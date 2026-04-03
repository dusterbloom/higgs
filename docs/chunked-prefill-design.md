# Chunked Prefill + Chunkwise-Parallel GDN Kernel Design

## Status: DRAFT -- needs profiling data before implementation

## Problem

For prompts of 5k+ tokens on Qwen3.5-35B-A3B (hybrid GDN + attention), TTFT is
dominated by the GDN layers. 48 of 64 layers are GDN, each running a sequential
`for (int t = 0; t < T; ++t)` recurrence over the full prompt length.

The current Metal kernel (`gated_delta_step` in `qwen3_next.rs:402-477`)
parallelizes across Dk (32 SIMD threads), Dv (128 grid.y), and B*Hv (32 grid.z)
= 131K threads. But the time dimension is fully serial.

## Model Dimensions (Qwen3.5-35B-A3B)

| Param | Value | Notes |
|-------|-------|-------|
| Dk (linear_key_head_dim) | 128 | |
| Dv (linear_value_head_dim) | 128 | |
| Hk (linear_num_key_heads) | 16 | |
| Hv (linear_num_value_heads) | 32 | GQA ratio = 2 |
| full_attention_interval | 4 | 3/4 layers are GDN |
| num_hidden_layers | 64 | 48 GDN + 16 attention |
| n_per_t (Dk / SIMD_WIDTH) | 4 | elements per thread |
| State shape | [B, 32, 128, 128] | 512K elements/batch |
| Grid size | (32, 128, 32) | 131K threads |

## Step 0: Profile Before Optimizing

**We must know where time actually goes before choosing a strategy.**

Add `Instant::now()` + `eval()` barriers around each component in a single
forward pass for T=5000:

```
embed         -> ??? ms
per layer:
  norm        -> ??? ms
  GDN kernel  -> ??? ms   (48 layers)
  Attn        -> ??? ms   (16 layers)
  residual    -> ??? ms
  norm        -> ??? ms
  MLP/MoE     -> ??? ms
  residual    -> ??? ms
lm_head       -> ??? ms
total         -> ??? ms
```

This tells us whether GDN is 30% or 80% of TTFT, which determines the ceiling
for any GDN optimization.

**Benchmark command:**
```
cargo test -p higgs-models --release -- bench_prefill_breakdown --nocapture --ignored
```

## Analysis: Current Kernel Performance

### Per-step work (one iteration of the T-loop)

Per thread (32 threads per SIMD group, n_per_t=4):

| Operation | FLOPs | Barriers | Memory |
|-----------|-------|----------|--------|
| state decay (state *= g) | 4 mul | 0 | 0 |
| kv_mem (dot state,k) | 4 mul | 1 simd_sum | 4 reads (k) |
| delta = (v - kv_mem)*beta | 2 | 0 | 1 read (v) |
| state update (+= k*delta) | 8 (4 mul+4 add) | 0 | 4 reads (k, again) |
| output (dot state,q) | 4 mul | 1 simd_sum | 4 reads (q) |
| y write | 0 | 0 | 1 write |
| **Total** | **22** | **2** | **13 reads + 1 write** |

Arithmetic intensity: 22 FLOPs / 28 bytes = 0.79 FLOPs/byte (memory-bound regime)

### Key bottlenecks per step
1. Two `simd_sum` barriers (cross-lane reductions stall the SIMD group)
2. 13 global memory reads (q, k read twice, v)
3. 1 global memory write (y)

### What limits throughput
- At 131K threads, the GPU is well-occupied
- But each step has only 22 FLOPs with 2 sync barriers
- The sequential loop means no thread can advance past the current timestep
- Memory reads are streaming/sequential (good for prefetch) but spread across
  large buffers for T=5000

---

## Design: Two-Component Architecture

### Component 1: Engine-Level Chunked Prefill

**Change `forward_hidden` to process the prompt in chunks of C tokens.**

```rust
// qwen3_next.rs: Qwen3NextCausalLM::forward_hidden()
// Instead of one forward() with T=5000:

let chunk_size = 128; // tunable
for chunk_start in (0..T).step_by(chunk_size) {
    let chunk_end = std::cmp::min(chunk_start + chunk_size, T);
    let chunk = inputs.index((.., chunk_start..chunk_end, ..));
    // Process chunk through ALL layers
    // Cache state carries over automatically (GDN state + KV cache)
}
```

**Why this works without kernel changes:**
- GDN `forward()` already accepts arbitrary seq_len S
- `ArraysCache` carries SSM state between calls
- `SteppingKeyValueCache` incrementally appends K/V for attention layers
- The `AttentionMask::Causal` handles variable offsets

**Expected benefits:**
- Working set per GDN dispatch: ~2 MB (fits GPU L2) vs ~82 MB (spills to DRAM)
- Shorter Metal command buffers -> better GPU pipeline interleaving
- Reduced peak memory (important on 32GB shared-memory M4)

**Expected cost:**
- 39 kernel dispatches per GDN layer instead of 1 (for T=5000, C=128)
- ~10-20us dispatch overhead each -> 0.4-0.8ms per GDN layer overhead
- Must verify this doesn't negate the L2 benefit

**Net gain: TBD by profiling. Estimated 1.5-3x if memory-bound, ~1x if compute-bound.**

### Component 2: Chunkwise-Parallel GDN Kernel

Replace the sequential kernel with a two-phase kernel for chunk-sized sequences.

#### The Math

The DeltaNet recurrence per dv_idx (independently for each of Dv=128 rows):

```
s_t = g_t * s_{t-1} + k_t * (v_t[dv] - dot(k_t, s_{t-1})) * beta_t
y_t[dv] = dot(q_t, s_t)
```

Rewritten as linear recurrence:

```
s_t = M_t * s_{t-1} + b_t
M_t = g_t * I - beta_t * k_t * k_t^T   (Dk x Dk, rank-1 perturbation of scaled I)
b_t = beta_t * v_t[dv] * k_t            (Dk vector)
```

Defining the corrected "value" after delta rule:

```
w_t = beta_t * (v_t[dv] - dot(k_t, s_{t-1}))    -- scalar, requires sequential state
```

Then output becomes gated linear attention:

```
y_t[dv] = SUM_{j<=t} decay(j->t) * dot(q_t, k_j) * w_j
decay(j->t) = PROD_{l=j+1}^{t} g_l = G_t / G_j
```

Where G_t = PROD_{l=0}^{t} g_l is the cumulative gate product.

#### Two-Phase Kernel

**Phase 1: Sequential State Scan** (C steps, state update only)

```metal
// SIMD group: 32 threads handle dk dimension
// Each threadgroup: one head, one dv_idx, one chunk

float state[n_per_t];           // in registers (4 floats)
threadgroup float w_cache[C];   // corrected values -> tgmem (C floats)
threadgroup float G_cache[C];   // cumulative gate products -> tgmem

float G_cumul = 1.0f;

for (int t = 0; t < C; ++t) {
    float g_val = compute_gate(a, a_log, dt_bias, t);
    float beta_val = compute_beta(b, t);
    G_cumul *= g_val;

    // State decay
    for (int i = 0; i < n_per_t; ++i)
        state[i] *= g_val;

    // kv_mem = dot(state, k_t)
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i)
        kv_mem += state[i] * k_[n_per_t * dk_idx + i];
    kv_mem = simd_sum(kv_mem);

    // Delta rule
    float delta = (v_[dv_idx] - kv_mem) * beta_val;

    // State update
    for (int i = 0; i < n_per_t; ++i)
        state[i] += k_[n_per_t * dk_idx + i] * delta;

    // Cache w and G for Phase 2 (only lane 0 writes)
    if (thread_index_in_simdgroup == 0) {
        w_cache[t] = delta;    // w_t = delta = beta * (v - k^T s)
        G_cache[t] = G_cumul;
    }

    // Advance pointers (k, v, a, b)
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    a_ += Hv;
    b_ += Hv;
}

// Write final state for inter-chunk propagation
for (int i = 0; i < n_per_t; ++i)
    o_state[...] = state[i];
```

**Key savings vs current kernel:**
- NO output computation in the loop (no q reads, no second simd_sum, no y writes)
- 1 simd_sum barrier per step instead of 2
- ~40% fewer FLOPs per step

**Phase 2: Parallel Output Computation**

After Phase 1, we have `w_cache[0..C]` and `G_cache[0..C]` in threadgroup memory.
Now compute all C outputs in parallel.

```metal
threadgroup_barrier(mem_threadgroup);

// Each SIMD lane computes outputs for a range of time positions
// 32 threads / ceil(C/32) iterations = covers all C positions
for (int t = dk_idx; t < C; t += 32) {
    // y_t = SUM_{j<=t} (q_t . k_j) * (G_cache[t] / G_cache[j]) * w_cache[j]

    // Reload q_t from global memory
    auto q_t = q + chunk_offset + t * Hk * Dk + hk_idx * Dk;

    float out = 0.0f;
    float G_t = G_cache[t];

    for (int j = 0; j <= t; ++j) {
        // dot(q_t, k_j) -- need k_j from global memory
        auto k_j = k + chunk_offset + j * Hk * Dk + hk_idx * Dk;
        float qk = 0.0f;
        for (int d = 0; d < Dk; ++d)
            qk += float(q_t[d]) * float(k_j[d]);

        float decay = G_t / G_cache[j];
        out += qk * decay * w_cache[j];
    }

    y[chunk_offset + t * Hv * Dv + dv_idx] = InT(out);
}
```

**Wait -- this has a problem.** Each output needs a Dk-dimensional dot product
with ALL preceding k vectors. That's O(C^2 * Dk) FLOPs per dv position per head.
For C=128, Dk=128: 2M FLOPs per position. With Dv=128, Hv=32: total 8.4 GFLOPS
per chunk per layer. For 39 chunks * 48 layers = 16 TFLOPS. At M4's 2.5 TFLOPS
peak, that's 6.4 seconds -- MUCH slower than the current approach.

**The C^2 * Dk term kills us for large C.** Must use small C or a different output
formulation.

### Revised Phase 2: Checkpoint + Short Sequential

Instead of the O(C^2 * Dk) attention, store state checkpoints during Phase 1
and replay short segments:

```metal
// During Phase 1, checkpoint state every K steps:
constexpr int K = 8;  // checkpoint interval
threadgroup float state_ckpt[C/K][n_per_t]; // in tgmem

// In Phase 1 loop, every K steps:
if (t % K == 0 && thread_index_in_simdgroup == 0) {
    for (int i = 0; i < n_per_t; ++i)
        state_ckpt[t/K][i] = state[i];
}
```

Then Phase 2 replays at most K steps from the nearest checkpoint:

```metal
// Phase 2: parallel output from checkpoints
for (int t = dk_idx; t < C; t += 32) {
    int ckpt = t / K;
    int t_offset = t % K;

    // Load checkpoint state
    float local_state[n_per_t];
    for (int i = 0; i < n_per_t; ++i)
        local_state[i] = state_ckpt[ckpt][i];

    // Replay t_offset steps
    for (int s = ckpt * K; s <= t; ++s) {
        // state update (same as Phase 1)
        float g_val = ...; float beta_val = ...;
        for (int i = 0; i < n_per_t; ++i)
            local_state[i] *= g_val;
        // ... kv_mem, delta, update ...
    }

    // Compute output
    float out = 0.0f;
    auto q_t = q + ...;
    for (int i = 0; i < n_per_t; ++i)
        out += local_state[i] * float(q_t[n_per_t * dk_idx + i]);
    out = simd_sum(out);

    if (thread_index_in_simdgroup == 0)
        y[...] = InT(out);
}
```

**Problem**: SIMD threads now process DIFFERENT time positions, but the inner
replay loop requires `simd_sum` which needs all threads in sync. This breaks the
SIMD cooperation model.

**Fix**: Process outputs in lock-step groups of 32, where all threads replay the
same checkpoint but for different dk elements:

```metal
// Phase 2: for each time position (sequentially, but without state dependency)
for (int t = 0; t < C; ++t) {
    int ckpt = t / K;
    int t_offset = t % K;

    // All 32 threads load the same checkpoint (their dk slice)
    float local_state[n_per_t];
    for (int i = 0; i < n_per_t; ++i)
        local_state[i] = state_ckpt_full[ckpt][(n_per_t * dk_idx + i)];

    // Replay t_offset steps (all threads in sync)
    for (int s = ckpt * K; s <= t; ++s) {
        // ... same sequential recurrence ...
    }

    // Output (simd_sum works because all threads are at same t)
    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i)
        out += local_state[i] * float(q_t[n_per_t * dk_idx + i]);
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0)
        y[t * Hv * Dv + dv_idx] = InT(out);
}
```

This iterates over C positions, replaying at most K steps each. Total sequential
steps: C * K/2 (average replay) vs C (current). **This is SLOWER, not faster!**

---

## Honest Assessment

After rigorous analysis, here is the truth about each approach:

### What works

| Approach | Expected gain | Complexity | Confidence |
|----------|--------------|------------|------------|
| Engine chunked prefill | 1.5-3x TTFT | Low | Medium |
| Skip output in state loop | ~1.3x per GDN | Low | High |
| Fused k read (read once, use twice) | ~1.1x | Low | High |
| fp16 state accumulation | up to 2x | Low | Low (stability?) |

### What does NOT work

| Approach | Why |
|----------|-----|
| C*C attention for outputs | O(C^2 * Dk) FLOPs >> O(C * Dk) sequential |
| Parallel prefix scan on state | Dk*Dk transition matrices, O(T * Dk^3) total |
| Checkpoint + replay | Same or more total steps than sequential |

### The fundamental constraint

The DeltaNet recurrence `s_t = M_t s_{t-1} + b_t` where `M_t` is Dk*Dk has
irreducible O(T * Dk) sequential depth per head per dv position. Any parallel
formulation trades **more total FLOPs** for reduced depth, but on a GPU that's
already well-occupied (131K threads), adding work slows things down.

**The kernel is likely in the awkward middle**: not fully compute-bound (too
much sequential dependency), not fully memory-bound (streaming access is
prefetch-friendly). The sequential loop with 2 SIMD barriers per step creates
a latency-bound pipeline.

---

## Recommended Plan

### Phase 1: Measure (1 day)

Write `bench_prefill_breakdown` test in `qwen3_next.rs`:
- T = [128, 512, 1024, 2048, 5120]
- Measure per-component time with eval barriers
- Determine: what fraction of TTFT is GDN vs MLP vs attention?
- Determine: is GDN compute-bound, memory-bound, or latency-bound?

### Phase 2: Engine Chunked Prefill (2-3 days)

Implement in `Qwen3NextCausalLM::forward_hidden`:
- Chunk the input into C-token segments
- Process each chunk through all layers
- Attention layers: KV cache grows incrementally (already works)
- GDN layers: SSM state carries over (already works)
- Tune C: try 64, 128, 256, 512

Measure TTFT at T=5000 with and without chunking.

### Phase 3: Lean State Kernel (2-3 days)

Write `gated_delta_prefill` -- a variant of the GDN kernel optimized for prefill:

1. **Fuse k reads**: Load k[dk] once, use for both kv_mem and state update
2. **Remove unnecessary output writes**: During prefill chunks (not the last),
   we still need outputs for downstream layers. But we can batch-write them
   from threadgroup memory instead of per-step global writes.
3. **Precompute gate values**: Load a[] and b[] in bulk, compute g/beta upfront
4. **Use threadgroup memory for q, k, v**: Load the chunk's data into tgmem
   once, read from fast memory during the loop

Expected: 1.2-1.5x improvement on the kernel itself.

### Phase 4: Explore Parallel Formulations (research)

Only if profiling shows GDN is >60% of TTFT AND the kernel is latency-bound:

- **Small-C attention formulation**: C=16-32 where C^2*Dk overhead is tolerable
- **simdgroup_matrix acceleration**: Batch multiple dv positions per SIMD group
  to turn dot products into wider matrix ops
- **fp16 state**: Test numerical stability within chunk-sized sequences

---

## Test Plan

### Correctness
```rust
#[test]
fn test_chunked_prefill_matches_full() {
    // Run model.forward(full_prompt) and model.forward(chunks)
    // Compare final logits and cache states
    // Must match to fp16 tolerance
}
```

### Performance
```
bench_ttft --model Qwen3.5-35B-A3B-3bit --tokens 5120 --warmup 3 --runs 10
```

Compare: baseline vs chunked vs chunked+lean_kernel

### Regression
- All 275 existing tests must pass
- Decode tok/s must not regress (chunking only affects prefill path)

---

## Files to Modify

| File | Change |
|------|--------|
| `qwen3_next.rs:1756` | `forward_hidden`: add chunked prefill loop |
| `qwen3_next.rs:402` | New kernel source: `GATED_DELTA_PREFILL_SOURCE` |
| `qwen3_next.rs:481` | New kernel creation fn |
| `qwen3_next.rs:1305` | GDN forward: dispatch prefill kernel when S > threshold |
| `qwen3_next.rs` (tests) | `bench_prefill_breakdown`, `test_chunked_prefill_matches_full` |

No engine-level changes needed. No changes to batch_engine or simple.rs.
The chunking happens inside the model's forward_hidden, transparent to the engine.

---

## Open Questions

1. What is the actual per-component breakdown of TTFT at T=5000?
2. Is the GDN kernel compute-bound, memory-bound, or latency-bound?
3. Does fp16 state accumulation maintain output quality over C=128 steps?
4. What is the optimal chunk size C? (depends on L2 cache size, dispatch overhead)
5. Can MLX's lazy eval pipeline dispatches across chunks without explicit async_eval?
