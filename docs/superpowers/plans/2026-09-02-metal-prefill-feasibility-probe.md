# Metal Prefill Feasibility Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the smallest benchmark-only Metal probe that determines whether row-local scalar attention has enough measured headroom over MLX SDPA to justify corrected sparse prefill.

**Architecture:** Add private, test-only MSL and Rust FFI code beside Higgs' existing FastMetal kernels. The probe performs exact online-softmax attention over a deterministic fraction of visible KV blocks using `row4` and `head4` SIMD-group mappings. `Head4` groups up to four query heads from the same KV-head group and leaves surplus SIMD groups idle when the GQA ratio is below four. Full density is checked against MLX SDPA; lower densities provide optimistic hardware bounds without summaries, approximation policy, cache integration, or production dispatch.

**Tech Stack:** Rust, `mlx-rs`, `mlx-sys` FastMetal FFI, Metal Shading Language, ignored Cargo benchmarks, Metal System Trace.

## Global Constraints

- Keep the probe private and `#[cfg(test)]`; add no public API, runtime flag, cache field, model dispatch, or dependency.
- Change only `crates/higgs-models/src/metal_kernel.rs` until writing the result document.
- Reuse `CachedMetalKernel`, `cstr_vec`, `row_contiguous_copy`, `array_is_row_contiguous`, and the existing FFI error path.
- Disable FastMetal automatic row-contiguity conversion. Copy Q explicitly and time it; pass capacity-sized row-contiguous K/V with runtime `valid_len` and `capacity`.
- Support BF16/FP16, head dimensions 128/256, and every valid `Hq/Hkv` ratio.
- Use four SIMD groups per threadgroup and no K/V threadgroup staging.
- Allocate outside timing, warm JIT, force `eval`, alternate timing order, and use paired medians.
- At full density require maximum absolute error `<= 2e-3` and maximum row-relative L2 error `<= 2e-3` against MLX SDPA.
- Lower densities are optimistic performance bounds only.
- Run GitNexus upstream impact before changing any existing symbol; report HIGH or CRITICAL risk before editing. The new private probe symbols have no pre-existing blast radius.
- Run GitNexus `detect_changes` and `git diff --check` before every commit.

## File map

- Modify: `crates/higgs-models/src/metal_kernel.rs` — test-only MSL, FFI wrapper, correctness/layout tests, and ignored benchmark.
- Create after measurement: `docs/plans/indexless-metal-prefill-feasibility-results.md` — raw evidence and kill/promote verdict.

No cache, attention-dispatch, evaluator, or server files change in this plan.

---

### Task 1: Exact scalar probe and dense parity

**Files:**
- Modify/Test: `crates/higgs-models/src/metal_kernel.rs:6808-6845,8414-end`

**Interfaces:**

```rust
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
enum PrefillProbeSchedule { Row4, Head4 }

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
enum PrefillProbePattern { ContiguousTail, Interleaved }

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn prefill_dense_probe(
    queries: &Array, // row-contiguous [1,Hq,Lq,D]
    keys: &Array,    // row-contiguous [1,Hkv,capacity,D]
    values: &Array,
    valid_len: i32,
    query_offset: i32,
    density_num: i32,
    density_den: i32,
    schedule: PrefillProbeSchedule,
    pattern: PrefillProbePattern,
) -> Result<Array, Exception>;
```

- [ ] **Step 1: Write the failing dense-parity test**

Add deterministic bounded sine/cosine Q/K/V generation. Sweep:

```rust
for dtype in [Dtype::Bfloat16, Dtype::Float16] {
    for (len, d) in [(1, 32), (17, 128), (129, 256)] {
        for gqa in [1, 2, 4, 8] {
            // Hkv=1, Hq=gqa, full density, causal query_offset=0.
            // Alternate Row4 and Head4 across cases.
        }
    }
}
```

Compare with `mlx_rs::fast::scaled_dot_product_attention` using `ScaledDotProductAttentionMask::Causal` and scale `1.0 / sqrt(D)`. Add `assert_prefill_rows_close`: cast to FP32, calculate the global maximum absolute error, then calculate `sqrt(sum(error^2) / max(sum(reference^2), 1e-12))` independently for every `[D]` output row and take the maximum.

In the same RED step, add `prefill_probe_layout_contract` and `prefill_probe_schedules_match`. The layout test transposes Q, proves the view is non-row-contiguous, copies it explicitly, appends 17 rows to a default `SteppingKeyValueCache`, and compares backing-capacity probe output (`capacity=256`, `valid_len=17`) with MLX SDPA over the valid slice. The schedule test uses Q `[1,16,33,128]` and K/V `[1,2,256,128]`, comparing `row4` with `head4` at 100-percent contiguous, 25-percent contiguous, and 12.5-percent interleaved density. Low-density cases compare schedules only, never approximation quality against dense SDPA.

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p higgs-models --lib metal_kernel::tests::prefill_dense_probe_matches_mlx -- --exact --nocapture
cargo test -p higgs-models --lib metal_kernel::tests::prefill_probe_layout_contract -- --exact --nocapture
cargo test -p higgs-models --lib metal_kernel::tests::prefill_probe_schedules_match -- --exact --nocapture
```

Expected: compile failure because the new probe helpers do not exist.

- [ ] **Step 3: Implement the scalar online-softmax MSL**

Add `#[cfg(test)] const PREFILL_DENSE_PROBE_SOURCE`. Each SIMD group owns one output row/head. `row4` maps four groups to adjacent rows of one query head. `head4` maps them to the same row for up to four query heads inside one KV-head group, using `(kv_head, query_row, query_head_subgroup)` as the workgroup coordinates and leaving unused SIMD groups idle when necessary. Map GQA with `kv_head = q_head / (Hq/Hkv)`.

For every selected 128-token block and every causally visible key:

```metal
float partial = 0.0f;
for (uint dim = lane; dim < uint(D); dim += 32)
    partial += float(q[q_base + dim]) * float(k[k_base + dim]);
float score = simd_sum(partial) * rsqrt(float(D));
float next_max = max(running_max, score);
float old_scale = exp(running_max - next_max);
float weight = exp(score - next_max);
denominator = denominator * old_scale + weight;
for (uint dim = lane; dim < uint(D); dim += 32)
    out_fragment[dim / 32] = out_fragment[dim / 32] * old_scale
        + weight * float(v[v_base + dim]);
running_max = next_max;
```

Full density selects every visible block. Contiguous-tail density selects the newest `ceil(blocks*density_num/density_den)` blocks. Interleaved density distributes the same count deterministically across the visible range. Force the final visible block so every row has a denominator. This selector is benchmark mechanics, not production logic.

- [ ] **Step 4: Implement the FastMetal wrapper**

Use one cached kernel and a per-shape launch-config cache keyed by `(dtype,Hq,Hkv,Lq,capacity,D,schedule)`. Specialize MSL only for dtype, `D`, and schedule. Pass `Hq`, `Hkv`, `Lq`, `capacity`, `valid_len`, `query_offset`, density numerator/denominator, and pattern as scalar MLX arrays. Set a `(128,1,1)` threadgroup, grid enough for four work items per group, output `[1,Hq,Lq,D]`, and FastMetal `ensure_row_contiguous=false`.

Reject invalid ranks/shapes, `valid_len` outside `1..=capacity`, negative query offset, invalid density, `Hq % Hkv != 0`, unsupported dtype/dimension, and non-row-contiguous inputs with `Exception::custom("invalid prefill feasibility probe contract")`.

- [ ] **Step 5: Verify GREEN and the test namespace**

Run:

```bash
cargo fmt --all -- --check
cargo test -p higgs-models --lib metal_kernel::tests::prefill_dense_probe_matches_mlx -- --exact --nocapture
cargo test -p higgs-models --lib metal_kernel::tests
```

Expected: all commands exit 0 and dense parity passes 24 cases.

- [ ] **Step 6: Check scope and commit**

Run GitNexus `detect_changes({scope: "unstaged"})`, then:

```bash
git diff --check
git add crates/higgs-models/src/metal_kernel.rs
git commit -m "test(metal): add prefill feasibility probe"
```

### Task 2: Paired benchmark and hardware verdict

**Files:**
- Modify/Test: `crates/higgs-models/src/metal_kernel.rs`
- Create: `docs/plans/indexless-metal-prefill-feasibility-results.md`

**Interfaces:** Consumes Task 1's tested probe and produces JSONL timing rows plus one committed decision.

- [ ] **Step 1: Add the ignored benchmark**

Add `metal_kernel::tests::bench_prefill_dense_probe`, selected by `HIGGS_PREFILL_PROBE_CASE`, with default warmup 4 and samples 9. Accept these exact cases:

| Case | dtype | Hq:Hkv | D | Lq | Lkv |
|---|---:|---:|---:|---:|---:|
| `deploy4k` | BF16 | 16:2 | 256 | 4096 | 4096 |
| `deploy8k` | BF16 | 16:2 | 256 | 8192 | 8192 |
| `deploy16k` | BF16 | 16:2 | 256 | 16384 | 16384 |
| `control128_4k` | BF16 | 16:2 | 128 | 4096 | 4096 |
| `control128_8k` | BF16 | 16:2 | 128 | 8192 | 8192 |
| `gqa1_8k` | BF16 | 16:16 | 256 | 8192 | 8192 |
| `gqa2_8k` | BF16 | 16:8 | 256 | 8192 | 8192 |
| `gqa4_8k` | BF16 | 16:4 | 256 | 8192 | 8192 |
| `fp16_8k` | FP16 | 16:2 | 256 | 8192 | 8192 |
| `longk16k` | BF16 | 16:2 | 256 | 256 | 16384 |
| `longk32k` | BF16 | 16:2 | 256 | 256 | 32768 |
| `longk64k` | BF16 | 16:2 | 256 | 256 | 65536 |

Run both schedules at densities 100, 50, 37.5, 25, 12.5, and 6.25 percent with contiguous-tail selection; also run 25 and 12.5 percent interleaved.

- [ ] **Step 2: Implement paired timing and JSONL output**

Evaluate allocation before timing and Q-copy separately. Alternate order:

```text
even sample: MLX -> row4 -> head4
odd sample:  head4 -> row4 -> MLX
```

Force `eval` in every timed closure. Emit:

```text
case,dtype,hq,hkv,d,lq,lkv,schedule,pattern,density,
mlx_us,kernel_us,q_copy_us,copy_inclusive_us,ratio,
issued_kv_bytes,ideal_packed_kv_bytes,active_bytes,peak_bytes
```

Read memory with `mlx_get_active_memory` and `mlx_get_peak_memory`. Label issued bytes as an arithmetic model, not physical DRAM traffic.

- [ ] **Step 3: Compile, then run decisive cases in fresh processes**

Run:

```bash
cargo test -p higgs-models --release --lib metal_kernel::tests::bench_prefill_dense_probe --no-run
HIGGS_PREFILL_PROBE_CASE=deploy8k cargo test -p higgs-models --release --lib metal_kernel::tests::bench_prefill_dense_probe -- --ignored --exact --nocapture --test-threads=1
HIGGS_PREFILL_PROBE_CASE=deploy16k cargo test -p higgs-models --release --lib metal_kernel::tests::bench_prefill_dense_probe -- --ignored --exact --nocapture --test-threads=1
HIGGS_PREFILL_PROBE_CASE=gqa1_8k cargo test -p higgs-models --release --lib metal_kernel::tests::bench_prefill_dense_probe -- --ignored --exact --nocapture --test-threads=1
HIGGS_PREFILL_PROBE_CASE=longk32k cargo test -p higgs-models --release --lib metal_kernel::tests::bench_prefill_dense_probe -- --ignored --exact --nocapture --test-threads=1
```

Stop the remaining matrix if these hit a kill gate; otherwise run every remaining case in a fresh process.

- [ ] **Step 4: Apply the numeric decision gates**

Define `R(rho) = copy-inclusive probe latency / MLX dense latency`.

Promote scalar traversal only if all hold:

- `R(37.5%) <= 0.90` at 8K and 16K;
- `R(25%) <= 0.75` at 8K and 16K;
- GQA-8 retains at least 80 percent of GQA-1 speedup;
- Q copy is at most 20 percent of copy-inclusive latency;
- active/peak memory has no unexplained increase.

Kill scalar traversal if either 8K or 16K fails `R(12.5%) <= 0.80`. This redirects the next design to tiled SIMD-group matrix math or PackGQA; it does not reject corrected sparse attention.

If timing passes or is ambiguous, invoke the metal-gpu-profiling skill and capture one `row4` and one `head4` primary case. When GQA-8 traffic exceeds twice the ideal packed-head lower bound, require `head4` to recover at least 15 percent.

- [ ] **Step 5: Write the result document**

Record exact commands and commit; Mac/SoC/memory/macOS/Rust/MLX identity; raw JSONL or its artifact path; parity maxima; 8K/16K values for `R(37.5%)`, `R(25%)`, and `R(12.5%)`; row4/head4 and GQA comparisons; Q-copy share; memory/counters; and exactly one verdict:

```text
PROMOTE_SCALAR | PROMOTE_WITH_PACKGQA | REDESIGN_TILED | KILL_METAL_PREFILL
```

Name the single next experiment authorized by the verdict.

- [ ] **Step 6: Final verification and commit**

Run:

```bash
cargo fmt --all -- --check
cargo test -p higgs-models --lib metal_kernel::tests
cargo test -p higgs-models --release --lib metal_kernel::tests::bench_prefill_dense_probe --no-run
git diff --check
```

Run GitNexus `detect_changes({scope: "unstaged"})`, confirm only test-only probe/benchmark symbols and the result document, then:

```bash
git add crates/higgs-models/src/metal_kernel.rs docs/plans/indexless-metal-prefill-feasibility-results.md
git commit -m "bench(metal): measure prefill break-even"
```

## Stop boundary

This plan ends with a measured hardware verdict. It does not authorize QKV tracing, approximation evaluation, block summaries, cache lifecycle changes, `Qwen3NextAttention::forward_scheduled` dispatch, or a production sparse kernel. Those begin only if the committed result selects a viable execution primitive.
