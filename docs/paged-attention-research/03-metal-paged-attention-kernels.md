# Metal Paged Attention Kernels

## Source: mistralrs-paged-attn/src/metal/

mistral.rs has a complete Metal paged attention implementation. Key files:
- `kernels/pagedattention.metal` -- paged SDPA (v1 + v2)
- `kernels/reshape_and_cache.metal` -- slot-mapped KV cache write
- `kernels/gather_kv_cache.metal` -- gather paged blocks into contiguous tensors
- `kernels/copy_blocks.metal` -- block-level copy
- `kernels/kv_scale_update.metal` -- FP8 scale tracking
- `kernels/utils.metal` -- FP8 conversion, math helpers
- `backend/paged_attention.rs` -- Rust Metal dispatch (CustomOp1)

---

## 1. KV Cache Physical Layout

### Key Cache: `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
- `x` = element size in bytes (bf16: x=2, f32: x=4, fp8: x=1)
- This "x-packing" groups consecutive bytes for vectorized Metal loads
- Example for bf16, head_size=128, block_size=32: `[N, 8, 64, 32, 2]`

### Value Cache: `[num_blocks, num_kv_heads, head_size, block_size]`
- No x-packing (different access pattern for V)
- Example: `[N, 8, 128, 32]`

### Why different layouts?
- **K** is accessed as dot product Q*K^T -- needs head_dim elements contiguous per query position
- **V** is accessed as weighted sum softmax(scores)*V -- needs block_size elements contiguous per head_dim
- The x-packing for K enables vectorized loads of head_dim elements

---

## 2. reshape_and_cache Kernel (Cache Write)

Writes new K/V tokens into paged cache at slot-mapped positions.

```metal
template <typename KV_T, typename CACHE_T>
kernel void reshape_and_cache(
    const device KV_T *key,           // [num_tokens, num_heads, head_size]
    const device KV_T *value,         // [num_tokens, num_heads, head_size]
    device CACHE_T *key_cache,        // [num_blocks, num_heads, head_size/x, block_size, x]
    device CACHE_T *value_cache,      // [num_blocks, num_heads, head_size, block_size]
    const device int64_t *slot_mapping, // [num_tokens]
    const device float *k_scale,      // [1] (optional, for FP8)
    const device float *v_scale,      // [1] (optional, for FP8)
    // + stride/shape params
    uint gid [[threadgroup_position_in_grid]],    // = token_idx
    uint tid [[thread_position_in_threadgroup]]
);
```

**Slot mapping:** `slot_idx = block_id * block_size + offset_in_block`
- `block_idx = slot_idx / block_size`
- `block_offset = slot_idx % block_size`
- Padding tokens use `slot_idx = -1` (skipped)

**Indexing math (K):**
```
tgt_key_idx = block_idx * num_heads * (head_size/x) * block_size * x
            + head_idx * (head_size/x) * block_size * x
            + x_idx * block_size * x
            + block_offset * x
            + x_offset
```

**Indexing math (V):**
```
tgt_value_idx = block_idx * num_heads * head_size * block_size
              + head_idx * head_size * block_size
              + head_offset * block_size
              + block_offset
```

One threadgroup per token, threads cooperatively write `num_heads * head_size` elements.

---

## 3. gather_kv_cache Kernel (Paged -> Contiguous)

Gathers K/V from paged blocks into contiguous output tensors. Used during prefill.

```metal
template <typename CACHE_T, typename OUT_T>
kernel void gather_kv_cache(
    const device CACHE_T *key_cache,    // [num_blocks, kv_heads, head_size/x, block_size, x]
    const device CACHE_T *value_cache,  // [num_blocks, kv_heads, head_size, block_size]
    device OUT_T *k_out,                // [num_tokens, kv_heads, head_size]
    device OUT_T *v_out,                // [num_tokens, kv_heads, head_size]
    const device float *k_scale,        // optional FP8 dequant
    const device float *v_scale,
    const device int *block_table,      // [batch, max_blocks]
    const device int *cu_seq_lens,      // [batch + 1]
    // + shape params
    uint gid [[threadgroup_position_in_grid]],  // = output token_id
    uint tid [[thread_position_in_threadgroup]]
);
```

**Key algorithm:**
1. Binary search `cu_seq_lens` to find `batch_id` for this output token
2. Compute `batch_offset = token_id - cu_seq_lens[batch_id]`
3. Look up physical block: `block_id = block_table[batch_id * stride + batch_offset / block_size]`
4. Compute slot within block: `slot = batch_offset % block_size`
5. Copy K/V elements with x-unpacking (K) and transpose (V)

One threadgroup per output token. Threads cooperatively copy `kv_heads * head_size` elements.

---

## 4. paged_attention Kernel (Decode SDPA)

Full paged attention for decode (single query token per sequence).

Two variants:
- **V1:** Single pass, all blocks processed in one kernel launch
- **V2:** Two-pass with partitioning for long contexts

### V1 Kernel

```metal
kernel void paged_attention_v1(
    device T *out,                      // [num_seqs, num_heads, head_size]
    const device T *q,                  // [num_seqs, num_heads, head_size]
    const device CACHE_T *key_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const device CACHE_T *value_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const device int *block_tables,     // [num_seqs, max_blocks_per_seq]
    const device int *context_lens,     // [num_seqs]
    // + scale, softcapping, alibi_slopes, strides
);
```

**Thread/grid structure:**
- Grid: `[num_heads, num_seqs, 1]` -- one threadgroup per (seq, head) pair
- Each threadgroup iterates over all blocks for its sequence
- Uses SIMD group operations for reductions
- Softmax computed online (running max + exp_sum)

**Per-block processing:**
1. Load Q for this head (shared across all blocks)
2. For each block in `block_tables[seq_id]`:
   - Load K vectors from paged cache (with x-unpacking)
   - Compute Q*K^T dot products -> scores
   - Apply softcapping if needed
   - Track running max + exp_sum for online softmax
3. Second pass: weighted sum of V using softmax scores

### V2 Kernel (Partitioned)

For long contexts, splits blocks into partitions:
- **Pass 1:** Each partition computes partial softmax + partial V accumulation
- **Pass 2:** Reduce across partitions using log-sum-exp

```
partition_size = 512  // tokens per partition
max_num_partitions = max_context_len / partition_size
```

V1 used when: `max_partitions == 1` OR `num_seqs * num_heads > 512`

### Supported head sizes
Kernel supports: 64, 80, 96, 112, 128, 192, 256
(Higgs uses 128 -- fully supported)

### Supported dtypes
- float, bfloat16_t, half (query/output)
- Same + F8E4M3 (cache)
- FP8 dequant via k_scale/v_scale

---

## 5. MLX SDPA Comparison

### MLX's current SDPA (`scaled_dot_product_attention.metal`)
- Standard contiguous SDPA -- expects `[B, H, T, D]` layout
- `sdpa_vector` variant for single-query decode (similar to paged_attention_v1 but contiguous)
- Supports head dims: 64, 96, 128, 256
- NO paged attention support
- NO block table indirection

### Can MLX SDPA be extended for paging?

**For prefill:** No need -- gather paged blocks into contiguous tensors first, then use standard MLX SDPA. This is exactly what mistral.rs does.

**For decode:** Two options:
1. **Gather + MLX sdpa_vector** -- gather K/V into contiguous, then call MLX's sdpa_vector. Simple but adds a memory copy.
2. **Custom Metal kernel** -- port mistral.rs's `paged_attention_v1` to work with MLX's buffer API. More complex but avoids the gather copy.

**Recommendation:** Start with option 1 (gather + sdpa_vector). Benchmark. Only write custom kernel if gather overhead is significant for decode.

### MLX custom kernel integration
MLX supports custom Metal kernels via:
- `mlx::core::metal::device().register_library()` -- register metallib
- `mlx::core::fast::metal_kernel()` -- call custom kernel from Python
- From Rust via mlx-sys: register pipeline state, dispatch compute command

---

## 6. Block Size Recommendations for Apple Silicon

### mistral.rs defaults
- Default: **32 tokens per block**
- Supported: `[8, 16, 32]`

### Apple Silicon considerations
- M-series GPU has 32KB threadgroup memory
- Block size 32 with bf16, head_dim=128: `32 * 128 * 2 = 8 KB` per K or V block per head
- Block size 64: `64 * 128 * 2 = 16 KB` -- still fits in threadgroup memory
- Block size 128: `128 * 128 * 2 = 32 KB` -- at the limit

### Recommendation for Higgs
- **Block size 32** for paged attention kernel (matches mistral.rs, fits comfortably)
- **Block size 64** for SSD spill tier (larger blocks reduce I/O overhead, 5.88 MB per block)
- These can differ: the spill tier groups 2 paged blocks into 1 SSD block

---

## 7. Key Takeaways for Higgs Implementation

1. **The "gather then SDPA" pattern is the pragmatic approach.** Even mistral.rs uses it for prefill. We can start with gather for both prefill AND decode, then optimize decode later.

2. **Metal kernel exists and is MIT-licensed.** We could port the pagedattention.metal kernel, but our ML framework is MLX (not candle). The kernel interface is straightforward -- buffers + scalar params.

3. **The x-packing for K cache is important** for vectorized Metal loads. We need to adopt this layout if we write a custom kernel, but can ignore it if we gather into contiguous tensors first.

4. **FP8 KV cache support is free** if we adopt the same cache layout. The kernels handle dequant via k_scale/v_scale function constants.

5. **Slot mapping is the key abstraction** -- `slot_id = block_id * block_size + block_offset`. All cache writes go through this mapping. All cache reads use `block_table[seq_id][block_idx]` to find the physical block.
