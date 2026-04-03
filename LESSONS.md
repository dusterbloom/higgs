# Lessons Learned: MoE Inference on Apple Silicon (Metal)

Engineering reference from performance optimization of the Higgs MLX inference server.
Applies to any MoE model running quantized inference via `gather_qmm` on Metal GPUs.

---

## 1. MoE Expert Dispatch is the Bottleneck

In MoE models (DeepSeek-V2, Qwen3-MoE), expert dispatch accounts for ~83% of per-layer
compute during prefill. Attention and projection layers are not the bottleneck.

Measurements on DeepSeek-V2-Lite (64 experts, top_k=6, 4-bit):

| Component       | Share of per-layer compute |
|-----------------|---------------------------|
| MoE dispatch    | ~83%                      |
| Attention (SDPA)| ~14%                      |
| kv_b_proj       | ~2.5%                     |
| Other           | ~0.5%                     |

Implication: optimizing anything outside MoE dispatch yields negligible TTFT improvement.

## 2. Global Batch Sort for gather_qmm

**The single largest optimization found: 4-6x prefill speedup at L>=512.**

### Problem

Higgs kept tokens in original batch order when calling `gather_qmm`. Each expert's tokens
were scattered across memory. The Metal kernel cannot coalesce memory accesses when expert
indices jump around randomly.

### Solution

mlx-lm (Python reference) physically reorders tokens by expert index before calling
`gather_qmm`. The key function is `_gather_sort` from `mlx_lm/models/switch_layers.py`:

```python
def _gather_sort(x, indices):
    """Flatten all tokens, sort by expert index globally."""
    *_, M = indices.shape            # top_k
    indices_flat = indices.flatten()
    order = mx.argsort(indices_flat)
    inv_order = mx.argsort(order)
    x_sorted = x.reshape(-1, 1, x.shape[-1])[order // M]
    return x_sorted, indices_flat[order], inv_order
```

This produces monotonically non-decreasing expert indices. The Metal kernel processes each
expert's batch contiguously, enabling coalesced GPU memory access.

### Rust implementation

Core of `SwitchMlpWeights::forward_gather_global_sort` in `qwen3_next.rs`:

```rust
// Flatten [B, L, top_k] -> [N], argsort, derive token mapping
let idx_flat = indices.flatten(None, None)?;
let order = ops::argsort_axis(&idx_flat, 0)?;
let inv_order = ops::argsort_axis(&order, 0)?;
let token_idx = order.floor_divide(&top_k_arr)?;

// Reorder tokens, call gather_qmm with sorted_indices=true
let x_sorted = x_flat.take_axis(&token_idx, 0)?;
let gate_out = gather_qmm(&x_sorted, ..., true /* sorted_indices */)?;

// After MoE computation, restore original order
let out_unsorted = out_flat.take_axis(&inv_order, 0)?;
```

### Performance

Python microbenchmark (`bench_moe_sort.py`, DeepSeek-V2-Lite dimensions):

| SeqLen | Unsorted   | Global sort | Speedup |
|--------|------------|-------------|---------|
| 1      | 0.5 ms     | 0.5 ms      | 1.0x    |
| 128    | 6 ms       | 2 ms        | 3x      |
| 512    | 24 ms      | 5 ms        | 5x      |
| 2048   | 95 ms      | 16 ms       | 6x      |

Sort overhead is negligible (<2ms at L=2048). Decode (L=1) is unaffected.

## 3. MLX-RS / MLX-C Version Pinning

Switching `mlx-rs` from git pin `af21d79` (MLX-C 0.4.0) to crates.io 0.25.3 dropped
decode throughput from 60 tok/s to 37 tok/s (-38%). MLX-C 0.4.0 ships faster Metal kernels
not yet on crates.io.

**Critical**: after switching mlx-rs versions, you MUST delete the stale metallib:

```
rm target/release/mlx.metallib
```

A stale metallib causes silent performance regression with no errors or warnings.

API changes required for MLX-C 0.4.0:
- `gather_qmm`: wrap `group_size`/`bits` in `mlx_optional_int_`, add `mode="affine"` param
- `scaled_dot_product_attention`: added `sinks` param (`None::<&Array>`)
- `set_wired_limit_to_max`: uses `mlx_device_info_new/get` API (not `mlx_metal_device_info`)

## 4. Profiling Methodology

Rules that prevent false conclusions:

| Rule | Why |
|------|-----|
| Same binary, same machine, same power state | Battery vs AC causes 35-40% GPU throughput difference on M4 |
| Unique prompts per iteration | Defeats prefix cache (append run index or timestamp to prompt) |
| Python microbenchmarks first | 10 minutes to write, immediately confirms/rejects hypothesis before touching Rust |
| A/B against known reference | Compare against mlx-lm Python to isolate Rust-specific overhead |
| Median of N runs, with warmup | First 2-3 runs hit JIT compilation and Metal shader cache |

The Python `bench_moe_sort.py` that identified the root cause took 10 minutes to write
and immediately showed the 6x gap. Always prototype measurements in Python before
committing to Rust changes.

## 5. Dead Ends

These are documented to prevent re-investigation.

### Absorbed MLA (skip kv_b_proj)

**Hypothesis**: skip the kv_b_proj projection, cache the compressed latent directly.

**Result**: TTFT unchanged. kv_b_proj is only 2.5% of per-layer compute, so eliminating it
has no measurable impact. Decode regressed -12% because decompression now happens per-token
at each generation step instead of once during prefill. Reverted.

### Causal mask enum

**Hypothesis**: replacing the Array-based causal mask with an enum would let MLX use a
faster SDPA kernel path.

**Result**: no measurable TTFT change. MLX's SDPA already handles Array masks efficiently
and takes the same kernel path as the enum variant.

### Per-token sort (sort top-k within each token)

**Hypothesis**: sorting each token's top-k expert indices (axis=-1) would help `gather_qmm`.

**Result**: ~15% improvement vs 400-500% from global batch sort. The Metal kernel needs
cross-batch coalescing (all tokens for expert N contiguous in memory), not within-token
ordering. Per-token sort only helps if a single token has duplicate experts.

---

## Summary

The dominant cost in MoE inference on Metal is expert dispatch via `gather_qmm`. The fix
is straightforward: globally sort tokens by expert index before dispatch, pass
`sorted_indices=true`, then unsort the output. Everything else we tried was noise compared
to this single change.
