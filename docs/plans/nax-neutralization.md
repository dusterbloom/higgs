# NAX Neutralization Plan — Bit-Exact Custom Kernels for M5

## Problem

M5 introduces NAX (Neural Acceleration eXtensions), exposed via Metal
Performance Pack's `mpp::tensor_ops::matmul2d`. MLX 0.30.6 gates six NAX
kernel variants on `MACOS_SDK_VERSION >= 26.2 AND MLX_METAL_VERSION >= 400`.
When compiled in and running on M5 hardware, NAX replaces the
`simdgroup_matrix` (Metal 3) path for:

| NAX kernel | Replaces | Model impact |
|---|---|---|
| `steel_gemm_fused_nax` | fused steel GEMM | Attention QKV/O, lm_head |
| `steel_gemm_splitk_nax` | split-K steel GEMM | Large-K matmuls (K>=10240) |
| `quantized_nax` | quantized QMM | Affine 4-bit weight path |
| `fp_quantized_nax` | fp4/fp8 QMM | fp-quantized paths |
| `steel_attention_nax` | steel SDPA | **Biggest prefill cost** |
| `steel_gemm_gather_nax` | gather steel GEMM | MoE gather (unused — trellis GEMM is custom) |

NAX uses a different internal accumulation order than `simdgroup_matrix`.
Both are mathematically correct (fp32 accumulation, fp16/bf16 inputs) but
not bit-identical — the last few fp32 ULPs differ.

Staying on the non-NAX path costs **20% prefill / 4% decode** on M5.

## Current approach

The project stays on SDK 26.0 (< 26.2 gate), so `MLX_METAL_NO_NAX` is
defined at compile time and `is_nax_available()` short-circuits to false.
This keeps bit-exact output but pays the full 20%/4% cost on M5 hardware.
On M4 (current dev hardware), there is no cost — NAX doesn't exist.

## Scope of the problem

For Qwen3.6-35B-A3B-Escha-W2 (40 layers, full_attention_interval=4):

| Operation | % of prefill | NAX-affected? |
|---|---|---|
| Expert projections (trellis GEMM) | ~40% | No — custom kernel |
| Attention QKV/O (steel GEMM) | ~15% | Yes |
| Attention SDPA (steel attention) | ~15% | Yes |
| GDN fused projections | ~20% | Partially |
| Router/gate | ~3% | Yes (output -> top-k, robust) |
| Norms, embeddings, lm_head | ~7% | Mixed |

~30-35% of prefill compute goes through NAX-affected ops. The expert
projections (the largest single component) are already custom kernels.

## Approach: custom kernels via `mlx_fast_metal_kernel`

Write custom Metal kernels for the NAX-affected operations using
`simdgroup_matrix` (Metal 3 tensor-core ops). These are deterministic
across all hardware — same bits on M4, M5, and beyond. They bypass
MLX's NAX gate entirely (raw MSL source compiled by the JIT kernel
infrastructure higgs already uses for trellis GEMM and GDN recurrence).

### Why `simdgroup_matrix` and not NAX

- `simdgroup_matrix` maps to the GPU's tensor cores via stable Metal 3
  intrinsics. The instruction sequence is deterministic and documented.
- NAX (`mpp::tensor_ops::matmul2d`) is a Metal 4 black-box — the internal
  lane mapping and reduction tree are Apple-proprietary and may change
  between chip revisions.
- The trellis QGEMM kernel already proves scalar fma achieves 1.4 TFLOP/s
  of the 3.6 TFLOP/s peak. `simdgroup_matrix` would reach ~2.5+ TFLOP/s.
  The rejected BM8/BN32 prototype (`docs/DSPARK_MLX_DESIGN.md:329`) only
  rules out the narrow M=5 packed-Q1 case, not large-M GEMM or SDPA.

## Build phases

### Phase 1: Decode SDPA (2-3 days)

Fixes the 4% decode cost. Lowest risk.

At decode, Q is `[1,16,1,128]`, K/V are `[1,2,N+1,128]`, no mask. This is
a batched dot-product + weighted sum — structurally identical to the
existing TurboQuant `SCORE_KERNEL` + `VALUE_KERNEL` (turboquant.rs:1138-1201),
just operating on bf16 instead of quantized codes.

**Starting point:** GDN recurrence kernel (`qwen3_next.rs:1781-1862`).
Same grid `(32, Dv, B*Hv)`, same `simd_sum` reduction. Replace the
recurrence loop body with:
1. `simd_sum(Q .* K_i)` for each KV position → score
2. Scalar softmax over scores (sequential, N+1 elements)
3. `simd_sum(softmax_weight .* V_i)` → output element

**Deliverables:**
- MSL kernel ~150 lines
- Rust FFI wrapper ~100 lines (copy `CachedMetalKernel` pattern)
- Tests ~80 lines (sweep GQA ratios 8:1/4:1/2:1, context lengths
  1..4096, dtypes bf16/f16/f32)
- Gate: `max_rel_gap(custom, mlx_sdpa) <= 1e-3` for f32, `<= 2e-3` for bf16
- Integration: replace call site A at `qwen3_next.rs:4183-4194`

### Phase 2: Prefill flash attention (5-7 days)

Fixes the 20% prefill cost. The critical path.

At prefill, Q is `[1,16,L,128]`, K/V are `[1,2,T,128]`, causal mask.
Requires a real flash-attention kernel:

1. Tile the KV sequence into blocks (~64 keys per tile)
2. For each query block x KV tile:
   - Compute QK^T partial scores via `simdgroup_matrix` (16x16 blocks)
   - Track running max + sum for online softmax
   - Accumulate weighted V contribution
   - Renormalize when a new tile improves the max
3. Apply causal mask (`q_pos >= k_pos`) inside the tile loop

**Starting point:** trellis QGEMM (`metal_kernel.rs:741-940`).
The BM=32/BN=128/RM=4/RN=8 register-tiled block structure is the
template. The flash tiling adds the online softmax bookkeeping.

**De-risking strategy:** Start with a scalar-fma flash kernel to validate
correctness (will match current non-NAX steel speed), then upgrade the
inner loop to `simdgroup_matrix` for performance.

**Deliverables:**
- MSL kernel ~350-400 lines
- Rust FFI wrapper ~120 lines
- Tests ~100 lines (causal correctness with offset, chunked prefill
  integration, GQA 8:1, L sweep 32/128/512/1024)
- Integration: replace call sites B/C/D at `qwen3_next.rs:4390`,
  `4599`, `26266`
- Gate: `max_rel_gap(custom, mlx_sdpa) <= 1e-3`; causal masking
  bit-exact against reference

**GQA handling:** The kernel must handle H_q:H_kv = 8:1 internally
(8 query heads share 1 KV head). Grid: one threadgroup per
(query_head_group, query_position_block). Each threadgroup loads its
KV head once and iterates over the 8 query heads sharing it.

**Mask handling:**
- `kv_offset == 0`: use `Causal` enum (no buffer) — generate mask
  in-kernel via `threadgroup_position >= thread_position`
- `kv_offset > 0`: pass offset as scalar input, generate mask in-kernel
- Decode (L=1): no mask, separate kernel (Phase 1)

### Phase 3: simdgroup_matrix upgrade for QKV/O GEMM (1-2 days, optional)

Only if profiling shows projection matmuls are a bottleneck after SDPA
is custom. The trellis QGEMM already has register-tiled block structure
— upgrading the inner loop from scalar fma to `simdgroup_matrix`
multiply-add is localized (~50 lines MSL in the inner product loop).

The "ponytail" comment at `metal_kernel.rs:1007-1012` already identifies
this as the next step:
> "the inner product is scalar fma, which reaches about 1.4 TFLOP/s of
> the 3.6 TFLOP/s the chip gives. The next step is simdgroup_matrix."

**Deliverables:**
- MSL change ~50 lines (replace scalar fma loop with simdgroup tile ops)
- No new Rust code (same kernel, same dispatch)
- Tests: existing QGEMM test suite validates correctness

## Total estimate

| Phase | Fixes | Effort | Risk |
|---|---|---|---|
| 1: Decode SDPA | 4% decode | 2-3 days | Low |
| 2: Prefill flash attention | 20% prefill | 5-7 days | Medium |
| 3: simdgroup GEMM upgrade | Remaining proj cost | 1-2 days | Low |
| **Total** | **Full NAX independence** | **8-12 days** | |

## Reusable infrastructure

Everything below exists in the codebase today and needs zero changes:

| Component | Location | What it provides |
|---|---|---|
| `CachedMetalKernel` + `OnceLock` | `metal_kernel.rs:68-85` | Kernel registration |
| Config cache (thread-local HashMap) | `qwen3_next.rs:1367-1380` | Shape-specialized dispatch |
| Per-thread error handler | `metal_kernel.rs:48-56` | FFI error capture |
| `cstr_vec`, `ensure_row_contiguous` | `metal_kernel.rs:102-105, 6262` | Input preparation |
| GDN recurrence kernel (MSL template) | `qwen3_next.rs:1781-1862` | Closest SDPA analog |
| TurboQuant SCORE/VALUE kernels | `turboquant.rs:1138-1201` | Decode attention split |
| Trellis QGEMM (block GEMM template) | `metal_kernel.rs:741-940` | Register-tiled GEMM |
| `max_rel_gap` test helper | `eschamoe.rs:2788-2799` | Numerical comparison |
| `patterned_input` / `patterned_weights` | `metal_kernel.rs:7879` | Deterministic test data |
| SDPA test suite (GQA, causal, offset) | `utils.rs:346-735` | Reference patterns |
| Mask builders | `utils.rs:104-149` | Causal/Array mask format |

## Must build from scratch

| Component | Why new | Est. MSL lines |
|---|---|---|
| `simdgroup_matrix` preamble | Zero use in codebase | ~30 |
| Flash online softmax | No max-tracking pattern exists | ~80 |
| In-kernel causal mask | Masks are host-side today | ~10 |
| GQA broadcast in-kernel | MLX handles internally today | ~15 |

## NAX dispatch reference

From `mlx/backend/metal/` in MLX 0.30.6 (vendored at
`target/release/build/mlx-sys-*/out/build/_deps/mlx-src/`):

```
matmul.cpp:354     steel_matmul_regular_axpby_nax    (any GEMM)
matmul.cpp:950     steel_gemm_splitk_axpby_nax       (K>=10240, M*N>=2048^2)
matmul.cpp:2390    gather_mm_rhs_nax                  (M==1 gather matmul)
sdpa.cpp:177       sdpa_full_self_attention_nax       (flash attention)
quantized.cpp:695  qmm_nax                            (4-bit quantized matmul)
quantized.cpp:791  gather_qmm_nax                     (gather quantized matmul)
quantized.cpp:1136 gather_qmm_rhs_nax                 (M==1 gather quantized)
```

Dispatch condition (all sites):
```cpp
is_nax_available() && !complexfloating && (enable_tf32() || dtype != float32)
```

`enable_tf32()` defaults to 1 (`MLX_ENABLE_TF32=1`). For bf16/f16 inputs,
NAX is ALWAYS dispatched on M5 regardless of the tf32 flag. Setting
`MLX_ENABLE_TF32=0` only blocks NAX for fp32 inputs.

Runtime gate (`device.h:268-287`):
```cpp
__builtin_available(macOS 26.2, *) && arch_gen >= 17  // M-series
__builtin_available(macOS 26.2, *) && arch_gen >= 18  // P-series (iPhone)
```

## Alternative quick win: op-level NAX exemption

Before building custom kernels, validate whether full bit-exactness is
even needed at the token level:

1. Build with SDK 26.2 (NAX kernels compiled in)
2. Run the eschamoe test suite with NAX enabled on M5
3. Measure `max_rel_gap` between NAX and non-NAX logits
4. If < 1e-4 relative, the argmax matches for any practical prompt

If token-level equivalence holds, enable NAX freely and skip the custom
kernel work entirely. The 20%/4% cost is the price of fp32 bit-exactness,
which may be stricter than what the application actually needs.

## Environment

- Current dev hardware: Apple M4 MacBook Pro, 32 GB (no NAX, no cost)
- Target hardware: Apple M5+ (when available)
- MLX version: 0.30.6 (vendored via mlx-sys git rev f4aa309)
- SDK gate: MacOSX26.0.sdk (below 26.2 NAX threshold)
- The build also requires linking Xcode's static `libclang_rt.osx.a`
  for `___isPlatformVersionAtLeast` (used by `is_nax_available()`)
