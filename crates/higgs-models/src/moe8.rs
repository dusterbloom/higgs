//! Weight-once `MoE` expert passes for 1..=8 routed rows (Clover block preview,
//! self-speculative verify, diffusion commits).
//!
//! MLX's `gather_qmv` re-reads an expert's weights once per routed (row, slot):
//! 64 slots for 8 rows x top-8 against ~15-30 distinct experts. Here the
//! distinct experts of the rows are planned on the GPU and each expert is read
//! once, all 8 rows computed in 8x8x8 MMAs as in [`crate::qmm8`].
//!
//! After Kapitsa's `moe8_plan` / `qmm8_gu` / `moe8_down` (`verify8.metal`,
//! Apache-2.0, nace-ai/kapitsa):
//! 1. `plan` (one threadgroup, a thread per expert): the distinct experts of
//!    the rows' top-k, in ascending id order, become tiles; `slot[tile][row]`
//!    is the row's top-k slot for that expert (`0xffffffff`: the row does not
//!    use it). Prefix sums, no atomics: the tiles are a pure function of the
//!    routing.
//! 2. `gate_up` (grid: intermediate columns x tiles): gate and up of the tile's
//!    expert for all 8 rows, `SwiGLU` in fp32, `h` stored as half
//!    `[tile, 8, I]`.
//! 3. `down` (grid: hidden columns x tiles): `h[tile] @ W_down[e]^T`, each
//!    live row stored to its `[row, slot, D]` output. Every output element has
//!    exactly one writer and the per-element arithmetic does not depend on the
//!    other rows, so the result is deterministic.
//!
//! The grids are sized for the worst case (`rows * top_k` tiles); tiles past
//! `ntiles` exit at once, so there is no host sync and no indirect dispatch.
//!
//! Numerics follow `qmm8` (half-rounded activations, fp32 group epilogue), not
//! MLX's qmv, so this runs only inside `with_fast_rows`: never AR decode,
//! prefill or anything whose outputs reach a retained cache.

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, error::Exception};

use crate::metal_kernel::{FastKernel, TemplateArg};

/// Simdgroups per threadgroup of the two expert passes.
const NSG: i32 = 4;
/// Gate (and as many up) 8-column fragments per simdgroup in `gate_up`.
const NP: i32 = 2;
/// 8-column fragments per simdgroup in `down`.
const NF: i32 = 2;
/// 64-groups whose weight codes are loaded ahead of their MMAs (M4: 2 beats
/// 4 and 8, which spill).
const UNR: i32 = 2;

/// Shared by `gate_up` and `down`: the `qmm8` MMA chain over the 64-groups.
const HEADER: &str = r#"
// 4 consecutive 16-bit inputs (half or bfloat16 bits) as half.
template <typename T>
inline half4 v8_x4(const device T* p) {
  const ushort4 u = *reinterpret_cast<const device ushort4*>(p);
  if (metal::is_same<T, half>::value) return as_type<half4>(u);
  return half4(as_type<float4>(uint4(u) << 16));
}
// B operands of one 64-group for token rows fn, fn + 1 (k-slot fm of fragment f
// is physical input 16 (fm / 2) + 8 (f / 4) + f % 4 + 4 (fm % 2), so a weight
// row's 16 contiguous codes feed one lane's 8 fragments) and the rows' sums of
// the half-rounded inputs, over the 8 lanes sharing fn (lane bits 1, 2, 4).
template <typename T>
inline float2 v8_load_b(const device T* x0, const device T* x1, bool v0, bool v1, int g,
                        thread simdgroup_half8x8 (&B)[8]) {
  const half4 a0 = v0 ? v8_x4<T>(x0 + 64 * g) : half4(0.0h);
  const half4 a1 = v0 ? v8_x4<T>(x0 + 64 * g + 8) : half4(0.0h);
  const half4 b0 = v1 ? v8_x4<T>(x1 + 64 * g) : half4(0.0h);
  const half4 b1 = v1 ? v8_x4<T>(x1 + 64 * g + 8) : half4(0.0h);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 4; ++i) {
    B[i].thread_elements()[0] = a0[i];
    B[i].thread_elements()[1] = b0[i];
    B[4 + i].thread_elements()[0] = a1[i];
    B[4 + i].thread_elements()[1] = b1[i];
  }
  const float4 fa = float4(a0) + float4(a1), fb = float4(b0) + float4(b1);
  float2 sx = float2((fa.x + fa.y) + (fa.z + fa.w), (fb.x + fb.y) + (fb.z + fb.w));
  sx += simd_shuffle_xor(sx, 2);
  sx += simd_shuffle_xor(sx, 4);
  sx += simd_shuffle_xor(sx, 16);
  return sx;
}
// codes . x over one 64-group: the 16 codes of this lane's weight row as the A
// operand, exact in half as 1024 + q.
inline float2 v8_codes_mma(uint2 wq, thread simdgroup_half8x8 (&B)[8]) {
  simdgroup_half8x8 A;
  simdgroup_float8x8 C;
  C.thread_elements()[0] = 0.0f;
  C.thread_elements()[1] = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 8; ++f) {
    const uint word = (f < 4 ? wq.x : wq.y) >> (4 * (f & 3));
    const half2 a = as_type<half2>((word & 0x000F000Fu) | 0x64006400u);
    A.thread_elements()[0] = a.x;
    A.thread_elements()[1] = a.y;
    simdgroup_multiply_accumulate(C, A, B[f], C);
  }
  return float2(C.thread_elements()[0], C.thread_elements()[1]);
}
// acc[f] += x . dequant(row f) over G groups for NFR weight rows (codes wrow[f]
// at this lane's 8 bytes of each group, scales srow[f], biases brow[f]) against
// the inputs of rows fn, fn + 1. UNR groups' codes are loaded ahead of their
// MMAs; the epilogue is fp32: y += s * dot + (b - 1024 s) * sum(x).
template <typename T, typename S, int NFR, int G>
inline void v8_run(const device uint2* const thread* wrow, const device S* const thread* srow,
                   const device S* const thread* brow, const device T* x0, const device T* x1,
                   bool v0, bool v1, thread float2* acc) {
  for (int g0 = 0; g0 < G; g0 += UNR) {
    uint2 wq[UNR][NFR];
    _Pragma("clang loop unroll(full)")
    for (int u = 0; u < UNR; ++u)
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NFR; ++f) wq[u][f] = wrow[f][(g0 + u) * 4];
    _Pragma("clang loop unroll(full)")
    for (int u = 0; u < UNR; ++u) {
      const int g = g0 + u;
      simdgroup_half8x8 B[8];
      const float2 sx = v8_load_b<T>(x0, x1, v0, v1, g, B);
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < NFR; ++f) {
        const float s = float(srow[f][g]);
        const float b = float(brow[f][g]);
        acc[f] += s * v8_codes_mma(wq[u][f], B) + (b - 1024.0f * s) * sx;
      }
    }
  }
}
"#;

/// One threadgroup of `32 * NSGP >= NE` threads; thread e plans expert e.
const PLAN: &str = r"
  threadgroup uint cnt[32];
  const uint tid = thread_position_in_threadgroup.x;
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  uint my[8];
  uint rows = 0;
  for (int t = 0; t < 8; ++t) {
    my[t] = 0xffffffffu;
    if (t < R && tid < uint(NE))
      for (int j = 0; j < TK; ++j)
        if (uint(idx[t * TK + j]) == tid) my[t] = uint(j);
    rows += my[t] != 0xffffffffu ? 1u : 0u;
  }
  const uint used = rows > 0 ? 1u : 0u;
  const uint pre = simd_prefix_exclusive_sum(used);
  if (lane == 31) cnt[sgi] = pre + used;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base = 0;
  for (uint i = 0; i < sgi; ++i) base += cnt[i];
  if (used) {
    const uint u = base + pre;
    tiles[u] = tid;
    for (int t = 0; t < 8; ++t) slot[u * 8 + t] = my[t];
  }
  if (tid == 0) {
    uint total = 0;
    for (int i = 0; i < NSGP; ++i) total += cnt[i];
    ntiles[0] = total;
  }
";

/// Grid `(N / (8 NP) simdgroups, tiles)`: `h[tile][t][n] = silu(x_t . Wg[e][n]) * (x_t . Wu[e][n])`.
const GATE_UP: &str = r"
  const uint tile = threadgroup_position_in_grid.y;
  if (tile >= ntiles[0]) return;
  const uint lane = thread_index_in_simdgroup;
  const int c0 = (int(threadgroup_position_in_grid.x) * NSG + int(simdgroup_index_in_threadgroup)) * 8 * NP;
  if (c0 >= N) return;
  const uint e = tiles[tile];
  const uint q = lane >> 2;
  const int fm = int((q & 4) | ((lane >> 1) & 3));
  const int fn = int(((q & 2) << 1) | ((lane & 1) << 1));
  constexpr int G = K / 64;
  const int kb = 16 * (fm >> 1) + 4 * (fm & 1);
  const bool v0 = fn < M, v1 = fn + 1 < M;
  const device T* x0 = x + size_t(v0 ? fn : 0) * K + kb;
  const device T* x1 = x + size_t(v1 ? fn + 1 : 0) * K + kb;
  const size_t wexp = size_t(e) * size_t(N) * (K / 8);
  const size_t sexp = size_t(e) * size_t(N) * G;
  // fragments 0 .. NP - 1: gate rows c0 + 8 p + fm; NP .. 2 NP - 1: the up rows
  const device uint2* wrow[2 * NP];
  const device ST* srow[2 * NP];
  const device ST* brow[2 * NP];
  for (int p = 0; p < NP; ++p) {
    const size_t n = size_t(c0 + 8 * p + fm);
    wrow[p] = reinterpret_cast<const device uint2*>(wg + wexp + n * (K / 8)) + (fn >> 1);
    srow[p] = scg + sexp + n * G;
    brow[p] = bg + sexp + n * G;
    wrow[NP + p] = reinterpret_cast<const device uint2*>(wu + wexp + n * (K / 8)) + (fn >> 1);
    srow[NP + p] = scu + sexp + n * G;
    brow[NP + p] = bu + sexp + n * G;
  }
  float2 acc[2 * NP];
  for (int f = 0; f < 2 * NP; ++f) acc[f] = 0.0f;
  v8_run<T, ST, 2 * NP, G>(wrow, srow, brow, x0, x1, v0, v1, acc);
  device half* ht = h + size_t(tile) * 8 * N;
  for (int p = 0; p < NP; ++p) {
    const int n = c0 + 8 * p + fm;
    const float2 gt = acc[p];
    const float2 hv = gt / (1.0f + exp(-gt)) * acc[NP + p];
    ht[size_t(fn) * N + n] = half(hv.x);
    ht[size_t(fn + 1) * N + n] = half(hv.y);
  }
";

/// Grid `(N / (8 NF) simdgroups, tiles)`: `y[t][slot][n] = h[tile][t] . Wd[e][n]`.
const DOWN: &str = r"
  const uint tile = threadgroup_position_in_grid.y;
  if (tile >= ntiles[0]) return;
  const uint lane = thread_index_in_simdgroup;
  const int n0 = (int(threadgroup_position_in_grid.x) * NSG + int(simdgroup_index_in_threadgroup)) * 8 * NF;
  if (n0 >= N) return;
  const uint e = tiles[tile];
  const uint q = lane >> 2;
  const int fm = int((q & 4) | ((lane >> 1) & 3));
  const int fn = int(((q & 2) << 1) | ((lane & 1) << 1));
  constexpr int G = K / 64;
  const int kb = 16 * (fm >> 1) + 4 * (fm & 1);
  const device half* x0 = h + (size_t(tile) * 8 + fn) * K + kb;
  const device half* x1 = x0 + K;
  const size_t wexp = size_t(e) * size_t(N) * (K / 8);
  const size_t sexp = size_t(e) * size_t(N) * G;
  const device uint2* wrow[NF];
  const device ST* srow[NF];
  const device ST* brow[NF];
  for (int f = 0; f < NF; ++f) {
    const size_t n = size_t(n0 + 8 * f + fm);
    wrow[f] = reinterpret_cast<const device uint2*>(w + wexp + n * (K / 8)) + (fn >> 1);
    srow[f] = scales + sexp + n * G;
    brow[f] = biases + sexp + n * G;
  }
  float2 acc[NF];
  for (int f = 0; f < NF; ++f) acc[f] = 0.0f;
  v8_run<half, ST, NF, G>(wrow, srow, brow, x0, x1, true, true, acc);
  const uint j0 = slot[tile * 8 + fn], j1 = slot[tile * 8 + fn + 1];
  for (int f = 0; f < NF; ++f) {
    const int n = n0 + 8 * f + fm;
    if (j0 != 0xffffffffu) y[(size_t(fn) * TK + j0) * N + n] = OutT(acc[f].x);
    if (j1 != 0xffffffffu) y[(size_t(fn + 1) * TK + j1) * N + n] = OutT(acc[f].y);
  }
";

fn header() -> std::ffi::CString {
    std::ffi::CString::new(format!(
        "constant constexpr int NSG = {NSG};\nconstant constexpr int NP = {NP};\n\
         constant constexpr int NF = {NF};\nconstant constexpr int UNR = {UNR};\n{HEADER}"
    ))
    .unwrap_or_default()
}

fn plan_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_moe8_plan",
            &[c"idx"],
            &[c"tiles", c"slot", c"ntiles"],
            &std::ffi::CString::new(PLAN).unwrap_or_default(),
            c"",
        )
    })
}

fn gate_up_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_moe8_gate_up_affine4_g64",
            &[
                c"x", c"wg", c"scg", c"bg", c"wu", c"scu", c"bu", c"tiles", c"ntiles",
            ],
            &[c"h"],
            &std::ffi::CString::new(GATE_UP).unwrap_or_default(),
            &header(),
        )
    })
}

fn down_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_moe8_down_affine4_g64",
            &[
                c"h", c"w", c"scales", c"biases", c"tiles", c"ntiles", c"slot",
            ],
            &[c"y"],
            &std::ffi::CString::new(DOWN).unwrap_or_default(),
            &header(),
        )
    })
}

/// `(weight, scales, biases)` of one stacked expert projection
/// `[E, out, in / 8]` u32 / `[E, out, in / 64]`.
pub(crate) type Expert<'a> = (&'a Array, &'a Array, &'a Array);

/// Shapes the kernels take: `x` `[rows, D]` 16-bit with 1..=8 rows, `indices`
/// `[rows, top_k]` u32, gate/up `[E, I, D / 8]` and down `[E, D, I / 8]` u32,
/// D and I multiples of `64 * UNR`, E <= 1024.
pub(crate) fn supported(x: &Array, indices: &Array, wg: &Array, wu: &Array, wd: &Array) -> bool {
    let ([rows, d], [e, i, dp], [ed, dd, ip]) = (x.shape(), wg.shape(), wd.shape()) else {
        return false;
    };
    (1..=8).contains(rows)
        && matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16)
        && indices.shape() == [*rows, indices.dim(-1)]
        && indices.dtype() == Dtype::Uint32
        && wg.shape() == wu.shape()
        && [wg.dtype(), wu.dtype(), wd.dtype()] == [Dtype::Uint32; 3]
        && e == ed
        && (1..=1024).contains(e)
        && d == dd
        && *d == dp * 8
        && *i == ip * 8
        && d % (64 * UNR) == 0
        && i % (64 * UNR) == 0
}

/// The `SwiGLU` expert outputs `[rows, top_k, D]` (in `x`'s dtype) of every
/// routed (row, slot); call only when [`supported`].
pub(crate) fn moe8_affine4(
    x: &Array,
    indices: &Array,
    (wg, scg, bg): Expert<'_>,
    (wu, scu, bu): Expert<'_>,
    (wd, sd, bd): Expert<'_>,
) -> Result<Array, Exception> {
    let (rows, d) = (x.dim(0), x.dim(1));
    let (e, inter) = (wg.dim(0), wg.dim(1));
    let tk = indices.dim(-1);
    let tiles_max = (rows * tk).min(e);
    let plan_threads = (e + 31) / 32 * 32;
    let idx = indices.reshape(&[rows * tk])?;
    let mut plan = plan_kernel().apply(
        &[&idx],
        &[
            TemplateArg::Int(c"R", rows),
            TemplateArg::Int(c"TK", tk),
            TemplateArg::Int(c"NE", e),
            TemplateArg::Int(c"NSGP", plan_threads / 32),
        ],
        [plan_threads, 1, 1],
        [plan_threads, 1, 1],
        &[
            (&[tiles_max], Dtype::Uint32),
            (&[tiles_max * 8], Dtype::Uint32),
            (&[1], Dtype::Uint32),
        ],
    )?;
    let ntiles = plan.remove(2);
    let slot = plan.remove(1);
    let tiles = plan.remove(0);
    let h = gate_up_kernel()
        .apply(
            &[x, wg, scg, bg, wu, scu, bu, &tiles, &ntiles],
            &[
                TemplateArg::Dtype(c"T", x.dtype()),
                TemplateArg::Dtype(c"ST", scg.dtype()),
                TemplateArg::Int(c"K", d),
                TemplateArg::Int(c"N", inter),
                TemplateArg::Int(c"M", rows),
            ],
            [inter / (8 * NP) * 32, tiles_max, 1],
            [NSG * 32, 1, 1],
            &[(&[tiles_max, 8, inter], Dtype::Float16)],
        )?
        .remove(0);
    Ok(down_kernel()
        .apply(
            &[&h, wd, sd, bd, &tiles, &ntiles, &slot],
            &[
                TemplateArg::Dtype(c"ST", sd.dtype()),
                TemplateArg::Dtype(c"OutT", x.dtype()),
                TemplateArg::Int(c"K", inter),
                TemplateArg::Int(c"N", d),
                TemplateArg::Int(c"TK", tk),
            ],
            [d / (8 * NF) * 32, tiles_max, 1],
            [NSG * 32, 1, 1],
            &[(&[rows, tk, d], x.dtype())],
        )?
        .remove(0))
}

#[doc(hidden)]
pub mod probe {
    //! Test seam (higgs-models unit tests do not compile).
    use super::{Array, Exception};

    pub fn supported(x: &Array, indices: &Array, wg: &Array, wu: &Array, wd: &Array) -> bool {
        super::supported(x, indices, wg, wu, wd)
    }

    /// Each projection is `(weight, scales, biases)` from `mlx_rs::ops::quantize(.., 64, 4)`.
    pub fn moe8(
        x: &Array,
        indices: &Array,
        gate: (&Array, &Array, &Array),
        up: (&Array, &Array, &Array),
        down: (&Array, &Array, &Array),
    ) -> Result<Array, Exception> {
        super::moe8_affine4(x, indices, gate, up, down)
    }
}
