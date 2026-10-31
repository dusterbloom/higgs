//! Affine 4-bit / group-64 matmul for 1..=8 token rows that reads each weight
//! once (MLX's small-M path costs ~3x the one-row time at 8 rows on M4).
//!
//! Technique from Kapitsa's `verify8.metal` (Apache-2.0, nace-ai/kapitsa):
//! the 8 token rows are one 8x8 `simdgroup_matrix` dimension, so each
//! simdgroup computes `Y^T[8 outputs, 8 rows] = W . X^T` on 8x8x8 MMAs with
//! - the weight codes as the A operand, exact in half as `1024 + q`
//!   (`(word & 0x000F000F) | 0x64006400`: no dequantization arithmetic);
//! - one MMA chain per 64-group closed by an fp32 epilogue
//!   `y += s * dot + (b - 1024 s) * sum(x)` over the group's half-rounded x;
//! - a permuted k order inside the group so a lane's A elements of the 8
//!   fragments are 16 contiguous codes of its weight row (8 bytes);
//! - an "X8 table": the inputs of a 64-group already in per-lane B-fragment
//!   order (16 halfs = two 16-byte loads per lane) with the rows' group sums,
//!   so the per-group input work (scattered loads, bf16 conversion, shuffle
//!   reductions) is not repeated for every output block. Kapitsa builds it in
//!   a prepare dispatch; here each threadgroup builds its chunk of `GS` groups
//!   in threadgroup memory (one dispatch; its 4 simdgroups x 2 fragments share
//!   each chunk).
//!
//! Two departures from Kapitsa, measured on M4 (10 cores, ~105 GB/s):
//! - Weight lines. Lane (fm, fn) needs 8 bytes of row fm per group, so the 4
//!   lanes of a row would touch a 128-byte line a quarter at a time over 4
//!   iterations (83 GB/s asymptotically). Instead each lane loads 32 contiguous
//!   bytes (group 4 j + fn / 2) once per 4 groups, the simdgroup stages its 16
//!   rows' lines in threadgroup memory and reads back the 8-byte pieces: 97-102
//!   GB/s on the 12288 x 8192 and 248320 x 2048 shapes.
//! - Up to 3 projections sharing `x` (q|k|v, gate|up, qkvz|ba) run as segments
//!   of one dispatch (one table, one launch), and a lone small-N projection
//!   (the GDN out projection, the shared expert) is k-sliced over `S`
//!   threadgroups along grid y with its fp32 partials summed in slice order by
//!   a second small kernel (deterministic, no atomics).
//!
//! Numerics differ from MLX's qmv (half activations), so this is only used
//! where outputs are never committed to a cache (Clover block preview).

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, error::Exception};

use crate::metal_kernel::{FastKernel, TemplateArg};

/// 64-groups per threadgroup-memory chunk (8 KB of table).
const GS: i32 = 8;
/// 8-row fragments per simdgroup (16 weight rows).
const NF: i32 = 2;
/// Simdgroups per threadgroup: with `GS` this keeps the threadgroup memory
/// (table + staged weight lines) at 16 KB, two threadgroups per core (M4: 8
/// simdgroups per threadgroup or a 16-group table cost 5-15%).
const CS: i32 = 4;
/// Simdgroups the grid aims for (~64 per core on a 10-core M4); below it, a
/// lone projection is k-sliced over more threadgroups (4 groups per slice at
/// least): 2048 x 4096 and the shared expert's 512-wide shapes, -0.5 ms per
/// verify pass on M4.
const TARGET_SIMDGROUPS: i32 = 768;

const HEADER: &str = r#"
// 8 consecutive 16-bit inputs (half or bfloat16 bits) as two half4.
template <typename T>
inline void q8_ld8(const device T* p, thread half4& lo, thread half4& hi) {
  const ushort4 u0 = reinterpret_cast<const device ushort4*>(p)[0];
  const ushort4 u1 = reinterpret_cast<const device ushort4*>(p)[1];
  if (metal::is_same<T, half>::value) {
    lo = as_type<half4>(u0);
    hi = as_type<half4>(u1);
  } else {
    lo = half4(as_type<float4>(uint4(u0) << 16));
    hi = half4(as_type<float4>(uint4(u1) << 16));
  }
}
// 8x8 fragment layout: lane (fm, fn) holds row fm, columns fn, fn + 1.
inline uint q8_lane_of(uint fm, uint fn) {
  return ((fm >> 2) << 4) | ((fn >> 2) << 3) | ((fm & 3) << 1) | ((fn >> 1) & 1);
}
// 8 table halfs: rows (a, b) interleaved over 4 consecutive inputs.
inline uint4 q8_pack(half4 a, half4 b) {
  return uint4(as_type<uint>(half2(a.x, b.x)), as_type<uint>(half2(a.y, b.y)),
               as_type<uint>(half2(a.z, b.z)), as_type<uint>(half2(a.w, b.w)));
}
// A scale or bias (half or bfloat16 bits) as float.
template <typename S>
inline float q8_coef(ushort v) {
  if (metal::is_same<S, half>::value) return float(as_type<half>(v));
  return as_type<float>(uint(v) << 16);
}
// codes . x over one 64-group: the 16 codes of this lane's weight row as the A
// operand (exact in half as 1024 + q), the lane's 16 table halfs as B.
inline float2 q8_mma_group(uint2 wq, uint4 t0, uint4 t1) {
  simdgroup_half8x8 A, B;
  simdgroup_float8x8 C;
  C.thread_elements()[0] = 0.0f;
  C.thread_elements()[1] = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 8; ++f) {
    const uint word = (f < 4 ? wq.x : wq.y) >> (4 * (f & 3));
    const half2 a = as_type<half2>((word & 0x000F000Fu) | 0x64006400u);
    const half2 b = as_type<half2>(f < 4 ? t0[f] : t1[f - 4]);
    A.thread_elements()[0] = a.x;
    A.thread_elements()[1] = a.y;
    B.thread_elements()[0] = b.x;
    B.thread_elements()[1] = b.y;
    simdgroup_multiply_accumulate(C, A, B, C);
  }
  return float2(C.thread_elements()[0], C.thread_elements()[1]);
}
"#;

/// Grid `(threadgroups of CS simdgroups over the segments' column blocks,
/// S k-slices)`: segment i (weights `w<i>`, N<i> rows, outputs `y<i>`) owns
/// `ceil(N<i> / (8 NF CS))` consecutive threadgroups along x; unused segments
/// have N = 0. `y[slice][row][n]` (`[row][n]` when S == 1).
const SOURCE: &str = r#"
  constexpr int G = K / 64;
  constexpr int GPS = (G / 4 + S - 1) / S * 4;
  constexpr int TGB = 8 * NF * CS;  // output rows per threadgroup
  constexpr int TG0 = (N0 + TGB - 1) / TGB, TG1 = (N1 + TGB - 1) / TGB;
  threadgroup uint4 tab[GS * 64];
  threadgroup float2 tsum[GS * 4];
  // One 128-byte line (4 groups) of each simdgroup's NF x 8 weight rows.
  threadgroup uint4 stage[CS * NF * 8 * 8];
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int slice = int(threadgroup_position_in_grid.y);
  const int gs0 = slice * GPS;
  const int gs1 = min(gs0 + GPS, G);
  int tg = int(threadgroup_position_in_grid.x);
  int seg = 0;
  if (tg >= TG0) {
    tg -= TG0;
    seg = 1;
    if (tg >= TG1) {
      tg -= TG1;
      seg = 2;
    }
  }
  const int N = seg == 0 ? N0 : seg == 1 ? N1 : N2;
  const device uint32_t* w = seg == 0 ? w0 : seg == 1 ? w1 : w2;
  const device ST* scales = seg == 0 ? s0 : seg == 1 ? s1 : s2;
  const device ST* biases = seg == 0 ? b0 : seg == 1 ? b1 : b2;
  device OutT* y = seg == 0 ? y0 : seg == 1 ? y1 : y2;
  const int n0 = (tg * CS + sg) * 8 * NF;
  const uint q = lane >> 2;
  const int fm = int((q & 4) | ((lane >> 1) & 3));
  const int fn = int(((q & 2) << 1) | ((lane & 1) << 1));
  const int c = fn >> 1;
  // Weight row n of fragment f: the lane loads group 4 j + c's 32 bytes of
  // every 4-group line (the 4 lanes of a row cover the line); its A operands
  // (codes [16 c, 16 c + 16) of each group) are read back from the staged line.
  const device uint4* wline[NF];
  const device ST* srow[NF];
  const device ST* brow[NF];
  float2 acc[NF];
  for (int f = 0; f < NF; ++f) {
    const int n = min(n0 + 8 * f + fm, N - 1);
    wline[f] = reinterpret_cast<const device uint4*>(w + size_t(n) * (K / 8)) + 2 * c;
    srow[f] = scales + size_t(n) * G;
    brow[f] = biases + size_t(n) * G;
    acc[f] = 0.0f;
  }
  threadgroup uint2* st = reinterpret_cast<threadgroup uint2*>(stage + sg * (NF * 8 * 8));
  // Table prep, lane = (row pair pp, input octet pj) of one group: inputs
  // 8 pj .. 8 pj + 7 of rows 2 pp, 2 pp + 1. k-slot fm of fragment f is
  // physical input 16 (fm / 2) + 8 (f / 4) + f % 4 + 4 (fm % 2), so the octet's
  // first four inputs are fragments 4 (pj % 2) .. + 3 of lane (2 (pj / 2), 2 pp)
  // and its last four those of lane (2 (pj / 2) + 1, 2 pp): one 16-byte store each.
  const int pj = int(lane & 7), pp = int(lane >> 3);
  const bool l0 = 2 * pp < M, l1 = 2 * pp + 1 < M;
  const device T* xr0 = x + size_t(l0 ? 2 * pp : 0) * K + 8 * pj;
  const device T* xr1 = x + size_t(l1 ? 2 * pp + 1 : 0) * K + 8 * pj;
  const uint laneA = q8_lane_of(uint(2 * (pj >> 1)), uint(2 * pp));
  // One 4-group step of weights ahead (the loads overlap the previous step's MMAs).
  uint4 wl[NF][2];
  ushort4 sl[NF], bl[NF];
  for (int f = 0; f < NF; ++f) {
    wl[f][0] = wline[f][gs0 * 2];
    wl[f][1] = wline[f][gs0 * 2 + 1];
    sl[f] = *reinterpret_cast<const device ushort4*>(srow[f] + gs0);
    bl[f] = *reinterpret_cast<const device ushort4*>(brow[f] + gs0);
  }
  for (int c0 = gs0; c0 < gs1; c0 += GS) {
    const int nch = min(GS, gs1 - c0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int gi = sg; gi < nch; gi += CS) {
      const int g = c0 + gi;
      half4 a0 = 0.0h, a1 = 0.0h, b0 = 0.0h, b1 = 0.0h;
      if (l0) q8_ld8<T>(xr0 + 64 * g, a0, a1);
      if (l1) q8_ld8<T>(xr1 + 64 * g, b0, b1);
      tab[(gi * 32 + laneA) * 2 + (pj & 1)] = q8_pack(a0, b0);
      tab[(gi * 32 + laneA + 2) * 2 + (pj & 1)] = q8_pack(a1, b1);
      // Group sums of the half-rounded inputs over the octets (lane bits 0-2).
      const float4 fa = float4(a0) + float4(a1), fb = float4(b0) + float4(b1);
      float2 sx = float2((fa.x + fa.y) + (fa.z + fa.w), (fb.x + fb.y) + (fb.z + fb.w));
      sx += simd_shuffle_xor(sx, 1);
      sx += simd_shuffle_xor(sx, 2);
      sx += simd_shuffle_xor(sx, 4);
      if (pj == 0) tsum[gi * 4 + pp] = sx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int gi = 0; gi < nch; gi += 4) {
      const int gn = min(c0 + gi + 4, G - 4);
      uint4 wln[NF][2];
      ushort4 sln[NF], bln[NF];
      for (int f = 0; f < NF; ++f) {
        wln[f][0] = wline[f][gn * 2];
        wln[f][1] = wline[f][gn * 2 + 1];
        sln[f] = *reinterpret_cast<const device ushort4*>(srow[f] + gn);
        bln[f] = *reinterpret_cast<const device ushort4*>(brow[f] + gn);
      }
      for (int f = 0; f < NF; ++f) {
        st[((f * 8 + fm) * 8 + 2 * c) * 2] = wl[f][0].xy;
        st[((f * 8 + fm) * 8 + 2 * c) * 2 + 1] = wl[f][0].zw;
        st[((f * 8 + fm) * 8 + 2 * c) * 2 + 2] = wl[f][1].xy;
        st[((f * 8 + fm) * 8 + 2 * c) * 2 + 3] = wl[f][1].zw;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      _Pragma("clang loop unroll(full)")
      for (int u = 0; u < 4; ++u) {
        const uint4 t0 = tab[((gi + u) * 32 + lane) * 2], t1 = tab[((gi + u) * 32 + lane) * 2 + 1];
        const float2 sx = tsum[(gi + u) * 4 + c];
        for (int f = 0; f < NF; ++f) {
          const uint2 wq = st[(f * 8 + fm) * 16 + u * 4 + c];
          const float s = q8_coef<ST>(sl[f][u]), b = q8_coef<ST>(bl[f][u]);
          acc[f] += s * q8_mma_group(wq, t0, t1) + (b - 1024.0f * s) * sx;
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int f = 0; f < NF; ++f) {
        wl[f][0] = wln[f][0];
        wl[f][1] = wln[f][1];
        sl[f] = sln[f];
        bl[f] = bln[f];
      }
    }
  }
  for (int f = 0; f < NF; ++f) {
    const int n = n0 + 8 * f + fm;
    if (n >= N) continue;
    if (fn < M) y[(size_t(slice) * M + fn) * N + n] = OutT(acc[f].x);
    if (fn + 1 < M) y[(size_t(slice) * M + fn + 1) * N + n] = OutT(acc[f].y);
  }
"#;

/// `y[i] = sum over the S slices of part[slice][i]` in slice order.
const REDUCE: &str = r"
  const uint i = thread_position_in_grid.x;
  if (i >= uint(MN)) return;
  float a = 0.0f;
  for (int s = 0; s < S; ++s) a += part[size_t(s) * MN + i];
  y[i] = OutT(a);
";

fn kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_qmm8_affine4_g64",
            &[
                c"x", c"w0", c"s0", c"b0", c"w1", c"s1", c"b1", c"w2", c"s2", c"b2",
            ],
            &[c"y0", c"y1", c"y2"],
            &std::ffi::CString::new(SOURCE).unwrap_or_default(),
            &std::ffi::CString::new(HEADER).unwrap_or_default(),
        )
    })
}

fn reduce_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_qmm8_slice_sum",
            &[c"part"],
            &[c"y"],
            &std::ffi::CString::new(REDUCE).unwrap_or_default(),
            c"",
        )
    })
}

/// `(weight, scales, biases)` of one affine-4bit g64 projection `[N, K / 8]`
/// u32 / `[N, K / 64]`.
pub(crate) type Segment<'a> = (&'a Array, &'a Array, &'a Array);

/// Shapes the kernel takes: `x` `[.., K]` 16-bit with 1..=8 rows, `w`
/// `[N, K / 8]` u32, K a multiple of 256 (4-group lines), N a multiple of 8.
pub(crate) fn supported(x: &Array, w: &Array, bits: i32, group_size: i32) -> bool {
    bits == 4 && group_size == 64 && supported_multi(x, &[w])
}

/// [`supported`] for 1..=3 weights sharing `x` (one dispatch, see
/// [`qmm8_affine4_multi`]).
pub(crate) fn supported_multi(x: &Array, ws: &[&Array]) -> bool {
    let rows: i32 = x.shape().iter().rev().skip(1).product();
    let Some(&k) = x.shape().last() else {
        return false;
    };
    (1..=3).contains(&ws.len())
        && (1..=8).contains(&rows)
        && matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16)
        && k % 256 == 0
        && ws.iter().all(|w| {
            matches!(w.shape(), [n, kp] if k == kp * 8 && n % 8 == 0) && w.dtype() == Dtype::Uint32
        })
}

/// Fragments per simdgroup, simdgroups per threadgroup, k-slices.
struct Plan {
    nf: i32,
    cs: i32,
    s: i32,
}

fn plan(n: i32, k: i32) -> Plan {
    let nf = if n >= 128 { NF } else { 1 };
    let blocks = (n + 8 * nf - 1) / (8 * nf);
    let cs = blocks.min(CS);
    let max_slices = (k / 256).max(1);
    let s = ((TARGET_SIMDGROUPS + blocks - 1) / blocks).clamp(1, max_slices);
    Plan { nf, cs, s }
}

/// `x @ dequant(w)^T` for 1..=8 rows; output `[.., N]` in `scales`' dtype.
pub(crate) fn qmm8_affine4(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
) -> Result<Array, Exception> {
    qmm8_affine4_multi(x, &[(w, scales, biases)])?
        .pop()
        .ok_or_else(|| Exception::custom("qmm8: no output"))
}

/// `x @ dequant(w_i)^T` for 1..=3 projections sharing `x` (same K) in one
/// dispatch (the input table is built once; e.g. q|k|v, gate|up); one output
/// `[.., N_i]` per segment, in its `scales`' dtype. Call only when
/// [`supported_multi`].
pub(crate) fn qmm8_affine4_multi(x: &Array, segs: &[Segment<'_>]) -> Result<Vec<Array>, Exception> {
    let k = *x.shape().last().unwrap_or(&0);
    let rows: i32 = x.shape().iter().rev().skip(1).product();
    let x2 = x.reshape(&[rows, k])?;
    let Some(&first) = segs.first() else {
        return Ok(Vec::new());
    };
    let ns: Vec<i32> = segs.iter().map(|(w, _, _)| w.dim(0)).collect();
    let out_dtype = first.1.dtype();
    let Plan { nf, cs, s: slices } = plan(ns.iter().sum(), k);
    // k-slices only for a lone projection: the fused ones are wide enough.
    let s = if segs.len() == 1 { slices } else { 1 };
    let tgb = 8 * nf * cs;
    let tg_x: i32 = ns.iter().map(|n| (n + tgb - 1) / tgb).sum();
    // Unused segments bind segment 0's arrays, have N = 0 and a 1-element output.
    let seg = |i: usize| segs.get(i).copied().unwrap_or(first);
    let n_of = |i: usize| ns.get(i).copied().unwrap_or(0);
    let dtype = if s > 1 { Dtype::Float32 } else { out_dtype };
    let shapes: Vec<Vec<i32>> = (0..3)
        .map(|i| match ns.get(i) {
            Some(&n) if s > 1 => vec![s, rows, n],
            Some(&n) => vec![rows, n],
            None => vec![1],
        })
        .collect();
    let outputs: Vec<(&[i32], Dtype)> = shapes.iter().map(|sh| (sh.as_slice(), dtype)).collect();
    let (w0, s0, b0) = seg(0);
    let (w1, s1, b1) = seg(1);
    let (w2, s2, b2) = seg(2);
    let mut out = kernel().apply(
        &[&x2, w0, s0, b0, w1, s1, b1, w2, s2, b2],
        &[
            TemplateArg::Dtype(c"T", x.dtype()),
            TemplateArg::Dtype(c"ST", s0.dtype()),
            TemplateArg::Dtype(c"OutT", dtype),
            TemplateArg::Int(c"K", k),
            TemplateArg::Int(c"N0", n_of(0)),
            TemplateArg::Int(c"N1", n_of(1)),
            TemplateArg::Int(c"N2", n_of(2)),
            TemplateArg::Int(c"M", rows),
            TemplateArg::Int(c"NF", nf),
            TemplateArg::Int(c"CS", cs),
            TemplateArg::Int(c"S", s),
            TemplateArg::Int(c"GS", GS),
        ],
        [tg_x * 32 * cs, s, 1],
        [32 * cs, 1, 1],
        &outputs,
    )?;
    out.truncate(ns.len());
    let mut shape = x.shape().to_vec();
    out.into_iter()
        .zip(&ns)
        .map(|(mut y, &n)| {
            if s > 1 {
                let mn = rows * n;
                y = reduce_kernel()
                    .apply(
                        &[&y],
                        &[
                            TemplateArg::Dtype(c"OutT", out_dtype),
                            TemplateArg::Int(c"S", s),
                            TemplateArg::Int(c"MN", mn),
                        ],
                        [(mn + 255) / 256 * 256, 1, 1],
                        [256, 1, 1],
                        &[(&[rows, n], out_dtype)],
                    )?
                    .remove(0);
            }
            if let Some(last) = shape.last_mut() {
                *last = n;
            }
            y.reshape(&shape)
        })
        .collect()
}

#[doc(hidden)]
pub mod probe {
    //! Test seam (higgs-models unit tests do not compile).
    use super::{Array, Exception};

    pub fn supported(x: &Array, w: &Array) -> bool {
        super::supported(x, w, 4, 64)
    }

    pub fn qmm8(x: &Array, w: &Array, s: &Array, b: &Array) -> Result<Array, Exception> {
        super::qmm8_affine4(x, w, s, b)
    }

    /// One dispatch for 1..=3 `(w, scales, biases)` sharing `x`.
    pub fn qmm8_multi(
        x: &Array,
        segs: &[(&Array, &Array, &Array)],
    ) -> Result<Vec<Array>, Exception> {
        super::qmm8_affine4_multi(x, segs)
    }

    /// `(fragments per simdgroup, simdgroups per threadgroup, k-slices)` the
    /// kernel is launched with for an `[N, K]` weight.
    pub fn plan(n: i32, k: i32) -> (i32, i32, i32) {
        let p = super::plan(n, k);
        (p.nf, p.cs, p.s)
    }
}
