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
//!   fragments are 16 contiguous codes of its weight row (one 8-byte load).
//!
//! Numerics differ from MLX's qmv (half activations), so this is only used
//! where outputs are never committed to a cache (Clover block preview).

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, error::Exception};

use crate::metal_kernel::{FastKernel, TemplateArg};

/// Simdgroups per threadgroup (8 output rows each).
const NSG: i32 = 4;

const SOURCE: &str = r"
  const uint lane = thread_index_in_simdgroup;
  const int n0 = (int(threadgroup_position_in_grid.x) * NSG + int(simdgroup_index_in_threadgroup)) * 8;
  if (n0 >= N) return;
  // 8x8 fragment layout: lane holds row fm, columns fn, fn + 1.
  const uint q = lane >> 2;
  const int fm = int((q & 4) | ((lane >> 1) & 3));
  const int fn = int(((q & 2) << 1) | ((lane & 1) << 1));
  constexpr int G = K / 64;
  const int n = n0 + fm;
  // A: weight row n, codes [16 c, 16 c + 16) of each group, c = fn / 2.
  const device uint2* wrow = reinterpret_cast<const device uint2*>(w + size_t(n) * (K / 8)) + (fn >> 1);
  const device ST* srow = scales + size_t(n) * G;
  const device ST* brow = biases + size_t(n) * G;
  // B: k-slot fm of fragment f is physical input 16 (fm / 2) + 8 (f / 4) + f % 4 + 4 (fm % 2).
  const int kb = 16 * (fm >> 1) + 4 * (fm & 1);
  const bool v0 = fn < M, v1 = fn + 1 < M;
  const device T* x0 = x + size_t(v0 ? fn : 0) * K + kb;
  const device T* x1 = x + size_t(v1 ? fn + 1 : 0) * K + kb;

  float2 acc = 0.0f;
  for (int g = 0; g < G; ++g) {
    const uint2 wq = wrow[g * 4];
    half xa[8], xb[8];
    for (int i = 0; i < 4; ++i) {
      xa[i] = v0 ? half(x0[64 * g + i]) : half(0);
      xa[4 + i] = v0 ? half(x0[64 * g + 8 + i]) : half(0);
      xb[i] = v1 ? half(x1[64 * g + i]) : half(0);
      xb[4 + i] = v1 ? half(x1[64 * g + 8 + i]) : half(0);
    }
    // Group sums per token row: the 8 lanes sharing fn (lane bits 1, 2, 4) hold all 64 inputs.
    float2 sx = 0.0f;
    for (int i = 0; i < 8; ++i) sx += float2(float(xa[i]), float(xb[i]));
    sx += simd_shuffle_xor(sx, 2);
    sx += simd_shuffle_xor(sx, 4);
    sx += simd_shuffle_xor(sx, 16);

    simdgroup_half8x8 A, B;
    simdgroup_float8x8 C;
    C.thread_elements()[0] = 0.0f;
    C.thread_elements()[1] = 0.0f;
    for (int f = 0; f < 8; ++f) {
      const uint word = (f < 4 ? wq.x : wq.y) >> (4 * (f & 3));
      const half2 a = as_type<half2>((word & 0x000F000Fu) | 0x64006400u);
      A.thread_elements()[0] = a.x;
      A.thread_elements()[1] = a.y;
      B.thread_elements()[0] = xa[f];
      B.thread_elements()[1] = xb[f];
      simdgroup_multiply_accumulate(C, A, B, C);
    }
    const float s = float(srow[g]);
    const float b = float(brow[g]);
    acc += s * float2(C.thread_elements()[0], C.thread_elements()[1]) + (b - 1024.0f * s) * sx;
  }
  if (v0) y[size_t(fn) * N + n] = OutT(acc.x);
  if (v1) y[size_t(fn + 1) * N + n] = OutT(acc.y);
";

fn kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        let header = format!("constant constexpr int NSG = {NSG};\n");
        FastKernel::new(
            c"higgs_qmm8_affine4_g64",
            &[c"x", c"w", c"scales", c"biases"],
            &[c"y"],
            &std::ffi::CString::new(SOURCE).unwrap_or_default(),
            &std::ffi::CString::new(header).unwrap_or_default(),
        )
    })
}

/// Shapes the kernel takes: `x` `[.., K]` with 1..=8 rows, `w` `[N, K / 8]`
/// u32, K a multiple of 64, N a multiple of 8.
pub(crate) fn supported(x: &Array, w: &Array, bits: i32, group_size: i32) -> bool {
    let rows: i32 = x.shape().iter().rev().skip(1).product();
    let (Some(&k), [n, kp]) = (x.shape().last(), w.shape()) else {
        return false;
    };
    bits == 4
        && group_size == 64
        && (1..=8).contains(&rows)
        && w.dtype() == Dtype::Uint32
        && k == kp * 8
        && k % 64 == 0
        && n % 8 == 0
}

/// `x @ dequant(w)^T` for 1..=8 rows; output `[.., N]` in `scales`' dtype.
pub(crate) fn qmm8_affine4(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
) -> Result<Array, Exception> {
    let k = *x.shape().last().unwrap_or(&0);
    let n = *w.shape().first().unwrap_or(&0);
    let rows: i32 = x.shape().iter().rev().skip(1).product();
    let x2 = x.reshape(&[rows, k])?;
    let out_dtype = scales.dtype();
    let mut out = kernel().apply(
        &[&x2, w, scales, biases],
        &[
            TemplateArg::Dtype(c"T", x.dtype()),
            TemplateArg::Dtype(c"ST", scales.dtype()),
            TemplateArg::Dtype(c"OutT", out_dtype),
            TemplateArg::Int(c"K", k),
            TemplateArg::Int(c"N", n),
            TemplateArg::Int(c"M", rows),
        ],
        [n / 8 * 32, 1, 1],
        [NSG * 32, 1, 1],
        &[(&[rows, n], out_dtype)],
    )?;
    let mut shape = x.shape().to_vec();
    if let Some(last) = shape.last_mut() {
        *last = n;
    }
    out.remove(0).reshape(&shape)
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
}
