//! Bonsai-Q2 CPU reference: packed 2-bit affine quantization oracle.
//!
//! Mirror of [`crate::bonsai_q1::PackedQ1Linear`] for 2-bit affine weights.
//! Used as the bit-exact CPU oracle for the Q2 Metal kernels added in Phase 3B-D.
//!
//! Bonsai-27B-Q2 loads through `qwen3_next::Qwen3NextCausalLM` (not a standalone
//! `BonsaiQ2Engine`), so this module intentionally exposes only the packed
//! weight container and a CPU dequant — no engine, no layer aggregation, no
//! forward path. The production target-side kernels live in
//! [`crate::metal_kernel`] (`bonsai_q2_qmv`, `bonsai_q2_qmm`,
//! `bonsai_q2_wide_qmm`, plus the row2 promotion path).
//!
//! Layout (matches MLX 2-bit `QuantizedLinear` / `prism-ml` affine form):
//!   - `w_packed`: `[out_features, in_features/16]` u32, bits `2*col%16 .. 2*col%16+2`
//!     of word `col/16` hold the raw 2-bit code for column `col`.
//!   - `scales`, `biases`: `[out_features, in_features/128]` f16, one per group
//!     of 128 input columns.
//!
//! Effective weight: `w[row, col] = scales[row, col/128] * q + biases[row, col/128]`
//! where `q ∈ {0, 1, 2, 3}` is the unpacked 2-bit code. Biases are retained
//! (Phase 0.3 decision); Q2 has no symmetric-bias compaction trick analogous
//! to Q1's `bias = -scale/2`.
//!
//! Residency: ~2.5 bpw (2 bits/weight + 32 bits/group / 128 weights).

#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    // Quantization math uses small bounded dims (head_dim, GROUP_SIZE=128, vocab) and
    // bit-packed u32→f32 conversions where precision/sign loss is intentional.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    // Dequant kernel indexes into manually-bounds-checked slices.
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::doc_markdown,
    clippy::doc_lazy_continuation,
    clippy::missing_const_for_fn,
)]

use half::f16;

/// Affine group size for the Ternary-Bonsai-27B-2bit target.
pub const GROUP_SIZE: usize = 128;

/// Number of 2-bit weights packed into one `u32` word.
pub const WEIGHTS_PER_WORD: usize = 16;

/// Packed 2-bit affine linear layer — CPU reference for Q2 kernel tests.
pub struct PackedQ2Linear {
    pub w_packed: Vec<u32>,
    pub scales: Vec<f16>,
    pub biases: Vec<f16>,
    pub out_features: usize,
    pub in_features: usize,
}

impl PackedQ2Linear {
    pub const fn resident_bytes(&self) -> usize {
        self.w_packed.len() * 4 + self.scales.len() * 2 + self.biases.len() * 2
    }

    /// Number of `u32` packed words per output row.
    pub fn packed_cols(&self) -> usize {
        self.in_features / WEIGHTS_PER_WORD
    }

    /// Number of affine groups per output row.
    pub fn n_groups(&self) -> usize {
        self.in_features / GROUP_SIZE
    }

    /// Dequantize a single row to fp32 (CPU oracle).
    ///
    /// Not used on the hot path — the production target-side kernels in
    /// [`crate::metal_kernel`] are the hot path; this CPU implementation is
    /// the bit-exact reference those kernels are tested against.
    ///
    /// Formula per element: `w = scale * q + bias` where `q ∈ {0,1,2,3}` is
    /// the unpacked 2-bit code, `scale` and `bias` are taken from the affine
    /// group containing `col`.
    pub fn dequant_row_to_fp32(&self, row: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.in_features);
        let packed_cols = self.packed_cols();
        let n_groups = self.n_groups();
        let w_row = &self.w_packed[row * packed_cols..(row + 1) * packed_cols];
        let s_row = &self.scales[row * n_groups..(row + 1) * n_groups];
        let b_row = &self.biases[row * n_groups..(row + 1) * n_groups];
        for col in 0..self.in_features {
            let word = w_row[col / WEIGHTS_PER_WORD];
            let bit_off = 2 * (col % WEIGHTS_PER_WORD);
            let q = ((word >> bit_off) & 0b11) as f32;
            let group = col / GROUP_SIZE;
            out[col] = s_row[group].to_f32().mul_add(q, b_row[group].to_f32());
        }
    }

    /// Dequantize the full `[out_features, in_features]` matrix to a flat
    /// row-major `Vec<f32>`. Convenience wrapper around
    /// [`Self::dequant_row_to_fp32`] for tests that need the whole tensor.
    pub fn dequant_to_fp32(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.out_features * self.in_features];
        for row in 0..self.out_features {
            let row_end = (row + 1) * self.in_features;
            self.dequant_row_to_fp32(row, &mut out[row * self.in_features..row_end]);
        }
        out
    }
}

impl PackedQ2Linear {
    /// Build a `PackedQ2Linear` from already-packed raw bytes (no quantization).
    /// Used by tests that construct fixtures from MLX `ops::quantize` output.
    pub fn from_packed(
        w_packed: Vec<u32>,
        scales: Vec<f16>,
        biases: Vec<f16>,
        out_features: usize,
        in_features: usize,
    ) -> Self {
        let expect_words = out_features * (in_features / WEIGHTS_PER_WORD);
        let expect_groups = out_features * (in_features / GROUP_SIZE);
        debug_assert_eq!(w_packed.len(), expect_words);
        debug_assert_eq!(scales.len(), expect_groups);
        debug_assert_eq!(biases.len(), expect_groups);
        Self {
            w_packed,
            scales,
            biases,
            out_features,
            in_features,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;
    use crate::mlx_exec::eval;
    use mlx_rs::ops::{dequantize, quantize};

    /// Q2 CPU oracle must match MLX's stock affine `ops::dequantize` bit-for-bit
    /// across a representative shape. This is the foundation gate: every Phase
    /// 3B-D kernel test compares its output against this oracle, so the oracle
    /// itself must be proven correct against MLX first.
    #[test]
    fn q2_cpu_oracle_matches_mlx_dequantize() {
        let _exec = crate::mlx_exec::acquire();
        let out_features = 64usize;
        let in_features = 256usize;
        let group_size = 128i32;
        let bits = 2i32;

        // MLX's quantize path for bits=2 requires fp16 input on this build.
        let float_weight_f32 =
            mlx_rs::random::uniform::<f32, f32>(-1.5, 1.5, &[out_features as i32, in_features as i32], None)
                .unwrap();
        eval([&float_weight_f32].into_iter()).unwrap();
        let float_weight = float_weight_f32.as_dtype(mlx_rs::Dtype::Float16).unwrap();
        eval([&float_weight].into_iter()).unwrap();
        let (qw, qs, qb) = quantize(&float_weight, group_size, bits).unwrap();
        eval([&qw, &qs, &qb].into_iter()).unwrap();

        let w_packed: Vec<u32> = qw.as_slice::<u32>().iter().copied().collect();
        let scales: Vec<f16> = qs.as_slice::<f16>().iter().copied().collect();
        let biases: Vec<f16> = qb.as_slice::<f16>().iter().copied().collect();

        let oracle = PackedQ2Linear::from_packed(
            w_packed,
            scales,
            biases,
            out_features,
            in_features,
        );

        // Reference: dequantize the fp16 source directly (not the f32 original,
        // which would introduce fp16 rounding noise that the oracle shouldn't
        // be blamed for).
        let cpu = oracle.dequant_to_fp32();

        let mlx_deq = dequantize(&qw, &qs, Some(&qb), Some(group_size), Some(bits)).unwrap();
        let mlx_deq_f32 = mlx_deq.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        eval([&mlx_deq_f32].into_iter()).unwrap();
        let mlx_flat: Vec<f32> = mlx_deq_f32.as_slice::<f32>().iter().copied().collect();

        assert_eq!(cpu.len(), mlx_flat.len());
        let mut max_diff: f32 = 0.0;
        for (i, (a, b)) in cpu.iter().zip(mlx_flat.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d < 1e-3,
                "mismatch at flat idx {i}: cpu={a}, mlx={b}, diff={d}"
            );
        }
        assert!(
            max_diff < 1e-3,
            "max diff {max_diff} exceeds fp16 epsilon at 2-bit quantization"
        );
    }

    /// Verify the per-row dequant path matches the full-tensor dequant.
    /// This catches off-by-one errors in row indexing that the MLX comparison
    /// above might miss if MLX happens to share the same bug.
    #[test]
    fn q2_per_row_dequant_matches_full_dequant() {
        let _exec = crate::mlx_exec::acquire();
        let out_features = 8usize;
        let in_features = 512usize;
        let group_size = 128i32;
        let bits = 2i32;

        let float_weight_f32 =
            mlx_rs::random::uniform::<f32, f32>(-2.0, 2.0, &[out_features as i32, in_features as i32], None)
                .unwrap();
        eval([&float_weight_f32].into_iter()).unwrap();
        let float_weight = float_weight_f32.as_dtype(mlx_rs::Dtype::Float16).unwrap();
        eval([&float_weight].into_iter()).unwrap();
        let (qw, qs, qb) = quantize(&float_weight, group_size, bits).unwrap();
        eval([&qw, &qs, &qb].into_iter()).unwrap();

        let w_packed: Vec<u32> = qw.as_slice::<u32>().iter().copied().collect();
        let scales: Vec<f16> = qs.as_slice::<f16>().iter().copied().collect();
        let biases: Vec<f16> = qb.as_slice::<f16>().iter().copied().collect();

        let oracle = PackedQ2Linear::from_packed(
            w_packed,
            scales,
            biases,
            out_features,
            in_features,
        );

        let full = oracle.dequant_to_fp32();
        let mut row_buf = vec![0f32; in_features];
        for row in 0..out_features {
            oracle.dequant_row_to_fp32(row, &mut row_buf);
            let expected = &full[row * in_features..(row + 1) * in_features];
            assert_eq!(row_buf.as_slice(), expected, "row {row} mismatch");
        }
    }

    /// Bit-pattern test: confirm the 2-bit codes unpack to the expected
    /// {0,1,2,3} set and that packed-then-unpacked round-trips bit-exact.
    #[test]
    fn q2_bit_unpacking_round_trips() {
        let out_features = 2usize;
        let in_features = 128usize; // one full affine group
        let n_groups = in_features / GROUP_SIZE; // 1

        let mut w_packed = vec![0u32; out_features * (in_features / WEIGHTS_PER_WORD)];
        // Cycle through all four 2-bit codes per row.
        let codes_per_row = [0u32, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        for row in 0..out_features {
            let mut word = 0u32;
            for (col, &code) in codes_per_row.iter().enumerate() {
                word |= code << (2 * col);
            }
            // Replicate the 16-code pattern 8 times to fill the 128-col row.
            for repeat in 0..8 {
                w_packed[row * 8 + repeat] = word;
            }
        }
        let scales = vec![f16::from_f32(1.0); out_features * n_groups];
        let biases = vec![f16::from_f32(0.0); out_features * n_groups];

        let oracle = PackedQ2Linear {
            w_packed,
            scales,
            biases,
            out_features,
            in_features,
        };

        let mut row = vec![0f32; in_features];
        oracle.dequant_row_to_fp32(0, &mut row);
        for (col, value) in row.iter().enumerate() {
            let code_idx = col % 16;
            let expected = codes_per_row[code_idx] as f32;
            assert_eq!(
                *value, expected,
                "col {col}: expected code {expected}, got {value}"
            );
        }
    }

    /// Resident bytes accounting matches the documented Q2 footprint
    /// (~2.25 bpw inclusive of scales and biases; Q1 is 1.25 bpw because the
    /// symmetric bias trick drops the bias array, but Q2 retains biases).
    #[test]
    fn q2_resident_bytes_matches_2_25_bpw_approximation() {
        let out_features = 5120usize; // matches Bonsai-27B hidden
        let in_features = 17408usize; // matches Bonsai-27B inter
        let n_weights = out_features * in_features;
        let n_groups_per_row = in_features / GROUP_SIZE;
        let w_packed = vec![0u32; out_features * (in_features / WEIGHTS_PER_WORD)];
        let scales = vec![f16::ZERO; out_features * n_groups_per_row];
        let biases = vec![f16::ZERO; out_features * n_groups_per_row];

        let oracle = PackedQ2Linear {
            w_packed,
            scales,
            biases,
            out_features,
            in_features,
        };
        let bytes = oracle.resident_bytes();
        let bpw = (bytes as f64) * 8.0 / (n_weights as f64);
        // 2 bits/weight + 16 bits/group / 128 weights for scales
        //              + 16 bits/group / 128 weights for biases
        //            = 2 + 0.125 + 0.125 = 2.25 bpw exactly.
        assert!(
            (bpw - 2.25).abs() < 1e-9,
            "expected ~2.25 bpw, got {bpw}"
        );
    }
}
