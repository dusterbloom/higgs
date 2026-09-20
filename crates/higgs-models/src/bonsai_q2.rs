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
    use mlx_rs::ops::{dequantize, indexing::IndexOp, quantize};

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
        let float_weight_f32 = mlx_rs::random::uniform::<f32, f32>(
            -1.5,
            1.5,
            &[out_features as i32, in_features as i32],
            None,
        )
        .unwrap();
        eval([&float_weight_f32].into_iter()).unwrap();
        let float_weight = float_weight_f32.as_dtype(mlx_rs::Dtype::Float16).unwrap();
        eval([&float_weight].into_iter()).unwrap();
        let (qw, qs, qb) = quantize(&float_weight, group_size, bits).unwrap();
        eval([&qw, &qs, &qb].into_iter()).unwrap();

        let w_packed: Vec<u32> = qw.as_slice::<u32>().iter().copied().collect();
        let scales: Vec<f16> = qs.as_slice::<f16>().iter().copied().collect();
        let biases: Vec<f16> = qb.as_slice::<f16>().iter().copied().collect();

        let oracle =
            PackedQ2Linear::from_packed(w_packed, scales, biases, out_features, in_features);

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

        let float_weight_f32 = mlx_rs::random::uniform::<f32, f32>(
            -2.0,
            2.0,
            &[out_features as i32, in_features as i32],
            None,
        )
        .unwrap();
        eval([&float_weight_f32].into_iter()).unwrap();
        let float_weight = float_weight_f32.as_dtype(mlx_rs::Dtype::Float16).unwrap();
        eval([&float_weight].into_iter()).unwrap();
        let (qw, qs, qb) = quantize(&float_weight, group_size, bits).unwrap();
        eval([&qw, &qs, &qb].into_iter()).unwrap();

        let w_packed: Vec<u32> = qw.as_slice::<u32>().iter().copied().collect();
        let scales: Vec<f16> = qs.as_slice::<f16>().iter().copied().collect();
        let biases: Vec<f16> = qb.as_slice::<f16>().iter().copied().collect();

        let oracle =
            PackedQ2Linear::from_packed(w_packed, scales, biases, out_features, in_features);

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
        assert!((bpw - 2.25).abs() < 1e-9, "expected ~2.25 bpw, got {bpw}");
    }

    // -----------------------------------------------------------------
    // Phase 3B kernel bit-exactness tests (against the CPU oracle).
    // -----------------------------------------------------------------

    /// Deterministic PRNG (SplitMix-ish LCG), mirrors `bonsai_q1::tests::lcg`.
    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 32) as u32
    }

    /// Build a deterministic Q2 fixture with per-(row,group) scales/biases so
    /// a wrong group index produces a clear mismatch. Magnitudes are small and
    /// signed, like real affine params. Matches `bonsai_q1::make_packed`.
    fn make_packed_q2(out_features: usize, in_features: usize, seed: u64) -> PackedQ2Linear {
        let packed_cols = in_features / WEIGHTS_PER_WORD;
        let n_groups = in_features / GROUP_SIZE;
        let mut st = seed;
        let w_packed: Vec<u32> = (0..out_features * packed_cols)
            .map(|_| lcg(&mut st))
            .collect();
        let scales: Vec<f16> = (0..out_features * n_groups)
            .map(|i| f16::from_f32(0.05 + 0.013 * ((i % 7) as f32)))
            .collect();
        let biases: Vec<f16> = (0..out_features * n_groups)
            .map(|i| f16::from_f32(-0.03 + 0.011 * ((i % 5) as f32)))
            .collect();
        PackedQ2Linear {
            w_packed,
            scales,
            biases,
            out_features,
            in_features,
        }
    }

    /// CPU reference matvec: y = dequant(W) · x for one row of x.
    fn dense_matvec_reference(p: &PackedQ2Linear, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; p.out_features];
        let mut w_row = vec![0f32; p.in_features];
        for r in 0..p.out_features {
            p.dequant_row_to_fp32(r, &mut w_row);
            let mut acc = 0f32;
            for c in 0..p.in_features {
                acc += x[c] * w_row[c];
            }
            y[r] = acc;
        }
        y
    }

    fn qmm_error(actual: &mlx_rs::Array, reference: &[f32]) -> (f32, f32) {
        let actual = actual.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        actual.eval().unwrap();
        let got = actual.as_slice::<f32>();
        assert_eq!(got.len(), reference.len());
        let mut max_abs = 0.0_f32;
        let mut error_sq = 0.0_f32;
        let mut reference_sq = 0.0_f32;
        for (&value, &want) in got.iter().zip(reference) {
            let error = value - want;
            max_abs = max_abs.max(error.abs());
            error_sq += error * error;
            reference_sq += want * want;
            assert!(value.is_finite());
        }
        (max_abs, (error_sq / reference_sq.max(1.0e-12)).sqrt())
    }

    fn make_ternary_q2(out_features: usize, in_features: usize, seed: u64) -> PackedQ2Linear {
        let packed_cols = in_features / WEIGHTS_PER_WORD;
        let n_groups = in_features / GROUP_SIZE;
        let mut st = seed;
        let w_packed: Vec<u32> = (0..out_features * packed_cols)
            .map(|_| {
                let mut word = 0u32;
                for shift in 0..16 {
                    word |= (lcg(&mut st) % 3) << (2 * shift);
                }
                word
            })
            .collect();
        let scales: Vec<f16> = (0..out_features * n_groups)
            .map(|i| f16::from_f32(0.05 + 0.013 * ((i % 7) as f32)))
            .collect();
        let biases = scales
            .iter()
            .map(|&scale| f16::from_f32(-scale.to_f32()))
            .collect();
        PackedQ2Linear {
            w_packed,
            scales,
            biases,
            out_features,
            in_features,
        }
    }

    /// Upload a PackedQ2Linear's packed weight, scales, and biases to MLX
    /// arrays matching the kernel's expected layout.
    fn upload_to_mlx(p: &PackedQ2Linear) -> (mlx_rs::Array, mlx_rs::Array, mlx_rs::Array) {
        use mlx_rs::Array;
        let packed_cols = p.packed_cols();
        let n_groups = p.n_groups();
        let w = Array::from_slice(&p.w_packed, &[p.out_features as i32, packed_cols as i32]);
        let s = Array::from_slice(&p.scales, &[p.out_features as i32, n_groups as i32]);
        let b = Array::from_slice(&p.biases, &[p.out_features as i32, n_groups as i32]);
        (w, s, b)
    }

    fn make_q4_arrays(
        out_features: usize,
        in_features: usize,
        group_size: usize,
        seed: u64,
    ) -> (mlx_rs::Array, mlx_rs::Array, mlx_rs::Array) {
        let packed_cols = in_features / 8;
        let n_groups = in_features / group_size;
        let mut st = seed;
        let w_packed: Vec<u32> = (0..out_features * packed_cols)
            .map(|_| lcg(&mut st))
            .collect();
        let scales: Vec<f16> = (0..out_features * n_groups)
            .map(|i| f16::from_f32(0.015 + 0.004 * ((i % 11) as f32)))
            .collect();
        let biases: Vec<f16> = (0..out_features * n_groups)
            .map(|i| f16::from_f32(-0.02 + 0.003 * ((i % 13) as f32)))
            .collect();
        let w = mlx_rs::Array::from_slice(&w_packed, &[out_features as i32, packed_cols as i32]);
        let s = mlx_rs::Array::from_slice(&scales, &[out_features as i32, n_groups as i32]);
        let b = mlx_rs::Array::from_slice(&biases, &[out_features as i32, n_groups as i32]);
        (w, s, b)
    }

    /// Phase 3B foundation gate: `bonsai_q2_qmv` must match the CPU oracle
    /// bit-for-bit (within fp16 epsilon) across multiple shapes including the
    /// dominant Bonsai-27B verifier shapes.

    /// Phase C1' go/no-go: stock affine-2bit qmm vs convert-unpack JIT vs
    /// staged-x JIT at the checkpoint's dominant decode shapes (M=1, f16).
    /// Run: cargo test --release -p higgs-models q2_qmv_go_no_go -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q2_qmv_go_no_go() {
        let _exec = crate::mlx_exec::acquire();
        use std::time::Instant;

        let mut st = 0x7777_7777_u64;
        let lcg = |st: &mut u64| {
            *st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*st >> 32) as u32)
        };
        let f16_vec = |n: usize, st: &mut u64| -> Vec<half::f16> {
            (0..n)
                .map(|_| half::f16::from_f32((lcg(st) as f32 / u32::MAX as f32).mul_add(0.2, -0.1)))
                .collect()
        };
        let time = |f: &mut dyn FnMut() -> mlx_rs::Array| {
            let _ = f().eval().unwrap();
            let t0 = Instant::now();
            for _ in 0..30 {
                let _ = f().eval().unwrap();
            }
            t0.elapsed().as_micros() as f64 / 30.0
        };

        for (rows, k, label) in [
            (34_816usize, 5_120usize, "gate_up-fused"),
            (16_384, 5_120, "qkvz-fused"),
            (17_408, 5_120, "gate/up"),
            (5_120, 17_408, "down"),
        ] {
            let packed_cols = k / 16;
            let n_groups = k / GROUP_SIZE;
            let w = mlx_rs::Array::from_slice(
                &(0..rows * packed_cols).map(|_| lcg(&mut st)).collect::<Vec<u32>>(),
                &[rows as i32, packed_cols as i32],
            );
            let s = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let b = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let x = mlx_rs::Array::from_slice(&f16_vec(k, &mut st), &[1, k as i32]);

            let stock = time(&mut || {
                crate::quant_mode::quantized_matmul(
                    &x, &w, &s, Some(&b), true, 128, 2, crate::quant_mode::QuantMode::Affine,
                )
                .unwrap()
            });
            let convert = time(&mut || {
                crate::metal_kernel::bonsai_q2_qmv(&x, &w, &s, &b, GROUP_SIZE as i32).unwrap()
            });
            let staged = time(&mut || {
                crate::metal_kernel::bonsai_q2_qmv_staged(&x, &w, &s, &b, GROUP_SIZE as i32)
                    .unwrap()
            });
            println!(
                "Q2PROBE {label} [{rows},K={k}] M=1: stock={stock:.0}us convert={convert:.0}us staged={staged:.0}us staged-vs-stock={:.2}x",
                stock / staged
            );
        }

        // bf16 activation sweep (server reality): the two fused shapes only.
        for (rows, k, label) in [(34_816usize, 5_120usize, "gate_up-fused"), (16_384, 5_120, "qkvz-fused")] {
            let packed_cols = k / 16;
            let n_groups = k / GROUP_SIZE;
            let w = mlx_rs::Array::from_slice(
                &(0..rows * packed_cols).map(|_| lcg(&mut st)).collect::<Vec<u32>>(),
                &[rows as i32, packed_cols as i32],
            );
            let s = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let b = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let x = mlx_rs::Array::from_slice(&f16_vec(k, &mut st), &[1, k as i32])
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap();

            let stock = time(&mut || {
                crate::quant_mode::quantized_matmul(
                    &x, &w, &s, Some(&b), true, 128, 2, crate::quant_mode::QuantMode::Affine,
                )
                .unwrap()
            });
            let convert = time(&mut || {
                crate::metal_kernel::bonsai_q2_qmv(&x, &w, &s, &b, GROUP_SIZE as i32).unwrap()
            });
            let staged = time(&mut || {
                crate::metal_kernel::bonsai_q2_qmv_staged(&x, &w, &s, &b, GROUP_SIZE as i32)
                    .unwrap()
            });
            println!(
                "Q2PROBE-BF16 {label} [{rows},K={k}] M=1: stock={stock:.0}us convert={convert:.0}us staged={staged:.0}us convert-vs-stock={:.2}x",
                stock / convert
            );
        }
    }

    /// Controlled f16 vs bf16 stock-qmm comparison at the dominant decode
    /// shapes, interleaved (ABAB) to cancel thermal/drift, median of 5 trials.
    /// Run: cargo test --release -p higgs-models q2_qmm_dtype_f16_vs_bf16 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q2_qmm_dtype_f16_vs_bf16() {
        let _exec = crate::mlx_exec::acquire();
        use std::time::Instant;

        let mut st = 0x4242_4242_u64;
        let lcg = |st: &mut u64| {
            *st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*st >> 32) as u32)
        };
        let f16_vec = |n: usize, st: &mut u64| -> Vec<half::f16> {
            (0..n)
                .map(|_| half::f16::from_f32((lcg(st) as f32 / u32::MAX as f32).mul_add(0.2, -0.1)))
                .collect()
        };

        // (rows, k, calls_per_token)
        let shapes: &[(usize, usize, usize)] = &[
            (34_816, 5_120, 64),  // gate_up-fused
            (16_384, 5_120, 48),  // qkvz-fused
            (17_408, 5_120, 0),   // gate/up reference (unfused shape)
            (5_120, 17_408, 64),  // down
            (248_320, 5_120, 1),  // lm_head
        ];

        for &(rows, k, calls) in shapes {
            let packed_cols = k / 16;
            let n_groups = k / GROUP_SIZE;
            let w = mlx_rs::Array::from_slice(
                &(0..rows * packed_cols).map(|_| lcg(&mut st)).collect::<Vec<u32>>(),
                &[rows as i32, packed_cols as i32],
            );
            let s = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let b = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let x_f16 = mlx_rs::Array::from_slice(&f16_vec(k, &mut st), &[1, k as i32]);
            let x_bf16 = x_f16
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap();

            let run = |x: &mlx_rs::Array, iters: usize| -> f64 {
                for _ in 0..5 {
                    let _ = crate::quant_mode::quantized_matmul(
                        x, &w, &s, Some(&b), true, 128, 2, crate::quant_mode::QuantMode::Affine,
                    )
                    .unwrap()
                    .eval()
                    .unwrap();
                }
                let t0 = Instant::now();
                for _ in 0..iters {
                    let _ = crate::quant_mode::quantized_matmul(
                        x, &w, &s, Some(&b), true, 128, 2, crate::quant_mode::QuantMode::Affine,
                    )
                    .unwrap()
                    .eval()
                    .unwrap();
                }
                t0.elapsed().as_micros() as f64 / iters as f64
            };

            // Interleaved ABAB, 5 trials each.
            let mut f16_ts = Vec::with_capacity(5);
            let mut bf16_ts = Vec::with_capacity(5);
            for _ in 0..5 {
                f16_ts.push(run(&x_f16, 20));
                bf16_ts.push(run(&x_bf16, 20));
            }
            f16_ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            bf16_ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let f16_med = f16_ts[2];
            let bf16_med = bf16_ts[2];
            println!(
                "DTYPE [{rows},K={k}] calls={calls}: f16={f16_med:.0}us bf16={bf16_med:.0}us ratio={:.2}x (bf16/f16) per-token={:.2}ms@f16 / {:.2}ms@bf16",
                bf16_med / f16_med,
                f16_med * calls as f64 / 1000.0,
                bf16_med * calls as f64 / 1000.0,
            );
        }
    }

    /// Ternary-select (no-bias) kernel: bit-gate against the CPU reference on
    /// ternary data (codes {0,1,2}, bias == -scale exactly).
    #[test]
    fn q2_qmv_ternary_matches_cpu_reference() {
        let _exec = crate::mlx_exec::acquire();

        let make_ternary = |out_f: usize, in_f: usize, seed: u64| {
            let packed_cols = in_f / WEIGHTS_PER_WORD;
            let n_groups = in_f / GROUP_SIZE;
            let mut st = seed;
            let w_packed: Vec<u32> = (0..out_f * packed_cols)
                .map(|_| {
                    let mut word = 0u32;
                    for sh in 0..16 {
                        let c = lcg(&mut st) % 3; // {0,1,2}
                        word |= c << (2 * sh);
                    }
                    word
                })
                .collect();
            let scales: Vec<f16> = (0..out_f * n_groups)
                .map(|i| f16::from_f32(0.05 + 0.013 * ((i % 7) as f32)))
                .collect();
            let biases: Vec<f16> = scales.iter().map(|&sv| f16::from_f32(-sv.to_f32())).collect();
            PackedQ2Linear { w_packed, scales, biases, out_features: out_f, in_features: in_f }
        };

        for &(out_f, in_f, seed) in &[
            (96usize, 256usize, 0x1234_5678_u64),
            (96usize, 1280usize, 0x5EED_5EED_u64),
            (130usize, 4096usize, 0x0BAD_F00D_u64),
            (5120usize, 5120usize, 0xCAFEBABE_u64),
        ] {
            let p = make_ternary(out_f, in_f, seed);
            let (w, s, _b) = upload_to_mlx(&p);

            let mut st = 0xABCD_EF01_u64;
            let x_f32: Vec<f32> = (0..in_f)
                .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[1, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();
            let x_ref: Vec<f32> = x_f32
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect();

            let y = crate::metal_kernel::bonsai_q2_qmv_ternary(&x, &w, &s, GROUP_SIZE as i32)
                .unwrap();
            y.eval().unwrap();
            let got = y.as_slice::<half::f16>();
            let want = dense_matvec_reference(&p, &x_ref);
            let mut max_rel = 0f32;
            for r in 0..out_f {
                let gv = got[r].to_f32();
                let wv = want[r];
                if wv != 0.0 {
                    max_rel = max_rel.max((gv - wv).abs() / wv.abs());
                }
            }
            assert!(
                max_rel < 1e-2,
                "ternary qmv relative error {max_rel} at {out_f}x{in_f}"
            );
        }
    }

    /// Ternary qmv at M=8 (speculative-verify tile): the kernel's grid.z batches
    /// over input rows, so it must match the CPU oracle across all 8 rows.
    #[test]
    fn q2_qmv_ternary_m8_matches_cpu_reference() {
        let _exec = crate::mlx_exec::acquire();

        let make_ternary = |out_f: usize, in_f: usize, seed: u64| {
            let packed_cols = in_f / WEIGHTS_PER_WORD;
            let n_groups = in_f / GROUP_SIZE;
            let mut st = seed;
            let w_packed: Vec<u32> = (0..out_f * packed_cols)
                .map(|_| {
                    let mut word = 0u32;
                    for sh in 0..16 {
                        let c = lcg(&mut st) % 3;
                        word |= c << (2 * sh);
                    }
                    word
                })
                .collect();
            let scales: Vec<f16> = (0..out_f * n_groups)
                .map(|i| f16::from_f32(0.05 + 0.013 * ((i % 7) as f32)))
                .collect();
            let biases: Vec<f16> = scales
                .iter()
                .map(|&sv| f16::from_f32(-sv.to_f32()))
                .collect();
            PackedQ2Linear {
                w_packed,
                scales,
                biases,
                out_features: out_f,
                in_features: in_f,
            }
        };

        for &(out_f, in_f, seed) in &[
            (96usize, 256usize, 0x1234_5678_u64),
            (96usize, 1280usize, 0x5EED_5EED_u64),
            (5120usize, 5120usize, 0xCAFEBABE_u64),
        ] {
            let p = make_ternary(out_f, in_f, seed);
            let (w, s, _b) = upload_to_mlx(&p);

            let mut st = 0xABCD_EF01_u64;
            let x_f32: Vec<f32> = (0..(8 * in_f))
                .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[8, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();
            let x_ref: Vec<f32> = x_f32
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect();

            let y =
                crate::metal_kernel::bonsai_q2_qmv_ternary(&x, &w, &s, GROUP_SIZE as i32).unwrap();
            y.eval().unwrap();
            let got = y.as_slice::<half::f16>();
            assert_eq!(got.len(), 8 * out_f);

            for m_idx in 0..8 {
                let x_slice = &x_ref[m_idx * in_f..(m_idx + 1) * in_f];
                let want = dense_matvec_reference(&p, x_slice);
                let mut max_rel = 0f32;
                for r in 0..out_f {
                    let gv = got[m_idx * out_f + r].to_f32();
                    let wv = want[r];
                    if wv != 0.0 {
                        max_rel = max_rel.max((gv - wv).abs() / wv.abs());
                    }
                }
                assert!(
                    max_rel < 1e-2,
                    "ternary qmv m8 relative error {max_rel} at {out_f}x{in_f} row {m_idx}"
                );
            }
        }
    }

    /// Ternary M<=8 MMA verify kernel (bits=2 g128): `simdgroup_matrix<8,8>`
    /// with the affine-offset trick must match the CPU oracle across all rows.
    #[test]
    fn q2_mma_m8_matches_cpu_reference() {
        let _exec = crate::mlx_exec::acquire();

        let make_ternary = |out_f: usize, in_f: usize, seed: u64| {
            let packed_cols = in_f / WEIGHTS_PER_WORD;
            let n_groups = in_f / GROUP_SIZE;
            let mut st = seed;
            let w_packed: Vec<u32> = (0..out_f * packed_cols)
                .map(|_| {
                    let mut word = 0u32;
                    for sh in 0..16 {
                        let c = lcg(&mut st) % 3;
                        word |= c << (2 * sh);
                    }
                    word
                })
                .collect();
            let scales: Vec<f16> = (0..out_f * n_groups)
                .map(|i| f16::from_f32(0.05 + 0.013 * ((i % 7) as f32)))
                .collect();
            let biases: Vec<f16> = scales
                .iter()
                .map(|&sv| f16::from_f32(-sv.to_f32()))
                .collect();
            PackedQ2Linear {
                w_packed,
                scales,
                biases,
                out_features: out_f,
                in_features: in_f,
            }
        };

        for &(out_f, in_f, seed) in &[
            (512usize, 512usize, 0x1111_2222_u64),
            (5120usize, 5120usize, 0x5555_6666_u64),
            (5120usize, 17408usize, 0x7777_8888_u64),
        ] {
            let p = make_ternary(out_f, in_f, seed);
            let (w, s, b) = upload_to_mlx(&p);

            for m_rows in [4i32, 5, 8] {
            let mut st = 0xABCD_EF01_u64;
            let x_f32: Vec<f32> = (0..(m_rows as usize * in_f))
                .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[m_rows, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap();
            let x_ref: Vec<f32> = x_f32
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_f32())
                .collect();

            let y = crate::metal_kernel::bonsai_q2_mma_m8(&x, &w, &s, &b).unwrap();
            y.eval().unwrap();
            let got = y.as_slice::<half::bf16>();
            assert_eq!(got.len(), m_rows as usize * out_f);

            let mut w_row = vec![0f32; in_f];
            for m_idx in 0..m_rows as usize {
                let x_slice = &x_ref[m_idx * in_f..(m_idx + 1) * in_f];
                let want = dense_matvec_reference(&p, x_slice);
                let mut max_rel = 0f32;
                for r in 0..out_f {
                    let gv = got[m_idx * out_f + r].to_f32();
                    let wv = want[r];
                    if wv.abs() > 1e-3 {
                        max_rel = max_rel.max((gv - wv).abs() / wv.abs());
                    }
                }
                assert!(
                    max_rel < 2e-2,
                    "mma_m8 ({out_f}x{in_f}) M={m_rows} row {m_idx}: max_rel {max_rel}"
                );
            }
            }
        }
    }

    /// Go/no-go: does dropping the bias read (ternary no-bias kernel) beat
    /// stock MLX at the dominant M=1 shapes? This falsifies (or confirms) the
    /// "decode is DRAM-bound, fewer bytes = faster" thesis before the trit
    /// repack. Run: cargo test --release -p higgs-models q2_qmv_ternary_go_no_go -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q2_qmv_ternary_go_no_go() {
        let _exec = crate::mlx_exec::acquire();
        use std::time::Instant;

        let mut st = 0x7777_7777_u64;
        let lcg2 = |st: &mut u64| {
            *st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*st >> 32) as u32)
        };
        let f16_vec = |n: usize, st: &mut u64| -> Vec<half::f16> {
            (0..n)
                .map(|_| half::f16::from_f32((lcg2(st) as f32 / u32::MAX as f32).mul_add(0.2, -0.1)))
                .collect()
        };
        let time = |f: &mut dyn FnMut() -> mlx_rs::Array| {
            let _ = f().eval().unwrap();
            let t0 = Instant::now();
            for _ in 0..30 {
                let _ = f().eval().unwrap();
            }
            t0.elapsed().as_micros() as f64 / 30.0
        };

        for (rows, k, label) in [
            (34_816usize, 5_120usize, "gate_up-fused"),
            (16_384, 5_120, "qkvz-fused"),
            (17_408, 5_120, "gate/up"),
            (5_120, 17_408, "down"),
        ] {
            let packed_cols = k / 16;
            let n_groups = k / GROUP_SIZE;
            let mut w: Vec<u32> = Vec::with_capacity(rows * packed_cols);
            for _ in 0..rows * packed_cols {
                let mut word = 0u32;
                for sh in 0..16 {
                    let c = lcg2(&mut st) % 3;
                    word |= c << (2 * sh);
                }
                w.push(word);
            }
            let w = mlx_rs::Array::from_slice(&w, &[rows as i32, packed_cols as i32]);
            let s = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let b = mlx_rs::Array::from_slice(
                &s.as_slice::<half::f16>().iter().map(|&v| half::f16::from_f32(-v.to_f32())).collect::<Vec<_>>(),
                &[rows as i32, n_groups as i32],
            );
            let x = mlx_rs::Array::from_slice(&f16_vec(k, &mut st), &[1, k as i32]);

            let stock = time(&mut || {
                crate::quant_mode::quantized_matmul(
                    &x, &w, &s, Some(&b), true, 128, 2, crate::quant_mode::QuantMode::Affine,
                )
                .unwrap()
            });
            let ternary = time(&mut || {
                crate::metal_kernel::bonsai_q2_qmv_ternary(&x, &w, &s, GROUP_SIZE as i32).unwrap()
            });
            println!(
                "TERNARY-PROBE {label} [{rows},K={k}] M=1: stock={stock:.0}us ternary={ternary:.0}us ternary-vs-stock={:.2}x",
                stock / ternary
            );
        }

        // bf16 activation confirmation (server reality) on the two dominant shapes.
        for (rows, k, label) in [(34_816usize, 5_120usize, "gate_up-fused"), (16_384, 5_120, "qkvz-fused")] {
            let packed_cols = k / 16;
            let n_groups = k / GROUP_SIZE;
            let mut w: Vec<u32> = Vec::with_capacity(rows * packed_cols);
            for _ in 0..rows * packed_cols {
                let mut word = 0u32;
                for sh in 0..16 {
                    let c = lcg2(&mut st) % 3;
                    word |= c << (2 * sh);
                }
                w.push(word);
            }
            let w = mlx_rs::Array::from_slice(&w, &[rows as i32, packed_cols as i32]);
            let s = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let b = mlx_rs::Array::from_slice(
                &s.as_slice::<half::f16>().iter().map(|&v| half::f16::from_f32(-v.to_f32())).collect::<Vec<_>>(),
                &[rows as i32, n_groups as i32],
            );
            let x = mlx_rs::Array::from_slice(&f16_vec(k, &mut st), &[1, k as i32])
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap();

            let stock = time(&mut || {
                crate::quant_mode::quantized_matmul(
                    &x, &w, &s, Some(&b), true, 128, 2, crate::quant_mode::QuantMode::Affine,
                )
                .unwrap()
            });
            let ternary = time(&mut || {
                crate::metal_kernel::bonsai_q2_qmv_ternary(&x, &w, &s, GROUP_SIZE as i32).unwrap()
            });
            println!(
                "TERNARY-PROBE-BF16 {label} [{rows},K={k}] M=1: stock={stock:.0}us ternary={ternary:.0}us ternary-vs-stock={:.2}x",
                stock / ternary
            );
        }
    }

    /// Pack flat 2-bit ternary codes ({0,1,2}) into 26-byte/128-trit groups.
    /// `codes` is the full row-major u32 buffer (out_features * in_features/16
    /// words); trits are packed consecutively across the whole buffer.
    fn pack_trits_26(codes: &[u32]) -> Vec<u8> {
        let total_trits = codes.len() * 16;
        let n_groups = total_trits / 128;
        let mut out = vec![0u8; n_groups * 26];
        let get = |i: usize| -> u32 { (codes[i / 16] >> (2 * (i % 16))) & 3 };
        for g in 0..n_groups {
            let base = g * 26;
            for j in 0..25 {
                let mut v = 0u32;
                for t in 0..5 {
                    v += get(g * 128 + j * 5 + t) * 3u32.pow(t as u32);
                }
                out[base + j] = v as u8;
            }
            let mut v = 0u32;
            for t in 0..3 {
                v += get(g * 128 + 125 + t) * 3u32.pow(t as u32);
            }
            out[base + 25] = v as u8;
        }
        out
    }

    /// Plan A baseline: stock MLX batched (M>1) prefill bandwidth vs the
    /// 120 GB/s target, at the dominant shapes. Establishes headroom before
    /// building a batched ternary kernel.
    /// Run: cargo test --release -p higgs-models q2_prefill_bw_baseline -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q2_prefill_bw_baseline() {
        let _exec = crate::mlx_exec::acquire();
        use std::time::Instant;

        let mut st = 0x9999_9999_u64;
        let lcg = |st: &mut u64| { *st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407); ((*st >> 32) as u32) };
        let f16_vec = |n: usize, st: &mut u64| -> Vec<half::f16> {
            (0..n).map(|_| half::f16::from_f32((lcg(st) as f32 / u32::MAX as f32).mul_add(0.2, -0.1))).collect()
        };

        for (rows, k, label) in [(34_816usize, 5_120usize, "gate_up-fused"), (16_384, 5_120, "qkvz-fused")] {
            let packed_cols = k / 16;
            let n_groups = k / GROUP_SIZE;
            let mut w: Vec<u32> = Vec::with_capacity(rows * packed_cols);
            for _ in 0..rows * packed_cols {
                let mut word = 0u32;
                for sh in 0..16 { let c = lcg(&mut st) % 3; word |= c << (2 * sh); }
                w.push(word);
            }
            let w = mlx_rs::Array::from_slice(&w, &[rows as i32, packed_cols as i32]);
            let s = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let b = mlx_rs::Array::from_slice(&s.as_slice::<half::f16>().iter().map(|&v| half::f16::from_f32(-v.to_f32())).collect::<Vec<_>>(), &[rows as i32, n_groups as i32]);
            // weight bytes read per call (packed 2-bit + scale, no bias for ternary)
            let weight_bytes = (rows * packed_cols * 4 + rows * n_groups * 2) as f64;

            for m in [2i32, 4, 8, 16, 32, 64] {
                let x = mlx_rs::Array::from_slice(&f16_vec((k * m as usize) as usize, &mut st), &[m, k as i32]);
                let time = |iters: usize| -> f64 {
                    for _ in 0..3 {
                        let _ = crate::quant_mode::quantized_matmul(&x, &w, &s, Some(&b), true, 128, 2, crate::quant_mode::QuantMode::Affine).unwrap().eval().unwrap();
                    }
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = crate::quant_mode::quantized_matmul(&x, &w, &s, Some(&b), true, 128, 2, crate::quant_mode::QuantMode::Affine).unwrap().eval().unwrap();
                    }
                    t0.elapsed().as_micros() as f64 / iters as f64
                };
                let us = time(10);
                let gb_s = weight_bytes / (us * 1e-6) / 1e9;
                println!("PREFILL-BW {label} M={m}: {us:.0}us -> {gb_s:.0} GB/s");
            }
        }
    }

    /// Batched ternary qmm oracle: M>1 vs the CPU 2-bit reference.
    #[test]
    fn q2_qmm_ternary_matches_cpu_reference() {
        let _exec = crate::mlx_exec::acquire();

        for &(out_f, in_f, m, seed) in &[
            (96usize, 256usize, 1usize, 0x1111_1111_u64),
            (96usize, 256usize, 3usize, 0x2222_2222_u64),
            (130usize, 4096usize, 5usize, 0x3333_3333_u64),
            (96usize, 1280usize, 8usize, 0x4444_4444_u64),
        ] {
            let packed_cols = in_f / WEIGHTS_PER_WORD;
            let n_groups = in_f / GROUP_SIZE;
            let mut st = seed;
            let codes: Vec<u32> = (0..out_f * packed_cols)
                .map(|_| {
                    let mut word = 0u32;
                    for sh in 0..16 { let c = lcg(&mut st) % 3; word |= c << (2 * sh); }
                    word
                })
                .collect();
            let scales: Vec<f16> = (0..out_f * n_groups).map(|i| f16::from_f32(0.05 + 0.013 * ((i % 7) as f32))).collect();
            let p = PackedQ2Linear {
                w_packed: codes,
                scales: scales.clone(),
                biases: scales.iter().map(|&sv| f16::from_f32(-sv.to_f32())).collect(),
                out_features: out_f,
                in_features: in_f,
            };
            let (w, s, _b) = upload_to_mlx(&p);

            let mut st = 0xABCD_EF01_u64;
            let x_f32: Vec<f32> = (0..m * in_f).map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0)).collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[m as i32, in_f as i32]).as_dtype(mlx_rs::Dtype::Float16).unwrap();

            let y = crate::metal_kernel::bonsai_q2_qmm_ternary(&x, &w, &s, GROUP_SIZE as i32).unwrap();
            y.eval().unwrap();
            let got = y.as_slice::<half::f16>();

            let mut max_rel = 0f32;
            for mm in 0..m {
                let x_ref: Vec<f32> = x_f32[mm * in_f..(mm + 1) * in_f].iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
                let want = dense_matvec_reference(&p, &x_ref);
                for r in 0..out_f {
                    let gv = got[mm * out_f + r].to_f32();
                    let wv = want[r];
                    if wv != 0.0 { max_rel = max_rel.max((gv - wv).abs() / wv.abs()); }
                }
            }
            assert!(max_rel < 1e-2, "batched ternary qmm rel err {max_rel} at {out_f}x{in_f} M={m}");
        }
    }

    /// Batched ternary qmm vs stock MLX at prefill batch sizes.
    /// Run: cargo test --release -p higgs-models q2_qmm_ternary_go_no_go -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q2_qmm_ternary_go_no_go() {
        let _exec = crate::mlx_exec::acquire();
        use std::time::Instant;

        let mut st = 0x7777_7777_u64;
        let lcg2 = |st: &mut u64| { *st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407); ((*st >> 32) as u32) };
        let f16_vec = |n: usize, st: &mut u64| -> Vec<half::f16> { (0..n).map(|_| half::f16::from_f32((lcg2(st) as f32 / u32::MAX as f32).mul_add(0.2, -0.1))).collect() };
        let time = |f: &mut dyn FnMut() -> mlx_rs::Array| { let _ = f().eval().unwrap(); let t0 = Instant::now(); for _ in 0..10 { let _ = f().eval().unwrap(); } t0.elapsed().as_micros() as f64 / 10.0 };

        for (rows, k, label) in [(34_816usize, 5_120usize, "gate_up-fused"), (16_384, 5_120, "qkvz-fused")] {
            let packed_cols = k / 16; let n_groups = k / GROUP_SIZE;
            let mut codes: Vec<u32> = Vec::with_capacity(rows * packed_cols);
            for _ in 0..rows * packed_cols { let mut word = 0u32; for sh in 0..16 { let c = lcg2(&mut st) % 3; word |= c << (2 * sh); } codes.push(word); }
            let w = mlx_rs::Array::from_slice(&codes, &[rows as i32, packed_cols as i32]);
            let s = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let b = mlx_rs::Array::from_slice(&s.as_slice::<half::f16>().iter().map(|&v| half::f16::from_f32(-v.to_f32())).collect::<Vec<_>>(), &[rows as i32, n_groups as i32]);

            for m in [8i32, 16, 32, 64] {
                let x = mlx_rs::Array::from_slice(&f16_vec((k * m as usize) as usize, &mut st), &[m, k as i32]);
                let stock = time(&mut || crate::quant_mode::quantized_matmul(&x, &w, &s, Some(&b), true, 128, 2, crate::quant_mode::QuantMode::Affine).unwrap());
                let ternary = time(&mut || crate::metal_kernel::bonsai_q2_qmm_ternary(&x, &w, &s, GROUP_SIZE as i32).unwrap());
                println!("QMM-PROBE {label} M={m}: stock={stock:.0}us ternary={ternary:.0}us speedup={:.2}x", stock / ternary);
            }
        }
    }

    #[test]
    fn q2_qmm_tiled_ternary_matches_cpu_reference() {
        let _exec = crate::mlx_exec::acquire();

        for &(n, k, m, dtype, seed) in &[
            (96usize, 256usize, 9usize, mlx_rs::Dtype::Float16, 0x1234_5678_u64),
            (130, 4096, 17, mlx_rs::Dtype::Bfloat16, 0x5EED_5EED),
            (5120, 5120, 31, mlx_rs::Dtype::Bfloat16, 0x0BAD_F00D),
            (96, 17408, 33, mlx_rs::Dtype::Float16, 0xCAFEBABE),
        ] {
            let p = make_ternary_q2(n, k, seed);
            let (w, s, b) = upload_to_mlx(&p);
            let mut st = 0xABCD_EF01_u64;
            let x_f32: Vec<f32> = (0..m * k)
                .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[m as i32, k as i32])
                .as_dtype(dtype)
                .unwrap();
            let x_ref: Vec<f32> = x_f32
                .iter()
                .map(|&value| match dtype {
                    mlx_rs::Dtype::Float16 => half::f16::from_f32(value).to_f32(),
                    mlx_rs::Dtype::Bfloat16 => half::bf16::from_f32(value).to_f32(),
                    _ => unreachable!("oracle cases use fp16 or bf16"),
                })
                .collect();
            let mut reference = Vec::with_capacity(m * n);
            for row in 0..m {
                reference.extend(dense_matvec_reference(&p, &x_ref[row * k..(row + 1) * k]));
            }

            let s_matched = s.as_dtype(dtype).unwrap();
            let b_matched = b.as_dtype(dtype).unwrap();
            let native_stock = crate::quant_mode::quantized_matmul(
                &x,
                &w,
                &s,
                Some(&b),
                true,
                GROUP_SIZE as i32,
                2,
                crate::quant_mode::QuantMode::Affine,
            )
            .unwrap();
            let cast_matched_stock = crate::quant_mode::quantized_matmul(
                &x,
                &w,
                &s_matched,
                Some(&b_matched),
                true,
                GROUP_SIZE as i32,
                2,
                crate::quant_mode::QuantMode::Affine,
            )
            .unwrap();
            let candidate = crate::metal_kernel::bonsai_q2_qmm_tiled_ternary(
                &x,
                &w,
                &s,
                GROUP_SIZE as i32,
            )
            .unwrap();

            native_stock.eval().unwrap();
            cast_matched_stock.eval().unwrap();
            candidate.eval().unwrap();
            assert_eq!(candidate.shape(), &[m as i32, n as i32]);
            assert_eq!(candidate.dtype(), dtype);
            let native_metrics = qmm_error(&native_stock, &reference);
            let cast_metrics = qmm_error(&cast_matched_stock, &reference);
            let candidate_metrics = qmm_error(&candidate, &reference);
            println!(
                "TILED-QMM-ORACLE N={n} K={k} M={m} dtype={dtype:?}: native_stock={native_metrics:?} cast_matched_stock={cast_metrics:?} candidate={candidate_metrics:?}"
            );
            assert!(
                candidate_metrics.1 <= 1.0e-2,
                "tiled qmm normalized RMS error {} at N={n} K={k} M={m}",
                candidate_metrics.1
            );
            assert!(
                candidate_metrics.0 <= 0.25,
                "tiled qmm maximum absolute error {} at N={n} K={k} M={m}",
                candidate_metrics.0
            );
        }
    }

    #[test]
    fn q2_qmm_tiled_ternary_preserves_rank_and_checks_contracts() {
        let _exec = crate::mlx_exec::acquire();
        let n = 96usize;
        let k = 256usize;
        let p = make_ternary_q2(n, k, 0xD00D_F00D);
        let (w, s, _) = upload_to_mlx(&p);
        let values = vec![0.25_f32; 9 * k];
        let x = mlx_rs::Array::from_slice(&values, &[1, 9, k as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();
        let rank3 = crate::metal_kernel::bonsai_q2_qmm_tiled_ternary(
            &x,
            &w,
            &s,
            GROUP_SIZE as i32,
        )
        .unwrap();
        assert_eq!(rank3.shape(), &[1, 9, n as i32]);

        let expect_contract_error = |result: Result<mlx_rs::Array, mlx_rs::error::Exception>,
                                     expected_phrase: &str| {
            let error = match result {
                Ok(_) => panic!("expected {expected_phrase} validation error"),
                Err(error) => error,
            };
            let message = error.to_string();
            assert!(
                message.contains(expected_phrase),
                "expected contract phrase {expected_phrase:?} in error message, got: {message}"
            );
        };

        expect_contract_error(
            crate::metal_kernel::bonsai_q2_qmm_tiled_ternary(&x, &w, &s, 64),
            "group",
        );
        let short_k_values = vec![0.25_f32; 9 * (k - 16)];
        let short_k = mlx_rs::Array::from_slice(&short_k_values, &[9, (k - 16) as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();
        expect_contract_error(
            crate::metal_kernel::bonsai_q2_qmm_tiled_ternary(
                &short_k,
                &w,
                &s,
                GROUP_SIZE as i32,
            ),
            "input K mismatch",
        );
        let short_scales = mlx_rs::Array::from_slice(&p.scales[..n], &[n as i32, 1]);
        let flat_x = mlx_rs::Array::from_slice(&values, &[9, k as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();
        expect_contract_error(
            crate::metal_kernel::bonsai_q2_qmm_tiled_ternary(
                &flat_x,
                &w,
                &short_scales,
                GROUP_SIZE as i32,
            ),
            "scale",
        );
        let short_values = vec![0.25_f32; 8 * k];
        let short_m = mlx_rs::Array::from_slice(&short_values, &[8, k as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();
        expect_contract_error(
            crate::metal_kernel::bonsai_q2_qmm_tiled_ternary(
                &short_m,
                &w,
                &s,
                GROUP_SIZE as i32,
            ),
            "M must be > 8",
        );
    }

    #[allow(unsafe_code)]
    fn time_qmm_with_peak_memory(f: &mut dyn FnMut() -> mlx_rs::Array) -> (f64, usize) {
        use std::time::Instant;

        let mut before = 0_usize;
        unsafe {
            assert_eq!(mlx_sys::mlx_get_active_memory(&mut before), 0);
            assert_eq!(mlx_sys::mlx_reset_peak_memory(), 0);
        }
        let started = Instant::now();
        let output = f();
        output.eval().unwrap();
        let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        drop(output);
        let mut peak = 0_usize;
        unsafe {
            assert_eq!(mlx_sys::mlx_get_peak_memory(&mut peak), 0);
        }
        (elapsed_us, peak.saturating_sub(before))
    }

    fn median_us(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }

    /// Compare tiled ternary QMM against stock MLX; scalar QMM is informational.
    /// Run: cargo test --release -p higgs-models q2_qmm_tiled_ternary_go_no_go -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q2_qmm_tiled_ternary_go_no_go() {
        let _exec = crate::mlx_exec::acquire();
        let mut st = 0x7654_3210_u64;

        for &(n, k, label) in &[
            (34_816usize, 5_120usize, "gate_up"),
            (5_120, 17_408, "down"),
        ] {
            let p = make_ternary_q2(n, k, st);
            st = st.wrapping_add(1);
            let (w, s, b) = upload_to_mlx(&p);
            for m in [16usize, 32, 64, 512, 1024] {
                let x_values: Vec<f16> = (0..m * k)
                    .map(|_| f16::from_f32((lcg(&mut st) as f32 / u32::MAX as f32).mul_add(0.2, -0.1)))
                    .collect();
                let x = mlx_rs::Array::from_slice(&x_values, &[m as i32, k as i32]);
                let mut stock = || {
                    crate::quant_mode::quantized_matmul(
                        &x,
                        &w,
                        &s,
                        Some(&b),
                        true,
                        GROUP_SIZE as i32,
                        2,
                        crate::quant_mode::QuantMode::Affine,
                    )
                    .unwrap()
                };
                let mut scalar = || {
                    crate::metal_kernel::bonsai_q2_qmm_ternary(
                        &x,
                        &w,
                        &s,
                        GROUP_SIZE as i32,
                    )
                    .unwrap()
                };
                let mut candidate = || {
                    crate::metal_kernel::bonsai_q2_qmm_tiled_ternary(
                        &x,
                        &w,
                        &s,
                        GROUP_SIZE as i32,
                    )
                    .unwrap()
                };

                stock().eval().unwrap();
                scalar().eval().unwrap();
                candidate().eval().unwrap();
                let mut times = [Vec::with_capacity(7), Vec::with_capacity(7), Vec::with_capacity(7)];
                let mut peaks = [0_usize; 3];
                for round in 0..7 {
                    let order = match round % 3 {
                        0 => [0, 1, 2],
                        1 => [1, 2, 0],
                        _ => [2, 0, 1],
                    };
                    for path in order {
                        let (elapsed, peak) = match path {
                            0 => time_qmm_with_peak_memory(&mut stock),
                            1 => time_qmm_with_peak_memory(&mut scalar),
                            _ => time_qmm_with_peak_memory(&mut candidate),
                        };
                        times[path].push(elapsed);
                        peaks[path] = peaks[path].max(peak);
                    }
                }
                let medians = [
                    median_us(&mut times[0]),
                    median_us(&mut times[1]),
                    median_us(&mut times[2]),
                ];
                println!(
                    "TILED-QMM-BENCH {label} N={n} K={k} M={m}: stock={:.0}us/{}B scalar={:.0}us/{}B candidate={:.0}us/{}B candidate-vs-stock={:.2}x",
                    medians[0], peaks[0], medians[1], peaks[1], medians[2], peaks[2], medians[0] / medians[2]
                );
                assert!(
                    medians[2] < medians[0],
                    "tiled QMM candidate did not beat stock MLX for {label} N={n} K={k} M={m}: candidate={}us stock={}us",
                    medians[2], medians[0]
                );
            }
        }
    }

    /// Trit kernel oracle: bit-gate vs the CPU 2-bit reference on ternary data.
    #[test]
    fn q2_qmv_trit_matches_cpu_reference() {
        let _exec = crate::mlx_exec::acquire();

        for &(out_f, in_f, seed) in &[
            (96usize, 256usize, 0x1234_5678_u64),
            (96usize, 1280usize, 0x5EED_5EED_u64),
            (130usize, 4096usize, 0x0BAD_F00D_u64),
            (5120usize, 5120usize, 0xCAFEBABE_u64),
        ] {
            let packed_cols = in_f / WEIGHTS_PER_WORD;
            let n_groups = in_f / GROUP_SIZE;
            let mut st = seed;
            let codes: Vec<u32> = (0..out_f * packed_cols)
                .map(|_| {
                    let mut word = 0u32;
                    for sh in 0..16 {
                        let c = lcg(&mut st) % 3;
                        word |= c << (2 * sh);
                    }
                    word
                })
                .collect();
            let p = PackedQ2Linear {
                w_packed: codes.clone(),
                scales: (0..out_f * n_groups)
                    .map(|i| f16::from_f32(0.05 + 0.013 * ((i % 7) as f32)))
                    .collect(),
                biases: (0..out_f * n_groups).map(|_| f16::from_f32(0.0)).collect(),
                out_features: out_f,
                in_features: in_f,
            };
            let (w2, s, _b) = upload_to_mlx(&p);

            let trit = pack_trits_26(&codes);
            let wt = mlx_rs::Array::from_slice(&trit, &[out_f as i32, (in_f / 128 * 26) as i32]);

            let mut st = 0xABCD_EF01_u64;
            let x_f32: Vec<f32> = (0..in_f)
                .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[1, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();
            let x_ref: Vec<f32> = x_f32
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect();

            // reference via the 2-bit path (bias = -scale): build biases properly
            let p_ref = PackedQ2Linear {
                w_packed: codes,
                scales: p.scales.clone(),
                biases: p.scales.iter().map(|&sv| f16::from_f32(-sv.to_f32())).collect(),
                out_features: out_f,
                in_features: in_f,
            };
            let _ = w2;
            let y = crate::metal_kernel::bonsai_q2_qmv_trit(&x, &wt, &s, GROUP_SIZE as i32).unwrap();
            y.eval().unwrap();
            let got = y.as_slice::<half::f16>();
            let want = dense_matvec_reference(&p_ref, &x_ref);
            let mut max_rel = 0f32;
            for r in 0..out_f {
                let gv = got[r].to_f32();
                let wv = want[r];
                if wv != 0.0 {
                    max_rel = max_rel.max((gv - wv).abs() / wv.abs());
                }
            }
            assert!(
                max_rel < 1e-2,
                "trit qmv relative error {max_rel} at {out_f}x{in_f}"
            );
        }
    }

    /// Go/no-go: trit kernel (1.6 bits/weight) vs the 2-bit ternary kernel at
    /// the dominant M=1 shapes. Run: cargo test --release -p higgs-models q2_qmv_trit_go_no_go -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q2_qmv_trit_go_no_go() {
        let _exec = crate::mlx_exec::acquire();
        use std::time::Instant;

        let mut st = 0x7777_7777_u64;
        let lcg2 = |st: &mut u64| {
            *st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*st >> 32) as u32)
        };
        let f16_vec = |n: usize, st: &mut u64| -> Vec<half::f16> {
            (0..n)
                .map(|_| half::f16::from_f32((lcg2(st) as f32 / u32::MAX as f32).mul_add(0.2, -0.1)))
                .collect()
        };
        let time = |f: &mut dyn FnMut() -> mlx_rs::Array| {
            let _ = f().eval().unwrap();
            let t0 = Instant::now();
            for _ in 0..30 {
                let _ = f().eval().unwrap();
            }
            t0.elapsed().as_micros() as f64 / 30.0
        };

        for (rows, k, label) in [
            (34_816usize, 5_120usize, "gate_up-fused"),
            (16_384, 5_120, "qkvz-fused"),
            (5_120, 17_408, "down"),
        ] {
            let packed_cols = k / 16;
            let n_groups = k / GROUP_SIZE;
            let mut codes: Vec<u32> = Vec::with_capacity(rows * packed_cols);
            for _ in 0..rows * packed_cols {
                let mut word = 0u32;
                for sh in 0..16 {
                    let c = lcg2(&mut st) % 3;
                    word |= c << (2 * sh);
                }
                codes.push(word);
            }
            let w2 = mlx_rs::Array::from_slice(&codes, &[rows as i32, packed_cols as i32]);
            let s = mlx_rs::Array::from_slice(&f16_vec(rows * n_groups, &mut st), &[rows as i32, n_groups as i32]);
            let trit = pack_trits_26(&codes);
            let wt = mlx_rs::Array::from_slice(&trit, &[rows as i32, (k / 128 * 26) as i32]);
            let x = mlx_rs::Array::from_slice(&f16_vec(k, &mut st), &[1, k as i32]);

            let ternary = time(&mut || {
                crate::metal_kernel::bonsai_q2_qmv_ternary(&x, &w2, &s, GROUP_SIZE as i32).unwrap()
            });
            let trit = time(&mut || {
                crate::metal_kernel::bonsai_q2_qmv_trit(&x, &wt, &s, GROUP_SIZE as i32).unwrap()
            });
            println!(
                "TRIT-PROBE {label} [{rows},K={k}] M=1: ternary={ternary:.0}us trit={trit:.0}us trit-vs-ternary={:.2}x",
                ternary / trit
            );
        }
    }

    /// Staged-x variant must satisfy the same oracle gate as the base kernel.
    #[test]
    fn q2_qmv_staged_kernel_matches_cpu_reference() {
        let _exec = crate::mlx_exec::acquire();

        for &(out_f, in_f, seed) in &[
            (96usize, 256usize, 0x1234_5678_u64),
            (96usize, 1280usize, 0x5EED_5EED_u64), // main block + tail in one launch
            (130usize, 4096usize, 0x0BAD_F00D_u64),
            (5120usize, 5120usize, 0xCAFEBABE_u64),
        ] {
            let p = make_packed_q2(out_f, in_f, seed);
            let (w, s, b) = upload_to_mlx(&p);

            let mut st = 0xABCD_EF01_u64;
            let x_f32: Vec<f32> = (0..in_f)
                .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[1, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();
            let x_ref: Vec<f32> = x_f32
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect();

            let y =
                crate::metal_kernel::bonsai_q2_qmv_staged(&x, &w, &s, &b, GROUP_SIZE as i32)
                    .unwrap();
            y.eval().unwrap();
            let got = y.as_slice::<half::f16>();

            let want = dense_matvec_reference(&p, &x_ref);
            let mut max_rel = 0f32;
            for r in 0..out_f {
                let gv = got[r].to_f32();
                let wv = want[r];
                if wv != 0.0 {
                    max_rel = max_rel.max((gv - wv).abs() / wv.abs());
                }
            }
            assert!(
                max_rel < 1e-2,
                "staged qmv relative error {max_rel} at {out_f}x{in_f}"
            );
        }
    }

    #[test]
    fn q2_qmv_kernel_matches_cpu_reference() {
        let _exec = crate::mlx_exec::acquire();

        for &(out_f, in_f, seed) in &[
            (96usize, 256usize, 0x1234_5678_u64), // small shape, multiple groups
            (130usize, 4096usize, 0x0BAD_F00D_u64), // 32 packed words per row
            (5120usize, 5120usize, 0xCAFEBABE_u64), // Bonsai-27B hidden x hidden
        ] {
            let p = make_packed_q2(out_f, in_f, seed);
            let (w, s, b) = upload_to_mlx(&p);

            // Build a deterministic fp16 activation vector.
            let mut st = 0xABCD_EF01_u64;
            let x_f32: Vec<f32> = (0..in_f)
                .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[1, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();
            // Reference uses fp16-rounded activations to match kernel dtype exactly.
            let x_ref: Vec<f32> = x_f32
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect();

            let y = crate::metal_kernel::bonsai_q2_qmv(&x, &w, &s, &b, GROUP_SIZE as i32).unwrap();
            y.eval().unwrap();
            let got = y.as_slice::<half::f16>();
            assert_eq!(
                got.len(),
                out_f,
                "output length mismatch for {out_f}x{in_f}"
            );

            let want = dense_matvec_reference(&p, &x_ref);
            let mut max_rel = 0f32;
            for r in 0..out_f {
                let gv = got[r].to_f32();
                let wv = want[r];
                let tol = (1e-2 * wv.abs()).max(1e-3);
                let d = (gv - wv).abs();
                assert!(
                    d <= tol,
                    "qmv mismatch ({out_f}x{in_f}) row {r}: got {gv} want {wv} (|d|={d}, tol={tol})"
                );
                if d > 0.0 {
                    max_rel = max_rel.max(d / tol);
                }
            }
            // Sanity: we should not be sitting right at the tolerance edge.
            assert!(
                max_rel < 0.9,
                "max_rel={max_rel} for shape ({out_f}x{in_f}) — kernel within tolerance but suspiciously close"
            );
        }
    }

    /// Phase 3B M>1 gate: z-batched QMV must match CPU oracle for verifier
    /// shapes M=1..=5 (anchor + drafts). This is what `bonsai_q2_qmm` routes
    /// to and what the block verifier will exercise at M=5.
    #[test]
    fn q2_qmm_kernel_matches_cpu_reference_m1_through_m5() {
        let _exec = crate::mlx_exec::acquire();

        let (out_f, in_f) = (512usize, 512usize);
        let p = make_packed_q2(out_f, in_f, 0xFEED_FACE);
        let (w, s, b) = upload_to_mlx(&p);
        let mut w_row = vec![0f32; in_f];

        for m in 1..=5 {
            // Distinct deterministic activation per M row.
            let mut st = 0xDEAD_BEEF_u64.wrapping_mul(m as u64);
            let x_f32: Vec<f32> = (0..(m as usize * in_f))
                .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[m, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();
            let x_ref: Vec<f32> = x_f32
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect();

            let y = crate::metal_kernel::bonsai_q2_qmm(&x, &w, &s, &b, GROUP_SIZE as i32).unwrap();
            y.eval().unwrap();
            let got = y.as_slice::<half::f16>();
            assert_eq!(got.len(), m as usize * out_f);

            for m_idx in 0..m as usize {
                let x_slice = &x_ref[m_idx * in_f..(m_idx + 1) * in_f];
                for r in 0..out_f {
                    p.dequant_row_to_fp32(r, &mut w_row);
                    let mut acc = 0f32;
                    for c in 0..in_f {
                        acc += x_slice[c] * w_row[c];
                    }
                    let gv = got[m_idx * out_f + r].to_f32();
                    let tol = (1e-2 * acc.abs()).max(1e-3);
                    assert!(
                        (gv - acc).abs() <= tol,
                        "qmm M={m} row {r} (m_idx={m_idx}): got {gv} want {acc}"
                    );
                }
            }
        }
    }

    /// Phase 3D M=5 kernel gate: bonsai_q2_row2_m5_contract must match the
    /// CPU oracle across all 5 verifier rows for representative Bonsai-27B
    /// shapes. This is the make-or-break gate for the 1.45x target.
    #[test]
    fn q2_row2_m5_kernel_matches_cpu_reference() {
        let _exec = crate::mlx_exec::acquire();

        for &(out_f, in_f, seed) in &[
            (256usize, 256usize, 0x1111_2222_u64),   // small smoke
            (512usize, 512usize, 0x3333_4444_u64),   // 4 groups
            (5120usize, 5120usize, 0x5555_6666_u64), // Bonsai-27B hidden x hidden
        ] {
            let p = make_packed_q2(out_f, in_f, seed);
            let (w_canon, s_canon, b_canon) = upload_to_mlx(&p);
            let packed =
                crate::metal_kernel::BonsaiQ2Row2::from_row_major(&w_canon, &s_canon).unwrap();

            // Round-trip first -- if this fails, the layout transform itself is broken.
            let (w_rt, s_rt) = packed.to_row_major().unwrap();
            w_rt.eval().unwrap();
            s_rt.eval().unwrap();
            let w_got: Vec<u32> = w_rt.as_slice::<u32>().iter().copied().collect();
            let s_got: Vec<half::f16> = s_rt.as_slice::<half::f16>().iter().copied().collect();
            assert_eq!(
                w_got, p.w_packed,
                "row2 weights did not round-trip ({out_f}x{in_f})"
            );
            assert_eq!(
                s_got, p.scales,
                "row2 scales did not round-trip ({out_f}x{in_f})"
            );

            let packed_ref = packed.as_ref();

            // Build 5 distinct activation rows (anchor + 4 drafts).
            let mut st = 0x9876_5432_u64;
            let x_f32: Vec<f32> = (0..(5 * in_f))
                .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[5, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();
            let x_ref: Vec<f32> = x_f32
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect();

            let y =
                crate::metal_kernel::bonsai_q2_row2_m5_contract(&x, packed_ref, &b_canon).unwrap();
            y.eval().unwrap();
            let got = y.as_slice::<half::f16>();
            assert_eq!(got.len(), 5 * out_f);

            // CPU reference: full dequant then matvec per (verifier_row, output_row).
            let mut w_row = vec![0f32; in_f];
            for m_idx in 0..5 {
                let x_slice = &x_ref[m_idx * in_f..(m_idx + 1) * in_f];
                for r in 0..out_f {
                    p.dequant_row_to_fp32(r, &mut w_row);
                    let mut acc = 0f32;
                    for c in 0..in_f {
                        acc += x_slice[c] * w_row[c];
                    }
                    let gv = got[m_idx * out_f + r].to_f32();
                    let tol = 0.0030 + 1e-2 * acc.abs();
                    assert!(
                        (gv - acc).abs() <= tol,
                        "row2_m5 ({out_f}x{in_f}) verifier_row={m_idx} output_row={r}: got {gv} want {acc}"
                    );
                }
            }
        }
    }

    /// Phase 3D.4 kill-gate microbench: compare bonsai_q2_row2_m5_contract
    /// against the z-batched bonsai_q2_qmm baseline on the dominant Bonsai-27B
    /// verifier shapes. Promotion gate: >=1.30x speedup. Ignored by default
    /// because it requires real GPU time; run with `--ignored microbench`.
    #[test]
    #[ignore = "microbench: run with --ignored q2_row2_m5_microbench_kill_gate"]
    fn q2_row2_m5_microbench_kill_gate() {
        let _exec = crate::mlx_exec::acquire();

        // Dominant Bonsai-27B MLP verifier shapes:
        // - gate/up_proj: in=hidden=5120, out=inter=17408
        // - down_proj:    in=inter=17408, out=hidden=5120
        let shapes: &[(usize, usize, &str)] = &[
            (17408, 5120, "gate_up (N=inter, K=hidden)"),
            (5120, 17408, "down (N=hidden, K=inter)"),
        ];

        for &(out_f, in_f, label) in shapes {
            let p = make_packed_q2(out_f, in_f, 0xBEEF_BEEF);
            let (w_canon, s_canon, b_canon) = upload_to_mlx(&p);
            let packed =
                crate::metal_kernel::BonsaiQ2Row2::from_row_major(&w_canon, &s_canon).unwrap();
            let packed_ref = packed.as_ref();

            // 5-row activation tile matching the dSpark verifier (anchor + 4 drafts).
            let x_f32: Vec<f32> = (0..(5 * in_f))
                .map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5)
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[5, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();

            // Warmup
            for _ in 0..3 {
                let _ = crate::metal_kernel::bonsai_q2_qmm(&x, &w_canon, &s_canon, &b_canon, 128)
                    .unwrap()
                    .eval()
                    .unwrap();
                let _ = crate::metal_kernel::bonsai_q2_row2_m5_contract(&x, packed_ref, &b_canon)
                    .unwrap()
                    .eval()
                    .unwrap();
                let _ = crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&x, packed_ref)
                    .unwrap()
                    .eval()
                    .unwrap();
                let _ = crate::metal_kernel::bonsai_q2_row2_m5_ternary_splitk(&x, packed_ref, 2)
                    .unwrap()
                    .eval()
                    .unwrap();
                let _ = crate::metal_kernel::bonsai_q2_row2_m5_ternary_splitk(&x, packed_ref, 4)
                    .unwrap()
                    .eval()
                    .unwrap();
            }

            // Measure z-batched QMM baseline.
            let n_iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                let y = crate::metal_kernel::bonsai_q2_qmm(&x, &w_canon, &s_canon, &b_canon, 128)
                    .unwrap();
                y.eval().unwrap();
            }
            let qmm_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            // Measure row2 M=5 kernel.
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                let y = crate::metal_kernel::bonsai_q2_row2_m5_contract(&x, packed_ref, &b_canon)
                    .unwrap();
                y.eval().unwrap();
            }
            let row2_m5_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            // Measure strict-ternary row2 M=5 kernel.
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                let y =
                    crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&x, packed_ref).unwrap();
                y.eval().unwrap();
            }
            let ternary_row2_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            // Measure strict-ternary row2 split-K variants.
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                let y = crate::metal_kernel::bonsai_q2_row2_m5_ternary_splitk(&x, packed_ref, 2)
                    .unwrap();
                y.eval().unwrap();
            }
            let ternary_splitk2_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                let y = crate::metal_kernel::bonsai_q2_row2_m5_ternary_splitk(&x, packed_ref, 4)
                    .unwrap();
                y.eval().unwrap();
            }
            let ternary_splitk4_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            // Measure MLX stock affine Q2.
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                mlx_rs::ops::quantized_matmul(&x, &w_canon, &s_canon, &b_canon, true, 128, 2)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
            let mlx_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            eprintln!(
                "MICROBENCH {label} ({out_f}x{in_f}, M=5): mlx={mlx_us:.1}us qmm={qmm_us:.1}us row2_m5={row2_m5_us:.1}us ternary_row2={ternary_row2_us:.1}us splitk2={ternary_splitk2_us:.1}us splitk4={ternary_splitk4_us:.1}us mlx/ternary_row2={:.3}x mlx/splitk2={:.3}x mlx/splitk4={:.3}x qmm/ternary_row2={:.3}x",
                mlx_us / ternary_row2_us,
                mlx_us / ternary_splitk2_us,
                mlx_us / ternary_splitk4_us,
                qmm_us / ternary_row2_us
            );
        }
    }

    /// Simdgroup Q2 kernel vs MLX stock — the real comparison.
    #[test]
    #[ignore = "microbench: --ignored q2_simd_vs_mlx_stock"]
    fn q2_simd_vs_mlx_stock() {
        let _exec = crate::mlx_exec::acquire();
        for &(out_f, in_f, label) in &[
            (16384usize, 5120usize, "gdn_qkvz_like"),
            (4096usize, 5120usize, "gdn_ba_like"),
            (6144usize, 5120usize, "attn_q_o_like"),
            (1024usize, 5120usize, "attn_k_like"),
            (17408usize, 5120usize, "gate_up"),
            (5120usize, 17408usize, "down"),
            (248320usize, 5120usize, "lm_head"),
        ] {
            let p = make_packed_q2(out_f, in_f, 0xCAFE);
            let (w, s, b) = upload_to_mlx(&p);
            let x_f32: Vec<f32> = (0..in_f)
                .map(|i| i as f32 / in_f as f32 * 2.0 - 1.0)
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[1, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();
            for _ in 0..5 {
                let _ = mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, 128, 2)
                    .unwrap()
                    .eval()
                    .unwrap();
                let _ = crate::metal_kernel::bonsai_q2_qmv_simd(&x, &w, &s, &b, GROUP_SIZE as i32)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
            let n_iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                let y = mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, 128, 2).unwrap();
                y.eval().unwrap();
            }
            let mlx_us = t0.elapsed().as_micros() as f64 / n_iters as f64;
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                let y = crate::metal_kernel::bonsai_q2_qmv_simd(&x, &w, &s, &b, GROUP_SIZE as i32)
                    .unwrap();
                y.eval().unwrap();
            }
            let simd_us = t0.elapsed().as_micros() as f64 / n_iters as f64;
            let ratio = mlx_us / simd_us;
            eprintln!(
                "SIMD_VS_MLX {label} ({out_f}x{in_f}): mlx={mlx_us:.0}us simd={simd_us:.0}us ratio={ratio:.3}x"
            );
        }
    }

    #[test]
    #[ignore = "microbench: --ignored q2_m5_verifier_path_sweep"]
    fn q2_m5_verifier_path_sweep() {
        let _exec = crate::mlx_exec::acquire();
        for &(out_f, in_f, label) in &[
            (16384usize, 5120usize, "gdn_qkvz_like"),
            (4096usize, 5120usize, "gdn_ba_like"),
            (6144usize, 5120usize, "attn_q_o_like"),
            (1024usize, 5120usize, "attn_k_like"),
            (17408usize, 5120usize, "gate_up"),
            (5120usize, 17408usize, "down"),
            (248320usize, 5120usize, "lm_head"),
        ] {
            let p = make_packed_q2(out_f, in_f, 0x5151_5151);
            let (w, s, b) = upload_to_mlx(&p);
            let x_f32: Vec<f32> = (0..(5 * in_f))
                .map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5)
                .collect();
            let x = mlx_rs::Array::from_slice(&x_f32, &[5, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();

            for _ in 0..3 {
                mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, 128, 2)
                    .unwrap()
                    .eval()
                    .unwrap();
                crate::metal_kernel::bonsai_q2_qmm(&x, &w, &s, &b, GROUP_SIZE as i32)
                    .unwrap()
                    .eval()
                    .unwrap();
                crate::metal_kernel::bonsai_q2_qmv_simd(&x, &w, &s, &b, GROUP_SIZE as i32)
                    .unwrap()
                    .eval()
                    .unwrap();
                crate::metal_kernel::bonsai_q2_ternary_qmv_simd(&x, &w, &s, GROUP_SIZE as i32)
                    .unwrap()
                    .eval()
                    .unwrap();
            }

            let n_iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, 128, 2)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
            let mlx_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                crate::metal_kernel::bonsai_q2_qmm(&x, &w, &s, &b, GROUP_SIZE as i32)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
            let qmm_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                crate::metal_kernel::bonsai_q2_qmv_simd(&x, &w, &s, &b, GROUP_SIZE as i32)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
            let simd_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                crate::metal_kernel::bonsai_q2_ternary_qmv_simd(&x, &w, &s, GROUP_SIZE as i32)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
            let ternary_simd_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            eprintln!(
                "Q2_M5_SWEEP {label} ({out_f}x{in_f}): mlx={mlx_us:.1}us qmm={qmm_us:.1}us simd={simd_us:.1}us ternary_simd={ternary_simd_us:.1}us mlx/qmm={:.3}x qmm/simd={:.3}x mlx/simd={:.3}x mlx/ternary_simd={:.3}x",
                mlx_us / qmm_us,
                qmm_us / simd_us,
                mlx_us / simd_us,
                mlx_us / ternary_simd_us
            );
        }
    }

    #[test]
    #[ignore = "microbench: --ignored q2_m5_fused_gate_up_sweep"]
    fn q2_m5_fused_gate_up_sweep() {
        let _exec = crate::mlx_exec::acquire();
        let out_f = 17408usize;
        let in_f = 5120usize;
        let gate = make_packed_q2(out_f, in_f, 0x6174_6571);
        let up = make_packed_q2(out_f, in_f, 0x7570_7072);
        let (gw, gs, gb) = upload_to_mlx(&gate);
        let (uw, us, ub) = upload_to_mlx(&up);
        let fw = mlx_rs::ops::concatenate_axis(&[&gw, &uw], 0).unwrap();
        let fs = mlx_rs::ops::concatenate_axis(&[&gs, &us], 0).unwrap();
        let fb = mlx_rs::ops::concatenate_axis(&[&gb, &ub], 0).unwrap();
        let x_f32: Vec<f32> = (0..(5 * in_f))
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5)
            .collect();
        let x = mlx_rs::Array::from_slice(&x_f32, &[5, in_f as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();

        for _ in 0..3 {
            mlx_rs::ops::quantized_matmul(&x, &gw, &gs, &gb, true, 128, 2)
                .unwrap()
                .eval()
                .unwrap();
            mlx_rs::ops::quantized_matmul(&x, &uw, &us, &ub, true, 128, 2)
                .unwrap()
                .eval()
                .unwrap();
            mlx_rs::ops::quantized_matmul(&x, &fw, &fs, &fb, true, 128, 2)
                .unwrap()
                .eval()
                .unwrap();
        }

        let n_iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let gate_out = mlx_rs::ops::quantized_matmul(&x, &gw, &gs, &gb, true, 128, 2).unwrap();
            let up_out = mlx_rs::ops::quantized_matmul(&x, &uw, &us, &ub, true, 128, 2).unwrap();
            crate::mlx_exec::eval([&gate_out, &up_out].into_iter()).unwrap();
        }
        let separate_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            mlx_rs::ops::quantized_matmul(&x, &fw, &fs, &fb, true, 128, 2)
                .unwrap()
                .eval()
                .unwrap();
        }
        let fused_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        eprintln!(
            "Q2_M5_FUSED_GATE_UP gate+up ({out_f}x{in_f}) M=5: separate={separate_us:.1}us fused={fused_us:.1}us speedup={:.3}x",
            separate_us / fused_us
        );
    }

    #[test]
    #[ignore = "microbench: --ignored q2_m5_mlp_fused_gate_up_sweep"]
    fn q2_m5_mlp_fused_gate_up_sweep() {
        let _exec = crate::mlx_exec::acquire();
        let hidden_f = 5120usize;
        let intermediate_f = 17408usize;
        let gate = make_packed_q2(intermediate_f, hidden_f, 0x6761_7465);
        let up = make_packed_q2(intermediate_f, hidden_f, 0x7570_7570);
        let down = make_packed_q2(hidden_f, intermediate_f, 0x646f_776e);
        let (gw, gs, gb) = upload_to_mlx(&gate);
        let (uw, us, ub) = upload_to_mlx(&up);
        let (dw, ds, db) = upload_to_mlx(&down);
        let gate_row2 = crate::metal_kernel::BonsaiQ2Row2::from_row_major(&gw, &gs).unwrap();
        let up_row2 = crate::metal_kernel::BonsaiQ2Row2::from_row_major(&uw, &us).unwrap();
        let down_row2 = crate::metal_kernel::BonsaiQ2Row2::from_row_major(&dw, &ds).unwrap();
        let gate_row2 = gate_row2.as_ref();
        let up_row2 = up_row2.as_ref();
        let down_row2 = down_row2.as_ref();
        let fw = mlx_rs::ops::concatenate_axis(&[&gw, &uw], 0).unwrap();
        let fs = mlx_rs::ops::concatenate_axis(&[&gs, &us], 0).unwrap();
        let fb = mlx_rs::ops::concatenate_axis(&[&gb, &ub], 0).unwrap();
        let x_f32: Vec<f32> = (0..(5 * hidden_f))
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5)
            .collect();
        let x = mlx_rs::Array::from_slice(&x_f32, &[5, hidden_f as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();

        for _ in 0..3 {
            let gate_out = mlx_rs::ops::quantized_matmul(&x, &gw, &gs, &gb, true, 128, 2).unwrap();
            let up_out = mlx_rs::ops::quantized_matmul(&x, &uw, &us, &ub, true, 128, 2).unwrap();
            let act = gate_out
                .multiply(mlx_rs::nn::sigmoid(&gate_out).unwrap())
                .unwrap()
                .multiply(&up_out)
                .unwrap();
            mlx_rs::ops::quantized_matmul(&act, &dw, &ds, &db, true, 128, 2)
                .unwrap()
                .eval()
                .unwrap();

            let fused = mlx_rs::ops::quantized_matmul(&x, &fw, &fs, &fb, true, 128, 2).unwrap();
            let parts = fused
                .split_axis(&[intermediate_f as i32], Some(-1))
                .unwrap();
            let gate_out = parts.first().unwrap();
            let up_out = parts.get(1).unwrap();
            let act = gate_out
                .multiply(mlx_rs::nn::sigmoid(gate_out).unwrap())
                .unwrap()
                .multiply(up_out)
                .unwrap();
            mlx_rs::ops::quantized_matmul(&act, &dw, &ds, &db, true, 128, 2)
                .unwrap()
                .eval()
                .unwrap();

            let gate_out =
                crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&x, gate_row2).unwrap();
            let up_out =
                crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&x, up_row2).unwrap();
            let act = gate_out
                .multiply(mlx_rs::nn::sigmoid(&gate_out).unwrap())
                .unwrap()
                .multiply(&up_out)
                .unwrap();
            crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&act, down_row2)
                .unwrap()
                .eval()
                .unwrap();
            mlx_rs::ops::quantized_matmul(&act, &dw, &ds, &db, true, 128, 2)
                .unwrap()
                .eval()
                .unwrap();

            let fused_gu = crate::metal_kernel::bonsai_q2_row2_m5_ternary_fused_gate_up(
                &x, gate_row2, up_row2,
            )
            .unwrap();
            let parts = fused_gu
                .split_axis(&[intermediate_f as i32], Some(-1))
                .unwrap();
            let gate_out = parts.first().unwrap();
            let up_out = parts.get(1).unwrap();
            let act = gate_out
                .multiply(mlx_rs::nn::sigmoid(gate_out).unwrap())
                .unwrap()
                .multiply(up_out)
                .unwrap();
            crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&act, down_row2)
                .unwrap()
                .eval()
                .unwrap();
        }

        let n_iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let gate_out = mlx_rs::ops::quantized_matmul(&x, &gw, &gs, &gb, true, 128, 2).unwrap();
            let up_out = mlx_rs::ops::quantized_matmul(&x, &uw, &us, &ub, true, 128, 2).unwrap();
            let act = gate_out
                .multiply(mlx_rs::nn::sigmoid(&gate_out).unwrap())
                .unwrap()
                .multiply(&up_out)
                .unwrap();
            mlx_rs::ops::quantized_matmul(&act, &dw, &ds, &db, true, 128, 2)
                .unwrap()
                .eval()
                .unwrap();
        }
        let separate_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let fused = mlx_rs::ops::quantized_matmul(&x, &fw, &fs, &fb, true, 128, 2).unwrap();
            let parts = fused
                .split_axis(&[intermediate_f as i32], Some(-1))
                .unwrap();
            let gate_out = parts.first().unwrap();
            let up_out = parts.get(1).unwrap();
            let act = gate_out
                .multiply(mlx_rs::nn::sigmoid(gate_out).unwrap())
                .unwrap()
                .multiply(up_out)
                .unwrap();
            mlx_rs::ops::quantized_matmul(&act, &dw, &ds, &db, true, 128, 2)
                .unwrap()
                .eval()
                .unwrap();
        }
        let fused_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let gate_out =
                crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&x, gate_row2).unwrap();
            let up_out =
                crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&x, up_row2).unwrap();
            let act = gate_out
                .multiply(mlx_rs::nn::sigmoid(&gate_out).unwrap())
                .unwrap()
                .multiply(&up_out)
                .unwrap();
            crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&act, down_row2)
                .unwrap()
                .eval()
                .unwrap();
        }
        let ternary_row2_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let gate_out =
                crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&x, gate_row2).unwrap();
            let up_out =
                crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&x, up_row2).unwrap();
            let act = gate_out
                .multiply(mlx_rs::nn::sigmoid(&gate_out).unwrap())
                .unwrap()
                .multiply(&up_out)
                .unwrap();
            mlx_rs::ops::quantized_matmul(&act, &dw, &ds, &db, true, 128, 2)
                .unwrap()
                .eval()
                .unwrap();
        }
        let hybrid_row2_mlx_down_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let fused_gu = crate::metal_kernel::bonsai_q2_row2_m5_ternary_fused_gate_up(
                &x, gate_row2, up_row2,
            )
            .unwrap();
            let parts = fused_gu
                .split_axis(&[intermediate_f as i32], Some(-1))
                .unwrap();
            let gate_out = parts.first().unwrap();
            let up_out = parts.get(1).unwrap();
            let act = gate_out
                .multiply(mlx_rs::nn::sigmoid(gate_out).unwrap())
                .unwrap()
                .multiply(up_out)
                .unwrap();
            crate::metal_kernel::bonsai_q2_row2_m5_ternary_direct(&act, down_row2)
                .unwrap()
                .eval()
                .unwrap();
        }
        let fused_row2_gate_up_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        eprintln!(
            "Q2_M5_MLP_FUSED_GATE_UP hidden={hidden_f} intermediate={intermediate_f} M=5: separate={separate_us:.1}us fused={fused_us:.1}us ternary_row2={ternary_row2_us:.1}us hybrid_row2_mlx_down={hybrid_row2_mlx_down_us:.1}us fused_row2_gate_up={fused_row2_gate_up_us:.1}us fused_speedup={:.3}x row2_speedup={:.3}x hybrid_speedup={:.3}x fused_row2_gate_up_speedup={:.3}x",
            separate_us / fused_us,
            separate_us / ternary_row2_us,
            separate_us / hybrid_row2_mlx_down_us,
            separate_us / fused_row2_gate_up_us
        );
    }

    #[test]
    #[ignore = "microbench: --ignored q2_m5_rank2_vs_rank3_sweep"]
    fn q2_m5_rank2_vs_rank3_sweep() {
        let _exec = crate::mlx_exec::acquire();
        for &(out_f, in_f, label) in &[
            (16384usize, 5120usize, "gdn_qkvz_like"),
            (4096usize, 5120usize, "gdn_ba_like"),
            (6144usize, 5120usize, "attn_q_o_like"),
            (1024usize, 5120usize, "attn_k_like"),
            (17408usize, 5120usize, "gate_up"),
            (5120usize, 17408usize, "down"),
            (248320usize, 5120usize, "lm_head"),
        ] {
            let p = make_packed_q2(out_f, in_f, 0x7232_7233);
            let (w, s, b) = upload_to_mlx(&p);
            let x_f32: Vec<f32> = (0..(5 * in_f))
                .map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5)
                .collect();
            let x2 = mlx_rs::Array::from_slice(&x_f32, &[5, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();
            let x3 = mlx_rs::Array::from_slice(&x_f32, &[1, 5, in_f as i32])
                .as_dtype(mlx_rs::Dtype::Float16)
                .unwrap();

            for _ in 0..3 {
                mlx_rs::ops::quantized_matmul(&x2, &w, &s, &b, true, 128, 2)
                    .unwrap()
                    .eval()
                    .unwrap();
                mlx_rs::ops::quantized_matmul(&x3, &w, &s, &b, true, 128, 2)
                    .unwrap()
                    .eval()
                    .unwrap();
            }

            let n_iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                mlx_rs::ops::quantized_matmul(&x2, &w, &s, &b, true, 128, 2)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
            let rank2_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                mlx_rs::ops::quantized_matmul(&x3, &w, &s, &b, true, 128, 2)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
            let rank3_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            eprintln!(
                "Q2_M5_RANK2_VS_RANK3 {label} ({out_f}x{in_f}): rank2={rank2_us:.1}us rank3={rank3_us:.1}us rank2/rank3={:.3}x",
                rank2_us / rank3_us
            );
        }
    }

    #[test]
    #[ignore = "microbench: --ignored q2_m_scaling_sweep"]
    fn q2_m_scaling_sweep() {
        let _exec = crate::mlx_exec::acquire();
        for &(out_f, in_f, label) in &[
            (16384usize, 5120usize, "gdn_qkvz_like"),
            (17408usize, 5120usize, "gate_up"),
            (5120usize, 17408usize, "down"),
            (248320usize, 5120usize, "lm_head"),
        ] {
            let p = make_packed_q2(out_f, in_f, 0x6d73_636c);
            let (w, s, b) = upload_to_mlx(&p);
            let mut baseline_m1_us = None;
            for &m_rows in &[1usize, 2, 4, 5, 8] {
                let x_f32: Vec<f32> = (0..(m_rows * in_f))
                    .map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5)
                    .collect();
                let x = mlx_rs::Array::from_slice(&x_f32, &[m_rows as i32, in_f as i32])
                    .as_dtype(mlx_rs::Dtype::Float16)
                    .unwrap();
                for _ in 0..3 {
                    mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, 128, 2)
                        .unwrap()
                        .eval()
                        .unwrap();
                }
                let n_iters = 20;
                let t0 = std::time::Instant::now();
                for _ in 0..n_iters {
                    mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, 128, 2)
                        .unwrap()
                        .eval()
                        .unwrap();
                }
                let us = t0.elapsed().as_micros() as f64 / n_iters as f64;
                let m1 = *baseline_m1_us.get_or_insert(us);
                eprintln!(
                    "Q2_M_SCALING {label} ({out_f}x{in_f}) M={m_rows}: total={us:.1}us per_row={:.1}us speedup_vs_m1_serial={:.3}x",
                    us / m_rows as f64,
                    (m1 * m_rows as f64) / us
                );
            }
        }
    }

    #[test]
    #[ignore = "microbench: --ignored q2_m5_lm_head_argmax_candidate_sweep"]
    fn q2_m5_lm_head_argmax_candidate_sweep() {
        let _exec = crate::mlx_exec::acquire();
        let out_f = 248320usize;
        let in_f = 5120usize;
        let p = make_packed_q2(out_f, in_f, 0x6865_6164);
        let (w, s, _b) = upload_to_mlx(&p);
        let b = s.negative().unwrap();
        let x_f32: Vec<f32> = (0..(5 * in_f))
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5)
            .collect();
        let x = mlx_rs::Array::from_slice(&x_f32, &[5, in_f as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();

        for _ in 0..3 {
            let logits = mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, 128, 2).unwrap();
            mlx_rs::argmax_axis!(&logits, -1).unwrap().eval().unwrap();
            let (maxv, maxid) =
                crate::metal_kernel::bonsai_q2_m5_argmax_candidates(&x, &w, &s, &b, 128).unwrap();
            crate::mlx_exec::eval([&maxv, &maxid].into_iter()).unwrap();
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&maxv, &maxid)
                .unwrap()
                .eval()
                .unwrap();
            let (maxv, maxid) =
                crate::metal_kernel::bonsai_q2_m5_ternary_argmax_candidates(&x, &w, &s, 128)
                    .unwrap();
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&maxv, &maxid)
                .unwrap()
                .eval()
                .unwrap();
            let (maxv, maxid) = crate::metal_kernel::bonsai_q2_m5_ternary_argmax_candidates_splitk(
                &x, &w, &s, 128, 2,
            )
            .unwrap();
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&maxv, &maxid)
                .unwrap()
                .eval()
                .unwrap();
            let (maxv, maxid) = crate::metal_kernel::bonsai_q2_m5_ternary_argmax_candidates_splitk(
                &x, &w, &s, 128, 4,
            )
            .unwrap();
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&maxv, &maxid)
                .unwrap()
                .eval()
                .unwrap();
        }

        let logits = mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, 128, 2).unwrap();
        let ref_ids_arr = mlx_rs::argmax_axis!(&logits, -1).unwrap();
        ref_ids_arr.eval().unwrap();
        let ref_ids: Vec<u32> = ref_ids_arr.as_slice::<u32>().to_vec();
        let (maxv, maxid) =
            crate::metal_kernel::bonsai_q2_m5_argmax_candidates(&x, &w, &s, &b, 128).unwrap();
        crate::mlx_exec::eval([&maxv, &maxid].into_iter()).unwrap();
        let values = maxv.as_slice::<f32>();
        let ids = maxid.as_slice::<f32>();
        let blocks = values.len() / 5;
        let mut cand_ids = Vec::with_capacity(5);
        for row in 0..5 {
            let mut best_v = f32::NEG_INFINITY;
            let mut best_id = 0u32;
            for block in 0..blocks {
                let idx = row * blocks + block;
                let v = values[idx];
                let id = ids[idx] as u32;
                if v > best_v || (v == best_v && id < best_id) {
                    best_v = v;
                    best_id = id;
                }
            }
            cand_ids.push(best_id);
        }
        eprintln!("Q2_M5_HEAD_ARGMAX_PARITY ref={ref_ids:?} cand={cand_ids:?}");
        let gpu_ids_arr =
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&maxv, &maxid).unwrap();
        gpu_ids_arr.eval().unwrap();
        let gpu_ids: Vec<u32> = gpu_ids_arr.as_slice::<u32>().to_vec();
        eprintln!("Q2_M5_HEAD_ARGMAX_GPU_REDUCE gpu={gpu_ids:?}");
        let (tmaxv, tmaxid) =
            crate::metal_kernel::bonsai_q2_m5_ternary_argmax_candidates(&x, &w, &s, 128).unwrap();
        let ternary_gpu_ids_arr =
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&tmaxv, &tmaxid).unwrap();
        ternary_gpu_ids_arr.eval().unwrap();
        let ternary_gpu_ids: Vec<u32> = ternary_gpu_ids_arr.as_slice::<u32>().to_vec();
        eprintln!("Q2_M5_HEAD_ARGMAX_TERNARY_GPU_REDUCE gpu={ternary_gpu_ids:?}");
        let (s2maxv, s2maxid) =
            crate::metal_kernel::bonsai_q2_m5_ternary_argmax_candidates_splitk(&x, &w, &s, 128, 2)
                .unwrap();
        let splitk2_ids_arr =
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&s2maxv, &s2maxid).unwrap();
        splitk2_ids_arr.eval().unwrap();
        let splitk2_ids: Vec<u32> = splitk2_ids_arr.as_slice::<u32>().to_vec();
        let (s4maxv, s4maxid) =
            crate::metal_kernel::bonsai_q2_m5_ternary_argmax_candidates_splitk(&x, &w, &s, 128, 4)
                .unwrap();
        let splitk4_ids_arr =
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&s4maxv, &s4maxid).unwrap();
        splitk4_ids_arr.eval().unwrap();
        let splitk4_ids: Vec<u32> = splitk4_ids_arr.as_slice::<u32>().to_vec();
        eprintln!(
            "Q2_M5_HEAD_ARGMAX_TERNARY_SPLITK splitk2={splitk2_ids:?} splitk4={splitk4_ids:?}"
        );

        let n_iters = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let logits = mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, 128, 2).unwrap();
            mlx_rs::argmax_axis!(&logits, -1).unwrap().eval().unwrap();
        }
        let mlx_argmax_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let (maxv, maxid) =
                crate::metal_kernel::bonsai_q2_m5_argmax_candidates(&x, &w, &s, &b, 128).unwrap();
            crate::mlx_exec::eval([&maxv, &maxid].into_iter()).unwrap();
            let values = maxv.as_slice::<f32>();
            let ids = maxid.as_slice::<f32>();
            let blocks = values.len() / 5;
            let mut sink = 0u32;
            for row in 0..5 {
                let mut best_v = f32::NEG_INFINITY;
                let mut best_id = 0u32;
                for block in 0..blocks {
                    let idx = row * blocks + block;
                    let v = values[idx];
                    let id = ids[idx] as u32;
                    if v > best_v || (v == best_v && id < best_id) {
                        best_v = v;
                        best_id = id;
                    }
                }
                sink ^= best_id;
            }
            std::hint::black_box(sink);
        }
        let candidate_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let (maxv, maxid) =
                crate::metal_kernel::bonsai_q2_m5_argmax_candidates(&x, &w, &s, &b, 128).unwrap();
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&maxv, &maxid)
                .unwrap()
                .eval()
                .unwrap();
        }
        let gpu_reduce_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let (maxv, maxid) =
                crate::metal_kernel::bonsai_q2_m5_ternary_argmax_candidates(&x, &w, &s, 128)
                    .unwrap();
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&maxv, &maxid)
                .unwrap()
                .eval()
                .unwrap();
        }
        let ternary_gpu_reduce_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let (maxv, maxid) = crate::metal_kernel::bonsai_q2_m5_ternary_argmax_candidates_splitk(
                &x, &w, &s, 128, 2,
            )
            .unwrap();
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&maxv, &maxid)
                .unwrap()
                .eval()
                .unwrap();
        }
        let ternary_splitk2_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let (maxv, maxid) = crate::metal_kernel::bonsai_q2_m5_ternary_argmax_candidates_splitk(
                &x, &w, &s, 128, 4,
            )
            .unwrap();
            crate::metal_kernel::bonsai_q2_m5_argmax_reduce_ids(&maxv, &maxid)
                .unwrap()
                .eval()
                .unwrap();
        }
        let ternary_splitk4_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        eprintln!(
            "Q2_M5_HEAD_ARGMAX lm_head ({out_f}x{in_f}) M=5: mlx_qmm_argmax={mlx_argmax_us:.1}us candidate_cpu_reduce={candidate_us:.1}us candidate_gpu_reduce={gpu_reduce_us:.1}us ternary_gpu_reduce={ternary_gpu_reduce_us:.1}us ternary_splitk2={ternary_splitk2_us:.1}us ternary_splitk4={ternary_splitk4_us:.1}us cpu_speedup={:.3}x gpu_speedup={:.3}x ternary_speedup={:.3}x splitk2_speedup={:.3}x splitk4_speedup={:.3}x gpu_vs_cpu={:.3}x ternary_vs_affine_gpu={:.3}x splitk2_vs_direct={:.3}x splitk4_vs_direct={:.3}x",
            mlx_argmax_us / candidate_us,
            mlx_argmax_us / gpu_reduce_us,
            mlx_argmax_us / ternary_gpu_reduce_us,
            mlx_argmax_us / ternary_splitk2_us,
            mlx_argmax_us / ternary_splitk4_us,
            candidate_us / gpu_reduce_us,
            gpu_reduce_us / ternary_gpu_reduce_us,
            ternary_gpu_reduce_us / ternary_splitk2_us,
            ternary_gpu_reduce_us / ternary_splitk4_us
        );
    }

    #[test]
    #[ignore = "microbench: --ignored q4_m4_output_head_argmax_candidate_sweep"]
    fn q4_m4_output_head_argmax_candidate_sweep() {
        let _exec = crate::mlx_exec::acquire();
        let out_f = 248320usize;
        let in_f = 5120usize;
        let group_size = 32i32;
        let (w, s, b) = make_q4_arrays(out_f, in_f, group_size as usize, 0x6473_706b);
        let x_f32: Vec<f32> = (0..(4 * in_f))
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5)
            .collect();
        let x = mlx_rs::Array::from_slice(&x_f32, &[4, in_f as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();

        for _ in 0..3 {
            let logits =
                mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, group_size, 4).unwrap();
            mlx_rs::argmax_axis!(&logits, -1).unwrap().eval().unwrap();
            let (maxv, maxid) =
                crate::metal_kernel::bonsai_q4_m4_argmax_candidates(&x, &w, &s, &b, group_size)
                    .unwrap();
            crate::metal_kernel::bonsai_q4_m4_argmax_reduce_ids(&maxv, &maxid)
                .unwrap()
                .eval()
                .unwrap();
        }

        let logits = mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, group_size, 4).unwrap();
        let ref_ids_arr = mlx_rs::argmax_axis!(&logits, -1).unwrap();
        ref_ids_arr.eval().unwrap();
        let ref_ids: Vec<u32> = ref_ids_arr.as_slice::<u32>().to_vec();
        let (maxv, maxid) =
            crate::metal_kernel::bonsai_q4_m4_argmax_candidates(&x, &w, &s, &b, group_size)
                .unwrap();
        let gpu_ids_arr =
            crate::metal_kernel::bonsai_q4_m4_argmax_reduce_ids(&maxv, &maxid).unwrap();
        gpu_ids_arr.eval().unwrap();
        let gpu_ids: Vec<u32> = gpu_ids_arr.as_slice::<u32>().to_vec();
        eprintln!("Q4_M4_OUTPUT_HEAD_ARGMAX_PARITY ref={ref_ids:?} cand={gpu_ids:?}");

        let n_iters = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let logits =
                mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, group_size, 4).unwrap();
            mlx_rs::argmax_axis!(&logits, -1).unwrap().eval().unwrap();
        }
        let mlx_argmax_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let (maxv, maxid) =
                crate::metal_kernel::bonsai_q4_m4_argmax_candidates(&x, &w, &s, &b, group_size)
                    .unwrap();
            crate::metal_kernel::bonsai_q4_m4_argmax_reduce_ids(&maxv, &maxid)
                .unwrap()
                .eval()
                .unwrap();
        }
        let candidate_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        eprintln!(
            "Q4_M4_OUTPUT_HEAD_ARGMAX output_head ({out_f}x{in_f}) M=4: mlx_qmm_argmax={mlx_argmax_us:.1}us candidate_gpu_reduce={candidate_us:.1}us speedup={:.3}x",
            mlx_argmax_us / candidate_us
        );
    }

    #[test]
    #[ignore = "microbench: --ignored q4_m1_markov_biased_output_head_argmax_sweep"]
    fn q4_m1_markov_biased_output_head_argmax_sweep() {
        let _exec = crate::mlx_exec::acquire();
        let out_f = 248320usize;
        let in_f = 5120usize;
        let markov_rank = 256usize;
        let group_size = 32i32;
        let (w, s, b) = make_q4_arrays(out_f, in_f, group_size as usize, 0x6473_706b);
        let (mw, ms, mb) = make_q4_arrays(out_f, markov_rank, group_size as usize, 0x6d61_726b);
        let x_f32: Vec<f32> = (0..(4 * in_f))
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5)
            .collect();
        let markov_f32: Vec<f32> = (0..(4 * markov_rank))
            .map(|i| ((i as u32).wrapping_mul(2246822519) >> 8) as f32 / 16777216.0 - 0.5)
            .collect();
        let x = mlx_rs::Array::from_slice(&x_f32, &[4, in_f as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();
        let markov = mlx_rs::Array::from_slice(&markov_f32, &[4, markov_rank as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();

        for _ in 0..3 {
            let base = mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, group_size, 4).unwrap();
            for step in 0..4 {
                let base_row = base.index(step).reshape(&[1, out_f as i32]).unwrap();
                let markov_row = markov
                    .index(step)
                    .reshape(&[1, markov_rank as i32])
                    .unwrap();
                let bias =
                    mlx_rs::ops::quantized_matmul(&markov_row, &mw, &ms, &mb, true, group_size, 4)
                        .unwrap();
                mlx_rs::argmax_axis!(&base_row.add(&bias).unwrap(), -1)
                    .unwrap()
                    .eval()
                    .unwrap();

                let x_row = x.index(step);
                let (maxv, maxid) = crate::metal_kernel::bonsai_q4_m1_plus_q4_m1_argmax_candidates(
                    &x_row,
                    &w,
                    &s,
                    &b,
                    &markov_row,
                    &mw,
                    &ms,
                    &mb,
                    group_size,
                    group_size,
                )
                .unwrap();
                crate::metal_kernel::bonsai_q4_m1_argmax_reduce_ids(&maxv, &maxid)
                    .unwrap()
                    .eval()
                    .unwrap();

                let (maxv, maxid) = crate::metal_kernel::bonsai_q4_markov_m1_argmax_candidates(
                    &base_row,
                    &markov_row,
                    &mw,
                    &ms,
                    &mb,
                    group_size,
                )
                .unwrap();
                crate::metal_kernel::bonsai_q4_m1_argmax_reduce_ids(&maxv, &maxid)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
        }

        let base = mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, group_size, 4).unwrap();
        let mut ref_ids = Vec::new();
        let mut fused_ids = Vec::new();
        let mut markov_only_ids = Vec::new();
        for step in 0..4 {
            let base_row = base.index(step).reshape(&[1, out_f as i32]).unwrap();
            let markov_row = markov
                .index(step)
                .reshape(&[1, markov_rank as i32])
                .unwrap();
            let bias =
                mlx_rs::ops::quantized_matmul(&markov_row, &mw, &ms, &mb, true, group_size, 4)
                    .unwrap();
            let ref_id_arr = mlx_rs::argmax_axis!(&base_row.add(&bias).unwrap(), -1).unwrap();
            ref_id_arr.eval().unwrap();
            ref_ids.extend(ref_id_arr.as_slice::<u32>().iter().copied());

            let x_row = x.index(step);
            let (maxv, maxid) = crate::metal_kernel::bonsai_q4_m1_plus_q4_m1_argmax_candidates(
                &x_row,
                &w,
                &s,
                &b,
                &markov_row,
                &mw,
                &ms,
                &mb,
                group_size,
                group_size,
            )
            .unwrap();
            let fused_id_arr =
                crate::metal_kernel::bonsai_q4_m1_argmax_reduce_ids(&maxv, &maxid).unwrap();
            fused_id_arr.eval().unwrap();
            fused_ids.extend(fused_id_arr.as_slice::<u32>().iter().copied());

            let (maxv, maxid) = crate::metal_kernel::bonsai_q4_markov_m1_argmax_candidates(
                &base_row,
                &markov_row,
                &mw,
                &ms,
                &mb,
                group_size,
            )
            .unwrap();
            let markov_only_id_arr =
                crate::metal_kernel::bonsai_q4_m1_argmax_reduce_ids(&maxv, &maxid).unwrap();
            markov_only_id_arr.eval().unwrap();
            markov_only_ids.extend(markov_only_id_arr.as_slice::<u32>().iter().copied());
        }
        eprintln!(
            "Q4_M1_MARKOV_ARGMAX_PARITY ref={ref_ids:?} fused={fused_ids:?} markov_only={markov_only_ids:?}"
        );

        let n_iters = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let base = mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, group_size, 4).unwrap();
            for step in 0..4 {
                let base_row = base.index(step).reshape(&[1, out_f as i32]).unwrap();
                let markov_row = markov
                    .index(step)
                    .reshape(&[1, markov_rank as i32])
                    .unwrap();
                let bias =
                    mlx_rs::ops::quantized_matmul(&markov_row, &mw, &ms, &mb, true, group_size, 4)
                        .unwrap();
                mlx_rs::argmax_axis!(&base_row.add(&bias).unwrap(), -1)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
        }
        let mlx_markov_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            for step in 0..4 {
                let x_row = x.index(step);
                let markov_row = markov
                    .index(step)
                    .reshape(&[1, markov_rank as i32])
                    .unwrap();
                let (maxv, maxid) = crate::metal_kernel::bonsai_q4_m1_plus_q4_m1_argmax_candidates(
                    &x_row,
                    &w,
                    &s,
                    &b,
                    &markov_row,
                    &mw,
                    &ms,
                    &mb,
                    group_size,
                    group_size,
                )
                .unwrap();
                crate::metal_kernel::bonsai_q4_m1_argmax_reduce_ids(&maxv, &maxid)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
        }
        let fused_markov_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let base = mlx_rs::ops::quantized_matmul(&x, &w, &s, &b, true, group_size, 4).unwrap();
            for step in 0..4 {
                let base_row = base.index(step).reshape(&[1, out_f as i32]).unwrap();
                let markov_row = markov
                    .index(step)
                    .reshape(&[1, markov_rank as i32])
                    .unwrap();
                let (maxv, maxid) = crate::metal_kernel::bonsai_q4_markov_m1_argmax_candidates(
                    &base_row,
                    &markov_row,
                    &mw,
                    &ms,
                    &mb,
                    group_size,
                )
                .unwrap();
                crate::metal_kernel::bonsai_q4_m1_argmax_reduce_ids(&maxv, &maxid)
                    .unwrap()
                    .eval()
                    .unwrap();
            }
        }
        let markov_only_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        eprintln!(
            "Q4_M1_MARKOV_ARGMAX output_head ({out_f}x{in_f}) + markov_b ({out_f}x{markov_rank}) x4: mlx_m4_output_plus_markov={mlx_markov_us:.1}us fused_m1x4={fused_markov_us:.1}us markov_only_fused={markov_only_us:.1}us fused_speedup={:.3}x markov_only_speedup={:.3}x",
            mlx_markov_us / fused_markov_us,
            mlx_markov_us / markov_only_us
        );
    }
}
