#![allow(clippy::tests_outside_test_module, clippy::unwrap_used)]
//! Speculative verify must reproduce decode. Bonsai-2's ternary packed
//! projections at the verify width (2..=8 rows: a `DFlash` block or MTP draft
//! plus its anchor) are compared row by row with the same rows decoded one at
//! a time, both through the real `QLinear` dispatch.

use half::f16;
use higgs_models::ternary_linear_probe::forward;
use mlx_rs::{Array, Dtype};

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    u32::try_from(*state >> 32).unwrap()
}

/// Uniform in [0, 1] at 16-bit resolution.
fn unit(state: &mut u64) -> f32 {
    f32::from(u16::try_from(lcg(state) >> 16).unwrap()) / f32::from(u16::MAX)
}

fn to_f32(array: &Array) -> Vec<f32> {
    let values = array.as_dtype(Dtype::Float32).unwrap();
    values.eval().unwrap();
    values.as_slice::<f32>().to_vec()
}

fn dim(n: usize) -> i32 {
    i32::try_from(n).unwrap()
}

#[test]
fn verify_rows_match_decode_rows() {
    let _exec = higgs_models::mlx_exec::acquire();
    let block = 1024;
    // Bonsai-2's reduction widths (hidden 5120, GDN/attention output 6144,
    // MLP down 17408); output widths that end mid-tile, as in_proj_ba's 96.
    for (n, k, seed) in [(1000, 5120, 1), (96, 6144, 2), (520, 17_408, 3)] {
        let mut st = seed;
        // The pack's contract: codes {0,1,2}, FP16 group scales, biases == -scales.
        let codes: Vec<u32> = (0..n * k / 16)
            .map(|_| (0..16).fold(0, |w, i| w | (lcg(&mut st) % 3) << (2 * i)))
            .collect();
        let scales: Vec<f16> = (0..n * k / 128)
            .map(|_| f16::from_f32(0.04_f32.mul_add(unit(&mut st), 0.01)))
            .collect();
        let s_max = scales.iter().map(|s| s.to_f32()).fold(0.0, f32::max);
        let biases: Vec<f16> = scales.iter().map(|&s| -s).collect();
        let sign_values: Vec<f32> = (0..k)
            .map(|_| if lcg(&mut st) & 1 == 0 { 1.0 } else { -1.0 })
            .collect();
        let w = Array::from_slice(&codes, &[dim(n), dim(k / 16)]);
        let s = Array::from_slice(&scales, &[dim(n), dim(k / 128)]);
        let b = Array::from_slice(&biases, &[dim(n), dim(k / 128)]);
        let signs = Array::from_slice(&sign_values, &[dim(k)]);
        let project = |rows_of_x: &[f32], rows: usize| {
            let input = Array::from_slice(rows_of_x, &[1, dim(rows), dim(k)]);
            to_f32(&forward(&input, &w, &s, &b, &signs, block).unwrap())
        };

        // 2..=8 take the verify kernel; 9 must still fall back cleanly.
        for m in 2..=9 {
            // Activations with sparse large outliers, like residual-stream channels.
            let x: Vec<f32> = (0..m * k)
                .map(|i| unit(&mut st).mul_add(2.0, -1.0) * if i % 97 == 0 { 40.0 } else { 1.0 })
                .collect();
            let verify = project(&x, m);
            assert_eq!(verify.len(), m * n);
            for (row, (xr, verify_row)) in x.chunks_exact(k).zip(verify.chunks_exact(n)).enumerate()
            {
                let decode = project(xr, 1);
                // Both paths accumulate the same dot products in FP32. The
                // rotation is orthonormal, so s_max * sqrt(k) * |x|_2 bounds
                // sum|s w x_rotated|; 2^-16 of it is 256 FP32 unit roundoffs,
                // far above FP32 reordering, far below 16-bit inputs/outputs.
                let norm = xr.iter().map(|v| v * v).sum::<f32>().sqrt();
                let root_k = f32::from(u16::try_from(k).unwrap()).sqrt();
                let bound = s_max * root_k * norm / 65536.0;
                let worst = verify_row
                    .iter()
                    .zip(&decode)
                    .map(|(v, d)| (v - d).abs())
                    .fold(0.0, f32::max);
                assert!(
                    worst <= bound,
                    "[{n}x{k}] M={m} row {row}: verify vs decode {worst:e} > bound {bound:e}"
                );
            }
        }
    }
}
