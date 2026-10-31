//! `qmm8` (weight-once 8-row affine-4bit matmul) agrees with MLX's
//! `quantized_matmul` within bf16 rounding, for partial and full row tiles,
//! a partial last column block (N not a multiple of the 16-row block) and
//! k-slices with a trailing partial chunk.
#![allow(clippy::unwrap_used)]

use higgs_models::qmm8::probe;
use mlx_rs::{Array, Dtype, ops, random};

#[test]
fn qmm8_matches_quantized_matmul() {
    random::seed(7).unwrap();
    // (N, K): 72 x 256 is one fragment per simdgroup, 4 groups, one slice;
    // 136 x 512 is two fragments (the last block half live), 8 groups over
    // two k-slices of 4 (a chunk shorter than 8 groups, summed by the reduce).
    for (n, k) in [(72, 256), (136, 512)] {
        let (nf, _, s) = probe::plan(n, k);
        assert_eq!((nf, s), if n == 72 { (1, 1) } else { (2, 2) });
        let w = random::normal::<f32>(&[n, k], None, None, None)
            .unwrap()
            .multiply(Array::from_f32(0.05))
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let (wq, sc, b) = ops::quantize(&w, 64, 4).unwrap();
        for m in [1, 3, 8] {
            let x = random::normal::<f32>(&[1, m, k], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            assert!(probe::supported(&x, &wq));
            let want = ops::quantized_matmul(&x, &wq, &sc, &b, true, 64, 4)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            let got = probe::qmm8(&x, &wq, &sc, &b).unwrap();
            assert_eq!(got.shape(), &[1, m, n]);
            assert_eq!(got.dtype(), Dtype::Bfloat16);
            let got = got.as_dtype(Dtype::Float32).unwrap();
            let err: f32 = ops::abs(&got.subtract(&want).unwrap())
                .unwrap()
                .max(None)
                .unwrap()
                .item();
            let mag: f32 = ops::abs(&want).unwrap().max(None).unwrap().item();
            // Both round to bf16; half activations add ~2^-11 relative per input.
            assert!(
                err <= 0.02 * mag,
                "N={n} K={k} M={m}: max err {err} vs magnitude {mag}"
            );
        }
    }
}

/// Three projections sharing `x` in one dispatch (q|k|v: a wide and two
/// narrow segments) match the three separate matmuls.
#[test]
fn qmm8_multi_matches_separate_matmuls() {
    random::seed(11).unwrap();
    let k = 512;
    let ws: Vec<(Array, Array, Array)> = [136, 24, 8]
        .iter()
        .map(|&n| {
            let w = random::normal::<f32>(&[n, k], None, None, None)
                .unwrap()
                .multiply(Array::from_f32(0.05))
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            ops::quantize(&w, 64, 4).unwrap()
        })
        .collect();
    let x = random::normal::<f32>(&[1, 5, k], None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let segs: Vec<(&Array, &Array, &Array)> = ws.iter().map(|(w, s, b)| (w, s, b)).collect();
    let got = probe::qmm8_multi(&x, &segs).unwrap();
    assert_eq!(got.len(), 3);
    for ((w, s, b), y) in ws.iter().zip(&got) {
        let want = ops::quantized_matmul(&x, w, s, b, true, 64, 4)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        assert_eq!(y.shape(), want.shape());
        let y = y.as_dtype(Dtype::Float32).unwrap();
        let err: f32 = ops::abs(&y.subtract(&want).unwrap())
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        let mag: f32 = ops::abs(&want).unwrap().max(None).unwrap().item();
        assert!(
            err <= 0.02 * mag,
            "N={}: max err {err} vs magnitude {mag}",
            w.dim(0)
        );
    }
}
