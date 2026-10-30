//! `qmm8` (weight-once 8-row affine-4bit matmul) agrees with MLX's
//! `quantized_matmul` within bf16 rounding, for partial and full row tiles.
#![allow(clippy::unwrap_used)]

use higgs_models::qmm8::probe;
use mlx_rs::{Array, Dtype, ops, random};

#[test]
fn qmm8_matches_quantized_matmul() {
    random::seed(7).unwrap();
    let (n, k) = (72, 640); // N % 8 == 0, K = 10 groups, odd group count
    let w = random::normal::<f32>(&[n, k], None, None, None)
        .unwrap()
        .multiply(Array::from_f32(0.05))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let (wq, s, b) = ops::quantize(&w, 64, 4).unwrap();
    for m in [1, 3, 8] {
        let x = random::normal::<f32>(&[1, m, k], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        assert!(probe::supported(&x, &wq));
        let want = ops::quantized_matmul(&x, &wq, &s, &b, true, 64, 4)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let got = probe::qmm8(&x, &wq, &s, &b).unwrap();
        assert_eq!(got.shape(), &[1, m, n]);
        let got = got.as_dtype(Dtype::Float32).unwrap();
        let err: f32 = ops::abs(&got.subtract(&want).unwrap())
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        let mag: f32 = ops::abs(&want).unwrap().max(None).unwrap().item();
        // Both round to bf16; half activations add ~2^-11 relative per input.
        assert!(err <= 0.02 * mag, "M={m}: max err {err} vs magnitude {mag}");
    }
}
