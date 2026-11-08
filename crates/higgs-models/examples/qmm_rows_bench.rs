//! Time MLX affine-4bit g64 `quantized_matmul` vs row count M at Clover's
//! dense projection shapes, and the 8-row `qmm8` kernel against it (achieved
//! GB/s over the weight + scale + bias bytes; floor at 105 GB/s).
//!
//! `QMM_SHAPES=N:K,..` overrides the shapes; `QMM_M=1,8` the row counts.
#![allow(clippy::unwrap_used, clippy::print_stdout, clippy::cast_precision_loss)]

use std::time::Instant;

use higgs_models::qmm8::probe;
use mlx_rs::{Array, Dtype, ops, random, transforms::eval};

fn time(f: impl Fn() -> Array) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..30 {
        let t = Instant::now();
        // ponytail: 10 back-to-back calls per sample to amortize launch/sync
        let ys: Vec<Array> = (0..10).map(|_| f()).collect();
        eval(ys.iter()).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / 10.0);
    }
    best
}

fn main() {
    let shapes: Vec<(i32, i32)> = std::env::var("QMM_SHAPES").map_or_else(
        |_| {
            vec![
                (12288, 2048),
                (2048, 4096),
                (8192, 2048),
                (512, 2048),
                (2048, 512),
                (32, 2048),
                (248_320, 2048),
            ]
        },
        |v| {
            v.split(',')
                .map(|s| {
                    let (n, k) = s.split_once(':').unwrap();
                    (n.parse().unwrap(), k.parse().unwrap())
                })
                .collect()
        },
    );
    let ms: Vec<i32> = std::env::var("QMM_M").map_or(vec![1, 3, 8], |v| {
        v.split(',').map(|s| s.parse().unwrap()).collect()
    });
    for &(n, k) in &shapes {
        let w = random::normal::<f32>(&[n, k], None, None, None)
            .unwrap()
            .multiply(Array::from_f32(0.02))
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let (wq, s, b) = ops::quantize(&w, 64, 4).unwrap();
        eval([&wq, &s, &b]).unwrap();
        let bytes = (n as f64) * (k as f64) * (0.5 + 4.0 / 64.0);
        let (nf, cs, sl) = probe::plan(n, k);
        let mut line = format!(
            "N={n} K={k} [nf{nf} cs{cs} s{sl} floor {:.3} ms]:",
            bytes / 105e9 * 1e3
        );
        for &m in &ms {
            let x = random::normal::<f32>(&[m, k], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            eval([&x]).unwrap();
            let want = ops::quantized_matmul(&x, &wq, &s, &b, true, 64, 4).unwrap();
            let got = probe::qmm8(&x, &wq, &s, &b).unwrap();
            if let Err(e) = got.eval() {
                panic!("qmm8 eval: {e}");
            }
            let (wf, gf) = (
                want.as_dtype(Dtype::Float32).unwrap(),
                got.as_dtype(Dtype::Float32).unwrap(),
            );
            let err: f32 = ops::abs(&wf.subtract(&gf).unwrap())
                .unwrap()
                .max(None)
                .unwrap()
                .item();
            let mag: f32 = ops::abs(&wf).unwrap().max(None).unwrap().item();
            let tm = time(|| ops::quantized_matmul(&x, &wq, &s, &b, true, 64, 4).unwrap());
            let tq = time(|| probe::qmm8(&x, &wq, &s, &b).unwrap());
            line += &format!(
                " | M{m} mlx={tm:.3} qmm8={tq:.3} ({:.0} GB/s) err={:.1e}",
                bytes / (tq * 1e-3) / 1e9,
                err / mag
            );
        }
        println!("{line} ms");
    }
}
