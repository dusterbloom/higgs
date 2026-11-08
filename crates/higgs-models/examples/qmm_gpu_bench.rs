//! Enqueue many independent 8-row projections of one shape (for a Metal
//! System Trace: GPU busy per command buffer / ops per buffer = GPU time per
//! dispatch including launch gaps). `QMM_SHAPE=N:K`, `QMM_IMPL=qmm8|mlx`,
//! `QMM_OPS` ops per eval (default 400), `QMM_EVALS` (default 5).
#![allow(clippy::unwrap_used, clippy::print_stdout, clippy::cast_precision_loss)]

use std::time::Instant;

use higgs_models::qmm8::probe;
use mlx_rs::{Array, Dtype, ops, random, transforms::eval};

fn main() {
    let (n, k) = std::env::var("QMM_SHAPE").map_or((512, 2048), |v| {
        let (a, b) = v.split_once(':').unwrap();
        (a.parse().unwrap(), b.parse().unwrap())
    });
    let impl_ = std::env::var("QMM_IMPL").unwrap_or_else(|_| "qmm8".into());
    let ops_per: usize = std::env::var("QMM_OPS").map_or(400, |v| v.parse().unwrap());
    let evals: usize = std::env::var("QMM_EVALS").map_or(5, |v| v.parse().unwrap());
    let w = random::normal::<f32>(&[n, k], None, None, None).unwrap().as_dtype(Dtype::Bfloat16).unwrap();
    let (wq, s, b) = ops::quantize(&w, 64, 4).unwrap();
    let x = random::normal::<f32>(&[8, k], None, None, None).unwrap().as_dtype(Dtype::Bfloat16).unwrap();
    eval([&wq, &s, &b, &x]).unwrap();
    let f = |x: &Array| -> Array {
        if impl_ == "mlx" { ops::quantized_matmul(x, &wq, &s, &b, true, 64, 4).unwrap() } else { probe::qmm8(x, &wq, &s, &b).unwrap() }
    };
    for _ in 0..2 { let ys: Vec<Array> = (0..20).map(|_| f(&x)).collect(); eval(ys.iter()).unwrap(); }
    std::thread::sleep(std::time::Duration::from_millis(1500));
    let t = Instant::now();
    for _ in 0..evals {
        let ys: Vec<Array> = (0..ops_per).map(|_| f(&x)).collect();
        eval(ys.iter()).unwrap();
    }
    let total = t.elapsed().as_secs_f64();
    println!("{impl_} N={n} K={k}: {} ops in {total:.3} s = {:.1} us/op wall", ops_per * evals, total * 1e6 / (ops_per * evals) as f64);
}
