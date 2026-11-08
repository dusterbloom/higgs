//! CPU cost per lazy MLX op (graph construction) and per evaluated op, for a
//! native op (`add` on 8 x 2048 bf16), MLX's `quantized_matmul` and the
//! custom `qmm8` kernel at a tiny shape (N = 32): what a dispatch costs
//! before any bandwidth.
#![allow(clippy::unwrap_used, clippy::print_stdout, clippy::cast_precision_loss)]

use std::time::Instant;

use higgs_models::qmm8::probe;
use mlx_rs::{Array, Dtype, ops, random, transforms::eval};

fn main() {
    let (n, k, m) = (32, 2048, 8);
    let w = random::normal::<f32>(&[n, k], None, None, None).unwrap().as_dtype(Dtype::Bfloat16).unwrap();
    let (wq, s, b) = ops::quantize(&w, 64, 4).unwrap();
    let x = random::normal::<f32>(&[m, k], None, None, None).unwrap().as_dtype(Dtype::Bfloat16).unwrap();
    eval([&wq, &s, &b, &x]).unwrap();
    let chains: usize = std::env::args().nth(1).map_or(100, |v| v.parse().unwrap());
    let run = |name: &str, f: &dyn Fn(&Array) -> Array| {
        // chain `chains` ops so each depends on the previous (no fusion of independent work)
        for _ in 0..3 {
            let mut y = x.clone();
            for _ in 0..chains { y = f(&y); }
            eval([&y]).unwrap();
        }
        let mut best_build = f64::MAX;
        let mut best_total = f64::MAX;
        for _ in 0..10 {
            let t = Instant::now();
            let mut y = x.clone();
            for _ in 0..chains { y = f(&y); }
            let tb = t.elapsed().as_secs_f64() * 1e6 / chains as f64;
            eval([&y]).unwrap();
            let tt = t.elapsed().as_secs_f64() * 1e6 / chains as f64;
            best_build = best_build.min(tb);
            best_total = best_total.min(tt);
        }
        println!("{name:14} build {best_build:6.1} us/op   build+eval {best_total:6.1} us/op");
    };
    run("add", &|y| y.add(&x).unwrap());
    run("add+astype", &|y| y.add(&x).unwrap().as_dtype(Dtype::Bfloat16).unwrap());
    // the matmuls map [8, 2048] -> [8, 32]; pad back to [8, 2048] with a cheap native op
    let pad = |z: Array| ops::concatenate_axis(&[&z; 64], -1).unwrap();
    run("qmv(mlx)", &|y| pad(ops::quantized_matmul(y, &wq, &s, &b, true, 64, 4).unwrap()));
    run("qmm8(custom)", &|y| pad(probe::qmm8(y, &wq, &s, &b).unwrap()));
}
