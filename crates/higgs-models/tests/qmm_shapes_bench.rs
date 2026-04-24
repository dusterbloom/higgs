//! Isolated `quantized_matmul` kernel bench for Bonsai-Q1 decode shapes.
//!
//! Run:
//!   cargo test --release -p higgs-models --test qmm_shapes_bench \
//!       qmm_shapes_isolated -- --ignored --nocapture
//!
//! Why: end-to-end Bonsai-8B decode is 3.2x slower than mlx-lm Python on the
//! same kernels. This bench isolates qmm cost per shape so we can compare
//! apples-to-apples against `qmm_shapes_bench.py` (mlx 0.31.2.dev). If the
//! Rust numbers match Python at the same shape, the gap is in composition
//! (cache update, rope offset Array allocs, FFI shape queries). If Rust is
//! slower per call, the gap is in mlx-sys / MLX 0.31.1 vs 0.31.2.dev.

use mlx_rs::{ops, random, transforms::eval, Array, Dtype};

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    k: i32,
    m: i32,
}

const SHAPES: &[Shape] = &[
    // Bonsai-1.7B (hidden=2048, intermediate=6144, kv_dim=1024, vocab=151669)
    Shape { label: "1p7b/q_or_o",  k: 2048, m: 2048 },
    Shape { label: "1p7b/k_or_v",  k: 2048, m: 1024 },
    Shape { label: "1p7b/gate_up", k: 2048, m: 6144 },
    Shape { label: "1p7b/down",    k: 6144, m: 2048 },
    Shape { label: "1p7b/lm_head", k: 2048, m: 151669 },
    // Bonsai-8B (hidden=4096, intermediate=12288, kv_dim=1024, vocab=151669)
    Shape { label: "8b/q_or_o",    k: 4096,  m: 4096 },
    Shape { label: "8b/k_or_v",    k: 4096,  m: 1024 },
    Shape { label: "8b/gate_up",   k: 4096,  m: 12288 },
    Shape { label: "8b/down",      k: 12288, m: 4096 },
    Shape { label: "8b/lm_head",   k: 4096,  m: 151669 },
];

const GROUP_SIZE: i32 = 128;
const BITS: i32 = 1;
const WARMUP: usize = 50;
const ITERS: usize = 1000;

fn fresh_x(k: i32) -> Array {
    let x = random::normal::<f32>(&[1i32, 1, k], None, None, None).unwrap();
    x.as_dtype(Dtype::Float16).unwrap()
}

#[test]
#[ignore]
fn qmm_shapes_isolated() {
    println!();
    println!("qmm-isolated bench — group={GROUP_SIZE} bits={BITS}");
    println!("warmup={WARMUP} iters={ITERS} (release build, default device)");
    println!();
    println!(
        "{:<18} {:>6} {:>6} {:>14}",
        "shape", "K", "M", "ms/iter (sync)"
    );

    for shape in SHAPES {
        let w_full = random::normal::<f32>(&[shape.m, shape.k], None, None, None).unwrap();
        let w_full = w_full.as_dtype(Dtype::Float16).unwrap();
        let (qw, scales, biases) = ops::quantize(&w_full, GROUP_SIZE, BITS).unwrap();
        eval([&qw, &scales, &biases]).unwrap();

        let x = fresh_x(shape.k);
        eval([&x]).unwrap();

        for _ in 0..WARMUP {
            let y = ops::quantized_matmul(
                &x, &qw, &scales, Some(&biases), true, GROUP_SIZE, BITS,
            )
            .unwrap();
            eval([&y]).unwrap();
        }

        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            let y = ops::quantized_matmul(
                &x, &qw, &scales, Some(&biases), true, GROUP_SIZE, BITS,
            )
            .unwrap();
            eval([&y]).unwrap();
        }
        let ms_sync = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        println!(
            "{:<18} {:>6} {:>6} {:>14.4}",
            shape.label, shape.k, shape.m, ms_sync
        );
    }
}
