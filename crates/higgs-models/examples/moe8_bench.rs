//! Per-layer time of the weight-once `moe8` passes vs the generic switch-MLP
//! gather (`gather_qmv`) at Clover's expert shapes, for 8 rows x top-8 routed
//! to U distinct experts; against the byte floor at 120 GB/s.
//!
//! `cargo run --release -p higgs-models --example moe8_bench -- [ITERS]`
#![allow(clippy::unwrap_used, clippy::print_stdout, clippy::cast_precision_loss)]

use std::time::Instant;

use higgs_models::moe8::probe;
use higgs_models::qwen3_next::switch_mlp_gather_probe::SwitchMlpGather;
use mlx_rs::{Array, Dtype, ops, random, transforms::eval};

fn quantized(shape: &[i32], seed: u64) -> (Array, Array, Array) {
    let key = random::key(seed).unwrap();
    let w = random::uniform::<f32, f32>(-0.05, 0.05, shape, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let q = ops::quantize(&w, 64, 4).unwrap();
    eval([&q.0, &q.1, &q.2]).unwrap();
    q
}

fn median_us(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let iters: usize = std::env::args().nth(1).map_or(30, |s| s.parse().unwrap());
    let (e, d, i, tk, rows) = (256, 2048, 512, 8, 8);
    let gate = quantized(&[e, i, d], 1);
    let up = quantized(&[e, i, d], 2);
    let down = quantized(&[e, d, i], 3);
    let block = SwitchMlpGather::new(64, 4, [gate.clone(), up.clone(), down.clone()]).unwrap();
    let x = random::normal::<f32>(&[1, rows, d], None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let x2 = x.reshape(&[rows, d]).unwrap();
    eval([&x2]).unwrap();
    let bytes_per_expert = (2.0 * (i * d / 2 + 2 * i * d / 64 * 2) as f64) + (d * i / 2 + 2 * d * i / 64 * 2) as f64;
    println!("bytes/expert = {:.2} MB", bytes_per_expert / 1e6);
    let us: Vec<u32> = std::env::var("MOE8_U").map_or(vec![8, 16, 24, 32, 48, 64], |v| v.split(',').map(|s| s.parse().unwrap()).collect());
    for u in us {
        // (t, j) -> (8 t + j) mod U: U distinct experts, distinct within a row.
        let routing: Vec<u32> = (0..rows as u32 * tk as u32).map(|p| p % u).collect();
        let idx = Array::from_slice(&routing, &[1, rows, tk]);
        let idx2 = idx.reshape(&[rows, tk]).unwrap();
        eval([&idx2]).unwrap();
        let mut t_moe8 = Vec::new();
        let mut t_qmv = Vec::new();
        let layers: usize = std::env::var("MOE8_LAYERS").map_or(40, |v| v.parse().unwrap());
        for it in 0..iters + 5 {
            let t = Instant::now();
            let ys: Vec<Array> = (0..layers).map(|_| probe::moe8(&x2, &idx2, (&gate.0, &gate.1, &gate.2), (&up.0, &up.1, &up.2), (&down.0, &down.1, &down.2)).unwrap()).collect();
            eval(ys.iter()).unwrap();
            if it >= 5 {
                t_moe8.push(t.elapsed().as_secs_f64() * 1e6 / layers as f64);
            }
            let t = Instant::now();
            let ys: Vec<Array> = (0..layers).map(|_| block.forward(&x, &idx).unwrap()).collect();
            eval(ys.iter()).unwrap();
            if it >= 5 {
                t_qmv.push(t.elapsed().as_secs_f64() * 1e6 / layers as f64);
            }
        }
        let floor_us = u as f64 * bytes_per_expert / 120e9 * 1e6;
        let (m8, mq) = (median_us(t_moe8), median_us(t_qmv));
        println!(
            "U={u:2}  moe8 {m8:7.1} us ({:5.1} ms/40L, {:3.0} GB/s)  qmv {mq:7.1} us ({:5.1} ms/40L)  floor {floor_us:6.1} us",
            m8 * 40.0 / 1e3,
            u as f64 * bytes_per_expert / (m8 * 1e-6) / 1e9,
            mq * 40.0 / 1e3
        );
    }
}
