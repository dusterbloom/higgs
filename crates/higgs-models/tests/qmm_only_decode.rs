//! Stripped decode: only the 7 qmm per layer + lm_head, no rope/norm/sdpa/cache/residual.
//!
//! Run:
//!   cargo test --release -p higgs-models --test qmm_only_decode \
//!       qmm_only_decode_bench -- --ignored --nocapture
//!
//! If this hits ~14 ms/step on Bonsai-8B, the binding is fine and the gap
//! lives in our scaffolding (rope, cache, norms). If it's ≥30 ms/step, the
//! binding itself is serializing graph nodes.

use std::path::PathBuf;

use mlx_rs::{ops, random, transforms::eval, Array, Dtype};

use higgs_models::bonsai_q1::{BonsaiQ1Engine, BonsaiQ1Gpu};

const WARMUP: usize = 4;
const STEPS: usize = 64;

fn dir(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/lm-studio/models/prism-ml")
        .join(name);
    p.join("config.json").exists().then_some(p)
}

fn fresh_fp16(shape: &[i32]) -> Array {
    random::normal::<f32>(shape, None, None, None)
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap()
}

fn run(label: &str, gpu: &BonsaiQ1Gpu) {
    let hidden = gpu.config.hidden as i32;
    let inter = gpu.config.inter as i32;

    let do_step = |x_h: &Array, x_i: &Array| -> Array {
        let mut last = x_h.clone();
        for layer in &gpu.layers {
            let _ = layer.q_proj.forward(x_h).unwrap();
            let _ = layer.k_proj.forward(x_h).unwrap();
            let _ = layer.v_proj.forward(x_h).unwrap();
            let o = layer.o_proj.forward(x_h).unwrap();
            let _ = layer.gate_proj.forward(x_h).unwrap();
            let _ = layer.up_proj.forward(x_h).unwrap();
            let d = layer.down_proj.forward(x_i).unwrap();
            // chain something so the optimizer can't dead-strip
            last = ops::add(&o, &d).unwrap();
        }
        let logits_in = if gpu.lm_head.is_some() {
            &last
        } else {
            &last
        };
        match &gpu.lm_head {
            Some(lm) => lm.forward(logits_in).unwrap(),
            None => gpu.embed.forward(logits_in).unwrap(),
        }
    };

    // Warmup
    for _ in 0..WARMUP {
        let xh = fresh_fp16(&[1, 1, hidden]);
        let xi = fresh_fp16(&[1, 1, inter]);
        let y = do_step(&xh, &xi);
        eval([&y]).unwrap();
    }

    // Measured
    let t0 = std::time::Instant::now();
    for _ in 0..STEPS {
        let xh = fresh_fp16(&[1, 1, hidden]);
        let xi = fresh_fp16(&[1, 1, inter]);
        let y = do_step(&xh, &xi);
        eval([&y]).unwrap();
    }
    let elapsed = t0.elapsed();
    let ms = elapsed.as_secs_f64() * 1e3 / STEPS as f64;
    println!(
        "{label}: {STEPS} steps in {:.1} ms → {:.3} ms/step ({:.1} tok/s)",
        elapsed.as_secs_f64() * 1e3,
        ms,
        1000.0 / ms
    );
}

#[test]
#[ignore]
fn qmm_only_decode_bench() {
    println!();
    println!("qmm-only stripped-decode bench (no rope/norm/sdpa/cache/residual)");
    println!("warmup={WARMUP} steps={STEPS}");
    println!();

    if let Some(d) = dir("Bonsai-1.7B-mlx-1bit") {
        let gpu = BonsaiQ1Engine::load(&d).unwrap().to_gpu().unwrap();
        run("Bonsai-1.7B", &gpu);
    } else {
        eprintln!("Bonsai-1.7B not found, skipping");
    }

    if let Some(d) = dir("Bonsai-8B-mlx-1bit") {
        let gpu = BonsaiQ1Engine::load(&d).unwrap().to_gpu().unwrap();
        run("Bonsai-8B", &gpu);
    } else {
        eprintln!("Bonsai-8B not found, skipping");
    }
}
