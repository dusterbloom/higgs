//! Ceiling check: time MLX SDPA at Escha prefill shapes.
//! Q [1,16,L,256] f32, K/V [1,2,S,256] f32, causal. L=S in {2048, 8192}.
#![allow(clippy::unwrap_used, clippy::print_stdout)]

use mlx_rs::{Array, Dtype, fast, transforms::eval};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_sdpa(l: i32, trials: usize) {
    let s = l;
    let q = Array::ones::<f32>(&[1, 16, l, 256]).unwrap();
    let k = Array::ones::<f32>(&[1, 2, s, 256]).unwrap();
    let v = Array::ones::<f32>(&[1, 2, s, 256]).unwrap();
    let scale = 1.0f32 / 256.0f32.sqrt();
    // Warmup + verify.
    let out = fast::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        scale,
        Some(fast::ScaledDotProductAttentionMask::Causal),
        None::<&Array>,
    )
    .unwrap();
    eval([&out]).unwrap();
    let mut ms = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = std::time::Instant::now();
        let o = fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            Some(fast::ScaledDotProductAttentionMask::Causal),
            None::<&Array>,
        )
        .unwrap();
        eval([&o]).unwrap();
        ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    // Score bytes materialized by unfused path: [1,16,L,S] fp32.
    let score_gb = 16.0 * (l as f64) * (s as f64) * 4.0 / 1e9;
    let med = median(ms);
    println!(
        "L=S={l}: SDPA median {med:.1} ms over {trials} trials (unfused scores ~{score_gb:.2} GB/layer)"
    );
}

fn main() {
    println!("Dtype check: {:?}", Dtype::Float32);
    bench_sdpa(2048, 5);
    bench_sdpa(8192, 3);
}
