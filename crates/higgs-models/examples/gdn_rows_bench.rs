//! Time the fast-rows GDN kernels at Clover's layer shapes (Hk 16, Hv 32,
//! D 128, K 4): verify prep, the tape recurrence, the fold, and the preview
//! mixer, each amortised over back-to-back launches.
#![allow(clippy::unwrap_used, clippy::print_stdout)]

use std::time::Instant;

use higgs_models::gdn_rows::{PreviewInputs, probe};
use mlx_rs::{Array, Dtype, random, transforms::eval};

fn time(f: impl Fn() -> Vec<Array>) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..20 {
        let t = Instant::now();
        let ys: Vec<Array> = (0..10).flat_map(|_| f()).collect();
        eval(ys.iter()).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / 10.0);
    }
    best
}

fn normal(shape: &[i32], dtype: Dtype) -> Array {
    let a = random::normal::<f32>(shape, None, None, None)
        .unwrap()
        .as_dtype(dtype)
        .unwrap();
    eval([&a]).unwrap();
    a
}

fn main() {
    let (hk, hv, d, k) = (16, 32, 128, 4);
    let c = (2 * hk + hv) * d;
    let bf = Dtype::Bfloat16;
    let f32_ = Dtype::Float32;
    let state = normal(&[1, hv, d, d], f32_);
    let conv_w = normal(&[k, c], bf);
    let conv_r = normal(&[c, 1, k - 1], bf);
    let per_head = normal(&[hv], f32_);
    let norm = normal(&[d], bf);
    for rows in [4, 8] {
        let x = normal(&[1, rows, c], bf);
        let hist = normal(&[1, k - 1, c], bf);
        let a = normal(&[1, rows, hv], bf);
        let b = normal(&[1, rows, hv], bf);
        let g = probe::geometry(&x, &hist, &state, &conv_w, 8).unwrap();
        let prep = time(|| {
            let (q, k, v, w) = probe::prep_verify(&x, &hist, &conv_w, &norm, &norm, g).unwrap();
            vec![q, k, v, w]
        });
        let (q, kk, v, _) = probe::prep_verify(&x, &hist, &conv_w, &norm, &norm, g).unwrap();
        eval([&q, &kk, &v]).unwrap();
        let tape = time(|| {
            let (y, s, t) =
                probe::gated_delta_with_tape(&q, &kk, &v, &per_head, &a, &per_head, &b, &state, g)
                    .unwrap();
            vec![y, s, t]
        });
        let (_, _, delta) =
            probe::gated_delta_with_tape(&q, &kk, &v, &per_head, &a, &per_head, &b, &state, g)
                .unwrap();
        eval([&delta]).unwrap();
        let fold = time(|| {
            let (s, w) = probe::fold(
                &state, &kk, &a, &per_head, &per_head, &delta, &hist, &x, rows / 2, g,
            )
            .unwrap();
            vec![s, w]
        });
        let mixer = PreviewInputs {
            rows: &x,
            a: &a,
            history: &hist,
            state: &state,
            conv_w: &conv_w,
            conv_r: &conv_r,
            norm_q: &norm,
            norm_k: &norm,
            a_log: &per_head,
            dt_bias: &per_head,
            scale: &per_head,
            hist_bias: &per_head,
            hist_c: &per_head,
        };
        let preview = time(|| vec![probe::preview_mixer(&mixer).unwrap()]);
        let preview_mlx = time(|| vec![probe::preview_mixer_mlx(&mixer).unwrap()]);
        println!(
            "rows={rows}: prep_verify {prep:.3} ms, tape recurrence {tape:.3} ms, fold {fold:.3} ms, \
             preview mixer {preview:.3} ms (mlx chain {preview_mlx:.3} ms)"
        );
    }
}
