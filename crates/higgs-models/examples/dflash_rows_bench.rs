//! Per-kernel cost of the fused DFlash2 draft at the real geometry (hidden
//! 2048, 32/8 heads, 6 layers, intermediate 6144, 8 taps): each step is run
//! `REPS` times on independent random inputs inside one eval, so the number
//! is the GPU time per dispatch including its launch ramp (no per-eval
//! overhead). Context length is an argument (default 300).
//!
//! `cargo run --release -p higgs-models --example dflash_rows_bench -- [CTX] [ITERS]`
#![allow(clippy::unwrap_used, clippy::print_stdout, clippy::cast_precision_loss)]

use std::time::Instant;

use higgs_models::dflash_rows::probe::{self, Heads};
use higgs_models::qmm8::probe as qmm8;
use mlx_rs::{Array, Dtype, ops, random, transforms::eval};

const H: i32 = 2048;
const HQ: i32 = 32;
const HKV: i32 = 8;
const T: i32 = 8;
const INTER: i32 = 6144;
const NL: i32 = 6;
const TAPS: i32 = 8;
const REPS: usize = 12;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bf16(shape: &[i32], scale: f32) -> Array {
    random::normal::<f32>(shape, None, Some(scale), None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

fn quant(n: i32, k: i32) -> (Array, Array, Array) {
    ops::quantize(&bf16(&[n, k], 0.04), 64, 4).unwrap()
}

/// Median per-op GPU time over `iters` evals of `REPS` independent ops.
fn per_op<F: Fn(usize) -> Array>(iters: usize, f: F) -> f64 {
    let mut ms = Vec::new();
    for i in 0..iters + 2 {
        let t = Instant::now();
        let outs: Vec<Array> = (0..REPS).map(&f).collect();
        eval(outs.iter()).unwrap();
        if i >= 2 {
            ms.push(t.elapsed().as_secs_f64() * 1e3 / REPS as f64);
        }
    }
    median(ms)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ctx: i32 = args.get(1).map_or(300, |s| s.parse().unwrap());
    let iters: usize = args.get(2).map_or(10, |s| s.parse().unwrap());
    let heads = Heads {
        hq: HQ,
        hkv: HKV,
        window: 2048,
        eps: 1e-6,
        rope_base: 1e7,
    };
    let xs: Vec<Array> = (0..REPS).map(|_| bf16(&[T, H], 1.0)).collect();
    let xs3: Vec<Array> = (0..REPS).map(|_| bf16(&[3, H], 1.0)).collect();
    let taps: Vec<Array> = (0..REPS).map(|_| bf16(&[3, TAPS * H], 1.0)).collect();
    let w = bf16(&[H], 1.0);
    let base = bf16(&[2, 2, H], 0.3);
    let d = bf16(&[T, 4 * H / 16], 0.1);
    let (kw, ks, kb) = quant(4 * H / 16, H);
    let (qw, qs, qb) = quant(HQ * 128, H);
    let (kkw, kks, kkb) = quant(HKV * 128, H);
    let (vw, vs, vb) = quant(HKV * 128, H);
    let (ow, os, ob) = quant(H, HQ * 128);
    let (gw, gs, gb) = quant(INTER, H);
    let (uw, us, ub) = quant(INTER, H);
    let (dw, ds, db) = quant(H, INTER);
    let (fw, fs, fb) = quant(H, TAPS * H);
    let (skw, sks, skb) = quant(NL * 2 * HKV * 128, H);
    let (hpw, hps, hpb) = quant(256, H);
    let q: Vec<Array> = (0..REPS).map(|_| bf16(&[T, HQ * 128], 1.0)).collect();
    let kn: Vec<Array> = (0..REPS).map(|_| bf16(&[T, HKV * 128], 1.0)).collect();
    let ck: Vec<Array> = (0..REPS).map(|_| bf16(&[1, HKV, ctx, 128], 1.0)).collect();
    let cv: Vec<Array> = (0..REPS).map(|_| bf16(&[1, HKV, ctx, 128], 1.0)).collect();
    let kv: Vec<Array> = (0..REPS)
        .map(|_| bf16(&[3, NL * 2 * HKV * 128], 1.0))
        .collect();
    let acts: Vec<Array> = (0..REPS).map(|_| bf16(&[T, INTER], 1.0)).collect();
    let o: Vec<Array> = (0..REPS).map(|_| bf16(&[T, HQ * 128], 1.0)).collect();
    let hf: Vec<Array> = (0..REPS)
        .map(|_| random::normal::<f32>(&[T, H], None, None, None).unwrap())
        .collect();
    let logits: Vec<Array> = (0..REPS).map(|_| bf16(&[7, 98_304], 3.0)).collect();
    let pos = Array::from_slice(&[ctx], &[1]);
    let w128 = bf16(&[128], 1.0);
    let mut all = Vec::new();
    for arr in xs
        .iter()
        .chain(&q)
        .chain(&kn)
        .chain(&ck)
        .chain(&cv)
        .chain(&kv)
        .chain(&acts)
        .chain(&o)
        .chain(&hf)
        .chain(&logits)
        .chain(&taps)
        .chain(&xs3)
    {
        all.push(arr);
    }
    eval(all).unwrap();

    let mut rows: Vec<(&str, f64, f64)> = Vec::new();
    let mut add = |name: &'static str, per_draft: f64, ms: f64| {
        println!(
            "{name:28} {ms:7.3} ms/op x {per_draft:4.1} per draft = {:6.3} ms",
            ms * per_draft
        );
        rows.push((name, per_draft, ms));
    };
    add(
        "norm_rows [8,2048]",
        1.0,
        per_op(iters, |i| probe::norm_rows(&xs[i], &w, 1e-6).unwrap()),
    );
    add(
        "conv_prep",
        12.0,
        per_op(iters, |i| probe::conv_prep(&xs[i], &d, &base).unwrap()),
    );
    add(
        "conv_fin_res_norm",
        12.0,
        per_op(iters, |i| {
            probe::conv_fin_res_norm(&hf[i], &xs[i], &d, &base, &w, 1e-6)
                .unwrap()
                .1
        }),
    );
    add(
        "qmm8 conv proj 512x2048",
        12.0,
        per_op(iters, |i| qmm8::qmm8(&xs[i], &kw, &ks, &kb).unwrap()),
    );
    add(
        "qmm8 q|k|v 6144x2048",
        6.0,
        per_op(iters, |i| {
            qmm8::qmm8_multi(
                &xs[i],
                &[(&qw, &qs, &qb), (&kkw, &kks, &kkb), (&vw, &vs, &vb)],
            )
            .unwrap()
            .remove(0)
        }),
    );
    add(
        "attention (ctx)",
        6.0,
        per_op(iters, |i| {
            probe::attention(
                &q[i], &kn[i], &kn[i], &w128, &w128, &ck[i], &cv[i], &pos, heads,
            )
            .unwrap()
        }),
    );
    add(
        "qmm8 o_proj 2048x4096",
        6.0,
        per_op(iters, |i| qmm8::qmm8(&o[i], &ow, &os, &ob).unwrap()),
    );
    add(
        "qmm8 gate|up+swiglu 2x6144",
        6.0,
        per_op(iters, |i| {
            qmm8::qmm8_swiglu(&xs[i], (&gw, &gs, &gb), (&uw, &us, &ub)).unwrap()
        }),
    );
    add(
        "qmm8 down 2048x6144",
        6.0,
        per_op(iters, |i| qmm8::qmm8(&acts[i], &dw, &ds, &db).unwrap()),
    );
    add(
        "qmm8 fc 2048x16384 [3 rows]",
        1.0,
        per_op(iters, |i| qmm8::qmm8(&taps[i], &fw, &fs, &fb).unwrap()),
    );
    add(
        "qmm8 stacked kv 12288x2048",
        1.0,
        per_op(iters, |i| qmm8::qmm8(&xs3[i], &skw, &sks, &skb).unwrap()),
    );
    add(
        "ctx_commit (ctx)",
        6.0,
        per_op(iters, |i| {
            probe::ctx_commit(&kv[i], &w128, &pos, &ck[i], &cv[i], 0, NL, heads)
                .unwrap()
                .0
        }),
    );
    add(
        "qmm8 hidden_proj 256x2048",
        1.0,
        per_op(iters, |i| qmm8::qmm8(&xs[i], &hpw, &hps, &hpb).unwrap()),
    );
    add(
        "top16+merge [7,98304]",
        1.0,
        per_op(iters, |i| probe::top16(&logits[i]).unwrap().unwrap().0),
    );
    let total: f64 = rows.iter().map(|(_, n, ms)| n * ms).sum();
    println!(
        "sum of per-op costs per draft (stage + selector, no head): {total:.2} ms at ctx {ctx}"
    );
}
