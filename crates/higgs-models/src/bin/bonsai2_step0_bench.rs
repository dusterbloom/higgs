//! Step 0 decomposition bench for `prism_hadamard_qwen35` (Bonsai-2-27B) decode.
//!
//! Server-measured full decode step is ~200 ms/token (5.0 tok/s, M4 base,
//! model solo-resident). Per-op sync micro-benches overstate costs because
//! async dispatch pipelines; this bench instead submits whole-token op graphs
//! at the checkpoint's exact shapes and evaluates once per iteration:
//!   - the full per-token packed qmm graph (stock affine bits=2)
//!   - rotation-only graphs for the current wiring (~354 rotations/token,
//!     one per packed module) and the proposed shared wiring (~194/token,
//!     one per distinct input vector)
//!
//! current ≈ qmm + rot354, shared ≈ qmm + rot194. The difference against the
//! server's measured step time bounds the GDN + attention + glue residual.
//! Data is LCG-random (timing is data-blind).

use half::f16;
use higgs_models::quant_mode::quantized_matmul;
use higgs_models::quant_mode::QuantMode;
use mlx_rs::Array;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

fn f16_vec(n: usize, st: &mut u64, scale: f32) -> Vec<f16> {
    (0..n)
        .map(|_| f16::from_f32((lcg(st) as f32 / u32::MAX as f32).mul_add(scale, -scale / 2.0)))
        .collect()
}

/// The `hadamard_rotate` sequence from qwen3_next.rs.
fn hadamard_rotate(x: &Array, block: i32, signs: &Array) -> Result<Array, mlx_rs::error::Exception> {
    let shape = x.shape().to_vec();
    let y = x.multiply(&signs.as_dtype(x.dtype())?)?;
    y.reshape(&[-1, block])?
        .hadamard_transform(None)?
        .reshape(&shape)
}

/// One checkpoint weight site (packed affine bits=2 g128).
struct Site {
    label: &'static str,
    rows: usize,
    k_words: usize, // K / 16
    calls_per_token: usize,
}

const SITES: [Site; 9] = [
    Site { label: "qkvz-fused", rows: 16_384, k_words: 320, calls_per_token: 48 },
    Site { label: "linear-out", rows: 5_120, k_words: 384, calls_per_token: 48 },
    Site { label: "gate", rows: 17_408, k_words: 320, calls_per_token: 64 },
    Site { label: "up", rows: 17_408, k_words: 320, calls_per_token: 64 },
    Site { label: "down", rows: 5_120, k_words: 1_088, calls_per_token: 64 },
    Site { label: "q", rows: 12_288, k_words: 320, calls_per_token: 16 },
    Site { label: "k", rows: 1_024, k_words: 320, calls_per_token: 16 },
    Site { label: "v", rows: 1_024, k_words: 320, calls_per_token: 16 },
    Site { label: "o", rows: 5_120, k_words: 384, calls_per_token: 16 },
];

fn time_fn<F: FnMut() -> Result<Array, mlx_rs::error::Exception>>(
    label: &str,
    mut f: F,
) -> f64 {
    let _ = f().expect("warmup").eval().expect("warmup eval");
    let iters = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = f().expect("run").eval().expect("run eval");
    }
    let ms = t0.elapsed().as_micros() as f64 / iters as f64 / 1000.0;
    println!("GRAPH {label}: {ms:.2}ms");
    ms
}

fn qmm_token_graph(st: &mut u64) -> Result<Array, mlx_rs::error::Exception> {
    let mut last = None;
    for site in SITES.iter() {
        let n_groups = site.k_words * 16 / 128;
        let w = Array::from_slice(
            &(0..site.rows * site.k_words).map(|_| lcg(st)).collect::<Vec<u32>>(),
            &[site.rows as i32, site.k_words as i32],
        );
        let scales = Array::from_slice(
            &f16_vec(site.rows * n_groups, st, 0.1),
            &[site.rows as i32, n_groups as i32],
        );
        let biases = Array::from_slice(
            &f16_vec(site.rows * n_groups, st, 0.1),
            &[site.rows as i32, n_groups as i32],
        );
        let width = site.k_words * 16;
        let x = Array::from_slice(&f16_vec(width, st, 2.0), &[1, width as i32]);
        for _ in 0..site.calls_per_token {
            last = Some(quantized_matmul(
                &x, &w, &scales, Some(&biases), true, 128, 2, QuantMode::Affine,
            )?);
        }
    }
    last.ok_or_else(|| mlx_rs::error::Exception::custom("no sites"))
}

fn rotations_graph(count: usize, width: usize, st: &mut u64) -> Result<Array, mlx_rs::error::Exception> {
    let x = Array::from_slice(&f16_vec(width, st, 2.0), &[1, width as i32]);
    let signs = Array::from_slice(&f16_vec(width, st, 0.2), &[width as i32]);
    let mut last = x;
    for _ in 0..count {
        last = hadamard_rotate(&last, 1024, &signs)?;
    }
    Ok(last)
}

fn main() {

    let _exec = higgs_models::mlx_exec::acquire();
    let mut st = 0x1234_5678_u64;

    if std::env::var("PROBE").is_ok() {
        fusion_probe(&mut st);
        return;
    }

    let qmm = time_fn("qmm token graph (353 calls)", || qmm_token_graph(&mut st));
    let rot354 = time_fn("rotations, current wiring (354x [1,5120])", || {
        rotations_graph(354, 5120, &mut st)
    });
    let rot194 = time_fn("rotations, shared wiring (194x [1,5120])", || {
        rotations_graph(194, 5120, &mut st)
    });

    let current = qmm + rot354;
    let shared = qmm + rot194;
    println!(
        "SUMMARY server-step=200ms/token | current≈{current:.1}ms ({:.1} tok/s) | shared≈{shared:.1}ms ({:.1} tok/s) | rot tax {:.1}ms -> saves {:.1}ms",
        1000.0 / current,
        1000.0 / shared,
        rot354,
        rot354 - rot194
    );
    println!(
        "SUMMARY residual vs server (GDN+attn+glue+lm_head+sampling): {:.0}ms/token",
        (200.0 - current).max(0.0)
    );
}

// Phase B probe: fused gate+up ([34816,5120] one call) vs two [17408,5120] calls,
// and fused qkv for FA ([14336,5120]) vs three calls. Same graph methodology.
fn fusion_probe(st: &mut u64) {
    use std::time::Instant;
    let mk = |rows: usize, st: &mut u64| {
        let n_groups = 5120 / 128;
        let w = Array::from_slice(
            &(0..rows * 320).map(|_| lcg(st)).collect::<Vec<u32>>(),
            &[rows as i32, 320],
        );
        let scales = Array::from_slice(&f16_vec(rows * n_groups, st, 0.1), &[rows as i32, n_groups as i32]);
        let biases = Array::from_slice(&f16_vec(rows * n_groups, st, 0.1), &[rows as i32, n_groups as i32]);
        (w, scales, biases)
    };
    let x = Array::from_slice(&f16_vec(5120, st, 2.0), &[1, 5120]);
    let (wg, sg, bg) = mk(17_408, st);
    let (wu, su, bu) = mk(17_408, st);
    let (wf, sf, bf) = mk(34_816, st);
    let run = |f: &mut dyn FnMut() -> Result<Array, mlx_rs::error::Exception>| {
        let _ = f().unwrap().eval().unwrap();
        let t0 = Instant::now();
        for _ in 0..30 { let _ = f().unwrap().eval().unwrap(); }
        t0.elapsed().as_micros() as f64 / 30.0 / 1000.0
    };
    let two = run(&mut || {
        let a = quantized_matmul(&x, &wg, &sg, Some(&bg), true, 128, 2, QuantMode::Affine)?;
        let b = quantized_matmul(&x, &wu, &su, Some(&bu), true, 128, 2, QuantMode::Affine)?;
        mlx_rs::transforms::eval([&a, &b])?;
        Ok(b)
    });
    let fused = run(&mut || quantized_matmul(&x, &wf, &sf, Some(&bf), true, 128, 2, QuantMode::Affine));
    println!("PROBE gate_up: two-calls={two:.3}ms fused={fused:.3}ms ratio={:.2}x", two / fused);

    let (wq, sq, bq) = mk(12_288, st);
    let (wk, sk, bk) = mk(1_024, st);
    let (wv, sv, bv) = mk(1_024, st);
    let (wqkv, sqkv, bqkv) = mk(14_336, st);
    let three = run(&mut || {
        let a = quantized_matmul(&x, &wq, &sq, Some(&bq), true, 128, 2, QuantMode::Affine)?;
        let b = quantized_matmul(&x, &wk, &sk, Some(&bk), true, 128, 2, QuantMode::Affine)?;
        let c = quantized_matmul(&x, &wv, &sv, Some(&bv), true, 128, 2, QuantMode::Affine)?;
        mlx_rs::transforms::eval([&a, &b, &c])?;
        Ok(c)
    });
    let fused3 = run(&mut || quantized_matmul(&x, &wqkv, &sqkv, Some(&bqkv), true, 128, 2, QuantMode::Affine));
    println!("PROBE fa_qkv: three-calls={three:.3}ms fused={fused3:.3}ms ratio={:.2}x", three / fused3);
}
