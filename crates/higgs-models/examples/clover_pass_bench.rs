//! Time Clover's multi-row passes on real weights: an all-mask block preview
//! (read-only) and a causal block forward (what a commit / verify costs).
//!
//! `cargo run --release -p higgs-models --example clover_pass_bench -- MODEL_DIR [ROWS] [ITERS]`
#![allow(clippy::unwrap_used, clippy::print_stdout, clippy::cast_precision_loss)]

use std::time::Instant;

use higgs_models::qwen3_next::{load_qwen3_5_moe_model, with_fast_rows};
use mlx_rs::{Array, transforms::eval};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Prints `tracing` events (the `HIGGS_DFLASH_LAYER_*_TIMING` breakdowns).
struct Stderr;
impl tracing::Subscriber for Stderr {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct V;
        impl tracing::field::Visit for V {
            fn record_debug(&mut self, _: &tracing::field::Field, v: &dyn std::fmt::Debug) {
                eprintln!("{v:?}");
            }
        }
        event.record(&mut V);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn main() {
    if std::env::var("BENCH_TRACE").is_ok() {
        tracing::subscriber::set_global_default(Stderr).unwrap();
    }
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let rows: i32 = args.get(2).map_or(8, |s| s.parse().unwrap());
    let iters: usize = args.get(3).map_or(20, |s| s.parse().unwrap());
    const MASK: u32 = 151_669;

    let mut model = load_qwen3_5_moe_model(dir).unwrap();
    let mut cache = model.make_cache();
    // ponytail: synthetic 64-token prompt; pass cost barely depends on content
    let prompt: Vec<u32> = (0..64u32).map(|i| 1000 + i * 37).collect();
    let p = Array::from_slice(&prompt, &[1, 64]);
    let logits = model.forward(&p, None, &mut cache).unwrap();
    eval([&logits]).unwrap();

    let block = Array::from_slice(&vec![MASK; rows as usize], &[1, rows]);
    let mut preview = Vec::new();
    let mut preview_build = Vec::new();
    for i in 0..iters + 3 {
        let t = Instant::now();
        let l = model.forward_preview_block(&block, &cache).unwrap();
        let tb = t.elapsed().as_secs_f64() * 1e3;
        eval([&l]).unwrap();
        if i >= 3 {
            preview.push(t.elapsed().as_secs_f64() * 1e3);
            preview_build.push(tb);
        }
    }
    println!("rows={rows} preview_build_ms={:.1} (graph construction before eval)", median(preview_build));

    let toks: Vec<u32> = (0..rows as u32).map(|i| 2000 + i).collect();
    let causal_in = Array::from_slice(&toks, &[1, rows]);
    let mut causal = Vec::new();
    for i in 0..iters.min(8) + 2 {
        let t = Instant::now();
        let fast = std::env::var("BENCH_FAST_COMMIT").is_ok();
        let l = if fast {
            with_fast_rows(|| model.forward(&causal_in, None, &mut cache)).unwrap()
        } else {
            model.forward(&causal_in, None, &mut cache).unwrap()
        };
        eval([&l]).unwrap();
        if i >= 2 {
            causal.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }

    // BENCH_VERIFY=1: the self-speculative verify (tape-recording forward)
    // and a half-accept rollback, timed separately.
    if std::env::var("BENCH_VERIFY").is_ok() {
        use higgs_models::qwen3_next::{DFlashRowSchedule, LayerCache};
        let eval_cache = |cache: &Vec<Option<LayerCache>>| {
            let mut targets = Vec::new();
            for layer in cache.iter().flatten() {
                if let LayerCache::Arrays(a) = layer {
                    targets.extend(a.ssm_state.iter());
                    targets.extend(a.conv_state.iter());
                }
            }
            eval(targets).unwrap();
        };
        let (mut verify, mut rollback, mut verify_build) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..iters.min(8) + 2 {
            let t = Instant::now();
            let (l, _, tapes) = with_fast_rows(|| {
                model.forward_with_taps_tape_scheduled(
                    &causal_in,
                    None,
                    &mut cache,
                    &[],
                    None,
                    DFlashRowSchedule::NativeBatch,
                )
            })
            .unwrap();
            let tb = t.elapsed().as_secs_f64() * 1e3;
            eval([&l]).unwrap();
            eval_cache(&cache);
            let t_verify = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            model
                .replay_tape_rollback(&tapes, &mut cache, rows / 2, rows - rows / 2)
                .unwrap();
            eval_cache(&cache);
            if i >= 2 {
                verify.push(t_verify);
                verify_build.push(tb);
                rollback.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        println!(
            "rows={rows} verify_ms={:.1} rollback_ms={:.1} verify_build_ms={:.1}",
            median(verify),
            median(rollback),
            median(verify_build)
        );
    }

    let one = Array::from_slice(&[3000u32], &[1, 1]);
    let mut ar = Vec::new();
    for i in 0..iters + 3 {
        let t = Instant::now();
        let l = model.forward(&one, None, &mut cache).unwrap();
        eval([&l]).unwrap();
        if i >= 3 {
            ar.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    println!(
        "rows={rows} preview_ms={:.1} causal_ms={:.1} ar_step_ms={:.1}",
        median(preview),
        median(causal),
        median(ar)
    );
}
