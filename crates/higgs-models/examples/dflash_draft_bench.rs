//! Time one DFlash2 draft step on real weights, by phase: context tiles +
//! trunk (`stage_forward`), the target head on the masked slots, the
//! selector's `argpartition`, and the whole selector; fast rows off/on.
//!
//! `cargo run --release -p higgs-models --example dflash_draft_bench -- TARGET_DIR DRAFTER_DIR [ITERS] [KEPT_ROWS]`
#![allow(clippy::unwrap_used, clippy::print_stdout, clippy::cast_precision_loss)]

use std::time::Instant;

use higgs_models::{
    AnyModel,
    dflash::load_dflash_drafter,
    qwen3_next::{load_qwen3_5_moe_model, with_fast_rows},
};
use mlx_rs::{Array, Dtype, ops, ops::indexing::IndexOp, transforms::eval};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time<F: FnMut() -> Array>(iters: usize, mut f: F) -> f64 {
    let mut ms = Vec::new();
    for i in 0..iters + 3 {
        let t = Instant::now();
        let out = f();
        eval([&out]).unwrap();
        if i >= 3 {
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    median(ms)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: usize = args.get(3).map_or(20, |s| s.parse().unwrap());
    let kept: i32 = args.get(4).map_or(4, |s| s.parse().unwrap());
    let model = AnyModel::Qwen3Next(load_qwen3_5_moe_model(&args[1]).unwrap());
    let mut drafter = load_dflash_drafter(std::path::Path::new(&args[2])).unwrap();
    let bs = drafter.config.block_size;
    let h = drafter.config.hidden_size;
    let n_taps = drafter.config.num_taps();
    let mask = drafter.config.mask_token_id();
    let rand = |rows: i32| {
        mlx_rs::random::normal::<f32>(&[1, rows, h], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    };

    // 64 primed context rows (two full tiles), then `kept` pending rows per cycle.
    let mut cache = drafter.make_cache();
    let primed: Vec<Array> = (0..n_taps).map(|_| rand(64)).collect();
    drafter.prime_taps(&primed, &mut cache).unwrap();
    let pending: Vec<Array> = (0..n_taps).map(|_| rand(kept)).collect();
    eval(pending.iter()).unwrap();
    let mut ids = vec![mask; bs as usize];
    ids[0] = 1000;
    let noise = model
        .embed_token_ids(&Array::from_slice(&ids, &[1, bs]))
        .unwrap();
    eval([&noise]).unwrap();

    for fast in [false, true] {
        let run = |f: &mut dyn FnMut() -> Array| {
            if fast {
                with_fast_rows(|| f())
            } else {
                f()
            }
        };
        let stage = time(iters, || {
            run(&mut || {
                drafter
                    .stage_forward(&noise, &pending, &cache)
                    .unwrap()
                    .hidden()
                    .clone()
            })
        });
        let hidden = drafter
            .stage_forward(&noise, &pending, &cache)
            .unwrap()
            .hidden()
            .index((.., 1.., ..));
        eval([&hidden]).unwrap();
        let head = time(iters, || {
            run(&mut || model.forward_all_logits_from_hidden(&hidden).unwrap())
        });
        let logits = model.forward_all_logits_from_hidden(&hidden).unwrap();
        eval([&logits]).unwrap();
        // The reduced draft head (first 98304 ids), as `DFlashSpec` runs it.
        let parts = model.lm_head_fast_rows_parts().unwrap();
        let (hw, hs, hb) = drafter.reduced_head(parts, 98_304).unwrap().clone();
        let hidden2 = hidden.reshape(&[-1, h]).unwrap();
        let head98k = time(iters, || {
            run(&mut || higgs_models::qmm8::probe::qmm8(&hidden2, &hw, &hs, &hb).unwrap())
        });
        let top_k = drafter.config.selector_top_k();
        let part = time(iters, || {
            ops::argpartition_axis(&logits.negative().unwrap(), top_k - 1, -1).unwrap()
        });
        let select = time(iters, || {
            run(&mut || {
                drafter
                    .propose_dflash2_tokens(&hidden, &logits, 1000)
                    .unwrap()
                    .unwrap()
            })
        });
        let all = time(iters, || {
            run(&mut || {
                let tx = drafter.stage_forward(&noise, &pending, &cache).unwrap();
                let hidden = tx.hidden().index((.., 1.., ..));
                let logits = model.forward_all_logits_from_hidden(&hidden).unwrap();
                drafter
                    .propose_dflash2_tokens(&hidden, &logits, 1000)
                    .unwrap()
                    .unwrap()
            })
        });
        println!(
            "fast_rows={fast} kept={kept}: stage_ms={stage:.2} head_ms={head:.2} head98k_ms={head98k:.2} argpartition_ms={part:.2} selector_ms={select:.2} all_ms={all:.2} (all = stage + full head + selector)"
        );
    }
}
