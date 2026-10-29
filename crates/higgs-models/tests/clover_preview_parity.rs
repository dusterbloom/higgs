//! Clover parity against DiffuSpace's reference implementation on a 4-layer
//! slice. Build the fixture with `examples/clover_reference_fixture.py`, then:
//!
//! ```sh
//! HIGGS_CLOVER_FIXTURE_DIR=~/AI-Models/clover-fixture \
//!   cargo test -p higgs-models --release --test clover_preview_parity -- --ignored --nocapture
//! ```
use std::path::PathBuf;

use higgs_models::{
    cache::KeyValueCache,
    qwen3_next::{LayerCache, Qwen3NextCausalLM, load_qwen3_5_moe_model},
};
use mlx_rs::{Array, Dtype, ops::indexing::IndexOp};

struct Fixture {
    model: Qwen3NextCausalLM,
    logits: std::collections::HashMap<String, Array>,
    meta: serde_json::Value,
}

fn fixture() -> Option<Fixture> {
    let dir = PathBuf::from(std::env::var("HIGGS_CLOVER_FIXTURE_DIR").ok()?);
    let model = load_qwen3_5_moe_model(dir.join("slice")).unwrap();
    let logits = Array::load_safetensors(dir.join("fixture.safetensors")).unwrap();
    let meta =
        serde_json::from_str(&std::fs::read_to_string(dir.join("fixture.json")).unwrap()).unwrap();
    Some(Fixture {
        model,
        logits,
        meta,
    })
}

fn ids(meta: &serde_json::Value, key: &str) -> Vec<u32> {
    meta[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

/// Row-wise agreement of `got` vs the reference `want` (both `[rows, vocab]`).
/// bf16 activations over 4-bit weights vs the reference's dense dequantized
/// weights: top-1 must agree and the logit vectors must be near-collinear.
fn assert_close(tag: &str, got: &Array, want: &Array) {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let rows = want.shape()[0];
    let mut top1 = 0;
    let mut worst_cos = f32::MAX;
    for r in 0..rows {
        let (g, w) = (got.index(r), want.index(r));
        let g_arg: u32 = mlx_rs::ops::indexing::argmax(&g, None).unwrap().item();
        let w_arg: u32 = mlx_rs::ops::indexing::argmax(&w, None).unwrap().item();
        top1 += usize::from(g_arg == w_arg);
        let dot: f32 = g.multiply(&w).unwrap().sum(None).unwrap().item();
        let gn: f32 = g
            .square()
            .unwrap()
            .sum(None)
            .unwrap()
            .sqrt()
            .unwrap()
            .item();
        let wn: f32 = w
            .square()
            .unwrap()
            .sum(None)
            .unwrap()
            .sqrt()
            .unwrap()
            .item();
        worst_cos = worst_cos.min(dot / (gn * wn));
    }
    let rows = usize::try_from(rows).unwrap();
    println!("{tag}: top1 {top1}/{rows}, worst cosine {worst_cos:.5}");
    assert!(
        worst_cos > 0.99,
        "{tag}: logits diverge from reference (cosine {worst_cos})"
    );
    assert!(
        top1 * 100 >= rows * 95,
        "{tag}: top-1 agreement {top1}/{rows}"
    );
}

#[test]
#[ignore = "needs HIGGS_CLOVER_FIXTURE_DIR (examples/clover_reference_fixture.py)"]
fn causal_prefill_matches_reference() {
    let Some(mut f) = fixture() else { return };
    let prompt = ids(&f.meta, "prompt");
    let n = i32::try_from(prompt.len()).unwrap();
    let mut cache = f.model.make_cache();
    let logits = f
        .model
        .forward(&Array::from_slice(&prompt, &[1, n]), None, &mut cache)
        .unwrap()
        .reshape(&[1, -1])
        .unwrap();
    let want = f.logits["prefill_logits"].index((n - 1..n, ..));
    assert_close("prefill last position", &logits, &want);
}

/// Every committed cache array, materialised on the host: preview must leave
/// all of it bit-identical.
fn cache_snapshot(cache: &[Option<LayerCache>]) -> Vec<(i32, Vec<f32>)> {
    let host = |a: &Array| {
        let a = a.as_dtype(Dtype::Float32).unwrap();
        a.eval().unwrap();
        a.as_slice::<f32>().to_vec()
    };
    cache
        .iter()
        .flatten()
        .flat_map(|layer| match layer {
            LayerCache::Arrays(state) => vec![
                (
                    state.offset,
                    state.conv_state.as_ref().map(host).unwrap_or_default(),
                ),
                (
                    state.conv_pos,
                    state.ssm_state.as_ref().map(host).unwrap_or_default(),
                ),
            ],
            LayerCache::KV(kv) => kv
                .eval_targets()
                .into_iter()
                .map(|a| (kv.offset(), host(a)))
                .collect(),
        })
        .collect()
}

#[test]
#[ignore = "needs HIGGS_CLOVER_FIXTURE_DIR (examples/clover_reference_fixture.py)"]
fn block_preview_matches_reference_and_leaves_cache_untouched() {
    let Some(mut f) = fixture() else { return };
    let mask = u32::try_from(f.meta["mask_token_id"].as_u64().unwrap()).unwrap();
    let prompt = ids(&f.meta, "prompt");
    let mut cache = f.model.make_cache();
    f.model
        .forward(
            &Array::from_slice(&prompt, &[1, i32::try_from(prompt.len()).unwrap()]),
            None,
            &mut cache,
        )
        .unwrap()
        .eval()
        .unwrap();
    let committed = cache_snapshot(&cache);

    let preview = |model: &mut Qwen3NextCausalLM, cache: &[Option<LayerCache>], block: &[u32]| {
        let n = i32::try_from(block.len()).unwrap();
        let logits = model
            .forward_preview_block(&Array::from_slice(block, &[1, n]), cache)
            .unwrap();
        logits.reshape(&[n, -1]).unwrap()
    };
    let all_mask = vec![mask; 32];
    let got = preview(&mut f.model, &cache, &all_mask);
    assert_close(
        "preview all-mask 32",
        &got,
        &f.logits["preview_mask32_logits"],
    );
    let got = preview(&mut f.model, &cache, &ids(&f.meta, "partial_block"));
    assert_close(
        "preview partial 32",
        &got,
        &f.logits["preview_partial32_logits"],
    );
    let got = preview(&mut f.model, &cache, &[mask; 8]);
    assert_close(
        "preview all-mask 8",
        &got,
        &f.logits["preview_mask8_logits"],
    );
    assert!(
        cache_snapshot(&cache) == committed,
        "block preview mutated the committed cache"
    );

    // Commit the reference's resolved block causally, then preview the next.
    let first = ids(&f.meta, "first_block");
    f.model
        .forward(&Array::from_slice(&first, &[1, 32]), None, &mut cache)
        .unwrap()
        .eval()
        .unwrap();
    let got = preview(&mut f.model, &cache, &all_mask);
    assert_close(
        "preview after commit",
        &got,
        &f.logits["preview_after_commit_logits"],
    );
}
