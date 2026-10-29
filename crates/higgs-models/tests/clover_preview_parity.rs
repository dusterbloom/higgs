//! Clover parity against DiffuSpace's reference implementation on a 4-layer
//! slice. Build the fixture with `examples/clover_reference_fixture.py`, then:
//!
//! ```sh
//! HIGGS_CLOVER_FIXTURE_DIR=~/AI-Models/clover-fixture \
//!   cargo test -p higgs-models --release --test clover_preview_parity -- --ignored --nocapture
//! ```
use std::path::PathBuf;

use higgs_models::{
    AnyModel, SamplingParams,
    cache::KeyValueCache,
    clover::{AnyModelBlocks, BlockDecodeConfig, BlockModel, CloverPenalties, block_decode},
    qwen3_next::{LayerCache, Qwen3NextCausalLM, load_qwen3_5_moe_model},
};
use mlx_rs::{Array, Dtype, error::Exception, ops::indexing::IndexOp};

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

fn generate_case<'a>(meta: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    meta["generate"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("fixture lacks generate case {name}"))
}

/// The block-loop config the reference `generate` ran with for `case`.
fn case_config(meta: &serde_json::Value, case: &serde_json::Value) -> BlockDecodeConfig {
    let settings = &case["settings"];
    let int = |key: &str| usize::try_from(settings[key].as_u64().unwrap()).unwrap();
    BlockDecodeConfig {
        block_size: int("block_size"),
        #[allow(clippy::cast_possible_truncation)]
        threshold: settings["threshold"].as_f64().unwrap() as f32,
        mask_token_id: u32::try_from(meta["mask_token_id"].as_u64().unwrap()).unwrap(),
        max_new_tokens: int("max_new_tokens"),
        eos_token_ids: ids(settings, "eos_token_id"),
        pad_token_id: settings["pad_token_id"]
            .as_u64()
            .map(|v| u32::try_from(v).unwrap()),
        penalties: CloverPenalties::from_generation_config(settings),
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
    }
}

/// Replays the reference's own preview logits, one array per pass.
struct Replay {
    logits: Vec<Array>,
    commits: Vec<Vec<u32>>,
}

impl BlockModel for Replay {
    fn preview(&mut self, _block: &[u32]) -> Result<Array, Exception> {
        assert!(
            !self.logits.is_empty(),
            "more preview passes than the reference"
        );
        Ok(self.logits.remove(0))
    }

    fn commit(&mut self, tokens: &[u32]) -> Result<(), Exception> {
        self.commits.push(tokens.to_vec());
        Ok(())
    }
}

/// Block loop + penalties + ramps on the reference's exact logits must
/// reproduce the reference `generate` tokens bit-for-bit. Threshold 0 resolves
/// a block in one pass, so the fixture's all-mask previews (after the prompt,
/// and after committing the first block) are exactly what the reference saw.
#[test]
#[ignore = "needs HIGGS_CLOVER_FIXTURE_DIR (examples/clover_reference_fixture.py)"]
fn block_loop_on_reference_logits_matches_reference_generate() {
    let Some(f) = fixture() else { return };
    let _gate = higgs_models::mlx_exec::acquire();
    let first = ids(&f.meta, "first_block");
    for (name, blocks) in [
        ("greedy@0.0", 2),
        ("pack_penalties@0.0", 2),
        // Its second block follows a different first block; replay one.
        ("ramps@0.0", 1),
    ] {
        let case = generate_case(&f.meta, name);
        let mut cfg = case_config(&f.meta, case);
        cfg.max_new_tokens = 32 * blocks;
        let mut want = ids(case, "tokens");
        want.truncate(cfg.max_new_tokens);
        if blocks > 1 {
            assert_eq!(
                want[..32],
                first[..],
                "{name}: replayed block 2 follows first_block"
            );
        }
        let mut replay = Replay {
            logits: vec![
                f.logits["preview_mask32_logits"].clone(),
                f.logits["preview_after_commit_logits"].clone(),
            ],
            commits: Vec::new(),
        };
        let vocab = f.logits["preview_mask32_logits"].shape()[1];
        let out = block_decode(&cfg, vocab, &mut replay, |_| false).unwrap();
        assert_eq!(out.tokens, want, "{name}");
        assert_eq!(out.passes_per_block, vec![1; blocks], "{name}");
        assert_eq!(replay.commits, vec![first.clone(); blocks - 1], "{name}");
    }
}

/// Full higgs diffusion (preview forward + causal commits) on the slice vs the
/// reference `generate`. Exact for `greedy@0.0`. The other cases are printed,
/// not asserted: on this 4-layer slice they hinge on sub-ulp bf16 ties
/// (measured 2026-10-29: `pack_penalties@0.0` token 47, penalised 83 beats
/// 82 by 0.039 logits in the reference; `ramps@0.0` token 22, think-end beats
/// EOS by 0.020; every `@0.95` case unmasks one position per pass by
/// near-uniform ~1% confidences, first reordered at pass 5 where the
/// reference's top pending confidences are 0.01053/0.01047/0.01047).
#[test]
#[ignore = "needs HIGGS_CLOVER_FIXTURE_DIR (examples/clover_reference_fixture.py)"]
fn diffusion_generate_on_slice_matches_reference_tokens() {
    let Some(f) = fixture() else { return };
    let _gate = higgs_models::mlx_exec::acquire();
    let prompt = ids(&f.meta, "prompt");
    let mut model = AnyModel::Qwen3Next(f.model);
    for case in f.meta["generate"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cfg = case_config(&f.meta, case);
        let want = ids(case, "tokens");
        let mut cache = model.make_cache().unwrap();
        let n = i32::try_from(prompt.len()).unwrap();
        let logits = model
            .forward(&Array::from_slice(&prompt, &[1, n]), None, &mut cache)
            .unwrap();
        let vocab = *logits.shape().last().unwrap();
        let mut blocks = AnyModelBlocks {
            model: &mut model,
            cache: &mut cache,
        };
        let out = block_decode(&cfg, vocab, &mut blocks, |_| false).unwrap();
        let diverged = out
            .tokens
            .iter()
            .zip(&want)
            .position(|(a, b)| a != b)
            .or_else(|| (out.tokens.len() != want.len()).then(|| out.tokens.len().min(want.len())));
        println!(
            "{name}: {} tokens (reference {}), passes/block {:?}, first divergence {diverged:?}",
            out.tokens.len(),
            want.len(),
            out.passes_per_block
        );
        if name == "greedy@0.0" {
            assert_eq!(out.tokens, want, "{name}");
        }
    }
}
