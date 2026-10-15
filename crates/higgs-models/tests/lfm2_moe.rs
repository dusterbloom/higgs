//! `lfm2_moe` loader + forward invariants on a tiny synthetic checkpoint.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use std::collections::HashMap;

use std::path::Path;

use higgs_models::{
    LayerCache,
    lfm2_moe::{Lfm2MoeArgs, Lfm2MoeCausalLM, load_lfm2_moe_model},
};
use mlx_rs::{Array, ops, ops::indexing::IndexOp, random};

const D: i32 = 64;
const V: i32 = 96;
const E: i32 = 4;

fn normal(shape: &[i32]) -> Array {
    random::normal::<f32>(shape, None, None, None).unwrap()
}

/// Quantize `w` (last axis = input dim) and store it under `name` at `bits`.
fn put_q(w: &mut HashMap<String, Array>, name: &str, shape: &[i32], bits: i32) {
    let (q, s, b) = ops::quantize(
        &normal(shape).multiply(Array::from_f32(0.1)).unwrap(),
        32,
        bits,
    )
    .unwrap();
    w.insert(format!("{name}.weight"), q);
    w.insert(format!("{name}.scales"), s);
    w.insert(format!("{name}.biases"), b);
}

/// A 3-layer (conv+dense, attention+MoE, conv+MoE) checkpoint with a
/// per-tensor 8-bit embedding override, written the way mlx-lm lays it out.
fn write_tiny_checkpoint(dir: &Path) {
    let config = serde_json::json!({
        "model_type": "lfm2_moe", "hidden_size": D, "num_hidden_layers": 3,
        "num_attention_heads": 2, "num_key_value_heads": 1, "num_experts": E,
        "num_experts_per_tok": 2, "num_dense_layers": 1, "norm_eps": 1e-5,
        "conv_L_cache": 3, "layer_types": ["conv", "full_attention", "conv"],
        "norm_topk_prob": true, "use_expert_bias": true, "routed_scaling_factor": 1.0,
        "rope_parameters": {"rope_theta": 10000.0}, "vocab_size": V,
        "quantization": {"group_size": 32, "bits": 4, "mode": "affine",
            "model.embed_tokens": {"group_size": 32, "bits": 8}},
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let mut w = HashMap::new();
    put_q(&mut w, "model.embed_tokens", &[V, D], 8);
    w.insert(
        "model.embedding_norm.weight".into(),
        Array::ones::<f32>(&[D]).unwrap(),
    );
    for i in 0..3 {
        let p = format!("model.layers.{i}");
        for norm in ["operator_norm", "ffn_norm"] {
            w.insert(
                format!("{p}.{norm}.weight"),
                Array::ones::<f32>(&[D]).unwrap(),
            );
        }
        if i == 1 {
            for (n, out) in [
                ("q_proj", D),
                ("k_proj", 32),
                ("v_proj", 32),
                ("out_proj", D),
            ] {
                put_q(&mut w, &format!("{p}.self_attn.{n}"), &[out, D], 4);
            }
            for n in ["q_layernorm", "k_layernorm"] {
                w.insert(
                    format!("{p}.self_attn.{n}.weight"),
                    Array::ones::<f32>(&[32]).unwrap(),
                );
            }
        } else {
            w.insert(format!("{p}.conv.conv.weight"), normal(&[D, 3, 1]));
            put_q(&mut w, &format!("{p}.conv.in_proj"), &[3 * D, D], 4);
            put_q(&mut w, &format!("{p}.conv.out_proj"), &[D, D], 4);
        }
        let ff = format!("{p}.feed_forward");
        if i == 0 {
            put_q(&mut w, &format!("{ff}.gate_proj"), &[D, D], 4);
            put_q(&mut w, &format!("{ff}.up_proj"), &[D, D], 4);
            put_q(&mut w, &format!("{ff}.down_proj"), &[D, D], 4);
        } else {
            w.insert(format!("{ff}.gate.weight"), normal(&[E, D]));
            w.insert(format!("{ff}.expert_bias"), normal(&[E]));
            for n in ["gate_proj", "up_proj", "down_proj"] {
                put_q(&mut w, &format!("{ff}.switch_mlp.{n}"), &[E, D, D], 4);
            }
        }
    }
    Array::save_safetensors(&w, None, &dir.join("model.safetensors")).unwrap();
}

/// Feeding a sequence in chunks (prefill, multi-token continuation with an
/// offset mask, then single-token decode) must reproduce the one-shot
/// logits — this exercises conv state, KV cache, offsets and masking.
#[test]
fn chunked_decode_matches_one_shot_prefill() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_checkpoint(dir.path());
    let mut model = load_lfm2_moe_model(dir.path()).unwrap();
    let tokens = Array::from_slice(&[3_i32, 17, 42, 5, 88, 9, 61, 30], &[1, 8]);

    let one_shot = model
        .forward_all_logits(&tokens, None, &mut Vec::new())
        .unwrap();

    let mut cache = Vec::new();
    let chunks = [(0, 4), (4, 6), (6, 7), (7, 8)].map(|(a, b)| {
        model
            .forward_all_logits(&tokens.index((.., a..b)), None, &mut cache)
            .unwrap()
    });
    let chunked = ops::concatenate_axis(&chunks, 1).unwrap();

    assert_eq!(one_shot.shape(), &[1, 8, V]);
    let max_diff = one_shot
        .subtract(&chunked)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap();
    assert!(
        max_diff.item::<f32>() < 1e-3,
        "max |Δlogit| = {}",
        max_diff.item::<f32>()
    );
    let LayerCache::Arrays(conv) = cache[0].as_ref().unwrap() else {
        panic!("layer 0 is conv")
    };
    assert_eq!(conv.offset, 8, "conv offset must track the KV offset");
}

#[test]
fn rejects_layer_types_mismatch() {
    let args: Lfm2MoeArgs = serde_json::from_value(serde_json::json!({
        "model_type": "lfm2_moe", "hidden_size": D, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "num_experts_per_tok": 2,
        "num_dense_layers": 1, "norm_eps": 1e-5, "conv_L_cache": 3, "layer_types": ["conv"],
    }))
    .unwrap();
    assert!(Lfm2MoeCausalLM::new(args, None).is_err());
}
