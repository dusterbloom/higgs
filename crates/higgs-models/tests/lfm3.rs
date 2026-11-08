//! Native LFM3 loader and cache invariants on a tiny synthetic checkpoint.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use std::{collections::HashMap, path::Path};

use higgs_models::{
    LayerCache,
    lfm3::{Lfm3Args, Lfm3CausalLM, load_lfm3_model},
};
use mlx_rs::{Array, ops, ops::indexing::IndexOp};

const D: i32 = 64;
const H: i32 = 4;
const KV: i32 = 2;
const HD: i32 = 16;
const I: i32 = 128;
const V: i32 = 64;
const GROUP: i32 = 32;

fn normal(shape: &[i32]) -> Array {
    filled_f32(shape, 0.05)
}

fn filled_f32(shape: &[i32], value: f32) -> Array {
    let count = usize::try_from(shape.iter().product::<i32>()).unwrap();
    Array::from_slice(&vec![value; count], shape)
}

fn filled_u32(shape: &[i32], value: u32) -> Array {
    let count = usize::try_from(shape.iter().product::<i32>()).unwrap();
    Array::from_slice(&vec![value; count], shape)
}

fn put_q(weights: &mut HashMap<String, Array>, name: &str, shape: &[i32]) {
    let mut packed_shape = shape.to_vec();
    let input = *packed_shape.last().unwrap();
    *packed_shape.last_mut().unwrap() = input / 8;
    let mut group_shape = shape.to_vec();
    *group_shape.last_mut().unwrap() = input / GROUP;
    let weight = filled_u32(&packed_shape, 0);
    let scales = filled_f32(&group_shape, 0.02);
    let biases = normal(&group_shape);
    let p = format!("language_model.{name}");
    weights.insert(format!("{p}.weight"), weight);
    weights.insert(format!("{p}.scales"), scales);
    weights.insert(format!("{p}.biases"), biases);
}

fn put_norm(weights: &mut HashMap<String, Array>, name: &str, dim: i32) {
    weights.insert(
        format!("language_model.{name}.weight"),
        Array::ones::<f32>(&[dim]).unwrap(),
    );
}

/// A conv + sliding-attention + full-attention model, using the exact official
/// tensor names and the `language_model.` wrapper from the MLX checkpoint.
fn write_tiny_checkpoint(dir: &Path) {
    let config = serde_json::json!({
        "model_type": "lfm3", "hidden_size": D, "intermediate_size": I,
        "num_hidden_layers": 3, "num_attention_heads": H,
        "num_key_value_heads": KV, "head_dim": HD, "norm_eps": 1e-5,
        "conv_L_cache": 7, "conv_bias": false, "sliding_window": 4,
        "use_sandwich_norm": true, "vocab_size": V,
        "layer_types": ["conv", "sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "quantization": {"group_size": GROUP, "bits": 4, "mode": "affine"}
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let mut weights = HashMap::new();
    put_q(&mut weights, "model.embed_tokens", &[V, D]);
    put_norm(&mut weights, "model.embedding_norm", D);
    for layer in 0..3 {
        let p = format!("model.layers.{layer}");
        for norm in [
            "operator_norm",
            "post_operator_norm",
            "ffn_norm",
            "post_ffn_norm",
        ] {
            put_norm(&mut weights, &format!("{p}.{norm}"), D);
        }
        if layer == 0 {
            weights.insert(
                format!("language_model.{p}.conv.conv.weight"),
                normal(&[D, 7, 1]),
            );
            put_q(&mut weights, &format!("{p}.conv.in_proj"), &[3 * D, D]);
            put_q(&mut weights, &format!("{p}.conv.out_proj"), &[D, D]);
        } else {
            for (name, out) in [
                ("q_proj", H * HD),
                ("k_proj", KV * HD),
                ("v_proj", KV * HD),
                ("o_proj", D),
                ("gate_proj", H),
            ] {
                put_q(&mut weights, &format!("{p}.self_attn.{name}"), &[out, D]);
            }
            put_norm(&mut weights, &format!("{p}.self_attn.q_norm"), HD);
            put_norm(&mut weights, &format!("{p}.self_attn.k_norm"), HD);
        }
        put_q(&mut weights, &format!("{p}.feed_forward.w1"), &[I, D]);
        put_q(&mut weights, &format!("{p}.feed_forward.w2"), &[D, I]);
        put_q(&mut weights, &format!("{p}.feed_forward.w3"), &[I, D]);
    }
    Array::save_safetensors(&weights, None, &dir.join("model.safetensors")).unwrap();
}

fn valid_args_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "lfm3", "hidden_size": D, "intermediate_size": I,
        "num_hidden_layers": 1, "num_attention_heads": H,
        "num_key_value_heads": KV, "head_dim": HD, "norm_eps": 1e-5,
        "conv_L_cache": 7, "conv_bias": false, "sliding_window": 4,
        "use_sandwich_norm": true, "vocab_size": V,
        "layer_types": ["conv"], "rope_parameters": {}
    })
}

#[test]
fn loads_prefixed_checkpoint_and_produces_finite_logits() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_checkpoint(dir.path());
    let mut model = load_lfm3_model(dir.path()).unwrap();
    let tokens = Array::from_slice(&[1_i32, 2, 3, 4, 5], &[1, 5]);
    let logits = model
        .forward_all_logits(&tokens, None, &mut Vec::new())
        .unwrap();

    assert_eq!(model.args.head_dim(), HD);
    assert_eq!(logits.shape(), &[1, 5, V]);
    let max_abs = logits.abs().unwrap().max(None).unwrap().item::<f32>();
    assert!(max_abs.is_finite(), "logits contained a non-finite value");
}

#[test]
fn chunked_decode_matches_one_shot_across_sliding_window() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_checkpoint(dir.path());
    let mut model = load_lfm3_model(dir.path()).unwrap();
    let tokens = Array::from_slice(&[3_i32, 17, 42, 5, 8, 9, 61, 30, 7], &[1, 9]);
    let one_shot = model
        .forward_all_logits(&tokens, None, &mut Vec::new())
        .unwrap();

    let mut cache = Vec::new();
    let chunks = [(0, 3), (3, 7), (7, 8), (8, 9)].map(|(a, b)| {
        model
            .forward_all_logits(&tokens.index((.., a..b)), None, &mut cache)
            .unwrap()
    });
    let chunked = ops::concatenate_axis(&chunks, 1).unwrap();
    let max_diff = one_shot
        .subtract(&chunked)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item::<f32>();
    assert!(max_diff < 2e-3, "max |delta logit| = {max_diff}");

    let LayerCache::Arrays(conv) = cache[0].as_ref().unwrap() else {
        panic!("layer 0 must have a convolution cache")
    };
    assert_eq!(conv.offset, 9);
}

#[test]
fn rejects_invalid_layer_schedule() {
    let args: Lfm3Args = serde_json::from_value(serde_json::json!({
        "model_type": "lfm3", "hidden_size": D, "intermediate_size": I,
        "num_hidden_layers": 2, "num_attention_heads": H,
        "num_key_value_heads": KV, "head_dim": HD, "norm_eps": 1e-5,
        "conv_L_cache": 7, "conv_bias": false, "sliding_window": 4,
        "use_sandwich_norm": true, "vocab_size": V,
        "layer_types": ["conv"], "rope_parameters": {}
    }))
    .unwrap();
    assert!(Lfm3CausalLM::new(args, None).is_err());
}

#[test]
fn rejects_zero_attention_heads_without_panicking() {
    let mut value = valid_args_json();
    value["num_attention_heads"] = serde_json::json!(0);
    value.as_object_mut().unwrap().remove("head_dim");
    let args: Lfm3Args = serde_json::from_value(value).unwrap();

    assert!(Lfm3CausalLM::new(args, None).is_err());
}

#[test]
fn rejects_overflowing_attention_dimensions_without_panicking() {
    let mut value = valid_args_json();
    value["hidden_size"] = serde_json::json!(1);
    value["num_attention_heads"] = serde_json::json!(i32::MAX);
    value["num_key_value_heads"] = serde_json::json!(1);
    value["head_dim"] = serde_json::json!(i32::MAX);
    let args: Lfm3Args = serde_json::from_value(value).unwrap();

    let result = std::panic::catch_unwind(|| Lfm3CausalLM::new(args, None));
    assert!(result.is_ok(), "invalid dimensions must not panic");
    assert!(result.unwrap().is_err());
}

#[test]
fn rejects_untied_embeddings() {
    let mut value = valid_args_json();
    value["tie_word_embeddings"] = serde_json::json!(false);
    let args: Lfm3Args = serde_json::from_value(value).unwrap();

    assert!(Lfm3CausalLM::new(args, None).is_err());
}

#[test]
fn rejects_explicit_mask_before_initializing_cache() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_checkpoint(dir.path());
    let mut model = load_lfm3_model(dir.path()).unwrap();
    let tokens = Array::from_slice(&[1_i32], &[1, 1]);
    let mask = Array::from_slice(&[true], &[1, 1]);
    let mut cache = Vec::new();

    assert!(
        model
            .forward_hidden(&tokens, Some(&mask), &mut cache)
            .is_err()
    );
    assert!(cache.is_empty(), "mask rejection must not initialize cache");
}

#[test]
fn rejects_checkpoint_missing_required_quantization_tensor() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_checkpoint(dir.path());
    let path = dir.path().join("model.safetensors");
    let mut weights = Array::load_safetensors(&path).unwrap();
    weights.remove("language_model.model.layers.0.feed_forward.w1.scales");
    Array::save_safetensors(&weights, None, &path).unwrap();

    let error = load_lfm3_model(dir.path()).unwrap_err().to_string();
    assert!(error.contains("w1.scales"), "unexpected error: {error}");
}

#[test]
fn rejects_checkpoint_tensor_with_wrong_shape() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_checkpoint(dir.path());
    let path = dir.path().join("model.safetensors");
    let mut weights = Array::load_safetensors(&path).unwrap();
    weights.insert(
        "language_model.model.embedding_norm.weight".to_owned(),
        filled_f32(&[D - 1], 1.0),
    );
    Array::save_safetensors(&weights, None, &path).unwrap();

    let error = load_lfm3_model(dir.path()).unwrap_err().to_string();
    assert!(
        error.contains("embedding_norm.weight") && error.contains("shape"),
        "unexpected error: {error}"
    );
}

#[test]
fn rejects_non_affine_quantization_mode() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_checkpoint(dir.path());
    let config_path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&config_path).unwrap()).unwrap();
    config["quantization"]["mode"] = serde_json::json!("mxfp4");
    std::fs::write(&config_path, config.to_string()).unwrap();

    let error = load_lfm3_model(dir.path()).unwrap_err().to_string();
    assert!(error.contains("mxfp4"), "unexpected error: {error}");
}
