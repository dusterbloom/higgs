//! A flat Qwen3.5-family `config.json` (no `text_config`, e.g. Clover packs)
//! must resolve to the same model args as its nested equivalent, including the
//! outer-level per-path quantization overrides.
use higgs_models::qwen3_next::resolve_runtime_model_args;
use serde_json::json;

#[test]
fn flat_config_resolves_like_its_nested_form() {
    let text = json!({
        "model_type": "qwen3_5_moe", "hidden_size": 256, "num_hidden_layers": 4,
        "intermediate_size": 0, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 64, "rms_norm_eps": 1e-6, "vocab_size": 1024,
        "max_position_embeddings": 512, "full_attention_interval": 4,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 32,
        "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "num_experts": 4,
        "num_experts_per_tok": 2, "shared_expert_intermediate_size": 256,
        "moe_intermediate_size": 128, "norm_topk_prob": true, "tie_word_embeddings": false,
    });
    let quantization = json!({
        "group_size": 64, "bits": 4, "mode": "affine",
        "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
        "language_model.model.layers.0.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
    });
    let mut flat = text.clone();
    flat["quantization"] = quantization.clone();
    let nested = json!({
        "model_type": "qwen3_5_moe", "text_config": text,
        "quantization": quantization, "tie_word_embeddings": false,
    });

    let dir = tempfile::tempdir().unwrap();
    let flat_args = resolve_runtime_model_args(dir.path(), &flat).unwrap();
    let nested_args = resolve_runtime_model_args(dir.path(), &nested).unwrap();

    assert_eq!(
        flat_args
            .quant_overrides
            .get("model.layers.0.mlp.gate")
            .map(|q| q.bits),
        Some(8),
        "8-bit router override must survive a flat config"
    );
    assert_eq!(format!("{flat_args:?}"), format!("{nested_args:?}"));
}
