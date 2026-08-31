//! Assemble a parsed GGUF file into higgs's llama-style
//! [`transformer::Model`].
//!
//! The only work here is translation: GGUF metadata keys become the serde
//! config [`ModelArgs`] already parses, and GGUF tensor names become the
//! parameter keys the safetensors loaders already assign. Dequantization is
//! dispatched per tensor type through [`crate::gguf::dequant`].

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
use serde_json::json;

use crate::error::ModelError;
use crate::gguf::dequant::dequant_tensor;
use crate::gguf::parser::{GgufFile, GgufValue};
use crate::transformer::{self, Model};

/// GGUF metadata → the config JSON [`ModelArgs`] deserializes from.
/// Only the llama-family keys a GGUF file carries are mapped.
fn config_from_metadata(
    metadata: &HashMap<String, GgufValue>,
    tie_word_embeddings: bool,
) -> Result<serde_json::Value, String> {
    let get = |key: &str| -> Result<&GgufValue, String> {
        metadata
            .get(key)
            .ok_or_else(|| format!("GGUF metadata key {key} missing"))
    };
    let u32_of = |key: &str| -> Result<i32, String> {
        match get(key)? {
            GgufValue::U32(v) => Ok(*v as i32),
            other => Err(format!("{key} is not u32: {other:?}")),
        }
    };
    let f32_of = |key: &str| -> Result<f32, String> {
        match get(key)? {
            GgufValue::F32(v) => Ok(*v),
            other => Err(format!("{key} is not f32: {other:?}")),
        }
    };

    Ok(json!({
        "model_type": "llama",
        "hidden_size": u32_of("llama.embedding_length")?,
        "num_hidden_layers": u32_of("llama.block_count")?,
        "intermediate_size": u32_of("llama.feed_forward_length")?,
        "num_attention_heads": u32_of("llama.attention.head_count")?,
        "num_key_value_heads": u32_of("llama.attention.head_count_kv")?,
        "rms_norm_eps": f32_of("llama.attention.layer_norm_rms_epsilon")?,
        "vocab_size": u32_of("llama.vocab_size")?,
        "max_position_embeddings": u32_of("llama.context_length")?,
        "rope_theta": f32_of("llama.rope.freq_base")?,
        "tie_word_embeddings": tie_word_embeddings,
    }))
}

/// GGUF tensor name → higgs transformer parameter key. Returns `None` for
/// tensors the transformer does not carry (position embeddings, etc.).
pub fn param_key(name: &str) -> Option<String> {
    use std::sync::OnceLock;
    static GLOBAL: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    let global = GLOBAL.get_or_init(|| {
        [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
        ]
        .into_iter()
        .collect()
    });
    if let Some(key) = global.get(name) {
        return Some((*key).to_owned());
    }
    let rest = name.strip_prefix("blk.")?;
    let (index, tensor) = rest.split_once('.')?;
    let tensor = match tensor {
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{index}.{tensor}"))
}

/// Parse a GGUF file, dequantize every tensor, and assemble a
/// [`transformer::Model`]. Fails when a tensor the model expects cannot be
/// dequanted; unknown tensors are skipped with a warning.
pub fn load_transformer(path: &Path) -> Result<Model, ModelError> {
    let data = std::fs::read(path)?;
    let file = GgufFile::parse(data).map_err(ModelError::UnsupportedModel)?;

    let tied = !file.tensors.contains_key("output.weight");
    let config =
        config_from_metadata(&file.metadata, tied).map_err(ModelError::UnsupportedModel)?;
    let args: transformer::ModelArgs = serde_json::from_value(config)?;
    let mut model = Model::new(args)?;

    let mut assigned = 0usize;
    {
        let mut params = model.parameters_mut().flatten();
        for (name, info) in &file.tensors {
            let Some(key) = param_key(name) else {
                tracing::warn!(tensor = %name, "skipping tensor with no parameter mapping");
                continue;
            };
            let Some(target) = params.get_mut(key.as_str()) else {
                tracing::warn!(tensor = %name, key = %key, "no model parameter for tensor");
                continue;
            };
            let bytes = file
                .tensor_bytes(name)
                .ok_or_else(|| ModelError::MissingWeight(name.clone()))?
                .map_err(ModelError::UnsupportedModel)?;
            let values = dequant_tensor(info.dtype, bytes).map_err(ModelError::UnsupportedModel)?;
            // GGUF lists dims fastest-first; MLX weights are row-major
            // [out, in, ...] — the reverse.
            let shape: Vec<i32> = info.dims.iter().rev().map(|d| *d as i32).collect();
            **target = Array::from_slice(&values, &shape);
            assigned += 1;
        }
    }
    tracing::info!(tensors = assigned, "GGUF weights assigned");
    model.eval()?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_key_maps_llama_names() {
        assert_eq!(
            param_key("token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            param_key("blk.7.attn_q.weight").as_deref(),
            Some("model.layers.7.self_attn.q_proj.weight")
        );
        assert_eq!(
            param_key("blk.0.ffn_down.weight").as_deref(),
            Some("model.layers.0.mlp.down_proj.weight")
        );
        assert_eq!(param_key("rope_freqs.weight"), None);
    }

    /// End to end: real GGUF → weights → forward → greedy tokens.
    /// Opt in with HIGGS_GGUF_E2E_FILE=<gguf> and
    /// HIGGS_GGUF_E2E_TOKENIZER=<tokenizer.json>.
    #[test]
    fn real_file_forward_generation() {
        let (Ok(gguf_path), Ok(tok_path)) = (
            std::env::var("HIGGS_GGUF_E2E_FILE"),
            std::env::var("HIGGS_GGUF_E2E_TOKENIZER"),
        ) else {
            eprintln!("skipping: set HIGGS_GGUF_E2E_FILE and HIGGS_GGUF_E2E_TOKENIZER");
            return;
        };

        let _exec = crate::mlx_exec::acquire();
        let model = load_transformer(Path::new(&gguf_path)).expect("load");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");
        let mut any = crate::AnyModel::Transformer(model);
        let mut cache = any.make_cache().expect("cache");

        let prompt = "The capital of France is";
        let encoding = tokenizer.encode(prompt, false).expect("encode");
        let ids: Vec<i32> = encoding.get_ids().iter().map(|v| *v as i32).collect();

        // Prefill with the prompt, then greedy-decode a few tokens.
        let mut input = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let mut generated: Vec<u32> = Vec::new();
        for _ in 0..8 {
            let logits = any.forward(&input, None, &mut cache).expect("forward");
            logits.eval().expect("eval logits");
            let shape = logits.shape();
            let (seq, vocab) = (
                shape[shape.len() - 2] as usize,
                shape[shape.len() - 1] as usize,
            );
            let vals = logits.as_slice::<f32>();
            let last = &vals[(seq as usize - 1) * vocab as usize..seq as usize * vocab as usize];
            let next = last
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .expect("argmax");
            generated.push(next);
            input = Array::from_slice(&[next as i32], &[1, 1]);
        }

        let text = tokenizer.decode(&generated, false).expect("decode");
        eprintln!("prompt = {prompt:?} → {generated:?} = {text:?}");
        assert!(
            text.to_lowercase().contains("paris"),
            "greedy generation from {prompt:?} gave {text:?}"
        );
    }
}
