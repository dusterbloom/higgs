//! Apple Neural Engine draft model for speculative decoding.
//!
//! Provides two draft model backends:
//! 1. **CPU (Accelerate SGEMM)** — loads weights from MLX safetensors, runs on CPU
//! 2. **CoreML (ANE)** — loads a `.mlpackage`, dispatched to ANE by CoreML runtime
//!
//! Designed for split-silicon speculative decoding:
//! ANE/CPU runs the draft model while GPU runs the target model.

mod config;
pub mod coreml;
mod decode;
mod safetensors;
mod weights;

pub use config::ModelConfig;
pub use coreml::CoreMlDraftModel;
pub use decode::{DecodeResult, KvCache};
pub use weights::{LayerWeights, ModelWeights};

use std::path::Path;

/// Load a draft model from an MLX safetensors directory.
///
/// Parses `config.json` for model dimensions, then loads all weight tensors
/// (dequantizing if necessary). Returns the model weights ready for decode.
pub fn load_draft_model(dir: &Path) -> Result<ModelWeights, String> {
    let cfg = config::ModelConfig::from_dir(dir)?;
    ModelWeights::from_mlx_safetensors(dir, &cfg)
        .map_err(|e| format!("failed to load draft model from {}: {e}", dir.display()))
}

/// Create a fresh KV cache for the given model.
pub fn make_kv_cache(model: &ModelWeights, max_seq: usize) -> KvCache {
    KvCache::new(&model.cfg, model.layers.len(), max_seq)
}

/// Decode one token, returning logits over the vocabulary.
pub fn decode_step(model: &ModelWeights, token: u32, kv_cache: &mut KvCache) -> DecodeResult {
    decode::decode_step(model, token, kv_cache)
}

/// Sample argmax from logits.
pub fn sample_argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}
