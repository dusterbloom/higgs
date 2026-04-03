// SPDX-License-Identifier: Apache-2.0
//! Draft model management for SpecPrefill token scoring.

use std::path::{Path, PathBuf};
use mlx_rs::Array;
use crate::error::ModelError;
use crate::qwen3_next;
use super::scoring_attention;

#[derive(Debug, Clone)]
pub struct DraftModelConfig {
    pub model_path: PathBuf,
    pub n_lookahead: usize,
    pub temp: f32,
    pub top_p: f32,
    pub pool_kernel: usize,
}

impl Default for DraftModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("mlx-community/Qwen3-0.6B-4bit"),
            n_lookahead: 8,
            temp: 0.6,
            top_p: 0.95,
            pool_kernel: 13,
        }
    }
}

pub struct DraftModel {
    model: qwen3_next::Qwen3NextCausalLM,
    config: DraftModelConfig,
}

impl DraftModel {
    pub fn load(config: DraftModelConfig) -> Result<Self, ModelError> {
        tracing::info!(path = %config.model_path.display(), "Loading draft model for SpecPrefill");
        let model = qwen3_next::load_qwen3_next_model(&config.model_path)
            .map_err(|e| ModelError::UnsupportedModel(e.to_string()))?;
        tracing::info!("Draft model loaded successfully");
        Ok(Self { model, config })
    }
    
    pub fn config(&self) -> &DraftModelConfig {
        &self.config
    }
    
    pub fn score_tokens(&self, tokens: &[u32]) -> Result<Array, ModelError> {
        let mut model_clone = self.model.clone();
        scoring_attention::score_tokens_with_attention(&mut model_clone, tokens, &self.config)
    }
}

pub fn score_with_draft(draft_model: &DraftModel, tokens: &[u32]) -> Result<Array, ModelError> {
    draft_model.score_tokens(tokens)
}

pub fn auto_select_draft_model(_target_model_path: &Path) -> PathBuf {
    PathBuf::from("mlx-community/Qwen3-0.6B-4bit")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draft_model_config_defaults() {
        let config = DraftModelConfig::default();
        assert_eq!(config.n_lookahead, 8);
        assert!((config.temp - 0.6).abs() < 1e-6);
        assert!((config.top_p - 0.95).abs() < 1e-6);
        assert_eq!(config.pool_kernel, 13);
    }
}
