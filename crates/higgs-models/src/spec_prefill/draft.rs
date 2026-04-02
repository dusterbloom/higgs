// SPDX-License-Identifier: Apache-2.0
//! Draft model management for SpecPrefill token scoring.
//!
//! Loads and manages a small draft model (e.g., Qwen3.5-0.6B-4bit) for
//! attention-based token importance scoring.

use crate::error::ModelError;
use crate::qwen3_next;
use crate::registry::AnyModel;
use mlx_rs::{Array, Exception};
use std::path::{Path, PathBuf};

/// Draft model configuration.
#[derive(Debug, Clone)]
pub struct DraftModelConfig {
    /// Path to draft model directory.
    pub model_path: PathBuf,
    /// Number of lookahead decode steps (default 8).
    pub n_lookahead: usize,
    /// Sampling temperature for lookahead (default 0.6).
    pub temp: f32,
    /// Top-p for lookahead sampling (default 0.95).
    pub top_p: f32,
}

impl Default for DraftModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("mlx-community/Qwen3.5-0.6B-4bit"),
            n_lookahead: 8,
            temp: 0.6,
            top_p: 0.95,
        }
    }
}

/// Draft model wrapper for token scoring.
pub struct DraftModel {
    model: AnyModel,
    config: DraftModelConfig,
}

impl DraftModel {
    /// Load draft model from path.
    pub fn load(config: DraftModelConfig) -> Result<Self, ModelError> {
        tracing::info!(
            path = %config.model_path.display(),
            "Loading draft model for SpecPrefill"
        );

        // Load the draft model using the standard model loader
        // For now, we assume Qwen3.5 architecture
        let model =
            qwen3_next::load_qwen3_next_model(&config.model_path).map_err(ModelError::Model)?;

        tracing::info!("Draft model loaded successfully");

        Ok(Self {
            model: AnyModel::Qwen3Next(model),
            config,
        })
    }

    /// Get draft model config.
    pub fn config(&self) -> &DraftModelConfig {
        &self.config
    }

    /// Score token importance using attention patterns.
    ///
    /// This implements the SpecPrefill scoring algorithm:
    /// 1. Prefill draft model with all prompt tokens
    /// 2. Run N lookahead decode steps
    /// 3. Capture query vectors during attention
    /// 4. Compute importance = Q_lookahead @ K_prompt^T
    /// 5. Aggregate across heads/layers, apply smoothing
    ///
    /// # Arguments
    /// * `tokens` - Prompt token IDs
    ///
    /// # Returns
    /// Per-token importance scores (not normalized)
    pub fn score_tokens(&self, tokens: &[u32]) -> Result<Array, ModelError> {
        // TODO: Implement full attention-based scoring
        // For now, return uniform distribution as placeholder

        let n_tokens = tokens.len();
        let uniform = 1.0 / n_tokens as f32;
        let scores = Array::from_full(&[n_tokens as i32], uniform)
            .map_err(|e| ModelError::Mlx(e.to_string()))?;

        Ok(scores)
    }
}

/// Compute token importance using draft model.
///
/// # Arguments
/// * `draft_model` - Loaded draft model
/// * `tokens` - Prompt token IDs
/// * `n_lookahead` - Number of lookahead decode steps
/// * `pool_kernel` - Smoothing kernel size (0 to disable)
///
/// # Returns
/// Per-token importance scores
pub fn score_with_draft(draft_model: &DraftModel, tokens: &[u32]) -> Result<Array, ModelError> {
    draft_model.score_tokens(tokens)
}

/// Auto-select draft model based on target model size.
///
/// Returns recommended draft model path for the given target model.
pub fn auto_select_draft_model(target_model_path: &Path) -> PathBuf {
    // For Qwen3.5-35B, use Qwen3.5-0.6B as draft
    // For larger models, could use 4B variant
    PathBuf::from("mlx-community/Qwen3.5-0.6B-4bit")
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
    }
}
