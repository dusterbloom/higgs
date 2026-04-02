// SPDX-License-Identifier: Apache-2.0
//! SpecPrefill engine integration.
//!
//! Integrates SpecPrefill with the Higgs inference engine to enable
//! sparse prefill for long prompts.

use super::spec_prefill::{
    self,
    draft::{DraftModel, DraftModelConfig},
    scoring::TokenImportance,
};
use crate::constrained::ConstrainedGenerator;
use crate::error::ModelError;
use crate::simple::{Engine, PreparedGeneration, SamplingParams};
use mlx_rs::{Array, Exception};
use std::path::{Path, PathBuf};

/// SpecPrefill configuration.
#[derive(Debug, Clone)]
pub struct SpecPrefillConfig {
    /// Enable SpecPrefill (default: true for prompts >8k).
    pub enabled: bool,
    /// Draft model path (auto-selected if None).
    pub draft_model_path: Option<PathBuf>,
    /// Enable SpecPrefill for prompts longer than this (default 8192).
    pub threshold: usize,
    /// Disable SpecPrefill for prompts longer than this (default 65536).
    pub max_tokens: usize,
    /// Keep rate override (auto-computed from context if None).
    pub keep_rate: Option<f32>,
    /// Chunk size for token selection (default 32).
    pub chunk_size: usize,
}

impl Default for SpecPrefillConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            draft_model_path: None,
            threshold: 8192,
            max_tokens: 65536,
            keep_rate: None,
            chunk_size: 32,
        }
    }
}

/// SpecPrefill engine wrapper.
pub struct SpecPrefillEngine {
    config: SpecPrefillConfig,
    draft_model: Option<DraftModel>,
}

impl SpecPrefillEngine {
    /// Create new SpecPrefill engine.
    pub fn new(config: SpecPrefillConfig) -> Result<Self, ModelError> {
        let draft_model = if config.enabled && config.draft_model_path.is_some() {
            let draft_config = DraftModelConfig {
                model_path: config.draft_model_path.clone().unwrap(),
                ..Default::default()
            };
            Some(DraftModel::load(draft_config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            draft_model,
        })
    }

    /// Check if SpecPrefill should be used for given prompt length.
    pub fn should_use_spec_prefill(&self, prompt_len: usize) -> bool {
        self.config.enabled
            && prompt_len >= self.config.threshold
            && prompt_len <= self.config.max_tokens
    }

    /// Get keep rate for given prompt length.
    pub fn get_keep_rate(&self, prompt_len: usize) -> f32 {
        self.config
            .keep_rate
            .unwrap_or_else(|| spec_prefill::compute_keep_rate(prompt_len))
    }

    /// Prefill with SpecPrefill optimization.
    ///
    /// # Arguments
    /// * `engine` - Main inference engine
    /// * `prepared` - Prepared generation state
    /// * `params` - Sampling parameters
    ///
    /// # Returns
    /// (logits, logprob_data) - Same as standard prefill
    pub fn prefill_with_spec(
        &self,
        engine: &mut Engine,
        prepared: &mut PreparedGeneration,
        params: &SamplingParams,
        logprob_top_n: Option<u32>,
        constraint: Option<&ConstrainedGenerator>,
    ) -> Result<(Array, Option<crate::simple::LogprobArrays>), ModelError> {
        let prompt_tokens = prepared.prompt_tokens.clone();
        let prompt_len = prompt_tokens.len();

        // Score tokens with draft model
        let importance = if let Some(draft) = &self.draft_model {
            draft.score_tokens(&prompt_tokens)?
        } else {
            // Fallback to uniform scoring
            spec_prefill::score_tokens_uniform(prompt_len)?.scores
        };

        // Select chunks to keep
        let keep_rate = self.get_keep_rate(prompt_len);
        let selected_indices = {
            let importance = TokenImportance {
                scores: importance,
                n_tokens: prompt_len,
            };
            importance.select_chunks(keep_rate, self.config.chunk_size)?
        };

        tracing::info!(
            prompt_len,
            selected_len = selected_indices.len(),
            keep_rate,
            "SpecPrefill: selected {}/{} tokens ({:.1}%)",
            selected_indices.len(),
            prompt_len,
            (selected_indices.len() as f32 / prompt_len as f32) * 100.0
        );

        // Run sparse prefill
        let (logits, state) = spec_prefill::sparse_prefill(
            &mut prepared.model,
            &prepared.prompt_array,
            &selected_indices,
            &mut prepared.cache,
            0, // position_offset
        )?;

        // Store state for cleanup after generation
        // TODO: Add state storage to PreparedGeneration

        // Extract last token logits
        let last_logits = logits.index((.., -1, ..));

        // Apply constraint mask if present
        let constrained_logits = if let Some(cg) = constraint {
            cg.apply_mask(&last_logits)?
        } else {
            last_logits
        };

        // Sample token
        let current_token = crate::simple::sample(&constrained_logits, params)?;

        // Compute logprobs if requested
        let logprob_data = if let Some(top_n) = logprob_top_n {
            let scaled = if params.temperature <= f32::EPSILON {
                constrained_logits
            } else {
                constrained_logits.multiply(Array::from_f32(1.0 / params.temperature))?
            };
            Some(crate::simple::LogprobArrays::compute(
                &scaled,
                &current_token,
                Some(top_n),
            )?)
        } else {
            None
        };

        // Evaluate
        {
            let mut eval_targets: Vec<&Array> = vec![&current_token];
            if let Some(ref lp) = logprob_data {
                eval_targets.extend(lp.eval_targets());
            }
            if constraint.is_some() {
                mlx_rs::eval(eval_targets)?;
            } else {
                mlx_rs::async_eval(eval_targets)?;
            }
        }

        // Store in cache
        if prepared.pixel_values.is_none() {
            let mut pc = engine
                .prefix_cache
                .lock()
                .map_err(|e| ModelError::Generation(format!("Cache lock poisoned: {}", e)))?;
            pc.store(&prompt_tokens, prepared.cache.clone());
        }

        Ok((current_token, logprob_data))
    }
}

/// Integration point: wrap Engine::run_prefill to use SpecPrefill when appropriate.
///
/// This function should be called from Engine::generate_inner before the standard
/// prefill path.
pub fn try_spec_prefill(
    engine: &mut Engine,
    prepared: &mut PreparedGeneration,
    params: &SamplingParams,
    logprob_top_n: Option<u32>,
    constraint: Option<&ConstrainedGenerator>,
    spec_engine: &SpecPrefillEngine,
) -> Result<Option<(Array, Option<crate::simple::LogprobArrays>)>, ModelError> {
    let prompt_len = prepared.prompt_tokens.len();

    if spec_engine.should_use_spec_prefill(prompt_len) {
        tracing::debug!("Using SpecPrefill for {} token prompt", prompt_len);
        let result =
            spec_engine.prefill_with_spec(engine, prepared, params, logprob_top_n, constraint)?;
        Ok(Some(result))
    } else {
        // Fall through to standard prefill
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_prefill_config_defaults() {
        let config = SpecPrefillConfig::default();
        assert!(config.enabled);
        assert_eq!(config.threshold, 8192);
        assert_eq!(config.max_tokens, 65536);
        assert_eq!(config.chunk_size, 32);
    }

    #[test]
    fn test_should_use_spec_prefill() {
        let engine = SpecPrefillEngine::new(SpecPrefillConfig::default()).unwrap();

        assert!(!engine.should_use_spec_prefill(1000));
        assert!(!engine.should_use_spec_prefill(8191));
        assert!(engine.should_use_spec_prefill(8192));
        assert!(engine.should_use_spec_prefill(16384));
        assert!(!engine.should_use_spec_prefill(65537));
    }
}
