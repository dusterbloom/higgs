//! Speculative prefill — sparse prefill optimization (experimental).
//!
//! Selects a subset of prompt tokens for initial prefill, trading off
//! accuracy for reduced TTFT on long sequences.
//!
//! Currently disabled pending optimized RoPE implementation.

use higgs_models::error::ModelError;

/// Configuration for speculative prefill.
#[derive(Debug, Clone)]
pub struct SpecPrefillConfig {
    /// Minimum prompt length to trigger speculative prefill.
    pub min_prompt_len: usize,
    /// Fraction of tokens to keep during speculative prefill.
    pub keep_rate: f32,
}

impl Default for SpecPrefillConfig {
    fn default() -> Self {
        Self {
            min_prompt_len: 2048,
            keep_rate: 0.5,
        }
    }
}

/// Engine for speculative (sparse) prefill.
pub struct SpecPrefillEngine {
    config: SpecPrefillConfig,
}

impl SpecPrefillEngine {
    /// Create a new speculative prefill engine.
    ///
    /// Returns an error if `config.keep_rate` is outside `(0.0, 1.0]`.
    pub fn new(config: SpecPrefillConfig) -> Result<Self, ModelError> {
        if !(config.keep_rate > 0.0 && config.keep_rate <= 1.0) {
            return Err(ModelError::Mlx(mlx_rs::error::Exception::custom(format!(
                "SpecPrefill keep_rate must be in (0.0, 1.0], got {}",
                config.keep_rate
            ))));
        }
        Ok(Self { config })
    }

    /// Whether speculative prefill should be used for a given prompt length.
    pub fn should_use_spec_prefill(&self, prompt_len: usize) -> bool {
        prompt_len >= self.config.min_prompt_len
    }

    /// Get the keep rate for token selection.
    pub fn get_keep_rate(&self, _prompt_len: usize) -> f32 {
        self.config.keep_rate
    }
}
