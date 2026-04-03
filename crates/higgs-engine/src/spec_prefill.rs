// SPDX-License-Identifier: Apache-2.0
//! SpecPrefill engine integration stub.

use higgs_models::error::ModelError;

#[derive(Debug, Clone)]
pub struct SpecPrefillConfig {
    pub enabled: bool,
    pub draft_model_path: Option<std::path::PathBuf>,
    pub threshold: usize,
    pub max_tokens: usize,
    pub keep_rate: Option<f32>,
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

pub struct SpecPrefillEngine {
    config: SpecPrefillConfig,
}

impl SpecPrefillEngine {
    pub fn new(config: SpecPrefillConfig) -> Result<Self, ModelError> {
        Ok(Self { config })
    }
    
    pub fn should_use_spec_prefill(&self, prompt_len: usize) -> bool {
        self.config.enabled && prompt_len >= self.config.threshold && prompt_len <= self.config.max_tokens
    }
    
    pub fn get_keep_rate(&self, prompt_len: usize) -> f32 {
        self.config.keep_rate.unwrap_or_else(|| {
            if prompt_len < 8192 { 1.0 }
            else if prompt_len < 16384 { 0.30 }
            else if prompt_len < 32768 { 0.25 }
            else { 0.20 }
        })
    }
}
