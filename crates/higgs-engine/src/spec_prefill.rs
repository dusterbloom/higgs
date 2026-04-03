// SPDX-License-Identifier: Apache-2.0
//! SpecPrefill engine integration (stub).

use higgs_models::error::ModelError;

/// SpecPrefill configuration.
#[derive(Debug, Clone)]
pub struct SpecPrefillConfig {
    pub enabled: bool,
    pub threshold: usize,
    pub max_tokens: usize,
    pub keep_rate: Option<f32>,
    pub chunk_size: usize,
}

impl Default for SpecPrefillConfig {
    fn default() -> Self {
        Self {
            enabled: true,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_prefill_config_defaults() {
        let config = SpecPrefillConfig::default();
        assert!(config.enabled);
        assert_eq!(config.threshold, 8192);
        assert_eq!(config.max_tokens, 65536);
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
