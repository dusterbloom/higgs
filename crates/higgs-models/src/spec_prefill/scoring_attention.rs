// SPDX-License-Identifier: Apache-2.0
//! Attention-based token importance scoring stub.

use mlx_rs::Array;
use crate::error::ModelError;
use crate::qwen3_next::Qwen3NextCausalLM;
use super::draft::DraftModelConfig;

/// Stub implementation - returns uniform distribution
pub fn score_tokens_with_attention(
    _model: &mut Qwen3NextCausalLM,
    tokens: &[u32],
    _config: &DraftModelConfig,
) -> Result<Array, ModelError> {
    let n_prompt = tokens.len();
    let uniform = 1.0 / n_prompt as f32;
    Ok(Array::from_iter(vec![uniform; n_prompt], &[n_prompt as i32]))
}
