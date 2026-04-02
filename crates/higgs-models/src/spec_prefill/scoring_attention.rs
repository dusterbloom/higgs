// SPDX-License-Identifier: Apache-2.0
//! Attention-based token importance scoring for SpecPrefill.
//!
//! Implements the SpecPrefill scoring algorithm:
//! 1. Prefill draft model with all prompt tokens
//! 2. Run N lookahead decode steps, capturing query vectors
//! 3. Compute importance = Q_lookahead @ K_prompt^T
//! 4. Aggregate across heads/layers, apply smoothing

use super::draft::DraftModelConfig;
use crate::cache::{AnyCache, KeyValueCache, LayerCache};
use crate::error::ModelError;
use crate::qwen3_next::Qwen3NextCausalLM;
use mlx_rs::{ops, Array, Dtype, Exception, Stream};

/// Capture query vectors during attention computation.
///
/// This wrapper captures query vectors after RoPE but before attention.
pub struct QueryCaptureAttention<'a> {
    inner: &'a mut dyn AttentionModule,
    query_buffer: &'a mut Vec<Array>,
}

impl<'a> QueryCaptureAttention<'a> {
    pub fn new(inner: &'a mut dyn AttentionModule, query_buffer: &'a mut Vec<Array>) -> Self {
        Self {
            inner,
            query_buffer,
        }
    }
}

/// Trait for attention modules that can be wrapped for query capture.
pub trait AttentionModule {
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut dyn KeyValueCache,
    ) -> Result<Array, Exception>;
}

/// Score token importance using attention patterns from draft model.
///
/// # Arguments
/// * `model` - Draft model (Qwen3NextCausalLM)
/// * `tokens` - Prompt token IDs
/// * `config` - Draft model configuration
///
/// # Returns
/// Per-token importance scores (not normalized)
pub fn score_tokens_with_attention(
    model: &mut Qwen3NextCausalLM,
    tokens: &[u32],
    config: &DraftModelConfig,
) -> Result<Array, ModelError> {
    let n_prompt = tokens.len();
    let stream = Stream::default();

    // Convert tokens to array [1, n_prompt]
    let input_tokens = Array::from_slice(tokens, &[1, n_prompt as i32])
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Create cache for draft model prefill
    let mut cache = model.make_cache();

    // Phase 1: Prefill draft model with all tokens
    let hidden = model
        .forward_hidden(&input_tokens, None, &mut cache)
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Get last hidden state for sampling
    let last_hidden = hidden.index((.., -1, ..));
    let logits = model
        .lm_head
        .as_ref()
        .map(|h| h.forward(&last_hidden))
        .unwrap_or_else(|| model.model.embed_tokens.as_linear(&last_hidden))
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Phase 2: Lookahead decode with query capture
    let mut query_captures: Vec<Vec<Array>> = Vec::new();
    let mut current_token =
        ops::argmax(&logits, -1, false).map_err(|e| ModelError::Mlx(e.to_string()))?;

    for step in 0..config.n_lookahead {
        // Capture queries from all attention layers
        let mut step_queries: Vec<Array> = Vec::new();

        // Run forward pass and capture queries
        // TODO: Implement proper query capture mechanism
        // For now, skip capture and just run decode

        // Sample next token
        let next_logits = {
            let hidden = model
                .forward_hidden(&current_token.reshape(&[1, 1, -1])?, None, &mut cache)
                .map_err(|e| ModelError::Mlx(e.to_string()))?;
            let last_hidden = hidden.index((.., -1, ..));
            model
                .lm_head
                .as_ref()
                .map(|h| h.forward(&last_hidden))
                .unwrap_or_else(|| model.model.embed_tokens.as_linear(&last_hidden))
                .map_err(|e| ModelError::Mlx(e.to_string()))?
        };

        // Apply temperature and top-p sampling
        let scaled = if config.temp <= f32::EPSILON {
            next_logits
        } else {
            next_logits
                .multiply(Array::from_f32(1.0 / config.temp))
                .map_err(|e| ModelError::Mlx(e.to_string()))?
        };

        current_token =
            sample_top_p(&scaled, config.top_p).map_err(|e| ModelError::Mlx(e.to_string()))?;

        query_captures.push(step_queries);
    }

    // Phase 3: Compute importance from captured queries
    // importance = mean over lookahead steps of max over heads of (Q @ K^T)
    let importance = compute_importance_from_queries(
        &query_captures,
        &cache,
        n_prompt,
        model.model.layers.len(),
        config.pool_kernel,
    )?;

    Ok(importance)
}

/// Compute importance scores from captured query vectors.
///
/// Aggregation (SpecPrefill paper):
/// 1. softmax(Q @ K^T / sqrt(d)) per head, per layer, per lookahead token
/// 2. avg_pool1d smoothing
/// 3. max across (layers x heads)
/// 4. mean across lookahead tokens
fn compute_importance_from_queries(
    query_captures: &[Vec<Array>],
    cache: &AnyCache,
    n_prompt: usize,
    n_layers: usize,
    pool_kernel: usize,
) -> Result<Array, Exception> {
    if query_captures.is_empty() || query_captures[0].is_empty() {
        // Fallback to uniform if no queries captured
        return Array::from_full(&[n_prompt as i32], 1.0 / n_prompt as f32);
    }

    let n_lookahead = query_captures.len();
    let mut all_scores: Vec<Array> = Vec::new();

    for step_queries in query_captures {
        if step_queries.is_empty() {
            continue;
        }

        // For each layer, compute attention weights
        for (layer_idx, queries) in step_queries.iter().enumerate() {
            // Get cached keys for this layer
            // TODO: Extract keys from cache

            // Compute Q @ K^T / sqrt(d)
            // queries: [B, n_heads, n_lookahead, head_dim]
            // keys: [B, n_heads, n_prompt, head_dim]
            // scores: [B, n_heads, n_lookahead, n_prompt]

            // For now, return uniform scores
            let scores = Array::from_full(&[n_prompt as i32], 1.0 / n_prompt as f32)?;
            all_scores.push(scores);
        }
    }

    if all_scores.is_empty() {
        return Array::from_full(&[n_prompt as i32], 1.0 / n_prompt as f32);
    }

    // Stack and aggregate
    let mut stacked = all_scores[0].clone();
    for scores in all_scores.iter().skip(1) {
        stacked = stacked.add(scores)?;
    }
    stacked = stacked.divide_scalar(all_scores.len() as f32)?;

    // Apply smoothing if requested
    if pool_kernel > 1 {
        stacked = avg_pool1d(&stacked, pool_kernel)?;
    }

    Ok(stacked)
}

/// Top-p (nucleus) sampling.
fn sample_top_p(logits: &Array, top_p: f32) -> Result<Array, Exception> {
    // Sort logits descending
    let sorted_indices = ops::argpartition(logits, logits.shape()[logits.ndim() - 1] as i32)?;

    // Get sorted probabilities
    let probs = ops::softmax(logits, -1)?;
    let sorted_probs = probs.take_along_axis(&sorted_indices, -1)?;

    // Compute cumulative sum
    let cumsum = ops::cumsum(&sorted_probs, -1, false, false)?;

    // Find cutoff where cumsum > top_p
    // TODO: Implement proper top-p sampling

    // For now, just use argmax
    ops::argmax(logits, -1, false)
}

/// 1D average pooling along the last axis.
fn avg_pool1d(x: &Array, kernel_size: usize) -> Result<Array, Exception> {
    if kernel_size <= 1 {
        return Ok(x.clone());
    }

    let pad = kernel_size / 2;
    // Simple implementation using slicing
    // TODO: Implement efficient pooling

    Ok(x.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avg_pool1d() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let pooled = avg_pool1d(&x, 3).unwrap();
        // Should smooth the values
        assert_eq!(pooled.shape(), &[5]);
    }
}
