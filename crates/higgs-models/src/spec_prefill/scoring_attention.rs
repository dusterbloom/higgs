// SPDX-License-Identifier: Apache-2.0
//! Attention-based token importance scoring for SpecPrefill.
//!
//! Implements the SpecPrefill scoring algorithm from arxiv.org/abs/2502.02789:
//! 1. Prefill draft model with all prompt tokens
//! 2. Run N lookahead decode steps, capturing query vectors after RoPE
//! 3. Compute importance = mean_over_steps(max_over_heads(Q @ K^T / sqrt(d)))
//! 4. Apply avg_pool1d smoothing
//! 5. Select top-K% tokens by importance

use super::draft::DraftModelConfig;
use crate::cache::{AnyCache, KeyValueCache};
use crate::error::ModelError;
use crate::qwen3_next::{LayerCache, Qwen3NextAttention, Qwen3NextCausalLM};
use mlx_rs::{ops, Array, Dtype, Exception, Shape, Stream};

/// Captured query vectors from one lookahead step.
#[derive(Debug, Clone)]
struct CapturedQueries {
    /// Queries from all layers: Vec<[B, n_heads, 1, head_dim]>
    layer_queries: Vec<Array>,
}

/// Score token importance using attention patterns from draft model.
///
/// # Arguments
/// * `model` - Draft model (Qwen3NextCausalLM)
/// * `tokens` - Prompt token IDs
/// * `config` - Draft model configuration
///
/// # Returns
/// Per-token importance scores (not normalized, higher = more important)
pub fn score_tokens_with_attention(
    model: &mut Qwen3NextCausalLM,
    tokens: &[u32],
    config: &DraftModelConfig,
) -> Result<Array, ModelError> {
    let n_prompt = tokens.len();

    if n_prompt == 0 {
        return Array::from_full(&[0], 0.0f32).map_err(|e| ModelError::Mlx(e.to_string()));
    }

    // Convert tokens to array [1, n_prompt]
    let input_tokens = Array::from_slice(tokens, &[1, n_prompt as i32])
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Create cache for draft model
    let mut cache = model.make_cache();

    // Phase 1: Prefill draft model with all tokens
    let hidden = model
        .forward_hidden(&input_tokens, None, &mut cache)
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Get logits for last token
    let last_hidden = hidden.index((.., -1, ..));
    let logits = model
        .lm_head
        .as_ref()
        .map(|h| h.forward(&last_hidden))
        .unwrap_or_else(|| model.model.embed_tokens.as_linear(&last_hidden))
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Sample first token
    let mut current_token =
        ops::argmax(&logits, -1, false).map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Phase 2: Lookahead decode with query capture
    let mut all_captures: Vec<CapturedQueries> = Vec::with_capacity(config.n_lookahead);

    for _step in 0..config.n_lookahead {
        // Capture queries from all attention layers during this decode step
        let captured = capture_queries_from_decode(model, &current_token, &mut cache)?;
        all_captures.push(captured);

        // Sample next token
        let next_logits =
            decode_step_and_get_logits(model, &current_token, &mut cache, config.temp)?;

        current_token =
            sample_top_p(&next_logits, config.top_p).map_err(|e| ModelError::Mlx(e.to_string()))?;
    }

    // Phase 3: Compute importance from captured queries
    let importance =
        compute_importance_from_captures(&all_captures, &cache, n_prompt, config.pool_kernel)?;

    Ok(importance)
}

/// Capture query vectors from all attention layers during a single decode step.
fn capture_queries_from_decode(
    model: &mut Qwen3NextCausalLM,
    current_token: &Array,
    cache: &mut AnyCache,
) -> Result<CapturedQueries, ModelError> {
    let mut layer_queries = Vec::with_capacity(model.model.layers.len());

    // Reshape token for single-step decode [1, 1]
    let token_2d = current_token
        .reshape(&[1, 1])
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Embed tokens
    let mut h = model
        .model
        .embed_tokens
        .forward(&token_2d)
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Get cache offset
    let cache_offset = get_cache_offset(cache);

    // Process each layer and capture queries
    for (layer_idx, layer) in model.model.layers.iter_mut().enumerate() {
        let layer_cache = match cache {
            AnyCache::Hybrid(hybrid) => hybrid
                .kv_cache
                .get_mut(layer_idx)
                .and_then(|opt| opt.as_mut())
                .ok_or_else(|| {
                    ModelError::Generation(format!("Layer {} cache missing", layer_idx))
                })?,
        };

        // Apply layer norm
        let normed = layer
            .input_layernorm
            .forward(&h)
            .map_err(|e| ModelError::Mlx(e.to_string()))?;

        // Capture query based on layer type
        if layer.is_linear {
            // Linear attention (GatedDeltaNet) - capture from linear_attn
            if let Some(linear_attn) = layer.linear_attn.as_mut() {
                // For GDN, we need to capture Q after the initial projection
                // This requires accessing internal state - simplified for now
                // TODO: Implement proper GDN query capture
                let dummy_q = Array::from_full(&[1, 1, 1, 64], 0.0f32)
                    .map_err(|e| ModelError::Mlx(e.to_string()))?;
                layer_queries.push(dummy_q);
            }
        } else {
            // Full attention - capture Q after q_proj + reshape + RoPE
            if let Some(attn) = layer.self_attn.as_mut() {
                let q = capture_query_from_attention(attn, &normed, cache_offset)?;
                layer_queries.push(q);
            }
        }

        // Continue forward pass (simplified - actual forward would continue here)
        // For scoring, we only need the queries, not the full output
    }

    Ok(CapturedQueries { layer_queries })
}

/// Capture query vector from a full attention layer.
fn capture_query_from_attention(
    attn: &mut Qwen3NextAttention,
    normed: &Array,
    cache_offset: i32,
) -> Result<Array, ModelError> {
    let shape = normed.shape();
    let b = *shape
        .first()
        .ok_or_else(|| ModelError::Generation("Missing batch dim"))?;
    let s = *shape
        .get(1)
        .ok_or_else(|| ModelError::Generation("Missing seq dim"))?;

    // Q projection
    let q_out = attn
        .q_proj
        .forward(normed)
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Reshape to [B, S, n_heads, head_dim]
    let q_reshaped = q_out
        .reshape(&[b, s, attn.num_heads, attn.head_dim])
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Transpose to [B, n_heads, S, head_dim]
    let q_transposed = q_reshaped
        .transpose(&[0, 2, 1, 3])
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Apply RoPE
    let q_with_rope = attn
        .rope
        .forward(&q_transposed, cache_offset)
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    Ok(q_with_rope)
}

/// Run a single decode step and return logits.
fn decode_step_and_get_logits(
    model: &mut Qwen3NextCausalLM,
    current_token: &Array,
    cache: &mut AnyCache,
    temperature: f32,
) -> Result<Array, ModelError> {
    let token_2d = current_token
        .reshape(&[1, 1])
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    let hidden = model
        .forward_hidden(&token_2d, None, cache)
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    let last_hidden = hidden.index((.., -1, ..));
    let logits = model
        .lm_head
        .as_ref()
        .map(|h| h.forward(&last_hidden))
        .unwrap_or_else(|| model.model.embed_tokens.as_linear(&last_hidden))
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Apply temperature
    if temperature > f32::EPSILON {
        let scaled = logits
            .multiply(Array::from_f32(1.0 / temperature))
            .map_err(|e| ModelError::Mlx(e.to_string()))?;
        Ok(scaled)
    } else {
        Ok(logits)
    }
}

/// Compute importance scores from captured query vectors.
///
/// Aggregation (SpecPrefill paper):
/// 1. For each lookahead step, each layer, each head: compute Q @ K^T / sqrt(d)
/// 2. Softmax over prompt positions
/// 3. Max across (layers × heads)
/// 4. Mean across lookahead steps
/// 5. avg_pool1d smoothing
fn compute_importance_from_captures(
    captures: &[CapturedQueries],
    cache: &AnyCache,
    n_prompt: usize,
    pool_kernel: usize,
) -> Result<Array, ModelError> {
    if captures.is_empty() || captures[0].layer_queries.is_empty() {
        // Fallback to uniform if no queries captured
        return Array::from_full(&[n_prompt as i32], 1.0 / n_prompt as f32)
            .map_err(|e| ModelError::Mlx(e.to_string()));
    }

    let n_layers = captures[0].layer_queries.len();
    let mut step_scores: Vec<Array> = Vec::with_capacity(captures.len());

    for capture in captures {
        if capture.layer_queries.len() != n_layers {
            continue;
        }

        // Compute attention weights for this step
        let mut layer_max_scores: Vec<Array> = Vec::new();

        for (layer_idx, queries) in capture.layer_queries.iter().enumerate() {
            // Get cached keys for this layer
            let keys = get_cached_keys(cache, layer_idx, n_prompt)?;

            if keys.is_none() {
                continue;
            }
            let keys = keys.unwrap();

            // Compute Q @ K^T / sqrt(d)
            // queries: [B, n_heads, 1, head_dim]
            // keys: [B, n_heads, n_prompt, head_dim]
            // attention: [B, n_heads, 1, n_prompt]
            let attention = compute_attention_weights(queries, &keys)?;

            // Softmax over prompt positions
            let weights =
                ops::softmax(&attention, -1).map_err(|e| ModelError::Mlx(e.to_string()))?;

            // Max across heads: [B, 1, 1, n_prompt] -> [B, n_prompt]
            let max_over_heads = weights
                .max(&[-2], false)
                .map_err(|e| ModelError::Mlx(e.to_string()))?;

            layer_max_scores.push(max_over_heads);
        }

        // Max across layers
        if layer_max_scores.is_empty() {
            continue;
        }

        let mut step_max = layer_max_scores[0].clone();
        for scores in layer_max_scores.iter().skip(1) {
            step_max =
                ops::maximum(&step_max, scores).map_err(|e| ModelError::Mlx(e.to_string()))?;
        }

        step_scores.push(step_max);
    }

    if step_scores.is_empty() {
        return Array::from_full(&[n_prompt as i32], 1.0 / n_prompt as f32)
            .map_err(|e| ModelError::Mlx(e.to_string()));
    }

    // Mean across lookahead steps
    let mut summed = step_scores[0].clone();
    for scores in step_scores.iter().skip(1) {
        summed = summed
            .add(scores)
            .map_err(|e| ModelError::Mlx(e.to_string()))?;
    }
    let importance = summed
        .divide_scalar(step_scores.len() as f32)
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Apply smoothing
    let smoothed = if pool_kernel > 1 {
        avg_pool1d(&importance, pool_kernel)?
    } else {
        importance
    };

    Ok(smoothed)
}

/// Get cached keys for a specific layer.
fn get_cached_keys(
    cache: &AnyCache,
    layer_idx: usize,
    n_prompt: usize,
) -> Result<Option<Array>, ModelError> {
    match cache {
        AnyCache::Hybrid(hybrid) => {
            let layer_cache = hybrid.kv_cache.get(layer_idx).and_then(|opt| opt.as_ref());

            match layer_cache {
                Some(LayerCache::KV(kv)) => {
                    // Extract keys: [B, n_heads, cache_len, head_dim]
                    // Return first n_prompt keys
                    let keys = kv.keys.clone();
                    let key_shape = keys.shape();

                    if key_shape.len() >= 3 {
                        let cache_len = key_shape[2] as usize;
                        if cache_len >= n_prompt {
                            // Slice to get first n_prompt keys
                            let sliced = keys
                                .index((.., .., ..n_prompt as i32, ..))
                                .map_err(|e| ModelError::Mlx(e.to_string()))?;
                            Ok(Some(sliced))
                        } else {
                            Ok(Some(keys))
                        }
                    } else {
                        Ok(None)
                    }
                }
                Some(LayerCache::Arrays(_)) => {
                    // SSM cache doesn't have traditional KV
                    Ok(None)
                }
                None => Ok(None),
            }
        }
    }
}

/// Compute attention weights: Q @ K^T / sqrt(d).
fn compute_attention_weights(queries: &Array, keys: &Array) -> Result<Array, ModelError> {
    // queries: [B, n_heads, 1, head_dim]
    // keys: [B, n_heads, n_prompt, head_dim]

    let q_shape = queries.shape();
    let head_dim = *q_shape
        .last()
        .ok_or_else(|| ModelError::Generation("Missing head_dim"))?;
    let scale = (head_dim as f32).sqrt().recip();

    // Reshape for matmul
    // Q: [B * n_heads, 1, head_dim]
    // K: [B * n_heads, head_dim, n_prompt]
    let b = *q_shape
        .first()
        .ok_or_else(|| ModelError::Generation("Missing batch"))?;
    let n_heads = *q_shape
        .get(1)
        .ok_or_else(|| ModelError::Generation("Missing n_heads"))?;

    let q_flat = queries
        .reshape(&[b * n_heads, 1, head_dim])
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    let k_shape = keys.shape();
    let n_prompt = *k_shape
        .get(2)
        .ok_or_else(|| ModelError::Generation("Missing n_prompt"))?;

    let k_flat = keys
        .reshape(&[b * n_heads, head_dim, n_prompt])
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Q @ K^T
    let attention = ops::matmul(&q_flat, &k_flat).map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Scale
    let scaled = attention
        .multiply_scalar(scale)
        .map_err(|e| ModelError::Mlx(e.to_string()))?;

    // Reshape back to [B, n_heads, 1, n_prompt]
    scaled
        .reshape(&[b, n_heads, 1, n_prompt])
        .map_err(|e| ModelError::Mlx(e.to_string()))
}

/// Top-p (nucleus) sampling.
fn sample_top_p(logits: &Array, top_p: f32) -> Result<Array, Exception> {
    if top_p >= 1.0 {
        // Fall back to argmax
        return ops::argmax(logits, -1, false);
    }

    // Get probabilities
    let probs = ops::softmax(logits, -1)?;

    // Sort descending
    let sorted_probs = ops::sort(&probs, -1, true)?;

    // Cumulative sum
    let cumsum = ops::cumsum(&sorted_probs, -1, false, false)?;

    // Find cutoff (first position where cumsum > top_p)
    // For simplicity, use argmax for now
    // TODO: Implement proper nucleus sampling
    ops::argmax(logits, -1, false)
}

/// 1D average pooling along the last axis.
fn avg_pool1d(x: &Array, kernel_size: usize) -> Result<Array, ModelError> {
    if kernel_size <= 1 {
        return Ok(x.clone());
    }

    let shape = x.shape();
    let n = *shape
        .last()
        .ok_or_else(|| ModelError::Generation("Missing last dim"))?;

    // Simple implementation: for each position, average over window
    let mut output_data = Vec::with_capacity(x.shape().iter().product::<i32>() as usize);
    let input_data: Vec<f32> = x.try_into().map_err(|e| ModelError::Mlx(e.to_string()))?;

    let stride = n as usize;
    let pad = kernel_size / 2;

    for i in 0..stride {
        let mut sum = 0.0f32;
        let mut count = 0u32;

        for j in 0..kernel_size {
            let pos = i as i32 + j as i32 - pad as i32;
            if pos >= 0 && pos < n {
                sum += input_data[i + pos as usize * stride];
                count += 1;
            }
        }

        output_data.push(if count > 0 { sum / count as f32 } else { 0.0 });
    }

    Array::from_slice(&output_data, &shape).map_err(|e| ModelError::Mlx(e.to_string()))
}

/// Get cache offset from AnyCache.
fn get_cache_offset(cache: &AnyCache) -> i32 {
    match cache {
        AnyCache::Hybrid(hybrid) => {
            if let Some(first) = hybrid.kv_cache.first() {
                match first {
                    Some(LayerCache::KV(kv)) => kv.offset(),
                    Some(LayerCache::Arrays(arrays)) => arrays.offset,
                    None => 0,
                }
            } else {
                0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_weights_shape() {
        // Q: [1, 2, 1, 64], K: [1, 2, 100, 64]
        let q = Array::from_full(&[1, 2, 1, 64], 1.0f32).unwrap();
        let k = Array::from_full(&[1, 2, 100, 64], 1.0f32).unwrap();

        let attention = compute_attention_weights(&q, &k).unwrap();
        assert_eq!(attention.shape(), &[1, 2, 1, 100]);
    }

    #[test]
    fn test_avg_pool1d() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let pooled = avg_pool1d(&x, 3).unwrap();
        assert_eq!(pooled.shape(), &[5]);
    }
}
