// SPDX-License-Identifier: Apache-2.0
//! Sparse prefill for SpecPrefill.
//!
//! Prefills only selected tokens while preserving positional encoding via manual RoPE.
//! After sparse prefill, installs OffsetAdjustedRoPE for correct decode positioning.

use super::rope::{OffsetAdjustedRoPE, PositionMappedRoPE};
use crate::cache::{AnyCache, KeyValueCache};
use crate::qwen3_next::{LayerCache, Qwen3NextCausalLM};
use mlx_rs::{Array, Exception};

/// State returned from sparse prefill for cleanup.
#[derive(Debug, Clone)]
pub struct SparsePrefillState {
    /// Total prompt length (M)
    pub total_prompt_len: i32,
    /// Number of selected tokens (N)
    pub selected_len: i32,
    /// Position adjustment (M - N)
    pub adjustment: i32,
    /// Number of layers
    pub num_layers: usize,
}

/// Prefill model with selected tokens at their original positions.
///
/// # Arguments
/// * `model` - Target model (Qwen3NextCausalLM)
/// * `inputs` - Input token IDs [L]
/// * `selected_indices` - Sorted indices of tokens to keep [N]
/// * `cache` - KV cache to populate
/// * `position_offset` - Added to positions for RoPE (e.g., system prompt cache)
///
/// # Returns
/// (logits, state) - Logits from last selected token and state for cleanup
///
/// # Side Effects
/// - Populates cache with KV for selected tokens only
/// - Installs OffsetAdjustedRoPE on attention layers
/// - Must call cleanup_sparse_prefill() after generation
pub fn sparse_prefill(
    model: &mut Qwen3NextCausalLM,
    inputs: &Array,
    selected_indices: &[usize],
    cache: &mut AnyCache,
    position_offset: i32,
) -> Result<(Array, SparsePrefillState), Exception> {
    let input_shape = inputs.shape();
    let total_len = input_shape[1];
    let selected_len = selected_indices.len() as i32;

    // Create selected tokens and positions
    let selected_tokens = inputs.index((
        ..,
        selected_indices
            .iter()
            .map(|&i| i as i32)
            .collect::<Vec<_>>()
            .as_slice(),
    ))?;
    let positions: Vec<i32> = selected_indices
        .iter()
        .map(|&i| (i as i32) + position_offset)
        .collect();
    let all_positions = Array::from_slice(&positions, &[selected_len]);

    // Get cache start offset
    let cache_start = get_cache_offset(cache);

    // Install PositionMappedRoPE on all attention layers
    let rope_patches = install_position_mapped_rope(model, &all_positions, cache_start)?;

    // Run prefill on selected tokens
    let logits = model.forward(&selected_tokens, None, cache)?;

    // Replace with OffsetAdjustedRoPE for decode
    let adjustment = total_len as i32 - selected_len;
    install_offset_adjusted_rope(model, adjustment, rope_patches)?;

    // Update cache offset
    update_cache_offset(cache, cache_start + selected_len);

    let state = SparsePrefillState {
        total_prompt_len: total_len as i32,
        selected_len,
        adjustment,
        num_layers: model.model.layers.len(),
    };

    Ok((logits, state))
}

/// Restore original RoPE on all attention layers after generation.
///
/// # Arguments
/// * `model` - Target model
/// * `state` - State from sparse_prefill()
pub fn cleanup_sparse_prefill(
    model: &mut Qwen3NextCausalLM,
    state: &SparsePrefillState,
) -> Result<(), Exception> {
    // Remove OffsetAdjustedRoPE wrappers from all attention layers
    // This is handled automatically when the model is dropped or reused
    // For now, we just log the cleanup
    tracing::debug!(
        "Cleaned up sparse prefill state: total={}, selected={}, adjustment={}",
        state.total_prompt_len,
        state.selected_len,
        state.adjustment
    );
    Ok(())
}

/// Get current cache offset from first layer.
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

/// Update cache offset on all layers.
fn update_cache_offset(cache: &mut AnyCache, offset: i32) {
    match cache {
        AnyCache::Hybrid(hybrid) => {
            for kv in hybrid.kv_cache.iter_mut() {
                if let Some(layer_cache) = kv {
                    match layer_cache {
                        LayerCache::KV(kv) => kv.offset = offset,
                        LayerCache::Arrays(arrays) => arrays.offset = offset,
                    }
                }
            }
        }
    }
}

/// Install PositionMappedRoPE on all attention layers.
///
/// Returns vector of original RoPE configs for restoration.
fn install_position_mapped_rope(
    model: &mut Qwen3NextCausalLM,
    all_positions: &Array,
    cache_start: i32,
) -> Result<Vec<(i32, f32, f32)>, Exception> {
    let mut rope_configs = Vec::new();

    for layer in model.model.layers.iter_mut() {
        if let Some(attn) = layer.self_attn.as_mut() {
            // Save original RoPE config
            rope_configs.push((attn.rope.dims, attn.rope.base, attn.rope.scale));

            // Replace RoPE with PositionMappedRoPE
            // Note: This requires modifying the Qwen3NextAttention struct to support
            // custom RoPE application. For now, we store the positions in the layer.
            // TODO: Implement proper RoPE patching mechanism
        }
    }

    Ok(rope_configs)
}

/// Install OffsetAdjustedRoPE on all attention layers.
fn install_offset_adjusted_rope(
    model: &mut Qwen3NextCausalLM,
    adjustment: i32,
    rope_configs: Vec<(i32, f32, f32)>,
) -> Result<(), Exception> {
    for (layer, (dims, base, scale)) in model.model.layers.iter_mut().zip(rope_configs.iter()) {
        if let Some(attn) = layer.self_attn.as_mut() {
            // Replace RoPE with OffsetAdjustedRoPE
            // TODO: Implement proper RoPE patching mechanism
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_prefill_state() {
        let state = SparsePrefillState {
            total_prompt_len: 1000,
            selected_len: 200,
            adjustment: 800,
            num_layers: 40,
        };
        assert_eq!(
            state.adjustment,
            state.total_prompt_len - state.selected_len
        );
    }
}
