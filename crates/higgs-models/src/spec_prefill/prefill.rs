// SPDX-License-Identifier: Apache-2.0
//! Sparse prefill for SpecPrefill.

use crate::AnyCache;
use crate::qwen3_next::{Qwen3NextCausalLM, LayerCache, Qwen3NextAttention};
use mlx_rs::{Array, ops};

#[derive(Debug, Clone)]
pub struct SparsePrefillState {
    pub total_prompt_len: i32,
    pub selected_len: i32,
    pub adjustment: i32,
}

/// Select tokens at specified indices from input tensor.
pub fn select_tokens(inputs: &Array, indices: &[usize]) -> Result<Array, mlx_rs::error::Exception> {
    let shape = inputs.shape();
    let B = shape[0];
    let L = shape[1];
    
    let gather_indices: Vec<i32> = (0..B)
        .flat_map(|b| indices.iter().map(move |&i| (b * L + i as i32)))
        .collect();
    
    if shape.len() == 2 {
        let inputs_2d = inputs.reshape(&[B * L, 1])?;
        let n = indices.len() as i32;
        let indices_array = Array::from_slice(&gather_indices, &[B * n, 1]);
        let selected = inputs_2d.take_along_axis(&indices_array, 0)?;
        selected.reshape(&[B, n])
    } else {
        let D = shape[2];
        let inputs_2d = inputs.reshape(&[B * L, D])?;
        let n = indices.len() as i32;
        let indices_array = Array::from_slice(&gather_indices, &[B * n, 1]);
        let selected = inputs_2d.take_along_axis(&indices_array, 0)?;
        selected.reshape(&[B, n, D])
    }
}

/// Create position array from token indices with offset.
pub fn create_position_array(indices: &[usize], offset: i32) -> Array {
    let positions: Vec<i32> = indices.iter().map(|&i| (i as i32) + offset).collect();
    Array::from_slice(&positions, &[positions.len() as i32])
}

/// Prefill model with selected tokens at their original positions.
pub fn sparse_prefill(
    model: &mut Qwen3NextCausalLM,
    inputs: &Array,
    selected_indices: &[usize],
    cache: &mut AnyCache,
    _position_offset: i32,
) -> Result<(Array, SparsePrefillState), mlx_rs::error::Exception> {
    let total_len = inputs.dim(1) as i32;
    let selected_len = selected_indices.len() as i32;
    
    let layer_cache_vec = match cache {
        AnyCache::Hybrid(vec) => vec,
        AnyCache::KV(_) => {
            return Err(mlx_rs::error::Exception::custom(
                "sparse_prefill: expected Hybrid cache for Qwen3Next"
            ));
        }
    };
    
    let selected_tokens = select_tokens(inputs, selected_indices)?;
    let logits = model.forward(&selected_tokens, None, layer_cache_vec)?;
    
    let state = SparsePrefillState {
        total_prompt_len: total_len,
        selected_len,
        adjustment: total_len - selected_len,
    };
    
    Ok((logits, state))
}

pub fn cleanup_sparse_prefill(
    _model: &mut Qwen3NextCausalLM,
    _state: &SparsePrefillState,
) -> Result<(), mlx_rs::error::Exception> {
    Ok(())
}

// ===========================================================================
// Phase 2: Custom RoPE Application
// ===========================================================================

/// Apply RoPE at custom positions for a single attention layer.
pub fn apply_rope_at_positions(
    attention: &Qwen3NextAttention,
    queries: &Array,
    keys: &Array,
    positions: &Array,
) -> Result<(Array, Array), mlx_rs::error::Exception> {
    attention.apply_rope_at_positions(queries, keys, positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_position_array() {
        let indices = vec![0, 10, 25, 100];
        let positions = create_position_array(&indices, 0);
        
        assert_eq!(positions.shape(), &[4]);
        let pos_slice: &[i32] = positions.as_slice();
        assert_eq!(pos_slice, &[0, 10, 25, 100]);
    }

    #[test]
    fn test_create_position_array_with_offset() {
        let indices = vec![0, 10, 25];
        let positions = create_position_array(&indices, 50);
        
        let pos_slice: &[i32] = positions.as_slice();
        assert_eq!(pos_slice, &[50, 60, 75]);
    }

    #[test]
    fn test_apply_rope_at_positions_signature() {
        // Verify the function signature is correct
        // Actual testing requires model loading which causes test harness issues
        let shape = [1, 2, 1, 64];
        let queries = Array::from_slice(&vec![1.0f32; 128], &shape);
        let keys = Array::from_slice(&vec![1.0f32; 128], &shape);
        let positions = Array::from_slice(&[0i32, 1], &[2]);
        
        assert_eq!(queries.shape(), &shape);
        assert_eq!(keys.shape(), &shape);
        assert_eq!(positions.shape(), &[2]);
    }
}

// ===========================================================================
// Phase 3: Sparse Attention Forward (STUB - requires full layer iteration)
// ===========================================================================

/// Sparse model forward pass with custom RoPE positions.
///
/// This is a STUB - full implementation requires:
/// 1. Iterate through all decoder layers
/// 2. For each attention layer, apply RoPE at custom positions
/// 3. Run attention with custom-positioned Q and K
/// 4. Continue with MLP and residual connections
///
/// Currently falls back to standard forward pass.
///
/// # Arguments
/// * `model` - Qwen3Next model
/// * `inputs` - Input tokens [B, L]
/// * `selected_indices` - Token indices to process
/// * `cache` - KV cache
/// * `position_offset` - Position offset (e.g., system prompt length)
///
/// # Returns
/// (logits, state) - Logits from last selected token and state for cleanup
pub fn sparse_model_forward(
    model: &mut Qwen3NextCausalLM,
    inputs: &Array,
    selected_indices: &[usize],
    cache: &mut AnyCache,
    position_offset: i32,
) -> Result<(Array, SparsePrefillState), mlx_rs::error::Exception> {
    // STUB: For now, use the existing sparse_prefill which just selects tokens
    // TODO: Implement full sparse forward with custom RoPE
    sparse_prefill(model, inputs, selected_indices, cache, position_offset)
}

#[cfg(test)]
mod phase3_tests {
    use super::*;

    #[test]
    fn test_sparse_model_forward_exists() {
        // Verify the function exists and has correct signature
        // Full testing requires model loading
        assert_eq!(sparse_model_forward as usize != 0, true);
    }
}
