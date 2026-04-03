// SPDX-License-Identifier: Apache-2.0
//! Sparse prefill for SpecPrefill.

use crate::AnyCache;
use crate::qwen3_next::{Qwen3NextCausalLM, LayerCache};
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
}
