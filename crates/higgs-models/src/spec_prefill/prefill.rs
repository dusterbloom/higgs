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

/// Prefill model with selected tokens at their original positions.
pub fn sparse_prefill(
    model: &mut Qwen3NextCausalLM,
    inputs: &Array,
    selected_indices: &[usize],
    cache: &mut AnyCache,
    position_offset: i32,
) -> Result<(Array, SparsePrefillState), mlx_rs::error::Exception> {
    let total_len = inputs.dim(1) as i32;
    let selected_len = selected_indices.len() as i32;
    
    // Extract Vec<Option<LayerCache>> from AnyCache
    let layer_cache_vec = match cache {
        AnyCache::Hybrid(vec) => vec,
        AnyCache::KV(_) => {
            return Err(mlx_rs::error::Exception::custom(
                "sparse_prefill: expected Hybrid cache for Qwen3Next"
            ));
        }
    };
    
    // Extract selected tokens using take_along_axis
    let indices_array = Array::from_iter(
        selected_indices.iter().map(|&i| i as i32),
        &[1, selected_len]
    );
    
    // Reshape inputs to [B*L, D] then gather
    let shape = inputs.shape();
    let B = shape[0];
    let D = shape[shape.len() - 1];
    let inputs_2d = inputs.reshape(&[B * total_len, D])?;
    
    // Create indices for gather
    let gather_indices: Vec<i32> = (0..B)
        .flat_map(|b| selected_indices.iter().map(move |&i| (b * total_len + i as i32)))
        .collect();
    
    let indices_flat = Array::from_iter(gather_indices, &[B * selected_len, 1]);
    let selected_tokens = inputs_2d.take_along_axis(&indices_flat, 0)?;
    let selected_tokens = selected_tokens.reshape(&[B, selected_len, D])?;
    
    // Forward pass with selected tokens
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
