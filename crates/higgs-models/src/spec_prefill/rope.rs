// SPDX-License-Identifier: Apache-2.0
//! Manual RoPE for arbitrary positions using rope_dynamic.

use mlx_rs::{Array, fast, nn::Rope};

/// Apply RoPE at arbitrary (non-contiguous) positions.
///
/// # Arguments
/// * `x` - Input tensor [B, n_heads, L, head_dim]
/// * `positions` - Position indices [L] (can be non-contiguous)
/// * `rope` - RoPE module with dimensions, base, scale
///
/// # Returns
/// RoPE-rotated tensor with same shape as input
pub fn manual_rope_at_positions(
    x: &Array,
    positions: &Array,
    rope: &Rope,
) -> Result<Array, mlx_rs::error::Exception> {
    // x: [B, n_heads, L, head_dim]
    // positions: [L]
    // Use rope_dynamic with positions array
    fast::rope_dynamic(
        x,
        rope.dimensions,
        rope.traditional,
        Some(rope.base),
        rope.scale,
        positions,  // Array of positions!
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::nn::RopeBuilder;

    #[test]
    fn test_manual_rope_at_positions() {
        let rope = RopeBuilder::new(64).build().unwrap();
        let x = Array::from_iter(vec![1.0f32; 128], &[1, 2, 1, 64]);
        let positions = Array::from_slice(&[0i32, 100], &[2]);
        
        let result = manual_rope_at_positions(&x, &positions, &rope);
        assert!(result.is_ok());
    }
}
