// SPDX-License-Identifier: Apache-2.0
//! Manual RoPE for arbitrary positions in SpecPrefill.
//!
//! Standard RoPE assumes contiguous positions [0, 1, 2, ..., L-1].
//! SpecPrefill needs RoPE at arbitrary (non-contiguous) positions to preserve
//! the original positional encoding when prefilling only selected tokens.
//!
//! Key insight: RoPE is relative - Q_m @ K_p^T depends only on (m - p).
//! Selected keys stored contiguously with correct RoPE angles produce correct
//! attention during decode.

use mlx_rs::ops;
use mlx_rs::{Array, Dtype, Exception, Stream};

/// Apply RoPE at arbitrary (non-contiguous) positions.
///
/// # Arguments
/// * `x` - Input tensor [B, n_heads, L, head_dim]
/// * `positions` - Position indices [L] (can be non-contiguous)
/// * `dims` - Number of dimensions to rotate (typically head_dim)
/// * `base` - RoPE base frequency (default 10000.0)
/// * `scale` - Position scale divisor (default 1.0)
///
/// # Returns
/// RoPE-rotated tensor with same shape as input
pub fn manual_rope(
    x: &Array,
    positions: &Array,
    dims: i32,
    base: f32,
    scale: f32,
) -> Result<Array, Exception> {
    let half = dims / 2;

    // Compute inverse frequencies: 1 / (base^(2i/dims))
    let inv_freq = ops::arange(0, dims, 2, Dtype::Float32)?
        .divide_scalar(base.ln() / half as f32)?
        .exp()?
        .reciprocal()?;

    // Compute angles: positions * inv_freq
    // positions: [L], inv_freq: [half] -> angles: [L, half]
    let scaled_pos = positions.cast(Dtype::Float32)?.divide_scalar(scale)?;
    let angles = ops::matmul(&scaled_pos.reshape(&[-1, 1])?, &inv_freq.reshape(&[1, -1])?)?;

    // cos/sin of angles
    let cos_a = ops::cos(&angles)?.expand_dims(&[0, 1])?; // [1, 1, L, half]
    let sin_a = ops::sin(&angles)?.expand_dims(&[0, 1])?; // [1, 1, L, half]

    // Split x into rotated and pass-through parts
    let x_shape = x.shape();
    let x_rot = x.index((.., .., .., ..dims))?;
    let x_pass = x.index((.., .., .., dims..))?;

    // Split rotated part into pairs for rotation
    let x1 = x_rot.index((.., .., .., ..half))?;
    let x2 = x_rot.index((.., .., .., half..))?;

    // Apply RoPE: [x1*cos - x2*sin, x1*sin + x2*cos]
    let rotated = ops::concatenate_axis(
        &[
            &x1.multiply(&cos_a)?.subtract(&x2.multiply(&sin_a)?)?,
            &x1.multiply(&sin_a)?.add(&x2.multiply(&cos_a)?)?,
        ],
        -1,
    )?;

    // Concatenate rotated and pass-through parts
    ops::concatenate_axis(&[&rotated, &x_pass], -1)
}

/// Apply RoPE at arbitrary positions using pre-computed frequencies.
///
/// For custom RoPE variants (Llama3, Yarn, SuScaled) that store _freqs.
pub fn manual_rope_with_freqs(
    x: &Array,
    positions: &Array,
    dims: i32,
    freqs: &Array,
    pre_scale: f32,
) -> Result<Array, Exception> {
    let half = dims / 2;

    // Use pre-computed frequencies
    let inv_freq = freqs.cast(Dtype::Float32)?.reciprocal()?;

    // Compute angles
    let scaled_pos = positions.cast(Dtype::Float32)?.divide_scalar(pre_scale)?;
    let angles = ops::matmul(&scaled_pos.reshape(&[-1, 1])?, &inv_freq.reshape(&[1, -1])?)?;

    let cos_a = ops::cos(&angles)?.expand_dims(&[0, 1])?;
    let sin_a = ops::sin(&angles)?.expand_dims(&[0, 1])?;

    let x_rot = x.index((.., .., .., ..dims))?;
    let x_pass = x.index((.., .., .., dims..))?;

    let x1 = x_rot.index((.., .., .., ..half))?;
    let x2 = x_rot.index((.., .., .., half..))?;

    let x1_scaled = if pre_scale != 1.0 {
        x1.multiply_scalar(pre_scale)?
    } else {
        x1
    };
    let x2_scaled = if pre_scale != 1.0 {
        x2.multiply_scalar(pre_scale)?
    } else {
        x2
    };

    let rotated = ops::concatenate_axis(
        &[
            &x1_scaled
                .multiply(&cos_a)?
                .subtract(&x2_scaled.multiply(&sin_a)?)?,
            &x1_scaled
                .multiply(&sin_a)?
                .add(&x2_scaled.multiply(&cos_a)?)?,
        ],
        -1,
    )?;

    ops::concatenate_axis(&[&rotated, &x_pass], -1)
}

/// RoPE wrapper that applies RoPE at mapped positions during sparse prefill.
///
/// Maps cache offset to position array index:
///   positions = all_positions[(offset - cache_start) : (offset - cache_start) + L]
pub struct PositionMappedRoPE {
    dims: i32,
    base: f32,
    scale: f32,
    all_positions: Array,
    cache_start: i32,
    has_custom_freqs: bool,
    freqs: Option<Array>,
    pre_scale: f32,
}

impl PositionMappedRoPE {
    /// Create a position-mapped RoPE wrapper.
    ///
    /// # Arguments
    /// * `dims` - Rotary dimensions
    /// * `base` - RoPE base frequency
    /// * `scale` - Position scale
    /// * `all_positions` - All selected token positions [N]
    /// * `cache_start` - Initial cache offset (for system prompt cache)
    /// * `freqs` - Optional pre-computed frequencies for custom RoPE
    /// * `pre_scale` - Pre-scale factor for custom RoPE variants
    pub fn new(
        dims: i32,
        base: f32,
        scale: f32,
        all_positions: Array,
        cache_start: i32,
        freqs: Option<Array>,
        pre_scale: f32,
    ) -> Self {
        Self {
            dims,
            base,
            scale,
            all_positions,
            cache_start,
            has_custom_freqs: freqs.is_some(),
            freqs,
            pre_scale,
        }
    }

    /// Apply RoPE at mapped positions.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, n_heads, L, head_dim]
    /// * `offset` - Current cache offset
    pub fn apply(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let x_shape = x.shape();
        let L = x_shape[2];

        // Map cache offset to position array index
        let idx = offset - self.cache_start;

        // Select positions for this chunk
        let positions = self.all_positions.index((idx..idx + L as i32))?;

        // Apply manual RoPE
        if self.has_custom_freqs {
            let freqs = self.freqs.as_ref().unwrap();
            manual_rope_with_freqs(x, &positions, self.dims, freqs, self.pre_scale)
        } else {
            manual_rope(x, &positions, self.dims, self.base, self.scale)
        }
    }
}

/// RoPE wrapper that adds constant offset for decode after sparse prefill.
///
/// After sparse prefill of N tokens from M total:
///   cache.offset = N + i, desired position = M + i
///   adjustment = M - N
pub struct OffsetAdjustedRoPE {
    dims: i32,
    base: f32,
    scale: f32,
    adjustment: i32,
    has_custom_freqs: bool,
    freqs: Option<Array>,
    pre_scale: f32,
}

impl OffsetAdjustedRoPE {
    /// Create an offset-adjusted RoPE wrapper.
    ///
    /// # Arguments
    /// * `dims` - Rotary dimensions
    /// * `base` - RoPE base frequency
    /// * `scale` - Position scale
    /// * `adjustment` - Position offset (M - N, total - selected)
    /// * `freqs` - Optional pre-computed frequencies for custom RoPE
    /// * `pre_scale` - Pre-scale factor for custom RoPE variants
    pub fn new(
        dims: i32,
        base: f32,
        scale: f32,
        adjustment: i32,
        freqs: Option<Array>,
        pre_scale: f32,
    ) -> Self {
        Self {
            dims,
            base,
            scale,
            adjustment,
            has_custom_freqs: freqs.is_some(),
            freqs,
            pre_scale,
        }
    }

    /// Apply RoPE with offset adjustment.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, n_heads, L, head_dim]
    /// * `offset` - Current cache offset
    pub fn apply(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let adjusted_offset = offset + self.adjustment;
        let L = x.shape()[2];

        // Create contiguous positions with adjustment
        let positions = ops::arange(adjusted_offset, adjusted_offset + L as i32, 1, Dtype::Int32)?;

        // Apply manual RoPE
        if self.has_custom_freqs {
            let freqs = self.freqs.as_ref().unwrap();
            manual_rope_with_freqs(x, &positions, self.dims, freqs, self.pre_scale)
        } else {
            manual_rope(x, &positions, self.dims, self.base, self.scale)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_rope_shape() {
        let x = Array::ones::<f32>(&[1, 2, 4, 64]).unwrap();
        let positions = Array::from_slice(&[0i32, 1, 2, 3], &[4]);
        let result = manual_rope(&x, &positions, 64, 10000.0, 1.0).unwrap();
        assert_eq!(result.shape(), &[1, 2, 4, 64]);
    }

    #[test]
    fn test_manual_rope_non_contiguous() {
        let x = Array::ones::<f32>(&[1, 2, 2, 64]).unwrap();
        let positions = Array::from_slice(&[0i32, 100], &[2]);
        let result = manual_rope(&x, &positions, 64, 10000.0, 1.0).unwrap();
        assert_eq!(result.shape(), &[1, 2, 2, 64]);
    }

    #[test]
    fn test_position_mapped_rope() {
        let all_positions = Array::from_slice(&[0i32, 10, 20, 30], &[4]);
        let rope = PositionMappedRoPE::new(64, 10000.0, 1.0, all_positions, 0, None, 1.0);

        let x = Array::ones::<f32>(&[1, 2, 2, 64]).unwrap();
        let result = rope.apply(&x, 0).unwrap();
        assert_eq!(result.shape(), &[1, 2, 2, 64]);
    }

    #[test]
    fn test_offset_adjusted_rope() {
        let rope = OffsetAdjustedRoPE::new(64, 10000.0, 1.0, 100, None, 1.0);

        let x = Array::ones::<f32>(&[1, 2, 1, 64]).unwrap();
        let result = rope.apply(&x, 0).unwrap();
        assert_eq!(result.shape(), &[1, 2, 1, 64]);
    }
}
