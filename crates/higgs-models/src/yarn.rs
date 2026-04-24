//! YaRN RoPE helpers shared across models.
//!
//! Extracted verbatim from the original site in `deepseek_v2.rs`. The
//! `apply_yarn_rope` wrapper adds a `traditional` flag so Qwen3-family models
//! (Bonsai) can reuse the same freq precomputation with `traditional=false`,
//! while DeepSeek stays on `traditional=true`.

use std::f32::consts::PI;

use mlx_rs::{Array, error::Exception, fast};

fn yarn_find_correction_dim(num_rotations: f32, dim: i32, base: f32, max_pos: i32) -> f32 {
    let dim_f = f32::from(i16::try_from(dim).unwrap_or(i16::MAX));
    let max_pos_f = f32::from(i16::try_from(max_pos).unwrap_or(i16::MAX));
    (dim_f * (max_pos_f / (num_rotations * 2.0 * PI)).ln()) / (2.0 * base.ln())
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn yarn_find_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: i32,
    base: f32,
    max_pos: i32,
) -> (i32, i32) {
    let low = yarn_find_correction_dim(low_rot, dim, base, max_pos).floor() as i32;
    let high = yarn_find_correction_dim(high_rot, dim, base, max_pos).ceil() as i32;
    (low.max(0), high.min(dim - 1))
}

pub(crate) fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        (0.1 * mscale).mul_add(scale.ln(), 1.0)
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing
)]
pub(crate) fn compute_yarn_freqs(
    dim: i32,
    base: f32,
    scaling_factor: f32,
    orig_max_pos: i32,
    beta_fast: f32,
    beta_slow: f32,
) -> Array {
    let half_dim = dim / 2;
    let dim_f = f32::from(i16::try_from(dim).unwrap_or(i16::MAX));

    let mut freq_extra = Vec::with_capacity(half_dim as usize);
    let mut freq_inter = Vec::with_capacity(half_dim as usize);
    for i in 0..half_dim {
        let exp = f32::from(i16::try_from(2 * i).unwrap_or(0)) / dim_f;
        let theta = base.powf(exp);
        freq_extra.push(theta);
        freq_inter.push(scaling_factor * theta);
    }

    let (low, high) = yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max_pos);

    let low_f = f32::from(i16::try_from(low).unwrap_or(0));
    let high_f = f32::from(i16::try_from(high).unwrap_or(0));
    let range = if (high_f - low_f).abs() < 0.001 {
        high_f - low_f + 0.001
    } else {
        high_f - low_f
    };

    let mut freqs = Vec::with_capacity(half_dim as usize);
    for i in 0..half_dim as usize {
        let idx_f = f32::from(i16::try_from(i).unwrap_or(0));
        let ramp = ((idx_f - low_f) / range).clamp(0.0, 1.0);
        let mask = 1.0 - ramp;
        let inter = freq_inter[i];
        let extra = freq_extra[i];
        let denom = inter * mask + extra * (1.0 - mask);
        freqs.push((inter * extra) / denom);
    }

    Array::from_slice(&freqs, &[half_dim])
}

/// Apply YaRN-scaled RoPE.
///
/// When `mscale != 1.0`, inputs are pre-scaled before rotation (matches the
/// DeepSeek reference). `traditional=false` matches the Qwen3 / LLaMA rope
/// layout; `traditional=true` matches DeepSeek's packed complex layout.
pub(crate) fn apply_yarn_rope(
    x: &Array,
    dim: i32,
    base: f32,
    yarn_freqs: Option<&Array>,
    mscale: f32,
    offset: i32,
    traditional: bool,
) -> Result<Array, Exception> {
    let x_scaled = if (mscale - 1.0).abs() > f32::EPSILON {
        x.multiply(mlx_rs::array!(mscale))?
    } else {
        x.clone()
    };
    yarn_freqs.map_or_else(
        || {
            fast::rope(
                &x_scaled,
                dim,
                traditional,
                base,
                1.0,
                offset,
                None::<&Array>,
            )
        },
        |freqs| {
            fast::rope(
                &x_scaled,
                dim,
                traditional,
                None::<f32>,
                1.0,
                offset,
                Some(freqs),
            )
        },
    )
}
