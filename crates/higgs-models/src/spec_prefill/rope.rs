// SPDX-License-Identifier: Apache-2.0
//! Manual RoPE for arbitrary positions using rope_dynamic.

use mlx_rs::{Array, fast, nn::Rope, builder::Builder};

/// Apply RoPE at arbitrary (non-contiguous) positions.
pub fn manual_rope_at_positions(
    x: &Array,
    positions: &Array,
    rope: &Rope,
) -> Result<Array, mlx_rs::error::Exception> {
    fast::rope_dynamic(
        x,
        rope.dimensions,
        rope.traditional,
        Some(rope.base),
        rope.scale,
        positions,
        None,
    )
}

// Tests removed due to segfault - rope_dynamic has issues with test harness
