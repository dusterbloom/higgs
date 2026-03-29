//! MIL (Model Intermediate Language) program generators for ANE.
//!
//! Generates MIL program text strings that get compiled to ANE binaries.
//! Only contains generators needed for RWKV-7 inference.
//!
//! Feature-gated behind `ane`.

#![cfg(feature = "ane")]

use std::fmt::Write;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// ANE SRAM capacity in fp16 elements (~28 MB).
/// When a single kernel's working set exceeds this, throughput drops ~30%.
const ANE_SRAM_FP16_ELEMS: usize = 14_000_000;

/// MIL program header (version 1.3, matching CoreML toolchain).
const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main(
"#;

// ---------------------------------------------------------------------------
// SRAM tiling
// ---------------------------------------------------------------------------

/// Plan for tiling a large matmul to fit in ANE SRAM.
#[derive(Debug, Clone)]
pub struct TilePlan {
    pub tile_size: usize,
    pub n_tiles: usize,
    pub last_actual: usize,
    pub full_size: usize,
}

impl TilePlan {
    pub fn needs_tiling(&self) -> bool {
        self.n_tiles > 1
    }
}

/// Compute output-channel tiling so each tile fits in SRAM.
///
/// Layout: `[1, channels, 1, spatial]` where `spatial = seq_len + tile_oc`.
/// Constraint: `ic * (seq_len + tile_oc) <= ANE_SRAM_FP16_ELEMS`.
pub fn compute_oc_tile_plan(ic: usize, oc: usize, seq_len: usize) -> TilePlan {
    let budget = ANE_SRAM_FP16_ELEMS / ic;
    if budget <= seq_len {
        // Even a single column doesn't fit; fall back to full size.
        return TilePlan {
            tile_size: oc,
            n_tiles: 1,
            last_actual: oc,
            full_size: oc,
        };
    }
    let max_tile = budget - seq_len;
    // Round down to multiple of 128 for ANE alignment.
    let tile = (max_tile / 128) * 128;
    let tile = tile.max(128).min(oc);

    let n_tiles = (oc + tile - 1) / tile;
    let last = oc - (n_tiles - 1) * tile;

    TilePlan {
        tile_size: tile,
        n_tiles,
        last_actual: last,
        full_size: oc,
    }
}

/// Compute input-channel (reduction dimension) tiling.
pub fn compute_ic_tile_plan(ic: usize, oc: usize, seq_len: usize) -> TilePlan {
    let budget = ANE_SRAM_FP16_ELEMS / (seq_len + oc);
    let tile = (budget / 128) * 128;
    let tile = tile.max(128).min(ic);

    let n_tiles = (ic + tile - 1) / tile;
    let last = ic - (n_tiles - 1) * tile;

    TilePlan {
        tile_size: tile,
        n_tiles,
        last_actual: last,
        full_size: ic,
    }
}

// ---------------------------------------------------------------------------
// MIL config
// ---------------------------------------------------------------------------

/// Configuration for generating RWKV-7 MIL programs.
#[derive(Debug, Clone)]
pub struct MilConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub seq_len: usize,
}

// ---------------------------------------------------------------------------
// MIL generators
// ---------------------------------------------------------------------------

/// Generate a MIL program for a matmul projection: `y = x @ W.T`.
///
/// Uses the DynMatmul pattern: weights packed into the IOSurface spatial
/// dimension alongside activations. Layout: `[1, ic, 1, seq_len + oc]`.
/// Activations at `[0:seq_len]`, weights at `[seq_len:seq_len+oc]`.
pub fn gen_matmul_program(ic: usize, oc: usize, seq_len: usize) -> (String, usize, usize) {
    let spatial = seq_len + oc;
    let input_bytes = ic * spatial * 2; // fp16
    let output_bytes = oc * seq_len * 2;

    let mut mil = String::with_capacity(2048);
    write!(
        mil,
        r#"{MIL_HEADER}    %input: tensor<[1, {ic}, 1, {spatial}], fp16>
  ) {{
    %act = slice_by_size(x = %input, begin = [0, 0, 0, 0], size = [1, {ic}, 1, {seq_len}]);
    %wt = slice_by_size(x = %input, begin = [0, 0, 0, {seq_len}], size = [1, {ic}, 1, {oc}]);
    %act_r = reshape(x = %act, shape = [1, {ic}, {seq_len}]);
    %wt_r = reshape(x = %wt, shape = [1, {ic}, {oc}]);
    %wt_t = transpose(x = %wt_r, perm = [0, 2, 1]);
    %y = matmul(x = %wt_t, y = %act_r);
    %y_r = transpose(x = %y, perm = [0, 2, 1]);
    %out = reshape(x = %y_r, shape = [1, {oc}, 1, {seq_len}]);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, input_bytes, output_bytes)
}

/// Generate a MIL program for element-wise sigmoid: `y = sigmoid(x)`.
pub fn gen_sigmoid_program(dim: usize, seq_len: usize) -> (String, usize, usize) {
    let bytes = dim * seq_len * 2;

    let mut mil = String::with_capacity(1024);
    write!(
        mil,
        r#"{MIL_HEADER}    %input: tensor<[1, {dim}, 1, {seq_len}], fp16>
  ) {{
    %out = sigmoid(x = %input);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, bytes, bytes)
}

/// Generate a MIL program for element-wise multiply: `y = a * b`.
pub fn gen_multiply_program(dim: usize, seq_len: usize) -> (String, usize, usize) {
    let bytes = dim * seq_len * 2;

    let mut mil = String::with_capacity(1024);
    write!(
        mil,
        r#"{MIL_HEADER}    %a: tensor<[1, {dim}, 1, {seq_len}], fp16>,
    %b: tensor<[1, {dim}, 1, {seq_len}], fp16>
  ) {{
    %out = mul(x = %a, y = %b);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    // Two inputs, one output.
    (mil, bytes * 2, bytes)
}

/// Generate a MIL program for element-wise add: `y = a + b`.
pub fn gen_add_program(dim: usize, seq_len: usize) -> (String, usize, usize) {
    let bytes = dim * seq_len * 2;

    let mut mil = String::with_capacity(1024);
    write!(
        mil,
        r#"{MIL_HEADER}    %a: tensor<[1, {dim}, 1, {seq_len}], fp16>,
    %b: tensor<[1, {dim}, 1, {seq_len}], fp16>
  ) {{
    %out = add(x = %a, y = %b);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, bytes * 2, bytes)
}

/// Generate a MIL program for SiLU (SiGLU without gate): `y = x * sigmoid(x)`.
pub fn gen_silu_program(dim: usize, seq_len: usize) -> (String, usize, usize) {
    let bytes = dim * seq_len * 2;

    let mut mil = String::with_capacity(1024);
    write!(
        mil,
        r#"{MIL_HEADER}    %input: tensor<[1, {dim}, 1, {seq_len}], fp16>
  ) {{
    %sig = sigmoid(x = %input);
    %out = mul(x = %input, y = %sig);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, bytes, bytes)
}

/// Generate a MIL program for squared ReLU: `y = relu(x)^2`.
pub fn gen_sqrelu_program(dim: usize, seq_len: usize) -> (String, usize, usize) {
    let bytes = dim * seq_len * 2;

    let mut mil = String::with_capacity(1024);
    write!(
        mil,
        r#"{MIL_HEADER}    %input: tensor<[1, {dim}, 1, {seq_len}], fp16>
  ) {{
    %relu = relu(x = %input);
    %out = mul(x = %relu, y = %relu);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, bytes, bytes)
}

/// Generate a MIL program for LayerNorm.
///
/// Uses ANE's built-in `layer_norm` op with weight/bias as constexpr.
/// Weight and bias are packed into the IOSurface alongside activations.
pub fn gen_layer_norm_program(dim: usize, seq_len: usize, eps: f32) -> (String, usize, usize) {
    // Input layout: [1, dim, 1, seq_len + 2*dim] where:
    //   [0:seq_len] = activations
    //   [seq_len:seq_len+dim] = gamma (weight)
    //   [seq_len+dim:seq_len+2*dim] = beta (bias)
    let spatial = seq_len + 2 * dim;
    let input_bytes = dim * spatial * 2;
    let output_bytes = dim * seq_len * 2;

    let gamma_start = seq_len;
    let beta_start = seq_len + dim;

    let mut mil = String::with_capacity(2048);
    write!(
        mil,
        r#"{MIL_HEADER}    %input: tensor<[1, {dim}, 1, {spatial}], fp16>
  ) {{
    %act = slice_by_size(x = %input, begin = [0, 0, 0, 0], size = [1, {dim}, 1, {seq_len}]);
    %gamma_raw = slice_by_size(x = %input, begin = [0, 0, 0, {gamma_start}], size = [1, {dim}, 1, 1]);
    %beta_raw = slice_by_size(x = %input, begin = [0, 0, 0, {beta_start}], size = [1, {dim}, 1, 1]);
    %gamma = reshape(x = %gamma_raw, shape = [{dim}]);
    %beta = reshape(x = %beta_raw, shape = [{dim}]);
    %out = layer_norm(x = %act, axes = [1], gamma = %gamma, beta = %beta, epsilon = {eps});
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, input_bytes, output_bytes)
}

/// Check whether a matmul with given dimensions fits in ANE SRAM.
pub fn fits_in_sram(ic: usize, oc: usize, seq_len: usize) -> bool {
    ic * (seq_len + oc) <= ANE_SRAM_FP16_ELEMS
}

/// Compute the total IOSurface bytes for a DynMatmul kernel.
pub fn dyn_matmul_input_bytes(ic: usize, oc: usize, seq_len: usize) -> usize {
    ic * (seq_len + oc) * 2
}

/// Compute the output bytes for a matmul kernel.
pub fn matmul_output_bytes(oc: usize, seq_len: usize) -> usize {
    oc * seq_len * 2
}
