//! ANE forward pass orchestration for RWKV-7.
//!
//! Dispatches linear projections and element-wise ops to ANE,
//! runs the WKV recurrence on CPU (sequential dependency).
//!
//! Feature-gated behind `ane`.

#![cfg(feature = "ane")]

use crate::ane_bridge::{self, build_weight_blob, AneKernel};
use crate::ane_mil::{self, MilConfig};
use crate::rwkv7::Rwkv7LayerState;

// ---------------------------------------------------------------------------
// Compiled kernel cache for one RWKV-7 layer
// ---------------------------------------------------------------------------

/// Pre-compiled ANE kernels for a single RWKV-7 layer.
///
/// These are compiled once and reused across inference steps.
/// Weights are packed into IOSurfaces at compile time.
pub struct LayerKernels {
    /// r, k, v projections: dim -> dim.
    pub r_proj: AneKernel,
    pub k_proj: AneKernel,
    pub v_proj: AneKernel,
    /// Output projection: dim -> dim.
    pub o_proj: AneKernel,
    /// FFN key projection: dim -> intermediate_size.
    pub ffn_key: AneKernel,
    /// FFN value projection: intermediate_size -> dim.
    pub ffn_value: AneKernel,
}

/// All compiled ANE kernels for the full RWKV-7 model.
pub struct Rwkv7AneExecutor {
    pub layers: Vec<LayerKernels>,
    pub config: MilConfig,
    initialized: bool,
}

impl Rwkv7AneExecutor {
    /// Create an uninitialized executor. Call `compile()` to prepare kernels.
    pub fn new(config: MilConfig) -> Self {
        Self {
            layers: Vec::new(),
            config,
            initialized: false,
        }
    }

    /// Whether the executor has been compiled and is ready for inference.
    pub fn is_ready(&self) -> bool {
        self.initialized
    }

    /// Compile all ANE kernels for the model.
    ///
    /// `layer_weights` provides the weight data for each layer's projections.
    pub fn compile(&mut self, layer_weights: &[LayerWeightData]) -> Result<(), String> {
        ane_bridge::init()?;
        ane_bridge::set_quiet(true);

        let dim = self.config.dim;
        let inter = self.config.intermediate_size;
        let seq = self.config.seq_len;

        let mut layers = Vec::with_capacity(layer_weights.len());

        for (i, weights) in layer_weights.iter().enumerate() {
            tracing::debug!(layer = i, "Compiling ANE kernels for RWKV-7 layer");

            if !ane_mil::fits_in_sram(dim, dim, seq) {
                return Err(format!(
                    "Layer {i}: dim×dim matmul ({dim}×{dim}) doesn't fit in ANE SRAM at seq_len={seq}"
                ));
            }

            // Generate MIL programs.
            let (r_mil, r_in, r_out) = ane_mil::gen_matmul_program(dim, dim, seq);
            let (k_mil, _, _) = ane_mil::gen_matmul_program(dim, dim, seq);
            let (v_mil, _, _) = ane_mil::gen_matmul_program(dim, dim, seq);
            let (o_mil, _, _) = ane_mil::gen_matmul_program(dim, dim, seq);
            let (fk_mil, fk_in, fk_out) = ane_mil::gen_matmul_program(dim, inter, seq);
            let (fv_mil, fv_in, fv_out) = ane_mil::gen_matmul_program(inter, dim, seq);

            // Build weight blobs (f32 -> fp16 ANE format).
            let r_blob = build_weight_blob(&weights.r_proj, dim, dim);
            let k_blob = build_weight_blob(&weights.k_proj, dim, dim);
            let v_blob = build_weight_blob(&weights.v_proj, dim, dim);
            let o_blob = build_weight_blob(&weights.o_proj, dim, dim);
            let fk_blob = build_weight_blob(&weights.ffn_key, inter, dim);
            let fv_blob = build_weight_blob(&weights.ffn_value, dim, inter);

            // Compile kernels.
            let r_kern = AneKernel::compile(&r_mil, Some(&r_blob), &[r_in], &[r_out])
                .map_err(|e| format!("Layer {i} r_proj: {e}"))?;
            let k_kern = AneKernel::compile(&k_mil, Some(&k_blob), &[r_in], &[r_out])
                .map_err(|e| format!("Layer {i} k_proj: {e}"))?;
            let v_kern = AneKernel::compile(&v_mil, Some(&v_blob), &[r_in], &[r_out])
                .map_err(|e| format!("Layer {i} v_proj: {e}"))?;
            let o_kern = AneKernel::compile(&o_mil, Some(&o_blob), &[r_in], &[r_out])
                .map_err(|e| format!("Layer {i} o_proj: {e}"))?;
            let fk_kern = AneKernel::compile(&fk_mil, Some(&fk_blob), &[fk_in], &[fk_out])
                .map_err(|e| format!("Layer {i} ffn_key: {e}"))?;
            let fv_kern = AneKernel::compile(&fv_mil, Some(&fv_blob), &[fv_in], &[fv_out])
                .map_err(|e| format!("Layer {i} ffn_value: {e}"))?;

            layers.push(LayerKernels {
                r_proj: r_kern,
                k_proj: k_kern,
                v_proj: v_kern,
                o_proj: o_kern,
                ffn_key: fk_kern,
                ffn_value: fv_kern,
            });

            tracing::debug!(
                layer = i,
                cached = layers.last().map_or(false, |l| l.r_proj.was_cached()),
                "ANE kernels compiled"
            );
        }

        self.layers = layers;
        self.initialized = true;

        tracing::info!(
            num_layers = self.layers.len(),
            compiles = ane_bridge::compile_count(),
            loads = ane_bridge::load_count(),
            "RWKV-7 ANE executor ready"
        );

        ane_bridge::set_quiet(false);
        Ok(())
    }
}

/// Weight data for a single RWKV-7 layer (f32 slices).
pub struct LayerWeightData {
    pub r_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub ffn_key: Vec<f32>,
    pub ffn_value: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

/// Run a single decode step (seq_len=1) through the ANE executor.
///
/// The WKV recurrence runs on CPU between ANE dispatches:
///   ANE(projections) -> CPU(WKV recurrence) -> ANE(output+FFN)
///
/// Returns the hidden state as f32 after all layers.
pub fn forward_ane_decode(
    executor: &Rwkv7AneExecutor,
    input_hidden: &[f32],
    states: &mut [Option<Rwkv7LayerState>],
) -> Result<Vec<f32>, String> {
    if !executor.is_ready() {
        return Err("ANE executor not compiled".into());
    }

    let dim = executor.config.dim;
    let _num_heads = executor.config.num_heads;
    let _head_dim = executor.config.head_dim;

    let mut hidden = input_hidden.to_vec();

    for (layer_idx, layer_kernels) in executor.layers.iter().enumerate() {
        let state = states[layer_idx]
            .as_mut()
            .ok_or_else(|| format!("Missing state for layer {layer_idx}"))?;

        // --- Attention path ---
        // Write activations to projection kernels and evaluate on ANE.
        let act_bytes: Vec<u8> = hidden.iter().flat_map(|f| f.to_le_bytes()).collect();

        layer_kernels.r_proj.write_input(0, &act_bytes);
        layer_kernels.k_proj.write_input(0, &act_bytes);
        layer_kernels.v_proj.write_input(0, &act_bytes);

        layer_kernels
            .r_proj
            .eval()
            .map_err(|e| format!("Layer {layer_idx} r_proj: {e}"))?;
        layer_kernels
            .k_proj
            .eval()
            .map_err(|e| format!("Layer {layer_idx} k_proj: {e}"))?;
        layer_kernels
            .v_proj
            .eval()
            .map_err(|e| format!("Layer {layer_idx} v_proj: {e}"))?;

        // Read projection outputs.
        let mut r_out = vec![0u8; dim * 4];
        let mut k_out = vec![0u8; dim * 4];
        let mut v_out = vec![0u8; dim * 4];
        layer_kernels.r_proj.read_output(0, &mut r_out);
        layer_kernels.k_proj.read_output(0, &mut k_out);
        layer_kernels.v_proj.read_output(0, &mut v_out);

        // --- WKV recurrence on CPU ---
        // For T=1 decode, this is a single state update step.
        // TODO: Full WKV-7 recurrence with token shift, decay, attention bonus.
        // Placeholder: pass r_proj output through.
        let attn_out = r_out;

        // --- Output projection on ANE ---
        layer_kernels.o_proj.write_input(0, &attn_out);
        layer_kernels
            .o_proj
            .eval()
            .map_err(|e| format!("Layer {layer_idx} o_proj: {e}"))?;
        let mut o_out = vec![0u8; dim * 4];
        layer_kernels.o_proj.read_output(0, &mut o_out);

        // Residual add (CPU).
        let o_f32: Vec<f32> = o_out
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        for (h, o) in hidden.iter_mut().zip(o_f32.iter()) {
            *h += o;
        }

        // --- FFN path ---
        let ffn_bytes: Vec<u8> = hidden.iter().flat_map(|f| f.to_le_bytes()).collect();

        layer_kernels.ffn_key.write_input(0, &ffn_bytes);
        layer_kernels
            .ffn_key
            .eval()
            .map_err(|e| format!("Layer {layer_idx} ffn_key: {e}"))?;

        let inter = executor.config.intermediate_size;
        let mut fk_out = vec![0u8; inter * 4];
        layer_kernels.ffn_key.read_output(0, &mut fk_out);

        // sqReLU activation on CPU.
        let mut fk_f32: Vec<f32> = fk_out
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        for v in &mut fk_f32 {
            *v = v.max(0.0);
            *v *= *v;
        }

        let fk_bytes: Vec<u8> = fk_f32.iter().flat_map(|f| f.to_le_bytes()).collect();

        layer_kernels.ffn_value.write_input(0, &fk_bytes);
        layer_kernels
            .ffn_value
            .eval()
            .map_err(|e| format!("Layer {layer_idx} ffn_value: {e}"))?;

        let mut fv_out = vec![0u8; dim * 4];
        layer_kernels.ffn_value.read_output(0, &mut fv_out);

        // Residual add.
        let fv_f32: Vec<f32> = fv_out
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        for (h, f) in hidden.iter_mut().zip(fv_f32.iter()) {
            *h += f;
        }

        state.offset += 1;
    }

    Ok(hidden)
}
