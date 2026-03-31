//! ANE forward pass orchestration for RWKV-7.
//!
//! Dispatches linear projections and element-wise ops to ANE,
//! runs the WKV recurrence on CPU (sequential dependency).
//!
//! Architecture: ANE(r,k,v projections) → CPU(WKV recurrence) → ANE(o_proj + FFN)
//!
//! Feature-gated behind `ane`.

#![allow(unsafe_code)]

use crate::ane_bridge;
use crate::ane_mil::MilConfig;

// ---------------------------------------------------------------------------
// ANE-specific recurrent state (pure f32, no MLX dependency)
// ---------------------------------------------------------------------------

/// Per-layer recurrent state for ANE forward. All f32, no MLX arrays.
pub struct AneLayerState {
    /// WKV accumulator: `[num_heads * head_dim * head_dim]` (row-major).
    pub wkv_state: Vec<f32>,
    /// Previous token hidden for attention token-shift: `[dim]`.
    pub attn_shift_state: Vec<f32>,
    /// Previous token hidden for FFN token-shift: `[dim]`.
    pub ffn_shift_state: Vec<f32>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl AneLayerState {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        let dim = num_heads * head_dim;
        Self {
            wkv_state: vec![0.0; num_heads * head_dim * head_dim],
            attn_shift_state: vec![0.0; dim],
            ffn_shift_state: vec![0.0; dim],
            num_heads,
            head_dim,
        }
    }

    /// Access wkv_state element at `[h, i, j]`.
    #[inline]
    fn s(&self, h: usize, i: usize, j: usize) -> f32 {
        self.wkv_state[h * self.head_dim * self.head_dim + i * self.head_dim + j]
    }

    /// Mutable access to wkv_state element at `[h, i, j]`.
    #[inline]
    fn s_mut(&mut self, h: usize, i: usize, j: usize) -> &mut f32 {
        &mut self.wkv_state[h * self.head_dim * self.head_dim + i * self.head_dim + j]
    }
}

// ---------------------------------------------------------------------------
// CPU-side per-layer weight data (LoRA, mixing, norms)
// ---------------------------------------------------------------------------

/// Intermediate activation for LoRA projections.
#[derive(Debug, Clone, Copy)]
pub enum LoraActivation {
    Tanh,
    Sigmoid,
    Identity,
}

/// LoRA weights: down `[rank, input_dim]`, up_weight `[output_dim, rank]`, up_bias `[output_dim]`.
pub struct LoraWeights {
    pub down: Vec<f32>,    // [rank, input_dim]
    pub up_weight: Vec<f32>, // [output_dim, rank]
    pub up_bias: Vec<f32>,   // [output_dim]
    pub rank: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub activation: LoraActivation,
}

// Accelerate BLAS FFI for fast matrix-vector operations.
unsafe extern "C" {
    // cblas_sgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    fn cblas_sgemv(
        order: i32, trans: i32, m: i32, n: i32,
        alpha: f32, a: *const f32, lda: i32,
        x: *const f32, incx: i32,
        beta: f32, y: *mut f32, incy: i32,
    );
    // cblas_sger(order, m, n, alpha, x, incx, y, incy, a, lda)
    // Rank-1 update: A += alpha * x * y^T
    fn cblas_sger(
        order: i32, m: i32, n: i32,
        alpha: f32,
        x: *const f32, incx: i32,
        y: *const f32, incy: i32,
        a: *mut f32, lda: i32,
    );
}

/// Fast matrix-vector multiply using Accelerate: y = A @ x
/// A is [m, n] row-major, x is [n], y is [m].
fn gemv_f32(a: &[f32], m: usize, n: usize, x: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemv(
            101, // CblasRowMajor
            111, // CblasNoTrans
            m as i32, n as i32,
            1.0, a.as_ptr(), n as i32,
            x.as_ptr(), 1,
            0.0, y.as_mut_ptr(), 1,
        );
    }
}

/// Fast matrix-vector multiply with transpose: y = A^T @ x
/// A is [m, n] row-major, x is [m], y is [n].
pub fn gemv_trans_f32(a: &[f32], m: usize, n: usize, x: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemv(
            101, // CblasRowMajor
            112, // CblasTrans
            m as i32, n as i32,
            1.0, a.as_ptr(), n as i32,
            x.as_ptr(), 1,
            0.0, y.as_mut_ptr(), 1,
        );
    }
}

impl LoraWeights {
    /// Forward: x @ down.T -> activation -> h @ up.T + bias
    /// Uses Accelerate cblas_sgemv for fast matrix-vector multiply.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        // h = down @ x: [rank, input_dim] @ [input_dim] -> [rank]
        let mut h = vec![0.0f32; self.rank];
        gemv_f32(&self.down, self.rank, self.input_dim, x, &mut h);

        // Activation
        for v in &mut h {
            *v = match self.activation {
                LoraActivation::Tanh => v.tanh(),
                LoraActivation::Sigmoid => 1.0 / (1.0 + (-*v).exp()),
                LoraActivation::Identity => *v,
            };
        }

        // out = up_weight @ h + bias: [output_dim, rank] @ [rank] -> [output_dim]
        let mut out = vec![0.0f32; self.output_dim];
        gemv_f32(&self.up_weight, self.output_dim, self.rank, &h, &mut out);
        for (o, b) in out.iter_mut().zip(self.up_bias.iter()) {
            *o += b;
        }
        out
    }
}

/// All CPU-side weight data for a single RWKV-7 layer.
pub struct LayerCpuWeights {
    // Token shift mixing params: [dim] each.
    pub x_r: Vec<f32>,
    pub x_w: Vec<f32>,
    pub x_k: Vec<f32>,
    pub x_v: Vec<f32>,
    pub x_a: Vec<f32>,
    pub x_g: Vec<f32>,

    // Key normalization: [dim].
    pub k_k: Vec<f32>,
    pub k_a: Vec<f32>,

    // Recurrent gate: [num_heads * head_dim] (flattened from [H, D]).
    pub r_k: Vec<f32>,

    // LoRA projections.
    pub w_lora: LoraWeights,
    pub a_lora: LoraWeights,
    pub v_lora: Option<LoraWeights>, // None for layer 0
    pub g_lora: LoraWeights,

    // Group norm: [dim].
    pub g_norm_weight: Vec<f32>,
    pub g_norm_bias: Vec<f32>,

    // Layer norms: weight [dim], bias [dim] (optional).
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn_norm_bias: Vec<f32>,

    // FFN token shift mixing: [dim].
    pub ffn_x_k: Vec<f32>,

    // Pre-norm (only layer 0 when norm_first).
    pub pre_norm_weight: Option<Vec<f32>>,
    pub pre_norm_bias: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// CPU helper ops
// ---------------------------------------------------------------------------

/// Layer normalization on a 1D f32 slice.
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter().zip(bias.iter()))
        .map(|(&xi, (&w, &b))| (xi - mean) * inv_std * w + b)
        .collect()
}

/// Group normalization: split `x` into `num_groups` and normalize each.
fn group_norm_cpu(x: &[f32], weight: &[f32], bias: &[f32], num_groups: usize, eps: f32) -> Vec<f32> {
    let c = x.len();
    let group_size = c / num_groups;
    let mut out = vec![0.0f32; c];

    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let group = &x[start..end];

        let mean: f32 = group.iter().sum::<f32>() / group_size as f32;
        let var: f32 = group.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / group_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..group_size {
            let idx = start + i;
            out[idx] = (x[idx] - mean) * inv_std * weight[idx] + bias[idx];
        }
    }
    out
}

/// L2-normalize a vector.
fn l2_norm_cpu(x: &[f32]) -> Vec<f32> {
    let sq_sum: f32 = x.iter().map(|v| v * v).sum();
    let norm = (sq_sum + 1e-12).sqrt();
    x.iter().map(|v| v / norm).collect()
}

/// Token shift mixing: result[i] = x[i] + mix[i] * (shifted[i] - x[i])
/// Python: `torch.addcmul(hidden, delta, x_param)` = `x + (prev - x) * mix`
fn apply_shift_cpu(x: &[f32], shifted: &[f32], mix: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(shifted.iter().zip(mix.iter()))
        .map(|(&xi, (&si, &mi))| xi + mi * (si - xi))
        .collect()
}

/// Convert f32 to fp16 bits (IEEE 754 half-precision).
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;
    if exp == 0xFF {
        return (sign | 0x7C00 | if frac != 0 { 0x0200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        return sign as u16;
    }
    (sign | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

// ANE IOSurface I/O helpers (used when ANE kernels are active).
// Kept for future ANE integration.
#[allow(dead_code)]
fn _write_activation_f32(kernel: &crate::ane_bridge::AneKernel, activation: &[f32], padded_seq: usize) {
    if padded_seq <= 1 {
        kernel.write_input(0, &f32_to_bytes(activation));
    } else {
        let ic = activation.len();
        let mut padded = vec![0.0f32; ic * padded_seq];
        for c in 0..ic {
            padded[c * padded_seq] = activation[c];
        }
        kernel.write_input(0, &f32_to_bytes(&padded));
    }
}

#[allow(dead_code)]
fn _read_output_f32(kernel: &crate::ane_bridge::AneKernel, oc: usize, padded_seq: usize) -> Vec<f32> {
    let total = oc * padded_seq;
    let mut buf = vec![0u8; total * 4];
    kernel.read_output(0, &mut buf);
    let all = bytes_to_f32(&buf);
    if padded_seq <= 1 { all } else { (0..oc).map(|c| all[c * padded_seq]).collect() }
}

/// Read f32 values from little-endian byte buffer.
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Write f32 values to little-endian byte buffer.
fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------
// Compiled kernel cache for one RWKV-7 layer
// ---------------------------------------------------------------------------

pub struct Rwkv7AneExecutor {
    pub cpu_weights: Vec<LayerCpuWeights>,
    /// Projection weights (transposed [ic, oc] for BLAS gemv_trans fallback).
    pub ane_weights: Vec<LayerAneWeightData>,
    /// fp16 weights (original [oc, ic] layout for gemm_f16 — half bandwidth).
    fp16_weights: Vec<LayerFp16Weights>,
    pub config: MilConfig,
    pub padded_seq: usize,
    initialized: bool,
}

/// Weight data for projections (f32, transposed [ic, oc] for BLAS gemv_trans).
pub struct LayerAneWeightData {
    pub r_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub ffn_key: Vec<f32>,
    pub ffn_value: Vec<f32>,
}

/// Pre-converted fp16 weights (original [oc, ic] layout for gemm_f16).
pub struct LayerFp16Weights {
    pub r_proj: Vec<u16>,
    pub k_proj: Vec<u16>,
    pub v_proj: Vec<u16>,
    pub o_proj: Vec<u16>,
    pub ffn_key: Vec<u16>,
    pub ffn_value: Vec<u16>,
}

/// Convert f32 slice to fp16 u16 bits.
fn f32_vec_to_fp16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_f16(v)).collect()
}

impl Rwkv7AneExecutor {
    pub fn new(config: MilConfig) -> Self {
        let padded_seq = config.seq_len.max(crate::ane_mil::ANE_MIN_SPATIAL);
        Self {
            cpu_weights: Vec::new(),
            ane_weights: Vec::new(),
            fp16_weights: Vec::new(),
            config,
            padded_seq,
            initialized: false,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.initialized
    }

    /// Load weights for BLAS-accelerated decode.
    pub fn compile(
        &mut self,
        ane_weights: Vec<LayerAneWeightData>,
        cpu_weights: Vec<LayerCpuWeights>,
        fp16_weights: Vec<LayerFp16Weights>,
    ) -> Result<(), String> {
        self.cpu_weights = cpu_weights;
        self.ane_weights = ane_weights;
        self.fp16_weights = fp16_weights;
        self.initialized = true;
        tracing::info!(
            num_layers = self.cpu_weights.len(),
            "RWKV-7 BLAS executor ready"
        );
        Ok(())
    }

    /// Create fresh layer states for all layers.
    pub fn make_states(&self) -> Vec<AneLayerState> {
        (0..self.cpu_weights.len())
            .map(|_| AneLayerState::new(self.config.num_heads, self.config.head_dim))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Forward pass: ANE projections + CPU WKV recurrence
// ---------------------------------------------------------------------------

/// Full model state for ANE decode path.
pub struct AneModelState {
    pub layers: Vec<AneLayerState>,
    /// v_first from layer 0, used by subsequent layers for value interpolation.
    pub v_first: Vec<f32>,
}

/// Run a single decode step (seq_len=1) through the ANE executor.
///
/// Matches the corrected RWKV-7 DPLR Delta Rule recurrence from rwkv7.rs.
pub fn forward_ane_decode(
    executor: &Rwkv7AneExecutor,
    input_hidden: &[f32],
    model_state: &mut AneModelState,
) -> Result<Vec<f32>, String> {
    if !executor.is_ready() {
        return Err("ANE executor not compiled".into());
    }

    let dim = executor.config.dim;
    let inter = executor.config.intermediate_size;
    let num_heads = executor.config.num_heads;
    let head_dim = executor.config.head_dim;
    let gn_eps = (head_dim as f32) * 1e-5;

    let mut hidden = input_hidden.to_vec();

    // Pre-allocate all scratch buffers (eliminates ~480 malloc/free per token).
    let mut r = vec![0.0f32; dim];
    let mut k = vec![0.0f32; dim];
    let mut v = vec![0.0f32; dim];
    let mut o = vec![0.0f32; dim];
    let mut fk = vec![0.0f32; inter];
    let mut fv = vec![0.0f32; dim];
    let mut attn_out = vec![0.0f32; dim];
    let mut kk_a = vec![0.0f32; dim];

    let mut t_ane_us = 0u64;
    let mut t_cpu_us = 0u64;
    let mut t_io_us = 0u64;
    let mut t_lora_us = 0u64;
    let mut t_wkv_us = 0u64;

    for (layer_idx, cpu_w) in executor.cpu_weights.iter().enumerate() {
        let state = &mut model_state.layers[layer_idx];

        // --- Pre-norm (only layer 0 when norm_first) ---
        if let (Some(w), Some(b)) = (&cpu_w.pre_norm_weight, &cpu_w.pre_norm_bias) {
            hidden = layer_norm(&hidden, w, b, 1e-5);
        }

        // --- Attention path ---

        // 1. Layer norm
        let normed = layer_norm(&hidden, &cpu_w.attn_norm_weight, &cpu_w.attn_norm_bias, 1e-5);

        // 2. Token shift: delta = prev - curr, mix = curr + delta * param
        let shifted = &state.attn_shift_state;
        let xr = apply_shift_cpu(&normed, shifted, &cpu_w.x_r);
        let xw = apply_shift_cpu(&normed, shifted, &cpu_w.x_w);
        let xk = apply_shift_cpu(&normed, shifted, &cpu_w.x_k);
        let xv = apply_shift_cpu(&normed, shifted, &cpu_w.x_v);
        let xa = apply_shift_cpu(&normed, shifted, &cpu_w.x_a);
        let xg = apply_shift_cpu(&normed, shifted, &cpu_w.x_g);
        state.attn_shift_state.copy_from_slice(&normed);

        // 3. r/k/v projections via Accelerate sgemv (AMX-optimized, fastest for T=1)
        let t0 = std::time::Instant::now();
        let ane_w = &executor.ane_weights[layer_idx];
        gemv_trans_f32(&ane_w.r_proj, dim, dim, &xr, &mut r);
        gemv_trans_f32(&ane_w.k_proj, dim, dim, &xk, &mut k);
        gemv_trans_f32(&ane_w.v_proj, dim, dim, &xv, &mut v);
        t_ane_us += t0.elapsed().as_micros() as u64;

        // 4. LoRA projections on CPU
        let t0_lora = std::time::Instant::now();
        let w_raw = cpu_w.w_lora.forward(&xw);
        let a_raw = cpu_w.a_lora.forward(&xa);
        let g = cpu_w.g_lora.forward(&xg); // gate (used directly, no extra sigmoid)

        // Decay: w_log = -0.6065 * sigmoid(w_raw), then w = exp(w_log)
        let w: Vec<f32> = w_raw
            .iter()
            .map(|&wi| (-0.606_530_66 * (1.0 / (1.0 + (-wi).exp()))).exp())
            .collect();

        // Attention bonus: a = sigmoid(a_raw)
        let a: Vec<f32> = a_raw.iter().map(|&ai| 1.0 / (1.0 + (-ai).exp())).collect();

        // v_first: layer 0 stores, layers > 0 lerp
        if layer_idx == 0 {
            model_state.v_first = v.clone();
        } else if let Some(ref v_lora) = cpu_w.v_lora {
            let alpha_raw = v_lora.forward(&xv);
            for (i, vi) in v.iter_mut().enumerate() {
                let alpha = 1.0 / (1.0 + (-alpha_raw[i]).exp()); // sigmoid
                *vi += alpha * (model_state.v_first[i] - *vi); // lerp
            }
        }

        // kk: l2_norm(k * k_k) — normalized key for attention bonus
        let kk: Vec<f32> = k.iter().zip(cpu_w.k_k.iter()).map(|(&ki, &kki)| ki * kki).collect();
        let kk = l2_norm_cpu(&kk);

        // k update: k = k * (1 + (a - 1) * k_a)
        for (i, ki) in k.iter_mut().enumerate() {
            *ki *= 1.0 + (a[i] - 1.0) * cpu_w.k_a[i];
        }

        t_lora_us += t0_lora.elapsed().as_micros() as u64;

        // 5. DPLR Delta Rule recurrence (T=1, B=1) — BLAS-accelerated
        let t0_wkv = std::time::Instant::now();
        let d = head_dim;
        let dd = d * d;

        // Precompute kk*a into pre-allocated buffer
        for (i, (kki, ai)) in kk.iter().zip(a.iter()).enumerate() {
            kk_a[i] = kki * ai;
        }

        for h in 0..num_heads {
            let ho = h * d;
            let s_off = h * dd; // offset into flat wkv_state

            // 1. Decay: S[i,:] *= w[i] for each row
            for i in 0..d {
                let wi = w[ho + i];
                let row_start = s_off + i * d;
                for j in 0..d {
                    state.wkv_state[row_start + j] *= wi;
                }
            }

            // 2. S += outer(k, v): rank-1 update via cblas_sger
            //    S[d,d] += k[d] * v[d]^T
            unsafe {
                cblas_sger(
                    101, d as i32, d as i32,
                    1.0,
                    k[ho..].as_ptr(), 1,
                    v[ho..].as_ptr(), 1,
                    state.wkv_state[s_off..].as_mut_ptr(), d as i32,
                );
            }

            // 3. S -= outer(kk, kk_a): rank-1 update with alpha=-1
            unsafe {
                cblas_sger(
                    101, d as i32, d as i32,
                    -1.0,
                    kk[ho..].as_ptr(), 1,
                    kk_a[ho..].as_ptr(), 1,
                    state.wkv_state[s_off..].as_mut_ptr(), d as i32,
                );
            }

            // 4. out = r^T @ S: [1,d] @ [d,d] = [1,d]  via cblas_sgemv(Trans)
            //    out[j] = sum_i r[i] * S[i,j] = (S^T @ r)[j]
            unsafe {
                cblas_sgemv(
                    101, 112, // RowMajor, Trans
                    d as i32, d as i32,
                    1.0,
                    state.wkv_state[s_off..].as_ptr(), d as i32,
                    r[ho..].as_ptr(), 1,
                    0.0,
                    attn_out[ho..].as_mut_ptr(), 1,
                );
            }
        }

        t_wkv_us += t0_wkv.elapsed().as_micros() as u64;

        // 6. Group norm + gate output correction + gate
        let y = group_norm_cpu(&attn_out, &cpu_w.g_norm_weight, &cpu_w.g_norm_bias, num_heads, gn_eps);

        // correction = (r * k * r_k).sum_per_head * v — reuse `o` buffer as y_out
        for h in 0..num_heads {
            let ho = h * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += r[ho + d] * k[ho + d] * cpu_w.r_k[h * head_dim + d];
            }
            for d in 0..head_dim {
                o[ho + d] = (y[ho + d] + dot * v[ho + d]) * g[ho + d];
            }
        }
        // y_out is now in `o` buffer

        t_cpu_us += t0.elapsed().as_micros() as u64;

        // 7. o_proj via sgemv
        let t0 = std::time::Instant::now();
        gemv_trans_f32(&ane_w.o_proj, dim, dim, &o, &mut fv);
        t_ane_us += t0.elapsed().as_micros() as u64;

        // 8. Residual add (o_proj result is in fv buffer)
        for (h, val) in hidden.iter_mut().zip(fv.iter()) {
            *h += val;
        }

        // --- FFN path ---

        // 9. FFN norm
        let ffn_normed = layer_norm(&hidden, &cpu_w.ffn_norm_weight, &cpu_w.ffn_norm_bias, 1e-5);

        // FFN token shift (separate state from attention)
        let ffn_xk = apply_shift_cpu(&ffn_normed, &state.ffn_shift_state, &cpu_w.ffn_x_k);
        state.ffn_shift_state.copy_from_slice(&ffn_normed);

        // FFN via sgemv
        let t0 = std::time::Instant::now();
        gemv_trans_f32(&ane_w.ffn_key, dim, inter, &ffn_xk, &mut fk);
        for val in fk.iter_mut() { *val = val.max(0.0); *val *= *val; } // sqReLU
        gemv_trans_f32(&ane_w.ffn_value, inter, dim, &fk, &mut fv);
        t_ane_us += t0.elapsed().as_micros() as u64;

        // Residual add
        for (h, f) in hidden.iter_mut().zip(fv.iter()) {
            *h += f;
        }
    }

    eprintln!(
        "  timing: proj={}ms lora={}ms wkv={}ms other={}ms total={}ms",
        t_ane_us / 1000, t_lora_us / 1000, t_wkv_us / 1000,
        (t_cpu_us.saturating_sub(t_lora_us).saturating_sub(t_wkv_us)) / 1000,
        (t_ane_us + t_cpu_us + t_io_us) / 1000,
    );

    Ok(hidden)
}

// ---------------------------------------------------------------------------
// Full forward: token_id → logits
// ---------------------------------------------------------------------------

/// Full ANE decode: embedding → layers → norm → LM head.
pub fn forward_ane_full(
    executor: &Rwkv7AneExecutor,
    token_id: u32,
    model_state: &mut AneModelState,
    embedding: &[f32],
    lm_head_fp16: &[u16],
    final_norm_w: &[f32],
    final_norm_b: &[f32],
    vocab_size: usize,
    dim: usize,
) -> Result<Vec<f32>, String> {
    // 1. Embedding lookup
    let start = token_id as usize * dim;
    let end = start + dim;
    if end > embedding.len() {
        return Err(format!("Token {token_id} out of embedding range"));
    }
    let input_hidden = &embedding[start..end];

    // 2. Layer-by-layer decode
    let hidden = forward_ane_decode(executor, input_hidden, model_state)?;

    // 3. Final layer norm
    let normed = layer_norm(&hidden, final_norm_w, final_norm_b, 1e-5);

    // 4. LM head: lm_head [vocab, dim] @ normed [dim, 1] → logits [vocab]
    // gemm_f16(A_fp16, M, K, B_f32, N, C_out, alpha, beta)
    // A = lm_head [vocab × dim], B = normed [dim × 1], C = logits [vocab × 1]
    let mut logits = vec![0.0f32; vocab_size];
    ane_bridge::gemm_f16(lm_head_fp16, vocab_size, dim, &normed, 1, &mut logits, 1.0, 0.0);
    Ok(logits)
}
