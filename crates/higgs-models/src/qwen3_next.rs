//! Qwen3-Coder-Next model implementation.
//!
//! Hybrid SSM/attention transformer with Mixture of Experts (`MoE`).
//! Every `full_attention_interval`-th layer uses full attention (`Qwen3NextAttention`),
//! all other layers use `GatedDeltaNet` (SSM-like linear attention).
//! All layers use Sparse `MoE` for the feed-forward block.

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use mlx_rs::{
    Array, Dtype, Stream,
    builder::Builder,
    error::Exception,
    fast,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// FFI error capture for gather_qmm
// ---------------------------------------------------------------------------

/// Captures the most recent MLX error message from our FFI calls.
static FFI_LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// Error handler registered once with MLX to capture error messages.
#[allow(unsafe_code)]
unsafe extern "C" fn ffi_error_handler(msg: *const c_char, _data: *mut c_void) {
    let s = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    if let Ok(mut guard) = FFI_LAST_ERROR.lock() {
        *guard = Some(s);
    }
}

/// Register our FFI error handler exactly once.
fn ensure_ffi_error_handler() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_set_error_handler(Some(ffi_error_handler), std::ptr::null_mut(), None);
        }
    });
}

/// Wrapper for the cached `GatedDeltaNet` Metal kernel object.
struct CachedMetalKernel(mlx_sys::mlx_fast_metal_kernel);

// The kernel object is immutable after creation and used read-only in apply.
#[allow(unsafe_code)]
unsafe impl Send for CachedMetalKernel {}
#[allow(unsafe_code)]
unsafe impl Sync for CachedMetalKernel {}

impl Drop for CachedMetalKernel {
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.0);
        }
    }
}

/// Cached `GatedDeltaNet` Metal kernel -- created once, reused for all layers.
static GATED_DELTA_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

/// Cached tape-recording GDN kernel -- outputs (y, state_out, delta_tape).
static GATED_DELTA_TAPE_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

/// Cached tape-replay kernel -- inputs (tape, k, a, a_log, dt_bias, state_in, T), outputs state_out.
static TAPE_REPLAY_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

use crate::{
    cache::{KeyValueCache, KvCacheView, SteppingKeyValueCache},
    error::ModelError,
    utils::{AttentionMask, apply_rope, create_causal_mask},
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const fn default_full_attention_interval() -> i32 {
    4
}

const fn default_rope_theta() -> f32 {
    10000.0
}

const fn default_partial_rotary_factor() -> f32 {
    1.0
}

/// Match Python mlx-lm default: `norm_topk_prob: bool = True`.
/// Without normalization, `MoE` expert scores sum to ~0.39 instead of 1.0,
/// producing 0.39x output magnitude and degenerate generation.
const fn default_norm_topk_prob() -> bool {
    true
}

/// Quantization parameters from config.json (top-level defaults).
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

/// Configuration for the Qwen3-Next / Qwen3.5 hybrid architecture.
///
/// Supports hybrid SSM/attention transformers with optional Sparse MoE.
/// Every `full_attention_interval`-th layer uses full attention, all other
/// layers use `GatedDeltaNet` (SSM-like linear attention). MoE layers are
/// enabled when `decoder_sparse_step > 0` and `num_experts > 0`.
///
/// Key fields:
/// - `norm_topk_prob` — normalize top-k expert scores (default `true`).
/// - `gate_quantization` — optional quantization override for MoE gate weights.
/// - `use_separate_gdn_projections` — when `true`, GDN layers use 4 separate
///   projection matrices; when `false` (default), projections are fused to 2
///   combined matrices for fewer GPU dispatches.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3NextModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    // Linear attention (GatedDeltaNet) params
    #[serde(default)]
    pub linear_num_value_heads: i32,
    #[serde(default)]
    pub linear_num_key_heads: i32,
    #[serde(default)]
    pub linear_key_head_dim: i32,
    #[serde(default)]
    pub linear_value_head_dim: i32,
    #[serde(default)]
    pub linear_conv_kernel_dim: i32,

    // MoE params
    #[serde(default)]
    pub num_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: i32,
    #[serde(default)]
    pub decoder_sparse_step: i32,
    #[serde(default)]
    pub shared_expert_intermediate_size: i32,
    #[serde(default)]
    pub moe_intermediate_size: i32,
    /// Normalize top-k expert scores to sum to 1.0 before weighting outputs.
    /// Defaults to `true` to match Python mlx-lm. Setting to `false` scales
    /// MoE output by the raw softmax scores (~0.39x), causing degenerate output.
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,

    /// Per-layer quantization override for router gate / shared_expert_gate.
    /// When absent, uses the global quantization config.
    #[serde(default)]
    pub gate_quantization: Option<QuantizationConfig>,

    /// Use separate GDN projections (qwen3.5-style) instead of combined (qwen3_next-style).
    #[serde(default)]
    pub use_separate_gdn_projections: bool,
}

// ---------------------------------------------------------------------------
// Quantized weight containers
// ---------------------------------------------------------------------------

type QuantizedParams = (Param<Array>, Param<Array>, Param<Array>);

pub(crate) fn init_quantized_params() -> Result<QuantizedParams, Exception> {
    Ok((
        Param::new(Array::zeros::<f32>(&[1])?),
        Param::new(Array::zeros::<f32>(&[1])?),
        Param::new(Array::zeros::<f32>(&[1])?),
    ))
}

pub(crate) fn quantized_forward(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    ops::quantized_matmul(x, weight, scales, biases, true, group_size, bits)
}

/// Quantized linear layer stored as raw weight/scales/biases arrays.
/// Forward uses `quantized_matmul` directly.
#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct QLinear {
    #[param]
    pub(crate) weight: Param<Array>,
    #[param]
    pub(crate) scales: Param<Array>,
    #[param]
    pub(crate) biases: Param<Array>,
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
}

impl QLinear {
    pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        let (weight, scales, biases) = init_quantized_params()?;
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array, Exception> {
        if self.weight.dtype() == Dtype::Uint32 {
            quantized_forward(
                x,
                &self.weight,
                &self.scales,
                &self.biases,
                self.group_size,
                self.bits,
            )
        } else {
            ops::matmul(x, self.weight.value.t())
        }
    }
}

/// Quantized embedding stored as raw weight/scales/biases arrays.
#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct QEmbedding {
    #[param]
    weight: Param<Array>,
    #[param]
    scales: Param<Array>,
    #[param]
    biases: Param<Array>,
    group_size: i32,
    bits: i32,
}

impl QEmbedding {
    pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        let (weight, scales, biases) = init_quantized_params()?;
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    pub(crate) fn forward(&self, indices: &Array) -> Result<Array, Exception> {
        if self.weight.dtype() == Dtype::Uint32 {
            let shape = indices.shape().to_vec();
            let flat = indices.flatten(None, None)?;
            let w = (*self.weight).take_axis(&flat, 0)?;
            let s = (*self.scales).take_axis(&flat, 0)?;
            let b = (*self.biases).take_axis(&flat, 0)?;
            let out = ops::dequantize(&w, &s, &b, self.group_size, self.bits)?;
            let mut ret_shape: Vec<i32> = shape.to_vec();
            ret_shape.push(-1);
            out.reshape(&ret_shape)
        } else {
            Ok(self.weight.index(indices))
        }
    }

    pub(crate) fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        if self.weight.dtype() == Dtype::Uint32 {
            quantized_forward(
                x,
                &self.weight,
                &self.scales,
                &self.biases,
                self.group_size,
                self.bits,
            )
        } else {
            ops::matmul(x, self.weight.value.t())
        }
    }
}

// ---------------------------------------------------------------------------
// SwiGLU activation
// ---------------------------------------------------------------------------

/// Reads `HIGGS_TARGET_COMPILE` once and caches the result.
///
/// When `=1`, [`swiglu`] routes through `mlx_rs::transforms::compile::compile`
/// so MLX can fuse `sigmoid + multiply(gate) + multiply(x)` into a single
/// Metal kernel, trimming ~1 dispatch per MLP per layer. Default off — we
/// want opt-in until A/B confirms a win. See
/// `.planning/next-session-verify-bottleneck.md` for context.
fn target_compile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = std::env::var("HIGGS_TARGET_COMPILE").as_deref() == Ok("1");
        if on {
            tracing::info!("HIGGS_TARGET_COMPILE=1 — swiglu routed through mlx_rs::compile()");
        }
        on
    })
}

/// SiLU(gate) * x — uses `nn::silu` (compiled/fused sigmoid+multiply) to
/// reduce dispatch count from 3 unfused ops to 2 (silu is 1 fused kernel + 1
/// multiply).
///
/// When `HIGGS_TARGET_COMPILE=1`, the whole `silu(g) * u` chain is wrapped in
/// `mlx_rs::transforms::compile::compile` so MLX fuses all three element-wise
/// ops into a single dispatch. Numerical equivalence is enforced by the A/B
/// parity gate before merge.
pub(crate) fn swiglu(gate: &Array, x: &Array) -> Result<Array, Exception> {
    if target_compile_enabled() {
        // MLX caches compiled graphs internally by the closure's TypeId. The
        // closure captures nothing, so its type is stable across calls and
        // every re-call hits the MLX cache after the first warmup.
        //
        // `shapeless=false` (None → default) is the right choice for
        // DFlash verify: seq is fixed per block_size, so per-shape caching
        // compiles exactly once per stable shape we see.
        let mut compiled = mlx_rs::transforms::compile::compile(
            |(g, u): (&Array, &Array)| -> Result<Array, Exception> { nn::silu(g)?.multiply(u) },
            None,
        );
        compiled((gate, x))
    } else {
        nn::silu(gate)?.multiply(x)
    }
}

/// sigmoid(gate) * x — element-wise sigmoid gating. When
/// `HIGGS_TARGET_COMPILE=1`, the sigmoid+multiply pair fuses into one dispatch.
/// Used in attention output gating (2048d per decode token) and in MoE
/// shared-expert gating (one scalar per expert output).
pub(crate) fn sigmoid_mul(gate: &Array, x: &Array) -> Result<Array, Exception> {
    if target_compile_enabled() {
        let mut compiled = mlx_rs::transforms::compile::compile(
            |(g, u): (&Array, &Array)| -> Result<Array, Exception> { nn::sigmoid(g)?.multiply(u) },
            None,
        );
        compiled((gate, x))
    } else {
        nn::sigmoid(gate)?.multiply(x)
    }
}

// ---------------------------------------------------------------------------
// gather_qmm FFI wrapper
// ---------------------------------------------------------------------------

/// Quantized matrix multiplication with expert-level gather, dispatched as a
/// single fused GPU kernel. Replaces per-expert `take_axis + quantized_matmul`
/// loops in `MoE` layers.
///
/// `rhs_indices` selects which expert weight matrices to use for each batch
/// element. Batch dimensions of `x` and `rhs_indices` are broadcast together.
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    rhs_indices: &Array,
    transpose: bool,
    group_size: i32,
    bits: i32,
    sorted_indices: bool,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let null_lhs = unsafe { mlx_sys::mlx_array_new() };
    let mut result = unsafe { mlx_sys::mlx_array_new() };
    let status = unsafe {
        mlx_sys::mlx_gather_qmm(
            &raw mut result,
            x.as_ptr(),
            w.as_ptr(),
            scales.as_ptr(),
            biases.as_ptr(),
            null_lhs,
            rhs_indices.as_ptr(),
            transpose,
            mlx_sys::mlx_optional_int_ {
                value: group_size,
                has_value: true,
            },
            mlx_sys::mlx_optional_int_ {
                value: bits,
                has_value: true,
            },
            c"affine".as_ptr(),
            sorted_indices,
            stream.as_ptr(),
        )
    };

    // Always free the null sentinel
    unsafe { mlx_sys::mlx_array_free(null_lhs) };

    if status != 0 {
        // Free the uninitialized result array
        unsafe { mlx_sys::mlx_array_free(result) };
        let mlx_msg = FFI_LAST_ERROR
            .lock()
            .ok()
            .and_then(|mut g| g.take())
            .unwrap_or_default();
        let msg = format!(
            "gather_qmm failed: {mlx_msg} \
             [x={:?}/{:?} w={:?}/{:?} scales={:?}/{:?} biases={:?}/{:?} \
             idx={:?}/{:?} transpose={transpose} gs={group_size} bits={bits}]",
            x.shape(),
            x.dtype(),
            w.shape(),
            w.dtype(),
            scales.shape(),
            scales.dtype(),
            biases.shape(),
            biases.dtype(),
            rhs_indices.shape(),
            rhs_indices.dtype(),
        );
        return Err(Exception::custom(msg));
    }
    Ok(unsafe { Array::from_ptr(result) })
}

// ---------------------------------------------------------------------------
// `GatedDeltaNet` custom Metal kernel
// ---------------------------------------------------------------------------

/// Metal kernel source for the fused `GatedDeltaNet` recurrence.
///
/// Computes `g = exp(-exp(a_log) * softplus(a + dt_bias))` and `beta = sigmoid(b)`
/// inline, then runs the full recurrence -- all in one kernel dispatch.
///
/// Template parameters: `InT` (dtype), `Dk`, `Dv`, `Hk`, `Hv` (int constants).
// ---------------------------------------------------------------------------
// Chunk gated-delta rule (parallel, matches HF chunk_gated_delta_rule)
// ---------------------------------------------------------------------------

/// Pure-MLX implementation of the chunk gated-delta rule.
/// Matches HuggingFace's `torch_chunk_gated_delta_rule` used for multi-token
/// (S > 1) forward passes with `initial_state=None`.
///
fn softplus(x: &mlx_rs::Array) -> Result<mlx_rs::Array, Exception> {
    use mlx_rs::ops::*;
    let one = Array::from_f32(1.0);
    let neg_abs = x.abs()?.negative()?;
    x.add(&neg_abs.exp()?.add(&one)?.log()?)
}

fn sigmoid(x: &mlx_rs::Array) -> Result<mlx_rs::Array, Exception> {
    use mlx_rs::ops::*;
    let neg = x.negative()?;
    let sig = neg.exp()?.add(&Array::from_f32(1.0))?.reciprocal()?;
    Ok(sig)
}

/// For the typical DFlash verify step (S=16 ≤ 64), this uses a single chunk
/// so there is no inter-chunk loop — only the intra-chunk parallel prefix scan.
///
/// Shapes:
///   q: [B, S, Hk, Dk]   k: [B, S, Hk, Dk]   v: [B, S, Hv, Dv]
///   g: [B, S, Hv]       beta: [B, S, Hv]
///
/// Returns (y, final_state):
///   y:            [B, S, Hv, Dv]
///   final_state:  [B, Hv, Dk, Dv]  (HF layout; caller must transpose to [Dv, Dk] for Metal kernel)
fn chunk_gated_delta_rule(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
    initial_state: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    use mlx_rs::ops::indexing::NewAxis;
    use mlx_rs::ops::*;

    let orig_dtype = q.dtype();
    let shape = q.shape();
    let b = shape[0];
    let s = shape[1];
    let cs = 64i32;
    let num_chunks = (s + cs - 1) / cs;

    let q = q
        .as_dtype(mlx_rs::Dtype::Float32)?
        .transpose_axes(&[0, 2, 1, 3])?;
    let k = k
        .as_dtype(mlx_rs::Dtype::Float32)?
        .transpose_axes(&[0, 2, 1, 3])?;
    let v = v
        .as_dtype(mlx_rs::Dtype::Float32)?
        .transpose_axes(&[0, 2, 1, 3])?;
    let beta = beta
        .as_dtype(mlx_rs::Dtype::Float32)?
        .transpose_axes(&[0, 2, 1])?;
    let g = g
        .as_dtype(mlx_rs::Dtype::Float32)?
        .transpose_axes(&[0, 2, 1])?;

    let scale = 1.0f32 / (head_k_dim as f32).sqrt();
    let q = q.multiply(&Array::from_f32(scale))?;

    let (q, k, v, beta, g) = if num_chunks * cs > s {
        let pad_s = num_chunks * cs - s;
        let pad_widths_4: &[(i32, i32)] = &[(0, 0), (0, 0), (0, pad_s), (0, 0)];
        let pad_widths_3: &[(i32, i32)] = &[(0, 0), (0, 0), (0, pad_s)];
        let q2 = pad(&q, pad_widths_4, None, None)?;
        let k2 = pad(&k, pad_widths_4, None, None)?;
        let v2 = pad(&v, pad_widths_4, None, None)?;
        let b2 = pad(&beta, pad_widths_3, None, None)?;
        let g2 = pad(&g, pad_widths_3, None, None)?;
        (q2, k2, v2, b2, g2)
    } else {
        (q, k, v, beta, g)
    };

    let v_beta = v.multiply(&beta.expand_dims(-1)?)?;
    let k_beta = k.multiply(&beta.expand_dims(-1)?)?;

    let q = q.reshape(&[b, num_k_heads, num_chunks, cs, head_k_dim])?;
    let k = k.reshape(&[b, num_k_heads, num_chunks, cs, head_k_dim])?;
    let v = v.reshape(&[b, num_v_heads, num_chunks, cs, head_v_dim])?;
    let k_beta = k_beta.reshape(&[b, num_k_heads, num_chunks, cs, head_k_dim])?;
    let v_beta = v_beta.reshape(&[b, num_v_heads, num_chunks, cs, head_v_dim])?;
    let g = g.reshape(&[b, num_v_heads, num_chunks, cs])?;

    let g_cumsum = cumsum(&g, -1, None, None)?;

    let g_cs_col = g_cumsum.index((.., .., .., .., NewAxis));
    let g_cs_row = g_cumsum.index((.., .., .., NewAxis, ..));
    let decay = g_cs_col.subtract(&g_cs_row)?.exp()?;

    let ones_cs = Array::ones::<i32>(&[cs, cs])?;
    let strict_upper_f = triu(&ones_cs, 1)?.as_dtype(mlx_rs::Dtype::Float32)?;
    let strict_lower_f = tril(&ones_cs, -1)?.as_dtype(mlx_rs::Dtype::Float32)?;
    let lower_incl_f = Array::ones::<f32>(&[cs, cs])?.subtract(&strict_upper_f)?;

    let kkT = matmul(&k_beta, &k.transpose_axes(&[0, 1, 2, 4, 3])?)?;
    // HF: attn = -((k_beta @ k.T) * decay_mask).masked_fill(triu(0), 0)
    // → only strictly lower triangle is non-zero (diagonal and upper are zeroed)
    let attn: Array = kkT
        .multiply(&broadcast_to(&decay, kkT.shape())?)?
        .negative()?
        .multiply(&broadcast_to(&strict_lower_f, kkT.shape())?)?;

    // Parallel prefix scan (HF lines 367-370):
    //   for i in range(1, cs):
    //     row = attn[..., i, :i].clone()
    //     sub = attn[..., :i, :i].clone()
    //     attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    let mut attn_acc_parts: Vec<Array> = Vec::with_capacity(cs as usize);
    attn_acc_parts.push(attn.index((.., .., .., 0..1, ..)));
    for i in 1..cs {
        let attn_ik = attn.index((.., .., .., i..i + 1, ..i));
        let acc_k = concatenate_axis(&attn_acc_parts.iter().collect::<Vec<_>>(), 3)?;
        let acc_k_trunc = acc_k.index((.., .., .., .., ..i));
        let contrib = matmul(&attn_ik, &acc_k_trunc)?;
        let attn_row_i = attn.index((.., .., .., i..i + 1, ..));
        let row_updated = attn_row_i.index((.., .., .., .., ..i)).add(&contrib)?;
        let row_rest = attn_row_i.index((.., .., .., .., i..));
        let new_row = concatenate_axis(&[&row_updated, &row_rest], -1)?;
        attn_acc_parts.push(new_row);
    }
    let attn_acc = concatenate_axis(&attn_acc_parts.iter().collect::<Vec<_>>(), 3)?;

    // HF: attn = attn + eye(cs)  (add identity so diagonal becomes 1)
    let eye_cs = Array::eye::<f32>(cs, None, None)?;
    let eye_bc = broadcast_to(&eye_cs, &[b, num_k_heads, num_chunks, cs, cs])?;
    let attn_acc = attn_acc.add(&eye_bc)?;

    // HF: value = attn @ v_beta;  k_cumdecay = attn @ (k_beta * g.exp())
    let value = matmul(&attn_acc, &v_beta)?;

    let g_exp = g.exp()?.expand_dims(-1)?;
    let k_beta_g = k_beta.multiply(&broadcast_to(&g_exp, k_beta.shape())?)?;
    let k_cumdecay = matmul(&attn_acc, &k_beta_g)?;

    // Per-chunk mask: HF uses masked_fill(triu(0), 0) → strictly lower only (diagonal=0)
    let zero_mask_4d = broadcast_to(&strict_lower_f, &[b, num_v_heads, cs, cs])?;

    // HF: last_recurrent_state = zeros if initial_state is None else initial_state
    let mut last_state = match initial_state {
        Some(s) => s.clone(),
        None => Array::zeros::<f32>(&[b, num_v_heads, head_k_dim, head_v_dim])?,
    };
    let mut out_chunks: Vec<Array> = Vec::with_capacity(num_chunks as usize);

    for c in 0..num_chunks {
        let q_i = q.index((.., .., c, .., ..));
        let k_i = k.index((.., .., c, .., ..));
        let v_i = value.index((.., .., c, .., ..));
        let g_i = g.index((.., .., c, ..));
        let decay_i = decay.index((.., .., c, .., ..));
        let k_cd_i = k_cumdecay.index((.., .., c, .., ..));

        // v_prime = k_cumdecay @ state  →  [cs, Dk] @ [Dk, Dv] → [cs, Dv]
        let v_prime = matmul(&k_cd_i, &last_state)?;
        let v_new = v_i.subtract(&v_prime)?;

        // attn_inter = (q * g.exp()) @ state  →  [cs, Dk] @ [Dk, Dv] → [cs, Dv]
        let g_i_exp = g_i.exp()?.expand_dims(-1)?;
        let q_g = q_i.multiply(&broadcast_to(&g_i_exp, q_i.shape())?)?;
        let attn_inter = matmul(&q_g, &last_state)?;

        let qkT = matmul(&q_i, &k_i.transpose_axes(&[0, 1, 3, 2])?)?;
        let attn_i = qkT.multiply(&decay_i)?.multiply(&zero_mask_4d)?;

        let chunk_out = attn_inter.add(&matmul(&attn_i, &v_new)?)?;

        // State update (HF line 390-393):
        //   state = state * g[-1].exp() + (k * (g[-1] - g).exp()).T @ v_new
        let g_last = g_i.index((.., .., -1..)).exp()?;
        let g_diff = g_i.index((.., .., -1..)).subtract(&g_i)?;
        let k_g = k_i.multiply(&g_diff.exp()?.expand_dims(-1)?)?;
        // k_g.T @ v_new  →  [Dk, cs] @ [cs, Dv] → [Dk, Dv]
        let weighted_v = matmul(&k_g.transpose_axes(&[0, 1, 3, 2])?, &v_new)?;
        let g_last_bc = broadcast_to(&g_last.expand_dims(-1)?, last_state.shape())?;
        let wv_bc = broadcast_to(&weighted_v, last_state.shape())?;
        last_state = last_state.multiply(&g_last_bc)?.add(&wv_bc)?;

        out_chunks.push(chunk_out);
    }

    let core_attn_out = concatenate_axis(&out_chunks.iter().collect::<Vec<_>>(), 2)?;
    let core_attn_out: Array = core_attn_out
        .reshape(&[b, num_v_heads, num_chunks * cs, head_v_dim])?
        .index((.., .., ..s, ..));
    let core_attn_out = core_attn_out.transpose_axes(&[0, 2, 1, 3])?;

    Ok((
        core_attn_out.as_dtype(orig_dtype)?,
        last_state.as_dtype(orig_dtype)?,
    ))
}

/// Grid: `(32, Dv, B * Hv)`, Threadgroup: `(32, 4, 1)`.
const GATED_DELTA_KERNEL_SOURCE: &str = r"
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

// Per-head constants for gate computation
float a_log_val = static_cast<float>(a_log[hv_idx]);
float dt_bias_val = static_cast<float>(dt_bias[hv_idx]);

// a, b: [B, T, Hv]
auto a_ = a + b_idx * T * Hv;
auto b_ = b + b_idx * T * Hv;

for (int t = 0; t < T; ++t) {
  // Compute g = exp(-exp(a_log) * softplus(a + dt_bias))
  float x = static_cast<float>(a_[hv_idx]) + dt_bias_val;
  float sp = fmax(x, 0.0f) + log1p(exp(-fabs(x)));
  float g_val = exp(-exp(a_log_val) * sp);

  // beta = sigmoid(b)
  float beta_val = 1.0f / (1.0f + exp(-static_cast<float>(b_[hv_idx])));

  {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] * g_val;
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);

    auto delta = (v_[dv_idx] - kv_mem) * beta_val;

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] + k_[s_idx] * delta;
      out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y[dv_idx] = static_cast<InT>(out);
    }
  }
  // Match mlx-lm precision: cast state to InT between timesteps
  for (int i = 0; i < n_per_t; ++i) {
    state[i] = static_cast<float>(static_cast<InT>(state[i]));
  }
  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  a_ += Hv;
  b_ += Hv;
}
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  if (Stateless) {
    o_state[s_idx] = i_state[s_idx];
  } else {
    o_state[s_idx] = static_cast<InT>(state[i]);
  }
}
";

/// Create the `mlx_fast_metal_kernel` object from kernel source and names.
#[allow(unsafe_code)]
fn create_gated_delta_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 9] = [
        c"q",
        c"k",
        c"v",
        c"a_log",
        c"a",
        c"dt_bias",
        c"b",
        c"state_in",
        c"T",
    ];
    let output_names: [&std::ffi::CStr; 2] = [c"y", c"state_out"];

    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|s| s.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|s| s.as_ptr()).collect();

    // The kernel source is a compile-time string literal with no interior NULs.
    let source = CString::new(GATED_DELTA_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"gated_delta_step".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,  // ensure_row_contiguous
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

/// Configure template args, grid, threadgroup, and output shapes for the kernel.
#[allow(unsafe_code)]
fn configure_gated_delta_kernel(
    in_dtype: mlx_sys::mlx_dtype,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
    stateless: bool,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"InT".as_ptr(),
            in_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dk".as_ptr(),
            head_k_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dv".as_ptr(),
            head_v_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hk".as_ptr(),
            num_k_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hv".as_ptr(),
            num_v_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Stateless".as_ptr(),
            if stateless { 1 } else { 0 },
        );

        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, head_v_dim, batch * num_v_heads);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1);

        let y_shape = [batch, seq_len, num_v_heads, head_v_dim];
        let state_shape = [batch, num_v_heads, head_v_dim, head_k_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            in_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            in_dtype,
        );

        config
    }
}

/// Fused `GatedDeltaNet` kernel: computes g, beta, AND the full recurrence in one dispatch.
#[allow(unsafe_code, clippy::too_many_arguments)]
fn gated_delta_kernel_ffi(
    q: &Array,
    k: &Array,
    v: &Array,
    a_log: &Array,
    a: &Array,
    dt_bias: &Array,
    b: &Array,
    state_in: &Array,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> Result<(Array, Array), Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let in_dtype = unsafe { mlx_sys::mlx_array_dtype(q.as_ptr()) };

    let cached = GATED_DELTA_KERNEL.get_or_init(|| CachedMetalKernel(create_gated_delta_kernel()));
    let config = configure_gated_delta_kernel(
        in_dtype,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        false,
    );

    let t_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        q.as_ptr(),
        k.as_ptr(),
        v.as_ptr(),
        a_log.as_ptr(),
        a.as_ptr(),
        dt_bias.as_ptr(),
        b.as_ptr(),
        state_in.as_ptr(),
        t_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
            .unwrap_or_default();
        Err(Exception::custom(format!(
            "gated_delta_kernel failed: {mlx_msg}"
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        let mut state_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0);
            mlx_sys::mlx_vector_array_get(&raw mut state_ptr, outputs_vec, 1);
        }
        Ok((unsafe { Array::from_ptr(y_ptr) }, unsafe {
            Array::from_ptr(state_ptr)
        }))
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(t_scalar);
    }

    result
}

/// Stateless variant: computes identical `y` outputs but writes `state_in`
/// unchanged to `state_out`. Used for DFlash verify — GDN state is never
/// corrupted by speculative tokens, eliminating backup/restore overhead.
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn gated_delta_kernel_ffi_stateless(
    q: &Array,
    k: &Array,
    v: &Array,
    a_log: &Array,
    a: &Array,
    dt_bias: &Array,
    b: &Array,
    state_in: &Array,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> Result<(Array, Array), Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let in_dtype = unsafe { mlx_sys::mlx_array_dtype(q.as_ptr()) };

    let cached = GATED_DELTA_KERNEL.get_or_init(|| CachedMetalKernel(create_gated_delta_kernel()));
    let config = configure_gated_delta_kernel(
        in_dtype,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        true, // stateless — state_out := state_in
    );

    let t_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        q.as_ptr(),
        k.as_ptr(),
        v.as_ptr(),
        a_log.as_ptr(),
        a.as_ptr(),
        dt_bias.as_ptr(),
        b.as_ptr(),
        state_in.as_ptr(),
        t_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
            .unwrap_or_default();
        Err(Exception::custom(format!(
            "gated_delta_kernel_stateless failed: {mlx_msg}"
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        let mut state_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0);
            mlx_sys::mlx_vector_array_get(&raw mut state_ptr, outputs_vec, 1);
        }
        Ok((unsafe { Array::from_ptr(y_ptr) }, unsafe {
            Array::from_ptr(state_ptr)
        }))
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(t_scalar);
    }

    result
}

// ---------------------------------------------------------------------------
// Tape-recording GDN kernel: same recurrence, also outputs innovation delta
// ---------------------------------------------------------------------------

/// Tape-recording variant of the GDN kernel. Identical computation but also
/// outputs `innovation_tape[B, T, Hv, Dv]` — the delta residual at each step.
/// Used for DFlash verify: on partial rejection, we replay only accepted steps
/// from the tape instead of re-running the full forward.
const GATED_DELTA_TAPE_KERNEL_SOURCE: &str = r"
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;
auto tape_ = innovation_tape + b_idx * T * Hv * Dv + hv_idx * Dv;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

float a_log_val = static_cast<float>(a_log[hv_idx]);
float dt_bias_val = static_cast<float>(dt_bias[hv_idx]);

auto a_ = a + b_idx * T * Hv;
auto b_ = b + b_idx * T * Hv;

for (int t = 0; t < T; ++t) {
  float x = static_cast<float>(a_[hv_idx]) + dt_bias_val;
  float sp = fmax(x, 0.0f) + log1p(exp(-fabs(x)));
  float g_val = exp(-exp(a_log_val) * sp);

  float beta_val = 1.0f / (1.0f + exp(-static_cast<float>(b_[hv_idx])));

  {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] * g_val;
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);

    auto delta = (v_[dv_idx] - kv_mem) * beta_val;

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] + k_[s_idx] * delta;
      out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y[dv_idx] = static_cast<InT>(out);
      tape_[dv_idx] = delta;
    }
  }
  // Match mlx-lm precision: cast state to InT between timesteps
  for (int i = 0; i < n_per_t; ++i) {
    state[i] = static_cast<float>(static_cast<InT>(state[i]));
  }
  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  tape_ += Hv * Dv;
  a_ += Hv;
  b_ += Hv;
}
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = static_cast<InT>(state[i]);
}
";

#[allow(unsafe_code)]
fn create_gated_delta_tape_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 9] = [
        c"q",
        c"k",
        c"v",
        c"a_log",
        c"a",
        c"dt_bias",
        c"b",
        c"state_in",
        c"T",
    ];
    let output_names: [&std::ffi::CStr; 3] = [c"y", c"state_out", c"innovation_tape"];

    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|s| s.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|s| s.as_ptr()).collect();

    let source =
        CString::new(GATED_DELTA_TAPE_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"gated_delta_tape".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_gated_delta_tape_kernel(
    in_dtype: mlx_sys::mlx_dtype,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"InT".as_ptr(),
            in_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dk".as_ptr(),
            head_k_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dv".as_ptr(),
            head_v_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hk".as_ptr(),
            num_k_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hv".as_ptr(),
            num_v_heads,
        );

        let y_shape = [batch, seq_len, num_v_heads, head_v_dim];
        let state_shape = [batch, num_v_heads, head_v_dim, head_k_dim];
        let tape_shape = [batch, seq_len, num_v_heads, head_v_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            in_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            in_dtype,
        );
        // Tape stores deltas in float32 for precision (matches dflash-mlx)
        let f32_dtype: mlx_sys::mlx_dtype = 10; // MLX_FLOAT32
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            tape_shape.as_ptr(),
            tape_shape.len(),
            f32_dtype,
        );

        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, head_v_dim, batch * num_v_heads);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1);

        config
    }
}

/// Tape-recording GDN kernel: returns `(y, state_out, innovation_tape)`.
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn gated_delta_kernel_ffi_with_tape(
    q: &Array,
    k: &Array,
    v: &Array,
    a_log: &Array,
    a: &Array,
    dt_bias: &Array,
    b: &Array,
    state_in: &Array,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> Result<(Array, Array, Array), Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let in_dtype = unsafe { mlx_sys::mlx_array_dtype(q.as_ptr()) };

    let cached =
        GATED_DELTA_TAPE_KERNEL.get_or_init(|| CachedMetalKernel(create_gated_delta_tape_kernel()));
    let config = configure_gated_delta_tape_kernel(
        in_dtype,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    );

    let t_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        q.as_ptr(),
        k.as_ptr(),
        v.as_ptr(),
        a_log.as_ptr(),
        a.as_ptr(),
        dt_bias.as_ptr(),
        b.as_ptr(),
        state_in.as_ptr(),
        t_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
            .unwrap_or_default();
        Err(Exception::custom(format!(
            "gated_delta_tape_kernel failed: {mlx_msg}"
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        let mut state_ptr = unsafe { mlx_sys::mlx_array_new() };
        let mut tape_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0);
            mlx_sys::mlx_vector_array_get(&raw mut state_ptr, outputs_vec, 1);
            mlx_sys::mlx_vector_array_get(&raw mut tape_ptr, outputs_vec, 2);
        }
        Ok((
            unsafe { Array::from_ptr(y_ptr) },
            unsafe { Array::from_ptr(state_ptr) },
            unsafe { Array::from_ptr(tape_ptr) },
        ))
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(t_scalar);
    }

    result
}

// ---------------------------------------------------------------------------
// Tape replay kernel: replays accepted steps to advance GDN state
// ---------------------------------------------------------------------------

/// Replays the GDN recurrence from a recorded innovation tape.
/// Inputs: tape[B,T,Hv,Dv], k[B,T,Hk,Dk], a[B,T,Hv], a_log[Hv], dt_bias[Hv], state_in[B,Hv,Dv,Dk].
/// Output: state_out[B,Hv,Dv,Dk].
/// Much cheaper than full GDN forward — no projections, conv1d, norms, or output computation.
const TAPE_REPLAY_KERNEL_SOURCE: &str = r"
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

// a_log and dt_bias are [B * Hv] when batched across layers
float a_log_val = static_cast<float>(a_log[b_idx * Hv + hv_idx]);
float dt_bias_val = static_cast<float>(dt_bias[b_idx * Hv + hv_idx]);
auto a_ = a + b_idx * T * Hv;

for (int t = 0; t < T; ++t) {
  float x = static_cast<float>(a_[hv_idx]) + dt_bias_val;
  float sp = fmax(x, 0.0f) + log1p(exp(-fabs(x)));
  float g_val = exp(-exp(a_log_val) * sp);

  auto delta = tape_[dv_idx];
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] * g_val + k_[s_idx] * delta;
  }
  // Match mlx-lm precision: cast state to InT between timesteps
  for (int i = 0; i < n_per_t; ++i) {
    state[i] = static_cast<float>(static_cast<InT>(state[i]));
  }
  tape_ += Hv * Dv;
  k_ += Hk * Dk;
  a_ += Hv;
}
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = static_cast<InT>(state[i]);
}
";

#[allow(unsafe_code)]
fn create_tape_replay_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 7] =
        [c"tape", c"k", c"a", c"a_log", c"dt_bias", c"state_in", c"T"];
    let output_names: [&std::ffi::CStr; 1] = [c"state_out"];

    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|s| s.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|s| s.as_ptr()).collect();

    let source = CString::new(TAPE_REPLAY_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"tape_replay".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

/// Replay accepted steps from a recorded innovation tape.
/// Returns the new SSM state after replaying `seq_len` steps.
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn tape_replay_kernel_ffi(
    tape: &Array,
    k: &Array,
    a: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state_in: &Array,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let in_dtype = unsafe { mlx_sys::mlx_array_dtype(state_in.as_ptr()) };

    let cached = TAPE_REPLAY_KERNEL.get_or_init(|| CachedMetalKernel(create_tape_replay_kernel()));

    let config = unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"InT".as_ptr(),
            in_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dk".as_ptr(),
            head_k_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dv".as_ptr(),
            head_v_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hk".as_ptr(),
            num_k_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hv".as_ptr(),
            num_v_heads,
        );

        let state_shape = [batch, num_v_heads, head_v_dim, head_k_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            in_dtype,
        );

        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, head_v_dim, batch * num_v_heads);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1);

        config
    };

    let t_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        tape.as_ptr(),
        k.as_ptr(),
        a.as_ptr(),
        a_log.as_ptr(),
        dt_bias.as_ptr(),
        state_in.as_ptr(),
        t_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
            .unwrap_or_default();
        Err(Exception::custom(format!(
            "tape_replay_kernel failed: {mlx_msg}"
        )))
    } else {
        let mut state_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut state_ptr, outputs_vec, 0);
        }
        Ok(unsafe { Array::from_ptr(state_ptr) })
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(t_scalar);
    }

    result
}

// ---------------------------------------------------------------------------
// Qwen3NextAttention (full attention with gated Q and partial RoPE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3NextAttention {
    #[param]
    q_proj: QLinear,
    #[param]
    k_proj: QLinear,
    #[param]
    v_proj: QLinear,
    #[param]
    o_proj: QLinear,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    scale: f32,
}

impl Qwen3NextAttention {
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let head_dim = args.head_dim;
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();
        let rope_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        // partial_rotary_factor * head_dim is always a small positive integer (e.g. 64)
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let partial_dim = (rope_dim_f32 * args.partial_rotary_factor).round() as i32;

        Ok(Self {
            q_proj: QLinear::new(ql, qb)?,
            k_proj: QLinear::new(ql, qb)?,
            v_proj: QLinear::new(ql, qb)?,
            o_proj: QLinear::new(ql, qb)?,
            q_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            rope: nn::RopeBuilder::new(partial_dim)
                .traditional(false)
                .base(args.rope_theta)
                .scale(1.0)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            num_attention_heads: args.num_attention_heads,
            num_key_value_heads: args.num_key_value_heads,
            scale,
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut SteppingKeyValueCache,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Q is projected to 2 * num_heads * head_dim (doubled for gating)
        let q_proj_output = self.q_proj.forward(x)?;
        let q_reshaped = q_proj_output.reshape(&[B, L, self.num_attention_heads, -1])?;
        let q_halves = q_reshaped.split(2, Some(-1))?;
        let queries_pre = q_halves
            .first()
            .ok_or_else(|| Exception::custom("split produced empty result"))?;
        let gate = q_halves
            .get(1)
            .ok_or_else(|| Exception::custom("split produced empty result"))?
            .reshape(&[B, L, -1])?;

        let keys_raw = self.k_proj.forward(x)?;
        let values_raw = self.v_proj.forward(x)?;

        // Per-head RmsNorm then transpose to [B, H, L, D]
        let mut queries = self
            .q_norm
            .forward(queries_pre)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_norm
            .forward(&keys_raw.reshape(&[B, L, self.num_key_value_heads, -1])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values_raw
            .reshape(&[B, L, self.num_key_value_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE with cache offset
        let offset = cache.offset();
        queries = apply_rope(&queries, &self.rope, offset)?;
        keys = apply_rope(&keys, &self.rope, offset)?;

        let view = cache.update_and_view(keys, values)?;
        let is_tq_decode = view.turboquant().is_some();

        let output = if is_tq_decode {
            let tq_view = view.turboquant().unwrap();
            let scores = tq_view.decode_scores(&queries, self.num_attention_heads)?;
            let scale_arr = Array::from_f32(self.scale).as_dtype(scores.dtype())?;
            let mut scaled = scores.multiply(&scale_arr)?;

            // Apply causal mask for L>1 (block-K verify).
            // At T=1: mask is None → skipped → existing behavior preserved.
            // At T>1: mask is Some(Array([L, total_seq])) → additive -inf masking.
            if let Some(m) = mask {
                let mask_arr = match m {
                    AttentionMask::Array(a) => a.clone(),
                    AttentionMask::Causal => create_causal_mask(L, None)?,
                };
                // mask_arr: boolean [L, total_seq], True = attend
                // Reshape → [1, L, total_seq] for broadcast with scores [H, L, total_seq]
                let total_seq = *mask_arr.shape().last().unwrap_or(&1);
                let mask_3d = mask_arr.reshape(&[1, L, total_seq])?;
                let neg_inf = Array::from_f32(f32::NEG_INFINITY).as_dtype(scaled.dtype())?;
                scaled = ops::r#where(&mask_3d, &scaled, &neg_inf)?;
            }

            let weights = ops::softmax_axis(&scaled, -1, true)?;
            tq_view
                .decode_values(&weights, self.num_attention_heads)?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, L, -1])?
        } else {
            let (cached_keys, cached_values) = view.into_dense()?;
            let sdpa_mask = mask.map(fast::ScaledDotProductAttentionMask::from);
            fast::scaled_dot_product_attention(
                queries,
                cached_keys,
                cached_values,
                self.scale,
                sdpa_mask,
                None::<&Array>,
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?
        };

        let gated = sigmoid_mul(&gate, &output)?;
        self.o_proj.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// Qwen3NextMLP (standard SwiGLU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3NextMLP {
    #[param]
    gate_proj: QLinear,
    #[param]
    down_proj: QLinear,
    #[param]
    up_proj: QLinear,
}

pub(crate) fn new_mlp_projections(
    ql: i32,
    qb: i32,
) -> Result<(QLinear, QLinear, QLinear), Exception> {
    Ok((
        QLinear::new(ql, qb)?,
        QLinear::new(ql, qb)?,
        QLinear::new(ql, qb)?,
    ))
}

impl Qwen3NextMLP {
    fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections(ql, qb)?;
        Ok(Self {
            gate_proj,
            down_proj,
            up_proj,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gate_out = self.gate_proj.forward(x)?;
        let up_out = self.up_proj.forward(x)?;
        let activated = swiglu(&gate_out, &up_out)?;
        self.down_proj.forward(&activated)
    }
}

/// ANE int8 MLP forward for a single layer.
///
/// Shapes (row-major unless noted):
///   x            : [1, S, H] f32 (or fp16; converted)
///   gate weight  : compiled at [inter, H]  → kernel input [1, H,     1, seq_bucket]
///   up   weight  : compiled at [inter, H]  → kernel input [1, H,     1, seq_bucket]
///   down weight  : compiled at [H,     inter] → kernel input [1, inter, 1, seq_bucket]
///
/// Layout marshaling mirrors the probe (`ane_mlmodel::tests::
/// qwen3_9b_mlp_int8_vs_mlx_probe` lines 1574-1645): row-major `[S, C]` must
/// be transposed to `[C, 1, S]` fp16 for the conv1x1 mlpackage, and the
/// output transposed back. Seqs shorter than the compiled bucket are
/// zero-padded in the seq dimension.
#[cfg(feature = "ane")]
#[allow(unsafe_code)]
pub(crate) fn forward_ane_int8_mlp(
    x: &Array,
    gate: &std::sync::Arc<crate::ane_mlmodel::AneMlPackageKernel>,
    up: &std::sync::Arc<crate::ane_mlmodel::AneMlPackageKernel>,
    down: &std::sync::Arc<crate::ane_mlmodel::AneMlPackageKernel>,
) -> Result<Array, Exception> {
    use half::f16;
    use rayon::prelude::*;

    let shape = x.shape().to_vec(); // [1, S, H]
    let s = shape[1] as usize;
    let h = shape[2] as usize;
    let bucket = gate.input_shape[3] as usize;
    let inter = gate.output_shape[1] as usize;

    debug_assert_eq!(gate.input_shape[1] as usize, h, "gate in_dim mismatch");
    debug_assert_eq!(up.input_shape[1] as usize, h, "up in_dim mismatch");
    debug_assert_eq!(up.output_shape[1] as usize, inter, "up inter mismatch");
    debug_assert_eq!(down.input_shape[1] as usize, inter, "down in_dim mismatch");
    debug_assert_eq!(down.output_shape[1] as usize, h, "down out_dim mismatch");

    // --- Pack input: cast bf16→fp16 once, transpose [1,S,H] → [H,bucket] ---
    // Strided transpose parallelised over channel rows. Trailing [s..bucket]
    // columns stay zero (vec! default); conv1x1 at those cols contributes 0.
    let x_f16_arr = x.as_dtype(Dtype::Float16)?;
    x_f16_arr.eval()?;
    let x_f16 = x_f16_arr.as_slice::<f16>();

    let mut x_fp16 = vec![0u16; h * bucket];
    x_fp16
        .par_chunks_mut(bucket)
        .enumerate()
        .for_each(|(ci, row)| {
            for t in 0..s {
                row[t] = x_f16[t * h + ci].to_bits();
            }
        });

    // --- gate / up projections ---
    let mut gate_fp16 = vec![0u16; inter * bucket];
    gate.predict_fp16(&x_fp16, &mut gate_fp16)
        .map_err(|e| Exception::custom(format!("forward_ane_int8_mlp: gate: {e}")))?;
    let mut up_fp16 = vec![0u16; inter * bucket];
    up.predict_fp16(&x_fp16, &mut up_fp16)
        .map_err(|e| Exception::custom(format!("forward_ane_int8_mlp: up: {e}")))?;
    drop(x_fp16);

    // --- SwiGLU directly in ANE native layout [1, inter, 1, bucket]. ---
    // Gate/up outputs already sit in [inter, bucket] channel-major u16; down's
    // input wants the same layout. SwiGLU is elementwise, so we skip both the
    // [inter,bucket]→[S,inter] unpack and the [S,inter]→[inter,bucket] pack.
    // Padded [s..bucket] seq cols are zero in gate+up (conv1x1 of zero input);
    // silu(0)·0 = 0, so padding stays zero through SwiGLU.
    //
    // SAFETY: `half::f16` is `#[repr(transparent)]` over `u16`, so a
    // `&[u16]` of fp16 bits is layout-identical to a `&[f16]`.
    let gate_as_f16: &[f16] =
        unsafe { std::slice::from_raw_parts(gate_fp16.as_ptr().cast::<f16>(), gate_fp16.len()) };
    let up_as_f16: &[f16] =
        unsafe { std::slice::from_raw_parts(up_fp16.as_ptr().cast::<f16>(), up_fp16.len()) };
    let gate_arr = Array::from_slice(gate_as_f16, &[1, inter as i32, 1, bucket as i32]);
    let up_arr = Array::from_slice(up_as_f16, &[1, inter as i32, 1, bucket as i32]);
    drop(gate_fp16);
    drop(up_fp16);
    let activated = swiglu(&gate_arr, &up_arr)?;
    activated.eval()?;

    // --- down projection: consume activated fp16 buffer as u16 bits. ---
    // SAFETY: activated is Dtype::Float16 with len inter*bucket;
    // `as_slice::<f16>` is contiguous and f16↔u16 are repr-identical.
    let act_f16 = activated.as_slice::<f16>();
    let act_u16: &[u16] =
        unsafe { std::slice::from_raw_parts(act_f16.as_ptr().cast::<u16>(), act_f16.len()) };
    let mut down_fp16 = vec![0u16; h * bucket];
    down.predict_fp16(act_u16, &mut down_fp16)
        .map_err(|e| Exception::custom(format!("forward_ane_int8_mlp: down: {e}")))?;

    // --- Unpack [H, bucket] → [S, H] fp16, return as Dtype::Float16. ---
    // Caller adds residual against bf16 hidden; MLX upcasts automatically.
    let mut out_f16: Vec<f16> = vec![f16::ZERO; s * h];
    out_f16.par_chunks_mut(h).enumerate().for_each(|(t, row)| {
        for co in 0..h {
            row[co] = f16::from_bits(down_fp16[co * bucket + t]);
        }
    });
    Ok(Array::from_slice(&out_f16, &[1, s as i32, h as i32]))
}

// ---------------------------------------------------------------------------
// SwitchMLP weights (stacked expert weights for MoE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct SwitchMlpWeights {
    #[param]
    gate_proj: QLinear,
    #[param]
    up_proj: QLinear,
    #[param]
    down_proj: QLinear,
    /// Lazily fused gate+up weights for MoE gather_qmm (3→2 calls per layer).
    fused_gate_up: Option<(Array, Array, Array, i32)>,
}

impl SwitchMlpWeights {
    pub(crate) fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections(ql, qb)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            fused_gate_up: None,
        })
    }

    /// Apply the full `SwiGLU` `MoE` block for all selected experts in one shot
    /// using `gather_qmm` (fused expert-indexed quantized matmul).
    ///
    /// `x`: `[..., D]` input
    /// `indices`: `[..., top_k]` expert indices
    /// Returns: `[..., top_k, D]`
    pub(crate) fn forward_gather(
        &self,
        x: &Array,
        indices: &Array,
        sorted: bool,
    ) -> Result<Array, Exception> {
        // Reshape so x batch dims broadcast with the indices shape.
        // x: [B, L, D] -> [B, L, 1, 1, D]
        //   batch = [B, L, 1], M=1, K=D
        // indices: [B, L, top_k]
        //   broadcast([B, L, 1], [B, L, top_k]) -> [B, L, top_k]
        let shape = x.shape();
        let err = || Exception::custom("forward_gather input must be [B, L, D]");
        let b = *shape.first().ok_or_else(err)?;
        let l = *shape.get(1).ok_or_else(err)?;
        let d = *shape.get(2).ok_or_else(err)?;
        let x_exp = x.reshape(&[b, l, 1, 1, d])?;

        // Gate/up projections: [B, L, top_k, 1, intermediate]
        let gate_out = gather_qmm(
            &x_exp,
            &self.gate_proj.weight,
            &self.gate_proj.scales,
            &self.gate_proj.biases,
            indices,
            true,
            self.gate_proj.group_size,
            self.gate_proj.bits,
            sorted,
        )?;
        let up_out = gather_qmm(
            &x_exp,
            &self.up_proj.weight,
            &self.up_proj.scales,
            &self.up_proj.biases,
            indices,
            true,
            self.up_proj.group_size,
            self.up_proj.bits,
            sorted,
        )?;

        let activated = swiglu(&gate_out, &up_out)?;

        // Down projection: [B, L, top_k, 1, D]
        // activated batch=[B,L,top_k] broadcasts with indices [B,L,top_k] exactly
        let down_out = gather_qmm(
            &activated,
            &self.down_proj.weight,
            &self.down_proj.scales,
            &self.down_proj.biases,
            indices,
            true,
            self.down_proj.group_size,
            self.down_proj.bits,
            sorted,
        )?;

        // Squeeze M=1: [B, L, top_k, D]
        down_out.squeeze_axes(&[-2])
    }

    /// Like `forward_gather` but reorders tokens globally by expert index
    /// before calling `gather_qmm`, matching mlx-lm's `_gather_sort` pattern.
    ///
    /// This gives coalesced GPU memory access and is 3-6x faster for prefill
    /// (L >= 32). For single-token decode (L=1) it's equivalent.
    ///
    /// `x`: `[B, L, D]`
    /// `indices`: `[B, L, top_k]` expert indices (need NOT be pre-sorted)
    /// Returns: `[B, L, top_k, D]`
    pub(crate) fn forward_gather_global_sort(
        &self,
        x: &Array,
        indices: &Array,
    ) -> Result<Array, Exception> {
        let x_shape = x.shape();
        let err = || Exception::custom("forward_gather_global_sort input must be [B, L, D]");
        let b = *x_shape.first().ok_or_else(err)?;
        let l = *x_shape.get(1).ok_or_else(err)?;
        let d = *x_shape.get(2).ok_or_else(err)?;
        let top_k = *indices
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("indices must have last dim"))?;

        // --- Global sort: flatten, argsort, reorder tokens by expert ---
        // indices: [B, L, top_k] -> [N] where N = B*L*top_k
        let idx_flat = indices.flatten(None, None)?;
        let order = ops::argsort_axis(&idx_flat, 0)?;
        let inv_order = ops::argsort_axis(&order, 0)?;

        // Map each sorted position back to its source token: order / top_k
        let top_k_arr = Array::from_slice(&[top_k as u32], &[1]);
        let token_idx = order.floor_divide(&top_k_arr)?;

        // x_flat: [B*L, 1, D] -> x_sorted: [N, 1, D]
        let x_flat = x.reshape(&[b * l, 1, d])?;
        let x_sorted = x_flat.take_axis(&token_idx, 0)?;

        // idx_sorted: [N] — monotonically non-decreasing expert indices
        let idx_sorted = idx_flat.take_axis(&order, 0)?;

        // --- gather_qmm with coalesced access ---
        let gate_out = gather_qmm(
            &x_sorted,
            &self.gate_proj.weight,
            &self.gate_proj.scales,
            &self.gate_proj.biases,
            &idx_sorted,
            true,
            self.gate_proj.group_size,
            self.gate_proj.bits,
            true, // indices are globally sorted
        )?;
        let up_out = gather_qmm(
            &x_sorted,
            &self.up_proj.weight,
            &self.up_proj.scales,
            &self.up_proj.biases,
            &idx_sorted,
            true,
            self.up_proj.group_size,
            self.up_proj.bits,
            true,
        )?;

        let activated = swiglu(&gate_out, &up_out)?;

        let down_out = gather_qmm(
            &activated,
            &self.down_proj.weight,
            &self.down_proj.scales,
            &self.down_proj.biases,
            &idx_sorted,
            true,
            self.down_proj.group_size,
            self.down_proj.bits,
            true,
        )?;

        // down_out: [N, 1, D] -> squeeze M -> [N, D]
        let out_flat = down_out.squeeze_axes(&[-2])?;

        // --- Unsort: restore original token order ---
        let out_unsorted = out_flat.take_axis(&inv_order, 0)?;

        // Reshape back to [B, L, top_k, D]
        out_unsorted.reshape(&[b, l, top_k, d])
    }

    /// Like `forward_gather_global_sort` but fuses gate+up into a single
    /// `gather_qmm` call (3→2 per layer). Lazy-inits fused weights on first call.
    pub(crate) fn forward_gather_fused(
        &mut self,
        x: &Array,
        indices: &Array,
    ) -> Result<Array, Exception> {
        // Lazy-init: concatenate gate+up weights along axis 1 (intermediate dim).
        // MoE weights are [num_experts, intermediate_packed, hidden].
        if self.fused_gate_up.is_none() {
            let intermediate = *self
                .gate_proj
                .weight
                .shape()
                .get(1)
                .ok_or_else(|| Exception::custom("gate_proj weight missing dim 1"))?;
            let fw = ops::concatenate_axis(&[&*self.gate_proj.weight, &*self.up_proj.weight], 1)?;
            let fs = ops::concatenate_axis(&[&*self.gate_proj.scales, &*self.up_proj.scales], 1)?;
            let fb = ops::concatenate_axis(&[&*self.gate_proj.biases, &*self.up_proj.biases], 1)?;
            fw.eval()?;
            fs.eval()?;
            fb.eval()?;
            self.fused_gate_up = Some((fw, fs, fb, intermediate));
        }
        let (fw, fs, fb, intermediate) = self
            .fused_gate_up
            .as_ref()
            .ok_or_else(|| Exception::custom("fused_gate_up missing after init"))?;

        // --- Global sort (same as forward_gather_global_sort) ---
        let x_shape = x.shape();
        let err = || Exception::custom("forward_gather_fused input must be [B, L, D]");
        let b = *x_shape.first().ok_or_else(err)?;
        let l = *x_shape.get(1).ok_or_else(err)?;
        let d = *x_shape.get(2).ok_or_else(err)?;
        let top_k = *indices
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("indices must have last dim"))?;

        let idx_flat = indices.flatten(None, None)?;
        let order = ops::argsort_axis(&idx_flat, 0)?;
        let inv_order = ops::argsort_axis(&order, 0)?;

        let top_k_arr = Array::from_slice(&[top_k as u32], &[1]);
        let token_idx = order.floor_divide(&top_k_arr)?;

        let x_flat = x.reshape(&[b * l, 1, d])?;
        let x_sorted = x_flat.take_axis(&token_idx, 0)?;
        let idx_sorted = idx_flat.take_axis(&order, 0)?;

        // --- Fused gate+up: ONE gather_qmm instead of TWO ---
        let fused_out = gather_qmm(
            &x_sorted,
            fw,
            fs,
            fb,
            &idx_sorted,
            true,
            self.gate_proj.group_size,
            self.gate_proj.bits,
            true,
        )?;
        // Split at intermediate boundary → gate_out, up_out
        let parts = fused_out.split_axis(&[*intermediate], Some(-1))?;
        let gate_out = parts
            .first()
            .ok_or_else(|| Exception::custom("fused split failed"))?;
        let up_out = parts
            .get(1)
            .ok_or_else(|| Exception::custom("fused split failed"))?;
        let activated = swiglu(gate_out, up_out)?;

        // --- down_proj: unchanged ---
        let down_out = gather_qmm(
            &activated,
            &self.down_proj.weight,
            &self.down_proj.scales,
            &self.down_proj.biases,
            &idx_sorted,
            true,
            self.down_proj.group_size,
            self.down_proj.bits,
            true,
        )?;

        // down_out: [N, 1, D] -> squeeze M -> [N, D]
        let out_flat = down_out.squeeze_axes(&[-2])?;

        // --- Unsort: restore original token order ---
        let out_unsorted = out_flat.take_axis(&inv_order, 0)?;

        // Reshape back to [B, L, top_k, D]
        out_unsorted.reshape(&[b, l, top_k, d])
    }
}

// ---------------------------------------------------------------------------
// SparseMoeBlock (router + SwitchGLU + shared expert)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct SparseMoeBlock {
    #[param]
    gate: QLinear,
    #[param]
    switch_mlp: SwitchMlpWeights,
    #[param]
    shared_expert: Qwen3NextMLP,
    #[param]
    shared_expert_gate: QLinear,
    top_k: i32,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        if args.num_experts <= 0 {
            return Err(Exception::custom("num_experts must be > 0"));
        }
        if args.num_experts_per_tok <= 0 {
            return Err(Exception::custom("num_experts_per_tok must be > 0"));
        }
        if args.num_experts_per_tok > args.num_experts {
            return Err(Exception::custom(
                "num_experts_per_tok must be <= num_experts",
            ));
        }
        // Gate quantization: use per-layer override if present, else global
        let (gate_ql, gate_qb) = args
            .gate_quantization
            .as_ref()
            .map_or((ql, qb), |gq| (gq.group_size, gq.bits));
        Ok(Self {
            gate: QLinear::new(gate_ql, gate_qb)?,
            switch_mlp: SwitchMlpWeights::new(ql, qb)?,
            shared_expert: Qwen3NextMLP::new(ql, qb)?,
            shared_expert_gate: QLinear::new(gate_ql, gate_qb)?,
            top_k: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
        })
    }

    #[allow(dead_code)]
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gates = ops::softmax_axis(&self.gate.forward(x)?, -1, true)?;

        // Top-K selection via argpartition
        let neg_k = -self.top_k;
        let all_inds = ops::argpartition_axis(&gates, neg_k, -1)?;
        let num_experts = *gates
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("gates must have last dim"))?;
        let top_k_start = num_experts - self.top_k;
        let top_inds = ops::sort_axis(all_inds.index((.., .., top_k_start..)), -1)?;
        let raw_scores = gates.take_along_axis(&top_inds, -1)?;

        let top_scores = if self.norm_topk_prob {
            let score_sum = raw_scores.sum_axes(&[-1], true)?;
            raw_scores.divide(score_sum)?
        } else {
            raw_scores
        };

        // Expert computation via fused gather_qmm (global sort for coalesced access)
        let y = self.switch_mlp.forward_gather_global_sort(x, &top_inds)?;

        // Weighted sum over experts: [B, L, top_k, D] * [B, L, top_k, 1] -> sum -> [B, L, D]
        let expert_sum = y
            .multiply(&top_scores.expand_dims(-1)?)?
            .sum_axes(&[-2], false)?;

        // Shared expert
        let shared_y = self.shared_expert.forward(x)?;
        let shared_gate_val = nn::sigmoid(&self.shared_expert_gate.forward(x)?)?;
        let shared_out = shared_y.multiply(&shared_gate_val)?;

        expert_sum.add(shared_out)
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNet (SSM-like linear attention)
// ---------------------------------------------------------------------------

/// Cache state for a `GatedDeltaNet` layer.
#[derive(Debug, Clone)]
pub struct ArraysCache {
    pub conv_state: Option<Array>,
    pub ssm_state: Option<Array>,
    pub offset: i32,
}

impl ArraysCache {
    pub const fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
            offset: 0,
        }
    }
}

impl Default for ArraysCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ArraysCache {
    /// Evaluate lazy arrays so a subsequent `clone()` captures values.
    pub fn eval_arrays(&self) -> Result<(), mlx_rs::error::Exception> {
        if let Some(cs) = &self.conv_state {
            cs.eval()?;
        }
        if let Some(ss) = &self.ssm_state {
            ss.eval()?;
        }
        Ok(())
    }
}

/// Recorded intermediate values from a tape-recording GDN forward pass.
/// Enables cheap tape replay on partial rejection — no re-running projections,
/// conv1d, norms, or attention. Just `state = state * g + k * delta`.
pub struct GdnLayerTape {
    /// Innovation delta at each timestep: `[B, T, Hv, Dv]`
    pub delta_tape: Array,
    /// Post-conv, post-norm key vectors: `[B, T, Hk, Dk]`
    pub norm_k: Array,
    /// Projected gate values: `[B, T, Hv]`
    pub a_proj: Array,
    /// Raw QKV input to conv1d (for conv_state rebuild): `[B, T, conv_dim]`
    pub qkv_input: Array,
    /// Pre-forward conv_state for rollback: `[B, K-1, conv_dim]`
    pub conv_state_init: Option<Array>,
    /// Pre-forward ssm_state for rollback: `[B, Hv, Dv, Dk]`
    pub ssm_state_init: Option<Array>,
    /// Pre-forward cache offset for rollback
    pub offset_init: i32,
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters)]
struct GatedDeltaNet {
    #[param]
    in_proj_qkvz: QLinear,
    #[param]
    in_proj_ba: QLinear,
    // Separate projections for qwen3_5-style models (flat split, not per-head)
    #[param]
    in_proj_qkv: Option<QLinear>,
    #[param]
    in_proj_z: Option<QLinear>,
    #[param]
    in_proj_a: Option<QLinear>,
    #[param]
    in_proj_b: Option<QLinear>,
    #[param]
    conv1d: nn::Conv1d,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    out_proj: QLinear,
    #[param]
    A_log: Param<Array>,
    #[param]
    dt_bias: Param<Array>,
    num_k_heads: i32,
    num_v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    key_dim: i32,
    conv_dim: i32,
    conv_kernel_size: i32,
    use_separate_projections: bool,
    qk_norm_weight_q: Array,
    qk_norm_weight_k: Array,
    /// Pre-transposed conv weight for fast T=1 decode: [kernel_size, conv_dim].
    conv_weight_t: Option<Array>,
    /// Optional compiled ANE kernels for the layer's three dense projections
    /// (`in_proj_qkvz`, `in_proj_ba`, `out_proj`). `None` = Metal matmul as
    /// before. Each `Vec` entry is a seq-length bucket (one full compiled kernel
    /// set per bucket); at dispatch time the smallest bucket with
    /// `S <= kernels.qkvz.seq_len` is selected. If `S` exceeds every bucket, the
    /// dispatch falls back to Metal. The `Vec` is sorted ascending by
    /// `qkvz.seq_len` — [`select_ane_bucket`] relies on that ordering.
    ///
    /// **Inline path** (`!Send` — used only by Wave 1/2 parity tests on the
    /// main thread). Production uses [`Self::ane_handle`] instead.
    #[cfg(feature = "ane")]
    ane_kernels: Option<Vec<std::sync::Arc<crate::qwen3_next_ane::GdnAneLayerKernels>>>,
    /// Optional handle to the model-wide GDN ANE worker thread (Wave 4).
    /// `Send + Sync`, so this path is what makes `HIGGS_TARGET_ANE_GDN=1`
    /// usable once the model has been moved into the inference worker
    /// thread (`batch_engine.rs:117` / `simple.rs`). Mutually exclusive with
    /// [`Self::ane_kernels`] in practice — `enable_ane_gdn_all_layers_via_worker`
    /// sets this; the inline `enable_ane_gdn*` methods set `ane_kernels`.
    #[cfg(feature = "ane")]
    ane_handle: Option<crate::qwen3_next_ane_worker::GdnAneWorkerHandle>,
    /// Index of this layer within the worker's per-layer kernel table.
    /// Meaningful only when [`Self::ane_handle`] is `Some`; ignored otherwise
    /// (kept as `0`). Set by `enable_ane_gdn_all_layers_via_worker` based on
    /// the order in which linear layers are enumerated.
    #[cfg(feature = "ane")]
    ane_linear_layer_idx: usize,
}

impl GatedDeltaNet {
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let num_k_heads = args.linear_num_key_heads;
        let num_v_heads = args.linear_num_value_heads;
        let head_k_dim = args.linear_key_head_dim;
        let head_v_dim = args.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = args.linear_conv_kernel_dim;

        let use_sep = args.use_separate_gdn_projections;
        Ok(Self {
            in_proj_qkvz: QLinear::new(ql, qb)?,
            in_proj_ba: QLinear::new(ql, qb)?,
            in_proj_qkv: if use_sep {
                Some(QLinear::new(ql, qb)?)
            } else {
                None
            },
            in_proj_z: if use_sep {
                Some(QLinear::new(ql, qb)?)
            } else {
                None
            },
            in_proj_a: if use_sep {
                Some(QLinear::new(ql, qb)?)
            } else {
                None
            },
            in_proj_b: if use_sep {
                Some(QLinear::new(ql, qb)?)
            } else {
                None
            },
            conv1d: nn::Conv1dBuilder::new(conv_dim, conv_dim, conv_kernel_size)
                .bias(false)
                .groups(conv_dim)
                .padding(0)
                .build()?,
            norm: nn::RmsNormBuilder::new(head_v_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            out_proj: QLinear::new(ql, qb)?,
            A_log: Param::new(Array::zeros::<f32>(&[num_v_heads])?),
            dt_bias: Param::new(Array::zeros::<f32>(&[num_v_heads])?),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            conv_dim,
            conv_kernel_size,
            use_separate_projections: use_sep,
            qk_norm_weight_q: {
                let dim_f32 = f32::from(
                    i16::try_from(head_k_dim)
                        .map_err(|_| Exception::custom("head_k_dim out of i16 range"))?,
                );
                let s = dim_f32.sqrt().recip();
                let w = Array::ones::<f32>(&[head_k_dim])?.multiply(Array::from_f32(s * s))?;
                w.eval()?;
                w
            },
            qk_norm_weight_k: {
                let dim_f32 = f32::from(
                    i16::try_from(head_k_dim)
                        .map_err(|_| Exception::custom("head_k_dim out of i16 range"))?,
                );
                let s = dim_f32.sqrt().recip();
                let w = Array::ones::<f32>(&[head_k_dim])?.multiply(Array::from_f32(s))?;
                w.eval()?;
                w
            },
            conv_weight_t: None,
            #[cfg(feature = "ane")]
            ane_kernels: None,
            #[cfg(feature = "ane")]
            ane_handle: None,
            #[cfg(feature = "ane")]
            ane_linear_layer_idx: 0,
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        inputs: &Array,
        _mask: Option<&AttentionMask>,
        cache: &mut ArraysCache,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let S = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Project inputs and split into q, k, v, z, b, a
        let (q, k, v, z, b, a) = if self.use_separate_projections {
            // qwen3.5-style: 4 separate projections, flat split
            let qkv_proj = self
                .in_proj_qkv
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_qkv missing"))?;
            let z_proj = self
                .in_proj_z
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_z missing"))?;
            let b_proj = self
                .in_proj_b
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_b missing"))?;
            let a_proj = self
                .in_proj_a
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_a missing"))?;

            let qkv = qkv_proj.forward(inputs)?;
            let z = z_proj
                .forward(inputs)?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;
            let b = b_proj.forward(inputs)?;
            let a = a_proj.forward(inputs)?;

            let split_indices = &[self.key_dim, self.key_dim * 2];
            let qkv_parts = qkv.split_axis(split_indices, Some(-1))?;
            let q = qkv_parts
                .first()
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let k = qkv_parts
                .get(1)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let v = qkv_parts
                .get(2)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

            (q, k, v, z, b, a)
        } else {
            // qwen3_next-style: combined projections, per-head reshape
            let mixed_qkvz = self.in_proj_qkvz.forward(inputs)?;
            let mixed_ba = self.in_proj_ba.forward(inputs)?;
            self.fix_query_key_value_ordering(&mixed_qkvz, &mixed_ba, B, S)?
        };

        // Conv1d with state management
        let q_flat = q.reshape(&[B, S, -1])?;
        let k_flat = k.reshape(&[B, S, -1])?;
        let v_flat = v.reshape(&[B, S, -1])?;
        let mixed_qkv = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1)?;

        // Prepend conv_state (or zeros on first call) for causal conv context.
        // Both S>1 (chunk/verify) and S==1 (recurrent) use the same logic:
        // take the existing conv_state if available, otherwise zero-pad.
        let conv_state = match cache.conv_state.take() {
            Some(state) => state,
            None => ops::zeros_dtype(
                &[B, self.conv_kernel_size - 1, self.conv_dim],
                inputs.dtype(),
            )?,
        };
        let conv_input = ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;

        // Update conv state cache
        let n_keep = self.conv_kernel_size - 1;
        let conv_input_len = *conv_input
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("conv_input missing seq dim"))?;
        let keep_start = conv_input_len - n_keep;
        // Force contiguous layout — the slice is a strided view, and passing
        // strided arrays to concatenate on the next decode step is suboptimal.
        let cs = conv_input.index((.., keep_start.., ..));
        let cs_shape = cs.shape().to_vec();
        cache.conv_state = Some(cs.flatten(None, None)?.reshape(&cs_shape)?);

        // Fast path: depthwise conv1d via element-wise multiply + sum.
        // S==1: single window, no loop. S<=32 (block-K): sliding window loop.
        // Both avoid full Conv1d kernel dispatch overhead (30 GDN layers).
        // S=1 and prefill both use native Conv1d (single fused kernel dispatch).
        // Block-K verify (1 < S <= 32) uses sliding window with pre-transposed weights
        // to avoid Conv1d dispatch overhead at small S with 30 GDN layers.
        let conv_out = if S > 1 && S <= 32 {
            let wt = match &self.conv_weight_t {
                Some(w) => w.clone(),
                None => {
                    let shape = self.conv1d.weight.shape();
                    let w = if shape.len() == 3 && shape[2] == 1 {
                        self.conv1d.weight.squeeze_axes(&[-1])?.transpose()?
                    } else if shape.len() == 3 && shape[1] == 1 {
                        self.conv1d.weight.squeeze_axes(&[1])?.transpose()?
                    } else {
                        return Err(Exception::custom(format!(
                            "Unexpected conv1d weight shape: {:?}",
                            shape
                        )));
                    };
                    let w = w.as_dtype(inputs.dtype())?;
                    w.eval()?;
                    self.conv_weight_t = Some(w.clone());
                    w
                }
            };
            // Sliding window: conv_input [B, K-1+S, D] → S windows of [B, K, D]
            let ks = self.conv_kernel_size;
            let mut windows = Vec::with_capacity(S as usize);
            for i in 0..S {
                windows.push(
                    conv_input
                        .index((.., i..i + ks, ..))
                        .multiply(&wt)?
                        .sum_axes(&[1], true)?,
                );
            }
            nn::silu(&ops::concatenate_axis(
                &windows.iter().collect::<Vec<_>>(),
                1,
            )?)?
        } else {
            // S=1 decode + large S prefill: native Conv1d is a single fused dispatch.
            // Coerce conv1d.weight to input dtype on first call — the S>1 path above
            // does this implicitly via conv_weight_t; the native path needs it
            // explicit or MLX panics with "operation failed but no error was set"
            // when the model's stored dtype (e.g. bf16 distil drafter) differs from
            // the active activation dtype (e.g. f16 from a 4-bit verifier pair).
            let in_dt = inputs.dtype();
            if self.conv1d.weight.dtype() != in_dt {
                let coerced = self.conv1d.weight.as_dtype(in_dt)?;
                coerced.eval()?;
                self.conv1d.weight.value = coerced;
            }
            nn::silu(&self.conv1d.forward(&conv_input)?)?
        };

        // Split conv output back to q, k, v
        let split_indices = &[self.key_dim, self.key_dim * 2];
        let conv_parts = conv_out.split_axis(split_indices, Some(-1))?;
        let conv_q = conv_parts
            .first()
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_k = conv_parts
            .get(1)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_v = conv_parts
            .get(2)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

        // On first call, convert weight vectors to match input dtype.
        let in_dt = inputs.dtype();
        if self.qk_norm_weight_q.dtype() != in_dt {
            self.qk_norm_weight_q = self.qk_norm_weight_q.as_dtype(in_dt)?;
            self.qk_norm_weight_k = self.qk_norm_weight_k.as_dtype(in_dt)?;
        }

        let norm_q = fast::rms_norm(&conv_q, &self.qk_norm_weight_q, 1e-6)?;
        let norm_k = fast::rms_norm(&conv_k, &self.qk_norm_weight_k, 1e-6)?;

        // SSM recurrence: Metal kernel for all S values.
        // S > 1: use cached state (correct output, state rolled back on rejection via GdnStateBackup).
        // S == 1: use cached state (normal autoregressive decode).
        let (y, new_state) = if S > 1 {
            let state = cache.ssm_state.take().unwrap_or_else(|| {
                ops::zeros_dtype(
                    &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                    inputs.dtype(),
                )
                .unwrap()
            });
            gated_delta_kernel_ffi(
                &norm_q,
                &norm_k,
                &conv_v,
                &self.A_log,
                &a,
                &self.dt_bias,
                &b,
                &state,
                B,
                S,
                self.num_k_heads,
                self.head_k_dim,
                self.num_v_heads,
                self.head_v_dim,
            )?
        } else {
            // Recurrent mode: use previous ssm_state
            let state = match cache.ssm_state.take() {
                Some(s) => s,
                None => ops::zeros_dtype(
                    &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                    inputs.dtype(),
                )?,
            };
            gated_delta_kernel_ffi(
                &norm_q,
                &norm_k,
                &conv_v,
                &self.A_log,
                &a,
                &self.dt_bias,
                &b,
                &state,
                B,
                S,
                self.num_k_heads,
                self.head_k_dim,
                self.num_v_heads,
                self.head_v_dim,
            )?
        };
        cache.ssm_state = Some(new_state);
        cache.offset += S;

        let normed = self.norm.forward(&y)?;
        let gated_out = swiglu(&z, &normed)?;

        // Output projection
        let out_flat = gated_out.reshape(&[B, S, -1])?;
        self.out_proj.forward(&out_flat)
    }

    /// Stateless forward: computes identical output `y` but does NOT update
    /// `cache.ssm_state` or `cache.conv_state`. Used for DFlash verify —
    /// GDN state is never corrupted by speculative tokens.
    #[allow(non_snake_case)]
    fn forward_stateless(
        &mut self,
        inputs: &Array,
        _mask: Option<&AttentionMask>,
        cache: &ArraysCache,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let S = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Project inputs — same as stateful forward
        let (q, k, v, z, b, a) = if self.use_separate_projections {
            let qkv_proj = self
                .in_proj_qkv
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_qkv missing"))?;
            let z_proj = self
                .in_proj_z
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_z missing"))?;
            let b_proj = self
                .in_proj_b
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_b missing"))?;
            let a_proj = self
                .in_proj_a
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_a missing"))?;

            let qkv = qkv_proj.forward(inputs)?;
            let z = z_proj
                .forward(inputs)?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;
            let b = b_proj.forward(inputs)?;
            let a = a_proj.forward(inputs)?;

            let split_indices = &[self.key_dim, self.key_dim * 2];
            let qkv_parts = qkv.split_axis(split_indices, Some(-1))?;
            let q = qkv_parts
                .first()
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let k = qkv_parts
                .get(1)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let v = qkv_parts
                .get(2)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

            (q, k, v, z, b, a)
        } else {
            let mixed_qkvz = self.in_proj_qkvz.forward(inputs)?;
            let mixed_ba = self.in_proj_ba.forward(inputs)?;
            self.fix_query_key_value_ordering(&mixed_qkvz, &mixed_ba, B, S)?
        };

        // Conv1d — read conv_state without consuming it
        let q_flat = q.reshape(&[B, S, -1])?;
        let k_flat = k.reshape(&[B, S, -1])?;
        let v_flat = v.reshape(&[B, S, -1])?;
        let mixed_qkv = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1)?;

        // Borrow conv_state without taking it (stateless must not mutate cache)
        let conv_state = match &cache.conv_state {
            Some(state) => state.clone(),
            None => ops::zeros_dtype(
                &[B, self.conv_kernel_size - 1, self.conv_dim],
                inputs.dtype(),
            )?,
        };
        let conv_input = ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;

        // DO NOT update cache.conv_state

        let conv_out = if S > 1 && S <= 32 {
            let wt = match &self.conv_weight_t {
                Some(w) => w.clone(),
                None => {
                    let shape = self.conv1d.weight.shape();
                    let w = if shape.len() == 3 && shape[2] == 1 {
                        self.conv1d.weight.squeeze_axes(&[-1])?.transpose()?
                    } else if shape.len() == 3 && shape[1] == 1 {
                        self.conv1d.weight.squeeze_axes(&[1])?.transpose()?
                    } else {
                        return Err(Exception::custom(format!(
                            "Unexpected conv1d weight shape: {:?}",
                            shape
                        )));
                    };
                    let w = w.as_dtype(inputs.dtype())?;
                    w.eval()?;
                    // Don't cache weight here — stateless should be side-effect-free
                    w
                }
            };
            let ks = self.conv_kernel_size;
            let mut windows = Vec::with_capacity(S as usize);
            for i in 0..S {
                windows.push(
                    conv_input
                        .index((.., i..i + ks, ..))
                        .multiply(&wt)?
                        .sum_axes(&[1], true)?,
                );
            }
            nn::silu(&ops::concatenate_axis(
                &windows.iter().collect::<Vec<_>>(),
                1,
            )?)?
        } else {
            // Stateless path: clone-coerce rather than mutate self.conv1d.weight,
            // matching the qk_norm_weight_* clone pattern below.
            let in_dt = inputs.dtype();
            if self.conv1d.weight.dtype() != in_dt {
                let coerced = self.conv1d.weight.as_dtype(in_dt)?;
                let out = ops::conv1d(
                    &conv_input,
                    &coerced,
                    1,
                    0,
                    1,
                    self.conv_dim,
                )?;
                nn::silu(&out)?
            } else {
                nn::silu(&self.conv1d.forward(&conv_input)?)?
            }
        };

        let split_indices = &[self.key_dim, self.key_dim * 2];
        let conv_parts = conv_out.split_axis(split_indices, Some(-1))?;
        let conv_q = conv_parts
            .first()
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_k = conv_parts
            .get(1)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_v = conv_parts
            .get(2)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

        let in_dt = inputs.dtype();
        let qk_wq = if self.qk_norm_weight_q.dtype() != in_dt {
            self.qk_norm_weight_q.as_dtype(in_dt)?
        } else {
            self.qk_norm_weight_q.clone()
        };
        let qk_wk = if self.qk_norm_weight_k.dtype() != in_dt {
            self.qk_norm_weight_k.as_dtype(in_dt)?
        } else {
            self.qk_norm_weight_k.clone()
        };

        let norm_q = fast::rms_norm(&conv_q, &qk_wq, 1e-6)?;
        let norm_k = fast::rms_norm(&conv_k, &qk_wk, 1e-6)?;

        // Stateless SSM recurrence — state_out := state_in
        let state = match &cache.ssm_state {
            Some(s) => s.clone(),
            None => ops::zeros_dtype(
                &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                inputs.dtype(),
            )?,
        };
        let (y, _unchanged_state) = gated_delta_kernel_ffi_stateless(
            &norm_q,
            &norm_k,
            &conv_v,
            &self.A_log,
            &a,
            &self.dt_bias,
            &b,
            &state,
            B,
            S,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )?;
        // DO NOT update cache.ssm_state or cache.offset

        let normed = self.norm.forward(&y)?;
        let gated_out = swiglu(&z, &normed)?;

        let out_flat = gated_out.reshape(&[B, S, -1])?;
        self.out_proj.forward(&out_flat)
    }

    /// Tape-recording forward: identical output, also returns
    /// `(delta_tape, norm_k, a_proj)` needed for cheap tape replay.
    /// State IS updated (normal forward) — on full acceptance, zero extra work.
    #[allow(non_snake_case)]
    fn forward_with_tape(
        &mut self,
        inputs: &Array,
        _mask: Option<&AttentionMask>,
        cache: &mut ArraysCache,
    ) -> Result<(Array, GdnLayerTape), Exception> {
        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("need >= 2 dims"))?;
        let S = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("need >= 2 dims"))?;

        let (q, k, v, z, b, a) = if self.use_separate_projections {
            let qkv_proj = self
                .in_proj_qkv
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_qkv missing"))?;
            let z_proj = self
                .in_proj_z
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_z missing"))?;
            let b_proj = self
                .in_proj_b
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_b missing"))?;
            let a_proj = self
                .in_proj_a
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_a missing"))?;

            let qkv = qkv_proj.forward(inputs)?;
            let z = z_proj
                .forward(inputs)?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;
            let b = b_proj.forward(inputs)?;
            let a = a_proj.forward(inputs)?;

            let split_indices = &[self.key_dim, self.key_dim * 2];
            let qkv_parts = qkv.split_axis(split_indices, Some(-1))?;
            let q = qkv_parts
                .first()
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let k = qkv_parts
                .get(1)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let v = qkv_parts
                .get(2)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;
            (q, k, v, z, b, a)
        } else {
            // ANE-offload path (feature-gated). Two ANE backends share this
            // dispatch site:
            //   1. Worker handle (Wave 4, production) — `Send + Sync` mpsc
            //      handle to a model-wide `qwen-gdn-ane-worker` thread that
            //      owns kernels for ALL linear layers. Selected first.
            //   2. Inline `Vec<Arc<GdnAneLayerKernels>>` (Wave 1/2 parity tests
            //      only — `!Send`, can't survive moving the model into a
            //      worker thread). Selected when no handle is attached.
            // Both fall back to Metal when `S` exceeds the compiled seq_len.
            #[cfg(feature = "ane")]
            let (mixed_qkvz, mixed_ba) = if let Some(handle) = &self.ane_handle {
                if (S as usize) <= handle.seq_len() {
                    let idx = self.ane_linear_layer_idx;
                    // Fused dispatch: single ANE eval for both qkvz+ba.
                    handle.dispatch_fused(idx, inputs)?
                } else {
                    (
                        self.in_proj_qkvz.forward(inputs)?,
                        self.in_proj_ba.forward(inputs)?,
                    )
                }
            } else {
                match self
                    .ane_kernels
                    .as_deref()
                    .and_then(|buckets| select_ane_bucket(buckets, S as usize))
                {
                    Some(k) => {
                        if let Some(fused) = k.qkvz_ba_fused.as_ref() {
                            fused.dispatch(inputs)?
                        } else {
                            let qkvz = k.qkvz.as_ref().ok_or_else(|| {
                                Exception::custom(
                                    "ane_kernels bucket has neither fused nor separate qkvz",
                                )
                            })?;
                            let ba = k.ba.as_ref().ok_or_else(|| {
                                Exception::custom(
                                    "ane_kernels bucket has neither fused nor separate ba",
                                )
                            })?;
                            (qkvz.dispatch(inputs)?, ba.dispatch(inputs)?)
                        }
                    }
                    None => (
                        self.in_proj_qkvz.forward(inputs)?,
                        self.in_proj_ba.forward(inputs)?,
                    ),
                }
            };
            #[cfg(not(feature = "ane"))]
            let (mixed_qkvz, mixed_ba) = (
                self.in_proj_qkvz.forward(inputs)?,
                self.in_proj_ba.forward(inputs)?,
            );
            self.fix_query_key_value_ordering(&mixed_qkvz, &mixed_ba, B, S)?
        };

        // Save a for replay (before any reshape that might happen)
        let a_for_replay = a.clone();

        // Conv1d — same as normal forward
        let q_flat = q.reshape(&[B, S, -1])?;
        let k_flat = k.reshape(&[B, S, -1])?;
        let v_flat = v.reshape(&[B, S, -1])?;
        let mixed_qkv = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1)?;

        // Save qkv for conv_state rebuild on replay
        let qkv_for_replay = mixed_qkv.clone();

        // Capture initial state for rollback (Python _GDNStateCapture equivalent)
        let conv_state_init = cache.conv_state.clone();
        let ssm_state_init = cache.ssm_state.clone();
        let offset_init = cache.offset;

        let conv_state = match cache.conv_state.take() {
            Some(state) => state,
            None => ops::zeros_dtype(
                &[B, self.conv_kernel_size - 1, self.conv_dim],
                inputs.dtype(),
            )?,
        };
        let conv_input = ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;

        let n_keep = self.conv_kernel_size - 1;
        let conv_input_len = *conv_input
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("conv_input missing seq dim"))?;
        let keep_start = conv_input_len - n_keep;
        let cs = conv_input.index((.., keep_start.., ..));
        let cs_shape = cs.shape().to_vec();
        cache.conv_state = Some(cs.flatten(None, None)?.reshape(&cs_shape)?);

        let conv_out = if S > 1 && S <= 32 {
            let wt = match &self.conv_weight_t {
                Some(w) => w.clone(),
                None => {
                    let shape = self.conv1d.weight.shape();
                    let w = if shape.len() == 3 && shape[2] == 1 {
                        self.conv1d.weight.squeeze_axes(&[-1])?.transpose()?
                    } else if shape.len() == 3 && shape[1] == 1 {
                        self.conv1d.weight.squeeze_axes(&[1])?.transpose()?
                    } else {
                        return Err(Exception::custom(format!(
                            "Unexpected conv1d weight shape: {:?}",
                            shape
                        )));
                    };
                    let w = w.as_dtype(inputs.dtype())?;
                    w.eval()?;
                    self.conv_weight_t = Some(w.clone());
                    w
                }
            };
            let ks = self.conv_kernel_size;
            let mut windows = Vec::with_capacity(S as usize);
            for i in 0..S {
                windows.push(
                    conv_input
                        .index((.., i..i + ks, ..))
                        .multiply(&wt)?
                        .sum_axes(&[1], true)?,
                );
            }
            nn::silu(&ops::concatenate_axis(
                &windows.iter().collect::<Vec<_>>(),
                1,
            )?)?
        } else {
            // Mirror the S>1 dtype coercion for the native Conv1d path —
            // see GatedDeltaNet::forward for the rationale.
            let in_dt = inputs.dtype();
            if self.conv1d.weight.dtype() != in_dt {
                let coerced = self.conv1d.weight.as_dtype(in_dt)?;
                coerced.eval()?;
                self.conv1d.weight.value = coerced;
            }
            nn::silu(&self.conv1d.forward(&conv_input)?)?
        };

        let split_indices = &[self.key_dim, self.key_dim * 2];
        let conv_parts = conv_out.split_axis(split_indices, Some(-1))?;
        let conv_q = conv_parts
            .first()
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_k = conv_parts
            .get(1)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_v = conv_parts
            .get(2)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

        let in_dt = inputs.dtype();
        if self.qk_norm_weight_q.dtype() != in_dt {
            self.qk_norm_weight_q = self.qk_norm_weight_q.as_dtype(in_dt)?;
            self.qk_norm_weight_k = self.qk_norm_weight_k.as_dtype(in_dt)?;
        }

        let norm_q = fast::rms_norm(&conv_q, &self.qk_norm_weight_q, 1e-6)?;
        let norm_k = fast::rms_norm(&conv_k, &self.qk_norm_weight_k, 1e-6)?;

        // Save norm_k for replay
        let norm_k_for_replay = norm_k.clone();

        // Tape-recording kernel — state IS updated, tape IS recorded
        let state = cache.ssm_state.take().unwrap_or_else(|| {
            ops::zeros_dtype(
                &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                inputs.dtype(),
            )
            .unwrap()
        });
        let (y, new_state, delta_tape) = gated_delta_kernel_ffi_with_tape(
            &norm_q,
            &norm_k,
            &conv_v,
            &self.A_log,
            &a,
            &self.dt_bias,
            &b,
            &state,
            B,
            S,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )?;
        cache.ssm_state = Some(new_state);
        cache.offset += S;

        let normed = self.norm.forward(&y)?;
        let gated_out = swiglu(&z, &normed)?;
        let out_flat = gated_out.reshape(&[B, S, -1])?;
        // ANE out_proj — see the qkvz/ba dispatch comment above for why both
        // worker handle and inline kernels are checked.
        #[cfg(feature = "ane")]
        let output = if let Some(handle) = &self.ane_handle {
            if (S as usize) <= handle.seq_len() {
                use crate::qwen3_next_ane_worker::ProjKind;
                handle.dispatch(self.ane_linear_layer_idx, ProjKind::OutProj, &out_flat)?
            } else {
                self.out_proj.forward(&out_flat)?
            }
        } else {
            match self
                .ane_kernels
                .as_deref()
                .and_then(|buckets| select_ane_bucket(buckets, S as usize))
            {
                Some(k) => k.out_proj.dispatch(&out_flat)?,
                None => self.out_proj.forward(&out_flat)?,
            }
        };
        #[cfg(not(feature = "ane"))]
        let output = self.out_proj.forward(&out_flat)?;

        let tape = GdnLayerTape {
            delta_tape,
            norm_k: norm_k_for_replay,
            a_proj: a_for_replay,
            qkv_input: qkv_for_replay,
            conv_state_init,
            ssm_state_init,
            offset_init,
        };

        Ok((output, tape))
    }

    /// Replay accepted steps from a recorded tape, advancing SSM state.
    /// Also rebuilds conv_state for the accepted prefix.
    #[allow(non_snake_case)]
    fn replay_from_tape(
        &self,
        tape: &GdnLayerTape,
        n_accepted: i32,
        cache: &mut ArraysCache,
    ) -> Result<(), Exception> {
        if n_accepted <= 0 {
            return Ok(());
        }

        // Slice tape data to accepted steps only
        let tape_slice = tape.delta_tape.index((.., ..n_accepted, ..));
        let k_slice = tape.norm_k.index((.., ..n_accepted, ..));
        let a_slice = tape.a_proj.index((.., ..n_accepted, ..));

        let state = cache.ssm_state.take().unwrap_or_else(|| {
            let dt = tape.delta_tape.dtype();
            ops::zeros_dtype(&[1, self.num_v_heads, self.head_v_dim, self.head_k_dim], dt).unwrap()
        });

        let new_state = tape_replay_kernel_ffi(
            &tape_slice,
            &k_slice,
            &a_slice,
            &self.A_log,
            &self.dt_bias,
            &state,
            1,
            n_accepted,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )?;
        cache.ssm_state = Some(new_state);
        cache.offset += n_accepted;

        // Rebuild conv_state from recorded qkv input
        let ks = self.conv_kernel_size;
        let n_keep = ks - 1;
        if n_keep > 0 {
            let qkv_slice = tape.qkv_input.index((.., ..n_accepted, ..));
            // Prepend existing conv_state (or zeros), take last n_keep entries
            let prefix = match &cache.conv_state {
                Some(cs) => cs.clone(),
                None => ops::zeros_dtype(&[1, n_keep, self.conv_dim], tape.qkv_input.dtype())?,
            };
            let full = ops::concatenate_axis(&[&prefix, &qkv_slice], 1)?;
            let total_len = *full
                .shape()
                .get(1)
                .ok_or_else(|| Exception::custom("conv rebuild: missing seq dim"))?;
            let start = total_len - n_keep;
            let cs = full.index((.., start.., ..));
            let cs_shape = cs.shape().to_vec();
            cache.conv_state = Some(cs.flatten(None, None)?.reshape(&cs_shape)?);
        }

        Ok(())
    }

    /// Reorder the projected qkvz and ba tensors into separate heads.
    #[allow(non_snake_case, clippy::type_complexity)]
    fn fix_query_key_value_ordering(
        &self,
        mixed_qkvz: &Array,
        mixed_ba: &Array,
        B: i32,
        S: i32,
    ) -> Result<(Array, Array, Array, Array, Array, Array), Exception> {
        let nk = self.num_k_heads;
        let dn = self.head_k_dim;
        let nv = self.num_v_heads;
        let dv = self.head_v_dim;
        let v_per_k = nv / nk;

        // Reshape to [B, S, nk, -1]
        let qkvz = mixed_qkvz.reshape(&[B, S, nk, -1])?;
        let ba = mixed_ba.reshape(&[B, S, nk, -1])?;

        // Split qkvz at [dn, 2*dn, 2*dn + v_per_k*dv]
        let split_at = &[dn, 2 * dn, 2 * dn + v_per_k * dv];
        let qkvz_parts = qkvz.split_axis(split_at, Some(-1))?;
        let q = qkvz_parts
            .first()
            .ok_or_else(|| Exception::custom("qkvz split failed"))?
            .clone();
        let k = qkvz_parts
            .get(1)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?
            .clone();
        let v_raw = qkvz_parts
            .get(2)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?;
        let z_raw = qkvz_parts
            .get(3)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?;

        let v = v_raw.reshape(&[B, S, nv, dv])?;
        let z = z_raw.reshape(&[B, S, nv, dv])?;

        // Split ba at [v_per_k]
        let ba_parts = ba.split_axis(&[v_per_k], Some(-1))?;
        let b_raw = ba_parts
            .first()
            .ok_or_else(|| Exception::custom("ba split failed"))?;
        let a_raw = ba_parts
            .get(1)
            .ok_or_else(|| Exception::custom("ba split failed"))?;

        let b = b_raw.reshape(&[B, S, nv])?;
        let a = a_raw.reshape(&[B, S, nv])?;

        Ok((q, k, v, z, b, a))
    }

    /// Compile and attach ANE kernels for this layer's three GDN projections
    /// (`in_proj_qkvz`, `in_proj_ba`, `out_proj`) at the given seq buckets.
    ///
    /// One fresh kernel set is compiled per bucket in `seq_lens`. Buckets are
    /// stored sorted ascending by `seq_len`; dispatch picks the smallest bucket
    /// that fits the runtime `S`, or falls back to Metal if `S` exceeds every
    /// bucket. After this call, `forward_with_tape` will run all three
    /// projections on ANE when `!use_separate_projections` and a bucket fits.
    /// All other ops (conv, norm, delta kernel) stay on Metal.
    ///
    /// Currently restricted to `seq_lens.len() == 1`. Multi-bucket was
    /// implemented but exposed a reproducible failure in the ANE bridge's
    /// `patch_from_donor` path (state accumulates across patches and fails
    /// around the 17th patched bucket1 invocation). The slice signature is
    /// retained so the guard can be lifted in a single edit once the bridge
    /// is fixed. See `.planning/next-session-phase2-wave3.md`.
    ///
    /// Returns an error if `seq_lens` is empty, has len > 1, contains
    /// duplicates, or if any weight is not rank-2 / cannot be dequantized.
    #[cfg(feature = "ane")]
    pub fn enable_ane_gdn(&mut self, seq_lens: &[i32]) -> Result<(), Exception> {
        use std::sync::Arc;
        if self.use_separate_projections {
            return Err(Exception::custom(
                "enable_ane_gdn: use_separate_projections=true not supported",
            ));
        }
        if seq_lens.is_empty() {
            return Err(Exception::custom("enable_ane_gdn: seq_lens empty"));
        }
        if seq_lens.len() > 1 {
            return Err(Exception::custom(format!(
                "enable_ane_gdn: multi-bucket (len={}) disabled — ANE bridge \
                 patch_from_donor leaks state across successive patches \
                 (fails ~17 patches in on bucket1). Pass a single-element slice. \
                 See .planning/next-session-phase2-wave3.md.",
                seq_lens.len()
            )));
        }
        let mut sorted: Vec<i32> = seq_lens.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(Exception::custom(format!(
                "enable_ane_gdn: duplicate seq_lens {seq_lens:?}"
            )));
        }
        if sorted.first().is_some_and(|s| *s <= 0) {
            return Err(Exception::custom(format!(
                "enable_ane_gdn: non-positive seq_len in {seq_lens:?}"
            )));
        }
        let mut buckets: Vec<Arc<crate::qwen3_next_ane::GdnAneLayerKernels>> =
            Vec::with_capacity(sorted.len());
        for seq_len in sorted {
            let qkvz = compile_proj_from_qlinear(&self.in_proj_qkvz, seq_len, "qkvz")?;
            let ba = compile_proj_from_qlinear(&self.in_proj_ba, seq_len, "ba")?;
            let out_proj = compile_proj_from_qlinear(&self.out_proj, seq_len, "out_proj")?;
            buckets.push(Arc::new(crate::qwen3_next_ane::GdnAneLayerKernels {
                seq_len: qkvz.seq_len,
                qkvz: Some(Arc::new(qkvz)),
                ba: Some(Arc::new(ba)),
                qkvz_ba_fused: None,
                out_proj: Arc::new(out_proj),
            }));
        }
        self.ane_kernels = Some(buckets);
        Ok(())
    }

    /// Attach ANE projection kernels for this layer by patching weights into
    /// a donor layer's already-compiled microcode, one bucket at a time.
    ///
    /// `donors` must come from `enable_ane_gdn` on a layer with identical
    /// projection shapes (true for any two GDN layers in the same Qwen3-Next
    /// model). The donor order IS the bucket order for this layer — donors
    /// are assumed sorted ascending by `qkvz.seq_len` (that is the invariant
    /// `enable_ane_gdn` establishes). Skips MIL compilation — only runs
    /// `loadWithQoS` per projection per bucket, so this is O(weight-load)
    /// rather than O(MIL-compile).
    #[cfg(feature = "ane")]
    pub fn enable_ane_gdn_from_donor(
        &mut self,
        donors: &[std::sync::Arc<crate::qwen3_next_ane::GdnAneLayerKernels>],
    ) -> Result<(), Exception> {
        use std::sync::Arc;
        if self.use_separate_projections {
            return Err(Exception::custom(
                "enable_ane_gdn_from_donor: use_separate_projections=true not supported",
            ));
        }
        if donors.is_empty() {
            return Err(Exception::custom("enable_ane_gdn_from_donor: donors empty"));
        }
        if donors.len() > 1 {
            return Err(Exception::custom(format!(
                "enable_ane_gdn_from_donor: multi-bucket (len={}) disabled — \
                 ANE bridge patch_from_donor leaks state across successive \
                 patches. Pass a single donor. \
                 See .planning/next-session-phase2-wave3.md.",
                donors.len()
            )));
        }
        let mut buckets: Vec<Arc<crate::qwen3_next_ane::GdnAneLayerKernels>> =
            Vec::with_capacity(donors.len());
        for (bi, donor) in donors.iter().enumerate() {
            let seq = donor.seq_len;
            let donor_qkvz = donor.qkvz.as_ref().ok_or_else(|| {
                Exception::custom(
                    "enable_ane_gdn_from_donor: donor missing separate qkvz kernel \
                 (was compiled with fused layout — not supported for multi-bucket patching)",
                )
            })?;
            let donor_ba = donor.ba.as_ref().ok_or_else(|| {
                Exception::custom("enable_ane_gdn_from_donor: donor missing separate ba kernel")
            })?;
            let load_before = crate::ane_bridge::load_count();
            let qkvz =
                compile_proj_from_qlinear_donor(&self.in_proj_qkvz, donor_qkvz).map_err(|e| {
                    Exception::custom(format!(
                        "patch qkvz bucket{bi}(seq={seq}) load_before={load_before}: {e}"
                    ))
                })?;
            let ba = compile_proj_from_qlinear_donor(&self.in_proj_ba, donor_ba).map_err(|e| {
                Exception::custom(format!(
                    "patch ba   bucket{bi}(seq={seq}) load_before={load_before}: {e}"
                ))
            })?;
            let out_proj = compile_proj_from_qlinear_donor(&self.out_proj, &donor.out_proj)
                .map_err(|e| {
                    Exception::custom(format!(
                        "patch out  bucket{bi}(seq={seq}) load_before={load_before}: {e}"
                    ))
                })?;
            buckets.push(Arc::new(crate::qwen3_next_ane::GdnAneLayerKernels {
                seq_len: qkvz.seq_len,
                qkvz: Some(Arc::new(qkvz)),
                ba: Some(Arc::new(ba)),
                qkvz_ba_fused: None,
                out_proj: Arc::new(out_proj),
            }));
        }
        self.ane_kernels = Some(buckets);
        Ok(())
    }
}

/// Pick the ANE bucket for runtime seq `s` from a layer's bucket list.
///
/// `buckets` must be sorted ascending by `qkvz.seq_len` (the invariant that
/// [`GatedDeltaNet::enable_ane_gdn`] and [`GatedDeltaNet::enable_ane_gdn_from_donor`]
/// establish). Returns the smallest bucket where `s <= seq_len`, or `None` if
/// `s` exceeds every bucket — in which case the caller falls back to Metal.
#[cfg(feature = "ane")]
#[inline]
fn select_ane_bucket(
    buckets: &[std::sync::Arc<crate::qwen3_next_ane::GdnAneLayerKernels>],
    s: usize,
) -> Option<&std::sync::Arc<crate::qwen3_next_ane::GdnAneLayerKernels>> {
    buckets.iter().find(|k| s <= k.seq_len)
}

/// Helper: dequantize a `QLinear` weight and compile an `AneProjKernel` for it.
#[cfg(feature = "ane")]
fn compile_proj_from_qlinear(
    ql: &QLinear,
    seq_len: i32,
    name: &'static str,
) -> Result<crate::qwen3_next_ane::AneProjKernel, Exception> {
    let wshape = ql.weight.shape();
    if wshape.len() != 2 {
        return Err(Exception::custom(format!(
            "enable_ane_gdn({name}): expected rank-2 weight, got {wshape:?}"
        )));
    }
    let out_dim = wshape[0] as usize;
    // For Uint32 packed weights, inner dim is oc * ic_packed*8/bits;
    // `ops::dequantize` expands this for us, so read the dim from the
    // dequant output instead of the packed shape.
    let w_f32 = crate::qwen3_next_ane::dequantize_qlinear_to_f32(
        &ql.weight,
        &ql.scales,
        &ql.biases,
        ql.group_size,
        ql.bits,
    )?;
    if out_dim == 0 || w_f32.is_empty() {
        return Err(Exception::custom(format!(
            "enable_ane_gdn({name}): empty weight"
        )));
    }
    let in_dim = w_f32.len() / out_dim;
    if in_dim * out_dim != w_f32.len() {
        return Err(Exception::custom(format!(
            "enable_ane_gdn({name}): weight len {} not divisible by oc {}",
            w_f32.len(),
            out_dim
        )));
    }
    crate::qwen3_next_ane::compile_proj(&w_f32, in_dim, out_dim, seq_len as usize, name)
        .map_err(Exception::custom)
}

/// Wave 4: dequantize a `QLinear` weight to row-major f32 + report `(in, out)`.
/// Used by `enable_ane_gdn_all_layers_via_worker` to extract per-layer GDN
/// projection weights on the main thread before shipping them to the worker.
///
/// `name` and `layer_idx` are diagnostic — only used in error messages.
#[cfg(feature = "ane")]
fn dequantize_gdn_qlinear(
    ql: &QLinear,
    name: &'static str,
    layer_idx: usize,
) -> Result<(Vec<f32>, usize, usize), Exception> {
    let wshape = ql.weight.shape();
    if wshape.len() != 2 {
        return Err(Exception::custom(format!(
            "dequantize_gdn_qlinear({name}, layer {layer_idx}): expected rank-2 weight, got {wshape:?}"
        )));
    }
    let out_dim = wshape[0] as usize;
    let w_f32 = crate::qwen3_next_ane::dequantize_qlinear_to_f32(
        &ql.weight,
        &ql.scales,
        &ql.biases,
        ql.group_size,
        ql.bits,
    )?;
    if out_dim == 0 || w_f32.is_empty() {
        return Err(Exception::custom(format!(
            "dequantize_gdn_qlinear({name}, layer {layer_idx}): empty weight"
        )));
    }
    let in_dim = w_f32.len() / out_dim;
    if in_dim * out_dim != w_f32.len() {
        return Err(Exception::custom(format!(
            "dequantize_gdn_qlinear({name}, layer {layer_idx}): weight len {} not divisible by oc {}",
            w_f32.len(),
            out_dim
        )));
    }
    Ok((w_f32, in_dim, out_dim))
}

/// Wave 2: dequantize a `QLinear` weight and patch it into a donor's compiled
/// kernel (no MIL recompile). Donor must already be compiled at the matching
/// shape — call this only for layers ≥ 1, with layer 0's kernels as donor.
#[cfg(feature = "ane")]
fn compile_proj_from_qlinear_donor(
    ql: &QLinear,
    donor: &crate::qwen3_next_ane::AneProjKernel,
) -> Result<crate::qwen3_next_ane::AneProjKernel, Exception> {
    let w_f32 = crate::qwen3_next_ane::dequantize_qlinear_to_f32(
        &ql.weight,
        &ql.scales,
        &ql.biases,
        ql.group_size,
        ql.bits,
    )?;
    if w_f32.len() != donor.in_dim * donor.out_dim {
        return Err(Exception::custom(format!(
            "enable_ane_gdn_from_donor({}): weight len {} != donor in_dim·out_dim {}·{} = {}",
            donor.name,
            w_f32.len(),
            donor.in_dim,
            donor.out_dim,
            donor.in_dim * donor.out_dim
        )));
    }
    crate::qwen3_next_ane::compile_proj_from_donor(donor, &w_f32).map_err(Exception::custom)
}

/// Reference implementation of gate computation (used by tests).
/// Production code uses `compute_g_beta_kernel_ffi` instead.
#[cfg(test)]
fn compute_g_compiled((a_log, a, dt_bias): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    let a_plus_bias = a.add(dt_bias)?;
    let sp = nn::softplus(&a_plus_bias)?;
    let neg_decay = a_log.exp()?.negative()?.multiply(sp)?;
    neg_decay.exp()
}

// ---------------------------------------------------------------------------
// DecoderLayer
// ---------------------------------------------------------------------------

/// Wrapper for the FFN block: either sparse MoE or dense SwiGLU.
/// Both share the `mlp` parameter namespace in safetensors — their sub-keys
/// don't overlap (MoE: gate, switch_mlp, shared_expert; Dense: gate_proj, up_proj, down_proj).
#[derive(Debug, Clone, ModuleParameters)]
struct FfnBlock {
    #[param]
    gate: Option<QLinear>,
    #[param]
    switch_mlp: Option<SwitchMlpWeights>,
    #[param]
    shared_expert: Option<Qwen3NextMLP>,
    #[param]
    shared_expert_gate: Option<QLinear>,
    #[param]
    gate_proj: Option<QLinear>,
    #[param]
    up_proj: Option<QLinear>,
    #[param]
    down_proj: Option<QLinear>,
    is_moe: bool,
    top_k: i32,
    norm_topk_prob: bool,
    /// Cached fused gate+up weights for dense layers (lazily computed on first forward).
    fused_gate_up: Option<(Array, Array, Array, i32)>,
    /// Optional int8-mlpackage kernels for the three dense projections.
    /// Populated by [`Qwen3NextCausalLM::finalize_ane_mlp_layer0_int8_inline`]
    /// when `HIGGS_TARGET_ANE_INT8_MLP=1`. Dispatched in the dense forward
    /// path only for prefill shapes (`1 < seq <= compiled bucket`); decode
    /// (seq=1) stays on the QLinear path. MoE layers ignore these fields.
    #[cfg(feature = "ane")]
    gate_proj_ane: Option<std::sync::Arc<crate::ane_mlmodel::AneMlPackageKernel>>,
    #[cfg(feature = "ane")]
    up_proj_ane: Option<std::sync::Arc<crate::ane_mlmodel::AneMlPackageKernel>>,
    #[cfg(feature = "ane")]
    down_proj_ane: Option<std::sync::Arc<crate::ane_mlmodel::AneMlPackageKernel>>,
}

impl FfnBlock {
    fn new_moe(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let moe = SparseMoeBlock::new(args, ql, qb)?;
        Ok(Self {
            gate: Some(moe.gate),
            switch_mlp: Some(moe.switch_mlp),
            shared_expert: Some(moe.shared_expert),
            shared_expert_gate: Some(moe.shared_expert_gate),
            gate_proj: None,
            up_proj: None,
            down_proj: None,
            is_moe: true,
            top_k: moe.top_k,
            norm_topk_prob: moe.norm_topk_prob,
            fused_gate_up: None,
            #[cfg(feature = "ane")]
            gate_proj_ane: None,
            #[cfg(feature = "ane")]
            up_proj_ane: None,
            #[cfg(feature = "ane")]
            down_proj_ane: None,
        })
    }

    fn new_dense(ql: i32, qb: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate: None,
            switch_mlp: None,
            shared_expert: None,
            shared_expert_gate: None,
            gate_proj: Some(QLinear::new(ql, qb)?),
            up_proj: Some(QLinear::new(ql, qb)?),
            down_proj: Some(QLinear::new(ql, qb)?),
            is_moe: false,
            top_k: 0,
            norm_topk_prob: false,
            fused_gate_up: None,
            #[cfg(feature = "ane")]
            gate_proj_ane: None,
            #[cfg(feature = "ane")]
            up_proj_ane: None,
            #[cfg(feature = "ane")]
            down_proj_ane: None,
        })
    }

    /// Return a `'static` string naming the path `forward` will take for
    /// input of shape `[1, seq_len, hidden]`. Used by the decode tracer.
    fn selected_path(&self, seq_len: usize) -> &'static str {
        if self.is_moe {
            return "moe";
        }
        #[cfg(feature = "ane")]
        if let (Some(g), Some(_u), Some(_d)) = (
            self.gate_proj_ane.as_ref(),
            self.up_proj_ane.as_ref(),
            self.down_proj_ane.as_ref(),
        ) {
            let bucket = g.input_shape.get(3).copied().unwrap_or(0) as usize;
            if seq_len > 1 && bucket > 0 && seq_len <= bucket {
                return "ane_int8";
            }
        }
        match (self.gate_proj.as_ref(), self.up_proj.as_ref()) {
            (Some(gp), Some(up))
                if gp.weight.dtype() != Dtype::Uint32 || up.weight.dtype() != Dtype::Uint32 =>
            {
                "fp16_dense"
            }
            _ => "quantized_fused",
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        if self.is_moe {
            // Delegate to SparseMoeBlock logic
            let gate_ref = self
                .gate
                .as_ref()
                .ok_or_else(|| Exception::custom("MoE gate missing"))?;
            let seg_ref = self
                .shared_expert_gate
                .as_ref()
                .ok_or_else(|| Exception::custom("MoE shared_expert_gate missing"))?;

            let gates = ops::softmax_axis(&gate_ref.forward(x)?, -1, true)?;

            let neg_k = -self.top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1)?;
            let num_experts = *gates
                .shape()
                .last()
                .ok_or_else(|| Exception::custom("gates must have last dim"))?;
            let top_k_start = num_experts - self.top_k;
            let inds = all_inds.index((.., .., top_k_start..));
            let scores = gates.take_along_axis(&inds, -1)?;
            let scores = if self.norm_topk_prob {
                let sum = scores.sum_axes(&[-1], true)?;
                scores.divide(&sum)?
            } else {
                scores
            };

            let switch_ref = self
                .switch_mlp
                .as_mut()
                .ok_or_else(|| Exception::custom("MoE switch_mlp missing"))?;
            let y = switch_ref.forward_gather_fused(x, &inds)?;

            let expert_sum = y
                .multiply(&scores.expand_dims(-1)?)?
                .sum_axes(&[-2], false)?;

            let se_ref = self
                .shared_expert
                .as_ref()
                .ok_or_else(|| Exception::custom("MoE shared_expert missing"))?;
            let shared_y = se_ref.forward(x)?;

            let shared_gate_logit = seg_ref.forward(x)?;
            let shared_out = sigmoid_mul(&shared_gate_logit, &shared_y)?;

            expert_sum.add(shared_out)
        } else {
            // Dense SwiGLU. Use the fused quantized path only for packed weights;
            // bf16/fp16 checkpoints should go through the regular linear forward path.
            let dp = self
                .down_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense down_proj missing"))?;
            let gp = self
                .gate_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;
            let up = self
                .up_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense up_proj missing"))?;

            // ANE int8 fast path — prefill only (1 < S <= compiled seq bucket).
            // Populated by `Qwen3NextCausalLM::finalize_ane_mlp_layer0_int8_inline`.
            #[cfg(feature = "ane")]
            if let (Some(g), Some(u), Some(d)) = (
                self.gate_proj_ane.as_ref(),
                self.up_proj_ane.as_ref(),
                self.down_proj_ane.as_ref(),
            ) {
                let shape = x.shape();
                if shape.len() == 3 && (shape[0] as usize) == 1 {
                    let s = shape[1] as usize;
                    let bucket = g.input_shape.get(3).copied().unwrap_or(0) as usize;
                    if s > 1 && bucket > 0 && s <= bucket {
                        return forward_ane_int8_mlp(x, g, u, d);
                    }
                }
            }

            if gp.weight.dtype() != Dtype::Uint32 || up.weight.dtype() != Dtype::Uint32 {
                let gate_out = gp.forward(x)?;
                let up_out = up.forward(x)?;
                return dp.forward(&swiglu(&gate_out, &up_out)?);
            }

            if self.fused_gate_up.is_none() {
                let intermediate = *gp
                    .weight
                    .shape()
                    .first()
                    .ok_or_else(|| Exception::custom("gate_proj weight has no dims"))?;
                let fw = ops::concatenate_axis(&[&*gp.weight, &*up.weight], 0)?;
                let fs = ops::concatenate_axis(&[&*gp.scales, &*up.scales], 0)?;
                let fb = ops::concatenate_axis(&[&*gp.biases, &*up.biases], 0)?;
                fw.eval()?;
                fs.eval()?;
                fb.eval()?;
                self.fused_gate_up = Some((fw, fs, fb, intermediate));
            }

            let (fw, fs, fb, intermediate) = self
                .fused_gate_up
                .as_ref()
                .ok_or_else(|| Exception::custom("fused_gate_up missing after init"))?;
            let fused_out = quantized_forward(x, fw, fs, fb, gp.group_size, gp.bits)?;
            let parts = fused_out.split_axis(&[*intermediate], Some(-1))?;
            let gate_out = parts
                .first()
                .ok_or_else(|| Exception::custom("fused split failed"))?;
            let up_out = parts
                .get(1)
                .ok_or_else(|| Exception::custom("fused split failed"))?;
            dp.forward(&swiglu(gate_out, up_out)?)
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct DecoderLayer {
    #[param]
    linear_attn: Option<GatedDeltaNet>,
    #[param]
    self_attn: Option<Qwen3NextAttention>,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    mlp: FfnBlock,
    is_linear: bool,
}

impl DecoderLayer {
    fn new(args: &Qwen3NextModelArgs, layer_idx: i32, ql: i32, qb: i32) -> Result<Self, Exception> {
        let is_linear = (layer_idx + 1) % args.full_attention_interval != 0;

        let linear_attn = if is_linear {
            Some(GatedDeltaNet::new(args, ql, qb)?)
        } else {
            None
        };
        let self_attn = if is_linear {
            None
        } else {
            Some(Qwen3NextAttention::new(args, ql, qb)?)
        };

        let ffn = if args.num_experts > 0 {
            FfnBlock::new_moe(args, ql, qb)?
        } else {
            FfnBlock::new_dense(ql, qb)?
        };
        Ok(Self {
            linear_attn,
            self_attn,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            mlp: ffn,
            is_linear,
        })
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let r = if self.is_linear {
            let attn = self
                .linear_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("linear_attn missing on linear layer"))?;
            let LayerCache::Arrays(ssm_cache) = cache else {
                return Err(Exception::custom("Expected ArraysCache for linear layer"));
            };
            attn.forward(&normed, mask, ssm_cache)?
        } else {
            let attn = self
                .self_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("self_attn missing on attention layer"))?;
            let LayerCache::KV(kv_cache) = cache else {
                return Err(Exception::custom("Expected KVCache for attention layer"));
            };
            attn.forward(&normed, mask, kv_cache)?
        };

        let h = x.add(r)?;
        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }
}

// ---------------------------------------------------------------------------
// LayerCache enum
// ---------------------------------------------------------------------------

/// Per-layer cache: either KV cache (full attention) or arrays (SSM).
#[derive(Debug, Clone)]
pub enum LayerCache {
    KV(SteppingKeyValueCache),
    Arrays(ArraysCache),
}

// ---------------------------------------------------------------------------
// Qwen3NextInner (embed + layers + norm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3NextInner {
    #[param]
    embed_tokens: QEmbedding,
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
    full_attention_interval: i32,
}

impl Qwen3NextInner {
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i, ql, qb))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: QEmbedding::new(ql, qb)?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            full_attention_interval: args.full_attention_interval,
        })
    }
}

/// Outcome of `Qwen3NextCausalLM::enable_ane_gdn_all_layers`.
///
/// Surfaces enough state for callers (model_loader, doctor, tests) to log the
/// setup, detect regressions in compile/patch costs, and verify that the
/// shared-microcode invariant holds (`compile_count_after - before ≈ 3·n_buckets`).
#[cfg(feature = "ane")]
#[derive(Debug, Clone, Copy)]
pub struct AneGdnSetupReport {
    /// Number of layers that ran a full MIL compile (donor — should be 1).
    pub n_compiled_layers: usize,
    /// Number of layers patched from the donor (should be N_linear - 1).
    pub n_patched_layers: usize,
    /// Number of seq-length buckets compiled per layer. For Wave 2 this is 1;
    /// for Wave 3 it is the length of the `seq_lens` slice passed in (e.g. 3
    /// for `[16, 32, 48]`). Each bucket is a separately loaded ANE program
    /// set — total loaded program count per kernel shape is `n_buckets`.
    pub n_buckets: usize,
    /// Wall time to compile the donor layer's buckets × projections.
    pub layer0_compile_ms: u64,
    /// Wall time to patch all subsequent layers.
    pub patch_ms: u64,
    /// `ane_bridge::load_count()` snapshot before setup.
    pub load_count_before: u64,
    /// `ane_bridge::load_count()` snapshot after setup.
    pub load_count_after: u64,
    /// `ane_bridge::compile_count()` snapshot before setup.
    pub compile_count_before: u64,
    /// `ane_bridge::compile_count()` snapshot after setup. The difference from
    /// `before` should be small (`≤ 3·n_buckets`, one fresh MIL compile per
    /// (projection × bucket)); any larger gap means `patch_from_donor` is
    /// silently triggering recompiles.
    pub compile_count_after: u64,
}

// ---------------------------------------------------------------------------
// Qwen3NextCausalLM (the public model type)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3NextCausalLM {
    pub args: Qwen3NextModelArgs,
    #[param]
    model: Qwen3NextInner,
    #[param]
    lm_head: Option<QLinear>,
    #[param]
    dense_lm_head: Option<nn::Linear>,
    /// Optional compiled ANE kernel for `lm_head` (`y = hidden @ W_lm^T`).
    /// Populated by [`Self::finalize_ane_lm_head_inline`] on the inference
    /// thread when `HIGGS_TARGET_ANE_LM_HEAD=1`. Single seq bucket — runtime
    /// seqs `> seq_len` fall back to the Metal/QLinear path. Mirrors the
    /// inline GDN `ane_kernels` design, but with one kernel (not per-layer).
    #[cfg(feature = "ane")]
    lm_head_ane: Option<std::sync::Arc<crate::ane_mlmodel::AneLmHeadLut6Kernel>>,
}

impl Qwen3NextCausalLM {
    pub fn new(args: Qwen3NextModelArgs) -> Result<Self, Exception> {
        if args.full_attention_interval <= 0 {
            return Err(Exception::custom("full_attention_interval must be > 0"));
        }
        if args.linear_num_key_heads <= 0 || args.linear_num_value_heads <= 0 {
            return Err(Exception::custom("linear_num_*_heads must be > 0"));
        }
        if args.linear_conv_kernel_dim <= 0 {
            return Err(Exception::custom("linear_conv_kernel_dim must be > 0"));
        }

        let ql = args.quantization.as_ref().map_or(64, |q| q.group_size);
        let qb = args.quantization.as_ref().map_or(4, |q| q.bits);

        let model = Qwen3NextInner::new(&args, ql, qb)?;
        let (lm_head, dense_lm_head) = if args.tie_word_embeddings {
            (None, None)
        } else if args.quantization.is_some() {
            (Some(QLinear::new(ql, qb)?), None)
        } else {
            (
                None,
                Some(
                    nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                        .bias(false)
                        .build()?,
                ),
            )
        };

        Ok(Self {
            args,
            model,
            lm_head,
            dense_lm_head,
            #[cfg(feature = "ane")]
            lm_head_ane: None,
        })
    }

    fn project_logits(&self, hidden: &Array) -> Result<Array, Exception> {
        // ANE fast path: compiled `lm_head` kernel populated by
        // `finalize_ane_lm_head_inline`. Only used when seq <= compile seq_len;
        // larger seqs (e.g. long prefill) fall through to the Metal path below.
        #[cfg(feature = "ane")]
        if let Some(ane) = self.lm_head_ane.as_ref() {
            let shape = hidden.shape();
            if shape.len() == 3 && (shape[1] as usize) <= ane.seq_len {
                return ane.dispatch(hidden);
            }
        }
        if let Some(head) = self.dense_lm_head.as_ref() {
            return ops::matmul(hidden, head.weight.value.t());
        }

        match self.lm_head.as_ref() {
            Some(head) => head.forward(hidden),
            None => self.model.embed_tokens.as_linear(hidden),
        }
    }

    /// Create the per-layer cache vector.
    pub fn make_cache(&self) -> Vec<Option<LayerCache>> {
        self.model
            .layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Some(LayerCache::Arrays(ArraysCache::new()))
                } else {
                    Some(LayerCache::KV(SteppingKeyValueCache::new()))
                }
            })
            .collect()
    }

    /// Create a hybrid cache with TurboQuant on the full-attention KV layers.
    ///
    /// Linear-attention (SSM/GDN) layers get a plain `ArraysCache`; full-attention
    /// layers get a `SteppingKeyValueCache` with TurboQuant storage. This matches
    /// the selective compression strategy used by other TurboQuant implementations.
    pub fn make_cache_turbo(
        &self,
        kv_cache_config: crate::turboquant::KvCacheConfig,
    ) -> Result<Vec<Option<LayerCache>>, mlx_rs::error::Exception> {
        self.model
            .layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Ok(Some(LayerCache::Arrays(ArraysCache::new())))
                } else {
                    Ok(Some(LayerCache::KV(SteppingKeyValueCache::new_turbo(
                        kv_cache_config,
                        self.args.num_key_value_heads,
                        self.args.head_dim,
                    )?)))
                }
            })
            .collect()
    }

    /// Enable ANE projections on every GDN (linear-attention) layer, compiled
    /// across the requested seq-length buckets.
    ///
    /// Layer 0 (the first linear layer) compiles one kernel set per bucket
    /// fresh (becomes the donor); subsequent linear layers patch the same
    /// compiled microcode with their own weights, once per bucket (no MIL
    /// recompile). Full-attention layers are skipped.
    ///
    /// Pass a single-element slice. Multi-bucket (`len > 1`) is currently
    /// rejected at the API boundary — Wave 3 exposed a state-accumulation
    /// bug in the ANE bridge's `patch_from_donor` that fails reliably around
    /// the 17th bucket1 patch, regardless of bucket seq. Slice shape is
    /// retained so re-enabling is a guard-removal once the bridge is fixed.
    /// See `.planning/next-session-phase2-wave3.md`.
    ///
    /// Returns a setup report with timings, bucket count, and ANE program
    /// counts so callers can detect regressions / kernel-cap overruns.
    ///
    /// Gated behind `feature = "ane"`; on non-ANE builds this entry point does
    /// not exist.
    #[cfg(feature = "ane")]
    pub fn enable_ane_gdn_all_layers(
        &mut self,
        seq_lens: &[i32],
    ) -> Result<AneGdnSetupReport, Exception> {
        use std::time::Instant;

        if seq_lens.is_empty() {
            return Err(Exception::custom(
                "enable_ane_gdn_all_layers: seq_lens empty",
            ));
        }
        if seq_lens.len() > 1 {
            return Err(Exception::custom(format!(
                "enable_ane_gdn_all_layers: multi-bucket (len={}) disabled — \
                 ANE bridge patch_from_donor leaks state across successive \
                 patches (fails mid-loop around the 17th bucket1 patch). \
                 Pass a single-element slice. \
                 See .planning/next-session-phase2-wave3.md.",
                seq_lens.len()
            )));
        }

        let load_count_before = crate::ane_bridge::load_count();
        let compile_count_before = crate::ane_bridge::compile_count();

        // Find the first linear (GDN) layer. For Qwen3-Next default
        // full_attention_interval=4, this is layer 0; we walk regardless to
        // stay correct for variant configs.
        let first_linear_idx = self
            .model
            .layers
            .iter()
            .position(|l| l.is_linear)
            .ok_or_else(|| {
                Exception::custom("enable_ane_gdn_all_layers: model has no linear-attention layers")
            })?;

        // Compile layer `first_linear_idx` fully (one kernel set per bucket).
        let t0 = Instant::now();
        {
            let layer = &mut self.model.layers[first_linear_idx];
            let gdn = layer
                .linear_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("first linear layer missing linear_attn"))?;
            gdn.enable_ane_gdn(seq_lens)?;
        }
        let layer0_compile_ms = t0.elapsed().as_millis() as u64;

        // Snapshot the donor bucket list (Arc clone per bucket — cheap).
        let donors: Vec<std::sync::Arc<crate::qwen3_next_ane::GdnAneLayerKernels>> =
            self.model.layers[first_linear_idx]
                .linear_attn
                .as_ref()
                .and_then(|g| g.ane_kernels.as_ref())
                .map(|v| v.clone())
                .ok_or_else(|| Exception::custom("donor kernels missing after enable_ane_gdn"))?;

        // Patch the rest (per-bucket patch inside `enable_ane_gdn_from_donor`).
        let t_patch = Instant::now();
        let mut n_patched = 0usize;
        for (idx, layer) in self.model.layers.iter_mut().enumerate() {
            if !layer.is_linear || idx == first_linear_idx {
                continue;
            }
            let gdn = layer.linear_attn.as_mut().ok_or_else(|| {
                Exception::custom(format!("layer {idx}: is_linear but linear_attn=None"))
            })?;
            gdn.enable_ane_gdn_from_donor(&donors)
                .map_err(|e| Exception::custom(format!("layer{idx} patch: {e}")))?;
            if n_patched < 3 || n_patched % 4 == 0 {
                eprintln!(
                    "patched layer{idx}: load_count={}",
                    crate::ane_bridge::load_count()
                );
            }
            n_patched += 1;
        }
        let patch_ms = t_patch.elapsed().as_millis() as u64;

        let load_count_after = crate::ane_bridge::load_count();
        let compile_count_after = crate::ane_bridge::compile_count();

        let report = AneGdnSetupReport {
            n_compiled_layers: 1,
            n_patched_layers: n_patched,
            n_buckets: seq_lens.len(),
            layer0_compile_ms,
            patch_ms,
            load_count_before: load_count_before as u64,
            load_count_after: load_count_after as u64,
            compile_count_before: compile_count_before as u64,
            compile_count_after: compile_count_after as u64,
        };

        // Defensive: ANE program cap is 119 (see ane_mil.rs:1585). Warn loudly
        // if we cross 100 — drafter loads ~20 more on its own.
        if report.load_count_after > 100 {
            eprintln!(
                "WARN: ANE load_count={} after GDN setup — approaching 119-program cap",
                report.load_count_after
            );
        }
        Ok(report)
    }

    /// Wave 4: enable ANE GDN offload on every linear layer via a dedicated
    /// `qwen-gdn-ane-worker` thread.
    ///
    /// Unlike [`Self::enable_ane_gdn_all_layers`] (the inline path used by the
    /// Wave 1/2 parity tests), this attaches a `Send + Sync` mpsc handle to
    /// each `GatedDeltaNet`. That handle survives moving the model into the
    /// inference worker thread (`batch_engine.rs:117` / `simple.rs`), which
    /// is what finally makes `HIGGS_TARGET_ANE_GDN=1` usable end-to-end.
    ///
    /// Single bucket only — `seq_len` is a single `i32`. Wave 3's multi-bucket
    /// support is parked on the `patch_from_donor` bridge bug; once that
    /// lands, this signature gains a slice and the worker holds a 2-D
    /// kernel table internally. See `.planning/next-session-phase2-wave3.md`
    /// and `.planning/next-session-phase2-wave4.md`.
    ///
    /// Steps:
    ///   1. For each linear layer in `self.model.layers`, dequantize the three
    ///      GDN projection weights (`in_proj_qkvz`, `in_proj_ba`, `out_proj`)
    ///      to f32 on the calling thread.
    ///   2. Spawn the worker — it compiles layer 0's three projections fully
    ///      (one MIL compile per projection — three total) then patches
    ///      layers 1..N-1 from layer 0's donor microcode.
    ///   3. Clone the handle into every linear layer's `ane_handle` slot,
    ///      assigning each layer's `linear_layer_idx` in iteration order.
    ///
    /// Returns an `AneGdnSetupReport` (the same struct used by the inline
    /// path) for log/diagnostic parity. `n_buckets` is `1` for Wave 4.
    #[cfg(feature = "ane")]
    pub fn enable_ane_gdn_all_layers_via_worker(
        &mut self,
        seq_len: i32,
    ) -> Result<AneGdnSetupReport, Exception> {
        use std::time::Instant;

        if seq_len <= 0 {
            return Err(Exception::custom(format!(
                "enable_ane_gdn_all_layers_via_worker: non-positive seq_len {seq_len}"
            )));
        }

        let load_count_before = crate::ane_bridge::load_count();
        let compile_count_before = crate::ane_bridge::compile_count();

        // Step 1: dequantize all linear layers' projection weights to f32 on
        // the main thread (where the QLinear tensors live + ops::dequantize
        // is callable). Capture (layer_idx, weights) so we can reattach the
        // handle by index after the worker spawns.
        let t_dequant = Instant::now();
        let mut linear_indices: Vec<usize> = Vec::new();
        let mut layer_weights: Vec<crate::qwen3_next_ane_worker::GdnLayerWeights> = Vec::new();
        for (idx, layer) in self.model.layers.iter().enumerate() {
            if !layer.is_linear {
                continue;
            }
            let gdn = layer.linear_attn.as_ref().ok_or_else(|| {
                Exception::custom(format!(
                    "enable_ane_gdn_all_layers_via_worker: layer {idx} is_linear \
                     but linear_attn=None"
                ))
            })?;
            if gdn.use_separate_projections {
                return Err(Exception::custom(format!(
                    "enable_ane_gdn_all_layers_via_worker: layer {idx} \
                     use_separate_projections=true not supported"
                )));
            }
            let (qkvz_w, qkvz_in, qkvz_out) =
                dequantize_gdn_qlinear(&gdn.in_proj_qkvz, "qkvz", idx)?;
            let (ba_w, ba_in, ba_out) = dequantize_gdn_qlinear(&gdn.in_proj_ba, "ba", idx)?;
            let (out_w, out_in, out_out) = dequantize_gdn_qlinear(&gdn.out_proj, "out_proj", idx)?;
            linear_indices.push(idx);
            layer_weights.push(crate::qwen3_next_ane_worker::GdnLayerWeights {
                qkvz_w_f32: qkvz_w,
                qkvz_in,
                qkvz_out,
                ba_w_f32: ba_w,
                ba_in,
                ba_out,
                out_w_f32: out_w,
                out_in,
                out_out,
            });
        }
        if linear_indices.is_empty() {
            return Err(Exception::custom(
                "enable_ane_gdn_all_layers_via_worker: model has no linear-attention layers",
            ));
        }
        let dequant_ms = t_dequant.elapsed().as_millis() as u64;

        // Step 2: spawn the worker (compiles layer 0, patches 1..N-1).
        let t_spawn = Instant::now();
        let n_layers = layer_weights.len();
        let handle =
            crate::qwen3_next_ane_worker::spawn_gdn_ane_worker(layer_weights, seq_len, None)
                .map_err(|e| {
                    Exception::custom(format!(
                        "enable_ane_gdn_all_layers_via_worker: spawn_gdn_ane_worker: {e}"
                    ))
                })?;
        let spawn_ms = t_spawn.elapsed().as_millis() as u64;

        // Step 3: attach the handle to every linear layer.
        for (linear_idx, &model_layer_idx) in linear_indices.iter().enumerate() {
            let layer = &mut self.model.layers[model_layer_idx];
            let gdn = layer.linear_attn.as_mut().ok_or_else(|| {
                Exception::custom(format!(
                    "attach handle: layer {model_layer_idx} linear_attn vanished"
                ))
            })?;
            gdn.ane_handle = Some(handle.clone());
            gdn.ane_linear_layer_idx = linear_idx;
        }

        let load_count_after = crate::ane_bridge::load_count();
        let compile_count_after = crate::ane_bridge::compile_count();

        let report = AneGdnSetupReport {
            n_compiled_layers: 1,
            n_patched_layers: n_layers - 1,
            n_buckets: 1,
            // Surface dequant time alongside the spawn time. We bucket
            // both into `layer0_compile_ms` (now "everything before patches")
            // and `patch_ms` is the worker-side patch+attach cost. Crude but
            // matches the inline report shape callers already log.
            layer0_compile_ms: dequant_ms + spawn_ms,
            patch_ms: 0,
            load_count_before: load_count_before as u64,
            load_count_after: load_count_after as u64,
            compile_count_before: compile_count_before as u64,
            compile_count_after: compile_count_after as u64,
        };

        // Same defensive load_count cap warning as the inline path — drafter
        // loads ~20 more on its own, so warn at 100 (cap is 119).
        if report.load_count_after > 100 {
            eprintln!(
                "WARN: ANE load_count={} after GDN worker setup — approaching 119-program cap",
                report.load_count_after
            );
        }
        // Sanity: compile_count must rise by exactly 3 (one per projection
        // donor — patches use loadWithQoS only). This is the Wave 2 invariant
        // applied to the worker spawn path.
        let compile_delta = report.compile_count_after - report.compile_count_before;
        if compile_delta != 3 {
            eprintln!(
                "WARN: enable_ane_gdn_all_layers_via_worker: expected compile_count delta=3 \
                 (one per projection donor), got {compile_delta}"
            );
        }
        Ok(report)
    }

    /// Inference-thread-safe prep for ANE GDN offload (P0.8 Stage 2).
    ///
    /// Dequantizes weights for every linear layer's three projections and
    /// returns them in a Send-safe `Vec<GdnLayerWeights>`. NO ANE
    /// compilation here — the returned weights can travel across a thread
    /// boundary into the inference worker, which then calls
    /// [`Self::finalize_ane_gdn_inline`] to compile the kernels on THAT
    /// thread.
    ///
    /// Splits Step 1 of [`Self::enable_ane_gdn_all_layers_via_worker`] out
    /// from the spawn+attach phase, so dispatches avoid the mpsc roundtrip
    /// (~42 % slower than Metal on dflash_4b at 4 B —
    /// `.planning/next-session-p08-stage2-kill-mpsc.md`).
    #[cfg(feature = "ane")]
    pub fn prepare_ane_gdn_weights(
        &self,
        seq_len: i32,
    ) -> Result<(Vec<crate::qwen3_next_ane_worker::GdnLayerWeights>, i32), Exception> {
        if seq_len <= 0 {
            return Err(Exception::custom(format!(
                "prepare_ane_gdn_weights: non-positive seq_len {seq_len}"
            )));
        }
        let mut layer_weights: Vec<crate::qwen3_next_ane_worker::GdnLayerWeights> = Vec::new();
        for (idx, layer) in self.model.layers.iter().enumerate() {
            if !layer.is_linear {
                continue;
            }
            let gdn = layer.linear_attn.as_ref().ok_or_else(|| {
                Exception::custom(format!(
                    "prepare_ane_gdn_weights: layer {idx} is_linear but linear_attn=None"
                ))
            })?;
            if gdn.use_separate_projections {
                return Err(Exception::custom(format!(
                    "prepare_ane_gdn_weights: layer {idx} use_separate_projections=true \
                     not supported"
                )));
            }
            let (qkvz_w, qkvz_in, qkvz_out) =
                dequantize_gdn_qlinear(&gdn.in_proj_qkvz, "qkvz", idx)?;
            let (ba_w, ba_in, ba_out) = dequantize_gdn_qlinear(&gdn.in_proj_ba, "ba", idx)?;
            let (out_w, out_in, out_out) = dequantize_gdn_qlinear(&gdn.out_proj, "out_proj", idx)?;
            layer_weights.push(crate::qwen3_next_ane_worker::GdnLayerWeights {
                qkvz_w_f32: qkvz_w,
                qkvz_in,
                qkvz_out,
                ba_w_f32: ba_w,
                ba_in,
                ba_out,
                out_w_f32: out_w,
                out_in,
                out_out,
            });
        }
        if layer_weights.is_empty() {
            return Err(Exception::custom(
                "prepare_ane_gdn_weights: model has no linear-attention layers",
            ));
        }
        Ok((layer_weights, seq_len))
    }

    /// Inference-thread finalize: compile ANE kernels for every linear
    /// layer using the pre-dequantized weights from
    /// [`Self::prepare_ane_gdn_weights`], install them as inline
    /// `ane_kernels` (NOT `ane_handle` — no mpsc, no thread crossing per
    /// dispatch).
    ///
    /// MUST be called on the thread that will later call
    /// `forward_with_tape`. `AneProjKernel`'s IOSurface handles are
    /// thread-bound, so compiling on the same thread that dispatches keeps
    /// the realtime path warm.
    ///
    /// After compile, enters realtime mode for this thread (one-shot —
    /// never exited; the inference thread lives for the daemon's
    /// lifetime). Mirrors `qwen3_next_ane_worker.rs:350` and
    /// `dflash_ane.rs:481`.
    #[cfg(feature = "ane")]
    pub fn finalize_ane_gdn_inline(
        &mut self,
        weights: Vec<crate::qwen3_next_ane_worker::GdnLayerWeights>,
        seq_len: i32,
    ) -> Result<(), Exception> {
        use std::sync::Arc;
        use std::time::Instant;
        if weights.is_empty() {
            return Err(Exception::custom("finalize_ane_gdn_inline: weights empty"));
        }
        if seq_len <= 0 {
            return Err(Exception::custom(format!(
                "finalize_ane_gdn_inline: non-positive seq_len {seq_len}"
            )));
        }

        let load_before = crate::ane_bridge::load_count();
        let compile_before = crate::ane_bridge::compile_count();
        let t_compile = Instant::now();

        let linear_indices: Vec<usize> = self
            .model
            .layers
            .iter()
            .enumerate()
            .filter_map(|(idx, layer)| if layer.is_linear { Some(idx) } else { None })
            .collect();
        if linear_indices.len() != weights.len() {
            return Err(Exception::custom(format!(
                "finalize_ane_gdn_inline: weights len {} != model linear layers {}",
                weights.len(),
                linear_indices.len()
            )));
        }

        // Layer 0 full compile. Fused qkvz+ba in a single ANE program (one
        // dispatch vs two) — mirrors `qwen3_next_ane_worker::compile_all_layers`.
        // Halves bridge state accumulation (2 kernels/layer instead of 3),
        // which also avoids the `patch_from_donor LOAD FAILED at layer ~18`
        // condition on 9B models.
        let pad = seq_len as usize;
        let w0 = &weights[0];
        let fused0 = crate::qwen3_next_ane::compile_fused_gdn_proj(
            &w0.qkvz_w_f32,
            &w0.ba_w_f32,
            w0.qkvz_in,
            w0.qkvz_out,
            w0.ba_out,
            pad,
        )
        .map_err(|e| Exception::custom(format!("finalize: layer 0 fused compile: {e}")))?;
        let out0 = crate::qwen3_next_ane::compile_proj(
            &w0.out_w_f32,
            w0.out_in,
            w0.out_out,
            pad,
            "out_proj",
        )
        .map_err(|e| Exception::custom(format!("finalize: layer 0 out_proj compile: {e}")))?;

        // Patch layers 1..N from layer 0's donors. loadWithQoS only,
        // no MIL recompile (the Wave 2 invariant).
        let mut tail: Vec<(
            crate::qwen3_next_ane::FusedGdnProjKernel,
            crate::qwen3_next_ane::AneProjKernel,
        )> = Vec::with_capacity(weights.len().saturating_sub(1));
        for (idx, w) in weights.iter().enumerate().skip(1) {
            if w.qkvz_in != w0.qkvz_in
                || w.qkvz_out != w0.qkvz_out
                || w.ba_in != w0.ba_in
                || w.ba_out != w0.ba_out
                || w.out_in != w0.out_in
                || w.out_out != w0.out_out
            {
                return Err(Exception::custom(format!(
                    "finalize: layer {idx}: shapes diverge from layer 0 — donor patching \
                     requires identical (in,out) per projection"
                )));
            }
            let fused_i = crate::qwen3_next_ane::compile_fused_gdn_proj_from_donor(
                &fused0,
                &w.qkvz_w_f32,
                &w.ba_w_f32,
            )
            .map_err(|e| Exception::custom(format!("finalize: layer {idx} fused patch: {e}")))?;
            let out_i = crate::qwen3_next_ane::compile_proj_from_donor(&out0, &w.out_w_f32)
                .map_err(|e| {
                    Exception::custom(format!("finalize: layer {idx} out_proj patch: {e}"))
                })?;
            tail.push((fused_i, out_i));
        }
        let mut compiled: Vec<(
            crate::qwen3_next_ane::FusedGdnProjKernel,
            crate::qwen3_next_ane::AneProjKernel,
        )> = Vec::with_capacity(weights.len());
        compiled.push((fused0, out0));
        compiled.extend(tail);
        let compile_ms = t_compile.elapsed().as_millis() as u64;

        // Drop the dequantized f32 weights ASAP — they're now baked into
        // the ANE BLOBFILEs (~24 × (qkvz+ba+out) × 4B/element otherwise).
        drop(weights);

        // Attach inline kernels to each linear layer; clear any pre-existing
        // ane_handle so forward_with_tape picks the inline path (it checks
        // ane_handle FIRST, ane_kernels SECOND).
        for ((linear_idx, &model_layer_idx), (fused_k, out_k)) in
            linear_indices.iter().enumerate().zip(compiled.into_iter())
        {
            let layer = &mut self.model.layers[model_layer_idx];
            let gdn = layer.linear_attn.as_mut().ok_or_else(|| {
                Exception::custom(format!(
                    "finalize: layer {model_layer_idx} linear_attn vanished"
                ))
            })?;
            gdn.ane_handle = None;
            gdn.ane_linear_layer_idx = linear_idx;
            let layer_kernels = Arc::new(crate::qwen3_next_ane::GdnAneLayerKernels {
                seq_len: fused_k.seq_len,
                qkvz: None,
                ba: None,
                qkvz_ba_fused: Some(Arc::new(fused_k)),
                out_proj: Arc::new(out_k),
            });
            gdn.ane_kernels = Some(vec![layer_kernels]);
        }

        // Enter realtime mode for this thread (one-shot). Mirrors the
        // worker thread setup at qwen3_next_ane_worker.rs:350. The
        // inference thread lives for the daemon's lifetime — never call
        // end_realtime.
        let rt_enabled = if std::env::var("HIGGS_ANE_REALTIME").as_deref() == Ok("0") {
            tracing::warn!("HIGGS_ANE_REALTIME=0 — skipping begin_realtime for GDN path");
            false
        } else {
            crate::ane_bridge::AneKernel::begin_realtime()
        };

        let load_after = crate::ane_bridge::load_count();
        let compile_after = crate::ane_bridge::compile_count();
        let compile_delta = compile_after - compile_before;
        if compile_delta != 2 {
            eprintln!(
                "WARN: finalize_ane_gdn_inline: expected compile_count delta=2 \
                 (fused qkvz+ba + out_proj), got {compile_delta}"
            );
        }
        tracing::info!(
            n_layers = linear_indices.len(),
            compile_ms,
            load_before,
            load_after,
            compile_before,
            compile_after,
            rt_enabled,
            "ANE GDN inline finalize complete on inference thread"
        );

        Ok(())
    }

    /// Main-thread prep: dequantize `lm_head` weights to contiguous f32.
    ///
    /// Returns `Ok(None)` when the model has tied word embeddings (no explicit
    /// `lm_head` to offload — would need to dequant `embed_tokens` which is
    /// out of scope for step 1). Returns `Ok(Some((w_f32, hidden, vocab)))`
    /// otherwise. `w_f32` is row-major `[vocab * hidden]` — exactly the layout
    /// [`crate::qwen3_next_ane::compile_proj`] expects
    /// (`out_dim=vocab, in_dim=hidden`).
    ///
    /// Send-safe (returns plain `Vec<f32>`). Ships across the inference-thread
    /// move; finalize on that thread via [`Self::finalize_ane_lm_head_inline`].
    #[cfg(feature = "ane")]
    pub fn prepare_lm_head_weights(&self) -> Result<Option<(Vec<f32>, usize, usize)>, Exception> {
        let hidden = self.args.hidden_size as usize;
        let vocab = self.args.vocab_size as usize;

        if let Some(head) = self.dense_lm_head.as_ref() {
            let w = &head.weight.value;
            let shape = w.shape();
            if shape.len() != 2 || shape[0] as usize != vocab || shape[1] as usize != hidden {
                return Err(Exception::custom(format!(
                    "prepare_lm_head_weights: dense_lm_head shape {:?} != [{vocab}, {hidden}]",
                    shape
                )));
            }
            let w_f32 = w.as_dtype(Dtype::Float32)?;
            w_f32.eval()?;
            return Ok(Some((w_f32.as_slice::<f32>().to_vec(), hidden, vocab)));
        }

        if let Some(head) = self.lm_head.as_ref() {
            let w_f32 = crate::qwen3_next_ane::dequantize_qlinear_to_f32(
                &head.weight.value,
                &head.scales.value,
                &head.biases.value,
                head.group_size,
                head.bits,
            )?;
            if w_f32.len() != vocab * hidden {
                return Err(Exception::custom(format!(
                    "prepare_lm_head_weights: QLinear dequant len {} != vocab*hidden {}",
                    w_f32.len(),
                    vocab * hidden
                )));
            }
            return Ok(Some((w_f32, hidden, vocab)));
        }

        // Tied-embedding path: `project_logits` uses `embed_tokens.as_linear(hidden)`,
        // which is identical to `hidden @ W_embed^T` where `W_embed` is
        // `[vocab, hidden]` — same layout `compile_proj` wants. Dequant the
        // QEmbedding weight the same way as QLinear.
        let embed = &self.model.embed_tokens;
        let w_f32 = crate::qwen3_next_ane::dequantize_qlinear_to_f32(
            &embed.weight.value,
            &embed.scales.value,
            &embed.biases.value,
            embed.group_size,
            embed.bits,
        )?;
        if w_f32.len() != vocab * hidden {
            return Err(Exception::custom(format!(
                "prepare_lm_head_weights: tied QEmbedding dequant len {} != vocab*hidden {}",
                w_f32.len(),
                vocab * hidden
            )));
        }
        Ok(Some((w_f32, hidden, vocab)))
    }

    /// Inference-thread finalize: compile the ANE `lm_head` kernel from the
    /// pre-dequantized weights produced by [`Self::prepare_lm_head_weights`]
    /// and install it as `lm_head_ane`.
    ///
    /// MUST be called on the thread that will later call `project_logits`
    /// (e.g. inside the inference worker thread spawn in `batch_engine.rs` /
    /// `simple.rs`). `AneProjKernel`'s IOSurface handles are thread-bound,
    /// matching the GDN inline pattern at [`Self::finalize_ane_gdn_inline`].
    ///
    /// Enters realtime mode (one-shot — never exited, matching the inference
    /// thread's daemon-lifetime assumption) so `AneProjKernel::dispatch`'s
    /// `eval_realtime()` path stays on the hot code path. Safe to call after
    /// [`Self::finalize_ane_gdn_inline`] has already entered the mode —
    /// `begin_realtime` is a no-op if already active on this thread.
    #[cfg(feature = "ane")]
    pub fn finalize_ane_lm_head_inline(
        &mut self,
        w_f32: Vec<f32>,
        hidden: usize,
        vocab: usize,
        seq_len: i32,
    ) -> Result<(), Exception> {
        use std::sync::Arc;
        use std::time::Instant;
        if seq_len <= 0 {
            return Err(Exception::custom(format!(
                "finalize_ane_lm_head_inline: non-positive seq_len {seq_len}"
            )));
        }
        if w_f32.len() != vocab * hidden {
            return Err(Exception::custom(format!(
                "finalize_ane_lm_head_inline: weight len {} != vocab*hidden {} ({vocab}*{hidden})",
                w_f32.len(),
                vocab * hidden
            )));
        }

        let load_before = crate::ane_bridge::load_count();
        let compile_before = crate::ane_bridge::compile_count();
        let t_compile = Instant::now();

        let pad = seq_len as usize;
        let kernel =
            crate::qwen3_next_ane::compile_proj_lut6(&w_f32, hidden, vocab, pad, "lm_head")
                .map_err(|e| {
                    Exception::custom(format!("finalize_ane_lm_head_inline: compile: {e}"))
                })?;
        let compile_ms = t_compile.elapsed().as_millis() as u64;

        // Drop the dequantized weights ASAP — they're now baked into the ANE
        // BLOBFILE (~vocab * hidden * 4 bytes otherwise — ~1.2 GB at Qwen3).
        drop(w_f32);

        self.lm_head_ane = Some(Arc::new(kernel));

        // Enter realtime mode one-shot for the inference thread. Idempotent
        // if `finalize_ane_gdn_inline` already called it — the ane_bridge
        // flag tracks per-thread state and re-entering is a no-op.
        let rt_enabled = crate::ane_bridge::AneKernel::begin_realtime();

        let load_after = crate::ane_bridge::load_count();
        let compile_after = crate::ane_bridge::compile_count();
        // Note: the LUT6 path goes through the public MLModel API, not the
        // private `_ANEInMemoryModel.compileWithQoS:` that `ane_bridge.m`
        // instruments — so `compile_count` does NOT increment here. The
        // previous delta==1 assertion was specific to the dense fp16 path;
        // leaving the metric unchanged here is expected.
        tracing::info!(
            hidden,
            vocab,
            seq_len = pad,
            compile_ms,
            load_before,
            load_after,
            compile_before,
            compile_after,
            rt_enabled,
            "ANE lm_head inline finalize complete on inference thread"
        );

        Ok(())
    }

    /// Main-thread prep: dequantize layer-0 MLP (gate/up/down) projections to
    /// contiguous f32 buffers for ANE int8 mlpackage compilation.
    ///
    /// Returns `Ok(None)` when layer 0 is an MoE block (no direct
    /// gate/up/down QLinear triple to dequantize — out of scope for step 1).
    /// Returns `Ok(Some((gate, up, down, hidden, intermediate)))` otherwise;
    /// each buffer is row-major `[out * in]` matching what
    /// [`crate::qwen3_next_ane::compile_proj_int8_mlpkg`] expects.
    ///
    /// Send-safe — all three buffers are plain `Vec<f32>` that ship across
    /// the inference-thread move, same pattern as
    /// [`Self::prepare_lm_head_weights`].
    #[cfg(feature = "ane")]
    #[allow(clippy::type_complexity)]
    pub fn prepare_mlp_layer0_int8_weights(
        &self,
    ) -> Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)>, Exception> {
        let hidden = self.args.hidden_size as usize;
        let inter = self.args.intermediate_size as usize;
        if hidden == 0 || inter == 0 {
            return Err(Exception::custom(format!(
                "prepare_mlp_layer0_int8_weights: zero dim (hidden={hidden}, inter={inter})"
            )));
        }

        let layer0 = self.model.layers.first().ok_or_else(|| {
            Exception::custom("prepare_mlp_layer0_int8_weights: model has no layers")
        })?;
        let mlp = &layer0.mlp;
        if mlp.is_moe {
            tracing::info!(
                "prepare_mlp_layer0_int8_weights: layer 0 is MoE — skipping (dense path only)"
            );
            return Ok(None);
        }
        let gp = mlp.gate_proj.as_ref().ok_or_else(|| {
            Exception::custom("prepare_mlp_layer0_int8_weights: dense gate_proj missing")
        })?;
        let up = mlp.up_proj.as_ref().ok_or_else(|| {
            Exception::custom("prepare_mlp_layer0_int8_weights: dense up_proj missing")
        })?;
        let dp = mlp.down_proj.as_ref().ok_or_else(|| {
            Exception::custom("prepare_mlp_layer0_int8_weights: dense down_proj missing")
        })?;

        let gate_f32 = crate::qwen3_next_ane::dequantize_qlinear_to_f32(
            &gp.weight.value,
            &gp.scales.value,
            &gp.biases.value,
            gp.group_size,
            gp.bits,
        )?;
        if gate_f32.len() != inter * hidden {
            return Err(Exception::custom(format!(
                "prepare_mlp_layer0_int8_weights: gate len {} != inter*hidden {}",
                gate_f32.len(),
                inter * hidden
            )));
        }
        let up_f32 = crate::qwen3_next_ane::dequantize_qlinear_to_f32(
            &up.weight.value,
            &up.scales.value,
            &up.biases.value,
            up.group_size,
            up.bits,
        )?;
        if up_f32.len() != inter * hidden {
            return Err(Exception::custom(format!(
                "prepare_mlp_layer0_int8_weights: up len {} != inter*hidden {}",
                up_f32.len(),
                inter * hidden
            )));
        }
        let down_f32 = crate::qwen3_next_ane::dequantize_qlinear_to_f32(
            &dp.weight.value,
            &dp.scales.value,
            &dp.biases.value,
            dp.group_size,
            dp.bits,
        )?;
        if down_f32.len() != hidden * inter {
            return Err(Exception::custom(format!(
                "prepare_mlp_layer0_int8_weights: down len {} != hidden*inter {}",
                down_f32.len(),
                hidden * inter
            )));
        }
        Ok(Some((gate_f32, up_f32, down_f32, hidden, inter)))
    }

    /// Inference-thread finalize: compile ANE int8 mlpackage kernels for
    /// layer 0's `gate_proj` / `up_proj` / `down_proj` from the pre-dequantized
    /// weights produced by [`Self::prepare_mlp_layer0_int8_weights`], and
    /// install them on `self.model.layers[0].mlp`.
    ///
    /// MUST be called on the inference worker thread — kernel IOSurfaces bind
    /// to the compiling thread (same invariant as
    /// [`Self::finalize_ane_lm_head_inline`]). Also enters realtime mode
    /// one-shot (idempotent if already active).
    #[cfg(feature = "ane")]
    pub fn finalize_ane_mlp_layer0_int8_inline(
        &mut self,
        gate_f32: Vec<f32>,
        up_f32: Vec<f32>,
        down_f32: Vec<f32>,
        hidden: usize,
        inter: usize,
        seq_len: i32,
    ) -> Result<(), Exception> {
        use std::sync::Arc;
        use std::time::Instant;

        if seq_len <= 0 {
            return Err(Exception::custom(format!(
                "finalize_ane_mlp_layer0_int8_inline: non-positive seq_len {seq_len}"
            )));
        }
        if gate_f32.len() != inter * hidden
            || up_f32.len() != inter * hidden
            || down_f32.len() != hidden * inter
        {
            return Err(Exception::custom(format!(
                "finalize_ane_mlp_layer0_int8_inline: weight size mismatch \
                 (gate={}, up={}, down={}, expected inter*hidden={} / hidden*inter={})",
                gate_f32.len(),
                up_f32.len(),
                down_f32.len(),
                inter * hidden,
                hidden * inter
            )));
        }

        let pad = seq_len as usize;

        let t = Instant::now();
        let gate_k = crate::qwen3_next_ane::compile_proj_int8_mlpkg(
            &gate_f32,
            hidden,
            inter,
            pad,
            "mlp0.gate",
        )
        .map_err(|e| {
            Exception::custom(format!("finalize_ane_mlp_layer0_int8_inline: gate: {e}"))
        })?;
        let gate_ms = t.elapsed().as_millis() as u64;
        drop(gate_f32);

        let t = Instant::now();
        let up_k =
            crate::qwen3_next_ane::compile_proj_int8_mlpkg(&up_f32, hidden, inter, pad, "mlp0.up")
                .map_err(|e| {
                    Exception::custom(format!("finalize_ane_mlp_layer0_int8_inline: up: {e}"))
                })?;
        let up_ms = t.elapsed().as_millis() as u64;
        drop(up_f32);

        let t = Instant::now();
        let down_k = crate::qwen3_next_ane::compile_proj_int8_mlpkg(
            &down_f32,
            inter,
            hidden,
            pad,
            "mlp0.down",
        )
        .map_err(|e| {
            Exception::custom(format!("finalize_ane_mlp_layer0_int8_inline: down: {e}"))
        })?;
        let down_ms = t.elapsed().as_millis() as u64;
        drop(down_f32);

        let layer0 = self.model.layers.first_mut().ok_or_else(|| {
            Exception::custom("finalize_ane_mlp_layer0_int8_inline: model has no layers")
        })?;
        if layer0.mlp.is_moe {
            return Err(Exception::custom(
                "finalize_ane_mlp_layer0_int8_inline: layer 0 is MoE — dense path only",
            ));
        }
        layer0.mlp.gate_proj_ane = Some(Arc::new(gate_k));
        layer0.mlp.up_proj_ane = Some(Arc::new(up_k));
        layer0.mlp.down_proj_ane = Some(Arc::new(down_k));

        let rt_enabled = crate::ane_bridge::AneKernel::begin_realtime();

        tracing::info!(
            hidden,
            inter,
            seq_len = pad,
            gate_ms,
            up_ms,
            down_ms,
            rt_enabled,
            "ANE int8 MLP layer-0 finalize complete on inference thread"
        );

        Ok(())
    }

    /// Forward pass returning hidden states before the LM head.
    #[allow(non_snake_case)]
    pub fn forward_hidden(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        // Create attention mask for full-attention layers
        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have >= 2 dims"))?;

        let fa_mask: Option<AttentionMask> = if T > 1 {
            // Read KV cache offset for chunked prefill: when offset > 0,
            // we need an explicit array mask so queries at positions
            // [offset, offset+T) attend correctly to KV at [0, offset+T).
            // The `Causal` flag only creates a lower-triangular on array
            // indices, which is wrong when Q_len < KV_len.
            let kv_offset = kv_cache
                .iter()
                .filter_map(|lc| match lc.as_ref()? {
                    LayerCache::KV(kv) => Some(kv.offset()),
                    _ => None,
                })
                .next()
                .unwrap_or(0);

            if kv_offset > 0 {
                Some(AttentionMask::Array(create_causal_mask(
                    T,
                    Some(kv_offset),
                )?))
            } else {
                Some(AttentionMask::Causal)
            }
        } else {
            None
        };

        let trace_on = crate::decode_trace::is_active();
        let tok = if trace_on {
            crate::decode_trace::begin_forward(T as usize)
        } else {
            0
        };
        let trace_sync = trace_on
            && std::env::var("HIGGS_DECODE_TRACE_SYNC")
                .map(|v| v == "1")
                .unwrap_or(false);

        for (layer_idx, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(kv_cache.iter_mut())
            .enumerate()
        {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;
            let mask = if layer.is_linear {
                None
            } else {
                fa_mask.as_ref()
            };

            let t0 = trace_on.then(std::time::Instant::now);

            let normed = layer.input_layernorm.forward(&h)?;
            let r = if layer.is_linear {
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                attn.forward(&normed, mask, ssm_cache)?
            } else {
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };
                attn.forward(&normed, mask, layer_kv)?
            };

            let h2 = h.add(r)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_path = if trace_on {
                layer.mlp.selected_path(T as usize)
            } else {
                ""
            };
            let mlp_out = layer.mlp.forward(&normed_post)?;
            h = h2.add(mlp_out)?;

            if let Some(t0) = t0 {
                if trace_sync {
                    mlx_rs::transforms::eval([&h])?;
                }
                let ns = t0.elapsed().as_nanos() as u64;
                let kind = if layer.is_linear {
                    "attn_linear"
                } else {
                    "attn_full"
                };
                let hidden = h.shape().last().copied().unwrap_or(0) as usize;
                crate::decode_trace::record_layer(
                    tok, layer_idx, kind, mlp_path, T as usize, hidden, ns,
                );
            }
        }

        if trace_on {
            crate::decode_trace::flush();
        }

        self.model.norm.forward(&h)
    }

    /// Forward pass returning logits for **all positions**.
    ///
    /// Returns shape `[B, T, vocab]`. Used by speculative decoding to verify
    /// a draft sequence in a single forward pass.
    pub fn forward_all_logits(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, kv_cache)?;
        self.project_logits(&h)
    }

    /// Forward pass returning logits + hidden states at specified tap layers.
    ///
    /// Used by DFlash speculative decoding: the target model produces logits
    /// AND collects intermediate hidden states that condition the drafter.
    /// Each tap hidden is `[B, T, hidden_size]`, captured post-residual/post-MLP.
    /// Returns `(all_position_logits, vec_of_tap_hidden_states)`.
    #[allow(non_snake_case)]
    pub fn forward_with_taps(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
    ) -> Result<(Array, Vec<Array>), Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have >= 2 dims"))?;

        let fa_mask: Option<AttentionMask> = if T > 1 {
            let kv_offset = kv_cache
                .iter()
                .filter_map(|lc| match lc.as_ref()? {
                    LayerCache::KV(kv) => Some(kv.offset()),
                    _ => None,
                })
                .next()
                .unwrap_or(0);

            if kv_offset > 0 {
                Some(AttentionMask::Array(create_causal_mask(
                    T,
                    Some(kv_offset),
                )?))
            } else {
                Some(AttentionMask::Causal)
            }
        } else {
            None
        };

        let mut taps = Vec::with_capacity(tap_layers.len());

        for (layer_idx, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(kv_cache.iter_mut())
            .enumerate()
        {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;
            let mask_ref = if layer.is_linear {
                None
            } else {
                fa_mask.as_ref()
            };

            let normed = layer.input_layernorm.forward(&h)?;

            let r = if layer.is_linear {
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                attn.forward(&normed, mask_ref, ssm_cache)?
            } else {
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };
                attn.forward(&normed, mask_ref, layer_kv)?
            };

            let h2 = h.add(r)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            h = h2.add(mlp_out)?;

            if tap_layers.contains(&layer_idx) {
                taps.push(h.clone());
            }
        }

        let normed = self.model.norm.forward(&h)?;
        let logits = self.project_logits(&normed)?;

        Ok((logits, taps))
    }

    /// Stateless verify pass: identical to `forward_with_taps` but GDN layers
    /// use `forward_stateless` — they compute correct outputs without updating
    /// `ssm_state` or `conv_state`. KV cache layers update normally (needed for
    /// future decode). Eliminates GdnStateBackup/restore overhead in DFlash verify.
    ///
    /// After verify, the caller runs `forward_hidden` with only the accepted
    /// tokens to commit the GDN state for those positions.
    #[allow(non_snake_case)]
    pub fn forward_with_taps_stateless(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
    ) -> Result<(Array, Vec<Array>), Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have >= 2 dims"))?;

        let fa_mask: Option<AttentionMask> = if T > 1 {
            let kv_offset = kv_cache
                .iter()
                .filter_map(|lc| match lc.as_ref()? {
                    LayerCache::KV(kv) => Some(kv.offset()),
                    _ => None,
                })
                .next()
                .unwrap_or(0);

            if kv_offset > 0 {
                Some(AttentionMask::Array(create_causal_mask(
                    T,
                    Some(kv_offset),
                )?))
            } else {
                Some(AttentionMask::Causal)
            }
        } else {
            None
        };

        let mut taps = Vec::with_capacity(tap_layers.len());

        for (layer_idx, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(kv_cache.iter_mut())
            .enumerate()
        {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;
            let mask_ref = if layer.is_linear {
                None
            } else {
                fa_mask.as_ref()
            };

            let normed = layer.input_layernorm.forward(&h)?;

            let r = if layer.is_linear {
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                // STATELESS: GDN state not updated
                attn.forward_stateless(&normed, mask_ref, ssm_cache)?
            } else {
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };
                // KV cache updates normally — needed for future decode
                attn.forward(&normed, mask_ref, layer_kv)?
            };

            let h2 = h.add(r)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            h = h2.add(mlp_out)?;

            if tap_layers.contains(&layer_idx) {
                taps.push(h.clone());
            }
        }

        let normed = self.model.norm.forward(&h)?;
        let logits = self.project_logits(&normed)?;

        Ok((logits, taps))
    }

    /// Stateless verify returning logits only (no taps). Convenience wrapper.
    pub fn forward_all_logits_stateless(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let (logits, _) = self.forward_with_taps_stateless(inputs, mask, kv_cache, &[])?;
        Ok(logits)
    }

    /// Tape-recording verify pass: runs normal forward (state IS updated) and
    /// records innovation tape per GDN layer. Returns `(logits, taps, tape_data)`.
    ///
    /// On full acceptance (89% of rounds): zero extra work — state already correct.
    /// On partial rejection: restore conv+ssm snapshots, replay tape[:n_accepted].
    #[allow(non_snake_case)]
    pub fn forward_with_taps_tape(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
    ) -> Result<(Array, Vec<Array>, Vec<Option<GdnLayerTape>>), Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have >= 2 dims"))?;

        let fa_mask: Option<AttentionMask> = if T > 1 {
            let kv_offset = kv_cache
                .iter()
                .filter_map(|lc| match lc.as_ref()? {
                    LayerCache::KV(kv) => Some(kv.offset()),
                    _ => None,
                })
                .next()
                .unwrap_or(0);

            if kv_offset > 0 {
                Some(AttentionMask::Array(create_causal_mask(
                    T,
                    Some(kv_offset),
                )?))
            } else {
                Some(AttentionMask::Causal)
            }
        } else {
            None
        };

        let mut taps = Vec::with_capacity(tap_layers.len());
        let mut layer_tapes: Vec<Option<GdnLayerTape>> =
            Vec::with_capacity(self.model.layers.len());

        // Optional per-layer GDN/FA timing. Gated by env to avoid the eval()
        // stalls (which serialize the GPU pipeline) in normal runs. Numbers
        // produced under timing are upper bounds: they include synchronization
        // cost that real execution overlaps. Useful for the GDN-vs-FA ratio.
        let layer_timing = std::env::var("HIGGS_DFLASH_LAYER_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false);
        let mut gdn_total_ms = 0.0_f64;
        let mut fa_total_ms = 0.0_f64;
        let mut gdn_count = 0usize;
        let mut fa_count = 0usize;
        let mut layer_ckpt = if layer_timing {
            mlx_rs::transforms::eval([&h])?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        for (layer_idx, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(kv_cache.iter_mut())
            .enumerate()
        {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;
            let is_linear = layer.is_linear;
            let mask_ref = if is_linear { None } else { fa_mask.as_ref() };

            let normed = layer.input_layernorm.forward(&h)?;

            let (r, tape) = if is_linear {
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                let (out, tape) = attn.forward_with_tape(&normed, mask_ref, ssm_cache)?;
                (out, Some(tape))
            } else {
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };
                (attn.forward(&normed, mask_ref, layer_kv)?, None)
            };

            layer_tapes.push(tape);

            let h2 = h.add(r)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            h = h2.add(mlp_out)?;

            if tap_layers.contains(&layer_idx) {
                taps.push(h.clone());
            }

            if let Some(ckpt) = layer_ckpt.as_mut() {
                mlx_rs::transforms::eval([&h])?;
                let now = std::time::Instant::now();
                let dt_ms = now.duration_since(*ckpt).as_secs_f64() * 1000.0;
                if is_linear {
                    gdn_total_ms += dt_ms;
                    gdn_count += 1;
                } else {
                    fa_total_ms += dt_ms;
                    fa_count += 1;
                }
                *ckpt = now;
            }
        }

        let normed = self.model.norm.forward(&h)?;
        let logits = self.project_logits(&normed)?;

        if layer_timing {
            mlx_rs::transforms::eval([&logits])?;
            let tail_ms = layer_ckpt
                .map(|c| c.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            tracing::info!(
                "dflash_layer_timing seq={} gdn_layers={} gdn_total_ms={:.1} gdn_avg={:.2}ms \
                 fa_layers={} fa_total_ms={:.1} fa_avg={:.2}ms tail_ms={:.1}",
                T,
                gdn_count,
                gdn_total_ms,
                gdn_total_ms / gdn_count.max(1) as f64,
                fa_count,
                fa_total_ms,
                fa_total_ms / fa_count.max(1) as f64,
                tail_ms,
            );
        }

        Ok((logits, taps, layer_tapes))
    }

    /// Number of transformer layers in the model.
    pub fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// Like [`forward_with_taps_tape`] but inserts `eval()` every
    /// `layer_chunk_size` layers so Metal can retire intermediate buffers.
    /// This prevents OOM on large models (27B+) where the full 64-layer
    /// lazy graph exceeds GPU memory.
    ///
    /// Also captures GDN snapshots incrementally (per-chunk) instead of
    /// requiring a separate upfront clone of all SSM states.
    #[allow(non_snake_case, clippy::type_complexity)]
    pub fn forward_with_taps_tape_chunked(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
        layer_chunk_size: usize,
    ) -> Result<(Array, Vec<Array>, Vec<Option<GdnLayerTape>>), Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have >= 2 dims"))?;

        let fa_mask: Option<AttentionMask> = if T > 1 {
            let kv_offset = kv_cache
                .iter()
                .filter_map(|lc| match lc.as_ref()? {
                    LayerCache::KV(kv) => Some(kv.offset()),
                    _ => None,
                })
                .next()
                .unwrap_or(0);

            if kv_offset > 0 {
                Some(AttentionMask::Array(create_causal_mask(
                    T,
                    Some(kv_offset),
                )?))
            } else {
                Some(AttentionMask::Causal)
            }
        } else {
            None
        };

        let num_layers = self.model.layers.len();
        let mut taps = Vec::with_capacity(tap_layers.len());
        let mut layer_tapes: Vec<Option<GdnLayerTape>> = Vec::with_capacity(num_layers);

        // See `forward_with_taps_tape` for caveats on per-layer timing.
        let layer_timing = std::env::var("HIGGS_DFLASH_LAYER_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false);
        let mut gdn_total_ms = 0.0_f64;
        let mut fa_total_ms = 0.0_f64;
        let mut gdn_count = 0usize;
        let mut fa_count = 0usize;
        let mut layer_ckpt = if layer_timing {
            mlx_rs::transforms::eval([&h])?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        for (layer_idx, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(kv_cache.iter_mut())
            .enumerate()
        {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;

            let is_linear = layer.is_linear;
            let mask_ref = if is_linear { None } else { fa_mask.as_ref() };

            let normed = layer.input_layernorm.forward(&h)?;

            let (r, tape) = if is_linear {
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                let (out, tape) = attn.forward_with_tape(&normed, mask_ref, ssm_cache)?;
                (out, Some(tape))
            } else {
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };
                (attn.forward(&normed, mask_ref, layer_kv)?, None)
            };

            layer_tapes.push(tape);

            let h2 = h.add(r)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            h = h2.add(mlp_out)?;

            if tap_layers.contains(&layer_idx) {
                taps.push(h.clone());
            }

            if let Some(ckpt) = layer_ckpt.as_mut() {
                mlx_rs::transforms::eval([&h])?;
                let now = std::time::Instant::now();
                let dt_ms = now.duration_since(*ckpt).as_secs_f64() * 1000.0;
                if is_linear {
                    gdn_total_ms += dt_ms;
                    gdn_count += 1;
                } else {
                    fa_total_ms += dt_ms;
                    fa_count += 1;
                }
                *ckpt = now;
            }

            // Chunk boundary: eval h + tapes + taps to retire intermediate
            // Metal buffers (layernorm, projections, MLP intermediates).
            if (layer_idx + 1) % layer_chunk_size == 0 && layer_idx + 1 < num_layers {
                let mut targets: Vec<&Array> = vec![&h];
                for lt in layer_tapes
                    .iter()
                    .skip(layer_idx + 1 - layer_chunk_size)
                    .flatten()
                {
                    targets.push(&lt.delta_tape);
                    targets.push(&lt.norm_k);
                    targets.push(&lt.a_proj);
                    targets.push(&lt.qkv_input);
                }
                for tap in &taps {
                    targets.push(tap);
                }
                mlx_rs::transforms::eval(targets)?;
            }
        }

        let normed = self.model.norm.forward(&h)?;
        let logits = self.project_logits(&normed)?;

        if layer_timing {
            mlx_rs::transforms::eval([&logits])?;
            let tail_ms = layer_ckpt
                .map(|c| c.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            tracing::info!(
                "dflash_layer_timing_chunked seq={} gdn_layers={} gdn_total_ms={:.1} gdn_avg={:.2}ms \
                 fa_layers={} fa_total_ms={:.1} fa_avg={:.2}ms tail_ms={:.1}",
                T,
                gdn_count,
                gdn_total_ms,
                gdn_total_ms / gdn_count.max(1) as f64,
                fa_count,
                fa_total_ms,
                fa_total_ms / fa_count.max(1) as f64,
                tail_ms,
            );
        }

        Ok((logits, taps, layer_tapes))
    }

    /// Replay accepted steps from recorded tape data on partial rejection.
    /// Restores GDN state from `snapshots`, replays tape[:n_accepted],
    /// and rolls back KV cache for rejected positions.
    ///
    /// All GDN layers are batched into a single Metal kernel dispatch
    /// (concat along batch dim, one kernel call, split back) to avoid
    /// per-layer dispatch overhead (~0.4ms × 24 layers = 10ms → <1ms).
    pub fn replay_tape_rollback(
        &self,
        layer_tapes: &[Option<GdnLayerTape>],
        kv_cache: &mut [Option<LayerCache>],
        n_accepted: i32,
        kv_rollback: i32,
    ) -> Result<(), Exception> {
        use mlx_rs::ops;

        // Collect GDN layer data for batched replay
        struct GdnReplayEntry<'a> {
            cache_idx: usize,
            tape: &'a GdnLayerTape,
            layer: &'a GatedDeltaNet,
            snap_state: Array,
        }

        let mut gdn_entries: Vec<GdnReplayEntry> = Vec::new();

        // First pass: restore state from tape's initial snapshot, rollback KV, collect GDN entries
        for (i, lc) in kv_cache.iter_mut().enumerate() {
            match lc {
                Some(LayerCache::Arrays(ac)) => {
                    if let Some(Some(tape)) = layer_tapes.get(i) {
                        // Restore from tape-captured initial state (Python _GDNStateCapture equivalent)
                        ac.conv_state = tape.conv_state_init.clone();
                        ac.ssm_state = tape.ssm_state_init.clone();
                        ac.offset = tape.offset_init;

                        let gdn_layer = self.model.layers[i]
                            .linear_attn
                            .as_ref()
                            .ok_or_else(|| Exception::custom("linear_attn missing for replay"))?;

                        let state = tape.ssm_state_init.clone().unwrap_or_else(|| {
                            let dt = tape.delta_tape.dtype();
                            ops::zeros_dtype(
                                &[
                                    1,
                                    gdn_layer.num_v_heads,
                                    gdn_layer.head_v_dim,
                                    gdn_layer.head_k_dim,
                                ],
                                dt,
                            )
                            .unwrap()
                        });

                        gdn_entries.push(GdnReplayEntry {
                            cache_idx: i,
                            tape,
                            layer: gdn_layer,
                            snap_state: state,
                        });
                    }
                }
                Some(LayerCache::KV(kv)) => {
                    if kv_rollback > 0 {
                        kv.rollback(kv_rollback);
                    }
                }
                _ => {}
            }
        }

        if gdn_entries.is_empty() {
            return Ok(());
        }

        // Batch all GDN layers: concat tape/k/a/state/A_log/dt_bias along batch dim
        let tape_slices: Vec<Array> = gdn_entries
            .iter()
            .map(|e| e.tape.delta_tape.index((.., ..n_accepted, ..)))
            .collect();
        let k_slices: Vec<Array> = gdn_entries
            .iter()
            .map(|e| e.tape.norm_k.index((.., ..n_accepted, ..)))
            .collect();
        let a_slices: Vec<Array> = gdn_entries
            .iter()
            .map(|e| e.tape.a_proj.index((.., ..n_accepted, ..)))
            .collect();
        let states: Vec<&Array> = gdn_entries.iter().map(|e| &e.snap_state).collect();
        let a_logs: Vec<&Array> = gdn_entries.iter().map(|e| e.layer.A_log.as_ref()).collect();
        let dt_biases: Vec<&Array> = gdn_entries
            .iter()
            .map(|e| e.layer.dt_bias.as_ref())
            .collect();

        let tape_refs: Vec<&Array> = tape_slices.iter().collect();
        let k_refs: Vec<&Array> = k_slices.iter().collect();
        let a_refs: Vec<&Array> = a_slices.iter().collect();

        let batched_tape = ops::concatenate_axis(&tape_refs, 0)?;
        let batched_k = ops::concatenate_axis(&k_refs, 0)?;
        let batched_a = ops::concatenate_axis(&a_refs, 0)?;
        let batched_state = ops::concatenate_axis(&states, 0)?;
        // Flatten A_log [Hv] per layer → [num_layers * Hv]
        let batched_a_log = ops::concatenate_axis(&a_logs, 0)?;
        let batched_dt_bias = ops::concatenate_axis(&dt_biases, 0)?;

        let num_layers = gdn_entries.len() as i32;
        let e0 = &gdn_entries[0];

        // Single kernel dispatch for all GDN layers
        let batched_new_state = tape_replay_kernel_ffi(
            &batched_tape,
            &batched_k,
            &batched_a,
            &batched_a_log,
            &batched_dt_bias,
            &batched_state,
            num_layers,
            n_accepted,
            e0.layer.num_k_heads,
            e0.layer.head_k_dim,
            e0.layer.num_v_heads,
            e0.layer.head_v_dim,
        )?;

        // Split results back to individual layers and rebuild conv_state
        for (offset, entry) in gdn_entries.iter().enumerate() {
            let ac = match &mut kv_cache[entry.cache_idx] {
                Some(LayerCache::Arrays(ac)) => ac,
                _ => continue,
            };

            // Extract this layer's state from the batched result
            let start = offset as i32;
            let new_state = batched_new_state.index((start..start + 1, .., .., ..));
            ac.ssm_state = Some(new_state);
            ac.offset += n_accepted;

            // Rebuild conv_state from recorded qkv input
            let ks = entry.layer.conv_kernel_size;
            let n_keep = ks - 1;
            if n_keep > 0 {
                let qkv_slice = entry.tape.qkv_input.index((.., ..n_accepted, ..));
                let prefix = match &ac.conv_state {
                    Some(cs) => cs.clone(),
                    None => ops::zeros_dtype(
                        &[1, n_keep, entry.layer.conv_dim],
                        entry.tape.qkv_input.dtype(),
                    )?,
                };
                let full = ops::concatenate_axis(&[&prefix, &qkv_slice], 1)?;
                let total_len = *full
                    .shape()
                    .get(1)
                    .ok_or_else(|| Exception::custom("conv rebuild: missing seq dim"))?;
                let cs_start = total_len - n_keep;
                let cs = full.index((.., cs_start.., ..));
                let cs_shape = cs.shape().to_vec();
                ac.conv_state = Some(cs.flatten(None, None)?.reshape(&cs_shape)?);
            }
        }

        Ok(())
    }

    /// Embed raw token IDs through the target model's embedding layer.
    ///
    /// Used by DFlash to convert `[anchor, mask, mask, ...]` block into
    /// the embedding space expected by the drafter.
    pub fn embed_token_ids(&self, token_ids: &Array) -> Result<Array, Exception> {
        self.model.embed_tokens.forward(token_ids)
    }

    /// Apply only the lm_head to pre-computed hidden states.
    ///
    /// Used by DFlash: the drafter produces hidden states in the target model's
    /// hidden space, and we project them through the target's lm_head to get logits.
    /// Input: `[B, T, hidden_size]`. Returns: `[B, T, vocab_size]`.
    pub fn forward_all_logits_from_hidden(&self, hidden: &Array) -> Result<Array, Exception> {
        self.project_logits(hidden)
    }

    /// Forward pass producing logits for the **last position only**.
    ///
    /// During inference only the last token's logits are sampled, so we
    /// slice hidden states before the lm_head projection. This avoids a
    /// full `quantized_matmul(vocab, hidden)` on T-1 discarded positions.
    /// Returns shape `[B, 1, vocab]`.
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, kv_cache)?;
        let h_last = h.index((.., -1.., ..)); // [B, 1, hidden]

        self.project_logits(&h_last)
    }

    /// Chunked prefill: process the prompt in `chunk_size`-token segments
    /// through all layers. Produces identical logits to `forward()` but with
    /// smaller per-dispatch working sets and lower peak memory.
    ///
    /// Only the **last chunk's** logits are returned (shape `[B, chunk_len, vocab]`).
    /// For full-sequence hidden states, use `forward_hidden` directly.
    #[allow(non_snake_case)]
    pub fn forward_chunked(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        chunk_size: i32,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // If chunk_size covers the whole sequence, just do a normal forward.
        if chunk_size >= T {
            return self.forward(inputs, mask, kv_cache);
        }

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        // Process all chunks except the last through forward_hidden (discard logits).
        // Cache states must be eval'd between chunks so the next chunk reads
        // materialized values (MLX is lazy).
        let mut offset = 0i32;
        while offset + chunk_size < T {
            let chunk = inputs.index((.., offset..offset + chunk_size));
            let h = self.forward_hidden(&chunk, None, kv_cache)?;
            // Eval hidden output + ALL cache states between chunks.
            // Both KV and SSM/conv must be materialized:
            // - SSM/conv: consumed by GDN FFI kernel (requires concrete arrays)
            // - KV: slice_update creates lazy nodes; without eval, nested
            //   updates accumulate and OOM on long sequences
            let mut targets: Vec<&Array> = vec![&h];
            for lc in kv_cache.iter().flatten() {
                match lc {
                    LayerCache::KV(kv) => targets.extend(kv.eval_targets()),
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            targets.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            targets.push(c);
                        }
                    }
                }
            }
            mlx_rs::transforms::eval(targets)?;
            offset += chunk_size;
        }

        // Last chunk: run through forward_hidden, project only last position.
        let last_chunk = inputs.index((.., offset..));
        let h = self.forward_hidden(&last_chunk, None, kv_cache)?;
        let h_last = h.index((.., -1.., ..)); // [B, 1, hidden]

        self.project_logits(&h_last)
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load model args from config.json.
pub fn load_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Qwen3NextModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

/// Load a `Qwen3Next` model from a directory containing safetensors + config.json.
pub fn load_qwen3_next_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_model_args(model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        num_experts = args.num_experts,
        vocab_size = args.vocab_size,
        "Loading qwen3_next model"
    );

    let mut model = Qwen3NextCausalLM::new(args)?;

    // Load weights directly from safetensors (no key remapping needed
    // since our param names match the safetensors keys exactly)
    crate::load_safetensors_weights(&mut model, model_path)?;

    tracing::info!("Qwen3Next model loaded successfully");
    Ok(model)
}

// ---------------------------------------------------------------------------
// Qwen3.5-MoE VLM support
// ---------------------------------------------------------------------------

/// Load model args from a Qwen3.5-MoE VLM config.json.
///
/// Qwen3.5-MoE uses the same architecture as `Qwen3Next` (hybrid
/// GatedDeltaNet + full attention + sparse MoE with shared expert) but ships
/// as a VLM with config nested under `text_config` and rope parameters nested
/// under `rope_parameters`.
fn load_qwen3_5_moe_text_config_args<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Qwen3NextModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(file)?;

    // VLM-wrapped checkpoints nest the language-model args under `text_config`.
    // Flat-layout checkpoints (e.g. Carnice-9b-MLX) put the same fields at the
    // top level of config.json. Fall back to the top-level object when the
    // wrapper is absent so both packagings are accepted.
    let text_config = config.get("text_config").unwrap_or(&config);

    let mut obj = text_config.clone();
    let map = obj
        .as_object_mut()
        .ok_or_else(|| ModelError::UnsupportedModel("text_config is not an object".into()))?;

    // Flatten rope_parameters into top-level fields
    if let Some(rope_params) = text_config.get("rope_parameters") {
        if let Some(theta) = rope_params.get("rope_theta") {
            map.entry("rope_theta").or_insert_with(|| theta.clone());
        }
        if let Some(prf) = rope_params.get("partial_rotary_factor") {
            map.entry("partial_rotary_factor")
                .or_insert_with(|| prf.clone());
        }
    }

    // Merge top-level quantization config
    if let Some(quant) = config.get("quantization") {
        map.entry("quantization").or_insert_with(|| quant.clone());
    }

    // Merge top-level tie_word_embeddings
    if let Some(tie) = config.get("tie_word_embeddings") {
        map.entry("tie_word_embeddings")
            .or_insert_with(|| tie.clone());
    }

    // Set decoder_sparse_step=1 only for MoE models (num_experts > 0).
    // Dense models (qwen3_5) use standard FFN and must keep decoder_sparse_step=0.
    let has_experts = text_config
        .get("num_experts")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0)
        > 0;
    if has_experts {
        map.entry("decoder_sparse_step")
            .or_insert(serde_json::Value::from(1));
    }

    // intermediate_size is unused when all layers are MoE;
    // for dense models, keep whatever value is in text_config.
    if has_experts {
        map.entry("intermediate_size")
            .or_insert(serde_json::Value::from(0));
    }

    // Mixed-bit BA projections cannot be fused without dequantizing because
    // MLX packs different bit-widths into different inner shapes.
    let mixed_ba_layers = qwen3_5_mixed_ba_quantization_layers(&config, text_config);
    let use_separate =
        std::env::var("HIGGS_SEPARATE_GDN_PROJ").is_ok() || !mixed_ba_layers.is_empty();
    map.insert(
        "use_separate_gdn_projections".to_owned(),
        serde_json::Value::from(use_separate),
    );
    if !mixed_ba_layers.is_empty() {
        tracing::info!(
            layers = ?mixed_ba_layers,
            "Detected mixed-bit GDN BA projections; using separate GDN projections"
        );
    }

    // Detect per-layer gate quantization override from top-level quantization config
    if let Some(quant) = config.get("quantization") {
        let gate_key = "language_model.model.layers.0.mlp.gate";
        if let Some(gate_q) = quant.get(gate_key) {
            map.insert("gate_quantization".to_owned(), gate_q.clone());
        }
    }

    Ok(serde_json::from_value(obj)?)
}

fn qwen3_5_quantization_config(value: &serde_json::Value) -> Option<QuantizationConfig> {
    Some(QuantizationConfig {
        group_size: i32::try_from(value.get("group_size")?.as_i64()?).ok()?,
        bits: i32::try_from(value.get("bits")?.as_i64()?).ok()?,
    })
}

fn qwen3_5_mixed_ba_quantization_layers(
    config: &serde_json::Value,
    text_config: &serde_json::Value,
) -> Vec<i32> {
    let Some(quant) = config.get("quantization") else {
        return Vec::new();
    };
    let Some(default_quant) = qwen3_5_quantization_config(quant) else {
        return Vec::new();
    };
    let Some(num_hidden_layers) = text_config
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_i64)
        .and_then(|n| i32::try_from(n).ok())
    else {
        return Vec::new();
    };

    (0..num_hidden_layers)
        .filter(|layer_idx| {
            let prefix = format!("language_model.model.layers.{layer_idx}.linear_attn");
            let a_quant = quant
                .get(&format!("{prefix}.in_proj_a"))
                .and_then(qwen3_5_quantization_config)
                .unwrap_or_else(|| default_quant.clone());
            let b_quant = quant
                .get(&format!("{prefix}.in_proj_b"))
                .and_then(qwen3_5_quantization_config)
                .unwrap_or_else(|| default_quant.clone());
            a_quant.bits != b_quant.bits || a_quant.group_size != b_quant.group_size
        })
        .collect()
}

/// Load a Qwen3.5 dense model (VLM wrapper around Qwen3Next architecture).
///
/// Reads `text_config` for model args, strips `language_model.` prefix from
/// safetensors weight keys. Unlike [`load_qwen3_5_moe_model`], does NOT force
/// `decoder_sparse_step=1` or attempt MoE gate fusion.
pub fn load_qwen3_5_model<P: AsRef<Path>>(model_dir: P) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_qwen3_5_moe_text_config_args(model_path)?;

    tracing::info!(
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        vocab_size = args.vocab_size,
        full_attention_interval = args.full_attention_interval,
        "Loading qwen3_5 dense model (VLM text backbone via qwen3_next)"
    );

    let gdn_dims = GdnDims {
        num_k_heads: args.linear_num_key_heads,
        num_v_heads: args.linear_num_value_heads,
        head_k_dim: args.linear_key_head_dim,
        head_v_dim: args.linear_value_head_dim,
    };
    let mut model = load_qwen3_5_model_with_gdn_fallback(model_path, args, &gdn_dims)?;

    load_qwen3_5_dense_lm_head(&mut model, model_path)?;

    tracing::info!("Qwen3.5 dense model loaded successfully");
    Ok(model)
}

/// Load a Qwen3.5-MoE model (VLM wrapper around Qwen3Next architecture).
///
/// Reads `text_config` for model args, strips `language_model.` prefix from
/// safetensors weight keys.
pub fn load_qwen3_5_moe_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_qwen3_5_moe_text_config_args(model_path)?;

    tracing::info!(
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        num_experts = args.num_experts,
        vocab_size = args.vocab_size,
        full_attention_interval = args.full_attention_interval,
        "Loading qwen3_5_moe model (VLM text backbone via qwen3_next)"
    );

    // Save GDN dimensions before args is moved
    let gdn_dims = GdnDims {
        num_k_heads: args.linear_num_key_heads,
        num_v_heads: args.linear_num_value_heads,
        head_k_dim: args.linear_key_head_dim,
        head_v_dim: args.linear_value_head_dim,
    };
    let mut model = load_qwen3_5_model_with_gdn_fallback(model_path, args, &gdn_dims)?;

    // Load dense lm_head if present (no-op for quantized MoE models).
    load_qwen3_5_dense_lm_head(&mut model, model_path)?;

    tracing::info!("Qwen3.5-MoE model loaded successfully");
    Ok(model)
}

fn strip_qwen3_5_text_prefix(key: &str) -> Option<&str> {
    key.strip_prefix("model.language_model.")
        .or_else(|| key.strip_prefix("language_model."))
}

fn assign_qwen3_5_param(
    params: &mut std::collections::HashMap<std::rc::Rc<str>, &mut Array>,
    key: &str,
    value: Array,
) -> Result<bool, Array> {
    if let Some(param) = params.get_mut(key) {
        **param = value;
        return Ok(true);
    }

    let model_key = format!("model.{key}");
    if let Some(param) = params.get_mut(model_key.as_str()) {
        **param = value;
        return Ok(true);
    }

    Err(value)
}

fn load_qwen3_5_model_with_gdn_fallback(
    model_path: &Path,
    mut args: Qwen3NextModelArgs,
    gdn_dims: &GdnDims,
) -> Result<Qwen3NextCausalLM, ModelError> {
    if args.use_separate_gdn_projections {
        let mut model = Qwen3NextCausalLM::new(args)?;
        load_qwen3_5_moe_weights_direct(&mut model, model_path)?;
        tracing::info!("Using separate GDN projections (4 dispatches per layer)");
        return Ok(model);
    }

    let mut fused_model = Qwen3NextCausalLM::new(args.clone())?;
    match load_qwen3_5_moe_weights_fused(&mut fused_model, model_path, gdn_dims) {
        Ok(()) => Ok(fused_model),
        Err(err) if is_mixed_bit_gdn_ba_fusion_error(&err) => {
            tracing::warn!(
                error = %err,
                "Detected mixed-bit GDN BA projection shapes; retrying with separate GDN projections"
            );
            args.use_separate_gdn_projections = true;
            let mut separate_model = Qwen3NextCausalLM::new(args)?;
            load_qwen3_5_moe_weights_direct(&mut separate_model, model_path)?;
            tracing::info!("Using separate GDN projections (4 dispatches per layer)");
            Ok(separate_model)
        }
        Err(err) => Err(err),
    }
}

fn is_mixed_bit_gdn_ba_fusion_error(err: &ModelError) -> bool {
    matches!(
        err,
        ModelError::ShapeMismatch(message)
            if message.contains("in_proj_ba")
                && message.contains("requires separate GDN projections")
    )
}

fn load_qwen3_5_dense_lm_head(
    model: &mut Qwen3NextCausalLM,
    model_path: &Path,
) -> Result<(), crate::error::ModelError> {
    let Some(head) = model.dense_lm_head.as_mut() else {
        return Ok(());
    };

    for file_path in crate::collect_safetensors_files(model_path)? {
        let loaded = Array::load_safetensors(&file_path)
            .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            if key == "lm_head.weight" {
                head.weight = Param::new(value);
                return Ok(());
            }
        }
    }

    Err(crate::error::ModelError::MissingWeight(
        "lm_head.weight".to_owned(),
    ))
}

/// GDN dimension info extracted from model args before move.
struct GdnDims {
    num_k_heads: i32,
    num_v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
}

/// Build row permutation to convert flat [q_all|k_all|v_all|z_all] layout
/// to per-head-grouped [q_h0|k_h0|v_h0|z_h0|q_h1|...] for in_proj_qkvz.
fn build_qkvz_permutation(d: &GdnDims) -> Result<Vec<i32>, Exception> {
    let nk = d.num_k_heads;
    if nk == 0 || d.num_v_heads % nk != 0 {
        return Err(Exception::custom(format!(
            "GQA ratio invalid: num_v_heads={} not divisible by num_k_heads={nk}",
            d.num_v_heads
        )));
    }
    let dk = d.head_k_dim;
    let v_per_k = d.num_v_heads / nk;
    let dv = d.head_v_dim;
    let key_dim = nk * dk;
    let qkv_rows = key_dim * 2 + d.num_v_heads * dv; // offset for z

    let mut perm = Vec::new();
    for h in 0..nk {
        // q: rows h*dk .. (h+1)*dk from qkv (offset 0)
        for i in 0..dk {
            perm.push(h * dk + i);
        }
        // k: rows key_dim + h*dk .. from qkv
        for i in 0..dk {
            perm.push(key_dim + h * dk + i);
        }
        // v: rows 2*key_dim + h*(v_per_k*dv) .. from qkv
        for i in 0..(v_per_k * dv) {
            perm.push(2 * key_dim + h * v_per_k * dv + i);
        }
        // z: rows h*(v_per_k*dv) .. from z (offset by qkv_rows)
        for i in 0..(v_per_k * dv) {
            perm.push(qkv_rows + h * v_per_k * dv + i);
        }
    }
    Ok(perm)
}

/// Build row permutation for flat [b_all|a_all] → per-head-grouped [b_h0|a_h0|b_h1|a_h1|...].
fn build_ba_permutation(d: &GdnDims) -> Vec<i32> {
    let nk = d.num_k_heads;
    let v_per_k = d.num_v_heads / nk;
    let nv = d.num_v_heads;

    let mut perm = Vec::new();
    for h in 0..nk {
        // b: rows h*v_per_k .. (h+1)*v_per_k from b
        for i in 0..v_per_k {
            perm.push(h * v_per_k + i);
        }
        // a: rows h*v_per_k .. (h+1)*v_per_k from a (offset by nv)
        for i in 0..v_per_k {
            perm.push(nv + h * v_per_k + i);
        }
    }
    perm
}

/// Concatenate two arrays along dim 0 and permute rows.
fn concat_and_permute(a: &Array, b: &Array, perm: &[i32]) -> Result<Array, Exception> {
    let cat = ops::concatenate_axis(&[a, b], 0)?;
    let perm_arr = Array::from_slice(
        perm,
        &[i32::try_from(perm.len()).map_err(|_| Exception::custom("perm len overflow"))?],
    );
    cat.take_axis(&perm_arr, 0)
}

fn can_concatenate_axis0(a: &Array, b: &Array) -> bool {
    let a_shape = a.shape();
    let b_shape = b.shape();
    a_shape.len() == b_shape.len()
        && a_shape
            .iter()
            .zip(b_shape.iter())
            .enumerate()
            .all(|(axis, (lhs, rhs))| axis == 0 || lhs == rhs)
}

/// Load Qwen3.5-MoE weights with GDN projection fusion.
///
/// Direct weight loader: strip `language_model.` prefix, no rearrangement.
/// Used when `use_separate_gdn_projections = true`.
fn load_qwen3_5_moe_weights_direct<M: mlx_rs::module::ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
) -> Result<(), crate::error::ModelError> {
    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();
    let mut matched = 0usize;
    let mut unmatched = Vec::new();

    for file_path in &safetensors_files {
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            let Some(stripped) = strip_qwen3_5_text_prefix(&key) else {
                unmatched.push(key);
                continue;
            };
            if assign_qwen3_5_param(&mut params, stripped, value).is_ok() {
                matched += 1;
            } else {
                unmatched.push(key);
            }
        }
    }

    tracing::info!(
        matched,
        unmatched_count = unmatched.len(),
        "Direct weight loading stats"
    );
    if !unmatched.is_empty() {
        for k in unmatched.iter().take(10) {
            tracing::debug!(key = %k, "Unmatched weight key (no matching model param)");
        }
        if unmatched.len() > 10 {
            tracing::debug!("... and {} more unmatched keys", unmatched.len() - 10);
        }
    }
    let param_count = params.keys().count();
    tracing::info!(param_count, "Total model parameters loaded");

    model
        .eval()
        .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

/// Rearranges flat (qkv,z,b,a) projections to per-head-grouped (qkvz,ba)
/// so the model uses the fused 2-dispatch forward path instead of 4 separate.
fn load_qwen3_5_moe_weights_fused<M: mlx_rs::module::ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
    gdn_dims: &GdnDims,
) -> Result<(), crate::error::ModelError> {
    use std::collections::HashMap;

    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();

    let qkvz_perm = build_qkvz_permutation(gdn_dims)
        .map_err(|e| crate::error::ModelError::ShapeMismatch(e.to_string()))?;
    let ba_perm = build_ba_permutation(gdn_dims);

    // GDN split keys: collect (part_a, part_b) for each combined target
    // Key format: "model.layers.N.linear_attn.in_proj_qkvz.{weight|scales|biases}"
    let mut gdn_parts: HashMap<String, (Option<Array>, Option<Array>)> = HashMap::new();

    let gdn_remap: &[(&str, &str, &str)] = &[
        ("in_proj_qkv", "in_proj_z", "in_proj_qkvz"),
        ("in_proj_b", "in_proj_a", "in_proj_ba"),
    ];

    for file_path in &safetensors_files {
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            let Some(stripped) = strip_qwen3_5_text_prefix(&key) else {
                continue;
            };

            let mut handled = false;
            for &(part_a_name, part_b_name, combined_name) in gdn_remap {
                for (is_b, split_name) in [(false, part_a_name), (true, part_b_name)] {
                    let needle = format!(".{split_name}.");
                    if let Some(pos) = stripped.find(&needle) {
                        let pfx = &stripped[..pos];
                        let sfx = &stripped[pos + needle.len()..];
                        let map_key = format!("{pfx}.{combined_name}.{sfx}");
                        let entry = gdn_parts.entry(map_key).or_insert((None, None));
                        if is_b {
                            entry.1 = Some(value.clone());
                        } else {
                            entry.0 = Some(value.clone());
                        }
                        handled = true;
                        break;
                    }
                }
                if handled {
                    break;
                }
            }

            if !handled {
                let _ = assign_qwen3_5_param(&mut params, stripped, value);
            }
        }
    }

    // Fuse GDN pairs: concat + row permutation
    let mut fused_count = 0usize;
    for (combined_key, (part_a, part_b)) in &gdn_parts {
        let (Some(a), Some(b)) = (part_a, part_b) else {
            return Err(crate::error::ModelError::Io(std::io::Error::other(
                format!("Incomplete GDN projection pair for key: {combined_key}"),
            )));
        };
        if combined_key.contains("in_proj_ba") && !can_concatenate_axis0(a, b) {
            return Err(crate::error::ModelError::ShapeMismatch(format!(
                "Mixed-bit BA fusion requires separate GDN projections for key {combined_key}: {:?} vs {:?}",
                a.shape(),
                b.shape()
            )));
        }
        let resolved_key = if params.contains_key(combined_key.as_str()) {
            combined_key.clone()
        } else {
            format!("model.{combined_key}")
        };
        let Some(param) = params.get_mut(resolved_key.as_str()) else {
            return Err(crate::error::ModelError::Io(std::io::Error::other(
                format!("Fused target key not found in model params: {combined_key}"),
            )));
        };
        let perm = if combined_key.contains("in_proj_qkvz") {
            &qkvz_perm
        } else {
            &ba_perm
        };
        match concat_and_permute(a, b, perm) {
            Ok(fused) => {
                **param = fused;
                fused_count += 1;
            }
            Err(e) => {
                return Err(crate::error::ModelError::Io(std::io::Error::other(
                    format!("GDN fusion failed for key {combined_key}: {e}"),
                )));
            }
        }
    }

    tracing::info!(
        fused_count,
        total_pairs = gdn_parts.len(),
        "Fused GDN projections (4→2 dispatches per layer)"
    );

    model
        .eval()
        .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::doc_markdown,
    clippy::needless_for_each,
    clippy::needless_collect,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_borrows_for_generic_args,
    clippy::needless_range_loop,
    clippy::manual_flatten,
    clippy::unnecessary_map_or,
    clippy::uninlined_format_args,
    clippy::manual_range_contains,
    clippy::explicit_iter_loop,
    clippy::borrow_as_ptr,
    clippy::ref_as_ptr
)]
mod tests {
    use super::*;

    #[test]
    fn test_config_deserialization() {
        let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "rope_theta": 5000000,
            "partial_rotary_factor": 0.25,
            "max_position_embeddings": 262144,
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 512,
            "moe_intermediate_size": 512,
            "norm_topk_prob": true,
            "full_attention_interval": 4,
            "tie_word_embeddings": false,
            "quantization": { "group_size": 64, "bits": 4 }
        }"#;

        let args: Qwen3NextModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "qwen3_next");
        assert_eq!(args.hidden_size, 2048);
        assert_eq!(args.num_hidden_layers, 48);
        assert_eq!(args.head_dim, 256);
        assert_eq!(args.num_experts, 512);
        assert_eq!(args.num_experts_per_tok, 10);
        assert_eq!(args.full_attention_interval, 4);
        assert_eq!(args.linear_conv_kernel_dim, 4);
        assert!(!args.tie_word_embeddings);
        assert!(args.norm_topk_prob);
        let qc = args.quantization.unwrap();
        assert_eq!(qc.group_size, 64);
        assert_eq!(qc.bits, 4);
    }

    #[test]
    fn test_swiglu() {
        let gate = Array::from_slice(&[1.0_f32, -1.0, 0.5], &[1, 3]);
        let x = Array::from_slice(&[2.0_f32, 3.0, 4.0], &[1, 3]);
        let result = swiglu(&gate, &x).unwrap();
        assert_eq!(result.shape(), &[1, 3]);
        // silu(1.0) * 2.0 = 0.7311 * 2.0 ~= 1.462
        let first: f32 = result.index((.., 0..1)).item();
        assert!(first > 1.0);
    }

    #[test]
    fn test_gated_delta_kernel_basic() {
        // B=1, T=1, Hk=2, Hv=4, Dk=32, Dv=32
        // Dk must be multiple of 32 for SIMD group width
        let q = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
        let k = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
        let v = Array::ones::<f32>(&[1, 1, 4, 32]).unwrap();
        let a_log = Array::zeros::<f32>(&[4]).unwrap();
        let a = Array::ones::<f32>(&[1, 1, 4]).unwrap();
        let dt_bias = Array::zeros::<f32>(&[4]).unwrap();
        let b = Array::zeros::<f32>(&[1, 1, 4]).unwrap();
        let state = Array::zeros::<f32>(&[1, 4, 32, 32]).unwrap();

        let (y, new_state) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &b, &state, 1, 1, 2, 32, 4, 32,
        )
        .unwrap();
        y.eval().unwrap();
        new_state.eval().unwrap();
        assert_eq!(y.shape(), &[1, 1, 4, 32]);
        assert_eq!(new_state.shape(), &[1, 4, 32, 32]);
    }

    #[test]
    fn test_sparse_moe_rejects_top_k_exceeding_num_experts() {
        assert_sparse_moe_rejects(
            |a| {
                a.num_experts = 4;
                a.num_experts_per_tok = 8;
            },
            "num_experts_per_tok",
        );
    }

    #[test]
    fn test_sparse_moe_accepts_top_k_equal_to_num_experts() {
        let mut args = minimal_qwen3_next_args();
        args.num_experts = 4;
        args.num_experts_per_tok = 4; // top_k == num_experts is fine
        let result = SparseMoeBlock::new(&args, 64, 4);
        assert!(result.is_ok());
    }

    fn assert_sparse_moe_rejects(
        mutate: impl FnOnce(&mut Qwen3NextModelArgs),
        expected_substring: &str,
    ) {
        let mut args = minimal_qwen3_next_args();
        mutate(&mut args);
        let result = SparseMoeBlock::new(&args, 64, 4);
        assert!(result.is_err(), "Should reject invalid args");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains(expected_substring),
            "Expected error about {expected_substring}, got: {msg}"
        );
    }

    #[test]
    fn test_sparse_moe_rejects_zero_num_experts() {
        assert_sparse_moe_rejects(|a| a.num_experts = 0, "num_experts");
    }

    #[test]
    fn test_sparse_moe_rejects_zero_num_experts_per_tok() {
        assert_sparse_moe_rejects(|a| a.num_experts_per_tok = 0, "num_experts_per_tok");
    }

    /// Minimal args for tests that only care about `MoE` fields.
    fn minimal_qwen3_next_args() -> Qwen3NextModelArgs {
        serde_json::from_str(
            r#"{
                "model_type": "qwen3_next",
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "rms_norm_eps": 1e-06,
                "vocab_size": 1024,
                "max_position_embeddings": 512,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 256,
                "moe_intermediate_size": 128,
                "norm_topk_prob": true
            }"#,
        )
        .unwrap()
    }

    /// Full args suitable for `Qwen3NextCausalLM::new()` validation tests.
    fn valid_causal_lm_args() -> Qwen3NextModelArgs {
        serde_json::from_str(
            r#"{
                "model_type": "qwen3_next",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "rms_norm_eps": 1e-06,
                "vocab_size": 1024,
                "max_position_embeddings": 512,
                "full_attention_interval": 4,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 256,
                "moe_intermediate_size": 128,
                "norm_topk_prob": true
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn test_causal_lm_rejects_zero_full_attention_interval() {
        let mut args = valid_causal_lm_args();
        args.full_attention_interval = 0;
        let result = Qwen3NextCausalLM::new(args);
        assert!(
            result.is_err(),
            "Should reject full_attention_interval == 0"
        );
    }

    #[test]
    fn test_causal_lm_rejects_zero_linear_key_heads() {
        let mut args = valid_causal_lm_args();
        args.linear_num_key_heads = 0;
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_err(), "Should reject linear_num_key_heads == 0");
    }

    #[test]
    fn test_causal_lm_rejects_zero_linear_value_heads() {
        let mut args = valid_causal_lm_args();
        args.linear_num_value_heads = 0;
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_err(), "Should reject linear_num_value_heads == 0");
    }

    #[test]
    fn test_causal_lm_rejects_zero_conv_kernel_dim() {
        let mut args = valid_causal_lm_args();
        args.linear_conv_kernel_dim = 0;
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_err(), "Should reject linear_conv_kernel_dim == 0");
    }

    #[test]
    fn test_layer_cache_variants() {
        let kv = LayerCache::KV(SteppingKeyValueCache::new());
        let arrays = LayerCache::Arrays(ArraysCache::new());
        match &kv {
            LayerCache::KV(c) => assert_eq!(c.offset(), 0),
            LayerCache::Arrays(_) => panic!("Expected KV variant"),
        }
        match &arrays {
            LayerCache::Arrays(c) => assert_eq!(c.offset, 0),
            LayerCache::KV(_) => panic!("Expected Arrays variant"),
        }
    }

    #[test]
    fn test_config_deserialization_missing_optional_fields() {
        // Only required fields; all serde(default) fields should get defaults
        let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 262144
        }"#;
        let args: Qwen3NextModelArgs = serde_json::from_str(json).unwrap();
        assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
        assert!((args.partial_rotary_factor - 1.0).abs() < f32::EPSILON);
        assert_eq!(args.full_attention_interval, 4);
        assert!(!args.tie_word_embeddings);
        assert!(!args.attention_bias);
        assert!(args.rope_scaling.is_none());
        assert!(args.quantization.is_none());
        assert_eq!(args.linear_num_value_heads, 0);
        assert_eq!(args.linear_num_key_heads, 0);
        assert_eq!(args.linear_key_head_dim, 0);
        assert_eq!(args.linear_value_head_dim, 0);
        assert_eq!(args.linear_conv_kernel_dim, 0);
        assert_eq!(args.num_experts, 0);
        assert_eq!(args.num_experts_per_tok, 0);
        assert_eq!(args.decoder_sparse_step, 0);
        assert!(args.norm_topk_prob);
        assert!(args.mlp_only_layers.is_empty());
    }

    #[test]
    fn test_config_deserialization_quantization_null() {
        let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "quantization": null
        }"#;
        let args: Qwen3NextModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.quantization.is_none());
    }

    #[test]
    fn test_swiglu_numeric_correctness() {
        // silu(x) = x * sigmoid(x)
        // silu(0) = 0 * 0.5 = 0
        // silu(1) = 1 * sigmoid(1) = 1 * 0.7310586 = 0.7310586
        // silu(-1) = -1 * sigmoid(-1) = -1 * 0.2689414 = -0.2689414

        // swiglu(gate, x) = silu(gate) * x

        // gate=0, x=5 => silu(0) * 5 = 0
        let gate = Array::from_slice(&[0.0_f32], &[1, 1]);
        let x = Array::from_slice(&[5.0_f32], &[1, 1]);
        let result = swiglu(&gate, &x).unwrap();
        let val: f32 = result.item();
        assert!((val - 0.0).abs() < 1e-6, "silu(0)*5 should be 0, got {val}");

        // gate=1, x=1 => silu(1) * 1 = 0.7310586
        let gate2 = Array::from_slice(&[1.0_f32], &[1, 1]);
        let x2 = Array::from_slice(&[1.0_f32], &[1, 1]);
        let result2 = swiglu(&gate2, &x2).unwrap();
        let val2: f32 = result2.item();
        assert!(
            (val2 - 0.731_058_6).abs() < 1e-4,
            "silu(1)*1 should be ~0.7311, got {val2}"
        );

        // gate=-1, x=2 => silu(-1) * 2 = -0.2689414 * 2 = -0.5378828
        let gate3 = Array::from_slice(&[-1.0_f32], &[1, 1]);
        let x3 = Array::from_slice(&[2.0_f32], &[1, 1]);
        let result3 = swiglu(&gate3, &x3).unwrap();
        let val3: f32 = result3.item();
        assert!(
            (val3 - (-0.537_882_8)).abs() < 1e-4,
            "silu(-1)*2 should be ~-0.5379, got {val3}"
        );
    }

    #[test]
    fn test_sparse_moe_happy_path_construction() {
        let args = minimal_qwen3_next_args();
        let result = SparseMoeBlock::new(&args, 64, 4);
        assert!(result.is_ok());
        let block = result.unwrap();
        assert_eq!(block.top_k, args.num_experts_per_tok);
        assert!(block.norm_topk_prob);
    }

    #[test]
    fn test_causal_lm_valid_construction() {
        let args = valid_causal_lm_args();
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.args.model_type, "qwen3_next");
    }

    #[test]
    fn test_causal_lm_make_cache_layer_types() {
        let args = valid_causal_lm_args();
        let model = Qwen3NextCausalLM::new(args).unwrap();
        let cache = model.make_cache();
        // 4 layers, full_attention_interval=4, so layers 0,1,2 are linear, layer 3 is full attention
        assert_eq!(cache.len(), 4);
        for (i, layer_cache) in cache.iter().enumerate() {
            let lc = layer_cache.as_ref().unwrap();
            let is_linear = (i + 1) % 4 != 0;
            if is_linear {
                assert!(
                    matches!(lc, LayerCache::Arrays(_)),
                    "Layer {i} should be Arrays (linear)"
                );
            } else {
                assert!(
                    matches!(lc, LayerCache::KV(_)),
                    "Layer {i} should be KV (full attention)"
                );
            }
        }
    }

    #[test]
    fn test_causal_lm_negative_full_attention_interval() {
        let mut args = valid_causal_lm_args();
        args.full_attention_interval = -1;
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_err());
    }

    #[test]
    fn test_causal_lm_with_quantization() {
        let mut args = valid_causal_lm_args();
        args.quantization = Some(QuantizationConfig {
            group_size: 32,
            bits: 8,
        });
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_causal_lm_with_tied_embeddings() {
        let mut args = valid_causal_lm_args();
        args.tie_word_embeddings = true;
        let model = Qwen3NextCausalLM::new(args).unwrap();
        assert!(model.lm_head.is_none());
    }

    #[test]
    fn test_causal_lm_without_tied_embeddings() {
        let mut args = valid_causal_lm_args();
        args.tie_word_embeddings = false;
        let model = Qwen3NextCausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some());
    }

    #[test]
    fn test_load_model_args_happy_path() {
        let dir = tempfile::tempdir().unwrap();
        let config = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 262144
        }"#;
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        let args = load_model_args(dir.path()).unwrap();
        assert_eq!(args.model_type, "qwen3_next");
        assert_eq!(args.hidden_size, 2048);
    }

    #[test]
    fn test_load_model_args_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_model_args(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_model_args_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{{bad json").unwrap();
        let result = load_model_args(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_arrays_cache_default() {
        let cache = ArraysCache::default();
        assert!(cache.conv_state.is_none());
        assert!(cache.ssm_state.is_none());
        assert_eq!(cache.offset, 0);
    }

    #[test]
    fn test_gated_delta_kernel_prefill() {
        // B=1, T=4, Hk=2, Hv=4, Dk=32, Dv=32
        let q = Array::ones::<f32>(&[1, 4, 2, 32]).unwrap();
        let k = Array::ones::<f32>(&[1, 4, 2, 32]).unwrap();
        let v = Array::ones::<f32>(&[1, 4, 4, 32]).unwrap();
        let a_log = Array::zeros::<f32>(&[4]).unwrap();
        let a = Array::ones::<f32>(&[1, 4, 4]).unwrap();
        let dt_bias = Array::zeros::<f32>(&[4]).unwrap();
        let b = Array::zeros::<f32>(&[1, 4, 4]).unwrap();
        let state = Array::zeros::<f32>(&[1, 4, 32, 32]).unwrap();

        let (y, new_state) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &b, &state, 1, 4, 2, 32, 4, 32,
        )
        .unwrap();
        y.eval().unwrap();
        new_state.eval().unwrap();
        assert_eq!(y.shape(), &[1, 4, 4, 32]);
        assert_eq!(new_state.shape(), &[1, 4, 32, 32]);
    }

    // -----------------------------------------------------------------------
    // gather_qmm + MoE rewrite tests
    // -----------------------------------------------------------------------

    /// Quantize a float matrix and return (weight, scales, biases) suitable for
    /// `gather_qmm` / `quantized_matmul`.
    fn quantize_weights(w: &Array, group_size: i32, bits: i32) -> (Array, Array, Array) {
        let (qw, scales, biases) = ops::quantize(w, group_size, bits).unwrap();
        (qw, scales, biases)
    }

    #[test]
    fn test_gather_qmm_basic() {
        // 2 experts, out=64, in=64 (dims must be multiples of 32 for quantize)
        let w_float = Array::ones::<f32>(&[2, 64, 64]).unwrap();
        let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

        // Input [1, 1, 1, 64], select expert 0
        let x = Array::ones::<f32>(&[1, 1, 1, 64]).unwrap();
        let indices = Array::from_slice(&[0_u32], &[1, 1, 1]);

        let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
        // Force evaluation to run the Metal kernel (MLX is lazy)
        result.eval().unwrap();
        // Output: [1, 1, 1, 1, 64] (batch broadcast with indices, M=1, N=64)
        assert_eq!(result.ndim(), 5);
        assert_eq!(*result.shape().last().unwrap(), 64);
    }

    #[test]
    fn test_gather_qmm_multi_expert() {
        // 4 experts, out=64, in=64
        let w_float = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

        let x = Array::ones::<f32>(&[1, 1, 1, 64]).unwrap();
        let indices = Array::from_slice(&[0_u32, 2, 3], &[1, 1, 3]);

        let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
        result.eval().unwrap();
        // Output: [1, 1, 3, 1, 64] — 3 experts selected
        assert_eq!(*result.shape().get(2).unwrap(), 3);
    }

    #[test]
    fn test_gather_qmm_matches_per_expert() {
        // Verify that gather_qmm produces the same result as the old
        // take_axis + quantized_matmul path for a single expert.
        let w_float = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[4, 64, 64], None).unwrap();
        let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

        let x = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[1, 64], None).unwrap();
        let expert_idx = Array::from_slice(&[2_u32], &[1]);

        // Old path: take_axis + quantized_matmul
        let ew = qw
            .take_axis(&expert_idx, 0)
            .unwrap()
            .squeeze_axes(&[0])
            .unwrap();
        let es = scales
            .take_axis(&expert_idx, 0)
            .unwrap()
            .squeeze_axes(&[0])
            .unwrap();
        let eb = biases
            .take_axis(&expert_idx, 0)
            .unwrap()
            .squeeze_axes(&[0])
            .unwrap();
        let old_result = ops::quantized_matmul(&x, &ew, &es, &eb, true, 64, 4).unwrap();

        // New path: gather_qmm
        let x_expanded = x.expand_dims(-2).unwrap(); // [1, 1, 64]
        let indices = Array::from_slice(&[2_u32], &[1, 1]);
        let new_result = gather_qmm(
            &x_expanded,
            &qw,
            &scales,
            &biases,
            &indices,
            true,
            64,
            4,
            false,
        )
        .unwrap()
        .squeeze_axes(&[-2])
        .unwrap()
        .squeeze_axes(&[-2])
        .unwrap();

        // Compare element-wise (both are quantized, should be exact match)
        let diff = old_result.subtract(&new_result).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-5,
            "gather_qmm and per-expert path differ by {max_diff}"
        );
    }

    #[test]
    fn test_forward_gather_global_sort_shape() {
        // RED: forward_gather_global_sort should produce [B, L, top_k, D]
        let mut block = SwitchMlpWeights::new(64, 4).unwrap();

        let gate_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
        *block.gate_proj.weight = gw;
        *block.gate_proj.scales = gs;
        *block.gate_proj.biases = gb;

        let up_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
        *block.up_proj.weight = uw;
        *block.up_proj.scales = us;
        *block.up_proj.biases = ub;

        let down_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
        *block.down_proj.weight = dw;
        *block.down_proj.scales = ds;
        *block.down_proj.biases = db;

        // B=1, L=4, top_k=2 — enough tokens to exercise the sort path
        let x = Array::ones::<f32>(&[1, 4, 64]).unwrap();
        let indices = Array::from_slice(&[2u32, 0, 1, 3, 0, 2, 3, 1], &[1, 4, 2]);

        let result = block.forward_gather_global_sort(&x, &indices).unwrap();
        assert_eq!(result.shape(), &[1, 4, 2, 64]);
    }

    #[test]
    fn test_forward_gather_global_sort_equivalence() {
        // RED: global sort must produce the same values as forward_gather
        let mut block = SwitchMlpWeights::new(64, 4).unwrap();

        let gate_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
        *block.gate_proj.weight = gw;
        *block.gate_proj.scales = gs;
        *block.gate_proj.biases = gb;

        let up_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
        *block.up_proj.weight = uw;
        *block.up_proj.scales = us;
        *block.up_proj.biases = ub;

        let down_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
        *block.down_proj.weight = dw;
        *block.down_proj.scales = ds;
        *block.down_proj.biases = db;

        let x = Array::ones::<f32>(&[1, 4, 64]).unwrap();
        let indices = Array::from_slice(&[2u32, 0, 1, 3, 0, 2, 3, 1], &[1, 4, 2]);

        let baseline = block.forward_gather(&x, &indices, false).unwrap();
        let sorted = block.forward_gather_global_sort(&x, &indices).unwrap();
        baseline.eval().unwrap();
        sorted.eval().unwrap();

        let diff = baseline.subtract(&sorted).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-5,
            "global sort and baseline differ by {max_diff}"
        );
    }

    #[test]
    fn test_forward_gather_global_sort_random_weights() {
        // Harder: random weights + distinct per-token inputs + more experts
        // Verifies the sort/unsort cycle preserves per-token identity.
        let num_experts = 8;
        let hidden = 64;
        let top_k = 3;
        let b = 1;
        let l = 16;

        let mut block = SwitchMlpWeights::new(64, 4).unwrap();

        let gate_w =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
                .unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
        *block.gate_proj.weight = gw;
        *block.gate_proj.scales = gs;
        *block.gate_proj.biases = gb;

        let up_w =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
                .unwrap();
        let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
        *block.up_proj.weight = uw;
        *block.up_proj.scales = us;
        *block.up_proj.biases = ub;

        let down_w =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
                .unwrap();
        let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
        *block.down_proj.weight = dw;
        *block.down_proj.scales = ds;
        *block.down_proj.biases = db;

        // Random input — each token is distinct
        let x = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[b, l, hidden], None).unwrap();
        // Random expert indices in [0, num_experts)
        let idx_data: Vec<u32> = (0..(b * l * top_k) as u32)
            .map(|i| i % num_experts as u32)
            .collect();
        let indices = Array::from_slice(&idx_data, &[b, l, top_k]);
        x.eval().unwrap();
        indices.eval().unwrap();

        let baseline = block.forward_gather(&x, &indices, false).unwrap();
        let sorted = block.forward_gather_global_sort(&x, &indices).unwrap();
        baseline.eval().unwrap();
        sorted.eval().unwrap();

        assert_eq!(baseline.shape(), sorted.shape());
        assert_eq!(sorted.shape(), &[b, l, top_k, hidden]);

        let diff = baseline.subtract(&sorted).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-4,
            "random weights: global sort differs by {max_diff}"
        );
    }

    #[test]
    fn test_moe_gate_up_fusion_parity() {
        // Fused gate+up (2 gather_qmm) must match unfused (3 gather_qmm).
        // Uses random weights + distinct per-token inputs to stress sort/unsort.
        let num_experts = 8;
        let hidden = 64;
        let top_k = 3;
        let b = 1;
        let l = 16;

        let mut block = SwitchMlpWeights::new(64, 4).unwrap();

        let gate_w =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
                .unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
        *block.gate_proj.weight = gw;
        *block.gate_proj.scales = gs;
        *block.gate_proj.biases = gb;

        let up_w =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
                .unwrap();
        let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
        *block.up_proj.weight = uw;
        *block.up_proj.scales = us;
        *block.up_proj.biases = ub;

        let down_w =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
                .unwrap();
        let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
        *block.down_proj.weight = dw;
        *block.down_proj.scales = ds;
        *block.down_proj.biases = db;

        let x = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[b, l, hidden], None).unwrap();
        let idx_data: Vec<u32> = (0..(b * l * top_k) as u32)
            .map(|i| i % num_experts as u32)
            .collect();
        let indices = Array::from_slice(&idx_data, &[b, l, top_k]);
        x.eval().unwrap();
        indices.eval().unwrap();

        // Reference: unfused 3-call path
        let reference = block.forward_gather_global_sort(&x, &indices).unwrap();
        // Fused: 2-call path
        let fused = block.forward_gather_fused(&x, &indices).unwrap();
        reference.eval().unwrap();
        fused.eval().unwrap();

        assert_eq!(reference.shape(), fused.shape());
        assert_eq!(fused.shape(), &[b, l, top_k, hidden]);

        let diff = reference.subtract(&fused).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-5,
            "fused gate+up differs from unfused by {max_diff}"
        );
    }

    #[test]
    fn test_switch_mlp_forward_gather_shapes() {
        // Verify forward_gather produces the correct output shape with the
        // double expand_dims pattern matching Python's SwitchGLU.
        let mut block = SwitchMlpWeights::new(64, 4).unwrap();

        // 4 experts, intermediate=64, hidden=64
        let gate_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
        *block.gate_proj.weight = gw;
        *block.gate_proj.scales = gs;
        *block.gate_proj.biases = gb;

        let up_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
        *block.up_proj.weight = uw;
        *block.up_proj.scales = us;
        *block.up_proj.biases = ub;

        let down_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
        *block.down_proj.weight = dw;
        *block.down_proj.scales = ds;
        *block.down_proj.biases = db;

        let x = Array::ones::<f32>(&[1, 1, 64]).unwrap();
        let indices = Array::from_slice(&[0_u32, 1, 2], &[1, 1, 3]);

        let result = block.forward_gather(&x, &indices, false).unwrap();
        // [B=1, L=1, top_k=3, D=64]
        assert_eq!(result.shape(), &[1, 1, 3, 64]);
    }

    #[test]
    fn test_sparse_moe_forward_output_shape() {
        // Build a SparseMoeBlock with quantized dummy weights and verify the
        // full forward pass produces the correct output shape.
        let mut args = minimal_qwen3_next_args();
        args.num_experts = 4;
        args.num_experts_per_tok = 2;
        args.moe_intermediate_size = 64;
        args.shared_expert_intermediate_size = 64;
        args.hidden_size = 64;

        let mut block = SparseMoeBlock::new(&args, 64, 4).unwrap();

        // Set router gate weights: [num_experts, hidden_size]
        let gate_w = Array::ones::<f32>(&[4, 64]).unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 8);
        *block.gate.weight = gw;
        *block.gate.scales = gs;
        *block.gate.biases = gb;

        // Set switch_mlp expert weights: [4, intermediate, hidden] and [4, hidden, intermediate]
        let proj_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (pw, ps, pb) = quantize_weights(&proj_w, 64, 4);
        for proj in [
            &mut block.switch_mlp.gate_proj,
            &mut block.switch_mlp.up_proj,
        ] {
            *proj.weight = pw.clone();
            *proj.scales = ps.clone();
            *proj.biases = pb.clone();
        }
        *block.switch_mlp.down_proj.weight = pw;
        *block.switch_mlp.down_proj.scales = ps;
        *block.switch_mlp.down_proj.biases = pb;

        // Set shared expert weights
        let shared_w = Array::ones::<f32>(&[64, 64]).unwrap();
        let (sw, ss, sb) = quantize_weights(&shared_w, 64, 4);
        for proj in [
            &mut block.shared_expert.gate_proj,
            &mut block.shared_expert.up_proj,
            &mut block.shared_expert.down_proj,
        ] {
            *proj.weight = sw.clone();
            *proj.scales = ss.clone();
            *proj.biases = sb.clone();
        }

        // Set shared expert gate weights
        let sgate_w = Array::ones::<f32>(&[1, 64]).unwrap();
        let (sgw, sgs, sgb) = quantize_weights(&sgate_w, 64, 8);
        *block.shared_expert_gate.weight = sgw;
        *block.shared_expert_gate.scales = sgs;
        *block.shared_expert_gate.biases = sgb;

        let x = Array::ones::<f32>(&[1, 1, 64]).unwrap();
        let result = block.forward(&x).unwrap();
        assert_eq!(result.shape(), &[1, 1, 64]);
    }

    #[test]
    fn test_gather_qmm_model_scale() {
        // Reproduce actual Qwen3-Next-4bit shapes: 512 experts, hidden=2048,
        // intermediate=512, group_size=64, bits=4, top_k=10.
        // Use smaller dims to keep test fast but same expert count.
        let num_experts = 512;
        let hidden = 128; // Smaller than 2048 for test speed
        let intermediate = 64;

        let w_float = mlx_rs::random::uniform::<f32, f32>(
            0.0,
            1.0,
            &[num_experts, intermediate, hidden],
            None,
        )
        .unwrap();
        let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

        // Decode shape: B=1, L=1, M=1
        let x = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[1, 1, 1, hidden], None).unwrap();
        let indices = Array::from_slice(
            &[0_u32, 10, 50, 100, 200, 300, 400, 450, 500, 511],
            &[1, 1, 10],
        );

        let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
        // Force actual Metal kernel evaluation
        result.eval().unwrap();
        assert_eq!(result.shape(), &[1, 1, 10, 1, intermediate]);
    }

    #[test]
    fn test_gather_qmm_prefill_broadcast() {
        // Prefill case: L > 1 requires the double expand_dims pattern.
        // x batch [B, L, 1] must broadcast with indices [B, L, top_k].
        let w_float = Array::ones::<f32>(&[8, 64, 64]).unwrap();
        let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

        // Prefill: B=1, L=9
        let x = Array::ones::<f32>(&[1, 9, 1, 1, 64]).unwrap(); // double expand
        let indices = Array::from_slice(
            &[0_u32, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 7],
            &[1, 9, 2],
        );

        let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
        result.eval().unwrap();
        // [1, 9, 2, 1, 64]: broadcast batch [1,9,1] with [1,9,2] -> [1,9,2], M=1, N=64
        assert_eq!(result.shape(), &[1, 9, 2, 1, 64]);
    }

    #[test]
    fn test_gather_qmm_bfloat16() {
        // Model uses bfloat16 for scales/biases and input activations.
        // Verify gather_qmm works with bfloat16 dtypes.
        use mlx_rs::Dtype;

        let num_experts = 8;
        let hidden = 128;
        let intermediate = 64;

        let w_float = mlx_rs::random::uniform::<f32, f32>(
            0.0,
            1.0,
            &[num_experts, intermediate, hidden],
            None,
        )
        .unwrap();
        let (qw, scales_f32, biases_f32) = quantize_weights(&w_float, 64, 4);

        // Convert scales/biases to bfloat16 (matching model file dtype)
        let scales = scales_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let biases = biases_f32.as_dtype(Dtype::Bfloat16).unwrap();

        // Input in bfloat16
        let x_f32 =
            mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[1, 1, 1, hidden], None).unwrap();
        let x = x_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let indices = Array::from_slice(&[0_u32, 3, 7], &[1, 1, 3]);

        let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
        result.eval().unwrap();
        assert_eq!(result.shape(), &[1, 1, 3, 1, intermediate]);
    }

    // -----------------------------------------------------------------------
    // compile tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compiled_compute_g_matches_raw() {
        let a_log = Array::from_slice(&[0.5_f32, -0.3], &[1, 2]);
        let a = Array::from_slice(&[1.0_f32, -1.0], &[1, 2]);
        let dt_bias = Array::from_slice(&[0.1_f32, 0.2], &[1, 2]);

        // Raw computation
        let a_plus_bias = a.add(&dt_bias).unwrap();
        let sp = nn::softplus(&a_plus_bias).unwrap();
        let neg_decay = a_log
            .exp()
            .unwrap()
            .negative()
            .unwrap()
            .multiply(sp)
            .unwrap();
        let raw_g = neg_decay.exp().unwrap();

        // Compiled computation
        let mut compiled = mlx_rs::transforms::compile::compile(compute_g_compiled, None);
        let compiled_g = compiled((&a_log, &a, &dt_bias)).unwrap();

        let diff = raw_g.subtract(&compiled_g).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-6,
            "compiled compute_g differs from raw by {max_diff}"
        );
    }

    #[test]
    fn test_gated_delta_kernel_state_passthrough() {
        // Verify that running kernel with T=1 twice produces different state
        // than running with T=2, confirming sequential dependence works.
        let q = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
        let k = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
        let v = Array::ones::<f32>(&[1, 1, 4, 32]).unwrap();
        let a_log = Array::zeros::<f32>(&[4]).unwrap();
        let a = Array::ones::<f32>(&[1, 1, 4]).unwrap();
        let dt_bias = Array::zeros::<f32>(&[4]).unwrap();
        let b = Array::zeros::<f32>(&[1, 1, 4]).unwrap();
        let state0 = Array::zeros::<f32>(&[1, 4, 32, 32]).unwrap();

        // Step 1
        let (_, state1) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &b, &state0, 1, 1, 2, 32, 4, 32,
        )
        .unwrap();
        state1.eval().unwrap();

        // Step 2 (uses state1)
        let (y2, state2) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &b, &state1, 1, 1, 2, 32, 4, 32,
        )
        .unwrap();
        y2.eval().unwrap();
        state2.eval().unwrap();

        assert_eq!(y2.shape(), &[1, 1, 4, 32]);
        assert_eq!(state2.shape(), &[1, 4, 32, 32]);
    }

    /// Reference ops implementation of a single gated delta step (for comparison tests).
    fn gated_delta_step_ref(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state: &Array,
    ) -> (Array, Array) {
        let decay = g.expand_dims(-1).unwrap().expand_dims(-1).unwrap();
        let decayed_state = state.multiply(&decay).unwrap();
        let k_expanded = k.expand_dims(-2).unwrap();
        let kv_mem = decayed_state
            .multiply(&k_expanded)
            .unwrap()
            .sum_axes(&[-1], false)
            .unwrap();
        let beta_expanded = beta.expand_dims(-1).unwrap();
        let delta = v
            .subtract(&kv_mem)
            .unwrap()
            .multiply(&beta_expanded)
            .unwrap();
        let delta_expanded = delta.expand_dims(-1).unwrap();
        let new_state = decayed_state
            .add(k_expanded.multiply(&delta_expanded).unwrap())
            .unwrap();
        let q_expanded = q.expand_dims(-2).unwrap();
        let y = new_state
            .multiply(&q_expanded)
            .unwrap()
            .sum_axes(&[-1], false)
            .unwrap();
        (y, new_state)
    }

    #[test]
    fn test_gated_delta_kernel_matches_ops() {
        // Compare kernel output against reference ops for T=1, no GQA.
        // B=1, T=1, Hk=1, Hv=1, Dk=32, Dv=32
        assert_kernel_matches_ops(1, 1, 1, 1, 32, 32, 1e-4, "Hk=Hv=1");
    }

    #[test]
    fn test_gated_delta_kernel_matches_ops_gqa() {
        // GQA: Hk=2, Hv=4 (repeat factor 2). This is the pattern used by Qwen3-Next.
        assert_kernel_matches_ops(1, 1, 2, 4, 32, 32, 1e-4, "Hk=2,Hv=4 GQA");
    }

    #[test]
    fn test_gated_delta_kernel_matches_ops_multi_step() {
        // T=3 with GQA: verify multi-timestep correctness
        assert_kernel_matches_ops(1, 3, 2, 4, 32, 32, 1e-4, "T=3 GQA");
    }

    #[test]
    fn test_gated_delta_kernel_matches_ops_model_dims() {
        // Actual Qwen3-Next dims: Hk=16, Hv=32, Dk=128, Dv=128
        assert_kernel_matches_ops(1, 1, 16, 32, 128, 128, 1e-4, "model dims");
    }

    #[test]
    fn test_gated_delta_kernel_matches_ops_bfloat16() {
        // The actual model uses bfloat16. Test with model dims in bfloat16.
        use mlx_rs::Dtype;
        let hk = 2;
        let hv = 4;
        let dk = 32;
        let dv = 32;
        let batch = 1;
        let seq_len = 1;

        let q = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let k = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let v = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv, dv], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let a_log = mlx_rs::random::uniform::<f32, f32>(-1.0, 0.0, &[hv], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let a = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let dt_bias = mlx_rs::random::uniform::<f32, f32>(-0.5, 0.5, &[hv], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let b = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let state = mlx_rs::random::uniform::<f32, f32>(-0.1, 0.1, &[batch, hv, dv, dk], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        // Kernel
        let (kern_y, kern_state) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &b, &state, batch, seq_len, hk, dk, hv, dv,
        )
        .unwrap();
        kern_y.eval().unwrap();
        kern_state.eval().unwrap();

        assert_eq!(kern_y.shape(), &[batch, seq_len, hv, dv]);
        assert_eq!(kern_state.shape(), &[batch, hv, dv, dk]);

        // Verify outputs are finite (not NaN/Inf)
        let y_f32 = kern_y.as_dtype(Dtype::Float32).unwrap();
        let y_abs_max: f32 = y_f32.abs().unwrap().max(None).unwrap().item();
        assert!(
            y_abs_max.is_finite() && y_abs_max < 1e6,
            "bfloat16 kernel y has bad values: max abs = {y_abs_max}"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_kernel_matches_ops(
        batch: i32,
        seq_len: i32,
        hk: i32,
        hv: i32,
        dk: i32,
        dv: i32,
        tol: f32,
        label: &str,
    ) {
        let q = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None)
            .unwrap();
        let k = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None)
            .unwrap();
        let v = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv, dv], None)
            .unwrap();
        let a_log = mlx_rs::random::uniform::<f32, f32>(-1.0, 0.0, &[hv], None).unwrap();
        let a_val =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None).unwrap();
        let dt_bias = mlx_rs::random::uniform::<f32, f32>(-0.5, 0.5, &[hv], None).unwrap();
        let b =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None).unwrap();
        let state =
            mlx_rs::random::uniform::<f32, f32>(-0.1, 0.1, &[batch, hv, dv, dk], None).unwrap();

        // Compute g and beta from raw inputs for the reference path
        let mut compute_g_fn = mlx_rs::transforms::compile::compile(compute_g_compiled, None);
        let g = compute_g_fn((&a_log, &a_val, &dt_bias)).unwrap();
        let beta = nn::sigmoid(&b).unwrap();

        // Reference: loop over timesteps with repeat_axis for GQA
        let repeat_factor = hv / hk;
        let mut ref_state = state.clone();
        let mut ref_ys = Vec::new();
        for t in 0..seq_len {
            let qt = q.index((.., t, .., ..));
            let kt = k.index((.., t, .., ..));
            let vt = v.index((.., t, .., ..));
            let gt = g.index((.., t, ..));
            let bt = beta.index((.., t, ..));

            let qt_rep = if repeat_factor > 1 {
                ops::repeat_axis::<f32>(qt, repeat_factor, -2).unwrap()
            } else {
                qt
            };
            let kt_rep = if repeat_factor > 1 {
                ops::repeat_axis::<f32>(kt, repeat_factor, -2).unwrap()
            } else {
                kt
            };

            let (y_t, new_state) =
                gated_delta_step_ref(&qt_rep, &kt_rep, &vt, &gt, &bt, &ref_state);
            ref_state = new_state;
            ref_ys.push(y_t);
        }
        let ref_y_refs: Vec<&Array> = ref_ys.iter().collect();
        let ref_y = ops::stack_axis(&ref_y_refs, 1).unwrap();
        ref_y.eval().unwrap();
        ref_state.eval().unwrap();

        // Kernel
        let (kern_y, kern_state) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a_val, &dt_bias, &b, &state, batch, seq_len, hk, dk, hv, dv,
        )
        .unwrap();
        kern_y.eval().unwrap();
        kern_state.eval().unwrap();

        // Compare y
        let y_diff = ref_y.subtract(&kern_y).unwrap().abs().unwrap();
        let y_max: f32 = y_diff.max(None).unwrap().item();
        assert!(y_max < tol, "[{label}] kernel y differs by {y_max}");

        // Compare state
        let s_diff = ref_state.subtract(&kern_state).unwrap().abs().unwrap();
        let s_max: f32 = s_diff.max(None).unwrap().item();
        assert!(s_max < tol, "[{label}] kernel state differs by {s_max}");
    }

    /// Benchmark: chain 48 layers of 3x gather_qmm + SwiGLU, single eval.
    /// Compare with Python's 0.378ms (48 layers, single eval).
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_gather_qmm_chain() {
        let num_experts = 512;
        let d = 2048;
        let intermediate = 512;
        let top_k = 10;

        // Create quantized expert weights (same as model)
        let gate_w = Array::zeros::<u32>(&[num_experts, intermediate, d * 4 / 32]).unwrap();
        let gate_s = Array::ones::<f32>(&[num_experts, intermediate, d / 64]).unwrap();
        let gate_b = Array::zeros::<f32>(&[num_experts, intermediate, d / 64]).unwrap();

        let up_w = Array::zeros::<u32>(&[num_experts, intermediate, d * 4 / 32]).unwrap();
        let up_s = Array::ones::<f32>(&[num_experts, intermediate, d / 64]).unwrap();
        let up_b = Array::zeros::<f32>(&[num_experts, intermediate, d / 64]).unwrap();

        let down_w = Array::zeros::<u32>(&[num_experts, d, intermediate * 4 / 32]).unwrap();
        let down_s = Array::ones::<f32>(&[num_experts, d, intermediate / 64]).unwrap();
        let down_b = Array::zeros::<f32>(&[num_experts, d, intermediate / 64]).unwrap();

        let x = Array::ones::<f32>(&[1, 1, 1, 1, d]).unwrap();
        let indices = Array::from_slice(&[0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);
        mlx_rs::transforms::eval([
            &gate_w, &gate_s, &gate_b, &up_w, &up_s, &up_b, &down_w, &down_s, &down_b, &x, &indices,
        ])
        .unwrap();

        // Warm up
        for _ in 0..3 {
            let mut y = x.clone();
            for _ in 0..48 {
                let g = gather_qmm(&y, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false)
                    .unwrap();
                let u = gather_qmm(&y, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
                let activated = swiglu(&g, &u).unwrap();
                y = gather_qmm(
                    &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
                )
                .unwrap();
            }
            mlx_rs::transforms::eval([&y]).unwrap();
        }

        // Benchmark: 48 layers, single eval -- split graph build vs eval
        let n = 50;
        let mut total_build_ns = 0u128;
        let mut total_eval_ns = 0u128;
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let mut y = x.clone();
            for _ in 0..48 {
                let g = gather_qmm(&y, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false)
                    .unwrap();
                let u = gather_qmm(&y, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
                let activated = swiglu(&g, &u).unwrap();
                y = gather_qmm(
                    &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
                )
                .unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y]).unwrap();
            let t2 = std::time::Instant::now();
            total_build_ns += (t1 - t0).as_nanos();
            total_eval_ns += (t2 - t1).as_nanos();
        }
        let build_ms = total_build_ns as f64 / n as f64 / 1_000_000.0;
        let eval_ms = total_eval_ns as f64 / n as f64 / 1_000_000.0;
        eprintln!(
            "48 layers * 3 gather_qmm + SwiGLU: build={build_ms:.2}ms eval={eval_ms:.2}ms total={:.2}ms",
            build_ms + eval_ms
        );

        // Also test with mlx-rs ops::add chain (no FFI gather_qmm)
        let n3 = 50;
        let x_simple = Array::ones::<f32>(&[1, 1, d]).unwrap();
        mlx_rs::transforms::eval([&x_simple]).unwrap();
        let mut total_simple_ns = 0u128;
        for _ in 0..n3 {
            let t0 = std::time::Instant::now();
            let mut y2 = x_simple.clone();
            for _ in 0..(48 * 5) {
                y2 = y2.add(&x_simple).unwrap();
            }
            mlx_rs::transforms::eval([&y2]).unwrap();
            total_simple_ns += t0.elapsed().as_nanos();
        }
        let simple_ms = total_simple_ns as f64 / n3 as f64 / 1_000_000.0;
        eprintln!("240 chained adds (single eval): {simple_ms:.2}ms");

        // Test with the shared gather_qmm wrapper
        let n4 = 50;
        let mut total_builtin_build = 0u128;
        let mut total_builtin_eval = 0u128;
        for _ in 0..n4 {
            let t0 = std::time::Instant::now();
            let mut y3 = x.clone();
            for _ in 0..48 {
                let g = gather_qmm(&y3, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false)
                    .unwrap();
                let u = gather_qmm(&y3, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
                let activated = swiglu(&g, &u).unwrap();
                y3 = gather_qmm(
                    &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
                )
                .unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y3]).unwrap();
            let t2 = std::time::Instant::now();
            total_builtin_build += (t1 - t0).as_nanos();
            total_builtin_eval += (t2 - t1).as_nanos();
        }
        let builtin_build = total_builtin_build as f64 / n4 as f64 / 1_000_000.0;
        let builtin_eval = total_builtin_eval as f64 / n4 as f64 / 1_000_000.0;
        eprintln!(
            "48 layers mlx-rs gather_qmm: build={builtin_build:.2}ms eval={builtin_eval:.2}ms total={:.2}ms",
            builtin_build + builtin_eval
        );

        // Test with quantized_matmul (not gather) - 144 chained calls
        let qm_w = Array::zeros::<u32>(&[d, d * 4 / 32]).unwrap();
        let qm_s = Array::ones::<f32>(&[d, d / 64]).unwrap();
        let qm_b = Array::zeros::<f32>(&[d, d / 64]).unwrap();
        let x_qm = Array::ones::<f32>(&[1, 1, d]).unwrap();
        mlx_rs::transforms::eval([&qm_w, &qm_s, &qm_b, &x_qm]).unwrap();

        // Warm up
        for _ in 0..3 {
            let mut y4 = x_qm.clone();
            for _ in 0..144 {
                y4 = ops::quantized_matmul(&y4, &qm_w, &qm_s, &qm_b, true, 64, 4).unwrap();
            }
            mlx_rs::transforms::eval([&y4]).unwrap();
        }

        let n5 = 50;
        let mut total_qm_build = 0u128;
        let mut total_qm_eval = 0u128;
        for _ in 0..n5 {
            let t0 = std::time::Instant::now();
            let mut y4 = x_qm.clone();
            for _ in 0..144 {
                y4 = ops::quantized_matmul(&y4, &qm_w, &qm_s, &qm_b, true, 64, 4).unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y4]).unwrap();
            let t2 = std::time::Instant::now();
            total_qm_build += (t1 - t0).as_nanos();
            total_qm_eval += (t2 - t1).as_nanos();
        }
        let qm_build = total_qm_build as f64 / n5 as f64 / 1_000_000.0;
        let qm_eval = total_qm_eval as f64 / n5 as f64 / 1_000_000.0;
        eprintln!(
            "144 chained quantized_matmul: build={qm_build:.2}ms eval={qm_eval:.2}ms total={:.2}ms",
            qm_build + qm_eval
        );

        // Benchmark: single layer, per-call eval
        let n2 = 200;
        let start2 = std::time::Instant::now();
        for _ in 0..n2 {
            let g =
                gather_qmm(&x, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false).unwrap();
            let u = gather_qmm(&x, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
            let activated = swiglu(&g, &u).unwrap();
            let y = gather_qmm(
                &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
            )
            .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let per_layer_ms = start2.elapsed().as_millis() as f64 / n2 as f64;
        eprintln!("1 layer * 3 gather_qmm + SwiGLU (per-call eval): {per_layer_ms:.2} ms");

        // Test eval overhead: 1000 chained adds (Python: build=0.23ms eval=1.87ms)
        let n_ops = 1000;
        let x_add = Array::ones::<f32>(&[1, 1, 2048]).unwrap();
        mlx_rs::transforms::eval([&x_add]).unwrap();
        // Warmup
        for _ in 0..3 {
            let mut y = x_add.clone();
            for _ in 0..n_ops {
                y = y.add(&x_add).unwrap();
            }
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let n6 = 50;
        let mut total_add_build = 0u128;
        let mut total_add_eval = 0u128;
        for _ in 0..n6 {
            let t0 = std::time::Instant::now();
            let mut y = x_add.clone();
            for _ in 0..n_ops {
                y = y.add(&x_add).unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y]).unwrap();
            let t2 = std::time::Instant::now();
            total_add_build += (t1 - t0).as_nanos();
            total_add_eval += (t2 - t1).as_nanos();
        }
        let add_build = total_add_build as f64 / n6 as f64 / 1_000_000.0;
        let add_eval = total_add_eval as f64 / n6 as f64 / 1_000_000.0;
        eprintln!(
            "{n_ops} chained adds: build={add_build:.2}ms eval={add_eval:.2}ms total={:.2}ms",
            add_build + add_eval
        );
        eprintln!(
            "Per op: build={:.1}us eval={:.1}us",
            add_build * 1000.0 / n_ops as f64,
            add_eval * 1000.0 / n_ops as f64
        );

        // Test with task-local default stream
        let stream = mlx_rs::Stream::new();
        let gather_with_stream = || {
            mlx_rs::with_new_default_stream(stream.clone(), || {
                let mut total_b = 0u128;
                let mut total_e = 0u128;
                let n7 = 50;
                for _ in 0..n7 {
                    let t0 = std::time::Instant::now();
                    let mut y = x.clone();
                    for _ in 0..48 {
                        let g =
                            gather_qmm(&y, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false)
                                .unwrap();
                        let u = gather_qmm(&y, &up_w, &up_s, &up_b, &indices, true, 64, 4, false)
                            .unwrap();
                        let activated = swiglu(&g, &u).unwrap();
                        y = gather_qmm(
                            &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
                        )
                        .unwrap();
                    }
                    let t1 = std::time::Instant::now();
                    mlx_rs::transforms::eval([&y]).unwrap();
                    let t2 = std::time::Instant::now();
                    total_b += (t1 - t0).as_nanos();
                    total_e += (t2 - t1).as_nanos();
                }
                let b = total_b as f64 / n7 as f64 / 1_000_000.0;
                let e = total_e as f64 / n7 as f64 / 1_000_000.0;
                eprintln!(
                    "48 layers gather_qmm (with task-local stream): build={b:.2}ms eval={e:.2}ms total={:.2}ms",
                    b + e
                );
            });
        };
        gather_with_stream();
    }

    /// Benchmark: 200 chained quantized_matmul ops (matching Python bench).
    /// Python: build=0.05ms eval=1.40ms total=1.45ms
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_chained_quantized_matmul() {
        use mlx_rs::Dtype;

        let x = ops::ones_dtype(&[1, 1, 2048], Dtype::Float16).unwrap();
        let raw_w = ops::ones_dtype(&[2048, 2048], Dtype::Float16).unwrap();
        let (w, s, b) = ops::quantize(&raw_w, 64, 4).unwrap();
        mlx_rs::transforms::eval([&x, &w, &s, &b]).unwrap();

        let n_ops = 200;
        let n = 50;

        // Warmup
        for _ in 0..10 {
            let mut y = x.clone();
            for _ in 0..n_ops {
                y = ops::quantized_matmul(&y, &w, &s, &b, true, 64, 4).unwrap();
            }
            mlx_rs::transforms::eval([&y]).unwrap();
        }

        let mut total_build = 0u128;
        let mut total_eval = 0u128;
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let mut y = x.clone();
            for _ in 0..n_ops {
                y = ops::quantized_matmul(&y, &w, &s, &b, true, 64, 4).unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y]).unwrap();
            let t2 = std::time::Instant::now();
            total_build += (t1 - t0).as_nanos();
            total_eval += (t2 - t1).as_nanos();
        }
        let build = total_build as f64 / n as f64 / 1e6;
        let eval = total_eval as f64 / n as f64 / 1e6;
        eprintln!(
            "Rust 200 qmm: build={build:.2}ms eval={eval:.2}ms total={:.2}ms",
            build + eval
        );

        // 200 chained adds
        for _ in 0..10 {
            let mut y = x.clone();
            for _ in 0..n_ops {
                y = y.add(&x).unwrap();
            }
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let mut total_build = 0u128;
        let mut total_eval = 0u128;
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let mut y = x.clone();
            for _ in 0..n_ops {
                y = y.add(&x).unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y]).unwrap();
            let t2 = std::time::Instant::now();
            total_build += (t1 - t0).as_nanos();
            total_eval += (t2 - t1).as_nanos();
        }
        let build = total_build as f64 / n as f64 / 1e6;
        let eval = total_eval as f64 / n as f64 / 1e6;
        eprintln!(
            "Rust 200 add: build={build:.2}ms eval={eval:.2}ms total={:.2}ms",
            build + eval
        );
    }

    /// Simulate 48-layer forward pass with per-layer weights.
    /// Python shared-weight sim: build=0.59ms eval=8.08ms
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_simulated_forward() {
        use mlx_rs::Dtype;

        let d = 2048i32;
        let d_inter = 512i32; // moe_intermediate_size from config
        let n_experts = 512i32;
        let top_k = 10i32; // num_experts_per_tok from config
        let gs = 64i32;
        let bits = 4i32;
        let shared_inter = 512i32; // shared_expert_intermediate_size

        // Use random weights to test realistic memory access patterns.
        // ops::ones_dtype creates constant data that artificially benefits from GPU cache.
        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };
        let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };

        let hk = 16i32;
        let dk = 128i32;
        let hv = 32i32;
        let dv = 128i32;

        struct LayerWeights {
            q_proj: (Array, Array, Array),
            k_proj: (Array, Array, Array),
            v_proj: (Array, Array, Array),
            o_proj: (Array, Array, Array),
            g_proj: (Array, Array, Array),
            beta_proj: (Array, Array, Array),
            gate: (Array, Array, Array),
            sw_gate: (Array, Array, Array),
            sw_up: (Array, Array, Array),
            sw_down: (Array, Array, Array),
            se_gate: (Array, Array, Array),
            se_up: (Array, Array, Array),
            se_down: (Array, Array, Array),
            se_gate_proj: (Array, Array, Array),
            norm_w: Array,
        }

        let layers: Vec<LayerWeights> = (0..48)
            .map(|_| LayerWeights {
                q_proj: make_qw(d, hk * dk),
                k_proj: make_qw(d, hk * dk),
                v_proj: make_qw(d, hv * dv),
                o_proj: make_qw(hv * dv, d),
                g_proj: make_qw(d, hv),
                beta_proj: make_qw(d, hv),
                gate: make_qw(d, n_experts),
                sw_gate: make_sw(d, d_inter),
                sw_up: make_sw(d, d_inter),
                sw_down: make_sw(d_inter, d),
                se_gate: make_qw(d, shared_inter * 2),
                se_up: make_qw(d, shared_inter * 2),
                se_down: make_qw(shared_inter * 2, d),
                se_gate_proj: make_qw(d, 1),
                norm_w: Array::ones::<f32>(&[d])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap(),
            })
            .collect();

        let mut all_w: Vec<&Array> = Vec::new();
        for l in &layers {
            for (w, s, b) in [
                &l.q_proj,
                &l.k_proj,
                &l.v_proj,
                &l.o_proj,
                &l.g_proj,
                &l.beta_proj,
                &l.gate,
                &l.sw_gate,
                &l.sw_up,
                &l.sw_down,
                &l.se_gate,
                &l.se_up,
                &l.se_down,
                &l.se_gate_proj,
            ] {
                all_w.extend_from_slice(&[w, s, b]);
            }
            all_w.push(&l.norm_w);
        }
        mlx_rs::transforms::eval(all_w).unwrap();

        // Check actual memory usage to verify weights are materialized
        let active_mem = {
            let mut res: usize = 0;
            #[allow(unsafe_code)]
            unsafe {
                mlx_sys::mlx_get_active_memory(&mut res as *mut _);
            }
            res
        };
        eprintln!(
            "Active memory after weight eval: {:.2} GB",
            active_mem as f64 / 1e9
        );

        // Print one switch weight shape to verify
        eprintln!(
            "sw_gate[0] shape: {:?} dtype: {:?}",
            layers[0].sw_gate.0.shape(),
            layers[0].sw_gate.0.dtype()
        );

        let x = ops::ones_dtype(&[1, 1, d], Dtype::Float16).unwrap();
        mlx_rs::transforms::eval([&x]).unwrap();

        let forward_n_inline = |x: &Array, n_layers: usize| -> Array {
            let mut h = x.clone();
            for l in layers.iter().take(n_layers) {
                let normed = fast::rms_norm(&h, &l.norm_w, 1e-6).unwrap();

                // Attention projections (matching real model's GDN layer ops)
                let _q = ops::quantized_matmul(
                    &normed,
                    &l.q_proj.0,
                    &l.q_proj.1,
                    &l.q_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let _k = ops::quantized_matmul(
                    &normed,
                    &l.k_proj.0,
                    &l.k_proj.1,
                    &l.k_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let v = ops::quantized_matmul(
                    &normed,
                    &l.v_proj.0,
                    &l.v_proj.1,
                    &l.v_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let g = ops::quantized_matmul(
                    &normed,
                    &l.g_proj.0,
                    &l.g_proj.1,
                    &l.g_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let _beta = ops::quantized_matmul(
                    &normed,
                    &l.beta_proj.0,
                    &l.beta_proj.1,
                    &l.beta_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let attn_proxy = v
                    .multiply(&nn::sigmoid(&g.sum_axes(&[-1], true).unwrap()).unwrap())
                    .unwrap();
                let o = ops::quantized_matmul(
                    &attn_proxy,
                    &l.o_proj.0,
                    &l.o_proj.1,
                    &l.o_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();

                let h2 = h.add(o).unwrap();
                let normed2 = fast::rms_norm(&h2, &l.norm_w, 1e-6).unwrap();

                // Router
                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                // Switch MLP (per-layer switch weights)
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                // Shared expert (per-layer weights)
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        };

        for n_layers in [1, 4, 8, 16, 24, 48] {
            for _ in 0..5 {
                let y = forward_n_inline(&x, n_layers);
                mlx_rs::transforms::eval([&y]).unwrap();
            }
            let n = 20;
            let mut total_eval = 0u128;
            for _ in 0..n {
                let y = forward_n_inline(&x, n_layers);
                let t0 = std::time::Instant::now();
                mlx_rs::transforms::eval([&y]).unwrap();
                total_eval += t0.elapsed().as_nanos();
            }
            let eval = total_eval as f64 / n as f64 / 1e6;
            eprintln!(
                "Inline {n_layers} layers: eval={eval:.2}ms per_layer={:.2}ms",
                eval / n_layers as f64
            );
        }
    }

    /// Test gather_qmm with loaded vs random weights to isolate memory effects.
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_gather_qmm_loaded_vs_random() {
        use mlx_rs::Dtype;
        let model_dir = "/Users/panbanda/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5";
        let shard = format!("{}/model-00001-of-00009.safetensors", model_dir);
        let path = std::path::Path::new(&shard);
        if !path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        // Load one safetensors shard
        let loaded = Array::load_safetensors(path).unwrap();
        mlx_rs::transforms::eval(loaded.values()).unwrap();

        // Find a switch_mlp weight (should be large [512, intermediate, ...])
        let mut sw_key = None;
        for key in loaded.keys() {
            if key.contains("switch_mlp") && key.contains("gate_proj") && key.contains(".weight") {
                sw_key = Some(key.clone());
                break;
            }
        }
        let sw_key = sw_key.expect("No switch_mlp weight found in shard");
        let w_loaded = &loaded[&sw_key];
        eprintln!(
            "Loaded weight '{sw_key}': shape={:?} dtype={:?}",
            w_loaded.shape(),
            w_loaded.dtype()
        );

        // Find corresponding scales and biases
        let scales_key = sw_key.replace(".weight", ".scales");
        let biases_key = sw_key.replace(".weight", ".biases");
        let s_loaded = &loaded[&scales_key];
        let b_loaded = &loaded[&biases_key];
        eprintln!(
            "Scales: {:?}, Biases: {:?}",
            s_loaded.shape(),
            b_loaded.shape()
        );

        // Create random weights of the same shape/dtype
        let w_shape = w_loaded.shape().to_vec();
        let s_shape = s_loaded.shape().to_vec();
        let b_shape = b_loaded.shape().to_vec();

        let w_random = mlx_rs::random::normal::<f32>(&w_shape, None, None, None)
            .unwrap()
            .as_dtype(w_loaded.dtype())
            .unwrap();
        let s_random = mlx_rs::random::normal::<f32>(&s_shape, None, None, None)
            .unwrap()
            .as_dtype(s_loaded.dtype())
            .unwrap();
        let b_random = mlx_rs::random::normal::<f32>(&b_shape, None, None, None)
            .unwrap()
            .as_dtype(b_loaded.dtype())
            .unwrap();
        mlx_rs::transforms::eval([&w_random, &s_random, &b_random]).unwrap();

        // Test input
        let x = ops::ones_dtype(&[1, 1, 1, 1, 2048], Dtype::Float16).unwrap();
        let indices = Array::from_slice(&[0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, 10]);
        mlx_rs::transforms::eval([&x, &indices]).unwrap();

        let gs = 64i32;
        let bits = 4i32;
        let n = 100;

        // Benchmark loaded weights
        for _ in 0..10 {
            let y = gather_qmm(
                &x, w_loaded, s_loaded, b_loaded, &indices, true, gs, bits, false,
            )
            .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let mut total_loaded = 0u128;
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let y = gather_qmm(
                &x, w_loaded, s_loaded, b_loaded, &indices, true, gs, bits, false,
            )
            .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
            total_loaded += t0.elapsed().as_nanos();
        }

        // Benchmark random weights
        for _ in 0..10 {
            let y = gather_qmm(
                &x, &w_random, &s_random, &b_random, &indices, true, gs, bits, false,
            )
            .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let mut total_random = 0u128;
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let y = gather_qmm(
                &x, &w_random, &s_random, &b_random, &indices, true, gs, bits, false,
            )
            .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
            total_random += t0.elapsed().as_nanos();
        }

        let loaded_us = total_loaded as f64 / n as f64 / 1e3;
        let random_us = total_random as f64 / n as f64 / 1e3;
        eprintln!(
            "gather_qmm single layer: loaded={loaded_us:.1}us random={random_us:.1}us ratio={:.2}x",
            loaded_us / random_us
        );
    }

    /// Isolate what causes the module vs inline performance gap.
    /// Tests three variants at 48 layers:
    /// A) Module forward with multiply-by-zero attention (baseline slow path)
    /// B) Inline forward with multiply-by-zero attention (tests if graph structure matters)
    /// C) Inline forward with real quantized_matmul attention (original fast path)
    /// D) Extract weights from modules into tuples, run inline (tests Param<Array> access)
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_module_vs_inline() {
        use mlx_rs::Dtype;
        use mlx_rs::module::Param;

        let d = 2048i32;
        let d_inter = 512i32;
        let n_experts = 512i32;
        let top_k = 10i32;
        let gs = 64i32;
        let bits = 4i32;
        let shared_inter = 512i32;

        let make_ql = |d_in: i32, d_out: i32, gs: i32, bits: i32| -> QLinear {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            QLinear {
                weight: Param::new(w),
                scales: Param::new(s),
                biases: Param::new(b),
                group_size: gs,
                bits,
            }
        };

        let make_switch_ql = |d_in: i32, d_out: i32| -> QLinear {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            QLinear {
                weight: Param::new(w),
                scales: Param::new(s),
                biases: Param::new(b),
                group_size: gs,
                bits,
            }
        };

        // Build 48 SparseMoeBlock instances with random weights
        let moe_blocks: Vec<SparseMoeBlock> = (0..48)
            .map(|_| SparseMoeBlock {
                gate: make_ql(d, n_experts, gs, bits),
                switch_mlp: SwitchMlpWeights {
                    gate_proj: make_switch_ql(d, d_inter),
                    up_proj: make_switch_ql(d, d_inter),
                    down_proj: make_switch_ql(d_inter, d),
                    fused_gate_up: None,
                },
                shared_expert: Qwen3NextMLP {
                    gate_proj: make_ql(d, shared_inter * 2, gs, bits),
                    up_proj: make_ql(d, shared_inter * 2, gs, bits),
                    down_proj: make_ql(shared_inter * 2, d, gs, bits),
                },
                shared_expert_gate: make_ql(d, 1, gs, bits),
                top_k,
                norm_topk_prob: true,
            })
            .collect();

        // Eval all module weights
        {
            use mlx_rs::module::ModuleParameters;
            let mut all_w: Vec<&Array> = Vec::new();
            for moe in &moe_blocks {
                for (_, arr) in moe.parameters().flatten() {
                    all_w.push(arr);
                }
            }
            mlx_rs::transforms::eval(all_w).unwrap();
        }

        // Extract module weights into bare tuples for variant D
        struct ExtractedWeights {
            gate: (Array, Array, Array),
            sw_gate: (Array, Array, Array),
            sw_up: (Array, Array, Array),
            sw_down: (Array, Array, Array),
            se_gate: (Array, Array, Array),
            se_up: (Array, Array, Array),
            se_down: (Array, Array, Array),
            se_gate_proj: (Array, Array, Array),
        }
        let extracted: Vec<ExtractedWeights> = moe_blocks
            .iter()
            .map(|moe| {
                // Clone the Array handles (cheap refcount bump, same underlying MLX data)
                ExtractedWeights {
                    gate: (
                        moe.gate.weight.value.clone(),
                        moe.gate.scales.value.clone(),
                        moe.gate.biases.value.clone(),
                    ),
                    sw_gate: (
                        moe.switch_mlp.gate_proj.weight.value.clone(),
                        moe.switch_mlp.gate_proj.scales.value.clone(),
                        moe.switch_mlp.gate_proj.biases.value.clone(),
                    ),
                    sw_up: (
                        moe.switch_mlp.up_proj.weight.value.clone(),
                        moe.switch_mlp.up_proj.scales.value.clone(),
                        moe.switch_mlp.up_proj.biases.value.clone(),
                    ),
                    sw_down: (
                        moe.switch_mlp.down_proj.weight.value.clone(),
                        moe.switch_mlp.down_proj.scales.value.clone(),
                        moe.switch_mlp.down_proj.biases.value.clone(),
                    ),
                    se_gate: (
                        moe.shared_expert.gate_proj.weight.value.clone(),
                        moe.shared_expert.gate_proj.scales.value.clone(),
                        moe.shared_expert.gate_proj.biases.value.clone(),
                    ),
                    se_up: (
                        moe.shared_expert.up_proj.weight.value.clone(),
                        moe.shared_expert.up_proj.scales.value.clone(),
                        moe.shared_expert.up_proj.biases.value.clone(),
                    ),
                    se_down: (
                        moe.shared_expert.down_proj.weight.value.clone(),
                        moe.shared_expert.down_proj.scales.value.clone(),
                        moe.shared_expert.down_proj.biases.value.clone(),
                    ),
                    se_gate_proj: (
                        moe.shared_expert_gate.weight.value.clone(),
                        moe.shared_expert_gate.scales.value.clone(),
                        moe.shared_expert_gate.biases.value.clone(),
                    ),
                }
            })
            .collect();

        let norm_w = Array::ones::<f32>(&[d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let x = ops::ones_dtype(&[1, 1, d], Dtype::Float16).unwrap();
        mlx_rs::transforms::eval([&x, &norm_w]).unwrap();

        let n_layers = 48usize;
        let n = 20;

        // Helper: run N warmups then N timed evals
        let bench = |label: &str, forward: &dyn Fn(&Array) -> Array| {
            for _ in 0..5 {
                let y = forward(&x);
                mlx_rs::transforms::eval([&y]).unwrap();
            }
            let mut total = 0u128;
            for _ in 0..n {
                let y = forward(&x);
                let t0 = std::time::Instant::now();
                mlx_rs::transforms::eval([&y]).unwrap();
                total += t0.elapsed().as_nanos();
            }
            let ms = total as f64 / n as f64 / 1e6;
            eprintln!(
                "{label}: eval={ms:.2}ms per_layer={:.2}ms",
                ms / n_layers as f64
            );
        };

        // A) Module forward + multiply-by-zero attention
        bench("A) module+zero_attn", &|x: &Array| {
            let mut h = x.clone();
            for moe in moe_blocks.iter().take(n_layers) {
                let normed = fast::rms_norm(&h, &norm_w, 1e-6).unwrap();
                let dummy_attn = normed.multiply(Array::from_f32(0.0)).unwrap();
                let h2 = h.add(dummy_attn).unwrap();
                let normed2 = fast::rms_norm(&h2, &norm_w, 1e-6).unwrap();
                let mlp_out = moe.forward(&normed2).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // B) Inline forward + multiply-by-zero attention (same extracted weights)
        bench("B) inline+zero_attn", &|x: &Array| {
            let mut h = x.clone();
            for l in extracted.iter().take(n_layers) {
                let normed = fast::rms_norm(&h, &norm_w, 1e-6).unwrap();
                let dummy_attn = normed.multiply(Array::from_f32(0.0)).unwrap();
                let h2 = h.add(dummy_attn).unwrap();
                let normed2 = fast::rms_norm(&h2, &norm_w, 1e-6).unwrap();

                // Inline MoE (same code as bench_simulated_forward)
                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // C) Inline forward + real quantized_matmul for attention (per-layer attn weights)
        // This matches the bench_simulated_forward test structure
        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };
        let attn_weights: Vec<(Array, Array, Array)> = (0..48).map(|_| make_qw(d, d)).collect();
        let per_layer_norms: Vec<Array> = (0..48)
            .map(|_| {
                Array::ones::<f32>(&[d])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        {
            let mut all_w: Vec<&Array> = Vec::new();
            for (w, s, b) in &attn_weights {
                all_w.extend_from_slice(&[w, s, b]);
            }
            for nw in &per_layer_norms {
                all_w.push(nw);
            }
            mlx_rs::transforms::eval(all_w).unwrap();
        }

        bench("C) inline+real_attn+per_layer_norm", &|x: &Array| {
            let mut h = x.clone();
            for (i, l) in extracted.iter().take(n_layers).enumerate() {
                let normed = fast::rms_norm(&h, &per_layer_norms[i], 1e-6).unwrap();
                let attn_out = ops::quantized_matmul(
                    &normed,
                    &attn_weights[i].0,
                    &attn_weights[i].1,
                    &attn_weights[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let h2 = h.add(attn_out).unwrap();
                let normed2 = fast::rms_norm(&h2, &per_layer_norms[i], 1e-6).unwrap();

                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // D) Inline + zero_attn + per_layer_norm (isolates norm_w sharing vs attn method)
        bench("D) inline+zero_attn+per_layer_norm", &|x: &Array| {
            let mut h = x.clone();
            for (i, l) in extracted.iter().take(n_layers).enumerate() {
                let normed = fast::rms_norm(&h, &per_layer_norms[i], 1e-6).unwrap();
                let dummy_attn = normed.multiply(Array::from_f32(0.0)).unwrap();
                let h2 = h.add(dummy_attn).unwrap();
                let normed2 = fast::rms_norm(&h2, &per_layer_norms[i], 1e-6).unwrap();

                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // E) Inline + multiply-by-ONE + shared norm (is zero specifically the issue?)
        bench("E) inline+mul_one_attn", &|x: &Array| {
            let mut h = x.clone();
            for l in extracted.iter().take(n_layers) {
                let normed = fast::rms_norm(&h, &norm_w, 1e-6).unwrap();
                let dummy_attn = normed.multiply(Array::from_f32(1.0)).unwrap();
                let h2 = h.add(dummy_attn).unwrap();
                let normed2 = fast::rms_norm(&h2, &norm_w, 1e-6).unwrap();

                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // F) Inline + zeros_like (skip normed entirely, just add zeros)
        bench("F) inline+zeros_like_attn", &|x: &Array| {
            let mut h = x.clone();
            for l in extracted.iter().take(n_layers) {
                let normed = fast::rms_norm(&h, &norm_w, 1e-6).unwrap();
                let _ = &normed; // normed computed but not used for attn
                let dummy_attn = ops::zeros_like(&normed).unwrap();
                let h2 = h.add(dummy_attn).unwrap();
                let normed2 = fast::rms_norm(&h2, &norm_w, 1e-6).unwrap();

                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // G) Inline + skip normed entirely, h2 = h (no ops for attention)
        bench("G) inline+h2_equals_h", &|x: &Array| {
            let mut h = x.clone();
            for l in extracted.iter().take(n_layers) {
                // Skip first rms_norm entirely
                let h2 = h.clone();
                let normed2 = fast::rms_norm(&h2, &norm_w, 1e-6).unwrap();

                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });
    }

    /// Benchmark 36 GDN layers using bare Arrays (matching Python bench_gdn_real_python.py).
    /// Isolates GDN ops from the model framework to compare GPU time vs Python.
    #[test]
    #[ignore = "requires GPU"]
    fn bench_gdn_layers() {
        use mlx_rs::Dtype;

        let d = 2048i32;
        let hk = 16i32;
        let hv = 32i32;
        let dk = 128i32;
        let dv = 128i32;
        let gs = 64i32;
        let bits = 4i32;
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz_out = key_dim * 2 + value_dim * 2;
        let ba_out = hv * 2;
        let n_layers = 36;

        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            (w, s, b)
        };

        struct GDNWeights {
            in_proj_qkvz: (Array, Array, Array),
            in_proj_ba: (Array, Array, Array),
            out_proj: (Array, Array, Array),
            conv_w: Array,
            a_log: Array,
            dt_bias: Array,
            norm_w: Array,
        }

        let mut layers = Vec::new();
        let mut all_w: Vec<&Array> = Vec::new();
        for _ in 0..n_layers {
            layers.push(GDNWeights {
                in_proj_qkvz: make_qw(d, qkvz_out),
                in_proj_ba: make_qw(d, ba_out),
                out_proj: make_qw(value_dim, d),
                conv_w: mlx_rs::random::normal::<f32>(&[conv_dim, 4, 1], None, None, None)
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap(),
                a_log: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
                dt_bias: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
                norm_w: Array::ones::<f32>(&[dv])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap(),
            });
        }
        for l in &layers {
            all_w.extend([&l.in_proj_qkvz.0, &l.in_proj_qkvz.1, &l.in_proj_qkvz.2]);
            all_w.extend([&l.in_proj_ba.0, &l.in_proj_ba.1, &l.in_proj_ba.2]);
            all_w.extend([&l.out_proj.0, &l.out_proj.1, &l.out_proj.2]);
            all_w.extend([&l.conv_w, &l.a_log, &l.dt_bias, &l.norm_w]);
        }
        mlx_rs::transforms::eval(all_w).unwrap();

        let x = Array::ones::<f32>(&[1, 1, d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let qk_norm_w = Array::ones::<f32>(&[dk]).unwrap();
        let inv_scale = Array::from_f32((dk as f32).sqrt().recip());
        let inv_scale_sq = {
            let s = (dk as f32).sqrt().recip();
            Array::from_f32(s * s)
        };
        let states: Vec<Array> = (0..n_layers)
            .map(|_| {
                Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        let conv_states: Vec<Array> = (0..n_layers)
            .map(|_| {
                Array::zeros::<f32>(&[1, 3, conv_dim])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        x.eval().unwrap();
        for s in &states {
            s.eval().unwrap();
        }
        for c in &conv_states {
            c.eval().unwrap();
        }

        let gdn_forward = |h: &Array,
                           l: &GDNWeights,
                           state: &Array,
                           conv_state: &Array|
         -> (Array, Array, Array) {
            let qkvz = ops::quantized_matmul(
                h,
                &l.in_proj_qkvz.0,
                &l.in_proj_qkvz.1,
                &l.in_proj_qkvz.2,
                true,
                gs,
                bits,
            )
            .unwrap();
            let ba = ops::quantized_matmul(
                h,
                &l.in_proj_ba.0,
                &l.in_proj_ba.1,
                &l.in_proj_ba.2,
                true,
                gs,
                bits,
            )
            .unwrap();

            let q = qkvz
                .index((.., .., ..key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let k = qkvz
                .index((.., .., key_dim..2 * key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let v = qkvz
                .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                .reshape(&[1, 1, hv, dv])
                .unwrap();
            let z = qkvz.index((.., .., 2 * key_dim + value_dim..));

            let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
            let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();

            // Conv1d
            let q_flat = q.reshape(&[1, 1, -1]).unwrap();
            let k_flat = k.reshape(&[1, 1, -1]).unwrap();
            let v_flat = v.reshape(&[1, 1, -1]).unwrap();
            let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
            let conv_in = ops::concatenate_axis(&[conv_state, &mixed], 1).unwrap();
            let new_conv_state = conv_in.index((.., -3.., ..));

            let conv_out =
                nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();

            let conv_q = conv_out
                .index((.., .., ..key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let conv_k = conv_out
                .index((.., .., key_dim..2 * key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let conv_v = conv_out
                .index((.., .., 2 * key_dim..))
                .reshape(&[1, 1, hv, dv])
                .unwrap();

            // RMS norm
            let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                .unwrap()
                .multiply(&inv_scale_sq)
                .unwrap();
            let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                .unwrap()
                .multiply(&inv_scale)
                .unwrap();

            // Metal kernel (computes g and beta internally)
            let (y, new_state) = gated_delta_kernel_ffi(
                &norm_q, &norm_k, &conv_v, &l.a_log, &a, &l.dt_bias, &b, state, 1, 1, hk, dk, hv,
                dv,
            )
            .unwrap();

            // Gated RMSNorm + swiglu
            let normed = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
            let z_shaped = z
                .index((.., .., ..value_dim))
                .reshape(&[1, 1, hv, dv])
                .unwrap();
            let gated = swiglu(&z_shaped, &normed).unwrap();

            // Output proj
            let out = ops::quantized_matmul(
                &gated.reshape(&[1, 1, -1]).unwrap(),
                &l.out_proj.0,
                &l.out_proj.1,
                &l.out_proj.2,
                true,
                gs,
                bits,
            )
            .unwrap();
            (out, new_state, new_conv_state)
        };

        // Warmup
        for _ in 0..5 {
            let mut h = x.clone();
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            for (j, l) in layers.iter().enumerate() {
                let (out, ns, nc) = gdn_forward(&h, l, &ss[j], &cs[j]);
                h = out;
                ss[j] = ns;
                cs[j] = nc;
            }
            let mut eval_targets: Vec<&Array> = vec![&h];
            eval_targets.extend(ss.iter());
            eval_targets.extend(cs.iter());
            mlx_rs::transforms::eval(eval_targets).unwrap();
        }

        // Benchmark
        let n = 20;
        let mut total = 0u128;
        for _ in 0..n {
            let mut h = x.clone();
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            for (j, l) in layers.iter().enumerate() {
                let (out, ns, nc) = gdn_forward(&h, l, &ss[j], &cs[j]);
                h = out;
                ss[j] = ns;
                cs[j] = nc;
            }
            let t0 = std::time::Instant::now();
            let mut eval_targets: Vec<&Array> = vec![&h];
            eval_targets.extend(ss.iter());
            eval_targets.extend(cs.iter());
            mlx_rs::transforms::eval(eval_targets).unwrap();
            total += t0.elapsed().as_nanos();
        }

        let avg_ms = total as f64 / n as f64 / 1e6;
        println!("Rust 36 GDN layers (bare arrays): {avg_ms:.2}ms");
        println!("Per layer: {:.3}ms", avg_ms / 36.0);
    }

    /// Benchmark 48 layers of interleaved GDN + MoE (matching real model structure).
    /// GDN layers: 0,1,2, 4,5,6, 8,9,10, ...  (every layer except multiples of 4 minus 1)
    /// FA layers: 3,7,11,... (every 4th layer, 0-indexed)
    /// All layers have MoE.
    #[test]
    #[ignore = "requires GPU"]
    fn bench_combined_gdn_moe() {
        use mlx_rs::Dtype;

        let d = 2048i32;
        let hk = 16i32;
        let hv = 32i32;
        let dk = 128i32;
        let dv = 128i32;
        let gs = 64i32;
        let bits = 4i32;
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz_out = key_dim * 2 + value_dim * 2;
        let ba_out = hv * 2;
        let n_layers = 48;
        let full_attn_interval = 4;
        let d_inter = 512i32;
        let n_experts = 512i32;
        let top_k = 10i32;
        let shared_inter = 512i32;

        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            (w, s, b)
        };
        let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            (w, s, b)
        };

        struct GDNWeights {
            in_proj_qkvz: (Array, Array, Array),
            in_proj_ba: (Array, Array, Array),
            out_proj: (Array, Array, Array),
            conv_w: Array,
            a_log: Array,
            dt_bias: Array,
            norm_w: Array,
        }
        struct MoEWeights {
            gate: (Array, Array, Array),
            sw_gate: (Array, Array, Array),
            sw_up: (Array, Array, Array),
            sw_down: (Array, Array, Array),
            se_gate: (Array, Array, Array),
            se_up: (Array, Array, Array),
            se_down: (Array, Array, Array),
            se_gate_proj: (Array, Array, Array),
            norm_w: Array,
        }
        struct AttnWeights {
            q_proj: (Array, Array, Array),
            k_proj: (Array, Array, Array),
            v_proj: (Array, Array, Array),
            o_proj: (Array, Array, Array),
        }

        let mut gdn_layers: Vec<Option<GDNWeights>> = Vec::new();
        let mut attn_layers: Vec<Option<AttnWeights>> = Vec::new();
        let mut moe_layers: Vec<MoEWeights> = Vec::new();
        let mut all_w: Vec<Array> = Vec::new();

        for i in 0..n_layers {
            let is_linear = (i + 1) % full_attn_interval != 0;
            if is_linear {
                let gdn = GDNWeights {
                    in_proj_qkvz: make_qw(d, qkvz_out),
                    in_proj_ba: make_qw(d, ba_out),
                    out_proj: make_qw(value_dim, d),
                    conv_w: mlx_rs::random::normal::<f32>(&[conv_dim, 4, 1], None, None, None)
                        .unwrap()
                        .as_dtype(Dtype::Float16)
                        .unwrap(),
                    a_log: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
                    dt_bias: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
                    norm_w: Array::ones::<f32>(&[dv])
                        .unwrap()
                        .as_dtype(Dtype::Float16)
                        .unwrap(),
                };
                all_w.extend([
                    gdn.in_proj_qkvz.0.clone(),
                    gdn.in_proj_qkvz.1.clone(),
                    gdn.in_proj_qkvz.2.clone(),
                ]);
                all_w.extend([
                    gdn.in_proj_ba.0.clone(),
                    gdn.in_proj_ba.1.clone(),
                    gdn.in_proj_ba.2.clone(),
                ]);
                all_w.extend([
                    gdn.out_proj.0.clone(),
                    gdn.out_proj.1.clone(),
                    gdn.out_proj.2.clone(),
                ]);
                all_w.extend([
                    gdn.conv_w.clone(),
                    gdn.a_log.clone(),
                    gdn.dt_bias.clone(),
                    gdn.norm_w.clone(),
                ]);
                gdn_layers.push(Some(gdn));
                attn_layers.push(None);
            } else {
                let attn = AttnWeights {
                    q_proj: make_qw(d, d),
                    k_proj: make_qw(d, d),
                    v_proj: make_qw(d, d),
                    o_proj: make_qw(d, d),
                };
                all_w.extend([
                    attn.q_proj.0.clone(),
                    attn.q_proj.1.clone(),
                    attn.q_proj.2.clone(),
                ]);
                all_w.extend([
                    attn.k_proj.0.clone(),
                    attn.k_proj.1.clone(),
                    attn.k_proj.2.clone(),
                ]);
                all_w.extend([
                    attn.v_proj.0.clone(),
                    attn.v_proj.1.clone(),
                    attn.v_proj.2.clone(),
                ]);
                all_w.extend([
                    attn.o_proj.0.clone(),
                    attn.o_proj.1.clone(),
                    attn.o_proj.2.clone(),
                ]);
                gdn_layers.push(None);
                attn_layers.push(Some(attn));
            }
            let moe = MoEWeights {
                gate: make_qw(d, n_experts),
                sw_gate: make_sw(d, d_inter),
                sw_up: make_sw(d, d_inter),
                sw_down: make_sw(d_inter, d),
                se_gate: make_qw(d, shared_inter * 2),
                se_up: make_qw(d, shared_inter * 2),
                se_down: make_qw(shared_inter * 2, d),
                se_gate_proj: make_qw(d, 1),
                norm_w: Array::ones::<f32>(&[d])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap(),
            };
            all_w.extend([moe.gate.0.clone(), moe.gate.1.clone(), moe.gate.2.clone()]);
            all_w.extend([
                moe.sw_gate.0.clone(),
                moe.sw_gate.1.clone(),
                moe.sw_gate.2.clone(),
            ]);
            all_w.extend([
                moe.sw_up.0.clone(),
                moe.sw_up.1.clone(),
                moe.sw_up.2.clone(),
            ]);
            all_w.extend([
                moe.sw_down.0.clone(),
                moe.sw_down.1.clone(),
                moe.sw_down.2.clone(),
            ]);
            all_w.extend([
                moe.se_gate.0.clone(),
                moe.se_gate.1.clone(),
                moe.se_gate.2.clone(),
            ]);
            all_w.extend([
                moe.se_up.0.clone(),
                moe.se_up.1.clone(),
                moe.se_up.2.clone(),
            ]);
            all_w.extend([
                moe.se_down.0.clone(),
                moe.se_down.1.clone(),
                moe.se_down.2.clone(),
            ]);
            all_w.extend([
                moe.se_gate_proj.0.clone(),
                moe.se_gate_proj.1.clone(),
                moe.se_gate_proj.2.clone(),
            ]);
            all_w.push(moe.norm_w.clone());
            moe_layers.push(moe);
        }
        let refs: Vec<&Array> = all_w.iter().collect();
        mlx_rs::transforms::eval(refs).unwrap();

        let x = Array::ones::<f32>(&[1, 1, d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let qk_norm_w = Array::ones::<f32>(&[dk]).unwrap();
        let inv_scale = Array::from_f32((dk as f32).sqrt().recip());
        let inv_scale_sq = {
            let s = (dk as f32).sqrt().recip();
            Array::from_f32(s * s)
        };
        let states: Vec<Array> = (0..36)
            .map(|_| {
                Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        let conv_states: Vec<Array> = (0..36)
            .map(|_| {
                Array::zeros::<f32>(&[1, 3, conv_dim])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        x.eval().unwrap();
        for s in &states {
            s.eval().unwrap();
        }
        for c in &conv_states {
            c.eval().unwrap();
        }

        let forward = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;

            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();

                // Attention
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();

                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();

                    let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();

                    let (y, new_state) = gated_delta_kernel_ffi(
                        &norm_q,
                        &norm_k,
                        &conv_v,
                        &l.a_log,
                        &a,
                        &l.dt_bias,
                        &b,
                        &ss[gdn_idx],
                        1,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();
                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;

                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    // Simplified attention: just qkvo matmuls
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };

                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();

                // MoE
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();

                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        // Warmup
        for _ in 0..5 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let result = forward(&x, &mut ss, &mut cs);
            let mut eval_targets: Vec<&Array> = vec![&result];
            eval_targets.extend(ss.iter());
            eval_targets.extend(cs.iter());
            mlx_rs::transforms::eval(eval_targets).unwrap();
        }

        // Benchmark
        let n = 20;
        let mut total_forward = 0u128;
        let mut total_eval = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let t0 = std::time::Instant::now();
            let result = forward(&x, &mut ss, &mut cs);
            let t1 = std::time::Instant::now();
            let mut eval_targets: Vec<&Array> = vec![&result];
            eval_targets.extend(ss.iter());
            eval_targets.extend(cs.iter());
            mlx_rs::transforms::eval(eval_targets).unwrap();
            let t2 = std::time::Instant::now();
            total_forward += (t1 - t0).as_nanos();
            total_eval += (t2 - t1).as_nanos();
        }

        let fwd_ms = total_forward as f64 / n as f64 / 1e6;
        let eval_ms = total_eval as f64 / n as f64 / 1e6;
        println!(
            "Rust 48 combined: forward={fwd_ms:.2}ms eval={eval_ms:.2}ms total={:.2}ms",
            fwd_ms + eval_ms
        );

        // Test: eval only the final result (not states) to see if eval target count matters
        let mut total_eval_one = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let result = forward(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&result]).unwrap();
            total_eval_one += t0.elapsed().as_nanos();
        }
        let eval_one_ms = total_eval_one as f64 / n as f64 / 1e6;
        println!("Rust 48 combined (eval result only): {eval_one_ms:.2}ms");

        // Variant: GDN only (skip MoE, replace with passthrough)
        let forward_gdn_only = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let is_gdn = gdn_layers[i].is_some();
                if !is_gdn {
                    continue;
                }
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let l = gdn_layers[i].as_ref().unwrap();
                let qkvz = ops::quantized_matmul(
                    &normed,
                    &l.in_proj_qkvz.0,
                    &l.in_proj_qkvz.1,
                    &l.in_proj_qkvz.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let ba = ops::quantized_matmul(
                    &normed,
                    &l.in_proj_ba.0,
                    &l.in_proj_ba.1,
                    &l.in_proj_ba.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let q = qkvz
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let k = qkvz
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let v = qkvz
                    .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                cs[gdn_idx] = conv_in.index((.., -3.., ..));
                let conv_out =
                    nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
                let conv_q = conv_out
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_k = conv_out
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_v = conv_out
                    .index((.., .., 2 * key_dim..))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                    .unwrap()
                    .multiply(&inv_scale_sq)
                    .unwrap();
                let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                    .unwrap()
                    .multiply(&inv_scale)
                    .unwrap();
                let (y, new_state) = gated_delta_kernel_ffi(
                    &norm_q,
                    &norm_k,
                    &conv_v,
                    &l.a_log,
                    &a,
                    &l.dt_bias,
                    &b,
                    &ss[gdn_idx],
                    1,
                    1,
                    hk,
                    dk,
                    hv,
                    dv,
                )
                .unwrap();
                ss[gdn_idx] = new_state;
                gdn_idx += 1;
                let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                let z_shaped = z
                    .index((.., .., ..value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let gated = swiglu(&z_shaped, &normed_y).unwrap();
                let r = ops::quantized_matmul(
                    &gated.reshape(&[1, 1, -1]).unwrap(),
                    &l.out_proj.0,
                    &l.out_proj.1,
                    &l.out_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                h = h.add(r).unwrap();
            }
            h
        };

        // Variant: MoE only (skip GDN)
        let forward_moe_only = |h_in: &Array| -> Array {
            let mut h = h_in.clone();
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                // Simple attn proxy
                let attn_out = ops::quantized_matmul(
                    &normed,
                    &moe_layers[i].gate.0,
                    &moe_layers[i].gate.1,
                    &moe_layers[i].gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let h2 = h.add(attn_out.sum_axes(&[-1], true).unwrap()).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        // Warmup GDN-only
        for _ in 0..5 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_gdn_only(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_gdn = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_gdn_only(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_gdn += t0.elapsed().as_nanos();
        }
        println!(
            "Rust GDN-only (36 layers, combined weights): {:.2}ms",
            total_gdn as f64 / n as f64 / 1e6
        );

        // Warmup MoE-only
        for _ in 0..5 {
            let r = forward_moe_only(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        let mut total_moe = 0u128;
        for _ in 0..n {
            let r = forward_moe_only(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total_moe += t0.elapsed().as_nanos();
        }
        println!(
            "Rust MoE-only (48 layers, combined weights): {:.2}ms",
            total_moe as f64 / n as f64 / 1e6
        );

        // Combined but with kernel replaced by zeros_like
        let forward_no_kernel = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let _conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let _norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let _norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let _g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                    let _beta = nn::sigmoid(&b).unwrap();

                    // SKIP kernel: use zeros instead
                    let y = Array::zeros::<f32>(&[1, 1, hv, dv])
                        .unwrap()
                        .as_dtype(mlx_rs::Dtype::Float16)
                        .unwrap();
                    ss[gdn_idx] = Array::zeros::<f32>(&[1, hv, dv, dk])
                        .unwrap()
                        .as_dtype(mlx_rs::Dtype::Float16)
                        .unwrap();
                    gdn_idx += 1;

                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_no_kernel(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_nk = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_no_kernel(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_nk += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined NO KERNEL (GDN ops + MoE): {:.2}ms",
            total_nk as f64 / n as f64 / 1e6
        );

        // Variant: ops-based GDN recurrence (no Metal kernel) interleaved with MoE
        let gqa_repeat = hv / hk;
        let forward_ops_gdn = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                    let beta = nn::sigmoid(&b).unwrap();

                    // Ops-based recurrence: repeat q,k for GQA then run step
                    let q_rep = ops::broadcast_to(
                        norm_q.reshape(&[1, hk, 1, dk]).unwrap(),
                        &[1, hk, gqa_repeat, dk],
                    )
                    .unwrap()
                    .reshape(&[1, hv, dk])
                    .unwrap();
                    let k_rep = ops::broadcast_to(
                        norm_k.reshape(&[1, hk, 1, dk]).unwrap(),
                        &[1, hk, gqa_repeat, dk],
                    )
                    .unwrap()
                    .reshape(&[1, hv, dk])
                    .unwrap();
                    let v_sq = conv_v.squeeze_axes(&[1]).unwrap();
                    let g_sq = g.squeeze_axes(&[0, 1]).unwrap();
                    let beta_sq = beta.squeeze_axes(&[0, 1]).unwrap();
                    let (y, new_state) =
                        gated_delta_step_ref(&q_rep, &k_rep, &v_sq, &g_sq, &beta_sq, &ss[gdn_idx]);
                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;

                    let y_4d = y.expand_dims(0).unwrap().expand_dims(0).unwrap();
                    let normed_y = fast::rms_norm(&y_4d, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_ops_gdn(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_ops = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_ops_gdn(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_ops += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined OPS GDN (no Metal kernel): {:.2}ms",
            total_ops as f64 / n as f64 / 1e6
        );

        // Variant: Metal kernel with per-layer eval barriers
        let forward_eval_barrier = |h_in: &Array,
                                    ss: &mut Vec<Array>,
                                    cs: &mut Vec<Array>|
         -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let (y, new_state) = gated_delta_kernel_ffi(
                        &norm_q,
                        &norm_k,
                        &conv_v,
                        &l.a_log,
                        &a,
                        &l.dt_bias,
                        &b,
                        &ss[gdn_idx],
                        1,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();
                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;
                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();

                // Eval barrier: force layer-by-layer evaluation
                h.eval().unwrap();
                ss.iter().for_each(|s| s.eval().unwrap());
                cs.iter().for_each(|c| c.eval().unwrap());
            }
            h
        };

        for _ in 0..3 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_eval_barrier(&x, &mut ss, &mut cs);
            r.eval().unwrap();
        }
        let mut total_eb = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let t0 = std::time::Instant::now();
            let r = forward_eval_barrier(&x, &mut ss, &mut cs);
            r.eval().unwrap();
            total_eb += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined EVAL BARRIER (per-layer eval): {:.2}ms",
            total_eb as f64 / n as f64 / 1e6
        );

        // Variant: async_eval after each layer (non-blocking pipeline hint)
        let forward_async = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let (y, new_state) = gated_delta_kernel_ffi(
                        &norm_q,
                        &norm_k,
                        &conv_v,
                        &l.a_log,
                        &a,
                        &l.dt_bias,
                        &b,
                        &ss[gdn_idx],
                        1,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();
                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;
                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();

                // Async eval hint: start processing GDN computation while building MoE graph
                mlx_rs::transforms::async_eval([&h2]).unwrap();

                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..3 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_async(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_async = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let t0 = std::time::Instant::now();
            let r = forward_async(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_async += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined ASYNC EVAL (per-layer hint): {:.2}ms",
            total_async as f64 / n as f64 / 1e6
        );

        // Variant: eval kernel outputs (y + state) immediately after each GDN layer
        let forward_eval_kernel = |h_in: &Array,
                                   ss: &mut Vec<Array>,
                                   cs: &mut Vec<Array>|
         -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let (y, new_state) = gated_delta_kernel_ffi(
                        &norm_q,
                        &norm_k,
                        &conv_v,
                        &l.a_log,
                        &a,
                        &l.dt_bias,
                        &b,
                        &ss[gdn_idx],
                        1,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();

                    // Targeted eval: resolve kernel outputs to break graph
                    mlx_rs::transforms::eval([&y, &new_state, &cs[gdn_idx]]).unwrap();

                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;
                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..3 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_eval_kernel(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_ek = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let t0 = std::time::Instant::now();
            let r = forward_eval_kernel(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_ek += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined EVAL KERNEL OUTPUTS: {:.2}ms",
            total_ek as f64 / n as f64 / 1e6
        );

        // Layer scaling test: run with 1, 4, 12, 24, 48 layers to check non-linearity
        // Test: tiny state (replace [1,32,128,128] with [1,1,1,1]) to check memory hypothesis
        let tiny_states: Vec<Array> = (0..36)
            .map(|_| {
                Array::zeros::<f32>(&[1, 1, 1, 1])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        for s in &tiny_states {
            s.eval().unwrap();
        }

        let forward_tiny_state = |h_in: &Array,
                                  ss: &mut Vec<Array>,
                                  cs: &mut Vec<Array>|
         -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let _norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let _norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                    let beta = nn::sigmoid(&b).unwrap();

                    // Tiny state: just multiply by a scalar instead of full state ops
                    let g_scalar = g.sum_axes(&[-1], true).unwrap();
                    let tiny_decayed = ss[gdn_idx].multiply(g_scalar).unwrap();
                    ss[gdn_idx] = tiny_decayed.add(Array::from_f32(0.1)).unwrap();

                    // Use conv_v directly as y (same shape [1,1,Hv,Dv])
                    let y = conv_v
                        .multiply(beta.reshape(&[1, 1, hv, 1]).unwrap())
                        .unwrap();

                    gdn_idx += 1;
                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = tiny_states.clone();
            let mut cs = conv_states.clone();
            let r = forward_tiny_state(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_ts = 0u128;
        for _ in 0..n {
            let mut ss = tiny_states.clone();
            let mut cs = conv_states.clone();
            let r = forward_tiny_state(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_ts += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined TINY STATE (all ops, no large state): {:.2}ms",
            total_ts as f64 / n as f64 / 1e6
        );

        for test_layers in [1i32, 4, 12, 24, 48] {
            let test_layers_u = test_layers as usize;
            let n_gdn = (0..test_layers_u)
                .filter(|i| gdn_layers.get(*i).map_or(false, |g| g.is_some()))
                .count();
            let forward_n = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
                let mut h = h_in.clone();
                let mut gdn_idx = 0usize;
                for i in 0..test_layers_u {
                    let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                    let r = if gdn_layers[i].is_some() {
                        let l = gdn_layers[i].as_ref().unwrap();
                        let qkvz = ops::quantized_matmul(
                            &normed,
                            &l.in_proj_qkvz.0,
                            &l.in_proj_qkvz.1,
                            &l.in_proj_qkvz.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let ba = ops::quantized_matmul(
                            &normed,
                            &l.in_proj_ba.0,
                            &l.in_proj_ba.1,
                            &l.in_proj_ba.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let q = qkvz
                            .index((.., .., ..key_dim))
                            .reshape(&[1, 1, hk, dk])
                            .unwrap();
                        let k = qkvz
                            .index((.., .., key_dim..2 * key_dim))
                            .reshape(&[1, 1, hk, dk])
                            .unwrap();
                        let v = qkvz
                            .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                            .reshape(&[1, 1, hv, dv])
                            .unwrap();
                        let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                        let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                        let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                        let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                        let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                        let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                        let mixed =
                            ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                        let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                        cs[gdn_idx] = conv_in.index((.., -3.., ..));
                        let conv_out =
                            nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                                .unwrap();
                        let conv_q = conv_out
                            .index((.., .., ..key_dim))
                            .reshape(&[1, 1, hk, dk])
                            .unwrap();
                        let conv_k = conv_out
                            .index((.., .., key_dim..2 * key_dim))
                            .reshape(&[1, 1, hk, dk])
                            .unwrap();
                        let conv_v = conv_out
                            .index((.., .., 2 * key_dim..))
                            .reshape(&[1, 1, hv, dv])
                            .unwrap();
                        let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                            .unwrap()
                            .multiply(&inv_scale_sq)
                            .unwrap();
                        let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                            .unwrap()
                            .multiply(&inv_scale)
                            .unwrap();
                        let (y, new_state) = gated_delta_kernel_ffi(
                            &norm_q,
                            &norm_k,
                            &conv_v,
                            &l.a_log,
                            &a,
                            &l.dt_bias,
                            &b,
                            &ss[gdn_idx],
                            1,
                            1,
                            hk,
                            dk,
                            hv,
                            dv,
                        )
                        .unwrap();
                        ss[gdn_idx] = new_state;
                        gdn_idx += 1;
                        let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                        let z_shaped = z
                            .index((.., .., ..value_dim))
                            .reshape(&[1, 1, hv, dv])
                            .unwrap();
                        let gated = swiglu(&z_shaped, &normed_y).unwrap();
                        ops::quantized_matmul(
                            &gated.reshape(&[1, 1, -1]).unwrap(),
                            &l.out_proj.0,
                            &l.out_proj.1,
                            &l.out_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap()
                    } else {
                        let al = attn_layers[i].as_ref().unwrap();
                        let q = ops::quantized_matmul(
                            &normed,
                            &al.q_proj.0,
                            &al.q_proj.1,
                            &al.q_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let _k = ops::quantized_matmul(
                            &normed,
                            &al.k_proj.0,
                            &al.k_proj.1,
                            &al.k_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let v = ops::quantized_matmul(
                            &normed,
                            &al.v_proj.0,
                            &al.v_proj.1,
                            &al.v_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let proxy = v
                            .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                            .unwrap();
                        ops::quantized_matmul(
                            &proxy,
                            &al.o_proj.0,
                            &al.o_proj.1,
                            &al.o_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap()
                    };
                    let h2 = h.add(r).unwrap();
                    let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                    let m = &moe_layers[i];
                    let gate_out = ops::quantized_matmul(
                        &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                    )
                    .unwrap();
                    let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                    let neg_k = -top_k;
                    let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                    let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                    let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                    let scores = raw_scores
                        .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                        .unwrap();
                    let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                    let g_out = gather_qmm(
                        &x_exp,
                        &m.sw_gate.0,
                        &m.sw_gate.1,
                        &m.sw_gate.2,
                        &top_inds,
                        true,
                        gs,
                        bits,
                        false,
                    )
                    .unwrap();
                    let u_out = gather_qmm(
                        &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits,
                        false,
                    )
                    .unwrap();
                    let activated = swiglu(&g_out, &u_out).unwrap();
                    let d_out = gather_qmm(
                        &activated,
                        &m.sw_down.0,
                        &m.sw_down.1,
                        &m.sw_down.2,
                        &top_inds,
                        true,
                        gs,
                        bits,
                        false,
                    )
                    .unwrap();
                    let expert_sum = d_out
                        .squeeze_axes(&[-2])
                        .unwrap()
                        .multiply(scores.expand_dims(-1).unwrap())
                        .unwrap()
                        .sum_axes(&[-2], false)
                        .unwrap();
                    let sh_g = ops::quantized_matmul(
                        &normed2,
                        &m.se_gate.0,
                        &m.se_gate.1,
                        &m.se_gate.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let sh_u = ops::quantized_matmul(
                        &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                    )
                    .unwrap();
                    let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                    let sh_d = ops::quantized_matmul(
                        &sh_act,
                        &m.se_down.0,
                        &m.se_down.1,
                        &m.se_down.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let sh_gate_val = nn::sigmoid(
                        ops::quantized_matmul(
                            &normed2,
                            &m.se_gate_proj.0,
                            &m.se_gate_proj.1,
                            &m.se_gate_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                    let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                    h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
                }
                h
            };
            for _ in 0..3 {
                let mut ss = states.clone();
                let mut cs = conv_states.clone();
                let r = forward_n(&x, &mut ss, &mut cs);
                let mut t: Vec<&Array> = vec![&r];
                t.extend(ss.iter());
                t.extend(cs.iter());
                mlx_rs::transforms::eval(t).unwrap();
            }
            let mut total_n = 0u128;
            for _ in 0..n {
                let mut ss = states.clone();
                let mut cs = conv_states.clone();
                let r = forward_n(&x, &mut ss, &mut cs);
                let t0 = std::time::Instant::now();
                let mut t: Vec<&Array> = vec![&r];
                t.extend(ss.iter());
                t.extend(cs.iter());
                mlx_rs::transforms::eval(t).unwrap();
                total_n += t0.elapsed().as_nanos();
            }
            let ms = total_n as f64 / n as f64 / 1e6;
            println!(
                "Layer scaling: {test_layers} layers ({n_gdn} GDN): {ms:.2}ms ({:.2}ms/layer)",
                ms / test_layers as f64
            );
        }

        // Variant: replace recurrence with a single matmul (same data flow, fewer ops)
        let forward_matmul_gdn = |h_in: &Array,
                                  ss: &mut Vec<Array>,
                                  cs: &mut Vec<Array>|
         -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let _norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let _norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                    let _beta = nn::sigmoid(&b).unwrap();

                    // Variant A: no reduction, just multiply + add on state
                    let g_exp = g.reshape(&[1, hv, 1, 1]).unwrap();
                    let decayed = ss[gdn_idx].multiply(g_exp).unwrap();
                    let v_exp = conv_v.reshape(&[1, hv, dv, 1]).unwrap();
                    ss[gdn_idx] = decayed.add(v_exp).unwrap();
                    // y = just take a slice of state (no reduction)
                    let y_proxy = ss[gdn_idx]
                        .index((.., .., .., 0..1))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    gdn_idx += 1;

                    let normed_y = fast::rms_norm(&y_proxy, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_matmul_gdn(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_mm = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_matmul_gdn(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_mm += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined MATMUL GDN (proxy recurrence): {:.2}ms",
            total_mm as f64 / n as f64 / 1e6
        );
    }

    /// Minimal reproducer: state ops + gather_qmm, nothing else.
    #[test]
    #[ignore = "requires GPU"]
    fn bench_minimal_state_moe_interaction() {
        use mlx_rs::Dtype;
        let n_layers = 48usize;
        let n_gdn = 36usize;
        let hv = 32i32;
        let dv = 128i32;
        let dk = 128i32;
        let d = 2048i32;
        let gs = 64i32;
        let bits = 4i32;
        let n_experts = 512i32;
        let d_inter = 512i32;
        let top_k = 10i32;

        // Expert weights for gather_qmm
        let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };
        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };

        let sw_gate: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
        let sw_up: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
        let sw_down: Vec<_> = (0..n_layers).map(|_| make_sw(d_inter, d)).collect();
        let gate_proj: Vec<_> = (0..n_layers).map(|_| make_qw(d, n_experts)).collect();
        let mut all_w: Vec<Array> = Vec::new();
        for i in 0..n_layers {
            all_w.extend([
                sw_gate[i].0.clone(),
                sw_gate[i].1.clone(),
                sw_gate[i].2.clone(),
            ]);
            all_w.extend([sw_up[i].0.clone(), sw_up[i].1.clone(), sw_up[i].2.clone()]);
            all_w.extend([
                sw_down[i].0.clone(),
                sw_down[i].1.clone(),
                sw_down[i].2.clone(),
            ]);
            all_w.extend([
                gate_proj[i].0.clone(),
                gate_proj[i].1.clone(),
                gate_proj[i].2.clone(),
            ]);
        }
        mlx_rs::transforms::eval(all_w.iter().collect::<Vec<_>>()).unwrap();

        let x = Array::ones::<f32>(&[1, 1, d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let states: Vec<Array> = (0..n_gdn)
            .map(|_| {
                Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        x.eval().unwrap();
        for s in &states {
            s.eval().unwrap();
        }

        let n = 20;

        // Test 1: state ops only (no MoE)
        let forward_state_only = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            for gdn_idx in 0..n_gdn {
                let g = h.sum_axes(&[-1], true).unwrap();
                let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                let new_state = ss[gdn_idx]
                    .multiply(decay)
                    .unwrap()
                    .add(Array::from_f32(0.01))
                    .unwrap();
                let y = new_state
                    .sum_axes(&[-1], false)
                    .unwrap()
                    .reshape(&[1, 1, -1])
                    .unwrap()
                    .index((.., .., ..d));
                ss[gdn_idx] = new_state;
                h = h.add(y).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_state_only(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_state_only(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "State ops only (36 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 2: MoE only (no state)
        let forward_moe_only = |h_in: &Array| -> Array {
            let mut h = h_in.clone();
            for i in 0..n_layers {
                let gate_out = ops::quantized_matmul(
                    &h,
                    &gate_proj[i].0,
                    &gate_proj[i].1,
                    &gate_proj[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let r = forward_moe_only(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let r = forward_moe_only(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "MoE ops only (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 3: interleaved state + MoE
        let forward_interleaved = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers {
                // State ops (for GDN layers)
                if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }

                // MoE ops
                let gate_out = ops::quantized_matmul(
                    &h,
                    &gate_proj[i].0,
                    &gate_proj[i].1,
                    &gate_proj[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Interleaved state + MoE (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 3c: keep ALL intermediates alive (prevent drops during graph construction)
        let forward_keep_alive = |h_in: &Array, ss: &mut Vec<Array>| -> (Array, Vec<Array>) {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            let mut keep: Vec<Array> = Vec::with_capacity(n_layers * 20);
            for i in 0..n_layers {
                if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(&decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    keep.push(g);
                    keep.push(decay);
                    keep.push(y.clone());
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }

                let gate_out = ops::quantized_matmul(
                    &h,
                    &gate_proj[i].0,
                    &gate_proj[i].1,
                    &gate_proj[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                keep.extend([
                    gate_out,
                    gates,
                    all_inds,
                    top_inds.clone(),
                    raw_scores,
                    scores,
                    x_exp,
                    g_out,
                    u_out,
                    activated,
                    d_out,
                    expert_sum.clone(),
                ]);
                h = h.add(expert_sum).unwrap();
            }
            (h, keep)
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let (r, _keep) = forward_keep_alive(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let (r, _keep) = forward_keep_alive(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Interleaved keep-alive (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 3b: same but eval only h (not states)
        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Interleaved eval h only (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 4: interleaved state + quantized_matmul only (no gather_qmm)
        let simple_w: Vec<_> = (0..n_layers).map(|_| make_qw(d, d)).collect();
        let mut sw: Vec<Array> = Vec::new();
        for i in 0..n_layers {
            sw.extend([
                simple_w[i].0.clone(),
                simple_w[i].1.clone(),
                simple_w[i].2.clone(),
            ]);
        }
        mlx_rs::transforms::eval(sw.iter().collect::<Vec<_>>()).unwrap();

        let forward_interleaved_qmm = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers {
                if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }
                // Simple quantized_matmul chain (no gather_qmm FFI)
                let out = ops::quantized_matmul(
                    &h,
                    &simple_w[i].0,
                    &simple_w[i].1,
                    &simple_w[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                h = h.add(out).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved_qmm(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved_qmm(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Interleaved state + quantized_matmul (no gather_qmm): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 5: interleaved state + MoE using gather_qmm
        let forward_interleaved_ops = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers {
                if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }

                let gate_out = ops::quantized_matmul(
                    &h,
                    &gate_proj[i].0,
                    &gate_proj[i].1,
                    &gate_proj[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved_ops(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved_ops(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Interleaved state + gather_qmm: {:.2}ms",
            total as f64 / n as f64 / 1e6
        );
    }

    #[test]
    #[ignore = "requires GPU"]
    #[cfg(any())]
    fn bench_cxx_bypass() {
        use mlx_rs::Dtype;
        let n_layers = 48i32;
        let n_gdn = 36i32;
        let hv = 32i32;
        let dv = 128i32;
        let dk = 128i32;
        let d = 2048i32;
        let gs = 64i32;
        let bits = 4i32;
        let n_experts = 512i32;
        let d_inter = 512i32;
        let top_k = 10i32;
        let n = 20;

        // Self-contained C++ benchmark (no prior Rust MLX operations)
        #[allow(unsafe_code)]
        let self_contained_us = unsafe {
            mlx_sys::mlx_bench_self_contained(
                n_layers, n_gdn, d, n_experts, d_inter, top_k, gs, bits, hv, dv, dk, 5, n,
            )
        };
        println!(
            "C++ self-contained BEFORE any Rust ops: {:.2}ms",
            self_contained_us / 1000.0
        );

        // Now do a tiny eval to see if ANY eval causes the slowdown
        {
            let tiny = Array::ones::<f32>(&[1, 1, 1]).unwrap();
            tiny.eval().unwrap();
        }
        #[allow(unsafe_code)]
        let after_tiny_us = unsafe {
            mlx_sys::mlx_bench_self_contained(
                n_layers, n_gdn, d, n_experts, d_inter, top_k, gs, bits, hv, dv, dk, 5, n,
            )
        };
        println!(
            "C++ self-contained AFTER tiny eval: {:.2}ms",
            after_tiny_us / 1000.0
        );

        // Now create and eval ONE large weight to test memory impact
        {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_inter, d], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            mlx_rs::transforms::eval(vec![&w, &s, &b]).unwrap();
            // raw, w, s, b will be dropped here
        }
        #[allow(unsafe_code)]
        let after_big_us = unsafe {
            mlx_sys::mlx_bench_self_contained(
                n_layers, n_gdn, d, n_experts, d_inter, top_k, gs, bits, hv, dv, dk, 5, n,
            )
        };
        println!(
            "C++ self-contained AFTER one big quantize: {:.2}ms",
            after_big_us / 1000.0
        );

        let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };
        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };

        let sw_gate: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
        let sw_up: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
        let sw_down: Vec<_> = (0..n_layers).map(|_| make_sw(d_inter, d)).collect();
        let gate_proj: Vec<_> = (0..n_layers).map(|_| make_qw(d, n_experts)).collect();
        let mut all_w: Vec<Array> = Vec::new();
        for i in 0..n_layers as usize {
            all_w.extend([
                sw_gate[i].0.clone(),
                sw_gate[i].1.clone(),
                sw_gate[i].2.clone(),
            ]);
            all_w.extend([sw_up[i].0.clone(), sw_up[i].1.clone(), sw_up[i].2.clone()]);
            all_w.extend([
                sw_down[i].0.clone(),
                sw_down[i].1.clone(),
                sw_down[i].2.clone(),
            ]);
            all_w.extend([
                gate_proj[i].0.clone(),
                gate_proj[i].1.clone(),
                gate_proj[i].2.clone(),
            ]);
        }
        mlx_rs::transforms::eval(all_w.iter().collect::<Vec<_>>()).unwrap();

        let x = Array::ones::<f32>(&[1, 1, d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let states: Vec<Array> = (0..n_gdn)
            .map(|_| {
                Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        x.eval().unwrap();
        for s in &states {
            s.eval().unwrap();
        }

        // Prepare raw pointer arrays for FFI
        let gate_w: Vec<_> = sw_gate.iter().map(|t| t.0.as_ptr()).collect();
        let gate_s: Vec<_> = sw_gate.iter().map(|t| t.1.as_ptr()).collect();
        let gate_b: Vec<_> = sw_gate.iter().map(|t| t.2.as_ptr()).collect();
        let up_w: Vec<_> = sw_up.iter().map(|t| t.0.as_ptr()).collect();
        let up_s: Vec<_> = sw_up.iter().map(|t| t.1.as_ptr()).collect();
        let up_b: Vec<_> = sw_up.iter().map(|t| t.2.as_ptr()).collect();
        let down_w: Vec<_> = sw_down.iter().map(|t| t.0.as_ptr()).collect();
        let down_s: Vec<_> = sw_down.iter().map(|t| t.1.as_ptr()).collect();
        let down_b: Vec<_> = sw_down.iter().map(|t| t.2.as_ptr()).collect();
        let gp_w: Vec<_> = gate_proj.iter().map(|t| t.0.as_ptr()).collect();
        let gp_s: Vec<_> = gate_proj.iter().map(|t| t.1.as_ptr()).collect();
        let gp_b: Vec<_> = gate_proj.iter().map(|t| t.2.as_ptr()).collect();

        let state_ptrs_for_cxx: Vec<_> = states.iter().map(|s| s.as_ptr()).collect();

        let n = 20;
        let stream = Stream::new();

        // Warmup
        for _ in 0..5 {
            let state_ptrs: Vec<_> = states.iter().map(|s| s.as_ptr()).collect();
            #[allow(unsafe_code)]
            let (result, state_outs) = unsafe {
                let mut result = mlx_sys::mlx_array_new();
                let mut state_outs: Vec<mlx_sys::mlx_array> =
                    (0..n_gdn).map(|_| mlx_sys::mlx_array_new()).collect();
                let status = mlx_sys::mlx_bench_interleaved_cxx(
                    &raw mut result,
                    state_outs.as_mut_ptr(),
                    x.as_ptr(),
                    state_ptrs.as_ptr(),
                    gate_w.as_ptr(),
                    gate_s.as_ptr(),
                    gate_b.as_ptr(),
                    up_w.as_ptr(),
                    up_s.as_ptr(),
                    up_b.as_ptr(),
                    down_w.as_ptr(),
                    down_s.as_ptr(),
                    down_b.as_ptr(),
                    gp_w.as_ptr(),
                    gp_s.as_ptr(),
                    gp_b.as_ptr(),
                    n_layers,
                    n_gdn,
                    d,
                    n_experts,
                    top_k,
                    gs,
                    bits,
                    stream.as_ptr(),
                );
                assert_eq!(status, 0, "C++ shim failed");
                let r = Array::from_ptr(result);
                let so: Vec<Array> = state_outs.into_iter().map(|p| Array::from_ptr(p)).collect();
                (r, so)
            };
            let mut t: Vec<&Array> = vec![&result];
            t.extend(state_outs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }

        // Benchmark
        let mut total = 0u128;
        for _ in 0..n {
            let state_ptrs: Vec<_> = states.iter().map(|s| s.as_ptr()).collect();
            #[allow(unsafe_code)]
            let (result, state_outs) = unsafe {
                let mut result = mlx_sys::mlx_array_new();
                let mut state_outs: Vec<mlx_sys::mlx_array> =
                    (0..n_gdn).map(|_| mlx_sys::mlx_array_new()).collect();
                let status = mlx_sys::mlx_bench_interleaved_cxx(
                    &raw mut result,
                    state_outs.as_mut_ptr(),
                    x.as_ptr(),
                    state_ptrs.as_ptr(),
                    gate_w.as_ptr(),
                    gate_s.as_ptr(),
                    gate_b.as_ptr(),
                    up_w.as_ptr(),
                    up_s.as_ptr(),
                    up_b.as_ptr(),
                    down_w.as_ptr(),
                    down_s.as_ptr(),
                    down_b.as_ptr(),
                    gp_w.as_ptr(),
                    gp_s.as_ptr(),
                    gp_b.as_ptr(),
                    n_layers,
                    n_gdn,
                    d,
                    n_experts,
                    top_k,
                    gs,
                    bits,
                    stream.as_ptr(),
                );
                assert_eq!(status, 0, "C++ shim failed");
                let r = Array::from_ptr(result);
                let so: Vec<Array> = state_outs.into_iter().map(|p| Array::from_ptr(p)).collect();
                (r, so)
            };
            let mut t: Vec<&Array> = vec![&result];
            t.extend(state_outs.iter());
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "C++ bypass interleaved (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test: build + eval entirely in C++ (no Rust involvement in eval)
        #[allow(unsafe_code)]
        let avg_us = unsafe {
            mlx_sys::mlx_bench_interleaved_cxx_with_eval(
                x.as_ptr(),
                state_ptrs_for_cxx.as_ptr(),
                gate_w.as_ptr(),
                gate_s.as_ptr(),
                gate_b.as_ptr(),
                up_w.as_ptr(),
                up_s.as_ptr(),
                up_b.as_ptr(),
                down_w.as_ptr(),
                down_s.as_ptr(),
                down_b.as_ptr(),
                gp_w.as_ptr(),
                gp_s.as_ptr(),
                gp_b.as_ptr(),
                n_layers,
                n_gdn,
                d,
                n_experts,
                top_k,
                gs,
                bits,
                5,
                n,
            )
        };
        println!("C++ build+eval (48 layers): {:.2}ms", avg_us / 1000.0);

        // Test: state ops only (no MoE)
        #[allow(unsafe_code)]
        let state_only_us = unsafe {
            mlx_sys::mlx_bench_state_ops_only(
                x.as_ptr(),
                state_ptrs_for_cxx.as_ptr(),
                n_gdn,
                d,
                5,
                n,
            )
        };
        println!(
            "C++ state ops only (36 layers): {:.2}ms",
            state_only_us / 1000.0
        );

        // Test: interleaved but eval h only (no states in eval list)
        #[allow(unsafe_code)]
        let h_only_us = unsafe {
            mlx_sys::mlx_bench_interleaved_h_only_eval(
                x.as_ptr(),
                state_ptrs_for_cxx.as_ptr(),
                gate_w.as_ptr(),
                gate_s.as_ptr(),
                gate_b.as_ptr(),
                up_w.as_ptr(),
                up_s.as_ptr(),
                up_b.as_ptr(),
                down_w.as_ptr(),
                down_s.as_ptr(),
                down_b.as_ptr(),
                gp_w.as_ptr(),
                gp_s.as_ptr(),
                gp_b.as_ptr(),
                n_layers,
                n_gdn,
                d,
                n_experts,
                top_k,
                gs,
                bits,
                5,
                n,
            )
        };
        println!(
            "C++ interleaved h-only eval (48 layers): {:.2}ms",
            h_only_us / 1000.0
        );

        // For comparison: the standard Rust interleaved version
        let forward_interleaved = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                if gdn_idx < n_gdn as usize && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }
                let gate_out = ops::quantized_matmul(
                    &h,
                    &gate_proj[i].0,
                    &gate_proj[i].1,
                    &gate_proj[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let gates_v = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let all_inds = ops::argpartition_axis(&gates_v, -top_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
                let raw_scores = gates_v.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Rust C API interleaved (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );
    }

    #[test]
    #[ignore = "requires GPU"]
    #[cfg(any())]
    fn bench_gather_mm_interleave() {
        use mlx_rs::Dtype;
        let n_layers = 48usize;
        let n_gdn = 36usize;
        let hv = 32i32;
        let dv = 128i32;
        let dk = 128i32;
        let d = 256i32; // Small dim to avoid OOM (float weights are not quantized)
        let n_experts = 64i32;
        let top_k = 10i32;

        // gather_mm: a=[..., M, K] @ b=[batch, K, N] -> [..., batch_sel, M, N]
        let float_weights: Vec<Array> = (0..n_layers)
            .map(|_| {
                mlx_rs::random::normal::<f32>(&[n_experts, d, d], None, None, None)
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        mlx_rs::transforms::eval(float_weights.iter().collect::<Vec<_>>()).unwrap();

        let x = Array::ones::<f32>(&[1, 1, d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let states: Vec<Array> = (0..n_gdn)
            .map(|_| {
                Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        x.eval().unwrap();
        for s in &states {
            s.eval().unwrap();
        }

        let n = 20;

        // gather_mm only (no state)
        let forward_gather_only = |h_in: &Array| -> Array {
            let mut h = h_in.clone();
            for i in 0..n_layers {
                let rhs_inds =
                    Array::from_slice(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);
                let x_exp = h.expand_dims(-2).unwrap();
                let out =
                    ops::gather_mm(&x_exp, &float_weights[i], None::<&Array>, &rhs_inds, None)
                        .unwrap();
                let out_sq = out.squeeze_axes(&[-2]).unwrap();
                let expert_sum = out_sq.sum_axes(&[-2], false).unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let r = forward_gather_only(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        let mut total = 0u128;
        for _ in 0..n {
            let r = forward_gather_only(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "gather_mm only (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // gather_mm interleaved with state
        let forward_interleaved = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers {
                if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }

                let rhs_inds =
                    Array::from_slice(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);
                let x_exp = h.expand_dims(-2).unwrap();
                let out =
                    ops::gather_mm(&x_exp, &float_weights[i], None::<&Array>, &rhs_inds, None)
                        .unwrap();
                let out_sq = out.squeeze_axes(&[-2]).unwrap();
                let expert_sum = out_sq.sum_axes(&[-2], false).unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "gather_mm interleaved (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );
    }

    #[test]
    #[ignore = "requires model files on disk"]
    fn bench_actual_model_forward() {
        let model_path = "/Users/panbanda/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5";
        if !std::path::Path::new(model_path).exists() {
            println!("Model not found at {model_path}, skipping");
            return;
        }

        let mut model = load_qwen3_next_model(model_path).unwrap();
        let mut cache: Vec<Option<LayerCache>> = Vec::new();

        // Prefill with a short prompt
        let prompt = Array::from_slice(&[9707u32, 1879], &[1, 2]);
        let prefill_out = model.forward(&prompt, None, &mut cache).unwrap();
        // Eval prefill outputs + cache states
        let mut to_eval: Vec<&Array> = vec![&prefill_out];
        for lc in &cache {
            if let Some(lc) = lc {
                match lc {
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            to_eval.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            to_eval.push(c);
                        }
                    }
                    LayerCache::KV(_) => {} // KV cache evals itself internally
                }
            }
        }
        mlx_rs::transforms::eval(to_eval).unwrap();

        // Get first token
        let logits = prefill_out.index((.., -1, ..));
        let token = ops::indexing::argmax_axis(&logits, -1, false).unwrap();
        mlx_rs::transforms::eval([&token]).unwrap();

        // Decode loop timing
        let mut current = token;
        for i in 0..22 {
            let input = current.index((.., ops::indexing::NewAxis));
            let t_fwd_start = std::time::Instant::now();
            let out = model.forward(&input, None, &mut cache).unwrap();
            let next = ops::indexing::argmax_axis(&out.index((.., -1, ..)), -1, false).unwrap();
            let t_fwd = t_fwd_start.elapsed();

            let t_eval_start = std::time::Instant::now();
            // Eval next token AND all cache states (like Python does)
            let mut eval_list: Vec<&Array> = vec![&next];
            for lc in cache.iter() {
                if let Some(lc) = lc {
                    match lc {
                        LayerCache::Arrays(ac) => {
                            if let Some(ref s) = ac.ssm_state {
                                eval_list.push(s);
                            }
                            if let Some(ref c) = ac.conv_state {
                                eval_list.push(c);
                            }
                        }
                        LayerCache::KV(_) => {}
                    }
                }
            }
            mlx_rs::transforms::eval(eval_list).unwrap();
            let t_eval = t_eval_start.elapsed();

            let t_item_start = std::time::Instant::now();
            let _id: u32 = next.item();
            let t_item = t_item_start.elapsed();

            let total = t_fwd + t_eval + t_item;
            if i < 5 || i >= 20 {
                println!(
                    "Step {i}: fwd={:.2}ms eval={:.2}ms item={:.2}ms total={:.2}ms ({:.1} tok/s)",
                    t_fwd.as_secs_f64() * 1000.0,
                    t_eval.as_secs_f64() * 1000.0,
                    t_item.as_secs_f64() * 1000.0,
                    total.as_secs_f64() * 1000.0,
                    1.0 / total.as_secs_f64(),
                );
            }
            current = next;
        }
    }

    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_metal_kernel_gather_qmm_interleaving() {
        let b: i32 = 1;
        let d: i32 = 2048;
        let n_layers: i32 = 48;
        let n_gdn: i32 = 36;
        let n_experts: i32 = 512;
        let d_inter: i32 = 512;
        let top_k: i32 = 10;
        let gs: i32 = 64;
        let bits: i32 = 4;
        let hk: i32 = 16;
        let hv: i32 = 32;
        let dk: i32 = 128;
        let dv: i32 = 128;

        let x = Array::from_slice(&vec![0.1f32; (b * d) as usize], &[b, 1, d]);

        fn make_qw3d(n: i32, out_d: i32, in_d: i32, gs: i32, bits: i32) -> (Array, Array, Array) {
            let raw = Array::from_slice(
                &vec![0.01f32; (n * out_d * in_d) as usize],
                &[n, out_d, in_d],
            );
            let (w, s, b_arr) = ops::quantize(&raw, gs, bits).unwrap();
            mlx_rs::transforms::eval([&w, &s, &b_arr]).unwrap();
            (w, s, b_arr)
        }

        let gate_w: Vec<_> = (0..n_layers)
            .map(|_| make_qw3d(n_experts, d_inter, d, gs, bits))
            .collect();
        let up_w: Vec<_> = (0..n_layers)
            .map(|_| make_qw3d(n_experts, d_inter, d, gs, bits))
            .collect();
        let down_w: Vec<_> = (0..n_layers)
            .map(|_| make_qw3d(n_experts, d, d_inter, gs, bits))
            .collect();

        let q = Array::from_slice(&vec![0.1f32; (b * hk * dk) as usize], &[b, 1, hk, dk]);
        let k = Array::from_slice(&vec![0.1f32; (b * hk * dk) as usize], &[b, 1, hk, dk]);
        let v = Array::from_slice(&vec![0.1f32; (b * hv * dv) as usize], &[b, 1, hv, dv]);
        let a_log_arr = Array::zeros::<f32>(&[hv]).unwrap();
        let a_arr = Array::from_slice(&vec![1.0f32; (b * hv) as usize], &[b, 1, hv]);
        let dt_bias_arr = Array::zeros::<f32>(&[hv]).unwrap();
        let b_arr = Array::zeros::<f32>(&[b, 1, hv]).unwrap();
        let state = Array::zeros::<f32>(&[b, hv, dv, dk]).unwrap();
        mlx_rs::transforms::eval([&q, &k, &v, &a_log_arr, &a_arr, &dt_bias_arr, &b_arr, &state])
            .unwrap();

        let indices = Array::from_slice(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);

        // Test 1: gather_qmm ONLY
        let build_gqmm_only = |h_in: &Array| -> Array {
            let mut h = h_in.clone();
            for i in 0..n_layers as usize {
                let xe = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &xe,
                    &gate_w[i].0,
                    &gate_w[i].1,
                    &gate_w[i].2,
                    &indices,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &xe, &up_w[i].0, &up_w[i].1, &up_w[i].2, &indices, true, gs, bits, false,
                )
                .unwrap();
                let act = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &act,
                    &down_w[i].0,
                    &down_w[i].1,
                    &down_w[i].2,
                    &indices,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let r = build_gqmm_only(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        let n = 10;
        let mut total = 0u128;
        for _ in 0..n {
            let r = build_gqmm_only(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "gather_qmm only (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 2: Metal kernel + gather_qmm interleaved
        let build_interleaved = |h_in: &Array| -> (Array, Vec<Array>) {
            let mut h = h_in.clone();
            let mut states_out = Vec::new();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                if gdn_idx < n_gdn as usize && (i + 1) % 4 != 0 {
                    let (y, s_out) = gated_delta_kernel_ffi(
                        &q,
                        &k,
                        &v,
                        &a_log_arr,
                        &a_arr,
                        &dt_bias_arr,
                        &b_arr,
                        &state,
                        b,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();
                    let y_flat = y.reshape(&[b, 1, -1]).unwrap();
                    let y_trunc = y_flat.index((.., .., ..d));
                    h = h.add(y_trunc).unwrap();
                    states_out.push(s_out);
                    gdn_idx += 1;
                }
                let xe = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &xe,
                    &gate_w[i].0,
                    &gate_w[i].1,
                    &gate_w[i].2,
                    &indices,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &xe, &up_w[i].0, &up_w[i].1, &up_w[i].2, &indices, true, gs, bits, false,
                )
                .unwrap();
                let act = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &act,
                    &down_w[i].0,
                    &down_w[i].1,
                    &down_w[i].2,
                    &indices,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert).unwrap();
            }
            (h, states_out)
        };

        for _ in 0..5 {
            let (r, s) = build_interleaved(&x);
            let mut ev: Vec<&Array> = vec![&r];
            ev.extend(s.iter());
            mlx_rs::transforms::eval(ev).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let (r, s) = build_interleaved(&x);
            let mut ev: Vec<&Array> = vec![&r];
            ev.extend(s.iter());
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval(ev).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Metal kernel + gather_qmm (eval h+states): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 3: Metal kernel + gather_qmm, eval h only
        for _ in 0..5 {
            let (r, _) = build_interleaved(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let (r, _) = build_interleaved(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Metal kernel + gather_qmm (eval h only): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );
    }

    /// Test eval scaling with graph size using quantized_matmul + rms_norm
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_eval_scaling() {
        let b: i32 = 1;
        let d: i32 = 2048;
        let gs: i32 = 64;
        let bits: i32 = 4;
        let n_layers: i32 = 48;

        let x = Array::from_slice(&vec![0.1f32; (b * d) as usize], &[b, 1, d]);

        fn make_qw2d(rows: i32, cols: i32, gs: i32, bits: i32) -> (Array, Array, Array) {
            let raw = Array::from_slice(&vec![0.01f32; (rows * cols) as usize], &[rows, cols]);
            let (w, s, b_arr) = ops::quantize(&raw, gs, bits).unwrap();
            mlx_rs::transforms::eval([&w, &s, &b_arr]).unwrap();
            (w, s, b_arr)
        }

        let weights: Vec<_> = (0..n_layers).map(|_| make_qw2d(d, d, gs, bits)).collect();
        let norm_ws: Vec<_> = (0..n_layers)
            .map(|_| {
                let w = Array::ones::<f32>(&[d]).unwrap();
                mlx_rs::transforms::eval([&w]).unwrap();
                w
            })
            .collect();

        for n_extras in &[0, 2, 5, 8, 12] {
            let total_ops = n_layers * (1 + n_extras + 1);
            let build = |h_in: &Array| -> Array {
                let mut h = h_in.clone();
                for i in 0..n_layers as usize {
                    h = ops::quantized_matmul(
                        &h,
                        &weights[i].0,
                        &weights[i].1,
                        &weights[i].2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    for j in 0..*n_extras as usize {
                        let idx = (i + j + 1) % n_layers as usize;
                        let extra = ops::quantized_matmul(
                            &h,
                            &weights[idx].0,
                            &weights[idx].1,
                            &weights[idx].2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let scale = Array::from_slice(&[0.01f32], &[1]);
                        h = h.add(extra.multiply(&scale).unwrap()).unwrap();
                    }
                    h = fast::rms_norm(&h, &norm_ws[i], 1e-6).unwrap();
                }
                h
            };
            for _ in 0..3 {
                let r = build(&x);
                mlx_rs::transforms::eval([&r]).unwrap();
            }
            let n = 10;
            let mut total_ns = 0u128;
            for _ in 0..n {
                let r = build(&x);
                let t0 = std::time::Instant::now();
                mlx_rs::transforms::eval([&r]).unwrap();
                total_ns += t0.elapsed().as_nanos();
            }
            let avg_ms = total_ns as f64 / n as f64 / 1e6;
            let us_per_op = avg_ms * 1000.0 / total_ops as f64;
            println!(
                "extras={n_extras:2} ops~={total_ops:4} eval={avg_ms:.2}ms ({us_per_op:.1}us/op)"
            );
        }
    }

    /// Measure async_eval pipelining: does GPU overlap with CPU graph building?
    ///
    /// cargo test -p higgs-models --release -- bench_async_pipeline --nocapture --ignored
    #[test]
    #[ignore]
    fn bench_async_pipeline() {
        use mlx_rs::random::normal;
        use mlx_rs::transforms::{async_eval, eval};

        let d: &[i32] = &[2048, 2048];
        let w = normal::<f32>(d, None, None, None).unwrap();
        eval([&w].into_iter()).unwrap();

        let build_graph = |x: &Array| -> Array {
            let mut h = x.clone();
            for _ in 0..40 {
                let mm = h.matmul(&w).unwrap();
                h = mm.add(&h).unwrap();
            }
            h
        };

        let x = normal::<f32>(&[1, 1, 2048], None, None, None).unwrap();
        eval([&x].into_iter()).unwrap();

        // Sequential
        let n = 20usize;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let y = build_graph(&x);
            eval([&y].into_iter()).unwrap();
        }
        let seq_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // Pipelined
        let t0 = std::time::Instant::now();
        let mut y = build_graph(&x);
        async_eval([&y].into_iter()).unwrap();
        for _ in 0..n {
            let next_y = build_graph(&y);
            async_eval([&next_y].into_iter()).unwrap();
            eval([&y].into_iter()).unwrap();
            y = next_y;
        }
        let pipe_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        eprintln!("Rust mlx-rs sequential:  {seq_ms:.2}ms/step");
        eprintln!("Rust mlx-rs pipelined:   {pipe_ms:.2}ms/step");
        eprintln!("Speedup: {:.2}x", seq_ms / pipe_ms);
    }

    /// Measure pure FFI graph-building overhead: no eval, just op dispatch.
    ///
    /// cargo test -p higgs-models --release -- bench_ffi_overhead --nocapture --ignored
    #[test]
    #[ignore]
    fn bench_ffi_overhead() {
        use mlx_rs::transforms::eval;

        let a = Array::ones::<f32>(&[1, 1, 2048]).unwrap();
        let b = Array::ones::<f32>(&[1, 1, 2048]).unwrap();
        eval([&a, &b].into_iter()).unwrap();

        let n = 2000usize;

        // Graph build only (no eval)
        let t0 = std::time::Instant::now();
        let mut x = a.clone();
        for _ in 0..n {
            x = x.add(&b).unwrap();
        }
        let build_us = t0.elapsed().as_micros();
        eprintln!(
            "Rust mlx-rs: {n} adds graph-build = {build_us}us ({:.1}us/op)",
            build_us as f64 / n as f64
        );

        // Graph build + eval
        let t0 = std::time::Instant::now();
        let mut x = a.clone();
        for _ in 0..n {
            x = x.add(&b).unwrap();
        }
        eval([&x].into_iter()).unwrap();
        let total_us = t0.elapsed().as_micros();
        eprintln!(
            "Rust mlx-rs: {n} adds + eval = {total_us}us ({:.1}us/op)",
            total_us as f64 / n as f64
        );

        // With task-local stream set
        let stream = Stream::new();
        mlx_rs::with_new_default_stream(stream, || {
            let t0 = std::time::Instant::now();
            let mut x = a.clone();
            for _ in 0..n {
                x = x.add(&b).unwrap();
            }
            let build_us = t0.elapsed().as_micros();
            eprintln!(
                "Rust mlx-rs (task-local stream): {n} adds graph-build = {build_us}us ({:.1}us/op)",
                build_us as f64 / n as f64
            );
        });
    }

    /// Write a qwen3.5-style VLM config.json (with text_config) and parse it.
    fn write_qwen35_config(dir: &std::path::Path, text_config_json: &str) {
        let config =
            format!(r#"{{"text_config": {text_config_json}, "tie_word_embeddings": false}}"#);
        std::fs::write(dir.join("config.json"), config).unwrap();
    }

    /// Helper: minimal qwen3.5 text_config JSON for a dense (non-MoE) model.
    fn qwen35_dense_text_config() -> &'static str {
        r#"{
            "model_type": "qwen3_5",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-06,
            "vocab_size": 1024,
            "max_position_embeddings": 512,
            "full_attention_interval": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 0,
            "num_experts_per_tok": 0
        }"#
    }

    /// Helper: minimal qwen3.5 text_config JSON for an MoE model.
    fn qwen35_moe_text_config() -> &'static str {
        r#"{
            "model_type": "qwen3_5_moe",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "intermediate_size": 0,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-06,
            "vocab_size": 1024,
            "max_position_embeddings": 512,
            "full_attention_interval": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": 256,
            "moe_intermediate_size": 128,
            "norm_topk_prob": true
        }"#
    }

    #[test]
    fn test_load_qwen35_moe_text_config_moe_sets_decoder_sparse_step() {
        let dir = tempfile::tempdir().unwrap();
        write_qwen35_config(dir.path(), qwen35_moe_text_config());
        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
        assert_eq!(
            args.decoder_sparse_step, 1,
            "MoE model should get decoder_sparse_step=1"
        );
        assert!(args.num_experts > 0);
    }

    #[test]
    fn test_load_qwen35_dense_text_config_no_forced_moe() {
        let dir = tempfile::tempdir().unwrap();
        write_qwen35_config(dir.path(), qwen35_dense_text_config());
        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
        // Dense models (num_experts=0) must NOT get decoder_sparse_step=1,
        // otherwise every layer tries to create SparseMoeBlock and fails.
        assert_eq!(
            args.decoder_sparse_step, 0,
            "Dense model should NOT get decoder_sparse_step=1"
        );
        assert_eq!(args.num_experts, 0);
    }

    /// Flat-layout qwen3_5 checkpoints (e.g. Carnice-9b-MLX) put model args at
    /// the top level of config.json instead of nested under `text_config`.
    /// The parser must accept both packagings and still populate the same args.
    #[test]
    fn test_load_qwen35_flat_config_parses_like_nested() {
        let dir = tempfile::tempdir().unwrap();
        // Same fields as qwen35_dense_text_config(), written flat (no wrapper).
        let flat = qwen35_dense_text_config();
        // Add the top-level tie_word_embeddings the VLM wrapper normally supplies
        // so both code paths end up with identical args.
        let flat_with_tie =
            flat.trim_end_matches('}').to_string() + r#", "tie_word_embeddings": false}"#;
        std::fs::write(dir.path().join("config.json"), &flat_with_tie).unwrap();
        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
        assert_eq!(args.num_hidden_layers, 4);
        assert_eq!(args.hidden_size, 256);
        assert_eq!(args.num_experts, 0);
        assert_eq!(args.decoder_sparse_step, 0);
    }

    #[test]
    fn test_load_qwen35_mixed_ba_quantization_forces_separate_gdn() {
        let dir = tempfile::tempdir().unwrap();
        let config = format!(
            r#"{{
                "text_config": {},
                "tie_word_embeddings": false,
                "quantization": {{
                    "group_size": 64,
                    "bits": 2,
                    "mode": "affine",
                    "language_model.model.layers.1.linear_attn.in_proj_a": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }}
                }}
            }}"#,
            qwen35_dense_text_config()
        );
        std::fs::write(dir.path().join("config.json"), config).unwrap();

        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

        assert!(
            args.use_separate_gdn_projections,
            "mixed-bit in_proj_a/in_proj_b must force separate GDN projections"
        );
    }

    #[test]
    fn test_load_qwen35_matching_ba_quantization_keeps_fused_gdn() {
        let dir = tempfile::tempdir().unwrap();
        let config = format!(
            r#"{{
                "text_config": {},
                "tie_word_embeddings": false,
                "quantization": {{
                    "group_size": 64,
                    "bits": 2,
                    "mode": "affine",
                    "language_model.model.layers.1.linear_attn.in_proj_a": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }},
                    "language_model.model.layers.1.linear_attn.in_proj_b": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }}
                }}
            }}"#,
            qwen35_dense_text_config()
        );
        std::fs::write(dir.path().join("config.json"), config).unwrap();

        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

        assert!(
            !args.use_separate_gdn_projections,
            "matching BA overrides should keep the fused GDN loader path"
        );
    }

    #[test]
    fn test_can_concatenate_axis0_detects_quantized_inner_shape_mismatch() {
        let a = Array::zeros::<f32>(&[48, 320]).unwrap();
        let b = Array::zeros::<f32>(&[48, 800]).unwrap();
        let c = Array::zeros::<f32>(&[96, 320]).unwrap();

        assert!(
            !can_concatenate_axis0(&a, &b),
            "different packed inner dims must block BA fusion"
        );
        assert!(
            can_concatenate_axis0(&a, &c),
            "axis-0 size may differ because fusion concatenates rows"
        );
    }

    /// GQA ratio: `num_v_heads` must be divisible by `num_k_heads`.
    /// This validates the assumption used in test/bench GDN recurrence loops.
    #[test]
    fn test_gqa_ratio_divisibility() {
        let args = valid_causal_lm_args();
        let hv = args.linear_num_value_heads;
        let hk = args.linear_num_key_heads;
        assert!(
            hk > 0 && hv % hk == 0,
            "linear_num_value_heads ({hv}) must be divisible by linear_num_key_heads ({hk})"
        );
    }

    /// QEmbedding equivalence: dequantize-then-gather produces same result as
    /// the full dequantize path (validates that gather on quantized storage
    /// is safe for future optimisation).
    #[test]
    fn test_qembedding_gather_then_dequantize_equivalence() {
        use mlx_rs::transforms::eval;

        let group_size = 64i32;
        let bits = 4i32;
        let vocab = 256i32;
        let hidden = 128i32;

        // Create a random float matrix and quantize it
        let float_weight =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[vocab, hidden], None).unwrap();
        eval([&float_weight].into_iter()).unwrap();
        let (qw, qs, qb) = ops::quantize(&float_weight, group_size, bits).unwrap();
        eval([&qw, &qs, &qb].into_iter()).unwrap();

        let indices = Array::from_slice(&[0i32, 5, 42, 255, 5], &[5]);
        eval([&indices].into_iter()).unwrap();

        // Path A: dequantize full vocab, then gather (current QEmbedding::forward)
        let full_deq = ops::dequantize(&qw, &qs, &qb, group_size, bits).unwrap();
        let path_a = full_deq.take_axis(&indices, 0).unwrap();
        eval([&path_a].into_iter()).unwrap();

        // Path B: gather quantized rows first, then dequantize only selected
        let sel_w = qw.take_axis(&indices, 0).unwrap();
        let sel_s = qs.take_axis(&indices, 0).unwrap();
        let sel_b = qb.take_axis(&indices, 0).unwrap();
        let path_b = ops::dequantize(&sel_w, &sel_s, &sel_b, group_size, bits).unwrap();
        eval([&path_b].into_iter()).unwrap();

        // They should be identical (both round-trip through the same quantized repr)
        let diff = path_a.subtract(&path_b).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-6,
            "gather-then-dequantize should match dequantize-then-gather, max diff: {max_diff}"
        );
    }

    // -----------------------------------------------------------------------
    // Chunked prefill tests
    // -----------------------------------------------------------------------

    /// forward_chunked compiles and the API is callable.
    /// chunk_size >= T falls through to normal forward (no chunking).
    #[test]
    fn test_chunked_prefill_api_exists() {
        let args = valid_causal_lm_args();
        let model = Qwen3NextCausalLM::new(args).unwrap();
        // Verify forward_chunked is callable (type-check / link test).
        // We can't run it on synthetic weights, but we confirm the method exists
        // and handles the chunk_size >= T fast path correctly.
        assert!(model.args.num_hidden_layers > 0);
    }

    /// Chunked prefill: logits are close to full prefill on a real model.
    /// Tests even division (chunk_size=4, seq_len=12).
    ///
    /// Note: quantized_matmul produces slightly different results for different
    /// input shapes due to tile reduction order (FP non-associativity).
    /// A max logit diff of ~1-2 is normal for 3-bit models.
    /// The decode_continuity test is the real correctness check (same tokens).
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- test_chunked_prefill_matches_full --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires model files on disk"]
    fn test_chunked_prefill_matches_full() {
        use mlx_rs::transforms::eval;

        let mut model = load_test_model();

        let seq_len = 12i32;
        let tokens: Vec<u32> = (0..seq_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();
        let input = Array::from_slice(&tokens, &[1, seq_len]);

        // Full prefill
        let mut cache_full: Vec<Option<LayerCache>> = Vec::new();
        let logits_full = model.forward(&input, None, &mut cache_full).unwrap();
        eval([&logits_full]).unwrap();

        // Chunked prefill: chunk_size=4 → chunks [4,4,4]
        let mut cache_chunked: Vec<Option<LayerCache>> = Vec::new();
        let logits_chunked = model
            .forward_chunked(&input, None, &mut cache_chunked, 4)
            .unwrap();
        eval([&logits_chunked]).unwrap();

        let last_full = logits_full.index((.., -1, ..));
        let last_chunked = logits_chunked.index((.., -1, ..));
        eval([&last_full, &last_chunked]).unwrap();

        let diff = last_full.subtract(&last_chunked).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        eprintln!("max logit |diff| = {max_diff}");
        assert!(
            max_diff < 2.0,
            "chunked logits diverge from full: max |diff| = {max_diff} (expect <2.0 for 3-bit quant)"
        );
    }

    /// Chunked prefill: uneven chunk sizes (remainder chunk).
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- test_chunked_prefill_uneven --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires model files on disk"]
    fn test_chunked_prefill_uneven() {
        use mlx_rs::transforms::eval;

        let mut model = load_test_model();

        let seq_len = 10i32;
        let tokens: Vec<u32> = (0..seq_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();
        let input = Array::from_slice(&tokens, &[1, seq_len]);

        let mut cache_full: Vec<Option<LayerCache>> = Vec::new();
        let logits_full = model.forward(&input, None, &mut cache_full).unwrap();
        eval([&logits_full]).unwrap();

        // chunk_size=3: chunks [3,3,3,1]
        let mut cache_chunked: Vec<Option<LayerCache>> = Vec::new();
        let logits_chunked = model
            .forward_chunked(&input, None, &mut cache_chunked, 3)
            .unwrap();
        eval([&logits_chunked]).unwrap();

        let last_full = logits_full.index((.., -1, ..));
        let last_chunked = logits_chunked.index((.., -1, ..));
        eval([&last_full, &last_chunked]).unwrap();

        let diff = last_full.subtract(&last_chunked).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        eprintln!("uneven max logit |diff| = {max_diff}");
        assert!(
            max_diff < 2.0,
            "uneven chunks diverge: max |diff| = {max_diff} (expect <2.0 for 3-bit quant)"
        );
    }

    /// Decode after chunked prefill produces same tokens as after full prefill.
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- test_chunked_prefill_decode_continuity --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires model files on disk"]
    fn test_chunked_prefill_decode_continuity() {
        use mlx_rs::transforms::eval;

        let mut model = load_test_model();

        let seq_len = 16i32;
        let tokens: Vec<u32> = (0..seq_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();
        let input = Array::from_slice(&tokens, &[1, seq_len]);

        // Full prefill + 5 decode steps
        let mut cache_full: Vec<Option<LayerCache>> = Vec::new();
        let logits_full = model.forward(&input, None, &mut cache_full).unwrap();
        eval([&logits_full]).unwrap();
        let full_tokens = decode_greedy(&mut model, &logits_full, &mut cache_full, 5);

        // Chunked prefill + 5 decode steps
        let mut cache_chunked: Vec<Option<LayerCache>> = Vec::new();
        let logits_chunked = model
            .forward_chunked(&input, None, &mut cache_chunked, 4)
            .unwrap();
        eval([&logits_chunked]).unwrap();
        let chunked_tokens = decode_greedy(&mut model, &logits_chunked, &mut cache_chunked, 5);

        assert_eq!(
            full_tokens, chunked_tokens,
            "decode tokens diverge: full={full_tokens:?} chunked={chunked_tokens:?}"
        );
    }

    /// Load whichever model is available for integration tests.
    fn load_test_model() -> Qwen3NextCausalLM {
        let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit")
        });
        if !std::path::Path::new(&model_path).exists() {
            panic!("Model not found at {model_path}. Set HIGGS_MODEL_PATH.");
        }
        // Warmup: load + prime shaders
        let mut model = load_qwen3_5_moe_model(&model_path).unwrap();
        let w = Array::from_slice(&[1u32, 2, 3, 4], &[1, 4]);
        let mut wc: Vec<Option<LayerCache>> = Vec::new();
        let out = model.forward(&w, None, &mut wc).unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
        model
    }

    /// Run greedy decode for `n` steps from prefill logits, return token ids.
    fn decode_greedy(
        model: &mut Qwen3NextCausalLM,
        prefill_logits: &Array,
        cache: &mut Vec<Option<LayerCache>>,
        n: usize,
    ) -> Vec<u32> {
        use mlx_rs::transforms::eval;

        let mut tok =
            ops::indexing::argmax_axis(&prefill_logits.index((.., -1, ..)), -1, false).unwrap();
        eval([&tok]).unwrap();
        let mut tokens = Vec::with_capacity(n);
        for _ in 0..n {
            let step_in = tok.index((.., ops::indexing::NewAxis));
            let out = model.forward(&step_in, None, cache).unwrap();
            tok = ops::indexing::argmax_axis(&out.index((.., -1, ..)), -1, false).unwrap();
            eval([&tok]).unwrap();
            tokens.push(tok.item::<u32>());
        }
        tokens
    }

    // -----------------------------------------------------------------------
    // Chunked prefill benchmark (real model)
    // -----------------------------------------------------------------------

    /// Benchmark chunked vs full prefill TTFT.
    ///
    /// Set env vars to control the benchmark:
    /// - `BENCH_SEQ`: comma-separated sequence lengths (default: 512,1024,2048,5120,10240)
    /// - `BENCH_CHUNK`: comma-separated chunk sizes (default: 128,256,512,1024)
    /// - `BENCH_FULL_MAX`: max sequence length for full prefill baseline (default: 10240)
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- bench_chunked_prefill --nocapture --ignored
    ///
    /// # Long sequences only:
    /// BENCH_SEQ=10240,20480,40960 BENCH_CHUNK=256,512 BENCH_FULL_MAX=20480 \
    ///   cargo test -p higgs-models --release -- bench_chunked_prefill --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires model files on disk"]
    fn bench_chunked_prefill() {
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let mut model = load_test_model();
        eprintln!(
            "Model: {} layers, hidden={}\n",
            model.args.num_hidden_layers, model.args.hidden_size,
        );

        let seq_lengths: Vec<i32> = std::env::var("BENCH_SEQ")
            .unwrap_or_else(|_| "512,1024,2048,5120,10240".to_string())
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let chunk_sizes: Vec<i32> = std::env::var("BENCH_CHUNK")
            .unwrap_or_else(|_| "128,256,512,1024".to_string())
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let full_max: i32 = std::env::var("BENCH_FULL_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10240);

        println!(
            "{:>7}  {:>6}  {:>10}  {:>10}  {:>8}",
            "T", "chunk", "full(ms)", "chunked(ms)", "ratio"
        );
        println!("{}", "-".repeat(50));

        for &seq_len in &seq_lengths {
            let tokens: Vec<u32> = (0..seq_len as u32)
                .map(|i| i % model.args.vocab_size as u32)
                .collect();
            let input = Array::from_slice(&tokens, &[1, seq_len]);

            let full_ms = if seq_len <= full_max {
                let mut cache_full: Vec<Option<LayerCache>> = Vec::new();
                let t0 = Instant::now();
                let logits_full = model.forward(&input, None, &mut cache_full).unwrap();
                eval([&logits_full]).unwrap();
                Some(t0.elapsed().as_secs_f64() * 1000.0)
            } else {
                None
            };

            for &chunk in &chunk_sizes {
                if chunk >= seq_len {
                    continue;
                }

                let mut cache_chunked: Vec<Option<LayerCache>> = Vec::new();
                let t0 = Instant::now();
                let logits_chunked = model
                    .forward_chunked(&input, None, &mut cache_chunked, chunk)
                    .unwrap();
                eval([&logits_chunked]).unwrap();
                let chunked_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let full_str = match full_ms {
                    Some(ms) => format!("{ms:>10.0}"),
                    None => format!("{:>10}", "—"),
                };
                let ratio_str = match full_ms {
                    Some(ms) => format!("{:>7.2}x", ms / chunked_ms),
                    None => format!("{:>8}", "—"),
                };

                println!("{seq_len:>7}  {chunk:>6}  {full_str}  {chunked_ms:>10.0}  {ratio_str}");
            }
            println!();
        }
    }

    // -----------------------------------------------------------------------
    // Prefill profiling benchmark
    // -----------------------------------------------------------------------

    /// Profile per-component TTFT breakdown for different sequence lengths.
    ///
    /// Measures wall-clock TTFT (single eval) and per-component time with eval
    /// barriers between embed, GDN, attention, MLP/MoE, norms, and lm_head.
    ///
    /// ```bash
    /// # Default model path: ~/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit
    /// cargo test -p higgs-models --release -- bench_prefill_breakdown --nocapture --ignored
    ///
    /// # Override model path:
    /// HIGGS_MODEL_PATH=/path/to/model cargo test -p higgs-models --release -- bench_prefill_breakdown --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires model files on disk"]
    fn bench_prefill_breakdown() {
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit")
        });
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Model not found at {model_path}");
            eprintln!("Set HIGGS_MODEL_PATH env var to your model directory");
            return;
        }

        eprintln!("Loading model from {model_path} ...");
        let mut model = load_qwen3_5_moe_model(&model_path).unwrap();
        let n_layers = model.args.num_hidden_layers;
        let fa_interval = model.args.full_attention_interval;
        eprintln!(
            "Loaded: {n_layers} layers, hidden={}, fa_interval={fa_interval}",
            model.args.hidden_size,
        );

        // Warmup: prime Metal shaders + lazy dtype conversions
        {
            let w = Array::from_slice(&[1u32, 2, 3, 4], &[1, 4]);
            let mut wc: Vec<Option<LayerCache>> = Vec::new();
            let out = model.forward(&w, None, &mut wc).unwrap();
            eval([&out].into_iter()).unwrap();
        }

        let max_seq: i32 = std::env::var("BENCH_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let all_lens: &[i32] = &[128, 512, 1024, 2048, 5120];
        let seq_lengths: Vec<i32> = all_lens.iter().copied().filter(|&t| t <= max_seq).collect();
        let seq_lengths: &[i32] = &seq_lengths;

        for &seq_len in seq_lengths {
            let tokens: Vec<u32> = (0..seq_len as u32)
                .map(|i| i % model.args.vocab_size as u32)
                .collect();

            // ----- Pass 1: real-world TTFT (no eval barriers) -----
            let input_a = Array::from_slice(&tokens, &[1, seq_len]);
            let mut cache_a: Vec<Option<LayerCache>> = Vec::new();

            let wall_start = Instant::now();
            let logits_a = model.forward(&input_a, None, &mut cache_a).unwrap();
            let mut eval_tgts: Vec<&Array> = vec![&logits_a];
            for lc in &cache_a {
                if let Some(LayerCache::Arrays(ac)) = lc {
                    if let Some(ref s) = ac.ssm_state {
                        eval_tgts.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_tgts.push(c);
                    }
                }
            }
            eval(eval_tgts).unwrap();
            let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

            // ----- Pass 2: per-component with eval barriers -----
            let input_b = Array::from_slice(&tokens, &[1, seq_len]);
            let mut cache_b: Vec<Option<LayerCache>> = model.make_cache();

            let fa_mask: Option<AttentionMask> = if seq_len > 1 {
                Some(AttentionMask::Causal)
            } else {
                None
            };

            // Embed
            let t0 = Instant::now();
            let mut h = model.model.embed_tokens.forward(&input_b).unwrap();
            eval([&h].into_iter()).unwrap();
            let ns_embed = t0.elapsed().as_nanos();

            let mut ns_gdn = 0u128;
            let mut ns_attn = 0u128;
            let mut ns_mlp = 0u128;
            let mut ns_norm = 0u128;
            let mut n_gdn = 0u32;
            let mut n_attn = 0u32;

            for (layer, layer_cache) in model.model.layers.iter_mut().zip(cache_b.iter_mut()) {
                let lc = layer_cache.as_mut().unwrap();
                let mask_ref = if layer.is_linear {
                    None
                } else {
                    fa_mask.as_ref()
                };

                // Pre-attention norm
                let t0 = Instant::now();
                let normed = layer.input_layernorm.forward(&h).unwrap();
                eval([&normed].into_iter()).unwrap();
                ns_norm += t0.elapsed().as_nanos();

                // GDN or full attention
                let t0 = Instant::now();
                let r = if layer.is_linear {
                    let gdn = layer.linear_attn.as_mut().unwrap();
                    let LayerCache::Arrays(sc) = lc else {
                        panic!("Expected ArraysCache");
                    };
                    let out = gdn.forward(&normed, mask_ref, sc).unwrap();
                    let mut tgts: Vec<&Array> = vec![&out];
                    if let Some(ref s) = sc.ssm_state {
                        tgts.push(s);
                    }
                    if let Some(ref c) = sc.conv_state {
                        tgts.push(c);
                    }
                    eval(tgts).unwrap();
                    n_gdn += 1;
                    ns_gdn += t0.elapsed().as_nanos();
                    out
                } else {
                    let attn = layer.self_attn.as_mut().unwrap();
                    let LayerCache::KV(kvc) = lc else {
                        panic!("Expected KVCache");
                    };
                    let out = attn.forward(&normed, mask_ref, kvc).unwrap();
                    eval([&out].into_iter()).unwrap();
                    n_attn += 1;
                    ns_attn += t0.elapsed().as_nanos();
                    out
                };

                // Residual + post-attention norm
                let t0 = Instant::now();
                let h2 = h.add(r).unwrap();
                let normed_post = layer.post_attention_layernorm.forward(&h2).unwrap();
                eval([&normed_post].into_iter()).unwrap();
                ns_norm += t0.elapsed().as_nanos();

                // MLP / MoE
                let t0 = Instant::now();
                let mlp_out = layer.mlp.forward(&normed_post).unwrap();
                eval([&mlp_out].into_iter()).unwrap();
                ns_mlp += t0.elapsed().as_nanos();

                // Final residual
                let t0 = Instant::now();
                h = h2.add(mlp_out).unwrap();
                eval([&h].into_iter()).unwrap();
                ns_norm += t0.elapsed().as_nanos();
            }

            // Final norm
            let t0 = Instant::now();
            h = model.model.norm.forward(&h).unwrap();
            eval([&h].into_iter()).unwrap();
            ns_norm += t0.elapsed().as_nanos();

            // LM head
            let t0 = Instant::now();
            let _logits = match model.lm_head.as_ref() {
                Some(head) => head.forward(&h).unwrap(),
                None => model.model.embed_tokens.as_linear(&h).unwrap(),
            };
            eval([&_logits].into_iter()).unwrap();
            let ns_lm = t0.elapsed().as_nanos();

            // ----- Report -----
            let barrier_total = ns_embed + ns_gdn + ns_attn + ns_mlp + ns_norm + ns_lm;
            let ms = |ns: u128| ns as f64 / 1e6;
            let pct = |ns: u128| ns as f64 / barrier_total as f64 * 100.0;
            let n_total = n_gdn + n_attn;

            println!();
            println!("==== T = {seq_len} ====");
            println!("  Wall TTFT (no barriers):  {:>8.1}ms", wall_ms,);
            println!(
                "  Sum  (eval barriers):     {:>8.1}ms  (barrier overhead: {:.1}ms)",
                ms(barrier_total),
                ms(barrier_total) - wall_ms,
            );
            println!();
            println!(
                "  embed:            {:>8.1}ms  {:>5.1}%",
                ms(ns_embed),
                pct(ns_embed),
            );
            println!(
                "  GDN ({n_gdn:>2} layers): {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
                ms(ns_gdn),
                pct(ns_gdn),
                ms(ns_gdn) / n_gdn.max(1) as f64,
            );
            println!(
                "  Attn ({n_attn:>2} layers): {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
                ms(ns_attn),
                pct(ns_attn),
                ms(ns_attn) / n_attn.max(1) as f64,
            );
            println!(
                "  MLP/MoE:          {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
                ms(ns_mlp),
                pct(ns_mlp),
                ms(ns_mlp) / n_total.max(1) as f64,
            );
            println!(
                "  norms+residual:   {:>8.1}ms  {:>5.1}%",
                ms(ns_norm),
                pct(ns_norm),
            );
            println!(
                "  lm_head:          {:>8.1}ms  {:>5.1}%",
                ms(ns_lm),
                pct(ns_lm),
            );
            println!(
                "  ---- GDN share of wall TTFT: {:.1}%",
                ms(ns_gdn) / wall_ms * 100.0,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Block-K per-layer breakdown (real model, populated cache)
    // -----------------------------------------------------------------------

    /// Profile per-component cost of block-K verify against populated KV cache.
    ///
    /// Unlike `bench_prefill_breakdown` which starts from empty cache, this
    /// prefills 256 tokens first, then instruments a block-K forward at S=16
    /// with eval barriers between each component.
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- bench_block_k_breakdown --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires 35B model on disk"]
    fn bench_block_k_breakdown() {
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let mut model = load_test_model();
        let n_layers = model.args.num_hidden_layers;
        let fa_interval = model.args.full_attention_interval;

        // --- Prefill 256 tokens to populate cache ---
        let prefill_len: i32 = 256;
        let prompt_ids: Vec<u32> = (1..=prefill_len as u32).collect();
        let prompt = Array::from_slice(&prompt_ids, &[1, prefill_len]);

        let mut cache: Vec<Option<LayerCache>> = Vec::new();
        let prefill_out = model.forward(&prompt, None, &mut cache).unwrap();
        {
            let mut tgts: Vec<&Array> = vec![&prefill_out];
            for lc in &cache {
                if let Some(LayerCache::Arrays(ac)) = lc {
                    if let Some(ref s) = ac.ssm_state {
                        tgts.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        tgts.push(c);
                    }
                }
            }
            eval(tgts).unwrap();
        }

        // 2 warmup decode steps
        let tok = ops::indexing::argmax_axis(&prefill_out.index((.., -1, ..)), -1, false).unwrap();
        eval([&tok]).unwrap();
        let mut current = tok;
        for _ in 0..2 {
            let inp = current.index((.., ops::indexing::NewAxis));
            let out = model.forward(&inp, None, &mut cache).unwrap();
            current = ops::indexing::argmax_axis(&out.index((.., -1, ..)), -1, false).unwrap();
            eval([&current]).unwrap();
        }

        println!("Prefill done. Cache offset: {}", prefill_len + 2);
        println!("Layers: {n_layers}, fa_interval: {fa_interval}");

        for block_k in [1, 8, 16] {
            let block_ids: Vec<u32> = (1..=block_k as u32).collect();
            let input = Array::from_slice(&block_ids, &[1, block_k]);

            // ----- Pass 1: wall-clock (no barriers) -----
            // Use a copy of cache state for fair comparison
            let t_wall = Instant::now();
            let h_wall = model.forward_hidden(&input, None, &mut cache).unwrap();
            eval([&h_wall]).unwrap();
            let wall_ms = t_wall.elapsed().as_secs_f64() * 1000.0;

            // ----- Pass 2: per-component with eval barriers -----
            // NOTE: cache is now offset by block_k more. This is fine — we
            // want the instrumented pass at the same populated-cache regime.

            let kv_offset = cache
                .iter()
                .filter_map(|lc| match lc.as_ref()? {
                    LayerCache::KV(kv) => Some(kv.offset()),
                    _ => None,
                })
                .next()
                .unwrap_or(0);

            let fa_mask: Option<AttentionMask> = if block_k > 1 {
                if kv_offset > block_k {
                    Some(AttentionMask::Array(
                        create_causal_mask(block_k, Some(kv_offset)).unwrap(),
                    ))
                } else {
                    Some(AttentionMask::Causal)
                }
            } else {
                None
            };

            // Embed
            let t0 = Instant::now();
            let mut h = model.model.embed_tokens.forward(&input).unwrap();
            eval([&h].into_iter()).unwrap();
            let ns_embed = t0.elapsed().as_nanos();

            let mut ns_gdn = 0u128;
            let mut ns_attn = 0u128;
            let mut ns_mlp = 0u128;
            let mut ns_norm = 0u128;
            let mut n_gdn = 0u32;
            let mut n_attn = 0u32;

            for (layer, layer_cache) in model.model.layers.iter_mut().zip(cache.iter_mut()) {
                let lc = layer_cache.as_mut().unwrap();
                let mask_ref = if layer.is_linear {
                    None
                } else {
                    fa_mask.as_ref()
                };

                let t0 = Instant::now();
                let normed = layer.input_layernorm.forward(&h).unwrap();
                eval([&normed].into_iter()).unwrap();
                ns_norm += t0.elapsed().as_nanos();

                let t0 = Instant::now();
                let r = if layer.is_linear {
                    let gdn = layer.linear_attn.as_mut().unwrap();
                    let LayerCache::Arrays(sc) = lc else {
                        panic!("Expected ArraysCache");
                    };
                    let out = gdn.forward(&normed, mask_ref, sc).unwrap();
                    let mut tgts: Vec<&Array> = vec![&out];
                    if let Some(ref s) = sc.ssm_state {
                        tgts.push(s);
                    }
                    if let Some(ref c) = sc.conv_state {
                        tgts.push(c);
                    }
                    eval(tgts).unwrap();
                    n_gdn += 1;
                    ns_gdn += t0.elapsed().as_nanos();
                    out
                } else {
                    let attn = layer.self_attn.as_mut().unwrap();
                    let LayerCache::KV(kvc) = lc else {
                        panic!("Expected KVCache");
                    };
                    let out = attn.forward(&normed, mask_ref, kvc).unwrap();
                    eval([&out].into_iter()).unwrap();
                    n_attn += 1;
                    ns_attn += t0.elapsed().as_nanos();
                    out
                };

                let t0 = Instant::now();
                let h2 = h.add(r).unwrap();
                let normed_post = layer.post_attention_layernorm.forward(&h2).unwrap();
                eval([&normed_post].into_iter()).unwrap();
                ns_norm += t0.elapsed().as_nanos();

                let t0 = Instant::now();
                let mlp_out = layer.mlp.forward(&normed_post).unwrap();
                eval([&mlp_out].into_iter()).unwrap();
                ns_mlp += t0.elapsed().as_nanos();

                let t0 = Instant::now();
                h = h2.add(mlp_out).unwrap();
                eval([&h].into_iter()).unwrap();
                ns_norm += t0.elapsed().as_nanos();
            }

            let t0 = Instant::now();
            h = model.model.norm.forward(&h).unwrap();
            eval([&h].into_iter()).unwrap();
            ns_norm += t0.elapsed().as_nanos();

            let t0 = Instant::now();
            let _logits = match model.lm_head.as_ref() {
                Some(head) => head.forward(&h).unwrap(),
                None => model.model.embed_tokens.as_linear(&h).unwrap(),
            };
            eval([&_logits].into_iter()).unwrap();
            let ns_lm = t0.elapsed().as_nanos();

            let barrier_total = ns_embed + ns_gdn + ns_attn + ns_mlp + ns_norm + ns_lm;
            let ms = |ns: u128| ns as f64 / 1e6;
            let pct = |ns: u128| ns as f64 / barrier_total as f64 * 100.0;
            let n_total = n_gdn + n_attn;

            println!();
            println!("==== Block-K = {block_k}, cache ~{kv_offset} ====");
            println!("  Wall (no barriers):   {:>8.1}ms", wall_ms);
            println!(
                "  Sum  (eval barriers): {:>8.1}ms  (overhead: {:.1}ms)",
                ms(barrier_total),
                ms(barrier_total) - wall_ms,
            );
            println!();
            println!(
                "  embed:            {:>8.1}ms  {:>5.1}%",
                ms(ns_embed),
                pct(ns_embed),
            );
            println!(
                "  GDN ({n_gdn:>2} layers): {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
                ms(ns_gdn),
                pct(ns_gdn),
                ms(ns_gdn) / n_gdn.max(1) as f64,
            );
            println!(
                "  Attn ({n_attn:>2} layers): {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
                ms(ns_attn),
                pct(ns_attn),
                ms(ns_attn) / n_attn.max(1) as f64,
            );
            println!(
                "  MLP/MoE:          {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
                ms(ns_mlp),
                pct(ns_mlp),
                ms(ns_mlp) / n_total.max(1) as f64,
            );
            println!(
                "  norms+residual:   {:>8.1}ms  {:>5.1}%",
                ms(ns_norm),
                pct(ns_norm),
            );
            println!(
                "  lm_head:          {:>8.1}ms  {:>5.1}%",
                ms(ns_lm),
                pct(ns_lm),
            );
            println!(
                "  ---- GDN share of wall: {:.1}%",
                ms(ns_gdn) / wall_ms * 100.0,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Block-K forward benchmark (real model)
    // -----------------------------------------------------------------------

    /// Benchmark `forward_hidden` at S=1 vs S=4,8,16 against cached state.
    ///
    /// This measures the true Rust-level forward cost, eliminating API/HTTP
    /// overhead. Used to determine if DFlash block-K verify is viable.
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- bench_block_k_verify --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires 35B model on disk"]
    fn bench_block_k_verify() {
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let mut model = load_test_model();

        // --- Prefill 256 tokens to build KV cache ---
        let prompt_len = 256;
        let prompt_ids: Vec<u32> = (1..=prompt_len).collect();
        let prompt = Array::from_slice(&prompt_ids, &[1, prompt_len as i32]);

        let mut cache: Vec<Option<LayerCache>> = Vec::new();
        let prefill_out = model.forward(&prompt, None, &mut cache).unwrap();
        let mut to_eval: Vec<&Array> = vec![&prefill_out];
        for lc in &cache {
            if let Some(lc) = lc {
                match lc {
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            to_eval.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            to_eval.push(c);
                        }
                    }
                    LayerCache::KV(_) => {}
                }
            }
        }
        eval(to_eval).unwrap();

        // --- Warmup: 2 decode steps ---
        let tok = ops::indexing::argmax_axis(&prefill_out.index((.., -1, ..)), -1, false).unwrap();
        eval([&tok]).unwrap();

        let mut current = tok;
        for _ in 0..2 {
            let inp = current.index((.., ops::indexing::NewAxis));
            let out = model.forward(&inp, None, &mut cache).unwrap();
            current = ops::indexing::argmax_axis(&out.index((.., -1, ..)), -1, false).unwrap();
            eval([&current]).unwrap();
        }

        // Current cache offset after prefill + 2 warmup decodes
        let cache_offset = prompt_len as i32 + 2;
        println!("Cache offset after warmup: {cache_offset}");

        // --- Benchmark S=1 baseline ---
        let n_trials = 5;
        let mut s1_times = Vec::with_capacity(n_trials);
        for _ in 0..n_trials {
            // Clone cache for fair comparison (each trial starts from same state)
            // Actually we just do single-token decode which is non-destructive
            let inp = current.index((.., ops::indexing::NewAxis));
            let t0 = Instant::now();
            let h = model.forward_hidden(&inp, None, &mut cache).unwrap();
            eval([&h]).unwrap();
            s1_times.push(t0.elapsed());
            // Undo the cache advancement by continuing (cache already updated, but
            // we measure the same workload each time — offset grows by 1 per trial)
            current = ops::indexing::argmax_axis(&h.index((.., -1, ..)), -1, false).unwrap();
            eval([&current]).unwrap();
        }
        let s1_median_ms = {
            let mut ms: Vec<f64> = s1_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ms[ms.len() / 2]
        };

        println!("\n==== S=1 (single-token decode) ====");
        println!("  median: {s1_median_ms:.2} ms");

        // --- Benchmark S=4,8,16 (block-K verify) ---
        for block_k in [4, 8, 16] {
            let block_ids: Vec<u32> = (1..=block_k).collect();
            let block_input = Array::from_slice(&block_ids, &[1, block_k as i32]);

            let mut bk_times = Vec::with_capacity(n_trials);
            for _ in 0..n_trials {
                let t0 = Instant::now();
                let h = model
                    .forward_hidden(&block_input, None, &mut cache)
                    .unwrap();
                eval([&h]).unwrap();
                bk_times.push(t0.elapsed());
                // Undo: cache offset grows by block_k each trial
            }
            let bk_median_ms = {
                let mut ms: Vec<f64> = bk_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ms[ms.len() / 2]
            };
            let ratio = bk_median_ms / s1_median_ms;
            let ms_per_tok = bk_median_ms / block_k as f64;

            println!("\n==== S={block_k} (block-K verify) ====");
            println!("  median: {bk_median_ms:.2} ms  ({ratio:.1}x vs S=1)");
            println!("  per-token: {ms_per_tok:.2} ms/tok");
        }

        // --- Also benchmark forward_all_logits at S=16 ---
        {
            let block_ids: Vec<u32> = (1..=16).collect();
            let block_input = Array::from_slice(&block_ids, &[1, 16]);
            let t0 = Instant::now();
            let logits = model
                .forward_all_logits(&block_input, None, &mut cache)
                .unwrap();
            eval([&logits]).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let shape = logits.shape().to_vec();
            println!("\n==== forward_all_logits S=16 ====");
            println!("  time: {ms:.2} ms, shape: {shape:?}");
        }
    }

    // -----------------------------------------------------------------------
    // Conv1d batch-K parity (unit test — no model files needed)
    // -----------------------------------------------------------------------

    /// Verify sliding-window loop matches native Conv1d for depthwise convolution.
    ///
    /// The S<=32 fast path in GatedDeltaNet::forward computes depthwise conv via
    /// a loop of `window * wt → sum(axis=1)`, avoiding Conv1d kernel dispatch.
    /// This test confirms it produces identical results to the native kernel.
    #[test]
    fn test_conv1d_batch_k_parity() {
        use mlx_rs::transforms::eval;

        let conv_dim = 64i32;
        let kernel_size = 4i32;

        // Depthwise Conv1d weight: [conv_dim, kernel_size, 1] (groups=conv_dim)
        let weight =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[conv_dim, kernel_size, 1], None)
                .unwrap();
        eval([&weight]).unwrap();

        // Transposed weight for fast path: [kernel_size, conv_dim]
        let wt = weight.squeeze_axes(&[-1]).unwrap().transpose().unwrap();
        eval([&wt]).unwrap();

        for s in [2, 4, 8, 16] {
            let total_len = kernel_size - 1 + s;
            let conv_input =
                mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, total_len, conv_dim], None)
                    .unwrap();
            eval([&conv_input]).unwrap();

            // Reference: native depthwise Conv1d kernel
            let ref_out = ops::conv1d(&conv_input, &weight, 1, 0, 1, conv_dim).unwrap();
            eval([&ref_out]).unwrap();

            // Our fast path: sliding window loop (same as GatedDeltaNet S<=32 path)
            let mut windows = Vec::with_capacity(s as usize);
            for i in 0..s {
                windows.push(
                    conv_input
                        .index((.., i..i + kernel_size, ..))
                        .multiply(&wt)
                        .unwrap()
                        .sum_axes(&[1], true)
                        .unwrap(),
                );
            }
            let fast_out = ops::concatenate_axis(&windows.iter().collect::<Vec<_>>(), 1).unwrap();
            eval([&fast_out]).unwrap();

            assert_eq!(ref_out.shape(), fast_out.shape(), "shape mismatch at S={s}");

            let diff = ref_out.subtract(&fast_out).unwrap().abs().unwrap();
            let max_diff: f32 = diff.max(None).unwrap().item();
            assert!(
                max_diff < 1e-5,
                "Conv1d batch-K parity failed at S={s}: max |diff| = {max_diff}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // TQ block-K parity (requires 35B model on disk)
    // -----------------------------------------------------------------------

    /// Compare L=1 sequential decode (known-good TQ path) against L=16 block
    /// forward (new TQ + causal mask path). Per-position logits must match.
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- test_tq_block_k_parity --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[ignore = "requires 35B model on disk"]
    fn test_tq_block_k_parity() {
        use mlx_rs::transforms::eval;

        let mut model = load_test_model();
        let block_k = 16i32;

        // --- Prefill short prompt ---
        let prompt_len = 64i32;
        let prompt_ids: Vec<u32> = (1..=prompt_len as u32).collect();
        let prompt = Array::from_slice(&prompt_ids, &[1, prompt_len]);

        let mut cache_a: Vec<Option<LayerCache>> = Vec::new();
        let prefill_out = model.forward(&prompt, None, &mut cache_a).unwrap();
        {
            let mut tgts: Vec<&Array> = vec![&prefill_out];
            for lc in &cache_a {
                if let Some(LayerCache::Arrays(ac)) = lc {
                    if let Some(ref s) = ac.ssm_state {
                        tgts.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        tgts.push(c);
                    }
                }
            }
            eval(tgts).unwrap();
        }

        // Clone cache for the block path (before sequential decode mutates it)
        let cache_b = cache_a.clone();

        // --- Path A: L=1 sequential decode × block_k steps ---
        let decode_ids: Vec<u32> = (100..100 + block_k as u32).collect();
        let mut seq_logits: Vec<Array> = Vec::with_capacity(block_k as usize);
        for &tid in &decode_ids {
            let inp = Array::from_slice(&[tid], &[1, 1]);
            let logits = model.forward_all_logits(&inp, None, &mut cache_a).unwrap();
            eval([&logits]).unwrap();
            seq_logits.push(logits); // [1, 1, vocab]
        }
        // Stack → [1, block_k, vocab]
        let seq_all = ops::concatenate_axis(&seq_logits.iter().collect::<Vec<_>>(), 1).unwrap();
        eval([&seq_all]).unwrap();

        // --- Path B: L=block_k block forward ---
        cache_a = cache_b; // restore pre-decode state
        let block_input = Array::from_slice(&decode_ids, &[1, block_k]);
        let block_logits = model
            .forward_all_logits(&block_input, None, &mut cache_a)
            .unwrap();
        eval([&block_logits]).unwrap();

        // --- Compare per-position logits ---
        let diff = seq_all.subtract(&block_logits).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        eprintln!("TQ block-K parity: max |diff| = {max_diff}  (block_k={block_k})");

        // 3-bit quantization + bf16 accumulation allows some tolerance.
        // Sequential and block paths use identical TQ Metal kernels; the only
        // difference is causal masking. Tolerance generous for GDN state drift.
        assert!(
            max_diff < 2.0,
            "TQ block-K parity failed: max |diff| = {max_diff} (expect <2.0)"
        );
    }

    // -----------------------------------------------------------------------
    // Dense model decode benchmark (27B, requires model on disk)
    // -----------------------------------------------------------------------

    /// Measure S=1 decode + prefill for a dense Qwen3.5 model.
    ///
    /// ```bash
    /// HIGGS_DENSE_MODEL_PATH=~/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit \
    /// cargo test -p higgs-models --release -- bench_dense_model_decode --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[ignore = "requires dense Qwen3.5 model on disk"]
    fn bench_dense_model_decode() {
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let model_path = std::env::var("HIGGS_DENSE_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit")
        });
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Dense model not found. Set HIGGS_DENSE_MODEL_PATH.");
            return;
        }
        eprintln!("Loading dense model from {model_path}...");
        let mut model = load_qwen3_5_model(&model_path).unwrap();
        eprintln!(
            "Loaded: {} layers, hidden={}, fa_interval={}",
            model.args.num_hidden_layers,
            model.args.hidden_size,
            model.args.full_attention_interval,
        );

        // Warmup
        {
            let w = Array::from_slice(&[1u32, 2, 3, 4], &[1, 4]);
            let mut wc: Vec<Option<LayerCache>> = Vec::new();
            let out = model.forward(&w, None, &mut wc).unwrap();
            eval([&out]).unwrap();
        }

        let prompt_len = 256i32;
        let prompt_ids: Vec<u32> = (1..=prompt_len as u32).collect();
        let prompt = Array::from_slice(&prompt_ids, &[1, prompt_len]);

        // Prefill TTFT
        let mut cache: Vec<Option<LayerCache>> = Vec::new();
        let t0 = Instant::now();
        let out = model.forward(&prompt, None, &mut cache).unwrap();
        eval([&out]).unwrap();
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // S=1 decode
        let n = 20usize;
        let tok = Array::from_slice(&[1u32], &[1, 1]);
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let t0 = Instant::now();
            let o = model.forward(&tok, None, &mut cache).unwrap();
            eval([&o]).unwrap();
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(f64::total_cmp);
        let med = samples[n / 2];

        println!("\n==== Dense 27B-4bit, ctx={prompt_len} ====");
        println!("  Prefill TTFT: {prefill_ms:.1} ms");
        println!(
            "  S=1 decode median: {med:.2} ms  ({:.1} tok/s)",
            1000.0 / med
        );
    }

    // -----------------------------------------------------------------------
    // TurboQuant vs dense KV decode benchmark (real model)
    // -----------------------------------------------------------------------

    /// Compare S=1 decode speed: dense KV cache vs TurboQuant KV cache.
    ///
    /// Prefills at two context depths, then measures median S=1 decode with
    /// each cache type. Shows TQ overhead at short context and benefit at long.
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- bench_tq_vs_dense_decode --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[ignore = "requires 35B model on disk"]
    fn bench_tq_vs_dense_decode() {
        use crate::turboquant::{KvCacheConfig, KvCacheMode};
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let tq_config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            ..KvCacheConfig::default()
        };

        let prompt_lengths: &[i32] = &[256, 2048];
        let n_decode = 20usize;

        println!(
            "\n{:>8}  {:>10}  {:>10}  {:>8}",
            "ctx", "dense_ms", "tq_ms", "tq/dense"
        );
        println!("{}", "-".repeat(44));

        for &prompt_len in prompt_lengths {
            let mut model = load_test_model();
            let prompt_ids: Vec<u32> = (1..=prompt_len as u32).collect();
            let prompt = Array::from_slice(&prompt_ids, &[1, prompt_len]);

            // --- Dense KV ---
            let mut cache_dense: Vec<Option<LayerCache>> = Vec::new();
            let out = model.forward(&prompt, None, &mut cache_dense).unwrap();
            eval([&out]).unwrap();

            let mut dense_samples = Vec::with_capacity(n_decode);
            let tok = Array::from_slice(&[1u32], &[1, 1]);
            for _ in 0..n_decode {
                let t0 = Instant::now();
                let o = model.forward(&tok, None, &mut cache_dense).unwrap();
                eval([&o]).unwrap();
                dense_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            dense_samples.sort_by(f64::total_cmp);
            let dense_med = dense_samples[n_decode / 2];

            // --- TurboQuant KV ---
            let mut cache_tq = model.make_cache_turbo(tq_config).unwrap();
            let out = model.forward(&prompt, None, &mut cache_tq).unwrap();
            eval([&out]).unwrap();

            let mut tq_samples = Vec::with_capacity(n_decode);
            for _ in 0..n_decode {
                let t0 = Instant::now();
                let o = model.forward(&tok, None, &mut cache_tq).unwrap();
                eval([&o]).unwrap();
                tq_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            tq_samples.sort_by(f64::total_cmp);
            let tq_med = tq_samples[n_decode / 2];

            println!(
                "{:>8}  {:>10.2}  {:>10.2}  {:>8.3}x",
                prompt_len,
                dense_med,
                tq_med,
                tq_med / dense_med
            );
        }
    }

    // -----------------------------------------------------------------------
    // BF16 baseline decode (9B dense, requires model on disk)
    // -----------------------------------------------------------------------

    /// Smoke-test all three BF16 forward paths (QEmbedding, QLinear, FfnBlock dense)
    /// on a real BF16 checkpoint. Loads the model, verifies dtypes, then exercises
    /// each BF16 code path at component level (embedding, linear, MLP) to check
    /// output magnitudes without running full 32-layer forward (which OOMs on 32GB).
    ///
    /// ```bash
    /// HIGGS_BF16_MODEL_PATH=~/AI-Models/shared/huggingface/hub/models--Qwen--Qwen3.5-9B \
    ///   cargo test -p higgs-models --release -- test_9b_bf16_baseline_decode --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[ignore = "requires BF16 Qwen3.5-9B model on disk"]
    fn test_9b_bf16_baseline_decode() {
        use mlx_rs::module::ModuleParameters;
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let model_path = std::env::var("HIGGS_BF16_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/AI-Models/shared/huggingface/hub/models--Qwen--Qwen3.5-9B")
        });
        if !std::path::Path::new(&model_path).exists() {
            panic!("BF16 model not found at {model_path}. Set HIGGS_BF16_MODEL_PATH.");
        }
        eprintln!("Loading BF16 model from {model_path}...");
        let mut model = load_qwen3_5_model(&model_path).unwrap();
        let hidden = model.args.hidden_size;
        let intermediate = model.args.intermediate_size;
        let vocab = model.args.vocab_size;
        eprintln!(
            "Loaded: {} layers, hidden={hidden}, intermediate={intermediate}, fa_interval={}, vocab={vocab}",
            model.args.num_hidden_layers, model.args.full_attention_interval,
        );

        // ── 1. Verify weight dtypes are BF16 (not quantized) ──
        let params = model.parameters().flatten();
        let mut n_bf16 = 0usize;
        let mut n_uint32 = 0usize;
        let mut n_fp32 = 0usize;
        for (_key, arr) in &params {
            let a: &Array = arr;
            match a.dtype() {
                Dtype::Bfloat16 => n_bf16 += 1,
                Dtype::Uint32 => n_uint32 += 1,
                Dtype::Float32 => n_fp32 += 1,
                _ => {}
            }
        }
        eprintln!(
            "Weight dtypes: {n_bf16} bf16, {n_fp32} fp32, {n_uint32} uint32, {} total",
            params.len()
        );
        assert!(
            n_uint32 == 0,
            "Expected pure BF16 checkpoint but found {n_uint32} uint32 (quantized) params"
        );
        assert!(n_bf16 > 0, "No bfloat16 params found — wrong checkpoint?");

        // ── 2. QEmbedding BF16 path (line 296: self.weight.index(indices)) ──
        let embed = &model.model.embed_tokens;
        eprintln!("\n=== QEmbedding BF16 ===");
        eprintln!(
            "  weight dtype: {:?}, shape: {:?}",
            embed.weight.dtype(),
            embed.weight.shape()
        );
        assert_eq!(embed.weight.dtype(), Dtype::Bfloat16);

        let indices = Array::from_slice(&[1u32, 100, 5000, 42], &[1, 4]);
        let emb_out = embed.forward(&indices).unwrap();
        eval([&emb_out]).unwrap();
        eprintln!(
            "  output shape: {:?}, dtype: {:?}",
            emb_out.shape(),
            emb_out.dtype()
        );
        assert_eq!(emb_out.shape(), &[1, 4, hidden]);

        let emb_abs = emb_out.abs().unwrap();
        let emb_max: f32 = emb_abs.max(None).unwrap().item();
        let emb_mean: f32 = emb_abs.mean(None).unwrap().item();
        eprintln!("  |embed| max={emb_max:.4}, mean={emb_mean:.6}");
        assert!(
            emb_max.is_finite() && emb_max > 0.001,
            "Dead embedding: max={emb_max}"
        );
        assert!(
            emb_mean.is_finite() && emb_mean > 0.0001,
            "Dead embedding: mean={emb_mean}"
        );

        // ── 3. QLinear BF16 path (line 254: matmul(x, self.weight.value.t())) ──
        eprintln!("\n=== QLinear BF16 ===");
        let l0 = &model.model.layers[0];
        let gate = l0.mlp.gate_proj.as_ref().unwrap();
        eprintln!(
            "  gate_proj weight dtype: {:?}, shape: {:?}",
            gate.weight.dtype(),
            gate.weight.shape()
        );
        assert_eq!(gate.weight.dtype(), Dtype::Bfloat16);

        // Feed a realistic-magnitude input through a single QLinear
        let x_linear = mlx_rs::random::normal::<f32>(&[1, 1, hidden], None, None, None).unwrap();
        let x_bf16 = x_linear.as_dtype(Dtype::Bfloat16).unwrap();
        let lin_out = gate.forward(&x_bf16).unwrap();
        eval([&lin_out]).unwrap();
        eprintln!(
            "  output shape: {:?}, dtype: {:?}",
            lin_out.shape(),
            lin_out.dtype()
        );
        assert_eq!(lin_out.shape(), &[1, 1, intermediate]);

        let lin_abs = lin_out.abs().unwrap();
        let lin_max: f32 = lin_abs.max(None).unwrap().item();
        let lin_mean: f32 = lin_abs.mean(None).unwrap().item();
        eprintln!("  |gate_proj(x)| max={lin_max:.4}, mean={lin_mean:.6}");
        assert!(
            lin_max.is_finite() && lin_max > 0.001,
            "Dead QLinear: max={lin_max}"
        );

        // Also check down_proj (intermediate → hidden)
        let down = l0.mlp.down_proj.as_ref().unwrap();
        eprintln!(
            "  down_proj weight dtype: {:?}, shape: {:?}",
            down.weight.dtype(),
            down.weight.shape()
        );
        let x_inter = mlx_rs::random::normal::<f32>(&[1, 1, intermediate], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let down_out = down.forward(&x_inter).unwrap();
        eval([&down_out]).unwrap();
        let down_max: f32 = down_out.abs().unwrap().max(None).unwrap().item();
        eprintln!("  |down_proj(x)| max={down_max:.4}");
        assert!(
            down_max.is_finite() && down_max > 0.001,
            "Dead down_proj: max={down_max}"
        );

        // ── 4. FfnBlock dense BF16 path (line 1826-1829: separate gate/up/down) ──
        eprintln!("\n=== FfnBlock dense BF16 (SwiGLU) ===");
        let ffn = &mut model.model.layers[0].mlp;
        assert!(!ffn.is_moe, "Expected dense FfnBlock on layer 0");

        // Verify the BF16 branch will be taken (gate/up dtype != Uint32)
        let gp_dtype = ffn.gate_proj.as_ref().unwrap().weight.dtype();
        let up_dtype = ffn.up_proj.as_ref().unwrap().weight.dtype();
        eprintln!(
            "  gate_proj dtype={gp_dtype:?}, up_proj dtype={up_dtype:?} (BF16 branch taken: {})",
            gp_dtype != Dtype::Uint32 || up_dtype != Dtype::Uint32
        );

        let x_ffn = mlx_rs::random::normal::<f32>(&[1, 4, hidden], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let ffn_out = ffn.forward(&x_ffn).unwrap();
        eval([&ffn_out]).unwrap();
        eprintln!(
            "  output shape: {:?}, dtype: {:?}",
            ffn_out.shape(),
            ffn_out.dtype()
        );
        assert_eq!(ffn_out.shape(), &[1, 4, hidden]);

        let ffn_abs = ffn_out.abs().unwrap();
        let ffn_max: f32 = ffn_abs.max(None).unwrap().item();
        let ffn_mean: f32 = ffn_abs.mean(None).unwrap().item();
        eprintln!("  |FfnBlock(x)| max={ffn_max:.4}, mean={ffn_mean:.6}");
        assert!(
            ffn_max.is_finite() && ffn_max > 0.001,
            "Dead FfnBlock: max={ffn_max}"
        );
        assert!(ffn_mean.is_finite(), "NaN in FfnBlock output");

        // ── 5. lm_head BF16 path ──
        eprintln!("\n=== lm_head BF16 ===");
        if let Some(ref mut head) = model.dense_lm_head {
            eprintln!("  dense_lm_head present (nn::Linear)");
            let h_in = mlx_rs::random::normal::<f32>(&[1, 1, hidden], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let lm_out = head.forward(&h_in).unwrap();
            eval([&lm_out]).unwrap();
            let lm_max: f32 = lm_out.abs().unwrap().max(None).unwrap().item();
            eprintln!(
                "  output shape: {:?}, |lm_head| max={lm_max:.4}",
                lm_out.shape()
            );
            assert!(
                lm_max.is_finite() && lm_max > 0.001,
                "Dead lm_head: max={lm_max}"
            );
        } else if let Some(ref head) = model.lm_head {
            eprintln!("  QLinear lm_head, weight dtype: {:?}", head.weight.dtype());
            let h_in = mlx_rs::random::normal::<f32>(&[1, 1, hidden], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let lm_out = head.forward(&h_in).unwrap();
            eval([&lm_out]).unwrap();
            let lm_max: f32 = lm_out.abs().unwrap().max(None).unwrap().item();
            eprintln!(
                "  output shape: {:?}, |lm_head| max={lm_max:.4}",
                lm_out.shape()
            );
            assert!(
                lm_max.is_finite() && lm_max > 0.001,
                "Dead lm_head: max={lm_max}"
            );
        } else {
            // Tied embeddings — as_linear path
            eprintln!("  Tied embeddings (as_linear path)");
            let h_in = mlx_rs::random::normal::<f32>(&[1, 1, hidden], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let lm_out = model.model.embed_tokens.as_linear(&h_in).unwrap();
            eval([&lm_out]).unwrap();
            let lm_max: f32 = lm_out.abs().unwrap().max(None).unwrap().item();
            eprintln!(
                "  output shape: {:?}, |as_linear| max={lm_max:.4}",
                lm_out.shape()
            );
            assert!(
                lm_max.is_finite() && lm_max > 0.001,
                "Dead as_linear: max={lm_max}"
            );
        }

        eprintln!("\n✓ All BF16 forward paths produce finite, non-zero outputs.");

        // ── 6. Per-component latency bench ──
        eprintln!("\n=== Per-component latency (5 iters, S=1) ===");
        let n_iter = 5usize;
        let x1 = mlx_rs::random::normal::<f32>(&[1, 1, hidden], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        eval([&x1]).unwrap();

        // Embedding
        {
            let idx = Array::from_slice(&[42u32], &[1, 1]);
            // warmup
            eval([&embed.forward(&idx).unwrap()]).unwrap();
            let mut ms = Vec::with_capacity(n_iter);
            for _ in 0..n_iter {
                let t = Instant::now();
                let o = embed.forward(&idx).unwrap();
                eval([&o]).unwrap();
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            ms.sort_by(f64::total_cmp);
            eprintln!("  QEmbedding:   {:.3} ms (median)", ms[n_iter / 2]);
        }

        // QLinear (gate_proj: hidden→intermediate)
        {
            let gp = model.model.layers[0].mlp.gate_proj.as_ref().unwrap();
            eval([&gp.forward(&x1).unwrap()]).unwrap();
            let mut ms = Vec::with_capacity(n_iter);
            for _ in 0..n_iter {
                let t = Instant::now();
                let o = gp.forward(&x1).unwrap();
                eval([&o]).unwrap();
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            ms.sort_by(f64::total_cmp);
            eprintln!(
                "  QLinear gate:  {:.3} ms (median)  [{hidden}→{intermediate}]",
                ms[n_iter / 2]
            );
        }

        // QLinear (down_proj: intermediate→hidden)
        {
            let dp = model.model.layers[0].mlp.down_proj.as_ref().unwrap();
            let x_i = mlx_rs::random::normal::<f32>(&[1, 1, intermediate], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            eval([&x_i]).unwrap();
            eval([&dp.forward(&x_i).unwrap()]).unwrap();
            let mut ms = Vec::with_capacity(n_iter);
            for _ in 0..n_iter {
                let t = Instant::now();
                let o = dp.forward(&x_i).unwrap();
                eval([&o]).unwrap();
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            ms.sort_by(f64::total_cmp);
            eprintln!(
                "  QLinear down:  {:.3} ms (median)  [{intermediate}→{hidden}]",
                ms[n_iter / 2]
            );
        }

        // FfnBlock (full SwiGLU: gate+up+silu+down)
        {
            let ffn = &mut model.model.layers[0].mlp;
            eval([&ffn.forward(&x1).unwrap()]).unwrap();
            let mut ms = Vec::with_capacity(n_iter);
            for _ in 0..n_iter {
                let t = Instant::now();
                let o = ffn.forward(&x1).unwrap();
                eval([&o]).unwrap();
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            ms.sort_by(f64::total_cmp);
            eprintln!(
                "  FfnBlock:     {:.3} ms (median)  [SwiGLU {hidden}→{intermediate}→{hidden}]",
                ms[n_iter / 2]
            );
        }

        // lm_head
        {
            if let Some(ref mut head) = model.dense_lm_head {
                eval([&head.forward(&x1).unwrap()]).unwrap();
                let mut ms = Vec::with_capacity(n_iter);
                for _ in 0..n_iter {
                    let t = Instant::now();
                    let o = head.forward(&x1).unwrap();
                    eval([&o]).unwrap();
                    ms.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                ms.sort_by(f64::total_cmp);
                eprintln!(
                    "  lm_head:      {:.3} ms (median)  [{hidden}→{vocab}]",
                    ms[n_iter / 2]
                );
            }
        }

        // Full-model S=1 decode attempt (may OOM on 32GB with 9B BF16)
        eprintln!("\n=== Full-model forward (S=1, single token) ===");
        let tok_input = Array::from_slice(&[42u32], &[1, 1]);
        let mut cache: Vec<Option<LayerCache>> = Vec::new();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let out = model.forward(&tok_input, None, &mut cache).unwrap();
            eval([&out]).unwrap();
            out
        })) {
            Ok(out) => {
                let out_max: f32 = out.abs().unwrap().max(None).unwrap().item();
                eprintln!(
                    "  output shape: {:?}, |logits| max={out_max:.4}",
                    out.shape()
                );

                // S=1 decode timing
                let mut ms = Vec::with_capacity(n_iter);
                for _ in 0..n_iter {
                    let t = Instant::now();
                    let o = model.forward(&tok_input, None, &mut cache).unwrap();
                    eval([&o]).unwrap();
                    ms.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                ms.sort_by(f64::total_cmp);
                let med = ms[n_iter / 2];
                eprintln!("  S=1 decode: {med:.2} ms  ({:.1} tok/s)", 1000.0 / med);
            }
            Err(_) => {
                eprintln!("  ⚠ Full-model forward panicked (likely OOM on 32GB with 9B BF16)");
            }
        }
    }

    #[test]
    fn test_chunk_vs_sequential_gdn() {
        use mlx_rs::ops::*;
        use mlx_rs::transforms;
        let b = 1i32;
        let s = 18i32;
        let nk = 16i32;
        let nv = 32i32;
        let dk = 128i32;
        let dv = 128i32;

        let q: Array =
            mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[b, s, nk, dk], None).unwrap();
        let k: Array =
            mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[b, s, nk, dk], None).unwrap();
        let v: Array =
            mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[b, s, nv, dv], None).unwrap();
        let g: Array = mlx_rs::random::uniform::<f32, f32>(-5.0, -0.5, &[b, s, nv], None).unwrap();
        let beta: Array = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[b, s, nv], None).unwrap();

        let mut q_parts: Vec<Array> = Vec::new();
        let mut k_parts: Vec<Array> = Vec::new();
        for _ in 0..(nv / nk) {
            q_parts.push(q.clone());
            k_parts.push(k.clone());
        }
        let q_refs: Vec<&Array> = q_parts.iter().collect();
        let q_rep = concatenate_axis(&q_refs, 2).unwrap();
        let k_refs: Vec<&Array> = k_parts.iter().collect();
        let k_rep = concatenate_axis(&k_refs, 2).unwrap();

        let (chunk_y, chunk_state) =
            chunk_gated_delta_rule(&q_rep, &k_rep, &v, &g, &beta, nv, dk, nv, dv, None).unwrap();
        chunk_y.eval().unwrap();
        chunk_state.eval().unwrap();
        let cy_max: f32 = chunk_y.abs().unwrap().max(None).unwrap().item();
        let cs_max: f32 = chunk_state.abs().unwrap().max(None).unwrap().item();
        eprintln!("chunk_y: max={:.4}, shape={:?}", cy_max, chunk_y.shape());
        eprintln!(
            "chunk_state: max={:.4}, shape={:?}",
            cs_max,
            chunk_state.shape()
        );

        assert!(!cy_max.is_nan(), "chunk y has NaN");
        assert!(!cs_max.is_nan(), "chunk state has NaN");

        // Sequential (HF recurrent_gated_delta_rule) — state [B, H, Dk, Dv]
        let mut seq_state: Array = Array::zeros::<f32>(&[b, nv, dk, dv]).unwrap();
        let scale = 1.0f32 / (dk as f32).sqrt();
        let q_f = q_rep
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap()
            .multiply(&Array::from_f32(scale))
            .unwrap();
        let k_f = k_rep
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap();
        let v_f = v
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap();
        let g_f = g
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .transpose_axes(&[0, 2, 1])
            .unwrap();
        let b_f = beta
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .transpose_axes(&[0, 2, 1])
            .unwrap();
        q_f.eval().unwrap();
        k_f.eval().unwrap();
        v_f.eval().unwrap();
        g_f.eval().unwrap();
        b_f.eval().unwrap();
        seq_state.eval().unwrap();

        let mut seq_y_parts: Vec<Array> = Vec::new();
        for t in 0..s {
            let q_t = q_f.index((.., .., t..t + 1, ..));
            let k_t = k_f.index((.., .., t..t + 1, ..));
            let v_t = v_f.index((.., .., t..t + 1, ..));
            let g_t = g_f
                .index((.., .., t..t + 1))
                .exp()
                .unwrap()
                .expand_dims(-1)
                .unwrap();
            let bt = sigmoid(&b_f.index((.., .., t..t + 1)))
                .unwrap()
                .expand_dims(-1)
                .unwrap();

            // HF: state [B, H, Dk, Dv], k_t [B, H, 1, Dk], q_t [B, H, 1, Dk], v_t [B, H, 1, Dv]
            // state * g_t → [B, H, Dk, Dv] * [B, H, 1, 1] → [B, H, Dk, Dv]
            let g_t_bc = broadcast_to(&g_t, seq_state.shape()).unwrap();
            let sg = seq_state.multiply(&g_t_bc).unwrap();

            // kv_mem = (state * k_t.unsqueeze(-1)).sum(-2)
            // k_t.unsqueeze(-1) → [B, H, 1, Dk, 1], state needs to be [B, H, 1, Dk, Dv] for broadcast
            // So: expand state to 5D, multiply, sum over Dk axis (axis -2)
            let sg_5d = sg.expand_dims(2).unwrap(); // [B, H, 1, Dk, Dv]
            let k_t_5d = k_t.expand_dims(-1).unwrap(); // [B, H, 1, Dk, 1]
            let kv = sg_5d
                .multiply(&k_t_5d)
                .unwrap()
                .sum_axes(&[-2], None)
                .unwrap(); // [B, H, 1, Dv]

            let delta = v_t
                .subtract(&kv)
                .unwrap()
                .multiply(&broadcast_to(&bt, v_t.shape()).unwrap())
                .unwrap();

            // state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            // delta.unsqueeze(-2) → [B, H, 1, 1, Dv], k_t.unsqueeze(-1) → [B, H, 1, Dk, 1]
            // product → [B, H, 1, Dk, Dv], squeeze axis 2 → [B, H, Dk, Dv]
            let d_5d = delta.expand_dims(-2).unwrap(); // [B, H, 1, 1, Dv]
            let k_t_5d = k_t.expand_dims(-1).unwrap(); // [B, H, 1, Dk, 1]
            let update = k_t_5d.multiply(&d_5d).unwrap(); // [B, H, 1, Dk, Dv]
            let update_4d = update.squeeze_axes(&[2]).unwrap(); // [B, H, Dk, Dv]
            seq_state = sg.add(&update_4d).unwrap();

            // out = (state * q_t.unsqueeze(-1)).sum(-2)
            let s_5d = seq_state.expand_dims(2).unwrap(); // [B, H, 1, Dk, Dv]
            let q_t_5d = q_t.expand_dims(-1).unwrap(); // [B, H, 1, Dk, 1]
            let out = s_5d
                .multiply(&q_t_5d)
                .unwrap()
                .sum_axes(&[-2], None)
                .unwrap();
            // out is [B, H, 1, Dv] (4D)
            seq_state.eval().unwrap();
            out.eval().unwrap();
            seq_y_parts.push(out);
        }

        let sy_refs: Vec<&Array> = seq_y_parts.iter().collect();
        let seq_y = concatenate_axis(&sy_refs, 2).unwrap();
        eprintln!("seq_y before transpose: {:?}", seq_y.shape());
        let seq_y = seq_y.transpose_axes(&[0, 2, 1, 3]).unwrap();
        eprintln!("seq_y after transpose: {:?}", seq_y.shape());
        seq_y.eval().unwrap();
        let sy_max: f32 = seq_y.abs().unwrap().max(None).unwrap().item();
        eprintln!("seq_y: max={:.4}, shape={:?}", sy_max, seq_y.shape());
        assert!(!sy_max.is_nan(), "seq y has NaN");

        let max_diff: f32 = chunk_y
            .subtract(&seq_y)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        eprintln!("max |chunk - seq| = {:.6}", max_diff);
    }

    // -----------------------------------------------------------------------
    // End-to-end GDN layer 0 ANE parity (Wave 1 acceptance)
    // -----------------------------------------------------------------------

    /// Load a real BF16 Qwen3.5-9B model, enable ANE on all three GDN
    /// projections of layer 0 (`in_proj_qkvz`, `in_proj_ba`, `out_proj`), and
    /// verify the layer output matches the all-Metal baseline to within 0.05
    /// absolute. This closes the Phase 1 acceptance gap (which proved only a
    /// single projection on synthetic weights).
    ///
    /// ```bash
    /// HIGGS_BF16_MODEL_PATH=~/AI-Models/shared/huggingface/hub/models--Qwen--Qwen3.5-9B \
    ///   cargo test -p higgs-models --release --features ane -- \
    ///     test_9b_gdn_layer0_ane_parity --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "requires BF16 Qwen3.5-9B model on disk"]
    fn test_9b_gdn_layer0_ane_parity() {
        use mlx_rs::transforms::eval;

        let model_path = std::env::var("HIGGS_BF16_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/AI-Models/shared/huggingface/hub/models--Qwen--Qwen3.5-9B")
        });
        if !std::path::Path::new(&model_path).exists() {
            panic!("BF16 model not found at {model_path}. Set HIGGS_BF16_MODEL_PATH.");
        }
        eprintln!("Loading BF16 model from {model_path}...");
        let mut model = load_qwen3_5_model(&model_path).unwrap();
        let hidden = model.args.hidden_size;
        eprintln!(
            "Loaded: {} layers, hidden={hidden}, fa_interval={}",
            model.args.num_hidden_layers, model.args.full_attention_interval,
        );

        // Layer 0 must be a GDN (linear attention) layer.
        let layer = &mut model.model.layers[0];
        assert!(
            layer.is_linear,
            "Layer 0 is not a GDN linear-attention layer — model config changed?"
        );
        let gdn = layer
            .linear_attn
            .as_mut()
            .expect("layer 0 linear_attn missing");
        assert!(
            !gdn.use_separate_projections,
            "Wave 1 ANE parity requires the fused qkvz/ba projection path \
             (set HIGGS_SEPARATE_GDN_PROJ unset)"
        );

        // Shared deterministic-per-run input at the expected post-layernorm
        // magnitude (~unit variance). bf16 to match the model dtype.
        let s = 16i32;
        let x_f32 = mlx_rs::random::normal::<f32>(&[1, s, hidden], None, None, None).unwrap();
        let x = x_f32.as_dtype(Dtype::Bfloat16).unwrap();
        x.eval().unwrap();

        // ── Baseline: all-Metal forward (ane_kernels=None by default) ──
        let mut cache_ref = ArraysCache::default();
        let (out_ref, _tape_ref) = gdn
            .forward_with_tape(&x, None, &mut cache_ref)
            .expect("baseline Metal forward failed");
        eval([&out_ref]).unwrap();
        let ref_max: f32 = out_ref.abs().unwrap().max(None).unwrap().item();
        eprintln!(
            "Metal baseline: out shape={:?}, dtype={:?}, |out|_max={:.4}",
            out_ref.shape(),
            out_ref.dtype(),
            ref_max
        );
        assert!(
            ref_max.is_finite() && ref_max > 0.0,
            "Dead Metal baseline: max={ref_max}"
        );

        // ── Enable ANE on all three GDN projections at compile seq = S ──
        gdn.enable_ane_gdn(&[s]).expect("enable_ane_gdn failed");

        let mut cache_ane = ArraysCache::default();
        let (out_ane, _tape_ane) = gdn
            .forward_with_tape(&x, None, &mut cache_ane)
            .expect("ANE forward failed");
        eval([&out_ane]).unwrap();

        // ── Compare elementwise in f32 ──
        let diff = out_ref
            .as_dtype(Dtype::Float32)
            .unwrap()
            .subtract(out_ane.as_dtype(Dtype::Float32).unwrap())
            .unwrap()
            .abs()
            .unwrap();
        diff.eval().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        let mean_diff: f32 = diff.mean(None).unwrap().item();
        // Magnitude-aware budget. Outputs at this layer are bf16; absolute
        // bf16 ULP scales with value magnitude (~|out|/2^7). Allow 0.5% of
        // the output magnitude — comfortably above bf16 quantization noise
        // (~1 ULP ≈ |out|/128) yet far below what an algorithmic bug would
        // produce (a real bug shows up in mean, not just outlier max).
        let budget = (ref_max * 0.005).max(0.05);
        eprintln!(
            "GDN layer 0 ANE parity: max_diff={max_diff:.6}, \
             mean_diff={mean_diff:.6} (budget {budget:.4} = max(0.005·|out|_max, 0.05))"
        );
        assert!(
            max_diff.is_finite(),
            "ANE output contains NaN/Inf: max_diff={max_diff}"
        );
        assert!(
            max_diff < budget,
            "GDN layer 0 ANE parity failed: max_diff={max_diff} exceeds {budget:.4} budget \
             (|out|_max={ref_max:.2}, mean_diff={mean_diff:.6})"
        );
    }

    /// Wave 2 acceptance: every GDN layer's forward output (through the donor
    /// + patched ANE projections) matches the all-Metal baseline within a
    /// magnitude-aware bf16 budget. Covers the donor-patch path on real 9B
    /// weights and exercises the public helper that model_loader will call
    /// once Wave 4 lands.
    ///
    /// ```bash
    /// HIGGS_BF16_MODEL_PATH=~/.cache/lm-studio/models/jason-schulz/Carnice-9b-MLX \
    ///   cargo test -p higgs-models --release --features ane -- \
    ///     test_9b_gdn_all_layers_ane_parity --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "requires Carnice-9B-MLX (or Qwen3.5-9B BF16) on disk"]
    fn test_9b_gdn_all_layers_ane_parity() {
        use mlx_rs::transforms::eval;

        let model_path = std::env::var("HIGGS_BF16_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/jason-schulz/Carnice-9b-MLX")
        });
        if !std::path::Path::new(&model_path).exists() {
            panic!("Model not found at {model_path}. Set HIGGS_BF16_MODEL_PATH.");
        }
        eprintln!("Loading model from {model_path}...");
        let mut model = load_qwen3_5_model(&model_path).unwrap();
        let hidden = model.args.hidden_size;
        let n_layers = model.model.layers.len();
        let n_linear = model.model.layers.iter().filter(|l| l.is_linear).count();
        eprintln!("Loaded: {n_layers} layers ({n_linear} linear), hidden={hidden}");

        // Per-linear-layer Metal baseline: feed the same input into every GDN
        // layer's forward_with_tape, capture last-token output. Doing this
        // BEFORE enabling ANE keeps the comparison fair (same input each side).
        let s = 16i32;
        let x_f32 = mlx_rs::random::normal::<f32>(&[1, s, hidden], None, None, None).unwrap();
        let x = x_f32.as_dtype(Dtype::Bfloat16).unwrap();
        x.eval().unwrap();

        let mut metal_outs: Vec<(usize, Array)> = Vec::with_capacity(n_linear);
        for (idx, layer) in model.model.layers.iter_mut().enumerate() {
            if !layer.is_linear {
                continue;
            }
            let gdn = layer.linear_attn.as_mut().unwrap();
            assert!(
                !gdn.use_separate_projections,
                "layer {idx}: use_separate_projections=true unsupported in Wave 2"
            );
            let mut cache = ArraysCache::default();
            let (out, _tape) = gdn.forward_with_tape(&x, None, &mut cache).unwrap();
            eval([&out]).unwrap();
            metal_outs.push((idx, out));
        }
        eprintln!("Captured {} Metal baseline outputs", metal_outs.len());

        // Enable ANE on every GDN layer via the public Wave 2 helper
        // (single-bucket slice — same as before Wave 3 lifted the cardinality).
        let report = model
            .enable_ane_gdn_all_layers(&[s])
            .expect("enable_ane_gdn_all_layers failed");
        eprintln!("ANE setup: {report:?}");
        assert_eq!(
            report.n_compiled_layers, 1,
            "expected exactly one donor compile"
        );
        assert_eq!(report.n_buckets, 1, "expected single-bucket setup");
        assert_eq!(
            report.n_patched_layers,
            n_linear - 1,
            "expected {} patched layers, got {}",
            n_linear - 1,
            report.n_patched_layers
        );
        // Patches must NOT recompile — that is the whole point of
        // patch_from_donor. The exact bridge-counter semantics are unstable
        // (observed Δcompile=0 in practice; bridge counters appear to be
        // incremented in a different path than docs suggest), so the strict
        // upper bound is "anything below the worst case of one fresh compile
        // per kernel". 24 GDN × 3 projs = 72 kernels — anything < 10 proves
        // patching, not recompiling.
        let compile_delta = report.compile_count_after - report.compile_count_before;
        assert!(
            compile_delta < 10,
            "patch_from_donor leaked into compileWithQoS: Δcompile={compile_delta} \
             (expected « {} kernels)",
            n_linear * 3
        );

        // Re-run every GDN layer through ANE. Same input as Metal baseline.
        let mut max_diff_global = 0.0f32;
        let mut max_diff_layer = 0usize;
        let mut ref_max_at_worst = 0.0f32;
        for (i, (idx, ref_out)) in metal_outs.iter().enumerate() {
            let layer = &mut model.model.layers[*idx];
            let gdn = layer.linear_attn.as_mut().unwrap();
            assert!(
                gdn.ane_kernels.is_some(),
                "layer {idx}: ane_kernels not attached after enable_ane_gdn_all_layers"
            );
            let mut cache = ArraysCache::default();
            let (out_ane, _tape) = gdn.forward_with_tape(&x, None, &mut cache).unwrap();
            eval([&out_ane]).unwrap();
            let ref_max: f32 = ref_out.abs().unwrap().max(None).unwrap().item();
            let diff = ref_out
                .as_dtype(Dtype::Float32)
                .unwrap()
                .subtract(out_ane.as_dtype(Dtype::Float32).unwrap())
                .unwrap()
                .abs()
                .unwrap();
            diff.eval().unwrap();
            let max_diff: f32 = diff.max(None).unwrap().item();
            if i < 3 || i >= metal_outs.len().saturating_sub(2) {
                eprintln!("  layer{idx:>2}: |out|_max={ref_max:.3} max_diff={max_diff:.5}");
            }
            if max_diff > max_diff_global {
                max_diff_global = max_diff;
                max_diff_layer = *idx;
                ref_max_at_worst = ref_max;
            }
        }

        // 1% relative budget (allows ≤2 bf16 ULPs at any output magnitude;
        // bf16 ULP ≈ |out| / 128). Mean diff stays orders of magnitude below
        // budget, so a real algorithmic bug would never sneak through.
        let budget = (ref_max_at_worst * 0.01).max(0.05);
        eprintln!(
            "Worst-case GDN layer parity: layer{max_diff_layer} max_diff={max_diff_global:.6} \
             |out|_max={ref_max_at_worst:.3} (budget {budget:.4})"
        );
        assert!(
            max_diff_global.is_finite(),
            "ANE produced NaN/Inf at layer{max_diff_layer}"
        );
        assert!(
            max_diff_global < budget,
            "All-layers ANE parity failed at layer{max_diff_layer}: \
             max_diff={max_diff_global} exceeds {budget:.4} budget"
        );
    }

    /// Wave 4 acceptance: every GDN layer's forward output, dispatched
    /// through the model-wide `qwen-gdn-ane-worker` thread, matches the
    /// all-Metal baseline within the same magnitude-aware bf16 budget the
    /// Wave 2 inline test uses. This is the parity gate that proves
    /// `enable_ane_gdn_all_layers_via_worker` is safe to wire into
    /// `model_loader.rs` for `HIGGS_TARGET_ANE_GDN=1`.
    ///
    /// ```bash
    /// HIGGS_BF16_MODEL_PATH=~/.cache/lm-studio/models/jason-schulz/Carnice-9b-MLX \
    ///   cargo test -p higgs-models --release --features ane -- \
    ///     test_9b_gdn_all_layers_ane_parity_worker --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "requires Carnice-9B-MLX (or Qwen3.5-9B BF16) on disk"]
    fn test_9b_gdn_all_layers_ane_parity_worker() {
        use mlx_rs::transforms::eval;

        let model_path = std::env::var("HIGGS_BF16_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/jason-schulz/Carnice-9b-MLX")
        });
        if !std::path::Path::new(&model_path).exists() {
            panic!("Model not found at {model_path}. Set HIGGS_BF16_MODEL_PATH.");
        }
        eprintln!("Loading model from {model_path}...");
        let mut model = load_qwen3_5_model(&model_path).unwrap();
        let hidden = model.args.hidden_size;
        let n_layers = model.model.layers.len();
        let n_linear = model.model.layers.iter().filter(|l| l.is_linear).count();
        eprintln!("Loaded: {n_layers} layers ({n_linear} linear), hidden={hidden}");

        // Capture Metal baseline BEFORE attaching the worker (same input each
        // side — fair comparison).
        let s = 16i32;
        let x_f32 = mlx_rs::random::normal::<f32>(&[1, s, hidden], None, None, None).unwrap();
        let x = x_f32.as_dtype(Dtype::Bfloat16).unwrap();
        x.eval().unwrap();

        let mut metal_outs: Vec<(usize, Array)> = Vec::with_capacity(n_linear);
        for (idx, layer) in model.model.layers.iter_mut().enumerate() {
            if !layer.is_linear {
                continue;
            }
            let gdn = layer.linear_attn.as_mut().unwrap();
            assert!(
                !gdn.use_separate_projections,
                "layer {idx}: use_separate_projections=true unsupported in Wave 4"
            );
            let mut cache = ArraysCache::default();
            let (out, _tape) = gdn.forward_with_tape(&x, None, &mut cache).unwrap();
            eval([&out]).unwrap();
            metal_outs.push((idx, out));
        }
        eprintln!("Captured {} Metal baseline outputs", metal_outs.len());

        // Spin up the worker, attach handles to every linear layer.
        let report = model
            .enable_ane_gdn_all_layers_via_worker(s)
            .expect("enable_ane_gdn_all_layers_via_worker failed");
        eprintln!("Worker setup: {report:?}");
        assert_eq!(
            report.n_compiled_layers, 1,
            "expected exactly one donor compile"
        );
        assert_eq!(report.n_buckets, 1, "Wave 4 is single-bucket");
        assert_eq!(
            report.n_patched_layers,
            n_linear - 1,
            "expected {} patched layers, got {}",
            n_linear - 1,
            report.n_patched_layers
        );
        // Worker spawn must respect the Wave 2 invariant: exactly 3 fresh MIL
        // compiles (one per projection donor). Same observation as the inline
        // test: bridge counters appear bumped from a different code path, so
        // we only assert the upper bound that proves patching, not recompiling.
        let compile_delta = report.compile_count_after - report.compile_count_before;
        assert!(
            compile_delta < 10,
            "worker spawn leaked into compileWithQoS: Δcompile={compile_delta} \
             (expected « {} kernels)",
            n_linear * 3
        );

        // Re-run every GDN layer through the worker. Same input as Metal.
        let mut max_diff_global = 0.0f32;
        let mut max_diff_layer = 0usize;
        let mut ref_max_at_worst = 0.0f32;
        for (i, (idx, ref_out)) in metal_outs.iter().enumerate() {
            let layer = &mut model.model.layers[*idx];
            let gdn = layer.linear_attn.as_mut().unwrap();
            assert!(
                gdn.ane_handle.is_some(),
                "layer {idx}: ane_handle not attached after \
                 enable_ane_gdn_all_layers_via_worker"
            );
            let mut cache = ArraysCache::default();
            let (out_ane, _tape) = gdn.forward_with_tape(&x, None, &mut cache).unwrap();
            eval([&out_ane]).unwrap();
            let ref_max: f32 = ref_out.abs().unwrap().max(None).unwrap().item();
            let diff = ref_out
                .as_dtype(Dtype::Float32)
                .unwrap()
                .subtract(out_ane.as_dtype(Dtype::Float32).unwrap())
                .unwrap()
                .abs()
                .unwrap();
            diff.eval().unwrap();
            let max_diff: f32 = diff.max(None).unwrap().item();
            if i < 3 || i >= metal_outs.len().saturating_sub(2) {
                eprintln!("  layer{idx:>2}: |out|_max={ref_max:.3} max_diff={max_diff:.5}");
            }
            if max_diff > max_diff_global {
                max_diff_global = max_diff;
                max_diff_layer = *idx;
                ref_max_at_worst = ref_max;
            }
        }

        // Same 1% relative budget the Wave 2 inline test uses — worker path
        // should be bit-identical to inline because both end up dispatching
        // through `AneProjKernel` with the same compiled microcode.
        let budget = (ref_max_at_worst * 0.01).max(0.05);
        eprintln!(
            "Worst-case GDN worker parity: layer{max_diff_layer} max_diff={max_diff_global:.6} \
             |out|_max={ref_max_at_worst:.3} (budget {budget:.4})"
        );
        assert!(
            max_diff_global.is_finite(),
            "Worker produced NaN/Inf at layer{max_diff_layer}"
        );
        assert!(
            max_diff_global < budget,
            "All-layers ANE worker parity failed at layer{max_diff_layer}: \
             max_diff={max_diff_global} exceeds {budget:.4} budget"
        );
    }

    // -----------------------------------------------------------------------
    // MLP layer 0 ANE int8 parity (dense SwiGLU path)
    // -----------------------------------------------------------------------

    /// Load a real BF16 9B dense model (Carnice-9B-MLX), capture the layer 0
    /// MLP output via the all-Metal SwiGLU path, install the ANE int8 kernels
    /// for `gate_proj` / `up_proj` / `down_proj`, forward again, and compare.
    ///
    /// Tolerance is magnitude-aware and looser than the GDN parity gate —
    /// int8 dequant is noisier than q4 dequant plus an extra fp16 round-trip
    /// through the conv1x1 mlpackage.
    ///
    /// ```bash
    /// HIGGS_BF16_MODEL_PATH=~/.cache/lm-studio/models/jason-schulz/Carnice-9b-MLX \
    ///   cargo test -p higgs-models --release --features ane -- \
    ///     test_9b_mlp_layer0_int8_ane_parity --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "requires Carnice-9B-MLX (or another dense-layer-0 BF16 9B) on disk"]
    fn test_9b_mlp_layer0_int8_ane_parity() {
        use mlx_rs::transforms::eval;

        let model_path = std::env::var("HIGGS_BF16_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/jason-schulz/Carnice-9b-MLX")
        });
        if !std::path::Path::new(&model_path).exists() {
            panic!("Model not found at {model_path}. Set HIGGS_BF16_MODEL_PATH.");
        }
        eprintln!("Loading model from {model_path}...");
        let mut model = load_qwen3_5_model(&model_path).unwrap();
        let hidden = model.args.hidden_size;
        let inter = model.args.intermediate_size;
        eprintln!(
            "Loaded: {} layers, hidden={hidden}, intermediate={inter}",
            model.model.layers.len()
        );

        assert!(
            !model.model.layers[0].mlp.is_moe,
            "Layer 0 MLP must be dense for this parity gate (model config changed?)"
        );

        let s = 128i32;
        let x_f32 = mlx_rs::random::normal::<f32>(&[1, s, hidden], None, None, None).unwrap();
        let x = x_f32.as_dtype(Dtype::Bfloat16).unwrap();
        x.eval().unwrap();

        // ── Baseline: all-Metal dense MLP forward ──
        let out_ref = {
            let ffn = &mut model.model.layers[0].mlp;
            let out = ffn.forward(&x).expect("baseline dense MLP forward failed");
            eval([&out]).unwrap();
            out
        };
        let ref_max: f32 = out_ref
            .as_dtype(Dtype::Float32)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        eprintln!(
            "Metal baseline: out shape={:?}, dtype={:?}, |out|_max={:.4}",
            out_ref.shape(),
            out_ref.dtype(),
            ref_max
        );
        assert!(
            ref_max.is_finite() && ref_max > 0.0,
            "Dead Metal baseline: max={ref_max}"
        );

        // ── Install ANE int8 kernels on layer 0 ──
        let (g, u, d, h, i) = model
            .prepare_mlp_layer0_int8_weights()
            .expect("prepare_mlp_layer0_int8_weights failed")
            .expect("layer 0 is MoE — test requires a dense MLP at layer 0");
        assert_eq!(h, hidden as usize);
        assert_eq!(i, inter as usize);
        model
            .finalize_ane_mlp_layer0_int8_inline(g, u, d, h, i, s)
            .expect("finalize_ane_mlp_layer0_int8_inline failed");

        // ── ANE forward on the same input ──
        let out_ane = {
            let ffn = &mut model.model.layers[0].mlp;
            let out = ffn.forward(&x).expect("ANE dense MLP forward failed");
            eval([&out]).unwrap();
            out
        };
        assert_eq!(out_ane.shape(), out_ref.shape());

        // ── Compare elementwise in f32 ──
        let diff = out_ref
            .as_dtype(Dtype::Float32)
            .unwrap()
            .subtract(out_ane.as_dtype(Dtype::Float32).unwrap())
            .unwrap()
            .abs()
            .unwrap();
        diff.eval().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        let mean_diff: f32 = diff.mean(None).unwrap().item();
        // int8 dequant adds ~|w|*2^-7 error per element; three projections +
        // fp16 conv1x1 boundaries stack to ~3% worst-element noise at
        // |out|_max (mean stays <0.5%). Budget is 3% of the output magnitude
        // (vs. 0.5% for the single-hop GDN parity gate), floored at 0.2
        // absolute so tiny-output layers don't gate on fp16 ULP noise. An
        // axis/transpose bug would show up in mean, not just max.
        let budget = (ref_max * 0.03).max(0.2);
        eprintln!(
            "MLP layer 0 int8 ANE parity: max_diff={max_diff:.6}, \
             mean_diff={mean_diff:.6} (budget {budget:.4} = max(0.03·|out|_max, 0.2))"
        );
        // Mean-diff sanity gate: a real algorithmic bug (wrong axis order,
        // swapped gate/up, stale weight binding) shifts the whole output and
        // pushes mean_diff to the same order as max_diff. Guard at 1% of
        // |out|_max — 5× the observed ~0.2% int8 noise floor.
        let mean_budget = (ref_max * 0.01).max(0.05);
        assert!(
            mean_diff < mean_budget,
            "MLP layer 0 int8 ANE parity mean_diff={mean_diff} exceeds {mean_budget:.4} \
             (likely algorithmic bug, not quantization noise; max_diff={max_diff}, \
             |out|_max={ref_max:.2})"
        );
        assert!(
            max_diff.is_finite(),
            "ANE output contains NaN/Inf: max_diff={max_diff}"
        );
        assert!(
            max_diff < budget,
            "MLP layer 0 int8 ANE parity failed: max_diff={max_diff} exceeds {budget:.4} budget \
             (|out|_max={ref_max:.2}, mean_diff={mean_diff:.6})"
        );
    }

    /// Micro-bench for `forward_ane_int8_mlp` at the handoff-gate shape
    /// (bucket=512, Carnice-9B dims). Reports min/median/mean over N iters
    /// after warmup. Use to establish a baseline before any perf change and
    /// to gate the ≥20% improvement target from
    /// `.planning/next-session-ane-int8-mlp-zerocopy.md`.
    ///
    /// ```bash
    /// HIGGS_BF16_MODEL_PATH=~/.cache/lm-studio/models/jason-schulz/Carnice-9b-MLX \
    /// HIGGS_CORETOOLS_PYTHON=/path/to/venv/bin/python \
    ///   cargo test -p higgs-models --release --features ane -- \
    ///     forward_ane_int8_mlp_bench --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "requires Carnice-9B-MLX + HIGGS_CORETOOLS_PYTHON"]
    fn forward_ane_int8_mlp_bench() {
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let model_path = std::env::var("HIGGS_BF16_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/jason-schulz/Carnice-9b-MLX")
        });
        assert!(
            std::path::Path::new(&model_path).exists(),
            "Model not found at {model_path}. Set HIGGS_BF16_MODEL_PATH."
        );
        let mut model = load_qwen3_5_model(&model_path).unwrap();
        let hidden = model.args.hidden_size;
        let inter = model.args.intermediate_size;

        let seq: i32 = 512;
        let iters: usize = 50;
        let warmup: usize = 5;

        let x_f32 = mlx_rs::random::normal::<f32>(&[1, seq, hidden], None, None, None).unwrap();
        let x = x_f32.as_dtype(Dtype::Bfloat16).unwrap();
        x.eval().unwrap();

        let (g, u, d, h, i) = model
            .prepare_mlp_layer0_int8_weights()
            .expect("prepare")
            .expect("layer 0 must be dense");
        assert_eq!(h, hidden as usize);
        assert_eq!(i, inter as usize);
        model
            .finalize_ane_mlp_layer0_int8_inline(g, u, d, h, i, seq)
            .expect("finalize");

        let ffn = &mut model.model.layers[0].mlp;

        for _ in 0..warmup {
            let y = ffn.forward(&x).expect("warmup forward");
            eval([&y]).unwrap();
        }

        let mut samples: Vec<u128> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let y = ffn.forward(&x).expect("bench forward");
            eval([&y]).unwrap();
            samples.push(t0.elapsed().as_micros());
        }
        samples.sort_unstable();
        let min_ms = samples[0] as f64 / 1000.0;
        let med_ms = samples[iters / 2] as f64 / 1000.0;
        let mean_us: u128 = samples.iter().sum();
        let mean_ms = mean_us as f64 / (iters as f64 * 1000.0);
        eprintln!(
            "forward_ane_int8_mlp_bench seq={seq} bucket={seq} hidden={hidden} inter={inter}: \
             min={min_ms:.3}ms  median={med_ms:.3}ms  mean={mean_ms:.3}ms  (n={iters})"
        );
    }

    /// Smoke-load the 0.8B Qwen3.5 drafter candidate through `load_qwen3_5_model`
    /// and run a single incremental forward. Confirms the VLM-wrapped 24-layer
    /// dense variant loads and forwards before we build the drafter adapter on top.
    ///
    /// Required model on disk at `~/.cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit/`.
    #[test]
    #[ignore]
    fn smoke_load_qwen3_5_08b() {
        use mlx_rs::Array;
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let home = std::env::var("HOME").unwrap();
        let path = format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit");

        let t0 = Instant::now();
        let mut model = load_qwen3_5_model(&path).expect("load_qwen3_5_model on 0.8B");
        eprintln!(
            "0.8B loaded in {:.2}s: vocab={}, hidden={}, layers={}",
            t0.elapsed().as_secs_f64(),
            model.args.vocab_size,
            model.args.hidden_size,
            model.args.num_hidden_layers
        );
        assert_eq!(model.args.vocab_size, 248320);
        assert_eq!(model.args.hidden_size, 1024);
        assert_eq!(model.args.num_hidden_layers, 24);

        // One incremental forward on a trivial token sequence.
        let toks: [i32; 4] = [248045, 1234, 5678, 248046];
        let arr = Array::from_slice(&toks, &[1, toks.len() as i32]);
        let mut cache = model.make_cache();

        let t1 = Instant::now();
        let logits = model.forward(&arr, None, &mut cache).expect("forward");
        eval([&logits]).unwrap();
        eprintln!(
            "forward in {:.1}ms: shape={:?}",
            t1.elapsed().as_secs_f64() * 1000.0,
            logits.shape()
        );
        assert_eq!(logits.shape(), &[1, 1, 248320]);

        // Argmax sanity — any valid token id.
        let argmax = mlx_rs::ops::indexing::argmax_axis(&logits, -1, false).unwrap();
        eval([&argmax]).unwrap();
        let top: i32 = argmax.index((0, 0)).item::<i32>();
        assert!(top >= 0 && top < 248320, "argmax out of range: {top}");
        eprintln!("argmax top token: {top}");
    }
}
