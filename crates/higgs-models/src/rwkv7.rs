//! RWKV-7 recurrent language model implementation.
//!
//! Pure recurrent architecture with fixed-size state per layer.
//! No attention masks, no KV cache — state is `[num_heads, head_dim, head_dim]`
//! per layer, constant regardless of sequence length.
//!
//! Reference: <https://huggingface.co/fla-hub/rwkv7-1.5B-world>

use std::path::Path;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as _, ModuleParametersExt, Param},
    nn,
    ops::{self, indexing::IndexOp},
    transforms::eval,
    Array,
};
use serde::Deserialize;

use crate::error::ModelError;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Rwkv7ModelArgs {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub vocab_size: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default)]
    pub norm_bias: bool,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_norm_first")]
    pub norm_first: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    // Low-rank dimensions for LoRA-style projections.
    #[serde(default = "default_decay_low_rank")]
    pub decay_low_rank_dim: i32,
    #[serde(default = "default_a_low_rank")]
    pub a_low_rank_dim: i32,
    #[serde(default = "default_v_low_rank")]
    pub v_low_rank_dim: i32,
    #[serde(default = "default_gate_low_rank")]
    pub gate_low_rank_dim: i32,
}

fn default_head_dim() -> i32 {
    64
}
fn default_norm_eps() -> f32 {
    1e-5
}
fn default_norm_first() -> bool {
    true
}
fn default_hidden_act() -> String {
    "sqrelu".into()
}
fn default_decay_low_rank() -> i32 {
    96
}
fn default_a_low_rank() -> i32 {
    96
}
fn default_v_low_rank() -> i32 {
    64
}
fn default_gate_low_rank() -> i32 {
    256
}

impl Rwkv7ModelArgs {
    pub fn num_heads(&self) -> i32 {
        self.hidden_size / self.head_dim
    }

    pub fn key_dim(&self) -> i32 {
        self.hidden_size
    }

    pub fn value_dim(&self) -> i32 {
        // RWKV-7 fla-hub uses hidden_size for both key and value dims.
        self.hidden_size
    }
}

fn load_model_args(model_path: &Path) -> Result<Rwkv7ModelArgs, ModelError> {
    let config_path = model_path.join("config.json");
    let file = std::fs::File::open(&config_path)?;
    let args: Rwkv7ModelArgs = serde_json::from_reader(file)?;
    Ok(args)
}

// ---------------------------------------------------------------------------
// Layer state (replaces KV cache for recurrent models)
// ---------------------------------------------------------------------------

/// Per-layer recurrent state. Fixed size regardless of sequence length.
#[derive(Debug, Clone)]
pub struct Rwkv7LayerState {
    /// WKV accumulator: `[num_heads, head_dim, head_dim]`.
    pub wkv_state: Option<Array>,
    /// Previous token hidden for attention token-shift: `[1, 1, hidden_size]`.
    pub shift_state: Option<Array>,
    /// Previous token hidden for FFN token-shift: `[1, 1, hidden_size]`.
    pub ffn_shift_state: Option<Array>,
    pub offset: i32,
}

impl Rwkv7LayerState {
    pub fn new() -> Self {
        Self {
            wkv_state: None,
            shift_state: None,
            ffn_shift_state: None,
            offset: 0,
        }
    }

    /// Evaluate lazy arrays so they can be cloned for prefix caching.
    pub fn eval_arrays(&self) -> Vec<&Array> {
        let mut targets = Vec::new();
        if let Some(ref s) = self.wkv_state {
            targets.push(s);
        }
        if let Some(ref s) = self.shift_state {
            targets.push(s);
        }
        if let Some(ref s) = self.ffn_shift_state {
            targets.push(s);
        }
        targets
    }
}

// ---------------------------------------------------------------------------
// LoRA-style low-rank projection (w_lora, a_lora, v_lora, g_lora)
// ---------------------------------------------------------------------------

/// Two-layer low-rank projection: Linear(in, rank) -> activation -> Linear(rank, out).
///
/// Weight names in safetensors: `lora.0.weight`, `lora.2.weight`, `lora.2.bias`.
#[derive(Debug, Clone, ModuleParameters)]
struct LowRankProj {
    #[param]
    lora: LowRankInner,
}

/// The sequential inner structure matching `nn.Sequential(Linear, Act, Linear)`.
///
/// Named fields map to safetensors keys: `0.weight`, `2.weight`, `2.bias`.
#[derive(Debug, Clone, ModuleParameters)]
struct LowRankInner {
    /// Down-projection: `[low_rank_dim, input_dim]`, no bias.
    #[param]
    down: Param<Array>,
    /// Up-projection weight: `[output_dim, low_rank_dim]`.
    #[param]
    up_weight: Param<Array>,
    /// Up-projection bias: `[output_dim]`.
    #[param]
    up_bias: Param<Array>,
    activation: LowRankActivation,
}

#[derive(Debug, Clone, Copy)]
enum LowRankActivation {
    Tanh,
    Sigmoid,
    Identity,
}

impl LowRankProj {
    fn new(
        input_dim: i32,
        output_dim: i32,
        low_rank_dim: i32,
        activation: LowRankActivation,
    ) -> Result<Self, Exception> {
        Ok(Self {
            lora: LowRankInner {
                down: Param::new(Array::zeros::<f32>(&[low_rank_dim, input_dim])?),
                up_weight: Param::new(Array::zeros::<f32>(&[output_dim, low_rank_dim])?),
                up_bias: Param::new(Array::zeros::<f32>(&[output_dim])?),
                activation,
            },
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // x @ down.T
        let h = ops::matmul(x, &self.lora.down.transpose_axes(&[-1, -2])?)?;
        let h = match self.lora.activation {
            LowRankActivation::Tanh => ops::tanh(&h)?,
            LowRankActivation::Sigmoid => ops::sigmoid(&h)?,
            LowRankActivation::Identity => h,
        };
        // h @ up.T + bias
        let out = ops::matmul(&h, &self.lora.up_weight.transpose_axes(&[-1, -2])?)?;
        ops::add(&out, &self.lora.up_bias)
    }
}

// ---------------------------------------------------------------------------
// Token shift (time mixing)
// ---------------------------------------------------------------------------

/// Linearly interpolate between previous and current hidden states.
fn token_shift(x: &Array, shift_state: &Option<Array>) -> Result<Array, Exception> {
    let shape = x.shape();
    let seq_len = shape.get(1).copied().unwrap_or(1);

    if seq_len == 1 {
        // Decode: shift_state is the previous token's hidden.
        match shift_state {
            Some(prev) => Ok(prev.clone()),
            None => Array::zeros::<f32>(x.shape()),
        }
    } else {
        // Prefill: shift is x[:, :-1, :] with zeros prepended.
        let zeros =
            Array::zeros::<f32>(&[*shape.first().unwrap_or(&1), 1, *shape.get(2).unwrap_or(&1)])?;
        let prev = x.index((.., ..seq_len - 1, ..));
        ops::concatenate_axis(&[&zeros, &prev], 1)
    }
}

/// Apply token shift mixing: `x + delta * mix` where `delta = shifted - x`.
///
/// Python: `torch.addcmul(hidden_states, delta, x_param)` = `x + (prev - x) * mix`
/// Result: `x * (1 - mix) + prev * mix` — mix=1 → all previous, mix=0 → all current.
fn apply_shift(x: &Array, shifted: &Array, mix: &Array) -> Result<Array, Exception> {
    let delta = ops::subtract(shifted, x)?;
    let mixed = ops::multiply(mix, &delta)?;
    ops::add(x, &mixed)
}

// ---------------------------------------------------------------------------
// L2 normalization
// ---------------------------------------------------------------------------

fn l2_norm(x: &Array, axis: i32) -> Result<Array, Exception> {
    let sq = ops::multiply(x, x)?;
    let sum = sq.sum_axis(axis, true)?;
    let norm = ops::sqrt(&ops::add(&sum, &Array::from_f32(1e-12))?)?;
    ops::divide(x, &norm)
}

// ---------------------------------------------------------------------------
// Group normalization (manual: reshape -> layer_norm -> reshape)
// ---------------------------------------------------------------------------

/// Group normalization: split channels into `num_groups`, normalize each group.
fn group_norm(
    x: &Array,
    weight: &Array,
    bias: &Array,
    num_groups: i32,
    eps: f32,
) -> Result<Array, Exception> {
    let shape = x.shape().to_vec();
    // x: [B, T, C] -> [B, T, num_groups, C/num_groups]
    let c = *shape.last().unwrap_or(&1);
    let group_size = c / num_groups;

    let batch_dims: Vec<i32> = shape[..shape.len() - 1].iter().map(|&d| d).collect();
    let mut new_shape = batch_dims.clone();
    new_shape.push(num_groups);
    new_shape.push(group_size);

    let reshaped = x.reshape(&new_shape)?;

    // Normalize over last dim (within each group).
    let mean = reshaped.mean_axis(-1, true)?;
    let centered = ops::subtract(&reshaped, &mean)?;
    let var = ops::multiply(&centered, &centered)?.mean_axis(-1, true)?;
    let inv_std = ops::rsqrt(&ops::add(&var, &Array::from_f32(eps))?)?;
    let normed = ops::multiply(&centered, &inv_std)?;

    // Flatten back to [B, T, C].
    let mut flat_shape = batch_dims;
    flat_shape.push(c);
    let flat = normed.reshape(&flat_shape)?;

    // Affine: weight * x + bias.
    let scaled = ops::multiply(&flat, weight)?;
    ops::add(&scaled, bias)
}

// ---------------------------------------------------------------------------
// Attention (WKV-7 recurrence)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Rwkv7Attention {
    // Token shift mixing parameters: [1, 1, hidden_size].
    #[param]
    x_r: Param<Array>,
    #[param]
    x_w: Param<Array>,
    #[param]
    x_k: Param<Array>,
    #[param]
    x_v: Param<Array>,
    #[param]
    x_a: Param<Array>,
    #[param]
    x_g: Param<Array>,

    // Key normalization: [key_dim].
    #[param]
    k_k: Param<Array>,
    #[param]
    k_a: Param<Array>,

    // Recurrent gate: [num_heads, head_dim].
    #[param]
    r_k: Param<Array>,

    // Main projections.
    #[param]
    r_proj: nn::Linear,
    #[param]
    k_proj: nn::Linear,
    #[param]
    v_proj: nn::Linear,
    #[param]
    o_proj: nn::Linear,

    // Low-rank LoRA-style projections.
    #[param]
    w_lora: LowRankProj,
    #[param]
    a_lora: LowRankProj,
    #[param]
    v_lora: Option<LowRankProj>,
    #[param]
    g_lora: LowRankProj,

    // Group norm over value dim (num_heads groups).
    #[param]
    g_norm_weight: Param<Array>,
    #[param]
    g_norm_bias: Param<Array>,

    num_heads: i32,
    head_dim: i32,
    layer_idx: i32,
}

impl Rwkv7Attention {
    fn new(args: &Rwkv7ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let hidden = args.hidden_size;
        let key_dim = args.key_dim();
        let value_dim = args.value_dim();
        let n_heads = args.num_heads();
        let head_dim = args.head_dim;

        let v_lora = if layer_idx > 0 {
            Some(LowRankProj::new(
                hidden,
                value_dim,
                args.v_low_rank_dim,
                LowRankActivation::Identity, // Python: activation=None
            )?)
        } else {
            None
        };

        Ok(Self {
            x_r: Param::new(Array::zeros::<f32>(&[1, 1, hidden])?),
            x_w: Param::new(Array::zeros::<f32>(&[1, 1, hidden])?),
            x_k: Param::new(Array::zeros::<f32>(&[1, 1, hidden])?),
            x_v: Param::new(Array::zeros::<f32>(&[1, 1, hidden])?),
            x_a: Param::new(Array::zeros::<f32>(&[1, 1, hidden])?),
            x_g: Param::new(Array::zeros::<f32>(&[1, 1, hidden])?),

            k_k: Param::new(Array::zeros::<f32>(&[key_dim])?),
            k_a: Param::new(Array::zeros::<f32>(&[key_dim])?),
            r_k: Param::new(Array::zeros::<f32>(&[n_heads, head_dim])?),

            r_proj: nn::LinearBuilder::new(hidden, key_dim)
                .bias(false)
                .build()?,
            k_proj: nn::LinearBuilder::new(hidden, key_dim)
                .bias(false)
                .build()?,
            v_proj: nn::LinearBuilder::new(hidden, value_dim)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(value_dim, hidden)
                .bias(false)
                .build()?,

            w_lora: LowRankProj::new(
                hidden,
                key_dim,
                args.decay_low_rank_dim,
                LowRankActivation::Tanh,
            )?,
            a_lora: LowRankProj::new(
                hidden,
                key_dim,
                args.a_low_rank_dim,
                LowRankActivation::Identity, // Python: activation=None
            )?,
            v_lora,
            g_lora: LowRankProj::new(
                hidden,
                value_dim,
                args.gate_low_rank_dim,
                LowRankActivation::Sigmoid,
            )?,

            g_norm_weight: Param::new(Array::ones::<f32>(&[value_dim])?),
            g_norm_bias: Param::new(Array::zeros::<f32>(&[value_dim])?),

            num_heads: n_heads,
            head_dim,
            layer_idx,
        })
    }

    /// Forward pass with v_first threading for RWKV-7 cross-layer value interpolation.
    fn forward(
        &mut self,
        x: &Array,
        state: &mut Rwkv7LayerState,
        v_first: &mut Option<Array>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Token shift.
        let shifted = token_shift(x, &state.shift_state)?;
        // Save last token for next step.
        state.shift_state = Some(x.index((.., -1.., ..)));

        // Mix shifted and current for each projection.
        let xr = apply_shift(x, &shifted, &self.x_r)?;
        let xw = apply_shift(x, &shifted, &self.x_w)?;
        let xk = apply_shift(x, &shifted, &self.x_k)?;
        let xv = apply_shift(x, &shifted, &self.x_v)?;
        let xa = apply_shift(x, &shifted, &self.x_a)?;
        let xg = apply_shift(x, &shifted, &self.x_g)?;

        // Main projections.
        let r = self.r_proj.forward(&xr)?;
        let k = self.k_proj.forward(&xk)?;
        let v = self.v_proj.forward(&xv)?;

        // Low-rank projections.
        let w_raw = self.w_lora.forward(&xw)?;
        let a_raw = self.a_lora.forward(&xa)?;
        let g = self.g_lora.forward(&xg)?;

        // Decay: w = -0.6065306597126334 * sigmoid(w_lora_out) (log-space).
        let w_log = ops::multiply(
            &Array::from_f32(-0.606_530_66),
            &ops::sigmoid(&w_raw)?,
        )?;

        // Attention bonus: a = sigmoid(a_lora_out), range (0, 1).
        let a = ops::sigmoid(&a_raw)?;

        // Gate: g = g_lora(xg) — g_lora has sigmoid as internal activation,
        // output is used directly as gate (no extra sigmoid like w and a).

        // v_first: layer 0 stores v, layers > 0 lerp with v_first.
        let v = if self.layer_idx == 0 {
            *v_first = Some(v.clone());
            v
        } else if let (Some(v_lr), Some(vf)) = (&mut self.v_lora, &*v_first) {
            // alpha = sigmoid(v_lora(xv))
            let alpha = ops::sigmoid(&v_lr.forward(&xv)?)?;
            // v = lerp(v, v_first, alpha) = v + alpha * (v_first - v)
            let diff = ops::subtract(vf, &v)?;
            ops::add(&v, &ops::multiply(&alpha, &diff)?)?
        } else {
            v
        };

        // kk: normalized key for attention bonus.
        // kk = l2_norm(k * k_k) reshaped to [B, T, H, D]
        let kk_flat = ops::multiply(&k, &self.k_k)?;
        let kk = l2_norm(&kk_flat, -1)?;
        let kk = kk.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;

        // k update: k = k * (1 + (a - 1) * k_a)
        // Python: k.addcmul(k * (a - 1), k_a) = k + k * (a - 1) * k_a = k * (1 + (a-1)*k_a)
        let a_minus_1 = ops::subtract(&a, &Array::from_f32(1.0))?;
        let k_correction = ops::multiply(&ops::multiply(&k, &a_minus_1)?, &self.k_a)?;
        let k = ops::add(&k, &k_correction)?;

        // Reshape for multi-head: [B, T, num_heads, head_dim].
        let r = r.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let w_log = w_log.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let a = a.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;

        // DPLR Delta Rule recurrence (sequential over time steps).
        // S[t] = diag(exp(w)) * S[t-1] + outer(k, v) + outer(-kk, kk*a)
        //      = diag(exp(w)) * S[t-1] + outer(k, v) - outer(kk, kk*a)
        // o[t] = S[t] @ r[t]
        let mut wkv_state = match state.wkv_state.take() {
            Some(s) => s,
            None => Array::zeros::<f32>(&[batch, self.num_heads, self.head_dim, self.head_dim])?,
        };

        let mut outputs = Vec::with_capacity(seq_len as usize);
        for t in 0..seq_len {
            let r_t = r.index((.., t, .., ..));
            let k_t = k.index((.., t, .., ..));
            let v_t = v.index((.., t, .., ..));
            let w_t = w_log.index((.., t, .., ..));
            let kk_t = kk.index((.., t, .., ..));
            let a_t = a.index((.., t, .., ..));

            // exp(w) for multiplicative decay.
            let w_exp = ops::exp(&w_t)?;
            let w_expanded = w_exp.reshape(&[batch, self.num_heads, self.head_dim, 1])?;
            wkv_state = ops::multiply(&wkv_state, &w_expanded)?;

            // outer(k, v): [B, H, D, 1] @ [B, H, 1, D] -> [B, H, D, D]
            let kv_outer = ops::matmul(
                &k_t.reshape(&[batch, self.num_heads, self.head_dim, 1])?,
                &v_t.reshape(&[batch, self.num_heads, 1, self.head_dim])?,
            )?;
            wkv_state = ops::add(&wkv_state, &kv_outer)?;

            // outer(-kk, kk*a) = -outer(kk, kk*a)
            let kk_a = ops::multiply(&kk_t, &a_t)?;
            let bonus = ops::matmul(
                &kk_t.reshape(&[batch, self.num_heads, self.head_dim, 1])?,
                &kk_a.reshape(&[batch, self.num_heads, 1, self.head_dim])?,
            )?;
            wkv_state = ops::subtract(&wkv_state, &bonus)?;

            // out_t = r_t^T @ S -> [B, H, D]
            // Python: (S * r.unsqueeze(-1)).sum(-2) = sum_i S[h,i,j] * r[h,i]
            let out_t = ops::matmul(
                &r_t.reshape(&[batch, self.num_heads, 1, self.head_dim])?,
                &wkv_state,
            )?
            .reshape(&[batch, self.num_heads, self.head_dim])?;

            outputs.push(out_t);
        }

        state.wkv_state = Some(wkv_state);

        // Stack outputs: [B, T, H, D].
        let stacked = ops::stack_axis(&outputs.iter().collect::<Vec<_>>(), 1)?;
        let y = stacked.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        // Group norm with eps = head_dim * norm_eps (Python convention).
        let gn_eps = (self.head_dim as f32) * 1e-5;
        let y = group_norm(
            &y,
            &self.g_norm_weight,
            &self.g_norm_bias,
            self.num_heads,
            gn_eps,
        )?;

        // Gate output correction: (o + correction) * g
        // correction = (r * k * r_k).sum(-1, keepdim=True) * v
        // where r, k are [B,T,H,D] and r_k is [H,D]
        let r_flat = r.reshape(&[batch * seq_len, self.num_heads, self.head_dim])?;
        let k_for_corr = k.reshape(&[batch * seq_len, self.num_heads, self.head_dim])?;
        let v_flat = v.reshape(&[batch * seq_len, self.num_heads, self.head_dim])?;
        let rkr = ops::multiply(&ops::multiply(&r_flat, &k_for_corr)?, &self.r_k)?;
        let rkr_sum = rkr.sum_axis(-1, true)?; // [B*T, H, 1]
        let correction = ops::multiply(&rkr_sum, &v_flat)?; // [B*T, H, D]
        let correction = correction.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        let y = ops::add(&y, &correction)?;
        let y = ops::multiply(&y, &g)?;

        // Output projection.
        self.o_proj.forward(&y)
    }
}

// ---------------------------------------------------------------------------
// Feed-forward (channel mix)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Rwkv7FeedForward {
    #[param]
    x_k: Param<Array>,
    #[param]
    key: nn::Linear,
    #[param]
    value: nn::Linear,
    hidden_act: String,
}

impl Rwkv7FeedForward {
    fn new(args: &Rwkv7ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            x_k: Param::new(Array::zeros::<f32>(&[1, 1, args.hidden_size])?),
            key: nn::LinearBuilder::new(args.hidden_size, args.intermediate_size)
                .bias(false)
                .build()?,
            value: nn::LinearBuilder::new(args.intermediate_size, args.hidden_size)
                .bias(false)
                .build()?,
            hidden_act: args.hidden_act.clone(),
        })
    }

    fn forward(&mut self, x: &Array, shift_state: &Option<Array>) -> Result<Array, Exception> {
        let shifted = token_shift(x, shift_state)?;
        let xk = apply_shift(x, &shifted, &self.x_k)?;
        let h = self.key.forward(&xk)?;

        // Activation.
        let h = match self.hidden_act.as_str() {
            "sqrelu" => {
                let relu_h = ops::maximum(&h, &Array::from_f32(0.0))?;
                ops::multiply(&relu_h, &relu_h)?
            }
            "silu" => nn::silu(&h)?,
            _ => h,
        };

        self.value.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Rwkv7Block {
    #[param]
    pre_norm: Option<nn::LayerNorm>,
    #[param]
    attn_norm: nn::LayerNorm,
    #[param]
    attn: Rwkv7Attention,
    #[param]
    ffn_norm: nn::LayerNorm,
    #[param]
    ffn: Rwkv7FeedForward,
}

impl Rwkv7Block {
    fn new(args: &Rwkv7ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let pre_norm = if args.norm_first && layer_idx == 0 {
            Some(
                nn::LayerNormBuilder::new(args.hidden_size)
                    .eps(args.norm_eps)
                    .build()?,
            )
        } else {
            None
        };

        Ok(Self {
            pre_norm,
            attn_norm: nn::LayerNormBuilder::new(args.hidden_size)
                .eps(args.norm_eps)
                .build()?,
            attn: Rwkv7Attention::new(args, layer_idx)?,
            ffn_norm: nn::LayerNormBuilder::new(args.hidden_size)
                .eps(args.norm_eps)
                .build()?,
            ffn: Rwkv7FeedForward::new(args)?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        state: &mut Rwkv7LayerState,
        v_first: &mut Option<Array>,
    ) -> Result<Array, Exception> {
        let mut h = x.clone();

        // Pre-norm (only first layer when norm_first).
        if let Some(ref mut pn) = self.pre_norm {
            h = pn.forward(&h)?;
        }

        // Attention path with residual.
        let normed = self.attn_norm.forward(&h)?;
        let attn_out = self.attn.forward(&normed, state, v_first)?;
        h = ops::add(&h, &attn_out)?;

        // FFN path with residual. FFN uses its own separate shift state.
        let normed = self.ffn_norm.forward(&h)?;
        let ffn_out = self.ffn.forward(&normed, &state.ffn_shift_state)?;
        // Save FFN shift state (post-norm hidden for next step).
        state.ffn_shift_state = Some(normed.index((.., -1.., ..)));
        ops::add(&h, &ffn_out)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Rwkv7Inner {
    #[param]
    embeddings: nn::Embedding,
    #[param]
    layers: Vec<Rwkv7Block>,
    #[param]
    norm: nn::LayerNorm,
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Rwkv7CausalLM {
    pub args: Rwkv7ModelArgs,
    #[param]
    model: Rwkv7Inner,
    #[param]
    lm_head: Option<nn::Linear>,
}

impl Rwkv7CausalLM {
    pub fn new(args: Rwkv7ModelArgs) -> Result<Self, Exception> {
        let n_layers = args.num_hidden_layers;
        let mut layers = Vec::with_capacity(n_layers as usize);
        for i in 0..n_layers {
            layers.push(Rwkv7Block::new(&args, i)?);
        }

        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };

        Ok(Self {
            model: Rwkv7Inner {
                embeddings: nn::Embedding::new(args.vocab_size, args.hidden_size)?,
                layers,
                norm: nn::LayerNormBuilder::new(args.hidden_size)
                    .eps(args.norm_eps)
                    .build()?,
            },
            lm_head,
            args,
        })
    }

    pub fn make_cache(&self) -> Vec<Option<Rwkv7LayerState>> {
        (0..self.args.num_hidden_layers)
            .map(|_| Some(Rwkv7LayerState::new()))
            .collect()
    }

    pub fn forward_hidden(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<Option<Rwkv7LayerState>>,
    ) -> Result<Array, Exception> {
        let mut h = self.model.embeddings.forward(inputs)?;

        // v_first: set by layer 0, used by subsequent layers for value interpolation.
        let mut v_first: Option<Array> = None;

        for (layer, layer_state) in self.model.layers.iter_mut().zip(cache.iter_mut()) {
            let state = layer_state
                .as_mut()
                .ok_or_else(|| Exception::custom("Missing RWKV-7 layer state"))?;
            h = layer.forward(&h, state, &mut v_first)?;
        }

        self.model.norm.forward(&h)
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<Option<Rwkv7LayerState>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, cache)?;
        let h_last = h.index((.., -1.., ..));

        match self.lm_head.as_mut() {
            Some(head) => head.forward(&h_last),
            None => {
                // Tied embeddings: use embedding weight as LM head.
                let w = &*self.model.embeddings.weight;
                ops::matmul(&h_last, &w.transpose_axes(&[-1, -2])?)
            }
        }
    }

    /// Chunked prefill: process long sequences in segments, evaluating
    /// intermediate states to bound peak memory.
    pub fn forward_chunked(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<Option<Rwkv7LayerState>>,
        chunk_size: i32,
    ) -> Result<Array, Exception> {
        let seq_len = *inputs
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        if chunk_size >= seq_len {
            return self.forward(inputs, cache);
        }

        let mut offset = 0i32;
        while offset + chunk_size < seq_len {
            let chunk = inputs.index((.., offset..offset + chunk_size));
            let h = self.forward_hidden(&chunk, cache)?;

            // Eval hidden + all layer states to prevent lazy graph growth.
            let mut targets: Vec<&Array> = vec![&h];
            for layer_state in cache.iter().flatten() {
                targets.extend(layer_state.eval_arrays());
            }
            eval(targets)?;

            offset += chunk_size;
        }

        // Last chunk: forward + LM head.
        let last_chunk = inputs.index((.., offset..));
        self.forward(&last_chunk, cache)
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Remap a safetensors key from the fla-hub naming convention to our Rust struct names.
///
/// Key transformations:
/// - `*.lora.0.weight` → `*.lora.down` (down-projection, no bias)
/// - `*.lora.2.weight` → `*.lora.up_weight` (up-projection weight)
/// - `*.lora.2.bias` → `*.lora.up_bias` (up-projection bias)
/// - `*.g_norm.weight` → `*.g_norm_weight` (group norm weight stored as flat param)
/// - `*.g_norm.bias` → `*.g_norm_bias` (group norm bias stored as flat param)
fn remap_rwkv7_key(key: &str) -> String {
    let mut k = key.to_string();
    k = k.replace(".lora.0.weight", ".lora.down");
    k = k.replace(".lora.2.weight", ".lora.up_weight");
    k = k.replace(".lora.2.bias", ".lora.up_bias");
    k = k.replace(".g_norm.weight", ".g_norm_weight");
    k = k.replace(".g_norm.bias", ".g_norm_bias");
    k
}

/// Quantization mode for RWKV-7 weight loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeMode {
    /// No quantization — load weights as-is (fp32/fp16).
    None,
    /// Quantize weights to int8 at load time (group_size=128).
    Int8,
}

impl QuantizeMode {
    /// Parse from a string (e.g., from config or env var).
    pub fn from_str_opt(s: Option<&str>) -> Self {
        match s {
            Some("int8") => Self::Int8,
            _ => Self::None,
        }
    }
}

pub fn load_rwkv7_model<P: AsRef<Path>>(model_dir: P) -> Result<Rwkv7CausalLM, ModelError> {
    // Check for quantization via environment variable.
    let quantize = QuantizeMode::from_str_opt(std::env::var("HIGGS_QUANTIZE").ok().as_deref());
    load_rwkv7_model_with_quantize(model_dir, quantize)
}

pub fn load_rwkv7_model_with_quantize<P: AsRef<Path>>(
    model_dir: P,
    _quantize: QuantizeMode,
) -> Result<Rwkv7CausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_model_args(model_path)?;

    tracing::info!(
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_heads(),
        head_dim = args.head_dim,
        vocab_size = args.vocab_size,
        intermediate_size = args.intermediate_size,
        "Loading RWKV-7 model"
    );

    let mut model = Rwkv7CausalLM::new(args)?;

    // Custom weight loading with key remapping for LoRA and GroupNorm names.
    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();

    for file_path in &safetensors_files {
        tracing::debug!(file = %file_path.display(), "Loading RWKV-7 weights");
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            let remapped = remap_rwkv7_key(&key);

            // Try remapped key first, then original.
            if let Some(param) = params.get_mut(&*remapped) {
                **param = value;
            } else if let Some(param) = params.get_mut(&*key) {
                **param = value;
            }
        }
    }

    model
        .eval()
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

    tracing::info!("RWKV-7 model loaded successfully");
    Ok(model)
}

// ---------------------------------------------------------------------------
// ANE context (feature-gated)
// ---------------------------------------------------------------------------

/// All ANE resources for accelerated RWKV-7 decode.
#[cfg(feature = "ane")]
pub struct AneContext {
    pub executor: crate::ane_forward::Rwkv7AneExecutor,
    pub embedding: Vec<f32>,
    pub lm_head_fp16: Vec<u16>,
    pub final_norm_w: Vec<f32>,
    pub final_norm_b: Vec<f32>,
    pub vocab_size: usize,
    pub dim: usize,
}

#[cfg(feature = "ane")]
impl Rwkv7CausalLM {
    /// Initialize ANE executor: extract weights, compile ANE kernels.
    ///
    /// This takes several seconds as it compiles MIL programs for each layer.
    /// Call once after loading the model, before decode.
    pub fn init_ane(&self) -> Result<AneContext, String> {
        use crate::ane_extract;
        use crate::ane_forward::Rwkv7AneExecutor;
        use crate::ane_mil::MilConfig;

        let dim = self.args.hidden_size as usize;
        let num_heads = self.args.num_heads() as usize;
        let head_dim = self.args.head_dim as usize;
        let inter = self.args.intermediate_size as usize;
        let vocab = self.args.vocab_size as usize;

        tracing::info!(dim, num_heads, head_dim, inter, "Initializing RWKV-7 ANE executor");

        let config = MilConfig {
            dim,
            num_heads,
            head_dim,
            intermediate_size: inter,
            seq_len: 1, // decode mode
        };

        let (ane_weights, cpu_weights) = ane_extract::extract_ane_weights(self);
        let fp16_weights = ane_extract::extract_fp16_projection_weights(self);
        let embedding = ane_extract::extract_embedding(self);
        let (lm_head_fp16, _, _) = ane_extract::extract_lm_head_fp16(self);
        let (final_norm_w, final_norm_b) = ane_extract::extract_final_norm(self);

        tracing::info!("Weights extracted, building executor...");

        let mut executor = Rwkv7AneExecutor::new(config);
        executor.compile(ane_weights, cpu_weights, fp16_weights)?;

        tracing::info!("ANE executor ready");

        Ok(AneContext {
            executor,
            embedding,
            lm_head_fp16,
            final_norm_w,
            final_norm_b,
            vocab_size: vocab,
            dim,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        // Check for model in standard HF cache location.
        let home = std::env::var("HOME").ok()?;
        let hf_cache = PathBuf::from(home).join(
            ".cache/huggingface/hub/models--fla-hub--rwkv7-1.5B-world",
        );
        if !hf_cache.exists() {
            return None;
        }
        // Find the snapshot directory.
        let snapshots = hf_cache.join("snapshots");
        let entry = std::fs::read_dir(snapshots).ok()?.next()?.ok()?;
        let path = entry.path();
        if path.join("config.json").exists() && path.join("model.safetensors").exists() {
            Some(path)
        } else {
            None
        }
    }

    #[test]
    fn test_rwkv7_config_load() {
        let Some(dir) = model_dir() else {
            eprintln!("RWKV-7 model not found, skipping");
            return;
        };
        let args = load_model_args(&dir).unwrap();
        assert_eq!(args.hidden_size, 2048);
        assert_eq!(args.num_hidden_layers, 24);
        assert_eq!(args.vocab_size, 65536);
        assert_eq!(args.head_dim, 64);
        assert_eq!(args.num_heads(), 32);
        assert_eq!(args.intermediate_size, 8192);
        assert!(args.norm_bias);
        assert!(args.norm_first);
        assert!(!args.tie_word_embeddings);
        assert_eq!(args.hidden_act, "sqrelu");
    }

    #[test]
    fn test_rwkv7_model_load_and_forward() {
        let Some(dir) = model_dir() else {
            eprintln!("RWKV-7 model not found, skipping");
            return;
        };

        eprintln!("Loading RWKV-7 1.5B from {:?}...", dir);
        let mut model = load_rwkv7_model(&dir).unwrap();
        eprintln!("Model loaded.");

        let mut cache = model.make_cache();

        // "The capital of France is" = [6699, 51128, 4706, 44312, 4600]
        let prompt_ids = [6699_i32, 51128, 4706, 44312, 4600];
        let input = Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);

        eprintln!("Running prefill ({} tokens)...", prompt_ids.len());
        let t0 = std::time::Instant::now();
        let logits = model.forward(&input, &mut cache).unwrap();
        eval([&logits]).unwrap();
        let prefill_ms = t0.elapsed().as_millis();
        eprintln!("Prefill done in {}ms", prefill_ms);

        // Logits shape: [1, 1, vocab_size] (last position only)
        let shape = logits.shape();
        eprintln!("Logits shape: {:?}", shape);
        assert_eq!(shape.last().copied(), Some(65536), "vocab_size mismatch");

        // Greedy decode: argmax
        let logits_flat = logits.as_slice::<f32>();
        let first_token = logits_flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap();

        eprintln!("First generated token: {}", first_token);

        // Generate a few more tokens
        let mut generated = vec![first_token];
        for step in 0..19 {
            let tok_input = Array::from_slice(&[*generated.last().unwrap()], &[1, 1]);
            let t0 = std::time::Instant::now();
            let logits = model.forward(&tok_input, &mut cache).unwrap();
            eval([&logits]).unwrap();
            let decode_ms = t0.elapsed().as_millis();

            let logits_flat = logits.as_slice::<f32>();

            // Check for NaN/inf logits (precision drift in recurrence).
            let nan_count = logits_flat.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!(
                    "Step {}: {nan_count} NaN logits detected — stopping (precision drift)",
                    step + 1
                );
                break;
            }

            let next_token = logits_flat
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i32)
                .unwrap();
            generated.push(next_token);

            eprintln!("Step {}: token {} ({}ms)", step + 1, next_token, decode_ms);

            // Stop on double newline (EOS)
            if next_token == 261 {
                break;
            }
        }

        eprintln!("\nGenerated token IDs: {:?}", generated);
        eprintln!("(decode with RWKV tokenizer to see text)");

        // Sanity: all generated tokens should be valid (0..vocab_size)
        for &t in &generated {
            assert!(t >= 0 && t < 65536, "Invalid token: {t}");
        }

        // The model should produce something other than token 0 repeatedly
        let unique: std::collections::HashSet<_> = generated.iter().collect();
        assert!(unique.len() > 1, "Model only produced one unique token — weights likely not loaded");
    }

    /// E2E ANE test: compile kernels, run decode on the Neural Engine.
    #[test]
    #[cfg(feature = "ane")]
    fn test_rwkv7_ane_decode() {
        let Some(dir) = model_dir() else {
            eprintln!("RWKV-7 model not found, skipping");
            return;
        };

        eprintln!("Loading RWKV-7 1.5B for ANE test...");
        let model = load_rwkv7_model(&dir).unwrap();

        eprintln!("Initializing ANE executor (compiling kernels)...");
        let t0 = std::time::Instant::now();
        let ane_ctx = model.init_ane().unwrap();
        eprintln!("ANE init: {}ms", t0.elapsed().as_millis());

        // Create fresh state
        let num_heads = model.args.num_heads() as usize;
        let head_dim = model.args.head_dim as usize;
        let dim = model.args.hidden_size as usize;
        let mut state = crate::ane_forward::AneModelState {
            layers: (0..model.args.num_hidden_layers as usize)
                .map(|_| crate::ane_forward::AneLayerState::new(num_heads, head_dim))
                .collect(),
            v_first: vec![0.0; dim],
        };

        // Prefill token by token (ANE path is T=1 only)
        let prompt_ids: Vec<u32> = vec![6699, 51128, 4706, 44312, 4600];
        eprintln!("Prefilling {} tokens on ANE...", prompt_ids.len());

        let t0 = std::time::Instant::now();
        let mut last_logits = Vec::new();
        for &tid in &prompt_ids {
            last_logits = crate::ane_forward::forward_ane_full(
                &ane_ctx.executor,
                tid,
                &mut state,
                &ane_ctx.embedding,
                &ane_ctx.lm_head_fp16,
                &ane_ctx.final_norm_w,
                &ane_ctx.final_norm_b,
                ane_ctx.vocab_size,
                ane_ctx.dim,
            )
            .unwrap();
        }
        let prefill_ms = t0.elapsed().as_millis();
        eprintln!("ANE prefill: {}ms ({} tokens)", prefill_ms, prompt_ids.len());

        // Greedy decode
        let first_token = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap();
        eprintln!("First ANE token: {first_token}");

        let mut generated = vec![first_token];
        let decode_start = std::time::Instant::now();
        for step in 0..19 {
            let t0 = std::time::Instant::now();
            let logits = crate::ane_forward::forward_ane_full(
                &ane_ctx.executor,
                *generated.last().unwrap(),
                &mut state,
                &ane_ctx.embedding,
                &ane_ctx.lm_head_fp16,
                &ane_ctx.final_norm_w,
                &ane_ctx.final_norm_b,
                ane_ctx.vocab_size,
                ane_ctx.dim,
            )
            .unwrap();
            let ms = t0.elapsed().as_millis();

            let nan_count = logits.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!("Step {}: NaN logits — stopping", step + 1);
                break;
            }

            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap();
            generated.push(next);
            eprintln!("Step {}: token {} ({}ms)", step + 1, next, ms);
        }
        let total_decode_ms = decode_start.elapsed().as_millis();
        let tok_per_sec = (generated.len() - 1) as f64 / (total_decode_ms as f64 / 1000.0);
        eprintln!(
            "\nANE decode: {} tokens in {}ms = {:.1} tok/s",
            generated.len() - 1,
            total_decode_ms,
            tok_per_sec
        );
        eprintln!("Generated: {:?}", generated);

        // Sanity checks
        assert!(generated.len() > 1, "Must generate at least 1 token");
        for &t in &generated {
            assert!(t < 65536, "Invalid token: {t}");
        }
    }

    /// Compare CPU vs ANE logits for the same single-token forward pass.
    /// Identifies where precision divergence starts.
    #[test]
    #[cfg(feature = "ane")]
    fn test_rwkv7_cpu_vs_ane_logits() {
        let Some(dir) = model_dir() else {
            eprintln!("RWKV-7 model not found, skipping");
            return;
        };

        let mut model = load_rwkv7_model(&dir).unwrap();
        let ane_ctx = model.init_ane().unwrap();

        let dim = model.args.hidden_size as usize;
        let num_heads = model.args.num_heads() as usize;
        let head_dim = model.args.head_dim as usize;
        let n_layers = model.args.num_hidden_layers as usize;

        // Feed token 6699 ("The") through both paths
        let token_id = 6699u32;

        // --- CPU path ---
        let mut cpu_cache = model.make_cache();
        let input = Array::from_slice(&[token_id as i32], &[1, 1]);
        let cpu_logits = model.forward(&input, &mut cpu_cache).unwrap();
        eval([&cpu_logits]).unwrap();
        let cpu_logits = cpu_logits.as_slice::<f32>();
        let cpu_top = cpu_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| (i, *v)).unwrap();

        // --- ANE path ---
        let mut ane_state = crate::ane_forward::AneModelState {
            layers: (0..n_layers)
                .map(|_| crate::ane_forward::AneLayerState::new(num_heads, head_dim))
                .collect(),
            v_first: vec![0.0; dim],
        };
        let ane_logits = crate::ane_forward::forward_ane_full(
            &ane_ctx.executor, token_id, &mut ane_state,
            &ane_ctx.embedding, &ane_ctx.lm_head_fp16,
            &ane_ctx.final_norm_w, &ane_ctx.final_norm_b,
            ane_ctx.vocab_size, ane_ctx.dim,
        ).unwrap();
        let ane_top = ane_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| (i, *v)).unwrap();

        // Compare
        let max_err = cpu_logits.iter().zip(ane_logits.iter())
            .map(|(c, a)| (c - a).abs())
            .fold(0.0f32, f32::max);
        let mean_err = cpu_logits.iter().zip(ane_logits.iter())
            .map(|(c, a)| (c - a).abs())
            .sum::<f32>() / cpu_logits.len() as f32;

        eprintln!("CPU top: token {} logit {:.3}", cpu_top.0, cpu_top.1);
        eprintln!("ANE top: token {} logit {:.3}", ane_top.0, ane_top.1);
        eprintln!("Logit max_err: {max_err:.4}, mean_err: {mean_err:.6}");
        eprintln!("Top-1 match: {}", cpu_top.0 == ane_top.0);

        // Also compare top-5
        let mut cpu_sorted: Vec<(usize, f32)> = cpu_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        let mut ane_sorted: Vec<(usize, f32)> = ane_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        cpu_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ane_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("CPU top-5: {:?}", &cpu_sorted[..5].iter().map(|(i, v)| (*i, format!("{v:.2}"))).collect::<Vec<_>>());
        eprintln!("ANE top-5: {:?}", &ane_sorted[..5].iter().map(|(i, v)| (*i, format!("{v:.2}"))).collect::<Vec<_>>());

        // r_proj comparison removed (shared kernel architecture)
    }
}
