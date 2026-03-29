//! RWKV-7 recurrent language model implementation.
//!
//! Pure recurrent architecture with fixed-size state per layer.
//! No attention masks, no KV cache — state is `[num_heads, head_dim, head_dim]`
//! per layer, constant regardless of sequence length.
//!
//! Reference: <https://huggingface.co/fla-hub/rwkv7-1.5B-world>

use std::path::Path;

use mlx_rs::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt, Param},
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
    /// Previous token hidden for token-shift: `[1, 1, hidden_size]`.
    pub shift_state: Option<Array>,
    pub offset: i32,
}

impl Rwkv7LayerState {
    pub fn new() -> Self {
        Self {
            wkv_state: None,
            shift_state: None,
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

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // x @ down.T
        let h = ops::matmul(x, &ops::transpose(&self.lora.down, &[-1, -2])?)?;
        let h = match self.lora.activation {
            LowRankActivation::Tanh => ops::tanh(&h)?,
            LowRankActivation::Sigmoid => ops::sigmoid(&h)?,
        };
        // h @ up.T + bias
        let out = ops::matmul(&h, &ops::transpose(&self.lora.up_weight, &[-1, -2])?)?;
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
            None => Ok(Array::zeros_like(x)?),
        }
    } else {
        // Prefill: shift is x[:, :-1, :] with zeros prepended.
        let zeros =
            Array::zeros::<f32>(&[*shape.first().unwrap_or(&1), 1, *shape.get(2).unwrap_or(&1)])?;
        let prev = x.index((.., ..seq_len - 1, ..));
        ops::concatenate(&[&zeros, &prev], 1)
    }
}

/// Apply token shift mixing: `x_mix * x + (1 - x_mix) * shifted`.
fn apply_shift(x: &Array, shifted: &Array, mix: &Array) -> Result<Array, Exception> {
    // mix * x + (1 - mix) * shifted = shifted + mix * (x - shifted)
    let diff = ops::subtract(x, shifted)?;
    let mixed = ops::multiply(mix, &diff)?;
    ops::add(shifted, &mixed)
}

// ---------------------------------------------------------------------------
// L2 normalization
// ---------------------------------------------------------------------------

fn l2_norm(x: &Array, axis: i32) -> Result<Array, Exception> {
    let sq = ops::multiply(x, x)?;
    let sum = ops::sum_axis(&sq, &[axis], true)?;
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
    let mean = ops::mean_axis(&reshaped, &[-1], true)?;
    let centered = ops::subtract(&reshaped, &mean)?;
    let var = ops::mean_axis(&ops::multiply(&centered, &centered)?, &[-1], true)?;
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
                LowRankActivation::Tanh,
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
                LowRankActivation::Tanh,
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

    fn forward(&self, x: &Array, state: &mut Rwkv7LayerState) -> Result<Array, Exception> {
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
        let w = self.w_lora.forward(&xw)?; // decay
        let a = self.a_lora.forward(&xa)?; // attention scaling
        let g = self.g_lora.forward(&xg)?; // gate (sigmoid)

        // Value interpolation with v_first (layer 0 uses v directly).
        let v = if let Some(ref v_lora) = self.v_lora {
            let v_mix = v_lora.forward(&xv)?;
            ops::add(&v, &v_mix)?
        } else {
            v
        };

        // Key normalization: k = l2_norm(k * k_k)
        let k_scaled = ops::multiply(&k, &self.k_k)?;
        let k = l2_norm(&k_scaled, -1)?;

        // a scaling: a = sigmoid(a) * k_a
        let a = ops::multiply(&ops::sigmoid(&a)?, &self.k_a)?;

        // Reshape for multi-head: [B, T, num_heads, head_dim].
        let r = r.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let w = w.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let a = a.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;

        // Decay: w = exp(-exp(w)) element-wise.
        let w = ops::exp(&ops::negative(&ops::exp(&w)?)?)?;

        // WKV recurrence (sequential over time steps).
        let mut wkv_state = match state.wkv_state.take() {
            Some(s) => s,
            None => Array::zeros::<f32>(&[batch, self.num_heads, self.head_dim, self.head_dim])?,
        };

        let mut outputs = Vec::with_capacity(seq_len as usize);
        for t in 0..seq_len {
            let r_t = r.index((.., t, .., ..)); // [B, H, D]
            let k_t = k.index((.., t, .., ..)); // [B, H, D]
            let v_t = v.index((.., t, .., ..)); // [B, H, D]
            let w_t = w.index((.., t, .., ..)); // [B, H, D]
            let a_t = a.index((.., t, .., ..)); // [B, H, D]

            // State update: S = diag(w) * S + k_outer_v - a_outer_(S @ a)
            // Simplified: S = diag(w) * S + outer(k, v) (core WKV recurrence)
            //
            // Full RWKV-7 recurrence with attention bonus:
            //   sa = S @ a_t  (attention to state)
            //   S = diag(w_t) * S + outer(k_t, v_t) - outer(a_t, sa)
            //   out_t = S @ r_t

            // sa = einsum("bhij,bhj->bhi", S, a_t)
            let sa = ops::matmul(
                &wkv_state,
                &a_t.reshape(&[batch, self.num_heads, self.head_dim, 1])?,
            )?
            .reshape(&[batch, self.num_heads, self.head_dim])?;

            // S = diag(w_t) * S
            // w_t: [B, H, D] -> [B, H, D, 1] for broadcasting over last dim.
            let w_expanded = w_t.reshape(&[batch, self.num_heads, self.head_dim, 1])?;
            wkv_state = ops::multiply(&wkv_state, &w_expanded)?;

            // S += outer(k_t, v_t) - outer(a_t, sa)
            let kv_outer = ops::matmul(
                &k_t.reshape(&[batch, self.num_heads, self.head_dim, 1])?,
                &v_t.reshape(&[batch, self.num_heads, 1, self.head_dim])?,
            )?;
            let a_sa_outer = ops::matmul(
                &a_t.reshape(&[batch, self.num_heads, self.head_dim, 1])?,
                &sa.reshape(&[batch, self.num_heads, 1, self.head_dim])?,
            )?;
            wkv_state = ops::add(&wkv_state, &ops::subtract(&kv_outer, &a_sa_outer)?)?;

            // out_t = S @ r_t -> [B, H, D]
            let out_t = ops::matmul(
                &wkv_state,
                &r_t.reshape(&[batch, self.num_heads, self.head_dim, 1])?,
            )?
            .reshape(&[batch, self.num_heads, self.head_dim])?;

            outputs.push(out_t);
        }

        state.wkv_state = Some(wkv_state);

        // Stack outputs: [B, T, H, D].
        let stacked = ops::stack(&outputs.iter().collect::<Vec<_>>(), 1)?;
        let y = stacked.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        // Group norm + gate.
        let y = group_norm(
            &y,
            &self.g_norm_weight,
            &self.g_norm_bias,
            self.num_heads,
            1e-5,
        )?;

        // Apply r_k correction: flatten r_k from [H, D] to [H*D].
        let r_k_flat = self.r_k.reshape(&[self.num_heads * self.head_dim])?;
        let y = ops::multiply(&y, &r_k_flat)?;

        // Gate: y = y * g (sigmoid was already applied in g_lora).
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

    fn forward(&self, x: &Array, shift_state: &Option<Array>) -> Result<Array, Exception> {
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
                    .bias(args.norm_bias)
                    .build()?,
            )
        } else {
            None
        };

        Ok(Self {
            pre_norm,
            attn_norm: nn::LayerNormBuilder::new(args.hidden_size)
                .eps(args.norm_eps)
                .bias(args.norm_bias)
                .build()?,
            attn: Rwkv7Attention::new(args, layer_idx)?,
            ffn_norm: nn::LayerNormBuilder::new(args.hidden_size)
                .eps(args.norm_eps)
                .bias(args.norm_bias)
                .build()?,
            ffn: Rwkv7FeedForward::new(args)?,
        })
    }

    fn forward(&self, x: &Array, state: &mut Rwkv7LayerState) -> Result<Array, Exception> {
        let mut h = x.clone();

        // Pre-norm (only first layer when norm_first).
        if let Some(ref pn) = self.pre_norm {
            h = pn.forward(&h)?;
        }

        // Attention path with residual.
        let normed = self.attn_norm.forward(&h)?;
        let attn_out = self.attn.forward(&normed, state)?;
        h = ops::add(&h, &attn_out)?;

        // FFN path with residual. FFN uses its own shift from the attention shift state.
        let normed = self.ffn_norm.forward(&h)?;
        let ffn_out = self.ffn.forward(&normed, &state.shift_state)?;
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
                    .bias(args.norm_bias)
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

        for (layer, layer_state) in self.model.layers.iter().zip(cache.iter_mut()) {
            let state = layer_state
                .as_mut()
                .ok_or_else(|| Exception::custom("Missing RWKV-7 layer state"))?;
            h = layer.forward(&h, state)?;
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

        match self.lm_head.as_ref() {
            Some(head) => head.forward(&h_last),
            None => {
                // Tied embeddings: use embedding weight as LM head.
                let w = &*self.model.embeddings.weight;
                ops::matmul(&h_last, &ops::transpose(w, &[-1, -2])?)
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
    quantize: QuantizeMode,
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
        quantize = ?quantize,
        "Loading RWKV-7 model"
    );

    let raw_model = Rwkv7CausalLM::new(args)?;

    // Apply quantization structure if requested.
    let mut model = if quantize == QuantizeMode::Int8 {
        tracing::info!("Applying int8 quantization (group_size=128, bits=8)");
        mlx_rs::nn::quantize(raw_model, 128, 8).map_err(|e| {
            ModelError::ShapeMismatch(format!("Failed to quantize RWKV-7 model: {e}"))
        })?
    } else {
        raw_model
    };

    // Custom weight loading with key remapping for LoRA and GroupNorm names.
    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let is_quantized = quantize == QuantizeMode::Int8;
    let mut params = model.parameters_mut().flatten();

    for file_path in &safetensors_files {
        tracing::debug!(file = %file_path.display(), "Loading RWKV-7 weights");
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            let remapped = remap_rwkv7_key(&key);

            // Try remapped key first, then original.
            let matched = params.get_mut(&*remapped).or_else(|| params.get_mut(&*key));

            if let Some(param) = matched {
                **param = value;
            } else if is_quantized {
                // For quantized models, also try the `.inner.` remapping.
                let inner_key = crate::remap_quantized_key(&remapped)
                    .or_else(|| crate::remap_quantized_key(&key));
                if let Some(inner) = inner_key {
                    if let Some(param) = params.get_mut(&*inner) {
                        **param = value;
                    }
                }
            }
        }
    }

    model
        .eval()
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

    tracing::info!("RWKV-7 model loaded successfully");
    Ok(model)
}
