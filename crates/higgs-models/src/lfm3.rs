//! Native `LiquidAI` LFM3 causal language model.
//!
//! LFM3 interleaves gated short-convolution, sliding-window attention and
//! global-attention layers. Its global attention uses a zero-tail partial `RoPE`:
//! frequencies retain the progression of the full head dimension while the
//! inactive tail receives zero angle.

use std::{collections::HashMap, path::Path};

use mlx_rs::{
    Array, Dtype,
    builder::Builder,
    error::Exception,
    fast,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
};
use serde::Deserialize;

use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    error::ModelError,
    quant_config::{QuantizationSettings, TensorQuant},
    qwen3_next::{ArraysCache, LayerCache, QEmbedding, QLinear, swiglu},
    utils::{AttentionMask, create_causal_mask, create_windowed_causal_mask},
};

const fn default_conv_kernel() -> i32 {
    7
}

const fn default_sliding_window() -> i32 {
    256
}

const fn default_true() -> bool {
    true
}

const fn default_partial_rotary_factor() -> f32 {
    1.0
}

const fn default_rope_type() -> String {
    String::new()
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Lfm3Args {
    pub model_type: String,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(rename = "head_dim", default)]
    pub attention_head_dim: Option<i32>,
    pub norm_eps: f32,
    #[serde(rename = "conv_L_cache", default = "default_conv_kernel")]
    pub conv_l_cache: i32,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    #[serde(default = "default_true")]
    pub use_sandwich_norm: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub vocab_size: i32,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: HashMap<String, RopeParameters>,
}

impl Lfm3Args {
    pub fn head_dim(&self) -> i32 {
        self.attention_head_dim.unwrap_or_else(|| {
            self.hidden_size
                .checked_div(self.num_attention_heads)
                .unwrap_or(0)
        })
    }

    fn layer_type(&self, layer: usize) -> Option<LayerType> {
        match self.layer_types.get(layer).map(String::as_str) {
            Some("conv") => Some(LayerType::Conv),
            Some("sliding_attention") => Some(LayerType::SlidingAttention),
            Some("full_attention") => Some(LayerType::FullAttention),
            _ => None,
        }
    }

    fn projection_widths(&self) -> Option<(i32, i32, i32)> {
        let head_dim = self.head_dim();
        if self.hidden_size <= 0
            || self.intermediate_size <= 0
            || self.vocab_size <= 0
            || self.num_attention_heads <= 0
            || self.num_key_value_heads <= 0
            || head_dim <= 0
        {
            return None;
        }
        let query_width = self.num_attention_heads.checked_mul(head_dim)?;
        if query_width != self.hidden_size {
            return None;
        }
        Some((
            query_width,
            self.num_key_value_heads.checked_mul(head_dim)?,
            self.hidden_size.checked_mul(3)?,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerType {
    Conv,
    SlidingAttention,
    FullAttention,
}

fn linear(q: &QuantizationSettings, path: &str) -> Result<QLinear, Exception> {
    let tensor = resolve_quantization(q, path);
    QLinear::new(tensor.group_size(), tensor.bits())
}

fn resolve_quantization(q: &QuantizationSettings, path: &str) -> TensorQuant {
    let prefixed = format!("language_model.{path}");
    if q.is_overridden(path) {
        q.resolve(path)
    } else if q.is_overridden(&prefixed) {
        q.resolve(&prefixed)
    } else {
        q.resolve(path)
    }
}

fn rms_norm(dim: i32, eps: f32) -> Result<nn::RmsNorm, Exception> {
    nn::RmsNormBuilder::new(dim).eps(eps).build()
}

fn empty() -> Param<Array> {
    Param::new(Array::from_slice::<f32>(&[], &[0]))
}

// ---------------------------------------------------------------------------
// Gated width-K short convolution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct ConvWeight {
    /// MLX depthwise Conv1d layout: `[D, K, 1]`.
    #[param]
    weight: Param<Array>,
}

#[derive(Debug, Clone, ModuleParameters)]
struct ShortConv {
    #[param]
    conv: ConvWeight,
    #[param]
    in_proj: QLinear,
    #[param]
    out_proj: QLinear,
    kernel: i32,
    taps: Option<Array>,
}

impl ShortConv {
    fn taps(&mut self) -> Result<Array, Exception> {
        if let Some(taps) = &self.taps {
            return Ok(taps.clone());
        }
        let taps = self
            .conv
            .weight
            .reshape(&[-1, self.kernel])?
            .transpose_axes(&[1, 0])?
            .as_dtype(Dtype::Float32)?;
        self.taps = Some(taps.clone());
        Ok(taps)
    }

    fn forward(&mut self, x: &Array, cache: &mut ArraysCache) -> Result<Array, Exception> {
        let projected = self.in_proj.forward(x)?;
        let parts = projected.split(3, Some(-1))?;
        let [b, c, value] = parts.as_slice() else {
            return Err(Exception::custom(
                "LFM3 conv projection must split into B, C, x",
            ));
        };
        let bx = b.multiply(value)?;
        let (batch, length, dim) = (x.dim(0), x.dim(1), x.dim(2));
        let state = cache.conv_state.take().map_or_else(
            || ops::zeros_dtype(&[batch, self.kernel - 1, dim], bx.dtype()),
            Ok,
        )?;
        let padded = ops::concatenate_axis(&[state, bx], 1)?;
        cache.conv_state = Some(padded.index((.., -(self.kernel - 1).., ..)));
        cache.conv_pos = self.kernel - 2;
        cache.offset += length;

        let padded_f32 = padded.as_dtype(Dtype::Float32)?;
        let taps = self.taps()?;
        let mut convolved = padded_f32
            .index((.., 0..length, ..))
            .multiply(taps.index(0))?;
        for tap in 1..self.kernel {
            convolved = convolved.add(
                padded_f32
                    .index((.., tap..tap + length, ..))
                    .multiply(taps.index(tap))?,
            )?;
        }
        self.out_proj
            .forward(&c.multiply(convolved.as_dtype(c.dtype())?)?)
    }
}

// ---------------------------------------------------------------------------
// Attention with LFM3 zero-tail RoPE and per-head output gate
// ---------------------------------------------------------------------------

fn apply_zero_tail_rope(
    x: &Array,
    offset: i32,
    base: f32,
    partial_factor: f32,
) -> Result<Array, Exception> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(Exception::custom("LFM3 RoPE expects [B, H, L, D]"));
    }
    let length = x.dim(2);
    let dim = x.dim(3);
    if dim <= 0 || dim % 2 != 0 {
        return Err(Exception::custom(
            "LFM3 RoPE head_dim must be positive and even",
        ));
    }
    let half = dim / 2;
    let dim_f32 = f32::from(
        u16::try_from(dim).map_err(|_| Exception::custom("LFM3 RoPE head_dim out of range"))?,
    );
    let active_threshold = (dim_f32 * partial_factor).clamp(0.0, dim_f32);
    let inv_freq = (0..half)
        .map(|i| -> Result<f32, Exception> {
            let i_f32 = f32::from(
                u16::try_from(i)
                    .map_err(|_| Exception::custom("LFM3 RoPE frequency index out of range"))?,
            );
            if 2.0 * (i_f32 + 1.0) <= active_threshold {
                Ok(base.powf(-2.0 * i_f32 / dim_f32))
            } else {
                Ok(0.0)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let positions = mlx_rs::arange!(start = offset, stop = offset + length)?
        .as_dtype(Dtype::Float32)?
        .reshape(&[length, 1])?;
    let frequencies = Array::from_slice(&inv_freq, &[1, half]);
    let angles = positions.multiply(frequencies)?;
    let dtype = x.dtype();
    // Hugging Face returns the f32 trig tables in the activation dtype, then
    // promotes those rounded values back to f32 for the rotation itself.
    let cos = ops::cos(&angles)?
        .reshape(&[1, 1, length, half])?
        .as_dtype(dtype)?
        .as_dtype(Dtype::Float32)?;
    let sin = ops::sin(&angles)?
        .reshape(&[1, 1, length, half])?
        .as_dtype(dtype)?
        .as_dtype(Dtype::Float32)?;
    let x_f32 = x.as_dtype(Dtype::Float32)?;
    let first = x_f32.index((.., .., .., ..half));
    let second = x_f32.index((.., .., .., half..));
    let rotated_first = first.multiply(&cos)?.subtract(second.multiply(&sin)?)?;
    let rotated_second = first.multiply(&sin)?.add(second.multiply(&cos)?)?;
    ops::concatenate_axis(&[rotated_first, rotated_second], -1)?.as_dtype(dtype)
}

#[derive(Debug, Clone, ModuleParameters)]
struct Attention {
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
    gate_proj: QLinear,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_theta: f32,
    partial_rotary_factor: f32,
}

impl Attention {
    fn new(
        args: &Lfm3Args,
        quant: &QuantizationSettings,
        path: &str,
        layer_type: LayerType,
    ) -> Result<Self, Exception> {
        let rope_name = match layer_type {
            LayerType::SlidingAttention => "sliding_attention",
            LayerType::FullAttention => "full_attention",
            LayerType::Conv => return Err(Exception::custom("conv layer cannot build attention")),
        };
        let rope = args
            .rope_parameters
            .get(rope_name)
            .ok_or_else(|| Exception::custom(format!("missing RoPE parameters for {rope_name}")))?;
        if !rope.rope_type.is_empty() && rope.rope_type != "default" {
            return Err(Exception::custom(format!(
                "unsupported LFM3 rope_type {}",
                rope.rope_type
            )));
        }
        let head_dim = args.head_dim();
        Ok(Self {
            q_proj: linear(quant, &format!("{path}.q_proj"))?,
            k_proj: linear(quant, &format!("{path}.k_proj"))?,
            v_proj: linear(quant, &format!("{path}.v_proj"))?,
            o_proj: linear(quant, &format!("{path}.o_proj"))?,
            q_norm: rms_norm(head_dim, args.norm_eps)?,
            k_norm: rms_norm(head_dim, args.norm_eps)?,
            gate_proj: linear(quant, &format!("{path}.gate_proj"))?,
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            head_dim,
            scale: f32::from(
                u16::try_from(head_dim)
                    .map_err(|_| Exception::custom("LFM3 attention head_dim out of range"))?,
            )
            .sqrt()
            .recip(),
            rope_theta: rope.rope_theta,
            partial_rotary_factor: rope.partial_rotary_factor,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut SteppingKeyValueCache,
    ) -> Result<Array, Exception> {
        let (batch, length) = (x.dim(0), x.dim(1));
        let heads =
            |array: Array, count: i32| array.reshape(&[batch, length, count, self.head_dim]);
        let q = self
            .q_norm
            .forward(&heads(self.q_proj.forward(x)?, self.n_heads)?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k_norm
            .forward(&heads(self.k_proj.forward(x)?, self.n_kv_heads)?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = heads(self.v_proj.forward(x)?, self.n_kv_heads)?.transpose_axes(&[0, 2, 1, 3])?;
        let offset = cache.offset();
        let queries =
            apply_zero_tail_rope(&q, offset, self.rope_theta, self.partial_rotary_factor)?;
        let new_keys =
            apply_zero_tail_rope(&k, offset, self.rope_theta, self.partial_rotary_factor)?;
        let (keys, values) = cache.update_and_fetch(new_keys, v)?;
        let attention = fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            self.scale,
            mask.map(fast::ScaledDotProductAttentionMask::from),
            None::<&Array>,
        )?
        .transpose_axes(&[0, 2, 1, 3])?;
        let gate = nn::sigmoid(&self.gate_proj.forward(x)?)?.expand_dims(-1)?;
        self.o_proj
            .forward(&attention.multiply(gate)?.reshape(&[batch, length, -1])?)
    }
}

// ---------------------------------------------------------------------------
// Dense SwiGLU and decoder stack
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct FeedForward {
    #[param]
    w1: QLinear,
    #[param]
    w2: QLinear,
    #[param]
    w3: QLinear,
}

impl FeedForward {
    fn new(quant: &QuantizationSettings, path: &str) -> Result<Self, Exception> {
        Ok(Self {
            w1: linear(quant, &format!("{path}.w1"))?,
            w2: linear(quant, &format!("{path}.w2"))?,
            w3: linear(quant, &format!("{path}.w3"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        self.w2
            .forward(&swiglu(&self.w1.forward(x)?, &self.w3.forward(x)?)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct DecoderLayer {
    #[param]
    self_attn: Option<Attention>,
    #[param]
    conv: Option<ShortConv>,
    #[param]
    feed_forward: FeedForward,
    #[param]
    operator_norm: nn::RmsNorm,
    #[param]
    post_operator_norm: Option<nn::RmsNorm>,
    #[param]
    ffn_norm: nn::RmsNorm,
    #[param]
    post_ffn_norm: Option<nn::RmsNorm>,
    layer_type: LayerType,
}

impl DecoderLayer {
    fn new(args: &Lfm3Args, quant: &QuantizationSettings, index: usize) -> Result<Self, Exception> {
        let layer_type = args.layer_type(index).ok_or_else(|| {
            Exception::custom(format!("invalid LFM3 layer type at index {index}"))
        })?;
        let path = format!("model.layers.{index}");
        let (self_attn, conv) = match layer_type {
            LayerType::Conv => (
                None,
                Some(ShortConv {
                    conv: ConvWeight { weight: empty() },
                    in_proj: linear(quant, &format!("{path}.conv.in_proj"))?,
                    out_proj: linear(quant, &format!("{path}.conv.out_proj"))?,
                    kernel: args.conv_l_cache,
                    taps: None,
                }),
            ),
            LayerType::SlidingAttention | LayerType::FullAttention => (
                Some(Attention::new(
                    args,
                    quant,
                    &format!("{path}.self_attn"),
                    layer_type,
                )?),
                None,
            ),
        };
        Ok(Self {
            self_attn,
            conv,
            feed_forward: FeedForward::new(quant, &format!("{path}.feed_forward"))?,
            operator_norm: rms_norm(args.hidden_size, args.norm_eps)?,
            post_operator_norm: args
                .use_sandwich_norm
                .then(|| rms_norm(args.hidden_size, args.norm_eps))
                .transpose()?,
            ffn_norm: rms_norm(args.hidden_size, args.norm_eps)?,
            post_ffn_norm: args
                .use_sandwich_norm
                .then(|| rms_norm(args.hidden_size, args.norm_eps))
                .transpose()?,
            layer_type,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        global_mask: Option<&AttentionMask>,
        sliding_mask: Option<&AttentionMask>,
        cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        let normed = self.operator_norm.forward(x)?;
        let mismatch = || Exception::custom("LFM3 layer/cache type mismatch");
        let operator = match cache {
            LayerCache::KV(kv) => {
                let mask = if self.layer_type == LayerType::SlidingAttention {
                    sliding_mask
                } else {
                    global_mask
                };
                self.self_attn
                    .as_mut()
                    .ok_or_else(mismatch)?
                    .forward(&normed, mask, kv)?
            }
            LayerCache::Arrays(arrays) => self
                .conv
                .as_mut()
                .ok_or_else(mismatch)?
                .forward(&normed, arrays)?,
        };
        let normalized_operator = self
            .post_operator_norm
            .as_mut()
            .map_or(Ok(operator.clone()), |norm| norm.forward(&operator))?;
        let hidden = x.add(normalized_operator)?;
        let mut ffn = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&hidden)?)?;
        if let Some(norm) = &mut self.post_ffn_norm {
            ffn = norm.forward(&ffn)?;
        }
        hidden.add(ffn)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct Lfm3Inner {
    #[param]
    embed_tokens: QEmbedding,
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    embedding_norm: nn::RmsNorm,
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Lfm3CausalLM {
    pub args: Lfm3Args,
    #[param]
    model: Lfm3Inner,
}

impl Lfm3CausalLM {
    pub fn new(args: Lfm3Args, quant: Option<&QuantizationSettings>) -> Result<Self, Exception> {
        let layer_count = usize::try_from(args.num_hidden_layers).unwrap_or(0);
        let schedule_valid = layer_count > 0
            && args.layer_types.len() == layer_count
            && args.layer_types.iter().all(|kind| {
                matches!(
                    kind.as_str(),
                    "conv" | "sliding_attention" | "full_attention"
                )
            });
        if !schedule_valid || args.projection_widths().is_none() {
            return Err(Exception::custom(
                "invalid LFM3 dimensions or layer schedule",
            ));
        }
        let head_dim = args.head_dim();
        if head_dim <= 0
            || args.num_attention_heads % args.num_key_value_heads != 0
            || args.conv_l_cache <= 1
            || args.sliding_window <= 0
        {
            return Err(Exception::custom(
                "invalid LFM3 dimensions or layer schedule",
            ));
        }
        if !args.tie_word_embeddings {
            return Err(Exception::custom(
                "native LFM3 currently supports tied embeddings only",
            ));
        }
        if args.conv_bias {
            return Err(Exception::custom(
                "native LFM3 currently supports bias-free convolution checkpoints only",
            ));
        }
        let dense = QuantizationSettings::new(0, 0);
        let q = quant.unwrap_or(&dense);
        let embed = resolve_quantization(q, "model.embed_tokens");
        let model = Lfm3Inner {
            embed_tokens: QEmbedding::new(embed.group_size(), embed.bits())?,
            layers: (0..layer_count)
                .map(|index| DecoderLayer::new(&args, q, index))
                .collect::<Result<_, _>>()?,
            embedding_norm: rms_norm(args.hidden_size, args.norm_eps)?,
        };
        Ok(Self { args, model })
    }

    pub fn make_cache(&self) -> Vec<Option<LayerCache>> {
        self.model
            .layers
            .iter()
            .map(|layer| {
                Some(match layer.layer_type {
                    LayerType::Conv => LayerCache::Arrays(ArraysCache::new()),
                    LayerType::SlidingAttention | LayerType::FullAttention => {
                        LayerCache::KV(SteppingKeyValueCache::new())
                    }
                })
            })
            .collect()
    }

    pub fn forward_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        if mask.is_some() {
            return Err(Exception::custom(
                "explicit attention masks are not supported by native LFM3",
            ));
        }
        if cache.is_empty() {
            *cache = self.make_cache();
        }
        if cache.len() != self.model.layers.len() {
            return Err(Exception::custom("cache length must match LFM3 layers"));
        }
        let mut hidden = self.model.embed_tokens.forward(inputs)?;
        let length = hidden.dim(1);
        let offset = cache
            .iter()
            .find_map(|slot| match slot.as_ref()? {
                LayerCache::KV(kv) => Some(kv.offset()),
                LayerCache::Arrays(_) => None,
            })
            .unwrap_or(0);
        let global = if length > 1 {
            Some(AttentionMask::Array(create_causal_mask(
                length,
                Some(offset),
            )?))
        } else {
            None
        };
        let sliding = if length > 1 || offset + length > self.args.sliding_window {
            Some(AttentionMask::Array(create_windowed_causal_mask(
                length,
                offset,
                self.args.sliding_window,
            )?))
        } else {
            None
        };
        for (layer, slot) in self.model.layers.iter_mut().zip(cache.iter_mut()) {
            hidden = layer.forward(
                &hidden,
                global.as_ref(),
                sliding.as_ref(),
                slot.as_mut()
                    .ok_or_else(|| Exception::custom("missing LFM3 layer cache"))?,
            )?;
        }
        self.model.embedding_norm.forward(&hidden)
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let hidden = self.forward_hidden(inputs, mask, cache)?;
        self.model
            .embed_tokens
            .as_linear(&hidden.index((.., -1.., ..)))
    }

    pub fn forward_all_logits(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let hidden = self.forward_hidden(inputs, mask, cache)?;
        self.model.embed_tokens.as_linear(&hidden)
    }
}

fn resolved_lfm3_config(config: &serde_json::Value) -> &serde_json::Value {
    config.get("text_config").unwrap_or(config)
}

fn quantization_config(config: &serde_json::Value) -> Option<&serde_json::Value> {
    resolved_lfm3_config(config).get("quantization")
}

fn validate_affine_quantization(config: &serde_json::Value) -> Result<(), ModelError> {
    let quantization = quantization_config(config);
    let Some(serde_json::Value::Object(entries)) = quantization else {
        return Ok(());
    };
    for (path, value) in entries {
        let configured_mode = if path == "mode" {
            Some(value)
        } else {
            value.as_object().and_then(|object| object.get("mode"))
        };
        if let Some(mode_value) = configured_mode {
            let Some(mode_name) = mode_value.as_str() else {
                return Err(ModelError::UnsupportedModel(format!(
                    "LFM3 quantization mode for {path} must be a string"
                )));
            };
            if mode_name != "affine" {
                return Err(ModelError::UnsupportedModel(format!(
                    "LFM3 quantization mode {mode_name} is unsupported; expected affine"
                )));
            }
        }
    }
    Ok(())
}

fn push_quantized_specs(
    specs: &mut Vec<(String, Vec<i32>)>,
    quant: &QuantizationSettings,
    path: &str,
    rows: i32,
    input: i32,
) -> Result<(), ModelError> {
    if rows <= 0 || input <= 0 {
        return Err(ModelError::ShapeMismatch(format!(
            "{path}: projection dimensions must be positive, got rows={rows}, input={input}"
        )));
    }
    match resolve_quantization(quant, path) {
        TensorQuant::Unquantized => specs.push((format!("{path}.weight"), vec![rows, input])),
        TensorQuant::Quantized { group_size, bits } => {
            let Some(packed_bits) = input.checked_mul(bits) else {
                return Err(ModelError::ShapeMismatch(format!(
                    "{path}: affine packed width overflows for input={input}, bits={bits}"
                )));
            };
            if group_size <= 0 || bits <= 0 || input % group_size != 0 || packed_bits % 32 != 0 {
                return Err(ModelError::ShapeMismatch(format!(
                    "{path}: invalid affine geometry input={input}, group_size={group_size}, bits={bits}"
                )));
            }
            let packed = packed_bits / 32;
            let groups = input / group_size;
            specs.push((format!("{path}.weight"), vec![rows, packed]));
            specs.push((format!("{path}.scales"), vec![rows, groups]));
            specs.push((format!("{path}.biases"), vec![rows, groups]));
        }
    }
    Ok(())
}

fn expected_checkpoint_shapes(
    args: &Lfm3Args,
    quant: &QuantizationSettings,
) -> Result<Vec<(String, Vec<i32>)>, ModelError> {
    let mut specs = Vec::new();
    let hidden = args.hidden_size;
    let head_dim = args.head_dim();
    let Some((query_width, key_value_width, conv_input_width)) = args.projection_widths() else {
        return Err(ModelError::ShapeMismatch(
            "invalid or overflowing LFM3 projection dimensions".to_owned(),
        ));
    };
    push_quantized_specs(
        &mut specs,
        quant,
        "model.embed_tokens",
        args.vocab_size,
        hidden,
    )?;
    specs.push(("model.embedding_norm.weight".to_owned(), vec![hidden]));
    for (layer, layer_type) in args.layer_types.iter().enumerate() {
        let path = format!("model.layers.{layer}");
        for norm in ["operator_norm", "ffn_norm"] {
            specs.push((format!("{path}.{norm}.weight"), vec![hidden]));
        }
        if args.use_sandwich_norm {
            for norm in ["post_operator_norm", "post_ffn_norm"] {
                specs.push((format!("{path}.{norm}.weight"), vec![hidden]));
            }
        }
        if layer_type == "conv" {
            specs.push((
                format!("{path}.conv.conv.weight"),
                vec![hidden, args.conv_l_cache, 1],
            ));
            push_quantized_specs(
                &mut specs,
                quant,
                &format!("{path}.conv.in_proj"),
                conv_input_width,
                hidden,
            )?;
            push_quantized_specs(
                &mut specs,
                quant,
                &format!("{path}.conv.out_proj"),
                hidden,
                hidden,
            )?;
        } else {
            let attention = format!("{path}.self_attn");
            for norm in ["q_norm", "k_norm"] {
                specs.push((format!("{attention}.{norm}.weight"), vec![head_dim]));
            }
            for (name, rows) in [
                ("q_proj", query_width),
                ("k_proj", key_value_width),
                ("v_proj", key_value_width),
                ("o_proj", hidden),
                ("gate_proj", args.num_attention_heads),
            ] {
                push_quantized_specs(
                    &mut specs,
                    quant,
                    &format!("{attention}.{name}"),
                    rows,
                    hidden,
                )?;
            }
        }
        for (name, rows, input) in [
            ("w1", args.intermediate_size, hidden),
            ("w2", hidden, args.intermediate_size),
            ("w3", args.intermediate_size, hidden),
        ] {
            push_quantized_specs(
                &mut specs,
                quant,
                &format!("{path}.feed_forward.{name}"),
                rows,
                input,
            )?;
        }
    }
    Ok(specs)
}

fn validate_checkpoint(
    dir: &Path,
    args: &Lfm3Args,
    quant: &QuantizationSettings,
) -> Result<(), ModelError> {
    let mut tensors = HashMap::new();
    for file in crate::collect_safetensors_files(dir)? {
        let loaded = Array::load_safetensors(&file)
            .map_err(|error| ModelError::Io(std::io::Error::other(error.to_string())))?;
        for (key, value) in loaded {
            let normalized_key = key
                .strip_prefix("language_model.")
                .unwrap_or(&key)
                .to_owned();
            if tensors.insert(normalized_key.clone(), value).is_some() {
                return Err(ModelError::ShapeMismatch(format!(
                    "duplicate LFM3 checkpoint tensor after prefix normalization: {normalized_key}"
                )));
            }
        }
    }
    for (key, expected) in expected_checkpoint_shapes(args, quant)? {
        let tensor = tensors
            .get(&key)
            .ok_or_else(|| ModelError::MissingWeight(key.clone()))?;
        if tensor.shape() != expected.as_slice() {
            return Err(ModelError::ShapeMismatch(format!(
                "{key}: expected shape {expected:?}, got {:?}",
                tensor.shape()
            )));
        }
    }
    Ok(())
}

pub(crate) fn load_lfm3_model_from_value(
    dir: &Path,
    config: &serde_json::Value,
) -> Result<Lfm3CausalLM, ModelError> {
    validate_affine_quantization(config)?;
    let args: Lfm3Args = serde_json::from_value(resolved_lfm3_config(config).clone())?;
    let quant = quantization_config(config)
        .cloned()
        .ok_or_else(|| ModelError::UnsupportedModel("LFM3 requires affine quantization".to_owned()))
        .and_then(|value| serde_json::from_value(value).map_err(ModelError::from))?;
    validate_checkpoint(dir, &args, &quant)?;
    tracing::info!(
        layers = args.num_hidden_layers,
        hidden_size = args.hidden_size,
        "Loading LFM3 model"
    );
    let mut model = Lfm3CausalLM::new(args, Some(&quant))?;
    crate::load_quantized_safetensors_weights_optional_prefix_with_settings(
        &mut model,
        dir,
        false,
        "language_model.",
        Some(&quant),
    )?;
    Ok(model)
}

pub fn load_lfm3_model<P: AsRef<Path>>(model_dir: P) -> Result<Lfm3CausalLM, ModelError> {
    let dir = model_dir.as_ref();
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(dir.join("config.json"))?)?;
    load_lfm3_model_from_value(dir, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hugging Face computes frequencies in f32, rounds cos/sin to the
    /// activation dtype, then promotes those rounded values back to f32 for
    /// rotation. This fixture changes if that intermediate BF16 round-trip is
    /// removed, even though the final result is also BF16.
    #[test]
    fn zero_tail_rope_rounds_trig_values_through_activation_dtype() {
        let x = Array::from_slice(
            &[0.707_031_25_f32, 1.468_75, -0.079_589_844, 1.085_937_5],
            &[1, 1, 1, 4],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        let actual = apply_zero_tail_rope(&x, 37, 10_000.0, 1.0)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();

        assert_eq!(
            actual.as_slice::<f32>(),
            &[0.490_234_38, 0.980_468_75, -0.515_625, 1.546_875]
        );
    }
}
