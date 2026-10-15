//! `LiquidAI` LFM2-MoE (`lfm2_moe`), ported from mlx-lm `models/lfm2_moe.py`.
//!
//! A hybrid of gated short-convolution and GQA attention layers, with
//! sigmoid-routed sparse experts after the first `num_dense_layers`.
//!
//! Every `QLinear`/`QEmbedding` is built from its own checkpoint path through
//! [`QuantizationSettings::resolve`], so per-tensor overrides (e.g. the 6-bit
//! `model.embed_tokens` in the 4-bit release) load without special cases.

use std::{cell::RefCell, path::Path};

use mlx_rs::{
    Array, Dtype,
    builder::Builder,
    error::Exception,
    fast,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
    transforms::compile::compile,
};
use serde::Deserialize;

use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    error::ModelError,
    quant_config::QuantizationSettings,
    qwen3_next::{ArraysCache, LayerCache, QEmbedding, QLinear, SwitchMlpWeights, swiglu},
    utils::{AttentionMask, apply_rope, create_causal_mask},
};

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Lfm2MoeArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub num_experts_per_tok: i32,
    pub num_dense_layers: i32,
    pub norm_eps: f32,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: i32,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub use_expert_bias: bool,
    #[serde(default = "one")]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

const fn one() -> f32 {
    1.0
}

impl Lfm2MoeArgs {
    pub const fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    fn is_attention(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .is_some_and(|t| t == "full_attention")
    }
}

/// Build a linear whose storage format is whatever the checkpoint declares for `path`.
fn linear(q: &QuantizationSettings, path: &str) -> Result<QLinear, Exception> {
    let t = q.resolve(path);
    QLinear::new(t.group_size(), t.bits())
}

fn rms_norm(dim: i32, eps: f32) -> Result<nn::RmsNorm, Exception> {
    nn::RmsNormBuilder::new(dim).eps(eps).build()
}

fn empty() -> Param<Array> {
    Param::new(Array::from_slice::<f32>(&[], &[0]))
}

// Decode is bound by MLX's per-op CPU encode cost (~1.4k ops/token keeps the
// GPU ~65% busy), so the per-token elementwise chains run as compiled graphs.
// Only at L=1: prefill shapes vary per request and are compute-bound anyway.
type CompiledFn = dyn for<'a> FnMut(&'a [Array]) -> Result<Vec<Array>, Exception>;
type RouterKey = (i32, bool, Dtype);

thread_local! {
    static CONV_STEP: RefCell<Option<Box<CompiledFn>>> = const { RefCell::new(None) };
    static ROUTERS: RefCell<Vec<(RouterKey, Box<CompiledFn>)>> = const { RefCell::new(Vec::new()) };
}

fn two(outputs: Vec<Array>) -> Result<(Array, Array), Exception> {
    let mut it = outputs.into_iter();
    match (it.next(), it.next()) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Err(Exception::custom("compiled graph must return two arrays")),
    }
}

// ---------------------------------------------------------------------------
// Gated short convolution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct ConvWeight {
    /// MLX depthwise `Conv1d` layout `[D, K, 1]`.
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
    /// `[K, D]` f32 taps, derived once from `conv.weight`.
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

    /// `out_proj(C * conv(B * x))`, causal over the cached last `K-1` rows of `B*x`.
    ///
    /// The state is kept chronologically (`conv_pos = K-2` keeps it
    /// ring-compatible with `ArraysCache` consumers). Taps accumulate in f32
    /// like mlx `Conv1d`: per-step bf16 rounding doubled the logit error
    /// against an f32 oracle (`scripts/lfm2_parity.py`).
    // ponytail: K shifted multiply-adds instead of nn::Conv1d; the Metal conv
    // kernel only handles K=4 (qwen3_next.rs) and K=3 here is negligible work.
    fn forward(&mut self, x: &Array, cache: &mut ArraysCache) -> Result<Array, Exception> {
        let parts = self.in_proj.forward(x)?.split(3, Some(-1))?;
        let [b, c, xs] = parts.as_slice() else {
            return Err(Exception::custom("in_proj must split into B, C, x"));
        };
        let (k, len) = (self.kernel, x.dim(1));
        let taps = self.taps()?;
        let prior = cache.conv_state.take();
        cache.conv_pos = k - 2;
        cache.offset += len;

        if let (1, Some(state)) = (len, &prior) {
            let args = [b.clone(), xs.clone(), c.clone(), state.clone(), taps];
            let (y, next_state) = two(CONV_STEP.with(|cell| {
                let mut slot = cell.borrow_mut();
                let step = slot.get_or_insert_with(|| Box::new(compile(conv_step, None)));
                step(&args)
            })?)?;
            cache.conv_state = Some(next_state);
            return self.out_proj.forward(&y);
        }

        let bx = b.multiply(xs)?;
        let state = match prior {
            Some(state) => state,
            None => ops::zeros_dtype(&[x.dim(0), k - 1, x.dim(2)], bx.dtype())?,
        };
        let xp = ops::concatenate_axis(&[state, bx], 1)?;
        cache.conv_state = Some(xp.index((.., -(k - 1).., ..)));
        let xp32 = xp.as_dtype(Dtype::Float32)?;
        let mut y = xp32.index((.., 0..len, ..)).multiply(taps.index(0))?;
        for i in 1..k {
            y = y.add(xp32.index((.., i..i + len, ..)).multiply(taps.index(i))?)?;
        }
        self.out_proj.forward(&c.multiply(y.as_dtype(c.dtype())?)?)
    }
}

/// One decode token of [`ShortConv::forward`] (same f32 tap order), plus the
/// shifted state: `[b, x, c, state, taps]` -> `[C * conv, next_state]`.
fn conv_step(args: &[Array]) -> Result<Vec<Array>, Exception> {
    let [b, xs, c, state, taps] = args else {
        return Err(Exception::custom("conv_step takes b, x, c, state, taps"));
    };
    let bx = b.multiply(xs)?;
    let k = taps.dim(0);
    let tap = |row: &Array, i: i32| row.as_dtype(Dtype::Float32)?.multiply(taps.index(i));
    let mut y = tap(&state.index((.., 0..1, ..)), 0)?;
    for i in 1..k - 1 {
        y = y.add(tap(&state.index((.., i..i + 1, ..)), i)?)?;
    }
    y = y.add(tap(&bx, k - 1)?)?;
    let next_state = ops::concatenate_axis(&[state.index((.., 1.., ..)), bx], 1)?;
    Ok(vec![c.multiply(y.as_dtype(c.dtype())?)?, next_state])
}

// ---------------------------------------------------------------------------
// Attention (GQA with per-head QK RMSNorm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Attention {
    #[param]
    q_proj: QLinear,
    #[param]
    k_proj: QLinear,
    #[param]
    v_proj: QLinear,
    #[param]
    out_proj: QLinear,
    #[param]
    q_layernorm: nn::RmsNorm,
    #[param]
    k_layernorm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
    n_heads: i32,
    n_kv_heads: i32,
    scale: f32,
}

impl Attention {
    fn new(args: &Lfm2MoeArgs, q: &QuantizationSettings, p: &str) -> Result<Self, Exception> {
        let head_dim = args.head_dim();
        let theta = args
            .rope_parameters
            .as_ref()
            .map_or(1_000_000.0, |r| r.rope_theta);
        Ok(Self {
            q_proj: linear(q, &format!("{p}.q_proj"))?,
            k_proj: linear(q, &format!("{p}.k_proj"))?,
            v_proj: linear(q, &format!("{p}.v_proj"))?,
            out_proj: linear(q, &format!("{p}.out_proj"))?,
            q_layernorm: rms_norm(head_dim, args.norm_eps)?,
            k_layernorm: rms_norm(head_dim, args.norm_eps)?,
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(false)
                .base(theta)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            scale: f32::from(
                u16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of range"))?,
            )
            .sqrt()
            .recip(),
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut SteppingKeyValueCache,
    ) -> Result<Array, Exception> {
        let (b, l) = (x.dim(0), x.dim(1));
        let heads = |a: Array, n: i32| a.reshape(&[b, l, n, -1]);
        let to_bhld = |a: Array| a.transpose_axes(&[0, 2, 1, 3]);
        let q = self
            .q_layernorm
            .forward(&heads(self.q_proj.forward(x)?, self.n_heads)?)?;
        let k = self
            .k_layernorm
            .forward(&heads(self.k_proj.forward(x)?, self.n_kv_heads)?)?;
        let v = heads(self.v_proj.forward(x)?, self.n_kv_heads)?;
        let queries = apply_rope(&to_bhld(q)?, &self.rope, cache.offset())?;
        let new_keys = apply_rope(&to_bhld(k)?, &self.rope, cache.offset())?;
        let (keys, values) = cache.update_and_fetch(new_keys, to_bhld(v)?)?;
        let out = fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            self.scale,
            mask.map(fast::ScaledDotProductAttentionMask::from),
            None::<&Array>,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, l, -1])?;
        self.out_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// Feed-forward: dense SwiGLU or sigmoid-routed experts (flat fields, since
// both live directly under `feed_forward.` in the checkpoint)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct FeedForward {
    #[param]
    gate_proj: Option<QLinear>,
    #[param]
    up_proj: Option<QLinear>,
    #[param]
    down_proj: Option<QLinear>,
    #[param]
    gate: Option<QLinear>,
    #[param]
    switch_mlp: Option<SwitchMlpWeights>,
    /// Routing-only bias (`[E]`, f32); empty on dense layers and when unused.
    #[param]
    expert_bias: Param<Array>,
    top_k: i32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
}

impl FeedForward {
    fn dense(q: &QuantizationSettings, p: &str) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: Some(linear(q, &format!("{p}.gate_proj"))?),
            up_proj: Some(linear(q, &format!("{p}.up_proj"))?),
            down_proj: Some(linear(q, &format!("{p}.down_proj"))?),
            gate: None,
            switch_mlp: None,
            expert_bias: empty(),
            top_k: 0,
            norm_topk_prob: false,
            routed_scaling_factor: 1.0,
        })
    }

    fn sparse(args: &Lfm2MoeArgs, q: &QuantizationSettings, p: &str) -> Result<Self, Exception> {
        let experts = ["gate_proj", "up_proj", "down_proj"].map(|n| format!("{p}.switch_mlp.{n}"));
        let t = q
            .resolve_uniform(experts.iter().map(String::as_str))
            .map_err(Exception::custom)?;
        let router = format!("{p}.gate");
        // ponytail: an unlisted router is stored dense (LiquidAI's 4-bit release
        // ships it as f32); a listed one follows its override.
        let gate = if q.is_overridden(&router) {
            linear(q, &router)?
        } else {
            QLinear::new(0, 0)?
        };
        let expert_bias = if args.use_expert_bias {
            Param::new(Array::from_slice::<f32>(&[0.0], &[1]))
        } else {
            empty()
        };
        Ok(Self {
            gate_proj: None,
            up_proj: None,
            down_proj: None,
            gate: Some(gate),
            switch_mlp: Some(SwitchMlpWeights::from_quant(t.group_size(), t.bits())?),
            expert_bias,
            top_k: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
            routed_scaling_factor: args.routed_scaling_factor,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        if let Some(gate) = &self.gate {
            let (inds, weights) = self.route(x, gate)?;
            let experts = self
                .switch_mlp
                .as_mut()
                .ok_or_else(|| Exception::custom("MoE layer without experts"))?;
            // Decode (one token): gather directly. The global-sort path pads the
            // routed rows to `4 × num_experts` for prefill/decode rounding parity,
            // which at L=1 multiplies expert work ~64× for LFM2's 64 experts.
            let routed = if x.dim(1) == 1 {
                experts.forward_gather_fused_decode(x, &inds)?
            } else {
                experts.forward_gather_fused(x, &inds)?
            };
            return routed
                .multiply(weights.expand_dims(-1)?)?
                .sum_axes(&[-2], false);
        }
        let (Some(g), Some(u), Some(d)) = (&self.gate_proj, &self.up_proj, &self.down_proj) else {
            return Err(Exception::custom(
                "feed_forward has neither dense nor MoE weights",
            ));
        };
        d.forward(&swiglu(&g.forward(x)?, &u.forward(x)?)?)
    }

    /// Experts are *selected* on `sigmoid(gate) + expert_bias` but *weighted* by
    /// the unbiased sigmoid scores (HF `Lfm2Moe` reference). Returns `(inds, weights)`.
    fn route(&self, x: &Array, gate: &QLinear) -> Result<(Array, Array), Exception> {
        let key = (self.top_k, self.norm_topk_prob, x.dtype());
        let args = [
            gate.forward(x)?.as_dtype(Dtype::Float32)?,
            (*self.expert_bias).clone(),
            Array::from_f32(self.routed_scaling_factor),
        ];
        two(if x.dim(1) == 1 {
            ROUTERS.with(|cell| {
                let mut routers = cell.borrow_mut();
                let slot = routers
                    .iter()
                    .position(|(k, _)| *k == key)
                    .unwrap_or_else(|| {
                        routers.push((key, Box::new(compile(router(key), None))));
                        routers.len() - 1
                    });
                let (_, route) = routers
                    .get_mut(slot)
                    .ok_or_else(|| Exception::custom("router cache slot"))?;
                route(&args)
            })?
        } else {
            router(key)(&args)?
        })
    }
}

/// Sigmoid top-k routing: `[logits_f32, expert_bias, scale]` -> `[inds, weights]`.
/// An empty `expert_bias` means no selection bias.
fn router(
    (top_k, norm_topk_prob, dtype): RouterKey,
) -> impl for<'a> FnMut(&'a [Array]) -> Result<Vec<Array>, Exception> + Copy + 'static {
    move |args: &[Array]| {
        let [logits, bias, scale] = args else {
            return Err(Exception::custom("router takes logits, bias, scale"));
        };
        let probs = nn::sigmoid(logits)?;
        let selection = if bias.size() > 0 {
            probs.add(bias)?
        } else {
            probs.clone()
        };
        let inds = ops::argpartition_axis(&selection, -top_k, -1)?.index((.., .., -top_k..));
        let mut scores = probs.take_along_axis(&inds, -1)?;
        if norm_topk_prob {
            let denom = scores.sum_axes(&[-1], true)?.add(Array::from_f32(1e-6))?;
            scores = scores.divide(denom)?;
        }
        Ok(vec![inds, scores.multiply(scale)?.as_dtype(dtype)?])
    }
}

// ---------------------------------------------------------------------------
// Decoder layer / model
// ---------------------------------------------------------------------------

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
    ffn_norm: nn::RmsNorm,
}

impl DecoderLayer {
    fn new(args: &Lfm2MoeArgs, q: &QuantizationSettings, i: usize) -> Result<Self, Exception> {
        let p = format!("model.layers.{i}");
        let (self_attn, conv) = if args.is_attention(i) {
            (
                Some(Attention::new(args, q, &format!("{p}.self_attn"))?),
                None,
            )
        } else {
            let conv = ShortConv {
                conv: ConvWeight { weight: empty() },
                in_proj: linear(q, &format!("{p}.conv.in_proj"))?,
                out_proj: linear(q, &format!("{p}.conv.out_proj"))?,
                kernel: args.conv_l_cache,
                taps: None,
            };
            (None, Some(conv))
        };
        let ff = format!("{p}.feed_forward");
        let dense = i32::try_from(i).is_ok_and(|layer| layer < args.num_dense_layers);
        Ok(Self {
            self_attn,
            conv,
            feed_forward: if dense {
                FeedForward::dense(q, &ff)?
            } else {
                FeedForward::sparse(args, q, &ff)?
            },
            operator_norm: rms_norm(args.hidden_size, args.norm_eps)?,
            ffn_norm: rms_norm(args.hidden_size, args.norm_eps)?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        let normed = self.operator_norm.forward(x)?;
        let mismatch = || Exception::custom("lfm2_moe layer/cache type mismatch");
        let r = match cache {
            LayerCache::KV(c) => self
                .self_attn
                .as_mut()
                .ok_or_else(mismatch)?
                .forward(&normed, mask, c)?,
            LayerCache::Arrays(c) => self
                .conv
                .as_mut()
                .ok_or_else(mismatch)?
                .forward(&normed, c)?,
        };
        let h = x.add(r)?;
        h.add(self.feed_forward.forward(&self.ffn_norm.forward(&h)?)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct Lfm2Inner {
    #[param]
    embed_tokens: QEmbedding,
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    embedding_norm: nn::RmsNorm,
}

/// LFM2-MoE causal LM. The LM head is always tied to `embed_tokens`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Lfm2MoeCausalLM {
    pub args: Lfm2MoeArgs,
    #[param]
    model: Lfm2Inner,
}

impl Lfm2MoeCausalLM {
    pub fn new(args: Lfm2MoeArgs, quant: Option<&QuantizationSettings>) -> Result<Self, Exception> {
        let n = usize::try_from(args.num_hidden_layers).unwrap_or(0);
        if n == 0 || args.layer_types.len() != n || args.num_attention_heads <= 0 {
            return Err(Exception::custom(
                "lfm2_moe needs num_hidden_layers > 0 matching layer_types, and attention heads",
            ));
        }
        let dense = QuantizationSettings::new(0, 0);
        let q = quant.unwrap_or(&dense);
        let embed = q.resolve("model.embed_tokens");
        let model = Lfm2Inner {
            embed_tokens: QEmbedding::new(embed.group_size(), embed.bits())?,
            layers: (0..n)
                .map(|i| DecoderLayer::new(&args, q, i))
                .collect::<Result<_, _>>()?,
            embedding_norm: rms_norm(args.hidden_size, args.norm_eps)?,
        };
        Ok(Self { args, model })
    }

    /// One cache per layer: KV for attention, conv state for short-conv layers.
    pub fn make_cache(&self) -> Vec<Option<LayerCache>> {
        (0..self.model.layers.len())
            .map(|i| {
                Some(if self.args.is_attention(i) {
                    LayerCache::KV(SteppingKeyValueCache::new())
                } else {
                    LayerCache::Arrays(ArraysCache::new())
                })
            })
            .collect()
    }

    /// Hidden states after `embedding_norm` for every input position.
    pub fn forward_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        if cache.is_empty() {
            *cache = self.make_cache();
        }
        if cache.len() != self.model.layers.len() {
            return Err(Exception::custom("cache length must match num layers"));
        }
        let mut h = self.model.embed_tokens.forward(inputs)?;
        let attn_mask = match mask {
            Some(m) => Some(AttentionMask::Array(m.clone())),
            None => causal_mask(h.dim(1), cache)?,
        };
        for (layer, slot) in self.model.layers.iter_mut().zip(cache.iter_mut()) {
            let c = slot
                .as_mut()
                .ok_or_else(|| Exception::custom("missing layer cache"))?;
            h = layer.forward(&h, attn_mask.as_ref(), c)?;
        }
        self.model.embedding_norm.forward(&h)
    }

    /// Logits for the last input position.
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, cache)?;
        self.model.embed_tokens.as_linear(&h.index((.., -1.., ..)))
    }

    /// Logits for every input position.
    pub fn forward_all_logits(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, cache)?;
        self.model.embed_tokens.as_linear(&h)
    }
}

/// Causal mask for `t` new tokens, offset by what the attention caches already hold.
fn causal_mask(t: i32, cache: &[Option<LayerCache>]) -> Result<Option<AttentionMask>, Exception> {
    if t <= 1 {
        return Ok(None);
    }
    let offset = cache
        .iter()
        .find_map(|c| match c.as_ref()? {
            LayerCache::KV(kv) => Some(kv.offset()),
            LayerCache::Arrays(_) => None,
        })
        .unwrap_or(0);
    Ok(Some(if offset > 0 {
        AttentionMask::Array(create_causal_mask(t, Some(offset))?)
    } else {
        AttentionMask::Causal
    }))
}

pub fn load_lfm2_moe_model<P: AsRef<Path>>(model_dir: P) -> Result<Lfm2MoeCausalLM, ModelError> {
    let dir = model_dir.as_ref();
    let args: Lfm2MoeArgs = serde_json::from_reader(std::fs::File::open(dir.join("config.json"))?)?;
    let quant = crate::load_checkpoint_quantization_settings(dir)?;
    tracing::info!(
        layers = args.num_hidden_layers,
        hidden_size = args.hidden_size,
        experts_per_tok = args.num_experts_per_tok,
        "Loading lfm2_moe model"
    );
    let mut model = Lfm2MoeCausalLM::new(args, quant.as_ref())?;
    crate::load_quantized_safetensors_weights_with_settings(
        &mut model,
        dir,
        false,
        quant.as_ref(),
    )?;
    // One gate+up gather per MoE layer instead of two (fewer decode ops).
    for layer in &mut model.model.layers {
        if let Some(experts) = layer.feed_forward.switch_mlp.as_mut() {
            experts.fuse_gate_up_in_place()?;
        }
    }
    Ok(model)
}
