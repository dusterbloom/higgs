//! NVIDIA Nemotron-Labs-Diffusion — decoder-only masked-diffusion LLM.
//!
//! This implements **diffusion-mode** inference only: starting from a block of
//! `mask_token_id` tokens (a "canvas"), the model runs several denoising
//! forward passes, each predicting clean tokens and un-masking the
//! highest-confidence positions, until the block is fully resolved. Generation
//! proceeds block-by-block (`block_size`), appending each finalized block to the
//! context.
//!
//! In diffusion mode the model attends **bidirectionally** over the current
//! canvas (no causal mask) — `dlm_paradigm = "bidirectional"` in config.json.
//! The architecture is an otherwise-standard decoder (RMSNorm + GQA attention +
//! SiLU MLP) with **YaRN** RoPE; we reuse [`crate::yarn`] for the rope and keep
//! the rest local so the autoregressive engine path is untouched.
//!
//! Autoregressive decoding and self-speculation (the model's other two modes)
//! are intentionally out of scope here and handled in follow-up work.

#![allow(clippy::doc_markdown)] // YaRN, RoPE, RMSNorm, SiLU, GQA are domain terms.

use std::path::Path;

use mlx_rs::{
    Array, Dtype,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn, ops,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
};
use serde::Deserialize;

use crate::{
    error::ModelError,
    transformer::QuantizationConfig,
    utils::{create_causal_mask, scaled_dot_product_attention},
    yarn::{apply_yarn_rope, compute_yarn_freqs, yarn_get_mscale},
};

/// Callback invoked with each finalized block's token ids during diffusion
/// decode (used for streaming). `None` to discard block events.
pub type BlockCallback<'a> = Option<&'a mut dyn FnMut(&[u32])>;

const fn default_rope_theta() -> f32 {
    1_000_000.0
}
const fn default_yarn_factor() -> f32 {
    1.0
}
const fn default_orig_max_pos() -> i32 {
    16384
}
const fn default_beta_fast() -> f32 {
    32.0
}
const fn default_beta_slow() -> f32 {
    1.0
}
const fn default_mscale() -> f32 {
    1.0
}
const fn default_block_size() -> i32 {
    32
}
const fn default_diffusion_steps() -> i32 {
    32
}
const fn default_mask_token_id() -> u32 {
    // Nemotron-Labs-Diffusion uses mask_token_id = 100; kept as a default for
    // robustness, but the real value is read from config.json.
    100
}

/// Nested `rope_parameters` block (YaRN scaling), as shipped by Nemotron.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_yarn_factor")]
    factor: f32,
    #[serde(default = "default_orig_max_pos")]
    original_max_position_embeddings: i32,
    #[serde(default = "default_beta_fast")]
    beta_fast: f32,
    #[serde(default = "default_beta_slow")]
    beta_slow: f32,
    #[serde(default = "default_mscale")]
    mscale: f32,
    #[serde(default = "default_mscale")]
    mscale_all_dim: f32,
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_theta: default_rope_theta(),
            factor: default_yarn_factor(),
            original_max_position_embeddings: default_orig_max_pos(),
            beta_fast: default_beta_fast(),
            beta_slow: default_beta_slow(),
            mscale: default_mscale(),
            mscale_all_dim: default_mscale(),
        }
    }
}

/// Model configuration, deserialized from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct NemotronDiffusionArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_parameters: RopeParameters,
    /// Present for pre-quantized MLX checkpoints (e.g. 4-bit `group_size=64`).
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    /// End-of-sequence token; lets diffusion stop early instead of generating
    /// the full requested length past EOS.
    #[serde(default)]
    pub eos_token_id: Option<u32>,

    // Diffusion-specific.
    #[serde(default = "default_mask_token_id")]
    pub mask_token_id: u32,
    #[serde(default = "default_block_size")]
    pub block_size: i32,
    #[serde(default = "default_diffusion_steps")]
    pub diffusion_steps: i32,
}

impl NemotronDiffusionArgs {
    /// Head dimension: explicit `head_dim` from config, else
    /// `hidden_size / num_attention_heads`.
    fn checked_head_dim(&self) -> Result<i32, Exception> {
        if let Some(hd) = self.head_dim {
            if hd <= 0 {
                return Err(Exception::custom("head_dim must be positive"));
            }
            return Ok(hd);
        }
        if self.num_attention_heads <= 0 {
            return Err(Exception::custom("num_attention_heads must be positive"));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(Exception::custom(
                "hidden_size must be divisible by num_attention_heads",
            ));
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }
}

/// Precomputed YaRN rope constants (shared across layers; not a parameter).
#[derive(Debug, Clone)]
struct RopeState {
    freqs: Array,
    base: f32,
    mscale: f32,
}

/// Frozen per-layer prefix K/V for diffusion decoding. Each entry is
/// `(roped_keys, values)` shaped `[1, n_kv_heads, len, head_dim]`. Built by a
/// causal prefill of the prompt and extended (causally) after each block, so
/// the denoising steps only forward the current block.
struct DiffusionKvCache {
    per_layer: Vec<(Array, Array)>,
    len: i32,
}

/// Per-layer `(roped_keys, values)` returned by one encoder forward.
type LayerKv = Vec<(Array, Array)>;

/// LoRA adapter on `o_proj` for the linear-spec drafter. Stored transposed so
/// apply is two plain matmuls: `out += scale * (x @ a_t) @ b_t`. Loaded from the
/// bundled `linear_spec_lora/` adapter; `None` for plain diffusion decode.
#[derive(Debug, Clone)]
struct OProjLora {
    a_t: Array, // [in_features, r]
    b_t: Array, // [r, out_features]
    scale: f32, // lora_alpha / r
}

/// Stats from [`NemotronDiffusionLM::linear_spec_generate`] for throughput
/// analysis: total forward evals, blocks drafted, tokens accepted.
#[derive(Debug, Clone, Copy)]
pub struct LinSpecStats {
    pub nfe: usize,
    pub blocks: usize,
    pub accepted_total: usize,
}

/// Multi-head GQA attention with YaRN rope, bidirectional (no KV cache).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct NemotronAttention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,

    #[quantizable]
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,
    /// Optional linear-spec drafter LoRA on `o_proj`; added to the `o_proj`
    /// output only when the caller requests `lora` (the draft phase). Not a
    /// model parameter — loaded separately and untouched by quantization.
    o_lora: Option<OProjLora>,
}

impl NemotronAttention {
    fn new(args: &NemotronDiffusionArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.checked_head_dim()?;
        let head_dim_f = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f.sqrt().recip();

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, n_heads * head_dim)
                    .bias(false)
                    .build()?,
            ),
            k_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
                    .bias(false)
                    .build()?,
            ),
            v_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
                    .bias(false)
                    .build()?,
            ),
            o_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(n_heads * head_dim, dim)
                    .bias(false)
                    .build()?,
            ),
            o_lora: None,
        })
    }

    /// Attention over `x`, optionally prepending a frozen prefix K/V cache and
    /// applying `mask`. Queries/keys are roped at absolute position `offset`.
    /// When `lora` is set and an `o_lora` adapter is attached, its low-rank term
    /// is added to the `o_proj` output (linear-spec draft phase). Returns
    /// `(output, roped_keys, values)` so the caller can cache the K/V.
    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        x: &Array,
        rope: &RopeState,
        cache: Option<&(Array, Array)>,
        offset: i32,
        mask: Option<&Array>,
        lora: bool,
    ) -> Result<(Array, Array, Array), Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("input must be 3D"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("input must be 3D"))?;

        let queries = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = self
            .k_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = self
            .v_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let offset_arr = Array::from_int(offset);
        let queries_roped = apply_yarn_rope(
            &queries,
            self.head_dim,
            rope.base,
            Some(&rope.freqs),
            rope.mscale,
            &offset_arr,
            false,
        )?;
        let keys_roped = apply_yarn_rope(
            &keys,
            self.head_dim,
            rope.base,
            Some(&rope.freqs),
            rope.mscale,
            &offset_arr,
            false,
        )?;

        // Prepend the frozen prefix K/V when decoding a block; the new (block)
        // K/V are still returned so a committed block can extend the cache.
        let (attn_keys, attn_values) = match cache {
            Some((ck, cv)) => (
                ops::concatenate_axis(&[ck, &keys_roped], -2)?,
                ops::concatenate_axis(&[cv, &values], -2)?,
            ),
            None => (keys_roped.clone(), values.clone()),
        };

        let output =
            scaled_dot_product_attention(queries_roped, attn_keys, attn_values, self.scale, mask)?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, L, -1])?;
        let mut o = self.o_proj.forward(&output)?;
        if lora {
            if let Some(adapter) = self.o_lora.as_ref() {
                let a_t = adapter.a_t.as_dtype(output.dtype())?;
                let b_t = adapter.b_t.as_dtype(output.dtype())?;
                let scale = Array::from_f32(adapter.scale).as_dtype(output.dtype())?;
                let delta = output.matmul(&a_t)?.matmul(&b_t)?.multiply(&scale)?;
                o = o.add(&delta)?;
            }
        }
        Ok((o, keys_roped, values))
    }
}

/// SiLU-gated MLP.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct NemotronMlp {
    #[quantizable]
    #[param]
    gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    down_proj: MaybeQuantized<nn::Linear>,
}

impl NemotronMlp {
    fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, hidden_dim)
                    .bias(false)
                    .build()?,
            ),
            up_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, hidden_dim)
                    .bias(false)
                    .build()?,
            ),
            down_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(hidden_dim, dim)
                    .bias(false)
                    .build()?,
            ),
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gated = nn::silu(self.gate_proj.forward(x)?)?.multiply(self.up_proj.forward(x)?)?;
        self.down_proj.forward(&gated)
    }
}

/// One decoder layer: pre-norm attention + pre-norm MLP, residual.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct NemotronLayer {
    #[quantizable]
    #[param]
    self_attn: NemotronAttention,
    #[quantizable]
    #[param]
    mlp: NemotronMlp,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl NemotronLayer {
    fn new(args: &NemotronDiffusionArgs) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: NemotronAttention::new(args)?,
            mlp: NemotronMlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        rope: &RopeState,
        cache: Option<&(Array, Array)>,
        offset: i32,
        mask: Option<&Array>,
        lora: bool,
    ) -> Result<(Array, (Array, Array)), Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let (attn_out, k, v) = self
            .self_attn
            .forward(&normed, rope, cache, offset, mask, lora)?;
        let h = x.add(attn_out)?;
        let normed_post = self.post_attention_layernorm.forward(&h)?;
        Ok((h.add(self.mlp.forward(&normed_post)?)?, (k, v)))
    }
}

/// Embedding + layers + final norm (no LM head).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct NemotronModel {
    #[quantizable]
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    layers: Vec<NemotronLayer>,
    #[param]
    norm: nn::RmsNorm,
}

impl NemotronModel {
    fn new(args: &NemotronDiffusionArgs) -> Result<Self, Exception> {
        if args.vocab_size <= 0 || args.num_hidden_layers <= 0 || args.num_key_value_heads <= 0 {
            return Err(Exception::custom(
                "vocab_size, num_hidden_layers, num_key_value_heads must be positive",
            ));
        }
        Ok(Self {
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            layers: (0..args.num_hidden_layers)
                .map(|_| NemotronLayer::new(args))
                .collect::<Result<Vec<_>, _>>()?,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }

    /// Forward `inputs`, threading an optional per-layer prefix cache, position
    /// `offset` and `mask`. Returns the final hidden states and each layer's
    /// freshly-computed `(roped_keys, values)`.
    fn forward(
        &mut self,
        inputs: &Array,
        rope: &RopeState,
        caches: Option<&[(Array, Array)]>,
        offset: i32,
        mask: Option<&Array>,
        lora: bool,
    ) -> Result<(Array, Vec<(Array, Array)>), Exception> {
        let mut h = self.embed_tokens.forward(inputs)?;
        let mut new_kv = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = caches.and_then(|c| c.get(i));
            let (hh, kv) = layer.forward(&h, rope, layer_cache, offset, mask, lora)?;
            h = hh;
            new_kv.push(kv);
        }
        Ok((self.norm.forward(&h)?, new_kv))
    }
}

/// Nemotron-Labs-Diffusion causal LM (diffusion-mode inference).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct NemotronDiffusionLM {
    pub args: NemotronDiffusionArgs,
    /// Transformer body. Named `encoder` to match the checkpoint's weight keys
    /// (`encoder.layers.*`, `encoder.embed_tokens.*`, `encoder.norm.*`).
    #[quantizable]
    #[param]
    encoder: NemotronModel,
    /// Diffusion output projection (`diffusion_head.*` in the checkpoint) — a
    /// separate quantized head, not a tied/`lm_head` layer.
    #[quantizable]
    #[param]
    diffusion_head: MaybeQuantized<nn::Linear>,
    rope: RopeState,
}

impl NemotronDiffusionLM {
    pub fn new(args: NemotronDiffusionArgs) -> Result<Self, Exception> {
        let head_dim = args.checked_head_dim()?;
        let rp = &args.rope_parameters;
        let freqs = compute_yarn_freqs(
            head_dim,
            rp.rope_theta,
            rp.factor,
            rp.original_max_position_embeddings,
            rp.beta_fast,
            rp.beta_slow,
        );
        // YaRN attention temperature: factor of get_mscale(factor, mscale) over
        // get_mscale(factor, mscale_all_dim). Nemotron ships mscale ==
        // mscale_all_dim, so this is 1.0 (no q/k scaling) — only the frequency
        // interpolation applies.
        let mscale_num = yarn_get_mscale(rp.factor, rp.mscale);
        let mscale_den = yarn_get_mscale(rp.factor, rp.mscale_all_dim);
        let mscale = if mscale_den.abs() > f32::EPSILON {
            mscale_num / mscale_den
        } else {
            1.0
        };
        let rope = RopeState {
            freqs,
            base: rp.rope_theta,
            mscale,
        };

        let encoder = NemotronModel::new(&args)?;
        // The checkpoint ships a dedicated (quantized) diffusion output head;
        // there is no tied/`lm_head` path for this model.
        let diffusion_head = MaybeQuantized::Original(
            nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                .bias(false)
                .build()?,
        );

        Ok(Self {
            args,
            encoder,
            diffusion_head,
            rope,
        })
    }

    pub const fn hidden_size(&self) -> i32 {
        self.args.hidden_size
    }

    /// `(num_key_value_heads, head_dim)` — used by the engine for cache sizing.
    pub fn kv_cache_geometry(&self) -> Result<(i32, i32), Exception> {
        Ok((self.args.num_key_value_heads, self.args.checked_head_dim()?))
    }

    /// Causal prefill of the prompt → per-layer KV cache + block-0 seed token.
    fn prefill(&mut self, prompt_ids: &[u32]) -> Result<(DiffusionKvCache, u32), Exception> {
        let p =
            i32::try_from(prompt_ids.len()).map_err(|_| Exception::custom("prompt too long"))?;
        let input = Array::from_slice(prompt_ids, &[1, p]);
        let mask = create_causal_mask(p, None)?;
        let (hidden, per_layer) =
            self.encoder
                .forward(&input, &self.rope, None, 0, Some(&mask), false)?;
        materialize(&per_layer)?;
        let seed = self.last_token(&hidden, p)?;
        Ok((DiffusionKvCache { per_layer, len: p }, seed))
    }

    /// Argmax token of the last position's logits in `hidden` (`[1, len, H]`).
    fn last_token(&mut self, hidden: &Array, len: i32) -> Result<u32, Exception> {
        let last = hidden.index((0, len - 1, ..)).reshape(&[1, -1])?;
        let logit = self.diffusion_head.forward(&last)?;
        let tok = ops::indexing::argmax_axis(&logit, -1, false)?.as_dtype(Dtype::Uint32)?;
        mlx_rs::transforms::eval([&tok])?;
        tok.as_slice::<u32>()
            .first()
            .copied()
            .ok_or_else(|| Exception::custom("empty argmax"))
    }

    /// Forward the current block against the frozen prefix cache (bidirectional,
    /// no mask). Returns per-position argmax token + softmax confidence. Used by
    /// plain diffusion decode; the linear-spec draft is in `draft_and_verify`.
    fn predict_block(
        &mut self,
        block: &[u32],
        cache: &DiffusionKvCache,
    ) -> Result<(Vec<u32>, Vec<f32>), Exception> {
        let blk = i32::try_from(block.len()).map_err(|_| Exception::custom("block too long"))?;
        let input = Array::from_slice(block, &[1, blk]);
        let (hidden, _) = self.encoder.forward(
            &input,
            &self.rope,
            Some(&cache.per_layer),
            cache.len,
            None,
            false,
        )?;
        let logits = self.diffusion_head.forward(&hidden)?.reshape(&[blk, -1])?;
        let probs = ops::softmax_axis(&logits, -1, true)?;
        let conf = probs.max_axis(-1, false)?.as_dtype(Dtype::Float32)?;
        let preds = ops::indexing::argmax_axis(&logits, -1, false)?.as_dtype(Dtype::Uint32)?;
        mlx_rs::transforms::eval([&conf, &preds])?;
        Ok((
            preds.as_slice::<u32>().to_vec(),
            conf.as_slice::<f32>().to_vec(),
        ))
    }

    /// Commit a finalized block causally: append its K/V to `cache` and return
    /// the next block's seed token.
    fn commit_block(
        &mut self,
        cache: &mut DiffusionKvCache,
        block: &[u32],
    ) -> Result<u32, Exception> {
        let b = i32::try_from(block.len()).map_err(|_| Exception::custom("block too long"))?;
        let input = Array::from_slice(block, &[1, b]);
        let mask = create_causal_mask(b, Some(cache.len))?;
        let (hidden, block_kv) = self.encoder.forward(
            &input,
            &self.rope,
            Some(&cache.per_layer),
            cache.len,
            Some(&mask),
            false,
        )?;
        extend_cache_by(cache, &block_kv, b)?;
        self.last_token(&hidden, b)
    }

    /// One linear-spec round: draft the seeded block (bidirectional, LoRA on,
    /// argmax only — no softmax), then verify it causally (LoRA off), evaluating
    /// the drafted block and the AR argmax in a single GPU→host barrier so the
    /// per-block sync cost is one eval (plus the later cache-extend materialize).
    /// The block stays on-device between draft and verify (no host round-trip).
    /// Returns `(drafted_tokens, ar_tokens, verify_block_kv)`.
    fn draft_and_verify(
        &mut self,
        seed: u32,
        block_len: usize,
        cache: &DiffusionKvCache,
    ) -> Result<(Vec<u32>, Vec<u32>, LayerKv), Exception> {
        let l = i32::try_from(block_len).map_err(|_| Exception::custom("block too long"))?;
        let mask_id = self.args.mask_token_id;

        // Seeded block [seed, mask, ...] on device.
        let mut seeded: Vec<u32> = std::iter::repeat_n(mask_id, block_len).collect();
        if let Some(s) = seeded.first_mut() {
            *s = seed;
        }
        let block0 = Array::from_slice(&seeded, &[1, l]);

        // Draft: bidirectional, LoRA on, argmax only (no softmax/confidence).
        let (dhidden, _) = self.encoder.forward(
            &block0,
            &self.rope,
            Some(&cache.per_layer),
            cache.len,
            None,
            true,
        )?;
        let draft_logits = self.diffusion_head.forward(&dhidden)?.reshape(&[l, -1])?;
        let draft_arg = ops::indexing::argmax_axis(&draft_logits, -1, false)?
            .as_dtype(Dtype::Uint32)?
            .reshape(&[1, l])?;
        // Keep position 0 = verified seed; positions 1.. take the draft. (The
        // draft's prediction at 0 is unused.)
        let block_dev = if l > 1 {
            ops::concatenate_axis(
                &[
                    &Array::from_slice(&[seed], &[1, 1]),
                    &draft_arg.index((.., 1..l)),
                ],
                -1,
            )?
        } else {
            Array::from_slice(&[seed], &[1, 1])
        };

        // Verify: causal, LoRA off.
        let vmask = create_causal_mask(l, Some(cache.len))?;
        let (vhidden, block_kv) = self.encoder.forward(
            &block_dev,
            &self.rope,
            Some(&cache.per_layer),
            cache.len,
            Some(&vmask),
            false,
        )?;
        let verify_logits = self.diffusion_head.forward(&vhidden)?.reshape(&[l, -1])?;
        let ar = ops::indexing::argmax_axis(&verify_logits, -1, false)?.as_dtype(Dtype::Uint32)?;

        // Single barrier: drafted block + AR argmax together (verify depends on
        // the draft, so one eval computes both forwards).
        let block_flat = block_dev.reshape(&[l])?;
        mlx_rs::transforms::eval([&block_flat, &ar])?;
        Ok((
            block_flat.as_slice::<u32>().to_vec(),
            ar.as_slice::<u32>().to_vec(),
            block_kv,
        ))
    }

    /// Diffusion decode (the reference's block-diffusion `generate`): causal
    /// prefill, then per block seed position 0, denoise the rest against the
    /// frozen prefix cache (bidirectional), and commit the block causally.
    /// Greedy; `on_block` receives each finalized block (for streaming).
    #[allow(clippy::shadow_reuse)]
    pub fn diffusion_generate(
        &mut self,
        prompt_ids: &[u32],
        num_tokens: usize,
        steps: usize,
        block_size: usize,
        mut on_block: BlockCallback<'_>,
    ) -> Result<Vec<u32>, Exception> {
        let mask_id = self.args.mask_token_id;
        let eos_id = self.args.eos_token_id;
        let steps = steps.max(1);
        let block_size = block_size.max(1);

        let (mut cache, mut seed) = self.prefill(prompt_ids)?;
        let mut generated: Vec<u32> = Vec::with_capacity(num_tokens);

        while generated.len() < num_tokens {
            let blk = block_size.min(num_tokens - generated.len());
            // Position 0 is the causal seed; the rest start masked.
            let mut block: Vec<u32> = std::iter::repeat_n(mask_id, blk).collect();
            if let Some(slot) = block.first_mut() {
                *slot = seed;
            }

            for step in 0..steps {
                let masked: Vec<usize> = (0..blk)
                    .filter(|&p| block.get(p).copied() == Some(mask_id))
                    .collect();
                if masked.is_empty() {
                    break;
                }
                let (preds, conf) = self.predict_block(&block, &cache)?;
                let n_unmask = unmask_count(masked.len(), step, steps);
                let mut ranked: Vec<(usize, f32)> = masked
                    .iter()
                    .map(|&p| (p, conf.get(p).copied().unwrap_or(0.0)))
                    .collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                for &(pos, _) in ranked.iter().take(n_unmask) {
                    if let (Some(slot), Some(&tok)) = (block.get_mut(pos), preds.get(pos)) {
                        *slot = tok;
                    }
                }
            }

            // Fill any leftover masks with their last prediction.
            let still: Vec<usize> = (0..blk)
                .filter(|&p| block.get(p).copied() == Some(mask_id))
                .collect();
            if !still.is_empty() {
                let (preds, _) = self.predict_block(&block, &cache)?;
                for pos in still {
                    if let (Some(slot), Some(&tok)) = (block.get_mut(pos), preds.get(pos)) {
                        *slot = tok;
                    }
                }
            }

            if let Some(cb) = on_block.as_deref_mut() {
                cb(&block);
            }
            generated.extend_from_slice(&block);

            // Stop once EOS appears; otherwise commit the block and reseed.
            if eos_id.is_some_and(|e| block.contains(&e)) || generated.len() >= num_tokens {
                break;
            }
            seed = self.commit_block(&mut cache, &block)?;
        }

        Ok(generated)
    }

    /// Linear self-speculation (the reference `linear_spec_generate`): causal
    /// prefill, then per block draft the whole block in one bidirectional
    /// LoRA-on forward, verify it in one causal LoRA-off forward, accept the
    /// longest prefix where the AR argmax agrees with the draft plus the AR
    /// bonus token, and advance the KV cache by the accepted count.
    ///
    /// Greedy: output is AR-exact regardless of draft quality, so the LoRA only
    /// raises the acceptance length (the speedup). Returns `(tokens, stats)`;
    /// `tokens` excludes the prompt.
    pub fn linear_spec_generate(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        block_length: usize,
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, LinSpecStats), Exception> {
        let block_len = block_length.max(1);

        let (mut cache, mut seed) = self.prefill(prompt_ids)?;
        let mut stats = LinSpecStats {
            nfe: 1, // prefill
            blocks: 0,
            accepted_total: 0,
        };
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        generated.push(seed); // prefill seed is the first generated token
        if eos_token_id == Some(seed) {
            return Ok((generated, stats));
        }

        while generated.len() < max_new_tokens {
            // Draft (bidirectional, LoRA on) + verify (causal, LoRA off) in one
            // device round with a single host barrier.
            let (drafted, ar_tokens, block_kv) = self.draft_and_verify(seed, block_len, &cache)?;
            stats.nfe += 2;
            stats.blocks += 1;

            // Accept the longest run where AR agrees with the draft (shifted by
            // one), then the AR bonus token at the tail.
            let mut accepted = 0usize;
            for i in 0..block_len.saturating_sub(1) {
                if ar_tokens.get(i).copied() == drafted.get(i + 1).copied() {
                    accepted += 1;
                } else {
                    break;
                }
            }
            accepted = (accepted + 1).min(block_len).min(ar_tokens.len());
            stats.accepted_total += accepted;

            // Committed tokens are the AR argmax of the first `accepted` positions.
            let accepted_toks = ar_tokens.get(..accepted).unwrap_or(&[]);

            // EOS truncation within the accepted run.
            if let Some(eos) = eos_token_id {
                if let Some(p) = accepted_toks.iter().position(|&t| t == eos) {
                    let head = accepted_toks.get(..=p).unwrap_or(accepted_toks);
                    generated.extend_from_slice(head);
                    let acc =
                        i32::try_from(p + 1).map_err(|_| Exception::custom("accepted overflow"))?;
                    extend_cache_by(&mut cache, &block_kv, acc)?;
                    return Ok((generated, stats));
                }
            }

            generated.extend_from_slice(accepted_toks);
            let acc =
                i32::try_from(accepted).map_err(|_| Exception::custom("accepted overflow"))?;
            extend_cache_by(&mut cache, &block_kv, acc)?;
            seed = accepted
                .checked_sub(1)
                .and_then(|i| ar_tokens.get(i).copied())
                .ok_or_else(|| Exception::custom("empty accept"))?;
        }

        generated.truncate(max_new_tokens);
        Ok((generated, stats))
    }
}

/// Force the cached K/V arrays to be evaluated once, so the prefix isn't
/// recomputed on every denoising step.
fn materialize(per_layer: &[(Array, Array)]) -> Result<(), Exception> {
    for (k, v) in per_layer {
        mlx_rs::transforms::eval([k, v])?;
    }
    Ok(())
}

/// Append the first `accepted` positions of a verified block's per-layer K/V to
/// `cache` (partial commit for linear-spec acceptance). `accepted == block_len`
/// is the full-block case used by `commit_block`.
fn extend_cache_by(
    cache: &mut DiffusionKvCache,
    block_kv: &[(Array, Array)],
    accepted: i32,
) -> Result<(), Exception> {
    let mut extended = Vec::with_capacity(cache.per_layer.len());
    for ((pk, pv), (bk, bv)) in cache.per_layer.iter().zip(block_kv) {
        let bk_acc = bk.index((.., .., 0..accepted, ..));
        let bv_acc = bv.index((.., .., 0..accepted, ..));
        extended.push((
            ops::concatenate_axis(&[pk, &bk_acc], -2)?,
            ops::concatenate_axis(&[pv, &bv_acc], -2)?,
        ));
    }
    cache.per_layer = extended;
    cache.len += accepted;
    materialize(&cache.per_layer)?;
    Ok(())
}

/// Number of masked positions to un-mask at `step` of `steps`, given `n_masks`
/// currently masked. Linear schedule that always un-masks ≥1 and finishes the
/// block on the last step.
fn unmask_count(n_masks: usize, step: usize, steps: usize) -> usize {
    if n_masks == 0 {
        return 0;
    }
    let total = steps.max(1);
    if step + 1 >= total {
        return n_masks;
    }
    let keep = n_masks.saturating_mul(total - step - 1) / total;
    n_masks.saturating_sub(keep).max(1)
}

/// Load model args from config.json.
pub fn load_nemotron_diffusion_model_args<P: AsRef<Path>>(
    model_dir: P,
) -> Result<NemotronDiffusionArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

/// Extract `<i>` from an adapter key containing `layers.<i>.`.
fn layer_index_from_key(key: &str) -> Option<usize> {
    let after = key.split("layers.").nth(1)?;
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Load the bundled linear-spec drafter LoRA (`linear_spec_lora/`) if present.
/// Returns one [`OProjLora`] per layer, indexed by the `layers.<i>.` in the
/// adapter keys (matched by suffix, so the exact PEFT prefix is irrelevant), or
/// `None` when the adapter directory is absent or incomplete.
fn load_linear_spec_lora(
    model_dir: &Path,
    num_layers: usize,
) -> Result<Option<Vec<OProjLora>>, ModelError> {
    let dir = model_dir.join("linear_spec_lora");
    let weights = dir.join("adapter_model.safetensors");
    if !weights.exists() {
        return Ok(None);
    }

    // scale = lora_alpha / r from adapter_config.json (integers; fallback 1.0).
    let scale = std::fs::read_to_string(dir.join("adapter_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .map_or(1.0_f32, |cfg| {
            let alpha_opt = cfg
                .get("lora_alpha")
                .and_then(serde_json::Value::as_u64)
                .and_then(|v| u16::try_from(v).ok());
            let rank_opt = cfg
                .get("r")
                .and_then(serde_json::Value::as_u64)
                .and_then(|v| u16::try_from(v).ok());
            match (alpha_opt, rank_opt) {
                (Some(alpha), Some(rank)) if rank > 0 => f32::from(alpha) / f32::from(rank),
                _ => 1.0,
            }
        });

    let loaded = Array::load_safetensors(&weights)
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

    let mut a: std::collections::HashMap<usize, Array> = std::collections::HashMap::new();
    let mut b: std::collections::HashMap<usize, Array> = std::collections::HashMap::new();
    for (key, value) in loaded {
        let key_str: &str = &key;
        let Some(idx) = layer_index_from_key(key_str) else {
            continue;
        };
        if key_str.ends_with("lora_A.weight") {
            a.insert(idx, value);
        } else if key_str.ends_with("lora_B.weight") {
            b.insert(idx, value);
        }
    }

    let mut out = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let (Some(av), Some(bv)) = (a.get(&i), b.get(&i)) else {
            tracing::warn!(
                layer = i,
                "linear_spec_lora missing o_proj A/B; ignoring adapter"
            );
            return Ok(None);
        };
        // PEFT stores A=[r,in], B=[out,r]; store transposed so apply is two matmuls.
        out.push(OProjLora {
            a_t: av.transpose().map_err(ModelError::Mlx)?,
            b_t: bv.transpose().map_err(ModelError::Mlx)?,
            scale,
        });
    }
    Ok(Some(out))
}

/// Load a Nemotron-Labs-Diffusion model from a directory.
pub fn load_nemotron_diffusion_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<NemotronDiffusionLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_nemotron_diffusion_model_args(model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        vocab_size = args.vocab_size,
        mask_token_id = args.mask_token_id,
        block_size = args.block_size,
        diffusion_steps = args.diffusion_steps,
        "Loading Nemotron-Labs-Diffusion model (diffusion mode)"
    );

    let quantization = args.quantization.clone();
    let raw_model = NemotronDiffusionLM::new(args).map_err(ModelError::Mlx)?;
    // Pre-quantized MLX checkpoints store packed 4-bit weights; convert the
    // module structure to quantized form (and remap the flat `.scales`/`.biases`
    // keys) before loading, exactly as the autoregressive transformer path does.
    let mut model = if let Some(ref qc) = quantization {
        tracing::info!(
            group_size = qc.group_size,
            bits = qc.bits,
            "Applying quantization structure"
        );
        mlx_rs::nn::quantize(raw_model, qc.group_size, qc.bits).map_err(|e| {
            ModelError::ShapeMismatch(format!("Failed to quantize model structure: {e}"))
        })?
    } else {
        raw_model
    };
    crate::load_quantized_safetensors_weights(&mut model, model_path, quantization.is_some())?;

    // Attach the linear-spec drafter LoRA (o_proj) when the adapter ships with
    // the checkpoint; absent for plain diffusion-only checkpoints.
    if let Some(loras) = load_linear_spec_lora(model_path, model.encoder.layers.len())? {
        for (layer, lora) in model.encoder.layers.iter_mut().zip(loras) {
            layer.self_attn.o_lora = Some(lora);
        }
        tracing::info!("Attached linear_spec drafter LoRA (o_proj)");
    }

    tracing::info!("Nemotron-Labs-Diffusion model loaded successfully");
    Ok(model)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stderr,
    clippy::shadow_reuse
)]
mod tests {
    use super::*;

    fn small_args() -> NemotronDiffusionArgs {
        NemotronDiffusionArgs {
            model_type: "nemotron_labs_diffusion".to_owned(),
            hidden_size: 256,
            num_hidden_layers: 2,
            intermediate_size: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vocab_size: 320,
            rms_norm_eps: 1e-5,
            head_dim: Some(64),
            tie_word_embeddings: false,
            rope_parameters: RopeParameters::default(),
            quantization: None,
            eos_token_id: None,
            mask_token_id: 300,
            block_size: 8,
            diffusion_steps: 4,
        }
    }

    #[test]
    fn config_deserializes_with_nested_rope_and_diffusion_fields() {
        let json = r#"{
            "architectures": ["NemotronLabsDiffusionModel"],
            "model_type": "nemotron_labs_diffusion",
            "hidden_size": 3072,
            "num_hidden_layers": 26,
            "intermediate_size": 9216,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 131072,
            "rms_norm_eps": 1e-05,
            "head_dim": 128,
            "block_size": 32,
            "mask_token_id": 100,
            "tie_word_embeddings": false,
            "rope_parameters": {
                "rope_type": "yarn", "factor": 16.0, "rope_theta": 1000000.0,
                "original_max_position_embeddings": 16384,
                "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0
            }
        }"#;
        let args: NemotronDiffusionArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "nemotron_labs_diffusion");
        assert_eq!(args.mask_token_id, 100);
        assert_eq!(args.block_size, 32);
        assert_eq!(args.head_dim, Some(128));
        assert_eq!(args.checked_head_dim().unwrap(), 128);
        assert!((args.rope_parameters.rope_theta - 1_000_000.0).abs() < f32::EPSILON);
        assert!((args.rope_parameters.factor - 16.0).abs() < f32::EPSILON);
    }

    #[test]
    fn unmask_count_schedule_is_monotone_and_finishes() {
        // Always un-masks at least one and finishes on the last step.
        let steps = 4;
        let mut remaining = 8usize;
        for step in 0..steps {
            if remaining == 0 {
                break;
            }
            let u = unmask_count(remaining, step, steps);
            assert!(u >= 1, "must unmask at least one while masks remain");
            assert!(u <= remaining);
            remaining -= u;
        }
        assert_eq!(remaining, 0, "block must be fully unmasked after all steps");
        assert_eq!(unmask_count(0, 0, 4), 0);
        assert_eq!(unmask_count(5, 3, 4), 5, "last step unmasks all remaining");
    }

    #[test]
    fn diffusion_generate_resolves_all_masks() {
        // Tiny random model: exercises the full MLX graph (embed → layers →
        // yarn rope → bidirectional sdpa → diffusion_head) + the denoising loop,
        // with no weights on disk. Asserts the loop terminates with no masks.
        let args = small_args();
        let mask_id = args.mask_token_id;
        let mut model = NemotronDiffusionLM::new(args).unwrap();
        let prompt = [1_u32, 2, 3];
        let out = model.diffusion_generate(&prompt, 8, 4, 8, None).unwrap();
        assert_eq!(out.len(), 8);
        assert!(
            out.iter().all(|&t| t != mask_id),
            "no mask tokens should remain"
        );
        assert!(out.iter().all(|&t| t < 320), "tokens within vocab");
    }

    #[test]
    fn diffusion_generate_is_deterministic_greedy() {
        let mut model = NemotronDiffusionLM::new(small_args()).unwrap();
        let prompt = [4_u32, 5];
        let a = model.diffusion_generate(&prompt, 8, 4, 8, None).unwrap();
        let b = model.diffusion_generate(&prompt, 8, 4, 8, None).unwrap();
        assert_eq!(a, b, "greedy diffusion decode must be deterministic");
    }

    /// End-to-end check against real weights. Loads the model + tokenizer from
    /// `HIGGS_NEMOTRON_DIFFUSION_DIR`, runs greedy diffusion decode, and asserts
    /// the loop terminates with no mask tokens and non-empty text. The decoded
    /// output is printed for manual coherence inspection (`--nocapture`).
    #[test]
    #[ignore = "requires real Nemotron-Labs-Diffusion weights; set HIGGS_NEMOTRON_DIFFUSION_DIR"]
    fn diffusion_generate_real_model_terminates_and_decodes() {
        let Some(dir) = std::env::var_os("HIGGS_NEMOTRON_DIFFUSION_DIR") else {
            eprintln!("skipping: set HIGGS_NEMOTRON_DIFFUSION_DIR to a model directory");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let mut model = load_nemotron_diffusion_model(&dir).unwrap();
        let mask_id = model.args.mask_token_id;
        let tokenizer = crate::load_tokenizer(&dir).unwrap();

        let enc = tokenizer.encode("The capital of France is", true).unwrap();
        let prompt_ids = enc.get_ids();
        let out = model
            .diffusion_generate(prompt_ids, 24, 24, 32, None)
            .unwrap();
        let text = tokenizer.decode(&out, true).unwrap();
        eprintln!("Nemotron diffusion output: {text:?}");

        // EOS may stop generation before the full 24 requested tokens.
        assert!(
            out.len() <= 24 && !out.is_empty(),
            "got {} tokens",
            out.len()
        );
        assert!(
            out.iter().all(|&t| t != mask_id),
            "no mask tokens should remain"
        );
        assert!(!text.trim().is_empty(), "decoded text must be non-empty");
        // Coherence gate: greedy decode of this prompt must name the capital.
        // If the weights are mis-loaded (e.g. the output head isn't mapped) the
        // model emits noise and this fails -- which is exactly what we want.
        assert!(
            text.to_lowercase().contains("paris"),
            "expected a coherent answer naming Paris, got: {text:?}"
        );
    }

    /// Decode-only throughput baseline for diffusion (dlm) mode on real weights.
    /// Loads once, warms up Metal, then times `diffusion_generate` at two step
    /// counts to show the steps<->parallelism tradeoff (fewer steps = more
    /// tokens per forward = faster, lower quality). Run with `--ignored
    /// --nocapture`. This is the "before" baseline for the linear_spec drafter.
    #[test]
    #[ignore = "requires real Nemotron-Labs-Diffusion weights; set HIGGS_NEMOTRON_DIFFUSION_DIR"]
    fn dlm_throughput_baseline() {
        let Some(dir) = std::env::var_os("HIGGS_NEMOTRON_DIFFUSION_DIR") else {
            eprintln!("skipping: set HIGGS_NEMOTRON_DIFFUSION_DIR to a model directory");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let mut model = load_nemotron_diffusion_model(&dir).unwrap();
        let tokenizer = crate::load_tokenizer(&dir).unwrap();
        let enc = tokenizer
            .encode("Explain in detail how photosynthesis works:", true)
            .unwrap();
        let prompt = enc.get_ids();

        // Warm up Metal kernels (not timed).
        let _ = model.diffusion_generate(prompt, 16, 8, 32, None).unwrap();

        for (num_tokens, steps, block) in [
            (128usize, 32usize, 32usize),
            (128, 16, 32),
            (128, 12, 32),
            (128, 8, 32),
        ] {
            let t = std::time::Instant::now();
            let out = model
                .diffusion_generate(prompt, num_tokens, steps, block, None)
                .unwrap();
            let secs = t.elapsed().as_secs_f64();
            let toks = out.len();
            let tps = f64::from(u32::try_from(toks).unwrap()) / secs;
            eprintln!(
                "dlm baseline: {toks} tok in {secs:.3}s = {tps:.1} tok/s (req={num_tokens}, steps={steps}, block={block})"
            );
        }
    }

    /// Greedy `linear_spec_generate` is AR-exact: the block size only changes how
    /// many tokens are verified per forward, never the resulting sequence. With
    /// `block=1` the loop degrades to pure autoregressive decode, so a larger
    /// block must produce the identical tokens. Catches off-by-one errors in the
    /// accept / cache-crop / reseed logic (runs without a LoRA adapter).
    #[test]
    fn linear_spec_is_ar_exact_across_block_sizes() {
        let mut model = NemotronDiffusionLM::new(small_args()).unwrap();
        let prompt = [1u32, 2, 3, 4, 5];

        let (ar, ar_stats) = model.linear_spec_generate(&prompt, 16, 1, None).unwrap();
        let (blk, blk_stats) = model.linear_spec_generate(&prompt, 16, 8, None).unwrap();

        assert_eq!(ar.len(), 16, "block=1 should fill max_new_tokens");
        assert_eq!(blk.len(), 16);
        assert_eq!(
            ar, blk,
            "block size must not change greedy output (AR-exact)"
        );
        // block=1 verifies one token per forward; block=8 must use no more NFE.
        assert!(blk_stats.nfe <= ar_stats.nfe);
        assert!(blk_stats.blocks >= 1 && blk_stats.accepted_total >= 1);
    }

    /// End-to-end linear-spec decode on real weights + LoRA drafter: coherent
    /// ("Paris"), and prints throughput / NFE / TPF / mean acceptance to compare
    /// against the dlm baseline. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "requires real Nemotron-Labs-Diffusion weights; set HIGGS_NEMOTRON_DIFFUSION_DIR"]
    fn linear_spec_real_model_paris_and_throughput() {
        let Some(dir) = std::env::var_os("HIGGS_NEMOTRON_DIFFUSION_DIR") else {
            eprintln!("skipping: set HIGGS_NEMOTRON_DIFFUSION_DIR to a model directory");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let mut model = load_nemotron_diffusion_model(&dir).unwrap();
        let eos = model.args.eos_token_id;
        let tokenizer = crate::load_tokenizer(&dir).unwrap();
        let enc = tokenizer.encode("The capital of France is", true).unwrap();
        let prompt = enc.get_ids();

        let _ = model.linear_spec_generate(prompt, 8, 16, eos).unwrap(); // warm up

        // Sweep block_length: too wide wastes drafted positions the AR verify
        // rejects; the sweet spot sits near the acceptance length.
        let mut last_text = String::new();
        for block in [4usize, 8, 16, 32] {
            let t = std::time::Instant::now();
            let (out, stats) = model.linear_spec_generate(prompt, 128, block, eos).unwrap();
            let secs = t.elapsed().as_secs_f64();
            last_text = tokenizer.decode(&out, true).unwrap();
            let toks = f64::from(u32::try_from(out.len()).unwrap());
            let tps = toks / secs;
            let tpf = toks / f64::from(u32::try_from(stats.nfe).unwrap());
            let mean_accept = f64::from(u32::try_from(stats.accepted_total).unwrap())
                / f64::from(u32::try_from(stats.blocks.max(1)).unwrap());
            eprintln!(
                "linear_spec block={block}: {} tok in {secs:.3}s = {tps:.1} tok/s | nfe={} TPF={tpf:.2} mean_accept={mean_accept:.2}",
                out.len(),
                stats.nfe
            );
        }
        eprintln!("linear_spec output (block=32): {last_text:?}");

        assert!(
            last_text.to_lowercase().contains("paris"),
            "expected a coherent answer naming Paris, got: {last_text:?}"
        );
    }

    /// Drift-free head-to-head: one model load, one warm-up, one thermal window.
    /// Runs dlm at several step counts and linear_spec at a couple block sizes
    /// back-to-back so the tok/s are directly comparable (cross-process runs
    /// drift ~15-20% with thermal). Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "requires real Nemotron-Labs-Diffusion weights; set HIGGS_NEMOTRON_DIFFUSION_DIR"]
    fn dlm_vs_linear_spec_headtohead() {
        let Some(dir) = std::env::var_os("HIGGS_NEMOTRON_DIFFUSION_DIR") else {
            eprintln!("skipping: set HIGGS_NEMOTRON_DIFFUSION_DIR to a model directory");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let mut model = load_nemotron_diffusion_model(&dir).unwrap();
        let eos = model.args.eos_token_id;
        let tokenizer = crate::load_tokenizer(&dir).unwrap();
        let enc = tokenizer
            .encode("Explain in detail how photosynthesis works:", true)
            .unwrap();
        let prompt = enc.get_ids();

        // One warm-up for the whole comparison.
        let _ = model.diffusion_generate(prompt, 16, 8, 32, None).unwrap();
        let _ = model.linear_spec_generate(prompt, 16, 8, eos).unwrap();

        eprintln!("== head-to-head (one process, 128 tok, block 32) ==");
        for steps in [8usize, 12, 16, 32] {
            let t = std::time::Instant::now();
            let out = model
                .diffusion_generate(prompt, 128, steps, 32, None)
                .unwrap();
            let secs = t.elapsed().as_secs_f64();
            let tps = f64::from(u32::try_from(out.len()).unwrap()) / secs;
            eprintln!("  dlm steps={steps:>2}: {tps:>5.1} tok/s  (diffusion quality)");
        }
        for block in [4usize, 8] {
            let t = std::time::Instant::now();
            let (out, stats) = model.linear_spec_generate(prompt, 128, block, eos).unwrap();
            let secs = t.elapsed().as_secs_f64();
            let toks = f64::from(u32::try_from(out.len()).unwrap());
            let tps = toks / secs;
            let mean_accept = f64::from(u32::try_from(stats.accepted_total).unwrap())
                / f64::from(u32::try_from(stats.blocks.max(1)).unwrap());
            eprintln!(
                "  linear_spec block={block}: {tps:>5.1} tok/s  (AR-exact, mean_accept {mean_accept:.2})"
            );
            assert!(!out.is_empty());
        }
    }
}
