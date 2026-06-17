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
    utils::scaled_dot_product_attention,
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
        })
    }

    #[allow(non_snake_case)]
    fn forward(&mut self, x: &Array, rope: &RopeState) -> Result<Array, Exception> {
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

        // YaRN rope over absolute positions 0..L (diffusion re-forwards the
        // whole canvas each step, so the offset is always zero).
        let offset = Array::from_int(0);
        let queries_roped = apply_yarn_rope(
            &queries,
            self.head_dim,
            rope.base,
            Some(&rope.freqs),
            rope.mscale,
            &offset,
            false,
        )?;
        let keys_roped = apply_yarn_rope(
            &keys,
            self.head_dim,
            rope.base,
            Some(&rope.freqs),
            rope.mscale,
            &offset,
            false,
        )?;

        // Bidirectional attention: no mask.
        let output =
            scaled_dot_product_attention(queries_roped, keys_roped, values, self.scale, None)?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, L, -1])?;
        self.o_proj.forward(&output)
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

    fn forward(&mut self, x: &Array, rope: &RopeState) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let h = x.add(self.self_attn.forward(&normed, rope)?)?;
        let normed_post = self.post_attention_layernorm.forward(&h)?;
        h.add(self.mlp.forward(&normed_post)?)
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

    fn forward(&mut self, inputs: &Array, rope: &RopeState) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(inputs)?;
        for layer in &mut self.layers {
            h = layer.forward(&h, rope)?;
        }
        self.norm.forward(&h)
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

    /// Forward the whole canvas, returning logits for every position
    /// `[1, L, vocab]`. Bidirectional (no causal mask).
    fn forward_canvas(&mut self, canvas: &Array) -> Result<Array, Exception> {
        let hidden = self.encoder.forward(canvas, &self.rope)?;
        self.diffusion_head.forward(&hidden)
    }

    /// Diffusion decode: generate `num_tokens` tokens by block-wise masked
    /// denoising. Greedy (argmax) selection; `steps` denoising iterations per
    /// block of `block_size`. `on_block` is invoked with each finalized block's
    /// token ids (for streaming).
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
        let steps = steps.max(1);
        let block_size = block_size.max(1);

        let mut canvas: Vec<u32> = prompt_ids.to_vec();
        let mut generated: Vec<u32> = Vec::with_capacity(num_tokens);

        while generated.len() < num_tokens {
            let blk = block_size.min(num_tokens - generated.len());
            let block_start = canvas.len();
            canvas.extend(std::iter::repeat_n(mask_id, blk));

            for step in 0..steps {
                // Positions in the current block still masked.
                let masked: Vec<usize> = (block_start..block_start + blk)
                    .filter(|&p| canvas.get(p).copied() == Some(mask_id))
                    .collect();
                if masked.is_empty() {
                    break;
                }

                let (preds, confidences) = self.predict_block(&canvas, block_start, blk)?;

                // Rank masked positions by confidence (descending) and unmask
                // the scheduled count this step.
                let n_unmask = unmask_count(masked.len(), step, steps);
                let mut ranked: Vec<(usize, f32)> = Vec::with_capacity(masked.len());
                for &pos in &masked {
                    let local = pos - block_start;
                    let conf = confidences.get(local).copied().unwrap_or(0.0);
                    ranked.push((pos, conf));
                }
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

                for &(pos, _) in ranked.iter().take(n_unmask) {
                    let local = pos - block_start;
                    if let (Some(slot), Some(&tok)) = (canvas.get_mut(pos), preds.get(local)) {
                        *slot = tok;
                    }
                }
            }

            // Any positions left masked after the schedule (shouldn't happen on
            // the final step) are filled with their last prediction.
            let still_masked: Vec<usize> = (block_start..block_start + blk)
                .filter(|&p| canvas.get(p).copied() == Some(mask_id))
                .collect();
            if !still_masked.is_empty() {
                let (preds, _) = self.predict_block(&canvas, block_start, blk)?;
                for pos in still_masked {
                    let local = pos - block_start;
                    if let (Some(slot), Some(&tok)) = (canvas.get_mut(pos), preds.get(local)) {
                        *slot = tok;
                    }
                }
            }

            let block_ids: Vec<u32> = canvas
                .get(block_start..block_start + blk)
                .map(<[u32]>::to_vec)
                .unwrap_or_default();
            if let Some(cb) = on_block.as_deref_mut() {
                cb(&block_ids);
            }
            generated.extend_from_slice(&block_ids);
        }

        Ok(generated)
    }

    /// Run one forward over the canvas and return, for each position in the
    /// current block, the argmax token id and its softmax confidence.
    #[allow(non_snake_case)]
    fn predict_block(
        &mut self,
        canvas: &[u32],
        block_start: usize,
        blk: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), Exception> {
        let L = i32::try_from(canvas.len()).map_err(|_| Exception::custom("canvas too long"))?;
        let input = Array::from_slice(canvas, &[1, L]);
        let logits = self.forward_canvas(&input)?;

        let start = i32::try_from(block_start).map_err(|_| Exception::custom("block_start"))?;
        let len = i32::try_from(blk).map_err(|_| Exception::custom("blk"))?;
        // [1, blk, vocab] -> [blk, vocab]
        let block_logits = logits
            .index((0, start..start + len, ..))
            .reshape(&[len, -1])?;

        let probs = ops::softmax_axis(&block_logits, -1, true)?;
        let conf = probs.max_axis(-1, false)?.as_dtype(Dtype::Float32)?;
        let preds_arr =
            ops::indexing::argmax_axis(&block_logits, -1, false)?.as_dtype(Dtype::Uint32)?;
        mlx_rs::transforms::eval([&conf, &preds_arr])?;

        let conf_host: Vec<f32> = conf.as_slice::<f32>().to_vec();
        let preds: Vec<u32> = preds_arr.as_slice::<u32>().to_vec();
        Ok((preds, conf_host))
    }
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

        assert_eq!(out.len(), 24);
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
}
