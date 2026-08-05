//! LFM2 (Liquid Foundation Model 2) hybrid conv/attention architecture.
//!
//! Alternates between double-gated short convolution blocks (depthwise causal
//! conv with multiplicative gating) and standard GQA attention blocks, all
//! sharing a SwiGLu FFN. The layer pattern is config-driven via `layer_types`.
//!
//! See: https://huggingface.co/LiquidAI/LFM2-2.6B

use std::path::Path;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

use crate::cache::{KeyValueCache, SteppingKeyValueCache};
use crate::error::ModelError;
use crate::qwen3_next::{ArraysCache, LayerCache};
use crate::utils::apply_rope;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Lfm2Config {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_conv_l_cache")]
    pub conv_l_cache: i32,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub bits: u8,
    pub group_size: i32,
    #[serde(default = "default_quant_mode")]
    pub mode: String,
}

fn default_conv_l_cache() -> i32 { 3 }
fn default_rope_theta() -> f32 { 1_000_000.0 }
fn default_norm_eps() -> f32 { 1e-5 }
fn default_max_pos() -> i32 { 128_000 }
fn default_true() -> bool { true }
fn default_quant_mode() -> String { "affine".to_owned() }

impl Lfm2Config {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, ModelError> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| ModelError::Io(e))?;
        let config: Lfm2Config = serde_json::from_str(&config_str)
            .map_err(|e| ModelError::Json(e))?;
        if config.layer_types.len() != config.num_hidden_layers as usize {
            return Err(ModelError::UnsupportedModel(format!(
                "lfm2: layer_types has {} entries but num_hidden_layers is {}",
                config.layer_types.len(), config.num_hidden_layers
            )));
        }
        let conv = config.layer_types.iter().filter(|t| t.as_str() == "conv").count();
        let attn = config.layer_types.iter().filter(|t| t.as_str() == "full_attention").count();
        tracing::info!(
            hidden_size = config.hidden_size,
            layers = config.num_hidden_layers,
            attn_heads = config.num_attention_heads,
            kv_heads = config.num_key_value_heads,
            head_dim = config.head_dim(),
            conv_layers = conv, attn_layers = attn,
            conv_kernel = config.conv_l_cache,
            intermediate = config.intermediate_size,
            "Loading LFM2 model"
        );
        Ok(config)
    }

    pub const fn head_dim(&self) -> i32 { self.hidden_size / self.num_attention_heads }

    pub fn is_attention_layer(&self, idx: usize) -> bool {
        self.layer_types.get(idx).map(|t| t.as_str() == "full_attention").unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Short convolution block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Lfm2ShortConv {
    #[param]
    in_proj: MaybeQuantized<nn::Linear>,
    #[param]
    conv: nn::Conv1d,
    #[param]
    out_proj: MaybeQuantized<nn::Linear>,
    conv_kernel: i32,
    hidden_size: i32,
}

impl Lfm2ShortConv {
    pub fn new(config: &Lfm2Config) -> Result<Self, Exception> {
        let h = config.hidden_size;
        let k = config.conv_l_cache;
        Ok(Self {
            in_proj: MaybeQuantized::Original(nn::Linear::new(h, 3 * h)?),
            conv: nn::Conv1dBuilder::new(h, h, k).bias(false).groups(h).padding(0).build()?,
            out_proj: MaybeQuantized::Original(nn::Linear::new(h, h)?),
            conv_kernel: k,
            hidden_size: h,
        })
    }

    /// Prefill forward over T tokens. `hidden`: `[B, T, H]` → `[B, T, H]`.
    pub fn forward(&mut self, hidden: &Array, cache: &mut ArraysCache) -> Result<Array, Exception> {
        let shape = hidden.shape();
        let (b, t) = (shape[0], shape[1]);
        let h = self.hidden_size;

        let projected = self.in_proj.forward(hidden)?;
        let reshaped = projected.reshape(&[b, t, 3, h])?;
        let gate_b = reshaped.index((.., .., 0, ..));
        let gate_c = reshaped.index((.., .., 1, ..));
        let value_x = reshaped.index((.., .., 2, ..));

        let gated = gate_b.multiply(&value_x)?; // [B, T, H]
        let conv_in = gated.transpose_axes(&[0, 2, 1])?; // [B, H, T]

        // Left-pad by K-1, depthwise conv, trim to T
        if t > 0 {
            let pad = ops::zeros_dtype(&[b, h, self.conv_kernel - 1], conv_in.dtype())?;
            let padded = ops::concatenate_axis(&[&pad, &conv_in], -1)?;
            let conv_out = self.conv.forward(&padded)?;
            let conv_trimmed = conv_out.index((.., .., ..t));

            let gated_conv = gate_c.transpose_axes(&[0, 2, 1])?.multiply(&conv_trimmed)?;
            let out = gated_conv.transpose_axes(&[0, 2, 1])?;

            // Update conv_state: keep last K-1 of conv_in
            let keep_from = (t - (self.conv_kernel - 1)).max(0);
            cache.conv_state = Some(conv_in.index((.., .., keep_from..)).clone());

            self.out_proj.forward(&out)
        } else {
            self.out_proj.forward(&gated)
        }
    }

    /// Single-step decode. `hidden`: `[B, 1, H]` → `[B, 1, H]`.
    pub fn forward_decode(&mut self, hidden: &Array, cache: &mut ArraysCache) -> Result<Array, Exception> {
        let shape = hidden.shape();
        let (b, _t) = (shape[0], shape[1]);
        let h = self.hidden_size;
        let k = self.conv_kernel;

        let projected = self.in_proj.forward(hidden)?;
        let reshaped = projected.reshape(&[b, 1, 3, h])?;
        let gate_b = reshaped.index((.., .., 0, ..));
        let gate_c = reshaped.index((.., .., 1, ..));
        let value_x = reshaped.index((.., .., 2, ..));

        let gated = gate_b.multiply(&value_x)?;
        let conv_in = gated.transpose_axes(&[0, 2, 1])?; // [B, H, 1]

        // Build conv input from state + new token
        let conv_full = if let Some(prev) = cache.conv_state.take() {
            ops::concatenate_axis(&[&prev, &conv_in], -1)?
        } else {
            let zeros = ops::zeros_dtype(&[b, h, k - 1], conv_in.dtype())?;
            ops::concatenate_axis(&[&zeros, &conv_in], -1)?
        };

        // Keep last K-1 as new state
        let conv_len = conv_full.shape().last().copied().unwrap_or(0);
        let keep_from = (conv_len - (k - 1)).max(0);
        cache.conv_state = Some(conv_full.index((.., .., keep_from..)).clone());

        let conv_out = self.conv.forward(&conv_full)?; // [B, H, ?]
        let last_idx = conv_out.shape()[2] - 1;
        let conv_last = conv_out.index((.., .., last_idx..));

        let gated_conv = gate_c.transpose_axes(&[0, 2, 1])?.multiply(&conv_last)?;
        let out = gated_conv.transpose_axes(&[0, 2, 1])?;

        self.out_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// GQA Attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Lfm2Attention {
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,
    #[param]
    out_proj: MaybeQuantized<nn::Linear>,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope: nn::Rope,
}

impl Lfm2Attention {
    pub fn new(config: &Lfm2Config) -> Result<Self, Exception> {
        let h = config.hidden_size;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim();
        Ok(Self {
            q_proj: MaybeQuantized::Original(nn::Linear::new(h, nh * hd)?),
            k_proj: MaybeQuantized::Original(nn::Linear::new(h, nkv * hd)?),
            v_proj: MaybeQuantized::Original(nn::Linear::new(h, nkv * hd)?),
            out_proj: MaybeQuantized::Original(nn::Linear::new(h, nh * hd)?),
            q_norm: nn::RmsNormBuilder::new(hd).eps(config.norm_eps).build()?,
            k_norm: nn::RmsNormBuilder::new(hd).eps(config.norm_eps).build()?,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            scale: 1.0 / f32::sqrt(hd as f32),
            rope: nn::RopeBuilder::new(hd).traditional(false).base(config.rope_theta).build()?,
        })
    }

    pub fn forward(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        kv_cache: &mut SteppingKeyValueCache,
    ) -> Result<Array, Exception> {
        let shape = hidden.shape();
        let (b, t) = (shape[0], shape[1]);
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

        let q = self.q_proj.forward(hidden)?.reshape(&[b, t, nh, hd])?.transpose_axes(&[0, 2, 1, 3])?;
        let k = self.k_proj.forward(hidden)?.reshape(&[b, t, nkv, hd])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = self.v_proj.forward(hidden)?.reshape(&[b, t, nkv, hd])?.transpose_axes(&[0, 2, 1, 3])?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let offset = kv_cache.offset();
        let q = apply_rope(&q, &self.rope, offset)?;
        let k = apply_rope(&k, &self.rope, offset)?;

        let view = kv_cache.update_and_view(k, v)?;
        let (cached_keys, cached_values) = view.into_dense()?;

        let sdpa_mask = mask.map(mlx_rs::fast::ScaledDotProductAttentionMask::from);
        let out = mlx_rs::fast::scaled_dot_product_attention(
            &q, &cached_keys, &cached_values, self.scale, sdpa_mask, None::<&Array>,
        )?;

        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, nh * hd])?;
        self.out_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// SwiGLu MLP
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Lfm2MLP {
    #[param]
    w1: MaybeQuantized<nn::Linear>,
    #[param]
    w2: MaybeQuantized<nn::Linear>,
    #[param]
    w3: MaybeQuantized<nn::Linear>,
}

impl Lfm2MLP {
    pub fn new(config: &Lfm2Config) -> Result<Self, Exception> {
        let (h, i) = (config.hidden_size, config.intermediate_size);
        Ok(Self {
            w1: MaybeQuantized::Original(nn::Linear::new(h, i)?),
            w2: MaybeQuantized::Original(nn::Linear::new(i, h)?),
            w3: MaybeQuantized::Original(nn::Linear::new(h, i)?),
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.w1.forward(x)?;
        let up = self.w3.forward(x)?;
        let activated = gate.multiply(&nn::sigmoid(&gate)?)?.multiply(&up)?;
        self.w2.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Lfm2DecoderLayer {
    #[param]
    conv: Option<Lfm2ShortConv>,
    #[param]
    attn: Option<Lfm2Attention>,
    #[param]
    operator_norm: nn::RmsNorm,
    #[param]
    ffn_norm: nn::RmsNorm,
    #[param]
    mlp: Lfm2MLP,
    is_conv: bool,
}

impl Lfm2DecoderLayer {
    pub fn new(config: &Lfm2Config, layer_idx: usize) -> Result<Self, Exception> {
        let is_conv = !config.is_attention_layer(layer_idx);
        let (conv, attn) = if is_conv {
            (Some(Lfm2ShortConv::new(config)?), None)
        } else {
            (None, Some(Lfm2Attention::new(config)?))
        };
        Ok(Self {
            conv, attn,
            operator_norm: nn::RmsNormBuilder::new(config.hidden_size).eps(config.norm_eps).build()?,
            ffn_norm: nn::RmsNormBuilder::new(config.hidden_size).eps(config.norm_eps).build()?,
            mlp: Lfm2MLP::new(config)?,
            is_conv,
        })
    }

    pub fn forward(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        layer_cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        let t = hidden.shape().get(1).copied().unwrap_or(1);

        let normed = self.operator_norm.forward(hidden)?;
        let op_out = if self.is_conv {
            let LayerCache::Arrays(ac) = layer_cache else {
                return Err(Exception::custom("lfm2: conv layer expects ArraysCache"));
            };
            if t == 1 { self.conv.as_mut().unwrap().forward_decode(&normed, ac)? }
            else { self.conv.as_mut().unwrap().forward(&normed, ac)? }
        } else {
            let LayerCache::KV(kv) = layer_cache else {
                return Err(Exception::custom("lfm2: attn layer expects KV cache"));
            };
            self.attn.as_mut().unwrap().forward(&normed, mask, kv)?
        };

        let h = hidden.add(&op_out)?;
        let normed = self.ffn_norm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        h.add(&mlp_out)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Lfm2Model {
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    layers: Vec<Lfm2DecoderLayer>,
    #[param]
    embedding_norm: nn::RmsNorm,
    config: Lfm2Config,
}

impl Lfm2Model {
    pub fn new(config: &Lfm2Config) -> Result<Self, Exception> {
        let layers = config.layer_types.iter().enumerate()
            .map(|(i, _)| Lfm2DecoderLayer::new(config, i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(config.vocab_size, config.hidden_size)?),
            layers,
            embedding_norm: nn::RmsNormBuilder::new(config.hidden_size).eps(config.norm_eps).build()?,
            config: config.clone(),
        })
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut [Option<LayerCache>],
    ) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(inputs)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let lm = if self.config.is_attention_layer(i) { mask } else { None };
            h = layer.forward(&h, lm, cache[i].as_mut().unwrap())?;
        }
        self.embedding_norm.forward(&h)
    }

    pub fn as_linear(&self, hidden: &Array) -> Result<Array, Exception> {
        match &self.embed_tokens {
            MaybeQuantized::Original(e) => e.as_linear(hidden),
            MaybeQuantized::Quantized(e) => e.as_linear(hidden),
        }
    }
}

// ---------------------------------------------------------------------------
// Causal LM
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Lfm2CausalLM {
    pub config: Lfm2Config,
    #[param]
    model: Lfm2Model,
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Lfm2CausalLM {
    pub fn new(config: &Lfm2Config) -> Result<Self, Exception> {
        let model = Lfm2Model::new(config)?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(MaybeQuantized::Original(nn::Linear::new(config.hidden_size, config.vocab_size)?))
        };
        Ok(Self { config: config.clone(), model, lm_head })
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.model.forward(inputs, mask, cache)?;
        let h_last = h.index((.., -1.., ..));
        match &mut self.lm_head {
            Some(head) => head.forward(&h_last),
            None => self.model.as_linear(&h_last),
        }
    }

    pub fn make_cache(&self) -> Vec<Option<LayerCache>> {
        self.model.layers.iter().enumerate().map(|(i, _)| {
            if self.config.is_attention_layer(i) {
                Some(LayerCache::KV(SteppingKeyValueCache::new()))
            } else {
                Some(LayerCache::Arrays(ArraysCache::new()))
            }
        }).collect()
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

pub fn load_lfm2_model(model_dir: &Path) -> Result<Lfm2CausalLM, ModelError> {
    let config = Lfm2Config::from_model_dir(model_dir)?;
    let mut model = Lfm2CausalLM::new(&config).map_err(ModelError::Mlx)?;
    let quantized = config.quantization.is_some();
    crate::load_quantized_safetensors_weights(&mut model, model_dir, quantized)?;
    Ok(model)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lfm2_config_parses_layer_types() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"),
            r#"{"model_type":"lfm2","hidden_size":64,"intermediate_size":128,
               "num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,
               "conv_L_cache":3,"vocab_size":100,"max_position_embeddings":512,
               "layer_types":["conv","conv","full_attention","conv"],
               "tie_word_embeddings":true,"rope_theta":1000000.0}"#).unwrap();
        let config = Lfm2Config::from_model_dir(dir.path()).unwrap();
        assert_eq!(config.num_hidden_layers, 4);
        assert!(!config.is_attention_layer(0));
        assert!(config.is_attention_layer(2));
        assert_eq!(config.head_dim(), 16);
    }

    #[test]
    fn lfm2_config_rejects_mismatched_layer_types() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"),
            r#"{"model_type":"lfm2","hidden_size":64,"intermediate_size":128,
               "num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,
               "vocab_size":100,"layer_types":["conv","conv"]}"#).unwrap();
        assert!(Lfm2Config::from_model_dir(dir.path()).is_err());
    }
}
