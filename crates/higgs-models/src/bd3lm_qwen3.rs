//! BD3LM-Qwen3: block-diffusion LM on top of Qwen3.
//! Runs a bidirectional mask over a 64-token block, iteratively denoising
//! via a custom `denoise_head` instead of the base LM head.

use std::{collections::HashMap, path::Path};

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    module::{Module, Param},
    nn, ops,
};
use serde::Deserialize;

use crate::{cache::KeyValueCache, error::ModelError, transformer};

const fn default_block_size() -> i32 {
    64
}
const fn default_num_steps() -> i32 {
    8
}

#[derive(Debug, Clone, Deserialize)]
pub struct Bd3lmConfig {
    #[serde(default = "default_block_size")]
    pub block_size: i32,
    #[serde(default = "default_num_steps")]
    pub num_denoising_steps: i32,
    pub denoise_hidden: i32,
}

pub struct DenoiseHead {
    pub norm: nn::LayerNorm, // denoise_head.0
    pub fc1: nn::Linear,     // denoise_head.1 : [hidden, hidden]
    pub fc2: nn::Linear,     // denoise_head.3 : [hidden, vocab]
}

impl DenoiseHead {
    pub fn new(hidden: i32, vocab: i32) -> Result<Self, Exception> {
        Ok(Self {
            norm: nn::LayerNormBuilder::new(hidden)
                .eps(1e-5)
                .affine(true)
                .build()?,
            fc1: nn::LinearBuilder::new(hidden, hidden).bias(true).build()?,
            fc2: nn::LinearBuilder::new(hidden, vocab).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, h: &Array) -> Result<Array, Exception> {
        let x = self.norm.forward(h)?;
        let x = self.fc1.forward(&x)?;
        let x = nn::gelu(&x)?;
        self.fc2.forward(&x)
    }
}

pub struct Bd3lmQwen3CausalLM {
    pub base: transformer::Model,
    pub mask_emb: Array, // [hidden]
    pub denoise_head: DenoiseHead,
    pub bd3lm_cfg: Bd3lmConfig,
}

impl Bd3lmQwen3CausalLM {
    pub const fn hidden_size(&self) -> i32 {
        self.base.args.hidden_size
    }

    pub const fn num_hidden_layers(&self) -> i32 {
        self.base.args.num_hidden_layers
    }

    pub const fn num_key_value_heads(&self) -> i32 {
        self.base.args.num_key_value_heads
    }

    pub fn checked_head_dim(&self) -> Result<i32, Exception> {
        self.base
            .args
            .checked_head_dim()
            .map_err(|e| Exception::custom(e.to_string()))
    }

    pub fn vocab_size(&self) -> i32 {
        self.base.args.vocab_size
    }

    /// Run BD3LM forward: inject mask_emb at masked positions, then transformer, then denoise_head.
    /// `masked_mask` is [B, T] bool. `attn_mask` is [T_q, T_k] bool (true = attend).
    /// Returns logits [B, T, vocab].
    pub fn forward_bd3lm<C: KeyValueCache>(
        &mut self,
        input_ids: &Array,
        masked_mask: &Array,
        attn_mask: Option<&Array>,
        cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        // 1. Normal token embedding
        let h = self.base.embed_tokens(input_ids)?; // [B, T, H]

        // 2. Broadcast mask_emb to [B, T, H] and replace where masked_mask==true
        let h_shape = h.shape().to_vec();
        let b = h_shape[0];
        let t = h_shape[1];
        let hidden = h_shape[2];

        let mask_bcast =
            ops::broadcast_to(self.mask_emb.reshape(&[1, 1, hidden])?, &[b, t, hidden])?;
        let mm_bcast = ops::broadcast_to(masked_mask.reshape(&[b, t, 1])?, &[b, t, hidden])?;
        let h = ops::r#where(&mm_bcast, &mask_bcast, &h)?;

        // 3. Run transformer stack
        let hidden_out = self
            .base
            .forward_hidden_from_embeddings(&h, attn_mask, cache)?;

        // 4. Denoise head → logits
        self.denoise_head.forward(&hidden_out)
    }
}

/// Load a BD3LM-Qwen3 model from a directory.
pub fn load_bd3lm_qwen3_model(model_dir: &Path) -> Result<Bd3lmQwen3CausalLM, ModelError> {
    // Read bd3lm_config.json
    let bd3lm_cfg_path = model_dir.join("bd3lm_config.json");
    let bd3lm_cfg: Bd3lmConfig = serde_json::from_reader(std::fs::File::open(&bd3lm_cfg_path)?)?;

    // Read main config, then override model_type so transformer::Model treats it as qwen3
    let config_path = model_dir.join("config.json");
    let mut args: transformer::ModelArgs =
        serde_json::from_reader(std::fs::File::open(&config_path)?)?;
    if args.model_type != "bd3lm_qwen3" {
        return Err(ModelError::UnsupportedModel(format!(
            "expected model_type=bd3lm_qwen3, got {}",
            args.model_type
        )));
    }
    args.model_type = "qwen3".to_owned();

    // Build and load base
    let quantization = args.quantization.clone();
    let hidden = args.hidden_size;
    let vocab = args.vocab_size;
    let raw = transformer::Model::new(args)?;
    let mut base = if let Some(ref qc) = quantization {
        mlx_rs::nn::quantize(raw, qc.group_size, qc.bits)
            .map_err(|e| ModelError::ShapeMismatch(format!("quantize failed: {e}")))?
    } else {
        raw
    };
    crate::load_quantized_safetensors_weights(&mut base, model_dir, quantization.is_some())?;

    // Build extras (mask_emb + denoise_head)
    let mut denoise_head =
        DenoiseHead::new(hidden, vocab).map_err(|e| ModelError::ShapeMismatch(e.to_string()))?;

    // Load bd3lm_extras.safetensors manually
    let extras_path = model_dir.join("bd3lm_extras.safetensors");
    let loaded: HashMap<String, Array> = Array::load_safetensors(&extras_path)
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?
        .into_iter()
        .collect();

    // mask_emb: accept either "mask_emb" or "mask_emb.weight"
    let mask_emb = loaded
        .get("mask_emb")
        .cloned()
        .or_else(|| loaded.get("mask_emb.weight").cloned())
        .ok_or_else(|| ModelError::MissingWeight("bd3lm_extras missing mask_emb".into()))?;

    // Probe which denoise_head index layout is used
    let norm_key_candidates = [
        ("denoise_head.0.weight", "denoise_head.0.bias"),
        ("denoise_head.norm.weight", "denoise_head.norm.bias"),
    ];
    let fc1_key_candidates = [
        ("denoise_head.1.weight", "denoise_head.1.bias"),
        ("denoise_head.fc1.weight", "denoise_head.fc1.bias"),
    ];
    let fc2_key_candidates = [
        ("denoise_head.3.weight", "denoise_head.3.bias"),
        ("denoise_head.2.weight", "denoise_head.2.bias"),
        ("denoise_head.fc2.weight", "denoise_head.fc2.bias"),
    ];

    let pick = |cands: &[(&str, &str)]| -> Result<(Array, Option<Array>), ModelError> {
        for (wk, bk) in cands {
            if let Some(w) = loaded.get(*wk) {
                let b = loaded.get(*bk).cloned();
                return Ok((w.clone(), b));
            }
        }
        Err(ModelError::MissingWeight(format!(
            "none of {cands:?} found in bd3lm_extras"
        )))
    };

    let (norm_w, norm_b) = pick(&norm_key_candidates)?;
    let (fc1_w, fc1_b) = pick(&fc1_key_candidates)?;
    let (fc2_w, fc2_b) = pick(&fc2_key_candidates)?;

    // LayerNorm weight/bias are both Param<Option<Array>>
    denoise_head.norm.weight = Param::new(Some(norm_w));
    denoise_head.norm.bias = Param::new(norm_b);
    // Linear weight is Param<Array>, bias is Param<Option<Array>>
    denoise_head.fc1.weight = Param::new(fc1_w);
    denoise_head.fc1.bias = Param::new(fc1_b);
    denoise_head.fc2.weight = Param::new(fc2_w);
    denoise_head.fc2.bias = Param::new(fc2_b);

    tracing::info!(
        block_size = bd3lm_cfg.block_size,
        num_steps = bd3lm_cfg.num_denoising_steps,
        "BD3LM-Qwen3 model loaded successfully"
    );

    Ok(Bd3lmQwen3CausalLM {
        base,
        mask_emb,
        denoise_head,
        bd3lm_cfg,
    })
}
