//! DFlash block-diffusion drafter for speculative decoding.
//!
//! A 0.5B drafter that produces 16 draft tokens per round via a single
//! non-causal forward pass. Conditions on hidden states tapped from 5
//! target model layers during the previous verify step.
//!
//! Architecture: 8 decoder layers with dual-stream attention —
//! Q from noise embedding, K/V from `concat(target_hidden, noise)`.
//! No `embed_tokens` or `lm_head` — uses the target model's `lm_head`.
//!
//! Reference: `dflash.py` in `z-lab/Qwen3.5-35B-A3B-DFlash`.
use std::path::Path;

use mlx_rs::{
    builder::Builder, error::Exception, macros::ModuleParameters, module::Module, nn, ops, Array,
};
use serde::Deserialize;

use crate::{error::ModelError, utils::apply_rope};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct DFlashSubConfig {
    target_layer_ids: Vec<usize>,
    #[serde(default)]
    mask_token_id: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DFlashConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    pub intermediate_size: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_block_size")]
    pub block_size: i32,
    pub vocab_size: i32,
    dflash_config: DFlashSubConfig,
}

impl DFlashConfig {
    pub fn target_layer_ids(&self) -> &[usize] {
        &self.dflash_config.target_layer_ids
    }

    pub fn num_taps(&self) -> usize {
        self.dflash_config.target_layer_ids.len()
    }

    pub fn mask_token_id(&self) -> i32 {
        self.dflash_config.mask_token_id.unwrap_or(248_070)
    }
}

const fn default_head_dim() -> i32 {
    128
}

const fn default_rms_norm_eps() -> f32 {
    1e-6
}

const fn default_rope_theta() -> f32 {
    1e7
}

const fn default_block_size() -> i32 {
    16
}

// ---------------------------------------------------------------------------
// SwiGLU MLP (non-quantized)
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
struct DFlashMLP {
    #[param]
    gate_proj: nn::Linear,
    #[param]
    up_proj: nn::Linear,
    #[param]
    down_proj: nn::Linear,
}

impl DFlashMLP {
    fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            up_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(false)
                .build()?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = nn::sigmoid(&gate)?.multiply(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// DFlash dual-stream attention
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
struct DFlashAttention {
    #[param]
    q_proj: nn::Linear,
    #[param]
    k_proj: nn::Linear,
    #[param]
    v_proj: nn::Linear,
    #[param]
    o_proj: nn::Linear,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl DFlashAttention {
    fn new(config: &DFlashConfig) -> Result<Self, Exception> {
        let head_dim = config.head_dim;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let hidden = config.hidden_size;

        Ok(Self {
            q_proj: nn::LinearBuilder::new(hidden, n_heads * head_dim)
                .bias(false)
                .build()?,
            k_proj: nn::LinearBuilder::new(hidden, n_kv_heads * head_dim)
                .bias(false)
                .build()?,
            v_proj: nn::LinearBuilder::new(hidden, n_kv_heads * head_dim)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(n_heads * head_dim, hidden)
                .bias(false)
                .build()?,
            q_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(config.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(config.rms_norm_eps)
                .build()?,
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(false)
                .base(config.rope_theta)
                .scale(1.0)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv_heads,
            head_dim,
            scale: f32::from(
                i16::try_from(head_dim)
                    .map_err(|_| Exception::custom("head_dim out of i16 range"))?,
            )
            .sqrt()
            .recip(),
        })
    }

    /// Dual-stream attention: Q from noise, K/V from `concat(target, noise)`.
    ///
    /// `noise`: `[B, block_size, hidden]` — the 16 draft positions.
    /// `target_hidden`: `[B, ctx_len, hidden]` — projected+normed tap states.
    /// `cache`: optional (K, V) from prior rounds, shape `[B, n_kv, cached_len, head_dim]`.
    ///   Post-RoPE K and raw V. Updated in-place with the new K/V appended.
    /// `cache_offset`: absolute position offset for RoPE (= cached seq length).
    #[allow(non_snake_case, clippy::shadow_reuse)]
    fn forward(
        &mut self,
        noise: &Array,
        target_hidden: &Array,
        cache: &mut Option<(Array, Array)>,
        cache_offset: i32,
    ) -> Result<Array, Exception> {
        let B = *noise
            .shape()
            .first()
            .ok_or_else(|| Exception::custom("need 3D"))?;
        let q_len = *noise
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("need 3D"))?;
        let ctx_len = *target_hidden
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("need 3D"))?;
        // Q from noise only
        let q = self.q_proj.forward(noise)?;
        let q = q.reshape(&[B, q_len, self.num_attention_heads, self.head_dim])?;
        let q = self.q_norm.forward(&q)?.transpose_axes(&[0, 2, 1, 3])?;

        // K/V from context (target_hidden) — SEPARATE from noise
        let ctx_k = self.k_proj.forward(target_hidden)?;
        let ctx_v = self.v_proj.forward(target_hidden)?;
        let ctx_k = ctx_k.reshape(&[B, ctx_len, self.num_key_value_heads, self.head_dim])?;
        let ctx_k = self.k_norm.forward(&ctx_k)?.transpose_axes(&[0, 2, 1, 3])?;
        let ctx_v = ctx_v
            .reshape(&[B, ctx_len, self.num_key_value_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // K/V from noise — freshly computed every round, never cached
        let noise_k = self.k_proj.forward(noise)?;
        let noise_v = self.v_proj.forward(noise)?;
        let noise_k = noise_k.reshape(&[B, q_len, self.num_key_value_heads, self.head_dim])?;
        let noise_k = self.k_norm.forward(&noise_k)?.transpose_axes(&[0, 2, 1, 3])?;
        let noise_v = noise_v
            .reshape(&[B, q_len, self.num_key_value_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE with absolute positions:
        // Context K: [cache_offset .. cache_offset + ctx_len]
        // Noise K + Q: [cache_offset + ctx_len .. cache_offset + ctx_len + q_len]
        let q = apply_rope(&q, &self.rope, cache_offset + ctx_len)?;
        let ctx_k = apply_rope(&ctx_k, &self.rope, cache_offset)?;
        let noise_k = apply_rope(&noise_k, &self.rope, cache_offset + ctx_len)?;

        // Cache stores ONLY context K/V (append to prior rounds)
        let (ctx_k, ctx_v) = if let Some((k_cached, v_cached)) = cache.as_ref() {
            (
                ops::concatenate_axis(&[k_cached, &ctx_k], 2)?,
                ops::concatenate_axis(&[v_cached, &ctx_v], 2)?,
            )
        } else {
            (ctx_k, ctx_v)
        };
        *cache = Some((ctx_k.clone(), ctx_v.clone()));

        // Attention over cached_context + fresh_noise
        let k = ops::concatenate_axis(&[&ctx_k, &noise_k], 2)?;
        let v = ops::concatenate_axis(&[&ctx_v, &noise_v], 2)?;

        // Non-causal SDPA (no mask)
        let output = mlx_rs::fast::scaled_dot_product_attention(
            q,
            k,
            v,
            self.scale,
            None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            None::<&Array>,
        )?;

        // [B, n_heads, q_len, head_dim] -> [B, q_len, n_heads * head_dim]
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[B, q_len, -1])?;
        self.o_proj.forward(&output)
    }
}

// ---------------------------------------------------------------------------
// DFlash decoder layer
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
struct DFlashDecoderLayer {
    #[param]
    self_attn: DFlashAttention,
    #[param]
    mlp: DFlashMLP,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl DFlashDecoderLayer {
    fn new(config: &DFlashConfig) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: DFlashAttention::new(config)?,
            mlp: DFlashMLP::new(config.hidden_size, config.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    fn forward(
        &mut self,
        noise: &Array,
        target_hidden: &Array,
        cache: &mut Option<(Array, Array)>,
        cache_offset: i32,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(noise)?;
        let attn_out = self
            .self_attn
            .forward(&normed, target_hidden, cache, cache_offset)?;
        let h = noise.add(attn_out)?;
        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }
}

// ---------------------------------------------------------------------------
// DFlash drafter (top-level)
// ---------------------------------------------------------------------------

/// DFlash block-diffusion drafter.
///
/// Produces `block_size` (16) draft tokens per round. Does NOT have its own
/// embed_tokens or lm_head — uses the target model's lm_head on the output.
#[derive(Debug, ModuleParameters)]
pub struct DFlashDrafter {
    #[param]
    fc: nn::Linear,
    #[param]
    hidden_norm: nn::RmsNorm,
    #[param]
    layers: Vec<DFlashDecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
    pub config: DFlashConfig,
}

impl DFlashDrafter {
    pub fn new(config: DFlashConfig) -> Result<Self, Exception> {
        #[allow(clippy::as_conversions)]
        let fc_in = config.num_taps() as i32 * config.hidden_size;
        let layers = (0..config.num_hidden_layers)
            .map(|_| DFlashDecoderLayer::new(&config))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            fc: nn::LinearBuilder::new(fc_in, config.hidden_size)
                .bias(false)
                .build()?,
            hidden_norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            config,
        })
    }

    /// Create an empty per-layer KV cache for the drafter.
    pub fn make_cache(&self) -> Vec<Option<(Array, Array)>> {
        vec![None; self.layers.len()]
    }

    /// Run the drafter forward pass.
    ///
    /// - `noise`: `[B, block_size, hidden_size]` — embedded block tokens.
    /// - `taps`: slice of hidden states from the target model at tap layers,
    ///   each `[B, T, target_hidden_size]`. Concatenated along the last dim,
    ///   projected via `fc`, then normalized.
    /// - `cache`: per-layer KV cache. Grows each round; crop after verify.
    ///
    /// Returns `[B, block_size, hidden_size]` — pass to target's `lm_head` for logits.
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        noise: &Array,
        taps: &[Array],
        cache: &mut [Option<(Array, Array)>],
    ) -> Result<Array, Exception> {
        if taps.len() != self.config.num_taps() {
            return Err(Exception::custom(format!(
                "expected {} taps, got {}",
                self.config.num_taps(),
                taps.len()
            )));
        }

        // Cache offset = current cached seq length (0 on first round)
        let cache_offset = cache
            .first()
            .and_then(|c| c.as_ref())
            .map_or(0, |(k, _)| k.shape()[2] as i32);

        // Concatenate tap hidden states: [B, T, num_taps * hidden_size]
        let tap_refs: Vec<&Array> = taps.iter().collect();
        let target_cat = ops::concatenate_axis(&tap_refs, -1)?;

        // Project + norm: [B, T, hidden_size]
        let target_hidden = self.fc.forward(&target_cat)?;
        let target_hidden = self.hidden_norm.forward(&target_hidden)?;

        let mut h = noise.clone();
        for (layer, lc) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(&h, &target_hidden, lc, cache_offset)?;
        }

        self.norm.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// GDN state save/restore for hybrid models (Qwen3.5)
// ---------------------------------------------------------------------------

/// Saved state for all GDN/linear-attention layers in the target model.
/// Much smaller than cloning the full KV cache — only stores conv_state,
/// ssm_state, and offset for each ArraysCache layer.
pub struct GdnStateBackup {
    states: Vec<(Option<Array>, Option<Array>, i32)>,
}

impl GdnStateBackup {
    /// Save GDN (ArraysCache) state from all layers. Call BEFORE verify forward.
    /// KV layers are not saved — they use cheap offset-based rollback instead.
    pub fn save(kv_cache: &[Option<crate::qwen3_next::LayerCache>]) -> Result<Self, Exception> {
        let mut states = Vec::with_capacity(kv_cache.len());
        for lc in kv_cache.iter() {
            match lc {
                Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                    ac.eval_arrays()?;
                    states.push((ac.conv_state.clone(), ac.ssm_state.clone(), ac.offset));
                }
                _ => states.push((None, None, 0)),
            }
        }
        Ok(Self { states })
    }

    /// Restore GDN state and rollback KV offsets. On rejection, call this
    /// BEFORE re-running the accepted tokens.
    pub fn restore_and_rollback(
        &self,
        kv_cache: &mut [Option<crate::qwen3_next::LayerCache>],
        rollback: i32,
    ) {
        for (lc, (conv, ssm, offset)) in kv_cache.iter_mut().zip(self.states.iter()) {
            match lc {
                Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                    ac.conv_state = conv.clone();
                    ac.ssm_state = ssm.clone();
                    ac.offset = *offset;
                }
                Some(crate::qwen3_next::LayerCache::KV(kv)) => {
                    if rollback > 0 {
                        kv.rollback(rollback);
                    }
                }
                _ => {}
            }
        }
    }
}

/// Rollback only KV cache layers by `rollback` positions. GDN layers are
/// left untouched. Used with stateless GDN verify where GDN state was never
/// corrupted by speculative tokens.
pub fn rollback_kv_only(
    kv_cache: &mut [Option<crate::qwen3_next::LayerCache>],
    rollback: i32,
) {
    if rollback <= 0 {
        return;
    }
    for lc in kv_cache.iter_mut() {
        if let Some(crate::qwen3_next::LayerCache::KV(kv)) = lc {
            kv.rollback(rollback);
        }
    }
}

/// Crop the drafter KV cache to `keep_len` along the sequence dim.
/// Called after verify to discard rejected positions.
/// Cache tensors have shape `[B, n_kv_heads, seq_len, head_dim]`.
pub fn crop_drafter_cache(cache: &mut [Option<(Array, Array)>], keep_len: i32) {
    use mlx_rs::ops::indexing::IndexOp;
    for entry in cache.iter_mut() {
        if let Some((k, v)) = entry {
            *k = k.index((.., .., ..keep_len, ..));
            *v = v.index((.., .., ..keep_len, ..));
        }
    }
}

/// Trim `n` entries from the END of the drafter KV cache.
///
/// Reference: `trim_draft_cache(draft_cache, block_size)` in dflash-mlx.
/// After each draft forward, the cache has `prev + ctx_len + block_size` entries.
/// Trimming `block_size` removes the noise K/V while keeping the accumulated
/// target context K/V that conditions future rounds.
pub fn trim_drafter_cache(cache: &mut [Option<(Array, Array)>], n: i32) {
    use mlx_rs::ops::indexing::IndexOp;
    for entry in cache.iter_mut() {
        if let Some((k, v)) = entry {
            let seq_len = k.shape()[2] as i32;
            let keep = (seq_len - n).max(0);
            *k = k.index((.., .., ..keep, ..));
            *v = v.index((.., .., ..keep, ..));
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load a DFlash drafter from a directory containing config.json + model.safetensors.
pub fn load_dflash_drafter(model_path: &Path) -> Result<DFlashDrafter, ModelError> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| ModelError::Io(std::io::Error::other(format!("reading config.json: {e}"))))?;
    let config: DFlashConfig = serde_json::from_str(&config_str)
        .map_err(|e| ModelError::Io(std::io::Error::other(format!("parsing config.json: {e}"))))?;

    let mut drafter = DFlashDrafter::new(config)
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

    crate::load_safetensors_weights(&mut drafter, model_path)?;

    Ok(drafter)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;

    fn test_config() -> DFlashConfig {
        DFlashConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1e7,
            block_size: 16,
            vocab_size: 100,
            dflash_config: DFlashSubConfig {
                target_layer_ids: vec![0, 1],
                mask_token_id: None,
            },
        }
    }

    #[test]
    fn test_dflash_drafter_construction() {
        let config = test_config();
        let drafter = DFlashDrafter::new(config).unwrap();
        assert_eq!(drafter.layers.len(), 2);
    }

    #[test]
    fn test_dflash_forward_shape() {
        let config = test_config();
        let mut drafter = DFlashDrafter::new(config).unwrap();

        let noise = Array::zeros::<f32>(&[1, 4, 64]).unwrap();
        let tap0 = Array::zeros::<f32>(&[1, 8, 64]).unwrap();
        let tap1 = Array::zeros::<f32>(&[1, 8, 64]).unwrap();

        let mut cache = drafter.make_cache();
        let out = drafter.forward(&noise, &[tap0, tap1], &mut cache).unwrap();
        assert_eq!(out.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_dflash_config_parse() {
        let json = r#"{
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 6144,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000,
            "block_size": 16,
            "vocab_size": 248320,
            "dflash_config": {
                "mask_token_id": 248070,
                "target_layer_ids": [1, 10, 19, 28, 37]
            }
        }"#;
        let config: DFlashConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_taps(), 5);
        assert_eq!(config.target_layer_ids(), &[1, 10, 19, 28, 37]);
        assert_eq!(config.block_size, 16);
        assert_eq!(config.mask_token_id(), 248_070);
    }

    #[test]
    #[ignore] // requires model weights
    fn test_dflash_load_drafter() {
        let drafter_path = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash",
        );
        let snapshots = std::fs::read_dir(drafter_path.join("snapshots")).unwrap();
        let snap_dir = snapshots.filter_map(|e| e.ok()).next().unwrap().path();

        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();
        assert_eq!(drafter.layers.len(), 8);
        assert_eq!(drafter.config.hidden_size, 2048);
        assert_eq!(drafter.config.num_taps(), 5);

        // Smoke test forward with random inputs
        let noise = Array::zeros::<f32>(&[1, 16, 2048])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();
        let tap = Array::zeros::<f32>(&[1, 10, 2048])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();
        let taps: Vec<Array> = (0..5).map(|_| tap.clone()).collect();

        let mut cache = drafter.make_cache();
        let out = drafter.forward(&noise, &taps, &mut cache).unwrap();
        assert_eq!(out.shape(), &[1, 16, 2048]);
        println!("DFlash drafter loaded and forward OK: {:?}", out.shape());
    }

    #[test]
    #[ignore] // requires model weights — full draft→verify loop
    fn test_dflash_35b_a3b() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        // ---- paths ----
        let target_path = "/Users/peppi/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit";
        let drafter_base = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash",
        );
        let snap_dir = std::fs::read_dir(drafter_base.join("snapshots"))
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .unwrap()
            .path();

        // ---- load target model ----
        println!("Loading target model...");
        let t0 = Instant::now();
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Target loaded in {:.1}s", t0.elapsed().as_secs_f64());

        // ---- load drafter ----
        println!("Loading DFlash drafter...");
        let t0 = Instant::now();
        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();
        println!("Drafter loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let configured_block_size = drafter.config.block_size;
        let block_size = std::env::var("HIGGS_DFLASH_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|&n| n >= 2 && n <= configured_block_size)
            .unwrap_or(configured_block_size);
        let mask_id = drafter.config.mask_token_id();

        // ---- tokenize prompt ----
        // Chat-templated "Write a short paragraph about the history of computers."
        // with thinking disabled (no /think block content)
        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let eos_token: i32 = 248046; // <|im_end|>
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        // ---- prefill with full-sequence taps ----
        println!("Prefilling...");
        let t0 = Instant::now();
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        // Eval everything immediately: logits + taps + all cache states
        let mut eval_targets: Vec<&Array> = vec![&prefill_logits];
        for t in &taps {
            eval_targets.push(t);
        }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        eval_targets.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_targets.push(c);
                    }
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();
        let prefill_ms = t0.elapsed().as_millis();
        println!(
            "Prefill: {prefill_ms}ms, taps: {} (each shape: {:?})",
            taps.len(),
            taps.first().map(|t| t.shape().to_vec())
        );

        // Last-position argmax from prefill logits [1, seq, vocab]
        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am
            .reshape(&[-1])
            .unwrap()
            .as_slice::<u32>()
            .to_vec();
        let mut last_token: i32 = *am_flat.last().unwrap() as i32;

        // ---- generation loop ----
        let max_rounds = 20;
        let mut total_tokens = 0usize;
        let mut total_accepted = 0usize;
        let mut total_draft_ms = 0u128;
        let mut total_verify_ms = 0u128;
        let mut generated: Vec<i32> = vec![last_token];
        let mut current_taps = taps;
        let mut draft_cache = drafter.make_cache();
        let mut start = prompt_len; // absolute position in output sequence

        println!(
            "\n--- DFlash generation (mask_id={mask_id}, effective_block_size={block_size}) ---"
        );
        for round in 0..max_rounds {
            // a. Build block_ids = [anchor, mask, mask, ..., mask] (block_size tokens)
            //    Reference: output_ids[:, start:start+block_size] where pos 0 is the
            //    last accepted token and the rest are pre-filled with mask_token_id.
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);

            // b. Embed block through target's embedding layer (NOT random noise)
            //    Reference: noise_embedding = target.model.embed_tokens(block_output_ids)
            let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

            // c. Drafter forward
            let t0 = Instant::now();
            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps, &mut draft_cache)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            let draft_ms = t0.elapsed().as_millis();
            total_draft_ms += draft_ms;

            if std::env::var("DFLASH_NO_DRAFT_CACHE").is_ok() {
                draft_cache = drafter.make_cache();
            } else {
                crop_drafter_cache(&mut draft_cache, start);
            }

            let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));

            if round == 0 {
                let tap0 = &current_taps[0];
                println!(
                    "  taps: ctx_len={}, tap_shape={:?}, tap_dtype={:?}",
                    tap0.shape()[1],
                    tap0.shape(),
                    tap0.dtype()
                );
                println!(
                    "  noise_emb: shape={:?}, dtype={:?}",
                    noise_embedding.shape(),
                    noise_embedding.dtype()
                );
                println!(
                    "  draft_hidden: shape={:?}, dtype={:?}",
                    draft_hidden.shape(),
                    draft_hidden.dtype()
                );
                let cache_len = draft_cache
                    .first()
                    .and_then(|c| c.as_ref())
                    .map_or(0, |(k, _)| k.shape()[2]);
                println!(
                    "  draft_cache_len={cache_len} (should be ctx_len+block_size={})",
                    tap0.shape()[1] as i32 + block_size
                );
            }

            // e. Target lm_head on sliced hidden → logits → argmax → draft tokens
            let draft_logits = target
                .forward_all_logits_from_hidden(&draft_hidden_sliced)
                .unwrap();
            let draft_token_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_token_ids]).unwrap();

            let draft_u32: Vec<u32> = draft_token_ids
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            // f. Build verify input: [anchor, draft_0, ..., draft_14] = block_size tokens
            //    Reference: target forward on block_output_ids (which now has drafts at 1:)
            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_len = verify_tokens.len() as i32; // = block_size
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            // g. Verify: target forward_with_taps on verify sequence.
            //    Reference: past_key_values_target.crop(start) only crops KV for
            //    attention layers. GDN/linear attention layers' crop() is a no-op
            //    in HuggingFace — the SSM state stays advanced by full block_size.
            //    So we only need to rollback the KV cache, NOT the GDN state.
            // h. Save GDN state before verify (for rollback if tokens rejected)
            let gdn_backup = GdnStateBackup::save(&kv_cache).unwrap();

            let t0 = Instant::now();
            let (verify_logits, verify_taps) = target
                .forward_with_taps(&verify_input, None, &mut kv_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();
            let verify_ms = t0.elapsed().as_millis();
            total_verify_ms += verify_ms;

            // i. Argmax verify logits and accept prefix
            let verify_argmax_arr = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_argmax_arr
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();

            // Reference: acceptance_length = (block_output_ids[:,1:] == posterior[:,:-1]).cumprod().sum()
            // posterior = argmax(verify_logits) at each position.
            // verify_flat[i] = target's greedy token at position i of the verify input.
            // draft_u32[i] = our draft at position i+1 of the block.
            // Match: draft_u32[i] == verify_flat[i] (target's prediction for position i).
            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len(); // number of matching draft tokens

            // The correction token: target's greedy pick at the rejection point.
            // Reference line 373: output_ids[start + acceptance_length + 1] = posterior[acceptance_length]
            // This is verify_flat[n_accepted] — one past the last match.
            let correction_token = verify_flat[n_accepted] as i32;

            // Total tokens this round = n_accepted (matched drafts) + 1 (correction)
            let tokens_this_round = n_accepted + 1;

            if round == 0 {
                println!("  draft_u32:  {:?}", &draft_u32[..draft_u32.len().min(15)]);
                println!(
                    "  verify_flat: {:?}",
                    &verify_flat[..verify_flat.len().min(16)]
                );
                println!("  accepted:    {:?} + correction={}", accepted, correction_token);
                println!(
                    "  n_accepted:  {n_accepted} + 1 correction (of {} drafts)",
                    draft_u32.len()
                );
            }

            // Rollback KV cache AND GDN state for rejected tokens.
            // We keep n_accepted + 1 positions (accepted + correction).
            // Reference: past_key_values_target.crop(start + acceptance_length + 1)
            {
                let keep = tokens_this_round as i32; // = n_accepted + 1
                let rollback = verify_len - keep;
                if rollback > 0 {
                    GdnStateBackup::restore_and_rollback(&gdn_backup, &mut kv_cache, rollback);
                }
                // Reference line 376: target_hidden[:, :acceptance_length + 1, :]
                current_taps = verify_taps
                    .into_iter()
                    .map(|tap| tap.index((.., ..keep, ..)))
                    .collect();
            }

            total_accepted += tokens_this_round;
            total_tokens += tokens_this_round;

            for &tok in &accepted {
                generated.push(tok as i32);
            }
            generated.push(correction_token);
            // Reference: next anchor = correction token (target's greedy at rejection point)
            last_token = correction_token;

            start += tokens_this_round as i32;

            println!(
                "Round {round}: draft={draft_ms}ms verify={verify_ms}ms accepted={n_accepted}+1/{} draft={draft_flat:?}",
                block_size - 1
            );

            // Stop on EOS
            if generated.contains(&eos_token) {
                println!("EOS detected, stopping.");
                break;
            }
        }

        let total_ms = total_draft_ms + total_verify_ms;
        let tok_per_sec = if total_ms > 0 {
            total_tokens as f64 / (total_ms as f64 / 1000.0)
        } else {
            0.0
        };

        println!("\n--- Results ---");
        println!("Total tokens: {total_tokens}");
        println!("Total rounds: {max_rounds}");
        println!(
            "Avg acceptance: {:.1} tok/round",
            total_accepted as f64 / max_rounds as f64
        );
        println!("Total draft time: {total_draft_ms}ms");
        println!("Total verify time: {total_verify_ms}ms");
        println!("Throughput: {tok_per_sec:.1} tok/s");
        println!(
            "Generated tokens: {:?}",
            &generated[..generated.len().min(50)]
        );
    }

    #[test]
    #[ignore] // requires BOTH 4B-4bit and 4B-bf16 targets + drafter
    fn test_dflash_4b_mixed_precision_taps() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let bf16_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-bf16";
        let q4_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit";
        let drafter_base = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        );
        let snap_dir = std::fs::read_dir(drafter_base.join("snapshots"))
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .unwrap()
            .path();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let eos_token: i32 = 248046;
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        // --- Phase 1: Load BF16 model, extract taps, then DROP it ---
        println!("=== Phase 1: BF16 taps extraction ===");
        let bf16_taps;
        let bf16_last_token;
        {
            println!("Loading 4B BF16 target...");
            let t0 = Instant::now();
            let mut bf16_target = load_qwen3_5_model(bf16_path).unwrap();
            println!("BF16 target loaded in {:.1}s", t0.elapsed().as_secs_f64());

            let mut bf16_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
            let tap_layers = vec![1usize, 8, 15, 22, 29];
            let (logits, taps) = bf16_target
                .forward_with_taps(&input_ids, None, &mut bf16_cache, &tap_layers)
                .unwrap();
            let mut eval_t: Vec<&Array> = vec![&logits];
            for t in &taps {
                eval_t.push(t);
            }
            mlx_rs::transforms::eval(eval_t).unwrap();

            let am = mlx_rs::argmax_axis!(logits, -1).unwrap();
            let flat: Vec<u32> = am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            bf16_last_token = *flat.last().unwrap() as i32;
            bf16_taps = taps;
            println!(
                "BF16 taps: {} taps, shape={:?}, dtype={:?}, last_token={}",
                bf16_taps.len(),
                bf16_taps[0].shape(),
                bf16_taps[0].dtype(),
                bf16_last_token
            );
            // bf16_target + bf16_cache dropped here, freeing ~8GB
        }

        // --- Phase 2: Load 4-bit model, extract taps ---
        println!("\n=== Phase 2: 4-bit taps extraction ===");
        println!("Loading 4B 4-bit target...");
        let t0 = Instant::now();
        let mut q4_target = load_qwen3_5_model(q4_path).unwrap();
        println!("4-bit target loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let tap_layers = vec![1usize, 8, 15, 22, 29];
        let mut q4_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (q4_logits, q4_taps) = q4_target
            .forward_with_taps(&input_ids, None, &mut q4_cache, &tap_layers)
            .unwrap();
        let mut eval_t: Vec<&Array> = vec![&q4_logits];
        for t in &q4_taps {
            eval_t.push(t);
        }
        for lc in q4_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_t.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        eval_t.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_t.push(c);
                    }
                }
            }
        }
        mlx_rs::transforms::eval(eval_t).unwrap();

        let q4_am = mlx_rs::argmax_axis!(q4_logits, -1).unwrap();
        let q4_flat: Vec<u32> = q4_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
        let q4_last_token = *q4_flat.last().unwrap() as i32;
        println!(
            "4-bit taps: {} taps, shape={:?}, dtype={:?}, last_token={}",
            q4_taps.len(),
            q4_taps[0].shape(),
            q4_taps[0].dtype(),
            q4_last_token
        );

        // --- Phase 3: Cosine similarity between taps ---
        println!("\n=== Phase 3: Tap cosine similarity (BF16 vs 4-bit) ===");
        for (i, (bf, q4)) in bf16_taps.iter().zip(q4_taps.iter()).enumerate() {
            let bf_flat = bf.reshape(&[-1]).unwrap().as_dtype(mlx_rs::Dtype::Float32).unwrap();
            let q4_flat = q4.reshape(&[-1]).unwrap().as_dtype(mlx_rs::Dtype::Float32).unwrap();
            let dot = ops::sum(&bf_flat.multiply(&q4_flat).unwrap(), None).unwrap();
            let norm_bf = ops::sum(&bf_flat.multiply(&bf_flat).unwrap(), None)
                .unwrap()
                .sqrt()
                .unwrap();
            let norm_q4 = ops::sum(&q4_flat.multiply(&q4_flat).unwrap(), None)
                .unwrap()
                .sqrt()
                .unwrap();
            let cos = dot.divide(&norm_bf.multiply(&norm_q4).unwrap()).unwrap();
            mlx_rs::transforms::eval([&cos]).unwrap();
            let cos_val: f32 = cos.item();
            println!(
                "  Tap {} (layer {}): cosine={:.6}",
                i,
                tap_layers[i],
                cos_val
            );
        }

        // --- Phase 4: DFlash with BF16 taps ---
        println!("\n=== Phase 4: DFlash with BF16 taps (draft quality test) ===");
        println!("Loading DFlash drafter...");
        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        // Use BF16 taps but 4-bit model's embed_tokens and lm_head
        let mut current_taps_bf16 = bf16_taps;
        let mut last_token_bf16 = bf16_last_token;
        let mut draft_cache_bf16 = drafter.make_cache();
        let mut start_bf16 = prompt_len;
        let mut total_accepted_bf16 = 0usize;
        let max_rounds = 10;

        for round in 0..max_rounds {
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token_bf16;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = q4_target.embed_token_ids(&block_ids).unwrap();

            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps_bf16, &mut draft_cache_bf16)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            crop_drafter_cache(&mut draft_cache_bf16, start_bf16);

            let draft_logits = q4_target
                .forward_all_logits_from_hidden(&draft_hidden.index((.., 1.., ..)))
                .unwrap();
            let draft_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_ids]).unwrap();
            let draft_u32: Vec<u32> = draft_ids.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            let mut verify_tokens = vec![last_token_bf16];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_tokens.len() as i32]);

            let gdn_backup = GdnStateBackup::save(&q4_cache).unwrap();
            let (verify_logits, verify_taps) = q4_target
                .forward_with_taps(&verify_input, None, &mut q4_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();

            let verify_am = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();

            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len();
            let correction = verify_flat[n_accepted] as i32;
            let tokens_this_round = n_accepted + 1;

            let keep = tokens_this_round as i32;
            let rollback = verify_tokens.len() as i32 - keep;
            if rollback > 0 {
                GdnStateBackup::restore_and_rollback(&gdn_backup, &mut q4_cache, rollback);
            }
            // BF16 taps for next round: use VERIFY taps (from 4-bit model)
            // because we need the target's hidden states at accepted positions.
            // But we could also re-extract from BF16 model for pure test...
            // For now: use 4-bit verify taps (tests if BF16 PREFILL taps help round 0)
            current_taps_bf16 = verify_taps
                .into_iter()
                .map(|tap| tap.index((.., ..keep, ..)))
                .collect();

            total_accepted_bf16 += tokens_this_round;
            last_token_bf16 = correction;
            start_bf16 += tokens_this_round as i32;

            println!(
                "  BF16-taps Round {round}: accepted={n_accepted}+1/{} draft={draft_flat:?}",
                block_size - 1
            );

            if verify_tokens.contains(&(eos_token as i32)) {
                break;
            }
        }
        let bf16_avg = total_accepted_bf16 as f64 / max_rounds as f64;

        // --- Phase 5: DFlash with 4-bit taps (control) ---
        println!("\n=== Phase 5: DFlash with 4-bit taps (control) ===");
        // Reset: reload 4-bit cache from scratch
        let mut q4_cache2: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (q4_logits2, q4_taps2) = q4_target
            .forward_with_taps(&input_ids, None, &mut q4_cache2, &tap_layers)
            .unwrap();
        {
            let mut ev: Vec<&Array> = vec![&q4_logits2];
            for t in &q4_taps2 { ev.push(t); }
            for lc in q4_cache2.iter().flatten() {
                match lc {
                    crate::qwen3_next::LayerCache::KV(kv) => ev.extend(kv.eval_targets()),
                    crate::qwen3_next::LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state { ev.push(s); }
                        if let Some(ref c) = ac.conv_state { ev.push(c); }
                    }
                }
            }
            mlx_rs::transforms::eval(ev).unwrap();
        }

        let mut current_taps_q4 = q4_taps2;
        let mut last_token_q4 = q4_last_token;
        let mut drafter2 = load_dflash_drafter(&snap_dir).unwrap();
        let mut draft_cache_q4 = drafter2.make_cache();
        let mut start_q4 = prompt_len;
        let mut total_accepted_q4 = 0usize;

        for round in 0..max_rounds {
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token_q4;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = q4_target.embed_token_ids(&block_ids).unwrap();

            let draft_hidden = drafter2
                .forward(&noise_embedding, &current_taps_q4, &mut draft_cache_q4)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            crop_drafter_cache(&mut draft_cache_q4, start_q4);

            let draft_logits = q4_target
                .forward_all_logits_from_hidden(&draft_hidden.index((.., 1.., ..)))
                .unwrap();
            let draft_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_ids]).unwrap();
            let draft_u32: Vec<u32> = draft_ids.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            let mut verify_tokens = vec![last_token_q4];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_tokens.len() as i32]);

            let gdn_backup = GdnStateBackup::save(&q4_cache2).unwrap();
            let (verify_logits, verify_taps) = q4_target
                .forward_with_taps(&verify_input, None, &mut q4_cache2, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();

            let verify_am = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();

            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len();
            let correction = verify_flat[n_accepted] as i32;
            let tokens_this_round = n_accepted + 1;

            let keep = tokens_this_round as i32;
            let rollback = verify_tokens.len() as i32 - keep;
            if rollback > 0 {
                GdnStateBackup::restore_and_rollback(&gdn_backup, &mut q4_cache2, rollback);
            }
            current_taps_q4 = verify_taps
                .into_iter()
                .map(|tap| tap.index((.., ..keep, ..)))
                .collect();

            total_accepted_q4 += tokens_this_round;
            last_token_q4 = correction;
            start_q4 += tokens_this_round as i32;

            println!(
                "  4bit-taps Round {round}: accepted={n_accepted}+1/{} draft={draft_flat:?}",
                block_size - 1
            );

            if verify_tokens.contains(&(eos_token as i32)) {
                break;
            }
        }
        let q4_avg = total_accepted_q4 as f64 / max_rounds as f64;

        // --- Summary ---
        println!("\n=== RESULTS ===");
        println!("BF16 taps avg acceptance: {bf16_avg:.1} tok/round");
        println!("4-bit taps avg acceptance: {q4_avg:.1} tok/round");
        println!("Improvement: {:.1}x", bf16_avg / q4_avg);
        if bf16_avg > q4_avg * 1.5 {
            println!(">>> QUANTIZATION IS THE BOTTLENECK — mixed-precision taps would help");
        } else {
            println!(">>> Quantization is NOT the main issue — look for bugs in drafter forward");
        }
    }

    #[test]
    #[ignore] // requires 4B target + drafter model weights on disk
    fn test_dflash_4b() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path =
            "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit";
        let drafter_base = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        );
        let snap_dir = std::fs::read_dir(drafter_base.join("snapshots"))
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .unwrap()
            .path();

        println!("Loading 4B target (bf16)...");
        let t0 = Instant::now();
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Target loaded in {:.1}s", t0.elapsed().as_secs_f64());

        println!("Loading 4B DFlash drafter...");
        let t0 = Instant::now();
        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();
        println!("Drafter loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        println!(
            "Config: hidden_size={} taps={:?} block_size={} mask_id={}",
            drafter.config.hidden_size, tap_layers, block_size, mask_id
        );
        println!(
            "Drafter: {} layers, {} heads, {} kv_heads",
            drafter.config.num_hidden_layers,
            drafter.config.num_attention_heads,
            drafter.config.num_key_value_heads
        );

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let eos_token: i32 = 248046;
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        println!("Prefilling 4B target...");
        let t0 = Instant::now();
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        let mut eval_targets: Vec<&Array> = vec![&prefill_logits];
        for t in &taps {
            eval_targets.push(t);
        }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        eval_targets.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_targets.push(c);
                    }
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();
        let prefill_ms = t0.elapsed().as_millis();
        println!(
            "Prefill: {prefill_ms}ms, taps: {} (each shape: {:?})",
            taps.len(),
            taps.first().map(|t| t.shape().to_vec())
        );

        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am
            .reshape(&[-1])
            .unwrap()
            .as_slice::<u32>()
            .to_vec();
        let mut last_token: i32 = *am_flat.last().unwrap() as i32;

        let max_rounds = 20;
        let mut total_tokens = 0usize;
        let mut total_accepted = 0usize;
        let mut total_draft_ms = 0u128;
        let mut total_verify_ms = 0u128;
        let mut generated: Vec<i32> = vec![last_token];
        let mut current_taps = taps;
        let mut draft_cache = drafter.make_cache();
        let mut start = prompt_len;

        println!("\n--- 4B DFlash generation (mask_id={mask_id}, block_size={block_size}) ---");
        for round in 0..max_rounds {
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

            let t0 = Instant::now();
            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps, &mut draft_cache)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            let draft_ms = t0.elapsed().as_millis();
            total_draft_ms += draft_ms;

            if std::env::var("DFLASH_NO_DRAFT_CACHE").is_ok() {
                draft_cache = drafter.make_cache();
            } else {
                crop_drafter_cache(&mut draft_cache, start);
            }

            let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));

            if round == 0 {
                let tap0 = &current_taps[0];
                println!(
                    "  taps: ctx_len={}, tap_shape={:?}, tap_dtype={:?}",
                    tap0.shape()[1],
                    tap0.shape(),
                    tap0.dtype()
                );
                println!(
                    "  noise_emb: shape={:?}, dtype={:?}",
                    noise_embedding.shape(),
                    noise_embedding.dtype()
                );
                println!(
                    "  draft_hidden: shape={:?}, dtype={:?}",
                    draft_hidden.shape(),
                    draft_hidden.dtype()
                );
            }

            let draft_logits = target
                .forward_all_logits_from_hidden(&draft_hidden_sliced)
                .unwrap();
            let draft_token_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_token_ids]).unwrap();

            let draft_u32: Vec<u32> = draft_token_ids
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            let gdn_backup = GdnStateBackup::save(&kv_cache).unwrap();

            let t0 = Instant::now();
            let (verify_logits, verify_taps) = target
                .forward_with_taps(&verify_input, None, &mut kv_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();
            let verify_ms = t0.elapsed().as_millis();
            total_verify_ms += verify_ms;

            let verify_argmax_arr = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_argmax_arr
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();

            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len();

            if round == 0 {
                println!("  draft_u32:  {:?}", &draft_u32[..draft_u32.len().min(15)]);
                println!(
                    "  verify_flat: {:?}",
                    &verify_flat[..verify_flat.len().min(16)]
                );
                println!("  accepted:    {:?}", accepted);
                println!("  n_accepted:  {n_accepted} (of {} drafts)", draft_u32.len());
            }

            if (n_accepted as i32) < block_size {
                let rollback = verify_len - n_accepted as i32;
                GdnStateBackup::restore_and_rollback(&gdn_backup, &mut kv_cache, rollback);
                current_taps = verify_taps
                    .into_iter()
                    .map(|tap| tap.index((.., ..n_accepted as i32, ..)))
                    .collect();
            } else {
                current_taps = verify_taps
                    .into_iter()
                    .map(|tap| tap.index((.., ..n_accepted as i32, ..)))
                    .collect();
            }

            total_accepted += n_accepted;
            total_tokens += n_accepted;

            for &tok in &accepted {
                generated.push(tok as i32);
            }
            last_token = *accepted.last().unwrap() as i32;
            start += n_accepted as i32;

            println!(
                "Round {round}: draft={draft_ms}ms verify={verify_ms}ms accepted={n_accepted}/{} draft={draft_flat:?}",
                block_size - 1
            );

            if generated.contains(&eos_token) {
                println!("EOS detected, stopping.");
                break;
            }
        }

        let total_ms = total_draft_ms + total_verify_ms;
        let tok_per_sec = if total_ms > 0 {
            total_tokens as f64 / (total_ms as f64 / 1000.0)
        } else {
            0.0
        };

        println!("\n--- 4B DFlash Results ---");
        println!("Total tokens: {total_tokens}");
        println!("Total rounds: {max_rounds}");
        println!(
            "Avg acceptance: {:.1} tok/round",
            total_accepted as f64 / max_rounds as f64
        );
        println!("Total draft time: {total_draft_ms}ms");
        println!("Total verify time: {total_verify_ms}ms");
        println!("Throughput: {tok_per_sec:.1} tok/s");
        println!(
            "Generated tokens: {:?}",
            &generated[..generated.len().min(50)]
        );
    }

    #[test]
    #[ignore] // requires 9B target + drafter model weights on disk
    fn test_dflash_9b_smoke() {
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit";
        let drafter_path = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash",
        );

        println!("Loading 9B target model...");
        let t0 = Instant::now();
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Target loaded in {:.1}s", t0.elapsed().as_secs_f64());

        println!("Loading 9B DFlash drafter...");
        let t0 = Instant::now();
        let mut drafter = load_dflash_drafter(drafter_path).unwrap();
        println!("Drafter loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        println!(
            "Config check: hidden_size={} taps={:?} block_size={}",
            drafter.config.hidden_size, tap_layers, block_size
        );

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 11964, 264, 2820, 6804, 323, 1077, 248046, 198, 248045, 74455, 198,
            248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        println!("Prefilling 9B target...");
        let t0 = Instant::now();
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        let mut eval_targets: Vec<&Array> = vec![&prefill_logits];
        for t in &taps {
            eval_targets.push(t);
        }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        eval_targets.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_targets.push(c);
                    }
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();
        println!(
            "Prefill OK in {}ms, tap shapes={:?}",
            t0.elapsed().as_millis(),
            taps.iter().map(|t| t.shape().to_vec()).collect::<Vec<_>>()
        );

        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am
            .reshape(&[-1])
            .unwrap()
            .as_slice::<u32>()
            .to_vec();
        let last_token = *am_flat.last().unwrap() as i32;

        let mut block_tokens = vec![mask_id; block_size as usize];
        block_tokens[0] = last_token;
        let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
        let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

        let mut draft_cache = drafter.make_cache();
        let draft_hidden = drafter
            .forward(&noise_embedding, &taps, &mut draft_cache)
            .unwrap();
        mlx_rs::transforms::eval([&draft_hidden]).unwrap();
        assert_eq!(
            draft_hidden.shape(),
            &[1, block_size, drafter.config.hidden_size]
        );

        let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
        let draft_logits = target
            .forward_all_logits_from_hidden(&draft_hidden_sliced)
            .unwrap();
        let draft_token_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
        mlx_rs::transforms::eval([&draft_token_ids]).unwrap();
        let draft_flat: Vec<u32> = draft_token_ids
            .reshape(&[-1])
            .unwrap()
            .as_slice::<u32>()
            .to_vec();

        println!(
            "Smoke OK: last_token={} first_draft={:?}",
            last_token,
            &draft_flat[..draft_flat.len().min(8)]
        );
    }

    #[test]
    #[ignore] // requires 9B target + drafter model weights on disk
    fn test_dflash_9b_full_loop() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let no_draft_cache = std::env::var("DFLASH_NO_DRAFT_CACHE").is_ok();

        let target_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit";
        let drafter_path = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash",
        );

        println!("Loading 9B target (4-bit), no_draft_cache={no_draft_cache}...");
        let t0 = Instant::now();
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Target loaded in {:.1}s", t0.elapsed().as_secs_f64());

        println!("Loading 9B DFlash drafter...");
        let t0 = Instant::now();
        let mut drafter = load_dflash_drafter(drafter_path).unwrap();
        println!("Drafter loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 11964, 264, 2820, 6804, 323, 1077, 248046, 198, 248045, 74455, 198,
            248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        println!("Prefilling...");
        let t0 = Instant::now();
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        let mut eval_targets: Vec<&Array> = vec![&prefill_logits];
        for t in &taps {
            eval_targets.push(t);
        }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        eval_targets.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_targets.push(c);
                    }
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();
        let prefill_ms = t0.elapsed().as_millis();
        println!(
            "Prefill: {prefill_ms}ms, taps: {} (each shape: {:?})",
            taps.len(),
            taps.first().map(|t| t.shape().to_vec())
        );

        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am
            .reshape(&[-1])
            .unwrap()
            .as_slice::<u32>()
            .to_vec();
        let mut last_token: i32 = *am_flat.last().unwrap() as i32;

        let max_rounds = 20;
        let mut total_tokens = 0usize;
        let mut total_accepted = 0usize;
        let mut total_draft_ms = 0u128;
        let mut total_verify_ms = 0u128;
        let mut generated: Vec<i32> = vec![last_token];
        let mut current_taps = taps;
        let mut draft_cache = drafter.make_cache();
        let mut start = prompt_len;

        println!("\n--- 27B DFlash generation (mask_id={mask_id}, block_size={block_size}) ---");
        for round in 0..max_rounds {
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

            let t0 = Instant::now();
            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps, &mut draft_cache)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            let draft_ms = t0.elapsed().as_millis();
            total_draft_ms += draft_ms;

            // Crop draft cache AFTER forward to discard speculative entries
            crop_drafter_cache(&mut draft_cache, start);

            let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
            let draft_logits = target
                .forward_all_logits_from_hidden(&draft_hidden_sliced)
                .unwrap();
            let draft_token_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_token_ids]).unwrap();

            let draft_u32: Vec<u32> = draft_token_ids
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            let kv_cache_clone = kv_cache.clone();

            let t0 = Instant::now();
            let (verify_logits, verify_taps) = target
                .forward_with_taps(&verify_input, None, &mut kv_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();
            let verify_ms = t0.elapsed().as_millis();
            total_verify_ms += verify_ms;

            let verify_argmax_arr = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_argmax_arr
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();

            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len();

            total_accepted += n_accepted;
            total_tokens += n_accepted;
            for &tok in &accepted {
                generated.push(tok as i32);
            }
            last_token = *accepted.last().unwrap() as i32;

            if (n_accepted as i32) < block_size {
                // Restore target cache to pre-verify state, then re-run only accepted tokens.
                // This gives clean taps without stale speculative KV entries.
                // Skip anchor at [0] — it's already in the restored cache.
                kv_cache = kv_cache_clone;
                let rerun_len = (n_accepted - 1) as i32;
                let rerun_input =
                    Array::from_slice(&verify_tokens[1..n_accepted], &[1, rerun_len]);
                let (_rerun_logits, rerun_taps) = target
                    .forward_with_taps(&rerun_input, None, &mut kv_cache, &tap_layers)
                    .unwrap();
                mlx_rs::transforms::eval([&_rerun_logits]).unwrap();
                current_taps = rerun_taps;
                // No rollback needed — kv_cache_clone + rerun already gives clean state.
            } else {
                current_taps = verify_taps
                    .into_iter()
                    .map(|tap| tap.index((.., ..n_accepted as i32, ..)))
                    .collect();
            }

            start += n_accepted as i32;

            println!(
                "Round {round}: draft={draft_ms}ms verify={verify_ms}ms accepted={n_accepted}/{} draft={draft_flat:?}",
                block_size - 1
            );
        }

        let total_ms = total_draft_ms + total_verify_ms;
        let tok_per_sec = if total_ms > 0 {
            total_tokens as f64 / (total_ms as f64 / 1000.0)
        } else {
            0.0
        };

        println!("\n--- 9B Results ---");
        println!("Total tokens: {total_tokens}");
        println!("Total rounds: {max_rounds}");
        println!(
            "Avg acceptance: {:.1} tok/round",
            total_accepted as f64 / max_rounds as f64
        );
        println!("Total draft time: {total_draft_ms}ms");
        println!("Total verify time: {total_verify_ms}ms");
        println!("Throughput: {tok_per_sec:.1} tok/s");
        println!(
            "Generated tokens: {:?}",
            &generated[..generated.len().min(50)]
        );
    }

    #[test]
    #[ignore] // requires Qwen/Qwen3.5-9B + z-lab 9B drafter weights on disk
    fn test_dflash_9b_qwen_full_loop() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit";
        let drafter_path = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash",
        );

        println!("Loading 9B target model...");
        let t0 = Instant::now();
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Target loaded in {:.1}s", t0.elapsed().as_secs_f64());

        println!("Loading 9B DFlash drafter...");
        let t0 = Instant::now();
        let mut drafter = load_dflash_drafter(drafter_path).unwrap();
        println!("Drafter loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 11964, 264, 2820, 6804, 323, 1077, 248046, 198, 248045, 74455, 198,
            248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        println!("Prefilling...");
        let t0 = Instant::now();
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        let mut eval_targets: Vec<&Array> = vec![&prefill_logits];
        for t in &taps {
            eval_targets.push(t);
        }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        eval_targets.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_targets.push(c);
                    }
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();
        let prefill_ms = t0.elapsed().as_millis();
        println!(
            "Prefill: {prefill_ms}ms, taps: {} (each shape: {:?})",
            taps.len(),
            taps.first().map(|t| t.shape().to_vec())
        );

        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am
            .reshape(&[-1])
            .unwrap()
            .as_slice::<u32>()
            .to_vec();
        let mut last_token: i32 = *am_flat.last().unwrap() as i32;

        let max_rounds = 20;
        let mut total_tokens = 0usize;
        let mut total_accepted = 0usize;
        let mut total_draft_ms = 0u128;
        let mut total_verify_ms = 0u128;
        let mut generated: Vec<i32> = vec![last_token];
        let mut current_taps = taps;
        let mut draft_cache = drafter.make_cache();
        let mut start = prompt_len;

        println!(
            "\n--- 9B Qwen DFlash generation (mask_id={mask_id}, block_size={block_size}) ---"
        );
        for round in 0..max_rounds {
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

            let t0 = Instant::now();
            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps, &mut draft_cache)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            let draft_ms = t0.elapsed().as_millis();
            total_draft_ms += draft_ms;

            // Crop draft cache AFTER forward to discard speculative entries
            crop_drafter_cache(&mut draft_cache, start);

            let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
            let draft_logits = target
                .forward_all_logits_from_hidden(&draft_hidden_sliced)
                .unwrap();
            let draft_token_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_token_ids]).unwrap();

            let draft_u32: Vec<u32> = draft_token_ids
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            let kv_cache_clone = kv_cache.clone();

            let t0 = Instant::now();
            let (verify_logits, verify_taps) = target
                .forward_with_taps(&verify_input, None, &mut kv_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();
            let verify_ms = t0.elapsed().as_millis();
            total_verify_ms += verify_ms;

            let verify_argmax_arr = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_argmax_arr
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();

            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len();

            total_accepted += n_accepted;
            total_tokens += n_accepted;
            for &tok in &accepted {
                generated.push(tok as i32);
            }
            last_token = *accepted.last().unwrap() as i32;

            if (n_accepted as i32) < block_size {
                // Restore target cache, rerun accepted tokens for clean taps.
                // Skip anchor at [0] — it's already in the restored cache.
                kv_cache = kv_cache_clone;
                let rerun_len = (n_accepted - 1) as i32;
                let rerun_input =
                    Array::from_slice(&verify_tokens[1..n_accepted], &[1, rerun_len]);
                let (_rerun_logits, rerun_taps) = target
                    .forward_with_taps(&rerun_input, None, &mut kv_cache, &tap_layers)
                    .unwrap();
                mlx_rs::transforms::eval([&_rerun_logits]).unwrap();
                current_taps = rerun_taps;
            } else {
                current_taps = verify_taps
                    .into_iter()
                    .map(|tap| tap.index((.., ..n_accepted as i32, ..)))
                    .collect();
            }

            start += n_accepted as i32;

            println!(
                "Round {round}: draft={draft_ms}ms verify={verify_ms}ms accepted={n_accepted}/{} draft={draft_flat:?}",
                block_size - 1
            );
        }

        let total_ms = total_draft_ms + total_verify_ms;
        let tok_per_sec = if total_ms > 0 {
            total_tokens as f64 / (total_ms as f64 / 1000.0)
        } else {
            0.0
        };

        println!("\n--- 9B Qwen Results ---");
        println!("Total tokens: {total_tokens}");
        println!("Total rounds: {max_rounds}");
        println!(
            "Avg acceptance: {:.1} tok/round",
            total_accepted as f64 / max_rounds as f64
        );
        println!("Total draft time: {total_draft_ms}ms");
        println!("Total verify time: {total_verify_ms}ms");
        println!("Throughput: {tok_per_sec:.1} tok/s");
        println!(
            "Generated tokens: {:?}",
            &generated[..generated.len().min(50)]
        );
    }

    #[test]
    #[ignore]
    fn test_9b_4bit_baseline_decode() {
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit";
        println!("Loading 9B 4-bit target...");
        let t0 = Instant::now();
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 11964, 264, 2820, 6804, 323, 1077, 248046, 198, 248045, 74455, 198,
            248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let t0 = Instant::now();
        let prefill_logits = target.forward(&input_ids, None, &mut kv_cache).unwrap();
        mlx_rs::transforms::eval(
            std::iter::once(&prefill_logits)
                .chain(kv_cache.iter().flatten().flat_map(|lc| match lc {
                    crate::qwen3_next::LayerCache::KV(kv) => kv.eval_targets(),
                    crate::qwen3_next::LayerCache::Arrays(ac) => {
                        let mut t = vec![];
                        if let Some(ref s) = ac.ssm_state {
                            t.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            t.push(c);
                        }
                        t
                    }
                }))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        println!("Prefill: {}ms", t0.elapsed().as_millis());

        let mut last = {
            let am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
            am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec()[0] as i32
        };
        let mut generated = vec![last];

        let t0 = Instant::now();
        for _ in 0..40 {
            let inp = Array::from_slice(&[last], &[1, 1]);
            let logits = target.forward(&inp, None, &mut kv_cache).unwrap();
            mlx_rs::transforms::eval([&logits]).unwrap();
            let am = mlx_rs::argmax_axis!(logits, -1).unwrap();
            last = am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec()[0] as i32;
            generated.push(last);
            if last == 248044 {
                break;
            }
        }
        let decode_ms = t0.elapsed().as_millis();
        let n = generated.len() - 1; // exclude prefill token
        println!(
            "Decoded {} tokens in {}ms ({:.1} tok/s)",
            n,
            decode_ms,
            n as f64 / (decode_ms as f64 / 1000.0)
        );
        println!("Tokens: {:?}", generated);
    }

    #[test]
    #[ignore]
    fn test_4b_bf16_ar_baseline() {
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-bf16";
        println!("Loading 4B BF16 target...");
        let mut target = load_qwen3_5_model(target_path).unwrap();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 11964, 264, 2820, 6804, 323, 1077, 248046, 198, 248045, 74455, 198,
            248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let t0 = Instant::now();
        let prefill_logits = target.forward(&input_ids, None, &mut kv_cache).unwrap();
        mlx_rs::transforms::eval(
            std::iter::once(&prefill_logits)
                .chain(kv_cache.iter().flatten().flat_map(|lc| match lc {
                    crate::qwen3_next::LayerCache::KV(kv) => kv.eval_targets(),
                    crate::qwen3_next::LayerCache::Arrays(ac) => {
                        let mut t = vec![];
                        if let Some(ref s) = ac.ssm_state { t.push(s); }
                        if let Some(ref c) = ac.conv_state { t.push(c); }
                        t
                    }
                }))
                .collect::<Vec<_>>(),
        ).unwrap();
        println!("Prefill: {}ms", t0.elapsed().as_millis());

        let mut last = {
            let am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
            am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec()[0] as i32
        };
        let mut generated = vec![last];
        let n_tokens = 65; // match DFlash output length

        let t0 = Instant::now();
        for _ in 0..n_tokens {
            let inp = Array::from_slice(&[last], &[1, 1]);
            let logits = target.forward(&inp, None, &mut kv_cache).unwrap();
            mlx_rs::transforms::eval([&logits]).unwrap();
            let am = mlx_rs::argmax_axis!(logits, -1).unwrap();
            last = am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec()[0] as i32;
            generated.push(last);
        }
        let decode_ms = t0.elapsed().as_millis();
        println!(
            "AR decode: {} tokens in {}ms ({:.1} tok/s)",
            n_tokens, decode_ms,
            n_tokens as f64 / (decode_ms as f64 / 1000.0)
        );
        println!("First 30 tokens: {:?}", &generated[..generated.len().min(30)]);
    }

    /// Sweep generation lengths to measure acceptance vs context length.
    #[test]
    #[ignore]
    fn test_dflash_4b_length_sweep() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-bf16";
        let drafter_path = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        );
        let snap_dir = std::fs::read_dir(drafter_path.join("snapshots"))
            .unwrap().filter_map(|e| e.ok()).next().unwrap().path();

        println!("Loading models...");
        let mut target = load_qwen3_5_model(target_path).unwrap();
        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();
        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 11964, 264, 2820, 6804, 323, 1077, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers).unwrap();
        let mut eval_targets: Vec<&Array> = vec![&prefill_logits];
        for t in &taps { eval_targets.push(t); }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state { eval_targets.push(s); }
                    if let Some(ref c) = ac.conv_state { eval_targets.push(c); }
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();

        let am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let first_token = *am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec().last().unwrap() as i32;

        let mut current_taps = taps;
        mlx_rs::transforms::eval(current_taps.iter().collect::<Vec<_>>()).unwrap();
        let mut draft_cache = drafter.make_cache();
        let mut last_token = first_token;
        let mut start = prompt_len;
        let mut total_tokens = 0usize;
        let mut round = 0usize;
        let mut acceptance_list: Vec<i32> = Vec::new();
        let mut total_draft_ms = 0u128;
        let mut total_verify_ms = 0u128;
        let checkpoints = [512, 1024, 2048];
        let max_tokens = 2048;
        let t_global = Instant::now();

        println!("\n--- 4B BF16 length sweep (block_size={block_size}) ---");
        while total_tokens < max_tokens {
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

            let t0 = Instant::now();
            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps, &mut draft_cache).unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            total_draft_ms += t0.elapsed().as_millis();
            crop_drafter_cache(&mut draft_cache, start);

            let draft_logits = target
                .forward_all_logits_from_hidden(&draft_hidden.index((.., 1.., ..))).unwrap();
            let draft_am = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_am]).unwrap();
            let draft_u32: Vec<u32> = draft_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            let snapshots: Vec<(Option<Array>, Option<Array>, i32)> = kv_cache.iter()
                .map(|lc| match lc {
                    Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                        ac.eval_arrays().unwrap();
                        (ac.conv_state.clone(), ac.ssm_state.clone(), ac.offset)
                    }
                    _ => (None, None, 0),
                }).collect();

            let t0 = Instant::now();
            let (verify_logits, verify_taps, layer_tapes) = target
                .forward_with_taps_tape(&verify_input, None, &mut kv_cache, &tap_layers).unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();
            total_verify_ms += t0.elapsed().as_millis();

            let verify_am = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len() as i32;

            if n_accepted < block_size {
                let kv_rollback = verify_len - n_accepted;
                target.replay_tape_rollback(
                    &layer_tapes, &mut kv_cache,
                    n_accepted, kv_rollback,
                ).unwrap();
                let replay_states: Vec<&Array> = kv_cache.iter()
                    .filter_map(|lc| match lc {
                        Some(crate::qwen3_next::LayerCache::Arrays(ac)) => ac.ssm_state.as_ref(),
                        _ => None,
                    }).collect();
                if !replay_states.is_empty() {
                    mlx_rs::transforms::eval(replay_states).unwrap();
                }
            }
            current_taps = verify_taps.into_iter()
                .map(|tap| tap.index((.., ..n_accepted, ..))).collect();

            acceptance_list.push(n_accepted);
            total_tokens += n_accepted as usize;
            last_token = *accepted.last().unwrap() as i32;
            start += n_accepted;
            round += 1;

            // Print checkpoints
            for &cp in &checkpoints {
                if total_tokens >= cp && (total_tokens - n_accepted as usize) < cp {
                    let elapsed = t_global.elapsed().as_secs_f64();
                    let recent_50: f64 = if acceptance_list.len() >= 50 {
                        acceptance_list[acceptance_list.len()-50..].iter().sum::<i32>() as f64 / 50.0
                    } else {
                        acceptance_list.iter().sum::<i32>() as f64 / acceptance_list.len() as f64
                    };
                    let overall_avg = acceptance_list.iter().sum::<i32>() as f64 / acceptance_list.len() as f64;
                    let tps = total_tokens as f64 / elapsed;
                    println!(
                        "\n  >>> {cp} tokens: {round} rounds, avg_accept={overall_avg:.1}, last50_accept={recent_50:.1}, {tps:.1} tok/s, draft={total_draft_ms}ms verify={total_verify_ms}ms"
                    );
                }
            }

            if round % 50 == 0 {
                let avg = acceptance_list[acceptance_list.len().saturating_sub(50)..].iter().sum::<i32>() as f64
                    / acceptance_list[acceptance_list.len().saturating_sub(50)..].len() as f64;
                println!("  round {round}: {total_tokens} tokens, last50_accept={avg:.1}");
            }
        }

        let elapsed = t_global.elapsed().as_secs_f64();
        let overall_avg = acceptance_list.iter().sum::<i32>() as f64 / acceptance_list.len() as f64;
        println!("\n=== FINAL: {total_tokens} tokens in {elapsed:.1}s ===");
        println!("Rounds: {round}, Avg acceptance: {overall_avg:.1}");
        println!("Draft: {total_draft_ms}ms, Verify: {total_verify_ms}ms");
        println!("Throughput: {:.1} tok/s", total_tokens as f64 / elapsed);
    }

    /// Iterative refinement: run drafter N times, replacing masks with draft tokens
    /// from previous pass. Measures acceptance vs refinement passes.
    #[test]
    #[ignore]
    fn test_dflash_4b_iterative_refinement() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-bf16";
        let drafter_path = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        );
        let snap_dir = std::fs::read_dir(drafter_path.join("snapshots"))
            .unwrap().filter_map(|e| e.ok()).next().unwrap().path();

        println!("Loading models...");
        let mut target = load_qwen3_5_model(target_path).unwrap();
        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();
        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 11964, 264, 2820, 6804, 323, 1077, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        // Prefill
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers).unwrap();
        let mut eval_targets: Vec<&Array> = vec![&prefill_logits];
        for t in &taps { eval_targets.push(t); }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state { eval_targets.push(s); }
                    if let Some(ref c) = ac.conv_state { eval_targets.push(c); }
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();

        let am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
        let first_token = *am_flat.last().unwrap() as i32;

        // Test with 1, 2, 3 refinement passes
        for n_passes in 1..=3 {
            // Clone state for each config
            let mut kv = kv_cache.clone();
            // Force eval cloned cache
            for lc in kv.iter() {
                match lc {
                    Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                        if let Some(ref s) = ac.ssm_state { mlx_rs::transforms::eval([s]).unwrap(); }
                        if let Some(ref c) = ac.conv_state { mlx_rs::transforms::eval([c]).unwrap(); }
                    }
                    Some(crate::qwen3_next::LayerCache::KV(kvc)) => {
                        mlx_rs::transforms::eval(kvc.eval_targets()).unwrap();
                    }
                    _ => {}
                }
            }
            let mut current_taps = taps.clone();
            mlx_rs::transforms::eval(current_taps.iter().collect::<Vec<_>>()).unwrap();
            let mut draft_cache = drafter.make_cache();
            let mut last_token = first_token;
            let mut start = prompt_len;
            let mut total_tokens = 0usize;
            let mut total_draft_ms = 0u128;
            let mut total_verify_ms = 0u128;
            let mut acceptance_list = Vec::new();
            let max_rounds = 20;

            println!("\n=== {n_passes}-pass refinement (block_size={block_size}) ===");
            for round in 0..max_rounds {
                // --- Draft with N refinement passes ---
                let t0 = Instant::now();

                // Pass 1: masks
                let mut block_tokens = vec![mask_id; block_size as usize];
                block_tokens[0] = last_token;
                let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
                let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

                let draft_hidden = drafter
                    .forward(&noise_embedding, &current_taps, &mut draft_cache).unwrap();
                mlx_rs::transforms::eval([&draft_hidden]).unwrap();
                crop_drafter_cache(&mut draft_cache, start);

                let mut draft_logits = target
                    .forward_all_logits_from_hidden(&draft_hidden.index((.., 1.., ..))).unwrap();
                let mut draft_am = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
                mlx_rs::transforms::eval([&draft_am]).unwrap();

                // Refinement passes 2..N: replace masks with draft tokens
                for _pass in 1..n_passes {
                    let draft_u32: Vec<u32> = draft_am.reshape(&[-1]).unwrap()
                        .as_slice::<u32>().to_vec();
                    let mut refined = vec![last_token];
                    refined.extend(draft_u32.iter().map(|&x| x as i32));
                    let refined_ids = Array::from_slice(&refined, &[1, block_size]);
                    let refined_emb = target.embed_token_ids(&refined_ids).unwrap();

                    let refined_hidden = drafter
                        .forward(&refined_emb, &current_taps, &mut draft_cache).unwrap();
                    mlx_rs::transforms::eval([&refined_hidden]).unwrap();
                    crop_drafter_cache(&mut draft_cache, start);

                    draft_logits = target
                        .forward_all_logits_from_hidden(&refined_hidden.index((.., 1.., ..))).unwrap();
                    draft_am = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
                    mlx_rs::transforms::eval([&draft_am]).unwrap();
                }

                let draft_ms = t0.elapsed().as_millis();
                total_draft_ms += draft_ms;

                let draft_u32: Vec<u32> = draft_am.reshape(&[-1]).unwrap()
                    .as_slice::<u32>().to_vec();
                let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

                // --- Verify ---
                let mut verify_tokens = vec![last_token];
                verify_tokens.extend_from_slice(&draft_flat);
                let verify_len = verify_tokens.len() as i32;
                let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

                let snapshots: Vec<(Option<Array>, Option<Array>, i32)> = kv.iter()
                    .map(|lc| match lc {
                        Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                            ac.eval_arrays().unwrap();
                            (ac.conv_state.clone(), ac.ssm_state.clone(), ac.offset)
                        }
                        _ => (None, None, 0),
                    }).collect();

                let t0 = Instant::now();
                let (verify_logits, verify_taps, layer_tapes) = target
                    .forward_with_taps_tape(&verify_input, None, &mut kv, &tap_layers).unwrap();
                mlx_rs::transforms::eval([&verify_logits]).unwrap();
                let verify_ms = t0.elapsed().as_millis();
                total_verify_ms += verify_ms;

                let verify_am = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
                let verify_flat: Vec<u32> = verify_am.reshape(&[-1]).unwrap()
                    .as_slice::<u32>().to_vec();
                let accepted = accept_prefix(&draft_u32, &verify_flat);
                let n_accepted = accepted.len() as i32;

                if n_accepted < block_size {
                    let kv_rollback = verify_len - n_accepted;
                    target.replay_tape_rollback(
                        &layer_tapes, &mut kv,
                        n_accepted, kv_rollback,
                    ).unwrap();
                    let replay_states: Vec<&Array> = kv.iter()
                        .filter_map(|lc| match lc {
                            Some(crate::qwen3_next::LayerCache::Arrays(ac)) => ac.ssm_state.as_ref(),
                            _ => None,
                        }).collect();
                    if !replay_states.is_empty() {
                        mlx_rs::transforms::eval(replay_states).unwrap();
                    }
                }
                current_taps = verify_taps.into_iter()
                    .map(|tap| tap.index((.., ..n_accepted, ..))).collect();

                acceptance_list.push(n_accepted);
                total_tokens += n_accepted as usize;
                for &tok in &accepted { }
                last_token = *accepted.last().unwrap() as i32;
                start += n_accepted;

                println!("  Round {round}: draft={draft_ms}ms verify={verify_ms}ms accepted={n_accepted}/{}", block_size - 1);
            }

            let avg_accept = acceptance_list.iter().sum::<i32>() as f64 / max_rounds as f64;
            let total_ms = total_draft_ms + total_verify_ms;
            let tps = total_tokens as f64 / (total_ms as f64 / 1000.0);
            println!("\n--- {n_passes}-PASS RESULTS ---");
            println!("Acceptance per round: {:?}", acceptance_list);
            println!("Avg acceptance: {avg_accept:.1} tok/round");
            println!("Total: {total_tokens} tokens in {total_ms}ms");
            println!("Draft total: {total_draft_ms}ms, Verify total: {total_verify_ms}ms");
            println!("Throughput: {tps:.1} tok/s");
        }
    }

    #[test]
    #[ignore] // requires 27B 4-bit target + z-lab 27B DFlash drafter weights on disk
    fn test_dflash_27b_full_loop() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit";
        let drafter_base = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash",
        );
        let snap_dir = std::fs::read_dir(drafter_base.join("snapshots"))
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .unwrap()
            .path();

        println!("Loading 27B 4-bit target...");
        let t0 = Instant::now();
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Target loaded in {:.1}s", t0.elapsed().as_secs_f64());

        println!("Loading 27B DFlash drafter...");
        let t0 = Instant::now();
        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();
        println!("Drafter loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let configured_block_size = drafter.config.block_size;
        let block_size = std::env::var("HIGGS_DFLASH_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|&n| n >= 2 && n <= configured_block_size)
            .unwrap_or(configured_block_size);
        let mask_id = drafter.config.mask_token_id();

        println!(
            "Config: tap_layers={tap_layers:?} block_size={block_size} hidden={}",
            drafter.config.hidden_size
        );

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let eos_token: i32 = 248046;
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        println!("Prefilling 27B...");
        let t0 = Instant::now();
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        let mut eval_targets: Vec<&Array> = vec![&prefill_logits];
        for t in &taps {
            eval_targets.push(t);
        }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        eval_targets.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_targets.push(c);
                    }
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();
        let prefill_ms = t0.elapsed().as_millis();
        println!(
            "Prefill: {prefill_ms}ms, taps: {} (each shape: {:?})",
            taps.len(),
            taps.first().map(|t| t.shape().to_vec())
        );

        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am
            .reshape(&[-1])
            .unwrap()
            .as_slice::<u32>()
            .to_vec();
        let mut last_token: i32 = *am_flat.last().unwrap() as i32;

        let max_rounds = 20;
        let mut total_tokens = 0usize;
        let mut total_accepted = 0usize;
        let mut total_draft_ms = 0u128;
        let mut total_verify_ms = 0u128;
        let mut generated: Vec<i32> = vec![last_token];
        let mut current_taps = taps;
        let mut draft_cache = drafter.make_cache();
        let mut start = prompt_len;

        println!("\n--- 27B DFlash generation (mask_id={mask_id}, block_size={block_size}) ---");
        for round in 0..max_rounds {
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

            let t0 = Instant::now();
            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps, &mut draft_cache)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            let draft_ms = t0.elapsed().as_millis();
            total_draft_ms += draft_ms;

            // Crop draft cache AFTER forward to discard speculative entries
            crop_drafter_cache(&mut draft_cache, start);

            let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
            let draft_logits = target
                .forward_all_logits_from_hidden(&draft_hidden_sliced)
                .unwrap();
            let draft_token_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_token_ids]).unwrap();

            let draft_u32: Vec<u32> = draft_token_ids
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            let gdn_backup = GdnStateBackup::save(&kv_cache).unwrap();

            let t0 = Instant::now();
            let (verify_logits, verify_taps) = target
                .forward_with_taps(&verify_input, None, &mut kv_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();
            let verify_ms = t0.elapsed().as_millis();
            total_verify_ms += verify_ms;

            let verify_argmax_arr = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_argmax_arr
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();

            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len();

            // Rollback GDN state + trim rejected KV entries, slice verify taps.
            // Mirror-sd approach: no rerun, just trim + slice. The verify taps
            // at accepted positions are valid (causal attention layers) or
            // slightly contaminated (GDN layers see future tokens) but the
            // drafter tolerates this.
            if (n_accepted as i32) < block_size {
                let rollback = verify_len - n_accepted as i32;
                GdnStateBackup::restore_and_rollback(&gdn_backup, &mut kv_cache, rollback);
            }
            current_taps = verify_taps
                .into_iter()
                .map(|tap| tap.index((.., ..n_accepted as i32, ..)))
                .collect();

            if round == 0 {
                println!("  draft_u32:  {:?}", &draft_u32[..draft_u32.len().min(15)]);
                println!(
                    "  verify_flat: {:?}",
                    &verify_flat[..verify_flat.len().min(16)]
                );
                println!("  accepted:    {:?}", accepted);
            }

            total_accepted += n_accepted;
            total_tokens += n_accepted;
            for &tok in &accepted {
                generated.push(tok as i32);
            }
            last_token = *accepted.last().unwrap() as i32;

            start += n_accepted as i32;

            println!(
                "Round {round}: draft={draft_ms}ms verify={verify_ms}ms accepted={n_accepted}/{} draft={draft_flat:?}",
                block_size - 1
            );

            if generated.contains(&eos_token) {
                println!("EOS detected, stopping.");
                break;
            }
        }

        let total_ms = total_draft_ms + total_verify_ms;
        let tok_per_sec = if total_ms > 0 {
            total_tokens as f64 / (total_ms as f64 / 1000.0)
        } else {
            0.0
        };

        println!("\n--- 27B DFlash Results ---");
        println!("Total tokens: {total_tokens}");
        println!("Total rounds: {max_rounds}");
        println!(
            "Avg acceptance: {:.1} tok/round",
            total_accepted as f64 / max_rounds as f64
        );
        println!("Total draft time: {total_draft_ms}ms");
        println!("Total verify time: {total_verify_ms}ms");
        println!("Throughput: {tok_per_sec:.1} tok/s");
        println!(
            "Generated tokens: {:?}",
            &generated[..generated.len().min(50)]
        );
    }

    #[test]
    #[ignore]
    fn test_dflash_4b_bf16_full_loop() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-bf16";
        let drafter_path = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        );

        println!("Loading 4B BF16 target...");
        let t0 = Instant::now();
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Target loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let snap_dir = std::fs::read_dir(drafter_path.join("snapshots"))
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .unwrap()
            .path();

        println!("Loading 4B DFlash drafter...");
        let t0 = Instant::now();
        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();
        println!("Drafter loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 11964, 264, 2820, 6804, 323, 1077, 248046, 198, 248045, 74455, 198,
            248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        println!("Prefilling...");
        let t0 = Instant::now();
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        let mut eval_targets: Vec<&Array> = vec![&prefill_logits];
        for t in &taps {
            eval_targets.push(t);
        }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        eval_targets.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_targets.push(c);
                    }
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();
        let prefill_ms = t0.elapsed().as_millis();
        println!(
            "Prefill: {prefill_ms}ms, taps: {} (each shape: {:?})",
            taps.len(),
            taps.first().map(|t| t.shape().to_vec())
        );

        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am
            .reshape(&[-1])
            .unwrap()
            .as_slice::<u32>()
            .to_vec();
        let mut last_token: i32 = *am_flat.last().unwrap() as i32;

        let max_rounds = 20;
        let mut total_tokens = 0usize;
        let mut total_accepted = 0usize;
        let mut total_draft_ms = 0u128;
        let mut total_verify_ms = 0u128;
        let mut total_replay_ms = 0u128;
        let mut generated: Vec<i32> = vec![last_token];
        let mut current_taps = taps;
        let mut draft_cache = drafter.make_cache();
        let mut start = prompt_len;

        // Force-eval taps + prefill outputs to flush lazy graph before generation
        let tap_refs: Vec<&Array> = current_taps.iter().collect();
        mlx_rs::transforms::eval(tap_refs).unwrap();
        println!("Taps eval'd after prefill");

        println!(
            "\n--- 4B BF16 DFlash generation (mask_id={mask_id}, block_size={block_size}) ---"
        );
        for round in 0..max_rounds {
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

            let t0 = Instant::now();
            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps, &mut draft_cache)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            let draft_ms = t0.elapsed().as_millis();
            total_draft_ms += draft_ms;

            // Crop draft cache AFTER forward to discard speculative entries
            crop_drafter_cache(&mut draft_cache, start);

            let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
            let draft_logits = target
                .forward_all_logits_from_hidden(&draft_hidden_sliced)
                .unwrap();
            let draft_token_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_token_ids]).unwrap();

            let draft_u32: Vec<u32> = draft_token_ids
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            // Snapshot GDN state for tape replay
            let snapshots: Vec<(Option<Array>, Option<Array>, i32)> = kv_cache.iter()
                .map(|lc| match lc {
                    Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                        ac.eval_arrays().unwrap();
                        (ac.conv_state.clone(), ac.ssm_state.clone(), ac.offset)
                    }
                    _ => (None, None, 0),
                })
                .collect();

            let t0 = Instant::now();
            let (verify_logits, verify_taps, layer_tapes) = target
                .forward_with_taps_tape(&verify_input, None, &mut kv_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();
            let verify_ms = t0.elapsed().as_millis();
            total_verify_ms += verify_ms;

            let verify_argmax_arr = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_argmax_arr
                .reshape(&[-1])
                .unwrap()
                .as_slice::<u32>()
                .to_vec();

            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len();

            // Partial rejection — GDN-only replay from tape (no full rerun)
            let t0 = Instant::now();
            if (n_accepted as i32) < block_size {
                let kv_rollback = verify_len - n_accepted as i32;
                target.replay_tape_rollback(
                    &layer_tapes, &mut kv_cache,
                    n_accepted as i32, kv_rollback,
                ).unwrap();
                for lc in kv_cache.iter() {
                    if let Some(crate::qwen3_next::LayerCache::Arrays(ac)) = lc {
                        if let Some(ref s) = ac.ssm_state {
                            mlx_rs::transforms::eval([s]).unwrap();
                        }
                    }
                }
            }
            let replay_ms = t0.elapsed().as_millis();
            total_replay_ms += replay_ms;
            current_taps = verify_taps.into_iter()
                .map(|tap| tap.index((.., ..n_accepted as i32, ..)))
                .collect();

            if round == 0 {
                println!("  draft_u32:  {:?}", &draft_u32[..draft_u32.len().min(15)]);
                println!(
                    "  verify_flat: {:?}",
                    &verify_flat[..verify_flat.len().min(16)]
                );
                println!("  accepted:    {:?}", accepted);
                println!(
                    "  n_accepted:  {n_accepted} (of {} drafts)",
                    draft_u32.len()
                );
            }

            total_accepted += n_accepted;
            total_tokens += n_accepted;
            for &tok in &accepted {
                generated.push(tok as i32);
            }
            last_token = *accepted.last().unwrap() as i32;

            start += n_accepted as i32;

            println!(
                "Round {round}: draft={draft_ms}ms verify={verify_ms}ms replay={replay_ms}ms accepted={n_accepted}/{} draft={draft_flat:?}",
                block_size - 1
            );
        }

        let total_ms = total_draft_ms + total_verify_ms + total_replay_ms;
        let tok_per_sec = if total_ms > 0 {
            total_tokens as f64 / (total_ms as f64 / 1000.0)
        } else {
            0.0
        };

        println!("\n--- 4B BF16 Results (tape replay) ---");
        println!("Total tokens: {total_tokens}");
        println!("Total rounds: {max_rounds}");
        println!(
            "Avg acceptance: {:.1} tok/round",
            total_accepted as f64 / max_rounds as f64
        );
        println!("Total draft time: {total_draft_ms}ms");
        println!("Total verify time: {total_verify_ms}ms");
        println!("Total replay time: {total_replay_ms}ms (was ~30ms/round with full rerun)");
        println!("Throughput: {tok_per_sec:.1} tok/s");
        println!(
            "Generated tokens: {:?}",
            &generated[..generated.len().min(50)]
        );
    }

    /// Two consecutive forward_with_taps on 4B bf16 without DFlash.
    /// Isolates whether the verify-forward crash is DFlash-specific or
    /// inherent to multi-forward on the 4B model with S>1 + cached state.
    #[test]
    #[ignore]
    fn test_4b_bf16_two_forward_no_dflash() {
        use crate::cache::KeyValueCache;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path =
            "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit";
        if !std::path::Path::new(target_path).exists() {
            println!("Skipping: 4B model not found at {target_path}");
            return;
        }

        println!("Loading 4B BF16 target...");
        let t0 = Instant::now();
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Target loaded in {:.1}s", t0.elapsed().as_secs_f64());

        // Use the same tap layers as the 4B DFlash drafter
        let tap_layers: Vec<usize> = vec![1, 8, 15, 22, 29];

        // 1. Prefill: S=18 (same as the DFlash test prompt)
        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 11964, 264, 2820, 6804, 323, 1077, 248046, 198, 248045, 74455, 198,
            248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();

        println!("Forward 1 (prefill, S={prompt_len})...");
        let t0 = Instant::now();
        let (logits1, taps1) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        println!(
            "  forward_with_taps returned in {:.0}ms (lazy)",
            t0.elapsed().as_millis()
        );

        // Eval EVERYTHING: logits + taps + all cache arrays
        let mut eval_targets: Vec<&Array> = vec![&logits1];
        for t in &taps1 {
            eval_targets.push(t);
        }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => {
                    eval_targets.extend(kv.eval_targets());
                }
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        eval_targets.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_targets.push(c);
                    }
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();
        println!(
            "  eval'd in {:.0}ms total. logits1 shape: {:?}",
            t0.elapsed().as_millis(),
            logits1.shape()
        );

        // Print cache state for diagnostics
        let mut n_kv = 0;
        let mut n_gdn = 0;
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => {
                    n_kv += 1;
                    if n_kv == 1 {
                        println!(
                            "  First KV cache: offset={}, keys shape={:?}",
                            kv.offset(),
                            kv.keys().map(|k| k.shape().to_vec())
                        );
                    }
                }
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    n_gdn += 1;
                    if n_gdn == 1 {
                        println!(
                            "  First GDN cache: offset={}, ssm_state shape={:?}, conv_state shape={:?}",
                            ac.offset,
                            ac.ssm_state.as_ref().map(|s| s.shape().to_vec()),
                            ac.conv_state.as_ref().map(|c| c.shape().to_vec()),
                        );
                    }
                }
            }
        }
        println!("  Cache layers: {n_kv} KV + {n_gdn} GDN = {}", n_kv + n_gdn);

        // 2. Second forward: S=16 (simulates verify without any DFlash)
        //    Just feed 16 arbitrary tokens through the model with existing cache.
        let verify_tokens: Vec<i32> = vec![
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        ];
        let verify_len = verify_tokens.len() as i32;
        let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

        println!("\nForward 2 (verify-sized, S={verify_len})...");
        let t0 = Instant::now();
        let result = target.forward_with_taps(&verify_input, None, &mut kv_cache, &tap_layers);
        match result {
            Ok((logits2, taps2)) => {
                println!(
                    "  forward_with_taps returned in {:.0}ms (lazy)",
                    t0.elapsed().as_millis()
                );
                let mut eval_targets2: Vec<&Array> = vec![&logits2];
                for t in &taps2 {
                    eval_targets2.push(t);
                }
                for lc in kv_cache.iter().flatten() {
                    match lc {
                        crate::qwen3_next::LayerCache::KV(kv) => {
                            eval_targets2.extend(kv.eval_targets());
                        }
                        crate::qwen3_next::LayerCache::Arrays(ac) => {
                            if let Some(ref s) = ac.ssm_state {
                                eval_targets2.push(s);
                            }
                            if let Some(ref c) = ac.conv_state {
                                eval_targets2.push(c);
                            }
                        }
                    }
                }
                match mlx_rs::transforms::eval(eval_targets2) {
                    Ok(()) => {
                        println!(
                            "  eval'd in {:.0}ms. logits2 shape: {:?}",
                            t0.elapsed().as_millis(),
                            logits2.shape()
                        );
                        println!("\n  SUCCESS: Two consecutive forwards work on 4B bf16!");
                        println!("  => Crash is DFlash-specific, not inherent to multi-forward.");
                    }
                    Err(e) => {
                        println!(
                            "\n  CRASH during eval of forward 2: {e}\n  => Metal kernel or lazy graph issue, not a shape mismatch"
                        );
                        panic!("Forward 2 eval failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!(
                    "\n  CRASH during forward 2 graph build: {e}\n  => Shape mismatch in model forward"
                );
                panic!("Forward 2 graph build failed: {e}");
            }
        }

        // 3. Bonus: S=1 decode step (should always work if S>1 works)
        let decode_input = Array::from_slice(&[42_i32], &[1, 1]);
        println!("\nForward 3 (decode, S=1)...");
        let t0 = Instant::now();
        let (logits3, _) = target
            .forward_with_taps(&decode_input, None, &mut kv_cache, &tap_layers)
            .unwrap();
        mlx_rs::transforms::eval([&logits3]).unwrap();
        println!(
            "  S=1 decode: {:.0}ms, logits3 shape: {:?}",
            t0.elapsed().as_millis(),
            logits3.shape()
        );
        println!("  All three forwards succeeded!");
    }

    /// Verify that `forward_with_taps_stateless` produces identical logits
    /// to `forward_with_taps`, and that GDN state is NOT mutated.
    #[test]
    #[ignore] // requires 4B target model
    fn test_stateless_gdn_correctness() {
        use crate::qwen3_next::load_qwen3_5_model;

        let model_path = "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit";
        println!("Loading 4B model...");
        let mut target = load_qwen3_5_model(model_path).unwrap();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);
        let tap_layers = vec![1usize, 8, 15, 22, 29];

        // --- Run 1: stateful (normal) forward_with_taps ---
        let mut cache_stateful: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (logits_stateful, taps_stateful) = target
            .forward_with_taps(&input_ids, None, &mut cache_stateful, &tap_layers)
            .unwrap();
        mlx_rs::transforms::eval([&logits_stateful]).unwrap();
        for t in &taps_stateful {
            mlx_rs::transforms::eval([t]).unwrap();
        }

        // Save GDN states after stateful forward
        let gdn_states_after_stateful: Vec<Option<Array>> = cache_stateful
            .iter()
            .map(|lc| match lc {
                Some(crate::qwen3_next::LayerCache::Arrays(ac)) => ac.ssm_state.clone(),
                _ => None,
            })
            .collect();

        // --- Run 2: stateless forward_with_taps_stateless on fresh cache ---
        let mut cache_stateless: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (logits_stateless, taps_stateless) = target
            .forward_with_taps_stateless(&input_ids, None, &mut cache_stateless, &tap_layers)
            .unwrap();
        mlx_rs::transforms::eval([&logits_stateless]).unwrap();
        for t in &taps_stateless {
            mlx_rs::transforms::eval([t]).unwrap();
        }

        // --- Check 1: logits match ---
        let l1 = logits_stateful.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let l2 = logits_stateless.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let diff = l1.subtract(&l2).unwrap();
        let abs_diff = ops::abs(&diff).unwrap();
        let max_diff: f32 = ops::max(&abs_diff, None).unwrap().item();
        println!("Logits max abs diff: {max_diff:.6e}");
        assert!(
            max_diff < 1e-3,
            "Stateless logits diverge from stateful: max_diff={max_diff}"
        );

        // --- Check 2: taps match ---
        for (i, (t1, t2)) in taps_stateful.iter().zip(taps_stateless.iter()).enumerate() {
            let a = t1.as_dtype(mlx_rs::Dtype::Float32).unwrap();
            let b = t2.as_dtype(mlx_rs::Dtype::Float32).unwrap();
            let td: f32 = ops::max(&ops::abs(&a.subtract(&b).unwrap()).unwrap(), None)
                .unwrap()
                .item();
            println!("Tap {i} (layer {}) max diff: {td:.6e}", tap_layers[i]);
            assert!(td < 1e-3, "Tap {i} diverges: max_diff={td}");
        }

        // --- Check 3: GDN state NOT updated by stateless ---
        for (i, lc) in cache_stateless.iter().enumerate() {
            if let Some(crate::qwen3_next::LayerCache::Arrays(ac)) = lc {
                // ssm_state should be None (never written) or zero (initial)
                if let Some(ref state) = ac.ssm_state {
                    let max_s: f32 = ops::max(
                        &ops::abs(&state.as_dtype(mlx_rs::Dtype::Float32).unwrap()).unwrap(),
                        None,
                    )
                    .unwrap()
                    .item();
                    assert!(
                        max_s < 1e-10,
                        "Layer {i}: GDN state was mutated by stateless forward! max={max_s}"
                    );
                }
                assert_eq!(
                    ac.offset, 0,
                    "Layer {i}: GDN offset was mutated by stateless forward!"
                );
            }
        }

        // --- Check 4: stateful DID update GDN state ---
        let mut any_nonzero = false;
        for state_opt in &gdn_states_after_stateful {
            if let Some(state) = state_opt {
                let max_s: f32 = ops::max(
                    &ops::abs(&state.as_dtype(mlx_rs::Dtype::Float32).unwrap()).unwrap(),
                    None,
                )
                .unwrap()
                .item();
                if max_s > 1e-6 {
                    any_nonzero = true;
                }
            }
        }
        assert!(
            any_nonzero,
            "Stateful forward should have produced non-zero GDN states"
        );

        println!("All stateless correctness checks passed!");
    }

    /// DFlash 4B full loop using stateless GDN verify.
    /// No GdnStateBackup needed — stateless verify + KV rollback + replay.
    #[test]
    #[ignore] // requires 4B target + drafter model weights on disk
    fn test_dflash_4b_stateless_verify() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path =
            "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit";
        let drafter_base = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        );
        let snap_dir = std::fs::read_dir(drafter_base.join("snapshots"))
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .unwrap()
            .path();

        println!("Loading 4B target...");
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Loading 4B DFlash drafter...");
        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();

        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let eos_token: i32 = 248046;
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        // Prefill — stateful (establishes GDN + KV state)
        println!("Prefilling...");
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        {
            let mut ev: Vec<&Array> = vec![&prefill_logits];
            for t in &taps { ev.push(t); }
            for lc in kv_cache.iter().flatten() {
                match lc {
                    crate::qwen3_next::LayerCache::KV(kv) => ev.extend(kv.eval_targets()),
                    crate::qwen3_next::LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state { ev.push(s); }
                        if let Some(ref c) = ac.conv_state { ev.push(c); }
                    }
                }
            }
            mlx_rs::transforms::eval(ev).unwrap();
        }

        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
        let mut last_token: i32 = *am_flat.last().unwrap() as i32;

        let max_rounds = 20;
        let mut total_tokens = 0usize;
        let mut total_draft_ms = 0u128;
        let mut total_verify_ms = 0u128;
        let mut total_replay_ms = 0u128;
        let mut generated: Vec<i32> = vec![last_token];
        let mut current_taps = taps;
        let mut draft_cache = drafter.make_cache();
        let mut start = prompt_len;

        println!("\n--- 4B DFlash STATELESS verify (block_size={block_size}) ---");
        for round in 0..max_rounds {
            // a. Build draft block
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

            // b. Draft forward
            let t0 = Instant::now();
            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps, &mut draft_cache)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            let draft_ms = t0.elapsed().as_millis();
            total_draft_ms += draft_ms;
            crop_drafter_cache(&mut draft_cache, start);

            let draft_logits = target
                .forward_all_logits_from_hidden(&draft_hidden.index((.., 1.., ..)))
                .unwrap();
            let draft_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_ids]).unwrap();
            let draft_u32: Vec<u32> = draft_ids.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            // c. Build verify input
            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            // d. STATELESS verify — GDN state untouched, KV cache updated
            let t0 = Instant::now();
            let (verify_logits, _stateless_taps) = target
                .forward_with_taps_stateless(&verify_input, None, &mut kv_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();
            let verify_ms = t0.elapsed().as_millis();
            total_verify_ms += verify_ms;

            // e. Check acceptance
            let verify_am = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            // accept_prefix returns matched drafts + correction token
            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len(); // includes correction

            // f. Rollback ALL KV from stateless verify, then replay accepted
            //    This re-adds KV for accepted tokens and advances GDN state.
            rollback_kv_only(&mut kv_cache, verify_len);

            let replay_tokens: Vec<i32> = std::iter::once(last_token)
                .chain(accepted.iter().map(|&x| x as i32))
                .collect();
            let replay_input = Array::from_slice(&replay_tokens, &[1, replay_tokens.len() as i32]);

            let t0 = Instant::now();
            let (_replay_logits, replay_taps) = target
                .forward_with_taps(&replay_input, None, &mut kv_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&_replay_logits]).unwrap();
            let replay_ms = t0.elapsed().as_millis();
            total_replay_ms += replay_ms;

            // g. Use taps from replay (not stateless verify)
            current_taps = replay_taps;

            total_tokens += n_accepted;
            for &tok in &accepted {
                generated.push(tok as i32);
            }
            last_token = *accepted.last().unwrap() as i32;
            start += n_accepted as i32;

            let n_draft_accepted = n_accepted - 1; // exclude correction
            println!(
                "Round {round}: draft={draft_ms}ms verify={verify_ms}ms replay={replay_ms}ms accepted={n_draft_accepted}+1/{} draft={draft_flat:?}",
                block_size - 1
            );

            if generated.contains(&eos_token) {
                println!("EOS detected, stopping.");
                break;
            }
        }

        let total_ms = total_draft_ms + total_verify_ms + total_replay_ms;
        let tok_per_sec = if total_ms > 0 {
            total_tokens as f64 / (total_ms as f64 / 1000.0)
        } else {
            0.0
        };
        let rounds_run = max_rounds.min(generated.len());

        println!("\n--- 4B DFlash STATELESS Results ---");
        println!("Total tokens: {total_tokens}");
        println!(
            "Avg tok/round: {:.1}",
            total_tokens as f64 / rounds_run as f64
        );
        println!("Total draft: {total_draft_ms}ms, verify: {total_verify_ms}ms, replay: {total_replay_ms}ms");
        println!("Throughput: {tok_per_sec:.1} tok/s (no backup/restore overhead)");
    }

    /// DFlash 4B full loop with TAPE REPLAY — bstnxbt's approach.
    /// Normal (stateful) verify + tape recording. On full acceptance: zero work.
    /// On partial rejection: restore snapshot + replay tape[:n_accepted].
    #[test]
    #[ignore] // requires 4B target + drafter model weights on disk
    fn test_dflash_4b_tape_replay() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path =
            "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit";
        let drafter_base = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        );
        let snap_dir = std::fs::read_dir(drafter_base.join("snapshots"))
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .unwrap()
            .path();

        println!("Loading 4B target...");
        let mut target = load_qwen3_5_model(target_path).unwrap();
        println!("Loading 4B DFlash drafter...");
        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();

        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let eos_token: i32 = 248046;
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        // Prefill — stateful (normal)
        println!("Prefilling...");
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        {
            let mut ev: Vec<&Array> = vec![&prefill_logits];
            for t in &taps { ev.push(t); }
            for lc in kv_cache.iter().flatten() {
                match lc {
                    crate::qwen3_next::LayerCache::KV(kv) => ev.extend(kv.eval_targets()),
                    crate::qwen3_next::LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state { ev.push(s); }
                        if let Some(ref c) = ac.conv_state { ev.push(c); }
                    }
                }
            }
            mlx_rs::transforms::eval(ev).unwrap();
        }

        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
        let mut last_token: i32 = *am_flat.last().unwrap() as i32;

        let max_rounds = 20;
        let mut total_tokens = 0usize;
        let mut total_draft_ms = 0u128;
        let mut total_verify_ms = 0u128;
        let mut total_replay_ms = 0u128;
        let mut n_full_accept = 0usize;
        let mut generated: Vec<i32> = vec![last_token];
        let mut current_taps = taps;
        let mut draft_cache = drafter.make_cache();
        let mut start = prompt_len;

        println!("\n--- 4B DFlash TAPE REPLAY (block_size={block_size}) ---");
        for round in 0..max_rounds {
            // a. Draft
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

            let t0 = Instant::now();
            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps, &mut draft_cache)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            let draft_ms = t0.elapsed().as_millis();
            total_draft_ms += draft_ms;
            crop_drafter_cache(&mut draft_cache, start);

            let draft_logits = target
                .forward_all_logits_from_hidden(&draft_hidden.index((.., 1.., ..)))
                .unwrap();
            let draft_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_ids]).unwrap();
            let draft_u32: Vec<u32> = draft_ids.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            // b. Build verify input
            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            // c. Snapshot GDN state BEFORE verify
            let snapshots: Vec<(Option<Array>, Option<Array>, i32)> = kv_cache.iter()
                .map(|lc| match lc {
                    Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                        ac.eval_arrays().unwrap();
                        (ac.conv_state.clone(), ac.ssm_state.clone(), ac.offset)
                    }
                    _ => (None, None, 0),
                })
                .collect();

            // d. TAPE-RECORDING verify — state IS updated, tape IS recorded
            let t0 = Instant::now();
            let (verify_logits, verify_taps, layer_tapes) = target
                .forward_with_taps_tape(&verify_input, None, &mut kv_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();
            let verify_ms = t0.elapsed().as_millis();
            total_verify_ms += verify_ms;

            // e. Check acceptance — accept_prefix returns matched drafts + correction
            let verify_am = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len(); // includes correction token
            let n_draft_accepted = n_accepted - 1;
            let keep = n_accepted as i32; // anchor + accepted drafts + correction = n_accepted

            // f. Handle acceptance
            let t0 = Instant::now();
            if keep < verify_len {
                // Partial rejection — restore + replay accepted from tape
                let kv_rollback = verify_len - keep;
                target.replay_tape_rollback(
                    &layer_tapes, &mut kv_cache,
                    keep, // replay keep tokens (anchor + accepted drafts)
                    kv_rollback,
                ).unwrap();
                // Batch-eval all replayed GDN states in one call
                let replay_states: Vec<&Array> = kv_cache.iter()
                    .filter_map(|lc| match lc {
                        Some(crate::qwen3_next::LayerCache::Arrays(ac)) => ac.ssm_state.as_ref(),
                        _ => None,
                    })
                    .collect();
                if !replay_states.is_empty() {
                    mlx_rs::transforms::eval(replay_states).unwrap();
                }
            } else {
                // Full acceptance — state already correct, zero extra work!
                n_full_accept += 1;
            }
            let replay_ms = t0.elapsed().as_millis();
            total_replay_ms += replay_ms;

            // g. Taps from verify, sliced to accepted tokens
            current_taps = verify_taps.into_iter()
                .map(|tap| tap.index((.., ..keep, ..)))
                .collect();

            total_tokens += n_accepted;
            for &tok in &accepted {
                generated.push(tok as i32);
            }
            last_token = *accepted.last().unwrap() as i32;
            start += n_accepted as i32;

            let full_str = if keep == verify_len { " [FULL]" } else { "" };
            println!(
                "Round {round}: draft={draft_ms}ms verify={verify_ms}ms replay={replay_ms}ms accepted={n_draft_accepted}+1/{}{full_str}",
                block_size - 1
            );

            if generated.contains(&eos_token) {
                println!("EOS detected, stopping.");
                break;
            }
        }

        let total_ms = total_draft_ms + total_verify_ms + total_replay_ms;
        let tok_per_sec = if total_ms > 0 {
            total_tokens as f64 / (total_ms as f64 / 1000.0)
        } else {
            0.0
        };
        let rounds_run = max_rounds.min(generated.len());

        println!("\n--- 4B DFlash TAPE REPLAY Results ---");
        println!("Total tokens: {total_tokens}");
        println!(
            "Avg acceptance: {:.1} tok/round",
            total_tokens as f64 / rounds_run as f64
        );
        println!("Full acceptance rounds: {n_full_accept}/{rounds_run}");
        println!("Total draft: {total_draft_ms}ms, verify: {total_verify_ms}ms, replay: {total_replay_ms}ms");
        println!("Throughput: {tok_per_sec:.1} tok/s");
    }

    /// Lossless verification: DFlash output must match AR baseline token-for-token.
    /// Runs both AR decode and DFlash tape-replay on the same 4B-4bit model with
    /// the same prompt, then asserts identical generated sequences.
    #[test]
    #[ignore] // requires 4B target + drafter model weights on disk
    fn test_dflash_4b_lossless_vs_ar() {
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use std::time::Instant;

        let target_path =
            "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit";
        let drafter_base = Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        );
        let snap_dir = std::fs::read_dir(drafter_base.join("snapshots"))
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .unwrap()
            .path();

        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let eos_token: i32 = 248046;
        let max_tokens = 50;

        // ── Phase 1: AR baseline ──────────────────────────────────────
        println!("=== AR Baseline ===");
        let mut target = load_qwen3_5_model(target_path).unwrap();
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let prefill_logits = target
            .forward(&input_ids, None, &mut kv_cache)
            .unwrap();
        mlx_rs::transforms::eval([&prefill_logits]).unwrap();

        let mut ar_tokens: Vec<u32> = Vec::new();
        let first = {
            let am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
            let flat: Vec<u32> = am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            *flat.last().unwrap()
        };
        ar_tokens.push(first);
        let mut next_token = first as i32;

        let t0 = Instant::now();
        while ar_tokens.len() < max_tokens {
            if next_token == eos_token { break; }
            let tok_arr = Array::from_slice(&[next_token], &[1, 1]);
            let logits = target.forward(&tok_arr, None, &mut kv_cache).unwrap();
            mlx_rs::transforms::eval([&logits]).unwrap();
            let am = mlx_rs::argmax_axis!(logits, -1).unwrap();
            let tok: u32 = am.reshape(&[-1]).unwrap().as_slice::<u32>()[0];
            ar_tokens.push(tok);
            next_token = tok as i32;
        }
        let ar_ms = t0.elapsed().as_millis();
        println!("AR: {} tokens in {}ms, first 20: {:?}", ar_tokens.len(), ar_ms, &ar_tokens[..ar_tokens.len().min(20)]);

        // Reset model state
        drop(kv_cache);
        drop(target);

        // ── Phase 2: DFlash tape-replay ──────────────────────────────
        println!("\n=== DFlash Tape-Replay ===");
        let mut target = load_qwen3_5_model(target_path).unwrap();
        let mut drafter = load_dflash_drafter(&snap_dir).unwrap();
        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        {
            let mut ev: Vec<&Array> = vec![&prefill_logits];
            for t in &taps { ev.push(t); }
            for lc in kv_cache.iter().flatten() {
                match lc {
                    crate::qwen3_next::LayerCache::KV(kv) => ev.extend(kv.eval_targets()),
                    crate::qwen3_next::LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state { ev.push(s); }
                        if let Some(ref c) = ac.conv_state { ev.push(c); }
                    }
                }
            }
            mlx_rs::transforms::eval(ev).unwrap();
        }

        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
        let mut last_token: i32 = *am_flat.last().unwrap() as i32;

        let mut dflash_tokens: Vec<u32> = vec![last_token as u32];
        let mut current_taps = taps;
        let mut draft_cache = drafter.make_cache();
        let mut start = prompt_len;
        let mut rounds = 0usize;

        let t0 = Instant::now();
        while dflash_tokens.len() < max_tokens {
            if last_token == eos_token { break; }

            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
            let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

            let draft_hidden = drafter
                .forward(&noise_embedding, &current_taps, &mut draft_cache)
                .unwrap();
            mlx_rs::transforms::eval([&draft_hidden]).unwrap();
            crop_drafter_cache(&mut draft_cache, start);

            let draft_logits = target
                .forward_all_logits_from_hidden(&draft_hidden.index((.., 1.., ..)))
                .unwrap();
            let draft_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
            mlx_rs::transforms::eval([&draft_ids]).unwrap();
            let draft_u32: Vec<u32> = draft_ids.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_flat);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            // Snapshot before tape-recording verify
            let snapshots: Vec<(Option<Array>, Option<Array>, i32)> = kv_cache.iter()
                .map(|lc| match lc {
                    Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                        ac.eval_arrays().unwrap();
                        (ac.conv_state.clone(), ac.ssm_state.clone(), ac.offset)
                    }
                    _ => (None, None, 0),
                })
                .collect();

            let (verify_logits, verify_taps, layer_tapes) = target
                .forward_with_taps_tape(&verify_input, None, &mut kv_cache, &tap_layers)
                .unwrap();
            mlx_rs::transforms::eval([&verify_logits]).unwrap();

            let verify_am = mlx_rs::argmax_axis!(verify_logits, -1).unwrap();
            let verify_flat: Vec<u32> = verify_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            let accepted = accept_prefix(&draft_u32, &verify_flat);
            let n_accepted = accepted.len(); // includes correction
            let keep = n_accepted as i32;

            if keep < verify_len {
                let kv_rollback = verify_len - keep;
                target.replay_tape_rollback(
                    &layer_tapes, &mut kv_cache,
                    keep, kv_rollback,
                ).unwrap();
                for lc in kv_cache.iter() {
                    if let Some(crate::qwen3_next::LayerCache::Arrays(ac)) = lc {
                        if let Some(ref s) = ac.ssm_state {
                            mlx_rs::transforms::eval([s]).unwrap();
                        }
                    }
                }
            }

            current_taps = verify_taps.into_iter()
                .map(|tap| tap.index((.., ..keep, ..)))
                .collect();

            for &tok in &accepted {
                dflash_tokens.push(tok);
            }
            last_token = *accepted.last().unwrap() as i32;
            start += n_accepted as i32;
            rounds += 1;

            // DIAGNOSTIC: S=1 decode via stateless GDN (no GDN mutation)
            {
                let probe = Array::from_slice(&[last_token], &[1, 1]);
                let gdn_snap_diag = crate::dflash::GdnStateBackup::save(&kv_cache).unwrap();
                let probe_logits = target.forward(&probe, None, &mut kv_cache).unwrap();
                mlx_rs::transforms::eval([&probe_logits]).unwrap();
                let probe_tok: u32 = mlx_rs::argmax_axis!(probe_logits, -1).unwrap()
                    .reshape(&[-1]).unwrap().as_slice::<u32>()[0];
                let pos = dflash_tokens.len();
                let ar_at_pos = ar_tokens.get(pos).copied();
                if ar_at_pos != Some(probe_tok) {
                    println!("  !! Round {rounds} DIVERGES: pos={pos} probe={probe_tok} ar={ar_at_pos:?} last_tok={last_token} accepted={n_accepted}");
                } else {
                    println!("  Round {rounds} OK: pos={pos} tok={probe_tok}");
                }
                // Restore: rollback KV by 1, restore GDN state
                crate::dflash::GdnStateBackup::restore_and_rollback(&gdn_snap_diag, &mut kv_cache, 1);
                for lc in kv_cache.iter() {
                    if let Some(crate::qwen3_next::LayerCache::Arrays(ac)) = lc {
                        if let Some(ref s) = ac.ssm_state { mlx_rs::transforms::eval([s]).unwrap(); }
                    }
                }
            }
        }
        let dflash_ms = t0.elapsed().as_millis();
        println!("DFlash: {} tokens in {}ms ({} rounds), first 20: {:?}",
            dflash_tokens.len(), dflash_ms, rounds,
            &dflash_tokens[..dflash_tokens.len().min(20)]);

        // ── Phase 3: Compare ─────────────────────────────────────────
        let cmp_len = ar_tokens.len().min(dflash_tokens.len());
        let mut first_mismatch = None;
        for i in 0..cmp_len {
            if ar_tokens[i] != dflash_tokens[i] {
                first_mismatch = Some(i);
                break;
            }
        }

        if let Some(pos) = first_mismatch {
            println!("\n❌ MISMATCH at position {pos}:");
            println!("  AR:     {:?}", &ar_tokens[pos.saturating_sub(2)..ar_tokens.len().min(pos+5)]);
            println!("  DFlash: {:?}", &dflash_tokens[pos.saturating_sub(2)..dflash_tokens.len().min(pos+5)]);
            panic!("DFlash output diverges from AR at token {pos}: AR={} DFlash={}", ar_tokens[pos], dflash_tokens[pos]);
        } else if ar_tokens.len() != dflash_tokens.len() {
            println!("\n⚠ Length mismatch: AR={} DFlash={} (first {} tokens match)",
                ar_tokens.len(), dflash_tokens.len(), cmp_len);
        } else {
            println!("\n✅ LOSSLESS: {} tokens match exactly", cmp_len);
        }
    }

    /// Test that batch-forward (S=N) produces same logits as N sequential AR forwards.
    /// Isolates whether GDN multi-token processing matches 1-at-a-time.
    #[test]
    #[ignore]
    fn test_batch_vs_sequential_parity() {
        use crate::qwen3_next::load_qwen3_5_model;

        let target_path =
            "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit";
        let mut target = load_qwen3_5_model(target_path).unwrap();

        // Use first 10 AR-generated tokens as a verify-like block
        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13, 248046, 198,
            248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        // Generate 4 tokens via AR to get a verify block
        let mut kv_cache_ar: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let logits = target.forward(&input_ids, None, &mut kv_cache_ar).unwrap();
        mlx_rs::transforms::eval([&logits]).unwrap();

        let am = mlx_rs::argmax_axis!(logits, -1).unwrap();
        let flat: Vec<u32> = am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
        let first_tok = *flat.last().unwrap() as i32;

        let mut ar_tokens = vec![first_tok];
        let mut next = first_tok;
        for _ in 0..3 {
            let tok_arr = Array::from_slice(&[next], &[1, 1]);
            let l = target.forward(&tok_arr, None, &mut kv_cache_ar).unwrap();
            mlx_rs::transforms::eval([&l]).unwrap();
            let a = mlx_rs::argmax_axis!(l, -1).unwrap();
            let t: u32 = a.reshape(&[-1]).unwrap().as_slice::<u32>()[0];
            ar_tokens.push(t as i32);
            next = t as i32;
        }
        // Get logit at final position from AR
        let final_tok_arr = Array::from_slice(&[next], &[1, 1]);
        let final_logits_ar = target.forward(&final_tok_arr, None, &mut kv_cache_ar).unwrap();
        mlx_rs::transforms::eval([&final_logits_ar]).unwrap();
        let ar_final = mlx_rs::argmax_axis!(final_logits_ar, -1).unwrap();
        let ar_final_tok: u32 = ar_final.reshape(&[-1]).unwrap().as_slice::<u32>()[0];

        println!("AR tokens: {:?}, next: {ar_final_tok}", ar_tokens);

        // Now redo: prefill + batch forward of the same 4 tokens + 1
        drop(kv_cache_ar);
        let mut target2 = load_qwen3_5_model(target_path).unwrap();
        let mut kv_cache_batch: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let logits2 = target2.forward(&input_ids, None, &mut kv_cache_batch).unwrap();
        mlx_rs::transforms::eval([&logits2]).unwrap();

        // Batch forward all 5 tokens at once
        let mut all_tokens = ar_tokens.clone();
        all_tokens.push(next);
        let batch_ids = Array::from_slice(&all_tokens, &[1, all_tokens.len() as i32]);
        let batch_logits = target2.forward(&batch_ids, None, &mut kv_cache_batch).unwrap();
        mlx_rs::transforms::eval([&batch_logits]).unwrap();

        // Compare last-position logit
        let batch_am = mlx_rs::argmax_axis!(batch_logits, -1).unwrap();
        let batch_flat: Vec<u32> = batch_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
        let batch_final_tok = *batch_flat.last().unwrap();

        println!("Batch last-position argmax: {batch_final_tok}");
        println!("AR    last-position argmax: {ar_final_tok}");

        if ar_final_tok == batch_final_tok {
            println!("✅ PARITY: batch and sequential produce same output");
        } else {
            // Also check intermediate positions
            for (i, (ar_tok, batch_tok)) in ar_tokens.iter().zip(batch_flat.iter()).enumerate() {
                if *ar_tok as u32 != *batch_tok {
                    println!("Divergence at intermediate position {i}: AR={ar_tok} batch={batch_tok}");
                }
            }
            panic!("❌ DIVERGENCE: batch={batch_final_tok} vs AR={ar_final_tok}");
        }
    }
}
