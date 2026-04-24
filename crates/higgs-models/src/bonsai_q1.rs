//! Bonsai-Q1 target-capable engine: packed 1.25-bpw weight storage.
//!
//! Unlike `DiffusionEngine::load_q1` which dequantizes to fp32 at load (32 GB
//! residency on 8B), this engine holds MLX's `Q1_0_g128` affine encoding
//! verbatim: `w[row, col] = scales[row, col/128] * bit(col) + biases[row,
//! col/128]`. Dequant happens inline inside the matmul kernel (Metal in P2).
//!
//! Residency: ~1.25 GB for Bonsai-8B-mlx-1bit, ~260 MB for Bonsai-1.7B-mlx-1bit.
//!
//! Scope: P1 loads weights and exposes layer count. Forward is P2/P3.

#![allow(clippy::too_many_arguments)]

use half::f16;
use std::path::Path;

use mlx_rs::{Array, Dtype, error::Exception, fast, ops, ops::indexing::IndexOp};
use safetensors::SafeTensors;

use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    error::ModelError,
    utils::{cached_scaled_dot_product_attention, create_attention_mask},
    yarn::{apply_yarn_rope, compute_yarn_freqs, yarn_get_mscale},
};

/// Load and materialize a Bonsai-Q1 model from `model_dir` onto the GPU.
///
/// Adapts [`BonsaiQ1Engine::load`]'s `Result<_, String>` into [`ModelError`] so
/// the engine surface in `higgs-engine::model_loader` can route it through the
/// same `EngineError::Model` path used by all other architectures.
pub fn load_bonsai_q1<P: AsRef<Path>>(model_dir: P) -> Result<BonsaiQ1Gpu, ModelError> {
    let engine = BonsaiQ1Engine::load(model_dir).map_err(ModelError::ShapeMismatch)?;
    engine.to_gpu().map_err(ModelError::Mlx)
}

pub const GROUP_SIZE: usize = 128;
const BITS: i32 = 1;
const GROUP_SIZE_I32: i32 = GROUP_SIZE as i32;

/// Packed 1-bit linear layer with affine per-group dequant.
///
/// Layout (matches MLX 1-bit QuantizedLinear, PrismML fork):
///   - `w_packed`: `[out_features, in_features/32]` u32, bit `col%32` of word
///     `col/32` is the raw 1-bit weight for column `col`.
///   - `scales`, `biases`: `[out_features, in_features/128]` f16, one per group
///     of 128 input columns.
///
/// Effective: 1 bit/weight + 32 bits/group / 128 weights = **1.25 bpw**.
pub struct PackedQ1Linear {
    pub w_packed: Vec<u32>,
    pub scales: Vec<f16>,
    pub biases: Vec<f16>,
    pub out_features: usize,
    pub in_features: usize,
}

impl PackedQ1Linear {
    pub fn resident_bytes(&self) -> usize {
        self.w_packed.len() * 4 + self.scales.len() * 2 + self.biases.len() * 2
    }

    /// Dequantize a single row to fp32 (reference path for correctness tests).
    ///
    /// Not used on the hot path — P2 replaces this with a Metal kernel that
    /// fuses dequant into the matmul.
    pub fn dequant_row_to_fp32(&self, row: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.in_features);
        let n_groups = self.in_features / GROUP_SIZE;
        let packed_cols = self.in_features / 32;
        let w_row = &self.w_packed[row * packed_cols..(row + 1) * packed_cols];
        let s_row = &self.scales[row * n_groups..(row + 1) * n_groups];
        let b_row = &self.biases[row * n_groups..(row + 1) * n_groups];
        for col in 0..self.in_features {
            let word = w_row[col / 32];
            let bit = ((word >> (col % 32)) & 1) as f32;
            let group = col / GROUP_SIZE;
            out[col] = s_row[group].to_f32() * bit + b_row[group].to_f32();
        }
    }
}

pub struct BonsaiQ1LayerWeights {
    pub q_proj: PackedQ1Linear,
    pub k_proj: PackedQ1Linear,
    pub v_proj: PackedQ1Linear,
    pub o_proj: PackedQ1Linear,
    pub gate_proj: PackedQ1Linear,
    pub up_proj: PackedQ1Linear,
    pub down_proj: PackedQ1Linear,
    pub q_norm: Vec<f16>,
    pub k_norm: Vec<f16>,
    pub input_norm: Vec<f16>,
    pub post_attn_norm: Vec<f16>,
}

impl BonsaiQ1LayerWeights {
    pub fn resident_bytes(&self) -> usize {
        self.q_proj.resident_bytes()
            + self.k_proj.resident_bytes()
            + self.v_proj.resident_bytes()
            + self.o_proj.resident_bytes()
            + self.gate_proj.resident_bytes()
            + self.up_proj.resident_bytes()
            + self.down_proj.resident_bytes()
            + (self.q_norm.len()
                + self.k_norm.len()
                + self.input_norm.len()
                + self.post_attn_norm.len())
                * 2
    }
}

#[derive(Debug, Clone)]
pub struct BonsaiQ1Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// YARN scaling factor if present (Bonsai-8B uses `factor=4.0, original=16384`).
    pub rope_yarn_factor: Option<f64>,
    pub rope_original_max_seq: Option<usize>,
    pub tie_word_embeddings: bool,
}

pub struct BonsaiQ1Engine {
    pub config: BonsaiQ1Config,
    pub layers: Vec<BonsaiQ1LayerWeights>,
    /// Token embedding stored packed (dequants inline at embed lookup time).
    pub embed: PackedQ1Linear,
    /// Untied LM head for 8B (`tie_word_embeddings: false`). None for 1.7B.
    pub lm_head: Option<PackedQ1Linear>,
    pub final_norm: Vec<f16>,
}

impl BonsaiQ1Engine {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn resident_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(BonsaiQ1LayerWeights::resident_bytes)
            .sum();
        let lm_head_bytes = self
            .lm_head
            .as_ref()
            .map_or(0, PackedQ1Linear::resident_bytes);
        layer_bytes + self.embed.resident_bytes() + lm_head_bytes + self.final_norm.len() * 2
    }

    /// Load from a HuggingFace directory containing `config.json` +
    /// `model.safetensors` in MLX 1-bit affine-quant format.
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let dir = model_dir.as_ref();

        let cfg_txt = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let cfg: serde_json::Value =
            serde_json::from_str(&cfg_txt).map_err(|e| format!("config.json parse: {e}"))?;

        let u64_of = |k: &str| -> Result<u64, String> {
            cfg[k]
                .as_u64()
                .ok_or_else(|| format!("config.json missing u64 '{k}'"))
        };
        let hidden = u64_of("hidden_size")? as usize;
        let heads = u64_of("num_attention_heads")? as usize;
        let kv_heads = u64_of("num_key_value_heads")? as usize;
        let head_dim = cfg["head_dim"].as_u64().map(|v| v as usize).unwrap_or(128);
        let inter = u64_of("intermediate_size")? as usize;
        let layers_n = u64_of("num_hidden_layers")? as usize;
        let vocab = u64_of("vocab_size")? as usize;

        let rms_norm_eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let rope_theta = cfg["rope_theta"].as_f64().unwrap_or(1_000_000.0);
        let tie_word_embeddings = cfg["tie_word_embeddings"].as_bool().unwrap_or(false);

        let (rope_yarn_factor, rope_original_max_seq) = cfg
            .get("rope_scaling")
            .and_then(|rs| {
                if rs.get("rope_type").and_then(|v| v.as_str()) == Some("yarn") {
                    let f = rs.get("factor").and_then(serde_json::Value::as_f64);
                    let o = rs
                        .get("original_max_position_embeddings")
                        .and_then(serde_json::Value::as_u64)
                        .map(|v| v as usize);
                    Some((f, o))
                } else {
                    None
                }
            })
            .unwrap_or((None, None));

        let quant = cfg
            .get("quantization")
            .ok_or("missing quantization block")?;
        let q_bits = quant.get("bits").and_then(serde_json::Value::as_u64);
        let q_group = quant.get("group_size").and_then(serde_json::Value::as_u64);
        if q_bits != Some(1) || q_group != Some(GROUP_SIZE as u64) {
            return Err(format!(
                "expected quantization {{bits:1, group_size:{GROUP_SIZE}}}, got bits={q_bits:?} \
                 group_size={q_group:?}"
            ));
        }

        let st_path = dir.join("model.safetensors");
        let st_data = std::fs::read(&st_path).map_err(|e| format!("read safetensors: {e}"))?;
        let tensors = SafeTensors::deserialize(&st_data)
            .map_err(|e| format!("deserialize safetensors: {e}"))?;

        let config = BonsaiQ1Config {
            hidden,
            layers: layers_n,
            heads,
            kv_heads,
            head_dim,
            inter,
            vocab,
            rms_norm_eps,
            rope_theta,
            rope_yarn_factor,
            rope_original_max_seq,
            tie_word_embeddings,
        };

        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let embed = load_packed(
            &tensors,
            "model.embed_tokens",
            vocab,
            hidden,
            "embed_tokens",
        )?;
        let lm_head = if tie_word_embeddings {
            None
        } else {
            Some(load_packed(&tensors, "lm_head", vocab, hidden, "lm_head")?)
        };
        let final_norm = load_f16(&tensors, "model.norm.weight")?;
        if final_norm.len() != hidden {
            return Err(format!(
                "final_norm len {} != hidden {hidden}",
                final_norm.len()
            ));
        }

        let mut layers = Vec::with_capacity(layers_n);
        for i in 0..layers_n {
            let p = format!("model.layers.{i}");
            let attn = format!("{p}.self_attn");
            let mlp = format!("{p}.mlp");

            let layer = BonsaiQ1LayerWeights {
                q_proj: load_packed(&tensors, &format!("{attn}.q_proj"), q_dim, hidden, "q_proj")?,
                k_proj: load_packed(
                    &tensors,
                    &format!("{attn}.k_proj"),
                    kv_dim,
                    hidden,
                    "k_proj",
                )?,
                v_proj: load_packed(
                    &tensors,
                    &format!("{attn}.v_proj"),
                    kv_dim,
                    hidden,
                    "v_proj",
                )?,
                o_proj: load_packed(&tensors, &format!("{attn}.o_proj"), hidden, q_dim, "o_proj")?,
                gate_proj: load_packed(
                    &tensors,
                    &format!("{mlp}.gate_proj"),
                    inter,
                    hidden,
                    "gate_proj",
                )?,
                up_proj: load_packed(
                    &tensors,
                    &format!("{mlp}.up_proj"),
                    inter,
                    hidden,
                    "up_proj",
                )?,
                down_proj: load_packed(
                    &tensors,
                    &format!("{mlp}.down_proj"),
                    hidden,
                    inter,
                    "down_proj",
                )?,
                q_norm: load_f16(&tensors, &format!("{attn}.q_norm.weight"))?,
                k_norm: load_f16(&tensors, &format!("{attn}.k_norm.weight"))?,
                input_norm: load_f16(&tensors, &format!("{p}.input_layernorm.weight"))?,
                post_attn_norm: load_f16(
                    &tensors,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
            };
            layers.push(layer);
        }

        let engine = Self {
            config,
            layers,
            embed,
            lm_head,
            final_norm,
        };
        let resident_mb = engine.resident_bytes() as f64 / (1024.0 * 1024.0);
        eprintln!(
            "BonsaiQ1Engine::load: {}L hidden={} heads={}/{} head_dim={} inter={} vocab={} \
             tied_embed={} packed_resident={:.1}MB",
            engine.config.layers,
            engine.config.hidden,
            engine.config.heads,
            engine.config.kv_heads,
            engine.config.head_dim,
            engine.config.inter,
            engine.config.vocab,
            engine.config.tie_word_embeddings,
            resident_mb,
        );
        Ok(engine)
    }
}

// ---------------------------------------------------------------------------
// GPU-ready mirror — built once from the packed engine.
// ---------------------------------------------------------------------------

/// MLX-resident 1-bit linear: weight as uint32 packed, scales/biases as f16,
/// same shape as `PackedQ1Linear` but ready for `ops::quantized_matmul`.
pub struct BonsaiQ1GpuLinear {
    pub w: Array,
    pub scales: Array,
    pub biases: Array,
    pub out_features: i32,
    pub in_features: i32,
}

impl BonsaiQ1GpuLinear {
    fn from_packed(p: &PackedQ1Linear) -> Result<Self, Exception> {
        let out = i32::try_from(p.out_features)
            .map_err(|_| Exception::custom("out_features overflows i32"))?;
        let inf = i32::try_from(p.in_features)
            .map_err(|_| Exception::custom("in_features overflows i32"))?;
        let packed_cols = inf / 32;
        let n_groups = inf / GROUP_SIZE_I32;

        let w = Array::from_slice(&p.w_packed, &[out, packed_cols]);
        let scales_f32: Vec<f32> = p.scales.iter().map(|h| h.to_f32()).collect();
        let biases_f32: Vec<f32> = p.biases.iter().map(|h| h.to_f32()).collect();
        let scales = Array::from_slice(&scales_f32, &[out, n_groups]).as_dtype(Dtype::Float16)?;
        let biases = Array::from_slice(&biases_f32, &[out, n_groups]).as_dtype(Dtype::Float16)?;

        Ok(Self {
            w,
            scales,
            biases,
            out_features: out,
            in_features: inf,
        })
    }

    /// `y = x @ dequant(w, scales, biases).T` via fused bits=1 qmm.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        ops::quantized_matmul(
            x,
            &self.w,
            &self.scales,
            &self.biases,
            true,
            GROUP_SIZE_I32,
            BITS,
        )
    }
}

pub struct BonsaiQ1GpuLayer {
    pub q_proj: BonsaiQ1GpuLinear,
    pub k_proj: BonsaiQ1GpuLinear,
    pub v_proj: BonsaiQ1GpuLinear,
    pub o_proj: BonsaiQ1GpuLinear,
    pub gate_proj: BonsaiQ1GpuLinear,
    pub up_proj: BonsaiQ1GpuLinear,
    pub down_proj: BonsaiQ1GpuLinear,
    pub q_norm: Array,
    pub k_norm: Array,
    pub input_norm: Array,
    pub post_attn_norm: Array,
}

pub struct BonsaiQ1Gpu {
    pub config: BonsaiQ1Config,
    pub layers: Vec<BonsaiQ1GpuLayer>,
    pub embed: BonsaiQ1GpuLinear,
    pub lm_head: Option<BonsaiQ1GpuLinear>,
    pub final_norm: Array,
    /// YARN-scaled RoPE frequencies (per `head_dim/2`). None if no YARN.
    pub yarn_freqs: Option<Array>,
    pub yarn_mscale: f32,
    pub attention_scale: f32,
}

fn f16_vec_to_array(weights: &[f16]) -> Result<Array, Exception> {
    let f32s: Vec<f32> = weights.iter().map(|h| h.to_f32()).collect();
    let len =
        i32::try_from(weights.len()).map_err(|_| Exception::custom("norm len overflows i32"))?;
    Array::from_slice(&f32s, &[len]).as_dtype(Dtype::Float16)
}

impl BonsaiQ1Engine {
    /// Consume the packed engine and materialize MLX arrays.
    ///
    /// Frees the `Vec<u32>` / `Vec<f16>` residency once copied to MLX.
    pub fn to_gpu(self) -> Result<BonsaiQ1Gpu, Exception> {
        let mut gpu_layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            gpu_layers.push(BonsaiQ1GpuLayer {
                q_proj: BonsaiQ1GpuLinear::from_packed(&layer.q_proj)?,
                k_proj: BonsaiQ1GpuLinear::from_packed(&layer.k_proj)?,
                v_proj: BonsaiQ1GpuLinear::from_packed(&layer.v_proj)?,
                o_proj: BonsaiQ1GpuLinear::from_packed(&layer.o_proj)?,
                gate_proj: BonsaiQ1GpuLinear::from_packed(&layer.gate_proj)?,
                up_proj: BonsaiQ1GpuLinear::from_packed(&layer.up_proj)?,
                down_proj: BonsaiQ1GpuLinear::from_packed(&layer.down_proj)?,
                q_norm: f16_vec_to_array(&layer.q_norm)?,
                k_norm: f16_vec_to_array(&layer.k_norm)?,
                input_norm: f16_vec_to_array(&layer.input_norm)?,
                post_attn_norm: f16_vec_to_array(&layer.post_attn_norm)?,
            });
        }

        let embed = BonsaiQ1GpuLinear::from_packed(&self.embed)?;
        let lm_head = self
            .lm_head
            .as_ref()
            .map(BonsaiQ1GpuLinear::from_packed)
            .transpose()?;
        let final_norm = f16_vec_to_array(&self.final_norm)?;

        // YARN precompute.
        let head_dim_i = i32::try_from(self.config.head_dim)
            .map_err(|_| Exception::custom("head_dim overflows i32"))?;
        let base = self.config.rope_theta as f32;
        let (yarn_freqs, yarn_mscale) = match self.config.rope_yarn_factor {
            Some(factor) if factor > 1.0 => {
                let orig = i32::try_from(
                    self.config
                        .rope_original_max_seq
                        .unwrap_or(self.config.hidden),
                )
                .map_err(|_| Exception::custom("orig_max_seq overflows i32"))?;
                let factor_f = factor as f32;
                let freqs = compute_yarn_freqs(head_dim_i, base, factor_f, orig, 32.0, 1.0);
                (Some(freqs), yarn_get_mscale(factor_f, 1.0))
            }
            _ => (None, 1.0),
        };

        let head_dim_f = head_dim_i as f32;
        let attention_scale = head_dim_f.sqrt().recip();

        Ok(BonsaiQ1Gpu {
            config: self.config,
            layers: gpu_layers,
            embed,
            lm_head,
            final_norm,
            yarn_freqs,
            yarn_mscale,
            attention_scale,
        })
    }
}

impl BonsaiQ1Gpu {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Gather embedding rows for a token-ID tensor.
    ///
    /// Tries the MLX GPU dequantize path first (`take_axis` + `ops::dequantize`);
    /// if the bits=1 dequantize kernel is missing, falls back to CPU row-dequant
    /// from the original packed storage — but since `to_gpu` has already
    /// dropped that, we instead materialize via a full quantized_matmul against
    /// a one-hot. For small B*L that's acceptable; we expect the GPU path to
    /// succeed on the PrismML-forked MLX.
    fn embed_rows(&self, ids: &Array) -> Result<Array, Exception> {
        let shape = ids.shape().to_vec();
        let flat = ids.flatten(None, None)?;
        let w = self.embed.w.take_axis(&flat, 0)?;
        let s = self.embed.scales.take_axis(&flat, 0)?;
        let b = self.embed.biases.take_axis(&flat, 0)?;
        let out = ops::dequantize(&w, &s, &b, GROUP_SIZE_I32, BITS)?;
        let mut ret_shape: Vec<i32> = shape;
        ret_shape.push(-1);
        out.reshape(&ret_shape)
    }

    fn apply_rope(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let head_dim = i32::try_from(self.config.head_dim)
            .map_err(|_| Exception::custom("head_dim overflows i32"))?;
        let offset_array = Array::from_int(offset);
        apply_yarn_rope(
            x,
            head_dim,
            self.config.rope_theta as f32,
            self.yarn_freqs.as_ref(),
            self.yarn_mscale,
            &offset_array,
            false, // Qwen3 layout
        )
    }

    /// Run the decoder trunk and return final-normed hidden `[B, T, hidden]`.
    /// Shared body for `forward` (last-position logits) and
    /// `forward_all_logits` (all-position logits, used by spec-decode verify).
    #[allow(non_snake_case)]
    fn forward_trunk(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("inputs must have >= 2 dims"))?;
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("inputs must have >= 2 dims"))?;

        if cache.is_empty() {
            *cache = (0..self.layers.len())
                .map(|_| Some(SteppingKeyValueCache::new()))
                .collect();
        } else if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "cache len {} != num_layers {}",
                cache.len(),
                self.layers.len()
            )));
        }

        let mut h = self.embed_rows(inputs)?; // [B, L, hidden]

        let mask = create_attention_mask(&h, cache, None)?;

        let heads = i32::try_from(self.config.heads)
            .map_err(|_| Exception::custom("heads overflows i32"))?;
        let kv_heads = i32::try_from(self.config.kv_heads)
            .map_err(|_| Exception::custom("kv_heads overflows i32"))?;
        let rms_eps = self.config.rms_norm_eps;

        for (layer, layer_cache) in self.layers.iter().zip(cache.iter_mut()) {
            let normed = fast::rms_norm(&h, &layer.input_norm, rms_eps)?;

            // q/k/v projections
            let q = layer.q_proj.forward(&normed)?;
            let k = layer.k_proj.forward(&normed)?;
            let v = layer.v_proj.forward(&normed)?;

            // Reshape to [B, L, n_heads, head_dim] then transpose to [B, n_heads, L, head_dim].
            let q = q
                .reshape(&[B, T, heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = k
                .reshape(&[B, T, kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = v
                .reshape(&[B, T, kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;

            // QK-norm along last axis (per head_dim), then RoPE.
            let q = fast::rms_norm(&q, &layer.q_norm, rms_eps)?;
            let k = fast::rms_norm(&k, &layer.k_norm, rms_eps)?;

            let offset = layer_cache.as_ref().map_or(0, KeyValueCache::offset);
            let q = self.apply_rope(&q, offset)?;
            let k = self.apply_rope(&k, offset)?;

            let mask_arr = match &mask {
                Some(crate::utils::AttentionMask::Array(a)) => Some(a),
                _ => None,
            };
            let mask_arr_opt: Option<&Array> = mask_arr;

            let attn_out = match layer_cache.as_mut() {
                Some(c) => cached_scaled_dot_product_attention(
                    q,
                    c,
                    k,
                    v,
                    self.attention_scale,
                    mask_arr_opt,
                )?,
                None => fast::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    self.attention_scale,
                    mask_arr_opt.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array),
                    None::<&Array>,
                )?,
            };

            let attn_out = attn_out
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, T, -1])?;
            let attn_out = layer.o_proj.forward(&attn_out)?;
            let h_post_attn = h.add(&attn_out)?;

            let normed_post = fast::rms_norm(&h_post_attn, &layer.post_attn_norm, rms_eps)?;
            let gate = layer.gate_proj.forward(&normed_post)?;
            let up = layer.up_proj.forward(&normed_post)?;
            let mlp_hidden = mlx_rs::nn::silu(&gate)?.multiply(&up)?;
            let mlp_out = layer.down_proj.forward(&mlp_hidden)?;

            h = h_post_attn.add(&mlp_out)?;
        }

        fast::rms_norm(&h, &self.final_norm, rms_eps)
    }

    /// Apply LM head (or tied embed) to `[B, T, hidden]` → `[B, T, vocab]`.
    fn project_logits(&self, h: &Array) -> Result<Array, Exception> {
        match &self.lm_head {
            Some(head) => head.forward(h),
            None => self.embed.forward(h),
        }
    }

    /// Causal forward. Returns logits `[B, 1, vocab]` for the last position
    /// (mlx_lm convention).
    pub fn forward(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_trunk(inputs, cache)?;
        let t = *h
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("trunk hidden missing T dim"))?;
        let last = if t > 1 { h.index((.., -1.., ..)) } else { h };
        self.project_logits(&last)
    }

    /// Causal forward returning logits at **every** position `[B, T, vocab]`.
    /// Used by speculative-decode target verify: given the draft prefix,
    /// obtain one logits row per proposed token in a single forward pass.
    pub fn forward_all_logits(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_trunk(inputs, cache)?;
        self.project_logits(&h)
    }

    /// Profiled variant of `forward`: same result, but attributes per-section
    /// wall time into `times`. Forces `.eval()` after every section (kills
    /// lazy batching — that's the point: ratios matter, absolutes don't).
    ///
    /// Used by `bench_bonsai_q1_decode_breakdown` to answer the
    /// dispatch-bound-vs-matmul-bound question for Bonsai-8B AR parity.
    pub fn forward_profiled(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<SteppingKeyValueCache>>,
        times: &mut SectionTimes,
    ) -> Result<Array, Exception> {
        let h = self.forward_trunk_profiled(inputs, cache, times)?;
        let t0 = std::time::Instant::now();
        let t = *h
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("trunk hidden missing T dim"))?;
        let last = if t > 1 { h.index((.., -1.., ..)) } else { h };
        let logits = self.project_logits(&last)?;
        logits.eval()?;
        times.add("lm_head", t0.elapsed().as_nanos());
        Ok(logits)
    }

    /// Profiled mirror of `forward_trunk`. Inserts `eval + record` at each
    /// semantic section boundary. Sections are grouped by operation type
    /// (qkv projections together, mlp up+gate together, etc.) — per-layer
    /// noise is collapsed into section totals across all layers.
    #[allow(non_snake_case)]
    fn forward_trunk_profiled(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<SteppingKeyValueCache>>,
        times: &mut SectionTimes,
    ) -> Result<Array, Exception> {
        use std::time::Instant;

        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("inputs must have >= 2 dims"))?;
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("inputs must have >= 2 dims"))?;

        if cache.is_empty() {
            *cache = (0..self.layers.len())
                .map(|_| Some(SteppingKeyValueCache::new()))
                .collect();
        } else if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "cache len {} != num_layers {}",
                cache.len(),
                self.layers.len()
            )));
        }

        // Sync point: make sure prior work isn't folded into embed_rows time.
        inputs.eval()?;

        let t0 = Instant::now();
        let mut h = self.embed_rows(inputs)?;
        h.eval()?;
        times.add("embed_rows", t0.elapsed().as_nanos());

        let mask = create_attention_mask(&h, cache, None)?;

        let heads = i32::try_from(self.config.heads)
            .map_err(|_| Exception::custom("heads overflows i32"))?;
        let kv_heads = i32::try_from(self.config.kv_heads)
            .map_err(|_| Exception::custom("kv_heads overflows i32"))?;
        let rms_eps = self.config.rms_norm_eps;

        for (layer, layer_cache) in self.layers.iter().zip(cache.iter_mut()) {
            let t0 = Instant::now();
            let normed = fast::rms_norm(&h, &layer.input_norm, rms_eps)?;
            normed.eval()?;
            times.add("input_norm", t0.elapsed().as_nanos());

            // qkv projections — 3× quantized_matmul on the same input.
            let t0 = Instant::now();
            let q = layer.q_proj.forward(&normed)?;
            let k = layer.k_proj.forward(&normed)?;
            let v = layer.v_proj.forward(&normed)?;
            q.eval()?;
            k.eval()?;
            v.eval()?;
            times.add("qkv_proj", t0.elapsed().as_nanos());

            // Reshape to [B, L, n_heads, head_dim] then transpose to
            // [B, n_heads, L, head_dim]. Metadata-only; lumped with qk_norm.
            let q = q
                .reshape(&[B, T, heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = k
                .reshape(&[B, T, kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = v
                .reshape(&[B, T, kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;

            let t0 = Instant::now();
            let q = fast::rms_norm(&q, &layer.q_norm, rms_eps)?;
            let k = fast::rms_norm(&k, &layer.k_norm, rms_eps)?;
            q.eval()?;
            k.eval()?;
            times.add("qk_norm", t0.elapsed().as_nanos());

            let offset = layer_cache.as_ref().map_or(0, KeyValueCache::offset);
            let t0 = Instant::now();
            let q = self.apply_rope(&q, offset)?;
            let k = self.apply_rope(&k, offset)?;
            q.eval()?;
            k.eval()?;
            times.add("rope", t0.elapsed().as_nanos());

            let mask_arr = match &mask {
                Some(crate::utils::AttentionMask::Array(a)) => Some(a),
                _ => None,
            };
            let mask_arr_opt: Option<&Array> = mask_arr;

            let t0 = Instant::now();
            let attn_out = match layer_cache.as_mut() {
                Some(c) => cached_scaled_dot_product_attention(
                    q,
                    c,
                    k,
                    v,
                    self.attention_scale,
                    mask_arr_opt,
                )?,
                None => fast::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    self.attention_scale,
                    mask_arr_opt.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array),
                    None::<&Array>,
                )?,
            };
            attn_out.eval()?;
            times.add("sdpa_kv", t0.elapsed().as_nanos());

            let attn_out = attn_out
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, T, -1])?;

            let t0 = Instant::now();
            let attn_out = layer.o_proj.forward(&attn_out)?;
            attn_out.eval()?;
            times.add("o_proj", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            let h_post_attn = h.add(&attn_out)?;
            h_post_attn.eval()?;
            times.add("residual", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            let normed_post = fast::rms_norm(&h_post_attn, &layer.post_attn_norm, rms_eps)?;
            normed_post.eval()?;
            times.add("post_attn_norm", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            let gate = layer.gate_proj.forward(&normed_post)?;
            let up = layer.up_proj.forward(&normed_post)?;
            gate.eval()?;
            up.eval()?;
            times.add("mlp_up_gate", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            let mlp_hidden = mlx_rs::nn::silu(&gate)?.multiply(&up)?;
            mlp_hidden.eval()?;
            times.add("silu_mul", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            let mlp_out = layer.down_proj.forward(&mlp_hidden)?;
            mlp_out.eval()?;
            times.add("mlp_down", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            h = h_post_attn.add(&mlp_out)?;
            h.eval()?;
            times.add("residual", t0.elapsed().as_nanos());
        }

        let t0 = Instant::now();
        let out = fast::rms_norm(&h, &self.final_norm, rms_eps)?;
        out.eval()?;
        times.add("final_norm", t0.elapsed().as_nanos());
        Ok(out)
    }
}

/// Per-section wall-time accumulator for the Bonsai-Q1 forward pass.
///
/// Exists only to attribute the 45 ms/tok Bonsai-8B AR decode cost to
/// individual sections (embed / norms / qkv / rope / sdpa / o_proj / mlp / lm_head).
/// Each section's compute is force-`.eval()`'d to prevent MLX lazy batching
/// from pooling multiple sections into one materialization — ratios between
/// sections are meaningful even though absolutes will be slower than the
/// unprofiled path.
#[derive(Debug, Default, Clone)]
pub struct SectionTimes {
    totals: std::collections::BTreeMap<&'static str, (u128, u64)>,
}

impl SectionTimes {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, name: &'static str, ns: u128) {
        let e = self.totals.entry(name).or_insert((0, 0));
        e.0 += ns;
        e.1 += 1;
    }

    /// Total across all sections (ns).
    pub fn total_ns(&self) -> u128 {
        self.totals.values().map(|(t, _)| *t).sum()
    }

    /// Section totals: `(name, total_ns, call_count)`, sorted by ns descending.
    pub fn entries(&self) -> Vec<(&'static str, u128, u64)> {
        let mut v: Vec<_> = self.totals.iter().map(|(k, (t, n))| (*k, *t, *n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }
}

fn load_packed(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    who: &str,
) -> Result<PackedQ1Linear, String> {
    if in_features % GROUP_SIZE != 0 {
        return Err(format!(
            "{who}: in_features {in_features} not divisible by group_size {GROUP_SIZE}"
        ));
    }
    let packed_cols = in_features / 32;
    let n_groups = in_features / GROUP_SIZE;

    let w_view = tensors
        .tensor(&format!("{prefix}.weight"))
        .map_err(|e| format!("{who}: {prefix}.weight: {e}"))?;
    let s_view = tensors
        .tensor(&format!("{prefix}.scales"))
        .map_err(|e| format!("{who}: {prefix}.scales: {e}"))?;
    let b_view = tensors
        .tensor(&format!("{prefix}.biases"))
        .map_err(|e| format!("{who}: {prefix}.biases: {e}"))?;

    let w_bytes = w_view.data();
    let s_bytes = s_view.data();
    let b_bytes = b_view.data();

    let expected_w_bytes = out_features * packed_cols * 4;
    if w_bytes.len() != expected_w_bytes {
        return Err(format!(
            "{who}: weight byte-size mismatch: got {} expected {}",
            w_bytes.len(),
            expected_w_bytes,
        ));
    }
    let expected_sb_bytes = out_features * n_groups * 2;
    if s_bytes.len() != expected_sb_bytes {
        return Err(format!(
            "{who}: scales byte-size mismatch: got {} expected {}",
            s_bytes.len(),
            expected_sb_bytes,
        ));
    }
    if b_bytes.len() != expected_sb_bytes {
        return Err(format!(
            "{who}: biases byte-size mismatch: got {} expected {}",
            b_bytes.len(),
            expected_sb_bytes,
        ));
    }

    let w_packed: Vec<u32> = w_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let scales = bytes_to_f16_vec(s_bytes);
    let biases = bytes_to_f16_vec(b_bytes);

    Ok(PackedQ1Linear {
        w_packed,
        scales,
        biases,
        out_features,
        in_features,
    })
}

fn load_f16(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f16>, String> {
    let view = tensors.tensor(name).map_err(|e| format!("{name}: {e}"))?;
    Ok(bytes_to_f16_vec(view.data()))
}

fn bytes_to_f16_vec(b: &[u8]) -> Vec<f16> {
    b.chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn bonsai_1_7b_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/lm-studio/models/prism-ml/Bonsai-1.7B-mlx-1bit");
        dir.join("config.json").exists().then_some(dir)
    }

    fn bonsai_8b_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/lm-studio/models/prism-ml/Bonsai-8B-mlx-1bit");
        dir.join("config.json").exists().then_some(dir)
    }

    /// Load 1.7B packed and check dims + residency.
    #[test]
    fn test_load_bonsai_1_7b_packed() {
        let Some(dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };
        let t0 = std::time::Instant::now();
        let engine = BonsaiQ1Engine::load(&dir).unwrap();
        eprintln!("load 1.7B in {}ms", t0.elapsed().as_millis());

        assert_eq!(engine.config.layers, 28);
        assert_eq!(engine.config.hidden, 2048);
        assert_eq!(engine.config.heads, 16);
        assert_eq!(engine.config.kv_heads, 8);
        assert_eq!(engine.config.head_dim, 128);
        assert_eq!(engine.config.inter, 6144);
        assert_eq!(engine.config.vocab, 151669);
        assert!(engine.config.tie_word_embeddings, "1.7B is tied");
        assert!(engine.lm_head.is_none());
        assert_eq!(engine.num_layers(), 28);

        let mb = engine.resident_bytes() as f64 / (1024.0 * 1024.0);
        eprintln!("1.7B packed resident: {mb:.1}MB");
        // Packed target: ~250 MB weights + 2048 × 151669 × 0.15625 = 47 MB embed.
        assert!(
            mb < 400.0,
            "1.7B residency {mb:.1}MB exceeds 400MB cap (packed math)"
        );
    }

    /// Load 8B packed and check dims + residency (~1.25 GB target).
    #[test]
    fn test_load_bonsai_8b_packed() {
        let Some(dir) = bonsai_8b_dir() else {
            eprintln!("Bonsai-8B not found, skipping");
            return;
        };
        let t0 = std::time::Instant::now();
        let engine = BonsaiQ1Engine::load(&dir).unwrap();
        eprintln!("load 8B in {}ms", t0.elapsed().as_millis());

        assert_eq!(engine.config.layers, 36);
        assert_eq!(engine.config.hidden, 4096);
        assert_eq!(engine.config.heads, 32);
        assert_eq!(engine.config.kv_heads, 8);
        assert_eq!(engine.config.head_dim, 128);
        assert_eq!(engine.config.inter, 12288);
        assert_eq!(engine.config.vocab, 151669);
        assert!(!engine.config.tie_word_embeddings, "8B has untied lm_head");
        assert!(engine.lm_head.is_some());
        assert_eq!(engine.config.rope_yarn_factor, Some(4.0));
        assert_eq!(engine.config.rope_original_max_seq, Some(16384));

        let mb = engine.resident_bytes() as f64 / (1024.0 * 1024.0);
        eprintln!("8B packed resident: {mb:.1}MB");
        // Packed target: ~1250 MB. Enforce < 2 GB.
        assert!(
            mb < 2048.0,
            "8B residency {mb:.1}MB exceeds 2 GB cap — packing regression?"
        );
    }

    /// Verify dequant_row_to_fp32 against the existing dequant_q1_g128 oracle
    /// on a single row of a real layer.
    #[test]
    fn test_packed_row_matches_reference_dequant() {
        let Some(dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };
        let engine = BonsaiQ1Engine::load(&dir).unwrap();
        let layer0 = &engine.layers[0];
        let q = &layer0.q_proj;

        // Reference: use existing dequant_q1_g128 from diffusion module.
        // Reconstruct raw bytes for row 0 only.
        let row = 0usize;
        let in_f = q.in_features;
        let packed_cols = in_f / 32;
        let n_groups = in_f / GROUP_SIZE;

        let w_row_u32 = &q.w_packed[row * packed_cols..(row + 1) * packed_cols];
        let s_row_f16 = &q.scales[row * n_groups..(row + 1) * n_groups];
        let b_row_f16 = &q.biases[row * n_groups..(row + 1) * n_groups];

        // Rebuild "row bytes" and call the diffusion-side reference on 1 row.
        let w_bytes: Vec<u8> = w_row_u32.iter().flat_map(|w| w.to_le_bytes()).collect();
        let s_bytes: Vec<u8> = s_row_f16
            .iter()
            .flat_map(|f| f.to_bits().to_le_bytes())
            .collect();
        let b_bytes: Vec<u8> = b_row_f16
            .iter()
            .flat_map(|f| f.to_bits().to_le_bytes())
            .collect();

        let reference = crate::diffusion::dequant_q1_g128(&w_bytes, &s_bytes, &b_bytes, 1, in_f);

        let mut ours = vec![0.0f32; in_f];
        q.dequant_row_to_fp32(row, &mut ours);

        assert_eq!(reference.len(), ours.len());
        let max_err = reference
            .iter()
            .zip(ours.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("max_err vs reference dequant: {max_err}");
        assert!(max_err < 1e-6, "dequant mismatch: max_err={max_err}");
    }

    /// P2 acceptance gate — verify MLX `quantized_matmul` with `bits=1` through
    /// the PrismML-forked core + mlx-c v0.6.0-3 bindings matches a scalar
    /// dequant-then-dot oracle on a real Bonsai-1.7B layer. A passing test
    /// proves the 1-bit hot path is wired end-to-end: packed uint32 storage →
    /// mlx-rs `Array` → PrismML `quantize(bits=1)` kernel → fp32 result.
    #[test]
    fn test_mlx_quantized_matmul_bits1_matches_oracle() {
        let Some(dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };
        let engine = BonsaiQ1Engine::load(&dir).unwrap();

        // Pick k_proj of layer 0 — smallest per-layer linear in 1.7B:
        // in=hidden=2048, out=kv_heads*head_dim=8*128=1024. Keeps the oracle
        // loop cheap while exercising the full dispatch.
        let k = &engine.layers[0].k_proj;
        let in_f = k.in_features;
        let out_f = k.out_features;
        let packed_cols = in_f / 32;
        let n_groups = in_f / GROUP_SIZE;

        // Build MLX arrays directly from the packed tables (no dequant).
        let w_mlx = mlx_rs::Array::from_slice(&k.w_packed, &[out_f as i32, packed_cols as i32]);
        let s_f32: Vec<f32> = k.scales.iter().map(|h| h.to_f32()).collect();
        let b_f32: Vec<f32> = k.biases.iter().map(|h| h.to_f32()).collect();
        let s_mlx = mlx_rs::Array::from_slice(&s_f32, &[out_f as i32, n_groups as i32]);
        let b_mlx = mlx_rs::Array::from_slice(&b_f32, &[out_f as i32, n_groups as i32]);

        // Deterministic activation: two sinusoids, small magnitude so fp16
        // intermediate precision doesn't blow the tolerance.
        let x_f32: Vec<f32> = (0..in_f)
            .map(|i| 0.01 * ((i as f32 * 0.03).sin() + 0.5 * (i as f32 * 0.17).cos()))
            .collect();
        let x_mlx = mlx_rs::Array::from_slice(&x_f32, &[1, in_f as i32]);

        // Hot path: PrismML bits=1 quantized matmul.
        let y_mlx = mlx_rs::ops::quantized_matmul(
            &x_mlx,
            &w_mlx,
            &s_mlx,
            &b_mlx,
            true,
            GROUP_SIZE as i32,
            1,
        )
        .expect("quantized_matmul(bits=1) failed — PrismML mlx core swap missing?");
        y_mlx.eval().expect("eval failed");
        let y_mlx_vec: &[f32] = y_mlx.as_slice::<f32>();
        assert_eq!(y_mlx_vec.len(), out_f);

        // Oracle: per-row scalar dequant-then-dot.
        let mut y_ref = vec![0.0f32; out_f];
        let mut w_row = vec![0.0f32; in_f];
        for row in 0..out_f {
            k.dequant_row_to_fp32(row, &mut w_row);
            y_ref[row] = w_row.iter().zip(x_f32.iter()).map(|(a, b)| a * b).sum();
        }

        let max_err = y_ref
            .iter()
            .zip(y_mlx_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "bits=1 quantized_matmul[1,{in_f}]x[{out_f},{in_f}] vs scalar oracle: max_err={max_err}"
        );
        assert!(
            max_err < 1e-2,
            "bits=1 MLX matmul disagrees with oracle dequant: max_err={max_err}"
        );
    }

    /// P4 acceptance gate — run the full forward pass on Bonsai-1.7B, verify
    /// prefill produces finite vocab-sized logits, decode advances the cache
    /// by one, and repeated prefills are bitwise-deterministic.
    #[test]
    fn test_bonsai_q1_forward_prefill_decode() {
        let Some(dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };
        let engine = BonsaiQ1Engine::load(&dir).unwrap();
        let vocab = engine.config.vocab;
        let layers_n = engine.config.layers;
        let t0 = std::time::Instant::now();
        let gpu = engine.to_gpu().expect("to_gpu failed");
        eprintln!("to_gpu in {}ms", t0.elapsed().as_millis());
        assert_eq!(gpu.num_layers(), layers_n);

        // Deterministic 5-token prompt (skip tokenizer — we only need finiteness + shape).
        let ids: Vec<i32> = vec![1, 2, 3, 4, 5];
        let prompt = mlx_rs::Array::from_slice(&ids, &[1, 5]);

        let mut cache: Vec<Option<SteppingKeyValueCache>> = Vec::new();
        let t0 = std::time::Instant::now();
        let logits0 = gpu.forward(&prompt, &mut cache).expect("prefill forward");
        logits0.eval().expect("eval prefill logits");
        eprintln!("prefill 5 tokens in {}ms", t0.elapsed().as_millis());

        let shape0 = logits0.shape().to_vec();
        assert_eq!(
            shape0,
            &[1, 1, vocab as i32],
            "logits shape mismatch: {shape0:?}"
        );
        let s0: &[f32] = logits0.as_slice::<f32>();
        assert_eq!(s0.len(), vocab);
        assert!(
            s0.iter().all(|v| v.is_finite()),
            "prefill logits contain NaN/Inf"
        );

        // Cache offset after prefill.
        let offset_after_prefill = cache[0].as_ref().unwrap().offset();
        assert_eq!(
            offset_after_prefill, 5,
            "cache offset should equal prefill len"
        );

        // Argmax of prefill → decode input.
        let (mut best_idx, mut best_val) = (0i32, f32::NEG_INFINITY);
        for (i, v) in s0.iter().enumerate() {
            if *v > best_val {
                best_val = *v;
                best_idx = i as i32;
            }
        }
        let decode_in = mlx_rs::Array::from_slice(&[best_idx], &[1, 1]);
        let t0 = std::time::Instant::now();
        let logits1 = gpu.forward(&decode_in, &mut cache).expect("decode forward");
        logits1.eval().expect("eval decode logits");
        eprintln!("decode 1 token in {}ms", t0.elapsed().as_millis());

        assert_eq!(logits1.shape().to_vec(), &[1, 1, vocab as i32]);
        let s1: &[f32] = logits1.as_slice::<f32>();
        assert!(
            s1.iter().all(|v| v.is_finite()),
            "decode logits contain NaN/Inf"
        );
        assert_eq!(cache[0].as_ref().unwrap().offset(), 6);

        // Determinism: fresh cache, same prompt, bitwise-identical logits.
        let mut cache2: Vec<Option<SteppingKeyValueCache>> = Vec::new();
        let logits0b = gpu.forward(&prompt, &mut cache2).expect("repeat prefill");
        logits0b.eval().unwrap();
        let s0b: &[f32] = logits0b.as_slice::<f32>();
        assert_eq!(s0, s0b, "repeat prefill is non-deterministic");
        eprintln!("P4 forward OK: prefill+decode finite, determinism confirmed");
    }

    /// P5 acceptance gate — round-trip Bonsai-1.7B through `AnyModel`:
    /// `load_bonsai_q1` + `AnyModel::BonsaiQ1` variant + `make_cache()` +
    /// `AnyModel::forward()`. Asserts logits shape matches the direct
    /// `BonsaiQ1Gpu::forward` path, proving the dispatch table is wired.
    #[test]
    fn test_bonsai_q1_through_anymodel() {
        use crate::AnyModel;

        let Some(dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };
        let gpu = load_bonsai_q1(&dir).expect("load_bonsai_q1 failed");
        let vocab = gpu.config.vocab as i32;
        let layers_n = gpu.num_layers();

        let mut model = AnyModel::BonsaiQ1(gpu);
        assert_eq!(model.num_layers(), layers_n);

        let mut cache = model.make_cache();

        let ids: Vec<i32> = vec![1, 2, 3, 4, 5];
        let prompt = mlx_rs::Array::from_slice(&ids, &[1, 5]);
        let logits = model
            .forward(&prompt, None, &mut cache)
            .expect("AnyModel::forward on BonsaiQ1 failed");
        logits.eval().expect("eval");
        assert_eq!(
            logits.shape().to_vec(),
            &[1, 1, vocab],
            "BonsaiQ1 through AnyModel: logits shape mismatch"
        );
        let s: &[f32] = logits.as_slice::<f32>();
        assert!(
            s.iter().all(|v| v.is_finite()),
            "AnyModel BonsaiQ1 logits contain NaN/Inf"
        );
        eprintln!("P5 AnyModel::BonsaiQ1 forward OK: shape=[1,1,{vocab}]");
    }

    /// Path-A blocker removal: verify `AnyModel::forward_all_logits` returns
    /// `[B, T, vocab]` on BonsaiQ1 so the spec-decode target verify path
    /// (`simple.rs::speculative_generate` L2721) can use Bonsai-8B as target.
    /// Also asserts the last-position row matches `forward`'s logits (shared
    /// trunk correctness).
    #[test]
    fn test_bonsai_q1_forward_all_logits() {
        use crate::AnyModel;

        let Some(dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };
        let gpu = load_bonsai_q1(&dir).expect("load");
        let vocab = gpu.config.vocab as i32;
        let mut model = AnyModel::BonsaiQ1(gpu);

        let ids: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let x = mlx_rs::Array::from_slice(&ids, &[1, 8]);

        // All-logits path (spec-decode verify uses this).
        let mut c = model.make_cache();
        let all = model
            .forward_all_logits(&x, None, &mut c)
            .expect("forward_all_logits");
        all.eval().unwrap();
        assert_eq!(
            all.shape().to_vec(),
            &[1, 8, vocab],
            "all-logits shape mismatch"
        );
        let all_flat: &[f32] = all.as_slice::<f32>();
        let last_row = &all_flat[7 * vocab as usize..8 * vocab as usize];

        // Last-position path (must match all-logits row 7).
        let mut c2 = model.make_cache();
        let last = model.forward(&x, None, &mut c2).expect("forward");
        last.eval().unwrap();
        assert_eq!(last.shape().to_vec(), &[1, 1, vocab]);
        let last_slice: &[f32] = last.as_slice::<f32>();
        assert_eq!(last_slice.len(), vocab as usize);

        // Shared-trunk check: last-position logits must be bitwise-identical.
        assert_eq!(
            last_slice, last_row,
            "forward vs forward_all_logits[-1] diverge"
        );
        eprintln!("forward_all_logits OK: shape=[1,8,{vocab}], row[-1] matches forward()");
    }

    // ---------------------------------------------------------------------
    // Bench (ignored by default — run with `--ignored --nocapture`).
    // ---------------------------------------------------------------------

    fn synth_ids(n: i32) -> Vec<i32> {
        // Deterministic low-index stream; avoids special tokens near 0 and
        // keeps indices well below vocab for any Qwen3-shape target.
        (0..n).map(|i| 100 + (i * 131) % 50_000).collect()
    }

    fn bench_one(
        name: &str,
        dir: &std::path::Path,
        prefill_lens: &[i32],
        decode_n: i32,
    ) -> Option<String> {
        use crate::AnyModel;
        use std::time::Instant;

        let mut md = String::new();
        md.push_str(&format!("\n## {name}\n\n"));

        // --- Load + to_gpu ---
        let t0 = Instant::now();
        let engine = BonsaiQ1Engine::load(dir).ok()?;
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mb = engine.resident_bytes() as f64 / (1024.0 * 1024.0);
        let layers_n = engine.num_layers();
        let vocab = engine.config.vocab as i32;

        let t0 = Instant::now();
        let gpu = engine.to_gpu().expect("to_gpu");
        let to_gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        md.push_str(&format!(
            "- layers: {layers_n}, vocab: {vocab}, packed resident: {mb:.1} MB\n"
        ));
        md.push_str(&format!(
            "- load: {load_ms:.0} ms, to_gpu: {to_gpu_ms:.0} ms\n\n"
        ));
        eprintln!(
            "[{name}] load={load_ms:.0}ms to_gpu={to_gpu_ms:.0}ms resident={mb:.1}MB layers={layers_n}"
        );

        let mut model = AnyModel::BonsaiQ1(gpu);

        // --- Kernel warmup (1 forward discarded) ---
        {
            let mut c = model.make_cache();
            let warm_ids = synth_ids(8);
            let x = mlx_rs::Array::from_slice(&warm_ids, &[1, 8]);
            let y = model.forward(&x, None, &mut c).expect("warmup forward");
            y.eval().expect("warmup eval");
        }

        // --- Prefill matrix ---
        // Two passes per shape: the first warms the MLX per-shape kernel
        // cache (one-shot compile), the second is the measured run.
        md.push_str("### Prefill (per-shape kernel warmed, fresh KV cache)\n\n");
        md.push_str("| L (tokens) | ms | tok/s |\n|---:|---:|---:|\n");
        for &L in prefill_lens {
            let ids = synth_ids(L);
            // Warm (compile kernel for this T).
            {
                let x = mlx_rs::Array::from_slice(&ids, &[1, L]);
                let mut c = model.make_cache();
                let y = model.forward(&x, None, &mut c).expect("prefill warmup");
                y.eval().expect("eval warmup");
            }
            // Measure.
            let x = mlx_rs::Array::from_slice(&ids, &[1, L]);
            let mut c = model.make_cache();
            let t0 = Instant::now();
            let y = model.forward(&x, None, &mut c).expect("prefill");
            y.eval().expect("eval");
            let dt = t0.elapsed().as_secs_f64();
            let tps = f64::from(L) / dt;
            md.push_str(&format!("| {L} | {:.1} | {tps:.1} |\n", dt * 1000.0));
            eprintln!(
                "[{name}] prefill L={L}: {:.1}ms ({tps:.1} tok/s)",
                dt * 1000.0
            );
        }

        // --- Sustained decode ---
        md.push_str("\n### Sustained decode after 16-token prefill (autoregressive argmax)\n\n");
        let mut c = model.make_cache();
        // Pre-allocate KV cache for the full sequence. Without this, the
        // default 256-token step forces a grow+concat every 256 decode
        // steps, which breaks any compiled-decode trace (shape of
        // keys/values changes). See crates/higgs-models/src/cache.rs.
        c.reserve_max_tokens(16 + decode_n + 8); // slack for safety
        let prompt = synth_ids(16);
        let x = mlx_rs::Array::from_slice(&prompt, &[1, 16]);
        let y0 = model.forward(&x, None, &mut c).expect("decode prefill");
        y0.eval().expect("eval");

        // Argmax from prefill logits to seed decode.
        let s0: &[f32] = y0.as_slice::<f32>();
        let (mut tok, mut best) = (0i32, f32::NEG_INFINITY);
        for (i, v) in s0.iter().enumerate() {
            if *v > best {
                best = *v;
                tok = i as i32;
            }
        }

        // Warm 4 steps (settle), then time decode_n-4 steps.
        let warm_steps = 4.min(decode_n / 4);
        for _ in 0..warm_steps {
            let d = mlx_rs::Array::from_slice(&[tok], &[1, 1]);
            let y = model.forward(&d, None, &mut c).expect("decode");
            y.eval().expect("eval");
            let s: &[f32] = y.as_slice::<f32>();
            let (mut nb_i, mut nb_v) = (0i32, f32::NEG_INFINITY);
            for (i, v) in s.iter().enumerate() {
                if *v > nb_v {
                    nb_v = *v;
                    nb_i = i as i32;
                }
            }
            tok = nb_i;
        }

        let steady = decode_n - warm_steps;
        let t0 = Instant::now();
        for _ in 0..steady {
            let d = mlx_rs::Array::from_slice(&[tok], &[1, 1]);
            let y = model.forward(&d, None, &mut c).expect("decode");
            y.eval().expect("eval");
            let s: &[f32] = y.as_slice::<f32>();
            let (mut nb_i, mut nb_v) = (0i32, f32::NEG_INFINITY);
            for (i, v) in s.iter().enumerate() {
                if *v > nb_v {
                    nb_v = *v;
                    nb_i = i as i32;
                }
            }
            tok = nb_i;
        }
        let dt = t0.elapsed().as_secs_f64();
        let tps = f64::from(steady) / dt;
        let ms_per_tok = dt * 1000.0 / f64::from(steady);

        md.push_str(&format!(
            "| decode steps | total ms | ms/tok | tok/s |\n|---:|---:|---:|---:|\n"
        ));
        md.push_str(&format!(
            "| {steady} (after {warm_steps} warmup) | {:.1} | {ms_per_tok:.2} | {tps:.1} |\n",
            dt * 1000.0
        ));
        eprintln!(
            "[{name}] decode {steady} steps: {:.1}ms ({tps:.1} tok/s, {ms_per_tok:.2}ms/tok)",
            dt * 1000.0
        );

        Some(md)
    }

    #[test]
    #[ignore = "bench; run with --ignored --nocapture"]
    fn bench_bonsai_q1_anymodel_full_matrix() {
        use std::time::SystemTime;

        let mut report = String::new();
        report.push_str("# Bonsai-Q1 through AnyModel — full-matrix bench\n\n");
        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        report.push_str(&format!("Run unix: {ts}\n"));
        report.push_str("Path: AnyModel::BonsaiQ1 → BonsaiQ1Gpu::forward (P5 integration)\n");
        report.push_str("Workload: synthetic deterministic token IDs, argmax decode.\n");

        // 1.7B: full matrix. 8B: drop 2048 prefill (peak memory cap).
        let sections = [
            (
                "Bonsai-1.7B",
                bonsai_1_7b_dir(),
                &[1, 16, 128, 512, 2048][..],
                256,
            ),
            ("Bonsai-8B", bonsai_8b_dir(), &[1, 16, 128, 512][..], 128),
        ];

        let mut any_ran = false;
        for (name, maybe_dir, prefill, decode_n) in sections {
            match maybe_dir {
                Some(dir) => {
                    if let Some(section) = bench_one(name, &dir, prefill, decode_n) {
                        report.push_str(&section);
                        any_ran = true;
                    }
                }
                None => {
                    eprintln!("[{name}] not found, skipping");
                    report.push_str(&format!("\n## {name}\n\n- not found locally; skipped.\n"));
                }
            }
        }
        assert!(any_ran, "no Bonsai checkpoints found; nothing to bench");

        // Print the full markdown report to stdout.
        eprintln!(
            "\n========== BEGIN REPORT ==========\n{report}\n========== END REPORT =========="
        );

        // Try to write to .planning/measurements/p5-bonsai-q1-anymodel.md. Path
        // is derived from the crate dir (two levels up from higgs-models).
        let crate_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let out_dir = crate_dir
            .parent()
            .and_then(std::path::Path::parent)
            .map(|p| p.join(".planning/measurements"));
        if let Some(dir) = out_dir {
            if dir.exists() {
                let out = dir.join("p5-bonsai-q1-anymodel.md");
                match std::fs::write(&out, &report) {
                    Ok(()) => eprintln!("wrote report: {}", out.display()),
                    Err(e) => eprintln!("failed to write report {}: {e}", out.display()),
                }
            } else {
                eprintln!("measurements dir not found: {}", dir.display());
            }
        }
    }

    // ---------------------------------------------------------------------
    // Phase-A decode breakdown: attribute Bonsai-8B's 45 ms/tok to sections.
    // Gated by HIGGS_PROFILE_BONSAI=1 so it runs only when asked; bench
    // itself is `--ignored` anyway.
    // ---------------------------------------------------------------------

    #[test]
    #[ignore = "bench; run with HIGGS_PROFILE_BONSAI=1 --ignored --nocapture"]
    fn bench_bonsai_q1_decode_breakdown() {
        use std::time::Instant;

        if std::env::var("HIGGS_PROFILE_BONSAI").ok().as_deref() != Some("1") {
            eprintln!(
                "HIGGS_PROFILE_BONSAI != 1; skipping (set to 1 to enable per-section profile)"
            );
            return;
        }

        let Some(dir) = bonsai_8b_dir() else {
            eprintln!("Bonsai-8B not found; skipping");
            return;
        };
        let gpu = load_bonsai_q1(&dir).expect("load");
        let layers_n = gpu.num_layers();

        // Prefill once (16 tokens), discard logits. Warmup with 4 decode
        // steps on a FRESH profile accumulator, then measure 64 steps.
        let prompt = synth_ids(16);
        let x = mlx_rs::Array::from_slice(&prompt, &[1, 16]);
        let mut cache: Vec<Option<SteppingKeyValueCache>> = Vec::new();
        let y = gpu.forward(&x, &mut cache).expect("prefill");
        y.eval().expect("eval prefill");

        // Seed decode token from prefill argmax.
        let s0: &[f32] = y.as_slice::<f32>();
        let (mut tok, mut best) = (0i32, f32::NEG_INFINITY);
        for (i, v) in s0.iter().enumerate() {
            if *v > best {
                best = *v;
                tok = i as i32;
            }
        }

        // Warmup (4 steps, no profiling): settle MLX per-shape compile cache.
        for _ in 0..4 {
            let d = mlx_rs::Array::from_slice(&[tok], &[1, 1]);
            let y = gpu.forward(&d, &mut cache).expect("warm decode");
            y.eval().expect("eval");
            let s: &[f32] = y.as_slice::<f32>();
            let mut nb_i = 0i32;
            let mut nb_v = f32::NEG_INFINITY;
            for (i, v) in s.iter().enumerate() {
                if *v > nb_v {
                    nb_v = *v;
                    nb_i = i as i32;
                }
            }
            tok = nb_i;
        }

        // Measured run: 64 decode steps with per-section attribution.
        let steady: i32 = 64;
        let mut times = SectionTimes::new();
        let t0 = Instant::now();
        for _ in 0..steady {
            let d = mlx_rs::Array::from_slice(&[tok], &[1, 1]);
            let y = gpu
                .forward_profiled(&d, &mut cache, &mut times)
                .expect("profiled decode");
            // forward_profiled already evals; still touch logits to pick next tok.
            let s: &[f32] = y.as_slice::<f32>();
            let mut nb_i = 0i32;
            let mut nb_v = f32::NEG_INFINITY;
            for (i, v) in s.iter().enumerate() {
                if *v > nb_v {
                    nb_v = *v;
                    nb_i = i as i32;
                }
            }
            tok = nb_i;
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Aggregate.
        let entries = times.entries();
        let total_ns = times.total_ns() as f64;
        let steady_f = f64::from(steady);
        let ms_per_step = wall_ms / steady_f;
        let accounted_ms = total_ns / 1_000_000.0 / steady_f;
        let accounted_pct = if ms_per_step > 0.0 {
            100.0 * accounted_ms / ms_per_step
        } else {
            0.0
        };

        // Build report.
        let mut md = String::new();
        md.push_str("# Bonsai-8B decode breakdown (Phase A)\n\n");
        md.push_str(&format!("- layers: {layers_n}\n"));
        md.push_str(&format!("- prefill: 16 synthetic tokens\n"));
        md.push_str(&format!("- warmup: 4 decode steps (discarded)\n"));
        md.push_str(&format!(
            "- measured: {steady} decode steps, profiled (eval per section)\n"
        ));
        md.push_str(&format!(
            "- wall ms/step: **{ms_per_step:.2}** (accounted {accounted_ms:.2} ms, {accounted_pct:.1}%)\n\n"
        ));
        md.push_str("| Section | μs/step | % accounted | calls/step |\n");
        md.push_str("|---|---:|---:|---:|\n");
        let denom_ns = total_ns.max(1.0);
        for (name, section_ns, calls) in &entries {
            let us_per_step = (*section_ns as f64) / 1000.0 / steady_f;
            let pct = 100.0 * (*section_ns as f64) / denom_ns;
            let calls_per_step = (*calls as f64) / steady_f;
            md.push_str(&format!(
                "| {name} | {us_per_step:.1} | {pct:.1}% | {calls_per_step:.1} |\n"
            ));
        }
        md.push_str(
            "\n**Note:** eval-per-section kills lazy batching, so ms/step > unprofiled 45 ms.\n",
        );
        md.push_str("Ratios (% accounted) are the signal. Unprofiled 8B decode is 22.2 tok/s\n");
        md.push_str("(~45 ms/tok) per REPORT.md.\n");

        eprintln!("\n========== DECODE BREAKDOWN ==========\n{md}\n========== END ==========");

        // Write to .planning/measurements/bonsai-parity/decode-breakdown.md.
        let crate_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if let Some(out_dir) = crate_dir
            .parent()
            .and_then(std::path::Path::parent)
            .map(|p| p.join(".planning/measurements/bonsai-parity"))
        {
            if out_dir.exists() {
                let out = out_dir.join("decode-breakdown.md");
                match std::fs::write(&out, &md) {
                    Ok(()) => eprintln!("wrote: {}", out.display()),
                    Err(e) => eprintln!("write failed {}: {e}", out.display()),
                }
            } else {
                eprintln!(
                    "measurements/bonsai-parity dir not found: {}",
                    out_dir.display()
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // P6 verify-cost probe: does `forward_all_logits` scale linearly in K?
    // Path A (ANE drafter + GPU target) pivots on this. If 8B verify at K=8
    // scales super-linearly, the split-silicon cycle collapses even with
    // good acceptance. Risk #4 from session-22 recap.
    // ---------------------------------------------------------------------

    #[test]
    #[ignore = "bench; run with --ignored --nocapture"]
    fn bench_bonsai_q1_verify_cost_8b() {
        use crate::AnyModel;
        use std::time::Instant;

        let Some(dir) = bonsai_8b_dir() else {
            eprintln!("Bonsai-8B not found; skipping");
            return;
        };
        let gpu = load_bonsai_q1(&dir).expect("load");
        let vocab = gpu.config.vocab as i32;
        let layers_n = gpu.num_layers();
        let mut model = AnyModel::BonsaiQ1(gpu);

        // Prime the cache with a 64-token "mid-generation" prefix so every K
        // measurement runs against an attn K-dim ~64+K, close to real spec-
        // decode conditions (prefill + some accepted tokens already).
        let prefix_len = 64i32;
        let prefix_ids = synth_ids(prefix_len);

        let ks: &[i32] = &[1, 4, 8, 12, 16];
        let iters = 5;

        let mut md = String::new();
        md.push_str("# Bonsai-8B verify-cost probe\n\n");
        md.push_str(&format!("- layers: {layers_n}, vocab: {vocab}\n"));
        md.push_str(&format!(
            "- prime prefix: {prefix_len} tokens (fresh KV per K measurement)\n"
        ));
        md.push_str(&format!(
            "- timing: min of {iters} iters per K after per-shape warmup\n\n"
        ));
        md.push_str("| K | min ms | ms/tok | vs K=1 ratio | super-linear? |\n");
        md.push_str("|---:|---:|---:|---:|:---|\n");

        let mut baseline_ms = f64::NAN;

        for &k in ks {
            let ids = synth_ids(k);

            // Warmup: prefill + one all-logits call to compile kernels for this K.
            {
                let mut c = model.make_cache();
                let px = mlx_rs::Array::from_slice(&prefix_ids, &[1, prefix_len]);
                let y = model.forward(&px, None, &mut c).expect("prefill warmup");
                y.eval().expect("eval");
                let x = mlx_rs::Array::from_slice(&ids, &[1, k]);
                let yv = model
                    .forward_all_logits(&x, None, &mut c)
                    .expect("verify warmup");
                yv.eval().expect("eval");
            }

            // Measure: min over `iters`. Fresh cache each time so timings aren't
            // polluted by state carried over from the previous K.
            let mut best_ms = f64::INFINITY;
            for _ in 0..iters {
                let mut c = model.make_cache();
                let px = mlx_rs::Array::from_slice(&prefix_ids, &[1, prefix_len]);
                let y = model.forward(&px, None, &mut c).expect("prefill");
                y.eval().expect("eval");

                let x = mlx_rs::Array::from_slice(&ids, &[1, k]);
                let t0 = Instant::now();
                let yv = model.forward_all_logits(&x, None, &mut c).expect("verify");
                yv.eval().expect("eval");
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                if ms < best_ms {
                    best_ms = ms;
                }
            }

            if k == 1 {
                baseline_ms = best_ms;
            }
            let ratio = best_ms / baseline_ms;
            let per_tok = best_ms / f64::from(k);
            let expected_ratio = f64::from(k);
            let super_linear = ratio / expected_ratio > 1.25;
            let flag = if super_linear { "YES (>1.25×)" } else { "no" };

            md.push_str(&format!(
                "| {k} | {best_ms:.2} | {per_tok:.2} | {ratio:.2}× (ideal {expected_ratio:.0}×) | {flag} |\n"
            ));
            eprintln!(
                "[verify-cost 8B] K={k}: {best_ms:.2}ms ({per_tok:.2}ms/tok, {ratio:.2}× baseline)"
            );
        }

        md.push_str("\n**Interpretation:** if any K flags super-linear, Path A's verify\n");
        md.push_str("budget must be re-estimated before running the daemon experiment.\n");
        md.push_str("Linear-ish scaling (≤1.25× over ideal) means K=8 verify ≈ 8× K=1 cost,\n");
        md.push_str("which preserves the session-22 back-of-envelope.\n");

        eprintln!("\n========== VERIFY-COST REPORT ==========\n{md}\n========== END ==========");

        let crate_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if let Some(out_dir) = crate_dir
            .parent()
            .and_then(std::path::Path::parent)
            .map(|p| p.join(".planning/measurements"))
        {
            if out_dir.exists() {
                let out = out_dir.join("p6-verify-cost-8b.md");
                match std::fs::write(&out, &md) {
                    Ok(()) => eprintln!("wrote: {}", out.display()),
                    Err(e) => eprintln!("write failed {}: {e}", out.display()),
                }
            }
        }
    }
}
