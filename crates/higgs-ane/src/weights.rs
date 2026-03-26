//! Model weights loaded from MLX safetensors format.

use crate::config::ModelConfig;
use crate::safetensors::{MmapTensorStore, bf16_to_f32, dequant_nbit};
use std::io;
use std::path::Path;

/// Per-layer transformer weights (f32, dequantized at load time).
pub struct LayerWeights {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
    pub rms_att: Vec<f32>,
    pub rms_ffn: Vec<f32>,
}

/// Full model weights for CPU/ANE decode.
pub struct ModelWeights {
    pub cfg: ModelConfig,
    pub layers: Vec<LayerWeights>,
    pub rms_final: Vec<f32>,
    pub embed: Vec<f32>,
    pub vocab_size: usize,
}

impl ModelWeights {
    /// Load weights from an MLX safetensors directory.
    ///
    /// Handles quantized weights (N-bit packed + BF16 scales/biases) and
    /// plain BF16 weights. GQA KV heads are NOT expanded (kept at kv_dim).
    pub fn from_mlx_safetensors(dir: &Path, cfg: &ModelConfig) -> io::Result<Self> {
        let store = MmapTensorStore::open(dir)?;

        let get_weight = |base: &str| -> io::Result<Vec<f32>> {
            let base = store.resolve_weight_base(base);
            let w_key = format!("{base}.weight");
            let s_key = format!("{base}.scales");
            let b_key = format!("{base}.biases");

            if let (Some(w), Some(s), Some(b)) =
                (store.get(&w_key), store.get(&s_key), store.get(&b_key))
            {
                let (_, shape) = store.meta(&w_key).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("no meta for {w_key}"))
                })?;
                let rows = shape[0];
                let cols = shape[1] * 32 / cfg.bits;
                let sc = bf16_to_f32(s);
                let bi = bf16_to_f32(b);
                Ok(dequant_nbit(w, &sc, &bi, rows, cols, cfg.group_size, cfg.bits))
            } else if let Some(data) = store.get(&format!("{base}.weight")) {
                let (dtype, _) = store.meta(&format!("{base}.weight")).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("no meta for {base}.weight"))
                })?;
                if dtype == "BF16" {
                    Ok(bf16_to_f32(data))
                } else if dtype == "F32" {
                    Ok(data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect())
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unsupported dtype {dtype} for {base}"),
                    ))
                }
            } else {
                Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("missing tensor: {base}"),
                ))
            }
        };

        let get_bf16 = |name: &str| -> io::Result<Vec<f32>> {
            let name = store.resolve_tensor_name(name);
            let data = store.get(&name).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("missing: {name}"))
            })?;
            let dtype = store.meta(&name).map(|(d, _)| d.as_str()).unwrap_or("BF16");
            Ok(match dtype {
                "F32" => data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                _ => bf16_to_f32(data),
            })
        };

        let dim = cfg.dim;

        // Embedding
        let embed = get_weight("model.embed_tokens")?;
        let expected = cfg.vocab_size * dim;
        if embed.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("embed size mismatch: got {}, expected {expected}", embed.len()),
            ));
        }

        // Layers
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let prefix = format!("model.layers.{l}");

            let wq = get_weight(&format!("{prefix}.self_attn.q_proj"))?;
            let wk = get_weight(&format!("{prefix}.self_attn.k_proj"))?;
            let wv = get_weight(&format!("{prefix}.self_attn.v_proj"))?;
            let wo = get_weight(&format!("{prefix}.self_attn.o_proj"))?;

            // FFN: try gate_proj/up_proj/down_proj, then gate_up_proj (fused)
            let (w1, w3, w2) = if let Ok(g) =
                get_weight(&format!("{prefix}.mlp.gate_proj"))
            {
                let u = get_weight(&format!("{prefix}.mlp.up_proj"))?;
                let d = get_weight(&format!("{prefix}.mlp.down_proj"))?;
                (g, u, d)
            } else {
                // Fused gate_up_proj: [2*hidden, dim] → split by rows
                let fused = get_weight(&format!("{prefix}.mlp.gate_up_proj"))?;
                let mid = fused.len() / 2;
                let d = get_weight(&format!("{prefix}.mlp.down_proj"))?;
                (fused[..mid].to_vec(), fused[mid..].to_vec(), d)
            };

            let rms_att = get_bf16(&format!("{prefix}.input_layernorm.weight"))?;
            let rms_ffn = get_bf16(&format!("{prefix}.post_attention_layernorm.weight"))?;

            layers.push(LayerWeights { wq, wk, wv, wo, w1, w2, w3, rms_att, rms_ffn });

            if l == 0 {
                tracing::debug!(
                    wq = layers[0].wq.len(),
                    wk = layers[0].wk.len(),
                    w1 = layers[0].w1.len(),
                    "draft model: loaded layer 0"
                );
            }
        }

        let rms_final = get_bf16("model.norm.weight")?;

        Ok(Self {
            cfg: cfg.clone(),
            layers,
            rms_final,
            embed,
            vocab_size: cfg.vocab_size,
        })
    }
}
