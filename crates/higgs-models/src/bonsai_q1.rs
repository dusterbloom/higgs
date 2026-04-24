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

use safetensors::SafeTensors;

pub const GROUP_SIZE: usize = 128;

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
        let layer_bytes: usize = self.layers.iter().map(BonsaiQ1LayerWeights::resident_bytes).sum();
        let lm_head_bytes = self.lm_head.as_ref().map_or(0, PackedQ1Linear::resident_bytes);
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

        let quant = cfg.get("quantization").ok_or("missing quantization block")?;
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

        let embed =
            load_packed(&tensors, "model.embed_tokens", vocab, hidden, "embed_tokens")?;
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
    let view = tensors
        .tensor(name)
        .map_err(|e| format!("{name}: {e}"))?;
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
        let s_bytes: Vec<u8> = s_row_f16.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_row_f16.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect();

        let reference = crate::diffusion::dequant_q1_g128(&w_bytes, &s_bytes, &b_bytes, 1, in_f);

        let mut ours = vec![0.0f32; in_f];
        q.dequant_row_to_fp32(row, &mut ours);

        assert_eq!(reference.len(), ours.len());
        let max_err =
            reference.iter().zip(ours.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
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
        let w_mlx = mlx_rs::Array::from_slice(
            &k.w_packed,
            &[out_f as i32, packed_cols as i32],
        );
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
}
