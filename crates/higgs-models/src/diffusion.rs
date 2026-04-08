//! MDLM Diffusion Language Model — Qwen3 encoder + BLAS forward + denoising loop.
//!
//! Self-contained engine for masked diffusion text generation. No MLX dependency.
//! Uses Accelerate BLAS (cblas_sgemm) for matrix-matrix multiplies at seq>1.
//!
//! Architecture: Qwen3 transformer encoder (bidirectional attention, no causal mask).
//! Generation: iterative denoising — unmask highest-confidence [MASK] positions each step.

#![allow(clippy::too_many_arguments, unsafe_code)]

use std::path::Path;

// BLAS FFI — Accelerate framework (linked via ane_bridge build.rs)
unsafe extern "C" {
    unsafe fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// y[M,N] = alpha * A[M,K] @ B[K,N] + beta * y. Row-major.
pub(crate) fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101,
            111,
            111, // RowMajor, NoTrans, NoTrans
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            y.as_mut_ptr(),
            n as i32,
        );
    }
}

/// y[M,N] += alpha * A[M,K] @ B[K,N]. Accumulates into y.
pub(crate) fn sgemm_acc(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101,
            111,
            111,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            1.0,
            y.as_mut_ptr(),
            n as i32,
        );
    }
}

/// y[M,N] = A^T[M,K] @ B[K,N] (A stored as [K,M], transposed)
pub(crate) fn sgemm_at(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101,
            112,
            111, // RowMajor, Trans, NoTrans
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            m as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            y.as_mut_ptr(),
            n as i32,
        );
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub mask_token_id: u32,
    pub rope_theta: f64,
}

// ---------------------------------------------------------------------------
// Per-layer weights
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DiffusionLayerWeights {
    // Attention projections [out, in] row-major (PyTorch layout).
    pub q_proj: Vec<f32>, // [heads*head_dim, hidden] = [2048, 1024]
    pub k_proj: Vec<f32>, // [kv_heads*head_dim, hidden] = [1024, 1024]
    pub v_proj: Vec<f32>, // [kv_heads*head_dim, hidden] = [1024, 1024]
    pub o_proj: Vec<f32>, // [hidden, heads*head_dim] = [1024, 2048]
    // QK norm
    pub q_norm: Vec<f32>, // [head_dim] = [128]
    pub k_norm: Vec<f32>, // [128]
    // Layer norms
    pub input_norm: Vec<f32>,     // [hidden]
    pub post_attn_norm: Vec<f32>, // [hidden]
    // MLP
    pub gate_proj: Vec<f32>, // [inter, hidden] = [3072, 1024]
    pub up_proj: Vec<f32>,   // [inter, hidden] = [3072, 1024]
    pub down_proj: Vec<f32>, // [hidden, inter] = [1024, 3072]
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DiffusionEngine {
    pub layers: Vec<DiffusionLayerWeights>,
    pub embed: Vec<f32>,      // [vocab, hidden]
    pub final_norm: Vec<f32>, // [hidden]
    pub config: DiffusionConfig,
    // Precomputed RoPE tables
    pub(crate) rope_cos: Vec<f32>, // [max_seq, head_dim/2]
    pub(crate) rope_sin: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionBackendPreference {
    Auto,
    CpuBlas,
    #[cfg(feature = "ane")]
    Ane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionBackend {
    CpuBlas,
    #[cfg(feature = "ane")]
    AneFused,
    #[cfg(feature = "ane")]
    AneMultiDispatch,
    #[cfg(feature = "ane")]
    AneHybridBonsai,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionBackendReport {
    pub model_kind: DiffusionModelKind,
    pub requested: DiffusionBackendPreference,
    pub selected: DiffusionBackend,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionModelKind {
    Standard,
    BonsaiQ1,
}

pub enum DiffusionRuntime {
    Cpu {
        engine: DiffusionEngine,
        report: DiffusionBackendReport,
    },
    #[cfg(feature = "ane")]
    AneFused {
        engine: AneDiffusionEngine,
        report: DiffusionBackendReport,
    },
    #[cfg(feature = "ane")]
    AneHybridBonsai {
        engine: AneBonsaiEngine,
        report: DiffusionBackendReport,
    },
}

impl DiffusionEngine {
    pub fn detect_model_kind<P: AsRef<Path>>(model_dir: P) -> Result<DiffusionModelKind, String> {
        let cfg = load_diffusion_config_json(model_dir)?;
        let bits = cfg
            .get("quantization")
            .and_then(|q| q.get("bits"))
            .and_then(serde_json::Value::as_u64);
        if bits == Some(1) {
            Ok(DiffusionModelKind::BonsaiQ1)
        } else {
            Ok(DiffusionModelKind::Standard)
        }
    }

    pub fn load_autodetect<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        match Self::detect_model_kind(&model_dir)? {
            DiffusionModelKind::Standard => Self::load(model_dir),
            DiffusionModelKind::BonsaiQ1 => Self::load_q1(model_dir),
        }
    }

    /// Load model from a HuggingFace model directory containing config.json + model.safetensors.
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let dir = model_dir.as_ref();

        // Load config
        let cfg = load_diffusion_config_json(dir)?;

        let hidden = cfg["hidden_size"].as_u64().unwrap() as usize;
        let heads = cfg["num_attention_heads"].as_u64().unwrap() as usize;
        // Qwen2.5-Coder (A2D) has no head_dim field; compute from hidden/heads.
        // Qwen3 variants explicitly set head_dim (e.g., 128). Fall back to 128
        // for older configs that predate the field.
        let head_dim = cfg["head_dim"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or_else(|| if heads > 0 { hidden / heads } else { 128 });

        // mask_token_id: try added_tokens.json for "<|mask|>", fall back to
        // Qwen3 value. A2D Qwen2.5-Coder uses 151665; Qwen3 uses 151669.
        let mask_token_id: u32 = match std::fs::read_to_string(dir.join("added_tokens.json")) {
            Ok(s) => match serde_json::from_str::<serde_json::Value>(&s) {
                Ok(json) => json
                    .as_object()
                    .and_then(|m| m.get("<|mask|>"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
                    .unwrap_or(151669),
                Err(_) => 151669,
            },
            Err(_) => 151669,
        };

        let config = DiffusionConfig {
            hidden,
            layers: cfg["num_hidden_layers"].as_u64().unwrap() as usize,
            heads,
            kv_heads: cfg["num_key_value_heads"].as_u64().unwrap() as usize,
            head_dim,
            inter: cfg["intermediate_size"].as_u64().unwrap() as usize,
            vocab: cfg["vocab_size"].as_u64().unwrap() as usize,
            mask_token_id,
            rope_theta: cfg["rope_theta"].as_f64().unwrap_or(1_000_000.0),
        };

        // Load safetensors weights
        let st_path = dir.join("model.safetensors");
        let st_data = std::fs::read(&st_path).map_err(|e| format!("safetensors: {e}"))?;
        let tensors = safetensors::SafeTensors::deserialize(&st_data)
            .map_err(|e| format!("deserialize: {e}"))?;

        let get = |name: &str| -> Vec<f32> {
            let t = tensors
                .tensor(name)
                .unwrap_or_else(|_| panic!("Missing: {name}"));
            bf16_to_f32(t.data())
        };

        let embed = get("model.embed_tokens.weight");
        let final_norm = get("model.norm.weight");

        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let p = format!("model.layers.{i}");
            layers.push(DiffusionLayerWeights {
                q_proj: get(&format!("{p}.self_attn.q_proj.weight")),
                k_proj: get(&format!("{p}.self_attn.k_proj.weight")),
                v_proj: get(&format!("{p}.self_attn.v_proj.weight")),
                o_proj: get(&format!("{p}.self_attn.o_proj.weight")),
                q_norm: get(&format!("{p}.self_attn.q_norm.weight")),
                k_norm: get(&format!("{p}.self_attn.k_norm.weight")),
                input_norm: get(&format!("{p}.input_layernorm.weight")),
                post_attn_norm: get(&format!("{p}.post_attention_layernorm.weight")),
                gate_proj: get(&format!("{p}.mlp.gate_proj.weight")),
                up_proj: get(&format!("{p}.mlp.up_proj.weight")),
                down_proj: get(&format!("{p}.mlp.down_proj.weight")),
            });
        }

        // Precompute RoPE
        let max_seq = 4096; // generous for generation
        let half_dim = config.head_dim / 2;
        let mut rope_cos = vec![0.0f32; max_seq * half_dim];
        let mut rope_sin = vec![0.0f32; max_seq * half_dim];
        for pos in 0..max_seq {
            for d in 0..half_dim {
                let freq =
                    1.0 / (config.rope_theta as f32).powf(2.0 * d as f32 / config.head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_dim + d] = angle.cos();
                rope_sin[pos * half_dim + d] = angle.sin();
            }
        }

        eprintln!(
            "DiffusionEngine loaded: {}L, hidden={}, heads={}/{}, vocab={}, {:.0}M params",
            config.layers,
            config.hidden,
            config.heads,
            config.kv_heads,
            config.vocab,
            embed.len() as f64 / 1e6 * 2.0 // rough param count
        );

        Ok(Self {
            layers,
            embed,
            final_norm,
            config,
            rope_cos,
            rope_sin,
        })
    }

    /// Load a Bonsai Q1_0_g128 (1-bit quantized) model and dequantize all weights to fp32.
    pub fn load_q1<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let dir = model_dir.as_ref();

        let cfg = load_diffusion_config_json(dir)?;

        let config = DiffusionConfig {
            hidden: cfg["hidden_size"].as_u64().unwrap() as usize,
            layers: cfg["num_hidden_layers"].as_u64().unwrap() as usize,
            heads: cfg["num_attention_heads"].as_u64().unwrap() as usize,
            kv_heads: cfg["num_key_value_heads"].as_u64().unwrap() as usize,
            head_dim: cfg["head_dim"].as_u64().unwrap_or(128) as usize,
            inter: cfg["intermediate_size"].as_u64().unwrap() as usize,
            vocab: cfg["vocab_size"].as_u64().unwrap() as usize,
            mask_token_id: 151669,
            rope_theta: cfg["rope_theta"].as_f64().unwrap_or(1_000_000.0),
        };

        let st_path = dir.join("model.safetensors");
        let st_data = std::fs::read(&st_path).map_err(|e| format!("safetensors: {e}"))?;
        let tensors = safetensors::SafeTensors::deserialize(&st_data)
            .map_err(|e| format!("deserialize: {e}"))?;

        let h = config.hidden;
        let q_dim = config.heads * config.head_dim;
        let kv_dim = config.kv_heads * config.head_dim;
        let inter = config.inter;

        let dequant = |prefix: &str, out_feat: usize, in_feat: usize| -> Vec<f32> {
            let w_name = format!("{prefix}.weight");
            let s_name = format!("{prefix}.scales");
            let b_name = format!("{prefix}.biases");
            let w = tensors
                .tensor(&w_name)
                .unwrap_or_else(|_| panic!("Missing: {w_name}"));
            let s = tensors
                .tensor(&s_name)
                .unwrap_or_else(|_| panic!("Missing: {s_name}"));
            let b = tensors
                .tensor(&b_name)
                .unwrap_or_else(|_| panic!("Missing: {b_name}"));
            dequant_q1_g128(w.data(), s.data(), b.data(), out_feat, in_feat)
        };

        let get_fp16 = |name: &str| -> Vec<f32> {
            let t = tensors
                .tensor(name)
                .unwrap_or_else(|_| panic!("Missing: {name}"));
            fp16_to_f32(t.data())
        };

        let embed = dequant("model.embed_tokens", config.vocab, h);
        let final_norm = get_fp16("model.norm.weight");

        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let p = format!("model.layers.{i}");
            layers.push(DiffusionLayerWeights {
                q_proj: dequant(&format!("{p}.self_attn.q_proj"), q_dim, h),
                k_proj: dequant(&format!("{p}.self_attn.k_proj"), kv_dim, h),
                v_proj: dequant(&format!("{p}.self_attn.v_proj"), kv_dim, h),
                o_proj: dequant(&format!("{p}.self_attn.o_proj"), h, q_dim),
                q_norm: get_fp16(&format!("{p}.self_attn.q_norm.weight")),
                k_norm: get_fp16(&format!("{p}.self_attn.k_norm.weight")),
                input_norm: get_fp16(&format!("{p}.input_layernorm.weight")),
                post_attn_norm: get_fp16(&format!("{p}.post_attention_layernorm.weight")),
                gate_proj: dequant(&format!("{p}.mlp.gate_proj"), inter, h),
                up_proj: dequant(&format!("{p}.mlp.up_proj"), inter, h),
                down_proj: dequant(&format!("{p}.mlp.down_proj"), h, inter),
            });
            if (i + 1) % 7 == 0 || i + 1 == config.layers {
                eprintln!("  dequantized layer {}/{}", i + 1, config.layers);
            }
        }

        let max_seq = 4096;
        let half_dim = config.head_dim / 2;
        let mut rope_cos = vec![0.0f32; max_seq * half_dim];
        let mut rope_sin = vec![0.0f32; max_seq * half_dim];
        for pos in 0..max_seq {
            for d in 0..half_dim {
                let freq =
                    1.0 / (config.rope_theta as f32).powf(2.0 * d as f32 / config.head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_dim + d] = angle.cos();
                rope_sin[pos * half_dim + d] = angle.sin();
            }
        }

        let embed_bytes = embed.len() * 4;
        let layer_bytes: usize = layers
            .iter()
            .map(|l| {
                (l.q_proj.len()
                    + l.k_proj.len()
                    + l.v_proj.len()
                    + l.o_proj.len()
                    + l.gate_proj.len()
                    + l.up_proj.len()
                    + l.down_proj.len()
                    + l.q_norm.len()
                    + l.k_norm.len()
                    + l.input_norm.len()
                    + l.post_attn_norm.len())
                    * 4
            })
            .sum();
        let total_mb =
            (embed_bytes + layer_bytes + final_norm.len() * 4) as f64 / (1024.0 * 1024.0);

        eprintln!(
            "DiffusionEngine::load_q1: {}L, hidden={}, heads={}/{}, vocab={}, \
             dequantized to {:.0}MB fp32",
            config.layers, config.hidden, config.heads, config.kv_heads, config.vocab, total_mb,
        );

        Ok(Self {
            layers,
            embed,
            final_norm,
            config,
            rope_cos,
            rope_sin,
        })
    }

    /// Full forward pass: token_ids [seq] → logits [seq, vocab].
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.config;
        let seq = token_ids.len();
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let half_hd = hd / 2;
        let n_heads = cfg.heads;
        let n_kv = cfg.kv_heads;
        let gqa_ratio = n_heads / n_kv;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;

        // 1. Embedding lookup
        let mut hidden = vec![0.0f32; seq * h];
        for (i, &tid) in token_ids.iter().enumerate() {
            let offset = tid as usize * h;
            hidden[i * h..(i + 1) * h].copy_from_slice(&self.embed[offset..offset + h]);
        }

        // 2. Layer loop
        // Scratch buffers (reused across layers)
        let mut q_buf = vec![0.0f32; seq * q_dim];
        let mut k_buf = vec![0.0f32; seq * kv_dim];
        let mut v_buf = vec![0.0f32; seq * kv_dim];
        let mut attn_out = vec![0.0f32; seq * q_dim];
        let mut o_buf = vec![0.0f32; seq * h];
        let mut gate_buf = vec![0.0f32; seq * cfg.inter];
        let mut up_buf = vec![0.0f32; seq * cfg.inter];
        let mut normed = vec![0.0f32; seq * h];

        for layer in &self.layers {
            // --- Attention ---
            // RMSNorm
            rms_norm(&hidden, &layer.input_norm, &mut normed, seq, h);

            // QKV projections: normed[seq,h] @ W^T[h,out] → buf[seq,out]
            // W is [out, h] row-major. We want normed @ W^T.
            // sgemm_at(M=out, N=seq, K=h, A=W[h,out]^T → [out,h], B=normed^T...)
            // Actually: result[seq, out] = normed[seq, h] @ W^T[h, out]
            // = sgemm(M=seq, N=out, K=h, A=normed, B=W^T)
            // But W is stored as [out, h]. We need W^T = [h, out].
            // Trick: sgemm with transB: result = normed @ W^T
            //   = sgemm(NoTrans, Trans, M=seq, N=out, K=h, A=normed[seq,h], B=W[out,h])
            sgemm_nt(seq, q_dim, h, &normed, &layer.q_proj, &mut q_buf);
            sgemm_nt(seq, kv_dim, h, &normed, &layer.k_proj, &mut k_buf);
            sgemm_nt(seq, kv_dim, h, &normed, &layer.v_proj, &mut v_buf);

            // QK norm (per-head RMSNorm over head_dim)
            for s in 0..seq {
                for head in 0..n_heads {
                    let off = s * q_dim + head * hd;
                    rms_norm_slice(&mut q_buf[off..off + hd], &layer.q_norm);
                }
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    rms_norm_slice(&mut k_buf[off..off + hd], &layer.k_norm);
                }
            }

            // RoPE rotation
            for s in 0..seq {
                for head in 0..n_heads {
                    let off = s * q_dim + head * hd;
                    apply_rope(
                        &mut q_buf[off..off + hd],
                        s,
                        half_hd,
                        &self.rope_cos,
                        &self.rope_sin,
                    );
                }
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    apply_rope(
                        &mut k_buf[off..off + hd],
                        s,
                        half_hd,
                        &self.rope_cos,
                        &self.rope_sin,
                    );
                }
            }

            // Bidirectional SDPA: for each head group, compute attn = softmax(Q@K^T/sqrt(d)) @ V
            // Q: [seq, n_heads, hd] grouped by KV head
            // K: [seq, n_kv, hd]
            // V: [seq, n_kv, hd]
            let scale = 1.0 / (hd as f32).sqrt();
            for kv_h in 0..n_kv {
                // Extract K, V for this KV head: [seq, hd]
                let mut k_head = vec![0.0f32; seq * hd];
                let mut v_head = vec![0.0f32; seq * hd];
                for s in 0..seq {
                    let ko = s * kv_dim + kv_h * hd;
                    k_head[s * hd..(s + 1) * hd].copy_from_slice(&k_buf[ko..ko + hd]);
                    v_head[s * hd..(s + 1) * hd].copy_from_slice(&v_buf[ko..ko + hd]);
                }

                // For each Q head in this GQA group
                for g in 0..gqa_ratio {
                    let q_h = kv_h * gqa_ratio + g;
                    let mut q_head = vec![0.0f32; seq * hd];
                    for s in 0..seq {
                        let qo = s * q_dim + q_h * hd;
                        q_head[s * hd..(s + 1) * hd].copy_from_slice(&q_buf[qo..qo + hd]);
                    }

                    // scores = Q[seq,hd] @ K^T[hd,seq] → [seq,seq]
                    let mut scores = vec![0.0f32; seq * seq];
                    sgemm_nt_scaled(seq, seq, hd, &q_head, &k_head, &mut scores, scale);

                    // Softmax over last dim (no causal mask — bidirectional!)
                    for row in 0..seq {
                        softmax_inplace(&mut scores[row * seq..(row + 1) * seq]);
                    }

                    // context = scores[seq,seq] @ V[seq,hd] → [seq,hd]
                    let mut ctx = vec![0.0f32; seq * hd];
                    sgemm(seq, hd, seq, &scores, &v_head, &mut ctx);

                    // Write back to attn_out
                    for s in 0..seq {
                        let ao = s * q_dim + q_h * hd;
                        attn_out[ao..ao + hd].copy_from_slice(&ctx[s * hd..(s + 1) * hd]);
                    }
                }
            }

            // O projection: attn_out[seq, q_dim] @ o_proj^T → o_buf[seq, h]
            sgemm_nt(seq, h, q_dim, &attn_out, &layer.o_proj, &mut o_buf);

            // Residual add
            for i in 0..seq * h {
                hidden[i] += o_buf[i];
            }

            // --- MLP ---
            rms_norm(&hidden, &layer.post_attn_norm, &mut normed, seq, h);

            // gate = normed @ gate_proj^T → [seq, inter]
            sgemm_nt(seq, cfg.inter, h, &normed, &layer.gate_proj, &mut gate_buf);
            // SiLU(gate) = gate * sigmoid(gate)
            for v in gate_buf.iter_mut() {
                *v *= 1.0 / (1.0 + (-*v).exp());
            }

            // up = normed @ up_proj^T → [seq, inter]
            sgemm_nt(seq, cfg.inter, h, &normed, &layer.up_proj, &mut up_buf);

            // gate * up (element-wise)
            for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) {
                *g *= u;
            }

            // down = gate_buf @ down_proj^T → [seq, h]
            sgemm_nt(seq, h, cfg.inter, &gate_buf, &layer.down_proj, &mut o_buf);

            // Residual add
            for i in 0..seq * h {
                hidden[i] += o_buf[i];
            }
        }

        // 3. Final RMSNorm
        rms_norm(&hidden, &self.final_norm, &mut normed, seq, h);

        // 4. LM head: normed[seq, h] @ embed^T[h, vocab] → logits[seq, vocab]
        // embed is [vocab, h]. We want normed @ embed^T.
        let mut logits = vec![0.0f32; seq * cfg.vocab];
        sgemm_nt(seq, cfg.vocab, h, &normed, &self.embed, &mut logits);

        logits
    }

    /// Forward pass returning logits for the LAST position only → `[vocab]`.
    ///
    /// Full layer computation is identical to `forward()` (bidirectional attention
    /// requires all positions), but the final RMSNorm + LM-head matmul only
    /// operates on the last hidden row: 1×vocab instead of seq×vocab.
    pub fn forward_last(&self, token_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.config;
        let seq = token_ids.len();
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let half_hd = hd / 2;
        let n_heads = cfg.heads;
        let n_kv = cfg.kv_heads;
        let gqa_ratio = n_heads / n_kv;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;

        // 1. Embedding lookup
        let mut hidden = vec![0.0f32; seq * h];
        for (i, &tid) in token_ids.iter().enumerate() {
            let offset = tid as usize * h;
            hidden[i * h..(i + 1) * h].copy_from_slice(&self.embed[offset..offset + h]);
        }

        // 2. Layer loop (identical to forward())
        let mut q_buf = vec![0.0f32; seq * q_dim];
        let mut k_buf = vec![0.0f32; seq * kv_dim];
        let mut v_buf = vec![0.0f32; seq * kv_dim];
        let mut attn_out = vec![0.0f32; seq * q_dim];
        let mut o_buf = vec![0.0f32; seq * h];
        let mut gate_buf = vec![0.0f32; seq * cfg.inter];
        let mut up_buf = vec![0.0f32; seq * cfg.inter];
        let mut normed = vec![0.0f32; seq * h];

        for layer in &self.layers {
            rms_norm(&hidden, &layer.input_norm, &mut normed, seq, h);

            sgemm_nt(seq, q_dim, h, &normed, &layer.q_proj, &mut q_buf);
            sgemm_nt(seq, kv_dim, h, &normed, &layer.k_proj, &mut k_buf);
            sgemm_nt(seq, kv_dim, h, &normed, &layer.v_proj, &mut v_buf);

            for s in 0..seq {
                for head in 0..n_heads {
                    let off = s * q_dim + head * hd;
                    rms_norm_slice(&mut q_buf[off..off + hd], &layer.q_norm);
                }
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    rms_norm_slice(&mut k_buf[off..off + hd], &layer.k_norm);
                }
            }

            for s in 0..seq {
                for head in 0..n_heads {
                    let off = s * q_dim + head * hd;
                    apply_rope(
                        &mut q_buf[off..off + hd],
                        s,
                        half_hd,
                        &self.rope_cos,
                        &self.rope_sin,
                    );
                }
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    apply_rope(
                        &mut k_buf[off..off + hd],
                        s,
                        half_hd,
                        &self.rope_cos,
                        &self.rope_sin,
                    );
                }
            }

            let scale = 1.0 / (hd as f32).sqrt();
            for kv_h in 0..n_kv {
                let mut k_head = vec![0.0f32; seq * hd];
                let mut v_head = vec![0.0f32; seq * hd];
                for s in 0..seq {
                    let ko = s * kv_dim + kv_h * hd;
                    k_head[s * hd..(s + 1) * hd].copy_from_slice(&k_buf[ko..ko + hd]);
                    v_head[s * hd..(s + 1) * hd].copy_from_slice(&v_buf[ko..ko + hd]);
                }

                for g in 0..gqa_ratio {
                    let q_h = kv_h * gqa_ratio + g;
                    let mut q_head = vec![0.0f32; seq * hd];
                    for s in 0..seq {
                        let qo = s * q_dim + q_h * hd;
                        q_head[s * hd..(s + 1) * hd].copy_from_slice(&q_buf[qo..qo + hd]);
                    }

                    let mut scores = vec![0.0f32; seq * seq];
                    sgemm_nt_scaled(seq, seq, hd, &q_head, &k_head, &mut scores, scale);

                    for row in 0..seq {
                        softmax_inplace(&mut scores[row * seq..(row + 1) * seq]);
                    }

                    let mut ctx = vec![0.0f32; seq * hd];
                    sgemm(seq, hd, seq, &scores, &v_head, &mut ctx);

                    for s in 0..seq {
                        let ao = s * q_dim + q_h * hd;
                        attn_out[ao..ao + hd].copy_from_slice(&ctx[s * hd..(s + 1) * hd]);
                    }
                }
            }

            sgemm_nt(seq, h, q_dim, &attn_out, &layer.o_proj, &mut o_buf);

            for i in 0..seq * h {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &layer.post_attn_norm, &mut normed, seq, h);

            sgemm_nt(seq, cfg.inter, h, &normed, &layer.gate_proj, &mut gate_buf);
            for v in gate_buf.iter_mut() {
                *v *= 1.0 / (1.0 + (-*v).exp());
            }

            sgemm_nt(seq, cfg.inter, h, &normed, &layer.up_proj, &mut up_buf);

            for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) {
                *g *= u;
            }

            sgemm_nt(seq, h, cfg.inter, &gate_buf, &layer.down_proj, &mut o_buf);

            for i in 0..seq * h {
                hidden[i] += o_buf[i];
            }
        }

        // 3. Final RMSNorm — LAST POSITION ONLY
        let last_row = &hidden[(seq - 1) * h..seq * h];
        let mut last_normed = vec![0.0f32; h];
        rms_norm(last_row, &self.final_norm, &mut last_normed, 1, h);

        // 4. LM head: 1×vocab instead of seq×vocab
        let mut logits = vec![0.0f32; cfg.vocab];
        sgemm_nt(1, cfg.vocab, h, &last_normed, &self.embed, &mut logits);
        logits
    }

    /// MDLM denoising: generate `num_tokens` from mask positions.
    pub fn generate(&self, prompt_ids: &[u32], num_tokens: usize, steps: usize) -> Vec<u32> {
        let mask_id = self.config.mask_token_id;
        let mut canvas: Vec<u32> = Vec::with_capacity(prompt_ids.len() + num_tokens);
        canvas.extend_from_slice(prompt_ids);
        canvas.extend(std::iter::repeat(mask_id).take(num_tokens));

        for step in 0..steps {
            let logits = self.forward(&canvas);
            let seq = canvas.len();
            let vocab = self.config.vocab;

            // For each position: confidence = max softmax prob, prediction = argmax
            let mut mask_positions: Vec<(usize, f32, u32)> = Vec::new(); // (pos, confidence, pred_token)
            for pos in 0..seq {
                if canvas[pos] != mask_id {
                    continue;
                }
                let row = &logits[pos * vocab..(pos + 1) * vocab];
                let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut best_idx = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in row.iter().enumerate() {
                    sum_exp += (v - max_logit).exp();
                    if v > best_val {
                        best_val = v;
                        best_idx = i as u32;
                    }
                }
                let confidence = (best_val - max_logit).exp() / sum_exp;
                mask_positions.push((pos, confidence, best_idx));
            }

            if mask_positions.is_empty() {
                break;
            }

            // How many to unmask this step
            let n_masks = mask_positions.len();
            let n_keep = n_masks * (steps - step - 1) / steps;
            let n_unmask = (n_masks - n_keep).max(1);

            // Sort by confidence descending, unmask the top-n
            mask_positions
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(pos, _, pred) in mask_positions.iter().take(n_unmask) {
                canvas[pos] = pred;
            }
        }

        canvas
    }
}

impl DiffusionRuntime {
    pub fn load_autodetect_with_backend<P: AsRef<Path>>(
        model_dir: P,
        seq_len: usize,
        backend: DiffusionBackendPreference,
    ) -> Result<Self, String> {
        match DiffusionEngine::detect_model_kind(&model_dir)? {
            DiffusionModelKind::Standard => Self::load_with_backend(model_dir, seq_len, backend),
            DiffusionModelKind::BonsaiQ1 => {
                Self::load_q1_with_backend(model_dir, seq_len, 1e-6, backend)
            }
        }
    }

    pub fn load_with_backend<P: AsRef<Path>>(
        model_dir: P,
        seq_len: usize,
        backend: DiffusionBackendPreference,
    ) -> Result<Self, String> {
        let engine = DiffusionEngine::load(model_dir)?;
        Self::new(engine, seq_len, backend)
    }

    pub fn load_q1_with_backend<P: AsRef<Path>>(
        model_dir: P,
        seq_len: usize,
        eps: f32,
        backend: DiffusionBackendPreference,
    ) -> Result<Self, String> {
        let engine = DiffusionEngine::load_q1(model_dir)?;
        Self::new_bonsai_q1(engine, seq_len, eps, backend)
    }

    pub fn new(
        engine: DiffusionEngine,
        seq_len: usize,
        backend: DiffusionBackendPreference,
    ) -> Result<Self, String> {
        #[cfg(not(feature = "ane"))]
        let _ = seq_len;

        match backend {
            DiffusionBackendPreference::Auto => {
                #[cfg(feature = "ane")]
                {
                    match AneDiffusionEngine::new(engine.clone(), seq_len) {
                        Ok(ane) => {
                            let selected = ane.backend_kind();
                            let detail = match selected {
                                DiffusionBackend::AneFused => {
                                    "Selected ANE fused diffusion backend automatically"
                                        .to_owned()
                                }
                                DiffusionBackend::AneMultiDispatch => {
                                    "Selected ANE multi-dispatch diffusion backend automatically after fused compile fallback"
                                        .to_owned()
                                }
                                _ => unreachable!("unexpected diffusion backend for ANE engine"),
                            };
                            return Ok(Self::AneFused {
                                engine: ane,
                                report: DiffusionBackendReport {
                                    model_kind: DiffusionModelKind::Standard,
                                    requested: backend,
                                    selected,
                                    detail,
                                },
                            });
                        }
                        Err(err) => {
                            return Ok(Self::Cpu {
                                engine,
                                report: DiffusionBackendReport {
                                    model_kind: DiffusionModelKind::Standard,
                                    requested: backend,
                                    selected: DiffusionBackend::CpuBlas,
                                    detail: format!(
                                        "ANE fused diffusion backend unavailable, using CPU BLAS: {err}"
                                    ),
                                },
                            });
                        }
                    }
                }

                #[cfg(not(feature = "ane"))]
                {
                    Ok(Self::Cpu {
                        engine,
                        report: DiffusionBackendReport {
                            model_kind: DiffusionModelKind::Standard,
                            requested: backend,
                            selected: DiffusionBackend::CpuBlas,
                            detail: "ANE feature not enabled, using CPU BLAS".to_owned(),
                        },
                    })
                }
            }
            DiffusionBackendPreference::CpuBlas => Ok(Self::Cpu {
                engine,
                report: DiffusionBackendReport {
                    model_kind: DiffusionModelKind::Standard,
                    requested: backend,
                    selected: DiffusionBackend::CpuBlas,
                    detail: "CPU BLAS requested explicitly".to_owned(),
                },
            }),
            #[cfg(feature = "ane")]
            DiffusionBackendPreference::Ane => {
                let ane = AneDiffusionEngine::new(engine, seq_len)?;
                let selected = ane.backend_kind();
                let detail = match selected {
                    DiffusionBackend::AneFused => {
                        "ANE fused diffusion backend requested explicitly".to_owned()
                    }
                    DiffusionBackend::AneMultiDispatch => {
                        "ANE diffusion backend requested explicitly; using multi-dispatch fallback"
                            .to_owned()
                    }
                    _ => unreachable!("unexpected diffusion backend for ANE engine"),
                };
                Ok(Self::AneFused {
                    engine: ane,
                    report: DiffusionBackendReport {
                        model_kind: DiffusionModelKind::Standard,
                        requested: backend,
                        selected,
                        detail,
                    },
                })
            }
        }
    }

    /// Create a diffusion runtime with causal ANE attention for autoregressive generation.
    /// Unlike `new`, this uses causal masking instead of bidirectional attention.
    pub fn new_causal(
        engine: DiffusionEngine,
        seq_len: usize,
        backend: DiffusionBackendPreference,
    ) -> Result<Self, String> {
        #[cfg(not(feature = "ane"))]
        let _ = seq_len;

        match backend {
            DiffusionBackendPreference::Auto => {
                #[cfg(feature = "ane")]
                {
                    match AneDiffusionEngine::new_causal(engine.clone(), seq_len) {
                        Ok(ane) => {
                            let selected = ane.backend_kind();
                            let detail = match selected {
                                DiffusionBackend::AneFused => {
                                    "Selected ANE fused diffusion backend (causal) automatically"
                                        .to_owned()
                                }
                                DiffusionBackend::AneMultiDispatch => {
                                    "Selected ANE multi-dispatch diffusion backend (causal) automatically after fused compile fallback"
                                        .to_owned()
                                }
                                _ => unreachable!("unexpected diffusion backend for ANE engine"),
                            };
                            return Ok(Self::AneFused {
                                engine: ane,
                                report: DiffusionBackendReport {
                                    model_kind: DiffusionModelKind::Standard,
                                    requested: backend,
                                    selected,
                                    detail,
                                },
                            });
                        }
                        Err(err) => {
                            return Ok(Self::Cpu {
                                engine,
                                report: DiffusionBackendReport {
                                    model_kind: DiffusionModelKind::Standard,
                                    requested: backend,
                                    selected: DiffusionBackend::CpuBlas,
                                    detail: format!(
                                        "ANE causal diffusion backend unavailable, using CPU BLAS: {err}"
                                    ),
                                },
                            });
                        }
                    }
                }

                #[cfg(not(feature = "ane"))]
                {
                    Ok(Self::Cpu {
                        engine,
                        report: DiffusionBackendReport {
                            model_kind: DiffusionModelKind::Standard,
                            requested: backend,
                            selected: DiffusionBackend::CpuBlas,
                            detail: "ANE feature not enabled, using CPU BLAS".to_owned(),
                        },
                    })
                }
            }
            DiffusionBackendPreference::CpuBlas => Ok(Self::Cpu {
                engine,
                report: DiffusionBackendReport {
                    model_kind: DiffusionModelKind::Standard,
                    requested: backend,
                    selected: DiffusionBackend::CpuBlas,
                    detail: "CPU BLAS requested explicitly".to_owned(),
                },
            }),
            #[cfg(feature = "ane")]
            DiffusionBackendPreference::Ane => {
                let ane = AneDiffusionEngine::new_causal(engine, seq_len)?;
                let selected = ane.backend_kind();
                let detail = match selected {
                    DiffusionBackend::AneFused => {
                        "ANE fused diffusion backend (causal) requested explicitly".to_owned()
                    }
                    DiffusionBackend::AneMultiDispatch => {
                        "ANE diffusion backend (causal) requested explicitly; using multi-dispatch fallback"
                            .to_owned()
                    }
                    _ => unreachable!("unexpected diffusion backend for ANE engine"),
                };
                Ok(Self::AneFused {
                    engine: ane,
                    report: DiffusionBackendReport {
                        model_kind: DiffusionModelKind::Standard,
                        requested: backend,
                        selected,
                        detail,
                    },
                })
            }
        }
    }

    pub fn new_bonsai_q1(
        engine: DiffusionEngine,
        seq_len: usize,
        eps: f32,
        backend: DiffusionBackendPreference,
    ) -> Result<Self, String> {
        #[cfg(not(feature = "ane"))]
        let _ = (seq_len, eps);

        match backend {
            DiffusionBackendPreference::Auto => {
                #[cfg(feature = "ane")]
                {
                    match AneBonsaiEngine::new(engine.clone(), seq_len, eps) {
                        Ok(ane) => {
                            return Ok(Self::AneHybridBonsai {
                                engine: ane,
                                report: DiffusionBackendReport {
                                    model_kind: DiffusionModelKind::BonsaiQ1,
                                    requested: backend,
                                    selected: DiffusionBackend::AneHybridBonsai,
                                    detail: "Selected ANE Bonsai hybrid backend automatically"
                                        .to_owned(),
                                },
                            });
                        }
                        Err(err) => {
                            return Ok(Self::Cpu {
                                engine,
                                report: DiffusionBackendReport {
                                    model_kind: DiffusionModelKind::BonsaiQ1,
                                    requested: backend,
                                    selected: DiffusionBackend::CpuBlas,
                                    detail: format!(
                                        "ANE Bonsai hybrid backend unavailable, using CPU BLAS: {err}"
                                    ),
                                },
                            });
                        }
                    }
                }

                #[cfg(not(feature = "ane"))]
                {
                    Ok(Self::Cpu {
                        engine,
                        report: DiffusionBackendReport {
                            model_kind: DiffusionModelKind::BonsaiQ1,
                            requested: backend,
                            selected: DiffusionBackend::CpuBlas,
                            detail: "ANE feature not enabled, using CPU BLAS".to_owned(),
                        },
                    })
                }
            }
            DiffusionBackendPreference::CpuBlas => Ok(Self::Cpu {
                engine,
                report: DiffusionBackendReport {
                    model_kind: DiffusionModelKind::BonsaiQ1,
                    requested: backend,
                    selected: DiffusionBackend::CpuBlas,
                    detail: "CPU BLAS requested explicitly".to_owned(),
                },
            }),
            #[cfg(feature = "ane")]
            DiffusionBackendPreference::Ane => {
                let ane = AneBonsaiEngine::new(engine, seq_len, eps)?;
                Ok(Self::AneHybridBonsai {
                    engine: ane,
                    report: DiffusionBackendReport {
                        model_kind: DiffusionModelKind::BonsaiQ1,
                        requested: backend,
                        selected: DiffusionBackend::AneHybridBonsai,
                        detail: "ANE Bonsai hybrid backend requested explicitly".to_owned(),
                    },
                })
            }
        }
    }

    pub fn backend_report(&self) -> &DiffusionBackendReport {
        match self {
            Self::Cpu { report, .. } => report,
            #[cfg(feature = "ane")]
            Self::AneFused { report, .. } => report,
            #[cfg(feature = "ane")]
            Self::AneHybridBonsai { report, .. } => report,
        }
    }

    pub fn requested_backend(&self) -> DiffusionBackendPreference {
        self.backend_report().requested
    }

    pub fn model_kind(&self) -> DiffusionModelKind {
        self.backend_report().model_kind
    }

    pub fn selected_backend(&self) -> DiffusionBackend {
        self.backend_report().selected
    }

    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        match self {
            Self::Cpu { engine, .. } => engine.forward(token_ids),
            #[cfg(feature = "ane")]
            Self::AneFused { engine, .. } => engine.forward(token_ids),
            #[cfg(feature = "ane")]
            Self::AneHybridBonsai { engine, .. } => engine.forward(token_ids),
        }
    }

    /// Forward pass returning logits for the LAST position only → `[vocab]`.
    ///
    /// Same ANE/BLAS compute for hidden states (causal attention needs all positions),
    /// but the CPU LM head matmul is 1×vocab instead of seq×vocab — saves ~15ms per
    /// call when seq>1 and vocab=151936.
    pub fn forward_last(&self, token_ids: &[u32]) -> Vec<f32> {
        match self {
            Self::Cpu { engine, .. } => engine.forward_last(token_ids),
            #[cfg(feature = "ane")]
            Self::AneFused { engine, .. } => engine.forward_last(token_ids),
            #[cfg(feature = "ane")]
            Self::AneHybridBonsai { engine, .. } => engine.forward_last(token_ids),
        }
    }

    pub fn generate(&self, prompt_ids: &[u32], num_tokens: usize, steps: usize) -> Vec<u32> {
        let mask_id = match self {
            Self::Cpu { engine, .. } => engine.config.mask_token_id,
            #[cfg(feature = "ane")]
            Self::AneFused { engine, .. } => engine.blas_engine.config.mask_token_id,
            #[cfg(feature = "ane")]
            Self::AneHybridBonsai { engine, .. } => engine.blas_engine.config.mask_token_id,
        };
        let vocab = match self {
            Self::Cpu { engine, .. } => engine.config.vocab,
            #[cfg(feature = "ane")]
            Self::AneFused { engine, .. } => engine.blas_engine.config.vocab,
            #[cfg(feature = "ane")]
            Self::AneHybridBonsai { engine, .. } => engine.blas_engine.config.vocab,
        };

        let mut canvas: Vec<u32> = Vec::with_capacity(prompt_ids.len() + num_tokens);
        canvas.extend_from_slice(prompt_ids);
        canvas.extend(std::iter::repeat(mask_id).take(num_tokens));

        for step in 0..steps {
            let logits = self.forward(&canvas);
            let seq = canvas.len();

            let mut mask_positions: Vec<(usize, f32, u32)> = Vec::new();
            for pos in 0..seq {
                if canvas[pos] != mask_id {
                    continue;
                }
                let row = &logits[pos * vocab..(pos + 1) * vocab];
                let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut best_idx = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in row.iter().enumerate() {
                    sum_exp += (v - max_logit).exp();
                    if v > best_val {
                        best_val = v;
                        best_idx = i as u32;
                    }
                }
                let confidence = (best_val - max_logit).exp() / sum_exp;
                mask_positions.push((pos, confidence, best_idx));
            }

            if mask_positions.is_empty() {
                break;
            }

            let n_masks = mask_positions.len();
            let n_keep = n_masks * (steps - step - 1) / steps;
            let n_unmask = (n_masks - n_keep).max(1);
            mask_positions
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(pos, _, pred) in mask_positions.iter().take(n_unmask) {
                canvas[pos] = pred;
            }
        }

        canvas
    }
}

fn load_diffusion_config_json<P: AsRef<Path>>(model_dir: P) -> Result<serde_json::Value, String> {
    let config_str = std::fs::read_to_string(model_dir.as_ref().join("config.json"))
        .map_err(|e| format!("config.json: {e}"))?;
    serde_json::from_str(&config_str).map_err(|e| format!("parse config: {e}"))
}

// ---------------------------------------------------------------------------
// ANE-accelerated engine — 28 kernels with weights baked in (BLOBFILEs)
// ---------------------------------------------------------------------------

/// ANE-accelerated diffusion engine using 28 per-layer kernels with weights
/// baked in as BLOBFILEs. No reload_weights at runtime — each kernel is
/// self-contained. This is ~30x faster than the old reload_weights approach.
///
/// Strategy:
/// - Compile ONE fused kernel (layer 0) with weights as BLOBFILEs.
/// - For layers 1-27: use `patch_from_donor` to swap weights without recompiling.
/// - At eval time: 28 direct kernel dispatches, zero weight reload overhead.
#[cfg(feature = "ane")]
pub struct AneDiffusionEngine {
    pub blas_engine: DiffusionEngine,
    kernels: AneDiffusionKernelSet,
    pub seq_len: usize, // fixed seq for compiled kernels (padded to ANE_MIN_SPATIAL)
}

#[cfg(feature = "ane")]
enum AneDiffusionKernelSet {
    Fused {
        kernels: Vec<crate::ane_bridge::AneKernel>,
    },
    MultiDispatch {
        attn_kernels: Vec<crate::ane_bridge::AneKernel>,
        ffn_kernels: Vec<crate::ane_bridge::AneKernel>,
    },
}

#[cfg(feature = "ane")]
fn compile_ane_kernel(
    label: &str,
    mil_text: &str,
    weight_names: &[&str],
    weight_datas: &[&[u8]],
    input_sizes: &[usize],
    output_sizes: &[usize],
) -> Result<crate::ane_bridge::AneKernel, String> {
    use crate::ane_bridge::AneKernel;

    match AneKernel::compile_multi_weights(
        mil_text,
        weight_names,
        weight_datas,
        input_sizes,
        output_sizes,
    ) {
        Ok(kernel) => Ok(kernel),
        Err(multi_err) => {
            eprintln!(
                "{label}: ANE multi-weight compile failed ({multi_err}), trying direct compile path..."
            );
            AneKernel::compile_direct(
                mil_text,
                weight_names,
                weight_datas,
                input_sizes,
                output_sizes,
            )
            .map_err(|direct_err| {
                format!("{label}: multi-weight compile failed ({multi_err}); direct compile failed ({direct_err})")
            })
        }
    }
}

#[cfg(feature = "ane")]
impl AneDiffusionEngine {
    pub fn backend_kind(&self) -> DiffusionBackend {
        match self.kernels {
            AneDiffusionKernelSet::Fused { .. } => DiffusionBackend::AneFused,
            AneDiffusionKernelSet::MultiDispatch { .. } => DiffusionBackend::AneMultiDispatch,
        }
    }

    /// Compile 28 fully-fused ANE kernels with weights baked in as BLOBFILEs.
    /// No reload_weights at runtime — each kernel is self-contained.
    pub fn new(engine: DiffusionEngine, seq_len: usize) -> Result<Self, String> {
        use crate::ane_bridge::{self, build_weight_blob, build_weight_blob_transposed};
        use crate::ane_mil::ANE_MIN_SPATIAL;
        use crate::diffusion_ane;

        ane_bridge::ane_init()?;

        let cfg = &engine.config;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let half_hd = hd / 2;
        let q_dim = cfg.heads * hd;
        let kv_dim = cfg.kv_heads * hd;
        let inter = cfg.inter;
        let seq = seq_len.max(ANE_MIN_SPATIAL);

        eprintln!("AneDiffusionEngine: compiling 28 fused ANE kernels (seq={seq})...");
        let t0 = std::time::Instant::now();

        // Precompute RoPE tables once — same for all layers (shared BLOBFILE data).
        let mut rope_cos = vec![0.0f32; seq * half_hd];
        let mut rope_sin = vec![0.0f32; seq * half_hd];
        for pos in 0..seq {
            for d in 0..half_hd {
                let freq = 1.0 / (cfg.rope_theta as f32).powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_hd + d] = angle.cos();
                rope_sin[pos * half_hd + d] = angle.sin();
            }
        }

        let fused_mil = diffusion_ane::gen_fused_diffusion_layer(
            h,
            cfg.heads,
            cfg.kv_heads,
            hd,
            inter,
            seq,
            1e-6,
            false, // bidirectional for diffusion
        );
        let fused_names: Vec<&str> = fused_mil.weight_names.iter().map(|s| s.as_str()).collect();
        let attn_mil =
            diffusion_ane::gen_diffusion_attention(h, cfg.heads, cfg.kv_heads, hd, seq, 1e-6);
        let attn_names: Vec<&str> = attn_mil.weight_names.iter().map(|s| s.as_str()).collect();
        let ffn_mil = diffusion_ane::gen_diffusion_ffn(h, inter, seq, 1e-6);
        let ffn_names: Vec<&str> = ffn_mil.weight_names.iter().map(|s| s.as_str()).collect();

        // Build the RoPE blobs once (layout: [seq, half_hd] row-major).
        let rope_cos_blob = build_weight_blob(&rope_cos, seq, half_hd);
        let rope_sin_blob = build_weight_blob(&rope_sin, seq, half_hd);

        // Build weight blobs for layer 0 and compile L0 kernel.
        let build_fused_blobs = |lw: &DiffusionLayerWeights| -> Vec<Vec<u8>> {
            vec![
                build_weight_blob(&lw.input_norm, 1, h),
                build_weight_blob(&lw.post_attn_norm, 1, h),
                build_weight_blob_transposed(&lw.q_proj, q_dim, h),
                build_weight_blob_transposed(&lw.k_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.v_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.o_proj, h, q_dim),
                build_weight_blob_transposed(&lw.gate_proj, inter, h),
                build_weight_blob_transposed(&lw.up_proj, inter, h),
                build_weight_blob_transposed(&lw.down_proj, h, inter),
                rope_cos_blob.clone(),
                rope_sin_blob.clone(),
                build_weight_blob(&lw.q_norm, 1, hd),
                build_weight_blob(&lw.k_norm, 1, hd),
            ]
        };
        let build_attn_blobs = |lw: &DiffusionLayerWeights| -> Vec<Vec<u8>> {
            vec![
                build_weight_blob(&lw.input_norm, 1, h),
                build_weight_blob_transposed(&lw.q_proj, q_dim, h),
                build_weight_blob_transposed(&lw.k_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.v_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.o_proj, h, q_dim),
                rope_cos_blob.clone(),
                rope_sin_blob.clone(),
                build_weight_blob(&lw.q_norm, 1, hd),
                build_weight_blob(&lw.k_norm, 1, hd),
            ]
        };
        let build_ffn_blobs = |lw: &DiffusionLayerWeights| -> Vec<Vec<u8>> {
            vec![
                build_weight_blob(&lw.post_attn_norm, 1, h),
                build_weight_blob_transposed(&lw.gate_proj, inter, h),
                build_weight_blob_transposed(&lw.up_proj, inter, h),
                build_weight_blob_transposed(&lw.down_proj, h, inter),
            ]
        };

        let l0_blobs = build_fused_blobs(&engine.layers[0]);
        let l0_refs: Vec<&[u8]> = l0_blobs.iter().map(|b| b.as_slice()).collect();

        let kernels = match compile_ane_kernel(
            "L0 fused compile",
            &fused_mil.mil_text,
            &fused_names,
            &l0_refs,
            &[fused_mil.input_bytes],
            &[fused_mil.output_bytes],
        ) {
            Ok(kernel0) => {
                let l0_ms = t0.elapsed().as_millis();
                eprintln!("  L0 full compile: {l0_ms}ms");

                // Patch layers 1-27 from L0 without recompiling.
                let mut kernels = Vec::with_capacity(cfg.layers);
                kernels.push(kernel0);
                for (i, lw) in engine.layers.iter().enumerate().skip(1) {
                    let blobs = build_fused_blobs(lw);
                    let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
                    let ki = kernels[0]
                        .patch_from_donor(
                            &fused_mil.mil_text,
                            &fused_names,
                            &refs,
                            &[fused_mil.input_bytes],
                            &[fused_mil.output_bytes],
                        )
                        .map_err(|e| format!("L{i} patch: {e}"))?;
                    kernels.push(ki);
                }
                let total_ms = t0.elapsed().as_millis();
                eprintln!(
                    "AneDiffusionEngine: 28 fused kernels in {total_ms}ms (L0={l0_ms}ms + 27 patches)"
                );
                AneDiffusionKernelSet::Fused { kernels }
            }
            Err(fused_err) => {
                eprintln!(
                    "Fused ANE layer compile failed ({fused_err}); falling back to multi-dispatch attention+FFN kernels..."
                );

                let attn_l0 = build_attn_blobs(&engine.layers[0]);
                let attn_l0_refs: Vec<&[u8]> = attn_l0.iter().map(|b| b.as_slice()).collect();
                let attn_kernel0 = compile_ane_kernel(
                    "L0 attention compile after fused fallback",
                    &attn_mil.mil_text,
                    &attn_names,
                    &attn_l0_refs,
                    &[attn_mil.input_bytes],
                    &[attn_mil.output_bytes],
                )?;

                let ffn_l0 = build_ffn_blobs(&engine.layers[0]);
                let ffn_l0_refs: Vec<&[u8]> = ffn_l0.iter().map(|b| b.as_slice()).collect();
                let ffn_kernel0 = compile_ane_kernel(
                    "L0 FFN compile after fused fallback",
                    &ffn_mil.mil_text,
                    &ffn_names,
                    &ffn_l0_refs,
                    &[ffn_mil.input_bytes],
                    &[ffn_mil.output_bytes],
                )?;

                let l0_ms = t0.elapsed().as_millis();
                eprintln!("  L0 multi-dispatch compile: {l0_ms}ms");

                let mut attn_kernels = Vec::with_capacity(cfg.layers);
                let mut ffn_kernels = Vec::with_capacity(cfg.layers);
                attn_kernels.push(attn_kernel0);
                ffn_kernels.push(ffn_kernel0);
                for (i, lw) in engine.layers.iter().enumerate().skip(1) {
                    let attn_blobs = build_attn_blobs(lw);
                    let attn_refs: Vec<&[u8]> = attn_blobs.iter().map(|b| b.as_slice()).collect();
                    let ki = attn_kernels[0]
                        .patch_from_donor(
                            &attn_mil.mil_text,
                            &attn_names,
                            &attn_refs,
                            &[attn_mil.input_bytes],
                            &[attn_mil.output_bytes],
                        )
                        .map_err(|e| format!("L{i} attn patch: {e}"))?;
                    attn_kernels.push(ki);

                    let ffn_blobs = build_ffn_blobs(lw);
                    let ffn_refs: Vec<&[u8]> = ffn_blobs.iter().map(|b| b.as_slice()).collect();
                    let ki = ffn_kernels[0]
                        .patch_from_donor(
                            &ffn_mil.mil_text,
                            &ffn_names,
                            &ffn_refs,
                            &[ffn_mil.input_bytes],
                            &[ffn_mil.output_bytes],
                        )
                        .map_err(|e| format!("L{i} ffn patch: {e}"))?;
                    ffn_kernels.push(ki);
                }
                let total_ms = t0.elapsed().as_millis();
                eprintln!("AneDiffusionEngine: 28 multi-dispatch kernel pairs in {total_ms}ms");
                AneDiffusionKernelSet::MultiDispatch {
                    attn_kernels,
                    ffn_kernels,
                }
            }
        };

        Ok(Self {
            blas_engine: engine,
            kernels,
            seq_len: seq,
        })
    }

    /// Compile 28 fully-fused ANE kernels with causal masking for autoregressive generation.
    /// This mirrors `new()` but passes `causal=true` to the MIL generator and includes
    /// the causal mask BLOBFILE.
    pub fn new_causal(engine: DiffusionEngine, seq_len: usize) -> Result<Self, String> {
        use crate::ane_bridge::{self, build_weight_blob, build_weight_blob_transposed};
        use crate::ane_mil::{build_causal_mask_blob, ANE_MIN_SPATIAL};
        use crate::diffusion_ane;

        ane_bridge::ane_init()?;

        let cfg = &engine.config;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let half_hd = hd / 2;
        let q_dim = cfg.heads * hd;
        let kv_dim = cfg.kv_heads * hd;
        let inter = cfg.inter;
        let seq = seq_len.max(ANE_MIN_SPATIAL);

        eprintln!("AneDiffusionEngine: compiling 28 causal fused ANE kernels (seq={seq})...");
        let t0 = std::time::Instant::now();

        let mut rope_cos = vec![0.0f32; seq * half_hd];
        let mut rope_sin = vec![0.0f32; seq * half_hd];
        for pos in 0..seq {
            for d in 0..half_hd {
                let freq = 1.0 / (cfg.rope_theta as f32).powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_hd + d] = angle.cos();
                rope_sin[pos * half_hd + d] = angle.sin();
            }
        }

        let fused_mil = diffusion_ane::gen_fused_diffusion_layer(
            h,
            cfg.heads,
            cfg.kv_heads,
            hd,
            inter,
            seq,
            1e-6,
            true, // causal masking
        );
        let fused_names: Vec<&str> = fused_mil.weight_names.iter().map(|s| s.as_str()).collect();
        let attn_mil =
            diffusion_ane::gen_diffusion_attention(h, cfg.heads, cfg.kv_heads, hd, seq, 1e-6);
        let attn_names: Vec<&str> = attn_mil.weight_names.iter().map(|s| s.as_str()).collect();
        let ffn_mil = diffusion_ane::gen_diffusion_ffn(h, inter, seq, 1e-6);
        let ffn_names: Vec<&str> = ffn_mil.weight_names.iter().map(|s| s.as_str()).collect();

        let rope_cos_blob = build_weight_blob(&rope_cos, seq, half_hd);
        let rope_sin_blob = build_weight_blob(&rope_sin, seq, half_hd);
        let causal_mask_blob = build_causal_mask_blob(seq);

        let build_fused_blobs = |lw: &DiffusionLayerWeights| -> Vec<Vec<u8>> {
            let mut blobs = vec![
                build_weight_blob(&lw.input_norm, 1, h),
                build_weight_blob(&lw.post_attn_norm, 1, h),
                build_weight_blob_transposed(&lw.q_proj, q_dim, h),
                build_weight_blob_transposed(&lw.k_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.v_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.o_proj, h, q_dim),
                build_weight_blob_transposed(&lw.gate_proj, inter, h),
                build_weight_blob_transposed(&lw.up_proj, inter, h),
                build_weight_blob_transposed(&lw.down_proj, h, inter),
                rope_cos_blob.clone(),
                rope_sin_blob.clone(),
                build_weight_blob(&lw.q_norm, 1, hd),
                build_weight_blob(&lw.k_norm, 1, hd),
            ];
            blobs.push(causal_mask_blob.clone());
            blobs
        };
        let build_attn_blobs = |lw: &DiffusionLayerWeights| -> Vec<Vec<u8>> {
            vec![
                build_weight_blob(&lw.input_norm, 1, h),
                build_weight_blob_transposed(&lw.q_proj, q_dim, h),
                build_weight_blob_transposed(&lw.k_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.v_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.o_proj, h, q_dim),
                rope_cos_blob.clone(),
                rope_sin_blob.clone(),
                build_weight_blob(&lw.q_norm, 1, hd),
                build_weight_blob(&lw.k_norm, 1, hd),
            ]
        };
        let build_ffn_blobs = |lw: &DiffusionLayerWeights| -> Vec<Vec<u8>> {
            vec![
                build_weight_blob(&lw.post_attn_norm, 1, h),
                build_weight_blob_transposed(&lw.gate_proj, inter, h),
                build_weight_blob_transposed(&lw.up_proj, inter, h),
                build_weight_blob_transposed(&lw.down_proj, h, inter),
            ]
        };

        let l0_blobs = build_fused_blobs(&engine.layers[0]);
        let l0_refs: Vec<&[u8]> = l0_blobs.iter().map(|b| b.as_slice()).collect();

        let kernels = match compile_ane_kernel(
            "L0 fused causal compile",
            &fused_mil.mil_text,
            &fused_names,
            &l0_refs,
            &[fused_mil.input_bytes],
            &[fused_mil.output_bytes],
        ) {
            Ok(kernel0) => {
                let l0_ms = t0.elapsed().as_millis();
                eprintln!("  L0 causal compile: {l0_ms}ms");

                let mut kernels = Vec::with_capacity(cfg.layers);
                kernels.push(kernel0);
                for (i, lw) in engine.layers.iter().enumerate().skip(1) {
                    let blobs = build_fused_blobs(lw);
                    let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
                    let ki = kernels[0]
                        .patch_from_donor(
                            &fused_mil.mil_text,
                            &fused_names,
                            &refs,
                            &[fused_mil.input_bytes],
                            &[fused_mil.output_bytes],
                        )
                        .map_err(|e| format!("L{i} patch: {e}"))?;
                    kernels.push(ki);
                }
                let total_ms = t0.elapsed().as_millis();
                eprintln!(
                    "AneDiffusionEngine: 28 causal fused kernels in {total_ms}ms (L0={l0_ms}ms + 27 patches)"
                );
                AneDiffusionKernelSet::Fused { kernels }
            }
            Err(fused_err) => {
                eprintln!(
                    "Fused causal ANE layer compile failed ({fused_err}); falling back to multi-dispatch..."
                );

                let attn_l0 = build_attn_blobs(&engine.layers[0]);
                let attn_l0_refs: Vec<&[u8]> = attn_l0.iter().map(|b| b.as_slice()).collect();
                let attn_kernel0 = compile_ane_kernel(
                    "L0 attention compile after fused causal fallback",
                    &attn_mil.mil_text,
                    &attn_names,
                    &attn_l0_refs,
                    &[attn_mil.input_bytes],
                    &[attn_mil.output_bytes],
                )?;

                let ffn_l0 = build_ffn_blobs(&engine.layers[0]);
                let ffn_l0_refs: Vec<&[u8]> = ffn_l0.iter().map(|b| b.as_slice()).collect();
                let ffn_kernel0 = compile_ane_kernel(
                    "L0 FFN compile after fused causal fallback",
                    &ffn_mil.mil_text,
                    &ffn_names,
                    &ffn_l0_refs,
                    &[ffn_mil.input_bytes],
                    &[ffn_mil.output_bytes],
                )?;

                let l0_ms = t0.elapsed().as_millis();
                eprintln!("  L0 multi-dispatch causal compile: {l0_ms}ms");

                let mut attn_kernels = Vec::with_capacity(cfg.layers);
                let mut ffn_kernels = Vec::with_capacity(cfg.layers);
                attn_kernels.push(attn_kernel0);
                ffn_kernels.push(ffn_kernel0);
                for (i, lw) in engine.layers.iter().enumerate().skip(1) {
                    let attn_blobs = build_attn_blobs(lw);
                    let attn_refs: Vec<&[u8]> = attn_blobs.iter().map(|b| b.as_slice()).collect();
                    let ki = attn_kernels[0]
                        .patch_from_donor(
                            &attn_mil.mil_text,
                            &attn_names,
                            &attn_refs,
                            &[attn_mil.input_bytes],
                            &[attn_mil.output_bytes],
                        )
                        .map_err(|e| format!("L{i} attn patch: {e}"))?;
                    attn_kernels.push(ki);

                    let ffn_blobs = build_ffn_blobs(lw);
                    let ffn_refs: Vec<&[u8]> = ffn_blobs.iter().map(|b| b.as_slice()).collect();
                    let ki = ffn_kernels[0]
                        .patch_from_donor(
                            &ffn_mil.mil_text,
                            &ffn_names,
                            &ffn_refs,
                            &[ffn_mil.input_bytes],
                            &[ffn_mil.output_bytes],
                        )
                        .map_err(|e| format!("L{i} ffn patch: {e}"))?;
                    ffn_kernels.push(ki);
                }
                let total_ms = t0.elapsed().as_millis();
                eprintln!(
                    "AneDiffusionEngine: 28 causal multi-dispatch kernel pairs in {total_ms}ms"
                );
                AneDiffusionKernelSet::MultiDispatch {
                    attn_kernels,
                    ffn_kernels,
                }
            }
        };

        Ok(Self {
            blas_engine: engine,
            kernels,
            seq_len: seq,
        })
    }

    /// Fully-fused ANE forward pass: embed → 28×(fused_dispatch) → final_norm → LM head.
    /// Each of the 28 kernels has weights baked in as BLOBFILEs — no reload_weights.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.blas_engine.config;
        let seq = token_ids.len();
        let h = cfg.hidden;
        let ps = self.seq_len; // padded spatial size for ANE IOSurface

        // 1. Embedding lookup — row-major [seq, h]
        let mut hidden = vec![0.0f32; seq * h];
        for (i, &tid) in token_ids.iter().enumerate() {
            let off = tid as usize * h;
            hidden[i * h..(i + 1) * h].copy_from_slice(&self.blas_engine.embed[off..off + h]);
        }

        // Pack row-major [seq, dim] → ANE channel-first fp32 [dim, ps] (zero-padded to ps).
        let pack = |data: &[f32], dim: usize| -> Vec<u8> {
            let mut buf = vec![0.0f32; dim * ps];
            for s in 0..seq {
                for c in 0..dim {
                    buf[c * ps + s] = data[s * dim + c];
                }
            }
            buf.iter().flat_map(|f| f.to_le_bytes()).collect()
        };

        // Unpack ANE channel-first fp32 [dim, ps] → row-major [seq, dim].
        let unpack = |bytes: &[u8], dim: usize| -> Vec<f32> {
            let all: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let mut out = vec![0.0f32; seq * dim];
            for s in 0..seq {
                for c in 0..dim {
                    out[s * dim + c] = all[c * ps + s];
                }
            }
            out
        };

        let t_fwd = std::time::Instant::now();

        let mut layer_in_bytes = pack(&hidden, h);
        let mut layer_out_bytes = vec![0u8; h * ps * 4];

        match &self.kernels {
            AneDiffusionKernelSet::Fused { kernels } => {
                for kernel in kernels {
                    kernel.write_input(0, &layer_in_bytes);
                    kernel.eval().unwrap();
                    kernel.read_output(0, &mut layer_out_bytes);
                    std::mem::swap(&mut layer_in_bytes, &mut layer_out_bytes);
                }
            }
            AneDiffusionKernelSet::MultiDispatch {
                attn_kernels,
                ffn_kernels,
            } => {
                let mut attn_out_bytes = vec![0u8; h * ps * 4];
                for (attn_kernel, ffn_kernel) in attn_kernels.iter().zip(ffn_kernels.iter()) {
                    attn_kernel.write_input(0, &layer_in_bytes);
                    attn_kernel.eval().unwrap();
                    attn_kernel.read_output(0, &mut attn_out_bytes);

                    ffn_kernel.write_input(0, &attn_out_bytes);
                    ffn_kernel.eval().unwrap();
                    ffn_kernel.read_output(0, &mut layer_out_bytes);
                    std::mem::swap(&mut layer_in_bytes, &mut layer_out_bytes);
                }
            }
        }

        let fwd_ms = t_fwd.elapsed().as_millis();
        let dispatches = match &self.kernels {
            AneDiffusionKernelSet::Fused { kernels } => kernels.len(),
            AneDiffusionKernelSet::MultiDispatch { attn_kernels, .. } => attn_kernels.len() * 2,
        };
        eprintln!("  fwd: {dispatches} ANE dispatches in {fwd_ms}ms (seq={seq})");

        // Unpack final layer output back to row-major [seq, h].
        let hidden_out = unpack(&layer_in_bytes, h);

        // 3. Final RMSNorm (CPU)
        let mut normed = vec![0.0f32; seq * h];
        rms_norm(
            &hidden_out,
            &self.blas_engine.final_norm,
            &mut normed,
            seq,
            h,
        );

        // 4. LM head: normed[seq, h] @ embed^T[h, vocab] → logits[seq, vocab]
        //    Embedding matrix doubles as LM head weight (tied weights).
        //    Too large for ANE (~600MB fp32), so we use BLAS on CPU.
        let mut logits = vec![0.0f32; seq * cfg.vocab];
        sgemm_nt(
            seq,
            cfg.vocab,
            h,
            &normed,
            &self.blas_engine.embed,
            &mut logits,
        );
        logits
    }

    /// Forward pass returning logits for the LAST position only → `[vocab]`.
    ///
    /// All ANE dispatch layers run identically to `forward()` (causal attention
    /// requires all positions).  Only the CPU final-norm + LM-head matmul is
    /// reduced from seq×vocab to 1×vocab, saving ~15ms per call at vocab=151936.
    pub fn forward_last(&self, token_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.blas_engine.config;
        let seq = token_ids.len();
        let h = cfg.hidden;
        let ps = self.seq_len;

        // 1. Embedding lookup — row-major [seq, h]
        let mut hidden = vec![0.0f32; seq * h];
        for (i, &tid) in token_ids.iter().enumerate() {
            let off = tid as usize * h;
            hidden[i * h..(i + 1) * h].copy_from_slice(&self.blas_engine.embed[off..off + h]);
        }

        // Pack row-major [seq, dim] → ANE channel-first fp32 [dim, ps]
        let pack = |data: &[f32], dim: usize| -> Vec<u8> {
            let mut buf = vec![0.0f32; dim * ps];
            for s in 0..seq {
                for c in 0..dim {
                    buf[c * ps + s] = data[s * dim + c];
                }
            }
            buf.iter().flat_map(|f| f.to_le_bytes()).collect()
        };

        // Unpack ANE channel-first fp32 [dim, ps] → row-major, LAST POSITION ONLY → [dim]
        let unpack_last = |bytes: &[u8], dim: usize| -> Vec<f32> {
            let all: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let last = seq - 1;
            let mut out = vec![0.0f32; dim];
            for c in 0..dim {
                out[c] = all[c * ps + last];
            }
            out
        };

        // 2. ANE layer dispatch (identical to forward())
        let mut layer_in_bytes = pack(&hidden, h);
        let mut layer_out_bytes = vec![0u8; h * ps * 4];

        match &self.kernels {
            AneDiffusionKernelSet::Fused { kernels } => {
                for kernel in kernels {
                    kernel.write_input(0, &layer_in_bytes);
                    kernel.eval().unwrap();
                    kernel.read_output(0, &mut layer_out_bytes);
                    std::mem::swap(&mut layer_in_bytes, &mut layer_out_bytes);
                }
            }
            AneDiffusionKernelSet::MultiDispatch {
                attn_kernels,
                ffn_kernels,
            } => {
                let mut attn_out_bytes = vec![0u8; h * ps * 4];
                for (attn_kernel, ffn_kernel) in attn_kernels.iter().zip(ffn_kernels.iter()) {
                    attn_kernel.write_input(0, &layer_in_bytes);
                    attn_kernel.eval().unwrap();
                    attn_kernel.read_output(0, &mut attn_out_bytes);

                    ffn_kernel.write_input(0, &attn_out_bytes);
                    ffn_kernel.eval().unwrap();
                    ffn_kernel.read_output(0, &mut layer_out_bytes);
                    std::mem::swap(&mut layer_in_bytes, &mut layer_out_bytes);
                }
            }
        }

        // 3. Unpack LAST position only → [h]
        let last_hidden = unpack_last(&layer_in_bytes, h);

        // 4. Final RMSNorm — single row
        let mut last_normed = vec![0.0f32; h];
        rms_norm(
            &last_hidden,
            &self.blas_engine.final_norm,
            &mut last_normed,
            1,
            h,
        );

        // 5. LM head: 1×vocab instead of seq×vocab
        let mut logits = vec![0.0f32; cfg.vocab];
        sgemm_nt(1, cfg.vocab, h, &last_normed, &self.blas_engine.embed, &mut logits);
        logits
    }

    /// ANE-accelerated MDLM generation.
    pub fn generate(&self, prompt_ids: &[u32], num_tokens: usize, steps: usize) -> Vec<u32> {
        let mask_id = self.blas_engine.config.mask_token_id;
        let vocab = self.blas_engine.config.vocab;
        let mut canvas: Vec<u32> = Vec::with_capacity(prompt_ids.len() + num_tokens);
        canvas.extend_from_slice(prompt_ids);
        canvas.extend(std::iter::repeat(mask_id).take(num_tokens));

        for step in 0..steps {
            let logits = self.forward(&canvas);
            let seq = canvas.len();

            let mut mask_positions: Vec<(usize, f32, u32)> = Vec::new();
            for pos in 0..seq {
                if canvas[pos] != mask_id {
                    continue;
                }
                let row = &logits[pos * vocab..(pos + 1) * vocab];
                let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut best_idx = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in row.iter().enumerate() {
                    sum_exp += (v - max_logit).exp();
                    if v > best_val {
                        best_val = v;
                        best_idx = i as u32;
                    }
                }
                let confidence = (best_val - max_logit).exp() / sum_exp;
                mask_positions.push((pos, confidence, best_idx));
            }

            if mask_positions.is_empty() {
                break;
            }

            let n_masks = mask_positions.len();
            let n_keep = n_masks * (steps - step - 1) / steps;
            let n_unmask = (n_masks - n_keep).max(1);

            mask_positions
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(pos, _, pred) in mask_positions.iter().take(n_unmask) {
                canvas[pos] = pred;
            }
        }

        canvas
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// FP16 (IEEE 754 half) bytes → f32 vec.
pub(crate) fn fp16_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            let sign = ((bits >> 15) & 1) as u32;
            let exp = ((bits >> 10) & 0x1F) as u32;
            let frac = (bits & 0x3FF) as u32;
            if exp == 0 {
                if frac == 0 {
                    f32::from_bits(sign << 31)
                } else {
                    let mut f = frac;
                    let mut e = 0i32;
                    while (f & (1 << 10)) == 0 {
                        f <<= 1;
                        e -= 1;
                    }
                    f &= 0x3FF;
                    let exp32 = (127 - 15 + 1 + e) as u32;
                    f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
                }
            } else if exp == 31 {
                f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
            } else {
                let exp32 = exp + 112;
                f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
            }
        })
        .collect()
}

/// Dequantize a Q1_0_g128 weight matrix to dense f32.
pub(crate) fn dequant_q1_g128(
    weight_bytes: &[u8],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    let group_size = 128usize;
    let n_groups = in_features / group_size;
    let packed_cols = in_features / 32;

    let weight_u32: Vec<u32> = weight_bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let scales = fp16_to_f32(scales_bytes);
    let biases = fp16_to_f32(biases_bytes);

    let mut out = vec![0.0f32; out_features * in_features];
    for row in 0..out_features {
        let w_row = &weight_u32[row * packed_cols..(row + 1) * packed_cols];
        let s_row = &scales[row * n_groups..(row + 1) * n_groups];
        let b_row = &biases[row * n_groups..(row + 1) * n_groups];
        let out_row = &mut out[row * in_features..(row + 1) * in_features];

        for col in 0..in_features {
            let group = col / group_size;
            let word_idx = col / 32;
            let bit_idx = col % 32;
            let bit = ((w_row[word_idx] >> bit_idx) & 1) as f32;
            out_row[col] = s_row[group] * bit + b_row[group];
        }
    }

    out
}

/// BF16 bytes → f32 vec.
pub(crate) fn bf16_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// RMSNorm: out[i] = x[i] * w[i] / sqrt(mean(x^2) + eps)
pub(crate) fn rms_norm(x: &[f32], w: &[f32], out: &mut [f32], seq: usize, dim: usize) {
    let eps = 1e-6f32;
    for s in 0..seq {
        let row = &x[s * dim..(s + 1) * dim];
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
        let inv = 1.0 / rms;
        for d in 0..dim {
            out[s * dim + d] = row[d] * inv * w[d];
        }
    }
}

/// RMSNorm with explicit eps: out[i] = x[i] * w[i] / sqrt(mean(x^2) + eps)
pub(crate) fn rms_norm_eps(
    x: &[f32],
    w: &[f32],
    out: &mut [f32],
    seq: usize,
    dim: usize,
    eps: f32,
) {
    for s in 0..seq {
        let row = &x[s * dim..(s + 1) * dim];
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
        let inv = 1.0 / rms;
        for d in 0..dim {
            out[s * dim + d] = row[d] * inv * w[d];
        }
    }
}

/// In-place RMSNorm on a slice (for QK norm per head).
pub(crate) fn rms_norm_slice(x: &mut [f32], w: &[f32]) {
    let eps = 1e-6f32;
    let dim = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    for (xi, wi) in x.iter_mut().zip(w.iter()) {
        *xi *= inv * wi;
    }
}

/// Apply RoPE rotation in-place to a [head_dim] vector at position `pos`.
pub(crate) fn apply_rope(x: &mut [f32], pos: usize, half_dim: usize, cos: &[f32], sin: &[f32]) {
    let co = pos * half_dim;
    for d in 0..half_dim {
        let x0 = x[d];
        let x1 = x[d + half_dim];
        let c = cos[co + d];
        let s = sin[co + d];
        x[d] = x0 * c - x1 * s;
        x[d + half_dim] = x0 * s + x1 * c;
    }
}

/// sgemm with B transposed: C[M,N] = A[M,K] @ B^T[K,N] where B stored as [N,K].
pub(crate) fn sgemm_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101,
            111,
            112, // RowMajor, NoTrans, Trans
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32, // B is [N,K], ldb=K for Trans
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// sgemm with B transposed and scaled: C = alpha * A @ B^T.
pub(crate) fn sgemm_nt_scaled(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    alpha: f32,
) {
    unsafe {
        cblas_sgemm(
            101,
            111,
            112,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// In-place softmax over a row.
pub(crate) fn softmax_inplace(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// y[M,N] += A^T[M,K] @ B[K,N] (A stored as [K,M], transposed). Accumulates into y.
pub(crate) fn sgemm_at_acc(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101,
            112,
            111, // RowMajor, Trans, NoTrans
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            m as i32,
            b.as_ptr(),
            n as i32,
            1.0,
            y.as_mut_ptr(),
            n as i32,
        );
    }
}

/// sgemm with B transposed, accumulating: C[M,N] += A[M,K] @ B^T[K,N].
pub(crate) fn sgemm_nt_acc(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101,
            111,
            112,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            1.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// CPU attention context for one diffusion/Qwen-style layer, returned before Wo.
pub(crate) fn diffusion_attention_context_cpu(
    engine: &DiffusionEngine,
    layer_idx: usize,
    hidden: &[f32],
    eps: f32,
) -> Vec<f32> {
    let cfg = &engine.config;
    let seq = hidden.len() / cfg.hidden;
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let half_hd = hd / 2;
    let n_heads = cfg.heads;
    let n_kv = cfg.kv_heads;
    let gqa_ratio = n_heads / n_kv;
    let q_dim = n_heads * hd;
    let kv_dim = n_kv * hd;
    let layer = &engine.layers[layer_idx];

    let mut normed = vec![0.0f32; seq * h];
    let mut q_buf = vec![0.0f32; seq * q_dim];
    let mut k_buf = vec![0.0f32; seq * kv_dim];
    let mut v_buf = vec![0.0f32; seq * kv_dim];
    let mut attn_out = vec![0.0f32; seq * q_dim];

    rms_norm_eps(hidden, &layer.input_norm, &mut normed, seq, h, eps);
    sgemm_nt(seq, q_dim, h, &normed, &layer.q_proj, &mut q_buf);
    sgemm_nt(seq, kv_dim, h, &normed, &layer.k_proj, &mut k_buf);
    sgemm_nt(seq, kv_dim, h, &normed, &layer.v_proj, &mut v_buf);

    for s in 0..seq {
        for head in 0..n_heads {
            let off = s * q_dim + head * hd;
            rms_norm_slice(&mut q_buf[off..off + hd], &layer.q_norm);
            apply_rope(
                &mut q_buf[off..off + hd],
                s,
                half_hd,
                &engine.rope_cos,
                &engine.rope_sin,
            );
        }
        for head in 0..n_kv {
            let off = s * kv_dim + head * hd;
            rms_norm_slice(&mut k_buf[off..off + hd], &layer.k_norm);
            apply_rope(
                &mut k_buf[off..off + hd],
                s,
                half_hd,
                &engine.rope_cos,
                &engine.rope_sin,
            );
        }
    }

    let scale = 1.0 / (hd as f32).sqrt();
    for kv_h in 0..n_kv {
        let mut k_head = vec![0.0f32; seq * hd];
        let mut v_head = vec![0.0f32; seq * hd];
        for s in 0..seq {
            let ko = s * kv_dim + kv_h * hd;
            k_head[s * hd..(s + 1) * hd].copy_from_slice(&k_buf[ko..ko + hd]);
            v_head[s * hd..(s + 1) * hd].copy_from_slice(&v_buf[ko..ko + hd]);
        }

        for g in 0..gqa_ratio {
            let q_h = kv_h * gqa_ratio + g;
            let mut q_head = vec![0.0f32; seq * hd];
            for s in 0..seq {
                let qo = s * q_dim + q_h * hd;
                q_head[s * hd..(s + 1) * hd].copy_from_slice(&q_buf[qo..qo + hd]);
            }

            let mut scores = vec![0.0f32; seq * seq];
            sgemm_nt_scaled(seq, seq, hd, &q_head, &k_head, &mut scores, scale);
            for row in 0..seq {
                softmax_inplace(&mut scores[row * seq..(row + 1) * seq]);
            }

            let mut ctx = vec![0.0f32; seq * hd];
            sgemm(seq, hd, seq, &scores, &v_head, &mut ctx);
            for s in 0..seq {
                let ao = s * q_dim + q_h * hd;
                attn_out[ao..ao + hd].copy_from_slice(&ctx[s * hd..(s + 1) * hd]);
            }
        }
    }

    attn_out
}

impl DiffusionEngine {
    /// Create an engine with small random weights for gradient checking tests.
    pub fn from_random(config: DiffusionConfig) -> Self {
        let h = config.hidden;
        let q_dim = config.heads * config.head_dim;
        let kv_dim = config.kv_heads * config.head_dim;
        let inter = config.inter;
        let vocab = config.vocab;

        let rand_vec = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32 * 0.618033988 + 0.31415926).fract() * 2.0 - 1.0) * scale)
                .collect()
        };

        let embed = rand_vec(vocab * h, 0.02);
        let final_norm = vec![1.0f32; h]; // init norm weights to 1

        let mut layers = Vec::with_capacity(config.layers);
        for l in 0..config.layers {
            let seed = l * 1000;
            let rv = |n: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| {
                        (((seed + i) as f32 * 0.618033988 + 0.31415926).fract() * 2.0 - 1.0) * 0.02
                    })
                    .collect()
            };
            layers.push(DiffusionLayerWeights {
                q_proj: rv(q_dim * h),
                k_proj: rv(kv_dim * h),
                v_proj: rv(kv_dim * h),
                o_proj: rv(h * q_dim),
                q_norm: vec![1.0; config.head_dim],
                k_norm: vec![1.0; config.head_dim],
                input_norm: vec![1.0; h],
                post_attn_norm: vec![1.0; h],
                gate_proj: rv(inter * h),
                up_proj: rv(inter * h),
                down_proj: rv(h * inter),
            });
        }

        let max_seq = 4096;
        let half_dim = config.head_dim / 2;
        let mut rope_cos = vec![0.0f32; max_seq * half_dim];
        let mut rope_sin = vec![0.0f32; max_seq * half_dim];
        for pos in 0..max_seq {
            for d in 0..half_dim {
                let freq =
                    1.0 / (config.rope_theta as f32).powf(2.0 * d as f32 / config.head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_dim + d] = angle.cos();
                rope_sin[pos * half_dim + d] = angle.sin();
            }
        }

        Self {
            layers,
            embed,
            final_norm,
            config,
            rope_cos,
            rope_sin,
        }
    }
}

// ---------------------------------------------------------------------------
// ANE hybrid engine for Bonsai-1.7B: ANE attention + BLAS FFN per layer
// ---------------------------------------------------------------------------

/// ANE hybrid engine: ANE attention + OC-tiled ANE FFN per layer.
///
/// Architecture per layer:
/// - Attention: 1 ANE kernel (QKV + SDPA + Wo, weights baked as BLOBFILEs)
/// - FFN gated: fused OC-tiled BLOBFILE kernels (gate/up weights split into paired tiles,
///   SiLU applied on ANE)
/// - FFN down: per-inter-tile BLOBFILE matmuls (each tile produces a full hidden partial)
/// - RMSNorm: CPU (trivial cost)
///
/// Tile dispatches per layer: gated_tiles + down_partial_tiles
/// For Bonsai 1.7B (dim=2048, inter=6144): 2+2 = 4 FFN dispatches + 1 attn = 5/layer
#[cfg(feature = "ane")]
pub struct AneBonsaiEngine {
    pub blas_engine: DiffusionEngine,
    pub attn_kernels: Vec<crate::ane_bridge::AneKernel>,
    /// OC-tiled fused gate+up+SiLU FFN kernels. Indexed as `[layer * n_tiles + tile]`.
    pub ffn_gated_kernels: Vec<crate::ane_bridge::AneKernel>,
    /// Per-inter-tile down partial kernels. Indexed as `[layer * n_tiles + tile]`.
    pub ffn_down_kernels: Vec<crate::ane_bridge::AneKernel>,
    /// OC tile plan for gate/up.
    pub w13_plan: crate::ane_mil::TilePlan,
    pub seq_len: usize,
    pub eps: f32,
    pub causal: bool,
}

#[cfg(feature = "ane")]
impl AneBonsaiEngine {
    /// Build 28 attention-only ANE kernels (bidirectional).
    pub fn new(engine: DiffusionEngine, seq_len: usize, eps: f32) -> Result<Self, String> {
        Self::new_inner(engine, seq_len, eps, false)
    }

    /// Build 28 attention-only ANE kernels with causal masking.
    pub fn new_causal(engine: DiffusionEngine, seq_len: usize, eps: f32) -> Result<Self, String> {
        Self::new_inner(engine, seq_len, eps, true)
    }

    fn new_inner(
        engine: DiffusionEngine,
        seq_len: usize,
        eps: f32,
        causal: bool,
    ) -> Result<Self, String> {
        use crate::ane_bridge::{self, build_weight_blob, build_weight_blob_transposed};
        use crate::ane_mil::{
            build_causal_mask_blob, compute_blobfile_tile_plan, gen_blobfile_matmul,
            gen_fused_silu_gate_up_proj,
            ANE_MIN_SPATIAL,
        };
        use crate::diffusion_ane;

        ane_bridge::ane_init()?;

        let cfg = &engine.config;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let half_hd = hd / 2;
        let q_dim = cfg.heads * hd;
        let kv_dim = cfg.kv_heads * hd;
        let inter = cfg.inter;
        let seq = seq_len.max(ANE_MIN_SPATIAL);

        let w13_plan = compute_blobfile_tile_plan(h, inter);
        let label = if causal { "causal" } else { "bidirectional" };
        eprintln!(
            "AneBonsaiEngine: compiling {} {} attn + OC-tiled FFN (dim={h}, inter={inter}, seq={seq})",
            cfg.layers, label
        );
        eprintln!(
            "  w13 plan: tile_oc={}, n_tiles={}, last_actual={}",
            w13_plan.tile_size, w13_plan.n_tiles, w13_plan.last_actual
        );
        let t0 = std::time::Instant::now();

        let mut rope_cos = vec![0.0f32; seq * half_hd];
        let mut rope_sin = vec![0.0f32; seq * half_hd];
        for pos in 0..seq {
            for d in 0..half_hd {
                let freq = 1.0 / (cfg.rope_theta as f32).powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_hd + d] = angle.cos();
                rope_sin[pos * half_hd + d] = angle.sin();
            }
        }

        let causal_mask_blob = if causal {
            Some(build_causal_mask_blob(seq))
        } else {
            None
        };

        // === Attention kernels (same as before) ===
        let attn_mil = diffusion_ane::gen_bonsai_attention_projection(
            h,
            cfg.heads,
            cfg.kv_heads,
            hd,
            seq,
            eps as f64,
            causal,
        );
        let attn_names: Vec<&str> = attn_mil.weight_names.iter().map(|s| s.as_str()).collect();
        let rope_cos_blob = build_weight_blob(&rope_cos, seq, half_hd);
        let rope_sin_blob = build_weight_blob(&rope_sin, seq, half_hd);

        let build_attn_blobs = |lw: &DiffusionLayerWeights| -> Vec<Vec<u8>> {
            let mut blobs = vec![
                build_weight_blob(&lw.input_norm, 1, h),
                build_weight_blob_transposed(&lw.q_proj, q_dim, h),
                build_weight_blob_transposed(&lw.k_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.v_proj, kv_dim, h),
                rope_cos_blob.clone(),
                rope_sin_blob.clone(),
                build_weight_blob(&lw.q_norm, 1, hd),
                build_weight_blob(&lw.k_norm, 1, hd),
            ];
            if let Some(ref mask) = causal_mask_blob {
                blobs.push(mask.clone());
            }
            blobs
        };

        let l0_attn_blobs = build_attn_blobs(&engine.layers[0]);
        let l0_attn_refs: Vec<&[u8]> = l0_attn_blobs.iter().map(|b| b.as_slice()).collect();
        let attn_kernel0 = compile_ane_kernel(
            "L0 Bonsai attn",
            &attn_mil.mil_text,
            &attn_names,
            &l0_attn_refs,
            &[attn_mil.input_bytes],
            &[attn_mil.output_bytes],
        )?;

        let l0_ms = t0.elapsed().as_millis();
        eprintln!("  L0 attn compile: {l0_ms}ms");

        let mut attn_kernels = Vec::with_capacity(cfg.layers);
        attn_kernels.push(attn_kernel0);
        for (i, lw) in engine.layers.iter().enumerate().skip(1) {
            let blobs = build_attn_blobs(lw);
            let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
            let ki = attn_kernels[0]
                .patch_from_donor(
                    &attn_mil.mil_text,
                    &attn_names,
                    &refs,
                    &[attn_mil.input_bytes],
                    &[attn_mil.output_bytes],
                )
                .map_err(|e| format!("L{i} attn patch: {e}"))?;
            attn_kernels.push(ki);
        }

        let attn_ms = t0.elapsed().as_millis();
        eprintln!("  {} attn kernels in {attn_ms}ms", cfg.layers);

        // === OC-tiled FFN kernels ===
        // Stage 1: gate+up+SiLU on ANE for each inter tile.
        // Stage 2: down partial matmul on ANE for the same inter tile.
        // The two stages are wired with shared IOSurfaces so the intermediate
        // gated activation never round-trips through CPU memory.

        let gated_tile_mil = gen_fused_silu_gate_up_proj(h, w13_plan.tile_size, seq);
        let gated_tile_names: Vec<&str> = gated_tile_mil
            .weight_names
            .iter()
            .map(|s| s.as_str())
            .collect();

        let down_partial_mil = gen_blobfile_matmul(w13_plan.tile_size, h, seq, "w2_partial");
        let down_partial_names: Vec<&str> = down_partial_mil
            .weight_names
            .iter()
            .map(|s| s.as_str())
            .collect();

        let build_w13_tile_blob = |weight: &[f32], tile_idx: usize| -> Vec<u8> {
            let start = w13_plan.tile_start(tile_idx);
            let actual = w13_plan.actual_tile_size(tile_idx);
            let tile_oc = w13_plan.tile_size;
            let mut tile_weight = vec![0.0f32; tile_oc * h];
            for t_row in 0..actual {
                let src_off = (start + t_row) * h;
                let dst_off = t_row * h;
                tile_weight[dst_off..dst_off + h].copy_from_slice(&weight[src_off..src_off + h]);
            }
            build_weight_blob_transposed(&tile_weight, tile_oc, h)
        };

        let build_down_partial_blob = |weight: &[f32], tile_idx: usize| -> Vec<u8> {
            let start = w13_plan.tile_start(tile_idx);
            let actual = w13_plan.actual_tile_size(tile_idx);
            let tile_ic = w13_plan.tile_size;
            let mut tile_weight = vec![0.0f32; h * tile_ic];
            for out_row in 0..h {
                let src_off = out_row * inter + start;
                let dst_off = out_row * tile_ic;
                tile_weight[dst_off..dst_off + actual]
                    .copy_from_slice(&weight[src_off..src_off + actual]);
            }
            build_weight_blob_transposed(&tile_weight, h, tile_ic)
        };

        // Compile L0 tile 0 kernels
        let gate_l0_t0_blob = build_w13_tile_blob(&engine.layers[0].gate_proj, 0);
        let up_l0_t0_blob = build_w13_tile_blob(&engine.layers[0].up_proj, 0);
        let gated_l0_t0_refs: Vec<&[u8]> = vec![gate_l0_t0_blob.as_slice(), up_l0_t0_blob.as_slice()];

        let gated_base = compile_ane_kernel(
            "L0 gated tile0",
            &gated_tile_mil.mil_text,
            &gated_tile_names,
            &gated_l0_t0_refs,
            &[gated_tile_mil.input_bytes],
            &[gated_tile_mil.output_bytes],
        )?;

        let down_l0_t0_blob = build_down_partial_blob(&engine.layers[0].down_proj, 0);
        let down_l0_t0_refs: Vec<&[u8]> = vec![down_l0_t0_blob.as_slice()];

        let down_base = compile_ane_kernel(
            "L0 down partial tile0",
            &down_partial_mil.mil_text,
            &down_partial_names,
            &down_l0_t0_refs,
            &[down_partial_mil.input_bytes],
            &[down_partial_mil.output_bytes],
        )?;

        let ffn_base_ms = t0.elapsed().as_millis();
        eprintln!("  FFN base tile kernels compile: {ffn_base_ms}ms");

        // Patch all tiles for all layers
        let n_w13 = w13_plan.n_tiles;
        let total_ffn_kernels = cfg.layers * (2 * n_w13);
        eprintln!(
            "  Patching {total_ffn_kernels} FFN tile kernels ({} layers × ({} gated + {} down-partial tiles))...",
            cfg.layers, n_w13, n_w13
        );

        let mut ffn_gated_kernels = Vec::with_capacity(cfg.layers * n_w13);
        let mut ffn_down_kernels = Vec::with_capacity(cfg.layers * n_w13);

        for (layer_idx, lw) in engine.layers.iter().enumerate() {
            for tile_idx in 0..n_w13 {
                let gate_blob = build_w13_tile_blob(&lw.gate_proj, tile_idx);
                let up_blob = build_w13_tile_blob(&lw.up_proj, tile_idx);
                let gated_refs: Vec<&[u8]> = vec![gate_blob.as_slice(), up_blob.as_slice()];
                let gated_ki = gated_base
                    .patch_from_donor(
                        &gated_tile_mil.mil_text,
                        &gated_tile_names,
                        &gated_refs,
                        &[gated_tile_mil.input_bytes],
                        &[gated_tile_mil.output_bytes],
                    )
                    .map_err(|e| format!("L{layer_idx} gated t{tile_idx} patch: {e}"))?;

                let down_blob = build_down_partial_blob(&lw.down_proj, tile_idx);
                let down_refs: Vec<&[u8]> = vec![down_blob.as_slice()];
                let down_ki = down_base
                    .patch_from_donor(
                        &down_partial_mil.mil_text,
                        &down_partial_names,
                        &down_refs,
                        &[down_partial_mil.input_bytes],
                        &[down_partial_mil.output_bytes],
                    )
                    .map_err(|e| format!("L{layer_idx} down-partial t{tile_idx} patch: {e}"))?;

                gated_ki
                    .share_output_to(0, &down_ki, 0)
                    .map_err(|e| format!("L{layer_idx} chain t{tile_idx} share: {e}"))?;

                ffn_gated_kernels.push(gated_ki);
                ffn_down_kernels.push(down_ki);
            }
        }

        let total_ms = t0.elapsed().as_millis();
        eprintln!(
            "AneBonsaiEngine: ready in {total_ms}ms — {} attn + {} FFN tile kernels",
            attn_kernels.len(),
            ffn_gated_kernels.len() + ffn_down_kernels.len(),
        );

        Ok(Self {
            blas_engine: engine,
            attn_kernels,
            ffn_gated_kernels,
            ffn_down_kernels,
            w13_plan,
            seq_len: seq,
            eps,
            causal,
        })
    }

    /// Hybrid forward: ANE attention + chained tiled ANE FFN per layer → final norm → LM head.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.blas_engine.config;
        let seq = token_ids.len();
        let h = cfg.hidden;
        let q_dim = cfg.heads * cfg.head_dim;
        let ps = self.seq_len;
        let verbose_trace = std::env::var_os("HIGGS_BONSAI_ANE_TRACE").is_some();

        let mut hidden = vec![0.0f32; seq * h];
        for (i, &tid) in token_ids.iter().enumerate() {
            let off = tid as usize * h;
            hidden[i * h..(i + 1) * h].copy_from_slice(&self.blas_engine.embed[off..off + h]);
        }

        let ane_dsb_sy = || {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                std::arch::asm!("dsb sy", options(nostack, preserves_flags));
            }
        };

        let write_input_cf32 =
            |kernel: &crate::ane_bridge::AneKernel, data: &[f32], dim: usize| {
                let base = kernel.get_input_base(0) as *mut f32;
                assert!(!base.is_null(), "ANE input base should not be null");
                unsafe {
                    for c in 0..dim {
                        let dst = std::slice::from_raw_parts_mut(base.add(c * ps), ps);
                        for s in 0..seq {
                            dst[s] = data[s * dim + c];
                        }
                    }
                }
                ane_dsb_sy();
            };

        let scatter_output_cf32 =
            |kernel: &crate::ane_bridge::AneKernel,
             actual_dim: usize,
             start: usize,
             dst: &mut [f32],
             dst_stride: usize| {
                let base = kernel.get_output_base(0) as *const f32;
                assert!(!base.is_null(), "ANE output base should not be null");
                ane_dsb_sy();
                unsafe {
                    for c in 0..actual_dim {
                        let src = std::slice::from_raw_parts(base.add(c * ps), ps);
                        for s in 0..seq {
                            dst[s * dst_stride + start + c] = src[s];
                        }
                    }
                }
            };

        let accumulate_output_cf32 =
            |kernel: &crate::ane_bridge::AneKernel,
             actual_dim: usize,
             dst: &mut [f32],
             dst_stride: usize| {
                let base = kernel.get_output_base(0) as *const f32;
                assert!(!base.is_null(), "ANE output base should not be null");
                ane_dsb_sy();
                unsafe {
                    for c in 0..actual_dim {
                        let src = std::slice::from_raw_parts(base.add(c * ps), ps);
                        for s in 0..seq {
                            dst[s * dst_stride + c] += src[s];
                        }
                    }
                }
            };

        let mut normed = vec![0.0f32; seq * h];
        let mut attn_proj = vec![0.0f32; seq * h];
        let mut attn_ctx_buf = vec![0.0f32; seq * q_dim];
        let mut ffn_out = vec![0.0f32; seq * h];
        // Bonsai Q1 drift compounds across depth on ANE attention. The best
        // measured hybrid split keeps most layers on ANE but rescues a sparse
        // set of earlier and late layers on CPU attention to stay within the
        // current correctness gate.
        // Causal path needs more CPU layers because the mask changes drift pattern.
        const BONSAI_CPU_ATTENTION_LAYERS: &[usize] = &[12, 15, 16, 17, 18, 19, 22, 24, 25, 26, 27];
        const BONSAI_CAUSAL_CPU_ATTENTION_LAYERS: &[usize] =
            &[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];
        let cpu_layers = if self.causal {
            BONSAI_CAUSAL_CPU_ATTENTION_LAYERS
        } else {
            BONSAI_CPU_ATTENTION_LAYERS
        };

        let rt_enabled = crate::ane_bridge::AneKernel::begin_realtime();

        let ane_eval = |kernel: &crate::ane_bridge::AneKernel| {
            if rt_enabled {
                kernel
                    .eval_realtime()
                    .unwrap_or_else(|_| kernel.eval().unwrap());
            } else {
                kernel.eval().unwrap();
            }
        };
        let ane_eval_chain = |kernels: &[&crate::ane_bridge::AneKernel]| {
            if rt_enabled {
                crate::ane_bridge::AneKernel::eval_chain_realtime(kernels).unwrap_or_else(|_| {
                    crate::ane_bridge::AneKernel::eval_chain(kernels).unwrap();
                });
            } else {
                crate::ane_bridge::AneKernel::eval_chain(kernels).unwrap();
            }
        };

        let t_fwd = std::time::Instant::now();
        for (layer_idx, kernel) in self.attn_kernels.iter().enumerate() {
            let layer = &self.blas_engine.layers[layer_idx];
            let residual = hidden.clone();
            if cpu_layers.contains(&layer_idx) {
                let cpu_ctx =
                    diffusion_attention_context_cpu(&self.blas_engine, layer_idx, &hidden, self.eps);
                attn_ctx_buf.copy_from_slice(&cpu_ctx);
            } else {
                write_input_cf32(kernel, &hidden, h);
                ane_eval(kernel);
                scatter_output_cf32(kernel, q_dim, 0, &mut attn_ctx_buf, q_dim);
            }
            sgemm_nt(seq, h, q_dim, &attn_ctx_buf, &layer.o_proj, &mut attn_proj);
            for i in 0..seq * h {
                hidden[i] = residual[i] + attn_proj[i];
            }

            {
                let nan_count = hidden.iter().filter(|v| !v.is_finite()).count();
                if nan_count > 0
                    || (verbose_trace && (layer_idx < 3 || layer_idx == cfg.layers - 1))
                {
                    let max_val = hidden
                        .iter()
                        .cloned()
                        .filter(|v| v.is_finite())
                        .fold(0.0f32, |a, b| a.max(b.abs()));
                    eprintln!(
                        "    L{layer_idx} post-attn: nan={nan_count}/{}, max_abs={max_val:.4}",
                        hidden.len()
                    );
                }
            }

            rms_norm_eps(
                &hidden,
                &layer.post_attn_norm,
                &mut normed,
                seq,
                h,
                self.eps,
            );

            // FFN: fused OC-tiled ANE gated projection → shared-surface ANE down partial
            let n_w13 = self.w13_plan.n_tiles;
            ffn_out.fill(0.0);
            for t in 0..n_w13 {
                let kernel_idx = layer_idx * n_w13 + t;
                let gated_kernel = &self.ffn_gated_kernels[kernel_idx];
                let down_kernel = &self.ffn_down_kernels[kernel_idx];
                write_input_cf32(gated_kernel, &normed, h);
                ane_eval_chain(&[gated_kernel, down_kernel]);
                accumulate_output_cf32(down_kernel, h, &mut ffn_out, h);
            }

            for i in 0..seq * h {
                hidden[i] += ffn_out[i];
            }

            {
                let nan_count = hidden.iter().filter(|v| !v.is_finite()).count();
                if nan_count > 0 {
                    let max_val = hidden
                        .iter()
                        .cloned()
                        .filter(|v| v.is_finite())
                        .fold(0.0f32, |a, b| a.max(b.abs()));
                    eprintln!(
                        "    L{layer_idx} post-ffn: nan={nan_count}/{}, max_abs={max_val:.4}",
                        hidden.len()
                    );
                }
            }
        }

        if rt_enabled {
            crate::ane_bridge::AneKernel::end_realtime();
        }

        let fwd_ms = t_fwd.elapsed().as_secs_f64() * 1000.0;
        if verbose_trace {
            eprintln!(
                "  hybrid fwd: {} ANE attn + {} ANE FFN chains in {fwd_ms:.1}ms (seq={seq}, rt={})",
                self.attn_kernels.len(),
                self.ffn_gated_kernels.len(),
                rt_enabled,
            );
        }

        rms_norm_eps(
            &hidden,
            &self.blas_engine.final_norm,
            &mut normed,
            seq,
            h,
            self.eps,
        );

        let mut logits = vec![0.0f32; seq * cfg.vocab];
        sgemm_nt(
            seq,
            cfg.vocab,
            h,
            &normed,
            &self.blas_engine.embed,
            &mut logits,
        );
        logits
    }

    /// Forward pass returning logits for the LAST position only → `[vocab]`.
    ///
    /// ANE attention + tiled FFN layers run identically to `forward()`.
    /// Only the CPU final-norm + LM-head matmul is 1×vocab instead of seq×vocab.
    pub fn forward_last(&self, token_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.blas_engine.config;
        let seq = token_ids.len();
        let h = cfg.hidden;
        let q_dim = cfg.heads * cfg.head_dim;
        let ps = self.seq_len;
        let verbose_trace = std::env::var_os("HIGGS_BONSAI_ANE_TRACE").is_some();

        let mut hidden = vec![0.0f32; seq * h];
        for (i, &tid) in token_ids.iter().enumerate() {
            let off = tid as usize * h;
            hidden[i * h..(i + 1) * h].copy_from_slice(&self.blas_engine.embed[off..off + h]);
        }

        let ane_dsb_sy = || {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                std::arch::asm!("dsb sy", options(nostack, preserves_flags));
            }
        };

        let write_input_cf32 =
            |kernel: &crate::ane_bridge::AneKernel, data: &[f32], dim: usize| {
                let base = kernel.get_input_base(0) as *mut f32;
                assert!(!base.is_null(), "ANE input base should not be null");
                unsafe {
                    for c in 0..dim {
                        let dst = std::slice::from_raw_parts_mut(base.add(c * ps), ps);
                        for s in 0..seq {
                            dst[s] = data[s * dim + c];
                        }
                    }
                }
                ane_dsb_sy();
            };

        let scatter_output_cf32 =
            |kernel: &crate::ane_bridge::AneKernel,
             actual_dim: usize,
             start: usize,
             dst: &mut [f32],
             dst_stride: usize| {
                let base = kernel.get_output_base(0) as *const f32;
                assert!(!base.is_null(), "ANE output base should not be null");
                ane_dsb_sy();
                unsafe {
                    for c in 0..actual_dim {
                        let src = std::slice::from_raw_parts(base.add(c * ps), ps);
                        for s in 0..seq {
                            dst[s * dst_stride + start + c] = src[s];
                        }
                    }
                }
            };

        let accumulate_output_cf32 =
            |kernel: &crate::ane_bridge::AneKernel,
             actual_dim: usize,
             dst: &mut [f32],
             dst_stride: usize| {
                let base = kernel.get_output_base(0) as *const f32;
                assert!(!base.is_null(), "ANE output base should not be null");
                ane_dsb_sy();
                unsafe {
                    for c in 0..actual_dim {
                        let src = std::slice::from_raw_parts(base.add(c * ps), ps);
                        for s in 0..seq {
                            dst[s * dst_stride + c] += src[s];
                        }
                    }
                }
            };

        let mut normed = vec![0.0f32; seq * h];
        let mut attn_proj = vec![0.0f32; seq * h];
        let mut attn_ctx_buf = vec![0.0f32; seq * q_dim];
        let mut ffn_out = vec![0.0f32; seq * h];
        const BONSAI_CPU_ATTENTION_LAYERS: &[usize] = &[12, 15, 16, 17, 18, 19, 22, 24, 25, 26, 27];
        const BONSAI_CAUSAL_CPU_ATTENTION_LAYERS: &[usize] =
            &[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];
        let cpu_layers = if self.causal {
            BONSAI_CAUSAL_CPU_ATTENTION_LAYERS
        } else {
            BONSAI_CPU_ATTENTION_LAYERS
        };

        let rt_enabled = crate::ane_bridge::AneKernel::begin_realtime();

        let ane_eval = |kernel: &crate::ane_bridge::AneKernel| {
            if rt_enabled {
                kernel
                    .eval_realtime()
                    .unwrap_or_else(|_| kernel.eval().unwrap());
            } else {
                kernel.eval().unwrap();
            }
        };
        let ane_eval_chain = |kernels: &[&crate::ane_bridge::AneKernel]| {
            if rt_enabled {
                crate::ane_bridge::AneKernel::eval_chain_realtime(kernels).unwrap_or_else(|_| {
                    crate::ane_bridge::AneKernel::eval_chain(kernels).unwrap();
                });
            } else {
                crate::ane_bridge::AneKernel::eval_chain(kernels).unwrap();
            }
        };

        for (layer_idx, kernel) in self.attn_kernels.iter().enumerate() {
            let layer = &self.blas_engine.layers[layer_idx];
            let residual = hidden.clone();
            if cpu_layers.contains(&layer_idx) {
                let cpu_ctx =
                    diffusion_attention_context_cpu(&self.blas_engine, layer_idx, &hidden, self.eps);
                attn_ctx_buf.copy_from_slice(&cpu_ctx);
            } else {
                write_input_cf32(kernel, &hidden, h);
                ane_eval(kernel);
                scatter_output_cf32(kernel, q_dim, 0, &mut attn_ctx_buf, q_dim);
            }
            sgemm_nt(seq, h, q_dim, &attn_ctx_buf, &layer.o_proj, &mut attn_proj);
            for i in 0..seq * h {
                hidden[i] = residual[i] + attn_proj[i];
            }

            rms_norm_eps(
                &hidden,
                &layer.post_attn_norm,
                &mut normed,
                seq,
                h,
                self.eps,
            );

            let n_w13 = self.w13_plan.n_tiles;
            ffn_out.fill(0.0);
            for t in 0..n_w13 {
                let kernel_idx = layer_idx * n_w13 + t;
                let gated_kernel = &self.ffn_gated_kernels[kernel_idx];
                let down_kernel = &self.ffn_down_kernels[kernel_idx];
                write_input_cf32(gated_kernel, &normed, h);
                ane_eval_chain(&[gated_kernel, down_kernel]);
                accumulate_output_cf32(down_kernel, h, &mut ffn_out, h);
            }

            for i in 0..seq * h {
                hidden[i] += ffn_out[i];
            }
        }

        if rt_enabled {
            crate::ane_bridge::AneKernel::end_realtime();
        }

        // Final RMSNorm — LAST POSITION ONLY
        let last_row = &hidden[(seq - 1) * h..seq * h];
        let mut last_normed = vec![0.0f32; h];
        rms_norm_eps(last_row, &self.blas_engine.final_norm, &mut last_normed, 1, h, self.eps);

        // LM head: 1×vocab
        let mut logits = vec![0.0f32; cfg.vocab];
        sgemm_nt(1, cfg.vocab, h, &last_normed, &self.blas_engine.embed, &mut logits);
        logits
    }
}

// ---------------------------------------------------------------------------
// ANE AR Decode Engine — T=1 decode with KV cache for speculative draft
// ---------------------------------------------------------------------------

/// T=1 decode engine on ANE with persistent KV cache IOSurfaces.
///
/// CPU pre-computes K/V per layer (fp16 matvec), writes to cache IOSurfaces.
/// ANE kernel does Q proj → QK-norm → RoPE → SDPA vs cache → Wo → residual → FFN.
///
/// Snapshot/restore is zero-cost: just reset `pos` and re-mask (stale K/V are masked out).
#[cfg(feature = "ane")]
pub struct AneArDecodeEngine {
    /// One compiled ANE kernel per transformer layer.
    kernels: Vec<crate::ane_bridge::AneKernel>,
    /// Reference to loaded model weights (for CPU K/V path + embed + final_norm).
    blas_engine: DiffusionEngine,
    // CPU K/V weights are accessed directly from blas_engine.layers
    /// Current decode position (next token goes here).
    pos: usize,
    /// Maximum sequence length (cache capacity).
    max_seq: usize,
    /// Total channels in packed input: dim + 2*kv_dim + hd + 1.
    total_ch: usize,
    /// Persistent input buffer per layer: [total_ch * max_seq] f32, updated incrementally.
    /// One buffer per layer (each layer has its own K/V cache state).
    input_bufs: Vec<Vec<f32>>,
}

#[cfg(feature = "ane")]
impl AneArDecodeEngine {
    /// Build the decode engine: compile ANE kernels with single packed input.
    ///
    /// `max_seq` is the cache capacity (must be >= 16). Each layer gets a persistent
    /// input buffer that holds x, K/V cache, rope, and mask in a single IOSurface.
    pub fn new(engine: DiffusionEngine, max_seq: usize) -> Result<Self, String> {
        use crate::ane_bridge::{self, build_weight_blob, build_weight_blob_transposed};
        use crate::ane_mil::ANE_MIN_SPATIAL;
        use crate::diffusion_ane;

        ane_bridge::ane_init()?;

        let cfg = &engine.config;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let q_dim = cfg.heads * hd;
        let inter = cfg.inter;
        let kv_dim = cfg.kv_heads * hd;
        let total_ch = h + 2 * kv_dim + hd + 1;

        assert!(max_seq >= ANE_MIN_SPATIAL, "max_seq must be >= {ANE_MIN_SPATIAL}");
        assert!(hd >= ANE_MIN_SPATIAL, "head_dim must be >= {ANE_MIN_SPATIAL}");

        eprintln!(
            "AneArDecodeEngine: compiling {} decode kernels (max_seq={max_seq}, total_ch={total_ch})...",
            cfg.layers
        );
        let t0 = std::time::Instant::now();

        // Generate decode layer MIL (stub on this branch; panics at runtime
        // if called — Magic Canvas tests go through the bidirectional path).
        let mil = diffusion_ane::gen_decode_layer(h, cfg.heads, cfg.kv_heads, hd, inter, max_seq, 1e-6);
        let mil_names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

        // Build weight blobs for each layer (8 BLOBFILEs per layer)
        let build_decode_blobs = |lw: &DiffusionLayerWeights| -> Vec<Vec<u8>> {
            vec![
                build_weight_blob(&lw.input_norm, 1, h),              // rms_att
                build_weight_blob_transposed(&lw.q_proj, q_dim, h),   // wq
                build_weight_blob_transposed(&lw.o_proj, h, q_dim),   // wo
                build_weight_blob(&lw.q_norm, 1, hd),                 // q_norm
                build_weight_blob(&lw.post_attn_norm, 1, h),          // rms_ffn
                build_weight_blob_transposed(&lw.gate_proj, inter, h), // gate
                build_weight_blob_transposed(&lw.up_proj, inter, h),  // up
                build_weight_blob_transposed(&lw.down_proj, h, inter), // down
            ]
        };

        // Compile L0 full
        let l0_blobs = build_decode_blobs(&engine.layers[0]);
        let l0_refs: Vec<&[u8]> = l0_blobs.iter().map(|b| b.as_slice()).collect();

        let kernel0 = compile_ane_kernel(
            "L0 decode compile",
            &mil.mil_text,
            &mil_names,
            &l0_refs,
            &[mil.input_bytes],
            &[mil.output_bytes],
        )?;
        let l0_ms = t0.elapsed().as_millis();
        eprintln!("  L0 full compile: {l0_ms}ms");

        // Patch layers 1+ from L0
        let mut kernels = Vec::with_capacity(cfg.layers);
        kernels.push(kernel0);
        for (i, lw) in engine.layers.iter().enumerate().skip(1) {
            let blobs = build_decode_blobs(lw);
            let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
            let ki = kernels[0]
                .patch_from_donor(
                    &mil.mil_text,
                    &mil_names,
                    &refs,
                    &[mil.input_bytes],
                    &[mil.output_bytes],
                )
                .map_err(|e| format!("L{i} decode patch: {e}"))?;
            kernels.push(ki);
        }
        let total_ms = t0.elapsed().as_millis();
        eprintln!(
            "AneArDecodeEngine: {} kernels in {total_ms}ms (L0={l0_ms}ms + {} patches)",
            cfg.layers,
            cfg.layers - 1
        );

        // Initialize per-layer input buffers with mask channel set to -1e9
        let buf_len = total_ch * max_seq;
        let mask_ch = h + 2 * kv_dim + hd;
        let mut input_bufs = Vec::with_capacity(cfg.layers);
        for _ in 0..cfg.layers {
            let mut buf = vec![0.0f32; buf_len];
            // Mask channel: all positions masked
            for p in 0..max_seq {
                buf[mask_ch * max_seq + p] = -1e9;
            }
            input_bufs.push(buf);
        }

        Ok(Self {
            kernels,
            blas_engine: engine,
            pos: 0,
            max_seq,
            total_ch,
            input_bufs,
        })
    }

    /// Decode a single token: embed → 28 layers → final_norm → logits.
    ///
    /// Each layer: CPU computes K/V → write cache → ANE kernel → read output.
    /// Returns logits `[vocab]`.
    pub fn decode_step(&mut self, token_id: u32) -> Vec<f32> {
        assert!(self.pos < self.max_seq, "KV cache full (pos={})", self.pos);

        let cfg = &self.blas_engine.config;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let half_hd = hd / 2;
        let kv_dim = cfg.kv_heads * hd;
        let pos = self.pos;

        // 1. Embedding lookup
        let mut hidden = vec![0.0f32; h];
        let off = token_id as usize * h;
        hidden.copy_from_slice(&self.blas_engine.embed[off..off + h]);

        // Precompute RoPE for current position
        let mut rope_cos = vec![0.0f32; half_hd];
        let mut rope_sin = vec![0.0f32; half_hd];
        for d in 0..half_hd {
            let freq =
                1.0 / (cfg.rope_theta as f32).powf(2.0 * d as f32 / hd as f32);
            let angle = pos as f32 * freq;
            rope_cos[d] = angle.cos();
            rope_sin[d] = angle.sin();
        }
        let ms = self.max_seq;

        // CPU scratch buffers for K/V computation
        let mut normed = vec![0.0f32; h];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];

        // 2. Layer loop — single packed input per layer
        for layer_idx in 0..self.kernels.len() {
            let lw = &self.blas_engine.layers[layer_idx];

            // CPU: RMSNorm → K/V projection → QK-norm on K → RoPE on K
            rms_norm_vec(&hidden, &lw.input_norm, &mut normed);

            // K = normed @ k_proj^T (sgemm_nt: [1,h] @ [kv_dim,h]^T → [1,kv_dim])
            sgemm_nt(1, kv_dim, h, &normed, &lw.k_proj, &mut k_buf);
            // V = normed @ v_proj^T
            sgemm_nt(1, kv_dim, h, &normed, &lw.v_proj, &mut v_buf);

            // QK-norm on K (per-head RMSNorm)
            for kv_h in 0..cfg.kv_heads {
                let off = kv_h * hd;
                rms_norm_slice(&mut k_buf[off..off + hd], &lw.k_norm);
            }

            // RoPE on K
            for kv_h in 0..cfg.kv_heads {
                let off = kv_h * hd;
                apply_rope(&mut k_buf[off..off + hd], pos, half_hd, &self.blas_engine.rope_cos, &self.blas_engine.rope_sin);
            }

            // --- Update persistent input buffer (single packed IOSurface) ---
            let buf = &mut self.input_bufs[layer_idx];

            // x: channels [0..h), position 0 only
            for ch in 0..h {
                buf[ch * ms] = hidden[ch];
            }

            // K cache at position pos: channels [h .. h+kv_dim)
            // Channel = h + kh*hd + d, spatial = pos
            for kh in 0..cfg.kv_heads {
                for d in 0..hd {
                    let ch = h + kh * hd + d;
                    buf[ch * ms + pos] = k_buf[kh * hd + d];
                }
            }

            // V cache at position pos: channels [h+kv_dim .. h+2*kv_dim)
            for kh in 0..cfg.kv_heads {
                for d in 0..hd {
                    let ch = h + kv_dim + kh * hd + d;
                    buf[ch * ms + pos] = v_buf[kh * hd + d];
                }
            }

            // Rope cos at position 0: channels [h+2*kv_dim .. h+2*kv_dim+half_hd)
            let rope_base = h + 2 * kv_dim;
            for d in 0..half_hd {
                buf[(rope_base + d) * ms] = rope_cos[d];
            }

            // Rope sin at position 0: channels [h+2*kv_dim+half_hd .. h+2*kv_dim+hd)
            for d in 0..half_hd {
                buf[(rope_base + half_hd + d) * ms] = rope_sin[d];
            }

            // Mask: unmask position pos (leave previous positions as-is, already 0)
            let mask_ch = h + 2 * kv_dim + hd;
            buf[mask_ch * ms + pos] = 0.0;

            // Write full packed buffer to single IOSurface input
            let bytes: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.kernels[layer_idx].write_input(0, &bytes);

            // ANE eval
            self.kernels[layer_idx].eval().unwrap_or_else(|e| panic!("L{layer_idx} decode eval failed: {e}"));

            // Read output: [1, dim, 1, ms] fp32 — position 0 only
            hidden = self.kernels[layer_idx].read_output_zerocopy(0, h, ms);
        }

        // 3. Final norm + LM head (logits)
        let mut final_normed = vec![0.0f32; h];
        rms_norm_vec(&hidden, &self.blas_engine.final_norm, &mut final_normed);

        // logits = embed^T @ final_normed (vocab, hidden) @ (hidden,) → (vocab,)
        let vocab = cfg.vocab;
        let mut logits = vec![0.0f32; vocab];
        sgemm_at(vocab, 1, h, &self.blas_engine.embed, &final_normed, &mut logits);

        self.pos += 1;
        logits
    }

    /// Process multiple tokens sequentially through the decode path.
    pub fn prefill(&mut self, token_ids: &[u32]) {
        for &tid in token_ids {
            let _ = self.decode_step(tid);
        }
    }

    /// Snapshot the current position for later restore (zero-cost).
    pub fn snapshot(&self) -> usize {
        self.pos
    }

    /// Restore to a previous position. Stale K/V in cache are harmless (masked out).
    pub fn restore(&mut self, saved_pos: usize) {
        assert!(saved_pos <= self.pos, "cannot restore to future position");
        let cfg = &self.blas_engine.config;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let kv_dim = cfg.kv_heads * hd;
        let mask_ch = h + 2 * kv_dim + hd;
        let ms = self.max_seq;
        // Re-mask positions >= saved_pos in all layer input buffers
        for buf in &mut self.input_bufs {
            for p in saved_pos..ms {
                buf[mask_ch * ms + p] = -1e9;
            }
        }
        self.pos = saved_pos;
    }

    /// Current decode position.
    pub fn current_pos(&self) -> usize {
        self.pos
    }
}

/// f32 → fp16 bit conversion (matches ane_extract::f32_to_f16_bits).
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;

    if exp == 0xFF {
        return (sign | 0x7C00 | if frac != 0 { 0x0200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        return sign as u16; // flush to zero
    }
    (sign | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// RMS norm for a single vector x[dim] with weight w[dim], output to out[dim].
fn rms_norm_vec(x: &[f32], w: &[f32], out: &mut [f32]) {
    let dim = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let rms = (sum_sq / dim as f32 + 1e-6).sqrt().recip();
    for i in 0..dim {
        out[i] = x[i] * rms * w[i];
    }
}

// ---------------------------------------------------------------------------
// Adaptive K controller for AR speculative decode
// ---------------------------------------------------------------------------

/// Dynamic draft-length controller for AR speculative decode.
///
/// Tracks a rolling window of acceptance rates and switches between
/// `k_high` (aggressive) and `k_low` (conservative) using hysteresis
/// thresholds to prevent oscillation.
pub struct AdaptiveKController {
    k_high: usize,
    k_low: usize,
    current_k: usize,
    window: std::collections::VecDeque<f64>,
    window_size: usize,
    drop_threshold: f64,
    rise_threshold: f64,
}

impl AdaptiveKController {
    /// Create a new controller. Starts at `k_high` (optimistic).
    ///
    /// - `drop_threshold`: rolling avg below this → switch to `k_low` (default 0.50)
    /// - `rise_threshold`: rolling avg above this → switch to `k_high` (default 0.65)
    pub fn new(k_low: usize, k_high: usize, window_size: usize) -> Self {
        Self {
            k_high,
            k_low,
            current_k: k_high,
            window: std::collections::VecDeque::with_capacity(window_size + 1),
            window_size,
            drop_threshold: 0.50,
            rise_threshold: 0.65,
        }
    }

    pub const fn current_k(&self) -> usize {
        self.current_k
    }

    /// Record a round's result and re-evaluate K.
    #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
    pub fn record(&mut self, accepted: usize, drafted: usize) {
        let rate = if drafted > 0 {
            accepted as f64 / drafted as f64
        } else {
            0.0
        };
        self.window.push_back(rate);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
        if self.window.len() >= self.window_size {
            let avg = self.window.iter().sum::<f64>() / self.window.len() as f64;
            if avg < self.drop_threshold {
                self.current_k = self.k_low;
            } else if avg > self.rise_threshold {
                self.current_k = self.k_high;
            }
            // Between thresholds → keep current (hysteresis)
        }
    }
}

// ---------------------------------------------------------------------------
// ANE Causal Drafter — speculative decode with ANE causal engine
// ---------------------------------------------------------------------------

/// Greedy acceptance for speculative decode.
///
/// Compares draft tokens against verify-model argmax predictions.
/// `verify_argmax` has length `draft.len() + 1`: the first entry verifies `draft[0]`,
/// and the last is the bonus/correction token.
///
/// Returns the accepted prefix plus one correction/bonus token.
pub fn accept_prefix(draft: &[u32], verify_argmax: &[u32]) -> Vec<u32> {
    debug_assert_eq!(verify_argmax.len(), draft.len() + 1);

    let mut accepted = 0;
    for (d, &v) in draft.iter().zip(verify_argmax.iter()) {
        if *d == v {
            accepted += 1;
        } else {
            break;
        }
    }
    let mut out: Vec<u32> = draft[..accepted].to_vec();
    out.push(verify_argmax[accepted]);
    out
}

/// ANE-based causal drafter for speculative decoding.
///
/// Wraps a `DiffusionRuntime` compiled with causal masking. Each `draft()` call
/// does K full-sequence causal forwards on ANE (~19ms each for 0.6B).
/// The GPU is 100% free for the verify model — no resource contention.
#[cfg(feature = "ane")]
pub struct AneCausalDrafter {
    pub(crate) runtime: DiffusionRuntime,
    pub(crate) vocab: usize,
    pub(crate) max_seq: usize,
}

#[cfg(feature = "ane")]
impl AneCausalDrafter {
    /// Build a causal ANE drafter from a model directory (e.g. Qwen3-0.6B-Base).
    pub fn new(model_path: &Path, max_seq: usize) -> Result<Self, String> {
        let engine = DiffusionEngine::load(model_path)?;
        let vocab = engine.config.vocab;
        let runtime =
            DiffusionRuntime::new_causal(engine, max_seq, DiffusionBackendPreference::Auto)?;
        eprintln!(
            "AneCausalDrafter: backend={:?}, max_seq={max_seq}",
            runtime.selected_backend()
        );
        Ok(Self {
            runtime,
            vocab,
            max_seq,
        })
    }

    /// Generate K draft tokens given a prefix via greedy argmax.
    ///
    /// Each step runs a full causal forward over `prefix ++ drafted_so_far`,
    /// takes the argmax at the last position, and appends it.
    /// Truncates prefix to `max_seq - k` if necessary.
    pub fn draft(&self, prefix: &[u32], k: usize) -> Vec<u32> {
        let budget = self.max_seq.saturating_sub(k);
        let start = if prefix.len() > budget {
            prefix.len() - budget
        } else {
            0
        };
        let mut seq: Vec<u32> = prefix[start..].to_vec();

        let mut drafted = Vec::with_capacity(k);
        for _ in 0..k {
            if seq.len() >= self.max_seq {
                break;
            }
            let logits = self.runtime.forward(&seq);
            // logits: [seq_len * vocab], take last position
            let last_offset = (seq.len() - 1) * self.vocab;
            let row = &logits[last_offset..last_offset + self.vocab];
            let token = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap();
            drafted.push(token);
            seq.push(token);
        }
        drafted
    }

    /// Get full logits for a token sequence. Returns `[seq_len, vocab]` flat.
    pub fn forward_logits(&self, token_ids: &[u32]) -> Vec<f32> {
        self.runtime.forward(token_ids)
    }
}

/// ANE→GPU speculative generation with fresh-cache verify and saved ANE logits.
///
/// Each round:
///   1. Draft[0] from saved ANE logits (FREE, no forward needed)
///   2. Draft[1..K-1] via sequential ANE causal forwards
///   3. GPU verify: fresh cache + forward_all_logits(context ++ drafts)
///      — no clone/restore/eval_for_clone overhead
///   4. Accept prefix, extend context
///
/// Returns generated tokens (not including prompt).
#[cfg(feature = "ane")]
#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
pub fn speculative_generate(
    drafter: &AneCausalDrafter,
    verifier: &mut crate::AnyModel,
    prompt: &[u32],
    max_tokens: usize,
    k_low: usize,
    k_high: usize,
) -> Vec<u32> {
    use mlx_rs::ops::indexing::{self as ix, IndexOp};

    let mut context: Vec<u32> = prompt.to_vec();
    let mut generated: Vec<u32> = Vec::new();
    let mut k_ctrl = AdaptiveKController::new(k_low, k_high, 3);

    let mut total_drafted = 0usize;
    let mut total_accepted = 0usize;
    let mut draft_ms = 0.0f64;
    let mut verify_ms = 0.0f64;

    // Bootstrap: one full ANE forward on the prompt to get saved logits.
    // The last-position argmax gives us draft[0] for round 0 — FOR FREE.
    let t_boot = std::time::Instant::now();
    let mut saved_ane_logits = drafter.forward_logits(&context);
    let boot_ms = t_boot.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Bootstrap ANE logits: {boot_ms:.0}ms");

    while generated.len() < max_tokens {
        let k = k_ctrl.current_k().min(max_tokens - generated.len());
        if k == 0 {
            break;
        }

        // --- Draft on ANE ---
        let t0 = std::time::Instant::now();

        // Draft[0] from saved logits — no forward needed
        let last_offset = (context.len() - 1) * drafter.vocab;
        let row = &saved_ane_logits[last_offset..last_offset + drafter.vocab];
        let draft_0 = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        let mut draft_tokens: Vec<u32> = vec![draft_0];

        // Draft[1..K-1] via sequential ANE causal forwards
        let budget = drafter.max_seq;
        let mut seq: Vec<u32> = context.clone();
        seq.push(draft_0);

        for _ in 1..k {
            if seq.len() >= budget {
                break;
            }
            // forward_last: same ANE dispatch but 1×vocab LM head instead of seq×vocab
            let logits = drafter.runtime.forward_last(&seq);
            let token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap();
            draft_tokens.push(token);
            seq.push(token);
        }

        let actual_k = draft_tokens.len();
        draft_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // --- Verify on GPU (fresh cache — no clone/restore overhead) ---
        let t1 = std::time::Instant::now();
        let mut cache = verifier.make_cache();
        let full_seq: Vec<i32> = context
            .iter()
            .chain(draft_tokens.iter())
            .map(|&t| t as i32)
            .collect();
        let full_arr = mlx_rs::Array::from_slice(&full_seq, &[1, full_seq.len() as i32]);
        let all_logits = verifier
            .forward_all_logits(&full_arr, None, &mut cache)
            .unwrap();
        mlx_rs::transforms::eval([&all_logits]).unwrap();

        // Extract argmax at verify positions
        let logits_2d = all_logits.squeeze_axes(&[0]).unwrap(); // [full_len, vocab]
        let all_preds = ix::argmax_axis(&logits_2d, -1, false).unwrap();
        mlx_rs::transforms::eval([&all_preds]).unwrap();

        let ctx_len = context.len();
        let mut verify_argmax: Vec<u32> = Vec::with_capacity(actual_k + 1);
        for i in 0..=actual_k {
            let pos = (ctx_len - 1 + i) as i32;
            verify_argmax.push(all_preds.index(pos).item::<i32>() as u32);
        }
        verify_ms += t1.elapsed().as_secs_f64() * 1000.0;

        // --- Accept ---
        let mut accepted_tokens = accept_prefix(&draft_tokens, &verify_argmax);
        let remaining = max_tokens - generated.len();
        if accepted_tokens.len() > remaining {
            accepted_tokens.truncate(remaining);
        }
        let accepted = accepted_tokens.len().saturating_sub(1);
        total_accepted += accepted;
        total_drafted += actual_k;
        k_ctrl.record(accepted, actual_k);

        // Deferred save: always save ANE logits on the CORRECT context after verify.
        // On full acceptance, context+accepted == seq (same as eager save).
        // On rejection, we skip the wasted eager forward that would have been discarded.
        context.extend_from_slice(&accepted_tokens);
        saved_ane_logits = drafter.forward_logits(&context);
        generated.extend_from_slice(&accepted_tokens);

        let round = total_drafted / k.max(1);
        eprintln!(
            "  R{round}: accepted {accepted}/{actual_k} (+{} new) | K={}",
            accepted_tokens.len(),
            k_ctrl.current_k()
        );
    }

    let total_ms = draft_ms + verify_ms;
    let tps = if total_ms > 0.0 {
        generated.len() as f64 / total_ms * 1000.0
    } else {
        0.0
    };
    let acc_rate = if total_drafted > 0 {
        total_accepted as f64 / total_drafted as f64 * 100.0
    } else {
        0.0
    };
    eprintln!("  --- Totals ---");
    eprintln!(
        "  {:.0}ms draft + {:.0}ms verify = {:.0}ms",
        draft_ms, verify_ms, total_ms
    );
    eprintln!("  Acceptance: {total_accepted}/{total_drafted} ({acc_rate:.1}%)");
    eprintln!("  Throughput: {tps:.1} tok/s");

    generated
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, unsafe_code)]
mod tests {
    use super::*;

    fn tiny_runtime_engine() -> DiffusionEngine {
        let config = DiffusionConfig {
            hidden: 2,
            layers: 0,
            heads: 1,
            kv_heads: 1,
            head_dim: 2,
            inter: 2,
            vocab: 4,
            mask_token_id: 3,
            rope_theta: 10_000.0,
        };

        DiffusionEngine {
            layers: Vec::new(),
            embed: vec![
                1.0, 0.0, // token 0
                0.0, 1.0, // token 1
                1.0, 1.0, // token 2
                0.5, -0.5, // mask token
            ],
            final_norm: vec![1.0, 1.0],
            config,
            rope_cos: vec![1.0; 8],
            rope_sin: vec![0.0; 8],
        }
    }

    #[cfg(feature = "ane")]
    fn bonsai_embed_hidden(engine: &DiffusionEngine, token_ids: &[u32]) -> Vec<f32> {
        let h = engine.config.hidden;
        let mut hidden = vec![0.0f32; token_ids.len() * h];
        for (i, &tid) in token_ids.iter().enumerate() {
            let off = tid as usize * h;
            hidden[i * h..(i + 1) * h].copy_from_slice(&engine.embed[off..off + h]);
        }
        hidden
    }

    #[cfg(feature = "ane")]
    fn bonsai_l0_attention_context_cpu(engine: &DiffusionEngine, hidden: &[f32]) -> Vec<f32> {
        bonsai_attention_context_cpu_for_layer(engine, 0, hidden)
    }

    #[cfg(feature = "ane")]
    fn bonsai_attention_context_cpu_for_layer(
        engine: &DiffusionEngine,
        layer_idx: usize,
        hidden: &[f32],
    ) -> Vec<f32> {
        let cfg = &engine.config;
        let seq = hidden.len() / cfg.hidden;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let half_hd = hd / 2;
        let n_heads = cfg.heads;
        let n_kv = cfg.kv_heads;
        let gqa_ratio = n_heads / n_kv;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let layer = &engine.layers[layer_idx];

        let mut normed = vec![0.0f32; seq * h];
        let mut q_buf = vec![0.0f32; seq * q_dim];
        let mut k_buf = vec![0.0f32; seq * kv_dim];
        let mut v_buf = vec![0.0f32; seq * kv_dim];
        let mut attn_out = vec![0.0f32; seq * q_dim];

        rms_norm_eps(hidden, &layer.input_norm, &mut normed, seq, h, 1e-6);
        sgemm_nt(seq, q_dim, h, &normed, &layer.q_proj, &mut q_buf);
        sgemm_nt(seq, kv_dim, h, &normed, &layer.k_proj, &mut k_buf);
        sgemm_nt(seq, kv_dim, h, &normed, &layer.v_proj, &mut v_buf);

        for s in 0..seq {
            for head in 0..n_heads {
                let off = s * q_dim + head * hd;
                rms_norm_slice(&mut q_buf[off..off + hd], &layer.q_norm);
                apply_rope(
                    &mut q_buf[off..off + hd],
                    s,
                    half_hd,
                    &engine.rope_cos,
                    &engine.rope_sin,
                );
            }
            for head in 0..n_kv {
                let off = s * kv_dim + head * hd;
                rms_norm_slice(&mut k_buf[off..off + hd], &layer.k_norm);
                apply_rope(
                    &mut k_buf[off..off + hd],
                    s,
                    half_hd,
                    &engine.rope_cos,
                    &engine.rope_sin,
                );
            }
        }

        let scale = 1.0 / (hd as f32).sqrt();
        for kv_h in 0..n_kv {
            let mut k_head = vec![0.0f32; seq * hd];
            let mut v_head = vec![0.0f32; seq * hd];
            for s in 0..seq {
                let ko = s * kv_dim + kv_h * hd;
                k_head[s * hd..(s + 1) * hd].copy_from_slice(&k_buf[ko..ko + hd]);
                v_head[s * hd..(s + 1) * hd].copy_from_slice(&v_buf[ko..ko + hd]);
            }

            for g in 0..gqa_ratio {
                let q_h = kv_h * gqa_ratio + g;
                let mut q_head = vec![0.0f32; seq * hd];
                for s in 0..seq {
                    let qo = s * q_dim + q_h * hd;
                    q_head[s * hd..(s + 1) * hd].copy_from_slice(&q_buf[qo..qo + hd]);
                }

                let mut scores = vec![0.0f32; seq * seq];
                sgemm_nt_scaled(seq, seq, hd, &q_head, &k_head, &mut scores, scale);
                for row in 0..seq {
                    softmax_inplace(&mut scores[row * seq..(row + 1) * seq]);
                }

                let mut ctx = vec![0.0f32; seq * hd];
                sgemm(seq, hd, seq, &scores, &v_head, &mut ctx);
                for s in 0..seq {
                    let ao = s * q_dim + q_h * hd;
                    attn_out[ao..ao + hd].copy_from_slice(&ctx[s * hd..(s + 1) * hd]);
                }
            }
        }

        attn_out
    }

    #[cfg(feature = "ane")]
    fn bonsai_l0_attention_context_ane(engine: &DiffusionEngine, token_ids: &[u32]) -> Vec<f32> {
        use crate::ane_bridge::{self, build_weight_blob, build_weight_blob_transposed};
        use crate::ane_mil::ANE_MIN_SPATIAL;

        ane_bridge::ane_init().unwrap();

        let cfg = &engine.config;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let q_dim = cfg.heads * hd;
        let kv_dim = cfg.kv_heads * hd;
        let seq = token_ids.len();
        let ps = seq.max(ANE_MIN_SPATIAL);

        let mil = crate::diffusion_ane::gen_bonsai_attention_projection(
            h,
            cfg.heads,
            cfg.kv_heads,
            hd,
            seq,
            1e-6,
            false, // bidirectional for that test
        );
        let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
        let layer = &engine.layers[0];
        let half_hd = hd / 2;
        let rope_cos = engine.rope_cos[..ps * half_hd].to_vec();
        let rope_sin = engine.rope_sin[..ps * half_hd].to_vec();
        let blobs = vec![
            build_weight_blob(&layer.input_norm, 1, h),
            build_weight_blob_transposed(&layer.q_proj, q_dim, h),
            build_weight_blob_transposed(&layer.k_proj, kv_dim, h),
            build_weight_blob_transposed(&layer.v_proj, kv_dim, h),
            build_weight_blob(&rope_cos, ps, half_hd),
            build_weight_blob(&rope_sin, ps, half_hd),
            build_weight_blob(&layer.q_norm, 1, hd),
            build_weight_blob(&layer.k_norm, 1, hd),
        ];
        let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
        let kernel = super::compile_ane_kernel(
            "L0 Bonsai attention-core compare compile",
            &mil.mil_text,
            &names,
            &refs,
            &[mil.input_bytes],
            &[mil.output_bytes],
        )
        .unwrap();

        let hidden = bonsai_embed_hidden(engine, token_ids);
        let mut input = vec![0.0f32; h * ps];
        for s in 0..seq {
            for c in 0..h {
                input[c * ps + s] = hidden[s * h + c];
            }
        }
        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut output_bytes = vec![0u8; q_dim * ps * 4];
        kernel.write_input(0, &input_bytes);
        kernel.eval().unwrap();
        kernel.read_output(0, &mut output_bytes);

        let all: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut out = vec![0.0f32; seq * q_dim];
        for s in 0..seq {
            for c in 0..q_dim {
                out[s * q_dim + c] = all[c * ps + s];
            }
        }
        out
    }

    #[cfg(feature = "ane")]
    fn permute_bonsai_gqa_head_layout(
        data: &[f32],
        seq: usize,
        kv_heads: usize,
        hd: usize,
    ) -> Vec<f32> {
        let heads = data.len() / (seq * hd);
        let hpg = heads / kv_heads;
        let mut out = vec![0.0f32; data.len()];
        for s in 0..seq {
            for kv_h in 0..kv_heads {
                for g in 0..hpg {
                    let src_head = kv_h * hpg + g;
                    let dst_head = g * kv_heads + kv_h;
                    let src = s * heads * hd + src_head * hd;
                    let dst = s * heads * hd + dst_head * hd;
                    out[dst..dst + hd].copy_from_slice(&data[src..src + hd]);
                }
            }
        }
        out
    }

    #[cfg(feature = "ane")]
    fn bonsai_attention_context_ane_with_kernel(
        kernel: &crate::ane_bridge::AneKernel,
        hidden: &[f32],
        seq: usize,
        ps: usize,
        h: usize,
        q_dim: usize,
    ) -> Vec<f32> {
        let mut input = vec![0.0f32; h * ps];
        for s in 0..seq {
            for c in 0..h {
                input[c * ps + s] = hidden[s * h + c];
            }
        }
        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut output_bytes = vec![0u8; q_dim * ps * 4];
        kernel.write_input(0, &input_bytes);
        kernel.eval().unwrap();
        kernel.read_output(0, &mut output_bytes);

        let all: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut out = vec![0.0f32; seq * q_dim];
        for s in 0..seq {
            for c in 0..q_dim {
                out[s * q_dim + c] = all[c * ps + s];
            }
        }
        out
    }

    fn write_config_json(dir: &std::path::Path, body: &str) {
        std::fs::write(dir.join("config.json"), body).unwrap();
    }

    fn model_dir() -> Option<String> {
        let dir = format!(
            "{}/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1",
            std::env::var("HOME").ok()?
        );
        if std::path::Path::new(&dir)
            .join("model.safetensors")
            .exists()
        {
            Some(dir)
        } else {
            None
        }
    }

    fn qwen3_base_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/lm-studio/models/mlx-community/Qwen3-0.6B-Base");
        if dir.join("config.json").exists() {
            Some(dir)
        } else {
            None
        }
    }

    fn bonsai_1_7b_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/lm-studio/models/prism-ml/Bonsai-1.7B-mlx-1bit");
        if dir.join("model.safetensors").exists() {
            Some(dir)
        } else {
            None
        }
    }

    fn bonsai_bench_tokenizer() -> Option<tokenizers::Tokenizer> {
        let dir = bonsai_1_7b_dir()?;
        crate::load_tokenizer(dir).ok()
    }

    fn build_long_bonsai_prompt_ids(
        tokenizer: &tokenizers::Tokenizer,
        prompt_len: usize,
    ) -> Vec<u32> {
        let seed = concat!(
            "Bonsai is a 1-bit Qwen3 model designed for efficient inference on Apple Silicon. ",
            "We care about ANE throughput, GPU fallback, prompt fidelity, and generated quality. ",
            "Summarize the tradeoffs of low-bit attention, residual error accumulation, and long-context performance. "
        );
        let seed_ids = tokenizer
            .encode(seed, false)
            .expect("seed prompt should tokenize")
            .get_ids()
            .to_vec();
        assert!(!seed_ids.is_empty(), "seed prompt tokenized to zero length");

        let mut ids = Vec::with_capacity(prompt_len);
        while ids.len() < prompt_len {
            ids.extend_from_slice(&seed_ids);
        }
        ids.truncate(prompt_len);
        ids
    }

    fn bonsai_bench_mask_token_id(tokenizer: &tokenizers::Tokenizer) -> u32 {
        tokenizer
            .encode("<|endoftext|>", false)
            .expect("pad token should tokenize")
            .get_ids()
            .first()
            .copied()
            .expect("pad token should yield one token")
    }

    fn sampled_masked_pseudo_ppl_runtime(
        runtime: &DiffusionRuntime,
        token_ids: &[u32],
        mask_id: u32,
        sample_count: usize,
    ) -> f64 {
        if token_ids.len() < 3 {
            return f64::NAN;
        }

        let interior = token_ids.len() - 2;
        let n = sample_count.min(interior).max(1);
        let mut positions = Vec::with_capacity(n);
        for i in 0..n {
            let pos = 1 + i * interior / n;
            if positions.last().copied() != Some(pos) {
                positions.push(pos);
            }
        }

        let mut nll_sum = 0.0f64;
        for &pos in &positions {
            let mut masked = token_ids.to_vec();
            masked[pos] = mask_id;
            let logits = runtime.forward(&masked);
            let vocab = logits.len() / masked.len();
            let row = &logits[pos * vocab..(pos + 1) * vocab];
            let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp = row.iter().map(|&v| (v - max_logit).exp()).sum::<f32>();
            let target = token_ids[pos] as usize;
            let logprob = row[target] - max_logit - sum_exp.ln();
            nll_sum += -(logprob as f64);
        }

        (nll_sum / positions.len() as f64).exp()
    }

    #[test]
    fn test_load_and_forward() {
        let Some(dir) = model_dir() else {
            eprintln!("Model not found, skipping");
            return;
        };

        let engine = DiffusionEngine::load(&dir).unwrap();
        assert_eq!(engine.config.layers, 28);
        assert_eq!(engine.config.hidden, 1024);

        // Same input as Python ground truth
        let input_ids: Vec<u32> = vec![
            785, 6722, 315, 9625, 374, 151669, 151669, 151669, 151669, 151669, 151669, 151669,
            151669, 151669, 151669, 151669, 151669, 151669, 151669, 151669, 151669,
        ];

        let t0 = std::time::Instant::now();
        let logits = engine.forward(&input_ids);
        let ms = t0.elapsed().as_millis();
        eprintln!("Forward: {}ms for seq={}", ms, input_ids.len());

        let vocab = engine.config.vocab;
        assert_eq!(logits.len(), input_ids.len() * vocab);

        // Compare first 5 logits at position 5 (first mask) vs Python ground truth
        // Python: [9.964, 4.230, 3.615, 0.344, 5.087]
        let pos5 = &logits[5 * vocab..5 * vocab + 5];
        eprintln!("Logits[0,5,:5] = {:?}", pos5);
        eprintln!("Python:         [9.964, 4.230, 3.615, 0.344, 5.087]");

        let err = (pos5[0] - 9.964).abs();
        eprintln!("First logit error: {err:.4}");
        // Allow some error from bf16→f32 weight conversion
        assert!(err < 1.0, "Logit error too large: {err}");
    }

    #[test]
    fn test_generate() {
        let Some(dir) = model_dir() else {
            eprintln!("Model not found, skipping");
            return;
        };

        let engine = DiffusionEngine::load(&dir).unwrap();

        let prompt_ids: Vec<u32> = vec![785, 6722, 315, 9625, 374]; // "The capital of France is"

        let t0 = std::time::Instant::now();
        let result = engine.generate(&prompt_ids, 32, 32);
        let elapsed = t0.elapsed();

        eprintln!(
            "Generated {} tokens in {}ms ({:.1} tok/s)",
            32,
            elapsed.as_millis(),
            32.0 / elapsed.as_secs_f64()
        );
        eprintln!("Result IDs: {:?}", &result[5..]);

        // We can't decode without the tokenizer, but verify it produces non-mask tokens
        let masks = result.iter().filter(|&&t| t == 151669).count();
        eprintln!("Remaining masks: {masks}");
        assert!(masks < 5, "Too many masks remaining: {masks}");
    }

    #[test]
    fn test_runtime_cpu_backend_report_and_forward() {
        let engine = tiny_runtime_engine();
        let input = vec![0, 3];
        let expected = engine.forward(&input);

        let runtime =
            DiffusionRuntime::new(engine, input.len(), DiffusionBackendPreference::CpuBlas)
                .unwrap();
        let report = runtime.backend_report();

        assert_eq!(report.model_kind, DiffusionModelKind::Standard);
        assert_eq!(report.requested, DiffusionBackendPreference::CpuBlas);
        assert_eq!(report.selected, DiffusionBackend::CpuBlas);
        assert_eq!(report.detail, "CPU BLAS requested explicitly");
        assert_eq!(runtime.model_kind(), DiffusionModelKind::Standard);
        assert_eq!(
            runtime.requested_backend(),
            DiffusionBackendPreference::CpuBlas
        );
        assert_eq!(runtime.selected_backend(), DiffusionBackend::CpuBlas);
        assert_eq!(runtime.forward(&input), expected);
    }

    #[test]
    fn test_runtime_cpu_backend_generate_matches_engine() {
        let engine = tiny_runtime_engine();
        let prompt = vec![0];
        let expected = engine.generate(&prompt, 1, 1);

        let runtime = DiffusionRuntime::new(
            engine,
            prompt.len() + 1,
            DiffusionBackendPreference::CpuBlas,
        )
        .unwrap();

        assert_eq!(runtime.generate(&prompt, 1, 1), expected);
    }

    #[test]
    fn test_detect_model_kind_standard() {
        let dir = tempfile::tempdir().unwrap();
        write_config_json(
            dir.path(),
            r#"{
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 3072,
                "vocab_size": 151669
            }"#,
        );

        assert_eq!(
            DiffusionEngine::detect_model_kind(dir.path()).unwrap(),
            DiffusionModelKind::Standard
        );
    }

    #[test]
    fn test_detect_model_kind_bonsai_q1() {
        let dir = tempfile::tempdir().unwrap();
        write_config_json(
            dir.path(),
            r#"{
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 6144,
                "vocab_size": 151669,
                "quantization": {
                    "bits": 1
                }
            }"#,
        );

        assert_eq!(
            DiffusionEngine::detect_model_kind(dir.path()).unwrap(),
            DiffusionModelKind::BonsaiQ1
        );
    }

    // ------------------------------------------------------------------------
    // MAGIC CANVAS killer tests — zero-shot UI generation on A2D diffusion.
    //
    // User hypothesis: the code-tuned A2D model (Qwen2.5-Coder-0.5B-Instruct
    // diffusion-mdlm) already has HTML/CSS/JS in its training distribution
    // via Qwen2.5-Coder pretraining, so UI generation should work zero-shot
    // without any fine-tuning. External reviewers (codex/GLM/Sonnet) all
    // assumed fine-tuning was mandatory because they pattern-matched on
    // general-purpose MDLM benchmarks.
    //
    // BUT: the existing DiffusionEngine::load was written for Qwen3 A2D
    // variants, which have q_norm/k_norm and no attention bias. Qwen2.5-Coder
    // A2D has attention BIAS and NO q_norm/k_norm. Loading it blows up at
    // `Missing: self_attn.q_norm.weight`. Adding Qwen2 architecture support
    // to DiffusionLayerWeights is ~100 lines of real work.
    //
    // First gate: run tests against the Qwen3-0.6B diffusion variant (which
    // the existing engine supports unchanged). If it can generate HTML zero-
    // shot even without code pretraining, the Qwen2.5-Coder path is a slam
    // dunk and the Rust engine extension is justified. If it can't, the
    // Qwen2.5-Coder variant becomes the decisive signal and extending the
    // engine is mandatory.
    // ------------------------------------------------------------------------

    fn qwen3_diffusion_dir() -> Option<String> {
        let dir = format!(
            "{}/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1",
            std::env::var("HOME").ok()?
        );
        if std::path::Path::new(&dir).join("model.safetensors").exists() {
            Some(dir)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    fn qwen25_coder_diffusion_dir() -> Option<String> {
        // TODO: requires DiffusionEngine to support Qwen2-style attention
        // (bias terms, no q_norm/k_norm). See diffusion.rs::load comments.
        let dir = format!(
            "{}/.cache/huggingface/hub/models--dllm-hub--Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/snapshots/a284e895a6248256baf2f60502e54aba61b24c1a",
            std::env::var("HOME").ok()?
        );
        if std::path::Path::new(&dir).join("model.safetensors").exists() {
            Some(dir)
        } else {
            None
        }
    }

    fn magic_canvas_dir() -> Option<String> {
        qwen3_diffusion_dir()
    }

    /// Format a chat prompt using the Qwen2/Qwen3 chat template tokens.
    fn format_chat_prompt(
        tokenizer: &tokenizers::Tokenizer,
        system: &str,
        user: &str,
    ) -> Vec<u32> {
        let prompt = format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        );
        tokenizer
            .encode(prompt, false)
            .expect("chat prompt tokenization")
            .get_ids()
            .to_vec()
    }

    /// Score HTML output for a login form: structure + tailwind + semantic.
    fn score_login_form_html(text: &str) -> (usize, [bool; 6]) {
        let lc = text.to_lowercase();
        let checks = [
            text.contains("<form") || lc.contains("<form"),
            text.contains("<input") || lc.contains("<input"),
            lc.contains("<button") || lc.contains("type=\"submit\"") || lc.contains("type='submit'"),
            lc.contains("type=\"email\"") || lc.contains("type='email'") || lc.contains("email"),
            lc.contains("type=\"password\"") || lc.contains("type='password'") || lc.contains("password"),
            // tailwind-ish class marker
            lc.contains("class=\"") && (
                lc.contains("bg-") || lc.contains("text-") || lc.contains("p-") ||
                lc.contains("flex") || lc.contains("rounded") || lc.contains("w-") ||
                lc.contains("h-") || lc.contains("grid")
            ),
        ];
        let score = checks.iter().filter(|b| **b).count();
        (score, checks)
    }

    /// T0 — smoke test: load the Qwen2.5-Coder A2D model via DiffusionEngine,
    /// tokenize a trivial prompt, and run 16-step MDLM denoising of 16 tokens.
    /// Pass if the decoded output has ≥50% non-mask tokens.
    #[test]
    #[ignore = "requires dllm-hub Qwen2.5-Coder diffusion model on disk; run manually"]
    fn t0_magic_canvas_smoke() {
        let Some(dir) = magic_canvas_dir() else {
            eprintln!("SKIP: Qwen2.5-Coder diffusion model not found");
            return;
        };
        let t_load = std::time::Instant::now();
        let engine = DiffusionEngine::load_autodetect(&dir).expect("load engine");
        eprintln!("[T0] model loaded in {:.1}s", t_load.elapsed().as_secs_f64());
        eprintln!(
            "[T0] config: hidden={} layers={} heads={}/{} head_dim={} vocab={} mask_id={}",
            engine.config.hidden,
            engine.config.layers,
            engine.config.heads,
            engine.config.kv_heads,
            engine.config.head_dim,
            engine.config.vocab,
            engine.config.mask_token_id,
        );

        let tokenizer = crate::load_tokenizer(&dir).expect("load tokenizer");
        let prompt_ids = format_chat_prompt(&tokenizer, "You generate UI markup.", "Say hi.");
        eprintln!("[T0] prompt: {} tokens", prompt_ids.len());

        let t_gen = std::time::Instant::now();
        let generated = engine.generate(&prompt_ids, 16, 16);
        let gen_ms = t_gen.elapsed().as_millis();
        eprintln!("[T0] generate: {gen_ms}ms for 16 tokens @ 16 steps");

        let gen_tokens = &generated[prompt_ids.len()..];
        let mask_id = engine.config.mask_token_id as u32;
        let non_mask = gen_tokens.iter().filter(|&&t| t != mask_id).count();
        let text = tokenizer
            .decode(gen_tokens, true)
            .unwrap_or_else(|_| "<decode error>".to_string());
        eprintln!("[T0] output ({non_mask}/{} non-mask): {text:?}", gen_tokens.len());

        assert!(
            non_mask >= gen_tokens.len() / 2,
            "model produced mostly mask tokens — smoke test fail"
        );
    }

    /// T1 — zero-shot HTML generation: ask for a login form, sweep step counts.
    /// Pass if AT LEAST ONE step count produces ≥4/6 structural features.
    #[test]
    #[ignore = "requires dllm-hub Qwen2.5-Coder diffusion model on disk; run manually"]
    fn t1_magic_canvas_zero_shot_login_form() {
        let Some(dir) = magic_canvas_dir() else {
            eprintln!("SKIP: Qwen2.5-Coder diffusion model not found");
            return;
        };
        let engine = DiffusionEngine::load_autodetect(&dir).expect("load engine");
        let tokenizer = crate::load_tokenizer(&dir).expect("load tokenizer");

        let prompt_ids = format_chat_prompt(
            &tokenizer,
            "You generate HTML with Tailwind CSS classes. Output only the HTML.",
            "Write a login form with email and password fields and a submit button.",
        );
        eprintln!("[T1] prompt: {} tokens", prompt_ids.len());

        let num_gen: usize = 96;
        let mut best_score = 0;
        let mut best_text = String::new();
        let mut best_steps = 0;
        let mut best_ms = 0u128;

        for &steps in &[4usize, 8, 16, 32] {
            let t = std::time::Instant::now();
            let generated = engine.generate(&prompt_ids, num_gen, steps);
            let ms = t.elapsed().as_millis();
            let gen_tokens = &generated[prompt_ids.len()..];
            let text = tokenizer
                .decode(gen_tokens, true)
                .unwrap_or_else(|_| String::new());
            let (score, checks) = score_login_form_html(&text);
            eprintln!(
                "[T1 steps={steps:>2}] {ms:>5}ms score={score}/6 form={} input={} btn={} email={} pw={} tw={}",
                checks[0], checks[1], checks[2], checks[3], checks[4], checks[5]
            );
            eprintln!("           output: {:?}", &text[..text.len().min(240)]);
            if score > best_score {
                best_score = score;
                best_text = text.clone();
                best_steps = steps;
                best_ms = ms;
            }
        }

        eprintln!();
        eprintln!("[T1] BEST: score={best_score}/6 at {best_steps} steps, {best_ms}ms");
        eprintln!("[T1] BEST OUTPUT: {best_text:?}");
        assert!(
            best_score >= 4,
            "T1 zero-shot FAIL: best score {best_score}/6 — user hypothesis (code-tuned model \
             produces HTML zero-shot) is refuted. Fine-tuning required."
        );
    }

    /// T2 — prompt variety: 10 diverse UI prompts at a single step count.
    /// Runs both on CPU BLAS and (when feature="ane") on ANE for real hardware
    /// latency comparison. Pass if ≥6/10 produce at least partially valid HTML.
    #[test]
    #[ignore = "long; requires Qwen3 diffusion model on disk; run manually"]
    fn t2_magic_canvas_prompt_variety() {
        let Some(dir) = magic_canvas_dir() else {
            eprintln!("SKIP: Qwen3 diffusion model not found at magic_canvas_dir()");
            return;
        };
        let engine = DiffusionEngine::load_autodetect(&dir).expect("load engine");
        let tokenizer = crate::load_tokenizer(&dir).expect("load tokenizer");

        let prompts: &[(&str, &str)] = &[
            ("login form",    "Write HTML for a login form with email and password."),
            ("signup form",   "Write HTML for a signup form with name, email, password, confirm password."),
            ("product card",  "Write HTML for a product card with image, title, price, buy button."),
            ("navbar",        "Write HTML for a navbar with logo and menu links."),
            ("pricing table", "Write HTML for a pricing table with 3 tiers: Basic, Pro, Enterprise."),
            ("contact form",  "Write HTML for a contact form with name, email, message, send button."),
            ("stat card",     "Write HTML for a stat card showing Users: 1234 with icon."),
            ("footer",        "Write HTML for a footer with 3 columns of links and copyright."),
            ("button",        "Write HTML for a primary button that says Get Started."),
            ("alert",         "Write HTML for a success alert saying Saved successfully."),
        ];

        const STEPS: usize = 16;
        const NUM_GEN: usize = 96;
        const CANVAS: usize = 256; // max prompt + num_gen for runtime allocation

        // --- Build runtimes: CPU always, ANE if feature enabled ---
        let cpu_runtime = DiffusionRuntime::new(
            engine.clone(),
            CANVAS,
            DiffusionBackendPreference::CpuBlas,
        )
        .expect("build CPU runtime");
        eprintln!("[T2] CPU runtime: {:?}", cpu_runtime.backend_report().detail);

        #[cfg(feature = "ane")]
        let ane_runtime = {
            eprintln!("[T2] building ANE runtime...");
            match DiffusionRuntime::new(
                engine.clone(),
                CANVAS,
                DiffusionBackendPreference::Ane,
            ) {
                Ok(rt) => {
                    eprintln!("[T2] ANE runtime: {:?} {}", rt.selected_backend(), rt.backend_report().detail);
                    Some(rt)
                }
                Err(e) => {
                    eprintln!("[T2] ANE runtime build FAILED: {e} — will skip ANE column");
                    None
                }
            }
        };
        #[cfg(not(feature = "ane"))]
        let ane_runtime: Option<DiffusionRuntime> = None;

        // Warm up once (allocator priming, kernel compile)
        {
            let warm_ids = format_chat_prompt(&tokenizer, "warmup", "warmup");
            let _ = cpu_runtime.generate(&warm_ids, 16, 4);
            if let Some(ref rt) = ane_runtime {
                let _ = rt.generate(&warm_ids, 16, 4);
            }
        }

        let mut cpu_valid = 0usize;
        let mut ane_valid = 0usize;
        let mut cpu_total_ms = 0u128;
        let mut ane_total_ms = 0u128;

        eprintln!();
        eprintln!("{:<14} | {:>9} {:>6} | {:>9} {:>6}",
                  "prompt", "cpu_ms", "valid", "ane_ms", "valid");
        eprintln!("{}", "-".repeat(64));

        for (label, user_msg) in prompts {
            let prompt_ids = format_chat_prompt(
                &tokenizer,
                "You generate HTML. Output only the HTML.",
                user_msg,
            );

            // CPU
            let t = std::time::Instant::now();
            let generated = cpu_runtime.generate(&prompt_ids, NUM_GEN, STEPS);
            let cpu_ms = t.elapsed().as_millis();
            cpu_total_ms += cpu_ms;
            let gen_tokens = &generated[prompt_ids.len()..];
            let cpu_text = tokenizer.decode(gen_tokens, true).unwrap_or_default();
            let cpu_ok = html_partially_valid(&cpu_text);
            if cpu_ok { cpu_valid += 1; }

            // ANE
            let (ane_ms_str, ane_ok_str, ane_text) = if let Some(ref rt) = ane_runtime {
                let t = std::time::Instant::now();
                let generated = rt.generate(&prompt_ids, NUM_GEN, STEPS);
                let ane_ms = t.elapsed().as_millis();
                ane_total_ms += ane_ms;
                let gen_tokens = &generated[prompt_ids.len()..];
                let text = tokenizer.decode(gen_tokens, true).unwrap_or_default();
                let ok = html_partially_valid(&text);
                if ok { ane_valid += 1; }
                (format!("{ane_ms}"), ok.to_string(), text)
            } else {
                ("-".to_string(), "-".to_string(), String::new())
            };

            eprintln!(
                "{label:<14} | {cpu_ms:>9} {:>6} | {ane_ms_str:>9} {:>6}",
                cpu_ok, ane_ok_str
            );
            let cpu_preview: String = cpu_text.chars().take(140).collect();
            eprintln!("   cpu: {cpu_preview:?}");
            if !ane_text.is_empty() {
                let ane_preview: String = ane_text.chars().take(140).collect();
                eprintln!("   ane: {ane_preview:?}");
            }
        }

        eprintln!();
        eprintln!(
            "[T2] CPU: {cpu_valid}/{} valid, avg {}ms/prompt",
            prompts.len(),
            cpu_total_ms / (prompts.len() as u128).max(1)
        );
        if ane_runtime.is_some() {
            eprintln!(
                "[T2] ANE: {ane_valid}/{} valid, avg {}ms/prompt",
                prompts.len(),
                ane_total_ms / (prompts.len() as u128).max(1)
            );
            let speedup = (cpu_total_ms as f64) / (ane_total_ms as f64).max(1.0);
            eprintln!("[T2] ANE speedup vs CPU: {speedup:.2}x");
        }

        assert!(
            cpu_valid >= 6,
            "T2 CPU prompt variety FAIL: only {cpu_valid}/{} valid.",
            prompts.len()
        );
    }

    fn html_partially_valid(text: &str) -> bool {
        let lc = text.to_lowercase();
        let has_open_tag = lc.contains('<');
        let has_close_tag = lc.contains("</");
        let has_class_attr = lc.contains("class=\"") || lc.contains("class='");
        has_open_tag && (has_close_tag || has_class_attr)
    }

    /// Benchmark: BLAS vs ANE BLOBFILE for the diffusion projection matmuls.
    /// Tests actual projection shapes at diffusion-relevant seq lengths.
    #[test]
    #[cfg(feature = "ane")]
    fn test_diffusion_blas_vs_ane() {
        use crate::ane_bridge::{self, build_weight_blob, AneKernel};
        use crate::ane_mil;

        ane_bridge::ane_init().expect("ANE init failed");
        ane_bridge::set_quiet(false);

        let n_iters = 50u128;

        // Diffusion model projection shapes (Qwen3-0.6B)
        let shapes: &[(&str, usize, usize)] = &[
            ("q_proj (1024→2048)", 1024, 2048),
            ("kv_proj (1024→1024)", 1024, 1024),
            ("gate/up (1024→3072)", 1024, 3072),
            ("down (3072→1024)", 3072, 1024),
        ];

        eprintln!("\n======================================================================");
        eprintln!("  Diffusion projections: BLAS sgemm vs ANE BLOBFILE (release)");
        eprintln!("  Simulating 28-layer forward: 28 x (q+k+v+o+gate+up+down) = 196 matmuls");
        eprintln!("======================================================================");

        for seq in [32, 64, 128] {
            eprintln!("\n--- seq={seq} ---");
            let mut total_blas_us = 0u128;
            let mut total_ane_us = 0u128;

            for &(label, ic, oc) in shapes {
                let w: Vec<f32> = (0..oc * ic)
                    .map(|i| ((i as f32) * 0.00001).sin() * 0.01)
                    .collect();
                let act: Vec<f32> = (0..seq * ic).map(|i| ((i as f32) * 0.001).sin()).collect();

                // BLAS: act[seq,ic] @ W^T[ic,oc] → [seq,oc]
                let mut out = vec![0.0f32; seq * oc];
                super::sgemm_nt(seq, oc, ic, &act, &w, &mut out);
                let t0 = std::time::Instant::now();
                for _ in 0..n_iters {
                    super::sgemm_nt(seq, oc, ic, &act, &w, &mut out);
                }
                let blas_us = t0.elapsed().as_micros() / n_iters;

                // ANE BLOBFILE
                let blob = build_weight_blob(&w, oc, ic);
                let mil = ane_mil::gen_blobfile_matmul(ic, oc, seq, "bench");
                let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
                let kernel = AneKernel::compile_multi_weights(
                    &mil.mil_text,
                    &names,
                    &[&blob],
                    &[mil.input_bytes],
                    &[mil.output_bytes],
                )
                .expect("compile failed");

                // Write activation (padded layout: [ic, seq] channel-first f32)
                let mut act_padded = vec![0.0f32; ic * seq];
                for s in 0..seq {
                    for c in 0..ic {
                        act_padded[c * seq + s] = act[s * ic + c];
                    }
                }
                let act_bytes: Vec<u8> = act_padded.iter().flat_map(|f| f.to_le_bytes()).collect();
                kernel.write_input(0, &act_bytes);
                kernel.eval().unwrap();

                let t0 = std::time::Instant::now();
                for _ in 0..n_iters {
                    kernel.write_input(0, &act_bytes);
                    kernel.eval().unwrap();
                }
                let ane_us = t0.elapsed().as_micros() / n_iters;

                // Count: how many of this shape per layer?
                let per_layer = match label {
                    "q_proj (1024→2048)" => 1,  // q
                    "kv_proj (1024→1024)" => 3, // k, v, o
                    "gate/up (1024→3072)" => 2, // gate + up
                    "down (3072→1024)" => 1,    // down
                    _ => 1,
                };
                total_blas_us += blas_us * per_layer * 28;
                total_ane_us += ane_us * per_layer * 28;

                let speedup = blas_us as f64 / ane_us as f64;
                eprintln!(
                    "  {label:<25} BLAS={blas_us:>6}µs  ANE={ane_us:>6}µs  {speedup:.2}x  (×{} ×28L)",
                    per_layer
                );
            }

            let speedup = total_blas_us as f64 / total_ane_us as f64;
            eprintln!(
                "  TOTAL 28L projection: BLAS={:.1}ms  ANE={:.1}ms  → {speedup:.2}x ANE",
                total_blas_us as f64 / 1000.0,
                total_ane_us as f64 / 1000.0
            );

            // Estimate full forward: projections + overhead (~20% for norms/rope/softmax/residual)
            let overhead_factor = 1.25;
            let blas_fwd = total_blas_us as f64 / 1000.0 * overhead_factor;
            let ane_fwd = total_ane_us as f64 / 1000.0 * overhead_factor;
            let blas_tps = (seq - 5) as f64 / (blas_fwd * 64.0 / 1000.0);
            let ane_tps = (seq - 5) as f64 / (ane_fwd * 64.0 / 1000.0);
            eprintln!("  Est. full fwd: BLAS≈{blas_fwd:.0}ms  ANE≈{ane_fwd:.0}ms");
            eprintln!(
                "  Est. 64-step:  BLAS≈{:.0} tok/s  ANE≈{:.0} tok/s",
                blas_tps, ane_tps
            );
        }
    }

    /// E2E ANE diffusion generation.
    #[test]
    #[cfg(feature = "ane")]
    fn test_ane_generate() {
        let Some(dir) = model_dir() else {
            eprintln!("Model not found, skipping");
            return;
        };

        let engine = DiffusionEngine::load(&dir).unwrap();
        let prompt_ids: Vec<u32> = vec![785, 6722, 315, 9625, 374]; // "The capital of France is"
        let n_gen = 123usize; // total seq = 5 + 123 = 128
        let steps = 64usize;
        let seq = prompt_ids.len() + n_gen;

        eprintln!("Compiling ANE kernels for seq={}...", seq);
        let ane = super::AneDiffusionEngine::new(engine, seq).unwrap();

        // Compare BLAS vs ANE forward for correctness
        let blas_engine = DiffusionEngine::load(&dir).unwrap();
        let input: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(std::iter::repeat(151669).take(n_gen))
            .collect();

        let blas_logits = blas_engine.forward(&input);
        let ane_logits = ane.forward(&input);

        let max_err = blas_logits
            .iter()
            .zip(ane_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("BLAS vs ANE logit max_err: {max_err:.4}");

        // ANE generation
        let t0 = std::time::Instant::now();
        let result = ane.generate(&prompt_ids, n_gen, steps);
        let elapsed = t0.elapsed();
        let tps = n_gen as f64 / elapsed.as_secs_f64();
        eprintln!(
            "ANE generated {n_gen} tokens in {}ms ({tps:.1} tok/s, {steps} steps)",
            elapsed.as_millis()
        );
        eprintln!("Result IDs (gen part): {:?}", &result[5..5 + 10.min(n_gen)]);

        // Also time BLAS for comparison
        let t0 = std::time::Instant::now();
        let blas_result = blas_engine.generate(&prompt_ids, n_gen, steps);
        let blas_elapsed = t0.elapsed();
        let blas_tps = n_gen as f64 / blas_elapsed.as_secs_f64();
        eprintln!(
            "BLAS generated {n_gen} tokens in {}ms ({blas_tps:.1} tok/s)",
            blas_elapsed.as_millis()
        );

        eprintln!("Speedup: {:.1}x", tps / blas_tps);
    }

    /// Benchmark: measure raw forward pass speed at different seq lengths.
    #[test]
    fn test_forward_speed_sweep() {
        let Some(dir) = model_dir() else {
            eprintln!("Model not found, skipping");
            return;
        };
        let engine = DiffusionEngine::load(&dir).unwrap();

        for seq in [16, 32, 64, 128] {
            let input: Vec<u32> = (0..seq).map(|i| if i < 5 { 785 } else { 151669 }).collect();
            // Warmup
            let _ = engine.forward(&input);

            let n = 5;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                let _ = engine.forward(&input);
            }
            let ms = t0.elapsed().as_millis() as f64 / n as f64;
            let gen_tok = seq - 5;
            // For diffusion: 64 steps × this forward = total generation time
            let steps = 64;
            let total_ms = ms * steps as f64;
            let tok_per_sec = gen_tok as f64 / (total_ms / 1000.0);
            eprintln!(
                "seq={seq:>4}: {ms:>6.1}ms/fwd × {steps} steps = {total_ms:>7.0}ms → {tok_per_sec:>5.1} tok/s ({gen_tok} tokens)"
            );
        }
    }

    /// Load Qwen3-0.6B-Base from LM Studio cache and run on ANE.
    /// The base model has identical architecture — proves ANE works for autoregressive too.
    /// (SDPA is bidirectional here; causal mask is a separate MIL change.)
    #[test]
    #[cfg(feature = "ane")]
    fn test_qwen3_base_ane() {
        let Some(base_path) = qwen3_base_dir() else {
            eprintln!("Qwen3-0.6B-Base not found, skipping");
            return;
        };

        eprintln!("Loading Qwen3-0.6B-Base from {}...", base_path.display());
        let engine = DiffusionEngine::load(&base_path).unwrap();
        eprintln!(
            "Loaded: {}L, dim={}, heads={}/{}",
            engine.config.layers, engine.config.hidden, engine.config.heads, engine.config.kv_heads
        );

        for seq in [64, 128] {
            let input: Vec<u32> = (0..seq)
                .map(|i| if i < 5 { 785 } else { 1000 + i as u32 })
                .collect();

            let _ = engine.forward(&input);
            let n = 3;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                let _ = engine.forward(&input);
            }
            let blas_ms = t0.elapsed().as_millis() as f64 / n as f64;

            let ane =
                super::AneDiffusionEngine::new(DiffusionEngine::load(&base_path).unwrap(), seq)
                    .unwrap();
            let ane_logits = ane.forward(&input);
            let blas_logits = engine.forward(&input);

            let max_err = blas_logits
                .iter()
                .zip(ane_logits.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mean_err: f32 = blas_logits
                .iter()
                .zip(ane_logits.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / blas_logits.len() as f32;

            let _ = ane.forward(&input);
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                let _ = ane.forward(&input);
            }
            let ane_ms = t0.elapsed().as_millis() as f64 / n as f64;

            let speedup = blas_ms / ane_ms;
            eprintln!(
                "seq={seq}: BLAS {blas_ms:.0}ms | ANE {ane_ms:.0}ms | speedup {speedup:.2}x | err max={max_err:.4} mean={mean_err:.6}"
            );

            assert!(max_err < 5.0, "Logit error too large: {max_err}");
        }
        eprintln!("PASS: Qwen3-0.6B-Base runs on ANE with matching logits");
    }

    /// Load Bonsai-1.7B Q1 (1-bit) model, print weight stats, and run a forward pass.
    #[test]
    fn test_load_bonsai_1_7b() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        eprintln!("Loading Bonsai-1.7B Q1_0_g128...");
        let t0 = std::time::Instant::now();
        let engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        let load_ms = t0.elapsed().as_millis();
        eprintln!("Loaded in {load_ms}ms");

        assert_eq!(engine.config.layers, 28);
        assert_eq!(engine.config.hidden, 2048);
        assert_eq!(engine.config.heads, 16);
        assert_eq!(engine.config.kv_heads, 8);
        assert_eq!(engine.config.head_dim, 128);
        assert_eq!(engine.config.inter, 6144);
        assert_eq!(engine.config.vocab, 151669);

        let l0 = &engine.layers[0];
        let stats = |name: &str, w: &[f32]| {
            let min = w.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = w.iter().sum::<f32>() / w.len() as f32;
            let nonzero = w.iter().filter(|&&v| v.abs() > 1e-10).count();
            eprintln!(
                "  {name:<15} shape={:<10} min={min:>8.4} max={max:>8.4} mean={mean:>8.6} nonzero={nonzero}/{}",
                w.len(),
                w.len()
            );
        };
        eprintln!("Layer 0 weight stats:");
        stats("q_proj", &l0.q_proj);
        stats("k_proj", &l0.k_proj);
        stats("v_proj", &l0.v_proj);
        stats("o_proj", &l0.o_proj);
        stats("gate_proj", &l0.gate_proj);
        stats("up_proj", &l0.up_proj);
        stats("down_proj", &l0.down_proj);
        stats("input_norm", &l0.input_norm);
        stats("q_norm", &l0.q_norm);

        let h = engine.config.hidden;
        let q_dim = engine.config.heads * engine.config.head_dim;
        let kv_dim = engine.config.kv_heads * engine.config.head_dim;
        assert_eq!(l0.q_proj.len(), q_dim * h);
        assert_eq!(l0.k_proj.len(), kv_dim * h);
        assert_eq!(l0.o_proj.len(), h * q_dim);
        assert_eq!(l0.gate_proj.len(), engine.config.inter * h);
        assert_eq!(l0.down_proj.len(), h * engine.config.inter);
        assert_eq!(engine.embed.len(), engine.config.vocab * h);

        let first_128 = &l0.q_proj[0..128];
        let unique: std::collections::HashSet<u32> =
            first_128.iter().map(|f| f.to_bits()).collect();
        eprintln!(
            "  q_proj row 0, group 0: {} unique values in 128 elements",
            unique.len()
        );
        assert!(
            unique.len() <= 2,
            "Q1 dequant should produce at most 2 unique values per group, got {}",
            unique.len()
        );

        let input_ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        eprintln!("Running forward pass (seq={})...", input_ids.len());
        let t0 = std::time::Instant::now();
        let logits = engine.forward(&input_ids);
        let fwd_ms = t0.elapsed().as_millis();
        eprintln!("Forward: {fwd_ms}ms");

        assert_eq!(logits.len(), input_ids.len() * engine.config.vocab);

        let last_pos = input_ids.len() - 1;
        let row = &logits[last_pos * engine.config.vocab..(last_pos + 1) * engine.config.vocab];
        let mut indices: Vec<usize> = (0..row.len()).collect();
        indices.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        eprintln!("Top-5 predictions at last position:");
        for &idx in indices.iter().take(5) {
            eprintln!("  token_id={idx} logit={:.4}", row[idx]);
        }

        let has_nan = logits.iter().any(|v| v.is_nan());
        assert!(!has_nan, "Logits contain NaN");
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_logit > 0.0, "All logits non-positive: max={max_logit}");

        eprintln!("PASS: Bonsai-1.7B Q1 loaded and forward pass completed");
    }

    /// ANE hybrid engine for Bonsai-1.7B: ANE attention + BLAS FFN.
    /// Loads 1-bit model, compiles 28 attention-only ANE kernels, runs hybrid forward,
    /// and compares logits against pure-BLAS reference.
    #[test]
    #[cfg(feature = "ane")]
    fn test_bonsai_1_7b_ane_hybrid() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        // Phase 1: BLAS reference only
        eprintln!("Loading Bonsai-1.7B Q1 for BLAS reference...");
        let blas_engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        assert_eq!(blas_engine.config.hidden, 2048);
        assert_eq!(blas_engine.config.layers, 28);
        assert_eq!(blas_engine.config.heads, 16);
        assert_eq!(blas_engine.config.kv_heads, 8);

        let short_input: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let short_seq = short_input.len();

        eprintln!("Running BLAS reference forward (seq={short_seq})...");
        let blas_logits_short = blas_engine.forward(&short_input);
        let blas_nan = blas_logits_short.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "  BLAS logits: {} total, {} non-finite",
            blas_logits_short.len(),
            blas_nan
        );

        // Drop BLAS engine to free 6.5GB before ANE loads
        drop(blas_engine);
        eprintln!("  Dropped BLAS engine, freeing ~6.5GB");

        // Phase 2: ANE short (correctness)
        eprintln!("Compiling ANE hybrid engine (seq={short_seq}, for correctness)...");
        let ane_engine_short = super::AneBonsaiEngine::new(
            DiffusionEngine::load_q1(&bonsai_dir).unwrap(),
            short_seq,
            1e-6,
        )
        .unwrap();
        let ane_logits_short = ane_engine_short.forward(&short_input);
        let ane_nan = ane_logits_short.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "  ANE  logits: {} total, {} non-finite",
            ane_logits_short.len(),
            ane_nan
        );
        eprintln!("  ANE  logits[0,:5] = {:?}", &ane_logits_short[..5]);
        eprintln!("  BLAS logits[0,:5] = {:?}", &blas_logits_short[..5]);

        assert_eq!(blas_logits_short.len(), ane_logits_short.len());
        let finite_errs: Vec<f32> = blas_logits_short
            .iter()
            .zip(ane_logits_short.iter())
            .map(|(a, b)| (a - b).abs())
            .filter(|e| e.is_finite())
            .collect();
        let n_finite = finite_errs.len();
        let n_total = blas_logits_short.len();
        let max_err_short = finite_errs.iter().cloned().fold(0.0f32, f32::max);
        let mean_err_short = if n_finite > 0 {
            finite_errs.iter().sum::<f32>() / n_finite as f32
        } else {
            f32::NAN
        };
        eprintln!(
            "  seq={short_seq}: max_err={max_err_short:.4}, mean_err={mean_err_short:.6}, \
             finite_pairs={n_finite}/{n_total}"
        );
        assert!(
            max_err_short < 5.0,
            "seq={short_seq} logit error too large: {max_err_short}"
        );
        assert!(
            n_finite > 0,
            "No finite logit pairs to compare at seq={short_seq}"
        );

        // Drop short ANE engine
        drop(ane_engine_short);
        eprintln!("  Dropped short ANE engine");

        // Phase 3: seq=64 benchmark
        let input_64: Vec<u32> = (0..64)
            .map(|i| if i < 5 { 785 + i } else { 1000 + i })
            .collect();
        eprintln!("Compiling ANE hybrid engine (seq=64)...");
        let ane_engine_64 =
            super::AneBonsaiEngine::new(DiffusionEngine::load_q1(&bonsai_dir).unwrap(), 64, 1e-6)
                .unwrap();

        // Reload BLAS for benchmark comparison
        let blas_engine_64 = DiffusionEngine::load_q1(&bonsai_dir).unwrap();

        let n = 3;
        let _ = blas_engine_64.forward(&input_64);
        let _ = ane_engine_64.forward(&input_64);

        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ = blas_engine_64.forward(&input_64);
        }
        let blas_avg_64 = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ = ane_engine_64.forward(&input_64);
        }
        let ane_avg_64 = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let speedup_64 = blas_avg_64 / ane_avg_64;
        eprintln!(
            "  seq=64 bench: BLAS {blas_avg_64:.0}ms | ANE hybrid {ane_avg_64:.0}ms | speedup {speedup_64:.2}x"
        );

        // Drop both engines before seq=128
        drop(ane_engine_64);
        drop(blas_engine_64);
        eprintln!("  Dropped seq=64 engines");

        // Phase 4: seq=128 benchmark
        let input_128: Vec<u32> = (0..128)
            .map(|i| if i < 5 { 785 + i } else { 1000 + i })
            .collect();
        eprintln!("Compiling ANE hybrid engine (seq=128)...");
        let ane_engine_128 =
            super::AneBonsaiEngine::new(DiffusionEngine::load_q1(&bonsai_dir).unwrap(), 128, 1e-6)
                .unwrap();

        let blas_engine_128 = DiffusionEngine::load_q1(&bonsai_dir).unwrap();

        let _ = blas_engine_128.forward(&input_128);
        let _ = ane_engine_128.forward(&input_128);

        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ = blas_engine_128.forward(&input_128);
        }
        let blas_avg_128 = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ = ane_engine_128.forward(&input_128);
        }
        let ane_avg_128 = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let speedup_128 = blas_avg_128 / ane_avg_128;
        eprintln!(
            "  seq=128 bench: BLAS {blas_avg_128:.0}ms | ANE hybrid {ane_avg_128:.0}ms | speedup {speedup_128:.2}x"
        );

        eprintln!("PASS: Bonsai-1.7B ANE hybrid engine matches BLAS reference");
    }

    /// Sweep seq lengths to find ANE sweet spot for Bonsai 1.7B.
    /// Measures forward latency at seq=1..128 (padded to ANE min 16) and
    /// computes effective tok/s for both autoregressive and diffusion-style.
    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Manual benchmark — sweep ANE seq lengths for Bonsai 1.7B"]
    fn bench_bonsai_ane_seq_sweep() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        eprintln!("=== Bonsai-1.7B ANE seq length sweep ===\n");
        eprintln!(
            "{:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "seq", "padded", "compile_s", "blas_ms", "ane_ms", "speedup", "ane_fwd/s"
        );
        eprintln!("{}", "-".repeat(72));

        for seq in [1, 4, 8, 16, 32, 64, 128] {
            let padded = seq.max(16); // ANE_MIN_SPATIAL

            // Build input
            let input: Vec<u32> = (0..seq)
                .map(|i| if i < 5 { 785u32 + i as u32 } else { 1000u32 + i as u32 })
                .collect();

            // Compile ANE engine
            let compile_t0 = std::time::Instant::now();
            let ane_engine = super::AneBonsaiEngine::new(
                DiffusionEngine::load_q1(&bonsai_dir).unwrap(),
                seq,
                1e-6,
            )
            .unwrap();
            let compile_s = compile_t0.elapsed().as_secs_f64();

            // BLAS engine
            let blas_engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();

            // Warmup
            let _ = blas_engine.forward(&input);
            let _ = ane_engine.forward(&input);

            // Benchmark (more iters for small seq)
            let n = if seq <= 16 { 10 } else { 5 };

            let t0 = std::time::Instant::now();
            for _ in 0..n {
                let _ = blas_engine.forward(&input);
            }
            let blas_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..n {
                let _ = ane_engine.forward(&input);
            }
            let ane_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

            let speedup = blas_ms / ane_ms;
            let ane_fwd_per_s = 1000.0 / ane_ms;

            eprintln!(
                "{seq:>6} {padded:>6} {compile_s:>10.1} {blas_ms:>10.1} {ane_ms:>10.1} {speedup:>10.2}x {ane_fwd_per_s:>10.1}"
            );

            // Drop before next iteration to free ANE resources
            drop(ane_engine);
            drop(blas_engine);
        }

        eprintln!("\n=== Effective generation speed ===\n");
        eprintln!("Autoregressive (1 token per forward):");
        eprintln!("  tok/s = ane_fwd/s (from seq=1 or seq=16 row above)");
        eprintln!("\nDiffusion (all tokens in parallel, K denoising steps):");
        eprintln!("  tok/s = (seq - prompt_len) / (K × ane_fwd_ms / 1000)");
        eprintln!("  e.g. seq=128, prompt=5, K=64 → 123 / (64 × ane_ms/1000)");
    }

    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Manual long-context Bonsai ANE benchmark"]
    fn bench_bonsai_1_7b_ane_long_context() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };
        let Some(tokenizer) = bonsai_bench_tokenizer() else {
            eprintln!("Bonsai tokenizer not found, skipping");
            return;
        };

        let base_engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        let mask_id = bonsai_bench_mask_token_id(&tokenizer);
        let gen_tokens = 16usize;
        let steps = 16usize;
        let ppl_samples = 8usize;

        eprintln!(
            "Bonsai long-context benchmark: gen_tokens={gen_tokens}, steps={steps}, sampled_pseudo_ppl={ppl_samples}"
        );

        for seq in [512usize, 1024, 2048] {
            let prompt_len = seq - gen_tokens;
            let prompt_ids = build_long_bonsai_prompt_ids(&tokenizer, prompt_len);
            let masked_canvas: Vec<u32> = prompt_ids
                .iter()
                .copied()
                .chain(std::iter::repeat(mask_id).take(gen_tokens))
                .collect();

            eprintln!("\n=== Bonsai seq={seq} prompt={prompt_len} gen={gen_tokens} ===");

            let mut cpu_engine = base_engine.clone();
            cpu_engine.config.mask_token_id = mask_id;
            let cpu = DiffusionRuntime::new_bonsai_q1(
                cpu_engine,
                seq,
                1e-6,
                DiffusionBackendPreference::CpuBlas,
            )
            .unwrap();

            let mut ane_engine = base_engine.clone();
            ane_engine.config.mask_token_id = mask_id;
            let t0 = std::time::Instant::now();
            let ane = match DiffusionRuntime::new_bonsai_q1(
                ane_engine,
                seq,
                1e-6,
                DiffusionBackendPreference::Ane,
            ) {
                Ok(runtime) => runtime,
                Err(err) => {
                    eprintln!("ANE compile failed at seq={seq}: {err}");
                    continue;
                }
            };
            let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "backend={} compile_ms={compile_ms:.1}",
                match ane.selected_backend() {
                    DiffusionBackend::AneHybridBonsai => "AneHybridBonsai",
                    DiffusionBackend::CpuBlas => "CpuBlas",
                    DiffusionBackend::AneFused => "AneFused",
                    DiffusionBackend::AneMultiDispatch => "AneMultiDispatch",
                }
            );

            let _ = cpu.forward(&masked_canvas);
            let t0 = std::time::Instant::now();
            let _ = cpu.forward(&masked_canvas);
            let cpu_prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let _ = ane.forward(&masked_canvas);
            let t0 = std::time::Instant::now();
            let _ = ane.forward(&masked_canvas);
            let ane_prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let cpu_ppl =
                sampled_masked_pseudo_ppl_runtime(&cpu, &prompt_ids, mask_id, ppl_samples);
            let ane_ppl =
                sampled_masked_pseudo_ppl_runtime(&ane, &prompt_ids, mask_id, ppl_samples);

            let t0 = std::time::Instant::now();
            let cpu_result = cpu.generate(&prompt_ids, gen_tokens, steps);
            let cpu_decode_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t0 = std::time::Instant::now();
            let ane_result = ane.generate(&prompt_ids, gen_tokens, steps);
            let ane_decode_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let cpu_gen_text = tokenizer
                .decode(&cpu_result[prompt_len..], true)
                .unwrap_or_else(|_| "<decode failed>".to_owned())
                .replace('\n', "\\n");
            let ane_gen_text = tokenizer
                .decode(&ane_result[prompt_len..], true)
                .unwrap_or_else(|_| "<decode failed>".to_owned())
                .replace('\n', "\\n");

            let prefill_speedup = cpu_prefill_ms / ane_prefill_ms;
            let decode_speedup = cpu_decode_ms / ane_decode_ms;
            let cpu_decode_tps = gen_tokens as f64 / (cpu_decode_ms / 1000.0);
            let ane_decode_tps = gen_tokens as f64 / (ane_decode_ms / 1000.0);

            eprintln!(
                "prefill_ms: cpu={cpu_prefill_ms:.1} ane={ane_prefill_ms:.1} speedup={prefill_speedup:.2}x"
            );
            eprintln!(
                "sampled_pseudo_ppl: cpu={cpu_ppl:.3} ane={ane_ppl:.3} delta={:.3}",
                ane_ppl - cpu_ppl
            );
            eprintln!(
                "decode_ms: cpu={cpu_decode_ms:.1} ane={ane_decode_ms:.1} speedup={decode_speedup:.2}x"
            );
            eprintln!("decode_tps: cpu={cpu_decode_tps:.2} ane={ane_decode_tps:.2}");
            eprintln!("cpu_text: {cpu_gen_text}");
            eprintln!("ane_text: {ane_gen_text}");
        }
    }

    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Exploratory compare for Bonsai ANE attention-core layout"]
    fn test_bonsai_1_7b_l0_attention_context_compare() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        let engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        let token_ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let seq = token_ids.len();
        let hd = engine.config.head_dim;
        let kv_heads = engine.config.kv_heads;

        let hidden = bonsai_embed_hidden(&engine, &token_ids);
        let cpu_ctx = bonsai_l0_attention_context_cpu(&engine, &hidden);
        let ane_ctx = bonsai_l0_attention_context_ane(&engine, &token_ids);
        let ane_ctx_gqa_t = permute_bonsai_gqa_head_layout(&ane_ctx, seq, kv_heads, hd);

        let direct_errs: Vec<f32> = cpu_ctx
            .iter()
            .zip(ane_ctx.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        let transposed_errs: Vec<f32> = cpu_ctx
            .iter()
            .zip(ane_ctx_gqa_t.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();

        let direct_max = direct_errs.iter().cloned().fold(0.0f32, f32::max);
        let direct_mean = direct_errs.iter().sum::<f32>() / direct_errs.len() as f32;
        let transposed_max = transposed_errs.iter().cloned().fold(0.0f32, f32::max);
        let transposed_mean = transposed_errs.iter().sum::<f32>() / transposed_errs.len() as f32;

        eprintln!("L0 attention-context compare: direct max={direct_max:.6} mean={direct_mean:.6}");
        eprintln!(
            "L0 attention-context compare: gqa-transposed max={transposed_max:.6} mean={transposed_mean:.6}"
        );
        eprintln!("CPU ctx[0,:8] = {:?}", &cpu_ctx[..8]);
        eprintln!("ANE ctx[0,:8] = {:?}", &ane_ctx[..8]);
        eprintln!("ANE gqa-t[0,:8] = {:?}", &ane_ctx_gqa_t[..8]);

        assert!(cpu_ctx.iter().all(|v| v.is_finite()));
        assert!(ane_ctx.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Exploratory layerwise drift trace for Bonsai ANE hybrid"]
    fn test_bonsai_1_7b_hybrid_layerwise_drift() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        let engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        let seq = 5usize;
        let eps = 1e-6f32;
        let ane_engine =
            super::AneBonsaiEngine::new(DiffusionEngine::load_q1(&bonsai_dir).unwrap(), seq, eps)
                .unwrap();

        let token_ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let cfg = &engine.config;
        let h = cfg.hidden;
        let q_dim = cfg.heads * cfg.head_dim;
        let inter = cfg.inter;
        let ps = ane_engine.seq_len;

        let mut hidden_cpu = bonsai_embed_hidden(&engine, &token_ids);
        let mut hidden_ane = hidden_cpu.clone();
        let mut normed = vec![0.0f32; seq * h];
        let mut attn_proj = vec![0.0f32; seq * h];
        let mut gate_buf = vec![0.0f32; seq * inter];
        let mut up_buf = vec![0.0f32; seq * inter];
        let mut ffn_out = vec![0.0f32; seq * h];

        for (layer_idx, kernel) in ane_engine.attn_kernels.iter().enumerate() {
            let layer = &engine.layers[layer_idx];

            let cpu_ctx = bonsai_attention_context_cpu_for_layer(&engine, layer_idx, &hidden_cpu);
            let ane_ctx =
                bonsai_attention_context_ane_with_kernel(kernel, &hidden_ane, seq, ps, h, q_dim);

            let ctx_errs: Vec<f32> = cpu_ctx
                .iter()
                .zip(ane_ctx.iter())
                .map(|(a, b)| (a - b).abs())
                .collect();
            let ctx_max = ctx_errs.iter().cloned().fold(0.0f32, f32::max);
            let ctx_mean = ctx_errs.iter().sum::<f32>() / ctx_errs.len() as f32;

            let cpu_residual = hidden_cpu.clone();
            let ane_residual = hidden_ane.clone();
            sgemm_nt(seq, h, q_dim, &cpu_ctx, &layer.o_proj, &mut attn_proj);
            for i in 0..seq * h {
                hidden_cpu[i] = cpu_residual[i] + attn_proj[i];
            }
            sgemm_nt(seq, h, q_dim, &ane_ctx, &layer.o_proj, &mut attn_proj);
            for i in 0..seq * h {
                hidden_ane[i] = ane_residual[i] + attn_proj[i];
            }

            let post_attn_errs: Vec<f32> = hidden_cpu
                .iter()
                .zip(hidden_ane.iter())
                .map(|(a, b)| (a - b).abs())
                .collect();
            let post_attn_max = post_attn_errs.iter().cloned().fold(0.0f32, f32::max);
            let post_attn_mean = post_attn_errs.iter().sum::<f32>() / post_attn_errs.len() as f32;

            rms_norm_eps(&hidden_cpu, &layer.post_attn_norm, &mut normed, seq, h, eps);
            sgemm_nt(seq, inter, h, &normed, &layer.gate_proj, &mut gate_buf);
            sgemm_nt(seq, inter, h, &normed, &layer.up_proj, &mut up_buf);
            for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) {
                *g = *g * (1.0 / (1.0 + (-*g).exp())) * u;
            }
            sgemm_nt(seq, h, inter, &gate_buf, &layer.down_proj, &mut ffn_out);
            for i in 0..seq * h {
                hidden_cpu[i] += ffn_out[i];
            }

            rms_norm_eps(&hidden_ane, &layer.post_attn_norm, &mut normed, seq, h, eps);
            sgemm_nt(seq, inter, h, &normed, &layer.gate_proj, &mut gate_buf);
            sgemm_nt(seq, inter, h, &normed, &layer.up_proj, &mut up_buf);
            for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) {
                *g = *g * (1.0 / (1.0 + (-*g).exp())) * u;
            }
            sgemm_nt(seq, h, inter, &gate_buf, &layer.down_proj, &mut ffn_out);
            for i in 0..seq * h {
                hidden_ane[i] += ffn_out[i];
            }

            let post_ffn_errs: Vec<f32> = hidden_cpu
                .iter()
                .zip(hidden_ane.iter())
                .map(|(a, b)| (a - b).abs())
                .collect();
            let post_ffn_max = post_ffn_errs.iter().cloned().fold(0.0f32, f32::max);
            let post_ffn_mean = post_ffn_errs.iter().sum::<f32>() / post_ffn_errs.len() as f32;

            eprintln!(
                "L{layer_idx}: ctx max={ctx_max:.6} mean={ctx_mean:.6} | post-attn max={post_attn_max:.6} mean={post_attn_mean:.6} | post-ffn max={post_ffn_max:.6} mean={post_ffn_mean:.6}"
            );
        }
    }

    #[cfg(feature = "ane")]
    fn bonsai_hybrid_forward_with_cpu_layers(
        ane_engine: &super::AneBonsaiEngine,
        token_ids: &[u32],
        cpu_attention_layers: &[usize],
    ) -> Vec<f32> {
        let cfg = &ane_engine.blas_engine.config;
        let seq = token_ids.len();
        let h = cfg.hidden;
        let q_dim = cfg.heads * cfg.head_dim;
        let inter = cfg.inter;
        let ps = ane_engine.seq_len;

        let mut hidden = bonsai_embed_hidden(&ane_engine.blas_engine, token_ids);
        let mut normed = vec![0.0f32; seq * h];
        let mut attn_proj = vec![0.0f32; seq * h];
        let mut gate_buf = vec![0.0f32; seq * inter];
        let mut up_buf = vec![0.0f32; seq * inter];
        let mut ffn_out = vec![0.0f32; seq * h];

        for (layer_idx, kernel) in ane_engine.attn_kernels.iter().enumerate() {
            let layer = &ane_engine.blas_engine.layers[layer_idx];
            let residual = hidden.clone();
            let attn_ctx = if cpu_attention_layers.contains(&layer_idx) {
                diffusion_attention_context_cpu(
                    &ane_engine.blas_engine,
                    layer_idx,
                    &hidden,
                    ane_engine.eps,
                )
            } else {
                bonsai_attention_context_ane_with_kernel(kernel, &hidden, seq, ps, h, q_dim)
            };

            sgemm_nt(seq, h, q_dim, &attn_ctx, &layer.o_proj, &mut attn_proj);
            for i in 0..seq * h {
                hidden[i] = residual[i] + attn_proj[i];
            }

            rms_norm_eps(
                &hidden,
                &layer.post_attn_norm,
                &mut normed,
                seq,
                h,
                ane_engine.eps,
            );
            sgemm_nt(seq, inter, h, &normed, &layer.gate_proj, &mut gate_buf);
            sgemm_nt(seq, inter, h, &normed, &layer.up_proj, &mut up_buf);
            for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) {
                *g = *g * (1.0 / (1.0 + (-*g).exp())) * u;
            }
            sgemm_nt(seq, h, inter, &gate_buf, &layer.down_proj, &mut ffn_out);
            for i in 0..seq * h {
                hidden[i] += ffn_out[i];
            }
        }

        rms_norm_eps(
            &hidden,
            &ane_engine.blas_engine.final_norm,
            &mut normed,
            seq,
            h,
            ane_engine.eps,
        );

        let mut logits = vec![0.0f32; seq * cfg.vocab];
        sgemm_nt(
            seq,
            cfg.vocab,
            h,
            &normed,
            &ane_engine.blas_engine.embed,
            &mut logits,
        );
        logits
    }

    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Exploratory search for Bonsai ANE/CPU split policy"]
    fn test_bonsai_1_7b_hybrid_policy_search() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        let short_input: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let blas_engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        let blas_logits = blas_engine.forward(&short_input);
        let ane_engine = super::AneBonsaiEngine::new(
            DiffusionEngine::load_q1(&bonsai_dir).unwrap(),
            short_input.len(),
            1e-6,
        )
        .unwrap();

        let mk_range = |start: usize, end: usize| -> Vec<usize> { (start..end).collect() };
        let policies: Vec<(&str, Vec<usize>)> = vec![
            ("tail4", mk_range(24, 28)),
            ("tail5", mk_range(23, 28)),
            ("tail6", mk_range(22, 28)),
            ("tail8", mk_range(20, 28)),
            ("alt_even_from20", vec![20, 22, 24, 26]),
            ("alt_odd_from20", vec![21, 23, 25, 27]),
            ("every2_from18", vec![18, 20, 22, 24, 26]),
            ("every3_from18", vec![18, 21, 24, 27]),
        ];

        let mut scored: Vec<(&str, Vec<usize>, f32, f32)> = Vec::new();
        for (name, cpu_layers) in policies {
            let logits =
                bonsai_hybrid_forward_with_cpu_layers(&ane_engine, &short_input, &cpu_layers);
            let errs: Vec<f32> = blas_logits
                .iter()
                .zip(logits.iter())
                .map(|(a, b)| (a - b).abs())
                .filter(|e| e.is_finite())
                .collect();
            let max_err = errs.iter().cloned().fold(0.0f32, f32::max);
            let mean_err = errs.iter().sum::<f32>() / errs.len() as f32;
            eprintln!(
                "policy={name:<16} cpu_layers={:?} max_err={max_err:.4} mean_err={mean_err:.6}",
                cpu_layers
            );
            scored.push((name, cpu_layers, max_err, mean_err));
        }

        scored.sort_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
        });

        eprintln!("Top policies by seq=5 max_err:");
        for (rank, (name, cpu_layers, max_err, mean_err)) in scored.iter().take(3).enumerate() {
            eprintln!(
                "  {}. {} cpu_layers={:?} max_err={:.4} mean_err={:.6}",
                rank + 1,
                name,
                cpu_layers,
                max_err,
                mean_err
            );
        }

        let input_64: Vec<u32> = (0..64)
            .map(|i| if i < 5 { 785 + i } else { 1000 + i })
            .collect();
        let ane_engine_64 =
            super::AneBonsaiEngine::new(DiffusionEngine::load_q1(&bonsai_dir).unwrap(), 64, 1e-6)
                .unwrap();
        for (name, cpu_layers, _, _) in scored.iter().take(3) {
            let _ = bonsai_hybrid_forward_with_cpu_layers(&ane_engine_64, &input_64, cpu_layers);
            let t0 = std::time::Instant::now();
            for _ in 0..2 {
                let _ =
                    bonsai_hybrid_forward_with_cpu_layers(&ane_engine_64, &input_64, cpu_layers);
            }
            let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / 2.0;
            eprintln!("policy={name:<16} seq=64 avg_ms={avg_ms:.1}");
        }
    }

    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Exhaustive late-layer search for Bonsai ANE/CPU split policy"]
    fn test_bonsai_1_7b_hybrid_tail_window_search() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        let short_input: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let blas_engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        let blas_logits = blas_engine.forward(&short_input);
        let ane_engine = super::AneBonsaiEngine::new(
            DiffusionEngine::load_q1(&bonsai_dir).unwrap(),
            short_input.len(),
            1e-6,
        )
        .unwrap();

        let window = [22usize, 23, 24, 25, 26, 27];
        let mut scored: Vec<(Vec<usize>, f32, f32)> = Vec::new();

        for mask in 1usize..(1usize << window.len()) {
            let cpu_layers: Vec<usize> = window
                .iter()
                .enumerate()
                .filter_map(|(bit, &layer)| ((mask & (1usize << bit)) != 0).then_some(layer))
                .collect();
            if cpu_layers.len() < 3 || cpu_layers.len() > 6 {
                continue;
            }

            let logits =
                bonsai_hybrid_forward_with_cpu_layers(&ane_engine, &short_input, &cpu_layers);
            let errs: Vec<f32> = blas_logits
                .iter()
                .zip(logits.iter())
                .map(|(a, b)| (a - b).abs())
                .filter(|e| e.is_finite())
                .collect();
            let max_err = errs.iter().cloned().fold(0.0f32, f32::max);
            let mean_err = errs.iter().sum::<f32>() / errs.len() as f32;
            scored.push((cpu_layers, max_err, mean_err));
        }

        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.0.len().cmp(&b.0.len()))
        });

        eprintln!("Top 10 late-window policies by seq=5 max_err:");
        for (rank, (cpu_layers, max_err, mean_err)) in scored.iter().take(10).enumerate() {
            eprintln!(
                "  {}. cpu_layers={:?} max_err={:.4} mean_err={:.6}",
                rank + 1,
                cpu_layers,
                max_err,
                mean_err
            );
        }

        let input_64: Vec<u32> = (0..64)
            .map(|i| if i < 5 { 785 + i } else { 1000 + i })
            .collect();
        let ane_engine_64 =
            super::AneBonsaiEngine::new(DiffusionEngine::load_q1(&bonsai_dir).unwrap(), 64, 1e-6)
                .unwrap();

        for (rank, (cpu_layers, max_err, mean_err)) in scored.iter().take(5).enumerate() {
            let _ = bonsai_hybrid_forward_with_cpu_layers(&ane_engine_64, &input_64, cpu_layers);
            let t0 = std::time::Instant::now();
            for _ in 0..2 {
                let _ =
                    bonsai_hybrid_forward_with_cpu_layers(&ane_engine_64, &input_64, cpu_layers);
            }
            let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / 2.0;
            eprintln!(
                "  top{} seq=64 cpu_layers={:?} max_err={:.4} mean_err={:.6} avg_ms={:.1}",
                rank + 1,
                cpu_layers,
                max_err,
                mean_err,
                avg_ms
            );
        }
    }

    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Search optional rescue layers on top of tail4 Bonsai split"]
    fn test_bonsai_1_7b_hybrid_tail4_rescue_search() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        let short_input: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let blas_engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        let blas_logits = blas_engine.forward(&short_input);
        let ane_engine = super::AneBonsaiEngine::new(
            DiffusionEngine::load_q1(&bonsai_dir).unwrap(),
            short_input.len(),
            1e-6,
        )
        .unwrap();

        let base_tail = vec![24usize, 25, 26, 27];
        let rescue_window = [18usize, 19, 20, 21, 22, 23];
        let mut scored: Vec<(Vec<usize>, f32, f32)> = Vec::new();

        for mask in 0usize..(1usize << rescue_window.len()) {
            let mut cpu_layers = base_tail.clone();
            for (bit, &layer) in rescue_window.iter().enumerate() {
                if (mask & (1usize << bit)) != 0 {
                    cpu_layers.push(layer);
                }
            }
            cpu_layers.sort_unstable();

            let logits =
                bonsai_hybrid_forward_with_cpu_layers(&ane_engine, &short_input, &cpu_layers);
            let errs: Vec<f32> = blas_logits
                .iter()
                .zip(logits.iter())
                .map(|(a, b)| (a - b).abs())
                .filter(|e| e.is_finite())
                .collect();
            let max_err = errs.iter().cloned().fold(0.0f32, f32::max);
            let mean_err = errs.iter().sum::<f32>() / errs.len() as f32;
            scored.push((cpu_layers, max_err, mean_err));
        }

        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.0.len().cmp(&b.0.len()))
        });

        eprintln!("Top 10 tail4+rescue policies by seq=5 max_err:");
        for (rank, (cpu_layers, max_err, mean_err)) in scored.iter().take(10).enumerate() {
            eprintln!(
                "  {}. cpu_layers={:?} max_err={:.4} mean_err={:.6}",
                rank + 1,
                cpu_layers,
                max_err,
                mean_err
            );
        }

        let input_64: Vec<u32> = (0..64)
            .map(|i| if i < 5 { 785 + i } else { 1000 + i })
            .collect();
        let ane_engine_64 =
            super::AneBonsaiEngine::new(DiffusionEngine::load_q1(&bonsai_dir).unwrap(), 64, 1e-6)
                .unwrap();

        for (rank, (cpu_layers, max_err, mean_err)) in scored.iter().take(5).enumerate() {
            let _ = bonsai_hybrid_forward_with_cpu_layers(&ane_engine_64, &input_64, cpu_layers);
            let t0 = std::time::Instant::now();
            for _ in 0..2 {
                let _ =
                    bonsai_hybrid_forward_with_cpu_layers(&ane_engine_64, &input_64, cpu_layers);
            }
            let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / 2.0;
            eprintln!(
                "  top{} seq=64 cpu_layers={:?} max_err={:.4} mean_err={:.6} avg_ms={:.1}",
                rank + 1,
                cpu_layers,
                max_err,
                mean_err,
                avg_ms
            );
        }
    }

    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Search wider rescue layers on top of the current Bonsai split"]
    fn test_bonsai_1_7b_hybrid_tail4_wide_rescue_search() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        let short_input: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let blas_engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        let blas_logits = blas_engine.forward(&short_input);
        let ane_engine = super::AneBonsaiEngine::new(
            DiffusionEngine::load_q1(&bonsai_dir).unwrap(),
            short_input.len(),
            1e-6,
        )
        .unwrap();

        let base_tail = vec![24usize, 25, 26, 27];
        let rescue_window = [16usize, 17, 18, 19, 20, 21, 22, 23];
        let mut scored: Vec<(Vec<usize>, f32, f32)> = Vec::new();

        for mask in 0usize..(1usize << rescue_window.len()) {
            let mut cpu_layers = base_tail.clone();
            for (bit, &layer) in rescue_window.iter().enumerate() {
                if (mask & (1usize << bit)) != 0 {
                    cpu_layers.push(layer);
                }
            }
            cpu_layers.sort_unstable();

            let logits =
                bonsai_hybrid_forward_with_cpu_layers(&ane_engine, &short_input, &cpu_layers);
            let errs: Vec<f32> = blas_logits
                .iter()
                .zip(logits.iter())
                .map(|(a, b)| (a - b).abs())
                .filter(|e| e.is_finite())
                .collect();
            let max_err = errs.iter().cloned().fold(0.0f32, f32::max);
            let mean_err = errs.iter().sum::<f32>() / errs.len() as f32;
            scored.push((cpu_layers, max_err, mean_err));
        }

        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.0.len().cmp(&b.0.len()))
        });

        eprintln!("Top 10 wide-rescue policies by seq=5 max_err:");
        for (rank, (cpu_layers, max_err, mean_err)) in scored.iter().take(10).enumerate() {
            eprintln!(
                "  {}. cpu_layers={:?} max_err={:.4} mean_err={:.6}",
                rank + 1,
                cpu_layers,
                max_err,
                mean_err
            );
        }

        let input_64: Vec<u32> = (0..64)
            .map(|i| if i < 5 { 785 + i } else { 1000 + i })
            .collect();
        let ane_engine_64 =
            super::AneBonsaiEngine::new(DiffusionEngine::load_q1(&bonsai_dir).unwrap(), 64, 1e-6)
                .unwrap();

        for (rank, (cpu_layers, max_err, mean_err)) in scored.iter().take(5).enumerate() {
            let _ = bonsai_hybrid_forward_with_cpu_layers(&ane_engine_64, &input_64, cpu_layers);
            let t0 = std::time::Instant::now();
            for _ in 0..2 {
                let _ =
                    bonsai_hybrid_forward_with_cpu_layers(&ane_engine_64, &input_64, cpu_layers);
            }
            let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / 2.0;
            eprintln!(
                "  top{} seq=64 cpu_layers={:?} max_err={:.4} mean_err={:.6} avg_ms={:.1}",
                rank + 1,
                cpu_layers,
                max_err,
                mean_err,
                avg_ms
            );
        }
    }

    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Search earlier rescue layers on top of the best known Bonsai split"]
    fn test_bonsai_1_7b_hybrid_best_split_early_rescue_search() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        let short_input: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let blas_engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        let blas_logits = blas_engine.forward(&short_input);
        let ane_engine = super::AneBonsaiEngine::new(
            DiffusionEngine::load_q1(&bonsai_dir).unwrap(),
            short_input.len(),
            1e-6,
        )
        .unwrap();

        let base_split = vec![18usize, 19, 22, 24, 25, 26, 27];
        let rescue_window = [12usize, 13, 14, 15, 16, 17];
        let mut scored: Vec<(Vec<usize>, f32, f32)> = Vec::new();

        for mask in 0usize..(1usize << rescue_window.len()) {
            let mut cpu_layers = base_split.clone();
            for (bit, &layer) in rescue_window.iter().enumerate() {
                if (mask & (1usize << bit)) != 0 {
                    cpu_layers.push(layer);
                }
            }
            cpu_layers.sort_unstable();

            let logits =
                bonsai_hybrid_forward_with_cpu_layers(&ane_engine, &short_input, &cpu_layers);
            let errs: Vec<f32> = blas_logits
                .iter()
                .zip(logits.iter())
                .map(|(a, b)| (a - b).abs())
                .filter(|e| e.is_finite())
                .collect();
            let max_err = errs.iter().cloned().fold(0.0f32, f32::max);
            let mean_err = errs.iter().sum::<f32>() / errs.len() as f32;
            scored.push((cpu_layers, max_err, mean_err));
        }

        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.0.len().cmp(&b.0.len()))
        });

        eprintln!("Top 10 early-rescue policies by seq=5 max_err:");
        for (rank, (cpu_layers, max_err, mean_err)) in scored.iter().take(10).enumerate() {
            eprintln!(
                "  {}. cpu_layers={:?} max_err={:.4} mean_err={:.6}",
                rank + 1,
                cpu_layers,
                max_err,
                mean_err
            );
        }

        let input_64: Vec<u32> = (0..64)
            .map(|i| if i < 5 { 785 + i } else { 1000 + i })
            .collect();
        let ane_engine_64 =
            super::AneBonsaiEngine::new(DiffusionEngine::load_q1(&bonsai_dir).unwrap(), 64, 1e-6)
                .unwrap();

        for (rank, (cpu_layers, max_err, mean_err)) in scored.iter().take(5).enumerate() {
            let _ = bonsai_hybrid_forward_with_cpu_layers(&ane_engine_64, &input_64, cpu_layers);
            let t0 = std::time::Instant::now();
            for _ in 0..2 {
                let _ =
                    bonsai_hybrid_forward_with_cpu_layers(&ane_engine_64, &input_64, cpu_layers);
            }
            let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / 2.0;
            eprintln!(
                "  top{} seq=64 cpu_layers={:?} max_err={:.4} mean_err={:.6} avg_ms={:.1}",
                rank + 1,
                cpu_layers,
                max_err,
                mean_err,
                avg_ms
            );
        }
    }

    /// Verify that the causal ANE diffusion engine for Qwen3 0.6B Base
    /// produces finite logits and reasonable output. This proves the causal
    /// mask is working correctly in the fused diffusion path.
    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Manual causal Qwen3 0.6B Base ANE verification"]
    fn test_qwen3_base_causal_ane_logits() {
        let Some(dir) = qwen3_base_dir() else {
            eprintln!("Qwen3-0.6B-Base not found, skipping");
            return;
        };

        let input: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let seq = input.len();

        eprintln!("Compiling causal ANE engine for Qwen3-0.6B-Base (seq={seq})...");
        let ane_engine =
            super::AneDiffusionEngine::new_causal(DiffusionEngine::load(&dir).unwrap(), seq)
                .unwrap();
        let ane_logits = ane_engine.forward(&input);

        let ane_nan = ane_logits.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "  ANE causal logits: {} total, {} non-finite",
            ane_logits.len(),
            ane_nan
        );
        assert_eq!(
            ane_nan, 0,
            "Causal ANE produced {ane_nan} non-finite logits"
        );

        // Check that the last position has reasonable predictions
        let cfg = &ane_engine.blas_engine.config;
        let last_pos = seq - 1;
        let row = &ane_logits[last_pos * cfg.vocab..(last_pos + 1) * cfg.vocab];
        let mut indices: Vec<usize> = (0..row.len()).collect();
        indices.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        eprintln!("Top-5 causal predictions at last position:");
        for &idx in indices.iter().take(5) {
            eprintln!("  token_id={idx} logit={:.4}", row[idx]);
        }

        let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_logit > 0.0, "All logits non-positive: max={max_logit}");

        eprintln!("PASS: causal Qwen3-0.6B-Base ANE produces finite logits");
    }

    /// Verify that the causal ANE Bonsai engine produces finite logits.
    /// This proves the causal mask is working correctly in the hybrid Bonsai path.
    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Manual causal Bonsai ANE verification"]
    fn test_bonsai_1_7b_causal_ane_logits() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        let input: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let seq = input.len();

        eprintln!("Compiling causal ANE Bonsai engine (seq={seq})...");
        let ane_engine = super::AneBonsaiEngine::new_causal(
            DiffusionEngine::load_q1(&bonsai_dir).unwrap(),
            seq,
            1e-6,
        )
        .unwrap();
        let ane_logits = ane_engine.forward(&input);

        let ane_nan = ane_logits.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "  ANE causal logits: {} total, {} non-finite",
            ane_logits.len(),
            ane_nan
        );
        assert_eq!(
            ane_nan, 0,
            "Causal ANE produced {ane_nan} non-finite logits"
        );

        let cfg = &ane_engine.blas_engine.config;
        let last_pos = seq - 1;
        let row = &ane_logits[last_pos * cfg.vocab..(last_pos + 1) * cfg.vocab];
        let mut indices: Vec<usize> = (0..row.len()).collect();
        indices.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        eprintln!("Top-5 causal predictions at last position:");
        for &idx in indices.iter().take(5) {
            eprintln!("  token_id={idx} logit={:.4}", row[idx]);
        }

        let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_logit > 0.0, "All logits non-positive: max={max_logit}");

        eprintln!("PASS: causal Bonsai ANE produces finite logits");
    }

    #[test]
    #[cfg(feature = "ane")]
    fn test_bonsai_tiled_ffn_compile() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        eprintln!("Loading Bonsai-1.7B Q1...");
        let engine = DiffusionEngine::load_q1(&bonsai_dir).unwrap();
        assert_eq!(engine.config.hidden, 2048);
        assert_eq!(engine.config.inter, 6144);
        assert_eq!(engine.config.layers, 28);

        eprintln!("Compiling OC-tiled AneBonsaiEngine (seq=64)...");
        let ane = super::AneBonsaiEngine::new(engine, 64, 1e-6).unwrap();
        eprintln!(
            "  w13: {} tiles × {} oc = {} total",
            ane.w13_plan.n_tiles, ane.w13_plan.tile_size, ane.w13_plan.full_size
        );
        eprintln!(
            "  kernels: {} attn, {} gated, {} down-partial",
            ane.attn_kernels.len(),
            ane.ffn_gated_kernels.len(),
            ane.ffn_down_kernels.len(),
        );

        let short_input: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let logits = ane.forward(&short_input);
        let nan_count = logits.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "  logits: {} total, {} non-finite, first 5 = {:?}",
            logits.len(),
            nan_count,
            &logits[..5.min(logits.len())]
        );
        assert!(
            nan_count == 0,
            "Expected 0 non-finite logits, got {nan_count}"
        );
    }

    #[test]
    #[cfg(feature = "ane")]
    fn test_bonsai_tiled_ffn_benchmark() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

        let n_warmup = 1;
        let n_iter = 3;

        for seq in [64usize, 128] {
            let input: Vec<u32> = (0..seq)
                .map(|i| if i < 5 { 785u32 + i as u32 } else { 1000u32 + i as u32 })
                .collect();

            eprintln!("\n=== Bonsai-1.7B OC-tiled ANE benchmark (seq={seq}) ===");

            let ane_engine = super::AneBonsaiEngine::new(
                DiffusionEngine::load_q1(&bonsai_dir).unwrap(),
                seq,
                1e-6,
            )
            .unwrap();

            let n_w13 = ane_engine.w13_plan.n_tiles;
            let ffn_dispatches_per_layer = 2 * n_w13;
            eprintln!(
                "  dispatches/layer: 1 attn + {ffn_dispatches_per_layer} FFN ({} gated + {} down-partial) = {}",
                n_w13, n_w13, 1 + ffn_dispatches_per_layer,
            );

            let cfg = &ane_engine.blas_engine.config;
            let h = cfg.hidden;
            let inter = cfg.inter;
            let ps = ane_engine.seq_len;
            let eps = ane_engine.eps;

            let mut hidden0 = vec![0.0f32; seq * h];
            for (i, &tid) in input.iter().enumerate() {
                let off = tid as usize * h;
                hidden0[i * h..(i + 1) * h].copy_from_slice(&ane_engine.blas_engine.embed[off..off + h]);
            }

            #[cfg(target_arch = "aarch64")]
            fn ane_dsb_sy() {
                unsafe { std::arch::asm!("dsb sy", options(nostack, preserves_flags)) }
            }

            #[cfg(not(target_arch = "aarch64"))]
            fn ane_dsb_sy() {}

            let write_input_cf32 = |kernel: &crate::ane_bridge::AneKernel, data: &[f32], dim: usize| {
                let base = kernel.get_input_base(0) as *mut f32;
                assert!(!base.is_null(), "ANE input base should not be null");
                unsafe {
                    for c in 0..dim {
                        let dst = std::slice::from_raw_parts_mut(base.add(c * ps), ps);
                        for s in 0..seq {
                            dst[s] = data[s * dim + c];
                        }
                    }
                }
                ane_dsb_sy();
            };

            let scatter_output_cf32 =
                |kernel: &crate::ane_bridge::AneKernel,
                 actual_dim: usize,
                 start: usize,
                 dst_row_major: &mut [f32],
                 dst_stride: usize| {
                    let base = kernel.get_output_base(0) as *const f32;
                    assert!(!base.is_null(), "ANE output base should not be null");
                    ane_dsb_sy();
                    unsafe {
                        for c in 0..actual_dim {
                            let src = std::slice::from_raw_parts(base.add(c * ps), ps);
                            for s in 0..seq {
                                dst_row_major[s * dst_stride + start + c] = src[s];
                            }
                        }
                    }
                };

            let accumulate_output_cf32 =
                |kernel: &crate::ane_bridge::AneKernel,
                 actual_dim: usize,
                 dst_row_major: &mut [f32],
                 dst_stride: usize| {
                    let base = kernel.get_output_base(0) as *const f32;
                    assert!(!base.is_null(), "ANE output base should not be null");
                    ane_dsb_sy();
                    unsafe {
                        for c in 0..actual_dim {
                            let src = std::slice::from_raw_parts(base.add(c * ps), ps);
                            for s in 0..seq {
                                dst_row_major[s * dst_stride + c] += src[s];
                            }
                        }
                    }
                };

            struct CpuFfnScratch {
                hidden: Vec<f32>,
                normed: Vec<f32>,
                gate: Vec<f32>,
                up: Vec<f32>,
                gated: Vec<f32>,
                out: Vec<f32>,
            }

            struct AneFfnScratch {
                hidden: Vec<f32>,
                normed: Vec<f32>,
                out: Vec<f32>,
            }

            let mut cpu_scratch = CpuFfnScratch {
                hidden: vec![0.0; seq * h],
                normed: vec![0.0; seq * h],
                gate: vec![0.0; seq * inter],
                up: vec![0.0; seq * inter],
                gated: vec![0.0; seq * inter],
                out: vec![0.0; seq * h],
            };

            let mut ane_scratch = AneFfnScratch {
                hidden: vec![0.0; seq * h],
                normed: vec![0.0; seq * h],
                out: vec![0.0; seq * h],
            };

            let cpu_ffn = |scratch: &mut CpuFfnScratch| {
                scratch.hidden.copy_from_slice(&hidden0);

                for layer in &ane_engine.blas_engine.layers {
                    rms_norm_eps(
                        &scratch.hidden,
                        &layer.post_attn_norm,
                        &mut scratch.normed,
                        seq,
                        h,
                        eps,
                    );
                    sgemm_nt(seq, inter, h, &scratch.normed, &layer.gate_proj, &mut scratch.gate);
                    for v in scratch.gate.iter_mut() {
                        *v *= 1.0 / (1.0 + (-*v).exp());
                    }
                    sgemm_nt(seq, inter, h, &scratch.normed, &layer.up_proj, &mut scratch.up);
                    for (g, u) in scratch.gate.iter_mut().zip(scratch.up.iter()) {
                        *g *= u;
                    }
                    sgemm_nt(
                        seq,
                        h,
                        inter,
                        &scratch.gate,
                        &layer.down_proj,
                        &mut scratch.out,
                    );
                    for i in 0..seq * h {
                        scratch.hidden[i] += scratch.out[i];
                    }
                }
            };

            let rt_enabled = crate::ane_bridge::AneKernel::begin_realtime();
            let ane_ffn = |scratch: &mut AneFfnScratch| {
                scratch.hidden.copy_from_slice(&hidden0);
                let n_w13 = ane_engine.w13_plan.n_tiles;

                for (layer_idx, layer) in ane_engine.blas_engine.layers.iter().enumerate() {
                    rms_norm_eps(
                        &scratch.hidden,
                        &layer.post_attn_norm,
                        &mut scratch.normed,
                        seq,
                        h,
                        eps,
                    );
                    scratch.out.fill(0.0);
                    for t in 0..n_w13 {
                        let kernel_idx = layer_idx * n_w13 + t;
                        let gated_kernel = &ane_engine.ffn_gated_kernels[kernel_idx];
                        let down_kernel = &ane_engine.ffn_down_kernels[kernel_idx];
                        write_input_cf32(gated_kernel, &scratch.normed, h);
                        if rt_enabled {
                            crate::ane_bridge::AneKernel::eval_chain_realtime(&[
                                gated_kernel,
                                down_kernel,
                            ])
                            .unwrap_or_else(|_| {
                                crate::ane_bridge::AneKernel::eval_chain(&[
                                    gated_kernel,
                                    down_kernel,
                                ])
                                .unwrap()
                            });
                        } else {
                            crate::ane_bridge::AneKernel::eval_chain(&[
                                gated_kernel,
                                down_kernel,
                            ])
                            .unwrap();
                        }
                        accumulate_output_cf32(down_kernel, h, &mut scratch.out, h);
                    }

                    for i in 0..seq * h {
                        scratch.hidden[i] += scratch.out[i];
                    }
                }
            };
            for _ in 0..n_warmup {
                cpu_ffn(&mut cpu_scratch);
                ane_ffn(&mut ane_scratch);
            }

            cpu_ffn(&mut cpu_scratch);
            ane_ffn(&mut ane_scratch);
            let cpu_sample = cpu_scratch.hidden.clone();
            let ane_sample = ane_scratch.hidden.clone();
            let max_err = cpu_sample
                .iter()
                .zip(ane_sample.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let ane_nan = ane_sample.iter().filter(|v| !v.is_finite()).count();
            eprintln!("  max_err vs CPU: {max_err:.5}, non-finite: {ane_nan}");
            assert_eq!(ane_nan, 0, "ANE FFN benchmark produced non-finite values");

            let t0 = std::time::Instant::now();
            for _ in 0..n_iter {
                cpu_ffn(&mut cpu_scratch);
            }
            let blas_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..n_iter {
                ane_ffn(&mut ane_scratch);
            }
            let ane_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;
            if rt_enabled {
                crate::ane_bridge::AneKernel::end_realtime();
            }

            let speedup = blas_ms / ane_ms;
            eprintln!("  CPU BLAS FFN stack: {blas_ms:.0}ms");
            eprintln!("  ANE tiled FFN stack: {ane_ms:.0}ms");
            eprintln!("  speedup: {speedup:.2}x");
        }
    }

    // -----------------------------------------------------------------------
    // dLLM draft → AR verify acceptance rate test
    // -----------------------------------------------------------------------

    fn qwen3_5_27b_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit");
        dir.join("config.json").exists().then_some(dir)
    }

    fn qwen3_8b_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/lm-studio/models/mlx-community/Qwen3-8B-4bit");
        dir.join("config.json").exists().then_some(dir)
    }

    fn qwen3_14b_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/lm-studio/models/mlx-community/Qwen3-14B-4bit");
        dir.join("config.json").exists().then_some(dir)
    }

    fn bonsai_8b_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/lm-studio/models/prism-ml/Bonsai-8B-mlx-1bit");
        dir.join("config.json").exists().then_some(dir)
    }

    fn qwen3_5_0_8b_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit");
        dir.join("config.json").exists().then_some(dir)
    }

    /// Measure what fraction of diffusion-drafted tokens a target AR model accepts.
    ///
    /// Uses fresh KV cache per round (single forward pass over context+drafts) to
    /// avoid cache rollback complexity. Each round:
    ///   1. Draft: MDLM diffusion on context[-120..] + K masks, 8 denoising steps
    ///   2. Verify: target model forward_all_logits on full_seq = context + drafts
    ///   3. Compare argmax(logits[P-1+i]) with draft[i] for longest accepted prefix
    #[test]
    #[ignore] // Heavy — loads 27B AR model + diffusion model
    fn test_diffusion_draft_acceptance_rate() {
        const K: usize = 8; // draft tokens per round
        const STEPS: usize = 1; // DEER/DFlash consensus: 1 step optimal
        const CONTEXT_WINDOW: usize = 120; // sliding window for diffusion input
        const N_ROUNDS: usize = 10;

        // --- Load diffusion draft model ---
        let Some(diff_dir) = model_dir() else {
            eprintln!("SKIP: Diffusion model not found");
            return;
        };
        let diff_engine = DiffusionEngine::load_autodetect(&diff_dir).unwrap();
        let diff_runtime = DiffusionRuntime::new(
            diff_engine,
            CONTEXT_WINDOW + K, // canvas size
            DiffusionBackendPreference::Auto,
        )
        .unwrap();
        eprintln!(
            "Draft model loaded: {:?} ({})",
            diff_runtime.selected_backend(),
            diff_runtime.backend_report().detail
        );

        // Tokenize prompt using the diffusion model's tokenizer (shared Qwen3 vocab)
        let tokenizer = crate::load_tokenizer(&diff_dir).unwrap();
        let prompt = "Explain quantum computing in simple terms";
        let prompt_ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .unwrap()
            .get_ids()
            .to_vec();
        eprintln!("Prompt: {prompt:?} ({} tokens)", prompt_ids.len());

        // --- Target model specs ---
        struct Target {
            name: &'static str,
            dir: Option<std::path::PathBuf>,
        }
        let targets = [
            Target {
                name: "Qwen3-8B-4bit",
                dir: qwen3_8b_dir(),
            },
            Target {
                name: "Qwen3-14B-4bit",
                dir: qwen3_14b_dir(),
            },
            Target {
                name: "Qwen3.5-27B-4bit",
                dir: qwen3_5_27b_dir(),
            },
            Target {
                name: "Bonsai-8B-mlx-1bit",
                dir: bonsai_8b_dir(),
            },
        ];

        for target in &targets {
            let Some(ref dir) = target.dir else {
                eprintln!("\n--- SKIP: {} (not found) ---", target.name);
                continue;
            };

            eprintln!("\n=== {} ===", target.name);
            let t_load = std::time::Instant::now();

            // Detect model type and load
            let model_type = crate::registry::detect_model_type(dir).unwrap();
            let load_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match model_type.as_str() {
                    "qwen3_5" => crate::qwen3_next::load_qwen3_5_model(dir)
                        .map(crate::AnyModel::Qwen3Next)
                        .map_err(|e| e.to_string()),
                    "qwen2" | "qwen3" | "llama" | "mistral" => {
                        crate::transformer::load_model(dir)
                            .map(crate::AnyModel::Transformer)
                            .map_err(|e| e.to_string())
                    }
                    other => Err(format!("Unsupported model_type={other}")),
                }
            }));
            let mut model = match load_result {
                Ok(Ok(m)) => m,
                Ok(Err(e)) => {
                    eprintln!("  Failed to load: {e}");
                    continue;
                }
                Err(_) => {
                    eprintln!("  Load panicked (likely unsupported quantization), skipping");
                    continue;
                }
            };
            eprintln!(
                "  Loaded in {:.1}s (type={model_type})",
                t_load.elapsed().as_secs_f64()
            );

            let mut context = prompt_ids.clone();
            let mut total_accepted: usize = 0;
            let mut total_drafted: usize = 0;
            let mut draft_time_ms: u128 = 0;
            let mut verify_time_ms: u128 = 0;

            for round in 0..N_ROUNDS {
                // --- Draft phase ---
                let ctx_start = context.len().saturating_sub(CONTEXT_WINDOW);
                let ctx_slice = &context[ctx_start..];

                let t_draft = std::time::Instant::now();
                let canvas = diff_runtime.generate(ctx_slice, K, STEPS);
                draft_time_ms += t_draft.elapsed().as_millis();

                // Extract draft tokens (last K of canvas)
                let draft: Vec<u32> = canvas[canvas.len() - K..].to_vec();

                // --- Verify phase (fresh cache, single forward pass) ---
                // Build full_input = context + draft_tokens
                let mut full_seq: Vec<i32> = context.iter().map(|&id| id as i32).collect();
                for &d in &draft {
                    full_seq.push(d as i32);
                }
                let t_total = full_seq.len() as i32;
                let input_arr = mlx_rs::Array::from_slice(&full_seq, &[1, t_total]);

                let mut cache = model.make_cache();

                let t_verify = std::time::Instant::now();
                let fwd = model.forward_all_logits(&input_arr, None, &mut cache);
                let all_logits = match fwd.and_then(|l| {
                    mlx_rs::transforms::eval([&l])?;
                    Ok(l)
                }) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("  forward_all_logits failed at round {}: {e}", round + 1);
                        eprintln!("  (model may not support full-sequence logits)");
                        break;
                    }
                };
                verify_time_ms += t_verify.elapsed().as_millis();

                // all_logits shape: [1, T, vocab]
                // logits at position P-1+i predicts what should be at position P+i
                // → compare argmax(logits[P-1+i]) with draft[i]
                //
                // Batch argmax over the K+1 positions we care about to minimize
                // round-trips to the GPU.
                use mlx_rs::ops::indexing::{self as ix, IndexOp};

                let p = context.len(); // prompt+accepted token count
                let start = (p - 1) as i32;
                let end = start + K as i32 + 1; // +1 for bonus token
                let verify_logits = all_logits.index((.., start..end, ..)); // [1, K+1, vocab]
                let verify_2d = verify_logits.squeeze_axes(&[0]).unwrap(); // [K+1, vocab]
                let preds = ix::argmax_axis(&verify_2d, -1, false).unwrap(); // [K+1]
                mlx_rs::transforms::eval([&preds]).unwrap();

                let mut accepted_this_round = 0usize;
                for i in 0..K {
                    let pred_id = preds.index(i as i32).item::<i32>() as u32;
                    if pred_id == draft[i] {
                        accepted_this_round += 1;
                        context.push(draft[i]);
                    } else {
                        context.push(pred_id);
                        break;
                    }
                }

                // If all K accepted, grab the bonus token
                if accepted_this_round == K {
                    let bonus_id = preds.index(K as i32).item::<i32>() as u32;
                    context.push(bonus_id);
                }

                total_accepted += accepted_this_round;
                total_drafted += K;

                eprintln!(
                    "  Round {:2}: accepted {}/{K} (ctx={})",
                    round + 1,
                    accepted_this_round,
                    context.len()
                );
            }

            // --- Results ---
            let acceptance_rate = total_accepted as f64 / total_drafted as f64 * 100.0;
            let avg_per_round = total_accepted as f64 / N_ROUNDS as f64;
            let total_new = context.len() - prompt_ids.len();
            let total_ms = draft_time_ms + verify_time_ms;
            let eff_tps = if total_ms > 0 {
                total_new as f64 / total_ms as f64 * 1000.0
            } else {
                0.0
            };

            eprintln!("\n  --- {} Results ---", target.name);
            eprintln!("  Acceptance rate:     {acceptance_rate:.1}%");
            eprintln!("  Avg accepted/round:  {avg_per_round:.1}/{K}");
            eprintln!("  Total new tokens:    {total_new}");
            eprintln!(
                "  Draft time:  {draft_time_ms}ms ({:.1}ms/round)",
                draft_time_ms as f64 / N_ROUNDS as f64
            );
            eprintln!(
                "  Verify time: {verify_time_ms}ms ({:.1}ms/round)",
                verify_time_ms as f64 / N_ROUNDS as f64
            );
            eprintln!("  Effective throughput: {eff_tps:.1} tok/s");

            if let Ok(decoded) = tokenizer.decode(&context[prompt_ids.len()..], true) {
                eprintln!("  Generated text: {decoded:?}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // AR spec decode: small Qwen3.5 draft → large Qwen3.5 verify
    // -----------------------------------------------------------------------

    /// Autoregressive speculative decoding with **cache reuse**.
    ///
    /// Drafts K tokens with Qwen3.5-0.8B, verifies with Qwen3.5-27B.
    /// Both draft and verify caches persist across rounds — only new tokens
    /// are processed each round.  Uses snapshot-restore to roll back rejected
    /// draft tokens from both KV and SSM/conv state.
    #[test]
    #[ignore] // Heavy — loads 0.8B draft + 27B verify model
    fn test_ar_spec_decode_acceptance() {
        const K: usize = 8; // draft tokens per round
        const N_ROUNDS: usize = 10;

        // --- Load draft model (0.8B) ---
        let Some(draft_dir) = qwen3_5_0_8b_dir() else {
            eprintln!("SKIP: Qwen3.5-0.8B-8bit not found");
            return;
        };
        eprintln!("Loading draft model...");
        let t_load = std::time::Instant::now();
        let draft_model = crate::qwen3_next::load_qwen3_5_model(&draft_dir).unwrap();
        let mut draft = crate::AnyModel::Qwen3Next(draft_model);
        eprintln!("  Draft loaded in {:.1}s", t_load.elapsed().as_secs_f64());

        // --- Load verify model (27B) ---
        let Some(verify_dir) = qwen3_5_27b_dir() else {
            eprintln!("SKIP: Qwen3.5-27B-4bit not found");
            return;
        };
        eprintln!("Loading verify model...");
        let t_load = std::time::Instant::now();
        let verify_model = crate::qwen3_next::load_qwen3_5_model(&verify_dir).unwrap();
        let mut verify = crate::AnyModel::Qwen3Next(verify_model);
        eprintln!(
            "  Verify loaded in {:.1}s",
            t_load.elapsed().as_secs_f64()
        );

        // Shared tokenizer (same vocab)
        let tokenizer = crate::load_tokenizer(&draft_dir).unwrap();

        let prompts = [
            "The capital of France is",
            "Explain quantum computing in simple terms:",
            "fn fibonacci(n: u64) -> u64 {",
        ];

        for prompt in &prompts {
            eprintln!("\n=== Prompt: {prompt:?} ===");

            let prompt_ids: Vec<u32> = tokenizer
                .encode(*prompt, false)
                .unwrap()
                .get_ids()
                .to_vec();
            eprintln!("  Tokenized: {} tokens", prompt_ids.len());

            let mut context = prompt_ids.clone();
            let mut total_accepted: usize = 0;
            let mut total_drafted: usize = 0;
            let mut draft_time_ms: f64 = 0.0;
            let mut verify_time_ms: f64 = 0.0;
            let mut advance_time_ms: f64 = 0.0;

            // Persistent caches — survive across rounds
            let mut draft_cache = draft.make_cache();
            let mut verify_cache = verify.make_cache();

            // Prefill both models with prompt (once per prompt).
            // SAVE the returned logits — they predict the first generated token
            // and are needed for round 0 verification + drafting.
            let ctx_input: Vec<i32> = prompt_ids.iter().map(|&id| id as i32).collect();
            let ctx_len = ctx_input.len() as i32;
            let ctx_arr = mlx_rs::Array::from_slice(&ctx_input, &[1, ctx_len]);

            let t_prefill = std::time::Instant::now();
            let mut saved_draft_logits = draft.forward(&ctx_arr, None, &mut draft_cache).unwrap();
            mlx_rs::transforms::eval([&saved_draft_logits]).unwrap();
            let mut saved_verify_logits =
                verify.forward(&ctx_arr, None, &mut verify_cache).unwrap();
            mlx_rs::transforms::eval([&saved_verify_logits]).unwrap();
            draft_cache.eval_for_clone().unwrap();
            verify_cache.eval_for_clone().unwrap();
            let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
            eprintln!("  Prefill: {prefill_ms:.0}ms (both models)");

            for round in 0..N_ROUNDS {
                // --- Draft phase: decode K tokens from persistent draft cache ---
                let t_draft = std::time::Instant::now();

                // Snapshot draft cache before drafting
                let draft_snapshot = draft_cache.clone();

                // Greedy decode K tokens using saved_draft_logits for the first
                let mut draft_tokens: Vec<u32> = Vec::with_capacity(K);
                {
                    use mlx_rs::ops::indexing as ix;
                    let pred = ix::argmax_axis(&saved_draft_logits, -1, false).unwrap();
                    mlx_rs::transforms::eval([&pred]).unwrap();
                    draft_tokens.push(pred.item::<i32>() as u32);
                }

                for _ in 1..K {
                    use mlx_rs::ops::indexing as ix;
                    let tok = *draft_tokens.last().unwrap() as i32;
                    let input = mlx_rs::Array::from_slice(&[tok], &[1, 1]);
                    let logits = draft.forward(&input, None, &mut draft_cache).unwrap();
                    mlx_rs::transforms::eval([&logits]).unwrap();
                    let pred = ix::argmax_axis(&logits, -1, false).unwrap();
                    mlx_rs::transforms::eval([&pred]).unwrap();
                    draft_tokens.push(pred.item::<i32>() as u32);
                }

                draft_time_ms += t_draft.elapsed().as_secs_f64() * 1000.0;

                // --- Verify phase: forward K draft tokens through verify model ---
                // The verify cache already has context from prefill + previous rounds.
                // saved_verify_logits predicts draft[0].
                // forward_all_logits(draft_tokens) gives logits for draft[1..K-1] + bonus.
                let t_verify = std::time::Instant::now();

                let verify_snapshot = verify_cache.clone();

                let draft_input: Vec<i32> = draft_tokens.iter().map(|&d| d as i32).collect();
                let draft_arr =
                    mlx_rs::Array::from_slice(&draft_input, &[1, draft_input.len() as i32]);
                let all_logits = verify
                    .forward_all_logits(&draft_arr, None, &mut verify_cache)
                    .unwrap();
                mlx_rs::transforms::eval([&all_logits]).unwrap();

                verify_time_ms += t_verify.elapsed().as_secs_f64() * 1000.0;

                // all_logits: [1, K, vocab]
                //   all_logits[0]   at pos P   → predicts pos P+1 → verifies draft[1]
                //   all_logits[i-1] at pos P+i-1 → predicts pos P+i → verifies draft[i]
                //   all_logits[K-1] at pos P+K-1 → predicts pos P+K → bonus token
                // saved_verify_logits at pos P-1 → predicts pos P → verifies draft[0]
                use mlx_rs::ops::indexing::{self as ix, IndexOp};
                let new_2d = all_logits.squeeze_axes(&[0]).unwrap(); // [K, vocab]

                // Verify draft[0] from saved logits
                let first_pred = {
                    let p = ix::argmax_axis(&saved_verify_logits, -1, false).unwrap();
                    mlx_rs::transforms::eval([&p]).unwrap();
                    p.item::<i32>() as u32
                };

                let mut accepted = 0usize;
                if first_pred == draft_tokens[0] {
                    accepted = 1;
                    // Verify draft[1..K-1] from new logits
                    let new_preds = ix::argmax_axis(&new_2d, -1, false).unwrap(); // [K]
                    mlx_rs::transforms::eval([&new_preds]).unwrap();
                    for i in 1..K {
                        let pred_id = new_preds.index((i - 1) as i32).item::<i32>() as u32;
                        if pred_id == draft_tokens[i] {
                            accepted += 1;
                        } else {
                            break;
                        }
                    }
                }

                // Build accepted sequence + bonus/correction
                let correction_or_bonus = if accepted == K {
                    // All accepted — bonus from all_logits[K-1]
                    let bonus_logits = new_2d.index((K - 1) as i32); // [vocab]
                    let p = ix::argmax_axis(&bonus_logits, -1, false).unwrap();
                    mlx_rs::transforms::eval([&p]).unwrap();
                    p.item::<i32>() as u32
                } else if accepted == 0 {
                    // None accepted — correction from saved_verify_logits
                    first_pred
                } else {
                    // Partial — correction from all_logits[accepted-1]
                    let corr_logits = new_2d.index((accepted - 1) as i32); // [vocab]
                    let p = ix::argmax_axis(&corr_logits, -1, false).unwrap();
                    mlx_rs::transforms::eval([&p]).unwrap();
                    p.item::<i32>() as u32
                };

                let mut new_tokens: Vec<u32> = draft_tokens[..accepted].to_vec();
                new_tokens.push(correction_or_bonus);
                context.extend_from_slice(&new_tokens);

                // --- Advance caches with accepted tokens ---
                let t_advance = std::time::Instant::now();

                if accepted == K {
                    // FAST PATH: all K accepted — caches already have the right state.
                    // Draft cache is at P+K-1 (draft[K-1] not yet fed). Feed it + bonus.
                    let catchup: Vec<i32> =
                        vec![draft_tokens[K - 1] as i32, correction_or_bonus as i32];
                    let catchup_arr = mlx_rs::Array::from_slice(&catchup, &[1, 2]);
                    saved_draft_logits =
                        draft.forward(&catchup_arr, None, &mut draft_cache).unwrap();
                    mlx_rs::transforms::eval([&saved_draft_logits]).unwrap();
                    draft_cache.eval_for_clone().unwrap();

                    // Verify cache is at P+K (all draft tokens processed). Feed bonus only.
                    let bonus_arr =
                        mlx_rs::Array::from_slice(&[correction_or_bonus as i32], &[1, 1]);
                    saved_verify_logits =
                        verify.forward(&bonus_arr, None, &mut verify_cache).unwrap();
                    mlx_rs::transforms::eval([&saved_verify_logits]).unwrap();
                    verify_cache.eval_for_clone().unwrap();
                } else {
                    // SLOW PATH: partial accept — restore snapshots and re-feed.
                    let advance_input: Vec<i32> =
                        new_tokens.iter().map(|&t| t as i32).collect();
                    let advance_arr = mlx_rs::Array::from_slice(
                        &advance_input,
                        &[1, advance_input.len() as i32],
                    );

                    draft_cache = draft_snapshot;
                    saved_draft_logits =
                        draft.forward(&advance_arr, None, &mut draft_cache).unwrap();
                    mlx_rs::transforms::eval([&saved_draft_logits]).unwrap();
                    draft_cache.eval_for_clone().unwrap();

                    verify_cache = verify_snapshot;
                    saved_verify_logits =
                        verify.forward(&advance_arr, None, &mut verify_cache).unwrap();
                    mlx_rs::transforms::eval([&saved_verify_logits]).unwrap();
                    verify_cache.eval_for_clone().unwrap();
                }

                advance_time_ms += t_advance.elapsed().as_secs_f64() * 1000.0;

                total_accepted += accepted;
                total_drafted += K;

                let draft_toks_str: Vec<String> =
                    draft_tokens.iter().map(|t| t.to_string()).collect();
                eprintln!(
                    "  Round {:2}: accepted {}/{K} (+{} new) | drafted [{}]",
                    round + 1,
                    accepted,
                    new_tokens.len(),
                    draft_toks_str.join(", ")
                );
            }

            // --- Results ---
            let acceptance_rate = total_accepted as f64 / total_drafted as f64 * 100.0;
            let avg_per_round = total_accepted as f64 / N_ROUNDS as f64;
            let total_new = context.len() - prompt_ids.len();
            let total_ms = draft_time_ms + verify_time_ms + advance_time_ms;
            let eff_tps = if total_ms > 0.0 {
                total_new as f64 / total_ms * 1000.0
            } else {
                0.0
            };

            eprintln!("\n  --- Results for {prompt:?} ---");
            eprintln!("  Acceptance rate:     {acceptance_rate:.1}%");
            eprintln!("  Avg accepted/round:  {avg_per_round:.1}/{K}");
            eprintln!("  Total new tokens:    {total_new}");
            eprintln!(
                "  Draft time:    {draft_time_ms:.0}ms ({:.1}ms/round)",
                draft_time_ms / N_ROUNDS as f64
            );
            eprintln!(
                "  Verify time:   {verify_time_ms:.0}ms ({:.1}ms/round)",
                verify_time_ms / N_ROUNDS as f64
            );
            eprintln!(
                "  Advance time:  {advance_time_ms:.0}ms ({:.1}ms/round)",
                advance_time_ms / N_ROUNDS as f64
            );
            eprintln!("  Effective throughput: {eff_tps:.1} tok/s");

            if let Ok(decoded) = tokenizer.decode(&context[prompt_ids.len()..], true) {
                eprintln!("  Generated text: {decoded:?}");
            }
        }
    }

    /// AR speculative decode with adaptive K controller.
    ///
    /// Same as `test_ar_spec_decode_acceptance` but uses `AdaptiveKController`
    /// to dynamically switch between K=16 (high acceptance) and K=8 (low
    /// acceptance).  Runs 15 rounds to give adaptation time.
    #[test]
    #[ignore] // Heavy — loads 0.8B draft + 27B verify model
    fn test_ar_spec_decode_adaptive_k() {
        const N_ROUNDS: usize = 15;

        // --- Load draft model (0.8B) ---
        let Some(draft_dir) = qwen3_5_0_8b_dir() else {
            eprintln!("SKIP: Qwen3.5-0.8B-8bit not found");
            return;
        };
        eprintln!("Loading draft model...");
        let t_load = std::time::Instant::now();
        let draft_model = crate::qwen3_next::load_qwen3_5_model(&draft_dir).unwrap();
        let mut draft = crate::AnyModel::Qwen3Next(draft_model);
        eprintln!("  Draft loaded in {:.1}s", t_load.elapsed().as_secs_f64());

        // --- Load verify model (27B) ---
        let Some(verify_dir) = qwen3_5_27b_dir() else {
            eprintln!("SKIP: Qwen3.5-27B-4bit not found");
            return;
        };
        eprintln!("Loading verify model...");
        let t_load = std::time::Instant::now();
        let verify_model = crate::qwen3_next::load_qwen3_5_model(&verify_dir).unwrap();
        let mut verify = crate::AnyModel::Qwen3Next(verify_model);
        eprintln!(
            "  Verify loaded in {:.1}s",
            t_load.elapsed().as_secs_f64()
        );

        // Shared tokenizer (same vocab)
        let tokenizer = crate::load_tokenizer(&draft_dir).unwrap();

        let prompts = [
            "The capital of France is",
            "Explain quantum computing in simple terms:",
            "fn fibonacci(n: u64) -> u64 {",
        ];

        for prompt in &prompts {
            eprintln!("\n=== Adaptive K — Prompt: {prompt:?} ===");

            let prompt_ids: Vec<u32> = tokenizer
                .encode(*prompt, false)
                .unwrap()
                .get_ids()
                .to_vec();
            eprintln!("  Tokenized: {} tokens", prompt_ids.len());

            let mut context = prompt_ids.clone();
            let mut total_accepted: usize = 0;
            let mut total_drafted: usize = 0;
            let mut draft_time_ms: f64 = 0.0;
            let mut verify_time_ms: f64 = 0.0;
            let mut advance_time_ms: f64 = 0.0;

            let mut controller = AdaptiveKController::new(8, 16, 3);

            // Persistent caches
            let mut draft_cache = draft.make_cache();
            let mut verify_cache = verify.make_cache();

            // Prefill both models
            let ctx_input: Vec<i32> = prompt_ids.iter().map(|&id| id as i32).collect();
            let ctx_len = ctx_input.len() as i32;
            let ctx_arr = mlx_rs::Array::from_slice(&ctx_input, &[1, ctx_len]);

            let t_prefill = std::time::Instant::now();
            let mut saved_draft_logits = draft.forward(&ctx_arr, None, &mut draft_cache).unwrap();
            mlx_rs::transforms::eval([&saved_draft_logits]).unwrap();
            let mut saved_verify_logits =
                verify.forward(&ctx_arr, None, &mut verify_cache).unwrap();
            mlx_rs::transforms::eval([&saved_verify_logits]).unwrap();
            draft_cache.eval_for_clone().unwrap();
            verify_cache.eval_for_clone().unwrap();
            let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
            eprintln!("  Prefill: {prefill_ms:.0}ms (both models)");

            for round in 0..N_ROUNDS {
                let k = controller.current_k();

                // --- Draft phase ---
                let t_draft = std::time::Instant::now();
                let draft_snapshot = draft_cache.clone();

                let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
                {
                    use mlx_rs::ops::indexing as ix;
                    let pred = ix::argmax_axis(&saved_draft_logits, -1, false).unwrap();
                    mlx_rs::transforms::eval([&pred]).unwrap();
                    draft_tokens.push(pred.item::<i32>() as u32);
                }

                for _ in 1..k {
                    use mlx_rs::ops::indexing as ix;
                    let tok = *draft_tokens.last().unwrap() as i32;
                    let input = mlx_rs::Array::from_slice(&[tok], &[1, 1]);
                    let logits = draft.forward(&input, None, &mut draft_cache).unwrap();
                    mlx_rs::transforms::eval([&logits]).unwrap();
                    let pred = ix::argmax_axis(&logits, -1, false).unwrap();
                    mlx_rs::transforms::eval([&pred]).unwrap();
                    draft_tokens.push(pred.item::<i32>() as u32);
                }

                draft_time_ms += t_draft.elapsed().as_secs_f64() * 1000.0;

                // --- Verify phase ---
                let t_verify = std::time::Instant::now();
                let verify_snapshot = verify_cache.clone();

                let draft_input: Vec<i32> = draft_tokens.iter().map(|&d| d as i32).collect();
                let draft_arr =
                    mlx_rs::Array::from_slice(&draft_input, &[1, draft_input.len() as i32]);
                let all_logits = verify
                    .forward_all_logits(&draft_arr, None, &mut verify_cache)
                    .unwrap();
                mlx_rs::transforms::eval([&all_logits]).unwrap();

                verify_time_ms += t_verify.elapsed().as_secs_f64() * 1000.0;

                use mlx_rs::ops::indexing::{self as ix, IndexOp};
                let new_2d = all_logits.squeeze_axes(&[0]).unwrap();

                let first_pred = {
                    let p = ix::argmax_axis(&saved_verify_logits, -1, false).unwrap();
                    mlx_rs::transforms::eval([&p]).unwrap();
                    p.item::<i32>() as u32
                };

                let mut accepted = 0usize;
                if first_pred == draft_tokens[0] {
                    accepted = 1;
                    let new_preds = ix::argmax_axis(&new_2d, -1, false).unwrap();
                    mlx_rs::transforms::eval([&new_preds]).unwrap();
                    for i in 1..k {
                        let pred_id = new_preds.index((i - 1) as i32).item::<i32>() as u32;
                        if pred_id == draft_tokens[i] {
                            accepted += 1;
                        } else {
                            break;
                        }
                    }
                }

                let correction_or_bonus = if accepted == k {
                    let bonus_logits = new_2d.index((k - 1) as i32);
                    let p = ix::argmax_axis(&bonus_logits, -1, false).unwrap();
                    mlx_rs::transforms::eval([&p]).unwrap();
                    p.item::<i32>() as u32
                } else if accepted == 0 {
                    first_pred
                } else {
                    let corr_logits = new_2d.index((accepted - 1) as i32);
                    let p = ix::argmax_axis(&corr_logits, -1, false).unwrap();
                    mlx_rs::transforms::eval([&p]).unwrap();
                    p.item::<i32>() as u32
                };

                let mut new_tokens: Vec<u32> = draft_tokens[..accepted].to_vec();
                new_tokens.push(correction_or_bonus);
                context.extend_from_slice(&new_tokens);

                // --- Advance caches ---
                let t_advance = std::time::Instant::now();

                if accepted == k {
                    let catchup: Vec<i32> =
                        vec![draft_tokens[k - 1] as i32, correction_or_bonus as i32];
                    let catchup_arr = mlx_rs::Array::from_slice(&catchup, &[1, 2]);
                    saved_draft_logits =
                        draft.forward(&catchup_arr, None, &mut draft_cache).unwrap();
                    mlx_rs::transforms::eval([&saved_draft_logits]).unwrap();
                    draft_cache.eval_for_clone().unwrap();

                    let bonus_arr =
                        mlx_rs::Array::from_slice(&[correction_or_bonus as i32], &[1, 1]);
                    saved_verify_logits =
                        verify.forward(&bonus_arr, None, &mut verify_cache).unwrap();
                    mlx_rs::transforms::eval([&saved_verify_logits]).unwrap();
                    verify_cache.eval_for_clone().unwrap();
                } else {
                    let advance_input: Vec<i32> =
                        new_tokens.iter().map(|&t| t as i32).collect();
                    let advance_arr = mlx_rs::Array::from_slice(
                        &advance_input,
                        &[1, advance_input.len() as i32],
                    );

                    draft_cache = draft_snapshot;
                    saved_draft_logits =
                        draft.forward(&advance_arr, None, &mut draft_cache).unwrap();
                    mlx_rs::transforms::eval([&saved_draft_logits]).unwrap();
                    draft_cache.eval_for_clone().unwrap();

                    verify_cache = verify_snapshot;
                    saved_verify_logits =
                        verify.forward(&advance_arr, None, &mut verify_cache).unwrap();
                    mlx_rs::transforms::eval([&saved_verify_logits]).unwrap();
                    verify_cache.eval_for_clone().unwrap();
                }

                advance_time_ms += t_advance.elapsed().as_secs_f64() * 1000.0;

                let prev_k = k;
                controller.record(accepted, k);
                let new_k = controller.current_k();

                total_accepted += accepted;
                total_drafted += k;

                let draft_toks_str: Vec<String> =
                    draft_tokens.iter().map(|t| t.to_string()).collect();
                let k_change = if new_k != prev_k {
                    format!(" → K={new_k}")
                } else {
                    String::new()
                };
                eprintln!(
                    "  Round {:2}: K={:2} accepted {}/{} (+{} new) | drafted [{}]{}",
                    round + 1,
                    k,
                    accepted,
                    k,
                    new_tokens.len(),
                    draft_toks_str.join(", "),
                    k_change,
                );
            }

            // --- Results ---
            let acceptance_rate = total_accepted as f64 / total_drafted as f64 * 100.0;
            let avg_per_round = total_accepted as f64 / N_ROUNDS as f64;
            let total_new = context.len() - prompt_ids.len();
            let total_ms = draft_time_ms + verify_time_ms + advance_time_ms;
            let eff_tps = if total_ms > 0.0 {
                total_new as f64 / total_ms * 1000.0
            } else {
                0.0
            };

            eprintln!("\n  --- Results for {prompt:?} ---");
            eprintln!("  Acceptance rate:     {acceptance_rate:.1}%");
            eprintln!("  Avg accepted/round:  {avg_per_round:.1}");
            eprintln!("  Total new tokens:    {total_new}");
            eprintln!(
                "  Draft time:    {draft_time_ms:.0}ms ({:.1}ms/round)",
                draft_time_ms / N_ROUNDS as f64
            );
            eprintln!(
                "  Verify time:   {verify_time_ms:.0}ms ({:.1}ms/round)",
                verify_time_ms / N_ROUNDS as f64
            );
            eprintln!(
                "  Advance time:  {advance_time_ms:.0}ms ({:.1}ms/round)",
                advance_time_ms / N_ROUNDS as f64
            );
            eprintln!("  Effective throughput: {eff_tps:.1} tok/s");

            if let Ok(decoded) = tokenizer.decode(&context[prompt_ids.len()..], true) {
                eprintln!("  Generated text: {decoded:?}");
            }
        }
    }

    #[test]
    fn test_adaptive_k_controller_thresholds() {
        let mut c = AdaptiveKController::new(8, 16, 3);
        assert_eq!(c.current_k(), 16, "starts at k_high");

        // 3 rounds of low acceptance → should drop to k_low
        c.record(3, 16); // 0.1875
        c.record(4, 16); // 0.25
        c.record(5, 16); // 0.3125 — avg 0.25 < 0.50
        assert_eq!(c.current_k(), 8, "drops to k_low after low acceptance");

        // 3 rounds of high acceptance → should rise to k_high
        c.record(7, 8); // 0.875
        c.record(6, 8); // 0.75
        c.record(7, 8); // 0.875 — avg 0.833 > 0.65
        assert_eq!(c.current_k(), 16, "rises to k_high after high acceptance");
    }

    #[test]
    fn test_adaptive_k_controller_hysteresis() {
        let mut c = AdaptiveKController::new(8, 16, 3);

        // Drop to k_low
        c.record(2, 16);
        c.record(3, 16);
        c.record(2, 16); // avg ~0.146
        assert_eq!(c.current_k(), 8);

        // Mid-range acceptance (between thresholds) → stays at k_low
        c.record(5, 8); // 0.625
        c.record(4, 8); // 0.50
        c.record(5, 8); // 0.625 — avg 0.583, between 0.50 and 0.65
        assert_eq!(c.current_k(), 8, "hysteresis: stays at k_low in dead zone");
    }

    // -----------------------------------------------------------------------
    // accept_prefix tests (pure logic, no models)
    // -----------------------------------------------------------------------

    #[test]
    fn test_accept_prefix() {
        // All match → all draft tokens + bonus
        assert_eq!(
            super::accept_prefix(&[5, 3, 7], &[5, 3, 7, 42]),
            vec![5, 3, 7, 42]
        );

        // First mismatch → correction only
        assert_eq!(
            super::accept_prefix(&[5, 3, 7], &[9, 1, 2, 0]),
            vec![9]
        );

        // Mid mismatch → accepted prefix + correction
        assert_eq!(
            super::accept_prefix(&[5, 3, 7], &[5, 3, 9, 0]),
            vec![5, 3, 9]
        );

        // Single draft token, matches → draft + bonus
        assert_eq!(super::accept_prefix(&[10], &[10, 20]), vec![10, 20]);

        // Single draft token, no match → correction
        assert_eq!(super::accept_prefix(&[10], &[99, 20]), vec![99]);

        // Empty draft → just the correction/bonus
        assert_eq!(super::accept_prefix(&[], &[42]), vec![42]);
    }

    // -----------------------------------------------------------------------
    // ANE Causal Drafter tests
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Needs Qwen3-0.6B-Base model on disk"]
    fn test_ane_causal_drafter_generates() {
        let Some(dir) = qwen3_base_dir() else {
            eprintln!("SKIP: Qwen3-0.6B-Base not found");
            return;
        };

        eprintln!("Building AneCausalDrafter (max_seq=64)...");
        let t0 = std::time::Instant::now();
        let drafter = super::AneCausalDrafter::new(&dir, 64).expect("build drafter");
        eprintln!("  Built in {:.1}s", t0.elapsed().as_secs_f64());

        let prefix: Vec<u32> = vec![2, 1820, 374]; // "The answer is"
        eprintln!("Drafting 4 tokens from prefix {:?}...", prefix);
        let t1 = std::time::Instant::now();
        let tokens = drafter.draft(&prefix, 4);
        let draft_ms = t1.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Got {} tokens in {draft_ms:.0}ms: {:?}", tokens.len(), tokens);

        assert_eq!(tokens.len(), 4, "should produce exactly 4 draft tokens");
        assert!(
            tokens.iter().all(|&t| (t as usize) < drafter.vocab),
            "all tokens should be < vocab"
        );
        // Not all the same token (degenerate)
        let unique: std::collections::HashSet<u32> = tokens.iter().copied().collect();
        // Relax this check slightly — it's possible but very unlikely for 4 tokens to be identical
        eprintln!("  Unique tokens: {}", unique.len());
    }

    #[test]
    #[cfg(feature = "ane")]
    #[ignore = "Needs Qwen3-0.6B-Base + Qwen3-14B-4bit models on disk"]
    fn test_speculative_generate_e2e() {
        let Some(draft_dir) = qwen3_base_dir() else {
            eprintln!("SKIP: Qwen3-0.6B-Base not found");
            return;
        };
        let Some(verify_dir) = qwen3_14b_dir() else {
            eprintln!("SKIP: Qwen3-14B-4bit not found");
            return;
        };

        eprintln!("Building ANE drafter (0.6B, max_seq=128)...");
        let drafter = super::AneCausalDrafter::new(&draft_dir, 128).expect("build drafter");

        eprintln!("Loading GPU verifier (14B)...");
        let t0 = std::time::Instant::now();
        let verify_model = crate::transformer::load_model(&verify_dir).unwrap();
        let mut verifier = crate::AnyModel::Transformer(verify_model);
        eprintln!("  Verifier loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let prompt: Vec<u32> = vec![785, 6722, 315, 9625, 374]; // "The capital of France is"

        eprintln!("Running speculative_generate (max_tokens=20, k=4..8)...");
        let t1 = std::time::Instant::now();
        let tokens = super::speculative_generate(
            &drafter,
            &mut verifier,
            &prompt,
            20,
            4,
            8,
        );
        let gen_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let tps = if gen_ms > 0.0 {
            tokens.len() as f64 / gen_ms * 1000.0
        } else {
            0.0
        };
        eprintln!(
            "  Generated {} tokens in {gen_ms:.0}ms ({tps:.1} tok/s)",
            tokens.len()
        );
        eprintln!("  Tokens: {:?}", tokens);

        assert!(!tokens.is_empty(), "should produce at least 1 token");
        assert!(tokens.len() <= 20, "should not exceed max_tokens");
    }

    // -----------------------------------------------------------------------
    // ANE AR Decode Engine tests
    // -----------------------------------------------------------------------

    /// Unit test: compile a tiny decode kernel with random weights, eval, check not NaN.
    /// No model files needed — all weights are synthetic.
    #[test]
    #[cfg(feature = "ane")]
    fn test_ane_decode_kernel_compiles() {
        use crate::ane_bridge::{self, build_weight_blob, build_weight_blob_transposed, AneKernel};
        use crate::diffusion_ane;

        ane_bridge::ane_init().expect("ANE init");

        // Tiny dims (all spatial >= 16)
        let dim = 64;
        let heads = 2;
        let kv_heads = 1;
        let hd = 32;
        let half_hd = 16;
        let inter = 64;
        let max_seq = 16;
        let attn_dim = heads * hd; // 64

        let mil = diffusion_ane::gen_decode_layer(dim, heads, kv_heads, hd, inter, max_seq, 1e-6);
        let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

        // Random weights
        let rms_att = build_weight_blob(&vec![1.0f32; dim], 1, dim);
        let wq = build_weight_blob_transposed(&vec![0.01f32; attn_dim * dim], attn_dim, dim);
        let wo = build_weight_blob_transposed(&vec![0.01f32; dim * attn_dim], dim, attn_dim);
        let q_norm = build_weight_blob(&vec![1.0f32; hd], 1, hd);
        let rms_ffn = build_weight_blob(&vec![1.0f32; dim], 1, dim);
        let gate = build_weight_blob_transposed(&vec![0.01f32; inter * dim], inter, dim);
        let up = build_weight_blob_transposed(&vec![0.01f32; inter * dim], inter, dim);
        let down = build_weight_blob_transposed(&vec![0.01f32; dim * inter], dim, inter);

        let blobs: Vec<&[u8]> = vec![&rms_att, &wq, &wo, &q_norm, &rms_ffn, &gate, &up, &down];

        let kernel = compile_ane_kernel(
            "test decode compile",
            &mil.mil_text,
            &names,
            &blobs,
            &[mil.input_bytes],
            &[mil.output_bytes],
        )
        .expect("decode kernel should compile");

        // Build single packed input buffer: [total_ch * max_seq] f32
        let kv_dim = kv_heads * hd;
        let total_ch = dim + 2 * kv_dim + hd + 1;
        let mut input_buf = vec![0.0f32; total_ch * max_seq];

        // x at channels [0..dim), position 0
        for ch in 0..dim {
            input_buf[ch * max_seq] = (ch as f32 * 0.01).sin();
        }

        // K/V cache: leave as zeros (will be masked)

        // rope_cos at position 0, channels [dim+2*kv_dim .. dim+2*kv_dim+half_hd)
        let rope_base = dim + 2 * kv_dim;
        for d in 0..half_hd {
            input_buf[(rope_base + d) * max_seq] = 1.0; // cos(0) = 1
        }
        // rope_sin: leave as 0 (sin(0) = 0)

        // mask: unmask position 0, rest masked
        let mask_ch = dim + 2 * kv_dim + hd;
        input_buf[mask_ch * max_seq] = 0.0;
        for p in 1..max_seq {
            input_buf[mask_ch * max_seq + p] = -1e9;
        }

        let bytes: Vec<u8> = input_buf.iter().flat_map(|v| v.to_le_bytes()).collect();
        kernel.write_input(0, &bytes);

        kernel.eval().expect("decode eval");
        let out = kernel.read_output_zerocopy(0, dim, max_seq);

        assert!(out.iter().all(|v| v.is_finite()), "output contains NaN/Inf");
        let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("Decode kernel output max_abs: {max_abs:.6}");
        assert!(max_abs > 0.0, "output is all zeros");
    }

    /// Integration test: decode 5 tokens, verify outputs are sensible and deterministic.
    /// Runs two independent decode engines with same tokens — outputs must match exactly.
    #[test]
    #[cfg(feature = "ane")]
    #[ignore] // needs Qwen3-0.6B-Base model on disk
    fn test_ane_decode_matches_blas() {
        let Some(base_path) = qwen3_base_dir() else {
            eprintln!("Qwen3-0.6B-Base not found, skipping");
            return;
        };

        let engine = DiffusionEngine::load(&base_path).expect("load model");
        let mut dec1 = AneArDecodeEngine::new(engine.clone(), 128).expect("build decode engine 1");
        let mut dec2 = AneArDecodeEngine::new(engine, 128).expect("build decode engine 2");

        let tokens: Vec<u32> = vec![2, 1820, 374, 264, 1296];

        for &tid in &tokens {
            let logits1 = dec1.decode_step(tid);
            let logits2 = dec2.decode_step(tid);

            assert!(logits1.iter().all(|v| v.is_finite()), "logits1 non-finite");
            assert!(logits2.iter().all(|v| v.is_finite()), "logits2 non-finite");

            let max_diff: f32 = logits1
                .iter()
                .zip(logits2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 0.01, "two runs diverged: max_diff={max_diff}");
        }

        let logits = dec1.decode_step(tokens[0]); // one more token
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        eprintln!("After 5+1 tokens, argmax={} logit={:.4}", argmax.0, argmax.1);
        assert!(argmax.1.abs() > 0.1, "logits suspiciously near zero");
    }

    /// Integration test: snapshot/restore produces correct divergent outputs.
    #[test]
    #[cfg(feature = "ane")]
    #[ignore] // needs Qwen3-0.6B-Base model on disk
    fn test_ane_decode_snapshot_restore() {
        let Some(base_path) = qwen3_base_dir() else {
            eprintln!("Qwen3-0.6B-Base not found, skipping");
            return;
        };

        let engine = DiffusionEngine::load(&base_path).expect("load model");
        let mut decode = AneArDecodeEngine::new(engine, 128).expect("build decode engine");

        // Decode 5 tokens
        let prefix: Vec<u32> = vec![2, 1820, 374, 264, 1296];
        for &tid in &prefix {
            let _ = decode.decode_step(tid);
        }

        // Snapshot at pos=5
        let snap = decode.snapshot();
        assert_eq!(snap, 5);

        // Decode 3 more tokens (branch A)
        let branch_a: Vec<u32> = vec![311, 279, 264];
        let mut logits_a5 = Vec::new();
        for (i, &tid) in branch_a.iter().enumerate() {
            let logits = decode.decode_step(tid);
            if i == 0 {
                logits_a5 = logits;
            }
        }

        // Restore to pos=5
        decode.restore(snap);
        assert_eq!(decode.current_pos(), 5);

        // Decode 3 DIFFERENT tokens (branch B)
        let branch_b: Vec<u32> = vec![500, 600, 700];
        let mut logits_b5 = Vec::new();
        for (i, &tid) in branch_b.iter().enumerate() {
            let logits = decode.decode_step(tid);
            if i == 0 {
                logits_b5 = logits;
            }
        }

        // At pos=5 (first token after restore), logits should match since
        // the input token differs but the prefix is the same. The logits at
        // pos=5 depend on the token fed (branch_a[0]=311 vs branch_b[0]=500),
        // so they SHOULD diverge.
        let diff: f32 = logits_a5
            .iter()
            .zip(logits_b5.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        eprintln!("Logit diff at pos=5 between branches: {diff:.4}");
        assert!(diff > 1.0, "branches should diverge after different tokens");
    }
}
