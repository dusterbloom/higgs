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

        let config = DiffusionConfig {
            hidden: cfg["hidden_size"].as_u64().unwrap() as usize,
            layers: cfg["num_hidden_layers"].as_u64().unwrap() as usize,
            heads: cfg["num_attention_heads"].as_u64().unwrap() as usize,
            kv_heads: cfg["num_key_value_heads"].as_u64().unwrap() as usize,
            head_dim: cfg["head_dim"].as_u64().unwrap_or(128) as usize,
            inter: cfg["intermediate_size"].as_u64().unwrap() as usize,
            vocab: cfg["vocab_size"].as_u64().unwrap() as usize,
            mask_token_id: 151669, // <|mask|> in Qwen3 tokenizer
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
// ANE-accelerated engine — fully-fused (1 dispatch per layer)
// ---------------------------------------------------------------------------

/// Pre-built weight blobs for one transformer layer (ready for ANE reload).
#[cfg(feature = "ane")]
pub struct AneLayerWeightBlobs {
    /// Weight blobs in the order expected by the compiled MIL kernel.
    pub blobs: Vec<Vec<u8>>,
}

#[cfg(feature = "ane")]
enum AneDiffusionKernelSet {
    Fused {
        kernel: crate::ane_bridge::AneKernel,
        layer_blobs: Vec<AneLayerWeightBlobs>,
    },
    MultiDispatch {
        attn_kernel: crate::ane_bridge::AneKernel,
        attn_layer_blobs: Vec<AneLayerWeightBlobs>,
        ffn_kernel: crate::ane_bridge::AneKernel,
        ffn_layer_blobs: Vec<AneLayerWeightBlobs>,
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

/// ANE-accelerated diffusion engine using a single fused kernel + per-layer weight reload.
///
/// Strategy to work around the ANE's ~12 concurrent kernel limit:
/// - Compile ONE kernel (layer 0). The ANE caches the compiled microcode.
/// - Pre-build weight blobs for all 28 layers.
/// - At eval time: reload_weights(layer_blobs) → eval → next layer.
///   `reload_weights` uses the delta cache (no recompile), just swaps weight BLOBFILEs.
///
/// This gives 1 ANE dispatch per layer with zero CPU computation in the inner loop.
#[cfg(feature = "ane")]
pub struct AneDiffusionEngine {
    pub blas_engine: DiffusionEngine,
    kernels: AneDiffusionKernelSet,
    pub seq_len: usize, // fixed seq for compiled kernels (padded to ANE_MIN_SPATIAL)
}

#[cfg(feature = "ane")]
impl AneDiffusionEngine {
    pub fn backend_kind(&self) -> DiffusionBackend {
        match self.kernels {
            AneDiffusionKernelSet::Fused { .. } => DiffusionBackend::AneFused,
            AneDiffusionKernelSet::MultiDispatch { .. } => DiffusionBackend::AneMultiDispatch,
        }
    }

    /// Compile one fully-fused ANE kernel and pre-build 28 sets of weight blobs.
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

        eprintln!(
            "Compiling 1 fully-fused ANE layer kernel (seq={seq}) + prebuilding {} weight sets...",
            cfg.layers
        );
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

        // Pre-build weight blobs for all layers.
        //
        // Layout rules:
        //   build_weight_blob(w, rows, cols)            → blob [rows, cols] row-major (as fp16)
        //   build_weight_blob_transposed(w, rows, cols) → blob [cols, rows] transposed (as fp16)
        //
        // PyTorch projection weights are stored as [out, in]. The MIL matmul expects
        // the weight tensor as [1, 1, in_ch, out_ch] (i.e. [ic, oc]).
        // build_weight_blob_transposed(w, out, in) → transposes [out,in] to [in,out] = [ic,oc]. ✓
        //
        // Norm weights are 1-D [dim] vectors, stored as [1, dim, 1, 1] in MIL.
        // build_weight_blob(w, 1, dim) packs them correctly.
        let mut fused_layer_blobs: Vec<AneLayerWeightBlobs> = Vec::with_capacity(cfg.layers);
        let mut attn_layer_blobs: Vec<AneLayerWeightBlobs> = Vec::with_capacity(cfg.layers);
        let mut ffn_layer_blobs: Vec<AneLayerWeightBlobs> = Vec::with_capacity(cfg.layers);
        for lw in &engine.layers {
            let fused_blobs = vec![
                build_weight_blob(&lw.input_norm, 1, h),            // rms_att
                build_weight_blob(&lw.post_attn_norm, 1, h),        // rms_ffn
                build_weight_blob_transposed(&lw.q_proj, q_dim, h), // wq
                build_weight_blob_transposed(&lw.k_proj, kv_dim, h), // wk
                build_weight_blob_transposed(&lw.v_proj, kv_dim, h), // wv
                build_weight_blob_transposed(&lw.o_proj, h, q_dim), // wo
                build_weight_blob_transposed(&lw.gate_proj, inter, h), // gate
                build_weight_blob_transposed(&lw.up_proj, inter, h), // up
                build_weight_blob_transposed(&lw.down_proj, h, inter), // down
                rope_cos_blob.clone(),                              // rope_cos
                rope_sin_blob.clone(),                              // rope_sin
                build_weight_blob(&lw.q_norm, 1, hd),               // q_norm
                build_weight_blob(&lw.k_norm, 1, hd),               // k_norm
            ];
            let attn_blobs = vec![
                build_weight_blob(&lw.input_norm, 1, h),
                build_weight_blob_transposed(&lw.q_proj, q_dim, h),
                build_weight_blob_transposed(&lw.k_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.v_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.o_proj, h, q_dim),
                rope_cos_blob.clone(),
                rope_sin_blob.clone(),
                build_weight_blob(&lw.q_norm, 1, hd),
                build_weight_blob(&lw.k_norm, 1, hd),
            ];
            let ffn_blobs = vec![
                build_weight_blob(&lw.post_attn_norm, 1, h),
                build_weight_blob_transposed(&lw.gate_proj, inter, h),
                build_weight_blob_transposed(&lw.up_proj, inter, h),
                build_weight_blob_transposed(&lw.down_proj, h, inter),
            ];
            fused_layer_blobs.push(AneLayerWeightBlobs { blobs: fused_blobs });
            attn_layer_blobs.push(AneLayerWeightBlobs { blobs: attn_blobs });
            ffn_layer_blobs.push(AneLayerWeightBlobs { blobs: ffn_blobs });
        }

        let fused_l0 = &fused_layer_blobs[0];
        let fused_l0_refs: Vec<&[u8]> = fused_l0.blobs.iter().map(|b| b.as_slice()).collect();

        let kernels = match compile_ane_kernel(
            "L0 fused compile",
            &fused_mil.mil_text,
            &fused_names,
            &fused_l0_refs,
            &[fused_mil.input_bytes],
            &[fused_mil.output_bytes],
        ) {
            Ok(kernel) => {
                let compile_ms = t0.elapsed().as_millis();
                eprintln!(
                    "ANE compiled: 1 fused kernel in {}ms, {} weight blobs prebuilt",
                    compile_ms, cfg.layers,
                );
                AneDiffusionKernelSet::Fused {
                    kernel,
                    layer_blobs: fused_layer_blobs,
                }
            }
            Err(fused_err) => {
                eprintln!(
                    "Fused ANE layer compile failed ({fused_err}); falling back to multi-dispatch attention+FFN kernels..."
                );

                let attn_l0 = &attn_layer_blobs[0];
                let attn_l0_refs: Vec<&[u8]> = attn_l0.blobs.iter().map(|b| b.as_slice()).collect();
                let attn_kernel = compile_ane_kernel(
                    "L0 attention compile after fused fallback",
                    &attn_mil.mil_text,
                    &attn_names,
                    &attn_l0_refs,
                    &[attn_mil.input_bytes],
                    &[attn_mil.output_bytes],
                )?;

                let ffn_l0 = &ffn_layer_blobs[0];
                let ffn_l0_refs: Vec<&[u8]> = ffn_l0.blobs.iter().map(|b| b.as_slice()).collect();
                let ffn_kernel = compile_ane_kernel(
                    "L0 FFN compile after fused fallback",
                    &ffn_mil.mil_text,
                    &ffn_names,
                    &ffn_l0_refs,
                    &[ffn_mil.input_bytes],
                    &[ffn_mil.output_bytes],
                )?;

                let compile_ms = t0.elapsed().as_millis();
                eprintln!(
                    "ANE compiled: multi-dispatch fallback in {}ms (attention + FFN, {} layers prebuilt)",
                    compile_ms, cfg.layers,
                );
                AneDiffusionKernelSet::MultiDispatch {
                    attn_kernel,
                    attn_layer_blobs,
                    ffn_kernel,
                    ffn_layer_blobs,
                }
            }
        };

        Ok(Self {
            blas_engine: engine,
            kernels,
            seq_len: seq,
        })
    }

    /// Fully-fused ANE forward pass: embed → 28×(reload+fused_dispatch) → final_norm → LM head.
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
            AneDiffusionKernelSet::Fused {
                kernel,
                layer_blobs,
            } => {
                for lb in layer_blobs {
                    let blob_refs: Vec<&[u8]> = lb.blobs.iter().map(|b| b.as_slice()).collect();
                    kernel.reload_weights(&blob_refs).unwrap();

                    kernel.write_input(0, &layer_in_bytes);
                    kernel.eval().unwrap();
                    kernel.read_output(0, &mut layer_out_bytes);
                    std::mem::swap(&mut layer_in_bytes, &mut layer_out_bytes);
                }
            }
            AneDiffusionKernelSet::MultiDispatch {
                attn_kernel,
                attn_layer_blobs,
                ffn_kernel,
                ffn_layer_blobs,
            } => {
                let mut attn_out_bytes = vec![0u8; h * ps * 4];
                for (attn_lb, ffn_lb) in attn_layer_blobs.iter().zip(ffn_layer_blobs.iter()) {
                    let attn_refs: Vec<&[u8]> =
                        attn_lb.blobs.iter().map(|b| b.as_slice()).collect();
                    attn_kernel.reload_weights(&attn_refs).unwrap();
                    attn_kernel.write_input(0, &layer_in_bytes);
                    attn_kernel.eval().unwrap();
                    attn_kernel.read_output(0, &mut attn_out_bytes);

                    let ffn_refs: Vec<&[u8]> = ffn_lb.blobs.iter().map(|b| b.as_slice()).collect();
                    ffn_kernel.reload_weights(&ffn_refs).unwrap();
                    ffn_kernel.write_input(0, &attn_out_bytes);
                    ffn_kernel.eval().unwrap();
                    ffn_kernel.read_output(0, &mut layer_out_bytes);
                    std::mem::swap(&mut layer_in_bytes, &mut layer_out_bytes);
                }
            }
        }

        let fwd_ms = t_fwd.elapsed().as_millis();
        let dispatches = match &self.kernels {
            AneDiffusionKernelSet::Fused { layer_blobs, .. } => layer_blobs.len(),
            AneDiffusionKernelSet::MultiDispatch {
                attn_layer_blobs, ..
            } => attn_layer_blobs.len() * 2,
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

/// ANE hybrid engine: attention on ANE (~24MB per layer, fits in 32MB), FFN on BLAS (72MB, too large).
#[cfg(feature = "ane")]
pub struct AneBonsaiEngine {
    pub blas_engine: DiffusionEngine,
    /// 28 attention-only ANE kernels (one per layer, weights baked in).
    pub attn_kernels: Vec<crate::ane_bridge::AneKernel>,
    pub seq_len: usize,
    pub eps: f32,
}

#[cfg(feature = "ane")]
impl AneBonsaiEngine {
    /// Build 28 attention-only ANE kernels + keep BLAS engine for FFN.
    pub fn new(engine: DiffusionEngine, seq_len: usize, eps: f32) -> Result<Self, String> {
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
        let seq = seq_len.max(ANE_MIN_SPATIAL);

        eprintln!(
            "AneBonsaiEngine: compiling {} attention ANE kernels (dim={h}, seq={seq}, eps={eps})...",
            cfg.layers
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

        let mil = diffusion_ane::gen_bonsai_attention_projection(
            h,
            cfg.heads,
            cfg.kv_heads,
            hd,
            seq,
            eps as f64,
        );
        let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
        let rope_cos_blob = build_weight_blob(&rope_cos, seq, half_hd);
        let rope_sin_blob = build_weight_blob(&rope_sin, seq, half_hd);

        let build_attn_blobs = |lw: &DiffusionLayerWeights| -> Vec<Vec<u8>> {
            vec![
                build_weight_blob(&lw.input_norm, 1, h),
                build_weight_blob_transposed(&lw.q_proj, q_dim, h),
                build_weight_blob_transposed(&lw.k_proj, kv_dim, h),
                build_weight_blob_transposed(&lw.v_proj, kv_dim, h),
                rope_cos_blob.clone(),
                rope_sin_blob.clone(),
                build_weight_blob(&lw.q_norm, 1, hd),
                build_weight_blob(&lw.k_norm, 1, hd),
            ]
        };

        let l0_blobs = build_attn_blobs(&engine.layers[0]);
        let l0_refs: Vec<&[u8]> = l0_blobs.iter().map(|b| b.as_slice()).collect();
        let kernel0 = compile_ane_kernel(
            "L0 Bonsai attention compile",
            &mil.mil_text,
            &names,
            &l0_refs,
            &[mil.input_bytes],
            &[mil.output_bytes],
        )?;

        let l0_ms = t0.elapsed().as_millis();
        eprintln!("  L0 full compile: {l0_ms}ms");

        let mut attn_kernels = Vec::with_capacity(cfg.layers);
        attn_kernels.push(kernel0);

        for (i, lw) in engine.layers.iter().enumerate().skip(1) {
            let blobs = build_attn_blobs(lw);
            let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
            let ki = attn_kernels[0]
                .patch_from_donor(
                    &mil.mil_text,
                    &names,
                    &refs,
                    &[mil.input_bytes],
                    &[mil.output_bytes],
                )
                .map_err(|e| format!("L{i} patch: {e}"))?;
            attn_kernels.push(ki);
        }

        let total_ms = t0.elapsed().as_millis();
        eprintln!(
            "AneBonsaiEngine: {} attn kernels in {total_ms}ms (L0={l0_ms}ms + {} patches in {}ms)",
            cfg.layers,
            cfg.layers - 1,
            total_ms - l0_ms,
        );

        Ok(Self {
            blas_engine: engine,
            attn_kernels,
            seq_len: seq,
            eps,
        })
    }

    /// Hybrid forward: ANE attention + BLAS FFN per layer → final norm → LM head.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.blas_engine.config;
        let seq = token_ids.len();
        let h = cfg.hidden;
        let q_dim = cfg.heads * cfg.head_dim;
        let inter = cfg.inter;
        let ps = self.seq_len;
        let verbose_trace = std::env::var_os("HIGGS_BONSAI_ANE_TRACE").is_some();

        let mut hidden = vec![0.0f32; seq * h];
        for (i, &tid) in token_ids.iter().enumerate() {
            let off = tid as usize * h;
            hidden[i * h..(i + 1) * h].copy_from_slice(&self.blas_engine.embed[off..off + h]);
        }

        let pack = |data: &[f32], dim: usize| -> Vec<u8> {
            let mut buf = vec![0.0f32; dim * ps];
            for s in 0..seq {
                for c in 0..dim {
                    buf[c * ps + s] = data[s * dim + c];
                }
            }
            buf.iter().flat_map(|f| f.to_le_bytes()).collect()
        };

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

        let mut normed = vec![0.0f32; seq * h];
        let mut attn_proj = vec![0.0f32; seq * h];
        let mut gate_buf = vec![0.0f32; seq * inter];
        let mut up_buf = vec![0.0f32; seq * inter];
        let mut ffn_out = vec![0.0f32; seq * h];
        // Bonsai Q1 drift compounds across depth on ANE attention. The best
        // measured hybrid split keeps most layers on ANE but rescues a sparse
        // set of earlier and late layers on CPU attention to stay within the
        // current correctness gate.
        const BONSAI_CPU_ATTENTION_LAYERS: &[usize] = &[12, 15, 16, 17, 18, 19, 22, 24, 25, 26, 27];

        let t_fwd = std::time::Instant::now();
        for (layer_idx, kernel) in self.attn_kernels.iter().enumerate() {
            let layer = &self.blas_engine.layers[layer_idx];
            let residual = hidden.clone();
            let attn_ctx = if BONSAI_CPU_ATTENTION_LAYERS.contains(&layer_idx) {
                diffusion_attention_context_cpu(&self.blas_engine, layer_idx, &hidden, self.eps)
            } else {
                let input_bytes = pack(&hidden, h);
                kernel.write_input(0, &input_bytes);
                kernel.eval().unwrap();
                let mut output_bytes = vec![0u8; q_dim * ps * 4];
                kernel.read_output(0, &mut output_bytes);
                unpack(&output_bytes, q_dim)
            };
            sgemm_nt(seq, h, q_dim, &attn_ctx, &layer.o_proj, &mut attn_proj);
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
            sgemm_nt(seq, inter, h, &normed, &layer.gate_proj, &mut gate_buf);
            sgemm_nt(seq, inter, h, &normed, &layer.up_proj, &mut up_buf);

            for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) {
                *g = *g * (1.0 / (1.0 + (-*g).exp())) * u;
            }

            sgemm_nt(seq, h, inter, &gate_buf, &layer.down_proj, &mut ffn_out);
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

        let fwd_ms = t_fwd.elapsed().as_secs_f64() * 1000.0;
        if verbose_trace {
            eprintln!(
                "  hybrid fwd: {} ANE attn + {} BLAS FFN in {fwd_ms:.1}ms (seq={seq})",
                self.attn_kernels.len(),
                self.attn_kernels.len(),
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

    /// Benchmark: BLAS vs ANE BLOBFILE for the diffusion projection matmuls.
    /// Tests actual projection shapes at diffusion-relevant seq lengths.
    #[test]
    #[cfg(feature = "ane")]
    fn test_diffusion_blas_vs_ane() {
        use crate::ane_bridge::{self, AneKernel, build_weight_blob};
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
    #[ignore = "Recovered exploratory test; archived run produced non-finite ANE logits"]
    fn test_bonsai_1_7b_ane_hybrid() {
        let Some(bonsai_dir) = bonsai_1_7b_dir() else {
            eprintln!("Bonsai-1.7B not found, skipping");
            return;
        };

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

        eprintln!("Compiling ANE hybrid engine (seq={short_seq}, for correctness)...");
        let ane_engine_short = super::AneBonsaiEngine::new(
            DiffusionEngine::load_q1(&bonsai_dir).unwrap(),
            short_seq,
            1e-6,
        )
        .unwrap();
        let ane_logits_short = ane_engine_short.forward(&short_input);

        let blas_nan = blas_logits_short.iter().filter(|v| !v.is_finite()).count();
        let ane_nan = ane_logits_short.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "  BLAS logits: {} total, {} non-finite",
            blas_logits_short.len(),
            blas_nan
        );
        eprintln!(
            "  ANE  logits: {} total, {} non-finite",
            ane_logits_short.len(),
            ane_nan
        );
        eprintln!("  BLAS logits[0,:5] = {:?}", &blas_logits_short[..5]);
        eprintln!("  ANE  logits[0,:5] = {:?}", &ane_logits_short[..5]);

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

        let input_64: Vec<u32> = (0..64)
            .map(|i| if i < 5 { 785 + i } else { 1000 + i })
            .collect();
        eprintln!("Compiling ANE hybrid engine (seq=64)...");
        let ane_engine_64 =
            super::AneBonsaiEngine::new(DiffusionEngine::load_q1(&bonsai_dir).unwrap(), 64, 1e-6)
                .unwrap();

        let n = 3;
        let _ = blas_engine.forward(&input_64);
        let _ = ane_engine_64.forward(&input_64);

        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ = blas_engine.forward(&input_64);
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

        let input_128: Vec<u32> = (0..128)
            .map(|i| if i < 5 { 785 + i } else { 1000 + i })
            .collect();
        eprintln!("Compiling ANE hybrid engine (seq=128)...");
        let ane_engine_128 =
            super::AneBonsaiEngine::new(DiffusionEngine::load_q1(&bonsai_dir).unwrap(), 128, 1e-6)
                .unwrap();

        let _ = blas_engine.forward(&input_128);
        let _ = ane_engine_128.forward(&input_128);

        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ = blas_engine.forward(&input_128);
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
}
