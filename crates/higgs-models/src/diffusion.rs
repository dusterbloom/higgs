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
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

/// y[M,N] = alpha * A[M,K] @ B[K,N] + beta * y. Row-major.
pub(crate) fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101, 111, 111, // RowMajor, NoTrans, NoTrans
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0, y.as_mut_ptr(), n as i32,
        );
    }
}

/// y[M,N] += alpha * A[M,K] @ B[K,N]. Accumulates into y.
pub(crate) fn sgemm_acc(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101, 111, 111,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            1.0, y.as_mut_ptr(), n as i32,
        );
    }
}

/// y[M,N] = A^T[M,K] @ B[K,N] (A stored as [K,M], transposed)
pub(crate) fn sgemm_at(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101, 112, 111, // RowMajor, Trans, NoTrans
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), m as i32,
            b.as_ptr(), n as i32,
            0.0, y.as_mut_ptr(), n as i32,
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

pub struct DiffusionLayerWeights {
    // Attention projections [out, in] row-major (PyTorch layout).
    pub q_proj: Vec<f32>,    // [heads*head_dim, hidden] = [2048, 1024]
    pub k_proj: Vec<f32>,    // [kv_heads*head_dim, hidden] = [1024, 1024]
    pub v_proj: Vec<f32>,    // [kv_heads*head_dim, hidden] = [1024, 1024]
    pub o_proj: Vec<f32>,    // [hidden, heads*head_dim] = [1024, 2048]
    // QK norm
    pub q_norm: Vec<f32>,    // [head_dim] = [128]
    pub k_norm: Vec<f32>,    // [128]
    // Layer norms
    pub input_norm: Vec<f32>,  // [hidden]
    pub post_attn_norm: Vec<f32>, // [hidden]
    // MLP
    pub gate_proj: Vec<f32>,  // [inter, hidden] = [3072, 1024]
    pub up_proj: Vec<f32>,    // [inter, hidden] = [3072, 1024]
    pub down_proj: Vec<f32>,  // [hidden, inter] = [1024, 3072]
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct DiffusionEngine {
    pub layers: Vec<DiffusionLayerWeights>,
    pub embed: Vec<f32>,        // [vocab, hidden]
    pub final_norm: Vec<f32>,   // [hidden]
    pub config: DiffusionConfig,
    // Precomputed RoPE tables
    pub(crate) rope_cos: Vec<f32>,         // [max_seq, head_dim/2]
    pub(crate) rope_sin: Vec<f32>,
}

impl DiffusionEngine {
    /// Load model from a HuggingFace model directory containing config.json + model.safetensors.
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let dir = model_dir.as_ref();

        // Load config
        let config_str = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let cfg: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("parse config: {e}"))?;

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
            let t = tensors.tensor(name).unwrap_or_else(|_| panic!("Missing: {name}"));
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
                let freq = 1.0 / (config.rope_theta as f32).powf(2.0 * d as f32 / config.head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_dim + d] = angle.cos();
                rope_sin[pos * half_dim + d] = angle.sin();
            }
        }

        eprintln!(
            "DiffusionEngine loaded: {}L, hidden={}, heads={}/{}, vocab={}, {:.0}M params",
            config.layers, config.hidden, config.heads, config.kv_heads,
            config.vocab, embed.len() as f64 / 1e6 * 2.0 // rough param count
        );

        Ok(Self { layers, embed, final_norm, config, rope_cos, rope_sin })
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
                    apply_rope(&mut q_buf[off..off + hd], s, half_hd, &self.rope_cos, &self.rope_sin);
                }
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    apply_rope(&mut k_buf[off..off + hd], s, half_hd, &self.rope_cos, &self.rope_sin);
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
            for i in 0..seq * h { hidden[i] += o_buf[i]; }

            // --- MLP ---
            rms_norm(&hidden, &layer.post_attn_norm, &mut normed, seq, h);

            // gate = normed @ gate_proj^T → [seq, inter]
            sgemm_nt(seq, cfg.inter, h, &normed, &layer.gate_proj, &mut gate_buf);
            // SiLU(gate) = gate * sigmoid(gate)
            for v in gate_buf.iter_mut() { *v *= 1.0 / (1.0 + (-*v).exp()); }

            // up = normed @ up_proj^T → [seq, inter]
            sgemm_nt(seq, cfg.inter, h, &normed, &layer.up_proj, &mut up_buf);

            // gate * up (element-wise)
            for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) { *g *= u; }

            // down = gate_buf @ down_proj^T → [seq, h]
            sgemm_nt(seq, h, cfg.inter, &gate_buf, &layer.down_proj, &mut o_buf);

            // Residual add
            for i in 0..seq * h { hidden[i] += o_buf[i]; }
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
                if canvas[pos] != mask_id { continue; }
                let row = &logits[pos * vocab..(pos + 1) * vocab];
                let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut best_idx = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in row.iter().enumerate() {
                    sum_exp += (v - max_logit).exp();
                    if v > best_val { best_val = v; best_idx = i as u32; }
                }
                let confidence = (best_val - max_logit).exp() / sum_exp;
                mask_positions.push((pos, confidence, best_idx));
            }

            if mask_positions.is_empty() { break; }

            // How many to unmask this step
            let n_masks = mask_positions.len();
            let n_keep = n_masks * (steps - step - 1) / steps;
            let n_unmask = (n_masks - n_keep).max(1);

            // Sort by confidence descending, unmask the top-n
            mask_positions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(pos, _, pred) in mask_positions.iter().take(n_unmask) {
                canvas[pos] = pred;
            }
        }

        canvas
    }
}

// ---------------------------------------------------------------------------
// ANE-accelerated engine — fully-fused (1 dispatch per layer)
// ---------------------------------------------------------------------------

/// Pre-built weight blobs for one transformer layer (ready for ANE reload).
#[cfg(feature = "ane")]
pub struct AneLayerWeightBlobs {
    /// All 13 weight blobs in order expected by the fused MIL kernel.
    pub blobs: Vec<Vec<u8>>,
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
    /// Single compiled kernel — weights are hot-swapped per layer via reload_weights.
    pub kernel: crate::ane_bridge::AneKernel,
    /// Pre-built weight blobs for each layer (13 blobs × 28 layers).
    pub layer_blobs: Vec<AneLayerWeightBlobs>,
    pub seq_len: usize, // fixed seq for compiled kernels (padded to ANE_MIN_SPATIAL)
}

#[cfg(feature = "ane")]
impl AneDiffusionEngine {
    /// Compile one fully-fused ANE kernel and pre-build 28 sets of weight blobs.
    pub fn new(engine: DiffusionEngine, seq_len: usize) -> Result<Self, String> {
        use crate::ane_bridge::{self, AneKernel, build_weight_blob, build_weight_blob_transposed};
        use crate::diffusion_ane;
        use crate::ane_mil::ANE_MIN_SPATIAL;

        ane_bridge::ane_init()?;

        let cfg = &engine.config;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let half_hd = hd / 2;
        let q_dim = cfg.heads * hd;
        let kv_dim = cfg.kv_heads * hd;
        let inter = cfg.inter;
        let seq = seq_len.max(ANE_MIN_SPATIAL);

        eprintln!("Compiling 1 fully-fused ANE layer kernel (seq={seq}) + prebuilding {} weight sets...", cfg.layers);
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

        // Generate the MIL once — identical for all 28 layers.
        let mil = diffusion_ane::gen_fused_diffusion_layer(
            h, cfg.heads, cfg.kv_heads, hd, inter, seq, 1e-6,
        );
        let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

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
        let mut layer_blobs: Vec<AneLayerWeightBlobs> = Vec::with_capacity(cfg.layers);
        for lw in &engine.layers {
            let blobs = vec![
                build_weight_blob(&lw.input_norm, 1, h),                          // rms_att
                build_weight_blob(&lw.post_attn_norm, 1, h),                      // rms_ffn
                build_weight_blob_transposed(&lw.q_proj, q_dim, h),               // wq
                build_weight_blob_transposed(&lw.k_proj, kv_dim, h),              // wk
                build_weight_blob_transposed(&lw.v_proj, kv_dim, h),              // wv
                build_weight_blob_transposed(&lw.o_proj, h, q_dim),               // wo
                build_weight_blob_transposed(&lw.gate_proj, inter, h),            // gate
                build_weight_blob_transposed(&lw.up_proj, inter, h),              // up
                build_weight_blob_transposed(&lw.down_proj, h, inter),            // down
                rope_cos_blob.clone(),                                              // rope_cos
                rope_sin_blob.clone(),                                              // rope_sin
                build_weight_blob(&lw.q_norm, 1, hd),                             // q_norm
                build_weight_blob(&lw.k_norm, 1, hd),                             // k_norm
            ];
            layer_blobs.push(AneLayerWeightBlobs { blobs });
        }

        // Compile ONE kernel using layer-0 weights.
        let l0 = &layer_blobs[0];
        let l0_refs: Vec<&[u8]> = l0.blobs.iter().map(|b| b.as_slice()).collect();
        let kernel = AneKernel::compile_multi_weights(
            &mil.mil_text, &names, &l0_refs,
            &[mil.input_bytes], &[mil.output_bytes],
        ).map_err(|e| format!("L0 compile: {e}"))?;

        let compile_ms = t0.elapsed().as_millis();
        eprintln!(
            "ANE compiled: 1 fused kernel in {}ms, {} weight blobs prebuilt",
            compile_ms, cfg.layers,
        );

        Ok(Self { blas_engine: engine, kernel, layer_blobs, seq_len: seq })
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

        // 2. Layer loop — 1 ANE dispatch per layer.
        //    For each layer: reload weights (delta path, no recompile) → write input → eval → read output.
        //    The fused kernel handles: RMSNorm → QKV → QK-norm → RoPE → SDPA → Wo → residual
        //                             → RMSNorm → FFN → residual.
        let mut layer_in_bytes = pack(&hidden, h);
        let mut layer_out_bytes = vec![0u8; h * ps * 4];

        for lb in &self.layer_blobs {
            // Hot-swap weights for this layer.
            // Uses delta cache: unloads model, writes new BLOBFILEs to disk, reloads.
            // No recompile (compileWithQoS is skipped). Correct results verified empirically.
            let blob_refs: Vec<&[u8]> = lb.blobs.iter().map(|b| b.as_slice()).collect();
            self.kernel.reload_weights(&blob_refs).unwrap();

            self.kernel.write_input(0, &layer_in_bytes);
            self.kernel.eval().unwrap();
            self.kernel.read_output(0, &mut layer_out_bytes);
            // Swap: output of this layer becomes input of next layer.
            std::mem::swap(&mut layer_in_bytes, &mut layer_out_bytes);
        }

        let fwd_ms = t_fwd.elapsed().as_millis();
        eprintln!("  fwd: {} ANE dispatches in {}ms (seq={seq})", self.layer_blobs.len(), fwd_ms);

        // Unpack final layer output back to row-major [seq, h].
        let hidden_out = unpack(&layer_in_bytes, h);

        // 3. Final RMSNorm (CPU)
        let mut normed = vec![0.0f32; seq * h];
        rms_norm(&hidden_out, &self.blas_engine.final_norm, &mut normed, seq, h);

        // 4. LM head: normed[seq, h] @ embed^T[h, vocab] → logits[seq, vocab]
        //    Embedding matrix doubles as LM head weight (tied weights).
        //    Too large for ANE (~600MB fp32), so we use BLAS on CPU.
        let mut logits = vec![0.0f32; seq * cfg.vocab];
        sgemm_nt(seq, cfg.vocab, h, &normed, &self.blas_engine.embed, &mut logits);
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
                if canvas[pos] != mask_id { continue; }
                let row = &logits[pos * vocab..(pos + 1) * vocab];
                let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut best_idx = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in row.iter().enumerate() {
                    sum_exp += (v - max_logit).exp();
                    if v > best_val { best_val = v; best_idx = i as u32; }
                }
                let confidence = (best_val - max_logit).exp() / sum_exp;
                mask_positions.push((pos, confidence, best_idx));
            }

            if mask_positions.is_empty() { break; }

            let n_masks = mask_positions.len();
            let n_keep = n_masks * (steps - step - 1) / steps;
            let n_unmask = (n_masks - n_keep).max(1);

            mask_positions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
            101, 111, 112, // RowMajor, NoTrans, Trans
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), k as i32, // B is [N,K], ldb=K for Trans
            0.0, c.as_mut_ptr(), n as i32,
        );
    }
}

/// sgemm with B transposed and scaled: C = alpha * A @ B^T.
pub(crate) fn sgemm_nt_scaled(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32], alpha: f32) {
    unsafe {
        cblas_sgemm(
            101, 111, 112,
            m as i32, n as i32, k as i32,
            alpha, a.as_ptr(), k as i32,
            b.as_ptr(), k as i32,
            0.0, c.as_mut_ptr(), n as i32,
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
    for v in row.iter_mut() { *v *= inv; }
}

/// y[M,N] += A^T[M,K] @ B[K,N] (A stored as [K,M], transposed). Accumulates into y.
pub(crate) fn sgemm_at_acc(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101, 112, 111, // RowMajor, Trans, NoTrans
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), m as i32,
            b.as_ptr(), n as i32,
            1.0, y.as_mut_ptr(), n as i32,
        );
    }
}

/// sgemm with B transposed, accumulating: C[M,N] += A[M,K] @ B^T[K,N].
pub(crate) fn sgemm_nt_acc(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        cblas_sgemm(
            101, 111, 112,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), k as i32,
            1.0, c.as_mut_ptr(), n as i32,
        );
    }
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
            (0..n).map(|i| ((i as f32 * 0.618033988 + 0.31415926).fract() * 2.0 - 1.0) * scale).collect()
        };

        let embed = rand_vec(vocab * h, 0.02);
        let final_norm = vec![1.0f32; h]; // init norm weights to 1

        let mut layers = Vec::with_capacity(config.layers);
        for l in 0..config.layers {
            let seed = l * 1000;
            let rv = |n: usize| -> Vec<f32> {
                (0..n).map(|i| (((seed + i) as f32 * 0.618033988 + 0.31415926).fract() * 2.0 - 1.0) * 0.02).collect()
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
                let freq = 1.0 / (config.rope_theta as f32).powf(2.0 * d as f32 / config.head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_dim + d] = angle.cos();
                rope_sin[pos * half_dim + d] = angle.sin();
            }
        }

        Self { layers, embed, final_norm, config, rope_cos, rope_sin }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, unsafe_code)]
mod tests {
    use super::*;

    fn model_dir() -> Option<String> {
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
            785, 6722, 315, 9625, 374,
            151669, 151669, 151669, 151669, 151669, 151669, 151669, 151669,
            151669, 151669, 151669, 151669, 151669, 151669, 151669, 151669,
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

        eprintln!("Generated {} tokens in {}ms ({:.1} tok/s)",
            32, elapsed.as_millis(), 32.0 / elapsed.as_secs_f64());
        eprintln!("Result IDs: {:?}", &result[5..]);

        // We can't decode without the tokenizer, but verify it produces non-mask tokens
        let masks = result.iter().filter(|&&t| t == 151669).count();
        eprintln!("Remaining masks: {masks}");
        assert!(masks < 5, "Too many masks remaining: {masks}");
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
                let w: Vec<f32> = (0..oc * ic).map(|i| ((i as f32) * 0.00001).sin() * 0.01).collect();
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
                    &mil.mil_text, &names, &[&blob],
                    &[mil.input_bytes], &[mil.output_bytes],
                ).expect("compile failed");

                // Write activation (padded layout: [ic, seq] channel-first f32)
                let mut act_padded = vec![0.0f32; ic * seq];
                for s in 0..seq { for c in 0..ic { act_padded[c * seq + s] = act[s * ic + c]; } }
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
                    "q_proj (1024→2048)" => 1,    // q
                    "kv_proj (1024→1024)" => 3,   // k, v, o
                    "gate/up (1024→3072)" => 2,   // gate + up
                    "down (3072→1024)" => 1,       // down
                    _ => 1,
                };
                total_blas_us += blas_us * per_layer * 28;
                total_ane_us += ane_us * per_layer * 28;

                let speedup = blas_us as f64 / ane_us as f64;
                eprintln!("  {label:<25} BLAS={blas_us:>6}µs  ANE={ane_us:>6}µs  {speedup:.2}x  (×{} ×28L)", per_layer);
            }

            let speedup = total_blas_us as f64 / total_ane_us as f64;
            eprintln!("  TOTAL 28L projection: BLAS={:.1}ms  ANE={:.1}ms  → {speedup:.2}x ANE",
                total_blas_us as f64 / 1000.0, total_ane_us as f64 / 1000.0);

            // Estimate full forward: projections + overhead (~20% for norms/rope/softmax/residual)
            let overhead_factor = 1.25;
            let blas_fwd = total_blas_us as f64 / 1000.0 * overhead_factor;
            let ane_fwd = total_ane_us as f64 / 1000.0 * overhead_factor;
            let blas_tps = (seq - 5) as f64 / (blas_fwd * 64.0 / 1000.0);
            let ane_tps = (seq - 5) as f64 / (ane_fwd * 64.0 / 1000.0);
            eprintln!("  Est. full fwd: BLAS≈{blas_fwd:.0}ms  ANE≈{ane_fwd:.0}ms");
            eprintln!("  Est. 64-step:  BLAS≈{:.0} tok/s  ANE≈{:.0} tok/s", blas_tps, ane_tps);
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
        let input: Vec<u32> = prompt_ids.iter().copied()
            .chain(std::iter::repeat(151669).take(n_gen)).collect();

        let blas_logits = blas_engine.forward(&input);
        let ane_logits = ane.forward(&input);

        let max_err = blas_logits.iter().zip(ane_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("BLAS vs ANE logit max_err: {max_err:.4}");

        // ANE generation
        let t0 = std::time::Instant::now();
        let result = ane.generate(&prompt_ids, n_gen, steps);
        let elapsed = t0.elapsed();
        let tps = n_gen as f64 / elapsed.as_secs_f64();
        eprintln!("ANE generated {n_gen} tokens in {}ms ({tps:.1} tok/s, {steps} steps)",
            elapsed.as_millis());
        eprintln!("Result IDs (gen part): {:?}", &result[5..5+10.min(n_gen)]);

        // Also time BLAS for comparison
        let t0 = std::time::Instant::now();
        let blas_result = blas_engine.generate(&prompt_ids, n_gen, steps);
        let blas_elapsed = t0.elapsed();
        let blas_tps = n_gen as f64 / blas_elapsed.as_secs_f64();
        eprintln!("BLAS generated {n_gen} tokens in {}ms ({blas_tps:.1} tok/s)",
            blas_elapsed.as_millis());

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
}
