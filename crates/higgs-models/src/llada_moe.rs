//! LLaDA-MoE-7B-A1B — Diffusion Language Model with Mixture of Experts.
//!
//! Self-contained BLAS engine: loads safetensors, runs forward, generates via MDLM denoising.
//! Architecture: Qwen3-style bidirectional attention + top-8/64 MoE FFN, 16 layers, dim=2048.
//! Active params per token: ~1B (of 7B total). Perfect for ANE due to small active compute.

#![allow(clippy::too_many_arguments, unsafe_code)]

use std::collections::HashMap;
use std::path::Path;

// BLAS FFI — Accelerate framework
unsafe extern "C" {
    unsafe fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

/// y[M,N] = A[M,K] @ B[K,N]. Row-major.
fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(101, 111, 111, m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32, b.as_ptr(), n as i32,
            0.0, y.as_mut_ptr(), n as i32);
    }
}

/// y[M,N] = A[M,K] @ B^T[K,N] where B stored as [N,K].
fn sgemm_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(101, 111, 112, m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32, b.as_ptr(), k as i32,
            0.0, y.as_mut_ptr(), n as i32);
    }
}

/// y[M,N] += A[M,K] @ B^T[K,N] where B stored as [N,K]. Accumulates.
fn sgemm_nt_acc(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemm(101, 111, 112, m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32, b.as_ptr(), k as i32,
            1.0, y.as_mut_ptr(), n as i32);
    }
}

/// y[M,N] = alpha * A[M,K] @ B^T[K,N].
fn sgemm_nt_scaled(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32], alpha: f32) {
    unsafe {
        cblas_sgemm(101, 111, 112, m as i32, n as i32, k as i32,
            alpha, a.as_ptr(), k as i32, b.as_ptr(), k as i32,
            0.0, y.as_mut_ptr(), n as i32);
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LladaMoeConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub expert_inter: usize,  // 1024
    pub num_experts: usize,   // 64
    pub top_k: usize,         // 8
    pub vocab: usize,
    pub mask_token_id: u32,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

// ---------------------------------------------------------------------------
// BF16 packed weight: raw bytes, decoded to f32 on demand per matmul.
// 14.7GB bf16 in RAM (vs 29.4GB f32). Lossless. ~1.3GB scratch for decode.
// ---------------------------------------------------------------------------

pub struct Bf16Weight {
    pub data: Vec<u8>,    // raw bf16 bytes [rows * cols * 2]
    pub rows: usize,
    pub cols: usize,
}

impl Bf16Weight {
    /// Store raw bf16 bytes with shape metadata.
    fn new(data: Vec<u8>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols * 2, "bf16 size mismatch: got {} expected {}", data.len(), rows * cols * 2);
        Self { data, rows, cols }
    }

    /// Decode bf16→f32 into scratch buffer for BLAS matmul.
    fn decode_into(&self, buf: &mut Vec<f32>) {
        let n = self.rows * self.cols;
        buf.resize(n, 0.0);
        for (i, chunk) in self.data.chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            buf[i] = f32::from_bits((bits as u32) << 16);
        }
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

pub struct ExpertWeights {
    pub gate_proj: Bf16Weight, // [expert_inter, hidden]
    pub up_proj: Bf16Weight,   // [expert_inter, hidden]
    pub down_proj: Bf16Weight, // [hidden, expert_inter]
}

pub struct LladaMoeLayerWeights {
    // Attention
    pub q_proj: Bf16Weight,       // [hidden, hidden]
    pub k_proj: Bf16Weight,       // [hidden, hidden]
    pub v_proj: Bf16Weight,       // [hidden, hidden]
    pub o_proj: Bf16Weight,       // [hidden, hidden]
    pub q_norm: Vec<f32>,         // [head_dim] — small, keep f32
    pub k_norm: Vec<f32>,         // [head_dim]
    pub input_norm: Vec<f32>,     // [hidden]
    pub post_attn_norm: Vec<f32>, // [hidden]
    // MoE
    pub router: Bf16Weight,       // [num_experts, hidden]
    pub experts: Vec<ExpertWeights>,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct LladaMoeEngine {
    pub layers: Vec<LladaMoeLayerWeights>,
    pub embed: Vec<f32>,       // [vocab, hidden] — f32 for embedding lookup
    pub lm_head: Bf16Weight,   // [vocab, hidden]
    pub final_norm: Vec<f32>,  // [hidden]
    pub config: LladaMoeConfig,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl LladaMoeEngine {
    /// Load from HuggingFace model directory (multi-shard safetensors).
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let dir = model_dir.as_ref();

        // Load config
        let config_str = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let cfg: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("parse config: {e}"))?;

        let hidden = cfg["hidden_size"].as_u64().unwrap() as usize;
        let heads = cfg["num_attention_heads"].as_u64().unwrap() as usize;
        let head_dim = hidden / heads;

        let config = LladaMoeConfig {
            hidden,
            layers: cfg["num_hidden_layers"].as_u64().unwrap() as usize,
            heads,
            kv_heads: cfg["num_key_value_heads"].as_u64().unwrap() as usize,
            head_dim,
            expert_inter: cfg["expert_intermediate_size"].as_u64().unwrap() as usize,
            num_experts: cfg["num_experts"].as_u64().unwrap() as usize,
            top_k: cfg["num_experts_per_tok"].as_u64().unwrap() as usize,
            vocab: cfg["vocab_size"].as_u64().unwrap() as usize,
            mask_token_id: 156895, // <|mask|> token — NOT eos_token_id (156892)
            rope_theta: cfg["rope_theta"].as_f64().unwrap_or(50000.0),
            rms_norm_eps: cfg["rms_norm_eps"].as_f64().unwrap_or(1e-5),
        };

        // Load shards one at a time, store raw bf16 bytes directly.
        // Peak: 1 shard (~5GB) + accumulated bf16 (~14.7GB) + embed f32 (~1.3GB) ≈ 21GB on 32GB.
        let shard_paths = find_safetensor_shards(dir)?;

        // Collect raw bf16 weights + small f32 norms
        let mut bf16_map: HashMap<String, Vec<u8>> = HashMap::new();
        let mut f32_map: HashMap<String, Vec<f32>> = HashMap::new();

        for (shard_idx, path) in shard_paths.iter().enumerate() {
            let data = std::fs::read(path).map_err(|e| format!("read shard {}: {e}", path.display()))?;
            let st = safetensors::SafeTensors::deserialize(&data)
                .map_err(|e| format!("deserialize shard {}: {e}", path.display()))?;
            let names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
            let n_tensors = names.len();

            for name in names {
                let t = st.tensor(&name).unwrap();
                let raw = t.data();

                // Small tensors (norms, embeddings) → f32. Large tensors → keep bf16 raw.
                if name.ends_with("_norm.weight") || name.ends_with("layernorm.weight")
                    || name == "model.norm.weight"
                {
                    f32_map.insert(name, bf16_to_f32(raw));
                } else if name == "model.embed_tokens.weight" {
                    // Embedding must be f32 for lookup
                    f32_map.insert(name, bf16_to_f32(raw));
                } else {
                    // Store raw bf16 bytes — decoded to f32 on demand during forward
                    bf16_map.insert(name, raw.to_vec());
                }
            }
            eprintln!("  shard {}/{}: {} tensors from {}",
                shard_idx + 1, shard_paths.len(), n_tensors,
                path.file_name().unwrap().to_string_lossy());
            // `data` dropped here — reclaim shard memory before loading next
        }

        // Assemble into structured weights
        let ei = config.expert_inter;
        let ne = config.num_experts;

        let take_bf16 = |map: &mut HashMap<String, Vec<u8>>, name: &str, rows: usize, cols: usize| -> Bf16Weight {
            let data = map.remove(name).unwrap_or_else(|| panic!("Missing bf16: {name}"));
            Bf16Weight::new(data, rows, cols)
        };
        let take_f32 = |map: &mut HashMap<String, Vec<f32>>, name: &str| -> Vec<f32> {
            map.remove(name).unwrap_or_else(|| panic!("Missing f32: {name}"))
        };

        let embed = take_f32(&mut f32_map, "model.embed_tokens.weight");
        let lm_head = take_bf16(&mut bf16_map, "lm_head.weight", config.vocab, hidden);
        let final_norm = take_f32(&mut f32_map, "model.norm.weight");

        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let p = format!("model.layers.{i}");

            let mut experts = Vec::with_capacity(ne);
            for e in 0..ne {
                experts.push(ExpertWeights {
                    gate_proj: take_bf16(&mut bf16_map, &format!("{p}.mlp.experts.{e}.gate_proj.weight"), ei, hidden),
                    up_proj: take_bf16(&mut bf16_map, &format!("{p}.mlp.experts.{e}.up_proj.weight"), ei, hidden),
                    down_proj: take_bf16(&mut bf16_map, &format!("{p}.mlp.experts.{e}.down_proj.weight"), hidden, ei),
                });
            }

            layers.push(LladaMoeLayerWeights {
                q_proj: take_bf16(&mut bf16_map, &format!("{p}.self_attn.q_proj.weight"), hidden, hidden),
                k_proj: take_bf16(&mut bf16_map, &format!("{p}.self_attn.k_proj.weight"), hidden, hidden),
                v_proj: take_bf16(&mut bf16_map, &format!("{p}.self_attn.v_proj.weight"), hidden, hidden),
                o_proj: take_bf16(&mut bf16_map, &format!("{p}.self_attn.o_proj.weight"), hidden, hidden),
                q_norm: take_f32(&mut f32_map, &format!("{p}.self_attn.q_norm.weight")),
                k_norm: take_f32(&mut f32_map, &format!("{p}.self_attn.k_norm.weight")),
                input_norm: take_f32(&mut f32_map, &format!("{p}.input_layernorm.weight")),
                post_attn_norm: take_f32(&mut f32_map, &format!("{p}.post_attention_layernorm.weight")),
                router: take_bf16(&mut bf16_map, &format!("{p}.mlp.gate.weight"), ne, hidden),
                experts,
            });
        }

        // Precompute RoPE
        let max_seq = 4096;
        let half_dim = head_dim / 2;
        let mut rope_cos = vec![0.0f32; max_seq * half_dim];
        let mut rope_sin = vec![0.0f32; max_seq * half_dim];
        for pos in 0..max_seq {
            for d in 0..half_dim {
                let freq = 1.0 / (config.rope_theta as f32).powf(2.0 * d as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_dim + d] = angle.cos();
                rope_sin[pos * half_dim + d] = angle.sin();
            }
        }

        let param_b = (embed.len() + lm_head.rows * lm_head.cols + final_norm.len()
            + layers.iter().map(|l| {
                l.q_proj.rows * l.q_proj.cols + l.k_proj.rows * l.k_proj.cols
                + l.v_proj.rows * l.v_proj.cols + l.o_proj.rows * l.o_proj.cols
                + l.q_norm.len() + l.k_norm.len() + l.input_norm.len() + l.post_attn_norm.len()
                + l.router.rows * l.router.cols
                + l.experts.iter().map(|e| {
                    e.gate_proj.rows * e.gate_proj.cols
                    + e.up_proj.rows * e.up_proj.cols
                    + e.down_proj.rows * e.down_proj.cols
                }).sum::<usize>()
            }).sum::<usize>()) as f64 / 1e9;

        let mem_mb = (embed.len() * 4 + lm_head.data.len() + final_norm.len() * 4
            + layers.iter().map(|l| {
                l.q_proj.data.len() + l.k_proj.data.len() + l.v_proj.data.len() + l.o_proj.data.len()
                + (l.q_norm.len() + l.k_norm.len() + l.input_norm.len() + l.post_attn_norm.len()) * 4
                + l.router.data.len()
                + l.experts.iter().map(|e| e.gate_proj.data.len() + e.up_proj.data.len() + e.down_proj.data.len()).sum::<usize>()
            }).sum::<usize>()) as f64 / 1e6;

        eprintln!(
            "LladaMoeEngine loaded (bf16): {}L, hidden={}, heads={}, {}experts(top-{}), vocab={}, \
             {:.1}B params, {:.0}MB resident",
            config.layers, config.hidden, config.heads, config.num_experts, config.top_k,
            config.vocab, param_b, mem_mb
        );

        Ok(Self { layers, embed, lm_head, final_norm, config, rope_cos, rope_sin })
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
        let q_dim = n_heads * hd; // = h for MHA
        let kv_dim = n_kv * hd;   // = h for MHA

        // 1. Embedding lookup
        let mut hidden = vec![0.0f32; seq * h];
        for (i, &tid) in token_ids.iter().enumerate() {
            let off = tid as usize * h;
            hidden[i * h..(i + 1) * h].copy_from_slice(&self.embed[off..off + h]);
        }

        // 2. Layer loop — scratch buffers
        let mut q_buf = vec![0.0f32; seq * q_dim];
        let mut k_buf = vec![0.0f32; seq * kv_dim];
        let mut v_buf = vec![0.0f32; seq * kv_dim];
        let mut attn_out = vec![0.0f32; seq * q_dim];
        let mut o_buf = vec![0.0f32; seq * h];
        let mut normed = vec![0.0f32; seq * h];
        let mut moe_out = vec![0.0f32; seq * h];

        // MoE scratch
        let mut router_logits = vec![0.0f32; seq * cfg.num_experts];
        let mut gate_buf = vec![0.0f32; seq * cfg.expert_inter];
        let mut up_buf = vec![0.0f32; seq * cfg.expert_inter];

        // Dequant scratch — largest weight is lm_head [vocab, hidden], reused per matmul
        let max_w = cfg.vocab * h; // lm_head is the largest
        let mut dq = vec![0.0f32; max_w];

        for layer in &self.layers {
            // === Attention ===
            rms_norm(&hidden, &layer.input_norm, &mut normed, seq, h, cfg.rms_norm_eps);

            layer.q_proj.decode_into(&mut dq);
            sgemm_nt(seq, q_dim, h, &normed, &dq, &mut q_buf);
            layer.k_proj.decode_into(&mut dq);
            sgemm_nt(seq, kv_dim, h, &normed, &dq, &mut k_buf);
            layer.v_proj.decode_into(&mut dq);
            sgemm_nt(seq, kv_dim, h, &normed, &dq, &mut v_buf);

            // QK norm
            for s in 0..seq {
                for head in 0..n_heads {
                    let off = s * q_dim + head * hd;
                    rms_norm_slice(&mut q_buf[off..off + hd], &layer.q_norm, cfg.rms_norm_eps);
                }
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    rms_norm_slice(&mut k_buf[off..off + hd], &layer.k_norm, cfg.rms_norm_eps);
                }
            }

            // RoPE
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

            // Bidirectional SDPA (no causal mask)
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

            // O projection + residual
            layer.o_proj.decode_into(&mut dq);
            sgemm_nt(seq, h, q_dim, &attn_out, &dq, &mut o_buf);
            for i in 0..seq * h { hidden[i] += o_buf[i]; }

            // === MoE FFN ===
            rms_norm(&hidden, &layer.post_attn_norm, &mut normed, seq, h, cfg.rms_norm_eps);

            // Router: normed[seq, h] @ gate^T[h, num_experts] → [seq, num_experts]
            layer.router.decode_into(&mut dq);
            sgemm_nt(seq, cfg.num_experts, h, &normed, &dq, &mut router_logits);

            // Softmax routing + top-k
            let mut routing_weights = vec![0.0f32; seq * cfg.num_experts];
            for s in 0..seq {
                let row = &router_logits[s * cfg.num_experts..(s + 1) * cfg.num_experts];
                let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for (i, &v) in row.iter().enumerate() {
                    let e = (v - max_v).exp();
                    routing_weights[s * cfg.num_experts + i] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for i in 0..cfg.num_experts {
                    routing_weights[s * cfg.num_experts + i] *= inv;
                }
            }

            // Top-k selection per position
            let mut top_experts = vec![0u32; seq * cfg.top_k];
            let mut top_weights = vec![0.0f32; seq * cfg.top_k];
            for s in 0..seq {
                let rw = &routing_weights[s * cfg.num_experts..(s + 1) * cfg.num_experts];
                // Simple top-k via partial sort
                let mut indices: Vec<usize> = (0..cfg.num_experts).collect();
                indices.sort_unstable_by(|&a, &b| rw[b].partial_cmp(&rw[a]).unwrap());
                for k in 0..cfg.top_k {
                    top_experts[s * cfg.top_k + k] = indices[k] as u32;
                    top_weights[s * cfg.top_k + k] = rw[indices[k]];
                }
            }

            // MoE forward — inverted index: O(seq×top_k) build, skip inactive experts
            moe_out.fill(0.0);
            let mut expert_positions: Vec<Vec<(usize, f32)>> = vec![Vec::new(); cfg.num_experts];
            for s in 0..seq {
                for k in 0..cfg.top_k {
                    let eidx = top_experts[s * cfg.top_k + k] as usize;
                    expert_positions[eidx].push((s, top_weights[s * cfg.top_k + k]));
                }
            }

            for (expert_idx, positions) in expert_positions.iter().enumerate() {
                if positions.is_empty() { continue; }

                let n_tok = positions.len();
                let ei = cfg.expert_inter;
                let expert = &layer.experts[expert_idx];

                // Gather tokens → [n_tok, h]
                let mut gathered = vec![0.0f32; n_tok * h];
                for (t, &(pos, _)) in positions.iter().enumerate() {
                    gathered[t * h..(t + 1) * h].copy_from_slice(&normed[pos * h..(pos + 1) * h]);
                }

                // Expert MLP: SiLU(gate(x)) * up(x) → down
                let gate_slice = &mut gate_buf[..n_tok * ei];
                let up_slice = &mut up_buf[..n_tok * ei];
                expert.gate_proj.decode_into(&mut dq);
                sgemm_nt(n_tok, ei, h, &gathered, &dq, gate_slice);
                expert.up_proj.decode_into(&mut dq);
                sgemm_nt(n_tok, ei, h, &gathered, &dq, up_slice);

                // SiLU(gate) * up
                for (g, u) in gate_slice.iter_mut().zip(up_slice.iter()) {
                    *g = *g * (1.0 / (1.0 + (-*g).exp())) * u;
                }

                // Down: [n_tok, ei] @ down^T[ei, h] → [n_tok, h]
                let mut expert_out = vec![0.0f32; n_tok * h];
                expert.down_proj.decode_into(&mut dq);
                sgemm_nt(n_tok, h, ei, gate_slice, &dq, &mut expert_out);

                // Scatter-add with routing weight
                for (t, &(pos, weight)) in positions.iter().enumerate() {
                    for d in 0..h {
                        moe_out[pos * h + d] += expert_out[t * h + d] * weight;
                    }
                }
            }

            // Residual
            for i in 0..seq * h { hidden[i] += moe_out[i]; }
        }

        // 3. Final RMSNorm
        rms_norm(&hidden, &self.final_norm, &mut normed, seq, h, cfg.rms_norm_eps);

        // 4. LM head (separate weight, NOT tied)
        let mut logits = vec![0.0f32; seq * cfg.vocab];
        self.lm_head.decode_into(&mut dq);
        sgemm_nt(seq, cfg.vocab, h, &normed, &dq, &mut logits);

        logits
    }

    /// MDLM denoising generation.
    pub fn generate(&self, prompt_ids: &[u32], num_tokens: usize, steps: usize) -> Vec<u32> {
        let mask_id = self.config.mask_token_id;
        let vocab = self.config.vocab;
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
// Weight loading helpers
// ---------------------------------------------------------------------------

/// Find safetensor shard files in a directory.
fn find_safetensor_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    // Single file?
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    // Multi-shard: model-NNNNN-of-MMMMM.safetensors
    let mut shard_paths: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("readdir: {e}"))? {
        let entry = entry.map_err(|e| format!("entry: {e}"))?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("model-") && name.ends_with(".safetensors") {
            shard_paths.push(entry.path());
        }
    }
    shard_paths.sort();

    if shard_paths.is_empty() {
        return Err("No safetensors files found".to_string());
    }

    eprintln!("Loading {} safetensors shards (int8 quantization)...", shard_paths.len());
    Ok(shard_paths)
}

/// BF16 bytes → f32 vec.
fn bf16_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn rms_norm(x: &[f32], w: &[f32], out: &mut [f32], seq: usize, dim: usize, eps: f64) {
    let eps = eps as f32;
    for s in 0..seq {
        let row = &x[s * dim..(s + 1) * dim];
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
        let inv = 1.0 / rms;
        for d in 0..dim {
            out[s * dim + d] = row[d] * inv * w[d];
        }
    }
}

fn rms_norm_slice(x: &mut [f32], w: &[f32], eps: f64) {
    let eps = eps as f32;
    let dim = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    for (xi, wi) in x.iter_mut().zip(w.iter()) {
        *xi *= inv * wi;
    }
}

fn apply_rope(x: &mut [f32], pos: usize, half_dim: usize, cos: &[f32], sin: &[f32]) {
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

fn softmax_inplace(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() { *v *= inv; }
}

// ---------------------------------------------------------------------------
// ANE Hybrid Engine: attention on ANE, MoE FFN on BLAS
// ---------------------------------------------------------------------------

#[cfg(feature = "ane")]
pub struct AneLladaMoeEngine {
    pub blas_engine: LladaMoeEngine,
    /// Single compiled attention kernel — weights hot-swapped per layer.
    pub attn_kernel: crate::ane_bridge::AneKernel,
    /// Pre-built attention weight blobs (9 blobs × 16 layers).
    pub attn_blobs: Vec<Vec<Vec<u8>>>,
    pub seq_len: usize,
}

#[cfg(feature = "ane")]
impl AneLladaMoeEngine {
    /// Compile ANE attention kernel, pre-build weight blobs for all layers.
    pub fn new(engine: LladaMoeEngine, seq_len: usize) -> Result<Self, String> {
        use crate::ane_bridge::{self, AneKernel, build_weight_blob, build_weight_blob_transposed};
        use crate::diffusion_ane;
        use crate::ane_mil::ANE_MIN_SPATIAL;

        ane_bridge::ane_init()?;

        let cfg = &engine.config;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let half_hd = hd / 2;
        let heads = cfg.heads;
        let kv_heads = cfg.kv_heads;
        let seq = seq_len.max(ANE_MIN_SPATIAL);

        eprintln!("Compiling ANE attention kernel (dim={h}, seq={seq}) + {} layer weight sets...", cfg.layers);
        let t0 = std::time::Instant::now();

        // Precompute RoPE
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
        let rope_cos_blob = build_weight_blob(&rope_cos, seq, half_hd);
        let rope_sin_blob = build_weight_blob(&rope_sin, seq, half_hd);

        // Generate attention MIL (reuse diffusion attention — same arch: QK-norm, RoPE, bidir SDPA)
        let mil = diffusion_ane::gen_diffusion_attention(h, heads, kv_heads, hd, seq, cfg.rms_norm_eps);
        let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

        // Pre-build attention weight blobs for all layers.
        // Bf16Weight → decode to f32 → build fp16 blob for ANE.
        let mut attn_blobs: Vec<Vec<Vec<u8>>> = Vec::with_capacity(cfg.layers);
        let mut dq = Vec::new(); // reusable f32 scratch

        for lw in &engine.layers {
            // Decode bf16 → f32, then build transposed fp16 ANE blobs
            lw.q_proj.decode_into(&mut dq);
            let wq = build_weight_blob_transposed(&dq, heads * hd, h);
            lw.k_proj.decode_into(&mut dq);
            let wk = build_weight_blob_transposed(&dq, kv_heads * hd, h);
            lw.v_proj.decode_into(&mut dq);
            let wv = build_weight_blob_transposed(&dq, kv_heads * hd, h);
            lw.o_proj.decode_into(&mut dq);
            let wo = build_weight_blob_transposed(&dq, h, heads * hd);

            let blobs = vec![
                build_weight_blob(&lw.input_norm, 1, h),   // rms_att
                wq,                                         // wq
                wk,                                         // wk
                wv,                                         // wv
                wo,                                         // wo
                rope_cos_blob.clone(),                      // rope_cos
                rope_sin_blob.clone(),                      // rope_sin
                build_weight_blob(&lw.q_norm, 1, hd),      // q_norm
                build_weight_blob(&lw.k_norm, 1, hd),      // k_norm
            ];
            attn_blobs.push(blobs);
        }

        // Compile ONE kernel using layer-0 weights
        let l0_refs: Vec<&[u8]> = attn_blobs[0].iter().map(|b| b.as_slice()).collect();
        let attn_kernel = AneKernel::compile_multi_weights(
            &mil.mil_text, &names, &l0_refs,
            &[mil.input_bytes], &[mil.output_bytes],
        ).map_err(|e| format!("L0 attn compile: {e}"))?;

        let compile_ms = t0.elapsed().as_millis();
        let blob_mb: f64 = attn_blobs[0].iter().map(|b| b.len() as f64).sum::<f64>() / 1e6;
        eprintln!(
            "ANE hybrid compiled: 1 attn kernel in {}ms, {:.1}MB/layer, {} layers",
            compile_ms, blob_mb, cfg.layers,
        );

        Ok(Self { blas_engine: engine, attn_kernel, attn_blobs, seq_len: seq })
    }

    /// Hybrid forward: ANE attention + BLAS MoE FFN.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.blas_engine.config;
        let seq = token_ids.len();
        let h = cfg.hidden;
        let ps = self.seq_len; // padded spatial

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

        // Unpack ANE channel-first fp32 [dim, ps] → row-major [seq, dim]
        let unpack = |bytes: &[u8], dim: usize| -> Vec<f32> {
            let all: Vec<f32> = bytes.chunks_exact(4)
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

        // Scratch buffers for MoE
        let ei = cfg.expert_inter;
        let mut normed = vec![0.0f32; seq * h];
        let mut moe_out = vec![0.0f32; seq * h];
        let mut router_logits = vec![0.0f32; seq * cfg.num_experts];
        let mut gate_buf = vec![0.0f32; seq * ei];
        let mut up_buf = vec![0.0f32; seq * ei];
        let mut dq = vec![0.0f32; h * h.max(ei).max(cfg.num_experts)]; // dequant scratch
        let mut attn_out_bytes = vec![0u8; h * ps * 4];

        let t_fwd = std::time::Instant::now();

        // 2. Layer loop: ANE attention → BLAS MoE
        for (layer_idx, layer) in self.blas_engine.layers.iter().enumerate() {
            // === ANE Attention dispatch ===
            let blob_refs: Vec<&[u8]> = self.attn_blobs[layer_idx].iter().map(|b| b.as_slice()).collect();
            self.attn_kernel.reload_weights(&blob_refs).unwrap();

            let attn_in = pack(&hidden, h);
            self.attn_kernel.write_input(0, &attn_in);
            self.attn_kernel.eval().unwrap();
            self.attn_kernel.read_output(0, &mut attn_out_bytes);
            hidden = unpack(&attn_out_bytes, h);

            // === BLAS MoE FFN ===
            rms_norm(&hidden, &layer.post_attn_norm, &mut normed, seq, h, cfg.rms_norm_eps);

            // Router
            layer.router.decode_into(&mut dq);
            sgemm_nt(seq, cfg.num_experts, h, &normed, &dq, &mut router_logits);

            // Softmax routing + top-k
            let mut routing_weights = vec![0.0f32; seq * cfg.num_experts];
            for s in 0..seq {
                let row = &router_logits[s * cfg.num_experts..(s + 1) * cfg.num_experts];
                let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for (i, &v) in row.iter().enumerate() {
                    let e = (v - max_v).exp();
                    routing_weights[s * cfg.num_experts + i] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for i in 0..cfg.num_experts {
                    routing_weights[s * cfg.num_experts + i] *= inv;
                }
            }

            let mut top_experts = vec![0u32; seq * cfg.top_k];
            let mut top_weights = vec![0.0f32; seq * cfg.top_k];
            for s in 0..seq {
                let rw = &routing_weights[s * cfg.num_experts..(s + 1) * cfg.num_experts];
                let mut indices: Vec<usize> = (0..cfg.num_experts).collect();
                indices.sort_unstable_by(|&a, &b| rw[b].partial_cmp(&rw[a]).unwrap());
                for k in 0..cfg.top_k {
                    top_experts[s * cfg.top_k + k] = indices[k] as u32;
                    top_weights[s * cfg.top_k + k] = rw[indices[k]];
                }
            }

            // MoE forward — inverted index: O(seq×top_k) build, skip inactive experts
            moe_out.fill(0.0);
            let mut expert_positions: Vec<Vec<(usize, f32)>> = vec![Vec::new(); cfg.num_experts];
            for s in 0..seq {
                for k in 0..cfg.top_k {
                    let eidx = top_experts[s * cfg.top_k + k] as usize;
                    expert_positions[eidx].push((s, top_weights[s * cfg.top_k + k]));
                }
            }

            let mut n_active = 0u32;
            for (expert_idx, positions) in expert_positions.iter().enumerate() {
                if positions.is_empty() { continue; }
                n_active += 1;

                let n_tok = positions.len();
                let expert = &layer.experts[expert_idx];

                let mut gathered = vec![0.0f32; n_tok * h];
                for (t, &(pos, _)) in positions.iter().enumerate() {
                    gathered[t * h..(t + 1) * h].copy_from_slice(&normed[pos * h..(pos + 1) * h]);
                }

                let gate_slice = &mut gate_buf[..n_tok * ei];
                let up_slice = &mut up_buf[..n_tok * ei];
                expert.gate_proj.decode_into(&mut dq);
                sgemm_nt(n_tok, ei, h, &gathered, &dq, gate_slice);
                expert.up_proj.decode_into(&mut dq);
                sgemm_nt(n_tok, ei, h, &gathered, &dq, up_slice);

                for (g, u) in gate_slice.iter_mut().zip(up_slice.iter()) {
                    *g = *g * (1.0 / (1.0 + (-*g).exp())) * u;
                }

                let mut expert_out = vec![0.0f32; n_tok * h];
                expert.down_proj.decode_into(&mut dq);
                sgemm_nt(n_tok, h, ei, gate_slice, &dq, &mut expert_out);

                for (t, &(pos, weight)) in positions.iter().enumerate() {
                    for d in 0..h {
                        moe_out[pos * h + d] += expert_out[t * h + d] * weight;
                    }
                }
            }
            if layer_idx == 0 {
                eprintln!("    L0: {n_active}/{} experts active", cfg.num_experts);
            }

            // Residual
            for i in 0..seq * h { hidden[i] += moe_out[i]; }
        }

        let fwd_ms = t_fwd.elapsed().as_millis();
        eprintln!("  hybrid fwd: {} ANE attn + BLAS MoE in {}ms (seq={seq})", cfg.layers, fwd_ms);

        // 3. Final RMSNorm
        rms_norm(&hidden, &self.blas_engine.final_norm, &mut normed, seq, h, cfg.rms_norm_eps);

        // 4. LM head (BLAS — too large for ANE)
        let mut logits = vec![0.0f32; seq * cfg.vocab];
        self.blas_engine.lm_head.decode_into(&mut dq);
        sgemm_nt(seq, cfg.vocab, h, &normed, &dq, &mut logits);
        logits
    }

    /// ANE-hybrid MDLM generation.
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn model_dir() -> Option<String> {
        let hub = format!(
            "{}/.cache/huggingface/hub/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct",
            std::env::var("HOME").ok()?
        );
        // HF hub stores files in snapshots/<hash>/ — find latest snapshot with safetensors
        let snap_dir = std::path::Path::new(&hub).join("snapshots");
        if snap_dir.exists() {
            for entry in std::fs::read_dir(&snap_dir).ok()? {
                let p = entry.ok()?.path();
                if p.join("model-00001-of-00003.safetensors").exists() {
                    return Some(p.to_string_lossy().to_string());
                }
            }
        }
        // Fallback: check root dir (older HF layout)
        if std::path::Path::new(&hub).join("model-00001-of-00003.safetensors").exists()
            || std::path::Path::new(&hub).join("model.safetensors").exists()
        {
            return Some(hub);
        }
        None
    }

    #[test]
    fn test_load_and_forward() {
        let Some(dir) = model_dir() else {
            eprintln!("LLaDA-MoE weights not found, skipping");
            return;
        };

        let engine = LladaMoeEngine::load(&dir).unwrap();
        assert_eq!(engine.config.layers, 16);
        assert_eq!(engine.config.hidden, 2048);
        assert_eq!(engine.config.num_experts, 64);
        assert_eq!(engine.config.top_k, 8);

        // Forward with mask tokens
        let mask_id = engine.config.mask_token_id;
        let input: Vec<u32> = (0..32).map(|i| if i < 5 { 1000 + i } else { mask_id }).collect();

        let t0 = std::time::Instant::now();
        let logits = engine.forward(&input);
        let ms = t0.elapsed().as_millis();
        eprintln!("Forward: {}ms for seq={}", ms, input.len());
        assert_eq!(logits.len(), input.len() * engine.config.vocab);

        // Check logits are finite
        assert!(logits.iter().all(|v| v.is_finite()), "Non-finite logits!");
        eprintln!("Logits[0,:5] = {:?}", &logits[..5]);
    }

    #[test]
    fn test_generate() {
        let Some(dir) = model_dir() else {
            eprintln!("LLaDA-MoE weights not found, skipping");
            return;
        };

        let engine = LladaMoeEngine::load(&dir).unwrap();

        // "The capital of France is" → generate 27 tokens with 16 denoising steps
        let prompt: Vec<u32> = vec![678, 7706, 300, 11406, 341]; // "The capital of France is"

        let t0 = std::time::Instant::now();
        let result = engine.generate(&prompt, 27, 16);
        let elapsed = t0.elapsed();
        let n_gen = 27;
        let tps = n_gen as f64 / elapsed.as_secs_f64();
        eprintln!("Generated {n_gen} tokens in {}ms ({tps:.1} tok/s, 16 steps)", elapsed.as_millis());
        eprintln!("Result IDs (gen part): {:?}", &result[5..15]);

        let masks = result.iter().filter(|&&t| t == engine.config.mask_token_id).count();
        eprintln!("Remaining masks: {masks}");
        assert!(masks < 5, "Too many masks remaining: {masks}");

        // Model should produce at least some non-eos tokens (e.g. "Paris.<|role_end|>")
        let non_eos = result[5..].iter().filter(|&&t| t != 156892).count();
        eprintln!("Non-eos tokens: {non_eos}/27");
        assert!(non_eos >= 1, "Model generated all eos — possible correctness issue");
    }

    #[test]
    fn test_forward_speed() {
        let Some(dir) = model_dir() else {
            eprintln!("LLaDA-MoE weights not found, skipping");
            return;
        };

        let engine = LladaMoeEngine::load(&dir).unwrap();
        let mask_id = engine.config.mask_token_id;

        for seq in [32, 64, 128] {
            let input: Vec<u32> = (0..seq).map(|i| if i < 5 { 1000 } else { mask_id }).collect();
            let _ = engine.forward(&input); // warmup

            let n = 3;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                let _ = engine.forward(&input);
            }
            let ms = t0.elapsed().as_millis() as f64 / n as f64;
            let steps = 64;
            let gen_tok = seq - 5;
            let tps = gen_tok as f64 / (ms * steps as f64 / 1000.0);
            eprintln!("seq={seq:>4}: {ms:>6.1}ms/fwd × {steps} steps → {tps:>5.1} tok/s ({gen_tok} tokens)");
        }
    }

    /// ANE hybrid engine at seq=128: compile, forward, compare to BLAS, benchmark.
    #[test]
    #[cfg(feature = "ane")]
    fn test_ane_hybrid() {
        let Some(dir) = model_dir() else {
            eprintln!("LLaDA-MoE weights not found, skipping");
            return;
        };

        let engine = LladaMoeEngine::load(&dir).unwrap();
        let mask_id = engine.config.mask_token_id;
        let prompt: Vec<u32> = vec![678, 7706, 300, 11406, 341]; // "The capital of France is"

        // seq=128 — ANE sweet spot (4.4x over BLAS for attention alone)
        let seq = 128;
        let input: Vec<u32> = (0..seq).map(|i| if i < 5 { prompt[i] } else { mask_id }).collect();

        // BLAS forward for reference
        let blas_logits = engine.forward(&input);
        eprintln!("BLAS forward done (seq={seq}): {} logits", blas_logits.len());

        // Build ANE hybrid engine at seq=128
        let ane = super::AneLladaMoeEngine::new(engine, seq).unwrap();

        // ANE hybrid forward
        let ane_logits = ane.forward(&input);
        eprintln!("ANE hybrid forward done (seq={seq}): {} logits", ane_logits.len());

        // Compare logits — ANE uses fp16 for attention, so some error expected
        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f32;
        let n = blas_logits.len();
        for i in 0..n {
            let err = (blas_logits[i] - ane_logits[i]).abs();
            max_err = max_err.max(err);
            sum_err += err;
        }
        let mean_err = sum_err / n as f32;
        eprintln!("Logit error vs BLAS: max={max_err:.4}, mean={mean_err:.6}");
        assert!(max_err < 50.0, "Max error too large: {max_err}");

        // Benchmark sweep: BLAS vs ANE hybrid at seq=128
        // Warmup
        let _ = ane.blas_engine.forward(&input);
        let _ = ane.forward(&input);

        let n_runs = 3;
        let blas_t0 = std::time::Instant::now();
        for _ in 0..n_runs { let _ = ane.blas_engine.forward(&input); }
        let blas_ms = blas_t0.elapsed().as_millis() as f64 / n_runs as f64;

        let ane_t0 = std::time::Instant::now();
        for _ in 0..n_runs { let _ = ane.forward(&input); }
        let ane_ms = ane_t0.elapsed().as_millis() as f64 / n_runs as f64;

        let speedup = blas_ms / ane_ms;
        let steps = 64;
        let gen_tok = seq - 5;
        let blas_tps = gen_tok as f64 / (blas_ms * steps as f64 / 1000.0);
        let ane_tps = gen_tok as f64 / (ane_ms * steps as f64 / 1000.0);
        eprintln!("seq={seq}: BLAS {blas_ms:.0}ms ({blas_tps:.1} tok/s) | ANE hybrid {ane_ms:.0}ms ({ane_tps:.1} tok/s) | speedup {speedup:.2}x");
    }

    /// ANE hybrid at seq=1024: attention should dominate, ANE wins big.
    #[test]
    #[cfg(feature = "ane")]
    fn test_ane_hybrid_1024() {
        let Some(dir) = model_dir() else {
            eprintln!("LLaDA-MoE weights not found, skipping");
            return;
        };

        let engine = LladaMoeEngine::load(&dir).unwrap();
        let mask_id = engine.config.mask_token_id;
        let prompt: Vec<u32> = vec![678, 7706, 300, 11406, 341];

        let seq = 1024;
        let input: Vec<u32> = (0..seq).map(|i| if i < 5 { prompt[i] } else { mask_id }).collect();

        // BLAS baseline
        let blas_t0 = std::time::Instant::now();
        let blas_logits = engine.forward(&input);
        let blas_ms = blas_t0.elapsed().as_millis();
        eprintln!("BLAS forward (seq={seq}): {}ms, {} logits", blas_ms, blas_logits.len());

        // ANE hybrid
        let ane = super::AneLladaMoeEngine::new(engine, seq).unwrap();

        // Warmup
        let _ = ane.forward(&input);

        let ane_t0 = std::time::Instant::now();
        let ane_logits = ane.forward(&input);
        let ane_ms = ane_t0.elapsed().as_millis();

        // Error check
        let mut max_err = 0.0f32;
        for i in 0..blas_logits.len() {
            max_err = max_err.max((blas_logits[i] - ane_logits[i]).abs());
        }
        eprintln!("Logit error: max={max_err:.4}");

        let speedup = blas_ms as f64 / ane_ms as f64;
        let steps = 64;
        let gen_tok = seq - 5;
        let blas_tps = gen_tok as f64 / (blas_ms as f64 * steps as f64 / 1000.0);
        let ane_tps = gen_tok as f64 / (ane_ms as f64 * steps as f64 / 1000.0);
        eprintln!("seq={seq}: BLAS {blas_ms}ms ({blas_tps:.2} tok/s) | ANE hybrid {ane_ms}ms ({ane_tps:.2} tok/s) | speedup {speedup:.2}x");
    }
}
