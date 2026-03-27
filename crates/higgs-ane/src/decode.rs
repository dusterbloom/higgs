//! Single-token autoregressive decode with KV cache.
//!
//! CPU-only baseline using Accelerate SGEMM for matmuls.
//! Ported from nanobot-rs ane_decode.rs.

use crate::config::ModelConfig;
use crate::weights::ModelWeights;

// ---------------------------------------------------------------------------
// CPU math primitives
// ---------------------------------------------------------------------------

/// RMSNorm: out = x * w / sqrt(mean(x^2) + eps)
fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], dim: usize, eps: f32) {
    let mut ss = 0.0f32;
    for i in 0..dim {
        ss += x[i] * x[i];
    }
    ss = 1.0 / (ss / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * ss * w[i];
    }
}

// ---------------------------------------------------------------------------
// Accelerate SGEMM (Apple BLAS on AMX)
// ---------------------------------------------------------------------------
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

/// Matrix-vector multiply: out[m] = W[m,k] @ x[k]
/// W is row-major [m, k], x is [k], returns [m].
/// Uses Accelerate cblas_sgemm on macOS (GEMM with n=1).
fn cpu_matvec(w: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            101, // CblasRowMajor
            111, // CblasNoTrans (A)
            111, // CblasNoTrans (B)
            m as i32, 1, k as i32,
            1.0, w.as_ptr(), k as i32,
            x.as_ptr(), 1,
            0.0, out.as_mut_ptr(), 1,
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            let row = &w[i * k..(i + 1) * k];
            let mut dot = 0.0f32;
            for j in 0..k {
                dot += row[j] * x[j];
            }
            out[i] = dot;
        }
    }
    out
}

fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

fn vec_add_inplace(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// KV cache for autoregressive decode.
pub struct KvCache {
    /// K cache per layer: `[n_kv_heads * max_seq * head_dim]`
    k: Vec<Vec<f32>>,
    /// V cache per layer: `[n_kv_heads * max_seq * head_dim]`
    v: Vec<Vec<f32>>,
    /// Current position (next token written here).
    pos: usize,
    max_seq: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    pub fn new(cfg: &ModelConfig, n_layers: usize, max_seq: usize) -> Self {
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let layer_size = n_kv_heads * max_seq * head_dim;
        Self {
            k: vec![vec![0.0f32; layer_size]; n_layers],
            v: vec![vec![0.0f32; layer_size]; n_layers],
            pos: 0,
            max_seq,
            n_kv_heads,
            head_dim,
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Append K, V for one token at one layer.
    fn append(&mut self, layer: usize, k_new: &[f32], v_new: &[f32]) {
        let hd = self.head_dim;
        let pos = self.pos;
        for kv_h in 0..self.n_kv_heads {
            let base = kv_h * self.max_seq * hd + pos * hd;
            self.k[layer][base..base + hd]
                .copy_from_slice(&k_new[kv_h * hd..(kv_h + 1) * hd]);
            self.v[layer][base..base + hd]
                .copy_from_slice(&v_new[kv_h * hd..(kv_h + 1) * hd]);
        }
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    /// Rollback to a previous position (for speculation rejection).
    pub fn rollback_to(&mut self, new_pos: usize) {
        self.pos = new_pos;
    }
}

// ---------------------------------------------------------------------------
// RoPE at single position
// ---------------------------------------------------------------------------

fn rope_at_pos(
    q: &mut [f32],
    k: &mut [f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f64,
) {
    let half_hd = head_dim / 2;
    for h in 0..n_q_heads {
        let base = h * head_dim;
        for i in 0..half_hd {
            let inv_freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            let cos = angle.cos() as f32;
            let sin = angle.sin() as f32;
            let r = q[base + i];
            let c = q[base + half_hd + i];
            q[base + i] = r * cos - c * sin;
            q[base + half_hd + i] = r * sin + c * cos;
        }
    }
    for h in 0..n_kv_heads {
        let base = h * head_dim;
        for i in 0..half_hd {
            let inv_freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            let cos = angle.cos() as f32;
            let sin = angle.sin() as f32;
            let r = k[base + i];
            let c = k[base + half_hd + i];
            k[base + i] = r * cos - c * sin;
            k[base + half_hd + i] = r * sin + c * cos;
        }
    }
}

// ---------------------------------------------------------------------------
// SDPA with KV cache
// ---------------------------------------------------------------------------

fn sdpa_cached(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    n_q_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n_kv_heads = cache.n_kv_heads;
    let cache_len = cache.pos;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let gqa_ratio = n_q_heads / n_kv_heads;

    let mut out = vec![0.0f32; n_q_heads * head_dim];

    for qh in 0..n_q_heads {
        let kv_h = qh / gqa_ratio;
        let q_off = qh * head_dim;
        let q_vec = &q[q_off..q_off + head_dim];
        let k_base = kv_h * cache.max_seq * head_dim;

        // Scores: q @ k[t]^T
        let mut scores = vec![0.0f32; cache_len];
        for t in 0..cache_len {
            let k_off = k_base + t * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_vec[d] * cache.k[layer][k_off + d];
            }
            scores[t] = dot * scale;
        }

        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }

        // Weighted sum of V
        let out_h = &mut out[q_off..q_off + head_dim];
        for t in 0..cache_len {
            let v_off = k_base + t * head_dim;
            let w = scores[t];
            for d in 0..head_dim {
                out_h[d] += w * cache.v[layer][v_off + d];
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Decode step
// ---------------------------------------------------------------------------

/// Result of a single decode step.
pub struct DecodeResult {
    pub logits: Vec<f32>,
}

/// Single-token decode: embed → layers → norm → classifier.
pub fn decode_step(model: &ModelWeights, token: u32, kv_cache: &mut KvCache) -> DecodeResult {
    let cfg = &model.cfg;
    let dim = cfg.dim;
    let n_layers = model.layers.len();
    let n_q_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let pos = kv_cache.pos();

    // 1. Embedding
    let mut x = vec![0.0f32; dim];
    let embed_off = token as usize * dim;
    x.copy_from_slice(&model.embed[embed_off..embed_off + dim]);

    // 2. Layers
    for l in 0..n_layers {
        let lw = &model.layers[l];

        // Attention norm
        let mut xnorm = vec![0.0f32; dim];
        rmsnorm(&mut xnorm, &x, &lw.rms_att, dim, cfg.rms_eps);

        // Q, K, V projections
        let q_proj_dim = cfg.q_proj_dim();
        let q_raw = cpu_matvec(&lw.wq, &xnorm, q_proj_dim, dim);
        let k_full = cpu_matvec(&lw.wk, &xnorm, lw.wk.len() / dim, dim);
        let v_full = cpu_matvec(&lw.wv, &xnorm, lw.wv.len() / dim, dim);

        // Collapse expanded KV heads back to kv_dim
        let k_proj_dim = lw.wk.len() / dim;
        let mut k = if k_proj_dim > kv_dim {
            let hpg = k_proj_dim / kv_dim;
            let mut collapsed = vec![0.0f32; kv_dim];
            for h in 0..n_kv_heads {
                let src = h * hpg * head_dim;
                let dst = h * head_dim;
                collapsed[dst..dst + head_dim].copy_from_slice(&k_full[src..src + head_dim]);
            }
            collapsed
        } else {
            k_full
        };
        let v_proj_dim = lw.wv.len() / dim;
        let v = if v_proj_dim > kv_dim {
            let hpg = v_proj_dim / kv_dim;
            let mut collapsed = vec![0.0f32; kv_dim];
            for h in 0..n_kv_heads {
                let src = h * hpg * head_dim;
                let dst = h * head_dim;
                collapsed[dst..dst + head_dim].copy_from_slice(&v_full[src..src + head_dim]);
            }
            collapsed
        } else {
            v_full
        };

        // Split Q and gate (attn_output_gate)
        let attn_dim = n_q_heads * head_dim;
        let (mut q, attn_gate) = if cfg.attn_output_gate {
            let mut q = vec![0.0f32; attn_dim];
            let mut gate = vec![0.0f32; attn_dim];
            for h in 0..n_q_heads {
                let src = h * 2 * head_dim;
                let dst = h * head_dim;
                q[dst..dst + head_dim].copy_from_slice(&q_raw[src..src + head_dim]);
                gate[dst..dst + head_dim]
                    .copy_from_slice(&q_raw[src + head_dim..src + 2 * head_dim]);
            }
            (q, Some(gate))
        } else {
            (q_raw, None)
        };

        // QKNorm: per-head RMSNorm on Q and K before RoPE (Qwen3)
        if let Some(ref qn) = lw.q_norm {
            for h in 0..n_q_heads {
                let off = h * head_dim;
                let mut tmp = vec![0.0f32; head_dim];
                rmsnorm(&mut tmp, &q[off..off + head_dim], qn, head_dim, cfg.rms_eps);
                q[off..off + head_dim].copy_from_slice(&tmp);
            }
        }
        if let Some(ref kn) = lw.k_norm {
            for h in 0..n_kv_heads {
                let off = h * head_dim;
                let mut tmp = vec![0.0f32; head_dim];
                rmsnorm(&mut tmp, &k[off..off + head_dim], kn, head_dim, cfg.rms_eps);
                k[off..off + head_dim].copy_from_slice(&tmp);
            }
        }

        // RoPE
        rope_at_pos(&mut q, &mut k, n_q_heads, n_kv_heads, head_dim, pos, cfg.rope_theta);

        // Append to KV cache
        kv_cache.append(l, &k, &v);
        let save_pos = kv_cache.pos;
        kv_cache.pos = pos + 1;

        // SDPA
        let mut attn_out = sdpa_cached(&q, kv_cache, l, n_q_heads, head_dim);
        kv_cache.pos = save_pos;

        // Apply attention output gate
        if let Some(ref gate) = attn_gate {
            for i in 0..attn_dim {
                let sig = 1.0 / (1.0 + (-gate[i]).exp());
                attn_out[i] *= sig;
            }
        }

        // Output projection
        let o = cpu_matvec(&lw.wo, &attn_out, dim, attn_dim);
        vec_add_inplace(&mut x, &o);

        // FFN
        let mut x2norm = vec![0.0f32; dim];
        rmsnorm(&mut x2norm, &x, &lw.rms_ffn, dim, cfg.rms_eps);
        let hidden = cfg.hidden_dim;
        let mut h1 = cpu_matvec(&lw.w1, &x2norm, hidden, dim);
        let h3 = cpu_matvec(&lw.w3, &x2norm, hidden, dim);
        silu_inplace(&mut h1);
        for i in 0..hidden {
            h1[i] *= h3[i];
        }
        let ffn_out = cpu_matvec(&lw.w2, &h1, dim, hidden);
        vec_add_inplace(&mut x, &ffn_out);
    }

    // Advance KV cache
    kv_cache.advance();

    // 3. Final norm
    let mut x_final = vec![0.0f32; dim];
    rmsnorm(&mut x_final, &x, &model.rms_final, dim, cfg.rms_eps);

    // 4. Classifier: logits = embed^T @ x_final (tied embeddings)
    let logits = cpu_matvec(&model.embed, &x_final, model.vocab_size, dim);

    DecodeResult { logits }
}
