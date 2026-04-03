//! LoRA adapters and training backward pass for MDLM diffusion model.
//!
//! Row-major [seq, dim] layout throughout (matching higgs DiffusionEngine).
//! Phase 1: CPU backward via Accelerate BLAS. Phase 2: ANE backward kernels.
//!
//! Ported from nanobot-rs ane_lora.rs / ane_backward.rs, adapted for:
//! - Row-major layout (nanobot-rs uses channel-first [dim, seq])
//! - Bidirectional attention (nanobot-rs uses causal)
//! - MDLM diffusion loss (nanobot-rs uses autoregressive CE)
//! - GQA with 16 Q heads / 8 KV heads (nanobot-rs uses MHA)

#![allow(clippy::too_many_arguments, unsafe_code)]

use crate::diffusion::{
    apply_rope, rms_norm, rms_norm_slice, sgemm, sgemm_acc, sgemm_at, sgemm_at_acc,
    sgemm_nt, sgemm_nt_scaled, softmax_inplace, DiffusionConfig, DiffusionEngine,
};

// ---------------------------------------------------------------------------
// LoRA config
// ---------------------------------------------------------------------------

/// LoRA configuration with JIT LoRA proven defaults (from nanobot-rs).
#[derive(Clone)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    /// Which projections to adapt: "q", "v", "o", "down".
    pub targets: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        LoraConfig {
            rank: 32,
            alpha: 32.0,
            targets: vec!["q".into(), "v".into(), "o".into(), "down".into()],
        }
    }
}

impl LoraConfig {
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

// ---------------------------------------------------------------------------
// LoRA adapter
// ---------------------------------------------------------------------------

/// Single LoRA adapter: δW ≈ B @ A where A∈R^{rank×d_in}, B∈R^{d_out×rank}.
///
/// Row-major storage (matching all higgs BLAS operations):
///   A: [rank, d_in] — A[r * d_in + i]
///   B: [d_out, rank] — B[o * rank + r]
///
/// Forward (row-major): h = x @ A^T, δy = h @ B^T (using sgemm_nt).
/// B is zero-initialized so the model starts unmodified.
#[derive(Clone)]
pub struct LoraAdapter {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub d_in: usize,
    pub d_out: usize,
}

impl LoraAdapter {
    /// Create with Kaiming uniform init for A, zeros for B.
    pub fn new(rank: usize, d_in: usize, d_out: usize) -> Self {
        let bound = (6.0f32 / d_in as f32).sqrt();
        let a: Vec<f32> = (0..rank * d_in)
            .map(|i| ((i as f32 * 0.618033988 + 0.31415926).fract() * 2.0 - 1.0) * bound)
            .collect();
        let b = vec![0.0f32; d_out * rank];
        LoraAdapter { a, b, rank, d_in, d_out }
    }

    /// LoRA forward (row-major): x[seq, d_in] → δy[seq, d_out], h[seq, rank].
    pub fn forward(&self, x: &[f32], seq: usize) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0f32; seq * self.rank];
        sgemm_nt(seq, self.rank, self.d_in, x, &self.a, &mut h);
        let mut dy = vec![0.0f32; seq * self.d_out];
        sgemm_nt(seq, self.d_out, self.rank, &h, &self.b, &mut dy);
        (dy, h)
    }

    /// Total trainable parameters.
    pub fn num_params(&self) -> usize {
        self.rank * self.d_in + self.d_out * self.rank
    }
}

// ---------------------------------------------------------------------------
// Per-layer LoRA adapters
// ---------------------------------------------------------------------------

/// LoRA adapters for one transformer layer.
pub struct LoraLayerAdapters {
    pub q: Option<LoraAdapter>,
    pub v: Option<LoraAdapter>,
    pub o: Option<LoraAdapter>,
    pub down: Option<LoraAdapter>,
}

// ---------------------------------------------------------------------------
// Full LoRA model
// ---------------------------------------------------------------------------

pub struct DiffusionLoraModel {
    pub layers: Vec<LoraLayerAdapters>,
    pub config: LoraConfig,
}

impl DiffusionLoraModel {
    /// Initialize LoRA adapters for all layers.
    pub fn new(cfg: LoraConfig, diff_cfg: &DiffusionConfig) -> Self {
        let rank = cfg.rank;
        let h = diff_cfg.hidden;
        let q_dim = diff_cfg.heads * diff_cfg.head_dim;
        let kv_dim = diff_cfg.kv_heads * diff_cfg.head_dim;
        let inter = diff_cfg.inter;

        let layers = (0..diff_cfg.layers)
            .map(|_| LoraLayerAdapters {
                q: if cfg.targets.iter().any(|t| t == "q") {
                    Some(LoraAdapter::new(rank, h, q_dim))
                } else {
                    None
                },
                v: if cfg.targets.iter().any(|t| t == "v") {
                    Some(LoraAdapter::new(rank, h, kv_dim))
                } else {
                    None
                },
                o: if cfg.targets.iter().any(|t| t == "o") {
                    Some(LoraAdapter::new(rank, q_dim, h))
                } else {
                    None
                },
                down: if cfg.targets.iter().any(|t| t == "down") {
                    Some(LoraAdapter::new(rank, inter, h))
                } else {
                    None
                },
            })
            .collect();

        DiffusionLoraModel { layers, config: cfg }
    }

    pub fn scale(&self) -> f32 {
        self.config.scale()
    }

    /// Total trainable parameters across all layers.
    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|l| {
            l.q.as_ref().map_or(0, |a| a.num_params())
                + l.v.as_ref().map_or(0, |a| a.num_params())
                + l.o.as_ref().map_or(0, |a| a.num_params())
                + l.down.as_ref().map_or(0, |a| a.num_params())
        }).sum()
    }
}

// ---------------------------------------------------------------------------
// Saved activations for backward pass
// ---------------------------------------------------------------------------

/// Per-layer activations saved during forward for backward.
pub struct LayerActivations {
    pub hidden_in: Vec<f32>,       // [seq, h] input to this layer
    pub normed_attn: Vec<f32>,     // [seq, h] after input_norm
    pub q_pre_norm: Vec<f32>,      // [seq, q_dim] Q before QK-norm (for QK-norm bwd)
    pub k_pre_norm: Vec<f32>,      // [seq, kv_dim] K before QK-norm
    pub q_final: Vec<f32>,         // [seq, q_dim] Q after norm+RoPE (for SDPA bwd)
    pub k_final: Vec<f32>,         // [seq, kv_dim] K after norm+RoPE
    pub v: Vec<f32>,               // [seq, kv_dim]
    pub attn_probs: Vec<Vec<f32>>, // n_heads × [seq*seq] per-head softmax output
    pub attn_out: Vec<f32>,        // [seq, q_dim] attention output before O-proj
    pub normed_ffn: Vec<f32>,      // [seq, h] after post_attn_norm
    pub gate_pre_silu: Vec<f32>,   // [seq, inter] gate before SiLU
    pub up_out: Vec<f32>,          // [seq, inter] up proj output
    pub gate_times_up: Vec<f32>,   // [seq, inter] SiLU(gate) * up = input to down_proj
    // LoRA intermediates (saved for weight gradient computation)
    pub lora_h_q: Option<Vec<f32>>,    // [seq, rank]
    pub lora_h_v: Option<Vec<f32>>,
    pub lora_h_o: Option<Vec<f32>>,
    pub lora_h_down: Option<Vec<f32>>,
}

/// All activations from a training forward pass.
pub struct DiffusionActivations {
    pub token_ids: Vec<u32>,
    pub layer_acts: Vec<LayerActivations>,
    pub hidden_out: Vec<f32>,   // [seq, h] final layer output (before final norm)
    pub final_normed: Vec<f32>, // [seq, h] after final RMSNorm
}

// ---------------------------------------------------------------------------
// LoRA gradients
// ---------------------------------------------------------------------------

pub struct LoraAdapterGrads {
    pub da: Vec<f32>, // [rank, d_in]
    pub db: Vec<f32>, // [d_out, rank]
}

impl LoraAdapterGrads {
    pub fn zeros(adapter: &LoraAdapter) -> Self {
        LoraAdapterGrads {
            da: vec![0.0; adapter.rank * adapter.d_in],
            db: vec![0.0; adapter.d_out * adapter.rank],
        }
    }
}

pub struct LoraLayerGrads {
    pub q: Option<LoraAdapterGrads>,
    pub v: Option<LoraAdapterGrads>,
    pub o: Option<LoraAdapterGrads>,
    pub down: Option<LoraAdapterGrads>,
}

pub struct DiffusionLoraGrads {
    pub layers: Vec<LoraLayerGrads>,
}

// ---------------------------------------------------------------------------
// Forward pass with activation saving
// ---------------------------------------------------------------------------

/// Forward pass through base model + LoRA, saving all activations for backward.
///
/// Returns logits [seq, vocab] and saved activations.
pub fn forward_train(
    engine: &DiffusionEngine,
    lora: &DiffusionLoraModel,
    token_ids: &[u32],
) -> (Vec<f32>, DiffusionActivations) {
    let cfg = &engine.config;
    let seq = token_ids.len();
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let half_hd = hd / 2;
    let n_heads = cfg.heads;
    let n_kv = cfg.kv_heads;
    let gqa_ratio = n_heads / n_kv;
    let q_dim = n_heads * hd;
    let kv_dim = n_kv * hd;
    let scale = lora.scale();

    // 1. Embedding lookup
    let mut hidden = vec![0.0f32; seq * h];
    for (i, &tid) in token_ids.iter().enumerate() {
        let offset = tid as usize * h;
        hidden[i * h..(i + 1) * h].copy_from_slice(&engine.embed[offset..offset + h]);
    }

    // Scratch buffers
    let mut q_buf = vec![0.0f32; seq * q_dim];
    let mut k_buf = vec![0.0f32; seq * kv_dim];
    let mut v_buf = vec![0.0f32; seq * kv_dim];
    let mut attn_out = vec![0.0f32; seq * q_dim];
    let mut o_buf = vec![0.0f32; seq * h];
    let mut gate_buf = vec![0.0f32; seq * cfg.inter];
    let mut up_buf = vec![0.0f32; seq * cfg.inter];
    let mut normed = vec![0.0f32; seq * h];

    let mut layer_acts = Vec::with_capacity(cfg.layers);

    // 2. Layer loop
    for (li, layer) in engine.layers.iter().enumerate() {
        let hidden_in = hidden.clone();
        let lora_layer = &lora.layers[li];

        // --- Attention ---
        rms_norm(&hidden, &layer.input_norm, &mut normed, seq, h);
        let normed_attn = normed.clone();

        // QKV projections: normed[seq,h] @ W^T → buf
        sgemm_nt(seq, q_dim, h, &normed, &layer.q_proj, &mut q_buf);
        sgemm_nt(seq, kv_dim, h, &normed, &layer.k_proj, &mut k_buf);
        sgemm_nt(seq, kv_dim, h, &normed, &layer.v_proj, &mut v_buf);

        // LoRA on Q
        let lora_h_q = if let Some(lora_q) = &lora_layer.q {
            let (dy, hh) = lora_q.forward(&normed, seq);
            for i in 0..seq * q_dim { q_buf[i] += scale * dy[i]; }
            Some(hh)
        } else {
            None
        };

        // LoRA on V
        let lora_h_v = if let Some(lora_v) = &lora_layer.v {
            let (dy, hh) = lora_v.forward(&normed, seq);
            for i in 0..seq * kv_dim { v_buf[i] += scale * dy[i]; }
            Some(hh)
        } else {
            None
        };

        // Save Q, K before QK-norm (needed for QK-norm backward)
        let q_pre_norm = q_buf.clone();
        let k_pre_norm = k_buf.clone();

        // QK norm (per-head RMSNorm)
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

        // RoPE
        for s in 0..seq {
            for head in 0..n_heads {
                let off = s * q_dim + head * hd;
                apply_rope(&mut q_buf[off..off + hd], s, half_hd, &engine.rope_cos, &engine.rope_sin);
            }
            for head in 0..n_kv {
                let off = s * kv_dim + head * hd;
                apply_rope(&mut k_buf[off..off + hd], s, half_hd, &engine.rope_cos, &engine.rope_sin);
            }
        }

        // Save Q, K after norm+RoPE and V (for SDPA backward)
        let q_final = q_buf.clone();
        let k_final = k_buf.clone();
        let v_saved = v_buf.clone();

        // Bidirectional SDPA
        let attn_scale = 1.0 / (hd as f32).sqrt();
        let mut all_probs: Vec<Vec<f32>> = Vec::with_capacity(n_heads);
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
                sgemm_nt_scaled(seq, seq, hd, &q_head, &k_head, &mut scores, attn_scale);

                for row in 0..seq {
                    softmax_inplace(&mut scores[row * seq..(row + 1) * seq]);
                }

                // Save attention probabilities for backward
                all_probs.push(scores.clone());

                let mut ctx = vec![0.0f32; seq * hd];
                sgemm(seq, hd, seq, &scores, &v_head, &mut ctx);

                for s in 0..seq {
                    let ao = s * q_dim + q_h * hd;
                    attn_out[ao..ao + hd].copy_from_slice(&ctx[s * hd..(s + 1) * hd]);
                }
            }
        }

        let attn_out_saved = attn_out.clone();

        // O projection: attn_out[seq, q_dim] @ o_proj^T → o_buf[seq, h]
        sgemm_nt(seq, h, q_dim, &attn_out, &layer.o_proj, &mut o_buf);

        // LoRA on O
        let lora_h_o = if let Some(lora_o) = &lora_layer.o {
            let (dy, hh) = lora_o.forward(&attn_out, seq);
            for i in 0..seq * h { o_buf[i] += scale * dy[i]; }
            Some(hh)
        } else {
            None
        };

        // Residual add
        for i in 0..seq * h { hidden[i] += o_buf[i]; }

        // --- MLP ---
        rms_norm(&hidden, &layer.post_attn_norm, &mut normed, seq, h);
        let normed_ffn = normed.clone();

        sgemm_nt(seq, cfg.inter, h, &normed, &layer.gate_proj, &mut gate_buf);
        let gate_pre_silu = gate_buf.clone();

        // SiLU(gate)
        for v in gate_buf.iter_mut() { *v *= 1.0 / (1.0 + (-*v).exp()); }

        sgemm_nt(seq, cfg.inter, h, &normed, &layer.up_proj, &mut up_buf);
        let up_out = up_buf.clone();

        // gate * up
        for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) { *g *= u; }
        let gate_times_up = gate_buf.clone();

        // down = gate_buf @ down_proj^T → [seq, h]
        sgemm_nt(seq, h, cfg.inter, &gate_buf, &layer.down_proj, &mut o_buf);

        // LoRA on down_proj
        let lora_h_down = if let Some(lora_down) = &lora_layer.down {
            let (dy, hh) = lora_down.forward(&gate_buf, seq);
            for i in 0..seq * h { o_buf[i] += scale * dy[i]; }
            Some(hh)
        } else {
            None
        };

        // Residual add
        for i in 0..seq * h { hidden[i] += o_buf[i]; }

        layer_acts.push(LayerActivations {
            hidden_in,
            normed_attn,
            q_pre_norm,
            k_pre_norm,
            q_final,
            k_final,
            v: v_saved,
            attn_probs: all_probs,
            attn_out: attn_out_saved,
            normed_ffn,
            gate_pre_silu,
            up_out,
            gate_times_up,
            lora_h_q,
            lora_h_v,
            lora_h_o,
            lora_h_down,
        });
    }

    // 3. Final RMSNorm
    let hidden_out = hidden.clone();
    let mut final_normed = vec![0.0f32; seq * h];
    rms_norm(&hidden, &engine.final_norm, &mut final_normed, seq, h);

    // 4. LM head: normed[seq, h] @ embed^T → logits[seq, vocab]
    let mut logits = vec![0.0f32; seq * cfg.vocab];
    sgemm_nt(seq, cfg.vocab, h, &final_normed, &engine.embed, &mut logits);

    let acts = DiffusionActivations {
        token_ids: token_ids.to_vec(),
        layer_acts,
        hidden_out,
        final_normed,
    };

    (logits, acts)
}

// ---------------------------------------------------------------------------
// Backward primitives
// ---------------------------------------------------------------------------

/// RMSNorm backward (row-major [seq, dim]).
///
/// Given dy[seq, dim], x[seq, dim], w[dim]:
/// Returns dx[seq, dim], accumulates dw[dim].
fn rmsnorm_bwd(
    dx: &mut [f32],
    dw: &mut [f32],
    dy: &[f32],
    x: &[f32],
    w: &[f32],
    seq: usize,
    dim: usize,
) {
    let eps = 1e-6f32;
    let inv_dim = 1.0 / dim as f32;
    for s in 0..seq {
        let row_x = &x[s * dim..(s + 1) * dim];
        let row_dy = &dy[s * dim..(s + 1) * dim];

        let ss: f32 = row_x.iter().map(|v| v * v).sum();
        let rrms = 1.0 / (ss * inv_dim + eps).sqrt();

        let mut dot = 0.0f32;
        for d in 0..dim {
            dot += row_dy[d] * row_x[d] * w[d];
        }
        dot *= rrms * rrms * inv_dim;

        for d in 0..dim {
            dx[s * dim + d] = rrms * (w[d] * row_dy[d] - row_x[d] * dot);
            dw[d] += row_dy[d] * row_x[d] * rrms;
        }
    }
}

/// Per-head RMSNorm backward (for QK-norm).
///
/// x_pre[hd]: pre-norm input, dy[hd]: upstream gradient, w[hd]: norm weight.
/// Returns dx[hd] (in-place into dx_out).
fn qk_norm_bwd_head(dx_out: &mut [f32], dy: &[f32], x_pre: &[f32], w: &[f32], hd: usize) {
    let eps = 1e-6f32;
    let inv_dim = 1.0 / hd as f32;
    let ss: f32 = x_pre.iter().map(|v| v * v).sum();
    let rrms = 1.0 / (ss * inv_dim + eps).sqrt();

    let mut dot = 0.0f32;
    for d in 0..hd {
        dot += dy[d] * x_pre[d] * w[d];
    }
    dot *= rrms * rrms * inv_dim;

    for d in 0..hd {
        dx_out[d] = rrms * (w[d] * dy[d] - x_pre[d] * dot);
    }
}

/// RoPE backward: inverse rotation.
///
/// Forward: y[d] = x[d]*cos - x[d+half]*sin, y[d+half] = x[d]*sin + x[d+half]*cos
/// Backward: dx[d] = dy[d]*cos + dy[d+half]*sin, dx[d+half] = -dy[d]*sin + dy[d+half]*cos
fn rope_bwd(dx: &mut [f32], dy: &[f32], pos: usize, half_dim: usize, cos: &[f32], sin: &[f32]) {
    let co = pos * half_dim;
    for d in 0..half_dim {
        let c = cos[co + d];
        let s = sin[co + d];
        dx[d] = dy[d] * c + dy[d + half_dim] * s;
        dx[d + half_dim] = -dy[d] * s + dy[d + half_dim] * c;
    }
}

/// Softmax backward: ds[i] = p[i] * (dp[i] - sum_j(dp[j]*p[j])).
fn softmax_bwd(ds: &mut [f32], dp: &[f32], p: &[f32], n: usize) {
    let dot: f32 = (0..n).map(|j| dp[j] * p[j]).sum();
    for i in 0..n {
        ds[i] = p[i] * (dp[i] - dot);
    }
}

/// SiLU backward.
///
/// Forward: gate_out = silu(gate) * up = gate*sigmoid(gate) * up
/// Given d_gate_times_up, gate_pre_silu, up:
/// Returns d_gate_pre_silu and d_up.
fn silu_gate_bwd(
    d_gate: &mut [f32],
    d_up: &mut [f32],
    d_out: &[f32],
    gate_pre_silu: &[f32],
    up: &[f32],
    n: usize,
) {
    for i in 0..n {
        let sig = 1.0 / (1.0 + (-gate_pre_silu[i]).exp());
        let silu_val = gate_pre_silu[i] * sig;
        d_up[i] = d_out[i] * silu_val;
        let silu_deriv = sig * (1.0 + gate_pre_silu[i] * (1.0 - sig));
        d_gate[i] = d_out[i] * up[i] * silu_deriv;
    }
}

/// LoRA backward at one injection point.
///
/// Given upstream grad dy[seq, d_out], saved input x[seq, d_in], saved h[seq, rank]:
/// - Computes dx_lora contribution (accumulated into dx)
/// - Returns (dA, dB) weight gradients.
fn lora_bwd(
    dx: &mut [f32],
    dy: &[f32],
    x: &[f32],
    h: &[f32],
    adapter: &LoraAdapter,
    scale: f32,
    seq: usize,
) -> LoraAdapterGrads {
    let rank = adapter.rank;
    let d_in = adapter.d_in;
    let d_out = adapter.d_out;

    // dh = scale * dy @ B: [seq, d_out] @ [d_out, rank] → [seq, rank]
    let mut dh = vec![0.0f32; seq * rank];
    sgemm(seq, rank, d_out, dy, &adapter.b, &mut dh);
    for v in dh.iter_mut() { *v *= scale; }

    // dx_lora = dh @ A: [seq, rank] @ [rank, d_in] → [seq, d_in] (accumulated into dx)
    sgemm_acc(seq, d_in, rank, &dh, &adapter.a, dx);

    // dB = scale * dy^T @ h: [d_out, seq] @ [seq, rank] → [d_out, rank]
    let mut db = vec![0.0f32; d_out * rank];
    sgemm_at(d_out, rank, seq, dy, h, &mut db);
    for v in db.iter_mut() { *v *= scale; }

    // dA = dh^T @ x: [rank, seq] @ [seq, d_in] → [rank, d_in]
    let mut da = vec![0.0f32; rank * d_in];
    sgemm_at(rank, d_in, seq, &dh, x, &mut da);

    LoraAdapterGrads { da, db }
}

// ---------------------------------------------------------------------------
// Full backward pass
// ---------------------------------------------------------------------------

/// Full backward: logit gradients → LoRA weight gradients.
///
/// `d_logits`: [seq, vocab] gradient of loss w.r.t. logits.
/// Returns LoRA parameter gradients for all layers.
pub fn backward(
    engine: &DiffusionEngine,
    lora: &DiffusionLoraModel,
    acts: &DiffusionActivations,
    d_logits: &[f32],
) -> DiffusionLoraGrads {
    let cfg = &engine.config;
    let seq = acts.token_ids.len();
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let half_hd = hd / 2;
    let n_heads = cfg.heads;
    let n_kv = cfg.kv_heads;
    let gqa_ratio = n_heads / n_kv;
    let q_dim = n_heads * hd;
    let kv_dim = n_kv * hd;
    let scale = lora.scale();

    // --- LM head backward ---
    // logits = final_normed @ embed^T (tied weights)
    // d_final_normed = d_logits @ embed: [seq, vocab] @ [vocab, h] → [seq, h]
    let mut d_hidden = vec![0.0f32; seq * h];
    sgemm(seq, h, cfg.vocab, d_logits, &engine.embed, &mut d_hidden);

    // --- Final RMSNorm backward ---
    let mut d_pre_norm = vec![0.0f32; seq * h];
    let mut _d_final_norm_w = vec![0.0f32; h]; // not training norm weights
    rmsnorm_bwd(
        &mut d_pre_norm, &mut _d_final_norm_w, &d_hidden,
        &acts.hidden_out, &engine.final_norm, seq, h,
    );
    d_hidden = d_pre_norm;

    // --- Per-layer backward (reverse order) ---
    let mut layer_grads = Vec::with_capacity(cfg.layers);

    for li in (0..cfg.layers).rev() {
        let layer = &engine.layers[li];
        let lora_layer = &lora.layers[li];
        let la = &acts.layer_acts[li];

        // d_hidden is the gradient flowing into this layer's output.
        // The layer has two residual paths:
        //   hidden_out = hidden_in + attn_path + ffn_path
        // So d_hidden flows to both the FFN and attention backward, AND continues
        // as d_hidden_in (the residual).

        // ==================== FFN backward ====================
        // FFN: normed_ffn = rmsnorm(hidden_mid); gate = normed@gate^T; up = normed@up^T;
        //      silu_gate = silu(gate)*up; out = silu_gate@down^T
        // d_hidden flows to d_ffn_out = d_hidden (the FFN residual path)

        // d_gate_times_up = d_hidden @ down_proj: [seq, h] @ [h, inter] → [seq, inter]
        let mut d_gate_times_up = vec![0.0f32; seq * cfg.inter];
        sgemm(seq, cfg.inter, h, &d_hidden, &layer.down_proj, &mut d_gate_times_up);

        // LoRA on down_proj
        let down_grads = if let (Some(adapter), Some(lora_h)) = (&lora_layer.down, &la.lora_h_down) {
            Some(lora_bwd(&mut d_gate_times_up, &d_hidden, &la.gate_times_up, lora_h, adapter, scale, seq))
        } else {
            None
        };

        // SiLU backward
        let mut d_gate_pre_silu = vec![0.0f32; seq * cfg.inter];
        let mut d_up = vec![0.0f32; seq * cfg.inter];
        silu_gate_bwd(
            &mut d_gate_pre_silu, &mut d_up, &d_gate_times_up,
            &la.gate_pre_silu, &la.up_out, seq * cfg.inter,
        );

        // d_normed_ffn from gate: d_gate_pre_silu @ gate_proj: [seq, inter] @ [inter, h] → [seq, h]
        let mut d_normed_ffn = vec![0.0f32; seq * h];
        sgemm(seq, h, cfg.inter, &d_gate_pre_silu, &layer.gate_proj, &mut d_normed_ffn);
        // d_normed_ffn from up: d_up @ up_proj (accumulated)
        sgemm_acc(seq, h, cfg.inter, &d_up, &layer.up_proj, &mut d_normed_ffn);

        // post_attn_norm backward: d_normed_ffn → d_hidden_mid
        let mut d_hidden_mid = vec![0.0f32; seq * h];
        let mut _d_post_norm_w = vec![0.0f32; h];
        // The hidden state before FFN norm = hidden_in + attn_output
        // We need the actual hidden_mid value. It equals layer input + attn residual.
        // But we saved hidden_in (before the layer). The hidden_mid = hidden_in + o_buf.
        // We didn't explicitly save hidden_mid. Let's reconstruct it:
        // hidden_mid = hidden_in + attn_path_output
        // We can compute: normed_ffn was saved, and it's rmsnorm(hidden_mid).
        // For rmsnorm backward, we need the pre-norm input (hidden_mid).
        // We need to reconstruct hidden_mid from what we have.
        // Actually: the gradient d_hidden from the residual skip at the end is split:
        //   hidden = hidden_mid + ffn_output  →  d_hidden_mid = d_hidden (residual copy)
        //                                        d_ffn_out = d_hidden (into FFN backward above)
        // So d_hidden_mid should get BOTH the residual d_hidden AND the FFN norm backward output.
        //
        // Wait, let me be more precise about the residual connections:
        //   hidden_mid = hidden_in + o_buf          (after attention residual)
        //   hidden_out = hidden_mid + ffn_out        (after FFN residual)
        //   d_hidden_out arrives from the next layer / loss
        //   d_hidden_mid = d_hidden_out              (residual: FFN output)
        //                + rmsnorm_bwd(d_normed_ffn) (through FFN norm path)
        //   d_hidden_in  = d_hidden_mid              (residual: attention output)
        //                + input_norm_bwd(d_normed_attn) (through attention path)
        //
        // We need hidden_mid for rmsnorm_bwd. Reconstruct from saved values:
        // We know normed_ffn = rmsnorm(hidden_mid, post_attn_norm).
        // To compute rmsnorm_bwd we need hidden_mid (the pre-norm input).

        // Reconstruct hidden_mid: we didn't save it, but we can derive it.
        // hidden_mid[s] = hidden_in[s] + attn_residual[s]
        // The attn_residual = o_proj(attn_out) + lora_o(attn_out).
        // We'd need to recompute it... OR just save hidden_mid in the forward.
        // For now, let's recompute it from normed_ffn using the norm inverse.
        // Actually that's lossy. Let me just compute it as:
        //   hidden_mid = hidden_in + (the attention block output)
        // The attention block output was added to hidden via the residual.
        // Since hidden_out = hidden_mid + ffn_out, and we know hidden_out...
        // Actually hidden_out is the FINAL hidden after all layers, not this layer's output.
        //
        // OK, simplest fix: save hidden_mid during forward. But I already defined
        // the activations struct... Let me just recompute the O-proj + LoRA output here.

        // Recompute attn residual: o_proj(attn_out) + scale*lora_o(attn_out)
        let mut attn_residual = vec![0.0f32; seq * h];
        sgemm_nt(seq, h, q_dim, &la.attn_out, &layer.o_proj, &mut attn_residual);
        if let (Some(adapter), Some(hh)) = (&lora_layer.o, &la.lora_h_o) {
            let mut dy = vec![0.0f32; seq * h];
            sgemm_nt(seq, h, adapter.rank, hh, &adapter.b, &mut dy);
            for i in 0..seq * h { attn_residual[i] += scale * dy[i]; }
        }
        let hidden_mid: Vec<f32> = la.hidden_in.iter().zip(attn_residual.iter()).map(|(a, b)| a + b).collect();

        rmsnorm_bwd(
            &mut d_hidden_mid, &mut _d_post_norm_w, &d_normed_ffn,
            &hidden_mid, &layer.post_attn_norm, seq, h,
        );

        // Add residual: d_hidden_mid += d_hidden (the skip connection from FFN output)
        for i in 0..seq * h { d_hidden_mid[i] += d_hidden[i]; }

        // ==================== Attention backward ====================
        // O projection backward: d_attn_out = d_hidden_mid @ o_proj: [seq, h] @ [h, q_dim] → [seq, q_dim]
        // Wait: o_proj is [h, q_dim]. Forward was: o_buf = attn_out @ o_proj^T.
        // Backward: d_attn_out = d_o_buf @ o_proj (NO transpose)
        // d_o_buf = d_hidden_mid (the gradient flowing into the attention residual path)
        // But d_o_buf is the gradient of the O-projection output, which is d_hidden_mid
        // (since hidden_mid = hidden_in + o_buf, and we split the residual).
        // Actually no: d_hidden_mid already has both the residual AND the FFN path.
        // The d through the attention residual is just d_hidden_mid.
        // So d_o_buf = d_hidden_mid (this is the gradient of o_buf + hidden_in).
        // For the o_proj backward: d_attn_out = d_hidden_mid @ o_proj
        // where o_proj is [h, q_dim], so this is [seq, h] @ [h, q_dim] → [seq, q_dim].

        let mut d_attn_out = vec![0.0f32; seq * q_dim];
        sgemm(seq, q_dim, h, &d_hidden_mid, &layer.o_proj, &mut d_attn_out);

        // LoRA on O: backward adds to d_attn_out
        let o_grads = if let (Some(adapter), Some(lora_h)) = (&lora_layer.o, &la.lora_h_o) {
            Some(lora_bwd(&mut d_attn_out, &d_hidden_mid, &la.attn_out, lora_h, adapter, scale, seq))
        } else {
            None
        };

        // --- SDPA backward ---
        // Reverse the per-head attention computation.
        let attn_scale = 1.0 / (hd as f32).sqrt();
        let mut dq_buf = vec![0.0f32; seq * q_dim];
        let mut dk_buf = vec![0.0f32; seq * kv_dim];
        let mut dv_buf = vec![0.0f32; seq * kv_dim];

        for kv_h in 0..n_kv {
            // Extract K, V for this KV head
            let mut k_head = vec![0.0f32; seq * hd];
            let mut v_head = vec![0.0f32; seq * hd];
            for s in 0..seq {
                let ko = s * kv_dim + kv_h * hd;
                k_head[s * hd..(s + 1) * hd].copy_from_slice(&la.k_final[ko..ko + hd]);
                v_head[s * hd..(s + 1) * hd].copy_from_slice(&la.v[ko..ko + hd]);
            }

            let mut dk_head = vec![0.0f32; seq * hd];
            let mut dv_head = vec![0.0f32; seq * hd];

            for g in 0..gqa_ratio {
                let q_h = kv_h * gqa_ratio + g;

                // Extract d_ctx and Q for this head
                let mut d_ctx = vec![0.0f32; seq * hd];
                let mut q_head = vec![0.0f32; seq * hd];
                for s in 0..seq {
                    let ao = s * q_dim + q_h * hd;
                    d_ctx[s * hd..(s + 1) * hd].copy_from_slice(&d_attn_out[ao..ao + hd]);
                    q_head[s * hd..(s + 1) * hd].copy_from_slice(&la.q_final[ao..ao + hd]);
                }

                let p = &la.attn_probs[q_h]; // [seq*seq]

                // dV += P^T @ d_ctx: [seq, seq]^T @ [seq, hd] → [seq, hd]
                sgemm_at_acc(seq, hd, seq, p, &d_ctx, &mut dv_head);

                // dP = d_ctx @ V^T: [seq, hd] @ [hd, seq] → [seq, seq]
                let mut dp = vec![0.0f32; seq * seq];
                sgemm_nt(seq, seq, hd, &d_ctx, &v_head, &mut dp);

                // Softmax backward: dS = P * (dP - rowsum(dP*P))
                let mut ds = vec![0.0f32; seq * seq];
                for row in 0..seq {
                    softmax_bwd(
                        &mut ds[row * seq..(row + 1) * seq],
                        &dp[row * seq..(row + 1) * seq],
                        &p[row * seq..(row + 1) * seq],
                        seq,
                    );
                }

                // dQ = dS @ K * scale: [seq, seq] @ [seq, hd] → [seq, hd]
                let mut dq_head = vec![0.0f32; seq * hd];
                sgemm(seq, hd, seq, &ds, &k_head, &mut dq_head);
                for v in dq_head.iter_mut() { *v *= attn_scale; }

                // dK += dS^T @ Q * scale
                let mut dk_contrib = vec![0.0f32; seq * hd];
                sgemm_at(seq, hd, seq, &ds, &q_head, &mut dk_contrib);
                for (a, b) in dk_head.iter_mut().zip(dk_contrib.iter()) { *a += b * attn_scale; }

                // Scatter dQ back into dq_buf
                for s in 0..seq {
                    let ao = s * q_dim + q_h * hd;
                    for d in 0..hd { dq_buf[ao + d] += dq_head[s * hd + d]; }
                }
            }

            // Scatter dK, dV back into dk_buf, dv_buf
            for s in 0..seq {
                let ko = s * kv_dim + kv_h * hd;
                for d in 0..hd {
                    dk_buf[ko + d] += dk_head[s * hd + d];
                    dv_buf[ko + d] += dv_head[s * hd + d];
                }
            }
        }

        // --- RoPE backward ---
        let mut dq_pre_rope = vec![0.0f32; seq * q_dim];
        let mut dk_pre_rope = vec![0.0f32; seq * kv_dim];
        for s in 0..seq {
            for head in 0..n_heads {
                let off = s * q_dim + head * hd;
                rope_bwd(
                    &mut dq_pre_rope[off..off + hd],
                    &dq_buf[off..off + hd],
                    s, half_hd, &engine.rope_cos, &engine.rope_sin,
                );
            }
            for head in 0..n_kv {
                let off = s * kv_dim + head * hd;
                rope_bwd(
                    &mut dk_pre_rope[off..off + hd],
                    &dk_buf[off..off + hd],
                    s, half_hd, &engine.rope_cos, &engine.rope_sin,
                );
            }
        }

        // --- QK-norm backward ---
        let mut dq_proj = vec![0.0f32; seq * q_dim];
        let mut dk_proj = vec![0.0f32; seq * kv_dim];
        for s in 0..seq {
            for head in 0..n_heads {
                let off = s * q_dim + head * hd;
                qk_norm_bwd_head(
                    &mut dq_proj[off..off + hd],
                    &dq_pre_rope[off..off + hd],
                    &la.q_pre_norm[off..off + hd],
                    &layer.q_norm,
                    hd,
                );
            }
            for head in 0..n_kv {
                let off = s * kv_dim + head * hd;
                qk_norm_bwd_head(
                    &mut dk_proj[off..off + hd],
                    &dk_pre_rope[off..off + hd],
                    &la.k_pre_norm[off..off + hd],
                    &layer.k_norm,
                    hd,
                );
            }
        }

        // --- QKV projection backward ---
        // dQ_proj[seq, q_dim] → d_normed from Q path: [seq, q_dim] @ [q_dim, h] → [seq, h]
        // q_proj is [q_dim, h], forward was: q = normed @ q_proj^T
        // backward: d_normed += dq_proj @ q_proj
        let mut d_normed_attn = vec![0.0f32; seq * h];
        sgemm(seq, h, q_dim, &dq_proj, &layer.q_proj, &mut d_normed_attn);
        // K path
        sgemm_acc(seq, h, kv_dim, &dk_proj, &layer.k_proj, &mut d_normed_attn);
        // V path
        sgemm_acc(seq, h, kv_dim, &dv_buf, &layer.v_proj, &mut d_normed_attn);

        // LoRA on Q
        let q_grads = if let (Some(adapter), Some(lora_h)) = (&lora_layer.q, &la.lora_h_q) {
            Some(lora_bwd(&mut d_normed_attn, &dq_proj, &la.normed_attn, lora_h, adapter, scale, seq))
        } else {
            None
        };

        // LoRA on V
        let v_grads = if let (Some(adapter), Some(lora_h)) = (&lora_layer.v, &la.lora_h_v) {
            Some(lora_bwd(&mut d_normed_attn, &dv_buf, &la.normed_attn, lora_h, adapter, scale, seq))
        } else {
            None
        };

        // --- Input norm backward ---
        let mut d_hidden_in = vec![0.0f32; seq * h];
        let mut _d_input_norm_w = vec![0.0f32; h];
        rmsnorm_bwd(
            &mut d_hidden_in, &mut _d_input_norm_w, &d_normed_attn,
            &la.hidden_in, &layer.input_norm, seq, h,
        );

        // Add residual: d_hidden_in += d_hidden_mid
        for i in 0..seq * h { d_hidden_in[i] += d_hidden_mid[i]; }

        // This becomes d_hidden for the previous layer
        d_hidden = d_hidden_in;

        layer_grads.push(LoraLayerGrads {
            q: q_grads,
            v: v_grads,
            o: o_grads,
            down: down_grads,
        });
    }

    // Reverse the grads to match layer order (we built them in reverse)
    layer_grads.reverse();

    DiffusionLoraGrads { layers: layer_grads }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn small_config() -> DiffusionConfig {
        DiffusionConfig {
            hidden: 64,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            head_dim: 32,
            inter: 128,
            vocab: 256,
            mask_token_id: 255,
            rope_theta: 10000.0,
        }
    }

    #[test]
    fn test_lora_forward_matches_base_when_b_zero() {
        let cfg = small_config();
        let engine = DiffusionEngine::from_random(cfg.clone());
        let lora_cfg = LoraConfig::default();
        let lora = DiffusionLoraModel::new(lora_cfg, &cfg);

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 255, 255, 255];

        let base_logits = engine.forward(&tokens);
        let (lora_logits, _acts) = forward_train(&engine, &lora, &tokens);

        let max_err = base_logits.iter().zip(lora_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!("Base vs LoRA (B=0) max error: {max_err:.6e}");
        assert!(max_err < 1e-4, "LoRA with B=0 should match base. Error: {max_err}");
    }

    #[test]
    fn test_lora_params_count() {
        let cfg = small_config();
        let lora_cfg = LoraConfig { rank: 8, ..LoraConfig::default() };
        let lora = DiffusionLoraModel::new(lora_cfg, &cfg);

        let n_params = lora.num_params();
        // Per layer with rank=8:
        //   q: 8*64 + 128*8 = 512 + 1024 = 1536  (d_in=64=h, d_out=128=4*32=q_dim)
        //   v: 8*64 + 64*8 = 512 + 512 = 1024   (d_out=64=2*32=kv_dim)
        //   o: 8*128 + 64*8 = 1024 + 512 = 1536  (d_in=128=q_dim, d_out=64=h)
        //   down: 8*128 + 64*8 = 1024 + 512 = 1536  (d_in=128=inter, d_out=64=h)
        // Per layer: 5632, 2 layers: 11264
        eprintln!("LoRA params: {n_params}");
        assert_eq!(n_params, 2 * (1536 + 1024 + 1536 + 1536));
    }

    #[test]
    fn test_finite_difference_gradient_check() {
        let cfg = small_config();
        let engine = DiffusionEngine::from_random(cfg.clone());
        let lora_cfg = LoraConfig { rank: 4, ..LoraConfig::default() };
        let mut lora = DiffusionLoraModel::new(lora_cfg, &cfg);

        // Give B matrices larger non-zero values so gradients are well above noise floor
        for layer in &mut lora.layers {
            for adapter in [&mut layer.q, &mut layer.v, &mut layer.o, &mut layer.down] {
                if let Some(a) = adapter {
                    for (i, v) in a.b.iter_mut().enumerate() {
                        *v = ((i as f32 * 0.618 + 0.1).fract() * 2.0 - 1.0) * 0.05;
                    }
                }
            }
        }

        let tokens: Vec<u32> = vec![1, 2, 3, 255, 255, 255, 255, 255];
        let mask_positions: Vec<usize> = (3..8).collect();

        // Use f64 for loss to avoid catastrophic cancellation in FD
        let loss_fn = |lora: &DiffusionLoraModel| -> f64 {
            let (logits, _) = forward_train(&engine, lora, &tokens);
            let vocab = cfg.vocab;
            let mut total = 0.0f64;
            for &pos in &mask_positions {
                let row = &logits[pos * vocab..(pos + 1) * vocab];
                let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum_exp: f64 = row.iter().map(|v| (*v as f64 - max_l).exp()).sum();
                let log_sum = max_l + sum_exp.ln();
                total -= logits[pos * vocab + 1] as f64 - log_sum;
            }
            total / mask_positions.len() as f64
        };

        let (logits, acts) = forward_train(&engine, &lora, &tokens);
        let vocab = cfg.vocab;

        let mut d_logits = vec![0.0f32; tokens.len() * vocab];
        for &pos in &mask_positions {
            let row = &logits[pos * vocab..(pos + 1) * vocab];
            let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|v| (v - max_l).exp()).sum();
            for v_idx in 0..vocab {
                let softmax_v = (row[v_idx] - max_l).exp() / sum_exp;
                let target = if v_idx == 1 { 1.0 } else { 0.0 };
                d_logits[pos * vocab + v_idx] = (softmax_v - target) / mask_positions.len() as f32;
            }
        }

        let grads = backward(&engine, &lora, &acts, &d_logits);

        let eps = 5e-4f64;
        let abs_tol = 1e-5f64;
        let mut n_pass = 0u32;
        let mut n_fail = 0u32;
        let mut n_skip = 0u32;

        // Collect (name, layer, target, matrix, index, analytical_grad) tuples
        // target: 0=q, 1=v, 2=o, 3=down; matrix: 0=a, 1=b
        let n = 8usize;
        let mut checks: Vec<(String, usize, usize, usize, usize, f32)> = Vec::new();

        for li in 0..cfg.layers {
            for (ti, name) in [(0, "q"), (1, "v"), (2, "o"), (3, "down")] {
                let grad_opt = match ti {
                    0 => grads.layers[li].q.as_ref(),
                    1 => grads.layers[li].v.as_ref(),
                    2 => grads.layers[li].o.as_ref(),
                    _ => grads.layers[li].down.as_ref(),
                };
                if let Some(grad) = grad_opt {
                    for i in 0..n.min(grad.db.len()) {
                        checks.push((format!("L{li}.{name}.B[{i}]"), li, ti, 1, i, grad.db[i]));
                    }
                    for i in 0..n.min(grad.da.len()) {
                        checks.push((format!("L{li}.{name}.A[{i}]"), li, ti, 0, i, grad.da[i]));
                    }
                }
            }
        }

        for (name, li, ti, mat, idx, analytical) in &checks {
            // Get/set the parameter
            let get_param = |l: &DiffusionLoraModel| -> f32 {
                let adapter = match ti { 0 => &l.layers[*li].q, 1 => &l.layers[*li].v, 2 => &l.layers[*li].o, _ => &l.layers[*li].down };
                let a = adapter.as_ref().unwrap();
                if *mat == 0 { a.a[*idx] } else { a.b[*idx] }
            };
            let set_param = |l: &mut DiffusionLoraModel, val: f32| {
                let adapter = match ti { 0 => &mut l.layers[*li].q, 1 => &mut l.layers[*li].v, 2 => &mut l.layers[*li].o, _ => &mut l.layers[*li].down };
                let a = adapter.as_mut().unwrap();
                if *mat == 0 { a.a[*idx] = val; } else { a.b[*idx] = val; }
            };

            let original = get_param(&lora);
            set_param(&mut lora, original + eps as f32);
            let loss_plus = loss_fn(&lora);
            set_param(&mut lora, original - eps as f32);
            let loss_minus = loss_fn(&lora);
            set_param(&mut lora, original);

            let fd = (loss_plus - loss_minus) / (2.0 * eps);
            let ag = *analytical as f64;

            if fd.abs() < abs_tol && ag.abs() < abs_tol {
                n_skip += 1;
                continue;
            }

            let rel_err = (fd - ag).abs() / fd.abs().max(ag.abs());
            if rel_err < 0.1 {
                n_pass += 1;
            } else {
                n_fail += 1;
                eprintln!("  FAIL {name}: fd={fd:.6e}, analytical={ag:.6e}, rel_err={rel_err:.3e}");
            }
        }

        let total = n_pass + n_fail + n_skip;
        eprintln!("Gradient check: {n_pass} pass, {n_fail} fail, {n_skip} skip (tiny), {total} total");
        let fail_rate = n_fail as f32 / (n_pass + n_fail).max(1) as f32;
        assert!(
            fail_rate < 0.15,
            "Too many gradient check failures: {n_fail}/{} ({:.0}%)",
            n_pass + n_fail, fail_rate * 100.0
        );
    }
}
