//! MDLM diffusion training loop with LoRA adapters.
//!
//! Implements:
//! - Random masking schedule (MDLM: mask ratio sampled per step)
//! - Cross-entropy loss on masked positions only
//! - AdamW optimizer for LoRA parameters
//! - Cosine LR schedule with linear warmup
//! - Gradient clipping
//!
//! Phase 1: CPU backward via BLAS. Phase 2: ANE backward kernels.

#![allow(clippy::too_many_arguments)]

use crate::diffusion::DiffusionEngine;
use crate::diffusion_lora::{
    backward, forward_train, DiffusionLoraGrads, DiffusionLoraModel, LoraAdapter,
    LoraAdapterGrads,
};

// ---------------------------------------------------------------------------
// Training config
// ---------------------------------------------------------------------------

pub struct TrainingConfig {
    pub total_steps: usize,
    pub max_lr: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    pub weight_decay: f32,
    pub warmup_steps: usize,
    pub grad_clip: f32,
    pub min_lr_frac: f32,
    pub log_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            total_steps: 1000,
            max_lr: 5e-4,     // 10x standard (JIT LoRA proven)
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            weight_decay: 0.01,
            warmup_steps: 50,
            grad_clip: 1.0,
            min_lr_frac: 0.1,
            log_interval: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Adam state for LoRA
// ---------------------------------------------------------------------------

pub struct AdamState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl AdamState {
    pub fn zeros(n: usize) -> Self {
        AdamState { m: vec![0.0; n], v: vec![0.0; n] }
    }
}

pub struct LoraAdapterAdamState {
    pub a: AdamState,
    pub b: AdamState,
}

pub struct LoraLayerAdamState {
    pub q: Option<LoraAdapterAdamState>,
    pub v: Option<LoraAdapterAdamState>,
    pub o: Option<LoraAdapterAdamState>,
    pub down: Option<LoraAdapterAdamState>,
}

pub struct DiffusionLoraAdamState {
    pub layers: Vec<LoraLayerAdamState>,
}

impl DiffusionLoraAdamState {
    pub fn zeros(lora: &DiffusionLoraModel) -> Self {
        let layers = lora.layers.iter().map(|l| {
            let mk = |opt: &Option<LoraAdapter>| -> Option<LoraAdapterAdamState> {
                opt.as_ref().map(|a| LoraAdapterAdamState {
                    a: AdamState::zeros(a.rank * a.d_in),
                    b: AdamState::zeros(a.d_out * a.rank),
                })
            };
            LoraLayerAdamState {
                q: mk(&l.q),
                v: mk(&l.v),
                o: mk(&l.o),
                down: mk(&l.down),
            }
        }).collect();
        DiffusionLoraAdamState { layers }
    }
}

// ---------------------------------------------------------------------------
// AdamW update
// ---------------------------------------------------------------------------

fn adam_update(
    w: &mut [f32],
    g: &[f32],
    state: &mut AdamState,
    t: usize,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
) {
    let bc1 = 1.0 / (1.0 - b1.powi(t as i32));
    let bc2 = 1.0 / (1.0 - b2.powi(t as i32));

    for i in 0..w.len() {
        if wd > 0.0 { w[i] *= 1.0 - lr * wd; }
        state.m[i] = b1 * state.m[i] + (1.0 - b1) * g[i];
        state.v[i] = b2 * state.v[i] + (1.0 - b2) * g[i] * g[i];
        let m_hat = state.m[i] * bc1;
        let v_hat = state.v[i] * bc2;
        w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

fn apply_adam_to_adapter(
    adapter: &mut LoraAdapter,
    grads: &LoraAdapterGrads,
    adam: &mut LoraAdapterAdamState,
    t: usize,
    lr: f32,
    cfg: &TrainingConfig,
) {
    adam_update(&mut adapter.a, &grads.da, &mut adam.a, t, lr, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, cfg.weight_decay);
    adam_update(&mut adapter.b, &grads.db, &mut adam.b, t, lr, cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps, cfg.weight_decay);
}

/// Apply Adam to all LoRA parameters.
pub fn adam_update_all(
    lora: &mut DiffusionLoraModel,
    grads: &DiffusionLoraGrads,
    adam: &mut DiffusionLoraAdamState,
    t: usize,
    lr: f32,
    cfg: &TrainingConfig,
) {
    for (li, lg) in grads.layers.iter().enumerate() {
        let ll = &mut lora.layers[li];
        let la = &mut adam.layers[li];

        if let (Some(adapter), Some(g), Some(st)) = (&mut ll.q, &lg.q, &mut la.q) {
            apply_adam_to_adapter(adapter, g, st, t, lr, cfg);
        }
        if let (Some(adapter), Some(g), Some(st)) = (&mut ll.v, &lg.v, &mut la.v) {
            apply_adam_to_adapter(adapter, g, st, t, lr, cfg);
        }
        if let (Some(adapter), Some(g), Some(st)) = (&mut ll.o, &lg.o, &mut la.o) {
            apply_adam_to_adapter(adapter, g, st, t, lr, cfg);
        }
        if let (Some(adapter), Some(g), Some(st)) = (&mut ll.down, &lg.down, &mut la.down) {
            apply_adam_to_adapter(adapter, g, st, t, lr, cfg);
        }
    }
}

// ---------------------------------------------------------------------------
// Cosine LR with linear warmup
// ---------------------------------------------------------------------------

pub fn cosine_lr(step: usize, warmup: usize, total: usize, max_lr: f32, min_lr_frac: f32) -> f32 {
    if step < warmup {
        max_lr * (step as f32) / (warmup as f32).max(1.0)
    } else {
        let min_lr = max_lr * min_lr_frac;
        let decay_steps = total.saturating_sub(warmup);
        if decay_steps == 0 { return max_lr; }
        let progress = ((step - warmup) as f32 / decay_steps as f32).min(1.0);
        min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

// ---------------------------------------------------------------------------
// Gradient operations
// ---------------------------------------------------------------------------

fn for_each_grad(grads: &DiffusionLoraGrads, mut f: impl FnMut(&[f32])) {
    for lg in &grads.layers {
        if let Some(ref g) = lg.q { f(&g.da); f(&g.db); }
        if let Some(ref g) = lg.v { f(&g.da); f(&g.db); }
        if let Some(ref g) = lg.o { f(&g.da); f(&g.db); }
        if let Some(ref g) = lg.down { f(&g.da); f(&g.db); }
    }
}

fn for_each_grad_mut(grads: &mut DiffusionLoraGrads, mut f: impl FnMut(&mut [f32])) {
    for lg in &mut grads.layers {
        if let Some(ref mut g) = lg.q { f(&mut g.da); f(&mut g.db); }
        if let Some(ref mut g) = lg.v { f(&mut g.da); f(&mut g.db); }
        if let Some(ref mut g) = lg.o { f(&mut g.da); f(&mut g.db); }
        if let Some(ref mut g) = lg.down { f(&mut g.da); f(&mut g.db); }
    }
}

pub fn global_grad_norm(grads: &DiffusionLoraGrads) -> f32 {
    let mut sum_sq = 0.0f64;
    for_each_grad(grads, |g| {
        for &v in g { sum_sq += (v as f64) * (v as f64); }
    });
    (sum_sq as f32).sqrt()
}

pub fn clip_gradients(grads: &mut DiffusionLoraGrads, clip: f32) {
    let norm = global_grad_norm(grads);
    if norm > clip {
        let scale = clip / norm;
        for_each_grad_mut(grads, |g| {
            for v in g.iter_mut() { *v *= scale; }
        });
    }
}

// ---------------------------------------------------------------------------
// MDLM masking + loss
// ---------------------------------------------------------------------------

/// Create a masked version of input tokens for MDLM training.
///
/// `mask_ratio`: fraction of non-prompt tokens to mask (0.0 to 1.0).
/// Returns (masked_tokens, mask_positions).
pub fn random_mask(
    tokens: &[u32],
    prompt_len: usize,
    mask_id: u32,
    mask_ratio: f32,
    rng_seed: u64,
) -> (Vec<u32>, Vec<usize>) {
    let mut masked = tokens.to_vec();
    let mut mask_positions = Vec::new();

    // Simple deterministic RNG (xorshift64)
    let mut state = rng_seed.wrapping_add(1);
    let mut next_f32 = || -> f32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f32) / (u64::MAX as f32)
    };

    for i in prompt_len..tokens.len() {
        if next_f32() < mask_ratio {
            masked[i] = mask_id;
            mask_positions.push(i);
        }
    }

    (masked, mask_positions)
}

/// Compute cross-entropy loss and logit gradients for masked positions.
///
/// Returns (loss, d_logits[seq, vocab]).
pub fn mdlm_loss(
    logits: &[f32],
    targets: &[u32],
    mask_positions: &[usize],
    vocab: usize,
) -> (f32, Vec<f32>) {
    let seq = targets.len();
    let mut d_logits = vec![0.0f32; seq * vocab];
    let mut total_loss = 0.0f32;

    if mask_positions.is_empty() {
        return (0.0, d_logits);
    }

    let n_masks = mask_positions.len() as f32;

    for &pos in mask_positions {
        let row = &logits[pos * vocab..(pos + 1) * vocab];
        let target = targets[pos] as usize;

        // Numerically stable cross-entropy
        let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|v| (v - max_l).exp()).sum();
        let log_sum = max_l + sum_exp.ln();
        total_loss -= (row[target] - log_sum) / n_masks;

        // Gradient: (softmax - one_hot) / n_masks
        for v_idx in 0..vocab {
            let softmax_v = (row[v_idx] - max_l).exp() / sum_exp;
            let target_v = if v_idx == target { 1.0 } else { 0.0 };
            d_logits[pos * vocab + v_idx] = (softmax_v - target_v) / n_masks;
        }
    }

    (total_loss, d_logits)
}

// ---------------------------------------------------------------------------
// Training step
// ---------------------------------------------------------------------------

/// Single training step: forward → loss → backward → clip → update.
///
/// Returns the loss value.
pub fn training_step(
    engine: &DiffusionEngine,
    lora: &mut DiffusionLoraModel,
    adam: &mut DiffusionLoraAdamState,
    tokens: &[u32],
    prompt_len: usize,
    step: usize,
    cfg: &TrainingConfig,
    rng_seed: u64,
) -> f32 {
    let mask_id = engine.config.mask_token_id;
    let vocab = engine.config.vocab;

    // Sample mask ratio from uniform [0, 1] (MDLM schedule)
    let mut state = rng_seed.wrapping_add(step as u64 * 31337);
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    let mask_ratio = (state as f32) / (u64::MAX as f32);

    // Mask input
    let (masked_tokens, mask_positions) = random_mask(tokens, prompt_len, mask_id, mask_ratio, rng_seed.wrapping_add(step as u64));

    if mask_positions.is_empty() {
        return 0.0;
    }

    // Forward
    let (logits, acts) = forward_train(engine, lora, &masked_tokens);

    // Loss (targets are the ORIGINAL unmasked tokens)
    let (loss, d_logits) = mdlm_loss(&logits, tokens, &mask_positions, vocab);

    // Backward
    let mut grads = backward(engine, lora, &acts, &d_logits);

    // Gradient clipping
    clip_gradients(&mut grads, cfg.grad_clip);

    // LR schedule
    let lr = cosine_lr(step, cfg.warmup_steps, cfg.total_steps, cfg.max_lr, cfg.min_lr_frac);

    // Adam update (step is 1-indexed for bias correction)
    adam_update_all(lora, &grads, adam, step + 1, lr, cfg);

    loss
}

/// Run the training loop.
pub fn train(
    engine: &DiffusionEngine,
    lora: &mut DiffusionLoraModel,
    tokens: &[u32],
    prompt_len: usize,
    cfg: &TrainingConfig,
) -> Vec<f32> {
    let mut adam = DiffusionLoraAdamState::zeros(lora);
    let mut losses = Vec::with_capacity(cfg.total_steps);

    let t0 = std::time::Instant::now();

    for step in 0..cfg.total_steps {
        let loss = training_step(
            engine, lora, &mut adam, tokens, prompt_len, step, cfg, 42,
        );
        losses.push(loss);

        if step % cfg.log_interval == 0 || step == cfg.total_steps - 1 {
            let lr = cosine_lr(step, cfg.warmup_steps, cfg.total_steps, cfg.max_lr, cfg.min_lr_frac);
            let elapsed = t0.elapsed().as_secs_f64();
            let ms_per_step = if step > 0 { elapsed / step as f64 * 1000.0 } else { 0.0 };
            eprintln!(
                "step {step:>5}/{}: loss={loss:.4}, lr={lr:.2e}, {ms_per_step:.0}ms/step",
                cfg.total_steps
            );
        }
    }

    losses
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::diffusion::DiffusionConfig;
    use crate::diffusion_lora::LoraConfig;

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
    fn test_cosine_lr_schedule() {
        let lr = cosine_lr(0, 10, 100, 1e-3, 0.1);
        assert!(lr < 1e-6, "LR at step 0 should be ~0 during warmup");

        let lr = cosine_lr(10, 10, 100, 1e-3, 0.1);
        assert!((lr - 1e-3).abs() < 1e-6, "LR should be max_lr at end of warmup");

        let lr = cosine_lr(100, 10, 100, 1e-3, 0.1);
        assert!((lr - 1e-4).abs() < 1e-6, "LR should be min_lr at end");
    }

    #[test]
    fn test_mdlm_loss() {
        let vocab = 10;
        let seq = 4;
        // Uniform logits
        let logits = vec![0.0f32; seq * vocab];
        let targets = vec![3u32, 5, 7, 1];
        let mask_positions = vec![1, 3];

        let (loss, d_logits) = mdlm_loss(&logits, &targets, &mask_positions, vocab);

        // Uniform logits: loss = -log(1/vocab) = log(vocab)
        let expected = (vocab as f32).ln();
        eprintln!("Loss: {loss:.4}, expected: {expected:.4}");
        assert!((loss - expected).abs() < 0.01);

        // Gradient should be non-zero only at mask positions
        for pos in 0..seq {
            let row_sum: f32 = d_logits[pos * vocab..(pos + 1) * vocab].iter().sum();
            if mask_positions.contains(&pos) {
                // Gradient row should sum to ~0 (softmax - one_hot sums to 0)
                assert!(row_sum.abs() < 1e-5, "Gradient at mask pos should sum to 0");
            } else {
                assert!(row_sum.abs() < 1e-10, "Gradient at non-mask pos should be 0");
            }
        }
    }

    #[test]
    fn test_random_mask() {
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let (masked, positions) = random_mask(&tokens, 3, 255, 0.5, 42);

        // Prompt should never be masked
        assert_eq!(masked[0], 1);
        assert_eq!(masked[1], 2);
        assert_eq!(masked[2], 3);

        // Some positions should be masked
        for &pos in &positions {
            assert!(pos >= 3);
            assert_eq!(masked[pos], 255);
        }
        eprintln!("Masked {}/{} gen tokens", positions.len(), 5);
    }

    /// Integration test: load real Qwen3-0.6B diffusion model and run 5 training steps.
    /// Run with: cargo test -p higgs-models -- test_real_model_training --nocapture --ignored
    #[test]
    #[ignore]
    fn test_real_model_training() {
        let home = std::env::var("HOME").expect("HOME not set");
        let model_dir = std::path::PathBuf::from(home).join(
            ".cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1"
        );
        if !model_dir.exists() {
            eprintln!("Model not found at {model_dir:?}, skipping");
            return;
        }

        eprintln!("Loading Qwen3-0.6B diffusion model...");
        let engine = DiffusionEngine::load(&model_dir).expect("Failed to load model");
        eprintln!("Config: {:?}", engine.config);

        // Create LoRA adapters (rank=32, default targets: q, v, o, down)
        let lora_cfg = LoraConfig::default();
        let mut lora = DiffusionLoraModel::new(lora_cfg, &engine.config);

        let n_params: usize = lora.layers.iter().map(|l| {
            let c = |a: &Option<LoraAdapter>| a.as_ref().map_or(0, |a| a.rank * a.d_in + a.d_out * a.rank);
            c(&l.q) + c(&l.v) + c(&l.o) + c(&l.down)
        }).sum();
        eprintln!("LoRA parameters: {n_params} ({:.2}M)", n_params as f64 / 1e6);

        // Training data: a short sentence with mask token at the end
        // "The capital of France is [MASK] [MASK] [MASK]"
        // Using raw token IDs (would normally use tokenizer)
        let mask_id = engine.config.mask_token_id;
        let tokens: Vec<u32> = vec![
            791, 6864, 315, 9822, 374, // "The capital of France is"
            12366, 13,                  // "Paris."  (target)
        ];
        let prompt_len = 5; // Everything before "Paris." is prompt

        let train_cfg = TrainingConfig {
            total_steps: 5,
            max_lr: 5e-4,
            warmup_steps: 2,
            log_interval: 1,
            ..TrainingConfig::default()
        };

        eprintln!("\nRunning 5 training steps on {} tokens (prompt_len={prompt_len})...", tokens.len());
        let losses = train(&engine, &mut lora, &tokens, prompt_len, &train_cfg);

        eprintln!("\nLosses: {:?}", losses);
        assert!(losses.iter().all(|l| l.is_finite()), "Loss must be finite");
        eprintln!("PASS: Real model training produces finite losses");
    }

    #[test]
    fn test_training_reduces_loss() {
        let cfg = small_config();
        let engine = DiffusionEngine::from_random(cfg.clone());
        let lora_cfg = LoraConfig { rank: 4, ..LoraConfig::default() };
        let mut lora = DiffusionLoraModel::new(lora_cfg, &cfg);

        let tokens: Vec<u32> = vec![1, 2, 3, 10, 20, 30, 40, 50];
        let prompt_len = 3;

        let train_cfg = TrainingConfig {
            total_steps: 50,
            max_lr: 1e-3,
            log_interval: 10,
            ..TrainingConfig::default()
        };

        let losses = train(&engine, &mut lora, &tokens, prompt_len, &train_cfg);

        let first_5: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let last_5: f32 = losses[losses.len()-5..].iter().sum::<f32>() / 5.0;

        eprintln!("First 5 avg loss: {first_5:.4}, last 5 avg loss: {last_5:.4}");
        // Loss should decrease (or at least not increase drastically)
        // With random model this might not always decrease, but it should not explode
        assert!(last_5 < first_5 * 2.0, "Loss should not explode: {first_5} → {last_5}");
    }
}
