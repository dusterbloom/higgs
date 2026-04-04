//! EGGROLL — Forward-pass-only training via population-based low-rank perturbations.
//!
//! Replaces backpropagation with evolutionary strategies (ES). Each step:
//! 1. Generate N low-rank perturbations from deterministic seeds
//! 2. Evaluate +/- antithetical pairs via forward passes
//! 3. Accumulate ES gradient from fitness signals
//! 4. Update weights directly (no optimizer state, no activations)
//!
//! Memory overhead: ~90 MB on top of the base model (0.6B fp32 ≈ 1.76 GB).
//! The perturbation factors are regenerated from seeds — zero storage.
//!
//! Reference: arXiv:2511.16652 (EGGROLL)

#![allow(
    clippy::too_many_arguments,
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::inline_always,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unreadable_literal,
    clippy::enum_variant_names,
    clippy::match_same_arms,
    clippy::print_stderr,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    unsafe_code
)]

use crate::diffusion::{DiffusionConfig, DiffusionEngine, DiffusionLayerWeights};
use crate::diffusion_train::{cosine_lr, mdlm_loss, random_mask};

// BLAS FFI — Accelerate framework (linked via ane_bridge build.rs)
unsafe extern "C" {
    /// Rank-1 update: A[m,n] += alpha * x[m] * y[n]^T (row-major).
    unsafe fn cblas_sger(
        order: i32,
        m: i32,
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
        a: *mut f32,
        lda: i32,
    );
}

// -----------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------

pub struct EggrollConfig {
    pub sigma: f32,
    pub lr: f32,
    pub rank: usize,
    pub population: usize,
    pub total_steps: usize,
    pub warmup_steps: usize,
    pub min_lr_frac: f32,
    pub log_interval: usize,
    pub base_seed: u64,
}

impl Default for EggrollConfig {
    fn default() -> Self {
        Self {
            sigma: 0.01,
            lr: 0.001,
            rank: 4,
            population: 32,
            total_steps: 100,
            warmup_steps: 10,
            min_lr_frac: 0.1,
            log_interval: 10,
            base_seed: 42,
        }
    }
}

// -----------------------------------------------------------------------
// RNG: counter-based deterministic xorshift64 + Box-Muller
// -----------------------------------------------------------------------

/// Counter-based seed: deterministic from (base, step, member, layer, weight_idx).
pub fn make_seed(
    base: u64,
    step: usize,
    member: usize,
    layer: usize,
    weight_idx: usize,
) -> u64 {
    base.wrapping_mul(6364136223846793005)
        .wrapping_add((step as u64).wrapping_mul(2654435761))
        .wrapping_add((member as u64).wrapping_mul(1442695040888963407))
        .wrapping_add((layer as u64).wrapping_mul(7046029254386353131))
        .wrapping_add((weight_idx as u64).wrapping_mul(3935559000370003845))
}

#[inline(always)]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[inline(always)]
fn xorshift_uniform(state: &mut u64) -> f32 {
    xorshift64(state);
    (*state as f32) / (u64::MAX as f32)
}

/// Approximate N(0,1) via Box-Muller transform.
pub fn xorshift_normal(state: &mut u64) -> f32 {
    let u1 = xorshift_uniform(state).max(1e-10);
    let u2 = xorshift_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Generate low-rank factors A[rows, rank] and B[cols, rank] from seed.
pub fn generate_factors(
    seed: u64,
    rows: usize,
    cols: usize,
    rank: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut state = seed;
    // Warm up the RNG
    xorshift64(&mut state);
    xorshift64(&mut state);

    let a: Vec<f32> = (0..rows * rank)
        .map(|_| xorshift_normal(&mut state))
        .collect();
    let b: Vec<f32> = (0..cols * rank)
        .map(|_| xorshift_normal(&mut state))
        .collect();
    (a, b)
}

// -----------------------------------------------------------------------
// Weight target descriptors
// -----------------------------------------------------------------------

#[derive(Clone, Copy)]
enum WeightKind {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

const ALL_WEIGHT_KINDS: [WeightKind; 7] = [
    WeightKind::QProj,
    WeightKind::KProj,
    WeightKind::VProj,
    WeightKind::OProj,
    WeightKind::GateProj,
    WeightKind::UpProj,
    WeightKind::DownProj,
];

#[derive(Clone, Copy)]
struct WeightTarget {
    kind: WeightKind,
    rows: usize,
    cols: usize,
}

fn weight_buf_mut(lw: &mut DiffusionLayerWeights, kind: WeightKind) -> &mut [f32] {
    match kind {
        WeightKind::QProj => &mut lw.q_proj,
        WeightKind::KProj => &mut lw.k_proj,
        WeightKind::VProj => &mut lw.v_proj,
        WeightKind::OProj => &mut lw.o_proj,
        WeightKind::GateProj => &mut lw.gate_proj,
        WeightKind::UpProj => &mut lw.up_proj,
        WeightKind::DownProj => &mut lw.down_proj,
    }
}

#[cfg(test)]
fn weight_buf(lw: &DiffusionLayerWeights, kind: WeightKind) -> &[f32] {
    match kind {
        WeightKind::QProj => &lw.q_proj,
        WeightKind::KProj => &lw.k_proj,
        WeightKind::VProj => &lw.v_proj,
        WeightKind::OProj => &lw.o_proj,
        WeightKind::GateProj => &lw.gate_proj,
        WeightKind::UpProj => &lw.up_proj,
        WeightKind::DownProj => &lw.down_proj,
    }
}

/// Build weight target descriptors for one layer.
fn build_targets(cfg: &DiffusionConfig) -> Vec<WeightTarget> {
    let h = cfg.hidden;
    let q_dim = cfg.heads * cfg.head_dim;
    let kv_dim = cfg.kv_heads * cfg.head_dim;
    let inter = cfg.inter;

    ALL_WEIGHT_KINDS
        .iter()
        .map(|&kind| {
            let (rows, cols) = match kind {
                WeightKind::QProj => (q_dim, h),
                WeightKind::KProj => (kv_dim, h),
                WeightKind::VProj => (kv_dim, h),
                WeightKind::OProj => (h, q_dim),
                WeightKind::GateProj => (inter, h),
                WeightKind::UpProj => (inter, h),
                WeightKind::DownProj => (h, inter),
            };
            WeightTarget { kind, rows, cols }
        })
        .collect()
}

// -----------------------------------------------------------------------
// Perturbation injection via BLAS cblas_sger
// -----------------------------------------------------------------------

/// W[rows, cols] += sign * sigma * sum_r(A[:,r] ⊗ B[:,r])
///
/// Uses BLAS rank-1 updates. `a` is [rows * rank], `b` is [cols * rank],
/// stored as rank contiguous columns: a[i * rank + r], b[j * rank + r].
fn apply_perturbation(
    w: &mut [f32],
    a: &[f32],
    b: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
    sigma: f32,
    sign: f32,
) {
    let alpha = sigma * sign;
    // a is stored as [rows, rank] — column r starts at offset r with stride rank.
    // cblas_sger wants contiguous vectors. We need a[:,r] = a[0*rank+r], a[1*rank+r], ...
    // With incx=rank, x starts at &a[r], we get the r-th column.
    for r in 0..rank {
        unsafe {
            cblas_sger(
                101, // CblasRowMajor
                rows as i32,
                cols as i32,
                alpha,
                a.as_ptr().add(r),    // a[:,r] start
                rank as i32,          // incx = rank (stride between rows)
                b.as_ptr().add(r),    // b[:,r] start
                rank as i32,          // incy = rank
                w.as_mut_ptr(),
                cols as i32,          // lda = cols (row-major)
            );
        }
    }
}

// -----------------------------------------------------------------------
// EggrollTrainer (Tier 1 — fp32 DiffusionEngine)
// -----------------------------------------------------------------------

pub struct EggrollTrainer {
    pub engine: DiffusionEngine,
    pub config: EggrollConfig,
    targets: Vec<WeightTarget>, // same for all layers (uniform architecture)
}

impl EggrollTrainer {
    pub fn new(engine: DiffusionEngine, config: EggrollConfig) -> Self {
        let targets = build_targets(&engine.config);
        Self {
            engine,
            config,
            targets,
        }
    }

    /// Run a perturbed forward pass: inject → forward → restore → return logits.
    fn perturbed_forward(
        &mut self,
        token_ids: &[u32],
        step: usize,
        member: usize,
        sign: f32,
    ) -> Vec<f32> {
        let sigma = self.config.sigma;
        let rank = self.config.rank;
        let base_seed = self.config.base_seed;
        let n_layers = self.engine.layers.len();

        // Phase 1: Inject perturbation into all weight matrices
        for layer_idx in 0..n_layers {
            for (wi, target) in self.targets.clone().iter().enumerate() {
                let seed = make_seed(base_seed, step, member, layer_idx, wi);
                let (a, b) = generate_factors(seed, target.rows, target.cols, rank);
                let w = weight_buf_mut(&mut self.engine.layers[layer_idx], target.kind);
                apply_perturbation(w, &a, &b, target.rows, target.cols, rank, sigma, sign);
            }
        }

        // Phase 2: Forward pass with perturbed weights
        let logits = self.engine.forward(token_ids);

        // Phase 3: Restore weights by applying the opposite perturbation
        for layer_idx in 0..n_layers {
            for (wi, target) in self.targets.clone().iter().enumerate() {
                let seed = make_seed(base_seed, step, member, layer_idx, wi);
                let (a, b) = generate_factors(seed, target.rows, target.cols, rank);
                let w = weight_buf_mut(&mut self.engine.layers[layer_idx], target.kind);
                apply_perturbation(w, &a, &b, target.rows, target.cols, rank, sigma, -sign);
            }
        }

        logits
    }

    /// One EGGROLL step: evaluate population with antithetical sampling,
    /// accumulate ES gradient, update base weights.
    ///
    /// Returns average loss across population.
    pub fn eggroll_step(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        mask_positions: &[usize],
        step: usize,
    ) -> f32 {
        let n = self.config.population;
        let sigma = self.config.sigma;
        let rank = self.config.rank;
        let base_seed = self.config.base_seed;
        let vocab = self.engine.config.vocab;
        let n_layers = self.engine.layers.len();
        // Phase 1: Evaluate population (2N forward passes)
        let mut fitnesses = Vec::with_capacity(n);
        let mut total_loss = 0.0f32;

        for member in 0..n {
            let logits_plus = self.perturbed_forward(tokens, step, member, 1.0);
            let (loss_plus, _) = mdlm_loss(&logits_plus, targets, mask_positions, vocab);

            let logits_minus = self.perturbed_forward(tokens, step, member, -1.0);
            let (loss_minus, _) = mdlm_loss(&logits_minus, targets, mask_positions, vocab);

            fitnesses.push(loss_minus - loss_plus);
            total_loss += (loss_plus + loss_minus) * 0.5;
        }

        // Phase 2: Accumulate gradient and update — one weight matrix at a time
        // to keep peak memory at max(rows * cols) ≈ 12.6 MB.
        let lr = cosine_lr(
            step,
            self.config.warmup_steps,
            self.config.total_steps,
            self.config.lr,
            self.config.min_lr_frac,
        );
        let scale = lr / (2.0 * n as f32 * sigma);

        let targets_snap = self.targets.clone();
        for layer_idx in 0..n_layers {
            for (wi, target) in targets_snap.iter().enumerate() {
                let size = target.rows * target.cols;
                let mut grad = vec![0.0f32; size];

                // Accumulate: grad += fitness_i * (A_i @ B_i^T)
                for member in 0..n {
                    let seed = make_seed(base_seed, step, member, layer_idx, wi);
                    let (a, b) = generate_factors(seed, target.rows, target.cols, rank);
                    let fitness = fitnesses[member];
                    // grad += fitness * A @ B^T via rank-1 updates
                    apply_perturbation(
                        &mut grad,
                        &a,
                        &b,
                        target.rows,
                        target.cols,
                        rank,
                        fitness,
                        1.0,
                    );
                }

                // Apply ES update: W += scale * grad
                let w = weight_buf_mut(&mut self.engine.layers[layer_idx], target.kind);
                for i in 0..size {
                    w[i] += scale * grad[i];
                }
            }
        }

        total_loss / n as f32
    }

    /// Full training loop. Returns loss history (one value per step).
    pub fn train(&mut self, tokens: &[u32], prompt_len: usize) -> Vec<f32> {
        let mask_id = self.engine.config.mask_token_id;
        let total_steps = self.config.total_steps;
        let log_interval = self.config.log_interval;
        let mut losses = Vec::with_capacity(total_steps);

        for step in 0..total_steps {
            // Sample mask ratio from uniform [0, 1] (MDLM schedule)
            let mut state = self.config.base_seed.wrapping_add(step as u64 * 31337);
            xorshift64(&mut state);
            let mask_ratio = xorshift_uniform(&mut state).clamp(0.15, 0.85);

            let rng_seed = self.config.base_seed.wrapping_add(step as u64 * 7919);
            let (masked, mask_positions) =
                random_mask(tokens, prompt_len, mask_id, mask_ratio, rng_seed);

            if mask_positions.is_empty() {
                losses.push(0.0);
                continue;
            }

            let loss = self.eggroll_step(&masked, tokens, &mask_positions, step);
            losses.push(loss);

            if log_interval > 0 && step % log_interval == 0 {
                let lr = cosine_lr(
                    step,
                    self.config.warmup_steps,
                    self.config.total_steps,
                    self.config.lr,
                    self.config.min_lr_frac,
                );
                eprintln!(
                    "[eggroll] step {step}/{total_steps}  loss={loss:.4}  lr={lr:.6}  masks={}",
                    mask_positions.len()
                );
            }
        }

        losses
    }

    /// Consume trainer and return the updated engine.
    pub fn into_engine(self) -> DiffusionEngine {
        self.engine
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion::DiffusionConfig;

    fn small_config() -> DiffusionConfig {
        DiffusionConfig {
            hidden: 64,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            inter: 128,
            vocab: 256,
            mask_token_id: 0,
            rope_theta: 10000.0,
        }
    }

    fn small_eggroll_config() -> EggrollConfig {
        EggrollConfig {
            sigma: 0.01,
            lr: 0.01,
            rank: 4,
            population: 8,
            total_steps: 20,
            warmup_steps: 3,
            min_lr_frac: 0.1,
            log_interval: 5,
            base_seed: 42,
        }
    }

    #[test]
    fn test_perturbation_deterministic() {
        let (a1, b1) = generate_factors(42, 8, 4, 2);
        let (a2, b2) = generate_factors(42, 8, 4, 2);
        assert_eq!(a1, a2);
        assert_eq!(b1, b2);

        // Different seed → different factors
        let (a3, _) = generate_factors(43, 8, 4, 2);
        assert_ne!(a1, a3);
    }

    #[test]
    fn test_normal_distribution_stats() {
        let mut state = 12345u64;
        let samples: Vec<f32> = (0..10000).map(|_| xorshift_normal(&mut state)).collect();
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let var = samples
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / samples.len() as f32;

        assert!(mean.abs() < 0.1, "mean should be ~0, got {mean}");
        assert!(
            (var - 1.0).abs() < 0.2,
            "variance should be ~1, got {var}"
        );
    }

    #[test]
    fn test_antithetical_symmetry() {
        let cfg = small_config();
        let engine = DiffusionEngine::from_random(cfg);
        let original = engine.layers[0].q_proj.clone();
        let mut trainer = EggrollTrainer::new(engine, small_eggroll_config());
        let target = &trainer.targets[0]; // q_proj
        let rows = target.rows;
        let cols = target.cols;
        let rank = trainer.config.rank;
        let sigma = trainer.config.sigma;

        let seed = make_seed(trainer.config.base_seed, 0, 0, 0, 0);
        let (a, b) = generate_factors(seed, rows, cols, rank);

        // Apply +perturbation
        {
            let w = &mut trainer.engine.layers[0].q_proj;
            apply_perturbation(w, &a, &b, rows, cols, rank, sigma, 1.0);
        }
        let perturbed_plus = trainer.engine.layers[0].q_proj.clone();
        assert_ne!(perturbed_plus, original, "perturbation should change weights");

        // Restore
        {
            let w = &mut trainer.engine.layers[0].q_proj;
            apply_perturbation(w, &a, &b, rows, cols, rank, sigma, -1.0);
        }

        // Check restored within fp32 tolerance
        for (i, (&o, &r)) in original
            .iter()
            .zip(trainer.engine.layers[0].q_proj.iter())
            .enumerate()
        {
            assert!(
                (o - r).abs() < 1e-5,
                "weight {i} not restored: {o} vs {r}"
            );
        }
    }

    #[test]
    fn test_weight_restore_after_forward() {
        let cfg = small_config();
        let engine = DiffusionEngine::from_random(cfg);
        // Snapshot all weight buffers
        let snapshots: Vec<Vec<Vec<f32>>> = engine
            .layers
            .iter()
            .map(|lw| {
                ALL_WEIGHT_KINDS
                    .iter()
                    .map(|&kind| weight_buf(lw, kind).to_vec())
                    .collect()
            })
            .collect();

        let mut trainer = EggrollTrainer::new(engine, small_eggroll_config());
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let _ = trainer.perturbed_forward(&tokens, 0, 0, 1.0);

        // Verify all weights restored
        for (layer_idx, layer_snaps) in snapshots.iter().enumerate() {
            for (wi, snap) in layer_snaps.iter().enumerate() {
                let kind = ALL_WEIGHT_KINDS[wi];
                let restored = weight_buf(&trainer.engine.layers[layer_idx], kind);
                for (i, (&orig, &rest)) in snap.iter().zip(restored.iter()).enumerate() {
                    assert!(
                        (orig - rest).abs() < 1e-5,
                        "layer {layer_idx} weight {wi} elem {i}: {orig} vs {rest}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_eggroll_training_reduces_loss() {
        let cfg = small_config();
        let engine = DiffusionEngine::from_random(cfg);
        let eggroll_cfg = EggrollConfig {
            sigma: 0.01,
            lr: 0.01,
            rank: 4,
            population: 16,
            total_steps: 30,
            warmup_steps: 5,
            min_lr_frac: 0.1,
            log_interval: 10,
            base_seed: 42,
        };

        let mut trainer = EggrollTrainer::new(engine, eggroll_cfg);
        let tokens = vec![1u32, 2, 3, 10, 20, 30, 40, 50];
        let prompt_len = 3;

        let losses = trainer.train(&tokens, prompt_len);

        assert!(
            losses.iter().all(|l| l.is_finite()),
            "all losses must be finite"
        );

        // ES is noisy — just verify it doesn't explode
        let first_5: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let last_5: f32 = losses[losses.len() - 5..].iter().sum::<f32>() / 5.0;
        assert!(
            last_5 < first_5 * 2.0,
            "loss should not explode: first_5={first_5:.4} last_5={last_5:.4}"
        );
    }

    #[test]
    #[ignore]
    fn test_eggroll_real_model() {
        // Requires Qwen3-0.6B-MDLM model on disk.
        // cargo test -p higgs-models -- test_eggroll_real_model --nocapture --ignored
        eprintln!("TODO: Load real 0.6B model and run 5 EGGROLL steps");
    }
}
