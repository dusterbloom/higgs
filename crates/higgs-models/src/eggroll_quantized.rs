//! EGGROLL Tier 2 — Quantized MLX model training via additive perturbation.
//!
//! Key insight: instead of mutating bit-packed quantized weights directly,
//! apply the low-rank perturbation as an additive term in the forward pass:
//!
//!   y = quantized_matmul(W, x) + sigma * A @ (B^T @ x)
//!
//! This is mathematically equivalent to `(W + sigma * A @ B^T) @ x` but:
//! - No dequant/requant per forward pass
//! - Works with any quantization (1-4 bit, any group size)
//! - B^T @ x is a rank-r vector (cheap), A @ result maps back to full dim
//!
//! Weight updates accumulate into fp32 delta buffers. Periodic merge via
//! dequant(W) + delta → requant replaces the base quantized weights.

#![allow(
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::str_to_string,
    clippy::missing_const_for_fn,
)]

use std::collections::HashMap;

use mlx_rs::{Array, error::Exception, ops};

use crate::diffusion_eggroll::{EggrollConfig, generate_factors, make_seed};
use crate::qwen3_next::QLinear;

// -----------------------------------------------------------------------
// Perturbed QLinear forward
// -----------------------------------------------------------------------

/// Forward pass through a quantized linear layer with optional delta + perturbation.
///
/// y = qmm(W, x) + x @ delta^T + scale * (x @ B) @ A^T
///
/// where x is [batch, seq, in_dim], and we compute the linear as x @ W^T
/// (MLX quantized_matmul with transpose=true handles this).
///
/// `delta`: accumulated fp32 update, shape [out_dim, in_dim]
/// `perturbation`: (A [out_dim, rank], B [in_dim, rank], scale = sigma * sign)
pub(crate) fn perturbed_qlinear_forward(
    x: &Array,
    qlinear: &QLinear,
    delta: Option<&Array>,
    perturbation: Option<(&Array, &Array, f32)>,
) -> Result<Array, Exception> {
    // Base quantized forward: y = x @ dequant(W)^T via fused kernel
    let mut y = qlinear.forward(x)?;

    // Add accumulated delta: y += x @ delta^T
    if let Some(d) = delta {
        let dt = d.transpose()?;
        y = y.add(&x.matmul(&dt)?)?;
    }

    // Add perturbation: y += scale * (x @ B) @ A^T
    // where B is [in_dim, rank], A is [out_dim, rank]
    if let Some((a, b, scale)) = perturbation {
        let xb = x.matmul(b)?; // [batch, seq, rank] — cheap
        let at = a.transpose()?; // [rank, out_dim]
        let pert = xb.matmul(&at)?; // [batch, seq, out_dim]
        let scale_arr = Array::from_f32(scale);
        y = y.add(&pert.multiply(&scale_arr)?)?;
    }

    Ok(y)
}

// -----------------------------------------------------------------------
// Per-layer state
// -----------------------------------------------------------------------

/// Tracks per-projection delta buffers for a single layer.
pub struct QuantizedLayerState {
    /// Per-projection fp32 delta buffers: "q_proj" -> [out_dim, in_dim]
    pub deltas: HashMap<String, Array>,
}

impl Default for QuantizedLayerState {
    fn default() -> Self {
        Self {
            deltas: HashMap::new(),
        }
    }
}

impl QuantizedLayerState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate an ES update into the delta buffer.
    /// delta[name] += scale * A @ B^T
    pub fn accumulate_delta(
        &mut self,
        name: &str,
        a: &Array,
        b: &Array,
        scale: f32,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), Exception> {
        let bt = b.transpose()?; // [rank, in_dim]
        let ab = a.matmul(&bt)?; // [out_dim, in_dim]
        let scale_arr = Array::from_f32(scale);
        let update = ab.multiply(&scale_arr)?;

        let current = if let Some(d) = self.deltas.get(name) {
            d.add(&update)?
        } else {
            let zeros = Array::zeros::<f32>(&[out_dim as i32, in_dim as i32])?;
            zeros.add(&update)?
        };
        self.deltas.insert(name.into(), current);
        Ok(())
    }
}

// -----------------------------------------------------------------------
// Quantized EGGROLL trainer
// -----------------------------------------------------------------------

/// Projection descriptor for quantized weight targets.
#[derive(Clone)]
pub struct QWeightTarget {
    pub name: String,
    pub out_dim: usize,
    pub in_dim: usize,
}

pub struct EggrollQuantizedTrainer {
    pub layers: Vec<QuantizedLayerState>,
    pub config: EggrollConfig,
    pub weight_targets: Vec<QWeightTarget>,
}

impl EggrollQuantizedTrainer {
    /// Create trainer with empty delta buffers.
    /// `weight_targets` describes each projection in a layer (same for all layers).
    pub fn new(
        n_layers: usize,
        config: EggrollConfig,
        weight_targets: Vec<QWeightTarget>,
    ) -> Self {
        let layers = (0..n_layers)
            .map(|_| QuantizedLayerState::new())
            .collect();
        Self {
            layers,
            config,
            weight_targets,
        }
    }

    /// Generate MLX Array perturbation factors for a specific weight target.
    pub fn generate_perturbation_arrays(
        &self,
        step: usize,
        member: usize,
        layer: usize,
        weight_idx: usize,
        target: &QWeightTarget,
    ) -> (Array, Array) {
        let seed = make_seed(self.config.base_seed, step, member, layer, weight_idx);
        let (a_vec, b_vec) =
            generate_factors(seed, target.out_dim, target.in_dim, self.config.rank);

        let a = Array::from_slice(
            &a_vec,
            &[target.out_dim as i32, self.config.rank as i32],
        );
        let b = Array::from_slice(
            &b_vec,
            &[target.in_dim as i32, self.config.rank as i32],
        );
        (a, b)
    }

    /// Accumulate ES gradient for one step into delta buffers.
    ///
    /// `fitnesses[member]` = loss_minus - loss_plus for each population member.
    pub fn accumulate_es_gradient(
        &mut self,
        fitnesses: &[f32],
        step: usize,
        lr: f32,
    ) -> Result<(), Exception> {
        let n = self.config.population;
        let sigma = self.config.sigma;
        let scale_base = lr / (2.0 * n as f32 * sigma);
        let n_layers = self.layers.len();
        let targets = self.weight_targets.clone();

        for layer_idx in 0..n_layers {
            for (wi, target) in targets.iter().enumerate() {
                for member in 0..n {
                    let fitness = fitnesses[member];
                    if fitness.abs() < 1e-12 {
                        continue;
                    }
                    let (a, b) = self.generate_perturbation_arrays(
                        step, member, layer_idx, wi, target,
                    );
                    let scale = scale_base * fitness;
                    self.layers[layer_idx].accumulate_delta(
                        &target.name,
                        &a,
                        &b,
                        scale,
                        target.out_dim,
                        target.in_dim,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Merge accumulated delta into a QLinear's quantized weights.
    /// Performs: dequant(W) + delta → requant → replace W.
    ///
    /// This is expensive and should be done infrequently (every ~100 steps).
    pub(crate) fn merge_delta(
        &mut self,
        layer_idx: usize,
        proj_name: &str,
        qlinear: &mut QLinear,
    ) -> Result<(), Exception> {
        let delta = match self.layers[layer_idx].deltas.get(proj_name) {
            Some(d) => d.clone(),
            None => return Ok(()),
        };

        // Dequantize current weights to fp32
        let deq = ops::dequantize(
            &*qlinear.weight,
            &*qlinear.scales,
            &*qlinear.biases,
            qlinear.group_size,
            qlinear.bits,
        )?;

        // Add delta and requantize
        let merged = deq.add(&delta)?;
        let (new_w, new_s, new_b) =
            ops::quantize(&merged, qlinear.group_size, qlinear.bits)?;

        // Replace weights
        *qlinear.weight = new_w;
        *qlinear.scales = new_s;
        *qlinear.biases = new_b;

        // Clear the delta buffer
        let zeros = Array::zeros::<f32>(delta.shape())?;
        self.layers[layer_idx]
            .deltas
            .insert(proj_name.into(), zeros);

        Ok(())
    }

    /// Get delta buffer for a projection (if any).
    pub fn get_delta(&self, layer_idx: usize, proj_name: &str) -> Option<&Array> {
        self.layers[layer_idx].deltas.get(proj_name)
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion_eggroll::{EggrollConfig, xorshift_normal};

    fn small_config() -> EggrollConfig {
        EggrollConfig {
            sigma: 0.01,
            lr: 0.001,
            rank: 2,
            population: 4,
            total_steps: 10,
            warmup_steps: 2,
            min_lr_frac: 0.1,
            log_interval: 5,
            base_seed: 42,
        }
    }

    #[test]
    fn test_additive_perturbation_equivalence() {
        // Verify: qmm(W, x) + sigma * A @ (B^T @ x) ≈ (dequant(W) + sigma * A @ B^T) @ x
        let out_dim = 64;
        let in_dim = 64;
        let group_size = 32;
        let bits = 4;
        let sigma = 0.1f32;
        let rank = 2;

        // Create random weight and quantize
        let mut state = 12345u64;
        let raw_data: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| xorshift_normal(&mut state) * 0.02)
            .collect();
        let raw = Array::from_slice(&raw_data, &[out_dim as i32, in_dim as i32]);
        let (qw, qs, qb) = ops::quantize(&raw, group_size, bits).unwrap();

        let qlinear = QLinear {
            weight: mlx_rs::module::Param::new(qw),
            scales: mlx_rs::module::Param::new(qs),
            biases: mlx_rs::module::Param::new(qb),
            group_size,
            bits,
        };

        // Create random input [1, 1, in_dim]
        let x_data: Vec<f32> = (0..in_dim)
            .map(|_| xorshift_normal(&mut state) * 0.1)
            .collect();
        let x = Array::from_slice(&x_data, &[1, 1, in_dim as i32]);

        // Create perturbation factors
        let a_data: Vec<f32> = (0..out_dim * rank)
            .map(|_| xorshift_normal(&mut state))
            .collect();
        let b_data: Vec<f32> = (0..in_dim * rank)
            .map(|_| xorshift_normal(&mut state))
            .collect();
        let a = Array::from_slice(&a_data, &[out_dim as i32, rank as i32]);
        let b = Array::from_slice(&b_data, &[in_dim as i32, rank as i32]);

        // Path A: additive perturbation via perturbed_qlinear_forward
        let y_additive = perturbed_qlinear_forward(
            &x,
            &qlinear,
            None,
            Some((&a, &b, sigma)),
        )
        .unwrap();

        // Path B: dequant + dense perturbation + matmul
        let deq = ops::dequantize(
            &*qlinear.weight,
            &*qlinear.scales,
            &*qlinear.biases,
            group_size,
            bits,
        )
        .unwrap();
        let abt = a.matmul(&b.transpose().unwrap()).unwrap(); // [out, in]
        let scale = Array::from_f32(sigma);
        let w_perturbed = deq.add(&abt.multiply(&scale).unwrap()).unwrap();
        let y_dense = x.matmul(&w_perturbed.transpose().unwrap()).unwrap();

        mlx_rs::transforms::eval([&y_additive, &y_dense]).unwrap();

        // Compare
        let a_flat: Vec<f32> = y_additive.as_slice().to_vec();
        let d_flat: Vec<f32> = y_dense.as_slice().to_vec();
        let max_diff = a_flat
            .iter()
            .zip(d_flat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 0.01,
            "additive vs dense perturbation max diff = {max_diff} (should be < 0.01)"
        );
    }

    #[test]
    fn test_delta_accumulation() {
        let config = small_config();
        let targets = vec![QWeightTarget {
            name: "q_proj".to_string(),
            out_dim: 64,
            in_dim: 64,
        }];
        let mut trainer = EggrollQuantizedTrainer::new(2, config, targets);

        // Simulate fitness signals
        let fitnesses = vec![0.1, -0.2, 0.3, -0.1];
        trainer
            .accumulate_es_gradient(&fitnesses, 0, 0.001)
            .unwrap();

        // Check that deltas exist and are non-zero
        for layer in &trainer.layers {
            let delta = layer.deltas.get("q_proj").unwrap();
            mlx_rs::transforms::eval(std::slice::from_ref(delta)).unwrap();
            let vals: Vec<f32> = delta.as_slice().to_vec();
            let sum_abs: f32 = vals.iter().map(|v| v.abs()).sum();
            assert!(
                sum_abs > 0.0,
                "delta should be non-zero after accumulation"
            );
        }

        // Second step should further accumulate
        let fitnesses2 = vec![0.05, -0.1, 0.2, -0.05];
        trainer
            .accumulate_es_gradient(&fitnesses2, 1, 0.001)
            .unwrap();

        let delta = trainer.layers[0].deltas.get("q_proj").unwrap();
        mlx_rs::transforms::eval(std::slice::from_ref(delta)).unwrap();
        let vals: Vec<f32> = delta.as_slice().to_vec();
        let sum_abs: f32 = vals.iter().map(|v| v.abs()).sum();
        assert!(
            sum_abs > 0.0,
            "delta should grow after second accumulation"
        );
    }

    #[test]
    fn test_quantized_eggroll_step() {
        // One ES step with a quantized model: verify loss is finite.
        let out_dim = 64;
        let in_dim = 64;
        let group_size = 32;
        let bits = 4;

        // Create a small quantized linear layer
        let mut state = 99u64;
        let raw_data: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| xorshift_normal(&mut state) * 0.02)
            .collect();
        let raw = Array::from_slice(&raw_data, &[out_dim as i32, in_dim as i32]);
        let (qw, qs, qb) = ops::quantize(&raw, group_size, bits).unwrap();

        let qlinear = QLinear {
            weight: mlx_rs::module::Param::new(qw),
            scales: mlx_rs::module::Param::new(qs),
            biases: mlx_rs::module::Param::new(qb),
            group_size,
            bits,
        };

        let config = small_config();
        let sigma = config.sigma;
        let rank = config.rank;
        let n = config.population;

        // Create random input
        let seq = 8;
        let x_data: Vec<f32> = (0..seq * in_dim)
            .map(|_| xorshift_normal(&mut state) * 0.1)
            .collect();
        let x = Array::from_slice(&x_data, &[1, seq as i32, in_dim as i32]);

        // Evaluate population: +/- perturbation forward passes
        let mut fitnesses = Vec::with_capacity(n);
        for member in 0..n {
            let seed = make_seed(config.base_seed, 0, member, 0, 0);
            let (a_vec, b_vec) = generate_factors(seed, out_dim, in_dim, rank);
            let a = Array::from_slice(&a_vec, &[out_dim as i32, rank as i32]);
            let b = Array::from_slice(&b_vec, &[in_dim as i32, rank as i32]);

            // +perturbation
            let y_plus =
                perturbed_qlinear_forward(&x, &qlinear, None, Some((&a, &b, sigma)))
                    .unwrap();
            // Simple L2 loss as proxy
            let loss_plus = y_plus.multiply(&y_plus).unwrap().mean(None).unwrap();

            // -perturbation
            let y_minus =
                perturbed_qlinear_forward(&x, &qlinear, None, Some((&a, &b, -sigma)))
                    .unwrap();
            let loss_minus = y_minus.multiply(&y_minus).unwrap().mean(None).unwrap();

            mlx_rs::transforms::eval([&loss_plus, &loss_minus]).unwrap();
            let lp = loss_plus.item::<f32>();
            let lm = loss_minus.item::<f32>();
            assert!(lp.is_finite(), "loss_plus should be finite");
            assert!(lm.is_finite(), "loss_minus should be finite");

            fitnesses.push(lm - lp);
        }

        // Accumulate gradient
        let targets = vec![QWeightTarget {
            name: "test".to_string(),
            out_dim,
            in_dim,
        }];
        let mut trainer = EggrollQuantizedTrainer::new(1, config, targets);
        trainer
            .accumulate_es_gradient(&fitnesses, 0, 0.001)
            .unwrap();

        // Verify delta is finite
        let delta = trainer.layers[0].deltas.get("test").unwrap();
        mlx_rs::transforms::eval(std::slice::from_ref(delta)).unwrap();
        let vals: Vec<f32> = delta.as_slice().to_vec();
        assert!(vals.iter().all(|v| v.is_finite()), "deltas must be finite");
    }
}
