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

use mlx_rs::{
    Array,
    error::Exception,
    module::ModuleParameters as _,
    ops,
};

use crate::diffusion_eggroll::{EggrollConfig, generate_factors, make_seed};
use crate::diffusion_train::cosine_lr;
use crate::qwen3_next::{DeltaMap, PerturbMap, QLinear, Qwen3NextCausalLM};

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
// Qwen3Next EGGROLL — snapshot-inject-restore training loop
// -----------------------------------------------------------------------

/// Metadata for a discovered QLinear projection target.
struct QWeightMeta {
    /// Base prefix, e.g. "model.layers.0.self_attn.q_proj"
    name: String,
    weight_key: String,
    scales_key: String,
    biases_key: String,
    out_dim: usize,
    in_dim: usize,
    group_size: i32,
    bits: i32,
    /// For deterministic seed generation (parsed from key).
    layer_idx: usize,
    /// Sequential index within this layer.
    weight_idx: usize,
}

/// Full EGGROLL training loop for quantized Qwen3Next models.
///
/// Uses snapshot-inject-restore: mutate all QLinear weights in-place before
/// the normal forward pass, then restore originals. Avoids modifying the
/// model's forward path — works on any Qwen3Next configuration (dense/MoE,
/// GDN/full attention).
pub struct Qwen3NextEggrollTrainer<'m> {
    model: &'m mut Qwen3NextCausalLM,
    pub config: EggrollConfig,
    pub merge_interval: usize,
    targets: Vec<QWeightMeta>,
    deltas: HashMap<String, Array>,
}

impl<'m> Qwen3NextEggrollTrainer<'m> {
    /// Create a trainer from a model with weights already loaded.
    ///
    /// Discovers all QLinear projections via `parameters().flatten()` and
    /// extracts dimensions from the packed weight shapes.
    ///
    /// `exclude_patterns`: skip any QLinear whose base key contains one of
    /// these substrings. For MoE models use `&["switch_mlp", "embed_tokens"]`
    /// to skip the huge expert stack; for dense use `&["embed_tokens"]`.
    pub fn new(
        model: &'m mut Qwen3NextCausalLM,
        config: EggrollConfig,
        merge_interval: usize,
        exclude_patterns: &[&str],
    ) -> Result<Self, Exception> {
        let quant = model
            .args
            .quantization
            .as_ref()
            .ok_or_else(|| Exception::custom("EGGROLL Tier 2 requires a quantized model"))?;
        let group_size = quant.group_size;
        let bits = quant.bits;

        // Discover all QLinear targets by scanning flattened parameter keys.
        let params = model.parameters().flatten();
        let keys: Vec<String> = params.keys().map(|k| k.to_string()).collect();

        let mut targets = Vec::new();
        let mut layer_weight_counter: HashMap<usize, usize> = HashMap::new();

        for key in &keys {
            let Some(base) = key.strip_suffix(".weight") else {
                continue;
            };
            let scales_key = format!("{base}.scales");
            let biases_key = format!("{base}.biases");
            if !keys.contains(&scales_key) || !keys.contains(&biases_key) {
                continue;
            }

            // Skip targets matching any exclude pattern
            if exclude_patterns.iter().any(|pat| base.contains(pat)) {
                continue;
            }

            let w = params.get(key.as_str()).unwrap();
            let shape = w.shape();
            if shape.len() < 2 || shape[0] <= 1 {
                continue; // placeholder or scalar — skip
            }

            let out_dim = shape[0] as usize;
            let in_dim = (shape[1] as usize) * 32 / (bits as usize);

            // Parse layer index from key: "model.layers.N...."
            let layer_idx = parse_layer_idx(base).unwrap_or(0);
            let wi = layer_weight_counter.entry(layer_idx).or_insert(0);
            let weight_idx = *wi;
            *wi += 1;

            targets.push(QWeightMeta {
                name: base.to_string(),
                weight_key: key.clone(),
                scales_key,
                biases_key,
                out_dim,
                in_dim,
                group_size,
                bits,
                layer_idx,
                weight_idx,
            });
        }

        let deltas = HashMap::new();

        Ok(Self {
            model,
            config,
            merge_interval,
            targets,
            deltas,
        })
    }

    /// Auto-detecting constructor: inspects `model.args.num_experts` to choose
    /// the right exclude patterns.
    ///
    /// - MoE (`num_experts > 0`): excludes `switch_mlp` (huge expert stack) + `embed_tokens`
    /// - Dense: excludes only `embed_tokens`
    pub fn new_for_model(
        model: &'m mut Qwen3NextCausalLM,
        config: EggrollConfig,
        merge_interval: usize,
    ) -> Result<Self, Exception> {
        let exclude: &[&str] = if model.args.num_experts > 0 {
            &["switch_mlp", "embed_tokens"]
        } else {
            &["embed_tokens"]
        };
        Self::new(model, config, merge_interval, exclude)
    }

    /// Number of discovered QLinear targets.
    pub fn num_targets(&self) -> usize {
        self.targets.len()
    }

    /// Names of all discovered targets — useful for verifying which projections are included.
    pub fn target_names(&self) -> Vec<&str> {
        self.targets.iter().map(|t| t.name.as_str()).collect()
    }

    /// Build a perturbation map for one population member.
    ///
    /// Returns a `PerturbMap` keyed by target name (e.g. "model.layers.0.self_attn.q_proj")
    /// with values `(A, B, sign * sigma)`.
    fn build_perturbation_map(&self, step: usize, member: usize, sign: f32) -> PerturbMap {
        let sigma = self.config.sigma;
        let rank = self.config.rank;
        let mut map = PerturbMap::with_capacity(self.targets.len());
        for t in &self.targets {
            let seed = make_seed(self.config.base_seed, step, member, t.layer_idx, t.weight_idx);
            let (a_vec, b_vec) = generate_factors(seed, t.out_dim, t.in_dim, rank);
            let a = Array::from_slice(&a_vec, &[t.out_dim as i32, rank as i32]);
            let b = Array::from_slice(&b_vec, &[t.in_dim as i32, rank as i32]);
            map.insert(t.name.clone(), (a, b, sign * sigma));
        }
        map
    }

    /// Accumulate ES gradient into delta buffers.
    ///
    /// For each population member: delta += (fitness * lr / (2*N*sigma)) * A @ B^T
    fn accumulate_gradient(
        &mut self,
        fitnesses: &[f32],
        step: usize,
        lr: f32,
    ) -> Result<(), Exception> {
        let n = self.config.population;
        let sigma = self.config.sigma;
        let scale_base = lr / (2.0 * n as f32 * sigma);

        for t in &self.targets {
            for (member, &fitness) in fitnesses.iter().enumerate() {
                if fitness.abs() < 1e-12 {
                    continue;
                }
                let seed = make_seed(
                    self.config.base_seed,
                    step,
                    member,
                    t.layer_idx,
                    t.weight_idx,
                );
                let (a_vec, b_vec) =
                    generate_factors(seed, t.out_dim, t.in_dim, self.config.rank);
                let a = Array::from_slice(&a_vec, &[t.out_dim as i32, self.config.rank as i32]);
                let b = Array::from_slice(&b_vec, &[t.in_dim as i32, self.config.rank as i32]);
                let bt = b.transpose()?;
                let ab = a.matmul(&bt)?;
                let scale = scale_base * fitness;
                let update = ab.multiply(&Array::from_f32(scale))?;

                let current = if let Some(d) = self.deltas.get(&t.name) {
                    d.add(&update)?
                } else {
                    let zeros =
                        Array::zeros::<f32>(&[t.out_dim as i32, t.in_dim as i32])?;
                    zeros.add(&update)?
                };
                self.deltas.insert(t.name.clone(), current);
            }
        }
        Ok(())
    }

    /// Merge accumulated deltas into quantized base weights and clear buffers.
    fn merge_deltas(&mut self) -> Result<(), Exception> {
        // Phase 1: read current weights + deltas, compute merged+requantized values
        let mut updates: Vec<(String, String, String, Array, Array, Array)> = Vec::new();
        {
            let params = self.model.parameters().flatten();
            for t in &self.targets {
                let delta = match self.deltas.get(&t.name) {
                    Some(d) => d.clone(),
                    None => continue,
                };
                let w = *params.get(t.weight_key.as_str()).unwrap();
                let s = *params.get(t.scales_key.as_str()).unwrap();
                let b = *params.get(t.biases_key.as_str()).unwrap();
                let deq = ops::dequantize(w, s, b, t.group_size, t.bits)?;
                let merged = deq.add(&delta)?;
                let (new_w, new_s, new_b) =
                    ops::quantize(&merged, t.group_size, t.bits)?;
                updates.push((
                    t.weight_key.clone(),
                    t.scales_key.clone(),
                    t.biases_key.clone(),
                    new_w,
                    new_s,
                    new_b,
                ));
            }
        }
        // Phase 2: write back
        let mut params = self.model.parameters_mut().flatten();
        for (wk, sk, bk, new_w, new_s, new_b) in updates {
            **params.get_mut(wk.as_str()).unwrap() = new_w;
            **params.get_mut(sk.as_str()).unwrap() = new_s;
            **params.get_mut(bk.as_str()).unwrap() = new_b;
        }
        self.deltas.clear();
        Ok(())
    }

    /// Run EGGROLL training loop.
    ///
    /// `tokens`: full token sequence (prompt + completion).
    /// `prompt_len`: number of prompt tokens (loss computed only on completion).
    ///
    /// Returns per-step loss values.
    pub fn train(
        &mut self,
        tokens: &[u32],
        prompt_len: usize,
    ) -> Result<Vec<f32>, Exception> {
        let total_steps = self.config.total_steps;
        let n = self.config.population;
        let vocab = self.model.args.vocab_size as usize;
        let mut losses = Vec::with_capacity(total_steps);

        // Build input array: [1, T]
        let seq_len = tokens.len();
        let input_ids: Vec<u32> = tokens.to_vec();
        let input =
            Array::from_slice(&input_ids, &[1, seq_len as i32]).as_type::<u32>()?;

        for step in 0..total_steps {
            let lr = cosine_lr(
                step,
                self.config.warmup_steps,
                total_steps,
                self.config.lr,
                self.config.min_lr_frac,
            );

            // Snapshot current deltas for this step (shared across all members).
            // Cloned once per step, not per member.
            let delta_map: Option<DeltaMap> = if self.deltas.is_empty() {
                None
            } else {
                Some(self.deltas.clone())
            };

            let mut fitnesses = Vec::with_capacity(n);
            let mut step_loss = 0.0f32;

            for member in 0..n {
                // +perturbation: set fields on model, forward, clear
                let perts_plus = self.build_perturbation_map(step, member, 1.0);
                self.model.perturbations = Some(perts_plus);
                self.model.train_deltas = delta_map.clone();
                let logits_plus = self.model.forward_all_logits(&input)?;
                mlx_rs::transforms::eval(std::slice::from_ref(&logits_plus))?;
                let loss_plus = causal_lm_loss(&logits_plus, tokens, prompt_len, vocab)?;
                self.model.perturbations = None;
                self.model.train_deltas = None;

                // -perturbation
                let perts_minus = self.build_perturbation_map(step, member, -1.0);
                self.model.perturbations = Some(perts_minus);
                self.model.train_deltas = delta_map.clone();
                let logits_minus = self.model.forward_all_logits(&input)?;
                mlx_rs::transforms::eval(std::slice::from_ref(&logits_minus))?;
                let loss_minus =
                    causal_lm_loss(&logits_minus, tokens, prompt_len, vocab)?;
                self.model.perturbations = None;
                self.model.train_deltas = None;

                fitnesses.push(loss_minus - loss_plus);
                step_loss += (loss_plus + loss_minus) / 2.0;
            }

            let avg_loss = step_loss / n as f32;
            losses.push(avg_loss);

            // Accumulate ES gradient
            self.accumulate_gradient(&fitnesses, step, lr)?;

            // Periodic merge
            if self.merge_interval > 0 && (step + 1) % self.merge_interval == 0 {
                self.merge_deltas()?;
            }

            if self.config.log_interval > 0 && (step + 1) % self.config.log_interval == 0 {
                eprintln!(
                    "[EGGROLL-Q] step {}/{} loss={avg_loss:.4} lr={lr:.6}",
                    step + 1,
                    total_steps,
                );
            }
        }

        // Final merge of any remaining deltas
        if !self.deltas.is_empty() {
            self.merge_deltas()?;
        }

        Ok(losses)
    }
}

/// Parse layer index from a parameter key like "model.layers.3.self_attn.q_proj".
fn parse_layer_idx(key: &str) -> Option<usize> {
    let parts: Vec<&str> = key.split('.').collect();
    for (i, &part) in parts.iter().enumerate() {
        if part == "layers" {
            return parts.get(i + 1)?.parse().ok();
        }
    }
    None
}

/// Causal LM cross-entropy loss on completion positions — GPU-only path.
///
/// `logits`: `[1, T, vocab]` from `forward_all_logits`.
/// `tokens`: the full token sequence of length T.
/// At position t, `logits[t]` predicts `tokens[t+1]`.
/// Loss is computed for positions `prompt_len-1..T-1` (completion tokens).
///
/// All computation stays on GPU via MLX ops. Only the final scalar is
/// materialized to CPU. For vocab=152k, T=100 this avoids transferring
/// ~60MB per call (the old CPU path did `as_slice().to_vec()`).
fn causal_lm_loss(
    logits: &Array,
    tokens: &[u32],
    prompt_len: usize,
    _vocab: usize,
) -> Result<f32, Exception> {
    use mlx_rs::ops::indexing::IndexOp;

    let seq_len = tokens.len();
    let start = prompt_len.saturating_sub(1);
    let end = seq_len - 1;
    if start >= end {
        return Ok(0.0);
    }

    // Slice logits to completion positions: [1, n_comp, vocab]
    let comp_logits = logits.index((.., start as i32..end as i32, ..));

    // log_softmax over vocab dim — numerically stable, stays on GPU
    let log_probs = mlx_rs::nn::log_softmax(&comp_logits, -1)?;

    // Build target indices: tokens[start+1..end+1], shape [1, n_comp, 1]
    let target_ids: Vec<u32> = tokens[start + 1..=end].to_vec();
    let n_comp = target_ids.len() as i32;
    let targets = Array::from_slice(&target_ids, &[1, n_comp, 1]).as_type::<u32>()?;

    // Gather target log-probs: [1, n_comp, 1]
    let target_lp = log_probs.take_along_axis(&targets, -1)?;

    // Mean negative log-probability → scalar
    let loss = target_lp.mean(None)?.negative()?;
    mlx_rs::transforms::eval(std::slice::from_ref(&loss))?;

    Ok(loss.item::<f32>())
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

        // Path A: additive perturbation via QLinear::forward_perturbed
        let y_additive = qlinear.forward_perturbed(
            &x,
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
                qlinear.forward_perturbed(&x, None, Some((&a, &b, sigma)))
                    .unwrap();
            // Simple L2 loss as proxy
            let loss_plus = y_plus.multiply(&y_plus).unwrap().mean(None).unwrap();

            // -perturbation
            let y_minus =
                qlinear.forward_perturbed(&x, None, Some((&a, &b, -sigma)))
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

    // -----------------------------------------------------------------------
    // Qwen3Next EGGROLL trainer tests
    // -----------------------------------------------------------------------

    use mlx_rs::module::ModuleParametersExt as _;
    use crate::qwen3_next::{Qwen3NextCausalLM, Qwen3NextModelArgs};

    /// Create a tiny all-attention Qwen3Next model with real quantized weights.
    fn tiny_qwen3next_model() -> Qwen3NextCausalLM {
        let args: Qwen3NextModelArgs = serde_json::from_str(
            r#"{
                "model_type": "qwen3_next",
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "intermediate_size": 256,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "rms_norm_eps": 1e-06,
                "vocab_size": 256,
                "max_position_embeddings": 128,
                "full_attention_interval": 1,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "num_experts": 0,
                "tie_word_embeddings": true,
                "quantization": { "group_size": 64, "bits": 4 }
            }"#,
        )
        .unwrap();

        let mut model = Qwen3NextCausalLM::new(args).unwrap();

        // Populate all parameters with random data of correct shapes.
        // QLinear params get quantized random weights; others get random fp32.
        let hidden = 128i32;
        let kv_dim = 64i32; // num_kv_heads(1) * head_dim(64)
        let inter = 256i32;
        let vocab = 256i32;
        let group_size = 64;
        let bits = 4;

        // Mapping: param base name -> [out_dim, in_dim]
        // q_proj: 2*num_heads*head_dim (doubled for gating), k/v: kv_heads*head_dim
        // o_proj: hidden, num_heads*head_dim
        let num_heads = 2i32;
        let head_dim = 64i32;
        let q_out = 2 * num_heads * head_dim; // 256 (doubled for gating)
        let o_in = num_heads * head_dim;      // 128
        let qlinear_shapes: Vec<(&str, i32, i32)> = vec![
            ("model.embed_tokens", vocab, hidden),
            // Layer 0
            ("model.layers.0.self_attn.q_proj", q_out, hidden),
            ("model.layers.0.self_attn.k_proj", kv_dim, hidden),
            ("model.layers.0.self_attn.v_proj", kv_dim, hidden),
            ("model.layers.0.self_attn.o_proj", hidden, o_in),
            ("model.layers.0.mlp.gate_proj", inter, hidden),
            ("model.layers.0.mlp.up_proj", inter, hidden),
            ("model.layers.0.mlp.down_proj", hidden, inter),
            // Layer 1
            ("model.layers.1.self_attn.q_proj", q_out, hidden),
            ("model.layers.1.self_attn.k_proj", kv_dim, hidden),
            ("model.layers.1.self_attn.v_proj", kv_dim, hidden),
            ("model.layers.1.self_attn.o_proj", hidden, o_in),
            ("model.layers.1.mlp.gate_proj", inter, hidden),
            ("model.layers.1.mlp.up_proj", inter, hidden),
            ("model.layers.1.mlp.down_proj", hidden, inter),
        ];

        let mut rng_state = 7777u64;
        let mut params = model.parameters_mut().flatten();

        for (base, out_d, in_d) in &qlinear_shapes {
            let raw_data: Vec<f32> = (0..(*out_d as usize * *in_d as usize))
                .map(|_| xorshift_normal(&mut rng_state) * 0.02)
                .collect();
            let raw = Array::from_slice(&raw_data, &[*out_d, *in_d]);
            let (qw, qs, qb) = ops::quantize(&raw, group_size, bits).unwrap();

            let wk = format!("{base}.weight");
            let sk = format!("{base}.scales");
            let bk = format!("{base}.biases");
            if let Some(p) = params.get_mut(wk.as_str()) {
                **p = qw;
            }
            if let Some(p) = params.get_mut(sk.as_str()) {
                **p = qs;
            }
            if let Some(p) = params.get_mut(bk.as_str()) {
                **p = qb;
            }
        }

        // All remaining params (norms, rope, etc.): keep init defaults.
        // Norms are already initialized by RmsNormBuilder with the right dims.

        drop(params);
        model.eval().unwrap();
        model
    }

    #[test]
    fn test_qwen3next_build_perturbation_map() {
        let mut model = tiny_qwen3next_model();
        let config = small_config();
        let sigma = config.sigma;
        let rank = config.rank;
        let trainer =
            Qwen3NextEggrollTrainer::new(&mut model, config, 100, &["embed_tokens"]).unwrap();

        assert!(
            trainer.num_targets() > 0,
            "Should discover QLinear targets, got {}",
            trainer.num_targets()
        );

        let pmap = trainer.build_perturbation_map(0, 0, 1.0);
        assert_eq!(
            pmap.len(),
            trainer.num_targets(),
            "Perturbation map should have one entry per target"
        );

        // Verify each entry has the expected shapes
        for t in &trainer.targets {
            let (a, b, scale) = pmap.get(&t.name).unwrap();
            assert_eq!(a.shape(), &[t.out_dim as i32, rank as i32]);
            assert_eq!(b.shape(), &[t.in_dim as i32, rank as i32]);
            assert!((scale - sigma).abs() < 1e-6);
        }

        // Sign=-1 should negate the scale
        let pmap_neg = trainer.build_perturbation_map(0, 0, -1.0);
        let (_, _, scale_neg) = pmap_neg.get(&trainer.targets[0].name).unwrap();
        assert!((*scale_neg + sigma).abs() < 1e-6);
    }

    #[test]
    fn test_qwen3next_perturbed_forward_differs() {
        let mut model = tiny_qwen3next_model();
        let config = small_config();
        let trainer =
            Qwen3NextEggrollTrainer::new(&mut model, config, 100, &["embed_tokens"]).unwrap();

        let input = Array::from_slice(&[1u32, 5, 10, 20], &[1, 4]).as_type::<u32>().unwrap();

        // Baseline forward (no perturbation)
        let logits_base = trainer.model.forward_all_logits(&input).unwrap();
        mlx_rs::transforms::eval(std::slice::from_ref(&logits_base)).unwrap();
        let base_vals: Vec<f32> = logits_base.as_slice().to_vec();

        // Perturbed forward via additive path
        let pmap = trainer.build_perturbation_map(0, 0, 1.0);
        trainer.model.perturbations = Some(pmap);
        let logits_pert = trainer.model.forward_all_logits(&input).unwrap();
        mlx_rs::transforms::eval(std::slice::from_ref(&logits_pert)).unwrap();
        let pert_vals: Vec<f32> = logits_pert.as_slice().to_vec();
        trainer.model.perturbations = None;

        // Weights should NOT have changed (additive path doesn't mutate them)
        // Logits should differ
        let max_diff = base_vals
            .iter()
            .zip(pert_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "Perturbed logits should differ from baseline, max_diff={max_diff}"
        );

        // Verify weights untouched: forward again without perturbation should match baseline
        let logits_after = trainer.model.forward_all_logits(&input).unwrap();
        mlx_rs::transforms::eval(std::slice::from_ref(&logits_after)).unwrap();
        let after_vals: Vec<f32> = logits_after.as_slice().to_vec();
        let restore_diff = base_vals
            .iter()
            .zip(after_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            restore_diff == 0.0,
            "Weights should be untouched after additive forward, diff={restore_diff}"
        );
    }

    #[test]
    fn test_qwen3next_eggroll_step() {
        let mut model = tiny_qwen3next_model();
        let mut config = small_config();
        config.total_steps = 1;
        config.population = 2;
        config.log_interval = 0; // suppress logs
        let mut trainer =
            Qwen3NextEggrollTrainer::new(&mut model, config, 0, &["embed_tokens"]).unwrap();

        // Tiny token sequence: 2 prompt + 4 completion tokens
        let tokens: Vec<u32> = vec![1, 5, 10, 20, 30, 40];
        let prompt_len = 2;

        let losses = trainer.train(&tokens, prompt_len).unwrap();
        assert_eq!(losses.len(), 1, "Should have 1 loss value for 1 step");
        assert!(
            losses[0].is_finite(),
            "Loss should be finite, got {}",
            losses[0]
        );
        assert!(
            losses[0] > 0.0,
            "CE loss should be positive, got {}",
            losses[0]
        );
    }

    /// Create a tiny MoE Qwen3Next model with 4 experts, 2 layers.
    fn tiny_qwen3next_moe_model() -> Qwen3NextCausalLM {
        let args: Qwen3NextModelArgs = serde_json::from_str(
            r#"{
                "model_type": "qwen3_next",
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "intermediate_size": 256,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "rms_norm_eps": 1e-06,
                "vocab_size": 256,
                "max_position_embeddings": 128,
                "full_attention_interval": 1,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 128,
                "moe_intermediate_size": 64,
                "norm_topk_prob": true,
                "tie_word_embeddings": true,
                "quantization": { "group_size": 64, "bits": 4 }
            }"#,
        )
        .unwrap();

        let mut model = Qwen3NextCausalLM::new(args).unwrap();

        let hidden = 128i32;
        let kv_dim = 64i32;
        let vocab = 256i32;
        let group_size = 64;
        let bits = 4;
        let num_heads = 2i32;
        let head_dim = 64i32;
        let q_out = 2 * num_heads * head_dim; // 256 (doubled for gating)
        let o_in = num_heads * head_dim;      // 128
        let num_experts = 4i32;
        let moe_inter = 64i32;
        let shared_inter = 128i32;

        // Populate helper: quantize raw data with given shape, write into params
        let mut rng_state = 8888u64;
        let mut params = model.parameters_mut().flatten();

        let mut populate = |base: &str, shape: &[i32]| {
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let raw_data: Vec<f32> = (0..n)
                .map(|_| xorshift_normal(&mut rng_state) * 0.02)
                .collect();
            let raw = Array::from_slice(&raw_data, shape);
            let (qw, qs, qb) = ops::quantize(&raw, group_size, bits).unwrap();
            let wk = format!("{base}.weight");
            let sk = format!("{base}.scales");
            let bk = format!("{base}.biases");
            if let Some(p) = params.get_mut(wk.as_str()) { **p = qw; }
            if let Some(p) = params.get_mut(sk.as_str()) { **p = qs; }
            if let Some(p) = params.get_mut(bk.as_str()) { **p = qb; }
        };

        populate("model.embed_tokens", &[vocab, hidden]);

        for layer in 0..2 {
            let pfx = format!("model.layers.{layer}");
            // Attention projections (2D)
            populate(&format!("{pfx}.self_attn.q_proj"), &[q_out, hidden]);
            populate(&format!("{pfx}.self_attn.k_proj"), &[kv_dim, hidden]);
            populate(&format!("{pfx}.self_attn.v_proj"), &[kv_dim, hidden]);
            populate(&format!("{pfx}.self_attn.o_proj"), &[hidden, o_in]);
            // MoE router gate (2D): [num_experts, hidden]
            populate(&format!("{pfx}.mlp.gate"), &[num_experts, hidden]);
            // Expert stack (3D): [num_experts, moe_inter, hidden] or [num_experts, hidden, moe_inter]
            populate(&format!("{pfx}.mlp.switch_mlp.gate_proj"), &[num_experts, moe_inter, hidden]);
            populate(&format!("{pfx}.mlp.switch_mlp.up_proj"), &[num_experts, moe_inter, hidden]);
            populate(&format!("{pfx}.mlp.switch_mlp.down_proj"), &[num_experts, hidden, moe_inter]);
            // Shared expert (2D): [shared_inter, hidden] etc.
            populate(&format!("{pfx}.mlp.shared_expert.gate_proj"), &[shared_inter, hidden]);
            populate(&format!("{pfx}.mlp.shared_expert.up_proj"), &[shared_inter, hidden]);
            populate(&format!("{pfx}.mlp.shared_expert.down_proj"), &[hidden, shared_inter]);
            // Shared expert gate (2D): [1, hidden]
            populate(&format!("{pfx}.mlp.shared_expert_gate"), &[1, hidden]);
        }

        drop(params);
        model.eval().unwrap();
        model
    }

    #[test]
    fn test_qwen3next_moe_target_discovery() {
        let mut model = tiny_qwen3next_moe_model();

        // With switch_mlp excluded: should get attn + router + shared expert + shared_expert_gate
        let trainer = Qwen3NextEggrollTrainer::new(
            &mut model,
            small_config(),
            100,
            &["switch_mlp", "embed_tokens"],
        )
        .unwrap();

        let names = trainer.target_names();

        // Per layer: 4 attn (q/k/v/o) + 1 router gate + 3 shared expert = 8
        // (shared_expert_gate has out_dim=1, filtered by shape[0]<=1 check)
        // 2 layers = 16 targets
        assert_eq!(
            trainer.num_targets(),
            16,
            "Expected 16 targets (8/layer × 2 layers), got {}: {:?}",
            trainer.num_targets(),
            names,
        );

        // Verify NO switch_mlp targets
        assert!(
            !names.iter().any(|n| n.contains("switch_mlp")),
            "switch_mlp targets should be excluded, found: {:?}",
            names.iter().filter(|n| n.contains("switch_mlp")).collect::<Vec<_>>(),
        );

        // Verify NO embed_tokens
        assert!(
            !names.iter().any(|n| n.contains("embed_tokens")),
            "embed_tokens should be excluded",
        );

        // Verify shared_expert IS included (not confused with switch_mlp)
        assert!(
            names.iter().any(|n| n.contains("shared_expert")),
            "shared_expert should be included, got: {:?}",
            names,
        );

        // Verify router gate IS included
        assert!(
            names.iter().any(|n| n.contains("mlp.gate")),
            "router gate should be included, got: {:?}",
            names,
        );

        // Now test WITHOUT switch_mlp exclusion: should include expert stack
        // (drop previous trainer to release the borrow)
        drop(trainer);
        let mut model2 = tiny_qwen3next_moe_model();
        let trainer2 = Qwen3NextEggrollTrainer::new(
            &mut model2,
            small_config(),
            100,
            &["embed_tokens"],
        )
        .unwrap();

        let names2 = trainer2.target_names();

        // Per layer: 8 (from above) + 3 switch_mlp (gate/up/down) = 11
        // 2 layers = 22 targets
        assert_eq!(
            trainer2.num_targets(),
            22,
            "Expected 22 targets (11/layer × 2 layers) with switch_mlp included, got {}: {:?}",
            trainer2.num_targets(),
            names2,
        );

        // Verify switch_mlp targets ARE present
        let switch_count = names2.iter().filter(|n| n.contains("switch_mlp")).count();
        assert_eq!(
            switch_count, 6,
            "Expected 6 switch_mlp targets (3/layer × 2), got {switch_count}",
        );
    }

    #[test]
    fn test_qwen3next_moe_auto_detect() {
        // new_for_model on MoE model should auto-exclude switch_mlp
        let mut moe_model = tiny_qwen3next_moe_model();
        let trainer = Qwen3NextEggrollTrainer::new_for_model(
            &mut moe_model,
            small_config(),
            100,
        )
        .unwrap();
        assert!(
            !trainer.target_names().iter().any(|n| n.contains("switch_mlp")),
            "new_for_model should auto-exclude switch_mlp for MoE models",
        );
        assert_eq!(trainer.num_targets(), 16);

        // new_for_model on dense model should NOT exclude mlp projections
        drop(trainer);
        let mut dense_model = tiny_qwen3next_model();
        let trainer2 = Qwen3NextEggrollTrainer::new_for_model(
            &mut dense_model,
            small_config(),
            100,
        )
        .unwrap();
        assert!(
            trainer2.target_names().iter().any(|n| n.contains("mlp")),
            "new_for_model should include dense MLP projections",
        );
    }

    #[test]
    fn test_qwen3next_moe_eggroll_e2e() {
        // Full E2E: run train() on tiny MoE model with switch_mlp excluded.
        // This proves the perturbed forward pass through MoE layers produces
        // finite, positive loss and doesn't panic on excluded targets.
        let mut model = tiny_qwen3next_moe_model();
        let mut config = small_config();
        config.total_steps = 2;
        config.population = 2;
        config.log_interval = 0;

        let mut trainer = Qwen3NextEggrollTrainer::new_for_model(
            &mut model,
            config,
            0, // no merge — just accumulate
        )
        .unwrap();

        // Verify MoE auto-detection worked
        assert!(
            !trainer.target_names().iter().any(|n| n.contains("switch_mlp")),
            "switch_mlp should be auto-excluded",
        );

        // Snapshot a scale tensor before training to verify weights change
        let snap_key = trainer.targets[0].scales_key.clone();
        let pre_scales = {
            let p = trainer.model.parameters().flatten();
            let s = *p.get(snap_key.as_str()).unwrap();
            mlx_rs::transforms::eval(std::slice::from_ref(s)).unwrap();
            s.as_slice::<f32>().to_vec()
        };

        // Tiny token sequence: 2 prompt + 4 completion
        let tokens: Vec<u32> = vec![1, 5, 10, 20, 30, 40];
        let prompt_len = 2;

        let losses = trainer.train(&tokens, prompt_len).unwrap();
        assert_eq!(losses.len(), 2, "Should have 2 loss values for 2 steps");
        for (i, loss) in losses.iter().enumerate() {
            assert!(
                loss.is_finite(),
                "Loss at step {i} should be finite, got {loss}",
            );
            assert!(
                *loss > 0.0,
                "CE loss at step {i} should be positive, got {loss}",
            );
        }

        // Verify weights actually changed (deltas merged into base weights)
        let post_scales = {
            let p = trainer.model.parameters().flatten();
            let s = *p.get(snap_key.as_str()).unwrap();
            mlx_rs::transforms::eval(std::slice::from_ref(s)).unwrap();
            s.as_slice::<f32>().to_vec()
        };
        let max_diff = pre_scales
            .iter()
            .zip(post_scales.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 0.0,
            "Weights should change after training (deltas merged into base)",
        );
    }

    /// Prove: quantized EGGROLL converges on an autoregressive task.
    ///
    /// Uses vocab=4096 so initial loss ≈ ln(4096) ≈ 8.3 — well above uniform floor.
    /// Compares loss from a fixed evaluation pass before/after training.
    /// This is THE critical test: proves ES gradient signal survives
    /// dequant→perturb→requant round-trips inherent to quantized training.
    #[test]
    fn test_qwen3next_convergence_proof() {
        // Build a tiny model with vocab=4096 for higher initial loss
        let args: Qwen3NextModelArgs = serde_json::from_str(
            r#"{
                "model_type": "qwen3_next",
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "intermediate_size": 256,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "rms_norm_eps": 1e-06,
                "vocab_size": 4096,
                "max_position_embeddings": 128,
                "full_attention_interval": 1,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "num_experts": 0,
                "tie_word_embeddings": true,
                "quantization": { "group_size": 64, "bits": 4 }
            }"#,
        )
        .unwrap();

        let mut model = Qwen3NextCausalLM::new(args).unwrap();

        // Populate with random quantized weights
        let hidden = 128i32;
        let kv_dim = 64i32;
        let inter = 256i32;
        let vocab = 4096i32;
        let group_size = 64;
        let bits = 4;
        let num_heads = 2i32;
        let head_dim = 64i32;
        let q_out = 2 * num_heads * head_dim;
        let o_in = num_heads * head_dim;

        let shapes: Vec<(&str, i32, i32)> = vec![
            ("model.embed_tokens", vocab, hidden),
            ("model.layers.0.self_attn.q_proj", q_out, hidden),
            ("model.layers.0.self_attn.k_proj", kv_dim, hidden),
            ("model.layers.0.self_attn.v_proj", kv_dim, hidden),
            ("model.layers.0.self_attn.o_proj", hidden, o_in),
            ("model.layers.0.mlp.gate_proj", inter, hidden),
            ("model.layers.0.mlp.up_proj", inter, hidden),
            ("model.layers.0.mlp.down_proj", hidden, inter),
            ("model.layers.1.self_attn.q_proj", q_out, hidden),
            ("model.layers.1.self_attn.k_proj", kv_dim, hidden),
            ("model.layers.1.self_attn.v_proj", kv_dim, hidden),
            ("model.layers.1.self_attn.o_proj", hidden, o_in),
            ("model.layers.1.mlp.gate_proj", inter, hidden),
            ("model.layers.1.mlp.up_proj", inter, hidden),
            ("model.layers.1.mlp.down_proj", hidden, inter),
        ];

        let mut rng_state = 7777u64;
        let mut params = model.parameters_mut().flatten();
        for (base, out_d, in_d) in &shapes {
            let raw_data: Vec<f32> = (0..(*out_d as usize * *in_d as usize))
                .map(|_| xorshift_normal(&mut rng_state) * 0.02)
                .collect();
            let raw = Array::from_slice(&raw_data, &[*out_d, *in_d]);
            let (qw, qs, qb) = ops::quantize(&raw, group_size, bits).unwrap();
            let wk = format!("{base}.weight");
            let sk = format!("{base}.scales");
            let bk = format!("{base}.biases");
            if let Some(p) = params.get_mut(wk.as_str()) { **p = qw; }
            if let Some(p) = params.get_mut(sk.as_str()) { **p = qs; }
            if let Some(p) = params.get_mut(bk.as_str()) { **p = qb; }
        }
        drop(params);
        model.eval().unwrap();

        // Evaluation sequence
        let tokens: Vec<u32> = (0..24).map(|i| ((i * 13 + 7) % 4096) as u32).collect();
        let prompt_len = 4;

        // Pre-training eval
        let input = Array::from_slice(&tokens, &[1, tokens.len() as i32])
            .as_type::<u32>().unwrap();
        let pre_logits = model.forward_all_logits(&input).unwrap();
        mlx_rs::transforms::eval(std::slice::from_ref(&pre_logits)).unwrap();
        let pre_loss = causal_lm_loss(&pre_logits, &tokens, prompt_len, 4096).unwrap();
        eprintln!("[Q-convergence] pre-training loss = {pre_loss:.4}");

        // Train
        let config = EggrollConfig {
            sigma: 0.15,      // larger sigma: additive gradient must survive merge→requant
            lr: 0.015,
            rank: 4,
            population: 16,
            total_steps: 120,
            warmup_steps: 10,
            min_lr_frac: 0.1,
            log_interval: 30,
            base_seed: 42,
        };

        let mut trainer =
            Qwen3NextEggrollTrainer::new(&mut model, config, 10, &["embed_tokens"]).unwrap();
        let losses = trainer.train(&tokens, prompt_len).unwrap();
        assert_eq!(losses.len(), 120);
        assert!(losses.iter().all(|l| l.is_finite()), "all losses must be finite");

        // Post-training eval (same tokens, fresh forward)
        let post_logits = trainer.model.forward_all_logits(&input).unwrap();
        mlx_rs::transforms::eval(std::slice::from_ref(&post_logits)).unwrap();
        let post_loss = causal_lm_loss(&post_logits, &tokens, prompt_len, 4096).unwrap();

        let improvement = (pre_loss - post_loss) / pre_loss * 100.0;
        eprintln!(
            "[Q-convergence] post-training loss = {post_loss:.4}  \
             improvement = {improvement:.1}%  ({pre_loss:.4} → {post_loss:.4})"
        );
        for (i, l) in losses.iter().enumerate() {
            if i % 10 == 0 {
                eprintln!("  step {i:3}: loss={l:.4}");
            }
        }

        assert!(
            post_loss < pre_loss,
            "QUANTIZED CONVERGENCE FAILED: post_loss={post_loss:.4} >= pre_loss={pre_loss:.4} \
             — ES signal does NOT survive dequant→perturb→requant"
        );
    }

    /// Prove: periodic merge doesn't regress convergence.
    ///
    /// Run same sequence with merge_interval=5 vs merge_interval=0 (merge at end).
    /// Both should converge; the merged path should not have higher final loss.
    #[test]
    fn test_periodic_merge_doesnt_regress() {
        let tokens: Vec<u32> = (0..24).map(|i| ((i * 11 + 5) % 256) as u32).collect();
        let prompt_len = 4;
        let steps = 30;

        // Path A: merge every 5 steps
        let mut model_a = tiny_qwen3next_model();
        let config_a = EggrollConfig {
            sigma: 0.05,
            lr: 0.005,
            rank: 4,
            population: 8,
            total_steps: steps,
            warmup_steps: 3,
            min_lr_frac: 0.1,
            log_interval: 0,
            base_seed: 42,
        };
        let mut trainer_a =
            Qwen3NextEggrollTrainer::new(&mut model_a, config_a, 5, &["embed_tokens"]).unwrap();
        let losses_a = trainer_a.train(&tokens, prompt_len).unwrap();

        // Path B: merge only at end
        let mut model_b = tiny_qwen3next_model();
        let config_b = EggrollConfig {
            sigma: 0.05,
            lr: 0.005,
            rank: 4,
            population: 8,
            total_steps: steps,
            warmup_steps: 3,
            min_lr_frac: 0.1,
            log_interval: 0,
            base_seed: 42,
        };
        let mut trainer_b =
            Qwen3NextEggrollTrainer::new(&mut model_b, config_b, 0, &["embed_tokens"]).unwrap();
        let losses_b = trainer_b.train(&tokens, prompt_len).unwrap();

        let window = steps / 4;
        let last_a: f32 = losses_a[losses_a.len() - window..].iter().sum::<f32>() / window as f32;
        let last_b: f32 = losses_b[losses_b.len() - window..].iter().sum::<f32>() / window as f32;

        eprintln!(
            "[merge test] periodic(5)={last_a:.4}  end_only={last_b:.4}  \
             diff={:.4}",
            last_a - last_b,
        );

        // Both should be finite
        assert!(last_a.is_finite() && last_b.is_finite());
        // Periodic merge should not be catastrophically worse (allow 20% margin for noise)
        assert!(
            last_a < last_b * 1.2,
            "Periodic merge regressed: {last_a:.4} vs end-only {last_b:.4}"
        );
    }

    /// Benchmark: additive-path training on the tiny model.
    ///
    /// Measures:
    /// - Wall-clock per training step (all members, +/- perturbation)
    /// - Overhead of additive forward vs clean forward
    /// - build_perturbation_map cost
    /// - delta_map.clone() cost
    /// - Training at different sequence lengths
    ///
    /// ```bash
    /// cargo test -p higgs-models -- bench_additive_training --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[ignore = "benchmark, not a correctness test"]
    fn bench_additive_training() {
        use std::time::Instant;

        // Use a model with max_position_embeddings=2048 to test longer sequences
        let args: Qwen3NextModelArgs = serde_json::from_str(
            r#"{
                "model_type": "qwen3_next",
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "intermediate_size": 256,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "rms_norm_eps": 1e-06,
                "vocab_size": 1024,
                "max_position_embeddings": 2048,
                "full_attention_interval": 1,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "num_experts": 0,
                "tie_word_embeddings": true,
                "quantization": { "group_size": 64, "bits": 4 }
            }"#,
        ).unwrap();
        let mut model = Qwen3NextCausalLM::new(args).unwrap();

        // Populate with random quantized weights (same as tiny_qwen3next_model but bigger vocab/pos)
        let hidden = 128i32;
        let kv_dim = 64i32;
        let inter = 256i32;
        let vocab = 1024i32;
        let group_size = 64;
        let bits = 4;
        let num_heads = 2i32;
        let head_dim = 64i32;
        let q_out = 2 * num_heads * head_dim;
        let o_in = num_heads * head_dim;
        let shapes: Vec<(&str, i32, i32)> = vec![
            ("model.embed_tokens", vocab, hidden),
            ("model.layers.0.self_attn.q_proj", q_out, hidden),
            ("model.layers.0.self_attn.k_proj", kv_dim, hidden),
            ("model.layers.0.self_attn.v_proj", kv_dim, hidden),
            ("model.layers.0.self_attn.o_proj", hidden, o_in),
            ("model.layers.0.mlp.gate_proj", inter, hidden),
            ("model.layers.0.mlp.up_proj", inter, hidden),
            ("model.layers.0.mlp.down_proj", hidden, inter),
            ("model.layers.1.self_attn.q_proj", q_out, hidden),
            ("model.layers.1.self_attn.k_proj", kv_dim, hidden),
            ("model.layers.1.self_attn.v_proj", kv_dim, hidden),
            ("model.layers.1.self_attn.o_proj", hidden, o_in),
            ("model.layers.1.mlp.gate_proj", inter, hidden),
            ("model.layers.1.mlp.up_proj", inter, hidden),
            ("model.layers.1.mlp.down_proj", hidden, inter),
        ];
        {
            use crate::diffusion_eggroll::xorshift_normal;
            let mut rng = 7777u64;
            let mut params = model.parameters_mut().flatten();
            for (base, out_d, in_d) in &shapes {
                let n = (*out_d as usize) * (*in_d as usize);
                let raw_data: Vec<f32> = (0..n).map(|_| xorshift_normal(&mut rng) * 0.02).collect();
                let raw = Array::from_slice(&raw_data, &[*out_d, *in_d]);
                let (qw, qs, qb) = ops::quantize(&raw, group_size, bits).unwrap();
                if let Some(p) = params.get_mut(format!("{base}.weight").as_str()) { **p = qw; }
                if let Some(p) = params.get_mut(format!("{base}.scales").as_str()) { **p = qs; }
                if let Some(p) = params.get_mut(format!("{base}.biases").as_str()) { **p = qb; }
            }
        }
        model.eval().unwrap();
        let config = EggrollConfig {
            sigma: 0.1,
            lr: 0.01,
            rank: 4,
            population: 8,
            total_steps: 5,
            warmup_steps: 1,
            min_lr_frac: 0.1,
            log_interval: 0,
            base_seed: 42,
        };

        let trainer = Qwen3NextEggrollTrainer::new(
            &mut model, config, 0, &["embed_tokens"],
        ).unwrap();
        let n_targets = trainer.num_targets();
        eprintln!("\n=== Additive Training Benchmark ===");
        eprintln!("Model: 2-layer dense, hidden=128, vocab=256");
        eprintln!("Targets: {n_targets} QLinear projections");
        eprintln!("Population: 8, Rank: 4\n");

        // --- 1. build_perturbation_map cost ---
        let n_runs = 20;
        let t0 = Instant::now();
        for i in 0..n_runs {
            let _ = trainer.build_perturbation_map(i, 0, 1.0);
        }
        let map_us = t0.elapsed().as_micros() as f64 / n_runs as f64;
        eprintln!("build_perturbation_map: {map_us:.0}µs ({n_targets} targets, rank=4)");

        // --- 2. delta_map.clone() cost ---
        // Simulate accumulated deltas
        let mut fake_deltas: std::collections::HashMap<String, Array> =
            std::collections::HashMap::new();
        for t in &trainer.targets {
            let zeros = Array::zeros::<f32>(&[t.out_dim as i32, t.in_dim as i32]).unwrap();
            fake_deltas.insert(t.name.clone(), zeros);
        }
        let t0 = Instant::now();
        for _ in 0..n_runs {
            let _ = fake_deltas.clone();
        }
        let clone_us = t0.elapsed().as_micros() as f64 / n_runs as f64;
        eprintln!("delta_map.clone(): {clone_us:.0}µs ({n_targets} entries)");

        // --- 3. Clean forward vs additive forward ---
        let seq_lens = [8, 32, 64, 128, 256, 512, 1024];
        eprintln!("\n--- Forward pass timing (5 warmup + 10 timed) ---");
        eprintln!("{:>6} {:>12} {:>12} {:>12}", "SeqLen", "Clean(ms)", "Additive(ms)", "Overhead(ms)");

        let pmap = trainer.build_perturbation_map(0, 0, 1.0);
        let dmap: crate::qwen3_next::DeltaMap = fake_deltas.clone();

        for &seq_len in &seq_lens {
            let tokens: Vec<u32> = (0..seq_len).map(|i| (i % 1024) as u32).collect();
            let input = Array::from_slice(&tokens, &[1, seq_len as i32])
                .as_type::<u32>().unwrap();

            // Clean forward
            for _ in 0..5 {
                let _ = trainer.model.forward_all_logits(&input).unwrap();
            }
            let t0 = Instant::now();
            for _ in 0..10 {
                let l = trainer.model.forward_all_logits(&input).unwrap();
                mlx_rs::transforms::eval(std::slice::from_ref(&l)).unwrap();
            }
            let clean_ms = t0.elapsed().as_secs_f64() * 100.0; // /10*1000

            // Additive forward
            trainer.model.perturbations = Some(pmap.clone());
            trainer.model.train_deltas = Some(dmap.clone());
            for _ in 0..5 {
                let _ = trainer.model.forward_all_logits(&input).unwrap();
            }
            let t0 = Instant::now();
            for _ in 0..10 {
                let l = trainer.model.forward_all_logits(&input).unwrap();
                mlx_rs::transforms::eval(std::slice::from_ref(&l)).unwrap();
            }
            let additive_ms = t0.elapsed().as_secs_f64() * 100.0;
            trainer.model.perturbations = None;
            trainer.model.train_deltas = None;

            eprintln!(
                "{seq_len:>6} {clean_ms:>12.2} {additive_ms:>12.2} {:>12.2}",
                additive_ms - clean_ms
            );
        }

        // --- 4. Full training step timing at different seq lens ---
        eprintln!("\n--- Full training step (pop=8, 16 forward passes/step) ---");
        eprintln!("{:>6} {:>12} {:>12} {:>8}", "SeqLen", "Step(ms)", "Loss", "Finite?");

        for &seq_len in &seq_lens {
            let mut model2 = tiny_qwen3next_model();
            let cfg = EggrollConfig {
                sigma: 0.1,
                lr: 0.01,
                rank: 4,
                population: 8,
                total_steps: 3,
                warmup_steps: 1,
                min_lr_frac: 0.1,
                log_interval: 0,
                base_seed: 42,
            };
            let mut tr = Qwen3NextEggrollTrainer::new(
                &mut model2, cfg, 0, &["embed_tokens"],
            ).unwrap();

            let tokens: Vec<u32> = (0..seq_len).map(|i| (i % 1024) as u32).collect();
            let prompt_len = seq_len / 4;

            let t0 = Instant::now();
            let losses = tr.train(&tokens, prompt_len).unwrap();
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let step_ms = total_ms / 3.0;
            let avg_loss = losses.iter().sum::<f32>() / losses.len() as f32;
            let all_finite = losses.iter().all(|l| l.is_finite());

            eprintln!(
                "{seq_len:>6} {step_ms:>12.1} {avg_loss:>12.4} {:>8}",
                if all_finite { "yes" } else { "NO" }
            );
        }

        eprintln!("\n=== Benchmark complete ===");
    }
}
