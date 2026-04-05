//! PCAST — Path-Constrained Additive Self-Training via backprop.
//!
//! Replaces the ES-based EGGROLL trainer with exact gradient computation:
//!
//!   y = stop_gradient(quantized_matmul(W, x)) + x @ delta.T
//!
//! where `delta = lora_a @ lora_b.T` is a low-rank trainable perturbation.
//! Gradients flow only through the delta path (base quantized weights are frozen
//! via `stop_gradient`). GDN layers are also `stop_gradient`'d since their
//! sequential scan produces impractical O(T) backward graphs.
//!
//! Compared to ES (eggroll_quantized.rs):
//! - 1 fwd + 1 bwd per step (vs 2N forwards for population N)
//! - Exact gradients (vs ES estimates)
//! - 320 targets including MoE experts (vs 200 excl. switch_mlp)
//! - ~3s/step (vs ~15s/step)

#![allow(
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::str_to_string,
    clippy::unwrap_used,
    clippy::use_debug,
    clippy::print_stderr,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    clippy::rc_clone_in_vec_init,
    clippy::clone_on_ref_ptr,
)]

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::{
    Array,
    error::Exception,
    module::ModuleParameters as _,
    ops,
    transforms::{KeyedGrad, keyed_value_and_grad},
};

use crate::diffusion_eggroll::EggrollConfig;
use crate::diffusion_train::cosine_lr;
use crate::qwen3_next::{DeltaMap, Qwen3NextCausalLM};

// -----------------------------------------------------------------------
// Gradient trainer config
// -----------------------------------------------------------------------

/// Configuration for the gradient-based PCAST trainer.
///
/// Reuses `EggrollConfig` for most fields (lr, rank, total_steps, etc.)
/// but `sigma` and `population` are ignored (no ES perturbations).
pub struct GradientTrainerConfig {
    pub rank: usize,
    pub lr: f32,
    pub total_steps: usize,
    pub warmup_steps: usize,
    pub min_lr_frac: f32,
    pub clip_ratio: f32,
    pub delta_decay: f32,
    pub log_interval: usize,
}

impl From<&EggrollConfig> for GradientTrainerConfig {
    fn from(ec: &EggrollConfig) -> Self {
        Self {
            rank: ec.rank,
            lr: ec.lr,
            total_steps: ec.total_steps,
            warmup_steps: ec.warmup_steps,
            min_lr_frac: ec.min_lr_frac,
            clip_ratio: ec.clip_ratio,
            delta_decay: ec.delta_decay,
            log_interval: ec.log_interval,
        }
    }
}

// -----------------------------------------------------------------------
// Delta target metadata
// -----------------------------------------------------------------------

/// Metadata for a single trainable target (QLinear or SwitchMlp projection).
struct DeltaTarget {
    /// Full key prefix, e.g. "model.layers.0.self_attn.q_proj"
    name: String,
    out_dim: usize,
    in_dim: usize,
}

// -----------------------------------------------------------------------
// GradientTrainer
// -----------------------------------------------------------------------

/// Gradient-based self-training for quantized models.
///
/// Uses `keyed_value_and_grad` to differentiate through low-rank delta paths
/// while the quantized base weights are frozen via `stop_gradient`.
pub struct GradientTrainer {
    /// Low-rank delta arrays: "target.lora_a" -> [out, r], "target.lora_b" -> [in, r]
    deltas: HashMap<Rc<str>, Array>,
    /// Target metadata for reconstructing DeltaMap
    targets: Vec<DeltaTarget>,
    /// Adam first moment (m)
    adam_m: HashMap<Rc<str>, Array>,
    /// Adam second moment (v)
    adam_v: HashMap<Rc<str>, Array>,
    /// Current training step
    step: usize,
    config: GradientTrainerConfig,
}

impl GradientTrainer {
    /// Create a gradient trainer, discovering all QLinear + SwitchMlp targets.
    pub fn new(
        model: &Qwen3NextCausalLM,
        config: GradientTrainerConfig,
    ) -> Result<Self, Exception> {
        let rank = config.rank;
        let exclude = &["embed_tokens", "lm_head", "mlp.gate", "linear_attn"];

        let params = model.parameters().flatten();
        let keys: Vec<String> = params.keys().map(|k| k.to_string()).collect();

        let quant = model
            .args
            .quantization
            .as_ref()
            .ok_or_else(|| Exception::custom("Gradient trainer requires a quantized model"))?;
        let group_size = quant.group_size as usize;

        let mut targets = Vec::new();

        for key in &keys {
            let Some(base) = key.strip_suffix(".scales") else {
                continue;
            };
            let weight_key = format!("{base}.weight");
            if !keys.contains(&weight_key) {
                continue;
            }
            if exclude.iter().any(|pat| base.contains(pat)) {
                continue;
            }

            let s = params.get(key.as_str()).unwrap();
            let s_shape = s.shape();

            // Distinguish QLinear (2D scales [out, n_groups]) from
            // SwitchMlp (3D scales [num_experts, out, n_groups])
            let (out_dim, in_dim) = if s_shape.len() == 3 {
                // SwitchMlp: [num_experts, out_dim, n_groups]
                let out = s_shape[1] as usize;
                let n_groups = s_shape[2] as usize;
                (out, n_groups * group_size)
            } else if s_shape.len() >= 2 {
                // QLinear: [out_dim, n_groups]
                let out = s_shape[0] as usize;
                let n_groups = s_shape[1] as usize;
                (out, n_groups * group_size)
            } else {
                continue;
            };

            targets.push(DeltaTarget {
                name: base.to_string(),
                out_dim,
                in_dim,
            });
        }

        // Initialize low-rank deltas with random values
        let mut deltas = HashMap::new();
        let adam_m = HashMap::new();
        let adam_v = HashMap::new();

        let scale = 1e-3 / (rank as f32).sqrt();
        for t in &targets {
            let a_key: Rc<str> = format!("{}.lora_a", t.name).into();
            let b_key: Rc<str> = format!("{}.lora_b", t.name).into();
            let a = ops::multiply(
                &mlx_rs::random::normal::<f32>(&[t.out_dim as i32, rank as i32], None, None, None)?,
                &Array::from_f32(scale),
            )?;
            let b = ops::multiply(
                &mlx_rs::random::normal::<f32>(&[t.in_dim as i32, rank as i32], None, None, None)?,
                &Array::from_f32(scale),
            )?;
            deltas.insert(a_key, a);
            deltas.insert(b_key, b);
        }

        // Eval all initial deltas
        let delta_refs: Vec<&Array> = deltas.values().collect();
        mlx_rs::transforms::eval(delta_refs)?;

        eprintln!(
            "[PCAST] {} targets, {} delta arrays, rank={rank}",
            targets.len(),
            deltas.len(),
        );
        for t in targets.iter().take(5) {
            eprintln!("[PCAST]   {} [{}x{}]", t.name, t.out_dim, t.in_dim);
        }
        if targets.len() > 5 {
            eprintln!("[PCAST]   ... and {} more", targets.len() - 5);
        }

        Ok(Self {
            deltas,
            targets,
            adam_m,
            adam_v,
            step: 0,
            config,
        })
    }

    /// Reconstruct full-rank `DeltaMap` from the keyed low-rank arrays.
    ///
    /// For each target: `delta = lora_a @ lora_b.T` → [out, in]
    fn build_delta_map(
        targets: &[DeltaTarget],
        params: &HashMap<Rc<str>, Array>,
    ) -> Result<DeltaMap, Exception> {
        let mut map = DeltaMap::with_capacity(targets.len());
        for t in targets {
            let a_key: Rc<str> = format!("{}.lora_a", t.name).into();
            let b_key: Rc<str> = format!("{}.lora_b", t.name).into();
            let a = params
                .get(&a_key)
                .ok_or_else(|| Exception::custom(format!("missing delta {a_key}")))?;
            let b = params
                .get(&b_key)
                .ok_or_else(|| Exception::custom(format!("missing delta {b_key}")))?;
            // delta = lora_a @ lora_b.T  →  [out, r] @ [r, in]  →  [out, in]
            let delta = a.matmul(&b.transpose()?)?;
            map.insert(t.name.clone(), delta);
        }
        Ok(map)
    }

    /// Adam update on a single parameter.
    fn adam_update(
        param: &Array,
        grad: &Array,
        m: &mut Option<Array>,
        v: &mut Option<Array>,
        lr: f32,
        step: usize,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Result<Array, Exception> {
        let t = (step + 1) as f32;

        // Initialize moments on first step
        let m_arr = match m.take() {
            Some(prev) => {
                let new_m = ops::add(
                    &ops::multiply(&Array::from_f32(beta1), &prev)?,
                    &ops::multiply(&Array::from_f32(1.0 - beta1), grad)?,
                )?;
                new_m
            }
            None => ops::multiply(&Array::from_f32(1.0 - beta1), grad)?,
        };
        let v_arr = match v.take() {
            Some(prev) => {
                let new_v = ops::add(
                    &ops::multiply(&Array::from_f32(beta2), &prev)?,
                    &ops::multiply(&Array::from_f32(1.0 - beta2), &ops::multiply(grad, grad)?)?,
                )?;
                new_v
            }
            None => ops::multiply(&Array::from_f32(1.0 - beta2), &ops::multiply(grad, grad)?)?,
        };

        // Bias correction
        let m_hat = ops::divide(&m_arr, &Array::from_f32(1.0 - beta1.powf(t)))?;
        let v_hat = ops::divide(&v_arr, &Array::from_f32(1.0 - beta2.powf(t)))?;

        // Update: param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
        let v_sqrt = ops::sqrt(&v_hat)?;
        let denom = ops::add(&v_sqrt, &Array::from_f32(eps))?;
        let mut update = ops::divide(&m_hat, &denom)?;
        if weight_decay > 0.0 {
            update = ops::add(&update, &ops::multiply(&Array::from_f32(weight_decay), param)?)?;
        }
        let result = ops::subtract(param, &ops::multiply(&Array::from_f32(lr), &update)?)?;

        *m = Some(m_arr);
        *v = Some(v_arr);

        Ok(result)
    }

    /// Run a single gradient training step.
    ///
    /// Returns the loss value for this step.
    pub fn train_step(
        &mut self,
        model: &mut Qwen3NextCausalLM,
        tokens: &[u32],
        prompt_len: usize,
    ) -> Result<f32, Exception> {
        let step = self.step;
        let lr = cosine_lr(
            step,
            self.config.warmup_steps,
            self.config.total_steps,
            self.config.lr,
            self.config.min_lr_frac,
        );

        let seq_len = tokens.len();
        let token_vec: Vec<u32> = tokens.to_vec();
        let input = Array::from_slice(&token_vec, &[1, seq_len as i32]).as_type::<u32>()?;

        // Clone targets for the closure
        let targets: Vec<(String, usize, usize)> = self
            .targets
            .iter()
            .map(|t| (t.name.clone(), t.out_dim, t.in_dim))
            .collect();

        // Loss closure: takes keyed delta params, returns [loss]
        // Captures &mut model to run forward pass.
        let loss_fn = |params: HashMap<Rc<str>, Array>, _args: i32| -> Result<Vec<Array>, Exception> {
            // Reconstruct DeltaMap from traced low-rank arrays
            let mut delta_map = DeltaMap::with_capacity(targets.len());
            for (name, _, _) in &targets {
                let a_key: Rc<str> = format!("{name}.lora_a").into();
                let b_key: Rc<str> = format!("{name}.lora_b").into();
                let a = params
                    .get(&a_key)
                    .ok_or_else(|| Exception::custom(format!("missing {a_key}")))?;
                let b = params
                    .get(&b_key)
                    .ok_or_else(|| Exception::custom(format!("missing {b_key}")))?;
                let delta = a.matmul(&b.transpose()?)?;
                delta_map.insert(name.clone(), delta);
            }

            // Set deltas on model and forward
            model.train_deltas = Some(delta_map);
            let logits = model.forward_all_logits(&input)?;
            model.train_deltas = None;

            // Differentiable causal LM loss on completion tokens
            let start = prompt_len.saturating_sub(1) as i32;
            let end = (seq_len - 1) as i32;
            if start >= end {
                return Ok(vec![Array::from_f32(0.0)]);
            }

            let comp_logits = logits.index((.., start..end, ..));
            let log_probs = mlx_rs::nn::log_softmax(&comp_logits, -1)?;

            let target_ids: Vec<u32> = token_vec[start as usize + 1..=end as usize].to_vec();
            let n_comp = target_ids.len() as i32;
            let target_arr = Array::from_slice(&target_ids, &[1, n_comp, 1]).as_type::<u32>()?;
            let target_lp = log_probs.take_along_axis(&target_arr, -1)?;
            let loss = target_lp.mean(None)?.negative()?;

            Ok(vec![loss])
        };

        // Compute value and gradients
        let mut vg = keyed_value_and_grad(loss_fn);
        let (values, grads): (Vec<Array>, KeyedGrad) = vg(self.deltas.clone(), 0)?;

        let loss_val = {
            mlx_rs::transforms::eval(std::slice::from_ref(&values[0]))?;
            values[0].item::<f32>()
        };

        // Adam update on each delta parameter
        let mut new_deltas = HashMap::with_capacity(self.deltas.len());
        for (key, param) in &self.deltas {
            let grad: &Array = grads
                .get(key)
                .ok_or_else(|| Exception::custom(format!("no gradient for {key}")))?;

            let m_entry = self.adam_m.remove(key);
            let v_entry = self.adam_v.remove(key);
            let mut m_opt = m_entry;
            let mut v_opt = v_entry;

            let updated = Self::adam_update(
                param,
                grad,
                &mut m_opt,
                &mut v_opt,
                lr,
                step,
                0.9,   // beta1
                0.999, // beta2
                1e-8,  // eps
                self.config.delta_decay,
            )?;

            if let Some(m) = m_opt {
                self.adam_m.insert(key.clone(), m);
            }
            if let Some(v) = v_opt {
                self.adam_v.insert(key.clone(), v);
            }
            new_deltas.insert(key.clone(), updated);
        }

        // Eval updated deltas
        let refs: Vec<&Array> = new_deltas.values().collect();
        mlx_rs::transforms::eval(refs)?;

        self.deltas = new_deltas;
        self.step += 1;

        Ok(loss_val)
    }

    /// Run full training loop over the given tokens.
    ///
    /// Returns per-step loss values.
    pub fn train(
        &mut self,
        model: &mut Qwen3NextCausalLM,
        tokens: &[u32],
        prompt_len: usize,
    ) -> Result<Vec<f32>, Exception> {
        let total_steps = self.config.total_steps;
        let log_interval = self.config.log_interval;
        let mut losses = Vec::with_capacity(total_steps);

        eprintln!(
            "[PCAST] training: targets={} seq_len={} prompt={} steps={total_steps}",
            self.targets.len(),
            tokens.len(),
            prompt_len,
        );

        for step in 0..total_steps {
            let t0 = std::time::Instant::now();
            let loss = self.train_step(model, tokens, prompt_len)?;
            let elapsed = t0.elapsed();
            losses.push(loss);

            if log_interval > 0 && ((step + 1) % log_interval == 0 || step == 0) {
                eprintln!(
                    "[PCAST] step {}/{total_steps} loss={loss:.4} ({:.0}ms)",
                    step + 1,
                    elapsed.as_millis(),
                );
            }
        }

        // Persist final deltas on model for inference
        let delta_map = Self::build_delta_map(&self.targets, &self.deltas)?;
        model.train_deltas = Some(delta_map);

        eprintln!(
            "[PCAST] done: {} steps, final_loss={:.4}",
            losses.len(),
            losses.last().copied().unwrap_or(0.0),
        );

        Ok(losses)
    }

    /// Number of trainable targets.
    pub fn num_targets(&self) -> usize {
        self.targets.len()
    }

    /// Total number of trainable parameters (lora_a + lora_b across all targets).
    pub fn num_params(&self) -> usize {
        self.targets
            .iter()
            .map(|t| (t.out_dim + t.in_dim) * self.config.rank)
            .sum()
    }
}

// -----------------------------------------------------------------------
// Indexing helper for differentiable loss
// -----------------------------------------------------------------------

use mlx_rs::ops::indexing::IndexOp;
