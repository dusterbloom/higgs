//! Weight extraction from a loaded RWKV-7 model for ANE compilation.
//!
//! Converts MLX `Array` parameters into flat `Vec<f32>` / `Vec<u16>` (fp16)
//! suitable for ANE kernel compilation and CPU-side compute.
//!
//! Feature-gated behind `ane`.

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::module::ModuleParameters;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::ane_forward::{LayerAneWeightData, LayerCpuWeights, LoraActivation, LoraWeights};
use crate::rwkv7::Rwkv7CausalLM;

type ParamMap<'a> = HashMap<Rc<str>, &'a Array>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a single parameter as f32 vec, converting dtype if needed. Panics if key is missing.
fn get_f32(params: &ParamMap<'_>, key: &str) -> Vec<f32> {
    let arr = params
        .get(key)
        .unwrap_or_else(|| panic!("Missing weight: {key}"));
    // Convert to f32 if the array is in a different dtype (e.g., bf16).
    if arr.dtype() == mlx_rs::Dtype::Float32 {
        arr.as_slice::<f32>().to_vec()
    } else {
        let converted = arr
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap_or_else(|e| panic!("Failed to convert {key} to f32: {e}"));
        eval([&converted]).unwrap_or_else(|e| panic!("Failed to eval {key}: {e}"));
        converted.as_slice::<f32>().to_vec()
    }
}

/// Extract a single parameter as f32 vec, returning None if absent.
fn get_f32_opt(params: &ParamMap<'_>, key: &str) -> Option<Vec<f32>> {
    params.get(key).map(|arr| {
        if arr.dtype() == mlx_rs::Dtype::Float32 {
            arr.as_slice::<f32>().to_vec()
        } else {
            let converted = arr
                .as_dtype(mlx_rs::Dtype::Float32)
                .expect("dtype conversion failed");
            eval([&converted]).expect("eval failed");
            converted.as_slice::<f32>().to_vec()
        }
    })
}

/// Flatten a [1, 1, dim] parameter to [dim].
fn get_f32_flat(params: &ParamMap<'_>, key: &str) -> Vec<f32> {
    get_f32(params, key)
}

/// Extract a weight matrix and transpose from [rows, cols] to [cols, rows].
/// PyTorch linear weight is [out_features, in_features]; ANE MIL matmul needs [in_features, out_features].
fn get_f32_transposed(params: &ParamMap<'_>, key: &str, rows: usize, cols: usize) -> Vec<f32> {
    let data = get_f32(params, key);
    assert_eq!(data.len(), rows * cols, "{key}: expected {rows}x{cols}={}, got {}", rows * cols, data.len());
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Convert a single f32 to IEEE 754 fp16 (u16 bits).
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf/NaN
        return (sign | 0x7C00 | if frac != 0 { 0x0200 } else { 0 }) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow → Inf
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        // Underflow → zero (skip denormals for speed)
        return sign as u16;
    }

    (sign | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Convert f32 slice to fp16 (u16 bits) for gemm_f16.
fn f32_to_fp16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_f16_bits(v)).collect()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Extract all weights for ANE compilation + CPU compute.
///
/// Evaluates all model parameters and returns them as flat f32 vectors.
/// The model must already be loaded with `load_rwkv7_model()`.
pub fn extract_ane_weights(
    model: &Rwkv7CausalLM,
) -> (Vec<LayerAneWeightData>, Vec<LayerCpuWeights>) {
    let params_nested = model.parameters();
    let params = params_nested.flatten();

    // Eval all parameters so as_slice works.
    let all_arrays: Vec<&Array> = params.values().copied().collect();
    eval(all_arrays).expect("Failed to eval model parameters");

    let n_layers = model.args.num_hidden_layers as usize;
    let mut ane_weights = Vec::with_capacity(n_layers);
    let mut cpu_weights = Vec::with_capacity(n_layers);

    for i in 0..n_layers {
        let pfx = format!("model.layers.{i}");

        // --- ANE weights (large projection matrices, transposed for MIL matmul) ---
        // PyTorch linear stores [out, in]; MIL matmul needs [in, out].
        let dim = model.args.hidden_size as usize;
        let inter = model.args.intermediate_size as usize;
        ane_weights.push(LayerAneWeightData {
            r_proj: get_f32_transposed(&params, &format!("{pfx}.attn.r_proj.weight"), dim, dim),
            k_proj: get_f32_transposed(&params, &format!("{pfx}.attn.k_proj.weight"), dim, dim),
            v_proj: get_f32_transposed(&params, &format!("{pfx}.attn.v_proj.weight"), dim, dim),
            o_proj: get_f32_transposed(&params, &format!("{pfx}.attn.o_proj.weight"), dim, dim),
            ffn_key: get_f32_transposed(&params, &format!("{pfx}.ffn.key.weight"), inter, dim),
            ffn_value: get_f32_transposed(&params, &format!("{pfx}.ffn.value.weight"), dim, inter),
        });

        // --- CPU weights (mixing, LoRA, norms, gating) ---

        // LoRA projections
        let w_lora = LoraWeights {
            down: get_f32(&params, &format!("{pfx}.attn.w_lora.lora.down")),
            up_weight: get_f32(&params, &format!("{pfx}.attn.w_lora.lora.up_weight")),
            up_bias: get_f32(&params, &format!("{pfx}.attn.w_lora.lora.up_bias")),
            rank: model.args.decay_low_rank_dim as usize,
            input_dim: dim,
            output_dim: dim,
            activation: LoraActivation::Tanh,
        };

        let a_lora = LoraWeights {
            down: get_f32(&params, &format!("{pfx}.attn.a_lora.lora.down")),
            up_weight: get_f32(&params, &format!("{pfx}.attn.a_lora.lora.up_weight")),
            up_bias: get_f32(&params, &format!("{pfx}.attn.a_lora.lora.up_bias")),
            rank: model.args.a_low_rank_dim as usize,
            input_dim: dim,
            output_dim: dim,
            activation: LoraActivation::Identity,
        };

        let g_lora = LoraWeights {
            down: get_f32(&params, &format!("{pfx}.attn.g_lora.lora.down")),
            up_weight: get_f32(&params, &format!("{pfx}.attn.g_lora.lora.up_weight")),
            up_bias: get_f32_opt(&params, &format!("{pfx}.attn.g_lora.lora.up_bias"))
                .unwrap_or_else(|| vec![0.0; dim]),
            rank: model.args.gate_low_rank_dim as usize,
            input_dim: dim,
            output_dim: dim,
            activation: LoraActivation::Sigmoid,
        };

        let v_lora = if i > 0 {
            Some(LoraWeights {
                down: get_f32(&params, &format!("{pfx}.attn.v_lora.lora.down")),
                up_weight: get_f32(&params, &format!("{pfx}.attn.v_lora.lora.up_weight")),
                up_bias: get_f32(&params, &format!("{pfx}.attn.v_lora.lora.up_bias")),
                rank: model.args.v_low_rank_dim as usize,
                input_dim: dim,
                output_dim: dim,
                activation: LoraActivation::Identity,
            })
        } else {
            None
        };

        // Pre-norm (layer 0 only when norm_first)
        let (pre_norm_weight, pre_norm_bias) = if i == 0 && model.args.norm_first {
            (
                Some(get_f32(&params, &format!("{pfx}.pre_norm.weight"))),
                Some(get_f32(&params, &format!("{pfx}.pre_norm.bias"))),
            )
        } else {
            (None, None)
        };

        cpu_weights.push(LayerCpuWeights {
            x_r: get_f32_flat(&params, &format!("{pfx}.attn.x_r")),
            x_w: get_f32_flat(&params, &format!("{pfx}.attn.x_w")),
            x_k: get_f32_flat(&params, &format!("{pfx}.attn.x_k")),
            x_v: get_f32_flat(&params, &format!("{pfx}.attn.x_v")),
            x_a: get_f32_flat(&params, &format!("{pfx}.attn.x_a")),
            x_g: get_f32_flat(&params, &format!("{pfx}.attn.x_g")),

            k_k: get_f32(&params, &format!("{pfx}.attn.k_k")),
            k_a: get_f32(&params, &format!("{pfx}.attn.k_a")),
            r_k: get_f32(&params, &format!("{pfx}.attn.r_k")),

            w_lora,
            a_lora,
            v_lora,
            g_lora,

            g_norm_weight: get_f32(&params, &format!("{pfx}.attn.g_norm_weight")),
            g_norm_bias: get_f32(&params, &format!("{pfx}.attn.g_norm_bias")),

            attn_norm_weight: get_f32(&params, &format!("{pfx}.attn_norm.weight")),
            attn_norm_bias: get_f32(&params, &format!("{pfx}.attn_norm.bias")),
            ffn_norm_weight: get_f32(&params, &format!("{pfx}.ffn_norm.weight")),
            ffn_norm_bias: get_f32(&params, &format!("{pfx}.ffn_norm.bias")),

            ffn_x_k: get_f32_flat(&params, &format!("{pfx}.ffn.x_k")),

            pre_norm_weight,
            pre_norm_bias,
        });
    }

    (ane_weights, cpu_weights)
}

/// Extract projection weights as fp16 in ORIGINAL PyTorch layout `[oc, ic]` for `gemm_f16`.
/// This halves memory bandwidth vs f32 for BLAS projections.
pub fn extract_fp16_projection_weights(model: &Rwkv7CausalLM) -> Vec<crate::ane_forward::LayerFp16Weights> {
    let params = model.parameters().flatten();
    let all_arrays: Vec<&Array> = params.values().copied().collect();
    eval(all_arrays).expect("Failed to eval");

    let n_layers = model.args.num_hidden_layers as usize;
    let dim = model.args.hidden_size as usize;
    let inter = model.args.intermediate_size as usize;

    (0..n_layers)
        .map(|i| {
            let pfx = format!("model.layers.{i}");
            // Original layout [oc, ic] — NOT transposed. gemm_f16 handles the matmul.
            crate::ane_forward::LayerFp16Weights {
                r_proj: f32_to_fp16(&get_f32(&params, &format!("{pfx}.attn.r_proj.weight"))),
                k_proj: f32_to_fp16(&get_f32(&params, &format!("{pfx}.attn.k_proj.weight"))),
                v_proj: f32_to_fp16(&get_f32(&params, &format!("{pfx}.attn.v_proj.weight"))),
                o_proj: f32_to_fp16(&get_f32(&params, &format!("{pfx}.attn.o_proj.weight"))),
                // ffn_key: [inter, dim], ffn_value: [dim, inter]
                ffn_key: f32_to_fp16(&get_f32(&params, &format!("{pfx}.ffn.key.weight"))),
                ffn_value: f32_to_fp16(&get_f32(&params, &format!("{pfx}.ffn.value.weight"))),
            }
        })
        .collect()
}

/// Extract embedding table as flat f32 vector `[vocab_size * dim]`.
pub fn extract_embedding(model: &Rwkv7CausalLM) -> Vec<f32> {
    let params = model.parameters().flatten();
    let all_arrays: Vec<&Array> = params.values().copied().collect();
    eval(all_arrays).expect("Failed to eval");
    get_f32(&params, "model.embeddings.weight")
}

/// Extract LM head as fp16 (u16 bits) for `gemm_f16`. Returns `(data, rows, cols)`.
pub fn extract_lm_head_fp16(model: &Rwkv7CausalLM) -> (Vec<u16>, usize, usize) {
    let params = model.parameters().flatten();
    let all_arrays: Vec<&Array> = params.values().copied().collect();
    eval(all_arrays).expect("Failed to eval");
    let w = get_f32(&params, "lm_head.weight");
    let rows = model.args.vocab_size as usize;
    let cols = model.args.hidden_size as usize;
    (f32_to_fp16(&w), rows, cols)
}

/// Extract final layer norm weights and biases.
pub fn extract_final_norm(model: &Rwkv7CausalLM) -> (Vec<f32>, Vec<f32>) {
    let params = model.parameters().flatten();
    let all_arrays: Vec<&Array> = params.values().copied().collect();
    eval(all_arrays).expect("Failed to eval");
    (
        get_f32(&params, "model.norm.weight"),
        get_f32(&params, "model.norm.bias"),
    )
}
