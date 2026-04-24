//! Bisection: progressively re-enable scaffolding ops to find the dominant
//! cost in the ~30 ms/step Rust↔Python decode gap on Bonsai-8B.
//!
//! Baseline (from `qmm_only_decode.rs`, session-27): qmm-only = **2.14 ms/step**
//! on 8B. Production `forward_trunk_free` = **44.5 ms/step**. mlx-lm Python
//! reference = **14 ms/step**. The 42 ms of scaffolding lives somewhere in:
//! rope (× 72/step), rms_norm (× 144/step), KV cache update + SDPA (× 36/step),
//! residual adds (× 72/step).
//!
//! Variants on top of qmm-only:
//!   v2  : + rope (Array::from_int(offset) per call × 72)
//!   v2b : + rope (Array::from_int once per step, shared across 72 calls) ← hypothesis test
//!   v3  : v2 + rms_norm × 4/layer (input + q_norm + k_norm + post_attn)
//!   v4  : v3 + KV cache update + SDPA × 1/layer (real growing cache)
//!   v5  : v4 + 2 residual adds/layer (= production-equivalent forward)
//!   v6  : v5 + as_slice::<f32>() readback + CPU argmax (= production decode loop)
//!
//! Run:
//!   cargo test --release -p higgs-models --test bisect_decode \
//!       bisect_decode_bench -- --ignored --nocapture

#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
#![allow(clippy::unwrap_used, clippy::cast_sign_loss)]

use std::path::PathBuf;
use std::time::Instant;

use mlx_rs::{Array, Dtype, fast, ops, random, transforms::eval};

use higgs_models::bonsai_q1::{BonsaiQ1Engine, BonsaiQ1Gpu};
use higgs_models::cache::{KeyValueCache, SteppingKeyValueCache};

const WARMUP: usize = 4;
const STEPS: usize = 64;

fn dir(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/lm-studio/models/prism-ml")
        .join(name);
    p.join("config.json").exists().then_some(p)
}

fn fresh_fp16(shape: &[i32]) -> Array {
    random::normal::<f32>(shape, None, None, None)
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap()
}

fn time_loop(label: &str, mut do_step: impl FnMut(usize)) {
    for s in 0..WARMUP {
        do_step(s);
    }
    let t0 = Instant::now();
    for s in 0..STEPS {
        do_step(WARMUP + s);
    }
    let elapsed = t0.elapsed();
    let ms = elapsed.as_secs_f64() * 1e3 / STEPS as f64;
    println!(
        "  {label}: {:.2} ms/step ({:.1} tok/s)",
        ms,
        1000.0 / ms
    );
}

// ---------- Variants ----------

fn v1_qmm_only(gpu: &BonsaiQ1Gpu) {
    let hidden = gpu.config.hidden as i32;
    let inter = gpu.config.inter as i32;
    time_loop("v1 qmm-only                        ", |_s| {
        let xh = fresh_fp16(&[1, 1, hidden]);
        let xi = fresh_fp16(&[1, 1, inter]);
        let mut last = xh.clone();
        for layer in &gpu.layers {
            let _ = layer.q_proj.forward(&xh).unwrap();
            let _ = layer.k_proj.forward(&xh).unwrap();
            let _ = layer.v_proj.forward(&xh).unwrap();
            let o = layer.o_proj.forward(&xh).unwrap();
            let _ = layer.gate_proj.forward(&xh).unwrap();
            let _ = layer.up_proj.forward(&xh).unwrap();
            let d = layer.down_proj.forward(&xi).unwrap();
            last = ops::add(&o, &d).unwrap();
        }
        let logits = match &gpu.lm_head {
            Some(lm) => lm.forward(&last).unwrap(),
            None => gpu.embed.forward(&last).unwrap(),
        };
        eval([&logits]).unwrap();
    });
}

fn v2_plus_rope_per_call(gpu: &BonsaiQ1Gpu) {
    let hidden = gpu.config.hidden as i32;
    let inter = gpu.config.inter as i32;
    let heads = gpu.config.heads as i32;
    let kv_heads = gpu.config.kv_heads as i32;
    let head_dim = gpu.config.head_dim as i32;
    let base = gpu.config.rope_theta as f32;
    let yarn_freqs = gpu.yarn_freqs.as_ref();
    let use_freqs = yarn_freqs.is_some();

    time_loop("v2 +rope (alloc per call × 72)     ", |s| {
        let off = s as i32;
        let xh = fresh_fp16(&[1, 1, hidden]);
        let xi = fresh_fp16(&[1, 1, inter]);
        let mut last = xh.clone();
        for layer in &gpu.layers {
            let q = layer
                .q_proj
                .forward(&xh)
                .unwrap()
                .reshape(&[1, 1, heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let k = layer
                .k_proj
                .forward(&xh)
                .unwrap()
                .reshape(&[1, 1, kv_heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let _ = layer.v_proj.forward(&xh).unwrap();

            // Hot path under test: fresh Array::from_int per call (× 2 per layer).
            let off_q = Array::from_int(off);
            let _q = if use_freqs {
                fast::rope_dynamic(&q, head_dim, false, None::<f32>, 1.0, &off_q, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&q, head_dim, false, base, 1.0, &off_q, None::<&Array>).unwrap()
            };
            let off_k = Array::from_int(off);
            let _k = if use_freqs {
                fast::rope_dynamic(&k, head_dim, false, None::<f32>, 1.0, &off_k, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&k, head_dim, false, base, 1.0, &off_k, None::<&Array>).unwrap()
            };

            let o = layer.o_proj.forward(&xh).unwrap();
            let _ = layer.gate_proj.forward(&xh).unwrap();
            let _ = layer.up_proj.forward(&xh).unwrap();
            let d = layer.down_proj.forward(&xi).unwrap();
            last = ops::add(&o, &d).unwrap();
        }
        let logits = match &gpu.lm_head {
            Some(lm) => lm.forward(&last).unwrap(),
            None => gpu.embed.forward(&last).unwrap(),
        };
        eval([&logits]).unwrap();
    });
}

fn v2b_plus_rope_prealloc(gpu: &BonsaiQ1Gpu) {
    let hidden = gpu.config.hidden as i32;
    let inter = gpu.config.inter as i32;
    let heads = gpu.config.heads as i32;
    let kv_heads = gpu.config.kv_heads as i32;
    let head_dim = gpu.config.head_dim as i32;
    let base = gpu.config.rope_theta as f32;
    let yarn_freqs = gpu.yarn_freqs.as_ref();
    let use_freqs = yarn_freqs.is_some();

    time_loop("v2b +rope (alloc 1× per step)      ", |s| {
        let off = s as i32;
        // ONE alloc per step, shared across all 72 rope calls.
        let off_arr = Array::from_int(off);
        let xh = fresh_fp16(&[1, 1, hidden]);
        let xi = fresh_fp16(&[1, 1, inter]);
        let mut last = xh.clone();
        for layer in &gpu.layers {
            let q = layer
                .q_proj
                .forward(&xh)
                .unwrap()
                .reshape(&[1, 1, heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let k = layer
                .k_proj
                .forward(&xh)
                .unwrap()
                .reshape(&[1, 1, kv_heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let _ = layer.v_proj.forward(&xh).unwrap();

            let _q = if use_freqs {
                fast::rope_dynamic(&q, head_dim, false, None::<f32>, 1.0, &off_arr, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&q, head_dim, false, base, 1.0, &off_arr, None::<&Array>)
                    .unwrap()
            };
            let _k = if use_freqs {
                fast::rope_dynamic(&k, head_dim, false, None::<f32>, 1.0, &off_arr, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&k, head_dim, false, base, 1.0, &off_arr, None::<&Array>)
                    .unwrap()
            };

            let o = layer.o_proj.forward(&xh).unwrap();
            let _ = layer.gate_proj.forward(&xh).unwrap();
            let _ = layer.up_proj.forward(&xh).unwrap();
            let d = layer.down_proj.forward(&xi).unwrap();
            last = ops::add(&o, &d).unwrap();
        }
        let logits = match &gpu.lm_head {
            Some(lm) => lm.forward(&last).unwrap(),
            None => gpu.embed.forward(&last).unwrap(),
        };
        eval([&logits]).unwrap();
    });
}

fn v3_plus_rms_norms(gpu: &BonsaiQ1Gpu) {
    let hidden = gpu.config.hidden as i32;
    let inter = gpu.config.inter as i32;
    let heads = gpu.config.heads as i32;
    let kv_heads = gpu.config.kv_heads as i32;
    let head_dim = gpu.config.head_dim as i32;
    let base = gpu.config.rope_theta as f32;
    let rms_eps = gpu.config.rms_norm_eps;
    let yarn_freqs = gpu.yarn_freqs.as_ref();
    let use_freqs = yarn_freqs.is_some();

    time_loop("v3 +rms_norm × 4/layer             ", |s| {
        let off = s as i32;
        let off_arr = Array::from_int(off);
        let xh = fresh_fp16(&[1, 1, hidden]);
        let xi = fresh_fp16(&[1, 1, inter]);
        let mut last = xh.clone();
        for layer in &gpu.layers {
            let normed = fast::rms_norm(&xh, &layer.input_norm, rms_eps).unwrap();
            let q = layer
                .q_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let k = layer
                .k_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, kv_heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let _ = layer.v_proj.forward(&normed).unwrap();

            let q = fast::rms_norm(&q, &layer.q_norm, rms_eps).unwrap();
            let k = fast::rms_norm(&k, &layer.k_norm, rms_eps).unwrap();

            let _q = if use_freqs {
                fast::rope_dynamic(&q, head_dim, false, None::<f32>, 1.0, &off_arr, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&q, head_dim, false, base, 1.0, &off_arr, None::<&Array>)
                    .unwrap()
            };
            let _k = if use_freqs {
                fast::rope_dynamic(&k, head_dim, false, None::<f32>, 1.0, &off_arr, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&k, head_dim, false, base, 1.0, &off_arr, None::<&Array>)
                    .unwrap()
            };

            let normed_post = fast::rms_norm(&xh, &layer.post_attn_norm, rms_eps).unwrap();
            let o = layer.o_proj.forward(&xh).unwrap();
            let _ = layer.gate_proj.forward(&normed_post).unwrap();
            let _ = layer.up_proj.forward(&normed_post).unwrap();
            let d = layer.down_proj.forward(&xi).unwrap();
            last = ops::add(&o, &d).unwrap();
        }
        let logits = match &gpu.lm_head {
            Some(lm) => lm.forward(&last).unwrap(),
            None => gpu.embed.forward(&last).unwrap(),
        };
        eval([&logits]).unwrap();
    });
}

fn v4_plus_kv_sdpa(gpu: &BonsaiQ1Gpu) {
    let hidden = gpu.config.hidden as i32;
    let inter = gpu.config.inter as i32;
    let heads = gpu.config.heads as i32;
    let kv_heads = gpu.config.kv_heads as i32;
    let head_dim = gpu.config.head_dim as i32;
    let base = gpu.config.rope_theta as f32;
    let rms_eps = gpu.config.rms_norm_eps;
    let scale = gpu.attention_scale;
    let yarn_freqs = gpu.yarn_freqs.as_ref();
    let use_freqs = yarn_freqs.is_some();

    let mut cache: Vec<SteppingKeyValueCache> = (0..gpu.layers.len())
        .map(|_| SteppingKeyValueCache::new())
        .collect();

    time_loop("v4 +KV cache update + SDPA         ", |s| {
        let off = s as i32;
        let off_arr = Array::from_int(off);
        let xh = fresh_fp16(&[1, 1, hidden]);
        let xi = fresh_fp16(&[1, 1, inter]);
        let mut last = xh.clone();
        for (layer, c) in gpu.layers.iter().zip(cache.iter_mut()) {
            let normed = fast::rms_norm(&xh, &layer.input_norm, rms_eps).unwrap();
            let q = layer
                .q_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let k = layer
                .k_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, kv_heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let v = layer
                .v_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, kv_heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();

            let q = fast::rms_norm(&q, &layer.q_norm, rms_eps).unwrap();
            let k = fast::rms_norm(&k, &layer.k_norm, rms_eps).unwrap();

            let q = if use_freqs {
                fast::rope_dynamic(&q, head_dim, false, None::<f32>, 1.0, &off_arr, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&q, head_dim, false, base, 1.0, &off_arr, None::<&Array>)
                    .unwrap()
            };
            let k = if use_freqs {
                fast::rope_dynamic(&k, head_dim, false, None::<f32>, 1.0, &off_arr, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&k, head_dim, false, base, 1.0, &off_arr, None::<&Array>)
                    .unwrap()
            };

            // KV cache update + SDPA (Dense view → into_dense → SDPA).
            let view = c.update_and_view(k, v).unwrap();
            let (full_k, full_v) = view.into_dense().unwrap();
            let _attn = fast::scaled_dot_product_attention(
                q,
                full_k,
                full_v,
                scale,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
                None::<&Array>,
            )
            .unwrap();

            let normed_post = fast::rms_norm(&xh, &layer.post_attn_norm, rms_eps).unwrap();
            let o = layer.o_proj.forward(&xh).unwrap();
            let _ = layer.gate_proj.forward(&normed_post).unwrap();
            let _ = layer.up_proj.forward(&normed_post).unwrap();
            let d = layer.down_proj.forward(&xi).unwrap();
            last = ops::add(&o, &d).unwrap();
        }
        let logits = match &gpu.lm_head {
            Some(lm) => lm.forward(&last).unwrap(),
            None => gpu.embed.forward(&last).unwrap(),
        };
        eval([&logits]).unwrap();
    });
}

fn v5_full_equivalent(gpu: &BonsaiQ1Gpu) {
    let hidden = gpu.config.hidden as i32;
    let heads = gpu.config.heads as i32;
    let kv_heads = gpu.config.kv_heads as i32;
    let head_dim = gpu.config.head_dim as i32;
    let base = gpu.config.rope_theta as f32;
    let rms_eps = gpu.config.rms_norm_eps;
    let scale = gpu.attention_scale;
    let yarn_freqs = gpu.yarn_freqs.as_ref();
    let use_freqs = yarn_freqs.is_some();

    let mut cache: Vec<SteppingKeyValueCache> = (0..gpu.layers.len())
        .map(|_| SteppingKeyValueCache::new())
        .collect();

    time_loop("v5 +residual adds (= production)   ", |s| {
        let off = s as i32;
        let off_arr = Array::from_int(off);
        let mut h = fresh_fp16(&[1, 1, hidden]);
        for (layer, c) in gpu.layers.iter().zip(cache.iter_mut()) {
            let normed = fast::rms_norm(&h, &layer.input_norm, rms_eps).unwrap();
            let q = layer
                .q_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let k = layer
                .k_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, kv_heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let v = layer
                .v_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, kv_heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();

            let q = fast::rms_norm(&q, &layer.q_norm, rms_eps).unwrap();
            let k = fast::rms_norm(&k, &layer.k_norm, rms_eps).unwrap();

            let q = if use_freqs {
                fast::rope_dynamic(&q, head_dim, false, None::<f32>, 1.0, &off_arr, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&q, head_dim, false, base, 1.0, &off_arr, None::<&Array>)
                    .unwrap()
            };
            let k = if use_freqs {
                fast::rope_dynamic(&k, head_dim, false, None::<f32>, 1.0, &off_arr, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&k, head_dim, false, base, 1.0, &off_arr, None::<&Array>)
                    .unwrap()
            };

            let view = c.update_and_view(k, v).unwrap();
            let (full_k, full_v) = view.into_dense().unwrap();
            let attn = fast::scaled_dot_product_attention(
                q,
                full_k,
                full_v,
                scale,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
                None::<&Array>,
            )
            .unwrap();

            let attn_out = attn
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap()
                .reshape(&[1, 1, -1])
                .unwrap();
            let attn_out = layer.o_proj.forward(&attn_out).unwrap();
            let h_post_attn = h.add(&attn_out).unwrap();

            let normed_post = fast::rms_norm(&h_post_attn, &layer.post_attn_norm, rms_eps).unwrap();
            let gate = layer.gate_proj.forward(&normed_post).unwrap();
            let up = layer.up_proj.forward(&normed_post).unwrap();
            let mlp_hidden = mlx_rs::nn::silu(&gate).unwrap().multiply(&up).unwrap();
            let mlp_out = layer.down_proj.forward(&mlp_hidden).unwrap();

            h = h_post_attn.add(&mlp_out).unwrap();
        }
        let final_h = fast::rms_norm(&h, &gpu.final_norm, rms_eps).unwrap();
        let logits = match &gpu.lm_head {
            Some(lm) => lm.forward(&final_h).unwrap(),
            None => gpu.embed.forward(&final_h).unwrap(),
        };
        eval([&logits]).unwrap();
    });
}

fn v6_with_readback_argmax(gpu: &BonsaiQ1Gpu) {
    let hidden = gpu.config.hidden as i32;
    let heads = gpu.config.heads as i32;
    let kv_heads = gpu.config.kv_heads as i32;
    let head_dim = gpu.config.head_dim as i32;
    let base = gpu.config.rope_theta as f32;
    let rms_eps = gpu.config.rms_norm_eps;
    let scale = gpu.attention_scale;
    let yarn_freqs = gpu.yarn_freqs.as_ref();
    let use_freqs = yarn_freqs.is_some();

    let mut cache: Vec<SteppingKeyValueCache> = (0..gpu.layers.len())
        .map(|_| SteppingKeyValueCache::new())
        .collect();

    time_loop("v6 +readback + argmax (= prod loop) ", |s| {
        let off = s as i32;
        let off_arr = Array::from_int(off);
        let mut h = fresh_fp16(&[1, 1, hidden]);
        for (layer, c) in gpu.layers.iter().zip(cache.iter_mut()) {
            let normed = fast::rms_norm(&h, &layer.input_norm, rms_eps).unwrap();
            let q = layer
                .q_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let k = layer
                .k_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, kv_heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let v = layer
                .v_proj
                .forward(&normed)
                .unwrap()
                .reshape(&[1, 1, kv_heads, head_dim])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();

            let q = fast::rms_norm(&q, &layer.q_norm, rms_eps).unwrap();
            let k = fast::rms_norm(&k, &layer.k_norm, rms_eps).unwrap();

            let q = if use_freqs {
                fast::rope_dynamic(&q, head_dim, false, None::<f32>, 1.0, &off_arr, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&q, head_dim, false, base, 1.0, &off_arr, None::<&Array>)
                    .unwrap()
            };
            let k = if use_freqs {
                fast::rope_dynamic(&k, head_dim, false, None::<f32>, 1.0, &off_arr, yarn_freqs)
                    .unwrap()
            } else {
                fast::rope_dynamic(&k, head_dim, false, base, 1.0, &off_arr, None::<&Array>)
                    .unwrap()
            };

            let view = c.update_and_view(k, v).unwrap();
            let (full_k, full_v) = view.into_dense().unwrap();
            let attn = fast::scaled_dot_product_attention(
                q,
                full_k,
                full_v,
                scale,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
                None::<&Array>,
            )
            .unwrap();

            let attn_out = attn
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap()
                .reshape(&[1, 1, -1])
                .unwrap();
            let attn_out = layer.o_proj.forward(&attn_out).unwrap();
            let h_post_attn = h.add(&attn_out).unwrap();

            let normed_post = fast::rms_norm(&h_post_attn, &layer.post_attn_norm, rms_eps).unwrap();
            let gate = layer.gate_proj.forward(&normed_post).unwrap();
            let up = layer.up_proj.forward(&normed_post).unwrap();
            let mlp_hidden = mlx_rs::nn::silu(&gate).unwrap().multiply(&up).unwrap();
            let mlp_out = layer.down_proj.forward(&mlp_hidden).unwrap();

            h = h_post_attn.add(&mlp_out).unwrap();
        }
        let final_h = fast::rms_norm(&h, &gpu.final_norm, rms_eps).unwrap();
        let logits = match &gpu.lm_head {
            Some(lm) => lm.forward(&final_h).unwrap(),
            None => gpu.embed.forward(&final_h).unwrap(),
        };
        // Production decode loop: cast to f32, eval, readback to CPU, argmax.
        let logits_f32 = logits.as_dtype(Dtype::Float32).unwrap();
        logits_f32.eval().unwrap();
        let s: &[f32] = logits_f32.as_slice::<f32>();
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = 0i32;
        for (i, v) in s.iter().enumerate() {
            if *v > best_v {
                best_v = *v;
                best_i = i as i32;
            }
        }
        std::hint::black_box(best_i);
    });
}

fn v7_production_forward(gpu: &BonsaiQ1Gpu) {
    // Calls the ACTUAL gpu.forward — uses cached_scaled_dot_product_attention,
    // embed_rows, create_attention_mask, etc. This is the production path.
    let mut cache: Vec<Option<SteppingKeyValueCache>> = (0..gpu.layers.len())
        .map(|_| Some(SteppingKeyValueCache::new()))
        .collect();
    let mut tok: i32 = 0;
    time_loop("v7 gpu.forward (= prod entry)      ", |_s| {
        let d = Array::from_slice(&[tok], &[1, 1]);
        let y = gpu.forward(&d, &mut cache).unwrap();
        let logits_f32 = y.as_dtype(Dtype::Float32).unwrap();
        logits_f32.eval().unwrap();
        let s: &[f32] = logits_f32.as_slice::<f32>();
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = 0i32;
        for (i, v) in s.iter().enumerate() {
            if *v > best_v {
                best_v = *v;
                best_i = i as i32;
            }
        }
        tok = best_i;
    });
}

fn run_all(gpu: &BonsaiQ1Gpu, label: &str) {
    println!();
    println!(
        "==== {label}  layers={}  hidden={}  heads={}  kv_heads={}  head_dim={}  inter={} ====",
        gpu.config.layers,
        gpu.config.hidden,
        gpu.config.heads,
        gpu.config.kv_heads,
        gpu.config.head_dim,
        gpu.config.inter
    );
    v1_qmm_only(gpu);
    v2_plus_rope_per_call(gpu);
    v2b_plus_rope_prealloc(gpu);
    v3_plus_rms_norms(gpu);
    v4_plus_kv_sdpa(gpu);
    v5_full_equivalent(gpu);
    v6_with_readback_argmax(gpu);
    v7_production_forward(gpu);
}

#[test]
#[ignore]
fn bisect_decode_bench() {
    println!();
    println!("Bisection decode bench (warmup={WARMUP}, steps={STEPS})");
    println!("Each variant adds one scaffolding component on top of qmm-only.");

    if let Some(d) = dir("Bonsai-1.7B-mlx-1bit") {
        let gpu = BonsaiQ1Engine::load(&d).unwrap().to_gpu().unwrap();
        run_all(&gpu, "Bonsai-1.7B");
    } else {
        eprintln!("Bonsai-1.7B not found, skipping");
    }

    if let Some(d) = dir("Bonsai-8B-mlx-1bit") {
        let gpu = BonsaiQ1Engine::load(&d).unwrap().to_gpu().unwrap();
        run_all(&gpu, "Bonsai-8B");
    } else {
        eprintln!("Bonsai-8B not found, skipping");
    }
}
