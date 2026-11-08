//! Prefill op microbench at Clover-35B-A3B shapes with synthetic weights:
//! dense bf16 GEMM ceiling, 4-bit `quantized_matmul` at M rows for every dense
//! projection, the switch-MLP expert gather at L routed tokens (kernel chosen
//! by `HIGGS_MOE_GATHER_KERNEL`, as in production), unfused D=256 SDPA per
//! full-attention layer vs KV length, the fused GDN recurrence kernel at T
//! (copy of the production kernel source), and the depthwise conv proxy.
//!
//! `BENCH=dense|moe|sdpa|gdn|conv|all` (default all), `ROWS=512,1024,2048`,
//! `KV=1024,4096,8192`.
#![allow(
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    unsafe_code
)]

use std::ffi::{CString, c_char};
use std::time::Instant;

use higgs_models::qwen3_next::switch_mlp_gather_probe::SwitchMlpGather;
use mlx_rs::{Array, Dtype, Stream, fast, ops, random, transforms::eval};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// ms per call: median over `samples` batches of `reps` back-to-back lazy calls.
fn time_ms(reps: usize, samples: usize, mut f: impl FnMut() -> Vec<Array>) -> f64 {
    for _ in 0..2 {
        let ys: Vec<Array> = (0..reps).flat_map(|_| f()).collect();
        eval(ys.iter()).unwrap();
    }
    let mut out = Vec::new();
    for _ in 0..samples {
        let t = Instant::now();
        let ys: Vec<Array> = (0..reps).flat_map(|_| f()).collect();
        eval(ys.iter()).unwrap();
        out.push(t.elapsed().as_secs_f64() * 1e3 / reps as f64);
    }
    median(out)
}

fn normal(shape: &[i32], dtype: Dtype) -> Array {
    let a = random::normal::<f32>(shape, None, None, None)
        .unwrap()
        .as_dtype(dtype)
        .unwrap();
    eval([&a]).unwrap();
    a
}

fn quant(shape: &[i32], bits: i32) -> (Array, Array, Array) {
    let w = normal(shape, Dtype::Bfloat16);
    let q = ops::quantize(&w, 64, bits).unwrap();
    eval([&q.0, &q.1, &q.2]).unwrap();
    q
}

fn env_list(name: &str, default: &[i32]) -> Vec<i32> {
    std::env::var(name).map_or_else(
        |_| default.to_vec(),
        |v| v.split(',').map(|s| s.trim().parse().unwrap()).collect(),
    )
}

const BF: Dtype = Dtype::Bfloat16;

fn bench_dense(rows: &[i32]) {
    println!("== dense bf16 GEMM ceiling");
    for (m, k, n) in [(1024, 2048, 2048), (1024, 2048, 12288), (2048, 2048, 2048)] {
        let a = normal(&[m, k], BF);
        let b = normal(&[k, n], BF);
        let t = time_ms(3, 5, || vec![a.matmul(&b).unwrap()]);
        println!(
            "  matmul M={m} K={k} N={n}: {t:.3} ms = {:.2} TFLOPS",
            2.0 * m as f64 * k as f64 * n as f64 / (t * 1e-3) / 1e12
        );
    }
    // (name, N, K, bits, layers using it)
    let shapes: [(&str, i32, i32, i32, i32); 11] = [
        ("gdn in_proj_qkvz", 12288, 2048, 4, 30),
        ("gdn in_proj_ba", 64, 2048, 4, 30),
        ("gdn out_proj", 2048, 4096, 4, 30),
        ("attn q_proj(+gate)", 8192, 2048, 4, 10),
        ("attn k_proj", 512, 2048, 4, 10),
        ("attn v_proj", 512, 2048, 4, 10),
        ("attn o_proj", 2048, 4096, 4, 10),
        ("shared gate", 512, 2048, 4, 40),
        ("shared up", 512, 2048, 4, 40),
        ("shared down", 2048, 512, 4, 40),
        ("router gate 8b", 256, 2048, 8, 40),
    ];
    println!("== 4-bit g64 quantized_matmul at Clover dense shapes");
    for &m in rows {
        let mut total = 0.0;
        for (name, n, k, bits, layers) in shapes {
            let (wq, s, b) = quant(&[n, k], bits);
            let x = normal(&[1, m, k], BF);
            let t = time_ms(3, 5, || {
                vec![ops::quantized_matmul(&x, &wq, &s, &b, true, 64, bits).unwrap()]
            });
            let tflops = 2.0 * m as f64 * k as f64 * n as f64 / (t * 1e-3) / 1e12;
            total += t * f64::from(layers);
            println!(
                "  M={m:4} {name:20} N={n:5} K={k:4} {bits}b: {t:7.3} ms {tflops:5.2} TFLOPS  x{layers:2} = {:7.1} ms",
                t * f64::from(layers)
            );
        }
        println!(
            "  M={m} dense projections per chunk: {total:.1} ms  ({:.0} tok/s if alone)",
            m as f64 / (total * 1e-3)
        );
    }
}

fn bench_moe(rows: &[i32]) {
    let (e, d, i, tk) = (256, 2048, 512, 8);
    let kernel = std::env::var("HIGGS_MOE_GATHER_KERNEL").unwrap_or_else(|_| "qmv(default)".into());
    let pad = std::env::var("HIGGS_MOE_GATHER_PAD").is_ok();
    println!("== switch-MLP expert gather (E={e} I={i} D={d} top{tk}) kernel={kernel} pad={pad}");
    let gate = quant(&[e, i, d], 4);
    let up = quant(&[e, i, d], 4);
    let down = quant(&[e, d, i], 4);
    let mut block = SwitchMlpGather::new(64, 4, [gate, up, down]).unwrap();
    let bytes_all_experts = 3.0 * (e * i * d) as f64 * (0.5 + 4.0 / 64.0);
    for &l in rows {
        let x = normal(&[1, l, d], BF);
        // Uniform routing, distinct experts within a row: expert j*32 + hash.
        let routing: Vec<u32> = (0..l as u32)
            .flat_map(|t| {
                (0..tk as u32).map(move |j| {
                    j * 32 + (t.wrapping_mul(2_654_435_761).wrapping_add(j * 40_503) >> 7) % 32
                })
            })
            .collect();
        let idx = Array::from_slice(&routing, &[1, l, tk]);
        eval([&idx]).unwrap();
        let flops = l as f64 * f64::from(tk) * 3.0 * 2.0 * f64::from(i) * f64::from(d);
        let t = time_ms(2, 5, || vec![block.forward(&x, &idx).unwrap()]);
        println!(
            "  L={l:4} sorted gather: {t:8.2} ms/layer  x40 = {:7.0} ms/chunk  {:.2} TFLOPS  (all-expert byte floor {:.1} ms @120GB/s)",
            t * 40.0,
            flops / (t * 1e-3) / 1e12,
            bytes_all_experts / 120e9 * 1e3
        );
        if std::env::var("MOE_FUSED").is_ok() {
            let tf = time_ms(2, 5, || vec![block.forward_fused(&x, &idx).unwrap()]);
            println!(
                "  L={l:4} fused gate|up:  {tf:8.2} ms/layer  x40 = {:7.0} ms/chunk  {:.2} TFLOPS",
                tf * 40.0,
                flops / (tf * 1e-3) / 1e12
            );
        }
    }
}

fn causal_mask(l: i32, kv: i32) -> Array {
    let offset = kv - l;
    let rows = Array::from_iter(0..kv, &[1, kv]);
    let cols = Array::from_iter(offset..kv, &[l, 1]);
    let m = cols.ge(&rows).unwrap();
    eval([&m]).unwrap();
    m
}

fn bench_sdpa(l: i32, kvs: &[i32]) {
    let (hq, hk) = (16, 2);
    println!("== SDPA per full-attention layer, Hq={hq} Hkv={hk}, L={l} queries");
    for d in [256, 128] {
        let scale = 1.0 / (d as f32).sqrt();
        for &kv in kvs {
            if d == 128 && kv != l {
                continue;
            }
            let q = normal(&[1, hq, l, d], BF);
            let k = normal(&[1, hk, kv, d], BF);
            let v = normal(&[1, hk, kv, d], BF);
            let mask = if kv == l { None } else { Some(causal_mask(l, kv)) };
            let t = time_ms(2, 5, || {
                let m = mask.as_ref().map_or(
                    fast::ScaledDotProductAttentionMask::Causal,
                    fast::ScaledDotProductAttentionMask::Array,
                );
                vec![fast::scaled_dot_product_attention(&q, &k, &v, scale, m, None::<&Array>).unwrap()]
            });
            let flops = 2.0 * 2.0 * f64::from(hq) * f64::from(l) * f64::from(kv) * f64::from(d);
            println!(
                "  D={d} KV={kv:5} mask={}: {t:7.2} ms/FA layer  x10 = {:6.0} ms  ({:.2} TFLOPS dense-equiv)",
                if kv == l { "causal " } else { "array  " },
                t * 10.0,
                flops / (t * 1e-3) / 1e12
            );
        }
    }
}

// --- fused GDN recurrence kernel (copy of production source) ---------------

const GDN_PREAMBLE: &str = r"
#define HIGGS_GDN_GATE(gate, a_value, dt_bias_value, a_log_value) \
  float x = static_cast<float>(a_value) + dt_bias_value; \
  float sp = fmax(x, 0.0f) + log1p(exp(-fabs(x))); \
  float gate = exp(-exp(a_log_value) * sp)
#define HIGGS_GDN_BETA(beta, b_value) \
  float beta = 1.0f / (1.0f + exp(-static_cast<float>(b_value)))
#define HIGGS_GDN_DECAY(state_value, gate) \
  state_value = state_value * gate
#define HIGGS_GDN_UPDATE(state_value, key_value, delta) \
  state_value = state_value + key_value * delta
#define HIGGS_GDN_RECORD_TAPE 0
";

const GDN_BODY: &str = r"
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;
auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;
auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;
auto i_state = reinterpret_cast<const device StateT*>(state_in) + (n * Dv + dv_idx) * Dk;
auto o_state = reinterpret_cast<device StateT*>(state_out) + (n * Dv + dv_idx) * Dk;
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}
float a_log_val = static_cast<float>(a_log[hv_idx]);
float dt_bias_val = static_cast<float>(dt_bias[hv_idx]);
auto a_ = a + b_idx * T * Hv;
auto b_ = b + b_idx * T * Hv;
for (int t = 0; t < T; ++t) {
  HIGGS_GDN_GATE(g_val, a_[hv_idx], dt_bias_val, a_log_val);
  HIGGS_GDN_BETA(beta_val, b_[hv_idx]);
  {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      HIGGS_GDN_DECAY(state[i], g_val);
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);
    auto delta = (v_[dv_idx] - kv_mem) * beta_val;
    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      HIGGS_GDN_UPDATE(state[i], k_[s_idx], delta);
      out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y[dv_idx] = static_cast<InT>(out);
    }
  }
  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  a_ += Hv;
  b_ += Hv;
}
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = static_cast<StateT>(state[i]);
}
";

struct GdnKernel(mlx_sys::mlx_fast_metal_kernel);

impl GdnKernel {
    fn new() -> Self {
        let inputs: [&std::ffi::CStr; 9] = [
            c"q", c"k", c"v", c"a_log", c"a", c"dt_bias", c"b", c"state_in", c"T",
        ];
        let outputs: [&std::ffi::CStr; 2] = [c"y", c"state_out"];
        let in_ptrs: Vec<*const c_char> = inputs.iter().map(|s| s.as_ptr()).collect();
        let out_ptrs: Vec<*const c_char> = outputs.iter().map(|s| s.as_ptr()).collect();
        let source = CString::new(format!("{GDN_PREAMBLE}{GDN_BODY}")).unwrap();
        unsafe {
            let in_vec = mlx_sys::mlx_vector_string_new_data(in_ptrs.as_ptr().cast_mut(), in_ptrs.len());
            let out_vec =
                mlx_sys::mlx_vector_string_new_data(out_ptrs.as_ptr().cast_mut(), out_ptrs.len());
            let k = mlx_sys::mlx_fast_metal_kernel_new(
                c"bench_gated_delta".as_ptr(),
                in_vec,
                out_vec,
                source.as_ptr(),
                c"".as_ptr(),
                true,
                false,
            );
            mlx_sys::mlx_vector_string_free(in_vec);
            mlx_sys::mlx_vector_string_free(out_vec);
            Self(k)
        }
    }

    fn config(
        in_dtype: mlx_sys::mlx_dtype,
        state_dtype: mlx_sys::mlx_dtype,
        b: i32,
        t: i32,
        hk: i32,
        dk: i32,
        hv: i32,
        dv: i32,
    ) -> mlx_sys::mlx_fast_metal_kernel_config {
        unsafe {
            let c = mlx_sys::mlx_fast_metal_kernel_config_new();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(c, c"InT".as_ptr(), in_dtype);
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                c,
                c"StateT".as_ptr(),
                state_dtype,
            );
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(c, c"Dk".as_ptr(), dk);
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(c, c"Dv".as_ptr(), dv);
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(c, c"Hk".as_ptr(), hk);
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(c, c"Hv".as_ptr(), hv);
            mlx_sys::mlx_fast_metal_kernel_config_set_grid(c, 32, dv, b * hv);
            mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(c, 32, 4, 1);
            let y_shape = [b, t, hv, dv];
            let s_shape = [b, hv, dv, dk];
            mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(c, y_shape.as_ptr(), 4, in_dtype);
            mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(c, s_shape.as_ptr(), 4, state_dtype);
            c
        }
    }

    fn apply(
        &self,
        config: mlx_sys::mlx_fast_metal_kernel_config,
        ins: [&Array; 8],
        t: i32,
    ) -> (Array, Array) {
        let stream = Stream::task_local_or_default();
        unsafe {
            let t_scalar = mlx_sys::mlx_array_new_int(t);
            let ptrs = [
                ins[0].as_ptr(),
                ins[1].as_ptr(),
                ins[2].as_ptr(),
                ins[3].as_ptr(),
                ins[4].as_ptr(),
                ins[5].as_ptr(),
                ins[6].as_ptr(),
                ins[7].as_ptr(),
                t_scalar,
            ];
            let in_vec = mlx_sys::mlx_vector_array_new_data(ptrs.as_ptr(), ptrs.len());
            let mut out_vec = mlx_sys::mlx_vector_array_new();
            let status =
                mlx_sys::mlx_fast_metal_kernel_apply(&raw mut out_vec, self.0, in_vec, config, stream.as_ptr());
            assert_eq!(status, 0, "gdn kernel apply failed");
            let mut y = mlx_sys::mlx_array_new();
            let mut s = mlx_sys::mlx_array_new();
            mlx_sys::mlx_vector_array_get(&raw mut y, out_vec, 0);
            mlx_sys::mlx_vector_array_get(&raw mut s, out_vec, 1);
            mlx_sys::mlx_vector_array_free(in_vec);
            mlx_sys::mlx_vector_array_free(out_vec);
            mlx_sys::mlx_array_free(t_scalar);
            (Array::from_ptr(y), Array::from_ptr(s))
        }
    }
}

fn bench_gdn(ts: &[i32]) {
    let (hk, dk, hv, dv) = (16, 128, 32, 128);
    println!("== fused GDN recurrence kernel (Hk={hk} Dk={dk} Hv={hv} Dv={dv}), grid (32,{dv},{hv}) tg (32,4,1)");
    let kernel = GdnKernel::new();
    let a_log = normal(&[hv], Dtype::Float32);
    let dt_bias = normal(&[hv], Dtype::Float32);
    let state = Array::zeros::<f32>(&[1, hv, dv, dk]).unwrap();
    eval([&state]).unwrap();
    for &t in ts {
        let q = normal(&[1, t, hk, dk], BF);
        let k = normal(&[1, t, hk, dk], BF);
        let v = normal(&[1, t, hv, dv], BF);
        let a = normal(&[1, t, hv], BF);
        let b = normal(&[1, t, hv], BF);
        let in_dtype = unsafe { mlx_sys::mlx_array_dtype(q.as_ptr()) };
        let st_dtype = unsafe { mlx_sys::mlx_array_dtype(state.as_ptr()) };
        let config = GdnKernel::config(in_dtype, st_dtype, 1, t, hk, dk, hv, dv);
        let ms = time_ms(2, 5, || {
            let (y, s) = kernel.apply(config, [&q, &k, &v, &a_log, &a, &dt_bias, &b, &state], t);
            vec![y, s]
        });
        // Useful FLOPs: per step per (hv, dv): 2*Dk MACs for kv_mem/out + 2*Dk for decay/update.
        let flops = t as f64 * f64::from(hv) * f64::from(dv) * 4.0 * f64::from(dk) * 2.0;
        println!(
            "  T={t:4}: {ms:8.2} ms/layer  x30 = {:7.0} ms/chunk  ({:.1} us/step, {:.2} TFLOPS-equiv)",
            ms * 30.0,
            ms * 1e3 / t as f64,
            flops / (ms * 1e-3) / 1e12
        );
        unsafe { mlx_sys::mlx_fast_metal_kernel_config_free(config) };
    }
}

fn bench_conv(ts: &[i32]) {
    let c = 8192;
    println!("== depthwise conv1d K=4 over {c} channels (MLX conv1d proxy for the canonical conv kernel)");
    let w = normal(&[c, 4, 1], BF);
    for &t in ts {
        let x = normal(&[1, t + 3, c], BF);
        let ms = time_ms(3, 5, || vec![ops::conv1d(&x, &w, 1, 0, 1, c).unwrap()]);
        println!("  T={t:4}: {ms:6.2} ms/layer  x30 = {:5.0} ms/chunk", ms * 30.0);
    }
}

fn main() {
    let which = std::env::var("BENCH").unwrap_or_else(|_| "all".into());
    let rows = env_list("ROWS", &[512, 1024, 2048]);
    let kvs = env_list("KV", &[1024, 2048, 4096, 8192]);
    let all = which == "all";
    if all || which == "dense" {
        bench_dense(&rows);
    }
    if all || which == "moe" {
        bench_moe(&rows);
    }
    if all || which == "sdpa" {
        bench_sdpa(1024, &kvs);
    }
    if all || which == "gdn" {
        bench_gdn(&rows);
    }
    if all || which == "conv" {
        bench_conv(&rows);
    }
}
