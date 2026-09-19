//! Spike v1: fused FlashAttention via TensorOps, mask-none path.
//! One threadgroup (128 threads, 4 simdgroups) per (q-head, 64-row Q tile).
//! S-loop with online softmax; D chunked in 64s (Rust-unrolled per chunk);
//! SV reuses P coop directly (Metal 27). Exactness vs MLX SDPA.
#![allow(clippy::unwrap_used, clippy::print_stdout)]

use std::ffi::{CStr, CString};

use mlx_rs::{Array, Stream, fast, transforms::eval};

const HEADER: &str = r"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
";

// Template ints: HQ, HKV, GQA, L, S, D, NTILE, SCALE_BITS.
// Inputs: q [1,HQ,L,D], k [1,HKV,S,D], v [1,HKV,S,D] f32 row-major.
// Output: o [1,HQ,L,D] f32.
// {CHUNK} is expanded per D-chunk (Rust-unrolled): declares oCc, accumulates,
// normalizes and stores chunk c.
fn source(nchunks: usize) -> String {
    let mut chunk_accum = String::new();
    let mut chunk_store = String::new();
    for c in 0..nchunks {
        chunk_accum.push_str(&format!(
            r"
  // ---- D chunk {c} ----
  auto oC{c} = sv_acc_op.get_destination_cooperative_tensor<PInTy, KTy, float>();
  for (uint i = 0; i < oC{c}.get_capacity(); ++i) {{ oC{c}[i] = 0.0f; }}
"
        ));
    }
    chunk_accum.push_str(
        r"
  for (int s0 = 0; s0 < S; s0 += 64) {
    tensor<device float, dextents<int, 2>, tensor_inline> tK(const_cast<device float*>(kh_base) + s0 * D, dextents<int, 2>{D, 64}, array<int, 2>{1, D});
    qk_op.run(tQ, tK, ctS);
    for (auto it = ctS.begin(); it != ctS.end(); ++it) { *it *= scale; }
    reduce_rows(ctS, ctMax, reduction_operation::max, metal::numeric_limits<float>::lowest());
    for (uint i = 0; i < nrow; ++i) {
      float m_old = mC[i];
      float m_new = metal::max(m_old, ctMax[i]);
      float a = metal::exp(m_old - m_new);
      mC[i] = m_new;
      aC[i] = a;
      lC[i] *= a;
    }
    for (auto it = ctS.begin(); it != ctS.end(); ++it) {
      auto mit = mC.map_iterator(it);
      *it = metal::exp(*it - *mit);
    }
    reduce_rows(ctS, ctSum, reduction_operation::sum, 0.0f);
    for (uint i = 0; i < nrow; ++i) { lC[i] += ctSum[i]; }
    // Fresh P adaptor each block: it must see this block's P values.
    auto pIn = sv_acc_op.get_left_input_cooperative_tensor<float, float, float>(ctS);
",
    );
    for c in 0..nchunks {
        chunk_accum.push_str(&format!(
            r"
    {{
      tensor<device float, dextents<int, 2>, tensor_inline> tV(
          const_cast<device float*>(vh_base) + s0 * D + {c} * 64, dextents<int, 2>{{64, 64}}, array<int, 2>{{1, D}});
      for (auto it = oC{c}.begin(); it != oC{c}.end(); ++it) {{
        auto ait = aC.map_iterator(it);
        *it *= *ait;
      }}
      sv_acc_op.run(pIn, tV, oC{c});
    }}
"
        ));
    }
    chunk_accum.push_str("  }\n");
    for c in 0..nchunks {
        chunk_store.push_str(&format!(
            r"
  {{
    if (NORM_ON != 0) {{
      for (auto it = oC{c}.begin(); it != oC{c}.end(); ++it) {{
        auto lit = lC.map_iterator(it);
        *it /= *lit;
      }}
    }}
    tensor<device float, dextents<int, 2>, tensor_inline> tO(
        oh + {c} * 64, dextents<int, 2>{{64, 16}}, array<int, 2>{{1, D}});
    oC{c}.store(tO);
  }}
"
        ));
    }
    r"
uint g = threadgroup_position_in_grid.x;
uint h = g / NTILE;
uint q0 = (g % NTILE) * 16;
uint hk = h / GQA;
float scale = as_type<float>(uint(SCALE_BITS));
if (SG_DIAG != 0) {
  o[thread_position_in_grid.x] = (float)(simdgroup_index_in_threadgroup * 1000 + threadgroup_position_in_grid.x);
  return;
}

const device float* qh = q + (h * L + q0) * D;
const device float* kh_base = k + (hk * S) * D;
const device float* vh_base = v + (hk * S) * D;
device float* oh = o + (h * L + q0) * D;

tensor<device float, dextents<int, 2>, tensor_inline> tQ(const_cast<device float*>(qh), dextents<int, 2>{D, 16}, array<int, 2>{1, D});

constexpr auto qk_desc = matmul2d_descriptor(16, 64, static_cast<int>(dynamic_extent), false, true, false);
matmul2d<qk_desc, execution_simdgroup> qk_op;
constexpr auto sv_acc_desc = matmul2d_descriptor(16, 64, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<sv_acc_desc, execution_simdgroup> sv_acc_op;

using QTy = metal::remove_addrspace_t<decltype(tQ)>;
tensor<device float, dextents<int, 2>, tensor_inline> tK0(const_cast<device float*>(kh_base), dextents<int, 2>{D, 64}, array<int, 2>{1, D});
using KTy = metal::remove_addrspace_t<decltype(tK0)>;

auto ctS = qk_op.get_destination_cooperative_tensor<QTy, KTy, float>();
auto ctMax = qk_op.get_row_reduction_destination_cooperative_tensor<QTy, KTy, float>();
auto ctSum = qk_op.get_row_reduction_destination_cooperative_tensor<QTy, KTy, float>();
auto mC = qk_op.get_row_reduction_destination_cooperative_tensor<QTy, KTy, float>();
auto lC = qk_op.get_row_reduction_destination_cooperative_tensor<QTy, KTy, float>();
auto aC = qk_op.get_row_reduction_destination_cooperative_tensor<QTy, KTy, float>();
uint nrow = mC.get_capacity();
for (uint i = 0; i < nrow; ++i) { mC[i] = -metal::numeric_limits<float>::infinity(); lC[i] = 0.0f; }

auto pIn0 = sv_acc_op.get_left_input_cooperative_tensor<float, float, float>(ctS);
using PInTy = metal::remove_addrspace_t<decltype(pIn0)>;

if (SG_DIAG == 2) {
  // Ground truth: QK run into fresh dest coop, then capacity + store.
  // Q [16,D], K [64,D] ones => scores all = D. D=64.
  tensor<device float, dextents<int, 2>, tensor_inline> tK(const_cast<device float*>(kh_base), dextents<int, 2>{D, 64}, array<int, 2>{1, D});
  qk_op.run(tQ, tK, ctS);
  uint lane = thread_position_in_grid.x;
  o[lane] = (float)ctS.get_capacity();
  tensor<device float, dextents<int, 2>, tensor_inline> tO(
      oh, dextents<int, 2>{64, 16}, array<int, 2>{1, D});
  ctS.store(tO);
  // Rowmax of scores (=D=64) into ctMax; stash first element at o[2048].
  reduce_rows(ctS, ctMax, reduction_operation::max, metal::numeric_limits<float>::lowest());
  if (lane < 32) { o[2048 + lane] = ctMax[lane % 16]; }
  return;
}

/*ACCUM*/
/*STORE*/
"
    .replace("/*ACCUM*/", &chunk_accum)
    .replace("/*STORE*/", &chunk_store)
}

#[allow(unsafe_code)]
fn cstr_vec(names: &[&CStr]) -> mlx_sys::mlx_vector_string {
    let ptrs: Vec<*const std::ffi::c_char> = names.iter().map(|s| s.as_ptr()).collect();
    unsafe { mlx_sys::mlx_vector_string_new_data(ptrs.as_ptr().cast_mut(), ptrs.len()) }
}

fn main() {
    const HQ: i32 = 2;
    const HKV: i32 = 1;
    let l_env: i32 = std::env::var("HIGGS_FA_L")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let s_env: i32 = std::env::var("HIGGS_FA_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    const D: i32 = 64;
    assert!(
        l_env % 64 == 0 && s_env % 64 == 0,
        "L,S must be multiples of 64"
    );
    let (L, S) = (l_env, s_env);
    const NTILE_DIV: i32 = 16;
    let ntile: i32 = L / NTILE_DIV;
    const GQA: i32 = HQ / HKV;
    let scale: f32 = 1.0 / (D as f32).sqrt();
    let groups = HQ * ntile;

    let nq = (HQ * L * D) as usize;
    let nkv = (HKV * S * D) as usize;
    let qv: Vec<f32> = (0..nq)
        .map(|i| ((i % 29) as f32) * 0.03125 - 0.4375)
        .collect();
    let kv: Vec<f32> = (0..nkv)
        .map(|i| ((i % 23) as f32) * 0.03125 - 0.34375)
        .collect();
    let q = Array::from_slice(&qv, &[1, HQ, L, D]);
    let k = Array::from_slice(&kv, &[1, HKV, S, D]);
    let v = Array::from_slice(&kv, &[1, HKV, S, D]);
    eval([&q, &k, &v]).unwrap();

    let src = source((D / 64) as usize);
    #[allow(unsafe_code)]
    let out: Result<Array, String> = unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let mut status = mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, groups * 32, 1, 1)
            | mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1)
            | mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                [1, HQ, L, D].as_ptr(),
                4,
                mlx_sys::mlx_dtype__MLX_FLOAT32,
            );
        for (name, val) in [
            ("HQ", HQ),
            ("HKV", HKV),
            ("GQA", GQA),
            ("L", L),
            ("S", S),
            ("D", D),
            ("NTILE", ntile),
            (
                "NORM_ON",
                if std::env::var("HIGGS_FA_NONORM").is_ok() {
                    0
                } else {
                    1
                },
            ),
            ("SCALE_BITS", scale.to_bits() as i32),
            (
                "SG_DIAG",
                if std::env::var("HIGGS_FA_SGDIAG").is_ok() {
                    1
                } else if std::env::var("HIGGS_FA_STOREDIAG").is_ok() {
                    2
                } else {
                    0
                },
            ),
        ] {
            let cname = CString::new(name).unwrap_or_default();
            status |= mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config,
                cname.as_ptr(),
                val,
            );
        }
        if status != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            return println!("config failed");
        }
        let in_vec = cstr_vec(&[c"q", c"k", c"v", c"z"]);
        let out_vec = cstr_vec(&[c"o"]);
        let source_c = CString::new(src).unwrap_or_default();
        let header = CString::new(HEADER).unwrap_or_default();
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_fused_attn_v1".as_ptr(),
            in_vec,
            out_vec,
            source_c.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        let stream = Stream::task_local_or_default();
        let z = Array::from_slice(&[3.0f32; 16], &[16]);
        eval([&z]).unwrap();
        let input_ptrs = [q.as_ptr(), k.as_ptr(), v.as_ptr(), z.as_ptr()];
        let inputs_vec = mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len());
        let mut outputs_vec = mlx_sys::mlx_vector_array_new();
        let apply_status = mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            kernel,
            inputs_vec,
            config,
            stream.as_ptr(),
        );
        mlx_sys::mlx_fast_metal_kernel_free(kernel);
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        if apply_status != 0 {
            mlx_sys::mlx_vector_array_free(outputs_vec);
            Err("apply failed (see MLX error above)".to_owned())
        } else {
            let mut output_ptr = mlx_sys::mlx_array_new();
            let get_status = mlx_sys::mlx_vector_array_get(&raw mut output_ptr, outputs_vec, 0);
            mlx_sys::mlx_vector_array_free(outputs_vec);
            if get_status == 0 {
                Ok(Array::from_ptr(output_ptr))
            } else {
                mlx_sys::mlx_array_free(output_ptr);
                Err("output readback failed".to_owned())
            }
        }
    };

    match out {
        Ok(o) => {
            eval([&o]).unwrap();
            let got = o.as_slice::<f32>().to_vec();
            let reference = fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                scale,
                None::<fast::ScaledDotProductAttentionMask>,
                None::<&Array>,
            )
            .unwrap();
            eval([&reference]).unwrap();
            let want = reference.as_slice::<f32>().to_vec();
            let max_rel = got
                .iter()
                .zip(want.iter())
                .map(|(g, w)| {
                    if w.abs() > 1e-5 {
                        (g - w).abs() / w.abs()
                    } else {
                        (g - w).abs()
                    }
                })
                .fold(0.0f32, f32::max);
            if std::env::var("HIGGS_FA_SGDIAG").is_ok() {
                println!("sg diag first 160: {:?}", &got[..160]);
            }
            if std::env::var("HIGGS_FA_STOREDIAG").is_ok() {
                let caps: Vec<f32> = got[..512].to_vec();
                let sum: f32 = caps.iter().sum();
                println!("post-run capacity sum over 512 lanes = {sum} (expect 512*32=16384)");
                // Rowmax check: kernel stored ctMax[lane%16] at o[2048+lane]
                // for lanes < 32 (TG0). Compare vs CPU rowmax, head 0 rows 0..15.
                let mut max_rel_m = 0.0f32;
                for r in 0..16 {
                    let mut mx = f32::NEG_INFINITY;
                    for c in 0..64 {
                        let mut s = 0.0;
                        for d in 0..64 {
                            s += qv[r * 64 + d] * kv[c * 64 + d];
                        }
                        if s > mx {
                            mx = s;
                        }
                    }
                    let gotv = got[2048 + r];
                    max_rel_m = max_rel_m.max((gotv - mx).abs() / mx.abs().max(1e-5));
                }
                println!("rowmax vs CPU max_rel={max_rel_m:.3e}");
            }
            if std::env::var("HIGGS_FA_NONORM").is_ok() {
                // Unnormalized reference only valid for single S-block runs.
                // O_unnorm[r] = sum_c P[r,c] * V[c], head 0 rows 0..64, S=64.
                let s_cur = s_env;
                let mut max_rel_u = 0.0f32;
                for r in 0..64 {
                    // P row from kernel output rows (normalized off => raw accum).
                    // Reference: recompute P then O.
                    let mut mx = f32::NEG_INFINITY;
                    for c in 0..s_cur as usize {
                        let mut s = 0.0;
                        for d in 0..64 {
                            s += qv[r * 64 + d] * kv[c * 64 + d];
                        }
                        s *= scale;
                        if s > mx {
                            mx = s;
                        }
                    }
                    for dc in 0..64 {
                        let mut o_ref = 0.0;
                        for c in 0..s_cur as usize {
                            let mut s = 0.0;
                            for d in 0..64 {
                                s += qv[r * 64 + d] * kv[c * 64 + d];
                            }
                            s *= scale;
                            o_ref += (s - mx).exp() * kv[c * 64 + dc];
                        }
                        let gotv = got[r * 64 + dc];
                        let denom = o_ref.abs().max(1e-5);
                        max_rel_u = max_rel_u.max((gotv - o_ref).abs() / denom);
                    }
                }
                println!("unnorm O vs CPU max_rel={max_rel_u:.3e}");
                // Locate unnorm mismatches (single-block => rescale is no-op).
                {
                    let mut bad: Vec<usize> = Vec::new();
                    for r in 0..128 {
                        for dc in 0..64 {
                            let mut mx = f32::NEG_INFINITY;
                            for c in 0..s_env as usize {
                                let mut s = 0.0;
                                for d in 0..64 {
                                    s += qv[r * 64 + d] * kv[c * 64 + d];
                                }
                                s *= scale;
                                if s > mx {
                                    mx = s;
                                }
                            }
                            let mut o_ref = 0.0;
                            for c in 0..s_env as usize {
                                let mut s = 0.0;
                                for d in 0..64 {
                                    s += qv[r * 64 + d] * kv[c * 64 + d];
                                }
                                s *= scale;
                                o_ref += (s - mx).exp() * kv[c * 64 + dc];
                            }
                            // head 0 only here; got layout [1,2,128,64].
                            let gotv = got[r * 64 + dc];
                            if o_ref.abs() > 1e-5 && (gotv - o_ref).abs() / o_ref.abs() > 1e-4 {
                                bad.push(r * 64 + dc);
                                if bad.len() >= 16 {
                                    break;
                                }
                            }
                        }
                        if bad.len() >= 16 {
                            break;
                        }
                    }
                    println!("unnorm bad head0 (first 16): {bad:?}");
                    println!(
                        "got row16[:8]={:?} got row0[:8]={:?}",
                        &got[1024..1032],
                        &got[..8]
                    );
                }
                // Show first output elements vs ref for calibration.
                {
                    let mut mx = f32::NEG_INFINITY;
                    for c in 0..s_env as usize {
                        let mut s = 0.0;
                        for d in 0..64 {
                            s += qv[d] * kv[c * 64 + d];
                        }
                        s *= scale;
                        if s > mx {
                            mx = s;
                        }
                    }
                    let mut refs = Vec::new();
                    for dc in 0..4 {
                        let mut o_ref = 0.0;
                        for c in 0..s_env as usize {
                            let mut s = 0.0;
                            for d in 0..64 {
                                s += qv[d] * kv[c * 64 + d];
                            }
                            s *= scale;
                            o_ref += (s - mx).exp() * kv[c * 64 + dc];
                        }
                        refs.push(o_ref);
                    }
                    println!("unnorm got[0..4]={:?} ref={refs:?}", &got[..4]);
                }
            }
            if std::env::var("HIGGS_FA_DIAG").is_ok() {
                // P-diag: rows of got hold P (exp scores) for all 64 query rows.
                // Compare against CPU softmax numerators.
                let rows = got.len() / 64;
                let sums: Vec<f32> = (0..rows.min(8))
                    .map(|r| got[r * 64..(r + 1) * 64].iter().sum())
                    .collect();
                println!("P row sums (first 8): {sums:?}");
                // CPU reference P for head 0, rows 0..64.
                let qh: Vec<f32> = qv[0..(64 * 64)].to_vec();
                let kh: Vec<f32> = kv[0..(64 * 64)].to_vec();
                let mut max_rel_p = 0.0f32;
                for r in 0..64 {
                    let mut mx = f32::NEG_INFINITY;
                    for c in 0..64 {
                        let mut s = 0.0;
                        for d in 0..64 {
                            s += qh[r * 64 + d] * kh[c * 64 + d];
                        }
                        s *= scale;
                        if s > mx {
                            mx = s;
                        }
                    }
                    for c in 0..64 {
                        let mut s = 0.0;
                        for d in 0..64 {
                            s += qh[r * 64 + d] * kh[c * 64 + d];
                        }
                        s *= scale;
                        let pref = (s - mx).exp();
                        let gotv = got[r * 64 + c];
                        let denom = pref.abs().max(1e-5);
                        max_rel_p = max_rel_p.max((gotv - pref).abs() / denom);
                    }
                }
                println!("P vs CPU max_rel={max_rel_p:.3e}");
                // Show ref row 0 distribution for calibration.
                {
                    let qh: Vec<f32> = qv[0..(64 * 64)].to_vec();
                    let kh: Vec<f32> = kv[0..(64 * 64)].to_vec();
                    let mut scores: Vec<f32> = Vec::with_capacity(64);
                    for c in 0..64 {
                        let mut s = 0.0;
                        for d in 0..64 {
                            s += qh[d] * kh[c * 64 + d];
                        }
                        scores.push(s * scale);
                    }
                    let mx = scores.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
                    let mn = scores.iter().fold(f32::INFINITY, |a, b| a.min(*b));
                    let prefs: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                    let psum: f32 = prefs.iter().sum();
                    println!(
                        "ref row0: max={mx:.4} min={mn:.4} prefsum={psum:.2} prefs[:8]={:?}",
                        &prefs[..8]
                    );
                    println!("got row0 [:8]={:?}", &got[..8]);
                }
            }
            if max_rel < 1e-4 {
                println!("FUSED ATTN V1: OK");
            } else {
                println!(
                    "FUSED ATTN V1: MISMATCH got8={:?} want8={:?}",
                    &got[..8],
                    &want[..8]
                );
                let bad_idx: Vec<usize> = got
                    .iter()
                    .zip(want.iter())
                    .enumerate()
                    .filter(|(_, (g, w))| {
                        (**w).abs() > 1e-5 && (**g - **w).abs() / (**w).abs() > 1e-4
                    })
                    .map(|(i, _)| i)
                    .take(16)
                    .collect();
                let nbad = got
                    .iter()
                    .zip(want.iter())
                    .filter(|(g, w)| (**w).abs() > 1e-5 && (**g - **w).abs() / (**w).abs() > 1e-4)
                    .count();
                println!("bad count={nbad} first16={bad_idx:?}");
            }
        }
        Err(e) => println!("FUSED ATTN V1: FAILED: {e}"),
    }
}
