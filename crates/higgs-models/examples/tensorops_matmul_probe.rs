//! Probe stage 2: run a real `mpp::tensor_ops::matmul2d` (64x32x128,
//! fp16 -> fp32) through MLX's `mlx_fast_metal_kernel` path and compare
//! against `mlx_rs` matmul.
#![allow(clippy::unwrap_used, clippy::print_stdout)]

use std::ffi::{CStr, CString};

use mlx_rs::{Array, Dtype, Stream, transforms::eval};

const HEADER: &str = r"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
";

// Single-tile GEMM: A [64,128] fp16 row-major, B [128,32] fp16 row-major,
// C [64,32] fp32 row-major (zero-initialized via init_value).
// Metal tensor coords are transposed vs row-major: extent(0) = contiguous dim.
const SOURCE: &str = r"
uint3 tgid = threadgroup_position_in_grid;
tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(const_cast<device half*>(a), dextents<int32_t, 2>{128, 64}, array<int32_t, 2>{1, 128});
tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(const_cast<device half*>(b), dextents<int32_t, 2>{32, 128}, array<int32_t, 2>{1, 32});
tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(c, dextents<int32_t, 2>{32, 64}, array<int32_t, 2>{1, 32});
constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);
matmul2d<desc, execution_simdgroups<4>> op;
auto mA = tA.slice(0, tgid.y * 64);
auto mB = tB.slice(tgid.x * 32, 0);
auto mC = tC.slice(tgid.x * 32, tgid.y * 64);
op.run(mA, mB, mC);
";

#[allow(unsafe_code)]
fn cstr_vec(names: &[&CStr]) -> mlx_sys::mlx_vector_string {
    let ptrs: Vec<*const std::ffi::c_char> = names.iter().map(|s| s.as_ptr()).collect();
    unsafe { mlx_sys::mlx_vector_string_new_data(ptrs.as_ptr().cast_mut(), ptrs.len()) }
}

fn main() {
    // Deterministic small inputs: all-ones => every C element must equal K.
    let a_vals: Vec<f32> = vec![1.0; 64 * 128];
    let b_vals: Vec<f32> = vec![1.0; 128 * 32];
    let a = Array::from_slice(&a_vals, &[64, 128])
        .as_dtype(Dtype::Float16)
        .unwrap();
    let b = Array::from_slice(&b_vals, &[128, 32])
        .as_dtype(Dtype::Float16)
        .unwrap();
    eval([&a, &b]).unwrap();

    #[allow(unsafe_code)]
    let out: Result<Array, String> = unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let out_shape = [64i32, 32];
        let status = mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 128, 1, 1)
            | mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1)
            | mlx_sys::mlx_fast_metal_kernel_config_set_init_value(config, 0.0)
            | mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                out_shape.as_ptr(),
                out_shape.len(),
                mlx_sys::mlx_dtype__MLX_FLOAT32,
            );
        if status != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            return println!("config failed");
        }
        let in_vec = cstr_vec(&[c"a", c"b"]);
        let out_vec = cstr_vec(&[c"c"]);
        let source = CString::new(SOURCE).unwrap_or_default();
        let header = CString::new(HEADER).unwrap_or_default();
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_tensorops_matmul".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        let stream = Stream::task_local_or_default();
        let input_ptrs = [a.as_ptr(), b.as_ptr()];
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
        Ok(c) => {
            eval([&c]).unwrap();
            let got = c.as_slice::<f32>().to_vec();
            let reference = a
                .as_dtype(Dtype::Float32)
                .unwrap()
                .matmul(&b.as_dtype(Dtype::Float32).unwrap())
                .unwrap();
            eval([&reference]).unwrap();
            let want = reference.as_slice::<f32>().to_vec();
            let max_abs = got
                .iter()
                .zip(want.iter())
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            let max_rel = got
                .iter()
                .zip(want.iter())
                .map(|(g, w)| {
                    if w.abs() > 1e-6 {
                        (g - w).abs() / w.abs()
                    } else {
                        (g - w).abs()
                    }
                })
                .fold(0.0f32, f32::max);
            println!("matmul2d ran: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
            if max_rel < 1e-2 {
                println!("TENSOROPS MATMUL: OK");
            } else {
                println!("TENSOROPS MATMUL: MISMATCH");
            }
        }
        Err(e) => println!("TENSOROPS MATMUL: FAILED: {e}"),
    }
}
