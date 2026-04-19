//! Public-CoreML MLModel wrapper for the ANE `lm_head` LUT6 path.
//!
//! Parallel to [`crate::qwen3_next_ane::AneProjKernel`] — same abstract role
//! (`y = x @ W^T` on ANE) but different compile/load pipeline. Weights are
//! 6-bit palettized via `scripts/palettize_lm_head.py` and the `.mlmodelc`
//! is loaded through the public `MLModel` API (see `ane_bridge_mlmodel.m`
//! for rationale).
//!
//! Thread-safety: `MLModel` is thread-safe per Apple's docs
//! (<https://developer.apple.com/documentation/coreml/mlmodel>). The handle
//! can be used from any thread after construction.

#![allow(
    unsafe_code,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::multiple_unsafe_ops_per_block,
    clippy::undocumented_unsafe_blocks
)]

use std::ffi::{CString, c_char, c_void};
use std::os::raw::c_int;

use half::f16;
use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype};

// --- FFI --------------------------------------------------------------------

unsafe extern "C" {
    fn ane_mlmodel_load(mlmodelc_path: *const c_char, error_out: *mut *mut c_char)
        -> *mut c_void;

    fn ane_mlmodel_predict_fp16(
        handle: *mut c_void,
        input_name: *const c_char,
        x_fp16: *const u16,
        x_count: usize,
        x_shape: *const i64,
        x_rank: c_int,
        output_name: *const c_char,
        y_fp16: *mut u16,
        y_count: usize,
        error_out: *mut *mut c_char,
    ) -> bool;

    fn ane_mlmodel_predict_fp16_multi(
        handle: *mut c_void,
        input_name: *const c_char,
        x_fp16: *const u16,
        x_count: usize,
        x_shape: *const i64,
        x_rank: c_int,
        output_names: *const *const c_char,
        y_fp16_buffers: *const *mut u16,
        y_counts: *const usize,
        n_outputs: c_int,
        error_out: *mut *mut c_char,
    ) -> bool;

    fn ane_mlmodel_free(handle: *mut c_void);

    fn ane_mlmodel_verify_ane_dispatch(
        mlmodelc_path: *const c_char,
        out_report: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> bool;

    // `free` from the C runtime — used to release NUL-terminated error
    // strings the bridge allocates with `malloc`.
    fn free(ptr: *mut c_void);
}

/// Must match `INPUT_NAME` / `OUTPUT_NAME` in `scripts/palettize_lm_head.py`.
const INPUT_NAME: &str = "x";
const OUTPUT_NAME: &str = "logits";

// --- Error helper ----------------------------------------------------------

unsafe fn take_error(ptr: *mut c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: the bridge malloc'd a NUL-terminated C string; we take ownership
    // by reading and then freeing with `libc::free` (malloc/free from the
    // same heap on macOS).
    let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
    let s = cstr.to_string_lossy().into_owned();
    unsafe { free(ptr.cast()); }
    s
}

// --- Kernel -----------------------------------------------------------------

pub struct AneLmHeadLut6Kernel {
    handle: *mut c_void,
    pub vocab: usize,
    pub hidden: usize,
    pub seq_len: usize,
}

// SAFETY: `MLModel` is documented thread-safe; the handle only wraps a
// retained CFTypeRef. No interior mutability.
#[allow(unsafe_code)]
unsafe impl Send for AneLmHeadLut6Kernel {}
#[allow(unsafe_code)]
unsafe impl Sync for AneLmHeadLut6Kernel {}

impl std::fmt::Debug for AneLmHeadLut6Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AneLmHeadLut6Kernel")
            .field("vocab", &self.vocab)
            .field("hidden", &self.hidden)
            .field("seq_len", &self.seq_len)
            .finish()
    }
}

impl Drop for AneLmHeadLut6Kernel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle came from `ane_mlmodel_load`; freed exactly once
            // here since we zero it out.
            unsafe { ane_mlmodel_free(self.handle); }
            self.handle = std::ptr::null_mut();
        }
    }
}

impl AneLmHeadLut6Kernel {
    /// Load a compiled `.mlmodelc` directory and configure it for CPU+ANE.
    pub fn load(
        mlmodelc_path: &str,
        vocab: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Result<Self, String> {
        let path = CString::new(mlmodelc_path)
            .map_err(|e| format!("mlmodelc_path contains NUL: {e}"))?;
        let mut err_ptr: *mut c_char = std::ptr::null_mut();
        // SAFETY: pointers are valid; bridge writes to `err_ptr` on failure.
        let handle = unsafe { ane_mlmodel_load(path.as_ptr(), &mut err_ptr) };
        if handle.is_null() {
            let msg = unsafe { take_error(err_ptr) };
            return Err(if msg.is_empty() {
                format!("ane_mlmodel_load({mlmodelc_path}) returned NULL")
            } else {
                msg
            });
        }
        Ok(Self { handle, vocab, hidden, seq_len })
    }

    /// Run `y = x @ W^T` through the compiled MLModel.
    ///
    /// `x`: `[B, S, hidden]`, any fp dtype. `S <= seq_len`. Returns
    /// `[B, S, vocab]` in the input's dtype.
    pub fn dispatch(&self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(Exception::custom(format!(
                "AneLmHeadLut6Kernel::dispatch expects rank-3 input, got {shape:?}"
            )));
        }
        let b = shape[0] as usize;
        let s = shape[1] as usize;
        let h = shape[2] as usize;
        if h != self.hidden {
            return Err(Exception::custom(format!(
                "lm_head hidden mismatch: input {h}, kernel {}",
                self.hidden
            )));
        }
        if s > self.seq_len {
            return Err(Exception::custom(format!(
                "lm_head seq too long: input {s}, kernel seq_len {}",
                self.seq_len
            )));
        }
        if b == 0 || s == 0 {
            return Err(Exception::custom(format!(
                "lm_head degenerate shape: B={b}, S={s}"
            )));
        }

        // --- f32 view of the activation ---
        let x_f32 = if x.dtype() == Dtype::Float32 {
            x.clone()
        } else {
            x.as_dtype(Dtype::Float32)?
        };
        x_f32.eval()?;
        let x_slice: &[f32] = x_f32.as_slice::<f32>();

        let pad = self.seq_len;
        let vocab = self.vocab;

        // Prepare an fp16 `[1, pad, hidden]` input buffer, reused per batch.
        let mut in_fp16 = vec![0u16; pad * h];
        let mut out_fp16 = vec![0u16; pad * vocab];
        let in_shape: [i64; 3] = [1, pad as i64, h as i64];

        let in_name = CString::new(INPUT_NAME).expect("INPUT_NAME not NUL-terminated");
        let out_name = CString::new(OUTPUT_NAME).expect("OUTPUT_NAME not NUL-terminated");

        let mut out_all = vec![0.0f32; b * s * vocab];

        for bi in 0..b {
            let src = &x_slice[bi * s * h..(bi + 1) * s * h];
            // Copy S rows into the first S rows of the padded buffer; rest
            // stay zero. Convert f32 → f16 as we go.
            for (i, &v) in src.iter().enumerate() {
                in_fp16[i] = f16::from_f32(v).to_bits();
            }
            for slot in &mut in_fp16[s * h..] {
                *slot = 0;
            }

            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            // SAFETY: pointers are valid, sizes match, shape is rank-3.
            let ok = unsafe {
                ane_mlmodel_predict_fp16(
                    self.handle,
                    in_name.as_ptr(),
                    in_fp16.as_ptr(),
                    in_fp16.len(),
                    in_shape.as_ptr(),
                    3,
                    out_name.as_ptr(),
                    out_fp16.as_mut_ptr(),
                    out_fp16.len(),
                    &mut err_ptr,
                )
            };
            if !ok {
                let msg = unsafe { take_error(err_ptr) };
                return Err(Exception::custom(format!(
                    "ane_mlmodel_predict_fp16 failed: {msg}"
                )));
            }

            // Slice [1, pad, vocab] → [s, vocab] and convert to f32.
            let out_dst = &mut out_all[bi * s * vocab..(bi + 1) * s * vocab];
            for (i, slot) in out_dst.iter_mut().enumerate() {
                *slot = f16::from_bits(out_fp16[i]).to_f32();
            }
        }

        let out = Array::from_slice(&out_all, &[b as i32, s as i32, vocab as i32]);
        if x.dtype() == Dtype::Float32 {
            Ok(out)
        } else {
            out.as_dtype(x.dtype())
        }
    }
}

// ---------------------------------------------------------------------------
// Generic `.mlpackage` bridge — sibling to the raw-MIL `AneKernel` path.
//
// Role: load a pre-compiled `.mlmodelc` that may contain int8 weights (via
// `constexpr_affine_dequantize`) and dispatch through the *public* `MLModel`
// API. The existing raw-MIL path (`AneKernel::compile_multi_weights`) stays
// fp16-only; this one is the int8-capable entry point.
//
// Scope is intentionally narrow: fp16 activation in, fp16 activation out,
// arbitrary rank, caller-supplied input/output feature names. The offline
// `.mlpackage` builder (`benchmarks/ane_int8_mlpackage_probe/`) owns shape
// and quant details; this kernel just loads, verifies dispatch, and runs.
// ---------------------------------------------------------------------------

/// Verify that a compiled `.mlmodelc` will prefer ANE for at least one op
/// under `MLComputeUnitsCPUAndNeuralEngine`. Returns `(on_ane, per_op_report)`.
///
/// Use this *before* any parity test per the AB7 trap: toy shapes silently
/// prefer CPU, which would let the int8 path pass parity while running on
/// the same core as the fp32 reference. A production-shape failure here
/// means ANE is not engaged at all, regardless of correctness.
pub fn verify_ane_dispatch(mlmodelc_path: &str) -> Result<(bool, String), String> {
    let path = CString::new(mlmodelc_path)
        .map_err(|e| format!("mlmodelc_path contains NUL: {e}"))?;
    let mut report_ptr: *mut c_char = std::ptr::null_mut();
    let mut err_ptr: *mut c_char = std::ptr::null_mut();
    // SAFETY: pointers valid; bridge fills report/error on success/failure.
    let on_ane = unsafe {
        ane_mlmodel_verify_ane_dispatch(path.as_ptr(), &mut report_ptr, &mut err_ptr)
    };
    // Always take both strings so neither leaks.
    let report = unsafe { take_error(report_ptr) };
    let err_msg = unsafe { take_error(err_ptr) };
    if !err_msg.is_empty() {
        return Err(err_msg);
    }
    Ok((on_ane, report))
}

/// Shape-agnostic kernel over a compiled `.mlpackage` → `.mlmodelc`.
pub struct AneMlPackageKernel {
    handle: *mut c_void,
    input_name: CString,
    output_name: CString,
    /// Rank-N input shape baked into the compiled model (for bounds checks).
    pub input_shape: Vec<i64>,
    /// Rank-M output shape baked into the compiled model.
    pub output_shape: Vec<i64>,
    pub mlmodelc_path: String,
}

// SAFETY: same rationale as `AneLmHeadLut6Kernel`. `MLModel` is documented
// thread-safe and the handle only wraps a retained CFTypeRef.
#[allow(unsafe_code)]
unsafe impl Send for AneMlPackageKernel {}
#[allow(unsafe_code)]
unsafe impl Sync for AneMlPackageKernel {}

impl std::fmt::Debug for AneMlPackageKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AneMlPackageKernel")
            .field("mlmodelc_path", &self.mlmodelc_path)
            .field("input_shape", &self.input_shape)
            .field("output_shape", &self.output_shape)
            .finish()
    }
}

impl Drop for AneMlPackageKernel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle from `ane_mlmodel_load`; freed once here.
            unsafe { ane_mlmodel_free(self.handle); }
            self.handle = std::ptr::null_mut();
        }
    }
}

impl AneMlPackageKernel {
    /// Load a compiled `.mlmodelc` and configure it for CPU+ANE dispatch.
    ///
    /// `input_shape` / `output_shape` are the shapes the caller will pass to
    /// `predict_fp16`. They must match the shapes baked into the `.mlpackage`
    /// at build time (coremltools `mb.TensorSpec`).
    pub fn load(
        mlmodelc_path: &str,
        input_name: &str,
        output_name: &str,
        input_shape: Vec<i64>,
        output_shape: Vec<i64>,
    ) -> Result<Self, String> {
        if input_shape.is_empty() || output_shape.is_empty() {
            return Err(format!(
                "empty shape: input={input_shape:?} output={output_shape:?}"
            ));
        }
        let path = CString::new(mlmodelc_path)
            .map_err(|e| format!("mlmodelc_path contains NUL: {e}"))?;
        let in_name = CString::new(input_name)
            .map_err(|e| format!("input_name contains NUL: {e}"))?;
        let out_name = CString::new(output_name)
            .map_err(|e| format!("output_name contains NUL: {e}"))?;
        let mut err_ptr: *mut c_char = std::ptr::null_mut();
        // SAFETY: path is a valid C string; err_ptr receives malloc'd msg.
        let handle = unsafe { ane_mlmodel_load(path.as_ptr(), &mut err_ptr) };
        if handle.is_null() {
            let msg = unsafe { take_error(err_ptr) };
            return Err(if msg.is_empty() {
                format!("ane_mlmodel_load({mlmodelc_path}) returned NULL")
            } else {
                msg
            });
        }
        Ok(Self {
            handle,
            input_name: in_name,
            output_name: out_name,
            input_shape,
            output_shape,
            mlmodelc_path: mlmodelc_path.to_owned(),
        })
    }

    /// Convenience: verify the compiled `.mlmodelc` backing this kernel
    /// will prefer ANE for at least one op. Equivalent to
    /// `verify_ane_dispatch(self.mlmodelc_path)` — exposed as a method so
    /// callers don't need to track the path separately.
    pub fn verify_ane_dispatch(&self) -> Result<(bool, String), String> {
        verify_ane_dispatch(&self.mlmodelc_path)
    }

    fn prod(shape: &[i64]) -> usize {
        shape.iter().map(|&d| d as usize).product()
    }

    /// Raw fp16 prediction. `x_fp16.len()` must equal `prod(input_shape)`,
    /// `y_fp16.len()` must equal `prod(output_shape)`. Both are row-major.
    pub fn predict_fp16(&self, x_fp16: &[u16], y_fp16: &mut [u16]) -> Result<(), String> {
        let want_in = Self::prod(&self.input_shape);
        let want_out = Self::prod(&self.output_shape);
        if x_fp16.len() != want_in {
            return Err(format!(
                "input count {} != prod(input_shape) {}",
                x_fp16.len(),
                want_in
            ));
        }
        if y_fp16.len() != want_out {
            return Err(format!(
                "output count {} != prod(output_shape) {}",
                y_fp16.len(),
                want_out
            ));
        }
        let mut err_ptr: *mut c_char = std::ptr::null_mut();
        // SAFETY: handle alive, lengths validated, shape points to rank-N.
        let ok = unsafe {
            ane_mlmodel_predict_fp16(
                self.handle,
                self.input_name.as_ptr(),
                x_fp16.as_ptr(),
                x_fp16.len(),
                self.input_shape.as_ptr(),
                self.input_shape.len() as c_int,
                self.output_name.as_ptr(),
                y_fp16.as_mut_ptr(),
                y_fp16.len(),
                &mut err_ptr,
            )
        };
        if !ok {
            let msg = unsafe { take_error(err_ptr) };
            return Err(if msg.is_empty() {
                "ane_mlmodel_predict_fp16 failed (no message)".to_owned()
            } else {
                msg
            });
        }
        Ok(())
    }

    /// Multi-output variant. `x_fp16` still has shape `input_shape`; the
    /// caller supplies `(output_name, output_buffer)` pairs for every MIL
    /// output to pull. One underlying `predictionFromFeatures:` call.
    ///
    /// Used by the int8 fusion probe: if ANE amortizes dispatch cost across
    /// outputs sharing one MIL program, `min_time(fused_N) < N *
    /// min_time(single)` by roughly `(N-1) * per_dispatch_overhead`.
    pub fn predict_fp16_multi(
        &self,
        x_fp16: &[u16],
        outputs: &mut [(&str, &mut [u16])],
    ) -> Result<(), String> {
        let want_in = Self::prod(&self.input_shape);
        if x_fp16.len() != want_in {
            return Err(format!(
                "input count {} != prod(input_shape) {}",
                x_fp16.len(),
                want_in,
            ));
        }
        if outputs.is_empty() {
            return Err("predict_fp16_multi requires ≥1 output".to_owned());
        }

        let names_cstr: Vec<CString> = outputs
            .iter()
            .map(|(n, _)| {
                CString::new(*n).map_err(|e| format!("output name contains NUL: {e}"))
            })
            .collect::<Result<_, _>>()?;
        let name_ptrs: Vec<*const c_char> = names_cstr.iter().map(|c| c.as_ptr()).collect();
        let counts: Vec<usize> = outputs.iter().map(|(_, b)| b.len()).collect();
        let buf_ptrs: Vec<*mut u16> =
            outputs.iter_mut().map(|(_, b)| b.as_mut_ptr()).collect();

        let mut err_ptr: *mut c_char = std::ptr::null_mut();
        // SAFETY: handle alive, x length validated, every output buffer
        // comes from a mutable &mut [u16] so its pointer is writable for
        // `counts[i]` u16s. Parallel arrays are all length n_outputs.
        let ok = unsafe {
            ane_mlmodel_predict_fp16_multi(
                self.handle,
                self.input_name.as_ptr(),
                x_fp16.as_ptr(),
                x_fp16.len(),
                self.input_shape.as_ptr(),
                self.input_shape.len() as c_int,
                name_ptrs.as_ptr(),
                buf_ptrs.as_ptr(),
                counts.as_ptr(),
                outputs.len() as c_int,
                &mut err_ptr,
            )
        };
        if !ok {
            let msg = unsafe { take_error(err_ptr) };
            return Err(if msg.is_empty() {
                "ane_mlmodel_predict_fp16_multi failed (no message)".to_owned()
            } else {
                msg
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests — require Python + coremltools, hence `#[ignore]`.
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]
mod tests {
    use super::*;
    use mlx_rs::{ops::matmul, random};

    /// End-to-end parity: random weights, compile via palettize_lm_head.py
    /// + coremlcompiler, dispatch, compare to MLX matmul.
    ///
    /// Tolerance is far looser than dense fp16 (~0.05) — palettization at
    /// 6 bits / group_size=16 is intentionally lossy. 0.5 absolute is the
    /// load-bearing threshold: if the output is further from matmul than
    /// that, the pipeline is broken, not just quantized.
    #[test]
    #[ignore = "requires Python venv with coremltools; run explicitly"]
    fn lut6_synthetic_parity() {
        use std::process::Command;
        use std::fs;

        let in_dim: usize = 512;
        let out_dim: usize = 1024;
        let s: usize = 15;
        let pad: usize = 16;

        let w = random::uniform::<f32, f32>(-0.05, 0.05,
            &[out_dim as i32, in_dim as i32], None).expect("random w");
        w.eval().unwrap();
        let w_vec = w.as_slice::<f32>().to_vec();

        let x = random::uniform::<f32, f32>(-1.0, 1.0,
            &[1, s as i32, in_dim as i32], None).expect("random x");
        x.eval().unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let weights_bin = tmp.path().join("w.bin");
        let out_dir = tmp.path().join("out");
        fs::create_dir_all(&out_dir).unwrap();

        // Write f32 LE bytes.
        let mut bytes = Vec::with_capacity(w_vec.len() * 4);
        for v in &w_vec { bytes.extend_from_slice(&v.to_le_bytes()); }
        fs::write(&weights_bin, &bytes).unwrap();

        let script = env!("CARGO_MANIFEST_DIR").to_string()
            + "/scripts/palettize_lm_head.py";
        let out = Command::new("python3")
            .arg(&script)
            .arg("--weights-bin").arg(&weights_bin)
            .arg("--vocab").arg(out_dim.to_string())
            .arg("--hidden").arg(in_dim.to_string())
            .arg("--seq-len").arg(pad.to_string())
            .arg("--out-dir").arg(&out_dir)
            .output().expect("spawn python3");
        assert!(out.status.success(),
            "palettize failed: stderr={}", String::from_utf8_lossy(&out.stderr));

        let response: serde_json::Value =
            serde_json::from_slice(&out.stdout).expect("json");
        let mlmodelc = response["mlmodelc"].as_str().unwrap();

        let kernel = AneLmHeadLut6Kernel::load(mlmodelc, out_dim, in_dim, pad)
            .expect("load");

        let y_ref = matmul(&x, &w.t()).unwrap();
        y_ref.eval().unwrap();
        let y_ref_vec: Vec<f32> = y_ref.as_slice::<f32>().to_vec();

        let y_ane = kernel.dispatch(&x).expect("dispatch");
        y_ane.eval().unwrap();
        let y_ane_vec: Vec<f32> = y_ane.as_slice::<f32>().to_vec();

        let mut max_diff = 0.0f32;
        for (a, b) in y_ane_vec.iter().zip(y_ref_vec.iter()) {
            let d = (a - b).abs();
            if d > max_diff { max_diff = d; }
        }
        // 6-bit palettization at group_size=16 is lossy — see module docstring.
        assert!(max_diff < 0.5,
            "LUT6 parity: max_diff={max_diff} (budget 0.5)");
    }

    /// Second compile hits the filesystem cache via the caller
    /// (`compile_proj_lut6` in `qwen3_next_ane.rs`). This test lives at the
    /// caller level because the cache logic is there, not in `load`.
    #[test]
    #[ignore = "cache is owned by compile_proj_lut6; covered in that layer"]
    fn lut6_cache_hit() {
        // Intentionally empty: see `qwen3_next_ane::compile_proj_lut6`
        // tests once that landing is wired. The Rust wrapper here just
        // loads whatever path it's handed — caching is orthogonal.
    }

    /// AB6/AB7 smoke test: given the int8-conv1x1 `.mlmodelc` produced by
    /// `benchmarks/ane_int8_mlpackage_probe/build_realistic.py`, the
    /// verifier must report ANE for the conv op. Falsifies regressions in
    /// the MLComputePlan bridge itself (separate from any parity concern).
    ///
    /// Path is env-var driven so CI on a freshly-cloned tree doesn't fail
    /// just because the probe hasn't been run yet.
    #[test]
    #[ignore = "requires pre-built .mlmodelc; set HIGGS_INT8_O_PROJ_MLMODELC"]
    fn verify_o_proj_4b_prefers_ane() {
        let path = std::env::var("HIGGS_INT8_O_PROJ_MLMODELC").expect(
            "set HIGGS_INT8_O_PROJ_MLMODELC=<path to int8_o_proj_4b.mlmodelc>",
        );
        let (on_ane, report) =
            verify_ane_dispatch(&path).expect("verify_ane_dispatch");
        println!("{report}");
        assert!(
            on_ane,
            "conv op did NOT prefer ANE at 3072x3072 — AB7 regression?\n{report}"
        );
        assert!(
            report.contains("ANE"),
            "report missing 'ANE' tag: {report}"
        );
    }

    /// End-to-end int8 parity at DFlash-4B `o_proj` shape (3072×3072, seq=16).
    ///
    /// Pipeline:
    ///   1. Random fp32 W [out, in] + fp32 x [1, in, 1, seq] (conv1x1 layout).
    ///   2. `quantize_int8_proj.py` (via the 3.13 sidecar) → `.mlmodelc`.
    ///   3. `AneMlPackageKernel::load` + `verify_ane_dispatch` (expect ANE).
    ///   4. Dispatch at fp16, convert output back to fp32.
    ///   5. Reference: fp32 matmul W @ x_flat on CPU via MLX.
    ///   6. Assert max|ane - ref| ≤ 0.08 (per-layer GDN budget; int8 adds
    ///      quantization noise above the fp16 path's 0.034 baseline).
    ///
    /// Env vars (required):
    ///   HIGGS_CORETOOLS_PYTHON — absolute path to a Python 3.13 interpreter
    ///                            with working coremltools 9.0
    ///                            (project .venv 3.14 will NOT work; see T1).
    #[test]
    #[ignore = "requires coremltools-3.13 sidecar; set HIGGS_CORETOOLS_PYTHON"]
    fn dflash_ane_o_proj_int8_parity() {
        use mlx_rs::{ops::matmul, random};
        use std::fs;
        use std::process::Command;

        let py = std::env::var("HIGGS_CORETOOLS_PYTHON")
            .expect("set HIGGS_CORETOOLS_PYTHON=<3.13 sidecar python>");
        let in_dim: usize = 3072;
        let out_dim: usize = 3072;
        let seq: usize = 16;

        // --- Random fp32 weights + activation ---
        // Uniform over a tight range keeps scale small so int8 quant
        // resolution is usable; larger ranges would blow the 0.08 budget
        // by the scale = max(|w|)/127 definition.
        let w = random::uniform::<f32, f32>(-0.05, 0.05,
            &[out_dim as i32, in_dim as i32], None).expect("random W");
        w.eval().unwrap();
        let w_vec: Vec<f32> = w.as_slice::<f32>().to_vec();

        // x is [1, seq, in_dim] in MLX convention; conv1x1 wants [1, in, 1, seq].
        let x = random::uniform::<f32, f32>(-1.0, 1.0,
            &[1, seq as i32, in_dim as i32], None).expect("random x");
        x.eval().unwrap();
        let x_vec: Vec<f32> = x.as_slice::<f32>().to_vec();

        // --- Write weight bin, invoke Python builder ---
        let tmp = tempfile::tempdir().unwrap();
        let weights_bin = tmp.path().join("w.bin");
        let out_dir = tmp.path().join("out");
        fs::create_dir_all(&out_dir).unwrap();

        let mut bytes = Vec::with_capacity(w_vec.len() * 4);
        for v in &w_vec { bytes.extend_from_slice(&v.to_le_bytes()); }
        fs::write(&weights_bin, &bytes).unwrap();

        let script = env!("CARGO_MANIFEST_DIR").to_string()
            + "/scripts/quantize_int8_proj.py";
        let out = Command::new(&py)
            .arg(&script)
            .arg("--weights-bin").arg(&weights_bin)
            .arg("--out-features").arg(out_dim.to_string())
            .arg("--in-features").arg(in_dim.to_string())
            .arg("--seq-len").arg(seq.to_string())
            .arg("--out-dir").arg(&out_dir)
            .output().expect("spawn python");
        assert!(out.status.success(),
            "quantize_int8_proj failed\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr));

        let resp: serde_json::Value =
            serde_json::from_slice(&out.stdout).expect("json response");
        let mlmodelc = resp["mlmodelc"].as_str().unwrap();
        let in_name = resp["input_name"].as_str().unwrap();
        let out_name = resp["output_name"].as_str().unwrap();
        let scale = resp["scale"].as_f64().unwrap();
        let max_abs_w = resp["max_abs_w"].as_f64().unwrap();
        eprintln!(
            "built mlmodelc={mlmodelc}  scale={scale:.3e}  max|w|={max_abs_w:.3e}"
        );

        // --- Load the kernel and verify ANE dispatch ---
        let input_shape: Vec<i64> = vec![1, in_dim as i64, 1, seq as i64];
        let output_shape: Vec<i64> = vec![1, out_dim as i64, 1, seq as i64];
        let kernel = AneMlPackageKernel::load(
            mlmodelc, in_name, out_name, input_shape, output_shape,
        ).expect("load mlpackage kernel");

        let (on_ane, report) = kernel.verify_ane_dispatch()
            .expect("verify_ane_dispatch");
        eprintln!("{report}");
        assert!(on_ane,
            "3072x3072 seq=16 should dispatch to ANE (AB7) — got:\n{report}");

        // --- Pack x into conv1x1 layout [1, in, 1, seq] as fp16 ---
        let pin = in_dim * seq;
        let pout = out_dim * seq;
        let mut x_fp16 = vec![0u16; pin];
        // x is [1, seq, in] row-major; conv1x1 wants [1, in, 1, seq] so we
        // transpose: dst[ci*seq + t] = src[t*in + ci].
        for t in 0..seq {
            for ci in 0..in_dim {
                let src = x_vec[t * in_dim + ci];
                x_fp16[ci * seq + t] = f16::from_f32(src).to_bits();
            }
        }
        let mut y_fp16 = vec![0u16; pout];
        kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");

        // --- Unpack [1, out, 1, seq] → [1, seq, out] fp32 ---
        let mut y_ane = vec![0.0f32; seq * out_dim];
        for co in 0..out_dim {
            for t in 0..seq {
                y_ane[t * out_dim + co] =
                    f16::from_bits(y_fp16[co * seq + t]).to_f32();
            }
        }

        // --- Reference: fp32 matmul W @ x_flat on CPU (MLX) ---
        // y_ref = x [1, seq, in] @ W.T [in, out] → [1, seq, out]
        let y_ref = matmul(&x, &w.t()).expect("matmul");
        y_ref.eval().unwrap();
        let y_ref_vec: Vec<f32> = y_ref.as_slice::<f32>().to_vec();

        let mut max_diff = 0.0f32;
        let mut max_abs_ref = 0.0f32;
        for (a, b) in y_ane.iter().zip(y_ref_vec.iter()) {
            let d = (a - b).abs();
            if d > max_diff { max_diff = d; }
            let ab = b.abs();
            if ab > max_abs_ref { max_abs_ref = ab; }
        }
        eprintln!(
            "int8 parity: max_diff={max_diff:.6}  max|ref|={max_abs_ref:.3}"
        );
        // Budget matches the plan's acceptance criterion 1.5:
        //   int8 adds ~quant_noise above fp16's ~0.034 baseline; ≤ 0.08 total.
        assert!(max_diff <= 0.08,
            "int8 parity max_diff={max_diff} > 0.08 (budget)");
    }

    /// Plan Step 1.6 gate: time `predict_fp16` on the int8 path at
    /// DFlash-4B `o_proj` shape across ctx ∈ {16, 64, 256} and report
    /// effective weight bandwidth. Purpose is not pass/fail — it's to
    /// confirm ANE is actually engaged (wall clock must reflect ~9 MB
    /// weight fetch, not 18 MB, otherwise the int8 payoff is missing).
    ///
    /// This test is informational: it always succeeds after running as
    /// long as dispatch is on ANE. Numbers must be compared against the
    /// fp16 baseline from `bench_9b_blocksize_sweep.sh` by the caller.
    #[test]
    #[ignore = "benchmark; set HIGGS_CORETOOLS_PYTHON"]
    fn dflash_ane_o_proj_int8_latency() {
        use std::fs;
        use std::process::Command;
        use std::time::Instant;

        let py = std::env::var("HIGGS_CORETOOLS_PYTHON")
            .expect("set HIGGS_CORETOOLS_PYTHON=<3.13 sidecar python>");
        let in_dim: usize = 3072;
        let out_dim: usize = 3072;
        // int8 weight bytes; fp16 baseline would be 2×.
        let w_bytes: f64 = (in_dim * out_dim) as f64;
        let w_fp16_bytes: f64 = w_bytes * 2.0;

        // Build once per seq — each compile is ~5 s.
        let seqs: Vec<usize> = vec![16, 64, 256];
        let iters: usize = 50;
        let warmup: usize = 5;

        // Random weights shared across seqs; saves rebuild cost isn't
        // available (weight-hash would differ) but at least the Rust work
        // is shared.
        let seed_w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| ((i as f32 * 0.12345).sin() * 0.05))
            .collect();
        let tmp = tempfile::tempdir().unwrap();
        let weights_bin = tmp.path().join("w.bin");
        let mut bytes = Vec::with_capacity(seed_w.len() * 4);
        for v in &seed_w { bytes.extend_from_slice(&v.to_le_bytes()); }
        fs::write(&weights_bin, &bytes).unwrap();

        eprintln!("| seq | min_ms | med_ms | GB/s (int8 bw) | GB/s (vs fp16 baseline) |");
        eprintln!("|-----|--------|--------|----------------|-------------------------|");

        for &seq in &seqs {
            let out_dir = tmp.path().join(format!("out_{seq}"));
            fs::create_dir_all(&out_dir).unwrap();

            let script = env!("CARGO_MANIFEST_DIR").to_string()
                + "/scripts/quantize_int8_proj.py";
            let out = Command::new(&py)
                .arg(&script)
                .arg("--weights-bin").arg(&weights_bin)
                .arg("--out-features").arg(out_dim.to_string())
                .arg("--in-features").arg(in_dim.to_string())
                .arg("--seq-len").arg(seq.to_string())
                .arg("--out-dir").arg(&out_dir)
                .output().expect("spawn python");
            assert!(out.status.success(),
                "quantize_int8_proj failed\nstderr: {}",
                String::from_utf8_lossy(&out.stderr));
            let resp: serde_json::Value =
                serde_json::from_slice(&out.stdout).expect("json");
            let mlmodelc = resp["mlmodelc"].as_str().unwrap();
            let in_name = resp["input_name"].as_str().unwrap();
            let out_name = resp["output_name"].as_str().unwrap();

            let input_shape: Vec<i64> = vec![1, in_dim as i64, 1, seq as i64];
            let output_shape: Vec<i64> = vec![1, out_dim as i64, 1, seq as i64];
            let kernel = AneMlPackageKernel::load(
                mlmodelc, in_name, out_name, input_shape, output_shape,
            ).expect("load");

            let (on_ane, _report) = kernel.verify_ane_dispatch().expect("verify");
            assert!(on_ane, "seq={seq} did not dispatch to ANE");

            // Fixed input; content doesn't matter for timing.
            let x_fp16 = vec![f16::from_f32(0.1).to_bits(); in_dim * seq];
            let mut y_fp16 = vec![0u16; out_dim * seq];

            // Warmup.
            for _ in 0..warmup {
                kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");
            }

            let mut samples_us: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");
                samples_us.push(t0.elapsed().as_micros());
            }
            samples_us.sort_unstable();
            let min_ms = samples_us[0] as f64 / 1000.0;
            let med_ms = samples_us[iters / 2] as f64 / 1000.0;

            // Effective weight bandwidth: fetch the int8 weights once per
            // forward. GB/s = w_bytes / (min_ms * 1e-3) / 1e9.
            let int8_gbs = w_bytes / (min_ms * 1e-3) / 1e9;
            let vs_fp16_gbs = w_fp16_bytes / (min_ms * 1e-3) / 1e9;
            eprintln!(
                "| {seq:>3} | {min_ms:>6.3} | {med_ms:>6.3} | {int8_gbs:>14.2} | {vs_fp16_gbs:>23.2} |",
            );
        }
    }

    /// Plan Step 1 (int8-e2e-decode handoff): bound the per-dispatch fixed
    /// cost of `AneMlPackageKernel::predict_fp16`. The DFlash-4B drafter
    /// needs 35 dispatches per decode step; a ≥100 µs fixed cost means
    /// fanout can't beat the fused raw-MIL fp16 path.
    ///
    /// Two probes at seq=32 (aligned):
    ///   tiny  64×64    — falls to CPU; measures FFI + Cocoa floor alone.
    ///   ane   3072×3072 — engages ANE (per handoff table); min - bw-bound
    ///                      compute floor isolates the ANE-dispatch delta.
    ///
    /// Decision thresholds (ANE overhead, handoff):
    ///   ≤ 50 µs  → 35 dispatches ≈ 1.75 ms. Fanout viable.
    ///   100–200 µs → 35 dispatches ≈ 3.5–7 ms. Wash; need fusion.
    ///   ≥ 300 µs → fanout cannot beat raw-MIL fp16.
    ///
    /// Informational: always passes. Reports both numbers so the handoff
    /// has measured evidence for which branch of the plan to take.
    #[test]
    #[ignore = "benchmark; set HIGGS_CORETOOLS_PYTHON"]
    fn dflash_ane_dispatch_overhead_probe() {
        use std::fs;
        use std::process::Command;
        use std::time::Instant;

        let py = std::env::var("HIGGS_CORETOOLS_PYTHON")
            .expect("set HIGGS_CORETOOLS_PYTHON=<3.13 sidecar python>");
        let seq: usize = 32;
        // ANE peak DRAM bandwidth ceiling used for the compute floor. This
        // is intentionally optimistic — a *lower* compute floor means a
        // *larger* implied overhead, so errs toward the safer (more
        // pessimistic) fanout verdict.
        let ane_peak_gbs: f64 = 60.0;

        // (label, out_features, in_features, iters, warmup)
        let shapes = [
            ("tiny(cpu-fallback)",   64usize,   64usize, 1000usize, 50usize),
            ("ane(3072x3072)",     3072usize, 3072usize,  200usize, 20usize),
        ];

        eprintln!("| shape              | on_ane | min_us | p50_us | p99_us | compute_floor_us | overhead_us | 35× ms |");
        eprintln!("|--------------------|--------|--------|--------|--------|------------------|-------------|--------|");

        let tmp = tempfile::tempdir().unwrap();
        let mut ane_overhead_us: Option<f64> = None;
        for (label, out_dim, in_dim, iters, warmup) in shapes {
            let seed_w: Vec<f32> = (0..in_dim * out_dim)
                .map(|i| ((i as f32 * 0.12345).sin() * 0.05))
                .collect();
            let weights_bin = tmp.path().join(format!("{label}.bin"));
            let mut bytes = Vec::with_capacity(seed_w.len() * 4);
            for v in &seed_w { bytes.extend_from_slice(&v.to_le_bytes()); }
            fs::write(&weights_bin, &bytes).unwrap();

            let out_dir = tmp.path().join(format!("out_{label}"));
            fs::create_dir_all(&out_dir).unwrap();
            let script = env!("CARGO_MANIFEST_DIR").to_string()
                + "/scripts/quantize_int8_proj.py";
            let out = Command::new(&py)
                .arg(&script)
                .arg("--weights-bin").arg(&weights_bin)
                .arg("--out-features").arg(out_dim.to_string())
                .arg("--in-features").arg(in_dim.to_string())
                .arg("--seq-len").arg(seq.to_string())
                .arg("--out-dir").arg(&out_dir)
                .output().expect("spawn python");
            assert!(out.status.success(),
                "{label}: quantize_int8_proj failed\nstderr: {}",
                String::from_utf8_lossy(&out.stderr));
            let resp: serde_json::Value =
                serde_json::from_slice(&out.stdout).expect("json");
            let mlmodelc = resp["mlmodelc"].as_str().unwrap();
            let in_name = resp["input_name"].as_str().unwrap();
            let out_name = resp["output_name"].as_str().unwrap();

            let input_shape: Vec<i64> = vec![1, in_dim as i64, 1, seq as i64];
            let output_shape: Vec<i64> = vec![1, out_dim as i64, 1, seq as i64];
            let kernel = AneMlPackageKernel::load(
                mlmodelc, in_name, out_name, input_shape, output_shape,
            ).expect("load");
            let (on_ane, _rep) = kernel.verify_ane_dispatch().expect("verify");

            let x_fp16 = vec![f16::from_f32(0.1).to_bits(); in_dim * seq];
            let mut y_fp16 = vec![0u16; out_dim * seq];
            for _ in 0..warmup {
                kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");
            }

            let mut samples_us: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");
                samples_us.push(t0.elapsed().as_micros());
            }
            samples_us.sort_unstable();
            let min_us = samples_us[0] as f64;
            let p50_us = samples_us[iters / 2] as f64;
            let p99_us = samples_us[(iters * 99) / 100] as f64;

            let w_bytes = (in_dim * out_dim) as f64; // int8
            let act_bytes = (in_dim * seq + out_dim * seq) as f64 * 2.0; // fp16
            let compute_floor_us = (w_bytes + act_bytes) / (ane_peak_gbs * 1.0e9) * 1.0e6;
            let overhead_us = (min_us - compute_floor_us).max(0.0);
            let projected_35_ms = overhead_us * 35.0 / 1000.0;

            eprintln!(
                "| {label:<18} | {on_ane:<6} | {min_us:>6.1} | {p50_us:>6.1} | {p99_us:>6.1} | {compute_floor_us:>16.2} | {overhead_us:>11.1} | {projected_35_ms:>6.2} |",
            );
            if on_ane { ane_overhead_us = Some(overhead_us); }
        }

        eprintln!();
        eprintln!("decision thresholds (ANE overhead, handoff int8-e2e-decode):");
        eprintln!("  ≤  50 µs → fanout viable (35× ≈ ≤1.75 ms)");
        eprintln!("  100-200 µs → wash; need fusion to beat raw-MIL fp16");
        eprintln!("  ≥ 300 µs → fusion mandatory");
        if let Some(o) = ane_overhead_us {
            let verdict = if o <= 50.0 { "FANOUT-VIABLE" }
                else if o < 250.0 { "WASH/FUSION-NEEDED" }
                else { "FUSION-MANDATORY" };
            eprintln!("ANE-engaged overhead ≈ {o:.1} µs → {verdict}");
        } else {
            eprintln!("WARN: no shape reported on_ane=true; verdict deferred.");
        }
    }

    /// Plan Step 3 (int8-e2e-decode handoff): does ANE amortize dispatch
    /// overhead when multiple conv1x1 ops share one MIL program input?
    ///
    /// Setup at DFlash-4B QKV shapes, seq=32 (aligned):
    ///   Q: in=3072 out=3072
    ///   K: in=3072 out=1024
    ///   V: in=3072 out=1024
    ///
    /// Two paths measured:
    ///   individual: 3 separate mlpackages + 3 sequential `predict_fp16` calls
    ///   fused:      1 fused mlpackage + 1 `predict_fp16_multi` call
    ///
    /// Decision:
    ///   fused_overhead ≈ single_overhead  → full amortization; fusion ships
    ///   fused_overhead ≈ 3× single_overhead → ANE serializes; kill fusion
    ///   anywhere between → partial amortization; quantify the savings vs
    ///   the 35-dispatch DFlash step.
    ///
    /// Informational test; always passes if both paths engage ANE.
    #[test]
    #[ignore = "benchmark; set HIGGS_CORETOOLS_PYTHON"]
    fn dflash_ane_fusion_probe() {
        use std::fs;
        use std::process::Command;
        use std::time::Instant;

        let py = std::env::var("HIGGS_CORETOOLS_PYTHON")
            .expect("set HIGGS_CORETOOLS_PYTHON=<3.13 sidecar python>");
        let in_dim: usize = 3072;
        let seq: usize = 32;
        let ane_peak_gbs: f64 = 60.0;
        let iters: usize = 200;
        let warmup: usize = 20;

        // (label, out_features) — DFlash-4B GQA attention projections.
        let projs: [(&str, usize); 3] = [("q", 3072), ("k", 1024), ("v", 1024)];

        // Reproducible seeds so the individual and fused builds see the
        // same quantization scale per-projection (parity isn't the goal,
        // but matching compute is).
        let tmp = tempfile::tempdir().unwrap();
        let mut wbins: Vec<std::path::PathBuf> = Vec::new();
        for (label, out_dim) in projs {
            let seed: Vec<f32> = (0..in_dim * out_dim)
                .map(|i| ((i as f32 * 0.1234).sin() * 0.05))
                .collect();
            let wb = tmp.path().join(format!("{label}.bin"));
            let mut bytes = Vec::with_capacity(seed.len() * 4);
            for v in &seed { bytes.extend_from_slice(&v.to_le_bytes()); }
            fs::write(&wb, &bytes).unwrap();
            wbins.push(wb);
        }

        // --- Individual path: 3 separate mlpackages ---
        let single_script = env!("CARGO_MANIFEST_DIR").to_string()
            + "/scripts/quantize_int8_proj.py";
        let mut singles: Vec<AneMlPackageKernel> = Vec::new();
        let mut single_outs: Vec<(usize, usize)> = Vec::new(); // (out, seq)
        for (i, (label, out_dim)) in projs.iter().enumerate() {
            let out_dir = tmp.path().join(format!("ind_{label}"));
            fs::create_dir_all(&out_dir).unwrap();
            let o = Command::new(&py)
                .arg(&single_script)
                .arg("--weights-bin").arg(&wbins[i])
                .arg("--out-features").arg(out_dim.to_string())
                .arg("--in-features").arg(in_dim.to_string())
                .arg("--seq-len").arg(seq.to_string())
                .arg("--out-dir").arg(&out_dir)
                .output().expect("spawn");
            assert!(o.status.success(), "{label}: single build failed\nstderr: {}",
                String::from_utf8_lossy(&o.stderr));
            let resp: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
            let mlc = resp["mlmodelc"].as_str().unwrap();
            let iname = resp["input_name"].as_str().unwrap();
            let oname = resp["output_name"].as_str().unwrap();
            let input_shape: Vec<i64> = vec![1, in_dim as i64, 1, seq as i64];
            let output_shape: Vec<i64> = vec![1, *out_dim as i64, 1, seq as i64];
            let k = AneMlPackageKernel::load(mlc, iname, oname, input_shape, output_shape)
                .expect("load single");
            let (on_ane, _) = k.verify_ane_dispatch().unwrap();
            assert!(on_ane, "individual {label} not on ANE");
            singles.push(k);
            single_outs.push((*out_dim, seq));
        }

        // --- Fused path: 1 mlpackage, 3 named outputs ---
        let fused_script = env!("CARGO_MANIFEST_DIR").to_string()
            + "/scripts/quantize_int8_fused.py";
        let fused_out_dir = tmp.path().join("fused");
        fs::create_dir_all(&fused_out_dir).unwrap();
        let mut fused_cmd = Command::new(&py);
        fused_cmd.arg(&fused_script)
            .arg("--in-features").arg(in_dim.to_string())
            .arg("--seq-len").arg(seq.to_string())
            .arg("--out-dir").arg(&fused_out_dir);
        for (i, (label, out_dim)) in projs.iter().enumerate() {
            fused_cmd.arg("--proj").arg(format!(
                "{label}:{out_dim}:{}", wbins[i].display()
            ));
        }
        let fo = fused_cmd.output().expect("spawn fused");
        assert!(fo.status.success(), "fused build failed\nstderr: {}",
            String::from_utf8_lossy(&fo.stderr));
        let fresp: serde_json::Value = serde_json::from_slice(&fo.stdout).unwrap();
        let fmlc = fresp["mlmodelc"].as_str().unwrap();
        let finput = fresp["input_name"].as_str().unwrap().to_owned();
        // Output names from the script are "y_<label>" — keep that ABI.
        let fused_output_names: Vec<String> = projs.iter()
            .map(|(label, _)| format!("y_{label}"))
            .collect();

        // Fused kernel input_shape matches the single one. output_shape is
        // unused for the multi path (we pass names + counts explicitly), so
        // set it to the Q output shape to satisfy the load invariants.
        let fused_input_shape: Vec<i64> = vec![1, in_dim as i64, 1, seq as i64];
        let fused_output_shape: Vec<i64> = vec![1, 3072, 1, seq as i64];
        let fused = AneMlPackageKernel::load(
            fmlc, &finput, &fused_output_names[0],
            fused_input_shape, fused_output_shape,
        ).expect("load fused");
        let (on_ane, report) = fused.verify_ane_dispatch().unwrap();
        assert!(on_ane, "fused QKV not on ANE\n{report}");

        // --- Warmup + time individual path ---
        let x_fp16 = vec![f16::from_f32(0.1).to_bits(); in_dim * seq];
        let mut ys: Vec<Vec<u16>> = projs.iter()
            .map(|(_, out)| vec![0u16; out * seq])
            .collect();

        for _ in 0..warmup {
            for (i, k) in singles.iter().enumerate() {
                k.predict_fp16(&x_fp16, &mut ys[i]).unwrap();
            }
        }
        let mut indiv_samples_us: Vec<u128> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            for (i, k) in singles.iter().enumerate() {
                k.predict_fp16(&x_fp16, &mut ys[i]).unwrap();
            }
            indiv_samples_us.push(t0.elapsed().as_micros());
        }
        indiv_samples_us.sort_unstable();

        // --- Warmup + time fused path ---
        let mut fused_ys: Vec<Vec<u16>> = projs.iter()
            .map(|(_, out)| vec![0u16; out * seq])
            .collect();
        for _ in 0..warmup {
            // Build the &mut slice of (name, buf) pairs fresh each call —
            // mutable borrows can't be reused.
            let mut pairs: Vec<(&str, &mut [u16])> = fused_output_names.iter()
                .map(|s| s.as_str())
                .zip(fused_ys.iter_mut().map(|v| v.as_mut_slice()))
                .collect();
            fused.predict_fp16_multi(&x_fp16, &mut pairs).unwrap();
        }
        let mut fused_samples_us: Vec<u128> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let mut pairs: Vec<(&str, &mut [u16])> = fused_output_names.iter()
                .map(|s| s.as_str())
                .zip(fused_ys.iter_mut().map(|v| v.as_mut_slice()))
                .collect();
            fused.predict_fp16_multi(&x_fp16, &mut pairs).unwrap();
            fused_samples_us.push(t0.elapsed().as_micros());
        }
        fused_samples_us.sort_unstable();

        // --- Analysis ---
        // Compute floor = sum over projections of (w_bytes + act_bytes)/bw.
        // act_bytes here is per-projection I/O (shared input x counted once
        // for fused, three times for individual — both are over-estimates of
        // the weight-dominated floor so the overhead figure is conservative).
        let mut total_w_bytes = 0.0;
        let mut total_single_act_bytes = 0.0;
        for (_, out_dim) in projs {
            total_w_bytes += (in_dim * out_dim) as f64;
            total_single_act_bytes +=
                ((in_dim * seq) + (out_dim * seq)) as f64 * 2.0;
        }
        let fused_act_bytes = (in_dim * seq) as f64 * 2.0
            + projs.iter().map(|(_, o)| (o * seq) as f64 * 2.0).sum::<f64>();
        let indiv_floor_us = (total_w_bytes + total_single_act_bytes)
            / (ane_peak_gbs * 1.0e9) * 1.0e6;
        let fused_floor_us = (total_w_bytes + fused_act_bytes)
            / (ane_peak_gbs * 1.0e9) * 1.0e6;

        let indiv_min_us = indiv_samples_us[0] as f64;
        let indiv_p50_us = indiv_samples_us[iters / 2] as f64;
        let fused_min_us = fused_samples_us[0] as f64;
        let fused_p50_us = fused_samples_us[iters / 2] as f64;

        let indiv_overhead_us = (indiv_min_us - indiv_floor_us).max(0.0);
        let fused_overhead_us = (fused_min_us - fused_floor_us).max(0.0);
        // Per-dispatch share implied by each path.
        let indiv_per_dispatch = indiv_overhead_us / projs.len() as f64;
        // Saved overhead relative to a no-fusion world.
        let saved_us = (indiv_overhead_us - fused_overhead_us).max(0.0);
        let amortization = if indiv_overhead_us > 0.0 {
            saved_us / indiv_overhead_us * 100.0
        } else { 0.0 };

        eprintln!(
            "DFlash-4B QKV fusion probe (seq={seq}, iters={iters}):"
        );
        eprintln!(
            "  individual: min={indiv_min_us:.1}us p50={indiv_p50_us:.1}us  \
             floor≈{indiv_floor_us:.1}us  overhead≈{indiv_overhead_us:.1}us  \
             (/3 = {indiv_per_dispatch:.1}us per dispatch)"
        );
        eprintln!(
            "  fused:      min={fused_min_us:.1}us p50={fused_p50_us:.1}us  \
             floor≈{fused_floor_us:.1}us  overhead≈{fused_overhead_us:.1}us"
        );
        eprintln!(
            "  saved overhead ≈ {saved_us:.1}us ({amortization:.0}% amortization)"
        );
        let verdict = if amortization >= 60.0 {
            "FUSION AMORTIZES — ship layer-level fusion"
        } else if amortization >= 30.0 {
            "PARTIAL AMORTIZATION — quantify vs 35-dispatch cost"
        } else {
            "NO AMORTIZATION — abandon public-MLModel fanout"
        };
        eprintln!("  verdict: {verdict}");
    }

    /// Full-layer int8 weight-work estimate at DFlash-4B, seq=32.
    /// 4 kernels/layer: QKV fused, O single, gate+up fused, down single.
    /// Measures wall-time for 1 layer forward-weight work, projects 5×
    /// for the full drafter stack. Compares to today's 18.5 ms fp16
    /// decode step. If projected << 18.5 ms, the int8+fusion path wins.
    #[test]
    #[ignore = "benchmark; set HIGGS_CORETOOLS_PYTHON"]
    fn dflash_ane_full_layer_int8_probe() {
        use std::fs;
        use std::process::Command;
        use std::time::Instant;

        let py = std::env::var("HIGGS_CORETOOLS_PYTHON")
            .expect("set HIGGS_CORETOOLS_PYTHON");
        let hidden: usize = 3072;
        let kv_head: usize = 1024;
        let ffn: usize = 9728;
        let seq: usize = 32;
        let iters: usize = 200;
        let warmup: usize = 20;

        let tmp = tempfile::tempdir().unwrap();
        let write_w = |label: &str, out: usize, in_: usize| -> std::path::PathBuf {
            let seed: Vec<f32> = (0..in_ * out)
                .map(|i| ((i as f32 * 0.1234).sin() * 0.05))
                .collect();
            let p = tmp.path().join(format!("{label}.bin"));
            let mut b = Vec::with_capacity(seed.len() * 4);
            for v in &seed { b.extend_from_slice(&v.to_le_bytes()); }
            fs::write(&p, &b).unwrap();
            p
        };
        let q_bin = write_w("q", hidden, hidden);
        let k_bin = write_w("k", kv_head, hidden);
        let v_bin = write_w("v", kv_head, hidden);
        let o_bin = write_w("o", hidden, hidden);
        let g_bin = write_w("g", ffn, hidden);
        let u_bin = write_w("u", ffn, hidden);
        let d_bin = write_w("d", hidden, ffn);

        let fused_script = env!("CARGO_MANIFEST_DIR").to_string()
            + "/scripts/quantize_int8_fused.py";
        let single_script = env!("CARGO_MANIFEST_DIR").to_string()
            + "/scripts/quantize_int8_proj.py";

        // Build QKV fused (in=hidden, seq=32).
        let qkv_dir = tmp.path().join("qkv"); fs::create_dir_all(&qkv_dir).unwrap();
        let o1 = Command::new(&py).arg(&fused_script)
            .arg("--in-features").arg(hidden.to_string())
            .arg("--seq-len").arg(seq.to_string())
            .arg("--out-dir").arg(&qkv_dir)
            .arg("--proj").arg(format!("q:{hidden}:{}", q_bin.display()))
            .arg("--proj").arg(format!("k:{kv_head}:{}", k_bin.display()))
            .arg("--proj").arg(format!("v:{kv_head}:{}", v_bin.display()))
            .output().expect("qkv build");
        assert!(o1.status.success(), "qkv: {}", String::from_utf8_lossy(&o1.stderr));
        let qkv_resp: serde_json::Value = serde_json::from_slice(&o1.stdout).unwrap();
        let qkv_mlc = qkv_resp["mlmodelc"].as_str().unwrap();

        // Build gate+up fused.
        let gu_dir = tmp.path().join("gu"); fs::create_dir_all(&gu_dir).unwrap();
        let o2 = Command::new(&py).arg(&fused_script)
            .arg("--in-features").arg(hidden.to_string())
            .arg("--seq-len").arg(seq.to_string())
            .arg("--out-dir").arg(&gu_dir)
            .arg("--proj").arg(format!("g:{ffn}:{}", g_bin.display()))
            .arg("--proj").arg(format!("u:{ffn}:{}", u_bin.display()))
            .output().expect("gu build");
        assert!(o2.status.success(), "gu: {}", String::from_utf8_lossy(&o2.stderr));
        let gu_resp: serde_json::Value = serde_json::from_slice(&o2.stdout).unwrap();
        let gu_mlc = gu_resp["mlmodelc"].as_str().unwrap();

        // Build O single (in=hidden, out=hidden).
        let build_single = |label: &str, wbin: &std::path::Path, out: usize, in_: usize| -> String {
            let d = tmp.path().join(format!("single_{label}"));
            fs::create_dir_all(&d).unwrap();
            let o = Command::new(&py).arg(&single_script)
                .arg("--weights-bin").arg(wbin)
                .arg("--out-features").arg(out.to_string())
                .arg("--in-features").arg(in_.to_string())
                .arg("--seq-len").arg(seq.to_string())
                .arg("--out-dir").arg(&d)
                .output().expect("single build");
            assert!(o.status.success(), "{label}: {}", String::from_utf8_lossy(&o.stderr));
            let r: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
            r["mlmodelc"].as_str().unwrap().to_owned()
        };
        let o_mlc = build_single("o", &o_bin, hidden, hidden);
        let d_mlc = build_single("d", &d_bin, hidden, ffn);

        // Load all kernels.
        let qkv = AneMlPackageKernel::load(qkv_mlc, "x", "y_q",
            vec![1, hidden as i64, 1, seq as i64],
            vec![1, hidden as i64, 1, seq as i64]).unwrap();
        let gu = AneMlPackageKernel::load(gu_mlc, "x", "y_g",
            vec![1, hidden as i64, 1, seq as i64],
            vec![1, ffn as i64, 1, seq as i64]).unwrap();
        let o = AneMlPackageKernel::load(&o_mlc, "x", "y",
            vec![1, hidden as i64, 1, seq as i64],
            vec![1, hidden as i64, 1, seq as i64]).unwrap();
        let d = AneMlPackageKernel::load(&d_mlc, "x", "y",
            vec![1, ffn as i64, 1, seq as i64],
            vec![1, hidden as i64, 1, seq as i64]).unwrap();

        for k in [&qkv, &gu, &o, &d] {
            let (on_ane, _) = k.verify_ane_dispatch().unwrap();
            assert!(on_ane, "kernel not on ANE: {}", k.mlmodelc_path);
        }

        let x_hid = vec![f16::from_f32(0.1).to_bits(); hidden * seq];
        let x_ffn = vec![f16::from_f32(0.1).to_bits(); ffn * seq];
        let mut y_q = vec![0u16; hidden * seq];
        let mut y_k = vec![0u16; kv_head * seq];
        let mut y_v = vec![0u16; kv_head * seq];
        let mut y_o = vec![0u16; hidden * seq];
        let mut y_g = vec![0u16; ffn * seq];
        let mut y_u = vec![0u16; ffn * seq];
        let mut y_d = vec![0u16; hidden * seq];

        // One "layer" = 4 dispatches (simulates weight-work, skips norm/attn/rope/residual).
        let run_layer = |y_q: &mut [u16], y_k: &mut [u16], y_v: &mut [u16],
                         y_o: &mut [u16], y_g: &mut [u16], y_u: &mut [u16],
                         y_d: &mut [u16]| {
            let mut qkv_pairs: Vec<(&str, &mut [u16])> = vec![
                ("y_q", y_q), ("y_k", y_k), ("y_v", y_v)
            ];
            qkv.predict_fp16_multi(&x_hid, &mut qkv_pairs).unwrap();
            o.predict_fp16(&x_hid, y_o).unwrap();
            let mut gu_pairs: Vec<(&str, &mut [u16])> = vec![
                ("y_g", y_g), ("y_u", y_u)
            ];
            gu.predict_fp16_multi(&x_hid, &mut gu_pairs).unwrap();
            d.predict_fp16(&x_ffn, y_d).unwrap();
        };

        for _ in 0..warmup {
            run_layer(&mut y_q, &mut y_k, &mut y_v, &mut y_o,
                      &mut y_g, &mut y_u, &mut y_d);
        }
        let mut samples_us: Vec<u128> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            run_layer(&mut y_q, &mut y_k, &mut y_v, &mut y_o,
                      &mut y_g, &mut y_u, &mut y_d);
            samples_us.push(t0.elapsed().as_micros());
        }
        samples_us.sort_unstable();
        let min_us = samples_us[0] as f64;
        let p50_us = samples_us[iters / 2] as f64;
        let p99_us = samples_us[(iters * 99) / 100] as f64;

        let layers: f64 = 5.0;
        let fp16_baseline_ms: f64 = 18.5;
        let proj_ms = min_us * layers / 1000.0;
        let p50_proj_ms = p50_us * layers / 1000.0;
        let savings_ms = fp16_baseline_ms - proj_ms;
        let pct = savings_ms / fp16_baseline_ms * 100.0;

        eprintln!(
            "DFlash-4B full-layer int8 weight-work (seq=32, iters={iters}):"
        );
        eprintln!("  1 layer: min={min_us:.1}us p50={p50_us:.1}us p99={p99_us:.1}us");
        eprintln!("  5 layers projected: min={proj_ms:.2}ms p50={p50_proj_ms:.2}ms");
        eprintln!("  fp16 baseline step: {fp16_baseline_ms} ms");
        eprintln!(
            "  naive delta (ignores non-weight work in baseline): \
             {savings_ms:+.2}ms ({pct:+.1}%)"
        );
        eprintln!(
            "  note: baseline 18.5ms includes norm/attn-core/rope/residual/lm_head;"
        );
        eprintln!(
            "        int8 savings apply only to weight kernels. The projected"
        );
        eprintln!(
            "        5-layer figure above is just the int8 weight-work portion;"
        );
        eprintln!(
            "        it must be smaller than the fp16 weight-work portion by"
        );
        eprintln!(
            "        ≥15% of the total baseline (2.8 ms) to ship."
        );
    }

    /// DFlash-4B MLP chain: gate/up [9728, 3072] and down [3072, 9728] at
    /// seq=64 (aligned). This is where AB3's bandwidth win lives — 747 MB
    /// fp16 → 375 MB int8 across the full chain. Target: each kernel hits
    /// ≥70 GB/s int8 (= ≥140 GB/s fp16-equivalent) to confirm ANE engaged.
    #[test]
    #[ignore = "benchmark; set HIGGS_CORETOOLS_PYTHON"]
    fn dflash_ane_mlp_chain_int8_latency() {
        use std::fs;
        use std::process::Command;
        use std::time::Instant;

        let py = std::env::var("HIGGS_CORETOOLS_PYTHON")
            .expect("set HIGGS_CORETOOLS_PYTHON");
        let seq: usize = std::env::var("HIGGS_BENCH_SEQ")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        let iters: usize = 50;
        let warmup: usize = 10;

        // (label, out_features, in_features)
        let shapes = [
            ("gate", 9728usize, 3072usize),
            ("up",   9728usize, 3072usize),
            ("down", 3072usize, 9728usize),
        ];

        eprintln!("| kernel | out×in        | min_ms | med_ms | int8 GB/s | fp16-eq GB/s | on_ane |");
        eprintln!("|--------|---------------|--------|--------|-----------|--------------|--------|");

        let tmp = tempfile::tempdir().unwrap();
        for (label, out_dim, in_dim) in shapes {
            let seed: Vec<f32> = (0..in_dim * out_dim)
                .map(|i| ((i as f32 * 0.1234).sin() * 0.05))
                .collect();
            let wbin = tmp.path().join(format!("{label}.bin"));
            let mut bytes = Vec::with_capacity(seed.len() * 4);
            for v in &seed { bytes.extend_from_slice(&v.to_le_bytes()); }
            fs::write(&wbin, &bytes).unwrap();

            let out_dir = tmp.path().join(format!("out_{label}"));
            fs::create_dir_all(&out_dir).unwrap();
            let script = env!("CARGO_MANIFEST_DIR").to_string()
                + "/scripts/quantize_int8_proj.py";
            let out = Command::new(&py)
                .arg(&script)
                .arg("--weights-bin").arg(&wbin)
                .arg("--out-features").arg(out_dim.to_string())
                .arg("--in-features").arg(in_dim.to_string())
                .arg("--seq-len").arg(seq.to_string())
                .arg("--out-dir").arg(&out_dir)
                .output().expect("spawn python");
            assert!(out.status.success(),
                "{label}: quantize failed\nstderr: {}",
                String::from_utf8_lossy(&out.stderr));
            let resp: serde_json::Value =
                serde_json::from_slice(&out.stdout).expect("json");
            let mlmodelc = resp["mlmodelc"].as_str().unwrap();
            let in_name = resp["input_name"].as_str().unwrap();
            let out_name = resp["output_name"].as_str().unwrap();

            let input_shape: Vec<i64> = vec![1, in_dim as i64, 1, seq as i64];
            let output_shape: Vec<i64> = vec![1, out_dim as i64, 1, seq as i64];
            let kernel = AneMlPackageKernel::load(
                mlmodelc, in_name, out_name, input_shape, output_shape,
            ).expect("load");
            let (on_ane, _rep) = kernel.verify_ane_dispatch().expect("verify");

            let x_fp16 = vec![f16::from_f32(0.1).to_bits(); in_dim * seq];
            let mut y_fp16 = vec![0u16; out_dim * seq];
            for _ in 0..warmup {
                kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");
            }
            let mut samples: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");
                samples.push(t0.elapsed().as_micros());
            }
            samples.sort_unstable();
            let min_ms = samples[0] as f64 / 1000.0;
            let med_ms = samples[iters / 2] as f64 / 1000.0;
            let w_bytes = (in_dim * out_dim) as f64;
            let int8_gbs = w_bytes / (min_ms * 1e-3) / 1e9;
            let fp16_eq_gbs = int8_gbs * 2.0;
            eprintln!(
                "| {label:<6} | {out_dim:>4}×{in_dim:<5} | {min_ms:>6.3} | {med_ms:>6.3} | {int8_gbs:>9.2} | {fp16_eq_gbs:>12.2} | {on_ane} |",
            );
        }
    }

    /// Qwen3-9B prefill go/no-go probe. Compares ANE int8 mlpackage forward
    /// vs MLX f32 matmul at the actual gate_proj shape used by Carnice-9B
    /// (hidden=4096, intermediate=12288) at seq=128.
    ///
    /// Decision rule: ANE int8 must beat MLX f32 matmul to justify the
    /// multi-session prefill wiring. MLX q4 (production path) is faster than
    /// f32 matmul, so f32 is a generous upper bound for "MLX baseline" — if
    /// ANE loses to f32 matmul, it loses harder to q4.
    ///
    /// Reports both wall-clock and parity of ANE int8 vs the f32 reference.
    #[test]
    #[ignore = "go/no-go probe; set HIGGS_CORETOOLS_PYTHON"]
    fn qwen3_9b_mlp_int8_vs_mlx_probe() {
        use mlx_rs::{ops::matmul, random};
        use std::fs;
        use std::process::Command;
        use std::time::Instant;

        let py = std::env::var("HIGGS_CORETOOLS_PYTHON")
            .expect("set HIGGS_CORETOOLS_PYTHON");
        // Carnice-9B (qwen3_5) MLP gate_proj shape.
        let in_dim: usize = 4096;
        let out_dim: usize = 12288;
        let seq: usize = 128;
        let iters: usize = 30;
        let warmup: usize = 5;

        // --- Synthetic weight + activation ---
        let w = random::uniform::<f32, f32>(-0.05, 0.05,
            &[out_dim as i32, in_dim as i32], None).expect("random W");
        w.eval().unwrap();
        let w_vec: Vec<f32> = w.as_slice::<f32>().to_vec();

        let x = random::uniform::<f32, f32>(-1.0, 1.0,
            &[1, seq as i32, in_dim as i32], None).expect("random x");
        x.eval().unwrap();
        let x_vec: Vec<f32> = x.as_slice::<f32>().to_vec();

        // --- Build int8 mlpackage ---
        let tmp = tempfile::tempdir().unwrap();
        let weights_bin = tmp.path().join("w.bin");
        let out_dir = tmp.path().join("out");
        fs::create_dir_all(&out_dir).unwrap();
        let mut bytes = Vec::with_capacity(w_vec.len() * 4);
        for v in &w_vec { bytes.extend_from_slice(&v.to_le_bytes()); }
        fs::write(&weights_bin, &bytes).unwrap();

        let script = env!("CARGO_MANIFEST_DIR").to_string()
            + "/scripts/quantize_int8_proj.py";
        let out = Command::new(&py)
            .arg(&script)
            .arg("--weights-bin").arg(&weights_bin)
            .arg("--out-features").arg(out_dim.to_string())
            .arg("--in-features").arg(in_dim.to_string())
            .arg("--seq-len").arg(seq.to_string())
            .arg("--out-dir").arg(&out_dir)
            .output().expect("spawn python");
        assert!(out.status.success(),
            "quantize failed\nstderr: {}",
            String::from_utf8_lossy(&out.stderr));
        let resp: serde_json::Value =
            serde_json::from_slice(&out.stdout).expect("json");
        let mlmodelc = resp["mlmodelc"].as_str().unwrap();
        let in_name = resp["input_name"].as_str().unwrap();
        let out_name = resp["output_name"].as_str().unwrap();

        // --- Load + verify ANE ---
        let input_shape: Vec<i64> = vec![1, in_dim as i64, 1, seq as i64];
        let output_shape: Vec<i64> = vec![1, out_dim as i64, 1, seq as i64];
        let kernel = AneMlPackageKernel::load(
            mlmodelc, in_name, out_name, input_shape, output_shape,
        ).expect("load");
        let (on_ane, report) = kernel.verify_ane_dispatch().expect("verify");
        eprintln!("{report}");
        assert!(on_ane, "shape should dispatch to ANE — got:\n{report}");

        // --- Pack input into conv1x1 layout fp16 ---
        let pin = in_dim * seq;
        let pout = out_dim * seq;
        let mut x_fp16 = vec![0u16; pin];
        for t in 0..seq {
            for ci in 0..in_dim {
                let src = x_vec[t * in_dim + ci];
                x_fp16[ci * seq + t] = f16::from_f32(src).to_bits();
            }
        }
        let mut y_fp16 = vec![0u16; pout];

        // --- Bench ANE ---
        for _ in 0..warmup {
            kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");
        }
        let mut ane_samples: Vec<u128> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");
            ane_samples.push(t0.elapsed().as_micros());
        }
        ane_samples.sort_unstable();
        let ane_min_ms = ane_samples[0] as f64 / 1000.0;
        let ane_med_ms = ane_samples[iters / 2] as f64 / 1000.0;

        // --- Bench MLX f32 matmul (generous upper bound) ---
        let wt = w.t();
        wt.eval().unwrap();
        for _ in 0..warmup {
            let y = matmul(&x, &wt).expect("matmul");
            y.eval().unwrap();
        }
        let mut mlx_samples: Vec<u128> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let y = matmul(&x, &wt).expect("matmul");
            y.eval().unwrap();
            mlx_samples.push(t0.elapsed().as_micros());
        }
        mlx_samples.sort_unstable();
        let mlx_min_ms = mlx_samples[0] as f64 / 1000.0;
        let mlx_med_ms = mlx_samples[iters / 2] as f64 / 1000.0;

        // --- Bench MLX q4 matmul (production baseline; group_size=64 bits=4) ---
        let (qw, qs, qb) = mlx_rs::ops::quantize(&w, 64, 4).expect("quantize q4");
        qw.eval().unwrap(); qs.eval().unwrap(); qb.eval().unwrap();
        for _ in 0..warmup {
            let y = mlx_rs::ops::quantized_matmul(&x, &qw, &qs, &qb, true, 64, 4)
                .expect("qmm");
            y.eval().unwrap();
        }
        let mut q4_samples: Vec<u128> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let y = mlx_rs::ops::quantized_matmul(&x, &qw, &qs, &qb, true, 64, 4)
                .expect("qmm");
            y.eval().unwrap();
            q4_samples.push(t0.elapsed().as_micros());
        }
        q4_samples.sort_unstable();
        let q4_min_ms = q4_samples[0] as f64 / 1000.0;
        let q4_med_ms = q4_samples[iters / 2] as f64 / 1000.0;

        // --- Parity ---
        let mut y_ane = vec![0.0f32; seq * out_dim];
        for co in 0..out_dim {
            for t in 0..seq {
                y_ane[t * out_dim + co] =
                    f16::from_bits(y_fp16[co * seq + t]).to_f32();
            }
        }
        let y_ref = matmul(&x, &wt).expect("matmul");
        y_ref.eval().unwrap();
        let y_ref_vec: Vec<f32> = y_ref.as_slice::<f32>().to_vec();
        let mut max_diff = 0.0f32;
        let mut max_abs_ref = 0.0f32;
        for (a, b) in y_ane.iter().zip(y_ref_vec.iter()) {
            let d = (a - b).abs();
            if d > max_diff { max_diff = d; }
            let ab = b.abs();
            if ab > max_abs_ref { max_abs_ref = ab; }
        }

        let w_bytes = (in_dim * out_dim) as f64;
        let ane_int8_gbs = w_bytes / (ane_min_ms * 1e-3) / 1e9;
        let speedup_f32 = mlx_min_ms / ane_min_ms;
        let speedup_q4 = q4_min_ms / ane_min_ms;

        eprintln!("");
        eprintln!("=== Qwen3-9B gate_proj prefill probe ({out_dim}×{in_dim} seq={seq}) ===");
        eprintln!("ANE int8: min={ane_min_ms:.3} ms  med={ane_med_ms:.3} ms  ({ane_int8_gbs:.1} GB/s int8)");
        eprintln!("MLX f32 : min={mlx_min_ms:.3} ms  med={mlx_med_ms:.3} ms");
        eprintln!("MLX q4  : min={q4_min_ms:.3} ms  med={q4_med_ms:.3} ms  (production baseline)");
        eprintln!("speedup vs f32: {speedup_f32:.2}x");
        eprintln!("speedup vs q4 : {speedup_q4:.2}x  <-- DECISION GATE (need >1.5x to wire)");
        eprintln!("parity  : max_diff={max_diff:.4}  max|ref|={max_abs_ref:.2}");
        eprintln!("");

        // Informational — never panic. The verdict is the printed numbers.
        assert!(max_diff <= 0.5,
            "parity sanity: max_diff={max_diff} exceeds 0.5 (something is wrong)");
    }

    /// Layer-0 MLP projection sweep — runs the int8-vs-q4 probe for BOTH
    /// shape orientations used by Qwen3-9B MLP:
    ///   gate/up : in=4096, out=12288  (hidden → intermediate)
    ///   down    : in=12288, out=4096  (intermediate → hidden)
    ///
    /// The original probe only covered gate/up. Before wiring all three
    /// projections we need confirmation that the reverse orientation also
    /// clears the >1.5× gate vs MLX q4 (bandwidth argument predicts yes
    /// — 50M weights either way — but compute-cliff behavior can differ).
    #[test]
    #[ignore = "go/no-go probe; set HIGGS_CORETOOLS_PYTHON"]
    fn qwen3_9b_mlp_projections_probe() {
        use mlx_rs::{ops::matmul, random};
        use std::fs;
        use std::process::Command;
        use std::time::Instant;

        let py = std::env::var("HIGGS_CORETOOLS_PYTHON")
            .expect("set HIGGS_CORETOOLS_PYTHON");
        let seq: usize = std::env::var("HIGGS_ANE_INT8_PROBE_SEQ")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(128);
        let iters: usize = 30;
        let warmup: usize = 5;

        // Qwen3-9B MLP projection shapes. hidden=4096, intermediate=12288.
        let cases: &[(usize, usize, &str)] = &[
            (4096, 12288, "gate/up"),
            (12288, 4096, "down"),
        ];

        eprintln!("");
        eprintln!("=== Qwen3-9B MLP projections probe (seq={seq}) ===");
        eprintln!("| proj    | shape         | ANE int8 | MLX f32  | MLX q4   | vs q4  | on_ane | max_diff |");
        eprintln!("|---------|---------------|----------|----------|----------|--------|--------|----------|");

        let mut all_pass = true;
        for &(in_dim, out_dim, label) in cases {
            // --- Synthetic weight + activation ---
            let w = random::uniform::<f32, f32>(-0.05, 0.05,
                &[out_dim as i32, in_dim as i32], None).expect("random W");
            w.eval().unwrap();
            let w_vec: Vec<f32> = w.as_slice::<f32>().to_vec();

            let x = random::uniform::<f32, f32>(-1.0, 1.0,
                &[1, seq as i32, in_dim as i32], None).expect("random x");
            x.eval().unwrap();
            let x_vec: Vec<f32> = x.as_slice::<f32>().to_vec();

            // --- Build int8 mlpackage ---
            let tmp = tempfile::tempdir().unwrap();
            let weights_bin = tmp.path().join("w.bin");
            let out_dir = tmp.path().join("out");
            fs::create_dir_all(&out_dir).unwrap();
            let mut bytes = Vec::with_capacity(w_vec.len() * 4);
            for v in &w_vec { bytes.extend_from_slice(&v.to_le_bytes()); }
            fs::write(&weights_bin, &bytes).unwrap();

            let script = env!("CARGO_MANIFEST_DIR").to_string()
                + "/scripts/quantize_int8_proj.py";
            let out = Command::new(&py)
                .arg(&script)
                .arg("--weights-bin").arg(&weights_bin)
                .arg("--out-features").arg(out_dim.to_string())
                .arg("--in-features").arg(in_dim.to_string())
                .arg("--seq-len").arg(seq.to_string())
                .arg("--out-dir").arg(&out_dir)
                .output().expect("spawn python");
            assert!(out.status.success(),
                "quantize failed ({label})\nstderr: {}",
                String::from_utf8_lossy(&out.stderr));
            let resp: serde_json::Value =
                serde_json::from_slice(&out.stdout).expect("json");
            let mlmodelc = resp["mlmodelc"].as_str().unwrap();
            let in_name = resp["input_name"].as_str().unwrap();
            let out_name = resp["output_name"].as_str().unwrap();

            // --- Load + verify ANE ---
            let input_shape: Vec<i64> = vec![1, in_dim as i64, 1, seq as i64];
            let output_shape: Vec<i64> = vec![1, out_dim as i64, 1, seq as i64];
            let kernel = AneMlPackageKernel::load(
                mlmodelc, in_name, out_name, input_shape, output_shape,
            ).expect("load");
            let (on_ane, _report) = kernel.verify_ane_dispatch().expect("verify");

            // --- Pack input into conv1x1 layout fp16 ---
            let pin = in_dim * seq;
            let pout = out_dim * seq;
            let mut x_fp16 = vec![0u16; pin];
            for t in 0..seq {
                for ci in 0..in_dim {
                    let src = x_vec[t * in_dim + ci];
                    x_fp16[ci * seq + t] = f16::from_f32(src).to_bits();
                }
            }
            let mut y_fp16 = vec![0u16; pout];

            // --- Bench ANE ---
            for _ in 0..warmup {
                kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");
            }
            let mut ane_samples: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                kernel.predict_fp16(&x_fp16, &mut y_fp16).expect("predict");
                ane_samples.push(t0.elapsed().as_micros());
            }
            ane_samples.sort_unstable();
            let ane_min_ms = ane_samples[0] as f64 / 1000.0;

            // --- Bench MLX f32 matmul ---
            let wt = w.t();
            wt.eval().unwrap();
            for _ in 0..warmup {
                let y = matmul(&x, &wt).expect("matmul");
                y.eval().unwrap();
            }
            let mut mlx_samples: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let y = matmul(&x, &wt).expect("matmul");
                y.eval().unwrap();
                mlx_samples.push(t0.elapsed().as_micros());
            }
            mlx_samples.sort_unstable();
            let mlx_min_ms = mlx_samples[0] as f64 / 1000.0;

            // --- Bench MLX q4 (production baseline; group_size=64 bits=4) ---
            let (qw, qs, qb) = mlx_rs::ops::quantize(&w, 64, 4).expect("quantize q4");
            qw.eval().unwrap(); qs.eval().unwrap(); qb.eval().unwrap();
            for _ in 0..warmup {
                let y = mlx_rs::ops::quantized_matmul(&x, &qw, &qs, &qb, true, 64, 4)
                    .expect("qmm");
                y.eval().unwrap();
            }
            let mut q4_samples: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let y = mlx_rs::ops::quantized_matmul(&x, &qw, &qs, &qb, true, 64, 4)
                    .expect("qmm");
                y.eval().unwrap();
                q4_samples.push(t0.elapsed().as_micros());
            }
            q4_samples.sort_unstable();
            let q4_min_ms = q4_samples[0] as f64 / 1000.0;

            // --- Parity ---
            let mut y_ane = vec![0.0f32; seq * out_dim];
            for co in 0..out_dim {
                for t in 0..seq {
                    y_ane[t * out_dim + co] =
                        f16::from_bits(y_fp16[co * seq + t]).to_f32();
                }
            }
            let y_ref = matmul(&x, &wt).expect("matmul");
            y_ref.eval().unwrap();
            let y_ref_vec: Vec<f32> = y_ref.as_slice::<f32>().to_vec();
            let mut max_diff = 0.0f32;
            for (a, b) in y_ane.iter().zip(y_ref_vec.iter()) {
                let d = (a - b).abs();
                if d > max_diff { max_diff = d; }
            }

            let speedup_q4 = q4_min_ms / ane_min_ms;
            let shape = format!("{out_dim}×{in_dim}");
            eprintln!(
                "| {label:<7} | {shape:<13} | {ane_min_ms:>6.3} ms | {mlx_min_ms:>6.3} ms | {q4_min_ms:>6.3} ms | {speedup_q4:>5.2}x | {on_ane:>6} | {max_diff:>8.4} |",
            );

            if speedup_q4 < 1.5 || !on_ane {
                all_pass = false;
            }
            assert!(max_diff <= 0.5,
                "parity sanity ({label}): max_diff={max_diff} exceeds 0.5");
        }
        eprintln!("");
        eprintln!("DECISION GATE: all shapes must clear >1.5x vs q4 and dispatch on ANE.");
        if !all_pass {
            eprintln!("WARN: at least one projection did not clear the gate — probe informational, not a hard fail.");
        }
    }
}
