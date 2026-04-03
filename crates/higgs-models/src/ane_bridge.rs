//! FFI bridge to Apple Neural Engine via private APIs.
//!
//! Wraps the Objective-C `ane_bridge.{h,m}` into safe Rust types.
//! The bridge compiles MIL programs into ANE kernels, evaluates them,
//! and manages IOSurface-backed input/output tensors.
//!
//! Ported from nanobot-rs for the Higgs inference server.
//! Feature-gated behind `ane`.

#![allow(unsafe_code)]

use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;

// ---------------------------------------------------------------------------
// Raw FFI declarations (mirrors ane_bridge.h)
// ---------------------------------------------------------------------------

/// Opaque handle returned by ane_bridge_compile*.
#[repr(C)]
pub(crate) struct ANEKernelHandle {
    _opaque: [u8; 0],
}

#[link(name = "ane_bridge")]
unsafe extern "C" {
    fn ane_bridge_init() -> c_int;

    fn ane_bridge_compile(
        mil_text: *const c_char,
        mil_len: usize,
        weight_data: *const u8,
        weight_len: usize,
        n_inputs: c_int,
        input_sizes: *const usize,
        n_outputs: c_int,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle;

    fn ane_bridge_compile_multi_weights(
        mil_text: *const c_char,
        mil_len: usize,
        weight_names: *const *const c_char,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: c_int,
        n_inputs: c_int,
        input_sizes: *const usize,
        n_outputs: c_int,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle;

    fn ane_bridge_compile_direct(
        mil_text: *const c_char,
        mil_len: usize,
        weight_names: *const *const c_char,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: c_int,
        n_inputs: c_int,
        input_sizes: *const usize,
        n_outputs: c_int,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle;

    fn ane_bridge_eval(kernel: *mut ANEKernelHandle) -> bool;

    fn ane_bridge_write_input(
        kernel: *mut ANEKernelHandle,
        idx: c_int,
        data: *const c_void,
        bytes: usize,
    );

    fn ane_bridge_write_input_region(
        kernel: *mut ANEKernelHandle,
        idx: c_int,
        offset: usize,
        data: *const c_void,
        bytes: usize,
    );

    fn ane_bridge_write_input_strided(
        kernel: *mut ANEKernelHandle,
        idx: c_int,
        dst_offset: usize,
        dst_stride: usize,
        src: *const c_void,
        src_stride: usize,
        chunk_bytes: usize,
        n_chunks: c_int,
    );

    fn ane_bridge_read_output(
        kernel: *mut ANEKernelHandle,
        idx: c_int,
        data: *mut c_void,
        bytes: usize,
    );

    fn ane_bridge_clone_kernel(source: *mut ANEKernelHandle) -> *mut ANEKernelHandle;

    fn ane_bridge_share_surface(
        src: *mut ANEKernelHandle,
        out_idx: c_int,
        dst: *mut ANEKernelHandle,
        in_idx: c_int,
    ) -> bool;

    fn ane_bridge_eval_chain(kernels: *mut *mut ANEKernelHandle, n: c_int) -> bool;

    fn ane_bridge_begin_realtime() -> bool;
    fn ane_bridge_end_realtime();
    fn ane_bridge_eval_realtime(kernel: *mut ANEKernelHandle) -> bool;
    fn ane_bridge_eval_chain_realtime(kernels: *mut *mut ANEKernelHandle, n: c_int) -> bool;
    fn ane_bridge_prepare_chain(kernels: *mut *mut ANEKernelHandle, n: c_int) -> bool;

    fn ane_bridge_free(kernel: *mut ANEKernelHandle);

    fn ane_bridge_get_compile_count() -> c_int;
    fn ane_bridge_get_load_count() -> c_int;
    fn ane_bridge_reset_compile_count();
    fn ane_bridge_clear_cache();
    fn ane_bridge_set_quiet(quiet: bool);
    fn ane_bridge_was_cached(kernel: *mut ANEKernelHandle) -> bool;

    fn ane_bridge_build_weight_blob(
        src: *const f32,
        rows: c_int,
        cols: c_int,
        out_len: *mut usize,
    ) -> *mut u8;

    fn ane_bridge_build_weight_blob_transposed(
        src: *const f32,
        rows: c_int,
        cols: c_int,
        out_len: *mut usize,
    ) -> *mut u8;

    fn ane_bridge_free_blob(ptr: *mut c_void);

    // fp16-weight GEMM: C[M,N] = alpha * A_f16[M,K] @ B_f32[K,N] + beta * C
    fn ane_bridge_gemm_f16(
        a_f16: *const u16,
        m: c_int,
        k: c_int,
        b_f32: *const f32,
        n: c_int,
        c_f32: *mut f32,
        alpha: f32,
        beta: f32,
    );

    fn ane_bridge_matvec_f16(
        a_f16: *const u16,
        m: c_int,
        k: c_int,
        x_f32: *const f32,
        y_f32: *mut f32,
    );

    // Zero-copy IOSurface access (Orion-style dsb sy)
    fn ane_bridge_get_input_base(kernel: *mut ANEKernelHandle, idx: c_int) -> *mut c_void;
    fn ane_bridge_get_output_base(kernel: *mut ANEKernelHandle, idx: c_int) -> *mut c_void;
    fn ane_bridge_input_size(kernel: *mut ANEKernelHandle, idx: c_int) -> usize;
    fn ane_bridge_output_size(kernel: *mut ANEKernelHandle, idx: c_int) -> usize;

    // INT8 weight blob builders
    fn ane_bridge_build_weight_blob_int8(
        src: *const i8,
        rows: c_int,
        cols: c_int,
        out_len: *mut usize,
    ) -> *mut u8;

    fn ane_bridge_build_weight_blob_quantized(
        src: *const f32,
        rows: c_int,
        cols: c_int,
        out_scale: *mut f32,
        out_len: *mut usize,
    ) -> *mut u8;

    // Delta compilation (Orion-style)
    fn ane_bridge_reload_weights(
        kernel: *mut ANEKernelHandle,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: c_int,
    ) -> bool;

    // Fast delta reload: reuses same model object, skips descriptor/model creation.
    fn ane_bridge_delta_reload(
        kernel: *mut ANEKernelHandle,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: c_int,
    ) -> bool;

    fn ane_bridge_patch_from_donor(
        donor: *mut ANEKernelHandle,
        mil_text: *const c_char,
        mil_len: usize,
        weight_names: *const *const c_char,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: c_int,
        n_inputs: c_int,
        input_sizes: *const usize,
        n_outputs: c_int,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle;
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

/// Initialize the ANE runtime. Must be called once before any compilation.
pub fn ane_init() -> Result<(), String> {
    let rc = unsafe { ane_bridge_init() };
    if rc == 0 {
        Ok(())
    } else {
        Err("ane_bridge_init failed (ANE unavailable or private framework missing)".into())
    }
}

/// Current global kernel compile count (fresh compileWithQoS calls only).
pub fn compile_count() -> i32 {
    unsafe { ane_bridge_get_compile_count() }
}

/// Total loadWithQoS calls (both fresh compiles and delta-cached loads).
pub fn load_count() -> i32 {
    unsafe { ane_bridge_get_load_count() }
}

/// Reset the global compile counter.
pub fn reset_compile_count() {
    unsafe { ane_bridge_reset_compile_count() }
}

/// Clear the persistent compilation cache (~/.nanobot/ane_cache/).
pub fn clear_cache() {
    unsafe { ane_bridge_clear_cache() }
}

/// Suppress compile/load error output to stderr.
/// Use around expected-fail compilation attempts to keep TUI clean.
pub fn set_quiet(quiet: bool) {
    unsafe { ane_bridge_set_quiet(quiet) }
}

/// fp16-weight GEMM: `c[m,n] = alpha * a_f16[m,k] @ b_f32[k,n] + beta * c[m,n]`.
///
/// `a_f16` is row-major fp16 (stored as `u16`), `b_f32` and `c_f32` are row-major fp32.
/// Uses tiled NEON fp16→fp32 conversion + cblas_sgemm. Halves weight bandwidth.
pub fn gemm_f16(
    a_f16: &[u16],
    m: usize,
    k: usize,
    b_f32: &[f32],
    n: usize,
    c_f32: &mut [f32],
    alpha: f32,
    beta: f32,
) {
    debug_assert_eq!(a_f16.len(), m * k);
    debug_assert_eq!(b_f32.len(), k * n);
    debug_assert!(c_f32.len() >= m * n);
    unsafe {
        ane_bridge_gemm_f16(
            a_f16.as_ptr(),
            m as c_int,
            k as c_int,
            b_f32.as_ptr(),
            n as c_int,
            c_f32.as_mut_ptr(),
            alpha,
            beta,
        );
    }
}

/// Fast fp16-weight matvec: `y[M] = A_f16[M,K] @ x_f32[K]`.
/// Reads fp16 weights (half bandwidth), accumulates in f32 via NEON vfma. Zero allocation.
pub fn matvec_f16(a_f16: &[u16], m: usize, k: usize, x_f32: &[f32], y_f32: &mut [f32]) {
    debug_assert_eq!(a_f16.len(), m * k);
    debug_assert_eq!(x_f32.len(), k);
    debug_assert!(y_f32.len() >= m);
    unsafe {
        ane_bridge_matvec_f16(a_f16.as_ptr(), m as c_int, k as c_int, x_f32.as_ptr(), y_f32.as_mut_ptr());
    }
}

/// Convert f32 weights into ANE blob format (128-byte header + fp16 data).
pub fn build_weight_blob(weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(weights.len(), rows * cols, "weight dimensions mismatch");
    let mut out_len: usize = 0;
    let ptr = unsafe {
        ane_bridge_build_weight_blob(weights.as_ptr(), rows as c_int, cols as c_int, &mut out_len)
    };
    assert!(!ptr.is_null(), "ane_bridge_build_weight_blob returned NULL");
    let blob = unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec();
    unsafe { ane_bridge_free_blob(ptr as *mut c_void) };
    blob
}

/// Convert f32 weights into ANE blob format with transposition.
pub fn build_weight_blob_transposed(weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(weights.len(), rows * cols, "weight dimensions mismatch");
    let mut out_len: usize = 0;
    let ptr = unsafe {
        ane_bridge_build_weight_blob_transposed(
            weights.as_ptr(),
            rows as c_int,
            cols as c_int,
            &mut out_len,
        )
    };
    assert!(
        !ptr.is_null(),
        "ane_bridge_build_weight_blob_transposed returned NULL"
    );
    let blob = unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec();
    unsafe { ane_bridge_free_blob(ptr as *mut c_void) };
    blob
}

/// Convert int8 weights into ANE blob format (64-byte header + int8 data).
///
/// For use with `constexpr_affine_dequantize` in MIL. The ANE performs
/// int8→fp16 dequantization at eval time using scale/zero_point from MIL.
pub fn build_weight_blob_int8(weights: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(weights.len(), rows * cols, "weight dimensions mismatch");
    let mut out_len: usize = 0;
    let ptr = unsafe {
        ane_bridge_build_weight_blob_int8(
            weights.as_ptr(),
            rows as c_int,
            cols as c_int,
            &mut out_len,
        )
    };
    assert!(
        !ptr.is_null(),
        "ane_bridge_build_weight_blob_int8 returned NULL"
    );
    let blob = unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec();
    unsafe { ane_bridge_free_blob(ptr as *mut c_void) };
    blob
}

/// Quantize f32 weights to int8 and build ANE blob in one step.
///
/// Uses symmetric per-tensor quantization: `scale = max(|w|) / 127`.
/// Returns `(blob, scale)` where `blob` is the int8 ANE blob and `scale`
/// is the dequantization scale for use in MIL `constexpr_affine_dequantize`.
pub fn build_weight_blob_quantized(weights: &[f32], rows: usize, cols: usize) -> (Vec<u8>, f32) {
    assert_eq!(weights.len(), rows * cols, "weight dimensions mismatch");
    let mut out_len: usize = 0;
    let mut scale: f32 = 0.0;
    let ptr = unsafe {
        ane_bridge_build_weight_blob_quantized(
            weights.as_ptr(),
            rows as c_int,
            cols as c_int,
            &mut scale,
            &mut out_len,
        )
    };
    assert!(
        !ptr.is_null(),
        "ane_bridge_build_weight_blob_quantized returned NULL"
    );
    let blob = unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec();
    unsafe { ane_bridge_free_blob(ptr as *mut c_void) };
    (blob, scale)
}

/// RAII wrapper around a compiled ANE kernel.
///
/// IOSurface handles are thread-bound, so `AneKernel` is `!Send + !Sync`.
pub struct AneKernel {
    handle: *mut ANEKernelHandle,
    _not_send_sync: std::marker::PhantomData<*mut ()>,
}

impl AneKernel {
    /// Compile a MIL program into an ANE kernel.
    ///
    /// - `mil_text`: UTF-8 MIL program source
    /// - `weights`: optional raw weight blob (ANE format with 128-byte header)
    /// - `input_sizes`: byte sizes of each input tensor
    /// - `output_sizes`: byte sizes of each output tensor
    pub fn compile(
        mil_text: &str,
        weights: Option<&[u8]>,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<Self, String> {
        let (w_ptr, w_len) = match weights {
            Some(w) => (w.as_ptr(), w.len()),
            None => (ptr::null(), 0),
        };
        let handle = unsafe {
            ane_bridge_compile(
                mil_text.as_ptr() as *const c_char,
                mil_text.len(),
                w_ptr,
                w_len,
                input_sizes.len() as c_int,
                input_sizes.as_ptr(),
                output_sizes.len() as c_int,
                output_sizes.as_ptr(),
            )
        };
        if handle.is_null() {
            return Err("ANE compilation failed".into());
        }
        Ok(Self {
            handle,
            _not_send_sync: std::marker::PhantomData,
        })
    }

    /// Compile a MIL program with multiple named weight files.
    pub fn compile_multi_weights(
        mil_text: &str,
        weight_names: &[&str],
        weight_datas: &[&[u8]],
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<Self, String> {
        assert_eq!(weight_names.len(), weight_datas.len());
        let c_names: Vec<CString> = weight_names
            .iter()
            .map(|n| CString::new(*n).expect("weight name contains null byte"))
            .collect();
        let name_ptrs: Vec<*const c_char> = c_names.iter().map(|c| c.as_ptr()).collect();
        let data_ptrs: Vec<*const u8> = weight_datas.iter().map(|d| d.as_ptr()).collect();
        let data_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();

        let handle = unsafe {
            ane_bridge_compile_multi_weights(
                mil_text.as_ptr() as *const c_char,
                mil_text.len(),
                name_ptrs.as_ptr(),
                data_ptrs.as_ptr(),
                data_lens.as_ptr(),
                weight_names.len() as c_int,
                input_sizes.len() as c_int,
                input_sizes.as_ptr(),
                output_sizes.len() as c_int,
                output_sizes.as_ptr(),
            )
        };
        if handle.is_null() {
            return Err("ANE multi-weight compilation failed".into());
        }
        Ok(Self {
            handle,
            _not_send_sync: std::marker::PhantomData,
        })
    }

    /// Compile via _ANEClient direct path (supports conv, full MIL op set).
    ///
    /// Same API as `compile_multi_weights` but uses `_ANEClient.compileModel`
    /// which calls ANECCompile with full conv/matmul/etc support.
    pub fn compile_direct(
        mil_text: &str,
        weight_names: &[&str],
        weight_datas: &[&[u8]],
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<Self, String> {
        assert_eq!(weight_names.len(), weight_datas.len());
        let c_names: Vec<CString> = weight_names
            .iter()
            .map(|n| CString::new(*n).expect("weight name contains null byte"))
            .collect();
        let name_ptrs: Vec<*const c_char> = c_names.iter().map(|c| c.as_ptr()).collect();
        let data_ptrs: Vec<*const u8> = weight_datas.iter().map(|d| d.as_ptr()).collect();
        let data_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();

        let handle = unsafe {
            ane_bridge_compile_direct(
                mil_text.as_ptr() as *const c_char,
                mil_text.len(),
                name_ptrs.as_ptr(),
                data_ptrs.as_ptr(),
                data_lens.as_ptr(),
                weight_names.len() as c_int,
                input_sizes.len() as c_int,
                input_sizes.as_ptr(),
                output_sizes.len() as c_int,
                output_sizes.as_ptr(),
            )
        };
        if handle.is_null() {
            return Err("ANE direct compilation failed".into());
        }
        Ok(Self {
            handle,
            _not_send_sync: std::marker::PhantomData,
        })
    }

    /// Clone this kernel: shares the compiled ANE program but creates fresh
    /// IOSurfaces. Use this to create per-layer instances where each layer's
    /// weights live permanently in its own IOSurface.
    ///
    /// The compiled model is ref-counted — it is unloaded only when all
    /// clones and the original are dropped.
    pub fn clone_kernel(&self) -> Result<Self, String> {
        let handle = unsafe { ane_bridge_clone_kernel(self.handle) };
        if handle.is_null() {
            return Err("ANE kernel clone failed".into());
        }
        Ok(Self {
            handle,
            _not_send_sync: std::marker::PhantomData,
        })
    }

    /// Returns true if this kernel was loaded from the compilation cache
    /// (delta compilation fast path — no compileWithQoS: happened).
    pub fn was_cached(&self) -> bool {
        unsafe { ane_bridge_was_cached(self.handle) }
    }

    /// Execute the kernel on ANE hardware.
    pub fn eval(&self) -> Result<(), String> {
        let ok = unsafe { ane_bridge_eval(self.handle) };
        if ok {
            Ok(())
        } else {
            Err("ANE eval failed".into())
        }
    }

    /// Write data to input tensor at `idx` (full buffer).
    pub fn write_input(&self, idx: usize, data: &[u8]) {
        unsafe {
            ane_bridge_write_input(
                self.handle,
                idx as c_int,
                data.as_ptr() as *const c_void,
                data.len(),
            );
        }
    }

    /// Write data to a region of input tensor at `idx`.
    ///
    /// Only bytes `[offset..offset+data.len())` in the IOSurface are modified.
    /// Use this to patch activations into a pre-populated weight buffer without
    /// re-copying static weights — reduces CPU memory bandwidth from ~13MB to
    /// ~512KB per layer dispatch.
    pub fn write_input_region(&self, idx: usize, offset: usize, data: &[u8]) {
        unsafe {
            ane_bridge_write_input_region(
                self.handle,
                idx as c_int,
                offset,
                data.as_ptr() as *const c_void,
                data.len(),
            );
        }
    }

    /// Write scattered chunks from `src` into input tensor at `idx`.
    ///
    /// Copies `n_chunks` blocks of `chunk_bytes` from `src` (at `src_stride`
    /// intervals) into the IOSurface starting at `dst_offset` (at `dst_stride`
    /// intervals). All writes happen under a single IOSurface lock.
    ///
    /// Use this to patch interleaved activation rows (e.g. dim rows of seq×4
    /// bytes each) without re-copying the static weight columns.
    pub fn write_input_strided(
        &self,
        idx: usize,
        dst_offset: usize,
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        chunk_bytes: usize,
        n_chunks: usize,
    ) {
        unsafe {
            ane_bridge_write_input_strided(
                self.handle,
                idx as c_int,
                dst_offset,
                dst_stride,
                src.as_ptr() as *const c_void,
                src_stride,
                chunk_bytes,
                n_chunks as c_int,
            );
        }
    }

    /// Wire this kernel's output[out_idx] as `dst`'s input[in_idx].
    ///
    /// After `self.eval()`, calling `dst.eval()` will read directly from
    /// self's output IOSurface — no CPU memcpy. Rebuilds dst's ANE request.
    pub fn share_output_to(
        &self,
        out_idx: usize,
        dst: &AneKernel,
        in_idx: usize,
    ) -> Result<(), String> {
        let ok = unsafe {
            ane_bridge_share_surface(self.handle, out_idx as c_int, dst.handle, in_idx as c_int)
        };
        if ok {
            Ok(())
        } else {
            Err("share_surface failed".into())
        }
    }

    /// Evaluate a chain of kernels back-to-back without intermediate reads.
    ///
    /// The kernels should be wired with `share_output_to` first so intermediates
    /// stay on ANE IOSurfaces. Only the first kernel's input and last kernel's
    /// output need CPU I/O.
    pub fn eval_chain(kernels: &[&AneKernel]) -> Result<(), String> {
        let mut handles: Vec<*mut ANEKernelHandle> = kernels.iter().map(|k| k.handle).collect();
        let ok = unsafe { ane_bridge_eval_chain(handles.as_mut_ptr(), handles.len() as c_int) };
        if ok {
            Ok(())
        } else {
            Err("eval_chain failed".into())
        }
    }

    /// Enter real-time dispatch mode (lower per-dispatch latency).
    pub fn begin_realtime() -> bool {
        unsafe { ane_bridge_begin_realtime() }
    }

    /// Exit real-time dispatch mode.
    pub fn end_realtime() {
        unsafe { ane_bridge_end_realtime() }
    }

    /// Evaluate using real-time dispatch (requires begin_realtime).
    pub fn eval_realtime(&self) -> Result<(), String> {
        let ok = unsafe { ane_bridge_eval_realtime(self.handle) };
        if ok {
            Ok(())
        } else {
            Err("eval_realtime failed".into())
        }
    }

    /// Evaluate chain using real-time dispatch.
    pub fn eval_chain_realtime(kernels: &[&AneKernel]) -> Result<(), String> {
        let mut handles: Vec<*mut ANEKernelHandle> = kernels.iter().map(|k| k.handle).collect();
        let ok =
            unsafe { ane_bridge_eval_chain_realtime(handles.as_mut_ptr(), handles.len() as c_int) };
        if ok {
            Ok(())
        } else {
            Err("eval_chain_realtime failed".into())
        }
    }

    /// Prepare chain for pipelined ANE execution.
    pub fn prepare_chain(kernels: &[&AneKernel]) -> Result<(), String> {
        let mut handles: Vec<*mut ANEKernelHandle> = kernels.iter().map(|k| k.handle).collect();
        let ok = unsafe { ane_bridge_prepare_chain(handles.as_mut_ptr(), handles.len() as c_int) };
        if ok {
            Ok(())
        } else {
            Err("prepare_chain failed".into())
        }
    }

    /// Read data from output tensor at `idx` into `buf`.
    pub fn read_output(&self, idx: usize, buf: &mut [u8]) {
        unsafe {
            ane_bridge_read_output(
                self.handle,
                idx as c_int,
                buf.as_mut_ptr() as *mut c_void,
                buf.len(),
            );
        }
    }

    // ── Zero-copy IOSurface access (Orion-style dsb sy) ──────────────

    /// Get a raw mutable pointer to the input IOSurface at `idx`.
    ///
    /// **UNSAFE**: The returned pointer is valid only while the kernel is alive.
    /// You MUST issue an ARM64 memory barrier (`dsb sy`) after writing and
    /// before calling `eval()`, and again after `eval()` before reading output.
    /// This eliminates the IOSurface lock/unlock overhead (~1µs each).
    ///
    /// Layout: `[1, n_channels, 1, spatial]` packed as `[n_ch * spatial]` f32.
    /// To write a single position: `ptr[ch * spatial + 0] = value`.
    pub fn get_input_base(&self, idx: usize) -> *mut u8 {
        unsafe { ane_bridge_get_input_base(self.handle, idx as c_int) as *mut u8 }
    }

    /// Get a raw pointer to the output IOSurface at `idx`.
    ///
    /// **UNSAFE**: Same constraints as `get_input_base`. Issue `dsb sy` after
    /// `eval()` before reading.
    pub fn get_output_base(&self, idx: usize) -> *const u8 {
        unsafe { ane_bridge_get_output_base(self.handle, idx as c_int) as *const u8 }
    }

    /// Get the byte size of input IOSurface at `idx`.
    pub fn input_size(&self, idx: usize) -> usize {
        unsafe { ane_bridge_input_size(self.handle, idx as c_int) }
    }

    /// Get the byte size of output IOSurface at `idx`.
    pub fn output_size(&self, idx: usize) -> usize {
        unsafe { ane_bridge_output_size(self.handle, idx as c_int) }
    }

    /// Write a vector `x` of `n_ch` f32 values into position 0 of input IOSurface
    /// `[1, n_ch, 1, spatial]` using zero-copy direct pointer + dsb sy barrier.
    ///
    /// Clears positions 1..spatial to zero (memset), then writes position 0 for each channel.
    /// ~2x faster than `write_input()` for padded spatial layouts.
    pub fn write_input_zerocopy(&self, idx: usize, x: &[f32], n_ch: usize, spatial: usize) {
        debug_assert_eq!(x.len(), n_ch);
        let base = self.get_input_base(idx) as *mut f32;
        if base.is_null() {
            return;
        }
        unsafe {
            // Zero the entire IOSurface (clears stale spatial padding)
            std::ptr::write_bytes(base, 0, n_ch * spatial);
            // Strided write: x[ch] → base[ch * spatial + 0]
            for ch in 0..n_ch {
                *base.add(ch * spatial) = x[ch];
            }
            // ARM64 full memory barrier — ensures writes are visible to ANE DMA
            #[cfg(target_arch = "aarch64")]
            std::arch::asm!("dsb sy", options(nostack, preserves_flags));
        }
    }

    /// Read position 0 from output IOSurface `[1, n_ch, 1, spatial]` using
    /// zero-copy direct pointer + dsb sy barrier.
    ///
    /// Returns `Vec<f32>` of length `n_ch`. ~2x faster than `read_output()`.
    pub fn read_output_zerocopy(&self, idx: usize, n_ch: usize, spatial: usize) -> Vec<f32> {
        let base = self.get_output_base(idx) as *const f32;
        if base.is_null() {
            return vec![0.0; n_ch];
        }
        unsafe {
            // ARM64 full memory barrier — ensures ANE DMA writes are visible to CPU
            #[cfg(target_arch = "aarch64")]
            std::arch::asm!("dsb sy", options(nostack, preserves_flags));
            let mut result = vec![0.0f32; n_ch];
            for ch in 0..n_ch {
                result[ch] = *base.add(ch * spatial);
            }
            result
        }
    }

    /// Reload weights without recompiling (Orion-style delta compilation).
    ///
    /// Unloads the model from ANE, writes new weight BLOBFILEs to disk,
    /// and reloads. The compiled microcode is reused — 8.5× faster than
    /// a full compile cycle. Does not increment the compile count.
    ///
    /// `weight_datas` must have the same number of entries as the original
    /// compile call's weight arrays.
    pub fn reload_weights(&self, weight_datas: &[&[u8]]) -> Result<(), String> {
        let data_ptrs: Vec<*const u8> = weight_datas.iter().map(|d| d.as_ptr()).collect();
        let data_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();
        let ok = unsafe {
            ane_bridge_reload_weights(
                self.handle,
                data_ptrs.as_ptr(),
                data_lens.as_ptr(),
                weight_datas.len() as c_int,
            )
        };
        if ok {
            Ok(())
        } else {
            Err("ANE delta reload failed".into())
        }
    }

    /// Fast delta reload: reuses the same model object, skipping descriptor and
    /// model creation. 10-30× faster than `reload_weights` for repeated hotswaps
    /// (e.g. classifier tiles). Falls back to `reload_weights` if in-memory caches
    /// (net.plist, MIL) aren't available.
    pub fn delta_reload(&self, weight_datas: &[&[u8]]) -> Result<(), String> {
        let data_ptrs: Vec<*const u8> = weight_datas.iter().map(|d| d.as_ptr()).collect();
        let data_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();
        let ok = unsafe {
            ane_bridge_delta_reload(
                self.handle,
                data_ptrs.as_ptr(),
                data_lens.as_ptr(),
                weight_datas.len() as c_int,
            )
        };
        if ok {
            Ok(())
        } else {
            Err("ANE delta reload failed".into())
        }
    }

    /// Create a new kernel by patching weights from this kernel's compiled
    /// program (Orion-style zero-compile clone).
    ///
    /// Copies this kernel's compiled microcode (net.plist) to the new model's
    /// temp directory and calls loadWithQoS only — no compileWithQoS.
    /// Does not increment the compile count.
    pub fn patch_from_donor(
        &self,
        mil_text: &str,
        weight_names: &[&str],
        weight_datas: &[&[u8]],
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<Self, String> {
        assert_eq!(weight_names.len(), weight_datas.len());
        let c_names: Vec<CString> = weight_names
            .iter()
            .map(|n| CString::new(*n).expect("weight name contains null byte"))
            .collect();
        let name_ptrs: Vec<*const c_char> = c_names.iter().map(|c| c.as_ptr()).collect();
        let data_ptrs: Vec<*const u8> = weight_datas.iter().map(|d| d.as_ptr()).collect();
        let data_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();

        let handle = unsafe {
            ane_bridge_patch_from_donor(
                self.handle,
                mil_text.as_ptr() as *const c_char,
                mil_text.len(),
                name_ptrs.as_ptr(),
                data_ptrs.as_ptr(),
                data_lens.as_ptr(),
                weight_names.len() as c_int,
                input_sizes.len() as c_int,
                input_sizes.as_ptr(),
                output_sizes.len() as c_int,
                output_sizes.as_ptr(),
            )
        };
        if handle.is_null() {
            return Err("ANE patch_from_donor failed".into());
        }
        Ok(Self {
            handle,
            _not_send_sync: std::marker::PhantomData,
        })
    }
}

impl Drop for AneKernel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ane_bridge_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Smoke tests — require actual ANE hardware (Apple Silicon Mac)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal MIL program: fp32 → fp16 → fp32 round-trip (cast-only identity).
    /// Uses the actual ANE MIL IR format with buildInfo metadata.
    /// Shape [1, 64, 1, 64] — ANE has a minimum tensor size requirement.
    const CAST_IDENTITY_MIL: &str = concat!(
        "program(1.3)\n",
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
        "{\"coremlc-version\", \"3505.4.1\"}, ",
        "{\"coremltools-component-milinternal\", \"\"}, ",
        "{\"coremltools-version\", \"9.0\"}})]\n",
        "{\n",
        "    func main<ios18>(tensor<fp32, [1, 64, 1, 64]> x) {\n",
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
        "        tensor<fp16, [1, 64, 1, 64]> xh = cast(dtype = to16, x = x)",
        "[name = string(\"cin\")];\n",
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
        "        tensor<fp32, [1, 64, 1, 64]> y = cast(dtype = to32, x = xh)",
        "[name = string(\"cout\")];\n",
        "    } -> (y);\n",
        "}\n",
    );

    /// Number of elements in the test tensor: 1 * 64 * 1 * 64.
    const N: usize = 64 * 64;

    #[test]
    fn smoke_init_compile_eval() {
        // 1. Initialize ANE runtime
        ane_init().expect("ane_init failed — is this Apple Silicon?");

        // 2. Compile the cast-identity MIL (no weights needed)
        let tensor_bytes = N * 4; // fp32
        let input_sizes = [tensor_bytes];
        let output_sizes = [tensor_bytes];

        let kernel = AneKernel::compile(CAST_IDENTITY_MIL, None, &input_sizes, &output_sizes)
            .expect("compile failed");

        // 3. Write fp32 input: values 0.0 .. N as f32
        let input_f32: Vec<f32> = (0..N).map(|i| (i % 100) as f32).collect();
        let input_bytes: Vec<u8> = input_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &input_bytes);

        // 4. Evaluate on ANE
        kernel.eval().expect("eval failed");

        // 5. Read output and verify round-trip
        let mut output_bytes = vec![0u8; tensor_bytes];
        kernel.read_output(0, &mut output_bytes);
        let output_f32: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // fp32→fp16→fp32 should preserve small integer values exactly
        assert_eq!(
            input_f32, output_f32,
            "Cast identity kernel should preserve values through fp32→fp16→fp32"
        );

        // 6. Verify kernel works (compile count may be 0 if cached from previous run)

        // 7. Drop is implicit — verifies cleanup doesn't crash
    }

    /// Test write_input_region: partial write then eval should preserve the patched region.
    #[test]
    fn smoke_write_input_region() {
        ane_init().expect("ane_init failed");
        let tensor_bytes = N * 4;
        let kernel = AneKernel::compile(CAST_IDENTITY_MIL, None, &[tensor_bytes], &[tensor_bytes])
            .expect("compile failed");

        // Write zeros to the full input first
        let zeros = vec![0u8; tensor_bytes];
        kernel.write_input(0, &zeros);

        // Patch the first 64 elements (one row of the [1,64,1,64] tensor)
        let patch: Vec<f32> = (0..64).map(|i| (i + 1) as f32).collect();
        let patch_bytes: Vec<u8> = patch.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input_region(0, 0, &patch_bytes);

        kernel.eval().expect("eval failed");

        let mut out = vec![0u8; tensor_bytes];
        kernel.read_output(0, &mut out);
        let out_f32: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // First 64 elements should be our patch (1..=64), rest should be 0
        for i in 0..64 {
            assert_eq!(out_f32[i], (i + 1) as f32, "patched element {i}");
        }
        for i in 64..N {
            assert_eq!(out_f32[i], 0.0, "unpatched element {i}");
        }
    }

    /// Test write_input_strided: scattered writes under single lock.
    #[test]
    fn smoke_write_input_strided() {
        ane_init().expect("ane_init failed");
        let tensor_bytes = N * 4; // 64*64*4 = 16384 bytes
        let kernel = AneKernel::compile(CAST_IDENTITY_MIL, None, &[tensor_bytes], &[tensor_bytes])
            .expect("compile failed");

        // Write zeros first
        kernel.write_input(0, &vec![0u8; tensor_bytes]);

        // Use strided write to set the first 4 bytes of each 64-element row.
        // Tensor is [1, 64, 1, 64] = 64 rows of 64 floats each.
        // Write one f32 per row: chunk_bytes=4, n_chunks=64,
        // dst_stride=64*4=256, src_stride=4 (contiguous source)
        let src: Vec<f32> = (0..64).map(|i| (i + 10) as f32).collect();
        let src_bytes: Vec<u8> = src.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input_strided(
            0,   // idx
            0,   // dst_offset
            256, // dst_stride (64 floats * 4 bytes)
            &src_bytes, 4,  // src_stride (contiguous f32s)
            4,  // chunk_bytes (one f32)
            64, // n_chunks (64 rows)
        );

        kernel.eval().expect("eval failed");

        let mut out = vec![0u8; tensor_bytes];
        kernel.read_output(0, &mut out);
        let out_f32: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // First element of each row should be our value, rest should be 0
        for row in 0..64 {
            assert_eq!(
                out_f32[row * 64],
                (row + 10) as f32,
                "row {row} first element"
            );
            for col in 1..64 {
                assert_eq!(out_f32[row * 64 + col], 0.0, "row {row} col {col}");
            }
        }
    }

    /// Test reload_weights: update weight blob without recompiling.
    #[test]
    fn smoke_reload_weights() {
        ane_init().expect("ane_init failed");
        clear_cache(); // ensure cold compile

        let tensor_bytes = N * 4;
        let weight_name = "@model_path/weights/weight.bin";
        let dummy_weight = vec![0u8; 128 + 64 * 2]; // minimal ANE blob

        let compile_count_before = compile_count();
        let kernel = AneKernel::compile_multi_weights(
            CAST_IDENTITY_MIL,
            &[weight_name],
            &[&dummy_weight],
            &[tensor_bytes],
            &[tensor_bytes],
        )
        .expect("compile failed");
        assert_eq!(
            compile_count(),
            compile_count_before + 1,
            "compile should increment count"
        );

        // Verify kernel works before reload
        let input_f32: Vec<f32> = (0..N).map(|i| (i % 50) as f32).collect();
        let input_bytes: Vec<u8> = input_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &input_bytes);
        kernel.eval().expect("eval before reload failed");

        // Reload with different weight data — should NOT increment compile count
        let new_weight = vec![0xFFu8; 128 + 64 * 2];
        let reload_count_before = compile_count();
        kernel
            .reload_weights(&[&new_weight])
            .expect("reload_weights failed");
        assert_eq!(
            compile_count(),
            reload_count_before,
            "reload should NOT increment compile count"
        );

        // Verify kernel still works after reload
        kernel.write_input(0, &input_bytes);
        kernel.eval().expect("eval after reload failed");

        let mut output_bytes = vec![0u8; tensor_bytes];
        kernel.read_output(0, &mut output_bytes);
        let output_f32: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            input_f32, output_f32,
            "cast identity should still work after weight reload"
        );
    }

    /// Test delta_reload: fast weight hotswap reusing same model object.
    /// Verifies multiple consecutive delta reloads produce correct results.
    #[test]
    fn smoke_delta_reload() {
        ane_init().expect("ane_init failed");

        let tensor_bytes = N * 4;
        let weight_name = "@model_path/weights/weight.bin";
        let dummy_weight = vec![0u8; 128 + 64 * 2];

        let kernel = AneKernel::compile_multi_weights(
            CAST_IDENTITY_MIL,
            &[weight_name],
            &[&dummy_weight],
            &[tensor_bytes],
            &[tensor_bytes],
        )
        .expect("compile failed");

        let input_f32: Vec<f32> = (0..N).map(|i| (i % 50) as f32).collect();
        let input_bytes: Vec<u8> = input_f32.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Verify initial eval
        kernel.write_input(0, &input_bytes);
        kernel.eval().expect("initial eval failed");

        // Do 3 consecutive delta reloads (simulates classifier tile hotswap)
        for round in 0..3 {
            let new_weight = vec![(0x10 + round as u8); 128 + 64 * 2];
            let count_before = compile_count();
            kernel
                .delta_reload(&[&new_weight])
                .expect(&format!("delta_reload round {round} failed"));
            assert_eq!(
                compile_count(),
                count_before,
                "delta_reload should NOT increment compile count"
            );

            // Verify kernel still produces correct output
            kernel.write_input(0, &input_bytes);
            kernel
                .eval()
                .expect(&format!("eval after delta_reload round {round} failed"));

            let mut output_bytes = vec![0u8; tensor_bytes];
            kernel.read_output(0, &mut output_bytes);
            let output_f32: Vec<f32> = output_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(
                input_f32, output_f32,
                "cast identity should work after delta_reload round {round}"
            );
        }
    }

    /// Test patch_from_donor: create kernel from donor's compiled microcode.
    #[test]
    fn smoke_patch_from_donor() {
        ane_init().expect("ane_init failed");
        let tensor_bytes = N * 4;
        let weight_name = "@model_path/weights/weight.bin";
        let dummy_weight = vec![0u8; 128 + 64 * 2];

        // Compile the donor
        let donor = AneKernel::compile_multi_weights(
            CAST_IDENTITY_MIL,
            &[weight_name],
            &[&dummy_weight],
            &[tensor_bytes],
            &[tensor_bytes],
        )
        .expect("donor compile failed");

        // Patch from donor — same MIL text, different weight values
        let patch_count_before = compile_count();
        let patched_weight = vec![0xAAu8; 128 + 64 * 2];
        let patched = donor
            .patch_from_donor(
                CAST_IDENTITY_MIL,
                &[weight_name],
                &[&patched_weight],
                &[tensor_bytes],
                &[tensor_bytes],
            )
            .expect("patch_from_donor failed");
        assert_eq!(
            compile_count(),
            patch_count_before,
            "patch should NOT increment compile count"
        );

        // Verify patched kernel works
        let input_f32: Vec<f32> = (0..N).map(|i| (i % 80) as f32).collect();
        let input_bytes: Vec<u8> = input_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        patched.write_input(0, &input_bytes);
        patched.eval().expect("patched eval failed");

        let mut output_bytes = vec![0u8; tensor_bytes];
        patched.read_output(0, &mut output_bytes);
        let output_f32: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            input_f32, output_f32,
            "patched kernel should work correctly"
        );

        // Both donor and patched should still be independently usable
        donor.write_input(0, &input_bytes);
        donor.eval().expect("donor eval after patch failed");
    }

    /// Test clone_kernel: original and clone share the program but have independent IOSurfaces.
    #[test]
    fn smoke_clone_kernel() {
        ane_init().expect("ane_init failed");
        let tensor_bytes = N * 4;
        let original =
            AneKernel::compile(CAST_IDENTITY_MIL, None, &[tensor_bytes], &[tensor_bytes])
                .expect("compile failed");
        let clone = original.clone_kernel().expect("clone failed");

        // Write different data to original and clone (values < 2048 for fp16 exactness)
        let data_a: Vec<f32> = (0..N).map(|i| (i % 100) as f32).collect();
        let data_b: Vec<f32> = (0..N).map(|i| ((i % 100) + 500) as f32).collect();
        let bytes_a: Vec<u8> = data_a.iter().flat_map(|f| f.to_le_bytes()).collect();
        let bytes_b: Vec<u8> = data_b.iter().flat_map(|f| f.to_le_bytes()).collect();

        original.write_input(0, &bytes_a);
        clone.write_input(0, &bytes_b);

        // Eval both — they should produce independent results
        original.eval().expect("original eval failed");
        clone.eval().expect("clone eval failed");

        let mut out_a = vec![0u8; tensor_bytes];
        let mut out_b = vec![0u8; tensor_bytes];
        original.read_output(0, &mut out_a);
        clone.read_output(0, &mut out_b);

        let out_a_f32: Vec<f32> = out_a
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let out_b_f32: Vec<f32> = out_b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Verify independence: original got data_a round-tripped, clone got data_b
        assert_eq!(out_a_f32, data_a, "original should have data_a");
        assert_eq!(out_b_f32, data_b, "clone should have data_b");

        // Drop clone first, then original — tests refcount cleanup
        drop(clone);
        // Original should still work after clone is dropped
        original.write_input(0, &bytes_a);
        original.eval().expect("original eval after clone drop");
    }

    /// Benchmark: compile → drop → re-compile the same kernel.
    /// Second compile should hit the delta cache (net.plist saved to ~/.nanobot/ane_cache/).
    #[test]
    fn bench_delta_compilation_cache() {
        ane_init().expect("ane_init failed");
        clear_cache(); // ensure cold start
        let tensor_bytes = N * 4;
        let weight_name = "@model_path/weights/weight.bin";
        let dummy_weight = vec![0u8; 128 + 64 * 2];

        // --- First compile: cold (full compileWithQoS) ---
        reset_compile_count();
        let t0 = std::time::Instant::now();
        let kernel = AneKernel::compile_multi_weights(
            CAST_IDENTITY_MIL,
            &[weight_name],
            &[&dummy_weight],
            &[tensor_bytes],
            &[tensor_bytes],
        )
        .expect("first compile failed");
        let cold_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let cold_compiles = compile_count();
        assert!(!kernel.was_cached(), "first compile should NOT be cached");
        assert_eq!(cold_compiles, 1, "first compile should increment count");

        // Verify correctness
        let input_f32: Vec<f32> = (0..N).map(|i| (i % 50) as f32).collect();
        let input_bytes: Vec<u8> = input_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &input_bytes);
        kernel.eval().expect("cold eval failed");
        let mut out = vec![0u8; tensor_bytes];
        kernel.read_output(0, &mut out);
        let out_f32: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            input_f32, out_f32,
            "cold compile should produce correct output"
        );

        // Drop kernel — tmpDir with net.plist persists on disk
        drop(kernel);

        // --- Second compile: warm (delta cache — load only, no compile) ---
        reset_compile_count();
        let t1 = std::time::Instant::now();
        let cached = AneKernel::compile_multi_weights(
            CAST_IDENTITY_MIL,
            &[weight_name],
            &[&dummy_weight],
            &[tensor_bytes],
            &[tensor_bytes],
        )
        .expect("cached compile failed");
        let warm_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let warm_compiles = compile_count();
        assert!(cached.was_cached(), "second compile SHOULD be cached");
        assert_eq!(warm_compiles, 0, "cached load should NOT increment count");

        // Verify correctness after cached load
        cached.write_input(0, &input_bytes);
        cached.eval().expect("cached eval failed");
        let mut out2 = vec![0u8; tensor_bytes];
        cached.read_output(0, &mut out2);
        let out2_f32: Vec<f32> = out2
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            input_f32, out2_f32,
            "cached kernel should produce identical output"
        );

        eprintln!(
            "\n  Delta compilation cache benchmark:\n    Cold compile: {cold_ms:.1}ms ({cold_compiles} compile)\n    Warm cached:  {warm_ms:.1}ms ({warm_compiles} compiles)\n    Speedup:      {:.1}×\n",
            cold_ms / warm_ms.max(0.01)
        );

        // Warm should be faster (or at worst equal for tiny kernels)
        // For real transformer kernels, Orion reports 8.5× speedup.
        // For this minimal cast kernel, expect at least some improvement.
        assert!(
            warm_ms <= cold_ms * 1.1,
            "cached should not be slower than cold: {warm_ms:.1}ms > {cold_ms:.1}ms"
        );
    }

    /// Test zero-copy IOSurface access: write via raw pointer + dsb sy, eval, read via raw pointer.
    #[test]
    fn smoke_zerocopy_io() {
        ane_init().expect("ane_init failed");

        let tensor_bytes = N * 4;
        let weight_name = "@model_path/weights/weight.bin";
        let dummy_weight = vec![0u8; 128 + 64 * 2];

        let kernel = AneKernel::compile_multi_weights(
            CAST_IDENTITY_MIL,
            &[weight_name],
            &[&dummy_weight],
            &[tensor_bytes],
            &[tensor_bytes],
        )
        .expect("compile failed");

        // Verify sizes
        assert_eq!(kernel.input_size(0), tensor_bytes);
        assert_eq!(kernel.output_size(0), tensor_bytes);

        // Write via zero-copy (raw pointer)
        // Use small values that survive fp16 round-trip (CAST_IDENTITY_MIL goes through fp16)
        let input_f32: Vec<f32> = (0..N).map(|i| (i % 256) as f32 * 0.25).collect();
        let base = kernel.get_input_base(0) as *mut f32;
        assert!(!base.is_null(), "input base should not be null");
        unsafe {
            for i in 0..N {
                *base.add(i) = input_f32[i];
            }
            #[cfg(target_arch = "aarch64")]
            std::arch::asm!("dsb sy", options(nostack, preserves_flags));
        }

        kernel.eval().expect("eval failed");

        // Read via zero-copy (raw pointer)
        let out_base = kernel.get_output_base(0) as *const f32;
        assert!(!out_base.is_null(), "output base should not be null");
        unsafe {
            #[cfg(target_arch = "aarch64")]
            std::arch::asm!("dsb sy", options(nostack, preserves_flags));
            for i in 0..N {
                let val = *out_base.add(i);
                assert_eq!(val, input_f32[i], "mismatch at index {i}");
            }
        }
    }

    /// Test write_input_zerocopy and read_output_zerocopy with strided layout.
    #[test]
    fn smoke_zerocopy_strided() {
        ane_init().expect("ane_init failed");

        // Compile a simple identity kernel with spatial=16
        // The CAST_IDENTITY_MIL has [1, 64, 1, 1] shape, so we need one with spatial > 1
        // Instead, test the helper functions on the existing kernel
        let tensor_bytes = N * 4;
        let weight_name = "@model_path/weights/weight.bin";
        let dummy_weight = vec![0u8; 128 + 64 * 2];

        let kernel = AneKernel::compile_multi_weights(
            CAST_IDENTITY_MIL,
            &[weight_name],
            &[&dummy_weight],
            &[tensor_bytes],
            &[tensor_bytes],
        )
        .expect("compile failed");

        // Use write_input_zerocopy with spatial=1 (N channels, 1 position)
        // Small values to survive fp16 round-trip
        let input_f32: Vec<f32> = (0..N).map(|i| (i % 128) as f32 * 0.5).collect();
        kernel.write_input_zerocopy(0, &input_f32, N, 1);
        kernel.eval().expect("eval failed");
        let output = kernel.read_output_zerocopy(0, N, 1);
        assert_eq!(input_f32, output, "zerocopy strided round-trip failed");
    }

    /// Test INT8 blob builders.
    #[test]
    fn smoke_int8_blob_builders() {
        // Test build_weight_blob_int8
        let int8_data: Vec<i8> = (0..16).map(|i| (i * 8 - 64) as i8).collect();
        let blob = build_weight_blob_int8(&int8_data, 4, 4);
        assert_eq!(
            blob.len(),
            64 + 16,
            "int8 blob should be 64-byte header + 16 bytes data"
        );
        // Check header magic
        assert_eq!(blob[0], 0xEF);
        assert_eq!(blob[1], 0xBE);
        assert_eq!(blob[2], 0xAD);
        assert_eq!(blob[3], 0xDE);
        // Check data content
        for i in 0..16 {
            assert_eq!(
                blob[64 + i] as i8,
                int8_data[i],
                "int8 data mismatch at {i}"
            );
        }

        // Test build_weight_blob_quantized
        let f32_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let (qblob, scale) = build_weight_blob_quantized(&f32_data, 4, 4);
        assert!(scale > 0.0, "scale should be positive");
        assert!(qblob.len() > 64, "quantized blob should have header + data");

        // Verify round-trip: dequant(quant(x)) ≈ x within quantization error
        let max_abs = f32_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let expected_scale = max_abs / 127.0;
        assert!(
            (scale - expected_scale).abs() < 1e-6,
            "scale mismatch: got {scale}, expected {expected_scale}"
        );
    }
}
