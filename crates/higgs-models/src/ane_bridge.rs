//! Safe Rust wrappers around the ANE bridge C FFI.
//!
//! The bridge uses private Apple frameworks (`_ANEInMemoryModel`, `_ANEClient`)
//! to compile and execute MIL programs on the Apple Neural Engine.
//!
//! All types are `!Send + !Sync` because IOSurface handles are thread-bound.
//! Feature-gated behind `ane`.

#![cfg(feature = "ane")]

use std::ffi::CString;
use std::ptr;

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type ANEKernelHandle = std::ffi::c_void;

#[link(name = "ane_bridge")]
extern "C" {
    fn ane_bridge_init() -> i32;

    fn ane_bridge_compile(
        mil_text: *const i8,
        mil_len: usize,
        weight_data: *const u8,
        weight_len: usize,
        n_inputs: i32,
        input_sizes: *const usize,
        n_outputs: i32,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle;

    fn ane_bridge_compile_multi_weights(
        mil_text: *const i8,
        mil_len: usize,
        weight_names: *const *const i8,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: i32,
        n_inputs: i32,
        input_sizes: *const usize,
        n_outputs: i32,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle;

    fn ane_bridge_compile_direct(
        mil_text: *const i8,
        mil_len: usize,
        weight_names: *const *const i8,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: i32,
        n_inputs: i32,
        input_sizes: *const usize,
        n_outputs: i32,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle;

    fn ane_bridge_eval(kernel: *mut ANEKernelHandle) -> bool;

    fn ane_bridge_write_input(
        kernel: *mut ANEKernelHandle,
        idx: i32,
        data: *const std::ffi::c_void,
        bytes: usize,
    );

    fn ane_bridge_write_input_region(
        kernel: *mut ANEKernelHandle,
        idx: i32,
        offset: usize,
        data: *const std::ffi::c_void,
        bytes: usize,
    );

    fn ane_bridge_read_output(
        kernel: *mut ANEKernelHandle,
        idx: i32,
        data: *mut std::ffi::c_void,
        bytes: usize,
    );

    fn ane_bridge_clone_kernel(source: *mut ANEKernelHandle) -> *mut ANEKernelHandle;

    fn ane_bridge_share_surface(
        src: *mut ANEKernelHandle,
        out_idx: i32,
        dst: *mut ANEKernelHandle,
        in_idx: i32,
    ) -> bool;

    fn ane_bridge_eval_chain(kernels: *mut *mut ANEKernelHandle, n: i32) -> bool;

    fn ane_bridge_begin_realtime() -> bool;
    fn ane_bridge_end_realtime();
    fn ane_bridge_eval_realtime(kernel: *mut ANEKernelHandle) -> bool;
    fn ane_bridge_eval_chain_realtime(kernels: *mut *mut ANEKernelHandle, n: i32) -> bool;
    fn ane_bridge_prepare_chain(kernels: *mut *mut ANEKernelHandle, n: i32) -> bool;

    fn ane_bridge_free(kernel: *mut ANEKernelHandle);

    fn ane_bridge_get_compile_count() -> i32;
    fn ane_bridge_get_load_count() -> i32;
    fn ane_bridge_reset_compile_count();
    fn ane_bridge_set_quiet(quiet: bool);
    fn ane_bridge_clear_cache();
    fn ane_bridge_was_cached(kernel: *mut ANEKernelHandle) -> bool;

    fn ane_bridge_reload_weights(
        kernel: *mut ANEKernelHandle,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: i32,
    ) -> bool;

    fn ane_bridge_delta_reload(
        kernel: *mut ANEKernelHandle,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: i32,
    ) -> bool;

    fn ane_bridge_patch_from_donor(
        donor: *mut ANEKernelHandle,
        mil_text: *const i8,
        mil_len: usize,
        weight_names: *const *const i8,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: i32,
        n_inputs: i32,
        input_sizes: *const usize,
        n_outputs: i32,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle;

    fn ane_bridge_build_weight_blob(
        src: *const f32,
        rows: i32,
        cols: i32,
        out_len: *mut usize,
    ) -> *mut u8;

    fn ane_bridge_build_weight_blob_transposed(
        src: *const f32,
        rows: i32,
        cols: i32,
        out_len: *mut usize,
    ) -> *mut u8;

    fn ane_bridge_free_blob(ptr: *mut std::ffi::c_void);

    fn ane_bridge_gemm_f16(
        a_f16: *const u16,
        m: i32,
        k: i32,
        b_f32: *const f32,
        n: i32,
        c_f32: *mut f32,
        alpha: f32,
        beta: f32,
    );

    fn ane_bridge_get_input_base(kernel: *mut ANEKernelHandle, idx: i32) -> *mut std::ffi::c_void;
    fn ane_bridge_get_output_base(kernel: *mut ANEKernelHandle, idx: i32) -> *mut std::ffi::c_void;
    fn ane_bridge_input_size(kernel: *mut ANEKernelHandle, idx: i32) -> usize;
    fn ane_bridge_output_size(kernel: *mut ANEKernelHandle, idx: i32) -> usize;

    fn ane_bridge_build_weight_blob_int8(
        src: *const i8,
        rows: i32,
        cols: i32,
        out_len: *mut usize,
    ) -> *mut u8;

    fn ane_bridge_build_weight_blob_quantized(
        src: *const f32,
        rows: i32,
        cols: i32,
        out_scale: *mut f32,
        out_len: *mut usize,
    ) -> *mut u8;
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

/// Initialize the ANE runtime. Must be called before any other function.
/// Returns `Ok(())` on success or `Err` if the private framework is unavailable.
pub fn init() -> Result<(), &'static str> {
    #[allow(unsafe_code)]
    let rc = unsafe { ane_bridge_init() };
    if rc == 0 {
        Ok(())
    } else {
        Err("Failed to initialize ANE runtime (private framework unavailable)")
    }
}

/// A compiled ANE kernel with associated IOSurfaces.
///
/// Not `Send` or `Sync` — IOSurface handles are thread-bound.
pub struct AneKernel {
    handle: *mut ANEKernelHandle,
}

impl Drop for AneKernel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            #[allow(unsafe_code)]
            unsafe {
                ane_bridge_free(self.handle);
            }
        }
    }
}

impl AneKernel {
    /// Compile a MIL program with a single weight blob.
    pub fn compile(
        mil_text: &str,
        weight_data: Option<&[u8]>,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Option<Self> {
        let (w_ptr, w_len) = match weight_data {
            Some(data) => (data.as_ptr(), data.len()),
            None => (ptr::null(), 0),
        };

        #[allow(unsafe_code)]
        let handle = unsafe {
            ane_bridge_compile(
                mil_text.as_ptr().cast(),
                mil_text.len(),
                w_ptr,
                w_len,
                input_sizes.len() as i32,
                input_sizes.as_ptr(),
                output_sizes.len() as i32,
                output_sizes.as_ptr(),
            )
        };

        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    /// Compile with multiple named weight files.
    pub fn compile_multi(
        mil_text: &str,
        weight_names: &[CString],
        weight_datas: &[&[u8]],
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Option<Self> {
        let name_ptrs: Vec<*const i8> = weight_names.iter().map(|n| n.as_ptr()).collect();
        let data_ptrs: Vec<*const u8> = weight_datas.iter().map(|d| d.as_ptr()).collect();
        let data_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();

        #[allow(unsafe_code)]
        let handle = unsafe {
            ane_bridge_compile_multi_weights(
                mil_text.as_ptr().cast(),
                mil_text.len(),
                name_ptrs.as_ptr(),
                data_ptrs.as_ptr(),
                data_lens.as_ptr(),
                weight_datas.len() as i32,
                input_sizes.len() as i32,
                input_sizes.as_ptr(),
                output_sizes.len() as i32,
                output_sizes.as_ptr(),
            )
        };

        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    /// Compile via the direct `_ANEClient` path (supports conv1x1).
    pub fn compile_direct(
        mil_text: &str,
        weight_names: &[CString],
        weight_datas: &[&[u8]],
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Option<Self> {
        let name_ptrs: Vec<*const i8> = weight_names.iter().map(|n| n.as_ptr()).collect();
        let data_ptrs: Vec<*const u8> = weight_datas.iter().map(|d| d.as_ptr()).collect();
        let data_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();

        #[allow(unsafe_code)]
        let handle = unsafe {
            ane_bridge_compile_direct(
                mil_text.as_ptr().cast(),
                mil_text.len(),
                name_ptrs.as_ptr(),
                data_ptrs.as_ptr(),
                data_lens.as_ptr(),
                weight_datas.len() as i32,
                input_sizes.len() as i32,
                input_sizes.as_ptr(),
                output_sizes.len() as i32,
                output_sizes.as_ptr(),
            )
        };

        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    /// Run the kernel on ANE.
    pub fn eval(&self) -> bool {
        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_eval(self.handle)
        }
    }

    /// Run using the real-time dispatch path (lower latency).
    pub fn eval_realtime(&self) -> bool {
        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_eval_realtime(self.handle)
        }
    }

    /// Write data to input IOSurface.
    pub fn write_input(&self, idx: i32, data: &[u8]) {
        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_write_input(self.handle, idx, data.as_ptr().cast(), data.len());
        }
    }

    /// Write to a region of an input IOSurface (partial update).
    pub fn write_input_region(&self, idx: i32, offset: usize, data: &[u8]) {
        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_write_input_region(
                self.handle,
                idx,
                offset,
                data.as_ptr().cast(),
                data.len(),
            );
        }
    }

    /// Read data from output IOSurface.
    pub fn read_output(&self, idx: i32, buf: &mut [u8]) {
        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_read_output(self.handle, idx, buf.as_mut_ptr().cast(), buf.len());
        }
    }

    /// Wire this kernel's output to another kernel's input (zero-copy).
    pub fn share_output_to(&self, out_idx: i32, dst: &AneKernel, in_idx: i32) -> bool {
        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_share_surface(self.handle, out_idx, dst.handle, in_idx)
        }
    }

    /// Clone: shares compiled model, creates fresh IOSurfaces.
    pub fn clone_kernel(&self) -> Option<Self> {
        #[allow(unsafe_code)]
        let handle = unsafe { ane_bridge_clone_kernel(self.handle) };
        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    /// Reload weights without recompiling (reuses compiled microcode).
    pub fn reload_weights(&self, weight_datas: &[&[u8]]) -> bool {
        let data_ptrs: Vec<*const u8> = weight_datas.iter().map(|d| d.as_ptr()).collect();
        let data_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();

        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_reload_weights(
                self.handle,
                data_ptrs.as_ptr(),
                data_lens.as_ptr(),
                weight_datas.len() as i32,
            )
        }
    }

    /// Fast delta reload: reuses same model object (10-30x faster).
    pub fn delta_reload(&self, weight_datas: &[&[u8]]) -> bool {
        let data_ptrs: Vec<*const u8> = weight_datas.iter().map(|d| d.as_ptr()).collect();
        let data_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();

        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_delta_reload(
                self.handle,
                data_ptrs.as_ptr(),
                data_lens.as_ptr(),
                weight_datas.len() as i32,
            )
        }
    }

    /// Whether this kernel was loaded from the compilation cache.
    pub fn was_cached(&self) -> bool {
        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_was_cached(self.handle)
        }
    }

    /// Get raw pointer to input IOSurface (zero-copy access).
    ///
    /// # Safety
    /// Caller must issue ARM64 `dsb sy` barrier before eval and after eval.
    #[allow(unsafe_code)]
    pub unsafe fn get_input_base(&self, idx: i32) -> *mut u8 {
        ane_bridge_get_input_base(self.handle, idx).cast()
    }

    /// Get raw pointer to output IOSurface (zero-copy access).
    ///
    /// # Safety
    /// Caller must issue ARM64 `dsb sy` barrier after eval.
    #[allow(unsafe_code)]
    pub unsafe fn get_output_base(&self, idx: i32) -> *mut u8 {
        ane_bridge_get_output_base(self.handle, idx).cast()
    }

    /// Byte size of input IOSurface.
    pub fn input_size(&self, idx: i32) -> usize {
        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_input_size(self.handle, idx)
        }
    }

    /// Byte size of output IOSurface.
    pub fn output_size(&self, idx: i32) -> usize {
        #[allow(unsafe_code)]
        unsafe {
            ane_bridge_output_size(self.handle, idx)
        }
    }

    /// Get the raw handle (for chain operations).
    pub(crate) fn raw(&self) -> *mut ANEKernelHandle {
        self.handle
    }
}

// ---------------------------------------------------------------------------
// Chain evaluation
// ---------------------------------------------------------------------------

/// Evaluate multiple kernels back-to-back without intermediate reads.
pub fn eval_chain(kernels: &[&AneKernel]) -> bool {
    let mut handles: Vec<*mut ANEKernelHandle> = kernels.iter().map(|k| k.handle).collect();
    #[allow(unsafe_code)]
    unsafe {
        ane_bridge_eval_chain(handles.as_mut_ptr(), handles.len() as i32)
    }
}

/// Evaluate chain using real-time dispatch (lower latency).
pub fn eval_chain_realtime(kernels: &[&AneKernel]) -> bool {
    let mut handles: Vec<*mut ANEKernelHandle> = kernels.iter().map(|k| k.handle).collect();
    #[allow(unsafe_code)]
    unsafe {
        ane_bridge_eval_chain_realtime(handles.as_mut_ptr(), handles.len() as i32)
    }
}

/// Prepare a chain for pipelined dispatch.
pub fn prepare_chain(kernels: &[&AneKernel]) -> bool {
    let mut handles: Vec<*mut ANEKernelHandle> = kernels.iter().map(|k| k.handle).collect();
    #[allow(unsafe_code)]
    unsafe {
        ane_bridge_prepare_chain(handles.as_mut_ptr(), handles.len() as i32)
    }
}

/// Begin real-time task mode.
pub fn begin_realtime() -> bool {
    #[allow(unsafe_code)]
    unsafe {
        ane_bridge_begin_realtime()
    }
}

/// End real-time task mode.
pub fn end_realtime() {
    #[allow(unsafe_code)]
    unsafe {
        ane_bridge_end_realtime();
    }
}

// ---------------------------------------------------------------------------
// Weight blob builders
// ---------------------------------------------------------------------------

/// An ANE weight blob (fp16 format with 128-byte header).
pub struct WeightBlob {
    data: *mut u8,
    len: usize,
}

impl Drop for WeightBlob {
    fn drop(&mut self) {
        if !self.data.is_null() {
            #[allow(unsafe_code)]
            unsafe {
                ane_bridge_free_blob(self.data.cast());
            }
        }
    }
}

impl WeightBlob {
    /// Build a weight blob from f32 weights.
    pub fn from_f32(weights: &[f32], rows: i32, cols: i32) -> Option<Self> {
        let mut len = 0usize;
        #[allow(unsafe_code)]
        let data = unsafe { ane_bridge_build_weight_blob(weights.as_ptr(), rows, cols, &mut len) };
        if data.is_null() {
            None
        } else {
            Some(Self { data, len })
        }
    }

    /// Build a transposed weight blob from f32 weights.
    pub fn from_f32_transposed(weights: &[f32], rows: i32, cols: i32) -> Option<Self> {
        let mut len = 0usize;
        #[allow(unsafe_code)]
        let data = unsafe {
            ane_bridge_build_weight_blob_transposed(weights.as_ptr(), rows, cols, &mut len)
        };
        if data.is_null() {
            None
        } else {
            Some(Self { data, len })
        }
    }

    /// Build an int8 weight blob from pre-quantized int8 weights.
    pub fn from_int8(weights: &[i8], rows: i32, cols: i32) -> Option<Self> {
        let mut len = 0usize;
        #[allow(unsafe_code)]
        let data =
            unsafe { ane_bridge_build_weight_blob_int8(weights.as_ptr(), rows, cols, &mut len) };
        if data.is_null() {
            None
        } else {
            Some(Self { data, len })
        }
    }

    /// Quantize f32 weights to int8 and build blob in one step.
    /// Returns the blob and the per-tensor scale factor.
    pub fn from_f32_quantized(weights: &[f32], rows: i32, cols: i32) -> Option<(Self, f32)> {
        let mut len = 0usize;
        let mut scale = 0.0f32;
        #[allow(unsafe_code)]
        let data = unsafe {
            ane_bridge_build_weight_blob_quantized(
                weights.as_ptr(),
                rows,
                cols,
                &mut scale,
                &mut len,
            )
        };
        if data.is_null() {
            None
        } else {
            Some((Self { data, len }, scale))
        }
    }

    /// Raw byte slice of the blob.
    pub fn as_bytes(&self) -> &[u8] {
        if self.data.is_null() || self.len == 0 {
            &[]
        } else {
            #[allow(unsafe_code)]
            unsafe {
                std::slice::from_raw_parts(self.data, self.len)
            }
        }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the blob is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Get the total number of ANE compilations since init.
pub fn compile_count() -> i32 {
    #[allow(unsafe_code)]
    unsafe {
        ane_bridge_get_compile_count()
    }
}

/// Get the total number of ANE model loads.
pub fn load_count() -> i32 {
    #[allow(unsafe_code)]
    unsafe {
        ane_bridge_get_load_count()
    }
}

/// Reset the compile counter.
pub fn reset_compile_count() {
    #[allow(unsafe_code)]
    unsafe {
        ane_bridge_reset_compile_count();
    }
}

/// Suppress ANE error output to stderr.
pub fn set_quiet(quiet: bool) {
    #[allow(unsafe_code)]
    unsafe {
        ane_bridge_set_quiet(quiet);
    }
}

/// Clear the persistent compilation cache.
pub fn clear_cache() {
    #[allow(unsafe_code)]
    unsafe {
        ane_bridge_clear_cache();
    }
}

/// fp16-weight GEMM: C = alpha * A_f16 @ B_f32 + beta * C.
///
/// # Safety
/// Caller must ensure buffer sizes are correct.
#[allow(unsafe_code)]
pub unsafe fn gemm_f16(
    a_f16: &[u16],
    m: i32,
    k: i32,
    b_f32: &[f32],
    n: i32,
    c_f32: &mut [f32],
    alpha: f32,
    beta: f32,
) {
    ane_bridge_gemm_f16(
        a_f16.as_ptr(),
        m,
        k,
        b_f32.as_ptr(),
        n,
        c_f32.as_mut_ptr(),
        alpha,
        beta,
    );
}
