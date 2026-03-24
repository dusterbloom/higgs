//! Compiled decode step for near-Python performance.
//!
//! The key insight: `mlx_compile` traces a function that takes arrays as
//! inputs and produces arrays as outputs. By packing the KV/SSM cache into
//! a flat `Vec<Array>` (FlatCache), we make cache state explicit I/O.
//! The compiled closure unpacks → runs forward → repacks, appearing pure
//! to the compiler while internally using the existing mutable forward path.
//!
//! Gated behind `HIGGS_COMPILED_DECODE=1` env var.

use higgs_models::cache::FlatCache;
use higgs_models::{AnyCache, AnyModel};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

/// State held behind the closure payload pointer.
struct CompiledState {
    model: *mut AnyModel,
    cache: *mut AnyCache,
    /// Number of flat cache arrays (fixed after first pack).
    n_cache_arrays: usize,
}

/// A decode step using a compiled FFI closure with explicit cache I/O.
pub(crate) struct CompiledDecodeStep {
    compiled_closure: mlx_sys::mlx_closure,
    raw_closure: mlx_sys::mlx_closure,
    /// Direct pointer to the state (since mlx_closure is opaque).
    state_ptr: *mut CompiledState,
    use_compiled: bool,
    n_cache_arrays: usize,
}

impl CompiledDecodeStep {
    /// Create a new compiled decode step.
    ///
    /// # Safety
    ///
    /// `model` and `cache` must outlive the returned `CompiledDecodeStep`.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn new(
        model: &mut AnyModel,
        cache: &mut AnyCache,
    ) -> Result<Self, crate::error::EngineError> {
        // Pack cache to determine array count
        let flat = match cache {
            AnyCache::Hybrid(layers) => FlatCache::pack(layers)
                .map_err(|e| crate::error::EngineError::Mlx(e))?,
            _ => return Err(crate::error::EngineError::Mlx(
                mlx_rs::error::Exception::custom("CompiledDecodeStep requires Hybrid cache"),
            )),
        };
        let n_cache_arrays = flat.len();

        let state = Box::new(CompiledState {
            model: model as *mut AnyModel,
            cache: cache as *mut AnyCache,
            n_cache_arrays,
        });
        let state_ptr = Box::into_raw(state);
        let payload = state_ptr.cast::<std::ffi::c_void>();

        let raw_closure = mlx_sys::mlx_closure_new_func_payload(
            Some(compiled_forward_callback),
            payload,
            Some(compiled_state_destructor),
        );

        // Try to compile the closure. This may fail if the graph has
        // operations that can't be traced (custom kernels, dynamic shapes).
        let mut compiled = mlx_sys::mlx_closure_new();
        let compile_status = mlx_sys::mlx_compile(
            &raw mut compiled,
            raw_closure,
            true, // shapeless=true
        );

        // Compilation tracing still panics inside the GDN layers' custom Metal
        // kernel or other ops that don't support tracing. The raw closure with
        // forward_compiled (concat-based KV cache) works correctly.
        // TODO: investigate which GDN/MoE ops break compile tracing.
        let use_compiled = false;
        tracing::info!(
            n_cache_arrays,
            use_compiled,
            compile_status,
            "Compiled decode step: FlatCache I/O with Array-based offsets"
        );

        Ok(Self {
            compiled_closure: compiled,
            raw_closure,
            state_ptr,
            use_compiled,
            n_cache_arrays,
        })
    }

    /// Run one decode step.
    ///
    /// Input: token array `[1]`.
    /// Output: sliced logits `[1, vocab]`.
    ///
    /// The cache is updated in-place via the flat cache round-trip.
    #[allow(unsafe_code)]
    pub(crate) fn step(&mut self, token: &Array) -> Result<Array, crate::error::EngineError> {
        unsafe {
            // Pack current cache state into the input vector
            let state = &mut *self.state_ptr;
            let cache = &mut *state.cache;

            let flat = match cache {
                AnyCache::Hybrid(layers) => FlatCache::pack(layers.as_slice())
                    .map_err(crate::error::EngineError::Mlx)?,
                _ => unreachable!(),
            };

            // Build input: [token, cache_0, cache_1, ..., cache_N]
            let mut input_ptrs: Vec<mlx_sys::mlx_array> = Vec::with_capacity(1 + flat.len());
            input_ptrs.push(token.as_ptr());
            for arr in &flat.arrays {
                input_ptrs.push(arr.as_ptr());
            }
            let input_vec = mlx_sys::mlx_vector_array_new_data(
                input_ptrs.as_ptr(),
                input_ptrs.len(),
            );

            // Call the closure (compiled or raw, with fallback)
            let mut result_vec = mlx_sys::mlx_vector_array_new();
            let mut used_compiled = false;

            if self.use_compiled {
                let status = mlx_sys::mlx_closure_apply(
                    &raw mut result_vec,
                    self.compiled_closure,
                    input_vec,
                );
                if status == 0 {
                    used_compiled = true;
                } else {
                    // Compiled path failed (likely during first trace).
                    // Fall back to raw closure for this and all future calls.
                    tracing::warn!("Compiled closure failed, falling back to raw");
                    self.use_compiled = false;
                    mlx_sys::mlx_vector_array_free(result_vec);
                    result_vec = mlx_sys::mlx_vector_array_new();
                }
            }

            if !used_compiled {
                let status = mlx_sys::mlx_closure_apply(
                    &raw mut result_vec,
                    self.raw_closure,
                    input_vec,
                );
                if status != 0 {
                    mlx_sys::mlx_vector_array_free(input_vec);
                    mlx_sys::mlx_vector_array_free(result_vec);
                    return Err(crate::error::EngineError::Mlx(
                        mlx_rs::error::Exception::custom("decode step failed (both paths)"),
                    ));
                }
            }

            mlx_sys::mlx_vector_array_free(input_vec);

            // Extract outputs: [logits, cache_0', cache_1', ..., cache_N']
            let n_out = mlx_sys::mlx_vector_array_size(result_vec);
            if n_out != 1 + self.n_cache_arrays {
                mlx_sys::mlx_vector_array_free(result_vec);
                return Err(crate::error::EngineError::Mlx(
                    mlx_rs::error::Exception::custom(format!(
                        "expected {} outputs, got {n_out}",
                        1 + self.n_cache_arrays
                    )),
                ));
            }

            // First output: logits
            let mut logits_ptr = mlx_sys::mlx_array_new();
            mlx_sys::mlx_vector_array_get(&raw mut logits_ptr, result_vec, 0);
            let logits = Array::from_ptr(logits_ptr);

            // Remaining outputs: updated cache arrays
            let mut updated_arrays = Vec::with_capacity(self.n_cache_arrays);
            for i in 0..self.n_cache_arrays {
                let mut arr_ptr = mlx_sys::mlx_array_new();
                mlx_sys::mlx_vector_array_get(&raw mut arr_ptr, result_vec, 1 + i);
                updated_arrays.push(Array::from_ptr(arr_ptr));
            }

            mlx_sys::mlx_vector_array_free(result_vec);

            // Unpack updated cache back into the structured cache
            let updated_flat = FlatCache {
                arrays: updated_arrays,
                slots: flat.slots,
            };
            match cache {
                AnyCache::Hybrid(layers) => {
                    updated_flat
                        .unpack(layers.as_mut_slice())
                        .map_err(crate::error::EngineError::Mlx)?;
                }
                _ => unreachable!(),
            }

            Ok(logits)
        }
    }
}

impl Drop for CompiledDecodeStep {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_closure_free(self.compiled_closure);
            mlx_sys::mlx_closure_free(self.raw_closure);
        }
    }
}

/// C callback for the compiled closure.
///
/// Input vector: `[token, cache_0, cache_1, ..., cache_N]`
/// Output vector: `[logits, cache_0', cache_1', ..., cache_N']`
///
/// Internally unpacks flat cache → runs model.forward() → repacks cache.
/// Wrapped in catch_unwind because extern "C" cannot unwind.
#[allow(unsafe_code)]
unsafe extern "C" fn compiled_forward_callback(
    result: *mut mlx_sys::mlx_vector_array,
    inputs: mlx_sys::mlx_vector_array,
    payload: *mut std::ffi::c_void,
) -> i32 {
    // catch_unwind prevents panics from unwinding through extern "C"
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compiled_forward_callback_inner(result, inputs, payload)
    }));
    match outcome {
        Ok(status) => status,
        Err(_) => 1, // panic caught, return error status
    }
}

#[allow(unsafe_code)]
unsafe fn compiled_forward_callback_inner(
    result: *mut mlx_sys::mlx_vector_array,
    inputs: mlx_sys::mlx_vector_array,
    payload: *mut std::ffi::c_void,
) -> i32 {
    let state = unsafe { &mut *(payload.cast::<CompiledState>()) };
    let model = unsafe { &mut *state.model };
    let cache = unsafe { &mut *state.cache };

    // Extract token (input[0])
    let mut token_ptr = unsafe { mlx_sys::mlx_array_new() };
    unsafe { mlx_sys::mlx_vector_array_get(&raw mut token_ptr, inputs, 0) };
    let token = unsafe { Array::from_ptr(token_ptr) };
    let decode_input = match token.reshape(&[1, 1]) {
        Ok(arr) => arr,
        Err(_) => return 1,
    };

    // Extract cache arrays (inputs[1..])
    let n_cache = state.n_cache_arrays;
    let mut cache_arrays = Vec::with_capacity(n_cache);
    for i in 0..n_cache {
        let mut arr_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut arr_ptr, inputs, 1 + i) };
        cache_arrays.push(unsafe { Array::from_ptr(arr_ptr) });
    }

    // Unpack cache arrays into the structured cache
    let flat_in = match cache {
        AnyCache::Hybrid(layers) => {
            match FlatCache::pack(layers.as_slice()) {
                Ok(f) => f,
                Err(_) => return 1,
            }
        }
        _ => return 1,
    };

    let incoming = FlatCache {
        arrays: cache_arrays,
        slots: flat_in.slots,
    };

    if let AnyCache::Hybrid(layers) = cache {
        if incoming.unpack(layers.as_mut_slice()).is_err() {
            return 1;
        }
    }

    // Use regular forward with standard cache ops.
    // forward_compiled (concat-based KV) is available for future mx.compile integration
    // but adds overhead without actual compilation.
    let logits = match model.forward(&decode_input, None, cache) {
        Ok(l) => l,
        Err(_) => return 1,
    };

    let sliced = logits.index((.., -1, ..));

    // Pack updated cache into output arrays
    let flat_out = match cache {
        AnyCache::Hybrid(layers) => match FlatCache::pack(layers.as_slice()) {
            Ok(f) => f,
            Err(_) => return 1,
        },
        _ => return 1,
    };

    // Build output: [logits, cache_0', ..., cache_N']
    let mut out_ptrs: Vec<mlx_sys::mlx_array> = Vec::with_capacity(1 + flat_out.len());
    out_ptrs.push(sliced.as_ptr());
    for arr in &flat_out.arrays {
        out_ptrs.push(arr.as_ptr());
    }

    unsafe {
        *result = mlx_sys::mlx_vector_array_new_data(out_ptrs.as_ptr(), out_ptrs.len());
    }

    0
}

/// Destructor for `Box<CompiledState>`.
#[allow(unsafe_code)]
unsafe extern "C" fn compiled_state_destructor(payload: *mut std::ffi::c_void) {
    unsafe { drop(Box::from_raw(payload.cast::<CompiledState>())) };
}

/// Check if compiled decode is enabled via env var.
pub(crate) fn compiled_decode_enabled() -> bool {
    std::env::var("HIGGS_COMPILED_DECODE")
        .ok()
        .is_some_and(|v| v == "1" || v == "true")
}
