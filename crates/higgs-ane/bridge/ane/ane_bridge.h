// ane_bridge.h — C-callable bridge to ANE private APIs for Python ctypes
// Wraps _ANEInMemoryModel via private AppleNeuralEngine.framework

#ifndef ANE_BRIDGE_H
#define ANE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque kernel handle
typedef struct ANEKernelHandle ANEKernelHandle;

// Initialize ANE runtime (load private framework, resolve classes)
// Returns 0 on success, -1 on failure
int ane_bridge_init(void);

// Compile a MIL program with weight blobs into an ANE kernel
// mil_text: UTF-8 MIL program text
// mil_len: length of MIL text
// weight_data: raw weight blob (can be NULL)
// weight_len: length of weight blob
// n_inputs: number of input tensors
// input_sizes: array of byte sizes for each input
// n_outputs: number of output tensors
// output_sizes: array of byte sizes for each output
// Returns kernel handle or NULL on failure
ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes);

// Compile with multiple named weight files (for transformer kernels)
// weight_names: array of weight file paths (e.g. "@model_path/weights/wq.bin")
// weight_datas: array of weight data pointers
// weight_lens: array of weight data lengths
// n_weights: number of weight files
ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// Compile via _ANEClient direct path (supports conv, full MIL op set).
// Same API as ane_bridge_compile_multi_weights but uses a different
// compilation pipeline that supports conv1x1 and other ops blocked
// by _ANEInMemoryModel.compileWithQoS.
ANEKernelHandle *ane_bridge_compile_direct(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// Evaluate (run) a compiled kernel on ANE
// Returns true on success
bool ane_bridge_eval(ANEKernelHandle *kernel);

// Write data to kernel input tensor (full buffer)
void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes);

// Write data to a region of kernel input tensor (partial update).
// Only the bytes at [offset..offset+bytes) are modified; the rest of
// the IOSurface is left untouched. Use this to patch activations into
// a pre-populated weight buffer without re-copying static weights.
void ane_bridge_write_input_region(ANEKernelHandle *kernel, int idx,
                                    size_t offset, const void *data,
                                    size_t bytes);

// Write scattered chunks to kernel input tensor under a single IOSurface lock.
// Copies n_chunks blocks of chunk_bytes from src (at src_stride intervals)
// to input[idx] starting at dst_offset (at dst_stride intervals).
// Use this to patch interleaved activation rows without touching static weights.
void ane_bridge_write_input_strided(ANEKernelHandle *kernel, int idx,
                                     size_t dst_offset, size_t dst_stride,
                                     const void *src, size_t src_stride,
                                     size_t chunk_bytes, int n_chunks);

// Read data from kernel output tensor
void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes);

// Clone a kernel: shares the compiled model but creates fresh IOSurfaces.
// Use this to create per-layer kernel instances where each layer's weights
// live permanently in its own IOSurface. The clone is ref-counted with the
// source — the model is unloaded only when all clones + source are freed.
ANEKernelHandle *ane_bridge_clone_kernel(ANEKernelHandle *source);

// Wire kernel B's input[in_idx] to kernel A's output[out_idx].
// After eval(A), eval(B) will read A's output directly — no CPU memcpy.
// Rebuilds B's ANE request to reference the shared IOSurface.
// Returns true on success.
bool ane_bridge_share_surface(ANEKernelHandle *src, int out_idx,
                               ANEKernelHandle *dst, int in_idx);

// Evaluate N kernels back-to-back without reading intermediate results.
// Much faster than eval+read+write for each kernel.
// kernels: array of kernel handles
// n: number of kernels
// Returns true if ALL evaluations succeeded.
bool ane_bridge_eval_chain(ANEKernelHandle **kernels, int n);

// --- Real-time evaluation path ---

// Begin real-time task mode (lower dispatch latency).
// Must be called before eval_realtime. Returns true on success.
bool ane_bridge_begin_realtime(void);

// End real-time task mode.
void ane_bridge_end_realtime(void);

// Evaluate a single kernel using the real-time path.
// Requires begin_realtime() to have been called.
bool ane_bridge_eval_realtime(ANEKernelHandle *kernel);

// Evaluate N kernels using real-time dispatch (lower latency per dispatch).
bool ane_bridge_eval_chain_realtime(ANEKernelHandle **kernels, int n);

// Prepare a chain of kernels for pipelined ANE execution.
// Uses _ANEClient's prepareChainingWithModel: for batched dispatch.
bool ane_bridge_prepare_chain(ANEKernelHandle **kernels, int n);

// Free a compiled kernel and all associated resources
void ane_bridge_free(ANEKernelHandle *kernel);

// Get compile count (for exec() restart budgeting)
int ane_bridge_get_compile_count(void);

// Total loadWithQoS calls (both fresh and delta-cached)
int ane_bridge_get_load_count(void);

// Reset compile count
void ane_bridge_reset_compile_count(void);

// Suppress compile/load error output to stderr.
// Use around expected-fail compilation attempts to keep TUI clean.
void ane_bridge_set_quiet(bool quiet);

// Clear the persistent compilation cache (~/.nanobot/ane_cache/).
// Use in tests to ensure a cold-start compilation.
void ane_bridge_clear_cache(void);

// Returns true if this kernel was loaded from the compilation cache
// (no compileWithQoS: happened — delta compilation fast path).
bool ane_bridge_was_cached(ANEKernelHandle *kernel);

// --- Delta compilation (Orion-style weight reload without recompile) ---

// Reload weights for an already-compiled kernel.
// Unloads the model from ANE, writes new weight files to disk, and reloads.
// The compiled microcode (net.plist) is reused — no recompilation.
// weight_datas/weight_lens must match the weight count from the original compile.
// Does NOT increment the compile count.
// Returns true on success.
bool ane_bridge_reload_weights(ANEKernelHandle *kernel,
                                const uint8_t **weight_datas,
                                const size_t *weight_lens,
                                int n_weights);

// Fast delta reload: reuses the same model object, skipping descriptor/model creation.
// 10-30× faster than reload_weights for repeated hotswaps (e.g. classifier tiles).
// Unloads model, repopulates tmpDir from in-memory caches (net.plist + MIL + new weights),
// reloads same model. Falls back to reload_weights if caches aren't available.
bool ane_bridge_delta_reload(ANEKernelHandle *kernel,
                              const uint8_t **weight_datas,
                              const size_t *weight_lens,
                              int n_weights);

// Create a new kernel by patching weights from a donor's compiled program.
// The donor's net.plist (compiled microcode) is copied to the new model's
// temp directory. Only loadWithQoS: is called — no compileWithQoS:.
// Use this when the MIL structure is the same but tensor shapes may differ
// (e.g. different seq_len buckets with identical layer topology).
// Does NOT increment the compile count.
// Returns kernel handle or NULL on failure.
ANEKernelHandle *ane_bridge_patch_from_donor(
    ANEKernelHandle *donor,
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// Build a weight blob in ANE format (128-byte header + fp16 data)
// src: float32 weights [rows x cols]
// Returns allocated buffer and sets out_len. Caller must free().
uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len);

// Build a transposed weight blob in ANE format
uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
                                                   size_t *out_len);

// Free a blob allocated by ane_bridge_build_weight_blob*
void ane_bridge_free_blob(void *ptr);

// fp16-weight GEMM: C[M,N] = A_f16[M,K] @ B_f32[K,N]
// Weights (A) stored as fp16 row-major [M,K], activations (B) as fp32 row-major [K,N].
// Output (C) is fp32 row-major [M,N]. Uses tiled fp16→fp32 conversion + cblas_sgemm.
// alpha/beta: C = alpha * A @ B + beta * C
void ane_bridge_gemm_f16(
    const uint16_t *a_f16, int M, int K,
    const float *b_f32, int N,
    float *c_f32,
    float alpha, float beta);

// ── Zero-copy IOSurface access (Orion-style dsb sy) ─────────────────
// Get raw base address of input/output IOSurface without locking.
// The caller MUST issue an ARM64 memory barrier (dsb sy) before eval
// and after eval to ensure cache coherency.
// Returns NULL if kernel/idx invalid.
void *ane_bridge_get_input_base(ANEKernelHandle *kernel, int idx);
void *ane_bridge_get_output_base(ANEKernelHandle *kernel, int idx);

// Get the byte size of an input/output IOSurface
size_t ane_bridge_input_size(ANEKernelHandle *kernel, int idx);
size_t ane_bridge_output_size(ANEKernelHandle *kernel, int idx);

// ── INT8 weight blob builders ───────────────────────────────────────
// Build an int8 weight blob in ANE format (64-byte header + int8 data).
// src: int8 weights [rows x cols]
// For use with constexpr_affine_dequantize in MIL.
// Returns allocated buffer and sets out_len. Caller must free via ane_bridge_free_blob().
uint8_t *ane_bridge_build_weight_blob_int8(const int8_t *src, int rows, int cols,
                                            size_t *out_len);

// Quantize float32 weights to int8 and build ANE blob in one step.
// Computes per-tensor symmetric scale: scale = max(abs) / 127.
// Returns allocated buffer, sets out_scale and out_len. Caller must free via ane_bridge_free_blob().
uint8_t *ane_bridge_build_weight_blob_quantized(const float *src, int rows, int cols,
                                                 float *out_scale, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif // ANE_BRIDGE_H
