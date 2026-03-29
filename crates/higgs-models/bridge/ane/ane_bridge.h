// ane_bridge.h — C-callable bridge to ANE private APIs
// Wraps _ANEInMemoryModel via private AppleNeuralEngine.framework
//
// Ported from nanobot-rs for the Higgs inference server.

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
ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
    const uint8_t *weight_data, size_t weight_len,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// Compile with multiple named weight files (for transformer kernels)
ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// Compile via _ANEClient direct path (supports conv, full MIL op set)
ANEKernelHandle *ane_bridge_compile_direct(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// Evaluate (run) a compiled kernel on ANE
bool ane_bridge_eval(ANEKernelHandle *kernel);

// Write data to kernel input tensor (full buffer)
void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
    const void *data, size_t bytes);

// Write data to a region of kernel input tensor (partial update)
void ane_bridge_write_input_region(ANEKernelHandle *kernel, int idx,
    size_t offset, const void *data, size_t bytes);

// Write scattered chunks to kernel input tensor under a single IOSurface lock
void ane_bridge_write_input_strided(ANEKernelHandle *kernel, int idx,
    size_t dst_offset, size_t dst_stride,
    const void *src, size_t src_stride,
    size_t chunk_bytes, int n_chunks);

// Read data from kernel output tensor
void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
    void *data, size_t bytes);

// Clone a kernel: shares the compiled model but creates fresh IOSurfaces
ANEKernelHandle *ane_bridge_clone_kernel(ANEKernelHandle *source);

// Wire kernel B's input to kernel A's output (zero-copy IOSurface chaining)
bool ane_bridge_share_surface(ANEKernelHandle *src, int out_idx,
    ANEKernelHandle *dst, int in_idx);

// Evaluate N kernels back-to-back without reading intermediate results
bool ane_bridge_eval_chain(ANEKernelHandle **kernels, int n);

// --- Real-time evaluation path ---

bool ane_bridge_begin_realtime(void);
void ane_bridge_end_realtime(void);
bool ane_bridge_eval_realtime(ANEKernelHandle *kernel);
bool ane_bridge_eval_chain_realtime(ANEKernelHandle **kernels, int n);
bool ane_bridge_prepare_chain(ANEKernelHandle **kernels, int n);

// Free a compiled kernel and all associated resources
void ane_bridge_free(ANEKernelHandle *kernel);

// Get/reset compile count (for exec() restart budgeting)
int ane_bridge_get_compile_count(void);
int ane_bridge_get_load_count(void);
void ane_bridge_reset_compile_count(void);

// Suppress compile/load error output to stderr
void ane_bridge_set_quiet(bool quiet);

// Clear the persistent compilation cache
void ane_bridge_clear_cache(void);

// Returns true if this kernel was loaded from the compilation cache
bool ane_bridge_was_cached(ANEKernelHandle *kernel);

// --- Delta compilation (weight reload without recompile) ---

bool ane_bridge_reload_weights(ANEKernelHandle *kernel,
    const uint8_t **weight_datas, const size_t *weight_lens, int n_weights);

bool ane_bridge_delta_reload(ANEKernelHandle *kernel,
    const uint8_t **weight_datas, const size_t *weight_lens, int n_weights);

ANEKernelHandle *ane_bridge_patch_from_donor(
    ANEKernelHandle *donor,
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// --- Weight blob builders ---

uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
    size_t *out_len);

uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
    size_t *out_len);

void ane_bridge_free_blob(void *ptr);

// fp16-weight GEMM: C[M,N] = A_f16[M,K] @ B_f32[K,N]
void ane_bridge_gemm_f16(
    const uint16_t *a_f16, int M, int K,
    const float *b_f32, int N,
    float *c_f32,
    float alpha, float beta);

// --- Zero-copy IOSurface access ---

void *ane_bridge_get_input_base(ANEKernelHandle *kernel, int idx);
void *ane_bridge_get_output_base(ANEKernelHandle *kernel, int idx);
size_t ane_bridge_input_size(ANEKernelHandle *kernel, int idx);
size_t ane_bridge_output_size(ANEKernelHandle *kernel, int idx);

// --- INT8 weight blob builders ---

uint8_t *ane_bridge_build_weight_blob_int8(const int8_t *src, int rows, int cols,
    size_t *out_len);

uint8_t *ane_bridge_build_weight_blob_quantized(const float *src, int rows, int cols,
    float *out_scale, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif // ANE_BRIDGE_H
