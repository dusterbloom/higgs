// coreml_bridge.h — C-callable bridge for CoreML model inference
// Wraps MLModel/MLMultiArray to run a compiled .mlpackage from Rust FFI

#ifndef COREML_BRIDGE_H
#define COREML_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a loaded CoreML model.
// Created by coreml_load(), freed by coreml_free().
typedef struct CoreMLHandle CoreMLHandle;

// Load a .mlpackage from disk.
// The compiled .mlmodelc is cached next to the package so reloads are fast.
// Returns NULL on failure (error logged via NSLog).
CoreMLHandle *coreml_load(const char *path);

// Run single-token decode prediction.
//
// Parameters:
//   handle       - model handle from coreml_load()
//   input_ids    - pointer to 1 int32 token id
//   position     - pointer to 1 int32 position index
//   kv_cache_in  - fp16 buffer [n_layers*2, n_kv_heads, max_seq, head_dim]
//                  read-only; wrapped zero-copy as MLMultiArray input
//   kv_cache_out - caller-allocated fp16 buffer (same shape as kv_cache_in)
//                  filled with the updated KV cache after prediction
//   logits_out   - caller-allocated fp16 buffer [vocab_size]
//                  filled with next-token logits
//
// Returns 0 on success, -1 on failure (error logged via NSLog).
int coreml_predict(CoreMLHandle *handle,
                   const int32_t *input_ids,
                   const int32_t *position,
                   const void *kv_cache_in,
                   void *kv_cache_out,
                   void *logits_out);

// Return the vocab size as reported by the model's output feature description.
// Returns -1 if the handle is NULL or the dimension cannot be determined.
int coreml_get_vocab_size(CoreMLHandle *handle);

// Return the total byte size of the KV cache buffers (kv_cache_in / kv_cache_out).
// Computed as n_layers*2 * n_kv_heads * max_seq * head_dim * sizeof(fp16).
// Returns -1 if the handle is NULL.
int coreml_get_kv_cache_size(CoreMLHandle *handle);

// Free the model handle and release all associated resources.
// Passing NULL is a no-op.
void coreml_free(CoreMLHandle *handle);

#ifdef __cplusplus
}
#endif

#endif // COREML_BRIDGE_H
