// coreml_bridge.h — C-callable bridge for CoreML model inference
// Supports both stateless (explicit KV cache) and stateful (MLState) models.

#ifndef COREML_BRIDGE_H
#define COREML_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef struct CoreMLHandle CoreMLHandle;
typedef struct CoreMLState CoreMLState;

// ─── Model lifecycle ────────────────────────────────────────────────────

// Load a .mlpackage from disk.
// The compiled .mlmodelc is cached next to the package so reloads are fast.
// Returns NULL on failure (error logged via NSLog).
CoreMLHandle *coreml_load(const char *path);

// Return the vocab size from the model's output feature description.
int coreml_get_vocab_size(CoreMLHandle *handle);

// Return the total byte size of one KV cache buffer (fp16).
// Returns 0 for stateful models (no external KV cache).
int coreml_get_kv_cache_size(CoreMLHandle *handle);

// Return the max sequence length (KV cache dim 2) for stateless models.
// Returns 0 for stateful models.
int coreml_get_max_seq(CoreMLHandle *handle);

// Return 1 if the model has MLState (stateful KV cache), 0 otherwise.
int coreml_is_stateful(CoreMLHandle *handle);

// Free the model handle and release all associated resources.
void coreml_free(CoreMLHandle *handle);

// ─── Stateful API (MLState — no KV cache copy) ─────────────────────────

// Create a new zero-initialized state for the model.
// Returns NULL on failure or if the model is not stateful.
CoreMLState *coreml_new_state(CoreMLHandle *handle);

// Free a state object.
void coreml_free_state(CoreMLState *state);

// Run single-token decode using MLState (KV cache stays on-device).
//
// Parameters:
//   handle       - model handle from coreml_load()
//   state        - state object from coreml_new_state()
//   input_ids    - pointer to 1 int32 token id
//   position     - pointer to 1 int32 position index
//   logits_out   - caller-allocated fp16 buffer [vocab_size]
//
// Returns 0 on success, -1 on failure.
int coreml_predict_stateful(CoreMLHandle *handle,
                            CoreMLState *state,
                            const int32_t *input_ids,
                            const int32_t *position,
                            void *logits_out);

// ─── Stateless API (explicit KV cache — legacy) ────────────────────────

// Run single-token decode with explicit KV cache buffers.
int coreml_predict(CoreMLHandle *handle,
                   const int32_t *input_ids,
                   const int32_t *position,
                   const void *kv_cache_in,
                   void *kv_cache_out,
                   void *logits_out);

#ifdef __cplusplus
}
#endif

#endif // COREML_BRIDGE_H
