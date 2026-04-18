// ane_bridge_mlmodel.h — Public-CoreML bridge for the ANE lm_head path.
//
// Companion to `ane_bridge.m` (private `_ANEInMemoryModel` API, used by GDN
// projections). This header exposes the public `MLModel` API so we can load
// a pre-compiled `.mlmodelc` produced by `palettize_lm_head.py`. The public
// path is the only way to use `constexpr_lut_to_dense` on ANE today — the
// private `compileWithQoS:` path rejects that op (see `probe_lut6_constexpr`
// in `qwen3_next_ane.rs`).
//
// Deliberately narrow: load one fp16 input, run prediction, read one fp16
// output. No microcode reuse, no IOSurface poking, no real-time mode.
// MLModel handles all of that internally and is thread-safe.

#ifndef ANE_BRIDGE_MLMODEL_H
#define ANE_BRIDGE_MLMODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANEMLModelHandle ANEMLModelHandle;

/// Load a compiled `.mlmodelc` bundle and configure it for CPU+ANE dispatch.
///
/// `mlmodelc_path` is an absolute path to a directory ending in `.mlmodelc`
/// (produced by `xcrun coremlcompiler compile`).
///
/// If `error_out` is non-NULL and loading fails, writes a malloc'd NUL-
/// terminated string describing the failure. Caller must free() it.
/// Returns NULL on failure.
ANEMLModelHandle *ane_mlmodel_load(const char *mlmodelc_path, char **error_out);

/// Run a single `y = f(x)` prediction with fp16 I/O.
///
/// Inputs:
///   handle      : from `ane_mlmodel_load`
///   input_name  : MIL feature name (e.g. "x")
///   x_fp16      : fp16 data, row-major, `x_count` half-floats
///   x_count     : element count (must equal the product of x_shape)
///   x_shape     : rank-`x_rank` shape array
///   x_rank      : number of dimensions in `x_shape`
///   output_name : MIL feature name (e.g. "logits")
///   y_fp16      : output buffer, filled on success (`y_count` half-floats)
///   y_count     : output element count (must match the model's output size)
///
/// If `error_out` is non-NULL and prediction fails, writes a malloc'd
/// NUL-terminated message. Caller must free() it.
///
/// Returns true on success.
bool ane_mlmodel_predict_fp16(
    ANEMLModelHandle *handle,
    const char *input_name,
    const uint16_t *x_fp16,
    size_t x_count,
    const int64_t *x_shape,
    int x_rank,
    const char *output_name,
    uint16_t *y_fp16,
    size_t y_count,
    char **error_out);

/// Multi-output variant: one `predictionFromFeatures:` call, N named outputs
/// pulled from the result. Used by the int8 fusion probe to measure whether
/// CoreML amortizes dispatch cost across outputs that share a single MIL
/// program (vs re-dispatching each op independently).
///
///   output_names : array of `n_outputs` NUL-terminated MIL feature names
///   y_fp16_buffers : array of `n_outputs` fp16 buffers, each filled on success
///   y_counts     : element counts, one per output buffer
///
/// Returns true iff every output was produced and memcpy'd into its buffer.
bool ane_mlmodel_predict_fp16_multi(
    ANEMLModelHandle *handle,
    const char *input_name,
    const uint16_t *x_fp16,
    size_t x_count,
    const int64_t *x_shape,
    int x_rank,
    const char *const *output_names,
    uint16_t *const *y_fp16_buffers,
    const size_t *y_counts,
    int n_outputs,
    char **error_out);

/// Release the MLModel and all associated resources.
void ane_mlmodel_free(ANEMLModelHandle *handle);

/// Verify an `.mlmodelc` will dispatch at least one op to the Neural Engine
/// under `MLComputeUnitsCPUAndNeuralEngine` using `MLComputePlan`. Uses the
/// *preferred* compute device (a scheduling preference, not a runtime
/// guarantee — pair with wall-clock or Instruments ANE signposts for the
/// full picture).
///
/// Inputs:
///   mlmodelc_path : absolute path to `.mlmodelc` directory
///   out_report    : non-NULL receives a malloc'd human-readable per-op
///                   device report (caller must free())
///   error_out     : non-NULL receives a malloc'd error message on failure
///
/// Returns true iff at least one `mlprogram` op prefers
/// `MLNeuralEngineComputeDevice`. False means every op preferred CPU/GPU,
/// the model has no program, or plan-load failed.
bool ane_mlmodel_verify_ane_dispatch(
    const char *mlmodelc_path,
    char **out_report,
    char **error_out);

#ifdef __cplusplus
}
#endif

#endif  // ANE_BRIDGE_MLMODEL_H
