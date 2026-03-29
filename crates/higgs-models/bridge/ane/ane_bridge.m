// ane_bridge.m — C-callable bridge to ANE private APIs
//
// This file wraps Apple's private AppleNeuralEngine.framework to compile and
// execute MIL programs on the Apple Neural Engine without going through CoreML.
//
// Ported from nanobot-rs (github.com/dusterbloom/nanobot-rs).
// The full implementation requires the private _ANEInMemoryModel and _ANEClient
// classes which are only available on macOS with Apple Silicon.
//
// To complete this port, copy the full implementation from:
//   nanobot-rs/bridge/ane/ane_bridge.m (refactoring/maximum-speed-with-less-code branch)
//
// Key components that need to be ported:
//   1. Framework loading via dlopen (AppleNeuralEngine.framework)
//   2. Class resolution (_ANEInMemoryModel, _ANEInMemoryModelDescriptor,
//      _ANERequest, _ANEIOSurfaceObject, _ANEClient)
//   3. IOSurface creation with write-combining (inputs) and cached (outputs) modes
//   4. Compilation via _ANEInMemoryModel.compileWithQoS and _ANEClient.compileModel
//   5. Delta compilation cache (~/.config/higgs/ane_cache/)
//   6. Real-time dispatch via _ANEClient.evaluateRealTimeWithModel
//   7. Weight blob builders (fp16 and int8 with ANE header format)

#import "ane_bridge.h"
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#import <dlfcn.h>

// ═══════════════════════════════════════════════════════════════
// SECTION 1 — Private class handles (resolved at runtime)
// ═══════════════════════════════════════════════════════════════

static Class cls_ANEInMemoryModelDescriptor = nil;
static Class cls_ANEInMemoryModel = nil;
static Class cls_ANERequest = nil;
static Class cls_ANEIOSurfaceObject = nil;
static Class cls_ANEClient = nil;

static bool g_initialized = false;
static bool g_quiet = false;
static int g_compile_count = 0;
static int g_load_count = 0;

// ═══════════════════════════════════════════════════════════════
// SECTION 2 — Kernel handle structure
// ═══════════════════════════════════════════════════════════════

struct ANEKernelHandle {
    id model;                    // _ANEInMemoryModel instance
    id request;                  // _ANERequest instance
    IOSurfaceRef *inputs;        // Input IOSurfaces
    IOSurfaceRef *outputs;       // Output IOSurfaces
    int n_inputs;
    int n_outputs;
    size_t *input_sizes;
    size_t *output_sizes;
    bool was_cached;
    // Delta compilation support
    NSString *tmpDir;
    NSData *netPlistData;
    NSString *milText;
    int n_weights;
};

// ═══════════════════════════════════════════════════════════════
// SECTION 3 — Initialization
// ═══════════════════════════════════════════════════════════════

int ane_bridge_init(void) {
    if (g_initialized) return 0;

    // Load the private ANE framework.
    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_LAZY);
    if (!handle) {
        if (!g_quiet) fprintf(stderr, "ane_bridge: failed to load AppleNeuralEngine.framework\n");
        return -1;
    }

    // Resolve private classes.
    cls_ANEInMemoryModelDescriptor = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    cls_ANEInMemoryModel = NSClassFromString(@"_ANEInMemoryModel");
    cls_ANERequest = NSClassFromString(@"_ANERequest");
    cls_ANEIOSurfaceObject = NSClassFromString(@"_ANEIOSurfaceObject");
    cls_ANEClient = NSClassFromString(@"_ANEClient");

    if (!cls_ANEInMemoryModel || !cls_ANERequest) {
        if (!g_quiet) fprintf(stderr, "ane_bridge: failed to resolve ANE classes\n");
        return -1;
    }

    g_initialized = true;
    return 0;
}

// ═══════════════════════════════════════════════════════════════
// SECTION 4 — Stub implementations
// ═══════════════════════════════════════════════════════════════
//
// TODO: Port the full implementations from nanobot-rs.
// The stubs below allow the Rust code to compile and link.
// They return failure values so ANE is gracefully unavailable
// until the full Obj-C implementation is ported.

ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
    const uint8_t *weight_data, size_t weight_len,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes) {
    // TODO: Full implementation from nanobot-rs
    if (!g_quiet) fprintf(stderr, "ane_bridge_compile: stub — port from nanobot-rs\n");
    return NULL;
}

ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes) {
    if (!g_quiet) fprintf(stderr, "ane_bridge_compile_multi_weights: stub\n");
    return NULL;
}

ANEKernelHandle *ane_bridge_compile_direct(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes) {
    if (!g_quiet) fprintf(stderr, "ane_bridge_compile_direct: stub\n");
    return NULL;
}

bool ane_bridge_eval(ANEKernelHandle *kernel) {
    return false;
}

void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
    const void *data, size_t bytes) {}

void ane_bridge_write_input_region(ANEKernelHandle *kernel, int idx,
    size_t offset, const void *data, size_t bytes) {}

void ane_bridge_write_input_strided(ANEKernelHandle *kernel, int idx,
    size_t dst_offset, size_t dst_stride,
    const void *src, size_t src_stride,
    size_t chunk_bytes, int n_chunks) {}

void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
    void *data, size_t bytes) {}

ANEKernelHandle *ane_bridge_clone_kernel(ANEKernelHandle *source) {
    return NULL;
}

bool ane_bridge_share_surface(ANEKernelHandle *src, int out_idx,
    ANEKernelHandle *dst, int in_idx) {
    return false;
}

bool ane_bridge_eval_chain(ANEKernelHandle **kernels, int n) {
    return false;
}

bool ane_bridge_begin_realtime(void) { return false; }
void ane_bridge_end_realtime(void) {}
bool ane_bridge_eval_realtime(ANEKernelHandle *kernel) { return false; }
bool ane_bridge_eval_chain_realtime(ANEKernelHandle **kernels, int n) { return false; }
bool ane_bridge_prepare_chain(ANEKernelHandle **kernels, int n) { return false; }

void ane_bridge_free(ANEKernelHandle *kernel) {
    if (!kernel) return;
    if (kernel->inputs) free(kernel->inputs);
    if (kernel->outputs) free(kernel->outputs);
    if (kernel->input_sizes) free(kernel->input_sizes);
    if (kernel->output_sizes) free(kernel->output_sizes);
    free(kernel);
}

int ane_bridge_get_compile_count(void) { return g_compile_count; }
int ane_bridge_get_load_count(void) { return g_load_count; }
void ane_bridge_reset_compile_count(void) { g_compile_count = 0; }
void ane_bridge_set_quiet(bool quiet) { g_quiet = quiet; }
void ane_bridge_clear_cache(void) {}
bool ane_bridge_was_cached(ANEKernelHandle *kernel) {
    return kernel ? kernel->was_cached : false;
}

bool ane_bridge_reload_weights(ANEKernelHandle *kernel,
    const uint8_t **weight_datas, const size_t *weight_lens, int n_weights) {
    return false;
}

bool ane_bridge_delta_reload(ANEKernelHandle *kernel,
    const uint8_t **weight_datas, const size_t *weight_lens, int n_weights) {
    return false;
}

ANEKernelHandle *ane_bridge_patch_from_donor(
    ANEKernelHandle *donor,
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes) {
    return NULL;
}

// ═══════════════════════════════════════════════════════════════
// SECTION 5 — Weight blob builders (implemented, no ANE dependency)
// ═══════════════════════════════════════════════════════════════

uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
    size_t *out_len) {
    // ANE weight format: 128-byte header + fp16 data (row-major).
    size_t n_elems = (size_t)rows * cols;
    size_t data_bytes = n_elems * 2;
    size_t total = 128 + data_bytes;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    if (!buf) return NULL;

    // Header: minimal — just store dimensions.
    uint32_t *hdr = (uint32_t *)buf;
    hdr[0] = (uint32_t)rows;
    hdr[1] = (uint32_t)cols;

    // Convert f32 → fp16.
    uint16_t *dst = (uint16_t *)(buf + 128);
    for (size_t i = 0; i < n_elems; i++) {
        // Use vImage (Accelerate) for proper f32→f16 conversion.
        vImage_Buffer src_buf = { .data = (void *)&src[i], .height = 1, .width = 1, .rowBytes = 4 };
        vImage_Buffer dst_buf = { .data = &dst[i], .height = 1, .width = 1, .rowBytes = 2 };
        vImageConvert_PlanarFtoPlanar16F(&src_buf, &dst_buf, 0);
    }

    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
    size_t *out_len) {
    // Transpose then build blob.
    size_t n = (size_t)rows * cols;
    float *transposed = (float *)malloc(n * sizeof(float));
    if (!transposed) return NULL;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            transposed[c * rows + r] = src[r * cols + c];
        }
    }

    uint8_t *result = ane_bridge_build_weight_blob(transposed, cols, rows, out_len);
    free(transposed);
    return result;
}

void ane_bridge_free_blob(void *ptr) {
    free(ptr);
}

void ane_bridge_gemm_f16(
    const uint16_t *a_f16, int M, int K,
    const float *b_f32, int N,
    float *c_f32,
    float alpha, float beta) {
    // Convert fp16 A to fp32 tile, then use cblas_sgemm.
    float *a_f32 = (float *)malloc((size_t)M * K * sizeof(float));
    if (!a_f32) return;

    for (int i = 0; i < M * K; i++) {
        vImage_Buffer src_buf = { .data = (void *)&a_f16[i], .height = 1, .width = 1, .rowBytes = 2 };
        vImage_Buffer dst_buf = { .data = &a_f32[i], .height = 1, .width = 1, .rowBytes = 4 };
        vImageConvert_Planar16FtoPlanarF(&src_buf, &dst_buf, 0);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, a_f32, K, b_f32, N, beta, c_f32, N);
    free(a_f32);
}

void *ane_bridge_get_input_base(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->n_inputs) return NULL;
    return IOSurfaceGetBaseAddress(kernel->inputs[idx]);
}

void *ane_bridge_get_output_base(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->n_outputs) return NULL;
    return IOSurfaceGetBaseAddress(kernel->outputs[idx]);
}

size_t ane_bridge_input_size(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->n_inputs) return 0;
    return kernel->input_sizes[idx];
}

size_t ane_bridge_output_size(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->n_outputs) return 0;
    return kernel->output_sizes[idx];
}

uint8_t *ane_bridge_build_weight_blob_int8(const int8_t *src, int rows, int cols,
    size_t *out_len) {
    size_t n_elems = (size_t)rows * cols;
    size_t total = 64 + n_elems;  // 64-byte header for int8
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    if (!buf) return NULL;

    uint32_t *hdr = (uint32_t *)buf;
    hdr[0] = (uint32_t)rows;
    hdr[1] = (uint32_t)cols;

    memcpy(buf + 64, src, n_elems);
    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_quantized(const float *src, int rows, int cols,
    float *out_scale, size_t *out_len) {
    size_t n = (size_t)rows * cols;

    // Compute per-tensor symmetric scale.
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > max_abs) max_abs = a;
    }
    float scale = (max_abs > 0) ? max_abs / 127.0f : 1.0f;
    *out_scale = scale;

    // Quantize to int8.
    int8_t *quantized = (int8_t *)malloc(n);
    if (!quantized) return NULL;

    float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < n; i++) {
        float v = src[i] * inv_scale;
        v = fmaxf(-127.0f, fminf(127.0f, roundf(v)));
        quantized[i] = (int8_t)v;
    }

    uint8_t *blob = ane_bridge_build_weight_blob_int8(quantized, rows, cols, out_len);
    free(quantized);
    return blob;
}
