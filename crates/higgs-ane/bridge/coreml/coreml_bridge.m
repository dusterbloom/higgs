// coreml_bridge.m — Objective-C implementation of the CoreML bridge
// Uses public CoreML framework APIs (MLModel, MLMultiArray, MLDictionaryFeatureProvider)

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#include <stdint.h>
#include <string.h>
#include "coreml_bridge.h"

// ---------------------------------------------------------------------------
// Internal model spec constants
// These match the expected I/O layout of the compiled draft model.
// n_layers*2 = 56 (28 layers * 2 for K and V)
// ---------------------------------------------------------------------------
#define KV_N_LAYERS2   56
#define KV_N_KV_HEADS   8
#define KV_MAX_SEQ    512
#define KV_HEAD_DIM   128

// Total number of fp16 elements in one KV cache tensor
#define KV_ELEM_COUNT  ((size_t)KV_N_LAYERS2 * KV_N_KV_HEADS * KV_MAX_SEQ * KV_HEAD_DIM)
// Total byte size of one KV cache buffer (fp16 = 2 bytes)
#define KV_BYTE_SIZE   (KV_ELEM_COUNT * sizeof(uint16_t))

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------
struct CoreMLHandle {
    void *model;       // MLModel *, retained via CFBridgingRetain
    int vocab_size;    // cached from model description on load
    int kv_cache_size; // total bytes of one kv_cache buffer
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Derive the compiled model cache path: <package_path>.mlmodelc
static NSString *compiled_model_cache_path(NSString *package_path) {
    // Strip trailing slash if present
    if ([package_path hasSuffix:@"/"]) {
        package_path = [package_path substringToIndex:package_path.length - 1];
    }
    return [package_path stringByAppendingString:@".mlmodelc"];
}

// Compile (if needed) and return the URL to the .mlmodelc directory.
// The result is cached next to the .mlpackage so repeated loads skip compilation.
static NSURL *ensure_compiled_model(NSString *package_path, NSError **out_error) {
    NSString *cache_path = compiled_model_cache_path(package_path);
    NSFileManager *fm = [NSFileManager defaultManager];

    // If the compiled model already exists and is newer than the package, reuse it.
    if ([fm fileExistsAtPath:cache_path]) {
        NSDictionary *cache_attrs = [fm attributesOfItemAtPath:cache_path error:nil];
        NSDictionary *src_attrs   = [fm attributesOfItemAtPath:package_path error:nil];
        NSDate *cache_date = cache_attrs[NSFileModificationDate];
        NSDate *src_date   = src_attrs[NSFileModificationDate];
        if (cache_date && src_date && [cache_date compare:src_date] != NSOrderedAscending) {
            NSLog(@"[coreml_bridge] reusing cached compiled model: %@", cache_path);
            return [NSURL fileURLWithPath:cache_path];
        }
        // Source is newer — remove stale cache
        [fm removeItemAtPath:cache_path error:nil];
    }

    NSLog(@"[coreml_bridge] compiling model: %@", package_path);
    NSURL *src_url  = [NSURL fileURLWithPath:package_path];
    NSURL *compiled = [MLModel compileModelAtURL:src_url error:out_error];
    if (!compiled) {
        return nil;
    }

    // Move compiled output to our stable cache location so it persists.
    NSURL *dest_url = [NSURL fileURLWithPath:cache_path];
    [fm removeItemAtURL:dest_url error:nil]; // remove any partial
    NSError *move_err = nil;
    if (![fm moveItemAtURL:compiled toURL:dest_url error:&move_err]) {
        NSLog(@"[coreml_bridge] warning: could not cache compiled model: %@", move_err);
        // Fall back to using the temp location (valid for this process lifetime)
        return compiled;
    }

    NSLog(@"[coreml_bridge] compiled model cached at: %@", cache_path);
    return dest_url;
}

// Parse vocab_size from the model's output feature description for "logits".
// Expected shape: [1, vocab_size]
static int parse_vocab_size(MLModel *model) {
    MLModelDescription *desc = model.modelDescription;
    MLFeatureDescription *logits_desc = desc.outputDescriptionsByName[@"logits"];
    if (!logits_desc) {
        NSLog(@"[coreml_bridge] warning: no 'logits' output feature found");
        return -1;
    }
    if (logits_desc.type != MLFeatureTypeMultiArray) {
        NSLog(@"[coreml_bridge] warning: 'logits' output is not a MultiArray");
        return -1;
    }
    MLMultiArrayConstraint *constraint = logits_desc.multiArrayConstraint;
    if (!constraint || constraint.shape.count < 2) {
        NSLog(@"[coreml_bridge] warning: 'logits' shape has fewer than 2 dims");
        return -1;
    }
    // shape = [1, vocab_size]
    return (int)[constraint.shape[1] integerValue];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

CoreMLHandle *coreml_load(const char *path) {
    if (!path) {
        NSLog(@"[coreml_bridge] coreml_load: path is NULL");
        return NULL;
    }

    NSString *package_path = [NSString stringWithUTF8String:path];
    NSError *error = nil;

    NSURL *compiled_url = ensure_compiled_model(package_path, &error);
    if (!compiled_url) {
        NSLog(@"[coreml_bridge] compilation failed: %@", error);
        return NULL;
    }

    // Configure prediction options to prefer ANE.
    // Setting computeUnits to .all lets CoreML choose ANE when available.
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsAll;

    MLModel *model = [MLModel modelWithContentsOfURL:compiled_url
                                       configuration:config
                                               error:&error];
    if (!model) {
        NSLog(@"[coreml_bridge] failed to load model: %@", error);
        return NULL;
    }

    int vocab_size = parse_vocab_size(model);
    NSLog(@"[coreml_bridge] loaded model, vocab_size=%d, kv_cache_bytes=%zu",
          vocab_size, KV_BYTE_SIZE);

    CoreMLHandle *handle = (CoreMLHandle *)calloc(1, sizeof(CoreMLHandle));
    if (!handle) {
        NSLog(@"[coreml_bridge] out of memory allocating handle");
        return NULL;
    }

    // Pin the MLModel to the C struct lifetime via CFBridgingRetain (+1 retain).
    // This prevents ARC from releasing the model when it goes out of scope.
    handle->model = (void *)CFBridgingRetain(model);
    handle->vocab_size    = vocab_size;
    handle->kv_cache_size = (int)KV_BYTE_SIZE;

    return handle;
}

int coreml_predict(CoreMLHandle *handle,
                   const int32_t *input_ids,
                   const int32_t *position,
                   const void *kv_cache_in,
                   void *kv_cache_out,
                   void *logits_out)
{
    if (!handle || !input_ids || !position || !kv_cache_in || !kv_cache_out || !logits_out) {
        NSLog(@"[coreml_bridge] coreml_predict: NULL argument");
        return -1;
    }

    @autoreleasepool {
        NSError *error = nil;

        // ----------------------------------------------------------------
        // Build input "input_ids": shape [1, 1], type Int32
        // ----------------------------------------------------------------
        NSArray<NSNumber *> *ids_shape = @[@1, @1];
        MLMultiArray *input_ids_array =
            [[MLMultiArray alloc] initWithShape:ids_shape
                                      dataType:MLMultiArrayDataTypeInt32
                                         error:&error];
        if (!input_ids_array) {
            NSLog(@"[coreml_bridge] failed to create input_ids array: %@", error);
            return -1;
        }
        ((int32_t *)input_ids_array.dataPointer)[0] = input_ids[0];

        // ----------------------------------------------------------------
        // Build input "position": shape [1], type Int32
        // ----------------------------------------------------------------
        NSArray<NSNumber *> *pos_shape = @[@1];
        MLMultiArray *position_array =
            [[MLMultiArray alloc] initWithShape:pos_shape
                                      dataType:MLMultiArrayDataTypeInt32
                                         error:&error];
        if (!position_array) {
            NSLog(@"[coreml_bridge] failed to create position array: %@", error);
            return -1;
        }
        ((int32_t *)position_array.dataPointer)[0] = position[0];

        // ----------------------------------------------------------------
        // Build input "kv_cache": shape [56, 8, 512, 128], type Float16
        // Zero-copy: wrap caller's buffer directly.
        //
        // MLMultiArray initWithDataPointer:shape:dataType:strides:deallocator:error:
        // does NOT copy — the caller's buffer must remain valid for the duration
        // of this call. Since we do not return until prediction completes, this
        // is safe.
        // ----------------------------------------------------------------
        NSArray<NSNumber *> *kv_shape = @[
            @(KV_N_LAYERS2), @(KV_N_KV_HEADS), @(KV_MAX_SEQ), @(KV_HEAD_DIM)
        ];

        // CoreML internally uses Float32. Convert fp16 input to fp32.
        float *kv_f32 = (float *)malloc(KV_ELEM_COUNT * sizeof(float));
        if (!kv_f32) {
            NSLog(@"[coreml_bridge] OOM allocating kv_f32 buffer");
            return -1;
        }
        {
            vImage_Buffer srcBuf = { .data = (void *)kv_cache_in, .height = 1, .width = KV_ELEM_COUNT, .rowBytes = KV_BYTE_SIZE };
            vImage_Buffer dstBuf = { .data = kv_f32, .height = 1, .width = KV_ELEM_COUNT, .rowBytes = KV_ELEM_COUNT * 4 };
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0);
        }

        // Strides for Float32 in element counts (row-major)
        NSArray<NSNumber *> *kv_strides_f32 = @[
            @((NSInteger)(KV_N_KV_HEADS * KV_MAX_SEQ * KV_HEAD_DIM)),
            @((NSInteger)(KV_MAX_SEQ * KV_HEAD_DIM)),
            @((NSInteger)(KV_HEAD_DIM)),
            @1
        ];

        MLMultiArray *kv_cache_array =
            [[MLMultiArray alloc] initWithDataPointer:kv_f32
                                                shape:kv_shape
                                             dataType:MLMultiArrayDataTypeFloat32
                                             strides:kv_strides_f32
                                         deallocator:^(void *ptr) { free(ptr); }
                                               error:&error];
        if (!kv_cache_array) {
            NSLog(@"[coreml_bridge] failed to wrap kv_cache_in: %@", error);
            free(kv_f32);
            return -1;
        }

        // ----------------------------------------------------------------
        // Assemble feature provider
        // ----------------------------------------------------------------
        NSDictionary<NSString *, MLFeatureValue *> *input_dict = @{
            @"input_ids": [MLFeatureValue featureValueWithMultiArray:input_ids_array],
            @"position":  [MLFeatureValue featureValueWithMultiArray:position_array],
            @"kv_cache":  [MLFeatureValue featureValueWithMultiArray:kv_cache_array],
        };

        MLDictionaryFeatureProvider *input_provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict
                                                              error:&error];
        if (!input_provider) {
            NSLog(@"[coreml_bridge] failed to create feature provider: %@", error);
            return -1;
        }

        // ----------------------------------------------------------------
        // Run prediction
        // ----------------------------------------------------------------
        MLModel *model = (__bridge MLModel *)handle->model;
        id<MLFeatureProvider> output = [model predictionFromFeatures:input_provider
                                                               error:&error];
        if (!output) {
            NSLog(@"[coreml_bridge] prediction failed: %@", error);
            return -1;
        }

        // ----------------------------------------------------------------
        // Copy output "logits": shape [1, vocab_size], type Float16
        // ----------------------------------------------------------------
        MLFeatureValue *logits_fv = [output featureValueForName:@"logits"];
        if (!logits_fv || logits_fv.type != MLFeatureTypeMultiArray) {
            NSLog(@"[coreml_bridge] missing or wrong-type 'logits' output");
            return -1;
        }
        MLMultiArray *logits_array = logits_fv.multiArrayValue;
        if (logits_array.count == 0) {
            NSLog(@"[coreml_bridge] 'logits' output is empty");
            return -1;
        }
        // CoreML may return Float32 even when Float16 was requested.
        // Convert to fp16 if needed before copying to caller's buffer.
        {
            NSInteger count = logits_array.count;
            if (logits_array.dataType == MLMultiArrayDataTypeFloat32) {
                const float *src = (const float *)logits_array.dataPointer;
                uint16_t *dst = (uint16_t *)logits_out;
                vImage_Buffer srcBuf = { .data = (void *)src, .height = 1, .width = (size_t)count, .rowBytes = (size_t)count * 4 };
                vImage_Buffer dstBuf = { .data = dst, .height = 1, .width = (size_t)count, .rowBytes = (size_t)count * 2 };
                vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0);
            } else {
                memcpy(logits_out, logits_array.dataPointer, (size_t)count * sizeof(uint16_t));
            }
        }

        // ----------------------------------------------------------------
        // Copy output "kv_cache_out": shape [56, 8, 512, 128]
        // ----------------------------------------------------------------
        MLFeatureValue *kv_out_fv = [output featureValueForName:@"kv_cache_out"];
        if (!kv_out_fv || kv_out_fv.type != MLFeatureTypeMultiArray) {
            NSLog(@"[coreml_bridge] missing or wrong-type 'kv_cache_out' output");
            return -1;
        }
        MLMultiArray *kv_out_array = kv_out_fv.multiArrayValue;
        if ((size_t)kv_out_array.count != KV_ELEM_COUNT) {
            NSLog(@"[coreml_bridge] kv_cache_out element count mismatch: got %ld, expected %zu",
                  (long)kv_out_array.count, KV_ELEM_COUNT);
            return -1;
        }
        {
            NSInteger count = kv_out_array.count;
            if (kv_out_array.dataType == MLMultiArrayDataTypeFloat32) {
                const float *src = (const float *)kv_out_array.dataPointer;
                uint16_t *dst = (uint16_t *)kv_cache_out;
                vImage_Buffer srcBuf = { .data = (void *)src, .height = 1, .width = (size_t)count, .rowBytes = (size_t)count * 4 };
                vImage_Buffer dstBuf = { .data = dst, .height = 1, .width = (size_t)count, .rowBytes = (size_t)count * 2 };
                vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0);
            } else {
                memcpy(kv_cache_out, kv_out_array.dataPointer, (size_t)count * sizeof(uint16_t));
            }
        }

        return 0;
    } // @autoreleasepool
}

int coreml_get_vocab_size(CoreMLHandle *handle) {
    if (!handle) return -1;
    return handle->vocab_size;
}

int coreml_get_kv_cache_size(CoreMLHandle *handle) {
    if (!handle) return -1;
    return handle->kv_cache_size;
}

void coreml_free(CoreMLHandle *handle) {
    if (!handle) return;
    // Release the retained MLModel object (balances CFBridgingRetain in load)
    if (handle->model) {
        CFRelease(handle->model);
        handle->model = NULL;
    }
    free(handle);
}
