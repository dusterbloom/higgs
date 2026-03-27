// coreml_bridge.m — Objective-C implementation of the CoreML bridge
// Supports both stateless (explicit KV cache) and stateful (MLState) models.

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#include <stdint.h>
#include <string.h>
#include "coreml_bridge.h"

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------
struct CoreMLHandle {
    void *model;           // MLModel *, retained via CFBridgingRetain
    int vocab_size;
    int kv_cache_size;     // total bytes (fp16), 0 for stateful models
    int is_stateful;       // 1 if model has MLState
    // KV cache dimensions (parsed from model input description)
    int kv_ndim;
    int kv_dim[4];         // [n_layers*2, n_kv_heads, max_seq, head_dim]
};

struct CoreMLState {
    void *state;           // MLState *, retained via CFBridgingRetain
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static NSString *compiled_model_cache_path(NSString *package_path) {
    if ([package_path hasSuffix:@"/"]) {
        package_path = [package_path substringToIndex:package_path.length - 1];
    }
    return [package_path stringByAppendingString:@".mlmodelc"];
}

static NSURL *ensure_compiled_model(NSString *package_path, NSError **out_error) {
    NSString *cache_path = compiled_model_cache_path(package_path);
    NSFileManager *fm = [NSFileManager defaultManager];

    if ([fm fileExistsAtPath:cache_path]) {
        NSDictionary *cache_attrs = [fm attributesOfItemAtPath:cache_path error:nil];
        NSDictionary *src_attrs   = [fm attributesOfItemAtPath:package_path error:nil];
        NSDate *cache_date = cache_attrs[NSFileModificationDate];
        NSDate *src_date   = src_attrs[NSFileModificationDate];
        if (cache_date && src_date && [cache_date compare:src_date] != NSOrderedAscending) {
            NSLog(@"[coreml_bridge] reusing cached compiled model: %@", cache_path);
            return [NSURL fileURLWithPath:cache_path];
        }
        [fm removeItemAtPath:cache_path error:nil];
    }

    NSLog(@"[coreml_bridge] compiling model: %@", package_path);
    NSURL *src_url  = [NSURL fileURLWithPath:package_path];
    NSURL *compiled = [MLModel compileModelAtURL:src_url error:out_error];
    if (!compiled) return nil;

    NSURL *dest_url = [NSURL fileURLWithPath:cache_path];
    [fm removeItemAtURL:dest_url error:nil];
    NSError *move_err = nil;
    if (![fm moveItemAtURL:compiled toURL:dest_url error:&move_err]) {
        NSLog(@"[coreml_bridge] warning: could not cache compiled model: %@", move_err);
        return compiled;
    }

    NSLog(@"[coreml_bridge] compiled model cached at: %@", cache_path);
    return dest_url;
}

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
    return (int)[constraint.shape[1] integerValue];
}

// ---------------------------------------------------------------------------
// Model lifecycle
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

    // Detect if model is stateful (has MLState descriptors)
    int is_stateful = 0;
    if (@available(macOS 15.0, *)) {
        NSDictionary *states = model.modelDescription.stateDescriptionsByName;
        is_stateful = (states && states.count > 0) ? 1 : 0;
    }

    // Parse KV cache dimensions from model input description
    int kv_ndim = 0;
    int kv_dim[4] = {0, 0, 0, 0};
    int kv_cache_size = 0;

    if (!is_stateful) {
        MLFeatureDescription *kv_desc =
            model.modelDescription.inputDescriptionsByName[@"kv_cache"];
        if (kv_desc && kv_desc.type == MLFeatureTypeMultiArray) {
            MLMultiArrayConstraint *c = kv_desc.multiArrayConstraint;
            kv_ndim = (int)c.shape.count;
            size_t elem_count = 1;
            for (int d = 0; d < kv_ndim && d < 4; d++) {
                kv_dim[d] = (int)[c.shape[d] integerValue];
                elem_count *= (size_t)kv_dim[d];
            }
            kv_cache_size = (int)(elem_count * sizeof(uint16_t));
        } else {
            NSLog(@"[coreml_bridge] warning: no 'kv_cache' input — stateless predict won't work");
        }
    }

    NSLog(@"[coreml_bridge] loaded model, vocab_size=%d, stateful=%d, "
          @"kv_cache_bytes=%d, kv_shape=[%d,%d,%d,%d]",
          vocab_size, is_stateful, kv_cache_size,
          kv_dim[0], kv_dim[1], kv_dim[2], kv_dim[3]);

    CoreMLHandle *handle = (CoreMLHandle *)calloc(1, sizeof(CoreMLHandle));
    if (!handle) return NULL;

    handle->model = (void *)CFBridgingRetain(model);
    handle->vocab_size    = vocab_size;
    handle->kv_cache_size = kv_cache_size;
    handle->is_stateful   = is_stateful;
    handle->kv_ndim       = kv_ndim;
    for (int d = 0; d < 4; d++) handle->kv_dim[d] = kv_dim[d];

    return handle;
}

int coreml_get_vocab_size(CoreMLHandle *handle) {
    if (!handle) return -1;
    return handle->vocab_size;
}

int coreml_get_kv_cache_size(CoreMLHandle *handle) {
    if (!handle) return -1;
    return handle->kv_cache_size;
}

int coreml_get_max_seq(CoreMLHandle *handle) {
    if (!handle || handle->is_stateful) return 0;
    // max_seq is dim[2] of the kv_cache shape [n_layers*2, n_kv_heads, max_seq, head_dim]
    return (handle->kv_ndim >= 3) ? handle->kv_dim[2] : 0;
}

int coreml_is_stateful(CoreMLHandle *handle) {
    if (!handle) return 0;
    return handle->is_stateful;
}

void coreml_free(CoreMLHandle *handle) {
    if (!handle) return;
    if (handle->model) {
        CFRelease(handle->model);
        handle->model = NULL;
    }
    free(handle);
}

// ---------------------------------------------------------------------------
// Stateful API (MLState)
// ---------------------------------------------------------------------------

CoreMLState *coreml_new_state(CoreMLHandle *handle) {
    if (!handle || !handle->is_stateful) {
        NSLog(@"[coreml_bridge] coreml_new_state: model is not stateful");
        return NULL;
    }

    if (@available(macOS 15.0, *)) {
        MLModel *model = (__bridge MLModel *)handle->model;
        MLState *mlstate = [model newState];
        if (!mlstate) {
            NSLog(@"[coreml_bridge] failed to create MLState");
            return NULL;
        }

        CoreMLState *state = (CoreMLState *)calloc(1, sizeof(CoreMLState));
        if (!state) return NULL;
        state->state = (void *)CFBridgingRetain(mlstate);
        return state;
    } else {
        NSLog(@"[coreml_bridge] MLState requires macOS 15.0+");
        return NULL;
    }
}

void coreml_free_state(CoreMLState *state) {
    if (!state) return;
    if (state->state) {
        CFRelease(state->state);
        state->state = NULL;
    }
    free(state);
}

int coreml_predict_stateful(CoreMLHandle *handle,
                            CoreMLState *state,
                            const int32_t *input_ids,
                            const int32_t *position,
                            void *logits_out)
{
    if (!handle || !state || !input_ids || !position || !logits_out) {
        NSLog(@"[coreml_bridge] coreml_predict_stateful: NULL argument");
        return -1;
    }

    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            NSError *error = nil;

            // Build input "input_ids": [1, 1] Int32
            MLMultiArray *input_ids_array =
                [[MLMultiArray alloc] initWithShape:@[@1, @1]
                                          dataType:MLMultiArrayDataTypeInt32
                                             error:&error];
            if (!input_ids_array) {
                NSLog(@"[coreml_bridge] failed to create input_ids: %@", error);
                return -1;
            }
            ((int32_t *)input_ids_array.dataPointer)[0] = input_ids[0];

            // Build input "position": [1] Int32
            MLMultiArray *position_array =
                [[MLMultiArray alloc] initWithShape:@[@1]
                                          dataType:MLMultiArrayDataTypeInt32
                                             error:&error];
            if (!position_array) {
                NSLog(@"[coreml_bridge] failed to create position: %@", error);
                return -1;
            }
            ((int32_t *)position_array.dataPointer)[0] = position[0];

            // Assemble feature provider (no KV cache — it's in the state)
            NSDictionary<NSString *, MLFeatureValue *> *input_dict = @{
                @"input_ids": [MLFeatureValue featureValueWithMultiArray:input_ids_array],
                @"position":  [MLFeatureValue featureValueWithMultiArray:position_array],
            };
            MLDictionaryFeatureProvider *provider =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict
                                                                  error:&error];
            if (!provider) {
                NSLog(@"[coreml_bridge] failed to create feature provider: %@", error);
                return -1;
            }

            // Run prediction WITH state (KV cache stays on-device)
            MLModel *model = (__bridge MLModel *)handle->model;
            MLState *mlstate = (__bridge MLState *)state->state;
            id<MLFeatureProvider> output = [model predictionFromFeatures:provider
                                                             usingState:mlstate
                                                                  error:&error];
            if (!output) {
                NSLog(@"[coreml_bridge] stateful prediction failed: %@", error);
                return -1;
            }

            // Copy logits output (fp32 → fp16 due to rdar://92239209)
            MLFeatureValue *logits_fv = [output featureValueForName:@"logits"];
            if (!logits_fv || logits_fv.type != MLFeatureTypeMultiArray) {
                NSLog(@"[coreml_bridge] missing 'logits' output");
                return -1;
            }
            MLMultiArray *logits_array = logits_fv.multiArrayValue;
            NSInteger count = logits_array.count;
            if (logits_array.dataType == MLMultiArrayDataTypeFloat32) {
                const float *src = (const float *)logits_array.dataPointer;
                vImage_Buffer srcBuf = { .data = (void *)src, .height = 1, .width = (size_t)count, .rowBytes = (size_t)count * 4 };
                vImage_Buffer dstBuf = { .data = logits_out, .height = 1, .width = (size_t)count, .rowBytes = (size_t)count * 2 };
                vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0);
            } else {
                memcpy(logits_out, logits_array.dataPointer, (size_t)count * sizeof(uint16_t));
            }

            return 0;
        }
    } else {
        NSLog(@"[coreml_bridge] MLState requires macOS 15.0+");
        return -1;
    }
}

// ---------------------------------------------------------------------------
// Stateless API (explicit KV cache — legacy)
// ---------------------------------------------------------------------------

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

        // input_ids [1,1] Int32
        MLMultiArray *input_ids_array =
            [[MLMultiArray alloc] initWithShape:@[@1, @1]
                                      dataType:MLMultiArrayDataTypeInt32
                                         error:&error];
        if (!input_ids_array) {
            NSLog(@"[coreml_bridge] failed to create input_ids array: %@", error);
            return -1;
        }
        ((int32_t *)input_ids_array.dataPointer)[0] = input_ids[0];

        // position [1] Int32
        MLMultiArray *position_array =
            [[MLMultiArray alloc] initWithShape:@[@1]
                                      dataType:MLMultiArrayDataTypeInt32
                                         error:&error];
        if (!position_array) {
            NSLog(@"[coreml_bridge] failed to create position array: %@", error);
            return -1;
        }
        ((int32_t *)position_array.dataPointer)[0] = position[0];

        // kv_cache: fp16 → fp32 conversion (rdar://92239209)
        int d0 = handle->kv_dim[0], d1 = handle->kv_dim[1];
        int d2 = handle->kv_dim[2], d3 = handle->kv_dim[3];
        size_t kv_elem_count = (size_t)d0 * d1 * d2 * d3;
        NSArray<NSNumber *> *kv_shape = @[@(d0), @(d1), @(d2), @(d3)];
        float *kv_f32 = (float *)malloc(kv_elem_count * sizeof(float));
        if (!kv_f32) {
            NSLog(@"[coreml_bridge] OOM allocating kv_f32 buffer");
            return -1;
        }
        {
            vImage_Buffer srcBuf = { .data = (void *)kv_cache_in, .height = 1, .width = kv_elem_count, .rowBytes = kv_elem_count * 2 };
            vImage_Buffer dstBuf = { .data = kv_f32, .height = 1, .width = kv_elem_count, .rowBytes = kv_elem_count * 4 };
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0);
        }

        NSArray<NSNumber *> *kv_strides_f32 = @[
            @((NSInteger)((size_t)d1 * d2 * d3)),
            @((NSInteger)((size_t)d2 * d3)),
            @((NSInteger)(d3)),
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

        MLModel *model = (__bridge MLModel *)handle->model;
        id<MLFeatureProvider> output = [model predictionFromFeatures:input_provider
                                                               error:&error];
        if (!output) {
            NSLog(@"[coreml_bridge] prediction failed: %@", error);
            return -1;
        }

        // Copy logits
        MLFeatureValue *logits_fv = [output featureValueForName:@"logits"];
        if (!logits_fv || logits_fv.type != MLFeatureTypeMultiArray) {
            NSLog(@"[coreml_bridge] missing or wrong-type 'logits' output");
            return -1;
        }
        MLMultiArray *logits_array = logits_fv.multiArrayValue;
        {
            NSInteger count = logits_array.count;
            if (logits_array.dataType == MLMultiArrayDataTypeFloat32) {
                const float *src = (const float *)logits_array.dataPointer;
                vImage_Buffer srcBuf = { .data = (void *)src, .height = 1, .width = (size_t)count, .rowBytes = (size_t)count * 4 };
                vImage_Buffer dstBuf = { .data = logits_out, .height = 1, .width = (size_t)count, .rowBytes = (size_t)count * 2 };
                vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0);
            } else {
                memcpy(logits_out, logits_array.dataPointer, (size_t)count * sizeof(uint16_t));
            }
        }

        // Copy kv_cache_out
        MLFeatureValue *kv_out_fv = [output featureValueForName:@"kv_cache_out"];
        if (!kv_out_fv || kv_out_fv.type != MLFeatureTypeMultiArray) {
            NSLog(@"[coreml_bridge] missing or wrong-type 'kv_cache_out' output");
            return -1;
        }
        MLMultiArray *kv_out_array = kv_out_fv.multiArrayValue;
        {
            NSInteger count = kv_out_array.count;
            if (kv_out_array.dataType == MLMultiArrayDataTypeFloat32) {
                const float *src = (const float *)kv_out_array.dataPointer;
                vImage_Buffer srcBuf = { .data = (void *)src, .height = 1, .width = (size_t)count, .rowBytes = (size_t)count * 4 };
                vImage_Buffer dstBuf = { .data = kv_cache_out, .height = 1, .width = (size_t)count, .rowBytes = (size_t)count * 2 };
                vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0);
            } else {
                memcpy(kv_cache_out, kv_out_array.dataPointer, (size_t)count * sizeof(uint16_t));
            }
        }

        return 0;
    }
}
