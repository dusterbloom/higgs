// ane_bridge_mlmodel.m — Public CoreML `MLModel` implementation of the
// ANE lm_head bridge. See `ane_bridge_mlmodel.h` for the rationale.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <string.h>
#include <stdlib.h>

#include "ane_bridge_mlmodel.h"

// --- Handle struct ----------------------------------------------------------

struct ANEMLModelHandle {
    MLModel *model;  // Retained via ARC.
};

// --- Helpers ----------------------------------------------------------------

static char *ane_mlmodel_copy_cstr(NSString *s) {
    if (!s) { return NULL; }
    const char *utf = [s UTF8String];
    if (!utf) { return NULL; }
    size_t n = strlen(utf);
    char *buf = (char *)malloc(n + 1);
    if (!buf) { return NULL; }
    memcpy(buf, utf, n + 1);
    return buf;
}

static void ane_mlmodel_report_error(char **error_out, NSString *msg) {
    if (error_out) {
        *error_out = ane_mlmodel_copy_cstr(msg);
    }
}

// --- load ------------------------------------------------------------------

ANEMLModelHandle *ane_mlmodel_load(const char *mlmodelc_path, char **error_out) {
    if (!mlmodelc_path) {
        ane_mlmodel_report_error(error_out, @"ane_mlmodel_load: NULL path");
        return NULL;
    }
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:mlmodelc_path];
        if (!path) {
            ane_mlmodel_report_error(error_out, @"ane_mlmodel_load: invalid UTF-8 path");
            return NULL;
        }
        NSURL *url = [NSURL fileURLWithPath:path isDirectory:YES];

        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        NSError *err = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:url
                                           configuration:cfg
                                                   error:&err];
        if (!model) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_load: modelWithContentsOfURL failed: %@",
                err ? err.localizedDescription : @"unknown error"];
            ane_mlmodel_report_error(error_out, msg);
            return NULL;
        }

        ANEMLModelHandle *h = (ANEMLModelHandle *)calloc(1, sizeof(ANEMLModelHandle));
        if (!h) {
            ane_mlmodel_report_error(error_out, @"ane_mlmodel_load: calloc failed");
            return NULL;
        }
        h->model = model;
        // ARC transfers: store the strong reference in the C struct by
        // bridging to __bridge_retained so the handle keeps the model alive
        // past this autorelease pool.
        CFTypeRef retained = CFBridgingRetain(model);
        h->model = (__bridge MLModel *)retained;
        return h;
    }
}

// --- predict ---------------------------------------------------------------

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
    char **error_out)
{
    if (!handle || !handle->model) {
        ane_mlmodel_report_error(error_out, @"ane_mlmodel_predict_fp16: invalid handle");
        return false;
    }
    if (!input_name || !output_name || !x_fp16 || !y_fp16 || !x_shape || x_rank <= 0) {
        ane_mlmodel_report_error(error_out, @"ane_mlmodel_predict_fp16: invalid arguments");
        return false;
    }

    @autoreleasepool {
        NSString *inName = [NSString stringWithUTF8String:input_name];
        NSString *outName = [NSString stringWithUTF8String:output_name];
        if (!inName || !outName) {
            ane_mlmodel_report_error(error_out,
                @"ane_mlmodel_predict_fp16: invalid UTF-8 feature name");
            return false;
        }

        // --- Build the input MLMultiArray (fp16) ---
        NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:x_rank];
        size_t expected = 1;
        for (int i = 0; i < x_rank; ++i) {
            if (x_shape[i] <= 0) {
                ane_mlmodel_report_error(error_out,
                    @"ane_mlmodel_predict_fp16: non-positive dim in x_shape");
                return false;
            }
            [shape addObject:@(x_shape[i])];
            expected *= (size_t)x_shape[i];
        }
        if (expected != x_count) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16: x_count=%zu != prod(x_shape)=%zu",
                x_count, expected];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }

        NSError *err = nil;
        MLMultiArray *inArr =
            [[MLMultiArray alloc] initWithShape:shape
                                       dataType:MLMultiArrayDataTypeFloat16
                                          error:&err];
        if (!inArr) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16: MLMultiArray alloc failed: %@",
                err ? err.localizedDescription : @"unknown"];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }

        // Direct pointer copy — MLMultiArray's dataPointer is row-major
        // contiguous by default for freshly-allocated arrays.
        memcpy(inArr.dataPointer, x_fp16, x_count * sizeof(uint16_t));

        // --- Wrap the input in a feature provider ---
        MLFeatureValue *inVal = [MLFeatureValue featureValueWithMultiArray:inArr];
        NSDictionary *inDict = @{inName: inVal};
        MLDictionaryFeatureProvider *features =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inDict error:&err];
        if (!features) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16: feature provider failed: %@",
                err ? err.localizedDescription : @"unknown"];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }

        // --- Run prediction ---
        id<MLFeatureProvider> out =
            [handle->model predictionFromFeatures:features error:&err];
        if (!out) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16: predictionFromFeatures failed: %@",
                err ? err.localizedDescription : @"unknown"];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }

        MLFeatureValue *outVal = [out featureValueForName:outName];
        if (!outVal) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16: output '%@' not in prediction", outName];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }
        MLMultiArray *outArr = outVal.multiArrayValue;
        if (!outArr) {
            ane_mlmodel_report_error(error_out,
                @"ane_mlmodel_predict_fp16: output is not an MLMultiArray");
            return false;
        }
        if (outArr.dataType != MLMultiArrayDataTypeFloat16) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16: output dtype %ld, expected Float16",
                (long)outArr.dataType];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }

        size_t outCount = 1;
        for (NSNumber *d in outArr.shape) { outCount *= (size_t)d.unsignedIntegerValue; }
        if (outCount != y_count) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16: output count %zu != y_count %zu",
                outCount, y_count];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }

        // `getBytesWithHandler:` exposes a stride-aware contiguous view; for
        // fresh-from-MLModel outputs the strides are contiguous row-major, so
        // a flat memcpy is correct. We still iterate with strides to be safe
        // if the runtime ever hands us a non-contiguous view.
        NSArray<NSNumber *> *strides = outArr.strides;
        NSArray<NSNumber *> *oshape = outArr.shape;
        bool contiguous = true;
        size_t expected_stride = 1;
        for (NSInteger i = oshape.count - 1; i >= 0; --i) {
            if ((size_t)strides[i].unsignedIntegerValue != expected_stride) {
                contiguous = false;
                break;
            }
            expected_stride *= (size_t)oshape[i].unsignedIntegerValue;
        }

        if (contiguous) {
            memcpy(y_fp16, outArr.dataPointer, y_count * sizeof(uint16_t));
        } else {
            // Slow-path generic copy using MLMultiArray's subscript interface.
            const uint16_t *src = (const uint16_t *)outArr.dataPointer;
            NSUInteger rank = oshape.count;
            NSUInteger idx[16] = {0};
            if (rank > 16) {
                ane_mlmodel_report_error(error_out,
                    @"ane_mlmodel_predict_fp16: rank > 16 not supported");
                return false;
            }
            for (size_t k = 0; k < y_count; ++k) {
                NSUInteger flat = 0;
                for (NSUInteger i = 0; i < rank; ++i) {
                    flat += idx[i] * (NSUInteger)strides[i].unsignedIntegerValue;
                }
                y_fp16[k] = src[flat];
                // Increment multi-index (last dim fastest).
                for (NSInteger i = rank - 1; i >= 0; --i) {
                    idx[i]++;
                    if (idx[i] < (NSUInteger)oshape[i].unsignedIntegerValue) { break; }
                    idx[i] = 0;
                }
            }
        }
    }
    return true;
}

// --- predict (multi-output) ------------------------------------------------
//
// Shares the input-building path with `ane_mlmodel_predict_fp16`; differs
// only in how outputs are pulled from the prediction. One
// `predictionFromFeatures:` call regardless of `n_outputs` — that is the
// whole point of this entry point (see int8-e2e-decode handoff, Step 3).

static bool ane_mlmodel_copy_output_array(
    MLMultiArray *outArr, uint16_t *y_fp16, size_t y_count, NSString *outName,
    char **error_out)
{
    if (outArr.dataType != MLMultiArrayDataTypeFloat16) {
        NSString *msg = [NSString stringWithFormat:
            @"ane_mlmodel_predict_fp16_multi: '%@' dtype %ld, expected Float16",
            outName, (long)outArr.dataType];
        ane_mlmodel_report_error(error_out, msg);
        return false;
    }
    size_t outCount = 1;
    for (NSNumber *d in outArr.shape) { outCount *= (size_t)d.unsignedIntegerValue; }
    if (outCount != y_count) {
        NSString *msg = [NSString stringWithFormat:
            @"ane_mlmodel_predict_fp16_multi: '%@' count %zu != y_count %zu",
            outName, outCount, y_count];
        ane_mlmodel_report_error(error_out, msg);
        return false;
    }
    NSArray<NSNumber *> *strides = outArr.strides;
    NSArray<NSNumber *> *oshape = outArr.shape;
    bool contiguous = true;
    size_t expected_stride = 1;
    for (NSInteger i = oshape.count - 1; i >= 0; --i) {
        if ((size_t)strides[i].unsignedIntegerValue != expected_stride) {
            contiguous = false;
            break;
        }
        expected_stride *= (size_t)oshape[i].unsignedIntegerValue;
    }
    if (contiguous) {
        memcpy(y_fp16, outArr.dataPointer, y_count * sizeof(uint16_t));
        return true;
    }
    const uint16_t *src = (const uint16_t *)outArr.dataPointer;
    NSUInteger rank = oshape.count;
    NSUInteger idx[16] = {0};
    if (rank > 16) {
        ane_mlmodel_report_error(error_out,
            @"ane_mlmodel_predict_fp16_multi: rank > 16 not supported");
        return false;
    }
    for (size_t k = 0; k < y_count; ++k) {
        NSUInteger flat = 0;
        for (NSUInteger i = 0; i < rank; ++i) {
            flat += idx[i] * (NSUInteger)strides[i].unsignedIntegerValue;
        }
        y_fp16[k] = src[flat];
        for (NSInteger i = rank - 1; i >= 0; --i) {
            idx[i]++;
            if (idx[i] < (NSUInteger)oshape[i].unsignedIntegerValue) { break; }
            idx[i] = 0;
        }
    }
    return true;
}

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
    char **error_out)
{
    if (!handle || !handle->model) {
        ane_mlmodel_report_error(error_out,
            @"ane_mlmodel_predict_fp16_multi: invalid handle");
        return false;
    }
    if (!input_name || !x_fp16 || !x_shape || x_rank <= 0
        || !output_names || !y_fp16_buffers || !y_counts || n_outputs <= 0) {
        ane_mlmodel_report_error(error_out,
            @"ane_mlmodel_predict_fp16_multi: invalid arguments");
        return false;
    }

    @autoreleasepool {
        NSString *inName = [NSString stringWithUTF8String:input_name];
        if (!inName) {
            ane_mlmodel_report_error(error_out,
                @"ane_mlmodel_predict_fp16_multi: invalid UTF-8 input name");
            return false;
        }

        NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:x_rank];
        size_t expected = 1;
        for (int i = 0; i < x_rank; ++i) {
            if (x_shape[i] <= 0) {
                ane_mlmodel_report_error(error_out,
                    @"ane_mlmodel_predict_fp16_multi: non-positive dim in x_shape");
                return false;
            }
            [shape addObject:@(x_shape[i])];
            expected *= (size_t)x_shape[i];
        }
        if (expected != x_count) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16_multi: x_count=%zu != prod(x_shape)=%zu",
                x_count, expected];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }

        NSError *err = nil;
        MLMultiArray *inArr =
            [[MLMultiArray alloc] initWithShape:shape
                                       dataType:MLMultiArrayDataTypeFloat16
                                          error:&err];
        if (!inArr) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16_multi: MLMultiArray alloc failed: %@",
                err ? err.localizedDescription : @"unknown"];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }
        memcpy(inArr.dataPointer, x_fp16, x_count * sizeof(uint16_t));

        MLFeatureValue *inVal = [MLFeatureValue featureValueWithMultiArray:inArr];
        NSDictionary *inDict = @{inName: inVal};
        MLDictionaryFeatureProvider *features =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inDict error:&err];
        if (!features) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16_multi: feature provider failed: %@",
                err ? err.localizedDescription : @"unknown"];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }

        id<MLFeatureProvider> out =
            [handle->model predictionFromFeatures:features error:&err];
        if (!out) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_predict_fp16_multi: predictionFromFeatures failed: %@",
                err ? err.localizedDescription : @"unknown"];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }

        for (int i = 0; i < n_outputs; ++i) {
            if (!output_names[i] || !y_fp16_buffers[i]) {
                NSString *msg = [NSString stringWithFormat:
                    @"ane_mlmodel_predict_fp16_multi: NULL output[%d]", i];
                ane_mlmodel_report_error(error_out, msg);
                return false;
            }
            NSString *outName = [NSString stringWithUTF8String:output_names[i]];
            if (!outName) {
                NSString *msg = [NSString stringWithFormat:
                    @"ane_mlmodel_predict_fp16_multi: invalid UTF-8 output name[%d]", i];
                ane_mlmodel_report_error(error_out, msg);
                return false;
            }
            MLFeatureValue *outVal = [out featureValueForName:outName];
            if (!outVal) {
                NSString *msg = [NSString stringWithFormat:
                    @"ane_mlmodel_predict_fp16_multi: output '%@' not in prediction",
                    outName];
                ane_mlmodel_report_error(error_out, msg);
                return false;
            }
            MLMultiArray *outArr = outVal.multiArrayValue;
            if (!outArr) {
                NSString *msg = [NSString stringWithFormat:
                    @"ane_mlmodel_predict_fp16_multi: '%@' is not MLMultiArray", outName];
                ane_mlmodel_report_error(error_out, msg);
                return false;
            }
            if (!ane_mlmodel_copy_output_array(
                    outArr, y_fp16_buffers[i], y_counts[i], outName, error_out)) {
                return false;
            }
        }
    }
    return true;
}

// --- verify ANE dispatch ---------------------------------------------------
//
// Uses `MLComputePlan` (macOS 14.4+) to introspect which compute device each
// `mlprogram` operation is scheduled on under CPU_AND_NE. Gates the int8
// bridge per AB7 (scheduler threshold can silently route small shapes to
// CPU — we want to fail loud if that happens at a DFlash projection shape).

bool ane_mlmodel_verify_ane_dispatch(
    const char *mlmodelc_path,
    char **out_report,
    char **error_out)
{
    if (!mlmodelc_path) {
        ane_mlmodel_report_error(error_out,
            @"ane_mlmodel_verify_ane_dispatch: NULL path");
        return false;
    }
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:mlmodelc_path];
        if (!path) {
            ane_mlmodel_report_error(error_out,
                @"ane_mlmodel_verify_ane_dispatch: invalid UTF-8 path");
            return false;
        }
        NSURL *url = [NSURL fileURLWithPath:path isDirectory:YES];

        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        __block MLComputePlan *plan = nil;
        __block NSError *loadErr = nil;
        dispatch_semaphore_t sema = dispatch_semaphore_create(0);

        [MLComputePlan loadContentsOfURL:url
                           configuration:cfg
                       completionHandler:^(MLComputePlan *p, NSError *e) {
            plan = p;
            loadErr = e;
            dispatch_semaphore_signal(sema);
        }];
        dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);

        if (!plan) {
            NSString *msg = [NSString stringWithFormat:
                @"ane_mlmodel_verify_ane_dispatch: MLComputePlan load failed: %@",
                loadErr ? loadErr.localizedDescription : @"unknown"];
            ane_mlmodel_report_error(error_out, msg);
            return false;
        }

        MLModelStructureProgram *program = plan.modelStructure.program;
        if (!program) {
            ane_mlmodel_report_error(error_out,
                @"ane_mlmodel_verify_ane_dispatch: not an mlprogram");
            return false;
        }

        NSMutableString *report = [NSMutableString string];
        int aneCount = 0;
        int cpuCount = 0;
        int otherCount = 0;
        int totalOps = 0;

        NSDictionary<NSString *, MLModelStructureProgramFunction *> *functions =
            program.functions;
        for (NSString *fnName in functions) {
            MLModelStructureProgramFunction *fn = functions[fnName];
            MLModelStructureProgramBlock *block = fn.block;
            for (MLModelStructureProgramOperation *op in block.operations) {
                totalOps++;
                MLComputePlanDeviceUsage *usage =
                    [plan computeDeviceUsageForMLProgramOperation:op];
                NSString *opName = op.operatorName ?: @"<unnamed>";
                if (!usage) {
                    [report appendFormat:@"  %@ : <no usage>\n", opName];
                    continue;
                }
                id pref = usage.preferredComputeDevice;
                NSString *devName;
                if ([pref isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
                    devName = @"ANE"; aneCount++;
                } else if ([pref isKindOfClass:[MLCPUComputeDevice class]]) {
                    devName = @"CPU"; cpuCount++;
                } else {
                    devName = NSStringFromClass([pref class]);
                    otherCount++;
                }
                [report appendFormat:@"  %@ : %@\n", opName, devName];
            }
        }
        [report appendFormat:
            @"total_ops=%d ane=%d cpu=%d other=%d\n",
            totalOps, aneCount, cpuCount, otherCount];

        if (out_report) {
            *out_report = ane_mlmodel_copy_cstr(report);
        }
        return aneCount > 0;
    }
}

// --- free ------------------------------------------------------------------

void ane_mlmodel_free(ANEMLModelHandle *handle) {
    if (!handle) { return; }
    if (handle->model) {
        // Balance the CFBridgingRetain in load.
        CFRelease((__bridge CFTypeRef)handle->model);
        handle->model = nil;
    }
    free(handle);
}
