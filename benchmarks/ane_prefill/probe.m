// Benchmark-only public Core ML FP16 boundary. Compile with -fobjc-arc.
// All host input/output allocations and feature objects precede timed calls.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <time.h>
#include <stdio.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 5) {
            fprintf(stderr, "usage: probe model.mlpackage input.fp16 output-directory cpu|ane\n");
            return 2;
        }
        NSError *error = nil;
        NSURL *source = [NSURL fileURLWithPath:@(argv[1])];
        uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        NSURL *compiled = [MLModel compileModelAtURL:source error:&error];
        MLModelConfiguration *config = [MLModelConfiguration new];
        config.computeUnits = strcmp(argv[4], "cpu") == 0 ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndNeuralEngine;
        MLModel *model = compiled ? [MLModel modelWithContentsOfURL:compiled configuration:config error:&error] : nil;
        if (!model) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
        double loadMs = (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start) / 1e6;
        __block MLComputePlan *plan = nil;
        dispatch_semaphore_t ready = dispatch_semaphore_create(0);
        [MLComputePlan loadContentsOfURL:compiled configuration:config completionHandler:^(MLComputePlan *p, NSError *e) {
            if (e) fprintf(stderr, "compute plan: %s\n", e.description.UTF8String);
            plan = p; dispatch_semaphore_signal(ready);
        }];
        if (dispatch_semaphore_wait(ready, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)) || !plan) return 3;
        NSMutableArray *placements = [NSMutableArray array];
        for (MLModelStructureProgramOperation *op in plan.modelStructure.program.functions[@"main"].block.operations) {
            id device = [plan computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice;
            [placements addObject:@{@"op": op.operatorName, @"preferred": device ? NSStringFromClass([device class]) : @"unknown"}];
        }
        NSString *inputName = model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
        MLMultiArrayConstraint *constraint = model.modelDescription.inputDescriptionsByName[inputName].multiArrayConstraint;
        MLMultiArray *input = [[MLMultiArray alloc] initWithShape:constraint.shape dataType:MLMultiArrayDataTypeFloat16 error:&error];
        NSData *bytes = [NSData dataWithContentsOfFile:@(argv[2])];
        if (!input || bytes.length != (NSUInteger)input.count * 2) return 4;
        memcpy(input.dataPointer, bytes.bytes, bytes.length);
        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{inputName: [MLFeatureValue featureValueWithMultiArray:input]} error:&error];
        NSMutableDictionary *backings = [NSMutableDictionary dictionary];
        NSArray *names = [model.modelDescription.outputDescriptionsByName.allKeys sortedArrayUsingSelector:@selector(compare:)];
        for (NSString *name in names) {
            MLMultiArrayConstraint *outputConstraint = model.modelDescription.outputDescriptionsByName[name].multiArrayConstraint;
            MLMultiArray *output = [[MLMultiArray alloc] initWithShape:outputConstraint.shape dataType:MLMultiArrayDataTypeFloat16 error:&error];
            if (!output) return 5;
            backings[name] = output;
        }
        MLPredictionOptions *options = [MLPredictionOptions new];
        options.outputBackings = backings;
        NSMutableArray *samples = [NSMutableArray array];
        BOOL reused = YES;
        id<MLFeatureProvider> prediction = nil;
        for (int i = 0; i < 16; ++i) {
            @autoreleasepool {
                start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
                prediction = [model predictionFromFeatures:provider options:options error:&error];
                double ms = (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start) / 1e6;
                if (!prediction) { fprintf(stderr, "%s\n", error.description.UTF8String); return 6; }
                if (i >= 3) [samples addObject:@(ms)];
                for (NSString *name in names) {
                    MLMultiArray *actual = [prediction featureValueForName:name].multiArrayValue;
                    reused &= actual.dataPointer == ((MLMultiArray *)backings[name]).dataPointer;
                }
            }
        }
        NSMutableArray *outputs = [NSMutableArray array];
        for (NSString *name in names) {
            MLMultiArray *actual = [prediction featureValueForName:name].multiArrayValue;
            NSString *path = [@(argv[3]) stringByAppendingPathComponent:[name stringByAppendingString:@".fp16"]];
            // Verify canonical contiguous strides before writing raw FP16.
            NSUInteger stride = 1;
            for (NSInteger axis = actual.shape.count - 1; axis >= 0; --axis) {
                if (actual.shape[axis].unsignedIntegerValue > 1 && actual.strides[axis].unsignedIntegerValue != stride) return 7;
                stride *= actual.shape[axis].unsignedIntegerValue;
            }
            if (actual.dataType != MLMultiArrayDataTypeFloat16) return 8;
            if (![[NSData dataWithBytes:actual.dataPointer length:actual.count * 2] writeToFile:path atomically:YES]) return 9;
            [outputs addObject:@{@"name": name, @"shape": actual.shape, @"strides": actual.strides}];
        }
        NSDictionary *result = @{@"load_compile_ms": @(loadMs), @"prediction_ms": samples,
            @"output_backing_reused": @(reused), @"placements": placements, @"outputs": outputs,
            @"input_bytes": @(input.count * 2)};
        NSData *json = [NSJSONSerialization dataWithJSONObject:result options:0 error:&error];
        puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
        return 0;
    }
}
