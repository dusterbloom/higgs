// ane_bridge.m — Objective-C implementation of ANE bridge for Python ctypes
// Wraps _ANEInMemoryModel private APIs into C-callable functions

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <CommonCrypto/CommonDigest.h>
#import <mach/mach_time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <Accelerate/Accelerate.h>
#include "ane_bridge.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// --- Private class references ---
static Class g_ANEDesc = nil;
static Class g_ANEInMem = nil;
static Class g_ANEReq = nil;
static Class g_ANEIO = nil;
static bool g_initialized = false;
static int g_compile_count = 0;
static int g_load_count = 0;
static bool g_quiet = false;  // suppress compile/load error output

// Persistent cache directory for compiled net.plist files.
// ANE's unloadWithQoS: deletes its temp dir, so we copy net.plist
// to our own cache before unloading. On next compile with the same
// hexId, we restore it — skipping compileWithQoS: entirely.
static NSString *g_cache_dir = nil;

static NSString *ane_cache_dir(void) {
    if (!g_cache_dir) {
        NSString *home = NSHomeDirectory();
        g_cache_dir = [home stringByAppendingPathComponent:@".nanobot/ane_cache"];
        [[NSFileManager defaultManager] createDirectoryAtPath:g_cache_dir
            withIntermediateDirectories:YES attributes:nil error:nil];
    }
    return g_cache_dir;
}

// --- Kernel handle struct ---
struct ANEKernelHandle {
    id model;               // _ANEInMemoryModel
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    id request;             // _ANERequest
    NSString *tmpDir;
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
    int *refcount;          // shared refcount for model lifetime (NULL = legacy)
    bool loaded;            // ANE load state (for delta reload path)
    bool cached;            // true if loaded from compilation cache (no compileWithQoS:)
    bool direct;            // true if compiled via _ANEClient (conv support)
    // Weight metadata for delta reload — parallel arrays, length = nWeightFiles
    int nWeightFiles;
    char **weightRelPaths;  // relative paths within tmpDir (e.g. "weights/wq.bin")
    NSData *milData;        // cached MIL text (avoids reading model.mil from disk on reload)
    NSData *netPlistData;   // cached net.plist (compiled microcode) for delta reload
};

// --- _ANEClient direct path (supports conv, full op set) ---
static Class g_ANEClient = nil;
static Class g_ANEModelCls = nil;
static id    g_ane_client = nil;

// --- Public API ---

int ane_bridge_init(void) {
    if (g_initialized) return 0;

    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "ane_bridge: Failed to load AppleNeuralEngine.framework\n");
        return -1;
    }

    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        fprintf(stderr, "ane_bridge: Failed to resolve ANE private classes\n");
        return -1;
    }

    // Resolve _ANEClient + _ANEModel for direct compilation path (conv support)
    g_ANEClient  = NSClassFromString(@"_ANEClient");
    g_ANEModelCls = NSClassFromString(@"_ANEModel");
    if (g_ANEClient && g_ANEModelCls) {
        g_ane_client = ((id(*)(Class,SEL))objc_msgSend)(g_ANEClient, @selector(sharedConnection));
    }

    g_initialized = true;
    g_compile_count = 0;

    // Clear stale system ANE compilation cache on init.
    // The e5bundlecache can hold failed compilations that block subsequent loads.
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *cachesDir = [NSSearchPathForDirectoriesInDomains(
        NSCachesDirectory, NSUserDomainMask, YES) firstObject];
    if (cachesDir) {
        NSString *e5cache = [cachesDir stringByAppendingPathComponent:
            @"com.apple.e5rt.e5bundlecache"];
        if ([fm fileExistsAtPath:e5cache]) {
            [fm removeItemAtPath:e5cache error:nil];
        }
    }

    return 0;
}

void ane_bridge_set_quiet(bool quiet) {
    g_quiet = quiet;
}

// Default (cached) IOSurface — used for outputs where CPU reads need cache coherency.
static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

// Write-combining IOSurface — used for inputs where CPU only writes.
// Bypasses the LLC so training's IOSurface writes don't evict GPU inference data.
// ANE reads via DMA from DRAM directly, so write-combining is transparent to it.
// kIOMapWriteCombineCache = 0x0400 (IOKit/IOTypes.h)
static IOSurfaceRef create_surface_wc(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
        (id)kIOSurfaceCacheMode: @(0x0400)
    });
}

ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes)
{
    @autoreleasepool {
        if (!g_initialized) {
            fprintf(stderr, "ane_bridge: Not initialized\n");
            return NULL;
        }

        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];
        NSError *e = nil;

        // Build weight dictionary
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            wdict[name] = @{@"offset": @0, @"data": data};
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) {
            fprintf(stderr, "ane_bridge: modelWithMILText failed\n");
            return NULL;
        }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            fprintf(stderr, "ane_bridge: inMemoryModelWithDescriptor failed\n");
            return NULL;
        }

        // Pre-populate temp dir
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            // Extract filename from path like "@model_path/weights/wq.bin" -> "weights/wq.bin"
            NSString *relPath = name;
            if ([name hasPrefix:@"@model_path/"]) {
                relPath = [name substringFromIndex:12];
            }
            NSString *fullPath = [td stringByAppendingPathComponent:relPath];
            NSString *dir = [fullPath stringByDeletingLastPathComponent];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            [data writeToFile:fullPath atomically:YES];
        }

        // Delta compilation cache: if net.plist already exists from a previous
        // run (same MIL text + weight keys → same hexId → same tmpDir), skip
        // compileWithQoS: and go straight to loadWithQoS:. This is the Orion
        // delta compilation fast path — 8.5× faster per kernel.
        // Delta compilation cache: check our persistent cache for a pre-compiled
        // net.plist with this hexId. ANE's unloadWithQoS: deletes its temp dir,
        // so we maintain our own cache at ~/.nanobot/ane_cache/<hexId>/net.plist.
        NSString *plistPath = [td stringByAppendingPathComponent:@"net.plist"];
        NSString *cachePlist = [[ane_cache_dir()
            stringByAppendingPathComponent:hx]
            stringByAppendingPathComponent:@"net.plist"];
        bool cached = false;

        if ([fm fileExistsAtPath:cachePlist]) {
            // Restore net.plist from our persistent cache (exact hexId match)
            NSError *cpErr = nil;
            if ([fm copyItemAtPath:cachePlist toPath:plistPath error:&cpErr]) {
                cached = true;
            }
        } else {
            // MIL-prefix fallback: net.plist is weight-independent (Bug 7/11).
            // Search for ANY cached entry with the same MIL hash prefix.
            // hexId format: <mil_hash>_<weight_hash>_<aux_hash>
            NSString *milPrefix = [[hx componentsSeparatedByString:@"_"] firstObject];
            if (milPrefix && milPrefix.length > 0) {
                NSString *cacheDir = ane_cache_dir();
                NSArray *entries = [fm contentsOfDirectoryAtPath:cacheDir error:nil];
                for (NSString *entry in entries) {
                    if ([entry hasPrefix:milPrefix]) {
                        NSString *donorPlist = [[cacheDir
                            stringByAppendingPathComponent:entry]
                            stringByAppendingPathComponent:@"net.plist"];
                        if ([fm fileExistsAtPath:donorPlist]) {
                            NSError *cpErr = nil;
                            if ([fm copyItemAtPath:donorPlist toPath:plistPath error:&cpErr]) {
                                cached = true;
                                // Also save to exact hexId cache for next time
                                NSString *exactDir = [[cacheDir stringByAppendingPathComponent:hx]
                                    stringByDeletingLastPathComponent];
                                NSString *exactCacheDir = [cacheDir stringByAppendingPathComponent:hx];
                                [fm createDirectoryAtPath:exactCacheDir
                                    withIntermediateDirectories:YES attributes:nil error:nil];
                                [fm copyItemAtPath:donorPlist
                                    toPath:cachePlist error:nil];
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (!cached) {
            // Full compile (cold path)
            if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
                if (!g_quiet) fprintf(stderr, "ane_bridge: ANE compile failed: %s\n",
                        e ? [[e description] UTF8String] : "unknown");
                [fm removeItemAtPath:td error:nil];
                return NULL;
            }
            g_compile_count++;
        }

        // Load (with one retry after a brief pause for ANE slot reclamation)
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!loaded) {
            if (!g_quiet) fprintf(stderr, "ane_bridge: ANE load %s (retrying in 100ms): %s\n",
                    cached ? "cached" : "fresh",
                    e ? [[e description] UTF8String] : "unknown");
            if (cached) {
                // Cached net.plist may be stale — fall back to full compile
                e = nil;
                if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
                    if (!g_quiet) fprintf(stderr, "ane_bridge: ANE fallback compile failed: %s\n",
                            e ? [[e description] UTF8String] : "unknown");
                    [fm removeItemAtPath:td error:nil];
                    return NULL;
                }
                g_compile_count++;
                e = nil;
                loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            } else {
                usleep(100000); // 100ms
                e = nil;
                loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            }
        }
        if (!loaded) {
            if (!g_quiet) fprintf(stderr, "ane_bridge: ANE load failed after retry: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        g_load_count++;

        // Create kernel handle
        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        k->model = mdl;
        k->tmpDir = td;
        k->refcount = (int *)malloc(sizeof(int));
        *k->refcount = 1;
        k->loaded = true;
        k->cached = cached;
        k->nInputs = n_inputs;
        k->nOutputs = n_outputs;
        k->inputBytes = (size_t *)malloc(n_inputs * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        memcpy(k->inputBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(k->outputBytes, output_sizes, n_outputs * sizeof(size_t));

        // Store weight relative paths for delta reload
        k->nWeightFiles = n_weights;
        if (n_weights > 0) {
            k->weightRelPaths = (char **)malloc(n_weights * sizeof(char *));
            for (int i = 0; i < n_weights; i++) {
                NSString *name = [NSString stringWithUTF8String:weight_names[i]];
                NSString *relPath = name;
                if ([name hasPrefix:@"@model_path/"]) {
                    relPath = [name substringFromIndex:12];
                }
                k->weightRelPaths[i] = strdup([relPath UTF8String]);
            }
        } else {
            k->weightRelPaths = NULL;
        }

        // Cache MIL text for reload_weights (avoids fragile disk reads)
        k->milData = [milData copy];

        // Cache net.plist for delta reload (avoids disk I/O on weight hotswap)
        k->netPlistData = [NSData dataWithContentsOfFile:plistPath];

        // Create IOSurfaces
        // Inputs use write-combining: CPU only writes, ANE reads via DMA.
        // This prevents training's weight copies from polluting the LLC
        // (which would evict GPU inference working set on unified memory).
        k->ioInputs = (IOSurfaceRef *)malloc(n_inputs * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(n_outputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs; i++)
            k->ioInputs[i] = create_surface_wc(input_sizes[i]);
        for (int i = 0; i < n_outputs; i++)
            k->ioOutputs[i] = create_surface(output_sizes[i]);

        // Build request
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        return k;
    }
}

ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes) {
    if (weight_data && weight_len > 0) {
        const char *name = "@model_path/weights/weight.bin";
        return ane_bridge_compile_multi_weights(
            mil_text, mil_len,
            &name, &weight_data, &weight_len, 1,
            n_inputs, input_sizes,
            n_outputs, output_sizes);
    } else {
        return ane_bridge_compile_multi_weights(
            mil_text, mil_len,
            NULL, NULL, NULL, 0,
            n_inputs, input_sizes,
            n_outputs, output_sizes);
    }
}

/// Compile via _ANEClient direct path (supports conv, full MIL op set).
///
/// Unlike `ane_bridge_compile_multi_weights` which uses _ANEInMemoryModel
/// (restricted op subset, conv blocked), this path goes through _ANEClient
/// which calls ANECCompile with full support for conv, matmul, etc.
///
/// The calling convention and returned ANEKernelHandle are identical —
/// same IOSurface setup, same eval/write/read API.
ANEKernelHandle *ane_bridge_compile_direct(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes)
{
    @autoreleasepool {
        if (!g_initialized || !g_ane_client || !g_ANEModelCls) {
            fprintf(stderr, "ane_bridge: _ANEClient not available\n");
            return NULL;
        }

        NSFileManager *fm = [NSFileManager defaultManager];
        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];

        // Create unique bundle directory based on MIL hash
        CC_SHA256_CTX sha;
        CC_SHA256_Init(&sha);
        CC_SHA256_Update(&sha, mil_text, (CC_LONG)mil_len);
        for (int i = 0; i < n_weights; i++)
            CC_SHA256_Update(&sha, weight_datas[i], (CC_LONG)weight_lens[i]);
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256_Final(digest, &sha);
        NSMutableString *hashStr = [NSMutableString stringWithCapacity:64];
        for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++)
            [hashStr appendFormat:@"%02x", digest[i]];
        NSString *programKey = [NSString stringWithFormat:@"nanobot_direct_%@", hashStr];

        NSString *bundleDir = [NSTemporaryDirectory()
            stringByAppendingPathComponent:programKey];
        NSString *modelDir = [bundleDir stringByAppendingPathComponent:@"model"];
        [fm createDirectoryAtPath:[modelDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];

        // Write MIL text
        [milData writeToFile:[modelDir stringByAppendingPathComponent:@"model.mil"]
                  atomically:NO];

        // Write weight BLOBFILEs
        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            NSString *relPath = name;
            if ([name hasPrefix:@"@model_path/"]) {
                relPath = [name substringFromIndex:12];
            }
            NSString *fullPath = [modelDir stringByAppendingPathComponent:relPath];
            NSString *dir = [fullPath stringByDeletingLastPathComponent];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES
                           attributes:nil error:nil];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            [data writeToFile:fullPath atomically:NO];
        }

        // Create _ANEModel from bundle URL
        NSURL *bundleURL = [NSURL fileURLWithPath:bundleDir];
        id mdl = ((id(*)(Class,SEL,id,id))objc_msgSend)(
            g_ANEModelCls, @selector(modelAtURL:key:),
            bundleURL, programKey);
        if (!mdl) {
            fprintf(stderr, "ane_bridge: _ANEModel modelAtURL failed\n");
            return NULL;
        }

        // Compile via _ANEClient (full op support — conv works here)
        NSError *e = nil;
        NSDictionary *compileOpts = @{
            @"kANEFModelType": @"kANEFModelMIL",
            @"NetworkSourceFileName": @"model/model.mil",
        };
        BOOL compiled = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
            g_ane_client, @selector(compileModel:options:qos:error:),
            mdl, compileOpts, 21, &e);
        if (!compiled) {
            if (!g_quiet) fprintf(stderr, "ane_bridge: _ANEClient compile failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:bundleDir error:nil];
            return NULL;
        }
        g_compile_count++;

        // Load onto ANE
        e = nil;
        BOOL loaded = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
            g_ane_client, @selector(loadModel:options:qos:error:),
            mdl, @{}, 21, &e);
        if (!loaded) {
            if (!g_quiet) fprintf(stderr, "ane_bridge: _ANEClient load failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            return NULL;
        }

        g_load_count++;

        // Build kernel handle (same structure as _ANEInMemoryModel path)
        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        k->model = mdl;  // _ANEModel, not _ANEInMemoryModel — eval uses g_ane_client
        k->tmpDir = modelDir;
        k->refcount = (int *)malloc(sizeof(int));
        *k->refcount = 1;
        k->loaded = true;
        k->cached = false;
        k->direct = true;
        k->nInputs = n_inputs;
        k->nOutputs = n_outputs;
        k->inputBytes = (size_t *)malloc(n_inputs * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        memcpy(k->inputBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(k->outputBytes, output_sizes, n_outputs * sizeof(size_t));

        k->nWeightFiles = n_weights;
        if (n_weights > 0) {
            k->weightRelPaths = (char **)malloc(n_weights * sizeof(char *));
            for (int i = 0; i < n_weights; i++) {
                NSString *name = [NSString stringWithUTF8String:weight_names[i]];
                NSString *relPath = name;
                if ([name hasPrefix:@"@model_path/"])
                    relPath = [name substringFromIndex:12];
                k->weightRelPaths[i] = strdup([relPath UTF8String]);
            }
        } else {
            k->weightRelPaths = NULL;
        }

        k->milData = [milData copy];

        // IOSurfaces (same as _ANEInMemoryModel path)
        k->ioInputs = (IOSurfaceRef *)malloc(n_inputs * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(n_outputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs; i++)
            k->ioInputs[i] = create_surface_wc(input_sizes[i]);
        for (int i = 0; i < n_outputs; i++)
            k->ioOutputs[i] = create_surface(output_sizes[i]);

        // Build request (same as _ANEInMemoryModel path)
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        return k;
    }
}

bool ane_bridge_eval(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel || !kernel->model) return false;
        NSError *e = nil;
        BOOL ok;
        if (kernel->direct && g_ane_client) {
            // _ANEClient path: evaluateWithModel:options:request:qos:error:
            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_ane_client,
                @selector(evaluateWithModel:options:request:qos:error:),
                kernel->model, @{}, kernel->request, 21, &e);
        } else {
            // _ANEInMemoryModel path: evaluateWithQoS:options:request:error:
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                kernel->model, @selector(evaluateWithQoS:options:request:error:),
                21, @{}, kernel->request, &e);
        }
        if (!ok && e && !g_quiet) {
            NSLog(@"[ANE] eval failed: %@", e);
        }
        return ok;
    }
}

void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes) {
    if (!kernel || idx < 0 || idx >= kernel->nInputs) return;
    IOSurfaceLock(kernel->ioInputs[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(kernel->ioInputs[idx]), data, bytes);
    IOSurfaceUnlock(kernel->ioInputs[idx], 0, NULL);
}

void ane_bridge_write_input_region(ANEKernelHandle *kernel, int idx,
                                    size_t offset, const void *data,
                                    size_t bytes) {
    if (!kernel || idx < 0 || idx >= kernel->nInputs) return;
    if (offset + bytes > kernel->inputBytes[idx]) return;
    IOSurfaceLock(kernel->ioInputs[idx], 0, NULL);
    void *base = IOSurfaceGetBaseAddress(kernel->ioInputs[idx]);
    memcpy((uint8_t *)base + offset, data, bytes);
    IOSurfaceUnlock(kernel->ioInputs[idx], 0, NULL);
}

void ane_bridge_write_input_strided(ANEKernelHandle *kernel, int idx,
                                     size_t dst_offset, size_t dst_stride,
                                     const void *src, size_t src_stride,
                                     size_t chunk_bytes, int n_chunks) {
    if (!kernel || idx < 0 || idx >= kernel->nInputs) return;
    size_t max_dst = dst_offset + (n_chunks > 0 ? (n_chunks - 1) : 0) * dst_stride + chunk_bytes;
    if (max_dst > kernel->inputBytes[idx]) return;
    IOSurfaceLock(kernel->ioInputs[idx], 0, NULL);
    uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(kernel->ioInputs[idx]);
    const uint8_t *s = (const uint8_t *)src;
    for (int i = 0; i < n_chunks; i++) {
        memcpy(base + dst_offset + (size_t)i * dst_stride,
               s + (size_t)i * src_stride,
               chunk_bytes);
    }
    IOSurfaceUnlock(kernel->ioInputs[idx], 0, NULL);
}

void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes) {
    if (!kernel || idx < 0 || idx >= kernel->nOutputs) return;
    IOSurfaceLock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(kernel->ioOutputs[idx]), bytes);
    IOSurfaceUnlock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
}

bool ane_bridge_share_surface(ANEKernelHandle *src, int out_idx,
                               ANEKernelHandle *dst, int in_idx) {
    @autoreleasepool {
        if (!src || !dst) return false;
        if (out_idx < 0 || out_idx >= src->nOutputs) return false;
        if (in_idx < 0 || in_idx >= dst->nInputs) return false;

        // Verify byte sizes match
        if (src->outputBytes[out_idx] != dst->inputBytes[in_idx]) {
            NSLog(@"[ANE] share_surface: size mismatch (src out=%zu, dst in=%zu)",
                  src->outputBytes[out_idx], dst->inputBytes[in_idx]);
            return false;
        }

        // Release dst's old input surface and point to src's output surface
        CFRelease(dst->ioInputs[in_idx]);
        dst->ioInputs[in_idx] = src->ioOutputs[out_idx];
        CFRetain(dst->ioInputs[in_idx]);

        // Rebuild dst's ANE request with the new IOSurface references
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:dst->nInputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:dst->nInputs];
        for (int i = 0; i < dst->nInputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), dst->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:dst->nOutputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:dst->nOutputs];
        for (int i = 0; i < dst->nOutputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), dst->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        dst->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        return true;
    }
}

bool ane_bridge_eval_chain(ANEKernelHandle **kernels, int n) {
    @autoreleasepool {
        if (!kernels || n <= 0) return false;
        for (int i = 0; i < n; i++) {
            ANEKernelHandle *k = kernels[i];
            if (!k || !k->model) return false;
            NSError *e = nil;
            BOOL ok;
            if (k->direct && g_ane_client) {
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_ane_client,
                    @selector(evaluateWithModel:options:request:qos:error:),
                    k->model, @{}, k->request, 21, &e);
            } else {
                ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    k->model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, k->request, &e);
            }
            if (!ok) {
                if (e) NSLog(@"[ANE] eval_chain step %d failed: %@", i, e);
                return false;
            }
        }
        return true;
    }
}

// Evaluate using the real-time path — lower dispatch overhead.
// Requires beginRealTimeTask to be called first.
static bool g_realtime_active = false;

bool ane_bridge_begin_realtime(void) {
    if (!g_ane_client) return false;
    if (g_realtime_active) return true;
    @try {
        ((void(*)(id,SEL))objc_msgSend)(g_ane_client, @selector(beginRealTimeTask));
        g_realtime_active = true;
        return true;
    } @catch (NSException *ex) {
        if (!g_quiet) NSLog(@"[ANE] beginRealTimeTask exception: %@", ex);
        return false;
    }
}

void ane_bridge_end_realtime(void) {
    if (!g_ane_client || !g_realtime_active) return;
    @try {
        ((void(*)(id,SEL))objc_msgSend)(g_ane_client, @selector(endRealTimeTask));
    } @catch (NSException *ex) {
        if (!g_quiet) NSLog(@"[ANE] endRealTimeTask exception: %@", ex);
    }
    g_realtime_active = false;
}

bool ane_bridge_eval_realtime(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel || !kernel->model || !g_ane_client) return false;
        @try {
            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_ane_client,
                @selector(evaluateRealTimeWithModel:options:request:error:),
                kernel->model, @{}, kernel->request, &e);
            if (!ok && e && !g_quiet) NSLog(@"[ANE] eval_realtime failed: %@", e);
            return ok;
        } @catch (NSException *ex) {
            if (!g_quiet) NSLog(@"[ANE] eval_realtime exception: %@", ex);
            return false;
        }
    }
}

bool ane_bridge_eval_chain_realtime(ANEKernelHandle **kernels, int n) {
    @autoreleasepool {
        if (!kernels || n <= 0 || !g_ane_client) return false;
        for (int i = 0; i < n; i++) {
            ANEKernelHandle *k = kernels[i];
            if (!k || !k->model) return false;
            @try {
                NSError *e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    g_ane_client,
                    @selector(evaluateRealTimeWithModel:options:request:error:),
                    k->model, @{}, k->request, &e);
                if (!ok) {
                    if (e && !g_quiet) NSLog(@"[ANE] eval_chain_realtime step %d failed: %@", i, e);
                    return false;
                }
            } @catch (NSException *ex) {
                if (!g_quiet) NSLog(@"[ANE] eval_chain_realtime step %d exception: %@", i, ex);
                return false;
            }
        }
        return true;
    }
}

bool ane_bridge_prepare_chain(ANEKernelHandle **kernels, int n) {
    @autoreleasepool {
        if (!kernels || n <= 0 || !g_ane_client) return false;
        for (int i = 0; i < n; i++) {
            ANEKernelHandle *k = kernels[i];
            if (!k || !k->model) return false;
            @try {
                NSError *e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_ane_client,
                    @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                    k->model, @{}, k->request, 21, &e);
                if (!ok) {
                    if (e && !g_quiet) NSLog(@"[ANE] prepare_chain step %d failed: %@", i, e);
                    return false;
                }
            } @catch (NSException *ex) {
                if (!g_quiet) NSLog(@"[ANE] prepare_chain step %d exception: %@", i, ex);
                return false;
            }
        }
        return true;
    }
}

ANEKernelHandle *ane_bridge_clone_kernel(ANEKernelHandle *source) {
    @autoreleasepool {
        if (!source || !source->model) return NULL;

        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        k->model = source->model;    // share compiled model (ARC retains)
        k->tmpDir = source->tmpDir;  // share tmpDir (not owned by clone)
        k->loaded = source->loaded;
        k->nInputs = source->nInputs;
        k->nOutputs = source->nOutputs;
        k->nWeightFiles = 0;         // clones don't own weight paths
        k->weightRelPaths = NULL;

        // Share refcount
        k->refcount = source->refcount;
        if (k->refcount) (*k->refcount)++;

        k->inputBytes = (size_t *)malloc(k->nInputs * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(k->nOutputs * sizeof(size_t));
        memcpy(k->inputBytes, source->inputBytes, k->nInputs * sizeof(size_t));
        memcpy(k->outputBytes, source->outputBytes, k->nOutputs * sizeof(size_t));

        // Create fresh IOSurfaces (write-combining for inputs, cached for outputs)
        k->ioInputs = (IOSurfaceRef *)malloc(k->nInputs * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(k->nOutputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < k->nInputs; i++)
            k->ioInputs[i] = create_surface_wc(source->inputBytes[i]);
        for (int i = 0; i < k->nOutputs; i++)
            k->ioOutputs[i] = create_surface(source->outputBytes[i]);

        // Build new request bound to the fresh IOSurfaces
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:k->nInputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:k->nInputs];
        for (int i = 0; i < k->nInputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:k->nOutputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:k->nOutputs];
        for (int i = 0; i < k->nOutputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        return k;
    }
}

void ane_bridge_free(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel) return;

        // Decrement refcount; only unload model + clean tmpDir when last reference drops
        bool last_ref = true;
        if (kernel->refcount) {
            last_ref = (--*kernel->refcount == 0);
        }

        if (last_ref) {
            // Save net.plist to persistent cache before unload destroys it.
            // ANE's unloadWithQoS: deletes the entire temp directory.
            if (kernel->tmpDir && kernel->model) {
                NSFileManager *fmSave = [NSFileManager defaultManager];
                NSString *plist = [kernel->tmpDir
                    stringByAppendingPathComponent:@"net.plist"];
                if ([fmSave fileExistsAtPath:plist]) {
                    id hexId = ((id(*)(id,SEL))objc_msgSend)(
                        kernel->model, @selector(hexStringIdentifier));
                    NSString *cacheSubdir = [ane_cache_dir()
                        stringByAppendingPathComponent:hexId];
                    [fmSave createDirectoryAtPath:cacheSubdir
                        withIntermediateDirectories:YES attributes:nil error:nil];
                    NSString *cacheDst = [cacheSubdir
                        stringByAppendingPathComponent:@"net.plist"];
                    // Overwrite if exists (may be updated by newer compile)
                    [fmSave removeItemAtPath:cacheDst error:nil];
                    [fmSave copyItemAtPath:plist toPath:cacheDst error:nil];
                }
            }

            NSError *e = nil;
            if (kernel->model) {
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    kernel->model, @selector(unloadWithQoS:error:), 21, &e);
            }
            // Preserve tmpDir for delta compilation cache — net.plist persists
            // across process restarts so subsequent launches skip compileWithQoS:.
            // macOS cleans /tmp on reboot. Weight files are small (~KBs).
            // Only delete if this was a patched kernel (no compile happened)
            // and it created a NEW directory (not reusing donor's dir).
            // For simplicity: always preserve. OS reclaims on reboot.
            free(kernel->refcount);
            kernel->model = nil;
            kernel->tmpDir = nil;
        } else {
            // Clone: don't unload shared model or delete tmpDir
            kernel->model = nil;
            kernel->tmpDir = nil;
        }

        for (int i = 0; i < kernel->nInputs; i++)
            if (kernel->ioInputs[i]) CFRelease(kernel->ioInputs[i]);
        for (int i = 0; i < kernel->nOutputs; i++)
            if (kernel->ioOutputs[i]) CFRelease(kernel->ioOutputs[i]);
        free(kernel->ioInputs);
        free(kernel->ioOutputs);
        free(kernel->inputBytes);
        free(kernel->outputBytes);
        for (int i = 0; i < kernel->nWeightFiles; i++)
            free(kernel->weightRelPaths[i]);
        free(kernel->weightRelPaths);

        kernel->request = nil;
        free(kernel);
    }
}

int ane_bridge_get_compile_count(void) {
    return g_compile_count;
}

int ane_bridge_get_load_count(void) {
    return g_load_count;
}

void ane_bridge_reset_compile_count(void) {
    g_compile_count = 0;
}

void ane_bridge_clear_cache(void) {
    NSString *dir = ane_cache_dir();
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm removeItemAtPath:dir error:nil];
    [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
}

bool ane_bridge_was_cached(ANEKernelHandle *kernel) {
    return kernel ? kernel->cached : false;
}

// --- Delta compilation: reload weights without recompiling ---
//
// Orion-style delta patching: the compiled ANE microcode (net.plist) is
// shape-dependent but weight-value-independent. After the initial compile,
// we can update weights by writing new BLOBFILEs to disk and calling
// loadWithQoS: — the ANE picks up the new data without recompilation.
//
// Two paths:
//   reload_weights — same model, unload → patch files → reload (fast path)
//   patch_from_donor — new model reuses donor's net.plist (zero-compile clone)

bool ane_bridge_reload_weights(ANEKernelHandle *kernel,
                                const uint8_t **weight_datas,
                                const size_t *weight_lens,
                                int n_weights) {
    if (!kernel || !kernel->model || !kernel->tmpDir) return false;
    if (n_weights != kernel->nWeightFiles) {
        fprintf(stderr, "ane_bridge: reload_weights: weight count mismatch (%d vs %d)\n",
                n_weights, kernel->nWeightFiles);
        return false;
    }

    @autoreleasepool {
        // Delta weight reload via fresh _ANEModel per call.
        //
        // Creates a new model with updated weights, populates its tmpDir with
        // net.plist (from old model's persistent cache → skip compile) + weight
        // files, then loads. The fresh model reads new weights from disk.
        //
        // Model caching doesn't help because unloadWithQoS deletes the tmpDir,
        // and loadWithQoS on a cached model without its tmpDir triggers a full
        // recompile (slower than creating a fresh model with a populated tmpDir).

        NSFileManager *fm = [NSFileManager defaultManager];

        // Use cached MIL text (avoids fragile disk reads — tmpDir may be gone after unload)
        NSData *milData = kernel->milData;
        if (!milData) {
            // Fallback: read from disk (legacy kernels without cached milData)
            NSString *milPath = [kernel->tmpDir stringByAppendingPathComponent:@"model.mil"];
            milData = [NSData dataWithContentsOfFile:milPath];
        }
        if (!milData) {
            fprintf(stderr, "ane_bridge: reload_weights: no MIL data (cached or on disk)\n");
            return false;
        }

        // Build descriptor with new weight data
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:kernel->weightRelPaths[i]];
            NSString *fullName = [@"@model_path/" stringByAppendingString:name];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            wdict[fullName] = @{@"offset": @0, @"data": data};
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) {
            fprintf(stderr, "ane_bridge: reload_weights: descriptor failed\n");
            return false;
        }

        id newModel = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!newModel) {
            fprintf(stderr, "ane_bridge: reload_weights: model creation failed\n");
            return false;
        }

        // Populate new model's tmpDir
        id newHexId = ((id(*)(id,SEL))objc_msgSend)(newModel, @selector(hexStringIdentifier));
        NSString *newTmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:newHexId];
        [fm createDirectoryAtPath:[newTmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[newTmpDir stringByAppendingPathComponent:@"model.mil"] atomically:NO];
        for (int i = 0; i < n_weights; i++) {
            NSString *fullPath = [newTmpDir
                stringByAppendingPathComponent:
                    [NSString stringWithUTF8String:kernel->weightRelPaths[i]]];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            [data writeToFile:fullPath atomically:NO];
        }

        // Copy net.plist to skip compile (from persistent cache or old model's dir)
        NSString *newPlist = [newTmpDir stringByAppendingPathComponent:@"net.plist"];
        if (![fm fileExistsAtPath:newPlist]) {
            NSString *oldPlist = [kernel->tmpDir stringByAppendingPathComponent:@"net.plist"];
            if ([fm fileExistsAtPath:oldPlist]) {
                [fm copyItemAtPath:oldPlist toPath:newPlist error:nil];
            } else {
                id oldHex = ((id(*)(id,SEL))objc_msgSend)(kernel->model, @selector(hexStringIdentifier));
                NSString *cachePlist = [[ane_cache_dir()
                    stringByAppendingPathComponent:oldHex]
                    stringByAppendingPathComponent:@"net.plist"];
                if ([fm fileExistsAtPath:cachePlist]) {
                    [fm copyItemAtPath:cachePlist toPath:newPlist error:nil];
                }
            }
        }

        // Unload old model (frees ANE slot)
        if (kernel->loaded) {
            NSError *ue = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                kernel->model, @selector(unloadWithQoS:error:), 21, &ue);
            kernel->loaded = false;
        }

        // Load new model
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            newModel, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            if (!g_quiet) fprintf(stderr, "ane_bridge: reload_weights LOAD FAILED: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            return false;
        }

        // Transfer ownership
        kernel->model = newModel;
        kernel->tmpDir = newTmpDir;
        kernel->loaded = true;

        // Rebuild ANE request with existing IOSurfaces
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:kernel->nInputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:kernel->nInputs];
        for (int i = 0; i < kernel->nInputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), kernel->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:kernel->nOutputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:kernel->nOutputs];
        for (int i = 0; i < kernel->nOutputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), kernel->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        kernel->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        return true;
    }
}

// --- Delta reload: reuse same model object, skip descriptor/model creation ---
//
// 10-30× faster than reload_weights for repeated hotswaps (e.g. classifier tiles).
// Unloads the model, repopulates tmpDir from in-memory caches (net.plist + MIL +
// new weights), then reloads the SAME _ANEModel object. Avoids creating a new
// descriptor, model, and hexId computation per call.
//
// Prerequisites: kernel must have milData and netPlistData cached (set at compile time).
bool ane_bridge_delta_reload(ANEKernelHandle *kernel,
                              const uint8_t **weight_datas,
                              const size_t *weight_lens,
                              int n_weights) {
    if (!kernel || !kernel->model || !kernel->tmpDir) return false;
    if (n_weights != kernel->nWeightFiles) {
        fprintf(stderr, "ane_bridge: delta_reload: weight count mismatch (%d vs %d)\n",
                n_weights, kernel->nWeightFiles);
        return false;
    }
    if (!kernel->milData || !kernel->netPlistData) {
        // Fall back to full reload if caches not available
        return ane_bridge_reload_weights(kernel, weight_datas, weight_lens, n_weights);
    }

    @autoreleasepool {
        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *td = kernel->tmpDir;

        // 1. Unload model (frees ANE SRAM — ANE deletes tmpDir)
        if (kernel->loaded) {
            NSError *ue = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                kernel->model, @selector(unloadWithQoS:error:), 21, &ue);
            kernel->loaded = false;
        }

        // 2. Repopulate tmpDir from in-memory caches + new weight data
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [kernel->netPlistData writeToFile:[td stringByAppendingPathComponent:@"net.plist"]
                               atomically:NO];
        [kernel->milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"]
                          atomically:NO];
        for (int i = 0; i < n_weights; i++) {
            NSString *fullPath = [td stringByAppendingPathComponent:
                [NSString stringWithUTF8String:kernel->weightRelPaths[i]]];
            NSData *data = [NSData dataWithBytesNoCopy:(void *)weight_datas[i]
                                                length:weight_lens[i]
                                          freeWhenDone:NO];
            [data writeToFile:fullPath atomically:NO];
        }

        // 4. Reload same model object (reads fresh weights from repopulated tmpDir)
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            kernel->model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            if (!g_quiet) fprintf(stderr, "ane_bridge: delta_reload LOAD FAILED: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            return false;
        }

        kernel->loaded = true;

        // 5. Rebuild ANE request with existing IOSurfaces
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:kernel->nInputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:kernel->nInputs];
        for (int i = 0; i < kernel->nInputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), kernel->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:kernel->nOutputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:kernel->nOutputs];
        for (int i = 0; i < kernel->nOutputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), kernel->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        kernel->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        return true;
    }
}

ANEKernelHandle *ane_bridge_patch_from_donor(
    ANEKernelHandle *donor,
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes)
{
    if (!g_initialized || !donor || !donor->tmpDir) return NULL;

    @autoreleasepool {
        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];
        NSFileManager *fm = [NSFileManager defaultManager];

        // Build weight dictionary for descriptor
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            wdict[name] = @{@"offset": @0, @"data": data};
        }

        // Create descriptor + model (gives us the hexStringIdentifier)
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) {
            fprintf(stderr, "ane_bridge: patch_from_donor: descriptor failed\n");
            return NULL;
        }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            fprintf(stderr, "ane_bridge: patch_from_donor: model creation failed\n");
            return NULL;
        }

        id hexId = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *newDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        bool sameDir = [newDir isEqualToString:donor->tmpDir];

        if (sameDir) {
            // Same MIL text + weight dict keys → same hexId → reuse directory.
            // Just update weight files in place.
        } else {
            // Different hexId — copy compiled microcode from donor
            [fm removeItemAtPath:newDir error:nil];
            [fm createDirectoryAtPath:newDir withIntermediateDirectories:YES
                           attributes:nil error:nil];

            NSString *srcPlist = [donor->tmpDir
                stringByAppendingPathComponent:@"net.plist"];
            NSString *dstPlist = [newDir
                stringByAppendingPathComponent:@"net.plist"];
            NSError *copyErr = nil;
            if (![fm copyItemAtPath:srcPlist toPath:dstPlist error:&copyErr]) {
                fprintf(stderr, "ane_bridge: patch_from_donor: net.plist copy failed: %s\n",
                        copyErr ? [[copyErr description] UTF8String] : "unknown");
                [fm removeItemAtPath:newDir error:nil];
                return NULL;
            }
        }

        // Write MIL text + weight files
        [milData writeToFile:[newDir stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];
        [fm createDirectoryAtPath:[newDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];

        // Track relative paths for delta reload
        char **relPaths = n_weights > 0 ? (char **)malloc(n_weights * sizeof(char *)) : NULL;
        bool dataWritten = false;

        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            NSString *relPath = name;
            if ([name hasPrefix:@"@model_path/"]) {
                relPath = [name substringFromIndex:12];
            }
            relPaths[i] = strdup([relPath UTF8String]);

            NSString *fullPath = [newDir stringByAppendingPathComponent:relPath];
            NSString *dir = [fullPath stringByDeletingLastPathComponent];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES
                           attributes:nil error:nil];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            [data writeToFile:fullPath atomically:NO];

            // First blob also becomes the data file (ANE compiler convention)
            if (!dataWritten) {
                [data writeToFile:[newDir stringByAppendingPathComponent:@"data"]
                       atomically:NO];
                dataWritten = true;
            }
        }

        // Load WITHOUT compiling — the key delta optimization
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            if (!g_quiet) fprintf(stderr, "ane_bridge: patch_from_donor LOAD FAILED: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            for (int i = 0; i < n_weights; i++) free(relPaths[i]);
            free(relPaths);
            if (!sameDir) [fm removeItemAtPath:newDir error:nil];
            return NULL;
        }

        // Build kernel handle
        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        k->model = mdl;
        k->tmpDir = newDir;
        k->loaded = true;
        k->refcount = (int *)malloc(sizeof(int));
        *k->refcount = 1;
        k->nInputs = n_inputs;
        k->nOutputs = n_outputs;
        k->inputBytes = (size_t *)malloc(n_inputs * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        memcpy(k->inputBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(k->outputBytes, output_sizes, n_outputs * sizeof(size_t));
        k->nWeightFiles = n_weights;
        k->weightRelPaths = relPaths;
        k->milData = [milData copy];

        // Create IOSurfaces + request (same as compile path)
        k->ioInputs = (IOSurfaceRef *)malloc(n_inputs * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(n_outputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs; i++)
            k->ioInputs[i] = create_surface_wc(input_sizes[i]);
        for (int i = 0; i < n_outputs; i++)
            k->ioOutputs[i] = create_surface(output_sizes[i]);

        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        // Note: does NOT increment g_compile_count
        return k;
    }
}

uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len) {
    int wsize = rows * cols * 2; // fp16
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    // ANE blob header
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;

    // Convert float32 -> float16 (NEON-vectorized: 4 values per cycle)
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    int count = rows * cols;
#if defined(__aarch64__)
    int i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(v);
        vst1_f16((__fp16 *)(fp16 + i), h);
    }
    for (; i < count; i++) {
        fp16[i] = (_Float16)src[i];
    }
#else
    for (int i = 0; i < count; i++) {
        fp16[i] = (_Float16)src[i];
    }
#endif

    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
                                                   size_t *out_len) {
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;

    // Convert float32 -> float16 with transpose (NEON for inner loop)
    _Float16 *fp16 = (_Float16 *)(buf + 128);
#if defined(__aarch64__)
    for (int i = 0; i < rows; i++) {
        int j = 0;
        for (; j + 3 < cols; j += 4) {
            float32x4_t v = vld1q_f32(src + i * cols + j);
            float16x4_t h = vcvt_f16_f32(v);
            // Scatter-store transposed (can't vectorize the scatter)
            __fp16 tmp[4];
            vst1_f16(tmp, h);
            fp16[(j+0) * rows + i] = (_Float16)tmp[0];
            fp16[(j+1) * rows + i] = (_Float16)tmp[1];
            fp16[(j+2) * rows + i] = (_Float16)tmp[2];
            fp16[(j+3) * rows + i] = (_Float16)tmp[3];
        }
        for (; j < cols; j++) {
            fp16[j * rows + i] = (_Float16)src[i * cols + j];
        }
    }
#else
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fp16[j * rows + i] = (_Float16)src[i * cols + j];
#endif

    *out_len = total;
    return buf;
}

void ane_bridge_free_blob(void *ptr) {
    free(ptr);
}

// ---------------------------------------------------------------------------
// fp16-weight GEMM: C[M,N] = alpha * A_f16[M,K] @ B_f32[K,N] + beta * C
//
// Tiles the M dimension so the fp16→fp32 expansion fits in L2 cache.
// Each micro-tile: convert fp16→fp32 via NEON vcvt, then cblas_sgemm.
// Halves main memory reads vs pure fp32 cblas_sgemm.
// ---------------------------------------------------------------------------
void ane_bridge_gemm_f16(
    const uint16_t *a_f16, int M, int K,
    const float *b_f32, int N,
    float *c_f32,
    float alpha, float beta)
{
    // Tile size: rows per micro-tile. 512 rows × K cols × 4 bytes = 4MB fp32 at K=2048.
    const int TILE_ROWS = 512;

    float *tile_f32 = (float *)malloc((size_t)TILE_ROWS * K * sizeof(float));
    if (!tile_f32) return;

    for (int m_start = 0; m_start < M; m_start += TILE_ROWS) {
        int m_tile = (M - m_start < TILE_ROWS) ? (M - m_start) : TILE_ROWS;
        size_t tile_elems = (size_t)m_tile * K;
        const uint16_t *src = a_f16 + (size_t)m_start * K;

#if defined(__aarch64__)
        // NEON vectorized fp16→fp32: 8 elements per iteration
        size_t i = 0;
        for (; i + 7 < tile_elems; i += 8) {
            float16x8_t h = vld1q_f16((const __fp16 *)(src + i));
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
            vst1q_f32(tile_f32 + i, lo);
            vst1q_f32(tile_f32 + i + 4, hi);
        }
        for (; i < tile_elems; i++) {
            tile_f32[i] = (float)(*(const __fp16 *)(src + i));
        }
#else
        for (size_t i = 0; i < tile_elems; i++) {
            uint16_t h = src[i];
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp_bits = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp_bits == 0) { f = sign; }
            else if (exp_bits == 31) { f = sign | 0x7F800000 | (mant << 13); }
            else { f = sign | ((exp_bits + 112) << 23) | (mant << 13); }
            memcpy(&tile_f32[i], &f, 4);
        }
#endif

        // C_tile[m_tile,N] = alpha * A_tile[m_tile,K] @ B[K,N] + beta * C_tile
        // Each tile writes to its own row range of C — no accumulation across tiles.
        cblas_sgemm(101, 111, 111,
                    m_tile, N, K,
                    alpha,
                    tile_f32, K,
                    b_f32, N,
                    beta,
                    c_f32 + (size_t)m_start * N, N);
    }

    free(tile_f32);
}

// ── Zero-copy IOSurface access (Orion-style) ─────────────────────────

void *ane_bridge_get_input_base(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->nInputs) return NULL;
    return IOSurfaceGetBaseAddress(kernel->ioInputs[idx]);
}

void *ane_bridge_get_output_base(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->nOutputs) return NULL;
    return IOSurfaceGetBaseAddress(kernel->ioOutputs[idx]);
}

size_t ane_bridge_input_size(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->nInputs) return 0;
    return kernel->inputBytes[idx];
}

size_t ane_bridge_output_size(ANEKernelHandle *kernel, int idx) {
    if (!kernel || idx < 0 || idx >= kernel->nOutputs) return 0;
    return kernel->outputBytes[idx];
}

// ── INT8 weight blob builders ────────────────────────────────────────

uint8_t *ane_bridge_build_weight_blob_int8(const int8_t *src, int rows, int cols,
                                            size_t *out_len) {
    int wsize = rows * cols;  // 1 byte per int8 element
    int total = 64 + wsize;   // 64-byte header + data
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    if (!buf) { *out_len = 0; return NULL; }

    // ANE int8 blob header (matches Orion format)
    buf[0] = 0xEF; buf[1] = 0xBE; buf[2] = 0xAD; buf[3] = 0xDE;
    buf[4] = 0x01;
    buf[10] = 0x08;  // 8-bit element marker

    memcpy(buf + 64, src, wsize);
    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_quantized(const float *src, int rows, int cols,
                                                 float *out_scale, size_t *out_len) {
    int n = rows * cols;

    // Find global max abs for symmetric quantization
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = src[i] < 0 ? -src[i] : src[i];
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs / 127.0f;
    if (scale == 0.0f) scale = 1.0f;

    // Quantize to int8
    int8_t *qdata = (int8_t *)malloc(n);
    if (!qdata) { *out_len = 0; *out_scale = 0; return NULL; }
    for (int i = 0; i < n; i++) {
        float v = src[i] / scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        qdata[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }

    uint8_t *blob = ane_bridge_build_weight_blob_int8(qdata, rows, cols, out_len);
    free(qdata);
    *out_scale = scale;
    return blob;
}
