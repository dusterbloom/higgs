# Tiled Ternary QMM Prototype

**Status:** Approved for a bounded, default-off prototype

**Date:** 2026-09-20

**Target:** Ternary-Bonsai-2-27B prefill on the current Apple Silicon host

## Goal

Measure whether a Metal `simdgroup_matrix` kernel specialized for the Prism
2-bit ternary format can beat MLX's stock quantized matrix multiplication on
the model's real prefill projection shapes. The prototype may enter the model
path only after it passes isolated correctness, performance, and memory gates.

The serving baseline remains the committed stock MLX QMM path. The existing
scalar `bonsai_q2_qmm_ternary` kernel is a test and benchmark reference, not the
performance baseline.

## Scope

The prototype covers affine Q2 projections whose loader metadata proves all of
these conditions:

- `bits == 2`
- `group_size == 128`
- a Prism Hadamard transform is present
- codes are ternary `{0, 1, 2}`
- `bias == -scale`
- weights use canonical `u32 [N, K/16]` storage
- scales use `[N, K/128]` storage
- the flattened activation row count is greater than eight

The first implementation uses existing packed weights. It does not convert the
checkpoint, keep a second packed layout, cache dense weights, change decode, or
enable the new path by default.

## Kernel Design

The first tile uses:

- `BM = 8` activation rows
- `BN = 32` output rows
- `BK = 128`, matching one quantization group
- 128 threads and four SIMD groups per threadgroup
- four independent 8 by 8 output fragments
- BF16 matrix operands and FP32 accumulators
- no split K and no double buffering

For each K group, the threadgroup stages an `8 x 128` activation tile and a
`32 x 128` decoded weight tile. Packed codes are decoded to `{-1, 0, +1}` and
combined with their per-output-row scales. Each SIMD group performs sixteen
8-wide matrix multiply-accumulate steps. Estimated threadgroup memory is about
10 KiB before alignment and bookkeeping.

M and N tails are zero-filled before all uniform barriers and matrix
operations. Output stores are guarded. Production K dimensions are divisible
by 128; unsupported dimensions stay on the stock path.

The current Hadamard rotation can produce FP32 activations while Metal matrix
operations use BF16 operands. That conversion is an explicit numerical change.
The benchmark and quality checks must compare:

1. the current FP32-Hadamard stock path;
2. stock QMM with the same candidate cast; and
3. the candidate kernel.

This separates conversion error from kernel arithmetic error. Tolerances are
fixed from native stock behavior before candidate results are examined.

## Host Integration

`metal_kernel.rs` owns the versioned MSL source, cached kernel handle, launch
configuration, contract validation, and output reconstruction. The wrapper
uses the task-local MLX stream and preserves the caller's output rank and dtype.

`qwen3_next.rs` adds a default-off prefill switch and two guarded dispatch
points after the operator passes its gate:

1. `QLinear::forward`, which covers down and standalone projections;
2. the dense fused gate/up path, which bypasses `QLinear::forward`.

Unsupported shapes and formats select stock MLX before dispatch. Runtime Metal
errors can be lazy, so the design does not promise recovery merely because the
wrapper returned `Ok`. JIT execution is proved in an isolated smoke run before
model integration.

The M=1 ternary QMV path and the opt-in M<=8 verifier remain unchanged.

## Correctness Checks

The CPU oracle uses the existing packed Q2 dequantization semantics. Candidate
results are also compared with stock MLX. Coverage includes:

- FP16 and BF16 inputs;
- rank-2 and rank-3 inputs;
- M tails around the dispatch boundary and tile boundary;
- N tails;
- K values 5120 and 17408;
- zeros, cancellation, extreme valid scales, and finite-value checks;
- invalid group, dtype, shape, and provenance fallback;
- output shape and dtype preservation;
- fixed real-model logits and deterministic greedy prompts.

Checks report maximum absolute error, relative error with a zero-safe rule, and
normalized RMS error. A changed greedy token is reviewed rather than waived by
loosening a numerical tolerance.

## Performance and Memory Gates

Compilation is warmed separately. Measurements force evaluation, alternate
stock and candidate order, and report repeated medians, spread, peak MLX active
memory, peak allocation, and process RSS.

The decisive operator shapes are:

- fused gate/up: `N=34816, K=5120`
- down: `N=5120, K=17408`
- activation rows: `M=512` and `M=1024`

Small M values are used only for safe JIT and correctness smoke runs. Gate/up
and down are promoted independently. Each projection must reach at least 1.2x
the stock operator at both real M values, including casts and allocations.

After guarded integration, the exact 512-token and 4096-token prefill workloads
must improve unprofiled latency by at least 10 percent beyond observed run
noise. Decode must not regress. The candidate must fit loaded-model headroom
without reducing the configured chunk size or retaining dense matrices.

Only one process runs GPU work. The coordinator stops the port 9000 serving
process for GPU validation and restores it afterward.

## Stop Conditions

Stop after the first correct tile and at most one evidence-driven tile variant
when any of these conditions holds:

- the operator gain stays below 1.2x stock;
- the native precision or fixed-prompt quality checks fail;
- the runtime requires unsupported matrix element-layout assumptions;
- memory pressure forces a smaller chunk size or erases the speedup;
- end-to-end prefill improves by less than 10 percent.

Persistent matrix-friendly weight repacking is outside this prototype. It may
be proposed separately only if profiling identifies packed-weight access as the
winning kernel's remaining bottleneck.

## Team and Order

Work proceeds in gated waves:

1. Luna A owns `metal_kernel.rs` and implements the kernel and wrapper.
2. Luna B owns `bonsai_q2.rs` and prepares oracle cases and operator benchmarks.
3. Astra reviews the wrapper contract, precision, barriers, tails, and evidence.
4. After the operator gate passes, Luna B owns the two `qwen3_next.rs` dispatch
   points and the default-off switch.
5. The coordinator alone runs GPU correctness, performance, and full-model A/B
   measurements.
6. Astra reviews the final evidence before any default behavior or merge is
   proposed.

No two workers edit `metal_kernel.rs` or `qwen3_next.rs` concurrently.

## Expected Effort and Performance Range

The bounded prototype is estimated at three to seven focused engineering days.
Two workers parallelize the independent kernel and fixture work; runtime JIT,
quality, and performance tuning remain sequential.

The instrumented 4096-token attribution assigns about 59.6 percent of prefill
time to gate/up and down QMM. As an estimate, a 2x QMM improvement would move
50 tok/s to about 71 tok/s, 2.5x to about 78 tok/s, and 4x to about 90 tok/s.
Reaching 100 tok/s from this kernel alone would require roughly a 6.2x QMM
improvement if the non-QMM time remains unchanged. Final claims use unprofiled
end-to-end measurements.
