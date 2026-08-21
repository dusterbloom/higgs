# Affine Q2 Gate/Up Dispatch Design

## Goal

Determine, with token-parity and repeatable end-to-end measurements, whether
separating the dense SwiGLU gate and up projections lets Qwen3.8-27B affine-Q2
use the existing Q2 SIMD decoder faster than the current fused projection.

## Scope

This is priority 3 only. It does not change native Escha Trellis QMV, native
Trellis prefill, checkpoint conversion, quantization format, or model defaults
until the measured promotion gate is met.

For this Escha W2 experiment, `HIGGS_ESCHA_AFFINE_BITS=2` changes only the
conversion bit width. The conversion target retains its checkpoint-compatible
affine group size of 64, so every Q2 test and any later Escha-specific automatic
policy uses group size 64. Bonsai-Q2's unrelated group-size-128 fixtures are
not the target contract.

## Confirmed Current Path

For a dense affine-Q2 MLP decode token, `FfnBlock::forward` selects
`dense_hidden_fused` by default on the target Apple GPUs. On a base Apple M4,
the existing safe-default policy instead selects the separate path when
`HIGGS_DENSE_FFN_GATE_UP` is unset. The benchmark must therefore record the
hardware and include an unset-environment production baseline.

The fused path concatenates gate and up weights, scales, and biases on the
output dimension, then calls `quantized_forward`. The resulting fused
projection is `[34816, 320]` packed (`N=34816`, `K=5120`).

`affine_q2_simd_forward` only routes to `bonsai_q2_qmv_simd` when all of the
following are true:

- `HIGGS_BONSAI_Q2_SIMD=1`;
- flattened activation row count is one;
- weight shape is `[17408, 320]`, hence logical `K=5120`.

The fused `[34816, 320]` projection therefore always uses stock MLX. With
`HIGGS_DENSE_FFN_GATE_UP=0`, `dense_hidden_separate` calls each original
`[17408, 320]` projection independently; each becomes eligible for the SIMD
route when Q2 SIMD is enabled.

## Alternatives

### A. Measure the existing switches first — selected

Do not alter the production selection policy initially. Use the two existing
environment switches from a clean server process to isolate fusion from the Q2
SIMD kernel. This makes the experiment reversible and prevents a microbenchmark
result from silently changing model behavior.

### B. Automatically split eligible Q2 gate/up pairs

After a passing experiment, add a narrow automatic policy for affine-Q2 dense
SwiGLU decode only. The policy must require the exact known shape and leave all
other models, bit widths, sequence lengths, and prefill behavior unchanged.

### C. Keep fusion and widen the SIMD kernel to `N=34816`

This is a later kernel project. It changes the Metal execution shape and cannot
be justified before establishing whether the two existing `N=17408` SIMD
dispatches recover enough end-to-end decode time.

## Correctness Contract

The experiment changes scheduling only. It must retain the checkpoint,
prompt, tokenizer, sampling configuration, context/cache behavior, and model
load configuration.

Before a runtime policy can be promoted:

1. A unit test must protect the exact SIMD eligibility boundary: single-token
   `[17408, 320]` is eligible with the flag; `[34816, 320]`, another token
   count, or another packed-K dimension is not.
2. The existing dense fused/separate hidden-output test must cover affine Q2
   parameters as well as its current Q4 fixture, comparing evaluated hidden
   values within an explicitly documented FP16 tolerance.
3. The SIMD QMV route must be compared with stock MLX on deterministic Q2
   tensors using the same tolerance already used by Q2 kernel reference tests.
4. Full-model greedy generation must emit one identical token-ID sequence
   across every valid production-baseline, A, B, C, and D run. The check
   includes a rebuilt prefill/cache and at least 128 generated decode tokens;
   candidate-internal determinism alone is not sufficient.
5. A process-local dispatch trace must prove the expected routing for every
   cell: no SIMD dispatch for the production baseline, A, B, or C unless that
   is explicitly expected from the target's pre-existing defaults, and one or
   more SIMD dispatches for D. The trace is diagnostic-only and must not add
   work when disabled.

A change in floating-point reduction order is not accepted merely because it
is numerically close: the full-model token gate is the release criterion.

## Measurement Matrix

Run each cell in a new server process. `DENSE_FFN_FUSE_GATE_UP` is cached in a
`OnceLock`; `HIGGS_BONSAI_Q2_SIMD` is read at every dispatch. Fresh processes
are still mandatory because fusion creates lazy persistent arrays on the first
decode token and because a clean process controls Metal pipeline compilation,
cache construction, and the remaining process state.

| Cell | `HIGGS_DENSE_FFN_GATE_UP` | `HIGGS_BONSAI_Q2_SIMD` | Purpose |
| --- | --- | --- | --- |
| P | unset | unset | Actual production baseline on this hardware |
| A | `1` | `0` | Current fused/stock baseline |
| B | `1` | `1` | Control: fused shape remains stock MLX |
| C | `0` | `0` | Cost of separate stock MLX projections |
| D | `0` | `1` | Candidate: separate projections plus SIMD QMV |

Every run uses Escha Qwen3.8-27B in affine-Q2 mode, a fixed cached prompt and
greedy decoding. Rebuild the same prefill/cache in every process, discard 32
generated warmup tokens after that prefill, and then measure exactly 512
post-cache greedy decode tokens. Record model load time, prefill tokens/s,
warmup completion, and post-cache decode tokens/s separately.

Use three ABBA pairs (six samples per comparison) for P-versus-D on hardware
where P is fused, alternating thermal and clock drift rather than grouping all
candidates together. Run A, B, and C as diagnostic controls with the same
warmup and fixed decode length. Report every sample, the median of six, and
the worst sample. Keep AC power, unchanged power mode, identical wired memory
limit, no competing model/server, and a recorded `pmset`/memory-pressure
snapshot. Reject a run with a model reload failure, cache miss, missing dispatch
trace, different cross-cell token sequence, or competing GPU work.

## Promotion Gate

Promote only if D is at least 3% faster than the actual production baseline P
in median post-cache decode, does not regress the worst valid run by more than
1%, and passes every full-model token-ID parity run. On a base M4 where P is
already separate, this experiment may establish only the opt-in SIMD benefit;
it does not justify changing the gate/up default. B confirms the expected
dispatch boundary; C quantifies whether fusion itself dominates.

If D does not pass, retain the present opt-in switches and make no default
change. The earlier 586ba2eac end-to-end win and its 475efb6ea reversal show
that isolated QMV speed does not predict full-model AR decode; a valid null
result is an intended outcome.

If D passes, the next implementation is a decode-only, narrowly shape-gated
automatic policy plus the tests above. It requires affine-Q2, group size 64,
non-empty biases, canonical layout, exactly `[17408, 320]` weights for both
gate and up, and `seq_len == 1`. Prefill remains on the present path. The
automatic decision must be per-call from the eligible dense MLP path; it must
not silently redefine unset `HIGGS_BONSAI_Q2_SIMD` for unrelated affine-Q2
projections in a multi-model process. Explicit environment settings remain
kill-switches.

## Verification

The first implementation adds only pure dispatch eligibility tests, Q2/G64 SIMD
numeric parity coverage, Q2/G64 fused/separate coverage, and the
disabled-by-default dispatch diagnostic needed for the matrix. It must run focused model tests,
formatting, the relevant ignored Q2 microbenchmark for diagnostic context, and
the fresh-process end-to-end matrix. The final review must inspect the diff and
verify that no automatic runtime behavior changed before the promotion gate is
met.
