# Affine Q2 Gate/Up Dispatch Design

## Goal

Determine, with token-parity and repeatable end-to-end measurements, whether
separating the dense SwiGLU gate and up projections lets Qwen3.8-27B affine-Q2
use the existing Q2 SIMD decoder faster than the current fused projection.

## Scope

This is priority 3 only. It does not change native Escha Trellis QMV, native
Trellis prefill, checkpoint conversion, quantization format, or model defaults
until the measured promotion gate is met.

## Confirmed Current Path

For a dense affine-Q2 MLP decode token, `FfnBlock::forward` selects
`dense_hidden_fused` by default. That path concatenates gate and up weights,
scales, and biases on the output dimension, then calls `quantized_forward`.
The resulting fused projection is `[34816, 320]` packed (`N=34816`,
`K=5120`).

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
process-start environment switches to isolate fusion from the Q2 SIMD kernel.
This makes the experiment reversible and prevents a microbenchmark result from
silently changing model behavior.

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
4. Full-model greedy generation must emit the identical token-ID sequence for
   a fixed prompt and at least 128 generated tokens in every candidate run.

A change in floating-point reduction order is not accepted merely because it
is numerically close: the full-model token gate is the release criterion.

## Measurement Matrix

Run each cell in a new server process. `OnceLock` caches both environment
choices, so changing an environment variable inside an existing Higgs process
does not create a valid comparison.

| Cell | `HIGGS_DENSE_FFN_GATE_UP` | `HIGGS_BONSAI_Q2_SIMD` | Purpose |
| --- | --- | --- | --- |
| A | `1` | `0` | Current fused/stock baseline |
| B | `1` | `1` | Control: fused shape remains stock MLX |
| C | `0` | `0` | Cost of separate stock MLX projections |
| D | `0` | `1` | Candidate: separate projections plus SIMD QMV |

Every run uses Escha Qwen3.8-27B in affine-Q2 mode, a fixed cached prompt and
greedy decoding. Record model load time separately from warm decode. Record
prefill tokens/s and post-cache decode tokens/s separately; priority 3 is
judged only on post-cache decode.

Use three ABBA pairs for each comparison against A, alternating thermal and
clock drift rather than grouping all candidates together. Keep AC power,
unchanged power mode, identical wired memory limit, no competing model/server,
and a recorded `pmset`/memory-pressure snapshot. Reject a run with a model
reload failure, cache miss, different token sequence, or competing GPU work.

## Promotion Gate

Promote only if D is at least 3% faster than A in median post-cache decode,
does not regress the worst valid run by more than 1%, and passes every
full-model token-ID parity run. B confirms the expected dispatch boundary; C
quantifies whether fusion itself dominates.

If D does not pass, retain the present opt-in switches and make no default
change. If D passes, the next implementation is a narrowly shape-gated
automatic policy plus the tests above; it does not generalize the kernel to
other shapes.

## Verification

The implementation must run the focused model tests, formatting, the relevant
ignored Q2 microbenchmark for diagnostic context, and the fresh-process
end-to-end matrix. The final review must inspect the diff and verify that the
only changed runtime behavior is the proven affine-Q2 dense decode selection.
