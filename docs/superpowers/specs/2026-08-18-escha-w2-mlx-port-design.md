# Escha-W2 MLX/K3 Port Design

## Context

The `nightly` branch already has a resident native Escha trellis path, direct Metal QMV for short routed domains, and a trellis QGEMM path for longer prefill. The K3 roadmap's useful lesson is to eliminate avoidable route-boundary and dense-weight movement, but its pinned host staging and selected-slab uploads target PCIe systems. Apple unified memory does not have that transfer boundary, so copying selected expert slabs would add work rather than remove it.

## Goals

1. Make the native Escha path safe for short routed domains and unusual but supported shapes.
2. Establish exactness and greedy-token gates before changing a reduction or dtype boundary.
3. Evaluate the existing direct-read trellis QGEMM as a prefill-only optimization, without changing the default until an end-to-end result earns promotion.
4. Keep the work small, reversible, and suitable for upstreaming to `panbanda/higgs`.

## Non-goals

- Port CUDA-only P23/P25/P26 host placement, selected-slab upload, or pinned DMA staging.
- Add adaptive expert dropping, D3 approximation, CPU/GPU split placement, or a persistent multi-gigabyte decoded workspace.
- Combine Escha changes with MTP/DSpark speculation; the Escha MTP expert weights are unavailable and there is no verifier contract for changed logits.
- Promote a numerically different Steel or simdgroup reduction as a drop-in replacement.

## Design

### Phase 1: correctness boundary

Inspect and correct the native `SwitchMlpWeights` routing order so QMM-specific short-domain padding cannot leak into the native Escha branch. Preserve the native activation/ID row correspondence for domains below four rows per expert and across the 32-row QMV/QGEMM dispatch boundary. Add focused tests for short rows, batch/layout equivalence, sorted and unsorted expert IDs, and the 32/33 boundary.

Add fail-closed validation at the native gather boundary for unsupported K/dimensions/dtypes, expert-count/code-shape mismatch, and out-of-range expert IDs. Validation must occur before device pointer arithmetic.

### Phase 2: exactness gates

Treat each native kernel as a distinct arithmetic contract:

- Tile decode must compare FP16 bit patterns.
- A faithful QMV port must preserve lane mapping, FMA order, and `simd_sum`, and compare FP32 output bits with the existing native QMV.
- A faithful scalar QGEMM port must compare bits with the existing scalar QGEMM, including ragged expert walks.
- Any Steel/simdgroup or changed FP16/FP32 boundary is a new backend and requires a separate quality qualification.

Add a real-checkpoint greedy trajectory gate: fixed prompts, first-token parity, and a 128–256-token token digest against the current native baseline. Native-vs-affine logit differences remain diagnostic only; affine is not an exact trellis oracle.

### Phase 3: prefill-only QGEMM evaluation

Keep decode on direct QMV. Compare the existing scratch and direct trellis QGEMM paths in fresh processes at fixed 64/1K/4K prefill and a fixed decode window. Report median/tail latency, TTFT, causal decode rate, load time, and active/peak memory. Cover sorted, ragged, and fragmented routing.

Retain `HIGGS_ESCHA_NATIVE=0` as the affine escape hatch and `HIGGS_ESCHA_TRELLIS_GEMM=1` as the experiment switch. Do not default-enable QGEMM or add public configuration until correctness and full-model measurements pass.

## Acceptance gates

- Existing workspace tests remain green, with no new warnings or formatting changes.
- Native short-row tests pass without shape or expert-ID drift.
- Invalid expert IDs and mismatched code/expert dimensions fail before pointer formation.
- Exact kernel contracts pass for the tested production domains; tolerance-only comparisons are not sufficient for a claimed faithful port.
- Greedy token trajectory remains unchanged for the fixed real-checkpoint suite.
- Any promoted path demonstrates a material end-to-end gain, unchanged model residency expectations, and no regression on ragged routing.

## Rollback and upstream shape

Keep changes localized to `crates/higgs-models/src/eschamoe.rs`, `crates/higgs-models/src/metal_kernel.rs`, `crates/higgs-models/src/qwen3_next.rs`, and focused tests/docs. Preserve the existing affine and scratch paths as explicit fallbacks. Split correctness/validation from optional performance changes so the first upstream review can land without requiring a benchmark-dependent default change.

