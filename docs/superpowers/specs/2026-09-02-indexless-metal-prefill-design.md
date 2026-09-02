# Indexless Corrected Metal Prefill

## Status

Approved direction. This document defines the design boundary; it does not authorize broadening the first implementation beyond the evaluator and gates described below.

## Objective

Reduce long-context prefill latency on Apple Silicon by replacing eligible dense multi-token attention with a Metal-native approximation inspired by [FlashPrefill V2](https://arxiv.org/abs/2608.19758), while preserving every prompt token and the ordinary full-length KV cache.

The first release targets a single causal prompt prefill over contiguous BF16 or FP16 KV storage. Decode remains dense and unchanged. The design favors a small, measurable Metal path over a CUDA-compatible attention-backend abstraction.

## Success criteria

The work is successful only if all of the following hold:

- The offline trace evaluator identifies a stable approximation Pareto point across early and late full-attention layers.
- The Metal kernel matches the chosen approximation oracle within the BF16/FP16 numerical tolerance defined below.
- The eligible path produces a clear operator-level win above a measured context-length knee and no regression below it because dense fallback remains active.
- Qwen3-Next time to first token improves measurably despite only every fourth layer using full attention.
- RULER and needle-in-a-haystack quality remain within the release gates below.
- The resident KV cache still contains one row per source token; no token pruning or RoPE-position compression is introduced.

## Core decision

Do not port FlashPrefill V2's two-stage CUDA/Hopper architecture. Higgs will use row-local, indexless selection inside the attention kernel.

For a query row `q` and KV block `j` with valid count `n_j`, key mean `k_mean_j`, and value mean `v_mean_j`, define

```text
a_j = scale * dot(q, k_mean_j)
```

The kernel performs three cheap summary scans:

1. Find `a_max` over blocks that are fully visible to the query. Exclude the partial causal-boundary block from all summary scans.
2. Count selected blocks and choose sparse or dense execution for the row.
3. Traverse blocks and update one FP32 online-softmax state:
   - selected block: process its causally visible tokens exactly;
   - omitted block: process one pseudo-token with logit `a_j + log(n_j)` and value `v_mean_j`.

A block is selected when

```text
a_j >= a_max + log(alpha)
```

or when it belongs to the forced sink region, local window, or causal-boundary region. The implementation branches before taking the logarithm: `alpha <= 0` is the dense correctness mode and selects every visible block.

The partial causal-boundary block is exact but does not contribute a pooled score. This prevents future keys within that block from changing the threshold or the selection of earlier blocks. If no complete block is visible, the row uses exact causal attention only.

Selection is per query row rather than shared across a large query tile. This removes cross-head/block unions and reduces the selection rule to a logit comparison. Summary scores are recomputed instead of written to a score tensor or sparse index.

At block size `B`, three key-summary scans plus one value-summary correction scan add approximately `2/B` of dense QK+PV arithmetic. At `B = 128`, this is about 1.6 percent before exact selected-block work.

## Why this is the Apple design

The reference implementation relies on CUTLASS/CuTe, TMA, `wgmma`, warp specialization, CSR compaction, and Hopper-oriented ping-pong pipelines. Those mechanisms do not transfer directly to Apple GPUs.

Higgs has also measured that doubling threadgroup staging from 10.6 KB to 21.2 KB halved resident threadgroups and regressed latency on M4. The initial kernel therefore uses:

- no ping-pong staging;
- no global score or index workspace;
- no prefix sum or compaction;
- no persistent scheduler metadata;
- no threadgroup allocation above 10 KB without profiler evidence;
- direct contiguous K/V reads and FP32 accumulator state held by each SIMD group.

The initial schedule assigns one `(query row, query head)` to one SIMD group and four rows to a 128-thread threadgroup. The exact lane mapping and number of output channels per lane are compile-time choices for the target head dimension. They may change after profiling without changing the algorithmic contract.

## Approximation selection

The first Metal implementation uses one centroid per 128-token block unless the trace evaluator rejects it.

Three candidates are evaluated at matched exact-token density:

### Mean

Store `(k_mean, v_mean)` and use the correction described above. This is the minimal baseline and default implementation candidate.

### Mean plus risk

Additionally compute scalar within-block dispersion. The evaluator compares mass-only selection with an error-risk proxy derived from

```text
risk_j ~= estimated_mass_j * sigma_k_j * sigma_v_j
```

The product is motivated by a first-order bound on the omitted block's key/value covariance error. It is a heuristic because scalar dispersion discards query direction. It is promoted only if it reduces exact density by at least 25 percent at equal output error across the trace suite.

### Haar-2

Represent a logical 128-token block by the means of its two 64-token halves. An omitted block contributes two pseudo-tokens with `log(64)` count shifts.

Refining one mean into two half means monotonically improves the denominator approximation by Jensen's inequality and captures the between-half key/value covariance mode. It doubles summary and correction work. It is promoted only if it halves p99 correction error or permits a 35--50 percent lower exact density at equal output error.

No learned centroids, PCA, covariance matrices, clustering, or random-feature attention are in scope.

## Trace evaluator gate

The trace evaluator precedes Metal implementation so approximation error and kernel error are never debugged simultaneously.

An environment-gated hook captures the exact SDPA inputs after query/key normalization and RoPE:

- one early and one late full-attention layer;
- one KV head and its grouped query heads;
- K/V for representative 4K, 8K, 16K, and longer prefixes when memory permits;
- sampled query rows near block boundaries, retrieval positions, and prompt tails.

Traces use safetensors and introduce no dependency. The evaluator sweeps:

- block size 64 and 128;
- selection threshold;
- sink and local-window sizes;
- mean, mean-plus-risk, and Haar-2 correction.

It reports:

- exact-token density and selected-block count;
- attention-output relative L2 and cosine error;
- p50, p95, and p99 row error;
- worst row/head and prompt position;
- next-token KL when logits are available;
- correlation between each selection proxy and actual correction error.

The evaluator chooses the local approximation. It does not replace end-to-end quality tests.

## Semantic oracle

Before Metal dispatch, a simple Rust/MLX oracle implements the chosen row-local algorithm. It covers:

- dense mode;
- corrected sparse mode;
- grouped-query head mapping;
- causal visibility with nonzero offsets;
- partial final blocks and true block counts;
- forced sink, local, and diagonal blocks.

The oracle is not a production fast path. It exists to specify Metal behavior and to isolate approximation drift from kernel defects.

## Metal kernel

The custom kernel is a private Higgs operation built through the existing `mlx_fast_metal_kernel` FFI pattern.

Each SIMD group owns one query row/head and keeps:

- query fragments;
- FP32 running maximum;
- FP32 softmax denominator;
- FP32 output fragments.

For selected blocks, lanes cooperate on QK reductions with `simd_sum`, compute stable online-softmax weights, and accumulate the corresponding V fragments. For corrected blocks, the same state consumes the block pseudo-token. The partial causal block is always exact.

The first version deliberately uses direct loads and scalar/SIMD reductions. SIMD-group matrix multiplication, PackGQA-style sharing, and single-buffer staging are measurement-driven follow-ups, not prerequisites.

Kernel specialization is bounded to dtype, block size, and supported head dimensions. Sequence lengths, valid cache length, head counts, GQA ratio, query offset, scale, and `log(alpha)` are runtime inputs so arbitrary prompt lengths do not create an unbounded JIT cache.

## Initial integration seam

The first target is `Qwen3NextAttention::forward_scheduled`, immediately after dense K/V append and before the existing multi-token MLX SDPA call.

The approximation is eligible only when all of these conditions hold:

- batch size is one;
- query length is greater than one;
- this is the first prompt prefill with zero prior cache offset;
- the mask is ordinary causal attention;
- cache storage is dense and contiguous;
- dtype is BF16 or FP16;
- head dimension has a compiled kernel specialization;
- sequence length exceeds a fixed, benchmarked activation threshold.

Everything else falls back to existing MLX SDPA. In particular, decode, canonical speculative rows, TurboQuant, MLA, arbitrary array masks, soft-capping, and model-specific sliding-window behavior are unchanged.

The documented Qwen3-Next configuration uses head dimension 256. The implementation must inspect the actual benchmark checkpoint and specialize its real dimension; it must not assume the reference CUDA repository's dimensions or hardcode 128.

After this path wins, generic Qwen/Llama integration may route eligible calls through `cached_scaled_dot_product_attention`. No public backend trait is introduced before that evidence exists.

## Block summaries

The first offset-zero implementation computes block means per attention call using ordinary MLX reductions. This is linear work beside quadratic attention and avoids changing cache lifecycle code before the kernel proves useful.

Persistent block summaries are a later optimization for chunked prefill and prefix reuse. If added, they form a private sidecar to `SteppingKeyValueCache` and must:

- recompute only the previous open tail block and newly appended blocks;
- track the valid length separately from backing allocation capacity;
- participate in deep clone, evaluation targets, and memory accounting;
- rebuild lazily after prefix restore;
- invalidate on trim, prune, rollback, cache quantization, or incompatible representation changes.

TurboQuant summaries must never be mixed with exact blocks from a different representation.

## Dense fallback

Sparse execution is not universally faster or more accurate. Diffuse rows, heterogeneous blocks, or disjoint GQA requirements may select most blocks.

The kernel's second summary scan counts selected blocks. If density exceeds a benchmarked break-even value, the row traverses all visible tokens exactly. This gate is deterministic and uses a compiled default; the first release has no runtime autotuner.

The outer dispatch also retains dense MLX SDPA below the measured sequence-length knee.

## Correctness requirements

- `alpha <= 0` matches dense MLX SDPA within `2e-3` maximum absolute/relative error for supported BF16/FP16 shapes.
- Constant K/V blocks are exact under correction even when no non-forced block is selected.
- Future K/V perturbations cannot affect earlier causal queries.
- Partial and diagonal blocks are always exact, mask future tokens internally, and never contribute pooled scores to selection.
- A selected block is never also corrected.
- Block count, not nominal block size, supplies the tail block's logarithmic shift.
- Grouped-query head mapping is correct for every supported GQA ratio.
- Kernel output matches the semantic oracle within `2e-3`.
- Existing one-token decode and canonical-row tests remain unchanged.

## Performance gates

All timings warm the JIT, force evaluation, and use paired medians. Measure 4K, 8K, 16K, 32K, and 64K where the checkpoint supports them.

The kernel must show:

- no end-to-end regression below the activation knee because fallback remains active;
- a clear attention-operator win above the knee;
- a measurable Qwen3-Next TTFT win;
- no unexplained increase in active or peak MLX memory;
- no threadgroup allocation above 10 KB without Metal profiling that demonstrates a net gain.

Report selector scans, exact traversal, correction traversal, total attention latency, selected density, and selected-run contiguity separately.

## Quality gates

Run dense versus approximate prefill on RULER, needle-in-a-haystack, and representative LongBench tasks at supported long-context lengths.

The initial release target is:

- no more than one point average loss versus dense;
- no more than two points loss on any individual gated task;
- full resident KV length preserved;
- ordinary dense decode machinery preserved;
- adversarial needles placed inside heterogeneous blocks included in the suite.

Approximate prefill can change later-layer K/V and generated tokens. "Decode unchanged" means the decode algorithm and dispatch remain unchanged, not that generation is bit-identical to dense prefill.

## Failure handling

Unsupported shapes or semantics always take the dense path. Kernel compilation or runtime failures surface through the existing MLX exception path and must not silently return approximate output.

Configuration is private and experimental until the performance and quality gates pass. The first implementation does not expose server flags.

## Deferred work

The following are explicitly outside the first implementation:

- paged device KV addressing;
- continuous or variable-length prefill batching;
- chunked-prefill acceleration and persistent summary sidecars;
- FP8;
- TurboQuant sparse prefill;
- MLA;
- arbitrary masks, soft-capping, and model-specific window semantics;
- global score tensors, CSR indices, compaction, prefix sums, and scheduler workspaces;
- PackGQA, SIMD-group matrix attention, cross-layer selection reuse, and double buffering;
- learned or model-specific selection policies;
- public attention-backend APIs and runtime autotuning;
- combining this approximation with PFlash token pruning.

## Implementation sequence

1. Add the QKV trace hook and offline evaluator.
2. Select mean, mean-plus-risk, or Haar-2 using the stated promotion gates.
3. Add the semantic oracle and correctness tests.
4. Add the private Metal kernel in dense mode, then corrected sparse mode.
5. Add the narrow Qwen3-Next offset-zero dispatch and dense fallback.
6. Run operator, TTFT, memory, and quality gates.
7. Only after a passing result, decide whether persistent summaries, chunked prefill, or generic transformer integration deserve separate designs.

Approximate first-implementation footprint is 700--1,000 lines across the evaluator, oracle, Metal wrapper/source, tests, and narrow dispatch. Persistent cache summaries would add roughly 250--350 lines in a later phase.
