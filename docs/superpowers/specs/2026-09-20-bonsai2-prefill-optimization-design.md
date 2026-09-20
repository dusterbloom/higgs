# Ternary Bonsai 2 Prefill Optimization Program

## Status

Approved design. This document authorizes measurement and controlled experiments in the order below. It does not authorize source implementation until the relevant gate passes.

Target:

- model: `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`;
- hardware: base Apple M4, 10-core GPU, 32 GB unified memory;
- runtime: Higgs 1.8 on branch `fix/mlx-031-compat`;
- verified starting revision: `fcca407727babd66a0afad54114e283317216bc5`;
- server port: `9000` when HTTP timing is required.

Decode optimization is parked. This program measures and changes prompt prefill only.

## Objective

Establish a reproducible prefill baseline, identify the dominant cost with persisted evidence, and improve 4K prompt prefill without sacrificing short-prompt latency, numerical correctness, memory stability, or model behavior.

The program has two separate tracks:

1. **Exact dense prefill**, which preserves the current packed model computation within fixed floating-point tolerances.
2. **Approximate sparse prefill**, which may skip prompt-token computation and must be exposed and evaluated as a distinct behavior-changing mode.

The aspirational 4K target is at least 100 prompt tokens/s. It is a pass/fail target for experiments, not a promised result or an assumed hardware ceiling.

## Existing evidence

Historical cold request observations are retained as anchors:

| Prompt tokens | Wall time | Derived rate |
| ---: | ---: | ---: |
| 512 | 31.946 s | 16.027 tok/s |
| 4096 | 270.215 s | 15.158 tok/s |

These are not established prefill-only baselines. Their timing may include one decode token, request setup, cache publication, graph compilation, and first-use initialization.

The available `PROFILE-MLP` sample attributes 99.93% of sampled MLP wall time to named MLP stages, with fused gate/up accounting for 86.4% of that sampled MLP time. The profiler samples the first three GDN layers and first full-attention layer, extrapolates to 48 + 16 layers, and forces evaluations that can change scheduling. These percentages therefore do not establish whole-request attribution.

The current wide custom ternary kernels cover small `M` shapes, while prefill uses large `M` through MLX quantized matrix multiplication. Gate/up fusion already exists, and `HIGGS_DENSE_FFN_GATE_UP=separate` supplies an exact existing-path control.

At the 4K historical rate, 100 tok/s requires a 6.597x total speedup and a complete request-to-first-token budget of 40.960 seconds. Exact dense work has a conditional planning range of 1.2x to 1.7x, or roughly 18 to 26 tok/s from that anchor. About 30 tok/s is a stretch milestone. No current evidence supports 100 tok/s through exact dense kernel work alone.

PFlash-L7 with roughly 10% retained blocks has theoretical target-work headroom of about 8.5x to 9.9x for eligible long-history prompts. It is approximate: omitted tokens remove GDN recurrence updates and attention KV rows. Its scoring happens outside normal AR request timing, and hard-kept current-user content may leave a single long user prompt almost uncompressed.

## Design principles

- Persist raw evidence before changing code.
- Measure request-to-first-token externally and record internal phases separately.
- Keep first-use results separate from resident-process results with fresh target prompt state.
- Change one independent variable at a time.
- Resolve existing runtime controls before writing a custom kernel.
- Use measured end-to-end time fractions for speedup projections.
- Treat exact and sparse results as different products and report them separately.
- Preserve failed and invalidated runs with the reason they were rejected.
- Run GitNexus impact analysis before any symbol edit and graph change analysis before every commit.

## Experiment 1: exact baseline and attribution

This is the first authorized GPU experiment. Use the unchanged verified starting revision.

### Configuration

- Test exact 512-token and 4096-token rendered prompts.
- Save prompt JSON, rendered token IDs, token count, and token hash.
- Generate one output token. Decode throughput is excluded.
- Disable PFlash, speculative decoding, MTP, prefix cache lookup and publication, session reuse, disk reuse, and plan reuse.
- Use fresh target prompt state for each measured request.
- Record effective runtime profile, dtype, KV policy, threshold, chunk size, gate/up selection, model snapshot, tensor manifest, binary hash, build identity, OS, MLX version, thermal state, memory pressure, and competing GPU processes.
- If the historical chunk configuration cannot be proven, declare a new baseline at threshold 1024 and chunk size 1024.
- Record every actual `query_tokens` value, body/suffix split, chunk boundary, processed-token count, and reused-token count.

### Samples

For each prompt length:

- retain process first use as a separate observation;
- warm the required shapes without retaining target prompt state;
- collect three profile-off repetitions;
- collect one profile-on diagnostic after equivalent warmup;
- persist complete stdout/stderr, `PROFILE`, `PROFILE-MLP`, phase logs, external timestamps, allocator/process memory, and cache counters.

### Baseline gate

Proceed only when:

- token hashes match across comparisons;
- processed prompt tokens equal the input count exactly once;
- reused prompt tokens are zero;
- all effective flags and versions are recorded;
- no concurrent GPU work, thermal throttle, swap, or memory-pressure event is observed;
- profile-off coefficient of variation is at most 3% at each length.

A difference greater than 10% from historical wall time requires an explanation or designation as a new baseline.

If profile-on wall time differs from profile-off by more than 5%, or extrapolated layer totals differ from measured prefill by more than 10%, profiler shares cannot size an optimization. Capture one uninstrumented Metal System Trace and use the full-request timeline for attribution.

## Experiment 2: prefill chunk sweep

Run with profiling off and the valid Experiment 1 configuration.

At 4096 tokens compare chunk sizes 512, 1024, and 2048. The 1024 baseline samples may be reused. A single initial run may reject an OOM or clearly slower size. Confirm the best contender and baseline with three alternating paired runs. Test 1536 only if the trend needs a midpoint. Test a 4096 single chunk only after 2048 passes the memory gate.

Record body/suffix partitioning, QMM `M` shapes, phase and external timing, allocator peak, process footprint, swap, and memory pressure.

Adopt a chunk size only when:

- median 4K prefill latency falls by at least 5%;
- every confirming pair improves by at least 3%;
- the 512-token control regresses by no more than 3%;
- at least 4 GiB system memory headroom remains;
- no swap-out or memory-pressure warning is attributable to the run;
- the exact correctness gate passes.

If no size passes, retain the baseline. Completion of the sweep still unlocks the next experiment.

## Experiment 3: fused versus separate gate/up

At the selected chunk size, compare the default fused gate/up path with `HIGGS_DENSE_FFN_GATE_UP=separate`.

Use fresh processes because the choice is resolved at construction. Run matched warmup and three alternating paired profile-off measurements. Apply the same performance, memory, and correctness adoption rules as the chunk sweep. Use a persisted full-request timeline to attribute any result, including repeated Hadamard transforms, allocation, and setup costs.

A custom large-`M` ternary QMM becomes eligible for design and implementation only if independently measured affected projection work accounts for at least 50% of unprofiled prefill after resolving this existing control.

## Experiment 4: exact large-M ternary QMM

This experiment requires the Experiment 3 implementation gate plus fresh GitNexus index, symbol-level upstream impact analysis, and risk review before editing.

The first prototype covers the production fused gate/up matrix `N=34816, K=5120` and down matrix `N=5120, K=17408`, using the actual group-128 packed tensors and activation dtype. It must benchmark the selected production chunk `M`, `M=512`, observed remainder shapes, and unaligned row tails.

The prototype must validate affine bias/scale relations, packed ternary codes, layout, padding, Hadamard signs, dtype, and fallback behavior. Incompatible tensors retain stock dispatch. Dequantized weights or a changed dtype cannot be presented as a pack-compatible result.

The microbenchmark passes only when:

- time-weighted affected projection cost improves by at least 1.5x;
- no required production shape slows by more than 5%;
- all numerical checks pass.

The integrated path passes only when:

- median 4K prefill latency improves by at least 10%;
- the 512-token case regresses by no more than 3%;
- the memory gate passes;
- the complete exact correctness gate passes.

Failure at either gate ends this kernel proposal. QKV fusion, fewer evaluation barriers, and whole-model compilation do not become rescue work without new attribution. QKV or evaluation work requires its own measured share of at least 10% of prefill and must deliver at least a 5% total improvement.

## Experiment 5: sparse feasibility

Sparse feasibility begins only after the exact baseline and chunk sweep. It remains a separate behavior-changing mode.

First run a 100%-survivor identity target plan against dense exact with matched prompt, plain KV storage, and fresh state. It must pass the exact numerical, state, cache, and continuation checks. This isolates sparse execution machinery from token selection. A 95% selector run is not an identity control.

Then evaluate the existing PFlash-L7 configuration with:

- fixed floor and ceiling of 0.10;
- 32-token blocks;
- exit index 7;
- eight lookahead steps;
- plan cache disabled;
- all current hard-keep protections enabled.

Use one 4096-token old-history/short-current-turn prompt and one 4096-token mostly-current-user control. Persist scored-source length, exact appended tail, final survivor positions, actual retention, scorer time, selection time, target time, fallback reason, memory, and external request-to-first-token time. The time budget includes scorer and request setup.

The 100 tok/s feasibility gate passes only if each of three fresh-prompt eligible 4K runs completes within 40.960 seconds with no reused target or plan tokens. Failure ends the 100 tok/s claim while allowing measured smaller gains to be reported. Hard-keeps and prompt family remain fixed. If fallback or scoring adds more than 5% to the matched exact time on the mostly-current-user control, the mode cannot be enabled automatically.

## Sparse behavior gate

Before a large quality run, freeze 24 adversarial cases covering:

- early, middle, and late retrieval;
- 32-token boundaries;
- multiple facts and conflicting updates;
- exact tool arguments;
- long current-user input;
- instruction hierarchy.

Any new failure where exact succeeds stops the candidate.

Release evaluation then uses at least 300 independent representative prompts with predetermined answers or rubrics. It includes at least 60 direct-retrieval or tool-argument cases and requires zero new task failures versus exact. The initial 24 adversarial cases are excluded from that statistical set. Selection policy, keep ratio, and scorer cannot be tuned on the held-out set. Behavior-affecting tuning requires a new held-out set.

Report every failure, subgroup results, retention distribution, latency distribution, and scorer-inclusive end-to-end rate. Sparse results cannot be labeled exact cold prefill.

## Exact correctness gate

An exact-computation change must preserve tensor shapes and dtypes, introduce no non-finite values, and satisfy all of these frozen tolerances against the verified baseline:

- normalized RMS error `||new-ref||2 / max(||ref||2, 1e-12) <= 0.002`;
- normalized maximum error `max|new-ref| / max(max|ref|, 1e-12) <= 0.02`;
- final-logit maximum absolute difference `<= 0.05`;
- forward KL `KL(softmax(ref) || softmax(new)) <= 1e-4`;
- identical greedy first token.

Baseline self-repeats must fit the tolerances before candidate comparison. Cover 512 and 4096 tokens, selected chunk boundary minus one and plus one, real suffix/remainder shapes, and the 24 frozen semantic cases. Verify every layer's cache length and cursor, GDN and convolution state, original position handling, and full prompt coverage. Require identical 32-token greedy continuations on all 24 semantic cases. Decode timing remains excluded.

A bitwise-exact claim requires bitwise-equal final states and logits.

## Evidence and decision records

Each experiment writes a manifest and raw artifacts to a uniquely named directory containing the revision, UTC timestamp, model identifier, and experiment name. The manifest records every command and effective environment variable needed to reproduce the run.

The summary for each experiment contains:

- the raw artifact paths and hashes;
- accepted and rejected samples with reasons;
- median, spread, pairwise comparisons, and actual token counts;
- peak memory, headroom, thermal state, and contention check;
- numerical and behavior-gate results;
- a single decision: adopt, retain baseline, stop proposal, or unlock the next experiment.

No derived percentage is reported without its timing boundary. No modeled speedup is reported as measured throughput.

## Exclusions

This program does not authorize:

- decode optimization;
- advertising prefix reuse as cold prefill throughput;
- removal of sparse hard-keeps to improve a benchmark;
- layer or channel skipping beyond the defined sparse feasibility experiment;
- new QKV fusion without measured attribution;
- reduced evaluation frequency without measured scheduling cost and cache-state proof;
- whole-model compilation;
- broad kernel implementation before the preceding experiment gate passes.

## Completion

The exact track completes when the best gated exact configuration is measured and either adopted or rejected with preserved evidence. The sparse track completes when scorer-inclusive feasibility and behavior gates pass or the proposal is stopped.

The final report states separately:

- exact 512-token and 4096-token prefill rates;
- sparse eligible and protected-control rates;
- request-to-first-token boundaries;
- numerical and behavior status;
- whether the 100 tok/s target was reached, and under which exact workload.
