# M4 prefill investigation

**Goal:** Explain the corrected 32K/45K baseline, measure bounded GPU tuning,
and collect sparse-attention feasibility/quality evidence before choosing a
larger optimization. User authorized stages 1–3 in this task.

**Architecture:** Reuse the existing release server, explicit private config,
request payloads, memory watchdog and profiling hooks. Keep diagnostic timings
separate from uninstrumented HTTP latency. Reuse the existing indexless design
and offline probes; no approximate serving default is installed from screening.

## Constraints

- Work in the existing isolated worktree; preserve all preexisting edits.
- Same checkpoint and request bytes; pinned binary SHA256 and environment.
- Match `raise_wired_limit=true`, dense KV and cold prompt cache.
- Serialize all GPU/model/build/trace/indexer runs; stop only owned processes.
- Reject timed runs with new system swapouts; retain every failure.
- Record power, physical footprint, semantic checks and cached-token counts.
- Compare uninstrumented controls around candidates; report drift and single
  observations without presenting them as a stable speedup distribution.
- Graph impact before symbol edits; complete graph checks before commits.

## Tasks

- [x] 1. Profile the existing serving binary at long context with HIGGS_PROFILE.
  Parse per-cycle GDN, full-attention and MLP timings across token positions.
  Check available native trace tooling. Quantify attribution limitations.
- [x] 2. First isolate the existing serial-token Metal GDN recurrence and
  benchmark exact launch/register-layout variants on identical inputs and final
  state. A parallel tiled recurrence is a separate algorithm experiment, not
  equivalent to outer prefill chunk tuning. Then screen existing GPU controls,
  including prefill chunk sizes 512,
  1024 and 2048 if memory permits. Use one variable per arm and matched request
  payloads. Bracket a promising candidate with controls at 32K/45K; reject
  memory or retrieval regressions. Follow the profile if another existing
  control offers a better bounded experiment.
- [x] 3. Inspect/reuse the indexless feasibility probe and real QKV trace hooks.
  Measure dense versus reduced-work attention cost at relevant shapes and,
  where captures are available, approximation error on real model activations.
  Separate oracle/gather ceilings from actual Metal kernel speed. If a gate
  fails, retain the negative result and quantify the next useful experiment.
- [x] 4. Independently review data interpretation, preserve raw evidence,
  report attainable whole-request bounds and unresolved quality requirements,
  verify user changes preserved, and commit scoped experiment artifacts.

No end-to-end sparse speedup is claimed without an implemented path, matched
serving measurements and behavioral evaluation. Two retrieval facts alone are
not a broad quality gate. Existing baseline runs are context, not paired arms.

## New external evidence

The linked Strix Halo journey was supplied during execution. Its reported
ablation makes tiled GDN recurrence the first exact-kernel hypothesis and leaves
sparse attention trace-gated. Its 191.25→1160.02 tok/s aggregate is not an M4
prediction or a product of independent portable speedups. WMMA/DPP/LDS/PM4,
IOMMU settings and Linux scattered-file-read behavior are not direct Metal
implementations. BF16/dequant caching must preserve the checkpoint's resolved
weights and fit the memory gate. Token pruning/PFlash and KVA remain separate
approximate experiments with model-level quality gates. This campaign measures
the existing full-token attention approximation oracle only, not KVA or PFlash.

## Measured outcome

See `benchmarks/prefill_investigation/RESULTS.md`. All four 32K arms completed,
but 19% drift between identical controls invalidates a chunk speedup claim;
no candidate qualified for the prepared 45K follow-up. Corrected GDN launch
pairs are neutral. The scalar sparse kernel loses even at nominal 12.5% block
density; real-activation approximation error is strongly layer/head dependent.
The next kernel hypothesis is exact FP32/D256 tiled attention, with separate
GDN algorithm and approximation work rather than enabling the failed probes.
