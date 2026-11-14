# Higgs TTFT cache cost hillclimb

Compare radix prefix reuse and retained session continuation on identical Clover inputs. Preserve cache independence, cancellation fallback, numerical behavior and memory accounting. Treat matching, restoration, suffix prefill, publication and first-token generation as separate costs; a cache hit alone is not a performance result.

## Steps

1. Refresh the GitNexus index including large inference files. Trace both cache paths and run upstream impact analysis before production edits.
2. Build a clean baseline. Add a bounded synthetic cache benchmark at Clover geometry (40 layers, 10 attention layers, 30 recurrent layers), measuring snapshot, radix store, radix restore and retained-style deep clone at multiple context lengths. Run without a second model load. Report synchronization and physical byte costs.
3. Measure actual TTFT on equivalent requests using radix versus numeric session IDs, with identical reasoning/speculation settings and suffixes. Record cached and uncached token counts, queue interference and machine pressure; reject contaminated comparisons. Do not benchmark while a user request is active.
4. Profile the dominant measured cache phase. Add a regression test for the smallest safe optimization, verify it fails, implement, then rerun correctness tests and the same benchmark. Retain only measured wins.
5. Record before/after results, explain hybrid exact-endpoint matching and retained-session reconciliation costs, and produce a reviewable candidate. Do not replace the running binary until the candidate and rollback are ready.

## Validation

Cache reconstruction must preserve values, offsets and isolation when the source or returned live cache is mutated. Include paged and whole-clone fallback behavior, recurrent layers, and existing cache tests. Report CPU-only checks separately from Metal-dependent tests. Performance comparisons use repeated runs after warmup and give context length, dtype, retained bytes and active memory. Run graph change analysis on the actual worktree before any commit.

## Outcome

Completed the cache benchmark, isolated HTTP comparison, independent review and regression checks. The disposable-tail optimization was rejected after stock/candidate/stock repeats failed to demonstrate a convincing gain. Retained phase instrumentation and reproducible benchmark/evidence files; no binary installation or production configuration change. Results: `docs/superpowers/results/2026-11-13-ttft-cache-costs.md`.
