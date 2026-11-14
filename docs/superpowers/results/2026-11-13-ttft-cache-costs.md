# Clover TTFT: radix versus retained sessions

2026-11-13, M4 MacBook, 32 GiB unified memory. Base commit: `a73e40dd8b6b04626f58c1983c5fef83ec388c87`. Isolated worktree: branch `perf/ttft-cache-costs`. Installed binary and user configuration were not replaced. The previous port-9000 process stopped during investigation; these measurements used an isolated port-19001 server, now stopped.

## Findings

The observed Nodecode prefix loss and the extra cost of radix continuation are different problems. Historical request reconstruction showed changing tool-description order early in the prompt: reuse fell to 1,344 tokens despite a roughly 21k-token prompt. Identical cache keys cannot make different token sequences reusable. That turn's roughly 72-second TTFT was dominated by recomputing the suffix, rather than trie lookup or copying cache tensors. A hit counter counts even a short usable prefix; use cached tokens, uncached tokens and elapsed prefill together.

The controlled benchmark here holds prompt, model, reasoning, output limit and suffix constant. It does not reproduce the full Nodecode workload or establish a causal harness comparison. Both Clover and MiniCPM5 were resident. All 84 HTTP requests recorded zero other requests. Both seed requests are excluded from the table. The retained seed can hit the radix entry created by the preceding seed and is not a cold comparison; first-use shader compilation and weight warmup varied greatly and are not optimization wins.

| Run | Starting prefix | Radix median TTFT | Retained median TTFT |
|---|---:|---:|---:|
| Stock A | ~2k | 891 ms | 660 ms |
| Candidate B | ~2k | 864 ms | 663 ms |
| Stock C | ~2k | 872 ms | 660 ms |
| Stock A | ~8k | 983 ms | 770 ms |
| Candidate B | ~8k | 1,003 ms | 791 ms |
| Stock C | ~8k | 1,032 ms | 806 ms |

Each median covers six follow-ups. Each adds 150 prompt tokens, requests no thinking, temperature zero, self-spec, and at most four output tokens. Inputs use a canonical assistant reply `OK` for both routes. Each run starts a fresh server/cache. Radix always runs first in each pair; order bias is a remaining limitation. There was no simultaneous compilation during these measurements. Thermal counters were unavailable; the unchanged retained control exposes some run-to-run drift. These are exploratory measurements, not confidence intervals.

Retained reuse saved about 0.2–0.25 seconds on this workload. Example final ~2k turn: prompt 2,988 tokens; radix cached 2,816, retained cached 2,838. The historical multi-second gap between harnesses is much larger than the controlled route difference and cannot be attributed to cache transport alone.

## Why the paths differ

### Radix

`PagedPrefixCache::find_longest_prefix` matches exact token sequences. Clover combines attention KV with recurrent GDN state. KV can be sliced into blocks, but the GDN snapshot is valid only at its stored endpoint; arbitrary mid-edge reuse would require a state that is not stored. Therefore the deepest token match is not necessarily a usable hybrid checkpoint.

Restoration gathers attention blocks into a request-local KV cache and deep-clones recurrent state. Clover self-spec stock-prefills the uncached body to a block-aligned recurring boundary, snapshots that stock state, then forwards the remaining alignment slack and generation prompt separately. Publication can share attention blocks across branches, while each recurrent endpoint requires its own state.

Early prompt edits or tool-order changes leave a short reusable prefix. Context compaction and history eviction also change tokens and must invalidate the affected suffix. Reusing a session ID or cache key cannot bypass this correctness constraint.

### Retained continuation

`fork_target_continuable` looks up the session, validates the prior prompt against the new tokens (including the supported continuation reconciliation), and forks a complete cache. This is not zero-copy. The old checkpoint survives cancellation until a successor is published.

`generate_continued_self_spec_locked` forwards the uncached suffix with `store=false`, avoiding radix body/tail splitting and publication. It can reuse the entire preceding prompt, including its generation suffix, rather than the earlier block-aligned radix endpoint. It publishes a stock prompt snapshot before speculative verification.

Crucially, current Clover self-spec retains the input prompt checkpoint, not its fast speculative completion state. The next turn recomputes the previous answer along with new tool/user text. Long generated answers can therefore make the next TTFT expensive even when continuation succeeds. Other execution modes need separate measurement; this finding is specifically for this Clover path.

## Copying and memory costs

Cache-only synthetic measurements use full Clover geometry: 40 layers, 10 attention KV layers, 30 recurrent layers; FP16 KV, FP32 recurrent state. They load no weights and synchronize evaluated outputs. Initial iteration is excluded. Input `ones` tensors may be broadcast views, so these numbers characterize the tested operations rather than real production DRAM throughput.

| Context | Logical checkpoint bytes | Retained-style fork | Radix store | Radix restore |
|---|---:|---:|---:|---:|
| 2,048 | 107,806,720 (~103 MiB) | ~16 ms | ~22 ms | ~13 ms |
| 8,192 | 233,635,840 (~223 MiB) | ~20 ms | ~45 ms | ~17 ms |

Actual phase tracing showed snapshots roughly 27–64 ms in one run; uncached forwards cost hundreds of milliseconds. These operations are not interchangeable isolated samples, and total radix cost also includes snapshot/publication.

Logical Clover checkpoint size is `20,480 * tokens + 65,863,680` bytes at this dtype/geometry. At 32,768 tokens this is 736,952,320 bytes (~703 MiB), excluding weights, allocator capacity/slack, metadata, temporary activations, and drafts. A retained session holds a full checkpoint and temporarily forks another live cache. Radix can share old attention blocks between endpoints but pays ~63 MiB recurrent state per stored endpoint. Duplicate retained/radix checkpoints, additional sessions and transient live/snapshot copies all matter; a 4 GiB configured cache budget is not a 4 GiB total-process memory cap.

## Hillclimb outcome

Tested using existing fast-row kernels only for the disposable tail after a stock shared checkpoint. A real four-layer Clover regression first failed on stock code, then passed with the candidate; shared checkpoints stayed bit-identical to stock. All four candidate cache tests and 68 paged-prefix tests passed. After removal, all three existing real-fixture cache regressions passed again. Independent review found no cache-corruption concern in the final diagnostics-only diff. All 36 recorded retained follow-ups were audited for a continuation increment, zero prefix-miss increment and the expected prior-prompt token boundary.

An initial comparison appeared to improve radix by ~33%, but the unchanged retained control improved ~22% too. Repeating stock/candidate/stock reduced the result to roughly 2–3% at ~2k and no convincing benefit at ~8k. The numerical behavior change was rejected and removed. The final code only adds environment-gated phase diagnostics (`HIGGS_DIAG_SESSION_TIMING=1`); it does not change cache semantics or kernel selection. Timers include synchronization performed by the existing forward path; `body_and_snapshot_ms` includes snapshot duration, so do not add those fields together.

Next work, in order:

1. Stabilize the harness's early tool/prompt rendering. Keep descriptions and ordering stable; put changing context late. Do this at the source rather than silently rewriting user message content in Higgs.
2. Use retained session continuation for stable conversations, with the existing exact-token guard and radix fallback. Track previous-answer recomputation separately from truly new input.
3. Investigate uncached stock prefill using real Nodecode prompt lengths, suffix sizes and tool-turn traces. Small copy savings cannot fix tens of thousands of unnecessarily recomputed tokens.
4. Consider an exact committed completion checkpoint only with a demonstrated stock-equivalent state; publishing fast verification state directly would violate the existing reusable-cache contract.

## Reproduction and evidence

- `crates/higgs-engine/examples/ttft_cache_costs.rs`: `cargo run --offline --release -p higgs-engine --example ttft_cache_costs -- 2048 8192`.
- `crates/higgs-engine/examples/ttft_cache_routes.py`: run against a fresh isolated Clover server with `--base-url http://127.0.0.1:19001 --output result.json --prefix-words 2048 --turns 6 --case-key ttft-reference-repeat-2048`. Use identical case keys and fresh servers for comparisons. Repeat with 8192.
- Raw JSON and summarized timings: [ttft-cache-costs](ttft-cache-costs/summary.json); cache-only operations: [cache-only.jsonl](ttft-cache-costs/cache-only.jsonl).
- Experimental stock/fast executables and full server traces remain under `/private/tmp/higgs-ttft-*`; they are not installed artifacts. The rejected fast-tail change is described above rather than retained in production source.
- GitNexus upstream impact for `self_spec_stock_prefill`: CRITICAL, ten impacted nodes, two direct callers, five processes; warned before edits. Broader paged-cache operations were also high/critical and were not edited. Graph change analysis is complete with no partial/truncation flags. Final diagnostics still touch a high-impact path and require normal review before integration.
