# Session Context Governor Brief

Date: 2026-07-22

This is the working design brief for agents investigating Higgs + Nanobot long-session
latency, retained KV cliffs, PFlash session bootstrap, LCM admission, and duplicate tool
loops. Keep changes source-grounded. Do not use stale docs as authority when current
source disagrees.

## Live Failure Being Designed Against

Live Higgs logs showed a retained session growing normally:

- `retained_tokens=24038`
- `candidate_tokens=25332`
- then next request: `retained_tokens=0`, `outcome=ExactBootstrap`
- reason: `ColdPromptTooLarge { prompt_tokens: 26497, max_prefill_tokens: 8192 }`

Current active Higgs config confirms the local cliff:

- `kv_max_sessions = 1`
- `kv_max_session_tokens = 24576`
- `kv_cache_bytes = 1073741824`
- `prefill_compression = "auto"`
- `prefill_keep_ratio = 0.10`

The observed behavior is not mysterious: retained KV hit the per-session cap, the
retained session disappeared, and the next session request exact-prefilled a 26k prompt.

## Source-Grounded Current Behavior

Higgs:

- `crates/higgs-engine/src/simple.rs`
  - `session_prefill_strategy`: cold prompt over suffix cap becomes `BootstrapExact`.
  - `stash_into`: removes retained session when `state.tokens().len() > max_session_tokens`.
  - `pflash_compress_if_eligible`: rejects requests with `session_id.is_some()`.
  - session routed paths call `generate_continued_impl_locked`, which exact-prefills when
    `take_continuable` returns `None`.
- `crates/higgs/src/routes/chat.rs`
  - `session_id` opts into retained-session generation when request shape is compatible.
  - incompatible request shapes fall back to stateless, where radix/PFlash can still run.
- `crates/higgs-models/src/turboquant.rs`
  - default KV mode is `Off`; TurboQuant is explicit.
  - `max_session_tokens = 0` means unlimited in config.
- `crates/higgs-engine/src/cache/paired.rs`
  - target-only retained state may use the historical TurboQuant cap exemption.
  - paired target+dSpark retained state may not, because the dSpark snapshot remains
    uncompressed and grows with context.

Nanobot:

- `src/agent/lcm.rs`
  - LCM triggers on estimated active conversation tokens versus Nanobot context budget.
  - It does not know Higgs retained-session token cap.
- `src/agent/agent_loop/shared.rs`
  - prefix cache watermark protects already-sent prompt prefix.
  - pending LCM checkpoints can be deferred to avoid warm-cache invalidation.
  - Higgs session ids are epoch-derived and old epochs can be dropped.
- `src/session/filters.rs`
  - history filtering preserves cache-stable windows and caps replayed tool bodies.
- `src/agent/tool_guard.rs`
  - current in-turn duplicate guard is call-local and result-local.
  - read-only calls can be blocked after cached result is present.
- `src/agent/tools/stash_search.rs`
  - large stashed tool outputs can be searched/sliced without recalling full bodies.

## Design Principle

Split three concerns that are currently coupled:

1. Conversation identity: "this is the same user session."
2. Exact retained KV continuity: "the next prompt is an exact prefix extension."
3. Prefill strategy: exact, PFlash compressed, compaction checkpoint, or reject.

`session_id` must not mean "forbid PFlash forever." It should identify continuity;
the admission controller chooses the prefill strategy.

## Target Architecture

Introduce a cross-layer Local Inference Admission Controller.

Nanobot owns foreground admission before each local LLM call:

- Estimate full rendered prompt tokens.
- Know or derive Higgs limits:
  - model context window
  - retained session token cap
  - max exact suffix prefill
  - PFlash availability
  - current retained token count or last observed retained/cached tokens
- Decide one of:
  - `ContinueExact`: send same Higgs session epoch.
  - `StartAsyncCompaction`: below hard cap but approaching pressure.
  - `BlockForCompaction`: prompt would cross retained/session cap.
  - `ResetEpochAfterCompaction`: install LCM checkpoint, bump Higgs session epoch, drop old retained session id.
  - `AllowPFlashBootstrap`: emergency path when foreground prompt is already cold/oversized.

Higgs owns execution after admission:

- Keep exact retained continuation as the fast path.
- Add explicit `PFlashBootstrap` for cold oversized compatible session requests.
- Do not publish PFlash-compressed/sparse KV as exact retained KV unless retained state
  carries a compressed/survivor manifest and future continuations validate against it.
- Emit trace outcome and metrics that Nanobot can consume.

## Required Invariants

- Never silently treat compressed/sparse PFlash prompt KV as exact retained session KV.
- Never drop old KV without also changing the prompt through a lossless summary/snapshot.
- Never let a prompt reach `ColdPromptTooLarge` exact bootstrap when PFlash is available and
  the request shape is compatible.
- Nanobot compaction must be installed before foreground inference when predicted prompt
  pressure exceeds the Higgs retained-session hard threshold.
- Duplicate read/search tool calls must not execute again after an identical successful
  result exists for the same mutation epoch.
- Tool-result retrieval must prefer bounded `search_tool_result` or `slice_tool_result`
  over full recall.

## Workstreams

### H1: Higgs Session Admission Outcomes

Owner files:

- `crates/higgs-engine/src/simple.rs`
- `crates/higgs/src/routes/chat.rs`

Add explicit session outcomes:

- `Continued`
- `ExactBootstrap`
- `PFlashBootstrap`
- `RejectedNeedsCompaction` if we decide to expose hard rejection later

Tests first:

- cold oversized session with PFlash enabled chooses `PFlashBootstrap`, not `ExactBootstrap`.
- incompatible session request still follows exact/stateless compatibility rules.
- metrics distinguish exact bootstrap from PFlash bootstrap.

Risk:

- Serving path and retained-session map. Run GitNexus impact before editing indexed owners.
- If private helper symbols are not indexed, run impact on closest indexed owner and document
  the gap before changing code.

### H2: Higgs PFlash Bootstrap Execution

Owner files:

- `crates/higgs-engine/src/simple.rs`

Implement an emergency path for cold oversized compatible session prompts:

- Permit PFlash planning even when the request has `session_id`, but only under an explicit
  bootstrap mode.
- Run compressed prefill using existing stateless PFlash machinery.
- For dSpark:
  - use dSpark only if sparse target taps are available for the PFlash plan.
  - otherwise use AR for that turn.
- Do not stash the resulting compressed/sparse prompt as exact retained state in v1.
- Return usage/trace that tells the caller this was a degraded bootstrap.

Tests first:

- PFlash bootstrap returns no exact retained-session publication.
- dSpark eligibility follows existing sparse-tap rules.
- required PFlash errors are visible, not silently exact-prefilled.

### N1: Nanobot Higgs Capability And Pressure Model

Owner files:

- `src/providers/openai_compat.rs`
- `src/agent/agent_core.rs`
- `src/agent/agent_loop/shared.rs`
- config schema if new knobs are needed

Needed data:

- configured/local Higgs retained cap
- session epoch
- last prompt tokens
- last retained/cached tokens
- last prefilled tokens
- last session outcome
- tool definition token cost

The first version may use config/static capabilities. The better version should consume Higgs
metrics or a small capabilities endpoint.

Tests first:

- pressure threshold uses the lower of Nanobot context budget and Higgs retained cap.
- at 80-90 percent of retained cap, Nanobot schedules or blocks for compaction.
- after compaction, Nanobot bumps Higgs session epoch and sends drop ids for old epochs.

### N2: Nanobot Foreground Compaction Admission

Owner files:

- `src/agent/agent_loop/shared.rs`
- `src/agent/lcm.rs`
- `src/agent/agent_core.rs`

Change policy:

- LCM should not trigger only from Nanobot's max context window.
- Add a retained-KV pressure threshold:
  - soft: start async compaction before cap pressure matters.
  - hard: block/install compaction before foreground inference.
- Pending compaction should stop being deferred if preserving the warm prefix would cross the
  Higgs retained cap. One re-prefill of compacted context is cheaper than a 26k cold exact prefill.

Tests first:

- a prompt below Nanobot max context but above Higgs retained hard threshold compacts before
  foreground call.
- pending LCM checkpoint installs under retained-KV hard pressure even with a warm prefix.
- old session epoch is queued for Higgs drop after checkpoint.

### T1: Tool Result Ledger And Duplicate Read/Search Control

Owner files:

- `src/agent/tool_guard.rs`
- `src/agent/router.rs`
- `src/agent/tool_engine.rs`
- `src/session/db.rs` if durable ledger is needed

Design:

- Canonical tool key = tool name + canonical JSON args + mutation epoch.
- Writes and shell execution advance mutation epoch for filesystem-affecting reads.
- Repeated read/search after successful result returns a synthetic receipt, not a new execution:
  - existing `tool_call_id`
  - result preview status
  - "use search_tool_result or slice_tool_result" guidance for large outputs
- This must work across reordered JSON args and within a single tool loop.

Tests first:

- reordered args dedupe.
- duplicate search_files does not execute after successful result.
- write/apply_patch invalidates stale read/search keys.
- synthetic duplicate receipt does not balloon context.

### O1: Observability

Expose enough signal to debug live sessions without screenshots:

- Higgs trace outcome: exact continue, exact bootstrap, PFlash bootstrap.
- retained cap and retained tokens.
- PFlash attempted/used/fallback reason.
- Nanobot admission decision and thresholds.
- compaction pending/in-flight/installed plus reason.
- duplicate tool call blocked/executed status.

## KV Cache Limit Answer

Current live cap is not 64k. It is `kv_max_session_tokens = 24576`.

Mechanically, the config allows larger values, and `0` means unlimited. But the practical limit
is set by memory pressure, model context length, paired dSpark retention, MLX/Metal allocation
behavior, and thermal throughput.

Important distinction:

- `kv_max_session_tokens` bounds retained per-session KV continuity.
- `kv_cache_bytes` is the resident-byte budget for prefix/radix cache entries. It is not the
  only bound on the retained session map.

Rough dense target KV lower-bound for Bonsai/Qwen3.5-style metadata from logs:

- layers: 64
- KV heads: 4
- assume head dim: 128
- K and V
- fp16 bytes: 2

Approximate dense target KV per token:

`64 layers * 2 K/V * 4 kv_heads * 128 dim * 2 bytes = 131072 bytes/token`

That is about:

- 24k tokens: roughly 3 GiB target KV
- 64k tokens: roughly 8 GiB target KV

This excludes dSpark retained snapshot/state, temporary prefill allocations, generated-token
growth, prefix cache entries, model weights, PFlash scorer, OS pressure, and low-power thermal
effects. TurboQuant can reduce target-only retained KV, but the current paired dSpark retained
path cannot simply exempt itself from the token cap because dSpark side state remains uncompressed.

Recommendation:

- Do not make 64k retained KV the main solution.
- Use 64k only as an experimental stress cap with memory/thermal telemetry.
- The structural answer is compaction/admission before the retained cap, plus PFlash emergency
  bootstrap if a cold oversized request still slips through.

## Suggested Implementation Order

1. Add tests for Higgs outcomes and Nanobot retained-cap admission.
2. Implement Higgs `PFlashBootstrap` as explicit degraded bootstrap with no exact publication.
3. Add Nanobot pressure model using static config or discovered Higgs capabilities.
4. Install LCM checkpoint before foreground inference under retained-cap hard pressure.
5. Add durable/in-turn tool ledger for duplicate reads/searches.
6. Add metrics and live log assertions.
7. Only then tune caps such as 24k, 32k, or 64k.

## Non-Goals For First Pass

- Do not implement arbitrary KV eviction without prompt rewriting.
- Do not silently publish compressed/sparse PFlash state as exact retained KV.
- Do not solve long sessions by only increasing `kv_max_session_tokens`.
- Do not depend on stale docs when source and logs are available.

## Agent Operating Rules

- Run GitNexus impact before editing any indexed symbol.
- Warn on HIGH or CRITICAL impact before editing.
- Prefer TDD red-to-green.
- Keep workstreams disjoint where possible.
- Do not revert unrelated dirty worktree changes.
- Report exact files changed and tests run.
