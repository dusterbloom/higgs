# `prompt_cache_key` support — design + plan

Status: **SHELVED after Phase 0 + codex review** (see Outcome at the end).
Date: 2026-10-27.

## Problem

nodecode (and any OpenAI-wire agent) sends `prompt_cache_key` on every
`/v1/chat/completions` request. Higgs drops it: `ChatCompletionRequest`
(`crates/higgs/src/types/openai.rs:5-119`) only understands a numeric
`session_id`. Result in the 2026-10-27 nodecode run: every turn logged
`large stateless prefill: no session controls`. Prompts grew 8.4k → 24.7k
tokens, and the run ended in a Metal OOM abort during a 24.7k-token prefill.

nodecode key semantics (`~/.local/share/nodecode/current/src/src/engine/provider.lisp:6140`):
`sha256(version, provider, model, credential-discriminator, base system prompt, tool names)`.
**It is one key per static prompt, shared by every conversation, fork and
subagent.** It is deliberately not a session identity. It is a
cache-routing namespace, matching OpenAI's semantics.

Consequence: mapping `prompt_cache_key → session_id` directly is wrong.
Concurrent conversations would collide on one retained session and get 409s
from the divergence guard (`simple.rs:4716-4752`).

## What already exists (do not rebuild)

- Stateless radix prefix cache: `PagedPrefixCache`, token-exact, 32-token
  blocks, up to 8 endpoints, `kv_cache_bytes` budget (`simple.rs:5576-5722`,
  `paged_prefix_cache.rs:681-860`).
- Hybrid (Qwen3.6 / `Qwen3Next`, GDN state) stores **one snapshot per
  request** at `hybrid_checkpoint_boundary(body_len)` (floor 32,
  `simple.rs:178`, `:5817-5905`). Lookup accepts only exact stored endpoints
  (`paged_prefix_cache.rs:751`, `:828`). So:
  - same conversation, turn N+1: **should hit** (turn N's body is a prefix);
  - new conversation, fork or subagent sharing only system+tools: **always
    misses**.
- Retained sessions (`retained: HashMap<u64, RetainedKv>`). They keep
  generated tokens, use strict-prefix validation, allow `kv_max_sessions`
  with LRU, and are bounded by `kv_max_retained_bytes`.
- `/metrics` JSON exposes `CacheMetrics` (`radix_lookups`, `radix_hits`,
  `prefill_saved_tokens`, …). Hits are logged at `debug!` only. Misses have no
  log line.

## Unknowns that gate the design

The run's logs never showed whether the stateless radix actually hit across
nodecode turns. Two things could defeat it:

- nodecode window eviction or tool-result thinning rewriting earlier text;
- template re-rendering, e.g. the Qwen template dropping earlier assistant
  `<think>` content.

Either would move the divergence point before the stored endpoint. **Phase 0
measures this before any feature work.**

## Design

### D1. Parse and surface

- Add `prompt_cache_key: Option<String>` to `ChatCompletionRequest`.
- Validate it: at most 256 bytes; treat empty as absent.
- It is never a correctness input. Radix and retained matching stay
  token-exact. The key only scopes **routing, checkpoint placement and
  eviction**.
- The Anthropic route is out of scope, because that API uses `cache_control`.

### D2. Implicit retained sessions (uses the existing KV session system)

When a request has `prompt_cache_key`, no `session_id`, and no feature from
`session_routing_disabled_reasons`:

1. **Resolve.** Among retained sessions tagged with this key, pick the one
   whose retained tokens are the **longest strict prefix** of the new prompt
   tokens. Skip sessions whose lock is held, so concurrent conversations never
   share one. Reuse `retained_continuation_prior_len` / the
   `retained_session_prefix_len` hint.
2. **Hit.** Route through `generate_session_routed_*` with that internal id.
   Generated tokens are reused, so only the new tool result or user text is
   prefilled. The radix path can't do this, because it never stores
   generated tokens.
3. **Miss or divergence.** Never return 409. The client did not ask for a
   session, so fall through to the stateless radix path. Then store the
   result as a **new** implicit session under the key.
   - Allocate implicit ids from a reserved range (top bit set) so they can
     never collide with client-chosen `session_id`s.
   - Divergence of an implicit candidate only means "not this one". It does
     not delete the candidate.
4. **Limits.** Existing `kv_max_sessions`, `kv_max_retained_bytes`,
   `kv_retained_idle_secs` and LRU apply unchanged. Implicit sessions evict
   before explicit ones at the same LRU age, because explicit sessions are a
   stronger client signal.
5. **Response.** `usage.prompt_tokens_details.cached_tokens` reports the reused
   tokens. Do not echo the internal id. It is invisible to clients.

Everything here composes existing machinery. The new parts are a
`key → Vec<u64>` index next to `retained`, plus a resolver.

### D3. Learned shared-prefix checkpoint (hybrid cross-conversation reuse)

The goal is a radix hit for new conversations and subagents on the ~8k-token
static prefix.

- Per key, remember the last few stored prompt token prefixes. A cheap
  option is block hashes, which the radix already computes.
- On a stateless miss under key K, compute the LCP with K's previous prompts.
  If `LCP ≥ HIGGS_PCK_MIN_SHARED_TOKENS` (default 1024), record
  `boundary[K] = floor32(LCP)`.
- Future hybrid prefills under K whose prompt extends past `boundary[K]` take
  an **extra** `deep_clone` snapshot at that boundary. This makes the existing
  two-phase prefill three-phase: shared boundary, then body boundary, then
  suffix. Store it as its own radix endpoint.
- The boundary endpoint gets eviction priority over per-turn endpoints under
  the same key (pinned-by-key, still under `kv_cache_bytes`).
- No template parsing is needed: the boundary is learned from real token LCPs.
- Dense models already get partial-prefix reuse, so D3 is a no-op for them.

### D4. Observability (Phase 0 deliverable, kept permanently)

- `info!` per request: `prefix_cache=hit|miss|bypass`, `reused_tokens`,
  `prompt_tokens`, `prompt_cache_key` (first 8 chars), and the resolved
  path (`implicit_session|radix|cold`).
- On a miss: the LCP against the nearest stored endpoint, so "diverged at
  token 412 of 19k" is visible.
- `set_cached_tokens` on non-streaming paths too. It is missing today, at
  `chat.rs:1848` (streaming only).

## Non-goals

- Mapping `prompt_cache_key` to a client-visible session.
- Batch engine (`state.rs:1249-1290` ignores the prefix-cache args).
  Follow up only if Phase 0 shows batch in use.
- Fixing the OOM abort. That is a separate track: a free-memory admission
  guard for large non-PFlash prefills, extending the PFlash guard at
  `simple.rs:9886`. It is listed so it isn't lost.
- Suffix Cache Reuse (`docs/suffix-cache-reuse.md`) for mid-prompt edits.

## Plan

Each phase ends with its gate. Don't start the next phase if a gate fails.

### Phase 0 — measure (half a day)

1. Implement D4 logging (route + engine), with no behaviour change.
2. Replay a real nodecode session against Escha. Capture with
   `HIGGS_LOG=info` and `curl :9000/metrics`.
3. Report a table: per turn, `prompt_tokens`, `reused_tokens`, path, and
   divergence point.
4. **Gate.** Branch on what the table shows:
   - Same-conversation reuse ≥ 80%: D2 is a small win (generated tokens only),
     so do D3 first.
   - Reuse is low because of early divergence: root-cause it (template or
     nodecode rewrite) before D2 or D3. Neither feature helps if tokens
     diverge early.

### Phase 1 — D1 + D2 implicit sessions

- Types: field, validation, and a `doctor` check if any config knob is
  added.
- Engine: `implicit_index: Mutex<HashMap<String, Vec<u64>>>`, resolver, and
  reserved id allocator. Hook into the existing retained stash/evict paths so
  the index never holds dead ids.
- Route: resolution before the `session_continuation_id` branch, in both
  streaming and non-streaming flows.
- Tests (minimal, general):
  - Resolver unit test: three retained sessions under one key; the prompt
    extends B. Expect B to be chosen, a locked session to be skipped, and a
    diverged session to cause a fallthrough without deletion.
  - Route test with `Engine::test_stub`: two interleaved conversations on the
    same key. Expect no 409, each routed to its own session, and
    `cached_tokens > 0` on turn 2 of each.
- **Gate.** Replay of the Phase 0 session shows end-to-end TTFT before/after
  as a table, and greedy output identical to cold.

### Phase 2 — D3 learned shared-prefix checkpoint

- Engine: per-key boundary learning, the extra hybrid snapshot in the
  prefill, and pinned eviction.
- Tests:
  - Unit: the LCP → boundary rule (floor 32, minimum threshold).
  - Ignored real-model test, extending `cross_turn_hybrid_reuse.rs`: a
    conversation B with a different first user turn but the same
    system+tools hits at the boundary, and greedy output equals cold.
- **Gate.** A new nodecode subagent or conversation shows
  `reused_tokens ≈ boundary` on its first request. Report end-to-end TTFT.

### Phase 3 — docs

- README (request fields table), the `daemon.rs` template if a knob is
  added, and doc comments.
- `cargo clippy -p higgs`, `cargo fmt -p higgs -- --check`,
  `cargo test -p higgs -- --test-threads=1`, plus engine tests.

## Risks / questions for review

1. Memory. Implicit sessions plus a pinned boundary snapshot add retained KV
   on a box that just OOM'd. Both stay under existing budgets, but should D2
   default off until the OOM guard lands?
2. The extra `deep_clone` snapshot costs time and memory during prefill.
   Does the GDN state clone stay small enough at 35B-A3B?
3. Template stability. Does retained-session text reconciliation
   (`continued_prompt_tokens_from_retained`, `simple.rs:7263`) cope with the
   Qwen template dropping earlier `<think>` blocks? If not, implicit sessions
   will always miss on turn N+2.
4. Is a per-key LRU index the right granularity, or should resolution scan
   all retained sessions (key as filter only)?
5. Strict-prefix scan cost: O(sessions × prompt_len) token compare per
   request. With `kv_max_sessions` ≤ ~8 it's negligible; confirm.

## Outcome (2026-10-27)

### Phase 0 answered from existing data

nodecode records higgs's `usage.prompt_tokens_details.cached_tokens` per
round. They are stored as `turn.usage` events in `~/.nodecode/history.db`.
The crashing session (Escha, `s-…ES60QF`) shows the stateless radix cache
hitting on every round after the first:

| Round | Prompt | Cached | Newly prefilled |
|---|---|---|---|
| 10:52:22 | 11,075 | 10,656 | 256 |
| 10:53:00 | 13,839 | 10,880 | 2,798 |
| 10:54:11 | 19,250 | 13,664 | 5,546 |
| 10:54:30 | 19,453 | 19,296 | 119 |
| 10:54:39 (crash) | 24,737 | ~19.4k expected | ~5.3k |

Same-conversation reuse was 70–99%. New conversations cold-prefill only
about 1.5–1.7k tokens of system prompt. The premise of this plan ("nodecode
bypasses the KV system") was wrong. The `large stateless prefill` warning fires
before radix lookup (`chat.rs:2374`), so it does not mean "cold".

### Codex review (gpt-6-sol, read-only) — key findings

- D2 has no reservation between resolve and worker. Two requests can pick the
  same idle id, and the divergence guard then deletes it (`simple.rs:6925`,
  `:4716`).
- No reserved id range exists. `session_id` accepts any `u64` from clients.
- A stateless miss returns no KV to stash. "Store as implicit session" would
  be a new engine path (`engine.rs:238`).
- D3's extra `deep_clone` copies KV and GDN state while prefill is live. Peak
  GPU memory exceeds the steady-state cache budgets, on the machine that just
  OOM'd.
- D3 can learn a single-conversation LCP instead of the shared static prefix.
- Hybrid reuse is approximate (`cross_turn_hybrid_reuse.rs:1`). The release
  gate would have to be a bounded-divergence test, not greedy equality.

### Decision

- D2 and D3 are not worth their risk. Same-conversation reuse already works.
  The cross-conversation gain is about 1.5k tokens per new conversation.
- `prompt_cache_key` stays an ignored hint.
- The real defect is the crash: a Metal OOM during a ~5.3k-token suffix
  prefill on top of a ~19.4k-token restored prefix. That is the **memory
  admission guard** track listed under Non-goals, and it is the next piece of
  work.
