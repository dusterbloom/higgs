# KV-Cache Efficiency — Higgs Side (B/C) Implementation Plan

> **SUPERSEDED (2026-08-10):** Replaced by the cross-repository
> `nanobot-rs/docs/superpowers/plans/2026-08-10-c-first-kv-cache-reliability.md`.
> Do not implement `ContinueFromPrefix`, same-session compaction reuse, or
> hybrid-cache truncation from this document.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the retained-session KV cache reusable across (B) prefix-stable rewrites and (C) LCM auto-expand, and expose the cache decisions so the client can measure efficiency. Client-side counterpart plan: `nanobot-rs/docs/superpowers/plans/2026-08-10-kv-cache-bce-nanobot.md`.

**Architecture:** Extend the existing retained-session prefill strategies (`SessionPrefillStrategy::Continue | BootstrapExact | BootstrapPFlash`) with a prefix-preserving strategy, add session pinning for expansion, and surface the already-computed `session_prompt_trace_metrics` (common_prefix_tokens, divergence_token, outcome) over a response header and `/metrics`.

**Tech Stack:** Rust 2021, axum, tokio, MLX (higgs-engine `simple.rs`), tracing.

## Global Constraints

- `cargo build` + `cargo test` must pass in `crates/higgs-engine` and `crates/higgs`.
- No new semantic variants behind flags; the new strategy is a real branch of `session_prefill_strategy`.
- Memory: retained KV on this 32 GB box must stay bounded (SIGABRT 2026-07-03 with unbounded KV). Model fact for budgeting: Qwen3.6-35B-A3B-Escha-W2 = 40 layers × 2 KV heads × head_dim 256 → ≈ 80 KiB/token fp16 (≈ 20 KiB/token 4-bit); a 19.6k-token session ≈ 1.6 GB fp16.
- Idle-TTL and LRU eviction must never evict a pinned session while a request is mid-flight under it.
- Existing `/v1/cache/sessions/{session_id}` DELETE keeps working (it forces eviction even for pinned sessions).

---

### Task 1: Surface the session prompt trace (visibility)

**Files:**
- Modify: `crates/higgs/src/routes/chat.rs` (attach response header on streaming + blocking paths)
- Modify: `crates/higgs-engine/src/simple.rs` (`session_prompt_trace_metrics` already computes the fields — add a `fmt_trace_header` helper)
- Modify: `crates/higgs/src/metrics.rs` (counters + gauge)
- Test: `crates/higgs-engine/src/simple.rs` tests, `crates/higgs/src/routes/chat.rs` tests

**Interfaces:**
- Consumes: `session_prompt_trace_metrics(...)` return type (verified at simple.rs:680 — fields `prompt_tokens`, `retained_tokens`, `candidate_tokens`, `suffix_tokens`, `common_prefix_tokens`, `divergence_token`, `boundary_splice`, `outcome`).
- Produces:
  - `SimpleEngine::format_session_trace(&SessionPromptTraceMetrics) -> String` → `"outcome=Continued;common_prefix=13428;suffix=20089;divergence=None"`
  - Response header `X-Higgs-Session-Trace` on `/v1/chat/completions` when the request carried `session_id`.
  - Metrics: `higgs_session_prefill_strategy_total{outcome="continued|bootstrap_exact|bootstrap_pflash", reason}` counter; `higgs_retained_sessions_mem_bytes` gauge (sum over `RetainedKv` entries of `state.paired_estimated_bytes()` plus dense estimate).

- [ ] **Step 1: Write the failing test — header format**

```rust
#[test]
fn session_trace_header_round_trips() {
    let metrics = SessionPromptTraceMetrics {
        prompt_tokens: 33517, retained_tokens: 13428, candidate_tokens: 33517,
        suffix_tokens: 20089, common_prefix_tokens: 13428,
        divergence_token: None, boundary_splice: false,
        tool_result_messages: 0, tool_result_bytes: 0, tool_result_largest_bytes: 0,
        outcome: SessionPromptTraceOutcome::Continued,
    };
    let header = format_session_trace(&metrics);
    assert!(header.contains("outcome=Continued"));
    assert!(header.contains("common_prefix=13428"));
    assert!(header.contains("suffix=20089"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-engine session_trace_header_round_trips`
Expected: FAIL — `format_session_trace` missing.

- [ ] **Step 3: Implement `format_session_trace`**

In `simple.rs` beside `session_prompt_trace_metrics`:

```rust
pub fn format_session_trace(m: &SessionPromptTraceMetrics) -> String {
    let outcome = match m.outcome {
        SessionPromptTraceOutcome::Continued => "continued",
        SessionPromptTraceOutcome::ExactBootstrap => "bootstrap_exact",
        SessionPromptTraceOutcome::PFlashBootstrap => "bootstrap_pflash",
    };
    format!(
        "outcome={outcome};common_prefix={};suffix={};divergence={}",
        m.common_prefix_tokens,
        m.suffix_tokens,
        m.divergence_token.map_or_else(|| "none".into(), |t| t.to_string())
    )
}
```

(Adjust to the exact enum variant names in the tree — `SessionPromptTraceOutcome` at simple.rs:5290 area; if variants differ, keep the mapping exhaustive and update the test string.)

- [ ] **Step 4: Attach the header in routes/chat.rs**

On both the blocking and streaming completion paths, after the strategy is resolved, when `request_session_id.is_some()`:

```rust
let trace = engine.format_session_trace(&trace_metrics);
response_headers.insert("x-higgs-session-trace", HeaderValue::from_str(&trace)?);
```

For streaming, insert into the SSE response headers (the `initial` SSE event) before the first data chunk.

- [ ] **Step 5: Add /metrics counters**

In `metrics.rs`, register a counter family keyed by `(outcome, reason)` and increment where each strategy arm resolves in `generate_session_routed_*`; gauge `higgs_retained_sessions_mem_bytes` updated in `cache_snapshot` (simple.rs:3293 area, which already walks retained sessions).

- [ ] **Step 6: Run tests + build + commit**

Run: `cargo test -p higgs-engine session_trace && cargo test -p higgs session_trace && cargo build`
Expected: PASS.
Commit: `git add crates/higgs-engine/src/simple.rs crates/higgs/src/routes/chat.rs crates/higgs/src/metrics.rs && git commit -m "feat(cache): surface session prefill trace via header and metrics"`

---

### Task 2: `ContinueFromPrefix` — keep the matched system prefix on rewrites (B)

**Files:**
- Modify: `crates/higgs-engine/src/simple.rs` (strategy enum ~576, `session_prefill_strategy` ~602, `generate_session_routed_*` match arms ~5324/5425, `handle_session_exact_bootstrap`)
- Test: `crates/higgs-engine/src/simple.rs` tests (extend `session_cache_test_engine` area ~13500)

**Interfaces:**
- Consumes: `retained_session_tokens(session_id) -> Option<Vec<u32>>`, `session_max_suffix_prefill_tokens`, the dense `RetainedState` in `RetainedKv`.
- Produces: `SessionPrefillStrategy::ContinueFromPrefix { session_id, common_prefix_tokens: usize }`; new `SessionBootstrapReason::PrefixTooShort` (when common prefix < `MIN_KEEP_PREFIX_TOKENS = 256`); `SimpleEngine::truncate_retained_to(session_id, common_prefix_tokens) -> Result<(), EngineError>` (keeps KV for `[..common_prefix_tokens]`, drops the rest, so `generate_continued_impl_locked` can continue from it).
- Consumes from nanobot plan Task 2: the client stops rotating the session id when the system prefix is byte-stable, so the same session id's retained KV is offered for a head-replaced prompt.

- [ ] **Step 1: Write the failing strategy test**

```rust
#[test]
fn strategy_keeps_common_prefix_when_head_replaced() {
    let retained = vec![10, 20, 30, 40]; // system, orig1, orig2, tail
    let new_prompt = vec![10, 20, 50, 60]; // system, summary, tail
    let s = session_prefill_strategy(
        7, Some(&retained), &new_prompt, /* max_prefill_tokens */ 8192, false,
    );
    match s {
        SessionPrefillStrategy::ContinueFromPrefix { session_id, common_prefix_tokens } => {
            assert_eq!(session_id, 7);
            assert_eq!(common_prefix_tokens, 2); // system prefix survives
        }
        other => panic!("expected ContinueFromPrefix, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-engine strategy_keeps_common_prefix_when_head_replaced`
Expected: FAIL — today this returns `BootstrapExact(DivergedOrNotGrowing)`.

- [ ] **Step 3: Implement the strategy decision**

In `session_prefill_strategy`, between the full-prefix `Continue` check and the `DivergedOrNotGrowing` return, add:

```rust
let common = common_prefix_token_len(retained_tokens, continuation_candidate);
if common >= MIN_KEEP_PREFIX_TOKENS {
    let suffix_tokens = continuation_candidate.len() - common;
    if suffix_tokens <= max_prefill_tokens {
        return SessionPrefillStrategy::ContinueFromPrefix {
            session_id,
            common_prefix_tokens: common,
        };
    }
    return SessionPrefillStrategy::BootstrapExact {
        session_id,
        reason: SessionBootstrapReason::LargeSuffix {
            suffix_tokens,
            max_prefill_tokens,
        },
    };
}
```

(`common_prefix_token_len` already exists — verified in `session_prompt_trace_metrics`.)

- [ ] **Step 4: Implement `truncate_retained_to`**

Truncate the dense retained KV state to `common_prefix_tokens` (slice the per-layer KV arrays to the token count; if the retained state is paged/TurboQuant and slicing is not possible, return `Err` and the caller falls back to `BootstrapExact`). Reuse the slicing helpers already present in `paged_prefix_cache.rs` (`slice_into_blocks` family) where the retained state is a paged cache.

```rust
pub fn truncate_retained_to(&self, session_id: u64, keep_tokens: usize) -> Result<(), EngineError> {
    let mut retained = self.retained.lock();
    let Some(entry) = retained.get_mut(&session_id) else {
        return Err(EngineError::SessionCacheMiss);
    };
    entry.state.truncate_to(keep_tokens) // RetainedState method; Err if unsupported
}
```

- [ ] **Step 5: Wire the match arm**

In both `generate_session_routed_with_thinking` and the streaming twin, add:

```rust
SessionPrefillStrategy::ContinueFromPrefix { session_id, common_prefix_tokens } => {
    self.record_and_log_session_prompt_trace(/* ... */ SessionPromptTraceOutcome::ContinuedFromPrefix);
    self.truncate_retained_to(session_id, common_prefix_tokens)
        .unwrap_or_else(|_| self.handle_session_exact_bootstrap(session_id, SessionBootstrapReason::DivergedOrNotGrowing));
    self.generate_continued_impl_locked(session_id, &continued_prompt, max_tokens, params, enable_thinking, sender_or_none, timing, total_start)
}
```

Note: `continued_prompt` from `continued_prompt_tokens_from_retained` must be recomputed against the truncated retained (call it after truncation, or pass the candidate through unchanged — the candidate is the full new prompt; after truncation the engine continues with `prompt_tokens[common..]` plus the retained prefix). Keep the trace outcome enum exhaustive.

- [ ] **Step 6: Run the strategy + engine tests**

Run: `cargo test -p higgs-engine session_prefill_strategy && cargo test -p higgs-engine continued && cargo build`
Expected: PASS.

- [ ] **Step 7: Commit**

Commit: `git add crates/higgs-engine/src/simple.rs && git commit -m "feat(cache): ContinueFromPrefix keeps matched system prefix on head-replaced prompts"`

---

### Task 3: Pin retained sessions for expansion (C)

**Files:**
- Modify: `crates/higgs-engine/src/simple.rs` (`RetainedKv`, eviction fns ~1021-1100, `drop_retained_session` ~3553)
- Modify: `crates/higgs/src/types/openai.rs` (`pin_session_id` request field), `crates/higgs/src/routes/chat.rs` (forward the field)
- Modify: `crates/higgs/src/config.rs` (optional `kv_pin_expansion_sessions` default true — client sends the pin explicitly; config only gates the feature)
- Test: `crates/higgs-engine/src/simple.rs` (extend `simple_engine_drop_retained_session_*` tests)

**Interfaces:**
- Consumes: `RetainedKv { state, last_used }` (simple.rs:~1890), `evict_idle_except_from` (simple.rs:1091), LRU insert at simple.rs:1021-1058.
- Produces: `ChatRequest.pin_session_id: Option<u64>` (OpenAI body field, ignored by non-higgs clients); `SimpleEngine::pin_retained_session(session_id: u64, pin: bool) -> bool`; `RetainedKv.pinned: bool`.
- Consumes from nanobot plan Task 5: the client sends `pin_session_id = <old session id>` on the first request after a compaction, and the old id is offered as `session_id` on expansion requests.

- [ ] **Step 1: Write the failing test — pinned session survives LRU pressure**

```rust
#[test]
fn pinned_session_survives_lru_eviction() {
    let engine = session_cache_test_engine(); // max_retained_sessions = 2
    engine.retain_session_for_tests(SESSION_ID, &[1, 2, 3]);
    engine.retain_session_for_tests(OTHER_ID, &[4, 5, 6]);
    engine.pin_retained_session(SESSION_ID, true);
    engine.retain_session_for_tests(THIRD_ID, &[7, 8, 9]); // forces LRU eviction
    assert!(engine.retained_session_tokens(SESSION_ID).is_some(),
        "pinned session must not be LRU-evicted");
    assert!(engine.retained_session_tokens(OTHER_ID).is_none());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-engine pinned_session_survives_lru_eviction`
Expected: FAIL — LRU evicts SESSION_ID.

- [ ] **Step 3: Implement pinning**

```rust
struct RetainedKv {
    state: RetainedState,
    last_used: std::time::Instant,
    pinned: bool,
}

pub fn pin_retained_session(&self, session_id: u64, pin: bool) -> bool {
    let mut retained = self.retained.lock();
    match retained.get_mut(&session_id) {
        Some(entry) => { entry.pinned = pin; true }
        None => false,
    }
}
```

In the LRU insert path (simple.rs:1021-1058) and `evict_idle_except_from` (simple.rs:1091), skip entries with `pinned == true`. Keep `drop_retained_session` (simple.rs:3553) unconditional — explicit drops must still free memory (the client unpins via a later `pin_session_id: 0` or the nanobot drop path).

- [ ] **Step 4: Parse and forward `pin_session_id`**

In `openai.rs` add `pub pin_session_id: Option<u64>` to `ChatRequest` (defaults via serde `#[serde(default)]`), and in `routes/chat.rs` after drop handling (line ~1400):

```rust
if let Some(pin) = req.pin_session_id {
    if engine.pin_retained_session(pin, true) {
        tracing::debug!(session_id = pin, "pinned retained session for expansion");
    }
}
```

- [ ] **Step 5: Run tests + build + commit**

Run: `cargo test -p higgs-engine pinned_session && cargo test -p higgs pin_session && cargo build`
Expected: PASS.
Commit: `git add crates/higgs-engine/src/simple.rs crates/higgs/src/types/openai.rs crates/higgs/src/routes/chat.rs && git commit -m "feat(cache): pin retained sessions for LCM expansion"`

---

### Task 4: Config, doctor, and memory budget for multiple retained sessions

**Files:**
- Modify: `crates/higgs/src/config.rs` (document `kv_max_sessions` for expansion; `kv_max_suffix_prefill_tokens` guidance)
- Modify: `crates/higgs/src/doctor.rs` (add a cache-budget check near the existing `kv_max_session_tokens` warning at doctor.rs:287)
- Modify: `~/.config/higgs/config.toml` (user config — the Escha model entry, currently commented out, or the runtime defaults)
- Test: `crates/higgs/src/config.rs` tests, `crates/higgs/src/doctor.rs` tests

**Interfaces:**
- Consumes: `kv_max_sessions` (default 8, config.rs:691), `kv_max_session_tokens` (default 0 = unbounded), `kv_max_suffix_prefill_tokens` (default 8192, config.rs:699), `kv_cache_bytes` (config.rs ~521).
- Produces: doctor recommendation text; documented config block for the Escha model.

- [ ] **Step 1: Write the failing doctor test**

```rust
#[test]
fn doctor_warns_when_retained_sessions_cannot_fit_expansion() {
    // kv_max_sessions=1 AND kv_cache_bytes < 2 GB with a 35B model → warn
    let cfg = model_config_with_kv(/* kv_max_sessions */ 1, /* kv_cache_bytes */ 1 << 30);
    let findings = cache_budget_findings(&cfg);
    assert!(findings.iter().any(|f| f.severity == Severity::Warn
        && f.message.contains("expansion")));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs doctor_warns_when_retained_sessions_cannot_fit_expansion`
Expected: FAIL — no such check.

- [ ] **Step 3: Implement the doctor check**

In `doctor.rs`, next to the `kv_max_session_tokens` warning, add a check: if the model is 35B-class (any `qwen3_5_moe` or layers × kv_heads × head_dim ≥ 20k dims/token) and (`kv_max_sessions < 2` or `kv_cache_bytes < 1.6 GiB`), warn:

```
model {label} cannot keep a second retained session for LCM expansion: kv_max_sessions={n}
would evict the pre-compaction KV. Set kv_max_sessions >= 2 and kv_cache_bytes >= 2 GiB,
or expansion falls back to a full re-prefill.
```

- [ ] **Step 4: Update the user config**

In `~/.config/higgs/config.toml`, uncomment/complete the Escha model entry with expansion-safe bounds (this is a config change, run through the normal config review):

```toml
# (Escha model entry)
kv_max_sessions = 2
kv_max_session_tokens = 32768
kv_max_suffix_prefill_tokens = 24576
kv_cache_bytes = 2147483648
```

(`kv_max_suffix_prefill_tokens = 24576` so a full 19.6k-token expansion stays in `Continue`/`ContinueFromPrefix` instead of flipping to `BootstrapExact(LargeSuffix)`; the observed 15:02:06 request had a 20,089-token suffix.)

- [ ] **Step 5: Run tests + build + commit**

Run: `cargo test -p higgs doctor_cache_budget && cargo build`
Expected: PASS.
Commit: `git add crates/higgs/src/config.rs crates/higgs/src/doctor.rs && git commit -m "feat(doctor): warn when retained-session budget cannot fit LCM expansion"`

---

## Self-Review

**Spec coverage:**
- B: Task 2 (`ContinueFromPrefix` keeps the matched system prefix when nanobot stops rotating on byte-stable rewrites) + Task 1 (trace shows whether it engaged).
- C: Task 3 (pinned retained sessions so the pre-compaction KV survives until expansion) + Task 4 (config/doctor budget so two sessions coexist within the 32 GB box, with the observed 20k-token suffix staying under `kv_max_suffix_prefill_tokens`).
- E: no server change needed (tool block is a template concern; client freezes it — nanobot plan Task 1).
- Visibility: Task 1 (`X-Higgs-Session-Trace` + metrics counters).

**Soundness note:** `ContinueFromPrefix` only keeps KV whose conditioning prefix is byte-identical (the system/developer region); everything after the divergence point is re-prefilled. Expansion pinning keeps the pre-compaction session whose token vector is a true prefix of the expansion prompt — the only fresh tokens are the new user question. No KV for re-conditioned regions is ever reused.

**Placeholder scan:** every signature referenced exists in the current tree (`session_prefill_strategy`, `RetainedKv`, `session_prompt_trace_metrics`, `common_prefix_token_len`, `retained_session_tokens`, `drop_retained_session`, `evict_idle_except_from`, `kv_max_*` config fields). The one deliberate adapter point is the `SessionPromptTraceOutcome` variant spelling, flagged inline.

**Type consistency:** `pin_session_id` (Task 3) is consumed by the nanobot plan Task 5's compaction step; `format_session_trace` (Task 1) is read by the nanobot Task 6 metrics step; `ContinueFromPrefix` (Task 2) is the mechanism that makes the nanobot Task 2 prefix-keep actually save tokens rather than just bookkeeping.
