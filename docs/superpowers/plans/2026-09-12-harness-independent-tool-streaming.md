# Harness-independent tool streaming implementation plan

> Approved implementation executed on `codex/tool-streaming`. The checklist below preserves the original plan; current results and remaining limits are recorded in [the validation report](2026-09-12-tool-streaming-validation.md).

**Goal:** Remove server-induced silence during tool generation for clients of Higgs's supported streaming APIs, preserve argument correctness, and release work reliably on disconnect.

**Architecture:** A shared incremental model-tool parser produces ordered text, tool-start, argument-fragment, tool-end, and failure events. Existing API routes serialize these events into their native protocols. Transport keepalives remain transport keepalives; correctness and progress must not depend on a harness recognizing comments.

**Tech stack:** Rust, existing Tokio/Axum SSE infrastructure, serde_json, existing model generation workers; pinned SDK clients for interoperability tests.

## Evidence and scope

The September 12 harness history explicitly records a 300,000 ms model-event idle timeout and retries after the writing-plans announcement. Completed calls had approximately 95% aggregate cache reuse. A 1,120-token tool call took approximately 141 seconds and was exposed only at completion. The chat route buffers calls until their closing delimiter and emits SSE comments while holding them. The harness watchdog wraps parsed model events.

This establishes an interoperability defect, but does not prove that all five minutes of each failed attempt were productive decoding. First capture generation progress and parser state to distinguish buffering from an actual inference stall. Do not call the overall performance problem solved merely because a timeout disappears.

Target OpenAI Chat Completions and the existing Anthropic Messages route. Do not add a Responses endpoint as part of this work. No changes to harness settings, prompts, model quantization, or automatic history compaction. A client that imposes a total response deadline, ignores tool deltas, or times out before prefill completes cannot be made universally compatible by streaming arguments.

## Alternatives

1. Increase harness timeouts: temporary mitigation; long calls remain invisible and retries remain expensive.
2. Send empty content or fabricated progress: clients can filter it; can contaminate output; does not establish real model progress.
3. Stream real tool arguments through standard events: recommended. Requires careful incremental parsing and terminal semantics, but fixes the server-side cause across compliant clients.

## Task 1 — Capture and reproduce the failure

**Files:** existing tests in `crates/higgs/src/routes/chat.rs`; new `crates/higgs/tests/tool_streaming.rs`; `crates/higgs/src/metrics.rs` and `crates/higgs/src/metrics_log.rs` if additional persisted fields are needed.

- [ ] Restore GitNexus tooling before symbol edits. The prior CLI context call failed with database version 42 versus runtime version 40. Use the analyzer/runtime matching the index, or reindex with the supported installation. Preserve embeddings if present. Run upstream impact for each modified symbol, report callers/processes/risk, and warn before HIGH/CRITICAL edits. Do not treat missing graph results as zero impact.
- [ ] Preserve current unrelated working-tree changes; use an isolated checkout for implementation.
- [ ] Create a deterministic scripted generation fixture: introductory text followed by a long Qwen XML file-write call, with regular model chunks and no closing tag until after a shortened client idle deadline. Use fake/paused time for unit tests and a generous scaled wall-clock deadline for SDK integration tests.
- [ ] Test two observers: raw SSE bytes and parsed tool events. Confirm that current transport comments arrive while parsed tool progress stops and the client times out.
- [ ] Replay the affected request locally with opt-in diagnostics; retain only request ID, timing, token counts, parser state and byte counts in normal logs. Raw tool text is local debug evidence only.
- [ ] Measure queue wait, prefill, cached/new prompt tokens, first generated token, first semantic event, generated tokens while holding, maximum semantic-event gap, and terminal reason. If generation itself stalls, retain a separate engine investigation with those measurements.

## Task 2 — Incremental tool parsing

**Files:** `crates/higgs-engine/src/tool_parser.rs`; new `crates/higgs-engine/src/tool_parser/streaming.rs` for the state machine and its tests. Keep existing batch parsing as an independent parity oracle for valid fixtures.

**Internal event contract:** ordered `Text(String)`, `ToolStart { index: usize, name: String }`, `ArgumentsDelta { index: usize, fragment: String }`, `ToolEnd { index: usize }`. Parsing failure is a typed error returned with any already-produced events. Route code allocates one stable wire ID per start. Do not group text and calls into separate vectors that lose their original order.

- [ ] Write failing prefix tests first: tool start appears when the name is known; argument bytes appear before the closing tag; concatenation yields exactly the expected JSON value after completion.
- [ ] Support all existing formats: legacy JSON envelopes, Qwen function/parameter XML (including Escha), MiniCPM functions and CDATA. Identify string quoting and nesting incrementally; do not locate structural delimiters inside JSON strings by substring search.
- [ ] For schema-declared XML strings, JSON-escape characters as they arrive. Retain only the unresolved delimiter/CDATA suffix and whatever bounded whitespace is required to preserve current normalization. Cover quotes, backslashes, newlines, Unicode and delimiter-like text.
- [ ] Explicitly address existing type coercion. Current numeric/object/array parsing can fall back to a string only after seeing the whole value; emitted JSON cannot be retracted. Use an incremental strict parser for declared structured/scalar types, with malformed typed values ending the stream in an error. Align batch and streaming behavior for this documented malformed-input policy. Do not silently preserve whole-value buffering for large arrays or objects. For absent/ambiguous schemas, preserve bounded buffering and disclose this progress limitation; never promise incremental conversion when the final type is unknowable.
- [ ] Reject duplicate argument keys or ambiguous envelopes before claiming successful completion. Test argument-before-name envelopes; buffer until identity is known within the existing size limit.
- [ ] Preserve the existing 1 MiB call limit and add bounded nesting/token bookkeeping. Track total call bytes even when emitted bytes are discarded. Avoid rescanning or copying the full accumulated call per token; verify approximately linear scaling with increasing payload sizes.
- [ ] Partition each fixture at every valid string boundary and randomized chunk boundaries. Compare complete valid results against batch parsing. Test network byte splits separately in the HTTP client layer.

## Task 3 — Native protocol adapters and failure semantics

**Files:** `crates/higgs/src/routes/chat.rs`, `crates/higgs/src/routes/anthropic.rs`, `crates/higgs/src/sse.rs`; existing protocol types and translator only where their actual callers require changes.

- [ ] Map starts to OpenAI `delta.tool_calls` identity/name, then send append-only `function.arguments` fragments with the same index. Emit identity once and do not resend the complete argument object at tool end. Reference: https://developers.openai.com/api/docs/guides/function-calling#streaming .
- [ ] Use the same internal parser for Anthropic tool block start, `input_json_delta`, and block stop. Validate against the official Messages streaming contract during implementation. Audit the existing proxy translator for assumptions that each delta contains a complete call.
- [ ] Cover both stateless and retained-session generation paths, plus `tool_choice` auto, none, required and named. The current `required_hold` must not reintroduce full-call buffering. Preserve constraint enforcement; tentative streamed arguments are not proof of validity.
- [ ] Before any tool-start is exposed, malformed ordinary text may retain existing text fallback. After exposure, never convert the partial call back to assistant prose or repeat its prefix. Malformed JSON/XML, overflow or missing closer must produce an explicit failed/incomplete terminal according to the endpoint contract, with no successful tool-call finish.
- [ ] Exercise EOS, output-token limit, capacity interruption and worker failure mid-call. Preserve capacity terminal fields and ordering. Emit usage/termination according to existing endpoint contracts; no content follows a terminal failure. A truncated JSON object is never a successful tool call.
- [ ] Write route tests proving exact event order, stable IDs, no duplicated bytes, multiple calls, mixed reasoning/text/tools, and valid final argument reconstruction. Keep nonstreaming results unchanged for valid inputs.

## Task 4 — Cancellation and accountable timings

**Files:** `crates/higgs-engine/src/simple.rs`, `crates/higgs/src/routes/chat.rs`, `crates/higgs/src/routes/anthropic.rs`, `crates/higgs/src/metrics.rs`; capacity lifecycle code only if tests demonstrate a gap.

- [ ] Disconnect during queued work, prefill, plain decode and tool-argument decode. Verify the worker observes cancellation at its next cooperative boundary, drops reservations and leaves no detached retry work. An in-flight GPU command may complete; do not promise instantaneous hardware cancellation.
- [ ] Test a new request immediately after cancellation: no orphaned predecessor continues consuming generation capacity. Retained caches must either remain valid at a known committed boundary or be invalidated.
- [ ] Persist one terminal record for success, disconnect, capacity interruption and generation failure, including elapsed time and partial output count. Finalization must be idempotent across stream drop and worker completion.
- [ ] Keep diagnostics off the semantic output channel. Track generated-versus-emitted progress separately so future silent generation can be distinguished from a stalled worker.

## Task 5 — Compatibility and release gates

**Files:** new `tests/compat/tool_streaming/` with pinned client dependencies, synthetic fixtures and a README; update existing API documentation with streaming and malformed-input behavior.

- [ ] Validate using a raw SSE consumer that discards comments, official OpenAI Python and Node SDKs, the installed pi-ai adapter used by deepseek-harness, and an Anthropic SDK on the Messages route. Verify SDKs receive argument deltas rather than merely keeping the socket open.
- [ ] Run a long-call fixture whose total duration exceeds the client idle deadline while genuine argument fragments arrive more frequently than that deadline. Require successful completion, no retry, one completed call, and exact argument reconstruction.
- [ ] Run the inverse fixture: the worker produces no progress. It must still fail/cancel normally; heartbeats must not fabricate semantic activity.
- [ ] Use inert tool declarations for live Escha tests so generated file writes are inspected, not executed. Test cold and warm prompts near the incident's 20K-token size, and a multi-turn sequence. Record time-to-first-token, max argument-event gap, decode throughput and cancellation latency separately.
- [ ] Run `cargo test -p higgs-engine tool_parser`, relevant route tests, then `cargo test -p higgs-engine -p higgs` under the repository's supported MLX build environment. Run `cargo fmt --all -- --check` and required CI checks. Record any hardware-dependent checks that cannot run; do not count them as passing.
- [ ] Before each implementation commit, run GitNexus detect_changes and inspect expected symbol/process scope. Refresh the index after commits, preserving embeddings. Review parser correctness and protocol traces before release.

## Definition of done

Large valid calls stream real arguments on both supported APIs and all tested clients reconstruct them correctly without timeout changes. Malformed and interrupted calls never claim success. Cancellation releases generation ownership. Terminal metrics include interrupted attempts. No material throughput regression in the controlled before/after benchmark. Remaining prefill or decode slowness is reported separately, with measured evidence.

The initial planning step changed only documentation. Implementation and isolated live validation were subsequently authorized; consult the validation report for current execution status.
