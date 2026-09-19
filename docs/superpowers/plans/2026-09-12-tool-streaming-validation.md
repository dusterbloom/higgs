# Tool streaming validation

Implementation branch: `codex/tool-streaming`, based on `d8eb59032`.

## Result

The final candidate streams real tool arguments on both native APIs, passes the
SDK/harness interoperability checks, and cancels retained prefill at the next
cooperative chunk boundary. The final controlled warm runs show no material
throughput regression. No installed executable or harness configuration was
replaced; all temporary servers were stopped after validation.

## Incident evidence and confirmed defects

The supplied warnings count replayed tool-output bytes, not request duration.
The saved deepseek-harness session records a **300,000 ms parsed-model-event idle
timeout** and retries. The old route held model tool calls until their closing
delimiter; its SSE comments did not reset that watchdog. A completed 1,120-token
tool call took about 141 seconds. Completed requests had approximately 95% cache
reuse. This does not prove the failed attempts were continuously decoding.

Live validation also reproduced a distinct cancellation defect: an abandoned
retained-session prefill kept the generation worker for roughly 177 seconds,
even though HTTP metrics had already recorded 499. The immediate follow-up's
first event arrived 180.88 seconds after disconnect. This demonstrates a worker
cleanup defect; it is not proof that this exact phase caused every original retry.

## Implemented behavior

- Shared incremental Qwen JSON/XML and MiniCPM parsing for OpenAI Chat
  Completions and native Anthropic Messages streaming.
- Stable tool identities and append-only JSON argument fragments; exact final
  reconstruction across real clients.
- Bounded 1 MiB calls and 256-level nesting, explicit malformed/incomplete errors,
  duplicate-key rejection, and no successful terminal after interruption.
- Required/named completion validation while streaming tentative arguments.
- Native Anthropic error terminals, including capacity interruption.
- Disconnect checks after the serialized generation gate and at retained
  prefill boundaries, including AR/MTP, prefill-only and DFlash paths.
- DFlash cancellation is classified before cold-retry eligibility, preserving
  typed pressure/drain/watchdog reasons through paired-cache error wrapping.
- Idempotent terminal metrics with partial output counts, and optional aggregate
  generated/emitted, parser, serialization, detokenization and channel timings.

Native Anthropic nonstream `any`/named `tool` choices explicitly require
`stream:true`; this does not add a nonstream tool-use representation. Ambiguous
XML types retain bounded buffering. Clients must await successful completion
before executing tools. No fake progress or timeout changes are used.

## Automated verification

| Check | Result |
|---|---|
| Full release/all-features workspace suite, before final cancellation follow-up | 2,334 passed, zero failed, 83 ignored |
| Final engine/server library suites after cancellation and diagnostics | 693 + 705 passed, zero failed, 5 + 1 ignored |
| Parser coverage | 69 passed, including legacy parity, strict JSON and delimiter boundaries |
| New retained/DFlash cancellation regressions | Dropped receiver, wrapped reason mapping and unrelated-error preservation passed |
| Raw SSE, comments ignored | 82 argument events over about 2.37s; exact reconstruction |
| OpenAI Node 6.40.0 / Python 2.26.0 | Streaming and interrupted-call checks passed |
| Anthropic Node 0.123.0 / Python 0.84.0 | Streaming and interrupted-call checks passed |
| pi-ai 0.85.1 + harness adapter 0.1.5-rc.2 | Real one-second idle watchdog accepted long fixture; no retries |
| Inverse pi-ai idle fixture | Watchdog failed as expected; disconnect freed gate for follow-up |
| Validation scrubber / runner self-test | 3 tests and offline runner passed |
| Independent Sol review | Parser, protocol, cancellation and diagnostics blockers resolved |

Latest Node and Python suites ran sequentially because the fixture deliberately
shares the real serialized worker gate. A prior concurrent run confounded the
follow-up latency test with normal queueing. A preliminary sandboxed Rust rerun
aborted when MLX could not enumerate a Metal device; the final run with local
device access passed.

Repository-wide formatting and Clippy still fail in unchanged base code. New
Rust changes are formatted, and changed-line Clippy checks report no new
diagnostics. Existing formatting differences remain in `simple.rs`, model
`cache.rs` and model `lib.rs`. Ignored hardware tests are not counted as passing.

## Final controlled live measurements

Same Escha model, generation/KV settings, tool declarations and growing prompt
sequence; speculation disabled, disk prefix caching disabled, one fresh process
per build, no concurrent model or compilation. The baseline is the exact base
commit built with the same compiler and Metal bundle. Tools are inert.

| Case | Base first args / total | Final first args / total | Final argument events |
|---|---|---|---|
| Cold | 168.60s / 168.60s | 145.55s / 159.80s | 146 |
| Warm continuation | 15.96s / 15.96s | 2.05s / 15.72s | 142 |
| Multi-turn | 16.76s / 16.76s | 2.05s / 16.63s | 149 |

The base emitted one argument event per call. Actual initial input was 18,835
tokens; output counts match at 165/162/169. Warm cached counts match at
18,999/19,218. Final inter-argument gaps were at most 0.48/0.55/0.53 seconds.
The scripted one-second watchdog test does not imply a one-second live
first-token deadline: initial prompt processing remains substantial.

Engine phase diagnostics separate this from delivery: baseline warm decode was
14.65/15.43s; final warm decode was 14.41/15.31s, with roughly 1.1s prefill.
Warm parser time was about 0.5ms; detokenization and channel send together about
2.7ms. SSE serialization and yield suspension were below 1ms combined.
The cold difference is not claimed as an inference optimization.

The final full probe cancelled during arguments and initial waiting. Follow-up
first events arrived at **3.81s** and **10.04s** after disconnect, respectively,
and completed successfully. Both satisfy the probe's 30-second limit. The
initial-wait case repeats the original 18,842-token cancellation sequence.
A separate focused run showed worker release at 6.59s; follow-up latency also
includes that request's own prefill/decode. Cancellation remains cooperative,
not instantaneous cancellation of an in-flight GPU command.

An earlier candidate artifact showed 31–36s warm totals. This did not reproduce
in the final matching-build run; its cause was not established. The final
measurements and aggregate diagnostics support no material regression for this
sequence, not a universal performance guarantee. Further same-binary buffered/
streamed comparison remains an available diagnostic if that anomaly recurs.

## Artifact provenance

SHA-256:

- Exact-base build: `c61666f54a52b52c4bca9a221bd2553b0f89757f4daeb9068eeb538ec940ba5a`
- Final candidate: `bdb6cd1de44ec81623716eaa31895bd00cf7f0462b3b6da82f3d45ef1f58de43`
- Earlier candidate with anomalous warm timings: `b479e4d6e1c84cee5e40fb9f5625002b826ef8510176e2794140f57f36d7a1c6`
- Initially installed baseline: `b9efb80fbd177fb03fe6f26c60bc93b43bfb6d246c71df5614bad2ee19b77c60` (precise source commit not established)

The initial installed artifact used its own Metal bundle. The final controlled
pair uses the same bundle and build command. Reproduce with the inert
[compatibility and live probes](../../../tests/compat/tool_streaming/README.md).
