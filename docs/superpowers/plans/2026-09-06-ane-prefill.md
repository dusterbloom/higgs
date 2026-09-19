# ANE-assisted long-context prefill implementation plan

> **For agentic workers:** Use the subagent-driven-development or executing-plans skill when execution is authorized. This is a proposed plan, not authorization to implement it now.

**Goal:** Produce an upstream-ready, evidence-backed M4 ANE prefill improvement for native Escha W2 without damaging long-context recovery or capacity.

**Architecture:** One existing Higgs prefill path, with an optional concrete token-local z projection resource. The GPU keeps recurrent qkv, native W2 experts and decode. Reuse the public CoreML branch selectively and retain one benchmark/scoring contract.

**Tech stack:** Rust, mlx-rs/Metal, a minimal Objective-C public CoreML bridge, existing Python measurement tools, release builds, tmux.

**Design:** `docs/superpowers/specs/2026-09-06-ane-prefill-design.md`.

## Global constraints

- Plan-only until design review. No code, checkout, installation or runtime change is part of this planning turn.
- Start implementation from a clean isolated checkout containing the reviewed long-context fixes; current nightly has uncommitted work. Do not copy or commit unrelated dirty changes. Record the exact prerequisite commit set before creating the implementation branch.
- `feat/magic-canvas:89141aa6e` supplies reusable CoreML concepts; never merge the whole branch. Preserve all existing worktrees.
- Apply GitNexus upstream impact before each edited symbol and graph change detection before each commit. Current index is 37 commits stale; the planning query failed partially and did not resolve GatedDeltaNet. Refresh during implementation; UNKNOWN requires textual caller corroboration, not an all-clear.
- All hardware runs and builds serialized in tmux. Release builds/tests only. No local GPU inference while compiling or running another model.
- Use the design's numerical, memory, cancellation and acceptance gates verbatim. A failed gate stops progression; don't widen precision/error limits or change the corpus to pass.
- Paths below are relative to the Higgs checkout unless explicitly marked nanobot. The implementation owner must preserve others' edits and is not alone in the codebase.

## Task 1 — Reproducible baseline and operator attribution

**Own:** existing `benchmarks/` harness where applicable; proposed `benchmarks/ane_prefill/measure.py`, `benchmarks/ane_prefill/test_measure.py`, `benchmarks/ane_prefill/README.md`; diagnostic instrumentation only in `crates/higgs-models/src/qwen3_next.rs` and `crates/higgs-engine/src/simple.rs` when existing counters are insufficient.

**Consumes:** current model/tokenizer and successful 45K synthetic replay. **Produces:** a JSONL record per run with `arm`, `checkpoint_sha256`, `binary_sha256`, `prompt_sha256`, `prompt_tokens`, `cached_tokens`, `prefill_ms`, `request_ms`, `decode_tokens`, `decode_ms`, `ane_dispatches`, `peak_footprint_bytes`, `swapouts_delta`, `outcome`, `semantic_pass`; phase records for projection, recurrence, attention and native experts. Missing phase timing is null, never inferred from total time minus an assumed decode speed.

- [ ] Pin baseline build, checkpoint, corpus, server settings, OS/power and current escha-mlx revision. Reuse existing tokenization, HTTP auth, memory sampling and SSE parsing rather than maintaining a second version.
- [ ] Add a failing measurement regression using a synthetic stream: total prompt 45,003, processed 44,992, elapsed 467,320 ms, cached zero. Expected processed rate is `44992 / 467.320`, not `45003 / request_seconds`. A terminal server error makes `outcome` failed even if the harness process exits zero. A subchunk stream without final timing produces null prefill rate.
- [ ] Implement the shared record/scorer and run its standard-library unit tests: `python3 -m unittest discover -s benchmarks/ane_prefill -p 'test_*.py'`.
- [ ] Capture current full-model phase costs at early and late positions. Diagnostic synchronizations are allowed only in attribution runs; separately measure their perturbation against uninstrumented whole-request timing. A sum of forced-sync timings is not the release performance baseline.
- [ ] Run the uninstrumented prompt ladder once, then bounded paired controls at 32K/45K. Record whether z has enough removable cost: a 20% z-latency cut yields only `0.20 * f` whole-prefill latency reduction. If even the ideal z bound cannot plausibly meet the 15% release target, stop z integration and document the dominant measured operator instead.
- [ ] Review and commit only harness/diagnostic changes after graph detection. No new engine path yet.

The record/scorer tests must include these exact assertions (using the proposed measurement module):

```python
import math
from measure import processed_rate, run_passed

assert math.isclose(processed_rate(44992, 467320), 96.27664127364547)
assert processed_rate(496, None) is None
assert not run_passed({"outcome": "server_error", "semantic_pass": True})
assert not run_passed({"outcome": "finished", "semantic_pass": False})
assert run_passed({"outcome": "finished", "semantic_pass": True})
```

Proposed helper signatures: `processed_rate(processed_tokens: int, prefill_ms: int | None) -> float | None`; `run_passed(record: dict) -> bool`, true only for a finished run with semantic_pass exactly true. Validate negative counts/times as malformed records; a zero duration has no defined rate. Keep these small helpers in the shared scorer, not a separate service.

## Task 2 — Extract one safe CoreML projection resource

**Own:** proposed `crates/higgs-models/src/ane_projection.rs`, `crates/higgs-models/bridge/ane_projection.{h,m}`, Cargo/build integration, and proposed `benchmarks/ane_prefill/export_projection.py`. Use existing build/bridge locations when they already satisfy these responsibilities; do not retain lm-head/drafter-specific prototype abstractions.

**Consumes:** selected current z rows and Task 1 record format. **Produces:** `AneProjection`, `AneTicket`, `AneError`, `AneMemory` as specified in the design; a versioned compiled-artifact manifest. `AneError` distinguishes unsupported configuration, preparation, execution, Busy and cancelled work. No session IDs or KV state enter this API.

- [ ] Extract the branch's public CoreML code with its source license and attribution. Remove unrelated LUT6 lm-head, DFlash, private-backend and model-specific assumptions from the reused slice. Inspect Objective-C ownership and error allocation before wrapping it.
- [ ] First write tests for wrong shape/dtype, corrupted artifact, checkpoint/OS/tile mismatch, second submission while a ticket owns buffers, ticket-drop ordering, and external memory returning to baseline after shutdown. A fake native completion barrier must prove buffers outlive work; hardware tests additionally confirm actual ANE placement.
- [ ] Implement explicit create/load/submit/finish/drop ownership. Native failure returns a typed error and cannot be interpreted as a valid output array. Build on supported macOS with usable CoreML; preserve the existing GPU-only build elsewhere.
- [ ] Export selected rows only, using actual QLinear group size/bits/scales/biases and row ordering. Match the exact loaded checkpoint. Install artifacts atomically after validation; never dequantize all W2 experts or silently convert source weights in place.
- [ ] Run release unit tests and one real projection at T1024. Report cold compile/load, warm compute, packing, copy, wait, merge, residency and peak separately, plus copy-inclusive latency and the fixed numerical screening gates.
- [ ] Reject the route unless placement is confirmed and its measured projected end-to-end saving survives accounting. If public CoreML cannot expose useful overlap, record that result; a private API backend requires a new reviewed design, not another hidden fallback.
- [ ] Review/commit this isolated adapter after graph detection. Do not install it on user requests.

## Task 3 — Integrate the existing GDN seam and resource accounting

**Own:** `crates/higgs-models/src/qwen3_next.rs` (`GatedDeltaNet`, loader fusion/layout seam), model-loader hooks already used by nightly, `crates/higgs/src/state.rs` and existing capacity accounting. Touch engine code only where lifecycle/admission genuinely requires it; do not introduce a second forward loop.

**Consumes:** Task 2 adapter and Task 1 measurement contract. **Produces:** optional `AneZPlan` on the model and a measured hybrid path behind the existing benchmark diagnostic selection.

- [ ] First add labeled-row tests for both separate and combined/per-head projection layouts. Unknown layouts decline eligibility. Test exact original recurrent weight bytes, GPU reference tolerances on identical inputs, native W2 experts untouched, and comparison to the existing fused GPU baseline.
- [ ] Build the actual split: GPU excludes only the ANE z rows, retains all recurrent rows, and runs concurrently. Restore original output ordering once before existing continuation. A CPU wait before independent GPU submission must fail the schedule test.
- [ ] Keep one T1024 tile, one ticket, one worker. Nonmatching tails and decode stay on existing GPU execution. Do not pad real tokens or positions; no static ratio copied from another Qwen model.
- [ ] Account incremental CoreML resident/temporary memory exactly once before installing the plan. Add tests showing additional memory affects admission, insufficient headroom selects GPU before work, cancellation doesn't free live native buffers, and model unload retires resources before removing their charge.
- [ ] Failure tests: preparation error selects GPU; execution error retires ticket then permits at most one uncommitted projection recomputation; pressure/cancellation never triggers recomputation. Cache/recurrent state is unchanged after failed projection. Existing typed watchdog progress and retry regressions must remain green.
- [ ] Extend cache identity with effective projection-plan fingerprint where approximate state can differ. Test full-prefix hit, partial restore, append, plan disable/reload, and cancellation followed by a new request. Existing compatible GPU-only cache identity remains stable.
- [ ] Run release model/engine/server suites: `cargo test --release -p higgs-models --lib`, `cargo test --release -p higgs-engine --lib`, `cargo test --release -p higgs --lib`; build `cargo build --release -p higgs --bin higgs`. Use the baseline's documented environment and platform skips; do not count skipped model-dependent tests as hardware validation.
- [ ] Benchmark one full 32K/45K pair before broader evaluation. Stop on any quality, memory, cancellation or latency regression. Review and commit integration only with evidence and graph detection.

## Task 4 — Long-context quality, endurance and matched performance

**Own:** Task 1 benchmark/scorer, proposed `benchmarks/ane_prefill/cases.json`, and existing nanobot `src/agent/agent_loop/recovery_eval.rs` / `endurance_eval.rs` only if their current harness cannot express the needed isolated cases. Keep results in `benchmarks/ane_prefill/results/<run-id>/` with immutable source and output manifests.

**Consumes:** exact baseline and candidate binaries. **Produces:** 24-case fixed quality matrix, paired performance report, admission/recovery report and a go/no-go decision.

- [ ] Freeze expected outcomes before candidate execution. Six task families x two lengths x cold/partial-restore = 24 cases. Run both arms twice with speculation disabled and matched generation settings. Known GPU failures are reported, not quietly removed. Any new critical behavioral failure rejects the candidate.
- [ ] Test scorer failures before running models: fabricated success, wrong correction precedence, missing evidence, duplicate prohibited retry and a harness exit-zero with `pass:false` must fail the report. Do not require a canonical enum string unless the original task specifies it.
- [ ] Run a small six-case screen before the full matrix. Abort the expensive campaign at the first critical regression. Retain both failed and successful outputs.
- [ ] Collect five GPU/hybrid/GPU timed blocks at 32K/45K and the short-prompt ladder. Reuse quality-run timing only when its request/outputs exactly satisfy the performance protocol. Report drift, medians, ranges, measured ANE dispatches, CPU/GPU synchronization costs and external memory. The release gate is >=15% whole-request latency reduction at both long lengths, each uncontaminated pair >=5%, decode <=3% slower, short prompts <=5% slower, incremental peak <=1GiB, no new swap-outs and no loss of 45K admission.
- [ ] Use the existing isolated library-test nanobot harness to run a tool-heavy long session, real LCM summary/recovery and capacity wait/resume with completed-tool receipts. No user DB writes, no duplicate tools, no competing singleton CLI, no lossy emergency trimming counted as success.
- [ ] Run matched current escha-mlx on this same machine sequentially, restoring Higgs afterward. Report checkpoint/layout differences explicitly. Publish ranking claims only for the workloads actually measured; a win against our baseline is not proof of being fastest.

### Validation cost

The full 24-case matrix, two arms and two repetitions is 96 long requests. The 32K/45K timed blocks add 30 requests unless compatible existing measurements can be reused. With the observed ~8-minute cold 45K request, this is an overnight campaign (roughly 12–18 hours before failures/repeats), not a quick smoke test. Record actual time after the screen; never run these in parallel on the same GPU to shorten wall time. Stop at a failed gate and retain its artifacts rather than finishing an invalid campaign.

## Task 5 — Upstream review package

**Own:** Higgs contribution documentation and scoped commits; no provider/protocol changes in nanobot.

- [ ] Produce a compact evidence report with exact commands, environment, binary/tensor hashes, scores, raw timings, memory and failure cases. Describe the approximate INT8 numerical contract and partial-cache behavior clearly.
- [ ] Propose upstream's public activation policy only after the gates pass. Default enablement needs reproducible hardware/model qualification and maintainer agreement; benchmark diagnostics must not become a permanent matrix of env switches.
- [ ] Structure review as harness, adapter, integration/accounting, then evidence/docs. Independently review native ownership, lazy scheduling, numerical policy and cancellation. Run graph detection for each commit; inspect unrelated dirty work before staging.
- [ ] Prepare draft PR text locally, with the exact trigger and measured before/after behavior. No push, public PR, maintainer message or remote publication occurs until authorized.

## Review of this plan

- [x] Current branch and dirty-worktree provenance distinguished.
- [x] Single forward path, concrete ownership and shared contracts specified.
- [x] Recurrent precision, approximate z quality and cache compatibility covered.
- [x] Native lifetime, memory admission, cancellation and fallback bounded.
- [x] Baseline, statistical limitations, failure scoring and upstream evidence covered.
- [x] No runtime or code changes performed in this planning turn.
