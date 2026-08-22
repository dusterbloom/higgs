# Escha Performance Scorecard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject speculative benchmark results that do not reproduce the AR baseline's greedy output for a local Escha Qwen3.8 model.

**Architecture:** `bench_speculative` keeps the small, existing fresh-server loop. It derives a deterministic digest from each visible completion and compares each non-baseline run to the baseline with the same repeat index before the result is persisted. Server-side MTP telemetry remains opaque log text for this milestone.

**Tech Stack:** Rust, Clap, Tokio, serde, existing `higgs-bench` helpers.

## Global Constraints

- No new dependencies or environment flags.
- Do not start a Higgs server or use the GPU during unit/build verification.
- Preserve the user's unrelated `AGENTS.md`, `CLAUDE.md`, `.omen/`, and Escha-document edits.
- A live benchmark is valid only with thinking disabled, temperature zero, a fixed prompt, and a matching thermal state.

---

### Task 1: Add parity-gated trial accounting

**Files:**
- Modify: `crates/higgs-bench/src/bin/bench_speculative.rs`
- Test: `crates/higgs-bench/src/bin/bench_speculative.rs`

**Interfaces:**
- Consumes: `CompletionPayload { completion_tokens, content }` returned by `request_completion`.
- Produces: `TrialRun { output_digest, parity_with_baseline }` and a fallible comparison before `BenchOutput` is persisted.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn rejects_a_visible_completion_that_differs_from_its_baseline() {
    let baseline = OutputSignature::new(4, "same");
    let candidate = OutputSignature::new(4, "different");
    assert!(compare_to_baseline(&baseline, &candidate).is_err());
}

#[test]
fn rejects_a_completion_token_count_that_differs_from_its_baseline() {
    let baseline = OutputSignature::new(4, "same");
    let candidate = OutputSignature::new(5, "same");
    assert!(compare_to_baseline(&baseline, &candidate).is_err());
}
```

- [ ] **Step 2: Verify the tests fail for the missing parity contract**

Run: `cargo test -p higgs-bench --bin bench_speculative rejects_a_visible_completion_that_differs_from_its_baseline`

Expected: compilation failure because `OutputSignature` and `compare_to_baseline` do not exist.

- [ ] **Step 3: Implement the smallest deterministic comparison**

```rust
struct OutputSignature { completion_tokens: u32, digest: u64 }

fn compare_to_baseline(baseline: &OutputSignature, candidate: &OutputSignature) -> Result<()> {
    if baseline == candidate { Ok(()) } else { anyhow::bail!("greedy output differs from baseline") }
}
```

Use a local FNV-1a byte loop for `digest`; do not add a hashing crate. Store each baseline by repeat index, compare subsequent modes, and add the digest/parity fields to `TrialRun`.

- [ ] **Step 4: Verify focused and crate tests**

Run: `cargo test -p higgs-bench --bin bench_speculative`

Expected: PASS.

- [ ] **Step 5: Format, inspect impact, and commit**

Run: `cargo fmt --all`, `git diff --check`, and GitNexus `detect_changes` before staging only the two new docs and `bench_speculative.rs`.

Commit: `feat(bench): gate speculative runs on greedy parity`

### Task 2: Prove the runnable scorecard without GPU contention

**Files:**
- Modify: none

**Interfaces:**
- Consumes: the finished `bench_speculative` CLI.
- Produces: a documented exact live command, deferred until the active Higgs process is stopped.

- [ ] **Step 1: Compile the release benchmark**

Run: `cargo build -p higgs-bench --release --bin bench_speculative`

Expected: PASS; do not run the binary because it starts a server.

- [ ] **Step 2: Record the live protocol**

Use, after confirming no active Higgs process:

```bash
./target/release/bench_speculative \
  --model-path "$HOME/.cache/lm-studio/models/EschaLabs/Qwen3.8-27B-Escha-W2" \
  --model-name escha-27b \
  --trials baseline,mtp_default,mtp_adaptive \
  --max-tokens 320 --repeats 3
```

Follow with `bench_frontier --frontiers 2048,4096,8192,16384 --probe-tokens 64 --runs 3` and capture a matched 10-second Metal System Trace of the live server.
