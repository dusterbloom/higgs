# Qwen3.8 Low-Power Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Qwen3.8 27B Higgs throughput on Apple M4 in normal and Low Power Mode while preserving exact greedy token trajectories and production cache semantics.

**Architecture:** Use the existing ignored production `mtp_cycle` benchmark as the measurement boundary. First establish paired depth, dispatch, and power-mode evidence; then make only isolated cross-row or FastMetal configuration changes that pass bit-exact tests and trace-guided timing gates.

**Tech Stack:** Rust, Cargo release tests, MLX/MLX-C FastMetal kernels, Metal System Trace (`xctrace`), local Qwen3.8 27B checkpoint, macOS `pmset`/`powermetrics` telemetry.

## Global Constraints

- Target model: `/Users/peppi/AI-Models/qwen38-higgs`.
- Target branch: `codex/qwen38-higgs-port`, isolated from the original checkout.
- Preserve draft/verifier token IDs, measured and whole-trajectory digests, cache offsets, and accepted-token counts.
- Preserve stock fallback for unsupported shapes, quantization, row counts, and disabled feature settings.
- Keep each benchmark instrumentation or production optimization in a separate revertible commit.
- Do not leave Low Power Mode or other system power settings changed after testing.
- Retain a candidate only after focused exactness tests, release checks, and paired benchmark evidence.

---

### Task 1: Stabilize the benchmark and record the depth baseline

**Files:**
- Modify: `docs/superpowers/sdd/2026-08-17-qwen38-higgs-performance-port/task-6-report.md` only for raw-result summaries.
- Read: `crates/higgs-engine/src/mtp.rs:1230-1385`.
- Artifacts: `/private/tmp/higgs-qwen38-sweep/` (not committed).

**Interfaces:**
- Consumes: existing `bench_production_mtp_cycle_real_model`, `HIGGS_MTP_DRAFT_N_MAX`, `HIGGS_CROSSROW_QMV`, `HIGGS_MTP_ADAPTIVE_DRAFT`.
- Produces: depth-by-depth medians with measured/whole digest parity and power-state annotations.

- [ ] **Step 1: Confirm the physical power baseline.**

Run `pmset -g ps`, `pmset -g batt`, and `pmset -g custom`. Continue only when the charger is recognized and the battery is not discharging. Record `lowpowermode` and reject any trial with visible thermal pressure or battery drain.

- [ ] **Step 2: Create the raw-output directory.**

```bash
mkdir -p /private/tmp/higgs-qwen38-sweep
```

- [ ] **Step 3: Run three fresh-process trials for each grouped depth.**

Use `BENCH_PROMPT_LEN=256`, `BENCH_DECODE_STEPS=64`, `HIGGS_MTP_ADAPTIVE_DRAFT=0`, `HIGGS_CROSSROW_QMV=1`, and each `HIGGS_MTP_DRAFT_N_MAX` value from 1 through 8:

```bash
HIGGS_MODEL_PATH=/Users/peppi/AI-Models/qwen38-higgs \
BENCH_PROMPT_LEN=256 BENCH_DECODE_STEPS=64 \
HIGGS_MTP_DRAFT_N_MAX=8 HIGGS_MTP_ADAPTIVE_DRAFT=0 HIGGS_CROSSROW_QMV=1 \
cargo test -p higgs-engine --release --lib \
  mtp::tests::bench_production_mtp_cycle_real_model -- \
  --ignored --exact --nocapture
```

Repeat the command with `HIGGS_MTP_DRAFT_N_MAX=1`, `2`, `3`, `4`, `5`, `6`, and `7`, saving each complete stdout under `/private/tmp/higgs-qwen38-sweep/grouped-depth-N-run-M.log`.

- [ ] **Step 4: Run the stock paired trials.**

Repeat the same depth/run matrix with `HIGGS_CROSSROW_QMV=0`, saving logs as `/private/tmp/higgs-qwen38-sweep/stock-depth-N-run-M.log`.

- [ ] **Step 5: Check exactness before ranking speed.**

For every grouped/stock depth pair, require equal verifier rows, drafted count, accepted count, measured emitted count, measured digest, whole count, and whole digest. Exclude any pair that differs before calculating medians.

- [ ] **Step 6: Record the result and commit the report.**

Add the raw median table, power-state metadata, and the selected depth to `task-6-report.md`, then run `git diff --check` and commit with:

```bash
git add -f .superpowers/sdd/2026-08-17-qwen38-higgs-performance-port/task-6-report.md
git commit -m "docs: record qwen depth sweep"
```

---

### Task 2: Compare verifier dispatch paths and identify the trace target

**Files:**
- Read: `crates/higgs-models/src/qwen3_next.rs:900-1020,2914-3660`.
- Modify: `docs/superpowers/sdd/2026-08-17-qwen38-higgs-performance-port/task-6-report.md` for results.
- Artifacts: `/private/tmp/higgs-qwen38-sweep/dispatch-*.log`.

**Interfaces:**
- Consumes: selected depth from Task 1, `HIGGS_CROSSROW_QMV`, `HIGGS_QGEMM_VERIFY`, production benchmark output.
- Produces: a dispatch decision for `T=2..9` and one representative run per mode for Metal tracing.

- [ ] **Step 1: Run the selected depth with grouped cross-row.**

Use the Task 1 selected depth, `HIGGS_CROSSROW_QMV=1`, and `BENCH_DECODE_STEPS=128`. Save the complete output and record the digest pair.

- [ ] **Step 2: Run the same workload with stock dispatch.**

Use identical settings with `HIGGS_CROSSROW_QMV=0`. Require exact digest parity and compare median tok/s and average cycle time.

- [ ] **Step 3: Exercise the existing QGEMM verifier gate.**

Run the same workload with `HIGGS_QGEMM_VERIFY=1` and `HIGGS_CROSSROW_QMV=0`; the QGEMM path is defined in `qwen3_next.rs` and must not be combined with the grouped cross-row setting for this comparison. Record whether it is accepted, rejected, or falls back, plus its exact digest.

- [ ] **Step 4: Select the trace pair.**

Choose the fastest exact grouped/stock pair at the selected depth. If QGEMM is exact and materially faster, include it as the third trace condition; otherwise document it as a rejected dispatch candidate.

---

### Task 3: Capture warmed Metal System Traces and classify the bottleneck

**Files:**
- Read: `crates/higgs-models/src/crossrow_qmv.rs:248-390` and `crates/higgs-models/src/qwen3_next.rs:3425-3545`.
- Artifacts: `/private/tmp/higgs-metal/qwen38-*.trace`, `.xml`, and `.log` (not committed).

**Interfaces:**
- Consumes: exact trace conditions from Task 2.
- Produces: GPU/CPU/launch classification that gates code changes in Tasks 4–6.

- [ ] **Step 1: Build the benchmark binary once.**

```bash
CARGO_TARGET_DIR=/private/tmp/higgs-qwen38-target \
cargo test -p higgs-engine --release --lib \
  mtp::tests::bench_production_mtp_cycle_real_model --no-run
```

- [ ] **Step 2: Capture grouped and stock traces.**

Use `xcrun xctrace record --template "Metal System Trace"` with the exact Task 2 environment, `--time-limit 10m`, and output names `/private/tmp/higgs-metal/qwen38-grouped.trace` and `/private/tmp/higgs-metal/qwen38-stock.trace`. Launch the already-built test binary with `--ignored --exact --nocapture` and save stdout to matching `.log` files.

- [ ] **Step 3: Export trace table-of-contents data.**

```bash
xcrun xctrace export --input /private/tmp/higgs-metal/qwen38-grouped.trace \
  --toc --output /private/tmp/higgs-metal/qwen38-grouped-toc.xml
xcrun xctrace export --input /private/tmp/higgs-metal/qwen38-stock.trace \
  --toc --output /private/tmp/higgs-metal/qwen38-stock-toc.xml
```

If sandbox permissions prevent Instruments capture, record that limitation and use benchmark wall time plus `powermetrics` annotations; do not fabricate GPU conclusions.

- [ ] **Step 4: Classify the dominant cost.**

Mark the workload as kernel-bound, host/configuration-bound, or draft synchronization-bound by comparing encoder duration, CPU launch gaps, and per-cycle benchmark output. Only the matching optimization task may proceed.

---

### Task 4: Prototype and verify cross-row metadata broadcast

**Files:**
- Modify: `crates/higgs-models/src/crossrow_qmv.rs:132-151`.
- Test: existing `crossrow_qmv::tests::crossrow_bit_exact_vs_stock_rows` and `crossrow_qmv::tests::crossrow_metal_schedule`.

**Interfaces:**
- Consumes: Task 3 classification showing cross-row kernel cost is material.
- Produces: a separate candidate commit that preserves the `fetch_block` nibble/accumulation order.

- [ ] **Step 1: Add the failing source-level invariant test.**

Add `crossrow_kernel_source_uses_quartet_metadata_broadcast` beside the existing schedule-source tests. Assert that `crossrow_metal_schedule_source()` contains `simd_broadcast`, `scl[r]`, `bia[r]`, and `int(lid) / 4`. Run `cargo test -p higgs-models --lib crossrow_kernel_source_uses_quartet_metadata_broadcast -- --exact`; it must fail before the candidate implementation because the current source has no `simd_broadcast` call.

- [ ] **Step 2: Implement quartet broadcast.**

In `fetch_block`, load `sc[gi]` and `bi[gi]` from the first lane of each four-lane quartet and broadcast those values with the Metal SIMD broadcast primitive. Leave packed weight loads, nibble extraction, `qdot4`, `qdot4_pair`, `qdot4_triple`, and row accumulation unchanged.

- [ ] **Step 3: Run focused exactness tests.**

```bash
cargo test -p higgs-models --release --lib crossrow_qmv -- --nocapture
```

Require all existing supported-row and fallback tests to pass before timing.

- [ ] **Step 4: Benchmark the candidate in normal and Low Power Mode.**

Repeat the exact Task 2 grouped workload for five paired fresh-process trials. Retain the candidate only if all digests match and the median target throughput improves by at least 3%, or Low Power Mode improves by at least 3% while normal mode regresses by no more than 1%.

- [ ] **Step 5: Commit or revert the candidate.**

For a passing candidate, run `cargo fmt --all -- --check` with the known unchanged workspace baseline exception documented, run `git diff --check`, and commit:

```bash
git add crates/higgs-models/src/crossrow_qmv.rs
git commit -m "perf(crossrow): broadcast quartet metadata"
```

For a failing candidate, restore only the candidate file to its pre-task state and record the rejected result in the report.

---

### Task 5: Test row-group scheduling and reusable FastMetal configurations

**Files:**
- Modify: `crates/higgs-models/src/crossrow_qmv.rs:48-84,300-390`.
- Test: `crates/higgs-models/src/crossrow_qmv.rs:430-486` plus release cross-row tests.

**Interfaces:**
- Consumes: Task 3 bottleneck classification and Task 1 hot verifier widths.
- Produces: at most one schedule/config-cache candidate commit; rejected alternatives remain documented only.

- [ ] **Step 1: Add the failing schedule assertion.**

Change `crossrow_group_layout_covers_supported_rows` to expect `M=5` as `&[2, 2, 1][..]`, then run `cargo test -p higgs-models --lib crossrow_group_layout_covers_supported_rows -- --exact`; it must fail before the production match arm changes because the current layout is `[3, 2]`.

- [ ] **Step 2: Benchmark the schedule alternatives in isolation.**

Implement one compile-time selected `M=5` alternative at a time, preserving generated starts/sizes and row arithmetic. Run the cross-row exactness suite and the real benchmark at depths that produce `T=5`, with `HIGGS_CROSSROW_QMV=1` and `0` as controls.

- [ ] **Step 3: Add a thread-local configuration cache only if traces show launch overhead.**

Follow the existing `CachedMetalKernelConfig` and `RefCell<HashMap<...>>` pattern in `qwen3_next.rs`. Define `CrossrowQmvKernelConfigKey { out_dtype, t_rows, k_dim, n_rows, grid_x, grid_y }`, add a thread-local `CROSSROW_QMV_CONFIG_CACHE`, and cache `CachedMetalKernelConfig` values for the finite production shape set. Keep request-owned configs for nonpersistent shapes and preserve `mlx_fast_metal_kernel_config_free` through the existing owner type.

- [ ] **Step 4: Verify exactness and release behavior.**

```bash
cargo test -p higgs-models --release --lib crossrow_qmv -- --nocapture
cargo check --release -p higgs-models -p higgs-engine
```

- [ ] **Step 5: Retain only measured wins.**

Use five paired trials at the production depth in normal and Low Power Mode. Commit a candidate only when it meets the global retention rule and leaves fallback tests unchanged.

---

### Task 6: Long-run validation, independent review, and handoff

**Files:**
- Modify: `docs/superpowers/sdd/2026-08-17-qwen38-higgs-performance-port/task-6-report.md`.
- Review: complete branch diff against `nightly`.

**Interfaces:**
- Consumes: all retained candidate commits and raw logs from Tasks 1–5.
- Produces: final benchmark evidence and a clean, reviewable branch.

- [ ] **Step 1: Run long decode validation.**

Use the retained configuration with `BENCH_PROMPT_LEN=256` and `BENCH_DECODE_STEPS=256` for five normal and five Low Power fresh-process trials, alternating modes. Require stable acceptance, emitted counts, and both digest fields.

- [ ] **Step 2: Run a prompt-length check.**

Repeat three paired trials at `BENCH_PROMPT_LEN=1024` and `BENCH_DECODE_STEPS=128` to catch prompt/cache-shape regressions without changing the model or token construction.

- [ ] **Step 3: Run final verification.**

```bash
cargo test -p higgs-models --release --lib crossrow_qmv -- --nocapture
cargo test -p higgs-engine --release --lib mtp -- --nocapture
cargo check --release -p higgs-models -p higgs-engine
cargo build --release
rustfmt --edition 2024 --check crates/higgs-engine/src/mtp.rs
git diff --check
git diff nightly...HEAD --check
```

- [ ] **Step 4: Update the report with raw evidence.**

Record model path, power state, trial order, medians, exact digests, thermal/battery exclusions, rejected candidates, and retained commits in `task-6-report.md`.

- [ ] **Step 5: Request independent review.**

Generate a scoped review package for each retained candidate and dispatch a fresh reviewer. Resolve all Important/Critical findings, rerun affected tests, and keep the worktree clean.

- [ ] **Step 6: Commit the final report.**

```bash
git add -f .superpowers/sdd/2026-08-17-qwen38-higgs-performance-port/task-6-report.md
git commit -m "docs: record low power validation"
```
