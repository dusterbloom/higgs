# Qwen3.8 Performance Port to Higgs Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Improve Qwen3.8/Qwen3.5-MoE speculative verification on Higgs nightly by adding the promoted cross-row row grouping where it is compatible with Higgs's existing exact Rust/MLX path, then integrate only measured-safe MTP tuning.

**Architecture:** Keep QLinear::forward as the eligibility and fallback boundary. Make the row-group schedule a Rust-owned table used for host dispatch and Metal-source generation, so M2..M9 layouts cannot drift. The Metal kernel will share packed weight reads across groups of up to three input rows while retaining the existing scalar affine4/group64 arithmetic and output dtype. MTP policy changes remain separate and model-agnostic, gated by focused controller tests and end-to-end token/throughput evidence.

**Tech Stack:** Rust 2024 workspace, mlx-rs/MLX 0.31, custom Metal kernels through mlx_sys::mlx_fast_metal_kernel, Cargo unit tests, Apple Metal GPU benchmark tests.

## Global Constraints

- Work only in the isolated worktree on codex/qwen38-higgs-port, based on nightly.
- Preserve exact greedy verification: cross-row outputs must match the stock per-row affine4/group64 quantized matmul under the existing exactness contract.
- Preserve unsupported-shape, disabled-feature, and kernel-error fallbacks.
- Do not copy challenge manifests, submission metadata, or Swift implementation details into Higgs.
- Do not modify the original pr-187 checkout or its docs/superpowers/ and qwen38-challenge/ changes.
- Keep optimization slices in separate commits with focused tests and benchmark evidence.
- Use HIGGS_CROSSROW_QMV=0 to compare the stock fallback against the cross-row path.
- Cap any MTP draft-depth change at the existing MAX_MTP_DRAFT_N_MAX limit of 8.

---

## File Map

- crates/higgs-models/src/crossrow_qmv.rs: owns the Rust row-group table, generated Metal schedule helpers, grouped affine4 kernel, host grid sizing, and cross-row exactness tests.
- crates/higgs-models/src/qwen3_next.rs: remains the QLinear eligibility boundary; modify only if grouped dispatch needs a narrow shape or environment correction.
- crates/higgs-engine/src/mtp.rs: owns architecture-neutral adaptive depth behavior and controller tests; modify only for a measured policy improvement.
- crates/higgs-engine/src/mlx_tuning.rs: owns default/configured MTP depth; modify only when the baseline/AB comparison demonstrates a safe Qwen3.8-compatible default.
- crates/higgs-engine/src/simple.rs: owns the runtime adaptive-depth wiring; modify only if controller telemetry needs a minimal integration change.
- docs/superpowers/specs/2026-08-17-qwen38-higgs-performance-port-design.md: approved design constraints and delivery boundary.
- docs/superpowers/plans/2026-08-17-qwen38-higgs-performance-port.md: this execution plan.

## Task 1: Establish the clean nightly baseline

**Files:**
- Read: crates/higgs-models/src/crossrow_qmv.rs
- Read: crates/higgs-engine/src/mtp.rs
- Read: crates/higgs-models/src/qwen3_next.rs
- Read: crates/higgs-engine/src/mlx_tuning.rs
- Record: local benchmark output outside the repository or in the handoff, not as a source change.

**Interfaces:**
- Consumes: clean nightly checkout at d499fe22434198c0802f21a91939647077e7ae6c.
- Produces: verified baseline test results, cross-row enabled/disabled timing, and MTP depth/acceptance observations used by later gates.

- [ ] Step 1: Confirm the worktree and baseline commit are clean

Run:

~~~bash
git status --short
git branch --show-current
git log -1 --oneline
git check-ignore -q .worktrees
~~~

Expected: no status output, branch codex/qwen38-higgs-port, HEAD at the nightly tip, and .worktrees ignored.

- [ ] Step 2: Run the focused cross-row correctness baseline

Run:

~~~bash
cargo test -p higgs-models --lib crossrow_qmv -- --nocapture
~~~

Expected: the existing M2..M9 exactness test passes on the Apple Metal host.

- [ ] Step 3: Run the adaptive MTP controller baseline

Run:

~~~bash
cargo test -p higgs-engine --lib mtp -- --nocapture
~~~

Expected: the existing acceptance/depth controller tests pass.

- [ ] Step 4: Run the real-model baseline when the local checkpoint exists

Run with the locally available Qwen3.8 challenge checkpoint, for example:

~~~bash
HIGGS_MODEL_PATH="$HOME/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit" \
BENCH_PROMPT_LEN=256 BENCH_DECODE_STEPS=32 \
cargo test -p higgs-models --release --lib \
  bench_actual_qwen3_5_mtp_decode --ignored --exact --nocapture
~~~

Run the same test with HIGGS_CROSSROW_QMV=0 and record total tokens/sec, cycle time, drafted/accepted behavior, and any model-load or unsupported-path messages. If the checkpoint is unavailable, record that the GPU benchmark is unavailable and continue with the deterministic focused tests.

- [ ] Step 5: Record the current MTP defaults before editing them

Inspect default_mtp_draft_n_max and the HIGGS_MTP_DRAFT_N_MAX override. Record the effective depth for the target model class and do not change it in this task. This prevents a grouping result from being confused with a policy change.

## Task 2: Add a single Rust-owned row-group schedule

**Files:**
- Modify: crates/higgs-models/src/crossrow_qmv.rs
- Test: crates/higgs-models/src/crossrow_qmv.rs unit-test module

**Interfaces:**
- Consumes: t_rows values already restricted by QLinear::crossrow_qmv_forward to M2..M9.
- Produces: crossrow_group_layout(t_rows) -> &'static [i32] and crossrow_group_count(t_rows) -> i32, used by both host grid setup and Metal source generation.

- [ ] Step 1: Write the failing schedule tests

Add a pure Rust test that asserts the full supported table:

~~~rust
#[test]
fn crossrow_group_layout_covers_supported_rows() {
    let expected = [
        (2, &[2][..]),
        (3, &[2, 1][..]),
        (4, &[2, 2][..]),
        (5, &[3, 2][..]),
        (6, &[3, 3][..]),
        (7, &[3, 2, 2][..]),
        (8, &[3, 3, 2][..]),
        (9, &[3, 3, 3][..]),
    ];
    for (rows, groups) in expected {
        assert_eq!(crossrow_group_layout(rows), groups);
        assert_eq!(crossrow_group_count(rows), groups.len() as i32);
        assert_eq!(groups.iter().sum::<i32>(), rows);
    }
}
~~~

Also assert that unsupported M values return an empty layout and zero group count. The M8/M9 entries are the promoted grouping; M2..M7 retain two-row sharing where possible and use a three-row group only when it reduces the number of groups.

- [ ] Step 2: Run the schedule test to verify it fails

~~~bash
cargo test -p higgs-models --lib crossrow_group_layout -- --nocapture
~~~

Expected: compilation/test failure because the schedule functions do not yet exist.

- [ ] Step 3: Implement the minimal Rust schedule

Add const match-based functions in crossrow_qmv.rs with exactly these layouts:

~~~rust
const fn crossrow_group_layout(t_rows: i32) -> &'static [i32] {
    match t_rows {
        2 => &[2],
        3 => &[2, 1],
        4 => &[2, 2],
        5 => &[3, 2],
        6 => &[3, 3],
        7 => &[3, 2, 2],
        8 => &[3, 3, 2],
        9 => &[3, 3, 3],
        _ => &[],
    }
}

const fn crossrow_group_count(t_rows: i32) -> i32 {
    crossrow_group_layout(t_rows).len() as i32
}
~~~

Keep the functions private to the module and use them for all later host-side layout decisions.

- [ ] Step 4: Run the schedule tests to verify they pass

~~~bash
cargo test -p higgs-models --lib crossrow_group_layout -- --nocapture
~~~

Expected: PASS.

- [ ] Step 5: Commit the schedule slice

~~~bash
git add crates/higgs-models/src/crossrow_qmv.rs
git commit -m "test(crossrow): define grouped row schedules"
~~~

## Task 3: Implement grouped cross-row Metal verification

**Files:**
- Modify: crates/higgs-models/src/crossrow_qmv.rs
- Test: crates/higgs-models/src/crossrow_qmv.rs exactness module

**Interfaces:**
- Consumes: crossrow_group_layout and crossrow_group_count from Task 2.
- Produces: one cross-row kernel that handles group sizes 1, 2, and 3, plus host dispatch sized by the schedule.

- [ ] Step 1: Add a source-generation test for the schedule mirror

Add a pure Rust test for the generated Metal schedule source. For each M2..M9, assert that the generated source includes the group start/size entries derived from crossrow_group_layout(M). The test must also assert that M8 contains 3, 3, 2 and M9 contains 3, 3, 3, preventing a future host/kernel mismatch.

- [ ] Step 2: Run the source-generation test to verify it fails

~~~bash
cargo test -p higgs-models --lib crossrow_metal_schedule -- --nocapture
~~~

Expected: FAIL because the generated schedule helper does not yet exist.

- [ ] Step 3: Generate compile-time Metal group helpers from the Rust table

Replace the fixed pair-only source assembly with a helper that emits Metal template functions for crossrow_group_start<M>(group) and crossrow_group_size<M>(group). Generate one switch branch for M2..M9 using the Rust table, then concatenate it between CROSSROW_QMV_SOURCE and the kernel entry. Keep the kernel cached in the existing OnceLock.

The generated functions must return the same row starts as the table. They must not add a runtime schedule buffer or change the MLX kernel argument list.

- [ ] Step 4: Replace pair-only accumulation with up-to-three-row groups

In CROSSROW_QMV_ENTRY, retain load_x4, fetch_block, and the existing four-nibble extraction order. Add a third-row accumulation form that performs the same scalar per-row multiply/add sequence as qdot4 and qdot4_pair. For each group:

1. Compute first_m and group_size from the generated helpers.
2. Load row 0 and conditionally rows 1 and 2 from the same K block.
3. Reuse the fetched packed weight words and scale/bias values for each input row in the group.
4. Reduce each row independently with simd_sum and store only the rows in the group.

Use fixed-size local storage for three rows so Metal has no dynamic allocation. Do not combine rows' scalar accumulations into a shared sum; exact per-row arithmetic is the correctness boundary.

- [ ] Step 5: Size the host grid from the Rust schedule

In crossrow_qmv_verify, replace (t_rows + 1) / 2 with crossrow_group_count(t_rows) and keep the existing 64-thread threadgroup and 8-output-tile y grid. Reject an unsupported schedule before kernel dispatch with the existing error style, although the caller's M2..M9 eligibility makes that unreachable in normal use.

- [ ] Step 6: Run formatting and the focused exactness suite

~~~bash
cargo fmt --all -- --check
cargo test -p higgs-models --lib crossrow_qmv -- --nocapture
~~~

Expected: formatting passes and the stock-vs-cross-row test passes for all M2..M9 and all existing shapes.

- [ ] Step 7: Add adversarial nibble/activation cases before committing

Extend crossrow_bit_exact_vs_stock_rows with deterministic cases containing all-zero nibbles, maximum 4-bit nibbles, alternating signs, zero activations, negative activations, and a nontrivial third row for M6/M8/M9. Compare each result against the existing per-row quantized_matmul reference with the same max diff == 0.0 assertion.

- [ ] Step 8: Run the expanded exactness test and commit

~~~bash
cargo test -p higgs-models --lib crossrow_qmv -- --nocapture
git add crates/higgs-models/src/crossrow_qmv.rs
git commit -m "perf(crossrow): share weights across grouped rows"
~~~

## Task 4: Verify QLinear integration and fallback behavior

**Files:**
- Read/modify only if required: crates/higgs-models/src/qwen3_next.rs
- Test: existing QLinear/model tests plus cross-row tests

**Interfaces:**
- Consumes: grouped crossrow_qmv_verify from Task 3.
- Produces: unchanged public QLinear behavior with grouped M8/M9 dispatch and stock fallback for all other cases.

- [ ] Step 1: Add or extend a pure eligibility test if the current module has one

Cover these cases at the existing QLinear eligibility boundary: affine bits 4, group 64, K divisible by 512, N divisible by 8, M2..M9, disabled HIGGS_CROSSROW_QMV, M1, M10, non-affine mode, and non-eligible K/N shapes. The expected result is cross-row only for the first eligible set and None for every fallback case.

- [ ] Step 2: Run the eligibility test before changing production code

~~~bash
cargo test -p higgs-models --lib crossrow -- --nocapture
~~~

Expected: the existing behavior passes or the new test exposes the exact eligibility helper that needs a minimal correction.

- [ ] Step 3: Make the smallest integration correction, if needed

Keep the current ordering after qgemm verification and before the stock mode match. Do not broaden the quantization or shape domain while integrating the group schedule. Preserve the HIGGS_CROSSROW_QMV=0 opt-out.

- [ ] Step 4: Run model-package verification and commit only integration edits

~~~bash
cargo test -p higgs-models --lib crossrow -- --nocapture
cargo check -p higgs-models
git diff --check
~~~

If qwen3_next.rs was unchanged, do not create a no-op commit. If it changed, commit it separately:

~~~bash
git add crates/higgs-models/src/qwen3_next.rs
git commit -m "fix(qwen3): preserve crossrow fallback boundaries"
~~~

## Task 5: Measure and integrate MTP policy only when evidence supports it

**Files:**
- Test/modify: crates/higgs-engine/src/mtp.rs
- Modify only on an approved measured result: crates/higgs-engine/src/mlx_tuning.rs, crates/higgs-engine/src/simple.rs

**Interfaces:**
- Consumes: unchanged token/cache behavior and real-model results from Tasks 1–4.
- Produces: either a documented no-change decision or a bounded, tested MTP policy change that improves throughput without changing tokens.

- [ ] Step 1: Run the real-model AB matrix

For the same checkpoint, prompt length, decode length, and warm-up, run:

~~~bash
HIGGS_MODEL_PATH="$TARGET_MODEL" HIGGS_CROSSROW_QMV=1 HIGGS_MTP_ADAPTIVE_DRAFT=0 HIGGS_MTP_DRAFT_N_MAX=2 cargo test -p higgs-models --release --lib bench_actual_qwen3_5_mtp_decode --ignored --exact --nocapture
HIGGS_MODEL_PATH="$TARGET_MODEL" HIGGS_CROSSROW_QMV=1 HIGGS_MTP_ADAPTIVE_DRAFT=1 HIGGS_MTP_DRAFT_N_MAX=3 cargo test -p higgs-models --release --lib bench_actual_qwen3_5_mtp_decode --ignored --exact --nocapture
HIGGS_MODEL_PATH="$TARGET_MODEL" HIGGS_CROSSROW_QMV=1 HIGGS_MTP_ADAPTIVE_DRAFT=1 HIGGS_MTP_DRAFT_N_MAX=4 cargo test -p higgs-models --release --lib bench_actual_qwen3_5_mtp_decode --ignored --exact --nocapture
~~~

Repeat with HIGGS_CROSSROW_QMV=0 only when the first baseline was available. Record token trajectory, average cycle time, accepted drafts, and throughput.

- [ ] Step 2: Add controller tests for any policy change before implementation

If the AB matrix shows a stable gain from policy behavior rather than the grouped kernel, add tests in mtp.rs covering: full acceptance increases depth by one, a rejection at or below 25% decreases depth by one, 75% acceptance does not exceed the configured maximum, zero drafts resets to the minimum, and the depth never leaves 1..=8. The tests must assert only controller state and must not require a model or GPU.

- [ ] Step 3: Implement the smallest model-agnostic policy change

Keep AdaptiveDraftDepth bounded by MAX_MTP_DRAFT_N_MAX, preserve the environment override, and do not add challenge-specific model-name checks. If the evidence supports only a higher target depth, change the tuning default or runtime cap in its owning file rather than hard-coding a value in mtp_cycle. If no policy change is faster and token-identical, leave production MTP code unchanged and record that the grouped verifier is the accepted port.

- [ ] Step 4: Run controller and engine tests

~~~bash
cargo test -p higgs-engine --lib mtp -- --nocapture
cargo test -p higgs-engine --lib mlx_tuning -- --nocapture
cargo check -p higgs-engine
~~~

- [ ] Step 5: Commit only a measured MTP change

~~~bash
git add crates/higgs-engine/src/mtp.rs crates/higgs-engine/src/mlx_tuning.rs crates/higgs-engine/src/simple.rs
git commit -m "perf(mtp): tune verified draft depth"
~~~

Do not stage untouched files; if no MTP change is justified, omit this commit.

## Task 6: Verify the complete port and prepare review

**Files:**
- Read: all changed files and the approved design/plan.
- No new production file is needed unless a benchmark harness is required by an observed gap.

**Interfaces:**
- Consumes: all committed optimization slices and their focused test results.
- Produces: clean verified branch, performance comparison, and reviewer-ready upstream handoff.

- [ ] Step 1: Run the affected package tests and checks

~~~bash
cargo fmt --all -- --check
cargo test -p higgs-models --lib crossrow_qmv -- --nocapture
cargo test -p higgs-engine --lib mtp -- --nocapture
cargo test -p higgs-engine --lib mlx_tuning -- --nocapture
cargo check -p higgs-models -p higgs-engine
git diff --check
~~~

- [ ] Step 2: Run the real-model final AB

Use identical warm-up, prompt, decode length, and environment for stock fallback, grouped cross-row, and any retained MTP policy. Run enough repeated trials to compare medians, and retain the raw output in the handoff. Confirm the emitted token sequence is identical between stock and optimized modes.

- [ ] Step 3: Inspect the final diff and history

Run:

~~~bash
git status --short
git diff nightly...HEAD --stat
git diff nightly...HEAD --check
git log --oneline --decorate nightly..HEAD
rg -n 'TODO|TBD|FIXME|XXX|HACK|<<<<<<<|=======|>>>>>>>' $(git diff --name-only nightly...HEAD)
~~~

Expected: only intentional design and optimization files are changed, no conflict markers or placeholder notes remain, and the original nightly ref is unchanged.

- [ ] Step 4: Request independent code review

Use the requesting-code-review workflow with the branch diff and test/benchmark evidence. Address only actionable findings, rerun the affected tests, and commit review fixes separately.

- [ ] Step 5: Stop at the delivery decision

After verification and review, present the branch status, exact benchmark comparison, commit list, and the explicit choice to keep the branch, merge locally, or push/create a PR against origin/nightly. Do not move nightly or push unreviewed code.

