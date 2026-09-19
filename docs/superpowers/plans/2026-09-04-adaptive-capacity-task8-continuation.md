# Adaptive Capacity Task 8 Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the synchronized Higgs/Nanobot adaptive-capacity release by making optional caches yield to the minimum working request, preserving native Escha execution, and completing real-hardware and cross-binary validation.

**Architecture:** Keep Higgs's existing `CapacityController` as the only byte solver. `cache_allocations_with` will binary-search only the non-frozen optional-cache budget and validate each candidate with fresh conservative controllers for the prospective model and every non-draining active model; draining allocations remain a hard floor, while the existing desired/published maximum remains authoritative until engine acknowledgement. The remaining interrupted smoke changes will be cleaned and verified before a native-only Escha replay and Nanobot HTTP conformance run.

**Tech Stack:** Rust 2024, MLX/Metal, Tokio/Axum, Cargo release profiles, tmux, GitNexus CLI, OpenCode CLI with `opencode/muse-spark-1.3-contributor-free`.

## Global Constraints

- Run every build and test in release mode; never create debug artifacts.
- Run long-lived servers and hardware replay commands in `tmux`.
- Preserve unrelated `AGENTS.md` and `CLAUDE.md` worktree changes.
- Keep `HIGGS_ESCHA_NATIVE=1` explicit in every Escha hardware command; affine execution is a failed gate.
- Do not weaken the 20%/30% protected reserve, critical-pressure rejection, the 1,024-token completion floor, frozen draining allocations, or desired/published cache acknowledgement.
- A positive capacity envelope must come from the existing checked byte solver; do not add a second heuristic or hardware table.
- The required upstream impact analysis already classified `cache_allocations_with` as CRITICAL with 41 downstream symbols; the user explicitly approved this edit. Treat registration, pressure recomputation, model load/unload, cache acknowledgement, and request admission as the verification scope.

---

### Task 1: Make optional caches preserve the minimum request

**Files:**
- Modify: `crates/higgs/src/capacity/registry.rs:1927-2180`
- Test: `crates/higgs/src/capacity/registry.rs:3200-4100`

**Interfaces:**
- Consumes: `cache_allocations_with(&RegistryState, Option<&ModelCapacityFacts>)`, `ModelCapacityFacts::controller(SharedLedger, MemoryPressure)`, `fair_cache_allocations`, `extend_fair_allocations`, `shared_ledger_with`, `apply_allocation_totals`.
- Produces: a cache plan whose frozen bytes are unchanged and whose flexible bytes are the largest fair budget that leaves every prospective/non-draining model's conservative `CapacityDecision` available.

- [x] **Step 1: Write the failing constrained-registration test**

  Keep `registration_reduces_optional_caches_to_preserve_minimum_request`. It models 12 GiB loaded residency, a 4 GiB cold transient, a 24 GiB authority under constrained pressure, and a 1,024-token minimum. It must assert successful registration, `safe_total_tokens == 1_024`, and a positive cache allocation below the configured 3 GiB ceiling.

- [x] **Step 2: Run the test and record the expected failure**

  Run:

  ```bash
  cargo test -p higgs --release capacity::registry::tests::registration_reduces_optional_caches_to_preserve_minimum_request -- --exact --nocapture
  ```

  Expected before the fix: panic on `InsufficientCapacity("escha")`.

- [x] **Step 3: Add the monotonic budget test before production code**

  Add this second test beside the registration regression. `CapacityController` receives retained and prefix allocations as one summed byte charge, so reducing their total must never change a failed minimum request into a larger failure:

  ```rust
  #[test]
  fn minimum_request_fit_is_monotonic_in_total_cache_bytes() {
      let registry = CapacityRegistry::new(["escha".to_owned()]);
      let mut model = facts("escha", 12 * GIB);
      model.costs.transient_prefill.base_bytes = 4 * GIB;
      model.configured_total_token_ceiling = Some(1_024);
      model.configured_output_token_ceiling = Some(1_024);
      registry.refresh_memory(model.memory);
      registry.apply_pressure_observation(PressureObservation {
          pressure: MemoryPressure::Constrained,
          swap_out_delta: 0,
          compressor_delta: 0,
      });

      assert!(minimum_requests_fit_with_cache_bytes(&registry.lock(), Some(&model), 0).unwrap());
      assert!(minimum_requests_fit_with_cache_bytes(
          &registry.lock(),
          Some(&model),
          128 * 1024 * 1024
      ).unwrap());
      assert!(!minimum_requests_fit_with_cache_bytes(
          &registry.lock(),
          Some(&model),
          3 * GIB
      ).unwrap());
  }
  ```

  Run the existing draining-cache, stale-revision, FIFO handoff, and live-reservation tests unchanged as the hard-floor and acknowledgement regressions; do not duplicate those fixtures.

- [x] **Step 4: Implement byte-budget search inside the existing cache path**

  After the existing raw cache envelope and frozen-total checks, binary-search `0..=min(raw_flexible_envelope, requested_flexible_total)`. Each tested `cache_bytes` is `frozen_total + flexible_midpoint`; the search never scales or redistributes the frozen portion. Apply the total to a copied `SharedLedger`, set `active_reservation_bytes = 0`, and require a fresh conservative controller to report `Available` for `added` plus each non-draining active model.

  The helper boundary must remain local to `registry.rs`:

  ```rust
  fn minimum_requests_fit_with_cache_bytes(
      state: &RegistryState,
      added: Option<&ModelCapacityFacts>,
      cache_bytes: u64,
  ) -> Result<bool, RegistrationError>;
  ```

  This search computes the future **desired** cache plan only. It intentionally excludes in-flight reservations and old published bytes because neither may permanently shrink the client-visible semantic envelope; the unchanged caller still evaluates `max(desired, published)` with live reservation bytes before registration or admission can succeed. Thus a stale engine acknowledgement remains conservative, and a learned profile that makes the exact candidate unavailable is still rejected by `commit_active`.

  Use checked arithmetic. `Ok(false)` moves the binary-search bound, arithmetic errors abort, and a frozen-floor failure for an added model becomes `RegistrationError::InsufficientCapacity`. Ordinary recomputation may publish zero flexible bytes without changing the frozen floor. Do not mutate active controllers during the search.

- [x] **Step 5: Prove targeted and registry behavior**

  Run:

  ```bash
  cargo test -p higgs --release capacity::registry::tests::registration_reduces_optional_caches_to_preserve_minimum_request -- --exact --nocapture
  cargo test -p higgs --release capacity::registry::tests -- --nocapture
  cargo test -p higgs --release capacity::tests -- --nocapture
  ```

  Expected: all pass; existing FIFO, stale-revision, drain, rollback, and pressure tests remain green.

### Task 2: Clean and validate the interrupted native-Escha smoke changes

**Files:**
- Modify: `crates/higgs-engine/src/mlx_tuning.rs:300-380`
- Modify: `crates/higgs-models/src/eschamoe.rs:1360-1785,3740-3815`
- Modify: `crates/higgs/src/doctor.rs:740-930`
- Modify: `crates/higgs/src/main.rs:1210-1280`
- Modify: `crates/higgs/src/state.rs:1440-1580,1900-2030`

**Interfaces:**
- Consumes: `higgs_models::eschamoe::native_mode`, `resident_estimate_bytes`, `model_load_estimate`, `MlxMemorySnapshot::measure`, `CapacityRegistry::refresh_memory`.
- Produces: one native-Escha load estimate shared with doctor, one boot measurement, and a post-conversion allocator snapshot without diagnostic-only tracing.

- [x] **Step 1: Add/retain native-estimator regression coverage**

  Test flat and nested `text_config`, `qwen3_5_vl` classification, native trellis `layer_meta`, affine fallback math, and fail-closed fallback when architecture fields are missing. Compare the estimator with the observed load peak during Task 4 rather than making a unit test depend on a user-local model path or a fixed machine-specific byte count.

  Keep preload and in-load accounting identical: a native estimate's resident floor is `required_process_bytes - workspace_upper_bound_bytes`, while missing/overflowing architecture metadata retains the full-artifact fallback. Cover that bound directly and keep the existing Gemma full-artifact regression unchanged.

- [x] **Step 2: Remove interrupted-session debris**

  Remove `HIGGS_TRACE_CAPACITY_LOAD` instrumentation from `build_engine_with_capacity`, eliminate the duplicate non-boot memory refresh in `AppState::with_capacity_registry`, restore exactly one `#[cfg(test)]`/lint attribute block before `mod tests` and `mod convert_dump` in `eschamoe.rs`, and warn if the one boot authority measurement fails before the existing fail-closed rejection.

- [x] **Step 3: Keep native execution explicit and fail observable**

  Preserve the default-native implementation:

  ```rust
  pub fn native_mode() -> bool {
      !std::env::var("HIGGS_ESCHA_NATIVE").is_ok_and(|value| value == "0")
  }
  ```

  Keep `execution_mode` equal to `simple:eschamoe-native` for native Escha. Do not introduce automatic affine fallback.

- [x] **Step 4: Run focused release checks**

  ```bash
  cargo fmt --all -- --check
  cargo test -p higgs-models --release eschamoe -- --nocapture
  cargo test -p higgs-engine --release model_load_estimate -- --nocapture
  cargo test -p higgs --release doctor -- --nocapture
  HIGGS_ESCHA_NATIVE=1 target/release/higgs doctor -c /Users/peppi/.config/higgs/config.toml
  ```

  Expected doctor evidence: the checkpoint remains in trellis form and uses the Metal kernel; no affine-path warning.

### Task 3: Run the full Higgs release gate

**Files:**
- Verify only; no planned source edits.

**Interfaces:**
- Consumes: completed Tasks 1-2.
- Produces: release binaries and fresh correctness evidence.

- [x] **Step 1: Run full release tests and build serially in tmux**

  ```bash
  tmux new-session -d -s higgs-cap-release 'cd /private/tmp/higgs-adaptive-capacity && set -o pipefail && cargo test --release 2>&1 | tee /private/tmp/higgs-cap-release-test.log; code=$?; echo __HIGGS_TEST_EXIT__=$code | tee -a /private/tmp/higgs-cap-release-test.log; exit $code'
  tmux new-session -d -s higgs-cap-build-final 'cd /private/tmp/higgs-adaptive-capacity && set -o pipefail && cargo build --release 2>&1 | tee /private/tmp/higgs-cap-release-build.log; code=$?; echo __HIGGS_BUILD_EXIT__=$code | tee -a /private/tmp/higgs-cap-release-build.log; exit $code'
  ```

  Start `higgs-cap-build-final` only after `higgs-cap-release` exits successfully. Inspect the final `test result`/`Finished release` lines and require both exit markers to equal zero.

- [x] **Step 2: Run Git hygiene checks**

  ```bash
  git diff --check
  cargo fmt --all -- --check
  ```

  Expected: no whitespace or formatting failures.

### Task 4: Replay native Escha on real hardware

**Files:**
- Evidence: `/private/tmp/higgs-cap-native.log`
- Evidence: `/private/tmp/higgs-cap-native-memory.log`

**Interfaces:**
- Consumes: `target/release/higgs`, real Escha checkpoint, production Higgs config.
- Produces: native runtime identity, capacity JSON, metrics, request results, and before/after VM counters.

- [x] **Step 1: Start server and monitor in tmux**

  ```bash
  tmux new-session -d -s higgs-cap-native 'cd /private/tmp/higgs-adaptive-capacity && HIGGS_ESCHA_NATIVE=1 RUST_LOG=info target/release/higgs serve -c /Users/peppi/.config/higgs/config.toml 2>&1 | tee /private/tmp/higgs-cap-native.log'
  ```

  Record `memory_pressure`, `vm_stat` swap-outs, compressor pages, and Higgs RSS before load, at peak, after publication, and after the replay. The replay fails and no commit is allowed if swap-outs increase, pressure reaches critical, or constrained pressure persists for more than 60 seconds after the load completes.

- [x] **Step 2: Verify publication and native identity**

  Query authenticated `/v1/capacity?model=escha-35b-a3b` and `/metrics`. Require `schemaVersion == 1`, `availability: available`, `recommendedOutputTokens >= 1024`, nonnegative fair cache budgets, `pressure != critical`, and logs containing `Installed native trellis expert weights`. Unit coverage must assert `execution_mode == "simple:eschamoe-native"`.

- [x] **Step 3: Exercise genuine capacity bands**

  Run progressively larger cold requests, one retained warm session with tool-shaped turns, a cache-only control, and three cold starts near the published prompt boundary. Require zero new swap-outs, no sustained constrained/critical pressure, no typed 413 below the published envelope, and the expected typed response above it.

### Task 5: Complete synchronized Nanobot conformance and review

**Files:**
- Verify: `/private/tmp/nac`
- Modify only if a synchronized-contract test exposes a real defect.

**Interfaces:**
- Consumes: running Higgs server and Nanobot commit `9c3975f`.
- Produces: cross-binary schema/error/retry evidence, release test/build logs, and matched turn benchmark results.

- [x] **Step 1: Pin the synchronized branch and run real HTTP conformance**

  Confirm `/private/tmp/nac` is at Nanobot commit `9c3975f` before testing. Validate `schemaVersion == 1`, all capacity fields, exact 413 fields, typed 503 behavior, retained-session rotation after compaction, one retry for one logical turn, and no duplicated tool calls/results. Keep the same Higgs `bootId`/generation evidence with each case.

- [x] **Step 2: Run Nanobot compatibility and release gates**

  From `/private/tmp/nac`:

  ```bash
  cargo test --release
  cargo build --release
  scripts/turn_bench.sh
  git diff --check
  cargo fmt --all -- --check
  ```

  Run long commands in named tmux sessions and retain their logs.

- [x] **Step 3: Request OpenCode Muse/Spark review**

  Use `opencode/muse-spark-1.3-contributor-free` in read-only plan/review mode. Give it the final diff, the original adaptive-capacity spec, the failing/passing regression output, native replay evidence, and cross-binary results. Resolve correctness findings; do not accept stylistic expansion or affine fallback.

- [x] **Step 4: Run final GitNexus scope checks and commit**

  In each changed repository prefer the checked-in runner required by `AGENTS.md`:

  ```bash
  node .gitnexus/run.cjs detect-changes --scope all --repo .
  node .gitnexus/run.cjs detect-changes --scope compare --base-ref main --repo .
  ```

  Require non-partial results whose affected symbols and processes match the plan. Commit only task-owned files; preserve unrelated `AGENTS.md` and `CLAUDE.md` changes.
