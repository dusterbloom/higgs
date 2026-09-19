# Runtime Model Load Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the lifecycle and security gaps found in the PR #187 review while preserving the public `AppState` construction surface.

**Architecture:** Move runtime-load coordination into the private `Router`, which owns both the concurrent-load semaphore and resident-model permits. Resolve and authorize a single canonical model path before loading, and move the concurrency permit into the blocking load task so cancellation cannot release it early. Retain each resident permit through engine draining on unload.

**Tech Stack:** Rust, Tokio, Axum, MLX, `std::path`, existing unit/integration tests.

## Global Constraints

- Runtime loads accept Hugging Face IDs resolved from the local cache or canonical local paths under configured roots.
- Runtime loading remains opt-in and requires a meaningful, valid `server.api_key`.
- Startup-configured engines do not consume runtime resident-model permits.
- Runtime resident permits remain held until the engine's final `Arc` reference is dropped.
- No network download or process-wide MLX cache-clear claim may appear in runtime-load docs.
- Preserve public `AppState` struct-literal compatibility by keeping coordination state private to `Router`.

---

### Task 1: Bind runtime authorization to the loaded path

**Files:**
- Modify: `crates/higgs/src/model_resolver.rs`
- Modify: `crates/higgs/src/routes/models.rs`
- Test: `crates/higgs/src/model_resolver.rs`

- [ ] **Step 1: Write failing path-binding tests**

Add tests that resolve an HF cache ID to one concrete directory and verify the route loader can consume that returned path without re-resolving the caller string; retain the existing relative-directory and symlink-escape tests.

- [ ] **Step 2: Run the resolver tests and confirm the missing single-resolution API fails**

Run: `cargo test -p higgs --lib runtime_policy`

Expected: the new test fails because the resolver currently returns only an authorization result and the route calls `resolve` again.

- [ ] **Step 3: Implement one-shot runtime resolution**

Add a resolver returning the canonical `PathBuf` used for loading. HF IDs must resolve through the HF cache helper without checking the current working directory; local paths must canonicalize once and pass component-aware root containment. Update `load_model` to pass that returned `PathBuf` to `build_engine`.

- [ ] **Step 4: Run the resolver and route tests**

Run: `cargo test -p higgs --lib model_resolver::tests::test_runtime_policy`

Expected: all path-policy tests pass.

### Task 2: Make load concurrency cancellation-safe

**Files:**
- Modify: `crates/higgs/src/router.rs`
- Modify: `crates/higgs/src/routes/models.rs`
- Test: `crates/higgs/src/routes/models.rs`

- [ ] **Step 1: Write a failing permit-lifetime test**

Add a deterministic test using a blocking task and an owned semaphore permit that asserts a second load cannot acquire the permit until the first blocking task exits, even if the outer future is dropped.

- [ ] **Step 2: Run the focused test and confirm the current permit is released too early**

Run: `cargo test -p higgs --lib runtime_load_permit_survives_outer_cancellation`

Expected: failure against the current route-owned permit.

- [ ] **Step 3: Move coordination into `Router` and capture the owned permit in `spawn_blocking`**

Initialize the load semaphore from validated config inside `Router::from_config`. Return a runtime-load guard whose owned permit is moved into the blocking closure and only released when that closure returns.

- [ ] **Step 4: Run focused concurrency tests**

Run: `cargo test -p higgs --lib runtime_load_permit_survives_outer_cancellation` and `cargo test -p higgs --lib runtime_load_gate_uses_configured_concurrency`

Expected: all focused tests pass.

### Task 3: Retain resident quota through insertion and unload drain

**Files:**
- Modify: `crates/higgs/src/router.rs`
- Modify: `crates/higgs/src/routes/models.rs`
- Modify: `crates/higgs/src/state.rs`
- Test: `crates/higgs/src/router.rs`
- Test: `crates/higgs/src/routes/models.rs`
- Test: `crates/higgs/tests/integration/api_contract.rs`
- Test: `crates/higgs/tests/integration/proxy_e2e.rs`

- [ ] **Step 1: Write failing quota-lifecycle tests**

Test that startup engines do not consume runtime capacity, a queued second load cannot insert after the first fills the quota, and an unload with an in-flight `Arc` keeps the resident permit held until the drain finishes.

- [ ] **Step 2: Run the quota tests and confirm current name-count behavior fails**

Run: `cargo test -p higgs --lib runtime_engine_count_excludes_startup_engines`

Expected: the lifecycle assertions fail because the current count is tied to routing-table membership rather than engine lifetime.

- [ ] **Step 3: Store private resident permits with runtime engine metadata**

Create the resident semaphore from `runtime_max_loaded_models`, acquire its owned permit before the blocking load, store it with the runtime engine on insertion, and return it with removed engines. Pass the permit into both immediate and detached drain paths so a new load cannot acquire capacity until final engine destruction.

- [ ] **Step 4: Remove the public `AppState` semaphore field**

Keep `AppState` fields source-compatible and route runtime coordination through `Router` methods. Update constructors and tests accordingly.

- [ ] **Step 5: Run quota and integration tests**

Run: `cargo test -p higgs --tests`

Expected: all package tests pass.

### Task 4: Validate API keys and semaphore bounds

**Files:**
- Modify: `crates/higgs/src/config.rs`
- Modify: `crates/higgs/src/doctor.rs`
- Test: `crates/higgs/src/config.rs`
- Test: `crates/higgs/src/doctor.rs`

- [ ] **Step 1: Write failing tests**

Add tests rejecting blank/whitespace/invalid-header API keys and rejecting zero or over-limit runtime semaphore values.

- [ ] **Step 2: Run the focused validation tests**

Run: `cargo test -p higgs --lib api_key` and `cargo test -p higgs --lib runtime_concurrent`

Expected: the new tests fail against permissive validation.

- [ ] **Step 3: Implement validation and doctor behavior**

Reject blank or header-unsafe keys during config validation, make doctor treat blank keys as unauthenticated, and reject values above `tokio::sync::Semaphore::MAX_PERMITS` before constructing semaphores.

- [ ] **Step 4: Run focused validation tests**

Run: `cargo test -p higgs --lib api_key` and `cargo test -p higgs --lib runtime_concurrent`

Expected: all focused validation tests pass.

### Task 5: Complete configuration and unload documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/configuration.md`
- Modify: `crates/higgs/src/daemon.rs`

- [ ] **Step 1: Update docs**

Document the three runtime controls, their defaults, root semantics, API-key requirement, local/cache-only resolution, and exact `204` versus `202` unload behavior without claiming process-wide MLX cache clearing.

- [ ] **Step 2: Run formatting and documentation-adjacent tests**

Run: `cargo fmt --all -- --check && cargo test -p higgs --tests`

Expected: formatting and all package tests pass.

### Task 6: Final verification

**Files:**
- Verify: all changed files.

- [ ] **Step 1: Run lint and tests**

Run: `cargo clippy -p higgs --all-targets -- -D warnings && cargo test -p higgs --tests`

Expected: exit 0, with 491 library tests, 2 binary tests, and 100 passing integration tests plus the existing 10 ignored tests.

- [ ] **Step 2: Inspect the final diff**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; the pre-existing untracked `qwen38-challenge/` checkout remains untouched.
