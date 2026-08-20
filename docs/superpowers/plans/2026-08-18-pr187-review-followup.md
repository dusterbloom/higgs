# PR 187 Review Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address every still-valid, non-resolved review finding on upstream PR 187 and verify every review thread against the final diff and tests.

**Architecture:** Keep the existing runtime model design intact. Tighten local Higgs routing so an explicit model rewrite never falls through to an arbitrary engine, make runtime-load failures typed instead of string-inspected, and improve unload diagnostics without changing drain ownership. Add focused API error-contract tests and document the runtime configuration safety contract.

**Tech Stack:** Rust, Tokio, Axum, serde, Cargo test, GitHub PR review threads.

## Global Constraints

- Preserve the existing API behavior and status codes except where review feedback identifies a bug or misleading message.
- Runtime model loading remains opt-in and API-key protected.
- Local runtime paths remain restricted by `local.runtime_model_roots`; Hugging Face resolution remains cache-only.
- Do not resolve or reply to GitHub threads during implementation; verification is read-only unless explicitly requested later.
- Follow TDD for behavior changes: add a focused failing test, observe the expected failure, implement the minimal fix, then rerun the test.

---

### Task 1: Fix local routing fallback and typed runtime-load errors

**Owner:** Terra

**Files:**
- Modify: `crates/higgs/src/router.rs`
- Modify: `crates/higgs/src/routes/models.rs`
- Test: Rust unit tests in the same two modules

**Interfaces:**
- `Router::resolve` must return the existing missing-model error when `model == "auto"` but `model_rewrite` names an unloaded model.
- `Router::acquire_runtime_load` must expose a typed budget-rejection case so the HTTP handler maps budget exhaustion to `400` without inspecting error-message text.

- [ ] **Step 1: Add a failing routing regression test.** Configure a Higgs route with `model = "missing"`, load a different stub engine, resolve `"auto"`, and assert the result is an error mentioning `missing` rather than a successful route to the stub engine.
- [ ] **Step 2: Run the focused router test and confirm it fails because the explicit rewrite falls through to the arbitrary-engine fallback.**
- [ ] **Step 3: Restrict the Higgs `auto` fallback to `model_rewrite.is_none()`.** Keep the virtual-`auto` fallback for the unrewritten default route and preserve all direct-model behavior.
- [ ] **Step 4: Add or update a failing runtime-load mapping test that distinguishes resident-budget exhaustion from other gate failures.**
- [ ] **Step 5: Replace stringly-typed `acquire_runtime_load` errors with a small internal error type (budget reached versus gate closed), and map those variants explicitly in `load_model` while preserving the existing response messages and status codes.**
- [ ] **Step 6: Tighten `startup_engines_do_not_consume_runtime_model_budget` to assert the exact expected resolver failure, so a future budget regression cannot satisfy the test accidentally.
- [ ] **Step 7: Add periodic warning logging to `drain_in_background` at a fixed interval, including elapsed drain time and `Arc::strong_count`, without changing polling, permit ownership, or cleanup.
- [ ] **Step 8: Fix the unload-disabled `403` message to say runtime model unloading is disabled.
- [ ] **Step 9: Restrict `remove_engine` so callers cannot accidentally use the non-draining removal path for runtime-loaded engines; keep the runtime endpoint on `remove_runtime_engine` and preserve the existing unit coverage.
- [ ] **Step 10: Run the focused router/models tests, then the relevant `cargo fmt --check` and package test targets.

---

### Task 2: Cover error contracts and document runtime configuration semantics

**Owner:** Sol

**Files:**
- Modify: `crates/higgs/src/error.rs`
- Modify: `docs/configuration.md`
- Modify: `crates/higgs/src/doctor.rs`
- Modify: `crates/higgs/src/model_resolver.rs`

**Interfaces:**
- `ServerError::Conflict` must serialize as HTTP `409`, type `conflict`, and the supplied message.
- `ServerError::Forbidden` must serialize as HTTP `403`, type `forbidden`, and the supplied message.
- The generated configuration example must explain the enable/auth gate, path-root policy, startup-model budget exclusion, and concurrent-load limit.

- [ ] **Step 1: Add two focused failing response tests for `Conflict` and `Forbidden`, asserting status, `error.type`, message, and null code.
- [ ] **Step 2: Run only those tests and confirm they fail because the new variants have no direct contract coverage.
- [ ] **Step 3: Keep the existing `IntoResponse` mapping if it already satisfies the assertions; make only the minimal production change needed.
- [ ] **Step 4: Expand the `[local]` comments in `docs/configuration.md` to state that runtime load/unload requires `allow_runtime_model_load = true` and a non-empty `server.api_key`, that empty roots allow only cached Hugging Face IDs, that configured roots constrain local paths, that startup models do not count toward `runtime_max_loaded_models`, and that `runtime_max_concurrent_loads` limits simultaneous runtime attempts.
- [ ] **Step 5: Add doctor validation for every configured `local.runtime_model_roots` entry, with a focused test for an unresolvable root; preserve the existing hard failure for runtime loading without a non-empty API key.
- [ ] **Step 6: Replace the test-only `runtime_load_path_allowed` policy helper with tests that call the production `resolve_runtime_model_with_cache(..., None)` path, asserting allowed HF-shaped IDs and rejection of untrusted local paths/roots using the production error messages.
- [ ] **Step 7: Run `cargo fmt --check` and the focused doctor/model-resolver/error tests, then inspect the documentation diff for the requested configuration semantics.

---

### Task 3: Whole-PR review-thread verification

**Owner:** Main agent

**Files:**
- Read-only: GitHub PR 187 review metadata, reviews, comments, and inline threads
- Read-only: final working-tree diff and test output

- [ ] **Step 1: Inspect both agents’ diffs and confirm they changed only their assigned files.
- [ ] **Step 2: Run `cargo fmt --check`, `cargo clippy -p higgs -- -W clippy::nursery`, and `cargo test -p higgs -- --test-threads=1`.
- [ ] **Step 3: Re-fetch PR 187’s review threads and classify each thread as addressed, already addressed before this turn, or still open with a technical reason.
- [ ] **Step 4: Verify the historical maintainer security comments against current code: doctor hard-fails missing/blank API keys, runtime paths are allowlisted/cache-only, runtime load counts are bounded, and docs match cache-only load/unload behavior.
- [ ] **Step 5: Report the final thread-by-thread status, test evidence, merge-conflict/CLA status, and any feedback that remains intentionally unaddressed.

---

### Task 4: Final review fixes

**Owners:** Terra and Sol

**Files:**
- Terra modifies: `crates/higgs/src/router.rs`, `crates/higgs/src/routes/models.rs`
- Sol modifies: `crates/higgs/src/config.rs`, `crates/higgs/src/model_resolver.rs`

- [ ] **Step 1: Terra adds a production-flow regression test showing that an auto-router classification which selects a missing explicit rewrite returns an error instead of being swallowed and falling through to the default Higgs route; propagate selected-target errors from `try_auto_route` while preserving “no classification” fallback behavior.
- [ ] **Step 2: Sol adds normal config validation requiring a non-blank `server.api_key` whenever runtime model load/unload is enabled, with a failing config test first; preserve existing API-key character validation.
- [ ] **Step 3: Sol hardens runtime-root validation to reject blank entries and canonical paths that are not directories, with focused tests for both cases.
- [ ] **Step 4: Terra updates affected runtime-model test fixtures to provide an API key where runtime loading is intentionally enabled, and adds an endpoint/config regression test for the unauthenticated configuration if needed.
- [ ] **Step 5: Run focused red-green tests, `cargo fmt --check`, Clippy, and the full `cargo test -p higgs -- --test-threads=1` suite.
