### Task 5A: Bind one process-wide capacity registry to model/server lifecycle and expose `/v1/capacity`

**Owner:** deep worker. This is a planned CRITICAL edit. Own a narrow registry implementation (`crates/higgs/src/capacity/registry.rs` if justified), `crates/higgs/src/capacity.rs`, `crates/higgs/src/capacity/pressure.rs` only for a registry pressure-sink seam, `crates/higgs/src/state.rs`, `crates/higgs/src/main.rs`, `crates/higgs/src/lib.rs`, `crates/higgs/src/routes/mod.rs`, `crates/higgs/src/routes/models.rs`, new `crates/higgs/src/routes/capacity.rs`, `crates/higgs/Cargo.toml` plus `Cargo.lock` for the direct existing `sha2 = "0.11"` dependency, exact AppState fixture updates, focused/integration tests, and this report. Do not edit root `Cargo.toml`, chat/completions/Anthropic generation hot paths, or loader internals in 5A. Preserve unrelated `AGENTS.md`/`CLAUDE.md`.

**Risk disclosed:** refreshed GitNexus rates `build_engine` CRITICAL (75 impacted / 41 direct / 7 processes), `AppState` CRITICAL (76 / 3 direct / 43 processes), `build_router` CRITICAL (124 / 90 direct / 7 processes), and `cmd_serve`, boot/runtime load/unload, router insert/remove/from_config CRITICAL. Re-run exact impact before every existing symbol edit and keep the diff constrained. Task 5B separately owns loader-boundary callbacks and bounded allocation aborts; do not claim those here.

- [x] **Step 1: Add failing registry/lifecycle/route tests first.**

  Cover: a single registry authority with immutable known-model catalog and active records; one random process boot ID shared across model snapshots; monotonic per-model generation; conservative snapshot immediately after successful registration; exact content fingerprint stability and one-byte/path mutation invalidation; process-wide shared-residency ledger preventing models from independently spending the same headroom; boot load and runtime load register before route visibility; unload enters draining/unavailable without deleting identity and final unregister occurs only through a Task6-compatible drain seam; known unloaded returns HTTP 200 unavailable with all token/cache fields zero; unknown returns exact typed 404; route returns the stored snapshot unchanged; authenticated success and unauthenticated rejection use the existing API auth layer; observer starts exactly once and is explicitly stopped/joined after graceful shutdown.

  Include rollback tests: failed registration does not expose a route; failed router insertion rolls back active capacity; failed observer startup leaves no live handle; shutdown still joins observer when server returns an error. AppState test construction must have one helper rather than duplicating registry setup across every literal.

- [x] **Step 2: Add one process-wide `CapacityRegistry` authority.**

  The registry owns the process boot ID, known-model catalog, active per-model records/controllers, pressure state, generation counters, and one shared allocation/residency ledger. Individual model cost solvers may remain internal, but they cannot independently claim the same process headroom. Make snapshots atomic/consistent under one narrow synchronization boundary; do not hold locks across I/O or await. Expose lifecycle operations such as known/unavailable, begin registration, commit active, begin drain, and finish unregister. Task 6 must be able to delay final unregister until reservations drain.

  Adapt Task 4 through a narrow pressure-sink trait/callback if necessary so the one observer feeds the registry rather than a throwaway single-model controller. Do not start a second observer or one observer per model.

- [x] **Step 3: Build exact model identity without leaking loader internals.**

  Compute a deterministic exact content fingerprint over the sorted relevant model artifact set: domain/version tag plus normalized relative path, length, and streamed bytes. Fail closed on unreadable files, overflow, path escape, or directory symlink; count regular file symlinks according to the already-reviewed Task 2 policy without following directory symlinks. Include model/quantization/execution/KV/drafter identity in the controller profile key. Do not reuse the disk-prefix cache’s heuristic name/size/mtime identity.

- [x] **Step 4: Attach registry to startup and model lifecycle.**

  Add `Arc<CapacityRegistry>` to `AppState` exactly once. Preserve resolved model facts through boot loading and register after a load succeeds but before exposing the engine through the router. Runtime load follows the same transactional order. Unload first marks the model draining/unavailable, then removes route visibility; retain the record until Task 6’s drain callback finalizes it. Known configured-but-unloaded models remain in the catalog and publish zero fields.

  This task may perform pre-load definite-too-large rejection using Task 2's strict artifact byte count plus the existing coarse loader workspace bound, and post-load minimum-working-request rejection using measured facts. It must release/rollback on failure. Per-shard/conversion pressure checkpoints belong to 5B.

- [x] **Step 5: Start/stop the one pressure observer with the server.**

  Construct the registry before `AppState`, start Task 4's observer once, retain its handle across `axum::serve`, and ensure every graceful/error shutdown path consumes `stop().await` before PID cleanup/process exit. No detached task and no sync-Drop-only happy path. The observer pressure sink updates all active controllers plus registry admission state without holding a registry lock across await.

- [x] **Step 6: Expose authenticated `/v1/capacity`.**

  Add the route to the existing authenticated `api_routes` subtree. Require a nonblank query `model`; unknown is exact typed model-not-found 404; known unloaded/draining is HTTP 200 unavailable with all token/cache values zero; active returns the registry's already-computed `CapacitySnapshot` without route-side token math. Preserve exact Task 1 schema and no body/content-type generic 404 behavior when the route is absent on old servers.

- [x] **Step 7: Validate serialized release gates.**

  Run package-local formatting, focused registry/lifecycle/capacity-route/observer tests, `cargo test --release -p higgs --lib`, relevant integration API contract tests, `cargo build --release -p higgs`, and `git diff --check`. Use only fake memory/model fixtures where possible; do not load the 35B artifact or induce pressure in unit gates.

- [x] **Step 8: Detect scope, report, and commit.**

  Run staged/task and compare-to-main GitNexus `detect_changes`, explicitly assess every CRITICAL flow. Write `.superpowers/sdd/2026-09-02-higgs-adaptive-capacity/task-5a-report.md` with RED/GREEN/mutation evidence, lifecycle ordering, rollback evidence, and the explicit 5B boundary. Stage only owned files. Commit `feat(capacity): bind model lifecycle` and return exact SHA.
