# Task 5A report: bind model lifecycle

## Outcome

Higgs now owns one process-wide `CapacityRegistry` for every configured and
dynamically loaded local model. Startup and runtime loading publish capacity
only after successful measured registration; unload first marks the model
draining and removes route visibility, then joins the worker, clears allocator
cache, remeasures, and atomically finishes unregistering. One server-owned
pressure observer updates the registry and effective cache policy and is
explicitly stopped on every server exit.

`GET /v1/capacity?model=...` is under the existing authenticated API subtree.
It returns the stored snapshot without route-side capacity math, returns HTTP
200 with zero token/cache fields for known unloaded or draining models, and
returns the typed `higgs_capacity_model_not_found` 404 for unknown models.

## Impact and scope

Pre-edit GitNexus analysis rated the lifecycle surface CRITICAL:

- `build_engine`: 75 impacted, 41 direct, 7 processes.
- `AppState`: 76 impacted, 3 direct, 43 processes.
- `build_router`: 124 impacted, 90 direct, 7 processes.
- Boot/runtime model load, unload, router insert/remove/from-config, and server
  shutdown were treated as CRITICAL.
- `BatchEngine::load` was HIGH (5 upstream, 1 direct, 4 modules).
- `worker_loop` was LOW (3 upstream, 1 direct, 2 modules), with the index
  warning that the result was a lower bound.
- Newly added or changed unindexed registry/profile/cache helpers returned
  UNKNOWN and were conservatively treated as CRITICAL.

The implementation remained within the amended Task 5A ownership: Higgs
lifecycle/registry/routes, narrow engine worker-join and cache-policy seams,
focused fixtures/tests, `sha2 = "0.11"`, and this task record. Unrelated
`AGENTS.md` and `CLAUDE.md` changes were preserved and excluded.

## Lifecycle and rollback evidence

- Registration tickets and drain tokens use private immutable UUID nonces;
  capacity generations cannot strand or authorize lifecycle completion.
- Load measurement is serialized against inference and concurrent loads.
- A post-load rejection or insertion failure drops/joins the engine, invokes
  the shared allocator-cache cleanup, remeasures, and rolls back registration.
- `BatchEngine` owns its worker `JoinHandle`; acknowledged shutdown guarantees
  worker-owned models are destroyed before cleanup measurement.
- The Batch worker acquires the MLX process gate per work/control unit, so an
  idle Batch engine cannot deadlock co-resident Simple inference or loading.
- A draining model's exact pre-drain cache allocation is frozen across every
  pressure/registration recomputation. It is subtracted before fair-sharing
  the remainder and released only by drain cancellation or atomic
  `finish_unregister(final_snapshot)` after join/drop/cleanup.
- Router cache updates use one coherent registry allocation snapshot. An
  unchanged cap is a true no-op, including no PagedPrefixCache revision
  invalidation.
- Existing zero config semantics are preserved: zero cache ceilings mean
  automatic/no explicit ceiling, and zero retained-token maximum publishes the
  effective prompt maximum. Zero does not disable a model or other models.

## Identity and learned profiles

- Model identity hashes a versioned, sorted allowlist of authoritative weights,
  configuration, tokenizer/template, and drafter artifacts by normalized path,
  length, and streamed content. Runtime caches, logs, and unrelated files are
  excluded; directory symlinks and unreadable/empty roots fail closed.
- Learned profile storage is explicit and config-adjacent, uses atomic writes,
  and persists evidence only. Live pressure, capacity, generations,
  reservations, and boot IDs are never persisted.
- Profile matching includes exact platform/OS/backend authority, Higgs schema
  and executable plus adjacent `mlx.metallib` content, model/drafter/prefill
  identity, actual quantization/native-affine/KV mode, cache configuration, and
  resolved memory-relevant execution facts.
- PFlash full-score/free-memory values and DFlash block/min-block/dSpark cap
  values are resolved by the same five pure engine functions used at runtime;
  equal profile keys therefore cannot hide whitespace/default parser drift.

## TDD and mutation evidence

- Digest formatting initially failed to compile until exact SHA-256 encoding
  was implemented.
- Lifecycle nonce regressions failed when generation changes invalidated drain
  and publication rollback; immutable nonces made them pass.
- The Batch shutdown regression failed to compile before the acknowledged join
  API existed.
- The final drain regression failed with the allocation changing from exactly
  6 GiB + 6 GiB to 6,335,076,762 + 6,335,076,762 bytes after pressure. The
  frozen-allocation implementation makes the same test pass.
- Exact runtime-parser tests failed to compile before shared resolvers existed.
  After implementation, deliberately adding `.trim()` changed the whitespace
  result from the required default 8192 to 4096 and failed the test; restoring
  exact untrimmed parsing returned it to green.
- A prior mutation that removed the live zero-cap cache branch failed at the
  cache-emptiness assertion and passed after restoration.
- The prepared DFlash pair regression proves an unchanged PagedPrefixCache cap
  preserves its publication revision and live pair.

Three independent review passes were completed. The final re-review was CLEAN
after verifying immutable drain allocations and shared engine/profile parsers.

## Release verification

- Focused registry: 23 passed, 0 failed.
- Focused profile/parser regressions: 2 passed, 0 failed.
- `cargo test --release -p higgs-engine --lib -- --test-threads=1`:
  590 passed, 0 failed, 5 ignored.
- `cargo test --release -p higgs --lib -- --test-threads=1`:
  699 passed, 0 failed.
- `cargo test --release -p higgs --test integration_tests -- --test-threads=1`:
  107 passed, 0 failed, 10 ignored.
- `cargo build --release -p higgs`: passed.
- `git diff --check`: passed before staging.
- GitNexus staged detection: 23 files, 131 changed symbols, 81 affected
  execution flows, CRITICAL as expected for the authorized lifecycle surface.
- GitNexus compare-to-main detection: 234 files, 4,202 symbols, 296 flows,
  CRITICAL; this intentionally includes the preceding capacity-plan tasks on
  the feature branch rather than only Task 5A.

The build emitted pre-existing `higgs-models` unused-code warnings and reported
that `mlx.metallib` was absent from the local MLX build output. Production
runtime setup requires/restores that artifact; profile identity fails
conservatively when it is unavailable.

## Explicit Task 5B boundary

Task 5A intentionally contains no pre-load artifact-size admission claim and no
per-shard or per-conversion loader checkpoint. It performs the post-load
minimum-working-request check using measured facts. Task 5B owns the strict
weight scanner, architecture-specific workspace kind, optional sidecar policy,
loader-boundary abort callbacks, and partial-load cleanup instrumentation.
