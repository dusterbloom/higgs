# Task 5A report: bind model lifecycle

## Controller review correction round 1/5

- [x] Preserve controller pressure and rise hysteresis across shared-ledger updates.
- [x] Serialize revisioned registry cache publication through engine application.
- [x] Support non-macOS observer startup and seed pressure before boot loading.
- [x] Evict optional caches on critical pressure without releasing drain freezes.
- [x] Replace Higgs' partial environment key with one canonical engine/model runtime identity.
- [x] Make cancelled runtime loads own durable join/drop/cleanup/remeasure rollback.
- [x] Replace Batch cache busy-wait with cancellation-safe asynchronous acknowledgement.
- [x] Lower Simple retained limits by evicting unleased entries before leased entries.
- [x] Publish and allocate only cache classes each engine supports.
- [x] Use one exposed-name resolver for catalog, boot, and runtime model paths.
- [x] Re-review all corrections, rerun full release gates, and record final evidence.

### Round 1 implementation evidence

- Shared-ledger replacement now uses a controller-owned bounded recomputation;
  it preserves the current pressure downshift and cannot jump upward during
  normal recovery, registration, memory refresh, or drain completion.
- Cache allocation is a monotonic revisioned plan. One async Router publisher
  serializes plan application, retries a pressure/load race, and acknowledges
  only the exact registry revision applied before route visibility.
- Critical pressure assigns zero to supported optional cache classes while a
  draining model's frozen allocation remains reserved through atomic finish.
- Engine facts declare supported cache classes. Batch receives only a prefix
  allocation and publishes zero retained-session tokens; zero remains
  automatic for every supported class.
- Batch cache control uses bounded async queue send plus Tokio oneshot
  acknowledgement. The current-thread saturated-queue regression proves the
  runtime continues scheduling instead of busy-spinning or blocking.
- Simple retained-cap reductions evict least-recently-used unleased entries
  first, then break only the minimum earliest-expiring leases needed to reach
  the explicit cap. Critical zero policy revokes the optional retained cache.
- The observer starts and seeds before boot loads. Its coordinator applies
  pressure to the registry while no Router exists, then weak-attaches the
  Router and immediately publishes the latest revision. Non-macOS uses an
  owned, stoppable no-op source rather than failing server startup.
- Runtime loads retain their blocking JoinHandle in a cancellation guard. A
  detached cleanup supervisor joins a completed engine, clears allocator
  cache, remeasures, and refreshes the registry; a second provisional guard is
  disarmed only after cache policy, route insertion, and capacity publication.
- One typed engine-owned `ResolvedRuntimeIdentity` replaces Higgs' duplicate
  environment inventory. Qwen profile identity consumes the same path-aware
  resolver as all three Qwen adapter load arms, including wrapper flattening,
  outer quantization, checkpoint layout scans, GDN overrides, and Escha affine
  layout. Its mutation suite covers resolved choices across dense requant,
  Bonsai, compiled GDN/gating, async state eval, QGEMV, dSpark/Q2,
  PFlash/DFlash, Escha, MLA, and TurboQuant.
- Catalog and loaded-model identity use one exposed-name resolver for explicit
  aliases, local paths, configured Hugging Face IDs, and resolved HF snapshots;
  boot registration no longer seeds raw-path aliases.

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
- Load measurement is serialized against inference and concurrent loads and is
  published to a monotonic registry revision before the GPU/load gate is
  released. Out-of-order lifecycle commits solve from the latest published
  allocator state and cannot overwrite it with per-load facts.
- A post-load rejection or insertion failure drops/joins the engine, invokes
  the shared allocator-cache cleanup, remeasures, and rolls back registration.
- `BatchEngine` owns its worker `JoinHandle`; acknowledged shutdown guarantees
  worker-owned models are destroyed before cleanup measurement.
- The Batch worker acquires the MLX process gate per work/control unit, so an
  idle Batch engine cannot deadlock co-resident Simple inference or loading.
- A draining model's exact pre-drain cache allocation is frozen across every
  pressure/registration recomputation. It is subtracted before fair-sharing
  the remainder and released only by drain cancellation or atomic
  `finish_unregister(published_measurement)` after join/drop/cleanup.
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
- Reordered load commits initially replaced a 10 GiB allocator snapshot with a
  stale 5 GiB snapshot; reordered unload completion replaced a later 8 GiB
  snapshot with stale zero. Publishing measurements before gate release and
  removing lifecycle snapshot assignment makes both deterministic regressions
  pass.
- The unsupported-cache mutation published 49,152 retained-session tokens for
  Batch. Capability-gated publication now returns zero while Simple preserves
  zero-as-automatic semantics.
- The Qwen identity mutation resolved nested outer affine Q2 as `disabled`
  instead of `escha_qwen38`; the shared execution resolver makes that test and
  the top-level `qwen3_5_moe` regression pass.
- The shutdown-order regression initially failed to compile before the shared
  stop-then-cleanup seam existed. It now proves that observer cancellation and
  join complete before learned-profile persistence or PID-file removal, on
  both successful and failed server exits.

The final independent controller re-review inspected all eleven corrections,
the subsequent measurement-order, Qwen resolver, cache-capability,
cancellation, and shutdown-order fixes, and reported no remaining findings.

## Release verification

- Focused registry correction suite: 32 passed, 0 failed.
- Focused Qwen identity correction: 1 passed, 0 failed.
- Focused disabled-route cancellation correction: 1 passed, 0 failed.
- `higgs-models` release library: 753 passed, 0 failed, 46 ignored.
- `higgs-engine` release library: 596 passed, 0 failed, 5 ignored.
- Runtime-identity integration: 3 passed, 0 failed.
- Higgs release library: 717 passed, 0 failed.
- Higgs release integration: 107 passed, 0 failed, 10 ignored.
- Higgs release binary: 4 passed, 0 failed.
- Higgs release build: passed.
- GitNexus staged detection: 17 indexed files, 205 changed symbols, 184
  affected processes, CRITICAL as expected for the lifecycle/router surface.
- GitNexus compare-to-`main` detection: 236 files, 4,519 changed symbols, 858
  affected processes, CRITICAL across the complete multi-task branch; the
  Task 5A staged set remains the bounded 17-file indexed subset above.

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
