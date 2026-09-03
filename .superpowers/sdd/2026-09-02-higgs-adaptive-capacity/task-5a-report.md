# Task 5A report: bind model lifecycle

## Controller review correction round 3/5

- [x] Make measured allocator decrease the sole authority for zero-capacity
  recovery.
- [x] Preserve zero across unpublished-registration rollback and unregister
  without a qualifying measurement.
- [x] Reject equal, increased, stale, cross-boot, non-adjacent, and saturated
  measurement epochs as recovery evidence.
- [x] Re-run independent review, focused lifecycle tests, and the full serialized
  release matrix.

### Round 3 implementation evidence

- `PublishedMemoryMeasurement` now carries its private boot identity, adjacent
  previous/current measurement epochs, and exact previous/current MLX active
  bytes. Bounded recovery requires the same boot, an exact `+1` epoch, the
  registry's current epoch and active-byte value, and a strict active-byte
  decrease. Epoch saturation fails closed.
- Dropping an unpublished `ActiveRegistration` only removes provisional
  metadata and recomputes with `Preserve`; it cannot treat model removal as
  proof that engine-owned allocations were released.
- `finish_unregister` preserves zero for no measurement and for equal,
  increased, or stale measurements. It permits the bounded minimum only when
  the supplied token is the exact current decreased measurement produced after
  engine shutdown and allocator-cache cleanup. Critical pressure remains
  unavailable regardless of decrease evidence.

### Round 3 TDD and mutation evidence

- The provisional-cancellation regression failed with 1,024 tokens restored
  immediately after registration rollback, before engine cleanup; it now stays
  zero until a later authoritative decrease is published.
- The unregister regression failed with 1,024 tokens restored by
  `finish_unregister(None)` and now preserves zero for absent/equal/increased
  evidence.
- A stale decreased token initially restored 1,024 after a newer measurement;
  exact-current epoch matching now keeps it at zero.
- The large-envelope lifecycle case starts above 8,192 tokens, drives the
  survivor to zero, then supplies a current 20 GiB to 5 GiB decrease after
  drain. It recovers to exactly 8,192 tokens, proving bounded recovery rather
  than a full-capacity jump.

## Controller review correction round 2/5

- [x] Make active capacity registration and route visibility one externally
  atomic publication.
- [x] Recover a zero envelope only from measured residency cleanup, with the
  bounded minimum and existing hysteresis.
- [x] Add normalized MLX allocator policy to the canonical runtime identity by
  sharing the exact resolver used by runtime setup.
- [x] Complete independent re-review, full release gates, and final GitNexus
  staged/compare verification.

### Round 2 implementation evidence

- Runtime insertion first installs a disabled route. The active-registration
  nonce, acknowledged cache-policy revision, capacity publication, generation
  bump, and non-awaiting route-ready flip are then committed while the registry
  transaction is held. Before that commit, capacity GET reports unavailable,
  chat resolution and model listing omit the route, and DELETE reports not
  found. Cancellation drops both rollback owners, removes the disabled route,
  joins and drops the engine, clears allocator cache, remeasures, and leaves no
  registration or drain conflict. After commit, concurrent DELETE can drain and
  durably clean the published engine while the load future is paused at the
  deterministic post-insertion gate.
- Shared-residency replacement distinguishes pressure/cache recomputation from
  genuine cleanup. A measured `active_bytes` decrease or fixed model-residency
  removal may seed at most 8,192 tokens from zero; all other recomputations
  preserve the current zero bound. Further headroom improvement remains subject
  to the controller's recovery hysteresis.
- The canonical engine runtime identity now serializes the resolved allocator
  policy as `disabled`, `legacy`, or `wired_limit`. The same pure resolver drives
  `set_wired_limit_to_max`: presence of `HIGGS_NO_MEM_LIMIT` disables limits;
  `legacy`, `safe`, and `caps` select legacy caps; all other wired-mode values
  select the default wired limit.

### Round 2 TDD and mutation evidence

- The atomic-publication regression initially observed capacity as
  `unavailable` after the route was already usable. It now gates immediately
  after successful insertion and proves capacity GET, chat resolution, list,
  and concurrent DELETE all agree with the single committed state.
- The cancellation regression now gates immediately after disabled insertion.
  Its readiness mutation exposed a route through direct/default resolution;
  filtering both lookup paths on `capacity_ready` makes the exact test pass.
- The rejected co-load cleanup regression initially remained unavailable after
  memory fell from 20 GiB to 5 GiB. It now recovers to exactly the bounded 8,192
  token seed and does not jump on a subsequent 4 GiB measurement.
- Independent review then found an over-broad recovery mutation: a 1,024-token
  constrained envelope fell to zero and the same shared-ledger pass restored it
  to 1,024. The exact RED reproduced that value; cause-scoped recovery now
  preserves zero while the cleanup regression remains green.
- Adding either allocator environment switch initially left the runtime
  identity unchanged. Mutation coverage now proves each resolved policy change
  changes identity while equivalent spellings normalize to one key.

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

- Focused registry correction suite: 37 passed, 0 failed.
- Focused Qwen identity correction: 1 passed, 0 failed.
- Focused route lifecycle correction suite: 21 passed, 0 failed.
- `higgs-models` release library: 753 passed, 0 failed, 46 ignored.
- `higgs-engine` release library: 596 passed, 0 failed, 5 ignored.
- Runtime-identity integration: 3 passed, 0 failed.
- Higgs release library: 723 passed, 0 failed.
- Higgs release integration: 107 passed, 0 failed, 10 ignored.
- Higgs release binary: 4 passed, 0 failed.
- Higgs release build: passed.
- GitNexus staged detection: 2 files, 22 changed symbols, 53
  affected processes, CRITICAL as expected for the lifecycle/router surface.
- GitNexus compare-to-`main` detection: 236 files, 4,631 changed symbols, 862
  affected processes, CRITICAL across the complete multi-task branch; the
  Task 5A staged set remains the bounded 2-file subset above.

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
