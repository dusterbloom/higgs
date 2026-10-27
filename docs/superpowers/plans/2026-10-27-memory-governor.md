# Memory governor: never abort on GPU OOM

Status: **DEFERRED** after codex review (see Review outcome at the end). Date: 2026-10-27.

## Problem

A Metal out-of-memory is fatal to the whole process.

- MLX reports it from the command-buffer completion handler
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`), on a dispatch thread, as an
  uncaught C++ exception. The result is `abort()`.
- One request therefore takes down every client, every retained session and
  the radix cache.

The memory leak (radix views pinning prefill buffers) is fixed. The failure
mode is not. Any request whose peak exceeds the GPU working set still kills
the server.

Facts this design rests on (verified):

- **The MLX memory limit cannot help.** In vendored MLX, `memory_limit` is a
  soft scheduling hint: `eval` waits for in-flight tasks while active memory
  is over it (`mlx/transforms.cpp:243-256`), and `malloc` only trims the buffer
  cache (`backend/metal/allocator.cpp:134-137`). `newBuffer` succeeds, and the
  OOM surfaces later on the GPU.
- **Admission checks tokens only.** It reserves no bytes and never reads MLX
  memory (`crates/higgs/src/capacity/registry.rs:124,154,237-249`).
- **Pressure telemetry is diagnostics only.** It never feeds admission
  (`capacity.rs:20`, `registry.rs:363-370`).
- **The reclaim code is unused.** `reclaim_unleased_retained_for_capacity`
  (`simple.rs:4648`) has no callers, and `StopReason::CriticalPressure` is
  raised only in tests.
- **The prefill progress sink runs after every chunk's `eval`**
  (`higgs-models/src/lib.rs:942-962`, Qwen3Next `qwen3_next.rs:10035`). It can
  already abort cleanly with a typed error (`progress.rs:212-228`, mapped at
  `simple.rs:13850`).
- Measured on the 2026-10-27 replay (Escha 35B-A3B, M4 32 GB): Metal
  recommended working set 25.0 GB, weights plus runtime about 11.8 GB, peak
  16.2 GB after the leak fix, and the per-chunk transient grows with context.

## Goal (user contract)

Higgs serves every request the machine can serve. A request it cannot serve
is refused cleanly, with an actionable non-retryable error, and the server
and every other session stay intact. In order of preference:

1. **Serve it**, reclaiming optional memory first: the MLX buffer pool, radix
   entries, and idle retained sessions.
2. **Degrade optional work.** Skip the radix store or hybrid snapshot for this
   request rather than OOM, because caching is an optimisation.
3. **Refuse up front or between chunks, never mid-kernel.** Use the existing
   413 `compact_and_retry` error, with a message naming the servable budget.
   Agents already handle that by compacting.

## Design

### M1. Budget

`budget = 0.92 × metal_recommended_working_set_bytes()`
(`mlx_tuning.rs:158`).

- MLX itself starts trimming its cache at `0.95 × WS` (its `gc_limit_`), and
  the OOM happened at 0.98 × WS.
- 0.92 leaves about 2 GB on a 25 GB working set for Metal command buffers,
  other processes' wired memory and estimation error.
- `HIGGS_MEMORY_BUDGET_FRACTION` overrides it for experiments. It is not a
  config field.

### M2. Chunk-boundary guard (the core)

A small `MemoryGovernor` owned by `SimpleEngine` is consulted from the prefill
progress sink after each chunk evaluates. It works from measured values:

- It reads `active` (accurate post-eval) and `peak` (monotonic since the
  `RequestMemorySampler` phase start; **it never resets the peak**).
- Chunk transient: `t_k = max(0, peak_now - active_before_chunk_k)`. This
  overestimates when an earlier chunk had the bigger peak, which is the safe
  direction.
- The projected next peak scales with context, because attention scores grow
  with keys:
  `projected = active + t_k × (ctx_{k+1} / ctx_k)`.
- If `projected > budget`, it reclaims (M3), re-measures, and re-projects.
  - If it fits: continue.
  - If not: return `Err(MemoryExhausted { needed, budget, prompt_tokens })`.
    The prefill aborts before the next chunk allocates.
- The governor also keeps a per-model EMA of
  `t_k / (chunk_tokens × ctx_tokens)` for M4.

### M3. Reclaim ladder (cheapest first; stop when it fits)

All of these run under the exec gate and take locks in gate → mutex order,
the same order as store.

1. `mlx_clear_cache` (the buffer pool, at most 256 MiB here). Free.
2. Radix: evict LRU endpoints until enough bytes are released. The current
   request's restored prefix already lives in its own live cache, so this is
   safe.
3. Idle, unleased retained sessions: `reclaim_unleased_retained_for_capacity`.
   Leased or active sessions are never touched.

Each step logs at warn with the bytes freed, and increments
`capacity.reclaims` / `capacity.memory_refusals` counters in `/metrics`.

### M4. Pre-flight check (before restore and the first chunk)

Before materialising a radix prefix and running prefill, estimate:

- `restore`: the matched prefix's KV bytes, known exactly from the radix
  ledger.
- the first chunk's transient: EMA × chunk × ctx, or 0 with no history yet.
  The chunk guard covers from chunk 1 on.
- `snapshot`: the hybrid `deep_clone` of the KV prefix, known exactly from
  the cache's nbytes.

If `active + estimate > budget`: reclaim (M3), then re-check. If it still
doesn't fit, refuse with 413 before any GPU work.

### M5. Optional-work degradation

- The hybrid body snapshot (`simple.rs` two-phase split) is skipped when
  `active + snapshot_bytes > budget`. The request still runs; it just isn't
  cached.
- The radix store (`owned_block_views` tail copies) is skipped when
  `active + tail_bytes > budget`.

### M6. Errors

- New `EngineError::MemoryExhausted { needed_bytes, budget_bytes,
  prompt_tokens }` maps to the existing `CapacityError::Exceeded` → **HTTP 413,
  code `compact_and_retry`**.
- The message reads: "prompt of N tokens needs about X GB; Y GB available
  after reclaim; compact the context".
- It is not retryable as-is, unlike a 503.

### Non-goals (stated assumptions)

- **Decode-time guard.** KV growth per token is small (about 20 KB per token
  here). A follow-up could check every N steps.
- **Batch engine.** Its own prefill loop would get the same sink hook later.
- **Byte-based admission at the HTTP layer.** M4 inside the engine covers it.
- **Calibrating with `RequestAllocationCapture`.** M2's live measurement
  replaces it.

## Tests (minimal, general)

1. **Governor unit test (pure, no MLX).** Feed `(active, peak, ctx)`
   sequences and check four rules:
   - it projects with ctx scaling;
   - it triggers reclaim, and continues when reclaim frees enough;
   - it returns `MemoryExhausted` when reclaim can't free enough;
   - it never decreases the transient estimate within a request.
2. **Engine test with an injected small budget** (a test-only override). A
   long prompt prefill:
   - with radix entries present, reclaims them and completes;
   - with nothing reclaimable, fails with `MemoryExhausted`, and the engine
     then serves a short prompt.
   This proves the engine stays alive.
3. **Route test.** `MemoryExhausted` maps to 413 `compact_and_retry`.
4. **E2E (manual).** Run the replay with
   `HIGGS_MEMORY_BUDGET_FRACTION=0.6` (a budget below the peak): the crash
   request gets 413, the server stays up, and the next request is served. Run
   it again at the default: no refusals, and the same throughput as the leak
   fix.

## Risks / questions for review

1. Is `peak - active_before` a sound per-chunk transient under MLX's async
   eval (the `RequestMemorySampler` phase reset at phase start, prefill
   restart)?
2. Do all prefill paths have a sink? That means hybrid two-phase body/suffix,
   MTP `capture_hidden` single-pass (unchunked!), PFlash, the DFlash paired
   path, and session continuation. An unchunked path has no boundary at all,
   so M4 has to cover it, or it should be forced through chunking.
3. Lock order in M3 from inside the sink, with gate held: prefix-cache mutex
   and retained-map mutex. Is any path holding these while waiting on the
   gate?
4. Is 0.92 × WS right, and should the budget subtract other processes' wired
   memory (the `host_statistics64` free count)?
5. Should 413 be 507 Insufficient Storage instead? 413 keeps the agents'
   existing `compact_and_retry` handling.

## Review outcome (codex gpt-6-sol, 2026-10-27) — deferred

**An in-process governor cannot guarantee "never abort".** A single MLX
`eval` can exceed GPU memory before any check runs: the first chunk with no
history, or any single-pass path. The vendored MLX then throws from the Metal
completion handler (`backend/metal/eval.cpp:18`).

Corrections to the draft:

- **The sink coverage claim is false.** Ordinary non-streaming prefill
  installs no progress sink. Hybrid `capture_hidden` (MTP) forwards are single
  pass. PFlash scoring has no hook. Final progress reports precede the final
  `eval`. DFlash paired prefill has only partial boundaries.
- **`peak − active_before` is not a per-chunk transient.**
  `RequestMemorySampler` resets the peak once per phase, so the value is the
  maximum since phase start.
- **Reclaiming retained sessions from inside the sink deadlocks.**
  Continuations take session → model → gate. Also,
  `reclaim_unleased_retained_for_capacity` returns bytes retained, not bytes
  freed.
- **The restore, snapshot and store byte figures are payload sizes, not copy
  peaks.**
- **A typed 413 needs new plumbing.** The progress sink returns `ModelError`,
  and `CapacityExceededError` has no byte or message fields. Already-started
  SSE streams cannot change status.
- **Decode and BatchEngine must be in scope** for the contract.

Options when resumed:

- **Isolate + guard.** A supervised inference worker process gives the real
  guarantee; it loses that worker's in-memory caches on a crash. Add a cheap
  preflight check and optional-cache shedding to make crashes rare.
- **Guard only.** Best effort.
- **Auto-restart only.** Smallest change.
