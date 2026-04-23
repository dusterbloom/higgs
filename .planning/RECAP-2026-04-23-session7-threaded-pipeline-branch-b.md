# Session 7 recap — Threaded spec-decode (Stage 1) + Branch B ANE wiring

## TL;DR

- **Landed Stage 1 of the pipelined spec-decode**: scoped-thread drafter + mpsc
  protocol, correctness-equivalent to serial, gated by
  `HIGGS_SPEC_DECODE_PIPELINE=1`. Zero tok/s change on its own — scaffolding
  for when drafter runs on ANE.
- **Landed Branch B**: `QwenNextCausalDrafter::from_dir` now attaches ANE GDN
  via the worker path when `HIGGS_DRAFTER_ANE_GDN=1`. Compiles, attaches, but
  this is the SLOW ANE path (3.2ms GPU sync per dispatch). Real win requires
  switching the drafter to the inline IOSurface path — work for next session.
- **Re-assessed option 3 ("batch mpsc messages")**: doesn't help. Dispatches
  within a forward pass are sequentially dependent via intervening GPU
  compute. The real fix is the inline path, which already exists in-tree for
  the verifier.

## Starting state

- Branch: `feat/magic-canvas`, head `96d3ee20` (session 6 recap).
- Baseline from session 6: coding 49% accept, 7.3 tok/s algo, 3.5 tok/s wall.

## Investigation outputs

### Q1-Q4 feasibility (first Explore agent)

- Both drafter and verifier are `qwen3_5` hybrid arch (GDN + full-attention),
  24 and 64 layers respectively.
- `QwenNextCausalDrafter::from_dir` at `diffusion.rs:4114` calls
  `load_qwen3_5_model` DIRECTLY, bypassing `load_model`'s
  `maybe_enable_ane_gdn`. So prior to this session, drafter never got ANE
  attachment.
- ANE worker is spawned per model (no singleton collision).
- Bucket seq_len=32 fits drafter K∈[1..16] via zero-padding.
- Verdict at start of session: Branch B is a 15-30 LOC loader change.

### ANE worker root cause (second Explore agent)

- `qwen3_next_ane_worker.rs:297` — `x_f32.eval()` forces GPU→CPU
  materialization per dispatch, ~3.2ms/call. NOT mpsc, NOT marshaling.
- Current dispatch topology: 24 GDN layers × 2 calls (qkvz+ba fused + out_proj)
  = 48 dispatches × 3.2ms ≈ 150ms/forward ANE-only overhead.
- Agent's "batch 2-3 dispatches in one mpsc msg" suggestion is wrong: qkvz+ba
  output feeds GPU SSM compute which produces out_proj's input — they cannot
  be batched without re-architecting the forward.
- `dispatch_fused` already fuses qkvz+ba; no further in-layer batching available.

## Changes this session

### New: `crates/higgs-models/src/speculative_threaded.rs` (~340 LOC)

- `DraftReq::{Prefill, AdvanceAndDraft}` / `DraftResp::{Drafts, Err}` protocol.
- `drafter_thread_loop`: owns `d_cache`, `saved_draft_logits`, `d_snapshot`;
  handles fast-path (full accept) vs slow-path (partial accept) cache advance.
- `speculative_generate_next_threaded`: `std::thread::scope`-based; drafter
  borrows `&mut QwenNextCausalDrafter` for the call duration.

### Modified: `crates/higgs-models/src/diffusion.rs:speculative_generate_next`

- Dispatch: if `HIGGS_SPEC_DECODE_PIPELINE=1`, delegate to threaded variant.
- Default behavior unchanged.

### Modified: `crates/higgs-models/src/diffusion.rs:QwenNextCausalDrafter::from_dir`

- New env var `HIGGS_DRAFTER_ANE_GDN=1`: calls
  `model.enable_ane_gdn_all_layers_via_worker(32)` (worker path).
- Known limitation: worker path is the slow one. Next session should switch
  to inline path (see below).

### Modified: `crates/higgs-models/src/lib.rs`

- Registered `pub mod speculative_threaded;`.

## Validation

- `HIGGS_SPEC_DECODE_PIPELINE=1 cargo test ... test_speculative_generate_next_natural_prompts`:
  coding 49.0% accept, **7.2 tok/s algo** (matches 7.3 baseline), QA 12.4% accept,
  2.8 tok/s algo. Correctness preserved.
- Combined `HIGGS_SPEC_DECODE_PIPELINE=1 HIGGS_DRAFTER_ANE_GDN=1` +
  `--features ane`: **OOM/SIGKILL** during verifier prefill. Both 14GB of
  model weights + ANE kernel compile memory exceeded limit on this box. ANE
  compile DID fire (log confirmed) before kill.

## Known issues surfaced

- `WARN: enable_ane_gdn_all_layers_via_worker: expected compile_count delta=3,
  got 2` at `qwen3_next.rs:4192`. Pre-existing off-by-one in worker sanity
  check — `dispatch_fused` reduced donors from 3 to 2 but the assert wasn't
  updated. Not a functional bug.

## Next session — inline IOSurface drafter

The real option 3 win. The inline path already exists and works for the
verifier via `batch_engine::spawn` thread, which calls
`qwen.finalize_ane_gdn_inline(weights, seq_len)` at `batch_engine.rs:127` on
the inference thread. IOSurface-bound kernels, zero-copy GPU↔ANE, no 3.2ms
sync.

### Concrete plan (~80-120 LOC)

1. Change `QwenNextCausalDrafter::from_dir` to return
   `(Self, Option<PendingAneGdn>)`. Gate via `HIGGS_DRAFTER_ANE_GDN_INLINE=1`
   (or repurpose `HIGGS_DRAFTER_ANE_GDN=1` with worker as opt-out fallback).
2. In `speculative_threaded::drafter_thread_loop`, before the `rx.recv()`
   loop: if `pending` is present, call
   `drafter.model.finalize_ane_gdn_inline(weights, seq_len)`. This binds
   IOSurfaces on the drafter thread — exactly where dispatches happen.
3. For the serial path (`HIGGS_SPEC_DECODE_PIPELINE=0`), caller finalizes
   immediately after `from_dir` on the current thread.
4. Update callers of `from_dir`:
   - Test at `diffusion.rs:8317` — receive + finalize.
   - `crates/higgs-engine/src/simple.rs` — scan usage, update.

### Acceptance criteria

1. Combined run (`HIGGS_SPEC_DECODE_PIPELINE=1 HIGGS_DRAFTER_ANE_GDN_INLINE=1`)
   completes without OOM on a box with ≥32GB unified memory.
2. Coding prompt hits ≥ 10 tok/s algo (goal 14 tok/s, floor at threaded-serial
   baseline of 7.2).
3. No accept-rate regression on synthetic or natural prompts.
4. Trace confirms drafter dispatches routed to ANE (existing trace machinery).

### Stage 2 speculation — defer

Once inline path works and we see partial overlap: add speculative ahead-run
where drafter produces round N+1 drafts assuming round N fully accepts, with
rollback on partial. Estimated +20% on top of inline win.

## Commands

```bash
cd /Users/peppi/Dev/higgs
git status                     # confirm session 7 changes uncommitted
git log --oneline -5

# Re-confirm Stage 1 correctness on fresh context:
HIGGS_SPEC_DECODE_PIPELINE=1 cargo test -p higgs-models --release \
  test_speculative_generate_next_natural_prompts \
  -- --ignored --nocapture --test-threads=1 2>&1 | tail -40

# Next session's first experiment (after inline wiring):
HIGGS_SPEC_DECODE_PIPELINE=1 HIGGS_DRAFTER_ANE_GDN_INLINE=1 \
  cargo test -p higgs-models --release --features ane \
  test_speculative_generate_next_natural_prompts \
  -- --ignored --nocapture --test-threads=1 2>&1 | tail -60
```

## Files touched (uncommitted)

- `crates/higgs-models/src/speculative_threaded.rs` (new)
- `crates/higgs-models/src/lib.rs` (+1 line)
- `crates/higgs-models/src/diffusion.rs` (env-var dispatcher + Branch B wiring)

Plus session 5+6 uncommitted work still in the tree (persistent caches, natural
prompt test). Commit all of session 5+6+7 together, or split as three commits
before next session.

## Wins-banked / can't-fight reference

| Fight | Verdict | Location |
|---|---|---|
| Drafter ANE plumbing attached | DONE this session (worker path) | `diffusion.rs:4110` |
| Threaded drafter scaffolding | DONE this session | `speculative_threaded.rs` |
| Inline IOSurface ANE (zero-copy) | Next session — pattern already in `batch_engine.rs:127` | see above |
| Verifier 560ms floor | Phase 2 verify_build surgery (battle plan) | out of scope this session |
| QA accept rate | Can't win without bigger drafter | accepted |
| Cold-start JIT (36s first prompt) | Amortized across calls — don't fight | accepted |
