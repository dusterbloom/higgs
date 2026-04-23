# Session 6 recap (2026-04-23) — persistent drafter cache + natural prompts

## TL;DR

1. **Shipped P0: persistent drafter cache** in `speculative_generate_next`.
   Both drafter and verifier now keep persistent KV/SSM caches across
   rounds with snapshot/clone + fast/slow advance. End-of-round
   `drafter.prefill(&context)` eliminated.
2. **Shipped P2: natural-prompt validation.** New
   `test_speculative_generate_next_natural_prompts` runs a coding prompt
   (49.0% accept, **7.3 tok/s algorithm** / 3.5 tok/s wall) and a QA
   prompt (12.4% accept, 2.7 tok/s). Confirms the recap's hypothesis:
   synthetic "The capital of France is" under-samples acceptance.
3. **Algorithm throughput did NOT materially improve on the baseline
   synthetic prompt.** Session 5 = 6.7 tok/s, Session 6 = 6.7 tok/s (same
   bucket). Wall ticked up 2.5 → 2.8 tok/s (bootstrap variance).
   Per-round drafter-prefill savings (≈-40ms) were absorbed by
   `d_cache.clone()` snapshot cost (≈+40ms) + drafter-advance bucket
   shift. Correctness-neutral.
4. **12 tok/s ceiling remains gated by the same constraints:** 467ms
   verify floor on K=8 warm + ≤ 50% acceptance without a stronger
   drafter. Next unlock is ANE drafter (P1), now with concrete findings.

## Per-round timing comparison

### Synthetic prompt ("The capital of France is", 5 tokens, K=8..16, 60 gen)

| Round/K | Session 5 draft | Session 6 draft | Session 5 verify | Session 6 verify |
|---------|----------------:|----------------:|-----------------:|-----------------:|
| R1 K=16 cold | 267ms | 261ms | 1525ms | 1330ms |
| R2 K=16      | 167ms | 175ms | 558ms  | 564ms  |
| R7 K=8       | 82ms  | 90ms  | 466ms  | 468ms  |
| R8 K=8       | 83ms  | 91ms  | 467ms  | 467ms  |
| R9 K=8 8/8   | 82ms  | 93ms  | 467ms  | 473ms  |
| R11 K=16 8/8 | 86ms  | 90ms  | 467ms  | 468ms  |
| R6 K=16 16/16| 181ms | 179ms | 559ms  | 571ms  |
| R7 K=16 15/16| 184ms | 170ms | 560ms  | 558ms  |
| **Totals** | 1383/6093/1451 | 1427/5932/1644 | (same 41.7% accept, 6.7 tok/s) |

Draft bucket is unchanged; see "Why no throughput gain" below.

### Natural prompts (session 6, new test)

**Coding prompt** `"// Implement a binary tree node in Rust:\n"` (10 tokens):
```
R1 K=16 1/16 draft=278ms verify=1531ms  ← cold
R2 K=16 16/16 draft=156ms verify=558ms  ← FULL ACCEPT
R3 K=16 2/16 draft=159ms verify=558ms
R4 K=16 16/16 draft=168ms verify=562ms  ← FULL ACCEPT
R7 K=8 8/8 draft=80ms verify=470ms      ← FULL ACCEPT
...
Totals: 1171ms draft + 5507ms verify + 1494ms advance = 8172ms
Acceptance: 51/104 (49.0%)
Throughput: 7.3 tok/s (algorithm), 3.5 tok/s (wall)
```
Output: `"//\n// struct Node {\n//     val: i32,\n//     left: Option<Rc<RefCell<Node>>>,\n..."`

**QA prompt** `"Q: Explain how MLX achieves zero-copy on Apple Silicon. A:"` (15 tokens):
```
30+ rounds, mostly 0-2 accepts per round. Rare 4/8 peaks (R14, R18, R27).
Totals: 2842ms draft + 13855ms verify + 5874ms advance = 22571ms
Acceptance: 31/250 (12.4%)
Throughput: 2.7 tok/s (algorithm), 2.6 tok/s (wall)
```
Output: `" MLX leverages Apple's Metal framework and its memory management system to achieve zero-copy..."`

## Why no throughput gain on synthetic prompt

P0's theoretical savings:
- Removed per-round `drafter.prefill(seed)` at start of round: ≈-40ms/round
- Removed per-round `drafter.prefill(&context)` at end of round (unbucketed): ≈-40ms/round

P0's added cost:
- `d_cache.clone()` per round for slow-path snapshot: ≈+40ms/round
- Drafter advance (1-2 token forward) per round: ≈+20ms/round

Net algorithm-bucket delta: ~0. Wall-time savings come from eliminating
the unbucketed end-of-round prefill (~360ms total), but that's masked by
bootstrap variance (14.9s → 12.6s).

**The cheap wins are gone.** Further improvement requires:
1. Snapshot-only-when-needed (can't — we don't know accept count until
   after verify, and cache is mutated by then).
2. Lower-overhead clone (`d_cache.clone()` clones 24 layers × KV/SSM/conv
   arrays; MLX arrays are refcounted so "clone" should be O(layers), not
   O(tokens) — but still visible at ~40ms).
3. ANE drafter (moves the drafter off GPU entirely → parallel with
   verifier → drafter cost disappears from critical path).

## Natural-prompt insight

The 37-point acceptance gap between coding (49%) and QA (12.4%) on the
**same drafter/verifier pair** confirms that per-token predictability
dominates throughput — not algorithm overhead. Full-accept streaks on
the coding prompt (R2, R4, R7: 16/16, 16/16, 8/8) show the algorithm
and caches are working correctly; the 0.8B drafter just can't predict
QA prose reliably.

This means P1 (ANE drafter) has a lower ceiling than the recap assumed
unless paired with a larger drafter. Even with zero drafter latency, QA
throughput capped at 12.4% × K × (1/verify_ms) = 12.4% × 8 × (1/0.47s) ≈
2.1 tok/s at K=8, or 12.4% × 16 × (1/0.56s) ≈ 3.5 tok/s at K=16. A
larger drafter would move both acceptance and latency.

Coding prompt projection with free drafter: 49% × 16 / 0.56s ≈ 14 tok/s.
**This is the natural ceiling to target for P1.**

## Files modified this session

- `crates/higgs-models/src/diffusion.rs`:
  - `QwenNextCausalDrafter::eval_cache_with` L4161: `fn` → `pub(crate) fn`
    (needed to call it from `speculative_generate_next` after inlining
    the prefill logic).
  - Doc comment for `speculative_generate_next` L4185-4192: rewrote to
    describe the persistent-cache design.
  - `speculative_generate_next` bootstrap L4215-4244: removed
    `drafter.prefill(&context)`, replaced with direct
    `drafter.model.forward` + `eval_cache_with`. Shared `prompt_arr`
    with verifier.
  - `speculative_generate_next` draft phase L4252-4271: removed per-round
    `drafter.prefill(&seed)`, replaced with `argmax` of
    `saved_draft_logits` + `d_cache.clone()` snapshot + step loop.
  - `speculative_generate_next` advance L4376-4402: added symmetric
    drafter-cache advance (fast path feeds `[draft[K-1], bonus]`; slow
    path restores snapshot + re-feeds accepted_tokens).
  - Deleted end-of-round `drafter.prefill(&context)` at old L4371-4374.
  - New test `test_speculative_generate_next_natural_prompts` L8292-8370:
    coding + QA prompts via tokenizer.encode.

## Files created this session

- `.planning/RECAP-2026-04-23-session6-persistent-drafter-cache.md` —
  this file.

## P1 — ANE drafter (next session, inherits from session 5 + new findings)

Qwen3Next ANE path exists; concrete locations from this session's
exploration:

1. **Dispatch**: `crates/higgs-models/src/qwen3_next.rs:2806-2846`
   (qkvz + ba projections) and `2967-2983` (out_proj). Only GDN projection
   matrices go to ANE; Conv1d, norm, attention stay on GPU.
2. **Selector**: `HIGGS_TARGET_ANE_GDN=1` at
   `crates/higgs-engine/src/model_loader.rs:207`.
3. **Kernel struct**: `GdnAneLayerKernels` at
   `crates/higgs-models/src/qwen3_next_ane.rs:312-321` holds the three
   per-layer projection kernels.
4. **Kernels are compiled per model load at fixed seq buckets.** To route
   the 0.8B drafter through ANE, we need one of:
   - A second ANE worker thread + second compile (duplicates memory for
     two full kernel sets).
   - A shared worker with per-model dispatch multiplexing (bigger
     refactor, single compile pass runs on shared hardware).
   - A smaller compile target matching the 0.8B drafter's dims.

**Expected savings:** Drafter currently costs ~80-100ms/round at K=8 on
the coding prompt. Routing to ANE eliminates GPU contention (ANE runs in
parallel with GPU verifier) → drafter becomes ~free on the critical path
→ projected +2-3 tok/s on coding prompt, bringing it toward the 14 tok/s
ceiling derived above.

**Research questions for P1:**
- Do the 0.8B drafter's GDN layer dims match a compiled bucket? (Need
  dims from `load_qwen3_5_model` output + compare to the bucket
  registry.)
- Is the ANE worker thread re-entrant across models, or does it assume
  single-model ownership? (See `qwen3_next_ane_worker` — probably the
  latter.)
- What's the bucket size floor — can we add a 0.8B-specific small bucket
  without blowing up kernel compile time?

## Pre-flight at next session start

```bash
cd /Users/peppi/Dev/higgs
git status                  # expect: diffusion.rs dirty + recap files
git diff --stat crates/higgs-models/src/diffusion.rs

# Re-bench baseline (should reproduce 6.7 tok/s algorithm, 2.8 tok/s wall):
cargo test -p higgs-models --release test_speculative_generate_next_e2e \
  -- --ignored --nocapture --test-threads=1 2>&1 | tail -20

# Re-run natural prompts (7.3 tok/s coding, 2.7 tok/s QA):
cargo test -p higgs-models --release test_speculative_generate_next_natural_prompts \
  -- --ignored --nocapture --test-threads=1 2>&1 | tail -60
```

## Last committed sha

`96d3ee20` on `feat/magic-canvas` — unchanged. Session 5 + 6 work is
still uncommitted.
