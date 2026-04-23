# Session 5 recap (2026-04-23) — persistent verify cache + 12 tok/s push

## TL;DR

1. **Shipped the persistent verify cache fix** from session 4's handoff:
   `speculative_generate_next` now uses snapshot/restore on a single
   `verify_cache` (not fresh-cache-per-round). Verify per round dropped from
   ~3s (cold full-prefill) to ~467ms (warm K-token append).
2. **Bumped K from 4..8 → 8..16** in the test (K=16 matches the historical
   12.7 tok/s baseline). Acceptance jumped 28.6% → 41.7%.
3. **Algorithm throughput: 0.4 → 6.7 tok/s** (algorithm-only). Wall is still
   bootstrap-dominated at small N.
4. **Did NOT hit 12 tok/s.** Ceiling on this hardware/model pair is bounded
   by the 27B GDN/SSM verify forward floor (~467ms warm, ~559ms at K=16).
   The pathological "187543 187544…" digit-loop in the test prompt also
   suppresses acceptance below natural-prompt levels.
5. **Persistent drafter cache + skip end-of-round prefill: designed, not
   implemented.** Estimated ~+1.5 tok/s → ~8 tok/s. See "Next session"
   below for the design.
6. **ANE drafter path: research done, but the agent's analysis of
   "no qwen3_next→ANE plumbing" was WRONG per user correction. Qwen3 verify
   already has ANE.** Re-investigate next session.

## Per-round timing (test_speculative_generate_next_e2e, K=8..16, 60 tokens)

```
R1 (K=16, 0/16): draft=267ms verify=1525ms   ← cold, lazy weight load
R2 (K=16, 0/16): draft=167ms verify=558ms
R3 (K=8, 1/8):   draft=164ms verify=558ms    ← K-ctrl dropped to 8
R7 (K=8, 0/8):   draft=82ms  verify=466ms
R8 (K=8, 2/8):   draft=83ms  verify=467ms
R9 (K=8, 8/8):   draft=82ms  verify=467ms    ← FAST PATH (full accept)
R10 (K=8, 0/8):  draft=87ms  verify=467ms
R11 (K=16, 8/8): draft=86ms  verify=467ms    ← K-ctrl ramped to 16
R6 (K=16, 16/16): draft=181ms verify=559ms   ← +17 new tokens (peak)
R7 (K=16, 15/16): draft=184ms verify=560ms   ← +16 new tokens

Totals: 1383ms draft + 6093ms verify + 1451ms advance = 8927ms
Acceptance: 50/120 (41.7%)
Throughput: 6.7 tok/s (algorithm), 2.5 tok/s (wall, includes 14.9s bootstrap)
```

**Variance is enormous.** R6 peak: 17 tokens / 860ms = 19.7 tok/s.
R10 trough: 1 token / 700ms = 1.4 tok/s. The "187543 187544…" digit pattern
created by the synthetic 5-token prompt alternates between predictable
streaks (drafter accepts) and boundary tokens (drafter misses). Real
prose/code prompts should give tighter variance + higher mean acceptance.

## Files modified this session

- `crates/higgs-models/src/diffusion.rs`:
  - `speculative_generate_next` (lines ~4192–4400): rewrote with persistent
    `verify_cache`, snapshot/clone via `eval_for_clone()`, fast/slow paths
    mirroring the canonical pattern at line 7430+. Added per-round phase
    timing (`draft=Xms verify=Yms`). Drafter is still rebuilt per round.
  - Test `test_speculative_generate_next_e2e` (lines ~8202–8260):
    `max_seq=512 → 1024`, `max_tokens=20 → 60`, `k=4..8 → 8..16`, assertion
    bound updated to `<=60`.

## Next session — concrete to-do (in priority order)

### P0: Persistent drafter cache + skip e-o-r prefill (~30min, +~1 tok/s)

Currently the drafter does TWO full prefills per round (start-of-round
`prefill(seed)` and end-of-round `prefill(context)`), totaling ~80–180ms
of wasted work. Mirror the persistent verify cache pattern.

**Design (mirrors verify pattern in `speculative_generate_next`):**

```rust
// Bootstrap (replace lines ~4216-4218):
let mut d_cache = drafter.model.make_cache();
let prompt_arr = mlx_rs::Array::from_slice(&prompt_i32, &[1, prompt_i32.len() as i32]);
let mut saved_draft_logits = drafter.model.forward(&prompt_arr, None, &mut d_cache)?;
mlx_rs::transforms::eval([&saved_draft_logits])?;
// Need eval_for_clone equivalent for the raw Vec<Option<LayerCache>>:
//   make `QwenNextCausalDrafter::eval_cache_with` pub(crate) and call it,
//   OR inline the logic from lib.rs:136-150.

// Round draft phase (replace lines ~4226-4250):
let draft_0 = ix::argmax_axis(&saved_draft_logits, -1, false)?
    .index((0, 0)).item::<i32>() as u32;  // verify shape — see Caveat below
let mut draft_tokens: Vec<u32> = vec![draft_0];
let d_snapshot: Vec<Option<LayerCache>> = d_cache.clone();
let mut last_tok = draft_0;
for _ in 1..k {
    if context.len() + draft_tokens.len() >= drafter.max_seq { break; }
    let next = drafter.step(last_tok, &mut d_cache)?;  // already evals
    draft_tokens.push(next);
    last_tok = next;
}

// Drafter advance (add alongside verify-cache advance):
if accepted == actual_k {
    // FAST: cache consumed draft[0..K-2]. Need cache at L+K+1.
    // Feed [draft[K-1], bonus] (2 tokens).
    let bonus = *accepted_tokens.last().unwrap();
    let advance = vec![draft_tokens[actual_k - 1] as i32, bonus as i32];
    let arr = mlx_rs::Array::from_slice(&advance, &[1, 2]);
    saved_draft_logits = drafter.model.forward(&arr, None, &mut d_cache)?;
} else {
    // SLOW: restore + feed accepted_tokens.
    d_cache = d_snapshot;
    let advance: Vec<i32> = accepted_tokens.iter().map(|&t| t as i32).collect();
    let arr = mlx_rs::Array::from_slice(&advance, &[1, advance.len() as i32]);
    saved_draft_logits = drafter.model.forward(&arr, None, &mut d_cache)?;
}
mlx_rs::transforms::eval([&saved_draft_logits])?;
// + eval_for_clone equivalent on d_cache.

// Delete the end-of-round drafter.prefill(&context) call entirely.
```

**Caveat — index shape uncertainty:** `Qwen3NextCausalLM::forward` shape
return is unverified (saw 5 forward methods at qwen3_next.rs lines 249,
290, 1532, 1664, 2115 — didn't trace which is the public LM head). The
existing `prefill` uses `.index((0, 0))` which works in tests, suggesting
forward returns `[1, T, vocab]` and `argmax(-1).index((0, 0))` gives
position 0's prediction (which would be WRONG for end-of-prompt). Yet
the test produces plausible tokens. **Spend 10min verifying shape before
implementing** — try `.index((0, T-1))` if needed, where `T = arr.len()`.

### P1: Re-investigate ANE drafter path (USER CORRECTED ME)

**The Explore agent's report was WRONG about ANE for qwen3_next.** Per
user correction: "we have ANE for Qwen3 verify". Need to:

1. Find where Qwen3 verify dispatches to ANE — search for `Qwen3NextCausalLM`
   + `ane`, `coreml`, `compile`, etc. The agent searched `diffusion.rs` and
   missed the qwen3_next ANE path.
2. Determine if the same ANE path can be used for the 0.8B drafter
   (`Qwen3.5-0.8B-8bit` loaded via `load_qwen3_5_model`).
3. If yes: route the drafter forward through ANE → drafter cost drops
   to ~zero (separate accelerator) → unblocks 12+ tok/s ceiling.

The agent DID confirm correctly:
- `AneCausalDrafter` exists at `diffusion.rs:3859` for the dense-Qwen3 path
- The 0.8B Qwen3.5-8bit IS loadable via `load_qwen3_5_model` (qwen3_next.rs)
- Vocab 248320 matches the 27B target
- A `QwenNextAneDrafter` adapter would be the right shape

**The unknown** is just: where's the Qwen3Next ANE backend selector?
That's the missing piece the agent didn't find. User says it exists.
Look at how the 27B verifier dispatches forward today — does it conditionally
hit ANE? Probably yes given `HIGGS_TARGET_COMPILE` mentioned in commits.

Search hints:
```bash
rg -n 'ANE|ane' crates/higgs-models/src/qwen3_next.rs | head -30
rg -n 'HIGGS_TARGET_COMPILE|target_compile' crates/higgs-models/src/
git log --oneline -20 -- crates/higgs-models/src/qwen3_next.rs
```

### P2: Run with a natural prompt (~10min, +~3-5 tok/s expected)

The "The capital of France is" test prompt is pathological — it produces
"187543 187544 187545…" digit loops where drafter alternates between
streak (high accept) and boundary (zero accept) tokens. Try:

- A 100-token coding prompt ("// Implement a binary tree node in Rust:\n")
- A 100-token QA prompt ("Q: Explain how MLX achieves zero-copy on Apple
  Silicon. A:")

Prediction: 50–60% acceptance, 10–14 tok/s on the persistent-cache path.

## ANE research agent's full report (verbatim, for reference)

The agent ran in parallel and found:
- `AneCausalDrafter` interface at `crates/higgs-models/src/diffusion.rs:3859–3923`
  with `forward_logits()`, `draft(prefix, k)`, etc.
- 27B verify is loaded via `load_qwen3_5_model` at `qwen3_next.rs:5856`
  (handles nested `text_config` + `language_model.` prefix).
- ANE backend exists as `DiffusionBackend::AneFused` enum + `DiffusionRuntime`
  dispatch at `diffusion.rs:1142–1253`.
- C++/Obj-C bridge: `crates/higgs-models/bridge/ane/ane_bridge.{h,m}`,
  `ane_mil.rs`, `ane_mlmodel.rs`, `diffusion_ane.rs`.

The agent CLAIMED there's no `Qwen3NextRuntime` with ANE selector. **User
says this is wrong.** Look harder next session.

## Pre-flight at next session start

```bash
cd /Users/peppi/Dev/higgs
git status                  # expect dirty: diffusion.rs (verify cache rewrite)
                            #            + this recap
# Verify the new code is in:
rg -n 'verify_snapshot' crates/higgs-models/src/diffusion.rs
# Should show 1 hit in speculative_generate_next.

# Re-bench to baseline (should reproduce 6.7 tok/s):
cargo test -p higgs-models --release test_speculative_generate_next_e2e \
  -- --ignored --nocapture --test-threads=1 2>&1 | tail -20
```

## Math for the 12 tok/s target

Per-round costs (current, K=16, warm):
- Drafter: ~180ms (2× prefills + steps)
- Verify forward: ~559ms (27B GDN, K=16)
- Advance: ~120ms (1 verify forward post-accept)
- **Total: ~860ms/round**

Tokens per round (avg): 41.7% × 16 + 1 = 7.7 (in theory)
Actual avg measured: 6.0 (variance from pathological prompt)

To hit 12 tok/s = need 10.3 tokens/860ms or 6 tokens/500ms.

| Optimization | Saves | New ceiling |
|--------------|-------|-------------|
| Persistent drafter cache | -100ms/round | ~7.5 tok/s |
| Natural prompt → 55% accept | +2 tok/round | ~10 tok/s |
| ANE drafter (zero GPU contention) | -180ms drafter | ~12 tok/s |
| All three combined | n/a | ~14 tok/s |

So the user's intuition is right: **with ANE drafter + a real prompt
+ persistent drafter cache, 12 tok/s is reachable**. The verify floor
is 467ms — that's the asymptote, ~14 tok/s with K=16 fully accepted.

## Last committed sha

`96d3ee20` on `feat/magic-canvas` — unchanged. All session 5 work is
uncommitted in the working tree.

## Files created this session

- `.planning/RECAP-2026-04-23-session5-persistent-verify-cache.md` — this file

## Files modified this session

- `crates/higgs-models/src/diffusion.rs` — persistent verify cache,
  per-round timing, K=8..16 test bump
