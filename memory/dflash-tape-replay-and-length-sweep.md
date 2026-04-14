# DFlash Tape Replay + Length Sweep Results (2026-04-14)

## What was done

1. **Replaced full model rerun with GDN-only tape replay** on partial speculative-decode rejection
   - Engine (`simple.rs`): uses `forward_with_taps_tape` + `replay_tape_rollback` instead of full rerun
   - Model dispatch (`lib.rs`): added `forward_with_taps_tape` and `replay_tape_rollback` on `AnyModel`
   - Batched all 24 GDN layers into single Metal kernel dispatch (was 24 separate dispatches)
   - Kernel modified to batch-index `a_log`/`dt_bias` for multi-layer batching

2. **Length sweep proving acceptance scales with generation length**

## Key Results — 4B BF16 on 120GB/s hardware

### AR Baseline
- 9.6 tok/s (104ms/token, 64% of bandwidth ceiling)

### DFlash Length Sweep
| Tokens | Rounds | Avg Accept | Last-50 Accept | tok/s | vs AR |
|--------|--------|-----------|----------------|-------|-------|
| 65     | 20     | 3.2       | 3.2            | 17.3  | 1.8x  |
| 512    | 160    | 3.2       | 4.1            | 14.6  | 1.5x  |
| 1024   | 269    | 3.8       | 6.4            | 17.5  | 1.8x  |
| 2048   | 351    | 5.8       | 12.7           | 26.3  | 2.7x  |

Acceptance at 2048 tokens (12.7) matches dflash-mlx's reported 11.97.

### Replay Cost
- Old (full model rerun): ~30ms/round
- New (batched tape replay): ~5ms/round warm, <1% of total time

### Bandwidth Ceiling Analysis (120GB/s)
- Weight load per forward: 8GB / 120GB/s = 66.7ms
- Current verify: 163ms (2.4x weight load — room to improve)
- Theoretical DFlash ceiling at accept=13, verify=67ms: 151 tok/s (10x AR)

## What DOESN'T work
- **Iterative refinement** (multi-pass draft): acceptance drops from 3.2 → 1.1. Drafter trained on mask inputs, not its own outputs. Would need retraining.

## Next priorities
1. **Verify speed**: 163ms → ~100ms. Something in GDN kernel or model forward has 60ms overhead beyond weight loading.
2. **ANE drafting**: make draft cost ~0 by running on Neural Engine. Even at current acceptance, eliminates 19ms/round. Enables future iterative refinement if drafter retrained.
3. **Test at 4096 tokens**: acceptance should climb further toward 13.5 (dflash-mlx reports this).

## Files changed
- `crates/higgs-engine/src/simple.rs` — tape-recording verify + batched replay in DFlash loop
- `crates/higgs-models/src/lib.rs` — `forward_with_taps_tape` and `replay_tape_rollback` on AnyModel
- `crates/higgs-models/src/qwen3_next.rs` — batched tape replay kernel (batch-indexed a_log/dt_bias), batched `replay_tape_rollback`
- `crates/higgs-models/src/dflash.rs` — updated tests: tape replay, length sweep, AR baseline, full loop with timing

## Python reference comparison
- dflash-mlx on M4 Max (400GB/s): 186 tok/s at 4028 tokens, accept=13.55
- Our hardware (120GB/s): 26.3 tok/s at 2048 tokens, accept=12.7
- Bandwidth ratio: 3.3x → speed ratio should be ~3.3x → 186/3.3 = 56 tok/s target
- Gap: 26.3 vs 56 → verify speed optimization is the remaining 2x
