# Bonsai-8B verify-cost probe (P6 sanity check)

Date: 2026-04-24
Binary: `cargo test -p higgs-models --release --lib bench_bonsai_q1_verify_cost_8b -- --ignored --nocapture`
Path: `AnyModel::BonsaiQ1 → BonsaiQ1Gpu::forward_all_logits`

## Setup

- Model: Bonsai-8B-mlx-1bit, 36 layers, vocab 151669, packed 1220.7 MB resident.
- Prime prefix: 64 tokens (fresh KV per K measurement, matches mid-generation cache depth).
- Timing: min of 5 iters per K after per-shape warmup.

## Data

| K | min ms | ms/tok | vs K=1 ratio | ideal ratio |
|---:|---:|---:|---:|---:|
| 1 | 42.38 | 42.38 | 1.00× | 1× |
| 4 | 193.31 | 48.33 | 4.56× | 4× |
| 8 | 414.69 | 51.84 | **9.79×** | 8× |
| 12 | 178.66 | 14.89 | **4.22×** | 12× |
| 16 | 178.03 | 11.13 | **4.20×** | 16× |

## Findings

1. **K=1 baseline = 42.4 ms.** Matches the session-22 decode rate (23.4 tok/s ≈ 42.7 ms/tok) — sanity check passes: one-token `forward_all_logits` ≈ one decode step.

2. **K=4 is linear.** 4.56× baseline (slight constant-cost overhead of ~6 ms).

3. **K=8 anomaly.** 9.79× baseline, 52 ms/tok — slightly *worse* than K=1 per-token. Likely a kernel-compile or MLX quantized-matmul tile-alignment artifact: 8 is below whatever threshold flips MLX into a fused `[1, T, vocab]` path.

4. **K=12 and K=16 collapse to sub-linear.** 178 ms regardless of K — a flat cost. Per-token drops to 14.9 ms (K=12) then 11.1 ms (K=16). Strong evidence that MLX's `quantized_matmul` on the LM-head projection `[1, T, 151669]` hits a different kernel at T≥12 that amortizes setup cost across the T dim.

## Impact on Path A experiment

Prior session-22 back-of-envelope assumed verify on Bonsai-8B for K=8 drafts would cost ~60–90 ms and yield a 1.6× win at α=0.7. This data says:

- **K=8 is the worst K to pick.** Verify cost (415 ms) > baseline cycle budget — Path A would lose badly.
- **K=12 is the sweet spot.** 178 ms verify, E[n] ≈ 4 at α=0.7 → 22.5 tok/s (~break-even with the 23.4 tok/s 8B baseline). Not a clear win.
- **K=16 could win slightly.** 178 ms (same as K=12!) with E[n] ≈ 4.5 at α=0.7 → 25.3 tok/s (~1.08× baseline). Marginal.

## Implications / next steps

- Path A's best case at K=12/16 is narrow (~1.1× at best). Not the 1.6× the session-22 math promised.
- The drafter cost (ANE) wasn't measured here — if it exceeds ~150 ms and can't hide behind verify, Path A loses.
- **Recommendation:** measure the ANE drafter latency for K=12 and K=16 before running the full daemon experiment. If ANE drafter for K=12 > 178 ms, Path A is dead.
- Sanity-rerun the K=8 measurement with longer warmup (≥10 iters) to confirm the anomaly is compile-artifact rather than a real workload-dependent regression.

## Caveats

- Fresh cache + fresh prefill per iter — some MLX state may not be fully warmed across iters.
- `min` of 5 iters is a floor estimate. Median or mean may tell a different story for K=8.
- Live spec-decode verify shares cache across cycles; per-cycle verify should be closer to the flat ~178 ms seen at K=12/16 than to the anomalous K=8 reading.

---

## Addendum (2026-04-24) — ANE drafter cross-check → Path A is dead

The P6 "next step" was: *measure the ANE drafter latency for K=12 and K=16
before running the full daemon experiment. If ANE drafter for K=12 > 178 ms,
Path A is dead.*

That measurement already exists across sessions 16 and 17 — no new probe
needed to reach a decision.

### Existing ANE drafter data

`AneBonsaiDraftModel::draft(K)` is **K sequential** `forward_last(ctx)`
calls (see `crates/higgs-engine/src/ane_bonsai_draft.rs:93-120`); the ANE
kernel is compiled for a fixed seq_len and each call re-runs the full
attention stack.

| source | seq_len | per-call | K=12 total | K=16 total |
|---|---:|---:|---:|---:|
| session 16 probe 2 (14B target, 1.7B drafter, live) | 2048 | ~1,875 ms | ~22.5 s | ~30 s |
| session 17 late-session probe (14B target, 1.7B drafter, live) | 256 | ~220 ms | ~2.64 s | ~3.52 s |

Linear seq_len scaling confirmed (session 17). The 256-token configuration
is already near the floor of useful drafter context.

### Break-even math vs the 8B baseline (23.4 tok/s ≈ 42.7 ms/tok)

To beat baseline, effective tok/s `= E[n] / cycle_ms × 1000 > 23.4`, i.e.
`cycle_ms / E[n] < 42.7`. With verify at its K=12/16 floor of **178 ms**
(flat, per the data above), and drafter running concurrently on ANE with
target on GPU so at best the first 178 ms of drafter is hidden:

- Exposed drafter = `max(0, K × per_call − 178)`.
- Cycle = 178 ms + exposed_drafter.

At seq_len=256 (best feasible for drafter with any working context):

| K | drafter total | exposed | cycle ms | E[n]@α=0.7 | eff tok/s | vs 23.4 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 880 | 702 | 880 | 2.5 | 2.8 | **−88 %** |
| 8 | 1,760 | 1,582 | 1,760 | 3.4 | 1.9 | **−92 %** |
| 12 | 2,640 | 2,462 | 2,640 | 4.2 | 1.6 | **−93 %** |
| 16 | 3,520 | 3,342 | 3,520 | 4.7 | 1.3 | **−94 %** |

Catastrophic loss at every K. The 178 ms verify-floor alone already
eats the full per-token budget for E[n]≈4, so drafter cost of **any
exposed amount** pushes the cycle below break-even.

### Sensitivity: what would Path A need?

Require `cycle ≤ 42.7 ms × E[n]`. At K=12, α=0.7, E[n]≈4.2 → cycle ≤ 179 ms
≈ the verify floor itself. Drafter must be fully hidden, i.e.
`K × per_call ≤ 178 ms` → **per-call ≤ 14.8 ms**.

Measured per-call at seq=256 is ~220 ms. Required speedup: **~15×**. That
is not a tuning-knob delta — it requires architectural change in the
drafter:

- **Stateful ANE drafter** (persistent per-layer KV). Removes the
  O(seq_len) recompute each call. Session-16 lever #4.
- **Batched-K in one ANE dispatch** (feed `[last, d₁ … d_{K-1}]` once,
  harvest all-position logits via causal mask). Session-16 lever #3.
- **BD3-LM block denoising** (parallel K-token denoise per step). Session-16
  lever #5.
- **GPU drafter** (Path B) — but it serializes with verify on the GPU,
  so it does not benefit from concurrency. Session-22 Path B estimate
  already showed marginal outcomes; with verify = 178 ms (not 70 ms as
  assumed earlier) the Path B math worsens.

### Conclusion

**Do not run the Path A full-daemon experiment.** Stateless ANE drafter
cannot beat a 23.4 tok/s GPU-only baseline for any (K, seq_len) in
feasible range. Recommend:

1. Skip the `/tmp/bonsai-split.toml` experiment described in
   `RECAP-…-session22` §"Concrete next-session work".
2. Redirect to the session-17 P1-P6 plan (packed q1 target + fused
   Metal 1-bit kernels) — that path targets PrismML's 57 tok/s 8B AR
   figure, which is a 2.4× win over our current 23.4 tok/s baseline
   without any speculative decoding at all, and without needing ANE
   to be fast enough.
3. Park Path A behind "stateful ANE drafter" as a later effort, only
   if packed-q1 AR plateaus below target.

### If decisive microbench still wanted

A self-contained bench is cheap to add:

```
#[test]
#[ignore]
fn bench_ane_bonsai_forward_last_per_call() {
    // Load AneBonsaiEngine on Bonsai-1.7B with seq_len ∈ {64,128,256,512,2048}.
    // For each seq_len, time a single forward_last(ctx) with ctx.len()=seq_len.
    // Per-shape warm × 1; min of 5 iters.
    // Expected: linear in seq_len, slope ≈ 0.86 ms/token.
}
```

Adds ~2–3 min run time (ANE kernel compile dominates at ~80 s per
seq_len; use one fixed seq=256 for a 90 s total run). Propose only if
hard numbers at exactly the seq_len we intend to use are needed for
the record.
