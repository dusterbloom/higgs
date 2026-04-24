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
