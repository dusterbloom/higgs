# DFlash Inventory — feat/magic-canvas

## What is DFlash

DFlash is higgs's diffusion-style block-speculative decoding pipeline. A small "drafter" model (z-lab `*-DFlash` checkpoints) emits a block of K candidate tokens by running 5 tap layers seeded from the target's hidden states; the target then runs a single batched verify-tape forward (`forward_with_taps_tape`) to accept the longest matching prefix. Block size, tap layers, mask token, and architecture are all read dynamically from the drafter's `dflash_config`. Implementation lives in `crates/higgs-models/src/dflash.rs` (drafter), `crates/higgs-engine/src/simple.rs::generate_dflash_inner` (loop), and `crates/higgs-models/src/qwen3_next.rs::forward_with_taps_tape` (verify path). The trace schema (`dflash_trace round= embed= draft= lm_draft= verify_build= verify= accept= replay= round_total= accepted= avg_accept= eff_tps=`) is the single source of per-round truth.

## Subprojects

### DFlash baseline + matched-arch
- Goal: net positive speedup on M4 base over pure decode.
- Approach: load `z-lab/Qwen3.5-{4B,9B,27B,35B-A3B}-DFlash` against matching MLX target; fully on GPU via `gpu_drafter`.
- Headline result: 4B BF16 self-draft 26.3 tok/s baseline (`d6daf3e0`), regressed to 21.3 on HEAD; 9B Carnice 22 tok/s warm AR vs 22.5 with DFlash on Natalia gsm8k prompt.
- Status: matched-arch checkpoints all on disk; net win is small (~1.12× on M4) and prompt-dependent.
- Key: `crates/higgs-engine/src/simple.rs:1418-1825`, `.planning/RECAP-2026-04-25-session31-dflash-matched-arch-ready.md`, `.planning/dflash-forensics-and-ane-target-plan.md`.

### DFlash drafter parity audit
- Goal: explain why our avg_accept (~3) trails Aryagm reference (~7.85 on same 4B pair).
- Approach: 10 architectural candidates probed (mask, RoPE, KV layout, tap point, hidden_norm…) cross-referenced against z-lab `dflash.py` and Aryagm `draft.py`.
- Headline result: 10/10 refuted. Real cause is GDN state rollback (Qwen3.5 hybrid SSM layers cannot rollback by offset). Fix `a7e2737c` raised 2.1 → 3.4; commit `bee1ee20` regressed it.
- Status: closed; redirected to regression doc.
- Key: `.planning/next-session-dflash-drafter-parity-audit.md`, commit `a7e2737c`.

### DFlash block-size sweeps
- Goal: pick a runtime default that wins across contexts.
- Approach: 9B, 4B, 27B, A3B sweeps at blocks 2/3/4/8/12/16, recording avg_accept + round_total.
- Headline result: 9B BS=12 → 19.5 tok/s (avg_accept 6.2). 4B short-context BS=3-4 wins (23.3/22.3 tok/s). Acceptance plateaus at 2.6-2.9 above the trained block.
- Status: default landed (BS=4); 27B logs show BS=2/3 even at trained=16 because of GDN rollback caveats.
- Key: `.planning/next-session-dflash-default-block-size.md`, `.planning/measurements/a3-dflash-b{2,3,12,16}.log`, `benchmarks/dflash_block_size_ab.sh`, `memory/dflash-regression-bee1ee20-handoff.md`.

### DFlash compile wrap (Bonsai b1)
- Goal: amortize MLX graph compile across rounds.
- Approach: compile-wrap shimmed via Bonsai b1 patches; verify-build cached.
- Headline result: see Bonsai inventory; affects `verify_build` timer in trace (otherwise 130-440 ms/round on 27B).
- Status: landed via session 25/26 recaps.
- Key: `.planning/RECAP-2026-04-24-session25-b1-compile-wrap-design.md`, `.planning/RECAP-2026-04-24-session26-b1-compile-wrap-landed.md`.

### DFlash FSM-aware verify (commits 394a79e2, 73777ab4, 38d33810)
- See dedicated section below.

### DFlash 27B regression / topology-B win
- Goal: survive 27B target without SIGKILL and produce a usable speedup.
- Approach: conditional `set_wired_limit_to_max` cap (`HIGGS_MLX_CAP_FRACTION=0.7`, `cap_mb=17891`); chunked verify; topology-B drafter geometry.
- Headline result: 27B baseline 6.25 tok/s → DFlash 5.97 tok/s, eff_tps 7.24 (warm48 run). topoB BS=12 measurement queued but CSV is header-only.
- Status: cap shipped, 27B no longer crashes; DFlash is still net-flat at 27B.
- Key: `.planning/next-session-27b-dflash-crash.md`, `benchmarks/dflash_27b_topoB_20260423_225643/`, `benchmarks/dflash_27b_warm48_20260416_154055.out`, `memory/dflash-27b-streaming-handoff.md`.

### DFlash A3B 6× regression
- Goal: enable DFlash on Qwen3.6-35B-A3B (40-layer MoE) toward 20 tps.
- Approach: matched drafter `z-lab/Qwen3.6-35B-A3B-DFlash` (taps `[1,10,19,28,37]`, BS=16); env `HIGGS_DFLASH_DISABLE_ANE=1`.
- Headline result: AR baseline 43.83 tok/s, DFlash median 6.71 tok/s — **6.5× net-negative**. Suspect: silent fp16→f32 upcasts not swept by `2de6ad03`.
- Status: open; drafter forward, verify-tape, and engine glue remain to be audited; OOM blocks ANE-on testing on 32 GB box.
- Key: `.planning/RECAP-2026-04-25-session33-a3b-dflash-6x-regression.md`, `.planning/measurements/qwen36-a3b-dflash-block16-trace.log`, `.planning/next-session-qwen36-dflash-repro.md`.

### DFlash Python parity vs Rust
- Goal: explain Rust 22.6 vs handoff-claimed Python 28.24 tok/s.
- Approach: ran Python+Rust on identical Natalia gsm8k prompt with byte-identical first-round drafts.
- Headline result: Rust ≈ Python on a single sample (22.5/22.6 vs 22.08); 28.24 was a 3-sample average. Gap is sample-dependent.
- Status: closed (Rust matches Python on apples-to-apples).
- Key: `.planning/next-session-dflash-python-parity.md`, ref `/Users/peppi/Dev/dflash/dflash/benchmark.py`.

### DFlash temperature sweeps
- Goal: characterize accept rate vs sampling temperature.
- Approach: 9B Carnice + drafter at temp ∈ {0.0, 0.3, 0.7, 1.0}, 3 reps × 100 tokens.
- Headline result: temp=0 → 23.39-24.07 tok/s, accept 5.94, eff_tps 29-30; degrades smoothly with temp.
- Status: data captured; not converted into runtime guidance.
- Key: `benchmarks/dflash_9b_temp_sweep_20260416_134301.out`, `benchmarks/dflash_9b_temp_sweep.sh`.

### DFlash drafter A/B (CPU vs GPU vs ANE)
- Goal: kill CPU-marshalled streaming drafter path.
- Approach: streaming `generate_dflash_streaming_inner` rewritten to mirror sync `gpu_drafter.forward`; dropped CPU f32 conversions.
- Headline result: 4B 13.3→20.5 tok/s (+54%); 9B 7.5→13.2 (+76%); `draft_ms` 114→0.1 ms (MLX fuses graph).
- Status: shipped uncommitted on `feat/magic-canvas`.
- Key: `benchmarks/dflash_9b_drafter_ab.sh`, `memory/dflash-cpu-drafter-handoff.md`, `memory/dflash-ane-drafter-handoff.md`.

### DFlash int8 weights (ANE bridge)
- Goal: halve ANE drafter weight bandwidth (fp16→int8) below 18.5 ms forward floor.
- Approach: NEW `.mlpackage`/`MLModel(.cpuAndNeuralEngine)` path — abandons the raw-MIL emitter (rejected `tensor<int8>` on macOS 26.3.1).
- Headline result: probe artifacts in `/tmp/higgs_int8_probe/` confirm CoreML scheduler picks ANE at realistic DFlash-4B shapes; not yet wired to engine.
- Status: open; needs `AneKernel::from_mlpackage` + offline build step + ANE-dispatch verifier.
- Key: `.planning/next-session-dflash-int8-weights.md`, `crates/higgs-models/src/ane_bridge.rs`.

## Performance evolution table

| Date | Config | tps / metric | SHA-or-file | Note |
|---|---|---|---|---|
| 2026-04-13 | 4B Q4, Python ref | 7.85 avg_accept | session `1cdfaead` | reference Aryagm |
| 2026-04-14 | 9B Carnice, M4, lucky burst | 58-60 tok/s | `dflash-forensics-and-ane-target-plan.md` | accept 16/16, rare |
| 2026-04-14 | 9B Carnice, mixed | 18-22 tok/s | same | accept ~4.1/16, net 1.12× |
| 2026-04-16 | 9B BS=12 + GDN-ANE | 19.5 tok/s, accept 6.2 | `memory/dflash-regression-bee1ee20-handoff.md` | recovered from bee1ee20 |
| 2026-04-16 | 27B Q4 baseline | 6.25 tok/s | `dflash_27b_warm48_…out` | warm |
| 2026-04-16 | 27B Q4 + DFlash | 5.97 tok/s, eff 7.24 | same | net flat |
| 2026-04-16 | 9B temp=0 sweep | 23.39-24.07, accept 5.94 | `dflash_9b_temp_sweep_…out` | best AB cell |
| 2026-04-19 | 9B noane_dflash regression | 6.7 tok/s | `matrix_20260419_174750/` | streaming CPU path bug |
| 2026-04-19 | Streaming drafter fix (4B) | 13.3 → 20.5 tok/s | `next-session-dflash-default-block-size.md` | +54% |
| 2026-04-21 | A3B BS=16 | eff_tps 4.2, accept 3.0 | `qwen36-a3b-dflash-block16-trace.log` | first run |
| 2026-04-24 | 4B BF16 self-draft, ctx 2048 | 21.3 tok/s, accept 4.3 | `golden-regression-HEAD.md` | −19% vs `d6daf3e0` |
| 2026-04-25 | 27B BS=12 | round_total 713 ms, avg_accept 3.2 | `a3-dflash-b12.log` | warm |
| 2026-04-25 | A3B AR vs DFlash | 43.83 → 6.71 tok/s | `RECAP-…session33` | 6.5× negative |
| 2026-04-26 | 4B + DFlash + FSM smoke | eff_tps 50.1, accept 3.1 | `RECAP-…session4` | green, JSON mode |

## FSM-aware verify (recent landed)

Three commits on `feat/magic-canvas` opened FSM-constrained generation through the speculative paths. `394a79e2` ("feat(spec-decode): FSM-aware verify in DFlash") lifted the gate at `simple.rs:1083` so `generate_dflash_inner` now accepts `&mut Option<ConstrainedGenerator>`; the verify-side mask is built per-block via the existing `peek_states_for_drafts`/`build_mask_rows` primitives, the early-exit on `is_finished()` runs after the cache fast/slow path (lines 1937-1944) to keep KV coherent. `73777ab4` ("AR-spec FSM-aware verify (Option A)") followed the same trait-callback pattern (`FsmHook` in `higgs-models::diffusion`, `ConstrainedFsmHook` wrapper in `higgs-engine`) so AR-spec gets the same mask/advance/finish hooks at `speculative_generate_next` (`diffusion.rs:4308-4535`). `38d33810` defaulted AR-spec K window to 2..3 and produced **+11-30% tps**. Smoke test on 4B + DFlash JSON-mode (RECAP session4): 7 rounds, accept histogram 4×full + 1×3 + 1×2 + 1×1, avg_accept settled ~3.1, peak eff_tps 54.9. 420/420 lib + 15/15 constrained tests green.

## Open threads

- `next-session-dflash-int8-weights.md` — `AneKernel::from_mlpackage`, offline `.mlpackage` build, ANE-dispatch verifier — none wired.
- `next-session-qwen36-dflash-repro.md` — A3B 6× regression dtype hunt (drafter forward, tape, engine glue) still open.
- `next-session-dflash-default-block-size.md` — runtime default ship + commit on `feat/magic-canvas`.
- `next-session-bench-matrix-dflash-handoff.md` — bench-matrix `start_server` stderr fix; rerun matrix once DFlash regression is closed.
- `next-session-27b-dflash-crash.md` — 9B sanity test + conditional MLX cap + 27B context sweep beyond Step 0.
- `dflash-forensics-and-ane-target-plan.md` — verify-on-ANE remains the only single lever to hit 4× claim on M4.

## Recommended cleanup

| filename | superseded by | reason |
|---|---|---|
| `.planning/next-session-dflash-drafter-parity-audit.md` | `.planning/next-session-dflash-regression.md` | parity audit closed (10/10 refuted); cause is GDN rollback not parity |
| `.planning/next-session-dflash-python-parity.md` | (closed) | resolved: Rust ≈ Python on apples-to-apples; 28.24 was 3-sample avg |
| `memory/dflash-regression-bee1ee20-handoff.md` | `.planning/RECAP-2026-04-25-session33…` | bee1ee20 root cause now subsumed under broader dtype-upcast hypothesis |
| `benchmarks/dflash_27b_topoB_20260423_225643/dflash_topoB.csv` | (rerun) | header-only file, never populated |
| `memory/dflash-ane-projections-{v1,v2}-handoff.md` | `memory/dflash-ane-projections-handoff.md` | versioned handoffs; keep latest only (ANE-specific, see ANE inventory) |
| `.planning/next-session-27b-dflash-crash.md` (Step 0 only) | RECAP session 31 (cap shipped) | conditional MLX cap landed; remaining steps belong to the open A3B/A2 baseline track |
