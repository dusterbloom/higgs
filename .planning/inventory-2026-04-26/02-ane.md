# ANE Inventory — feat/magic-canvas

## Subprojects

### ANE Drafter (DFlash inline drafter)
- **Goal**: Run the small DFlash drafter on Apple Neural Engine so it overlaps with GPU verify, taking draft cost off the critical path.
- **Approach**: Wire 4B/9B/27B/35B-A3B DFlash drafters through the `ane_bridge` IOSurface path; build matched-arch shim; iterate on parity (4B → 9B → 27B). Investigated the 39 ms `transfer` sub-timer that survives `HIGGS_DFLASH_DISABLE_ANE`.
- **Headline result**: ANE+GPU vs CPU-BLAS drafter on Qwen3.6-35B-A3B-4bit produced 0 ms improvement on `lm_draft` and 0 tps on `eff_tps` (both 7.0–7.1 tok/s) — `RECAP-2026-04-25-session34-ane-no-fix-transfer-39ms-found.md`.
- **Status**: BLOCKED — drafter location is irrelevant; mactop confirms ANE idle. Real cost is the 39 ms `transfer` sub-timer, not drafter compute.
- **Key files**: `dflash-ane-drafter-handoff.md`, `RECAP-2026-04-24-session16-ane-drafter-investigation.md`, `RECAP-2026-04-25-session34-ane-no-fix-transfer-39ms-found.md`, `next-session-ane-drafter-p1-handoff.md`, `next-session-B-measure-inline-ane-drafter.md`.

### ANE Prefill (long-context offload)
- **Goal**: Offload chunked prefill of GDN layers (75% of Qwen3.5-35B-A3B) to ANE for 30 k+ contexts at ~2 W instead of 62 W on GPU.
- **Approach**: Split-silicon design — GDN layers on ANE, FA layers on GPU; engine-side bucket=512 chunked prefill matched to ANE dispatch.
- **Headline result**: Reference target 268 tok/s on 0.8 B drafter at 0.22 W (vs 62.05 W GPU) cited in design doc — `docs/ane-prefill-design.md`. No end-to-end Higgs prefill measurement yet.
- **Status**: PAUSED — design landed; execution gated by 27B memory blocker and drafter pivot.
- **Key files**: `docs/ane-prefill-design.md`, `next-session-phase1-ane-memory-handoff.md`, `phase1-ane-memory-surgery-plan.md`.

### ANE Projections v1 / v2 (linear-layer offload)
- **Goal**: Run QKV / O / MLP projections of the DFlash drafter on ANE while target K/V stays on CPU BLAS.
- **Approach**: v1 separate ANE evals → v2 added realtime dispatch (`begin_realtime`/`eval_realtime`), silu→down IOSurface chain (`share_output_to`/`eval_chain_realtime`), NEON 4×4 block transpose, and `std::thread::scope` overlap with target K/V.
- **Headline result**: v2 median forward 28.0 ms at ctx=16 (v1 was 29.0 ms); **18.5 ms ANE floor** at ~55 GB/s effective DRAM bandwidth on 1 GB of fp16 weights — `dflash-ane-projections-v2-handoff.md`.
- **Status**: WORKING but bandwidth-bound; further scheduling tricks won't move the floor.
- **Key files**: `dflash-ane-projections-handoff.md`, `dflash-ane-projections-v1-handoff.md`, `dflash-ane-projections-v2-handoff.md`.

### ANE GDN / Recurrence / Replay
- **Goal**: Move GDN recurrence + qkvz/ba/out kernels onto ANE; remove per-dispatch GPU↔CPU fence via `eval_chain`.
- **Approach**: Wire `dispatch_recurrence` in `qwen3_next_ane_worker.rs`; fuse `qkvz_ba_fused` to compile 2 kernels instead of 3; tile/per-head dispatch to fit ANE compile limits.
- **Headline result**: Topology B (HIGGS_TARGET_ANE_GDN=0, BS=16) **22.49 tok/s vs 19.46 baseline (+15.5%)**; verify_build collapsed 230.9 → 2.2 ms after removing 72 ANE GDN dispatches — `topology-b-win-and-ar-ane-handoff.md`. ANE recurrence compilation **fails at 9B dims** (flat_w=4096 > ~64 limit) — `gate1-ane-worker-wiring-handoff.md`.
- **Status**: ABANDONED for target GDN — keep ANE-GDN off (net loss). Drafter-side GDN replay still in flight (eval_chain).
- **Key files**: `gate1-ane-worker-wiring-handoff.md`, `topology-b-win-and-ar-ane-handoff.md`, `next-session-ane-gdn-eval-chain.md`, `gdn-replay-optimization-handoff.md`.

### ANE INT8 MLP zero-copy
- **Goal**: Eliminate 6 element-wise f32↔fp16 transpose loops per forward in `forward_ane_int8_mlp`; default 512-bucket aligned with engine chunked prefill.
- **Approach**: Layer-0 int8 MLP wired (commit `89141aa6`); next step is zero-copy IOSurface I/O to drop the transposes.
- **Headline result**: Probe vs MLX q4 @ seq=128 — gate/up **2.15×**, down **1.58×**, both `on_ane=true`. Layer-0 parity `max_diff=0.37` (2.2%), `mean_diff=0.040` (0.24%) — `next-session-ane-int8-mlp-zerocopy.md`.
- **Status**: WORKING (layer-0 + bucket=512 landed); zero-copy refactor open.
- **Key files**: `next-session-ane-int8-mlp-zerocopy.md`.

### ANE LM head
- **Goal**: Evaluate ANE for the 152k-vocab LM head matmul.
- **Headline result**: C1-lm-head (M=128, N=152000, K=4096) on `cpuAndNe` = **154 GFLOP/s** (1036 ms warm) vs cpuOnly 718 GFLOP/s — ANE **loses by 4.7×** on this shape — `ane_c1_sustained_tflops.md`.
- **Status**: ABANDONED — vocab-size LM-head shape is anti-ANE.
- **Key files**: `.planning/measurements/ane_c1_sustained_tflops.md`.

### ANE 9B parity
- **Goal**: Get DFlash ANE forward to numerical parity on the 9B drafter (4B already passed).
- **Approach**: Two bug fixes: (1) MIL `concat` on innermost (oc/width) axis silently emits NaN — rewrote `emit_blobfile_matmul_tiled` to concat on channel-axis=1; (2) scaled `down_proj` weights to avoid fp16 saturation.
- **Headline result**: 4B `max_diff=0.033`, 9B `max_diff=0.082` (tolerance 0.15), no NaN/Inf — `next-session-ane-9b-parity.md`.
- **Status**: WORKING — RESOLVED (commits `068a14ef`, `c95a80c7`).
- **Key files**: `next-session-ane-9b-parity.md`, `memory/9b-optimization-report.md`.

### ANE matmul / sustained TFLOPS / dispatch round-trip
- **Goal**: Establish hardware ceilings for ANE on M4 base (synergy/reframe verification).
- **Approach**: Built `benchmarks/ane_matmul_bench/` (Rust + ObjC++ port of ccv CoreML dynamic-weights pattern), IOSurface-backed MLMultiArray, ios19 MIL (2× dequantize + 1× matmul). Swept C1 shapes and (16,16,16) dispatch.
- **Headline result**: Dispatch round-trip **PASS at 0.032 ms warm** on (16,16,16) cpuAndNe; sustained TFLOPS **FAIL** — max 11.3 TFLOPs (gate 18 TFLOPs) on prefill-big shape — `ane_g2_dispatch_roundtrip.md`, `ane_c1_sustained_tflops.md`.
- **Status**: WORKING measurements; informs all downstream gating.
- **Key files**: `ane_g2_dispatch_roundtrip.md`, `ane_c1_sustained_tflops.md`, `ane_bench_raw.json`, `next-session-ane-synergy-handoff.md`, `next-session-ane-reframe-verification.md`.

## Headline numbers table

| Probe | Date | Setup | Result | Source |
|---|---|---|---|---|
| G2 dispatch round-trip | 2026-04-17 | M4, (16,16,16), cpuAndNe, 1000 iters | 0.032 ms warm mean (PASS, gate 0.15 ms) | `ane_g2_dispatch_roundtrip.md` |
| C1 sustained TFLOPs (max) | 2026-04-17 | M4, prefill-big 1024×11008×4096, cpuAndNe | 11 337 GFLOP/s (FAIL, gate 18 000) | `ane_c1_sustained_tflops.md` |
| C1 attn-proj | 2026-04-17 | M4, 128×4096×4096, cpuAndNe | 3 790 GFLOP/s @ 1.13 ms warm | `ane_c1_sustained_tflops.md` |
| C1 moe-gateup | 2026-04-17 | M4, 128×11008×4096, cpuAndNe | 2 905 GFLOP/s @ 3.97 ms warm | `ane_c1_sustained_tflops.md` |
| C1 lm-head | 2026-04-17 | M4, 128×152000×4096, cpuAndNe | 154 GFLOP/s (ANE loses 4.7×) | `ane_c1_sustained_tflops.md` |
| Inline ANE drafter run | 2026-04-24 | K=12, 27B target | (raw log only — no quantitative summary in handoff) | `inline-ane-drafter/run-20260424-K12.log` |
| Topology B (ANE-GDN OFF) | 2026-04-16 | 9B, BS=16 | 22.49 tok/s (+15.5% vs 19.46 baseline) | `topology-b-win-and-ar-ane-handoff.md` |
| ANE projections v2 | 2026-04 | 4B drafter ctx=16 | 28.0 ms median (18.5 ms ANE floor, 55 GB/s) | `dflash-ane-projections-v2-handoff.md` |
| 9B DFlash ANE parity | 2026-04 | 5 layers | max_diff=0.082 (tol 0.15), 0 NaN/Inf | `next-session-ane-9b-parity.md` |
| ANE drafter on 35B-A3B-4bit | 2026-04-25 | K=16, 562-tok prompt | 7.0 tok/s (no improvement vs CPU BLAS 7.1) | `RECAP-2026-04-25-session34-ane-no-fix-transfer-39ms-found.md` |

## Open threads

- `next-session-ane-drafter-p1-handoff.md` — route 0.8 B drafter through Qwen3Next ANE path; projects 14 tok/s ceiling on coding prompts.
- `next-session-B-measure-inline-ane-drafter.md` — resume Point B inline-IOSurface drafter measurement on 27B; blocked on `--features ane` build break at `bd3lm_qwen3.rs:118`.
- `next-session-ane-fix-remaining.md` — 3 hot-path call sites left to finish ANE DFlash bit-rot fix; bench Carnice-9B follows.
- `next-session-ane-9b-parity.md` — superseded by RESOLVED status (commits landed); kept for record only.
- `next-session-ane-int8-mlp-zerocopy.md` — zero-copy `forward_ane_int8_mlp` to drop 6 transposes; scale to 30 k+ prefill.
- `next-session-ane-gdn-eval-chain.md` — kill per-dispatch GPU↔CPU fence in GDN drafter via `eval_chain`; uncommitted hunks pending split-add.
- `next-session-ane-reframe-verification.md` — verify load-bearing claims of M4-base projections doc and Draw Things reframe before committing to refactor.
- `next-session-ane-synergy-handoff.md` — ANE/GPU/CPU synergy follow-ups (D, E gates after C1/G2).
- `next-session-phase1-ane-memory-handoff.md` — 27B DFlash ANE blocked on OS jetsam SIGKILL at 23.5 GB RSS during `compile_dflash_ane`; MLX cap doesn't help (CPU f32 dequant transients).

## Recommended cleanup

- `next-session-ane-9b-parity.md` — superseded by — RESOLVED in commits `068a14ef` + `c95a80c7`; can be archived.
- `dflash-ane-projections-handoff.md` and `dflash-ane-projections-v1-handoff.md` — superseded by — `dflash-ane-projections-v2-handoff.md` (v2 is the current floor at 28 ms / 18.5 ms ANE).
- `RECAP-2026-04-24-session16-ane-drafter-investigation.md` — superseded by — `RECAP-2026-04-25-session34-ane-no-fix-transfer-39ms-found.md` (session 34 ruled out 3 of 4 hypotheses and pinned next move).
- `phase1-ane-memory-surgery-plan.md` — superseded by — `next-session-phase1-ane-memory-handoff.md` (battle plan's Phase 1 declared outdated; gate/up MIL split already implemented).
- `next-session-ane-synergy-handoff.md` and `next-session-ane-reframe-verification.md` — partially superseded by — `ane_c1_sustained_tflops.md` + `ane_g2_dispatch_roundtrip.md` (G2 PASS, C1 FAIL recorded); only the un-run D/E gates remain.
