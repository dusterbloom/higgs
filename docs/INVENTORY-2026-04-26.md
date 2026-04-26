# Higgs Inventory — feat/magic-canvas (2026-03-23 → 2026-04-26)

> Five-week snapshot of the active development branch. Sources: 115 commits, ~80 RECAP/handoff notes, ~40 measurement files, all under `feat/magic-canvas` (branch point `43eedc6b`). Per-domain detail lives in `.planning/inventory-2026-04-26/{01..05}.md`.

---

## Executive summary

**Branch identity.** `feat/magic-canvas` is where two product threads meet: a generative-UI / structured-output direction ("magic canvas", FSM-constrained decoding, structured CoT) and a speculative-decoding stack (PLD, AR-spec, DFlash, drafter trait) made FSM-aware so it compounds with constraints. 115 commits over five weeks, two authors, a clear acceleration through the last seven days.

**Where we are.**

- **AR throughput on Bonsai-class models is essentially closed.** Session-28 dtype fix (`1712b9ab`, fp16 hold-through in `apply_yarn_rope`) gave +2.83× on Bonsai-8B (22→64 tok/s) and +2.11× on 1.7B (87→184 tok/s). Now within ~12% of mlx-lm Python (`decode-breakdown.md`).
- **PLD with FSM-aware verify shipped end-to-end** (commits `2761a7ce`, `e72b5dee`, `871c5ddf`). Verbatim Carnice-9B benchmark: **1.84× median decode** (21.68 → 39.87 tps). JSON-mode smoke green on 0.6B-4bit; 127 cycles, `{`-prefix held through verify.
- **AR-spec FSM-aware verify landed today** (`73777ab4`) and the K-window default tuning (`38d33810`) gave **+11–30% tps** with no other code change. Carnice-9B + 0.8B-8bit: 47.3% → 72.7% acceptance, 14.66 → 19.10 tps.
- **DFlash FSM gate lifted** (`394a79e2`) — `generate_dflash_inner` now accepts a constrained generator. 4B + JSON smoke: 50–55 eff_tps, accept ~3.1, no panics.

**Where we're stuck.**

- **DFlash A3B 6× regression** (`session33`): 35B-A3B AR 43.83 tps → DFlash 6.71 tps. Root cause not yet identified; suspect surviving fp16→f32 upcasts not swept by `2de6ad03`.
- **BD3LM Bonsai-8B output is broken** — Rust generates 64 commas, Python parity check reproduces same pattern. Ruled out Rust bugs; pointing at denoise-head architecture / weight tying / activation index. Phase D bench gated.
- **27B DFlash net flat** (5.97 tps with DFlash vs 6.25 baseline), and Path A (stateless ANE drafter for 27B) was killed after the p6 verify-floor data: 178 ms flat verify ≥K=12, projected −88% to −94% break-even at 8B.
- **ANE drafter "39 ms transfer mystery"** survives `HIGGS_DFLASH_DISABLE_ANE`; mactop confirms ANE idle. Real cost is the transfer sub-timer, not drafter compute.

**The single biggest signal.** Three of the last week's wins (Bonsai-Q1 +2.83×, AR-spec +30%, PLD 1.84×) came from **dtype audits, default tuning, and harness fixes**, not new architecture. The remaining low-hanging fruit is the same shape: 3 known fp16 upcast suspects, a 30-LOC structured-CoT spike, and `--features ane` build break at `bd3lm_qwen3.rs:118`.

---

## Per-domain inventory

Detailed write-ups live alongside this file. Each is a self-contained section with goals, status, key files, and headline numbers.

| Domain | File | Highlights |
|---|---|---|
| Commit timeline | `.planning/inventory-2026-04-26/01-timeline.md` | 115 commits, 5 weeks, top-10 milestones, reverts/dead-ends |
| Apple Neural Engine | `.planning/inventory-2026-04-26/02-ane.md` | 8 subprojects, hardware-ceiling table, Topology-B win, prefill design |
| DFlash | `.planning/inventory-2026-04-26/03-dflash.md` | 11 subprojects, performance evolution table, A3B regression open |
| Diffusion / BD3LM / Bonsai / Eggroll | `.planning/inventory-2026-04-26/04-diffusion.md` | Bonsai-Q1 closed, BD3LM denoise blocker, Eggroll speculative |
| Magic Canvas / Gen UI / spec-decode | `.planning/inventory-2026-04-26/05-magic-canvas-spec-decode.md` | DraftModel trait, PLD, AR-spec, structured CoT spike |

---

## Headline benchmarks (cross-domain)

| Probe | Date | Setup | Result | Source |
|---|---|---|---|---|
| Bonsai-8B decode (post-dtype fix) | 2026-04-25 | 1-bit qmm, fp16 hold-through | **64 tok/s** (was 22 tok/s) | `decode-breakdown.md` |
| Bonsai-1.7B decode (post-dtype fix) | 2026-04-25 | same | **184 tok/s** (was 87 tok/s) | same |
| AR-spec K=2..3 (Carnice-9B + 0.8B) | 2026-04-26 | T=0, ctx=2k | **72.7% acc, 19.10 tps** (+30% vs K=4..8) | `RECAP-…session7-arspec-validated-k23-win.md` |
| PLD Carnice-9B verbatim | 2026-04-26 | 3-run median, max=384 | **1.84× decode** (21.68→39.87 tps) | `pld_carnice_20260426/results.json` |
| DFlash 4B + FSM JSON smoke | 2026-04-26 | BS=4, T=0 | **50.1 eff_tps, accept 3.1** | `RECAP-…session4-dflash-fsm-smoke-arspec-plan.md` |
| 9B Carnice DFlash temp=0 | 2026-04-16 | BS=12 | **24.07 tps, accept 5.94, eff 29-30** | `dflash_9b_temp_sweep_…out` |
| Topology-B (ANE-GDN OFF) | 2026-04-16 | 9B BS=16 | **22.49 tps** (+15.5% vs 19.46 baseline) | `topology-b-win-…handoff.md` |
| ANE int8 MLP probe vs MLX q4 | 2026-04 | seq=128 | gate/up **2.15×**, down **1.58×** | `next-session-ane-int8-mlp-zerocopy.md` |
| ANE projections v2 floor | 2026-04 | 4B drafter ctx=16 | **18.5 ms ANE floor** at 55 GB/s | `dflash-ane-projections-v2-handoff.md` |
| MLX memory cap removal | 2026-04-10 | engine-wide | **5× decode throughput** | commit `e5c47264` |
| MoE batch sort | 2026-03-26 | global gather_qmm | **4-5× TTFT** | commit `bba6c82e` |
| TurboQuant u32 packed words | 2026-03-27 | decode kernels | **4× fewer mem loads** | commit `d969879a` |
| Quantized embed gather | 2026-03-25 | dequantize-after-gather | **~1.8× decode** | commit `37e986d6` (squashed in `79694606`) |
| ANE C1 LM-head (fail) | 2026-04-17 | 128×152000×4096 cpuAndNe | 154 GFLOP/s — **ANE loses 4.7×** | `ane_c1_sustained_tflops.md` |
| ANE drafter on 35B-A3B (no-op) | 2026-04-25 | K=16 | 7.0 tps (≈ CPU-BLAS 7.1) | `RECAP-…session34-…transfer-39ms-found.md` |
| DFlash A3B regression | 2026-04-25 | BS=16 | 43.83 → **6.71 tps (6.5× negative)** | `RECAP-…session33-…6x-regression.md` |

---

## Open threads (active work)

Concentrated in three buckets — pick where to attack first based on cost vs blast radius.

### High-value, ready-to-execute
1. **Structured CoT spike** — 30 LOC across `simple.rs` and `chat.rs`, env-gated `HIGGS_STRUCTURED_THINK=1`, compounds with PLD (PLD speeds the trace, structured CoT shrinks it 22–43×). Plan locations are nailed in `RECAP-2026-04-26-structured-cot-spike-handoff.md`.
2. **Dtype audit (3 known suspects + sweep)** — `deepseek_v2.rs:219`, `deepseek_v2.rs:625`, `siglip.rs:108` plus a full sweep across qwen2/3/3_next/gemma2/phi3/starcoder2/transformer/llada_moe/diffusion. The Bonsai +2.83× win was exactly this shape; expect more.
3. **A3B DFlash 6× regression diagnosis** — Suspect surviving fp16→f32 upcasts in the A3B drafter forward / verify-tape / engine glue. Same hunting pattern as session 28.
4. **`--features ane` build break** at `bd3lm_qwen3.rs:118` — blocks Point B inline ANE drafter measurement. Single file fix.
5. **PLD `--pld` CLI flag landed but no realistic-prompt characterization curve** (carry-over from s2 #2).

### Hardware-ceiling work
6. **ANE int8 MLP zero-copy** — drop the 6 element-wise transposes per forward in `forward_ane_int8_mlp`. Bucket=512 already aligned with chunked prefill. Layer-0 + parity already landed.
7. **ANE drafter eval_chain** — kill per-dispatch GPU↔CPU fence in GDN drafter. Uncommitted hunks pending split-add.
8. **DFlash int8 weights via `.mlpackage`** — abandons raw-MIL emitter (rejected `tensor<int8>` on macOS 26.3.1). Probe artifacts confirm CoreML scheduler picks ANE at realistic shapes; needs `AneKernel::from_mlpackage` + offline build step + ANE-dispatch verifier.

### Research / speculative
9. **BD3LM denoise-head audit** — wrong base hash? wrong activation index? tied embedding? `mask_emb` scaling? Audit `/Users/peppi/Dev/diffusion_bonsai/` training script. Phase D bench is gated on this.
10. **AR-spec advance-overhead** — ~19 ms/tok flat tax in the AR-spec loop; ANE drafter overlap deferred as multi-day work.
11. **27B DFlash context sweep** — beyond Step 0 (cap landed); BS=2/3 default for trained-16 needs context-dependent override.
12. **Eggroll 35B validation at pop=4/20step** — has not run end-to-end against a live higgs server.

---

## Lessons / dead ends (posterity)

These are the paths that got tried, measured, and ruled out. Save the next reader 10 hours of bisection.

- **Async-eval inside GDN dispatch only moves the wait** — `5425cd34` reverts `14a6bd5b` after measurement. Don't try this again without changing the synchronization point.
- **Pipeline=true unconditional default was a dead end** — `c1f85ade` reverts; `22bf8f15` makes it env-gated `HIGGS_DFLASH_PIPELINE=1`.
- **TEAL on pure-MLX caps at ~5%, not 1.7×** — row-gather on `up_proj` doesn't scale; `gather_qmm` was wishcasting (`RECAP-2026-04-23-session10-teal-dead-pivot.md`).
- **ANE GDN compilation fails at 9B dims** (`flat_w=4096 > ~64 limit`). Topology-B (`HIGGS_TARGET_ANE_GDN=0`) is the practical win at +15.5%; further ANE-GDN-on attempts net-negative.
- **ANE LM-head loses 4.7× at vocab=152K shape** (`C1` benchmark). Don't offload LM-head to ANE on Qwen-class models.
- **Path A (stateless ANE drafter for 27B) is dead** — p6 verify-floor 178 ms flat ≥K=12, projected −88% to −94% break-even on 8B even at α=0.7.
- **Bonsai compile-wrap alone caps at ~1.5×, was actually a regression at 0.84×/0.94×** (session 26). The real win was the dtype bug fix in session 28 — *the "scaffolding gap" hypothesis was wrong; it was an f32 leak in `apply_yarn_rope`*.
- **DFlash drafter parity audit was a wild goose chase** — 10/10 architectural candidates refuted. Real cause was GDN state rollback (Qwen3.5 hybrid SSM layers cannot rollback by offset). Fix `a7e2737c`, regressed by `bee1ee20`.
- **DFlash Python-vs-Rust gap was sample-dependent** — Rust ≈ Python on apples-to-apples; the 28.24 tps was a 3-sample average. Don't average without reporting variance.
- **int8 raw-MIL decode path is fully dead** — `5d159425` documents AB9/AB10/AB11 verdict. Prefill-only is the surviving direction; macOS 26.3.1 rejects `tensor<int8>` for runtime decode.
- **`mlx_rs::array!(f32_value)` silently produces an f32 scalar** even inside an fp16 graph. Use `Array::from_f32(x).as_dtype(input.dtype())` instead. This single mistake cost ~28 ms/step on Bonsai-8B for weeks.

---

## Cleanup candidates

A focused list of `.planning/` files that are clearly superseded. Recommend moving to `.planning/archive/2026-04-cleanup/` rather than deleting (still recoverable via git).

### ANE
- `next-session-ane-9b-parity.md` — RESOLVED in commits `068a14ef` + `c95a80c7`.
- `dflash-ane-projections-handoff.md`, `dflash-ane-projections-v1-handoff.md` — superseded by `v2`.
- `RECAP-2026-04-24-session16-ane-drafter-investigation.md` — superseded by session-34 (3 of 4 hypotheses ruled out).
- `phase1-ane-memory-surgery-plan.md` — superseded by `next-session-phase1-ane-memory-handoff.md` (Phase 1 declared outdated).
- `next-session-ane-synergy-handoff.md` + `next-session-ane-reframe-verification.md` — partially superseded by `ane_c1_sustained_tflops.md` + `ane_g2_dispatch_roundtrip.md`.

### DFlash
- `next-session-dflash-drafter-parity-audit.md` — closed (10/10 refuted; cause is GDN rollback).
- `next-session-dflash-python-parity.md` — closed (Rust ≈ Python).
- `memory/dflash-regression-bee1ee20-handoff.md` — root cause subsumed under broader dtype-upcast hypothesis.
- `benchmarks/dflash_27b_topoB_20260423_225643/dflash_topoB.csv` — header-only, never populated.
- `next-session-27b-dflash-crash.md` (Step 0) — cap shipped; remaining steps belong to A3B/A2 baseline track.

### Diffusion / Bonsai
- `next-session-bd3lm-bonsai.md` → superseded by `next-session-bd3lm-phase-c.md` → `phase-c-bench.md` → `denoise-head-blocker.md`.
- `RECAP-2026-04-24-session{23,24,25,26}-…` — pre-dtype-fix Bonsai work; conclusions stale though numbers in `decode-breakdown.md` remain referenceable.
- `RECAP-2026-04-25-session27-bonsai-scaffolding-gap-isolated.md` — superseded by session-28 (gap was the dtype bug); bisect bench artifact still useful.

### Spec-decode / Magic Canvas
- `RECAP-2026-04-24-session{11,12,13,14}-*.md` — chained handoff sequence, superseded by session-15 (E2E green).
- `RECAP-2026-04-26-session5-arspec-fsm-handoff-shape-bug.md` — hypothesis falsified (root cause was dtype, not shape).
- `RECAP-2026-04-26-session4-dflash-fsm-smoke-arspec-plan.md` — plan executed in session-7.
- `RECAP-2026-04-26-pld-fsm-landed.md` — smoke completed; CLI fix landed.
- `.planning/measurements/session-14-spec-decode-K12.md` — pre-tokenizer-mismatch data.

**Quick action.** ~25 files identified above. Suggest `mkdir .planning/archive/2026-04-cleanup && git mv <list>` in a single PR titled `chore(planning): archive superseded handoffs`. Keep a one-line `INDEX.md` in the archive so future searches don't go cold.

---

## Proposed 4-week plan (2026-04-27 → 2026-05-25)

The plan is shaped by what's ready vs what's blocked, and biases toward landings that produce shippable user-facing wins (Magic Canvas pitch) while clearing the technical-debt overhang from five weeks of velocity.

### Week 1 — close the dtype frontier, ship the Magic Canvas spike (2026-04-27 → 2026-05-03)

**Goal:** convert the last week's diagnostic wins into a shippable release-candidate, plus land the user-facing structured-CoT/PLD demo.

- **Day 1–2: Dtype sweep.** Land the 3 known suspects (`deepseek_v2.rs:219`, `deepseek_v2.rs:625`, `siglip.rs:108`) with bisect-based verification per session-28 method. Sweep qwen2/3/3_next/gemma2/phi3/starcoder2/transformer/llada_moe. Target: any model with > 5% f32 leak in decode. Expected impact: 1.1–2× on at least one more model class.
- **Day 2–3: A3B DFlash regression diagnosis.** Apply the same hunting pattern to drafter forward / verify-tape / engine glue. If dtype-clean, suspect KV layout difference for MoE. Goal: reach AR parity (43.83 tps) with DFlash on, even before net-positive.
- **Day 3–4: Structured CoT spike (30 LOC).** Land env-gated `HIGGS_STRUCTURED_THINK=1` per `RECAP-2026-04-26-structured-cot-spike-handoff.md`. Demo on Carnice-9B with PLD. Expected demo number: PLD 1.84× × structured CoT trace shrink ≈ **3–8× wall-clock on think-heavy prompts**.
- **Day 5: Cleanup PR.** Move ~25 superseded files into `.planning/archive/2026-04-cleanup/`. Single PR, no logic changes.
- **Continuous: PLD characterization curve.** Run PLD across realistic-prompt classes (code, JSON, prose, RAG) to publish a "when does PLD win" table for README.

### Week 2 — ANE int8 prefill ship + DFlash int8 weights (2026-05-04 → 2026-05-10)

**Goal:** make ANE pay rent for the first time on the magic-canvas branch.

- **Day 1–2: Fix `--features ane` build break** at `bd3lm_qwen3.rs:118`. Re-run Point B inline ANE drafter measurement. Decide: is the 8.7 tps from session 11 still the gate?
- **Day 2–4: ANE int8 MLP zero-copy.** Eliminate the 6 element-wise f32↔fp16 transposes per forward. Layer-0 + bucket=512 already landed. Target: cut the 18.5 ms ANE projection floor to ≤14 ms. End-to-end at 30 k+ context.
- **Day 4–5: DFlash int8 weights via `.mlpackage`.** Add `AneKernel::from_mlpackage` + offline build step + ANE-dispatch verifier (mactop or `IODeviceTree:/AppleH13`). Validate on DFlash-4B realistic shape.
- **Day 5–6: AR-spec K=2..3 default ship.** Push `38d33810` to fork/main as a release candidate. Update README + benchmarks. Tag `higgs-vX.Y` once dtype sweep + structured CoT are also in.

### Week 3 — BD3LM blocker resolution + ANE drafter eval-chain (2026-05-11 → 2026-05-17)

**Goal:** unblock or formally kill BD3LM. Get the ANE drafter past its 39 ms transfer mystery.

- **Day 1–3: BD3LM denoise-head audit.**
  1. Audit `/Users/peppi/Dev/diffusion_bonsai/` training script for activation index 2 (GELU vs SiLU), embedding tying status, mask_emb scaling, base hash.
  2. If a config mismatch: fix Rust + rerun Phase D bench (target: PPL 16.2 ± 0.5, 5.5× algorithmic vs AR).
  3. If structurally broken: write `dead-end-bd3lm-bonsai.md`, archive, free the BF16 weights from disk.
- **Day 3–5: ANE drafter eval_chain landing.** Split-add the uncommitted hunks; kill the per-dispatch GPU↔CPU fence. Re-run inline ANE drafter on K=12 27B target. Question: does the 39 ms transfer collapse?
- **Day 5–6: 27B DFlash context-dependent block size.** Address the BS=2/3 default for trained-16 with a context-dependent override. Re-run 27B BS=12 sweep with the new policy.
- **Continuous: AR-spec advance-overhead.** Profile the ~19 ms/tok flat tax. Cheap wins likely (string-allocation, regex re-compile).

### Week 4 — Re-baseline, rebase, release (2026-05-18 → 2026-05-25)

**Goal:** stop accumulating, start shipping. Rebase magic-canvas onto current `main` (now at upstream `4e745855` after this morning's sync) and cut a release.

- **Day 1: Bench matrix re-run.** Fix `next-session-bench-matrix-dflash-handoff.md`'s `start_server` stderr issue. Re-run the full matrix across 4B/9B/27B/A3B with all this month's wins on (dtype fixes, K=2..3, PLD, structured CoT, ANE int8 if shipped). Publish a single `BENCHMARKS-2026-05-MAGIC-CANVAS.md`.
- **Day 2–3: Rebase magic-canvas onto main.** 113 upstream commits since branch point. Use `--rebase-merges` or interactive rebase; expect conflicts in `qwen3_next.rs`, `simple.rs`, `diffusion.rs`. Run full `cargo test --features ane` after.
- **Day 3–4: Release prep.** Update `README.md` headline numbers (Bonsai-8B 64 tok/s, Carnice-9B PLD 1.84×, AR-spec +30%). Doctor checks for new fields. `higgs init` template touch-ups. Changelog generation.
- **Day 4–5: Cut release.** Tag, push, write release notes. Highlight the Magic Canvas + spec-decode story arc.
- **Day 5: Retro.** What worked (dtype audits, sub-agent inventory, daily RECAP discipline). What didn't (DFlash A3B 6× silently sat for days; BD3LM Python-parity discovery should have been week-1 material). Capture in `.planning/RETRO-2026-05.md`.

### Stretch goals (if Week 1–2 finish ahead)
- **Eggroll 35B end-to-end at pop=4/20step.** Validation harness already in `scripts/`. Speculative but cheap to test.
- **DFlash temperature-runtime-policy.** Convert the 9B temp-sweep data into a runtime guidance flag (`HIGGS_DFLASH_TEMP_POLICY=accept_curve`).
- **`gitnexus_rename` audit pass.** Rename the most-overloaded function names (e.g. `forward_with_taps_tape` → `verify_tape_forward`) in a single safe-rename PR.

### Explicit non-goals for the next 4 weeks
- Going back to Bonsai compile-wrap (session-26 verdict stands).
- ANE LM-head offload (4.7× loss measured).
- Async-eval inside GDN dispatch (session `5425cd34` revert stands).
- Pure-MLX TEAL (session-10 dead-pivot stands).

---

## Notes for posterity (human / AI)

If you're reading this cold:

1. **Start with `01-timeline.md`'s Top-10 milestones**; they reconstruct the narrative arc in 10 lines.
2. **Then read the "Lessons / dead ends" section above**; it's the highest information density per word in this document.
3. **Don't trust handoff `RECAP-…` files in isolation.** They reflect the state at end-of-session and are often wrong by the next day. Cross-reference against the latest `next-session-*.md` for the same area, then check git log.
4. **The single source of truth for a measurement is the `.log`/`.json` file under `.planning/measurements/` or `benchmarks/`**. RECAPs paraphrase; they drift.
5. **Trace schemas** are stable across the branch — `dflash_trace round= embed= draft= …` and `spec_decode: cycle …` lines are greppable across all `.out` files.
6. **The Bonsai-Q1 dtype fix (commit `1712b9ab`) is the canonical "five weeks of work, eight chars of fix" anecdote for this branch.** When in doubt, suspect dtype.

---

*Generated by 5 parallel research agents over `feat/magic-canvas`, synthesized 2026-04-26. Sources: 115 commits (43eedc6b..HEAD), per-domain inventories at `.planning/inventory-2026-04-26/`.*
