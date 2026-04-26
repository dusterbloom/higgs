# Commit Timeline — feat/magic-canvas (43eedc6b..HEAD)

## Headline numbers

- **Total commits**: 115 (no-merges) on `feat/magic-canvas` since branch point at `43eedc6b`.
- **Date range**: 2026-03-23 → 2026-04-26 (35 calendar days, ~5 weeks of active work).
- **Distinct authors**: 2 — Peppi Littera (primary) and `Claude` (3 commits, all on 2026-03-24, addressing PR #60/#61 review findings).
- **Top scope prefixes by count**:
  - `fix` (untyped): 10 — broad fixes touching loaders, configs, regression patches.
  - `feat(models)`: 7 — diffusion, AR spec-decode, Wave 1 GDN offload, forward_all_logits, etc.
  - `perf(turboquant)`: 6 — Metal kernels, packed-word decode, prefill regression fixes.
  - `feat(speculative)`: 6 — DraftModel trait, ANE drafter, SimpleEngine wiring, tracing.
  - `perf(models)`: 5 — Conv1d fast paths, MoE gate+up fusion, target compile.
  - `perf(ane)`: 5 — int8 MLP path, GDN rowwise MIL, async_eval probes.
  - `feat(spec-decode)`: 5 — IOSurface ANE drafter, FSM-aware verify, AR-spec.
  - `feat(ane)`: 5 — int8 MLP layer-0 prefill, lm_head offload scaffolding, mlpackage bridge.
  - Long tail: `bench(bonsai-q1)` ×3, `feat(bonsai-q1)` ×4, `feat(diffusion)` ×2, `feat(dflash)` ×2, `feat(engine)` ×2, `wip` ×2, `refactor` ×2, `docs(planning)` ×2.
- **Daily peak**: 20 commits on 2026-04-24 (Bonsai-Q1 P1→P6 + spec-decode wiring landings). Second peak: 9 commits on both 2026-04-09 and 2026-04-26.

## Weekly chronology

- **Week of 2026-03-23** (12 commits, Mon–Wed): Branch opens with `qwen3_5_moe` model support (`2a2caaf4`), GDN projection 4→2 fusion (`fea46046`), dense FFN qwen3_5 alongside MoE (`70c504c2`), and the first speculative-decode infrastructure scaffold (`beb95d59`). Also lands a stack of upstream PR #60/#61 review fixes (`8f6a93b1`, `c264ef92`, `da878a8b`).
- **Week of 2026-03-23, late** (2026-03-25 → 2026-03-28, 14 commits): Performance push — quantized embed gather (`79694606`, ~1.8× decode), MLX-C 0.4.0 repin restores 60 tok/s (`baafddd5`), paged prefix cache (`826794b0`), causal-mask enum + chunked prefill (`cf0389db`), MoE batch sort 4-5× TTFT (`bba6c82e`), then the TurboQuant kernel run: 3-bit pack fix (`9f4e0fd2`), u32 packed words for decode (`d969879a`), deferred bulk quant (`994cc2ff`), GQA-fused score kernel enabling TQ on Qwen3Next (`3b941a6c`).
- **Week of 2026-03-30 → 2026-04-05** (4 commits): Diffusion + AR spec-decode foundation lands. `89ab5fdc` adds diffusion, ANE bridge, RWKV-7, LLaDA-MoE modules. `de5d1552` wires the AR speculative decode path (0.8B draft → 27B verify) and `2c5a44d2` adds the adaptive-K controller. `a79b904c` ships forward_last + deferred save reaching 12.7 tok/s.
- **Week of 2026-04-06 → 2026-04-12** (12 commits): Diffusion "Magic Canvas" theme starts to surface — `9917adcc` Magic Canvas killer tests + Qwen2/Qwen3 A2D loader fixes; `648c4251` Qwen2.5-Coder A2D on ANE. Heavy Tuesday landing on 2026-04-09: forward_all_logits (`06aa3199`), fused MoE gate+up 3→2 matmuls (`8c56888d`), Conv1d batch-K fast path (`267bf791`), TQ block-L Metal kernels (`517fd6e7`), 64-byte ANE alignment (`144ac21d`). 2026-04-10 lifts MLX memory caps for 5× decode (`e5c47264`).
- **Week of 2026-04-13 → 2026-04-19** (24 commits): GDN ANE offload waves and DFlash plumbing. Wave 1/2/4 GDN offload (`b72412a2`, `cddcb2a1`, `d5c20025`), worker-thread realtime eval (`fb54b77e`), 9B parity tiled-matmul fix (`068a14ef`), down_proj/up_proj fp16 saturation fixes (`c95a80c7`, `2de57ce5`), DFlash rejection sampling for T>0 (`782360c0`), pipeline env-gate (`22bf8f15`), and the int8 MLP run-up: prefill probe wins MLX q4 by 2.23× (`92f91f59`), int8 MLP SwiGLU 3.64× baseline (`8467d558`).
- **Week of 2026-04-20 → 2026-04-26** (43 commits — the busiest): Spec-decode and Bonsai-Q1 dominate. 2026-04-23 lands persistent caches + threaded pipeline + ANE drafter (`2e36b777`, `133ad797`). 2026-04-24 ships the DraftModel trait port (`1161b844`), AneBonsaiDraftModel (`2121953b`), draft_model config + doctor validation (`054494c2`, `feef8e47`), SimpleEngine wiring (`5108e8db`), tokenizer hash gate (`835dc291`), per-cycle tracing (`b285c2ca`), and end-to-end spec-decode on 27B + Bonsai-1.7B (`1cee9bd6`). Same day: Bonsai-Q1 P1→P6 (packed 1.25-bpw engine `aad4aeea`, bits=1 oracle `15c12e2c`, causal forward+KV `17bc471b`, AnyModel variant `9e0ea6b7`, forward_all_logits `1c8e6464`, AnyModel matrix bench `446b96cb`, P6 verify-cost probe `dd9b73bc`, dynamic rope+KV pre-alloc `a012c847`). 2026-04-25 lands the dtype fix: fp16 attention path → 2.7× decode on Bonsai-8B (`1712b9ab`) plus 8 hot-path upcast kills (`2de6ad03`). 2026-04-26 closes with PLD landing (`2761a7ce`), FSM-aware verify (`e72b5dee`, `394a79e2`, `73777ab4`), on-device draft-token verify (`d0a8d73a`), and AR-spec K=2..3 default (`38d33810`, +11-30% tps).

## Themes

### spec-decode (AR-spec, FSM verify, PLD, DraftModel trait)
- **Count**: ~22 commits. **Date range**: 2026-03-23 → 2026-04-26 (active through HEAD).
- **Status**: ACTIVE — most recent landings on the branch.
- Key commits:
  - `beb95d59` 2026-03-23 — speculative decode infrastructure (draft + verify loop), the original scaffold.
  - `de5d1552` 2026-04-03 — AR speculative decode 0.8B → 27B with cache reuse.
  - `1161b844` 2026-04-24 — port DraftModel trait + spec-decode core.
  - `1cee9bd6` 2026-04-24 — end-to-end spec-decode on 27B + Bonsai-1.7B.
  - `2761a7ce` 2026-04-26 — PLD: config, CLI flags, doctor checks, README, +14/9 unit tests.
  - `73777ab4` 2026-04-26 — AR-spec FSM-aware verify (Option A).
  - `38d33810` 2026-04-26 — default K window to 2..3, +11-30% tps.

### ANE (drafter, prefill, GDN, projections, lm_head)
- **Count**: ~20 commits. **Date range**: 2026-03-25 → 2026-04-26.
- **Status**: ACTIVE — recent FSM-verify lands depend on it; lm_head + int8 MLP probes paused after evidence.
- Key commits:
  - `b72412a2` 2026-04-14 — Wave 1: full GDN layer ANE offload (qkvz+ba+out_proj).
  - `cddcb2a1` 2026-04-15 — Wave 2: all 24 GDN layers via patch_from_donor.
  - `d5c20025` 2026-04-15 — Wave 4: GDN ANE worker thread (Send+Sync handle).
  - `92f91f59` 2026-04-18 — qwen3_9b prefill probe: ANE int8 wins MLX q4 by 2.23× (GREEN).
  - `8467d558` 2026-04-19 — int8 MLP SwiGLU native [1,inter,1,bucket] layout, 3.64× baseline.
  - `116c2f61` 2026-04-20 — GDN rowwise MIL output, +9% tok/s.
  - `133ad797` 2026-04-23 — inline IOSurface ANE drafter path + tunable test knobs.
  - `2121953b` 2026-04-24 — AneBonsaiDraftModel over native AneBonsaiEngine.

### DFlash (probes, parity, regressions)
- **Count**: ~8 commits. **Date range**: 2026-04-14 → 2026-04-26.
- **Status**: ACTIVE — FSM-aware verify on DFlash gate just lifted at HEAD.
- Key commits:
  - `068a14ef` 2026-04-14 — ANE DFlash 9B parity (tiled matmul + silu rewire).
  - `d6daf3e0` 2026-04-14 — GDN-only tape replay + batched Metal kernel for DFlash.
  - `c1f85ade` 2026-04-16 — revert pipeline=true, add verify_build_ms timer + 9b sweep.
  - `782360c0` 2026-04-16 — wire rejection sampling for temperature>0.
  - `394a79e2` 2026-04-26 — FSM-aware verify in DFlash (gate lifted at simple.rs:1083).

### Diffusion / BD3LM / Bonsai / Eggroll / Magic Canvas
- **Count**: ~13 commits (Bonsai-Q1 = 8, diffusion/A2D = 3, Magic Canvas tests = 1, related modules = 1).
- **Date range**: 2026-04-03 → 2026-04-25. **Status**: ACTIVE — Bonsai-Q1 just demonstrated 2.7× decode.
- Key commits:
  - `89ab5fdc` 2026-04-03 — diffusion, ANE bridge, RWKV-7, LLaDA-MoE modules.
  - `9917adcc` 2026-04-08 — Magic Canvas killer tests + Qwen2/Qwen3 A2D load fixes.
  - `648c4251` 2026-04-09 — Qwen2.5-Coder A2D on ANE: bias, untied lm_head, seq alignment.
  - `aad4aeea` 2026-04-24 — packed 1.25-bpw engine type (Bonsai P1).
  - `1712b9ab` 2026-04-25 — keep attention path in fp16, 2.7× decode speedup on 8B.

### TurboQuant / quantization
- **Count**: 8 commits. **Date range**: 2026-03-26 → 2026-04-09.
- **Status**: LANDED — block-K parity tests merged 2026-04-09; no activity after.
- Key commits:
  - `9f4e0fd2` 2026-03-27 — 3-bit pack_indices corruption fix + correctness tests.
  - `d969879a` 2026-03-27 — u32 packed words for decode kernels (4× fewer memory loads).
  - `994cc2ff` 2026-03-28 — deferred bulk quantization (no packing during prefill).
  - `3b941a6c` 2026-03-28 — GQA-fused score kernel, enable TurboQuant on Qwen3Next.
  - `517fd6e7` 2026-04-09 — block-L Metal kernels for TQ score+value.

### Qwen3-next / Qwen3.5 / Qwen3.6 model work
- **Count**: ~10 commits. **Date range**: 2026-03-23 → 2026-04-26.
- **Status**: STABLE — most recent is the dtype-coerce fix at HEAD.
- Key commits:
  - `2a2caaf4` 2026-03-23 — qwen3_5_moe support (VLM wrapper around qwen3_next).
  - `70c504c2` 2026-03-23 — dense FFN qwen3_5 alongside MoE.
  - `fea46046` 2026-03-23 — fuse GDN projections 4→2 with row permutation.
  - `061e500c` 2026-04-23 — handle mixed-bit Qwen3.5 GDN BA loading.
  - `26862aef` 2026-04-26 — coerce conv1d.weight dtype on S=1 native path.

### MoE perf
- **Count**: 3 commits. **Date range**: 2026-03-26 → 2026-04-09. **Status**: LANDED.
- Key: `bba6c82e` (global batch sort, 4-5× TTFT), `8c56888d` (fused gate+up 3→2 matmuls), `f981d984` (LESSONS.md + benches).

### Engine plumbing (cache, chunked prefill, streaming, tools, doctor)
- **Count**: ~10 commits. **Date range**: 2026-03-25 → 2026-04-26. **Status**: STABLE.
- Key: `826794b0` paged prefix cache; `d24e4a92` chunked prefill OOM fix at 24K+; `e5c47264` MLX memory caps removed → 5× decode; `fb48230c` AnyCache::trim_by; `feef8e47` doctor validates draft_model; `339612ae` lift HIGGS_MLX_CAP_FRACTION env-var gate to all models.

### CI / release / chore / docs
- **Count**: ~7 commits. Mostly bench-script collection (`c3d7a4bc`), AGENTS/CLAUDE updates (`d92949d4`), planning recaps (`358340b3`, `96d3ee20`, `e7214f85`), and gitnexus auto-stats refresh (`dd56cd62`). Continuous, low-volume.

## Top-10 milestone commits

1. `beb95d59` — feat: speculative decode infrastructure (draft + verify loop) — the original scaffold the entire spec-decode arc was built on.
2. `de5d1552` — feat(models): AR speculative decode 0.8B → 27B with cache reuse — first end-to-end speculative decode through the production stack.
3. `89ab5fdc` — feat(models): add diffusion, ANE bridge, RWKV-7, LLaDA-MoE modules — defines the model-architecture surface this branch is named after ("magic canvas").
4. `e5c47264` — perf(engine): remove MLX memory limits — 5× decode throughput fix — the single largest decode-throughput win on the branch.
5. `b72412a2` — feat(models): Wave 1 — full GDN layer ANE offload (qkvz + ba + out_proj) — opens the GDN-on-ANE pipeline that everything downstream depends on.
6. `92f91f59` — feat(ane): qwen3_9b prefill probe — ANE int8 wins MLX q4 by 2.23× (GREEN) — go/no-go evidence that ANE int8 prefill is real.
7. `1161b844` — feat(speculative): port DraftModel trait + spec-decode core — replaces the prototype with the production trait the rest of the engine consumes.
8. `1cee9bd6` — fix(speculative): end-to-end spec-decode on 27B + Bonsai-1.7B — the integration bring-up that proves the whole stack works on real targets.
9. `1712b9ab` — perf(bonsai-q1): keep attention path in fp16 — 2.7× decode speedup on 8B — the dtype-truth diagnosis that unlocks Bonsai-Q1's headline number.
10. `73777ab4` — feat(spec-decode): AR-spec FSM-aware verify (Option A) — the FSM verify landing that lets PLD compound with constrained decoding (current frontier of the branch).

## Reverts / dead-ends / superseded work

- **Explicit revert**: `5425cd34` 2026-04-20 — `Revert "perf(ane): try async_eval inside GDN dispatch — moves wait, doesn't cut it"`, reverting `14a6bd5b` from the same day. Async_eval inside GDN dispatch was tried, measured, and reversed once it became clear it only moved the wait point.
- **Pipeline-mode flip-flop**: `c1f85ade` 2026-04-16 explicitly *reverts* an earlier `pipeline=true` setting and adds `verify_build_ms` timing. Subsequent `22bf8f15` makes pipeline mode env-gated via `HIGGS_DFLASH_PIPELINE=1` — i.e., the unconditional pipeline default was a dead end and the work moved to opt-in.
- **Speculative dead code purge**: `1cea874f` 2026-03-25 — `refactor: strip speculative decode dead code` — clears out the original speculative scaffold ahead of the more thorough rebuild that lands later in `1161b844`.
- **WIP checkpoints superseded by later landings**: `4b5fc6c7` 2026-04-24 (`wip(magic-canvas): checkpoint inherited session 9-11 state`) and `bee1ee20` 2026-04-16 (`wip(ane): lm_head + GDN + DFlash wiring — E2E validated, DFlash regresses`). Both are explicitly WIP; the latter is superseded by `e318efa0` (proper `HIGGS_TARGET_ANE_LM_HEAD=1` scaffold) and the subsequent DFlash regression work.
- **Probes that documented "dead" paths**: `5d159425` 2026-04-18 — `docs(ane): AB9/AB10/AB11 — int8 raw-MIL fully dead, prefill-only verdict` — int8 raw-MIL decode path was killed off after evidence; prefill-only is the surviving direction.
- **PR #60/#61 review-fix churn**: `c264ef92`, `da878a8b`, `e884fc67`, `2d438aa9`, `8f6a93b1`, `6a18af07` — six small fixes across 2026-03-24/25 chasing review comments on upstream PRs; not reverts but iterative cleanups whose effect is partially superseded by later refactors of the same code paths.
