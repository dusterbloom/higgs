# Git Archaeology — ANE / DFlash / GDN Timeline

**Method:** `git log --follow` on ANE/DFlash source files, filtered to commits touching these paths on all branches since 2026-01-01. Commit messages are treated as **Tier D** (documented by the engineer at the time of commit). Measured numbers embedded in commit messages are **Tier E** when accompanied by a reproducible bench, otherwise downgraded to D.

**Phase 2 purpose:** anchor CLAIMS.md claims to specific commits, surface evidence that only lives in git, and build a chronological story of what was tried vs what stuck.

---

## Commit Chain (chronological, oldest → newest)

### Foundation — ANE bridge + BLOBFILE era

| Commit | Headline | Evidence |
|---|---|---|
| `89ab5fdc` | feat(models): add diffusion, ANE bridge, RWKV-7, LLaDA-MoE | Single-import commit for `ane_bridge.rs`, `ane_extract.rs`, `ane_forward.rs`. Entry point for the whole ANE subsystem. |
| `0f5bd628` | Consolidate causal ANE path: 28-kernel BLOBFILE, 3.1× BLAS speedup | **E:** Qwen3 0.6B Base ANE 83.6 tok/s (3.1× over BLAS 26.7); 28 ANE dispatches @ ~19ms for seq=128 (was 840ms with `reload_weights`). **Lesson:** weights-baked-as-BLOBFILEs is **30× faster** than reload_weights. |
| `2116d9dc` | Prototype chained Bonsai ANE tiled FFN path | Earliest tiled FFN on ANE. |
| `0c283c3b`→`8e3ae3dc` (4 commits) | Bonsai ANE hybrid split policy search | **D:** layers 14-27 on CPU for causal path. Hybrid split was a **policy search**, not a derived answer — implies the optimal split depends on model+config and must be empirically tuned. |
| `a79b904c` | forward_last + deferred save for ANE spec decode | **E:** 10.8→12.7 tok/s (+17.6%); draft time 916→613 ms (−33%). **Lesson:** 1×vocab LM-head matmul saves ~15 ms/intermediate call vs seq×vocab. |
| `e074c9b4` | feat: speculative decode infrastructure | **E-neg:** No speedup on MoE+GDN. Batch verify can't amortize because GDN is sequential and MoE loads different expert weights per token. Infrastructure shipped anyway for dense-attention models. **Lesson:** classic spec decode is structurally wrong for MoE+SSM. |

### GDN-on-ANE rollout (Waves 1-4)

| Commit | Headline | Evidence |
|---|---|---|
| `b72412a2` | Wave 1 — full GDN layer ANE offload (qkvz + ba + out_proj) | **E:** Qwen3.5-4B-MLX-bf16: max_diff 0.003906, mean 0.000386 (budget 0.05). Parity proven on real BF16 checkpoint. |
| `cddcb2a1` | Wave 2 — all 24 GDN layers via `patch_from_donor` | **E:** Carnice-9B (32 layers, 24 linear, hidden=4096). 1 donor compile (2.9s) + 23 patches (~107s). `load_count 0→3` — `patch_from_donor` **reuses microcode** without bumping `loadWithQoS`. **ANE program load cap: 119 per process.** Per-layer parity within 1% bf16 relative budget. |
| `d5c20025` | Wave 4 — GDN ANE worker thread (Send+Sync handle) | **D:** `AneKernel`/`AneProjKernel` never cross a thread. Handle = mpsc Sender clone. **E:** 0.12 ms/dispatch @ 1000-round synthetic stress. Parity <1% per-layer. |
| `ed01ae33` | Wire `HIGGS_TARGET_ANE_GDN=1` in model_loader | **D:** flag was **dead in production** pre-Wave 4 — `AneKernel` `!Send` blocked it. Wave 4's worker unblocks. Smoke-tested: coherent output, no NaN. |
| `144ac21d` | Enforce 64-byte spatial alignment in all ANE MIL generators | **E:** seq=137 fails universally with `status=0x1d`; seq=64/96/128/160 succeed across 3 model configs. **Rule: last axis must be 32-element (fp16) / 64-byte aligned.** Structural fix — protects spec decode, diffusion, Bonsai hybrid. |

### ANE Prefill Engine (dense path — answers the 4B story)

| Commit | Headline | Evidence |
|---|---|---|
| `e7172aa6` | ANE GDN prefill engine — multi-dispatch, cos=0.99999 | **E:** Verified cos=0.999991 on real 35B-A3B-3bit (layer 0). **But: ANE I/O overhead dominates — 38-68 ms vs GPU 3-17 ms.** Phase 2 promised to overlap ANE+GPU to hide this. Buckets [512, 1024, 2048]. |
| `cd78076c` | ANE GDN prefill zero-copy I/O — 7-35% faster projections | **E:** 7% faster @ seq=128, 35% faster @ seq=512. Direct IOSurface read/write via `get_input_base` / `get_output_base`. Shared input between qkvz + ba (copy_nonoverlapping). Pre-allocated scratch. **Lesson:** ANE-side compute was never the bottleneck; I/O path was. |
| `267bf791` | Conv1d batch-K fast path for S≤32 in GDN layers | **D:** extends S==1 element-wise fast path to S≤32 via sliding window loop. Avoids Conv1d kernel dispatch for all 30 GDN layers during block-K verify. S>32 falls through to native Conv1d. |

### DFlash evolution (the 38% regression story)

| Commit | Headline | Evidence |
|---|---|---|
| `a7e2737c` | fix(engine): replay accepted tokens through GDN on partial accept | **E:** Acceptance 2.1→3.4 tok/round (matches Python aryagm/dflash-mlx's 3.33). Token parity for first 20+ generated. 11.7→17.7 tok/s. **Bug:** state rolled back but not advanced through accepted tokens → cascading GDN state corruption. **Lesson:** every accepted token must be replayed through both KV and GDN. |
| `d6daf3e0` | perf(engine): GDN-only tape replay + batched Metal kernel | **E:** Full-model rerun on rejection replaced by SSM recurrence replay: ~5 ms vs ~30 ms/round. Batched 24 GDN layers into single Metal kernel (was 24 × ~0.4 ms). **E:** Acceptance scales with length — 3.2 @ 65 tok → 12.7 @ 2048 tok, reaching 26.3 tok/s (2.7× over AR on 120 GB/s). |
| `068a14ef` | ANE DFlash 9B parity — tiled matmul + silu rewire | **E:** **ANE MIL concat on axis=3 silently produces NaN.** Only axis=1 (channel) concat reliable. New `emit_blobfile_matmul_tiled` helper — per-tile matmul, transpose each to channel-first, concat on axis=1. Single-tile path produces same channel-first layout for callers. 4B: max_diff 0.034, 0 NaN/Inf. 9B: 65531/65536 finite (5 Infs in down_proj, inter=12288). Also clamps bf16→fp16 at ±65504 in `ane_bridge.m`. |
| `c95a80c7` | Scale down_proj weights to avoid fp16 saturation on 9B | **E:** 9B `inter=12288` pushes down_proj output past fp16 max → Inf propagates through residual → RMSNorm `sum_sq=Inf → scale=0` → whole block collapses to 4095 zeros + 1 NaN. Fix: halve weights at compile, restore ×2 in fp32 after `ane_to_cpu`. 9B: max_diff 0.082, 0 NaN/Inf (was all-NaN). |
| `2de57ce5` | Scale up_proj for ANE fp16 — mirror of down_proj fix | **E:** fused silu_gate_up runs entirely in fp16. `silu * um` can saturate on 9B. Halve up_proj; compose with existing 4.0× unscale at down readback (0.5 × 0.5 × 4 = 1). **Gate deliberately not scaled: silu(0.5x) ≠ 0.5·silu(x).** 4B: 0.033→0.0328; 9B: 0.082→0.0778. |
| `e893d465` | Fix: ANE worker dies after first round on Drop send | **E:** `DFlashAneWorkerHandle::drop` unconditionally sent Shutdown. Handle is Clone; simple.rs clones into per-round bg thread; first clone dropped → worker exited → next round panicked `SendError` → 500 after 2 rounds + poisoned lock. **Fix: remove Drop entirely.** mpsc cleanup on last tx drop is canonical. Validated: 40-round request HTTP 200 with zero panics. |
| `e318efa0` | Scaffold lm_head-on-ANE offload (HIGGS_TARGET_ANE_LM_HEAD=1) | **E:** NET-ZERO on Qwen3.5-4B-MLX-4bit. Compile fails with "ANE multi-weight compilation failed" at [vocab=248320, hidden=2560] — fp16 weights ~1.25 GB across ~130 tiles **exceed ANE per-kernel microcode budget**. shipstuff's LUT6 palettization is the unlock. Graceful fallback to Metal. |
| `fb54b77e` | feat(ane-gdn): eval_realtime in worker thread (P0.8 Stage 1) | **D:** `eval_realtime` is per-thread; worker IS a thread → pairing legal. Partial P0.8 — mpsc path still crosses threads; Stage 2 = inline on inference thread. Fixed 6 pre-existing stale `replay_tape_rollback` callsites in dflash.rs. |
| `bee1ee20` | WIP: lm_head + GDN + DFlash wiring — E2E validated, DFlash regresses | **E on Carnice-9b-MLX:** GDN-ANE only: 21.17 tok/s (28.3s, 600 tok). GDN-ANE + DFlash: 13.12 tok/s (45.9s, 602 tok). **38% regression confirmed.** 4B drafter has tied embeddings AND hidden=2560 (mismatches 9B drafter 4096) → needs Qwen3.5-4B-DFlash drafter (hidden=2560, 5 layers). |
| `c1f85ade` | revert pipeline=true, add verify_build_ms timer + 9b sweep | **E (root cause of 38% regression):** with `HIGGS_TARGET_ANE_GDN=1`, `forward_with_taps_tape` is **NOT lazy** — 72 blocking ANE dispatches/verify. Previous `t_verify_fwd` timer only started **after** this function returned → 228-280 ms was **invisible**. Pipeline=true regressed further (13→8.7 tok/s): CPU/ANE drafter slower than lazy GPU drafter, both contend for ANE queue, accept rate 5.5→4.0. **Full breakdown (block=16):** draft 0.1ms · lm_draft 53ms · verify_build 280ms · verify_fwd 28ms · replay 2ms · round_total 311ms · avg_accept 5.5/16. **Ceiling math:** perfect accept = 41 tok/s; 34% accept = 14 tok/s. **Acceptance rate is the bottleneck, not verify speed.** |
| `22bf8f15` | env-gate pipeline mode via HIGGS_DFLASH_PIPELINE | **E — FIRST DFLASH WIN:** Topology B (GDN=0, pipeline=0, BS=16) = **22.49 tok/s vs 19.46 AR baseline (+15.5%)**. Pipeline=true collapses accept 5.94 → 3.85 (BS=16) and 1.08 (BS=8) due to stale context. |
| `782360c0` | Wire rejection sampling for temperature>0 | **E:** Greedy unchanged (temp=0: 22.98/23.20/23.21 tok/s over 3 reps vs 22.49 baseline). Temp=0.7: 5.2/16 accept (~33%), 22.3-34.4 eff_tps, no NaN. `accept_prefix_rs` uses `min(1, p/q)` residual resampling. |

---

## The `feat/ane-prefill` branch (separate lineage)

Git confirms: `feat/ane-prefill` is **ahead of main** with the following ANE prefill work that never landed:

```
ba27b23a perf(engine): adaptive prefill chunk size — scale by model/GPU headroom
cdb8416c deps: bump mlx-c to v0.6.0+ (MLX v0.31.1) via local mlx-rs clone
cd78076c perf(models): ANE GDN prefill zero-copy I/O — 7-35% faster projections
e7172aa6 feat(models): ANE GDN prefill engine — multi-dispatch projections with cos=0.99999
a79b904c perf(models): forward_last + deferred save for ANE speculative decode — 12.7 tok/s
2c5a44d2 feat(models): adaptive K controller for AR speculative decode
de5d1552 feat(models): AR speculative decode — 0.8B draft → 27B verify with cache reuse
```

Unique files on the branch:
- `crates/higgs-models/src/ane_gdn_prefill.rs` — **NOT on main.** Self-contained GDN prefill engine.
- `crates/higgs-models/src/diffusion_ane_bwd.rs` — diffusion ANE backward kernels.
- `scripts/graft_mtp_weights.py` — MTP weight grafting.

Also: `cdb8416c` bumps to **MLX v0.31.1** via a local mlx-rs clone (vs `main`'s pinned `af21d79` → MLX-C 0.4.0). **X2 implication:** the prefill work assumed a newer MLX than ships on main.

**Status:** stranded. `ane_gdn_prefill.rs` never merged because the wire-through on main went via `HIGGS_TARGET_ANE_GDN` (the eager 72-dispatch path that regresses) rather than the prefill-specific lazy multi-dispatch engine.

### DSB SY barrier (branch-only ANE quirk)

`ane_gdn_prefill.rs:33-40` shows:

```rust
/// ARM64 data synchronization barrier — ensures stores are visible to ANE DMA
/// (write→eval) and DMA results are visible to CPU (eval→read).
#[inline(always)]
fn dsb_sy() { unsafe { std::arch::asm!("dsb sy", options(nostack, preserves_flags)); } }
```

**AQ8 (D):** ANE DMA visibility requires an ARM64 `dsb sy` full-system barrier around eval boundaries. Not on main. The comment implies cases existed where IOSurface writes *looked* committed to the CPU but the ANE read stale data (or vice versa). This is kernel-level driver behavior not surfaced anywhere in Apple's CoreML docs.

---

## Undocumented ANE Quirks — The GDN-Recurrence 5-Bug Gauntlet (2026-04-17)

**Source:** session transcripts `8126e869`, `a30877b1`, `b1f6d060` (2026-04-17), during `test_gdn_recurrence_ane_parity` debugging. All 5 bugs FIXED and encoded in `crates/higgs-models/src/ane_mil.rs` (~line 1457 region). Each is Tier E — surfaced by isolating a minimal parity reproducer and bisecting.

These quirks would be invisible to anyone reading Apple's docs. They belong at the top of `docs/ane-hardware-priors.md`.

### Bug 1 — fp16 fill-const syntax rejected
**Symptom:** MIL compile fails on `val=tensor<fp16,...>(X)` for any constant X.
**Fix:** replace fill-const with a `sub` op (e.g., `sub(x, x)` for a zero tensor).
**Generalization:** MIL ios18 doesn't accept fp16 scalar fill literals — synthesize constants from ops.

### Bug 2 — `state` is a reserved MIL ios18 keyword
**Symptom:** identical MIL program compiles when the input variable is named `a` but fails when named `state`. Two programs differ only by name.
**Fix:** rename `state` → `st`. Code comment added: `// NB: `state` is a reserved keyword in MIL ios18 — use `st` instead.`
**Generalization:** avoid generic names (`state`, `input`, `output`, `tensor`, likely more). Use disambiguated short prefixes (`st`, `g`, `k`). **No published reserved-word list exists** — discoverable only by rename-and-retry.

### Bug 3 — fp16 IOSurfaces cause `status=0x1d` at eval
**Symptom:** `MLModelError` with status code 0x1d when any IOSurface input is declared fp16.
**Fix:** **ALL working ANE programs in higgs use fp32 IOSurface inputs + `cast(fp32→fp16)` inside MIL.**
**Generalization:** IOSurface dtype ≠ MIL operand dtype. Bind fp32 at the surface, cast down inside the program. This is why `ane_bridge.m` has the bf16→fp16 clamp (068a14ef) — it runs on the fp16 path *inside* MIL, not at the surface.

### Bug 4 — Mixed-channel IOSurfaces cause `status=0x1d` with 3+ inputs
**Symptom:** 3+ IOSurface inputs with different channel dims (e.g., one C=1, another C=16) → eval fails 0x1d. 2 mixed inputs work. Same-channel N inputs work.
**Fix:** flatten heterogeneous inputs into a single uniform-channel layout (e.g., `[1, Dk, 1, flat_w]` where `flat_w = ane_align_seq(hv * dv)`), broadcast small inputs across channels before writing to IOSurface.
**Test preserved for regression:** `probe_broadcast_eval` (`#[ignore]` with constraint documentation).
**Generalization:** ANE's IOSurface binding table has an (undocumented) 2-input tolerance for channel mismatch; beyond that, uniformize.

### Bug 5 — ANE reorders IOSurface bindings alphabetically by parameter name
**Symptom:** **data corruption** — wrong tensors fed to wrong ops, **no error raised**. Parity fails silently.
**Root cause:** IOSurface index 0 maps to the **alphabetically-first parameter name**, not the first-declared. Code comment: "ANE may sort IOSurface bindings alphabetically or by first-use."
**Fix:** name params `a0, a1, a2, ...` so alphabetical order = declaration order = write order.
**Generalization:** This is the most dangerous quirk — no compile error, no eval error, just wrong outputs. When a MIL program compiles, evals, but produces garbage, **check parameter name ordering first.** Use `a0..aN` (or another monotonically-sorting prefix) as a matter of policy.

### Claim IDs for CLAIMS.md merge

- **AQ9 (E):** fp16 fill-const rejected — use `sub` op.
- **AQ10 (E):** `state` is reserved MIL ios18 keyword.
- **AQ11 (E):** fp16 IOSurfaces fail `status=0x1d`; use fp32 IOSurface + in-MIL cast.
- **AQ12 (E):** 3+ mixed-channel IOSurfaces fail `status=0x1d`; uniformize to single channel dim.
- **AQ13 (E):** ANE binds IOSurfaces by alphabetical name order, not declaration order — silent data corruption.

---

## Derived Learnings (not explicit in any single commit)

### L1 — The "invisible" timer problem
c1f85ade shows: **a regression's root cause can be invisible if the timer scope is wrong.** 228-280 ms/round was unaccounted for two commits (bee1ee20 → c1f85ade) because `t_verify_fwd` started too late. **Generalization:** when perf regresses, first check that the timer brackets cover the new code path. Forensic timers are cheap; guessing is expensive.

### L2 — Weight-value scaling is a ship-level workaround for fp16 saturation
c95a80c7 + 2de57ce5 compose to keep all intermediate values in fp16 range *without* MIL surgery: halve the weights going in, multiply back in fp32 at the IOSurface boundary. **Non-commuting steps (silu gate) must NOT be scaled** — the fix is asymmetric by design. This pattern will recur anywhere `inter` is large (larger models, MoE expert FFNs).

### L3 — ANE MIL has undocumented axis constraints
068a14ef discovered by parity failure: axis=3 concat silently NaNs. 144ac21d: seq must be 64-byte aligned or ANE returns `status=0x1d`. Both behaviors compile successfully and fail at eval. **Rule:** for any new MIL construct, add a parity vs Metal baseline test before wiring to production.

### L4 — ANE program load cap = 119
cddcb2a1 notes "load_count 0→3 — patch_from_donor reuses microcode without bumping loadWithQoS (well below the 119-program ANE cap)." **Implication:** patching ≫ recompiling for same-shape, different-weights kernels. For 24 GDN layers, compiling 24 donors would trip the cap; patch_from_donor stays at 1 compile + 23 patches.

### L5 — Classic spec decode ≠ DFlash for MoE+SSM
e074c9b4 shipped the infrastructure but proved it's useless for Qwen3.5-35B-A3B: GDN is sequential, MoE loads different experts per token. DFlash exists *because* classic batched verify doesn't work for this class. The DFlash regressions are intrinsic to the hybrid architecture, not the implementation.

### L6 — ANE prefill I/O dominates, not compute
e7172aa6 → cd78076c trajectory: compute was ~correct from day 1 (cos=0.99999 on real weights), but **ANE I/O overhead was 38-68 ms vs GPU 3-17 ms**. Zero-copy IOSurface path recovered 7-35%. The remaining gap is transpose + sync cost, not matmul.

### L7 — Worker thread pattern is load-bearing
e893d465 + d5c20025 + fb54b77e: the `!Send` IOSurface is a hardware constraint propagating through the whole stack. Every ANE consumer (dflash, GDN, lm_head scaffolding) ends up as a worker thread + mpsc handle. **Do not add Drop on the handle** — it kills the worker via Clone propagation.

### L8 — DFlash's bottleneck is acceptance rate, not speed
c1f85ade ceiling math (perfect accept = 41 tok/s, 34% = 14 tok/s) and 22bf8f15 Topology B win (GDN=0 + better accept = 22.49 tok/s) both point to: **optimize the drafter / tap layers to raise accept rate before optimizing verify latency.** The `bench_9b_blocksize_sweep.sh` script was added explicitly for this.

---

## Source-anchored answers to Open Questions

Updates the Q# list in CLAIMS.md where archaeology supplies answers.

| Q# | Question | Archaeology answer |
|---|---|---|
| Q6 | Was `HIGGS_TARGET_ANE_PREFILL` ever prototyped? | **No commit creates this flag.** `git log --all --grep='ANE_PREFILL'` returns empty. The `ane-gdn-prefill` work (e7172aa6, cd78076c, 267bf791) exists but is wired through `HIGGS_TARGET_ANE_GDN`, not a separate prefill flag. The roofline proposal remained a proposal. |
| Q7 | The 4B scaling story | **Multiple dimensions in the git record:** (a) `b72412a2` (Wave 1) proved 4B GDN parity on BF16 first (max_diff 0.003906). (b) `068a14ef` proved 4B DFlash tiled-matmul parity (5 layers, 0 NaN/Inf). (c) `bee1ee20` notes 4B has **tied word embeddings + hidden=2560** (≠9B 4096) — required Qwen3.5-4B-DFlash drafter variant with matching dims. (d) 4B didn't hit the fp16 saturation wall c95a80c7 fixed — smaller `inter` dim. **Summary:** 4B "scaled" because its dimensions sit inside fp16 range AND inside the tiled-matmul single-tile path. 9B broke both invariants. |
| Q2 | Does DFlash ANE draft help prefill vs GPU-only DFlash? | **No — worse.** `c1f85ade`: pipeline=true (which forces CPU/ANE drafter) regresses 13→8.7 tok/s. The "lazy GPU drafter" path is faster because MLX fuses draft compute into the lm_draft eval. |

## Source-anchored new claims (to merge into CLAIMS.md)

### DFlash — measured (Tier E)

- **DF1:** 38% regression on Carnice-9B confirmed — 21.17 AR vs 13.12 DFlash tok/s (bee1ee20).
- **DF2:** `verify_build_ms` (72 blocking ANE GDN dispatches) = 228-280 ms with GDN=1, previously untimed (c1f85ade).
- **DF3:** Pipeline mode regresses further (13→8.7 tok/s) — drafter contention + lower accept (c1f85ade).
- **DF4:** Topology B first DFlash win over AR baseline: 22.49 vs 19.46 tok/s (+15.5%) at GDN=0, pipeline=0, BS=16 (22bf8f15).
- **DF5:** Greedy rejection sampling: 22.98/23.20/23.21 tok/s (3 reps, temp=0) vs 22.49 baseline — no regression (782360c0).
- **DF6:** Full round breakdown @ block=16, GDN=1: draft 0.1ms · lm_draft 53ms · verify_build 280ms · verify_fwd 28ms · replay 2ms · total 311ms · accept 5.5/16 (c1f85ade).
- **DF7:** GDN partial-accept state corruption fix: accept 2.1→3.4 tok/round; 11.7→17.7 tok/s (a7e2737c).
- **DF8:** Tape replay + batched Metal kernel: 26.3 tok/s @ 2048 tok (2.7× AR); acceptance scales 3.2@65tok → 12.7@2048tok (d6daf3e0).
- **DF9:** `forward_last` optimization: 10.8→12.7 tok/s (+17.6%); draft time 916→613 ms (a79b904c).
- **DF10:** Classic spec decode (small draft + big verify) gave ZERO speedup on MoE+GDN (e074c9b4).

### ANE hardware/MIL quirks (Tier E)

- **AQ1:** MIL `concat` axis=3 (innermost) silently produces NaN; only axis=1 reliable (068a14ef).
- **AQ2:** Last tensor axis must be 64-byte / 32-fp16 aligned; seq=137 fails status=0x1d universally (144ac21d).
- **AQ3:** 9B `inter=12288` saturates fp16 in down_proj and up_proj; halve weights at compile + restore in fp32 (c95a80c7, 2de57ce5). **Gate weight must NOT be scaled** (silu nonlinearity).
- **AQ4:** ANE per-kernel microcode budget exceeded by lm_head [248320 × 2560] fp16 (~1.25 GB across ~130 tiles) — "ANE multi-weight compilation failed" (e318efa0).
- **AQ5:** bf16→fp16 cast must be clamped at ±65504 (068a14ef, ane_bridge.m).
- **AQ6:** ANE program load cap = 119; `patch_from_donor` reuses microcode without bumping `loadWithQoS` (cddcb2a1).
- **AQ7:** BLOBFILE-baked weights are **30× faster** than single-kernel + `reload_weights` (0f5bd628: 840ms → 19ms for seq=128).
- **AQ8:** ANE DMA requires `dsb sy` ARM64 barrier around eval for write→eval→read visibility (`feat/ane-prefill:ane_gdn_prefill.rs:33-40`).
- **AQ9:** MIL ios18 rejects `val=tensor<fp16,...>(X)` fill-const; synthesize via `sub` op.
- **AQ10:** `state` is a reserved MIL ios18 keyword; use `st` (rename was the fix in `ane_mil.rs:~1457`).
- **AQ11:** fp16 IOSurfaces fail eval `status=0x1d`; bind fp32 at surface, cast down inside MIL.
- **AQ12:** 3+ IOSurfaces with mixed channel dims fail `status=0x1d` (2 mixed is OK); uniformize.
- **AQ13:** ANE binds IOSurfaces by **alphabetical parameter name**, not declaration order — silent data corruption. Policy: name params `a0..aN`.

### ANE prefill engine (Tier E)

- **AP1:** ANE GDN prefill: cos=0.999991 on real 35B-A3B-3bit; ANE I/O = 38-68 ms vs GPU 3-17 ms per layer — I/O dominates (e7172aa6).
- **AP2:** Zero-copy IOSurface recovers 7-35% (cd78076c).
- **AP3:** Conv1d S≤32 sliding-window replaces kernel dispatch; S>32 falls through (267bf791).
- **AP4:** Bonsai hybrid CPU split: layers 14-27 on CPU for causal path (0f5bd628).

### Worker threading (Tier D)

- **W1:** Handle must NOT impl Drop — Clone propagation kills the worker on first handle drop (e893d465).
- **W2:** GDN worker dispatch: 0.12 ms/dispatch @ 1000-round synthetic stress (d5c20025).
- **W3:** `HIGGS_TARGET_ANE_GDN` was **dead in production pre-Wave 4** — `!Send` IOSurface blocked it (ed01ae33).

---

## ANE bandwidth wall — the 18.5 ms floor (from `memory/dflash-ane-projections-v2-handoff.md`)

The DFlash-ANE-projections-v2 handoff measures the **exact bandwidth ceiling** for the current fp16 path on Qwen3.5-4B-DFlash drafter (h=2560, q=4096, kv=1024, inter=9728, 5 layers, block=16):

| Component | fp16 weights | ANE time | Effective bandwidth |
|---|---|---|---|
| QKV | 157 MB | 3.5 ms | 44.9 GB/s |
| O proj | 105 MB | 2.3 ms | 45.7 GB/s |
| MLP chain (silu+down) | 747 MB | 12.7 ms | 58.8 GB/s |
| **Total** | **1010 MB** | **18.5 ms** | **54.6 GB/s** |

**Claim AB1 (E):** The DFlash-drafter ANE forward is DRAM-bandwidth-bound at **54.6 GB/s effective** — ~45% of M4's 120 GB/s peak. The MLP chain approaches the ceiling at 58.8 GB/s because it streams more data per kernel (747 MB) and amortizes dispatch better.

**Claim AB2 (E):** v1→v2 optimizations (realtime dispatch + silu→down IOSurface chain via `share_output_to` + NEON 4×4 block transpose + QKV||target_kv thread::scope overlap) moved ctx=16 from 29 ms → 28 ms (3.4%), ctx=64 from 31.6 → 31.2 ms (1.3%), ctx=256 from 50.4 → 47.5 ms (5.7%). **Scheduling tricks have hit diminishing returns against the bandwidth wall.**

**Claim AB3 (I):** The ONLY path below the 18.5 ms floor is **int8 weight blobs**: halving 1010 MB fp16 → ~505 MB int8 → ~9.2 ms ANE floor at the same 54.6 GB/s. Combined with existing CPU overlap: **estimated ~12-13 ms total forward.** 30-35% headroom.

**Claim AB4 (D — SUPERSEDED 2026-04-18):** ~~The infrastructure already exists: `build_weight_blob_int8` in `ane_bridge.rs:332`, MIL op `constexpr_affine_dequantize` handles int8→fp16 at eval time. Blocker is per-channel/per-tensor scale+zero_point plumbing and MIL generators updated to emit the int8 weight path.~~

Superseded by AB5/AB6 below — the `ane_mil.rs` emitter approach does NOT work on the engine's current ANE bridge, but a .mlpackage-based approach DOES.

**Source:** `memory/dflash-ane-projections-v2-handoff.md:1-70`.

### AB5 — Raw-MIL int8 path is dead (Tier E, re-measured 2026-04-18)

**Claim AB5 (E):** `constexpr_affine_dequantize` + `tensor<int8>` weights submitted through `_ANEDesc modelWithMILText:` (the engine's current `AneKernel::compile_multi_weights` path via `crates/higgs-models/bridge/ane/ane_bridge.m`) fails compile with `ANECCompile() FAILED: err=(InvalidMILProgram)`.

**Toolchain under test:** macOS 26.3.1 (25D771280a) · Xcode 26.0.1 (17A400) · coremlc 3505.4.1 / MIL 3510.2.1 · coremltools 9.0.

**Test reproduced:** `diffusion_ane::tests::test_int8_conv1x1_nanobot_pattern` (c_in=c_out=64, seq=16, conv1x1 + constexpr_affine_dequantize, scale=0.01, axis=0). The test was `#[ignore]`'d on 2026-04-03 with the same failure; 15 days and an Xcode bump did not change the outcome. Failure hash: `4E71B9B165...`.

**Scope of kill:** raw-MIL entry point only. Does NOT rule out int8 via other CoreML entry points. See AB6.

### AB6 — mlpackage int8 path is alive (Tier E, new 2026-04-18)

**Claim AB6 (E):** The same int8 weight + `constexpr_affine_dequantize` + conv1x1 chain builds, compiles, and schedules on ANE when delivered as an `.mlpackage` via coremltools' typed `mlprogram` path instead of raw-MIL text.

**Evidence:**
- `ct.convert(..., opset_version=ct.target.iOS18)` emits the mlpackage without warnings.
- `xcrun coremlcompiler compile` converts the mlpackage to `.mlmodelc` without errors.
- `MLComputePlan.load_from_path(..., compute_units=CPU_AND_NE)` reports: `conv` op `supported_compute_devices = [CPU, NeuralEngine]`.
- At toy shape (c_in=c_out=64, seq=16): `preferred_compute_device = MLCPUComputeDevice` (cost 0.96) — scheduler picks CPU for kernels too small to amortize ANE dispatch.
- At DFlash-4B o_proj shape (c_in=c_out=3072, seq=16, 9.4 MB int8 weights): **`preferred_compute_device = MLNeuralEngineComputeDevice`** (cost 0.54). The scheduler flips to ANE once the kernel is big enough.

**Probe artifacts:** `/tmp/higgs_int8_probe/` — `build_int8_mlpackage.py`, `build_realistic.py`, `compute_plan.py`, `plan_4b.py`, plus generated `.mlpackage` / `.mlmodelc` directories. Python 3.13 sidecar venv required because Python 3.14 wheels lack `libcoremlpython` (confirms CLAIMS.md T1).

**What this does NOT prove yet:**
1. Parity vs CPU reference at realistic shapes (toy-shape `predict()` path has a loader quirk — blocked on the MLModel-wrapper rather than the compute-plan API).
2. Latency at realistic shapes — compute-plan preferred-device is not a wall-clock measurement. The ANE could still be bandwidth-bound at the same 54.6 GB/s, or the scheduler could drop back to CPU under different `configuration.computeUnits` combinations.
3. Scalability to the MLP chain (ic=3072, oc=9728 per up/gate; 17408 for down on 27B where AB3 originally targeted 505 MB).

**Consequence for AB3:** AB3 (int8 halves bandwidth → ~9.2 ms floor) remains a live target. AB4 was wrong about *where* in the codebase the plumbing lives: NOT `ane_mil.rs` emitters, but a new `.mlpackage`-based bridge alongside the existing `_ANEDesc modelWithMILText:` path. See rewritten plan at `.planning/next-session-dflash-int8-weights.md`.

### Important cross-reference with CLAIMS.md R3

CLAIMS.md R3 says "Decode: ANE provides ZERO value — int8 reads 2.37× more bytes than 3-bit at M=1" (roofline.md:177-181). That compares **ANE int8** against **GPU 3-bit quantized** for the 35B-A3B decode case.

AB3 compares **ANE int8** against **ANE fp16** for the DFlash-drafter projection chain. Different denominators.

**Resolution:** R3 is about the ANE-vs-GPU choice for whole-model decode on 3-bit-quantized MoE (GPU wins on bandwidth). AB3 is about the fp16→int8 step *within* the ANE path for an already-ANE-committed code path (halves bandwidth, wins). **Both are correct.** The takeaway: if you've chosen to run on ANE, int8 weights are the next 30% win. If you're choosing between ANE and GPU for 3-bit quantized decode, GPU still wins because int8 is heavier than 3-bit.

---

## Wave 2 — Uncommitted Handoffs Scan (9/26 files)

Spawned Explore subagent across `memory/*.md` + `docs/*.md` uncommitted. 23 new Tier-E/D claims from the first 9 priority files. Denser than expected — full scan deferred pending triage.

### New ANE hardware constraints (E)

- **N5:** ANE SRAM cliff at ~28 MB working set → ~30% throughput drop (not gradual). 9B hits this at 32 MB. (`memory/9b-optimization-report.md:70-71`)
- **N6:** ANE minimum spatial dim = 16. Programs with seq < 16 compile but fail eval `status=0x1d`. Complements AQ2. (`memory/9b-optimization-report.md:73-74`)
- **N7 (refines AQ2):** Precise rule: `seq * 2 % 64 == 0` → `seq % 32 == 0` at fp16. seq=137 (prime) fails universally. 64/96/128/160 pass. (`memory/9b-optimization-report.md:76-77`)
- **N8:** Per-MIL-program tile cap ≈ 16 tiles. 20 tiles → `InvalidMILProgram`; 14 tiles OK. 27B DOWN budget 12 MB → 16 MB per tile to stay at 14 tiles. (`memory/9b-optimization-report.md:79-80`)
- **N10:** IOSurface lock/unlock ≈ 1 μs per op. Zero-copy via raw pointers + `dsb sy` eliminates this on hot path (Orion-style). Ties to cd78076c commit. (`memory/9b-optimization-report.md:85-86`)
- **N12:** `patch_from_donor` **LOAD FAILS** when projection shapes diverge (~layer 18 on 9B). Reaffirms the patch-only-same-shape invariant from cddcb2a1. (`memory/9b-optimization-report.md:91-92`)
- **N14:** CPU `[seq, channels]` row-major vs ANE `[1, channels, 1, seq]` channel-major — NEON 4×4 block transpose + scalar edges per dispatch. (`memory/9b-optimization-report.md:97-98`)
- **N21:** GDN recurrence ANE compile **FAILS** at 9B dims (Hv=32, Dk=128, Dv=128): `flat_w=4096` fp32 inputs `[1,128,1,4096]` ≈ 2 MB each exceed ANE input size limits. Works at tiny (Hv=4, Dk=16, Dv=16, flat_w=64). Workaround: per-head dispatch loop Hv×32 times with ~1 ms overhead each. (`memory/gate1-ane-worker-wiring-handoff.md:20-29`)
- **N23:** GDN FFN `gate_up` at seq=512 working set = 50.3 MB > ANE SRAM 28 MB → needs 2-3 tile tiling via `compute_oc_tile_plan`. (`docs/ane-prefill-design.md:97-100`)

### New DFlash measurements (E) — answers the multi-scale story

- **N1:** Topology B 22.49 tok/s (GDN=0 GPU verify), +15.5% over AR 19.46. `verify_build_ms` collapses 230.9 → 2.2 ms with 72 ANE dispatches removed. (`memory/topology-b-win-and-ar-ane-handoff.md:12-26`)
- **N15 (4B):** DFlash-drafter block-16 verify 95.1 ms (5.94 ms/tok), drafter 27.7 ms, amortization 3.45× (ideal ≤1.0). AR baseline 33.9 tok/s. Projected @accept=5: **40.7 tok/s (1.20× speedup).** (`memory/dflash-probe-4b-latency.md:4-6`)
- **N16 (27B):** Drafter 47.3 ms @ ctx=16. Verify 543.4 ms. AR baseline 6.4 tok/s. Projected @accept=5: **8.5 tok/s (1.33× speedup, poor amortization).** (`memory/dflash-probe-27b-latency.md:4-6`)
- **N17 (27B crash root cause — resolves Phase 1 open question):** 27B DFlash ANE drafter **FAILS compile** with `ANECCompile: InvalidMILProgram`. Fused SILU `gate_up` at 5120×17408 ≈ **178 MB bf16** exceeds the ~32 MB SRAM sweet spot (N5). Drafter alone ceilings at ~45 tok/s. (`memory/27b-dflash-ane-silu-diagnosis-handoff.md:7-9`)
- **N18 (D):** 27B target 30 tok/s requires `HIGGS_TARGET_ANE_LM_HEAD=1` to pull the 108 ms floor → 68 ms (ceiling 49 tok/s @ accept=3). Scaffold landed in e318efa0 but never end-to-end benched at 27B. (`memory/27b-dflash-ane-silu-diagnosis-handoff.md:10`)
- **N19:** Topology B BS sweep flat at 21.2-22.9 tok/s across BS ∈ {8, 12, 16, 24, 32}. Accept saturates ≈ 6 regardless. **No BS cliff** (the cliff existed at GDN=1). (`memory/topology-b-win-and-ar-ane-handoff.md:25`)
- **N2 (reinforces DF2):** GDN runs faster on GPU (144 ms compute) than ANE dispatch (230 ms) on 9B verify — confirms the regression is dispatch-bound, not compute-bound. (`memory/dflash-regression-bee1ee20-handoff.md:14-16`)

### New DFlash semantics (D)

- **N20:** Rejection sampling at temp=0 is **mathematically identical to strict argmax matching** — zero lift from accept criterion alone. Multi-candidate or better drafter needed to raise accept. (`memory/topology-b-win-and-ar-ane-handoff.md:33`)
- **N22 (from `docs/ane-prefill-design.md`):** Prefill strategy — GDN layers (75% of 35B total) are ANE-friendly, FA (full-attention) layers GPU-only. Split-silicon prefill at 512-token chunks (512 % 32 == 0 → AQ2 compliant). 30 GDN + 10 FA split on 35B-A3B. (`docs/ane-prefill-design.md:15-20`)

---

### Wave 2 continued — N24-N33 (remaining 16 files)

#### Prefill strategy (docs/ane-prefill-design.md)

- **N24 (E/I):** Hybrid ANE-GDN + GPU-FA split-silicon prefill targets **3,555 tok/s vs 450 GPU-only (7.9×)** at 144 ms GDN + 130 ms FA overlapped = 144 ms. **6× power reduction** (62 W → ~10 W). Grounded in nanobot-rs prior art (5.4× training, 17× GDN projection). Theoretical ceiling. (`docs/ane-prefill-design.md:163-180`)
- **N25 (D):** 40 layers × 4 seq buckets {128, 256, 512, 1024} = **160 programs > ANE 119-program cap**. Fix: share single compiled kernel across layers via `DynMatmul` (weights in IOSurface) or sliding-window unload/reload. Reaffirms AQ6 with a concrete budget breach. (`docs/ane-prefill-design.md:239-241`)
- **N26 (D):** 30 GDN layers × ~6 kernels × ~50 ms = **~9 s ANE compile cost at startup.** Mitigated by persistent cache at `~/.higgs/ane_cache` → second launch instant. (`docs/ane-prefill-design.md:198-202`)

#### DFlash drafter / verify — 27B path

- **N27 (E):** 27D fix lands 27B DFlash ANE drafter at **BS=12 6.82 tok/s** (vs 5.43 CPU-BLAS baseline, +26%). Draft time collapses from **32-73 ms → 0.3 ms at all BS.** New bottleneck: `verify_build = 410 ms` @ BS=12. (`memory/27d-verify-build-handoff.md:10-11`)
- **N31 (E/D):** Root cause of 410 ms `verify_build`: `ane_mlmodel.rs::dispatch()` line 172 calls `x_f32.eval()` which forces the **entire upstream lazy graph** (embed → 64 layers → norm) to materialize GPU-synchronously **before** the ANE lm_head matmul runs. Composition: full GPU forward + 60-80 ms ANE round-trip + pack/unpack. `HIGGS_DFLASH_PIPELINE=1` cannot hide this because the blocking work is on the main thread. (`memory/27e-verify-build-lmhead-block-handoff.md:14-24`)
- **N32 (D):** Proposed split — add `HIGGS_DFLASH_ANE_LM_HEAD` (default OFF) gated separately from `HIGGS_TARGET_ANE_LM_HEAD`. ANE lm_head matmul (~7 GFLOP ≈ 1-2 ms GPU) beats ANE round-trip (60-80 ms) **only when the hidden graph is lazy**; loses when forced to eval synchronously. (`memory/27e-verify-build-lmhead-block-handoff.md:99-120`)
- **N33 (D):** DFlash ANE drafter stack = **33 kernels** = 1 fc (variable ctx) + 8 layers × {fused_qkv, o_proj, fused_gate_up, down}. Target <15 ms drafter + 95 ms GPU verify → **52.6 tok/s @ accept=5** (vs 14.5 tok/s serial). (`memory/dflash-ane-projections-handoff.md:100-101`)

#### DFlash speculative + topology sweep

- **N28 (D):** New env flag `HIGGS_DFLASH_SPECULATIVE` — levels 0 (disabled) / 1 (single median anchor) / 2-3 (multi-anchor). Projected **21 → 28-30 tok/s** on 9B at T=12 via 65% hit rate on stale-context pre-drafts during the verify window. (`memory/speculative-predraft-handoff.md:190-193`)
- **N29 (D):** **Dimension mismatch blocker:** z-lab `Qwen3.5-4B-DFlash` has hidden=2560, Carnice-9B target has hidden=4096. DFlash drafters must match target's `forward_all_logits_from_hidden` input dim. Requires projection layer OR skip. Context for bee1ee20's note about needing a matching 4B drafter. (`memory/w1-temperature-fix-handoff.md:24-26`)
- **N30 (E — full topology sweep):** 9B results — **Pure baseline 19.46, Topology A (ANE-GDN=1) 18.08, Topology B (GDN=0) 22.49, Topology C (pipeline=1) 14.02.** ANE-GDN is **net loss** at 9B. (`memory/w1-temperature-fix-handoff.md:217-224`)

#### Strategic (not a measurement)

- **Merge strategy 2026-04-11** lists 8-PR sequence: Phase 0 merges PR stack #72-77 (Qwen3.5+TQ); Phases 1-2 five focused PRs (TQ fixes, prefill, decode, GEMV, TQ paging); Phase 3 three feature PRs (diffusion-ANE, magic-canvas block-K, ANE GDN prefill). Tech debt: split `qwen3_next.rs` (11.5K lines), `diffusion.rs` (7.2K lines). (`docs/merge-strategy-2026-04-11.md`)

---

## Contradictions (Wave 2)

- **X5 — Temperature handling (RESOLVED):** handoff said hardcoded argmax; current `simple.rs:1206` and `:1775` both branch `if params.temperature == 0.0` → argmax, else `accept_prefix_rs` rejection sampling. Handoff was pre-782360c0. Code comment at `:1815`: "Accept: greedy argmax+accept_prefix (temp=0) or rejection sampling (temp>0)." **Status:** temporal contradiction only — main is correct.
- **X6 — silu fusion on ANE (RESOLVED):** `memory/27c-silu-mil-split-plan.md:19-36` and `memory/27d-verify-build-handoff.md:1-12` confirm: the fused-silu-gate-up MIL (from 068a14ef/2de57ce5) **fails at the 32-weight ANE compiler threshold** (works at 20, fails at 32). The fix is **split-kernel**: separate `gate_proj` (16 weights) + `up_proj` (16 weights) kernels, with CPU doing `silu(gm) * um` fusion (100-300 μs/layer, <1.5 ms total for 5 DFlash layers). Trades IOSurface zero-copy for 2×`ane_to_cpu` + CPU loop + 1×`cpu_to_ane` round trip. Deployed in 27d. This is also the key that **unblocks 27B ANE drafter (N17).** So N13's "silu must happen on CPU" is now accurate *for the split path*; the fused path (068a14ef) exists for smaller models where it still fits under 32 weights.

---

## Contradictions surfaced or resolved

- **X4 (ANE helps prefill for dense):** Archaeology confirms **partially true**. The *ANE GDN prefill engine* (e7172aa6 → cd78076c) achieves correctness + modest I/O wins on dense projections *in isolation*. But when wired through `HIGGS_TARGET_ANE_GDN=1` end-to-end, the 72-dispatch-per-layer eager eval pattern (c1f85ade) regresses the whole verify path. **Resolution:** the kernel-level work is sound; the eager-eval wiring is the regression.
- **X1 (ANE for prefill flag):** Archaeology confirms **the flag was never implemented under that name**. The prefill work lives in the ANE GDN path, gated by the GDN flag. The roofline doc's proposal is stale.
