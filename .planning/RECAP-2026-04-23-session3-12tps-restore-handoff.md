# Session 3 recap (2026-04-23) — 12.7 tok/s restore + Phase 4 prep

## TL;DR

1. Yesterday's Mac crash during Phase 2 v2 bench: **bs16 config** (BS=16 + 27B
   4-bit + ANE compile on 36GB) triggered system-level crash during model load.
2. Salvaged v2 results (`v2/RESULTS.md`): **cap094 at 267ms verify_build, 3.1
   tok/s** is the survivor winner — but still 4× below the 10 tok/s gate.
3. **Baseline audit**: historical best was **12.7 tok/s** (commit `a79b904c`,
   2026-04-04), AR speculative decode (0.8B → 27B, K=16, ANE spec) — NOT
   DFlash. DFlash peak was 6.82 tok/s; today we measure 3 tok/s, a regression.
4. **Ceiling audit**: battle plan says **20 target, 22 hard ceiling**. The
   "24–30" number I earlier cited is NOT in any doc — that was my error.
5. Plan approved (`~/.claude/plans/giggly-honking-yao.md`): three phases —
   A validate 12.7 on feat/ane-prefill, B port to magic-canvas, C SMC-SD Lite.

## Blocker discovered mid-Phase-A (before executing)

The 0.8B drafter path that produced 12.7 tok/s is not trivially reproducible:

- **`Qwen3.5-0.8B-8bit`** (at `~/.cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit`)
  is a **VLM** (`model_type=qwen3_5`, nested `text_config`, layer_types include
  `linear_attention`). `DiffusionEngine::load` at `diffusion.rs:231` reads
  flat keys (`cfg["hidden_size"]`, `cfg["num_hidden_layers"]`) and will panic
  on this config — **cannot load as drafter**.
- **`Qwen3-0.6B-Base`** (at `~/.cache/lm-studio/models/mlx-community/Qwen3-0.6B-Base`)
  is a plain `Qwen3ForCausalLM` (dense) with vocab 151936. **DiffusionEngine
  CAN load it** (the existing test at diffusion.rs:7837 already uses it).
  But vocab mismatch with Qwen3.6-27B-4bit (vocab 248320) makes this pairing
  non-functional for speculative decode with the 27B on disk today.
- **Hypothesis**: the 12.7 tok/s run used a different 27B — likely
  `Qwen3-27B` dense (vocab 151936) that matches Qwen3-0.6B-Base. That model
  is NOT on disk under any 27B name I searched. Need to check git-log near
  a79b904c for the model path used in the actual bench, or check the
  `benchmarks/` dir on `feat/ane-prefill` for recorded bench commands.

## Assets available on disk (verified)

**Small drafters:**
- `/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3-0.6B-4bit/` — dense qwen3, vocab 151936
- `/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3-0.6B-Base/` — dense qwen3, vocab 151936
- `/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit/` — **VLM, incompatible**
- `/Users/peppi/AI-Models/shared/huggingface/hub/models--Qwen--Qwen3-0.6B/` — dense qwen3, vocab 151936
- `/Users/peppi/AI-Models/mlx/Qwen3-1.7B-MLX-8bit/` — larger option if 0.6B too weak

**27B candidates:**
- `/Users/peppi/.cache/lm-studio/models/NexVeridian/Qwen3.6-27B-4bit` — qwen3_5 VLM, vocab 248320 (what today's bench uses)
- *Qwen3-27B dense (vocab 151936) — NOT located yet*

## Next session — unstuck path (UPDATED after architecture probe)

**Disk hunt result**: only one 27B on disk — `Qwen3.6-27B-4bit` (qwen3_5 VLM,
vocab 248320, 64 layers, hidden 5120, head_dim 256, hybrid GDN + attention).
**No dense Qwen3-27B (vocab 151936) anywhere on disk.**

**Architecture probe of Qwen3.5-0.8B-8bit**: it's the same family as the
target — `model_type=qwen3_5`, vocab 248320 (matches 27B), hybrid layer_types
(`linear_attention` + `full_attention`), head_dim 256, 24 layers, weight
prefix `language_model.model.layers.N.linear_attn.*` with full GDN blocks
(`A_log`, `conv1d`, `dt_bias`, `in_proj_{a,b,qkv,z}`, `out_proj`).

**Conclusion**: `DiffusionEngine::load` CANNOT load this under any tweak —
it's designed for dense self_attn. The 0.8B needs `qwen3_next` (same engine
as the 27B target). `AneCausalDrafter` wraps `DiffusionEngine` → also
incompatible as-is.

So the a79b904c pairing on 2026-04-04 **could not have been these two files**.
The 12.7 must have been:
- (a) An older 27B that no longer exists on disk (dense qwen3 variant), OR
- (b) A different 0.8B model — search for any qwen3 variant with vocab 151936.

**Three real options for next session:**

**Option A — Widen disk hunt for the actual a79b904c pairing**:
```bash
# any 0.8B or 1B with vocab 151936 (matches dense Qwen3 family):
fd -t f config.json ~/AI-Models ~/.cache/lm-studio/models 2>/dev/null | \
  xargs grep -l '"vocab_size": *151936' 2>/dev/null
# then check which are 0.6B-1.7B (small enough to be a drafter)
# Qwen3-1.7B variants already located — could substitute.
```
Check git blob at a79b904c for any committed bench script or log mentioning
the model path. `git show a79b904c:benchmarks/` or `git log a79b904c..2cb7808b
--diff-filter=A -- benchmarks/` to find any runner that shipped.

**Option B — Port `AneCausalDrafter` to use `qwen3_next`** (bigger, matches
user's stated request):
- New `QwenNextCausalDrafter` wrapping `qwen3_next::Qwen3NextModel` as a
  small-max-seq drafter with `forward_last` semantics.
- Use Qwen3.5-0.8B-8bit as drafter + Qwen3.6-27B-4bit as target — vocabs and
  head_dim already match.
- Will require: (1) loading path in qwen3_next for the 24-layer 0.8B config
  (probably already works since it handles the 64-layer 27B), (2) a drafter
  adapter implementing the `draft(k)` API, (3) rewiring `speculative_generate`
  to accept the new drafter trait (or a second generate function). Budget 4–6h.

**Option C — Use Qwen3-1.7B-8bit as drafter** (at
`~/AI-Models/mlx/Qwen3-1.7B-MLX-8bit/` — dense qwen3, vocab 151936). Pair with
a downloaded dense Qwen3-27B. Needs ~54GB disk download, won't fit 36GB RAM
alongside drafter — **probably infeasible on this machine**.

**Recommended**: **Option B**. It matches "fuck DFlash + get phase 4 asap"
because (1) it reuses qwen3_next which is already the production target path,
(2) the 0.8B shares the drafter's strength (hybrid GDN) which is what made
12.7 tok/s possible on the ANE, (3) Phase 4 SMC-SD extends this same drafter
with N particles. Budget: 4–6h to get a bench number; if it's ≥12 tok/s,
Phase B collapses into it (same wiring). If it's <12, we've built the Phase 4
substrate anyway.

**Decision required from user before Phase A continues**: A, B, or C.

---

## USER DECISION (2026-04-23 end-of-session): **Option B**

Next session starts by porting `AneCausalDrafter` to wrap `qwen3_next` instead
of `DiffusionEngine`. Drafter = `Qwen3.5-0.8B-8bit`; target = `Qwen3.6-27B-4bit`.

### Option B execution outline (for next session opener)

1. **Recon qwen3_next loader** — confirm it can load the 24-layer 0.8B from
   `~/.cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit/` (weight prefix
   is `language_model.model.*` — same as the 27B target, so probably works).
   File: `crates/higgs-models/src/qwen3_next.rs`. Look for the config parser
   and weight loader — confirm nested `text_config` is read.
2. **Design drafter adapter**. Add `QwenNextCausalDrafter` next to
   `AneCausalDrafter` in `crates/higgs-models/src/diffusion.rs:3859`:
   - Wraps `qwen3_next::Qwen3NextModel` (or whatever the struct is called).
   - Exposes `draft(prefix: &[u32], k: usize) -> Vec<u32>` — greedy K-token
     continuation using the model's decode path.
   - Exposes `forward_logits` or equivalent for speculative_generate use.
3. **Fork `speculative_generate`** — new `speculative_generate_next` that
   takes `&QwenNextCausalDrafter` + `&mut AnyModel` (target). Keep the same
   overall algorithm: draft K with ANE + drafter, verify with target
   forward_all_logits, accept longest matching prefix.
4. **Wire into `simple.rs`** — env gate `HIGGS_AR_SPEC_DRAFT_PATH`. When set,
   load drafter at startup, route `generate_inner` through the new path.
   BEFORE DFlash check at `simple.rs:866`.
5. **Bench**. Same 200-token greedy prompt. Target ≥12 tok/s. If <12: the
   number isn't reproducible with qwen3_5 arch on this machine; fall back to
   Option A (hunt dense) OR accept current AR-spec number as the new floor
   and move to Phase 4 SMC-SD on top of the new substrate.
6. **Side benefit**: Phase 4 SMC-SD port builds directly on
   `QwenNextCausalDrafter` — N-particle drafting broadcasts over the same
   adapter. The work isn't wasted even if 12.7 doesn't reproduce.

### Risks specific to Option B

| Risk | Mitigation |
|------|------------|
| qwen3_next expects flat config, not nested `text_config` | Check at recon step 1. If broken, add nested-config parsing (~30 min). |
| 0.8B weights use `language_model.` prefix, 27B target code may hardcode prefix | Grep qwen3_next for `language_model.` to confirm. |
| ANE compile for 0.8B drafter might collide with 27B target's ANE session | Use drafter CPU-only first (skip ANE for drafter); if slow, add ANE offload later. |
| 0.8B + 27B both in memory → close to 36GB ceiling | Keep `HIGGS_MLX_CAP_FRACTION=0.94`; no BS=16; watch Activity Monitor. |
| `AneCausalDrafter` was the proof-of-12.7 path; new adapter may not reach same tok/s | Accept the real measured number. No faking. |

### Files to touch (Option B)

- `crates/higgs-models/src/diffusion.rs` — add `QwenNextCausalDrafter` +
  `speculative_generate_next`.
- `crates/higgs-models/src/qwen3_next.rs` — possibly add nested-config parsing
  if not already present.
- `crates/higgs-engine/src/simple.rs` — engine-level routing, env var gate.
- `crates/higgs/src/config.rs` + `doctor.rs` + `daemon.rs` — env var docs +
  doctor validation + init template.
- `README.md` — env var reference.

### Pre-flight at next session start

```bash
cd /Users/peppi/Dev/higgs
git status                                  # expect same dirty tree
git log -1 --oneline                        # expect ace5763b
ls -la ~/.cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit/
# Confirm model files still on disk.
```

Do NOT create a worktree for Option B — this is a magic-canvas integration,
not a feat/ane-prefill validation.

## What's on disk in worktrees

- No worktree created yet — Phase A step 1 not executed.
- Main checkout on `feat/magic-canvas` HEAD = `ace5763b`, dirty with BD3LM +
  planning files (unchanged from yesterday's session 2).

## Phase 2 salvage (for completeness)

`.planning/measurements/phase2-verify-build/v2/RESULTS.md` updated by
`parse_v2.sh` (new file). Summary:

| config | BS | cap | vbuild_med | eff_tps | accept_frac |
|--------|----|-----|-----------|---------|-------------|
| cap094 | 12 | 0.94 | 267ms | 3.1 | 0.18 |
| capunset | 12 | unset | 313ms | 2.8 | 0.18 |

Stop-criterion (vbuild ≤200, tps ≥10, accept ≥0.30) FAILS on every surviving
config. Don't resume Phase 2 — per the approved plan, Phase 2 is parked.

## Files written this session

- `.planning/measurements/phase2-verify-build/parse_v2.sh` — salvage parser.
- `.planning/measurements/phase2-verify-build/v2/RESULTS.md` — salvage table.
- `~/.claude/plans/giggly-honking-yao.md` — approved plan (A/B/C).
- `.planning/RECAP-2026-04-23-session3-12tps-restore-handoff.md` — this file.

## Tasks in the todo list

1. [in_progress] Locate Qwen3.5-0.8B drafter → **blocker above**, pivot needed.
2. [in_progress] Find or build AR-spec bench invocation → no bench scripts on
   feat/ane-prefill; will need to synthesize from test harness
   (`diffusion.rs:7837`) once drafter/target pairing is resolved.
3. [pending] Create worktree at feat/ane-prefill tip.
4. [pending] Build --release --features ane on worktree.
5. [pending] Run 200-tok bench, gate ≥12 tok/s.
6. [pending] Write ar-spec-12tps-baseline.md.
7. [pending] Phase B: wire AR-spec into simple.rs on magic-canvas.
8. [pending] Phase B: bench on magic-canvas, gate ≥12 tok/s.
9. [pending] Phase C: SMC-SD Lite port.

## Safety reminders

- `HIGGS_MLX_CAP_FRACTION=0.94` for any bench (winner from salvage).
- No BS=16 on 27B. That's what killed the Mac.
- Single server process. Graceful SIGTERM for `time -l` flush.

## Budget check

- Session time: context hit 78% before Phase A step 1 fired.
- Remaining work: A (~1h if drafter pairing resolved), B (~2–3h), C (~8h).
- Total road to Phase 4 complete: ~12h from an unblocked Phase A start.

## Last committed sha

`ace5763b` on `feat/magic-canvas` — unchanged.
