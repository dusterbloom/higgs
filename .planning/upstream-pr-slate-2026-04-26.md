# Upstream PR Slate — feat/magic-canvas → panbanda/higgs

> **REVISED 2026-04-26 after upstream audit.** Original optimistic slate replaced — most of our work cannot trivially land upstream because upstream has independently shipped competing implementations (MTP, chunked prefill, prefix cache, TurboQuant kernels, Qwen3.5).

---

## Audit findings — what upstream has now

In the 113 commits since branch point `43eedc6b`, upstream has shipped:

| Their feature | Their commit | Overlap with ours |
|---|---|---|
| **MTP** (multi-token prediction speculative decode) | `4fa09711`, `4396e884`, `785ffa8c` | Conceptually overlaps with our `DraftModel` trait, but theirs is model-built-in (Qwen3.5 MTP head). Could coexist. |
| Chunked prefill | `4fa09711` | Ours (`d24e4a92`) is independent — different design. **Theirs is canonical now.** |
| Prefix cache | `4fa09711`, `3e35b3f8`, `37d2ddd3` | Ours backports — superseded by theirs. |
| Spec-prefill (sparse prefill, currently disabled pending RoPE) | `0cdcb614`, `4e50c2aa`, in `simple.rs` | We have `crates/higgs-models/src/spec_prefill.rs` from upstream. We didn't touch it. |
| TurboQuant + benchmark suite | `f96995fa`, `1514737d`, `5b63e4c9` | Ours adds 296 LOC to `turboquant.rs`. **Need careful diff.** |
| **fp16 dtype fix (4× speedup, 18.6 → 75 tok/s)** | `8ece387a` (#18) | They already shipped a dtype fix. **Need to check which call sites they fixed.** |
| Qwen3.5 model architecture | `1514737d` | We have `qwen3_5.rs`, `qwen3_5_moe.rs` as new files; theirs lives in `qwen3_next.rs`. **Major design divergence.** |
| Doctor with multi-model routing | (multiple) | Their doctor is for a different system (provider routing) than ours (engine config). |
| Batch engine + scheduler | `crates/higgs-engine/src/batch_engine.rs`, `scheduler/` | We don't have these. |
| Thinking budget | `4fa09711`, `5b63e4c9` | We touch this in `simple.rs`; theirs is canonical. |
| Reasoning parser | `crates/higgs-engine/src/reasoning_parser.rs` | We don't have this. |

**Implication:** the cherry-pick model breaks for almost everything. Most of `feat/magic-canvas` is either (a) in new files that don't exist upstream (yarn.rs, bonsai_q1.rs, dflash.rs, diffusion.rs, qwen3_5.rs, speculative.rs, ane_bonsai_draft.rs, anycache.rs, constrained.rs work — though their constrained.rs exists too) or (b) modifies code paths upstream has already rewritten.

### Files that exist on BOTH sides (mergeable surface)
`qwen3_next.rs`, `simple.rs`, `turboquant.rs`, `doctor.rs`, `constrained.rs`, `deepseek_v2.rs`, `siglip.rs`, `transformer.rs`, `qwen3_moe.rs`, `gemma2.rs`, `phi3.rs`, `starcoder2.rs`, `siglip.rs`, `lib.rs`.

### Files only on our side (additive PRs only)
`yarn.rs`, `bonsai_q1.rs`, `qwen3_5.rs`, `qwen3_5_moe.rs`, `dflash.rs`, `diffusion.rs`, `speculative.rs`, `ane_bonsai_draft.rs`, `anycache.rs`.

---

## Revised strategy

The honest framing in three buckets:

### Bucket 1 — "Drop-in fixes to upstream code" (truly low-friction PRs)
Small surgical changes to files that **exist on upstream and still have the bug/inefficiency**. These are the only PRs that fit the "merge with ease" goal. Likely 1–3 PRs total.

### Bucket 2 — "RFC-class design proposals" (must coexist with upstream's design)
Larger features where upstream has shipped a different design we'd need to interoperate with. Open as **draft RFCs first**, get sign-off, then code. Examples: generic `DraftModel` trait that subsumes MTP; AR-spec FSM hooks for MTP.

### Bucket 3 — "Fork-resident features" (probably never upstream)
Niche, experimental, or fork-specific work. Ship on `nightly` branch on the fork; don't try to upstream. Examples: BD3LM, ANE drafter, Magic Canvas Gen UI, DFlash, Bonsai-Q1 1-bit engine, Qwen3.5 yarn extension.

---

## Quality bar (unchanged — applies to every PR)

1. Single coherent story (one `feat`, one `fix`, one `perf`).
2. Net diff < 500 LOC excluding tests/docs/generated.
3. Tests prove general logic, not custom asserts.
4. `cargo clippy -p higgs --all-targets -- -D warnings` clean.
5. `cargo fmt --check` clean.
6. `cargo test -p higgs -- --test-threads=1` green (project rule).
7. Doctor updated if config changed.
8. README + `higgs init` template updated if user-facing.
9. Conventional commit message — `type(scope): why-not-what`.
10. PR description follows template.

### PR description template

```markdown
## Problem
<1–3 sentences>

## Approach
<2–5 sentences — why this shape>

## Evidence
- Before: <number with source>
- After: <number with source>
- Δ: <ratio or %>

## Test plan
- [ ] Unit
- [ ] Integration / E2E
- [ ] Bench (if perf)

## Risk
<1 sentence>

## Out of scope
<bullets — keeps the diff small>
```

---

## Bucket 1 — Drop-in fixes (ready to ship)

| # | Title | Source commits | Net est | Verified upstream still has bug? | Status |
|---|-------|----------------|---------|-----------------------------------|--------|
| **1** | `perf(dtype): hold attention dtype through yarn_mscale and siglip scale` | extract from `2de6ad03` (deepseek_v2 + siglip parts only; drop rwkv7/yarn.rs/bonsai_q1.rs) | ~30 LOC + 1 test | ✅ deepseek_v2.rs:218 still has `multiply(mlx_rs::array!(yarn_mscale))`; siglip.rs:108 still has `.multiply(mlx_rs::array!(self.scale))` | **next** |
| 2 | `perf(dtype): scan-and-fix remaining f32-scalar promotion sites in upstream` | sweep — produce a fresh diff against `origin/main` HEAD | unknown until scanned | TBD — sweep needed | proposed |
| 3 | `fix(turboquant): 3-bit pack_indices corruption + correctness tests` | `b412a936` | ~150 LOC | TBD — diff our `turboquant.rs` against upstream's | proposed |
| 4 | `perf(turboquant): u32 packed words for decode kernels` | `d969879a` | ~250 LOC | TBD — diff against upstream's TQ kernels | proposed |

**PR #1 is verified-ready.** PRs 2–4 need a "before-cherry-pick" verification step that we now know to do for every PR.

---

## Bucket 2 — RFC-class design proposals (open as drafts)

These need design discussion before code. Each starts as a GitHub Discussion or draft PR with **no code**, just a problem statement + proposed shape + consequences. After upstream sign-off, then we code.

| # | RFC title | What we're proposing | Upstream's existing design | Friction |
|---|-----------|----------------------|----------------------------|----------|
| R1 | `RFC: generic DraftModel trait — MTP + AR-spec + PLD + DFlash as plugin impls` | Our `DraftModel` trait + cycle policy that subsumes MTP, AR-spec, PLD, DFlash, etc. | `mtp.rs` is hardcoded to model.mtp_draft; not pluggable | **high** — touches their core spec-decode design |
| R2 | `RFC: FSM-aware verify hooks for speculative decode (MTP first)` | `FsmHook` trait + adapter pattern from our `73777ab4` and `e72b5dee` | Their constrained.rs is upstream; MTP doesn't talk to it | medium |
| R3 | `RFC: Prompt-Lookup Decoding as a built-in DraftModel` | PLD lookup, configurable n-gram, FSM-aware | None | medium — additive only |
| R4 | `RFC: K-window adaptive controller for spec-decode` | The `38d33810` insight: K=2..3 default beats K=4..8 by 11–30% on real models | None | low — small fact-finding |

**Tactic:** open R4 first as a tiny RFC. It validates that we can have productive design conversations with the upstream maintainer before betting larger work on the relationship.

---

## Bucket 3 — Fork-resident (don't upstream)

Track these on the fork's `nightly` branch instead. They serve our use case but don't fit upstream's roadmap.

- **Bonsai-Q1 1-bit engine** (`bonsai_q1.rs`, `yarn.rs`) — niche, requires PrismML mlx-c cherry-picks, depends on `mlx_rs::ops::quantized_matmul(bits=1)`.
- **DFlash diffusion-style spec-decode** (`dflash.rs`) — competes with their MTP; A3B 6× regression unresolved.
- **ANE bridge** (`ane_bonsai_draft.rs`, ANE crate) — Apple-Silicon-only, blocked on 39 ms transfer mystery.
- **BD3LM Bonsai-8B** (`bd3lm_qwen3.rs`) — output broken, denoise-head audit pending.
- **Qwen3.5 yarn extension** (`yarn.rs`) — superseded by upstream's Qwen3.5 in `qwen3_next.rs`. Decide: discard or rebase onto theirs.
- **Magic Canvas / Gen UI / Structured CoT spike** — pre-spike; not yet shippable.
- **Eggroll runtime training** (scripts only) — research.
- **Speculative decode FSM stack on top of our DraftModel** — gated on R1/R2.
- **AneBonsaiDraftModel + native ANE drafter** — depends on ANE bridge.

---

## Per-PR workflow (unchanged but tightened)

1. **Verify still relevant.** `git log origin/main -- <touched-paths>` for the commits we want; read upstream's current version of any modified file. **If upstream already fixed this → drop the PR.**
2. **Branch off current `origin/main`:** `git checkout -b upstream-pr/<short-kebab-name> origin/main`.
3. **Cherry-pick** in order. Resolve conflicts by reading upstream's surrounding code; don't reflexively take "ours".
4. **Refresh tests.** Add tests if missing. Verify upstream's existing tests still pass.
5. **Pre-flight:** `cargo fmt --check && cargo clippy -p higgs --all-targets -- -D warnings && cargo test -p higgs -- --test-threads=1`. Plus relevant `-p higgs-engine -p higgs-models` runs.
6. **Squash** to clean commits with conventional messages.
7. **Update README / doctor / `higgs init` template** if user-facing.
8. **Push** to fork: `git push fork upstream-pr/<name>`.
9. **Open PR** against `panbanda/higgs:main` with the description template.
10. **Update this slate** — flip status to `open`, link the PR.

---

## Working-tree hygiene (one-time setup before PR work)

`feat/magic-canvas` has uncommitted work that should land on the fork branch first so the working tree is clean for cherry-pick operations.

Decision needed:
- (a) Commit the inventory + AGENTS + CLAUDE updates to `feat/magic-canvas` as `chore(planning): inventory + agent docs`, push to fork.
- (b) Stash and ignore until later.

**Preference: (a).** Keeps cherry-pick state clean.

---

## Tracking

| Date | PR # | Action | Notes |
|------|------|--------|-------|
| 2026-04-26 | — | Slate v1 created | Optimistic plan |
| 2026-04-26 | — | Slate v2 — upstream audit | Reset after finding upstream MTP, chunked prefill, prefix cache, Qwen3.5, dtype fix #18 already shipped |

---

## Recommended first move

**Step 0 — Setup (10 min):** Commit the inventory + agent docs to `feat/magic-canvas` and push. Working-tree clean.

**Step 1 — PR #1 (60 min):** Ship the verified-ready dtype fix (deepseek_v2 yarn_mscale + siglip attention scale). Two-line patch + one regression test. Smallest possible upstream-friendly first contact.

**Step 2 — Sweep (90 min):** Run `rg 'multiply\(mlx_rs::array!\(' crates/` against `origin/main` HEAD to find every f32-scalar promotion site that survived their PR #18. Open as PR #2 (or fold into PR #1 if scope is small).

**Step 3 — RFC R4 (30 min):** Open a GitHub Discussion proposing the K-window default tuning data. Tiny, fact-based, low-stakes way to test our relationship with the upstream maintainer.

**Step 4 — Decide R1.** Based on R4 reception, decide whether to invest in the bigger spec-decode RFC (R1) or stay fork-resident.

If steps 0–3 land cleanly, we have a working pattern. If they don't, the fork is the answer and the slate becomes "what to ship on `nightly` for our users".
