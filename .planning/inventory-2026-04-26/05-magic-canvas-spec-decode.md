# Magic Canvas / Gen UI + Speculative Decoding — feat/magic-canvas

## Branch identity

`feat/magic-canvas` is the integration branch where two product threads converge: a "magic canvas" / generative-UI direction (FSM-constrained structured output, OpenAI-style `response_format`, structured chain-of-thought) and a speculative-decoding stack (drafter trait, PLD, AR-spec, DFlash) made FSM-aware so it compounds with constrained generation. The user-facing pitch: structured outputs (JSON / GBNF / regex) that are also fast — every speculative-decode path consults the same FSM, so a `{`-prefix is enforceable and PLD/DFlash/AR-spec accept rates remain valid under constraint. Defining code paths: `crates/higgs-engine/src/{speculative.rs, ane_bonsai_draft.rs, constrained.rs, simple.rs}`, `crates/higgs-models/src/diffusion.rs::speculative_generate_next`, `crates/higgs/src/routes/chat.rs::build_constraint`.

## Speculative decoding stack

### Drafter trait + Bonsai shim — sessions 11/12 (commits `1161b844`, `2121953b`)
Goal: provide a generic `DraftModel` trait and a first concrete impl over the native ANE Bonsai engine, replacing the abandoned CoreML `MlxDraftModel` from `feat/ane-spec-decode`. Landed in session 12 as two new files: `crates/higgs-engine/src/speculative.rs` (370 lines, `accept_prefix`, `speculative_step`, `speculative_loop`, trait `DraftModel { prefill, draft, advance, rollback }` with `Send` bound, 19 pure-logic tests) and `crates/higgs-engine/src/ane_bonsai_draft.rs` (187 lines wrapping `AneBonsaiEngine`, stateless ctx-rebuild, `unsafe impl Send` mirroring `CachedMetalKernel` pattern, 6 ctx-truncation tests). 25/25 tests green; nothing wired to `SimpleEngine` yet at end of session.

### Spec-decode config — session 13 (commits `054494c2`, `feef8e47`, `fb48230c`)
Goal: wire `--draft-model`/`--num-draft` end-to-end without yet touching `simple.rs`. Landed: `ServeArgs` + `ModelConfig.{draft_model, num_draft}` plumbed through `build_simple_config` / `load_config_file`; 22 fixture sites updated; `doctor::check_draft_models` (path FAIL + batch-incompatibility WARN, 3 new tests, suite 26→29); `AnyCache::trim_by` + `SteppingKeyValueCache::trim_by` for rollback; daemon `higgs init` template + README updated. 88/88 higgs tests, doctor 29/29.

### Spec-decode engine wiring — session 14 (commits `5108e8db`, `835dc291`)
Goal: actually run draft→verify→accept inside `SimpleEngine`. Ported `293793ad` from `feat/ane-spec-decode`, swapping `MlxDraftModel` for `AneBonsaiDraftModel`. `SimpleEngine` gained `draft: Option<Mutex<Box<dyn DraftModel>>>` + `num_draft`, `load_with_draft()`, `speculative_generate()`, `speculative_streaming()`, branch points in `generate_inner` / `generate_streaming_inner` gated on `draft.is_some() && constraint.is_none() && !logprobs`. `build_bonsai_draft()` feature-gated on `ane`; env knobs `HIGGS_BONSAI_DRAFTER_SEQ_LEN`, `HIGGS_BONSAI_DRAFTER_EPS`. `tokenizer_hash` + `check_tokenizer_compat` reject mismatched drafter/target (override `HIGGS_SPEC_ALLOW_TOKENIZER_MISMATCH=1`). 410/410 + 88/88 green; no measurement.

### E2E green — session 15 (commit `1cee9bd6`)
Goal: first 200-OK on the speculative path. Three blockers fixed: (1) drafter OOM (jetsam) — `drop_blas_layers()` frees ~4 GB once ANE FFN tiles bake in; (2) `AnyModel::forward` slices last position only, so verify switched to `forward_all_logits` for K+1 rows; (3) drafter embed OOR on cycle 2 — clamp `tid` to `vocab − 1` in `AneBonsaiEngine::forward_last`. K=16 on Qwen3.6-27B-4bit + Bonsai-1.7B-q1: 1–2/16 acceptance, ~26 s draft + ~13 s verify. Crippled by tokenizer mismatch (151K vs 248K vocab) — kept under override flag, real win awaits a tokenizer-matched drafter (the "z-lab pivot" in the recap).

### AR-spec FSM-aware verify — sessions 4/5/6/7 (commits `394a79e2`, `73777ab4`, `26862aef`)
Goal: lift the AR-spec gate at `simple.rs:1071` and let the FSM mask verify rows. Cross-crate constraint: `speculative_generate_next` lives in `higgs-models`, `ConstrainedGenerator` in `higgs-engine` — solved with a `pub trait FsmHook { mask_verify_logits, advance_accepted, is_finished }` in `diffusion.rs` and `ConstrainedFsmHook<'a>` adapter in `constrained.rs`. Session 5 botched it via a unified-concat refactor that panicked at `mlx-rs/ops/convolution.rs:95`; session 6 confirmed the panic was a pre-existing dtype mismatch on iotaminer drafter S=1 conv (`bf16` weight vs `f16` activation), unrelated to FSM. Session 7 fixed the conv coercion (`26862aef`) and validated AR-spec on Qwen3.5-4B + Qwen3.5-0.8B-8bit: 48.6% accept / 19.96 tps, then on Carnice-9b + Qwen3.5-0.8B-8bit. The K-window default tuning (`38d33810`) drops `K_LOW=2 K_HIGH=3`: acceptance jumps 47.3%→72.7%, +30% tps on Carnice-9b, +11% on 4B (`speculation_policy.rs`).

### PLD — sessions 1/2 (commits `e72b5dee`, `871c5ddf`)
Prompt-Lookup Decoding shipped end-to-end with FSM-aware verify on the same primitives (`peek_states_for_drafts`, `apply_mask_at_state`, `advance`, `is_finished`). `simple.rs::load_with_pld` + `PldDraftModel` impl + 14 unit tests; CLI flags `--pld --pld-max-ngram --pld-min-ngram`; `ModelConfig.{pld, pld_max_ngram, pld_min_ngram}`; `doctor::check_pld` (mutual-exclusion + bounds + 9 tests); README "Prompt Lookup Decoding" section. Session 1 PLD smoke test (Qwen3-0.6B-4bit JSON) confirmed 127 `spec_decode: cycle` lines under FSM constraint — `{`-prefix held through verify rows. Session 2 caught a CLI fix: `build_simple_config` was hardcoding `pld: false` so `--pld` was silently dropped — fixed and verified.

### K window default tuning — commit `38d33810`
`HIGGS_AR_SPEC_K_LOW=2 / K_HIGH=3` made the new default in `simple.rs:465-472`: +11–30% tps with no code change beyond the constants. Default K=4..8 was over-eager for the drafter+verifier pairs we ship.

## Gen UI / Magic Canvas / Structured CoT

**Product story.** Constrained decoding (regex / JSON-schema via `outlines-core`) is already plumbed via `ConstrainedGenerator`. The "magic canvas" framing is: structured generative UI where the model emits regex-shaped tokens (e.g. `GOAL: …\nAPPROACH: …\nEDGE: …`) and the harness renders the canvas, so structured CoT compounds with PLD by speeding up whatever shape the FSM forces.

**Code/feature work.** Constrained-generation primitives extended in session 1 (`crates/higgs-engine/src/constrained.rs`) with `peek_states_for_drafts` (read-only walker over draft tokens), `apply_mask_at_state` (state-parameterized mask), private `mask_for_state` helper. These power FSM-aware verify across PLD/DFlash/AR-spec without changing `ConstrainedGenerator`'s public surface. Verify rows now mask per-position, `cg.advance(token)` runs per accepted token, and `is_finished()` short-circuits identical to the AR loop.

**Structured CoT spike.** Status: design + locations nailed (`RECAP-2026-04-26-structured-cot-spike-handoff.md`), code NOT written. Two-file ~30 LOC change: `simple.rs` drops `constraint = None` after `seen_think_close`, and `chat.rs::build_constraint` env-gates `HIGGS_STRUCTURED_THINK=1` to wrap the `<think>` span with a `GOAL/APPROACH/EDGE` regex from `andthattoo/structured-cot`. Promotion path (post-spike): `ModelConfig.structured_think_schema: Option<String>`, doctor check, cached `Arc<Index>` per model, README section. Compounds with PLD: PLD speeds the trace, structured CoT shrinks it 22–43×.

## Headline numbers

| Probe | Date | Config | Result | Source |
|---|---|---|---|---|
| PLD A/B Carnice-9b verbatim | 2026-04-26 | bf16, T=0, max_tokens=384, 3 runs | **1.84× median** decode (21.68→39.87 tps) | `benchmarks/pld_carnice_20260426/results.json` |
| PLD Qwen3-0.6B-4bit JSON smoke | 2026-04-26 | FSM-aware verify, max_tokens=128 | 127 `cycle` lines, `{`-prefix held, no panics | `RECAP-2026-04-26-pld-fsm-landed.md` |
| PLD demo (session 1) | 2026-04-26 | Carnice-9B prompt-overlap | 31.5 (PLD) vs 20.0 (off) → 1.57× | `RECAP-2026-04-26-structured-cot-spike-handoff.md` |
| AR-spec K=4..8 default | 2026-04-26 | Carnice-9b + Qwen3.5-0.8B-8bit | 47.3% acc, 14.66 tps | `RECAP-2026-04-26-session7-arspec-validated-k23-win.md` |
| AR-spec K=2..3 win | 2026-04-26 | Carnice-9b + Qwen3.5-0.8B-8bit | **72.7% acc, 19.10 tps (+30%)** | session7 recap; commit `38d33810` |
| AR-spec 4B | 2026-04-26 | Qwen3.5-4B-MLX-4bit + 0.8B-8bit | 48.6% acc, 19.96 tps; +11% at K=2..3 | session7 recap |
| Spec-decode E2E first run | 2026-04-24 | Qwen3.6-27B-4bit + Bonsai-1.7B-q1, K=16 | 1–2/16 acc, sub-0.1 tps (tokenizer mismatch) | `RECAP-2026-04-24-session15-...md` |
| Bonsai-8B verify floor | 2026-04-24 | K=12/16, prime=64 | **178 ms flat** (sub-linear ≥K=12) | `.planning/measurements/p6-verify-cost-8b.md` |
| Path A break-even (8B) | 2026-04-24 | ANE drafter, seq=256, α=0.7 | −88% to −94% vs 23.4 tps baseline → **dead** | p6 addendum |
| 27B AR baseline | 2026-04-25 | Qwen3.6-27B-4bit AR | **6.40 tps** | `27b-speculation-assumption-ledger-20260425.md` |

## Decision log

- 2026-04-24 — Port `DraftModel` trait from `feat/ane-spec-decode` `efa05ded` but **drop `MlxDraftModel`** (CoreML); native ANE bridge has surpassed it (s11).
- 2026-04-24 — Choose `unsafe impl Send` for `AneBonsaiDraftModel` mirroring existing `CachedMetalKernel` pattern; defer empirical IOSurface-thread-affinity check to first integration run (s12).
- 2026-04-24 — Tokenizer-hash gate on by default; override only via `HIGGS_SPEC_ALLOW_TOKENIZER_MISMATCH=1` with WARN (s14).
- 2026-04-24 — Spec-decode branch gated on `constraint.is_none() && !logprobs` initially; FSM-aware verify added later (s14 → s1 PLD).
- 2026-04-24 — Path A (stateless ANE drafter for Qwen3.6-27B) **abandoned** after p6 verify-floor data + ANE per-call probe; redirect to packed-q1 + fused Metal 1-bit kernels (`p6-verify-cost-8b.md` addendum).
- 2026-04-24 — On K=16 mismatch run, accept short-window finish_reason=stop as smoke pass; defer real measurement to z-lab tokenizer-matched drafter (s15).
- 2026-04-26 — PLD landed with FSM-aware verify in one commit (`e72b5dee`); 1.84× verbatim; CLI bug fixed via `871c5ddf`.
- 2026-04-26 — AR-spec FSM hook chosen as **trait callback** (`FsmHook`) rather than threading `ConstrainedGenerator` across crates (s4 plan).
- 2026-04-26 — AR-spec verify-argmax: **Option A (split branches)** preferred over unified concat to keep unconstrained path byte-identical; later validated dtype panic was the real bug, not the refactor (s5/s6).
- 2026-04-26 — K=2..3 made default after measured +30% on Carnice-9b; ANE drafter overlap deferred as multi-day work, not flag-gated (s7).

## Open threads

Unresolved next-session-*.md handoffs in this domain:

- `.planning/next-session-ane-drafter-p1-handoff.md` — ANE drafter P1 (stateful per-layer KV) needed to make Path A viable; gated on packed-q1 results.
- `.planning/next-session-B-measure-inline-ane-drafter.md` — original "Point B'" inline ANE drafter measurement that produced session 11's 8.7 tps (below 10 tps gate).
- `.planning/next-session-dflash-drafter-parity-audit.md` — DFlash drafter parity audit.
- `RECAP-2026-04-26-structured-cot-spike-handoff.md` — structured CoT spike not yet implemented; two-file change waiting.
- AR-spec advance-overhead (~19 ms/tok flat tax) and iotaminer distil-qwen35-4b drafter conv1d S=1 panic root cause (model-specific) deferred per session 7.
- DFlash + FSM smoke green (s4) but no realistic-prompt PLD characterization curve yet (carry-over from s2 #2).

## Recommended cleanup

| Filename | Superseded by | Reason |
|---|---|---|
| `.planning/RECAP-2026-04-24-session11-drafter-trait-integration-handoff.md` | s12 recap | Plan-only handoff; s12 executed steps 1+2. |
| `.planning/RECAP-2026-04-24-session12-drafter-trait-bonsai-shim-landed.md` | s13 recap | Steps 3+ covered downstream. |
| `.planning/RECAP-2026-04-24-session13-spec-decode-config-landed.md` | s14 recap | Engine wiring landed in 14. |
| `.planning/RECAP-2026-04-24-session14-spec-decode-engine-wired.md` | s15 recap | E2E now green. |
| `.planning/RECAP-2026-04-26-session5-arspec-fsm-handoff-shape-bug.md` | s6 recap | Hypothesis falsified; root cause was dtype, not shape. |
| `.planning/RECAP-2026-04-26-session4-dflash-fsm-smoke-arspec-plan.md` | s7 recap | Plan executed; AR-spec validated. |
| `.planning/RECAP-2026-04-26-pld-fsm-landed.md` | s2 recap | Smoke completed; CLI fix landed. |
| `.planning/measurements/session-14-spec-decode-K12.md` | session-15 K=16 numbers | Pre-tokenizer-mismatch data, superseded by E2E run. |
