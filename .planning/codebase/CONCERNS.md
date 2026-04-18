# Concerns

Technical debt, known issues, fragile areas, and risk surfaces in the Higgs codebase.

> Sourced from sub-agent exploration + existing `.planning/` handoff docs (`next-session-*.md`, `dflash-forensics-and-ane-target-plan.md`, `gdn-recurrence-debug-handoff.md`, `next-session-a3b-prefill-regression.md`).

## Performance Regressions

### DFlash speculative decoding — 38% perf loss on Qwen3.5-9B
- **Files:** `crates/higgs-models/src/dflash.rs` (3,963 lines), `dflash_ane.rs` (1,718 lines), `dflash_cpu.rs`
- **Symptom:** Speculative decoding path is slower than non-speculative baseline on 9B.
- **Related handoffs:** `.planning/next-session-dflash-regression.md`, `.planning/dflash-forensics-and-ane-target-plan.md`, `.planning/next-session-verify-bottleneck.md`.
- **Recent commits:** `782360c0 feat(dflash): wire rejection sampling for temperature>0`, `22bf8f15 feat(dflash): env-gate pipeline mode via HIGGS_DFLASH_PIPELINE`, `c1f85ade fix(dflash): revert pipeline=true, add verify_build_ms timer`.
- **Status:** Mitigated via env gate (`HIGGS_DFLASH_PIPELINE`) — pipeline=true reverted because it regresses. Root cause not resolved.

### A3B prefill regression
- **Handoff:** `.planning/next-session-a3b-prefill-regression.md` (16 KB) — active debugging document.
- **Risk:** Unclear if fixed; investigate before touching prefill paths.

### 27B DFlash crash
- **Handoff:** `.planning/next-session-27b-dflash-crash.md`.
- **Risk:** DFlash at 27B scale crashes; not clear if guard is in place.

## Fragile Architectural Areas

### `qwen3_next.rs` — 15,856-line monolith
- Hybrid SSM + attention + MoE + ANE offload all in one file.
- Split across `qwen3_next.rs`, `qwen3_next_ane.rs`, `qwen3_next_ane_worker.rs`.
- Modifying one region risks breaking unrelated ones.
- **Mitigation idea:** Decompose into sub-modules (attention block, SSM block, MoE routing, ANE worker FFI) before any major refactor.

### ANE bridge (`ane_bridge.rs`, `ane_mil.rs`, `ane_extract.rs`, `ane_forward.rs`, `ane_mlmodel.rs`)
- CoreML / ANE coordination untested under contention.
- Race conditions / deadlocks possible in multi-request scenarios.
- `ane_mil.rs` is 2,290 lines — heavy MIL generation logic.
- Metal kernel lifecycle: unsafe `Drop` implementations rely on process exit for cleanup.

### DFlash heterogeneous ANE + CPU execution
- Manual NEON intrinsics for CPU verify path (`dflash_cpu.rs`).
- Untested edge cases on boundary conditions (short sequences, batch=1 vs batch>1).
- Draft-accept rejection sampling recently added (`782360c0`) — coverage unclear.

### GDN recurrence
- **Handoff:** `.planning/gdn-recurrence-debug-handoff.md` — active debugging context.

## Dependency Risks

### MLX-RS version pinning
- Silent 38% regression if metallib cache is not cleared after version changes.
- **Mitigation:** Document cache-clear procedure; consider post-build cache invalidation.

### Unbounded upstream version constraints
- Some crates pinned to specific versions to avoid regression; others floating.
- A dependency audit pass would surface which are intentional pins vs accidental.

## Security

### Chat template Jinja rendering
- **File:** `crates/higgs-engine/src/chat_template.rs`.
- Templates come from model config — treated as trusted but executed as Jinja.
- If a compromised model config is loaded, template injection is possible.
- **Mitigation:** Sandbox renderer or validate template source.

### FFI boundaries (ANE bridge)
- Error propagation across Rust ↔ Objective-C / CoreML boundary uses raw pointers and status codes.
- Race conditions possible if errors are not handled synchronously.

## Test Coverage Gaps

| Area | Gap |
|------|-----|
| ANE coordination | No concurrency / stress tests |
| DFlash rejection sampling | No unit tests covering temperature > 0 sampling correctness |
| Streaming tokens | Limited coverage of mid-stream errors / disconnects |
| Multimodal images | `llava_qwen2.rs` + `siglip.rs` lack image-contract tests |
| Diffusion pipelines | `diffusion*.rs` (5 files) — training + inference paths mostly untested in integration |
| Router auto-selection | `auto_router.rs` — limited unit coverage |
| CLI exec | `cli_exec.rs` integration test exists but narrow |

## Code Hygiene

### Unused / dead code
- Scattered unused imports and helpers in model files.
- Examples mentioned by the exploration: `softplus` function, `KvCacheView` struct.
- **Mitigation:** `cargo clippy -- -W dead_code` sweep and prune.

### Benchmarks folder pollution
- `benchmarks/` contains dozens of timestamped log files checked into git (`bench_context_sweep_20260407_*.txt/json`, `ane_clean_rerun_*.log`).
- These should live in `benchmarks/results/` (gitignored) or be archived.

### `.planning/` accumulation
- Many `next-session-*.md` handoff files — some resolved, some active.
- Unclear which are current without reading each one.
- **Mitigation:** `/gsd:cleanup` or explicit archival step.

## Documentation Gaps

- ANE prefill design (`docs/ane-prefill-design.md`) exists but may not reflect post-regression state.
- No single entry document for DFlash state after the pipeline revert.
- Architecture diagram (`docs/higgs-architecture-diagram.md`) likely predates recent ANE/DFlash additions.

## Operational Risks

### Doctor validation lag
- Project rule: new config fields must update `crates/higgs/src/doctor.rs`.
- Enforcement is cultural, not automated — easy to forget on branches.

### Test parallelism
- `cargo test -p higgs` requires `--test-threads=1` due to shared port bindings.
- Anyone running `cargo test` without the flag will hit flaky failures.
- **Mitigation:** `.cargo/config.toml` test runner wrapper, or randomize port per test.

## Priority Summary

| Priority | Concern |
|----------|---------|
| HIGH | DFlash 38% regression (unresolved root cause) |
| HIGH | `qwen3_next.rs` 15k-line monolith (changes are high-risk) |
| HIGH | ANE coordination untested under contention |
| MED | A3B prefill regression (active handoff) |
| MED | 27B DFlash crash (active handoff) |
| MED | MLX-RS cache-stale silent regression |
| LOW | Benchmarks folder pollution, .planning accumulation |
| LOW | Dead code / unused imports |
