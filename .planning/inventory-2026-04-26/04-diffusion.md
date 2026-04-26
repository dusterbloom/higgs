# Diffusion / BD3LM / Bonsai / Eggroll Inventory — feat/magic-canvas

## Research thesis

higgs is being built on two parallel tracks: (1) push autoregressive (AR) Qwen-class throughput on Apple Silicon to ~20-200 tok/s via MLX optimization, and (2) explore non-AR / parallel decoding regimes (block-diffusion, masked diffusion, speculative drafting) that change the per-token compute curve. The diffusion track is motivated by `docs/sota-diffusion-lms-2026.md`, which argues the discrete/masked diffusion frontier (LLaDA 2.x, BD3LM, DiffuCoder, Dream-7B) now matches mid-size AR quality and offers algorithmic 5-10x speedups that are platform-independent. Bonsai (1-bit quantized Qwen3-class model) sits at the intersection: a high-density AR baseline that the user has already LoRA-tuned to a BD3LM block-diffusion variant in Colab (PPL 16.2). The goal of the diffusion subprojects is to land that exact-quality BD3LM Bonsai-8B locally, and to keep an opportunistic eye on adjacent acceleration ideas (Eggroll runtime training, ShadowKV, TEAL).

## Subprojects

### Bonsai-Q1 (1-bit Qwen3 AR engine)

- Goal: AR parity with PrismML's mlx-lm 1-bit Bonsai.
- Approach: route through `mlx_rs::ops::quantized_matmul(bits=1)` against the local PrismML cherry-picks in `mlx-sys/src/mlx-c` (commit `ed45aec`). Phase A instrumented `forward_trunk` with per-section timers; B1 added dynamic rope offset + KV pre-alloc + compile_with_state wrap; session-28 isolated a fp16->f32 silent upcast in `apply_yarn_rope`.
- Headline: **2.83x speedup on 8B** (44.46 -> 15.69 ms/step, 22 -> 64 tok/s); 2.11x on 1.7B (87 -> 184 tok/s). Now within ~12% of mlx-lm Python (14 ms/step).
- Status: AR parity essentially closed. Three identical fp16-upcast bugs still pending in `deepseek_v2.rs:219`, `deepseek_v2.rs:625`, `siglip.rs:108`.
- Key files: `crates/higgs-models/src/bonsai_q1.rs`, `crates/higgs-models/src/yarn.rs`, `crates/higgs-models/tests/bisect_decode.rs`, `crates/higgs-models/tests/qmm_only_decode.rs`.

### BD3LM Bonsai-8B (block-diffusion in higgs)

- Goal: run the user's Ternary-Bonsai-8B + BD3LM LoRA + denoise head + mask_emb locally with exact Colab parity (PPL 16.2, 5.5x algorithmic speedup vs AR).
- Approach: BF16 end-to-end (no quant), new `model_type="bd3lm_qwen3"`, `AnyModel::Bd3lmQwen3`, `AnyCache::Bd3lm`, denoise loop in `generate_bd3lm_inner`. Reuses existing `TransformerModel` via `forward_from_embeddings`; injects `mask_emb` at masked positions; KV rollback per denoising step.
- Headline: **dispatch works, output is broken** — Rust generates 64 commas on smoke prompt; Python parity check reproduces same comma pattern, ruling out Rust bugs and pointing at denoise_head architecture / base mismatch / weight tying.
- Status: blocked on denoise-head architecture audit. Rust scaffolding (Phase B) and dispatch (Phase C) complete; Phase D bench gated.
- Key files: `crates/higgs-models/src/bd3lm_qwen3.rs`, `crates/higgs-engine/src/simple.rs:generate_bd3lm_inner`, `scripts/bd3lm_parity_check.py`, `scripts/prep_bd3lm_bonsai.py`, `bonsai-bd3lm-merged-bf16/bd3lm_extras.safetensors` (~1.2 GB, off-tree).

### Eggroll (gradient-based runtime training)

- Goal: replace Evolution Strategies (pop x 2 forwards/step) with backprop through a `stop_gradient(qmm(W,x)) + delta @ x` decomposition. Train deltas on a frozen quantized base.
- Approach: pure mlx PoC (`scripts/eggroll_v2_poc.py`); 35B validation harness against running higgs server (`scripts/validate_eggroll_35b.py`) progressing through pop=1/1step -> pop=4/20step.
- Headline: PoC and validation scripts present; no commits to runtime engine. Targets `NexVeridian/Qwen3.5-35B-A3B-3bit`.
- Status: research scripts only; not wired into any crate. Speculative.
- Key files: `scripts/eggroll_v2_poc.py`, `scripts/validate_eggroll_35b.py`. Persistence via `~/.nanobot/experience.db`.

### Denoise-head experiments (BD3LM blocker investigation)

- Goal: figure out why `denoise_head` from `bd3lm_ckpt.pt` decodes commas instead of tokens.
- Approach: Python-side parity reproduces the failure; tests both LoRA-applied and not. Ruled out Rust forward bugs.
- Outstanding hypotheses (per `next-session-bd3lm-denoise-head-blocker.md`): wrong base hash, wrong activation between `denoise_head.1` and `.3` (assumed GELU, may be SiLU), tied embedding (`denoise_head.3.weight` shape matches `embed_tokens.weight`), undertrained checkpoint (step=6000), `mask_emb` scaling.
- Key files: `crates/higgs-models/src/bd3lm_qwen3.rs:35-58` (DenoiseHead), `scripts/bd3lm_parity_check.py`.

### Probe scripts (research, off the hot path)

- `scripts/probe_q_smoothness_and_krank.py` — per-layer post-RoPE Q smoothness + K rank for temporal-caching / sketch-archive feasibility on Qwen3.5-35B-A3B-3bit.
- `scripts/probe_analysis.py` — earlier residual-stream proxy version.
- `scripts/shadowkv_prototype.py` — Phase 0 quality gate for invariant-dim sketch attention selection on 8K context (mass recall, top-32 recall, KL gates).
- `scripts/dflash_probe_latency.py` / `scripts/dflash_probe_cpu_draft.py` — drafter+verify latency math for spec-decode bandwidth budgeting on M4.
- Status: all standalone, not wired into engine. Used to gate roadmap features.

## Bonsai parity report

From `.planning/measurements/bonsai-parity/REPORT.md` and `decode-breakdown.md`:

Pre-fix (2026-04-24, instrumented):

| Model | higgs decode | PrismML decode | Gap |
|---|---:|---:|---:|
| Bonsai-1.7B | 86.6 tok/s | 226.5 tok/s | 2.62x |
| Bonsai-8B | 22.2 tok/s | 71.5 tok/s | 3.22x |

Phase A decode breakdown (Bonsai-8B, 64 steps, eval-per-section, 99.8% accounted, 163.34 ms/step instrumented):

| Class | % accounted |
|---|---:|
| Matmul (qmm) — mlp_up_gate, qkv_proj, mlp_down, o_proj, lm_head | 59.4% |
| Norms (qk_norm, post_attn, input, final) | 15.5% |
| Residuals (2 adds/layer) | 8.0% |
| RoPE (2 fast::rope/layer) | 7.6% |
| SDPA + KV write | 4.9% |
| silu_mul | 4.3% |
| Embed + misc | 0.2% |

Per-section detail (microseconds/step): mlp_up_gate 36697.8 (22.5%), qkv_proj 27467.3 (16.8%), mlp_down 19196.6 (11.8%), residual 12977.0 (8.0%), rope 12568.7 (7.7%), qk_norm 12401.2 (7.6%), o_proj 11052.8 (6.8%), sdpa_kv 8101.0 (5.0%), silu_mul 7071.6 (4.3%), input_norm 6498.6 (4.0%), post_attn_norm 6242.9 (3.8%), lm_head 2360.2 (1.4%), embed_rows 201.3, final_norm 176.7.

Post-session-28 fix (fp16 dtype hold-through, `yarn.rs`):

| Model | Before | After | Speedup |
|---|---:|---:|---:|
| Bonsai-1.7B | 11.51 ms (87 t/s) | **5.45 ms (184 t/s)** | 2.11x |
| Bonsai-8B | 44.46 ms (22 t/s) | **15.69 ms (64 t/s)** | 2.83x |

Stripped qmm-only floors stand at 2.14 ms (8B), 1.66 ms (1.7B); the remaining ~13.5 ms/step on 8B is genuine scaffolding now in the same ballpark as Python (14 ms/step ref).

## Decision log

- 2026-04-23 — TEAL killed for pure-MLX path: row-gather on up_proj caps at ~5%, not 1.7x; `gather_qmm` API was wishcasting (`RECAP-2026-04-23-session10-teal-dead-pivot.md`).
- 2026-04-24 (s12) — Bonsai shim integrated into drafter trait; foundational adapter landed.
- 2026-04-24 (s18) — Bonsai-Q1 packed loader landed (P2/bits=1 weight format wired).
- 2026-04-24 (s22) — `bench_bonsai_q1_anymodel_full_matrix` becomes the headline AR bench.
- 2026-04-24 (s23) — Phase A decode breakdown: 59.4% matmul, 35.4% scaffolding; "compile-wrap alone caps at ~1.5x" verdict written.
- 2026-04-24 (s24) — B1 steps 1+2 land: `fast::rope_dynamic` + KV pre-alloc to `max_tokens`.
- 2026-04-24 (s25) — B1 compile-wrap design committed: `forward_trunk_free`, hand-rolled `Updatable` over 705 arrays.
- 2026-04-24 (s26) — B1 compile-wrap LANDED but **regression** (1.7B 0.84x, 8B 0.94x); state-positional-swap costs > Metal fusion gain. Decision: stop wrapper plumbing, target matmul width / dequant fusion next.
- 2026-04-25 (s27) — Scaffolding gap isolated to `gpu.forward` wrapper code via `bisect_decode.rs` ladder (v6 vs v7 = 28 ms hidden cost).
- 2026-04-25 (s28) — Root cause = silent fp16->f32 upcast in `apply_yarn_rope` (`mlx_rs::array!(mscale)` with `mscale: f32` makes f32 scalar). Fix: `Array::from_f32(mscale).as_dtype(x.dtype())`. **2.83x on 8B / 2.11x on 1.7B.**
- 2026-04-25 (BD3LM) — denoise_head investigation declared blocker; scope frozen until Python parity coherent.

## Open threads

- BD3LM denoise-head fix: audit `/Users/peppi/Dev/diffusion_bonsai/` training script for activation index 2, embedding-tying status, mask_emb scaling.
- BD3LM AR-parity sanity: rerun Phase D with `num_denoising_steps=64` once decoder coherent (`next-session-bd3lm-phase-c-bench.md`).
- Bonsai-Q1: B2 matmul throughput push (qmm decode-shape qmv profiling, MLX tiling for B=1).
- Dtype audit: 3 known HIGH suspects (`deepseek_v2.rs:219`, `deepseek_v2.rs:625`, `siglip.rs:108`) plus full sweep across qwen2/3/3_next, gemma2, phi3, starcoder2, transformer, llada_moe, diffusion.
- LoRA-merge verification of `bonsai-bd3lm-merged-bf16/model.safetensors` (deferred behind denoise blocker).
- SIGSEGV when running all `bonsai_q1::tests` together (memory pressure 8B+1.7B in same process); pre-existing.
- Eggroll: validation against live higgs server has not run end-to-end at pop=4/20step.

## Recommended cleanup

- `next-session-bd3lm-bonsai.md` — superseded by `next-session-bd3lm-phase-c.md` then `next-session-bd3lm-phase-c-bench.md` then `next-session-bd3lm-denoise-head-blocker.md`. Original Phase A handoff; obsolete.
- `next-session-bd3lm-phase-c.md` — superseded by `next-session-bd3lm-phase-c-bench.md`. Design doc; consumed.
- `next-session-bd3lm-phase-c-bench.md` — superseded by `next-session-bd3lm-denoise-head-blocker.md`. Bench plan blocked behind denoise-head fix.
- `RECAP-2026-04-24-session24-b1-steps1-2-handoff.md` — superseded by `RECAP-2026-04-24-session26-b1-compile-wrap-landed.md`. Intermediate B1 step.
- `RECAP-2026-04-24-session25-b1-compile-wrap-design.md` — superseded by session-26 (design landed and benched).
- `RECAP-2026-04-24-session26-b1-compile-wrap-landed.md` — superseded by `RECAP-2026-04-25-session28-bonsai-decode-2.7x-dtype-fix.md`. Compile wrap was a regression; the real fix was the dtype bug. Keep for forensic context but mark superseded.
- `RECAP-2026-04-24-session23-bonsai-decode-breakdown.md` — superseded by session-28 root-cause findings (the 35% "scaffolding" hypothesis was wrong; it was f32 leak). Numbers in `decode-breakdown.md` remain referenceable; recap conclusions are stale.
- `RECAP-2026-04-25-session27-bonsai-scaffolding-gap-isolated.md` — superseded by session-28 (scaffolding gap was the dtype bug). Bisect bench it produced is still useful artifact.
