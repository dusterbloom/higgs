# Higgs Handoff — session close 2026-08-31 · Branch: nightly · Head: see git log

## State: clean. Full suite 761 passed, 0 failed.

## What changed this session

| item | commit | status |
|---|---|---|
| BM=64 "SG1-7 zeros" bug ROOT-CAUSED + fixed | 279725324 | ✅ kernel was never broken — launch grid.x was `blocks_n * 128` threads with 256-thread threadgroups → half the column blocks never dispatched. Grid is in THREADS. |
| QGEMM row block selectable | 279725324 | ✅ BM=32 default; `HIGGS_ESCHA_QGEMM_BM=64` runs the variant. BM=64 only wins decode-heavy even-run down-proj (1.87x vs 1.59x); ragged parity. |
| GGUF parser real-file correct | 0d486769f | ✅ u64 counts + u64-len strings + all 13 value types + data_start/alignment. Validated on real GGUF. |
| Q4_K dequant spec-correct | 1fcb8e0ac, 0d486769f | ✅ formula `d*sc*q - dmin*m`, real nibble layout (32 lows then 32 highs per page), scale high-bits per gguf-py. Bit-exact vs oracle on real file. |
| GGUF end-to-end forward | aff95a79a | ↩️ REVERTED — see below |
| Q5_0/Q8_0/Q6_K/F32/F16/BF16 dequant | aff95a79a | ↩️ REVERTED — see below |

## DECISION: GGUF work fully REVERTED — it was a detour

The whole gguf module (parser + dequant + e2e) was removed after research showed the
GGUF-trending catalog is qwen3_5/gemma4 arch — models higgs already runs natively with
better kernels. Ingress = competing with llama.cpp on its home turf, no differentiation.
Recovery if ever needed: commits 1fcb8e0ac → 0d486769f → aff95a79a (in order), plus the
original WIP skeleton at 2a105668d. Test assets in ~/.cache/higgs-gguf/. Do NOT resume
without a new product reason.

## Mission status update (2026-08-31, post-IQ-bench): fit-goal reachable via stock affine Q2

DOUBLE-BUFFER EXPERIMENT (closed, do not retry blind): applied m4-prefill-engine's
ping-pong threadgroup staging (cited repo, Apache-2.0) to eschamoe gather_qgemm_simd.
Correct (bit-identical semantics, 47 tests green) but consistently SLOWER at the
default (128,32,40) config: gate_up even 43.8 → 45.7/47.5 ms, down even 18.8 →
19.8/21.5 ms. Cause: doubling threadgroup footprint 10.6→21.2 KB halves resident
threadgroups; cross-tg latency hiding > intra-tg overlap on 10-core M4. A future retry
needs either an occupancy-neutral scheme (f16 staging halves footprint) or a shape
where staging truly dominates (BM=64 variant untested).

IQ codebook kernel project (tag archive/iq-codebook) ran its Phase-2 gate and LOST:
bit-exact IQ2_XS/IQ3_XXS qmv kernels benched 2.0-2.3x SLOWER than stock MLX affine
bits=2 at M=1 (546 vs 1185 µs @ 17408x5120). At 2.31 bpw there is no bandwidth prize
over affine-2; unwired per kill criterion. Recovery: tag archive/iq-codebook (format +
quantizer + kernels, golden-vector validated vs gguf-py; design doc in
.planning/DESIGN-iq-codebook-kernels.md). Only revisit if a QUALITY case for codebook
vs affine at 2.3 bpw is proven (needs a cosine harness first).

The 70B-on-32GB fit-goal does NOT need IQ: stock affine Q2/g64 = 2.25 bpw ≈ 19.7 GB,
conversion pipeline already exists (eschamoe.rs convert_checkpoint + quantize_affine),
native MLX kernels win decode. Cheapest honest next step if the mission resumes:
Phase-0 proof — convert an existing 70B to Q2/g64, measure RAM/tok-s on the M4.

## Facts worth keeping

- Old BM=64 perf claims in 2bdfd8bf3's message are INVALID (measured with the grid bug).
- `mlx_rs::ops::quantization` exposes `quantize`/`quantized_matmul`/`gather_qmm` — MLX's
  native quant kernels are reachable from Rust if ever needed. No custom Metal needed for
  standard affine shapes.
- 2026 model landscape: Qwen3.5/3.6/3.8 (+Ornith-1.5, Tiel-Coder) are ALL `qwen3_5` hybrid
  arch (ssm + full-attn interval 4) → higgs qwen3_next path. GLM-5.3 = glm5_next (not supported).
- No real MoE router weights exist on this machine (all stubs) — "actual router" measurements
  are blocked until weights return.
- GGUF test assets: ~/.cache/higgs-gguf/ (SmolLM2-135M Q4_K_M + tokenizer.json).
- Env-gated tests: HIGGS_GGUF_TEST_FILE, HIGGS_GGUF_ORACLE_TENSOR, HIGGS_GGUF_E2E_FILE,
  HIGGS_GGUF_E2E_TOKENIZER. No .venv in repo root (Rust project).
