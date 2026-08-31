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

## The mission (unchanged): 70B-class on 32 GB via native IQ kernels

- IQ2/IQ3 codebook-kernel path (PLAN.md "GGUF Q4_K + IQ quant" section): the eschamoe
  simdgroup GEMM pattern (gather → threadgroup decode → fragment MMA) transfers to the
  codebook-lookup decode. This is the differentiator; next session starts here.
- BM=64 decode-halving insight feeds the same kernel work.

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
