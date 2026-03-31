# ANE Diffusion LM — Session Handoff

## What Was Built (2 commits on `feat/rwkv7-ane-v2`)

### Commit 1: `1d33453` — RWKV-7 + Diffusion + ANE infrastructure (6,510 lines)
### Commit 2: `63891d9` — Fused 1-dispatch-per-layer ANE engine

## Working E2E Systems

### 1. RWKV-7 CPU Engine (`crates/higgs-models/src/rwkv7.rs`)
- Loads `fla-hub/rwkv7-1.5B-world` from `~/.cache/huggingface/hub/`
- 11.5 tok/s decode (release), token-identical to Python reference
- BLAS sgemv for projections, BLAS sger+sgemv for WKV recurrence
- Registry wired: auto-detects `model_type: "rwkv7"`

### 2. Diffusion LM BLAS Engine (`crates/higgs-models/src/diffusion.rs`)
- Loads `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` from `~/.cache/huggingface/hub/`
- 350-line self-contained Qwen3 encoder, no MLX dependency
- `DiffusionEngine::load(path)` → `forward(token_ids)` → `generate(prompt, n_tokens, steps)`
- 22 tok/s (BLAS, release), logits match Python to 0.0005 error
- Generates "The capital of France is Paris." correctly

### 3. Diffusion LM ANE Engine (`diffusion.rs` + `diffusion_ane.rs`)
- `AneDiffusionEngine::new(engine, seq_len)` compiles 28 fused BLOBFILE kernels
- Each kernel = full transformer layer: RMSNorm→QKV→QKnorm→RoPE→SDPA→Wo→residual→RMSNorm→FFN→residual
- 99-104 tok/s at seq=128, 3.6W system power, 35 mJ/token
- Layer 0 does full compile, layers 1-27 use `patch_from_donor` (fast)
- **First diffusion LM on ANE ever**

### 4. Multi-dispatch generators (NEW — `diffusion_ane.rs`)
- `gen_diffusion_attention()` — attention-only MIL (9 BLOBFILEs, ~12MB at dim=1024)
- `gen_diffusion_ffn()` — FFN-only MIL (4 BLOBFILEs, ~18MB at dim=1024)
- Chained output is **bit-identical** to fused (max_err = 0.000000)
- 55 tok/s at seq=128 (1.81x overhead vs fused due to 56 vs 28 dispatches)
- Each extra dispatch costs ~564µs (490µs hardware floor + I/O copy)

## Key Architecture Decisions

### ANE Constraints (hard-won, don't re-learn these)
- **32MB BLOBFILE limit per program** — 0.6B model at fp16 = 30MB/layer (barely fits)
- **~119 loaded program limit** — why we use 28 fused kernels not 196 individual
- **spatial >= 16 required** — seq < 16 compiles but fails eval with `0x1d`
- **concat op fails on `_ANEClient` path** — must use `_ANEInMemoryModel` (compile_multi_weights)
- **~490µs dispatch floor** — hardware scheduling overhead per eval()
- **Channel-first layout**: IOSurface is `[1, channels, 1, spatial]` fp32
- **Int8 BLOBFILE: DEAD** — `tensor<int8>` rejected by `ANECCompile()`, `constexpr_affine_dequantize` also rejected. Tested 3 patterns (cast, dequant matmul, dequant conv1x1). nanobot-rs code is dead/untested. Not a viable path.

### BLAS vs ANE Crossover
- **seq=1**: BLAS wins (130µs vs 224µs) — ANE dispatch overhead dominates
- **seq≥16**: ANE wins (2.6x at seq=16, 4.4x at seq=128)
- **Diffusion LMs operate at seq=64-128** → ANE sweet spot

### Multi-dispatch vs Fused (benchmarked)
- **Fused (1 dispatch/layer)**: 99 tok/s, 19.4ms/step — use when layer fits under 32MB
- **Multi-dispatch (2 dispatches/layer)**: 55 tok/s, 35.2ms/step — use when layer > 32MB
- Overhead: 1.81x (extra dispatch + IOSurface read/write between kernels)
- Correctness: bit-identical (fp16→fp32→fp16 round-trip is lossless)

### Weight Blob Layout
- PyTorch stores `[out_features, in_features]`
- MIL matmul expects `[1, 1, in_channels, out_channels]`
- Use `build_weight_blob_transposed(weights, out, in)` to transpose
- Norm weights: `build_weight_blob(norm_w, 1, dim)` → `[1, dim, 1, 1]`
- RoPE: `build_weight_blob(cos, seq, half_hd)` → `[seq, half_hd]`

## Files Map

| File | Purpose | Status |
|------|---------|--------|
| `crates/higgs-models/src/diffusion.rs` | BLAS engine + AneDiffusionEngine | ✅ Working |
| `crates/higgs-models/src/diffusion_ane.rs` | `gen_fused_diffusion_layer()` + `gen_diffusion_attention()` + `gen_diffusion_ffn()` | ✅ Working |
| `crates/higgs-models/src/rwkv7.rs` | RWKV-7 CPU model + AneContext | ✅ Working |
| `crates/higgs-models/src/ane_bridge.rs` | ANE FFI bridge (compile, eval, IOSurface, blobs) | ✅ Working |
| `crates/higgs-models/src/ane_mil.rs` | MIL codegen (blobfile_matmul, fused_qkv, etc.) | ✅ Working |
| `crates/higgs-models/src/ane_extract.rs` | Weight extraction from MLX Arrays | ✅ Working |
| `crates/higgs-models/src/ane_forward.rs` | RWKV-7 BLAS decode (LoRA, WKV recurrence) | ✅ Working |

## What to Build Next (Priority Order)

### ~~1. Int8 BLOBFILE Weights~~ — DEAD END
**Confirmed dead**: ANE's raw MIL compiler (`_ANEDesc modelWithMILText:`) rejects `tensor<int8>` entirely. Three patterns tested:
1. `tensor<int8>` + `cast(dtype="fp16")` → InvalidMILProgram
2. `tensor<int8>` + `constexpr_affine_dequantize` (matmul) → InvalidMILProgram
3. `tensor<int8>` + `constexpr_affine_dequantize` (conv1x1, nanobot-rs pattern) → InvalidMILProgram

The nanobot-rs `gen_conv1x1_int8_blob()` at `ane_mil.rs:4318` was never called in production — dead code.

### 1. LLaDA-MoE-7B-A1B on ANE (needs Opus for architecture)

**Model**: `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` — config already downloaded.
- hidden=2048, 16 layers, 16 heads (MHA), 64 experts, 8 active per token
- expert_intermediate=1024 (tiny!), dense_intermediate=8192

**Problem**: dim=2048 MHA doesn't fit in 2 dispatches:
- Attention: 4 × 2048×2048 × fp16 = 32MB exactly + norms/RoPE → **33.6MB (over limit)**
- FFN MoE: 8 active experts × 3 × 2048×1024 × fp16 = **100.7MB (way over)**

**Possible architectures** (needs investigation):
- **3+ dispatches/layer**: Split attention into (QKV dispatch ~24MB) + (SDPA+Wo dispatch ~8MB). Split MoE into per-expert dispatches (~12.6MB each). Total: ~3 attn + 8 expert = 11 dispatches/layer × 16 layers = 176 dispatches. At 564µs/dispatch = ~99ms/step. At seq=128, 64 steps → ~1.6 tok/s. **Probably too slow.**
- **Hybrid CPU+ANE**: Do expert routing + MoE on CPU (BLAS), attention on ANE. More practical.
- **Wait for smaller diffusion MoE models**: dim=1024 MoE would fit fused.

**Needs**: Weights downloaded, expert routing implementation, MoE MIL generator.

### 2. Wire Multi-dispatch into AneDiffusionEngine (Sonnet can do)

Currently `AneDiffusionEngine::new()` only uses `gen_fused_diffusion_layer()`. Add a fallback path:
- If `gen_fused` BLOBFILE total > 32MB → use `gen_diffusion_attention` + `gen_diffusion_ffn`
- Compile 2 kernels instead of 1
- Forward loop: reload_weights for attn → eval → read → reload_weights for FFN → eval → read
- This makes the engine work for any model dim without changing the public API

### 3. LoRA Training on ANE (2 weeks, needs Opus)

nanobot-rs has backward kernels (`gen_fused_layer_bwd`). Port for diffusion:
- Forward: existing fused kernel
- Backward: reverse the computation graph
- Weight update: Adam on CPU + delta_reload to ANE
- LoRA: tiny adapter matrices as IOSurface inputs (zero recompile on swap)

## Test Commands

```bash
# RWKV-7 CPU forward (no ANE feature needed)
cargo test -p higgs-models --release -- rwkv7::tests::test_rwkv7_model_load_and_forward --nocapture

# Diffusion BLAS forward + generate
cargo test -p higgs-models --release -- diffusion::tests::test_load_and_forward --nocapture
cargo test -p higgs-models --release -- diffusion::tests::test_generate --nocapture

# ANE fused layer benchmark (needs --features ane)
cargo test -p higgs-models --features ane --release -- diffusion_ane::tests::test_fused_layer --nocapture

# ANE multi-dispatch correctness + benchmark
cargo test -p higgs-models --features ane --release -- diffusion_ane::tests::test_multi_dispatch_vs_fused --nocapture

# ANE E2E generate
cargo test -p higgs-models --features ane --release -- diffusion::tests::test_ane_generate --nocapture

# Full MIL benchmark sweep (BLAS vs ANE vs GPU)
cargo test -p higgs-models --features ane --release -- ane_mil::tests::test_blas_vs_ane --nocapture
```

## Models Downloaded

| Model | Location | Size | Status |
|-------|----------|------|--------|
| RWKV-7 1.5B | `~/.cache/huggingface/hub/models--fla-hub--rwkv7-1.5B-world/` | ~3GB | ✅ Full |
| Qwen3-0.6B-diffusion | `~/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/` | ~1.2GB | ✅ Full |
| DiffutronLM-0.3B (Turkish) | `~/.cache/huggingface/hub/models--diffutron--DiffutronLM-0.3B-Instruct/` | ~0.6GB | ✅ Full |
| LLaDA-MoE-7B-A1B | `~/.cache/huggingface/hub/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/` | Config only | ⬜ Weights not downloaded |

## Benchmarks (M4 Max, release mode, on battery)

| Engine | Model | tok/s | mJ/token | System W |
|--------|-------|-------|----------|----------|
| BLAS sgemv | RWKV-7 1.5B (decode, seq=1) | 11.5 | 341 | 3.9 |
| BLAS sgemm | Qwen3-0.6B diffusion (seq=128) | 10 | 354 | 3.5 |
| **ANE fused** | **Qwen3-0.6B diffusion (seq=128)** | **99-104** | **35** | **3.6** |
| ANE multi-dispatch | Qwen3-0.6B diffusion (seq=128) | 55 | ~65 | ~3.6 |

### Multi-dispatch overhead breakdown
| Dispatches | Time (28 layers) | tok/s (64-step) | Per-dispatch |
|-----------|-----------------|-----------------|-------------|
| 28 (fused) | 19.4ms | 99 | 693µs |
| 56 (multi) | 35.2ms | 55 | 629µs |
| Delta: +28 | +15.8ms | -44 | ~564µs/extra |

## Agent Delegation Guide

**Opus**: Architecture decisions, debugging ANE compile failures, MoE expert routing design, MIL program generation for new architectures. Use sparingly.

**Sonnet**: Well-specified coding tasks with clear input/output. Good for:
- Wiring multi-dispatch into `AneDiffusionEngine` (fallback path)
- Writing new tests
- Wiring existing components together
- Weight extraction for new models

**Haiku**: File reads, greps, simple searches. Good for:
- Checking weight shapes in safetensors
- Reading configs
- Running existing tests
- Searching HuggingFace for models
