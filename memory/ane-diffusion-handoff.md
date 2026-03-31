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
- 104 tok/s at seq=128, 3.6W system power, 35 mJ/token
- Layer 0 does full compile, layers 1-27 use `patch_from_donor` (fast)
- **First diffusion LM on ANE ever**

## Key Architecture Decisions

### ANE Constraints (hard-won, don't re-learn these)
- **32MB BLOBFILE limit per program** — 0.6B model at fp16 = 30MB/layer (barely fits)
- **~119 loaded program limit** — why we use 28 fused kernels not 196 individual
- **spatial >= 16 required** — seq < 16 compiles but fails eval with `0x1d`
- **concat op fails on `_ANEClient` path** — must use `_ANEInMemoryModel` (compile_multi_weights)
- **~490µs dispatch floor** — hardware scheduling overhead per eval()
- **Channel-first layout**: IOSurface is `[1, channels, 1, spatial]` fp32

### BLAS vs ANE Crossover
- **seq=1**: BLAS wins (130µs vs 224µs) — ANE dispatch overhead dominates
- **seq≥16**: ANE wins (2.6x at seq=16, 4.4x at seq=128)
- **Diffusion LMs operate at seq=64-128** → ANE sweet spot

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
| `crates/higgs-models/src/diffusion_ane.rs` | `gen_fused_diffusion_layer()` MIL generator | ✅ Working |
| `crates/higgs-models/src/rwkv7.rs` | RWKV-7 CPU model + AneContext | ✅ Working |
| `crates/higgs-models/src/ane_bridge.rs` | ANE FFI bridge (compile, eval, IOSurface, blobs) | ✅ Working |
| `crates/higgs-models/src/ane_mil.rs` | MIL codegen (blobfile_matmul, fused_qkv, etc.) | ✅ Working |
| `crates/higgs-models/src/ane_extract.rs` | Weight extraction from MLX Arrays | ✅ Working |
| `crates/higgs-models/src/ane_forward.rs` | RWKV-7 BLAS decode (LoRA, WKV recurrence) | ✅ Working |

## What to Build Next (Priority Order)

### 1. Int8 BLOBFILE Weights (1-2 days, Sonnet can do with clear spec)

**Goal**: Halve BLOBFILE size → fit 1.2B models in 32MB limit.

**How**: In `diffusion_ane.rs`, change each BLOBFILE const from:
```mil
tensor<fp16, [1,1,{ic},{oc}]> W = const()[name=string("W"),
  val=tensor<fp16, [1,1,{ic},{oc}]>(BLOBFILE(...))]
```
to:
```mil
tensor<int8, [1,1,{ic},{oc}]> W_int8 = const()[name=string("Wi"),
  val=tensor<int8, [1,1,{ic},{oc}]>(BLOBFILE(...))]
fp16 W_scale = const()[name=string("Ws"), val=fp16({scale})]
fp16 W_zero = const()[name=string("Wz"), val=fp16(0)]
tensor<fp16, [1,1,{ic},{oc}]> W = constexpr_affine_dequantize(
  quantized_data=W_int8, scale=W_scale, zero_point=W_zero, axis=0)
```

Use `build_weight_blob_quantized(weights, rows, cols)` which returns `(blob, scale)`.

**Test**: Compare int8 logits vs fp16 logits — max error should be < 1.0.

### 2. LLaDA-MoE-7B-A1B on ANE (1 week, needs Opus for architecture)

**Model**: `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` — config already downloaded.
- hidden=2048, 16 layers, 16 heads (MHA), 64 experts, 8 active per token
- expert_intermediate=1024 (tiny!), dense_intermediate=8192

**Architecture**:
- Attention kernel (int8): ~16MB → fits one kernel
- MLP experts: 8 active × 3 × 2048 × 1024 = ~50MB fp16, ~25MB int8 → fits one kernel
- 2 dispatches/layer × 16 layers = 32 dispatches
- Estimated: ~100 tok/s at seq=128

**Needs**: Expert routing on CPU (top-8 selection), expert weight gathering, new MIL for MoE MLP.

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
| **ANE fused** | **Qwen3-0.6B diffusion (seq=128)** | **104** | **35** | **3.6** |

## Agent Delegation Guide

**Opus**: Architecture decisions, debugging ANE compile failures, MoE expert routing design, MIL program generation for new architectures. Use sparingly.

**Sonnet**: Well-specified coding tasks with clear input/output. Good for:
- Int8 BLOBFILE swap (spec above is complete)
- Writing new tests
- Wiring existing components together
- Weight extraction for new models

**Haiku**: File reads, greps, simple searches. Good for:
- Checking weight shapes in safetensors
- Reading configs
- Running existing tests
- Searching HuggingFace for models
