# LFM3 1.7B feasibility and implementation plan

**Goal:** Run LiquidAI LFM3-1.7B natively in Higgs on Apple Silicon, initially with the official MLX 4-bit checkpoint.

**Status:** Planning estimate, not an approved implementation specification. No model downloaded or runtime code changed.

**Architecture:** Add a dedicated LFM3 model implementation using existing MLX quantization, hybrid-cache, convolution and attention primitives where their semantics match. Keep the existing LFM2-MoE path intact. Establish correctness before custom Metal optimization.

**Tech stack:** Rust, mlx-rs, safetensors; Python/Transformers for independent reference outputs.

## Evidence and uncertainty

- Repository: higgs at `91d4a5e0398c0ac12fb873f1e2f0a074fa44b5c5`; GitNexus index matches HEAD.
- `lfm2_moe.rs` already implements gated short convolutions, attention with Q/K normalization, and dense feed-forward layers alongside MoE.
- `gemma3.rs` supplies an existing sliding-window masking pattern.
- `adapter.rs`, `registry.rs`, `lib.rs` and engine `model_loader.rs` require model wiring; adding a registry string alone is insufficient.
- Official LFM3 model card describes 24 layers: 13 convolution, 5 sliding-window attention, 6 global attention; 256K advertised context.
- Official MLX 4-bit model card reports 0.97 GB of weights, affine quantization, group size 64. It explicitly says MLX-LM support is in progress.
- Initial browser retrieval failed; subsequent inspection fetched upstream source/config/template and the MLX safetensors header successfully. Findings appear below. No tensor payloads were downloaded and no inference was executed.

Sources: https://huggingface.co/LiquidAI/LFM3-1.7B and https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit

## Scope

Initial release: single-request text generation, prefill and incremental decode, official 4-bit weights, streaming chat, correct stop behavior. Then validate tool calls and retained sessions before advertising agent support.

Defer full 256K validation, speculative decoding, new custom kernels and multimodal variants. Weight size excludes KV cache and runtime buffers.

## Work packages and gates

| Work | Estimate (focused engineer hours) | Deliverable / acceptance gate |
| --- | ---: | --- |
| 1. Inspect pinned upstream code, download checkpoints, establish reference | 3–5 | Record config, tensor names/shapes, quantization overrides, tokenizer and stop IDs. Produce deterministic reference logits and replies using reviewed upstream Transformers code; confirm execution on available hardware. |
| 2. Implement model and checkpoint loading | 8–14 | New `crates/higgs-models/src/lfm3.rs`. Implement exact layer schedule, convolution, local/global attention, feed-forward, norms and output head. Synthetic checkpoint tests in `crates/higgs-models/tests/lfm3.rs` cover load, invalid config and finite logits. |
| 3. Integrate model and cache lifecycle | 4–8 | Wire `registry.rs`, `adapter.rs`, `lib.rs`, engine `model_loader.rs`. Verify one-shot vs chunked prefill, incremental decode, positions across local-window boundaries, reset and independent-request isolation. Disable unsupported capabilities explicitly. |
| 4. Numerical parity and real generation | 5–9 | New `scripts/lfm3_parity.py` and model example `lfm3_logits.rs`, following the LFM2 parity harness. Compare same token IDs against reference, locate first divergent layer, test multiple chunk sizes. Set dtype-aware tolerances from reference self-comparisons rather than borrowing LFM2 thresholds blindly. |
| 5. Chat/API and agent behavior | 4–8 | New engine template test `crates/higgs-engine/tests/lfm3_template.rs`. Match upstream template bytes for plain, multi-turn and tool conversations. Verify streaming, EOS, tool arguments and reasoning behavior if present; modify existing parsers only if fixtures show a mismatch. |
| 6. Regression, benchmarks and docs | 3–5 | Run model/engine regressions including LFM2. Measure cold load, prefill, decode and peak memory at stated context lengths. Update `docs/models.md` and configuration example with tested limits. |

Total: **27–49 focused hours**, approximately **4–7 working days** for one engineer familiar with this codebase. Initial load-and-generate milestone: roughly **12–24 hours**, with limited correctness assurance until parity gates pass. Budget **1–2 calendar weeks** including investigation and review. Unusual reference/backend or numerical issues may add **2–4 days**.

## Execution order

- [ ] Complete reference inspection and revise this estimate against actual tensor/operator differences.
- [ ] Add failing synthetic model tests, implement model, and verify those tests.
- [ ] Add cache equivalence tests before integrating production cache reuse.
- [ ] Establish real-checkpoint numerical parity before accepting fluent output as evidence.
- [ ] Validate chat and tools through both direct and streaming API paths.
- [ ] Run `cargo test -p higgs-models --test lfm3`, `cargo test -p higgs-models --test lfm2_moe`, and `cargo test -p higgs-engine`; add checkpoint-backed tests as explicit opt-in tests.
- [ ] Benchmark measured context lengths and publish results without extrapolating to 256K.

Before runtime edits, run unambiguous GitNexus upstream impact analysis on affected symbols, report callers/processes/risk, and warn for HIGH/CRITICAL. An initial query for `load_model` was ambiguous (UNKNOWN), so it is not an authoritative blast-radius verdict. Resolve it by file/symbol identity before implementation. Before any commit, run complete graph change detection and resolve partial/truncated results.

## Alternatives

Recommended: native dedicated implementation reusing proven primitives. Extending LFM2-MoE in place risks coupling two architectures before their differences are established. An external inference server can provide earlier access if its LFM3 support is verified, but does not deliver native Higgs support.

## Inspection findings (2026-11-01)

Inspected official `modeling_lfm3.py` (42,359 bytes), `lfm3_conv.py`, chat template, MLX config and 74,994-byte tensor header. The MLX repository tree identifies the modeling source as Git blob `0ccffc7ed179c2e3aaa19ab93bba735abb1d124d`, config as `00840a1a0e83d11b0ac26876a0cf62f089d45010`, and the weight payload SHA-256 as `1be5bf672013cd91aa62148f77c5eea8d2784a95845d5fe826e08c7b84823815` (973,083,386 bytes). The source body was fetched from the original text checkpoint; the recorded blob is from the MLX tree, not a verified hash of that downloaded body.

### Confirmed implementation differences

| Area | Actual LFM3 behavior | Higgs work |
| --- | --- | --- |
| Convolution | Width 7; `out_proj(C * conv(B*x))`; no bias in this checkpoint | Existing LFM2 convolution loops/kernel parameterize K; arithmetic is a strong reuse candidate. Preserve six chronological prior inputs and verify chunk boundaries. |
| Attention | 16 Q heads, 4 KV heads, head dimension 128; Q/K RMSNorm; sigmoid output gate with one scalar per Q head | Existing GQA/QK norm primitives plus new quantized `gate_proj` and output multiplication. |
| Local attention | Five layers with window 256; six global layers | Reuse window-mask helper; mask decode too when past length exceeds 256. Initially retaining all KV is simpler but does not give bounded local-cache memory. |
| Global RoPE | Theta 1e6; 50% rotary frequencies, computed using full dimension 128, zero tail | Do NOT substitute ordinary 64-dimensional partial RoPE. Frequencies pair dimensions 0–31 with 64–95; dimensions 32–63 and 96–127 remain unchanged. Implement explicit full-head split-half rotation first. |
| Local RoPE | Theta 1e4; all 128 dimensions rotated | Existing full-head rotation is a candidate, subject to numeric parity. |
| Residual branches | Pre-norm plus `post_operator_norm` and `post_ffn_norm` before residual additions | Add both post-norms; existing LFM2 lacks them. |
| Dense MLP | `w2(silu(w1(x)) * w3(x))`, intermediate 7680 | Reuse quantized linears/SwiGLU, retain checkpoint names w1/w2/w3. No MoE routing needed. |
| Checkpoint | 594 tensors; all under `language_model.`; conv tensor `[2048,7,1]`; affine 4-bit/group 64; tied embeddings | Existing optional-prefix loader can strip wrapper. Convolution layout already matches the existing MLX convention. Quantized gate tensor `[16,256]` with scales/biases `[16,32]`. |
| Chat/tools | ChatML, optional `<think>`, Qwen-style `<tool_call><function=…><parameter=…>` | Existing parser recognizes this family; fixture validation remains necessary. It differs from LFM2 Pythonic tool-call lists. |

### Revised two-hour assessment

No new GPU primitive appears architecturally necessary for ordinary single-request inference. A two-hour coordinated prototype remains plausible, with moderate confidence, but is not a production-support commitment. The most likely blockers are numeric parity (RoPE and norm rounding), checkpoint integration and compile/debug time. A current Transformers 5.12.1-compatible reference environment must still be established; installed MLX-LM support is not assumed.

Suggested timebox, starting after this inspection:

1. 0–15 minutes: freeze module/API boundaries and checkpoint identity; obtain weights/tokenizer and prepare fixtures.
2. 15–65 minutes: model implementation, loader/AnyModel wiring, and independent test/reference preparation in parallel with distinct file ownership.
3. 65–100 minutes: compile, load official weights, verify incremental vs chunked prefill, correct discovered integration problems.
4. 100–120 minutes: run short chat/streaming checks and available reference comparison; report exact passed/failed gates and remaining gaps.

Stop claiming a usable prototype if weights do not load, logits are non-finite, or incremental/cache tests fail. Fluent output alone is insufficient. Full-reference parity, tools and long context may exceed the timebox; label the result experimental if those gates remain open.
