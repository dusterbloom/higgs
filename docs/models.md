# Supported Models

Higgs detects local model support from `config.json` `model_type`. The tables below are representative rather than exhaustive.

## Supported Architectures

| Architecture | `model_type` | Examples |
|---|---|---|
| LLaMA | `llama` | Llama 3 and CodeLlama |
| Mistral | `mistral` | Mistral 7B |
| Qwen2 | `qwen2` | Qwen2 and Qwen2.5 |
| Qwen3 | `qwen3` | Qwen3 |
| Qwen3.5 (dense) | `qwen3_5` | Qwen3.5 dense MLX checkpoints |
| Qwen3.5 / Qwen3.6 MoE | `qwen3_5_moe` | Qwen3.5-35B-A3B, Qwen3.6-35B-A3B |
| Qwen3-Next | `qwen3_next` | Qwen3-Coder hybrid checkpoints |
| Qwen3-MoE | `qwen3_moe` | Qwen3-30B-A3B |
| Nanbeige | `nanbeige` | Nanbeige4.2 |
| Gemma 2 | `gemma2` | Gemma 2 2B, 9B, and 27B |
| Gemma 3 | `gemma3`, `gemma3_text` | Gemma 3 1B, 4B, 12B, and 27B |
| Gemma 4 | `gemma4`, `gemma4_text`, `gemma4_unified` | Gemma 4 E2B, E4B (edge); 12B, 31B; 26B-A4B (MoE) |
| Phi-3 | `phi3` | Phi-3 Mini, Small, and Medium |
| Starcoder2 | `starcoder2` | Starcoder2 3B, 7B, and 15B |
| DeepSeek-V2 | `deepseek_v2` | DeepSeek-V2-Lite |
| LLaVA-Qwen2 | `llava-qwen2` | nanoLLaVA-1.5 |

### Gemma 3 / Gemma 4 notes

- These are text-language-model implementations. Multimodal checkpoints
  (`gemma3`, `gemma4`) load fine — their vision/audio tower weights are skipped
  and only the text model runs. The text weights may be nested under
  `language_model.` in such checkpoints; Higgs strips that prefix automatically.
- Gemma 4 E2B/E4B (per-layer-input embeddings + cross-layer KV sharing) and dense
  text variants are supported. The MoE variant (`gemma4` with
  `enable_moe_block`, e.g. 26B-A4B) is supported only with **unquantized** expert
  weights; a checkpoint with quantized experts is rejected at load with a clear
  error rather than producing incorrect output.

## Continuous Batching Support

`batch=true` enables true batched decode only for these `model_type` values:

- `llama`
- `mistral`
- `qwen2`
- `qwen3`

Other supported architectures still serve normally in simple mode, but Higgs now rejects `batch=true` during config load, `doctor`, and server startup.

Nanbeige uses repeated shared-weight decoder loops with loop-aware KV cache slots, so it is not included in true batched decode support.

## Representative Working MLX Model IDs

| Family | Example model IDs |
|---|---|
| LLaMA | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Qwen2.5 | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| Qwen3 | `mlx-community/Qwen3-1.7B-4bit` |
| Qwen3-Next | `mlx-community/Qwen3-Coder-Next-4bit` |
| Qwen3.5 dense | `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit` |
| Qwen3.5 MoE | `NexVeridian/Qwen3.5-35B-A3B-3bit` |
| Qwen3.6 MoE | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| Qwen3.6 MoE (eschamoe) | `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` (converted at load; see below) |
| Nanbeige | `MercuriusDream/Nanbeige4.2-3B-mlx-6bit` |
| DeepSeek-V2 | `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` |

## Qwen 3.6 Notes

- `Qwen3.6` support currently uses the `qwen3_5_moe` loader path.
- The cached-model smoke matrix covered `mlx-community/Qwen3.6-35B-A3B-4bit` plus `mlx-community/Llama-3.2-1B-Instruct-4bit`, `mlx-community/Qwen2.5-3B-Instruct-4bit`, `mlx-community/Qwen3-1.7B-4bit`, and `mlx-community/Qwen3-Coder-Next-4bit`.
- OpenAI-style chat requests use non-thinking mode by default for `Qwen3.6` unless the request explicitly opts into reasoning.

## EschaLabs `eschamoe` Checkpoints

Higgs loads EschaLabs trellis-quantized (`eschamoe`) checkpoints, for example
`EschaLabs/Qwen3.6-35B-A3B-Escha-W2` (a 2-bit trellis release of
Qwen3.6-35B-A3B). No config field is needed — detection is automatic:

- A checkpoint is treated as `eschamoe` when `quantize_config.json` declares
  `quant_method: "eschamoe"`, or, as a fallback, when `config.json` declares it
  under `quantization_config.quant_method`. `quantize_config.json` wins when
  both files are present.
- `model_type` stays `qwen3_5_moe`, so the model serves through the existing
  Qwen3.5/3.6-MoE path.

At load, Higgs decodes the trellis weights on the CPU and requantizes them in
memory to MLX affine 4-bit (group size 64) — the same layout as
`mlx-community/Qwen3.6-35B-A3B-4bit`.

**Memory: the on-disk size is misleading.** The 2-bit trellis download is small
(12.3 GB for the 35B release), but the converted model is roughly 20 GB
resident. Size your machine for the resident number, not the download.
`higgs doctor` estimates the resident size from `config.json` and warns when it
crowds system RAM.

**Slow first load.** The trellis decode is CPU-bound: about 140 s for the 35B
checkpoint. A long first start is expected — it is not a hang.

**Limitations.**

- Support is currently text-only.
- The MTP draft head in the published checkpoint is not usable: it ships the
  MoE router and shared expert but no routed expert weights, so Higgs disables
  MTP for this checkpoint and decodes without speculation.

Worked config entry:

```toml
[[models]]
name = "qwen36-escha"
path = "EschaLabs/Qwen3.6-35B-A3B-Escha-W2"
```

## Model Input Requirements

- Local models can be referenced by Hugging Face model ID or local path.
- The model must be in MLX `safetensors` format. EschaLabs `eschamoe` trellis
  checkpoints are the exception; Higgs converts them in memory at load (see
  above).
- The checkpoint must use a supported `config.json` `model_type`.
- macOS local serving requires `mlx.metallib` next to the executable. Release artifacts bundle it, and source builds restore it from Cargo build output when possible.

For configuration details, see [configuration.md](configuration.md).
