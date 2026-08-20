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
| Architecture | `model_type` | Examples | Vision |
|---|---|---|---|
| LLaMA | `llama` | Llama 3 and CodeLlama | none |
| Mistral | `mistral` | Mistral 7B | none |
| Qwen2 | `qwen2` | Qwen2 and Qwen2.5 | none |
| Qwen3 | `qwen3` | Qwen3 | none |
| Qwen3.5+ (dense) | `qwen3_5`, `qwen3_5_text` | Qwen3.5 dense checkpoints; Qwen3.8-27B | none |
| Qwen3.5+ (MoE) | `qwen3_5_moe`, `qwen3_5_text_moe` | Qwen3.5-35B-A3B, Qwen3.6-35B-A3B | none |
| Qwen3-Next | `qwen3_next` | Qwen3-Coder hybrid checkpoints | none |
| Qwen3-MoE | `qwen3_moe` | Qwen3-30B-A3B | none |
| Qwen-VL | `qwen3_5_vl`, `qwen3_vl`, `qwen2_5_vl` | Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL | supported |
| Gemma 2 | `gemma2` | Gemma 2 2B, 9B, and 27B | none |
| Gemma 3 | `gemma3`, `gemma3_text` | Gemma 3 1B, 4B, 12B, and 27B | tower-ignored† |
| Gemma 4 | `gemma4`, `gemma4_text`, `gemma4_unified` | Gemma 4 E2B, E4B (edge); 12B, 31B; 26B-A4B (MoE) | tower-ignored† |
| Phi-3 | `phi3` | Phi-3 Mini, Small, and Medium | none |
| Starcoder2 | `starcoder2` | Starcoder2 3B, 7B, and 15B | none |
| DeepSeek-V2 | `deepseek_v2` | DeepSeek-V2-Lite | none |
| LLaVA-Qwen2 | `llava-qwen2` | nanoLLaVA-1.5 | supported |

† `higgs doctor` reports vision at the adapter level. A multimodal `gemma3` /
`gemma4` checkpoint (with `vision_config` and `vision_tower.` weights) shows
`vision: tower-ignored` even though the runtime loads the tower and runs
pan-and-scan image input; `gemma3_text` / `gemma4_text` checkpoints show
`vision: none`.

### Gemma 3 / Gemma 4 notes

- Multimodal `gemma3` / `gemma4` checkpoints carry a SigLIP-style vision tower
  under `vision_tower.`; Higgs loads it and processes images with the
  pan-and-scan scheme (see [Vision-capable models](#vision-capable-models)).
  Text-only `gemma3_text` / `gemma4_text` checkpoints have no tower and remain
  text-only. The audio tower weights on multimodal checkpoints are skipped. The
  text weights of multimodal checkpoints may be nested under `language_model.`;
  Higgs strips that prefix automatically.
- Current constraint: the multimodal forward requires the vision tower's hidden
  size to equal the language model's hidden size — the learned multi-modal
  projector is not yet implemented, so checkpoints where the two differ error at
  prefill rather than producing wrong output.
- Gemma 4 E2B/E4B (per-layer-input embeddings + cross-layer KV sharing) and dense
  text variants are supported. The MoE variant (`gemma4` with
  `enable_moe_block`, e.g. 26B-A4B) is supported only with **unquantized** expert
  weights; a checkpoint with quantized experts is rejected at load with a clear
  error rather than producing incorrect output.

## Vision-Capable Models

Image input is supported on the OpenAI chat endpoint (`/v1/chat/completions`,
streaming and non-streaming) for three families. Each family implements a
different preprocessing scheme; the per-family defaults come from the
checkpoint's `config.json` (`vision_config`, `mm_tokens_per_image`,
`min_pixels` / `max_pixels`).

| Family | `model_type` | Preprocessing | Per-family defaults |
|---|---|---|---|
| LLaVA-Qwen2 | `llava-qwen2` | Square resize | Every image is resized to `vision_config.image_size` (384 for nanoLLaVA); `detail: "low"` on every image halves the target (floored at 128 px), `auto`/`high` use the full size |
| Qwen-VL | `qwen2_5_vl`, `qwen3_vl`, `qwen3_5_vl` | Dynamic resolution | `smart_resize` into the `[min_pixels, max_pixels]` budget (`256·28²` / `1280·28²` defaults), rounding to multiples of `patch × merge` (28); 2×2 spatial merge of the patch grid |
| Gemma 3 / Gemma 4 | `gemma3`, `gemma4` (multimodal) | Pan-and-scan | Shorter side scaled to the tower's `image_size`; aspect-aware 6-crop grid (2 rows × 3 cols, transposed to 3 × 2 for portrait); `mm_tokens_per_image` embeddings per crop (256 for Gemma 3 27B, 1024 for 4B; fallback `num_patches / 4`) |

Notes:

- **Markers**: LLaVA uses `<image>`; Qwen-VL uses
  `<|vision_start|><|image_pad|><|vision_end|>`; Gemma uses
  `<start_of_image><end_of_image>`. Markers are spliced at each image's true
  position, so multiple images per request are supported in all three families.
- **Backbones**: Qwen-VL runs on the Qwen3Next text backbone, loaded through the
  Qwen3.5 dense/MoE loaders. escha-w2 (eschamoe) backbones are expected to work
  under the same wrapper per the design spec — escha quantization is confined to
  expert projections and is orthogonal to the vision wrapper — but there is no
  in-tree escha checkpoint to verify against.
- **Doctor status**: `higgs doctor` reports a `vision:` status in each model's
  capability line: `vision: supported (<model_type>)`, `vision: tower-ignored
  (<model_type>)`, or `vision: none`. `supported` means the resolved adapter
  implements vision; `tower-ignored` means the checkpoint declares vision
  weights (`vision_config` or a `*_vl` model type) that the resolved text
  adapter skips; `none` means neither.
- **Current constraints**:
  - Qwen-VL's SigLIP-shaped tower supports its nominal grid only (fixed learned
    position table); an image that `smart_resize`s to a different patch grid
    fails prefill with a clear error (the Qwen-VL RoPE tower is not yet
    implemented).
  - Gemma 3/4 multimodal forward requires the tower's hidden size to equal the
    language model's hidden size (see the Gemma notes above).
- **Caching and MTP**: image requests never reuse or populate the in-memory or
  disk prefix cache (their KV state reflects merged image features and would not
  match a text-only prefix), and MTP speculative decode is disabled for image
  requests.

## Continuous Batching Support

`batch=true` enables true batched decode only for these `model_type` values:

- `llama`
- `mistral`
- `qwen2`
- `qwen3`
- `llava-qwen2`
- `qwen3_5_vl` (and the other Qwen-VL types `qwen3_vl` / `qwen2_5_vl`)

Other supported architectures still serve normally in simple mode, but Higgs now rejects `batch=true` during config load, `doctor`, and server startup. In batch mode, vision-capable families preprocess images inside the worker thread and route multimodal requests through the batched engine.

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
| Qwen3.8 dense | `mlx-community/Qwen3.8-27B-4bit` |
| DeepSeek-V2 | `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` |

## Qwen 3.5+ Adapter and Version Notes

- Qwen 3.5, 3.6, and 3.8 dense and MoE checkpoints use the Qwen 3.5 adapters. This includes `*ForConditionalGeneration` wrapper configs: Higgs detects the top-level wrapper and consumes the nested `text_config` used by the text loader.
- `_text` aliases such as `qwen3_5_text` and `qwen3_5_text_moe` resolve to the corresponding dense or MoE adapter.
- Unknown newer versions within a supported family can use the nearest adapter only after the resolved config passes structural validation. Higgs logs an untested-version warning when it takes this tolerant path. A missing or invalid required field is rejected by name; an unknown family is rejected with the supported family/version list.
- `mlx-community/Qwen3.8-27B-4bit` (27B dense, 4-bit) is verified working through its top-level `qwen3_5` wrapper and nested `qwen3_5_text` config.
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

### Native and affine paths

Higgs has two ways to load these checkpoints. The native path is the default.

**Native (default).** The expert projections stay in their trellis form and a
Metal kernel decodes them during the forward pass. Only the non-expert weights
convert, to MLX affine 4-bit (group size 64). The 35B release holds about
11 GB and loads in a few seconds.

**Affine (`HIGGS_ESCHA_NATIVE=0`).** Every expert decodes on the CPU and
requantizes in memory to MLX affine 4-bit — the same layout as
`mlx-community/Qwen3.6-35B-A3B-4bit`. The same 35B release then holds about
22 GB and takes roughly 140 s to start, so it needs a machine with memory to
spare. The path stays available for comparing the kernel against a plain
affine baseline.

`higgs doctor` estimates the resident size for whichever mode is active and
warns when it crowds system RAM. On the native path it reads the trellis rate
of each projection from `quantization_config.layer_meta`; note that the rate
varies per projection, so a checkpoint named `W2` is not uniformly 2-bit — the
35B release uses 2 bits for `gate_up_proj` and 3 for `down_proj`.

**Memory: the on-disk size is misleading on the affine path.** The 2-bit
trellis download is 12.3 GB for the 35B release; converted, it is roughly
22 GB. Size the machine for the resident number, not the download.

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
