# First-Class Vision Support — Design

**Date:** 2026-08-19
**Branch:** `feat/vision-models`
**Status:** Approved in brainstorm; awaiting implementation plan

## Summary

Make vision a first-class capability in Higgs: every model family that ships
vision weights runs its **native** image preprocessing, images are accepted on
both the OpenAI and Anthropic APIs (streaming and non-streaming) at their true
position in the conversation (multi-turn, interleaved with text, tool results),
resolution control (`detail`, `max_width`/`max_height`) is honored, and
requests that don't match model capabilities fail with strict, explicit 400s.

Target families, in implementation order: **LLaVA-Qwen2** (existing, generalized
to multi-image), **Gemma 3 / Gemma 4** vision towers (weights already present in
checkpoints but currently skipped), **Qwen-VL** (Qwen3.5-VL / Qwen3-VL /
Qwen2.5-VL) including checkpoints whose text backbone is **Qwen3.5** (dense and
MoE) or **escha-w2** (EschaLabs `eschamoe` trellis-quantized, built on the
Qwen3Next path).

## Current State (baseline)

- One working VLM: `llava-qwen2` (nanoLLaVA) with a SigLIP vision encoder.
- OpenAI-style `image_url` parts accepted, **base64 data URIs only**; HTTP URLs
  explicitly not fetched.
- **Single image only**: all images are extracted, only the first is used;
  `merge_embeddings` hard-errors on more than one image position.
- Preprocessing is a naive square resize to `image_size` (default 384) via
  Lanczos3 — correct only for LLaVA/SigLIP.
- Anthropic `Image` blocks are **silently dropped** in
  `anthropic_messages_to_engine`.
- Gemma 3/4 multimodal checkpoints load, but `vision_tower.*` / `audio_tower.*`
  weights are skipped — text-only.
- Prefix caching is disabled for multimodal; the batch engine rejects
  multimodal inputs outright.
- No doctor validation, config surface, or docs for vision.

## Architecture

### `VisionModel` trait in `higgs-models::vision` (new module)

```rust
/// Shared input item, position-preserving: produced by the route layer from
/// either OpenAI parts or Anthropic blocks, in client order.
pub struct ImageInput {
    pub position: usize,       // index among all content parts (interleaving)
    pub message_index: usize,  // for error messages: "image in message 2"
    pub bytes: Vec<u8>,
    pub media_type: String,    // "image/png", "image/jpeg", ...
    pub detail: ImageDetail,   // Low | High | Auto
    pub max_dims: Option<(u32, u32)>, // Anthropic max_width/max_height
}

/// Opaque to the engine; only the model impl knows the internal arrangement.
pub struct ImageBatch {
    pub pixel_values: Array,          // family-native layout, N images
    pub per_image_tokens: Vec<usize>, // embedding rows each image expands to
    pub layout: ImageTokenLayout,     // produced per batch (k is data-dependent)
}

pub struct ImageTokenLayout {
    pub start: Option<u32>, // e.g. Gemma <start_of_image>, Qwen <|vision_start|>
    pub end: Option<u32>,   // e.g. Gemma <end_of_image>, Qwen <|vision_end|>
    pub pad: Option<u32>,   // e.g. Qwen <|image_pad|> — replaced by features
}

pub struct VisionCapabilities {
    pub families: Vec<&'static str>,      // "llava-qwen2", "gemma3", "qwen3_5_vl", ...
    pub image_sizes: Vec<i32>,            // native preprocessing sizes
    pub supported_media: Vec<&'static str>, // "image/png", "image/jpeg", ...
    pub layout_kind: ImageTokenLayoutKind,  // Sentinel | StartEndPad
}

pub trait VisionModel {
    fn vision_capabilities(&self) -> VisionCapabilities;
    fn preprocess_images(&self, images: &[ImageInput]) -> Result<ImageBatch, VisionError>;
    fn forward_multimodal(&mut self, input_ids: &Array, batch: &ImageBatch,
                          cache: &mut AnyCache) -> Result<Array, Exception>;
}
```

`AnyModel` gains `as_vision(&mut self) -> Option<&mut dyn VisionModel>`; the
engine and route layer talk to vision only through the trait. The existing
`is_vlm()` / `image_size()` `matches!` helpers are replaced by
`vision_capabilities()`.

### Embedding-merge is generalized, and stays before the transformer

LLaVA already merges projected image features into the text embedding sequence
**before** the transformer runs. `merge_embeddings` is generalized from "exactly
1 sentinel position, 1 image" to "N image regions, each expanding to
`per_image_tokens[i]` rows, with optional start/end/pad tokens". Because merging
happens before the transformer, the **batch engine can share the same path** —
each sequence in the batch gets its own merged embedding array.

### Backbone requirement: `forward_from_embeddings` on `Qwen3NextCausalLM`

**Critical gap.** LLaVA relies on `Transformer::forward_from_embeddings`
(embedding-merge → transformer). `Qwen3NextCausalLM` — which carries **Qwen3.5
dense (`qwen3_5`), Qwen3.5 MoE (`qwen3_5_moe`), Qwen3-Next, and on the escha
branch `eschamoe`** — has no such method; its `forward_hidden` / `forward` /
`forward_last_token` / `forward_chunked` all call
`self.model.embed_tokens.forward(inputs)` internally.

Add `Qwen3NextCausalLM::forward_from_embeddings(&mut self, embeddings: &Array,
mask: Option<&Array>, kv_cache: &mut Vec<Option<LayerCache>>) -> Result<Array,
Exception>` mirroring `Transformer::forward_from_embeddings`: take the merged
`[1, L, D]` array, run the layer stack from there (skip `embed_tokens` lookup),
same mask/cache semantics as `forward_hidden`. This serves **all**
Qwen3Next-backed families at once — including eschamoe, whose `EschaSwitchMlp`
expert layers live inside `SwitchMlpWeights` and are unaffected by the embedding
entry point.

### Qwen-VL is its own adapter

New `LoadKind::QwenVl` adapter:

- Detects wrapper `model_type` values (`qwen3_5_vl`, `qwen3_vl`, `qwen2_5_vl`,
  ...) and their nested `text_config`; the existing `DetectedModel` plumbing
  already surfaces `wrapper_model_type` and nested `model_type`.
- Resolves the text backbone through the existing qwen3.5 / qwen3_next arg
  loaders (which already handle `language_model.` prefix stripping).
- Builds `QwenVlModel { vision_tower, mm_projector, language_model:
  Qwen3NextCausalLM }` implementing `VisionModel`.
- `qwen_revision()` currently returns `None` for a `_vl` suffix — the tolerant
  matcher must learn the suffix so `qwen3_5_vl` maps to the QwenVl adapter.
- Vision tower weights load from the wrapper checkpoint's `vision_tower.*`
  prefix; the text backbone from `text_config`.

### escha-w2 composition is orthogonal

Escha quantization is confined to expert projections inside the backbone, so
the vision wrapper is orthogonal to it: a Qwen3.5-VL checkpoint that is
Escha-quantized works through the same wrapper as long as the backbone exposes
`forward_from_embeddings` and `embed_token`. The wrapper level must not
hardcode dense-expert assumptions that break under `force_eschamoe_quant_layout`.
The escha work (`codex/escha-w2-mlx-roadmap`) merges into the same crate; the
trait must be escha-neutral.

## Media Pipeline & API Surface

New `higgs/src/media.rs` module — one shared extraction pipeline for both APIs
and both route paths (streaming + non-streaming), replacing
`extract_images` / `inject_image_placeholders`:

```
API request (OpenAI parts | Anthropic blocks)
  → MediaExtractor::from_request()  → Vec<MediaItem>   (client order, positions kept)
  → fetcher resolves data URIs locally, HTTP URLs via reqwest (cap + timeout)
  → capability gate: engine has vision?  else 400 (strict)
  → per-item validation (media_type, size cap)  → 400 with message index
  → prompt rendering with family markers at the item's position
  → VisionModel::preprocess_images()  → ImageBatch
  → forward_multimodal() with merged embeddings
```

### Extraction rules

- **OpenAI parts**: `text` and `image_url` parts collected in order;
  interleaving preserved ("look at this → [img] → now describe it"). HTTP(S)
  `image_url` values fetched with a configurable client (byte cap + timeout);
  `data:` URIs decoded locally.
- **Anthropic blocks**: `text` and `image` blocks collected in order;
  `source.type = base64` decoded, `source.type = url` fetched. Images in
  `tool_result` blocks collected too. System images rejected (Anthropic spec)
  with 400.
- A message with N images produces N `MediaItem`s at their true positions.

### Strict errors (as chosen)

| Condition | Response |
|---|---|
| Images sent to text-only model | 400, names the model, says it has no vision support |
| Unknown/unsupported `media_type` (audio, video) | 400 with message index |
| Image bytes exceed cap | 400 with the cap value |
| HTTP fetch failure / timeout / non-image content-type | 400 |
| Image in system prompt (Anthropic) | 400 |
| Base64 decode failure | 400 with message index |

### Resolution semantics (`detail`, `max_dims`)

- `detail: low` → downscale so the long edge ≤ model-specific low threshold
  (e.g. 384 LLaVA, 896 Gemma, 512 Qwen-VL low).
- `detail: high` → keep native resolution up to the family cap; the family
  processor still does its native transform (LLaVA squares, Gemma
  pans-and-scans, Qwen-VL dynamic resolution with `max_pixels`).
- `detail: auto` → family default.
- Anthropic `max_width` / `max_height` → downscale to fit the box before family
  processing; `max_dims` wins over `detail`.
- Per-request hard cap (configurable, default 4096px long edge) bounds memory.

### Anthropic parity

`anthropic_messages_to_engine` no longer drops images; both routes call the
same chat-completions core with a unified `Vec<MediaItem>`, so streaming and
tool-calling behave identically on both APIs.

## Family Preprocessing & Token Layout

Each family implements `preprocess_images` and produces the per-batch
`ImageTokenLayout`. Preprocessing happens once per request in the route layer;
the `ImageBatch` flows to whichever engine path runs the request.

### LLaVA-Qwen2 / SigLIP (generalize existing)

- Preprocess: keep `siglip::preprocess_image` (square resize, Lanczos3) but
  batch it — N images → `[N, H, W, 3]` pixel values.
- Layout: 1 sentinel per image; `per_image_tokens = [1, 1, …]`;
  `start/end/pad = None`.
- Merge: walk the token sequence; at each sentinel splice `image_features[i]`
  (`[num_patches, hidden]`); multiple images allowed.

### Gemma 3 / Gemma 4 vision (new `gemma_vision.rs`)

- Preprocess: family-native **pan-and-scan**. Resize to the target square
  (896 for Gemma 3; from checkpoint `vision_config`) preserving aspect ratio,
  then take up to 6 crops (center + corners, or the configured crop set). Each
  crop `image_size × image_size`, normalized with family mean/std. **Positional
  offsets** per crop are part of the preprocessing output (an offsets array
  telling the transformer where each crop belongs), not a separate forward.
- Layout: `<start_of_image>` + k × feature rows + `<end_of_image>`; k is the
  per-image patch count from the tower output (actual k read from
  `vision_config`, not hardcoded). `start`/`end` are real tokenizer tokens.
- Merge: splice k feature rows between start and end tokens; `<start_of_image>`
  and `<end_of_image>` stay in the sequence as regular token embeddings (the
  standard Gemma 3 arrangement). Confirm against the reference implementation
  during the Gemma step and adjust if it differs.
- Weight loading: Gemma 3/4 loaders consume `vision_tower.*` weights when
  present (under the existing `language_model.` prefix handling) and remain
  text-only when absent (`gemma3_text`, `gemma4_text` unchanged).

### Qwen-VL (new `qwen_vl.rs`)

- Preprocess: **dynamic resolution**. Compute an aspect-ratio-preserving
  resolution from `min_pixels`/`max_pixels` (e.g. `max_pixels = 1280×28×28`,
  `min_pixels = 256×28×28`), resize, split into `grid_h × grid_w` tiles of
  `patch_size × patch_size` (e.g. 14). **Pixel-shuffle** merges tiles into the
  text embedding dimension (2×2 merge with `merge_size=2`), so a single image
  becomes `(grid_h × grid_w) / 4` embedding rows. `per_image_tokens[i]` comes
  out of this.
- Layout: `<|vision_start|>` + `per_image_tokens[i]` × `<|image_pad|>` +
  `<|vision_end|>`; k is resolution-dependent, so the layout is produced per
  batch after preprocessing.
- Merge: splice pixel-shuffled features at pad positions; start/end stay as
  embeddings.
- Backbone: `QwenVlModel { vision_tower, mm_projector, language_model:
  Qwen3NextCausalLM }` using the new `forward_from_embeddings`.

### Shared preprocessing utilities (`vision.rs`)

- Image decode + resize + normalize primitives (promoted from `siglip.rs`).
- `detail` / `max_dims` → target-size resolution applied **before** the family
  processor.
- `ImageBatch` is opaque to the engine.

### Chat-template rendering with markers

The engine renders the chat template with **family markers** at media positions
instead of the old `<image>\n` prefix:

- LLaVA: marker text `<image>` at the part's true position.
- Gemma 3/4: `<start_of_image><end_of_image>` (per family template).
- Qwen-VL: `<|vision_start|><|image_pad|><|vision_end|>`; post-tokenization
  expansion replaces the pad with k pads.

Then the trait's post-processor rewrites token ids into the exact sentinel run.
`replace_image_tokens` / `inject_image_placeholders` are removed.

## Engine Integration

### Simple engine (primary path, staged first)

- `pixel_values: Option<Array>` parameters become `image_batch: Option<ImageBatch>`
  in `generate_with_thinking` / `generate_streaming_with_thinking`.
- Prefill: multimodal path calls `model.forward_multimodal(&prompt_array,
  &batch, &mut cache)` — N-image aware via `per_image_tokens` + `layout`.
- Prefix caching stays disabled for multimodal (image-specific KV), documented
  in trait docs and config reference. Disk prefix store likewise skipped
  (already true).
- Multimodal requests never chunk prefill (image features are inserted
  mid-sequence; a chunk boundary cannot split them). Documented.

### Batch engine (staged later — real work)

Two gaps must close for batch vision:

1. **Multimodal prefill** in `prefill_request` / `advance_prefill`: today prefill
   slices token chunks and calls `model.forward` on token ids. Multimodal
   prefill runs the **merged-embedding forward** (`forward_multimodal`) once per
   request — image features can't span chunk boundaries. `BatchRequest` gains
   `image_batch: Option<ImageBatch>`; `start_prefill` skips the prefix-cache
   lookup when present.
2. **Batched decode for VLM families**: `forward_batched` exists only for
   `Transformer`. Batched decode only ever sees single text tokens against
   per-request KV caches (image fused into KV during prefill), so this is
   *delegating to the backbone's batched decode*: `LlavaQwen2` → its Qwen2
   `Transformer::forward_batched`; `QwenVl` → new
   `Qwen3NextCausalLM::forward_batched`; Gemma 3/4 equivalents when batch is
   wanted. The batch-mode config gate (`batch=true` allowed only for
   transformer families) extends family-by-family as each wrapper lands batched
   decode.

**Sequencing**: batch vision lands after simple-mode vision per family.
Multimodal never uses prefix or disk cache at any stage.

### Metrics & token accounting

- Prompt token count includes image-token expansion: `len(prompt_tokens)`
  already reflects the sentinel run after post-processing, so
  `prompt_token_count` (usage + metrics) is correct **if** expansion happens
  before the count is read. Capture after expansion; count exactly once; both
  APIs identical.
- `image_count`, `total_image_pixels`, `preprocess_ms` added to metrics
  (opt-in debug level; no schema break).
- Tokenizer-visible markers are expanded *after* `prepare_chat_prompt`, so the
  tokenizer and template never see sentinel ids (same invariant as today).

### MTP / speculative decode interaction

`Qwen3Next` models can carry an MTP head, which assumes token-id inputs.
Multimodal prefill must **not** run the MTP head over a prompt containing image
features (draft logits for image positions are meaningless). Rule: a request
with images disables MTP/speculative decode for that request and runs the plain
decode loop — per-request, not per-model.

### Embeddings endpoint

`/v1/embeddings` stays text-only; images in an embeddings request → 400.
(Image embeddings were explicitly out of scope.)

## Config, Doctor, Docs

### New config fields (all optional, defaults preserve current behavior)

**`ServerSection`**:

```toml
[server]
max_image_bytes = 20971520      # per-image decoded byte cap (default 20 MiB)
image_fetch_timeout = 10.0      # remote image URL timeout in seconds
max_image_dimension = 4096      # long-edge pixel cap before family processing
```

Follow the `max_body_size` pattern (serde defaults, doctor validation, daemon
template).

**`ModelConfig`**:

```toml
[[models]]
path = "mlx-community/Qwen3.5-VL-..."
disable_vision = false  # escape hatch: force-disable vision if tower fails to load
```

Without `disable_vision`, a checkpoint whose vision weights fail to load is a
hard startup error (strict semantics).

### Doctor validation (`cargo test -p higgs -- --test-threads=1`)

1. **Server section**: `max_image_bytes < max_body_size` (else warn — body cap
   smaller than image cap means images can never arrive); `max_image_dimension`
   in 64..=16384; `image_fetch_timeout > 0`.
2. **Per-model vision checks**:
   - `disable_vision = true` on a model with no vision tower → warn (no-op).
   - Checkpoint shows vision capability (wrapper `*_vl` `model_type`,
     `vision_config` present, or vision-tower weight keys) but the resolved
     adapter can't run vision → warn "checkpoint contains vision weights that
     Higgs will ignore" (mirrors today's gemma3/4 skip) — until the family is
     implemented; then error only if `disable_vision` unset and loading fails.
   - `batch = true` on a VLM family without batched decode → error (extends the
     existing batch-support gate).
3. **Capability report**: `higgs doctor` gains a vision column: `vision: none |
   supported (LLaVA) | tower-ignored (Gemma4)`.

### `higgs init` template (daemon.rs)

Add the three server fields + `disable_vision` to the generated `config.toml`
template with `#` comments, per template conventions.

### Docs

- **README.md**: "Vision support" section — families, `detail`/`max_dims`
  semantics, HTTP URL support, 400 behavior, config reference additions.
- **docs/models.md**: mark VLM rows with vision status; add "Vision-capable
  models" table (square resize / pan-and-scan / dynamic resolution, per-family
  defaults).
- **docs/configuration.md**: document the new fields and the
  "multimodal requests never use prefix/disk cache" behavior.
- **Doc comments** on new config fields and `VisionModel` trait methods.

## Testing

### Unit

- Preprocessing per family: pixel-value shapes, pan-and-scan crop sets +
  offsets, dynamic-resolution grid computation + pixel-shuffle, `detail` /
  `max_dims` → target size.
- Token layout: marker → sentinel-run expansion for LLaVA (1), Gemma
  (start/k/end), Qwen-VL (start/k pads/end with data-dependent k).
- `merge_embeddings` for N images and mixed text/image interleaving.
- `Qwen3NextCausalLM::forward_from_embeddings` parity with
  `forward_hidden` on text-only inputs (logits equal).
- `qwen_revision` / tolerant matcher learns `_vl` suffix; adapter resolution
  for `qwen3_5_vl`, `qwen3_vl`, `qwen2_5_vl`.
- Anthropic block parsing (base64 + URL, tool_result, system rejection);
  OpenAI part parsing; extractor ordering.
- HTTP fetch (mocked): success, timeout, oversize, non-image content-type.
- Doctor rules: new server fields, disable_vision no-op warn, batch-gate error,
  capability report.

### Integration

- Image request through OpenAI and Anthropic endpoints, streaming and
  non-streaming, single and multi-image, interleaved positions, tools+images,
  multi-turn with images.
- Text-only model + image → 400; oversize → 400; bad base64 → 400.
- Batch mode with images once per-family batched decode lands.
- MTP disabled for image requests.
- `cargo clippy -p higgs` clean (nursery), `cargo fmt --check` pass,
  `cargo test -p higgs -- --test-threads=1` pass.
- Extend `scripts/release_smoke_cached_models.sh` with a cached vision model
  (nanoLLaVA) covering single-image streaming/non-streaming; optional
  environment-gated addition for a Qwen-VL checkpoint if cached.

## Out of Scope (explicitly deferred)

- Audio input (Gemma 4 audio towers remain skipped; unified multimodal pipeline
  rejected as YAGNI).
- Image embeddings in `/v1/embeddings`.
- Image-aware prefix caching (keying by image hash) — perf work, not required.
- CUDA-specific Escha host-placement / selected-slab upload (per escha branch
  non-goals).

## Sequencing (implementation order)

1. `VisionModel` trait + `vision.rs` + `as_vision()` + capabilities metadata.
2. Shared media extraction (`media.rs`), HTTP fetch, strict errors, both APIs.
3. LLaVA multi-image: batched preprocessing, generalized merge, marker
   rendering, detail/max_dims.
4. `Qwen3NextCausalLM::forward_from_embeddings` (unblocks Qwen3.5 / escha).
5. Qwen-VL adapter (detection, tower, dynamic resolution, layout).
6. Gemma 3/4 vision towers (pan-and-scan, positional offsets, weight loading).
7. Batch engine vision (multimodal prefill + batched decode per family).
8. Config + doctor + docs + smoke harness.
9. MTP-disable for image requests (can land earlier; kept here for completeness).

Each step is independently reviewable and commit-ready.
