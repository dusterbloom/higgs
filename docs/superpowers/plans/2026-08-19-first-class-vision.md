# First-Class Vision Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make vision a first-class capability in Higgs — every model family that ships vision weights runs its native preprocessing, images work on both the OpenAI and Anthropic APIs (streaming and non-streaming) at their true conversation position, resolution control is honored, and capability mismatches fail with strict 400s.

**Architecture:** A `VisionModel` trait in `higgs-models::vision` is implemented per family (LLaVA-Qwen2, Gemma 3/4, Qwen-VL). The route layer extracts position-preserving `ImageInput`s from either API, the engine renders family marker tokens at those positions and post-processes them into sentinel runs, and image features are merged into the text embedding sequence **before** the transformer runs — the same trick LLaVA already uses, generalized to N images and shared with the batch engine. `Qwen3NextCausalLM` (the backbone for Qwen3.5 dense/MoE, Qwen3-Next, and eschamoe) gains `forward_from_embeddings` to support the merge.

**Tech Stack:** Rust workspace (`higgs`, `higgs-engine`, `higgs-models`), `mlx-rs`, `reqwest` (HTTP image fetch), `image` crate (decode/resize), `axum` routes, `cargo test/clippy/fmt`.

## Global Constraints

- `cargo clippy -p higgs` must stay clean (nursery lints enabled).
- `cargo fmt -p higgs -- --check` must pass.
- `cargo test -p higgs -- --test-threads=1` must pass (thread limit required by shared port bindings).
- When adding or changing config fields, update `crates/higgs/src/doctor.rs` validation and run the doctor tests.
- When changing user-facing behavior (config fields, CLI flags, API surface), update `README.md`, `crates/higgs/src/daemon.rs` (`higgs init` template), and doc comments on public structs/fields.
- Multimodal requests must never use the prefix cache or disk prefix store (image-specific KV states).
- A request with images must never run the MTP head (draft logits for image positions are meaningless).
- `/v1/embeddings` stays text-only; images there → 400.
- No new public config is added without a serde default preserving current behavior.
- Every task must be independently reviewable and committed separately.

---

## File Map

**New files:**
- `crates/higgs-models/src/vision.rs` — `VisionModel` trait, `ImageInput`, `ImageBatch`, `ImageTokenLayout`, `VisionCapabilities`, `VisionError`, generalized `merge_embeddings`, shared preprocessing primitives.
- `crates/higgs/src/media.rs` — `MediaExtractor`, `MediaItem`, HTTP fetch, strict validation.
- `crates/higgs-models/src/qwen_vl.rs` — Qwen-VL model: vision tower, projector, dynamic-resolution preprocessing, token layout.
- `crates/higgs-models/src/gemma_vision.rs` — Gemma 3/4 pan-and-scan preprocessing + vision tower loading.

**Modified files:**
- `crates/higgs-models/src/lib.rs` — `as_vision()`, `vision_capabilities()`, `forward_multimodal(&ImageBatch)`, remove `is_vlm`/`image_size` helpers.
- `crates/higgs-models/src/llava_qwen2.rs` — multi-image preprocessing, marker text, postprocess, generalized merge call.
- `crates/higgs-models/src/siglip.rs` — batched `preprocess_images`, shared primitives.
- `crates/higgs-models/src/qwen3_next.rs` — `forward_from_embeddings` (+ `forward_batched` in Phase 7).
- `crates/higgs-models/src/gemma3.rs`, `gemma4.rs` — load vision towers when present.
- `crates/higgs-models/src/adapter.rs` — `LoadKind::QwenVl`, `_vl` suffix in `qwen_revision`.
- `crates/higgs-engine/src/simple.rs` — `image_batch: Option<ImageBatch>` params, postprocess, marker text.
- `crates/higgs-engine/src/batch_engine.rs` — multimodal prefill + batched decode (Phase 7).
- `crates/higgs-engine/src/chat_template.rs` — marker-aware `convert_messages` (moved to media layer; renderer unchanged).
- `crates/higgs/src/routes/chat.rs` — use `media.rs`, pass `ImageBatch`.
- `crates/higgs/src/anthropic_adapter.rs` — stop dropping images.
- `crates/higgs/src/state.rs` — engine delegation updates.
- `crates/higgs/src/config.rs`, `doctor.rs`, `daemon.rs` — config fields + validation + init template.
- `README.md`, `docs/models.md`, `docs/configuration.md` — vision docs.

---

## Phase 1: Vision trait foundation

### Task 1: `vision.rs` — trait, shared types, capabilities

**Files:**
- Create: `crates/higgs-models/src/vision.rs`
- Modify: `crates/higgs-models/src/lib.rs` (add `pub mod vision;`)

**Interfaces:**
- Produces:
  - `pub enum ImageDetail { Low, High, Auto }` (with `Default = Auto`)
  - `pub struct ImageInput { pub position: usize, pub message_index: usize, pub bytes: Vec<u8>, pub media_type: String, pub detail: ImageDetail, pub max_dims: Option<(u32, u32)> }`
  - `pub struct ImageTokenLayout { pub start: Option<u32>, pub end: Option<u32>, pub pad: Option<u32> }`
  - `pub enum ImageTokenLayoutKind { Sentinel, StartEndPad }`
  - `pub struct VisionCapabilities { pub families: Vec<&'static str>, pub image_sizes: Vec<i32>, pub supported_media: Vec<&'static str>, pub layout_kind: ImageTokenLayoutKind }`
  - `pub struct ImageBatch { pub pixel_values: Array, pub per_image_tokens: Vec<usize>, pub layout: ImageTokenLayout }`
  - `pub enum VisionError { UnsupportedMediaType(String), ImageTooLarge(usize, usize), Decode(String), Preprocess(String) }` (impl `Display` + `std::error::Error`)
  - `pub const IMAGE_TOKEN_INDEX: i32 = -200;` (moved here from `llava_qwen2`; re-export from there for compat during migration)
  - `pub trait VisionModel { fn vision_capabilities(&self) -> VisionCapabilities; fn image_marker_text(&self) -> &'static str; fn preprocess_images(&self, images: &[ImageInput]) -> Result<ImageBatch, VisionError>; fn postprocess_image_tokens(&self, tokens: &mut Vec<u32>, tokenizer: &Tokenizer, batch: &ImageBatch) -> Result<(), VisionError>; fn forward_multimodal(&mut self, input_ids: &Array, batch: &ImageBatch, cache: &mut AnyCache) -> Result<Array, Exception>; }`

- [ ] **Step 1: Write the failing test**

Create `crates/higgs-models/src/vision.rs` and a test module asserting the types exist and serialize/construct correctly:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_detail_default_is_auto() {
        assert!(matches!(ImageDetail::default(), ImageDetail::Auto));
    }

    #[test]
    fn vision_error_display() {
        let e = VisionError::UnsupportedMediaType("audio/mp3".to_owned());
        assert!(e.to_string().contains("audio/mp3"));
    }

    #[test]
    fn image_token_index_is_negative_sentinel() {
        assert_eq!(IMAGE_TOKEN_INDEX, -200);
    }

    #[test]
    fn image_batch_layout_fields() {
        let batch = ImageBatch {
            pixel_values: Array::from_slice(&[0.0f32; 3], &[1, 1, 1, 3]),
            per_image_tokens: vec![1],
            layout: ImageTokenLayout { start: None, end: None, pad: None },
        };
        assert_eq!(batch.per_image_tokens, vec![1]);
        assert!(batch.layout.start.is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-models vision::tests --lib`
Expected: FAIL — `vision` module not found.

- [ ] **Step 3: Write the implementation**

```rust
//! Vision-language model support: shared trait, types, and preprocessing.
//!
//! Every VLM family implements [`VisionModel`]. The route layer produces
//! position-preserving [`ImageInput`]s from OpenAI parts or Anthropic blocks;
//! the engine renders family marker tokens, post-processes them into sentinel
//! runs, and merges image features into the text embedding sequence before the
//! transformer runs (see [`merge_embeddings`]).

use mlx_rs::{Array, Exception};

/// Sentinel token id marking image-feature positions in the token stream.
/// Negative so it can never collide with a real token id. Replaced by image
/// features during embedding merge.
pub const IMAGE_TOKEN_INDEX: i32 = -200;

/// Resolution control from the request (`detail` / `max_width` semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
}

/// One image from a request, in client order, with position metadata.
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Index among all content parts in the message (preserves interleaving).
    pub position: usize,
    /// Message index within the request (for error messages).
    pub message_index: usize,
    /// Decoded image bytes.
    pub bytes: Vec<u8>,
    /// MIME type, e.g. "image/png".
    pub media_type: String,
    /// OpenAI `detail` value (ignored by families without resolution tiers).
    pub detail: ImageDetail,
    /// Anthropic `max_width`/`max_height` cap, if provided.
    pub max_dims: Option<(u32, u32)>,
}

/// How image markers appear in the tokenized prompt for a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageTokenLayoutKind {
    /// LLaVA: a single `<image>` token becomes one sentinel position.
    Sentinel,
    /// Gemma/Qwen: `<start>` + k sentinels + `<end>`; start/end stay as
    /// regular token embeddings.
    StartEndPad,
}

/// Capability metadata shared by the route layer and `higgs doctor`.
#[derive(Debug, Clone, Default)]
pub struct VisionCapabilities {
    pub families: Vec<&'static str>,
    pub image_sizes: Vec<i32>,
    pub supported_media: Vec<&'static str>,
    pub layout_kind: ImageTokenLayoutKind,
}

/// Start/end/pad token ids for the family's image marker run.
///
/// `pad` marks positions whose embeddings are replaced by image features;
/// `start`/`end` (when present) stay as regular token embeddings.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageTokenLayout {
    pub start: Option<u32>,
    pub end: Option<u32>,
    pub pad: Option<u32>,
}

/// Preprocessed images, opaque to the engine. Only the model impl knows the
/// internal arrangement of `pixel_values`.
#[derive(Debug, Clone)]
pub struct ImageBatch {
    /// Family-native pixel layout for all images (e.g. `[N, H, W, 3]`).
    pub pixel_values: Array,
    /// Number of embedding rows each image expands to, in image order.
    pub per_image_tokens: Vec<usize>,
    /// Token layout for this batch (k can depend on preprocessing output).
    pub layout: ImageTokenLayout,
}

/// Errors produced while processing images before generation.
#[derive(Debug)]
pub enum VisionError {
    UnsupportedMediaType(String),
    ImageTooLarge(usize, usize),
    Decode(String),
    Preprocess(String),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedMediaType(m) => write!(f, "unsupported media type: {m}"),
            Self::ImageTooLarge(bytes, cap) => {
                write!(f, "image is {bytes} bytes, exceeding the {cap}-byte cap")
            }
            Self::Decode(e) => write!(f, "failed to decode image: {e}"),
            Self::Preprocess(e) => write!(f, "image preprocessing failed: {e}"),
        }
    }
}

impl std::error::Error for VisionError {}

/// A vision-language model.
pub trait VisionModel {
    /// Capability metadata for capability gating and doctor reports.
    fn vision_capabilities(&self) -> VisionCapabilities;
    /// Marker text injected into the prompt at each image position **before**
    /// tokenization (e.g. `"<image>"`, `"<|vision_start|><|image_pad|><|vision_end|>"`).
    fn image_marker_text(&self) -> &'static str;
    /// Family-native preprocessing for all images in a request.
    fn preprocess_images(&self, images: &[ImageInput]) -> Result<ImageBatch, VisionError>;
    /// Expand marker token ids in `tokens` into the exact sentinel run for
    /// this batch (called after `prepare_chat_prompt`).
    fn postprocess_image_tokens(
        &self,
        tokens: &mut Vec<u32>,
        tokenizer: &Tokenizer,
        batch: &ImageBatch,
    ) -> Result<(), VisionError>;
    /// Forward pass for text + images. `input_ids` is `[1, L]` with
    /// `IMAGE_TOKEN_INDEX` at image-feature positions.
    fn forward_multimodal(
        &mut self,
        input_ids: &Array,
        batch: &ImageBatch,
        cache: &mut AnyCache,
    ) -> Result<Array, Exception>;
}

/// Whether an image media type is accepted by any family (shared validation).
pub fn is_supported_image_media_type(media_type: &str) -> bool {
    matches!(
        media_type,
        "image/png" | "image/jpeg" | "image/jpg" | "image/webp" | "image/gif" | "image/bmp"
    )
}
```

Note: `Tokenizer` is `mlx_rs::tokenizer::Tokenizer` — import as `use mlx_rs::tokenizer::Tokenizer;` and `use crate::AnyCache;` in the same module. If `AnyCache` is not `pub` at crate root, make it `pub` (it already is — `crate::AnyCache` is used across modules).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs-models vision::tests --lib`
Expected: PASS (4 tests).

- [ ] **Step 5: Wire the module and commit**

Add `pub mod vision;` to `crates/higgs-models/src/lib.rs` (alphabetical position near `utils`).

```bash
git add crates/higgs-models/src/vision.rs crates/higgs-models/src/lib.rs
git commit -m "feat(models): add VisionModel trait and shared vision types"
```

---

### Task 2: `as_vision()` + capabilities on `AnyModel`; engine delegation

**Files:**
- Modify: `crates/higgs-models/src/lib.rs`
- Modify: `crates/higgs-engine/src/simple.rs:520-551`
- Modify: `crates/higgs/src/state.rs:120-140`

**Interfaces:**
- Consumes: `VisionModel`, `VisionCapabilities`, `ImageBatch` from Task 1.
- Produces:
  - `AnyModel::as_vision(&mut self) -> Option<&mut dyn VisionModel>`
  - `AnyModel::vision_capabilities(&self) -> Option<VisionCapabilities>`
  - `AnyModel::forward_multimodal(&mut self, input_ids: &Array, batch: &ImageBatch, cache: &mut AnyCache) -> Result<Array, Exception>` (signature change: `pixel_values: &Array` → `batch: &ImageBatch`)
  - `SimpleEngine::is_vlm(&self) -> bool` (unchanged signature, new impl), `SimpleEngine::image_marker_text(&self) -> Option<&'static str>`, `SimpleEngine::postprocess_image_tokens(&self, tokens: &mut Vec<u32>, batch: &ImageBatch) -> Result<(), VisionError>`, `SimpleEngine::vision_capabilities(&self) -> Option<VisionCapabilities>`
  - `state::Engine::is_vlm()`, `image_marker_text()`, `postprocess_image_tokens()`, `vision_capabilities()` (delegating; `vlm_image_size` and `replace_image_tokens` removed)
  - `SimpleEngine::generate_with_thinking` / `generate_streaming_with_thinking`: parameter `pixel_values: Option<Array>` → `image_batch: Option<ImageBatch>`

- [ ] **Step 1: Write the failing test (models crate)**

The cheap, decisive test is on text-only models (no safetensors needed). LLaVA model construction requires real weights, so cover the positive path at the engine level with the `Stub` variant in `state.rs` (which is `#[cfg(test)]`), plus a trait-level test asserting a hand-built `ImageBatch` flows through `merge_embeddings` (Task 7 covers that; here keep it to capabilities):

```rust
#[test]
fn text_models_report_no_vision() {
    // Build a tiny Transformer/Qwen3Next/Gemma model through the same
    // constructors the existing test modules use (they build real small
    // models from generated weights). Assert:
    //   model.as_vision().is_none()
    //   model.vision_capabilities().is_none()
    //   model.is_vlm() == false
}

#[test]
fn llava_reports_vision_capabilities() {
    // Only if the file already has a cheap way to build a LlavaQwen2Model
    // (check llava_qwen2.rs's test module). If not, delete this test and rely
    // on the engine-level Stub test below plus Task 7's merge test.
}
```

And in `state.rs`'s test module (Stub exists):

```rust
#[test]
fn stub_engine_reports_no_vision() {
    let engine = Engine::Stub(StubEngine::default());
    assert!(!engine.is_vlm());
    assert!(engine.vision_capabilities().is_none());
}
```

If `StubEngine` lacks a default constructor, use the exact construction the existing `state.rs` tests use.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-models vision --lib && cargo test -p higgs state::tests --lib`
Expected: FAIL — `as_vision` / `vision_capabilities` do not exist.

- [ ] **Step 3: Implement `as_vision` and capabilities on `AnyModel`**

In `crates/higgs-models/src/lib.rs`, replace the existing `is_vlm` and `image_size` methods (currently `matches!(self, Self::LlavaQwen2(_))` etc.) with:

```rust
/// Whether this model has a vision tower and can accept image input.
pub fn is_vlm(&self) -> bool {
    self.as_vision().is_some()
}

/// Capability metadata if this model supports vision.
pub fn vision_capabilities(&self) -> Option<VisionCapabilities> {
    self.as_vision().map(|v| v.vision_capabilities())
}

/// The vision implementation for this model, if it has one.
pub fn as_vision(&mut self) -> Option<&mut dyn VisionModel> {
    match self {
        Self::LlavaQwen2(m) => Some(m),
        Self::Transformer(_)
        | Self::Qwen3Next(_)
        | Self::Qwen3Moe(_)
        | Self::Gemma2(_)
        | Self::Gemma3(_)
        | Self::Gemma4(_)
        | Self::Phi3(_)
        | Self::Starcoder2(_)
        | Self::DeepSeekV2(_)
        | Self::BonsaiQ1(_) => None,
    }
}
```

Note: `as_vision` requires `&mut self` while `is_vlm`/`vision_capabilities` are `&self` — keep `is_vlm` as a `matches!` on the enum (it never changes) or restructure: make `as_vision(&self) -> Option<&dyn VisionModel>` for the shared-`&self` methods and add `as_vision_mut(&mut self) -> Option<&mut dyn VisionModel>` for forward. **Decision: use both** — `as_vision(&self)` for capabilities, `as_vision_mut(&mut self)` for `forward_multimodal`. Update the trait usage accordingly:

```rust
pub fn as_vision(&self) -> Option<&dyn VisionModel> {
    match self {
        Self::LlavaQwen2(m) => Some(m),
        _ => None,
    }
}
pub fn as_vision_mut(&mut self) -> Option<&mut dyn VisionModel> {
    match self {
        Self::LlavaQwen2(m) => Some(m),
        _ => None,
    }
}
```

Then change `forward_multimodal` to take the batch:

```rust
pub fn forward_multimodal(
    &mut self,
    input_ids: &Array,
    batch: &ImageBatch,
    cache: &mut AnyCache,
) -> Result<Array, Exception> {
    match self.as_vision_mut() {
        Some(v) => v.forward_multimodal(input_ids, batch, cache),
        None => Err(Exception::custom("Model does not support multimodal forward")),
    }
}
```

Remove `image_size()` (replaced by `vision_capabilities().image_sizes`).

- [ ] **Step 4: Implement `LlavaQwen2` as `VisionModel` (partial)**

In `crates/higgs-models/src/llava_qwen2.rs`:

```rust
impl VisionModel for LlavaQwen2Model {
    fn vision_capabilities(&self) -> VisionCapabilities {
        VisionCapabilities {
            families: vec!["llava-qwen2"],
            image_sizes: vec![self.image_size],
            supported_media: vec!["image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"],
            layout_kind: ImageTokenLayoutKind::Sentinel,
        }
    }

    fn image_marker_text(&self) -> &'static str {
        "<image>"
    }

    fn preprocess_images(&self, images: &[ImageInput]) -> Result<ImageBatch, VisionError> {
        // Full multi-image implementation lands in Task 6; for now implement
        // the single-image path so the trait compiles and Phase 1 tests pass:
        let mut pixel_values = Vec::with_capacity(images.len());
        for img in images {
            pixel_values.push(crate::siglip::preprocess_image(&img.bytes, self.image_size as u32)?);
        }
        Ok(ImageBatch {
            pixel_values: if pixel_values.is_empty() {
                Array::from_slice(&[0.0f32; 3], &[0, 1, 1, 3])
            } else {
                mlx_rs::ops::concatenate_axis(&pixel_values.iter().collect::<Vec<_>>(), 0)?
            },
            per_image_tokens: vec![1; images.len()],
            layout: ImageTokenLayout::default(),
        })
    }

    fn postprocess_image_tokens(
        &self,
        tokens: &mut Vec<u32>,
        tokenizer: &Tokenizer,
        _batch: &ImageBatch,
    ) -> Result<(), VisionError> {
        let Some(marker_id) = tokenizer.token_to_id("<image>") else {
            return Ok(()); // tokenizer without <image>: nothing to expand
        };
        for t in tokens.iter_mut() {
            if *t == marker_id {
                *t = IMAGE_TOKEN_INDEX as u32;
            }
        }
        Ok(())
    }

    fn forward_multimodal(
        &mut self,
        input_ids: &Array,
        batch: &ImageBatch,
        cache: &mut AnyCache,
    ) -> Result<Array, Exception> {
        let AnyCache::KV(c) = cache else {
            return Err(Exception::custom("LLaVA-Qwen2 requires a KV cache"));
        };
        // Single-image path for Phase 1; multi-image merge lands in Task 7.
        self.forward_multimodal_single(input_ids, &batch.pixel_values, c)
    }
}
```

`forward_multimodal_single` is the existing `forward_multimodal` body renamed (keeps `batch=1` check and the old `merge_embeddings`). Update `llava_qwen2.rs`'s `IMAGE_TOKEN_INDEX` const to re-export `crate::vision::IMAGE_TOKEN_INDEX` so existing references (`simple.rs`, tests) keep compiling: `pub use crate::vision::IMAGE_TOKEN_INDEX;`.

Also update `simple.rs`:

```rust
// simple.rs — replace is_vlm/vlm_image_size/replace_image_tokens block
pub fn is_vlm(&self) -> bool {
    self.model.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_vlm()
}

pub fn image_marker_text(&self) -> Option<&'static str> {
    let model = self.model.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    model.as_vision().map(|v| v.image_marker_text())
}

pub fn vision_capabilities(&self) -> Option<VisionCapabilities> {
    let model = self.model.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    model.as_vision().map(|v| v.vision_capabilities())
}

pub fn postprocess_image_tokens(
    &self,
    tokens: &mut Vec<u32>,
    batch: &ImageBatch,
) -> Result<(), VisionError> {
    let model = self.model.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(v) = model.as_vision() else {
        return Ok(());
    };
    v.postprocess_image_tokens(tokens, &self.tokenizer, batch)
}
```

Change both generate signatures in `simple.rs` from `pixel_values: Option<Array>` to `image_batch: Option<ImageBatch>` and thread it into `prepare_generation` / `run_prefill` (the prefill calls `model.forward_multimodal(&prepared.prompt_array, pixel_values, ...)` → `batch, ...`). Update `state.rs` similarly, deleting `vlm_image_size`/`replace_image_tokens` and adding the new delegators.

- [ ] **Step 5: Update call sites in routes to compile**

Temporarily change `crates/higgs/src/routes/chat.rs` (both streaming and non-streaming paths) so the project compiles: replace the `pixel_values` construction with a placeholder that passes `None` and keep `engine.replace_image_tokens(&mut prompt_tokens)` behind `#[allow(deprecated)]`-style temporary — **or** simpler: keep `replace_image_tokens` temporarily on the engine (marked `#[deprecated]`) and pass `None` for `image_batch`. Full route rewiring happens in Task 8. The goal of this step is only: `cargo build` green with the new signatures.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p higgs-models --lib`, then `cargo test -p higgs -- --test-threads=1`
Expected: PASS (existing behavior preserved; vision tests pass).

- [ ] **Step 7: Commit**

```bash
git add crates/higgs-models/src/lib.rs crates/higgs-models/src/llava_qwen2.rs crates/higgs-engine/src/simple.rs crates/higgs/src/state.rs crates/higgs/src/routes/chat.rs
git commit -m "feat(models): expose vision through VisionModel trait on AnyModel"
```

---

## Phase 2: Media pipeline

### Task 3: `media.rs` — extraction, fetch, validation

**Files:**
- Create: `crates/higgs/src/media.rs`
- Modify: `crates/higgs/src/lib.rs` (add `pub mod media;`)

**Interfaces:**
- Consumes: `ImageInput`, `ImageDetail`, `VisionError` (Task 1); `ServerSection` fields `max_image_bytes`, `image_fetch_timeout`, `max_image_dimension` (Task 16 — **implement Task 16 first**, or use constants `MAX_IMAGE_BYTES_DEFAULT = 20 << 20`, `IMAGE_FETCH_TIMEOUT_DEFAULT = 10.0`, `MAX_IMAGE_DIMENSION_DEFAULT = 4096` in this task and wire config in Task 16).
- Produces:
  - `pub struct MediaItem { pub position: usize, pub message_index: usize, pub bytes: Vec<u8>, pub media_type: String, pub detail: ImageDetail, pub max_dims: Option<(u32, u32)> }` with `impl From<MediaItem> for ImageInput`
  - `pub struct MediaExtractor { pub max_image_bytes: usize, pub fetch_timeout: std::time::Duration, pub max_image_dimension: u32, pub http_client: reqwest::Client }` with `MediaExtractor::new(max_image_bytes: usize, fetch_timeout_secs: f64, max_image_dimension: u32) -> Result<Self, ServerError>` (primitives, not `&ServerSection`, so this task builds before Task 16 adds the config fields)
  - `pub fn extract_openai(&self, messages: &[ChatCompletionMessage]) -> Result<Vec<MediaItem>, ServerError>` — returns items in client order with `position` = index among all content parts across the message's parts; text parts advance the position but produce no item.
  - `pub fn extract_anthropic(&self, messages: &[AnthropicMessage], system: Option<&SystemPrompt>) -> Result<Vec<MediaItem>, ServerError>` — collects `Image` blocks (base64 + url sources) and `tool_result` images; rejects images in `system` with 400.
  - `fn resolve_bytes(&self, url: &str, position: usize, message_index: usize) -> Result<Vec<u8>, ServerError>` — decodes `data:` URIs, fetches `http(s)://`, applies `max_image_bytes`.
  - `fn validate_media_type(&self, media_type: &str) -> Result<(), ServerError>`

- [ ] **Step 1: Write the failing test**

Create `crates/higgs/src/media.rs` with a test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn extractor() -> MediaExtractor {
        MediaExtractor {
            max_image_bytes: 1 << 20,
            fetch_timeout: std::time::Duration::from_secs(1),
            max_image_dimension: 4096,
            http_client: reqwest::Client::new(),
        }
    }

    fn base64_png_url() -> String {
        // 1x1 red PNG
        const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        format!("data:image/png;base64,{PNG}")
    }

    #[test]
    fn decodes_base64_data_uri() {
        let e = extractor();
        let bytes = e.resolve_bytes(&base64_png_url(), 0, 0).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn rejects_oversize_image() {
        let e = extractor(); // 1 MiB cap
        let big = format!("data:image/png;base64,{}", "A".repeat(2 << 20));
        let err = e.resolve_bytes(&big, 0, 0).unwrap_err();
        assert!(err.to_string().contains("cap"));
    }

    #[test]
    fn rejects_non_image_media_type() {
        let e = extractor();
        let err = e.validate_media_type("audio/mp3").unwrap_err();
        assert!(err.to_string().contains("unsupported media type"));
    }

    #[test]
    fn preserves_position_order_across_parts() {
        // Build a message: text, image, text, image -> positions 1 and 3
        let msg = ChatCompletionMessage {
            role: "user".to_owned(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: "a".into() },
                ContentPart::ImageUrl { image_url: ImageUrl { url: base64_png_url() } },
                ContentPart::Text { text: "b".into() },
                ContentPart::ImageUrl { image_url: ImageUrl { url: base64_png_url() } },
            ])),
            ..Default::default()
        };
        let items = extractor.extract_openai(&[msg]).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].position, 1);
        assert_eq!(items[1].position, 3);
    }
}
```

Note: `ChatCompletionMessage` may not implement `Default` — construct it fully or add `#[derive(Default)]`-compatible construction using the actual field names from `crates/higgs/src/types/openai.rs` (check the struct; fill `reasoning_content: None, tool_calls: None, tool_call_id: None`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs media::tests --lib`
Expected: FAIL — `media` module not found.

- [ ] **Step 3: Write the implementation**

```rust
//! Media extraction and validation for vision requests.
//!
//! One shared pipeline for OpenAI parts and Anthropic blocks. Produces
//! position-preserving [`MediaItem`]s in client order; the engine renders
//! family markers at those positions.

use base64::Engine as _;
use reqwest::Client;

use crate::types::anthropic::{AnthropicMessage, ContentBlock, SystemPrompt};
use crate::types::openai::{ChatCompletionMessage, ContentPart, MessageContent};
use crate::ServerError;

/// Default decoded-byte cap per image (20 MiB).
pub const MAX_IMAGE_BYTES_DEFAULT: usize = 20 << 20;
/// Default HTTP fetch timeout for remote image URLs (seconds).
pub const IMAGE_FETCH_TIMEOUT_DEFAULT: f64 = 10.0;
/// Default long-edge pixel cap applied before family preprocessing.
pub const MAX_IMAGE_DIMENSION_DEFAULT: u32 = 4096;

/// A validated image from a request, in client order.
#[derive(Debug, Clone)]
pub struct MediaItem {
    pub position: usize,
    pub message_index: usize,
    pub bytes: Vec<u8>,
    pub media_type: String,
    pub detail: higgs_models::vision::ImageDetail,
    pub max_dims: Option<(u32, u32)>,
}

impl From<MediaItem> for higgs_models::vision::ImageInput {
    fn from(m: MediaItem) -> Self {
        higgs_models::vision::ImageInput {
            position: m.position,
            message_index: m.message_index,
            bytes: m.bytes,
            media_type: m.media_type,
            detail: m.detail,
            max_dims: m.max_dims,
        }
    }
}

/// Extracts and validates media items from API requests.
pub struct MediaExtractor {
    pub max_image_bytes: usize,
    pub fetch_timeout: std::time::Duration,
    pub max_image_dimension: u32,
    http_client: Client,
}

impl MediaExtractor {
    pub fn new(
        max_image_bytes: usize,
        fetch_timeout_secs: f64,
        max_image_dimension: u32,
    ) -> Result<Self, ServerError> {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs_f64(fetch_timeout_secs.max(0.1)))
            .build()
            .map_err(|e| ServerError::InternalError(format!("HTTP client build failed: {e}")))?;
        Ok(Self {
            max_image_bytes,
            fetch_timeout: std::time::Duration::from_secs_f64(fetch_timeout_secs.max(0.1)),
            max_image_dimension,
            http_client,
        })
    }

    /// Extract media from OpenAI-style chat messages, preserving part order.
    pub fn extract_openai(
        &self,
        messages: &[ChatCompletionMessage],
    ) -> Result<Vec<MediaItem>, ServerError> {
        let mut items = Vec::new();
        for (mi, msg) in messages.iter().enumerate() {
            let Some(content) = &msg.content else { continue };
            match content {
                MessageContent::Text(_) => {}
                MessageContent::Parts(parts) => {
                    for (pi, part) in parts.iter().enumerate() {
                        if let ContentPart::ImageUrl { image_url } = part {
                            let url = &image_url.url;
                            let (media_type, bytes) = self.resolve_url(url, pi, mi)?;
                            self.validate_media_type(&media_type)?;
                            items.push(MediaItem {
                                position: pi,
                                message_index: mi,
                                bytes,
                                media_type,
                                detail: image_url.detail.unwrap_or_default(),
                                max_dims: None,
                            });
                        }
                    }
                }
            }
        }
        Ok(items)
    }

    /// Extract media from Anthropic-style messages and blocks.
    pub fn extract_anthropic(
        &self,
        messages: &[AnthropicMessage],
        system: Option<&SystemPrompt>,
    ) -> Result<Vec<MediaItem>, ServerError> {
        // Anthropic spec: no images in system.
        if let Some(sys) = system {
            if sys.has_image() {
                return Err(ServerError::BadRequest(
                    "images are not allowed in the system prompt".to_owned(),
                ));
            }
        }
        let mut items = Vec::new();
        for (mi, msg) in messages.iter().enumerate() {
            let crate::types::anthropic::AnthropicContent::Blocks(blocks) = &msg.content else {
                continue;
            };
            for (bi, block) in blocks.iter().enumerate() {
                let ContentBlock::Image { source } = block else { continue };
                let source = source.as_object().ok_or_else(|| {
                    ServerError::BadRequest(format!(
                        "image source in message {mi} is not an object"
                    ))
                })?;
                let media_type = source
                    .get("media_type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("image/png")
                    .to_owned();
                self.validate_media_type(&media_type)?;
                let (max_w, max_h) = (
                    source.get("max_width").and_then(serde_json::Value::as_u64),
                    source.get("max_height").and_then(serde_json::Value::as_u64),
                );
                let src_type = source.get("type").and_then(serde_json::Value::as_str);
                let data = source.get("data").and_then(serde_json::Value::as_str);
                let bytes = match (src_type, data) {
                    (Some("base64"), Some(data)) => base64::engine::general_purpose::STANDARD
                        .decode(data)
                        .map_err(|e| {
                            ServerError::BadRequest(format!(
                                "invalid base64 image in message {mi}: {e}"
                            ))
                        })?,
                    (Some("url"), Some(url)) => self.resolve_url(url, bi, mi)?.1,
                    _ => {
                        return Err(ServerError::BadRequest(format!(
                            "unsupported image source type in message {mi}"
                        )));
                    }
                };
                items.push(MediaItem {
                    position: bi,
                    message_index: mi,
                    bytes,
                    media_type,
                    detail: higgs_models::vision::ImageDetail::Auto,
                    max_dims: match (max_w, max_h) {
                        (Some(w), Some(h)) => Some((w as u32, h as u32)),
                        _ => None,
                    },
                });
            }
        }
        Ok(items)
    }

    /// Resolve a `data:` URI or `http(s)://` URL to decoded bytes.
    pub fn resolve_url(
        &self,
        url: &str,
        position: usize,
        message_index: usize,
    ) -> Result<(String, Vec<u8>), ServerError> {
        if let Some(data) = url.strip_prefix("data:") {
            let media_type = data
                .split(';')
                .next()
                .unwrap_or("image/png")
                .to_owned();
            let b64 = data
                .split_once(";base64,")
                .map(|(_, b)| b)
                .ok_or_else(|| {
                    ServerError::BadRequest(format!(
                        "malformed data URI at part {position} of message {message_index}"
                    ))
                })?;
            let bytes = base64::engine::general_purpose::STANDARD.decode(b64).map_err(|e| {
                ServerError::BadRequest(format!(
                    "invalid base64 image at part {position} of message {message_index}: {e}"
                ))
            })?;
            self.check_size(&bytes, position, message_index)?;
            Ok((media_type, bytes))
        } else if url.starts_with("http://") || url.starts_with("https://") {
            let resp = self.http_client.get(url).send().map_err(|e| {
                ServerError::BadRequest(format!("failed to fetch image URL: {e}"))
            })?;
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("image/png")
                .to_owned();
            let bytes = resp.bytes().map_err(|e| {
                ServerError::BadRequest(format!("failed to read image response: {e}"))
            })?;
            let bytes = bytes.to_vec();
            self.check_size(&bytes, position, message_index)?;
            Ok((content_type, bytes))
        } else {
            Err(ServerError::BadRequest(format!(
                "unsupported image URL scheme at part {position} of message {message_index}"
            )))
        }
    }

    fn check_size(
        &self,
        bytes: &[u8],
        position: usize,
        message_index: usize,
    ) -> Result<(), ServerError> {
        if bytes.len() > self.max_image_bytes {
            return Err(ServerError::BadRequest(format!(
                "image at part {position} of message {message_index} is {} bytes, exceeding the {}-byte cap",
                bytes.len(),
                self.max_image_bytes
            )));
        }
        Ok(())
    }

    fn validate_media_type(&self, media_type: &str) -> Result<(), ServerError> {
        let base = media_type.split(';').next().unwrap_or(media_type).trim();
        if higgs_models::vision::is_supported_image_media_type(base) {
            Ok(())
        } else {
            Err(ServerError::BadRequest(format!(
                "unsupported media type: {media_type}"
            )))
        }
    }
}
```

Notes:
- `ImageUrl` in `crates/higgs/src/types/openai.rs` currently has only `url` — **Task 3 must also add `detail: Option<ImageDetail>`** with `#[serde(default)]`, and `ImageDetail` here is the OpenAI enum (`low|high|auto`) — reuse `higgs_models::vision::ImageDetail` with `serde` deserialization: `#[derive(Deserialize)] #[serde(rename_all = "lowercase")]` on the model enum (add `serde` derives to the Task 1 enum — extend it there: `#[derive(serde::Deserialize)] #[serde(rename_all = "lowercase")]`).
- `SystemPrompt::has_image()` may not exist — add it to `crates/higgs/src/types/anthropic.rs` (iterates blocks, `matches!(b, ContentBlock::Image { .. })`).
- `MessageContent::Parts` matching uses the existing enum shapes from `openai.rs` (`ContentPart::Text { text }`, `ContentPart::ImageUrl { image_url }`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs media::tests --lib`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/higgs/src/media.rs crates/higgs/src/lib.rs crates/higgs/src/types/openai.rs crates/higgs/src/types/anthropic.rs crates/higgs-models/src/vision.rs
git commit -m "feat: add shared media extraction pipeline for vision requests"
```

---

### Task 4: Strict capability + media gating in chat routes

**Files:**
- Modify: `crates/higgs/src/routes/chat.rs` (both `chat_completions_non_streaming` and `chat_completions_streaming`)

**Interfaces:**
- Consumes: `MediaExtractor`, `MediaItem` (Task 3); `Engine::is_vlm()`, `Engine::image_marker_text()` (Task 2).
- Produces: route-layer behavior — 400s per spec table; `Vec<MediaItem>` flowing into prompt building (Task 8 consumes this).

- [ ] **Step 1: Write the failing integration test**

In `crates/higgs/tests/integration/` add (or extend an existing request-validation test file):

```rust
#[tokio::test]
async fn image_to_text_only_model_returns_400() {
    // Serve a text-only cached model (Llama-3.2-1B-4bit), POST a chat request
    // with an image_url part, expect 400 with "no vision" in the body.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs --test integration request_validation -- --test-threads=1`
Expected: FAIL — request currently ignores images (or the test file doesn't exist yet; create it following the pattern in `crates/higgs/tests/integration/request_validation.rs`).

- [ ] **Step 3: Implement the gate**

At the top of both chat handlers, after `extract_images` is replaced:

```rust
// Replace extract_images + inject_image_placeholders with the media extractor.
let media_extractor = MediaExtractor::new(
    state.config.server.max_image_bytes,
    state.config.server.image_fetch_timeout,
    state.config.server.max_image_dimension,
)?;
let media = media_extractor.extract_openai(&req.messages)?;

if !media.is_empty() && !engine.is_vlm() {
    return Err(ServerError::BadRequest(format!(
        "model {} does not support vision (image input); \
         use a vision-capable model (e.g. llava-qwen2)",
        engine.model_name()
    )));
}

// Build effective messages: text parts with marker text at image positions.
// (Full marker rendering lands in Task 8; for now preserve text-only behavior:
//  if media.is_empty(), messages pass through unchanged.)
let effective_messages = if media.is_empty() {
    req.messages.clone()
} else {
    inject_markers(&req.messages, media.len(), engine.image_marker_text())
};
```

Where `inject_markers` (Task 8 replaces with position-aware version) temporarily keeps the old prefix behavior:

```rust
/// TEMPORARY: prefix markers like the old `<image>\n` behavior. Task 8 makes
/// this position-aware using MediaItem.position.
fn inject_markers(
    messages: &[ChatCompletionMessage],
    count: usize,
    marker: Option<&'static str>,
) -> Vec<ChatCompletionMessage> {
    let marker = marker.unwrap_or("<image>");
    messages
        .iter()
        .map(|m| {
            let Some(content) = &m.content else { return m.clone() };
            if !content.has_images() {
                return m.clone();
            }
            let prefix = format!("{marker}\n");
            ChatCompletionMessage {
                role: m.role.clone(),
                content: Some(MessageContent::Text(format!(
                    "{prefix}{}",
                    content.text()
                ))),
                reasoning_content: m.reasoning_content.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            }
        })
        .collect()
}
```

Keep passing `image_batch: None` for now (Task 8 wires the real batch).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs --test integration request_validation -- --test-threads=1`
Expected: PASS — 400 with "does not support vision".

- [ ] **Step 5: Commit**

```bash
git add crates/higgs/src/routes/chat.rs crates/higgs/tests/integration/
git commit -m "feat: strict 400 when images are sent to text-only models"
```

---

## Phase 3: LLaVA multi-image

### Task 5: Batched SigLIP preprocessing

**Files:**
- Modify: `crates/higgs-models/src/siglip.rs`

**Interfaces:**
- Consumes: `ImageInput`, `VisionError` (Task 1).
- Produces:
  - `pub fn preprocess_images_batch(images: &[&[u8]], image_size: u32) -> Result<Array, VisionError>` — returns `[N, H, W, 3]`, `N == images.len()`.
  - `pub fn preprocess_image_resized(image_bytes: &[u8], target: (u32, u32), filter: image::imageops::FilterType) -> Result<Array, VisionError>` — decode, resize to exact target, normalize SigLIP-style ([0,1] → mean 0.5/std 0.5).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn batch_preprocess_shape_matches_count() {
    let png = include_bytes!("../../../../tests/fixtures/1x1.png"); // or the base64 from Task 3
    let arr = preprocess_images_batch(&[png.as_slice(), png.as_slice()], 32).unwrap();
    assert_eq!(arr.shape(), &[2, 32, 32, 3]);
}
```

Add `tests/fixtures/1x1.png` (1×1 red PNG; can be generated via `image` crate in the test itself if fixtures are undesirable — use `image::RgbImage::new(1,1)` and write bytes in-test).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-models siglip::tests --lib`
Expected: FAIL — `preprocess_images_batch` not defined.

- [ ] **Step 3: Implement**

```rust
/// Preprocess N images into a single `[N, H, W, 3]` NHWC array.
pub fn preprocess_images_batch(
    images: &[&[u8]],
    image_size: u32,
) -> Result<Array, crate::vision::VisionError> {
    let arrays: Vec<Array> = images
        .iter()
        .map(|b| preprocess_image(b, image_size))
        .collect::<Result<_, _>>()
        .map_err(|e| crate::vision::VisionError::Preprocess(e.to_string()))?;
    if arrays.is_empty() {
        return Ok(Array::from_slice(&[0.0f32; 3], &[0, 1, 1, 3]));
    }
    let refs: Vec<&Array> = arrays.iter().collect();
    mlx_rs::ops::concatenate_axis(&refs, 0)
        .map_err(|e| crate::vision::VisionError::Preprocess(e.to_string()))
}

/// Decode, resize to an exact target, and apply SigLIP normalization.
pub fn preprocess_image_resized(
    image_bytes: &[u8],
    target: (u32, u32),
    filter: image::imageops::FilterType,
) -> Result<Array, crate::vision::VisionError> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| crate::vision::VisionError::Decode(e.to_string()))?;
    let resized = img.resize_exact(target.0, target.1, filter);
    let rgb = resized.to_rgb8();
    let (w, h) = rgb.dimensions();
    let pixels = rgb.into_raw();
    let mut float_pixels: Vec<f32> = pixels.iter().map(|&p| f32::from(p) / 255.0).collect();
    for pixel in &mut float_pixels {
        *pixel = (*pixel - 0.5) / 0.5;
    }
    #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
    Ok(Array::from_slice(&float_pixels, &[1, h as i32, w as i32, 3]))
}
```

Refactor `preprocess_image` to delegate to `preprocess_image_resized` (keeps existing callers/tests working).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs-models siglip::tests --lib`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/higgs-models/src/siglip.rs crates/higgs-models/src/vision.rs
git commit -m "feat(models): batched SigLIP preprocessing for multi-image"
```

---

### Task 6: LLaVA multi-image preprocessing + marker position-aware rendering

**Files:**
- Modify: `crates/higgs-models/src/llava_qwen2.rs` (`preprocess_images` full impl)
- Modify: `crates/higgs/src/routes/chat.rs` (`inject_markers` → position-aware)

**Interfaces:**
- Consumes: `preprocess_images_batch` (Task 5), `MediaItem.position` (Task 3).
- Produces: `LlavaQwen2Model::preprocess_images` returning `per_image_tokens = [1; N]`; position-aware marker rendering producing one `<image>` marker per image **at the part's true position** within the message text.

- [ ] **Step 1: Write the failing test (models)**

```rust
#[test]
fn llava_multi_image_batch() {
    let model = build_test_model();
    let png = /* 1x1 png bytes */;
    let inputs = vec![
        ImageInput { position: 0, message_index: 0, bytes: png.clone(), media_type: "image/png".into(), detail: ImageDetail::Auto, max_dims: None },
        ImageInput { position: 2, message_index: 0, bytes: png, media_type: "image/png".into(), detail: ImageDetail::Auto, max_dims: None },
    ];
    let batch = model.preprocess_images(&inputs).unwrap();
    assert_eq!(batch.per_image_tokens, vec![1, 1]);
    assert_eq!(batch.pixel_values.shape().first(), Some(&2));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-models llava_qwen2::tests --lib`
Expected: FAIL — multi-image batch not handled.

- [ ] **Step 3: Implement `preprocess_images` fully**

```rust
fn preprocess_images(&self, images: &[ImageInput]) -> Result<ImageBatch, VisionError> {
    // detail/max_dims resolution: LLaVA is a fixed square processor, so
    // detail=Low downscales the target; High/Auto use image_size.
    let target = match images.iter().map(|i| i.detail).max().unwrap_or_default() {
        ImageDetail::Low => (self.image_size / 2).max(128),
        _ => self.image_size,
    } as u32;
    let bytes: Vec<&[u8]> = images.iter().map(|i| i.bytes.as_slice()).collect();
    let pixel_values = crate::siglip::preprocess_images_batch(&bytes, target)?;
    Ok(ImageBatch {
        pixel_values,
        per_image_tokens: vec![1; images.len()],
        layout: ImageTokenLayout::default(),
    })
}
```

- [ ] **Step 4: Position-aware marker rendering in chat.rs**

Replace the temporary `inject_markers` with a version that walks each message's parts in order and splices the marker text between text parts:

```rust
/// Rebuild message content with the family marker inserted at each image
/// part's true position. Text parts keep their relative order.
fn render_markers(
    messages: &[ChatCompletionMessage],
    marker: Option<&'static str>,
) -> Vec<ChatCompletionMessage> {
    let marker = marker.unwrap_or("<image>");
    messages
        .iter()
        .map(|m| {
            let Some(content) = &m.content else { return m.clone() };
            let MessageContent::Parts(parts) = content else { return m.clone() };
            let mut out = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => out.push_str(text),
                    ContentPart::ImageUrl { .. } => out.push_str(marker),
                }
            }
            ChatCompletionMessage {
                role: m.role.clone(),
                content: Some(MessageContent::Text(out)),
                reasoning_content: m.reasoning_content.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            }
        })
        .collect()
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p higgs-models llava_qwen2::tests --lib && cargo test -p higgs -- --test-threads=1`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/higgs-models/src/llava_qwen2.rs crates/higgs/src/routes/chat.rs
git commit -m "feat(vlm): multi-image LLaVA preprocessing with position-aware markers"
```

---

### Task 7: Generalized `merge_embeddings` (N images)

**Files:**
- Modify: `crates/higgs-models/src/vision.rs` (add `merge_embeddings`)
- Modify: `crates/higgs-models/src/llava_qwen2.rs` (use it; delete local `merge_embeddings`)

**Interfaces:**
- Consumes: `ImageBatch` (Task 1).
- Produces:
  - `pub fn merge_embeddings(input_ids: &Array, text_embeddings: &Array, image_features: &Array, batch: &ImageBatch) -> Result<Array, Exception>`
    - `image_features` is `[sum(per_image_tokens), hidden_size]` (concatenated feature rows in image order; the family impl concatenates per-image features).
    - Walks the token sequence; at each `IMAGE_TOKEN_INDEX` position, consumes the next feature row; start/end/pad tokens (non-sentinel) keep their token embeddings.
    - Validates that the number of sentinel positions equals `sum(per_image_tokens)`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn merge_two_images_in_one_sequence() {
    // input_ids: [text, SENTINEL, text, SENTINEL, text]
    let ids = Array::from_slice(&[1i32, IMAGE_TOKEN_INDEX, 2, IMAGE_TOKEN_INDEX, 3], &[1, 5]);
    // text embeddings for ids 1,2,3 are [1, 5, 4] (lookup table of 4-dim rows)
    let table = Array::from_slice(
        &[0.0f32, 0.1, 0.2, 0.3,  /* id0 */
          1.0, 1.1, 1.2, 1.3,  /* id1 */
          2.0, 2.1, 2.2, 2.3,  /* id2 */
          3.0, 3.1, 3.2, 3.3], /* id3 */
        &[4, 4],
    );
    // build text_embeddings [1,5,4] by gathering ids -> use mlx_rs::ops::take or manual
    // image features: 2 images, 1 token each, dim 4
    let features = Array::from_slice(&[9.0f32, 9.1, 9.2, 9.3, 8.0, 8.1, 8.2, 8.3], &[2, 4]);
    let batch = ImageBatch {
        pixel_values: Array::from_slice(&[0.0f32; 3], &[1, 1, 1, 3]),
        per_image_tokens: vec![1, 1],
        layout: ImageTokenLayout::default(),
    };
    let merged = merge_embeddings(&ids, &text_embeddings, &features, &batch).unwrap();
    assert_eq!(merged.shape(), &[1, 5, 4]);
    // position 1 row == features[0], position 3 row == features[1]
}
```

(The exact embedding-gather code should use `mlx_rs::ops::take(&table, &ids, 0)` — verify against the existing `merge_embeddings` in `llava_qwen2.rs`, which already handles the single-image case with `image_feats.index(0)`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-models vision::tests --lib`
Expected: FAIL — `merge_embeddings` not defined.

- [ ] **Step 3: Implement**

Move and generalize the existing `merge_embeddings` from `llava_qwen2.rs` (lines ~200-270) into `vision.rs`. The existing logic: find `image_positions`, error if empty, take `image_feats = features.index(0)`, split the sequence around the position, and splice. Generalization:

```rust
pub fn merge_embeddings(
    input_ids: &Array,
    text_embeddings: &Array,
    image_features: &Array,
    batch: &ImageBatch,
) -> Result<Array, Exception> {
    eval([input_ids])?;
    let ids: Vec<i32> = input_ids.index(0).as_slice::<i32>().to_vec();

    let expected: usize = batch.per_image_tokens.iter().sum();
    let sentinel_count = ids.iter().filter(|id| **id == IMAGE_TOKEN_INDEX).count();
    if sentinel_count != expected {
        return Err(Exception::custom(format!(
            "expected {expected} image feature positions, found {sentinel_count}"
        )));
    }
    if expected == 0 {
        return Ok(text_embeddings.clone());
    }

    // image_features: [sum(per_image_tokens), hidden]
    let hidden = image_features.shape().last().copied().unwrap_or(0);
    let mut feature_rows: Vec<Array> = Vec::with_capacity(expected);
    for i in 0..expected {
        feature_rows.push(image_features.index((i as i64..i as i64 + 1, ..))); // [1, hidden]
    }

    let mut segments: Vec<Array> = Vec::new();
    let mut feat_idx = 0usize;
    for (i, id) in ids.iter().enumerate() {
        if *id == IMAGE_TOKEN_INDEX {
            segments.push(feature_rows[feat_idx].clone());
            feat_idx += 1;
        } else {
            segments.push(text_embeddings.index((.., i as i64..i as i64 + 1, ..)));
        }
    }
    // Concatenate segments along axis 1.
    let refs: Vec<&Array> = segments.iter().collect();
    mlx_rs::ops::concatenate_axis(&refs, 1)
}
```

Notes:
- `index` calls must match the existing code's conventions in `llava_qwen2.rs` (`IndexOp` import, `NewAxis` handling). Port the exact indexing style from the existing function to avoid `mlx_rs` API mismatch.
- The existing function returns `text_embeddings.clone()` early when `image_positions.is_empty()` — preserved via the `expected == 0` branch.
- `start`/`end`/`pad` tokens are **not** sentinels, so they naturally fall into the else branch (kept as token embeddings) — the layout only matters for families that expand pads into sentinels during `postprocess_image_tokens`.

- [ ] **Step 4: Rewire LLaVA to use it**

In `llava_qwen2.rs`, replace the single-image body of `forward_multimodal` (via `forward_multimodal_single` → new `forward_multimodal`) so multi-image works:

```rust
fn forward_multimodal(
    &mut self,
    input_ids: &Array,
    batch: &ImageBatch,
    cache: &mut AnyCache,
) -> Result<Array, Exception> {
    let AnyCache::KV(c) = cache else {
        return Err(Exception::custom("LLaVA-Qwen2 requires a KV cache"));
    };
    // Validate batch=1 (unchanged).
    let batch_size = input_ids.shape().first().copied().unwrap_or(0);
    if batch_size != 1 {
        return Err(Exception::custom(format!(
            "LLaVA-Qwen2 only supports batch_size=1, got {batch_size}"
        )));
    }
    let image_features = self.encode_image_batch(&batch.pixel_values)?; // [sum(per_image_tokens), hidden]
    let sentinel = Array::from_slice(&[IMAGE_TOKEN_INDEX], &[1]);
    let is_sentinel = input_ids.eq(&sentinel)?;
    let zero = Array::from_slice(&[0_i32], &[1]);
    let safe_ids = mlx_rs::ops::r#where(&is_sentinel, &zero, input_ids)?;
    let text_embeddings = self.language_model.embed_tokens(&safe_ids)?;
    let combined = crate::vision::merge_embeddings(input_ids, &text_embeddings, &image_features, batch)?;
    self.language_model.forward_from_embeddings(&combined, None, c)
}
```

Add `encode_image_batch` to `LlavaQwen2Model` — encodes each image through the tower+projector and concatenates rows:

```rust
/// Encode N images. `pixel_values`: `[N, H, W, 3]`.
/// Returns `[sum(per_image_tokens), hidden]` — for LLaVA, `[N, num_patches, hidden]`
/// reshaped to `[N * num_patches, hidden]`.
pub fn encode_image_batch(&mut self, pixel_values: &Array) -> Result<Array, Exception> {
    let n = pixel_values.shape().first().copied().unwrap_or(0);
    if n <= 1 {
        // Single-image path (kept for exactness with existing behavior).
        let feats = self.encode_image(pixel_values)?; // [1, num_patches, hidden]
        return feats.index(0);
    }
    let mut rows = Vec::new();
    for i in 0..n {
        let single = pixel_values.index((i as i64..i as i64 + 1, .., .., ..));
        let feats = self.encode_image(&single)?; // [1, num_patches, hidden]
        rows.push(feats.index(0)); // [num_patches, hidden]
    }
    let refs: Vec<&Array> = rows.iter().collect();
    mlx_rs::ops::concatenate_axis(&refs, 0)
}
```

Remove the old local `merge_embeddings` from `llava_qwen2.rs`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p higgs-models --lib && cargo test -p higgs -- --test-threads=1`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/higgs-models/src/vision.rs crates/higgs-models/src/llava_qwen2.rs
git commit -m "feat(vlm): generalized embedding merge for multi-image VLM forward"
```

---

### Task 8: Wire `ImageBatch` through the simple engine and routes

**Files:**
- Modify: `crates/higgs/src/routes/chat.rs` (both handlers — replace `pixel_values` block with real batch construction)
- Modify: `crates/higgs-engine/src/simple.rs` (postprocess call in generate paths)

**Interfaces:**
- Consumes: `MediaExtractor`, `MediaItem`, `render_markers` (Task 3/6), `Engine::postprocess_image_tokens`, `Engine::is_vlm` (Task 2), `generate_with_thinking(…, image_batch: Option<ImageBatch>)` (Task 2).
- Produces: end-to-end LLaVA multi-image serving on the OpenAI path — streaming and non-streaming.

- [ ] **Step 1: Write the failing integration test**

Extend the integration suite:

```rust
#[tokio::test]
async fn llava_single_image_chat_non_streaming() {
    // Serve cached nanoLLaVA (or a small llava-qwen2 checkpoint), POST a chat
    // request with one image_url (base64 1x1 png) + text, expect 200 and a
    // text completion.
}

#[tokio::test]
async fn llava_two_images_chat_non_streaming() {
    // Same with two image parts; expect 200 (multi-image no longer errors).
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs --test integration -- --test-threads=1`
Expected: FAIL — second image errors (old single-image path) or batch not wired.

- [ ] **Step 3: Wire the batch in both handlers**

Replace the `pixel_values` block in `chat_completions_non_streaming`:

```rust
let image_batch = if !media.is_empty() && engine.is_vlm() {
    let inputs: Vec<higgs_models::vision::ImageInput> =
        media.into_iter().map(MediaItem::into).collect();
    let batch = engine
        .preprocess_images(&inputs)
        .map_err(|e| ServerError::BadRequest(e.to_string()))?;
    engine
        .postprocess_image_tokens(&mut prompt_tokens, &batch)
        .map_err(ServerError::Engine)?;
    Some(batch)
} else {
    None
};
```

`preprocess_images` is a new engine/state method (no `as_vision_model` — the state layer owns the model lock):

```rust
// state.rs
pub fn preprocess_images(
    &self,
    images: &[higgs_models::vision::ImageInput],
) -> Result<higgs_models::vision::ImageBatch, higgs_models::vision::VisionError> {
    match self {
        Self::Simple(e) => e.preprocess_images(images),
        Self::Batch(_) => Err(higgs_models::vision::VisionError::Preprocess(
            "batch engine has no vision model".to_owned(),
        )),
        #[cfg(test)]
        Self::Stub(_) => Err(higgs_models::vision::VisionError::Preprocess("stub".to_owned())),
    }
}
```

And on `SimpleEngine`:

```rust
pub fn preprocess_images(
    &self,
    images: &[higgs_models::vision::ImageInput],
) -> Result<higgs_models::vision::ImageBatch, higgs_models::vision::VisionError> {
    let model = self.model.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let v = model.as_vision().ok_or_else(|| {
        higgs_models::vision::VisionError::Preprocess("model has no vision".to_owned())
    })?;
    v.preprocess_images(images)
}
```

Then pass `image_batch` (not `pixel_values`) to `engine.generate_with_thinking(..., image_batch)`. Do the same in the streaming handler.

In `simple.rs` `run_prefill`, the multimodal branch becomes:

```rust
if let Some(ref batch) = prepared.image_batch {
    prepared
        .model
        .forward_multimodal(&prepared.prompt_array, batch, &mut prepared.cache)
        .map_err(EngineError::Mlx)?
} else {
    // ... existing text-only paths
}
```

And `prepare_generation`'s `has_images` check uses `image_batch.is_some()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs --test integration -- --test-threads=1`
Expected: PASS — single and double image requests both complete.

- [ ] **Step 5: Commit**

```bash
git add crates/higgs/src/routes/chat.rs crates/higgs/src/state.rs crates/higgs-engine/src/simple.rs
git commit -m "feat(vlm): end-to-end multi-image LLaVA on OpenAI chat API"
```

---

## Phase 4: Qwen3Next backbone

### Task 9: `Qwen3NextCausalLM::forward_from_embeddings`

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next.rs` (near `forward_hidden`, ~line 3891)

**Interfaces:**
- Produces:
  - `pub fn forward_from_embeddings(&mut self, embeddings: &Array, mask: Option<&Array>, kv_cache: &mut Vec<Option<LayerCache>>) -> Result<Array, Exception>` — identical semantics to `forward_hidden` but skips `embed_tokens` lookup (the caller merges image features into the embedding array first). Returns logits for the **last position only** (like `forward`), matching what the engines consume. Also `pub fn forward_from_embeddings_hidden(...)` if hidden states are needed by MTP-adjacent paths.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn forward_from_embeddings_matches_forward_hidden_on_text() {
    // Build a tiny Qwen3Next model via the existing test constructor.
    // tokenize a short prompt -> tokens.
    // h1 = model.forward_hidden(&ids_array, None, &mut cache1)?;
    // emb = model.embed_tokens_from_ids(&tokens)?;  // need public embed path
    // h2 = model.forward_from_embeddings(&emb, None, &mut cache2)?;
    // assert h1 == h2 (bitwise, or within 1e-6 after eval).
}
```

If `embed_tokens_from_ids` is private, use the existing public `embed_token(id)` per token and concatenate, or make `embed_tokens_from_ids` pub(crate) for the test. Follow whatever the file's existing test helpers do (there are large test modules in `qwen3_next.rs`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-models qwen3_next::tests --lib`
Expected: FAIL — `forward_from_embeddings` not defined.

- [ ] **Step 3: Implement**

Refactor `forward_raw_hidden` so the embedding lookup is a parameter:

```rust
fn forward_raw_hidden(
    &mut self,
    inputs: &Array,
    _mask: Option<&Array>,
    kv_cache: &mut Vec<Option<LayerCache>>,
) -> Result<Array, Exception> {
    let h = self.model.embed_tokens.forward(inputs)?;
    self.forward_raw_from_hidden(h, kv_cache)
}

/// Layer stack from a pre-computed hidden/embedding array (no embed lookup).
fn forward_raw_from_hidden(
    &mut self,
    h: Array,
    kv_cache: &mut Vec<Option<LayerCache>>,
) -> Result<Array, Exception> {
    // body of the old forward_raw_hidden starting from `let mut h = ...`
    // minus the embed_tokens lookup
}

/// Forward from pre-merged embeddings (VLM path). Returns last-position logits.
pub fn forward_from_embeddings(
    &mut self,
    embeddings: &Array,
    mask: Option<&Array>,
    kv_cache: &mut Vec<Option<LayerCache>>,
) -> Result<Array, Exception> {
    let h = self.forward_raw_from_hidden(embeddings.clone(), kv_cache)?;
    let h_normed = self.model.norm.forward(&h)?;
    let h_last = h_normed.index((.., -1.., ..));
    match self.lm_head.as_ref() {
        Some(head) => head.forward(&h_last),
        None => self.model.embed_tokens.as_linear(&h_last),
    }
}
```

Careful: the existing `forward_raw_hidden` builds the attention mask from `h.shape()` and handles the `kv_cache` sizing — all of that moves into `forward_raw_from_hidden` unchanged. `forward_with_hidden` and `forward_hidden` keep working through the refactor. The `_mask: Option<&Array>` param is currently ignored in `forward_raw_hidden` — preserve that behavior exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs-models qwen3_next::tests --lib`
Expected: PASS — parity test and all existing qwen3_next tests.

- [ ] **Step 5: Commit**

```bash
git add crates/higgs-models/src/qwen3_next.rs
git commit -m "feat(models): forward_from_embeddings for Qwen3Next backbone (Qwen3.5/eschamoe)"
```

---

## Phase 5: Qwen-VL adapter

### Task 10: Adapter detection — `_vl` suffix + `LoadKind::QwenVl`

**Files:**
- Modify: `crates/higgs-models/src/adapter.rs`

**Interfaces:**
- Consumes: `DetectedModel`, `classify`, `strip_text_alias` (existing).
- Produces:
  - `qwen_revision` learns `_vl` suffix: `Some((minor, false))` for `qwen3_5_vl` etc. (keep MoE distinction for `*_moe`; a `*_vl` wrapper is resolved by the wrapper detection path, so the text backbone inside is `qwen3_5`/`qwen3_5_moe` via `text_config`).
  - `LoadKind::QwenVl` adapter registered with `is_exact` on `qwen3_5_vl` | `qwen3_vl` | `qwen2_5_vl` (and tolerant match on the `_vl` suffix family).
  - `AdapterInfo` for the new adapter (`family: "Qwen-VL"`).
  - Loading delegates to `crate::qwen_vl::load_qwen_vl_model_from_value(dir, raw)` (Task 11).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn qwen_revision_accepts_vl_suffix() {
    assert_eq!(qwen_revision("qwen3_5_vl"), Some((5, false)));
    assert_eq!(qwen_revision("qwen3_vl"), Some((3, false)));
    assert_eq!(qwen_revision("qwen2_5_vl"), None); // qwen2_5 is a different prefix
}

#[test]
fn qwen_vl_adapter_detected() {
    // detect() on a config.json with model_type=qwen3_5_vl and nested
    // text_config -> resolve() returns the QwenVl adapter.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-models adapter::tests --lib`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `adapter.rs`:

```rust
fn qwen_revision(model_type: &str) -> Option<(u32, bool)> {
    let normalized = strip_text_alias(model_type);
    let rest = normalized.strip_prefix("qwen3_")?;
    let (minor_text, suffix) = rest
        .split_once('_')
        .map_or((rest, None), |(minor, suffix)| (minor, Some(suffix)));
    let minor = minor_text.parse().ok()?;
    match suffix {
        None => Some((minor, false)),
        Some("moe") => Some((minor, true)),
        Some("vl") => Some((minor, false)), // VL wrapper: text backbone inside text_config
        Some(_) => None,
    }
}
```

Add the adapter constant:

```rust
static QWEN_VL: BuiltinAdapter = BuiltinAdapter::new(
    "qwen_vl",
    "Qwen-VL",
    "Qwen-VL vision-language family on a Qwen3Next text backbone",
    LoadKind::QwenVl,
    true, // vision
);
```

Register in `BUILTINS` (order matters for resolution priority — place before the qwen3_5 adapters so `qwen3_5_vl` hits it first), add the `LoadKind::QwenVl =>` arm in `is_exact` and `load`, and the arm in `tolerant_match`:

```rust
LoadKind::QwenVl => matches!(text_alias.as_ref(), "qwen3_5_vl" | "qwen3_vl" | "qwen2_5_vl"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs-models adapter::tests --lib`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/higgs-models/src/adapter.rs
git commit -m "feat(models): detect Qwen-VL wrapper model types"
```

---

### Task 11: Qwen-VL model — tower, dynamic resolution, layout

**Files:**
- Create: `crates/higgs-models/src/qwen_vl.rs`
- Modify: `crates/higgs-models/src/lib.rs` (add `pub mod qwen_vl;` + `AnyModel::QwenVl(QwenVlModel)` variant + `as_vision`/`as_vision_mut`/`forward_multimodal` arms)

**Interfaces:**
- Consumes: `VisionModel`, `ImageBatch`, `ImageInput`, `merge_embeddings` (Task 1/7), `Qwen3NextCausalLM` (Task 9), `SigLipVisionModel`/`SigLipVisionConfig` (existing).
- Produces:
  - `pub struct QwenVlConfig { pub vision_config: SigLipVisionConfig, pub text_config: serde_json::Value, pub min_pixels: i32, pub max_pixels: i32, pub merge_size: i32, pub mm_hidden_size: i32, pub patch_size: i32 }` (parse from wrapper config; `vision_config` may be `SigLipVisionConfig`-shaped or a ViT config with `hidden_size`/`intermediate_size`/`num_hidden_layers`/`num_attention_heads`/`patch_size`).
  - `pub struct QwenVlModel { vision_tower: SigLipVisionModel, mm_projector: nn::Linear, language_model: Qwen3NextCausalLM, config: QwenVlConfig }` implementing `VisionModel`.
  - `pub fn load_qwen_vl_model_from_value(dir: &Path, raw: &serde_json::Value) -> Result<AnyModel, ModelError>` — parses config, loads vision tower weights from `vision_tower.*`, projector from `mm_projector.*`, text backbone from `text_config` via `load_qwen3_5_text_config_args_from_value` + `load_qwen3_5_model_with_args` (or the moe variant when `text_config` has experts).
  - `fn smart_resize(height: u32, width: u32, min_pixels: i32, max_pixels: i32) -> Result<(u32, u32), VisionError>` — the Qwen-VL dynamic-resolution algorithm: try resolutions by repeatedly dividing by 2 (or the standard mlx-vlm loop), pick the largest ≤ max_pixels and ≥ min_pixels; **port from the Qwen2-VL reference** (`mlx-vlm` `qwen2_5_vl.py` or transformers `image_processing_qwen2_vl.py`).
  - `fn get_grid(h: u32, w: u32, patch: u32, merge: u32) -> (u32, u32)` — `((h / patch + merge - 1) / merge, (w / patch + merge - 1) / merge)` style computation.

- [ ] **Step 1: Write the failing test (preprocessing only — no model load)**

```rust
#[test]
fn smart_resize_respects_max_pixels() {
    // 1920x1080 image, min_pixels=256*28*28, max_pixels=1280*28*28
    let (h, w) = smart_resize(1080, 1920, 256 * 28 * 28, 1280 * 28 * 28).unwrap();
    assert!(h * w <= 1280 * 28 * 28);
    assert!(h * w >= 256 * 28 * 28 || h <= 28 || w <= 28);
}

#[test]
fn grid_computation() {
    assert_eq!(get_grid(448, 448, 14, 2), (16, 16));
    assert_eq!(get_grid(28, 56, 14, 2), (1, 2));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-models qwen_vl::tests --lib`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement preprocessing + layout**

```rust
//! Qwen-VL vision-language model (Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL).

use std::path::Path;
use mlx_rs::{Array, Exception};
use mlx_rs::nn;
use crate::error::ModelError;
use crate::qwen3_next::Qwen3NextCausalLM;
use crate::siglip::{SigLipVisionConfig, SigLipVisionModel};
use crate::vision::{
    ImageBatch, ImageDetail, ImageInput, ImageTokenLayout, VisionCapabilities,
    VisionError, VisionModel, IMAGE_TOKEN_INDEX,
};
use crate::AnyCache;

/// Resolution candidates: repeatedly halve until ≤ max_pixels (Qwen-VL scheme).
pub fn smart_resize(
    height: u32,
    width: u32,
    min_pixels: i32,
    max_pixels: i32,
) -> Result<(u32, u32), VisionError> {
    let min_pixels = min_pixels.max(1) as u64;
    let max_pixels = max_pixels.max(1) as u64;
    let mut h = height;
    let mut w = width;
    loop {
        let cur = u64::from(h) * u64::from(w);
        if cur <= max_pixels {
            break;
        }
        // halve the larger dimension first (round up to keep parity)
        if h >= w {
            h = (h + 1) / 2;
        } else {
            w = (w + 1) / 2;
        }
    }
    if u64::from(h) * u64::from(w) < min_pixels {
        // enforce floor: scale up the smaller dimension
        let scale = (min_pixels as f64 / (u64::from(h) * u64::from(w)) as f64).sqrt();
        let new_h = ((f64::from(h) * scale).round() as u32).max(1);
        let new_w = ((f64::from(w) * scale).round() as u32).max(1);
        h = new_h;
        w = new_w;
    }
    Ok((h, w))
}

/// Grid of patches after pixel-shuffle merge.
pub fn get_grid(h: u32, w: u32, patch: u32, merge: u32) -> (u32, u32) {
    let gh = (h / patch).max(1);
    let gw = (w / patch).max(1);
    (((gh + merge - 1) / merge).max(1), ((gw + merge - 1) / merge).max(1))
}
```

**Full preprocessing (in `preprocess_images`)**: for each image — decode → `smart_resize` → resize → pad to multiple of `patch_size * merge_size` → split into `(gh*gw)` patches of `patch*patch` → **pixel-shuffle**: reshape to `[gh, gw, merge*merge, patch*merge, patch*merge, 3]`-style and transpose to merge 2×2 blocks into one token (the exact transpose per the Qwen2-VL reference; mlx-vlm's `qwen2_5_vl.py::preprocess` is the source of truth) → the vision tower outputs per-patch features → `per_image_tokens[i] = gh * gw`.

**Layout**:

```rust
fn postprocess_image_tokens(
    &self,
    tokens: &mut Vec<u32>,
    tokenizer: &Tokenizer,
    batch: &ImageBatch,
) -> Result<(), VisionError> {
    let start = tokenizer.token_to_id("<|vision_start|>");
    let pad = tokenizer.token_to_id("<|image_pad|>");
    let end = tokenizer.token_to_id("<|vision_end|>");
    let (Some(start), Some(pad), Some(end)) = (start, pad, end) else {
        return Err(VisionError::Preprocess(
            "tokenizer missing Qwen-VL vision tokens".to_owned(),
        ));
    };
    let mut out = Vec::with_capacity(tokens.len() + batch.per_image_tokens.iter().sum::<usize>());
    let mut img_idx = 0usize;
    let mut i = 0usize;
    while i < tokens.len() {
        let t = tokens[i];
        if t == start {
            out.push(t);
            i += 1;
            // consume <|image_pad|> and expand to k sentinels
            let k = batch.per_image_tokens.get(img_idx).copied().unwrap_or(1);
            img_idx += 1;
            while i < tokens.len() && tokens[i] == pad {
                i += 1;
            }
            for _ in 0..k {
                out.push(IMAGE_TOKEN_INDEX as u32);
            }
        } else {
            out.push(t);
            i += 1;
        }
    }
    *tokens = out;
    Ok(())
}
```

Note: the marker text rendered by the route (`<|vision_start|><|image_pad|><|vision_end|>`) tokenizes to exactly one pad between start/end; the expansion replaces it with `k` sentinels. Validate `img_idx == batch.per_image_tokens.len()` at the end.

- [ ] **Step 4: Implement the model + load**

`QwenVlModel` fields and `VisionModel` impl: `vision_capabilities` reports `qwen3_5_vl`/`qwen_vl`; `image_marker_text` returns `"<|vision_start|><|image_pad|><|vision_end|>"`; `forward_multimodal` computes `image_features = encode_all(batch)` → concatenated `[sum, hidden]` → `text_embeddings = language_model.embed_sequence(safe_ids)` (need a public batch-embed on Qwen3Next: add `pub fn embed_tokens_batch(&self, ids: &Array) -> Result<Array, Exception>` wrapping `self.model.embed_tokens.forward(ids)`) → `merge_embeddings` → `language_model.forward_from_embeddings(&combined, None, c)`.

`load_qwen_vl_model_from_value`:
- Parse `vision_config` (fall back to `vision_tower.vision_model`-style nesting if present) into `SigLipVisionConfig` (its fields match Qwen-VL ViT: hidden_size/intermediate_size/num_hidden_layers/num_attention_heads/patch_size/image_size or `temporal_patch_size`-style extras ignored).
- `min_pixels`/`max_pixels` from top-level config (defaults `256*28*28`, `1280*28*28`); `merge_size` default 2.
- Text: reuse `crate::qwen3_next::load_qwen3_5_text_config_args_from_value(&raw)` then `load_qwen3_5_model_with_args(dir, args)` (or `_moe_` variant when `text_config.model_type` ends `_moe`), yielding `Qwen3NextCausalLM`; wrap in `AnyModel::QwenVl`.
- Load safetensors: vision tower under `vision_tower.*` (use `load_siglip_weights` with prefix `"vision_tower.vision_model."` or `"vision_tower.vision_tower.vision_model."` — try both, mirroring the LLaVA loader's prefix handling), projector `mm_projector.*` (linear: `weight` + optional `bias`), text under `language_model.` prefix (the qwen3.5 loaders already strip it).

Because `Qwen3NextCausalLM` weights were loaded by the qwen3.5 loader from the **same safetensors files** (text backbone), and the vision weights come from the same files, the loader must: (1) load the text backbone with the qwen3.5 loader; (2) load vision + projector separately with `load_siglip_weights`-style code that skips already-consumed text keys. Follow the exact pattern in `llava_qwen2.rs::load_llava_qwen2_model_from_value` (which already splits vision/text loading from one checkpoint).

- [ ] **Step 5: Add `AnyModel::QwenVl` variant**

In `lib.rs`: add the variant, wire `as_vision`/`as_vision_mut`/`forward_multimodal`/`make_cache`/`supports_batched_decode` arms (cache: `QwenVl` uses the backbone's `make_cache` — reuse the `Qwen3Next` arm logic by delegating to `m.language_model.make_cache()`), and the adapter `load` arm from Task 10 calls `crate::qwen_vl::load_qwen_vl_model_from_value`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p higgs-models qwen_vl::tests --lib && cargo test -p higgs-models --lib`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/higgs-models/src/qwen_vl.rs crates/higgs-models/src/lib.rs crates/higgs-models/src/qwen3_next.rs crates/higgs-models/src/adapter.rs
git commit -m "feat(models): Qwen-VL model with dynamic-resolution vision"
```

---

## Phase 6: Gemma 3/4 vision

### Task 12: Gemma pan-and-scan preprocessing

**Files:**
- Create: `crates/higgs-models/src/gemma_vision.rs` (preprocessing only; tower loading in Task 13)

**Interfaces:**
- Consumes: `ImageInput`, `VisionError`, `preprocess_image_resized` (Task 5).
- Produces:
  - `pub struct GemmaVisionConfig { pub image_size: i32, pub patch_size: i32, pub num_patches: i32, pub crop_set: Vec<(i32, i32)> }` (from `vision_config`; `num_patches` = `(image_size/patch_size)^2`).
  - `pub fn pan_and_scan(image_bytes: &[u8], config: &GemmaVisionConfig) -> Result<PanAndScan, VisionError>` where `pub struct PanAndScan { pub crops: Vec<Array>, pub offsets: Vec<i32> }` — each crop `[1, image_size, image_size, 3]`, `offsets` per crop in patch units (e.g. `0`, `(image_size/patch_size) - 1`, ...) telling the transformer the positional offset of each crop.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn pan_and_scan_produces_expected_crop_count() {
    // config: image_size 896, patch_size 16 -> 6 crops (center + 4 corners + full)
    let cfg = GemmaVisionConfig {
        image_size: 896,
        patch_size: 16,
        num_patches: 56 * 56,
        crop_set: vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)],
    };
    let ps = pan_and_scan(&png_bytes(), &cfg).unwrap();
    assert_eq!(ps.crops.len(), cfg.crop_set.len());
    assert_eq!(ps.offsets.len(), cfg.crop_set.len());
    assert_eq!(ps.crops[0].shape(), &[1, 896, 896, 3]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-models gemma_vision::tests --lib`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```rust
//! Gemma 3/4 pan-and-scan vision preprocessing.

/// Pan-and-scan crop set: (row_frac, col_frac) in {0,1,2}² — center + corners.
/// The exact set is checkpoint-specific; `vision_config.crop_size` or the
/// reference gemma3 processor defines it. Default = 6 crops.
pub fn default_crop_set() -> Vec<(i32, i32)> {
    vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)]
}

pub fn pan_and_scan(
    image_bytes: &[u8],
    config: &GemmaVisionConfig,
) -> Result<PanAndScan, VisionError> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| VisionError::Decode(e.to_string()))?;
    let (w, h) = (img.width(), img.height());
    // aspect-ratio-preserving resize to the target square via letterbox,
    // then crop `crop_set.len()` windows of image_size².
    let resized = img.resize(config.image_size as u32, config.image_size as u32, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();
    let mut crops = Vec::new();
    let mut offsets = Vec::new();
    let patch = config.patch_size as i32;
    let grid = config.image_size / patch;
    for (r, c) in &config.crop_set {
        // offsets in patch units along each axis
        let offset_row = (*r * (grid - 1)) / 2;
        let offset_col = (*c * (grid - 1)) / 2;
        offsets.push(offset_row * grid + offset_col);
        let crop = crop_square(&rgb, *r, *c, config.image_size as u32);
        crops.push(to_normalized_array(crop));
    }
    Ok(PanAndScan { crops, offsets })
}
```

`crop_square` returns the `image_size²` RGB window for the given (row_frac, col_frac) placement (the exact crop anchor math must match the gemma3 reference processor — the anchor is `(frac * (orig - target))` clamped; verify against the HF `gemma3` image processor during implementation). `to_normalized_array` applies Gemma normalization (mean/std from `vision_config`; default mean 0.5/std 0.5 unless the checkpoint specifies `image_mean`/`image_std`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs-models gemma_vision::tests --lib`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/higgs-models/src/gemma_vision.rs crates/higgs-models/src/lib.rs
git commit -m "feat(models): Gemma 3/4 pan-and-scan image preprocessing"
```

---

### Task 13: Gemma vision towers — load weights + `VisionModel` impl

**Files:**
- Modify: `crates/higgs-models/src/gemma3.rs`, `gemma4.rs` (load vision towers when present)
- Modify: `crates/higgs-models/src/lib.rs` (`AnyModel::Gemma3`/`Gemma4` gain vision via inner wrapper — **decision:** wrap in a new `GemmaVisionModel { language_model: Gemma3CausalLM | Gemma4CausalLM, vision_tower: SigLipVisionModel, config }` variant OR make `Gemma3`/`Gemma4` implement `VisionModel` directly when a tower is present)

**Interfaces:**
- Consumes: `GemmaVisionConfig`, `PanAndScan` (Task 12), `SigLipVisionModel` (existing).
- Produces:
  - `Gemma3CausalLM`/`Gemma4CausalLM` optionally hold a vision tower; `AnyModel` arms for `as_vision` return `Some` when a tower is loaded.
  - `preprocess_images`: pan-and-scan per image → crops + offsets → `ImageBatch { pixel_values, per_image_tokens: vec![k; N] where k = num_patches / 4 (compression) or the checkpoint's image-token count, layout: { start: <start_of_image>, end: <end_of_image>, pad: None } }`.
  - `postprocess_image_tokens`: expand `<start_of_image><end_of_image>` runs into `start + k×IMAGE_TOKEN_INDEX + end`.
  - `forward_multimodal`: encode crops, apply positional offsets (position-ids array), merge features, run backbone `forward_from_embeddings` (Gemma 3/4 have their own backbone forward — **add `forward_from_embeddings` to `Gemma3CausalLM`/`Gemma4CausalLM` mirroring Task 9** if not present).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn gemma3_marker_expansion() {
    // token stream [.., start_of_image, end_of_image, ..] with k=256
    let mut tokens = vec![1u32, 2, 3]; // pretend 2 == start_of_image, 3 == end_of_image
    let batch = ImageBatch { pixel_values: Array::from_slice(&[0.0f32; 3], &[1, 1, 1, 3]),
        per_image_tokens: vec![256], layout: ImageTokenLayout { start: Some(2), end: Some(3), pad: None } };
    // call the gemma postprocessor with a tokenizer stub whose token_to_id
    // maps "<start_of_image>"->2, "<end_of_image>"->3
    // expect tokens == [1, 2, -200 x 256, 3]
}
```

(The tokenizer stub: `Tokenizer` is concrete — if it can't be stubbed, factor the expansion into a pure helper `expand_gemma_markers(tokens, start, end, per_image_tokens)` and test that directly.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-models gemma_vision::tests --lib`
Expected: FAIL.

- [ ] **Step 3: Implement the expansion helper + weight loading**

```rust
/// Pure expansion used by Gemma postprocess: `[start][end]` runs become
/// `[start][-200 × k][end]`.
pub fn expand_gemma_markers(
    tokens: &mut Vec<u32>,
    start: u32,
    end: u32,
    per_image_tokens: &[usize],
) {
    let mut out = Vec::with_capacity(tokens.len() + per_image_tokens.iter().sum::<usize>());
    let mut img_idx = 0usize;
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == start && i + 1 < tokens.len() && tokens[i + 1] == end {
            out.push(start);
            let k = per_image_tokens.get(img_idx).copied().unwrap_or(1);
            img_idx += 1;
            for _ in 0..k {
                out.push(IMAGE_TOKEN_INDEX as u32);
            }
            out.push(end);
            i += 2;
        } else {
            out.push(tokens[i]);
            i += 1;
        }
    }
    *tokens = out;
}
```

Weight loading: in `gemma3.rs`/`gemma4.rs` loaders, after the existing `load_quantized_safetensors_weights_optional_prefix_with_settings` call, detect vision-tower keys (any key containing `vision_tower.`) and, if present, load a `SigLipVisionModel` from `vision_config` (gemma3's vision config nests under `vision_config` at top level; gemma4 under `vision_config` or `vision_tower.vision_model`), using `load_siglip_weights` with the appropriate prefix (`vision_tower.vision_model.` etc.). Store it in the model struct as `Option<SigLipVisionModel>`; `as_vision` returns `Some` only when present. Keep `gemma3_text`/`gemma4_text` behavior identical (no tower keys → `None`).

- [ ] **Step 4: Implement `VisionModel` for the Gemma variants**

`preprocess_images` (pan-and-scan per image; `per_image_tokens = [k; N]` with `k = config.num_patches / 4` — verify the compression factor from `vision_config`), `image_marker_text = "<start_of_image><end_of_image>"`, `forward_multimodal` (encode crops → apply positional offsets via the backbone's position-id support or an offset tensor the attention layers consume — **the Gemma 3 reference uses `position_ids` per crop**; if the current Gemma3 backbone has no position-id input, add a minimal `forward_from_embeddings_with_offsets` that accepts an offsets array and feeds it into RoPE application. Verify against `gemma3.rs`'s RoPE handling during implementation and match the reference gemma3 model's positional-offset semantics exactly.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p higgs-models --lib && cargo test -p higgs -- --test-threads=1`
Expected: PASS (gemma3_text/gemma4_text unchanged; vision variants covered by unit tests).

- [ ] **Step 6: Commit**

```bash
git add crates/higgs-models/src/gemma_vision.rs crates/higgs-models/src/gemma3.rs crates/higgs-models/src/gemma4.rs crates/higgs-models/src/lib.rs
git commit -m "feat(models): Gemma 3/4 vision towers with pan-and-scan"
```

---

## Phase 7: Batch engine + MTP + embeddings

### Task 14: Batch engine multimodal prefill + batched decode for VLM families

> **Pre-flight ruling (human approved):** `Qwen3Next` models use
> `AnyCache::Hybrid(Vec<Option<LayerCache>>)`, and `AnyModel::forward_batched`
> rejects Hybrid caches outright. This task therefore implements
> `Qwen3NextCausalLM::forward_batched` for **Hybrid caches** (per-request
> offsets + batched projections over the hybrid SSM/attention stack, mirroring
> `Transformer::forward_batched`), so Qwen-VL (Qwen3Next-backed) gets true
> batched decode. LLaVA (Transformer-backed) delegates to the inner
> `Transformer::forward_batched`.

**Files:**
- Modify: `crates/higgs-engine/src/batch_engine.rs`
- Modify: `crates/higgs-models/src/qwen3_next.rs` (`forward_batched` for the backbone, Hybrid cache)
- Modify: `crates/higgs-models/src/llava_qwen2.rs` (delegate batched decode to the Qwen2 transformer)
- Modify: `crates/higgs-models/src/lib.rs` (`AnyModel::forward_batched` arms — accept `AnyCache::Hybrid` for `Qwen3Next`, `AnyCache::KV` for `LlavaQwen2`)
- Modify: `crates/higgs/src/doctor.rs` + startup gate + README (extend the `batch=true` family allowlist)

**Interfaces:**
- Consumes: `ImageBatch` (Task 1), `forward_multimodal` (Task 2/7/11), `BatchRequest` (existing), `Transformer::forward_batched` (existing, the porting reference).
- Produces:
  - `BatchRequest { … , image_batch: Option<ImageBatch> }`
  - `start_prefill`/`advance_prefill` multimodal path (single-pass merged-embedding forward; skip prefix-cache lookup)
  - `Qwen3NextCausalLM::forward_batched(&mut self, inputs: &Array, kv_caches: &mut [&mut Vec<Option<LayerCache>>]) -> Result<Array, Exception>` — port `Transformer::forward_batched`: per-request position offsets from each cache, batched projections over the stacked `[N, 1]` inputs, per-request RoPE/cache updates inside the layer loop, LM head applied per request. Must handle both `LayerCache::KV` and `LayerCache::Arrays` entries per request, and the hybrid SSM/attention layer mix (`qwen3_next.rs` layers are a `Vec<DecoderLayer>` where each layer may be attention or SSM — follow how `forward_raw_from_hidden` iterates them).
  - `AnyModel::forward_batched` arms: `LlavaQwen2` → inner `Transformer::forward_batched` (KV cache); `QwenVl` → `m.language_model.forward_batched` (Hybrid cache); `AnyCache::Hybrid` no longer rejected when the model is `Qwen3Next`/`QwenVl`.
  - `batch=true` allowlist extended to `llava-qwen2` and `qwen3_5_vl` (doctor + startup + README).

- [ ] **Step 1: Write the failing test**

Batch engine tests live in `batch_engine.rs`'s test module; add:

```rust
#[test]
fn multimodal_batch_request_skips_prefix_cache_and_runs_forward_multimodal() {
    // Construct a BatchRequest with image_batch=Some(...); call start_prefill;
    // assert prefix_cache is not consulted (prompt fully re-prefilled) and the
    // prefill runs forward_multimodal (observable via a stub model or by
    // asserting no panic + correct logits shape on a real LLaVA model).
}
```

And in `qwen3_next.rs`'s test module, a batched-decode parity test:

```rust
#[test]
fn forward_batched_matches_forward_on_single_request() {
    // Build a tiny Qwen3Next model (existing test constructor). Two requests:
    // (a) run forward() on one token with a fresh cache
    // (b) run forward_batched(&[token]) with one cache
    // assert logits equal (after eval) within 1e-5.
}

#[test]
fn forward_batched_two_requests_different_offsets() {
    // Request A has 3 cached tokens (offset 3), request B has 1 (offset 1).
    // Stack [A_next, B_next] through forward_batched; assert each output row
    // matches a single-request forward() at its own offset.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs-engine batch_engine::tests --lib && cargo test -p higgs-models qwen3_next::tests --lib`
Expected: FAIL — `image_batch` field doesn't exist; `forward_batched` not defined on `Qwen3NextCausalLM`.

- [ ] **Step 3: Implement**

In `batch_engine.rs`:
- Add `image_batch: Option<ImageBatch>` to `BatchRequest` (default `None` in existing constructors/tests).
- `start_prefill`: `let prefix_match = if req.image_batch.is_some() { None } else { prefix_cache.find_longest_prefix(&req.prompt_tokens) };`
- `advance_prefill`: if `prefill.req.image_batch` is `Some`, run the whole prompt in one merged-embedding forward:

```rust
if let Some(batch) = &prefill.req.image_batch {
    // Single-pass multimodal prefill: image features can't span chunks.
    let input = Array::from(prefill.tokens.as_slice()).index(NewAxis);
    let logits = model.forward_multimodal(&input, batch, &mut prefill.cache)?;
    let last_logits = logits.index((.., -1, ..));
    prefill.offset = prefill.tokens.len();
    return complete_prefill(..., last_logits).map(PrefillAdvance::Complete);
}
```

Then the batched decode work, in order:
1. **Port `Transformer::forward_batched` to `Qwen3NextCausalLM`.** Read `Transformer::forward_batched` (in `crates/higgs-models/src/transformer.rs`, ~line 697) carefully first — it is the reference: per-request offsets from `kv_caches[i].first().offset()`, batched projections over `[N, 1]` stacked tokens, then a per-request loop for RoPE/attention/cache writes. `Qwen3Next` differs in: cache entries are `LayerCache` (KV or Arrays), and layers mix attention/SSM (each `DecoderLayer` has its own forward that consumes the cache). The faithful approach for the first version: **loop layers; for each layer, loop requests** — per-request `layer.forward` with that request's cache — which is correct but not batched across requests; then, as a second pass within the same task, batch the heavy projections (Q/K/V, gate/up) across requests where the layer structure permits, keeping per-request RoPE/SSM state writes. The parity tests above must pass at the end; if full projection batching proves infeasible for SSM layers, the per-request loop over layers is acceptable **only if** the parity tests pass and the batch decode path is still one forward call per round from the engine's perspective.
2. **`AnyModel::forward_batched` arms:** accept `AnyCache::Hybrid` when the model is `Qwen3Next` (convert `&mut AnyCache` → `&mut Vec<Option<LayerCache>>`), and `AnyCache::KV` when the model is `LlavaQwen2` (delegate to the inner Qwen2 `Transformer`). Keep rejecting Hybrid for non-Qwen3Next models (Gemma 3/4 don't use it).
3. **`batch=true` gate:** extend the allowlist (`llama`, `mistral`, `qwen2`, `qwen3` + `llava-qwen2` + `qwen3_5_vl`) in `doctor.rs`, the server-startup batch check, and the README batch-support section.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs-engine --lib && cargo test -p higgs-models --lib && cargo test -p higgs -- --test-threads=1`
Expected: PASS — parity tests and all existing tests green.

- [ ] **Step 5: Commit**

```bash
git add crates/higgs-engine/src/batch_engine.rs crates/higgs-models/src/qwen3_next.rs crates/higgs-models/src/llava_qwen2.rs crates/higgs-models/src/lib.rs crates/higgs/src/doctor.rs README.md
git commit -m "feat(engine): batch-mode vision with multimodal prefill and batched decode"
```

---

### Task 15: MTP disable for image requests + embeddings rejection

**Files:**
- Modify: `crates/higgs-engine/src/simple.rs` (MTP gate)
- Modify: `crates/higgs-engine/src/batch_engine.rs` (MTP gate)
- Modify: `crates/higgs/src/routes/embeddings.rs` (reject images)

**Interfaces:**
- Consumes: `ImageBatch` (Task 1).
- Produces: behavior — a request with `image_batch.is_some()` never enters the MTP/speculative path; `/v1/embeddings` with image parts → 400.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn image_request_disables_mtp() {
    // engine with an MTP-enabled Qwen3Next model
    // generate_with_thinking(..., image_batch=Some(...)) must not call the
    // MTP draft path — observable via log line or a stub MTP head.
}
```

And an embeddings integration test: POST `/v1/embeddings` with a message containing an `image_url` part → 400.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs -- --test-threads=1`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `simple.rs`/`batch_engine.rs`, find where the MTP/speculative path is selected (grep `mtp` / `has_mtp` / `draft`) and gate it:

```rust
let use_mtp = self.model.lock()...has_mtp() && image_batch.is_none();
```

In `embeddings.rs`, before processing, scan the request content for image parts (reuse `MessageContent::has_images`) and return 400:

```rust
if req_has_images {
    return Err(ServerError::BadRequest(
        "images are not supported in embeddings requests".to_owned(),
    ));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs -- --test-threads=1`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/higgs-engine/src/simple.rs crates/higgs-engine/src/batch_engine.rs crates/higgs/src/routes/embeddings.rs
git commit -m "feat(engine): disable MTP for image requests; reject images in embeddings"
```

---

## Phase 8: Config, doctor, docs

### Task 16: Config fields + doctor validation + init template

**Files:**
- Modify: `crates/higgs/src/config.rs` (`ServerSection`, `ModelConfig`)
- Modify: `crates/higgs/src/doctor.rs`
- Modify: `crates/higgs/src/daemon.rs` (init template)

**Interfaces:**
- Produces:
  - `ServerSection.max_image_bytes: usize` (default `20 << 20`), `image_fetch_timeout: f64` (default `10.0`), `max_image_dimension: u32` (default `4096`).
  - `ModelConfig.disable_vision: bool` (default `false`).
  - Doctor checks per spec §5.2.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn config_defaults_preserve_behavior() {
    let cfg: ServerSection = toml::from_str("").unwrap();
    assert_eq!(cfg.max_image_bytes, 20 << 20);
    assert_eq!(cfg.image_fetch_timeout, 10.0);
    assert_eq!(cfg.max_image_dimension, 4096);
    let m: ModelConfig = toml::from_str(r#"path = "x""#).unwrap();
    assert!(!m.disable_vision);
}

#[test]
fn doctor_warns_when_image_cap_exceeds_body_cap() {
    // config with max_image_bytes > max_body_size -> doctor warn entry
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p higgs config::tests doctor::tests --lib -- --test-threads=1`
Expected: FAIL — fields don't exist.

- [ ] **Step 3: Implement**

Add the fields with serde defaults (follow `max_body_size` exactly), wire `MediaExtractor::new(state.config.server.max_image_bytes, state.config.server.image_fetch_timeout, state.config.server.max_image_dimension)` in the chat routes (replacing the constants used in Task 3 if any). Add doctor checks:

```rust
// doctor.rs, in the server section validation:
if server.max_image_bytes > server.max_body_size {
    warn("server.max_image_bytes > server.max_body_size: images can never arrive within the body cap", result);
}
if !(64..=16384).contains(&server.max_image_dimension) {
    fail("server.max_image_dimension must be within 64..=16384", result);
}
if server.image_fetch_timeout <= 0.0 {
    fail("server.image_fetch_timeout must be positive", result);
}
// per-model:
if model.disable_vision && /* model has no vision tower */ {
    warn("disable_vision=true on a model without vision is a no-op", result);
}
if /* checkpoint has vision weights but adapter can't run vision */ {
    warn("checkpoint contains vision weights that Higgs will ignore", result);
}
if model.batch && /* VLM family without batched decode */ {
    fail("batch=true is not supported for this vision model family yet", result);
}
```

Add the vision status to the doctor report (the `vision: none | supported (LLaVA) | tower-ignored (Gemma4)` column). Update `daemon.rs`'s `config.toml` template with the four fields + comments.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p higgs -- --test-threads=1`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/higgs/src/config.rs crates/higgs/src/doctor.rs crates/higgs/src/daemon.rs crates/higgs/src/routes/chat.rs
git commit -m "feat: vision config fields with doctor validation"
```

---

### Task 17: Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/models.md`
- Modify: `docs/configuration.md`

**Content:**
- README: "Vision support" section — families (LLaVA, Gemma 3/4, Qwen-VL), `detail`/`max_dims` semantics, HTTP URL support, 400 behavior, new config fields in the reference.
- docs/models.md: mark VLM rows with vision status; "Vision-capable models" table (square resize / pan-and-scan / dynamic resolution + per-family defaults); note escha-w2/Qwen3.5 backbones work under Qwen-VL.
- docs/configuration.md: the four new fields + "multimodal requests never use prefix/disk cache" behavior.

- [ ] **Step 1: Write the docs**

Follow the existing doc style in each file (tables, short sections, links).

- [ ] **Step 2: Commit**

```bash
git add README.md docs/models.md docs/configuration.md
git commit -m "docs: vision support, config fields, and model capabilities"
```

---

## Phase 9: Smoke harness

### Task 18: Smoke test extension

**Files:**
- Modify: `scripts/release_smoke_cached_models.sh`

- [ ] **Step 1: Add vision coverage**

Following the existing harness pattern, add a single-image non-streaming + streaming request against a cached `llava-qwen2` model (e.g. `nanoLLaVA-1.5` if cached), asserting 200 and non-empty content. Add an optional env-gated section (`HIGGS_SMOKE_INCLUDE_OPTIONAL_MODELS=1`) for a Qwen-VL checkpoint if cached. Include a negative case: an image request to a text-only cached model expecting 400.

- [ ] **Step 2: Run the harness**

Run: `HIGGS_SMOKE_INCLUDE_OPTIONAL_MODELS=0 scripts/release_smoke_cached_models.sh`
Expected: all steps pass with the new vision coverage.

- [ ] **Step 3: Commit**

```bash
git add scripts/release_smoke_cached_models.sh
git commit -m "test: smoke coverage for vision requests"
```

---

## Self-Review Notes

- **Spec coverage:** every spec section maps to tasks — trait/architecture (T1-2), media pipeline + strict errors + detail/max_dims (T3-4, T6), family preprocessing + layouts (T5-7 LLaVA, T10-11 Qwen-VL, T12-13 Gemma), backbone gap `forward_from_embeddings` (T9), simple engine (T2, T8), batch engine + MTP + embeddings (T14-15), config/doctor/docs (T16-17), smoke (T18), escha-w2/Qwen3.5 compatibility (T9-T11 — same `Qwen3NextCausalLM` backbone, no dense-expert assumptions).
- **Sequencing risk:** Task 3 references `ServerSection` fields added in Task 16 — Task 3 uses constants (listed) so it's buildable standalone; Task 16 swaps constants for config.
- **`forward_multimodal` signature change** ripples to `state.rs` and both engines — Task 2 covers all call sites.
- **AnyModel arms** (`make_cache`, `supports_batched_decode`, `forward_batched`, `is_vlm`) must be updated in every task that adds a variant (T2, T11, T13, T14) — each task lists its arms.
