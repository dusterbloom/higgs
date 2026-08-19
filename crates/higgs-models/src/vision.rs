//! Vision-language model support: shared trait, types, and preprocessing.
//!
//! Every VLM family implements [`VisionModel`]. The route layer produces
//! position-preserving [`ImageInput`]s from `OpenAI` parts or `Anthropic` blocks;
//! the engine renders family marker tokens, post-processes them into sentinel
//! runs, and merges image features into the text embedding sequence before the
//! transformer runs (see [`merge_embeddings`]).

use mlx_rs::{
    Array,
    error::Exception,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
};
use tokenizers::Tokenizer;

use crate::AnyCache;

/// Sentinel token id marking image-feature positions in the token stream.
/// Negative so it can never collide with a real token id. Replaced by image
/// features during embedding merge.
pub const IMAGE_TOKEN_INDEX: i32 = -200;

/// Resolution control from the request (`detail` / `max_width` semantics).
///
/// Serde parses the `OpenAI` wire values `"auto"` / `"low"` / `"high"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
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
    /// `OpenAI` `detail` value (ignored by families without resolution tiers).
    pub detail: ImageDetail,
    /// Anthropic `max_width`/`max_height` cap, if provided.
    pub max_dims: Option<(u32, u32)>,
}

/// How image markers appear in the tokenized prompt for a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageTokenLayoutKind {
    /// `LLaVA`: a single `<image>` token becomes one sentinel position.
    #[default]
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

/// Merge text embeddings and image features at `IMAGE_TOKEN_INDEX` positions.
///
/// `input_ids` is `[1, seq_len]` with `IMAGE_TOKEN_INDEX` at image-feature
/// positions; `text_embeddings` is the matching `[1, seq_len, hidden]` token
/// embedding sequence. `image_features` holds `sum(batch.per_image_tokens)`
/// rows of `[hidden]` in image order (the family impl concatenates per-image
/// features). Walking the token sequence, each sentinel position consumes the
/// next feature row; every other position keeps its token embedding. The
/// result is `[1, seq_len, hidden]` — one merged row per input position.
///
/// Errors when the number of sentinel positions does not match
/// `sum(batch.per_image_tokens)`. An empty batch returns the text embeddings
/// unchanged.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
pub fn merge_embeddings(
    input_ids: &Array,
    text_embeddings: &Array,
    image_features: &Array,
    batch: &ImageBatch,
) -> Result<Array, Exception> {
    // input_ids: [1, seq_len], text_embeddings: [1, seq_len, hidden]
    // image_features: [sum(per_image_tokens), hidden]
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

    // Build one [1, 1, hidden] segment per input position: text segments come
    // from text_embeddings; sentinel segments consume the next feature row
    // (expanded from [1, hidden] with NewAxis so all segments concatenate
    // along axis 1).
    let mut segments: Vec<Array> = Vec::with_capacity(ids.len());
    let mut feat_idx = 0i32;
    for (i, id) in ids.iter().enumerate() {
        if *id == IMAGE_TOKEN_INDEX {
            segments.push(
                image_features
                    .index((feat_idx..feat_idx + 1, ..))
                    .index(NewAxis),
            );
            feat_idx += 1;
        } else {
            segments.push(text_embeddings.index((.., i as i32..i as i32 + 1, ..)));
        }
    }

    let seg_refs: Vec<&Array> = segments.iter().collect();
    mlx_rs::ops::concatenate_axis(&seg_refs, 1)
}

#[cfg(test)]
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::panic,
    clippy::unwrap_used
)]
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
            layout: ImageTokenLayout {
                start: None,
                end: None,
                pad: None,
            },
        };
        assert_eq!(batch.per_image_tokens, vec![1]);
        assert!(batch.layout.start.is_none());
    }

    #[test]
    fn merge_embeddings_two_images_in_one_sequence() {
        // input_ids: [text, SENTINEL, text, SENTINEL, text]
        let ids = Array::from_slice(&[1i32, IMAGE_TOKEN_INDEX, 2, IMAGE_TOKEN_INDEX, 3], &[1, 5]);
        // text_embeddings [1, 5, 4]: row j is all `j as f32`, distinct per position
        let text_data: Vec<f32> = (0..5)
            .flat_map(|j| std::iter::repeat(j as f32).take(4))
            .collect();
        let text_embeddings = Array::from_slice(&text_data, &[1, 5, 4]);
        // image features: 2 images, 1 token each, dim 4
        let features = Array::from_slice(&[9.0f32, 9.1, 9.2, 9.3, 8.0, 8.1, 8.2, 8.3], &[2, 4]);
        let batch = ImageBatch {
            pixel_values: Array::from_slice(&[0.0f32; 3], &[1, 1, 1, 3]),
            per_image_tokens: vec![1, 1],
            layout: ImageTokenLayout::default(),
        };
        let merged = merge_embeddings(&ids, &text_embeddings, &features, &batch).unwrap();
        assert_eq!(merged.shape(), &[1, 5, 4]);
        let flat = merged.as_slice::<f32>();
        let row = |i: usize| &flat[i * 4..(i + 1) * 4];
        // text positions keep their token embeddings; sentinels get feature rows
        assert_eq!(row(0), &[0.0, 0.0, 0.0, 0.0]); // text id 1
        assert_eq!(row(1), &[9.0, 9.1, 9.2, 9.3]); // features[0]
        assert_eq!(row(2), &[2.0, 2.0, 2.0, 2.0]); // text id 2
        assert_eq!(row(3), &[8.0, 8.1, 8.2, 8.3]); // features[1]
        assert_eq!(row(4), &[4.0, 4.0, 4.0, 4.0]); // text id 3
    }

    #[test]
    fn merge_embeddings_one_image_two_tokens() {
        // One image that expands to two feature rows: [text, SENTINEL, SENTINEL, text]
        let ids = Array::from_slice(&[5i32, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 6], &[1, 4]);
        let text_data: Vec<f32> = (0..4)
            .flat_map(|j| std::iter::repeat(j as f32).take(3))
            .collect();
        let text_embeddings = Array::from_slice(&text_data, &[1, 4, 3]);
        let features = Array::from_slice(&[7.0f32, 7.1, 7.2, 6.0, 6.1, 6.2], &[2, 3]);
        let batch = ImageBatch {
            pixel_values: Array::from_slice(&[0.0f32; 3], &[1, 1, 1, 3]),
            per_image_tokens: vec![2],
            layout: ImageTokenLayout::default(),
        };
        let merged = merge_embeddings(&ids, &text_embeddings, &features, &batch).unwrap();
        assert_eq!(merged.shape(), &[1, 4, 3]);
        let flat = merged.as_slice::<f32>();
        let row = |i: usize| &flat[i * 3..(i + 1) * 3];
        assert_eq!(row(0), &[0.0, 0.0, 0.0]);
        assert_eq!(row(1), &[7.0, 7.1, 7.2]); // features[0]
        assert_eq!(row(2), &[6.0, 6.1, 6.2]); // features[1]
        assert_eq!(row(3), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn merge_embeddings_rejects_sentinel_count_mismatch() {
        let ids = Array::from_slice(&[1i32, IMAGE_TOKEN_INDEX, 2], &[1, 3]);
        let text_data: Vec<f32> = (0..3)
            .flat_map(|j| std::iter::repeat(j as f32).take(2))
            .collect();
        let text_embeddings = Array::from_slice(&text_data, &[1, 3, 2]);
        let features = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let batch = ImageBatch {
            pixel_values: Array::from_slice(&[0.0f32; 3], &[1, 1, 1, 3]),
            per_image_tokens: vec![2], // expects 2 sentinels, ids has 1
            layout: ImageTokenLayout::default(),
        };
        let err = merge_embeddings(&ids, &text_embeddings, &features, &batch).unwrap_err();
        assert!(
            err.to_string()
                .contains("expected 2 image feature positions, found 1")
        );
    }

    #[test]
    fn merge_embeddings_without_sentinels_returns_text_embeddings() {
        let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let text_data: Vec<f32> = (0..3)
            .flat_map(|j| std::iter::repeat(j as f32).take(2))
            .collect();
        let text_embeddings = Array::from_slice(&text_data, &[1, 3, 2]);
        let empty_features = Array::from_slice(&[0.0f32; 0], &[0, 2]);
        let batch = ImageBatch {
            pixel_values: Array::from_slice(&[0.0f32; 3], &[1, 1, 1, 3]),
            per_image_tokens: vec![],
            layout: ImageTokenLayout::default(),
        };
        let merged = merge_embeddings(&ids, &text_embeddings, &empty_features, &batch).unwrap();
        assert_eq!(merged.shape(), &[1, 3, 2]);
        assert_eq!(merged.as_slice::<f32>(), text_embeddings.as_slice::<f32>());
    }
}
