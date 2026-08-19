//! Vision-language model support: shared trait, types, and preprocessing.
//!
//! Every VLM family implements [`VisionModel`]. The route layer produces
//! position-preserving [`ImageInput`]s from `OpenAI` parts or `Anthropic` blocks;
//! the engine renders family marker tokens, post-processes them into sentinel
//! runs, and merges image features into the text embedding sequence before the
//! transformer runs (see [`merge_embeddings`]).

use mlx_rs::{Array, error::Exception};
use tokenizers::Tokenizer;

use crate::AnyCache;

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
            layout: ImageTokenLayout {
                start: None,
                end: None,
                pad: None,
            },
        };
        assert_eq!(batch.per_image_tokens, vec![1]);
        assert!(batch.layout.start.is_none());
    }
}
