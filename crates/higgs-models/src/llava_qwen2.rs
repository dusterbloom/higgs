//! LLaVA-Qwen2 vision-language model (nanoLLaVA architecture).
//!
//! Combines a `SigLIP` vision encoder with a Qwen2 language model through an
//! MLP projector. The vision encoder processes images into patch embeddings,
//! which are projected into the LLM's embedding space and concatenated with
//! text token embeddings at positions marked by `IMAGE_TOKEN_INDEX`.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    module::{Module, Param},
    nn,
    ops::indexing::IndexOp,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::AnyCache;
use crate::cache::KeyValueCache;
use crate::error::ModelError;
use crate::siglip::{
    SigLipVisionConfig, SigLipVisionModel, load_siglip_weights, preprocess_image_resized,
    preprocess_images_batch,
};
use crate::transformer;
use crate::vision::{
    ImageBatch, ImageDetail, ImageInput, ImageTokenLayout, ImageTokenLayoutKind,
    VisionCapabilities, VisionError, VisionModel,
};

/// Token ID used as a placeholder for image positions in the input sequence.
pub use crate::vision::IMAGE_TOKEN_INDEX;

/// Full LLaVA-Qwen2 config from config.json.
#[derive(Debug, Deserialize)]
pub struct LlavaQwen2Config {
    pub hidden_size: i32,
    pub mm_hidden_size: i32,
    pub mm_projector_type: String,
    pub num_hidden_layers: i32,
    pub vision_config: SigLipVisionConfig,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

pub use crate::quant_config::QuantizationSettings as QuantizationConfig;

// ---------------------------------------------------------------------------
// Multimodal Projector (MLP)
// ---------------------------------------------------------------------------

/// MLP projector mapping vision hidden states to the LLM's embedding space.
/// For `mlp2x_gelu`: Linear -> GELU -> Linear
pub struct MmProjector {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl MmProjector {
    fn new(vision_dim: i32, lm_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            linear_1: nn::LinearBuilder::new(vision_dim, lm_dim).build()?,
            linear_2: nn::LinearBuilder::new(lm_dim, lm_dim).build()?,
        })
    }

    fn forward(&mut self, input: &Array) -> Result<Array, Exception> {
        let hidden = self.linear_1.forward(input)?;
        let activated = nn::gelu(&hidden)?;
        self.linear_2.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// LLaVA-Qwen2 Model
// ---------------------------------------------------------------------------

/// Combined vision-language model.
pub struct LlavaQwen2Model {
    vision_tower: SigLipVisionModel,
    mm_projector: MmProjector,
    language_model: transformer::Model,
    image_size: i32,
}

impl LlavaQwen2Model {
    /// Get the hidden size of the language model.
    pub const fn hidden_size(&self) -> i32 {
        self.language_model.args.hidden_size
    }

    /// Number of transformer layers.
    pub const fn num_hidden_layers(&self) -> i32 {
        self.language_model.args.num_hidden_layers
    }

    /// Number of KV heads used by the language model.
    pub(crate) const fn num_key_value_heads(&self) -> i32 {
        self.language_model.args.num_key_value_heads
    }

    /// Attention head dimension used by the language model KV cache.
    pub(crate) fn head_dim(&self) -> Result<i32, crate::error::ModelError> {
        self.language_model.args.checked_head_dim()
    }

    /// Get the image size expected by the vision encoder.
    pub const fn image_size(&self) -> i32 {
        self.image_size
    }

    /// Forward pass for text-only (no image).
    pub fn forward_text<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        self.language_model.forward(inputs, mask, cache)
    }

    /// Forward pass for text-only, returning hidden states.
    pub fn forward_text_hidden<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        self.language_model.forward_hidden(inputs, mask, cache)
    }

    /// Forward pass for text-only, returning logits for every input position.
    pub fn forward_text_all_logits<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        self.language_model.forward_all_logits(inputs, mask, cache)
    }

    /// Encode an image through the vision tower and projector.
    ///
    /// Input: `pixel_values` with shape `[1, H, W, 3]` (NHWC).
    /// Output: projected features `[1, num_patches, lm_hidden_size]`.
    pub fn encode_image(&mut self, pixel_values: &Array) -> Result<Array, Exception> {
        let hidden_states = self.vision_tower.forward_with_hidden_states(pixel_values)?;

        // nanoLLaVA uses the second-to-last layer output
        let num_states = hidden_states.len();
        let vision_features = hidden_states
            .get(num_states.saturating_sub(2))
            .or_else(|| hidden_states.last())
            .ok_or_else(|| Exception::custom("empty hidden states from vision encoder"))?;

        self.mm_projector.forward(vision_features)
    }

    /// Encode N images through the vision tower and projector.
    ///
    /// Input: `pixel_values` with shape `[N, H, W, 3]` (NHWC).
    /// Output: projected features `[sum(per_image_tokens), hidden]` — for
    /// `LLaVA` each image expands to `num_patches` rows, so
    /// `[N * num_patches, hidden]` with one row per patch.
    pub fn encode_image_batch(&mut self, pixel_values: &Array) -> Result<Array, Exception> {
        let n = pixel_values.shape().first().copied().unwrap_or(0);
        if n <= 1 {
            // Single-image path (kept for exactness with existing behavior).
            let feats = self.encode_image(pixel_values)?; // [1, num_patches, hidden]
            return Ok(feats.index(0));
        }
        let mut rows = Vec::new();
        for i in 0..n {
            let single = pixel_values.index((i..i + 1, .., .., ..));
            let feats = self.encode_image(&single)?; // [1, num_patches, hidden]
            rows.push(feats.index(0)); // [num_patches, hidden]
        }
        let refs: Vec<&Array> = rows.iter().collect();
        mlx_rs::ops::concatenate_axis(&refs, 0)
    }
}

/// Resolve the shared `LLaVA` preprocessing target from the request's detail tiers.
///
/// `LLaVA` is a fixed square processor, so every image in the batch is resized
/// to the same target. The highest requested tier wins: only when *every*
/// image is `Low` do we downscale to half the encoder's native size (floored
/// at 128px); `Auto`/`High` (and empty inputs) use `image_size` as-is.
#[allow(clippy::as_conversions, clippy::cast_sign_loss)]
fn llava_target_size(image_size: i32, details: &[ImageDetail]) -> u32 {
    let all_low = !details.is_empty() && details.iter().all(|d| *d == ImageDetail::Low);
    let size = if all_low {
        (image_size / 2).max(128)
    } else {
        image_size
    };
    size as u32
}

/// Expand `<image>` marker ids into `num_patches` consecutive sentinels.
///
/// [`merge_embeddings`](crate::vision::merge_embeddings) requires exactly
/// `sum(batch.per_image_tokens)` sentinel positions; `LLaVA` expands every
/// `<image>` marker into `num_patches` (one per vision patch) so a batch of N
/// images yields `N * num_patches` feature positions. Tokens that are not the
/// marker (including other special tokens) pass through unchanged.
fn expand_image_markers(tokens: &mut Vec<u32>, marker_id: u32, num_patches: usize, sentinel: u32) {
    if num_patches == 0 {
        return;
    }
    let marker_count = tokens.iter().filter(|&&t| t == marker_id).count();
    if marker_count == 0 {
        return;
    }
    tokens.reserve(marker_count * (num_patches - 1));
    let original = std::mem::take(tokens);
    for t in original {
        if t == marker_id {
            tokens.extend(std::iter::repeat_n(sentinel, num_patches));
        } else {
            tokens.push(t);
        }
    }
}

impl VisionModel for LlavaQwen2Model {
    fn vision_capabilities(&self) -> VisionCapabilities {
        VisionCapabilities {
            families: vec!["llava-qwen2"],
            image_sizes: vec![self.image_size],
            supported_media: vec![
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/gif",
                "image/bmp",
            ],
            layout_kind: ImageTokenLayoutKind::Sentinel,
        }
    }

    fn image_marker_text(&self) -> &'static str {
        "<image>"
    }

    fn preprocess_images(&self, images: &[ImageInput]) -> Result<ImageBatch, VisionError> {
        let details: Vec<ImageDetail> = images.iter().map(|i| i.detail).collect();
        let target = llava_target_size(self.image_size, &details);

        let bytes: Vec<&[u8]> = images.iter().map(|i| i.bytes.as_slice()).collect();
        let pixel_values = match preprocess_images_batch(&bytes, target) {
            Ok(pixel_values) => pixel_values,
            Err(batch_err) => {
                // `preprocess_images_batch` collapses every failure into
                // `Preprocess`; re-run the offending image through the decode
                // path so malformed bytes surface as `VisionError::Decode`
                // rather than a server-side preprocessing error.
                for img in images {
                    preprocess_image_resized(
                        &img.bytes,
                        (target, target),
                        image::imageops::FilterType::Lanczos3,
                    )?;
                }
                return Err(batch_err);
            }
        };
        let num_patches = usize::try_from(self.vision_tower.num_patches())
            .map_err(|e| VisionError::Preprocess(format!("invalid vision num_patches: {e}")))?;
        Ok(ImageBatch {
            pixel_values,
            // Each image expands to `num_patches` feature rows in the merge.
            per_image_tokens: vec![num_patches; images.len()],
            // LLaVA resizes every image to the same square target, so the
            // batch canvas is unpadded and sizes are uniform.
            image_sizes: vec![(target, target); images.len()],
            layout: ImageTokenLayout::default(),
        })
    }

    #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
    fn postprocess_image_tokens(
        &self,
        tokens: &mut Vec<u32>,
        tokenizer: &Tokenizer,
        _batch: &ImageBatch,
    ) -> Result<(), VisionError> {
        let Some(marker_id) = tokenizer.token_to_id("<image>") else {
            return Ok(()); // tokenizer without <image>: nothing to expand
        };
        let num_patches = usize::try_from(self.vision_tower.num_patches())
            .map_err(|e| VisionError::Preprocess(format!("invalid vision num_patches: {e}")))?;
        // Each <image> marker becomes `num_patches` consecutive sentinels so
        // the sentinel count matches `sum(batch.per_image_tokens)` required by
        // `merge_embeddings` (N images -> N * num_patches feature rows).
        expand_image_markers(tokens, marker_id, num_patches, IMAGE_TOKEN_INDEX as u32);
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
        // Validate batch=1 assumption.
        let batch_size = input_ids.shape().first().copied().unwrap_or(0);
        if batch_size != 1 {
            return Err(Exception::custom(format!(
                "LLaVA-Qwen2 only supports batch_size=1, got {batch_size}"
            )));
        }
        let image_features = self.encode_image_batch(&batch.pixel_values)?; // [sum(per_image_tokens), hidden]
        // Replace IMAGE_TOKEN_INDEX sentinel with 0 before embedding lookup to
        // avoid out-of-bounds access. merge_embeddings overwrites these positions.
        let sentinel = Array::from_slice(&[IMAGE_TOKEN_INDEX], &[1]);
        let is_sentinel = input_ids.eq(&sentinel)?;
        let zero = Array::from_slice(&[0_i32], &[1]);
        let safe_ids = mlx_rs::ops::r#where(&is_sentinel, &zero, input_ids)?;
        let text_embeddings = self.language_model.embed_tokens(&safe_ids)?;
        let combined =
            crate::vision::merge_embeddings(input_ids, &text_embeddings, &image_features, batch)?;
        self.language_model
            .forward_from_embeddings(&combined, None, c)
    }
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

/// Load a LLaVA-Qwen2 model from a directory.
pub fn load_llava_qwen2_model(model_dir: &Path) -> Result<LlavaQwen2Model, ModelError> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let raw: serde_json::Value = serde_json::from_str(&config_str)?;
    load_llava_qwen2_model_from_value(model_dir, &raw)
}

pub(crate) fn load_llava_qwen2_model_from_value(
    model_dir: &Path,
    raw: &serde_json::Value,
) -> Result<LlavaQwen2Model, ModelError> {
    let config: LlavaQwen2Config = serde_json::from_value(raw.clone())?;
    load_llava_qwen2_model_with_config(model_dir, &config, raw)
}

fn load_llava_qwen2_model_with_config(
    model_dir: &Path,
    config: &LlavaQwen2Config,
    raw: &serde_json::Value,
) -> Result<LlavaQwen2Model, ModelError> {
    tracing::info!(
        image_size = config.vision_config.image_size,
        vision_layers = config.vision_config.num_hidden_layers,
        vision_hidden = config.vision_config.hidden_size,
        lm_hidden = config.hidden_size,
        lm_layers = config.num_hidden_layers,
        projector = %config.mm_projector_type,
        "Loading LLaVA-Qwen2 model"
    );

    // Build vision encoder
    let mut vision_tower = SigLipVisionModel::new(&config.vision_config)?;

    // Build projector
    let mut mm_projector = MmProjector::new(config.mm_hidden_size, config.hidden_size)?;

    // Build language model (reads text_config, strips language_model. prefix)
    let language_args = transformer::text_model_args_from_value(raw)?;
    let language_model = transformer::load_vlm_language_model_with_args(model_dir, language_args)?;

    // Load all safetensor weights for vision and projector
    let weights = load_safetensor_weights(model_dir)?;

    let vision_prefix = "vision_tower.vision_tower.vision_model.";
    load_siglip_weights(&mut vision_tower, &weights, vision_prefix)?;
    load_projector_weights(&mut mm_projector, &weights)?;

    let image_size = config.vision_config.image_size;

    tracing::info!("LLaVA-Qwen2 model loaded successfully");

    Ok(LlavaQwen2Model {
        vision_tower,
        mm_projector,
        language_model,
        image_size,
    })
}

fn load_projector_weights(
    projector: &mut MmProjector,
    weights: &HashMap<String, Array>,
) -> Result<(), ModelError> {
    let get = |name: &str| -> Result<Array, ModelError> {
        weights
            .get(name)
            .cloned()
            .ok_or_else(|| ModelError::MissingWeight(format!("Missing projector weight: {name}")))
    };

    projector.linear_1.weight = Param::new(get("mm_projector.linear_1.weight")?);
    projector.linear_1.bias = Param::new(Some(get("mm_projector.linear_1.bias")?));
    projector.linear_2.weight = Param::new(get("mm_projector.linear_2.weight")?);
    projector.linear_2.bias = Param::new(Some(get("mm_projector.linear_2.bias")?));

    Ok(())
}

/// Load all safetensor weights from a model directory into a `HashMap`.
fn load_safetensor_weights(model_dir: &Path) -> Result<HashMap<String, Array>, ModelError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");

    let files: Vec<std::path::PathBuf> = if index_path.exists() {
        let index_str = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_str)?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| ModelError::MissingWeight("Missing weight_map in index".to_owned()))?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_files.sort();
        shard_files.dedup();
        shard_files.into_iter().map(|f| model_dir.join(f)).collect()
    } else if single_path.exists() {
        vec![single_path]
    } else {
        return Err(ModelError::MissingWeight(
            "No safetensors file found".to_owned(),
        ));
    };

    let mut all_weights = HashMap::new();
    for path in &files {
        let loaded = Array::load_safetensors(path)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;
        all_weights.extend(loaded);
    }
    Ok(all_weights)
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llava_target_size_uses_native_size_unless_all_low() {
        assert_eq!(llava_target_size(384, &[]), 384);
        assert_eq!(llava_target_size(384, &[ImageDetail::Auto]), 384);
        assert_eq!(llava_target_size(384, &[ImageDetail::High]), 384);
        // Highest requested tier wins: a single Auto/High forces full size.
        assert_eq!(
            llava_target_size(384, &[ImageDetail::Auto, ImageDetail::Low]),
            384
        );
        assert_eq!(
            llava_target_size(384, &[ImageDetail::High, ImageDetail::Low]),
            384
        );
    }

    #[test]
    fn llava_target_size_low_downscales_to_half() {
        assert_eq!(llava_target_size(384, &[ImageDetail::Low]), 192);
        assert_eq!(
            llava_target_size(384, &[ImageDetail::Low, ImageDetail::Low]),
            192
        );
    }

    #[test]
    fn llava_target_size_low_is_floored_at_128() {
        assert_eq!(llava_target_size(64, &[ImageDetail::Low]), 128);
        assert_eq!(llava_target_size(0, &[ImageDetail::Low]), 128);
    }

    #[test]
    fn expand_image_markers_single_marker_becomes_num_patches_sentinels() {
        // One <image> marker (id 99) must expand to `num_patches` consecutive
        // sentinels so the count matches `sum(per_image_tokens)` in merge.
        let mut tokens = vec![1, 2, 99, 3];
        expand_image_markers(&mut tokens, 99, 4, IMAGE_TOKEN_INDEX as u32);
        let s = IMAGE_TOKEN_INDEX as u32;
        assert_eq!(tokens, vec![1, 2, s, s, s, s, 3]);
    }

    #[test]
    fn expand_image_markers_two_markers_expand_in_order() {
        let mut tokens = vec![1, 99, 2, 99, 3];
        expand_image_markers(&mut tokens, 99, 3, IMAGE_TOKEN_INDEX as u32);
        let s = IMAGE_TOKEN_INDEX as u32;
        assert_eq!(tokens, vec![1, s, s, s, 2, s, s, s, 3]);
        assert_eq!(tokens.len(), 9);
    }

    #[test]
    fn expand_image_markers_without_markers_is_unchanged() {
        let mut tokens = vec![1, 2, 3];
        expand_image_markers(&mut tokens, 99, 4, IMAGE_TOKEN_INDEX as u32);
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn expand_image_markers_uses_image_token_index_sentinel() {
        let mut tokens = vec![99];
        expand_image_markers(&mut tokens, 99, 2, IMAGE_TOKEN_INDEX as u32);
        assert_eq!(tokens, vec![IMAGE_TOKEN_INDEX as u32; 2]);
        assert_eq!(tokens[0], IMAGE_TOKEN_INDEX as u32);
        assert_eq!(tokens[0] as i32, IMAGE_TOKEN_INDEX);
    }
}
