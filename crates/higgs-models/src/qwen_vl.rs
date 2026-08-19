//! Qwen-VL vision-language model (Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL).
//!
//! Combines a `SigLIP`-shaped vision encoder with a `Qwen3Next` language model
//! through a linear projector. Images are preprocessed with the Qwen-VL
//! **dynamic-resolution** scheme: `smart_resize` fits the image into the
//! `[min_pixels, max_pixels]` budget, then the image is split into patch grid
//! whose 2×2 blocks are merged (`spatial_merge_size`) so each image expands to
//! `(grid_h × grid_w) / merge²` embedding rows.
//!
//! Prompt layout: `<|vision_start|><|image_pad|><|vision_end|>` — the pad is
//! expanded post-tokenization into `per_image_tokens[i]` consecutive
//! [`IMAGE_TOKEN_INDEX`] sentinels which `forward_multimodal` replaces with
//! projected image features before the transformer runs.

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
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::AnyCache;
use crate::AnyModel;
use crate::error::ModelError;
use crate::qwen3_next::{LayerCache, Qwen3NextCausalLM};
use crate::siglip::{SigLipVisionConfig, SigLipVisionModel, load_siglip_weights};
use crate::vision::{
    IMAGE_TOKEN_INDEX, ImageBatch, ImageInput, ImageTokenLayout, ImageTokenLayoutKind,
    VisionCapabilities, VisionError, VisionModel,
};

/// Image-size trait methods (`dimensions`) used in preprocessing.
use image::GenericImageView;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Parsed Qwen-VL wrapper config from config.json.
#[derive(Debug, Clone)]
pub struct QwenVlConfig {
    /// Vision tower config, parsed with Qwen-VL field aliases
    /// (`depth`/`num_heads`/`in_chans`/`in_channels`/`spatial_patch_size`).
    pub vision_config: SigLipVisionConfig,
    /// Raw nested `text_config` (kept for the qwen3.5 text-args loader).
    pub text_config: Value,
    /// Dynamic-resolution pixel budget floor.
    pub min_pixels: i32,
    /// Dynamic-resolution pixel budget ceiling.
    pub max_pixels: i32,
    /// Spatial merge size (`spatial_merge_size`, typically 2).
    pub merge_size: i32,
    /// Projector input width (`vision_hidden × merge²`, or `out_hidden_size`).
    pub mm_hidden_size: i32,
    /// Vision patch size.
    pub patch_size: i32,
}

impl QwenVlConfig {
    /// Parse a Qwen-VL wrapper config, tolerating the field-name drift between
    /// the `qwen2_5_vl` / `qwen3_vl` / `qwen3_5_vl` vision configs.
    pub(crate) fn from_value(raw: &Value) -> Result<Self, ModelError> {
        let vision = raw.get("vision_config").ok_or_else(|| {
            ModelError::UnsupportedModel("missing vision_config in Qwen-VL config.json".into())
        })?;
        // Some Qwen-VL wrappers (e.g. `qwen2_5_vl`) keep the text backbone
        // config at the top level; mirror `qwen35_args` and fall back to the
        // whole raw config when `text_config` is absent.
        let text_config = raw
            .get("text_config")
            .cloned()
            .unwrap_or_else(|| raw.clone());
        let vision_config = parse_vision_config(vision)?;
        let merge_size = get_i32(vision, &["spatial_merge_size"]).unwrap_or(2);
        let patch_size = vision_config.patch_size;
        // min/max pixel budgets come from the wrapper config when present;
        // the Qwen2.5-VL reference defaults are 256*28*28 / 1280*28*28.
        let min_pixels = get_i32(raw, &["min_pixels"]).unwrap_or(256 * 28 * 28);
        let max_pixels = get_i32(raw, &["max_pixels"]).unwrap_or(1280 * 28 * 28);
        let mm_hidden_size = get_i32(raw, &["mm_hidden_size"])
            .or_else(|| get_i32(vision, &["out_hidden_size"]))
            .unwrap_or_else(|| vision_config.hidden_size * merge_size * merge_size);
        Ok(Self {
            vision_config,
            text_config,
            min_pixels,
            max_pixels,
            merge_size,
            mm_hidden_size,
            patch_size,
        })
    }
}

/// Parse a Qwen-VL `vision_config` into [`SigLipVisionConfig`].
///
/// Qwen-VL vision configs use `depth` instead of `num_hidden_layers`,
/// `num_heads` instead of `num_attention_heads`, and `in_chans`/`in_channels`
/// instead of `num_channels`; `image_size` may be absent (dynamic resolution)
/// in which case a nominal value (`patch_size × 32`) sizes the tower's
/// positional-embedding table.
fn parse_vision_config(vision: &Value) -> Result<SigLipVisionConfig, ModelError> {
    let get = |keys: &[&str]| -> Result<i32, ModelError> {
        get_i32(vision, keys).ok_or_else(|| {
            ModelError::UnsupportedModel(format!("Qwen-VL vision_config missing one of {keys:?}"))
        })
    };
    let hidden_size = get(&["hidden_size", "embed_dim"])?;
    let patch_size = get(&["patch_size", "spatial_patch_size"])?;
    let nominal_image_size = get_i32(vision, &["image_size"]).unwrap_or(patch_size * 32);
    Ok(SigLipVisionConfig {
        hidden_size,
        intermediate_size: get(&["intermediate_size"])?,
        num_hidden_layers: get(&["num_hidden_layers", "depth"])?,
        num_attention_heads: get(&["num_attention_heads", "num_heads"])?,
        num_channels: get(&["num_channels", "in_channels", "in_chans"])?,
        patch_size,
        image_size: nominal_image_size,
        layer_norm_eps: get_f32(vision, &["layer_norm_eps"]).unwrap_or(1e-6),
        hidden_act: vision
            .get("hidden_act")
            .and_then(Value::as_str)
            .unwrap_or("gelu_pytorch_tanh")
            .to_owned(),
    })
}

fn get_i32(value: &Value, keys: &[&str]) -> Option<i32> {
    keys.iter().find_map(|k| {
        value
            .get(k)
            .and_then(Value::as_i64)
            .and_then(|v| i32::try_from(v).ok())
    })
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn get_f32(value: &Value, keys: &[&str]) -> Option<f32> {
    keys.iter()
        .find_map(|k| value.get(k).and_then(Value::as_f64).map(|v| v as f32))
}

// ---------------------------------------------------------------------------
// Dynamic-resolution preprocessing (pure helpers)
// ---------------------------------------------------------------------------

/// Round to the nearest whole number, ties to even — Python's `round()`, which
/// the Qwen-VL reference `smart_resize` relies on (banker's rounding).
fn round_half_to_even(x: f64) -> f64 {
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 {
        floor + 1.0
    } else if frac < 0.5 || floor % 2.0 == 0.0 {
        floor
    } else {
        floor + 1.0
    }
}

/// The Qwen-VL dynamic-resolution algorithm (default patch grid factor 28).
///
/// Thin wrapper over [`smart_resize_with_factor`] using the Qwen2.5-VL default
/// `factor = patch_size × merge_size = 28`.
pub fn smart_resize(
    height: u32,
    width: u32,
    min_pixels: i32,
    max_pixels: i32,
) -> Result<(u32, u32), VisionError> {
    smart_resize_with_factor(height, width, 28, min_pixels, max_pixels)
}

/// Port of the Qwen2-VL reference `smart_resize` (transformers
/// `image_processing_qwen2_vl.py`, mirrored by mlx-vlm): round both dimensions
/// to multiples of `factor`, then shrink (floor) or grow (ceil) to fit the
/// `[min_pixels, max_pixels]` budget while preserving aspect ratio.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn smart_resize_with_factor(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: i32,
    max_pixels: i32,
) -> Result<(u32, u32), VisionError> {
    let min_px = u64::try_from(min_pixels.max(1)).unwrap_or(u64::MAX);
    let max_px = u64::try_from(max_pixels.max(1)).unwrap_or(u64::MAX);
    let factor_u64 = u64::from(factor.max(1));
    let h_u64 = u64::from(height);
    let w_u64 = u64::from(width);
    if h_u64 == 0 || w_u64 == 0 {
        return Err(VisionError::Preprocess(
            "image has a zero dimension".to_owned(),
        ));
    }
    // Absolute aspect-ratio guard from the reference (> 200 is rejected).
    let ratio = if h_u64 > w_u64 {
        h_u64 as f64 / w_u64 as f64
    } else {
        w_u64 as f64 / h_u64 as f64
    };
    if ratio > 200.0 {
        return Err(VisionError::Preprocess(format!(
            "absolute aspect ratio must be smaller than 200, got {ratio}"
        )));
    }

    let h_rounded = (round_half_to_even(h_u64 as f64 / factor_u64 as f64) as u64) * factor_u64;
    let w_rounded = (round_half_to_even(w_u64 as f64 / factor_u64 as f64) as u64) * factor_u64;
    let (h_result, w_result) = if h_rounded * w_rounded > max_px {
        let beta = ((h_u64 * w_u64) as f64 / max_px as f64).sqrt();
        let hb =
            ((h_u64 as f64 / beta / factor_u64 as f64).floor() as u64 * factor_u64).max(factor_u64);
        let wb =
            ((w_u64 as f64 / beta / factor_u64 as f64).floor() as u64 * factor_u64).max(factor_u64);
        (hb, wb)
    } else if h_rounded * w_rounded < min_px {
        let beta = (min_px as f64 / (h_u64 * w_u64) as f64).sqrt();
        let hb = (h_u64 as f64 * beta / factor_u64 as f64).ceil() as u64 * factor_u64;
        let wb = (w_u64 as f64 * beta / factor_u64 as f64).ceil() as u64 * factor_u64;
        (hb, wb)
    } else {
        (h_rounded, w_rounded)
    };
    let h_out = u32::try_from(h_result)
        .map_err(|_| VisionError::Preprocess("resized height overflow".into()))?;
    let w_out = u32::try_from(w_result)
        .map_err(|_| VisionError::Preprocess("resized width overflow".into()))?;
    Ok((h_out, w_out))
}

/// Patch grid after the 2×2 spatial merge.
///
/// For a `patch × merge`-aligned image this equals
/// `(h / (patch * merge), w / (patch * merge))` — the number of embedding rows
/// per image dimension produced by the tower after merging.
pub fn get_grid(h: u32, w: u32, patch: u32, merge: u32) -> (u32, u32) {
    let gh = (h / patch).max(1);
    let gw = (w / patch).max(1);
    (gh.div_ceil(merge).max(1), gw.div_ceil(merge).max(1))
}

/// Merged feature rows an image of `height × width` yields after the tower and
/// the 2×2 spatial merge — the per-image token budget.
///
/// This is the pure per-image grid computation shared by preprocessing (token
/// budget) and encoding (row-count assertion), so a mixed-resolution batch is
/// always budgeted from each image's **own** dims.
fn image_feature_rows(height: u32, width: u32, patch: u32, merge: u32) -> usize {
    let (gh, gw) = get_grid(height, width, patch, merge);
    usize::try_from(gh * gw).unwrap_or(usize::MAX)
}

/// Reference Qwen-VL pixel-shuffle (transformers `image_processing_qwen2_vl.py`).
///
/// Rearranges a channels-first image `[C, H, W]` (H/W multiples of
/// `patch × merge`) into `[grid_h × grid_w, C × patch × patch]` patch rows so a
/// `patch × merge`-sized block becomes one row with its 2×2 sub-patches
/// interleaved — the exact transpose `(0,3,6,4,7,2,1,5,8)` of the reference
/// (still-image `temporal_patch_size = 1` simplification; the reference
/// duplicates the frame along the temporal axis, which is content-neutral for
/// the flattened rows).
pub fn pixel_shuffle(image: &Array, patch: u32, merge: u32) -> Result<Array, Exception> {
    let &[c, h, w] = image.shape() else {
        return Err(Exception::custom("pixel_shuffle: expected [C, H, W] input"));
    };
    let patch_i32 = i32::try_from(patch).map_err(|_| Exception::custom("patch size overflow"))?;
    let merge_i32 = i32::try_from(merge).map_err(|_| Exception::custom("merge size overflow"))?;
    if h % (patch_i32 * merge_i32) != 0 || w % (patch_i32 * merge_i32) != 0 {
        return Err(Exception::custom(format!(
            "pixel_shuffle: dims [{h}, {w}] must be multiples of patch*merge={}",
            patch_i32 * merge_i32
        )));
    }
    let gh = h / patch_i32;
    let gw = w / patch_i32;
    let reshaped = image.reshape(&[
        1,
        1,
        c,
        gh / merge_i32,
        merge_i32,
        patch_i32,
        gw / merge_i32,
        merge_i32,
        patch_i32,
    ])?;
    let transposed = reshaped.transpose_axes(&[0, 3, 6, 4, 7, 2, 1, 5, 8])?;
    transposed.reshape(&[gh * gw, c * patch_i32 * patch_i32])
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

/// Qwen2.5-VL normalization statistics (CLIP-style, from the reference
/// `preprocessor_config.json`; Qwen3-VL ships [0.5, 0.5, 0.5] instead).
const QWEN_VL_MEAN: [f32; 3] = [0.481_454_7, 0.457_827_5, 0.408_210_72];
const QWEN_VL_STD: [f32; 3] = [0.268_629_55, 0.261_302_6, 0.275_777_1];

/// Decode, resize to an exact target, and apply Qwen-VL normalization.
/// Returns `[1, H, W, 3]` NHWC (channel-last for MLX).
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::indexing_slicing
)]
fn preprocess_qwen_vl_image(image_bytes: &[u8], target: (u32, u32)) -> Result<Array, VisionError> {
    let img =
        image::load_from_memory(image_bytes).map_err(|e| VisionError::Decode(e.to_string()))?;
    // PIL BICUBIC ≈ image-rs CatmullRom.
    let resized = img.resize_exact(target.0, target.1, image::imageops::FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    let (w, h) = rgb.dimensions();
    let pixels = rgb.into_raw();
    let mut floats: Vec<f32> = Vec::with_capacity(pixels.len());
    for (i, &p) in pixels.iter().enumerate() {
        let ch = i % 3;
        floats.push((f32::from(p) / 255.0 - QWEN_VL_MEAN[ch]) / QWEN_VL_STD[ch]);
    }
    Ok(Array::from_slice(&floats, &[1, h as i32, w as i32, 3]))
}

/// Stack per-image `[1, H, W, 3]` arrays into one `[N, max_h, max_w, 3]` array,
/// zero-padding each image to the batch maxima (dynamic resolutions differ).
#[allow(clippy::indexing_slicing)]
fn stack_padded(images: &[Array], max_h: u32, max_w: u32) -> Result<Array, VisionError> {
    if images.is_empty() {
        return Ok(Array::from_slice::<f32>(&[], &[0, 1, 1, 3]));
    }
    let (max_h_usize, max_w_usize) = (
        usize::try_from(max_h).map_err(|_| VisionError::Preprocess("height overflow".into()))?,
        usize::try_from(max_w).map_err(|_| VisionError::Preprocess("width overflow".into()))?,
    );
    let mut data: Vec<f32> = vec![0.0; images.len() * max_h_usize * max_w_usize * 3];
    for (n, img) in images.iter().enumerate() {
        let &[_, h_i32, w_i32, 3] = img.shape() else {
            return Err(VisionError::Preprocess(
                "internal: expected [1, H, W, 3] image".into(),
            ));
        };
        let (h_usize, w_usize) = (
            usize::try_from(h_i32)
                .map_err(|_| VisionError::Preprocess("height overflow".into()))?,
            usize::try_from(w_i32).map_err(|_| VisionError::Preprocess("width overflow".into()))?,
        );
        let src = img.as_slice::<f32>();
        let base = n * max_h_usize * max_w_usize * 3;
        for r in 0..h_usize {
            let src_row = r * w_usize * 3;
            let dst_row = base + r * max_w_usize * 3;
            data[dst_row..dst_row + w_usize * 3]
                .copy_from_slice(&src[src_row..src_row + w_usize * 3]);
        }
    }
    let shape = [
        i32::try_from(images.len())
            .map_err(|_| VisionError::Preprocess("batch overflow".into()))?,
        i32::try_from(max_h_usize)
            .map_err(|_| VisionError::Preprocess("height overflow".into()))?,
        i32::try_from(max_w_usize).map_err(|_| VisionError::Preprocess("width overflow".into()))?,
        3,
    ];
    Ok(Array::from_slice(&data, &shape))
}

// ---------------------------------------------------------------------------
// Marker expansion
// ---------------------------------------------------------------------------

/// Expand `<|vision_start|>`…`<|image_pad|>`… runs into the exact sentinel run.
///
/// The route renders `<|vision_start|><|image_pad|><|vision_end|>` per image;
/// post-tokenization the single pad is replaced by `per_image_tokens[i]`
/// consecutive [`IMAGE_TOKEN_INDEX`] sentinels while start/end stay as regular
/// token embeddings — mirroring how the reference processor expands one
/// `<|image_pad|>` into `grid_thw.prod() // merge_size²` pads.
fn expand_qwen_vl_markers(
    tokens: &mut Vec<u32>,
    start: u32,
    pad: u32,
    per_image_tokens: &[usize],
    sentinel: u32,
) -> Result<(), VisionError> {
    let mut out = Vec::with_capacity(tokens.len() + per_image_tokens.iter().sum::<usize>());
    let mut img_idx = 0usize;
    let mut iter = tokens.iter().copied().peekable();
    while let Some(t) = iter.next() {
        if t == start {
            let k = per_image_tokens.get(img_idx).copied().ok_or_else(|| {
                VisionError::Preprocess(format!(
                    "more image markers in the prompt than images in the batch (marker {img_idx})"
                ))
            })?;
            img_idx += 1;
            // Consume every <|image_pad|> in the marker run.
            while iter.peek() == Some(&pad) {
                iter.next();
            }
            // <|vision_start|> stays as a real token embedding; the pad
            // position becomes `k` consecutive feature sentinels.
            out.push(t);
            out.extend(std::iter::repeat_n(sentinel, k));
            continue;
        }
        out.push(t);
    }
    if img_idx != per_image_tokens.len() {
        return Err(VisionError::Preprocess(format!(
            "expected {img_idx} image markers in the prompt, batch has {} images",
            per_image_tokens.len()
        )));
    }
    *tokens = out;
    Ok(())
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Qwen-VL model: `SigLIP`-shaped vision tower + linear projector on a
/// `Qwen3Next` text backbone.
pub struct QwenVlModel {
    vision_tower: SigLipVisionModel,
    mm_projector: nn::Linear,
    language_model: Qwen3NextCausalLM,
    config: QwenVlConfig,
}

impl QwenVlModel {
    /// Forward pass for text-only input (no images).
    pub fn forward_text(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        self.language_model.forward(inputs, mask, cache)
    }

    /// Forward pass for text-only input, returning hidden states.
    pub fn forward_text_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        self.language_model.forward_hidden(inputs, mask, cache)
    }

    /// Forward pass for text-only input, returning logits for every position.
    pub fn forward_text_all_logits(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let (_, logits) = self
            .language_model
            .forward_with_hidden(inputs, mask, cache)?;
        Ok(logits)
    }

    /// Batched decode over the `Qwen3Next` backbone (one token per request).
    ///
    /// Delegates to [`Qwen3NextCausalLM::forward_batched`] over the hybrid
    /// SSM/attention stack; the engine calls this through
    /// `AnyModel::forward_batched` with `AnyCache::Hybrid` caches.
    pub fn forward_text_batched(
        &mut self,
        inputs: &Array,
        kv_caches: &mut [&mut Vec<Option<LayerCache>>],
    ) -> Result<Array, Exception> {
        self.language_model.forward_batched(inputs, kv_caches)
    }

    /// Language-model hidden size.
    pub const fn hidden_size(&self) -> i32 {
        self.language_model.args.hidden_size
    }

    /// Language-model layer count.
    pub const fn num_hidden_layers(&self) -> i32 {
        self.language_model.args.num_hidden_layers
    }

    /// Language-model KV head count.
    pub const fn num_key_value_heads(&self) -> i32 {
        self.language_model.args.num_key_value_heads
    }

    /// Language-model attention head dimension.
    pub const fn head_dim(&self) -> i32 {
        self.language_model.args.head_dim
    }

    /// Fresh per-layer hybrid (KV + SSM) cache for the backbone.
    pub fn make_lm_cache(&self) -> Vec<Option<LayerCache>> {
        self.language_model.make_cache()
    }

    /// Fresh per-layer hybrid cache with `TurboQuant` on full-attention layers.
    pub fn make_lm_cache_turbo(
        &self,
        config: crate::turboquant::KvCacheConfig,
    ) -> Result<Vec<Option<LayerCache>>, Exception> {
        self.language_model.make_cache_turbo(config)
    }

    /// Encode every image in the batch through the tower + 2×2 merge +
    /// projector, returning `[sum(per_image_tokens), lm_hidden]` feature rows.
    fn encode_all_images(&mut self, batch: &ImageBatch) -> Result<Array, Exception> {
        let n =
            usize::try_from(batch.pixel_values.shape().first().copied().unwrap_or(0)).unwrap_or(0);
        if batch.image_sizes.len() != n {
            return Err(Exception::custom(format!(
                "Qwen-VL: batch has {n} pixel canvases but {} image_sizes entries",
                batch.image_sizes.len()
            )));
        }
        let patch = self.config.patch_size;
        let merge = self.config.merge_size;
        let mut rows: Vec<Array> = Vec::with_capacity(n);
        for (i, &(image_h, image_w)) in batch.image_sizes.iter().enumerate() {
            let (image_h_i32, image_w_i32) = (
                i32::try_from(image_h)
                    .map_err(|_| Exception::custom("Qwen-VL: image height overflow"))?,
                i32::try_from(image_w)
                    .map_err(|_| Exception::custom("Qwen-VL: image width overflow"))?,
            );
            let i32_idx =
                i32::try_from(i).map_err(|_| Exception::custom("Qwen-VL: batch index overflow"))?;
            // Crop to the image's own (unpadded) dims — the batch canvas is
            // zero-padded to the largest image, and padded regions must never
            // reach the tower.
            let pixel =
                batch
                    .pixel_values
                    .index((i32_idx..i32_idx + 1, ..image_h_i32, ..image_w_i32, ..));
            let &[_, h, w, _] = pixel.shape() else {
                return Err(Exception::custom(
                    "Qwen-VL: expected [1, H, W, 3] pixel slice",
                ));
            };
            let gh = h / patch;
            let gw = w / patch;
            // The SigLIP-shaped tower only supports its nominal grid today
            // (fixed learned position table); fail loudly instead of letting
            // the position lookup raise a cryptic broadcast error.
            let num_patches = self.vision_tower.num_patches();
            if gh * gw != num_patches {
                return Err(Exception::custom(format!(
                    "Qwen-VL: image grid {gh}×{gw} ({gh}*{gw} patches) does not match the \
                     tower's nominal {num_patches} patches — dynamic grids need the \
                     Qwen-VL RoPE tower (not yet implemented)"
                )));
            }
            // Tower features: one row per (patch-sized) grid cell.
            let feats = self.vision_tower.forward(&pixel)?; // [1, gh*gw, hidden]
            let hidden = self.config.vision_config.hidden_size;
            // Reference `merge_image_features`: 2×2 blocks of patch features
            // merge into one row of `hidden * merge²` — per_image_tokens is
            // `(gh / merge) * (gw / merge)` per image.
            let merged = feats
                .reshape(&[1, gh / merge, merge, gw / merge, merge, hidden])?
                .transpose_axes(&[0, 1, 3, 2, 4, 5])?
                .reshape(&[1, (gh / merge) * (gw / merge), hidden * merge * merge])?
                .index(0); // [tokens, hidden*merge*merge]
            rows.push(self.mm_projector.forward(&merged)?);
        }
        let image_features = if rows.is_empty() {
            Array::from_slice(&[0.0f32; 0], &[0, self.hidden_size()])
        } else {
            let refs: Vec<&Array> = rows.iter().collect();
            mlx_rs::ops::concatenate_axis(&refs, 0)?
        };
        // Loud check: the encoded row count must match the token budget the
        // prompt was expanded to. `merge_embeddings` only validates the
        // SENTINEL count (not the feature-row count), so any mismatch here
        // would otherwise silently mis-align features to tokens.
        let expected: usize = batch.per_image_tokens.iter().sum();
        let actual =
            usize::try_from(image_features.shape().first().copied().unwrap_or(0)).unwrap_or(0);
        if actual != expected {
            return Err(Exception::custom(format!(
                "Qwen-VL: encoded {actual} image feature rows, expected {expected} \
                 (sum of per_image_tokens)"
            )));
        }
        Ok(image_features)
    }
}

impl VisionModel for QwenVlModel {
    fn vision_capabilities(&self) -> VisionCapabilities {
        VisionCapabilities {
            families: vec!["qwen3_5_vl", "qwen3_vl", "qwen2_5_vl"],
            image_sizes: vec![self.config.patch_size * 32],
            supported_media: vec![
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/gif",
                "image/bmp",
            ],
            layout_kind: ImageTokenLayoutKind::StartEndPad,
        }
    }

    fn image_marker_text(&self) -> &'static str {
        "<|vision_start|><|image_pad|><|vision_end|>"
    }

    fn preprocess_images(&self, images: &[ImageInput]) -> Result<ImageBatch, VisionError> {
        let patch = u32::try_from(self.config.patch_size)
            .map_err(|_| VisionError::Preprocess("invalid patch_size".into()))?;
        let merge = u32::try_from(self.config.merge_size)
            .map_err(|_| VisionError::Preprocess("invalid merge_size".into()))?;
        let factor = patch * merge;
        let mut arrays = Vec::with_capacity(images.len());
        let mut per_image_tokens = Vec::with_capacity(images.len());
        let mut image_sizes = Vec::with_capacity(images.len());
        let mut max_h = 0u32;
        let mut max_w = 0u32;
        for img in images {
            let decoded = image::load_from_memory(&img.bytes)
                .map_err(|e| VisionError::Decode(e.to_string()))?;
            let (w, h) = decoded.dimensions();
            let (rh, rw) = smart_resize_with_factor(
                h,
                w,
                factor,
                self.config.min_pixels,
                self.config.max_pixels,
            )?;
            let array = preprocess_qwen_vl_image(&img.bytes, (rw, rh))?;
            arrays.push(array);
            // Token budget from the image's OWN resized dims; the padded batch
            // canvas must never change this (see `encode_all_images`).
            per_image_tokens.push(image_feature_rows(rh, rw, patch, merge));
            image_sizes.push((rh, rw));
            max_h = max_h.max(rh);
            max_w = max_w.max(rw);
        }
        let pixel_values = stack_padded(&arrays, max_h, max_w)?;
        Ok(ImageBatch {
            pixel_values,
            per_image_tokens,
            image_sizes,
            image_offsets: vec![],
            layout: ImageTokenLayout::default(),
        })
    }

    #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
    fn postprocess_image_tokens(
        &self,
        tokens: &mut Vec<u32>,
        tokenizer: &Tokenizer,
        batch: &ImageBatch,
    ) -> Result<(), VisionError> {
        let start_id = tokenizer.token_to_id("<|vision_start|>");
        let pad_id = tokenizer.token_to_id("<|image_pad|>");
        let end_id = tokenizer.token_to_id("<|vision_end|>");
        let (Some(start), Some(pad), Some(_end)) = (start_id, pad_id, end_id) else {
            return Err(VisionError::Preprocess(
                "tokenizer missing Qwen-VL vision tokens".to_owned(),
            ));
        };
        expand_qwen_vl_markers(
            tokens,
            start,
            pad,
            &batch.per_image_tokens,
            IMAGE_TOKEN_INDEX as u32,
        )
    }

    fn forward_multimodal(
        &mut self,
        input_ids: &Array,
        batch: &ImageBatch,
        cache: &mut AnyCache,
    ) -> Result<Array, Exception> {
        let AnyCache::Hybrid(c) = cache else {
            return Err(Exception::custom(
                "Qwen-VL requires a Hybrid cache (Qwen3Next backbone)",
            ));
        };
        // Validate batch=1 assumption.
        let batch_size = input_ids.shape().first().copied().unwrap_or(0);
        if batch_size != 1 {
            return Err(Exception::custom(format!(
                "Qwen-VL only supports batch_size=1, got {batch_size}"
            )));
        }
        let image_features = self.encode_all_images(batch)?; // [sum(per_image_tokens), lm_hidden]
        // Replace IMAGE_TOKEN_INDEX sentinels with 0 before embedding lookup;
        // merge_embeddings overwrites these positions with image features.
        let sentinel = Array::from_slice(&[IMAGE_TOKEN_INDEX], &[1]);
        let is_sentinel = input_ids.eq(&sentinel)?;
        let zero = Array::from_slice(&[0_i32], &[1]);
        let safe_ids = mlx_rs::ops::r#where(&is_sentinel, &zero, input_ids)?;
        let text_embeddings = self.language_model.embed_tokens_batch(&safe_ids)?;
        let combined =
            crate::vision::merge_embeddings(input_ids, &text_embeddings, &image_features, batch)?;
        self.language_model
            .forward_from_embeddings(&combined, None, c)
    }
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

/// Load a Qwen-VL model from a parsed config.json (wrapper `AnyModel`).
///
/// The text backbone loads through the qwen3.5 dense/MoE loaders (which strip
/// the `language_model.` prefix from the same safetensors files), then the
/// vision tower and projector weights are consumed from those files' remaining
/// `vision_tower.*` / `mm_projector.*` keys — mirroring
/// `llava_qwen2::load_llava_qwen2_model_from_value`.
pub(crate) fn load_qwen_vl_model_from_value(
    dir: &Path,
    raw: &Value,
) -> Result<AnyModel, ModelError> {
    let config = QwenVlConfig::from_value(raw)?;

    // Text backbone: the qwen3.5 dense or MoE loader (both strip the
    // `language_model.` prefix from the same safetensors files). Pick the MoE
    // variant when the text config declares experts.
    let is_moe = config
        .text_config
        .get("model_type")
        .and_then(Value::as_str)
        .is_some_and(|t| t.ends_with("_moe"))
        || config
            .text_config
            .get("num_experts")
            .and_then(Value::as_i64)
            .is_some_and(|n| n > 0);
    // Mirror `adapter::qwen35_args`: wrap a top-level text config so the
    // qwen3.5 arg loader sees a `text_config` node.
    let text_args = if raw.get("text_config").is_some() {
        crate::qwen3_next::load_qwen3_5_text_config_args_from_value(raw)?
    } else {
        let wrapped = serde_json::json!({ "text_config": raw.clone() });
        crate::qwen3_next::load_qwen3_5_text_config_args_from_value(&wrapped)?
    };
    let language_model = if is_moe {
        crate::qwen3_next::load_qwen3_5_moe_model_with_args(dir, text_args)?
    } else {
        crate::qwen3_next::load_qwen3_5_model_with_args(dir, text_args)?
    };

    tracing::info!(
        vision_layers = config.vision_config.num_hidden_layers,
        vision_hidden = config.vision_config.hidden_size,
        lm_hidden = language_model.args.hidden_size,
        lm_layers = language_model.args.num_hidden_layers,
        merge_size = config.merge_size,
        "Loading Qwen-VL model"
    );

    let mut vision_tower = SigLipVisionModel::new(&config.vision_config)?;
    let mut mm_projector =
        nn::LinearBuilder::new(config.mm_hidden_size, language_model.args.hidden_size).build()?;

    let weights = load_safetensor_weights(dir)?;

    // Vision tower weights: try both LLaVA-style prefix candidates.
    let prefixes = [
        "vision_tower.vision_model.",
        "vision_tower.vision_tower.vision_model.",
    ];
    let mut tower_err: Option<ModelError> = None;
    for prefix in prefixes {
        match load_siglip_weights(&mut vision_tower, &weights, prefix) {
            Ok(()) => {
                tower_err = None;
                break;
            }
            Err(err) => tower_err = Some(err),
        }
    }
    if let Some(err) = tower_err {
        return Err(err);
    }
    load_mm_projector_weights(&mut mm_projector, &weights)?;

    tracing::info!("Qwen-VL model loaded successfully");
    Ok(AnyModel::QwenVl(QwenVlModel {
        vision_tower,
        mm_projector,
        language_model,
        config,
    }))
}

fn load_mm_projector_weights(
    projector: &mut nn::Linear,
    weights: &HashMap<String, Array>,
) -> Result<(), ModelError> {
    let get = |name: &str| -> Result<Array, ModelError> {
        weights
            .get(name)
            .cloned()
            .ok_or_else(|| ModelError::MissingWeight(format!("Missing projector weight: {name}")))
    };
    projector.weight = Param::new(get("mm_projector.weight")?);
    projector.bias = Param::new(weights.get("mm_projector.bias").cloned());
    Ok(())
}

/// Load all safetensor weights from a model directory into a `HashMap`.
fn load_safetensor_weights(dir: &Path) -> Result<HashMap<String, Array>, ModelError> {
    let index_path = dir.join("model.safetensors.index.json");
    let single_path = dir.join("model.safetensors");

    let files: Vec<std::path::PathBuf> = if index_path.exists() {
        let index_str = std::fs::read_to_string(&index_path)?;
        let index: Value = serde_json::from_str(&index_str)?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| ModelError::MissingWeight("Missing weight_map in index".to_owned()))?;
        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(Value::as_str)
            .map(ToOwned::to_owned)
            .collect();
        shard_files.sort();
        shard_files.dedup();
        shard_files.into_iter().map(|f| dir.join(f)).collect()
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

#[cfg(test)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::shadow_reuse,
    clippy::unwrap_used
)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_respects_max_pixels() {
        // 1920x1080 image, min_pixels=256*28*28, max_pixels=1280*28*28
        let (h, w) = smart_resize(1080, 1920, 256 * 28 * 28, 1280 * 28 * 28).unwrap();
        assert!(h * w <= 1280 * 28 * 28);
        assert!(h * w >= 256 * 28 * 28 || h <= 28 || w <= 28);
    }

    #[test]
    fn smart_resize_matches_reference_values() {
        // Ported from the Qwen2-VL reference `smart_resize` (factor 28);
        // values cross-checked against an independent Python port.
        let mn = 256 * 28 * 28;
        let mx = 1280 * 28 * 28;
        assert_eq!(smart_resize(1080, 1920, mn, mx).unwrap(), (728, 1316));
        // Small image grows to the min pixel floor.
        assert_eq!(smart_resize(100, 100, mn, mx).unwrap(), (448, 448));
        // Large square shrinks to the max pixel budget.
        assert_eq!(smart_resize(4000, 4000, mn, mx).unwrap(), (980, 980));
        // Landscape and portrait aspect ratios keep their orientation.
        assert_eq!(smart_resize(768, 1024, mn, mx).unwrap(), (756, 1036));
        assert_eq!(smart_resize(1200, 900, mn, mx).unwrap(), (1148, 840));
    }

    #[test]
    fn smart_resize_enforces_min_and_max_pixels() {
        let (h, w) = smart_resize(100, 100, 256 * 28 * 28, 1280 * 28 * 28).unwrap();
        assert!(h * w >= 256 * 28 * 28);
        let (h, w) = smart_resize(4000, 4000, 256 * 28 * 28, 1280 * 28 * 28).unwrap();
        assert!(h * w <= 1280 * 28 * 28);
        // Both dims are multiples of the grid factor.
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn smart_resize_uses_bankers_rounding_like_python_round() {
        // 70/28 = 2.5 → Python round() → 2 (round-half-to-even), not 3.
        assert_eq!(
            smart_resize_with_factor(70, 70, 28, 1, 1_000_000).unwrap(),
            (56, 56)
        );
        // 98/28 = 3.5 → 4 (even) → 112.
        assert_eq!(
            smart_resize_with_factor(98, 98, 28, 1, 1_000_000).unwrap(),
            (112, 112)
        );
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect_ratios() {
        let err = smart_resize(1, 1000, 1, 1_000_000).unwrap_err();
        assert!(err.to_string().contains("aspect ratio"));
    }

    #[test]
    fn grid_computation() {
        assert_eq!(get_grid(448, 448, 14, 2), (16, 16));
        assert_eq!(get_grid(28, 56, 14, 2), (1, 2));
    }

    #[test]
    fn grid_computation_handles_non_aligned_inputs() {
        assert_eq!(get_grid(30, 60, 14, 2), (1, 2));
        assert_eq!(get_grid(14, 14, 14, 2), (1, 1));
    }

    #[test]
    fn per_image_feature_rows_reflect_own_resolutions() {
        // Each image's token budget comes from its OWN resized dims: 448×448 →
        // merged grid (16,16) → 256 rows; 336×336 → (12,12) → 144 rows.
        assert_eq!(image_feature_rows(448, 448, 14, 2), 256);
        assert_eq!(image_feature_rows(336, 336, 14, 2), 144);
        // Wider images keep their aspect: 1120×448 → (40,16) → 640 rows.
        assert_eq!(image_feature_rows(1120, 448, 14, 2), 640);
    }

    #[test]
    fn mixed_resolution_batch_budget_matches_per_image_rows() {
        // The review-caught bug: deriving grids from the PADDED canvas (the
        // batch max dims) inflates smaller images' rows — 336×336 would be
        // budgeted as 256 instead of 144, mis-aligning features to tokens.
        assert_ne!(
            image_feature_rows(448, 448, 14, 2),
            image_feature_rows(336, 336, 14, 2)
        );
        // Correct mixed-batch budget is 400 (= 256 + 144), not 512.
        let mixed: usize = [448, 336]
            .iter()
            .map(|&h| image_feature_rows(h, h, 14, 2))
            .sum();
        assert_eq!(mixed, 400);
    }

    #[test]
    fn expand_markers_single_image_pad_becomes_k_sentinels() {
        // [text, <|vision_start|>, <|image_pad|>, <|vision_end|>, text]
        let mut tokens = vec![1, 10, 20, 30, 2];
        expand_qwen_vl_markers(&mut tokens, 10, 20, &[4], IMAGE_TOKEN_INDEX as u32).unwrap();
        let s = IMAGE_TOKEN_INDEX as u32;
        assert_eq!(tokens, vec![1, 10, s, s, s, s, 30, 2]);
    }

    #[test]
    fn expand_markers_two_images_expand_in_order() {
        let mut tokens = vec![1, 10, 20, 30, 2, 10, 20, 30, 3];
        expand_qwen_vl_markers(&mut tokens, 10, 20, &[2, 3], IMAGE_TOKEN_INDEX as u32).unwrap();
        let s = IMAGE_TOKEN_INDEX as u32;
        assert_eq!(tokens, vec![1, 10, s, s, 30, 2, 10, s, s, s, 30, 3]);
        assert_eq!(tokens.len(), 12);
    }

    #[test]
    fn expand_markers_without_markers_is_unchanged() {
        let mut tokens = vec![1, 2, 3];
        expand_qwen_vl_markers(&mut tokens, 10, 20, &[], IMAGE_TOKEN_INDEX as u32).unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn expand_markers_uses_image_token_index_sentinel() {
        let mut tokens = vec![10, 20, 30];
        expand_qwen_vl_markers(&mut tokens, 10, 20, &[2], IMAGE_TOKEN_INDEX as u32).unwrap();
        let s = IMAGE_TOKEN_INDEX as u32;
        assert_eq!(tokens, vec![10, s, s, 30]);
        assert_eq!(tokens[1] as i32, IMAGE_TOKEN_INDEX);
    }

    #[test]
    fn expand_markers_errors_when_prompt_has_more_images_than_batch() {
        let mut tokens = vec![10, 20, 30, 10, 20, 30];
        let err = expand_qwen_vl_markers(&mut tokens, 10, 20, &[1], IMAGE_TOKEN_INDEX as u32)
            .unwrap_err();
        assert!(err.to_string().contains("more image markers"));
    }

    #[test]
    fn expand_markers_errors_when_batch_has_more_images_than_prompt() {
        let mut tokens = vec![10, 20, 30];
        let err = expand_qwen_vl_markers(&mut tokens, 10, 20, &[1, 1], IMAGE_TOKEN_INDEX as u32)
            .unwrap_err();
        assert!(err.to_string().contains("batch has 2 images"));
    }

    #[test]
    fn pixel_shuffle_matches_reference_block_layout() {
        // 4x4 single-channel image; patch=1, merge=2 → 16 rows of 1 element.
        // The reference transpose reorders rows as 2x2 blocks with their
        // sub-pixels interleaved.
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let img = Array::from_slice(&data, &[1, 4, 4]);
        let shuffled = pixel_shuffle(&img, 1, 2).unwrap();
        assert_eq!(shuffled.shape(), &[16, 1]);
        assert_eq!(
            shuffled.as_slice::<f32>(),
            &[
                0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 10.0, 11.0, 14.0,
                15.0
            ]
        );
    }

    #[test]
    fn pixel_shuffle_produces_grid_rows_per_patch_block() {
        // 28x28 RGB image, patch=14, merge=2 → 2x2 grid of rows, each
        // [3 * 14 * 14].
        let img = Array::from_slice(&[0.0f32; 3 * 28 * 28], &[3, 28, 28]);
        let shuffled = pixel_shuffle(&img, 14, 2).unwrap();
        assert_eq!(shuffled.shape(), &[4, 3 * 14 * 14]);
    }

    #[test]
    fn pixel_shuffle_rejects_non_aligned_dims() {
        let img = Array::from_slice(&[0.0f32; 3 * 30 * 30], &[3, 30, 30]);
        assert!(pixel_shuffle(&img, 14, 2).is_err());
    }

    #[test]
    fn config_parse_qwen25_style_vision_aliases() {
        let raw = serde_json::json!({
            "model_type": "qwen2_5_vl",
            "text_config": { "model_type": "qwen3_5_text", "hidden_size": 3584 },
            "vision_config": {
                "depth": 32,
                "hidden_act": "silu",
                "hidden_size": 1280,
                "intermediate_size": 3420,
                "num_heads": 16,
                "in_chans": 3,
                "out_hidden_size": 3584,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            }
        });
        let config = QwenVlConfig::from_value(&raw).unwrap();
        assert_eq!(config.vision_config.hidden_size, 1280);
        assert_eq!(config.vision_config.num_hidden_layers, 32);
        assert_eq!(config.vision_config.num_attention_heads, 16);
        assert_eq!(config.vision_config.num_channels, 3);
        assert_eq!(config.vision_config.patch_size, 14);
        assert_eq!(config.merge_size, 2);
        assert_eq!(config.patch_size, 14);
        // mm_hidden_size: out_hidden_size wins over hidden*merge² (5120).
        assert_eq!(config.mm_hidden_size, 3584);
        // Defaults when the wrapper omits min/max pixels.
        assert_eq!(config.min_pixels, 256 * 28 * 28);
        assert_eq!(config.max_pixels, 1280 * 28 * 28);
    }

    #[test]
    fn config_parse_qwen3_style_vision_derives_nominal_image_size() {
        let raw = serde_json::json!({
            "model_type": "qwen3_vl",
            "text_config": { "model_type": "qwen3_vl_text", "hidden_size": 4096 },
            "vision_config": {
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_heads": 16,
                "in_channels": 3,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304
            },
            "min_pixels": 65536,
            "max_pixels": 16777216
        });
        let config = QwenVlConfig::from_value(&raw).unwrap();
        assert_eq!(config.vision_config.num_hidden_layers, 27);
        assert_eq!(config.vision_config.num_attention_heads, 16);
        assert_eq!(config.vision_config.num_channels, 3);
        // No image_size: nominal = patch_size * 32 = 512.
        assert_eq!(config.vision_config.image_size, 512);
        assert_eq!(config.mm_hidden_size, 1152 * 2 * 2);
        assert_eq!(config.min_pixels, 65536);
        assert_eq!(config.max_pixels, 16777216);
    }

    #[test]
    fn config_parse_falls_back_to_top_level_text_config() {
        // `qwen2_5_vl`-style wrappers keep the text backbone at the top level.
        let raw = serde_json::json!({
            "model_type": "qwen2_5_vl",
            "hidden_size": 3584,
            "vocab_size": 152064,
            "vision_config": {
                "depth": 32,
                "hidden_size": 1280,
                "intermediate_size": 3420,
                "num_heads": 16,
                "in_chans": 3,
                "patch_size": 14,
                "spatial_merge_size": 2
            }
        });
        let config = QwenVlConfig::from_value(&raw).unwrap();
        assert_eq!(
            config
                .text_config
                .get("hidden_size")
                .and_then(Value::as_i64),
            Some(3584)
        );
        assert_eq!(config.vision_config.hidden_size, 1280);
    }

    #[test]
    fn config_parse_missing_vision_config_errors() {
        let raw = serde_json::json!({ "model_type": "qwen3_vl" });
        assert!(QwenVlConfig::from_value(&raw).is_err());
    }
}
