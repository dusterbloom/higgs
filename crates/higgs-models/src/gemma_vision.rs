//! Gemma 3/4 pan-and-scan vision preprocessing.
//!
//! The Gemma 3 vision tower consumes fixed `image_size x image_size` images.
//! For larger or non-square inputs, the model's *pan-and-scan* (`PaS`) scheme
//! crops several `image_size x image_size` windows out of the input and feeds
//! each through the tower separately, telling the language model where each
//! crop sits via a positional [`PanAndScan::offsets`] (in patch units).
//!
//! # Crop-anchor math (IMPORTANT for Task 13)
//!
//! The reference behavior described in the Task 12 plan is the `HuggingFace`
//! `gemma3` image processor (`pan_and_scan` / `make_batched_images`). Verified
//! against HF `transformers` (first `gemma3` commit `50d3530` through current
//! `main`): the real HF processor does **not** use a fixed crop set or
//! patch-unit offsets — it tiles the image into aspect-ratio-driven,
//! non-overlapping windows and resizes each (plus the original) to a square.
//! There is no `crop_set`/offset convention to copy, so this module implements
//! the plan's sketch faithfully and documents the exact convention here.
//! Task 13 must re-validate these choices against a real Gemma 3 checkpoint.
//!
//! Conventions implemented:
//!
//! 1. **Resize** — the image is scaled so its **shorter side** becomes
//!    `image_size` (aspect-preserving, no letterbox); the longer side scales
//!    proportionally and may exceed `image_size`. This is the only reading of
//!    the plan's note ("…the shorter side is typically ≥ `image_size` after
//!    resize, so crops fit") under which every `image_size²` window fits inside
//!    the resized image. The sketch's literal `resize(image_size, image_size)`
//!    call (longest side → `image_size`) would shrink the shorter side *below*
//!    `image_size`, forcing every window past the edge and collapsing all
//!    crops onto one anchor — no pan-and-scan at all.
//! 2. **Crop anchors** — for crop-set coordinate `(row, col)` the window's
//!    top-left corner is `anchor = clamp(round(frac * (span - target)), 0,
//!    span - target)` per axis, with `frac = coord / 2` — the same grid
//!    fraction the offset formula divides by. `0 -> 0` (leading edge),
//!    `1 -> (span - target + 1) / 2` (center), `2 -> span - target` (trailing
//!    edge). Rounding is half-up.
//! 3. **Offsets** — in patch units, exactly per the plan's sketch:
//!    `offset_row = (row * (grid - 1)) / 2`, `offset_col = (col * (grid - 1)) / 2`
//!    (truncating integer division), linearized as
//!    `offset_row * grid + offset_col`, `grid = image_size / patch_size`.
//!    These are grid-slot encodings (e.g. 0, 27, 55 for grid 56), not exact
//!    pixel positions in patch units; the transformer consumes them as
//!    positional offsets and Task 13 should match them against a real
//!    checkpoint.
//! 4. **Normalization** — `mean = std = 0.5` (the Gemma 3
//!    `preprocessor_config` values; identical to the `SigLIP` path), mapping
//!    `[0, 255] -> [-1, 1]`.
//! 5. **Aspect-aware crop set** — the grid is rotated to the image's long axis
//!    so its denser divisions sample the direction that actually extends
//!    beyond `image_size`: landscape/square inputs use the default 2-row x
//!    3-col set `[(0,0), (0,1), (1,0), (1,1), (0,2), (1,2)]`, portrait inputs
//!    use the transposed 3-row x 2-col set `[(0,0), (0,1), (1,0), (1,1),
//!    (2,0), (2,1)]`.
//! 6. **Distinct-window dedup** — the short axis collapses to exactly
//!    `image_size` after the resize, so its grid anchors coincide (e.g. the
//!    two row anchors of a landscape image). Windows that are pixel-identical
//!    are emitted **once** (first occurrence, keeping its offset), so
//!    identical pixels never carry two different positional offsets. A square
//!    input therefore yields a single whole-image crop. The crop/offset
//!    counts below are the *distinct* counts; `per_image_tokens` uses them.
//!
//! The default crop set is the plan's 3x2 grid: 4 corners + 2 centers,
//! `[(0,0), (0,1), (1,0), (1,1), (0,2), (1,2)]` (see [`default_crop_set`] and
//! [`portrait_crop_set`]).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{Array, error::Exception, ops, ops::indexing::IndexOp, transforms::eval};
use tokenizers::Tokenizer;

use crate::{
    error::ModelError,
    siglip::{SigLipVisionConfig, SigLipVisionModel, load_siglip_weights},
    vision::{
        IMAGE_TOKEN_INDEX, ImageBatch, ImageInput, ImageTokenLayout, ImageTokenLayoutKind,
        VisionCapabilities, VisionError,
    },
};

/// Configuration for the Gemma vision tower, read from `vision_config` in the
/// model's `config.json` (Task 13).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmaVisionConfig {
    /// Side length of the tower's square input (e.g. 896).
    pub image_size: i32,
    /// Vision tower patch size (e.g. 14 or 16).
    pub patch_size: i32,
    /// Number of patch embeddings per full image, `(image_size/patch_size)^2`
    /// (used by Task 13 for per-image token counts; not needed here).
    pub num_patches: i32,
    /// Pan-and-scan crop set: `(row, col)` grid coordinates in `{0, 1, 2}²`,
    /// e.g. [`default_crop_set`].
    pub crop_set: Vec<(i32, i32)>,
    /// Language-model tokens one full-image crop expands to after the 2x2
    /// patch pooling (`mm_tokens_per_image` from the checkpoint, else
    /// `num_patches / 4`; see [`gemma_tokens_per_crop`]).
    pub tokens_per_crop: usize,
}

impl GemmaVisionConfig {
    /// Build from a `SigLIP`-shaped `vision_config` plus the checkpoint's
    /// `mm_tokens_per_image` (which lives at the top level of `config.json`,
    /// not inside `vision_config`).
    pub fn from_siglip(
        siglip: &SigLipVisionConfig,
        mm_tokens_per_image: Option<i32>,
    ) -> Result<Self, ModelError> {
        let num_patches = siglip.num_patches();
        let tokens_per_crop = gemma_tokens_per_crop(num_patches, mm_tokens_per_image)?;
        Ok(Self {
            image_size: siglip.image_size,
            patch_size: siglip.patch_size,
            num_patches,
            crop_set: default_crop_set(),
            tokens_per_crop,
        })
    }
}

/// Number of language-model tokens one full-image crop expands to.
///
/// The reference Gemma 3 checkpoints store this as `mm_tokens_per_image` at
/// the top level of `config.json` (256 for `gemma-3-27b` with 896²/14px
/// patches pooled 4x4; 1024 for `gemma-3-4b` pooled 2x2) — never hardcode it.
/// When the field is absent, fall back to the plan's 4:1 (2x2) patch pooling
/// `num_patches / 4`.
pub fn gemma_tokens_per_crop(
    num_patches: i32,
    mm_tokens_per_image: Option<i32>,
) -> Result<usize, ModelError> {
    let grid = num_patches.isqrt();
    if grid <= 0 || grid * grid != num_patches {
        return Err(ModelError::UnsupportedModel(format!(
            "Gemma vision: num_patches {num_patches} is not a perfect square"
        )));
    }
    let tokens = match mm_tokens_per_image {
        Some(t) if t > 0 => t,
        _ => {
            if num_patches % 4 != 0 {
                return Err(ModelError::UnsupportedModel(format!(
                    "Gemma vision: num_patches {num_patches} is not divisible by 4, \
                     no 4:1 pooling fallback"
                )));
            }
            num_patches / 4
        }
    };
    let tokens_per_side = tokens.isqrt();
    if tokens_per_side <= 0 || tokens_per_side * tokens_per_side != tokens {
        return Err(ModelError::UnsupportedModel(format!(
            "Gemma vision: mm_tokens_per_image {tokens} is not a perfect square"
        )));
    }
    if grid % tokens_per_side != 0 {
        return Err(ModelError::UnsupportedModel(format!(
            "Gemma vision: grid {grid} is not divisible by tokens_per_side {tokens_per_side}"
        )));
    }
    usize::try_from(tokens)
        .map_err(|_| ModelError::UnsupportedModel("Gemma vision: tokens_per_crop overflow".into()))
}

/// Default pan-and-scan crop set: `(row_frac, col_frac)` over a 3x2 grid.
///
/// Four corners plus the two horizontal centers (2 row fractions x 3 column
/// fractions). The exact set is checkpoint-specific; `vision_config.crop_size`
/// or the reference gemma3 processor defines it. Used for square and
/// landscape inputs; [`portrait_crop_set`] is the 90° rotation used when the
/// image is taller than wide.
pub fn default_crop_set() -> Vec<(i32, i32)> {
    vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)]
}

/// Rotated crop set for portrait (taller-than-wide) images: 3 row fractions
/// x 2 column fractions, so the denser grid divisions lie along the vertical
/// (long) axis and the crops stay distinct.
pub fn portrait_crop_set() -> Vec<(i32, i32)> {
    vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
}

/// The output of [`pan_and_scan`]: one normalized crop per **distinct** anchor
/// of the aspect-aware crop set (pixel-identical windows from the short-axis
/// collapse are deduplicated), plus the matching positional offsets.
#[derive(Debug, Clone)]
pub struct PanAndScan {
    /// Crops in `[1, image_size, image_size, 3]` NHWC float32 form,
    /// normalized with `mean = std = 0.5`.
    pub crops: Vec<Array>,
    /// Positional offset of each crop in patch units (linearized
    /// `offset_row * grid + offset_col`; see the module docs).
    pub offsets: Vec<i32>,
}

/// Run Gemma pan-and-scan preprocessing on one image.
///
/// Decodes `image_bytes`, resizes the image so its shorter side becomes
/// `config.image_size`, crops one `image_size²` window per entry of the
/// aspect-aware crop set (anchored per the module docs), and normalizes each
/// crop to `[1, image_size, image_size, 3]`.
///
/// The crop set is rotated to the image's long axis ([`default_crop_set`] for
/// square/landscape, [`portrait_crop_set`] for portrait), and windows that are
/// pixel-identical because the short axis collapsed to `image_size` (e.g. the
/// two row anchors of a landscape image) are deduplicated: the output holds
/// one crop per **distinct anchor**, so identical pixels never carry two
/// different positional offsets. A square input therefore yields a single
/// (whole-image) crop.
pub fn pan_and_scan(
    image_bytes: &[u8],
    config: &GemmaVisionConfig,
) -> Result<PanAndScan, VisionError> {
    let image_size = config.image_size;
    let patch_size = config.patch_size;
    if image_size <= 0 || patch_size <= 0 || image_size % patch_size != 0 {
        return Err(VisionError::Preprocess(format!(
            "invalid Gemma vision config: image_size={image_size}, patch_size={patch_size}; \
             image_size must be a positive multiple of patch_size"
        )));
    }
    if config.crop_set.is_empty() {
        return Err(VisionError::Preprocess(
            "Gemma pan-and-scan crop_set must not be empty".to_owned(),
        ));
    }
    for &(row, col) in &config.crop_set {
        if !(0..=2).contains(&row) || !(0..=2).contains(&col) {
            return Err(VisionError::Preprocess(format!(
                "Gemma pan-and-scan crop coordinate ({row}, {col}) is outside the \
                 supported {{0, 1, 2}}^2 grid"
            )));
        }
    }

    let img =
        image::load_from_memory(image_bytes).map_err(|e| VisionError::Decode(e.to_string()))?;
    let (orig_w, orig_h) = (img.width(), img.height());
    // image_size > 0 was validated above, so this conversion always succeeds.
    let target = u32::try_from(image_size).unwrap_or(u32::MAX);

    // Aspect-aware crop set: the denser grid divisions lie along the long axis
    // so the crops stay distinct for non-square inputs.
    let crop_set = if orig_w < orig_h {
        portrait_crop_set()
    } else {
        default_crop_set()
    };

    // Resize so the shorter side becomes exactly `target`; the longer side is
    // then >= target, so every crop window fits. Aspect ratio is preserved and
    // nothing is letterboxed/padded. New dims use integer round-half-up of
    // `dim * target / short` (u64 to avoid overflow).
    let short = orig_w.min(orig_h);
    let new_w = u32::try_from(
        (u64::from(orig_w) * u64::from(target) + u64::from(short) / 2) / u64::from(short),
    )
    .unwrap_or(u32::MAX)
    .max(1);
    let new_h = u32::try_from(
        (u64::from(orig_h) * u64::from(target) + u64::from(short) / 2) / u64::from(short),
    )
    .unwrap_or(u32::MAX)
    .max(1);
    let resized = img
        .resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
        .to_rgb8();

    let grid = image_size / patch_size;
    let mut crops = Vec::with_capacity(crop_set.len());
    let mut offsets = Vec::with_capacity(crop_set.len());
    let mut seen_anchors: Vec<(u32, u32)> = Vec::with_capacity(crop_set.len());
    for &(row, col) in &crop_set {
        // Offsets in patch units along each axis — the plan's sketch verbatim
        // (truncating integer division).
        let offset_row = (row * (grid - 1)) / 2;
        let offset_col = (col * (grid - 1)) / 2;

        let (x, y) = crop_anchor(row, col, target, new_w, new_h);
        // Windows that are pixel-identical because the short axis collapsed to
        // `target` must not be emitted twice with different offsets — keep the
        // first occurrence only (its offset is the long-axis position).
        if seen_anchors.iter().any(|&(sx, sy)| sx == x && sy == y) {
            continue;
        }
        seen_anchors.push((x, y));
        offsets.push(offset_row * grid + offset_col);
        crops.push(to_normalized_array(
            crop_square(&resized, row, col, target),
            target,
        ));
    }

    Ok(PanAndScan { crops, offsets })
}

/// Top-left pixel of the `target x target` crop window for crop-set coordinate
/// `(row, col)`: `anchor = round((coord / 2) * (span - target))` clamped to
/// `[0, span - target]` per axis (see the module docs).
///
/// `span >= target` on both axes is guaranteed by the shorter-side resize in
/// [`pan_and_scan`], so the window always fits.
fn crop_anchor(row: i32, col: i32, target: u32, span_w: u32, span_h: u32) -> (u32, u32) {
    let anchor = |coord: i32, span: u32| -> u32 {
        let span_i = i64::from(span);
        let target_i = i64::from(target);
        let delta = span_i.saturating_sub(target_i); // >= 0 under shorter-side resize
        // round-half-up of (coord * delta) / 2, then clamp to [0, delta].
        let raw = (i64::from(coord) * delta + 1) / 2;
        let clamped = raw.clamp(0, delta);
        u32::try_from(clamped).unwrap_or(u32::MAX)
    };
    (anchor(col, span_w), anchor(row, span_h))
}

// ---------------------------------------------------------------------------
// Marker expansion + patch pooling (Task 13)
// ---------------------------------------------------------------------------

/// Pure expansion used by the Gemma postprocessor: every `<start_of_image>
/// <end_of_image>` marker run becomes `[start][IMAGE_TOKEN_INDEX × k][end]`,
/// consuming one entry of `per_image_tokens` per run.
///
/// Runs beyond the declared counts fall back to a single sentinel (`k = 1`),
/// matching the plan sketch.
#[allow(clippy::as_conversions, clippy::cast_sign_loss)]
pub fn expand_gemma_markers(
    tokens: &mut Vec<u32>,
    start: u32,
    end: u32,
    per_image_tokens: &[usize],
) {
    let mut out = Vec::with_capacity(tokens.len() + per_image_tokens.iter().sum::<usize>());
    let mut img_idx = 0usize;
    let mut i = 0usize;
    while let Some(&cur) = tokens.get(i) {
        if cur == start && tokens.get(i + 1) == Some(&end) {
            out.push(start);
            let k = per_image_tokens.get(img_idx).copied().unwrap_or(1);
            img_idx += 1;
            for _ in 0..k {
                out.push(IMAGE_TOKEN_INDEX as u32);
            }
            out.push(end);
            i += 2;
        } else {
            out.push(cur);
            i += 1;
        }
    }
    *tokens = out;
}

/// Average-pool the patch grid of a batch of crops to `tokens_per_side²` rows.
///
/// `features` is `[B, grid, grid, hidden]`; each `kernel × kernel` block
/// (`kernel = grid / tokens_per_side`) is averaged — the reference Gemma 3
/// `AvgPool2d(kernel_size, kernel_size)` in the multi-modal projector.
/// Returns `[B, tokens_per_side², hidden]`.
pub fn pool_patch_features(features: &Array, tokens_per_side: i32) -> Result<Array, Exception> {
    let shape = features.shape();
    let [b, g1, g2, hidden] = *shape else {
        return Err(Exception::custom(format!(
            "pool_patch_features: expected [B, grid, grid, hidden], got {shape:?}"
        )));
    };
    if g1 != g2 {
        return Err(Exception::custom(format!(
            "pool_patch_features: non-square patch grid {g1}x{g2}"
        )));
    }
    if g1 % tokens_per_side != 0 {
        return Err(Exception::custom(format!(
            "pool_patch_features: grid {g1} is not divisible by tokens_per_side {tokens_per_side}"
        )));
    }
    let kernel = g1 / tokens_per_side;
    // [B, s, kernel, s, kernel, H] -> mean over the two kernel axes.
    let reshaped =
        features.reshape(&[b, tokens_per_side, kernel, tokens_per_side, kernel, hidden])?;
    let pooled = reshaped.mean_axes(&[2, 4], false)?;
    pooled.reshape(&[b, tokens_per_side * tokens_per_side, hidden])
}

// ---------------------------------------------------------------------------
// Vision tower (Task 13)
// ---------------------------------------------------------------------------

/// The Gemma 3/4 vision stack: a SigLIP-style tower plus the pan-and-scan
/// config that describes how images are cropped and how many language-model
/// tokens each crop expands to.
#[derive(Debug, Clone)]
pub struct GemmaVisionTower {
    /// Pan-and-scan geometry + per-crop token count.
    pub config: GemmaVisionConfig,
    /// The vision encoder.
    pub tower: SigLipVisionModel,
}

impl GemmaVisionTower {
    /// Build the tower stack.
    pub const fn new(config: GemmaVisionConfig, tower: SigLipVisionModel) -> Self {
        Self { config, tower }
    }

    /// Language-model tokens per crop after patch pooling (`k`).
    pub const fn tokens_per_crop(&self) -> usize {
        self.config.tokens_per_crop
    }

    /// Capability metadata for the given family name (`"gemma3"` / `"gemma4"`).
    pub fn vision_capabilities(&self, families: Vec<&'static str>) -> VisionCapabilities {
        VisionCapabilities {
            families,
            image_sizes: vec![self.config.image_size],
            supported_media: vec![
                "image/png",
                "image/jpeg",
                "image/jpg",
                "image/webp",
                "image/gif",
                "image/bmp",
            ],
            layout_kind: ImageTokenLayoutKind::StartEndPad,
        }
    }

    /// Pan-and-scan preprocess for all images.
    ///
    /// Every image becomes `crop_set.len()` crops stacked along the batch
    /// axis; `per_image_tokens[i] = crops × tokens_per_crop` and
    /// [`ImageBatch::image_offsets`] holds each crop's positional offset in
    /// crop order (see [`pan_and_scan`]).
    pub fn preprocess_images(&self, images: &[ImageInput]) -> Result<ImageBatch, VisionError> {
        let k = self.config.tokens_per_crop;
        let mut crops: Vec<Array> = Vec::new();
        let mut per_image_tokens = Vec::with_capacity(images.len());
        let mut image_offsets: Vec<i32> = Vec::new();
        for img in images {
            let ps = pan_and_scan(&img.bytes, &self.config)?;
            let n = ps.crops.len();
            per_image_tokens.push(n * k);
            image_offsets.extend(ps.offsets);
            crops.extend(ps.crops);
        }
        let pixel_values = if crops.is_empty() {
            Array::from_slice::<f32>(&[], &[0, 1, 1, 3])
        } else {
            let refs: Vec<&Array> = crops.iter().collect();
            ops::concatenate_axis(&refs, 0).map_err(|e| VisionError::Preprocess(e.to_string()))?
        };
        let side = u32::try_from(self.config.image_size).unwrap_or(u32::MAX);
        Ok(ImageBatch {
            pixel_values,
            per_image_tokens,
            image_sizes: vec![(side, side); images.len()],
            image_offsets,
            layout: ImageTokenLayout::default(),
        })
    }

    /// Expand `<start_of_image><end_of_image>` marker runs into
    /// `start + k × IMAGE_TOKEN_INDEX + end` using the batch's per-image
    /// counts. Marker ids are resolved from the tokenizer.
    pub fn postprocess_image_tokens(
        &self,
        tokens: &mut Vec<u32>,
        tokenizer: &Tokenizer,
        batch: &ImageBatch,
    ) -> Result<(), VisionError> {
        let Some(start) = tokenizer.token_to_id("<start_of_image>") else {
            return Ok(()); // tokenizer without the marker: nothing to expand
        };
        let Some(end) = tokenizer.token_to_id("<end_of_image>") else {
            return Ok(());
        };
        expand_gemma_markers(tokens, start, end, &batch.per_image_tokens);
        Ok(())
    }

    /// Encode a batch of crops and average-pool each to `tokens_per_crop` rows.
    ///
    /// `pixel_values` is `[total_crops, image_size, image_size, 3]`; the
    /// result is `[sum(per_image_tokens), hidden]` in crop order, matching
    /// `merge_embeddings`'s expected feature layout.
    pub fn encode(&mut self, pixel_values: &Array) -> Result<Array, Exception> {
        let raw = self.tower.forward(pixel_values)?; // [C, num_patches, hidden]
        let grid = self.config.num_patches.isqrt();
        let tokens_per_side = i32::try_from(self.config.tokens_per_crop.isqrt())
            .map_err(|_| Exception::custom("Gemma vision: tokens_per_crop overflow for i32"))?;
        let c = *raw
            .shape()
            .first()
            .ok_or_else(|| Exception::custom("Gemma vision: empty crop batch"))?;
        let mut pooled = Vec::with_capacity(usize::try_from(c).unwrap_or(0));
        for i in 0..c {
            let crop = raw
                .index((i..i + 1, .., ..))
                .reshape(&[1, grid, grid, -1])?;
            let block = pool_patch_features(&crop, tokens_per_side)?;
            let block_shape = block.shape();
            let b = *block_shape
                .first()
                .ok_or_else(|| Exception::custom("Gemma vision: pooled crop has no batch dim"))?;
            let hidden = *block_shape
                .last()
                .ok_or_else(|| Exception::custom("Gemma vision: pooled crop has no hidden dim"))?;
            pooled.push(block.reshape(&[b * tokens_per_side * tokens_per_side, hidden])?);
        }
        let refs: Vec<&Array> = pooled.iter().collect();
        ops::concatenate_axis(&refs, 0)
    }

    /// Per-position `RoPE` offsets for a merged sequence (`input_ids` with
    /// `IMAGE_TOKEN_INDEX` at image rows).
    ///
    /// Text rows keep their natural sequence position; each image row uses
    /// `crop_offset + row_in_crop`, where `crop_offset` is the crop's
    /// pan-and-scan grid-slot offset (see [`pan_and_scan`]). This feeds the
    /// offsets into the backbone's `RoPE` application via
    /// `forward_from_embeddings_with_offsets`.
    pub fn build_position_offsets(
        &self,
        input_ids: &Array,
        batch: &ImageBatch,
    ) -> Result<Array, Exception> {
        eval([input_ids])?;
        let ids: Vec<i32> = input_ids.index(0).as_slice::<i32>().to_vec();
        let k = self.config.tokens_per_crop;
        let l = ids.len();

        // Cumulative crop index before each image.
        let mut cum_crops = Vec::with_capacity(batch.per_image_tokens.len());
        let mut acc = 0usize;
        for &t in &batch.per_image_tokens {
            cum_crops.push(acc);
            if t % k != 0 {
                return Err(Exception::custom(format!(
                    "Gemma vision: per_image_tokens {t} is not divisible by tokens_per_crop {k}"
                )));
            }
            acc += t / k;
        }
        if acc != batch.image_offsets.len() {
            return Err(Exception::custom(format!(
                "Gemma vision: {acc} crops but {} offsets",
                batch.image_offsets.len()
            )));
        }

        let mut offsets = vec![0i32; l];
        let mut image_idx = 0usize;
        let mut row_in_image = 0usize;
        for (i, &id) in ids.iter().enumerate() {
            if id == IMAGE_TOKEN_INDEX {
                let per_image = *batch.per_image_tokens.get(image_idx).ok_or_else(|| {
                    Exception::custom("Gemma vision: more image rows than per_image_tokens")
                })?;
                let crop = row_in_image / k;
                let row_in_crop = row_in_image % k;
                let crop_global = *cum_crops
                    .get(image_idx)
                    .ok_or_else(|| Exception::custom("Gemma vision: image index out of bounds"))?
                    + crop;
                let crop_offset = *batch.image_offsets.get(crop_global).ok_or_else(|| {
                    Exception::custom(format!(
                        "Gemma vision: missing offset for crop {crop_global}"
                    ))
                })?;
                let position = crop_offset
                    + i32::try_from(row_in_crop).map_err(|_| {
                        Exception::custom("Gemma vision: row index overflow for i32")
                    })?;
                if let Some(slot) = offsets.get_mut(i) {
                    *slot = position;
                }
                row_in_image += 1;
                if row_in_image == per_image {
                    row_in_image = 0;
                    image_idx += 1;
                }
            } else if let Some(slot) = offsets.get_mut(i) {
                *slot = i32::try_from(i)
                    .map_err(|_| Exception::custom("Gemma vision: sequence too long"))?;
            }
        }
        let l_i32 = i32::try_from(l)
            .map_err(|_| Exception::custom("Gemma vision: sequence too long for i32"))?;
        Ok(Array::from_slice(&offsets, &[l_i32]))
    }
}

/// Load a Gemma 3/4 vision tower when the checkpoint carries `vision_tower.`
/// keys; returns `None` for text-only checkpoints so `gemma3_text` /
/// `gemma4_text` loading behaves exactly as before.
///
/// The tower is a SigLIP-style encoder read from `vision_config` (a
/// `SigLipVisionConfig` shape) with the `vision_tower.vision_model.` weight
/// prefix (or the LLaVA-style `vision_tower.vision_tower.vision_model.`
/// fallback). `mm_tokens_per_image` is read from the top level of
/// `config.json` and drives the per-crop token count.
pub(crate) fn load_gemma_vision_tower(
    model_path: &Path,
) -> Result<Option<GemmaVisionTower>, ModelError> {
    if !crate::checkpoint_has_key_containing(model_path, "vision_tower.")? {
        return Ok(None);
    }

    let config_path = model_path.join("config.json");
    let file = std::fs::File::open(config_path)?;
    let raw: serde_json::Value = serde_json::from_reader(file)?;

    let vc = raw.get("vision_config").ok_or_else(|| {
        ModelError::MissingWeight(
            "checkpoint has vision_tower weights but config.json has no vision_config".to_owned(),
        )
    })?;
    let siglip_config: SigLipVisionConfig = serde_json::from_value(vc.clone())?;
    let mm_tokens = raw
        .get("mm_tokens_per_image")
        .and_then(serde_json::Value::as_i64)
        .map(|v| i32::try_from(v).unwrap_or(i32::MAX))
        .filter(|v| *v > 0);
    let config = GemmaVisionConfig::from_siglip(&siglip_config, mm_tokens)?;

    let mut tower = SigLipVisionModel::new(&siglip_config)?;

    // Load the weight map and keep only the tower keys, then pick the prefix
    // that actually exists in this checkpoint (gemma3 checkpoints use
    // `vision_tower.vision_model.`; some wrappers nest one level deeper).
    let mut weights = load_safetensor_weight_map(model_path)?;
    // Keep only the tower tensors; the LM tensors were already loaded by the
    // text loader and dropping them here avoids holding a second reference to
    // the full weight set (the arrays are lazy, so this is cheap).
    weights.retain(|k, _| k.starts_with("vision_tower."));
    let prefix = if weights
        .keys()
        .any(|k| k.starts_with("vision_tower.vision_tower.vision_model."))
    {
        "vision_tower.vision_tower.vision_model."
    } else {
        "vision_tower.vision_model."
    };
    load_siglip_weights(&mut tower, &weights, prefix)?;

    tracing::info!(
        image_size = config.image_size,
        patch_size = config.patch_size,
        num_patches = config.num_patches,
        tokens_per_crop = config.tokens_per_crop,
        crops = config.crop_set.len(),
        "Loaded Gemma vision tower"
    );
    Ok(Some(GemmaVisionTower::new(config, tower)))
}

/// Load every safetensors file in a model directory into one weight map.
fn load_safetensor_weight_map(model_path: &Path) -> Result<HashMap<String, Array>, ModelError> {
    let files = crate::collect_safetensors_files(model_path)?;
    let workspace_bytes = files.iter().try_fold(0_u64, |total, path| {
        let bytes = std::fs::metadata(path)?.len();
        total.checked_add(bytes).ok_or_else(|| {
            ModelError::LoadCapacity("Gemma vision artifact workspace overflow".to_owned())
        })
    })?;
    crate::progress::report_load_boundary(crate::progress::LoadBoundary::BeforeConversion {
        index: 0,
        bytes: workspace_bytes,
        kind: crate::progress::ConversionKind::FullArtifact,
    })?;
    let mut weights = HashMap::new();
    for (index, file_path) in files.iter().enumerate() {
        crate::progress::report_before_shard(index, file_path)?;
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;
        weights.extend(loaded);
        crate::progress::report_after_shard(index)?;
    }
    crate::progress::report_load_boundary(crate::progress::LoadBoundary::AfterConversion {
        index: 0,
        kind: crate::progress::ConversionKind::FullArtifact,
    })?;
    Ok(weights)
}

/// Extract the `target x target` RGB window for crop-set coordinate `(row,
/// col)`: top-left corner at `anchor = round((coord / 2) * (span - target))`
/// clamped to `[0, span - target]` per axis (see the module docs).
///
/// `span >= target` on both axes is guaranteed by the shorter-side resize in
/// [`pan_and_scan`], so the window always fits.
/// Extract the `target x target` RGB window for crop-set coordinate `(row,
/// col)` (anchored via [`crop_anchor`]; see the module docs).
///
/// `span >= target` on both axes is guaranteed by the shorter-side resize in
/// [`pan_and_scan`], so the window always fits.
fn crop_square(rgb: &image::RgbImage, row: i32, col: i32, target: u32) -> image::RgbImage {
    let (w, h) = rgb.dimensions();
    let (x, y) = crop_anchor(row, col, target, w, h);
    debug_assert!(
        x.checked_add(target).is_some_and(|end| end <= w),
        "pan-and-scan crop window overflows the resized image width"
    );
    debug_assert!(
        y.checked_add(target).is_some_and(|end| end <= h),
        "pan-and-scan crop window overflows the resized image height"
    );
    image::imageops::crop_imm(rgb, x, y, target, target).to_image()
}

/// Convert an exact `target x target` RGB crop into a `[1, target, target, 3]`
/// NHWC float32 array normalized with `mean = std = 0.5`
/// (`(pixel/255 - 0.5) / 0.5`, mapping `[0, 255] -> [-1, 1]`).
fn to_normalized_array(crop: image::RgbImage, target: u32) -> Array {
    // crop_square always returns an exact target x target window.
    let target_i32 = i32::try_from(target).unwrap_or(i32::MAX);
    let pixels = crop.into_raw();
    let float_pixels: Vec<f32> = pixels
        .iter()
        .map(|&p| (f32::from(p) / 255.0 - 0.5) / 0.5)
        .collect();
    Array::from_slice(&float_pixels, &[1, target_i32, target_i32, 3])
}

#[cfg(test)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]
mod tests {
    use super::*;

    /// 1x1 red PNG, generated in-memory (no fixture files on disk).
    fn test_png() -> Vec<u8> {
        let mut img = image::RgbImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png)
            .expect("encode test PNG");
        buf.into_inner()
    }

    /// `width`x`height` PNG with a distinct color per pixel.
    fn test_png_wh(width: u32, height: u32) -> Vec<u8> {
        let mut img = image::RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let v = (x + y * width) as u8 * 40;
                img.put_pixel(
                    x,
                    y,
                    image::Rgb([v, v.wrapping_add(60), v.wrapping_add(120)]),
                );
            }
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png)
            .expect("encode test PNG");
        buf.into_inner()
    }

    #[test]
    fn pan_and_scan_landscape_yields_distinct_long_axis_crops() {
        // 3x2 px landscape: the shorter side (h=2) is resized to 896 ->
        // 1344x896. The default 2x3 grid's row anchors collapse (h == target)
        // and the three column anchors 0/224/448 stay distinct; the row
        // duplicates are deduplicated, leaving one crop per distinct anchor.
        let cfg = GemmaVisionConfig {
            image_size: 896,
            patch_size: 16,
            num_patches: 56 * 56,
            tokens_per_crop: 784,
            crop_set: default_crop_set(),
        };
        let ps = pan_and_scan(&test_png_wh(3, 2), &cfg).unwrap();
        assert_eq!(ps.crops.len(), 3);
        assert_eq!(ps.offsets.len(), 3);
        assert_eq!(ps.crops[0].shape(), &[1, 896, 896, 3]);
        // All crops are pixel-distinct (no duplicated windows).
        assert_ne!(ps.crops[0].as_slice::<f32>(), ps.crops[1].as_slice::<f32>());
        assert_ne!(ps.crops[1].as_slice::<f32>(), ps.crops[2].as_slice::<f32>());
        assert_ne!(ps.crops[0].as_slice::<f32>(), ps.crops[2].as_slice::<f32>());
    }

    #[test]
    fn pan_and_scan_small_config_is_fast_and_shaped() {
        let cfg = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
            tokens_per_crop: 4,
            crop_set: default_crop_set(),
        };
        // A square input collapses to a single whole-image crop (every grid
        // anchor is (0,0)) — no panning is needed for a square image.
        let ps = pan_and_scan(&test_png(), &cfg).unwrap();
        assert_eq!(ps.crops.len(), 1);
        assert_eq!(ps.offsets.len(), 1);
        assert_eq!(ps.offsets[0], 0);
        assert_eq!(ps.crops[0].shape(), &[1, 32, 32, 3]);
    }

    #[test]
    fn pan_and_scan_offsets_match_sketch_formula() {
        // grid = 56: offset_row = (r * 55) / 2, offset_col = (c * 55) / 2,
        // offset = offset_row * 56 + offset_col. For landscape the kept crops
        // are the three column anchors (0,0), (0,1), (0,2) -> offsets 0, 27, 55.
        let cfg = GemmaVisionConfig {
            image_size: 896,
            patch_size: 16,
            num_patches: 56 * 56,
            tokens_per_crop: 784,
            crop_set: default_crop_set(),
        };
        let ps = pan_and_scan(&test_png_wh(3, 2), &cfg).unwrap();
        assert_eq!(ps.offsets, vec![0, 27, 55]);
        // Portrait rotates the grid: the kept crops are the three row anchors
        // (0,0), (1,0), (2,0) -> offsets 0, 1512, 3080.
        let ps_portrait = pan_and_scan(&test_png_wh(2, 3), &cfg).unwrap();
        assert_eq!(ps_portrait.offsets, vec![0, 1512, 3080]);
    }

    #[test]
    fn pan_and_scan_pans_across_the_long_axis() {
        // Portrait 2x3 px: resized to 32x48. The rotated 3x2 grid keeps the
        // three distinct row anchors (0, 8, 16); the short-axis (column)
        // duplicates are deduplicated, so every crop is pixel-distinct.
        let png = test_png_wh(2, 3);
        let cfg = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
            tokens_per_crop: 4,
            crop_set: default_crop_set(),
        };
        let ps = pan_and_scan(&png, &cfg).unwrap();
        assert_eq!(ps.crops.len(), 3);
        assert_eq!(ps.offsets.len(), 3);
        assert_ne!(ps.crops[0].as_slice::<f32>(), ps.crops[1].as_slice::<f32>());
        assert_ne!(ps.crops[1].as_slice::<f32>(), ps.crops[2].as_slice::<f32>());
        assert_ne!(ps.crops[0].as_slice::<f32>(), ps.crops[2].as_slice::<f32>());
    }

    #[test]
    fn pan_and_scan_normalizes_to_mean_half_std_half() {
        let cfg = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
            tokens_per_crop: 4,
            crop_set: default_crop_set(),
        };
        let ps = pan_and_scan(&test_png(), &cfg).unwrap();
        let flat = ps.crops[0].as_slice::<f32>();
        // Solid red 255,0,0 -> (255/255 - 0.5) / 0.5 = 1.0, and (0 - 0.5)/0.5 = -1.0.
        assert!((flat[0] - 1.0).abs() < 1e-4);
        assert!((flat[1] + 1.0).abs() < 1e-4);
        assert!((flat[2] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn pan_and_scan_rejects_invalid_bytes() {
        let cfg = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
            tokens_per_crop: 4,
            crop_set: default_crop_set(),
        };
        let err = pan_and_scan(b"not an image", &cfg).unwrap_err();
        assert!(matches!(err, VisionError::Decode(_)));
    }

    // -----------------------------------------------------------------------
    // expand_gemma_markers (Task 13 marker expansion)
    // -----------------------------------------------------------------------

    #[test]
    fn expand_gemma_markers_single_image_run() {
        // [text, start, end, text] with k=256 expands to
        // [text, start, -200 x 256, end, text].
        let mut tokens = vec![1u32, 2, 3, 4];
        expand_gemma_markers(&mut tokens, 2, 3, &[256]);
        assert_eq!(tokens.len(), 2 + 256 + 2);
        assert_eq!(tokens[0], 1);
        assert_eq!(tokens[1], 2);
        assert!(
            tokens[2..2 + 256]
                .iter()
                .all(|&t| t == IMAGE_TOKEN_INDEX as u32)
        );
        assert_eq!(tokens[2 + 256], 3);
        assert_eq!(tokens[2 + 256 + 1], 4);
    }

    #[test]
    fn expand_gemma_markers_multiple_images_in_order() {
        // Two images with different k: the second run consumes per_image_tokens[1].
        let mut tokens = vec![0u32, 2, 3, 5, 2, 3, 7];
        expand_gemma_markers(&mut tokens, 2, 3, &[4, 2]);
        // [0, 2, -200x4, 3, 5, 2, -200x2, 3, 7]
        assert_eq!(tokens.len(), 13);
        assert_eq!(tokens[0], 0);
        assert_eq!(tokens[1], 2);
        assert_eq!(tokens[2..6], vec![IMAGE_TOKEN_INDEX as u32; 4]);
        assert_eq!(tokens[6], 3);
        assert_eq!(tokens[7], 5);
        assert_eq!(tokens[8], 2);
        assert_eq!(tokens[9], IMAGE_TOKEN_INDEX as u32);
        assert_eq!(tokens[10], IMAGE_TOKEN_INDEX as u32);
        assert_eq!(tokens[11], 3);
        assert_eq!(tokens[12], 7);
    }

    #[test]
    fn expand_gemma_markers_no_markers_unchanged() {
        let mut tokens = vec![10u32, 11, 12];
        expand_gemma_markers(&mut tokens, 2, 3, &[256]);
        assert_eq!(tokens, vec![10, 11, 12]);
    }

    #[test]
    fn expand_gemma_markers_missing_k_falls_back_to_one() {
        // per_image_tokens exhausted -> each remaining run expands to a single
        // sentinel (the sketch's `unwrap_or(1)` behavior).
        let mut tokens = vec![2u32, 3, 2, 3];
        expand_gemma_markers(&mut tokens, 2, 3, &[1]);
        assert_eq!(
            tokens,
            vec![
                2,
                IMAGE_TOKEN_INDEX as u32,
                3,
                2,
                IMAGE_TOKEN_INDEX as u32,
                3
            ]
        );
    }

    // -----------------------------------------------------------------------
    // Pooled patch features (Task 13 per-crop token compression)
    // -----------------------------------------------------------------------

    #[test]
    fn pool_patch_features_2x2_means_blocks() {
        // [1, 4, 4, 2] grid, tokens_per_side 2 -> [1, 4, 2]. Each output row is
        // the mean of its 2x2 patch block; fill values so means are exact.
        let mut data = Vec::with_capacity(4 * 4 * 2);
        for r in 0..16 {
            let (i, j) = (r / 4, r % 4);
            // value = 10*i + j  (row, col), both channels identical
            let v = (10 * i + j) as f32;
            data.push(v);
            data.push(v);
        }
        let feats = Array::from_slice(&data, &[1, 4, 4, 2]);
        let pooled = pool_patch_features(&feats, 2).unwrap();
        assert_eq!(pooled.shape(), &[1, 4, 2]);
        mlx_rs::transforms::eval([&pooled]).unwrap();
        // Blocks (rows 0-1, cols 0-1): [0,1,10,11] -> 5.5;
        // (0-1, 2-3): [2,3,12,13] -> 7.5; (2-3, 0-1): [20,21,30,31] -> 25.5;
        // (2-3, 2-3): [22,23,32,33] -> 27.5. Both channels are identical.
        let vals = pooled.as_slice::<f32>();
        assert_eq!(vals, &[5.5, 5.5, 7.5, 7.5, 25.5, 25.5, 27.5, 27.5]);
    }

    #[test]
    fn pool_patch_features_4x4_kernel() {
        // [1, 8, 8, 1] with tokens_per_side 2 -> kernel 4 -> [1, 4, 1].
        let data: Vec<f32> = (0..64).map(|r| (r % 2) as f32).collect(); // row-parity blocks
        let feats = Array::from_slice(&data, &[1, 8, 8, 1]);
        let pooled = pool_patch_features(&feats, 2).unwrap();
        assert_eq!(pooled.shape(), &[1, 4, 1]);
    }

    #[test]
    fn pool_patch_features_rejects_non_divisible_grid() {
        let feats = Array::from_slice(&[0.0f32; 9 * 2], &[1, 3, 3, 2]);
        assert!(pool_patch_features(&feats, 2).is_err());
    }

    // -----------------------------------------------------------------------
    // tokens-per-crop math (Task 13 compression factor)
    // -----------------------------------------------------------------------

    #[test]
    fn gemma_tokens_per_crop_prefers_checkpoint_value() {
        // gemma-3-27b: 4096 patches (896/14)^2, mm_tokens_per_image 256.
        assert_eq!(gemma_tokens_per_crop(4096, Some(256)).unwrap(), 256);
        // gemma-3-4b: 4096 patches, mm_tokens_per_image 1024.
        assert_eq!(gemma_tokens_per_crop(4096, Some(1024)).unwrap(), 1024);
    }

    #[test]
    fn gemma_tokens_per_crop_falls_back_to_num_patches_div_4() {
        // (896/16)^2 = 3136 patches -> 4:1 compression -> 784.
        assert_eq!(gemma_tokens_per_crop(3136, None).unwrap(), 784);
    }

    #[test]
    fn gemma_tokens_per_crop_rejects_invalid_values() {
        // mm_tokens_per_image that is not a perfect square dividing the grid.
        assert!(gemma_tokens_per_crop(4096, Some(100)).is_err());
        // num_patches not divisible by 4 -> no 4:1 pooling fallback.
        assert!(gemma_tokens_per_crop(9, None).is_err());
        // num_patches that is not a perfect square at all.
        assert!(gemma_tokens_per_crop(10, None).is_err());
    }

    #[test]
    fn pan_and_scan_rejects_bad_config() {
        // patch_size must divide image_size.
        let cfg_undivisible = GemmaVisionConfig {
            image_size: 33,
            patch_size: 16,
            num_patches: 1,
            tokens_per_crop: 1,
            crop_set: default_crop_set(),
        };
        let err_undivisible = pan_and_scan(&test_png(), &cfg_undivisible).unwrap_err();
        assert!(matches!(err_undivisible, VisionError::Preprocess(_)));

        // The crop set must not be empty.
        let cfg_empty = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
            tokens_per_crop: 4,
            crop_set: vec![],
        };
        let err_empty = pan_and_scan(&test_png(), &cfg_empty).unwrap_err();
        assert!(matches!(err_empty, VisionError::Preprocess(_)));

        // Coordinates outside the {0, 1, 2} grid are rejected.
        let cfg_out_of_grid = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
            tokens_per_crop: 4,
            crop_set: vec![(3, 0)],
        };
        let err_out_of_grid = pan_and_scan(&test_png(), &cfg_out_of_grid).unwrap_err();
        assert!(matches!(err_out_of_grid, VisionError::Preprocess(_)));
    }

    // -----------------------------------------------------------------------
    // GemmaVisionTower pipeline (preprocess -> encode -> offsets)
    // -----------------------------------------------------------------------

    /// Tiny SigLIP config: image 16, patch 4 -> 16 patches, 1 layer.
    fn tiny_siglip_config() -> crate::siglip::SigLipVisionConfig {
        crate::siglip::SigLipVisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_channels: 3,
            patch_size: 4,
            image_size: 16,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu_pytorch_tanh".to_owned(),
        }
    }

    fn tiny_tower() -> GemmaVisionTower {
        let siglip = tiny_siglip_config();
        let config = GemmaVisionConfig {
            image_size: 16,
            patch_size: 4,
            num_patches: 16,
            tokens_per_crop: 4, // 2x2 pooling of the 4x4 patch grid
            crop_set: default_crop_set(),
        };
        GemmaVisionTower::new(
            config,
            crate::siglip::SigLipVisionModel::new(&siglip).unwrap(),
        )
    }

    #[test]
    fn tower_preprocess_expands_to_crops_times_tokens() {
        let tower = tiny_tower();
        let img = ImageInput {
            position: 0,
            message_index: 0,
            // 3x2 px landscape: resized to 24x16, the default 2x3 grid keeps
            // the three distinct column anchors (0, 4, 8) after dedup.
            bytes: test_png_wh(3, 2),
            media_type: "image/png".to_owned(),
            detail: crate::vision::ImageDetail::Auto,
            max_dims: None,
        };
        let batch = tower.preprocess_images(&[img]).unwrap();
        // 3 distinct crops x 4 tokens per crop.
        assert_eq!(batch.per_image_tokens, vec![12]);
        assert_eq!(batch.pixel_values.shape().first(), Some(&3));
        assert_eq!(batch.image_offsets.len(), 3);
        // Grid 4, landscape: kept crops (0,0), (0,1), (0,2) ->
        // offset_col=(c*3)/2 = 0, 1, 3.
        assert_eq!(batch.image_offsets, vec![0, 1, 3]);
    }

    #[test]
    fn tower_encode_pools_to_per_image_token_count() {
        let mut tower = tiny_tower();
        let img = ImageInput {
            position: 0,
            message_index: 0,
            bytes: test_png_wh(3, 2),
            media_type: "image/png".to_owned(),
            detail: crate::vision::ImageDetail::Auto,
            max_dims: None,
        };
        let batch = tower.preprocess_images(&[img]).unwrap();
        let features = tower.encode(&batch.pixel_values).unwrap();
        let expected_rows: usize = batch.per_image_tokens.iter().sum();
        assert_eq!(features.shape(), &[expected_rows as i32, 8]);
        mlx_rs::transforms::eval([&features]).unwrap();
        assert!(features.as_slice::<f32>().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tower_build_position_offsets_places_image_rows_at_crop_offsets() {
        let tower = tiny_tower();
        // [text, <12 image rows>, text] — one landscape image, 3 crops x 4.
        let mut ids = vec![1i32];
        ids.extend(std::iter::repeat_n(crate::vision::IMAGE_TOKEN_INDEX, 12));
        ids.push(2);
        // [1, L] like the engine's prompt array.
        let input_ids = Array::from_slice(&ids, &[1, 14]);

        let batch = ImageBatch {
            pixel_values: Array::from_slice(&[0.0f32; 3], &[1, 1, 1, 3]),
            per_image_tokens: vec![12],
            image_sizes: vec![(16, 16)],
            image_offsets: vec![0, 1, 3],
            layout: ImageTokenLayout::default(),
        };
        let offsets = tower.build_position_offsets(&input_ids, &batch).unwrap();
        mlx_rs::transforms::eval([&offsets]).unwrap();
        let vals: Vec<i32> = offsets.as_slice().to_vec();
        assert_eq!(vals.len(), 14);
        assert_eq!(vals[0], 0); // text keeps its natural position
        // Crop rows: crop c at offset o -> positions [o, o+1, o+2, o+3].
        let expected_rows: [i32; 12] = [
            0, 1, 2, 3, // crop 0 @ 0
            1, 2, 3, 4, // crop 1 @ 1
            3, 4, 5, 6, // crop 2 @ 3
        ];
        assert_eq!(&vals[1..13], &expected_rows);
        assert_eq!(vals[13], 13); // trailing text keeps its natural position
    }

    #[test]
    fn tower_build_position_offsets_rejects_malformed_batch() {
        let tower = tiny_tower();
        let input_ids = Array::from_slice(&[crate::vision::IMAGE_TOKEN_INDEX; 3], &[3]);
        let batch = ImageBatch {
            pixel_values: Array::from_slice(&[0.0f32; 3], &[1, 1, 1, 3]),
            per_image_tokens: vec![2], // not divisible by tokens_per_crop 4
            image_sizes: vec![(16, 16)],
            image_offsets: vec![0],
            layout: ImageTokenLayout::default(),
        };
        assert!(tower.build_position_offsets(&input_ids, &batch).is_err());
    }
}
