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
//!
//! The default crop set is the plan's 3x2 grid: 4 corners + 2 vertical
//! centers, `[(0,0), (0,1), (1,0), (1,1), (0,2), (1,2)]`.

use mlx_rs::Array;

use crate::vision::VisionError;

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
}

/// Default pan-and-scan crop set: `(row_frac, col_frac)` over a 3x2 grid.
///
/// Four corners plus the two vertical centers. The exact set is
/// checkpoint-specific; `vision_config.crop_size` or the reference gemma3
/// processor defines it.
pub fn default_crop_set() -> Vec<(i32, i32)> {
    vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)]
}

/// The output of [`pan_and_scan`]: one normalized crop per entry of
/// `config.crop_set`, plus the matching positional offsets.
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
/// `config.image_size`, crops one `image_size²` window per entry of
/// `config.crop_set` (anchored per the module docs), and normalizes each crop
/// to `[1, image_size, image_size, 3]`.
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
    let mut crops = Vec::with_capacity(config.crop_set.len());
    let mut offsets = Vec::with_capacity(config.crop_set.len());
    for &(row, col) in &config.crop_set {
        // Offsets in patch units along each axis — the plan's sketch verbatim
        // (truncating integer division).
        let offset_row = (row * (grid - 1)) / 2;
        let offset_col = (col * (grid - 1)) / 2;
        offsets.push(offset_row * grid + offset_col);

        let crop = crop_square(&resized, row, col, target);
        crops.push(to_normalized_array(crop, target));
    }

    Ok(PanAndScan { crops, offsets })
}

/// Extract the `target x target` RGB window for crop-set coordinate `(row,
/// col)`: top-left corner at `anchor = round((coord / 2) * (span - target))`
/// clamped to `[0, span - target]` per axis (see the module docs).
///
/// `span >= target` on both axes is guaranteed by the shorter-side resize in
/// [`pan_and_scan`], so the window always fits.
fn crop_square(rgb: &image::RgbImage, row: i32, col: i32, target: u32) -> image::RgbImage {
    let (w, h) = rgb.dimensions();
    let anchor = |coord: i32, span: u32| -> u32 {
        let span_i = i64::from(span);
        let target_i = i64::from(target);
        let delta = span_i.saturating_sub(target_i); // >= 0 under shorter-side resize
        // round-half-up of (coord * delta) / 2, then clamp to [0, delta].
        let raw = (i64::from(coord) * delta + 1) / 2;
        let clamped = raw.clamp(0, delta);
        u32::try_from(clamped).unwrap_or(u32::MAX)
    };
    let x = anchor(col, w);
    let y = anchor(row, h);
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
    fn pan_and_scan_produces_expected_crop_count() {
        // config: image_size 896, patch_size 16 -> 6 crops (3x2 pan-and-scan grid)
        let cfg = GemmaVisionConfig {
            image_size: 896,
            patch_size: 16,
            num_patches: 56 * 56,
            crop_set: default_crop_set(),
        };
        let ps = pan_and_scan(&test_png(), &cfg).unwrap();
        assert_eq!(ps.crops.len(), cfg.crop_set.len());
        assert_eq!(ps.offsets.len(), cfg.crop_set.len());
        assert_eq!(ps.crops[0].shape(), &[1, 896, 896, 3]);
    }

    #[test]
    fn pan_and_scan_small_config_is_fast_and_shaped() {
        let cfg = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
            crop_set: default_crop_set(),
        };
        let ps = pan_and_scan(&test_png(), &cfg).unwrap();
        assert_eq!(ps.crops.len(), 6);
        assert_eq!(ps.offsets.len(), 6);
        assert_eq!(ps.crops[0].shape(), &[1, 32, 32, 3]);
    }

    #[test]
    fn pan_and_scan_offsets_match_sketch_formula() {
        // grid = 56: offset_row = (r * 55) / 2, offset_col = (c * 55) / 2,
        // offset = offset_row * 56 + offset_col, for the default 3x2 crop set.
        let cfg = GemmaVisionConfig {
            image_size: 896,
            patch_size: 16,
            num_patches: 56 * 56,
            crop_set: default_crop_set(),
        };
        let ps = pan_and_scan(&test_png(), &cfg).unwrap();
        assert_eq!(ps.offsets, vec![0, 27, 1512, 1539, 55, 1567]);
    }

    #[test]
    fn pan_and_scan_pans_across_the_long_axis() {
        // Portrait 2x3 px: the shorter side (w=2) is resized to 32 -> 32x48,
        // so row anchors 0 and 8 (frac 0 vs 0.5) yield different crops, while
        // the column anchors coincide (w == target). crops[0] = (0,0) at
        // anchor (0,0); crops[2] = (1,0) at anchor (8,0).
        let png = test_png_wh(2, 3);
        let cfg = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
            crop_set: default_crop_set(),
        };
        let ps = pan_and_scan(&png, &cfg).unwrap();
        assert_ne!(ps.crops[0].as_slice::<f32>(), ps.crops[2].as_slice::<f32>());
        // (0,1) shares (0,0)'s anchor on the degenerate (short) axis.
        assert_eq!(ps.crops[0].as_slice::<f32>(), ps.crops[1].as_slice::<f32>());
    }

    #[test]
    fn pan_and_scan_normalizes_to_mean_half_std_half() {
        let cfg = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
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
            crop_set: default_crop_set(),
        };
        let err = pan_and_scan(b"not an image", &cfg).unwrap_err();
        assert!(matches!(err, VisionError::Decode(_)));
    }

    #[test]
    fn pan_and_scan_rejects_bad_config() {
        // patch_size must divide image_size.
        let cfg_undivisible = GemmaVisionConfig {
            image_size: 33,
            patch_size: 16,
            num_patches: 1,
            crop_set: default_crop_set(),
        };
        let err_undivisible = pan_and_scan(&test_png(), &cfg_undivisible).unwrap_err();
        assert!(matches!(err_undivisible, VisionError::Preprocess(_)));

        // The crop set must not be empty.
        let cfg_empty = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
            crop_set: vec![],
        };
        let err_empty = pan_and_scan(&test_png(), &cfg_empty).unwrap_err();
        assert!(matches!(err_empty, VisionError::Preprocess(_)));

        // Coordinates outside the {0, 1, 2} grid are rejected.
        let cfg_out_of_grid = GemmaVisionConfig {
            image_size: 32,
            patch_size: 8,
            num_patches: 4 * 4,
            crop_set: vec![(3, 0)],
        };
        let err_out_of_grid = pan_and_scan(&test_png(), &cfg_out_of_grid).unwrap_err();
        assert!(matches!(err_out_of_grid, VisionError::Preprocess(_)));
    }
}
