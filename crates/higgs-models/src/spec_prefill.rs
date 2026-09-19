//! Speculative / compressive prefill (PFlash) for higgs.
//!
//! Implements the *selection* half of SpecPrefill (Liu et al., arXiv:2502.02789)
//! — the algorithm Lucebox's PFlash uses verbatim. A small drafter (Qwen3-0.6B)
//! scores prompt-token importance; the heavy target then prefills only the kept
//! fraction, cutting target prefill FLOPs by ~1/keep_ratio.
//!
//! See `.planning/DESIGN-pflash-higgs.md` for the full design and
//! `docs/RESEARCH-pflash-prior-art.md` for prior art. Summary of the
//! SpecPrefill-Full-LAH recipe:
//!   1. Drafter forward + `lookahead` greedy decode; capture per-layer Q.
//!   2. Block-wise attention scoring (NEVER materialize `[H, S, S]` — see
//!      `SAFETY` below).
//!   3. `importance = mean_over_lookahead( max_over_(layers, heads)(attn) )`.
//!   4. 1D avgpool smoothing (`avgpool`, default 13).
//!   5. Chunk-top-K selection (`chunk`, default 32; `keep_ratio`, default 0.10).
//!   6. Restore original prompt positions on survivors (critical for NIAH).
//!
//! # SAFETY — the lesson from the probe crash
//!
//! A prior Python probe computed `Q @ K.T` as a full `[H, S, S]` tensor while
//! the Bonsai target was resident. At S=32K that is ~32 GB; the allocator OOM'd
//! and crashed the server.
//!
//! The scorer half (steps 1-3) MUST compute attention block-pair by block-pair:
//! one K-block of 128 at a time, producing a transient
//! `[lookahead, n_kv_heads, 128]` (~16 KB) and accumulating into a per-layer
//! `[lookahead, n_heads, S]` (~75 MB at S=128K), streamed to a running max so
//! peak is **~75 MB regardless of S**. S grows the accumulator linearly, never
//! quadratically.
//!
//! This module ships the model-free selection half (steps 4-6) plus the
//! S-linear per-layer attention primitive used by the dense drafter scorer.
//! Drafter orchestration lives in `transformer::Model::pflash_importance`.
//! Target-side sparse execution is implemented for Qwen3Next/Bonsai-hybrid
//! targets; dSpark/DFlash uses the compressed survivor sequence directly.

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]
mod tests;

/// Knobs for SpecPrefill selection. Defaults mirror the published recipe
/// (Cross-Family Appendix A.1: chunk=32, avgpool=13, lookahead=8) and the
/// "highest tradable point" keep_ratio=0.10 (RESEARCH §3.5).
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillScoreConfig {
    /// Fraction of source tokens kept after compression.
    pub keep_ratio: f32,
    /// Prompt-token block size for survivor selection.
    pub chunk: usize,
    /// 1D avgpool smoothing kernel width.
    pub avgpool: usize,
    /// Lookahead decoded tokens used for importance aggregation.
    pub lookahead: usize,
}

impl Default for PrefillScoreConfig {
    fn default() -> Self {
        Self {
            keep_ratio: 0.10,
            chunk: 32,
            avgpool: 13,
            lookahead: 8,
        }
    }
}

/// Map scorer uncertainty to an effective keep ratio.
///
/// The scorer produces a non-negative importance distribution over source
/// tokens. When that distribution is sharp, the prompt has a small number of
/// clear anchors and PFlash can compress aggressively. When it is diffuse, the
/// prompt is harder to summarize safely, so keep more of it.
#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
#[must_use]
pub fn adaptive_keep_ratio_from_importance(
    importance: &[f32],
    keep_floor: f32,
    keep_ceiling: f32,
) -> f32 {
    let floor = keep_floor.clamp(0.02, 0.95);
    let ceiling = keep_ceiling.clamp(floor, 0.95);
    if importance.len() < 2 || (ceiling - floor).abs() <= f32::EPSILON {
        return floor;
    }

    let finite_positive: Vec<f64> = importance
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(f64::from)
        .collect();
    let total: f64 = finite_positive.iter().sum();
    if total <= f64::EPSILON {
        return ceiling;
    }

    let entropy_nats = finite_positive.iter().fold(0.0_f64, |acc, score| {
        let p = *score / total;
        if p <= 0.0 { acc } else { acc - p * p.ln() }
    });
    let max_entropy = (importance.len() as f64).ln().max(f64::EPSILON);
    let normalized_entropy = (entropy_nats / max_entropy).clamp(0.0, 1.0) as f32;
    // Long real prompts naturally have high entropy even when the scorer still
    // provides usable anchors. A linear map therefore collapses the policy into
    // "mostly keep the ceiling" and erases the prefill win. Treat entropy below
    // 0.80 as compressible, then ramp quadratically only for truly diffuse
    // scorer output.
    let diffuse_pressure = ((normalized_entropy - 0.80) / 0.20).clamp(0.0, 1.0);
    floor + (ceiling - floor) * diffuse_pressure * diffuse_pressure
}

impl Default for PrefillScoreMode {
    fn default() -> Self {
        Self::Full
    }
}

/// Scorer variant used to produce a survivor plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrefillScoreMode {
    /// Full drafter scorer: prompt forward plus lookahead through the whole drafter.
    Full,
    /// Early-exit scorer at an intermediate drafter layer.
    L7,
}

/// Metadata that makes a compressed prefill plan safe to cache or reject.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillPlanMetadata {
    pub version: u32,
    pub score_mode: PrefillScoreMode,
    pub exit_layer: Option<usize>,
    pub keep_ratio: f32,
    pub chunk: usize,
    pub avgpool: usize,
    pub lookahead: usize,
}

impl PrefillPlanMetadata {
    pub const VERSION: u32 = 1;

    #[must_use]
    pub fn from_config(cfg: &PrefillScoreConfig) -> Self {
        Self {
            version: Self::VERSION,
            score_mode: PrefillScoreMode::Full,
            exit_layer: None,
            keep_ratio: cfg.keep_ratio,
            chunk: cfg.chunk,
            avgpool: cfg.avgpool,
            lookahead: cfg.lookahead,
        }
    }
}

/// A survivor plan: the kept token ids in their original order, plus the
/// **original prompt position** of each survivor (for RoPE position-id restore —
/// SpecPrefill §3.2.4; critical for NIAH and counting tasks).
#[derive(Debug, Clone, PartialEq)]
pub struct SurvivalPlan {
    pub token_ids: Vec<u32>,
    pub original_positions: Vec<i32>,
    pub source_token_count: usize,
    pub metadata: PrefillPlanMetadata,
}

/// Borrowed, validated target-prefill view of a [`SurvivalPlan`].
///
/// `logical_next_pos` is the absolute prompt position the next decoded token
/// must use. For lossy compression this is the source prompt length, not the
/// number of retained survivor rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetSparsePrefillPlan<'a> {
    pub token_ids: &'a [u32],
    pub original_positions: &'a [i32],
    pub logical_next_pos: i32,
}

/// Half-open source-token span that must survive PFlash selection exactly.
///
/// These spans carry executable prompt contracts (tool schemas, current action
/// tail) that are cheap enough to keep verbatim and expensive to corrupt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardKeepSpan {
    pub start: usize,
    pub end: usize,
}

impl HardKeepSpan {
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

impl TargetSparsePrefillPlan<'_> {
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    #[must_use]
    pub fn is_contiguous_identity(&self) -> bool {
        self.len() == usize::try_from(self.logical_next_pos).unwrap_or(usize::MAX)
            && self
                .original_positions
                .iter()
                .enumerate()
                .all(|(index, position)| *position == index as i32)
    }
}

impl SurvivalPlan {
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    #[must_use]
    pub fn with_scorer(mut self, score_mode: PrefillScoreMode, exit_layer: Option<usize>) -> Self {
        self.metadata.score_mode = score_mode;
        self.metadata.exit_layer = exit_layer;
        self
    }

    /// Build an identity/no-compression survivor plan. Useful when appending a
    /// small new turn to a cached compressed prefix: the old prefix stays
    /// compressed, while the new suffix remains exact.
    pub fn identity(tokens: &[u32], metadata: PrefillPlanMetadata) -> Result<Self, String> {
        let mut original_positions = Vec::with_capacity(tokens.len());
        for index in 0..tokens.len() {
            let position = i32::try_from(index).map_err(|_| {
                format!("SurvivalPlan::identity: token index {index} exceeds i32::MAX")
            })?;
            original_positions.push(position);
        }
        let plan = Self {
            token_ids: tokens.to_vec(),
            original_positions,
            source_token_count: tokens.len(),
            metadata,
        };
        plan.target_sparse_prefill_plan()?;
        Ok(plan)
    }

    /// Append a suffix plan whose original positions are relative to the suffix
    /// source. The returned plan's positions are relative to the concatenated
    /// source, preserving the frozen prefix survivor mapping across turns.
    pub fn append_suffix(
        &self,
        suffix: &Self,
        metadata: PrefillPlanMetadata,
    ) -> Result<Self, String> {
        let suffix_offset = self.source_token_count;
        let mut token_ids = Vec::with_capacity(self.token_ids.len() + suffix.token_ids.len());
        token_ids.extend_from_slice(&self.token_ids);
        token_ids.extend_from_slice(&suffix.token_ids);

        let mut original_positions =
            Vec::with_capacity(self.original_positions.len() + suffix.original_positions.len());
        original_positions.extend_from_slice(&self.original_positions);
        for &position in &suffix.original_positions {
            if position < 0 {
                return Err(format!(
                    "SurvivalPlan::append_suffix: negative suffix position {position}"
                ));
            }
            let suffix_position = usize::try_from(position).map_err(|_| {
                format!("SurvivalPlan::append_suffix: invalid suffix position {position}")
            })?;
            if suffix_position >= suffix.source_token_count {
                return Err(format!(
                    "SurvivalPlan::append_suffix: suffix position {position} outside source length {}",
                    suffix.source_token_count
                ));
            }
            let combined = suffix_offset
                .checked_add(suffix_position)
                .ok_or_else(|| "SurvivalPlan::append_suffix: position overflow".to_owned())?;
            let combined_i32 = i32::try_from(combined).map_err(|_| {
                format!(
                    "SurvivalPlan::append_suffix: combined position {combined} exceeds i32::MAX"
                )
            })?;
            original_positions.push(combined_i32);
        }

        let source_token_count = self
            .source_token_count
            .checked_add(suffix.source_token_count)
            .ok_or_else(|| "SurvivalPlan::append_suffix: source length overflow".to_owned())?;
        let plan = Self {
            token_ids,
            original_positions,
            source_token_count,
            metadata,
        };
        plan.target_sparse_prefill_plan()?;
        Ok(plan)
    }

    /// True only when the plan is a no-op contiguous prefill. Any lossy plan
    /// must go through a sparse-prefill path that applies RoPE at
    /// `original_positions` and advances decode from `source_token_count`.
    #[must_use]
    pub fn is_contiguous_identity(&self) -> bool {
        self.token_ids.len() == self.source_token_count
            && self.original_positions.len() == self.source_token_count
            && self
                .original_positions
                .iter()
                .enumerate()
                .all(|(index, position)| *position == index as i32)
    }

    pub fn target_sparse_prefill_plan(&self) -> Result<TargetSparsePrefillPlan<'_>, String> {
        if self.token_ids.len() != self.original_positions.len() {
            return Err(format!(
                "target_sparse_prefill_plan: token_ids ({}) and original_positions ({}) length mismatch",
                self.token_ids.len(),
                self.original_positions.len()
            ));
        }
        let logical_next_pos = i32::try_from(self.source_token_count).map_err(|_| {
            format!(
                "target_sparse_prefill_plan: source_token_count {} exceeds i32::MAX",
                self.source_token_count
            )
        })?;
        if self.source_token_count == 0 {
            if self.token_ids.is_empty() {
                return Ok(TargetSparsePrefillPlan {
                    token_ids: &self.token_ids,
                    original_positions: &self.original_positions,
                    logical_next_pos,
                });
            }
            return Err(
                "target_sparse_prefill_plan: non-empty survivors for empty source".to_owned(),
            );
        }
        if self.token_ids.is_empty() {
            return Err(
                "target_sparse_prefill_plan: non-empty source must retain at least one token"
                    .to_owned(),
            );
        }
        let mut previous = None;
        for &position in &self.original_positions {
            if position < 0 {
                return Err(format!(
                    "target_sparse_prefill_plan: negative original position {position}"
                ));
            }
            if usize::try_from(position).unwrap_or(usize::MAX) >= self.source_token_count {
                return Err(format!(
                    "target_sparse_prefill_plan: original position {position} outside source length {}",
                    self.source_token_count
                ));
            }
            if previous.is_some_and(|prev| position <= prev) {
                return Err(
                    "target_sparse_prefill_plan: original positions must be strictly increasing"
                        .to_owned(),
                );
            }
            previous = Some(position);
        }
        if self.original_positions.first().copied() != Some(0) {
            return Err(
                "target_sparse_prefill_plan: first source token at position 0 must be retained"
                    .to_owned(),
            );
        }
        let final_source_pos = i32::try_from(self.source_token_count - 1).map_err(|_| {
            format!(
                "target_sparse_prefill_plan: source_token_count {} exceeds i32::MAX",
                self.source_token_count
            )
        })?;
        if self.original_positions.last().copied() != Some(final_source_pos) {
            return Err(format!(
                "target_sparse_prefill_plan: final source token at position {final_source_pos} must be retained"
            ));
        }
        Ok(TargetSparsePrefillPlan {
            token_ids: &self.token_ids,
            original_positions: &self.original_positions,
            logical_next_pos,
        })
    }
}

/// 1D average-pool smoothing of a per-token importance vector (step 4).
///
/// Uses a **shrinking window at the edges**: samples near the start/end average
/// over fewer than `kernel` neighbors (no reflect/replicate padding). This is
/// intentional — the sink block (token 0) is force-kept by `select_survivors`
/// regardless of its smoothed score, so edge under-weighting does not lose it.
/// `kernel` must be odd and >= 1 so the window is symmetric around each sample
/// (the published kernel=13 is odd).
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn smooth_importance(importance: &[f32], kernel: usize) -> Result<Vec<f32>, String> {
    if importance.is_empty() {
        return Ok(Vec::new());
    }
    if kernel == 0 || kernel % 2 == 0 {
        return Err(format!(
            "smooth_importance: kernel must be odd and >= 1, got {kernel}"
        ));
    }
    let half = kernel / 2;
    let n = importance.len();
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let cnt = (hi - lo) as f32;
        let sum: f32 = importance[lo..hi].iter().sum();
        out[i] = sum / cnt;
    }
    Ok(out)
}

/// Select survivors from a smoothed importance vector (steps 5-6).
///
/// Chunks the prompt into `chunk`-sized blocks, scores each block by the max
/// smoothed importance it contains, and keeps the top `keep_ratio` fraction of
/// blocks. The first block (sink / system-prompt anchor) and the block holding
/// the final prompt token (whose logits the target samples from) are always
/// kept — SpecPrefill's sink convention.
///
/// Returns the kept token ids (a subsequence of `tokens`, original order) and
/// each survivor's original prompt position. Token-level positions are restored
/// verbatim (Option A in DESIGN §2.6); the target applies RoPE at these original
/// positions even though its KV cache stores only `M = survivors.len()` rows.
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn select_survivors(
    tokens: &[u32],
    importance: &[f32],
    cfg: &PrefillScoreConfig,
) -> Result<SurvivalPlan, String> {
    select_survivors_with_hard_keep(tokens, importance, cfg, &[])
}

/// Select survivors while forcing exact retention of caller-provided spans.
///
/// Block selection still controls the lossy budget for ordinary context. Hard
/// spans are applied at token granularity so preserving a tool name or the
/// current user turn does not force an entire neighboring chunk to survive.
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn select_survivors_with_hard_keep(
    tokens: &[u32],
    importance: &[f32],
    cfg: &PrefillScoreConfig,
    hard_keep_spans: &[HardKeepSpan],
) -> Result<SurvivalPlan, String> {
    if tokens.len() != importance.len() {
        return Err(format!(
            "select_survivors: tokens ({}) and importance ({}) length mismatch",
            tokens.len(),
            importance.len()
        ));
    }
    if tokens.is_empty() {
        return Ok(SurvivalPlan {
            token_ids: Vec::new(),
            original_positions: Vec::new(),
            source_token_count: 0,
            metadata: PrefillPlanMetadata::from_config(cfg),
        });
    }
    if !(0.02..=0.95).contains(&cfg.keep_ratio) {
        return Err(format!(
            "select_survivors: keep_ratio {} out of range [0.02, 0.95]",
            cfg.keep_ratio
        ));
    }
    if cfg.chunk == 0 {
        return Err("select_survivors: chunk must be >= 1".to_string());
    }
    for span in hard_keep_spans {
        if span.start > span.end || span.end > tokens.len() {
            return Err(format!(
                "select_survivors: hard keep span {}..{} outside source length {}",
                span.start,
                span.end,
                tokens.len()
            ));
        }
    }
    let smoothed = smooth_importance(importance, cfg.avgpool)?;

    let s = tokens.len();
    let n_blocks = s.div_ceil(cfg.chunk);
    // Block score = mean smoothed importance over the block (SpecPrefill §3.2.3:
    // "average importance within each chunk"). A multi-token salient span (a real
    // needle sentence) elevates the whole block, so mean ranks it correctly while
    // damping single-token noise spikes that `max` would over-promote.
    let mut block_score: Vec<(usize, f32)> = (0..n_blocks)
        .map(|b| {
            let lo = b * cfg.chunk;
            let hi = lo + cfg.chunk.min(s - lo);
            let sum: f32 = smoothed[lo..hi].iter().sum();
            (b, sum / (hi - lo) as f32)
        })
        .collect();
    // Stable descending sort by score: ties keep lower block index (earlier text).
    block_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let keep = (cfg.keep_ratio * n_blocks as f32).round() as usize;
    let keep = keep.clamp(1, n_blocks);
    let last_block = n_blocks - 1;
    let mut kept: Vec<bool> = vec![false; n_blocks];
    for &(b, _) in block_score.iter().take(keep) {
        kept[b] = true;
    }
    // Sink (block 0) + final-token block are mandatory.
    kept[0] = true;
    kept[last_block] = true;

    let mut hard_keep = vec![false; s];
    for span in hard_keep_spans {
        for keep in &mut hard_keep[span.start..span.end] {
            *keep = true;
        }
    }

    let mut token_ids = Vec::new();
    let mut original_positions = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        let b = i / cfg.chunk;
        if !kept[b] && !hard_keep[i] {
            continue;
        }
        token_ids.push(token);
        original_positions.push(i as i32);
    }
    Ok(SurvivalPlan {
        token_ids,
        original_positions,
        source_token_count: tokens.len(),
        metadata: PrefillPlanMetadata::from_config(cfg),
    })
}

// ---------------------------------------------------------------------------
// Scorer half (steps 1-3, the mlx-rs-heavy part).
// ---------------------------------------------------------------------------

use mlx_rs::ops;
use mlx_rs::{Array, error::Exception};

/// Per-layer importance contribution from the lookahead queries (step 2-3).
///
/// `q_lah` is `[n_heads, lah, head_dim]` — the post-RoPE/norm queries at the
/// `lah` lookahead positions for one drafter layer. `k` is
/// `[n_kv_heads, S, head_dim]` — the post-RoPE/norm keys over the prompt.
///
/// Returns `importance: [S] = mean_over_lah( max_over_heads( softmax(Q·K^T) ) )`.
///
/// # Memory safety (the lesson from the probe crash)
///
/// The attention tensor here is `[n_heads, lah, S]` — **S-linear, not S²**,
/// because queries are only the `lah=8` lookahead positions, not all prompt
/// positions. At `S = 128K`, `n_heads = 16`, `lah = 8`, f32: ~64 MB. The crash
/// came from running uncached full forwards (unbounded lazy graph) and from
/// treating all prompt tokens as queries; this function does neither. No
/// `[H, S, S]` is ever materialized.
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
pub fn layer_importance(
    q_lah: &Array,
    k: &Array,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
) -> Result<Array, Exception> {
    let k_shape = k.shape();
    let s = *k_shape.get(1).ok_or_else(|| {
        Exception::custom("layer_importance: k must be [n_kv_heads, S, head_dim]")
    })?;
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(Exception::custom(format!(
            "layer_importance: n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads}"
        )));
    }
    let group = n_heads / n_kv_heads;

    // Scale the queries (equivalent to scaling the scores; broadcasts a scalar
    // array — mlx-rs has no Mul<f32> overload, so go through ops::multiply).
    let q_scaled = ops::multiply(q_lah, &Array::from_f32(scale))?;

    // GQA: expand keys from [n_kv_heads, S, d] to [n_heads, S, d] by repeating
    // each kv head `group` times (matches the Qwen3 attention's head mapping).
    let k_expanded = if group == 1 {
        k.clone()
    } else {
        ops::broadcast_to(
            &k.reshape(&[n_kv_heads, 1, s, head_dim])?,
            &[n_kv_heads, group, s, head_dim],
        )?
        .reshape(&[n_heads, s, head_dim])?
    };
    let k_t = k_expanded.transpose_axes(&[0, 2, 1])?; // [n_heads, head_dim, S]

    // scores = q_scaled @ k_t -> [n_heads, lah, S]
    let scores = q_scaled.matmul(&k_t)?;
    let attn = ops::softmax_axis(&scores, -1, true)?;

    // importance = max over heads, then mean over lah -> [S].
    let max_h = ops::max_axis(&attn, 0, None)?;
    let importance = ops::mean_axis(&max_h, 0, None)?;
    Ok(importance)
}

// Internally (full signature once the drafter-forward capture lands):
//   pub fn score_prompt(
//       drafter: &AnyModel,
//       tokens: &[u32],
//       cfg: &PrefillScoreConfig,
//   ) -> Result<Vec<f32>, Exception> { ... }
//
// Qwen3-0.6B attention access (confirmed):
//   transformer::Attention exposes q_proj / k_proj (MaybeQuantized<nn::Linear>),
//   q_norm / k_norm (Option<RmsNorm>), rope (nn::Rope) — see
//   crates/higgs-models/src/transformer.rs:163-186.
