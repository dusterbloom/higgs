//! Clover `LowConfidence` block-diffusion decoding.
//!
//! Mirrors `DiffuSpace`'s reference `CloverMoeModelLM.generate` and
//! `sampling_clover_moe.py`: each block starts all-mask, is refined by up to
//! `block_size` preview passes (cache read-only), and every pass unmasks the
//! pending positions whose confidence exceeds the threshold (else the single
//! most confident one). Only the first `remaining` tokens of a resolved block
//! are returned, generation stops at the first EOS in positional order, and
//! the returned block is committed causally only when generation continues.
//!
//! Response penalties count committed response tokens only: tokens still in
//! the block being resolved never penalise themselves or other slots.

use std::path::Path;

use mlx_rs::{Array, error::Exception, ops};

use crate::mlx_exec::eval;

use crate::{AnyCache, AnyModel, SamplingParams};

/// Response penalties and stopping ramps (`generation_config.json` keys of the
/// Clover pack). Zero / one values disable the corresponding term.
#[derive(Debug, Clone, PartialEq)]
pub struct CloverPenalties {
    pub presence: f32,
    pub frequency: f32,
    pub repetition: f32,
    /// Bias added to the stop ids, ramped in from `length_start` over `length_window`.
    pub length_penalty: f32,
    pub length_start: u32,
    pub length_window: u32,
    /// Bias added to `think_end_token` until it is committed or visible in the block.
    pub think_end_bias: f32,
    pub think_end_start: u32,
    pub think_end_window: u32,
    pub think_end_token: Option<u32>,
}

impl Default for CloverPenalties {
    /// The reference `PENALTY_DEFAULTS`: every term disabled.
    fn default() -> Self {
        Self {
            presence: 0.0,
            frequency: 0.0,
            repetition: 1.0,
            length_penalty: 0.0,
            length_start: 8192,
            length_window: 8192,
            think_end_bias: 0.0,
            think_end_start: 4096,
            think_end_window: 4096,
            think_end_token: None,
        }
    }
}

impl CloverPenalties {
    /// Pack defaults from `generation_config.json`; absent keys keep the
    /// reference defaults (disabled).
    pub fn from_generation_config(value: &serde_json::Value) -> Self {
        let d = Self::default();
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        let f = |key: &str, default: f32| {
            value
                .get(key)
                .and_then(serde_json::Value::as_f64)
                .map_or(default, |v| v as f32)
        };
        let u = |key: &str, default: u32| {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(default)
        };
        Self {
            presence: f("presence_penalty", d.presence),
            frequency: f("frequency_penalty", d.frequency),
            repetition: f("repetition_penalty", d.repetition),
            length_penalty: f("dllm_length_penalty", d.length_penalty),
            length_start: u("dllm_length_penalty_start", d.length_start),
            length_window: u("dllm_length_penalty_window", d.length_window).max(1),
            think_end_bias: f("dllm_think_end_bias", d.think_end_bias),
            think_end_start: u("dllm_think_end_start", d.think_end_start),
            think_end_window: u("dllm_think_end_window", d.think_end_window).max(1),
            think_end_token: value
                .get("dllm_think_end_token_id")
                .and_then(serde_json::Value::as_u64)
                .and_then(|v| u32::try_from(v).ok()),
        }
    }

    /// Read `<model_dir>/generation_config.json`; missing or unreadable -> defaults.
    pub fn from_model_dir(model_dir: &Path) -> Self {
        std::fs::read_to_string(model_dir.join("generation_config.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .map_or_else(Self::default, |value| Self::from_generation_config(&value))
    }

    #[allow(clippy::float_cmp)]
    fn active(&self) -> bool {
        self.presence != 0.0
            || self.frequency != 0.0
            || self.repetition != 1.0
            || self.length_penalty != 0.0
            || self.think_end_bias != 0.0
    }
}

/// One block-diffusion request.
#[derive(Debug, Clone)]
pub struct BlockDecodeConfig {
    /// Preview block length; must divide the trained block size.
    pub block_size: usize,
    /// `LowConfidence` unmask threshold in `[0, 1]`.
    pub threshold: f32,
    pub mask_token_id: u32,
    pub max_new_tokens: usize,
    /// Generation ends after the first of these in emitted order.
    pub eos_token_ids: Vec<u32>,
    /// Pad id; joins the EOS ids as a target of the length ramp.
    pub pad_token_id: Option<u32>,
    pub penalties: CloverPenalties,
    /// `temperature == 0` is greedy argmax; otherwise [`crate::sample`] draws
    /// the prediction. Confidence always uses the T=1 (penalised) logits.
    pub sampling: SamplingParams,
}

/// The two model operations the block loop needs.
pub trait BlockModel {
    /// Logits for every block position (`[1, S, V]` or `[S, V]`); the
    /// committed state must not change.
    fn preview(&mut self, block: &[u32]) -> Result<Array, Exception>;
    /// Append `tokens` to the committed (causal) state.
    fn commit(&mut self, tokens: &[u32]) -> Result<(), Exception>;
}

/// [`BlockModel`] over a loaded model and its prefilled cache.
pub struct AnyModelBlocks<'a> {
    pub model: &'a mut AnyModel,
    pub cache: &'a mut AnyCache,
}

fn row(tokens: &[u32]) -> Result<Array, Exception> {
    let n = i32::try_from(tokens.len()).map_err(|_| Exception::custom("block too long"))?;
    Ok(Array::from_slice(tokens, &[1, n]))
}

impl BlockModel for AnyModelBlocks<'_> {
    fn preview(&mut self, block: &[u32]) -> Result<Array, Exception> {
        self.model.forward_preview_block(&row(block)?, self.cache)
    }

    /// The cache is request-local (diffusion mode never retains it), so the
    /// commit takes the fast multi-row kernels.
    fn commit(&mut self, tokens: &[u32]) -> Result<(), Exception> {
        let hidden = crate::qwen3_next::with_fast_rows(|| {
            self.model.forward_hidden(&row(tokens)?, None, self.cache)
        })?;
        let mut targets = vec![&hidden];
        targets.extend(self.cache.eval_targets());
        eval(targets)
    }
}

/// Result of [`block_decode`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockDecodeOutput {
    pub tokens: Vec<u32>,
    /// Preview passes spent on each block.
    pub passes_per_block: Vec<usize>,
    /// Ended on an EOS token.
    pub hit_eos: bool,
    /// Ended because `on_block` asked to stop.
    pub stopped_by_caller: bool,
}

/// Committed-response penalty state (`CloverPenaltyState`).
struct PenaltyState {
    p: CloverPenalties,
    active: bool,
    vocab: i32,
    /// Committed response length.
    length: usize,
    /// `[V]` f32 counts of committed response tokens.
    counts: Array,
    /// `[V]` bool, `counts > 0`.
    seen: Array,
    committed_close: bool,
    /// `[1, V]` f32 indicator of the length-ramp stop ids.
    stop_columns: Option<Array>,
    /// `[1, V]` f32 indicator of the think-end token.
    close_column: Option<Array>,
}

/// `clamp((positions - start) / window, 0, 1) * strength` for the block's
/// response positions `length + 1 ..= length + rows`, as an `[S, 1]` column.
// fp32 like the reference: positions stay exact far beyond any context length.
#[allow(clippy::cast_precision_loss, clippy::as_conversions)]
fn ramp(length: usize, rows: usize, start: u32, window: u32, strength: f32) -> Array {
    let values: Vec<f32> = (1..=rows)
        .map(|i| {
            let position = (length + i) as f32;
            ((position - start as f32) / window as f32).clamp(0.0, 1.0) * strength
        })
        .collect();
    Array::from_slice(&values, &[i32::try_from(rows).unwrap_or(i32::MAX), 1])
}

impl PenaltyState {
    fn new(p: CloverPenalties, vocab: i32, stop_token_ids: &[u32]) -> Result<Self, Exception> {
        let active = p.active();
        let mut stop_ids: Vec<u32> = stop_token_ids
            .iter()
            .copied()
            .filter(|&i| i64::from(i) < i64::from(vocab))
            .collect();
        stop_ids.sort_unstable();
        stop_ids.dedup();
        let stop_columns = (active && p.length_penalty != 0.0 && !stop_ids.is_empty())
            .then(|| indicator_counts(vocab, &stop_ids))
            .transpose()?;
        let close_column = match p.think_end_token {
            Some(close) if active && p.think_end_bias != 0.0 => {
                Some(indicator_counts(vocab, &[close])?)
            }
            _ => None,
        };
        let counts = ops::zeros::<f32>(&[vocab])?;
        let seen = counts.gt(Array::from_f32(0.0))?;
        Ok(Self {
            p,
            active,
            vocab,
            length: 0,
            counts,
            seen,
            committed_close: false,
            stop_columns,
            close_column,
        })
    }

    /// Count each returned response token once, after full block resolution.
    fn commit(&mut self, tokens: &[u32]) -> Result<(), Exception> {
        self.length += tokens.len();
        if !self.active || tokens.is_empty() {
            return Ok(());
        }
        self.committed_close |= self
            .p
            .think_end_token
            .is_some_and(|close| tokens.contains(&close));
        self.counts = self.counts.add(indicator_counts(self.vocab, tokens)?)?;
        self.seen = self.counts.gt(Array::from_f32(0.0))?;
        eval([&self.counts, &self.seen])
    }

    /// Apply the reference's operation order to fresh `[S, V]` f32 logits:
    /// frequency, presence, repetition, length ramp, think-end ramp.
    #[allow(clippy::float_cmp)]
    fn apply(&self, fresh: Array, block: &[u32], mask_id: u32) -> Result<Array, Exception> {
        if !self.active {
            return Ok(fresh);
        }
        let p = &self.p;
        let mut logits = fresh;
        if p.frequency != 0.0 {
            logits = logits.subtract(self.counts.multiply(Array::from_f32(p.frequency))?)?;
        }
        if p.presence != 0.0 {
            let seen = self.seen.as_dtype(mlx_rs::Dtype::Float32)?;
            logits = logits.subtract(seen.multiply(Array::from_f32(p.presence))?)?;
        }
        if p.repetition != 1.0 {
            let r = Array::from_f32(p.repetition);
            let penalised = ops::r#where(
                &logits.lt(Array::from_f32(0.0))?,
                &logits.multiply(&r)?,
                &logits.divide(&r)?,
            )?;
            logits = ops::r#where(&self.seen.expand_dims(0)?, &penalised, &logits)?;
        }
        let rows = block.len();
        if let Some(columns) = &self.stop_columns {
            let bias = ramp(
                self.length,
                rows,
                p.length_start,
                p.length_window,
                p.length_penalty,
            );
            logits = logits.add(bias.multiply(columns)?)?;
        }
        if let (Some(column), Some(close)) = (&self.close_column, p.think_end_token) {
            let visible = block.iter().any(|&t| t != mask_id && t == close);
            if !(self.committed_close || visible) {
                let bias = ramp(
                    self.length,
                    rows,
                    p.think_end_start,
                    p.think_end_window,
                    p.think_end_bias,
                );
                logits = logits.add(bias.multiply(column)?)?;
            }
        }
        Ok(logits)
    }
}

/// `[V]` f32 occurrence counts of `tokens` (a column indicator for distinct ids).
fn indicator_counts(vocab: i32, tokens: &[u32]) -> Result<Array, Exception> {
    let columns = ops::arange::<_, i32>(0, vocab, 1)?;
    let ids: Vec<i32> = tokens
        .iter()
        .filter_map(|&i| i32::try_from(i).ok())
        .collect();
    let n = i32::try_from(ids.len()).map_err(|_| Exception::custom("too many tokens"))?;
    columns
        .eq(Array::from_slice(&ids, &[n, 1]))?
        .as_dtype(mlx_rs::Dtype::Float32)?
        .sum_axis(0, false)
}

/// Prediction and T=1 confidence per row (`predict_with_confidence`).
fn predict(logits: &Array, sampling: &SamplingParams) -> Result<(Vec<u32>, Vec<f32>), Exception> {
    let prediction = if sampling.temperature == 0.0 {
        ops::indexing::argmax_axis(logits, -1, None)?
    } else {
        crate::sample(logits, sampling)?
    }
    .as_dtype(mlx_rs::Dtype::Uint32)?;
    let confidence = ops::softmax_axis(logits, -1, None)?
        .take_along_axis(&prediction.expand_dims(-1)?, -1)?
        .squeeze_axes(&[-1])?;
    eval([&prediction, &confidence])?;
    Ok((
        prediction.as_slice::<u32>().to_vec(),
        confidence.as_slice::<f32>().to_vec(),
    ))
}

/// Positions to unmask this pass: pending with confidence above `threshold`,
/// else the first most-confident pending position.
fn transfer_set(confidence: &[f32], pending: &[bool], threshold: f32) -> Vec<usize> {
    let candidates = || {
        confidence
            .iter()
            .zip(pending)
            .enumerate()
            .filter(|(_, (_, is_pending))| **is_pending)
            .map(|(i, (&c, _))| (i, c))
    };
    let above: Vec<usize> = candidates()
        .filter(|&(_, c)| c > threshold)
        .map(|(i, _)| i)
        .collect();
    if !above.is_empty() {
        return above;
    }
    let mut best: Option<(usize, f32)> = None;
    for (i, c) in candidates() {
        if best.is_none_or(|(_, b)| c > b) {
            best = Some((i, c));
        }
    }
    best.map(|(i, _)| i).into_iter().collect()
}

/// `LowConfidence` block decoding over an already-prefilled `model`.
///
/// `on_block` receives each block's emitted tokens (after EOS truncation)
/// and returns `true` to stop (stop sequence / disconnected client).
pub fn block_decode<M, F>(
    cfg: &BlockDecodeConfig,
    vocab: i32,
    model: &mut M,
    mut on_block: F,
) -> Result<BlockDecodeOutput, Exception>
where
    M: BlockModel,
    F: FnMut(&[u32]) -> bool,
{
    let bs = cfg.block_size;
    let rows = i32::try_from(bs).map_err(|_| Exception::custom("block_size overflows i32"))?;
    if bs == 0 {
        return Err(Exception::custom("block_size must be positive"));
    }
    let mut stop_ids = cfg.eos_token_ids.clone();
    stop_ids.extend(cfg.pad_token_id);
    let mut penalties = PenaltyState::new(cfg.penalties.clone(), vocab, &stop_ids)?;
    let mask_id = i32::try_from(cfg.mask_token_id).map_err(|_| Exception::custom("mask id"))?;
    let mask_column = ops::arange::<_, i32>(0, vocab, 1)?
        .eq(Array::from_int(mask_id))?
        .expand_dims(0)?;
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let mut out = BlockDecodeOutput::default();

    while out.tokens.len() < cfg.max_new_tokens {
        let remaining = bs.min(cfg.max_new_tokens - out.tokens.len());
        let mut block = vec![cfg.mask_token_id; bs];
        let mut pending = vec![true; bs];
        let mut passes = 0;
        while pending.contains(&true) {
            passes += 1;
            let raw = model
                .preview(&block)?
                .reshape(&[rows, -1])?
                .as_dtype(mlx_rs::Dtype::Float32)?;
            let unmasked = ops::r#where(&mask_column, &neg_inf, &raw)?;
            let logits = penalties.apply(unmasked, &block, cfg.mask_token_id)?;
            let (prediction, confidence) = predict(&logits, &cfg.sampling)?;
            for i in transfer_set(&confidence, &pending, cfg.threshold) {
                if let (Some(slot), Some(p), Some(&t)) =
                    (block.get_mut(i), pending.get_mut(i), prediction.get(i))
                {
                    *slot = t;
                    *p = false;
                }
            }
        }
        out.passes_per_block.push(passes);

        // Only the returned continuation is limited by `max_new_tokens`; stop
        // after full resolution at the first EOS in positional order.
        let mut new = block;
        new.truncate(remaining);
        let eos_at = new.iter().position(|t| cfg.eos_token_ids.contains(t));
        let mut emitted = new.clone();
        if let Some(i) = eos_at {
            emitted.truncate(i + 1);
        }
        out.hit_eos = eos_at.is_some();
        out.tokens.extend_from_slice(&emitted);
        penalties.commit(&emitted)?;
        out.stopped_by_caller = on_block(&emitted);
        if out.hit_eos || out.stopped_by_caller || out.tokens.len() >= cfg.max_new_tokens {
            break;
        }
        model.commit(&new)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Self-speculative decoding (DiffuSpace's SGLang `spec` mode, exact acceptance).
// ---------------------------------------------------------------------------

/// The extra operations self-speculation needs on top of [`BlockModel::preview`].
pub trait SpecModel: BlockModel {
    /// Causal forward of `tokens` into the state, tentatively; the greedy next
    /// token after each row (`out[i]` follows `tokens[..=i]`).
    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<u32>, Exception>;
    /// Keep the first `n` rows of the last [`Self::verify`], drop the rest.
    fn keep(&mut self, n: usize) -> Result<(), Exception>;
}

/// One self-speculative request.
#[derive(Debug, Clone)]
pub struct SelfSpecConfig {
    pub mask_token_id: u32,
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    /// Force one draft block size (rows incl. the pending token); `None`
    /// adapts between 4 and 8.
    pub draft_rows: Option<usize>,
}

/// Result of [`self_spec_decode`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelfSpecOutput {
    pub tokens: Vec<u32>,
    /// Draft + verify cycles run.
    pub cycles: usize,
    /// Drafts accepted over all cycles.
    pub accepted: usize,
    /// Cycles run at 8 rows (the rest at 4).
    pub wide_cycles: usize,
    pub hit_eos: bool,
    pub stopped_by_caller: bool,
}

/// Greedy self-speculative decoding with exact acceptance.
///
/// Each cycle drafts the tokens after `pending` with one block preview
/// `[pending, mask...]`,
/// verifies `[pending, drafts...]` causally, keeps the drafts that match the
/// causal argmax plus the causal token after them. The output is the verify
/// pass's greedy sequence whatever the drafts are; the multi-row verify rounds
/// differently from a one-row decode step, so near-tied tokens can differ from
/// plain AR decoding.
///
/// `first` is the greedy token after the prompt (not yet in the state).
/// `on_tokens` receives each cycle's new tokens and returns `true` to stop.
// Indices are bounded by construction: `rows >= 2` and `verify` returns one
// prediction per row.
#[allow(clippy::indexing_slicing)]
pub fn self_spec_decode<M, F>(
    cfg: &SelfSpecConfig,
    first: u32,
    model: &mut M,
    mut on_tokens: F,
) -> Result<SelfSpecOutput, Exception>
where
    M: SpecModel,
    F: FnMut(&[u32]) -> bool,
{
    let mut out = SelfSpecOutput::default();
    if cfg.max_new_tokens == 0 {
        return Ok(out);
    }
    out.tokens.push(first);
    out.hit_eos = cfg.eos_token_ids.contains(&first);
    out.stopped_by_caller = on_tokens(&[first]);
    let mut pending = first;
    // Share of the block accepted, smoothed. Hysteresis from Kapitsa's
    // measurements (M4): code keeps ~0.66 at 8 rows, prose ~0.73 at 4.
    let mut fill = 1.0_f32;
    let mut rows = cfg.draft_rows.unwrap_or(8);
    while !(out.hit_eos || out.stopped_by_caller) && out.tokens.len() < cfg.max_new_tokens {
        let mut block = vec![cfg.mask_token_id; rows];
        block[0] = pending;
        let w = i32::try_from(rows).map_err(|_| Exception::custom("draft rows overflow"))?;
        let logits = model.preview(&block)?.reshape(&[w, -1])?;
        let mask = i32::try_from(cfg.mask_token_id).map_err(|_| Exception::custom("mask id"))?;
        let vocab = logits.shape()[1];
        let mask_column = ops::arange::<_, i32>(0, vocab, 1)?.eq(Array::from_int(mask))?;
        let drafts = ops::indexing::argmax_axis(
            &ops::r#where(&mask_column, Array::from_f32(f32::NEG_INFINITY), &logits)?,
            -1,
            None,
        )?
        .as_dtype(mlx_rs::Dtype::Uint32)?;
        eval([&drafts])?;
        block[1..].copy_from_slice(&drafts.as_slice::<u32>()[1..]);

        let next = model.verify(&block)?;
        let mut n = 0;
        while n + 1 < rows && block[n + 1] == next[n] {
            n += 1;
        }
        model.keep(n + 1)?;
        out.cycles += 1;
        out.accepted += n;
        out.wide_cycles += usize::from(rows == 8);
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        {
            fill = 0.7_f32.mul_add(fill, 0.3 * (n + 1) as f32 / rows as f32);
        }
        if cfg.draft_rows.is_none() {
            rows = match rows {
                8 if fill < 0.5 => 4,
                4 if fill > 0.8 => 8,
                r => r,
            };
        }

        let mut new = block[1..=n].to_vec();
        new.push(next[n]);
        pending = next[n];
        new.truncate(cfg.max_new_tokens - out.tokens.len());
        if let Some(i) = new.iter().position(|t| cfg.eos_token_ids.contains(t)) {
            new.truncate(i + 1);
            out.hit_eos = true;
        }
        out.tokens.extend_from_slice(&new);
        out.stopped_by_caller = on_tokens(&new);
    }
    Ok(out)
}

/// [`SpecModel`] over a loaded model and a request-local cache: verify runs
/// in [`crate::qwen3_next::with_fast_rows`] and records the GDN tape so a
/// partial accept replays only the kept rows.
pub struct AnyModelSpec<'a> {
    pub model: &'a mut AnyModel,
    pub cache: &'a mut AnyCache,
    tapes: Vec<Option<crate::qwen3_next::GdnLayerTape>>,
    rows: usize,
}

impl<'a> AnyModelSpec<'a> {
    pub const fn new(model: &'a mut AnyModel, cache: &'a mut AnyCache) -> Self {
        Self {
            model,
            cache,
            tapes: Vec::new(),
            rows: 0,
        }
    }
}

impl BlockModel for AnyModelSpec<'_> {
    fn preview(&mut self, block: &[u32]) -> Result<Array, Exception> {
        self.model.forward_preview_block(&row(block)?, self.cache)
    }

    fn commit(&mut self, tokens: &[u32]) -> Result<(), Exception> {
        AnyModelBlocks {
            model: self.model,
            cache: self.cache,
        }
        .commit(tokens)
    }
}

impl SpecModel for AnyModelSpec<'_> {
    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<u32>, Exception> {
        let (logits, _, tapes) = crate::qwen3_next::with_fast_rows(|| {
            self.model.forward_with_taps_tape_scheduled(
                &row(tokens)?,
                None,
                self.cache,
                &[],
                None,
                crate::qwen3_next::DFlashRowSchedule::NativeBatch,
            )
        })?;
        let next =
            ops::indexing::argmax_axis(&logits, -1, None)?.as_dtype(mlx_rs::Dtype::Uint32)?;
        eval([&next])?;
        self.tapes = tapes;
        self.rows = tokens.len();
        Ok(next.as_slice::<u32>().to_vec())
    }

    fn keep(&mut self, n: usize) -> Result<(), Exception> {
        if n < self.rows {
            let kept = i32::try_from(n).map_err(|_| Exception::custom("rows overflow"))?;
            let dropped =
                i32::try_from(self.rows - n).map_err(|_| Exception::custom("rows overflow"))?;
            self.model
                .replay_tape_rollback(&self.tapes, self.cache, kept, dropped)?;
        }
        self.tapes.clear();
        self.rows = 0;
        Ok(())
    }
}
