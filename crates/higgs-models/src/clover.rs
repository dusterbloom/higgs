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

use mlx_rs::{Array, error::Exception, ops, ops::indexing::IndexOp};

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

    /// Block-diffusion penalties on fresh `[S, V]` f32 logits: every row sees
    /// the committed response only, and the think-end ramp is off once the
    /// close token is committed or visible anywhere in the block.
    fn apply(&self, fresh: Array, block: &[u32], mask_id: u32) -> Result<Array, Exception> {
        if !self.active {
            return Ok(fresh);
        }
        let open = self.p.think_end_token.is_some_and(|close| {
            !(self.committed_close || block.iter().any(|&t| t != mask_id && t == close))
        });
        let gate = open.then(|| Array::from_f32(1.0));
        self.penalize(fresh, &self.counts, &self.seen, block.len(), gate)
    }

    /// Causal penalties on fresh `[k + 1, V]` f32 verify logits of
    /// `[pending, drafts...]`: row `i` predicts the token after the committed
    /// response plus `drafts[..i]`, so its counts and think-end state include
    /// those drafts and a per-row argmax equals sequential penalised decoding.
    /// `drafts` (`[k]` u32) stays lazy, keeping the draft/verify overlap.
    fn apply_causal(&self, raw: Array, drafts: &Array) -> Result<Array, Exception> {
        if !self.active {
            return Ok(raw);
        }
        let fresh = raw.as_dtype(mlx_rs::Dtype::Float32)?;
        let k = drafts.dim(0);
        let rows = usize::try_from(k + 1).map_err(|_| Exception::custom("draft rows"))?;
        let ids = drafts.as_dtype(mlx_rs::Dtype::Int32)?.reshape(&[k, 1])?;
        let one_hot = ops::arange::<_, i32>(0, self.vocab, 1)?
            .eq(&ids)?
            .as_dtype(mlx_rs::Dtype::Float32)?;
        // Row i adds the drafts before it: a zero row, then the inclusive cumsum.
        let before = ops::concatenate_axis(&[&ops::zeros::<f32>(&[1, self.vocab])?, &one_hot], 0)?
            .cumsum(0, None, None)?;
        let counts = before.add(&self.counts)?;
        let seen = counts.gt(Array::from_f32(0.0))?;
        let gate = match self.p.think_end_token {
            Some(close) if !self.committed_close && self.close_column.is_some() => {
                let closes = drafts.eq(Array::from_slice(&[close], &[1]))?;
                let closed = ops::concatenate_axis(
                    &[
                        &ops::zeros::<f32>(&[1])?,
                        &closes.as_dtype(mlx_rs::Dtype::Float32)?,
                    ],
                    0,
                )?
                .cumsum(0, None, None)?;
                Some(
                    closed
                        .eq(Array::from_f32(0.0))?
                        .as_dtype(mlx_rs::Dtype::Float32)?
                        .reshape(&[k + 1, 1])?,
                )
            }
            _ => None,
        };
        self.penalize(fresh, &counts, &seen, rows, gate)
    }

    /// The reference's operation order: frequency, presence, repetition,
    /// length ramp, think-end ramp. `counts` / `seen` broadcast against the
    /// `rows` rows; `think_gate` (`None` = off) scales the think-end ramp.
    #[allow(clippy::float_cmp)]
    fn penalize(
        &self,
        fresh: Array,
        counts: &Array,
        seen: &Array,
        rows: usize,
        think_gate: Option<Array>,
    ) -> Result<Array, Exception> {
        let p = &self.p;
        let mut logits = fresh;
        if p.frequency != 0.0 {
            logits = logits.subtract(counts.multiply(Array::from_f32(p.frequency))?)?;
        }
        if p.presence != 0.0 {
            let presence = seen.as_dtype(mlx_rs::Dtype::Float32)?;
            logits = logits.subtract(presence.multiply(Array::from_f32(p.presence))?)?;
        }
        if p.repetition != 1.0 {
            let r = Array::from_f32(p.repetition);
            let penalised = ops::r#where(
                &logits.lt(Array::from_f32(0.0))?,
                &logits.multiply(&r)?,
                &logits.divide(&r)?,
            )?;
            logits = ops::r#where(seen, &penalised, &logits)?;
        }
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
        if let (Some(column), Some(gate)) = (&self.close_column, think_gate) {
            let bias = ramp(
                self.length,
                rows,
                p.think_end_start,
                p.think_end_window,
                p.think_end_bias,
            );
            logits = logits.add(bias.multiply(gate)?.multiply(column)?)?;
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

/// The three operations one greedy speculative cycle needs.
pub trait SpecModel {
    /// Draft up to `max` tokens to follow `pending`: a `[k]` u32 array with
    /// `k <= max`. It may stay lazy; the loop evaluates it with the verify
    /// pass so the drafter's GPU work overlaps the verify graph construction.
    fn draft(&mut self, pending: u32, max: usize) -> Result<Array, Exception>;
    /// Causal forward of the `[rows]` u32 `tokens` into the state,
    /// tentatively; the greedy next token after each row (`out[i]` follows
    /// `tokens[..=i]`) of `penalize(logits)`, `logits` being the `[rows, V]`
    /// verify logits. `penalize` stays inside the verify's eval.
    fn verify(&mut self, tokens: &Array, penalize: &Penalize<'_>) -> Result<Vec<u32>, Exception>;
    /// Keep the first `n` rows of the last [`Self::verify`], drop the rest.
    fn keep(&mut self, n: usize) -> Result<(), Exception>;
}

/// Logits processor a [`SpecModel::verify`] applies before its argmax.
pub type Penalize<'a> = dyn Fn(Array) -> Result<Array, Exception> + 'a;

/// One self-speculative request.
#[derive(Debug, Clone)]
pub struct SelfSpecConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    /// Verify block sizes (rows incl. the pending token, each `>= 2`) the
    /// loop chooses from; one entry fixes the size.
    pub rows: Vec<usize>,
    /// Cycle cost model in row units: a cycle at `r` rows costs `base + r`.
    /// Each cycle runs the block with the highest expected tokens per cost
    /// under the running per-position acceptance estimate; near a boundary
    /// the candidates' rates are nearly equal, so the choice is robust to
    /// the estimate's noise.
    pub cycle_base_rows: f32,
    /// Response penalties and ramps, applied causally per verify row.
    pub penalties: CloverPenalties,
    /// Vocabulary size of the verify logits.
    pub vocab: i32,
    /// Pad id; joins the EOS ids as a target of the length ramp.
    pub pad_token_id: Option<u32>,
}

/// Block preview drafting: a preview block must divide the trained block.
pub const PREVIEW_ROWS: [usize; 2] = [4, 8];
/// M4: preview ~35-41 ms + verify ~33.5 ms + 4.1 ms/row.
pub const PREVIEW_CYCLE_BASE_ROWS: f32 = 13.0;
/// `DFlash` drafting (block 8): M4 optima are 3 rows on prose, 6-8 on code.
pub const DFLASH_ROWS: [usize; 3] = [3, 6, 8];
/// M4 with the fused drafter: a cycle costs ~28.4 ms + 3.3 ms/row (38.3 /
/// 49.1 / 54.7 ms at 3 / 6 / 8 rows).
pub const DFLASH_CYCLE_BASE_ROWS: f32 = 8.6;

/// Expected tokens of a cycle at `rows`: the pending/bonus token plus the
/// accepted prefix, `p[i]` being the acceptance of draft `i + 1` given the
/// drafts before it were accepted. Acceptance does not rise with depth, so a
/// depth counts at most as the depth before it: a deep depth the loop has not
/// tested yet (still at its prior) cannot sell a wider block on its own.
fn expected_tokens(p: &[f32], rows: usize) -> f32 {
    let mut prefix = 1.0_f32;
    let mut total = 1.0_f32;
    let mut cap = 1.0_f32;
    for &pi in p.iter().take(rows.saturating_sub(1)) {
        cap = cap.min(pi);
        prefix *= cap;
        total += prefix;
    }
    total
}

/// The block size with the highest expected tokens per cycle cost.
fn best_rows(ladder: &[usize], p: &[f32], base: f32) -> usize {
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let rate = |r: usize| expected_tokens(p, r) / (base + r as f32);
    ladder
        .iter()
        .copied()
        .max_by(|&a, &b| rate(a).total_cmp(&rate(b)))
        .unwrap_or(8)
}

/// Whether a cycle's draft is evaluated asynchronously ahead of the verify.
fn async_draft() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("HIGGS_SPEC_ASYNC_DRAFT").map_or(true, |v| v != "0"))
}

/// Result of [`self_spec_decode`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelfSpecOutput {
    pub tokens: Vec<u32>,
    /// Draft + verify cycles run.
    pub cycles: usize,
    /// Drafts accepted over all cycles.
    pub accepted: usize,
    /// Cycles run at the widest block size.
    pub wide_cycles: usize,
    pub hit_eos: bool,
    pub stopped_by_caller: bool,
}

/// Greedy speculative decoding with exact acceptance.
///
/// Each cycle drafts the tokens after `pending`, verifies
/// `[pending, drafts...]` causally, keeps the drafts that match the causal
/// (penalised) argmax plus the causal token after them. Row `i`'s penalties
/// see the response through `drafts[..i]`, so with penalties the output is
/// still sequential penalised greedy decoding. The output is the verify pass's
/// greedy sequence whatever the drafts are; the multi-row verify rounds
/// differently from a one-row decode step, so near-tied tokens can differ
/// from plain AR decoding.
///
/// `first` is the greedy token after the prompt (not yet in the state).
/// ponytail: `first` is taken unpenalised: an empty response has no counts
/// and the ramps start past position 1 for any pack seen so far.
/// `on_tokens` receives each cycle's new tokens and returns `true` to stop.
// Indices are bounded by construction: `rows >= 2`, `k <= rows - 1` and
// `verify` returns one prediction per row.
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
    let mut stop_ids = cfg.eos_token_ids.clone();
    stop_ids.extend(cfg.pad_token_id);
    let mut penalties = PenaltyState::new(cfg.penalties.clone(), cfg.vocab, &stop_ids)?;
    out.tokens.push(first);
    penalties.commit(&[first])?;
    out.hit_eos = cfg.eos_token_ids.contains(&first);
    out.stopped_by_caller = on_tokens(&[first]);
    let mut pending = first;
    let ladder: Vec<usize> = cfg.rows.iter().map(|&r| r.max(2)).collect();
    let widest = ladder.iter().copied().max().unwrap_or(8);
    // Acceptance of each draft depth given the drafts before it held,
    // exponentially forgotten, prior 0.8 over two trials. Drafters lose
    // accuracy with depth, so one pooled rate would overrate wide blocks.
    let mut successes = vec![1.6_f32; widest - 1];
    let mut trials = vec![2.0_f32; widest - 1];
    let mut p = vec![0.8_f32; widest - 1];
    while !(out.hit_eos || out.stopped_by_caller) && out.tokens.len() < cfg.max_new_tokens {
        let rows = best_rows(&ladder, &p, cfg.cycle_base_rows);
        let drafts = model.draft(pending, rows - 1)?;
        // The draft's GPU work runs while the host builds the verify graph
        // (`HIGGS_SPEC_ASYNC_DRAFT=0` keeps both in the verify's eval).
        if async_draft() {
            crate::mlx_exec::async_eval([&drafts])?;
        }
        let k = usize::try_from(drafts.dim(0)).map_err(|_| Exception::custom("draft count"))?;
        if k >= rows {
            return Err(Exception::custom(format!(
                "drafter returned {k} tokens for {} slots",
                rows - 1
            )));
        }
        let block = ops::concatenate_axis(&[&Array::from_slice(&[pending], &[1]), &drafts], 0)?;
        let next = model.verify(&block, &|logits| penalties.apply_causal(logits, &drafts))?;
        if next.len() != k + 1 {
            return Err(Exception::custom(format!(
                "verify returned {} predictions for {} rows",
                next.len(),
                k + 1
            )));
        }
        let tokens = block.as_slice::<u32>();
        let mut n = 0;
        while n < k && tokens[n + 1] == next[n] {
            n += 1;
        }
        model.keep(n + 1)?;
        out.cycles += 1;
        out.accepted += n;
        out.wide_cycles += usize::from(rows == widest);
        // Only the depths this cycle tested (the accepted drafts and the
        // first rejected one) forget and update: an untested depth keeps its
        // estimate at full weight, else its evidence would thin out while the
        // loop runs narrower blocks and one later observation would swing it.
        for (depth, (s, t)) in successes.iter_mut().zip(trials.iter_mut()).enumerate() {
            if depth < n {
                *s = s.mul_add(0.95, 1.0);
                *t = t.mul_add(0.95, 1.0);
            } else if depth == n && n < k {
                *s *= 0.95;
                *t = t.mul_add(0.95, 1.0);
            } else {
                continue;
            }
            p[depth] = *s / *t;
        }

        let mut new = tokens[1..=n].to_vec();
        new.push(next[n]);
        pending = next[n];
        new.truncate(cfg.max_new_tokens - out.tokens.len());
        if let Some(i) = new.iter().position(|t| cfg.eos_token_ids.contains(t)) {
            new.truncate(i + 1);
            out.hit_eos = true;
        }
        out.tokens.extend_from_slice(&new);
        penalties.commit(&new)?;
        out.stopped_by_caller = on_tokens(&new);
    }
    Ok(out)
}

/// What one verify leaves behind: greedy next token per row, the post-layer
/// hiddens at the requested tap layers and the GDN tape for a partial keep.
type VerifyRows = (
    Vec<u32>,
    Vec<Array>,
    Vec<Option<crate::qwen3_next::GdnLayerTape>>,
);

/// Fast-rows tape verify of one `[rows]` block on a request-local cache; the
/// taps are materialized by the same eval as the logits.
fn verify_rows(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    tokens: &Array,
    tap_layers: &[usize],
    penalize: &Penalize<'_>,
) -> Result<VerifyRows, Exception> {
    let input = tokens.reshape(&[1, -1])?;
    let (logits, taps, tapes) = crate::qwen3_next::with_fast_rows(|| {
        model.forward_with_taps_tape_scheduled(
            &input,
            None,
            cache,
            tap_layers,
            None,
            crate::qwen3_next::DFlashRowSchedule::NativeBatch,
        )
    })?;
    let rows = tokens.dim(0);
    let penalized = penalize(logits.reshape(&[rows, -1])?)?;
    let next = ops::indexing::argmax_axis(&penalized, -1, None)?.as_dtype(mlx_rs::Dtype::Uint32)?;
    eval([&next])?;
    Ok((next.as_slice::<u32>().to_vec(), taps, tapes))
}

/// Keep the first `n` of the `rows` rows of the last verify (GDN replay from
/// the tape + KV trim); a no-op when every row is kept.
fn keep_rows(
    model: &AnyModel,
    cache: &mut AnyCache,
    tapes: &[Option<crate::qwen3_next::GdnLayerTape>],
    rows: usize,
    n: usize,
) -> Result<(), Exception> {
    if n < rows {
        let kept = i32::try_from(n).map_err(|_| Exception::custom("rows overflow"))?;
        let dropped = i32::try_from(rows - n).map_err(|_| Exception::custom("rows overflow"))?;
        model.replay_tape_rollback(tapes, cache, kept, dropped)?;
    }
    Ok(())
}

/// [`SpecModel`] over a loaded model and a request-local cache, drafting
/// with a block preview.
///
/// The preview of `[pending, mask...]` drafts; verify runs in
/// [`crate::qwen3_next::with_fast_rows`] and records the GDN tape so a
/// partial accept replays only the kept rows.
pub struct AnyModelSpec<'a> {
    pub model: &'a mut AnyModel,
    pub cache: &'a mut AnyCache,
    mask_token_id: u32,
    tapes: Vec<Option<crate::qwen3_next::GdnLayerTape>>,
    rows: usize,
}

impl<'a> AnyModelSpec<'a> {
    pub const fn new(model: &'a mut AnyModel, cache: &'a mut AnyCache, mask_token_id: u32) -> Self {
        Self {
            model,
            cache,
            mask_token_id,
            tapes: Vec::new(),
            rows: 0,
        }
    }
}

impl SpecModel for AnyModelSpec<'_> {
    /// One preview of `[pending, mask * max]`; the argmax of the masked
    /// positions (the mask id itself excluded) drafts them.
    fn draft(&mut self, pending: u32, max: usize) -> Result<Array, Exception> {
        let mut block = vec![self.mask_token_id; max + 1];
        if let Some(slot) = block.first_mut() {
            *slot = pending;
        }
        let w = i32::try_from(block.len()).map_err(|_| Exception::custom("draft rows overflow"))?;
        let logits = self
            .model
            .forward_preview_block(&row(&block)?, self.cache)?
            .reshape(&[w, -1])?
            .index((1.., ..));
        let mask = i32::try_from(self.mask_token_id).map_err(|_| Exception::custom("mask id"))?;
        let vocab = logits.dim(1);
        let mask_column = ops::arange::<_, i32>(0, vocab, 1)?.eq(Array::from_int(mask))?;
        ops::indexing::argmax_axis(
            &ops::r#where(&mask_column, Array::from_f32(f32::NEG_INFINITY), &logits)?,
            -1,
            None,
        )?
        .as_dtype(mlx_rs::Dtype::Uint32)
    }

    fn verify(&mut self, tokens: &Array, penalize: &Penalize<'_>) -> Result<Vec<u32>, Exception> {
        let (next, _, tapes) = verify_rows(self.model, self.cache, tokens, &[], penalize)?;
        self.tapes = tapes;
        self.rows = next.len();
        Ok(next)
    }

    fn keep(&mut self, n: usize) -> Result<(), Exception> {
        keep_rows(self.model, self.cache, &self.tapes, self.rows, n)?;
        self.tapes.clear();
        self.rows = 0;
        Ok(())
    }
}

/// Target tap rows (`[1, T, H]` per tap layer) the drafter has not consumed
/// yet, kept consistent with the rows the verify keeps.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct DraftTaps {
    /// Committed rows not yet in the drafter's context.
    pending: Vec<Array>,
    /// Every row of the last verify, rejected ones included.
    verified: Vec<Array>,
}

impl DraftTaps {
    #[must_use]
    pub const fn new(pending: Vec<Array>) -> Self {
        Self {
            pending,
            verified: Vec::new(),
        }
    }

    #[must_use]
    pub fn pending(&self) -> &[Array] {
        &self.pending
    }

    /// Rows in `pending`.
    pub fn pending_rows(&self) -> Result<i32, Exception> {
        self.pending.first().map_or(Ok(0), |tap| {
            tap.shape()
                .get(1)
                .copied()
                .ok_or_else(|| Exception::custom("tap has no row axis"))
        })
    }

    /// The drafter consumed `pending`.
    pub fn consume(&mut self) {
        self.pending.clear();
    }

    /// The verify processed these rows.
    pub fn record(&mut self, verified: Vec<Array>) {
        self.verified = verified;
    }

    /// The first `n` verified rows are committed: they become `pending`.
    pub fn keep(&mut self, n: usize) -> Result<(), Exception> {
        let rows = self
            .verified
            .first()
            .and_then(|tap| tap.shape().get(1).copied())
            .ok_or_else(|| Exception::custom("keep without a recorded verify"))?;
        let kept = i32::try_from(n).map_err(|_| Exception::custom("kept rows overflow"))?;
        if kept < 1 || kept > rows {
            return Err(Exception::custom(format!(
                "keep {kept} of {rows} verified rows"
            )));
        }
        if !self.pending.is_empty() {
            return Err(Exception::custom(
                "drafter has not consumed the pending taps",
            ));
        }
        self.pending = self
            .verified
            .drain(..)
            .map(|tap| tap.index((.., ..kept, ..)))
            .collect();
        Ok(())
    }
}

/// [`SpecModel`] whose drafts come from a `DFlash` drafter.
///
/// The `DFlash2` selector (or the target head's argmax for plain `DFlash`)
/// proposes; the verify also returns the target hiddens at the drafter's tap
/// layers, and the kept rows become the drafter's context for the next
/// draft, so the drafter's position tracks the request-local cache exactly.
pub struct DFlashSpec<'a> {
    pub model: &'a mut AnyModel,
    pub cache: &'a mut AnyCache,
    drafter: &'a mut crate::dflash::DFlashDrafter,
    draft_cache: crate::dflash::DFlashCache,
    tap_layers: Vec<usize>,
    taps: DraftTaps,
    mask_token_id: u32,
    /// Target rows committed so far (the drafter's position after it consumes
    /// the pending taps).
    committed: i32,
    tapes: Vec<Option<crate::qwen3_next::GdnLayerTape>>,
    rows: usize,
    /// The reduced draft head (see [`crate::dflash::DFlashDrafter::reduced_head`]);
    /// `None` runs the target's full head.
    draft_head: Option<(Array, Array, Array)>,
}

/// Rows of the target head the drafter's shortlist is drawn from:
/// `HIGGS_DFLASH_HEAD_ROWS` (0 = the full head), default 98304 (Kapitsa's;
/// every token of the M4 prose/code/count runs has an id below 95201).
fn draft_head_rows() -> Option<i32> {
    static ROWS: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("HIGGS_DFLASH_HEAD_ROWS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .map_or(Some(98_304), |rows| (rows > 0).then_some(rows))
    })
}

impl<'a> DFlashSpec<'a> {
    /// `draft_cache` holds the prompt except `taps`, the target hiddens of
    /// its last chunk (what the drafter's prefill returns).
    pub fn new(
        model: &'a mut AnyModel,
        cache: &'a mut AnyCache,
        drafter: &'a mut crate::dflash::DFlashDrafter,
        draft_cache: crate::dflash::DFlashCache,
        taps: Vec<Array>,
        mask_token_id: u32,
    ) -> Result<Self, Exception> {
        if drafter.config.is_dspark() {
            return Err(Exception::custom(
                "dSpark drafters cannot draft self-speculation",
            ));
        }
        let pending = DraftTaps::new(taps);
        let committed = draft_cache
            .position()
            .checked_add(pending.pending_rows()?)
            .ok_or_else(|| Exception::custom("drafter position overflow"))?;
        // A reduced head needs the fast-rows head format and fewer rows than
        // the vocabulary (in multiples of the kernel's 8-row block).
        let draft_head = match (draft_head_rows(), model.lm_head_fast_rows_parts()) {
            (Some(rows), Some(head)) if rows % 8 == 0 && rows < head.0.dim(0) => {
                Some(drafter.reduced_head(head, rows)?.clone())
            }
            _ => None,
        };
        Ok(Self {
            model,
            cache,
            tap_layers: drafter.config.target_layer_ids().to_vec(),
            drafter,
            draft_cache,
            taps: pending,
            mask_token_id,
            committed,
            tapes: Vec::new(),
            rows: 0,
            draft_head,
        })
    }
}

impl SpecModel for DFlashSpec<'_> {
    fn draft(&mut self, pending: u32, max: usize) -> Result<Array, Exception> {
        let bs = self.drafter.config.block_size;
        let width = usize::try_from(bs).map_err(|_| Exception::custom("block size"))?;
        let mut block = vec![self.mask_token_id; width.max(1)];
        if let Some(slot) = block.first_mut() {
            *slot = pending;
        }
        let noise = self.model.embed_token_ids(&row(&block)?)?;
        let anchor = i32::try_from(pending).map_err(|_| Exception::custom("token id overflow"))?;
        // Positions 1.. are the masked slots; the selector walks them in order
        // and a slot depends only on the slots before it, so only the first
        // `k` go through the target head and the top-k partition.
        let k = i32::try_from(max.min(width.saturating_sub(1)))
            .map_err(|_| Exception::custom("draft cap overflow"))?;
        let tokens = crate::qwen3_next::with_fast_rows(|| {
            let tx = self
                .drafter
                .stage_forward(&noise, self.taps.pending(), &self.draft_cache)?;
            if tx.position()? != self.committed {
                return Err(Exception::custom(format!(
                    "drafter context diverged: drafter {} target {}",
                    tx.position()?,
                    self.committed
                )));
            }
            let hidden = tx.hidden().index((.., 1..=k, ..));
            let logits = match &self.draft_head {
                Some((w, s, b)) => {
                    let hidden_width = hidden.dim(2);
                    crate::qmm8::qmm8_affine4(&hidden.reshape(&[k, hidden_width])?, w, s, b)?
                        .reshape(&[1, k, -1])?
                }
                None => self.model.forward_all_logits_from_hidden(&hidden)?,
            };
            let tokens = match self
                .drafter
                .propose_dflash2_tokens(&hidden, &logits, anchor)?
            {
                Some(tokens) => tokens,
                None => ops::indexing::argmax_axis(&logits, -1, None)?,
            };
            tx.commit(&mut self.draft_cache)?;
            Ok(tokens)
        })?;
        self.taps.consume();
        tokens.index(0).as_dtype(mlx_rs::Dtype::Uint32)
    }

    fn verify(&mut self, tokens: &Array, penalize: &Penalize<'_>) -> Result<Vec<u32>, Exception> {
        let (next, taps, tapes) =
            verify_rows(self.model, self.cache, tokens, &self.tap_layers, penalize)?;
        self.taps.record(taps);
        self.tapes = tapes;
        self.rows = next.len();
        Ok(next)
    }

    fn keep(&mut self, n: usize) -> Result<(), Exception> {
        keep_rows(self.model, self.cache, &self.tapes, self.rows, n)?;
        self.taps.keep(n)?;
        self.committed = self
            .committed
            .checked_add(i32::try_from(n).map_err(|_| Exception::custom("rows overflow"))?)
            .ok_or_else(|| Exception::custom("drafter position overflow"))?;
        self.tapes.clear();
        self.rows = 0;
        Ok(())
    }
}
