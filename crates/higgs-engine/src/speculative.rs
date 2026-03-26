use higgs_models::turboquant::KvCacheConfig;
use higgs_models::{AnyCache, AnyModel};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::error::EngineError;

/// Run one speculative decode cycle.
///
/// 1. Draft `num_draft` tokens with the draft model
/// 2. Build verify batch: `[last_token_id, draft_0, ..., draft_{K-1}]`
/// 3. Call `verify_fn` with the batch to get `K+1` target-sampled token IDs
/// 4. Accept the longest matching prefix
/// 5. Advance or rollback the draft model accordingly
///
/// Returns the accepted token IDs (1..=K+1).
pub fn speculative_step<F>(
    draft: &mut dyn DraftModel,
    last_token_id: u32,
    num_draft: usize,
    verify_fn: F,
) -> Result<Vec<u32>, EngineError>
where
    F: FnOnce(&[u32]) -> Result<Vec<u32>, EngineError>,
{
    let draft_ids = draft.draft(last_token_id, num_draft)?;
    let k = draft_ids.len();

    // Build verify batch: [last_token, draft_0, ..., draft_K-1]
    let mut verify_batch = Vec::with_capacity(k + 1);
    verify_batch.push(last_token_id);
    verify_batch.extend_from_slice(&draft_ids);

    let target_ids = verify_fn(&verify_batch)?;
    let accepted = accept_prefix(&draft_ids, &target_ids)?;

    // Count how many draft tokens were confirmed (excluding the divergent/bonus)
    let matched = if accepted.len() > k {
        k // all matched + bonus
    } else {
        accepted.len().saturating_sub(1) // last token is the divergence
    };

    if matched > 0 {
        draft.advance(matched)?;
    } else {
        draft.rollback()?;
    }

    Ok(accepted)
}

/// Result of one speculative decode cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepResult {
    /// Accepted token IDs from this cycle (1..=num_draft+1).
    pub tokens: Vec<u32>,
    /// Whether an EOS token was encountered.
    pub hit_eos: bool,
}

/// Run a full speculative decode loop until EOS or `max_tokens`.
///
/// `verify_fn` is called once per cycle with the verify batch (K+1 token IDs)
/// and must return K+1 target-sampled token IDs.
///
/// Returns all generated token IDs (not including the initial `last_token_id`).
pub fn speculative_loop<F>(
    draft: &mut dyn DraftModel,
    last_token_id: u32,
    num_draft: usize,
    max_tokens: usize,
    eos_ids: &[u32],
    mut verify_fn: F,
) -> Result<Vec<u32>, EngineError>
where
    F: FnMut(&[u32]) -> Result<Vec<u32>, EngineError>,
{
    let mut generated = Vec::new();
    let mut current_token = last_token_id;

    while generated.len() < max_tokens {
        let remaining = max_tokens - generated.len();
        let k = num_draft.min(remaining);
        if k == 0 {
            break;
        }

        let accepted = speculative_step(draft, current_token, k, |batch| verify_fn(batch))?;

        for &token in &accepted {
            if generated.len() >= max_tokens {
                break;
            }
            generated.push(token);
            if eos_ids.contains(&token) {
                return Ok(generated);
            }
        }

        if let Some(&last) = generated.last() {
            current_token = last;
        }
    }

    Ok(generated)
}

/// Compute the accepted prefix from a speculative decode cycle.
///
/// Given `draft_ids` (K tokens from the draft model) and `target_ids` (K+1
/// samples from the target model's verify logits), return the longest prefix
/// where draft and target agree, followed by the target's first divergent
/// token.
///
/// Invariants:
/// - `target_ids.len() == draft_ids.len() + 1`
/// - Returns 1..=K+1 tokens (always at least one token from the target)
pub fn accept_prefix(draft_ids: &[u32], target_ids: &[u32]) -> Result<Vec<u32>, EngineError> {
    let k = draft_ids.len();
    if target_ids.len() != k + 1 {
        return Err(EngineError::Generation(format!(
            "accept_prefix: target_ids.len() ({}) must be draft_ids.len() ({k}) + 1",
            target_ids.len(),
        )));
    }

    let mut accepted = Vec::with_capacity(k + 1);
    for i in 0..k {
        accepted.push(target_ids[i]);
        if target_ids[i] != draft_ids[i] {
            return Ok(accepted);
        }
    }
    // All K drafts matched — include the bonus token from the last verify position
    accepted.push(target_ids[k]);
    Ok(accepted)
}

/// Trait for a draft model that produces candidate tokens for speculative
/// decoding. Implementations may run on any device (GPU, ANE, CPU).
pub trait DraftModel: Send {
    /// Prefill the draft model with the given prompt tokens, resetting any
    /// prior cache state. Must be called once before the first `draft()` call
    /// in a new generation request.
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), EngineError>;

    /// Generate up to `num_draft` greedy tokens starting from `last_token_id`.
    fn draft(&mut self, last_token_id: u32, num_draft: usize) -> Result<Vec<u32>, EngineError>;

    /// Advance internal state by `n` accepted tokens.
    /// Called after verify confirms the first `n` draft tokens.
    fn advance(&mut self, n: usize) -> Result<(), EngineError>;

    /// Roll back to the state before the last `draft()` call.
    /// Called when the target rejects draft tokens and we need to resync.
    fn rollback(&mut self) -> Result<(), EngineError>;
}

/// Draft model backed by an MLX `AnyModel` running on GPU.
///
/// This is the baseline implementation: both draft and target share the GPU.
/// On Apple Silicon MoE models this doesn't speed up decode (see commit
/// `1cea874`), but it validates the trait integration and serves as fallback
/// when ANE is unavailable.
pub struct MlxDraftModel {
    model: AnyModel,
    cache: AnyCache,
    kv_cache_config: KvCacheConfig,
    /// Snapshot of the cache before the last `draft()` call.
    checkpoint: Option<AnyCache>,
}

impl MlxDraftModel {
    pub fn new(model: AnyModel, cache: AnyCache, kv_cache_config: KvCacheConfig) -> Self {
        Self {
            model,
            cache,
            kv_cache_config,
            checkpoint: None,
        }
    }
}

impl DraftModel for MlxDraftModel {
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), EngineError> {
        self.cache = self
            .model
            .make_cache_with_config(self.kv_cache_config)
            .map_err(EngineError::Mlx)?;
        self.checkpoint = None;

        let len =
            i32::try_from(prompt_tokens.len()).map_err(|_| EngineError::Generation("prompt too long for draft prefill".into()))?;
        let input = Array::from_slice(prompt_tokens, &[1, len]);
        let logits = self
            .model
            .forward(&input, None, &mut self.cache)
            .map_err(EngineError::Mlx)?;
        eval(std::slice::from_ref(&logits)).map_err(EngineError::Mlx)?;
        Ok(())
    }

    fn draft(&mut self, last_token_id: u32, num_draft: usize) -> Result<Vec<u32>, EngineError> {
        // Save checkpoint for rollback
        self.checkpoint = Some(self.cache.clone());

        let mut tokens = Vec::with_capacity(num_draft);
        let mut current = Array::from_slice(&[last_token_id], &[1]);

        for _ in 0..num_draft {
            let input = current.index((.., NewAxis));
            let logits = self
                .model
                .forward(&input, None, &mut self.cache)
                .map_err(EngineError::Mlx)?;
            let last_logits = logits.index((.., -1, ..));
            let next = mlx_rs::ops::indexing::argmax_axis(&last_logits, -1, false)
                .map_err(EngineError::Mlx)?;
            eval(std::slice::from_ref(&next)).map_err(EngineError::Mlx)?;

            let token_id: u32 = next.item();
            tokens.push(token_id);
            current = Array::from_slice(&[token_id], &[1]);
        }

        Ok(tokens)
    }

    fn advance(&mut self, _n: usize) -> Result<(), EngineError> {
        // Cache already has the right state for accepted tokens.
        // Draft quality slightly degrades after partial acceptance (stale cache
        // entries remain), but correctness is unaffected — the target always
        // verifies. A future optimization can trim the draft cache too.
        self.checkpoint = None;
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), EngineError> {
        if let Some(checkpoint) = self.checkpoint.take() {
            self.cache = checkpoint;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ANE/CPU draft model (feature-gated)
// ---------------------------------------------------------------------------

/// CPU-accelerated draft model that loads weights from MLX safetensors.
///
/// Runs entirely on CPU (using Accelerate SGEMM), freeing the GPU for the
/// target model. Uses its own KV cache with rollback support.
///
/// Architecture ready for ANE acceleration (BLOBFILE kernels) in a future pass.
#[cfg(feature = "ane")]
pub struct AneDraftModel {
    model: higgs_ane::ModelWeights,
    cache: higgs_ane::KvCache,
    checkpoint_pos: Option<usize>,
}

#[cfg(feature = "ane")]
impl AneDraftModel {
    /// Load a draft model from an MLX safetensors directory.
    pub fn load(dir: &std::path::Path, max_seq: usize) -> Result<Self, EngineError> {
        let model = higgs_ane::load_draft_model(dir)
            .map_err(|e| EngineError::Generation(format!("ANE draft load: {e}")))?;
        let cache = higgs_ane::make_kv_cache(&model, max_seq);
        Ok(Self {
            model,
            cache,
            checkpoint_pos: None,
        })
    }
}

#[cfg(feature = "ane")]
impl DraftModel for AneDraftModel {
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), EngineError> {
        self.cache.rollback_to(0);
        self.checkpoint_pos = None;
        for &token in prompt_tokens {
            higgs_ane::decode_step(&self.model, token, &mut self.cache);
        }
        Ok(())
    }

    fn draft(&mut self, last_token_id: u32, num_draft: usize) -> Result<Vec<u32>, EngineError> {
        self.checkpoint_pos = Some(self.cache.pos());
        let mut drafts = Vec::with_capacity(num_draft);
        let mut token = last_token_id;
        for _ in 0..num_draft {
            let result = higgs_ane::decode_step(&self.model, token, &mut self.cache);
            token = higgs_ane::sample_argmax(&result.logits);
            drafts.push(token);
        }
        Ok(drafts)
    }

    fn advance(&mut self, n: usize) -> Result<(), EngineError> {
        if let Some(pre) = self.checkpoint_pos.take() {
            self.cache.rollback_to(pre + n);
        }
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), EngineError> {
        if let Some(pre) = self.checkpoint_pos.take() {
            self.cache.rollback_to(pre);
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    // Compile-time check: MlxDraftModel satisfies DraftModel + Send
    const _: () = {
        fn _assert_send<T: DraftModel + Send>() {}
        fn _assert() {
            _assert_send::<MlxDraftModel>();
        }
    };

    // ── accept_prefix ──────────────────────────────────────────────────

    #[test]
    fn accept_prefix_all_match_returns_k_plus_one() {
        // Draft: [5, 3, 7], Target: [5, 3, 7, 42]
        // All 3 drafts match → return all 3 + bonus token 42
        let accepted = accept_prefix(&[5, 3, 7], &[5, 3, 7, 42]).unwrap();
        assert_eq!(accepted, vec![5, 3, 7, 42]);
    }

    #[test]
    fn accept_prefix_first_mismatch_returns_one() {
        // Draft: [5, 3, 7], Target: [9, 1, 2, 0]
        // First token diverges → return [9] (target's choice)
        let accepted = accept_prefix(&[5, 3, 7], &[9, 1, 2, 0]).unwrap();
        assert_eq!(accepted, vec![9]);
    }

    #[test]
    fn accept_prefix_mid_mismatch() {
        // Draft: [5, 3, 7], Target: [5, 3, 9, 0]
        // Match at 0,1 → diverge at 2 → return [5, 3, 9]
        let accepted = accept_prefix(&[5, 3, 7], &[5, 3, 9, 0]).unwrap();
        assert_eq!(accepted, vec![5, 3, 9]);
    }

    #[test]
    fn accept_prefix_single_draft_match() {
        let accepted = accept_prefix(&[10], &[10, 99]).unwrap();
        assert_eq!(accepted, vec![10, 99]);
    }

    #[test]
    fn accept_prefix_single_draft_mismatch() {
        let accepted = accept_prefix(&[10], &[20, 99]).unwrap();
        assert_eq!(accepted, vec![20]);
    }

    #[test]
    fn accept_prefix_empty_draft() {
        // Zero draft tokens → just the bonus token
        let accepted = accept_prefix(&[], &[42]).unwrap();
        assert_eq!(accepted, vec![42]);
    }

    #[test]
    fn accept_prefix_wrong_length_errors() {
        let err = accept_prefix(&[1, 2], &[1, 2]).unwrap_err();
        assert!(err.to_string().contains("must be"));
    }

    // ── DraftModel trait ───────────────────────────────────────────────

    /// Mock draft model that returns a fixed sequence of tokens.
    struct MockDraft {
        /// Tokens to return from `draft()`, cycling through them.
        sequence: Vec<u32>,
        cursor: usize,
        draft_count: usize,
    }

    impl MockDraft {
        fn new(sequence: Vec<u32>) -> Self {
            Self {
                sequence,
                cursor: 0,
                draft_count: 0,
            }
        }
    }

    impl DraftModel for MockDraft {
        fn prefill(&mut self, _prompt_tokens: &[u32]) -> Result<(), EngineError> {
            Ok(())
        }

        fn draft(&mut self, _last_token_id: u32, num_draft: usize) -> Result<Vec<u32>, EngineError> {
            let mut tokens = Vec::with_capacity(num_draft);
            for i in 0..num_draft {
                let idx = (self.cursor + i) % self.sequence.len();
                tokens.push(self.sequence[idx]);
            }
            self.draft_count = num_draft;
            Ok(tokens)
        }

        fn advance(&mut self, n: usize) -> Result<(), EngineError> {
            self.cursor = (self.cursor + n) % self.sequence.len();
            self.draft_count = 0;
            Ok(())
        }

        fn rollback(&mut self) -> Result<(), EngineError> {
            self.draft_count = 0;
            Ok(())
        }
    }

    #[test]
    fn mock_draft_produces_tokens() {
        let mut draft = MockDraft::new(vec![10, 20, 30]);
        let tokens = draft.draft(0, 3).unwrap();
        assert_eq!(tokens, vec![10, 20, 30]);
    }

    #[test]
    fn mock_draft_advance_shifts_cursor() {
        let mut draft = MockDraft::new(vec![10, 20, 30]);
        let _ = draft.draft(0, 2).unwrap();
        draft.advance(2).unwrap();
        let tokens = draft.draft(0, 2).unwrap();
        assert_eq!(tokens, vec![30, 10]);
    }

    #[test]
    fn mock_draft_rollback_preserves_cursor() {
        let mut draft = MockDraft::new(vec![10, 20, 30]);
        let _ = draft.draft(0, 2).unwrap();
        draft.rollback().unwrap();
        let tokens = draft.draft(0, 2).unwrap();
        assert_eq!(tokens, vec![10, 20]); // same as before
    }

    // ── speculative_step ───────────────────────────────────────────────

    #[test]
    fn step_all_accepted_returns_k_plus_one() {
        let mut draft = MockDraft::new(vec![10, 20, 30]);
        // Draft produces [10, 20, 30]. Target agrees on all + bonus 99.
        let accepted = speculative_step(&mut draft, 0, 3, |batch| {
            assert_eq!(batch, &[0, 10, 20, 30]); // verify batch
            Ok(vec![10, 20, 30, 99]) // K+1 target samples
        })
        .unwrap();
        assert_eq!(accepted, vec![10, 20, 30, 99]);
        // Draft advanced by 3 (all matched)
        assert_eq!(draft.cursor, 0); // 3 % 3 == 0
    }

    #[test]
    fn step_partial_accept_advances_draft() {
        let mut draft = MockDraft::new(vec![10, 20, 30]);
        // Draft produces [10, 20, 30]. Target matches first 2, diverges at 3rd.
        let accepted = speculative_step(&mut draft, 0, 3, |_| {
            Ok(vec![10, 20, 99, 55]) // diverge at position 2
        })
        .unwrap();
        assert_eq!(accepted, vec![10, 20, 99]);
        // Draft advanced by 2 (matched tokens)
        assert_eq!(draft.cursor, 2);
    }

    #[test]
    fn step_no_match_rollback() {
        let mut draft = MockDraft::new(vec![10, 20, 30]);
        // Draft produces [10, 20, 30]. Target rejects immediately.
        let accepted = speculative_step(&mut draft, 0, 3, |_| {
            Ok(vec![77, 0, 0, 0]) // first token differs
        })
        .unwrap();
        assert_eq!(accepted, vec![77]);
        // Draft rolled back (cursor unchanged)
        assert_eq!(draft.cursor, 0);
    }

    #[test]
    fn step_single_draft_match() {
        let mut draft = MockDraft::new(vec![10]);
        let accepted = speculative_step(&mut draft, 5, 1, |batch| {
            assert_eq!(batch, &[5, 10]);
            Ok(vec![10, 42]) // match + bonus
        })
        .unwrap();
        assert_eq!(accepted, vec![10, 42]);
    }

    #[test]
    fn step_verify_error_propagates() {
        let mut draft = MockDraft::new(vec![10, 20]);
        let err = speculative_step(&mut draft, 0, 2, |_| {
            Err(EngineError::Generation("GPU OOM".into()))
        })
        .unwrap_err();
        assert!(err.to_string().contains("GPU OOM"));
    }

    // ── speculative_loop ───────────────────────────────────────────────

    #[test]
    fn loop_generates_until_max_tokens() {
        let mut draft = MockDraft::new(vec![1, 2, 3]);
        // Target always agrees with draft → 3+1=4 tokens per cycle
        let tokens = speculative_loop(
            &mut draft,
            0,
            3,
            10, // max_tokens
            &[999], // EOS that won't appear
            |batch| {
                // Echo the draft tokens + bonus token 50
                let mut target = batch[1..].to_vec();
                target.push(50);
                Ok(target)
            },
        )
        .unwrap();
        assert_eq!(tokens.len(), 10);
    }

    #[test]
    fn loop_stops_on_eos() {
        let mut draft = MockDraft::new(vec![1, 2, 0]); // 0 = EOS
        let tokens = speculative_loop(
            &mut draft,
            99,
            3,
            100,
            &[0], // EOS token
            |batch| {
                let mut target = batch[1..].to_vec();
                target.push(50);
                Ok(target)
            },
        )
        .unwrap();
        // Should stop when token 0 (EOS) is generated
        assert!(tokens.contains(&0));
        assert!(tokens.len() < 100);
    }

    #[test]
    fn loop_with_partial_accepts_still_progresses() {
        let mut draft = MockDraft::new(vec![1, 2, 3]);
        let tokens = speculative_loop(
            &mut draft,
            0,
            3,
            6,
            &[999],
            |batch| {
                // Always reject after first token → 1 token per cycle
                // Return K+1 target tokens where K = batch.len() - 1
                let k = batch.len() - 1;
                let mut target = vec![77]; // diverge immediately
                target.resize(k + 1, 0);
                Ok(target)
            },
        )
        .unwrap();
        assert_eq!(tokens.len(), 6);
        // Every token should be 77 (target's choice on first rejection)
        assert!(tokens.iter().all(|&t| t == 77));
    }

    #[test]
    fn loop_empty_max_tokens() {
        let mut draft = MockDraft::new(vec![1]);
        let tokens = speculative_loop(
            &mut draft, 0, 3, 0, &[], |_| unreachable!(),
        )
        .unwrap();
        assert!(tokens.is_empty());
    }
}
