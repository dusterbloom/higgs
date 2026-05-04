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

    let mut verify_batch = Vec::with_capacity(k + 1);
    verify_batch.push(last_token_id);
    verify_batch.extend_from_slice(&draft_ids);

    let target_ids = verify_fn(&verify_batch)?;
    let accepted = accept_prefix(&draft_ids, &target_ids)?;

    let matched = if accepted.len() > k {
        k
    } else {
        accepted.len().saturating_sub(1)
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
    pub tokens: Vec<u32>,
    pub hit_eos: bool,
}

/// Run a full speculative decode loop until EOS or `max_tokens`.
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

    // Walk both slices in lock-step: types enforce bounds, no indexing needed.
    // The k+1th target token (the bonus when every draft matched) is appended
    // after the loop using `.last()` on target_ids.
    let mut accepted = Vec::with_capacity(k + 1);
    for (&target_token, &draft_token) in target_ids.iter().zip(draft_ids.iter()) {
        accepted.push(target_token);
        if target_token != draft_token {
            return Ok(accepted);
        }
    }
    // All k draft tokens matched — append the verify model's k+1th sample.
    // Safe because we validated `target_ids.len() == k + 1` above; the
    // .last() returns Some unless the slice is empty (impossible for k+1≥1).
    if let Some(&bonus_token) = target_ids.last() {
        accepted.push(bonus_token);
    }
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

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::unreachable
)]
mod tests {
    use super::*;

    #[test]
    fn accept_prefix_all_match_returns_k_plus_one() {
        let accepted = accept_prefix(&[5, 3, 7], &[5, 3, 7, 42]).unwrap();
        assert_eq!(accepted, vec![5, 3, 7, 42]);
    }

    #[test]
    fn accept_prefix_first_mismatch_returns_one() {
        let accepted = accept_prefix(&[5, 3, 7], &[9, 1, 2, 0]).unwrap();
        assert_eq!(accepted, vec![9]);
    }

    #[test]
    fn accept_prefix_mid_mismatch() {
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
        let accepted = accept_prefix(&[], &[42]).unwrap();
        assert_eq!(accepted, vec![42]);
    }

    #[test]
    fn accept_prefix_wrong_length_errors() {
        let err = accept_prefix(&[1, 2], &[1, 2]).unwrap_err();
        assert!(err.to_string().contains("must be"));
    }

    struct MockDraft {
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

        fn draft(
            &mut self,
            _last_token_id: u32,
            num_draft: usize,
        ) -> Result<Vec<u32>, EngineError> {
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
        assert_eq!(tokens, vec![10, 20]);
    }

    #[test]
    fn step_all_accepted_returns_k_plus_one() {
        let mut draft = MockDraft::new(vec![10, 20, 30]);
        let accepted = speculative_step(&mut draft, 0, 3, |batch| {
            assert_eq!(batch, &[0, 10, 20, 30]);
            Ok(vec![10, 20, 30, 99])
        })
        .unwrap();
        assert_eq!(accepted, vec![10, 20, 30, 99]);
        assert_eq!(draft.cursor, 0);
    }

    #[test]
    fn step_partial_accept_advances_draft() {
        let mut draft = MockDraft::new(vec![10, 20, 30]);
        let accepted = speculative_step(&mut draft, 0, 3, |_| Ok(vec![10, 20, 99, 55])).unwrap();
        assert_eq!(accepted, vec![10, 20, 99]);
        assert_eq!(draft.cursor, 2);
    }

    #[test]
    fn step_no_match_rollback() {
        let mut draft = MockDraft::new(vec![10, 20, 30]);
        let accepted = speculative_step(&mut draft, 0, 3, |_| Ok(vec![77, 0, 0, 0])).unwrap();
        assert_eq!(accepted, vec![77]);
        assert_eq!(draft.cursor, 0);
    }

    #[test]
    fn step_single_draft_match() {
        let mut draft = MockDraft::new(vec![10]);
        let accepted = speculative_step(&mut draft, 5, 1, |batch| {
            assert_eq!(batch, &[5, 10]);
            Ok(vec![10, 42])
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

    #[test]
    fn loop_generates_until_max_tokens() {
        let mut draft = MockDraft::new(vec![1, 2, 3]);
        let tokens = speculative_loop(&mut draft, 0, 3, 10, &[999], |batch| {
            let mut target = batch[1..].to_vec();
            target.push(50);
            Ok(target)
        })
        .unwrap();
        assert_eq!(tokens.len(), 10);
    }

    #[test]
    fn loop_stops_on_eos() {
        let mut draft = MockDraft::new(vec![1, 2, 0]);
        let tokens = speculative_loop(&mut draft, 99, 3, 100, &[0], |batch| {
            let mut target = batch[1..].to_vec();
            target.push(50);
            Ok(target)
        })
        .unwrap();
        assert!(tokens.contains(&0));
        assert!(tokens.len() < 100);
    }

    #[test]
    fn loop_with_partial_accepts_still_progresses() {
        let mut draft = MockDraft::new(vec![1, 2, 3]);
        let tokens = speculative_loop(&mut draft, 0, 3, 6, &[999], |batch| {
            let k = batch.len() - 1;
            let mut target = vec![77];
            target.resize(k + 1, 0);
            Ok(target)
        })
        .unwrap();
        assert_eq!(tokens.len(), 6);
        assert!(tokens.iter().all(|&t| t == 77));
    }

    #[test]
    fn loop_empty_max_tokens() {
        let mut draft = MockDraft::new(vec![1]);
        let tokens = speculative_loop(&mut draft, 0, 3, 0, &[], |_| unreachable!()).unwrap();
        assert!(tokens.is_empty());
    }
}
