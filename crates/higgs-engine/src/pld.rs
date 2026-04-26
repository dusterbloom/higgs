//! Prompt Lookup Decoding (PLD) drafter.
//!
//! Proposes draft tokens by matching the trailing n-gram of the running
//! sequence against earlier occurrences in the same sequence (prompt +
//! already-generated). When a match is found, the tokens immediately
//! following the match in the corpus are returned as the draft.
//!
//! No model, no weights, no GPU calls. The lookup is microseconds; the
//! verify step still runs the target model on the proposed batch.
//!
//! When no match is found, `draft()` returns an empty `Vec`. The verify
//! step then runs at length-1, equivalent to plain autoregressive decode.
//!
//! Reference: Apoorv Saxena, "Prompt Lookup Decoding"
//! (github.com/apoorvumang/prompt-lookup-decoding).

use crate::error::EngineError;
use crate::speculative::DraftModel;

pub struct PldDraftModel {
    /// Search corpus: prompt tokens plus every token confirmed by `advance()`.
    /// The next cycle's `last_token_id` is appended onto a temporary view
    /// during `draft()` so the suffix-match window includes it.
    committed: Vec<u32>,
    /// `(last_token_id, drafts)` from the most recent `draft()` call,
    /// awaiting `advance()` / `rollback()`. Mirrors `AneBonsaiDraftModel`.
    pending: Option<(u32, Vec<u32>)>,
    max_ngram: usize,
    min_ngram: usize,
}

impl PldDraftModel {
    pub fn new(max_ngram: usize, min_ngram: usize) -> Self {
        Self {
            committed: Vec::new(),
            pending: None,
            max_ngram,
            min_ngram,
        }
    }

    /// Find the rightmost occurrence of `pattern` inside `corpus[..corpus.len() - pattern.len()]`
    /// and return up to `num_draft` tokens immediately after the match.
    /// Returns `None` if no match exists or no continuation tokens are available.
    fn lookup(corpus: &[u32], pattern_len: usize, num_draft: usize) -> Option<Vec<u32>> {
        if pattern_len == 0 || pattern_len >= corpus.len() {
            return None;
        }
        let pattern_start = corpus.len() - pattern_len;
        let pattern = &corpus[pattern_start..];
        // Search rightmost-first so the most-recent match wins.
        let search_end = pattern_start;
        if search_end < pattern_len {
            return None;
        }
        for start in (0..=search_end - pattern_len).rev() {
            if &corpus[start..start + pattern_len] == pattern {
                let cont_start = start + pattern_len;
                let cont_end = cont_start.saturating_add(num_draft).min(corpus.len());
                if cont_end <= cont_start {
                    return None;
                }
                return Some(corpus[cont_start..cont_end].to_vec());
            }
        }
        None
    }
}

impl DraftModel for PldDraftModel {
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation(
                "PldDraftModel::prefill: empty prompt".into(),
            ));
        }
        if self.min_ngram == 0 || self.min_ngram > self.max_ngram {
            return Err(EngineError::Generation(format!(
                "PldDraftModel: invalid ngram window min={} max={}",
                self.min_ngram, self.max_ngram,
            )));
        }
        self.committed = prompt_tokens.to_vec();
        self.pending = None;
        Ok(())
    }

    fn draft(&mut self, last_token_id: u32, num_draft: usize) -> Result<Vec<u32>, EngineError> {
        if self.committed.is_empty() {
            return Err(EngineError::Generation(
                "PldDraftModel::draft called before prefill".into(),
            ));
        }
        if num_draft == 0 {
            self.pending = Some((last_token_id, Vec::new()));
            return Ok(Vec::new());
        }

        // Build a search corpus that includes `last_token_id` in the suffix
        // so the match window spans the most recent token. We don't mutate
        // `committed` — the corpus reverts on rollback by being thrown away.
        let mut corpus = Vec::with_capacity(self.committed.len() + 1);
        corpus.extend_from_slice(&self.committed);
        corpus.push(last_token_id);

        let mut drafts = Vec::new();
        let max_n = self.max_ngram.min(corpus.len().saturating_sub(1));
        for n in (self.min_ngram..=max_n).rev() {
            if let Some(found) = Self::lookup(&corpus, n, num_draft) {
                drafts = found;
                break;
            }
        }

        self.pending = Some((last_token_id, drafts.clone()));
        Ok(drafts)
    }

    fn advance(&mut self, n: usize) -> Result<(), EngineError> {
        let (last, drafts) = self.pending.take().ok_or_else(|| {
            EngineError::Generation("PldDraftModel::advance without prior draft".into())
        })?;
        if n > drafts.len() {
            return Err(EngineError::Generation(format!(
                "PldDraftModel::advance n={n} > drafts={}",
                drafts.len()
            )));
        }
        self.committed.push(last);
        self.committed.extend_from_slice(&drafts[..n]);
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), EngineError> {
        // The target's divergent token arrives as the next cycle's
        // `last_token_id` and gets folded into `committed` on the next
        // `advance()` — same self-healing pattern as AneBonsaiDraftModel.
        self.pending = None;
        Ok(())
    }
}

const _: () = {
    fn _assert_send<T: DraftModel + Send>() {}
    fn _assert() {
        _assert_send::<PldDraftModel>();
    }
};

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    // Convention used in the comments below:
    //   corpus = committed + [last_token_id]
    //   pattern (n-gram) = last n tokens of corpus
    //   search range = corpus[..corpus.len() - n]
    //   continuation = corpus[match_end..match_end + num_draft],
    //                  capped at corpus.len().

    fn fresh(max_ngram: usize, min_ngram: usize, prompt: &[u32]) -> PldDraftModel {
        let mut m = PldDraftModel::new(max_ngram, min_ngram);
        m.prefill(prompt).unwrap();
        m
    }

    #[test]
    fn no_match_returns_empty() {
        // corpus = [1,2,3,4,99]; no suffix at any n recurs.
        let mut m = fresh(3, 1, &[1, 2, 3, 4]);
        assert!(m.draft(99, 5).unwrap().is_empty());
    }

    #[test]
    fn single_match_returns_continuation() {
        // corpus = [10,20,30,40,50,10,20]; pattern "10 20" matches at idx 0.
        // continuation = corpus[2..5] = [30,40,50].
        let mut m = fresh(2, 1, &[10, 20, 30, 40, 50, 10]);
        assert_eq!(m.draft(20, 3).unwrap(), vec![30, 40, 50]);
    }

    #[test]
    fn multiple_matches_picks_rightmost() {
        // corpus = [10,20,1,2,30,10,20,40,50,10,20]; pattern "10 20" matches
        // at idx 0 (cont=[1,2,30]) and idx 5 (cont=[40,50,10]).
        // Rightmost → [40,50,10].
        let mut m = fresh(2, 1, &[10, 20, 1, 2, 30, 10, 20, 40, 50, 10]);
        assert_eq!(m.draft(20, 3).unwrap(), vec![40, 50, 10]);
    }

    #[test]
    fn descending_ngram_falls_back_to_shorter() {
        // corpus = [6,7,99,100,5,6,7]; 3-gram "5,6,7" does not recur but
        // 2-gram "6,7" matches at idx 0; continuation = [99,100,5].
        let mut m = fresh(3, 1, &[6, 7, 99, 100, 5, 6]);
        assert_eq!(m.draft(7, 3).unwrap(), vec![99, 100, 5]);
    }

    #[test]
    fn match_at_corpus_end_returns_empty() {
        // corpus = [1,2,3,4]; only matches would put cont_start at corpus.len(),
        // yielding zero continuation tokens.
        let mut m = fresh(2, 1, &[1, 2, 3]);
        assert!(m.draft(4, 3).unwrap().is_empty());
    }

    #[test]
    fn truncates_to_num_draft() {
        // corpus = [1,2,9,8,7,6,5,1,2]; pattern "1,2" matches at idx 0;
        // full continuation = [9,8,7,6,5,1,2] (7 tokens), truncated to 2.
        let mut m = fresh(2, 1, &[1, 2, 9, 8, 7, 6, 5, 1]);
        assert_eq!(m.draft(2, 2).unwrap(), vec![9, 8]);
    }

    #[test]
    fn advance_extends_committed_with_last_and_drafts() {
        let mut m = fresh(2, 1, &[10, 20, 30, 40, 50, 10]);
        let drafts = m.draft(20, 3).unwrap();
        assert_eq!(drafts, vec![30, 40, 50]);
        m.advance(2).unwrap();
        assert_eq!(m.committed, vec![10, 20, 30, 40, 50, 10, 20, 30, 40]);
        assert!(m.pending.is_none());
    }

    #[test]
    fn rollback_clears_pending_only_committed_unchanged() {
        let mut m = fresh(2, 1, &[10, 20, 30, 40, 50, 10]);
        let _ = m.draft(20, 3).unwrap();
        let snapshot = m.committed.clone();
        m.rollback().unwrap();
        assert_eq!(m.committed, snapshot);
        assert!(m.pending.is_none());
    }

    #[test]
    fn draft_before_prefill_errors() {
        let mut m = PldDraftModel::new(3, 1);
        let err = m.draft(1, 3).unwrap_err();
        assert!(err.to_string().contains("before prefill"));
    }

    #[test]
    fn empty_prompt_errors() {
        let mut m = PldDraftModel::new(3, 1);
        let err = m.prefill(&[]).unwrap_err();
        assert!(err.to_string().contains("empty prompt"));
    }

    #[test]
    fn invalid_ngram_window_errors() {
        let mut m = PldDraftModel::new(2, 5);
        let err = m.prefill(&[1, 2, 3]).unwrap_err();
        assert!(err.to_string().contains("invalid ngram"));

        let mut m2 = PldDraftModel::new(3, 0);
        let err2 = m2.prefill(&[1, 2, 3]).unwrap_err();
        assert!(err2.to_string().contains("invalid ngram"));
    }

    #[test]
    fn advance_without_prior_draft_errors() {
        let mut m = fresh(2, 1, &[1, 2, 3]);
        let err = m.advance(0).unwrap_err();
        assert!(err.to_string().contains("without prior draft"));
    }

    #[test]
    fn advance_n_larger_than_drafts_errors() {
        let mut m = fresh(2, 1, &[10, 20, 30, 40, 50, 10]);
        let _ = m.draft(20, 3).unwrap();
        let err = m.advance(99).unwrap_err();
        assert!(err.to_string().contains("n=99"));
    }

    #[test]
    fn num_draft_zero_returns_empty() {
        let mut m = fresh(2, 1, &[1, 2, 3]);
        let drafts = m.draft(4, 0).unwrap();
        assert!(drafts.is_empty());
        // pending still set so advance(0) is legal — folds last into committed.
        m.advance(0).unwrap();
        assert_eq!(m.committed, vec![1, 2, 3, 4]);
    }
}
