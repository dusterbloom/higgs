//! Draft model shim that adapts `AneBonsaiEngine` to the `DraftModel` trait.
//!
//! Bonsai on ANE is stateless — every forward re-runs the full sequence
//! through 28 attention kernels + OC-tiled FFN. This wrapper keeps a
//! running committed-token list and re-invokes `forward_last` once per
//! drafted token. Rollback is free (no KV cache to undo); advance pushes
//! the prior cycle's `last_token_id` + first N drafts onto the committed
//! list.
//!
//! Context is truncated from the front to `seq_len - num_draft` when it
//! would overflow the fixed ANE kernel seq length.

use crate::error::EngineError;
use crate::speculative::DraftModel;
use higgs_models::diffusion::AneBonsaiEngine;

pub struct AneBonsaiDraftModel {
    engine: AneBonsaiEngine,
    /// Prompt + all tokens confirmed by advance(). Does NOT include the
    /// `last_token_id` of an in-flight draft() call (that lives in pending).
    committed: Vec<u32>,
    /// `(last_token_id, drafts)` from the most recent draft() call, awaiting
    /// advance()/rollback(). `None` after advance/rollback or before first draft.
    pending: Option<(u32, Vec<u32>)>,
}

impl AneBonsaiDraftModel {
    pub fn new(engine: AneBonsaiEngine) -> Self {
        Self {
            engine,
            committed: Vec::new(),
            pending: None,
        }
    }

    fn seq_len(&self) -> usize {
        self.engine.seq_len
    }

    /// Build the input context for the next `forward_last` call:
    /// committed + [last_token_id] + drafts_so_far, trimmed from the front
    /// to fit `seq_len`.
    fn build_ctx(&self, last_token_id: u32, drafts_so_far: &[u32]) -> Vec<u32> {
        let cap = self.seq_len();
        let total = self.committed.len() + 1 + drafts_so_far.len();
        let start = total.saturating_sub(cap);
        let mut ctx = Vec::with_capacity(total - start);
        let committed_start = start.min(self.committed.len());
        ctx.extend_from_slice(&self.committed[committed_start..]);
        if start <= self.committed.len() {
            ctx.push(last_token_id);
            ctx.extend_from_slice(drafts_so_far);
        } else if start == self.committed.len() + 1 {
            ctx.extend_from_slice(drafts_so_far);
        } else {
            let drafts_start = start - self.committed.len() - 1;
            ctx.extend_from_slice(&drafts_so_far[drafts_start..]);
        }
        ctx
    }
}

fn argmax(logits: &[f32]) -> Result<u32, EngineError> {
    if logits.is_empty() {
        return Err(EngineError::Generation(
            "AneBonsaiDraftModel: empty logits".into(),
        ));
    }
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    u32::try_from(best_idx)
        .map_err(|_| EngineError::Generation("AneBonsaiDraftModel: token id overflow".into()))
}

impl DraftModel for AneBonsaiDraftModel {
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation(
                "AneBonsaiDraftModel::prefill: empty prompt".into(),
            ));
        }
        self.committed = prompt_tokens.to_vec();
        self.pending = None;
        Ok(())
    }

    fn draft(&mut self, last_token_id: u32, num_draft: usize) -> Result<Vec<u32>, EngineError> {
        if self.committed.is_empty() {
            return Err(EngineError::Generation(
                "AneBonsaiDraftModel::draft called before prefill".into(),
            ));
        }
        if num_draft == 0 {
            self.pending = Some((last_token_id, Vec::new()));
            return Ok(Vec::new());
        }

        let mut drafts: Vec<u32> = Vec::with_capacity(num_draft);
        for _ in 0..num_draft {
            let ctx = self.build_ctx(last_token_id, &drafts);
            if ctx.len() > self.seq_len() {
                return Err(EngineError::Generation(format!(
                    "AneBonsaiDraftModel: ctx len {} > engine seq_len {}",
                    ctx.len(),
                    self.seq_len(),
                )));
            }
            let logits = self.engine.forward_last(&ctx);
            drafts.push(argmax(&logits)?);
        }

        self.pending = Some((last_token_id, drafts.clone()));
        Ok(drafts)
    }

    fn advance(&mut self, n: usize) -> Result<(), EngineError> {
        let (last, drafts) = self.pending.take().ok_or_else(|| {
            EngineError::Generation("AneBonsaiDraftModel::advance without prior draft".into())
        })?;
        if n > drafts.len() {
            return Err(EngineError::Generation(format!(
                "AneBonsaiDraftModel::advance n={n} > drafts={}",
                drafts.len()
            )));
        }
        self.committed.push(last);
        self.committed.extend_from_slice(&drafts[..n]);
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), EngineError> {
        // Pending draft discarded entirely. The target's divergent token
        // will come in as the next cycle's `last_token_id`.
        self.pending = None;
        Ok(())
    }
}

// SAFETY: AneBonsaiEngine contains AneKernel which is !Send due to IOSurface
// thread-affinity. The DraftModel trait requires Send because SimpleEngine
// wraps the draft in Arc<Mutex<Box<dyn DraftModel>>>. The mutex serializes
// all access, so at any instant only one thread holds the engine. This
// matches the pattern used for CachedMetalKernel, AneProjKernel, etc. in
// higgs-models where the underlying handles are likewise serialized by an
// external lock before being accessed.
#[allow(unsafe_code)]
unsafe impl Send for AneBonsaiDraftModel {}

const _: () = {
    fn _assert_send<T: DraftModel + Send>() {}
    fn _assert() {
        _assert_send::<AneBonsaiDraftModel>();
    }
};

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    // build_ctx is pure logic — test it with a dummy struct that mimics layout.
    // We can't construct AneBonsaiEngine without real weights, so exercise
    // build_ctx indirectly via a helper that takes the committed list directly.

    fn build_ctx_pure(
        committed: &[u32],
        cap: usize,
        last_token_id: u32,
        drafts_so_far: &[u32],
    ) -> Vec<u32> {
        let total = committed.len() + 1 + drafts_so_far.len();
        let start = total.saturating_sub(cap);
        let mut ctx = Vec::with_capacity(total - start);
        let committed_start = start.min(committed.len());
        ctx.extend_from_slice(&committed[committed_start..]);
        if start <= committed.len() {
            ctx.push(last_token_id);
            ctx.extend_from_slice(drafts_so_far);
        } else if start == committed.len() + 1 {
            ctx.extend_from_slice(drafts_so_far);
        } else {
            let drafts_start = start - committed.len() - 1;
            ctx.extend_from_slice(&drafts_so_far[drafts_start..]);
        }
        ctx
    }

    #[test]
    fn ctx_fits_no_truncation() {
        let ctx = build_ctx_pure(&[1, 2, 3], 16, 9, &[10, 11]);
        assert_eq!(ctx, vec![1, 2, 3, 9, 10, 11]);
    }

    #[test]
    fn ctx_exactly_at_capacity() {
        let ctx = build_ctx_pure(&[1, 2, 3], 6, 9, &[10, 11]);
        assert_eq!(ctx, vec![1, 2, 3, 9, 10, 11]);
        assert_eq!(ctx.len(), 6);
    }

    #[test]
    fn ctx_truncates_committed_prefix() {
        // committed=4, last=1, drafts=2 → total=7, cap=5 → drop 2 from front
        let ctx = build_ctx_pure(&[1, 2, 3, 4], 5, 9, &[10, 11]);
        assert_eq!(ctx, vec![3, 4, 9, 10, 11]);
    }

    #[test]
    fn ctx_truncates_past_last_token() {
        // committed=2, last=1, drafts=3 → total=6, cap=3 → drop 3, keeps last 3
        let ctx = build_ctx_pure(&[1, 2], 3, 9, &[10, 11, 12]);
        assert_eq!(ctx, vec![10, 11, 12]);
    }

    #[test]
    fn ctx_drops_committed_but_keeps_last_and_drafts() {
        // committed=2, last=1, drafts=3 → total=6, cap=4 → drop both committed, keep last + 3 drafts
        let ctx = build_ctx_pure(&[1, 2], 4, 9, &[10, 11, 12]);
        assert_eq!(ctx, vec![9, 10, 11, 12]);
        assert_eq!(ctx.len(), 4);
    }

    #[test]
    fn ctx_drops_into_drafts_region() {
        // committed=1, last=1, drafts=4 → total=6, cap=3 → drop 3, drop last too, keep drafts[1..]
        let ctx = build_ctx_pure(&[5], 3, 9, &[10, 11, 12, 13]);
        assert_eq!(ctx, vec![11, 12, 13]);
    }
}
