//! Token-boundary accounting for cache-resident generation.

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_LEDGER_ID: AtomicU64 = AtomicU64::new(1);

/// The action required before this ledger can label a retained cache boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetentionAction {
    /// Every emitted token represented by the retention key has been forwarded.
    Ready { boundary: usize },
    /// Forward this visible non-EOS token once, without sampling another token.
    CacheOnlyForward { token: u32 },
    /// The trailing token is EOS: keep it in the response but exclude it from
    /// the retained key because it was never forwarded and is not conversational
    /// content.
    ExcludeEos { token: u32 },
    /// A cache-only forward has begun but has not yet succeeded.
    ForwardInFlight { token: u32 },
}

/// Token-ledger transition failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum LedgerError {
    #[error("token {token} is emitted but not forwarded")]
    PendingToken { token: u32 },
    #[error("cache-only forward for token {token} has not completed")]
    ForwardInFlight { token: u32 },
    #[error("no emitted pending token is available")]
    NoPendingToken,
    #[error("pending token {token} is not EOS")]
    PendingTokenIsNotEos { token: u32 },
    #[error("EOS token {token} was excluded; the ledger is terminal")]
    TerminalTokenExcluded { token: u32 },
    #[error("cache-only forward ticket belongs to another ledger or transition")]
    ForeignForwardTicket,
    #[error("token boundary overflow")]
    BoundaryOverflow,
    #[error("cache-only forward ticket counter overflow")]
    ForwardTicketOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TailState {
    Aligned,
    Pending { token: u32 },
    Forwarding { token: u32, ticket_id: u64 },
    ExcludedEos { token: u32 },
}

/// Opaque proof that a specific pending token entered a cache-only forward.
///
/// The ticket is intentionally neither `Clone` nor `Copy`: a successful model
/// forward consumes it exactly once when the ledger commits the token.
#[derive(Debug)]
pub(crate) struct ForwardTicket {
    ledger_id: u64,
    ticket_id: u64,
    token: u32,
}

/// Completion-token accounting for a cache-resident turn.
///
/// `emitted` is the response-visible sequence. `forwarded_len` is always a
/// prefix of it and is the only suffix allowed into a retention key. At most
/// one trailing emitted token may be pending; while it is pending or being
/// forwarded, [`Self::retainable_tokens`] fails closed.
#[derive(Debug)]
pub(crate) struct TokenLedger {
    base_boundary: usize,
    emitted: Vec<u32>,
    forwarded_len: usize,
    tail: TailState,
    ledger_id: u64,
    next_ticket_id: u64,
}

impl TokenLedger {
    #[must_use]
    pub(crate) fn new(base_boundary: usize) -> Self {
        Self {
            base_boundary,
            emitted: Vec::new(),
            forwarded_len: 0,
            tail: TailState::Aligned,
            ledger_id: NEXT_LEDGER_ID.fetch_add(1, Ordering::Relaxed),
            next_ticket_id: 1,
        }
    }

    /// Record a visible token whose target-cache transition already succeeded.
    pub(crate) fn record_forwarded(&mut self, token: u32) -> Result<(), LedgerError> {
        self.extend_forwarded([token])
    }

    /// Record a visible run already committed by a speculative verification
    /// transaction. This is atomic with respect to boundary overflow.
    pub(crate) fn extend_forwarded<I>(&mut self, tokens: I) -> Result<(), LedgerError>
    where
        I: IntoIterator<Item = u32>,
    {
        self.ensure_aligned_for_append()?;
        let incoming: Vec<u32> = tokens.into_iter().collect();
        let new_forwarded = self
            .forwarded_len
            .checked_add(incoming.len())
            .ok_or(LedgerError::BoundaryOverflow)?;
        self.base_boundary
            .checked_add(new_forwarded)
            .ok_or(LedgerError::BoundaryOverflow)?;
        self.emitted.extend(incoming);
        self.forwarded_len = new_forwarded;
        Ok(())
    }

    /// Emit one sampled token without claiming that the target cache contains
    /// it. No other token may be appended until this tail is forwarded or
    /// excluded as EOS.
    pub(crate) fn emit_pending(&mut self, token: u32) -> Result<(), LedgerError> {
        self.ensure_aligned_for_append()?;
        self.emitted.push(token);
        self.tail = TailState::Pending { token };
        Ok(())
    }

    /// Decide how to align the cache before publication.
    #[must_use]
    pub(crate) fn retention_action(&self, eos_token_ids: &[u32]) -> RetentionAction {
        match self.tail {
            TailState::Aligned | TailState::ExcludedEos { .. } => RetentionAction::Ready {
                boundary: self
                    .base_boundary
                    .checked_add(self.forwarded_len)
                    .unwrap_or(usize::MAX),
            },
            TailState::Pending { token } if eos_token_ids.contains(&token) => {
                RetentionAction::ExcludeEos { token }
            }
            TailState::Pending { token } => RetentionAction::CacheOnlyForward { token },
            TailState::Forwarding { token, .. } => RetentionAction::ForwardInFlight { token },
        }
    }

    /// Start the cache-only forward required for a visible non-EOS tail.
    pub(crate) fn begin_cache_only_forward(&mut self) -> Result<ForwardTicket, LedgerError> {
        let TailState::Pending { token } = self.tail else {
            return Err(self.tail_error());
        };
        let ticket_id = self.next_ticket_id;
        self.next_ticket_id = self
            .next_ticket_id
            .checked_add(1)
            .ok_or(LedgerError::ForwardTicketOverflow)?;
        self.tail = TailState::Forwarding { token, ticket_id };
        Ok(ForwardTicket {
            ledger_id: self.ledger_id,
            ticket_id,
            token,
        })
    }

    /// Commit a pending token only after its cache-only target forward succeeds.
    pub(crate) fn complete_cache_only_forward(
        &mut self,
        ticket: ForwardTicket,
    ) -> Result<(), LedgerError> {
        let TailState::Forwarding { token, ticket_id } = self.tail else {
            return Err(self.tail_error());
        };
        if ticket.ledger_id != self.ledger_id
            || ticket.ticket_id != ticket_id
            || ticket.token != token
        {
            return Err(LedgerError::ForeignForwardTicket);
        }
        let new_forwarded = self
            .forwarded_len
            .checked_add(1)
            .ok_or(LedgerError::BoundaryOverflow)?;
        self.base_boundary
            .checked_add(new_forwarded)
            .ok_or(LedgerError::BoundaryOverflow)?;
        if self.emitted.get(self.forwarded_len).copied() != Some(token) {
            return Err(LedgerError::ForeignForwardTicket);
        }
        self.forwarded_len = new_forwarded;
        self.tail = TailState::Aligned;
        Ok(())
    }

    /// Exclude an emitted-but-unforwarded EOS from the retention key while
    /// keeping it in the response-visible sequence.
    pub(crate) fn exclude_pending_eos(
        &mut self,
        eos_token_ids: &[u32],
    ) -> Result<u32, LedgerError> {
        let TailState::Pending { token } = self.tail else {
            return Err(self.tail_error());
        };
        if !eos_token_ids.contains(&token) {
            return Err(LedgerError::PendingTokenIsNotEos { token });
        }
        self.tail = TailState::ExcludedEos { token };
        Ok(token)
    }

    /// Response-visible completion tokens, including an excluded terminal EOS.
    #[must_use]
    pub(crate) fn emitted_tokens(&self) -> &[u32] {
        &self.emitted
    }

    /// The one emitted tail that is not yet known to be cache-resident.
    #[must_use]
    pub(crate) const fn pending_token(&self) -> Option<u32> {
        match self.tail {
            TailState::Pending { token } | TailState::Forwarding { token, .. } => Some(token),
            TailState::Aligned | TailState::ExcludedEos { .. } => None,
        }
    }

    /// Tokens safe to append to the prompt when constructing a retention key.
    ///
    /// This method is the publication gate: it never returns the emitted vector
    /// while a token is pending or an attempted forward is still in flight.
    pub(crate) fn retainable_tokens(&self) -> Result<&[u32], LedgerError> {
        match self.tail {
            TailState::Pending { token } => Err(LedgerError::PendingToken { token }),
            TailState::Forwarding { token, .. } => Err(LedgerError::ForwardInFlight { token }),
            TailState::Aligned | TailState::ExcludedEos { .. } => self
                .emitted
                .get(..self.forwarded_len)
                .ok_or(LedgerError::BoundaryOverflow),
        }
    }

    fn ensure_aligned_for_append(&self) -> Result<(), LedgerError> {
        match self.tail {
            TailState::Aligned => Ok(()),
            TailState::Pending { token } => Err(LedgerError::PendingToken { token }),
            TailState::Forwarding { token, .. } => Err(LedgerError::ForwardInFlight { token }),
            TailState::ExcludedEos { token } => Err(LedgerError::TerminalTokenExcluded { token }),
        }
    }

    const fn tail_error(&self) -> LedgerError {
        match self.tail {
            TailState::Aligned => LedgerError::NoPendingToken,
            TailState::Pending { token } => LedgerError::PendingToken { token },
            TailState::Forwarding { token, .. } => LedgerError::ForwardInFlight { token },
            TailState::ExcludedEos { token } => LedgerError::TerminalTokenExcluded { token },
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{LedgerError, RetentionAction, TokenLedger};

    #[test]
    fn pending_non_eos_requires_cache_only_forward_before_retention() {
        let mut ledger = TokenLedger::new(41);
        ledger.emit_pending(7).unwrap();

        assert_eq!(
            ledger.retention_action(&[99]),
            RetentionAction::CacheOnlyForward { token: 7 }
        );
        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::PendingToken { token: 7 }
        );

        let ticket = ledger.begin_cache_only_forward().unwrap();
        assert_eq!(
            ledger.retention_action(&[99]),
            RetentionAction::ForwardInFlight { token: 7 }
        );
        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::ForwardInFlight { token: 7 }
        );

        ledger.complete_cache_only_forward(ticket).unwrap();
        assert_eq!(
            ledger.retention_action(&[99]),
            RetentionAction::Ready { boundary: 42 }
        );
        assert_eq!(ledger.retainable_tokens().unwrap(), &[7]);
        assert_eq!(ledger.emitted_tokens(), &[7]);
    }

    #[test]
    fn emitted_eos_is_visible_but_excluded_from_the_cache_key() {
        let mut ledger = TokenLedger::new(12);
        ledger.record_forwarded(3).unwrap();
        ledger.emit_pending(99).unwrap();

        assert_eq!(
            ledger.retention_action(&[99, 100]),
            RetentionAction::ExcludeEos { token: 99 }
        );
        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::PendingToken { token: 99 }
        );

        assert_eq!(ledger.exclude_pending_eos(&[99, 100]).unwrap(), 99);
        assert_eq!(ledger.emitted_tokens(), &[3, 99]);
        assert_eq!(ledger.retainable_tokens().unwrap(), &[3]);
        assert_eq!(
            ledger.retention_action(&[99, 100]),
            RetentionAction::Ready { boundary: 13 }
        );
    }

    #[test]
    fn no_key_can_include_a_pending_or_in_flight_token() {
        let mut ledger = TokenLedger::new(5);
        ledger.extend_forwarded([10, 11]).unwrap();
        ledger.emit_pending(12).unwrap();

        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::PendingToken { token: 12 }
        );
        let ticket = ledger.begin_cache_only_forward().unwrap();
        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::ForwardInFlight { token: 12 }
        );
        ledger.complete_cache_only_forward(ticket).unwrap();
        assert_eq!(ledger.retainable_tokens().unwrap(), &[10, 11, 12]);
    }

    #[test]
    fn only_one_trailing_pending_token_is_permitted() {
        let mut ledger = TokenLedger::new(0);
        ledger.emit_pending(1).unwrap();
        assert_eq!(
            ledger.emit_pending(2).unwrap_err(),
            LedgerError::PendingToken { token: 1 }
        );
        assert_eq!(
            ledger.record_forwarded(2).unwrap_err(),
            LedgerError::PendingToken { token: 1 }
        );
    }

    #[test]
    fn eos_exclusion_rejects_a_non_eos_pending_token() {
        let mut ledger = TokenLedger::new(0);
        ledger.emit_pending(4).unwrap();
        assert_eq!(
            ledger.exclude_pending_eos(&[99]).unwrap_err(),
            LedgerError::PendingTokenIsNotEos { token: 4 }
        );
        assert_eq!(ledger.pending_token(), Some(4));
    }

    #[test]
    fn forward_tickets_are_bound_to_their_originating_ledger() {
        let mut left = TokenLedger::new(0);
        let mut right = TokenLedger::new(0);
        left.emit_pending(8).unwrap();
        right.emit_pending(8).unwrap();

        let left_ticket = left.begin_cache_only_forward().unwrap();
        let right_ticket = right.begin_cache_only_forward().unwrap();
        assert_eq!(
            right.complete_cache_only_forward(left_ticket).unwrap_err(),
            LedgerError::ForeignForwardTicket
        );
        right.complete_cache_only_forward(right_ticket).unwrap();
        assert_eq!(right.retainable_tokens().unwrap(), &[8]);
    }

    #[test]
    fn excluded_eos_is_terminal_for_the_ledger() {
        let mut ledger = TokenLedger::new(0);
        ledger.emit_pending(99).unwrap();
        ledger.exclude_pending_eos(&[99]).unwrap();

        assert_eq!(
            ledger.emit_pending(1).unwrap_err(),
            LedgerError::TerminalTokenExcluded { token: 99 }
        );
        assert_eq!(
            ledger.record_forwarded(1).unwrap_err(),
            LedgerError::TerminalTokenExcluded { token: 99 }
        );
    }

    #[test]
    fn boundary_overflow_is_reported_without_exposing_a_key() {
        let mut ledger = TokenLedger::new(usize::MAX);
        assert_eq!(
            ledger.record_forwarded(1).unwrap_err(),
            LedgerError::BoundaryOverflow
        );
        assert!(ledger.emitted_tokens().is_empty());
    }
}
