//! Correct-by-construction ownership for target/dSpark retained state.

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use higgs_models::{
    AnyCache,
    dflash::{DFlashCache, DFlashSnapshot},
};

use super::disk_prefix_cache::hash_tokens;

/// Private identity for the exact token boundary represented by both caches.
///
/// The hash is only a fast rejection hint. Exact token equality is the
/// authority, so even an equal-length FNV collision cannot claim continuity.
#[derive(Debug, PartialEq, Eq)]
struct PrefixStamp {
    hash: u64,
    len: usize,
    tokens: Box<[u32]>,
}

impl PrefixStamp {
    fn new(tokens: &[u32]) -> Self {
        Self {
            hash: hash_tokens(tokens),
            len: tokens.len(),
            tokens: tokens.into(),
        }
    }

    fn matches(&self, tokens: &[u32]) -> bool {
        self.matches_hashed(tokens, hash_tokens(tokens))
    }

    fn matches_hashed(&self, tokens: &[u32], hash: u64) -> bool {
        self.len == tokens.len() && self.hash == hash && self.tokens.as_ref() == tokens
    }

    fn boundary(&self) -> Result<i32, PairedCacheError> {
        i32::try_from(self.len)
            .map_err(|_| PairedCacheError::PrefixLengthOverflow { len: self.len })
    }
}

/// Construction failures for a target/dFlash retained pair.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PairedCacheError {
    #[error("prefix length {len} exceeds the cache boundary range")]
    PrefixLengthOverflow { len: usize },
    #[error(
        "retained prefix does not match requested tokens (stored length {stored_len}, requested length {requested_len})"
    )]
    PrefixMismatch {
        stored_len: usize,
        requested_len: usize,
    },
    #[error("target cache does not represent absolute boundary {expected}: {details}")]
    TargetBoundary { expected: i32, details: String },
    #[error(
        "dFlash cache boundary {actual} does not match the retained prefix boundary {expected}"
    )]
    DFlashBoundary { expected: i32, actual: i32 },
    #[error("failed to fork retained dFlash state: {details}")]
    DFlashFork { details: String },
    #[error("failed to materialize retained target state: {details}")]
    TargetMaterialization { details: String },
    #[error(
        "prepared paired prefix belongs to cache instance {prepared}, current instance is {current}"
    )]
    ForeignCacheInstance { prepared: u64, current: u64 },
    #[error("prepared paired prefix belongs to cache epoch {prepared}, current epoch is {current}")]
    StaleEpoch { prepared: u64, current: u64 },
    #[error(
        "prepared paired prefix belongs to cache publication revision {prepared}, current revision is {current}"
    )]
    StaleRevision { prepared: u64, current: u64 },
}

/// Immutable identity and accounting shared by every ownership form.
///
/// Radix plans clone this `Arc` for lock-free exact-key selection while the
/// single target+dFlash payload remains protected by its mutex.
#[derive(Debug)]
struct PairMetadata {
    stamp: PrefixStamp,
    target_bytes: usize,
    dflash_bytes: usize,
}

/// The one validated target+dFlash ownership core.
///
/// Session retention owns this directly. Radix retention puts the same type
/// behind a mutex and shares only its immutable metadata with lookup plans.
#[derive(Debug)]
struct SealedPair {
    target: AnyCache,
    dflash: DFlashSnapshot,
    metadata: Arc<PairMetadata>,
}

impl SealedPair {
    fn new(
        target: AnyCache,
        dflash: DFlashSnapshot,
        tokens: &[u32],
    ) -> Result<Self, PairedCacheError> {
        let stamp = PrefixStamp::new(tokens);
        let expected = stamp.boundary()?;
        Self::validate_boundaries(&target, dflash.position(), expected)?;
        let metadata = Arc::new(PairMetadata {
            target_bytes: target.estimated_bytes(),
            dflash_bytes: dflash.estimated_bytes(),
            stamp,
        });
        Ok(Self {
            target,
            dflash,
            metadata,
        })
    }

    fn validate_boundaries(
        target: &AnyCache,
        dflash_position: i32,
        expected: i32,
    ) -> Result<(), PairedCacheError> {
        target
            .validate_absolute_boundary(expected)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            })?;
        if dflash_position != expected {
            return Err(PairedCacheError::DFlashBoundary {
                expected,
                actual: dflash_position,
            });
        }
        Ok(())
    }

    #[must_use]
    #[cfg(test)]
    fn prefix_len(&self) -> usize {
        self.metadata.stamp.len
    }

    #[must_use]
    fn matches_prefix(&self, tokens: &[u32]) -> bool {
        self.metadata.stamp.matches(tokens)
    }

    #[cfg(test)]
    fn into_live(
        self,
        expected_tokens: &[u32],
    ) -> Result<(AnyCache, DFlashCache), PairedCacheError> {
        if !self.matches_prefix(expected_tokens) {
            return Err(PairedCacheError::PrefixMismatch {
                stored_len: self.prefix_len(),
                requested_len: expected_tokens.len(),
            });
        }
        Ok(self.into_live_unchecked())
    }

    fn into_live_unchecked(self) -> (AnyCache, DFlashCache) {
        let Self {
            target,
            dflash,
            metadata: _,
        } = self;
        (target, dflash.into_live())
    }

    #[must_use]
    fn demote(self) -> AnyCache {
        let Self {
            target,
            dflash,
            metadata: _,
        } = self;
        drop(dflash);
        target
    }

    /// Fork both live halves while the caller holds the radix payload mutex.
    fn try_fork_live(&self) -> Result<(AnyCache, DFlashCache), PairedCacheError> {
        let target = self.target.try_deep_clone().map_err(|error| {
            PairedCacheError::TargetMaterialization {
                details: error.to_string(),
            }
        })?;
        let dflash = self
            .dflash
            .fork_live()
            .map_err(|error| PairedCacheError::DFlashFork {
                details: error.to_string(),
            })?;
        let expected = self.metadata.stamp.boundary()?;
        Self::validate_boundaries(&target, dflash.position(), expected)?;
        Ok((target, dflash))
    }

    #[must_use]
    fn metadata(&self) -> Arc<PairMetadata> {
        Arc::clone(&self.metadata)
    }
}

/// One immutable retained target/dFlash boundary.
///
/// Both halves are private and the type is not `Clone`, so callers cannot
/// publish or move one retained half independently of the other.
#[derive(Debug)]
pub(crate) struct PairedCache {
    sealed: SealedPair,
}

impl PairedCache {
    /// Validate and publish one exact shared target/dFlash token boundary.
    ///
    /// The caller must have evaluated `target` while holding the process MLX
    /// execution gate before transferring it here. `DFlashSnapshot` is already
    /// evaluated by `DFlashDrafter::seal_after_taps`.
    pub(crate) fn new(
        target: AnyCache,
        dflash: DFlashSnapshot,
        tokens: &[u32],
    ) -> Result<Self, PairedCacheError> {
        SealedPair::new(target, dflash, tokens).map(|sealed| Self { sealed })
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn prefix_len(&self) -> usize {
        self.sealed.prefix_len()
    }

    /// Revalidate the lookup key before this pair is selected for reuse.
    #[must_use]
    pub(crate) fn matches_prefix(&self, tokens: &[u32]) -> bool {
        self.sealed.matches_prefix(tokens)
    }

    /// Consume both immutable halves into one live target/dFlash branch.
    ///
    /// Rechecking the private stamp here prevents a stale session/radix lookup
    /// from turning a structurally valid pair into continuity for another key.
    #[cfg(test)]
    pub(crate) fn into_live(
        self,
        expected_tokens: &[u32],
    ) -> Result<(AnyCache, DFlashCache), PairedCacheError> {
        self.sealed.into_live(expected_tokens)
    }

    /// Explicitly abandon speculative continuity and retain only target state.
    ///
    /// Consuming `self` makes demotion a whole-pair ownership transition; the
    /// drafter snapshot cannot remain accidentally associated with the target.
    #[must_use]
    pub(crate) fn demote(self) -> AnyCache {
        self.sealed.demote()
    }

    fn into_live_unchecked(self) -> (AnyCache, DFlashCache) {
        self.sealed.into_live_unchecked()
    }
}

/// One Arc-owned frozen target+dFlash boundary for radix reuse.
///
/// The payload mutex safely shares MLX's `!Sync` array handles. It is acquired
/// only after the radix lock has been released, and both halves are forked
/// while holding the same guard so they cannot be mixed across publications.
#[derive(Debug)]
pub(crate) struct RadixPairedSnapshot {
    sealed: Mutex<SealedPair>,
    metadata: Arc<PairMetadata>,
    #[cfg(test)]
    fail_next_fork: AtomicBool,
}

impl RadixPairedSnapshot {
    pub(crate) fn new(
        frozen_target: AnyCache,
        dflash: DFlashSnapshot,
        tokens: &[u32],
    ) -> Result<Self, PairedCacheError> {
        let sealed = SealedPair::new(frozen_target, dflash, tokens)?;
        let metadata = sealed.metadata();
        Ok(Self {
            sealed: Mutex::new(sealed),
            metadata,
            #[cfg(test)]
            fail_next_fork: AtomicBool::new(false),
        })
    }

    #[must_use]
    pub(crate) fn prefix_len(&self) -> usize {
        self.metadata.stamp.len
    }

    #[must_use]
    pub(crate) fn matches_prefix(&self, tokens: &[u32]) -> bool {
        self.metadata.stamp.matches(tokens)
    }

    #[must_use]
    pub(crate) fn target_bytes(&self) -> usize {
        self.metadata.target_bytes
    }

    #[must_use]
    pub(crate) fn dflash_bytes(&self) -> usize {
        self.metadata.dflash_bytes
    }

    /// Select the exact frozen pair without doing any MLX work.
    ///
    /// The returned plan owns the snapshot through an `Arc`, so callers may
    /// release the radix/prefix mutex before forking both live halves.
    pub(crate) fn plan_fork(
        self: &Arc<Self>,
        expected_tokens: &[u32],
    ) -> Result<RadixPairedForkPlan, PairedCacheError> {
        if !self.matches_prefix(expected_tokens) {
            return Err(PairedCacheError::PrefixMismatch {
                stored_len: self.prefix_len(),
                requested_len: expected_tokens.len(),
            });
        }
        Ok(RadixPairedForkPlan {
            snapshot: Arc::clone(self),
        })
    }

    fn fork_live(&self) -> Result<(AnyCache, DFlashCache), PairedCacheError> {
        debug_assert!(
            higgs_models::mlx_exec::held(),
            "radix paired fork requires the process MLX execution gate"
        );
        #[cfg(test)]
        if self.fail_next_fork.swap(false, Ordering::SeqCst) {
            return Err(PairedCacheError::TargetMaterialization {
                details: "injected paired fork failure".to_owned(),
            });
        }
        let sealed = self
            .sealed
            .lock()
            .map_err(|error| PairedCacheError::DFlashFork {
                details: format!("retained paired snapshot lock is poisoned: {error}"),
            })?;
        sealed.try_fork_live()
    }

    #[cfg(test)]
    fn fail_next_fork_for_test(&self) {
        self.fail_next_fork.store(true, Ordering::SeqCst);
    }
}

/// Owned, exact-identity plan for a post-prefix-lock paired fork.
#[derive(Debug)]
pub(crate) struct RadixPairedForkPlan {
    snapshot: Arc<RadixPairedSnapshot>,
}

impl RadixPairedForkPlan {
    #[must_use]
    pub(crate) fn prefix_len(&self) -> usize {
        self.snapshot.prefix_len()
    }

    pub(crate) fn materialize(self) -> Result<(AnyCache, DFlashCache), PairedCacheError> {
        self.snapshot.fork_live()
    }

    #[cfg(test)]
    pub(crate) fn fail_materialization_for_test(&self) {
        self.snapshot.fail_next_fork_for_test();
    }
}

/// Session/radix retained ownership, with paired state represented atomically.
#[derive(Debug)]
pub(crate) enum RetainedState {
    TargetOnly(AnyCache),
    Paired(PairedCache),
}

impl RetainedState {
    pub(crate) fn paired(
        target: AnyCache,
        dflash: DFlashSnapshot,
        tokens: &[u32],
    ) -> Result<Self, PairedCacheError> {
        PairedCache::new(target, dflash, tokens).map(Self::Paired)
    }

    /// Consume paired state only when it still matches the requested key.
    ///
    /// Target-only state and mismatched pairs are returned intact so the
    /// caller may explicitly demote or discard them.
    pub(crate) fn into_paired(
        self,
        expected_tokens: &[u32],
    ) -> Result<(AnyCache, DFlashCache), Self> {
        match self {
            Self::TargetOnly(target) => Err(Self::TargetOnly(target)),
            Self::Paired(pair) if pair.matches_prefix(expected_tokens) => {
                Ok(pair.into_live_unchecked())
            }
            Self::Paired(pair) => Err(Self::Paired(pair)),
        }
    }

    /// Consume the retained state and return target-only continuity.
    #[must_use]
    pub(crate) fn demote(self) -> AnyCache {
        match self {
            Self::TargetOnly(target) => target,
            Self::Paired(pair) => pair.demote(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use higgs_models::{
        AnyCache,
        cache::SteppingKeyValueCache,
        dflash::{DFlashConfig, DFlashDrafter, DFlashSnapshot},
    };
    use mlx_rs::Array;

    use super::{PairedCache, PairedCacheError, PrefixStamp, RetainedState};

    fn target_cache(boundary: i32) -> AnyCache {
        let layer = if boundary == 0 {
            SteppingKeyValueCache::new()
        } else {
            let keys = Array::zeros::<f32>(&[1, 1, boundary, 1]).unwrap();
            let values = Array::zeros::<f32>(&[1, 1, boundary, 1]).unwrap();
            SteppingKeyValueCache::from_arrays(keys, values).unwrap()
        };
        let cache = AnyCache::KV(vec![Some(layer)]);
        let _exec = higgs_models::mlx_exec::acquire();
        cache.eval().unwrap();
        cache
    }

    fn dflash_snapshot(boundary: i32) -> DFlashSnapshot {
        let config: DFlashConfig = serde_json::from_str(
            r#"{
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 8,
                "vocab_size": 8,
                "dflash_config": {
                    "target_layer_ids": [0]
                }
            }"#,
        )
        .unwrap();
        let mut drafter = DFlashDrafter::new(config).unwrap();
        let cache = drafter.make_cache();
        let taps = (boundary == 1)
            .then(|| Array::zeros::<f32>(&[1, 1, 4]).unwrap())
            .into_iter()
            .collect::<Vec<_>>();
        let _exec = higgs_models::mlx_exec::acquire();
        drafter.seal_after_taps(cache, &taps, boundary).unwrap()
    }

    #[test]
    fn paired_cache_accepts_one_exact_shared_boundary() {
        let pair = PairedCache::new(target_cache(0), dflash_snapshot(0), &[]).unwrap();

        assert_eq!(pair.prefix_len(), 0);
        assert!(pair.matches_prefix(&[]));
    }

    #[test]
    fn prefix_length_must_fit_the_model_boundary_type() {
        let len = usize::try_from(i32::MAX).unwrap() + 1;
        let stamp = PrefixStamp {
            hash: 0,
            len,
            tokens: Vec::new().into_boxed_slice(),
        };

        assert_eq!(
            stamp.boundary().unwrap_err(),
            PairedCacheError::PrefixLengthOverflow { len }
        );
    }

    #[test]
    fn paired_cache_rejects_target_boundary_mismatch() {
        let error = PairedCache::new(target_cache(0), dflash_snapshot(1), &[11]).unwrap_err();

        assert!(matches!(
            error,
            PairedCacheError::TargetBoundary { expected: 1, .. }
        ));
    }

    #[test]
    fn paired_cache_rejects_drafter_boundary_mismatch() {
        let error = PairedCache::new(target_cache(1), dflash_snapshot(0), &[11]).unwrap_err();

        assert_eq!(
            error,
            PairedCacheError::DFlashBoundary {
                expected: 1,
                actual: 0
            }
        );
    }

    #[test]
    fn same_length_different_prefix_does_not_match() {
        let pair = PairedCache::new(target_cache(1), dflash_snapshot(1), &[11]).unwrap();

        assert!(!pair.matches_prefix(&[12]));
    }

    #[test]
    fn exact_prefix_identity_rejects_a_simulated_hash_collision() {
        let stamp = PrefixStamp::new(&[11, 22]);

        assert!(
            !stamp.matches_hashed(&[11, 23], stamp.hash),
            "equal hash and length must not substitute for exact token equality"
        );
    }

    #[test]
    fn consuming_live_reuse_rechecks_the_exact_prefix() {
        let pair = PairedCache::new(target_cache(1), dflash_snapshot(1), &[11]).unwrap();

        let error = pair.into_live(&[12]).unwrap_err();
        assert_eq!(
            error,
            PairedCacheError::PrefixMismatch {
                stored_len: 1,
                requested_len: 1
            }
        );
    }

    #[test]
    fn consuming_live_reuse_moves_both_caches_together() {
        let pair = PairedCache::new(target_cache(1), dflash_snapshot(1), &[11]).unwrap();

        let (target, dflash) = pair.into_live(&[11]).unwrap();
        target.validate_absolute_boundary(1).unwrap();
        assert_eq!(dflash.position(), 1);
    }

    #[test]
    fn retained_pair_can_only_be_demoted_by_consuming_the_whole_pair() {
        let retained = RetainedState::paired(target_cache(0), dflash_snapshot(0), &[]).unwrap();

        assert!(matches!(retained, RetainedState::Paired(_)));
        let target = retained.demote();
        target.validate_absolute_boundary(0).unwrap();
    }

    #[test]
    fn target_only_state_demotes_without_special_cases() {
        let retained = RetainedState::TargetOnly(target_cache(0));

        retained.demote().validate_absolute_boundary(0).unwrap();
    }

    #[test]
    fn retained_reuse_returns_nonmatching_state_intact() {
        let retained = RetainedState::paired(target_cache(1), dflash_snapshot(1), &[11]).unwrap();

        let retained = retained.into_paired(&[12]).unwrap_err();
        assert!(matches!(retained, RetainedState::Paired(_)));
        retained.demote().validate_absolute_boundary(1).unwrap();
    }

    #[test]
    fn retained_reuse_moves_a_matching_pair_together() {
        let retained = RetainedState::paired(target_cache(1), dflash_snapshot(1), &[11]).unwrap();

        let (target, dflash) = retained.into_paired(&[11]).unwrap();
        target.validate_absolute_boundary(1).unwrap();
        assert_eq!(dflash.position(), 1);
    }
}
