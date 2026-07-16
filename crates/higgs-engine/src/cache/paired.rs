//! Correct-by-construction ownership for target/dSpark retained state.

#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use higgs_models::{
    AnyCache,
    dflash::{DFlashCache, DFlashSnapshot},
};

use super::disk_prefix_cache::hash_tokens;

/// Unforgeable-within-this-module identity for one live target/dSpark branch.
///
/// A fresh epoch is minted only by [`LivePair::cold`]. The exact forwarded
/// token ledger and both cache halves move with that epoch until sealing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PairBranchEpoch(u64);

#[allow(dead_code)] // removed when the shared coordinator migrates to LivePair
static NEXT_PAIR_BRANCH_EPOCH: AtomicU64 = AtomicU64::new(1);

#[allow(dead_code)] // removed when the shared coordinator migrates to LivePair
fn next_pair_branch_epoch() -> PairBranchEpoch {
    PairBranchEpoch(NEXT_PAIR_BRANCH_EPOCH.fetch_add(1, Ordering::Relaxed))
}

/// Private identity for the exact token boundary represented by both caches.
///
/// The hash is only a fast rejection hint. Exact token equality is the
/// authority, so even an equal-length FNV collision cannot claim continuity.
#[derive(Debug, PartialEq, Eq)]
struct PrefixStamp {
    /// `Some` only for the correct-by-construction [`LivePair`] path.
    ///
    /// `None` identifies the temporary compatibility constructor used by call
    /// sites that have not yet migrated to the shared paired coordinator.
    branch_epoch: Option<PairBranchEpoch>,
    hash: u64,
    len: usize,
    tokens: Box<[u32]>,
}

impl PrefixStamp {
    /// Temporary compatibility stamp. This proves exact lookup identity, but
    /// cannot prove that independently supplied cache halves came from it.
    fn new(tokens: &[u32]) -> Self {
        Self {
            branch_epoch: None,
            hash: hash_tokens(tokens),
            len: tokens.len(),
            tokens: tokens.into(),
        }
    }

    #[allow(dead_code)] // removed when the shared coordinator migrates to LivePair
    fn from_live_branch(branch_epoch: PairBranchEpoch, tokens: Vec<u32>) -> Self {
        Self {
            branch_epoch: Some(branch_epoch),
            hash: hash_tokens(&tokens),
            len: tokens.len(),
            tokens: tokens.into_boxed_slice(),
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
    #[allow(dead_code)] // exercised by the staged LivePair path pending migration
    #[error("live paired-cache advance carries a target half from another branch")]
    ForeignTargetBranch,
    #[allow(dead_code)] // exercised by the staged LivePair path pending migration
    #[error("live paired-cache advance carries a dFlash half from another paired branch")]
    ForeignDFlashPairBranch,
    #[allow(dead_code)] // exercised by the staged LivePair path pending migration
    #[error("live paired-cache branch revision overflow")]
    BranchRevisionOverflow,
    #[allow(dead_code)] // exercised by the staged LivePair path pending migration
    #[error("failed to seal live dFlash branch: {details}")]
    DFlashSeal { details: String },
    #[allow(dead_code)] // exercised by the staged LivePair path pending migration
    #[error("dFlash snapshot was sealed from another same-position live branch")]
    ForeignDFlashBranch,
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
        Self::from_stamp(target, dflash, PrefixStamp::new(tokens))
    }

    #[allow(dead_code)] // removed when the shared coordinator migrates to LivePair
    fn from_live_branch(
        target: AnyCache,
        dflash: DFlashSnapshot,
        branch_epoch: PairBranchEpoch,
        tokens: Vec<u32>,
    ) -> Result<Self, PairedCacheError> {
        Self::from_stamp(
            target,
            dflash,
            PrefixStamp::from_live_branch(branch_epoch, tokens),
        )
    }

    fn from_stamp(
        target: AnyCache,
        dflash: DFlashSnapshot,
        stamp: PrefixStamp,
    ) -> Result<Self, PairedCacheError> {
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

    #[must_use]
    fn estimated_bytes(&self) -> (usize, usize) {
        (self.metadata.target_bytes, self.metadata.dflash_bytes)
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
}

/// Move-owned live target/dSpark branch with an exact forwarded-token ledger.
///
/// This is the first correct-by-construction publication path. Neither target,
/// drafter nor token label can be supplied independently at seal time:
///
/// - [`Self::cold`] mints the private branch epoch at boundary zero;
/// - [`Self::begin_aligned_advance`] consumes the pair into two stamped halves;
/// - [`LivePairAdvance::commit`] reunites only halves from that same epoch and
///   appends the exact forwarded tokens after both boundaries validate;
/// - [`Self::seal`] consumes the owned drafter cache and rejects a snapshot
///   sealed from any other live DFlash branch, even at the same position.
///
/// The shared prefill/decode coordinator still needs to migrate onto this type.
/// Until then, [`PairedCache::new`] remains an explicitly unproven compatibility
/// layer for existing call sites.
#[derive(Debug)]
#[allow(dead_code)] // first provenance slice; production migration follows
struct LiveTargetHalf {
    epoch: PairBranchEpoch,
    cache: AnyCache,
}

#[derive(Debug)]
#[allow(dead_code)] // first provenance slice; production migration follows
struct LiveDFlashHalf {
    epoch: PairBranchEpoch,
    cache: DFlashCache,
}

#[derive(Debug)]
#[allow(dead_code)] // first provenance slice; production migration follows
pub(crate) struct LivePair {
    epoch: PairBranchEpoch,
    revision: u64,
    target: LiveTargetHalf,
    dflash: LiveDFlashHalf,
    tokens: Vec<u32>,
}

/// Move-only in-flight transition for one exact live pair.
///
/// Both mutable cache halves retain their private pair epoch while model work
/// happens. Reuniting a target half from branch A with a dFlash half from
/// same-length branch B therefore fails before either can be stamped.
#[derive(Debug)]
#[allow(dead_code)] // first provenance slice; production migration follows
pub(crate) struct LivePairAdvance {
    epoch: PairBranchEpoch,
    base_revision: u64,
    target: LiveTargetHalf,
    dflash: LiveDFlashHalf,
    prefix_tokens: Vec<u32>,
    forwarded_tokens: Box<[u32]>,
}

#[allow(dead_code)] // first provenance slice; production migration follows
impl LivePair {
    pub(crate) fn cold(target: AnyCache, dflash: DFlashCache) -> Result<Self, PairedCacheError> {
        SealedPair::validate_boundaries(&target, dflash.position(), 0)?;
        let epoch = next_pair_branch_epoch();
        Ok(Self {
            epoch,
            revision: 0,
            target: LiveTargetHalf {
                epoch,
                cache: target,
            },
            dflash: LiveDFlashHalf {
                epoch,
                cache: dflash,
            },
            tokens: Vec::new(),
        })
    }

    /// Begin an aligned transition through one exact target-token slice.
    ///
    /// The exact tokens are captured before either cache becomes mutable. The
    /// returned transition is move-only; dropping it after a partial failure
    /// drops both halves and cannot mint a publication stamp.
    pub(crate) fn begin_aligned_advance(
        self,
        forwarded_tokens: &[u32],
    ) -> Result<LivePairAdvance, PairedCacheError> {
        let new_len = self
            .tokens
            .len()
            .checked_add(forwarded_tokens.len())
            .ok_or(PairedCacheError::PrefixLengthOverflow { len: usize::MAX })?;
        i32::try_from(new_len)
            .map_err(|_| PairedCacheError::PrefixLengthOverflow { len: new_len })?;
        Ok(LivePairAdvance {
            epoch: self.epoch,
            base_revision: self.revision,
            target: self.target,
            dflash: self.dflash,
            prefix_tokens: self.tokens,
            forwarded_tokens: forwarded_tokens.into(),
        })
    }

    /// Consume this exact live branch into one publishable retained pair.
    ///
    /// The sealing callback receives the only owned live drafter cache. The
    /// returned snapshot carries model-level source-branch provenance, so a
    /// callback cannot substitute a same-position snapshot from another branch.
    pub(crate) fn seal<E, F>(self, seal: F) -> Result<PairedCache, PairedCacheError>
    where
        E: std::fmt::Display,
        F: FnOnce(DFlashCache, i32) -> Result<DFlashSnapshot, E>,
    {
        let Self {
            epoch,
            revision: _,
            target,
            dflash,
            tokens,
        } = self;
        if target.epoch != epoch {
            return Err(PairedCacheError::ForeignTargetBranch);
        }
        if dflash.epoch != epoch {
            return Err(PairedCacheError::ForeignDFlashPairBranch);
        }
        let expected = i32::try_from(tokens.len())
            .map_err(|_| PairedCacheError::PrefixLengthOverflow { len: tokens.len() })?;
        SealedPair::validate_boundaries(&target.cache, dflash.cache.position(), expected)?;
        let source_branch = dflash.cache.branch_id();
        let snapshot =
            seal(dflash.cache, expected).map_err(|error| PairedCacheError::DFlashSeal {
                details: error.to_string(),
            })?;
        if snapshot.source_branch_id() != source_branch {
            return Err(PairedCacheError::ForeignDFlashBranch);
        }
        let sealed = SealedPair::from_live_branch(target.cache, snapshot, epoch, tokens)?;
        Ok(PairedCache { sealed })
    }
}

#[allow(dead_code)] // first provenance slice; production migration follows
impl LivePairAdvance {
    /// Test seam for exercising the staged provenance transition before the
    /// production prefill coordinator migrates into this module.
    ///
    /// Production deliberately receives no raw `&mut AnyCache` escape hatch:
    /// the migration must add target-forward methods that consume the ticket's
    /// own token slice.
    #[cfg(test)]
    fn target_cache_mut(&mut self) -> &mut AnyCache {
        &mut self.target.cache
    }

    /// Test seam paired with [`Self::target_cache_mut`].
    #[cfg(test)]
    fn dflash_cache_mut(&mut self) -> &mut DFlashCache {
        &mut self.dflash.cache
    }

    /// Reunite the exact two halves after successful model work.
    pub(crate) fn commit(mut self) -> Result<LivePair, PairedCacheError> {
        if self.target.epoch != self.epoch {
            return Err(PairedCacheError::ForeignTargetBranch);
        }
        if self.dflash.epoch != self.epoch {
            return Err(PairedCacheError::ForeignDFlashPairBranch);
        }
        let new_len = self
            .prefix_tokens
            .len()
            .checked_add(self.forwarded_tokens.len())
            .ok_or(PairedCacheError::PrefixLengthOverflow { len: usize::MAX })?;
        let expected = i32::try_from(new_len)
            .map_err(|_| PairedCacheError::PrefixLengthOverflow { len: new_len })?;
        SealedPair::validate_boundaries(
            &self.target.cache,
            self.dflash.cache.position(),
            expected,
        )?;
        let revision = self
            .base_revision
            .checked_add(1)
            .ok_or(PairedCacheError::BranchRevisionOverflow)?;
        self.prefix_tokens.extend_from_slice(&self.forwarded_tokens);
        Ok(LivePair {
            epoch: self.epoch,
            revision,
            target: self.target,
            dflash: self.dflash,
            tokens: self.prefix_tokens,
        })
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
    ///
    /// # Temporary compatibility layer
    ///
    /// This constructor accepts independently supplied halves and therefore
    /// proves boundary/key equality but not shared prefill provenance. New code
    /// must use [`LivePair::seal`]. Delete this after `simple.rs` and radix
    /// publication migrate to the move-owned coordinator.
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

    #[must_use]
    pub(crate) fn estimated_bytes(&self) -> (usize, usize) {
        self.sealed.estimated_bytes()
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

/// Immutable dFlash sidecar attached to one exact radix endpoint.
///
/// The target half remains represented by the radix endpoint's existing
/// paged/cloned storage. This type owns only the drafter snapshot and the
/// private metadata that binds it to that exact target endpoint.
#[derive(Debug)]
pub(crate) struct RadixDFlashSnapshot {
    // `DFlashSnapshot` is immutable after sealing but its MLX arrays are
    // `!Sync`. This mutex safely shares one frozen snapshot across owned lookup
    // plans and is never acquired while the radix mutex is held.
    dflash: Mutex<DFlashSnapshot>,
    metadata: Arc<PairMetadata>,
    #[cfg(test)]
    fail_next_fork: AtomicBool,
}

impl RadixDFlashSnapshot {
    pub(crate) fn new(
        dflash: DFlashSnapshot,
        tokens: &[u32],
        target_bytes: usize,
    ) -> Result<Self, PairedCacheError> {
        let stamp = PrefixStamp::new(tokens);
        let expected = stamp.boundary()?;
        let actual = dflash.position();
        if actual != expected {
            return Err(PairedCacheError::DFlashBoundary { expected, actual });
        }
        let metadata = Arc::new(PairMetadata {
            target_bytes,
            dflash_bytes: dflash.estimated_bytes(),
            stamp,
        });
        Ok(Self {
            dflash: Mutex::new(dflash),
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
    ) -> Result<RadixDFlashForkPlan, PairedCacheError> {
        if !self.matches_prefix(expected_tokens) {
            return Err(PairedCacheError::PrefixMismatch {
                stored_len: self.prefix_len(),
                requested_len: expected_tokens.len(),
            });
        }
        Ok(RadixDFlashForkPlan {
            snapshot: Arc::clone(self),
        })
    }

    fn fork_live(&self) -> Result<DFlashCache, PairedCacheError> {
        debug_assert!(
            higgs_models::mlx_exec::held(),
            "radix dFlash fork requires the process MLX execution gate"
        );
        #[cfg(test)]
        if self.fail_next_fork.swap(false, Ordering::SeqCst) {
            return Err(PairedCacheError::DFlashFork {
                details: "injected dFlash fork failure".to_owned(),
            });
        }
        let snapshot = self
            .dflash
            .lock()
            .map_err(|error| PairedCacheError::DFlashFork {
                details: format!("retained dFlash snapshot lock is poisoned: {error}"),
            })?;
        let live = snapshot
            .fork_live()
            .map_err(|error| PairedCacheError::DFlashFork {
                details: error.to_string(),
            })?;
        let expected = self.metadata.stamp.boundary()?;
        let actual = live.position();
        if actual != expected {
            return Err(PairedCacheError::DFlashBoundary { expected, actual });
        }
        Ok(live)
    }

    #[cfg(test)]
    fn fail_next_fork_for_test(&self) {
        self.fail_next_fork.store(true, Ordering::SeqCst);
    }
}

/// Owned, exact-identity plan for a post-prefix-lock dFlash fork.
#[derive(Debug)]
pub(crate) struct RadixDFlashForkPlan {
    snapshot: Arc<RadixDFlashSnapshot>,
}

impl RadixDFlashForkPlan {
    #[must_use]
    pub(crate) fn prefix_len(&self) -> usize {
        self.snapshot.prefix_len()
    }

    pub(crate) fn materialize(self) -> Result<DFlashCache, PairedCacheError> {
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

    /// Only target-only state may use the historical TurboQuant exemption from
    /// `max_session_tokens`. A paired dSpark snapshot remains uncompressed and
    /// grows with context, so exempting the target half would leave the whole
    /// retained pair unbounded.
    #[must_use]
    pub(crate) const fn allows_target_only_cap_exemption(&self) -> bool {
        matches!(self, Self::TargetOnly(_))
    }

    #[must_use]
    pub(crate) fn paired_estimated_bytes(&self) -> Option<(usize, usize)> {
        match self {
            Self::TargetOnly(_) => None,
            Self::Paired(pair) => Some(pair.estimated_bytes()),
        }
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
        cache::{KeyValueCache, SteppingKeyValueCache},
        dflash::{DFlashConfig, DFlashDrafter, DFlashSnapshot},
    };
    use mlx_rs::Array;

    use super::{LivePair, PairedCache, PairedCacheError, PrefixStamp, RetainedState};

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

    fn advance_target_one(cache: &mut AnyCache, token: u32) {
        let AnyCache::KV(layers) = cache else {
            panic!("test target must use a KV cache");
        };
        let layer = layers
            .first_mut()
            .and_then(Option::as_mut)
            .expect("test target layer");
        let value = token as f32;
        let keys = Array::from_slice(&[value], &[1, 1, 1, 1]);
        let values = Array::from_slice(&[-value], &[1, 1, 1, 1]);
        layer.update_and_fetch(keys, values).unwrap();
    }

    fn pending_live_pair_at_one(token: u32) -> (super::LivePairAdvance, DFlashDrafter) {
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
        let mut advance = LivePair::cold(target_cache(0), drafter.make_cache())
            .unwrap()
            .begin_aligned_advance(&[token])
            .unwrap();
        let _exec = higgs_models::mlx_exec::acquire();
        advance_target_one(advance.target_cache_mut(), token);
        advance.target_cache_mut().eval().unwrap();
        let taps = [Array::from_slice(
            &[token as f32, 1.0, 2.0, 3.0],
            &[1, 1, 4],
        )];
        drafter
            .prime_taps(&taps, advance.dflash_cache_mut())
            .unwrap();
        (advance, drafter)
    }

    fn live_pair_at_one(token: u32) -> (LivePair, DFlashDrafter) {
        let (advance, drafter) = pending_live_pair_at_one(token);
        (advance.commit().unwrap(), drafter)
    }

    #[test]
    fn live_pair_seal_uses_its_forwarded_prefix_without_a_publication_label() {
        let (pair, mut drafter) = live_pair_at_one(11);
        let _exec = higgs_models::mlx_exec::acquire();

        let sealed = pair
            .seal(|dflash, boundary| drafter.seal_after_taps(dflash, &[], boundary))
            .unwrap();

        assert!(sealed.matches_prefix(&[11]));
        assert!(!sealed.matches_prefix(&[12]));
        assert!(
            sealed.sealed.metadata.stamp.branch_epoch.is_some(),
            "the correct-by-construction path must retain its live branch epoch"
        );
    }

    #[test]
    fn live_pair_rejects_a_same_length_snapshot_from_another_branch() {
        let (left, mut left_drafter) = live_pair_at_one(11);
        let (right, mut right_drafter) = live_pair_at_one(12);
        let right_boundary = i32::try_from(right.tokens.len()).unwrap();
        let _exec = higgs_models::mlx_exec::acquire();
        let wrong_snapshot = right_drafter
            .seal_after_taps(right.dflash.cache, &[], right_boundary)
            .unwrap();

        let error = left
            .seal(|owned_dflash, _boundary| {
                drop(owned_dflash);
                Ok::<_, &'static str>(wrong_snapshot)
            })
            .unwrap_err();

        assert_eq!(error, PairedCacheError::ForeignDFlashBranch);

        // Keep the intended drafter alive through the adversarial seal attempt:
        // the rejection is provenance-based, not caused by dropping model state.
        let _ = &mut left_drafter;
    }

    #[test]
    fn live_pair_rejects_a_same_length_target_half_from_another_branch() {
        let (mut left, _left_drafter) = pending_live_pair_at_one(11);
        let (mut right, _right_drafter) = pending_live_pair_at_one(12);
        std::mem::swap(&mut left.target, &mut right.target);

        let error = left.commit().unwrap_err();

        assert_eq!(error, PairedCacheError::ForeignTargetBranch);
    }

    #[test]
    fn live_pair_rejects_a_same_length_dflash_half_from_another_branch() {
        let (mut left, _left_drafter) = pending_live_pair_at_one(11);
        let (mut right, _right_drafter) = pending_live_pair_at_one(12);
        std::mem::swap(&mut left.dflash, &mut right.dflash);

        let error = left.commit().unwrap_err();

        assert_eq!(error, PairedCacheError::ForeignDFlashPairBranch);
    }

    #[test]
    fn paired_cache_accepts_one_exact_shared_boundary() {
        let pair = PairedCache::new(target_cache(0), dflash_snapshot(0), &[]).unwrap();

        assert_eq!(pair.prefix_len(), 0);
        assert!(pair.matches_prefix(&[]));
        assert!(
            pair.sealed.metadata.stamp.branch_epoch.is_none(),
            "legacy compatibility construction must remain visibly unproven"
        );
    }

    #[test]
    fn prefix_length_must_fit_the_model_boundary_type() {
        let len = usize::try_from(i32::MAX).unwrap() + 1;
        let stamp = PrefixStamp {
            branch_epoch: None,
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
