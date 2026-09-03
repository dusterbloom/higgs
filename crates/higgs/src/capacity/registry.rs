use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use higgs_engine::{EngineCostDescription, MlxMemorySnapshot, TransientPrefillEstimate};
use sha2::{Digest, Sha256};

use super::{
    CAPACITY_SCHEMA_VERSION, CapacityAvailability, CapacityBasis, CapacityController,
    CapacityInputs, CapacitySnapshot, LearnedProfile, LearnedProfileKey, LearnedProfileStore,
    MemoryPressure, PressureObservation,
};

const FINGERPRINT_DOMAIN: &[u8] = b"higgs:model-content:v1\0";

/// Exact, content-addressed identity and byte count for one model artifact tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelContentIdentity {
    pub fingerprint: String,
    pub artifact_bytes: u64,
}

/// Cache classes an engine can actually enforce. A zero ceiling remains
/// automatic for supported classes; unsupported classes receive no allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheCapabilities {
    pub retained_sessions: bool,
    pub prefix_cache: bool,
}

impl CacheCapabilities {
    pub const SIMPLE: Self = Self {
        retained_sessions: true,
        prefix_cache: true,
    };
    pub const BATCH: Self = Self {
        retained_sessions: false,
        prefix_cache: true,
    };
}

/// Immutable inputs captured after one model has loaded successfully.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelCapacityFacts {
    pub model: String,
    pub model_fingerprint: String,
    pub memory: MlxMemorySnapshot,
    pub costs: EngineCostDescription,
    pub loaded_model_bytes: u64,
    pub architectural_max_tokens: u64,
    pub prefill_chunk_tokens: u64,
    pub retained_session_tokens: u64,
    pub retained_resident_bytes: u64,
    pub prefix_cache_resident_bytes: u64,
    pub retained_bytes_ceiling: u64,
    pub prefix_cache_bytes_ceiling: u64,
    pub cache_capabilities: CacheCapabilities,
    pub configured_total_token_ceiling: Option<u64>,
    pub configured_output_token_ceiling: Option<u64>,
    pub quantization: String,
    pub execution_mode: String,
    pub kv_representation: String,
    pub prefill_model_identity: Option<String>,
    pub drafter_identity: Option<String>,
    pub learned_profile_key: Option<LearnedProfileKey>,
    pub startup_headroom_bytes: u64,
}

impl ModelCapacityFacts {
    fn controller(&self, shared: SharedLedger, pressure: MemoryPressure) -> CapacityController {
        CapacityController::new(CapacityInputs {
            memory: shared.memory,
            costs: self.costs,
            loaded_model_bytes: shared.loaded_model_bytes,
            architectural_max_tokens: self.architectural_max_tokens,
            prefill_chunk_tokens: self.prefill_chunk_tokens,
            retained_bytes: shared.retained_bytes,
            prefix_cache_bytes: shared.prefix_cache_bytes,
            active_reservation_bytes: shared.active_reservation_bytes,
            configured_total_token_ceiling: self.configured_total_token_ceiling,
            configured_output_token_ceiling: self.configured_output_token_ceiling,
            pressure,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RegistrationError {
    #[error("unknown model '{0}'")]
    UnknownModel(String),
    #[error("model '{0}' is already loaded or loading")]
    AlreadyRegistered(String),
    #[error("model '{0}' is not active")]
    NotActive(String),
    #[error("capacity arithmetic overflowed")]
    ArithmeticOverflow,
    #[error("model '{0}' cannot fit the minimum working request")]
    InsufficientCapacity(String),
}

#[derive(Clone, Copy, Debug, Default)]
struct SharedLedger {
    memory: MlxMemorySnapshot,
    loaded_model_bytes: u64,
    retained_bytes: u64,
    prefix_cache_bytes: u64,
    active_reservation_bytes: u64,
}

#[derive(Debug)]
struct ActiveModel {
    facts: ModelCapacityFacts,
    controller: CapacityController,
    basis: CapacityBasis,
    lifecycle_nonce: uuid::Uuid,
    drain_nonce: Option<uuid::Uuid>,
    draining: bool,
    frozen_cache_allocation: Option<CacheAllocation>,
    published: bool,
}

#[derive(Debug)]
struct ModelEntry {
    generation: u64,
    last_fingerprint: String,
    active: Option<ActiveModel>,
}

impl Default for ModelEntry {
    fn default() -> Self {
        Self {
            generation: 0,
            last_fingerprint: String::new(),
            active: None,
        }
    }
}

#[derive(Debug)]
struct RegistryState {
    models: BTreeMap<String, ModelEntry>,
    registering: BTreeSet<String>,
    pressure_controller: CapacityController,
    pressure: MemoryPressure,
    memory: MlxMemorySnapshot,
    memory_revision: u64,
    desired_cache_allocations: BTreeMap<String, CacheAllocation>,
    published_cache_allocations: BTreeMap<String, CacheAllocation>,
    cache_revision: u64,
    published_cache_revision: u64,
    cache_plan_pressure: MemoryPressure,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CacheAllocation {
    retained_bytes: u64,
    prefix_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheAllocationPlan {
    pub revision: u64,
    pub pressure: MemoryPressure,
    pub allocations: Vec<(String, u64, u64)>,
}

/// Proof that an allocator snapshot was published to this registry in the
/// same serialized GPU/load window in which it was measured.
pub struct PublishedMemoryMeasurement {
    boot_id: String,
    revision: u64,
}

/// The single process-wide authority for model capacity and shared residency.
#[derive(Debug)]
pub struct CapacityRegistry {
    boot_id: String,
    profile_dir: Option<PathBuf>,
    state: Mutex<RegistryState>,
}

impl CapacityRegistry {
    #[must_use]
    pub fn new(known_models: impl IntoIterator<Item = String>) -> Arc<Self> {
        Self::new_inner(known_models, None)
    }

    #[must_use]
    pub fn new_with_profile_dir(
        known_models: impl IntoIterator<Item = String>,
        profile_dir: PathBuf,
    ) -> Arc<Self> {
        Self::new_inner(known_models, Some(profile_dir))
    }

    fn new_inner(
        known_models: impl IntoIterator<Item = String>,
        profile_dir: Option<PathBuf>,
    ) -> Arc<Self> {
        Arc::new(Self {
            boot_id: uuid::Uuid::new_v4().to_string(),
            profile_dir,
            state: Mutex::new(RegistryState {
                models: known_models
                    .into_iter()
                    .map(|name| (name, ModelEntry::default()))
                    .collect(),
                registering: BTreeSet::new(),
                pressure_controller: registry_pressure_controller(),
                pressure: MemoryPressure::Normal,
                memory: MlxMemorySnapshot::default(),
                memory_revision: 0,
                desired_cache_allocations: BTreeMap::new(),
                published_cache_allocations: BTreeMap::new(),
                cache_revision: 0,
                published_cache_revision: 0,
                cache_plan_pressure: MemoryPressure::Normal,
            }),
        })
    }

    #[must_use]
    pub fn boot_id(&self) -> String {
        self.boot_id.clone()
    }

    pub fn snapshot(&self, model: &str) -> Result<CapacitySnapshot, RegistrationError> {
        let state = self.lock();
        let entry = state
            .models
            .get(model)
            .ok_or_else(|| RegistrationError::UnknownModel(model.to_owned()))?;
        Ok(snapshot_for(
            &self.boot_id,
            model,
            entry,
            state.pressure,
            state
                .published_cache_allocations
                .get(model)
                .copied()
                .unwrap_or_default(),
        ))
    }

    /// Atomically snapshot the effective per-model cache allocations so the
    /// router can apply one coherent process-wide policy revision.
    pub fn cache_allocations(&self) -> Vec<(String, u64, u64)> {
        let state = self.lock();
        state
            .desired_cache_allocations
            .iter()
            .map(|(model, allocation)| {
                (
                    model.clone(),
                    allocation.retained_bytes,
                    allocation.prefix_bytes,
                )
            })
            .collect()
    }

    #[must_use]
    pub fn cache_allocation_plan(&self) -> CacheAllocationPlan {
        let state = self.lock();
        CacheAllocationPlan {
            revision: state.cache_revision,
            pressure: state.pressure,
            allocations: state
                .desired_cache_allocations
                .iter()
                .map(|(model, allocation)| {
                    (
                        model.clone(),
                        allocation.retained_bytes,
                        allocation.prefix_bytes,
                    )
                })
                .collect(),
        }
    }

    /// Publish only the exact policy revision engines acknowledged. A newer
    /// pressure/load recomputation makes the caller retry the new plan.
    pub fn publish_cache_allocation_revision(&self, revision: u64) -> bool {
        let mut state = self.lock();
        if state.cache_revision != revision {
            return false;
        }
        let desired = state.desired_cache_allocations.clone();
        let old = std::mem::replace(&mut state.published_cache_allocations, desired);
        let published = state.published_cache_allocations.clone();
        state.published_cache_revision = revision;
        for (name, entry) in &mut state.models {
            if old.get(name) != published.get(name)
                && entry
                    .active
                    .as_ref()
                    .is_some_and(|active| active.published && !active.draining)
            {
                entry.generation = entry.generation.saturating_add(1);
            }
        }
        let mut shared = shared_ledger_with(&state, None, None).unwrap_or_default();
        let effective = conservative_cache_allocations(
            &state.desired_cache_allocations,
            &state.published_cache_allocations,
        );
        if apply_allocation_totals(&mut shared, &effective).is_err() {
            shared.active_reservation_bytes = u64::MAX;
        }
        recompute_active_models(&mut state, shared);
        true
    }

    /// Run the non-blocking route visibility flip while the acknowledged cache
    /// revision is still current. The closure must only mutate the router map;
    /// it must never await or call back into the registry.
    pub fn publish_route_if_current(&self, revision: u64, publish: impl FnOnce()) -> bool {
        let state = self.lock();
        if state.cache_revision != revision || state.published_cache_revision != revision {
            return false;
        }
        publish();
        true
    }

    pub fn begin_registration(
        self: &Arc<Self>,
        model: String,
    ) -> Result<RegistrationTicket, RegistrationError> {
        let mut state = self.lock();
        if state
            .models
            .get(&model)
            .is_some_and(|entry| entry.active.is_some())
            || state.registering.contains(&model)
        {
            return Err(RegistrationError::AlreadyRegistered(model));
        }
        let newly_known = !state.models.contains_key(&model);
        state.models.entry(model.clone()).or_default();
        state.registering.insert(model.clone());
        Ok(RegistrationTicket {
            registry: Arc::downgrade(self),
            model,
            nonce: uuid::Uuid::new_v4(),
            pending: true,
            newly_known,
        })
    }

    pub fn commit_active(
        self: &Arc<Self>,
        mut ticket: RegistrationTicket,
        facts: ModelCapacityFacts,
    ) -> Result<ActiveRegistration, RegistrationError> {
        if ticket.model != facts.model || !ticket.belongs_to(self) {
            return Err(RegistrationError::UnknownModel(facts.model));
        }
        // Disk I/O stays outside the registry lock. A corrupt, incomplete, or
        // mismatched file returns no profile and leaves the cold solver intact.
        let restored_profile = self.load_profile(&facts);
        let mut state = self.lock();
        if !state.registering.contains(&ticket.model) {
            return Err(RegistrationError::AlreadyRegistered(ticket.model.clone()));
        }

        let candidate_allocations = cache_allocations_with(&state, Some(&facts))?;
        let candidate_allocations = conservative_cache_allocations(
            &candidate_allocations,
            &state.published_cache_allocations,
        );
        let mut shared = shared_ledger_with(&state, Some(&facts), None)?;
        apply_allocation_totals(&mut shared, &candidate_allocations)?;
        let mut candidate = facts.controller(shared, state.pressure);
        let basis = restored_profile
            .as_ref()
            .map_or(CapacityBasis::Conservative, |profile| {
                let restored = facts.learned_profile_key.as_ref().is_some_and(|key| {
                    candidate.restore_profile(profile, key, facts.startup_headroom_bytes)
                });
                if restored
                    && profile
                        .evidence()
                        .iter()
                        .any(|band| band.cold_replacement_qualified)
                {
                    CapacityBasis::Learned
                } else {
                    CapacityBasis::Conservative
                }
            });
        if candidate.decision().availability != CapacityAvailability::Available {
            return Err(RegistrationError::InsufficientCapacity(facts.model));
        }

        let mut existing = Vec::new();
        for (name, entry) in &state.models {
            if let Some(active) = &entry.active {
                let mut controller = active.controller.transactional_copy();
                replace_shared_ledger(&mut controller, shared);
                if !active.draining
                    && controller.decision().availability != CapacityAvailability::Available
                {
                    return Err(RegistrationError::InsufficientCapacity(name.clone()));
                }
                existing.push((name.clone(), controller));
            }
        }

        for (name, controller) in existing {
            if let Some(entry) = state.models.get_mut(&name) {
                let changed = entry
                    .active
                    .as_ref()
                    .is_some_and(|active| active.controller.decision() != controller.decision());
                if let Some(active) = entry.active.as_mut() {
                    active.controller = controller;
                }
                if changed {
                    entry.generation = entry.generation.saturating_add(1);
                }
            }
        }
        // Re-apply the shared memory snapshot after the checked candidate build.
        replace_shared_ledger(&mut candidate, shared);
        let lifecycle_nonce = ticket.nonce;
        {
            let entry = state.models.entry(ticket.model.clone()).or_default();
            entry.generation = entry.generation.saturating_add(1);
            entry.last_fingerprint.clone_from(&facts.model_fingerprint);
            entry.active = Some(ActiveModel {
                facts,
                controller: candidate,
                basis,
                lifecycle_nonce,
                drain_nonce: None,
                draining: false,
                frozen_cache_allocation: None,
                published: false,
            });
        }
        state.registering.remove(&ticket.model);
        ticket.pending = false;
        recompute_registry(&mut state);

        Ok(ActiveRegistration {
            registry: Arc::downgrade(self),
            model: ticket.model.clone(),
            nonce: lifecycle_nonce,
            published: false,
            remove_on_rollback: ticket.newly_known,
        })
    }

    pub fn begin_drain(
        self: &Arc<Self>,
        model: &str,
    ) -> Result<DrainRegistration, RegistrationError> {
        let mut state = self.lock();
        let frozen_cache_allocation = state
            .desired_cache_allocations
            .get(model)
            .copied()
            .unwrap_or_default();
        let frozen_cache_allocation = max_cache_allocation(
            frozen_cache_allocation,
            state
                .published_cache_allocations
                .get(model)
                .copied()
                .unwrap_or_default(),
        );
        let entry = state
            .models
            .get_mut(model)
            .ok_or_else(|| RegistrationError::UnknownModel(model.to_owned()))?;
        let active = entry
            .active
            .as_mut()
            .ok_or_else(|| RegistrationError::NotActive(model.to_owned()))?;
        if active.draining {
            return Err(RegistrationError::NotActive(model.to_owned()));
        }
        let nonce = uuid::Uuid::new_v4();
        active.draining = true;
        active.drain_nonce = Some(nonce);
        active.frozen_cache_allocation = Some(frozen_cache_allocation);
        entry.generation = entry.generation.saturating_add(1);
        recompute_registry(&mut state);
        Ok(DrainRegistration {
            registry: Arc::downgrade(self),
            model: model.to_owned(),
            nonce,
            finished: false,
        })
    }

    pub fn finish_unregister(
        &self,
        mut drain: DrainRegistration,
        memory_after_release: Option<PublishedMemoryMeasurement>,
    ) -> io::Result<()> {
        if !drain.belongs_to(self) {
            return Ok(());
        }
        let mut state = self.lock();
        if let Some(measurement) = memory_after_release.as_ref()
            && (measurement.boot_id != self.boot_id || measurement.revision > state.memory_revision)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unload memory measurement does not belong to this registry revision",
            ));
        }
        let mut persisted = None;
        let mut removed = false;
        if let Some(entry) = state.models.get_mut(&drain.model)
            && entry
                .active
                .as_ref()
                .is_some_and(|active| active.draining && active.drain_nonce == Some(drain.nonce))
        {
            persisted = entry.active.as_ref().and_then(profile_record);
            entry.active = None;
            entry.generation = entry.generation.saturating_add(1);
            removed = true;
        }
        if removed {
            state.published_cache_allocations.remove(&drain.model);
            recompute_registry(&mut state);
        }
        drain.finished = true;
        drop(state);
        self.save_profile(persisted)
    }

    /// Persist all qualified learned evidence without serializing live capacity,
    /// pressure, reservations, generations, or boot identity.
    pub fn persist_profiles(&self) -> io::Result<()> {
        let profiles = {
            let state = self.lock();
            state
                .models
                .values()
                .filter_map(|entry| entry.active.as_ref())
                .filter_map(profile_record)
                .collect::<Vec<_>>()
        };
        for profile in profiles {
            self.save_profile(Some(profile))?;
        }
        Ok(())
    }

    pub fn apply_pressure_observation(&self, observation: PressureObservation) {
        let mut state = self.lock();
        state
            .pressure_controller
            .apply_pressure_observation(observation);
        let effective_pressure = state.pressure_controller.pressure();
        let normalized = PressureObservation {
            pressure: effective_pressure,
            ..observation
        };
        for entry in state.models.values_mut() {
            if let Some(active) = entry.active.as_mut() {
                let before = active.controller.decision();
                active.controller.apply_pressure_observation(normalized);
                if active.controller.decision() != before {
                    entry.generation = entry.generation.saturating_add(1);
                }
            }
        }
        state.pressure = effective_pressure;
        recompute_registry(&mut state);
    }

    pub fn refresh_memory(&self, memory: MlxMemorySnapshot) -> PublishedMemoryMeasurement {
        let mut state = self.lock();
        state.memory = memory;
        state.memory_revision = state.memory_revision.saturating_add(1);
        let revision = state.memory_revision;
        recompute_registry(&mut state);
        PublishedMemoryMeasurement {
            boot_id: self.boot_id.clone(),
            revision,
        }
    }

    fn rollback_active(&self, model: &str, nonce: uuid::Uuid, remove_on_rollback: bool) {
        let mut state = self.lock();
        let remove = state.models.get(model).is_some_and(|entry| {
            entry
                .active
                .as_ref()
                .is_some_and(|active| active.lifecycle_nonce == nonce)
        });
        if remove {
            if remove_on_rollback {
                state.models.remove(model);
            } else if let Some(entry) = state.models.get_mut(model) {
                entry.active = None;
                entry.generation = entry.generation.saturating_add(1);
            }
            recompute_registry(&mut state);
        }
    }

    fn publish_active(&self, model: &str, nonce: uuid::Uuid) -> bool {
        let mut state = self.lock();
        let Some(entry) = state.models.get_mut(model) else {
            return false;
        };
        let Some(active) = entry.active.as_mut() else {
            return false;
        };
        if active.lifecycle_nonce != nonce || active.draining {
            return false;
        }
        active.published = true;
        entry.generation = entry.generation.saturating_add(1);
        true
    }

    fn cancel_registration(&self, model: &str, newly_known: bool) {
        let mut state = self.lock();
        state.registering.remove(model);
        if newly_known
            && state
                .models
                .get(model)
                .is_some_and(|entry| entry.active.is_none())
        {
            state.models.remove(model);
        }
    }

    fn cancel_drain(&self, model: &str, nonce: uuid::Uuid) {
        let mut state = self.lock();
        if let Some(entry) = state.models.get_mut(model)
            && let Some(active) = entry.active.as_mut()
            && active.draining
            && active.drain_nonce == Some(nonce)
        {
            active.draining = false;
            active.drain_nonce = None;
            active.frozen_cache_allocation = None;
            entry.generation = entry.generation.saturating_add(1);
            recompute_registry(&mut state);
        }
    }

    fn lock(&self) -> MutexGuard<'_, RegistryState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn load_profile(&self, facts: &ModelCapacityFacts) -> Option<LearnedProfile> {
        let key = facts.learned_profile_key.as_ref()?;
        let store = self.profile_store(&facts.model)?;
        match store.load(key, facts.startup_headroom_bytes) {
            Ok(profile) => profile,
            Err(error) => {
                tracing::warn!(model = %facts.model, %error, "failed to load learned capacity profile");
                None
            }
        }
    }

    fn save_profile(&self, record: Option<(String, LearnedProfile)>) -> io::Result<()> {
        let Some((model, profile)) = record else {
            return Ok(());
        };
        let Some(store) = self.profile_store(&model) else {
            return Ok(());
        };
        store.save(&profile)
    }

    fn profile_store(&self, model: &str) -> Option<LearnedProfileStore> {
        let directory = self.profile_dir.as_ref()?;
        let mut hash = Sha256::new();
        hash.update(b"higgs:capacity-profile:model:v1\0");
        hash.update(model.as_bytes());
        let mut name = String::with_capacity(69);
        for byte in hash.finalize() {
            use std::fmt::Write as _;
            write!(&mut name, "{byte:02x}").expect("writing to String cannot fail");
        }
        name.push_str(".json");
        Some(LearnedProfileStore::new(directory.join(name)))
    }
}

fn profile_record(active: &ActiveModel) -> Option<(String, LearnedProfile)> {
    let key = active.facts.learned_profile_key.clone()?;
    active
        .controller
        .export_profile(key, active.facts.startup_headroom_bytes)
        .map(|profile| (active.facts.model.clone(), profile))
}

fn registry_pressure_controller() -> CapacityController {
    CapacityController::new(CapacityInputs {
        memory: MlxMemorySnapshot::default(),
        costs: EngineCostDescription {
            fixed_live_session_bytes: 0,
            persistent_bytes_per_token: 0,
            decode_workspace_bytes: 0,
            transient_prefill: TransientPrefillEstimate {
                base_bytes: 0,
                bytes_per_prompt_token: 0,
                bytes_per_chunk_token: 0,
                max_prompt_tokens: 0,
                max_chunk_tokens: 0,
            },
        },
        loaded_model_bytes: 0,
        architectural_max_tokens: 0,
        prefill_chunk_tokens: 0,
        retained_bytes: 0,
        prefix_cache_bytes: 0,
        active_reservation_bytes: 0,
        configured_total_token_ceiling: Some(0),
        configured_output_token_ceiling: Some(0),
        pressure: MemoryPressure::Normal,
    })
}

#[must_use = "dropping an unpublished registration rolls back active capacity"]
pub struct ActiveRegistration {
    registry: Weak<CapacityRegistry>,
    model: String,
    nonce: uuid::Uuid,
    published: bool,
    remove_on_rollback: bool,
}

impl ActiveRegistration {
    pub fn publish(mut self) {
        self.published = self
            .registry
            .upgrade()
            .is_some_and(|registry| registry.publish_active(&self.model, self.nonce));
    }
}

impl Drop for ActiveRegistration {
    fn drop(&mut self) {
        if !self.published
            && let Some(registry) = self.registry.upgrade()
        {
            registry.rollback_active(&self.model, self.nonce, self.remove_on_rollback);
        }
    }
}

#[must_use = "a registration ticket must be committed or rolled back"]
pub struct RegistrationTicket {
    registry: Weak<CapacityRegistry>,
    model: String,
    nonce: uuid::Uuid,
    pending: bool,
    newly_known: bool,
}

impl RegistrationTicket {
    fn belongs_to(&self, registry: &Arc<CapacityRegistry>) -> bool {
        self.registry
            .upgrade()
            .is_some_and(|owner| Arc::ptr_eq(&owner, registry))
    }
}

impl Drop for RegistrationTicket {
    fn drop(&mut self) {
        if self.pending
            && let Some(registry) = self.registry.upgrade()
        {
            registry.cancel_registration(&self.model, self.newly_known);
        }
    }
}

#[must_use = "a draining model must be finished after workers release it"]
pub struct DrainRegistration {
    registry: Weak<CapacityRegistry>,
    model: String,
    nonce: uuid::Uuid,
    finished: bool,
}

impl DrainRegistration {
    fn belongs_to(&self, registry: &CapacityRegistry) -> bool {
        self.registry
            .upgrade()
            .is_some_and(|owner| std::ptr::eq(Arc::as_ptr(&owner), registry))
    }
}

impl Drop for DrainRegistration {
    fn drop(&mut self) {
        if !self.finished
            && let Some(registry) = self.registry.upgrade()
        {
            registry.cancel_drain(&self.model, self.nonce);
        }
    }
}

fn shared_ledger_with(
    state: &RegistryState,
    added: Option<&ModelCapacityFacts>,
    removed: Option<&str>,
) -> Result<SharedLedger, RegistrationError> {
    let mut shared = SharedLedger {
        memory: state.memory,
        ..SharedLedger::default()
    };
    for (name, entry) in &state.models {
        if removed == Some(name.as_str()) {
            continue;
        }
        if let Some(active) = &entry.active {
            shared.loaded_model_bytes = shared
                .loaded_model_bytes
                .checked_add(active.facts.loaded_model_bytes)
                .ok_or(RegistrationError::ArithmeticOverflow)?;
            if active.facts.cache_capabilities.retained_sessions {
                shared.retained_bytes = shared
                    .retained_bytes
                    .checked_add(active.facts.retained_resident_bytes)
                    .ok_or(RegistrationError::ArithmeticOverflow)?;
            }
            if active.facts.cache_capabilities.prefix_cache {
                shared.prefix_cache_bytes = shared
                    .prefix_cache_bytes
                    .checked_add(active.facts.prefix_cache_resident_bytes)
                    .ok_or(RegistrationError::ArithmeticOverflow)?;
            }
        }
    }
    if let Some(facts) = added {
        shared.loaded_model_bytes = shared
            .loaded_model_bytes
            .checked_add(facts.loaded_model_bytes)
            .ok_or(RegistrationError::ArithmeticOverflow)?;
        if facts.cache_capabilities.retained_sessions {
            shared.retained_bytes = shared
                .retained_bytes
                .checked_add(facts.retained_resident_bytes)
                .ok_or(RegistrationError::ArithmeticOverflow)?;
        }
        if facts.cache_capabilities.prefix_cache {
            shared.prefix_cache_bytes = shared
                .prefix_cache_bytes
                .checked_add(facts.prefix_cache_resident_bytes)
                .ok_or(RegistrationError::ArithmeticOverflow)?;
        }
    }
    Ok(shared)
}

fn replace_shared_ledger(controller: &mut CapacityController, shared: SharedLedger) {
    controller.replace_shared_residency(
        shared.memory,
        shared.loaded_model_bytes,
        shared.retained_bytes,
        shared.prefix_cache_bytes,
        shared.active_reservation_bytes,
    );
}

fn recompute_active_models(state: &mut RegistryState, shared: SharedLedger) {
    for entry in state.models.values_mut() {
        if let Some(active) = entry.active.as_mut() {
            let before = active.controller.decision();
            replace_shared_ledger(&mut active.controller, shared);
            if active.controller.decision() != before {
                entry.generation = entry.generation.saturating_add(1);
            }
        }
    }
}

fn recompute_registry(state: &mut RegistryState) {
    let old_allocations = std::mem::take(&mut state.desired_cache_allocations);
    let allocations = cache_allocations_with(state, None);
    state.desired_cache_allocations = allocations.as_ref().cloned().unwrap_or_default();
    if old_allocations != state.desired_cache_allocations
        || state.cache_plan_pressure != state.pressure
    {
        state.cache_revision = state.cache_revision.saturating_add(1);
        state.cache_plan_pressure = state.pressure;
    }
    let mut shared = shared_ledger_with(state, None, None).unwrap_or_default();
    let effective = conservative_cache_allocations(
        &state.desired_cache_allocations,
        &state.published_cache_allocations,
    );
    if allocations.is_err() || apply_allocation_totals(&mut shared, &effective).is_err() {
        shared.active_reservation_bytes = u64::MAX;
    }
    recompute_active_models(state, shared);
}

fn max_cache_allocation(left: CacheAllocation, right: CacheAllocation) -> CacheAllocation {
    CacheAllocation {
        retained_bytes: left.retained_bytes.max(right.retained_bytes),
        prefix_bytes: left.prefix_bytes.max(right.prefix_bytes),
    }
}

fn conservative_cache_allocations(
    desired: &BTreeMap<String, CacheAllocation>,
    published: &BTreeMap<String, CacheAllocation>,
) -> BTreeMap<String, CacheAllocation> {
    let mut effective = published.clone();
    for (name, allocation) in desired {
        effective
            .entry(name.clone())
            .and_modify(|current| *current = max_cache_allocation(*current, *allocation))
            .or_insert(*allocation);
    }
    effective
}

fn cache_allocations_with(
    state: &RegistryState,
    added: Option<&ModelCapacityFacts>,
) -> Result<BTreeMap<String, CacheAllocation>, RegistrationError> {
    let mut requested_retained = BTreeMap::new();
    let mut requested_prefix = BTreeMap::new();
    let mut frozen = BTreeMap::new();
    let mut resident_cache = 0_u64;
    for (name, entry) in &state.models {
        if let Some(active) = entry.active.as_ref() {
            if let Some(allocation) = active.frozen_cache_allocation {
                frozen.insert(name.clone(), allocation);
            } else {
                if active.facts.cache_capabilities.retained_sessions {
                    requested_retained.insert(name.clone(), active.facts.retained_bytes_ceiling);
                }
                if active.facts.cache_capabilities.prefix_cache {
                    requested_prefix.insert(name.clone(), active.facts.prefix_cache_bytes_ceiling);
                }
            }
            if active.facts.cache_capabilities.retained_sessions {
                resident_cache = resident_cache
                    .checked_add(active.facts.retained_resident_bytes)
                    .ok_or(RegistrationError::ArithmeticOverflow)?;
            }
            if active.facts.cache_capabilities.prefix_cache {
                resident_cache = resident_cache
                    .checked_add(active.facts.prefix_cache_resident_bytes)
                    .ok_or(RegistrationError::ArithmeticOverflow)?;
            }
        }
    }
    if let Some(facts) = added {
        if facts.cache_capabilities.retained_sessions {
            requested_retained.insert(facts.model.clone(), facts.retained_bytes_ceiling);
        }
        if facts.cache_capabilities.prefix_cache {
            requested_prefix.insert(facts.model.clone(), facts.prefix_cache_bytes_ceiling);
        }
        if facts.cache_capabilities.retained_sessions {
            resident_cache = resident_cache
                .checked_add(facts.retained_resident_bytes)
                .ok_or(RegistrationError::ArithmeticOverflow)?;
        }
        if facts.cache_capabilities.prefix_cache {
            resident_cache = resident_cache
                .checked_add(facts.prefix_cache_resident_bytes)
                .ok_or(RegistrationError::ArithmeticOverflow)?;
        }
    }
    if requested_retained.is_empty() && requested_prefix.is_empty() && frozen.is_empty() {
        return Ok(BTreeMap::new());
    }
    let memory = added.map_or(state.memory, |facts| facts.memory);
    let authority = [
        memory.memory_limit_bytes,
        memory.metal_recommended_working_set_bytes,
    ]
    .into_iter()
    .flatten()
    .filter(|bytes| *bytes > 0)
    .min()
    .ok_or(RegistrationError::InsufficientCapacity(
        added.map_or_else(|| "process".to_owned(), |facts| facts.model.clone()),
    ))?;
    let automatic_ceiling = (authority / 16).max(1);
    for requested in requested_retained
        .values_mut()
        .chain(requested_prefix.values_mut())
    {
        if *requested == 0 {
            *requested = automatic_ceiling;
        }
    }
    if state.pressure == MemoryPressure::Critical {
        let mut allocations = frozen;
        for name in requested_retained.keys().chain(requested_prefix.keys()) {
            allocations.entry(name.clone()).or_default();
        }
        return Ok(allocations);
    }
    let percentage = match state.pressure {
        MemoryPressure::Normal => 20,
        MemoryPressure::Constrained | MemoryPressure::Critical => 30,
    };
    let reserve = authority
        .checked_mul(percentage)
        .map(|bytes| bytes / 100)
        .unwrap_or(u64::MAX)
        .max(4 * 1024 * 1024 * 1024);
    let cache_envelope = authority
        .saturating_sub(reserve)
        .saturating_sub(memory.active_bytes)
        .checked_add(resident_cache)
        .ok_or(RegistrationError::ArithmeticOverflow)?;
    let frozen_total = frozen.values().try_fold(0_u64, |sum, allocation| {
        sum.checked_add(allocation.retained_bytes)
            .and_then(|sum| sum.checked_add(allocation.prefix_bytes))
            .ok_or(RegistrationError::ArithmeticOverflow)
    })?;
    if added.is_some() && frozen_total > cache_envelope {
        return Err(RegistrationError::InsufficientCapacity(
            added.map_or_else(|| "process".to_owned(), |facts| facts.model.clone()),
        ));
    }
    let available_envelope = cache_envelope.saturating_sub(frozen_total);
    let mut retained = fair_cache_allocations(&requested_retained, available_envelope / 2);
    let mut prefix = fair_cache_allocations(
        &requested_prefix,
        available_envelope.saturating_sub(available_envelope / 2),
    );
    let used = allocation_total(&retained)?
        .checked_add(allocation_total(&prefix)?)
        .ok_or(RegistrationError::ArithmeticOverflow)?;
    let mut remaining = available_envelope.saturating_sub(used);
    remaining = extend_fair_allocations(&requested_retained, &mut retained, remaining)?;
    let _unused = extend_fair_allocations(&requested_prefix, &mut prefix, remaining)?;
    let mut allocations = frozen;
    for name in requested_retained.keys().chain(requested_prefix.keys()) {
        allocations.entry(name.clone()).or_insert(CacheAllocation {
            retained_bytes: retained.get(name).copied().unwrap_or(0),
            prefix_bytes: prefix.get(name).copied().unwrap_or(0),
        });
    }
    Ok(allocations)
}

fn fair_cache_allocations(
    requests: &BTreeMap<String, u64>,
    mut budget: u64,
) -> BTreeMap<String, u64> {
    let mut remaining = requests
        .iter()
        .filter(|(_, request)| **request > 0)
        .map(|(name, request)| (name.clone(), *request))
        .collect::<BTreeMap<_, _>>();
    let mut allocations: BTreeMap<String, u64> = BTreeMap::new();
    while budget > 0 && !remaining.is_empty() {
        let count = u64::try_from(remaining.len()).unwrap_or(u64::MAX);
        let share = budget.div_ceil(count).max(1);
        let names = remaining.keys().cloned().collect::<Vec<_>>();
        let mut progressed = false;
        for name in names {
            let request = remaining.get(&name).copied().unwrap_or(0);
            let grant = request.min(share).min(budget);
            allocations
                .entry(name.clone())
                .and_modify(|bytes| *bytes = bytes.saturating_add(grant))
                .or_insert(grant);
            budget = budget.saturating_sub(grant);
            progressed |= grant > 0;
            if grant == request {
                remaining.remove(&name);
            } else if let Some(request) = remaining.get_mut(&name) {
                *request -= grant;
            }
            if budget == 0 {
                break;
            }
        }
        if !progressed {
            break;
        }
    }
    allocations
}

fn extend_fair_allocations(
    requests: &BTreeMap<String, u64>,
    allocations: &mut BTreeMap<String, u64>,
    budget: u64,
) -> Result<u64, RegistrationError> {
    let remaining = requests
        .iter()
        .map(|(name, requested)| {
            (
                name.clone(),
                requested.saturating_sub(allocations.get(name).copied().unwrap_or(0)),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let additional = fair_cache_allocations(&remaining, budget);
    let used = allocation_total(&additional)?;
    for (name, bytes) in additional {
        allocations
            .entry(name)
            .and_modify(|allocated| *allocated = allocated.saturating_add(bytes))
            .or_insert(bytes);
    }
    Ok(budget.saturating_sub(used))
}

fn allocation_total(allocations: &BTreeMap<String, u64>) -> Result<u64, RegistrationError> {
    allocations.values().try_fold(0_u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or(RegistrationError::ArithmeticOverflow)
    })
}

fn apply_allocation_totals(
    shared: &mut SharedLedger,
    allocations: &BTreeMap<String, CacheAllocation>,
) -> Result<(), RegistrationError> {
    shared.retained_bytes = 0;
    shared.prefix_cache_bytes = 0;
    for allocation in allocations.values() {
        shared.retained_bytes = shared
            .retained_bytes
            .checked_add(allocation.retained_bytes)
            .ok_or(RegistrationError::ArithmeticOverflow)?;
        shared.prefix_cache_bytes = shared
            .prefix_cache_bytes
            .checked_add(allocation.prefix_bytes)
            .ok_or(RegistrationError::ArithmeticOverflow)?;
    }
    Ok(())
}

fn snapshot_for(
    boot_id: &str,
    model: &str,
    entry: &ModelEntry,
    pressure: MemoryPressure,
    allocation: CacheAllocation,
) -> CapacitySnapshot {
    let active = entry
        .active
        .as_ref()
        .filter(|active| active.published && !active.draining);
    let (availability, safe_total_tokens, recommended_output_tokens, max_prompt_tokens) = active
        .map(|active| {
            let decision = active.controller.decision();
            (
                decision.availability,
                decision.safe_total_tokens,
                decision.recommended_output_tokens,
                decision.max_prompt_tokens,
            )
        })
        .unwrap_or((CapacityAvailability::Unavailable, 0, 0, 0));
    CapacitySnapshot {
        schema_version: CAPACITY_SCHEMA_VERSION,
        model: model.to_owned(),
        model_fingerprint: entry.last_fingerprint.clone(),
        boot_id: boot_id.to_owned(),
        generation: entry.generation,
        availability,
        pressure,
        safe_total_tokens,
        recommended_output_tokens,
        max_prompt_tokens,
        retained_session_tokens: active.map_or(0, |active| {
            if !active.facts.cache_capabilities.retained_sessions {
                0
            } else if active.facts.retained_session_tokens == 0 {
                max_prompt_tokens
            } else {
                active.facts.retained_session_tokens.min(max_prompt_tokens)
            }
        }),
        retained_bytes: active.map_or(0, |_| allocation.retained_bytes),
        prefix_cache_bytes: active.map_or(0, |_| allocation.prefix_bytes),
        basis: active.map_or(CapacityBasis::Conservative, |active| active.basis),
    }
}

/// Hash authoritative model/config/tokenizer/template artifacts by normalized
/// relative path, length, and bytes. Runtime caches, logs, docs, and unrelated
/// files are deliberately excluded from model identity.
/// Directory symlinks are rejected; regular-file symlinks are streamed as files.
pub fn fingerprint_model_artifacts(root: &Path) -> io::Result<ModelContentIdentity> {
    let root_meta = fs::symlink_metadata(root)?;
    if !root_meta.is_dir() || root_meta.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "model artifact root must be a real directory",
        ));
    }
    let mut files = Vec::new();
    collect_artifacts(root, root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    if files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "model artifact tree contains no authoritative artifacts",
        ));
    }

    let mut hash = Sha256::new();
    hash.update(FINGERPRINT_DOMAIN);
    let mut artifact_bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    for (relative, path) in files {
        let metadata = fs::metadata(&path)?;
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "artifact changed type while fingerprinting",
            ));
        }
        let length = metadata.len();
        artifact_bytes = artifact_bytes
            .checked_add(length)
            .ok_or_else(|| io::Error::other("artifact byte count overflow"))?;
        let path_bytes = relative.as_bytes();
        hash.update(
            u64::try_from(path_bytes.len())
                .map_err(|_| io::Error::other("artifact path length overflow"))?
                .to_le_bytes(),
        );
        hash.update(path_bytes);
        hash.update(length.to_le_bytes());

        let mut file = File::open(path)?;
        let mut read_bytes = 0_u64;
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            read_bytes = read_bytes
                .checked_add(u64::try_from(count).map_err(io::Error::other)?)
                .ok_or_else(|| io::Error::other("artifact read count overflow"))?;
            hash.update(&buffer[..count]);
        }
        if read_bytes != length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "artifact length changed while fingerprinting",
            ));
        }
    }

    let digest = hash.finalize();
    let mut encoded = String::with_capacity(7 + digest.len() * 2);
    encoded.push_str("sha256:");
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(ModelContentIdentity {
        fingerprint: encoded,
        artifact_bytes,
    })
}

fn collect_artifacts(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            let target = fs::metadata(&path)?;
            if target.is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "model artifact directory symlink is not allowed",
                ));
            }
            if !is_relevant_model_artifact(&path) {
                continue;
            }
            if !target.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "model artifact symlink must target a regular file",
                ));
            }
            files.push((normalized_relative(root, &path)?, path));
        } else if metadata.is_dir() {
            collect_artifacts(root, &path, files)?;
        } else if metadata.is_file() && is_relevant_model_artifact(&path) {
            files.push((normalized_relative(root, &path)?, path));
        } else if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported model artifact file type",
            ));
        }
    }
    Ok(())
}

fn is_relevant_model_artifact(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    name.ends_with(".safetensors")
        || name.ends_with(".safetensors.index.json")
        || name.ends_with(".model")
        || name.ends_with(".tiktoken")
        || name.ends_with(".jinja")
        || name.ends_with(".tmpl")
        || matches!(
            name,
            "config.json"
                | "generation_config.json"
                | "quantize_config.json"
                | "tokenizer.json"
                | "tokenizer_config.json"
                | "special_tokens_map.json"
                | "preprocessor_config.json"
                | "processor_config.json"
                | "added_tokens.json"
                | "chat_template.json"
                | "vocab.json"
                | "vocab.txt"
                | "merges.txt"
        )
}

fn normalized_relative(root: &Path, path: &Path) -> io::Result<String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "artifact path escaped root"))?;
    let mut components = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(part) => components.push(part.to_str().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "artifact path is not UTF-8")
            })?),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "artifact path is not normalized",
                ));
            }
        }
    }
    Ok(components.join("/"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use higgs_engine::{EngineCostDescription, MlxMemorySnapshot, TransientPrefillEstimate};

    use super::*;
    use crate::capacity::{
        CapacityAvailability, CapacityBasis, LearnedBandEvidence, MemoryPressure,
        PressureObservation,
    };

    const GIB: u64 = 1024 * 1024 * 1024;

    fn facts(name: &str, loaded_model_bytes: u64) -> ModelCapacityFacts {
        ModelCapacityFacts {
            model: name.to_owned(),
            model_fingerprint: format!("sha256:{name}"),
            memory: MlxMemorySnapshot {
                active_bytes: loaded_model_bytes,
                peak_bytes: loaded_model_bytes,
                memory_limit_bytes: Some(24 * GIB),
                metal_recommended_working_set_bytes: Some(24 * GIB),
            },
            costs: EngineCostDescription {
                fixed_live_session_bytes: 256 * 1024 * 1024,
                persistent_bytes_per_token: 20_480,
                decode_workspace_bytes: 256 * 1024 * 1024,
                transient_prefill: TransientPrefillEstimate {
                    base_bytes: GIB,
                    bytes_per_prompt_token: 0,
                    bytes_per_chunk_token: 0,
                    max_prompt_tokens: 1_048_576,
                    max_chunk_tokens: 4_096,
                },
            },
            loaded_model_bytes,
            architectural_max_tokens: 1_048_576,
            prefill_chunk_tokens: 1_024,
            retained_session_tokens: 49_152,
            retained_resident_bytes: 0,
            prefix_cache_resident_bytes: 0,
            retained_bytes_ceiling: 2 * GIB,
            prefix_cache_bytes_ceiling: GIB,
            cache_capabilities: CacheCapabilities::SIMPLE,
            configured_total_token_ceiling: None,
            configured_output_token_ceiling: Some(4_096),
            quantization: "3bit".to_owned(),
            execution_mode: "native".to_owned(),
            kv_representation: "fp16".to_owned(),
            prefill_model_identity: None,
            drafter_identity: None,
            learned_profile_key: None,
            startup_headroom_bytes: 0,
        }
    }

    fn register(registry: &Arc<CapacityRegistry>, facts: ModelCapacityFacts) {
        registry.refresh_memory(facts.memory);
        let ticket = registry.begin_registration(facts.model.clone()).unwrap();
        registry.commit_active(ticket, facts).unwrap().publish();
        let plan = registry.cache_allocation_plan();
        assert!(registry.publish_cache_allocation_revision(plan.revision));
    }

    #[test]
    fn snapshots_publish_cache_bytes_only_after_exact_engine_acknowledgement() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        let model_facts = facts("model", 5 * GIB);
        registry.refresh_memory(model_facts.memory);
        let ticket = registry.begin_registration("model".to_owned()).unwrap();
        let active = registry.commit_active(ticket, model_facts).unwrap();
        let plan = registry.cache_allocation_plan();

        let before = registry.snapshot("model").unwrap();
        assert_eq!((before.retained_bytes, before.prefix_cache_bytes), (0, 0));
        assert_eq!(before.availability, CapacityAvailability::Unavailable);

        assert!(registry.publish_cache_allocation_revision(plan.revision));
        let acknowledged_but_hidden = registry.snapshot("model").unwrap();
        assert_eq!(
            (
                acknowledged_but_hidden.retained_bytes,
                acknowledged_but_hidden.prefix_cache_bytes,
            ),
            (0, 0),
            "a provisional model stays unavailable until route insertion commits"
        );
        active.publish();

        let published = registry.snapshot("model").unwrap();
        assert_eq!(
            (published.retained_bytes, published.prefix_cache_bytes),
            (plan.allocations[0].1, plan.allocations[0].2)
        );
        assert_eq!(published.availability, CapacityAvailability::Available);
    }

    fn learned_key(model: &str) -> LearnedProfileKey {
        LearnedProfileKey {
            hardware_identifier: "Mac15,9".into(),
            physical_memory_bytes: 64 * GIB,
            os_version: "15.6".into(),
            os_build: "24G90".into(),
            backend_authority_bytes: 24 * GIB,
            higgs_build: "sha256:higgs".into(),
            model_fingerprint: format!("sha256:{model}"),
            quantization: "3bit".into(),
            execution_mode: "native".into(),
            kv_representation: "fp16".into(),
            prefill_model_identity: None,
            execution_cache_fingerprint: "sha256:settings".into(),
            drafter_identity: None,
        }
    }

    #[test]
    fn matching_learned_profile_restores_across_process_boots() {
        let directory = tempfile::tempdir().unwrap();
        let seed = CapacityRegistry::new_with_profile_dir(
            ["escha".to_owned()],
            directory.path().to_owned(),
        );
        let profile = LearnedProfile::new(
            learned_key("escha"),
            12 * GIB,
            vec![LearnedBandEvidence {
                prompt_band: 65_536,
                cold_high_water_bytes: 2 * GIB,
                cold_replacement_qualified: true,
                retained_high_water_bytes: GIB,
                suffix_high_water_bytes: GIB / 2,
            }],
        );
        seed.profile_store("escha").unwrap().save(&profile).unwrap();

        let mut model = facts("escha", 5 * GIB);
        model.learned_profile_key = Some(learned_key("escha"));
        model.startup_headroom_bytes = 12 * GIB;
        register(&seed, model.clone());
        assert_eq!(
            seed.snapshot("escha").unwrap().basis,
            CapacityBasis::Learned
        );
        seed.persist_profiles().unwrap();

        let restarted = CapacityRegistry::new_with_profile_dir(
            ["escha".to_owned()],
            directory.path().to_owned(),
        );
        assert_ne!(seed.boot_id(), restarted.boot_id());
        register(&restarted, model);
        assert_eq!(
            restarted.snapshot("escha").unwrap().basis,
            CapacityBasis::Learned
        );
    }

    #[test]
    fn learned_profile_headroom_regression_falls_back_conservative() {
        let directory = tempfile::tempdir().unwrap();
        let registry = CapacityRegistry::new_with_profile_dir(
            ["escha".to_owned()],
            directory.path().to_owned(),
        );
        let profile = LearnedProfile::new(
            learned_key("escha"),
            12 * GIB,
            vec![LearnedBandEvidence {
                prompt_band: 65_536,
                cold_high_water_bytes: 2 * GIB,
                cold_replacement_qualified: true,
                retained_high_water_bytes: GIB,
                suffix_high_water_bytes: GIB / 2,
            }],
        );
        registry
            .profile_store("escha")
            .unwrap()
            .save(&profile)
            .unwrap();
        let mut model = facts("escha", 5 * GIB);
        model.learned_profile_key = Some(learned_key("escha"));
        model.startup_headroom_bytes = 12 * GIB - 1;
        register(&registry, model);
        assert_eq!(
            registry.snapshot("escha").unwrap().basis,
            CapacityBasis::Conservative
        );
    }

    #[test]
    fn known_unloaded_snapshot_is_zeroed_and_uses_process_boot_id() {
        let registry = CapacityRegistry::new(["escha".to_owned(), "draft".to_owned()]);

        let escha = registry.snapshot("escha").unwrap();
        let draft = registry.snapshot("draft").unwrap();
        assert_eq!(escha.boot_id, draft.boot_id);
        assert_eq!(escha.boot_id, registry.boot_id());
        assert_eq!(escha.availability, CapacityAvailability::Unavailable);
        assert_eq!(escha.safe_total_tokens, 0);
        assert_eq!(escha.recommended_output_tokens, 0);
        assert_eq!(escha.max_prompt_tokens, 0);
        assert_eq!(escha.retained_session_tokens, 0);
        assert_eq!(escha.retained_bytes, 0);
        assert_eq!(escha.prefix_cache_bytes, 0);
        assert_eq!(escha.basis, CapacityBasis::Conservative);
        assert!(registry.snapshot("unknown").is_err());
    }

    #[test]
    fn registration_is_transactional_and_generation_is_monotonic() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let first_generation = registry.snapshot("escha").unwrap().generation;
        {
            let _ticket = registry.begin_registration("escha".to_owned()).unwrap();
        }
        assert_eq!(
            registry.snapshot("escha").unwrap().generation,
            first_generation
        );

        let ticket = registry.begin_registration("escha".to_owned()).unwrap();
        let model_facts = facts("escha", 5 * GIB);
        registry.refresh_memory(model_facts.memory);
        let active = registry.commit_active(ticket, model_facts).unwrap();
        let active_generation = registry.snapshot("escha").unwrap().generation;
        assert!(active_generation > first_generation);
        drop(active);

        let rolled_back = registry.snapshot("escha").unwrap();
        assert_eq!(rolled_back.availability, CapacityAvailability::Unavailable);
        assert!(rolled_back.generation > active_generation);
    }

    #[test]
    fn failed_dynamic_registration_does_not_expand_the_known_catalog() {
        let registry = CapacityRegistry::new(std::iter::empty());
        {
            let _ticket = registry.begin_registration("dynamic".to_owned()).unwrap();
        }
        assert!(matches!(
            registry.snapshot("dynamic"),
            Err(RegistrationError::UnknownModel(_))
        ));

        let ticket = registry.begin_registration("dynamic".to_owned()).unwrap();
        let model_facts = facts("dynamic", 5 * GIB);
        registry.refresh_memory(model_facts.memory);
        let active = registry.commit_active(ticket, model_facts).unwrap();
        drop(active);
        assert!(matches!(
            registry.snapshot("dynamic"),
            Err(RegistrationError::UnknownModel(_))
        ));
    }

    #[test]
    fn all_models_share_one_residency_ledger() {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        register(&registry, facts("first", 5 * GIB));
        let before = registry.snapshot("first").unwrap().safe_total_tokens;

        register(&registry, facts("second", 5 * GIB));
        let first_after = registry.snapshot("first").unwrap();
        let second_after = registry.snapshot("second").unwrap();
        assert!(first_after.safe_total_tokens < before);
        assert_eq!(
            first_after.safe_total_tokens,
            second_after.safe_total_tokens
        );
        assert_eq!(first_after.retained_bytes, second_after.retained_bytes);
        assert_eq!(
            first_after.prefix_cache_bytes,
            second_after.prefix_cache_bytes
        );
        assert!(first_after.retained_bytes <= 2 * GIB);
        assert!(first_after.prefix_cache_bytes <= GIB);
    }

    #[test]
    fn zero_cache_ceiling_is_automatic_and_does_not_disable_other_models() {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        register(&registry, facts("first", 5 * GIB));
        let mut second = facts("second", 5 * GIB);
        second.retained_bytes_ceiling = 0;
        second.prefix_cache_bytes_ceiling = 0;
        register(&registry, second);

        let first = registry.snapshot("first").unwrap();
        let second = registry.snapshot("second").unwrap();
        assert!(first.retained_bytes > 0);
        assert!(first.prefix_cache_bytes > 0);
        assert!(second.retained_bytes > 0);
        assert!(second.prefix_cache_bytes > 0);
    }

    #[test]
    fn unsupported_cache_classes_are_neither_allocated_nor_charged() {
        let registry = CapacityRegistry::new(["batch".to_owned(), "simple".to_owned()]);
        let mut batch = facts("batch", 5 * GIB);
        batch.cache_capabilities = CacheCapabilities {
            retained_sessions: false,
            prefix_cache: true,
        };
        batch.retained_bytes_ceiling = 0;
        batch.prefix_cache_bytes_ceiling = 0;
        register(&registry, batch);

        let batch = registry.snapshot("batch").unwrap();
        assert_eq!(batch.retained_bytes, 0);
        assert_eq!(batch.retained_session_tokens, 0);
        assert!(batch.prefix_cache_bytes > 0);

        let mut simple = facts("simple", 5 * GIB);
        simple.retained_bytes_ceiling = 0;
        simple.prefix_cache_bytes_ceiling = 0;
        register(&registry, simple);
        let simple = registry.snapshot("simple").unwrap();
        assert!(simple.retained_bytes > 0);
        assert!(simple.prefix_cache_bytes > 0);
    }

    #[test]
    fn constrained_shared_ledger_recompute_preserves_pressure_downshift() {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        register(&registry, facts("first", 5 * GIB));
        let normal = registry.snapshot("first").unwrap().safe_total_tokens;

        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 1,
        });
        let constrained = registry.snapshot("first").unwrap().safe_total_tokens;
        assert!(constrained <= normal * 75 / 100);

        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 4 * GIB,
            peak_bytes: 5 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        assert!(
            registry.snapshot("first").unwrap().safe_total_tokens <= constrained,
            "a shared-ledger replacement must not discard the live pressure bound"
        );
    }

    #[test]
    fn pressure_seed_before_boot_registration_controls_first_policy() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 1,
        });
        register(&registry, facts("escha", 5 * GIB));

        let snapshot = registry.snapshot("escha").unwrap();
        assert_eq!(snapshot.pressure, MemoryPressure::Constrained);
        assert_eq!(
            registry.cache_allocation_plan().pressure,
            MemoryPressure::Constrained
        );
    }

    #[test]
    fn normal_shared_ledger_recompute_does_not_instantly_restore_capacity() {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        register(&registry, facts("first", 5 * GIB));
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 1,
        });
        let constrained = registry.snapshot("first").unwrap().safe_total_tokens;
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        let recovering = registry.snapshot("first").unwrap().safe_total_tokens;
        register(&registry, facts("second", 5 * GIB));
        assert!(registry.snapshot("first").unwrap().safe_total_tokens <= recovering);
        assert!(recovering <= constrained.saturating_add(131_072));
    }

    #[test]
    fn process_cache_allocations_stay_inside_one_shared_envelope() {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        register(&registry, facts("first", 5 * GIB));
        register(&registry, facts("second", 5 * GIB));

        let allocations = registry.cache_allocations();
        let total = allocations
            .iter()
            .map(|(_, retained, prefix)| retained.checked_add(*prefix).unwrap())
            .try_fold(0_u64, u64::checked_add)
            .unwrap();
        let process_envelope = 24 * GIB - (24 * GIB * 20 / 100) - 5 * GIB;
        assert!(total <= process_envelope);
        assert_eq!(allocations.len(), 2);
    }

    #[test]
    fn stale_cache_policy_revision_cannot_be_published() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        register(&registry, facts("escha", 5 * GIB));
        let first = registry.cache_allocation_plan();
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Critical,
            swap_out_delta: 1,
            compressor_delta: 1,
        });
        let second = registry.cache_allocation_plan();
        assert!(second.revision > first.revision);
        assert!(!registry.publish_cache_allocation_revision(first.revision));
        assert!(registry.publish_cache_allocation_revision(second.revision));
    }

    #[test]
    fn allocated_caches_are_reserved_by_every_models_solver() {
        let minimal_caches = CapacityRegistry::new(["escha".to_owned()]);
        let mut minimal = facts("escha", 5 * GIB);
        minimal.retained_bytes_ceiling = 1;
        minimal.prefix_cache_bytes_ceiling = 1;
        register(&minimal_caches, minimal);

        let with_caches = CapacityRegistry::new(["escha".to_owned()]);
        register(&with_caches, facts("escha", 5 * GIB));

        assert!(
            with_caches.snapshot("escha").unwrap().safe_total_tokens
                < minimal_caches.snapshot("escha").unwrap().safe_total_tokens
        );
    }

    #[test]
    fn retained_token_cap_never_exceeds_the_published_prompt_cap() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let mut model = facts("escha", 5 * GIB);
        model.retained_session_tokens = u64::MAX;
        register(&registry, model);

        let snapshot = registry.snapshot("escha").unwrap();
        assert_eq!(snapshot.retained_session_tokens, snapshot.max_prompt_tokens);
    }

    #[test]
    fn zero_retained_token_config_publishes_the_effective_prompt_cap() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let mut model = facts("escha", 5 * GIB);
        model.retained_session_tokens = 0;
        register(&registry, model);

        let snapshot = registry.snapshot("escha").unwrap();
        assert!(snapshot.max_prompt_tokens > 0);
        assert_eq!(snapshot.retained_session_tokens, snapshot.max_prompt_tokens);
    }

    #[test]
    fn draining_keeps_identity_unavailable_until_explicit_finish() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        register(&registry, facts("escha", 5 * GIB));
        let active_generation = registry.snapshot("escha").unwrap().generation;

        let drain = registry.begin_drain("escha").unwrap();
        let draining = registry.snapshot("escha").unwrap();
        assert_eq!(draining.availability, CapacityAvailability::Unavailable);
        assert_eq!(draining.model_fingerprint, "sha256:escha");
        assert!(draining.generation > active_generation);

        registry.finish_unregister(drain, None).unwrap();
        let unloaded = registry.snapshot("escha").unwrap();
        assert_eq!(unloaded.availability, CapacityAvailability::Unavailable);
        assert!(unloaded.generation > draining.generation);
        assert_eq!(unloaded.model_fingerprint, "sha256:escha");
    }

    #[test]
    fn dropping_drain_token_rolls_back_to_active() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        register(&registry, facts("escha", 5 * GIB));
        let before = registry.snapshot("escha").unwrap();
        drop(registry.begin_drain("escha").unwrap());
        let after = registry.snapshot("escha").unwrap();
        assert_eq!(after.availability, CapacityAvailability::Available);
        assert!(after.generation > before.generation);
    }

    #[test]
    fn final_unregister_refreshes_memory_without_bypassing_recovery_ramp() {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        register(&registry, facts("first", 5 * GIB));
        let mut second = facts("second", 5 * GIB);
        second.memory.active_bytes = 10 * GIB;
        second.memory.peak_bytes = 10 * GIB;
        register(&registry, second);
        let constrained = registry.snapshot("second").unwrap().safe_total_tokens;

        let drain = registry.begin_drain("first").unwrap();
        assert_eq!(
            registry.snapshot("second").unwrap().safe_total_tokens,
            constrained,
            "draining allocations stay reserved until the engine is destroyed"
        );
        let measurement = registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 5 * GIB,
            peak_bytes: 10 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        registry
            .finish_unregister(drain, Some(measurement))
            .unwrap();

        assert!(registry.snapshot("second").unwrap().safe_total_tokens <= constrained);
    }

    #[test]
    fn later_load_measurement_survives_out_of_order_registration_commits() {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        let first_ticket = registry.begin_registration("first".to_owned()).unwrap();
        let second_ticket = registry.begin_registration("second".to_owned()).unwrap();
        let mut first = facts("first", 5 * GIB);
        let mut second = facts("second", 5 * GIB);
        first.memory.active_bytes = 5 * GIB;
        second.memory.active_bytes = 10 * GIB;

        registry.refresh_memory(first.memory);
        registry.refresh_memory(second.memory);
        let second_active = registry.commit_active(second_ticket, second).unwrap();
        let first_active = registry.commit_active(first_ticket, first).unwrap();

        assert_eq!(registry.lock().memory.active_bytes, 10 * GIB);
        assert!(
            registry.snapshot("first").unwrap().safe_total_tokens
                <= registry.snapshot("second").unwrap().safe_total_tokens
        );
        drop((first_active, second_active));
    }

    #[test]
    fn stale_unload_measurement_cannot_replace_a_later_load_measurement() {
        let registry = CapacityRegistry::new(["old".to_owned(), "new".to_owned()]);
        register(&registry, facts("old", 5 * GIB));
        let drain = registry.begin_drain("old").unwrap();
        let after_unload = MlxMemorySnapshot {
            active_bytes: 0,
            peak_bytes: 5 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        };
        let unload_measurement = registry.refresh_memory(after_unload);

        let mut new = facts("new", 5 * GIB);
        new.memory.active_bytes = 8 * GIB;
        registry.refresh_memory(new.memory);
        let ticket = registry.begin_registration("new".to_owned()).unwrap();
        registry.commit_active(ticket, new).unwrap().publish();

        registry
            .finish_unregister(drain, Some(unload_measurement))
            .unwrap();
        assert_eq!(registry.lock().memory.active_bytes, 8 * GIB);
        assert!(registry.snapshot("new").unwrap().safe_total_tokens > 0);
    }

    #[test]
    fn draining_cache_allocation_is_frozen_across_pressure_recomputation() {
        let registry = CapacityRegistry::new(["draining".to_owned(), "new".to_owned()]);
        let mut draining = facts("draining", 5 * GIB);
        draining.retained_bytes_ceiling = 6 * GIB;
        draining.prefix_cache_bytes_ceiling = 6 * GIB;
        register(&registry, draining);

        let drain = registry.begin_drain("draining").unwrap();
        let frozen = registry
            .cache_allocations()
            .into_iter()
            .find(|(name, _, _)| name == "draining")
            .unwrap();

        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 1,
        });

        let after = registry
            .cache_allocations()
            .into_iter()
            .find(|(name, _, _)| name == "draining")
            .unwrap();
        assert_eq!(after, frozen);
        drop(drain);
    }

    #[test]
    fn critical_pressure_evicts_active_optional_caches_but_keeps_draining_reserved() {
        let registry = CapacityRegistry::new(["draining".to_owned(), "active".to_owned()]);
        register(&registry, facts("draining", 5 * GIB));
        register(&registry, facts("active", 5 * GIB));
        let drain = registry.begin_drain("draining").unwrap();
        let frozen = registry
            .cache_allocations()
            .into_iter()
            .find(|(name, _, _)| name == "draining")
            .unwrap();

        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Critical,
            swap_out_delta: 1,
            compressor_delta: 1,
        });
        let allocations = registry.cache_allocations();
        assert_eq!(
            allocations
                .iter()
                .find(|(name, _, _)| name == "draining")
                .unwrap(),
            &frozen
        );
        assert_eq!(
            allocations
                .iter()
                .find(|(name, _, _)| name == "active")
                .map(|(_, retained, prefix)| (*retained, *prefix)),
            Some((0, 0))
        );

        registry.finish_unregister(drain, None).unwrap();
        assert!(
            registry
                .cache_allocations()
                .iter()
                .all(|(name, _, _)| name != "draining")
        );
    }

    #[test]
    fn capacity_revision_during_drain_cannot_strand_the_lifecycle_token() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        register(&registry, facts("escha", 5 * GIB));
        let drain = registry.begin_drain("escha").unwrap();
        let draining_generation = registry.snapshot("escha").unwrap().generation;
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 1,
        });
        assert!(registry.snapshot("escha").unwrap().generation > draining_generation);
        registry.finish_unregister(drain, None).unwrap();

        assert!(
            registry.begin_registration("escha".to_owned()).is_ok(),
            "capacity generation changes must not invalidate the unique drain token"
        );
    }

    #[test]
    fn capacity_revision_during_publication_cannot_strand_the_lifecycle_token() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let model_facts = facts("escha", 5 * GIB);
        registry.refresh_memory(model_facts.memory);
        let ticket = registry.begin_registration("escha".to_owned()).unwrap();
        let active = registry.commit_active(ticket, model_facts).unwrap();
        let provisional_generation = registry.snapshot("escha").unwrap().generation;
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 1,
        });
        assert!(registry.snapshot("escha").unwrap().generation > provisional_generation);
        drop(active);

        assert!(
            registry.begin_registration("escha".to_owned()).is_ok(),
            "capacity generation changes must not invalidate unpublished rollback"
        );
    }

    #[test]
    fn pressure_updates_every_active_model_under_one_registry_revision() {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        register(&registry, facts("first", 4 * GIB));
        register(&registry, facts("second", 4 * GIB));

        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        });

        for name in ["first", "second"] {
            let snapshot = registry.snapshot(name).unwrap();
            assert_eq!(snapshot.pressure, MemoryPressure::Constrained);
            assert_eq!(snapshot.boot_id, registry.boot_id());
        }
    }

    #[test]
    fn process_pressure_state_is_retained_before_any_model_registers() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 1,
            compressor_delta: 0,
        });

        assert_eq!(
            registry.snapshot("escha").unwrap().pressure,
            MemoryPressure::Critical
        );
        let ticket = registry.begin_registration("escha".to_owned()).unwrap();
        assert!(matches!(
            registry.commit_active(ticket, facts("escha", 5 * GIB)),
            Err(RegistrationError::InsufficientCapacity(model)) if model == "escha"
        ));
        let unavailable = registry.snapshot("escha").unwrap();
        assert_eq!(unavailable.pressure, MemoryPressure::Critical);
        assert_eq!(unavailable.availability, CapacityAvailability::Unavailable);
    }

    #[test]
    fn fingerprint_is_stable_and_invalidated_by_content_or_relative_path() {
        let first = tempfile::tempdir().unwrap();
        std::fs::create_dir(first.path().join("weights")).unwrap();
        std::fs::write(first.path().join("config.json"), b"config").unwrap();
        std::fs::write(first.path().join("weights/a.safetensors"), b"weight-a").unwrap();

        let second = tempfile::tempdir().unwrap();
        std::fs::create_dir(second.path().join("weights")).unwrap();
        std::fs::write(second.path().join("weights/a.safetensors"), b"weight-a").unwrap();
        std::fs::write(second.path().join("config.json"), b"config").unwrap();
        assert_eq!(
            fingerprint_model_artifacts(first.path()).unwrap(),
            fingerprint_model_artifacts(second.path()).unwrap()
        );

        std::fs::write(second.path().join("config.json"), b"confiG").unwrap();
        assert_ne!(
            fingerprint_model_artifacts(first.path()).unwrap(),
            fingerprint_model_artifacts(second.path()).unwrap()
        );
        std::fs::write(second.path().join("config.json"), b"config").unwrap();
        std::fs::rename(
            second.path().join("weights/a.safetensors"),
            second.path().join("weights/b.safetensors"),
        )
        .unwrap();
        assert_ne!(
            fingerprint_model_artifacts(first.path()).unwrap(),
            fingerprint_model_artifacts(second.path()).unwrap()
        );
    }

    #[test]
    fn fingerprint_ignores_runtime_cache_logs_and_unrelated_files() {
        let model = tempfile::tempdir().unwrap();
        std::fs::write(model.path().join("config.json"), b"config").unwrap();
        std::fs::write(model.path().join("model.safetensors"), b"weights").unwrap();
        let before = fingerprint_model_artifacts(model.path()).unwrap();

        std::fs::write(
            model.path().join(".higgs-prefix-cache.bin"),
            b"runtime cache",
        )
        .unwrap();
        std::fs::write(model.path().join("server.log"), b"diagnostics").unwrap();
        std::fs::write(model.path().join("README.md"), b"documentation").unwrap();
        std::fs::write(model.path().join("runtime.json"), b"telemetry").unwrap();
        let after = fingerprint_model_artifacts(model.path()).unwrap();

        assert_eq!(after, before);
    }

    #[cfg(unix)]
    #[test]
    fn fingerprint_follows_file_symlinks_but_rejects_directory_symlinks() {
        use std::os::unix::fs::symlink;

        let model = tempfile::tempdir().unwrap();
        let blob = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(blob.path(), b"outside-blob").unwrap();
        symlink(blob.path(), model.path().join("model.safetensors")).unwrap();
        let identity = fingerprint_model_artifacts(model.path()).unwrap();
        assert_eq!(identity.artifact_bytes, 12);

        let linked_dir = tempfile::tempdir().unwrap();
        symlink(linked_dir.path(), model.path().join("linked-dir")).unwrap();
        assert!(fingerprint_model_artifacts(model.path()).is_err());
    }

    #[test]
    fn fingerprint_fails_closed_for_missing_or_non_directory_root() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("file");
        std::fs::write(&file, b"x").unwrap();
        assert!(fingerprint_model_artifacts(&file).is_err());
        assert!(fingerprint_model_artifacts(&root.path().join("missing")).is_err());
    }
}
