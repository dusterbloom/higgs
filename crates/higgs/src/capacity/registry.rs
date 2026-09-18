use super::{
    CAPACITY_SCHEMA_VERSION, CapacityAvailability, CapacityBasis, CapacityExceededError,
    CapacitySnapshot, CapacityUnavailableError, MemoryPressure, PressureObservation,
    PressureTelemetry, RequestCost,
};
use higgs_engine::MlxMemorySnapshot;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
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
    pub architectural_max_tokens: u64,
    pub retained_session_tokens: u64,
    pub retained_bytes_ceiling: u64,
    pub prefix_cache_bytes_ceiling: u64,
    pub cache_capabilities: CacheCapabilities,
    pub configured_total_token_ceiling: Option<u64>,
    pub configured_output_token_ceiling: Option<u64>,
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
    #[error("model '{0}' has no configured context")]
    InsufficientCapacity(String),
}
#[derive(Debug)]
struct ActiveModel {
    facts: ModelCapacityFacts,
    lifecycle_nonce: uuid::Uuid,
    drain_nonce: Option<uuid::Uuid>,
    draining: bool,
    published: bool,
}
#[derive(Debug, Default)]
struct ModelEntry {
    generation: u64,
    last_fingerprint: String,
    active: Option<ActiveModel>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheAllocationPlan {
    pub revision: u64,
    pub pressure: MemoryPressure,
    pub allocations: Vec<(String, u64, u64)>,
}
/// Measurement provenance used by model-unload cleanup, never admission.
pub struct PublishedMemoryMeasurement {
    boot_id: String,
    revision: u64,
}
impl PublishedMemoryMeasurement {
    #[cfg(test)]
    pub(crate) const fn revision(&self) -> u64 {
        self.revision
    }
}
#[derive(Debug)]
struct ActiveReservation {
    model: String,
    created: std::time::Instant,
    stop: higgs_engine::stop::GenerationStop,
}
#[derive(Debug)]
struct RegistryState {
    models: BTreeMap<String, ModelEntry>,
    registering: BTreeSet<String>,
    memory: MlxMemorySnapshot,
    memory_revision: u64,
    pressure: MemoryPressure,
    pressure_telemetry: PressureTelemetry,
    swap_out_delta: u64,
    compressor_delta: u64,
    cache_revision: u64,
    published_cache_revision: u64,
    active_reservations: BTreeMap<uuid::Uuid, ActiveReservation>,
    exceeded: u64,
    unavailable: u64,
}
/// Owns model publication and in-flight lifetimes. Request admission only checks
/// full prompt + requested output against the fixed model context.
#[derive(Debug)]
pub struct CapacityRegistry {
    boot_id: String,
    state: Mutex<RegistryState>,
    reservation_changed: tokio::sync::Notify,
    #[cfg(test)]
    pub(crate) test_memory_measurement:
        Mutex<Option<Result<MlxMemorySnapshot, higgs_engine::MlxMemoryProbeError>>>,
}
#[derive(Debug)]
pub enum CapacityAdmissionError {
    Exceeded(CapacityExceededError),
    Unavailable(CapacityUnavailableError),
}
impl CapacityAdmissionError {
    pub fn boot_id(&self) -> &str {
        match self {
            Self::Exceeded(e) => e.boot_id(),
            Self::Unavailable(e) => e.boot_id(),
        }
    }
    pub const fn generation(&self) -> u64 {
        match self {
            Self::Exceeded(e) => e.generation(),
            Self::Unavailable(e) => e.generation(),
        }
    }
}
/// An in-flight lifetime token, not a prediction or reservation of allocator bytes.
#[must_use = "keep the request guard alive until generation ends"]
#[derive(Debug)]
pub struct RequestReservation {
    registry: Weak<CapacityRegistry>,
    id: uuid::Uuid,
    stop: higgs_engine::stop::GenerationStop,
}
impl RequestReservation {
    pub fn stop(&self) -> higgs_engine::stop::GenerationStop {
        self.stop.clone()
    }
}
impl Drop for RequestReservation {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            registry.lock().active_reservations.remove(&self.id);
            registry.reservation_changed.notify_waiters();
        }
    }
}
#[derive(Debug)]
pub(crate) enum RequestReservationAttempt {
    Reserved(RequestReservation),
    Rejected(CapacityAdmissionError),
}
impl CapacityRegistry {
    pub fn new(known_models: impl IntoIterator<Item = String>) -> Arc<Self> {
        Arc::new(Self {
            boot_id: uuid::Uuid::new_v4().to_string(),
            state: Mutex::new(RegistryState {
                models: known_models
                    .into_iter()
                    .map(|m| (m, ModelEntry::default()))
                    .collect(),
                registering: BTreeSet::new(),
                memory: MlxMemorySnapshot::default(),
                memory_revision: 0,
                pressure: MemoryPressure::Normal,
                pressure_telemetry: PressureTelemetry::default(),
                swap_out_delta: 0,
                compressor_delta: 0,
                cache_revision: 0,
                published_cache_revision: 0,
                active_reservations: BTreeMap::new(),
                exceeded: 0,
                unavailable: 0,
            }),
            reservation_changed: tokio::sync::Notify::new(),
            #[cfg(test)]
            test_memory_measurement: Mutex::new(None),
        })
    }
    pub fn boot_id(&self) -> String {
        self.boot_id.clone()
    }
    pub async fn reserve_request(
        self: &Arc<Self>,
        model: &str,
        request: RequestCost,
    ) -> Result<RequestReservation, CapacityAdmissionError> {
        match self.try_reserve_request(model, request) {
            RequestReservationAttempt::Reserved(r) => Ok(r),
            RequestReservationAttempt::Rejected(e) => Err(e),
        }
    }
    pub(crate) fn try_reserve_request(
        self: &Arc<Self>,
        model: &str,
        request: RequestCost,
    ) -> RequestReservationAttempt {
        let mut state = self.lock();
        let entry = state.models.get(model);
        let generation = entry.map_or(0, |e| e.generation);
        let active = entry
            .and_then(|e| e.active.as_ref())
            .filter(|a| a.published && !a.draining);
        let Some(active) = active else {
            state.unavailable = state.unavailable.saturating_add(1);
            return RequestReservationAttempt::Rejected(CapacityAdmissionError::Unavailable(
                CapacityUnavailableError::new(self.boot_id.clone(), generation),
            ));
        };
        let total = fixed_total(&active.facts);
        let output_ceiling = active
            .facts
            .configured_output_token_ceiling
            .unwrap_or(u64::MAX);
        // Cached tokens still occupy model context. Checked addition also rejects
        // overflow before a worker can mutate any retained session.
        if request.output_tokens > output_ceiling
            || request
                .prompt_tokens
                .checked_add(request.output_tokens)
                .is_none_or(|n| n > total)
        {
            state.exceeded = state.exceeded.saturating_add(1);
            return RequestReservationAttempt::Rejected(CapacityAdmissionError::Exceeded(
                CapacityExceededError::new(
                    total.saturating_sub(request.output_tokens),
                    total,
                    self.boot_id.clone(),
                    generation,
                ),
            ));
        }
        let id = uuid::Uuid::new_v4();
        let stop = higgs_engine::stop::GenerationStop::default();
        state.active_reservations.insert(
            id,
            ActiveReservation {
                model: model.to_owned(),
                created: std::time::Instant::now(),
                stop: stop.clone(),
            },
        );
        RequestReservationAttempt::Reserved(RequestReservation {
            registry: Arc::downgrade(self),
            id,
            stop,
        })
    }
    pub(crate) fn refresh_measured_memory(&self) -> bool {
        #[cfg(test)]
        let measured = match *self.test_memory_measurement.lock().unwrap() {
            Some(measured) => measured,
            None => return true,
        };
        #[cfg(not(test))]
        let measured = MlxMemorySnapshot::measure();
        match measured {
            Ok(memory) => {
                self.refresh_memory(memory);
                true
            }
            Err(error) => {
                tracing::warn!(?error, "capacity allocator measurement failed");
                false
            }
        }
    }
    pub fn active_reservation_count(&self, model: &str) -> usize {
        self.lock()
            .active_reservations
            .values()
            .filter(|r| r.model == model)
            .count()
    }
    #[cfg(test)]
    pub(crate) fn admission_test_memory(&self) -> (MlxMemorySnapshot, u64) {
        let state = self.lock();
        (state.memory, state.memory_revision)
    }
    pub async fn wait_for_model_reservations(&self, model: &str) {
        loop {
            let changed = self.reservation_changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.active_reservation_count(model) == 0 {
                return;
            }
            changed.await;
        }
    }
    pub fn snapshot(&self, model: &str) -> Result<CapacitySnapshot, RegistrationError> {
        let state = self.lock();
        let entry = state
            .models
            .get(model)
            .ok_or_else(|| RegistrationError::UnknownModel(model.to_owned()))?;
        Ok(snapshot_for(&self.boot_id, model, entry, state.pressure))
    }
    pub fn cache_allocations(&self) -> Vec<(String, u64, u64)> {
        self.cache_allocation_plan().allocations
    }
    pub fn cache_allocation_plan(&self) -> CacheAllocationPlan {
        let state = self.lock();
        CacheAllocationPlan {
            revision: state.cache_revision,
            // Existing engine cache APIs use this field to select eviction policy.
            // Fixed ceilings never change because of measured pressure.
            pressure: MemoryPressure::Normal,
            allocations: state
                .models
                .iter()
                .filter_map(|(name, entry)| {
                    entry.active.as_ref().map(|a| {
                        let (retained, prefix) = cache_bounds(&a.facts);
                        (name.clone(), retained, prefix)
                    })
                })
                .collect(),
        }
    }
    pub fn publish_cache_allocation_revision(&self, revision: u64) -> bool {
        let mut state = self.lock();
        if state.cache_revision != revision {
            return false;
        }
        state.published_cache_revision = revision;
        true
    }
    fn publish_active_route_if_current(
        &self,
        model: &str,
        nonce: uuid::Uuid,
        revision: u64,
        publish: impl FnOnce(),
    ) -> bool {
        let mut state = self.lock();
        if state.cache_revision != revision || state.published_cache_revision != revision {
            return false;
        }
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
        if fixed_total(&facts) == 0 {
            return Err(RegistrationError::InsufficientCapacity(facts.model));
        }
        let mut state = self.lock();
        if !state.registering.contains(&ticket.model) {
            return Err(RegistrationError::AlreadyRegistered(ticket.model.clone()));
        }
        let entry = state.models.entry(ticket.model.clone()).or_default();
        entry.generation = entry.generation.saturating_add(1);
        entry.last_fingerprint.clone_from(&facts.model_fingerprint);
        entry.active = Some(ActiveModel {
            facts,
            lifecycle_nonce: ticket.nonce,
            drain_nonce: None,
            draining: false,
            published: false,
        });
        state.registering.remove(&ticket.model);
        state.cache_revision = state.cache_revision.saturating_add(1);
        ticket.pending = false;
        Ok(ActiveRegistration {
            registry: Arc::downgrade(self),
            model: ticket.model.clone(),
            nonce: ticket.nonce,
            published: false,
            remove_on_rollback: ticket.newly_known,
        })
    }
    pub fn begin_drain(
        self: &Arc<Self>,
        model: &str,
    ) -> Result<DrainRegistration, RegistrationError> {
        let mut state = self.lock();
        let entry = state
            .models
            .get_mut(model)
            .ok_or_else(|| RegistrationError::UnknownModel(model.to_owned()))?;
        let active = entry
            .active
            .as_mut()
            .filter(|a| !a.draining)
            .ok_or_else(|| RegistrationError::NotActive(model.to_owned()))?;
        let nonce = uuid::Uuid::new_v4();
        active.draining = true;
        active.drain_nonce = Some(nonce);
        entry.generation = entry.generation.saturating_add(1);
        // Explicit model unload cancels and joins workers before releasing MLX state.
        for r in state
            .active_reservations
            .values()
            .filter(|r| r.model == model)
        {
            r.stop.stop(higgs_engine::stop::StopReason::ModelDrain);
        }
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
        if state
            .active_reservations
            .values()
            .any(|r| r.model == drain.model)
        {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "model still has active requests",
            ));
        }
        if memory_after_release
            .is_some_and(|m| m.boot_id != self.boot_id || m.revision > state.memory_revision)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unload memory measurement does not belong to this registry revision",
            ));
        }
        if let Some(entry) = state.models.get_mut(&drain.model)
            && entry
                .active
                .as_ref()
                .is_some_and(|a| a.draining && a.drain_nonce == Some(drain.nonce))
        {
            entry.active = None;
            entry.generation = entry.generation.saturating_add(1);
            state.cache_revision = state.cache_revision.saturating_add(1);
        }
        drain.finished = true;
        Ok(())
    }
    pub fn apply_pressure_observation(&self, observation: PressureObservation) {
        let telemetry = PressureTelemetry {
            raw_pressure: observation.pressure,
            ..self.lock().pressure_telemetry
        };
        self.apply_pressure_observation_with_telemetry(observation, telemetry);
    }
    pub fn apply_pressure_observation_with_telemetry(
        &self,
        observation: PressureObservation,
        telemetry: PressureTelemetry,
    ) {
        let mut state = self.lock();
        state.pressure = observation.pressure;
        state.pressure_telemetry = telemetry;
        state.swap_out_delta = observation.swap_out_delta;
        state.compressor_delta = observation.compressor_delta;
    }
    pub fn refresh_memory(&self, memory: MlxMemorySnapshot) -> PublishedMemoryMeasurement {
        let mut state = self.lock();
        state.memory = memory;
        state.memory_revision = state.memory_revision.saturating_add(1);
        PublishedMemoryMeasurement {
            boot_id: self.boot_id.clone(),
            revision: state.memory_revision,
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
            state.cache_revision = state.cache_revision.saturating_add(1);
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
            entry.generation = entry.generation.saturating_add(1);
        }
    }
    fn lock(&self) -> MutexGuard<'_, RegistryState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
    pub fn diagnostics(&self) -> super::CapacityDiagnostics {
        let state = self.lock();
        let telemetry = state.pressure_telemetry;
        super::CapacityDiagnostics {
            boot_id: self.boot_id.clone(),
            active_reservations: state.active_reservations.len(),
            oldest_reservation_age_ms: state
                .active_reservations
                .values()
                .map(|r| u64::try_from(r.created.elapsed().as_millis()).unwrap_or(u64::MAX))
                .max(),
            pressure: state.pressure,
            raw_pressure: telemetry.raw_pressure,
            mlx_active_bytes: state.memory.active_bytes,
            mlx_peak_bytes: state.memory.peak_bytes,
            swap_out_delta: state.swap_out_delta,
            compressor_delta: state.compressor_delta,
            swap_outs_total: telemetry.swap_outs_total,
            compressions_total: telemetry.compressions_total,
            pressure_sampled_at_unix_ms: telemetry.sampled_at_unix_ms,
            pressure_sample_epoch: telemetry.sample_epoch,
            non_normal_pressure_event_epoch: telemetry.non_normal_pressure_event_epoch,
            rejections: super::CapacityRejectionDiagnostics {
                exceeded: state.exceeded,
                unavailable: state.unavailable,
            },
            models: state
                .models
                .iter()
                .map(|(name, entry)| {
                    let snapshot = snapshot_for(&self.boot_id, name, entry, state.pressure);
                    super::CapacityModelDiagnostics {
                        model: name.clone(),
                        generation: entry.generation,
                        available: snapshot.availability == CapacityAvailability::Available,
                        basis: CapacityBasis::Configured,
                        pressure: state.pressure,
                        safe_total_tokens: snapshot.safe_total_tokens,
                        recommended_output_tokens: snapshot.recommended_output_tokens,
                        max_prompt_tokens: snapshot.max_prompt_tokens,
                    }
                })
                .collect(),
            ..super::CapacityDiagnostics::default()
        }
    }
}
fn fixed_total(facts: &ModelCapacityFacts) -> u64 {
    facts
        .configured_total_token_ceiling
        .unwrap_or(u64::from(crate::config::default_max_context_tokens()))
        .min(facts.architectural_max_tokens)
}
fn cache_bounds(facts: &ModelCapacityFacts) -> (u64, u64) {
    (
        if facts.cache_capabilities.retained_sessions {
            facts.retained_bytes_ceiling
        } else {
            0
        },
        if facts.cache_capabilities.prefix_cache {
            facts.prefix_cache_bytes_ceiling
        } else {
            0
        },
    )
}
fn snapshot_for(
    boot: &str,
    model: &str,
    entry: &ModelEntry,
    pressure: MemoryPressure,
) -> CapacitySnapshot {
    let active = entry.active.as_ref().filter(|a| a.published && !a.draining);
    let total = active.map_or(0, |a| fixed_total(&a.facts));
    let output = active.map_or(0, |a| {
        a.facts
            .configured_output_token_ceiling
            .unwrap_or(4096)
            .min(4096)
            .min(total)
    });
    let (retained, prefix) = active.map_or((0, 0), |a| cache_bounds(&a.facts));
    CapacitySnapshot {
        schema_version: CAPACITY_SCHEMA_VERSION,
        model: model.to_owned(),
        model_fingerprint: entry.last_fingerprint.clone(),
        boot_id: boot.to_owned(),
        generation: entry.generation,
        availability: if active.is_some() {
            CapacityAvailability::Available
        } else {
            CapacityAvailability::Unavailable
        },
        pressure,
        safe_total_tokens: total,
        recommended_output_tokens: output,
        max_prompt_tokens: total.saturating_sub(output),
        retained_session_tokens: active.map_or(0, |a| {
            if a.facts.cache_capabilities.retained_sessions {
                if a.facts.retained_session_tokens == 0 {
                    total
                } else {
                    a.facts.retained_session_tokens.min(total)
                }
            } else {
                0
            }
        }),
        retained_bytes: retained,
        prefix_cache_bytes: prefix,
        basis: CapacityBasis::Configured,
    }
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
    pub(crate) fn publish_route_if_current(
        &mut self,
        revision: u64,
        publish: impl FnOnce(),
    ) -> bool {
        let published = self.registry.upgrade().is_some_and(|registry| {
            registry.publish_active_route_if_current(&self.model, self.nonce, revision, publish)
        });
        self.published |= published;
        published
    }

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

    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
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
mod fixed_contract_tests {
    use super::*;
    #[test]
    fn fixed_context_survives_pressure_and_counts_cached_prompt() {
        let registry = CapacityRegistry::new(["fixed".to_owned()]);
        let mut facts =
            super::super::route_test_facts("fixed", MlxMemorySnapshot::default(), 0, 32768);
        facts.configured_total_token_ceiling = Some(32768);
        facts.configured_output_token_ceiling = Some(4096);
        facts.architectural_max_tokens = 65536;
        let ticket = registry.begin_registration("fixed".to_owned()).unwrap();
        registry.commit_active(ticket, facts).unwrap().publish();
        let before = registry.snapshot("fixed").unwrap();
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Critical,
            swap_out_delta: 500,
            compressor_delta: 500,
        });
        let after = registry.snapshot("fixed").unwrap();
        assert_eq!(after.safe_total_tokens, 32768);
        assert_eq!(after.safe_total_tokens, before.safe_total_tokens);
        assert_eq!(after.availability, CapacityAvailability::Available);
        assert_eq!(serde_json::to_value(after.basis).unwrap(), "configured");
        let request = RequestCost {
            prompt_tokens: 32768,
            output_tokens: 1,
        };
        assert!(matches!(
            registry.try_reserve_request("fixed", request),
            RequestReservationAttempt::Rejected(CapacityAdmissionError::Exceeded(_))
        ));
    }
}

#[cfg(test)]
mod artifact_tests {
    use super::*;
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

#[cfg(test)]
mod fixed_lifecycle_tests {
    use super::*;
    fn active() -> Arc<CapacityRegistry> {
        let registry = CapacityRegistry::new(["fixed".to_owned()]);
        let mut facts =
            super::super::route_test_facts("fixed", MlxMemorySnapshot::default(), 0, 100);
        facts.architectural_max_tokens = 80;
        facts.configured_output_token_ceiling = Some(20);
        let ticket = registry.begin_registration("fixed".to_owned()).unwrap();
        registry.commit_active(ticket, facts).unwrap().publish();
        registry
    }
    #[tokio::test]
    async fn exact_context_output_limit_overflow_and_guard_cleanup() {
        let registry = active();
        assert_eq!(registry.snapshot("fixed").unwrap().safe_total_tokens, 80);
        let guard = registry
            .reserve_request(
                "fixed",
                RequestCost {
                    prompt_tokens: 60,
                    output_tokens: 20,
                },
            )
            .await
            .unwrap();
        assert_eq!(registry.active_reservation_count("fixed"), 1);
        // A second request uses the same fixed limit; no byte predictor queues it.
        let second = registry
            .reserve_request(
                "fixed",
                RequestCost {
                    prompt_tokens: 60,
                    output_tokens: 20,
                },
            )
            .await
            .unwrap();
        drop(second);
        drop(guard);
        assert_eq!(registry.active_reservation_count("fixed"), 0);
        for (prompt_tokens, output_tokens) in [(61, 20), (1, 21), (u64::MAX, 1)] {
            assert!(matches!(
                registry
                    .reserve_request(
                        "fixed",
                        RequestCost {
                            prompt_tokens,
                            output_tokens
                        }
                    )
                    .await,
                Err(CapacityAdmissionError::Exceeded(_))
            ));
        }
        let guard = registry
            .reserve_request(
                "fixed",
                RequestCost {
                    prompt_tokens: 1,
                    output_tokens: 1,
                },
            )
            .await
            .unwrap();
        let result: Result<(), &str> =
            super::super::run_reserved_generation(guard, None, || Err("allocation failed"));
        assert_eq!(result, Err("allocation failed"));
        assert_eq!(registry.active_reservation_count("fixed"), 0);
        let guard = registry
            .reserve_request(
                "fixed",
                RequestCost {
                    prompt_tokens: 1,
                    output_tokens: 1,
                },
            )
            .await
            .unwrap();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::super::run_reserved_generation(guard, None, || -> Result<(), ()> {
                panic!("allocation panic")
            })
        }));
        assert!(unwind.is_err());
        assert_eq!(registry.active_reservation_count("fixed"), 0);
    }
    #[tokio::test]
    async fn drain_waits_for_worker_and_cancels_future_admission() {
        let registry = active();
        let guard = registry
            .reserve_request(
                "fixed",
                RequestCost {
                    prompt_tokens: 1,
                    output_tokens: 1,
                },
            )
            .await
            .unwrap();
        let drain = registry.begin_drain("fixed").unwrap();
        assert_eq!(
            registry.snapshot("fixed").unwrap().availability,
            CapacityAvailability::Unavailable
        );
        assert!(matches!(
            registry
                .reserve_request(
                    "fixed",
                    RequestCost {
                        prompt_tokens: 1,
                        output_tokens: 1
                    }
                )
                .await,
            Err(CapacityAdmissionError::Unavailable(_))
        ));
        let worker_registry = Arc::clone(&registry);
        let waiter = tokio::spawn(async move {
            worker_registry.wait_for_model_reservations("fixed").await;
        });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        drop(guard);
        waiter.await.unwrap();
        registry.finish_unregister(drain, None).unwrap();
        assert!(registry.begin_registration("fixed".to_owned()).is_ok());
    }
    #[test]
    fn publication_rollback_and_cache_revision_are_transactional() {
        let registry = active();
        let before = registry.snapshot("fixed").unwrap();
        let pending = registry.begin_registration("pending".to_owned()).unwrap();
        let facts = super::super::route_test_facts("pending", MlxMemorySnapshot::default(), 0, 100);
        let registration = registry.commit_active(pending, facts).unwrap();
        assert_eq!(
            registry.snapshot("pending").unwrap().availability,
            CapacityAvailability::Unavailable
        );
        let stale = registry.cache_allocation_plan().revision;
        drop(registration);
        assert!(registry.snapshot("pending").is_err());
        assert!(!registry.publish_cache_allocation_revision(stale));
        assert_eq!(registry.snapshot("fixed").unwrap(), before);
    }
}
