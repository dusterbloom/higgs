use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use higgs_engine::{EngineCostDescription, MlxMemorySnapshot, TransientPrefillEstimate};
use sha2::{Digest, Sha256};

use super::{
    Admission, CAPACITY_SCHEMA_VERSION, CapacityAvailability, CapacityBasis, CapacityController,
    CapacityExceededError, CapacityInputs, CapacitySnapshot, CapacityUnavailableError,
    LearnedProfile, LearnedProfileKey, LearnedProfileStore, MemoryPressure, PressureObservation,
    RequestCost, ZeroCapacityRecovery,
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

/// One structured record per capacity transition: old/new envelope and
/// cause. Reductions additionally count as downshifts for diagnostics.
fn record_capacity_transition(
    state: &mut RegistryState,
    model: &str,
    before: crate::capacity::CapacityDecision,
    after: crate::capacity::CapacityDecision,
    cause: &'static str,
) {
    if after == before {
        return;
    }
    if after.safe_total_tokens < before.safe_total_tokens {
        state.counters.downshifts = state.counters.downshifts.saturating_add(1);
    }
    tracing::info!(
        model = %model,
        from_tokens = before.safe_total_tokens,
        to_tokens = after.safe_total_tokens,
        availability = ?after.availability,
        cause,
        "capacity envelope transition"
    );
}

fn stop_reason_label(reason: &higgs_engine::stop::StopReason) -> &'static str {
    match reason {
        higgs_engine::stop::StopReason::ClientDisconnect => "client_disconnect",
        higgs_engine::stop::StopReason::NoProgressWatchdog => "no_progress_watchdog",
        higgs_engine::stop::StopReason::CriticalPressure { .. } => "critical_pressure",
        higgs_engine::stop::StopReason::ModelDrain => "model_drain",
    }
}

/// Process-wide capacity diagnostics counters. Numbers only — no prompt or
/// request content.
#[derive(Debug, Default)]
struct CapacityCounters {
    rejections_exceeded: u64,
    rejections_unavailable: u64,
    /// Stop-reason label -> released-reservation count.
    stop_outcomes: BTreeMap<String, u64>,
    downshifts: u64,
    last_swap_out_delta: u64,
    last_compressor_delta: u64,
}

#[derive(Debug)]
struct RegistryState {
    models: BTreeMap<String, ModelEntry>,
    registering: BTreeSet<String>,
    pressure_controller: CapacityController,
    pressure: MemoryPressure,
    memory: MlxMemorySnapshot,
    memory_revision: u64,
    capacity_policy_revision: Option<u64>,
    desired_cache_allocations: BTreeMap<String, CacheAllocation>,
    published_cache_allocations: BTreeMap<String, CacheAllocation>,
    cache_revision: u64,
    published_cache_revision: u64,
    cache_plan_pressure: MemoryPressure,
    active_reservations: BTreeMap<uuid::Uuid, ActiveReservation>,
    admission_queue: VecDeque<AdmissionWaiter>,
    counters: CapacityCounters,
}

#[derive(Debug)]
struct ActiveReservation {
    model: String,
    bytes: u64,
    created: std::time::Instant,
    /// Stop signal shared with the owning worker. Critical pressure and model
    /// drain interrupt live reservations through it; the worker acknowledges
    /// at its next allocation boundary and releases the guard only after
    /// allocation has stopped.
    stop: higgs_engine::stop::GenerationStop,
}

#[derive(Debug)]
struct AdmissionWaiter {
    id: uuid::Uuid,
    model: String,
    request: RequestCost,
    boot_id: String,
    notify: Arc<tokio::sync::Notify>,
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

/// Lock-free-by-value facts sampled at a loader allocation boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadCapacitySnapshot {
    pub pressure: MemoryPressure,
    pub headroom_bytes: u64,
}

/// Proof that an allocator snapshot was published to this registry in the
/// same serialized GPU/load window in which it was measured.
pub struct PublishedMemoryMeasurement {
    boot_id: String,
    previous_revision: u64,
    revision: u64,
    previous_active_bytes: u64,
    active_bytes: u64,
    capacity_policy_revision: Option<u64>,
}

impl PublishedMemoryMeasurement {
    #[cfg(test)]
    pub(crate) const fn revision(&self) -> u64 {
        self.revision
    }

    fn authorizes_bounded_recovery(&self, boot_id: &str, state: &RegistryState) -> bool {
        self.boot_id == boot_id
            && self.previous_revision.checked_add(1) == Some(self.revision)
            && self.revision == state.memory_revision
            && self.active_bytes == state.memory.active_bytes
            && self.active_bytes < self.previous_active_bytes
            && self.capacity_policy_revision == state.capacity_policy_revision
            && self.capacity_policy_revision.is_some()
    }
}

/// The single process-wide authority for model capacity and shared residency.
#[derive(Debug)]
pub struct CapacityRegistry {
    boot_id: String,
    profile_dir: Option<PathBuf>,
    state: Mutex<RegistryState>,
    reservation_changed: tokio::sync::Notify,
}

#[derive(Debug)]
pub enum CapacityAdmissionError {
    Exceeded(CapacityExceededError),
    Unavailable(CapacityUnavailableError),
}

impl CapacityAdmissionError {
    #[must_use]
    pub fn boot_id(&self) -> &str {
        match self {
            Self::Exceeded(error) => error.boot_id(),
            Self::Unavailable(error) => error.boot_id(),
        }
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        match self {
            Self::Exceeded(error) => error.generation(),
            Self::Unavailable(error) => error.generation(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CacheReclamation {
    Prefix,
    Retained,
}

#[must_use = "dropping the request reservation releases its process-wide bytes"]
#[derive(Debug)]
pub struct RequestReservation {
    registry: Weak<CapacityRegistry>,
    id: uuid::Uuid,
    bytes: u64,
    released: bool,
    stop: higgs_engine::stop::GenerationStop,
}

impl RequestReservation {
    /// The stop signal shared with the capacity registry. The route worker
    /// installs it thread-locally beside this guard so engine allocation
    /// boundaries observe critical pressure, drain, and disconnect.
    #[must_use]
    pub fn stop(&self) -> higgs_engine::stop::GenerationStop {
        self.stop.clone()
    }

    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for RequestReservation {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        if let Some(registry) = self.registry.upgrade() {
            registry.release_reservation(self.id);
        }
        self.released = true;
    }
}

#[derive(Debug)]
pub(crate) enum RequestReservationAttempt {
    Reserved(RequestReservation),
    Contended,
    Rejected(CapacityAdmissionError),
}

struct QueuedReservation {
    registry: Arc<CapacityRegistry>,
    id: uuid::Uuid,
    notify: Arc<tokio::sync::Notify>,
    completed: bool,
}

impl Drop for QueuedReservation {
    fn drop(&mut self) {
        if !self.completed {
            self.registry.cancel_waiter(self.id);
        }
    }
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
                capacity_policy_revision: Some(0),
                desired_cache_allocations: BTreeMap::new(),
                published_cache_allocations: BTreeMap::new(),
                cache_revision: 0,
                published_cache_revision: 0,
                cache_plan_pressure: MemoryPressure::Normal,
                active_reservations: BTreeMap::new(),
                admission_queue: VecDeque::new(),
                counters: CapacityCounters::default(),
            }),
            reservation_changed: tokio::sync::Notify::new(),
        })
    }

    /// Process-wide adaptive-capacity diagnostics for `/metrics`: live
    /// reservation state, FIFO depth, pressure with swap/compressor deltas,
    /// rejection and cancellation outcome counters, downshift count, peak
    /// MLX allocation, and one row per model. No prompt content.
    #[must_use]
    pub fn diagnostics(&self) -> crate::capacity::CapacityDiagnostics {
        let state = self.lock();
        let now = std::time::Instant::now();
        let oldest = state
            .active_reservations
            .values()
            .map(|reservation| reservation.created)
            .min()
            .map(|created| {
                u64::try_from(now.saturating_duration_since(created).as_millis())
                    .unwrap_or(u64::MAX)
            });
        let models = state
            .models
            .iter()
            .filter_map(|(model, entry)| {
                let active = entry
                    .active
                    .as_ref()
                    .filter(|active| active.published && !active.draining)?;
                let decision = active.controller.decision();
                Some(crate::capacity::CapacityModelDiagnostics {
                    model: model.clone(),
                    generation: entry.generation,
                    available: decision.availability
                        == crate::capacity::CapacityAvailability::Available,
                    basis: active.basis,
                    pressure: state.pressure,
                    safe_total_tokens: decision.safe_total_tokens,
                    recommended_output_tokens: decision.recommended_output_tokens,
                    max_prompt_tokens: decision.max_prompt_tokens,
                    usable_bytes: decision.usable_bytes,
                })
            })
            .collect();
        crate::capacity::CapacityDiagnostics {
            boot_id: self.boot_id.clone(),
            active_reservations: state.active_reservations.len(),
            active_reservation_bytes: active_reservation_bytes(&state).unwrap_or(u64::MAX),
            oldest_reservation_age_ms: oldest,
            queued_waiters: state.admission_queue.len(),
            pressure: state.pressure,
            mlx_active_bytes: state.memory.active_bytes,
            mlx_peak_bytes: state.memory.peak_bytes,
            swap_out_delta: state.counters.last_swap_out_delta,
            compressor_delta: state.counters.last_compressor_delta,
            downshifts: state.counters.downshifts,
            rejections: crate::capacity::CapacityRejectionDiagnostics {
                exceeded: state.counters.rejections_exceeded,
                unavailable: state.counters.rejections_unavailable,
            },
            stop_outcomes: state.counters.stop_outcomes.clone(),
            models,
        }
    }

    #[must_use]
    pub fn boot_id(&self) -> String {
        self.boot_id.clone()
    }

    /// Atomically reserve the request peak, or wait in the process-wide FIFO
    /// when the request is individually safe but blocked by older reservations.
    pub async fn reserve_request(
        self: &Arc<Self>,
        model: &str,
        request: RequestCost,
    ) -> Result<RequestReservation, CapacityAdmissionError> {
        let id = uuid::Uuid::new_v4();
        let notify = Arc::new(tokio::sync::Notify::new());
        let attempt = {
            let mut state = self.lock();
            let attempt = self.try_reserve_locked(&mut state, model, request, false);
            if matches!(attempt, RequestReservationAttempt::Contended) {
                state.admission_queue.push_back(AdmissionWaiter {
                    id,
                    model: model.to_owned(),
                    request,
                    boot_id: self.boot_id.clone(),
                    notify: Arc::clone(&notify),
                });
            }
            attempt
        };
        match attempt {
            RequestReservationAttempt::Reserved(reservation) => Ok(reservation),
            RequestReservationAttempt::Rejected(error) => Err(error),
            RequestReservationAttempt::Contended => {
                let mut queued = QueuedReservation {
                    registry: Arc::clone(self),
                    id,
                    notify,
                    completed: false,
                };
                loop {
                    // Fresh allocator bytes before every grant attempt: the
                    // dequeue revalidation must not decide on memory that was
                    // cached before the waiter entered the FIFO.
                    self.refresh_measured_memory();
                    match self.try_grant_waiter(id, model) {
                        RequestReservationAttempt::Reserved(reservation) => {
                            queued.completed = true;
                            return Ok(reservation);
                        }
                        RequestReservationAttempt::Rejected(error) => {
                            queued.completed = true;
                            return Err(error);
                        }
                        RequestReservationAttempt::Contended => queued.notify.notified().await,
                    }
                }
            }
        }
    }

    pub(crate) fn try_reserve_request(
        self: &Arc<Self>,
        model: &str,
        request: RequestCost,
    ) -> RequestReservationAttempt {
        let mut state = self.lock();
        self.try_reserve_locked(&mut state, model, request, false)
    }

    /// Sample the live MLX allocator and publish it so admission decisions
    /// see fresh bytes, not the last observer's cached snapshot. Positive
    /// unaccounted MLX bytes left by prior work must be visible to an
    /// otherwise-successful admission. Test builds skip sampling: they
    /// publish deterministic memory explicitly via `refresh_memory`.
    pub(crate) fn refresh_measured_memory(&self) {
        if cfg!(test) {
            return;
        }
        if let Ok(memory) = MlxMemorySnapshot::measure() {
            self.refresh_memory(memory);
        }
    }

    fn try_reserve_locked(
        self: &Arc<Self>,
        state: &mut RegistryState,
        model: &str,
        request: RequestCost,
        queue_head: bool,
    ) -> RequestReservationAttempt {
        let Some(entry) = state.models.get(model) else {
            state.counters.rejections_unavailable += 1;
            return RequestReservationAttempt::Rejected(CapacityAdmissionError::Unavailable(
                CapacityUnavailableError::new(self.boot_id.clone(), 0),
            ));
        };
        let generation = entry.generation;
        let Some(active) = entry
            .active
            .as_ref()
            .filter(|active| active.published && !active.draining)
        else {
            state.counters.rejections_unavailable += 1;
            return RequestReservationAttempt::Rejected(CapacityAdmissionError::Unavailable(
                CapacityUnavailableError::new(self.boot_id.clone(), generation),
            ));
        };

        let shared = match effective_shared_ledger(state) {
            Ok(shared) => shared,
            Err(_) => {
                state.counters.rejections_unavailable += 1;
                return RequestReservationAttempt::Rejected(CapacityAdmissionError::Unavailable(
                    CapacityUnavailableError::new(self.boot_id.clone(), generation),
                ));
            }
        };
        let individual = active.controller.transactional_copy();
        let individual_ledger = match individual.admit(request) {
            Admission::Admitted(ledger) => ledger,
            other => {
                let error = admission_error(&self.boot_id, generation, other);
                match error {
                    CapacityAdmissionError::Exceeded(_) => {
                        state.counters.rejections_exceeded += 1;
                    }
                    CapacityAdmissionError::Unavailable(_) => {
                        state.counters.rejections_unavailable += 1;
                    }
                }
                return RequestReservationAttempt::Rejected(error);
            }
        };

        if !queue_head && !state.admission_queue.is_empty() {
            return RequestReservationAttempt::Contended;
        }
        let mut current = active.controller.transactional_copy();
        replace_shared_ledger(&mut current, shared, ZeroCapacityRecovery::Preserve);
        let current = current.admit(request);
        if !matches!(current, Admission::Admitted(_)) {
            return RequestReservationAttempt::Contended;
        }
        let Some(bytes) =
            request_reservation_bytes(individual_ledger, request.retained_growth_bytes)
        else {
            state.counters.rejections_unavailable += 1;
            return RequestReservationAttempt::Rejected(CapacityAdmissionError::Unavailable(
                CapacityUnavailableError::new(self.boot_id.clone(), generation),
            ));
        };
        let id = uuid::Uuid::new_v4();
        let stop = higgs_engine::stop::GenerationStop::default();
        state.active_reservations.insert(
            id,
            ActiveReservation {
                model: model.to_owned(),
                bytes,
                created: std::time::Instant::now(),
                stop: stop.clone(),
            },
        );
        RequestReservationAttempt::Reserved(RequestReservation {
            registry: Arc::downgrade(self),
            id,
            bytes,
            released: false,
            stop,
        })
    }

    fn try_grant_waiter(
        self: &Arc<Self>,
        id: uuid::Uuid,
        model: &str,
    ) -> RequestReservationAttempt {
        let (result, wake_next) = {
            let mut state = self.lock();
            let Some(waiter) = state.admission_queue.front() else {
                let generation = state.models.get(model).map_or(0, |entry| entry.generation);
                return RequestReservationAttempt::Rejected(CapacityAdmissionError::Unavailable(
                    CapacityUnavailableError::new(self.boot_id.clone(), generation),
                ));
            };
            if waiter.id != id && state.admission_queue.iter().any(|waiter| waiter.id == id) {
                return RequestReservationAttempt::Contended;
            }
            if waiter.id != id {
                let generation = state.models.get(model).map_or(0, |entry| entry.generation);
                return RequestReservationAttempt::Rejected(CapacityAdmissionError::Unavailable(
                    CapacityUnavailableError::new(self.boot_id.clone(), generation),
                ));
            }
            let model = waiter.model.clone();
            let request = waiter.request;
            let boot_matches = waiter.boot_id == self.boot_id;
            let result = if boot_matches {
                self.try_reserve_locked(&mut state, &model, request, true)
            } else {
                RequestReservationAttempt::Rejected(CapacityAdmissionError::Unavailable(
                    CapacityUnavailableError::new(self.boot_id.clone(), 0),
                ))
            };
            let wake_next = if matches!(
                result,
                RequestReservationAttempt::Reserved(_) | RequestReservationAttempt::Rejected(_)
            ) {
                state.admission_queue.pop_front();
                restore_cache_policy_if_admission_idle(&mut state);
                state
                    .admission_queue
                    .front()
                    .map(|waiter| Arc::clone(&waiter.notify))
            } else {
                None
            };
            (result, wake_next)
        };
        if let Some(notify) = wake_next {
            notify.notify_one();
        }
        result
    }

    fn cancel_waiter(&self, id: uuid::Uuid) {
        let wake_next = {
            let mut state = self.lock();
            let was_head = state
                .admission_queue
                .front()
                .is_some_and(|waiter| waiter.id == id);
            let Some(index) = state
                .admission_queue
                .iter()
                .position(|waiter| waiter.id == id)
            else {
                return;
            };
            state.admission_queue.remove(index);
            restore_cache_policy_if_admission_idle(&mut state);
            was_head
                .then(|| {
                    state
                        .admission_queue
                        .front()
                        .map(|waiter| Arc::clone(&waiter.notify))
                })
                .flatten()
        };
        if let Some(notify) = wake_next {
            notify.notify_one();
        }
    }

    fn release_reservation(&self, id: uuid::Uuid) {
        let wake = {
            let mut state = self.lock();
            let Some(released) = state.active_reservations.remove(&id) else {
                return;
            };
            if let Some(reason) = released.stop.reason() {
                let label = stop_reason_label(&reason).to_owned();
                *state.counters.stop_outcomes.entry(label).or_default() += 1;
            }
            restore_cache_policy_if_admission_idle(&mut state);
            state
                .admission_queue
                .front()
                .map(|waiter| Arc::clone(&waiter.notify))
        };
        self.reservation_changed.notify_waiters();
        if let Some(notify) = wake {
            notify.notify_one();
        }
    }

    #[must_use]
    pub fn active_reservation_count(&self, model: &str) -> usize {
        self.lock()
            .active_reservations
            .values()
            .filter(|reservation| reservation.model == model)
            .count()
    }

    #[must_use]
    pub fn active_reservation_bytes(&self) -> u64 {
        active_reservation_bytes(&self.lock()).unwrap_or(u64::MAX)
    }

    #[must_use]
    pub fn queued_waiter_count(&self) -> usize {
        self.lock().admission_queue.len()
    }

    #[cfg(test)]
    pub(crate) fn admission_test_memory(&self) -> (MlxMemorySnapshot, u64) {
        let state = self.lock();
        (state.memory, state.memory_revision)
    }

    pub async fn wait_for_model_reservations(&self, model: &str) {
        loop {
            let notified = self.reservation_changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.active_reservation_count(model) == 0 {
                return;
            }
            notified.await;
        }
    }

    /// Lower only desired optional cache budgets. Published accounting remains
    /// unchanged until every engine acknowledges the returned plan revision.
    pub(crate) fn request_cache_reclamation(&self, reclamation: CacheReclamation) -> bool {
        let mut state = self.lock();
        let frozen = state
            .models
            .iter()
            .filter(|(_, entry)| entry.active.as_ref().is_some_and(|active| active.draining))
            .map(|(model, _)| model.clone())
            .collect::<BTreeSet<_>>();
        let mut changed = false;
        for (model, allocation) in &mut state.desired_cache_allocations {
            if frozen.contains(model) {
                continue;
            }
            match reclamation {
                CacheReclamation::Prefix => {
                    changed |= allocation.prefix_bytes != 0;
                    allocation.prefix_bytes = 0;
                }
                CacheReclamation::Retained => {
                    changed |= allocation.retained_bytes != 0;
                    allocation.retained_bytes = 0;
                }
            }
        }
        if changed {
            state.cache_revision = state.cache_revision.saturating_add(1);
        }
        let acknowledgement_pending =
            state
                .published_cache_allocations
                .iter()
                .any(|(model, published)| {
                    if frozen.contains(model) {
                        return false;
                    }
                    let desired = state
                        .desired_cache_allocations
                        .get(model)
                        .copied()
                        .unwrap_or_default();
                    match reclamation {
                        CacheReclamation::Prefix => published.prefix_bytes > desired.prefix_bytes,
                        CacheReclamation::Retained => {
                            published.retained_bytes > desired.retained_bytes
                        }
                    }
                });
        changed || acknowledgement_pending
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

    /// Copy the current process load envelope while holding the registry lock;
    /// callers release it before any MLX work.
    pub fn load_snapshot(&self) -> Option<LoadCapacitySnapshot> {
        let state = self.lock();
        let authority = [
            state.memory.memory_limit_bytes,
            state.memory.metal_recommended_working_set_bytes,
        ]
        .into_iter()
        .flatten()
        .filter(|bytes| *bytes > 0)
        .min()?;
        let percentage = match state.pressure {
            MemoryPressure::Normal => 20,
            MemoryPressure::Constrained | MemoryPressure::Critical => 30,
        };
        let reserve = authority
            .checked_mul(percentage)?
            .checked_div(100)?
            .max(4 * 1024 * 1024 * 1024);
        Some(LoadCapacitySnapshot {
            pressure: state.pressure,
            headroom_bytes: authority
                .checked_sub(reserve)?
                .checked_sub(state.memory.active_bytes)?,
        })
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
        } else {
            shared.active_reservation_bytes = 0;
        }
        recompute_active_models(
            &mut state,
            shared,
            ZeroCapacityRecovery::Preserve,
            "cache reclamation remeasure",
        );
        let wake = state
            .admission_queue
            .front()
            .map(|waiter| Arc::clone(&waiter.notify));
        drop(state);
        if let Some(notify) = wake {
            notify.notify_one();
        }
        true
    }

    /// Publish the exact retained residency left after every engine atomically
    /// evicted only unleased entries for this policy revision.
    pub(crate) fn acknowledge_retained_reclamation(
        &self,
        revision: u64,
        retained_floors: &BTreeMap<String, u64>,
    ) -> bool {
        let publish_revision = {
            let mut state = self.lock();
            if state.cache_revision != revision {
                return false;
            }
            let mut changed = false;
            for (model, retained_bytes) in retained_floors {
                if let Some(allocation) = state.desired_cache_allocations.get_mut(model) {
                    changed |= allocation.retained_bytes != *retained_bytes;
                    allocation.retained_bytes = *retained_bytes;
                }
            }
            if changed {
                state.cache_revision = state.cache_revision.saturating_add(1);
            }
            state.cache_revision
        };
        self.publish_cache_allocation_revision(publish_revision)
    }

    /// Commit registration and route visibility while the acknowledged cache
    /// revision is still current. The closure must only mutate the router map;
    /// it must never await or call back into the registry.
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
        advance_capacity_policy_revision(&mut state);
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
                replace_shared_ledger(&mut controller, shared, ZeroCapacityRecovery::Preserve);
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
        replace_shared_ledger(&mut candidate, shared, ZeroCapacityRecovery::Preserve);
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
        recompute_registry(&mut state, "model registration");

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
        // Cancel-and-join: interrupt this model's live reservations so their
        // workers stop allocating at the next boundary and the final
        // unregister's wait-for-zero drains promptly instead of waiting for
        // max_tokens.
        for reservation in state.active_reservations.values() {
            if reservation.model == model {
                reservation
                    .stop
                    .stop(higgs_engine::stop::StopReason::ModelDrain);
            }
        }
        let mut removed_waiters = Vec::new();
        state.admission_queue.retain(|waiter| {
            if waiter.model == model {
                removed_waiters.push(Arc::clone(&waiter.notify));
                false
            } else {
                true
            }
        });
        recompute_registry(&mut state, "model drain");
        let wake = state
            .admission_queue
            .front()
            .map(|waiter| Arc::clone(&waiter.notify));
        let registration = DrainRegistration {
            registry: Arc::downgrade(self),
            model: model.to_owned(),
            nonce,
            finished: false,
        };
        drop(state);
        for notify in removed_waiters {
            notify.notify_one();
        }
        if let Some(notify) = wake {
            notify.notify_one();
        }
        Ok(registration)
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
            .any(|reservation| reservation.model == drain.model)
        {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "model still has active request reservations",
            ));
        }
        if let Some(measurement) = memory_after_release.as_ref()
            && (measurement.boot_id != self.boot_id || measurement.revision > state.memory_revision)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unload memory measurement does not belong to this registry revision",
            ));
        }
        let zero_recovery =
            memory_after_release
                .as_ref()
                .map_or(ZeroCapacityRecovery::Preserve, |measurement| {
                    if measurement.authorizes_bounded_recovery(&self.boot_id, &state) {
                        ZeroCapacityRecovery::BoundedMinimum
                    } else {
                        ZeroCapacityRecovery::Preserve
                    }
                });
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
            recompute_registry_with(&mut state, zero_recovery, "model unload");
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
        let mut transitions = Vec::new();
        for (model_name, entry) in state.models.iter_mut() {
            if let Some(active) = entry.active.as_mut() {
                let before = active.controller.decision();
                active.controller.apply_pressure_observation(normalized);
                let after = active.controller.decision();
                if after != before {
                    entry.generation = entry.generation.saturating_add(1);
                    transitions.push((model_name.to_owned(), before, after));
                }
            }
        }
        for (model_name, before, after) in transitions {
            record_capacity_transition(&mut state, &model_name, before, after, "memory pressure");
        }
        state.pressure = effective_pressure;
        state.counters.last_swap_out_delta = observation.swap_out_delta;
        state.counters.last_compressor_delta = observation.compressor_delta;
        if effective_pressure == MemoryPressure::Critical {
            // Interrupt every live reservation: each worker acknowledges at
            // its next allocation boundary and surfaces the typed capacity
            // interruption. Bytes are NOT released here — the worker-owned
            // guard still drops only after allocation has stopped.
            for reservation in state.active_reservations.values() {
                let generation = state
                    .models
                    .get(&reservation.model)
                    .map_or(0, |entry| entry.generation);
                reservation
                    .stop
                    .stop(higgs_engine::stop::StopReason::CriticalPressure {
                        boot_id: self.boot_id.clone(),
                        generation,
                    });
            }
        }
        recompute_registry(&mut state, "memory pressure");
        let wake = state
            .admission_queue
            .front()
            .map(|waiter| Arc::clone(&waiter.notify));
        drop(state);
        if let Some(notify) = wake {
            notify.notify_one();
        }
    }

    pub fn refresh_memory(&self, memory: MlxMemorySnapshot) -> PublishedMemoryMeasurement {
        let mut state = self.lock();
        let previous_revision = state.memory_revision;
        let previous_active_bytes = state.memory.active_bytes;
        state.memory = memory;
        state.memory_revision = state.memory_revision.saturating_add(1);
        let zero_recovery = if previous_revision.checked_add(1) == Some(state.memory_revision)
            && memory.active_bytes < previous_active_bytes
        {
            ZeroCapacityRecovery::BoundedMinimum
        } else {
            ZeroCapacityRecovery::Preserve
        };
        recompute_registry_with(&mut state, zero_recovery, "allocator memory");
        let measurement = PublishedMemoryMeasurement {
            boot_id: self.boot_id.clone(),
            previous_revision,
            revision: state.memory_revision,
            previous_active_bytes,
            active_bytes: memory.active_bytes,
            capacity_policy_revision: state.capacity_policy_revision,
        };
        let wake = state
            .admission_queue
            .front()
            .map(|waiter| Arc::clone(&waiter.notify));
        drop(state);
        if let Some(notify) = wake {
            notify.notify_one();
        }
        measurement
    }

    /// Publish the post-eviction allocator measurement without immediately
    /// recreating the optional cache budgets that admission just reclaimed.
    pub(crate) fn refresh_memory_after_reclamation(
        &self,
        memory: MlxMemorySnapshot,
    ) -> PublishedMemoryMeasurement {
        let mut state = self.lock();
        let previous_revision = state.memory_revision;
        let previous_active_bytes = state.memory.active_bytes;
        state.memory = memory;
        state.memory_revision = state.memory_revision.saturating_add(1);
        let zero_recovery = if previous_revision.checked_add(1) == Some(state.memory_revision)
            && memory.active_bytes < previous_active_bytes
        {
            ZeroCapacityRecovery::BoundedMinimum
        } else {
            ZeroCapacityRecovery::Preserve
        };
        let mut shared = shared_ledger_with(&state, None, None).unwrap_or_default();
        let effective = conservative_cache_allocations(
            &state.desired_cache_allocations,
            &state.published_cache_allocations,
        );
        if apply_allocation_totals(&mut shared, &effective).is_err() {
            shared.active_reservation_bytes = u64::MAX;
        } else {
            // Reservations are checked transactionally by admission and must
            // not shrink the client-visible semantic envelope while live.
            shared.active_reservation_bytes = 0;
        }
        recompute_active_models(
            &mut state,
            shared,
            zero_recovery,
            "cache reclamation remeasure",
        );
        let measurement = PublishedMemoryMeasurement {
            boot_id: self.boot_id.clone(),
            previous_revision,
            revision: state.memory_revision,
            previous_active_bytes,
            active_bytes: memory.active_bytes,
            capacity_policy_revision: state.capacity_policy_revision,
        };
        let wake = state
            .admission_queue
            .front()
            .map(|waiter| Arc::clone(&waiter.notify));
        drop(state);
        if let Some(notify) = wake {
            notify.notify_one();
        }
        measurement
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
            recompute_registry(&mut state, "registration rollback");
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
        advance_capacity_policy_revision(&mut state);
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
            recompute_registry(&mut state, "drain cancelled");
        }
        let wake = state
            .admission_queue
            .front()
            .map(|waiter| Arc::clone(&waiter.notify));
        drop(state);
        if let Some(notify) = wake {
            notify.notify_one();
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

fn shared_ledger_with(
    state: &RegistryState,
    added: Option<&ModelCapacityFacts>,
    removed: Option<&str>,
) -> Result<SharedLedger, RegistrationError> {
    let mut shared = SharedLedger {
        memory: state.memory,
        active_reservation_bytes: active_reservation_bytes(state)?,
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

fn active_reservation_bytes(state: &RegistryState) -> Result<u64, RegistrationError> {
    state
        .active_reservations
        .values()
        .try_fold(0_u64, |sum, reservation| {
            sum.checked_add(reservation.bytes)
                .ok_or(RegistrationError::ArithmeticOverflow)
        })
}

fn effective_shared_ledger(state: &RegistryState) -> Result<SharedLedger, RegistrationError> {
    let mut shared = shared_ledger_with(state, None, None)?;
    let effective = conservative_cache_allocations(
        &state.desired_cache_allocations,
        &state.published_cache_allocations,
    );
    apply_allocation_totals(&mut shared, &effective)?;
    Ok(shared)
}

fn request_reservation_bytes(ledger: super::ByteLedger, retained_growth_bytes: u64) -> Option<u64> {
    ledger
        .fixed_session_bytes
        .checked_add(ledger.prompt_bytes)?
        .checked_add(ledger.output_bytes)?
        .checked_add(ledger.decode_bytes)?
        .checked_add(retained_growth_bytes)?
        .checked_add(ledger.learned_retained_bytes)?
        .checked_add(ledger.transient_bytes)
}

fn admission_error(boot_id: &str, generation: u64, admission: Admission) -> CapacityAdmissionError {
    match admission {
        Admission::Exceeded(decision) => {
            CapacityAdmissionError::Exceeded(CapacityExceededError::new(
                decision.max_prompt_tokens,
                decision.safe_total_tokens,
                boot_id.to_owned(),
                generation,
            ))
        }
        Admission::FixedCostUnavailable | Admission::Unavailable | Admission::Admitted(_) => {
            CapacityAdmissionError::Unavailable(CapacityUnavailableError::new(
                boot_id.to_owned(),
                generation,
            ))
        }
    }
}

fn replace_shared_ledger(
    controller: &mut CapacityController,
    shared: SharedLedger,
    zero_recovery: ZeroCapacityRecovery,
) {
    controller.replace_shared_residency(
        shared.memory,
        shared.loaded_model_bytes,
        shared.retained_bytes,
        shared.prefix_cache_bytes,
        shared.active_reservation_bytes,
        zero_recovery,
    );
}

fn recompute_active_models(
    state: &mut RegistryState,
    shared: SharedLedger,
    zero_recovery: ZeroCapacityRecovery,
    cause: &'static str,
) {
    advance_capacity_policy_revision(state);
    let zero_recovery = if state.capacity_policy_revision.is_some() {
        zero_recovery
    } else {
        ZeroCapacityRecovery::Preserve
    };
    let mut transitions = Vec::new();
    for (model_name, entry) in state.models.iter_mut() {
        if let Some(active) = entry.active.as_mut() {
            let before = active.controller.decision();
            replace_shared_ledger(&mut active.controller, shared, zero_recovery);
            let after = active.controller.decision();
            if after != before {
                entry.generation = entry.generation.saturating_add(1);
                transitions.push((model_name.to_owned(), before, after));
            }
        }
    }
    for (model_name, before, after) in transitions {
        record_capacity_transition(state, &model_name, before, after, cause);
    }
}

fn advance_capacity_policy_revision(state: &mut RegistryState) {
    state.capacity_policy_revision = state
        .capacity_policy_revision
        .and_then(|revision| revision.checked_add(1));
}

fn recompute_registry(state: &mut RegistryState, cause: &'static str) {
    recompute_registry_with(state, ZeroCapacityRecovery::Preserve, cause);
}

fn restore_cache_policy_if_admission_idle(state: &mut RegistryState) {
    if state.active_reservations.is_empty() && state.admission_queue.is_empty() {
        // Reclaimed caches stay capped through the whole FIFO handoff. The
        // ordinary acknowledged policy may restore them only after the final
        // worker and waiter are both gone.
        recompute_registry(state, "admission idle restore");
    }
}

fn recompute_registry_with(
    state: &mut RegistryState,
    zero_recovery: ZeroCapacityRecovery,
    cause: &'static str,
) {
    let old_allocations = std::mem::take(&mut state.desired_cache_allocations);
    let allocations = cache_allocations_with(state, None);
    let mut desired = allocations.as_ref().cloned().unwrap_or_default();
    if !state.active_reservations.is_empty() || !state.admission_queue.is_empty() {
        // A request was admitted against the current cache ceiling — or is
        // queued to be. Pressure and memory observations may lower that
        // ceiling, but cannot restore reclaimed bytes until the FIFO is fully
        // idle: a recompute in the release→acquire handoff window (no active
        // reservations, waiters still queued) must not enlarge the ceiling a
        // queued waiter was admitted against.
        for (model, allocation) in &mut desired {
            let ceiling = old_allocations.get(model).copied().unwrap_or_default();
            allocation.retained_bytes = allocation.retained_bytes.min(ceiling.retained_bytes);
            allocation.prefix_bytes = allocation.prefix_bytes.min(ceiling.prefix_bytes);
        }
    }
    state.desired_cache_allocations = desired;
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
    let ledger_valid =
        allocations.is_ok() && apply_allocation_totals(&mut shared, &effective).is_ok();
    if !ledger_valid {
        shared.active_reservation_bytes = u64::MAX;
    }
    // Published limits describe semantic request shape. Live reservations are
    // a contention term checked by admission and must queue, not tell clients
    // to compact an otherwise safe request.
    if ledger_valid {
        shared.active_reservation_bytes = 0;
    }
    recompute_active_models(state, shared, zero_recovery, cause);
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
    let requested_total = requested_retained
        .values()
        .chain(requested_prefix.values())
        .try_fold(0_u64, |sum, requested| {
            sum.checked_add(*requested)
                .ok_or(RegistrationError::ArithmeticOverflow)
        })?;
    let mut low = 0_u64;
    let mut high = cache_envelope
        .saturating_sub(frozen_total)
        .min(requested_total);
    if !minimum_requests_fit_with_cache_bytes(state, added, frozen_total)? {
        if let Some(facts) = added {
            return Err(RegistrationError::InsufficientCapacity(facts.model.clone()));
        }
        high = 0;
    }
    while low < high {
        let midpoint = low + (high - low).div_ceil(2);
        let cache_bytes = frozen_total
            .checked_add(midpoint)
            .ok_or(RegistrationError::ArithmeticOverflow)?;
        if minimum_requests_fit_with_cache_bytes(state, added, cache_bytes)? {
            low = midpoint;
        } else {
            high = midpoint - 1;
        }
    }
    // Optional transport caches consume only the bytes left after every live
    // model can still serve its minimum semantic request. This uses the same
    // controller as publication/admission instead of a parallel cache rule.
    let available_envelope = low;
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

fn minimum_requests_fit_with_cache_bytes(
    state: &RegistryState,
    added: Option<&ModelCapacityFacts>,
    cache_bytes: u64,
) -> Result<bool, RegistrationError> {
    let mut shared = shared_ledger_with(state, added, None)?;
    shared.retained_bytes = cache_bytes;
    shared.prefix_cache_bytes = 0;
    // Live reservations are a contention term, not part of the published
    // semantic envelope whose minimum the cache plan must preserve.
    shared.active_reservation_bytes = 0;
    let fits = |facts: &ModelCapacityFacts| {
        facts
            .controller(shared, state.pressure)
            .decision()
            .availability
            == CapacityAvailability::Available
    };
    Ok(added.is_none_or(&fits)
        && state.models.values().all(|entry| {
            entry
                .active
                .as_ref()
                .is_none_or(|active| active.draining || fits(&active.facts))
        }))
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

    fn admission_facts(name: &str) -> ModelCapacityFacts {
        let mut facts = facts(name, 2 * GIB);
        facts.memory = MlxMemorySnapshot {
            active_bytes: 2 * GIB,
            peak_bytes: 2 * GIB,
            memory_limit_bytes: Some(10 * GIB),
            metal_recommended_working_set_bytes: Some(10 * GIB),
        };
        facts.costs = EngineCostDescription {
            fixed_live_session_bytes: 0,
            persistent_bytes_per_token: 1024 * 1024,
            decode_workspace_bytes: 0,
            transient_prefill: TransientPrefillEstimate {
                base_bytes: 0,
                bytes_per_prompt_token: 0,
                bytes_per_chunk_token: 0,
                max_prompt_tokens: 8_192,
                max_chunk_tokens: 1_024,
            },
        };
        facts.architectural_max_tokens = 8_192;
        facts.retained_bytes_ceiling = 0;
        facts.prefix_cache_bytes_ceiling = 0;
        facts.configured_total_token_ceiling = Some(4_096);
        facts.configured_output_token_ceiling = Some(2_048);
        facts
    }

    const fn admission_request(prompt_tokens: u64, output_tokens: u64) -> super::RequestCost {
        super::RequestCost {
            execution_path: crate::capacity::ExecutionPath::Cold,
            prompt_tokens,
            suffix_tokens: prompt_tokens,
            output_tokens,
            retained_growth_bytes: 0,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_requested_output_is_reserved_exactly() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        let mut facts = admission_facts("model");
        facts.memory.memory_limit_bytes = Some(12 * GIB);
        facts.memory.metal_recommended_working_set_bytes = Some(12 * GIB);
        register(&registry, facts);

        let zero_output = registry
            .reserve_request("model", admission_request(1_024, 0))
            .await
            .unwrap();
        assert_eq!(
            zero_output.bytes(),
            GIB,
            "zero requested output must reserve zero output bytes"
        );
        drop(zero_output);

        let requested_output = registry
            .reserve_request("model", admission_request(1_024, 2_048))
            .await
            .unwrap();
        assert_eq!(requested_output.bytes(), 3 * GIB);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_two_requests_cannot_reserve_the_same_bytes() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        register(&registry, admission_facts("model"));
        let first = registry
            .reserve_request("model", admission_request(1_024, 1_024))
            .await
            .unwrap();
        assert_eq!(registry.active_reservation_count("model"), 1);
        assert_eq!(registry.active_reservation_bytes(), 2 * GIB);

        let waiting_registry = Arc::clone(&registry);
        let (acquired, mut acquired_rx) = tokio::sync::mpsc::unbounded_channel();
        let waiter = tokio::spawn(async move {
            let guard = waiting_registry
                .reserve_request("model", admission_request(1_024, 1_024))
                .await
                .unwrap();
            acquired.send(guard).unwrap();
        });
        while registry.queued_waiter_count() != 1 {
            tokio::task::yield_now().await;
        }
        assert!(acquired_rx.try_recv().is_err());
        assert_eq!(registry.active_reservation_count("model"), 1);

        drop(first);
        let second = acquired_rx.recv().await.unwrap();
        assert_eq!(registry.active_reservation_count("model"), 1);
        drop(second);
        waiter.await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_contention_is_strict_fifo() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        register(&registry, admission_facts("model"));
        let first = registry
            .reserve_request("model", admission_request(1_024, 1_024))
            .await
            .unwrap();
        let (order_tx, mut order_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut releases = Vec::new();
        let mut tasks = Vec::new();
        for order in [2_u8, 3] {
            let waiting_registry = Arc::clone(&registry);
            let order_tx = order_tx.clone();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            releases.push(release_tx);
            tasks.push(tokio::spawn(async move {
                let guard = waiting_registry
                    .reserve_request("model", admission_request(1_024, 1_024))
                    .await
                    .unwrap();
                order_tx.send(order).unwrap();
                let _ = release_rx.await;
                drop(guard);
            }));
            while registry.queued_waiter_count() != usize::from(order - 1) {
                tokio::task::yield_now().await;
            }
        }

        drop(first);
        assert_eq!(order_rx.recv().await, Some(2));
        assert!(order_rx.try_recv().is_err());
        releases.remove(0).send(()).unwrap();
        assert_eq!(order_rx.recv().await, Some(3));
        releases.remove(0).send(()).unwrap();
        for task in tasks {
            task.await.unwrap();
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_dropping_queued_waiter_removes_it_and_wakes_progress() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        register(&registry, admission_facts("model"));
        let first = registry
            .reserve_request("model", admission_request(1_024, 1_024))
            .await
            .unwrap();
        let waiting_registry = Arc::clone(&registry);
        let cancelled = tokio::spawn(async move {
            waiting_registry
                .reserve_request("model", admission_request(1_024, 1_024))
                .await
        });
        while registry.queued_waiter_count() != 1 {
            tokio::task::yield_now().await;
        }
        let follower_registry = Arc::clone(&registry);
        let (acquired_tx, mut acquired_rx) = tokio::sync::mpsc::unbounded_channel();
        let follower = tokio::spawn(async move {
            let guard = follower_registry
                .reserve_request("model", admission_request(1_024, 1_024))
                .await
                .unwrap();
            acquired_tx.send(guard).unwrap();
        });
        while registry.queued_waiter_count() != 2 {
            tokio::task::yield_now().await;
        }
        cancelled.abort();
        while registry.queued_waiter_count() != 1 {
            tokio::task::yield_now().await;
        }

        drop(first);
        let replacement = acquired_rx.recv().await.unwrap();
        assert_eq!(registry.active_reservation_count("model"), 1);
        drop(replacement);
        follower.await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_diagnostics_expose_reservation_waiter_rejection_and_outcome_counters() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        register(&registry, admission_facts("model"));
        // 2 GiB per request against the byte-bound envelope: one fits, a
        // second contends in the FIFO, and a later allocator rise lowers the
        // published envelope.
        let base = admission_request(1_024, 1_024);

        // Semantic oversize is counted as a typed rejection.
        assert!(matches!(
            registry.try_reserve_request("model", admission_request(8_192, 8_192)),
            RequestReservationAttempt::Rejected(_)
        ));
        let diagnostics = registry.diagnostics();
        assert_eq!(diagnostics.rejections.exceeded, 1);

        // One live reservation with bytes and an oldest-age sample.
        let reservation = registry.reserve_request("model", base).await.unwrap();
        let diagnostics = registry.diagnostics();
        assert_eq!(diagnostics.active_reservations, 1);
        assert_eq!(diagnostics.active_reservation_bytes, 2 * GIB);
        assert!(diagnostics.oldest_reservation_age_ms.is_some());
        assert_eq!(diagnostics.queued_waiters, 0);

        // A contended follower is visible as a queued waiter.
        let waiting_registry = Arc::clone(&registry);
        let waiter =
            tokio::spawn(async move { waiting_registry.reserve_request("model", base).await });
        while registry.diagnostics().queued_waiters != 1 {
            assert!(!waiter.is_finished(), "follower did not enter FIFO");
            tokio::task::yield_now().await;
        }

        // Pressure observations expose the effective level; nonzero
        // swap/compressor deltas escalate the level, so the level assert and
        // the delta assert use separate observations.
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(registry.diagnostics().pressure, MemoryPressure::Constrained);

        // An envelope reduction from a measured allocator change is counted
        // as a downshift.
        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 7 * GIB / 2,
            peak_bytes: 7 * GIB / 2,
            memory_limit_bytes: Some(10 * GIB),
            metal_recommended_working_set_bytes: Some(10 * GIB),
        });
        let diagnostics = registry.diagnostics();
        assert!(
            diagnostics.downshifts >= 1,
            "a measured envelope reduction must be counted: {diagnostics:#?}"
        );

        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 42,
            compressor_delta: 7,
        });
        let diagnostics = registry.diagnostics();
        assert_eq!(diagnostics.swap_out_delta, 42);
        assert_eq!(diagnostics.compressor_delta, 7);

        // A stopped worker releases with its outcome counted. The critical
        // escalation from the delta observation marked the live reservation
        // first (first reason wins), so the recorded outcome is the pressure
        // interrupt — a later ModelDrain signal cannot rewrite it.
        drop(reservation);
        let diagnostics = registry.diagnostics();
        assert_eq!(diagnostics.active_reservations, 0);
        assert_eq!(diagnostics.stop_outcomes.get("critical_pressure"), Some(&1));
        assert_eq!(
            diagnostics.stop_outcomes.values().sum::<u64>(),
            1,
            "exactly one released-with-reason outcome: {:?}",
            diagnostics.stop_outcomes
        );

        // Unavailability rejections are counted separately.
        assert!(matches!(
            registry.try_reserve_request("missing-model", base),
            RequestReservationAttempt::Rejected(_)
        ));
        assert!(
            registry.diagnostics().rejections.unavailable >= 1,
            "{:?}",
            registry.diagnostics().rejections
        );
        let drain = registry.begin_drain("model").unwrap();
        let _ = drain;
        let _ = waiter.await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_critical_pressure_interrupts_live_reservations() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        register(&registry, admission_facts("model"));
        let reservation = registry
            .reserve_request("model", admission_request(1_024, 1_024))
            .await
            .unwrap();
        assert_eq!(reservation.stop().reason(), None);

        let generation_before = registry.snapshot("model").unwrap().generation;
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Critical,
            swap_out_delta: 0,
            compressor_delta: 0,
        });

        // The interrupt carries this boot's identity and the post-transition
        // generation nanobot will retry against.
        assert!(
            matches!(
                reservation.stop().reason(),
                Some(higgs_engine::stop::StopReason::CriticalPressure {
                    boot_id,
                    generation,
                }) if *boot_id == registry.boot_id().to_owned()
                    && generation >= generation_before
            ),
            "unexpected interrupt payload: {:?}",
            reservation.stop().reason()
        );
        // Bytes are still held: the worker releases its guard only after it
        // acknowledges at an allocation boundary.
        assert_eq!(registry.active_reservation_count("model"), 1);
        drop(reservation);
        assert_eq!(registry.active_reservation_count("model"), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_drain_cancels_and_joins_active_reservations() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        register(&registry, admission_facts("model"));
        let reservation = registry
            .reserve_request("model", admission_request(1_024, 1_024))
            .await
            .unwrap();
        assert_eq!(reservation.stop().reason(), None);

        let drain = registry.begin_drain("model").unwrap();
        assert_eq!(
            reservation.stop().reason(),
            Some(higgs_engine::stop::StopReason::ModelDrain)
        );
        drop(reservation);
        // With the interrupted worker released, the drain can complete.
        registry
            .finish_unregister(drain, None)
            .expect("drain finishes after worker release");
        assert_eq!(registry.active_reservation_count("model"), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_dequeue_revalidates_pressure_generation_memory_and_cost() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        register(&registry, admission_facts("model"));
        let initial = registry.snapshot("model").unwrap();
        let first = registry
            .reserve_request("model", admission_request(1_024, 1_024))
            .await
            .unwrap();
        let waiting_registry = Arc::clone(&registry);
        let waiter = tokio::spawn(async move {
            waiting_registry
                .reserve_request("model", admission_request(1_024, 1_024))
                .await
        });
        while registry.queued_waiter_count() != 1 {
            tokio::task::yield_now().await;
        }

        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 2 * GIB,
            peak_bytes: 2 * GIB,
            memory_limit_bytes: Some(10 * GIB),
            metal_recommended_working_set_bytes: Some(10 * GIB),
        });
        drop(first);
        let error = waiter.await.unwrap().unwrap_err();
        let current = registry.snapshot("model").unwrap();
        assert_eq!(current.pressure, MemoryPressure::Constrained);
        assert_eq!(error.boot_id(), initial.boot_id);
        assert_eq!(error.generation(), current.generation);
        assert!(current.generation > initial.generation);
        assert_eq!(registry.queued_waiter_count(), 0);
        assert_eq!(registry.active_reservation_count("model"), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_cache_reclamation_is_not_counted_before_acknowledgement() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        let mut model_facts = admission_facts("model");
        model_facts.loaded_model_bytes = GIB;
        model_facts.memory.active_bytes = GIB;
        model_facts.memory.peak_bytes = GIB;
        model_facts.prefix_cache_bytes_ceiling = 2 * GIB;
        register(&registry, model_facts);
        let request = admission_request(2_048, 2_048);
        assert!(matches!(
            registry.try_reserve_request("model", request),
            RequestReservationAttempt::Rejected(_)
        ));

        assert!(registry.request_cache_reclamation(CacheReclamation::Prefix));
        assert!(matches!(
            registry.try_reserve_request("model", request),
            RequestReservationAttempt::Rejected(_)
        ));
        assert_eq!(
            effective_shared_ledger(&registry.lock())
                .unwrap()
                .prefix_cache_bytes,
            2 * GIB
        );
        let plan = registry.cache_allocation_plan();
        assert_eq!(plan.allocations[0].2, 0);
        assert!(registry.publish_cache_allocation_revision(plan.revision));
        assert_eq!(
            effective_shared_ledger(&registry.lock())
                .unwrap()
                .prefix_cache_bytes,
            0
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_reclaimed_cache_stays_capped_until_final_guard_release() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        let mut model_facts = admission_facts("model");
        model_facts.loaded_model_bytes = GIB;
        model_facts.memory.active_bytes = GIB;
        model_facts.memory.peak_bytes = GIB;
        model_facts.prefix_cache_bytes_ceiling = 2 * GIB;
        register(&registry, model_facts);
        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 2 * GIB,
            peak_bytes: 2 * GIB,
            memory_limit_bytes: Some(10 * GIB),
            metal_recommended_working_set_bytes: Some(10 * GIB),
        });
        let mut request = admission_request(1_024, 1_024);
        request.retained_growth_bytes = GIB;

        assert!(registry.request_cache_reclamation(CacheReclamation::Prefix));
        let reclaimed = registry.cache_allocation_plan();
        assert!(registry.publish_cache_allocation_revision(reclaimed.revision));
        let guard = registry.reserve_request("model", request).await.unwrap();

        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(
            registry.cache_allocation_plan().allocations[0].2,
            0,
            "pressure publication cannot restore cache while reserved bytes are live"
        );
        assert!(matches!(
            registry.try_reserve_request("model", request),
            RequestReservationAttempt::Contended
        ));

        let waiting_registry = Arc::clone(&registry);
        let follower =
            tokio::spawn(async move { waiting_registry.reserve_request("model", request).await });
        while registry.queued_waiter_count() != 1 {
            assert!(!follower.is_finished(), "follower did not enter FIFO");
            tokio::task::yield_now().await;
        }
        drop(guard);
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(
            registry.cache_allocation_plan().allocations[0].2,
            0,
            "a queued-only FIFO handoff window must keep the reclaimed ceiling"
        );
        let follower_guard = follower.await.unwrap().unwrap();
        assert_eq!(
            registry.cache_allocation_plan().allocations[0].2,
            0,
            "the FIFO handoff retains the reclaimed ceiling"
        );
        drop(follower_guard);
        assert!(
            registry.cache_allocation_plan().allocations[0].2 > 0,
            "the ordinary ACK path may propose restoration after the final guard"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_queued_handoff_keeps_reclaimed_ceiling_through_recompute() {
        // Same reclaimed-ceiling state as the guard-release test above, but
        // the pressure publication lands in the window after the leader's
        // release and before the queued follower acquires: only waiters, no
        // active reservations. The recompute must not restore reclaimed
        // caches in that window.
        let registry = CapacityRegistry::new(["model".to_owned()]);
        let mut model_facts = admission_facts("model");
        model_facts.loaded_model_bytes = GIB;
        model_facts.memory.active_bytes = GIB;
        model_facts.memory.peak_bytes = GIB;
        model_facts.prefix_cache_bytes_ceiling = 2 * GIB;
        register(&registry, model_facts);
        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 2 * GIB,
            peak_bytes: 2 * GIB,
            memory_limit_bytes: Some(10 * GIB),
            metal_recommended_working_set_bytes: Some(10 * GIB),
        });
        let mut request = admission_request(1_024, 1_024);
        request.retained_growth_bytes = GIB;

        assert!(registry.request_cache_reclamation(CacheReclamation::Prefix));
        let reclaimed = registry.cache_allocation_plan();
        assert!(registry.publish_cache_allocation_revision(reclaimed.revision));
        let guard = registry.reserve_request("model", request).await.unwrap();

        let waiting_registry = Arc::clone(&registry);
        let follower =
            tokio::spawn(async move { waiting_registry.reserve_request("model", request).await });
        while registry.queued_waiter_count() != 1 {
            assert!(!follower.is_finished(), "follower did not enter FIFO");
            tokio::task::yield_now().await;
        }
        // Release the leader, then synchronously publish a restorative
        // observation before the follower task runs: `active_reservations`
        // is empty but the FIFO still holds the queued follower.
        drop(guard);
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(
            registry.cache_allocation_plan().allocations[0].2,
            0,
            "queued-only handoff keeps the reclaimed ceiling across recompute"
        );
        let follower_guard = follower.await.unwrap().unwrap();
        drop(follower_guard);
        assert!(
            registry.cache_allocation_plan().allocations[0].2 > 0,
            "restoration resumes only after the final waiter drains"
        );
    }

    #[test]
    fn capacity_admission_reclamation_preserves_detached_draining_cache_floor() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        let mut model_facts = admission_facts("model");
        model_facts.memory.memory_limit_bytes = Some(16 * GIB);
        model_facts.memory.metal_recommended_working_set_bytes = Some(16 * GIB);
        model_facts.retained_bytes_ceiling = 2 * GIB;
        model_facts.prefix_cache_bytes_ceiling = 2 * GIB;
        register(&registry, model_facts);
        let before = registry.cache_allocation_plan().allocations[0].clone();
        let drain = registry.begin_drain("model").unwrap();

        assert!(!registry.request_cache_reclamation(CacheReclamation::Prefix));
        assert!(!registry.request_cache_reclamation(CacheReclamation::Retained));
        assert_eq!(registry.cache_allocation_plan().allocations[0], before);
        drop(drain);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_guard_releases_on_success_error_and_unwind() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        register(&registry, admission_facts("model"));

        for outcome in ["success", "error", "unwind"] {
            let guard = registry
                .reserve_request("model", admission_request(1_024, 1_024))
                .await
                .unwrap();
            let worker_registry = Arc::clone(&registry);
            let joined = tokio::task::spawn_blocking(move || {
                let _guard = guard;
                assert_eq!(worker_registry.active_reservation_count("model"), 1);
                match outcome {
                    "success" => Ok::<(), &'static str>(()),
                    "error" => Err("engine error"),
                    "unwind" => panic!("engine unwind"),
                    _ => unreachable!(),
                }
            })
            .await;
            match outcome {
                "success" => assert!(joined.unwrap().is_ok()),
                "error" => assert!(joined.unwrap().is_err()),
                "unwind" => assert!(joined.is_err()),
                _ => unreachable!(),
            }
            assert_eq!(registry.active_reservation_count("model"), 0);
            assert_eq!(registry.active_reservation_bytes(), 0);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_drain_waits_for_worker_owned_reservations() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        register(&registry, admission_facts("model"));
        let guard = registry
            .reserve_request("model", admission_request(1_024, 1_024))
            .await
            .unwrap();
        let drain = registry.begin_drain("model").unwrap();
        let waiting_registry = Arc::clone(&registry);
        let (drained_tx, mut drained_rx) = tokio::sync::mpsc::unbounded_channel();
        let wait = tokio::spawn(async move {
            waiting_registry
                .wait_for_model_reservations(drain.model())
                .await;
            drained_tx.send(drain).unwrap();
        });
        assert!(drained_rx.try_recv().is_err());
        drop(guard);
        let drain = drained_rx.recv().await.unwrap();
        registry.finish_unregister(drain, None).unwrap();
        wait.await.unwrap();
        assert_eq!(
            registry.snapshot("model").unwrap().availability,
            CapacityAvailability::Unavailable
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_admission_drain_rejects_queued_work_with_current_generation() {
        let registry = CapacityRegistry::new(["model".to_owned()]);
        register(&registry, admission_facts("model"));
        let first = registry
            .reserve_request("model", admission_request(1_024, 1_024))
            .await
            .unwrap();
        let waiting_registry = Arc::clone(&registry);
        let waiter = tokio::spawn(async move {
            waiting_registry
                .reserve_request("model", admission_request(1_024, 1_024))
                .await
        });
        while registry.queued_waiter_count() != 1 {
            tokio::task::yield_now().await;
        }
        let drain = registry.begin_drain("model").unwrap();
        let generation = registry.snapshot("model").unwrap().generation;
        let error = waiter.await.unwrap().unwrap_err();
        assert!(matches!(error, CapacityAdmissionError::Unavailable(_)));
        assert_eq!(error.generation(), generation);
        assert_eq!(registry.queued_waiter_count(), 0);
        drop(first);
        drop(drain);
    }

    /// Omitting current MLX residency from load headroom admits a second model
    /// against bytes already owned by the first.
    #[test]
    fn load_snapshot_subtracts_active_residency_from_safe_envelope() {
        let registry = CapacityRegistry::new(std::iter::empty());
        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 5 * GIB,
            peak_bytes: 5 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(30 * GIB),
        });
        let snapshot = registry.load_snapshot().unwrap();
        assert_eq!(snapshot.pressure, MemoryPressure::Normal);
        assert_eq!(snapshot.headroom_bytes, 24 * GIB - (24 * GIB / 5) - 5 * GIB);
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
    fn constrained_zero_is_not_reseeded_by_shared_ledger_recompute() {
        let registry = CapacityRegistry::new(["first".to_owned()]);
        let mut first = facts("first", 5 * GIB);
        first.configured_total_token_ceiling = Some(1_024);
        first.configured_output_token_ceiling = Some(1_024);
        register(&registry, first);
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 1_024);

        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 1,
        });

        assert_eq!(
            registry.snapshot("first").unwrap().safe_total_tokens,
            0,
            "the shared-ledger pass must preserve the pressure controller's zero downshift"
        );
    }

    #[test]
    fn cancelled_provisional_needs_later_measured_decrease_to_recover_zero() {
        for pressure in [MemoryPressure::Constrained, MemoryPressure::Critical] {
            let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
            let mut first = facts("first", 5 * GIB);
            first.configured_total_token_ceiling = Some(1_024);
            first.configured_output_token_ceiling = Some(1_024);
            register(&registry, first);

            let ticket = registry.begin_registration("second".to_owned()).unwrap();
            let provisional = registry
                .commit_active(ticket, facts("second", 5 * GIB))
                .unwrap();
            registry.apply_pressure_observation(PressureObservation {
                pressure,
                swap_out_delta: u64::from(pressure == MemoryPressure::Critical),
                compressor_delta: 1,
            });
            assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);

            drop(provisional);
            assert_eq!(
                registry.snapshot("first").unwrap().safe_total_tokens,
                0,
                "rolling back metadata before engine cleanup is not recovery evidence"
            );

            let expected_after_decrease = if pressure == MemoryPressure::Critical {
                registry.apply_pressure_observation(PressureObservation {
                    pressure: MemoryPressure::Normal,
                    swap_out_delta: 0,
                    compressor_delta: 0,
                });
                0
            } else {
                1_024
            };
            registry.refresh_memory(MlxMemorySnapshot {
                active_bytes: 4 * GIB,
                peak_bytes: 5 * GIB,
                memory_limit_bytes: Some(24 * GIB),
                metal_recommended_working_set_bytes: Some(24 * GIB),
            });
            assert_eq!(
                registry.snapshot("first").unwrap().safe_total_tokens,
                expected_after_decrease,
                "a later authoritative decrease may seed only when pressure policy permits it"
            );
        }
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
    fn registration_reduces_optional_caches_to_preserve_minimum_request() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let mut model = facts("escha", 12 * GIB);
        model.costs.transient_prefill.base_bytes = 4 * GIB;
        model.configured_total_token_ceiling = Some(1_024);
        model.configured_output_token_ceiling = Some(1_024);
        registry.refresh_memory(model.memory);
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        });

        let ticket = registry.begin_registration(model.model.clone()).unwrap();
        registry.commit_active(ticket, model).unwrap().publish();
        let plan = registry.cache_allocation_plan();
        let allocated_cache = plan.allocations[0]
            .1
            .checked_add(plan.allocations[0].2)
            .unwrap();
        assert!(registry.publish_cache_allocation_revision(plan.revision));
        let snapshot = registry.snapshot("escha").unwrap();

        assert_eq!(snapshot.availability, CapacityAvailability::Available);
        assert_eq!(snapshot.safe_total_tokens, 1_024);
        assert!(allocated_cache > 0);
        assert!(allocated_cache < 3 * GIB);
    }

    #[test]
    fn minimum_request_fit_is_monotonic_in_total_cache_bytes() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let mut model = facts("escha", 12 * GIB);
        model.costs.transient_prefill.base_bytes = 4 * GIB;
        model.configured_total_token_ceiling = Some(1_024);
        model.configured_output_token_ceiling = Some(1_024);
        registry.refresh_memory(model.memory);
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        });

        assert!(minimum_requests_fit_with_cache_bytes(&registry.lock(), Some(&model), 0).unwrap());
        assert!(
            minimum_requests_fit_with_cache_bytes(
                &registry.lock(),
                Some(&model),
                128 * 1024 * 1024,
            )
            .unwrap()
        );
        assert!(
            !minimum_requests_fit_with_cache_bytes(&registry.lock(), Some(&model), 3 * GIB,)
                .unwrap()
        );
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

    fn draining_registry_with_zero_survivor() -> (Arc<CapacityRegistry>, DrainRegistration) {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        let mut first = facts("first", 5 * GIB);
        first.configured_total_token_ceiling = Some(1_024);
        first.configured_output_token_ceiling = Some(1_024);
        register(&registry, first);
        register(&registry, facts("second", 5 * GIB));
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 1,
        });
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);
        let drain = registry.begin_drain("second").unwrap();
        (registry, drain)
    }

    #[test]
    fn finish_unregister_requires_current_decreased_measurement_to_recover_zero() {
        let (registry, drain) = draining_registry_with_zero_survivor();
        registry.finish_unregister(drain, None).unwrap();
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);

        for active_bytes in [5 * GIB, 6 * GIB] {
            let (registry, drain) = draining_registry_with_zero_survivor();
            let measurement = registry.refresh_memory(MlxMemorySnapshot {
                active_bytes,
                peak_bytes: active_bytes,
                memory_limit_bytes: Some(24 * GIB),
                metal_recommended_working_set_bytes: Some(24 * GIB),
            });
            registry
                .finish_unregister(drain, Some(measurement))
                .unwrap();
            assert_eq!(
                registry.snapshot("first").unwrap().safe_total_tokens,
                0,
                "an equal or increased measurement is not recovery evidence"
            );
        }

        let (registry, drain) = draining_registry_with_zero_survivor();
        let decreased = registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 4 * GIB,
            peak_bytes: 5 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        registry.finish_unregister(drain, Some(decreased)).unwrap();
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 1_024);

        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        register(&registry, facts("first", 5 * GIB));
        register(&registry, facts("second", 5 * GIB));
        let full = registry.snapshot("first").unwrap().safe_total_tokens;
        assert!(full > 8_192);
        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 20 * GIB,
            peak_bytes: 20 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);
        let drain = registry.begin_drain("second").unwrap();
        let decreased = registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 5 * GIB,
            peak_bytes: 20 * GIB,
            memory_limit_bytes: Some(14 * GIB),
            metal_recommended_working_set_bytes: Some(14 * GIB),
        });
        assert_eq!(
            registry.snapshot("first").unwrap().safe_total_tokens,
            0,
            "the measured decrease alone has no headroom while the draining model is reserved"
        );
        registry.finish_unregister(drain, Some(decreased)).unwrap();
        let recovered = registry.snapshot("first").unwrap().safe_total_tokens;
        assert_eq!(recovered, 8_192);
        assert!(recovered < full, "measured recovery must not jump to full");
    }

    #[test]
    fn stale_decreased_measurement_cannot_authorize_unregister_recovery() {
        let (registry, drain) = draining_registry_with_zero_survivor();
        let stale_decrease = registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 4 * GIB,
            peak_bytes: 5 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 1,
        });
        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 5 * GIB,
            peak_bytes: 5 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        registry
            .finish_unregister(drain, Some(stale_decrease))
            .unwrap();
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);
    }

    #[test]
    fn pressure_after_measurement_invalidates_unregister_recovery() {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        let mut first = facts("first", 5 * GIB);
        first.configured_total_token_ceiling = Some(1_024);
        first.configured_output_token_ceiling = Some(1_024);
        register(&registry, first);
        register(&registry, facts("second", 5 * GIB));
        let drain = registry.begin_drain("second").unwrap();
        let decreased = registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 4 * GIB,
            peak_bytes: 5 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });

        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 1,
        });
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);
        registry.finish_unregister(drain, Some(decreased)).unwrap();
        assert_eq!(
            registry.snapshot("first").unwrap().safe_total_tokens,
            0,
            "a later pressure policy must invalidate earlier recovery evidence"
        );
    }

    fn draining_registry_with_bounded_recovery_token() -> (
        Arc<CapacityRegistry>,
        DrainRegistration,
        PublishedMemoryMeasurement,
    ) {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        register(&registry, facts("first", 5 * GIB));
        register(&registry, facts("second", 5 * GIB));
        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 20 * GIB,
            peak_bytes: 20 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);
        let drain = registry.begin_drain("second").unwrap();
        let decreased = registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 5 * GIB,
            peak_bytes: 20 * GIB,
            memory_limit_bytes: Some(14 * GIB),
            metal_recommended_working_set_bytes: Some(14 * GIB),
        });
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);
        (registry, drain, decreased)
    }

    #[test]
    fn cache_publication_invalidates_unregister_recovery() {
        let (registry, drain, decreased) = draining_registry_with_bounded_recovery_token();
        let plan = registry.cache_allocation_plan();
        assert!(registry.publish_cache_allocation_revision(plan.revision));

        registry.finish_unregister(drain, Some(decreased)).unwrap();
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);
    }

    #[test]
    fn normal_noop_policy_recompute_conservatively_invalidates_recovery() {
        let (registry, drain, decreased) = draining_registry_with_bounded_recovery_token();
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });

        registry.finish_unregister(drain, Some(decreased)).unwrap();
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);
    }

    #[test]
    fn policy_revision_overflow_permanently_disables_recovery_authority() {
        let (registry, drain, _) = draining_registry_with_bounded_recovery_token();
        registry.lock().capacity_policy_revision = Some(u64::MAX);
        let decreased = registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 4 * GIB,
            peak_bytes: 20 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        assert_eq!(registry.lock().capacity_policy_revision, None);

        registry.finish_unregister(drain, Some(decreased)).unwrap();
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);
        registry.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(registry.lock().capacity_policy_revision, None);
    }

    #[test]
    fn rejected_coload_cleanup_seeds_bounded_recovery_from_zero() {
        let registry = CapacityRegistry::new(["first".to_owned(), "second".to_owned()]);
        register(&registry, facts("first", 5 * GIB));
        let full = registry.snapshot("first").unwrap().safe_total_tokens;
        assert!(full > 8_192);

        let ticket = registry.begin_registration("second".to_owned()).unwrap();
        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 20 * GIB,
            peak_bytes: 20 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        assert_eq!(registry.snapshot("first").unwrap().safe_total_tokens, 0);
        assert!(matches!(
            registry.commit_active(ticket, facts("second", 5 * GIB)),
            Err(RegistrationError::InsufficientCapacity(_))
        ));

        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 5 * GIB,
            peak_bytes: 20 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        let recovered = registry.snapshot("first").unwrap();
        assert_eq!(recovered.availability, CapacityAvailability::Available);
        assert_eq!(recovered.safe_total_tokens, 8_192);
        assert!(recovered.safe_total_tokens < full);

        registry.refresh_memory(MlxMemorySnapshot {
            active_bytes: 4 * GIB,
            peak_bytes: 20 * GIB,
            memory_limit_bytes: Some(24 * GIB),
            metal_recommended_working_set_bytes: Some(24 * GIB),
        });
        assert_eq!(
            registry.snapshot("first").unwrap().safe_total_tokens,
            recovered.safe_total_tokens,
            "additional cleanup headroom must follow recovery hysteresis, not jump full"
        );
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
