use std::collections::BTreeMap;
use std::time::Instant;

use higgs_engine::{EngineCostDescription, MlxMemorySnapshot};
use serde::{Deserialize, Serialize};

#[allow(dead_code)] // Task 5 attaches this completed lifecycle to AppState.
pub(crate) mod pressure;
mod profile;
mod registry;

pub use profile::{LearnedBandEvidence, LearnedProfile, LearnedProfileKey, LearnedProfileStore};
pub use registry::{
    ActiveRegistration, CacheAllocationPlan, CacheCapabilities, CapacityRegistry,
    DrainRegistration, LoadCapacitySnapshot, ModelCapacityFacts, ModelContentIdentity,
    PublishedMemoryMeasurement, RegistrationError, RegistrationTicket, fingerprint_model_artifacts,
};
pub(crate) use registry::{
    CacheReclamation, CapacityAdmissionError, RequestReservation, RequestReservationAttempt,
};

/// Owned process observer; server shutdown must consume and join it.
#[must_use = "the process pressure observer must be stopped and joined"]
pub struct CapacityPressureObserver(Option<pressure::PressureObserverHandle>);

/// Sole observer sink. It records pressure before boot models load, then gains
/// a weak router attachment and immediately applies the latest cache policy.
pub struct CapacityPressureCoordinator {
    capacity: std::sync::Arc<CapacityRegistry>,
    state: std::sync::RwLock<Option<std::sync::Weak<crate::state::AppState>>>,
}

impl CapacityPressureCoordinator {
    #[must_use]
    pub fn new(capacity: std::sync::Arc<CapacityRegistry>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            capacity,
            state: std::sync::RwLock::new(None),
        })
    }

    pub async fn attach(&self, state: &crate::state::SharedState) -> Result<(), String> {
        *self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(std::sync::Arc::downgrade(state));
        state
            .router
            .apply_capacity_cache_allocations(&self.capacity)
            .await
    }
}

pub async fn start_capacity_pressure_observer(
    coordinator: std::sync::Arc<CapacityPressureCoordinator>,
) -> Result<CapacityPressureObserver, String> {
    pressure::system_observer_config()
        .start(coordinator)
        .await
        .map(|handle| CapacityPressureObserver(Some(handle)))
        .map_err(|error| error.to_string())
}

impl CapacityPressureObserver {
    pub async fn stop(mut self) -> Result<(), String> {
        let Some(handle) = self.0.take() else {
            return Ok(());
        };
        handle.stop().await.map_err(|error| error.to_string())
    }
}

/// Reserve one exact post-tokenization request peak. Optional caches are
/// reclaimed and acknowledged before a terminal capacity response; only
/// individually safe requests blocked by live reservations enter the FIFO.
pub(crate) async fn admit_generation_request(
    state: &crate::state::SharedState,
    model: &str,
    execution_path: ExecutionPath,
    prompt_tokens: usize,
    uncached_suffix_tokens: usize,
    output_tokens: u32,
) -> Result<RequestReservation, crate::error::ServerError> {
    #[cfg(test)]
    ensure_route_test_capacity(state, model);
    let generation = || {
        state
            .capacity
            .snapshot(model)
            .map(|snapshot| snapshot.generation)
            .unwrap_or(0)
    };
    let unavailable = || {
        crate::error::ServerError::CapacityUnavailable(CapacityUnavailableError::new(
            state.capacity.boot_id(),
            generation(),
        ))
    };
    // Admission charges only the uncached suffix. Callers obtain the
    // retained-prefix fact race-safely before this call and the session
    // worker revalidates it at acceptance; a suffix above the prompt can
    // only come from a stale fact and is clamped, never enlarged.
    let suffix_tokens =
        u64::try_from(uncached_suffix_tokens.min(prompt_tokens)).map_err(|_| unavailable())?;
    let prompt_tokens = u64::try_from(prompt_tokens).map_err(|_| unavailable())?;
    let request = RequestCost {
        execution_path,
        prompt_tokens,
        suffix_tokens: suffix_tokens.min(prompt_tokens),
        output_tokens: u64::from(output_tokens),
        retained_growth_bytes: 0,
    };

    // Fresh allocator bytes before the first atomic decision: positive
    // unaccounted MLX bytes from prior work must be visible even when the
    // decision would otherwise succeed, not only after a rejection.
    state.capacity.refresh_measured_memory();

    let mut attempt = state.capacity.try_reserve_request(model, request);
    for reclamation in [CacheReclamation::Prefix, CacheReclamation::Retained] {
        if !matches!(attempt, RequestReservationAttempt::Rejected(_)) {
            break;
        }
        if state.capacity.request_cache_reclamation(reclamation) {
            match reclamation {
                CacheReclamation::Prefix => {
                    state
                        .router
                        .apply_capacity_cache_allocations(&state.capacity)
                        .await
                }
                CacheReclamation::Retained => {
                    state
                        .router
                        .apply_capacity_retained_reclamation(&state.capacity)
                        .await
                }
            }
            .map_err(|_| unavailable())?;
            #[cfg(test)]
            let memory = state.capacity.admission_test_memory().0;
            #[cfg(not(test))]
            let memory = MlxMemorySnapshot::measure().map_err(|_| unavailable())?;
            state.capacity.refresh_memory_after_reclamation(memory);
            attempt = state.capacity.try_reserve_request(model, request);
        }
    }

    match attempt {
        RequestReservationAttempt::Reserved(reservation) => Ok(reservation),
        RequestReservationAttempt::Contended => state
            .capacity
            .reserve_request(model, request)
            .await
            .map_err(capacity_server_error),
        RequestReservationAttempt::Rejected(error) => Err(capacity_server_error(error)),
    }
}

#[cfg(test)]
fn ensure_route_test_capacity(state: &crate::state::SharedState, model: &str) {
    if state.capacity.snapshot(model).is_ok() {
        return;
    }
    const GIB: u64 = 1024 * 1024 * 1024;
    let memory = MlxMemorySnapshot {
        active_bytes: GIB,
        peak_bytes: GIB,
        memory_limit_bytes: Some(64 * GIB),
        metal_recommended_working_set_bytes: Some(64 * GIB),
    };
    let facts = route_test_facts(model, memory, 1, 1_048_576);
    state.capacity.refresh_memory(memory);
    if let Ok(ticket) = state.capacity.begin_registration(model.to_owned())
        && let Ok(active) = state.capacity.commit_active(ticket, facts)
    {
        active.publish();
        let plan = state.capacity.cache_allocation_plan();
        let _ = state
            .capacity
            .publish_cache_allocation_revision(plan.revision);
    }
}

#[cfg(test)]
fn route_test_facts(
    model: &str,
    memory: MlxMemorySnapshot,
    persistent_bytes_per_token: u64,
    token_ceiling: u64,
) -> ModelCapacityFacts {
    ModelCapacityFacts {
        model: model.to_owned(),
        model_fingerprint: format!("sha256:test-{model}"),
        memory,
        costs: EngineCostDescription {
            fixed_live_session_bytes: 0,
            persistent_bytes_per_token,
            decode_workspace_bytes: 0,
            transient_prefill: higgs_engine::TransientPrefillEstimate {
                base_bytes: 0,
                bytes_per_prompt_token: 0,
                bytes_per_chunk_token: 0,
                max_prompt_tokens: 1_048_576,
                max_chunk_tokens: 4_096,
            },
        },
        loaded_model_bytes: memory.active_bytes,
        architectural_max_tokens: 1_048_576,
        prefill_chunk_tokens: 1_024,
        retained_session_tokens: 0,
        retained_resident_bytes: 0,
        prefix_cache_resident_bytes: 0,
        retained_bytes_ceiling: 0,
        prefix_cache_bytes_ceiling: 0,
        cache_capabilities: CacheCapabilities {
            retained_sessions: false,
            prefix_cache: false,
        },
        configured_total_token_ceiling: Some(token_ceiling),
        configured_output_token_ceiling: Some(token_ceiling),
        quantization: "test".to_owned(),
        execution_mode: "test".to_owned(),
        kv_representation: "test".to_owned(),
        prefill_model_identity: None,
        drafter_identity: None,
        learned_profile_key: None,
        startup_headroom_bytes: 0,
    }
}

#[cfg(test)]
pub(crate) fn rejecting_route_test_state(
    model: &str,
) -> (
    crate::state::SharedState,
    std::sync::Arc<crate::state::Engine>,
) {
    use std::collections::HashMap;

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;
    let engine = std::sync::Arc::new(crate::state::Engine::test_stub(model));
    let dir = tempfile::tempdir().expect("test config directory");
    let path = dir.path().join("config.toml");
    std::fs::write(&path, "[provider.stub]\nurl = \"http://127.0.0.1:1\"\n").expect("test config");
    let config = crate::config::load_config_file(&path, None).expect("load test config");
    let router = crate::router::Router::from_config(
        &config,
        HashMap::from([(model.to_owned(), std::sync::Arc::clone(&engine))]),
    )
    .expect("test router");
    let state = std::sync::Arc::new(crate::state::AppState::new(
        router,
        config,
        reqwest::Client::new(),
        None,
    ));
    let memory = MlxMemorySnapshot {
        active_bytes: 2 * GIB,
        peak_bytes: 2 * GIB,
        memory_limit_bytes: Some(24 * GIB),
        metal_recommended_working_set_bytes: Some(24 * GIB),
    };
    state.capacity.refresh_memory(memory);
    let ticket = state
        .capacity
        .begin_registration(model.to_owned())
        .expect("begin test registration");
    let mut facts = route_test_facts(model, memory, MIB, 4_096);
    facts.cache_capabilities.prefix_cache = true;
    facts.prefix_cache_bytes_ceiling = GIB;
    state
        .capacity
        .commit_active(ticket, facts)
        .expect("commit test registration")
        .publish();
    let plan = state.capacity.cache_allocation_plan();
    assert!(
        state
            .capacity
            .publish_cache_allocation_revision(plan.revision)
    );
    (state, engine)
}

/// Route-test state whose admission envelope binds on bytes, not the token
/// ceiling, and whose transient prefill cost is real — so RetainedSuffix
/// charging of only the uncached suffix is observable through the route.
#[cfg(test)]
pub(crate) fn suffix_charging_route_test_state(
    model: &str,
) -> (
    crate::state::SharedState,
    std::sync::Arc<crate::state::Engine>,
) {
    use std::collections::HashMap;

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;
    let engine = std::sync::Arc::new(crate::state::Engine::test_stub(model));
    let dir = tempfile::tempdir().expect("test config directory");
    let path = dir.path().join("config.toml");
    std::fs::write(&path, "[provider.stub]\nurl = \"http://127.0.0.1:1\"\n").expect("test config");
    let config = crate::config::load_config_file(&path, None).expect("load test config");
    let router = crate::router::Router::from_config(
        &config,
        HashMap::from([(model.to_owned(), std::sync::Arc::clone(&engine))]),
    )
    .expect("test router");
    let state = std::sync::Arc::new(crate::state::AppState::new(
        router,
        config,
        reqwest::Client::new(),
        None,
    ));
    let memory = MlxMemorySnapshot {
        active_bytes: 2 * GIB,
        peak_bytes: 2 * GIB,
        memory_limit_bytes: Some(12 * GIB),
        metal_recommended_working_set_bytes: Some(12 * GIB),
    };
    state.capacity.refresh_memory(memory);
    let ticket = state
        .capacity
        .begin_registration(model.to_owned())
        .expect("begin test registration");
    let mut facts = route_test_facts(model, memory, MIB, 3_072);
    facts.costs.transient_prefill.base_bytes = MIB;
    facts.costs.transient_prefill.bytes_per_prompt_token = 4 * MIB;
    facts.costs.transient_prefill.max_prompt_tokens = 8_192;
    state
        .capacity
        .commit_active(ticket, facts)
        .expect("commit test registration")
        .publish();
    let plan = state.capacity.cache_allocation_plan();
    assert!(
        state
            .capacity
            .publish_cache_allocation_revision(plan.revision)
    );
    (state, engine)
}

/// Arm the reservation's stop with the configured request-timeout watchdog
/// and install it thread-locally for the worker's engine call. Engines
/// observe it at every bounded prefill chunk and decode step; critical
/// pressure and model drain are signalled through the same handle by the
/// registry.
pub(crate) fn install_reservation_stop(
    reservation: &RequestReservation,
    watchdog: Option<std::time::Duration>,
) -> higgs_engine::stop::GenerationStopGuard {
    let stop = reservation.stop();
    stop.set_watchdog(watchdog);
    higgs_engine::stop::install_generation_stop(stop)
}

/// The no-progress watchdog window: the configured server request timeout.
pub(crate) fn request_watchdog(state: &crate::state::SharedState) -> Option<std::time::Duration> {
    let timeout = state.config.server.timeout;
    (timeout > 0.0).then(|| std::time::Duration::from_secs_f64(timeout))
}

fn capacity_server_error(error: CapacityAdmissionError) -> crate::error::ServerError {
    match error {
        CapacityAdmissionError::Exceeded(error) => {
            crate::error::ServerError::CapacityExceeded(error)
        }
        CapacityAdmissionError::Unavailable(error) => {
            crate::error::ServerError::CapacityUnavailable(error)
        }
    }
}

pub const CAPACITY_SCHEMA_VERSION: u32 = 1;
pub const CAPACITY_RETRY_AFTER_MS: u64 = 5_000;

fn deserialize_schema_version<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let schema_version = u32::deserialize(deserializer)?;
    if schema_version != CAPACITY_SCHEMA_VERSION {
        return Err(<D::Error as serde::de::Error>::custom(format_args!(
            "unsupported capacity schemaVersion {schema_version}; expected {CAPACITY_SCHEMA_VERSION}"
        )));
    }
    Ok(schema_version)
}

/// Whether the requested model can currently accept its minimum working request.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityAvailability {
    Available,
    Unavailable,
}

/// Process memory pressure used to derive the published capacity envelope.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPressure {
    Normal,
    Constrained,
    Critical,
}

/// One content-free system observation delivered to the capacity controller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PressureObservation {
    pub pressure: MemoryPressure,
    pub swap_out_delta: u64,
    pub compressor_delta: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ZeroCapacityRecovery {
    Preserve,
    BoundedMinimum,
}

/// Evidence backing the current capacity envelope.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityBasis {
    Conservative,
    Learned,
}

/// Versioned capacity advertised for one model by this Higgs process.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CapacitySnapshot {
    #[serde(deserialize_with = "deserialize_schema_version")]
    pub schema_version: u32,
    pub model: String,
    pub model_fingerprint: String,
    pub boot_id: String,
    pub generation: u64,
    pub availability: CapacityAvailability,
    pub pressure: MemoryPressure,
    pub safe_total_tokens: u64,
    pub recommended_output_tokens: u64,
    pub max_prompt_tokens: u64,
    pub retained_session_tokens: u64,
    pub retained_bytes: u64,
    pub prefix_cache_bytes: u64,
    pub basis: CapacityBasis,
}

impl CapacitySnapshot {
    /// Revisions are process-local: a restarted server may reuse a generation number.
    #[must_use]
    pub fn is_same_revision(&self, other: &Self) -> bool {
        self.boot_id == other.boot_id && self.generation == other.generation
    }
}

/// OpenAI-compatible outer error object shared by HTTP errors and terminal SSE errors.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CapacityErrorEnvelope<T> {
    error: T,
}

impl<T> CapacityErrorEnvelope<T> {
    #[must_use]
    pub fn new(error: T) -> Self {
        Self { error }
    }
}

/// Request-specific limits returned before inference when a request is too large.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityExceededError {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    safe_prompt_tokens: u64,
    safe_total_tokens: u64,
    boot_id: String,
    generation: u64,
}

impl CapacityExceededError {
    #[must_use]
    pub fn new(
        safe_prompt_tokens: u64,
        safe_total_tokens: u64,
        boot_id: String,
        generation: u64,
    ) -> Self {
        Self {
            error_type: "higgs_capacity_exceeded",
            code: "compact_and_retry",
            safe_prompt_tokens,
            safe_total_tokens,
            boot_id,
            generation,
        }
    }

    #[must_use]
    pub fn boot_id(&self) -> &str {
        &self.boot_id
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

/// Temporary inability to fit even the minimum working request.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityUnavailableError {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    boot_id: String,
    generation: u64,
    retry_after_ms: u64,
}

impl CapacityUnavailableError {
    #[must_use]
    pub fn new(boot_id: String, generation: u64) -> Self {
        Self {
            error_type: "higgs_capacity_unavailable",
            code: "capacity_unavailable",
            boot_id,
            generation,
            retry_after_ms: CAPACITY_RETRY_AFTER_MS,
        }
    }

    #[must_use]
    pub fn boot_id(&self) -> &str {
        &self.boot_id
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

/// Terminal stream event emitted when pressure interrupts active generation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityInterruptedError {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    boot_id: String,
    generation: u64,
    partial_output_tokens: u64,
}

impl CapacityInterruptedError {
    #[must_use]
    pub fn new(boot_id: String, generation: u64, partial_output_tokens: u64) -> Self {
        Self {
            error_type: "higgs_capacity_interrupted",
            code: "capacity_interrupted",
            boot_id,
            generation,
            partial_output_tokens,
        }
    }
}

/// Typed unknown-model response for the capacity extension route.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityModelNotFoundError {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    model: String,
}

impl CapacityModelNotFoundError {
    #[must_use]
    pub fn new(model: String) -> Self {
        Self {
            error_type: "higgs_capacity_model_not_found",
            code: "model_not_found",
            model,
        }
    }
}

const GIBIBYTE: u64 = 1024 * 1024 * 1024;
const TOKEN_ALIGNMENT: u64 = 1024;
const MINIMUM_OUTPUT_RESERVE_TOKENS: u64 = 1024;
// Smallest aligned recovery envelope whose 12.5% ramp can advance by one aligned step.
const MINIMUM_RECOVERY_TOTAL_TOKENS: u64 = 8 * TOKEN_ALIGNMENT;
const DEFAULT_OUTPUT_TOKENS: u64 = 4096;

#[must_use]
pub const fn floor_1024(tokens: u64) -> u64 {
    tokens / TOKEN_ALIGNMENT * TOKEN_ALIGNMENT
}

/// Copied allocator, engine, cache, and user-ceiling facts used by the solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityInputs {
    pub memory: MlxMemorySnapshot,
    pub costs: EngineCostDescription,
    pub loaded_model_bytes: u64,
    pub architectural_max_tokens: u64,
    pub prefill_chunk_tokens: u64,
    pub retained_bytes: u64,
    pub prefix_cache_bytes: u64,
    pub active_reservation_bytes: u64,
    pub configured_total_token_ceiling: Option<u64>,
    pub configured_output_token_ceiling: Option<u64>,
    pub pressure: MemoryPressure,
}

impl CapacityInputs {
    #[must_use]
    pub fn working_set_authority_bytes(&self) -> Option<u64> {
        [
            self.memory.memory_limit_bytes,
            self.memory.metal_recommended_working_set_bytes,
        ]
        .into_iter()
        .flatten()
        .filter(|bytes| *bytes > 0)
        .min()
    }

    #[must_use]
    pub fn protected_reserve_bytes(&self) -> Option<u64> {
        let authority = self.working_set_authority_bytes()?;
        let percentage = match self.pressure {
            MemoryPressure::Normal => 20,
            MemoryPressure::Constrained | MemoryPressure::Critical => 30,
        };
        Some((authority.checked_mul(percentage)? / 100).max(4 * GIBIBYTE))
    }

    fn usable_bytes(&self) -> Option<u64> {
        self.working_set_authority_bytes()?
            .checked_sub(self.protected_reserve_bytes()?)
    }

    fn token_ceiling(&self) -> u64 {
        self.configured_total_token_ceiling
            .map_or(self.architectural_max_tokens, |configured| {
                configured.min(self.architectural_max_tokens)
            })
    }

    fn output_tokens(&self) -> u64 {
        self.configured_output_token_ceiling
            .map_or(DEFAULT_OUTPUT_TOKENS, |configured| {
                configured.min(DEFAULT_OUTPUT_TOKENS)
            })
    }
}

/// One request's semantic requirements not already committed process-wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestCost {
    pub execution_path: ExecutionPath,
    pub prompt_tokens: u64,
    pub suffix_tokens: u64,
    pub output_tokens: u64,
    pub retained_growth_bytes: u64,
}

/// Every term in the one checked admission inequality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteLedger {
    pub loaded_baseline_bytes: u64,
    pub unaccounted_active_bytes: u64,
    pub fixed_session_bytes: u64,
    pub prompt_bytes: u64,
    pub output_bytes: u64,
    pub decode_bytes: u64,
    pub retained_and_cache_bytes: u64,
    pub learned_retained_bytes: u64,
    pub transient_bytes: u64,
}

impl ByteLedger {
    #[must_use]
    pub fn total_bytes(&self) -> Option<u64> {
        self.loaded_baseline_bytes
            .checked_add(self.fixed_session_bytes)?
            .checked_add(self.unaccounted_active_bytes)?
            .checked_add(self.prompt_bytes)?
            .checked_add(self.output_bytes)?
            .checked_add(self.decode_bytes)?
            .checked_add(self.retained_and_cache_bytes)?
            .checked_add(self.learned_retained_bytes)?
            .checked_add(self.transient_bytes)
    }
}

/// Published result derived from the same ledger used by admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityDecision {
    pub availability: CapacityAvailability,
    pub safe_total_tokens: u64,
    pub recommended_output_tokens: u64,
    pub max_prompt_tokens: u64,
    pub usable_bytes: u64,
}

impl CapacityDecision {
    fn bounded_by(self, total_ceiling: u64) -> Self {
        let safe_total_tokens = self.safe_total_tokens.min(total_ceiling);
        let recommended_output_tokens = self.recommended_output_tokens.min(safe_total_tokens);
        Self {
            availability: if safe_total_tokens >= recommended_output_tokens.max(TOKEN_ALIGNMENT) {
                self.availability
            } else {
                CapacityAvailability::Unavailable
            },
            safe_total_tokens,
            recommended_output_tokens,
            max_prompt_tokens: safe_total_tokens
                .checked_sub(recommended_output_tokens)
                .unwrap_or(0),
            usable_bytes: self.usable_bytes,
        }
    }
}

/// Result of checking one request against the current byte and token envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    Admitted(ByteLedger),
    Exceeded(CapacityDecision),
    FixedCostUnavailable,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPath {
    Cold,
    RetainedSuffix,
    RadixHit,
}

/// Content-free, allocation-bearing evidence from one real request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationObservation {
    pub path: ExecutionPath,
    pub full_prompt_tokens: u64,
    pub suffix_tokens: u64,
    pub predicted_peak_bytes: u64,
    pub observed_peak_bytes: u64,
    pub observed_retained_bytes: u64,
    pub observed_suffix_transient_bytes: u64,
    pub pressure: MemoryPressure,
    pub swap_out_delta: u64,
    pub compressor_growth_bytes: u64,
    pub allocation_bearing: bool,
}

impl AllocationObservation {
    #[must_use]
    pub const fn clean(
        path: ExecutionPath,
        full_prompt_tokens: u64,
        suffix_tokens: u64,
        predicted_peak_bytes: u64,
        observed_peak_bytes: u64,
    ) -> Self {
        Self {
            path,
            full_prompt_tokens,
            suffix_tokens,
            predicted_peak_bytes,
            observed_peak_bytes,
            observed_retained_bytes: 0,
            observed_suffix_transient_bytes: 0,
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_growth_bytes: 0,
            allocation_bearing: true,
        }
    }

    fn is_clean(self) -> bool {
        self.allocation_bearing
            && self.observed_peak_bytes > 0
            && self.pressure == MemoryPressure::Normal
            && self.swap_out_delta == 0
            && self.compressor_growth_bytes == 0
    }
}

pub trait Clock: Clone {
    fn now_millis(&self) -> u64;
}

#[derive(Clone, Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Clock for SystemClock {
    fn now_millis(&self) -> u64 {
        self.origin
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }
}

#[derive(Clone, Debug, Default)]
struct RuntimeBandEvidence {
    cold_high_water_bytes: u64,
    cold_replacement_qualified: bool,
    retained_high_water_bytes: u64,
    suffix_high_water_bytes: u64,
    clean_cold_samples: Vec<(u64, u64)>,
}

/// Pure controller over copied byte-domain inputs and content-free evidence.
#[derive(Debug)]
pub struct CapacityController<C: Clock = SystemClock> {
    inputs: CapacityInputs,
    clock: C,
    boot_id: String,
    decision: CapacityDecision,
    evidence: BTreeMap<u64, RuntimeBandEvidence>,
    last_swap_out_millis: Option<u64>,
    strongest_pressure_since_normal: MemoryPressure,
}

impl CapacityController<SystemClock> {
    #[must_use]
    pub fn new(inputs: CapacityInputs) -> Self {
        Self::with_clock(inputs, SystemClock::default())
    }
}

impl<C: Clock> CapacityController<C> {
    pub(crate) fn transactional_copy(&self) -> Self {
        Self {
            inputs: self.inputs,
            clock: self.clock.clone(),
            boot_id: self.boot_id.clone(),
            decision: self.decision,
            evidence: self.evidence.clone(),
            last_swap_out_millis: self.last_swap_out_millis,
            strongest_pressure_since_normal: self.strongest_pressure_since_normal,
        }
    }

    /// Replace process-wide residency inputs without discarding a pressure
    /// downshift or bypassing the evidence-gated recovery ramp.
    pub(crate) fn replace_shared_residency(
        &mut self,
        memory: MlxMemorySnapshot,
        loaded_model_bytes: u64,
        retained_bytes: u64,
        prefix_cache_bytes: u64,
        active_reservation_bytes: u64,
        zero_recovery: ZeroCapacityRecovery,
    ) {
        let prior_total = self.decision.safe_total_tokens;
        self.inputs.memory = memory;
        self.inputs.loaded_model_bytes = loaded_model_bytes;
        self.inputs.retained_bytes = retained_bytes;
        self.inputs.prefix_cache_bytes = prefix_cache_bytes;
        self.inputs.active_reservation_bytes = active_reservation_bytes;
        let raw = self.solve_static();
        let recovery_ceiling = if prior_total == 0
            && raw.availability == CapacityAvailability::Available
            && zero_recovery == ZeroCapacityRecovery::BoundedMinimum
        {
            raw.safe_total_tokens.min(MINIMUM_RECOVERY_TOTAL_TOKENS)
        } else {
            prior_total
        };
        self.decision = raw.bounded_by(recovery_ceiling);
        if self.inputs.pressure == MemoryPressure::Critical {
            self.decision.availability = CapacityAvailability::Unavailable;
        }
    }

    #[must_use]
    pub fn with_clock(inputs: CapacityInputs, clock: C) -> Self {
        let initial_pressure = inputs.pressure;
        let mut controller = Self {
            inputs,
            clock,
            boot_id: uuid::Uuid::new_v4().to_string(),
            decision: unavailable_decision(),
            evidence: BTreeMap::new(),
            last_swap_out_millis: None,
            strongest_pressure_since_normal: initial_pressure,
        };
        controller.decision = controller.solve_static();
        controller
    }

    #[must_use]
    pub fn boot_id(&self) -> &str {
        &self.boot_id
    }

    #[must_use]
    pub const fn decision(&self) -> CapacityDecision {
        self.decision
    }

    #[must_use]
    pub const fn pressure(&self) -> MemoryPressure {
        self.inputs.pressure
    }

    /// Applies a live observation once. Repeated notifications at the same effective
    /// pressure freeze upward learning without repeatedly multiplying the envelope.
    pub fn apply_pressure_observation(
        &mut self,
        observation: PressureObservation,
    ) -> CapacityDecision {
        const SWAP_RECOVERY_MILLIS: u64 = 60_000;

        let now = self.clock.now_millis();
        if observation.swap_out_delta > 0 {
            self.last_swap_out_millis = Some(now);
        }
        let swap_is_sticky = self.last_swap_out_millis.is_some_and(|last_swap| {
            observation.pressure != MemoryPressure::Normal
                || now.saturating_sub(last_swap) < SWAP_RECOVERY_MILLIS
        });
        if self.last_swap_out_millis.is_some() && !swap_is_sticky {
            self.last_swap_out_millis = None;
        }
        let effective_pressure = if swap_is_sticky {
            MemoryPressure::Critical
        } else if observation.pressure == MemoryPressure::Normal && observation.compressor_delta > 0
        {
            // Compressor growth is deterministic evidence that a nominally normal
            // system is dirty. It freezes rises and uses the constrained reserve;
            // no arbitrary compression-rate threshold is invented.
            MemoryPressure::Constrained
        } else {
            observation.pressure
        };

        if effective_pressure != MemoryPressure::Normal
            || observation.swap_out_delta > 0
            || observation.compressor_delta > 0
        {
            self.clear_clean_windows();
        }
        if effective_pressure == self.inputs.pressure {
            return self.decision;
        }
        if effective_pressure == MemoryPressure::Normal {
            return self.recompute_for_pressure(effective_pressure, None);
        }
        if pressure_severity(effective_pressure)
            > pressure_severity(self.strongest_pressure_since_normal)
        {
            return self.recompute_for_pressure(effective_pressure, None);
        }

        // Warning and critical use the same 30% protected reserve. Moving to a
        // less severe state, or revisiting a severity already applied during the
        // same pressure episode, updates availability without multiplying the
        // already-downshifted token envelope again.
        let previous_total = self.decision.safe_total_tokens;
        let recomputed = self.recompute_for_pressure(effective_pressure, None);
        let transition_total = if previous_total == 0 {
            recomputed.safe_total_tokens
        } else {
            previous_total
        };
        let mut transition = self.decision_for_total(
            transition_total,
            recomputed.usable_bytes,
            recomputed.recommended_output_tokens,
        );
        if effective_pressure == MemoryPressure::Critical {
            transition.availability = CapacityAvailability::Unavailable;
        }
        self.decision = transition;
        self.decision
    }

    #[must_use]
    pub fn byte_ledger(&self, request: RequestCost) -> Option<ByteLedger> {
        self.byte_ledger_with_evidence(request)
    }

    #[must_use]
    pub fn admit(&self, request: RequestCost) -> Admission {
        if self.decision.availability != CapacityAvailability::Available {
            return Admission::Unavailable;
        }
        let total_tokens = match request.prompt_tokens.checked_add(request.output_tokens) {
            Some(tokens) => tokens,
            None => return Admission::Exceeded(unavailable_decision()),
        };
        let output_ceiling = self
            .inputs
            .configured_output_token_ceiling
            .unwrap_or(request.output_tokens);
        let bounded_output = request
            .output_tokens
            .min(output_ceiling)
            .min(self.decision.recommended_output_tokens);
        let minimum_total = TOKEN_ALIGNMENT.max(bounded_output);
        let minimum_request = RequestCost {
            execution_path: request.execution_path,
            prompt_tokens: minimum_total - bounded_output,
            suffix_tokens: request.suffix_tokens.min(minimum_total - bounded_output),
            output_tokens: bounded_output,
            retained_growth_bytes: request.retained_growth_bytes,
        };
        if self
            .byte_ledger_with_evidence(minimum_request)
            .and_then(|ledger| ledger.total_bytes())
            .is_none_or(|bytes| bytes > self.decision.usable_bytes)
        {
            return Admission::FixedCostUnavailable;
        }
        let request_bound = self
            .solve_for_shape(
                request.execution_path,
                bounded_output,
                request.retained_growth_bytes,
                request.suffix_tokens,
                false,
            )
            .bounded_by(self.decision.safe_total_tokens);
        if request_bound.availability != CapacityAvailability::Available {
            return Admission::Exceeded(request_bound);
        }
        let Some(ledger) = self.byte_ledger_with_evidence(request) else {
            return Admission::Exceeded(request_bound);
        };
        let Some(total_bytes) = ledger.total_bytes() else {
            return Admission::Exceeded(request_bound);
        };
        if request.output_tokens <= output_ceiling
            && total_tokens <= request_bound.safe_total_tokens
            && request.prompt_tokens <= request_bound.max_prompt_tokens
            && total_bytes <= request_bound.usable_bytes
        {
            Admission::Admitted(ledger)
        } else {
            Admission::Exceeded(request_bound)
        }
    }

    pub fn recompute_for_pressure(
        &mut self,
        pressure: MemoryPressure,
        memory: Option<MlxMemorySnapshot>,
    ) -> CapacityDecision {
        let previous = self.decision.safe_total_tokens;
        self.inputs.pressure = pressure;
        if pressure == MemoryPressure::Normal {
            self.strongest_pressure_since_normal = MemoryPressure::Normal;
        } else if pressure_severity(pressure)
            > pressure_severity(self.strongest_pressure_since_normal)
        {
            self.strongest_pressure_since_normal = pressure;
        }
        if pressure != MemoryPressure::Normal {
            self.clear_clean_windows();
        }
        if let Some(memory) = memory {
            self.inputs.memory = memory;
        }
        let recomputed = if pressure == MemoryPressure::Critical {
            self.solve_for_shape_at_current_limits(
                ExecutionPath::Cold,
                self.inputs.output_tokens(),
                0,
                u64::MAX,
                true,
            )
        } else {
            self.solve_static()
        };
        let fraction = match pressure {
            MemoryPressure::Normal => None,
            MemoryPressure::Constrained => Some(75),
            MemoryPressure::Critical => Some(50),
        };
        self.decision = if let Some(percent) = fraction {
            let downshift = if previous == 0
                && pressure == MemoryPressure::Constrained
                && recomputed.availability == CapacityAvailability::Available
            {
                recomputed
                    .safe_total_tokens
                    .min(MINIMUM_RECOVERY_TOTAL_TOKENS)
            } else {
                previous
                    .checked_mul(percent)
                    .map(|tokens| floor_1024(tokens / 100))
                    .unwrap_or(0)
            };
            let total = downshift.min(recomputed.safe_total_tokens);
            let mut decision = self.decision_for_total(
                total,
                recomputed.usable_bytes,
                recomputed.recommended_output_tokens,
            );
            if pressure == MemoryPressure::Critical {
                decision.availability = CapacityAvailability::Unavailable;
            }
            decision
        } else {
            let total =
                if previous == 0 && recomputed.availability == CapacityAvailability::Available {
                    recomputed
                        .safe_total_tokens
                        .min(MINIMUM_RECOVERY_TOTAL_TOKENS)
                } else {
                    previous.min(recomputed.safe_total_tokens)
                };
            self.decision_for_total(
                total,
                recomputed.usable_bytes,
                recomputed.recommended_output_tokens,
            )
        };
        self.decision
    }

    pub fn observe(&mut self, observation: AllocationObservation) {
        let band = prompt_band(observation.full_prompt_tokens);
        let now = self.clock.now_millis();
        if !observation.is_clean() {
            self.clear_clean_windows();
        }
        let underpredicted = observation.allocation_bearing
            && observation.observed_peak_bytes > observation.predicted_peak_bytes;
        let high_water =
            add_ten_percent_ceiling(observation.observed_peak_bytes).unwrap_or(u64::MAX);
        let mut cost_increased = false;
        if underpredicted {
            self.clear_qualifications();
            let evidence = self.evidence.entry(band).or_default();
            match observation.path {
                ExecutionPath::Cold => {
                    if high_water > evidence.cold_high_water_bytes {
                        evidence.cold_high_water_bytes = high_water;
                        cost_increased = true;
                    }
                }
                ExecutionPath::RetainedSuffix | ExecutionPath::RadixHit => {}
            }
        }
        if observation.allocation_bearing {
            let evidence = self.evidence.entry(band).or_default();
            if observation.observed_retained_bytes > 0 {
                let retained = add_ten_percent_ceiling(observation.observed_retained_bytes)
                    .unwrap_or(u64::MAX);
                if retained > evidence.retained_high_water_bytes {
                    evidence.retained_high_water_bytes = retained;
                    cost_increased = true;
                }
            }
            if observation.observed_suffix_transient_bytes > 0 {
                let suffix = add_ten_percent_ceiling(observation.observed_suffix_transient_bytes)
                    .unwrap_or(u64::MAX);
                if suffix > evidence.suffix_high_water_bytes {
                    evidence.suffix_high_water_bytes = suffix;
                    cost_increased = true;
                }
            }
        }
        if underpredicted || cost_increased {
            let recomputed = self.solve_static();
            self.decision = recomputed.bounded_by(self.decision.safe_total_tokens);
        }

        if observation.path == ExecutionPath::Cold
            && observation.is_clean()
            && !underpredicted
            && self.inputs.pressure == MemoryPressure::Normal
        {
            let evidence = self.evidence.entry(band).or_default();
            evidence
                .clean_cold_samples
                .push((now, observation.observed_peak_bytes));
            if evidence.clean_cold_samples.len() > 3 {
                evidence.clean_cold_samples.remove(0);
            }
            let ready = evidence.clean_cold_samples.len() == 3
                && evidence.clean_cold_samples[2]
                    .0
                    .checked_sub(evidence.clean_cold_samples[0].0)
                    .is_some_and(|elapsed| elapsed >= 5 * 60 * 1000);
            if ready {
                let observed_high_water = evidence
                    .clean_cold_samples
                    .iter()
                    .map(|(_, bytes)| *bytes)
                    .max()
                    .and_then(add_ten_percent_ceiling)
                    .unwrap_or(u64::MAX);
                evidence.cold_high_water_bytes = observed_high_water;
                evidence.cold_replacement_qualified = true;
                evidence.clean_cold_samples.clear();
            }
            if ready {
                let raw = self.solve_static();
                let step = floor_1024(self.decision.safe_total_tokens / 8).min(4096);
                let raised = self
                    .decision
                    .safe_total_tokens
                    .checked_add(step)
                    .unwrap_or(self.decision.safe_total_tokens)
                    .min(raw.safe_total_tokens)
                    .min(
                        if observation.full_prompt_tokens >= self.decision.max_prompt_tokens {
                            band.checked_mul(2)
                        } else {
                            Some(band)
                        }
                        .and_then(|prompt| prompt.checked_add(raw.recommended_output_tokens))
                        .unwrap_or(band),
                    );
                if raised > self.decision.safe_total_tokens {
                    self.decision = self.decision_for_total(
                        raised,
                        raw.usable_bytes,
                        raw.recommended_output_tokens,
                    );
                }
            }
        }
    }

    fn clear_clean_windows(&mut self) {
        for evidence in self.evidence.values_mut() {
            evidence.clean_cold_samples.clear();
        }
    }

    fn clear_qualifications(&mut self) {
        self.clear_clean_windows();
        for evidence in self.evidence.values_mut() {
            evidence.cold_replacement_qualified = false;
        }
    }

    #[must_use]
    pub fn learned_high_water_bytes(
        &self,
        path: ExecutionPath,
        full_prompt_tokens: u64,
    ) -> Option<u64> {
        let band = prompt_band(full_prompt_tokens);
        let bytes = self
            .evidence
            .range(..=band)
            .map(|(_, evidence)| match path {
                ExecutionPath::Cold => evidence.cold_high_water_bytes,
                ExecutionPath::RetainedSuffix | ExecutionPath::RadixHit => evidence
                    .retained_high_water_bytes
                    .max(evidence.suffix_high_water_bytes),
            })
            .max()?;
        (bytes > 0).then_some(bytes)
    }

    #[must_use]
    pub fn export_profile(
        &self,
        key: LearnedProfileKey,
        startup_headroom_bytes: u64,
    ) -> Option<LearnedProfile> {
        let evidence = self
            .evidence
            .iter()
            .filter_map(|(prompt_band, evidence)| {
                (evidence.cold_high_water_bytes > 0
                    || evidence.retained_high_water_bytes > 0
                    || evidence.suffix_high_water_bytes > 0)
                    .then_some(LearnedBandEvidence {
                        prompt_band: *prompt_band,
                        cold_high_water_bytes: evidence.cold_high_water_bytes,
                        cold_replacement_qualified: evidence.cold_replacement_qualified,
                        retained_high_water_bytes: evidence.retained_high_water_bytes,
                        suffix_high_water_bytes: evidence.suffix_high_water_bytes,
                    })
            })
            .collect();
        let profile = LearnedProfile::new(key, startup_headroom_bytes, evidence);
        profile.is_complete().then_some(profile)
    }

    pub fn restore_profile(
        &mut self,
        profile: &LearnedProfile,
        expected_key: &LearnedProfileKey,
        current_startup_headroom_bytes: u64,
    ) -> bool {
        if !profile.is_compatible(expected_key, current_startup_headroom_bytes) {
            return false;
        }
        let evidence = profile
            .evidence()
            .iter()
            .map(|persisted| {
                (
                    persisted.prompt_band,
                    RuntimeBandEvidence {
                        cold_high_water_bytes: persisted.cold_high_water_bytes,
                        cold_replacement_qualified: persisted.cold_replacement_qualified,
                        retained_high_water_bytes: persisted.retained_high_water_bytes,
                        suffix_high_water_bytes: persisted.suffix_high_water_bytes,
                        clean_cold_samples: Vec::new(),
                    },
                )
            })
            .collect();
        self.evidence = evidence;
        self.decision = self.solve_static();
        true
    }

    fn solve_static(&self) -> CapacityDecision {
        let output = self.inputs.output_tokens();
        self.solve_for_shape(ExecutionPath::Cold, output, 0, u64::MAX, true)
    }

    fn solve_for_shape(
        &self,
        execution_path: ExecutionPath,
        output: u64,
        retained_growth_bytes: u64,
        suffix_tokens: u64,
        publication_worst_path: bool,
    ) -> CapacityDecision {
        if self.inputs.pressure == MemoryPressure::Critical {
            return unavailable_decision();
        }
        self.solve_for_shape_at_current_limits(
            execution_path,
            output,
            retained_growth_bytes,
            suffix_tokens,
            publication_worst_path,
        )
    }

    fn solve_for_shape_at_current_limits(
        &self,
        execution_path: ExecutionPath,
        output: u64,
        retained_growth_bytes: u64,
        suffix_tokens: u64,
        publication_worst_path: bool,
    ) -> CapacityDecision {
        let Some(usable_bytes) = self.inputs.usable_bytes() else {
            return unavailable_decision();
        };
        let ceiling = floor_1024(self.inputs.token_ceiling());
        if ceiling < output || ceiling < TOKEN_ALIGNMENT {
            return unavailable_decision();
        }
        let mut low = 0;
        let mut high = ceiling / TOKEN_ALIGNMENT;
        while low < high {
            let midpoint = low + (high - low + 1) / 2;
            let total = midpoint * TOKEN_ALIGNMENT;
            let prompt = total.checked_sub(output).unwrap_or(0);
            let request = RequestCost {
                execution_path,
                prompt_tokens: prompt,
                suffix_tokens: suffix_tokens.min(prompt),
                output_tokens: output,
                retained_growth_bytes,
            };
            let fits = self
                .byte_ledger_with_evidence_for_policy(request, publication_worst_path)
                .and_then(|ledger| ledger.total_bytes())
                .is_some_and(|bytes| bytes <= usable_bytes);
            if fits {
                low = midpoint;
            } else {
                high = midpoint - 1;
            }
        }
        let total = low * TOKEN_ALIGNMENT;
        if total < output || total < TOKEN_ALIGNMENT {
            unavailable_decision()
        } else {
            self.decision_for_total(total, usable_bytes, output)
        }
    }

    fn decision_for_total(&self, total: u64, usable_bytes: u64, output: u64) -> CapacityDecision {
        let output = output.min(total);
        CapacityDecision {
            availability: if total >= output && total >= TOKEN_ALIGNMENT {
                CapacityAvailability::Available
            } else {
                CapacityAvailability::Unavailable
            },
            safe_total_tokens: total,
            recommended_output_tokens: output,
            max_prompt_tokens: total.checked_sub(output).unwrap_or(0),
            usable_bytes,
        }
    }

    fn byte_ledger_with_evidence(&self, request: RequestCost) -> Option<ByteLedger> {
        self.byte_ledger_with_evidence_for_policy(request, false)
    }

    fn byte_ledger_with_evidence_for_policy(
        &self,
        request: RequestCost,
        publication_worst_path: bool,
    ) -> Option<ByteLedger> {
        let prompt_bytes = self.inputs.costs.persistent_bytes(request.prompt_tokens)?;
        // Request admission charges the exact requested output. The
        // publication floor keeps published limits conservative for the
        // worst case of a request that grows its own output later.
        let output_charge_tokens = if publication_worst_path {
            request.output_tokens.max(MINIMUM_OUTPUT_RESERVE_TOKENS)
        } else {
            request.output_tokens
        };
        let output_bytes = self.inputs.costs.persistent_bytes(output_charge_tokens)?;
        let prefill_tokens = match request.execution_path {
            ExecutionPath::Cold => request.prompt_tokens,
            ExecutionPath::RetainedSuffix | ExecutionPath::RadixHit => {
                request.suffix_tokens.min(request.prompt_tokens)
            }
        };
        let static_transient = self
            .inputs
            .costs
            .transient_prefill
            .estimate_bytes(prefill_tokens, self.inputs.prefill_chunk_tokens)?;
        let band = prompt_band(request.prompt_tokens);
        let qualified_cold = self.evidence.get(&band).and_then(|evidence| {
            evidence
                .cold_replacement_qualified
                .then_some(evidence.cold_high_water_bytes)
        });
        let cold_transient = qualified_cold.unwrap_or_else(|| {
            static_transient.max(
                self.learned_high_water_bytes(ExecutionPath::Cold, request.prompt_tokens)
                    .unwrap_or(0),
            )
        });
        let suffix_transient = self
            .evidence
            .range(..=band)
            .map(|(_, evidence)| evidence.suffix_high_water_bytes)
            .max()
            .unwrap_or(0);
        let retained_evidence = self
            .evidence
            .range(..=band)
            .map(|(_, evidence)| evidence.retained_high_water_bytes)
            .max()
            .unwrap_or(0);
        let use_persisted_path = publication_worst_path
            || matches!(
                request.execution_path,
                ExecutionPath::RetainedSuffix | ExecutionPath::RadixHit
            );
        let transient_bytes = if use_persisted_path {
            cold_transient.max(suffix_transient)
        } else {
            cold_transient
        };
        let learned_retained_bytes = if use_persisted_path {
            retained_evidence
        } else {
            0
        };
        let known_active_bytes = self
            .inputs
            .loaded_model_bytes
            .checked_add(self.inputs.costs.fixed_live_session_bytes)?
            .checked_add(self.inputs.retained_bytes)?
            .checked_add(self.inputs.prefix_cache_bytes)?
            .checked_add(self.inputs.active_reservation_bytes)?;
        let unaccounted_active_bytes = self
            .inputs
            .memory
            .active_bytes
            .checked_sub(known_active_bytes)
            .unwrap_or(0);
        let retained_and_cache_bytes = self
            .inputs
            .retained_bytes
            .checked_add(self.inputs.prefix_cache_bytes)?
            .checked_add(self.inputs.active_reservation_bytes)?
            .checked_add(request.retained_growth_bytes)?;
        Some(ByteLedger {
            loaded_baseline_bytes: self.inputs.loaded_model_bytes,
            unaccounted_active_bytes,
            fixed_session_bytes: self.inputs.costs.fixed_live_session_bytes,
            prompt_bytes,
            output_bytes,
            decode_bytes: self.inputs.costs.decode_workspace_bytes,
            retained_and_cache_bytes,
            learned_retained_bytes,
            transient_bytes,
        })
    }
}

const fn unavailable_decision() -> CapacityDecision {
    CapacityDecision {
        availability: CapacityAvailability::Unavailable,
        safe_total_tokens: 0,
        recommended_output_tokens: 0,
        max_prompt_tokens: 0,
        usable_bytes: 0,
    }
}

fn prompt_band(tokens: u64) -> u64 {
    tokens
        .max(1)
        .checked_next_power_of_two()
        .unwrap_or(u64::MAX)
}

fn add_ten_percent_ceiling(bytes: u64) -> Option<u64> {
    let tenth = bytes / 10;
    let remainder = u64::from(bytes % 10 != 0);
    bytes.checked_add(tenth)?.checked_add(remainder)
}

const fn pressure_severity(pressure: MemoryPressure) -> u8 {
    match pressure {
        MemoryPressure::Normal => 0,
        MemoryPressure::Constrained => 1,
        MemoryPressure::Critical => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MODEL: &str = "escha-35b-a3b";
    const FINGERPRINT: &str =
        "sha256:7b2f5c8ae91a5b1d83f1364c2023e5e53b5530d0461a4193cf9bd37f4e70d821";
    const BOOT_ID: &str = "01993654-8af2-7b31-a420-c52ebc349287";

    fn available_snapshot() -> CapacitySnapshot {
        CapacitySnapshot {
            schema_version: 1,
            model: MODEL.to_owned(),
            model_fingerprint: FINGERPRINT.to_owned(),
            boot_id: BOOT_ID.to_owned(),
            generation: 7,
            availability: CapacityAvailability::Available,
            pressure: MemoryPressure::Normal,
            safe_total_tokens: 53_248,
            recommended_output_tokens: 4_096,
            max_prompt_tokens: 49_152,
            retained_session_tokens: 49_152,
            retained_bytes: 2_147_483_648,
            prefix_cache_bytes: 1_073_741_824,
            basis: CapacityBasis::Learned,
        }
    }

    #[test]
    fn capacity_snapshot_matches_schema_v1_json() {
        assert_eq!(
            serde_json::to_value(available_snapshot()).unwrap(),
            json!({
                "schemaVersion": 1,
                "model": MODEL,
                "modelFingerprint": FINGERPRINT,
                "bootId": BOOT_ID,
                "generation": 7,
                "availability": "available",
                "pressure": "normal",
                "safeTotalTokens": 53_248,
                "recommendedOutputTokens": 4_096,
                "maxPromptTokens": 49_152,
                "retainedSessionTokens": 49_152,
                "retainedBytes": 2_147_483_648_u64,
                "prefixCacheBytes": 1_073_741_824_u64,
                "basis": "learned"
            })
        );
    }

    #[test]
    fn known_but_unloaded_snapshot_is_unavailable_with_zero_token_fields() {
        let snapshot = CapacitySnapshot {
            availability: CapacityAvailability::Unavailable,
            safe_total_tokens: 0,
            recommended_output_tokens: 0,
            max_prompt_tokens: 0,
            retained_session_tokens: 0,
            retained_bytes: 0,
            prefix_cache_bytes: 0,
            basis: CapacityBasis::Conservative,
            ..available_snapshot()
        };

        let value = serde_json::to_value(snapshot).unwrap();
        assert_eq!(
            value,
            json!({
                "schemaVersion": 1,
                "model": MODEL,
                "modelFingerprint": FINGERPRINT,
                "bootId": BOOT_ID,
                "generation": 7,
                "availability": "unavailable",
                "pressure": "normal",
                "safeTotalTokens": 0,
                "recommendedOutputTokens": 0,
                "maxPromptTokens": 0,
                "retainedSessionTokens": 0,
                "retainedBytes": 0,
                "prefixCacheBytes": 0,
                "basis": "conservative"
            })
        );
    }

    #[test]
    fn capacity_enums_reject_unknown_values() {
        assert!(serde_json::from_str::<CapacityAvailability>("\"loading\"").is_err());
        assert!(serde_json::from_str::<MemoryPressure>("\"warning\"").is_err());
        assert!(serde_json::from_str::<CapacityBasis>("\"measured\"").is_err());
    }

    #[test]
    fn capacity_snapshot_rejects_non_v1_schema_at_deserialization() {
        let mut value = serde_json::to_value(available_snapshot()).unwrap();
        value["schemaVersion"] = json!(2);

        let error = serde_json::from_value::<CapacitySnapshot>(value).unwrap_err();
        assert_eq!(
            error.to_string(),
            "unsupported capacity schemaVersion 2; expected 1"
        );
    }

    #[test]
    fn generation_is_comparable_only_within_one_boot() {
        let current = available_snapshot();
        let same = available_snapshot();
        let restarted = CapacitySnapshot {
            boot_id: "01993654-8af2-7b31-a420-c52ebc349288".to_owned(),
            ..available_snapshot()
        };
        let advanced = CapacitySnapshot {
            generation: 8,
            ..available_snapshot()
        };

        assert!(current.is_same_revision(&same));
        assert!(!current.is_same_revision(&restarted));
        assert!(!current.is_same_revision(&advanced));
    }

    #[test]
    fn capacity_exceeded_matches_openai_shaped_413_body() {
        let body = CapacityErrorEnvelope::new(CapacityExceededError::new(
            36_864,
            40_960,
            BOOT_ID.to_owned(),
            8,
        ));

        assert_eq!(
            serde_json::to_value(body).unwrap(),
            json!({
                "error": {
                    "type": "higgs_capacity_exceeded",
                    "code": "compact_and_retry",
                    "safePromptTokens": 36_864,
                    "safeTotalTokens": 40_960,
                    "bootId": BOOT_ID,
                    "generation": 8
                }
            })
        );
    }

    #[test]
    fn capacity_unavailable_matches_openai_shaped_503_body() {
        let body = CapacityErrorEnvelope::new(CapacityUnavailableError::new(BOOT_ID.to_owned(), 9));

        assert_eq!(
            serde_json::to_value(body).unwrap(),
            json!({
                "error": {
                    "type": "higgs_capacity_unavailable",
                    "code": "capacity_unavailable",
                    "bootId": BOOT_ID,
                    "generation": 9,
                    "retryAfterMs": 5000
                }
            })
        );
    }

    #[test]
    fn capacity_interrupted_matches_terminal_sse_body() {
        let body =
            CapacityErrorEnvelope::new(CapacityInterruptedError::new(BOOT_ID.to_owned(), 10, 317));

        assert_eq!(
            serde_json::to_value(body).unwrap(),
            json!({
                "error": {
                    "type": "higgs_capacity_interrupted",
                    "code": "capacity_interrupted",
                    "bootId": BOOT_ID,
                    "generation": 10,
                    "partialOutputTokens": 317
                }
            })
        );
    }

    #[tokio::test]
    async fn typed_unknown_model_is_distinct_from_axum_legacy_route_absence() {
        use axum::{
            Router,
            body::Body,
            http::{Request, StatusCode},
        };
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let typed = serde_json::to_value(CapacityErrorEnvelope::new(
            CapacityModelNotFoundError::new("missing-model".to_owned()),
        ))
        .unwrap();
        let legacy = Router::new()
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=missing-model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(legacy.status(), StatusCode::NOT_FOUND);
        assert!(legacy.headers().get("content-type").is_none());
        let legacy_body = legacy.into_body().collect().await.unwrap().to_bytes();
        assert!(legacy_body.is_empty());

        assert_eq!(
            typed,
            json!({
                "error": {
                    "type": "higgs_capacity_model_not_found",
                    "code": "model_not_found",
                    "model": "missing-model"
                }
            })
        );
        assert_ne!(typed.to_string().as_bytes(), legacy_body.as_ref());
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    fn controller_inputs(memory_limit_gib: u64, metal_limit_gib: u64) -> CapacityInputs {
        CapacityInputs {
            memory: higgs_engine::MlxMemorySnapshot {
                active_bytes: 11 * GIB,
                peak_bytes: 11 * GIB,
                memory_limit_bytes: Some(memory_limit_gib * GIB),
                metal_recommended_working_set_bytes: Some(metal_limit_gib * GIB),
            },
            loaded_model_bytes: 11 * GIB,
            costs: higgs_engine::EngineCostDescription {
                fixed_live_session_bytes: 256 * 1024 * 1024,
                persistent_bytes_per_token: 20_480,
                decode_workspace_bytes: 256 * 1024 * 1024,
                transient_prefill: higgs_engine::TransientPrefillEstimate {
                    base_bytes: 2 * GIB,
                    bytes_per_prompt_token: 0,
                    bytes_per_chunk_token: 4 * 1024 * 1024,
                    max_prompt_tokens: 131_072,
                    max_chunk_tokens: 512,
                },
            },
            architectural_max_tokens: 131_072,
            prefill_chunk_tokens: 512,
            retained_bytes: 256 * 1024 * 1024,
            prefix_cache_bytes: 256 * 1024 * 1024,
            active_reservation_bytes: 0,
            configured_total_token_ceiling: None,
            configured_output_token_ceiling: None,
            pressure: MemoryPressure::Normal,
        }
    }

    fn request(prompt_tokens: u64, output_tokens: u64) -> RequestCost {
        RequestCost {
            execution_path: ExecutionPath::Cold,
            prompt_tokens,
            suffix_tokens: prompt_tokens,
            output_tokens,
            retained_growth_bytes: 64 * 1024 * 1024,
        }
    }

    fn learned_profile_key() -> LearnedProfileKey {
        LearnedProfileKey {
            hardware_identifier: "Mac15,9".into(),
            physical_memory_bytes: 64 * GIB,
            os_version: "15.6".into(),
            os_build: "24G90".into(),
            backend_authority_bytes: 48 * GIB,
            higgs_build: "abc123".into(),
            model_fingerprint: FINGERPRINT.into(),
            quantization: "3bit".into(),
            execution_mode: "native".into(),
            kv_representation: "fp16-hybrid".into(),
            prefill_model_identity: Some("prefill-v1".into()),
            execution_cache_fingerprint: "native;kv=fp16;radix=v1".into(),
            drafter_identity: None,
        }
    }

    #[test]
    fn controller_uses_smaller_nonzero_authority_and_protects_os_reserve() {
        let inputs = controller_inputs(28, 24);
        assert_eq!(inputs.working_set_authority_bytes(), Some(24 * GIB));
        assert_eq!(inputs.protected_reserve_bytes(), Some(24 * GIB / 5));

        let reversed = controller_inputs(20, 24);
        assert_eq!(reversed.working_set_authority_bytes(), Some(20 * GIB));
        assert_eq!(reversed.protected_reserve_bytes(), Some(4 * GIB));
    }

    #[test]
    fn missing_or_zero_authority_is_unavailable() {
        let mut inputs = controller_inputs(24, 24);
        inputs.memory.memory_limit_bytes = None;
        inputs.memory.metal_recommended_working_set_bytes = None;
        assert_eq!(
            CapacityController::new(inputs).decision().availability,
            CapacityAvailability::Unavailable
        );

        inputs.memory.memory_limit_bytes = Some(0);
        assert_eq!(
            CapacityController::new(inputs).decision().availability,
            CapacityAvailability::Unavailable
        );
    }

    #[test]
    fn numeric_configuration_only_lowers_automatic_capacity() {
        let automatic = CapacityController::new(controller_inputs(48, 48)).decision();
        let mut lower = controller_inputs(48, 48);
        lower.configured_total_token_ceiling = Some(16_384);
        let lowered = CapacityController::new(lower).decision();
        assert_eq!(lowered.safe_total_tokens, 16_384);
        assert!(lowered.safe_total_tokens < automatic.safe_total_tokens);

        let mut higher = controller_inputs(48, 48);
        higher.configured_total_token_ceiling = Some(automatic.safe_total_tokens + 16_384);
        assert_eq!(
            CapacityController::new(higher).decision().safe_total_tokens,
            automatic.safe_total_tokens
        );

        let mut small_output = controller_inputs(48, 48);
        small_output.configured_output_token_ceiling = Some(512);
        assert_eq!(
            CapacityController::new(small_output)
                .decision()
                .recommended_output_tokens,
            512
        );
    }

    #[test]
    fn normal_startup_has_no_invented_token_discount() {
        let controller = CapacityController::new(controller_inputs(48, 48));
        assert_eq!(controller.decision().safe_total_tokens, 131_072);
        assert!(matches!(
            controller.admit(request(127_076, 3996)),
            Admission::Admitted(_)
        ));
    }

    #[test]
    fn one_checked_ledger_accounts_every_cost_and_output_reserve() {
        let mut inputs = controller_inputs(24, 24);
        inputs.active_reservation_bytes = 128 * 1024 * 1024;
        let controller = CapacityController::new(inputs);
        let ledger = controller.byte_ledger(request(49_152, 4_096)).unwrap();
        assert_eq!(ledger.loaded_baseline_bytes, 11 * GIB);
        assert_eq!(ledger.fixed_session_bytes, 256 * 1024 * 1024);
        assert_eq!(ledger.prompt_bytes, 49_152 * 20_480);
        assert_eq!(ledger.output_bytes, 4_096 * 20_480);
        assert_eq!(ledger.decode_bytes, 256 * 1024 * 1024);
        assert_eq!(ledger.retained_and_cache_bytes, 704 * 1024 * 1024);
        assert_eq!(ledger.transient_bytes, 4 * GIB);
        assert!(ledger.total_bytes().is_some());

        let short_output = controller.byte_ledger(request(49_152, 512)).unwrap();
        assert_eq!(short_output.output_bytes, 512 * 20_480);

        let without_output = controller.byte_ledger(request(49_152, 0)).unwrap();
        assert_eq!(without_output.output_bytes, 0);
        assert_eq!(
            ledger.total_bytes().unwrap() - without_output.total_bytes().unwrap(),
            4_096 * 20_480
        );
    }

    #[test]
    fn minimum_completion_byte_reserve_fails_closed_near_the_boundary() {
        let mut inputs = controller_inputs(5, 5);
        let usable_bytes = 1280 * 1024 * 1024;
        inputs.memory = MlxMemorySnapshot {
            active_bytes: 0,
            peak_bytes: 0,
            memory_limit_bytes: Some(4 * GIB + usable_bytes),
            metal_recommended_working_set_bytes: Some(4 * GIB + usable_bytes),
        };
        inputs.loaded_model_bytes = 0;
        inputs.costs.fixed_live_session_bytes = 0;
        inputs.costs.persistent_bytes_per_token = 1024 * 1024;
        inputs.costs.decode_workspace_bytes = 0;
        inputs.costs.transient_prefill.base_bytes = 0;
        inputs.costs.transient_prefill.bytes_per_chunk_token = 0;
        inputs.retained_bytes = 0;
        inputs.prefix_cache_bytes = 0;
        inputs.configured_total_token_ceiling = Some(1024);
        inputs.configured_output_token_ceiling = Some(512);

        let controller = CapacityController::new(inputs);
        assert_eq!(
            controller.decision().availability,
            CapacityAvailability::Unavailable
        );
    }

    #[test]
    fn published_limits_and_admission_use_same_aligned_inequality() {
        let controller = CapacityController::new(controller_inputs(24, 24));
        let decision = controller.decision();
        assert_eq!(decision.safe_total_tokens % 1024, 0);
        assert_eq!(
            decision.max_prompt_tokens + decision.recommended_output_tokens,
            decision.safe_total_tokens
        );
        assert!(matches!(
            controller.admit(request(
                decision.max_prompt_tokens,
                decision.recommended_output_tokens
            )),
            Admission::Admitted(_)
        ));
        assert!(matches!(
            controller.admit(request(
                decision.max_prompt_tokens + 1024,
                decision.recommended_output_tokens
            )),
            Admission::Exceeded(_)
        ));
    }

    #[test]
    fn escha_geometry_reproduces_injected_32_and_64_gib_relationships() {
        assert_eq!(49_152 * 20_480, 960 * 1024 * 1024);
        let mut tier_32_inputs = controller_inputs(32, 24);
        tier_32_inputs.retained_bytes = 2 * GIB;
        tier_32_inputs.prefix_cache_bytes = GIB;
        let mut tier_64_inputs = controller_inputs(64, 48);
        tier_64_inputs.retained_bytes = 2 * GIB;
        tier_64_inputs.prefix_cache_bytes = GIB;
        let tier_32 = CapacityController::new(tier_32_inputs).decision();
        let tier_64 = CapacityController::new(tier_64_inputs).decision();
        assert!(tier_32.safe_total_tokens >= 1024);
        assert!(tier_64.safe_total_tokens > tier_32.safe_total_tokens);
    }

    #[test]
    fn checked_overflow_fails_unavailable() {
        let mut inputs = controller_inputs(24, 24);
        inputs.costs.persistent_bytes_per_token = u64::MAX;
        assert_eq!(
            CapacityController::new(inputs).decision().availability,
            CapacityAvailability::Unavailable
        );
    }

    #[test]
    fn pressure_downshift_uses_exact_aligned_floors() {
        let mut controller = CapacityController::new(controller_inputs(48, 48));
        let previous = controller.decision().safe_total_tokens;
        let warning_raw = controller.recompute_for_pressure(MemoryPressure::Constrained, None);
        assert_eq!(
            warning_raw.safe_total_tokens,
            floor_1024(previous * 75 / 100)
        );

        let before_critical = warning_raw.safe_total_tokens;
        let critical = controller.recompute_for_pressure(MemoryPressure::Critical, None);
        assert_eq!(
            critical.safe_total_tokens,
            floor_1024(before_critical * 50 / 100)
        );
        assert_eq!(critical.availability, CapacityAvailability::Unavailable);
    }

    #[test]
    fn critical_construction_and_restore_recover_bounded_on_constrained_observation() {
        let normal_inputs = controller_inputs(48, 48);
        let full_static = CapacityController::new(normal_inputs)
            .decision()
            .safe_total_tokens;
        let mut critical_inputs = normal_inputs;
        critical_inputs.pressure = MemoryPressure::Critical;

        let mut trained = CapacityController::new(normal_inputs);
        trained.observe(AllocationObservation::clean(
            ExecutionPath::Cold,
            8192,
            512,
            GIB,
            2 * GIB,
        ));
        let key = learned_profile_key();
        let profile = trained.export_profile(key.clone(), 8 * GIB).unwrap();

        let mut constructed = CapacityController::new(critical_inputs);
        let mut restored = CapacityController::new(critical_inputs);
        assert!(restored.restore_profile(&profile, &key, 8 * GIB));

        for controller in [&mut constructed, &mut restored] {
            let recovered = controller.apply_pressure_observation(PressureObservation {
                pressure: MemoryPressure::Constrained,
                swap_out_delta: 0,
                compressor_delta: 0,
            });
            assert_eq!(recovered.safe_total_tokens, 8_192);
            assert_eq!(recovered.availability, CapacityAvailability::Available);
            assert!(recovered.safe_total_tokens < full_static);
        }
    }

    #[test]
    fn direct_critical_recompute_marks_constrained_observation_as_recovery() {
        let mut controller = CapacityController::new(controller_inputs(48, 48));
        let critical = controller.recompute_for_pressure(MemoryPressure::Critical, None);
        assert_eq!(critical.safe_total_tokens, 65_536);

        let constrained = controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(constrained.safe_total_tokens, 65_536);
        assert_eq!(constrained.availability, CapacityAvailability::Available);
    }

    #[test]
    fn direct_normal_recompute_ends_episode_before_fresh_warning() {
        let mut controller = CapacityController::new(controller_inputs(48, 48));
        let critical = controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Critical,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(critical.safe_total_tokens, 65_536);
        let normal = controller.recompute_for_pressure(MemoryPressure::Normal, None);
        assert_eq!(normal.safe_total_tokens, 65_536);

        let warning = controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(warning.safe_total_tokens, 49_152);
        assert_eq!(warning.availability, CapacityAvailability::Available);
    }

    #[derive(Clone)]
    struct TestClock(std::sync::Arc<std::sync::atomic::AtomicU64>);

    impl Clock for TestClock {
        fn now_millis(&self) -> u64 {
            self.0.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl TestClock {
        fn set_seconds(&self, seconds: u64) {
            self.0
                .store(seconds * 1000, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[test]
    fn learning_requires_three_clean_allocation_observations_over_five_minutes() {
        let clock = TestClock(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        let mut inputs = controller_inputs(24, 24);
        inputs.costs.transient_prefill.base_bytes = 4 * GIB;
        let mut controller = CapacityController::with_clock(inputs, clock.clone());
        let initial = controller.decision().safe_total_tokens;
        let band_prompt = initial.saturating_sub(4096);
        let before_transient = controller
            .byte_ledger(request(band_prompt, 4096))
            .unwrap()
            .transient_bytes;

        for seconds in [0, 150] {
            clock.set_seconds(seconds);
            controller.observe(AllocationObservation::clean(
                ExecutionPath::Cold,
                band_prompt,
                512,
                6 * GIB,
                2 * GIB,
            ));
            assert_eq!(controller.decision().safe_total_tokens, initial);
        }
        clock.set_seconds(300);
        controller.observe(AllocationObservation::clean(
            ExecutionPath::Cold,
            band_prompt,
            512,
            6 * GIB,
            2 * GIB,
        ));
        let raised = controller.decision().safe_total_tokens;
        assert!(raised > initial);
        assert_eq!(before_transient, 6 * GIB);
        assert_eq!(
            controller
                .byte_ledger(request(band_prompt, 4096))
                .unwrap()
                .transient_bytes,
            2_362_232_013
        );
        assert!(raised - initial <= 4096);
        assert!(raised - initial <= floor_1024(initial / 8).max(1024));
    }

    #[test]
    fn retained_hits_never_lower_cold_evidence_and_pressure_freezes_rises() {
        let clock = TestClock(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        let mut controller =
            CapacityController::with_clock(controller_inputs(48, 48), clock.clone());
        let initial = controller.decision().safe_total_tokens;
        for (index, seconds) in [0, 150, 300].into_iter().enumerate() {
            clock.set_seconds(seconds);
            controller.observe(AllocationObservation::clean(
                ExecutionPath::RetainedSuffix,
                initial,
                64,
                1,
                1,
            ));
            assert_eq!(
                controller.decision().safe_total_tokens,
                initial,
                "hit {index} must not train cold capacity"
            );
        }
        controller.recompute_for_pressure(MemoryPressure::Constrained, None);
        let constrained = controller.decision().safe_total_tokens;
        for seconds in [301, 451, 601] {
            clock.set_seconds(seconds);
            controller.observe(AllocationObservation::clean(
                ExecutionPath::Cold,
                constrained,
                512,
                1,
                1,
            ));
        }
        assert_eq!(controller.decision().safe_total_tokens, constrained);
    }

    #[test]
    fn cold_evidence_cannot_skip_an_unobserved_prompt_band() {
        let clock = TestClock(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        let mut controller =
            CapacityController::with_clock(controller_inputs(48, 48), clock.clone());
        let initial = controller.decision().safe_total_tokens;
        let lower_band_prompt = initial / 4;
        for seconds in [0, 150, 300] {
            clock.set_seconds(seconds);
            controller.observe(AllocationObservation::clean(
                ExecutionPath::Cold,
                lower_band_prompt,
                512,
                1,
                1,
            ));
        }
        assert_eq!(controller.decision().safe_total_tokens, initial);
    }

    #[test]
    fn retained_and_radix_underprediction_train_only_retained_evidence() {
        let mut controller = CapacityController::new(controller_inputs(24, 24));
        let prompt = controller.decision().safe_total_tokens;
        let before = controller.decision().safe_total_tokens;
        let mut observation =
            AllocationObservation::clean(ExecutionPath::RetainedSuffix, prompt, 64, GIB, 8 * GIB);
        observation.observed_retained_bytes = 8 * GIB;
        controller.observe(observation);
        assert_eq!(
            controller.learned_high_water_bytes(ExecutionPath::RadixHit, prompt),
            Some(9_448_928_052)
        );
        assert_eq!(
            controller.learned_high_water_bytes(ExecutionPath::Cold, prompt),
            None
        );
        assert!(controller.decision().safe_total_tokens < before);
    }

    #[test]
    fn retained_request_charges_suffix_prefill_but_full_logical_kv() {
        let mut inputs = controller_inputs(48, 48);
        inputs.costs.transient_prefill.base_bytes = 0;
        inputs.costs.transient_prefill.bytes_per_prompt_token = 1024;
        inputs.costs.transient_prefill.bytes_per_chunk_token = 0;
        let controller = CapacityController::new(inputs);
        let cold = controller.byte_ledger(request(4096, 1024)).unwrap();
        let retained = controller
            .byte_ledger(RequestCost {
                execution_path: ExecutionPath::RetainedSuffix,
                suffix_tokens: 64,
                ..request(4096, 1024)
            })
            .unwrap();
        assert_eq!(cold.prompt_bytes, retained.prompt_bytes);
        assert_eq!(cold.transient_bytes, 4096 * 1024);
        assert_eq!(retained.transient_bytes, 64 * 1024);
    }

    #[test]
    fn allocator_underprediction_raises_high_water_ten_percent_and_recomputes() {
        let mut controller = CapacityController::new(controller_inputs(24, 24));
        let before = controller.decision().safe_total_tokens;
        let observed_peak = 6 * GIB;
        controller.observe(AllocationObservation::clean(
            ExecutionPath::Cold,
            before,
            512,
            GIB,
            observed_peak,
        ));
        assert_eq!(
            controller.learned_high_water_bytes(ExecutionPath::Cold, before),
            Some(7_086_696_039)
        );
        assert!(controller.decision().safe_total_tokens < before);
    }

    #[test]
    fn dirty_observation_and_pressure_reset_every_clean_window() {
        let clock = TestClock(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        let mut inputs = controller_inputs(24, 24);
        inputs.costs.transient_prefill.base_bytes = 4 * GIB;
        let mut controller = CapacityController::with_clock(inputs, clock.clone());
        let initial = controller.decision().safe_total_tokens;

        for seconds in [0, 150] {
            clock.set_seconds(seconds);
            controller.observe(AllocationObservation::clean(
                ExecutionPath::Cold,
                initial,
                512,
                6 * GIB,
                2 * GIB,
            ));
        }
        let mut dirty =
            AllocationObservation::clean(ExecutionPath::Cold, initial, 512, 6 * GIB, 2 * GIB);
        dirty.swap_out_delta = 1;
        clock.set_seconds(300);
        controller.observe(dirty);
        controller.recompute_for_pressure(MemoryPressure::Constrained, None);
        controller.recompute_for_pressure(MemoryPressure::Normal, None);
        let after_pressure = controller.decision().safe_total_tokens;

        for seconds in [301, 451] {
            clock.set_seconds(seconds);
            controller.observe(AllocationObservation::clean(
                ExecutionPath::Cold,
                after_pressure,
                512,
                6 * GIB,
                2 * GIB,
            ));
        }
        assert_eq!(controller.decision().safe_total_tokens, after_pressure);
    }

    #[test]
    fn exceeded_bound_rewrites_to_an_admissible_request_for_every_path() {
        for execution_path in [
            ExecutionPath::Cold,
            ExecutionPath::RetainedSuffix,
            ExecutionPath::RadixHit,
        ] {
            let mut controller = CapacityController::new(controller_inputs(24, 24));
            let observed_prompt = controller.decision().max_prompt_tokens;
            controller.observe(AllocationObservation::clean(
                execution_path,
                observed_prompt,
                512,
                GIB,
                4 * GIB,
            ));
            let oversized = RequestCost {
                execution_path,
                prompt_tokens: 131_072,
                suffix_tokens: 512,
                output_tokens: 4096,
                retained_growth_bytes: 64 * 1024 * 1024,
            };
            let Admission::Exceeded(bound) = controller.admit(oversized) else {
                panic!("oversized {execution_path:?} request was not rejected")
            };
            let rewritten = RequestCost {
                prompt_tokens: bound.max_prompt_tokens,
                suffix_tokens: 512.min(bound.max_prompt_tokens),
                output_tokens: bound.recommended_output_tokens,
                ..oversized
            };
            assert!(matches!(
                controller.admit(rewritten),
                Admission::Admitted(_)
            ));
        }
    }

    #[test]
    fn absolute_active_memory_charges_only_unaccounted_residual() {
        let mut inputs = controller_inputs(24, 24);
        inputs.active_reservation_bytes = 128 * 1024 * 1024;
        let known_beyond_model = inputs
            .costs
            .fixed_live_session_bytes
            .checked_add(inputs.retained_bytes)
            .unwrap()
            .checked_add(inputs.prefix_cache_bytes)
            .unwrap()
            .checked_add(inputs.active_reservation_bytes)
            .unwrap();
        inputs.memory.active_bytes = inputs.loaded_model_bytes + known_beyond_model;
        let controller = CapacityController::new(inputs);
        let ledger = controller
            .byte_ledger(RequestCost {
                retained_growth_bytes: 0,
                ..request(1024, 1024)
            })
            .unwrap();
        assert_eq!(ledger.loaded_baseline_bytes, 11 * GIB);
        assert_eq!(ledger.unaccounted_active_bytes, 0);
        assert_eq!(ledger.retained_and_cache_bytes, 640 * 1024 * 1024);
    }

    #[test]
    fn configured_output_ceiling_is_enforced_during_admission() {
        let mut inputs = controller_inputs(48, 48);
        inputs.configured_output_token_ceiling = Some(2048);
        let controller = CapacityController::new(inputs);
        let Admission::Exceeded(bound) = controller.admit(request(1024, 3072)) else {
            panic!("configured output ceiling was not enforced")
        };
        assert_eq!(bound.recommended_output_tokens, 2048);
        assert!(matches!(
            controller.admit(request(1024, 2048)),
            Admission::Admitted(_)
        ));
    }

    #[test]
    fn critical_is_intrinsic_to_construction_restore_and_admission() {
        let mut inputs = controller_inputs(48, 48);
        inputs.pressure = MemoryPressure::Critical;
        let full_static = CapacityController::new(controller_inputs(48, 48))
            .decision()
            .safe_total_tokens;
        let clock = TestClock(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        let mut critical = CapacityController::with_clock(inputs, clock.clone());
        assert_eq!(
            critical.decision().availability,
            CapacityAvailability::Unavailable
        );
        assert!(matches!(
            critical.admit(request(1024, 1024)),
            Admission::Unavailable
        ));
        let recovered = critical.recompute_for_pressure(MemoryPressure::Normal, None);
        assert_eq!(recovered.availability, CapacityAvailability::Available);
        assert!(recovered.safe_total_tokens > 0);
        assert!(recovered.safe_total_tokens < full_static);
        for seconds in [0, 150] {
            clock.set_seconds(seconds);
            critical.observe(AllocationObservation::clean(
                ExecutionPath::Cold,
                recovered.max_prompt_tokens,
                recovered.max_prompt_tokens,
                4 * GIB,
                4 * GIB,
            ));
            assert_eq!(critical.decision(), recovered);
        }
        clock.set_seconds(300);
        critical.observe(AllocationObservation::clean(
            ExecutionPath::Cold,
            recovered.max_prompt_tokens,
            recovered.max_prompt_tokens,
            4 * GIB,
            4 * GIB,
        ));
        let raised = critical.decision();
        assert_eq!(
            raised.safe_total_tokens - recovered.safe_total_tokens,
            floor_1024(recovered.safe_total_tokens / 8).min(4096)
        );
        assert!(raised.safe_total_tokens < full_static);

        let mut trained = CapacityController::new(controller_inputs(48, 48));
        trained.observe(AllocationObservation::clean(
            ExecutionPath::Cold,
            8192,
            512,
            GIB,
            2 * GIB,
        ));
        let key = learned_profile_key();
        let profile = trained.export_profile(key.clone(), 8 * GIB).unwrap();
        let restore_clock = TestClock(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        let mut critical_restore = CapacityController::with_clock(inputs, restore_clock);
        assert!(critical_restore.restore_profile(&profile, &key, 8 * GIB));
        assert_eq!(
            critical_restore.decision().availability,
            CapacityAvailability::Unavailable
        );
        let restored_recovery =
            critical_restore.recompute_for_pressure(MemoryPressure::Normal, None);
        assert_eq!(
            restored_recovery.availability,
            CapacityAvailability::Available
        );
        assert!(restored_recovery.safe_total_tokens > 0);
        assert!(restored_recovery.safe_total_tokens < full_static);
    }

    #[test]
    fn request_shape_failure_is_exceeded_while_model_capacity_is_available() {
        let controller = CapacityController::new(controller_inputs(48, 48));
        assert!(matches!(
            controller.admit(request(1024, 512)),
            Admission::Admitted(_)
        ));
        assert!(matches!(
            controller.admit(request(0, 200_000)),
            Admission::Exceeded(_)
        ));
        let fixed = RequestCost {
            retained_growth_bytes: u64::MAX,
            ..request(1024, 1024)
        };
        match controller.admit(fixed) {
            Admission::FixedCostUnavailable => {}
            Admission::Exceeded(bound) => assert!(matches!(
                controller.admit(RequestCost {
                    prompt_tokens: bound.max_prompt_tokens,
                    output_tokens: bound.recommended_output_tokens,
                    ..fixed
                }),
                Admission::Admitted(_)
            )),
            other => panic!("unexpected fixed-cost admission: {other:?}"),
        }
    }

    #[test]
    fn component_high_water_recomputes_publication_without_aggregate_underprediction() {
        for component in [ExecutionPath::RetainedSuffix, ExecutionPath::RadixHit] {
            let mut controller = CapacityController::new(controller_inputs(24, 24));
            let before = controller.decision();
            let mut observation = AllocationObservation::clean(
                component,
                before.max_prompt_tokens,
                512,
                8 * GIB,
                8 * GIB,
            );
            if component == ExecutionPath::RetainedSuffix {
                observation.observed_suffix_transient_bytes = 8 * GIB;
            } else {
                observation.observed_retained_bytes = 8 * GIB;
            }
            controller.observe(observation);
            let published = controller.decision();
            assert!(published.safe_total_tokens < before.safe_total_tokens);
            for path in [
                ExecutionPath::Cold,
                ExecutionPath::RetainedSuffix,
                ExecutionPath::RadixHit,
            ] {
                assert!(matches!(
                    controller.admit(RequestCost {
                        execution_path: path,
                        prompt_tokens: published.max_prompt_tokens,
                        suffix_tokens: published.max_prompt_tokens,
                        output_tokens: published.recommended_output_tokens,
                        retained_growth_bytes: 0,
                    }),
                    Admission::Admitted(_)
                ));
            }
        }
    }

    #[test]
    fn zero_cold_observations_cannot_replace_static_transient() {
        let clock = TestClock(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        let mut inputs = controller_inputs(24, 24);
        inputs.costs.transient_prefill.base_bytes = 4 * GIB;
        let mut controller = CapacityController::with_clock(inputs, clock.clone());
        let prompt = controller.decision().max_prompt_tokens;
        let before = controller.byte_ledger(request(prompt, 4096)).unwrap();
        for seconds in [0, 150, 300] {
            clock.set_seconds(seconds);
            controller.observe(AllocationObservation::clean(
                ExecutionPath::Cold,
                prompt,
                512,
                6 * GIB,
                0,
            ));
        }
        assert_eq!(controller.decision().max_prompt_tokens, prompt);
        assert_eq!(
            controller
                .byte_ledger(request(prompt, 4096))
                .unwrap()
                .transient_bytes,
            before.transient_bytes
        );
    }

    #[test]
    fn odd_high_water_rounds_ten_percent_up_and_non_allocations_do_not_train() {
        let mut controller = CapacityController::new(controller_inputs(48, 48));
        let prompt = controller.decision().safe_total_tokens;
        let mut non_allocation =
            AllocationObservation::clean(ExecutionPath::Cold, prompt, 512, 10, 11);
        non_allocation.allocation_bearing = false;
        controller.observe(non_allocation);
        assert_eq!(
            controller.learned_high_water_bytes(ExecutionPath::Cold, prompt),
            None
        );
        controller.observe(AllocationObservation::clean(
            ExecutionPath::Cold,
            prompt,
            512,
            10,
            11,
        ));
        assert_eq!(
            controller.learned_high_water_bytes(ExecutionPath::Cold, prompt),
            Some(13)
        );
    }

    #[test]
    fn allocator_underprediction_restarts_clean_qualification() {
        let clock = TestClock(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        let mut inputs = controller_inputs(24, 24);
        inputs.costs.transient_prefill.base_bytes = 4 * GIB;
        let mut controller = CapacityController::with_clock(inputs, clock.clone());
        let prompt = controller.decision().max_prompt_tokens;
        for seconds in [0, 150] {
            clock.set_seconds(seconds);
            controller.observe(AllocationObservation::clean(
                ExecutionPath::Cold,
                prompt,
                512,
                6 * GIB,
                2 * GIB,
            ));
        }
        clock.set_seconds(300);
        controller.observe(AllocationObservation::clean(
            ExecutionPath::Cold,
            prompt,
            512,
            GIB,
            3 * GIB,
        ));
        let after_underprediction = controller.decision();
        clock.set_seconds(600);
        controller.observe(AllocationObservation::clean(
            ExecutionPath::Cold,
            prompt,
            512,
            6 * GIB,
            2 * GIB,
        ));
        assert_eq!(controller.decision(), after_underprediction);
    }

    #[test]
    fn suffix_evidence_changes_transient_without_inventing_retained_residency() {
        let mut controller = CapacityController::new(controller_inputs(48, 48));
        let prompt = controller.decision().max_prompt_tokens;
        let request = RequestCost {
            execution_path: ExecutionPath::RetainedSuffix,
            prompt_tokens: prompt,
            suffix_tokens: 512,
            output_tokens: 4096,
            retained_growth_bytes: 0,
        };
        let before = controller.byte_ledger(request).unwrap();
        let mut observation =
            AllocationObservation::clean(ExecutionPath::RetainedSuffix, prompt, 512, GIB, 6 * GIB);
        observation.observed_retained_bytes = 0;
        observation.observed_suffix_transient_bytes = 6 * GIB;
        controller.observe(observation);
        let after = controller.byte_ledger(request).unwrap();
        assert_eq!(after.learned_retained_bytes, 0);
        assert!(after.transient_bytes > before.transient_bytes);
    }

    #[test]
    fn learned_profile_roundtrip_changes_decision_and_mismatch_is_transactional() {
        let clock = TestClock(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        let mut inputs = controller_inputs(24, 24);
        inputs.costs.transient_prefill.base_bytes = 4 * GIB;
        let mut trained = CapacityController::with_clock(inputs, clock.clone());
        let initial = trained.decision().safe_total_tokens;
        for seconds in [0, 150, 300] {
            clock.set_seconds(seconds);
            trained.observe(AllocationObservation::clean(
                ExecutionPath::Cold,
                initial,
                512,
                6 * GIB,
                2 * GIB,
            ));
        }
        let key = learned_profile_key();
        let profile = trained.export_profile(key.clone(), 8 * GIB).unwrap();
        let mut restored = CapacityController::new(inputs);
        let before_restore = restored.decision();
        assert!(restored.restore_profile(&profile, &key, 8 * GIB));
        assert!(restored.decision().safe_total_tokens > before_restore.safe_total_tokens);

        let mut wrong_key = key.clone();
        wrong_key.prefill_model_identity = Some("different".into());
        let after_restore = restored.decision();
        assert!(!restored.restore_profile(&profile, &wrong_key, 8 * GIB));
        assert_eq!(restored.decision(), after_restore);
        assert!(!restored.restore_profile(&profile, &key, 8 * GIB - 1));
        assert_eq!(restored.decision(), after_restore);

        let invalid = LearnedProfile::new(
            key.clone(),
            8 * GIB,
            vec![LearnedBandEvidence {
                prompt_band: 65_536,
                cold_high_water_bytes: 0,
                cold_replacement_qualified: true,
                retained_high_water_bytes: GIB,
                suffix_high_water_bytes: 0,
            }],
        );
        assert!(!restored.restore_profile(&invalid, &key, 8 * GIB));
        assert_eq!(restored.decision(), after_restore);
    }

    #[test]
    fn restored_suffix_high_water_changes_retained_admission() {
        let inputs = controller_inputs(24, 24);
        let mut trained = CapacityController::new(inputs);
        let prompt = trained.decision().max_prompt_tokens;
        let mut observation =
            AllocationObservation::clean(ExecutionPath::RetainedSuffix, prompt, 64, GIB, 8 * GIB);
        observation.observed_suffix_transient_bytes = 8 * GIB;
        trained.observe(observation);
        let key = learned_profile_key();
        let profile = trained.export_profile(key.clone(), 8 * GIB).unwrap();

        let request = RequestCost {
            execution_path: ExecutionPath::RetainedSuffix,
            prompt_tokens: prompt,
            suffix_tokens: 64,
            output_tokens: 4096,
            retained_growth_bytes: 0,
        };
        let fresh = CapacityController::new(inputs);
        assert!(matches!(fresh.admit(request), Admission::Admitted(_)));
        let mut restored = CapacityController::new(inputs);
        assert!(restored.restore_profile(&profile, &key, 8 * GIB));
        assert!(!matches!(restored.admit(request), Admission::Admitted(_)));
    }

    #[test]
    fn restored_published_boundary_admits_every_execution_path() {
        let clock = TestClock(std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        let mut inputs = controller_inputs(24, 24);
        inputs.costs.transient_prefill.base_bytes = 4 * GIB;
        let mut trained = CapacityController::with_clock(inputs, clock.clone());
        let prompt = trained.decision().max_prompt_tokens;
        for seconds in [0, 150, 300] {
            clock.set_seconds(seconds);
            trained.observe(AllocationObservation::clean(
                ExecutionPath::Cold,
                prompt,
                512,
                6 * GIB,
                2 * GIB,
            ));
        }
        let mut retained =
            AllocationObservation::clean(ExecutionPath::RetainedSuffix, prompt, 512, GIB, 2 * GIB);
        retained.observed_retained_bytes = 128 * 1024 * 1024;
        retained.observed_suffix_transient_bytes = 2 * GIB;
        trained.observe(retained);
        let key = learned_profile_key();
        let profile = trained.export_profile(key.clone(), 8 * GIB).unwrap();
        let mut restored = CapacityController::new(inputs);
        assert!(restored.restore_profile(&profile, &key, 8 * GIB));
        let published = restored.decision();
        for execution_path in [
            ExecutionPath::Cold,
            ExecutionPath::RetainedSuffix,
            ExecutionPath::RadixHit,
        ] {
            let boundary = RequestCost {
                execution_path,
                prompt_tokens: published.max_prompt_tokens,
                suffix_tokens: published.max_prompt_tokens,
                output_tokens: published.recommended_output_tokens,
                retained_growth_bytes: 0,
            };
            assert!(
                matches!(restored.admit(boundary), Admission::Admitted(_)),
                "{execution_path:?}"
            );
        }
    }

    #[test]
    fn each_controller_gets_a_new_boot_id() {
        let one = CapacityController::new(controller_inputs(24, 24));
        let two = CapacityController::new(controller_inputs(24, 24));
        assert_ne!(one.boot_id(), two.boot_id());
        assert!(uuid::Uuid::parse_str(one.boot_id()).is_ok());
    }
}
