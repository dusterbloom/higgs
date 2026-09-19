#[cfg(test)]
use higgs_engine::MlxMemorySnapshot;
use serde::{Deserialize, Serialize};

#[allow(dead_code)] // Task 5 attaches this completed lifecycle to AppState.
pub(crate) mod pressure;
mod registry;

pub use registry::{
    ActiveRegistration, CacheAllocationPlan, CacheCapabilities, CapacityRegistry,
    DrainRegistration, ModelCapacityFacts, ModelContentIdentity, PublishedMemoryMeasurement,
    RegistrationError, RegistrationTicket, fingerprint_model_artifacts,
};
pub(crate) use registry::{CapacityAdmissionError, RequestReservation};

/// Owned process observer; server shutdown must consume and join it.
#[must_use = "the process pressure observer must be stopped and joined"]
pub struct CapacityPressureObserver(Option<pressure::PressureObserverHandle>);

/// Observer sink records pressure as diagnostics without changing admission.
pub struct CapacityPressureCoordinator {
    capacity: std::sync::Arc<CapacityRegistry>,
}

impl CapacityPressureCoordinator {
    #[must_use]
    pub fn new(capacity: std::sync::Arc<CapacityRegistry>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self { capacity })
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

/// Explicit output requests are checked unchanged. An omitted output budget
/// uses the configured default, bounded by the context left after tokenization.
pub(crate) fn resolve_output_tokens(
    state: &crate::state::SharedState,
    model: &str,
    prompt_tokens: usize,
    requested: Option<u32>,
    default: u32,
) -> u32 {
    if let Some(requested) = requested {
        return requested;
    }
    #[cfg(test)]
    ensure_route_test_capacity(state, model);
    let default = default.min(state.config.server.max_tokens);
    state.capacity.snapshot(model).map_or(default, |snapshot| {
        let remaining = snapshot
            .safe_total_tokens
            .saturating_sub(prompt_tokens as u64);
        default.min(u32::try_from(remaining).unwrap_or(u32::MAX))
    })
}

/// Check fixed context before any generation or retained-session mutation.
pub(crate) async fn admit_generation_request(
    state: &crate::state::SharedState,
    model: &str,
    prompt_tokens: usize,
    output_tokens: u32,
) -> Result<RequestReservation, crate::error::ServerError> {
    #[cfg(test)]
    ensure_route_test_capacity(state, model);
    // Context counts all tokens, including retained/cached prefixes.
    state
        .capacity
        .reserve_request(
            model,
            RequestCost {
                prompt_tokens: prompt_tokens as u64,
                output_tokens: u64::from(output_tokens),
            },
        )
        .await
        .map_err(capacity_server_error)
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
pub(crate) fn route_test_facts(
    model: &str,
    _memory: MlxMemorySnapshot,
    _persistent_bytes_per_token: u64,
    token_ceiling: u64,
) -> ModelCapacityFacts {
    ModelCapacityFacts {
        model: model.to_owned(),
        model_fingerprint: format!("sha256:test-{model}"),
        architectural_max_tokens: 1_048_576,
        retained_session_tokens: 0,
        retained_bytes_ceiling: 0,
        prefix_cache_bytes_ceiling: 0,
        cache_capabilities: CacheCapabilities {
            retained_sessions: false,
            prefix_cache: false,
        },
        configured_total_token_ceiling: Some(token_ceiling),
        configured_output_token_ceiling: Some(token_ceiling),
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

/// Route-test state with a narrow fixed context and retained continuation support.
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
    let facts = route_test_facts(model, memory, MIB, 3_072);
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
/// observe it at every bounded prefill chunk and decode step; disconnect and
/// model drain are signalled through the same handle by the registry.
pub(crate) fn install_reservation_stop(
    reservation: &RequestReservation,
    watchdog: Option<std::time::Duration>,
) -> higgs_engine::stop::GenerationStopGuard {
    let stop = reservation.stop();
    stop.set_watchdog(watchdog);
    higgs_engine::stop::install_generation_stop(stop)
}

/// Hold the model lifetime and stop signal through generation, including errors.
pub(crate) fn run_reserved_generation<T, E>(
    reservation: RequestReservation,
    watchdog: Option<std::time::Duration>,
    generate: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let _stop_guard = install_reservation_stop(&reservation, watchdog);
    generate()
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

/// Whether the model is loaded and accepting requests.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityAvailability {
    Available,
    Unavailable,
}

/// Process memory pressure published for diagnostics only.
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
    /// Delta of Darwin's cumulative compression-activity counter.
    pub compressor_delta: u64,
}

/// Latest content-free facts produced by the process pressure observer.
///
/// The cumulative compression counter measures compression activity, not the
/// current size of the compressor. Optional fields remain unknown until a
/// valid VM-counter sample exists and retain that sample's timestamp and epoch
/// across transient probe failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PressureTelemetry {
    pub raw_pressure: MemoryPressure,
    pub swap_outs_total: Option<u64>,
    pub compressions_total: Option<u64>,
    /// Wall-clock time of the cumulative counters above.
    pub sampled_at_unix_ms: Option<u64>,
    /// Monotonic identity of the latest successful counter sample.
    pub sample_epoch: Option<u64>,
    /// Monotonic count of constrained or critical OS pressure events.
    pub non_normal_pressure_event_epoch: u64,
}

impl Default for PressureTelemetry {
    fn default() -> Self {
        Self {
            raw_pressure: MemoryPressure::Normal,
            swap_outs_total: None,
            compressions_total: None,
            sampled_at_unix_ms: None,
            sample_epoch: None,
            non_normal_pressure_event_epoch: 0,
        }
    }
}

/// Evidence backing the current capacity envelope.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityBasis {
    Configured,
}

/// Fixed context and process telemetry exposed by `/metrics`.
/// Numbers and identities only — never prompt or request content.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityDiagnostics {
    pub boot_id: String,
    pub active_reservations: usize,
    pub oldest_reservation_age_ms: Option<u64>,
    /// Latest observed pressure; it never changes context limits.
    pub pressure: MemoryPressure,
    /// Last pressure level reported by the operating-system event source.
    pub raw_pressure: MemoryPressure,
    pub mlx_active_bytes: u64,
    /// Process-global MLX peak since the latest serialized phase reset.
    pub mlx_peak_bytes: u64,
    pub swap_out_delta: u64,
    /// Latest compression-activity delta; this is not compressor byte growth.
    pub compressor_delta: u64,
    pub swap_outs_total: Option<u64>,
    /// Cumulative compression activity, not current compressor residency.
    pub compressions_total: Option<u64>,
    /// Wall-clock time of the cumulative pressure counters above.
    pub pressure_sampled_at_unix_ms: Option<u64>,
    /// Monotonic identity of the latest successful pressure-counter sample.
    pub pressure_sample_epoch: Option<u64>,
    /// Monotonic count of constrained or critical raw OS pressure events.
    pub non_normal_pressure_event_epoch: u64,
    pub rejections: CapacityRejectionDiagnostics,
    pub models: Vec<CapacityModelDiagnostics>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityRejectionDiagnostics {
    pub exceeded: u64,
    pub unavailable: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityModelDiagnostics {
    pub model: String,
    pub generation: u64,
    pub available: bool,
    pub basis: CapacityBasis,
    pub pressure: MemoryPressure,
    pub safe_total_tokens: u64,
    pub recommended_output_tokens: u64,
    pub max_prompt_tokens: u64,
}

impl Default for CapacityDiagnostics {
    fn default() -> Self {
        Self {
            boot_id: String::new(),
            active_reservations: 0,
            oldest_reservation_age_ms: None,
            pressure: MemoryPressure::Normal,
            raw_pressure: MemoryPressure::Normal,
            mlx_active_bytes: 0,
            mlx_peak_bytes: 0,
            swap_out_delta: 0,
            compressor_delta: 0,
            swap_outs_total: None,
            compressions_total: None,
            pressure_sampled_at_unix_ms: None,
            pressure_sample_epoch: None,
            non_normal_pressure_event_epoch: 0,
            rejections: CapacityRejectionDiagnostics::default(),
            models: Vec::new(),
        }
    }
}

/// Versioned fixed context advertised for one model by this Higgs process.
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

/// Semantic request shape. Cached tokens never reduce the context requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestCost {
    pub prompt_tokens: u64,
    pub output_tokens: u64,
}

#[cfg(test)]
mod output_default_tests {
    use super::*;
    #[tokio::test]
    async fn omitted_output_uses_exact_remaining_context_without_clamping_explicit_output() {
        let (state, _engine) = suffix_charging_route_test_state("output-default");
        assert_eq!(
            resolve_output_tokens(&state, "output-default", 3000, None, 32768),
            72
        );
        assert_eq!(
            resolve_output_tokens(&state, "output-default", 3000, Some(128), 32768),
            128
        );
        assert_eq!(
            resolve_output_tokens(&state, "output-default", 3072, None, 32768),
            0
        );
        assert_eq!(
            resolve_output_tokens(&state, "output-default", 4000, None, 32768),
            0
        );
        assert_eq!(
            resolve_output_tokens(&state, "output-default", 0, None, 16),
            16
        );
        assert_eq!(
            resolve_output_tokens(&state, "output-default", 3000, Some(0), 32768),
            0
        );
    }
}
