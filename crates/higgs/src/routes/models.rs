use std::sync::Arc;
use std::time::Duration;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use bytes::Bytes;

use crate::{
    capacity::{ActiveRegistration, DrainRegistration, ModelCapacityFacts, RegistrationError},
    config::{ModelConfig, validate_pflash_settings},
    error::ServerError,
    model_resolver,
    state::{
        Engine, SharedState, build_engine_with_capacity, measure_after_engine_drop,
        release_failed_engine,
    },
    types::openai::{ModelList, ModelObject},
};

/// How long `DELETE /v1/models/{name}` waits for in-flight requests to release a
/// model before detaching the final drop to a background task.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// Poll cadence while draining references to an unloaded model.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

type LoadedEngine = Result<(String, Engine, ModelCapacityFacts), String>;

/// Owns an in-progress blocking load. Dropping the awaiting HTTP future hands
/// the still-running task to a detached cleanup task instead of detaching the
/// model allocation itself.
struct RuntimeLoadGuard {
    task: Option<tokio::task::JoinHandle<LoadedEngine>>,
    capacity: Arc<crate::capacity::CapacityRegistry>,
    #[cfg(test)]
    cleanup_ack: Option<tokio::sync::oneshot::Sender<()>>,
}

impl RuntimeLoadGuard {
    fn spawn(
        capacity: Arc<crate::capacity::CapacityRegistry>,
        load: impl FnOnce() -> LoadedEngine + Send + 'static,
    ) -> Self {
        Self {
            task: Some(tokio::task::spawn_blocking(load)),
            capacity,
            #[cfg(test)]
            cleanup_ack: None,
        }
    }

    async fn finish(mut self) -> Result<LoadedEngine, tokio::task::JoinError> {
        let result = self.task.as_mut().expect("runtime load task exists").await;
        self.task.take();
        result
    }
}

impl Drop for RuntimeLoadGuard {
    fn drop(&mut self) {
        let Some(task) = self.task.take() else {
            return;
        };
        let capacity = Arc::clone(&self.capacity);
        #[cfg(test)]
        let cleanup_ack = self.cleanup_ack.take();
        tokio::spawn(async move {
            if let Ok(Ok((_name, engine, _facts))) = task.await {
                let _ =
                    tokio::task::spawn_blocking(move || release_failed_engine(engine, &capacity))
                        .await;
            }
            #[cfg(test)]
            if let Some(cleanup_ack) = cleanup_ack {
                let _ = cleanup_ack.send(());
            }
        });
    }
}

struct ProvisionalEngine {
    engine: Option<Arc<Engine>>,
    capacity: Arc<crate::capacity::CapacityRegistry>,
}

struct PublicationRollback {
    active: Option<ActiveRegistration>,
    state: SharedState,
}

impl PublicationRollback {
    fn new(active: ActiveRegistration, state: &SharedState) -> Self {
        Self {
            active: Some(active),
            state: Arc::clone(state),
        }
    }

    fn registration(&mut self) -> &mut ActiveRegistration {
        self.active.as_mut().expect("active registration exists")
    }

    fn publish(mut self) {
        self.active.take();
    }
}

impl Drop for PublicationRollback {
    fn drop(&mut self) {
        let Some(active) = self.active.take() else {
            return;
        };
        drop(active);
        let state = Arc::clone(&self.state);
        tokio::spawn(async move {
            if let Err(error) = state
                .router
                .apply_capacity_cache_allocations(&state.capacity)
                .await
            {
                tracing::warn!(%error, "failed to restore cache policy after cancelled publication");
            }
        });
    }
}

impl ProvisionalEngine {
    fn new(engine: Engine, capacity: Arc<crate::capacity::CapacityRegistry>) -> Self {
        Self {
            engine: Some(Arc::new(engine)),
            capacity,
        }
    }

    fn engine(&self) -> Arc<Engine> {
        Arc::clone(self.engine.as_ref().expect("provisional engine exists"))
    }

    fn publish(mut self) {
        self.engine.take();
    }

    async fn cleanup(mut self) {
        if let Some(engine) = self.engine.take() {
            let capacity = Arc::clone(&self.capacity);
            let cleanup = tokio::spawn(cleanup_shared_engine(engine, capacity));
            let _ = cleanup.await;
        }
    }
}

async fn cleanup_shared_engine(
    mut engine: Arc<Engine>,
    capacity: Arc<crate::capacity::CapacityRegistry>,
) {
    loop {
        match Arc::try_unwrap(engine) {
            Ok(engine) => {
                let _ = tokio::task::spawn_blocking(move || {
                    release_failed_engine(engine, &capacity);
                })
                .await;
                return;
            }
            Err(shared) => {
                engine = shared;
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }
    }
}

impl Drop for ProvisionalEngine {
    fn drop(&mut self) {
        let Some(engine) = self.engine.take() else {
            return;
        };
        let capacity = Arc::clone(&self.capacity);
        tokio::spawn(cleanup_shared_engine(engine, capacity));
    }
}

/// `GET /v1/models` -- list the locally loaded models.
pub async fn list_models(State(state): State<SharedState>) -> Json<ModelList> {
    let data = state
        .router
        .local_models_with_vlm()
        .into_iter()
        .map(|(name, vision)| model_object(name, vision))
        .collect();
    Json(ModelList {
        object: "list",
        data,
        runtime_model_load: state.config.local.allow_runtime_model_load,
    })
}

/// `POST /v1/models` -- load a model into GPU memory at runtime.
///
/// Opt-in via `local.allow_runtime_model_load`. The body mirrors a `[[models]]`
/// config entry (`path` required; `name`, `batch`, `mlx_profile`, `kv_*`
/// optional). Returns the created model object, `409` on a name collision, or
/// `403` when runtime loading is disabled.
pub async fn load_model(
    State(state): State<SharedState>,
    body: Bytes,
) -> Result<Json<ModelObject>, ServerError> {
    if !state.config.local.allow_runtime_model_load {
        return Err(ServerError::Forbidden(
            "runtime model loading is disabled; set local.allow_runtime_model_load = true to enable it"
                .to_owned(),
        ));
    }

    let model_cfg: ModelConfig = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("invalid model config: {e}")))?;

    // Validate the KV-cache config up front, mirroring `doctor::check_models`.
    model_cfg
        .kv_cache_config()
        .validate()
        .map_err(|e| ServerError::BadRequest(format!("invalid KV cache config: {e}")))?;
    if model_cfg.batch && model_cfg.kv_cache_config().is_turboquant() {
        return Err(ServerError::BadRequest(
            "unsupported combination: TurboQuant KV cache with batch=true".to_owned(),
        ));
    }
    validate_pflash_settings(&model_cfg)
        .map_err(|error| ServerError::BadRequest(format!("invalid PFlash config: {error}")))?;

    // Cheap collision pre-check when the caller named the model, so we don't pay
    // for a full load just to reject it. `insert_engine` re-checks under the
    // write lock, so this is an optimization, not the source of truth.
    if let Some(ref name) = model_cfg.name {
        if state.router.contains_engine(name) {
            return Err(ServerError::Conflict(format!(
                "model '{name}' is already loaded"
            )));
        }
    }

    // Resolve the path without prompting -- the server has no interactive stdin.
    let resolved = model_resolver::resolve(&model_cfg.path).map_err(|e| {
        ServerError::BadRequest(format!(
            "model '{}' not found locally: {e}; pre-download it (e.g. `huggingface-cli download {}`)",
            model_cfg.path, model_cfg.path
        ))
    })?;
    let exposed_name = crate::state::resolve_exposed_model_name(
        model_cfg.name.as_deref(),
        &model_cfg.path,
        &resolved,
    );
    if state.router.contains_engine(&exposed_name) {
        return Err(ServerError::Conflict(format!(
            "model '{exposed_name}' is already loaded"
        )));
    }

    // The weight load is blocking and GPU-bound; keep it off the async runtime.
    let config = state.config.clone();
    let capacity = Arc::clone(&state.capacity);
    let cfg = model_cfg.clone();
    let load_capacity = Arc::clone(&capacity);
    let (name, engine, facts) = RuntimeLoadGuard::spawn(load_capacity, move || {
        build_engine_with_capacity(&resolved, &cfg, &config, &capacity)
    })
    .finish()
    .await
    .map_err(|e| ServerError::InternalError(format!("model load task failed: {e}")))?
    .map_err(ServerError::BadRequest)?;
    let vision = engine.is_vlm();

    publish_loaded_engine(
        &state,
        name.clone(),
        engine,
        model_cfg.generation_defaults.clone(),
        facts,
    )
    .await?;

    tracing::info!(model_name = %name, "Model loaded at runtime");
    Ok(Json(model_object(name, vision)))
}

async fn publish_loaded_engine(
    state: &SharedState,
    name: String,
    engine: Engine,
    generation_defaults: crate::config::GenerationDefaults,
    facts: ModelCapacityFacts,
) -> Result<(), ServerError> {
    publish_loaded_engine_inner(
        state,
        name,
        engine,
        generation_defaults,
        facts,
        #[cfg(test)]
        None,
        #[cfg(test)]
        None,
    )
    .await
}

#[cfg(test)]
struct PublicationTestGate {
    arrived: tokio::sync::oneshot::Sender<()>,
    release: tokio::sync::oneshot::Receiver<()>,
}

async fn publish_loaded_engine_inner(
    state: &SharedState,
    name: String,
    engine: Engine,
    generation_defaults: crate::config::GenerationDefaults,
    facts: ModelCapacityFacts,
    #[cfg(test)] publication_gate: Option<PublicationTestGate>,
    #[cfg(test)] insertion_gate: Option<crate::router::RouteInsertionTestGate>,
) -> Result<(), ServerError> {
    let provisional = ProvisionalEngine::new(engine, Arc::clone(&state.capacity));
    let ticket = match state.capacity.begin_registration(name.clone()) {
        Ok(ticket) => ticket,
        Err(error) => {
            provisional.cleanup().await;
            return Err(registration_error(error));
        }
    };
    let active = match state.capacity.commit_active(ticket, facts) {
        Ok(active) => active,
        Err(error) => {
            provisional.cleanup().await;
            return Err(registration_error(error));
        }
    };
    let mut publication = PublicationRollback::new(active, state);
    if let Err((name, engine, error)) = state
        .router
        .insert_engine_with_capacity(
            name,
            provisional.engine(),
            generation_defaults,
            &state.capacity,
            publication.registration(),
            #[cfg(test)]
            insertion_gate,
        )
        .await
    {
        drop(engine);
        provisional.cleanup().await;
        let _ = state
            .router
            .apply_capacity_cache_allocations(&state.capacity)
            .await;
        return if error == "model name is already loaded" {
            Err(ServerError::Conflict(format!(
                "model '{name}' is already loaded"
            )))
        } else {
            Err(ServerError::InternalError(error))
        };
    }
    publication.publish();
    provisional.publish();
    #[cfg(test)]
    if let Some(gate) = publication_gate {
        let _ = gate.arrived.send(());
        let _ = gate.release.await;
    }
    Ok(())
}

fn registration_error(error: RegistrationError) -> ServerError {
    match error {
        RegistrationError::AlreadyRegistered(model) => {
            ServerError::Conflict(format!("model '{model}' is already loaded"))
        }
        RegistrationError::InsufficientCapacity(model) => ServerError::BadRequest(format!(
            "model '{model}' has insufficient capacity for the minimum working request"
        )),
        other => ServerError::InternalError(other.to_string()),
    }
}

/// `DELETE /v1/models/{name}` -- unload a model and free its GPU memory.
///
/// Returns `204` once the model is fully unloaded, `202` if a request was still
/// in flight past the drain timeout (the final free is detached), `404` if no
/// such model is loaded, or `409` if the model backs the auto-router.
pub async fn unload_model(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> Result<StatusCode, ServerError> {
    if state.router.auto_router_model_name() == Some(name.as_str()) {
        return Err(ServerError::Conflict(format!(
            "model '{name}' is bound to the auto-router and cannot be unloaded"
        )));
    }

    if !state.router.contains_engine(&name) {
        return Err(ServerError::ModelNotFound(name));
    }
    let drain = state
        .capacity
        .begin_drain(&name)
        .map_err(registration_error)?;
    let engine = state
        .router
        .remove_engine(&name)
        .ok_or_else(|| ServerError::ModelNotFound(name.clone()))?;
    let capacity_drain = CapacityDrain {
        state: Arc::clone(&state),
        registration: drain,
    };

    // The map entry is gone, so no new request can take a reference and the
    // strong count only decreases. Free GPU memory once the last in-flight
    // request releases its clone; detach past the timeout so a long generation
    // can't block the response.
    if drain_and_drop(engine, DRAIN_TIMEOUT, Some(capacity_drain)).await {
        tracing::info!(model_name = %name, "Model unloaded");
        Ok(StatusCode::NO_CONTENT)
    } else {
        tracing::info!(model_name = %name, "Model unload deferred; request still in flight");
        Ok(StatusCode::ACCEPTED)
    }
}

/// Wait until `engine` is solely owned here, then drop it (freeing GPU memory).
///
/// Returns `true` if dropped within `timeout`; `false` if it timed out and the
/// final drop was handed to a detached task. Dropping is intentionally not gated
/// on engine inference: teardown itself only releases buffers; allocator-cache
/// cleanup and the following measurement are serialized by the process GPU gate.
struct CapacityDrain {
    state: SharedState,
    registration: DrainRegistration,
}

impl CapacityDrain {
    async fn finish(self) {
        let memory_after_release =
            measure_after_engine_drop(&self.state.capacity, "model unload/swap");
        if let Err(error) = self
            .state
            .capacity
            .finish_unregister(self.registration, memory_after_release)
        {
            tracing::warn!(%error, "failed to persist learned capacity profile");
        }
        if let Err(error) = self
            .state
            .router
            .apply_capacity_cache_allocations(&self.state.capacity)
            .await
        {
            tracing::warn!(%error, "failed to apply cache allocations after model unload");
        }
    }
}

async fn drain_and_drop(
    engine: Arc<Engine>,
    timeout: Duration,
    capacity_drain: Option<CapacityDrain>,
) -> bool {
    let (finished, wait_finished) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        drain_in_background(engine, capacity_drain).await;
        let _ = finished.send(());
    });
    tokio::time::timeout(timeout, wait_finished).await.is_ok()
}

/// Poll until the detached engine reference is sole-owned, then drop it.
async fn drain_in_background(mut engine: Arc<Engine>, capacity_drain: Option<CapacityDrain>) {
    if let Some(drain) = capacity_drain.as_ref() {
        drain
            .state
            .capacity
            .wait_for_model_reservations(drain.registration.model())
            .await;
    }
    loop {
        match Arc::try_unwrap(engine) {
            Ok(owned) => {
                if let Err(error) = owned.shutdown() {
                    tracing::warn!(%error, "failed to join model engine during deferred unload");
                }
                if let Some(drain) = capacity_drain {
                    drain.finish().await;
                }
                return;
            }
            Err(shared) => {
                engine = shared;
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }
    }
}

fn model_object(name: String, vision: bool) -> ModelObject {
    ModelObject {
        id: name,
        object: "model",
        created: chrono::Utc::now().timestamp(),
        owned_by: "local".to_owned(),
        vision,
    }
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Instant;

    use crate::router::Router;
    use crate::state::AppState;
    use axum::body::Body;
    use higgs_engine::{EngineCostDescription, MlxMemorySnapshot, TransientPrefillEstimate};
    use http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn build_state(toml: &str, engines: HashMap<String, Arc<Engine>>) -> SharedState {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        // A stub provider satisfies the "at least one [[models]] or [provider.*]"
        // config rule; it does not affect direct local-engine routing.
        let full = format!("[provider.stub]\nurl = \"http://127.0.0.1:1\"\n\n{toml}");
        std::fs::write(&path, full).unwrap();
        let config = crate::config::load_config_file(&path, None).unwrap();
        let router = Router::from_config(&config, engines).unwrap();
        Arc::new(AppState::new(router, config, reqwest::Client::new(), None))
    }

    fn stub_engines(names: &[&str]) -> HashMap<String, Arc<Engine>> {
        names
            .iter()
            .map(|n| ((*n).to_owned(), Arc::new(Engine::test_stub(n))))
            .collect()
    }

    fn capacity_facts(name: &str) -> crate::capacity::ModelCapacityFacts {
        const GIB: u64 = 1024 * 1024 * 1024;
        crate::capacity::ModelCapacityFacts {
            model: name.to_owned(),
            model_fingerprint: format!("sha256:{name}"),
            memory: MlxMemorySnapshot {
                active_bytes: 5 * GIB,
                peak_bytes: 5 * GIB,
                memory_limit_bytes: Some(24 * GIB),
                metal_recommended_working_set_bytes: Some(24 * GIB),
            },
            costs: EngineCostDescription {
                fixed_live_session_bytes: 0,
                persistent_bytes_per_token: 20_480,
                decode_workspace_bytes: 0,
                transient_prefill: TransientPrefillEstimate {
                    base_bytes: GIB,
                    bytes_per_prompt_token: 0,
                    bytes_per_chunk_token: 0,
                    max_prompt_tokens: 131_072,
                    max_chunk_tokens: 4_096,
                },
            },
            loaded_model_bytes: 5 * GIB,
            architectural_max_tokens: 131_072,
            prefill_chunk_tokens: 1_024,
            retained_session_tokens: 49_152,
            retained_resident_bytes: 0,
            prefix_cache_resident_bytes: 0,
            retained_bytes_ceiling: 2 * GIB,
            prefix_cache_bytes_ceiling: GIB,
            cache_capabilities: crate::capacity::CacheCapabilities::SIMPLE,
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

    async fn publish_test_engine(
        state: &SharedState,
        name: String,
        engine: Engine,
        generation_defaults: crate::config::GenerationDefaults,
        facts: ModelCapacityFacts,
    ) -> Result<(), ServerError> {
        // Production facts are returned only after their allocator snapshot is
        // published inside the serialized MLX load window. Stub tests model
        // that boundary explicitly before entering lifecycle publication.
        state.capacity.refresh_memory(facts.memory);
        publish_loaded_engine(state, name, engine, generation_defaults, facts).await
    }

    #[tokio::test]
    async fn publish_loaded_engine_commits_capacity_before_route_visibility() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        assert!(!state.router.contains_engine("escha"));
        assert!(state.capacity.snapshot("escha").is_err());

        publish_test_engine(
            &state,
            "escha".to_owned(),
            Engine::test_stub("escha"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("escha"),
        )
        .await
        .unwrap();

        assert!(state.router.contains_engine("escha"));
        assert_eq!(
            state.capacity.snapshot("escha").unwrap().availability,
            crate::capacity::CapacityAvailability::Available
        );
        let allocation = state.capacity.snapshot("escha").unwrap();
        assert_eq!(
            state.router.local_engines()[0].route_test_capacity_cache_limits(),
            (allocation.retained_bytes, allocation.prefix_cache_bytes)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn successful_route_insertion_exposes_capacity_and_route_as_one_commit() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        let (arrived, wait_arrived) = tokio::sync::oneshot::channel();
        let (release, wait_release) = tokio::sync::oneshot::channel();
        let publish_state = Arc::clone(&state);
        let publication = tokio::spawn(async move {
            publish_state
                .capacity
                .refresh_memory(capacity_facts("second").memory);
            publish_loaded_engine_inner(
                &publish_state,
                "second".to_owned(),
                Engine::test_stub("second"),
                crate::config::GenerationDefaults::default(),
                capacity_facts("second"),
                Some(PublicationTestGate {
                    arrived,
                    release: wait_release,
                }),
                None,
            )
            .await
        });
        tokio::time::timeout(Duration::from_secs(1), wait_arrived)
            .await
            .expect("publication must pause after successful router insertion")
            .unwrap();

        let app = crate::build_router(Arc::clone(&state), 30.0, None, 0, 1024, None);
        let capacity_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=second")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let capacity_json: serde_json::Value = serde_json::from_slice(
            &capacity_response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes(),
        )
        .unwrap();
        let chat_visible = matches!(
            state.router.resolve("second", None).await,
            Ok(crate::router::ResolvedRoute::Higgs { .. })
        );
        let Json(models) = list_models(State(Arc::clone(&state))).await;

        let delete_state = Arc::clone(&state);
        let delete = tokio::spawn(async move {
            unload_model(State(delete_state), Path("second".to_owned())).await
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while state.router.contains_engine("second") {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("concurrent DELETE must observe and start draining the committed route");
        let _ = release.send(());

        assert_eq!(capacity_json["availability"], "available");
        assert!(
            chat_visible,
            "chat resolution must agree with capacity visibility"
        );
        assert!(models.data.iter().any(|model| model.id == "second"));
        publication.await.unwrap().unwrap();
        assert_eq!(delete.await.unwrap().unwrap(), StatusCode::NO_CONTENT);
        assert!(!state.router.contains_engine("second"));
        assert!(
            state
                .capacity
                .begin_registration("second".to_owned())
                .is_ok(),
            "publication/delete race must leave no registration or drain conflict"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pressure_racing_cache_publication_finishes_on_latest_revision() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        publish_test_engine(
            &state,
            "escha".to_owned(),
            Engine::test_stub("escha"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("escha"),
        )
        .await
        .unwrap();

        let pressure_state = Arc::clone(&state);
        let pressure = tokio::spawn(async move {
            pressure_state.capacity.apply_pressure_observation(
                crate::capacity::PressureObservation {
                    pressure: crate::capacity::MemoryPressure::Critical,
                    swap_out_delta: 1,
                    compressor_delta: 1,
                },
            );
            pressure_state
                .router
                .apply_capacity_cache_allocations(&pressure_state.capacity)
                .await
                .unwrap();
        });
        state
            .router
            .apply_capacity_cache_allocations(&state.capacity)
            .await
            .unwrap();
        pressure.await.unwrap();

        let allocation = state.capacity.snapshot("escha").unwrap();
        assert_eq!(
            allocation.pressure,
            crate::capacity::MemoryPressure::Critical
        );
        assert_eq!(
            state.router.local_engines()[0].route_test_capacity_cache_limits(),
            (allocation.retained_bytes, allocation.prefix_cache_bytes)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelling_committed_publication_restores_existing_engine_policy() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        let mut first_facts = capacity_facts("first");
        first_facts.retained_bytes_ceiling = 0;
        first_facts.prefix_cache_bytes_ceiling = 0;
        publish_test_engine(
            &state,
            "first".to_owned(),
            Engine::test_stub("first"),
            crate::config::GenerationDefaults::default(),
            first_facts,
        )
        .await
        .unwrap();
        let first_engine = state.router.local_engines()[0].clone();
        let baseline = first_engine.route_test_capacity_cache_limits();
        state.capacity.refresh_memory(MlxMemorySnapshot {
            active_bytes: 5 * 1024 * 1024 * 1024,
            peak_bytes: 5 * 1024 * 1024 * 1024,
            memory_limit_bytes: Some(128 * 1024 * 1024 * 1024),
            metal_recommended_working_set_bytes: Some(128 * 1024 * 1024 * 1024),
        });
        let expected_after_rollback = state
            .capacity
            .cache_allocation_plan()
            .allocations
            .into_iter()
            .find(|(name, _, _)| name == "first")
            .map(|(_, retained, prefix)| (retained, prefix))
            .unwrap();
        assert_ne!(expected_after_rollback, baseline);

        let (second_engine, arrived, release) = Engine::test_stub_with_cache_gate("second");
        let mut second_facts = capacity_facts("second");
        second_facts.memory.memory_limit_bytes = Some(128 * 1024 * 1024 * 1024);
        second_facts.memory.metal_recommended_working_set_bytes =
            second_facts.memory.memory_limit_bytes;
        second_facts.retained_bytes_ceiling = 0;
        second_facts.prefix_cache_bytes_ceiling = 0;
        let publish_state = Arc::clone(&state);
        let publication = tokio::spawn(async move {
            publish_test_engine(
                &publish_state,
                "second".to_owned(),
                second_engine,
                crate::config::GenerationDefaults::default(),
                second_facts,
            )
            .await
        });
        if tokio::time::timeout(Duration::from_secs(1), arrived.notified())
            .await
            .is_err()
        {
            panic!(
                "publication never reached cache gate: {:?}",
                publication.await
            );
        }
        assert_ne!(first_engine.route_test_capacity_cache_limits(), baseline);
        let published_first = state.capacity.snapshot("first").unwrap();
        assert_eq!(
            (
                published_first.retained_bytes,
                published_first.prefix_cache_bytes
            ),
            baseline,
            "capacity API must retain acknowledged bytes until the whole plan is applied"
        );
        assert_eq!(
            state.capacity.snapshot("second").unwrap().availability,
            crate::capacity::CapacityAvailability::Unavailable,
            "provisional model must remain unavailable before route publication"
        );
        publication.abort();
        release.notify_waiters();

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if state.capacity.snapshot("second").is_err()
                    && first_engine.route_test_capacity_cache_limits() == expected_after_rollback
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("publication rollback must restore the latest policy");
        assert!(!state.router.contains_engine("second"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelling_after_disabled_route_insertion_removes_the_uncommitted_entry() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        publish_test_engine(
            &state,
            "first".to_owned(),
            Engine::test_stub("first"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("first"),
        )
        .await
        .unwrap();

        let (arrived, wait_arrived) = tokio::sync::oneshot::channel();
        let (release, wait_release) = tokio::sync::oneshot::channel();
        let publish_state = Arc::clone(&state);
        let publication = tokio::spawn(async move {
            let facts = capacity_facts("second");
            publish_state.capacity.refresh_memory(facts.memory);
            publish_loaded_engine_inner(
                &publish_state,
                "second".to_owned(),
                Engine::test_stub("second"),
                crate::config::GenerationDefaults::default(),
                facts,
                None,
                Some(crate::router::RouteInsertionTestGate {
                    arrived,
                    release: wait_release,
                }),
            )
            .await
        });
        tokio::time::timeout(Duration::from_secs(1), wait_arrived)
            .await
            .expect("publication must pause immediately after disabled insertion")
            .unwrap();
        assert_eq!(
            state.capacity.snapshot("second").unwrap().availability,
            crate::capacity::CapacityAvailability::Unavailable
        );
        assert!(!matches!(
            state.router.resolve("second", None).await,
            Ok(crate::router::ResolvedRoute::Higgs { .. })
        ));
        let Json(models) = list_models(State(Arc::clone(&state))).await;
        assert!(models.data.iter().all(|model| model.id != "second"));
        assert!(matches!(
            unload_model(State(Arc::clone(&state)), Path("second".to_owned())).await,
            Err(ServerError::ModelNotFound(_))
        ));
        publication.abort();
        drop(release);

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let absent = state
                    .router
                    .local_engines()
                    .iter()
                    .all(|engine| engine.model_name() != "second");
                if absent && state.capacity.snapshot("second").is_err() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled disabled route must be removed and capacity rolled back");
        assert!(
            state
                .capacity
                .begin_registration("second".to_owned())
                .is_ok()
        );
    }

    #[tokio::test]
    async fn router_insertion_failure_rolls_back_active_capacity() {
        let state = build_state(
            "[local]\nallow_runtime_model_load = true\n",
            stub_engines(&["escha"]),
        );

        let error = publish_test_engine(
            &state,
            "escha".to_owned(),
            Engine::test_stub("replacement"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("escha"),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, ServerError::Conflict(_)));
        assert!(state.capacity.snapshot("escha").is_err());
    }

    #[test]
    fn test_model_objects_have_correct_fields() {
        let obj = model_object("test-model".to_owned(), false);
        assert_eq!(obj.object, "model");
        assert_eq!(obj.owned_by, "local");
        assert!(obj.created > 0);
        assert!(!obj.vision);
    }

    #[tokio::test]
    async fn list_models_reports_capabilities() {
        // runtime_model_load mirrors the config flag; vision is per-model
        // (stub engines are text-only); names stay sorted.
        let on = build_state(
            "[local]\nallow_runtime_model_load = true\n",
            stub_engines(&["zebra", "alpha"]),
        );
        let Json(list) = list_models(State(on)).await;
        assert!(list.runtime_model_load);
        let ids: Vec<&str> = list.data.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "zebra"]);
        assert!(list.data.iter().all(|m| !m.vision));

        let off = build_state(
            "[local]\nallow_runtime_model_load = false\n",
            HashMap::new(),
        );
        let Json(list) = list_models(State(off)).await;
        assert!(!list.runtime_model_load);
    }

    #[tokio::test]
    async fn load_disabled_returns_forbidden() {
        let state = build_state(
            "[local]\nallow_runtime_model_load = false\n",
            HashMap::new(),
        );
        let err = load_model(State(state), Bytes::from_static(b"{\"path\":\"x\"}"))
            .await
            .unwrap_err();
        assert!(matches!(err, ServerError::Forbidden(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelling_runtime_load_waits_for_cleanup_and_never_registers() {
        let capacity = crate::capacity::CapacityRegistry::new(["cancelled".to_owned()]);
        let (started, wait_started) = std::sync::mpsc::channel();
        let (release, wait_release) = std::sync::mpsc::channel();
        let (cleanup_ack, wait_cleanup) = tokio::sync::oneshot::channel();
        let mut guard = RuntimeLoadGuard::spawn(Arc::clone(&capacity), move || {
            started.send(()).unwrap();
            wait_release.recv().unwrap();
            Ok((
                "cancelled".to_owned(),
                Engine::test_stub("cancelled"),
                capacity_facts("cancelled"),
            ))
        });
        guard.cleanup_ack = Some(cleanup_ack);
        let waiter = tokio::spawn(async move { guard.finish().await });
        wait_started.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.abort();
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), wait_cleanup)
            .await
            .expect("cancellation cleanup must be durable")
            .unwrap();

        assert!(capacity.begin_registration("cancelled".to_owned()).is_ok());
    }

    #[tokio::test]
    async fn load_existing_name_conflicts() {
        // allow_runtime on + an already-loaded "llama"; naming it again is a
        // conflict caught before any (impossible, in-test) GPU load.
        let state = build_state(
            "[local]\nallow_runtime_model_load = true\n",
            stub_engines(&["llama"]),
        );
        let body = Bytes::from_static(b"{\"path\":\"some/path\",\"name\":\"llama\"}");
        let err = load_model(State(state), body).await.unwrap_err();
        assert!(matches!(err, ServerError::Conflict(_)));
    }

    #[tokio::test]
    async fn runtime_load_rejects_invalid_pflash_before_path_resolution() {
        use axum::response::IntoResponse as _;

        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        let body = Bytes::from_static(br#"{"path":"missing/model","prefill_keep_ratio":1.0}"#);

        let error = load_model(State(state), body).await.unwrap_err();
        assert!(
            error.to_string().contains("prefill_keep_ratio"),
            "route must report the invalid PFlash field before model resolution: {error}"
        );
        assert_eq!(error.into_response().status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unload_unknown_returns_not_found() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        let err = unload_model(State(state), Path("ghost".to_owned()))
            .await
            .unwrap_err();
        assert!(matches!(err, ServerError::ModelNotFound(_)));
    }

    #[tokio::test]
    async fn unload_auto_router_model_is_refused() {
        let toml = r#"
            [[models]]
            path = "/models/Arch-Router-1.5B-4bit"
            name = "router"

            [auto_router]
            enabled = true
            model = "router"
        "#;
        let state = build_state(toml, stub_engines(&["router"]));
        let err = unload_model(State(Arc::clone(&state)), Path("router".to_owned()))
            .await
            .unwrap_err();
        assert!(matches!(err, ServerError::Conflict(_)));
        // Still loaded.
        assert!(state.router.contains_engine("router"));
    }

    #[tokio::test]
    async fn unload_removes_engine_and_frees_it() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        publish_test_engine(
            &state,
            "llama".to_owned(),
            Engine::test_stub("llama"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("llama"),
        )
        .await
        .unwrap();
        let status = unload_model(State(Arc::clone(&state)), Path("llama".to_owned()))
            .await
            .unwrap();
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert!(!state.router.contains_engine("llama"));
        assert!(state.router.local_model_names().is_empty());
    }

    #[tokio::test]
    async fn unload_marks_capacity_unavailable_and_keeps_known_identity() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        publish_test_engine(
            &state,
            "escha".to_owned(),
            Engine::test_stub("escha"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("escha"),
        )
        .await
        .unwrap();
        let before = state.capacity.snapshot("escha").unwrap();

        let status = unload_model(State(Arc::clone(&state)), Path("escha".to_owned()))
            .await
            .unwrap();

        assert_eq!(status, StatusCode::NO_CONTENT);
        let after = state.capacity.snapshot("escha").unwrap();
        assert_eq!(
            after.availability,
            crate::capacity::CapacityAvailability::Unavailable
        );
        assert_eq!(after.model_fingerprint, before.model_fingerprint);
        assert!(after.generation > before.generation);
    }

    #[tokio::test]
    async fn capacity_admission_unload_waits_for_active_worker_reservation() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        publish_test_engine(
            &state,
            "reserved".to_owned(),
            Engine::test_stub("reserved"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("reserved"),
        )
        .await
        .unwrap();
        let reservation = state
            .capacity
            .reserve_request(
                "reserved",
                crate::capacity::RequestCost {
                    execution_path: crate::capacity::ExecutionPath::Cold,
                    prompt_tokens: 1,
                    suffix_tokens: 1,
                    output_tokens: 1,
                    retained_growth_bytes: 0,
                },
            )
            .await
            .unwrap();
        let unload_state = Arc::clone(&state);
        let unload = tokio::spawn(async move {
            unload_model(State(unload_state), Path("reserved".to_owned())).await
        });
        while state.router.contains_engine("reserved") {
            tokio::task::yield_now().await;
        }
        tokio::task::yield_now().await;
        assert!(!unload.is_finished());
        drop(reservation);
        assert_eq!(unload.await.unwrap().unwrap(), StatusCode::NO_CONTENT);
        assert_eq!(state.capacity.active_reservation_count("reserved"), 0);
    }

    #[tokio::test]
    async fn drain_and_drop_drops_sole_owner_immediately() {
        let engine = Arc::new(Engine::test_stub("solo"));
        assert!(drain_and_drop(engine, Duration::from_secs(1), None).await);
    }

    #[tokio::test]
    async fn drain_and_drop_waits_for_in_flight_clone() {
        let engine = Arc::new(Engine::test_stub("busy"));
        let clone = Arc::clone(&engine);
        // Release the simulated in-flight reference after a short delay.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(120)).await;
            drop(clone);
        });
        let start = Instant::now();
        assert!(drain_and_drop(engine, Duration::from_secs(5), None).await);
        assert!(
            start.elapsed() >= Duration::from_millis(100),
            "should have waited for the clone to drop"
        );
    }

    #[tokio::test]
    async fn drain_and_drop_times_out_and_detaches() {
        let engine = Arc::new(Engine::test_stub("stuck"));
        let clone = Arc::clone(&engine); // held past the timeout
        let drained = drain_and_drop(engine, Duration::from_millis(80), None).await;
        assert!(!drained, "should time out while a reference is held");
        drop(clone); // let the detached task finish
    }

    #[tokio::test]
    async fn cancelling_unload_keeps_drain_cleanup_owned_until_final_unregister() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        publish_test_engine(
            &state,
            "busy".to_owned(),
            Engine::test_stub("busy"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("busy"),
        )
        .await
        .unwrap();
        let lease = state.router.local_engines()[0].clone();
        let unload_state = Arc::clone(&state);
        let unload = tokio::spawn(async move {
            unload_model(State(unload_state), Path("busy".to_owned())).await
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while state.router.contains_engine("busy") {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        unload.abort();
        drop(lease);

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Ok(ticket) = state.capacity.begin_registration("busy".to_owned()) {
                    drop(ticket);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("detached drain must finish unregister after handler cancellation");
    }

    #[tokio::test]
    async fn cancelling_explicit_provisional_cleanup_does_not_drop_the_cleanup_owner() {
        let capacity = crate::capacity::CapacityRegistry::new(std::iter::empty());
        let provisional =
            ProvisionalEngine::new(Engine::test_stub("provisional"), Arc::clone(&capacity));
        let leased = provisional.engine();
        let weak = Arc::downgrade(&leased);
        let cleanup = tokio::spawn(provisional.cleanup());
        tokio::task::yield_now().await;
        cleanup.abort();
        drop(leased);

        tokio::time::timeout(Duration::from_secs(1), async {
            while weak.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("detached provisional cleanup must retain ownership through cancellation");
    }
}
