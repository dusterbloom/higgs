use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use bytes::Bytes;

use crate::{
    capacity::{DrainRegistration, ModelCapacityFacts, RegistrationError},
    config::{ModelConfig, validate_pflash_settings},
    error::ServerError,
    model_resolver,
    state::{
        Engine, SharedState, build_engine_with_capacity, measure_after_engine_drop,
        refresh_after_engine_drop, release_failed_engine,
    },
    types::openai::{ModelList, ModelObject},
};

/// How long `DELETE /v1/models/{name}` waits for in-flight requests to release a
/// model before detaching the final drop to a background task.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// Poll cadence while draining references to an unloaded model.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

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

    // The weight load is blocking and GPU-bound; keep it off the async runtime.
    let config = state.config.clone();
    let capacity = Arc::clone(&state.capacity);
    let cfg = model_cfg.clone();
    let (name, engine, facts) = tokio::task::spawn_blocking(move || {
        build_engine_with_capacity(&resolved, &cfg, &config, &capacity)
    })
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
    )?;

    tracing::info!(model_name = %name, "Model loaded at runtime");
    Ok(Json(model_object(name, vision)))
}

fn publish_loaded_engine(
    state: &SharedState,
    name: String,
    engine: Engine,
    generation_defaults: crate::config::GenerationDefaults,
    facts: ModelCapacityFacts,
) -> Result<(), ServerError> {
    let ticket = match state.capacity.begin_registration(name.clone()) {
        Ok(ticket) => ticket,
        Err(error) => {
            release_failed_engine(engine, &state.capacity);
            return Err(registration_error(error));
        }
    };
    let active = match state.capacity.commit_active(ticket, facts) {
        Ok(active) => active,
        Err(error) => {
            release_failed_engine(engine, &state.capacity);
            return Err(registration_error(error));
        }
    };
    let allocation = match state.capacity.snapshot(&name) {
        Ok(allocation) => allocation,
        Err(error) => {
            drop(active);
            release_failed_engine(engine, &state.capacity);
            return Err(registration_error(error));
        }
    };
    if let Err(error) =
        engine.apply_capacity_cache_limits(allocation.retained_bytes, allocation.prefix_cache_bytes)
    {
        drop(active);
        release_failed_engine(engine, &state.capacity);
        let _ = state
            .router
            .apply_capacity_cache_allocations(&state.capacity);
        return Err(ServerError::InternalError(format!(
            "failed to apply cache allocation for '{name}': {error}"
        )));
    }
    if let Err(error) = state
        .router
        .apply_capacity_cache_allocations(&state.capacity)
    {
        drop(active);
        release_failed_engine(engine, &state.capacity);
        let _ = state
            .router
            .apply_capacity_cache_allocations(&state.capacity);
        return Err(ServerError::InternalError(error));
    }
    if let Err((name, engine)) =
        state
            .router
            .insert_engine_with_defaults(name, Arc::new(engine), generation_defaults)
    {
        drop(active);
        if let Ok(engine) = Arc::try_unwrap(engine)
            && let Err(error) = engine.shutdown()
        {
            tracing::warn!(%error, "failed to join engine after publication failure");
        }
        let _ = refresh_after_engine_drop(&state.capacity, "failed model publication");
        let _ = state
            .router
            .apply_capacity_cache_allocations(&state.capacity);
        return Err(ServerError::Conflict(format!(
            "model '{name}' is already loaded"
        )));
    }
    active.publish();
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
    fn finish(self) {
        let memory_after_release = measure_after_engine_drop("model unload/swap");
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
        {
            tracing::warn!(%error, "failed to apply cache allocations after model unload");
        }
    }
}

async fn drain_and_drop(
    mut engine: Arc<Engine>,
    timeout: Duration,
    capacity_drain: Option<CapacityDrain>,
) -> bool {
    let start = Instant::now();
    loop {
        match Arc::try_unwrap(engine) {
            Ok(owned) => {
                if let Err(error) = owned.shutdown() {
                    tracing::warn!(%error, "failed to join model engine during unload");
                }
                // The engine's KV/prefix caches and weights are now freed, but
                // MLX parks the buffers in its allocator pool — return them to
                // the OS so a free-then-load swap actually reclaims memory. No
                // `eval`, so this is safe outside the GPU gate (same as `drop`).
                if let Some(drain) = capacity_drain {
                    drain.finish();
                }
                return true;
            }
            Err(shared) => {
                if start.elapsed() >= timeout {
                    tokio::spawn(drain_in_background(shared, capacity_drain));
                    return false;
                }
                engine = shared;
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }
    }
}

/// Poll until the detached engine reference is sole-owned, then drop it.
async fn drain_in_background(mut engine: Arc<Engine>, capacity_drain: Option<CapacityDrain>) {
    loop {
        match Arc::try_unwrap(engine) {
            Ok(owned) => {
                if let Err(error) = owned.shutdown() {
                    tracing::warn!(%error, "failed to join model engine during deferred unload");
                }
                if let Some(drain) = capacity_drain {
                    drain.finish();
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

    use crate::router::Router;
    use crate::state::AppState;
    use higgs_engine::{EngineCostDescription, MlxMemorySnapshot, TransientPrefillEstimate};

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

    #[test]
    fn publish_loaded_engine_commits_capacity_before_route_visibility() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        assert!(!state.router.contains_engine("escha"));
        assert!(state.capacity.snapshot("escha").is_err());

        publish_loaded_engine(
            &state,
            "escha".to_owned(),
            Engine::test_stub("escha"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("escha"),
        )
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

    #[test]
    fn router_insertion_failure_rolls_back_active_capacity() {
        let state = build_state(
            "[local]\nallow_runtime_model_load = true\n",
            stub_engines(&["escha"]),
        );

        let error = publish_loaded_engine(
            &state,
            "escha".to_owned(),
            Engine::test_stub("replacement"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("escha"),
        )
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
        publish_loaded_engine(
            &state,
            "llama".to_owned(),
            Engine::test_stub("llama"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("llama"),
        )
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
        publish_loaded_engine(
            &state,
            "escha".to_owned(),
            Engine::test_stub("escha"),
            crate::config::GenerationDefaults::default(),
            capacity_facts("escha"),
        )
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
}
