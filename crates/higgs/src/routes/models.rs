use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use bytes::Bytes;
use tokio::sync::OwnedSemaphorePermit;

use crate::{
    config::ModelConfig,
    error::ServerError,
    model_resolver,
    router::RuntimeLoadError,
    state::{Engine, SharedState, build_engine},
    types::openai::{ModelList, ModelObject},
};

/// How long `DELETE /v1/models/{name}` waits for in-flight requests to release a
/// model before detaching the final drop to a background task.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// Poll cadence while draining references to an unloaded model.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Warning cadence while a detached unload is still waiting on references.
const DRAIN_WARNING_INTERVAL: Duration = Duration::from_secs(30);

/// `GET /v1/models` -- list the locally loaded models.
pub async fn list_models(State(state): State<SharedState>) -> Json<ModelList> {
    let names = state.router.local_model_names();
    let data = model_objects_sorted(names.iter().map(String::as_str));
    Json(ModelList {
        object: "list",
        data,
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

    // Cheap collision pre-check when the caller named the model, so we don't pay
    // for a full load just to reject it. `insert_runtime_engine` re-checks under
    // the write lock, so this is an optimization, not the source of truth.
    if let Some(ref name) = model_cfg.name {
        if state.router.contains_engine(name) {
            return Err(ServerError::Conflict(format!(
                "model '{name}' is already loaded"
            )));
        }
    }

    // Resolve and authorize once without prompting -- the server has no
    // interactive stdin. The returned canonical path is the exact path passed
    // to the loader; arbitrary local directories are rejected.
    let resolved = model_resolver::resolve_runtime_model(
        &model_cfg.path,
        &state.config.local.runtime_model_roots,
    )
    .map_err(|e| {
        ServerError::BadRequest(format!(
            "model '{}' not found locally: {e}; pre-download it (e.g. `huggingface-cli download {}`)",
            model_cfg.path, model_cfg.path
        ))
    })?;

    // Serialize/bound concurrent loads: each load is GPU- and memory-heavy and
    // concurrent large loads can OOM the host. The owned permits are captured
    // by the blocking task so cancellation of this request cannot release them
    // while the load continues.
    let runtime_load_permit = state
        .router
        .acquire_runtime_load()
        .await
        .map_err(map_runtime_load_error)?;

    // The weight load is blocking and GPU-bound; keep it off the async runtime.
    let local = state.config.local.clone();
    let cfg = model_cfg.clone();
    let (name, engine, resident_permit) = tokio::task::spawn_blocking(move || {
        build_engine(&resolved, &cfg, &local)
            .map(|(name, engine)| (name, engine, runtime_load_permit.into_resident_permit()))
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("model load task failed: {e}")))?
    .map_err(ServerError::BadRequest)?;

    state
        .router
        .insert_runtime_engine(name.clone(), Arc::new(engine), resident_permit)
        .map_err(ServerError::Conflict)?;

    tracing::info!(model_name = %name, "Model loaded at runtime");
    Ok(Json(model_object(name)))
}

fn map_runtime_load_error(error: RuntimeLoadError) -> ServerError {
    let message = error.to_string();
    match error {
        RuntimeLoadError::ResidentBudgetExhausted => ServerError::BadRequest(message),
        RuntimeLoadError::LoadGateClosed | RuntimeLoadError::ResidentGateClosed => {
            ServerError::InternalError(message)
        }
    }
}

/// `DELETE /v1/models/{name}` -- unload a model and release its engine resources.
///
/// Returns `204` once the model is fully unloaded, `202` if a request was still
/// in flight past the drain timeout (the final free is detached), `404` if no
/// such model is loaded, or `409` if the model backs the auto-router.
pub async fn unload_model(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> Result<StatusCode, ServerError> {
    if !state.config.local.allow_runtime_model_load {
        return Err(ServerError::Forbidden(
            "runtime model unloading is disabled; set local.allow_runtime_model_load = true to enable it"
                .to_owned(),
        ));
    }

    if state.router.auto_router_model_name() == Some(name.as_str()) {
        return Err(ServerError::Conflict(format!(
            "model '{name}' is bound to the auto-router and cannot be unloaded"
        )));
    }

    let (engine, resident_permit) = state
        .router
        .remove_runtime_engine(&name)
        .ok_or_else(|| ServerError::ModelNotFound(name.clone()))?
        .into_parts();

    // The map entry is gone, so no new request can take a reference and the
    // strong count only decreases. Drop the engine once the last in-flight
    // request releases its clone; detach past the timeout so a long generation
    // cannot block the response. MLX may retain allocator/cache state after the
    // engine drop, so this endpoint does not promise a full process-wide cache
    // purge.
    // Own the drain in a detached task so cancellation of this request cannot
    // release the engine or its resident quota permit prematurely.
    let drained = tokio::spawn(drain_and_drop(engine, resident_permit, DRAIN_TIMEOUT))
        .await
        .map_err(|e| ServerError::InternalError(format!("model unload task failed: {e}")))?;
    if drained {
        tracing::info!(model_name = %name, "Model unloaded");
        Ok(StatusCode::NO_CONTENT)
    } else {
        tracing::info!(model_name = %name, "Model unload deferred; request still in flight");
        Ok(StatusCode::ACCEPTED)
    }
}

/// Wait until `engine` is solely owned here, then drop it.
///
/// Returns `true` if dropped within `timeout`; `false` if it timed out and the
/// final drop was handed to a detached task. Dropping is intentionally not gated
/// on the process-wide GPU gate: engine teardown frees MLX buffers but never
/// runs an `eval`, so it cannot race the cross-model output-array table that the
/// gate protects.
async fn drain_and_drop(
    mut engine: Arc<Engine>,
    resident_permit: Option<OwnedSemaphorePermit>,
    timeout: Duration,
) -> bool {
    let start = Instant::now();
    loop {
        match Arc::try_unwrap(engine) {
            Ok(owned) => {
                drop(owned);
                drop(resident_permit);
                return true;
            }
            Err(shared) => {
                if start.elapsed() >= timeout {
                    tokio::spawn(drain_in_background(shared, resident_permit));
                    return false;
                }
                engine = shared;
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }
    }
}

/// Poll until the detached engine reference is sole-owned, then drop it.
async fn drain_in_background(engine: Arc<Engine>, resident_permit: Option<OwnedSemaphorePermit>) {
    drain_in_background_with_warning_interval(engine, resident_permit, DRAIN_WARNING_INTERVAL)
        .await;
}

async fn drain_in_background_with_warning_interval(
    mut engine: Arc<Engine>,
    resident_permit: Option<OwnedSemaphorePermit>,
    warning_interval: Duration,
) {
    let start = Instant::now();
    let mut last_warning = start;
    loop {
        match Arc::try_unwrap(engine) {
            Ok(owned) => {
                drop(owned);
                drop(resident_permit);
                return;
            }
            Err(shared) => {
                let now = Instant::now();
                if now.duration_since(last_warning) >= warning_interval {
                    let elapsed_ms =
                        u64::try_from(now.duration_since(start).as_millis()).unwrap_or(u64::MAX);
                    let strong_count =
                        u64::try_from(Arc::strong_count(&shared)).unwrap_or(u64::MAX);
                    tracing::warn!(
                        elapsed_ms,
                        strong_count,
                        "Model unload is still waiting for references to drain"
                    );
                    last_warning = now;
                }
                engine = shared;
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }
    }
}

fn model_object(name: String) -> ModelObject {
    ModelObject {
        id: name,
        object: "model",
        created: chrono::Utc::now().timestamp(),
        owned_by: "local".to_owned(),
    }
}

/// Build a sorted, stable list of [`ModelObject`]s from an iterator of model names.
fn model_objects_sorted<'a>(names: impl Iterator<Item = &'a str>) -> Vec<ModelObject> {
    let mut sorted: Vec<&str> = names.collect();
    sorted.sort_unstable();
    sorted
        .into_iter()
        .map(|name| model_object(name.to_owned()))
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::significant_drop_tightening,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    use crate::router::{Router, RuntimeLoadError};
    use crate::state::AppState;
    use tracing::{Event, Subscriber, field::Visit};
    use tracing_subscriber::{Layer, layer::Context, prelude::*};

    #[derive(Clone, Default)]
    struct DrainWarningCapture {
        fields: Arc<Mutex<Vec<(u64, u64)>>>,
    }

    impl<S> Layer<S> for DrainWarningCapture
    where
        S: Subscriber,
    {
        fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
            if *event.metadata().level() != tracing::Level::WARN {
                return;
            }
            let mut visitor = DrainWarningVisitor::default();
            event.record(&mut visitor);
            if let (Some(elapsed_ms), Some(strong_count)) =
                (visitor.elapsed_ms, visitor.strong_count)
            {
                self.fields.lock().unwrap().push((elapsed_ms, strong_count));
            }
        }
    }

    #[derive(Default)]
    struct DrainWarningVisitor {
        elapsed_ms: Option<u64>,
        strong_count: Option<u64>,
    }

    impl Visit for DrainWarningVisitor {
        fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}

        fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
            match field.name() {
                "elapsed_ms" => self.elapsed_ms = Some(value),
                "strong_count" => self.strong_count = Some(value),
                _ => {}
            }
        }
    }

    fn build_state(toml: &str, engines: HashMap<String, Arc<Engine>>) -> SharedState {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        // A stub provider satisfies the "at least one [[models]] or [provider.*]"
        // config rule; it does not affect direct local-engine routing.
        let full = format!("[provider.stub]\nurl = \"http://127.0.0.1:1\"\n\n{toml}");
        std::fs::write(&path, full).unwrap();
        let config = crate::config::load_config_file(&path, None).unwrap();
        let router = Router::from_config(&config, engines).unwrap();
        Arc::new(AppState {
            router,
            config,
            http_client: reqwest::Client::new(),
            metrics: None,
        })
    }

    fn stub_engines(names: &[&str]) -> HashMap<String, Arc<Engine>> {
        names
            .iter()
            .map(|n| ((*n).to_owned(), Arc::new(Engine::test_stub(n))))
            .collect()
    }

    #[test]
    fn test_model_list_is_sorted_alphabetically() {
        let names = ["zebra", "alpha", "middle"];
        let data = model_objects_sorted(names.iter().copied());
        let ids: Vec<&str> = data.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "middle", "zebra"]);
    }

    #[test]
    fn test_model_list_empty_input() {
        let data = model_objects_sorted(std::iter::empty());
        assert!(data.is_empty());
    }

    #[test]
    fn test_model_objects_have_correct_fields() {
        let data = model_objects_sorted(std::iter::once("test-model"));
        let obj = data.first().unwrap();
        assert_eq!(obj.object, "model");
        assert_eq!(obj.owned_by, "local");
        assert!(obj.created > 0);
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
    async fn unload_disabled_returns_forbidden_with_unloading_message() {
        let state = build_state(
            "[local]\nallow_runtime_model_load = false\n",
            HashMap::new(),
        );

        let err = unload_model(State(state), Path("model".to_owned()))
            .await
            .unwrap_err();

        assert!(matches!(
            err,
            ServerError::Forbidden(message)
                if message == "runtime model unloading is disabled; set local.allow_runtime_model_load = true to enable it"
        ));
    }

    #[tokio::test]
    async fn load_existing_name_conflicts() {
        // allow_runtime on + an already-loaded "llama"; naming it again is a
        // conflict caught before any (impossible, in-test) GPU load.
        let state = build_state(
            "[server]\napi_key = \"test-api-key\"\n[local]\nallow_runtime_model_load = true\n",
            stub_engines(&["llama"]),
        );
        let body = Bytes::from_static(b"{\"path\":\"some/path\",\"name\":\"llama\"}");
        let err = load_model(State(state), body).await.unwrap_err();
        assert!(matches!(err, ServerError::Conflict(_)));
    }

    #[tokio::test]
    async fn startup_engines_do_not_consume_runtime_model_budget() {
        let state = build_state(
            "[server]\napi_key = \"test-api-key\"\n[local]\nallow_runtime_model_load = true\nruntime_max_loaded_models = 1\n",
            stub_engines(&["startup"]),
        );
        let body = Bytes::from_static(b"{\"path\":\"org/model\",\"name\":\"runtime\"}");
        let err = load_model(State(state), body).await.unwrap_err();
        let ServerError::BadRequest(message) = err else {
            panic!("expected runtime resolver bad request, got: {err}");
        };
        assert_eq!(
            message,
            "model 'org/model' not found locally: could not read HF cache ref for 'org/model': No such file or directory (os error 2); pre-download it (e.g. `huggingface-cli download org/model`)"
        );
    }

    #[test]
    fn runtime_load_errors_map_by_typed_variant() {
        assert!(matches!(
            map_runtime_load_error(RuntimeLoadError::ResidentBudgetExhausted),
            ServerError::BadRequest(message)
                if message == "runtime model budget reached; unload a runtime model before loading another"
        ));
        assert!(matches!(
            map_runtime_load_error(RuntimeLoadError::LoadGateClosed),
            ServerError::InternalError(message) if message == "runtime load gate closed"
        ));
        assert!(matches!(
            map_runtime_load_error(RuntimeLoadError::ResidentGateClosed),
            ServerError::InternalError(message) if message == "runtime resident-model gate closed"
        ));
    }

    #[tokio::test]
    async fn unload_unknown_returns_not_found() {
        let state = build_state(
            "[server]\napi_key = \"test-api-key\"\n[local]\nallow_runtime_model_load = true\n",
            HashMap::new(),
        );
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

            [server]
            api_key = "test-api-key"

            [local]
            allow_runtime_model_load = true
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
        let state = build_state(
            "[server]\napi_key = \"test-api-key\"\n[local]\nallow_runtime_model_load = true\n",
            stub_engines(&["llama"]),
        );
        let status = unload_model(State(Arc::clone(&state)), Path("llama".to_owned()))
            .await
            .unwrap();
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert!(!state.router.contains_engine("llama"));
        assert!(state.router.local_model_names().is_empty());
    }

    #[tokio::test]
    async fn drain_and_drop_drops_sole_owner_immediately() {
        let engine = Arc::new(Engine::test_stub("solo"));
        assert!(drain_and_drop(engine, None, Duration::from_secs(1)).await);
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
        assert!(drain_and_drop(engine, None, Duration::from_secs(5)).await);
        assert!(
            start.elapsed() >= Duration::from_millis(100),
            "should have waited for the clone to drop"
        );
    }

    #[tokio::test]
    async fn drain_and_drop_times_out_and_detaches() {
        let engine = Arc::new(Engine::test_stub("stuck"));
        let clone = Arc::clone(&engine); // held past the timeout
        let drained = drain_and_drop(engine, None, Duration::from_millis(80)).await;
        assert!(!drained, "should time out while a reference is held");
        drop(clone); // let the detached task finish
    }

    #[tokio::test(flavor = "current_thread")]
    async fn background_drain_warns_periodically_while_references_remain() {
        let capture = DrainWarningCapture::default();
        let captured_fields = Arc::clone(&capture.fields);
        let subscriber = tracing_subscriber::registry().with(capture);
        let _subscriber_guard = tracing::subscriber::set_default(subscriber);
        let engine = Arc::new(Engine::test_stub("stuck"));
        let in_flight = Arc::clone(&engine);

        let draining = tokio::spawn(drain_in_background_with_warning_interval(
            engine,
            None,
            Duration::from_millis(30),
        ));
        tokio::time::sleep(Duration::from_millis(125)).await;

        let warnings = captured_fields.lock().unwrap().clone();
        assert!(warnings.len() >= 2, "captured warnings: {warnings:?}");
        assert!(
            warnings
                .iter()
                .all(|(elapsed_ms, strong_count)| *elapsed_ms >= 30 && *strong_count == 2),
            "captured warnings: {warnings:?}"
        );
        drop(in_flight);
        tokio::time::timeout(Duration::from_secs(1), draining)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn resident_permit_stays_held_until_unload_drain_finishes() {
        let state = build_state(
            "[server]\napi_key = \"test-api-key\"\n[local]\nallow_runtime_model_load = true\nruntime_max_loaded_models = 1\n",
            HashMap::new(),
        );
        let load_permit = state.router.acquire_runtime_load().await.unwrap();
        state
            .router
            .insert_runtime_engine(
                "runtime".to_owned(),
                Arc::new(Engine::test_stub("runtime")),
                load_permit.into_resident_permit(),
            )
            .unwrap();
        let removed = state.router.remove_runtime_engine("runtime").unwrap();
        let (engine, resident_permit) = removed.into_parts();
        let in_flight = Arc::clone(&engine);
        let draining = tokio::spawn(drain_and_drop(
            engine,
            resident_permit,
            Duration::from_secs(1),
        ));

        assert!(state.router.acquire_runtime_load().await.is_err());
        drop(in_flight);
        assert!(draining.await.unwrap());
        assert!(state.router.acquire_runtime_load().await.is_ok());
    }

    #[tokio::test]
    async fn canceled_unload_handler_keeps_resident_permit_until_drain() {
        let state = build_state(
            "[server]\napi_key = \"test-api-key\"\n[local]\nallow_runtime_model_load = true\nruntime_max_loaded_models = 1\n",
            HashMap::new(),
        );
        let engine = Arc::new(Engine::test_stub("runtime"));
        let in_flight = Arc::clone(&engine);
        let load_permit = state.router.acquire_runtime_load().await.unwrap();
        state
            .router
            .insert_runtime_engine(
                "runtime".to_owned(),
                engine,
                load_permit.into_resident_permit(),
            )
            .unwrap();
        assert!(state.router.acquire_runtime_load().await.is_err());

        let unload = tokio::spawn(unload_model(
            State(Arc::clone(&state)),
            Path("runtime".to_owned()),
        ));
        assert!(
            tokio::time::timeout(Duration::from_secs(1), async {
                while state.router.contains_engine("runtime") {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .is_ok(),
            "unload should detach the engine promptly"
        );
        unload.abort();
        let _ = unload.await;

        assert!(
            tokio::time::timeout(
                Duration::from_millis(20),
                state.router.acquire_runtime_load()
            )
            .await
            .unwrap()
            .is_err(),
            "canceling unload released the resident permit too early"
        );
        drop(in_flight);
        let acquired = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if state.router.acquire_runtime_load().await.is_ok() {
                    break true;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert!(acquired);
    }
}
