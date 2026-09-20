use axum::{
    Json,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;

use crate::capacity::{CapacityErrorEnvelope, CapacityModelNotFoundError, RegistrationError};
use crate::error::ServerError;
use crate::state::SharedState;

#[derive(Deserialize)]
pub struct CapacityQuery {
    model: String,
    #[serde(default, rename = "schemaVersion")]
    schema_version: Option<u32>,
}

/// Return fixed model context limits; pressure is advisory telemetry.
pub async fn capacity(
    State(state): State<SharedState>,
    Query(query): Query<CapacityQuery>,
) -> Response {
    if query.model.trim().is_empty() {
        return ServerError::BadRequest("model must not be blank".to_owned()).into_response();
    }
    if query.schema_version == Some(2) {
        return match state.capacity.fast_session_contract(&query.model) {
            Ok(contract) => Json(contract).into_response(),
            Err(error) => ServerError::Conflict(error.to_string()).into_response(),
        };
    }
    match state.capacity.snapshot(&query.model) {
        Ok(snapshot) => Json(snapshot).into_response(),
        Err(RegistrationError::UnknownModel(model)) => (
            StatusCode::NOT_FOUND,
            Json(CapacityErrorEnvelope::new(CapacityModelNotFoundError::new(
                model,
            ))),
        )
            .into_response(),
        Err(error) => ServerError::InternalError(error.to_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use axum::body::Body;
    use higgs_engine::MlxMemorySnapshot;
    use http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::capacity::{CapacityRegistry, ModelCapacityFacts};
    use crate::router::Router;
    use crate::state::AppState;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn state_with(registry: Arc<CapacityRegistry>) -> Arc<AppState> {
        let config = crate::config::HiggsConfig::default();
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        Arc::new(AppState::with_capacity_registry(
            router,
            config,
            reqwest::Client::new(),
            None,
            registry,
        ))
    }

    fn active_facts() -> ModelCapacityFacts {
        super::super::super::capacity::route_test_facts(
            "escha",
            MlxMemorySnapshot::default(),
            0,
            32768,
        )
    }

    async fn body(response: axum::response::Response) -> serde_json::Value {
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
    }

    #[tokio::test]
    async fn known_unloaded_is_200_with_zero_capacity_fields() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let app = crate::build_router(state_with(registry), 30.0, None, 0, 1024, None);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=escha")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body(response).await;
        assert_eq!(json["availability"], "unavailable");
        for field in [
            "safeTotalTokens",
            "recommendedOutputTokens",
            "maxPromptTokens",
            "retainedSessionTokens",
            "retainedBytes",
            "prefixCacheBytes",
        ] {
            assert_eq!(json[field], 0, "{field} must be zero while unloaded");
        }
    }

    #[tokio::test]
    async fn active_route_returns_the_stored_snapshot_unchanged() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let facts = active_facts();
        registry.refresh_memory(MlxMemorySnapshot::default());
        let ticket = registry.begin_registration("escha".to_owned()).unwrap();
        registry.commit_active(ticket, facts).unwrap().publish();
        let expected = serde_json::to_value(registry.snapshot("escha").unwrap()).unwrap();
        let app = crate::build_router(state_with(Arc::clone(&registry)), 30.0, None, 0, 1024, None);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=escha")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body(response).await, expected);
    }

    #[tokio::test]
    async fn v2_route_returns_only_the_validated_retained_byte_contract() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let mut facts = active_facts();
        facts.retained_bytes_ceiling = 4 * GIB;
        facts.retained_session_tokens = 96_576;
        facts.guaranteed_retained_sessions = 1;
        facts.retained_bytes_per_token = 128 * 1024;
        registry.refresh_memory(MlxMemorySnapshot::default());
        let ticket = registry.begin_registration("escha".to_owned()).unwrap();
        registry.commit_active(ticket, facts).unwrap().publish();
        let app = crate::build_router(state_with(registry), 30.0, None, 0, 1024, None);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=escha&schemaVersion=2")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let json = body(response).await;
        assert_eq!(json["schemaVersion"], 2);
        assert_eq!(json["model"], "escha");
        assert_eq!(json["retainedBudgetBytes"], 4 * GIB);
        assert!(json.get("retainedSessionTokens").is_none());
    }

    #[tokio::test]
    async fn active_route_refreshes_allocator_before_returning_capacity() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let facts = active_facts();
        registry.refresh_memory(MlxMemorySnapshot::default());
        let ticket = registry.begin_registration("escha".to_owned()).unwrap();
        registry
            .commit_active(ticket, facts.clone())
            .unwrap()
            .publish();
        let before = registry.snapshot("escha").unwrap();
        assert!(before.max_prompt_tokens > 0);
        let revision = registry.admission_test_memory().1;
        *registry.test_memory_measurement.lock().unwrap() = Some(Ok(MlxMemorySnapshot {
            active_bytes: 24 * GIB,
            peak_bytes: 24 * GIB,
            ..MlxMemorySnapshot::default()
        }));
        let app = crate::build_router(state_with(Arc::clone(&registry)), 30.0, None, 0, 1024, None);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=escha")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body(response).await;
        assert_eq!(json["availability"], "available");
        assert_eq!(json["safeTotalTokens"], before.safe_total_tokens);
        assert_eq!(json["generation"], before.generation);
        assert_eq!(registry.admission_test_memory().1, revision);
        assert_eq!(registry.active_reservation_count("escha"), 0);
    }

    #[tokio::test]
    async fn active_route_measurement_failure_never_returns_stale_available_capacity() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let facts = active_facts();
        registry.refresh_memory(MlxMemorySnapshot::default());
        let ticket = registry.begin_registration("escha".to_owned()).unwrap();
        registry.commit_active(ticket, facts).unwrap().publish();
        assert!(registry.snapshot("escha").unwrap().max_prompt_tokens > 0);
        *registry.test_memory_measurement.lock().unwrap() = Some(Err(
            higgs_engine::MlxMemoryProbeError::QueryFailed("test allocator"),
        ));
        let app = crate::build_router(state_with(Arc::clone(&registry)), 30.0, None, 0, 1024, None);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=escha")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body(response).await;
        assert_eq!(json["availability"], "available");
        assert_eq!(json["safeTotalTokens"], 32768);
    }

    #[tokio::test]
    async fn unknown_model_is_exact_typed_404() {
        let registry = CapacityRegistry::new(Vec::new());
        let app = crate::build_router(state_with(registry), 30.0, None, 0, 1024, None);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=ghost")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body(response).await,
            serde_json::json!({
                "error": {
                    "type": "higgs_capacity_model_not_found",
                    "code": "model_not_found",
                    "model": "ghost"
                }
            })
        );
    }

    #[tokio::test]
    async fn capacity_route_uses_the_chat_authentication_layer() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        let app = crate::build_router(
            state_with(registry),
            30.0,
            Some("secret".to_owned()),
            0,
            1024,
            None,
        );
        let rejected = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=escha")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);

        let accepted = app
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=escha")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn missing_or_blank_model_is_bad_request_not_legacy_absence() {
        let registry = CapacityRegistry::new(["escha".to_owned()]);
        for uri in ["/v1/capacity", "/v1/capacity?model=%20%20"] {
            let app =
                crate::build_router(state_with(Arc::clone(&registry)), 30.0, None, 0, 1024, None);
            let response = app
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }
}
