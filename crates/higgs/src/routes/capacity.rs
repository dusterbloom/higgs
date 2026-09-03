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
}

/// Return the registry's already-computed snapshot without route-side policy math.
pub async fn capacity(
    State(state): State<SharedState>,
    Query(query): Query<CapacityQuery>,
) -> Response {
    if query.model.trim().is_empty() {
        return ServerError::BadRequest("model must not be blank".to_owned()).into_response();
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
    use higgs_engine::{EngineCostDescription, MlxMemorySnapshot, TransientPrefillEstimate};
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
        ModelCapacityFacts {
            model: "escha".to_owned(),
            model_fingerprint: "sha256:stored".to_owned(),
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
        registry.refresh_memory(facts.memory);
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
