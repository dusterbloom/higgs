use std::time::Instant;

use axum::{
    Json,
    extract::{Extension, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use bytes::Bytes;

use crate::{
    config::ApiFormat,
    error::ServerError,
    metrics::{RequestMetricsContext, RequestRecord},
    router::ResolvedRoute,
    state::SharedState,
    types::openai::{
        EmbeddingInput, EmbeddingObject, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    },
};

/// POST /v1/embeddings — text-only.
///
/// Images are rejected by the type system: [`EmbeddingInput`] accepts only a
/// plain string or an array of strings (see [`crate::types::openai::EmbeddingInput`]),
/// so a request carrying an `image_url` content part fails serde
/// deserialization with a 400 before this handler runs. There is no code path
/// that accepts image data here.
#[allow(clippy::too_many_lines)]
pub async fn embeddings(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ServerError> {
    let req: EmbeddingRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;
    request_metrics.set_requested_model(&req.model);

    let resolved = state
        .router
        .resolve(&req.model, None)
        .await
        .map_err(ServerError::ModelNotFound)?;

    match resolved {
        ResolvedRoute::Higgs {
            engine,
            model_name,
            routing_method,
        } => {
            let inputs = match &req.input {
                EmbeddingInput::Single(s) => vec![s.clone()],
                EmbeddingInput::Multiple(v) => v.clone(),
            };

            if inputs.is_empty() {
                return Err(ServerError::BadRequest(
                    "input must not be empty".to_owned(),
                ));
            }

            let start = Instant::now();
            let mut data = Vec::new();
            let mut total_tokens: u32 = 0;

            for (idx, text) in inputs.iter().enumerate() {
                let encoding = engine
                    .tokenizer()
                    .encode(text.as_str(), false)
                    .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;

                let token_ids = encoding.get_ids();
                let token_count: u32 = token_ids
                    .len()
                    .try_into()
                    .map_err(|_| ServerError::BadRequest("Input too long".to_owned()))?;
                total_tokens = total_tokens.saturating_add(token_count);

                let embedding = engine
                    .embed(token_ids)
                    .map_err(|e| ServerError::InternalError(format!("Embedding error: {e}")))?;

                let index: u32 = idx
                    .try_into()
                    .map_err(|_| ServerError::BadRequest("Too many inputs".to_owned()))?;

                data.push(EmbeddingObject {
                    object: "embedding",
                    embedding,
                    index,
                });
            }

            if let Some(ref metrics) = state.metrics {
                metrics.record(RequestRecord {
                    id: 0,
                    timestamp: Instant::now(),
                    wallclock: chrono::Utc::now(),
                    model: Some(model_name),
                    provider: Some("higgs".to_owned()),
                    routing_method: routing_method.into(),
                    status: 200,
                    duration: start.elapsed(),
                    input_tokens: u64::from(total_tokens),
                    output_tokens: 0,
                    error_body: None,
                });
                request_metrics.mark_recorded();
            }

            Ok(Json(EmbeddingResponse {
                object: "list",
                data,
                model: req.model,
                usage: EmbeddingUsage {
                    prompt_tokens: total_tokens,
                    total_tokens,
                },
            })
            .into_response())
        }
        ResolvedRoute::Remote {
            provider_name,
            provider_url,
            provider_format,
            strip_auth,
            api_key,
            model_rewrite,
            routing_method,
            ..
        } => {
            if provider_format != ApiFormat::OpenAi {
                return Err(ServerError::BadRequest(
                    "Embeddings proxy only supported for OpenAI-format providers".to_owned(),
                ));
            }
            let proxy_body = if let Some(ref rewrite) = model_rewrite {
                crate::proxy::rewrite_model_in_body(&body, rewrite)?
            } else {
                body
            };
            let start = Instant::now();
            let response = crate::proxy::proxy_request(
                &state.http_client,
                &provider_url,
                "/v1/embeddings",
                proxy_body,
                &headers,
                strip_auth,
                api_key.as_deref(),
            )
            .await;
            let metrics_model = model_rewrite.as_deref().unwrap_or(&req.model).to_owned();
            if let Some(ref metrics) = state.metrics {
                let status = response.as_ref().map_or(502, |resp| resp.status().as_u16());
                metrics.record(RequestRecord {
                    id: 0,
                    timestamp: Instant::now(),
                    wallclock: chrono::Utc::now(),
                    model: Some(metrics_model),
                    provider: Some(provider_name),
                    routing_method: routing_method.into(),
                    status,
                    duration: start.elapsed(),
                    input_tokens: 0,
                    output_tokens: 0,
                    error_body: None,
                });
                request_metrics.mark_recorded();
            }
            response
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// `EmbeddingInput` is `Single(String) | Multiple(Vec<String>)`, so an
    /// OpenAI-style `image_url` content part (an object) cannot deserialize —
    /// serde rejects it and the route 400s before the handler runs.
    #[test]
    fn embedding_input_rejects_image_url_content() {
        // Single-input with an image_url object.
        let single = json!({
            "model": "test-model",
            "input": {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        });
        let err = serde_json::from_value::<EmbeddingRequest>(single).unwrap_err();
        assert!(!err.to_string().is_empty());

        // Array-of-inputs with an image_url object inside.
        let multiple = json!({
            "model": "test-model",
            "input": [
                "plain text",
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}
            ]
        });
        assert!(serde_json::from_value::<EmbeddingRequest>(multiple).is_err());
    }

    /// Plain strings still deserialize (guards against over-strictness).
    #[test]
    fn embedding_input_accepts_plain_strings() {
        let single = json!({"model": "test-model", "input": "hello"});
        let req: EmbeddingRequest = serde_json::from_value(single).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Single(_)));

        let multiple = json!({"model": "test-model", "input": ["a", "b"]});
        let req: EmbeddingRequest = serde_json::from_value(multiple).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Multiple(_)));
    }
}
