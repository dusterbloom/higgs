use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json,
    extract::{Extension, State},
    http::HeaderMap,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
};
use bytes::Bytes;
use tokio_stream::Stream;

use crate::{
    anthropic_adapter::{
        anthropic_messages_to_engine, openai_finish_to_anthropic_stop, render_anthropic_markers,
    },
    config::ApiFormat,
    error::ServerError,
    media::{MediaExtractor, MediaItem},
    metrics::{MetricsStore, RequestMetricsContext},
    router::ResolvedRoute,
    routes::chat::{check_vision_capability, map_engine_error},
    state::{Engine, SharedState},
    types::anthropic::{
        AnthropicUsage, ContentBlockResponse, ContentBlockStartEvent, ContentBlockStartPayload,
        ContentBlockStopEvent, CountTokensRequest, CountTokensResponse, CreateMessageRequest,
        CreateMessageResponse, MessageDelta, MessageDeltaEvent, MessageStartEvent,
        MessageStartPayload, MessageStopEvent, TextDelta,
    },
};
use higgs_models::SamplingParams;

#[allow(clippy::too_many_lines)]
pub async fn create_message(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    let mut req: CreateMessageRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;
    request_metrics.set_requested_model(&req.model);

    if req.messages.is_empty() {
        return Err(ServerError::BadRequest(
            "messages array must not be empty".to_owned(),
        ));
    }

    let messages_json = serde_json::to_value(&req.messages).ok().and_then(|v| {
        if let serde_json::Value::Array(a) = v {
            Some(a)
        } else {
            None
        }
    });
    let resolved = state
        .router
        .resolve(&req.model, messages_json.as_deref())
        .await
        .map_err(ServerError::ModelNotFound)?;

    match resolved {
        ResolvedRoute::Higgs {
            engine,
            model_name,
            routing_method,
            ..
        } => {
            req.model = model_name;
            let start = Instant::now();
            if req.stream == Some(true) {
                let stream = create_message_stream(
                    Arc::clone(&state),
                    req,
                    engine,
                    state.metrics.clone(),
                    routing_method,
                )
                .await?;
                let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                if state.metrics.is_some() {
                    request_metrics.mark_recorded();
                }
                Ok(sse.into_response())
            } else {
                let response =
                    create_message_non_streaming(Arc::clone(&state), req, engine).await?;
                if let Some(ref metrics) = state.metrics {
                    metrics.record(crate::metrics::RequestRecord {
                        id: 0,
                        timestamp: Instant::now(),
                        wallclock: chrono::Utc::now(),
                        model: Some(response.model.clone()),
                        provider: Some("higgs".to_owned()),
                        routing_method: routing_method.into(),
                        status: 200,
                        duration: start.elapsed(),
                        input_tokens: u64::from(response.usage.input_tokens),
                        output_tokens: u64::from(response.usage.output_tokens),
                        error_body: None,
                    });
                    request_metrics.mark_recorded();
                }
                Ok(Json(response).into_response())
            }
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
            let start = Instant::now();
            let metrics_model = model_rewrite.as_deref().unwrap_or(&req.model).to_owned();
            let is_streaming = req.stream == Some(true);
            let mut usage = (0u64, 0u64);

            let result = match provider_format {
                ApiFormat::Anthropic => {
                    let proxy_body = if let Some(ref rewrite) = model_rewrite {
                        crate::proxy::rewrite_model_in_body(&body, rewrite)?
                    } else {
                        body
                    };
                    if is_streaming {
                        crate::proxy::proxy_request(
                            &state.http_client,
                            &provider_url,
                            "/v1/messages",
                            proxy_body,
                            &headers,
                            strip_auth,
                            api_key.as_deref(),
                        )
                        .await
                    } else {
                        let (status, resp_bytes) = crate::proxy::send_and_read(
                            &state.http_client,
                            &provider_url,
                            "/v1/messages",
                            proxy_body,
                            &headers,
                            strip_auth,
                            api_key.as_deref(),
                        )
                        .await?;
                        usage = crate::proxy::extract_usage(&resp_bytes);
                        Ok((
                            status,
                            [(axum::http::header::CONTENT_TYPE, "application/json")],
                            resp_bytes,
                        )
                            .into_response())
                    }
                }
                ApiFormat::OpenAi => {
                    let translated = crate::translate::anthropic_to_openai_request(&body)?;
                    let proxy_body = if let Some(ref rewrite) = model_rewrite {
                        crate::proxy::rewrite_model_in_body(&translated, rewrite)?
                    } else {
                        translated
                    };

                    let upstream = crate::proxy::send_to_provider(
                        &state.http_client,
                        &provider_url,
                        "/v1/chat/completions",
                        proxy_body,
                        &headers,
                        strip_auth,
                        api_key.as_deref(),
                    )
                    .await?;
                    let upstream_status = upstream.status().as_u16();

                    if upstream_status >= 400 {
                        let status_code = axum::http::StatusCode::from_u16(upstream_status)
                            .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
                        let resp_bytes = upstream.bytes().await.map_err(|e| {
                            ServerError::ProxyError(format!("Failed to read response: {e}"))
                        })?;
                        Ok((
                            status_code,
                            [(axum::http::header::CONTENT_TYPE, "application/json")],
                            resp_bytes,
                        )
                            .into_response())
                    } else if is_streaming {
                        let stream =
                            crate::translate::openai_stream_to_anthropic(upstream, req.model);
                        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                        Ok(sse.into_response())
                    } else {
                        let resp_bytes = upstream.bytes().await.map_err(|e| {
                            ServerError::ProxyError(format!("Failed to read response: {e}"))
                        })?;
                        usage = crate::proxy::extract_usage(&resp_bytes);
                        let translated_resp = crate::translate::openai_response_to_anthropic(
                            &resp_bytes,
                            &req.model,
                        )?;
                        Ok((
                            [(axum::http::header::CONTENT_TYPE, "application/json")],
                            translated_resp,
                        )
                            .into_response())
                    }
                }
            };
            if let Some(ref metrics) = state.metrics {
                let status = result.as_ref().map_or(502, |resp| resp.status().as_u16());
                if !(200..300).contains(&status) {
                    usage = (0, 0);
                }
                metrics.record(crate::metrics::RequestRecord {
                    id: 0,
                    timestamp: Instant::now(),
                    wallclock: chrono::Utc::now(),
                    model: Some(metrics_model),
                    provider: Some(provider_name),
                    routing_method: routing_method.into(),
                    status,
                    duration: start.elapsed(),
                    input_tokens: usage.0,
                    output_tokens: usage.1,
                    error_body: None,
                });
                request_metrics.mark_recorded();
            }
            result
        }
    }
}

async fn create_message_non_streaming(
    state: SharedState,
    req: CreateMessageRequest,
    engine: Arc<Engine>,
) -> Result<CreateMessageResponse, ServerError> {
    let max_tokens = req.max_tokens;
    let speculation =
        higgs_models::Speculation::parse(req.speculation.as_deref()).map_err(|v| {
            ServerError::BadRequest(format!(
                "invalid 'speculation' value '{v}' (expected auto|dflash|mtp|none)"
            ))
        })?;
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        speculation,
        ..SamplingParams::default()
    };
    let stop_sequences = req.stop_sequences.unwrap_or_default();

    // Extract media and gate on vision capability, mirroring the OpenAI chat
    // route: a strict 400 when images are sent to a model that cannot see them.
    let media_extractor = MediaExtractor::new(
        state.config.server.max_image_bytes,
        state.config.server.image_fetch_timeout,
        state.config.server.max_image_dimension,
    )?;
    let media = media_extractor
        .extract_anthropic(&req.messages, req.system.as_ref())
        .await?;
    check_vision_capability(&media, engine.is_vlm(), engine.model_name())?;

    // Build effective messages: text blocks with the family marker spliced at
    // each image block's true position. Text-only requests pass through
    // unchanged. The marker tokens are expanded into sentinel runs below.
    let effective_messages = if media.is_empty() {
        req.messages.clone()
    } else {
        render_anthropic_markers(&req.messages, engine.image_marker_text())
    };
    let engine_messages = anthropic_messages_to_engine(&effective_messages, req.system.as_ref());
    let tools = req.tools.as_deref();
    let thinking_enabled = crate::reasoning::effective_thinking_enabled(
        engine.enable_thinking(),
        &[engine.model_name(), req.model.as_str()],
        None,
        None,
    );

    let prompt_tokens = engine
        .prepare_chat_prompt_with_thinking(&engine_messages, tools, thinking_enabled)
        .map_err(ServerError::Engine)?;
    let reservation = crate::capacity::admit_generation_request(
        &state,
        &req.model,
        crate::capacity::ExecutionPath::Cold,
        prompt_tokens.len(),
        max_tokens,
    )
    .await?;

    // Multimodal requests: hand the raw decoded images to the engine, which
    // preprocesses them into a family-native `ImageBatch` and expands each
    // family marker token into its sentinel run. Preprocessing failures are
    // client problems (bad/malformed image data) and map to strict 400s via
    // `map_engine_error`.
    let image_inputs = (!media.is_empty() && engine.is_vlm())
        .then(|| media.into_iter().map(MediaItem::into).collect());

    let output = tokio::task::spawn_blocking(move || {
        let _reservation = reservation;
        engine.generate_with_thinking(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            false,
            None,
            thinking_enabled,
            None,
            image_inputs,
            None,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(map_engine_error)?;

    let stop_reason = openai_finish_to_anthropic_stop(&output.finish_reason);
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    let output_text = output.text;
    // When thinking is enabled, the template already opened `<think>` so
    // the model output starts inside the thinking block. We wrap it in a
    // properly closed `<think>...</think>` before parsing so the parser
    // always sees balanced tags, even if the model was length-stopped.
    let parse_input = if thinking_enabled {
        if output_text.contains("</think>") {
            format!("<think>{output_text}")
        } else {
            format!("<think>{output_text}</think>")
        }
    } else {
        output_text.clone()
    };
    let reasoning_result = higgs_engine::reasoning_parser::parse_reasoning(&parse_input);
    let visible_text = if reasoning_result.reasoning.is_some() {
        reasoning_result.text
    } else {
        output_text
    };

    Ok(CreateMessageResponse {
        id: msg_id,
        message_type: "message",
        role: "assistant",
        content: vec![ContentBlockResponse {
            block_type: "text",
            text: visible_text,
        }],
        model: req.model,
        stop_reason: Some(stop_reason),
        usage: AnthropicUsage {
            input_tokens: output.prompt_tokens,
            output_tokens: output.completion_tokens,
        },
    })
}

#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
async fn create_message_stream(
    state: SharedState,
    req: CreateMessageRequest,
    engine: Arc<Engine>,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, ServerError> {
    let max_tokens = req.max_tokens;
    let speculation =
        higgs_models::Speculation::parse(req.speculation.as_deref()).map_err(|v| {
            ServerError::BadRequest(format!(
                "invalid 'speculation' value '{v}' (expected auto|dflash|mtp|none)"
            ))
        })?;
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        speculation,
        ..SamplingParams::default()
    };
    let stop_sequences = req.stop_sequences.unwrap_or_default();

    // Extract media and gate on vision capability before the stream starts, so
    // images sent to a text-only model get a strict 400 rather than an
    // empty-looking stream.
    let media_extractor = MediaExtractor::new(
        state.config.server.max_image_bytes,
        state.config.server.image_fetch_timeout,
        state.config.server.max_image_dimension,
    )?;
    let media = media_extractor
        .extract_anthropic(&req.messages, req.system.as_ref())
        .await?;
    check_vision_capability(&media, engine.is_vlm(), engine.model_name())?;

    let effective_messages = if media.is_empty() {
        req.messages.clone()
    } else {
        render_anthropic_markers(&req.messages, engine.image_marker_text())
    };
    let engine_messages = anthropic_messages_to_engine(&effective_messages, req.system.as_ref());
    let tools = req.tools.as_deref();
    let thinking_enabled = crate::reasoning::effective_thinking_enabled(
        engine.enable_thinking(),
        &[engine.model_name(), req.model.as_str()],
        None,
        None,
    );

    let prompt_tokens = engine
        .prepare_chat_prompt_with_thinking(&engine_messages, tools, thinking_enabled)
        .map_err(ServerError::Engine)?;
    let reservation = crate::capacity::admit_generation_request(
        &state,
        &req.model,
        crate::capacity::ExecutionPath::Cold,
        prompt_tokens.len(),
        max_tokens,
    )
    .await?;

    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model = req.model;
    let prompt_token_count = u32::try_from(prompt_tokens.len())
        .map_err(|_| ServerError::BadRequest("Token count overflow".to_owned()))?;
    let image_inputs = (!media.is_empty() && engine.is_vlm())
        .then(|| media.into_iter().map(MediaItem::into).collect());

    // Spawn generation before creating the stream so prefill starts immediately
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::task::spawn_blocking(move || {
        let _reservation = reservation;
        let result = engine.generate_streaming_with_thinking(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            false,
            None,
            &tx,
            thinking_enabled,
            // Anthropic streaming does not surface prefill progress.
            false,
            None,
            image_inputs,
            None,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during Anthropic streaming");
        }
    });

    let start = Instant::now();
    let metrics_id = metrics.as_ref().map(|m| {
        m.record_pending(crate::metrics::RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: Some(model.clone()),
            provider: Some("higgs".to_owned()),
            routing_method: routing_method.into(),
            status: 200,
            duration: std::time::Duration::ZERO,
            input_tokens: u64::from(prompt_token_count),
            output_tokens: 0,
            error_body: None,
        })
    });

    let stream = async_stream::stream! {
        // 1. message_start
        let start_event = MessageStartEvent {
            event_type: "message_start",
            message: MessageStartPayload {
                id: msg_id.clone(),
                message_type: "message",
                role: "assistant",
                content: vec![],
                model: model.clone(),
                stop_reason: None,
                usage: AnthropicUsage {
                    input_tokens: prompt_token_count,
                    output_tokens: 0,
                },
            },
        };
        match serde_json::to_string(&start_event) {
            Ok(json) => yield Ok(Event::default().event("message_start").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 2. content_block_start
        let block_start = ContentBlockStartEvent {
            event_type: "content_block_start",
            index: 0,
            content_block: ContentBlockStartPayload {
                block_type: "text",
                text: String::new(),
            },
        };
        match serde_json::to_string(&block_start) {
            Ok(json) => yield Ok(Event::default().event("content_block_start").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 3. content_block_delta events (one per token)
        let mut final_stop_reason = None;
        let mut total_output_tokens: u32 = 0;
        let mut reasoning_tracker = if thinking_enabled {
            higgs_engine::reasoning_parser::StreamingReasoningTracker::new_inside_think()
        } else {
            higgs_engine::reasoning_parser::StreamingReasoningTracker::new()
        };

        let mut delta_writer = crate::sse::AnthropicDeltaWriter::new();
        while let Some(output) = rx.recv().await {
            let (visible, _reasoning) = reasoning_tracker.process(&output.new_text);

            if !visible.is_empty() {
                let td = TextDelta {
                    delta_type: "text_delta",
                    text: visible,
                };
                match delta_writer.write(&td) {
                    Ok(json) => yield Ok(Event::default().event("content_block_delta").data(json)),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            }
            total_output_tokens = output.completion_tokens;
            if let Some(reason) = output.finish_reason {
                final_stop_reason = Some(openai_finish_to_anthropic_stop(&reason));
            }
        }

        // Flush any remaining visible text buffered by the reasoning tracker
        let (flush_visible, _flush_reasoning) = reasoning_tracker.flush();
        if !flush_visible.is_empty() {
            let td = TextDelta {
                delta_type: "text_delta",
                text: flush_visible,
            };
            match delta_writer.write(&td) {
                Ok(json) => yield Ok(Event::default().event("content_block_delta").data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
            }
        }

        if let Some(ref m) = metrics {
            if let Some(id) = metrics_id {
                m.finalize_stream(id, u64::from(total_output_tokens), start.elapsed());
            }
        }

        // 4. content_block_stop
        let block_stop = ContentBlockStopEvent {
            event_type: "content_block_stop",
            index: 0,
        };
        match serde_json::to_string(&block_stop) {
            Ok(json) => yield Ok(Event::default().event("content_block_stop").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 5. message_delta
        let msg_delta = MessageDeltaEvent {
            event_type: "message_delta",
            delta: MessageDelta {
                stop_reason: final_stop_reason,
            },
            usage: AnthropicUsage {
                input_tokens: prompt_token_count,
                output_tokens: total_output_tokens,
            },
        };
        match serde_json::to_string(&msg_delta) {
            Ok(json) => yield Ok(Event::default().event("message_delta").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 6. message_stop
        let msg_stop = MessageStopEvent {
            event_type: "message_stop",
        };
        match serde_json::to_string(&msg_stop) {
            Ok(json) => yield Ok(Event::default().event("message_stop").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }
    };

    Ok(stream)
}

pub async fn count_tokens(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    let req: CountTokensRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;
    request_metrics.set_requested_model(&req.model);

    let messages_json = serde_json::to_value(&req.messages).ok().and_then(|v| {
        if let serde_json::Value::Array(a) = v {
            Some(a)
        } else {
            None
        }
    });
    let resolved = state
        .router
        .resolve(&req.model, messages_json.as_deref())
        .await
        .map_err(ServerError::ModelNotFound)?;

    match resolved {
        ResolvedRoute::Higgs {
            engine, model_name, ..
        } => {
            let engine_messages = anthropic_messages_to_engine(&req.messages, req.system.as_ref());
            let tools = req.tools.as_deref();
            let thinking_enabled = crate::reasoning::effective_thinking_enabled(
                engine.enable_thinking(),
                &[engine.model_name(), model_name.as_str()],
                None,
                None,
            );

            let tokens = engine
                .prepare_chat_prompt_with_thinking(&engine_messages, tools, thinking_enabled)
                .map_err(ServerError::Engine)?;

            let count = u32::try_from(tokens.len())
                .map_err(|_| ServerError::BadRequest("Token count overflow".to_owned()))?;

            Ok(Json(CountTokensResponse {
                input_tokens: count,
            })
            .into_response())
        }
        ResolvedRoute::Remote {
            stub_count_tokens,
            provider_url,
            provider_format,
            strip_auth,
            api_key,
            model_rewrite,
            ..
        } => {
            if stub_count_tokens || provider_format != ApiFormat::Anthropic {
                // OpenAI providers have no count_tokens equivalent; return stub
                return Ok(crate::proxy::stub_count_tokens_response());
            }
            let proxy_body = if let Some(ref rewrite) = model_rewrite {
                crate::proxy::rewrite_model_in_body(&body, rewrite)?
            } else {
                body
            };
            crate::proxy::proxy_request(
                &state.http_client,
                &provider_url,
                "/v1/messages/count_tokens",
                proxy_body,
                &headers,
                strip_auth,
                api_key.as_deref(),
            )
            .await
        }
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        CreateMessageRequest, ServerError, create_message_non_streaming, create_message_stream,
    };

    /// A valid 1x1 red PNG (passes byte-size and dimension checks).
    const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    fn image_message_body(stream: bool) -> serde_json::Value {
        serde_json::json!({
            "model": "stub-model",
            "max_tokens": 32,
            "stream": stream,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": TINY_PNG_B64
                }}
            ]}]
        })
    }

    async fn post_json(
        app: axum::Router,
        uri: &str,
        body: serde_json::Value,
    ) -> axum::http::Response<axum::body::Body> {
        use tower::ServiceExt as _;
        app.oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    async fn error_message(response: axum::http::Response<axum::body::Body>) -> (u16, String) {
        use http_body_util::BodyExt as _;
        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (
            status,
            json["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        )
    }

    fn stub_app() -> axum::Router {
        crate::build_router(
            crate::state::test_state_with_stub_engine("stub-model"),
            300.0,
            None,
            0,
            1024 * 1024,
            None,
        )
    }

    #[tokio::test]
    async fn test_create_message_image_on_text_model_returns_400() {
        let (status, msg) =
            error_message(post_json(stub_app(), "/v1/messages", image_message_body(false)).await)
                .await;
        assert_eq!(status, 400, "expected 400, got body: {msg}");
        assert!(
            msg.contains("does not support vision"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("stub-model"), "model name missing: {msg}");
    }

    #[tokio::test]
    async fn test_create_message_stream_image_on_text_model_returns_400() {
        // The gate runs before the stream starts, so even stream=true must
        // surface a strict 400 instead of an empty-looking SSE stream.
        let (status, msg) =
            error_message(post_json(stub_app(), "/v1/messages", image_message_body(true)).await)
                .await;
        assert_eq!(status, 400, "expected 400, got body: {msg}");
        assert!(
            msg.contains("does not support vision"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn test_create_message_text_request_passes_gate() {
        // A text-only request is not gated; the stub's generation failure
        // surfaces as a server error (500), proving the request got past the
        // vision gate and into the engine path.
        let body = serde_json::json!({
            "model": "stub-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}]
        });
        let (status, msg) = error_message(post_json(stub_app(), "/v1/messages", body).await).await;
        assert_eq!(
            status, 500,
            "stub generation must fail as a server error, got {status}: {msg}"
        );
    }

    #[tokio::test]
    async fn capacity_admission_rejects_both_anthropic_variants_before_worker_mutation() {
        let model = "prompt-limit-mutation-spy";
        let (state, engine) = crate::capacity::rejecting_route_test_state(model);
        let request = |stream| {
            serde_json::from_value::<CreateMessageRequest>(serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5120,
                "stream": stream
            }))
            .unwrap()
        };

        let blocking =
            create_message_non_streaming(Arc::clone(&state), request(false), Arc::clone(&engine))
                .await;
        assert!(matches!(blocking, Err(ServerError::CapacityExceeded(_))));
        let streaming = create_message_stream(
            Arc::clone(&state),
            request(true),
            Arc::clone(&engine),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await;
        assert!(matches!(streaming, Err(ServerError::CapacityExceeded(_))));
        assert!(engine.route_test_mutation_sequence().is_empty());
        assert_eq!(state.capacity.active_reservation_count(model), 0);
    }
}
