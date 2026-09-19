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
    metrics::{MetricsStore, RequestMetricsContext, StreamMetricsGuard},
    router::ResolvedRoute,
    routes::chat::{check_vision_capability, map_engine_error},
    state::{Engine, SharedState},
    types::anthropic::{
        AnthropicToolChoice, AnthropicUsage, ContentBlockResponse, CountTokensRequest,
        CountTokensResponse, CreateMessageRequest, CreateMessageResponse, MessageDelta,
        MessageDeltaEvent, MessageStartEvent, MessageStartPayload, MessageStopEvent, TextDelta,
    },
};
use higgs_models::SamplingParams;

struct AnthropicToolChoicePlan {
    prompt_tools: Option<Vec<serde_json::Value>>,
    constraint_schema: Option<serde_json::Value>,
    required_name: Option<String>,
}

impl AnthropicToolChoicePlan {
    const fn requires_call(&self) -> bool {
        self.constraint_schema.is_some()
    }
}

fn resolve_anthropic_tool_choice(
    choice: Option<&AnthropicToolChoice>,
    declared_tools: Option<&[serde_json::Value]>,
) -> Result<AnthropicToolChoicePlan, ServerError> {
    let available_tools = declared_tools.filter(|items| !items.is_empty());
    let required_name = match choice {
        None | Some(AnthropicToolChoice::Auto) => {
            return Ok(AnthropicToolChoicePlan {
                prompt_tools: available_tools.map(<[serde_json::Value]>::to_vec),
                constraint_schema: None,
                required_name: None,
            });
        }
        Some(AnthropicToolChoice::None) => {
            return Ok(AnthropicToolChoicePlan {
                prompt_tools: None,
                constraint_schema: None,
                required_name: None,
            });
        }
        Some(AnthropicToolChoice::Any) => None,
        Some(AnthropicToolChoice::Tool { name }) => {
            if name.is_empty() {
                return Err(ServerError::BadRequest(
                    "tool_choice tool name must not be empty".to_owned(),
                ));
            }
            Some(name.as_str())
        }
    };
    let required_tools = available_tools.ok_or_else(|| {
        ServerError::BadRequest("tool_choice requires at least one declared tool".to_owned())
    })?;
    let constraint_schema = anthropic_tool_call_schema(required_tools, required_name)?;
    Ok(AnthropicToolChoicePlan {
        prompt_tools: Some(required_tools.to_vec()),
        constraint_schema: Some(constraint_schema),
        required_name: required_name.map(str::to_owned),
    })
}

fn anthropic_tool_call_schema(
    tools: &[serde_json::Value],
    required_name: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let mut variants = Vec::new();
    let mut names = std::collections::HashSet::new();
    for tool in tools {
        let name = tool
            .get("name")
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| ServerError::BadRequest("tool name must not be empty".to_owned()))?;
        if !names.insert(name) {
            return Err(ServerError::BadRequest(format!(
                "duplicate tool name: {name}"
            )));
        }
        if required_name.is_some_and(|required| required != name) {
            continue;
        }
        let input_schema = tool.get("input_schema").cloned().unwrap_or_else(|| {
            serde_json::json!({
                "type": "object"
            })
        });
        if input_schema
            .get("type")
            .is_some_and(|kind| kind.as_str() != Some("object"))
        {
            return Err(ServerError::BadRequest(format!(
                "tool '{name}' input_schema must describe an object"
            )));
        }
        variants.push(serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"const": name},
                "arguments": input_schema
            },
            "required": ["name", "arguments"],
            "additionalProperties": false
        }));
    }

    if variants.is_empty() {
        return Err(ServerError::BadRequest(format!(
            "tool_choice tool '{}' was not declared",
            required_name.unwrap_or_default()
        )));
    }
    if let [variant] = variants.as_slice() {
        return Ok(variant.clone());
    }
    Ok(serde_json::json!({"oneOf": variants}))
}

fn build_anthropic_tool_constraint(
    schema: Option<&serde_json::Value>,
    engine: &Arc<Engine>,
) -> Result<Option<higgs_engine::constrained::ConstrainedGenerator>, ServerError> {
    let Some(constraint_schema) = schema else {
        return Ok(None);
    };
    let eos_id = engine.eos_token_ids().first().copied().unwrap_or(0);
    let vocab = higgs_engine::constrained::build_vocabulary(engine.tokenizer(), eos_id)
        .map_err(ServerError::Engine)?;
    higgs_engine::constrained::ConstrainedGenerator::from_tagged_json_schema(
        &constraint_schema.to_string(),
        &vocab,
    )
    .map(Some)
    .map_err(|error| ServerError::BadRequest(format!("Unsupported tool input schema: {error}")))
}

fn validate_anthropic_required_call(
    visible_text: &str,
    calls: &[(String, String, bool)],
    declared_tools: &[serde_json::Value],
    required_name: Option<&str>,
) -> Result<(), String> {
    let [(name, arguments, ended)] = calls else {
        return Err(format!(
            "required tool choice produced {} parser-visible tool calls, expected exactly one",
            calls.len()
        ));
    };
    if !visible_text.is_empty() {
        return Err(format!(
            "required tool call violated its grammar: {} bytes of visible text outside the tool_call envelope",
            visible_text.len()
        ));
    }
    if !ended {
        return Err("required tool call did not complete".to_owned());
    }
    if required_name.is_some_and(|expected| expected != name) {
        return Err(format!(
            "required tool call named '{name}', but tool_choice required '{}'",
            required_name.unwrap_or_default()
        ));
    }
    if !declared_tools
        .iter()
        .any(|tool| tool.get("name").and_then(serde_json::Value::as_str) == Some(name.as_str()))
    {
        return Err(format!(
            "required tool call named '{name}' is not a declared tool"
        ));
    }
    let parsed_arguments: serde_json::Value = serde_json::from_str(arguments)
        .map_err(|error| format!("tool '{name}' arguments are invalid JSON: {error}"))?;
    if !parsed_arguments.is_object() {
        return Err(format!("tool '{name}' arguments must be a JSON object"));
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub async fn create_message(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    // Captured before parsing and prompt preparation so TTFT measures
    // the client-observed wait, not only generation.
    let received_at = Instant::now();
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
                        timing: crate::metrics::RequestTiming::default(),
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
                    timing: crate::metrics::RequestTiming::default(),
                });
                request_metrics.mark_recorded();
            }
            result
        }
    }
}

#[allow(clippy::too_many_lines)]
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
    let tool_choice =
        resolve_anthropic_tool_choice(req.tool_choice.as_ref(), req.tools.as_deref())?;
    if tool_choice.requires_call() {
        return Err(ServerError::BadRequest(
            "Anthropic tool_choice 'any' and 'tool' currently require stream:true".to_owned(),
        ));
    }
    let tools = tool_choice.prompt_tools.as_deref();
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

    let watchdog = crate::capacity::request_watchdog(&state);
    let output = tokio::task::spawn_blocking(move || {
        crate::capacity::run_reserved_generation(reservation, watchdog, || {
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

fn anthropic_tool_schema(
    tools: Option<&[serde_json::Value]>,
) -> Option<higgs_engine::tool_parser::ToolSchema> {
    let normalized: Vec<serde_json::Value> = tools?
        .iter()
        .filter_map(|tool| {
            Some(serde_json::json!({
                "name": tool.get("name")?.as_str()?,
                "parameters": tool.get("input_schema")?.clone(),
            }))
        })
        .collect();
    higgs_engine::tool_parser::ToolSchema::from_tools(Some(&normalized))
}

fn anthropic_stream_error_json(message: &str) -> String {
    serde_json::json!({
        "type": "error",
        "error": {"type": "api_error", "message": message}
    })
    .to_string()
}

#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
async fn create_message_stream(
    state: SharedState,
    req: CreateMessageRequest,
    engine: Arc<Engine>,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
    received_at: Instant,
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
    let tool_choice =
        resolve_anthropic_tool_choice(req.tool_choice.as_ref(), req.tools.as_deref())?;
    let required_call = tool_choice.requires_call();
    let required_name = tool_choice.required_name.clone();
    let required_tools =
        required_call.then(|| tool_choice.prompt_tools.clone().unwrap_or_default());
    let tools = tool_choice.prompt_tools.as_deref();
    let stream_includes_tools = tools.is_some_and(|items| !items.is_empty());
    let tool_schema = anthropic_tool_schema(tools);
    let thinking_enabled = crate::reasoning::effective_thinking_enabled(
        engine.enable_thinking(),
        &[engine.model_name(), req.model.as_str()],
        None,
        None,
    ) && !required_call;
    let constraint =
        build_anthropic_tool_constraint(tool_choice.constraint_schema.as_ref(), &engine)?;

    let prompt_tokens = engine
        .prepare_chat_prompt_with_thinking(&engine_messages, tools, thinking_enabled)
        .map_err(ServerError::Engine)?;
    let reservation = crate::capacity::admit_generation_request(
        &state,
        &req.model,
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
    let (terminal_tx, terminal_rx) = tokio::sync::oneshot::channel::<crate::sse::WorkerTerminal>();
    let watchdog = crate::capacity::request_watchdog(&state);

    tokio::task::spawn_blocking(move || {
        let result = crate::capacity::run_reserved_generation(reservation, watchdog, || {
            engine.generate_streaming_with_thinking(
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
                constraint,
                image_inputs,
                None,
            )
        });
        if let Err(e) = &result {
            tracing::error!(error = %e, "Generation error during Anthropic streaming");
        }
        let _ = terminal_tx.send(crate::sse::WorkerTerminal::from_engine_result(result));
    });

    let start = received_at;
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
            timing: crate::metrics::RequestTiming::default(),
        })
    });

    let mut metrics_guard = StreamMetricsGuard::new(metrics.clone(), metrics_id, start);
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

        let mut final_stop_reason = None;
        let mut total_output_tokens: u32 = 0;
        let mut reasoning_tracker = if thinking_enabled {
            higgs_engine::reasoning_parser::StreamingReasoningTracker::new_inside_think()
        } else {
            higgs_engine::reasoning_parser::StreamingReasoningTracker::new()
        };

        let mut tool_tracker = higgs_engine::tool_parser::IncrementalToolCallTracker::new(
            stream_includes_tools,
            tool_schema,
        );
        let mut text_delta_writer = crate::sse::AnthropicDeltaWriter::new();
        let mut next_block_index = 0_u32;
        // `(wire block index, is_tool)` for the currently open content block.
        let mut open_block: Option<(u32, bool)> = None;
        let mut tool_blocks = Vec::<u32>::new();
        let mut required_visible = String::new();
        let mut required_calls = Vec::<(String, String, bool)>::new();

        macro_rules! emit_tool_output {
            ($output:expr) => {{
                let output = $output;
                let mut adapter_error = None;
                for event in output.events {
                    use higgs_engine::tool_parser::ToolStreamEvent;
                    match event {
                        ToolStreamEvent::Text(text) => {
                            if text.is_empty() {
                                continue;
                            }
                            if required_call {
                                required_visible.push_str(&text);
                                continue;
                            }
                            if open_block.is_some_and(|(_, is_tool)| is_tool) {
                                adapter_error = Some(
                                    "text arrived before the active tool block ended".to_owned(),
                                );
                                break;
                            }
                            let index = if let Some((index, false)) = open_block {
                                index
                            } else {
                                let index = next_block_index;
                                next_block_index = next_block_index.saturating_add(1);
                                open_block = Some((index, false));
                                let json = serde_json::json!({
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {"type": "text", "text": ""}
                                });
                                yield Ok(Event::default()
                                    .event("content_block_start")
                                    .data(json.to_string()));
                                index
                            };
                            metrics_guard.semantic_event();
                            if index == 0 {
                                let delta = TextDelta {
                                    delta_type: "text_delta",
                                    text,
                                };
                                match text_delta_writer.write(&delta) {
                                    Ok(json) => yield Ok(Event::default()
                                        .event("content_block_delta")
                                        .data(json.to_owned())),
                                    Err(error) => tracing::error!(
                                        error = %error,
                                        "Failed to serialize SSE chunk"
                                    ),
                                }
                            } else {
                                let json = serde_json::json!({
                                    "type": "content_block_delta",
                                    "index": index,
                                    "delta": {"type": "text_delta", "text": text}
                                });
                                yield Ok(Event::default()
                                    .event("content_block_delta")
                                    .data(json.to_string()));
                            }
                        }
                        ToolStreamEvent::ToolStart { index, name } => {
                            if required_call {
                                if index != required_calls.len() || index > 0 {
                                    adapter_error = Some(format!(
                                        "required tool choice produced an unexpected call index {index}"
                                    ));
                                    break;
                                }
                                if required_name
                                    .as_deref()
                                    .is_some_and(|expected| expected != name)
                                {
                                    adapter_error = Some(format!(
                                        "required tool call named '{name}', but tool_choice required '{}'",
                                        required_name.as_deref().unwrap_or_default()
                                    ));
                                    break;
                                }
                                if !required_tools.as_deref().unwrap_or(&[]).iter().any(|tool| {
                                    tool.get("name").and_then(serde_json::Value::as_str)
                                        == Some(name.as_str())
                                }) {
                                    adapter_error = Some(format!(
                                        "required tool call named '{name}' is not a declared tool"
                                    ));
                                    break;
                                }
                            }
                            match open_block {
                                Some((block_index, false)) => {
                                    let json = serde_json::json!({
                                        "type": "content_block_stop",
                                        "index": block_index
                                    });
                                    yield Ok(Event::default()
                                        .event("content_block_stop")
                                        .data(json.to_string()));
                                    open_block = None;
                                }
                                Some((_, true)) => {
                                    adapter_error = Some(
                                        "a tool call started before the active block ended"
                                            .to_owned(),
                                    );
                                    break;
                                }
                                None => {}
                            }
                            if index != tool_blocks.len() {
                                adapter_error = Some(format!(
                                    "tool stream started out-of-order call index {index}"
                                ));
                                break;
                            }
                            let block_index = next_block_index;
                            next_block_index = next_block_index.saturating_add(1);
                            tool_blocks.push(block_index);
                            if required_call {
                                required_calls.push((name.clone(), String::new(), false));
                            }
                            open_block = Some((block_index, true));
                            let json = serde_json::json!({
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": format!("toolu_{}", uuid::Uuid::new_v4().simple()),
                                    "name": name,
                                    "input": {}
                                }
                            });
                            metrics_guard.semantic_event();
                            yield Ok(Event::default()
                                .event("content_block_start")
                                .data(json.to_string()));
                        }
                        ToolStreamEvent::ArgumentsDelta { index, fragment } => {
                            let Some(&block_index) = tool_blocks.get(index) else {
                                adapter_error = Some(format!(
                                    "tool arguments arrived before call index {index}"
                                ));
                                break;
                            };
                            if open_block != Some((block_index, true)) {
                                adapter_error = Some(format!(
                                    "tool arguments arrived after call index {index} ended"
                                ));
                                break;
                            }
                            if required_call {
                                let Some((_, arguments, _)) = required_calls.get_mut(index) else {
                                    adapter_error = Some(format!(
                                        "required tool arguments arrived before call index {index}"
                                    ));
                                    break;
                                };
                                arguments.push_str(&fragment);
                            }
                            if !fragment.is_empty() {
                                let json = serde_json::json!({
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": fragment
                                    }
                                });
                                metrics_guard.semantic_event();
                                yield Ok(Event::default()
                                    .event("content_block_delta")
                                    .data(json.to_string()));
                            }
                        }
                        ToolStreamEvent::ToolEnd { index } => {
                            let Some(&block_index) = tool_blocks.get(index) else {
                                adapter_error = Some(format!(
                                    "tool end arrived before call index {index}"
                                ));
                                break;
                            };
                            if open_block != Some((block_index, true)) {
                                adapter_error = Some(format!(
                                    "tool call index {index} ended out of order"
                                ));
                                break;
                            }
                            if required_call {
                                let Some((_, _, ended)) = required_calls.get_mut(index) else {
                                    adapter_error = Some(format!(
                                        "required tool end arrived before call index {index}"
                                    ));
                                    break;
                                };
                                *ended = true;
                                // Arguments are tentative until the whole
                                // required-call postcondition succeeds. Leave
                                // the block open so a later violation cannot
                                // look like a completed tool use to clients.
                                continue;
                            }
                            let json = serde_json::json!({
                                "type": "content_block_stop",
                                "index": block_index
                            });
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(json.to_string()));
                            open_block = None;
                        }
                    }
                }
                if adapter_error.is_none() {
                    adapter_error = output.error.map(|error| error.to_string());
                }
                if let Some(error) = adapter_error {
                    metrics_guard.fail(error.clone());
                    yield Ok(Event::default()
                        .event("error")
                        .data(anthropic_stream_error_json(&error)));
                    return;
                }
            }};
        }

        while let Some(output) = rx.recv().await {
            if let Some(p) = output.prefill_progress {
                timing.cached_tokens = Some(u64::from(p.cached));
                continue;
            }
            if timing.ttft_ms.is_none() {
                timing.ttft_ms = Some(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX));
            }
            let (visible, _reasoning) = reasoning_tracker.process(&output.new_text);
            total_output_tokens = output.completion_tokens;
            metrics_guard.update(u64::from(total_output_tokens));
            emit_tool_output!(tool_tracker.process(&visible));
            if let Some(reason) = output.finish_reason {
                final_stop_reason = Some(openai_finish_to_anthropic_stop(&reason));
            }
        }

        let terminal = terminal_rx.await.unwrap_or_else(|error| {
            crate::sse::WorkerTerminal::Failed(format!(
                "streaming generation worker terminated unexpectedly: {error}"
            ))
        });
        match terminal {
            crate::sse::WorkerTerminal::Completed => {}
            crate::sse::WorkerTerminal::Failed(error) => {
                metrics_guard.fail(error.clone());
                yield Ok(Event::default()
                    .event("error")
                    .data(anthropic_stream_error_json(&error)));
                return;
            }
            crate::sse::WorkerTerminal::Capacity(info) => {
                metrics_guard.fail("capacity_interrupted".to_owned());
                let error = serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "higgs_capacity_interrupted",
                        "message": "generation interrupted by a capacity change",
                        "code": "capacity_interrupted",
                        "bootId": info.boot_id,
                        "generation": info.generation,
                        "partialOutputTokens": total_output_tokens,
                    }
                });
                yield Ok(Event::default().event("error").data(error.to_string()));
                return;
            }
        }

        let (flush_visible, _flush_reasoning) = reasoning_tracker.flush();
        emit_tool_output!(tool_tracker.process(&flush_visible));
        emit_tool_output!(tool_tracker.finish());

        if required_call {
            if let Err(error) = validate_anthropic_required_call(
                &required_visible,
                &required_calls,
                required_tools.as_deref().unwrap_or(&[]),
                required_name.as_deref(),
            ) {
                metrics_guard.fail(error.clone());
                yield Ok(Event::default()
                    .event("error")
                    .data(anthropic_stream_error_json(&error)));
                return;
            }
        }

        if let Some((index, _)) = open_block.take() {
            let json = serde_json::json!({
                "type": "content_block_stop",
                "index": index
            });
            yield Ok(Event::default()
                .event("content_block_stop")
                .data(json.to_string()));
        }

        if tool_tracker.completed_call_count() > 0 {
            final_stop_reason = Some("tool_use".to_owned());
        }

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

        let msg_stop = MessageStopEvent {
            event_type: "message_stop",
        };
        match serde_json::to_string(&msg_stop) {
            Ok(json) => yield Ok(Event::default().event("message_stop").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }
        metrics_guard.finish();
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
        AnthropicToolChoice, CreateMessageRequest, ServerError, create_message_non_streaming,
        create_message_stream, resolve_anthropic_tool_choice,
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

    async fn anthropic_tool_stream_body(model: &str) -> String {
        use axum::response::IntoResponse as _;
        use http_body_util::BodyExt as _;

        let request = serde_json::from_value::<CreateMessageRequest>(serde_json::json!({
            "model": model,
            "max_tokens": 4096,
            "stream": true,
            "messages": [{"role": "user", "content": "write the content"}],
            "tools": [{
                "name": "write",
                "input_schema": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"]
                }
            }]
        }))
        .unwrap();
        let state = crate::state::test_state_with_stub_engine(model);
        let stream = create_message_stream(
            state,
            request,
            Arc::new(crate::state::Engine::test_stub(model)),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await
        .unwrap();
        let response = axum::response::sse::Sse::new(stream).into_response();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    async fn anthropic_choice_stream_body(
        model: &str,
        choice: serde_json::Value,
    ) -> Result<String, ServerError> {
        use axum::response::IntoResponse as _;
        use http_body_util::BodyExt as _;

        let request = serde_json::from_value::<CreateMessageRequest>(serde_json::json!({
            "model": model,
            "max_tokens": 4096,
            "stream": true,
            "messages": [{"role": "user", "content": "use a tool"}],
            "tool_choice": choice,
            "tools": [
                {
                    "name": "weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                },
                {
                    "name": "write",
                    "input_schema": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"]
                    }
                }
            ]
        }))
        .unwrap();
        let state = crate::state::test_state_with_stub_engine(model);
        let stream = create_message_stream(
            state,
            request,
            Arc::new(crate::state::Engine::test_stub(model)),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await?;
        let response = axum::response::sse::Sse::new(stream).into_response();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        Ok(String::from_utf8(bytes.to_vec()).unwrap())
    }

    fn anthropic_stream_data(body: &str) -> Vec<serde_json::Value> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    #[test]
    fn native_required_choices_reject_missing_or_invalid_tools() {
        let any = AnthropicToolChoice::Any;
        assert!(matches!(
            resolve_anthropic_tool_choice(Some(&any), None),
            Err(ServerError::BadRequest(message)) if message.contains("at least one declared tool")
        ));

        let named = AnthropicToolChoice::Tool {
            name: "missing".to_owned(),
        };
        let tools = [serde_json::json!({
            "name": "write",
            "input_schema": {"type": "object"}
        })];
        assert!(matches!(
            resolve_anthropic_tool_choice(Some(&named), Some(&tools)),
            Err(ServerError::BadRequest(message)) if message.contains("was not declared")
        ));

        let empty = AnthropicToolChoice::Tool {
            name: String::new(),
        };
        assert!(matches!(
            resolve_anthropic_tool_choice(Some(&empty), Some(&tools)),
            Err(ServerError::BadRequest(message)) if message.contains("must not be empty")
        ));
    }

    #[tokio::test]
    async fn tool_stream_uses_native_content_blocks_and_argument_deltas() {
        let body = anthropic_tool_stream_body("required-stream-script-xml").await;
        let events = anthropic_stream_data(&body);
        let event_types: Vec<&str> = events
            .iter()
            .filter_map(|event| event["type"].as_str())
            .collect();

        let text_start = events
            .iter()
            .position(|event| {
                event["type"] == "content_block_start" && event["content_block"]["type"] == "text"
            })
            .unwrap();
        let text_stop = events
            .iter()
            .position(|event| event["type"] == "content_block_stop" && event["index"] == 0)
            .unwrap();
        let tool_start = events
            .iter()
            .position(|event| {
                event["type"] == "content_block_start"
                    && event["content_block"]["type"] == "tool_use"
            })
            .unwrap();
        let tool_stop = events
            .iter()
            .position(|event| event["type"] == "content_block_stop" && event["index"] == 1)
            .unwrap();
        assert!(text_start < text_stop && text_stop < tool_start && tool_start < tool_stop);

        let start = &events[tool_start];
        assert_eq!(start["index"], 1);
        assert_eq!(start["content_block"]["name"], "write");
        assert_eq!(start["content_block"]["input"], serde_json::json!({}));
        assert!(
            start["content_block"]["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("toolu_")),
            "{body}"
        );
        assert_eq!(body.matches("\"id\":\"toolu_").count(), 1, "{body}");

        let argument_fragments: Vec<&str> = events
            .iter()
            .filter(|event| {
                event["type"] == "content_block_delta"
                    && event["index"] == 1
                    && event["delta"]["type"] == "input_json_delta"
            })
            .filter_map(|event| event["delta"]["partial_json"].as_str())
            .collect();
        assert!(
            argument_fragments.len() > 1,
            "expected incremental deltas: {body}"
        );
        let arguments = argument_fragments.concat();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
            serde_json::json!({"content": "hello\nworld"})
        );
        assert!(events.iter().any(|event| {
            event["type"] == "message_delta" && event["delta"]["stop_reason"] == "tool_use"
        }));
        assert_eq!(event_types.last(), Some(&"message_stop"));
    }

    #[tokio::test]
    async fn native_any_choice_streams_tentative_arguments_then_validates() {
        let body = anthropic_choice_stream_body(
            "required-stream-script-required-json",
            serde_json::json!({"type": "any"}),
        )
        .await
        .unwrap();
        let events = anthropic_stream_data(&body);
        assert!(!events.iter().any(|event| {
            event["type"] == "content_block_start" && event["content_block"]["type"] == "text"
        }));
        let tool_start = events
            .iter()
            .position(|event| {
                event["type"] == "content_block_start"
                    && event["content_block"]["type"] == "tool_use"
            })
            .unwrap();
        assert_eq!(events[tool_start]["content_block"]["name"], "weather");
        let fragments: Vec<&str> = events
            .iter()
            .filter(|event| event["delta"]["type"] == "input_json_delta")
            .filter_map(|event| event["delta"]["partial_json"].as_str())
            .collect();
        assert!(
            fragments.len() > 1,
            "tentative arguments must stream: {body}"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&fragments.concat()).unwrap(),
            serde_json::json!({"city": "Rome"})
        );
        assert!(events.iter().any(|event| {
            event["type"] == "message_delta" && event["delta"]["stop_reason"] == "tool_use"
        }));
        assert_eq!(
            events.last().and_then(|event| event["type"].as_str()),
            Some("message_stop")
        );
    }

    #[tokio::test]
    async fn native_named_choice_rejects_wrong_identity_before_exposure() {
        let body = anthropic_choice_stream_body(
            "required-stream-script-required-json",
            serde_json::json!({"type": "tool", "name": "write"}),
        )
        .await
        .unwrap();
        let events = anthropic_stream_data(&body);
        assert!(
            !events
                .iter()
                .any(|event| event["content_block"]["type"] == "tool_use"),
            "{body}"
        );
        let error_index = events
            .iter()
            .position(|event| event["type"] == "error")
            .unwrap();
        assert!(
            events[error_index]["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("tool_choice required 'write'"))
        );
        assert_eq!(
            error_index + 1,
            events.len(),
            "nothing follows terminal failure: {body}"
        );
    }

    #[tokio::test]
    async fn native_required_leaked_text_never_completes_the_tentative_tool_block() {
        let body = anthropic_choice_stream_body(
            "required-stream-script-required-leaky",
            serde_json::json!({"type": "any"}),
        )
        .await
        .unwrap();
        let events = anthropic_stream_data(&body);
        let tool_start = events
            .iter()
            .position(|event| event["content_block"]["type"] == "tool_use")
            .unwrap();
        let tool_index = events[tool_start]["index"].clone();
        assert!(
            events.iter().any(|event| {
                event["type"] == "content_block_delta"
                    && event["index"] == tool_index
                    && event["delta"]["type"] == "input_json_delta"
            }),
            "{body}"
        );
        assert!(
            !events.iter().any(|event| {
                event["type"] == "content_block_stop" && event["index"] == tool_index
            }),
            "{body}"
        );
        assert!(
            !events.iter().any(|event| {
                event["type"] == "message_delta" || event["type"] == "message_stop"
            }),
            "{body}"
        );
        assert_eq!(
            events.last().and_then(|event| event["type"].as_str()),
            Some("error")
        );
    }

    #[tokio::test]
    async fn native_none_choice_disables_tool_parsing() {
        let body = anthropic_choice_stream_body(
            "required-stream-script-json",
            serde_json::json!({"type": "none"}),
        )
        .await
        .unwrap();
        let events = anthropic_stream_data(&body);
        assert!(
            !events
                .iter()
                .any(|event| event["content_block"]["type"] == "tool_use"),
            "{body}"
        );
        let text: String = events
            .iter()
            .filter(|event| event["delta"]["type"] == "text_delta")
            .filter_map(|event| event["delta"]["text"].as_str())
            .collect();
        assert!(text.contains("<tool_call>"), "{body}");
        assert_eq!(
            events.last().and_then(|event| event["type"].as_str()),
            Some("message_stop")
        );
    }

    #[tokio::test]
    async fn nonstream_required_choice_is_rejected_before_generation() {
        let model = "stub-model";
        let request = serde_json::from_value::<CreateMessageRequest>(serde_json::json!({
            "model": model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "use a tool"}],
            "tool_choice": {"type": "any"},
            "tools": [{"name": "write", "input_schema": {"type": "object"}}]
        }))
        .unwrap();
        let result = create_message_non_streaming(
            crate::state::test_state_with_stub_engine(model),
            request,
            Arc::new(crate::state::Engine::test_stub(model)),
        )
        .await;
        assert!(
            matches!(result, Err(ServerError::BadRequest(message)) if message.contains("stream:true"))
        );
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
