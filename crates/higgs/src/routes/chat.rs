use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
use higgs_engine::chat_template::ChatPromptMode;
use higgs_engine::simple::{SessionContinuationPolicy, SessionPromptTracePayloadStats};
use tokio_stream::Stream;

use crate::{
    config::{ApiFormat, GenerationDefaults},
    error::ServerError,
    media::{MediaExtractor, MediaItem},
    metrics::{MetricsStore, RequestMetricsContext, RequestRecord, StreamMetricsGuard},
    router::ResolvedRoute,
    state::{Engine, SharedState},
    types::openai::{
        ChatCompletionChoice, ChatCompletionDelta, ChatCompletionMessage, ChatCompletionRequest,
        ChatCompletionResponse, ChoiceLogprobs, CompletionUsage, ContentPart, MessageContent,
        RetentionReceipt, RetentionRequest, SessionCachePolicy, StopSequence, TokenLogprob,
        ToolCall, ToolCallDelta, ToolCallFunction, ToolCallFunctionDelta, ToolChoice,
        ToolChoiceMode, TopLogprob, merge_repetition_penalty,
    },
};
use higgs_models::SamplingParams;

const TOOL_RESULT_PROMPT_WARN_BYTES: usize = 16 * 1024;

struct ToolChoicePlan<'a> {
    prompt_tools: Option<&'a [serde_json::Value]>,
    constraint_schema: Option<serde_json::Value>,
    /// The pinned function name for a named tool_choice (`None` for the
    /// required-any mode).
    required_name: Option<String>,
}

impl ToolChoicePlan<'_> {
    fn requires_call(&self) -> bool {
        self.constraint_schema.is_some()
    }
}

fn resolve_tool_choice<'a>(
    choice: Option<&ToolChoice>,
    tools: Option<&'a [serde_json::Value]>,
    response_format: Option<&crate::types::openai::ResponseFormat>,
) -> Result<ToolChoicePlan<'a>, ServerError> {
    let tools = tools.filter(|tools| !tools.is_empty());
    let required_name: Option<&str> = match choice {
        None | Some(ToolChoice::Mode(ToolChoiceMode::Auto)) => {
            return Ok(ToolChoicePlan {
                prompt_tools: tools,
                constraint_schema: None,
                required_name: None,
            });
        }
        Some(ToolChoice::Mode(ToolChoiceMode::None)) => {
            return Ok(ToolChoicePlan {
                prompt_tools: None,
                constraint_schema: None,
                required_name: None,
            });
        }
        Some(ToolChoice::Mode(ToolChoiceMode::Required)) => None,
        Some(ToolChoice::Named(named)) => {
            if named.r#type != "function" {
                return Err(ServerError::BadRequest(format!(
                    "Unsupported tool_choice type: {}",
                    named.r#type
                )));
            }
            Some(named.function.name.as_str())
        }
    };

    if response_format.is_some_and(|format| format.r#type != "text") {
        return Err(ServerError::BadRequest(
            "tool_choice required/function cannot be combined with a JSON response_format"
                .to_owned(),
        ));
    }
    let tools = tools.ok_or_else(|| {
        ServerError::BadRequest("tool_choice requires at least one declared tool".to_owned())
    })?;
    let constraint_schema = tool_call_schema(tools, required_name)?;
    Ok(ToolChoicePlan {
        prompt_tools: Some(tools),
        constraint_schema: Some(constraint_schema),
        required_name: required_name.map(str::to_owned),
    })
}

fn tool_call_schema(
    tools: &[serde_json::Value],
    required_name: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let mut variants = Vec::new();
    let mut names = std::collections::HashSet::new();
    for tool in tools {
        let function = tool
            .get("function")
            .and_then(serde_json::Value::as_object)
            .filter(|_| tool.get("type").and_then(serde_json::Value::as_str) == Some("function"))
            .ok_or_else(|| {
                ServerError::BadRequest(
                    "tool_choice requires OpenAI function tool declarations".to_owned(),
                )
            })?;
        let name = function
            .get("name")
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                ServerError::BadRequest("tool function name must not be empty".to_owned())
            })?;
        if !names.insert(name) {
            return Err(ServerError::BadRequest(format!(
                "duplicate tool function name: {name}"
            )));
        }
        if required_name.is_some_and(|required| required != name) {
            continue;
        }
        let parameters = function.get("parameters").cloned().unwrap_or_else(|| {
            serde_json::json!({
                "type": "object"
            })
        });
        if parameters
            .get("type")
            .is_some_and(|kind| kind.as_str() != Some("object"))
        {
            return Err(ServerError::BadRequest(format!(
                "tool function '{name}' parameters must describe an object"
            )));
        }
        variants.push(serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"const": name},
                "arguments": parameters
            },
            "required": ["name", "arguments"],
            "additionalProperties": false
        }));
    }

    if variants.is_empty() {
        return Err(ServerError::BadRequest(format!(
            "tool_choice function '{}' was not declared",
            required_name.unwrap_or_default()
        )));
    }
    if let [variant] = variants.as_slice() {
        return Ok(variant.clone());
    }
    Ok(serde_json::json!({"oneOf": variants}))
}

/// Materialized response parts for the Required/named tool-choice
/// postcondition: exactly one parser-visible tool call, valid against the
/// declared tool schema, with no visible content around it.
///
/// Invariant: everything about the call's *arguments* (required keys, value
/// types, nested shape) is enforced by the exact grammar FSM that
/// `tool_call_schema` built — `from_tagged_json_schema` makes a malformed
/// argument sequence unsampleable — so this helper only rejects what the
/// grammar cannot express: wrong call count, leaked text outside the
/// `<tool_call>` envelope, an undeclared function name, a name that does not
/// match a named tool_choice, or non-object arguments. Shared by the
/// blocking, streaming, and session-routed materialization paths; every
/// deviation fails closed with a typed engine error.
fn required_tool_call_parts(
    parsed: &higgs_engine::tool_parser::ToolParseResult,
    declared_tools: &[serde_json::Value],
    required_name: Option<&str>,
) -> Result<(Option<MessageContent>, Option<Vec<ToolCall>>), ServerError> {
    let fail = |message: String| {
        ServerError::Engine(higgs_engine::error::EngineError::Generation(message))
    };
    let [call] = parsed.tool_calls.as_slice() else {
        return Err(fail(format!(
            "required tool choice produced {} parser-visible tool calls, \
             expected exactly one",
            parsed.tool_calls.len()
        )));
    };
    if !parsed.text.is_empty() {
        return Err(fail(format!(
            "required tool call violated its grammar: {} bytes of visible text \
             outside the tool_call envelope",
            parsed.text.len()
        )));
    }
    if let Some(expected) = required_name {
        if call.name != expected {
            return Err(fail(format!(
                "required tool call named '{}', but tool_choice required '{expected}'",
                call.name
            )));
        }
    }
    if !declared_tools.iter().any(|tool| {
        tool.get("function")
            .and_then(|f| f.get("name"))
            .and_then(serde_json::Value::as_str)
            == Some(call.name.as_str())
    }) {
        return Err(fail(format!(
            "required tool call named '{}' is not a declared tool",
            call.name
        )));
    }
    if !call.arguments.is_object() {
        return Err(fail(format!(
            "tool '{}' arguments must be a JSON object",
            call.name
        )));
    }
    let tool_call = ToolCall {
        id: format!("call_0_{}", uuid::Uuid::new_v4()),
        r#type: "function".to_owned(),
        function: ToolCallFunction {
            name: call.name.clone(),
            arguments: call.arguments.to_string(),
        },
    };
    Ok((None, Some(vec![tool_call])))
}

fn continuation_policy(policy: Option<SessionCachePolicy>) -> SessionContinuationPolicy {
    match policy.unwrap_or(SessionCachePolicy::BestEffort) {
        SessionCachePolicy::BestEffort => SessionContinuationPolicy::BestEffort,
        SessionCachePolicy::RequireContinuation => SessionContinuationPolicy::RequireContinuation,
    }
}

fn apply_required_retention(
    req: &mut ChatCompletionRequest,
) -> Result<Option<RetentionRequest>, ServerError> {
    let Some(retention) = req.retention.clone() else {
        return Ok(None);
    };
    if req.session_id.is_some_and(|id| id != retention.session_id)
        || req.session_cache_policy == Some(SessionCachePolicy::BestEffort)
        || req.drop_session_id == Some(retention.session_id)
        || req
            .drop_session_ids
            .as_deref()
            .is_some_and(|ids| ids.contains(&retention.session_id))
    {
        return Err(ServerError::BadRequest(
            "retention conflicts with legacy session controls".to_owned(),
        ));
    }
    req.session_id = Some(retention.session_id);
    req.session_cache_policy = Some(match retention.mode {
        crate::types::openai::RetentionMode::Required => SessionCachePolicy::RequireContinuation,
        crate::types::openai::RetentionMode::Seed => SessionCachePolicy::BestEffort,
    });
    Ok(Some(retention))
}

struct RetentionSeedClaim {
    state: SharedState,
    engine: Arc<Engine>,
    model: String,
    session_id: u64,
    reservation: Option<crate::state::EngineRetentionClaim>,
    published: bool,
}

impl RetentionSeedClaim {
    fn publish(mut self) -> bool {
        self.published = self.reservation.as_ref().is_some_and(|reservation| {
            self.state.publish_owned_retention_seed(
                &self.model,
                self.session_id,
                reservation.owner_id(),
            )
        });
        self.published
    }
}

impl Drop for RetentionSeedClaim {
    fn drop(&mut self) {
        if !self.published {
            if let Some(reservation) = self.reservation.take() {
                self.state.abort_owned_retention_seed(
                    &self.model,
                    self.session_id,
                    reservation.owner_id(),
                );
                self.engine.release_retained_reservation(reservation);
            }
        }
    }
}

fn validate_required_retention(
    state: &SharedState,
    engine: &Arc<Engine>,
    model: &str,
    prompt_tokens: &[u32],
    output_tokens: u32,
    retention: Option<&RetentionRequest>,
    retired_session_ids: &[u64],
) -> Result<Option<RetentionSeedClaim>, ServerError> {
    let Some(retention) = retention else {
        return Ok(None);
    };
    let error_context = crate::error::RetentionErrorContext {
        contract_revision: retention.contract_revision.clone(),
        session_id: retention.session_id,
        epoch: retention.epoch,
    };
    let contract = state
        .capacity
        .fast_session_contract(model)
        .map_err(|_| ServerError::RetentionCompactionRequired(error_context.clone()))?;
    let admission = match retention.mode {
        crate::types::openai::RetentionMode::Required => contract.admit_required(
            &retention.contract_revision,
            u64::try_from(prompt_tokens.len()).unwrap_or(u64::MAX),
            u64::from(output_tokens),
        ),
        crate::types::openai::RetentionMode::Seed => contract.admit_seed(
            &retention.contract_revision,
            u64::try_from(prompt_tokens.len()).unwrap_or(u64::MAX),
            u64::from(output_tokens),
        ),
    };
    admission.map_err(|error| match error {
        crate::capacity::RequiredRetentionAdmissionError::StaleContract => {
            ServerError::StaleRetentionContract(error_context.clone())
        }
        crate::capacity::RequiredRetentionAdmissionError::CompactionRequired => {
            ServerError::RetentionCompactionRequired(error_context.clone())
        }
    })?;
    match retention.mode {
        crate::types::openai::RetentionMode::Required => {
            if !state.retention_binding_matches(
                model,
                retention.session_id,
                &retention.contract_revision,
                retention.epoch,
            ) || !engine.retained_session_can_continue(retention.session_id, prompt_tokens)
            {
                return Err(ServerError::RequiredRetentionUnavailable(error_context));
            }
            if !engine.lease_retained_session(retention.session_id, 300) {
                return Err(ServerError::RequiredRetentionUnavailable(error_context));
            }
            Ok(None)
        }
        crate::types::openai::RetentionMode::Seed => {
            if engine
                .retained_session_receipt(retention.session_id)
                .is_some()
            {
                return Err(ServerError::Conflict(
                    "retention seed conflicts with an existing session identity".to_owned(),
                ));
            }
            let Some(reservation) = engine
                .reserve_retained_session_replacing(retention.session_id, retired_session_ids)
            else {
                return Err(ServerError::Conflict(
                    "retention seed conflicts with an existing session identity".to_owned(),
                ));
            };
            if !state.claim_exclusively_reserved_seed(
                model,
                retention.session_id,
                &retention.contract_revision,
                retention.epoch,
                &reservation,
            ) {
                engine.release_retained_reservation(reservation);
                return Err(ServerError::Conflict(
                    "retention seed conflicts with an existing session identity".to_owned(),
                ));
            }
            Ok(Some(RetentionSeedClaim {
                state: Arc::clone(state),
                engine: Arc::clone(engine),
                model: model.to_owned(),
                session_id: retention.session_id,
                reservation: Some(reservation),
                published: false,
            }))
        }
    }
}

fn map_session_engine_error(error: higgs_engine::error::EngineError) -> ServerError {
    match error {
        higgs_engine::error::EngineError::RetainedSessionUnavailable(session_id) => {
            ServerError::RetainedSessionUnavailable(session_id)
        }
        // Image preprocessing failures are client problems on the session path
        // too (mirrors `map_engine_error`).
        higgs_engine::error::EngineError::Vision(v) => ServerError::BadRequest(v.to_string()),
        other => ServerError::Engine(other),
    }
}

fn map_required_session_engine_error(
    error: higgs_engine::error::EngineError,
    retention: Option<&RetentionRequest>,
) -> ServerError {
    if matches!(
        error,
        higgs_engine::error::EngineError::RetainedSessionUnavailable(_)
    ) {
        if let Some(retention) = retention {
            return ServerError::RequiredRetentionUnavailable(
                crate::error::RetentionErrorContext {
                    contract_revision: retention.contract_revision.clone(),
                    session_id: retention.session_id,
                    epoch: retention.epoch,
                },
            );
        }
    }
    map_session_engine_error(error)
}

fn streaming_error_json(message: &str) -> String {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
            "code": "generation_error"
        }
    })
    .to_string()
}

#[allow(clippy::too_many_lines)]
pub async fn chat_completions(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    // Captured before parsing and prompt preparation so TTFT measures
    // the client-observed wait, not only generation.
    let received_at = Instant::now();
    let mut req: ChatCompletionRequest = serde_json::from_slice(&body)
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
            generation_defaults,
            routing_method,
        } => {
            req.model = model_name;
            if req.stream == Some(true) {
                let stream = chat_completions_stream(
                    Arc::clone(&state),
                    req,
                    engine,
                    generation_defaults,
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
                let start = Instant::now();
                let response = chat_completions_non_streaming(
                    Arc::clone(&state),
                    req,
                    engine,
                    generation_defaults,
                )
                .await?;
                if let Some(ref metrics) = state.metrics {
                    metrics.record(RequestRecord {
                        id: 0,
                        timestamp: Instant::now(),
                        wallclock: chrono::Utc::now(),
                        model: Some(response.model.clone()),
                        provider: Some("higgs".to_owned()),
                        routing_method: routing_method.into(),
                        status: 200,
                        duration: start.elapsed(),
                        input_tokens: u64::from(response.usage.prompt_tokens),
                        output_tokens: u64::from(response.usage.completion_tokens),
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
            let metrics_model = model_rewrite.as_deref().unwrap_or(&req.model).to_owned();
            let is_streaming = req.stream == Some(true);
            match provider_format {
                ApiFormat::OpenAi => {
                    let proxy_body = if let Some(ref rewrite) = model_rewrite {
                        crate::proxy::rewrite_model_in_body(&body, rewrite)?
                    } else {
                        body
                    };
                    let start = Instant::now();
                    let result = crate::proxy::proxy_request(
                        &state.http_client,
                        &provider_url,
                        "/v1/chat/completions",
                        proxy_body,
                        &headers,
                        strip_auth,
                        api_key.as_deref(),
                    )
                    .await;
                    if let Some(ref metrics) = state.metrics {
                        metrics.record(RequestRecord {
                            id: 0,
                            timestamp: Instant::now(),
                            wallclock: chrono::Utc::now(),
                            model: Some(metrics_model.clone()),
                            provider: Some(provider_name.clone()),
                            routing_method: routing_method.into(),
                            status: result.as_ref().map_or(502, |resp| resp.status().as_u16()),
                            duration: start.elapsed(),
                            input_tokens: 0,
                            output_tokens: 0,
                            error_body: None,
                            timing: crate::metrics::RequestTiming::default(),
                        });
                        request_metrics.mark_recorded();
                    }
                    result
                }
                ApiFormat::Anthropic => {
                    let translated = crate::translate::openai_to_anthropic_request(
                        &body,
                        state.config.server.max_tokens,
                    )?;
                    let proxy_body = if let Some(ref rewrite) = model_rewrite {
                        crate::proxy::rewrite_model_in_body(&translated, rewrite)?
                    } else {
                        translated
                    };

                    let start = Instant::now();
                    let upstream = crate::proxy::send_to_provider(
                        &state.http_client,
                        &provider_url,
                        "/v1/messages",
                        proxy_body,
                        &headers,
                        strip_auth,
                        api_key.as_deref(),
                    )
                    .await?;
                    let upstream_status = upstream.status().as_u16();

                    if is_streaming {
                        if let Some(ref metrics) = state.metrics {
                            metrics.record(RequestRecord {
                                id: 0,
                                timestamp: Instant::now(),
                                wallclock: chrono::Utc::now(),
                                model: Some(metrics_model.clone()),
                                provider: Some(provider_name.clone()),
                                routing_method: routing_method.into(),
                                status: upstream_status,
                                duration: start.elapsed(),
                                input_tokens: 0,
                                output_tokens: 0,
                                error_body: None,
                                timing: crate::metrics::RequestTiming::default(),
                            });
                            request_metrics.mark_recorded();
                        }
                        if upstream_status >= 400 {
                            let status_code = axum::http::StatusCode::from_u16(upstream_status)
                                .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
                            let resp_bytes = upstream.bytes().await.map_err(|e| {
                                ServerError::ProxyError(format!("Failed to read response: {e}"))
                            })?;
                            return Ok((
                                status_code,
                                [(axum::http::header::CONTENT_TYPE, "application/json")],
                                resp_bytes,
                            )
                                .into_response());
                        }
                        let stream =
                            crate::translate::anthropic_stream_to_openai(upstream, req.model);
                        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                        Ok(sse.into_response())
                    } else {
                        let resp_bytes = upstream.bytes().await.map_err(|e| {
                            ServerError::ProxyError(format!("Failed to read response: {e}"))
                        })?;
                        let usage = if (200..300).contains(&upstream_status) {
                            crate::proxy::extract_usage(&resp_bytes)
                        } else {
                            (0, 0)
                        };
                        if let Some(ref metrics) = state.metrics {
                            metrics.record(RequestRecord {
                                id: 0,
                                timestamp: Instant::now(),
                                wallclock: chrono::Utc::now(),
                                model: Some(metrics_model.clone()),
                                provider: Some(provider_name.clone()),
                                routing_method: routing_method.into(),
                                status: upstream_status,
                                duration: start.elapsed(),
                                input_tokens: usage.0,
                                output_tokens: usage.1,
                                error_body: None,
                                timing: crate::metrics::RequestTiming::default(),
                            });
                            request_metrics.mark_recorded();
                        }
                        let status_code = axum::http::StatusCode::from_u16(upstream_status)
                            .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
                        if upstream_status >= 400 {
                            Ok((
                                status_code,
                                [(axum::http::header::CONTENT_TYPE, "application/json")],
                                resp_bytes,
                            )
                                .into_response())
                        } else {
                            let translated_resp = crate::translate::anthropic_response_to_openai(
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
                }
            }
        }
    }
}

#[allow(clippy::too_many_lines)]
async fn chat_completions_non_streaming(
    state: SharedState,
    mut req: ChatCompletionRequest,
    engine: Arc<Engine>,
    generation_defaults: GenerationDefaults,
) -> Result<ChatCompletionResponse, ServerError> {
    let required_retention = apply_required_retention(&mut req)?;
    let max_tokens =
        resolved_max_tokens(&req, &generation_defaults, state.config.server.max_tokens);
    let prompt_mode = chat_prompt_mode(req.session_id, max_tokens);
    let continuation_policy = continuation_policy(req.session_cache_policy);
    let sampling = build_sampling_params(&req, &generation_defaults)?;
    let stop_sequences = StopSequence::extract(req.stop);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs;

    // Extract media and gate on vision capability: a strict 400 when images
    // are sent to a model that cannot see them.
    let media_extractor = MediaExtractor::new(
        state.config.server.max_image_bytes,
        state.config.server.image_fetch_timeout,
        state.config.server.max_image_dimension,
    )?;
    let media = media_extractor.extract_openai(&req.messages).await?;
    check_vision_capability(&media, engine.is_vlm(), engine.model_name())?;

    // Build effective messages: text parts with the family marker spliced at
    // each image part's true position. Text-only requests pass through
    // unchanged. The marker tokens are expanded into sentinel runs below.
    let effective_messages = if media.is_empty() {
        req.messages.clone()
    } else {
        render_markers(&req.messages, engine.image_marker_text())
    };

    let messages = convert_messages(&effective_messages);
    // Treat an empty `tools: []` as absent (mirrors the streaming path) so it
    // doesn't define `tools` in the template context or trigger tool parsing.
    let tool_choice = resolve_tool_choice(
        req.tool_choice.as_ref(),
        req.tools.as_deref(),
        req.response_format.as_ref(),
    )?;
    let tools = tool_choice.prompt_tools;
    let required_call = tool_choice.requires_call();
    let required_name = tool_choice.required_name.clone();
    let thinking_enabled = crate::reasoning::effective_thinking_enabled(
        engine.enable_thinking(),
        &[engine.model_name(), req.model.as_str()],
        req.reasoning.as_ref(),
        req.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.enable_thinking)
            .or(req.enable_thinking)
            .or(generation_defaults.enable_thinking),
        // A required-call grammar starts with visible `<tool_call>` text. Leaving
        // the template inside `<think>` would classify the entire call as reasoning.
    ) && !tool_choice.requires_call();

    let (prompt_tokens, pflash_policy) = engine
        .prepare_chat_prompt_with_pflash_policy(&messages, tools, thinking_enabled, prompt_mode)
        .map_err(ServerError::Engine)?;
    let max_tokens = crate::capacity::resolve_output_tokens(
        &state,
        &req.model,
        prompt_tokens.len(),
        req.max_tokens,
        max_tokens,
    );
    validate_prompt_limit(req.max_prompt_tokens, prompt_tokens.len())?;
    validate_session_lease_ttl(req.session_lease.map(|lease| lease.ttl_seconds))?;
    let dropped_binding_ids =
        retained_session_drop_ids(req.drop_session_id, req.drop_session_ids.as_deref());
    let seed_claim = validate_required_retention(
        &state,
        &engine,
        &req.model,
        &prompt_tokens,
        max_tokens,
        required_retention.as_ref(),
        &dropped_binding_ids,
    )?;
    // Multimodal requests: hand the raw decoded images to the engine, which
    // preprocesses them into a family-native `ImageBatch` (SimpleEngine under
    // its model lock; BatchEngine inside its worker thread) and expands each
    // family marker token into its sentinel run. Image preprocessing
    // failures are client problems (bad/malformed image data): they surface
    // as `EngineError::Vision` — Simple preprocesses synchronously in the
    // engine, and the batch worker marks worker-side preprocessing failures
    // so its generate tails reconstruct the same error — and `map_engine_error`
    // below maps them to strict 400s.
    let image_inputs = (!media.is_empty() && engine.is_vlm())
        .then(|| media.into_iter().map(MediaItem::into).collect());

    let constraint = build_constraint(
        req.response_format.as_ref(),
        tool_choice.constraint_schema.as_ref(),
        &engine,
    )?;

    // Opt-in multi-turn KV-cache reuse. Only honored for request shapes the
    // continued path can preserve; unsupported features fall back to normal
    // generation, where radix/PFlash stay available.
    //
    // BEST-EFFORT, not exact replay: the retained KV is TurboQuant-compressed
    // (lossy) and the prompt is reconciled in text space below, so a continued
    // turn may differ slightly from a stateless full prefill. Clients needing
    // bit-identical output should omit `session_id` — the radix prefix cache on
    // the normal path is exact. See `SimpleEngine::generate_continued`.
    let tokenizer = engine.tokenizer().clone();
    let checkpoint_id = req.checkpoint_id.clone();
    let session_id = session_continuation_id(
        req.session_id,
        image_inputs.is_some(),
        constraint.is_some(),
        checkpoint_id.as_deref(),
        want_logprobs,
        !stop_sequences.is_empty(),
    );
    if required_retention.as_ref().is_some_and(|retention| {
        retention.mode == crate::types::openai::RetentionMode::Seed
            && session_id != Some(retention.session_id)
    }) {
        return Err(ServerError::BadRequest(
            "retention seed is incompatible with request features that disable exact sessions"
                .to_owned(),
        ));
    }
    if continuation_policy == SessionContinuationPolicy::RequireContinuation && session_id.is_none()
    {
        engine.record_required_continuation_miss();
        return Err(ServerError::RetainedSessionUnavailable(
            req.session_id.unwrap_or_default(),
        ));
    }
    // Retained prefix facts are revalidated by the worker at acceptance.
    // They never discount the full prompt in the fixed context check.
    let retained_prefix = session_id
        .and_then(|sid| engine.retained_session_prefix_len(sid, &prompt_tokens))
        .unwrap_or(0);
    let reservation = crate::capacity::admit_generation_request(
        &state,
        &req.model,
        prompt_tokens.len(),
        max_tokens,
    )
    .await?;
    let assumed_retained_prefix = u64::try_from(retained_prefix).unwrap_or(u64::MAX);
    let watchdog = crate::capacity::request_watchdog(&state);
    state.drop_retention_bindings(&req.model, &dropped_binding_ids);
    drop_requested_retained_sessions(
        Arc::clone(&engine),
        req.drop_session_id,
        req.drop_session_ids.as_deref(),
    )
    .await?;
    let lease_active = req
        .session_lease
        .is_some_and(|lease| engine.lease_retained_session(lease.session_id, lease.ttl_seconds));

    let request_id = generate_request_id();
    let allow_prefix_cache = req.cache_mode.as_deref() != Some("bypass");
    let has_tools = tools.is_some();
    let tool_payload = tool_payload_stats(&effective_messages);
    warn_large_tool_payload(tool_payload);

    let output = if let Some(sid) = session_id {
        let engine_c = Arc::clone(&engine);
        let prompt_tokens_c = prompt_tokens.clone();
        let messages_c = messages.clone();
        let tools_c = tools.map(<[serde_json::Value]>::to_vec);
        let sampling_c = sampling.clone();
        let pflash_policy_c = pflash_policy.clone();
        let session_output = tokio::task::spawn_blocking(move || {
            crate::capacity::run_reserved_generation(reservation, watchdog, || {
                engine_c.generate_session_routed_with_thinking(
                    sid,
                    &prompt_tokens_c,
                    &messages_c,
                    tools_c.as_deref(),
                    max_tokens,
                    &sampling_c,
                    thinking_enabled,
                    tool_payload,
                    &pflash_policy_c,
                    continuation_policy,
                    assumed_retained_prefix,
                )
            })
        })
        .await
        .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
        .map_err(|error| map_required_session_engine_error(error, required_retention.as_ref()))?;

        let retention_receipt = required_retention.as_ref().and_then(|retention| {
            engine
                .retained_session_receipt(sid)
                .map(|(tokens, bytes)| RetentionReceipt {
                    outcome: match retention.mode {
                        crate::types::openai::RetentionMode::Seed => "seeded",
                        crate::types::openai::RetentionMode::Required => "continued",
                    },
                    session_id: sid,
                    epoch: retention.epoch,
                    retained_tokens: u64::try_from(tokens).unwrap_or(u64::MAX),
                    retained_bytes: u64::try_from(bytes).unwrap_or(u64::MAX),
                    contract_revision: retention.contract_revision.clone(),
                })
        });
        if retention_receipt.is_none() {
            if let Some(retention) = required_retention.as_ref() {
                return Err(ServerError::RequiredRetentionUnavailable(
                    crate::error::RetentionErrorContext {
                        contract_revision: retention.contract_revision.clone(),
                        session_id: retention.session_id,
                        epoch: retention.epoch,
                    },
                ));
            }
        }
        if let Some(seed_claim) = seed_claim {
            if !engine.lease_retained_session(sid, 300) {
                return Err(ServerError::RequiredRetentionUnavailable(
                    crate::error::RetentionErrorContext {
                        contract_revision: required_retention
                            .as_ref()
                            .map_or_else(String::new, |value| value.contract_revision.clone()),
                        session_id: sid,
                        epoch: required_retention.as_ref().map_or(0, |value| value.epoch),
                    },
                ));
            }
            if !seed_claim.publish() {
                return Err(ServerError::Conflict(
                    "retention seed publication lost its identity reservation".to_owned(),
                ));
            }
        }
        return build_session_response(
            &req.model,
            &request_id,
            session_output,
            tools,
            has_tools,
            thinking_enabled,
            lease_active,
            required_call,
            required_name.as_deref(),
            retention_receipt,
        );
    } else {
        tokio::task::spawn_blocking(move || {
            crate::capacity::run_reserved_generation(reservation, watchdog, || {
                engine.generate_with_thinking_and_pflash_policy_with_cache(
                    &prompt_tokens,
                    max_tokens,
                    &sampling,
                    &stop_sequences,
                    want_logprobs,
                    top_logprobs,
                    thinking_enabled,
                    constraint,
                    image_inputs,
                    checkpoint_id.as_deref(),
                    &pflash_policy,
                    allow_prefix_cache,
                )
            })
        })
        .await
        .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
        .map_err(map_engine_error)?
    };

    let logprobs_response = output
        .token_logprobs
        .as_ref()
        .map(|lps| logprobs_to_response(lps, &tokenizer));

    let output_text = output.text;
    // Parse reasoning (think tags) from the output.
    // When thinking mode is enabled, prefer the token-level split the engine
    // already performed (`output.reasoning_content` / `output.text`), which is
    // exact and never surfaces the `</think>` delimiter. Fall back to the
    // string parser only when the engine did not split (e.g. a model that
    // self-emits `<think>` tags, or thinking disabled at the engine layer).
    let (raw_text, reasoning_content) = if thinking_enabled {
        match output.reasoning_content {
            Some(r) => (output_text, Some(r)),
            None => {
                let parse_input = if output_text.contains("</think>") {
                    format!("<think>{output_text}")
                } else {
                    // Model was length-stopped mid-thinking — close the tag so the
                    // parser can extract reasoning instead of leaking raw `<think>`.
                    format!("<think>{output_text}</think>")
                };
                let reasoning_result =
                    higgs_engine::reasoning_parser::parse_reasoning(&parse_input);
                let raw_text = if reasoning_result.reasoning.is_some() {
                    reasoning_result.text
                } else {
                    output_text.clone()
                };
                (raw_text, reasoning_result.reasoning)
            }
        }
    } else {
        // Model-emitted reasoning (e.g. VibeThinker writes its own
        // `<think>...</think>`): parse it out without the prompt-injection
        // prepend used for template-opened thinking. No-op when absent, so it
        // matches the streaming path's `new()` tracker for every model.
        let reasoning_result = higgs_engine::reasoning_parser::parse_reasoning(&output_text);
        if reasoning_result.reasoning.is_some() {
            (reasoning_result.text, reasoning_result.reasoning)
        } else {
            (output_text, None)
        }
    };

    let (content, tool_calls, finish_reason) = if has_tools {
        let schema = higgs_engine::tool_parser::ToolSchema::from_tools(tools);
        let parsed = higgs_engine::tool_parser::parse_tool_calls(&raw_text, schema.as_ref());
        if required_call {
            let (content, calls) =
                required_tool_call_parts(&parsed, tools.unwrap_or(&[]), required_name.as_deref())?;
            (content, calls, "tool_calls".to_owned())
        } else if parsed.tool_calls.is_empty() {
            (
                Some(MessageContent::Text(raw_text)),
                None,
                output.finish_reason,
            )
        } else {
            let calls: Vec<ToolCall> = parsed
                .tool_calls
                .iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    id: format!("call_{i}_{}", uuid::Uuid::new_v4()),
                    r#type: "function".to_owned(),
                    function: ToolCallFunction {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    },
                })
                .collect();
            let text = if parsed.text.is_empty() {
                None
            } else {
                Some(MessageContent::Text(parsed.text))
            };
            (text, Some(calls), "tool_calls".to_owned())
        }
    } else {
        (
            Some(MessageContent::Text(raw_text)),
            None,
            output.finish_reason,
        )
    };

    Ok(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created: current_unix_timestamp(),
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_owned(),
                content,
                reasoning_content,
                tool_calls,
                tool_call_id: None,
            },
            finish_reason,
            logprobs: logprobs_response,
        }],
        // Stateless (no session_id) path: reuse is via the radix prefix cache,
        // surfaced through `GenerationOutput::cached_prompt_tokens` (mirrors
        // the streaming route's `PrefillProgress.cached`).
        usage: CompletionUsage::new(
            output.prompt_tokens,
            output.completion_tokens,
            output.cached_prompt_tokens,
        )
        .with_session_lease_active(lease_active),
    })
}

fn tool_payload_stats(messages: &[ChatCompletionMessage]) -> SessionPromptTracePayloadStats {
    let mut stats = SessionPromptTracePayloadStats::default();
    for message in messages {
        if !message.role.eq_ignore_ascii_case("tool") {
            continue;
        }
        let bytes = message
            .content
            .as_ref()
            .map_or(0, |content| content.text().len());
        stats.messages += 1;
        stats.bytes = stats.bytes.saturating_add(bytes);
        stats.largest_bytes = stats.largest_bytes.max(bytes);
    }
    stats
}

fn warn_large_tool_payload(stats: SessionPromptTracePayloadStats) {
    if stats.bytes >= TOOL_RESULT_PROMPT_WARN_BYTES {
        tracing::warn!(
            tool_result_messages = stats.messages,
            tool_result_bytes = stats.bytes,
            tool_result_largest_bytes = stats.largest_bytes,
            warn_threshold_bytes = TOOL_RESULT_PROMPT_WARN_BYTES,
            "large raw tool-result replay present in live prompt; compact to handles and recall exact output on demand"
        );
    }
}

/// Map a [`SessionGeneration`] (cache-resident continued turn) onto the same
/// `ChatCompletionResponse` shape as the normal path, preserving reasoning
/// extraction, tool-call parsing, and the `finish_reason: "tool_calls"`
/// override. The continued path uses greedy decode without logprobs, so
/// logprobs are absent; its actual engine finish reason is preserved.
///
/// Required/named tool-choice requests apply the same postcondition as the
/// blocking path and fail closed on any violation. Constraints and sessions
/// are mutually exclusive today (`session_continuation_id`), so this only
/// fires if that ever changes — enforcement here keeps the invariant local to
/// every materialization path.
#[allow(clippy::too_many_arguments)]
fn build_session_response(
    model: &str,
    request_id: &str,
    output: higgs_engine::simple::SessionGeneration,
    tools: Option<&[serde_json::Value]>,
    has_tools: bool,
    thinking_enabled: bool,
    lease_active: bool,
    required_call: bool,
    required_name: Option<&str>,
    retention_receipt: Option<RetentionReceipt>,
) -> Result<ChatCompletionResponse, ServerError> {
    let usage = session_usage(&output)
        .with_session_lease_active(lease_active)
        .with_retention_receipt(retention_receipt);
    let generation_finish_reason = output.finish_reason.clone();
    let output_text = output.text;
    // Same reasoning-tag handling as the normal path: the template opens
    // `<think>` in the prompt, so generated text starts inside the think block.
    let (raw_text, reasoning_content) = if thinking_enabled {
        let parse_input = if output_text.contains("</think>") {
            format!("<think>{output_text}")
        } else {
            format!("<think>{output_text}</think>")
        };
        let reasoning_result = higgs_engine::reasoning_parser::parse_reasoning(&parse_input);
        let raw_text = if reasoning_result.reasoning.is_some() {
            reasoning_result.text
        } else {
            output_text
        };
        (raw_text, reasoning_result.reasoning)
    } else {
        (output_text, None)
    };

    let (content, tool_calls, finish_reason) = if has_tools {
        let schema = higgs_engine::tool_parser::ToolSchema::from_tools(tools);
        let parsed = higgs_engine::tool_parser::parse_tool_calls(&raw_text, schema.as_ref());
        if required_call {
            let (content, calls) =
                required_tool_call_parts(&parsed, tools.unwrap_or(&[]), required_name)?;
            (content, calls, "tool_calls".to_owned())
        } else if parsed.tool_calls.is_empty() {
            (
                Some(MessageContent::Text(raw_text)),
                None,
                generation_finish_reason.clone(),
            )
        } else {
            let calls: Vec<ToolCall> = parsed
                .tool_calls
                .iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    id: format!("call_{i}_{}", uuid::Uuid::new_v4()),
                    r#type: "function".to_owned(),
                    function: ToolCallFunction {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    },
                })
                .collect();
            let text = if parsed.text.is_empty() {
                None
            } else {
                Some(MessageContent::Text(parsed.text))
            };
            (text, Some(calls), "tool_calls".to_owned())
        }
    } else {
        (
            Some(MessageContent::Text(raw_text)),
            None,
            generation_finish_reason,
        )
    };

    Ok(ChatCompletionResponse {
        id: request_id.to_owned(),
        object: "chat.completion",
        created: current_unix_timestamp(),
        model: model.to_owned(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_owned(),
                content,
                reasoning_content,
                tool_calls,
                tool_call_id: None,
            },
            finish_reason,
            logprobs: None,
        }],
        usage,
    })
}

/// Usage for a session-continuation turn. `cached_tokens` = the prompt tokens
/// served from the retained KV cache (everything not re-prefilled this turn).
/// Only a truly continued turn reused a prefix; a cold prefill reports 0.
fn session_usage(output: &higgs_engine::simple::SessionGeneration) -> CompletionUsage {
    let cached = if output.continued {
        let forwarded_prompt_tokens =
            if output.completion_tokens == 0 && output.finish_reason == "length" {
                output.prompt_tokens.saturating_sub(1)
            } else {
                output.prompt_tokens
            };
        forwarded_prompt_tokens.saturating_sub(output.prefilled_tokens)
    } else {
        0
    };
    CompletionUsage::new(output.prompt_tokens, output.completion_tokens, cached)
}

#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
async fn chat_completions_stream(
    state: SharedState,
    mut req: ChatCompletionRequest,
    engine: Arc<Engine>,
    generation_defaults: GenerationDefaults,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
) -> Result<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>, ServerError> {
    let required_retention = apply_required_retention(&mut req)?;
    let tool_choice = resolve_tool_choice(
        req.tool_choice.as_ref(),
        req.tools.as_deref(),
        req.response_format.as_ref(),
    )?;
    let prompt_tools = tool_choice.prompt_tools;
    let required_call = tool_choice.requires_call();
    let required_name = tool_choice.required_name.clone();
    // Owned copy so the 'static stream block can judge the required-call
    // postcondition at end-of-stream without borrowing the request.
    let required_tools = tool_choice
        .requires_call()
        .then(|| tool_choice.prompt_tools.unwrap_or(&[]).to_vec());
    let stream_includes_tools = prompt_tools.is_some();
    // Built here (before the `async_stream::stream!` block, which captures by
    // move) so the tracker can coerce XML-format tool-call values to their
    // declared JSON types.
    let tool_schema = higgs_engine::tool_parser::ToolSchema::from_tools(prompt_tools);

    if stream_includes_tools {
        tracing::debug!(
            request_model = req.model,
            tool_count = prompt_tools.map_or(0, <[serde_json::Value]>::len),
            "Streaming with tool-calls enabled; will emit incremental tool_calls deltas",
        );
    }

    let max_tokens =
        resolved_max_tokens(&req, &generation_defaults, state.config.server.max_tokens);
    let prompt_mode = chat_prompt_mode(req.session_id, max_tokens);
    let continuation_policy = continuation_policy(req.session_cache_policy);
    let sampling = build_sampling_params(&req, &generation_defaults)?;
    let stop_sequences = StopSequence::extract(req.stop);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs;

    // Extract media and gate on vision capability: a strict 400 when images
    // are sent to a model that cannot see them.
    let media_extractor = MediaExtractor::new(
        state.config.server.max_image_bytes,
        state.config.server.image_fetch_timeout,
        state.config.server.max_image_dimension,
    )?;
    let media = media_extractor.extract_openai(&req.messages).await?;
    check_vision_capability(&media, engine.is_vlm(), engine.model_name())?;

    // Build effective messages: text parts with the family marker spliced at
    // each image part's true position. Text-only requests pass through
    // unchanged. The marker tokens are expanded into sentinel runs below.
    let effective_messages = if media.is_empty() {
        req.messages.clone()
    } else {
        render_markers(&req.messages, engine.image_marker_text())
    };

    let messages = convert_messages(&effective_messages);
    let thinking_enabled_stream = crate::reasoning::effective_thinking_enabled(
        engine.enable_thinking(),
        &[engine.model_name(), req.model.as_str()],
        req.reasoning.as_ref(),
        req.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.enable_thinking)
            .or(req.enable_thinking)
            .or(generation_defaults.enable_thinking),
        // Keep the forced envelope in visible output for the streaming parser too.
    ) && !tool_choice.requires_call();

    // Pass tools into prompt rendering so the chat template emits the
    // tool spec the model recognises. The on-the-fly
    // [`IncrementalToolCallTracker`] below intercepts `<tool_call>…
    // </tool_call>` blocks the model produces and turns them into
    // structured `ToolCallDelta` SSE events.
    let (prompt_tokens, pflash_policy) = engine
        .prepare_chat_prompt_with_pflash_policy(
            &messages,
            prompt_tools,
            thinking_enabled_stream,
            prompt_mode,
        )
        .map_err(ServerError::Engine)?;
    let max_tokens = crate::capacity::resolve_output_tokens(
        &state,
        &req.model,
        prompt_tokens.len(),
        req.max_tokens,
        max_tokens,
    );
    validate_prompt_limit(req.max_prompt_tokens, prompt_tokens.len())?;
    validate_session_lease_ttl(req.session_lease.map(|lease| lease.ttl_seconds))?;
    let dropped_binding_ids =
        retained_session_drop_ids(req.drop_session_id, req.drop_session_ids.as_deref());
    let seed_claim = validate_required_retention(
        &state,
        &engine,
        &req.model,
        &prompt_tokens,
        max_tokens,
        required_retention.as_ref(),
        &dropped_binding_ids,
    )?;
    // Multimodal requests: hand the raw decoded images to the engine, which
    // preprocesses them into a family-native `ImageBatch` (SimpleEngine under
    // its model lock; BatchEngine inside its worker thread) and expands each
    // family marker token into its sentinel run. Image preprocessing
    // failures are client problems (bad/malformed image data): the
    // non-streaming path maps `EngineError::Vision` to strict 400s, and the
    // streaming path surfaces any engine failure as an error-finish chunk
    // instead of a silently truncated stream.
    let image_inputs = (!media.is_empty() && engine.is_vlm())
        .then(|| media.into_iter().map(MediaItem::into).collect());

    let constraint = build_constraint(
        req.response_format.as_ref(),
        tool_choice.constraint_schema.as_ref(),
        &engine,
    )?;

    let request_id = generate_request_id();
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|opts| opts.include_usage.unwrap_or(false));
    let return_progress = req.return_progress.unwrap_or(false);
    let collect_prefill_progress = return_progress || include_usage;
    let created = current_unix_timestamp();
    let request_session_id = req.session_id;
    let allow_prefix_cache = req.cache_mode.as_deref() != Some("bypass");
    let model = req.model;
    let checkpoint_id = req.checkpoint_id;
    let prompt_token_count = u32::try_from(prompt_tokens.len()).unwrap_or(0);
    let tool_payload = tool_payload_stats(&effective_messages);
    warn_large_tool_payload(tool_payload);

    let start = Instant::now();
    let stream_session_id = session_continuation_id(
        request_session_id,
        image_inputs.is_some(),
        constraint.is_some(),
        checkpoint_id.as_deref(),
        want_logprobs,
        !stop_sequences.is_empty(),
    );
    if required_retention.as_ref().is_some_and(|retention| {
        retention.mode == crate::types::openai::RetentionMode::Seed
            && stream_session_id != Some(retention.session_id)
    }) {
        return Err(ServerError::BadRequest(
            "retention seed is incompatible with request features that disable exact sessions"
                .to_owned(),
        ));
    }
    if continuation_policy == SessionContinuationPolicy::RequireContinuation
        && stream_session_id.is_none()
    {
        engine.record_required_continuation_miss();
        return Err(ServerError::RetainedSessionUnavailable(
            request_session_id.unwrap_or_default(),
        ));
    }
    // Retained prefix facts are revalidated by the worker at acceptance.
    // They never discount the full prompt in the fixed context check.
    let retained_prefix = stream_session_id
        .and_then(|sid| engine.retained_session_prefix_len(sid, &prompt_tokens))
        .unwrap_or(0);
    let reservation =
        crate::capacity::admit_generation_request(&state, &model, prompt_tokens.len(), max_tokens)
            .await?;
    let assumed_retained_prefix = u64::try_from(retained_prefix).unwrap_or(u64::MAX);
    let watchdog = crate::capacity::request_watchdog(&state);
    state.drop_retention_bindings(&model, &dropped_binding_ids);
    drop_requested_retained_sessions(
        Arc::clone(&engine),
        req.drop_session_id,
        req.drop_session_ids.as_deref(),
    )
    .await?;
    let lease_active = req
        .session_lease
        .is_some_and(|lease| engine.lease_retained_session(lease.session_id, lease.ttl_seconds));

    let (tx, mut rx) = tokio::sync::mpsc::channel(32);
    let (terminal_tx, terminal_rx) = tokio::sync::oneshot::channel::<crate::sse::WorkerTerminal>();
    let (acceptance, acceptance_rx) =
        if continuation_policy == SessionContinuationPolicy::RequireContinuation {
            let (tx, rx) = tokio::sync::oneshot::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

    // Cache-resident (session-continued) turns stream from the retained KV
    // cache; everything else does a fresh prefill. Both feed the same
    // `StreamingOutput` channel and the same delta/tool-call-tracking loop
    // below — the session path used to buffer the *entire* completion behind
    // a `spawn_blocking().await` before emitting a single burst of deltas
    // (the browser/client would see time-to-first-delta == total elapsed
    // time on every cache-resident turn). Streaming the retained-cache decode
    // loop itself (see `generate_continued_streaming_with_thinking`) fixes
    // that without changing the non-session path at all.
    if let Some(sid) = stream_session_id {
        let worker_engine = Arc::clone(&engine);
        let prompt_tools_c = prompt_tools.map(<[serde_json::Value]>::to_vec);
        let messages_c = messages.clone();
        let pflash_policy_c = pflash_policy.clone();
        tokio::task::spawn_blocking(move || {
            let mut result =
                crate::capacity::run_reserved_generation(reservation, watchdog, || {
                    worker_engine.generate_session_routed_streaming_with_thinking(
                        sid,
                        &prompt_tokens,
                        &messages_c,
                        prompt_tools_c.as_deref(),
                        max_tokens,
                        &sampling,
                        &tx,
                        thinking_enabled_stream,
                        tool_payload,
                        &pflash_policy_c,
                        continuation_policy,
                        acceptance,
                        assumed_retained_prefix,
                    )
                });
            if result.is_ok() {
                if let Some(seed_claim) = seed_claim {
                    if worker_engine.retained_session_receipt(sid).is_some() {
                        if !worker_engine.lease_retained_session(sid, 300) || !seed_claim.publish()
                        {
                            result = Err(higgs_engine::error::EngineError::Generation(
                                "retention seed publication lost its identity reservation"
                                    .to_owned(),
                            ));
                        }
                    } else {
                        result = Err(higgs_engine::error::EngineError::Generation(
                            "retention seed completed without exact publication".to_owned(),
                        ));
                    }
                }
            }
            match &result {
                Ok(()) => {}
                Err(higgs_engine::error::EngineError::Cancelled) => {
                    tracing::debug!(sid, "Session-routed streaming cancelled by client");
                }
                Err(e) => {
                    tracing::error!(error = %e, "Session-routed generation error during streaming");
                }
            }
            let _ = terminal_tx.send(crate::sse::WorkerTerminal::from_engine_result(result));
        });
    } else {
        let worker_engine = Arc::clone(&engine);
        tokio::task::spawn_blocking(move || {
            let result = crate::capacity::run_reserved_generation(reservation, watchdog, || {
                worker_engine.generate_streaming_with_thinking_and_pflash_policy_with_cache(
                    &prompt_tokens,
                    max_tokens,
                    &sampling,
                    &stop_sequences,
                    want_logprobs,
                    top_logprobs,
                    &tx,
                    thinking_enabled_stream,
                    collect_prefill_progress,
                    constraint,
                    image_inputs,
                    checkpoint_id.as_deref(),
                    &pflash_policy,
                    allow_prefix_cache,
                )
            });
            if let Err(ref e) = result {
                tracing::error!(error = %e, "Generation error during streaming");
            }
            let _ = terminal_tx.send(crate::sse::WorkerTerminal::from_engine_result(result));
        });
    }

    if let Some(acceptance_rx) = acceptance_rx {
        match acceptance_rx.await {
            Ok(Ok(())) => {}
            Ok(Err(session_id)) => {
                if let Some(retention) = required_retention.as_ref() {
                    return Err(ServerError::RequiredRetentionUnavailable(
                        crate::error::RetentionErrorContext {
                            contract_revision: retention.contract_revision.clone(),
                            session_id: retention.session_id,
                            epoch: retention.epoch,
                        },
                    ));
                }
                return Err(ServerError::RetainedSessionUnavailable(session_id));
            }
            Err(_) => {
                return Err(ServerError::InternalError(
                    "session continuation worker exited before acceptance".to_owned(),
                ));
            }
        }
    }

    let tokenizer = engine.tokenizer().clone();
    let metrics_id = metrics.as_ref().map(|m| {
        m.record_pending(RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: Some(model.clone()),
            provider: Some("higgs".to_owned()),
            routing_method: routing_method.into(),
            status: 200,
            duration: Duration::ZERO,
            input_tokens: u64::from(prompt_token_count),
            output_tokens: 0,
            error_body: None,
            timing: crate::metrics::RequestTiming::default(),
        })
    });

    let mut metrics_guard = StreamMetricsGuard::new(metrics.clone(), metrics_id, start);
    let stream = async_stream::stream! {
        let mut writer = crate::sse::ChatChunkWriter::new(&request_id, created, &model);
        let route_timing = std::env::var("HIGGS_DIAG_SESSION_TIMING").is_ok_and(|value| value == "1");
        let route_timer = route_timing.then(Instant::now);
        let mut tool_parser_elapsed = Duration::ZERO;
        let mut tool_parser_calls = 0_u64;
        let mut delta_serialize_elapsed = Duration::ZERO;
        let mut delta_yield_elapsed = Duration::ZERO;
        let mut delta_yields = 0_u64;

        // Helper to emit a chunk carrying a delta.
        macro_rules! emit_delta {
            ($delta:expr, $finish:expr, $logprobs:expr) => {{
                let serialize_timer = route_timing.then(Instant::now);
                let serialized = writer.write_delta($delta, $finish, $logprobs);
                if let Some(serialize_started_at) = serialize_timer {
                    delta_serialize_elapsed += serialize_started_at.elapsed();
                }
                match serialized {
                    Ok(json) => {
                        let yield_timer = route_timing.then(Instant::now);
                        yield Ok(Event::default().data(json));
                        if let Some(yield_started_at) = yield_timer {
                            delta_yield_elapsed += yield_started_at.elapsed();
                            delta_yields = delta_yields.saturating_add(1);
                        }
                    }
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            }};
        }

        // Send initial role chunk
        let role_delta = ChatCompletionDelta {
            role: Some("assistant".to_owned()),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        };
        emit_delta!(&role_delta, None, None);

        let mut reasoning_tracker = if thinking_enabled_stream {
            higgs_engine::reasoning_parser::StreamingReasoningTracker::new_inside_think()
        } else {
            higgs_engine::reasoning_parser::StreamingReasoningTracker::new()
        };
        // Streaming tool-call extractor — passthrough when no tools were
        // requested, otherwise watches for `<tool_call>…</tool_call>`
        // blocks and emits structured `ToolCallDelta` events.
        let mut tool_tracker = higgs_engine::tool_parser::IncrementalToolCallTracker::new(
            stream_includes_tools,
            tool_schema,
        );
        // Required/named calls stream append-only arguments immediately, but
        // retain enough bookkeeping to enforce the full postcondition before
        // emitting a successful finish.
        let mut required_visible = String::new();
        let mut required_calls: Vec<(String, String, bool)> = Vec::new();

        macro_rules! handle_tool_output {
            ($output:expr, $text_logprobs:expr) => {{
                let parser_timer = route_timing.then(Instant::now);
                let output = $output;
                if let Some(parser_started_at) = parser_timer {
                    tool_parser_elapsed += parser_started_at.elapsed();
                    tool_parser_calls = tool_parser_calls.saturating_add(1);
                }
                let mut adapter_error = None;
                for event in output.events {
                    use higgs_engine::tool_parser::ToolStreamEvent;
                    match event {
                        ToolStreamEvent::Text(text) => {
                            if required_call {
                                required_visible.push_str(&text);
                            } else if !text.is_empty() {
                                let d = ChatCompletionDelta {
                                    role: None,
                                    content: Some(text),
                                    reasoning_content: None,
                                    tool_calls: None,
                                };
                                metrics_guard.semantic_event();
                                emit_delta!(&d, None, $text_logprobs);
                            }
                        }
                        ToolStreamEvent::ToolStart { index, name } => {
                            if index != required_calls.len() {
                                adapter_error = Some(format!(
                                    "tool stream started out-of-order call index {index}"
                                ));
                                break;
                            }
                            if required_call {
                                let declared = required_tools
                                    .as_deref()
                                    .unwrap_or(&[])
                                    .iter()
                                    .any(|tool| {
                                        tool.get("function")
                                            .and_then(|function| function.get("name"))
                                            .and_then(serde_json::Value::as_str)
                                            == Some(name.as_str())
                                    });
                                if !declared
                                    || required_name
                                        .as_deref()
                                        .is_some_and(|expected| expected != name)
                                    || index > 0
                                {
                                    adapter_error = Some(format!(
                                        "required tool choice rejected call '{}' at index {index}",
                                        name
                                    ));
                                    break;
                                }
                            }
                            required_calls.push((name.clone(), String::new(), false));
                            let index = u32::try_from(index).unwrap_or(u32::MAX);
                            let d = ChatCompletionDelta {
                                role: None,
                                content: None,
                                reasoning_content: None,
                                tool_calls: Some(vec![ToolCallDelta {
                                    index,
                                    id: Some(format!("call_{index}_{}", uuid::Uuid::new_v4())),
                                    r#type: Some("function".to_owned()),
                                    function: Some(ToolCallFunctionDelta {
                                        name: Some(name),
                                        arguments: Some(String::new()),
                                    }),
                                }]),
                            };
                            metrics_guard.semantic_event();
                            emit_delta!(&d, None, None);
                        }
                        ToolStreamEvent::ArgumentsDelta { index, fragment } => {
                            let Some((_, arguments, _)) = required_calls.get_mut(index) else {
                                adapter_error = Some(format!(
                                    "tool arguments arrived before call index {index}"
                                ));
                                break;
                            };
                            arguments.push_str(&fragment);
                            if !fragment.is_empty() {
                                let index = u32::try_from(index).unwrap_or(u32::MAX);
                                let d = ChatCompletionDelta {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: Some(vec![ToolCallDelta {
                                        index,
                                        id: None,
                                        r#type: None,
                                        function: Some(ToolCallFunctionDelta {
                                            name: None,
                                            arguments: Some(fragment),
                                        }),
                                    }]),
                                };
                                metrics_guard.semantic_event();
                                emit_delta!(&d, None, None);
                            }
                        }
                        ToolStreamEvent::ToolEnd { index } => {
                            let Some((_, _, ended)) = required_calls.get_mut(index) else {
                                adapter_error = Some(format!(
                                    "tool end arrived before call index {index}"
                                ));
                                break;
                            };
                            *ended = true;
                        }
                    }
                }
                if adapter_error.is_none() {
                    adapter_error = output.error.map(|error| error.to_string());
                }
                if let Some(error) = adapter_error {
                    metrics_guard.fail(error.clone());
                    yield Ok(Event::default().data(streaming_error_json(&error)));
                    return;
                }
            }};
        }

        let mut output_token_count: u32 = 0;
        // Radix prefix-cache tokens reused this turn, taken from the prefill
        // progress events (`p.cached`). Reported as `prompt_tokens_details`.
        let mut cached_prompt_tokens: u32 = 0;
        let mut pending_finish_reason: Option<String> = None;
        let mut pending_finish_logprobs: Option<ChoiceLogprobs> = None;
        while let Some(output) = rx.recv().await {
            // Prefill-progress events carry no tokens: forward as
            // `prompt_progress` chunks when the client opted in, and keep
            // them away from the delta/tool trackers either way.
            if let Some(p) = output.prefill_progress {
                cached_prompt_tokens = cached_prompt_tokens.max(p.cached);
                metrics_guard.set_cached_tokens(u64::from(p.cached));
                if return_progress {
                    let time_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
                    let json = writer.write_prompt_progress(p.total, p.cached, p.processed, time_ms);
                    yield Ok(Event::default().data(json));
                }
                continue;
            }
            output_token_count = output.completion_tokens;
            metrics_guard.update(u64::from(output_token_count));
            let chunk_logprobs = output
                .token_logprob
                .as_ref()
                .map(|lp| logprobs_to_response(std::slice::from_ref(lp), &tokenizer));

            let (visible, reasoning) = reasoning_tracker.process(&output.new_text);

            if !reasoning.is_empty() {
                let d = ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: Some(reasoning),
                    tool_calls: None,
                };
                metrics_guard.semantic_event();
                emit_delta!(&d, None, None);
            }

            // Run the visible-text portion through the tool-call tracker
            // so `<tool_call>…</tool_call>` blocks become structured
            // deltas rather than being spoken aloud as plain text.
            let visible_is_empty = visible.is_empty();
            handle_tool_output!(tool_tracker.process(&visible), chunk_logprobs.as_ref());

            if let Some(finish_reason) = output.finish_reason {
                pending_finish_reason = Some(finish_reason);
                pending_finish_logprobs = if visible_is_empty { chunk_logprobs } else { None };
            }
        }

        let terminal = terminal_rx.await.unwrap_or_else(|error| {
            crate::sse::WorkerTerminal::Failed(format!(
                "streaming generation worker terminated unexpectedly: {error}"
            ))
        });
        let mut capacity_interrupted = false;
        match terminal {
            crate::sse::WorkerTerminal::Completed => {}
            crate::sse::WorkerTerminal::Capacity(info) => {
                capacity_interrupted = true;
                // Exact frozen v1 terminal event, then the normal tail
                // terminates the stream with [DONE].
                metrics_guard.fail("capacity_interrupted".to_owned());
                yield Ok(Event::default().data(crate::sse::capacity_interrupted_event_json(
                    &info,
                    u64::from(output_token_count),
                )));
            }
            crate::sse::WorkerTerminal::Failed(error) => {
                tracing::error!(error = %error, "Streaming generation terminated with an error");
                metrics_guard.fail(error.clone());
                yield Ok(Event::default().data(streaming_error_json(&error)));
                return;
            }
        }

        // A capacity terminal ends every stream. Discard buffered parser and
        // finish state so no content, tool, reasoning, or finish delta follows
        // the terminal; only optional usage and [DONE] remain.
        let discard_stream_remainder = capacity_interrupted;
        if discard_stream_remainder {
            pending_finish_reason = None;
            pending_finish_logprobs = None;
        }

        // Flush any remaining buffered content.
        let (flush_vis, flush_reas) = reasoning_tracker.flush();
        if !flush_reas.is_empty() && !discard_stream_remainder {
            let d = ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: Some(flush_reas),
                tool_calls: None,
            };
            metrics_guard.semantic_event();
            emit_delta!(&d, None, None);
        }
        if !discard_stream_remainder {
            handle_tool_output!(tool_tracker.process(&flush_vis), None);
            handle_tool_output!(tool_tracker.finish(), None);
        }

        // Required/named tool choice: judge the postcondition now that the
        // tracker has drained. Tentative arguments may already be visible,
        // but zero calls, leaked text, or multiple calls still end with the
        // typed error event and no successful finish. A capacity interruption
        // already ended the request with its own terminal.
        if !discard_stream_remainder {
            if required_call {
                let parsed_calls = required_calls
                    .into_iter()
                    .filter_map(|(name, arguments_text, ended)| {
                        ended.then(|| {
                            serde_json::from_str(&arguments_text)
                                .map(|arguments_value| higgs_engine::tool_parser::ParsedToolCall {
                                    name,
                                    arguments: arguments_value,
                                })
                        })
                    })
                    .collect::<Result<Vec<_>, _>>();
                let parsed = parsed_calls.map(|tool_calls| higgs_engine::tool_parser::ToolParseResult {
                    text: required_visible,
                    tool_calls,
                });
                let validation = parsed
                    .map_err(|error| error.to_string())
                    .and_then(|parsed_result| {
                        required_tool_call_parts(
                            &parsed_result,
                            required_tools.as_deref().unwrap_or(&[]),
                            required_name.as_deref(),
                        )
                        .map(|_| ())
                        .map_err(|error| error.to_string())
                    });
                if let Err(error) = validation {
                    metrics_guard.fail(error.clone());
                    yield Ok(Event::default().data(streaming_error_json(&error)));
                    return;
                }
            }
        }

        // Defer `finish_reason` until after the tracker has drained so we
        // know whether to report `"tool_calls"` or `"stop"`.
        if let Some(finish_reason) = pending_finish_reason {
            let effective_finish = if tool_tracker.completed_call_count() > 0 {
                "tool_calls".to_owned()
            } else {
                finish_reason
            };
            let d = ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: None,
            };
            emit_delta!(&d, Some(effective_finish.as_str()), pending_finish_logprobs.as_ref());
        }

        // Emit final chunk with usage only when explicitly requested.
        if include_usage || required_retention.is_some() {
            let retention_receipt = required_retention.as_ref().and_then(|retention| {
                stream_session_id.and_then(|session_id| {
                    engine.retained_session_receipt(session_id).map(|(tokens, bytes)| {
                        RetentionReceipt {
                            outcome: match retention.mode {
                                crate::types::openai::RetentionMode::Seed => "seeded",
                                crate::types::openai::RetentionMode::Required => "continued",
                            },
                            session_id,
                            epoch: retention.epoch,
                            retained_tokens: u64::try_from(tokens).unwrap_or(u64::MAX),
                            retained_bytes: u64::try_from(bytes).unwrap_or(u64::MAX),
                            contract_revision: retention.contract_revision.clone(),
                        }
                    })
                })
            });
            if let Some(retention) = required_retention.as_ref() {
                if retention_receipt.is_none() {
                    let json = serde_json::json!({
                        "error": {
                            "message": format!("Retained session {} is unavailable for required continuation", retention.session_id),
                            "type": "conflict",
                            "code": "retained_session_unavailable",
                            "contractRevision": retention.contract_revision,
                            "sessionId": retention.session_id,
                            "epoch": retention.epoch,
                        }
                    });
                    yield Ok(Event::default().data(json.to_string()));
                    return;
                }
            }
            let usage = CompletionUsage::new(
                prompt_token_count,
                output_token_count,
                cached_prompt_tokens,
            )
            .with_session_lease_active(lease_active)
            .with_retention_receipt(retention_receipt);
            match writer.write_usage(&usage) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize usage chunk"),
            }
        }

        metrics_guard.finish();

        #[allow(clippy::print_stderr)] // Existing env-gated session performance diagnostic.
        if let Some(route_started_at) = route_timer {
            eprintln!(
                "DIAG chat-stream-route: tool_parser_calls={tool_parser_calls} tool_parser={tool_parser_elapsed:.2?} delta_yields={delta_yields} delta_serialize={delta_serialize_elapsed:.2?} delta_yield_resume={delta_yield_elapsed:.2?} total={:.2?}",
                route_started_at.elapsed(),
            );
        }

        // Send [DONE] sentinel
        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(Box::pin(stream))
}

fn convert_messages(
    messages: &[ChatCompletionMessage],
) -> Vec<higgs_engine::chat_template::ChatMessage> {
    messages
        .iter()
        .map(|m| {
            let tool_calls_json = m.tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .filter_map(|tc| serde_json::to_value(tc).ok())
                    .map(|mut tc_value| {
                        // Make the tool call template-friendly: hoist
                        // `function.{name,arguments}` to the top level
                        // and parse string-encoded arguments to a JSON
                        // value. Without this, Qwen's chat template
                        // crashes on `tool_call.arguments|items`.
                        higgs_engine::chat_template::normalize_tool_call_for_template(
                            &mut tc_value,
                        );
                        tc_value
                    })
                    .collect()
            });
            let content = m
                .content
                .as_ref()
                .map_or_else(String::new, MessageContent::text);
            higgs_engine::chat_template::ChatMessage {
                role: m.role.clone(),
                content,
                tool_calls: tool_calls_json,
            }
        })
        .collect()
}

/// Map an engine error to a server error, surfacing vision preprocessing
/// failures (malformed client image data) as strict 400s and passing every
/// other engine error through as a 500.
///
/// Shared with the Anthropic route so both surfaces map `EngineError::Vision`
/// identically.
pub(crate) fn map_engine_error(e: higgs_engine::error::EngineError) -> ServerError {
    match e {
        higgs_engine::error::EngineError::Vision(v) => ServerError::BadRequest(v.to_string()),
        higgs_engine::error::EngineError::RetainedSessionUnavailable(session_id) => {
            ServerError::RetainedSessionUnavailable(session_id)
        }
        other @ (higgs_engine::error::EngineError::Model(_)
        | higgs_engine::error::EngineError::Mlx(_)
        | higgs_engine::error::EngineError::Tokenization(_)
        | higgs_engine::error::EngineError::Template(_)
        | higgs_engine::error::EngineError::Generation(_)
        | higgs_engine::error::EngineError::Cancelled
        | higgs_engine::error::EngineError::CapacityInterrupted { .. }) => {
            ServerError::Engine(other)
        }
    }
}

/// Reject images when the resolved model has no vision support.
///
/// Shared with the Anthropic route so both surfaces enforce the same 400 gate.
pub(crate) fn check_vision_capability(
    media: &[MediaItem],
    engine_is_vlm: bool,
    model_name: &str,
) -> Result<(), ServerError> {
    if !media.is_empty() && !engine_is_vlm {
        return Err(ServerError::BadRequest(format!(
            "model {model_name} does not support vision (image input); \
             use a vision-capable model (e.g. llava-qwen2)"
        )));
    }
    Ok(())
}

/// Rebuild message content with the family marker inserted at each image
/// part's true position. Text parts keep their relative order.
fn render_markers(
    messages: &[ChatCompletionMessage],
    marker: Option<&'static str>,
) -> Vec<ChatCompletionMessage> {
    let marker_text = marker.unwrap_or("<image>");
    messages
        .iter()
        .map(|m| {
            let Some(content) = &m.content else {
                return m.clone();
            };
            let MessageContent::Parts(parts) = content else {
                return m.clone();
            };
            let mut out = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => out.push_str(text),
                    ContentPart::ImageUrl { .. } => out.push_str(marker_text),
                }
            }
            ChatCompletionMessage {
                role: m.role.clone(),
                content: Some(MessageContent::Text(out)),
                reasoning_content: m.reasoning_content.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            }
        })
        .collect()
}

fn build_sampling_params(
    req: &ChatCompletionRequest,
    defaults: &GenerationDefaults,
) -> Result<SamplingParams, ServerError> {
    let speculation = higgs_models::Speculation::parse(
        req.speculation
            .as_deref()
            .or(defaults.speculation.as_deref()),
    )
    .map_err(|v| {
        ServerError::BadRequest(format!(
            "invalid 'speculation' value '{v}' (expected auto|dflash|mtp|none)"
        ))
    })?;
    let repetition_penalty = if req.repetition_penalty.is_some() || req.repeat_penalty.is_some() {
        merge_repetition_penalty(req.repetition_penalty, req.repeat_penalty)
    } else {
        defaults.repetition_penalty
    };
    Ok(SamplingParams {
        temperature: req.temperature.or(defaults.temperature).unwrap_or(0.0),
        top_p: req.top_p.or(defaults.top_p).unwrap_or(1.0),
        top_k: req.top_k.or(defaults.top_k),
        min_p: req.min_p.or(defaults.min_p),
        repetition_penalty,
        frequency_penalty: req.frequency_penalty.or(defaults.frequency_penalty),
        presence_penalty: req.presence_penalty.or(defaults.presence_penalty),
        speculation,
        thinking_budget: req.reasoning_budget,
    })
}

fn resolved_max_tokens(
    req: &ChatCompletionRequest,
    defaults: &GenerationDefaults,
    server_max_tokens: u32,
) -> u32 {
    req.max_tokens
        .or(defaults.max_tokens)
        .unwrap_or(server_max_tokens)
}

const fn chat_prompt_mode(session_id: Option<u64>, max_tokens: u32) -> ChatPromptMode {
    if session_id.is_some() && max_tokens == 0 {
        ChatPromptMode::SessionPrefill
    } else {
        ChatPromptMode::Generation
    }
}

/// Build a constrained generator from the request's response or tool format.
///
/// Returns `None` if no constraint is needed (text mode or absent).
fn build_constraint(
    response_format: Option<&crate::types::openai::ResponseFormat>,
    tool_call_schema: Option<&serde_json::Value>,
    engine: &std::sync::Arc<crate::state::Engine>,
) -> Result<Option<higgs_engine::constrained::ConstrainedGenerator>, ServerError> {
    if let Some(schema) = tool_call_schema {
        let eos_id = engine.eos_token_ids().first().copied().unwrap_or(0);
        let vocab = higgs_engine::constrained::build_vocabulary(engine.tokenizer(), eos_id)
            .map_err(ServerError::Engine)?;
        return higgs_engine::constrained::ConstrainedGenerator::from_tagged_json_schema(
            &schema.to_string(),
            &vocab,
        )
        .map(Some)
        .map_err(|error| {
            ServerError::BadRequest(format!("Unsupported tool parameter schema: {error}"))
        });
    }
    let Some(fmt) = response_format else {
        return Ok(None);
    };

    match fmt.r#type.as_str() {
        "text" => Ok(None),
        "json_object" | "json_schema" => {
            let eos_id = engine.eos_token_ids().first().copied().unwrap_or(0);
            let vocab = higgs_engine::constrained::build_vocabulary(engine.tokenizer(), eos_id)
                .map_err(ServerError::Engine)?;
            let constraint = if fmt.r#type == "json_schema" {
                if let Some(ref schema) = fmt.json_schema {
                    // OpenAI spec wraps the actual schema under a `schema` key:
                    // {"name": "...", "schema": {<actual schema>}}
                    // Fall back to the whole value for bare schemas.
                    let inner = schema
                        .get("schema")
                        .cloned()
                        .unwrap_or_else(|| schema.clone());
                    let schema_str = inner.to_string();
                    higgs_engine::constrained::ConstrainedGenerator::from_json_schema(
                        &schema_str,
                        &vocab,
                    )
                    .map_err(ServerError::Engine)?
                } else {
                    higgs_engine::constrained::ConstrainedGenerator::for_json_object(&vocab)
                        .map_err(ServerError::Engine)?
                }
            } else {
                higgs_engine::constrained::ConstrainedGenerator::for_json_object(&vocab)
                    .map_err(ServerError::Engine)?
            };

            Ok(Some(constraint))
        }
        other => Err(ServerError::BadRequest(format!(
            "Unsupported response_format type: {other}"
        ))),
    }
}

fn logprobs_to_response(
    infos: &[higgs_models::TokenLogprobInfo],
    tokenizer: &higgs_engine::tokenizers::Tokenizer,
) -> ChoiceLogprobs {
    let content = infos
        .iter()
        .map(|info| {
            let token_str = tokenizer
                .decode(&[info.token_id], false)
                .unwrap_or_default();
            let top = info
                .top_logprobs
                .iter()
                .map(|e| {
                    let t = tokenizer.decode(&[e.token_id], false).unwrap_or_default();
                    TopLogprob {
                        token: t,
                        logprob: e.logprob,
                    }
                })
                .collect();
            TokenLogprob {
                token: token_str,
                logprob: info.logprob,
                top_logprobs: top,
            }
        })
        .collect();
    ChoiceLogprobs { content }
}

fn generate_request_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

fn session_continuation_id(
    session_id: Option<u64>,
    has_image_inputs: bool,
    has_constraint: bool,
    checkpoint_id: Option<&str>,
    want_logprobs: bool,
    has_stop_sequences: bool,
) -> Option<u64> {
    session_id
        .filter(|_| !has_image_inputs)
        .filter(|_| !has_constraint)
        .filter(|_| checkpoint_id.is_none())
        .filter(|_| !want_logprobs)
        .filter(|_| !has_stop_sequences)
}

fn validate_prompt_limit(
    max_prompt_tokens: Option<u32>,
    prompt_tokens: usize,
) -> Result<(), ServerError> {
    if let Some(limit) = max_prompt_tokens {
        if prompt_tokens > usize::try_from(limit).unwrap_or(usize::MAX) {
            return Err(ServerError::ContextLengthExceeded {
                prompt_tokens,
                max_prompt_tokens: limit,
            });
        }
    }
    Ok(())
}

fn validate_session_lease_ttl(ttl_seconds: Option<u32>) -> Result<(), ServerError> {
    if ttl_seconds.is_some_and(|ttl| ttl == 0 || ttl > 300) {
        return Err(ServerError::BadRequest(
            "session_lease.ttl_seconds must be between 1 and 300".to_owned(),
        ));
    }
    Ok(())
}

async fn drop_requested_retained_sessions(
    engine: Arc<Engine>,
    session_id: Option<u64>,
    session_ids: Option<&[u64]>,
) -> Result<(), ServerError> {
    let ids = retained_session_drop_ids(session_id, session_ids);
    if ids.is_empty() {
        return Ok(());
    }

    tokio::task::spawn_blocking(move || {
        for session_id in ids {
            let dropped = engine.drop_retained_session(session_id);
            tracing::info!(
                session_id,
                dropped,
                "retained session drop requested by client"
            );
        }
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?;

    Ok(())
}

fn retained_session_drop_ids(session_id: Option<u64>, session_ids: Option<&[u64]>) -> Vec<u64> {
    let mut ids = Vec::new();
    if let Some(session_id) = session_id {
        ids.push(session_id);
    }
    if let Some(session_ids) = session_ids {
        ids.extend_from_slice(session_ids);
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

#[derive(Debug, serde::Deserialize)]
pub struct DropSessionsRequest {
    pub model: String,
    #[serde(default)]
    pub session_ids: Vec<u64>,
    pub session_id: Option<u64>,
}

/// Eager retained-session drop. Lets a client free a rotated session's
/// resident KV immediately — before the next (smaller) prompt prefills —
/// instead of piggybacking the drop on that request. Sessions still in
/// flight report `dropped: false` and remain covered by the piggyback path.
pub async fn drop_sessions(
    State(state): State<SharedState>,
    Json(req): Json<DropSessionsRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let ids = retained_session_drop_ids(req.session_id, Some(&req.session_ids));
    if ids.is_empty() {
        return Err(ServerError::BadRequest(
            "no session ids provided".to_owned(),
        ));
    }
    let resolved = state
        .router
        .resolve(&req.model, None)
        .await
        .map_err(ServerError::ModelNotFound)?;
    let ResolvedRoute::Higgs { engine, .. } = resolved else {
        return Err(ServerError::BadRequest(
            "session drop requires a higgs-routed model".to_owned(),
        ));
    };
    state.drop_retention_bindings(&req.model, &ids);
    let mut dropped = Vec::with_capacity(ids.len());
    for session_id in ids {
        dropped.push(serde_json::json!({
            "session_id": session_id,
            "dropped": engine.try_drop_retained_session(session_id),
        }));
    }
    Ok(Json(serde_json::json!({ "dropped": dropped })))
}

fn current_unix_timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}

#[allow(clippy::indexing_slicing, clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use axum::body::Body;
    use axum::response::IntoResponse;
    use axum::routing::post;
    use http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn streaming_test_state() -> SharedState {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[provider.stub]\nurl = \"http://127.0.0.1:1\"\n").unwrap();
        let config = crate::config::load_config_file(&path, None).unwrap();
        let router = crate::router::Router::from_config(&config, HashMap::new()).unwrap();
        Arc::new(crate::state::AppState::new(
            router,
            config,
            reqwest::Client::new(),
            None,
        ))
    }

    fn axum_session_test_app(engine_name: &str) -> (axum::Router, Arc<Engine>) {
        let engine = Arc::new(Engine::test_stub(engine_name));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[provider.stub]\nurl = \"http://127.0.0.1:1\"\n").unwrap();
        let config = crate::config::load_config_file(&path, None).unwrap();
        let router = crate::router::Router::from_config(
            &config,
            HashMap::from([(engine_name.to_owned(), Arc::clone(&engine))]),
        )
        .unwrap();
        let state = Arc::new(crate::state::AppState::new(
            router,
            config,
            reqwest::Client::new(),
            None,
        ));
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);
        (app, engine)
    }

    fn axum_chat_request(model: &str, extra: serde_json::Value) -> Request<Body> {
        let mut body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}]
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let mut request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        // The production router injects this extension via its metrics
        // middleware; the bare test router must do the same or the handler's
        // `Extension<RequestMetricsContext>` extractor rejects the request.
        request
            .extensions_mut()
            .insert(crate::metrics::RequestMetricsContext::default());
        request
    }

    fn axum_sse_events(body: &str) -> (Vec<serde_json::Value>, usize) {
        let mut events = Vec::new();
        let mut done = 0;
        for line in body.lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                done += 1;
            } else {
                events.push(serde_json::from_str(data).unwrap());
            }
        }
        (events, done)
    }

    fn chat_request(extra: serde_json::Value) -> ChatCompletionRequest {
        let mut request = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        });
        if let (Some(dst), Some(src)) = (request.as_object_mut(), extra.as_object()) {
            dst.extend(src.clone());
        }
        serde_json::from_value(request).unwrap()
    }

    #[test]
    fn streaming_error_uses_openai_error_envelope() {
        let value: serde_json::Value =
            serde_json::from_str(&streaming_error_json("decode failed")).unwrap();
        assert_eq!(value["error"]["message"], "decode failed");
        assert_eq!(value["error"]["type"], "server_error");
        assert_eq!(value["error"]["code"], "generation_error");
        assert_ne!(value, serde_json::json!("[DONE]"));
    }

    fn weather_and_shell_tools() -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "shell",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": false
                    }
                }
            }),
        ]
    }

    #[test]
    fn required_tool_choice_builds_one_schema_variant_per_function() {
        let tools = weather_and_shell_tools();
        let choice =
            crate::types::openai::ToolChoice::Mode(crate::types::openai::ToolChoiceMode::Required);
        let plan = resolve_tool_choice(Some(&choice), Some(&tools), None).unwrap();

        assert_eq!(plan.prompt_tools.unwrap().len(), 2);
        let variants = plan.constraint_schema.unwrap()["oneOf"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0]["properties"]["name"]["const"], "weather");
        assert_eq!(
            variants[1]["properties"]["arguments"]["required"][0],
            "command"
        );
    }

    #[test]
    fn named_tool_choice_restricts_schema_and_validates_name() {
        let tools = weather_and_shell_tools();
        let named =
            crate::types::openai::ToolChoice::Named(crate::types::openai::NamedToolChoice {
                r#type: "function".to_owned(),
                function: crate::types::openai::NamedToolChoiceFunction {
                    name: "shell".to_owned(),
                },
            });
        let plan = resolve_tool_choice(Some(&named), Some(&tools), None).unwrap();
        assert_eq!(
            plan.constraint_schema.unwrap()["properties"]["name"]["const"],
            "shell"
        );

        let missing =
            crate::types::openai::ToolChoice::Named(crate::types::openai::NamedToolChoice {
                r#type: "function".to_owned(),
                function: crate::types::openai::NamedToolChoiceFunction {
                    name: "missing".to_owned(),
                },
            });
        assert!(matches!(
            resolve_tool_choice(Some(&missing), Some(&tools), None),
            Err(ServerError::BadRequest(_))
        ));
    }

    #[test]
    fn automatic_and_none_tool_choices_do_not_add_constraints() {
        let tools = weather_and_shell_tools();
        let automatic =
            crate::types::openai::ToolChoice::Mode(crate::types::openai::ToolChoiceMode::Auto);
        let auto_plan = resolve_tool_choice(Some(&automatic), Some(&tools), None).unwrap();
        assert_eq!(auto_plan.prompt_tools.unwrap().len(), 2);
        assert!(auto_plan.constraint_schema.is_none());

        let none =
            crate::types::openai::ToolChoice::Mode(crate::types::openai::ToolChoiceMode::None);
        let none_plan = resolve_tool_choice(Some(&none), Some(&tools), None).unwrap();
        assert!(none_plan.prompt_tools.is_none());
        assert!(none_plan.constraint_schema.is_none());
    }

    #[test]
    fn required_tool_choice_rejects_missing_tools_and_json_response_format() {
        let required =
            crate::types::openai::ToolChoice::Mode(crate::types::openai::ToolChoiceMode::Required);
        assert!(matches!(
            resolve_tool_choice(Some(&required), None, None),
            Err(ServerError::BadRequest(_))
        ));

        let tools = weather_and_shell_tools();
        let response_format = crate::types::openai::ResponseFormat {
            r#type: "json_schema".to_owned(),
            json_schema: Some(serde_json::json!({"schema": {"type": "object"}})),
        };
        assert!(matches!(
            resolve_tool_choice(Some(&required), Some(&tools), Some(&response_format)),
            Err(ServerError::BadRequest(_))
        ));
    }

    #[test]
    fn required_tool_choice_rejects_non_object_arguments_and_duplicate_names() {
        let required =
            crate::types::openai::ToolChoice::Mode(crate::types::openai::ToolChoiceMode::Required);
        let non_object = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "bad",
                "parameters": {"type": "array", "items": {"type": "string"}}
            }
        })];
        assert!(matches!(
            resolve_tool_choice(Some(&required), Some(&non_object), None),
            Err(ServerError::BadRequest(_))
        ));

        let duplicate = vec![
            serde_json::json!({
                "type": "function",
                "function": {"name": "same", "parameters": {"type": "object"}}
            }),
            serde_json::json!({
                "type": "function",
                "function": {"name": "same", "parameters": {"type": "object"}}
            }),
        ];
        assert!(matches!(
            resolve_tool_choice(Some(&required), Some(&duplicate), None),
            Err(ServerError::BadRequest(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Required/named tool-choice materialization postcondition
    // -----------------------------------------------------------------------

    fn parsed_call(
        name: &str,
        arguments: serde_json::Value,
    ) -> higgs_engine::tool_parser::ParsedToolCall {
        higgs_engine::tool_parser::ParsedToolCall {
            name: name.to_owned(),
            arguments,
        }
    }

    fn parse_result(
        text: &str,
        calls: Vec<higgs_engine::tool_parser::ParsedToolCall>,
    ) -> higgs_engine::tool_parser::ToolParseResult {
        higgs_engine::tool_parser::ToolParseResult {
            text: text.to_owned(),
            tool_calls: calls,
        }
    }

    /// Assert the parse outcome fails closed with the typed engine error and
    /// return its message.
    fn expect_required_failure(
        parsed: &higgs_engine::tool_parser::ToolParseResult,
        required_name: Option<&str>,
    ) -> String {
        let tools = weather_and_shell_tools();
        let error = required_tool_call_parts(parsed, &tools, required_name)
            .expect_err("degenerate required-call materialization must fail closed");
        match error {
            ServerError::Engine(higgs_engine::error::EngineError::Generation(message)) => message,
            other => panic!("expected typed EngineError::Generation, got {other:?}"),
        }
    }

    #[test]
    fn blocking_required_materialization_accepts_exactly_one_valid_call() {
        let parsed = parse_result(
            "",
            vec![parsed_call("weather", serde_json::json!({"city": "Rome"}))],
        );
        let tools = weather_and_shell_tools();
        let (content, tool_calls) = required_tool_call_parts(&parsed, &tools, None)
            .expect("one schema-valid call is the only success shape");
        assert!(
            content.is_none(),
            "a required call carries no visible content"
        );
        let calls = tool_calls.expect("exactly one tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Rome"}"#);
    }

    #[test]
    fn blocking_required_materialization_fails_on_zero_calls() {
        let parsed = parse_result("no call at all", vec![]);
        let message = expect_required_failure(&parsed, None);
        assert!(message.contains("expected exactly one"), "{message}");
    }

    #[test]
    fn blocking_required_materialization_fails_on_leftover_text() {
        let parsed = parse_result(
            "sure, calling",
            vec![parsed_call("weather", serde_json::json!({"city": "Rome"}))],
        );
        let message = expect_required_failure(&parsed, None);
        assert!(
            message.contains("outside the tool_call envelope"),
            "{message}"
        );
    }

    #[test]
    fn blocking_required_materialization_fails_on_whitespace_only_visible_text() {
        // The postcondition is zero visible *bytes*: whitespace is still
        // leaked text, reported by its original byte length.
        let parsed = parse_result(
            " \t\n",
            vec![parsed_call("weather", serde_json::json!({"city": "Rome"}))],
        );
        let message = expect_required_failure(&parsed, None);
        assert!(message.contains("3 bytes of visible text"), "{message}");
    }

    #[test]
    fn blocking_required_materialization_fails_on_multiple_calls() {
        let call = parsed_call("weather", serde_json::json!({"city": "Rome"}));
        let parsed = parse_result("", vec![call.clone(), call]);
        let message = expect_required_failure(&parsed, None);
        assert!(message.contains("2 parser-visible tool calls"), "{message}");
    }

    #[test]
    fn blocking_required_materialization_fails_on_wrong_named_function() {
        let parsed = parse_result(
            "",
            vec![parsed_call("weather", serde_json::json!({"city": "Rome"}))],
        );
        let message = expect_required_failure(&parsed, Some("shell"));
        assert!(
            message.contains("tool_choice required 'shell'"),
            "{message}"
        );
    }

    #[test]
    fn blocking_required_materialization_fails_on_non_object_arguments() {
        // Argument *shape* beyond "must be an object" is the grammar FSM's job
        // (see `required_tool_call_parts`); a non-object arguments value is
        // rejected here.
        let parsed = parse_result("", vec![parsed_call("weather", serde_json::json!("Rome"))]);
        let message = expect_required_failure(&parsed, None);
        assert!(message.contains("must be a JSON object"), "{message}");
    }

    #[test]
    fn blocking_required_materialization_fails_on_undeclared_function() {
        let parsed = parse_result("", vec![parsed_call("ghost", serde_json::json!({}))]);
        let message = expect_required_failure(&parsed, None);
        assert!(message.contains("is not a declared tool"), "{message}");
    }

    #[test]
    fn streaming_required_materialization_rejects_degenerate_tracker_output() {
        // Mirrors the streaming route's accumulation: tracker output (visible
        // leaks plus completed calls) is validated once at end-of-stream with
        // the same helper as the blocking and session paths.
        let drain = |chunks: &[&str]| {
            let mut tracker = higgs_engine::tool_parser::StreamingToolCallTracker::new(
                true,
                higgs_engine::tool_parser::ToolSchema::from_tools(Some(&weather_and_shell_tools())),
            );
            let mut visible = String::new();
            let mut calls = Vec::new();
            for chunk in chunks {
                let out = tracker.process(chunk);
                visible.push_str(&out.visible);
                calls.extend(out.new_tool_calls);
            }
            let flush = tracker.flush();
            visible.push_str(&flush.visible);
            calls.extend(flush.new_tool_calls);
            let parsed = higgs_engine::tool_parser::ToolParseResult {
                text: visible,
                tool_calls: calls,
            };
            required_tool_call_parts(&parsed, &weather_and_shell_tools(), None)
        };

        let good = drain(&[
            "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Rome\"}}\n",
            "</tool_call>",
        ])
        .expect("a single well-formed streamed call must pass");
        assert_eq!(good.1.expect("call").len(), 1);

        // Two complete calls are a terminal failure even though both parsed.
        let double = drain(&[
            "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>",
            "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
        ])
        .expect_err("multiple streamed calls must fail closed");
        assert!(matches!(
            double,
            ServerError::Engine(higgs_engine::error::EngineError::Generation(_))
        ));

        // A truncated call flushes as visible text — never a silent success.
        let truncated = drain(&["<tool_call>\n{\"name\": \"weather\""])
            .expect_err("truncated streamed call must fail closed");
        assert!(matches!(
            truncated,
            ServerError::Engine(higgs_engine::error::EngineError::Generation(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Required/named streaming SSE behavior (stub engine)
    // -----------------------------------------------------------------------

    async fn required_stream_body(model: &str) -> String {
        let request = chat_request(serde_json::json!({
            "model": model,
            "stream": true,
            "tools": weather_and_shell_tools(),
            "tool_choice": "required"
        }));
        let stream = chat_completions_stream(
            streaming_test_state(),
            request,
            Arc::new(Engine::test_stub(model)),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await
        .expect("admitted required stream must start");
        let response = axum::response::sse::Sse::new(stream).into_response();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(body.to_vec()).unwrap()
    }

    async fn automatic_stream_body(model: &str) -> String {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"]
                }
            }
        })];
        let request = chat_request(serde_json::json!({
            "model": model,
            "stream": true,
            "tools": tools
        }));
        let stream = chat_completions_stream(
            streaming_test_state(),
            request,
            Arc::new(Engine::test_stub(model)),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await
        .unwrap();
        let response = axum::response::sse::Sse::new(stream).into_response();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(body.to_vec()).unwrap()
    }

    fn openai_stream_data(body: &str) -> Vec<serde_json::Value> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn automatic_long_tool_call_streams_append_only_argument_deltas() {
        let body = automatic_stream_body("required-stream-script-long").await;
        let chunks = openai_stream_data(&body);
        let tool_deltas: Vec<&serde_json::Value> = chunks
            .iter()
            .filter_map(|chunk| {
                let delta = &chunk["choices"][0]["delta"]["tool_calls"][0];
                delta.is_object().then_some(delta)
            })
            .collect();

        assert!(tool_deltas.len() > 2, "expected incremental deltas: {body}");
        assert_eq!(tool_deltas[0]["index"], 0);
        assert_eq!(tool_deltas[0]["type"], "function");
        assert_eq!(tool_deltas[0]["function"]["name"], "write");
        assert_eq!(tool_deltas[0]["function"]["arguments"], "");
        let call_id = tool_deltas[0]["id"].as_str().unwrap();
        assert!(call_id.starts_with("call_0_"), "{call_id}");
        assert_eq!(
            tool_deltas
                .iter()
                .filter(|delta| delta.get("id").is_some())
                .count(),
            1,
            "identity must be sent exactly once: {body}"
        );
        assert!(tool_deltas.iter().skip(1).all(|delta| {
            delta.get("id").is_none()
                && delta["function"].get("name").is_none()
                && delta["function"]["arguments"]
                    .as_str()
                    .is_some_and(|fragment| !fragment.is_empty())
        }));
        let arguments: String = tool_deltas
            .iter()
            .filter_map(|delta| delta["function"]["arguments"].as_str())
            .collect();
        let parsed: serde_json::Value = serde_json::from_str(&arguments).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({"content": "chunk ".repeat(16 * 80)})
        );

        let first_tool = chunks
            .iter()
            .position(|chunk| !chunk["choices"][0]["delta"]["tool_calls"].is_null())
            .unwrap();
        let text_before_tool: String = chunks[..first_tool]
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text_before_tool, "Preparing.\n", "{body}");
        assert!(
            chunks
                .iter()
                .any(|chunk| { chunk["choices"][0]["finish_reason"] == "tool_calls" })
        );
    }

    #[tokio::test]
    async fn required_streaming_emits_nothing_before_validation_then_one_call_and_done() {
        let body = required_stream_body("required-stream-valid-call").await;
        let chunks = openai_stream_data(&body);
        let tool_deltas: Vec<&serde_json::Value> = chunks
            .iter()
            .filter_map(|chunk| {
                let delta = &chunk["choices"][0]["delta"]["tool_calls"][0];
                delta.is_object().then_some(delta)
            })
            .collect();
        let arguments: String = tool_deltas
            .iter()
            .filter_map(|delta| delta["function"]["arguments"].as_str())
            .collect();

        assert!(!body.contains("\"content\":\""), "{body}");
        assert_eq!(
            tool_deltas
                .iter()
                .filter(|delta| delta.get("id").is_some())
                .count(),
            1,
            "{body}"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
            serde_json::json!({"city": "Rome"}),
            "{body}"
        );
        assert!(body.contains("\"finish_reason\":\"tool_calls\""), "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
        assert!(!body.contains("generation_error"), "{body}");
    }

    #[tokio::test]
    async fn required_streaming_leaked_text_emits_only_the_typed_error_event() {
        let body = required_stream_body("required-stream-leaky-text").await;
        let chunks = openai_stream_data(&body);
        let tool_deltas: Vec<&serde_json::Value> = chunks
            .iter()
            .filter_map(|chunk| {
                let delta = &chunk["choices"][0]["delta"]["tool_calls"][0];
                delta.is_object().then_some(delta)
            })
            .collect();
        let arguments: String = tool_deltas
            .iter()
            .filter_map(|delta| delta["function"]["arguments"].as_str())
            .collect();
        // Argument deltas are tentative: the route may expose them once the
        // call identity is valid, but leaked prose still prevents a successful
        // finish and nothing follows the terminal error.
        assert!(!body.contains("\"content\":\""), "{body}");
        assert_eq!(
            tool_deltas
                .iter()
                .filter(|delta| delta.get("id").is_some())
                .count(),
            1,
            "{body}"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
            serde_json::json!({"city": "Rome"}),
            "{body}"
        );
        assert!(body.contains("outside the tool_call envelope"), "{body}");
        assert!(!body.contains("\"finish_reason\":\"tool_calls\""), "{body}");
        assert!(!body.contains("data: [DONE]"), "{body}");
        let data_lines: Vec<&str> = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect();
        assert!(
            data_lines
                .last()
                .is_some_and(|data| data.contains("generation_error")),
            "{body}"
        );
    }

    #[tokio::test]
    async fn required_streaming_capacity_emits_only_capacity_terminal_then_done() {
        let body = required_stream_body("capacity-interrupted").await;
        // The capacity terminal has already ended the request: exactly one
        // capacity event, no required-call postvalidation error, no tool
        // deltas, then the normal tail terminator.
        assert_eq!(
            body.matches("higgs_capacity_interrupted").count(),
            1,
            "{body}"
        );
        assert!(!body.contains("generation_error"), "{body}");
        assert!(!body.contains("call_0_"), "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
    }

    #[tokio::test]
    async fn required_streaming_capacity_after_partial_call_emits_only_terminal_then_done() {
        // Partial `<tool_call>` bytes reach the trackers before the engine
        // dies: the buffered prefix and every tracker remainder must be
        // discarded after the typed capacity terminal — no content delta, no
        // tool delta, no generation error, only usage/[DONE] tail.
        let body = required_stream_body("required-stream-capacity-partial-call").await;
        assert_eq!(
            body.matches("higgs_capacity_interrupted").count(),
            1,
            "{body}"
        );
        assert!(!body.contains("<tool_call>"), "{body}");
        assert!(!body.contains("\"content\":\""), "{body}");
        assert!(!body.contains("call_0_"), "{body}");
        assert!(!body.contains("generation_error"), "{body}");
        // No finish delta may follow the capacity terminal: only [DONE].
        let terminal = body.find("higgs_capacity_interrupted").expect("terminal");
        let tail: String = body[terminal..]
            .split("\n\n")
            .skip(1)
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(tail.trim(), "data: [DONE]", "{body}");
    }

    #[test]
    fn session_required_materialization_failure_is_terminal() {
        let output = higgs_engine::simple::SessionGeneration {
            text: "plain text, no tool call".to_owned(),
            completion_tokens: 5,
            finish_reason: "stop".to_owned(),
            prompt_tokens: 3,
            prefilled_tokens: 3,
            continued: false,
            outcome: higgs_engine::simple::SessionOutcome::ExactBootstrap,
        };
        let tools = weather_and_shell_tools();
        let error = build_session_response(
            "model",
            "request",
            output,
            Some(&tools),
            true,
            false,
            false,
            true,
            None,
            None,
        )
        .expect_err("session-routed required materialization must fail closed");
        assert!(matches!(
            error,
            ServerError::Engine(higgs_engine::error::EngineError::Generation(_))
        ));
    }

    #[test]
    fn generation_defaults_fill_omitted_sampling_fields() {
        let req = chat_request(serde_json::json!({}));
        let defaults = GenerationDefaults {
            max_tokens: Some(4096),
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.0),
            repetition_penalty: Some(1.1),
            frequency_penalty: Some(0.2),
            presence_penalty: Some(0.3),
            speculation: Some("none".to_owned()),
            enable_thinking: Some(false),
        };

        assert_eq!(resolved_max_tokens(&req, &defaults, 1024), 4096);
        let sampling = build_sampling_params(&req, &defaults).unwrap();
        assert!((sampling.temperature - 0.7).abs() < f32::EPSILON);
        assert!((sampling.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(sampling.top_k, Some(20));
        assert_eq!(sampling.min_p, Some(0.0));
        assert_eq!(sampling.repetition_penalty, Some(1.1));
        assert_eq!(sampling.frequency_penalty, Some(0.2));
        assert_eq!(sampling.presence_penalty, Some(0.3));
        assert_eq!(sampling.speculation, higgs_models::Speculation::None);
    }

    #[test]
    fn request_sampling_fields_override_generation_defaults() {
        let req = chat_request(serde_json::json!({
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 5,
            "repeat_penalty": 1.2,
            "speculation": "auto"
        }));
        let defaults = GenerationDefaults {
            max_tokens: Some(4096),
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(20),
            repetition_penalty: Some(1.1),
            speculation: Some("none".to_owned()),
            ..GenerationDefaults::default()
        };

        assert_eq!(resolved_max_tokens(&req, &defaults, 1024), 64);
        let sampling = build_sampling_params(&req, &defaults).unwrap();
        assert!(sampling.temperature.abs() < f32::EPSILON);
        assert!((sampling.top_p - 1.0).abs() < f32::EPSILON);
        assert_eq!(sampling.top_k, Some(5));
        assert_eq!(sampling.repetition_penalty, Some(1.2));
        assert_eq!(sampling.speculation, higgs_models::Speculation::Auto);
    }

    #[test]
    fn session_usage_reports_reused_prefix_as_cached() {
        // A continued turn: 1000-token prompt, only the 120-token suffix was
        // prefilled, so 880 tokens came from the retained KV cache.
        let continued = higgs_engine::simple::SessionGeneration {
            text: String::new(),
            completion_tokens: 30,
            finish_reason: "length".to_owned(),
            prompt_tokens: 1000,
            prefilled_tokens: 120,
            continued: true,
            outcome: higgs_engine::simple::SessionOutcome::Continued,
        };
        let usage = session_usage(&continued);
        assert_eq!(usage.prompt_tokens, 1000);
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens),
            Some(880)
        );

        // A cold prefill re-ran the whole prompt: no cached tokens reported.
        let cold = higgs_engine::simple::SessionGeneration {
            text: String::new(),
            completion_tokens: 30,
            finish_reason: "stop".to_owned(),
            prompt_tokens: 1000,
            prefilled_tokens: 1000,
            continued: false,
            outcome: higgs_engine::simple::SessionOutcome::ExactBootstrap,
        };
        assert!(session_usage(&cold).prompt_tokens_details.is_none());

        let pflash = higgs_engine::simple::SessionGeneration {
            outcome: higgs_engine::simple::SessionOutcome::PFlashBootstrap,
            ..cold
        };
        assert!(session_usage(&pflash).prompt_tokens_details.is_none());
    }

    #[test]
    fn session_response_preserves_length_finish_reason() {
        let output = higgs_engine::simple::SessionGeneration {
            text: "partial".to_owned(),
            completion_tokens: 1,
            finish_reason: "length".to_owned(),
            prompt_tokens: 3,
            prefilled_tokens: 3,
            continued: false,
            outcome: higgs_engine::simple::SessionOutcome::ExactBootstrap,
        };

        let response = build_session_response(
            "model", "request", output, None, false, false, false, false, None, None,
        )
        .unwrap();
        assert_eq!(response.choices[0].finish_reason, "length");
    }

    #[test]
    fn session_response_marks_only_confirmed_lease() {
        let output = higgs_engine::simple::SessionGeneration {
            text: String::new(),
            completion_tokens: 0,
            finish_reason: "length".to_owned(),
            prompt_tokens: 3,
            prefilled_tokens: 2,
            continued: false,
            outcome: higgs_engine::simple::SessionOutcome::ExactBootstrap,
        };
        let response = build_session_response(
            "model", "request", output, None, false, false, true, false, None, None,
        )
        .unwrap();
        assert_eq!(response.usage.higgs_session_lease_active, Some(1));
    }

    #[tokio::test]
    async fn streaming_required_worker_rejection_returns_http_409_before_sse() {
        let engine = Arc::new(Engine::test_stub("raw-accept-worker-reject"));
        let request = chat_request(serde_json::json!({
            "stream": true,
            "session_id": 42,
            "session_cache_policy": "require_continuation"
        }));

        let error = match chat_completions_stream(
            streaming_test_state(),
            request,
            engine,
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await
        {
            Ok(_) => panic!("required miss opened an SSE response"),
            Err(error) => error,
        };
        let response = error.into_response();
        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "retained_session_unavailable");
    }

    #[tokio::test]
    async fn streaming_required_materialization_failure_errors_before_sse() {
        let request = chat_request(serde_json::json!({
            "stream": true,
            "session_id": 42,
            "max_tokens": 0,
            "session_cache_policy": "require_continuation"
        }));

        let engine = Arc::new(Engine::test_stub("zero-prefix-materialization-fail"));
        engine.test_retain_session_tokens(42, Vec::new());

        let error = match chat_completions_stream(
            streaming_test_state(),
            request,
            engine,
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await
        {
            Ok(_) => panic!("materialization failure opened an SSE response"),
            Err(error) => error,
        };
        assert_eq!(
            error.into_response().status(),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[tokio::test]
    async fn blocking_required_post_admission_miss_returns_http_409() {
        let request = chat_request(serde_json::json!({
            "session_id": 42,
            "max_tokens": 0,
            "session_cache_policy": "require_continuation"
        }));

        let error = chat_completions_non_streaming(
            streaming_test_state(),
            request,
            Arc::new(Engine::test_stub(
                "blocking-required-post-admission-evicted",
            )),
            GenerationDefaults::default(),
        )
        .await
        .unwrap_err();
        let response = error.into_response();
        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "retained_session_unavailable");
    }

    #[tokio::test]
    async fn capacity_interrupted_stream_emits_exact_terminal_event_then_done() {
        let model = "capacity-interrupted";
        let engine = Arc::new(Engine::test_stub(model));
        let request = chat_request(serde_json::json!({
            "model": model,
            "stream": true,
            "max_tokens": 8
        }));

        let stream = chat_completions_stream(
            streaming_test_state(),
            request,
            Arc::clone(&engine),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await
        .expect("admitted stream must start");
        let response = axum::response::sse::Sse::new(stream).into_response();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(body.to_vec()).unwrap();

        // Exact frozen v1 terminal payload.
        assert!(
            body.contains(r#""type":"higgs_capacity_interrupted""#),
            "{body}"
        );
        assert!(body.contains(r#""code":"capacity_interrupted""#), "{body}");
        assert!(body.contains(r#""bootId":"boot-route-test""#), "{body}");
        assert!(body.contains(r#""generation":4"#), "{body}");
        assert!(body.contains(r#""partialOutputTokens":0"#), "{body}");
        // The typed event precedes the normal terminator.
        let event_at = body.find("higgs_capacity_interrupted").expect("event");
        let done_at = body.find("[DONE]").expect("[DONE]");
        assert!(event_at < done_at, "{body}");
    }

    #[tokio::test]
    async fn auto_streaming_capacity_discards_every_buffered_remainder() {
        let model = "required-stream-capacity-partial-call";
        let request = chat_request(serde_json::json!({
            "model": model,
            "stream": true,
            "max_tokens": 8
        }));
        let stream = chat_completions_stream(
            streaming_test_state(),
            request,
            Arc::new(Engine::test_stub(model)),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await
        .expect("admitted Auto stream must start");
        let response = axum::response::sse::Sse::new(stream).into_response();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(body.to_vec()).unwrap();

        let terminal = body.find("higgs_capacity_interrupted").expect("terminal");
        let tail: String = body[terminal..]
            .split("\n\n")
            .skip(1)
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(tail.trim(), "data: [DONE]", "{body}");
    }

    #[tokio::test]
    async fn required_one_token_prefill_stream_accepts_zero_prefix_or_returns_409() {
        let request = || {
            chat_request(serde_json::json!({
                "stream": true,
                "stream_options": {"include_usage": true},
                "session_id": 42,
                "max_tokens": 0,
                "session_cache_policy": "require_continuation"
            }))
        };

        let accepted_engine = Arc::new(Engine::test_stub("zero-prefix-accept"));
        accepted_engine.test_retain_session_tokens(42, Vec::new());
        let accepted = chat_completions_stream(
            streaming_test_state(),
            request(),
            Arc::clone(&accepted_engine),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await
        .expect("explicit zero-prefix retention should be accepted");
        let response = axum::response::sse::Sse::new(accepted).into_response();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("\"prompt_tokens\":1"));
        assert!(body.contains("\"completion_tokens\":0"));
        assert!(body.contains("\"total_tokens\":1"));
        assert!(body.contains("\"finish_reason\":\"length\""));
        assert!(body.contains("data: [DONE]"));
        assert_eq!(
            accepted_engine.route_test_mutations(),
            1,
            "accepted zero-prefix prefill did not publish exactly once"
        );

        let rejected_engine = Arc::new(Engine::test_stub("zero-prefix-evicted"));
        let error = match chat_completions_stream(
            streaming_test_state(),
            request(),
            Arc::clone(&rejected_engine),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await
        {
            Ok(_) => panic!("evicted zero-prefix retention opened an SSE response"),
            Err(error) => error,
        };
        let response = error.into_response();
        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "retained_session_unavailable");
        assert_eq!(rejected_engine.route_test_mutations(), 0);
    }

    #[tokio::test]
    async fn axum_session_extensions_preserve_status_usage_and_mutation_order() {
        let (accepted_app, accepted_engine) = axum_session_test_app("zero-prefix-accept");
        let seed = accepted_app
            .clone()
            .oneshot(axum_chat_request(
                "zero-prefix-accept",
                serde_json::json!({
                    "session_id": 42,
                    "max_tokens": 0,
                    "max_prompt_tokens": 1,
                    "session_cache_policy": "best_effort"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(seed.status(), axum::http::StatusCode::OK);
        let seed_body = seed.into_body().collect().await.unwrap().to_bytes();
        let seed_body: serde_json::Value = serde_json::from_slice(&seed_body).unwrap();
        assert_eq!(seed_body["choices"].as_array().unwrap().len(), 1);
        assert_eq!(seed_body["choices"][0]["message"]["content"], "");
        assert!(
            seed_body["choices"][0]["message"]
                .get("tool_calls")
                .is_none()
        );
        assert_eq!(seed_body["choices"][0]["finish_reason"], "length");
        assert_eq!(seed_body["usage"]["prompt_tokens"], 1);
        assert_eq!(seed_body["usage"]["completion_tokens"], 0);
        assert_eq!(seed_body["usage"]["total_tokens"], 1);
        assert!(seed_body["usage"].get("prompt_tokens_details").is_none());
        assert!(
            seed_body["usage"]
                .get("higgs_session_lease_active")
                .is_none()
        );

        let leased = accepted_app
            .clone()
            .oneshot(axum_chat_request(
                "zero-prefix-accept",
                serde_json::json!({
                    "session_id": 43,
                    "max_tokens": 0,
                    "max_prompt_tokens": 1,
                    "session_lease": {"session_id": 42, "ttl_seconds": 300},
                    "session_cache_policy": "best_effort"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(leased.status(), axum::http::StatusCode::OK);
        let leased_body = leased.into_body().collect().await.unwrap().to_bytes();
        let leased_body: serde_json::Value = serde_json::from_slice(&leased_body).unwrap();
        assert_eq!(leased_body["usage"]["prompt_tokens"], 1);
        assert_eq!(leased_body["usage"]["completion_tokens"], 0);
        assert_eq!(leased_body["usage"]["total_tokens"], 1);
        assert!(leased_body["usage"].get("prompt_tokens_details").is_none());
        assert_eq!(leased_body["usage"]["higgs_session_lease_active"], 1);

        let continued = accepted_app
            .clone()
            .oneshot(axum_chat_request(
                "zero-prefix-accept",
                serde_json::json!({
                    "stream": true,
                    "stream_options": {"include_usage": true},
                    "session_id": 42,
                    "max_tokens": 0,
                    "max_prompt_tokens": 1,
                    "session_cache_policy": "require_continuation"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(continued.status(), axum::http::StatusCode::OK);
        let continued_body = continued.into_body().collect().await.unwrap().to_bytes();
        let (continued_events, done) =
            axum_sse_events(&String::from_utf8(continued_body.to_vec()).unwrap());
        assert_eq!(done, 1);
        let usage: Vec<_> = continued_events
            .iter()
            .filter_map(|event| event.get("usage"))
            .collect();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0]["prompt_tokens"], 1);
        assert_eq!(usage[0]["completion_tokens"], 0);
        assert_eq!(usage[0]["total_tokens"], 1);
        assert_eq!(usage[0]["prompt_tokens_details"]["cached_tokens"], 1);
        assert!(usage[0].get("higgs_session_lease_active").is_none());

        let singular = accepted_app
            .clone()
            .oneshot(axum_chat_request(
                "zero-prefix-accept",
                serde_json::json!({
                    "session_id": 44,
                    "drop_session_id": 42,
                    "max_tokens": 0,
                    "max_prompt_tokens": 1,
                    "session_cache_policy": "best_effort"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(singular.status(), axum::http::StatusCode::OK);
        let plural = accepted_app
            .oneshot(axum_chat_request(
                "zero-prefix-accept",
                serde_json::json!({
                    "session_id": 45,
                    "drop_session_id": 43,
                    "drop_session_ids": [44, 43, 44],
                    "max_tokens": 0,
                    "max_prompt_tokens": 1,
                    "session_cache_policy": "best_effort"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(plural.status(), axum::http::StatusCode::OK);
        assert_eq!(
            accepted_engine.route_test_mutation_sequence(),
            [
                "retain:42",
                "lease:42:300",
                "retain:43",
                "continue:42",
                "drop:42",
                "retain:44",
                "drop:43",
                "drop:44",
                "retain:45"
            ]
        );
        assert_eq!(accepted_engine.route_test_retained_sessions(), [45]);

        let (limited_app, limited_engine) = axum_session_test_app("prompt-limit-mutation-spy");
        let limited = limited_app
            .oneshot(axum_chat_request(
                "prompt-limit-mutation-spy",
                serde_json::json!({
                    "session_id": 42,
                    "max_tokens": 0,
                    "max_prompt_tokens": 2,
                    "drop_session_ids": [7, 7],
                    "session_lease": {"session_id": 8, "ttl_seconds": 300},
                    "session_cache_policy": "best_effort"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(limited.status(), axum::http::StatusCode::BAD_REQUEST);
        let limited_body = limited.into_body().collect().await.unwrap().to_bytes();
        let limited_body: serde_json::Value = serde_json::from_slice(&limited_body).unwrap();
        assert_eq!(limited_body["error"]["code"], "context_length_exceeded");
        assert_eq!(limited_engine.route_test_mutations(), 0);

        let (ttl_app, ttl_engine) = axum_session_test_app("prompt-limit-mutation-spy");
        let invalid_ttl = ttl_app
            .oneshot(axum_chat_request(
                "prompt-limit-mutation-spy",
                serde_json::json!({
                    "session_id": 42,
                    "max_tokens": 0,
                    "max_prompt_tokens": 3,
                    "drop_session_ids": [7, 8],
                    "session_lease": {"session_id": 8, "ttl_seconds": 301}
                }),
            ))
            .await
            .unwrap();
        assert_eq!(invalid_ttl.status(), axum::http::StatusCode::BAD_REQUEST);
        let invalid_ttl_body = invalid_ttl.into_body().collect().await.unwrap().to_bytes();
        let invalid_ttl_body: serde_json::Value =
            serde_json::from_slice(&invalid_ttl_body).unwrap();
        assert_eq!(invalid_ttl_body["error"]["type"], "invalid_request_error");
        assert_eq!(ttl_engine.route_test_mutations(), 0);

        let (missing_app, missing_engine) = axum_session_test_app("zero-prefix-evicted");
        let missing = missing_app
            .oneshot(axum_chat_request(
                "zero-prefix-evicted",
                serde_json::json!({
                    "stream": true,
                    "session_id": 42,
                    "max_tokens": 0,
                    "session_cache_policy": "require_continuation"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(missing.status(), axum::http::StatusCode::CONFLICT);
        let missing_body = missing.into_body().collect().await.unwrap().to_bytes();
        let missing_body: serde_json::Value = serde_json::from_slice(&missing_body).unwrap();
        assert_eq!(
            missing_body["error"]["code"],
            "retained_session_unavailable"
        );
        assert_eq!(missing_engine.route_test_mutations(), 0);
    }

    #[test]
    fn session_continuation_allows_plain_session_id() {
        assert_eq!(
            session_continuation_id(Some(42), false, false, None, false, false),
            Some(42)
        );
    }

    #[test]
    fn session_continuation_rejects_unsupported_request_shapes() {
        assert_eq!(
            session_continuation_id(Some(42), true, false, None, false, false),
            None
        );
        assert_eq!(
            session_continuation_id(Some(42), false, true, None, false, false),
            None
        );
        assert_eq!(
            session_continuation_id(None, false, false, None, false, false),
            None
        );
        assert_eq!(
            session_continuation_id(Some(42), false, false, None, true, false),
            None
        );
        assert_eq!(
            session_continuation_id(Some(42), false, false, None, false, true),
            None
        );
    }

    #[test]
    fn session_continuation_checkpoint_id_takes_precedence() {
        assert_eq!(
            session_continuation_id(Some(42), false, false, Some("checkpoint-a"), false, false),
            None
        );
    }

    #[test]
    fn retained_session_drop_ids_are_deduplicated_and_sorted() {
        assert_eq!(retained_session_drop_ids(None, None), Vec::<u64>::new());
        assert_eq!(
            retained_session_drop_ids(Some(9), Some(&[3, 9, 1, 3])),
            vec![1, 3, 9]
        );
    }

    #[test]
    fn prompt_limit_uses_authoritative_rendered_token_count() {
        assert!(validate_prompt_limit(None, 3).is_ok());
        assert!(validate_prompt_limit(Some(3), 3).is_ok());
        let error = validate_prompt_limit(Some(2), 3).unwrap_err();
        assert!(matches!(
            error,
            ServerError::ContextLengthExceeded {
                prompt_tokens: 3,
                max_prompt_tokens: 2
            }
        ));
    }

    #[test]
    fn session_prefill_mode_requires_session_and_zero_budget() {
        use higgs_engine::chat_template::ChatPromptMode;

        assert_eq!(
            chat_prompt_mode(Some(42), 0),
            ChatPromptMode::SessionPrefill
        );
        assert_eq!(chat_prompt_mode(None, 0), ChatPromptMode::Generation);
        assert_eq!(chat_prompt_mode(Some(42), 1), ChatPromptMode::Generation);
    }

    #[tokio::test]
    async fn axum_session_prefill_mode_is_authoritative_before_mutation() {
        let (app, engine) = axum_session_test_app("session-prefill-render-spy");

        let accepted = app
            .clone()
            .oneshot(axum_chat_request(
                "session-prefill-render-spy",
                serde_json::json!({
                    "session_id": 42,
                    "max_tokens": 0,
                    "max_prompt_tokens": 1,
                    "session_cache_policy": "best_effort"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(accepted.status(), axum::http::StatusCode::OK);

        let streaming_accepted = app
            .clone()
            .oneshot(axum_chat_request(
                "session-prefill-render-spy",
                serde_json::json!({
                    "stream": true,
                    "stream_options": {"include_usage": true},
                    "session_id": 45,
                    "max_tokens": 0,
                    "max_prompt_tokens": 1,
                    "session_cache_policy": "best_effort"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(streaming_accepted.status(), axum::http::StatusCode::OK);
        let body = streaming_accepted
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let (events, done) = axum_sse_events(&String::from_utf8(body.to_vec()).unwrap());
        assert_eq!(done, 1);
        assert!(events.iter().any(|event| {
            event["choices"][0]["finish_reason"] == serde_json::Value::String("length".to_owned())
        }));
        assert_eq!(engine.route_test_mutations(), 2);

        let streaming_rejected = app
            .clone()
            .oneshot(axum_chat_request(
                "session-prefill-render-spy",
                serde_json::json!({
                    "stream": true,
                    "session_id": 44,
                    "max_tokens": 1,
                    "max_prompt_tokens": 1,
                    "drop_session_ids": [42],
                    "session_cache_policy": "best_effort"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(
            streaming_rejected.status(),
            axum::http::StatusCode::BAD_REQUEST
        );
        let body = streaming_rejected
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "context_length_exceeded");
        assert_eq!(engine.route_test_mutations(), 2);

        let rejected = app
            .oneshot(axum_chat_request(
                "session-prefill-render-spy",
                serde_json::json!({
                    "session_id": 43,
                    "max_tokens": 1,
                    "max_prompt_tokens": 1,
                    "drop_session_ids": [42],
                    "session_cache_policy": "best_effort"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(rejected.status(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(engine.route_test_mutations(), 2);
    }

    #[tokio::test]
    async fn prompt_limit_rejects_before_blocking_and_streaming_session_mutations() {
        let engine = Arc::new(Engine::test_stub("prompt-limit-mutation-spy"));
        let request = || {
            chat_request(serde_json::json!({
                "session_id": 42,
                "max_prompt_tokens": 2,
                "drop_session_id": 7,
                "session_lease": {"session_id": 8, "ttl_seconds": 60}
            }))
        };

        let blocking = chat_completions_non_streaming(
            streaming_test_state(),
            request(),
            Arc::clone(&engine),
            GenerationDefaults::default(),
        )
        .await;
        assert!(matches!(
            blocking,
            Err(ServerError::ContextLengthExceeded {
                prompt_tokens: 3,
                max_prompt_tokens: 2
            })
        ));
        assert_eq!(engine.route_test_mutations(), 0);

        let streaming = chat_completions_stream(
            streaming_test_state(),
            request(),
            Arc::clone(&engine),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await;
        assert!(matches!(
            streaming,
            Err(ServerError::ContextLengthExceeded {
                prompt_tokens: 3,
                max_prompt_tokens: 2
            })
        ));
        assert_eq!(engine.route_test_mutations(), 0);
    }

    #[test]
    fn session_lease_ttl_is_bounded_to_wire_contract() {
        assert!(validate_session_lease_ttl(None).is_ok());
        assert!(validate_session_lease_ttl(Some(1)).is_ok());
        assert!(validate_session_lease_ttl(Some(300)).is_ok());
        assert!(validate_session_lease_ttl(Some(0)).is_err());
        assert!(validate_session_lease_ttl(Some(301)).is_err());
    }

    fn simple_message(role: &str, content: Option<&str>) -> ChatCompletionMessage {
        ChatCompletionMessage {
            role: role.to_owned(),
            content: content.map(|s| MessageContent::Text(s.to_owned())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            r#type: "function".to_owned(),
            function: ToolCallFunction {
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            },
        }
    }

    fn tool_message(role: &str, calls: Vec<ToolCall>) -> ChatCompletionMessage {
        ChatCompletionMessage {
            role: role.to_owned(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(calls),
            tool_call_id: None,
        }
    }

    #[test]
    fn test_convert_messages() {
        let msgs = vec![
            simple_message("user", Some("Hello")),
            simple_message("assistant", None),
        ];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted.first().map(|m| m.role.as_str()), Some("user"));
        assert_eq!(converted.first().map(|m| m.content.as_str()), Some("Hello"));
        assert_eq!(converted.get(1).map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_generate_request_id_format() {
        let id = generate_request_id();
        assert!(id.starts_with("chatcmpl-"));
        assert!(id.len() > "chatcmpl-".len());
    }

    #[test]
    fn test_convert_messages_with_tool_calls() {
        let msgs = vec![tool_message(
            "assistant",
            vec![tool_call("call_1", "get_weather", r#"{"city":"NYC"}"#)],
        )];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        let calls = converted
            .first()
            .and_then(|m| m.tool_calls.as_ref())
            .unwrap();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_convert_messages_empty_list() {
        let result = convert_messages(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_convert_messages_with_null_content() {
        let msgs = vec![simple_message("assistant", None)];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted.first().map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_convert_messages_with_tool_calls_complex_arguments() {
        let msgs = vec![tool_message(
            "assistant",
            vec![
                tool_call(
                    "call_1",
                    "search",
                    r#"{"query":"rust programming","filters":{"language":"en","year":2024}}"#,
                ),
                tool_call("call_2", "calculate", r#"{"expression":"2+2"}"#),
            ],
        )];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        let calls = converted
            .first()
            .and_then(|m| m.tool_calls.as_ref())
            .unwrap();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_generate_request_id_uniqueness() {
        let mut ids = std::collections::HashSet::new();
        for _ in 0..100 {
            let id = generate_request_id();
            assert!(ids.insert(id), "duplicate request ID generated");
        }
        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn test_current_unix_timestamp_reasonable_value() {
        let ts = current_unix_timestamp();
        assert!(ts > 1_700_000_000, "timestamp too old: {ts}");
    }

    fn image_item() -> MediaItem {
        MediaItem {
            position: 0,
            message_index: 0,
            bytes: vec![1, 2, 3],
            media_type: "image/png".to_owned(),
            detail: higgs_models::vision::ImageDetail::Auto,
            max_dims: None,
        }
    }

    fn parts_message(role: &str, parts: Vec<ContentPart>) -> ChatCompletionMessage {
        ChatCompletionMessage {
            role: role.to_owned(),
            content: Some(MessageContent::Parts(parts)),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn text_part(text: &str) -> ContentPart {
        ContentPart::Text {
            text: text.to_owned(),
        }
    }

    fn image_part() -> ContentPart {
        ContentPart::ImageUrl {
            image_url: crate::types::openai::ImageUrl {
                url: "data:image/png;base64,AAAA".to_owned(),
                detail: Some(higgs_models::vision::ImageDetail::Auto),
            },
        }
    }

    #[test]
    fn test_render_markers_splices_marker_at_image_positions() {
        let msgs = vec![parts_message(
            "user",
            vec![
                text_part("Look at "),
                image_part(),
                text_part(" then "),
                image_part(),
                text_part("."),
            ],
        )];
        let rendered = render_markers(&msgs, Some("<image>"));
        let content = rendered.first().and_then(|m| m.content.as_ref()).unwrap();
        let MessageContent::Text(text) = content else {
            panic!("expected rendered text content");
        };
        assert_eq!(text.as_str(), "Look at <image> then <image>.");
        assert_eq!(rendered.first().map(|m| m.role.as_str()), Some("user"));
    }

    #[test]
    fn test_render_markers_defaults_to_image_marker() {
        let msgs = vec![parts_message(
            "user",
            vec![text_part("A "), image_part(), text_part(" B")],
        )];
        let rendered = render_markers(&msgs, None);
        let content = rendered.first().and_then(|m| m.content.as_ref()).unwrap();
        let MessageContent::Text(text) = content else {
            panic!("expected rendered text content");
        };
        assert_eq!(text.as_str(), "A <image> B");
    }

    #[test]
    fn test_render_markers_passes_plain_text_messages_through() {
        let msgs = vec![simple_message("user", Some("plain text"))];
        let rendered = render_markers(&msgs, Some("<image>"));
        let content = rendered.first().and_then(|m| m.content.as_ref()).unwrap();
        let MessageContent::Text(text) = content else {
            panic!("expected text content");
        };
        assert_eq!(text.as_str(), "plain text");
    }

    #[test]
    fn test_check_vision_capability_rejects_images_on_text_model() {
        let media = vec![image_item()];
        let err = check_vision_capability(&media, false, "text-model").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("does not support vision"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("text-model"), "model name missing: {msg}");
    }

    #[test]
    fn test_check_vision_capability_accepts_images_on_vlm() {
        let media = vec![image_item()];
        assert!(check_vision_capability(&media, true, "vlm").is_ok());
    }

    #[test]
    fn test_check_vision_capability_accepts_no_images_on_text_model() {
        assert!(check_vision_capability(&[], false, "text-model").is_ok());
    }

    // -- Route-level vision gate (through the full axum router) --

    /// A valid 1x1 red PNG (passes byte-size and dimension checks).
    const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    fn image_chat_body(stream: bool) -> serde_json::Value {
        serde_json::json!({
            "model": "stub-model",
            "stream": stream,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{TINY_PNG_B64}")}}
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

    #[tokio::test]
    async fn test_chat_completions_image_on_text_model_returns_400() {
        let app = crate::build_router(
            crate::state::test_state_with_stub_engine("stub-model"),
            300.0,
            None,
            0,
            1024 * 1024,
            None,
        );
        let (status, msg) =
            error_message(post_json(app, "/v1/chat/completions", image_chat_body(false)).await)
                .await;
        assert_eq!(status, 400, "expected 400, got body: {msg}");
        assert!(
            msg.contains("does not support vision"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("stub-model"), "model name missing: {msg}");
    }

    #[tokio::test]
    async fn test_chat_completions_stream_image_on_text_model_returns_400() {
        let app = crate::build_router(
            crate::state::test_state_with_stub_engine("stub-model"),
            300.0,
            None,
            0,
            1024 * 1024,
            None,
        );
        let (status, msg) =
            error_message(post_json(app, "/v1/chat/completions", image_chat_body(true)).await)
                .await;
        assert_eq!(status, 400, "expected 400, got body: {msg}");
        assert!(
            msg.contains("does not support vision"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn capacity_admission_rejects_both_chat_variants_before_worker_mutation() {
        let model = "prompt-limit-mutation-spy";
        let (state, engine) = crate::capacity::rejecting_route_test_state(model);
        let initial_memory_revision = state.capacity.admission_test_memory().1;
        let request = |stream| {
            serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5120,
                "drop_session_id": 7,
                "session_lease": {"session_id": 8, "ttl_seconds": 60},
                "stream": stream
            }))
            .unwrap()
        };

        let blocking = chat_completions_non_streaming(
            Arc::clone(&state),
            request(false),
            Arc::clone(&engine),
            GenerationDefaults::default(),
        )
        .await;
        assert!(matches!(blocking, Err(ServerError::CapacityExceeded(_))));
        let streaming = chat_completions_stream(
            Arc::clone(&state),
            request(true),
            Arc::clone(&engine),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await;
        assert!(matches!(streaming, Err(ServerError::CapacityExceeded(_))));
        assert!(engine.route_test_mutation_sequence().is_empty());
        assert_eq!(state.capacity.active_reservation_count(model), 0);
        assert_eq!(
            state.capacity.admission_test_memory().1,
            initial_memory_revision
        );
    }

    #[tokio::test]
    async fn v2_retention_rejects_stale_and_oversized_before_worker_or_cache_mutation() {
        let model = "raw-accept-worker-reject";
        let (state, engine) = crate::capacity::retained_contract_route_test_state(
            model,
            4 * 1024 * 1024 * 1024,
            128 * 1024,
        );
        engine.test_retain_session_tokens(42, vec![7; 8]);
        let revision = state
            .capacity
            .fast_session_contract(model)
            .unwrap()
            .contract_revision()
            .to_owned();
        let request = |stream, contract_revision: &str| {
            serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4096,
                "stream": stream,
                "retention": {
                    "mode": "required",
                    "sessionId": 42,
                    "epoch": 7,
                    "contractRevision": contract_revision
                }
            }))
            .unwrap()
        };

        engine.test_set_chat_prompt_tokens(vec![7; 30_000]);
        let blocking = chat_completions_non_streaming(
            Arc::clone(&state),
            request(false, &revision),
            Arc::clone(&engine),
            GenerationDefaults::default(),
        )
        .await;
        let Err(ServerError::RetentionCompactionRequired(blocking_context)) = blocking else {
            panic!("blocking route must return the typed compact-required error");
        };
        assert_eq!(blocking_context.contract_revision, revision);
        assert_eq!(
            (blocking_context.session_id, blocking_context.epoch),
            (42, 7)
        );

        engine.test_set_chat_prompt_tokens(vec![7; 8]);
        let streaming = chat_completions_stream(
            Arc::clone(&state),
            request(true, "stale:revision"),
            Arc::clone(&engine),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await;
        let Err(ServerError::StaleRetentionContract(streaming_context)) = streaming else {
            panic!("streaming route must return the typed stale-contract error");
        };
        assert_eq!(streaming_context.contract_revision, "stale:revision");
        assert_eq!(
            (streaming_context.session_id, streaming_context.epoch),
            (42, 7)
        );
        assert_eq!(engine.route_test_retained_sessions(), [42]);
        assert!(engine.route_test_mutation_sequence().is_empty());
        assert_eq!(state.capacity.active_reservation_count(model), 0);
    }

    #[test]
    fn seed_claim_binds_revision_session_and_epoch_and_rejects_self_drop() {
        let model = "seed-binding";
        let (state, engine) = crate::capacity::retained_contract_route_test_state(
            model,
            4 * 1024 * 1024 * 1024,
            128 * 1024,
        );
        let revision = state
            .capacity
            .fast_session_contract(model)
            .unwrap()
            .contract_revision()
            .to_owned();
        let mut seed: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":model,"messages":[],"retention":{"mode":"seed","sessionId":42,"epoch":7,"contractRevision":revision}
        })).unwrap();
        let retention = apply_required_retention(&mut seed).unwrap().unwrap();
        assert_eq!(
            seed.session_cache_policy,
            Some(SessionCachePolicy::BestEffort)
        );
        let claim =
            validate_required_retention(&state, &engine, model, &[1, 2], 1, Some(&retention), &[])
                .unwrap()
                .unwrap();
        engine.test_retain_session_tokens(42, vec![1, 2]);
        assert!(claim.publish());

        let mut required = retention.clone();
        required.mode = crate::types::openai::RetentionMode::Required;
        assert!(
            validate_required_retention(
                &state,
                &engine,
                model,
                &[1, 2, 3],
                1,
                Some(&required),
                &[]
            )
            .is_ok()
        );
        required.epoch = 8;
        assert!(matches!(
            validate_required_retention(
                &state,
                &engine,
                model,
                &[1, 2, 3],
                1,
                Some(&required),
                &[]
            ),
            Err(ServerError::RequiredRetentionUnavailable(_))
        ));

        seed.drop_session_id = Some(42);
        assert!(apply_required_retention(&mut seed).is_err());
        assert_eq!(engine.route_test_retained_sessions(), [42]);
    }

    #[test]
    fn rotated_seed_atomically_retires_old_binding_before_claiming_slot() {
        let model = "rotated-seed";
        let (state, _) = crate::capacity::retained_contract_route_test_state(
            model,
            4 * 1024 * 1024 * 1024,
            128 * 1024,
        );
        assert!(state.claim_retention_seed(model, 41, "old", 6));
        assert!(state.publish_retention_seed(model, 41));

        assert!(state.claim_retention_seed_replacing(model, 42, "new", 7, &[41]));
        assert!(!state.retention_binding_matches(model, 41, "old", 6));
        assert!(!state.retention_binding_matches(model, 42, "new", 7));
        assert!(state.publish_retention_seed(model, 42));
        assert!(state.retention_binding_matches(model, 42, "new", 7));
    }

    #[tokio::test]
    async fn retained_suffix_admission_charges_uncached_suffix_and_revalidates_at_worker() {
        let model = "zero-prefix-accept";
        let (state, engine) = crate::capacity::suffix_charging_route_test_state(model);
        engine.test_set_chat_prompt_tokens(vec![7; 3_000]);
        engine.test_retain_session_tokens(42, vec![7; 2_900]);
        let request = |extra: serde_json::Value| {
            let mut request = serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}]
            });
            if let (Some(dst), Some(src)) = (request.as_object_mut(), extra.as_object()) {
                dst.extend(src.clone());
            }
            serde_json::from_value::<ChatCompletionRequest>(request).unwrap()
        };

        // Best-effort and required continuation use the same full context bound.
        let best_effort = chat_completions_non_streaming(
            Arc::clone(&state),
            request(serde_json::json!({"session_id": 42, "max_tokens": 16})),
            Arc::clone(&engine),
            GenerationDefaults::default(),
        )
        .await;
        assert!(
            best_effort.is_ok(),
            "fixed context permits full prompt: {best_effort:?}"
        );

        // Required continuation still verifies its retained prefix at acceptance.
        let admitted = chat_completions_non_streaming(
            Arc::clone(&state),
            request(serde_json::json!({
                "session_id": 42,
                "session_cache_policy": "require_continuation",
                "max_tokens": 16
            })),
            Arc::clone(&engine),
            GenerationDefaults::default(),
        )
        .await;
        assert!(
            admitted.is_ok(),
            "fixed-context session turn must admit: {admitted:?}"
        );
        assert_eq!(
            state.capacity.active_reservation_count(model),
            0,
            "worker-owned reservation must release after completion"
        );

        // A missing required session remains a typed 409 independent of context.
        let cold = chat_completions_non_streaming(
            Arc::clone(&state),
            request(serde_json::json!({
                "session_id": 43,
                "session_cache_policy": "require_continuation",
                "max_tokens": 16
            })),
            Arc::clone(&engine),
            GenerationDefaults::default(),
        )
        .await;
        assert!(
            matches!(cold, Err(ServerError::RetainedSessionUnavailable(43))),
            "missing required session must return 409: {cold:?}"
        );

        // Worker-acceptance revalidation: admission measures a 2,900-token
        // retained prefix, then the request itself drops session 42 before
        // the worker runs. The shrunk retained state must surface the
        // existing typed 409 instead of proceeding undercharged.
        let revalidated = chat_completions_non_streaming(
            Arc::clone(&state),
            request(serde_json::json!({
                "session_id": 42,
                "session_cache_policy": "require_continuation",
                "drop_session_ids": [42],
                "max_tokens": 16
            })),
            Arc::clone(&engine),
            GenerationDefaults::default(),
        )
        .await;
        assert!(
            matches!(
                revalidated,
                Err(ServerError::RetainedSessionUnavailable(42))
            ),
            "shrunk retained prefix must fail as the typed 409: {revalidated:?}"
        );
        assert_eq!(state.capacity.active_reservation_count(model), 0);
    }
    #[tokio::test]
    async fn omitted_output_uses_remaining_fixed_context_in_both_chat_paths() {
        let model = "zero-prefix-accept";
        let (state, engine) = crate::capacity::suffix_charging_route_test_state(model);
        engine.test_set_chat_prompt_tokens(vec![7; 3000]);
        let request = || {
            serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
                "model": model,
                "session_id": 42,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .unwrap()
        };
        let blocking = chat_completions_non_streaming(
            Arc::clone(&state),
            request(),
            Arc::clone(&engine),
            GenerationDefaults::default(),
        )
        .await;
        assert!(
            blocking.is_ok(),
            "omitted output must fit remaining context: {blocking:?}"
        );
        let streaming = chat_completions_stream(
            Arc::clone(&state),
            request(),
            Arc::clone(&engine),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await;
        assert!(
            streaming.is_ok(),
            "omitted output must fit remaining context"
        );
    }

    #[tokio::test]
    async fn cached_prompt_still_counts_toward_fixed_context_in_both_chat_paths() {
        let model = "zero-prefix-accept";
        let (state, engine) = crate::capacity::suffix_charging_route_test_state(model);
        engine.test_set_chat_prompt_tokens(vec![7; 3000]);
        engine.test_retain_session_tokens(42, vec![7; 2900]);
        let request = || {
            serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": 42,
                "session_cache_policy": "require_continuation",
                "max_tokens": 128
            }))
            .unwrap()
        };
        let blocking = chat_completions_non_streaming(
            Arc::clone(&state),
            request(),
            Arc::clone(&engine),
            GenerationDefaults::default(),
        )
        .await;
        assert!(matches!(blocking, Err(ServerError::CapacityExceeded(_))));
        let streaming = chat_completions_stream(
            Arc::clone(&state),
            request(),
            Arc::clone(&engine),
            GenerationDefaults::default(),
            None,
            crate::router::RoutingMethod::Direct,
        )
        .await;
        assert!(matches!(streaming, Err(ServerError::CapacityExceeded(_))));
        assert_eq!(state.capacity.active_reservation_count(model), 0);
    }

    #[tokio::test]
    async fn drop_sessions_route_reports_per_id_dropped_flags() {
        let engine_name = "drop-target";
        let engine = Arc::new(Engine::test_stub(engine_name));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[provider.stub]\nurl = \"http://127.0.0.1:1\"\n").unwrap();
        let config = crate::config::load_config_file(&path, None).unwrap();
        let router = crate::router::Router::from_config(
            &config,
            std::collections::HashMap::from([(
                engine_name.to_owned(),
                std::sync::Arc::clone(&engine),
            )]),
        )
        .unwrap();
        let state = Arc::new(crate::state::AppState::new(
            router,
            config,
            reqwest::Client::new(),
            None,
        ));
        assert!(state.claim_retention_seed(engine_name, 7, "rev", 1));
        assert!(state.publish_retention_seed(engine_name, 7));
        let app = axum::Router::new()
            .route("/v1/sessions/drop", axum::routing::post(drop_sessions))
            .with_state(Arc::clone(&state));

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/sessions/drop")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({
                            "model": engine_name,
                            "session_ids": [7, 9],
                            "session_id": 5,
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let json: serde_json::Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        let dropped = json["dropped"].as_array().unwrap();
        assert_eq!(dropped.len(), 3, "ids dedup + sort: 5, 7, 9");
        assert_eq!(dropped[0]["session_id"], 5);
        assert_eq!(dropped[1]["session_id"], 7);
        assert_eq!(dropped[2]["session_id"], 9);
        for entry in dropped {
            assert_eq!(entry["dropped"], false, "stub engine retains nothing");
        }
        assert!(
            !state.retention_binding_matches(engine_name, 7, "rev", 1),
            "the HTTP drop must retire the contract binding synchronously"
        );

        // Empty id list is a 400.
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/sessions/drop")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({ "model": engine_name }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    }
}
