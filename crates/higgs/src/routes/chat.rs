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
    metrics::{MetricsStore, RequestMetricsContext, RequestRecord},
    router::ResolvedRoute,
    state::{Engine, SharedState},
    types::openai::{
        ChatCompletionChoice, ChatCompletionDelta, ChatCompletionMessage, ChatCompletionRequest,
        ChatCompletionResponse, ChoiceLogprobs, CompletionUsage, ContentPart, MessageContent,
        SessionCachePolicy, StopSequence, TokenLogprob, ToolCall, ToolCallDelta, ToolCallFunction,
        ToolCallFunctionDelta, TopLogprob, merge_repetition_penalty,
    },
};
use higgs_models::SamplingParams;

const TOOL_RESULT_PROMPT_WARN_BYTES: usize = 16 * 1024;

fn continuation_policy(policy: Option<SessionCachePolicy>) -> SessionContinuationPolicy {
    match policy.unwrap_or(SessionCachePolicy::BestEffort) {
        SessionCachePolicy::BestEffort => SessionContinuationPolicy::BestEffort,
        SessionCachePolicy::RequireContinuation => SessionContinuationPolicy::RequireContinuation,
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
    req: ChatCompletionRequest,
    engine: Arc<Engine>,
    generation_defaults: GenerationDefaults,
) -> Result<ChatCompletionResponse, ServerError> {
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
    let tools = req.tools.as_deref().filter(|t| !t.is_empty());
    let thinking_enabled = crate::reasoning::effective_thinking_enabled(
        engine.enable_thinking(),
        &[engine.model_name(), req.model.as_str()],
        req.reasoning.as_ref(),
        req.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.enable_thinking)
            .or(req.enable_thinking)
            .or(generation_defaults.enable_thinking),
    );

    let (prompt_tokens, pflash_policy) = engine
        .prepare_chat_prompt_with_pflash_policy(&messages, tools, thinking_enabled, prompt_mode)
        .map_err(ServerError::Engine)?;
    validate_prompt_limit(req.max_prompt_tokens, prompt_tokens.len())?;
    validate_session_lease_ttl(req.session_lease.map(|lease| lease.ttl_seconds))?;
    drop_requested_retained_sessions(
        Arc::clone(&engine),
        req.drop_session_id,
        req.drop_session_ids.as_deref(),
    )
    .await?;
    let lease_active = req
        .session_lease
        .is_some_and(|lease| engine.lease_retained_session(lease.session_id, lease.ttl_seconds));

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

    let constraint = build_constraint(req.response_format.as_ref(), &engine)?;

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
    if continuation_policy == SessionContinuationPolicy::RequireContinuation && session_id.is_none()
    {
        engine.record_required_continuation_miss();
        return Err(ServerError::RetainedSessionUnavailable(
            req.session_id.unwrap_or_default(),
        ));
    }

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
            )
        })
        .await
        .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
        .map_err(map_session_engine_error)?;

        return Ok(build_session_response(
            &req.model,
            &request_id,
            session_output,
            tools,
            has_tools,
            thinking_enabled,
            lease_active,
        ));
    } else {
        tokio::task::spawn_blocking(move || {
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
        if parsed.tool_calls.is_empty() {
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
fn build_session_response(
    model: &str,
    request_id: &str,
    output: higgs_engine::simple::SessionGeneration,
    tools: Option<&[serde_json::Value]>,
    has_tools: bool,
    thinking_enabled: bool,
    lease_active: bool,
) -> ChatCompletionResponse {
    let usage = session_usage(&output).with_session_lease_active(lease_active);
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
        if parsed.tool_calls.is_empty() {
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

    ChatCompletionResponse {
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
    }
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
    req: ChatCompletionRequest,
    engine: Arc<Engine>,
    generation_defaults: GenerationDefaults,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
) -> Result<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>, ServerError> {
    let stream_includes_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());
    // Built here (before the `async_stream::stream!` block, which captures by
    // move) so the tracker can coerce XML-format tool-call values to their
    // declared JSON types.
    let tool_schema = higgs_engine::tool_parser::ToolSchema::from_tools(req.tools.as_deref());

    if stream_includes_tools {
        tracing::debug!(
            request_model = req.model,
            tool_count = req.tools.as_ref().map_or(0, Vec::len),
            "Streaming with tool-calls enabled; will emit tool_calls deltas via StreamingToolCallTracker",
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
    );

    // Pass tools into prompt rendering so the chat template emits the
    // tool spec the model recognises. The on-the-fly
    // [`StreamingToolCallTracker`] below intercepts `<tool_call>…
    // </tool_call>` blocks the model produces and turns them into
    // structured `ToolCallDelta` SSE events.
    let prompt_tools = req
        .tools
        .as_deref()
        .and_then(|t| if t.is_empty() { None } else { Some(t) });
    let (prompt_tokens, pflash_policy) = engine
        .prepare_chat_prompt_with_pflash_policy(
            &messages,
            prompt_tools,
            thinking_enabled_stream,
            prompt_mode,
        )
        .map_err(ServerError::Engine)?;
    validate_prompt_limit(req.max_prompt_tokens, prompt_tokens.len())?;
    validate_session_lease_ttl(req.session_lease.map(|lease| lease.ttl_seconds))?;
    drop_requested_retained_sessions(
        Arc::clone(&engine),
        req.drop_session_id,
        req.drop_session_ids.as_deref(),
    )
    .await?;
    let lease_active = req
        .session_lease
        .is_some_and(|lease| engine.lease_retained_session(lease.session_id, lease.ttl_seconds));

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

    let constraint = build_constraint(req.response_format.as_ref(), &engine)?;

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
    if continuation_policy == SessionContinuationPolicy::RequireContinuation
        && stream_session_id.is_none()
    {
        engine.record_required_continuation_miss();
        return Err(ServerError::RetainedSessionUnavailable(
            request_session_id.unwrap_or_default(),
        ));
    }

    let (tx, mut rx) = tokio::sync::mpsc::channel(32);
    let (terminal_tx, terminal_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
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
            let result = worker_engine.generate_session_routed_streaming_with_thinking(
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
            );
            match &result {
                Ok(()) => {}
                Err(higgs_engine::error::EngineError::Cancelled) => {
                    tracing::debug!(sid, "Session-routed streaming cancelled by client");
                }
                Err(e) => {
                    tracing::error!(error = %e, "Session-routed generation error during streaming");
                }
            }
            let _ = terminal_tx.send(result.map_err(|error| error.to_string()));
        });
    } else {
        let worker_engine = Arc::clone(&engine);
        tokio::task::spawn_blocking(move || {
            let result = worker_engine
                .generate_streaming_with_thinking_and_pflash_policy_with_cache(
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
                );
            if let Err(ref e) = result {
                tracing::error!(error = %e, "Generation error during streaming");
            }
            let _ = terminal_tx.send(result.map_err(|error| error.to_string()));
        });
    }

    if let Some(acceptance_rx) = acceptance_rx {
        match acceptance_rx.await {
            Ok(Ok(())) => {}
            Ok(Err(session_id)) => {
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
        })
    });

    let stream = async_stream::stream! {
        let mut writer = crate::sse::ChatChunkWriter::new(&request_id, created, &model);

        // Helper to emit a chunk carrying a delta.
        macro_rules! emit_delta {
            ($delta:expr, $finish:expr, $logprobs:expr) => {
                match writer.write_delta($delta, $finish, $logprobs) {
                    Ok(json) => yield Ok(Event::default().data(json)),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            };
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
        let mut tool_tracker = higgs_engine::tool_parser::StreamingToolCallTracker::new(
            stream_includes_tools,
            tool_schema,
        );

        // Closure that turns a `ParsedToolCall` into the OpenAI streaming
        // delta shape. Index is the running zero-based position of the
        // call in this response.
        let make_tool_delta = |index: u32, parsed: &higgs_engine::tool_parser::ParsedToolCall| {
            ToolCallDelta {
                index,
                id: Some(format!("call_{index}_{}", uuid::Uuid::new_v4())),
                r#type: Some("function".to_owned()),
                function: Some(ToolCallFunctionDelta {
                    name: Some(parsed.name.clone()),
                    arguments: Some(parsed.arguments.to_string()),
                }),
            }
        };

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
                if return_progress {
                    let time_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
                    let json = writer.write_prompt_progress(p.total, p.cached, p.processed, time_ms);
                    yield Ok(Event::default().data(json));
                }
                continue;
            }
            output_token_count = output.completion_tokens;
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
                emit_delta!(&d, None, None);
            }

            // Run the visible-text portion through the tool-call tracker
            // so `<tool_call>…</tool_call>` blocks become structured
            // deltas rather than being spoken aloud as plain text.
            let tool_out = tool_tracker.process(&visible);
            let visible_is_empty = tool_out.visible.is_empty();

            // Tool-call indices count up across the whole response. Each
            // chunk that closes N tool calls covers indices
            // `[base_index .. base_index+N)` where `base_index` is the
            // total completed *before* this chunk.
            let base_index = tool_tracker
                .completed_count()
                .saturating_sub(tool_out.new_tool_calls.len());
            for (i, parsed) in tool_out.new_tool_calls.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let idx = u32::try_from(base_index + i).unwrap_or(u32::MAX);
                let d = ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![make_tool_delta(idx, parsed)]),
                };
                emit_delta!(&d, None, None);
            }

            if !tool_out.visible.is_empty() {
                let d = ChatCompletionDelta {
                    role: None,
                    content: Some(tool_out.visible),
                    reasoning_content: None,
                    tool_calls: None,
                };
                emit_delta!(&d, None, chunk_logprobs.as_ref());
            }

            if let Some(finish_reason) = output.finish_reason {
                pending_finish_reason = Some(finish_reason);
                pending_finish_logprobs = if visible_is_empty { chunk_logprobs } else { None };
            }
        }

        let terminal_error = match terminal_rx.await {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(error) => Some(format!(
                "streaming generation worker terminated unexpectedly: {error}"
            )),
        };
        if let Some(error) = terminal_error {
            tracing::error!(error = %error, "Streaming generation terminated with an error");
            if let Some(ref m) = metrics {
                if let Some(id) = metrics_id {
                    m.fail_stream(
                        id,
                        u64::from(output_token_count),
                        start.elapsed(),
                        error.clone(),
                    );
                }
            }
            yield Ok(Event::default().data(streaming_error_json(&error)));
            return;
        }

        // Flush any remaining buffered content.
        let (flush_vis, flush_reas) = reasoning_tracker.flush();
        if !flush_reas.is_empty() {
            let d = ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: Some(flush_reas),
                tool_calls: None,
            };
            emit_delta!(&d, None, None);
        }
        // Drain the tool tracker (handles unclosed `<tool_call>` tags by
        // re-emitting their buffered prefix as visible content — never
        // silently drop tokens).
        let flush_tool_out = tool_tracker.process(&flush_vis);
        let flush_base_index = tool_tracker
            .completed_count()
            .saturating_sub(flush_tool_out.new_tool_calls.len());
        for (i, parsed) in flush_tool_out.new_tool_calls.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let idx = u32::try_from(flush_base_index + i).unwrap_or(u32::MAX);
            let d = ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![make_tool_delta(idx, parsed)]),
            };
            emit_delta!(&d, None, None);
        }
        if !flush_tool_out.visible.is_empty() {
            let d = ChatCompletionDelta {
                role: None,
                content: Some(flush_tool_out.visible),
                reasoning_content: None,
                tool_calls: None,
            };
            emit_delta!(&d, None, None);
        }
        let final_tool_out = tool_tracker.flush();
        if !final_tool_out.visible.is_empty() {
            let d = ChatCompletionDelta {
                role: None,
                content: Some(final_tool_out.visible),
                reasoning_content: None,
                tool_calls: None,
            };
            emit_delta!(&d, None, None);
        }

        // Defer `finish_reason` until after the tracker has drained so we
        // know whether to report `"tool_calls"` or `"stop"`.
        if let Some(finish_reason) = pending_finish_reason {
            let effective_finish = if tool_tracker.has_tool_calls() {
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
        if include_usage {
            let usage = CompletionUsage::new(
                prompt_token_count,
                output_token_count,
                cached_prompt_tokens,
            )
            .with_session_lease_active(lease_active);
            match writer.write_usage(&usage) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize usage chunk"),
            }
        }

        if let Some(ref m) = metrics {
            if let Some(id) = metrics_id {
                m.finalize_stream(id, u64::from(output_token_count), start.elapsed());
            }
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
        | higgs_engine::error::EngineError::Cancelled) => ServerError::Engine(other),
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

/// Build a constrained generator from the request's `response_format`.
///
/// Returns `None` if no constraint is needed (text mode or absent).
fn build_constraint(
    response_format: Option<&crate::types::openai::ResponseFormat>,
    engine: &std::sync::Arc<crate::state::Engine>,
) -> Result<Option<higgs_engine::constrained::ConstrainedGenerator>, ServerError> {
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

        let response =
            build_session_response("model", "request", output, None, false, false, false);
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
        let response = build_session_response("model", "request", output, None, false, false, true);
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

        let error = match chat_completions_stream(
            streaming_test_state(),
            request,
            Arc::new(Engine::test_stub("zero-prefix-materialization-fail")),
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
}
