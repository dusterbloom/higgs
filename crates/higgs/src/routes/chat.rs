use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
};
use bytes::Bytes;
use higgs_engine::simple::{SessionPromptTraceMetrics, SessionPromptTraceOutcome};
use tokio_stream::Stream;

use crate::{
    config::{ApiFormat, GenerationDefaults},
    error::ServerError,
    metrics::{MetricsStore, RequestRecord},
    router::ResolvedRoute,
    state::{Engine, SharedState},
    types::openai::{
        ChatCompletionChoice, ChatCompletionDelta, ChatCompletionMessage, ChatCompletionRequest,
        ChatCompletionResponse, ChoiceLogprobs, CompletionUsage, MessageContent, StopSequence,
        TokenLogprob, ToolCall, ToolCallDelta, ToolCallFunction, ToolCallFunctionDelta, TopLogprob,
        merge_repetition_penalty,
    },
};
use higgs_models::SamplingParams;

const TOOL_RESULT_PROMPT_WARN_BYTES: usize = 16 * 1024;

#[allow(clippy::too_many_lines)]
pub async fn chat_completions(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    let mut req: ChatCompletionRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;

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
                        model: response.model.clone(),
                        provider: "higgs".to_owned(),
                        routing_method: routing_method.into(),
                        status: 200,
                        duration: start.elapsed(),
                        input_tokens: u64::from(response.usage.prompt_tokens),
                        output_tokens: u64::from(response.usage.completion_tokens),
                        error_body: None,
                    });
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
                            model: metrics_model.clone(),
                            provider: provider_name.clone(),
                            routing_method: routing_method.into(),
                            status: result.as_ref().map_or(502, |resp| resp.status().as_u16()),
                            duration: start.elapsed(),
                            input_tokens: 0,
                            output_tokens: 0,
                            error_body: None,
                        });
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
                                model: metrics_model.clone(),
                                provider: provider_name.clone(),
                                routing_method: routing_method.into(),
                                status: upstream_status,
                                duration: start.elapsed(),
                                input_tokens: 0,
                                output_tokens: 0,
                                error_body: None,
                            });
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
                        let usage = crate::proxy::extract_usage(&resp_bytes);
                        if let Some(ref metrics) = state.metrics {
                            metrics.record(RequestRecord {
                                id: 0,
                                timestamp: Instant::now(),
                                wallclock: chrono::Utc::now(),
                                model: metrics_model.clone(),
                                provider: provider_name.clone(),
                                routing_method: routing_method.into(),
                                status: upstream_status,
                                duration: start.elapsed(),
                                input_tokens: usage.0,
                                output_tokens: usage.1,
                                error_body: None,
                            });
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
    drop_requested_retained_sessions(
        Arc::clone(&engine),
        req.drop_session_id,
        req.drop_session_ids.as_deref(),
    )
    .await?;

    let max_tokens =
        resolved_max_tokens(&req, &generation_defaults, state.config.server.max_tokens);
    let sampling = build_sampling_params(&req, &generation_defaults)?;
    let stop_sequences = StopSequence::extract(req.stop);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs;

    // Extract images and inject <image> placeholders for VLMs
    let images = extract_images(&req.messages);
    let effective_messages = if images.is_empty() {
        req.messages.clone()
    } else {
        inject_image_placeholders(&req.messages)
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

    let (mut prompt_tokens, pflash_policy) = engine
        .prepare_chat_prompt_with_pflash_policy(&messages, tools, thinking_enabled)
        .map_err(ServerError::Engine)?;

    // Preprocess images for VLM
    let pixel_values = if !images.is_empty() && engine.is_vlm() {
        engine.replace_image_tokens(&mut prompt_tokens);
        let image_size = engine.vlm_image_size().unwrap_or(384);
        #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
        let size = image_size as u32;
        let first_image = images
            .into_iter()
            .next()
            .ok_or_else(|| ServerError::BadRequest("Image data is empty".to_owned()))?;
        let pv = higgs_models::siglip::preprocess_image(&first_image, size)
            .map_err(|e| ServerError::InternalError(format!("Image preprocessing failed: {e}")))?;
        Some(pv)
    } else {
        None
    };

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
        pixel_values.is_some(),
        constraint.is_some(),
        checkpoint_id.as_deref(),
        want_logprobs,
        !stop_sequences.is_empty(),
    );
    let request_id = generate_request_id();
    let has_tools = tools.is_some();
    let tool_payload = tool_payload_stats(&effective_messages);
    warn_large_tool_payload(tool_payload);

    if let Some(sid) = session_id {
        let max_session_prefill_tokens = engine.session_max_suffix_prefill_tokens();
        let retained_tokens = engine.retained_session_tokens(sid);
        let continued_prompt = if retained_tokens.is_some() {
            continued_prompt_tokens(
                &engine,
                sid,
                &prompt_tokens,
                &messages,
                tools,
                thinking_enabled,
            )
        } else {
            prompt_tokens.clone()
        };

        let strategy = session_prefill_strategy(
            Some(sid),
            retained_tokens.as_deref(),
            &continued_prompt,
            max_session_prefill_tokens,
        );
        match strategy {
            SessionPrefillStrategy::Continue { session_id: sid } => {
                record_session_prompt_trace(
                    &engine,
                    sid,
                    retained_tokens.as_deref(),
                    &prompt_tokens,
                    &continued_prompt,
                    tool_payload,
                    SessionPromptTraceOutcome::Continued,
                );
                let sampling_c = sampling.clone();
                let engine_c = Arc::clone(&engine);
                let session_output = tokio::task::spawn_blocking(move || {
                    engine_c.generate_continued_with_thinking(
                        sid,
                        &continued_prompt,
                        max_tokens,
                        &sampling_c,
                        thinking_enabled,
                    )
                })
                .await
                .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
                .map_err(ServerError::Engine)?;

                return Ok(build_session_response(
                    &req.model,
                    &request_id,
                    session_output,
                    tools,
                    has_tools,
                    thinking_enabled,
                ));
            }
            SessionPrefillStrategy::BootstrapExact {
                session_id: sid,
                reason,
            } => {
                let bootstrap_route = session_bootstrap_route(
                    reason,
                    engine.pflash_can_run_stateless_for_prompt(&prompt_tokens),
                );
                match bootstrap_route {
                    SessionBootstrapRoute::ExactRetained => {
                        record_session_prompt_trace(
                            &engine,
                            sid,
                            retained_tokens.as_deref(),
                            &prompt_tokens,
                            &continued_prompt,
                            tool_payload,
                            SessionPromptTraceOutcome::ExactBootstrap,
                        );
                        handle_session_exact_bootstrap(sid, reason);
                        let sampling_c = sampling.clone();
                        let engine_c = Arc::clone(&engine);
                        let session_output = tokio::task::spawn_blocking(move || {
                            engine_c.generate_continued_with_thinking(
                                sid,
                                &continued_prompt,
                                max_tokens,
                                &sampling_c,
                                thinking_enabled,
                            )
                        })
                        .await
                        .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
                        .map_err(ServerError::Engine)?;

                        return Ok(build_session_response(
                            &req.model,
                            &request_id,
                            session_output,
                            tools,
                            has_tools,
                            thinking_enabled,
                        ));
                    }
                    SessionBootstrapRoute::StatelessPflash => {
                        record_session_prompt_trace(
                            &engine,
                            sid,
                            retained_tokens.as_deref(),
                            &prompt_tokens,
                            &continued_prompt,
                            tool_payload,
                            SessionPromptTraceOutcome::StatelessPflashBootstrap,
                        );
                        handle_session_stateless_pflash_bootstrap(sid, reason);
                    }
                }
            }
            SessionPrefillStrategy::Stateless(reason) => {
                record_session_prompt_trace(
                    &engine,
                    sid,
                    retained_tokens.as_deref(),
                    &prompt_tokens,
                    &continued_prompt,
                    tool_payload,
                    SessionPromptTraceOutcome::StatelessPrefill,
                );
                handle_session_stateless_prefill(sid, reason);
            }
        }
    }

    let output = tokio::task::spawn_blocking(move || {
        engine.generate_with_thinking_and_pflash_policy(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            want_logprobs,
            top_logprobs,
            thinking_enabled,
            constraint,
            pixel_values,
            checkpoint_id.as_deref(),
            &pflash_policy,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(ServerError::Engine)?;

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
        // which is not surfaced per-request through `GenerationOutput`, so
        // report no cached tokens rather than a fabricated value.
        usage: CompletionUsage::new(output.prompt_tokens, output.completion_tokens, 0),
    })
}

/// Build the token sequence fed to `generate_continued`. The engine guard
/// prefills only the suffix IFF the retained tokens are a strict token-prefix of
/// what we pass, so we hand it `retained ++ delta`.
///
/// `retained` carries the `<think>...</think>` block the chat template injected
/// for the prior assistant turn. The canonical multi-turn render strips
/// `<think>` from historical assistant turns, so match a think-stripped copy of
/// `retained` against the full render to find the new-turn delta, then append
/// that delta to the original retained tokens.
// `print_stderr`: env-gated (HIGGS_DIAG_SESSION_TIMING) diagnostics only.
#[allow(clippy::print_stderr)]
fn continued_prompt_tokens(
    engine: &Arc<Engine>,
    session_id: u64,
    prompt_tokens: &[u32],
    messages: &[higgs_engine::chat_template::ChatMessage],
    tools: Option<&[serde_json::Value]>,
    thinking_enabled: bool,
) -> Vec<u32> {
    let diag = std::env::var("HIGGS_DIAG_SESSION_TIMING").is_ok_and(|v| v == "1");
    let Some(retained) = engine.retained_session_tokens(session_id) else {
        if diag {
            eprintln!(
                "DIAG session-continue-fallback: reason=no_retained_cache session_id={session_id} prompt_tokens={}",
                prompt_tokens.len()
            );
        }
        return prompt_tokens.to_vec();
    };
    let retained_text = match engine.tokenizer().decode(&retained, false) {
        Ok(text) => text,
        Err(error) => {
            if diag {
                eprintln!(
                    "DIAG session-continue-fallback: reason=decode_retained_failed session_id={session_id} retained_tokens={} error={error}",
                    retained.len()
                );
            }
            return prompt_tokens.to_vec();
        }
    };
    let full_text = match engine.render_chat_prompt_with_thinking(messages, tools, thinking_enabled)
    {
        Ok(text) => text,
        Err(error) => {
            if diag {
                eprintln!(
                    "DIAG session-continue-fallback: reason=render_full_failed session_id={session_id} retained_tokens={} prompt_tokens={} error={error}",
                    retained.len(),
                    prompt_tokens.len()
                );
            }
            return prompt_tokens.to_vec();
        }
    };
    let Some(delta_text) = message_boundary_delta(&retained_text, &full_text) else {
        if diag {
            let common = common_prefix_bytes(&retained_text, &full_text);
            eprintln!(
                "DIAG session-continue-fallback: reason=boundary_splice_failed session_id={session_id} retained_tokens={} prompt_tokens={} retained_bytes={} full_bytes={} retained_msgs={} full_msgs={} common_bytes={} retained_tail={:?} full_at_mismatch={:?}",
                retained.len(),
                prompt_tokens.len(),
                retained_text.len(),
                full_text.len(),
                retained_text.matches(IM_START).count(),
                full_text.matches(IM_START).count(),
                common,
                preview_around(&retained_text, common),
                preview_around(&full_text, common)
            );
        }
        return prompt_tokens.to_vec();
    };
    let Ok(delta_enc) = engine.tokenizer().encode(delta_text, false) else {
        if diag {
            eprintln!(
                "DIAG session-continue-fallback: reason=encode_delta_failed session_id={session_id} delta_bytes={}",
                delta_text.len()
            );
        }
        return prompt_tokens.to_vec();
    };
    if diag {
        eprintln!(
            "DIAG session-continue: matched session_id={session_id} retained_tokens={} delta_tokens={} full_prompt_tokens={}",
            retained.len(),
            delta_enc.get_ids().len(),
            prompt_tokens.len()
        );
    }
    let mut combined = retained;
    combined.extend_from_slice(delta_enc.get_ids());
    combined
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionPrefillStrategy {
    Continue {
        session_id: u64,
    },
    BootstrapExact {
        session_id: u64,
        reason: SessionBootstrapReason,
    },
    Stateless(SessionPrefillFallback),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionBootstrapReason {
    ColdPromptTooLarge {
        prompt_tokens: usize,
        max_prefill_tokens: usize,
    },
    DivergedOrNotGrowing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionBootstrapRoute {
    ExactRetained,
    StatelessPflash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionPrefillFallback {
    NoSessionId,
    LargeSuffix {
        suffix_tokens: usize,
        max_prefill_tokens: usize,
    },
}

fn session_prefill_strategy(
    session_id: Option<u64>,
    retained_tokens: Option<&[u32]>,
    continuation_candidate: &[u32],
    max_prefill_tokens: usize,
) -> SessionPrefillStrategy {
    let Some(session_id) = session_id else {
        return SessionPrefillStrategy::Stateless(SessionPrefillFallback::NoSessionId);
    };

    let Some(retained_tokens) = retained_tokens else {
        if continuation_candidate.len() <= max_prefill_tokens {
            return SessionPrefillStrategy::Continue { session_id };
        }
        return SessionPrefillStrategy::BootstrapExact {
            session_id,
            reason: SessionBootstrapReason::ColdPromptTooLarge {
                prompt_tokens: continuation_candidate.len(),
                max_prefill_tokens,
            },
        };
    };

    let prior = retained_tokens.len();
    if prior == 0
        || prior >= continuation_candidate.len()
        || continuation_candidate.get(..prior) != Some(retained_tokens)
    {
        return SessionPrefillStrategy::BootstrapExact {
            session_id,
            reason: SessionBootstrapReason::DivergedOrNotGrowing,
        };
    }

    let suffix_tokens = continuation_candidate.len() - prior;
    if suffix_tokens <= max_prefill_tokens {
        SessionPrefillStrategy::Continue { session_id }
    } else {
        SessionPrefillStrategy::Stateless(SessionPrefillFallback::LargeSuffix {
            suffix_tokens,
            max_prefill_tokens,
        })
    }
}

fn session_bootstrap_route(
    _reason: SessionBootstrapReason,
    pflash_can_run_stateless: bool,
) -> SessionBootstrapRoute {
    if pflash_can_run_stateless {
        SessionBootstrapRoute::StatelessPflash
    } else {
        SessionBootstrapRoute::ExactRetained
    }
}

fn handle_session_stateless_prefill(session_id: u64, reason: SessionPrefillFallback) {
    tracing::info!(
        session_id,
        ?reason,
        "session continuation skipped; using stateless prefill path"
    );
}

fn handle_session_exact_bootstrap(session_id: u64, reason: SessionBootstrapReason) {
    tracing::info!(
        session_id,
        ?reason,
        "session continuation bootstrapping exact retained prefill path"
    );
}

fn handle_session_stateless_pflash_bootstrap(session_id: u64, reason: SessionBootstrapReason) {
    tracing::info!(
        session_id,
        ?reason,
        "session continuation cannot reuse retained KV; using stateless PFlash/cache path"
    );
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ToolPayloadStats {
    messages: usize,
    bytes: usize,
    largest_bytes: usize,
}

fn tool_payload_stats(messages: &[ChatCompletionMessage]) -> ToolPayloadStats {
    let mut stats = ToolPayloadStats::default();
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

fn warn_large_tool_payload(stats: ToolPayloadStats) {
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

fn record_session_prompt_trace(
    engine: &Arc<Engine>,
    session_id: u64,
    retained_tokens: Option<&[u32]>,
    prompt_tokens: &[u32],
    candidate_tokens: &[u32],
    tool_payload: ToolPayloadStats,
    outcome: SessionPromptTraceOutcome,
) {
    let trace = session_prompt_trace_metrics(
        retained_tokens,
        prompt_tokens,
        candidate_tokens,
        tool_payload,
        outcome,
    );
    tracing::info!(
        session_id,
        prompt_tokens = trace.prompt_tokens,
        retained_tokens = trace.retained_tokens,
        candidate_tokens = trace.candidate_tokens,
        suffix_tokens = trace.suffix_tokens,
        common_prefix_tokens = trace.common_prefix_tokens,
        divergence_token = ?trace.divergence_token,
        boundary_splice = trace.boundary_splice,
        tool_result_messages = trace.tool_result_messages,
        tool_result_bytes = trace.tool_result_bytes,
        tool_result_largest_bytes = trace.tool_result_largest_bytes,
        ?outcome,
        "session prompt/cache trace"
    );
    engine.record_session_prompt_trace(trace);
}

fn session_prompt_trace_metrics(
    retained_tokens: Option<&[u32]>,
    prompt_tokens: &[u32],
    candidate_tokens: &[u32],
    tool_payload: ToolPayloadStats,
    outcome: SessionPromptTraceOutcome,
) -> SessionPromptTraceMetrics {
    let Some(retained) = retained_tokens else {
        return SessionPromptTraceMetrics {
            prompt_tokens: prompt_tokens.len(),
            retained_tokens: 0,
            candidate_tokens: candidate_tokens.len(),
            suffix_tokens: candidate_tokens.len(),
            common_prefix_tokens: 0,
            divergence_token: None,
            boundary_splice: false,
            tool_result_messages: tool_payload.messages,
            tool_result_bytes: tool_payload.bytes,
            tool_result_largest_bytes: tool_payload.largest_bytes,
            outcome,
        };
    };

    let common_prefix_tokens = common_prefix_tokens(retained, candidate_tokens);
    let retained_prefix_of_candidate = !retained.is_empty()
        && retained.len() < candidate_tokens.len()
        && common_prefix_tokens == retained.len();
    let retained_prefix_of_prompt = !retained.is_empty()
        && retained.len() < prompt_tokens.len()
        && prompt_tokens.get(..retained.len()) == Some(retained);
    let divergence_token = if retained_prefix_of_candidate {
        None
    } else {
        Some(common_prefix_tokens)
    };
    let suffix_tokens = if retained_prefix_of_candidate {
        candidate_tokens.len().saturating_sub(retained.len())
    } else {
        candidate_tokens.len()
    };

    SessionPromptTraceMetrics {
        prompt_tokens: prompt_tokens.len(),
        retained_tokens: retained.len(),
        candidate_tokens: candidate_tokens.len(),
        suffix_tokens,
        common_prefix_tokens,
        divergence_token,
        boundary_splice: !retained_prefix_of_prompt && retained_prefix_of_candidate,
        tool_result_messages: tool_payload.messages,
        tool_result_bytes: tool_payload.bytes,
        tool_result_largest_bytes: tool_payload.largest_bytes,
        outcome,
    }
}

fn common_prefix_tokens(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn common_prefix_bytes(left: &str, right: &str) -> usize {
    left.as_bytes()
        .iter()
        .zip(right.as_bytes())
        .take_while(|(l, r)| l == r)
        .count()
}

fn preview_around(text: &str, byte_pos: usize) -> String {
    let start = char_boundary_at_or_before(text, byte_pos.saturating_sub(80));
    let end = char_boundary_at_or_after(text, byte_pos.saturating_add(80));
    text.get(start..end).unwrap_or_default().to_owned()
}

fn char_boundary_at_or_before(text: &str, byte_pos: usize) -> usize {
    let pos = byte_pos.min(text.len());
    (0..=pos)
        .rev()
        .find(|idx| text.is_char_boundary(*idx))
        .unwrap_or(0)
}

fn char_boundary_at_or_after(text: &str, byte_pos: usize) -> usize {
    let pos = byte_pos.min(text.len());
    (pos..=text.len())
        .find(|idx| text.is_char_boundary(*idx))
        .unwrap_or(text.len())
}

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

#[derive(Debug)]
struct RenderedMessageSegment<'a> {
    text_without_end: &'a str,
    end_start: Option<usize>,
    end_after: Option<usize>,
}

/// Splice point for a continued turn: the suffix of the canonical render that
/// covers the messages the retained KV does NOT yet cover.
///
/// The retained detokenization can never byte-match the canonical re-render of
/// the same messages — think blocks are stripped or replaced by a placeholder
/// depending on the turn's position relative to the conversation's last user
/// message, and content/tool-call join whitespace differs from what the model
/// generated. Instead of canonicalizing (template lore, position-dependent),
/// splice at message boundaries: the retained text covers as many messages as
/// it has `<|im_start|>` markers; everything from that boundary onward in the
/// fresh render is the delta to prefill. The retained prefix keeps the model's
/// original per-turn text (real thinking, original whitespace) — a known,
/// accepted divergence source of the best-effort continuation path.
///
/// The last retained message usually ends without `<|im_end|>` (the engine
/// pops the EOS token before stashing), so the delta starts AT the covered
/// messages' final `<|im_end|>`; when the retained text does end with it, the
/// delta starts right after instead. Returns `None` (caller falls back to a
/// full prefill) when the render has fewer messages than the retained text,
/// any covered complete message changed, or literal chat-template markers make
/// the text ambiguous to splice safely. The final retained message is commonly
/// the previously generated assistant turn without `<|im_end|>`; for that one,
/// the retained KV is the source of truth and only the role boundary must match.
fn message_boundary_delta<'a>(retained_text: &str, full_text: &'a str) -> Option<&'a str> {
    let retained_segments = rendered_message_segments(retained_text)?;
    let full_segments = rendered_message_segments(full_text)?;
    let covered = retained_segments.len();
    if covered == 0 || full_segments.len() < covered {
        return None;
    }

    for (idx, (retained, fresh)) in retained_segments
        .iter()
        .zip(full_segments.iter())
        .enumerate()
        .take(covered)
    {
        if idx + 1 == covered && generated_assistant_boundary(retained, fresh) {
            continue;
        }
        if normalized_segment(retained.text_without_end)
            != normalized_segment(fresh.text_without_end)
        {
            return None;
        }
    }

    let covered_segment = &full_segments[covered - 1];
    let end_start = covered_segment.end_start?;
    let end_after = covered_segment.end_after?;

    // `end_after` is just past the covered messages' final <|im_end|>. The
    // retained text usually lacks that closer (EOS popped at stash time), so
    // include it in the delta; skip it when the retained text already ends with
    // it.
    let trimmed = retained_text.trim_end_matches('\n');
    if trimmed.ends_with(IM_END) {
        full_text.get(end_after..)
    } else {
        full_text.get(end_start..)
    }
}

fn rendered_message_segments(text: &str) -> Option<Vec<RenderedMessageSegment<'_>>> {
    let mut segments = Vec::new();
    let mut pos = 0usize;

    while let Some(relative_start) = text.get(pos..)?.find(IM_START) {
        let start = pos + relative_start;
        if !text.get(pos..start)?.trim().is_empty() {
            return None;
        }

        let body_start = start + IM_START.len();
        let Some(relative_end) = text.get(body_start..)?.find(IM_END) else {
            let partial = text.get(start..)?;
            if partial.get(IM_START.len()..)?.contains(IM_START) {
                return None;
            }
            segments.push(RenderedMessageSegment {
                text_without_end: partial,
                end_start: None,
                end_after: None,
            });
            return Some(segments);
        };

        let end_start = body_start + relative_end;
        let end_after = end_start + IM_END.len();
        let body = text.get(body_start..end_start)?;
        if body.contains(IM_START) {
            return None;
        }

        segments.push(RenderedMessageSegment {
            text_without_end: text.get(start..end_start)?,
            end_start: Some(end_start),
            end_after: Some(end_after),
        });
        pos = end_after;
    }

    if !text.get(pos..)?.trim().is_empty() {
        return None;
    }
    Some(segments)
}

fn generated_assistant_boundary(
    retained: &RenderedMessageSegment<'_>,
    fresh: &RenderedMessageSegment<'_>,
) -> bool {
    retained.end_start.is_none()
        && fresh.end_start.is_some()
        && segment_role(retained.text_without_end) == Some("assistant")
        && segment_role(fresh.text_without_end) == Some("assistant")
}

fn segment_role(segment: &str) -> Option<&str> {
    segment
        .strip_prefix(IM_START)?
        .split_once('\n')
        .map(|(role, _)| role)
}

fn normalized_segment(segment: &str) -> String {
    let without_think = strip_think_blocks(segment);
    canonicalize_tool_calls(&without_think)
        .trim_end_matches('\n')
        .to_owned()
}

fn canonicalize_tool_calls(text: &str) -> String {
    let parsed = higgs_engine::tool_parser::parse_tool_calls(text, None);
    if parsed.tool_calls.is_empty() {
        return text.to_owned();
    }

    let mut out = parsed.text.trim_end_matches('\n').to_owned();
    for call in parsed.tool_calls {
        out.push_str("\n<tool_call:");
        out.push_str(&call.name);
        out.push(':');
        out.push_str(&canonical_json_value(&call.arguments));
        out.push('>');
    }
    out
}

fn canonical_json_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Array(values) => {
            let body = values
                .iter()
                .map(canonical_json_value)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        serde_json::Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            let body = entries
                .into_iter()
                .map(|(key, value)| {
                    let key = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_owned());
                    format!("{key}:{}", canonical_json_value(value))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
        _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_owned()),
    }
}

fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find(THINK_START) {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + THINK_START.len()..];
        let Some(end) = after_start.find(THINK_END) else {
            out.push_str(&rest[start..]);
            return out;
        };
        rest = &after_start[end + THINK_END.len()..];
        if let Some(after_blank) = rest.strip_prefix("\n\n") {
            rest = after_blank;
        }
    }

    out.push_str(rest);
    out
}

/// Map a [`SessionGeneration`] (cache-resident continued turn) onto the same
/// `ChatCompletionResponse` shape as the normal path, preserving reasoning
/// extraction, tool-call parsing, and the `finish_reason: "tool_calls"`
/// override. The continued path uses greedy decode without logprobs/streaming,
/// so logprobs are absent and `finish_reason` defaults to `"stop"`.
fn build_session_response(
    model: &str,
    request_id: &str,
    output: higgs_engine::simple::SessionGeneration,
    tools: Option<&[serde_json::Value]>,
    has_tools: bool,
    thinking_enabled: bool,
) -> ChatCompletionResponse {
    let usage = session_usage(&output);
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
                "stop".to_owned(),
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
            "stop".to_owned(),
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
        output.prompt_tokens.saturating_sub(output.prefilled_tokens)
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
    drop_requested_retained_sessions(
        Arc::clone(&engine),
        req.drop_session_id,
        req.drop_session_ids.as_deref(),
    )
    .await?;

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
    let sampling = build_sampling_params(&req, &generation_defaults)?;
    let stop_sequences = StopSequence::extract(req.stop);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs;

    // Extract images and inject <image> placeholders for VLMs
    let images = extract_images(&req.messages);
    let effective_messages = if images.is_empty() {
        req.messages.clone()
    } else {
        inject_image_placeholders(&req.messages)
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
    let (mut prompt_tokens, pflash_policy) = engine
        .prepare_chat_prompt_with_pflash_policy(&messages, prompt_tools, thinking_enabled_stream)
        .map_err(ServerError::Engine)?;

    // Preprocess images for VLM
    let pixel_values = if !images.is_empty() && engine.is_vlm() {
        engine.replace_image_tokens(&mut prompt_tokens);
        let image_size = engine.vlm_image_size().unwrap_or(384);
        #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
        let size = image_size as u32;
        let first_image = images
            .into_iter()
            .next()
            .ok_or_else(|| ServerError::BadRequest("Image data is empty".to_owned()))?;
        let pv = higgs_models::siglip::preprocess_image(&first_image, size)
            .map_err(|e| ServerError::InternalError(format!("Image preprocessing failed: {e}")))?;
        Some(pv)
    } else {
        None
    };

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
    let model = req.model;
    let checkpoint_id = req.checkpoint_id;
    let prompt_token_count = u32::try_from(prompt_tokens.len()).unwrap_or(0);
    let tool_payload = tool_payload_stats(&effective_messages);
    warn_large_tool_payload(tool_payload);

    let start = Instant::now();
    let metrics_id = metrics.as_ref().map(|m| {
        m.record_pending(RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: model.clone(),
            provider: "higgs".to_owned(),
            routing_method: routing_method.into(),
            status: 200,
            duration: Duration::ZERO,
            input_tokens: u64::from(prompt_token_count),
            output_tokens: 0,
            error_body: None,
        })
    });

    let stream_session_id = session_continuation_id(
        request_session_id,
        pixel_values.is_some(),
        constraint.is_some(),
        checkpoint_id.as_deref(),
        want_logprobs,
        !stop_sequences.is_empty(),
    );

    let tokenizer = engine.tokenizer().clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

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
        let max_session_prefill_tokens = engine.session_max_suffix_prefill_tokens();
        let retained_tokens = engine.retained_session_tokens(sid);
        let continued_prompt = if retained_tokens.is_some() {
            continued_prompt_tokens(
                &engine,
                sid,
                &prompt_tokens,
                &messages,
                prompt_tools,
                thinking_enabled_stream,
            )
        } else {
            prompt_tokens.clone()
        };

        let strategy = session_prefill_strategy(
            Some(sid),
            retained_tokens.as_deref(),
            &continued_prompt,
            max_session_prefill_tokens,
        );
        match strategy {
            SessionPrefillStrategy::Continue { session_id: sid } => {
                record_session_prompt_trace(
                    &engine,
                    sid,
                    retained_tokens.as_deref(),
                    &prompt_tokens,
                    &continued_prompt,
                    tool_payload,
                    SessionPromptTraceOutcome::Continued,
                );
                tokio::task::spawn_blocking(move || {
                    let result = engine.generate_continued_streaming_with_thinking(
                        sid,
                        &continued_prompt,
                        max_tokens,
                        &sampling,
                        &tx,
                        thinking_enabled_stream,
                    );
                    if let Err(e) = result {
                        tracing::error!(error = %e, "Session generation error during streaming");
                    }
                });
            }
            SessionPrefillStrategy::BootstrapExact {
                session_id: sid,
                reason,
            } => {
                let bootstrap_route = session_bootstrap_route(
                    reason,
                    engine.pflash_can_run_stateless_for_prompt(&prompt_tokens),
                );
                match bootstrap_route {
                    SessionBootstrapRoute::ExactRetained => {
                        record_session_prompt_trace(
                            &engine,
                            sid,
                            retained_tokens.as_deref(),
                            &prompt_tokens,
                            &continued_prompt,
                            tool_payload,
                            SessionPromptTraceOutcome::ExactBootstrap,
                        );
                        handle_session_exact_bootstrap(sid, reason);
                        tokio::task::spawn_blocking(move || {
                            let result = engine.generate_continued_streaming_with_thinking(
                                sid,
                                &continued_prompt,
                                max_tokens,
                                &sampling,
                                &tx,
                                thinking_enabled_stream,
                            );
                            if let Err(e) = result {
                                tracing::error!(error = %e, "Session generation error during streaming");
                            }
                        });
                    }
                    SessionBootstrapRoute::StatelessPflash => {
                        record_session_prompt_trace(
                            &engine,
                            sid,
                            retained_tokens.as_deref(),
                            &prompt_tokens,
                            &continued_prompt,
                            tool_payload,
                            SessionPromptTraceOutcome::StatelessPflashBootstrap,
                        );
                        handle_session_stateless_pflash_bootstrap(sid, reason);
                        tokio::task::spawn_blocking(move || {
                            let result = engine.generate_streaming_with_thinking_and_pflash_policy(
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
                                pixel_values,
                                checkpoint_id.as_deref(),
                                &pflash_policy,
                            );
                            if let Err(e) = result {
                                tracing::error!(error = %e, "Generation error during streaming");
                            }
                        });
                    }
                }
            }
            SessionPrefillStrategy::Stateless(reason) => {
                record_session_prompt_trace(
                    &engine,
                    sid,
                    retained_tokens.as_deref(),
                    &prompt_tokens,
                    &continued_prompt,
                    tool_payload,
                    SessionPromptTraceOutcome::StatelessPrefill,
                );
                handle_session_stateless_prefill(sid, reason);
                tokio::task::spawn_blocking(move || {
                    let result = engine.generate_streaming_with_thinking_and_pflash_policy(
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
                        pixel_values,
                        checkpoint_id.as_deref(),
                        &pflash_policy,
                    );
                    if let Err(e) = result {
                        tracing::error!(error = %e, "Generation error during streaming");
                    }
                });
            }
        }
    } else {
        tokio::task::spawn_blocking(move || {
            let result = engine.generate_streaming_with_thinking_and_pflash_policy(
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
                pixel_values,
                checkpoint_id.as_deref(),
                &pflash_policy,
            );
            if let Err(e) = result {
                tracing::error!(error = %e, "Generation error during streaming");
            }
        });
    }

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
            let usage =
                CompletionUsage::new(prompt_token_count, output_token_count, cached_prompt_tokens);
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

/// Extract image bytes from base64 data URIs in message content parts.
/// Returns decoded image bytes for each image found across all messages.
fn extract_images(messages: &[ChatCompletionMessage]) -> Vec<Vec<u8>> {
    use base64::Engine as _;
    let mut images = Vec::new();
    for msg in messages {
        let Some(content) = &msg.content else {
            continue;
        };
        for url in content.image_urls() {
            if let Some(data) = url.strip_prefix("data:") {
                // data:[<mediatype>];base64,<data>
                if let Some(base64_start) = data.find(";base64,") {
                    let encoded = &data[base64_start + 8..];
                    match base64::engine::general_purpose::STANDARD.decode(encoded) {
                        Ok(bytes) => images.push(bytes),
                        Err(e) => tracing::warn!(error = %e, "Failed to decode base64 image"),
                    }
                }
            }
            // HTTP/HTTPS URLs are not supported yet; could be fetched in the future
        }
    }
    images
}

/// Build text content with `<image>` placeholders injected for each image.
/// For VLMs, each image in a message gets a `<image>\n` prefix before the text.
fn inject_image_placeholders(messages: &[ChatCompletionMessage]) -> Vec<ChatCompletionMessage> {
    messages
        .iter()
        .map(|m| {
            let Some(content) = &m.content else {
                return m.clone();
            };
            if !content.has_images() {
                return m.clone();
            }

            let image_count = content.image_urls().len();
            let text = content.text();
            let prefix = "<image>\n".repeat(image_count);
            let combined = format!("{prefix}{text}");

            ChatCompletionMessage {
                role: m.role.clone(),
                content: Some(MessageContent::Text(combined)),
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
    has_pixel_values: bool,
    has_constraint: bool,
    checkpoint_id: Option<&str>,
    want_logprobs: bool,
    has_stop_sequences: bool,
) -> Option<u64> {
    session_id
        .filter(|_| !has_pixel_values)
        .filter(|_| !has_constraint)
        .filter(|_| checkpoint_id.is_none())
        .filter(|_| !want_logprobs)
        .filter(|_| !has_stop_sequences)
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

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

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
            prompt_tokens: 1000,
            prefilled_tokens: 120,
            continued: true,
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
            prompt_tokens: 1000,
            prefilled_tokens: 1000,
            continued: false,
        };
        assert!(session_usage(&cold).prompt_tokens_details.is_none());
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
    fn cold_long_session_bootstraps_exact_retained_cache() {
        const MAX_PREFILL_TOKENS: usize = 128;
        let prompt = vec![7; MAX_PREFILL_TOKENS + 1];

        assert_eq!(
            session_prefill_strategy(Some(42), None, &prompt, MAX_PREFILL_TOKENS),
            SessionPrefillStrategy::BootstrapExact {
                session_id: 42,
                reason: SessionBootstrapReason::ColdPromptTooLarge {
                    prompt_tokens: prompt.len(),
                    max_prefill_tokens: MAX_PREFILL_TOKENS,
                },
            }
        );
    }

    #[test]
    fn cold_short_session_still_seeds_retained_cache() {
        const MAX_PREFILL_TOKENS: usize = 128;
        let prompt = vec![7; MAX_PREFILL_TOKENS];

        assert_eq!(
            session_prefill_strategy(Some(42), None, &prompt, MAX_PREFILL_TOKENS),
            SessionPrefillStrategy::Continue { session_id: 42 }
        );
    }

    #[test]
    fn warm_session_with_small_exact_suffix_continues() {
        const MAX_PREFILL_TOKENS: usize = 128;
        let retained = vec![1, 2, 3];
        let mut candidate = retained.clone();
        candidate.extend(std::iter::repeat_n(9, MAX_PREFILL_TOKENS));

        assert_eq!(
            session_prefill_strategy(Some(42), Some(&retained), &candidate, MAX_PREFILL_TOKENS),
            SessionPrefillStrategy::Continue { session_id: 42 }
        );
    }

    #[test]
    fn warm_session_with_large_suffix_routes_to_stateless_pflash_path() {
        const MAX_PREFILL_TOKENS: usize = 128;
        let retained = vec![1, 2, 3];
        let mut candidate = retained.clone();
        candidate.extend(std::iter::repeat_n(9, MAX_PREFILL_TOKENS + 1));

        assert_eq!(
            session_prefill_strategy(Some(42), Some(&retained), &candidate, MAX_PREFILL_TOKENS),
            SessionPrefillStrategy::Stateless(SessionPrefillFallback::LargeSuffix {
                suffix_tokens: MAX_PREFILL_TOKENS + 1,
                max_prefill_tokens: MAX_PREFILL_TOKENS,
            })
        );
    }

    #[test]
    fn warm_session_with_diverged_candidate_bootstraps_exact_retained_cache() {
        const MAX_PREFILL_TOKENS: usize = 128;
        let retained = vec![1, 2, 3];
        let candidate = vec![1, 2, 4, 9];

        assert_eq!(
            session_prefill_strategy(Some(42), Some(&retained), &candidate, MAX_PREFILL_TOKENS),
            SessionPrefillStrategy::BootstrapExact {
                session_id: 42,
                reason: SessionBootstrapReason::DivergedOrNotGrowing,
            }
        );
    }

    #[test]
    fn warm_session_with_not_growing_candidate_bootstraps_exact_retained_cache() {
        const MAX_PREFILL_TOKENS: usize = 128;
        let retained = vec![1, 2, 3];

        assert_eq!(
            session_prefill_strategy(Some(42), Some(&retained), &retained, MAX_PREFILL_TOKENS),
            SessionPrefillStrategy::BootstrapExact {
                session_id: 42,
                reason: SessionBootstrapReason::DivergedOrNotGrowing,
            }
        );
    }

    #[test]
    fn session_bootstrap_uses_exact_retained_route_without_pflash() {
        assert_eq!(
            session_bootstrap_route(SessionBootstrapReason::DivergedOrNotGrowing, false),
            SessionBootstrapRoute::ExactRetained
        );
        assert_eq!(
            session_bootstrap_route(
                SessionBootstrapReason::ColdPromptTooLarge {
                    prompt_tokens: 129,
                    max_prefill_tokens: 128,
                },
                false,
            ),
            SessionBootstrapRoute::ExactRetained
        );
    }

    #[test]
    fn session_bootstrap_uses_stateless_pflash_route_when_available() {
        assert_eq!(
            session_bootstrap_route(SessionBootstrapReason::DivergedOrNotGrowing, true),
            SessionBootstrapRoute::StatelessPflash
        );
        assert_eq!(
            session_bootstrap_route(
                SessionBootstrapReason::ColdPromptTooLarge {
                    prompt_tokens: 129,
                    max_prefill_tokens: 128,
                },
                true,
            ),
            SessionBootstrapRoute::StatelessPflash
        );
    }

    #[test]
    fn boundary_delta_splices_after_covered_messages() {
        // Retained: system + user + generated assistant reply (EOS popped, so
        // no trailing <|im_end|>), with real thinking the render won't have.
        let retained = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\n<think>\nreasoning\n</think>\n\nanswer1";
        // Fresh render: same three messages canonically rendered (thinking
        // stripped, whitespace differs) plus a new user turn + gen suffix.
        let full = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\nanswer1<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n<think>";
        let delta = message_boundary_delta(retained, full).unwrap();
        // Delta starts at the covered messages' closing <|im_end|> (retained
        // lacks it) and carries everything new.
        assert_eq!(
            delta,
            "<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n<think>"
        );
    }

    #[test]
    fn boundary_delta_skips_closer_when_retained_has_it() {
        let retained = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\nans<|im_end|>";
        let full = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\nans<|im_end|>\n<|im_start|>user\nq2<|im_end|>";
        let delta = message_boundary_delta(retained, full).unwrap();
        assert_eq!(delta, "\n<|im_start|>user\nq2<|im_end|>");
    }

    #[test]
    fn boundary_delta_treats_generated_assistant_as_retained_source_of_truth() {
        let retained = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nI'm **NanoBot** - compact and direct.";
        let full = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\nI'm **NanoBot** (surname: Bonsai) - compact, direct, and local.<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

        let delta = message_boundary_delta(retained, full).unwrap();

        assert_eq!(
            delta,
            "<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn boundary_delta_rejects_partial_non_assistant_boundary() {
        let retained = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\npartial";
        let full = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\npartial edited<|im_end|>\n<|im_start|>assistant\n";

        assert!(message_boundary_delta(retained, full).is_none());
    }

    #[test]
    fn boundary_delta_rejects_mutated_first_message() {
        // Client rewrote the system prompt between turns: no splice.
        let retained = "<|im_start|>system\noriginal<|im_end|>\n<|im_start|>user\nq<|im_end|>";
        let full = "<|im_start|>system\nMUTATED!<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>user\nq2<|im_end|>";
        assert!(message_boundary_delta(retained, full).is_none());
    }

    #[test]
    fn boundary_delta_rejects_mutated_middle_message() {
        let retained = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nold context<|im_end|>\n<|im_start|>assistant\nanswer";
        let full = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nNEW context<|im_end|>\n<|im_start|>assistant\nanswer<|im_end|>\n<|im_start|>user\nq2<|im_end|>";
        assert!(message_boundary_delta(retained, full).is_none());
    }

    #[test]
    fn boundary_delta_rejects_late_system_prompt_mutation() {
        let original_tail = "a".repeat(300);
        let mutated_tail = format!("{}b", "a".repeat(299));
        let retained =
            format!("<|im_start|>system\n{original_tail}<|im_end|>\n<|im_start|>user\nq");
        let full =
            format!("<|im_start|>system\n{mutated_tail}<|im_end|>\n<|im_start|>user\nq<|im_end|>");
        assert!(message_boundary_delta(&retained, &full).is_none());
    }

    #[test]
    fn boundary_delta_rejects_template_marker_inside_message_content() {
        let retained = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nliteral <|im_start|>user\nnot a boundary<|im_end|>";
        let full = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nliteral <|im_start|>user\nnot a boundary<|im_end|>\n<|im_start|>assistant\n";
        assert!(message_boundary_delta(retained, full).is_none());
    }

    #[test]
    fn boundary_delta_rejects_render_with_fewer_messages() {
        // Render has fewer message closers than retained covers (history
        // shrank): no splice.
        let retained = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\nans";
        let full = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>";
        assert!(message_boundary_delta(retained, full).is_none());
    }

    #[test]
    fn boundary_delta_accepts_equivalent_tool_call_replay() {
        let retained = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nread it<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=read_file>\n<parameter=path>\nCargo.toml\n</parameter>\n</function>\n</tool_call>";
        let full = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nread it<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"arguments\":{\"path\":\"Cargo.toml\"},\"name\":\"read_file\"}\n</tool_call><|im_end|>\n<|im_start|>tool\n[workspace]\n<|im_end|>\n<|im_start|>assistant\n<think>";

        let delta = message_boundary_delta(retained, full).unwrap();

        assert_eq!(
            delta,
            "<|im_end|>\n<|im_start|>tool\n[workspace]\n<|im_end|>\n<|im_start|>assistant\n<think>"
        );
    }

    #[test]
    fn session_prompt_trace_reports_boundary_splice_and_tool_payload() {
        let retained = [1, 2, 3];
        let canonical = [9, 9, 9, 4, 5, 6];
        let candidate = [1, 2, 3, 4, 5, 6];
        let tool_payload = ToolPayloadStats {
            messages: 2,
            bytes: 100,
            largest_bytes: 80,
        };

        let trace = session_prompt_trace_metrics(
            Some(&retained),
            &canonical,
            &candidate,
            tool_payload,
            SessionPromptTraceOutcome::Continued,
        );

        assert_eq!(trace.common_prefix_tokens, retained.len());
        assert_eq!(trace.divergence_token, None);
        assert_eq!(trace.suffix_tokens, 3);
        assert!(trace.boundary_splice);
        assert_eq!(trace.tool_result_messages, 2);
        assert_eq!(trace.tool_result_bytes, 100);
        assert_eq!(trace.tool_result_largest_bytes, 80);
    }

    #[test]
    fn session_prompt_trace_reports_prefix_mismatch() {
        let retained = [1, 2, 3];
        let candidate = [1, 2, 4, 5];
        let trace = session_prompt_trace_metrics(
            Some(&retained),
            &candidate,
            &candidate,
            ToolPayloadStats::default(),
            SessionPromptTraceOutcome::StatelessPflashBootstrap,
        );

        assert_eq!(trace.common_prefix_tokens, 2);
        assert_eq!(trace.divergence_token, Some(2));
        assert_eq!(trace.suffix_tokens, candidate.len());
        assert!(!trace.boundary_splice);
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
}
