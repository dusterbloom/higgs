//! Batch engine with interleaved request processing.
//!
//! Unlike [`SimpleEngine`](crate::simple::SimpleEngine) which serializes requests
//! through a mutex, `BatchEngine` runs a dedicated background loop that interleaves
//! decode steps across multiple active requests. Each request gets one token per
//! iteration, so concurrent clients see tokens as soon as possible rather than
//! waiting for prior requests to fully complete.

use std::path::Path;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

use higgs_models::{
    AnyCache, AnyModel, LogprobArrays, SamplingParams, apply_penalties, sample,
    turboquant::KvCacheConfig,
    vision::{ImageBatch, ImageInput, VisionCapabilities, VisionError, VisionModel},
};
use mlx_rs::{
    Array, Stream,
    ops::indexing::{IndexOp, NewAxis},
    transforms::{async_eval, eval},
    with_new_default_stream,
};
use tokenizers::Tokenizer;

use crate::{
    chat_template::{ChatMessage, ChatTemplateRenderer},
    engine::{GenerationOutput, StreamingOutput},
    error::EngineError,
    model_loader,
    prompt_cache::PrefixCache,
    simple::{IncrementalDetok, find_stop_in_tail},
};

/// Default maximum number of cached prefixes.
const DEFAULT_PREFIX_CACHE_SIZE: usize = 8;

/// Maximum number of pending requests in the submission queue.
const REQUEST_QUEUE_CAPACITY: usize = 128;
static NEXT_PREFILL_ID: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// A generation request submitted to the batch engine.
struct BatchRequest {
    prompt_tokens: Vec<u32>,
    max_tokens: u32,
    params: SamplingParams,
    stop_sequences: Vec<String>,
    logprobs: bool,
    top_logprobs: Option<u32>,
    constraint: Option<crate::constrained::ConstrainedGenerator>,
    /// Raw decoded images for multimodal requests. The worker preprocesses
    /// them (inside [`start_prefill`], where the `AnyModel` lives) into an
    /// `ImageBatch` and expands the family marker tokens in the prompt. When
    /// present, the prompt is prefilled in a single merged-embedding forward
    /// (image features cannot span chunk boundaries) and the prefix cache is
    /// bypassed.
    image_inputs: Vec<ImageInput>,
    response_tx: tokio::sync::mpsc::Sender<StreamingOutput>,
}

/// An in-flight request being actively decoded.
struct ActiveRequest {
    cache: AnyCache,
    current_token: Array,
    generated_tokens: Vec<u32>,
    max_tokens: u32,
    params: SamplingParams,
    stop_sequences: Vec<String>,
    logprob_top_n: Option<u32>,
    constraint: Option<crate::constrained::ConstrainedGenerator>,
    response_tx: tokio::sync::mpsc::Sender<StreamingOutput>,
    prompt_len: u32,
    detok: IncrementalDetok,
}

/// A request whose prompt is being fed through the model over multiple turns
/// of the worker loop.
struct PendingPrefill {
    request_id: u64,
    req: BatchRequest,
    prompt_len: u32,
    tokens: Vec<u32>,
    offset: usize,
    cache: AnyCache,
    /// Worker-produced `ImageBatch` for multimodal requests (see
    /// [`start_prefill`]); `None` for text-only requests.
    image_batch: Option<ImageBatch>,
}

enum PrefillAdvance {
    InFlight(PendingPrefill),
    Complete(Option<ActiveRequest>),
}

#[cfg(test)]
fn prefill_chunk_ranges(token_count: usize, quantum: usize) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < token_count {
        let end = start.saturating_add(quantum).min(token_count);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

// ---------------------------------------------------------------------------
// BatchEngine
// ---------------------------------------------------------------------------

/// Inference engine with interleaved request processing.
///
/// Provides the same public API as `SimpleEngine` but runs all inference on a
/// dedicated background thread. Concurrent requests are interleaved at the
/// token level instead of being fully serialized.
pub struct BatchEngine {
    request_tx: tokio::sync::mpsc::Sender<BatchRequest>,
    tokenizer: Tokenizer,
    template: ChatTemplateRenderer,
    model_name: String,
    eos_token_ids: Vec<u32>,
    hidden_size: AtomicI32,
    /// Whether the loaded model is a vision-language model. Captured at load
    /// time because the `AnyModel` is moved into the worker thread and the
    /// handle cannot lock it afterwards.
    is_vlm: bool,
    /// The family marker text injected at each image position before
    /// tokenization, if the loaded model supports vision.
    image_marker_text: Option<&'static str>,
    /// Capability metadata for the loaded model, if it supports vision.
    vision_capabilities: Option<VisionCapabilities>,
}

impl BatchEngine {
    /// Load a model and start the background processing loop.
    pub fn load<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        raise_wired_limit: bool,
        prefill_yield_tokens: Option<u32>,
        disable_vision: bool,
    ) -> Result<Self, EngineError> {
        if kv_cache_config.is_turboquant() {
            return Err(EngineError::Generation(
                "TurboQuant is not supported with the batch engine".to_owned(),
            ));
        }
        let model_dir = dir.as_ref();
        let model_name = crate::simple::derive_model_name(model_dir);

        tracing::info!(model_dir = %model_dir.display(), "Loading model (batch engine)");

        let model = model_loader::load_model(model_dir, disable_vision)?;
        let tokenizer = model_loader::load_tokenizer(model_dir)?;
        let template = ChatTemplateRenderer::from_model_dir(model_dir)?;
        let eos_token_ids = crate::simple::extract_eos_tokens(model_dir);
        let hidden_size = model.hidden_size();

        // Capture vision metadata before the model is moved into the worker
        // thread; the handle needs it for route-level capability gating and
        // marker rendering but can never lock the model afterwards.
        let (is_vlm, image_marker_text, vision_capabilities) =
            capture_vision_capabilities(model.as_vision());

        crate::simple::set_wired_limit_to_max(raise_wired_limit);

        let (request_tx, request_rx) = tokio::sync::mpsc::channel(REQUEST_QUEUE_CAPACITY);
        let eos_ids = eos_token_ids.clone();
        let tok = tokenizer.clone();

        std::thread::Builder::new()
            .name("batch-engine".into())
            .spawn(move || {
                worker_loop(
                    model,
                    &tok,
                    &eos_ids,
                    request_rx,
                    prefill_yield_tokens.unwrap_or(0),
                );
            })
            .map_err(|e| EngineError::Generation(format!("Failed to spawn worker: {e}")))?;

        tracing::info!(
            model_name = %model_name,
            eos_tokens = ?eos_token_ids,
            "Batch engine ready"
        );

        Ok(Self {
            request_tx,
            tokenizer,
            template,
            model_name,
            eos_token_ids,
            hidden_size: AtomicI32::new(hidden_size),
            is_vlm,
            image_marker_text,
            vision_capabilities,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub const fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn hidden_size(&self) -> i32 {
        self.hidden_size.load(Ordering::Relaxed)
    }

    /// Whether the loaded model is a vision-language model.
    pub const fn is_vlm(&self) -> bool {
        self.is_vlm
    }

    /// The marker text injected at each image position before tokenization,
    /// if the loaded model supports vision.
    pub const fn image_marker_text(&self) -> Option<&'static str> {
        self.image_marker_text
    }

    /// Capability metadata for the loaded model, if it supports vision.
    pub fn vision_capabilities(&self) -> Option<VisionCapabilities> {
        self.vision_capabilities.clone()
    }

    /// Apply chat template and tokenize messages.
    pub fn prepare_chat_prompt_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        _enable_thinking: bool,
    ) -> Result<Vec<u32>, EngineError> {
        self.prepare_chat_prompt(messages, tools)
    }

    /// Apply chat template and tokenize messages.
    pub fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>, EngineError> {
        let prompt = self.template.apply(messages, tools, true)?;
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| EngineError::Tokenization(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Generate a complete (non-streaming) response.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
    ) -> Result<GenerationOutput, EngineError> {
        self.generate_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            false,
            constraint,
            image_inputs,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn generate_with_thinking(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        _enable_thinking: bool,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
    ) -> Result<GenerationOutput, EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }

        let prompt_len: u32 = prompt_tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;

        if max_tokens == 0 {
            return Ok(GenerationOutput {
                text: String::new(),
                finish_reason: "length".to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 0,
                token_logprobs: None,
            });
        }

        // Submit request and collect all streaming outputs.
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::channel(32);

        self.request_tx
            .blocking_send(BatchRequest {
                prompt_tokens: prompt_tokens.to_vec(),
                max_tokens,
                params: params.clone(),
                stop_sequences: stop_sequences.to_vec(),
                logprobs,
                top_logprobs,
                constraint,
                image_inputs: image_inputs.unwrap_or_default(),
                response_tx: internal_tx,
            })
            .map_err(|_| EngineError::Generation("Engine shut down".to_owned()))?;

        let mut full_text = String::new();
        let mut finish_reason = "length".to_owned();
        let mut completion_tokens: u32 = 0;
        let mut all_logprobs: Option<Vec<higgs_models::TokenLogprobInfo>> = logprobs.then(Vec::new);
        // The worker reports the post-preprocess prompt length on every chunk;
        // capture it from the first one so multimodal usage matches the
        // sentinel-expanded token count actually prefilled.
        let mut reported_prompt_len: Option<u32> = None;

        while let Some(output) = internal_rx.blocking_recv() {
            // Worker-sent error chunks (prefill/decode failures) fail the
            // request instead of surfacing as an empty success.
            if matches!(
                output.finish_reason.as_deref(),
                Some(WORKER_ERROR_FINISH | WORKER_ERROR_FINISH_VISION)
            ) {
                return Err(worker_error_from_output(&output));
            }
            if reported_prompt_len.is_none() {
                reported_prompt_len = Some(output.prompt_tokens);
            }
            full_text.push_str(&output.new_text);
            completion_tokens = output.completion_tokens;
            if let Some(ref reason) = output.finish_reason {
                finish_reason.clone_from(reason);
            }
            if let (Some(all_lp), Some(lp)) = (&mut all_logprobs, output.token_logprob) {
                all_lp.push(lp);
            }
            if output.finished {
                break;
            }
        }

        Ok(GenerationOutput {
            text: full_text,
            finish_reason,
            prompt_tokens: reported_prompt_len.unwrap_or(prompt_len),
            completion_tokens,
            token_logprobs: all_logprobs,
        })
    }

    /// Generate tokens one at a time via the provided channel.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn generate_streaming(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
    ) -> Result<(), EngineError> {
        self.generate_streaming_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            sender,
            false,
            false,
            constraint,
            image_inputs,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn generate_streaming_with_thinking(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        _enable_thinking: bool,
        // Batch engine does not emit prefill progress; accepts the flag to
        // share the streaming interface with SimpleEngine.
        _return_progress: bool,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
    ) -> Result<(), EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }

        let prompt_len: u32 = prompt_tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;

        if max_tokens == 0 {
            let _ = sender.blocking_send(StreamingOutput {
                new_text: String::new(),
                finished: true,
                finish_reason: Some("length".to_owned()),
                prompt_tokens: prompt_len,
                completion_tokens: 0,
                token_logprob: None,
                prefill_progress: None,
            });
            return Ok(());
        }

        // Submit request -- the background loop sends tokens directly to
        // an internal channel, and we forward them to the caller's sender.
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::channel(32);

        self.request_tx
            .blocking_send(BatchRequest {
                prompt_tokens: prompt_tokens.to_vec(),
                max_tokens,
                params: params.clone(),
                stop_sequences: stop_sequences.to_vec(),
                logprobs,
                top_logprobs,
                constraint,
                image_inputs: image_inputs.unwrap_or_default(),
                response_tx: internal_tx,
            })
            .map_err(|_| EngineError::Generation("Engine shut down".to_owned()))?;

        while let Some(output) = internal_rx.blocking_recv() {
            // Worker-sent error chunks (prefill/decode failures) fail the
            // request; the caller surfaces the failure (the route sends an
            // error-finish chunk to the SSE stream).
            if matches!(
                output.finish_reason.as_deref(),
                Some(WORKER_ERROR_FINISH | WORKER_ERROR_FINISH_VISION)
            ) {
                return Err(worker_error_from_output(&output));
            }
            let finished = output.finished;
            if sender.blocking_send(output).is_err() {
                break; // Client disconnected
            }
            if finished {
                break;
            }
        }

        Ok(())
    }

    /// Compute embeddings (delegates to a single forward pass).
    pub fn embed(&self, _token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        // Embeddings require direct model access. For now, return an error.
        // A proper implementation would submit an embed request to the worker.
        Err(EngineError::Generation(
            "Embeddings not yet supported in batch engine".to_owned(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Background worker loop
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn worker_loop(
    mut model: AnyModel,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    mut request_rx: tokio::sync::mpsc::Receiver<BatchRequest>,
    prefill_yield_tokens: u32,
) {
    let mut prefix_cache = PrefixCache::new(DEFAULT_PREFIX_CACHE_SIZE);
    let mut active: Vec<ActiveRequest> = Vec::new();
    let mut pending_prefill: Option<PendingPrefill> = None;

    loop {
        // 1. Prefill at most one pending request per iteration.
        //    This interleaves prefill with decode so long prefills don't
        //    starve active requests, keeping TTFT low for new arrivals
        //    while maintaining token throughput for in-flight requests.
        if pending_prefill.is_none() {
            if let Ok(req) = request_rx.try_recv() {
                if prefill_yield_tokens == 0 {
                    let response_tx = req.response_tx.clone();
                    match prefill_request(
                        &mut model,
                        &mut prefix_cache,
                        tokenizer,
                        eos_token_ids,
                        req,
                    ) {
                        Ok(Some(ar)) => active.push(ar),
                        Ok(None) => {}
                        Err(e) => {
                            send_prefill_error(&response_tx, &e);
                            tracing::error!(error = %e, "Prefill failed");
                        }
                    }
                } else {
                    let response_tx = req.response_tx.clone();
                    match start_prefill(&model, tokenizer, &mut prefix_cache, req) {
                        Ok(prefill) => pending_prefill = Some(prefill),
                        Err(e) => {
                            send_prefill_error(&response_tx, &e);
                            tracing::error!(error = %e, "Prefill failed");
                        }
                    }
                }
            }
        }

        if let Some(prefill) = pending_prefill.take() {
            let response_tx = prefill.req.response_tx.clone();
            match advance_prefill(
                &mut model,
                &mut prefix_cache,
                tokenizer,
                eos_token_ids,
                prefill,
                prefill_yield_tokens,
            ) {
                Ok(PrefillAdvance::InFlight(resumed_prefill)) => {
                    pending_prefill = Some(resumed_prefill);
                }
                Ok(PrefillAdvance::Complete(Some(ar))) => active.push(ar),
                Ok(PrefillAdvance::Complete(None)) => {}
                Err(e) => {
                    send_prefill_error(&response_tx, &e);
                    tracing::error!(error = %e, "Prefill failed");
                }
            }
        }

        // 2. If no active requests, block until one arrives.
        if active.is_empty() && pending_prefill.is_none() {
            if let Some(req) = request_rx.blocking_recv() {
                if prefill_yield_tokens == 0 {
                    let response_tx = req.response_tx.clone();
                    match prefill_request(
                        &mut model,
                        &mut prefix_cache,
                        tokenizer,
                        eos_token_ids,
                        req,
                    ) {
                        Ok(Some(ar)) => active.push(ar),
                        Ok(None) => continue,
                        Err(e) => {
                            send_prefill_error(&response_tx, &e);
                            tracing::error!(error = %e, "Prefill failed");
                            continue;
                        }
                    }
                } else {
                    let response_tx = req.response_tx.clone();
                    match start_prefill(&model, tokenizer, &mut prefix_cache, req) {
                        Ok(prefill) => pending_prefill = Some(prefill),
                        Err(e) => {
                            send_prefill_error(&response_tx, &e);
                            tracing::error!(error = %e, "Prefill failed");
                        }
                    }
                    continue;
                }
            } else {
                tracing::info!("Request channel closed, worker shutting down");
                return;
            }
        }

        // 3. Run one decode step per active request.
        //    Use batched decode when possible (Transformer architecture, >1
        //    request, no constrained generation). Falls back to pipelined
        //    sequential decode otherwise.
        let mut finished_indices = Vec::new();
        let use_batched = active.len() > 1
            && model.supports_batched_decode()
            && active.iter().all(|ar| ar.constraint.is_none());

        if use_batched {
            if let Err(e) = with_new_default_stream(Stream::new(), || {
                run_batched_decode_round(
                    &mut model,
                    &mut active,
                    tokenizer,
                    eos_token_ids,
                    &mut finished_indices,
                )
            }) {
                tracing::error!(error = %e, "Batched decode round failed");
                for (i, ar) in active.iter().enumerate() {
                    let _ = ar.response_tx.blocking_send(StreamingOutput {
                        new_text: format!("decode failed: {e}"),
                        finished: true,
                        finish_reason: Some("error".to_owned()),
                        prompt_tokens: ar.prompt_len,
                        completion_tokens: 0,
                        token_logprob: None,
                        prefill_progress: None,
                    });
                    finished_indices.push(i);
                }
            }
        } else {
            with_new_default_stream(Stream::new(), || {
                run_pipelined_decode_round(
                    &mut model,
                    &mut active,
                    tokenizer,
                    eos_token_ids,
                    &mut finished_indices,
                );
            });
        }

        // 4. Remove finished requests (reverse order to preserve indices).
        for i in finished_indices.into_iter().rev() {
            active.swap_remove(i);
        }
    }
}

/// Pipelined decode: build each request's computation graph with `async_eval`,
/// then materialize all results. GPU processes request N while CPU builds
/// request N+1's graph.
fn run_pipelined_decode_round(
    model: &mut AnyModel,
    active: &mut [ActiveRequest],
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    finished_indices: &mut Vec<usize>,
) {
    // Phase 1: Build computation graphs and submit to GPU.
    let mut graphs: Vec<Option<DecodeGraphResult>> = Vec::with_capacity(active.len());
    for (i, ar) in active.iter_mut().enumerate() {
        match build_decode_graph(model, ar) {
            Ok(result) => {
                let mut eval_targets: Vec<&Array> = vec![&result.next_token];
                if let Some(ref lp) = result.logprob_data {
                    eval_targets.extend(lp.eval_targets());
                }
                match async_eval(eval_targets) {
                    Ok(()) => graphs.push(Some(result)),
                    Err(e) => {
                        tracing::error!(error = %e, "async_eval failed");
                        let _ = ar.response_tx.blocking_send(StreamingOutput {
                            new_text: format!("decode step failed: {e}"),
                            finished: true,
                            finish_reason: Some("error".to_owned()),
                            prompt_tokens: ar.prompt_len,
                            completion_tokens: 0,
                            token_logprob: None,
                            prefill_progress: None,
                        });
                        finished_indices.push(i);
                        graphs.push(None);
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "Decode graph build failed");
                let _ = ar.response_tx.blocking_send(StreamingOutput {
                    new_text: format!("decode graph build failed: {e}"),
                    finished: true,
                    finish_reason: Some("error".to_owned()),
                    prompt_tokens: ar.prompt_len,
                    completion_tokens: 0,
                    token_logprob: None,
                    prefill_progress: None,
                });
                finished_indices.push(i);
                graphs.push(None);
            }
        }
    }

    // Phase 2: Materialize results.
    for (i, (ar, graph)) in active.iter_mut().zip(graphs).enumerate() {
        let Some(result) = graph else { continue };
        if materialize_decode_step(ar, result, tokenizer, eos_token_ids) {
            finished_indices.push(i);
        }
    }
}

/// Batched decode: single forward pass for all active requests.
///
/// Stacks N single-token inputs into `[N, 1]`, runs one batched forward pass,
/// then slices logits per-request for individual sampling.
#[allow(clippy::indexing_slicing)]
fn run_batched_decode_round(
    model: &mut AnyModel,
    active: &mut [ActiveRequest],
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    finished_indices: &mut Vec<usize>,
) -> Result<(), EngineError> {
    let n = active.len();

    // Stack all current tokens into [N, 1]
    let token_arrays: Vec<Array> = active
        .iter()
        .map(|ar| ar.current_token.index((.., NewAxis)))
        .collect();
    let token_refs: Vec<&Array> = token_arrays.iter().collect();
    let batched_input = mlx_rs::ops::concatenate_axis(&token_refs, 0).map_err(EngineError::Mlx)?;

    // Collect mutable cache references
    let mut cache_refs: Vec<&mut higgs_models::AnyCache> =
        active.iter_mut().map(|ar| &mut ar.cache).collect();

    // Batched forward pass: [N, 1] -> [N, 1, vocab_size]
    let batched_logits = model
        .forward_batched(&batched_input, &mut cache_refs)
        .map_err(EngineError::Mlx)?;
    let batched_last = batched_logits.index((.., -1, ..)); // [N, vocab_size]

    // Per-request: apply penalties, sample, compute logprobs, async_eval
    let mut results: Vec<Option<DecodeGraphResult>> = Vec::with_capacity(n);
    for (i, ar) in active.iter_mut().enumerate() {
        let idx = i32::try_from(i).unwrap_or(i32::MAX);
        let req_logits = batched_last.index((idx..idx + 1, ..));

        let penalized = apply_penalties(&req_logits, &ar.generated_tokens, &ar.params)
            .map_err(EngineError::Mlx)?;

        let next_token = sample(&penalized, &ar.params).map_err(EngineError::Mlx)?;

        let logprob_data = if let Some(top_n) = ar.logprob_top_n {
            let scaled = if ar.params.temperature <= f32::EPSILON {
                penalized
            } else {
                penalized
                    .multiply(mlx_rs::array!(1.0 / ar.params.temperature))
                    .map_err(EngineError::Mlx)?
            };
            Some(
                LogprobArrays::compute(&scaled, &next_token, Some(top_n))
                    .map_err(EngineError::Mlx)?,
            )
        } else {
            None
        };

        let mut eval_targets: Vec<&Array> = vec![&next_token];
        if let Some(ref lp) = logprob_data {
            eval_targets.extend(lp.eval_targets());
        }
        match async_eval(eval_targets) {
            Ok(()) => results.push(Some(DecodeGraphResult {
                next_token,
                logprob_data,
            })),
            Err(e) => {
                tracing::error!(error = %e, "async_eval failed in batched decode");
                finished_indices.push(i);
                results.push(None);
            }
        }
    }

    // Materialize all results
    for (i, (ar, result)) in active.iter_mut().zip(results).enumerate() {
        let Some(r) = result else { continue };
        if materialize_decode_step(ar, r, tokenizer, eos_token_ids) {
            finished_indices.push(i);
        }
    }

    Ok(())
}

/// Extract the vision metadata the HTTP layer needs from a loaded model's
/// vision implementation (if any), before the model is moved into the worker
/// thread. Returns `(is_vlm, image_marker_text, vision_capabilities)`.
fn capture_vision_capabilities(
    vision: Option<&dyn VisionModel>,
) -> (bool, Option<&'static str>, Option<VisionCapabilities>) {
    let Some(v) = vision else {
        return (false, None, None);
    };
    (
        true,
        Some(v.image_marker_text()),
        Some(v.vision_capabilities()),
    )
}

/// Preprocess a multimodal request inside the worker thread.
///
/// Decodes and preprocesses the raw [`ImageInput`]s into a family-native
/// [`ImageBatch`], then expands each family marker token in the prompt into
/// the sentinel run the batch expects (so `sum(batch.per_image_tokens)`
/// matches the sentinel count required by the embedding merge). The raw
/// inputs are consumed; the produced batch is returned for the single-pass
/// multimodal forward. Text-only requests pass through unchanged (`None`).
fn preprocess_images_in_worker(
    vision: Option<&dyn VisionModel>,
    tokenizer: &Tokenizer,
    req: &mut BatchRequest,
) -> Result<Option<ImageBatch>, EngineError> {
    if req.image_inputs.is_empty() {
        return Ok(None);
    }
    let v = vision.ok_or_else(|| {
        EngineError::Vision(VisionError::Preprocess(
            "model has no vision; cannot process images".to_owned(),
        ))
    })?;
    // Take the raw inputs so the (potentially large) decoded image bytes do
    // not linger on the request for the whole generation.
    let inputs = std::mem::take(&mut req.image_inputs);
    let batch = v.preprocess_images(&inputs)?;
    let mut tokens = std::mem::take(&mut req.prompt_tokens);
    v.postprocess_image_tokens(&mut tokens, tokenizer, &batch)?;
    req.prompt_tokens = tokens;
    Ok(Some(batch))
}

/// `finish_reason` sent by the worker on prefill failure; the error message
/// rides in `new_text` (the generate tails convert the chunk back into an
/// `Err`). The `:vision` form marks client-caused image preprocessing
/// failures so the tail reconstructs [`EngineError::Vision`] and the route
/// maps it to a strict 400.
const WORKER_ERROR_FINISH: &str = "error";
/// See [`WORKER_ERROR_FINISH`].
const WORKER_ERROR_FINISH_VISION: &str = "error:vision";

/// Send an error-finish chunk to a failed request's response channel so the
/// generate tail converts it into an `Err` — the client sees a failure
/// instead of an empty success. The message rides in `new_text`.
fn send_prefill_error(tx: &tokio::sync::mpsc::Sender<StreamingOutput>, error: &EngineError) {
    let (finish_reason, message) = match error {
        EngineError::Vision(_) => (WORKER_ERROR_FINISH_VISION, error.to_string()),
        EngineError::Model(_)
        | EngineError::Mlx(_)
        | EngineError::Tokenization(_)
        | EngineError::Template(_)
        | EngineError::Generation(_) => (WORKER_ERROR_FINISH, error.to_string()),
    };
    let _ = tx.blocking_send(StreamingOutput {
        new_text: message,
        finished: true,
        finish_reason: Some(finish_reason.to_owned()),
        prompt_tokens: 0,
        completion_tokens: 0,
        token_logprob: None,
        prefill_progress: None,
    });
}

/// Convert a worker-sent error chunk back into an [`EngineError`] so the
/// generate tails can fail the request (see [`send_prefill_error`]).
fn worker_error_from_output(output: &StreamingOutput) -> EngineError {
    match output.finish_reason.as_deref() {
        Some(WORKER_ERROR_FINISH_VISION) => {
            EngineError::Vision(VisionError::Preprocess(output.new_text.clone()))
        }
        _ => EngineError::Generation(if output.new_text.is_empty() {
            "generation failed".to_owned()
        } else {
            output.new_text.clone()
        }),
    }
}

/// Set up cache reuse once, before the prompt begins advancing through chunks.
fn start_prefill(
    model: &AnyModel,
    tokenizer: &Tokenizer,
    prefix_cache: &mut PrefixCache,
    mut req: BatchRequest,
) -> Result<PendingPrefill, EngineError> {
    // Multimodal requests are preprocessed here, in the worker thread, where
    // the model is owned: raw image inputs become an `ImageBatch` and marker
    // tokens expand into sentinels before the first forward pass.
    let image_batch = preprocess_images_in_worker(model.as_vision(), tokenizer, &mut req)?;
    // `prompt_len` reflects the post-preprocess token count so multimodal
    // usage reports match what the model actually prefilled (SimpleEngine
    // reports the same expanded length).
    let prompt_len = req
        .prompt_tokens
        .len()
        .try_into()
        .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;
    // Multimodal requests never reuse a prefix: image features are merged into
    // the embedding sequence, so the cached text-only KV/SSM state would not
    // match the prompt being prefilled.
    let prefix_match = if image_batch.is_some() {
        None
    } else {
        prefix_cache.find_longest_prefix(&req.prompt_tokens)
    };
    let (tokens, cache) = if let Some(matched) = prefix_match {
        let suffix = req
            .prompt_tokens
            .get(matched.prefix_len..)
            .unwrap_or_default();
        if suffix.is_empty() {
            // We don't cache logits, so an exact match must be replayed.
            (
                req.prompt_tokens.clone(),
                model.make_cache().map_err(EngineError::Mlx)?,
            )
        } else {
            (suffix.to_vec(), matched.cache)
        }
    } else {
        (
            req.prompt_tokens.clone(),
            model.make_cache().map_err(EngineError::Mlx)?,
        )
    };

    Ok(PendingPrefill {
        request_id: NEXT_PREFILL_ID.fetch_add(1, Ordering::Relaxed),
        req,
        prompt_len,
        tokens,
        offset: 0,
        cache,
        image_batch,
    })
}

fn advance_prefill(
    model: &mut AnyModel,
    prefix_cache: &mut PrefixCache,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    mut prefill: PendingPrefill,
    quantum: u32,
) -> Result<PrefillAdvance, EngineError> {
    let start = prefill.offset;
    let end = start
        .saturating_add(usize::try_from(quantum).unwrap_or(usize::MAX))
        .min(prefill.tokens.len());
    let is_complete = end == prefill.tokens.len();

    with_new_default_stream(Stream::new(), || {
        if let Some(batch) = &prefill.image_batch {
            // Single-pass multimodal prefill: image features are merged into
            // the embedding sequence and cannot span chunk boundaries, so the
            // whole prompt runs through one `forward_multimodal` call.
            let input = Array::from(prefill.tokens.as_slice()).index(NewAxis);
            let logits = model
                .forward_multimodal(&input, batch, &mut prefill.cache)
                .map_err(EngineError::Mlx)?;
            let last_logits = logits.index((.., -1, ..));
            prefill.offset = prefill.tokens.len();
            return complete_prefill(
                prefix_cache,
                tokenizer,
                eos_token_ids,
                prefill.req,
                prefill.prompt_len,
                prefill.cache,
                last_logits,
                true,
            )
            .map(PrefillAdvance::Complete);
        }

        let tokens = prefill
            .tokens
            .get(start..end)
            .ok_or_else(|| EngineError::Generation("Invalid prefill progress".to_owned()))?;
        let input = Array::from(tokens).index(NewAxis);
        let logits = model
            .forward(&input, None, &mut prefill.cache)
            .map_err(EngineError::Mlx)?;
        let last_logits = logits.index((.., -1, ..));
        prefill.offset = end;

        tracing::debug!(
            request_id = prefill.request_id,
            tokens_advanced = end - start,
            remaining = prefill.tokens.len() - end,
            "Prefill yield"
        );

        if !is_complete {
            eval([&last_logits]).map_err(EngineError::Mlx)?;
            return Ok(PrefillAdvance::InFlight(prefill));
        }

        complete_prefill(
            prefix_cache,
            tokenizer,
            eos_token_ids,
            prefill.req,
            prefill.prompt_len,
            prefill.cache,
            last_logits,
            false,
        )
        .map(PrefillAdvance::Complete)
    })
}

/// Prefill synchronously when yielding is disabled.
fn prefill_request(
    model: &mut AnyModel,
    prefix_cache: &mut PrefixCache,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    req: BatchRequest,
) -> Result<Option<ActiveRequest>, EngineError> {
    let prefill = start_prefill(model, tokenizer, prefix_cache, req)?;
    match advance_prefill(
        model,
        prefix_cache,
        tokenizer,
        eos_token_ids,
        prefill,
        u32::MAX,
    )? {
        PrefillAdvance::Complete(active) => Ok(active),
        PrefillAdvance::InFlight(_) => Err(EngineError::Generation(
            "Synchronous prefill did not complete".to_owned(),
        )),
    }
}

/// Finish a prefill after its final forward pass has produced logits.
///
/// `has_images` marks multimodal requests, which never enter the prefix cache
/// (their KV/SSM state reflects merged image features).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn complete_prefill(
    prefix_cache: &mut PrefixCache,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    req: BatchRequest,
    prompt_len: u32,
    cache: AnyCache,
    last_logits: Array,
    has_images: bool,
) -> Result<Option<ActiveRequest>, EngineError> {
    {
        let current_token = sample(&last_logits, &req.params).map_err(EngineError::Mlx)?;

        let logprob_top_n = req.logprobs.then(|| req.top_logprobs.unwrap_or(0));
        let first_logprob_data = if let Some(top_n) = logprob_top_n {
            let scaled = if req.params.temperature <= f32::EPSILON {
                last_logits
            } else {
                last_logits
                    .multiply(mlx_rs::array!(1.0 / req.params.temperature))
                    .map_err(EngineError::Mlx)?
            };
            Some(
                LogprobArrays::compute(&scaled, &current_token, Some(top_n))
                    .map_err(EngineError::Mlx)?,
            )
        } else {
            None
        };

        {
            let mut eval_targets: Vec<&Array> = vec![&current_token];
            if let Some(ref lp) = first_logprob_data {
                eval_targets.extend(lp.eval_targets());
            }
            eval(eval_targets).map_err(EngineError::Mlx)?;
        }

        // Cache the post-prefill state. Multimodal requests never enter the
        // prefix cache: their KV/SSM state reflects merged image features and
        // would not match a text-only prefix.
        if !has_images {
            prefix_cache.store(&req.prompt_tokens, cache.clone());
        }
        crate::simple::maybe_clear_mlx_cache(
            crate::simple::should_clear_mlx_cache_after_prefill(),
            "batch_post_prefill",
        );

        let first_token_id: u32 = current_token.item();
        let first_token_logprob = first_logprob_data
            .as_ref()
            .map(|lp| lp.materialize(first_token_id));

        // Decode the first token incrementally (routes through IncrementalDetok
        // so partial UTF-8 is held back and prefix-before-stop is correctly emitted).
        // Preserve content-bearing special tokens (e.g. MiniCPM tool-call markup)
        // while stripping control tokens, matching SimpleEngine's decode path.
        let skip_ids = std::sync::Arc::new(crate::simple::content_preserving_skip_ids(
            tokenizer,
            eos_token_ids,
        ));
        let mut detok = IncrementalDetok::new(String::new(), 0, skip_ids);
        let first_chunk = detok
            .append(tokenizer, &[first_token_id])
            .unwrap_or_default();
        let emitted_before = detok.text.len() - first_chunk.len();

        // Check if we're done after the first token
        let is_eos = eos_token_ids.contains(&first_token_id);
        let hit_stop = !req.stop_sequences.is_empty()
            && find_stop_in_tail(&detok.text, first_chunk.len(), &req.stop_sequences).is_some();
        let at_max = req.max_tokens <= 1;

        if is_eos || hit_stop || at_max {
            let finish_reason = if is_eos || hit_stop { "stop" } else { "length" };
            let mut send_text = if hit_stop {
                // Emit any prefix text before the stop sequence
                find_stop_in_tail(&detok.text, first_chunk.len(), &req.stop_sequences)
                    .and_then(|pos| detok.text.get(emitted_before..pos))
                    .unwrap_or_default()
                    .to_owned()
            } else {
                first_chunk
            };
            if !hit_stop {
                send_text.push_str(
                    &detok
                        .flush(tokenizer, &[first_token_id])
                        .unwrap_or_default(),
                );
            }
            let _ = req.response_tx.blocking_send(StreamingOutput {
                new_text: send_text,
                finished: true,
                finish_reason: Some(finish_reason.to_owned()),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprob: first_token_logprob,
                prefill_progress: None,
            });
            return Ok(None);
        }

        // Send first token
        if req
            .response_tx
            .blocking_send(StreamingOutput {
                new_text: first_chunk,
                finished: false,
                finish_reason: None,
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprob: first_token_logprob,
                prefill_progress: None,
            })
            .is_err()
        {
            return Ok(None); // Client disconnected
        }

        Ok(Some(ActiveRequest {
            cache,
            current_token,
            generated_tokens: vec![first_token_id],
            max_tokens: req.max_tokens,
            params: req.params,
            stop_sequences: req.stop_sequences,
            logprob_top_n,
            constraint: req.constraint,
            response_tx: req.response_tx,
            prompt_len,
            detok,
        }))
    }
}

/// Lazy arrays from building a decode step's computation graph.
struct DecodeGraphResult {
    next_token: Array,
    logprob_data: Option<LogprobArrays>,
}

/// Build the computation graph for one decode step without evaluating.
/// The caller should `async_eval` the result, then later call
/// `materialize_decode_step` to extract concrete token values.
fn build_decode_graph(
    model: &mut AnyModel,
    ar: &mut ActiveRequest,
) -> Result<DecodeGraphResult, EngineError> {
    let decode_input = ar.current_token.index((.., NewAxis));
    let logits = model
        .forward(&decode_input, None, &mut ar.cache)
        .map_err(EngineError::Mlx)?;
    let sliced = logits.index((.., -1, ..));

    let penalized =
        apply_penalties(&sliced, &ar.generated_tokens, &ar.params).map_err(EngineError::Mlx)?;

    let constrained = if let Some(ref cg) = ar.constraint {
        cg.apply_mask(&penalized).map_err(EngineError::Mlx)?
    } else {
        penalized
    };

    let next_token = sample(&constrained, &ar.params).map_err(EngineError::Mlx)?;

    let logprob_data = if let Some(top_n) = ar.logprob_top_n {
        let scaled = if ar.params.temperature <= f32::EPSILON {
            constrained
        } else {
            constrained
                .multiply(mlx_rs::array!(1.0 / ar.params.temperature))
                .map_err(EngineError::Mlx)?
        };
        Some(LogprobArrays::compute(&scaled, &next_token, Some(top_n)).map_err(EngineError::Mlx)?)
    } else {
        None
    };

    Ok(DecodeGraphResult {
        next_token,
        logprob_data,
    })
}

/// Materialize a decode step's results after `async_eval` has completed.
/// Returns `true` if the request is finished, `false` to continue.
fn materialize_decode_step(
    ar: &mut ActiveRequest,
    result: DecodeGraphResult,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
) -> bool {
    let token_id: u32 = result.next_token.item();

    // Advance constrained generator
    if let Some(ref mut cg) = ar.constraint {
        cg.advance(token_id);
    }

    let token_logprob = result
        .logprob_data
        .as_ref()
        .map(|lp| lp.materialize(token_id));

    ar.generated_tokens.push(token_id);
    ar.current_token = result.next_token;

    let completion_len: u32 = ar.generated_tokens.len().try_into().unwrap_or(u32::MAX);

    // Decode only the trailing token window for diff and stop checking
    let new_text = ar
        .detok
        .append(tokenizer, &ar.generated_tokens)
        .unwrap_or_default();
    let emitted_before = ar.detok.text.len() - new_text.len();

    let (mut final_new_text, hit_stop) = if ar.stop_sequences.is_empty() {
        (new_text, false)
    } else if let Some(pos) = find_stop_in_tail(&ar.detok.text, new_text.len(), &ar.stop_sequences)
    {
        let emit = ar
            .detok
            .text
            .get(emitted_before..pos)
            .unwrap_or_default()
            .to_owned();
        (emit, true)
    } else {
        (new_text, false)
    };

    let is_eos = eos_token_ids.contains(&token_id);
    let at_max = completion_len >= ar.max_tokens;
    let constraint_done = ar
        .constraint
        .as_ref()
        .is_some_and(crate::constrained::ConstrainedGenerator::is_finished);

    let finished = is_eos || at_max || hit_stop || constraint_done;
    if finished && !hit_stop {
        final_new_text.push_str(
            &ar.detok
                .flush(tokenizer, &ar.generated_tokens)
                .unwrap_or_default(),
        );
    }
    let finish_reason = if is_eos || hit_stop || constraint_done {
        Some("stop".to_owned())
    } else if at_max {
        Some("length".to_owned())
    } else {
        None
    };

    let disconnected = ar
        .response_tx
        .blocking_send(StreamingOutput {
            new_text: final_new_text,
            finished,
            finish_reason,
            prompt_tokens: ar.prompt_len,
            completion_tokens: completion_len,
            token_logprob,
            prefill_progress: None,
        })
        .is_err();

    finished || disconnected
}

#[allow(
    clippy::as_conversions,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]
#[cfg(test)]
mod tests {
    use super::*;
    use higgs_models::vision::{
        IMAGE_TOKEN_INDEX, ImageInput, ImageTokenLayout, ImageTokenLayoutKind, VisionCapabilities,
        VisionError, VisionModel,
    };
    use mlx_rs::error::Exception;

    // -----------------------------------------------------------------------
    // materialize_decode_step
    // -----------------------------------------------------------------------

    fn make_active_request(
        max_tokens: u32,
        stop_sequences: Vec<String>,
    ) -> (ActiveRequest, tokio::sync::mpsc::Receiver<StreamingOutput>) {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let ar = ActiveRequest {
            cache: AnyCache::KV(vec![]),
            current_token: Array::from_slice(&[0_u32], &[1]),
            generated_tokens: vec![],
            max_tokens,
            params: SamplingParams::default(),
            stop_sequences,
            logprob_top_n: None,
            constraint: None,
            response_tx: tx,
            prompt_len: 5,
            detok: IncrementalDetok::new(
                String::new(),
                0,
                std::sync::Arc::new(std::collections::HashSet::new()),
            ),
        };
        (ar, rx)
    }

    fn make_tokenizer() -> Tokenizer {
        Tokenizer::new(tokenizers::models::bpe::BPE::default())
    }

    #[test]
    fn materialize_decode_step_normal_token_returns_false() {
        let (mut ar, _rx) = make_active_request(100, vec![]);
        let tokenizer = make_tokenizer();
        let result = DecodeGraphResult {
            next_token: Array::from_slice(&[42_u32], &[1]),
            logprob_data: None,
        };
        let finished = materialize_decode_step(&mut ar, result, &tokenizer, &[0]);
        assert!(!finished, "Should not be finished after normal token");
        assert_eq!(ar.generated_tokens, vec![42]);
    }

    #[test]
    fn materialize_decode_step_eos_returns_true() {
        let (mut ar, _rx) = make_active_request(100, vec![]);
        let tokenizer = make_tokenizer();
        let eos_id = 50256_u32; // GPT-2 EOS
        let result = DecodeGraphResult {
            next_token: Array::from_slice(&[eos_id], &[1]),
            logprob_data: None,
        };
        let finished = materialize_decode_step(&mut ar, result, &tokenizer, &[eos_id]);
        assert!(finished, "Should be finished on EOS token");
    }

    #[test]
    fn materialize_decode_step_max_tokens_returns_true() {
        let (mut ar, _rx) = make_active_request(1, vec![]);
        let tokenizer = make_tokenizer();
        let result = DecodeGraphResult {
            next_token: Array::from_slice(&[42_u32], &[1]),
            logprob_data: None,
        };
        let finished = materialize_decode_step(&mut ar, result, &tokenizer, &[0]);
        assert!(finished, "Should be finished at max_tokens=1");
    }

    // -----------------------------------------------------------------------
    // Batched vs pipelined path selection
    // -----------------------------------------------------------------------

    #[test]
    fn batched_path_requires_multiple_requests() {
        // Single request should not use batched path
        let n_active = 1;
        let supports_batched = true;
        let all_unconstrained = true;
        let use_batched = n_active > 1 && supports_batched && all_unconstrained;
        assert!(!use_batched);
    }

    #[test]
    fn batched_path_requires_model_support() {
        let n_active = 4;
        let supports_batched = false;
        let all_unconstrained = true;
        let use_batched = n_active > 1 && supports_batched && all_unconstrained;
        assert!(!use_batched);
    }

    #[test]
    fn batched_path_disabled_with_constraints() {
        let n_active = 4;
        let supports_batched = true;
        let all_unconstrained = false;
        let use_batched = n_active > 1 && supports_batched && all_unconstrained;
        assert!(!use_batched);
    }

    #[test]
    fn batched_path_enabled_when_all_conditions_met() {
        let n_active = 4;
        let supports_batched = true;
        let all_unconstrained = true;
        let use_batched = n_active > 1 && supports_batched && all_unconstrained;
        assert!(use_batched);
    }

    #[test]
    fn prefill_chunk_ranges_cover_each_token_once() {
        let ranges = prefill_chunk_ranges(10, 4);
        assert_eq!(ranges, vec![0..4, 4..8, 8..10]);
    }

    // -----------------------------------------------------------------------
    // Vision capability capture
    // -----------------------------------------------------------------------

    /// A minimal `VisionModel` double so the pure capture/preprocess helpers
    /// can be unit-tested without loading real VLM weights (no vision-capable
    /// `AnyModel` is constructible from this crate with public APIs).
    struct TestVisionModel;

    impl VisionModel for TestVisionModel {
        fn vision_capabilities(&self) -> VisionCapabilities {
            VisionCapabilities {
                families: vec!["test-vision"],
                image_sizes: vec![16],
                supported_media: vec!["image/png"],
                layout_kind: ImageTokenLayoutKind::default(),
            }
        }

        fn image_marker_text(&self) -> &'static str {
            "<test-image>"
        }

        fn preprocess_images(&self, images: &[ImageInput]) -> Result<ImageBatch, VisionError> {
            Ok(ImageBatch {
                pixel_values: Array::from_slice(&[0.0_f32; 3], &[1, 1, 1, 3]),
                per_image_tokens: vec![2; images.len()],
                image_sizes: vec![(1, 1); images.len()],
                image_offsets: vec![],
                layout: ImageTokenLayout::default(),
            })
        }

        fn postprocess_image_tokens(
            &self,
            tokens: &mut Vec<u32>,
            _tokenizer: &Tokenizer,
            _batch: &ImageBatch,
        ) -> Result<(), VisionError> {
            // Marker token id 42 expands into two sentinels per image, the
            // count promised by `preprocess_images` above.
            let sentinel = IMAGE_TOKEN_INDEX as u32;
            let mut expanded = Vec::with_capacity(tokens.len());
            for &t in tokens.iter() {
                if t == 42 {
                    expanded.extend(std::iter::repeat_n(sentinel, 2));
                } else {
                    expanded.push(t);
                }
            }
            *tokens = expanded;
            Ok(())
        }

        fn forward_multimodal(
            &mut self,
            _input_ids: &Array,
            _batch: &ImageBatch,
            _cache: &mut AnyCache,
        ) -> Result<Array, Exception> {
            Err(Exception::custom("unused in unit test"))
        }
    }

    fn test_image_input() -> ImageInput {
        ImageInput {
            position: 0,
            message_index: 0,
            bytes: vec![0_u8; 8],
            media_type: "image/png".to_owned(),
            detail: higgs_models::vision::ImageDetail::Auto,
            max_dims: None,
        }
    }

    #[test]
    fn capture_vision_capabilities_extracts_vlm_metadata() {
        let vlm = TestVisionModel;
        let (is_vlm, marker, caps) = capture_vision_capabilities(Some(&vlm));
        assert!(is_vlm, "a loaded VLM must report vision");
        assert_eq!(marker, Some("<test-image>"));
        let caps_meta = caps.expect("VLM must expose vision capabilities");
        assert_eq!(caps_meta.families, vec!["test-vision"]);
        assert_eq!(caps_meta.image_sizes, vec![16]);
    }

    #[test]
    fn capture_vision_capabilities_reports_none_for_text_only_model() {
        let (is_vlm, marker, caps) = capture_vision_capabilities(None);
        assert!(!is_vlm, "a text-only model must not report vision");
        assert!(marker.is_none());
        assert!(caps.is_none());
    }

    // -----------------------------------------------------------------------
    // Worker-side multimodal preprocessing
    // -----------------------------------------------------------------------

    /// The worker preprocesses raw image inputs into an `ImageBatch` and
    /// expands marker tokens into the sentinel run the batch expects.
    #[test]
    fn worker_preprocess_produces_batch_and_expands_markers() {
        let tokenizer = make_tokenizer();
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let mut req = BatchRequest {
            prompt_tokens: vec![1, 2, 42, 3],
            max_tokens: 4,
            params: SamplingParams::default(),
            stop_sequences: vec![],
            logprobs: false,
            top_logprobs: None,
            constraint: None,
            response_tx: tx,
            image_inputs: vec![test_image_input()],
        };

        let batch = preprocess_images_in_worker(Some(&TestVisionModel), &tokenizer, &mut req)
            .unwrap()
            .expect("one image must produce a batch");
        assert_eq!(batch.per_image_tokens, vec![2]);
        let sentinel = IMAGE_TOKEN_INDEX as u32;
        assert_eq!(
            req.prompt_tokens,
            vec![1, 2, sentinel, sentinel, 3],
            "marker token must expand into the sentinel run"
        );
        assert!(
            req.image_inputs.is_empty(),
            "raw image inputs are consumed by preprocessing"
        );
    }

    /// Text-only requests skip preprocessing entirely.
    #[test]
    fn worker_preprocess_returns_none_without_images() {
        let tokenizer = make_tokenizer();
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let mut req = BatchRequest {
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 4,
            params: SamplingParams::default(),
            stop_sequences: vec![],
            logprobs: false,
            top_logprobs: None,
            constraint: None,
            response_tx: tx,
            image_inputs: vec![],
        };

        let batch =
            preprocess_images_in_worker(Some(&TestVisionModel), &tokenizer, &mut req).unwrap();
        assert!(batch.is_none(), "no images means no ImageBatch");
        assert_eq!(req.prompt_tokens, vec![1, 2, 3]);
    }

    // -----------------------------------------------------------------------
    // Multimodal prefill
    // -----------------------------------------------------------------------

    /// A multimodal request must bypass the prefix cache (the full prompt is
    /// re-prefilled so image features can be merged) and dispatch prefill to
    /// `forward_multimodal` instead of the chunked text forward.
    ///
    /// Since Task 14b, the worker preprocesses raw `ImageInput`s inside
    /// `start_prefill` (the model lives in the worker thread). This tiny
    /// model is text-only, so a multimodal request now fails at that
    /// preprocessing step — proving the request flows through the worker-side
    /// vision pipeline instead of being handed a pre-built `ImageBatch` with
    /// no model interaction.
    #[test]
    fn multimodal_batch_request_flows_through_worker_vision_preprocessing() {
        use higgs_models::qwen3_next::{Qwen3NextCausalLM, Qwen3NextModelArgs};

        // Tiny hybrid model (structure only; `make_cache` needs no weights).
        let args: Qwen3NextModelArgs = serde_json::from_str(
            r#"{
                "model_type": "qwen3_next",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 1e-06,
                "vocab_size": 128,
                "max_position_embeddings": 512,
                "full_attention_interval": 4,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 64,
                "moe_intermediate_size": 32,
                "norm_topk_prob": true
            }"#,
        )
        .unwrap();
        let model = AnyModel::Qwen3Next(Qwen3NextCausalLM::new(args).unwrap());

        // Prompt longer than MIN_PREFIX_LEN (16); seed the cache with a
        // 16-token prefix so a text request WOULD match it.
        let prompt: Vec<u32> = (0..24).collect();
        let mut prefix_cache = PrefixCache::new(8);
        let partial_cache = model.make_cache().unwrap();
        prefix_cache.store(&prompt[..16], partial_cache);

        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let make_req = |prompt_tokens: Vec<u32>, image_inputs: Vec<ImageInput>| BatchRequest {
            prompt_tokens,
            max_tokens: 4,
            params: SamplingParams::default(),
            stop_sequences: vec![],
            logprobs: false,
            top_logprobs: None,
            constraint: None,
            response_tx: tx.clone(),
            image_inputs,
        };
        let tokenizer = make_tokenizer();

        // Contrast: a text request with the same prompt reuses the cached
        // prefix and prefills only the suffix.
        let text_prefill = start_prefill(
            &model,
            &tokenizer,
            &mut prefix_cache,
            make_req(prompt.clone(), vec![]),
        )
        .unwrap();
        assert_eq!(
            text_prefill.tokens,
            prompt[16..].to_vec(),
            "text request should reuse the 16-token prefix cache"
        );

        // Multimodal: the worker must preprocess inside `start_prefill`. This
        // text-only model has no vision, so the request fails with a vision
        // preprocessing error rather than being prefilled without images.
        let Err(err) = start_prefill(
            &model,
            &tokenizer,
            &mut prefix_cache,
            make_req(prompt, vec![test_image_input()]),
        ) else {
            panic!("multimodal request on a text-only model must fail at preprocessing")
        };
        assert!(
            err.to_string().contains("has no vision"),
            "expected worker-side vision preprocessing failure, got: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // Worker prefill error protocol
    // -----------------------------------------------------------------------

    /// The worker's error-finish chunks must round-trip back into an
    /// `EngineError` in the generate tails, with vision failures typed as
    /// `EngineError::Vision` so the route can map them to 400s.
    #[test]
    fn worker_error_chunk_round_trips_into_engine_error() {
        let output = StreamingOutput {
            new_text: "image preprocessing failed: bad png".to_owned(),
            finished: true,
            finish_reason: Some(WORKER_ERROR_FINISH_VISION.to_owned()),
            prompt_tokens: 0,
            completion_tokens: 0,
            token_logprob: None,
            prefill_progress: None,
        };
        let err = worker_error_from_output(&output);
        assert!(
            matches!(err, EngineError::Vision(_)),
            "vision-marked chunk must reconstruct EngineError::Vision, got: {err}"
        );
        assert!(
            err.to_string().contains("bad png"),
            "error message must survive the round trip, got: {err}"
        );

        let generic = StreamingOutput {
            new_text: "generation failed: boom".to_owned(),
            finished: true,
            finish_reason: Some(WORKER_ERROR_FINISH.to_owned()),
            prompt_tokens: 0,
            completion_tokens: 0,
            token_logprob: None,
            prefill_progress: None,
        };
        let err = worker_error_from_output(&generic);
        assert!(
            matches!(err, EngineError::Generation(_)),
            "plain error chunk must reconstruct EngineError::Generation, got: {err}"
        );
        assert!(err.to_string().contains("boom"));

        // An empty message degrades to a readable generic message.
        let empty = StreamingOutput {
            new_text: String::new(),
            finished: true,
            finish_reason: Some(WORKER_ERROR_FINISH.to_owned()),
            prompt_tokens: 0,
            completion_tokens: 0,
            token_logprob: None,
            prefill_progress: None,
        };
        assert_eq!(
            worker_error_from_output(&empty).to_string(),
            "Generation error: generation failed"
        );
    }

    /// `send_prefill_error` emits the message and the vision marker so the
    /// tail can reconstruct the error.
    #[test]
    fn send_prefill_error_carries_message_and_vision_marker() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let vision_err = EngineError::Vision(VisionError::Decode("corrupt png".to_owned()));
        send_prefill_error(&tx, &vision_err);
        let chunk = rx.blocking_recv().unwrap();
        assert_eq!(
            chunk.finish_reason.as_deref(),
            Some(WORKER_ERROR_FINISH_VISION)
        );
        assert!(chunk.new_text.contains("corrupt png"));

        let gen_err = EngineError::Generation("prompt too long".to_owned());
        send_prefill_error(&tx, &gen_err);
        let chunk = rx.blocking_recv().unwrap();
        assert_eq!(chunk.finish_reason.as_deref(), Some(WORKER_ERROR_FINISH));
        assert!(chunk.new_text.contains("prompt too long"));
    }

    // -----------------------------------------------------------------------
    // Multimodal prefill dispatch + cache behaviour
    // -----------------------------------------------------------------------

    /// A multimodal `PendingPrefill` must dispatch its single prefill pass to
    /// `forward_multimodal` (not the chunked text forward), and `complete_prefill`
    /// must never store a multimodal post-prefill state into the prefix cache.
    #[test]
    fn multimodal_prefill_dispatches_to_forward_multimodal_and_skips_cache_store() {
        use higgs_models::qwen3_next::{Qwen3NextCausalLM, Qwen3NextModelArgs};
        use higgs_models::vision::ImageTokenLayout;

        let args: Qwen3NextModelArgs = serde_json::from_str(
            r#"{
                "model_type": "qwen3_next",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 1e-06,
                "vocab_size": 128,
                "max_position_embeddings": 512,
                "full_attention_interval": 4,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 64,
                "moe_intermediate_size": 32,
                "norm_topk_prob": true
            }"#,
        )
        .unwrap();
        let mut model = AnyModel::Qwen3Next(Qwen3NextCausalLM::new(args).unwrap());
        let tokenizer = make_tokenizer();
        let prompt: Vec<u32> = (0..24).collect();

        // --- Dispatch: a multimodal prefill must run `forward_multimodal`. ---
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let req = BatchRequest {
            prompt_tokens: prompt.clone(),
            max_tokens: 4,
            params: SamplingParams::default(),
            stop_sequences: vec![],
            logprobs: false,
            top_logprobs: None,
            constraint: None,
            response_tx: tx,
            image_inputs: vec![],
        };
        let dummy_batch = ImageBatch {
            pixel_values: Array::from_slice(&[0.0_f32; 3], &[1, 1, 1, 3]),
            per_image_tokens: vec![],
            image_sizes: vec![],
            image_offsets: vec![],
            layout: ImageTokenLayout::default(),
        };
        let mut prefix_cache = PrefixCache::new(8);
        let prefill = PendingPrefill {
            request_id: 1,
            req,
            prompt_len: 24,
            tokens: prompt.clone(),
            offset: 0,
            cache: model.make_cache().unwrap(),
            image_batch: Some(dummy_batch),
        };
        // A tiny quantum must still run ONE multimodal forward: the prefill
        // must dispatch to `forward_multimodal` (which errors on this
        // non-VLM stub) instead of the chunked text path.
        let Err(err) = advance_prefill(&mut model, &mut prefix_cache, &tokenizer, &[], prefill, 1)
        else {
            panic!("multimodal prefill must run forward_multimodal, not chunked forward")
        };
        assert!(
            err.to_string()
                .contains("does not support multimodal forward"),
            "expected forward_multimodal dispatch error, got: {err}"
        );

        // --- Cache: `complete_prefill` must not store multimodal state. ---
        // Seed a 16-token prefix so a text request WOULD match it; run
        // `complete_prefill` with the same 24-token prompt and observe whether
        // the cache gets extended to the full prompt.
        let run_complete = |has_images: bool| {
            let (tx, _rx) = tokio::sync::mpsc::channel(16);
            let req = BatchRequest {
                prompt_tokens: prompt.clone(),
                max_tokens: 4,
                params: SamplingParams::default(),
                stop_sequences: vec![],
                logprobs: false,
                top_logprobs: None,
                constraint: None,
                response_tx: tx,
                image_inputs: vec![],
            };
            let mut prefix_cache = PrefixCache::new(8);
            let partial_cache = model.make_cache().unwrap();
            prefix_cache.store(&prompt[..16], partial_cache);
            let last_logits = Array::from_slice(&[0.0_f32; 128], &[1, 128]);
            let cache = model.make_cache().unwrap();
            let result = complete_prefill(
                &mut prefix_cache,
                &tokenizer,
                &[],
                req,
                24,
                cache,
                last_logits,
                has_images,
            );
            if let Err(e) = result {
                panic!("complete_prefill must succeed: {e}");
            }
            let matched = prefix_cache
                .find_longest_prefix(&prompt)
                .expect("the seeded 16-token prefix must still match the 24-token prompt");
            matched.prefix_len
        };

        assert_eq!(
            run_complete(true),
            16,
            "multimodal prefill must NOT store its state into the prefix cache"
        );
        assert_eq!(
            run_complete(false),
            24,
            "text prefill stores the full prompt so the contrast is meaningful"
        );
    }
}
