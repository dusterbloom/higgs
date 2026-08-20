use std::path::Path;
use std::sync::Arc;

use higgs_engine::batch_engine::BatchEngine;
use higgs_engine::cache::DiskPrefixCacheConfig;
use higgs_engine::chat_template::{ChatMessage, ChatPromptMode};
use higgs_engine::engine::{GenerationOutput, StreamingOutput};
use higgs_engine::error::EngineError;
use higgs_engine::mlx_tuning::{MlxRuntimeTuning, resolve_runtime_tuning};
use higgs_engine::simple::{
    CacheStats, PFlashPromptPolicy, PrefillCompressionMode as EnginePrefillCompressionMode,
    SessionContinuationPolicy, SessionGeneration, SessionPromptTracePayloadStats,
    SessionStreamAcceptance, SimpleEngine,
};
use higgs_engine::tokenizers::Tokenizer;
use higgs_models::SamplingParams;
use higgs_models::turboquant::KvCacheConfig;
use higgs_models::vision::{ImageBatch, ImageInput, VisionCapabilities, VisionError};

use crate::config::{
    HiggsConfig, LocalConfig, ModelConfig, PrefillCompressionMode, resolved_model_supports_batch,
    validate_pflash_settings,
};
use crate::metrics::MetricsStore;
use crate::router::Router;

/// Process-wide GPU inference gate.
///
/// MLX's Metal backend keeps shared, non-stream-local state — notably the
/// output-array table mutated in `metal::CommandEncoder::set_output_array`. Two
/// co-resident models evaluating concurrently (each on its own `spawn_blocking`
/// thread, each under a fresh `with_new_default_stream(Stream::new())`) race on
/// that table and corrupt it → `EXC_BAD_ACCESS`/SIGSEGV inside
/// `set_output_array`. The per-engine `Mutex<AnyModel>` only serializes a single
/// model, not across the co-resident set (e.g. an SLM trio).
///
/// On a single-GPU host there is no real parallelism to lose, so all GPU eval is
/// serialized through this one gate. Held only for the duration of a
/// generate/embed call. NOTE: this also serializes concurrent requests to a
/// single `Batch` engine; if per-model batch interleaving is reintroduced, this
/// gate should be narrowed to cross-model boundaries.
static GPU_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub struct RouteTestStub {
    name: String,
    mutations: std::sync::atomic::AtomicU64,
    mutation_sequence: std::sync::Mutex<Vec<String>>,
    retained_sessions: std::sync::Mutex<std::collections::HashSet<u64>>,
}

#[cfg(test)]
impl RouteTestStub {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            mutations: std::sync::atomic::AtomicU64::new(0),
            mutation_sequence: std::sync::Mutex::new(Vec::new()),
            retained_sessions: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn record_mutation(&self) {
        self.mutations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn record_named_mutation(&self, mutation: String) {
        self.record_mutation();
        self.mutation_sequence.lock().unwrap().push(mutation);
    }

    fn route_session(&self, session_id: u64) -> bool {
        let continued = {
            let mut sessions = self.retained_sessions.lock().unwrap();
            let continued = sessions.contains(&session_id);
            sessions.insert(session_id);
            continued
        };
        let action = if continued { "continue" } else { "retain" };
        self.record_named_mutation(format!("{action}:{session_id}"));
        continued
    }

    fn retained_session_ids(&self) -> Vec<u64> {
        let mut ids: Vec<_> = self
            .retained_sessions
            .lock()
            .unwrap()
            .iter()
            .copied()
            .collect();
        ids.sort_unstable();
        ids
    }

    fn drop_session(&self, session_id: u64) -> bool {
        let dropped = self.retained_sessions.lock().unwrap().remove(&session_id);
        self.record_named_mutation(format!("drop:{session_id}"));
        dropped
    }

    fn lease_session(&self, session_id: u64, ttl_seconds: u32) -> bool {
        let retained = self.retained_sessions.lock().unwrap().contains(&session_id);
        if retained {
            self.record_named_mutation(format!("lease:{session_id}:{ttl_seconds}"));
        }
        retained
    }

    fn mutation_count(&self) -> u64 {
        self.mutations.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn mutation_sequence(&self) -> Vec<String> {
        self.mutation_sequence.lock().unwrap().clone()
    }
}

#[cfg(test)]
fn route_test_tokenizer() -> &'static Tokenizer {
    static TOKENIZER: std::sync::OnceLock<Tokenizer> = std::sync::OnceLock::new();
    TOKENIZER.get_or_init(|| {
        Tokenizer::from_bytes(
            br#"{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": null,
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "WordLevel",
                    "vocab": {"[UNK]": 0, "token": 7},
                    "unk_token": "[UNK]"
                }
            }"#,
        )
        .expect("valid route test tokenizer")
    })
}

/// Acquire the global GPU gate, recovering from poisoning so a panic mid-eval
/// cannot permanently wedge all inference.
fn gpu_gate() -> std::sync::MutexGuard<'static, ()> {
    GPU_GATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Unified engine interface wrapping either the simple (serialized) or batch
/// (interleaved) engine. Route handlers interact with this enum exclusively.
pub enum Engine {
    Simple(Box<SimpleEngine>),
    Batch(Box<BatchEngine>),
    #[cfg(test)]
    Stub(RouteTestStub),
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub fn load_simple<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        tuning: MlxRuntimeTuning,
        raise_wired_limit: bool,
        draft_model: Option<&Path>,
        prefill_drafter: Option<&Path>,
        prefill_compression: PrefillCompressionMode,
        prefill_keep_ratio: f32,
        prefill_threshold: usize,
        prefill_chunk: usize,
        prefill_avgpool: usize,
        prefill_lookahead: usize,
        prefill_score_mode: higgs_models::spec_prefill::PrefillScoreMode,
        prefill_exit_layer: usize,
        prefill_keep_ratio_max: f32,
        prefill_max_auto_prefill_ratio: f32,
        prefill_plan_cache: bool,
        prefill_plan_cache_entries: usize,
        prefill_suffix_identity_threshold: usize,
        session_max_suffix_prefill_tokens: usize,
        disk_cache_config: Option<DiskPrefixCacheConfig>,
    ) -> Result<Self, EngineError> {
        let prefill_compression = match prefill_compression {
            PrefillCompressionMode::Off => EnginePrefillCompressionMode::Off,
            PrefillCompressionMode::Auto => EnginePrefillCompressionMode::Auto,
            PrefillCompressionMode::Always => EnginePrefillCompressionMode::Always,
        };
        SimpleEngine::load_with_dflash(
            dir,
            kv_cache_config,
            tuning,
            raise_wired_limit,
            draft_model,
            disk_cache_config,
            prefill_drafter,
            prefill_compression,
            prefill_keep_ratio,
            prefill_threshold,
            prefill_chunk,
            prefill_avgpool,
            prefill_lookahead,
            prefill_score_mode,
            prefill_exit_layer,
            prefill_keep_ratio_max,
            prefill_max_auto_prefill_ratio,
            prefill_plan_cache,
            prefill_plan_cache_entries,
            prefill_suffix_identity_threshold,
            session_max_suffix_prefill_tokens,
        )
        .map(|e| Self::Simple(Box::new(e)))
    }

    pub fn load_batch<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        raise_wired_limit: bool,
    ) -> Result<Self, EngineError> {
        // The merged `BatchEngine::load` takes `prefill_yield_tokens` and
        // `disable_vision`; nightly keeps its 3-argument call shape, so both
        // stay at their defaults (`None` = synchronous prefill; `disable_vision`
        // is a documented no-op on nightly).
        BatchEngine::load(dir, kv_cache_config, raise_wired_limit, None, false)
            .map(|e| Self::Batch(Box::new(e)))
    }

    #[cfg(test)]
    pub fn test_stub(name: &str) -> Self {
        Self::Stub(RouteTestStub::new(name))
    }

    #[cfg(test)]
    pub(crate) fn route_test_mutations(&self) -> u64 {
        match self {
            Self::Stub(stub) => stub.mutation_count(),
            _ => 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn route_test_mutation_sequence(&self) -> Vec<String> {
        match self {
            Self::Stub(stub) => stub.mutation_sequence(),
            _ => Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn route_test_retained_sessions(&self) -> Vec<u64> {
        match self {
            Self::Stub(stub) => stub.retained_session_ids(),
            _ => Vec::new(),
        }
    }

    pub fn model_name(&self) -> &str {
        match self {
            Self::Simple(e) => e.model_name(),
            Self::Batch(e) => e.model_name(),
            #[cfg(test)]
            Self::Stub(stub) => stub.name(),
        }
    }

    #[cfg_attr(test, allow(clippy::unreachable))]
    pub fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Simple(e) => e.tokenizer(),
            Self::Batch(e) => e.tokenizer(),
            #[cfg(test)]
            Self::Stub(stub)
                if stub.name().starts_with("zero-prefix-")
                    || stub.name() == "blocking-required-post-admission-evicted"
                    || stub.name() == "session-prefill-render-spy" =>
            {
                route_test_tokenizer()
            }
            #[cfg(test)]
            Self::Stub(_) => panic!("Engine::test_stub has no tokenizer"),
        }
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        match self {
            Self::Simple(e) => e.eos_token_ids(),
            Self::Batch(e) => e.eos_token_ids(),
            #[cfg(test)]
            Self::Stub(_) => &[],
        }
    }

    pub fn hidden_size(&self) -> i32 {
        match self {
            Self::Simple(e) => e.hidden_size(),
            Self::Batch(e) => e.hidden_size(),
            #[cfg(test)]
            Self::Stub(_) => 0,
        }
    }

    pub fn enable_thinking(&self) -> bool {
        match self {
            Self::Simple(e) => e.enable_thinking(),
            Self::Batch(_) => false,
            #[cfg(test)]
            Self::Stub(_) => false,
        }
    }

    pub fn is_vlm(&self) -> bool {
        match self {
            Self::Simple(e) => e.is_vlm(),
            Self::Batch(e) => e.is_vlm(),
            #[cfg(test)]
            Self::Stub(_) => false,
        }
    }

    /// The marker text injected at each image position before tokenization.
    pub fn image_marker_text(&self) -> Option<&'static str> {
        match self {
            Self::Simple(e) => e.image_marker_text(),
            Self::Batch(e) => e.image_marker_text(),
            #[cfg(test)]
            Self::Stub(_) => None,
        }
    }

    /// Capability metadata for the loaded model, if it supports vision.
    pub fn vision_capabilities(&self) -> Option<VisionCapabilities> {
        match self {
            Self::Simple(e) => e.vision_capabilities(),
            Self::Batch(e) => e.vision_capabilities(),
            #[cfg(test)]
            Self::Stub(_) => None,
        }
    }

    /// Preprocess decoded images into a family-native [`ImageBatch`].
    ///
    /// Only the simple (serialized) engine preprocesses here; the batch engine
    /// preprocesses inside its worker thread (the model lives there), so its
    /// arm errors — the route must pass raw [`ImageInput`]s to `generate_*`
    /// instead.
    pub fn preprocess_images(&self, images: &[ImageInput]) -> Result<ImageBatch, VisionError> {
        match self {
            Self::Simple(e) => e.preprocess_images(images),
            Self::Batch(_) => Err(VisionError::Preprocess(
                "batch engine preprocesses images inside its worker; pass image_inputs to generate"
                    .to_owned(),
            )),
            #[cfg(test)]
            Self::Stub(_) => Err(VisionError::Preprocess("stub".to_owned())),
        }
    }

    /// Expand image marker tokens into the sentinel runs for `batch`.
    pub fn postprocess_image_tokens(
        &self,
        tokens: &mut Vec<u32>,
        batch: &ImageBatch,
    ) -> Result<(), VisionError> {
        match self {
            Self::Simple(e) => e.postprocess_image_tokens(tokens, batch),
            Self::Batch(_) => Ok(()),
            #[cfg(test)]
            Self::Stub(_) => Ok(()),
        }
    }

    pub fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>, EngineError> {
        match self {
            Self::Simple(e) => e.prepare_chat_prompt(messages, tools),
            Self::Batch(e) => e.prepare_chat_prompt(messages, tools),
            #[cfg(test)]
            Self::Stub(_) => Ok(Vec::new()),
        }
    }

    pub fn prepare_chat_prompt_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> Result<Vec<u32>, EngineError> {
        match self {
            Self::Simple(e) => {
                e.prepare_chat_prompt_with_thinking(messages, tools, enable_thinking)
            }
            Self::Batch(e) => e.prepare_chat_prompt_with_thinking(messages, tools, enable_thinking),
            #[cfg(test)]
            Self::Stub(_) => Ok(Vec::new()),
        }
    }

    pub fn prepare_chat_prompt_with_pflash_policy(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        mode: ChatPromptMode,
    ) -> Result<(Vec<u32>, PFlashPromptPolicy), EngineError> {
        match self {
            Self::Simple(e) => {
                e.prepare_chat_prompt_with_pflash_policy(messages, tools, enable_thinking, mode)
            }
            Self::Batch(e) => e
                .prepare_chat_prompt_for_mode(messages, tools, mode)
                .map(|tokens| (tokens, PFlashPromptPolicy::default())),
            #[cfg(test)]
            Self::Stub(stub) if stub.name() == "session-prefill-render-spy" => Ok((
                match mode {
                    ChatPromptMode::SessionPrefill => vec![7],
                    ChatPromptMode::Generation => vec![7, 8],
                },
                PFlashPromptPolicy::default(),
            )),
            #[cfg(test)]
            Self::Stub(stub) if stub.name() == "prompt-limit-mutation-spy" => {
                Ok((vec![1, 2, 3], PFlashPromptPolicy::default()))
            }
            #[cfg(test)]
            Self::Stub(stub) if stub.name().starts_with("zero-prefix-") => {
                Ok((vec![7], PFlashPromptPolicy::default()))
            }
            #[cfg(test)]
            Self::Stub(_) => Ok((Vec::new(), PFlashPromptPolicy::default())),
        }
    }

    /// Drop a retained per-session KV cache. Exact radix/disk prefix caches are
    /// independent and are intentionally left intact.
    pub fn drop_retained_session(&self, session_id: u64) -> bool {
        match self {
            Self::Simple(e) => e.drop_retained_session(session_id),
            Self::Batch(_) => false,
            #[cfg(test)]
            Self::Stub(stub) => {
                if stub.name() == "zero-prefix-accept" {
                    return stub.drop_session(session_id);
                }
                if stub.name() == "prompt-limit-mutation-spy" {
                    stub.record_mutation();
                }
                false
            }
        }
    }

    /// Confirm an idle-eviction lease only when the requested retained session exists.
    pub fn lease_retained_session(&self, session_id: u64, ttl_seconds: u32) -> bool {
        match self {
            Self::Simple(e) => e.lease_retained_session(
                session_id,
                std::time::Duration::from_secs(u64::from(ttl_seconds)),
            ),
            Self::Batch(_) => false,
            #[cfg(test)]
            Self::Stub(stub) => {
                if stub.name() == "zero-prefix-accept" {
                    return stub.lease_session(session_id, ttl_seconds);
                }
                if stub.name() == "prompt-limit-mutation-spy" {
                    stub.record_mutation();
                }
                false
            }
        }
    }

    pub fn retained_session_can_continue(&self, session_id: u64, prompt_tokens: &[u32]) -> bool {
        match self {
            Self::Simple(e) => e.retained_session_can_continue(session_id, prompt_tokens),
            Self::Batch(_) => false,
            #[cfg(test)]
            Self::Stub(stub) => stub.name() == "raw-accept-worker-reject",
        }
    }

    pub fn record_required_continuation_miss(&self) {
        if let Self::Simple(engine) = self {
            engine.record_required_continuation_miss();
        }
    }

    /// Cache-effectiveness snapshot for observability. Only the Simple engine
    /// has a cache-resident path; other variants report `None`.
    pub fn cache_stats(&self) -> Option<CacheStats> {
        match self {
            Self::Simple(e) => Some(e.cache_stats()),
            Self::Batch(_) => None,
            #[cfg(test)]
            Self::Stub(_) => None,
        }
    }

    /// Cache-resident multi-turn generation: prefill only the new suffix when
    /// the retained cache is an exact token-prefix of `prompt_tokens`, else a
    /// clean full prefill. Only the Simple engine supports this; other variants
    /// return an error so the caller can fall back to a normal generation.
    pub fn generate_continued(
        &self,
        session_id: u64,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
    ) -> Result<SessionGeneration, EngineError> {
        self.generate_continued_with_thinking(
            session_id,
            prompt_tokens,
            max_tokens,
            params,
            self.enable_thinking(),
        )
    }

    /// Cache-resident generation using the thinking mode already resolved for
    /// this request's chat template.
    pub fn generate_continued_with_thinking(
        &self,
        session_id: u64,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        enable_thinking: bool,
    ) -> Result<SessionGeneration, EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.generate_continued_with_thinking(
                session_id,
                prompt_tokens,
                max_tokens,
                params,
                enable_thinking,
            ),
            Self::Batch(_) => Err(EngineError::Generation(
                "session_id (continued generation) is only supported by the Simple engine"
                    .to_owned(),
            )),
            #[cfg(test)]
            Self::Stub(stub) => {
                if stub.name() == "prompt-limit-mutation-spy" {
                    stub.record_mutation();
                }
                Err(EngineError::Generation("test stub".to_owned()))
            }
        }
    }

    /// Streaming counterpart of [`Self::generate_continued_with_thinking`]:
    /// emits each decoded token via `sender` instead of buffering the whole
    /// completion.
    pub fn generate_continued_streaming_with_thinking(
        &self,
        session_id: u64,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        enable_thinking: bool,
    ) -> Result<(), EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.generate_continued_streaming_with_thinking(
                session_id,
                prompt_tokens,
                max_tokens,
                params,
                sender,
                enable_thinking,
            ),
            Self::Batch(_) => Err(EngineError::Generation(
                "session_id (continued generation) is only supported by the Simple engine"
                    .to_owned(),
            )),
            #[cfg(test)]
            Self::Stub(_) => Err(EngineError::Generation("test stub".to_owned())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_session_routed_with_thinking(
        &self,
        session_id: u64,
        prompt_tokens: &[u32],
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        max_tokens: u32,
        params: &SamplingParams,
        enable_thinking: bool,
        tool_payload: SessionPromptTracePayloadStats,
        pflash_policy: &PFlashPromptPolicy,
        continuation_policy: SessionContinuationPolicy,
    ) -> Result<SessionGeneration, EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.generate_session_routed_with_thinking(
                session_id,
                prompt_tokens,
                messages,
                tools,
                max_tokens,
                params,
                enable_thinking,
                tool_payload,
                pflash_policy,
                continuation_policy,
            ),
            Self::Batch(_) => Err(EngineError::Generation(
                "session_id (session-routed generation) is only supported by the Simple engine"
                    .to_owned(),
            )),
            #[cfg(test)]
            Self::Stub(stub) => {
                if stub.name() == "blocking-required-post-admission-evicted" {
                    return Err(EngineError::RetainedSessionUnavailable(session_id));
                }
                if stub.name() == "session-prefill-render-spy" {
                    stub.record_mutation();
                    return Ok(SessionGeneration {
                        text: String::new(),
                        completion_tokens: 0,
                        finish_reason: "length".to_owned(),
                        prompt_tokens: u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX),
                        prefilled_tokens: u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX),
                        continued: false,
                        outcome: higgs_engine::simple::SessionOutcome::ExactBootstrap,
                    });
                }
                if stub.name() == "zero-prefix-accept" {
                    let continued = stub.route_session(session_id);
                    return Ok(SessionGeneration {
                        text: String::new(),
                        completion_tokens: 0,
                        finish_reason: "length".to_owned(),
                        prompt_tokens: 1,
                        prefilled_tokens: if continued { 0 } else { 1 },
                        continued,
                        outcome: if continued {
                            higgs_engine::simple::SessionOutcome::Continued
                        } else {
                            higgs_engine::simple::SessionOutcome::ExactBootstrap
                        },
                    });
                }
                if stub.name() == "prompt-limit-mutation-spy" {
                    stub.record_mutation();
                }
                Err(EngineError::Generation("test stub".to_owned()))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_session_routed_streaming_with_thinking(
        &self,
        session_id: u64,
        prompt_tokens: &[u32],
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        max_tokens: u32,
        params: &SamplingParams,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        enable_thinking: bool,
        tool_payload: SessionPromptTracePayloadStats,
        pflash_policy: &PFlashPromptPolicy,
        continuation_policy: SessionContinuationPolicy,
        mut acceptance: Option<SessionStreamAcceptance>,
    ) -> Result<(), EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.generate_session_routed_streaming_with_thinking(
                session_id,
                prompt_tokens,
                messages,
                tools,
                max_tokens,
                params,
                sender,
                enable_thinking,
                tool_payload,
                pflash_policy,
                continuation_policy,
                acceptance,
            ),
            Self::Batch(_) => {
                if let Some(acceptance) = acceptance.take() {
                    let _ = acceptance.send(Err(session_id));
                }
                Err(EngineError::Generation(
                    "session_id (session-routed streaming) is only supported by the Simple engine"
                        .to_owned(),
                ))
            }
            #[cfg(test)]
            Self::Stub(stub) if stub.name() == "session-prefill-render-spy" => {
                stub.record_mutation();
                if let Some(acceptance) = acceptance.take() {
                    let _ = acceptance.send(Ok(()));
                }
                sender
                    .blocking_send(StreamingOutput {
                        new_text: String::new(),
                        finished: true,
                        finish_reason: Some("length".to_owned()),
                        prompt_tokens: u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX),
                        completion_tokens: 0,
                        token_logprob: None,
                        prefill_progress: None,
                    })
                    .map_err(|_| EngineError::Cancelled)
            }
            #[cfg(test)]
            Self::Stub(stub) if stub.name() == "zero-prefix-materialization-fail" => {
                Err(EngineError::Generation(
                    "injected retained-state materialization failure".to_owned(),
                ))
            }
            #[cfg(test)]
            Self::Stub(stub) if stub.name() == "zero-prefix-accept" => {
                let continued = stub.route_session(session_id);
                if let Some(acceptance) = acceptance.take() {
                    let _ = acceptance.send(Ok(()));
                }
                if continued {
                    sender
                        .blocking_send(StreamingOutput {
                            new_text: String::new(),
                            finished: false,
                            finish_reason: None,
                            prompt_tokens: 1,
                            completion_tokens: 0,
                            token_logprob: None,
                            prefill_progress: Some(higgs_engine::engine::PrefillProgress {
                                processed: 1,
                                cached: 1,
                                total: 1,
                            }),
                        })
                        .map_err(|_| EngineError::Cancelled)?;
                }
                sender
                    .blocking_send(StreamingOutput {
                        new_text: String::new(),
                        finished: true,
                        finish_reason: Some("length".to_owned()),
                        prompt_tokens: 1,
                        completion_tokens: 0,
                        token_logprob: None,
                        prefill_progress: None,
                    })
                    .map_err(|_| EngineError::Cancelled)
            }
            #[cfg(test)]
            Self::Stub(stub) => {
                if stub.name() == "prompt-limit-mutation-spy" {
                    stub.record_mutation();
                }
                if let Some(acceptance) = acceptance.take() {
                    let _ = acceptance.send(Err(session_id));
                }
                Err(EngineError::RetainedSessionUnavailable(session_id))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
        checkpoint_id: Option<&str>,
    ) -> Result<GenerationOutput, EngineError> {
        self.generate_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            self.enable_thinking(),
            constraint,
            image_inputs,
            checkpoint_id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_thinking(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        enable_thinking: bool,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
        checkpoint_id: Option<&str>,
    ) -> Result<GenerationOutput, EngineError> {
        self.generate_with_thinking_and_pflash_policy(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            enable_thinking,
            constraint,
            image_inputs,
            checkpoint_id,
            &PFlashPromptPolicy::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_thinking_and_pflash_policy(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        enable_thinking: bool,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
        checkpoint_id: Option<&str>,
        pflash_policy: &PFlashPromptPolicy,
    ) -> Result<GenerationOutput, EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.generate_with_thinking_and_pflash_policy(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                enable_thinking,
                constraint,
                image_inputs,
                checkpoint_id,
                pflash_policy,
            ),
            Self::Batch(e) => e.generate_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                enable_thinking,
                constraint,
                image_inputs,
            ),
            #[cfg(test)]
            Self::Stub(_) => Err(EngineError::Generation("test stub".to_owned())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_thinking_and_pflash_policy_with_cache(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        enable_thinking: bool,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
        checkpoint_id: Option<&str>,
        pflash_policy: &PFlashPromptPolicy,
        allow_prefix_cache: bool,
    ) -> Result<GenerationOutput, EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.generate_with_thinking_and_pflash_policy_with_cache(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                enable_thinking,
                constraint,
                image_inputs,
                checkpoint_id,
                pflash_policy,
                allow_prefix_cache,
            ),
            Self::Batch(e) => e.generate_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                enable_thinking,
                constraint,
                image_inputs,
            ),
            #[cfg(test)]
            Self::Stub(_) => Err(EngineError::Generation("test stub".to_owned())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
        checkpoint_id: Option<&str>,
    ) -> Result<(), EngineError> {
        self.generate_streaming_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            sender,
            self.enable_thinking(),
            // /v1/completions convenience entry never streams prefill progress.
            false,
            constraint,
            image_inputs,
            checkpoint_id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_thinking(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        enable_thinking: bool,
        return_progress: bool,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
        checkpoint_id: Option<&str>,
    ) -> Result<(), EngineError> {
        self.generate_streaming_with_thinking_and_pflash_policy(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            sender,
            enable_thinking,
            return_progress,
            constraint,
            image_inputs,
            checkpoint_id,
            &PFlashPromptPolicy::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_thinking_and_pflash_policy(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        enable_thinking: bool,
        return_progress: bool,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
        checkpoint_id: Option<&str>,
        pflash_policy: &PFlashPromptPolicy,
    ) -> Result<(), EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.generate_streaming_with_thinking_and_pflash_policy(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                sender,
                enable_thinking,
                return_progress,
                constraint,
                image_inputs,
                checkpoint_id,
                pflash_policy,
            ),
            Self::Batch(e) => e.generate_streaming_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                sender,
                enable_thinking,
                return_progress,
                constraint,
                image_inputs,
            ),
            #[cfg(test)]
            Self::Stub(_) => Err(EngineError::Generation("test stub".to_owned())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_thinking_and_pflash_policy_with_cache(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<higgs_engine::engine::StreamingOutput>,
        enable_thinking: bool,
        return_progress: bool,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        image_inputs: Option<Vec<ImageInput>>,
        checkpoint_id: Option<&str>,
        pflash_policy: &PFlashPromptPolicy,
        allow_prefix_cache: bool,
    ) -> Result<(), EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.generate_streaming_with_thinking_and_pflash_policy_with_cache(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                sender,
                enable_thinking,
                return_progress,
                constraint,
                image_inputs,
                checkpoint_id,
                pflash_policy,
                allow_prefix_cache,
            ),
            Self::Batch(e) => e.generate_streaming_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                sender,
                enable_thinking,
                return_progress,
                constraint,
                image_inputs,
            ),
            #[cfg(test)]
            Self::Stub(_) => Err(EngineError::Generation("test stub".to_owned())),
        }
    }

    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.embed(token_ids),
            Self::Batch(e) => e.embed(token_ids),
            #[cfg(test)]
            Self::Stub(_) => Ok(Vec::new()),
        }
    }
}

/// Build an engine from an already-resolved model directory and its config.
///
/// Shared by startup loading (`load_engines` in the binary) and the runtime
/// load endpoint (`POST /v1/models`). Path resolution and any download prompt
/// are the caller's responsibility, so this never blocks on stdin. Returns the
/// model's exposed name alongside the constructed engine.
pub fn build_engine(
    resolved: &Path,
    model_cfg: &ModelConfig,
    local: &LocalConfig,
) -> Result<(String, Engine), String> {
    validate_pflash_settings(model_cfg)
        .map_err(|error| format!("invalid PFlash settings: {error}"))?;
    if model_cfg.batch && !resolved_model_supports_batch(resolved)? {
        return Err(format!(
            "batch=true is only supported for transformer models (llama, mistral, qwen2, qwen3), llava-qwen2, and qwen3_5_vl; '{}' is not supported",
            model_cfg.path
        ));
    }
    let kv_cache_config = model_cfg.kv_cache_config();
    let engine = if model_cfg.batch {
        Engine::load_batch(resolved, kv_cache_config, local.raise_wired_limit)
            .map_err(|e| e.to_string())?
    } else {
        let tuning = resolve_runtime_tuning(resolved, model_cfg.requested_mlx_profile(local));
        Engine::load_simple(
            resolved,
            kv_cache_config,
            tuning,
            local.raise_wired_limit,
            model_cfg.draft_model.as_deref().map(Path::new),
            model_cfg.prefill_drafter.as_deref().map(Path::new),
            model_cfg.prefill_compression,
            model_cfg.prefill_keep_ratio,
            model_cfg.prefill_threshold,
            model_cfg.prefill_chunk,
            model_cfg.prefill_avgpool,
            model_cfg.prefill_lookahead,
            model_cfg.prefill_score_mode,
            model_cfg.prefill_exit_layer,
            model_cfg.prefill_keep_ratio_max,
            model_cfg.prefill_max_auto_prefill_ratio,
            model_cfg.prefill_plan_cache,
            model_cfg.prefill_plan_cache_entries,
            model_cfg.prefill_suffix_identity_threshold,
            model_cfg.kv_max_suffix_prefill_tokens,
            model_cfg.disk_prefix_cache_config(resolved),
        )
        .map_err(|e| e.to_string())?
    };
    let name = model_cfg
        .name
        .clone()
        .unwrap_or_else(|| engine.model_name().to_owned());
    Ok((name, engine))
}

/// Shared application state available to all route handlers.
pub struct AppState {
    /// Routes model names to local engines or remote providers.
    pub router: Router,
    /// Full server configuration.
    pub config: HiggsConfig,
    /// HTTP client for proxying requests to remote providers.
    pub http_client: reqwest::Client,
    /// Request metrics (present in config mode, absent in simple mode).
    pub metrics: Option<Arc<MetricsStore>>,
}

/// Type alias for the shared state used by Axum handlers.
pub type SharedState = Arc<AppState>;

/// Build a `SharedState` whose router serves `model` from a stub (non-VLM)
/// engine, for route-level tests of the vision capability gate.
///
/// The stub reports `is_vlm() == false`, so an image request routed to it must
/// hit the 400 gate before any tokenizer or generation code runs.
#[cfg(test)]
#[allow(clippy::expect_used)]
pub(crate) fn test_state_with_stub_engine(model: &str) -> SharedState {
    let config = crate::config::HiggsConfig::default();
    let mut engines = std::collections::HashMap::new();
    engines.insert(model.to_owned(), Arc::new(Engine::test_stub(model)));
    let router = crate::router::Router::from_config(&config, engines)
        .expect("default config builds a router");
    Arc::new(AppState {
        router,
        config,
        http_client: reqwest::Client::new(),
        metrics: None,
    })
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn build_engine_rejects_invalid_pflash_before_model_load() {
        let resolved = tempfile::tempdir().unwrap();
        let model = ModelConfig {
            path: "missing/model".to_owned(),
            prefill_keep_ratio: 1.0,
            ..ModelConfig::default()
        };

        let outcome = std::panic::catch_unwind(|| {
            build_engine(resolved.path(), &model, &LocalConfig::default())
        });
        assert!(outcome.is_ok(), "invalid PFlash config must not panic");
        let error = match outcome.unwrap() {
            Ok(_) => panic!("invalid PFlash config must fail before model load"),
            Err(error) => error,
        };
        assert!(
            error.contains("prefill_keep_ratio"),
            "expected PFlash validation error, got {error}"
        );
    }

    #[test]
    fn stub_engine_reports_no_vision() {
        let engine = Engine::test_stub("test-stub");
        assert!(!engine.is_vlm());
        assert!(engine.vision_capabilities().is_none());
    }

    #[test]
    fn stub_engine_preprocess_images_errors() {
        let engine = Engine::test_stub("test-stub");
        let err = engine.preprocess_images(&[]).unwrap_err();
        assert!(err.to_string().contains("stub"));
    }
}
