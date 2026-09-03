use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use higgs_engine::batch_engine::BatchEngine;
use higgs_engine::cache::DiskPrefixCacheConfig;
use higgs_engine::chat_template::{ChatMessage, ChatPromptMode};
use higgs_engine::engine::{GenerationOutput, StreamingOutput};
use higgs_engine::error::EngineError;
use higgs_engine::mlx_tuning::{
    MlxRuntimeTuning, ModelFootprint, TransientPrefillEstimate, resolve_runtime_tuning,
};
use higgs_engine::simple::{
    CacheStats, PFlashPromptPolicy, PrefillCompressionMode as EnginePrefillCompressionMode,
    SessionContinuationPolicy, SessionGeneration, SessionPromptTracePayloadStats,
    SessionStreamAcceptance, SimpleEngine,
};
use higgs_engine::tokenizers::Tokenizer;
use higgs_models::SamplingParams;
use higgs_models::turboquant::KvCacheConfig;
use higgs_models::vision::{ImageBatch, ImageInput, VisionCapabilities, VisionError};

use sha2::{Digest, Sha256};

use crate::capacity::CapacityRegistry;
use crate::capacity::{
    CAPACITY_SCHEMA_VERSION, LearnedProfileKey, ModelCapacityFacts, ModelContentIdentity,
    fingerprint_model_artifacts,
};
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
    capacity_cache_limits: std::sync::Mutex<(u64, u64)>,
    cache_apply_count: Arc<std::sync::atomic::AtomicU64>,
    cache_apply_gate:
        std::sync::Mutex<Option<(u64, Arc<tokio::sync::Notify>, Arc<tokio::sync::Notify>)>>,
}

#[cfg(test)]
impl RouteTestStub {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            mutations: std::sync::atomic::AtomicU64::new(0),
            mutation_sequence: std::sync::Mutex::new(Vec::new()),
            retained_sessions: std::sync::Mutex::new(std::collections::HashSet::new()),
            capacity_cache_limits: std::sync::Mutex::new((0, 0)),
            cache_apply_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            cache_apply_gate: std::sync::Mutex::new(None),
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

    fn set_capacity_cache_limits(&self, retained_bytes: u64, prefix_bytes: u64) {
        *self.capacity_cache_limits.lock().unwrap() = (retained_bytes, prefix_bytes);
    }

    fn capacity_cache_limits(&self) -> (u64, u64) {
        *self.capacity_cache_limits.lock().unwrap()
    }

    async fn wait_cache_apply_gate(&self) {
        let count = self
            .cache_apply_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .saturating_add(1);
        let gate = {
            let mut gate = self.cache_apply_gate.lock().unwrap();
            gate.as_ref()
                .is_some_and(|(gate_after, _, _)| count >= *gate_after)
                .then(|| gate.take())
                .flatten()
        };
        if let Some((_, arrived, release)) = gate {
            arrived.notify_one();
            release.notified().await;
        }
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

fn with_serialized_mlx_load<T>(operation: impl FnOnce() -> T) -> T {
    let _gpu = gpu_gate();
    operation()
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
    #[must_use]
    pub const fn cache_capabilities(&self) -> crate::capacity::CacheCapabilities {
        match self {
            Self::Simple(_) => crate::capacity::CacheCapabilities::SIMPLE,
            Self::Batch(_) => crate::capacity::CacheCapabilities::BATCH,
            #[cfg(test)]
            Self::Stub(_) => crate::capacity::CacheCapabilities::SIMPLE,
        }
    }

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
    pub(crate) fn test_stub_with_cache_gate(
        name: &str,
    ) -> (Self, Arc<tokio::sync::Notify>, Arc<tokio::sync::Notify>) {
        let stub = RouteTestStub::new(name);
        let arrived = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        *stub.cache_apply_gate.lock().unwrap() =
            Some((1, Arc::clone(&arrived), Arc::clone(&release)));
        (Self::Stub(stub), arrived, release)
    }

    #[cfg(test)]
    pub(crate) fn test_stub_with_cache_gate_after(
        name: &str,
        gate_after: u64,
    ) -> (
        Self,
        Arc<tokio::sync::Notify>,
        Arc<tokio::sync::Notify>,
        Arc<std::sync::atomic::AtomicU64>,
    ) {
        let stub = RouteTestStub::new(name);
        let arrived = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        *stub.cache_apply_gate.lock().unwrap() =
            Some((gate_after, Arc::clone(&arrived), Arc::clone(&release)));
        let count = Arc::clone(&stub.cache_apply_count);
        (Self::Stub(stub), arrived, release, count)
    }

    #[cfg(test)]
    pub(crate) fn route_test_cache_apply_count(&self) -> u64 {
        match self {
            Self::Stub(stub) => stub
                .cache_apply_count
                .load(std::sync::atomic::Ordering::Relaxed),
            _ => 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn route_test_mutations(&self) -> u64 {
        match self {
            Self::Stub(stub) => stub.mutation_count(),
            _ => 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn route_test_capacity_cache_limits(&self) -> (u64, u64) {
        match self {
            Self::Stub(stub) => stub.capacity_cache_limits(),
            _ => (0, 0),
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

    /// Apply the process registry's effective cache allocation. Batch engines
    /// acknowledge the worker-side eviction before this returns.
    pub async fn apply_capacity_cache_limits(
        &self,
        revision: u64,
        retained_bytes: u64,
        prefix_bytes: u64,
        pressure: crate::capacity::MemoryPressure,
    ) -> Result<(), EngineError> {
        let retained_limit = usize::try_from(retained_bytes).map_err(|_| {
            EngineError::Generation("retained cache allocation exceeds platform usize".to_owned())
        })?;
        let prefix_limit = usize::try_from(prefix_bytes).map_err(|_| {
            EngineError::Generation("prefix cache allocation exceeds platform usize".to_owned())
        })?;
        match self {
            Self::Simple(engine) => {
                engine.apply_capacity_cache_limits(
                    retained_limit,
                    prefix_limit,
                    if pressure == crate::capacity::MemoryPressure::Critical {
                        higgs_engine::simple::CachePressurePolicy::Critical
                    } else {
                        higgs_engine::simple::CachePressurePolicy::Normal
                    },
                );
                Ok(())
            }
            Self::Batch(engine) => {
                engine
                    .apply_capacity_cache_limit(revision, prefix_limit)
                    .await
            }
            #[cfg(test)]
            Self::Stub(stub) => {
                stub.wait_cache_apply_gate().await;
                stub.set_capacity_cache_limits(retained_bytes, prefix_bytes);
                Ok(())
            }
        }
    }

    /// Destroy any worker-owned model before allocator cleanup/measurement.
    pub fn shutdown(self) -> Result<(), EngineError> {
        match self {
            Self::Batch(engine) => (*engine).shutdown(),
            Self::Simple(engine) => {
                drop(engine);
                Ok(())
            }
            #[cfg(test)]
            Self::Stub(engine) => {
                drop(engine);
                Ok(())
            }
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
    let name = resolve_exposed_model_name(model_cfg.name.as_deref(), &model_cfg.path, resolved);
    Ok((name, engine))
}

/// Build one engine while capturing exact artifact identity and serialized MLX residency.
pub fn build_engine_with_capacity(
    resolved: &Path,
    model_cfg: &ModelConfig,
    config: &HiggsConfig,
    capacity: &CapacityRegistry,
) -> Result<(String, Engine, ModelCapacityFacts), String> {
    let identity = fingerprint_model_artifacts(resolved)
        .map_err(|error| format!("failed to fingerprint model artifacts: {error}"))?;
    let (name, engine, before, after) = with_serialized_mlx_load(|| {
        let before = higgs_engine::MlxMemorySnapshot::measure()
            .map_err(|error| format!("failed to measure MLX before model load: {error}"))?;
        let (name, engine) = match build_engine(resolved, model_cfg, &config.local) {
            Ok(loaded) => loaded,
            Err(error) => {
                higgs_engine::simple::maybe_clear_mlx_cache(true, "failed model load");
                if let Ok(memory) = higgs_engine::MlxMemorySnapshot::measure() {
                    capacity.refresh_memory(memory);
                }
                return Err(error);
            }
        };
        let after = match higgs_engine::MlxMemorySnapshot::measure() {
            Ok(memory) => memory,
            Err(error) => {
                if let Err(shutdown_error) = engine.shutdown() {
                    tracing::warn!(%shutdown_error, "failed to join engine after load measurement failure");
                }
                higgs_engine::simple::maybe_clear_mlx_cache(true, "failed model load measurement");
                if let Ok(memory) = higgs_engine::MlxMemorySnapshot::measure() {
                    capacity.refresh_memory(memory);
                }
                return Err(format!("failed to measure MLX after model load: {error}"));
            }
        };
        // Publish while the process GPU/load gate is still held. Lifecycle
        // commits may complete out of order, but can no longer regress the
        // allocator authority to an older per-load snapshot.
        capacity.refresh_memory(after);
        Ok((name, engine, before, after))
    })?;
    let cache_capabilities = engine.cache_capabilities();
    let facts = match build_capacity_facts_from_measurements(
        &name,
        resolved,
        model_cfg,
        config,
        identity,
        before,
        after,
        cache_capabilities,
    ) {
        Ok(facts) => facts,
        Err(error) => {
            release_failed_engine(engine, capacity);
            return Err(error);
        }
    };
    Ok((name, engine, facts))
}

#[doc(hidden)]
pub fn release_failed_engine(engine: Engine, capacity: &CapacityRegistry) {
    #[cfg(test)]
    if matches!(engine, Engine::Stub(_)) {
        drop(engine);
        return;
    }
    with_serialized_mlx_load(|| {
        if let Err(error) = engine.shutdown() {
            tracing::warn!(%error, "failed to join engine during failed publication cleanup");
        }
        refresh_after_engine_drop_locked(capacity, "failed model publication");
    });
}

#[doc(hidden)]
pub fn refresh_after_engine_drop(
    capacity: &CapacityRegistry,
    reason: &'static str,
) -> Option<crate::capacity::PublishedMemoryMeasurement> {
    #[cfg(test)]
    {
        let _ = (capacity, reason);
        None
    }
    #[cfg(not(test))]
    with_serialized_mlx_load(|| refresh_after_engine_drop_locked(capacity, reason))
}

pub(crate) fn measure_after_engine_drop(
    capacity: &CapacityRegistry,
    reason: &'static str,
) -> Option<crate::capacity::PublishedMemoryMeasurement> {
    #[cfg(test)]
    {
        let _ = (capacity, reason);
        None
    }
    #[cfg(not(test))]
    with_serialized_mlx_load(|| refresh_after_engine_drop_locked(capacity, reason))
}

fn refresh_after_engine_drop_locked(
    capacity: &CapacityRegistry,
    reason: &'static str,
) -> Option<crate::capacity::PublishedMemoryMeasurement> {
    let memory = measure_after_engine_drop_locked(reason)?;
    Some(capacity.refresh_memory(memory))
}

fn measure_after_engine_drop_locked(
    reason: &'static str,
) -> Option<higgs_engine::MlxMemorySnapshot> {
    higgs_engine::simple::maybe_clear_mlx_cache(true, reason);
    higgs_engine::MlxMemorySnapshot::measure().ok()
}

#[allow(clippy::too_many_arguments)]
fn build_capacity_facts_from_measurements(
    name: &str,
    resolved: &Path,
    model_cfg: &ModelConfig,
    config: &HiggsConfig,
    identity: ModelContentIdentity,
    before: higgs_engine::MlxMemorySnapshot,
    after: higgs_engine::MlxMemorySnapshot,
    cache_capabilities: crate::capacity::CacheCapabilities,
) -> Result<ModelCapacityFacts, String> {
    const MIB: u64 = 1024 * 1024;
    let footprint = ModelFootprint::from_load_measurements(resolved, before, after)
        .ok_or_else(|| "model residency or artifact measurement was invalid".to_owned())?;
    let model_json = std::fs::read(resolved.join("config.json"))
        .map_err(|error| format!("failed to read model config for capacity: {error}"))?;
    let model_json: serde_json::Value = serde_json::from_slice(&model_json)
        .map_err(|error| format!("failed to parse model config for capacity: {error}"))?;
    let architectural_max_tokens = config_u64(&model_json, "max_position_embeddings")
        .ok_or_else(|| "model config lacks max_position_embeddings".to_owned())?;
    let tuning = resolve_runtime_tuning(resolved, model_cfg.requested_mlx_profile(&config.local));
    let prefill_chunk_tokens = u64::try_from(tuning.chunked_prefill_chunk_size())
        .map_err(|_| "prefill chunk size is negative".to_owned())?;
    let transient_base = (identity.artifact_bytes / 3).max(512 * MIB);
    let transient = TransientPrefillEstimate {
        base_bytes: transient_base,
        bytes_per_prompt_token: 0,
        bytes_per_chunk_token: 0,
        max_prompt_tokens: architectural_max_tokens,
        max_chunk_tokens: prefill_chunk_tokens,
    };
    let costs = higgs_engine::EngineCostDescription::fp16_from_model_dir(
        resolved,
        256 * MIB,
        256 * MIB,
        transient,
    )
    .ok_or_else(|| "model config lacks safe KV geometry".to_owned())?;
    let quantize_json = read_optional_json(&resolved.join("quantize_config.json"))?;
    let quantization = quantize_json
        .as_ref()
        .or_else(|| model_json.get("quantization"))
        .or_else(|| model_json.get("quantization_config"))
        .map_or_else(|| "unquantized".to_owned(), serde_json::Value::to_string);
    let quantization_mode = model_json
        .get("quantization")
        .and_then(|value| value.get("mode"))
        .and_then(serde_json::Value::as_str);
    let is_eschamoe = higgs_models::eschamoe::is_eschamoe_checkpoint(resolved)
        .map_err(|error| format!("failed to identify model execution mode: {error}"))?;
    let runtime_environment = resolved_model_runtime_identity(
        resolved,
        &model_json,
        is_eschamoe,
        model_cfg.mla_latent_cache.unwrap_or(false),
    );
    let weight_execution = if is_eschamoe && higgs_models::eschamoe::native_mode() {
        "eschamoe-native"
    } else if is_eschamoe {
        "eschamoe-affine"
    } else {
        quantization_mode.unwrap_or("standard")
    };
    let execution_mode = format!(
        "{}:{weight_execution}",
        if model_cfg.batch { "batch" } else { "simple" }
    );
    let kv_representation = match model_cfg.kv_cache {
        higgs_models::turboquant::KvCacheMode::Off => "fp16".to_owned(),
        higgs_models::turboquant::KvCacheMode::Turboquant => format!(
            "turboquant:k{}:v{}:dense{}",
            model_cfg
                .kv_key_bits
                .unwrap_or(model_cfg.kv_bits.saturating_sub(1)),
            model_cfg.kv_value_bits.unwrap_or(model_cfg.kv_bits),
            model_cfg.kv_adaptive_dense_layers
        ),
    };
    let prefill_model_identity = optional_artifact_identity(model_cfg.prefill_drafter.as_deref())?;
    let effective_drafter_path = model_cfg
        .draft_model
        .clone()
        .or_else(|| std::env::var("HIGGS_DFLASH_PATH").ok());
    let drafter_identity = optional_artifact_identity(effective_drafter_path.as_deref())?;
    let startup_headroom_bytes = smaller_nonzero_memory_authority(before)
        .map_or(0, |authority| authority.saturating_sub(before.active_bytes));
    let learned_profile_key = learned_profile_key(
        model_cfg,
        config,
        &tuning,
        &identity.fingerprint,
        &quantization,
        &execution_mode,
        &kv_representation,
        prefill_model_identity.clone(),
        drafter_identity.clone(),
        runtime_environment,
        after,
    );

    Ok(ModelCapacityFacts {
        model: name.to_owned(),
        model_fingerprint: identity.fingerprint,
        memory: after,
        costs,
        loaded_model_bytes: footprint.loaded_mlx_bytes,
        architectural_max_tokens,
        prefill_chunk_tokens,
        retained_session_tokens: u64::try_from(model_cfg.kv_max_session_tokens)
            .map_err(|_| "retained-session token ceiling overflows u64".to_owned())?,
        retained_resident_bytes: 0,
        prefix_cache_resident_bytes: 0,
        retained_bytes_ceiling: u64::try_from(model_cfg.kv_max_retained_bytes)
            .map_err(|_| "retained byte ceiling overflows u64".to_owned())?,
        prefix_cache_bytes_ceiling: u64::try_from(model_cfg.kv_cache_bytes)
            .map_err(|_| "prefix-cache byte ceiling overflows u64".to_owned())?,
        cache_capabilities,
        configured_total_token_ceiling: None,
        configured_output_token_ceiling: Some(u64::from(config.server.max_tokens)),
        quantization,
        execution_mode,
        kv_representation,
        prefill_model_identity,
        drafter_identity,
        learned_profile_key,
        startup_headroom_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn learned_profile_key(
    model_cfg: &ModelConfig,
    config: &HiggsConfig,
    tuning: &MlxRuntimeTuning,
    model_fingerprint: &str,
    quantization: &str,
    execution_mode: &str,
    kv_representation: &str,
    prefill_model_identity: Option<String>,
    drafter_identity: Option<String>,
    runtime_environment: Option<higgs_engine::runtime_identity::ResolvedRuntimeIdentity>,
    memory: higgs_engine::MlxMemorySnapshot,
) -> Option<LearnedProfileKey> {
    let platform = platform_identity()?;
    let backend_authority_bytes = smaller_nonzero_memory_authority(memory)?;
    let higgs_build = executable_build_identity()?;
    let runtime_environment = runtime_environment?;
    let settings = serde_json::json!({
        "capacitySchema": CAPACITY_SCHEMA_VERSION,
        "batch": model_cfg.batch,
        "raiseWiredLimit": config.local.raise_wired_limit,
        "requestedMlxProfile": format!("{:?}", tuning.requested_profile()),
        "resolvedMlxProfile": format!("{:?}", tuning.resolved_profile()),
        "chunkedPrefillThreshold": tuning.chunked_prefill_threshold(),
        "chunkedPrefillChunkSize": tuning.chunked_prefill_chunk_size(),
        "clearCacheAfterPrefill": tuning.clear_cache_after_prefill(),
        "enableMtp": tuning.enable_mtp(),
        "mtpDraftNMax": tuning.mtp_draft_n_max(),
        "pagedKvTargetBytes": tuning.paged_kv_target_bytes(),
        "prefillYieldTokens": model_cfg.prefill_yield_tokens,
        "prefillCompression": format!("{:?}", model_cfg.prefill_compression),
        "prefillThreshold": model_cfg.prefill_threshold,
        "prefillKeepRatioBits": model_cfg.prefill_keep_ratio.to_bits(),
        "prefillChunk": model_cfg.prefill_chunk,
        "prefillAvgpool": model_cfg.prefill_avgpool,
        "prefillLookahead": model_cfg.prefill_lookahead,
        "prefillScoreMode": format!("{:?}", model_cfg.prefill_score_mode),
        "prefillExitLayer": model_cfg.prefill_exit_layer,
        "prefillKeepRatioMaxBits": model_cfg.prefill_keep_ratio_max.to_bits(),
        "prefillMaxAutoRatioBits": model_cfg.prefill_max_auto_prefill_ratio.to_bits(),
        "prefillPlanCache": model_cfg.prefill_plan_cache,
        "prefillPlanCacheEntries": model_cfg.prefill_plan_cache_entries,
        "prefillSuffixIdentityThreshold": model_cfg.prefill_suffix_identity_threshold,
        "diskCacheEnabled": model_cfg.disk_cache_enabled,
        "maxDiskBlocks": model_cfg.max_disk_blocks,
        "minTokensToPersist": model_cfg.min_tokens_to_persist,
        "kvMaxSessions": model_cfg.kv_max_sessions,
        "kvMaxSessionTokens": model_cfg.kv_max_session_tokens,
        "kvRetainedIdleSecs": model_cfg.kv_retained_idle_secs,
        "kvMaxRetainedBytes": model_cfg.kv_max_retained_bytes,
        "kvMaxSuffixPrefillTokens": model_cfg.kv_max_suffix_prefill_tokens,
        "kvCacheBytes": model_cfg.kv_cache_bytes,
        "kvDiskSpaceMb": model_cfg.kv_disk_space_mb,
        "runtimeEnvironment": runtime_environment,
    });
    let execution_cache_fingerprint = sha256_identity(
        b"higgs:capacity-execution-cache:v1\0",
        serde_json::to_vec(&settings).ok()?.as_slice(),
    );
    Some(LearnedProfileKey {
        hardware_identifier: platform.hardware_identifier,
        physical_memory_bytes: platform.physical_memory_bytes,
        os_version: platform.os_version,
        os_build: platform.os_build,
        backend_authority_bytes,
        higgs_build,
        model_fingerprint: model_fingerprint.to_owned(),
        quantization: quantization.to_owned(),
        execution_mode: execution_mode.to_owned(),
        kv_representation: kv_representation.to_owned(),
        prefill_model_identity,
        execution_cache_fingerprint,
        drafter_identity,
    })
}

fn resolved_model_runtime_identity(
    model_dir: &Path,
    model_json: &serde_json::Value,
    is_eschamoe: bool,
    mla_latent_cache: bool,
) -> Option<higgs_engine::runtime_identity::ResolvedRuntimeIdentity> {
    let runtime_config = model_json.get("text_config").unwrap_or(model_json);
    let model_type = runtime_config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if matches!(
        model_type,
        "qwen3_next" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe"
    ) {
        let args =
            higgs_models::qwen3_next::resolve_runtime_model_args(model_dir, model_json).ok()?;
        Some(
            higgs_engine::runtime_identity::resolved_runtime_identity_for_qwen(
                is_eschamoe,
                mla_latent_cache,
                &args,
            ),
        )
    } else {
        Some(higgs_engine::runtime_identity::resolved_runtime_identity(
            is_eschamoe,
            mla_latent_cache,
        ))
    }
}

fn smaller_nonzero_memory_authority(memory: higgs_engine::MlxMemorySnapshot) -> Option<u64> {
    [
        memory.memory_limit_bytes,
        memory.metal_recommended_working_set_bytes,
    ]
    .into_iter()
    .flatten()
    .filter(|bytes| *bytes > 0)
    .min()
}

struct PlatformIdentity {
    hardware_identifier: String,
    physical_memory_bytes: u64,
    os_version: String,
    os_build: String,
}

#[cfg(target_os = "macos")]
fn platform_identity() -> Option<PlatformIdentity> {
    fn sysctl(name: &str) -> Option<String> {
        let output = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", name])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let value = String::from_utf8(output.stdout).ok()?.trim().to_owned();
        (!value.is_empty()).then_some(value)
    }
    Some(PlatformIdentity {
        hardware_identifier: sysctl("hw.model")?,
        physical_memory_bytes: sysctl("hw.memsize")?
            .parse()
            .ok()
            .filter(|bytes| *bytes > 0)?,
        os_version: sysctl("kern.osproductversion")?,
        os_build: sysctl("kern.osversion")?,
    })
}

#[cfg(not(target_os = "macos"))]
fn platform_identity() -> Option<PlatformIdentity> {
    None
}

fn executable_build_identity() -> Option<String> {
    let path = std::env::current_exe().ok()?;
    let mut file = File::open(&path).ok()?;
    let mut hash = Sha256::new();
    hash.update(b"higgs:executable-build:v1\0");
    hash.update(CAPACITY_SCHEMA_VERSION.to_le_bytes());
    hash.update(file.metadata().ok()?.len().to_le_bytes());
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).ok()?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    if cfg!(target_os = "macos") {
        let metallib_path = path.with_file_name("mlx.metallib");
        let mut metallib = File::open(metallib_path).ok()?;
        hash.update(b"\0mlx.metallib\0");
        hash.update(metallib.metadata().ok()?.len().to_le_bytes());
        loop {
            let count = metallib.read(&mut buffer).ok()?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
    }
    Some(encode_sha256(hash.finalize()))
}

fn sha256_identity(domain: &[u8], bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(bytes);
    encode_sha256(hash.finalize())
}

fn encode_sha256(digest: impl IntoIterator<Item = u8>) -> String {
    let mut encoded = String::with_capacity(71);
    encoded.push_str("sha256:");
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn config_u64(config: &serde_json::Value, key: &str) -> Option<u64> {
    config
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|text| text.get(key))
                .and_then(serde_json::Value::as_u64)
        })
}

fn read_optional_json(path: &Path) -> Result<Option<serde_json::Value>, String> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("failed to read {}: {error}", path.display())),
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn optional_artifact_identity(path: Option<&str>) -> Result<Option<String>, String> {
    path.map(|path| {
        fingerprint_model_artifacts(Path::new(path))
            .map(|identity| identity.fingerprint)
            .map_err(|error| format!("failed to fingerprint sidecar '{path}': {error}"))
    })
    .transpose()
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
    /// Sole process-wide authority for local-model capacity and lifecycle state.
    pub capacity: Arc<CapacityRegistry>,
}

impl AppState {
    #[must_use]
    pub fn new(
        router: Router,
        config: HiggsConfig,
        http_client: reqwest::Client,
        metrics: Option<Arc<MetricsStore>>,
    ) -> Self {
        let known_models = config.models.iter().map(|model| {
            resolve_exposed_model_name(model.name.as_deref(), &model.path, Path::new(&model.path))
        });
        let capacity = CapacityRegistry::new(known_models);
        Self::with_capacity_registry(router, config, http_client, metrics, capacity)
    }

    #[must_use]
    pub fn with_capacity_registry(
        router: Router,
        config: HiggsConfig,
        http_client: reqwest::Client,
        metrics: Option<Arc<MetricsStore>>,
        capacity: Arc<CapacityRegistry>,
    ) -> Self {
        Self {
            router,
            config,
            http_client,
            metrics,
            capacity,
        }
    }
}

/// Apply the engine's exposed-name rule before a model enters either the known
/// catalog or the live routing table. Explicit aliases remain authoritative.
#[must_use]
pub fn resolve_exposed_model_name(
    configured_name: Option<&str>,
    configured_path: &str,
    resolved_path: &Path,
) -> String {
    if let Some(name) = configured_name {
        return name.to_owned();
    }
    let configured = Path::new(configured_path);
    if !configured.exists()
        && configured == resolved_path
        && crate::model_resolver::is_hf_model_id(configured_path)
    {
        return configured_path.to_owned();
    }
    higgs_engine::simple::exposed_model_name(resolved_path)
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
    Arc::new(AppState::new(router, config, reqwest::Client::new(), None))
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn exposed_model_name_is_identical_for_catalog_and_loaded_paths() {
        assert_eq!(
            resolve_exposed_model_name(Some("alias"), "/models/raw", Path::new("/models/raw")),
            "alias"
        );
        assert_eq!(
            resolve_exposed_model_name(None, "/models/Escha-35B", Path::new("/models/Escha-35B")),
            "Escha-35B"
        );
        assert_eq!(
            resolve_exposed_model_name(None, "NexVeridian/Escha", Path::new("NexVeridian/Escha")),
            "NexVeridian/Escha"
        );
        assert_eq!(
            resolve_exposed_model_name(
                None,
                "NexVeridian/Escha",
                Path::new("/cache/models--NexVeridian--Escha/snapshots/deadbeef"),
            ),
            "NexVeridian/Escha"
        );

        let root = tempfile::tempdir().unwrap();
        let local = root.path().join("models").join("foo");
        std::fs::create_dir_all(&local).unwrap();
        assert_eq!(
            resolve_exposed_model_name(None, "models/foo", &local),
            "foo"
        );
    }

    #[test]
    fn capacity_profile_environment_uses_the_canonical_engine_identity() {
        let environment = serde_json::to_value(
            higgs_engine::runtime_identity::resolved_runtime_identity_with(false, false, |name| {
                match name {
                    "HIGGS_PREFLASH_FULL_SCORE_MAX_TOKENS"
                    | "HIGGS_PREFLASH_MIN_FREE_MB"
                    | "HIGGS_DFLASH_BLOCK_SIZE"
                    | "HIGGS_DSPARK_DRAFT_CAP"
                    | "HIGGS_DFLASH_MIN_BLOCK" => Some(" 4096 ".to_owned()),
                    "HIGGS_DENSE_REQUANT_8BIT" => Some("1".to_owned()),
                    "HIGGS_BONSAI_QMV_KERNEL" => Some("legacy".to_owned()),
                    _ => None,
                }
            }),
        )
        .unwrap();
        assert_eq!(environment["pflashFullScoreMaxTokens"], 8192);
        assert_eq!(environment["pflashMinimumFreeMb"], 2048);
        assert!(environment["dflashBlockSize"].is_null());
        assert!(environment["dsparkDraftCapOverride"].is_null());
        assert!(environment["dflashMinimumBlock"].is_null());
        assert_eq!(environment["denseGdnRequant8Bit"], true);
        assert_eq!(environment["bonsaiQmvKernel"], "legacy");
    }

    #[test]
    fn runtime_identity_supports_non_qwen_and_nested_qwen_configs() {
        assert!(
            resolved_model_runtime_identity(
                Path::new("/does/not/need/model/artifacts"),
                &serde_json::json!({"model_type": "gemma3_text"}),
                false,
                false,
            )
            .is_some(),
            "non-Qwen families retain generic canonical profile identity"
        );

        let nested = serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": {
                "mode": "affine",
                "bits": 2,
                "group_size": 64
            },
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "intermediate_size": 17408,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "rms_norm_eps": 0.000001,
                "vocab_size": 1024,
                "max_position_embeddings": 512
            }
        });
        let model_dir = tempfile::tempdir().unwrap();
        let identity = resolved_model_runtime_identity(model_dir.path(), &nested, true, false)
            .expect("nested Qwen text config resolves typed identity");
        let value = serde_json::to_value(identity).unwrap();
        assert_ne!(value["denseFfnGateUp"], "auto");
        assert_eq!(value["bonsaiQ2Simd"], "escha_qwen38");

        let top_level_moe = nested["text_config"].clone();
        let identity =
            resolved_model_runtime_identity(model_dir.path(), &top_level_moe, false, false)
                .expect("top-level Qwen3.5 MoE config resolves typed identity");
        let value = serde_json::to_value(identity).unwrap();
        assert_ne!(value["denseFfnGateUp"], "auto");
        assert_ne!(value["bonsaiQ2Simd"], "auto");
    }

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

    #[test]
    fn model_load_measurement_window_is_serialized_by_the_inference_gate() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let barrier = Arc::new(std::sync::Barrier::new(3));
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let barrier = Arc::clone(&barrier);
            let active = Arc::clone(&active);
            let maximum = Arc::clone(&maximum);
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                with_serialized_mlx_load(|| {
                    let concurrent = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(concurrent, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }));
        }
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(maximum.load(Ordering::SeqCst), 1);

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let loader = std::thread::spawn(move || {
            with_serialized_mlx_load(|| {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
        });
        entered_rx.recv().unwrap();
        let (inference_done_tx, inference_done_rx) = std::sync::mpsc::channel();
        let inference = std::thread::spawn(move || {
            let engine = Engine::test_stub("contention");
            let _ = engine.generate_continued(1, &[], 1, &SamplingParams::default());
            inference_done_tx.send(()).unwrap();
        });
        assert!(
            inference_done_rx
                .recv_timeout(std::time::Duration::from_millis(20))
                .is_err(),
            "inference must wait until the measured load window releases the GPU gate"
        );
        release_tx.send(()).unwrap();
        inference_done_rx.recv().unwrap();
        loader.join().unwrap();
        inference.join().unwrap();
    }

    #[test]
    fn post_load_facts_bind_content_and_execution_identity_to_measured_residency() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let model = tempfile::tempdir().unwrap();
        std::fs::write(
            model.path().join("config.json"),
            br#"{
                "model_type":"qwen3_5_moe",
                "max_position_embeddings":131072,
                "num_hidden_layers":40,
                "num_key_value_heads":2,
                "head_dim":256,
                "full_attention_interval":4
            }"#,
        )
        .unwrap();
        std::fs::write(model.path().join("model.safetensors"), b"w").unwrap();
        std::fs::write(
            model.path().join("quantize_config.json"),
            br#"{"quant_method":"eschamoe","bits":3}"#,
        )
        .unwrap();
        let identity = crate::capacity::ModelContentIdentity {
            fingerprint: "sha256:exact".to_owned(),
            artifact_bytes: 12 * GIB,
        };
        let before = higgs_engine::MlxMemorySnapshot {
            active_bytes: GIB,
            peak_bytes: GIB,
            memory_limit_bytes: Some(40 * GIB),
            metal_recommended_working_set_bytes: Some(48 * GIB),
        };
        let after = higgs_engine::MlxMemorySnapshot {
            active_bytes: 12 * GIB,
            peak_bytes: 12 * GIB,
            ..before
        };
        let model_config = ModelConfig {
            path: model.path().display().to_string(),
            name: Some("escha".to_owned()),
            kv_max_session_tokens: 49_152,
            kv_max_retained_bytes: 2 * GIB as usize,
            kv_cache_bytes: GIB as usize,
            ..ModelConfig::default()
        };
        let config = HiggsConfig {
            models: vec![model_config.clone()],
            ..HiggsConfig::default()
        };

        let facts = build_capacity_facts_from_measurements(
            "escha",
            model.path(),
            &model_config,
            &config,
            identity,
            before,
            after,
            crate::capacity::CacheCapabilities::SIMPLE,
        )
        .unwrap();

        assert_eq!(facts.model_fingerprint, "sha256:exact");
        assert_eq!(facts.loaded_model_bytes, 11 * GIB);
        assert_eq!(facts.architectural_max_tokens, 131_072);
        assert_eq!(facts.costs.persistent_bytes_per_token, 20_480);
        assert_eq!(
            facts.quantization,
            r#"{"quant_method":"eschamoe","bits":3}"#
        );
        let escha_mode = if higgs_models::eschamoe::native_mode() {
            "simple:eschamoe-native"
        } else {
            "simple:eschamoe-affine"
        };
        assert_eq!(facts.execution_mode, escha_mode);
        assert_eq!(facts.retained_session_tokens, 49_152);
        assert_eq!(facts.retained_bytes_ceiling, 2 * GIB);
        assert_eq!(facts.prefix_cache_bytes_ceiling, GIB);
        assert_eq!(facts.startup_headroom_bytes, 39 * GIB);
        if let Some(key) = facts.learned_profile_key {
            assert_eq!(key.backend_authority_bytes, 40 * GIB);
        }
    }
}
