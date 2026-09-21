use std::path::{Path, PathBuf};
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

use crate::capacity::CapacityRegistry;
use crate::capacity::{ModelCapacityFacts, ModelContentIdentity, fingerprint_model_artifacts};
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
#[cfg(test)]
#[path = "streaming_fixtures.rs"]
pub(crate) mod streaming_fixtures;

static GPU_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub struct RouteTestStub {
    name: String,
    mutations: std::sync::atomic::AtomicU64,
    mutation_sequence: std::sync::Mutex<Vec<String>>,
    retained_sessions: std::sync::Mutex<std::collections::HashMap<u64, Vec<u32>>>,
    /// Optional prompt tokens the stub reports for chat-prompt preparation,
    /// letting route tests shape admission's prompt/suffix charge.
    chat_prompt_tokens: std::sync::Mutex<Vec<u32>>,
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
            retained_sessions: std::sync::Mutex::new(std::collections::HashMap::new()),
            chat_prompt_tokens: std::sync::Mutex::new(Vec::new()),
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
            let continued = sessions.contains_key(&session_id);
            sessions.entry(session_id).or_default();
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
            .keys()
            .copied()
            .collect();
        ids.sort_unstable();
        ids
    }

    fn drop_session(&self, session_id: u64) -> bool {
        let dropped = self
            .retained_sessions
            .lock()
            .unwrap()
            .remove(&session_id)
            .is_some();
        self.record_named_mutation(format!("drop:{session_id}"));
        dropped
    }

    fn lease_session(&self, session_id: u64, ttl_seconds: u32) -> bool {
        let retained = self
            .retained_sessions
            .lock()
            .unwrap()
            .contains_key(&session_id);
        if retained {
            self.record_named_mutation(format!("lease:{session_id}:{ttl_seconds}"));
        }
        retained
    }

    /// Seed retained session tokens so route tests can shape the uncached
    /// suffix fact admission charges.
    fn retain_session_tokens(&self, session_id: u64, tokens: Vec<u32>) {
        self.retained_sessions
            .lock()
            .unwrap()
            .insert(session_id, tokens);
    }

    /// Seed the prompt tokens reported by chat-prompt preparation.
    fn set_chat_prompt_tokens(&self, tokens: Vec<u32>) {
        *self.chat_prompt_tokens.lock().unwrap() = tokens;
    }

    fn chat_prompt_tokens(&self) -> Vec<u32> {
        self.chat_prompt_tokens.lock().unwrap().clone()
    }

    /// Common retained/prompt prefix length, mirroring
    /// `SimpleEngine::retained_session_prefix_len` for route tests.
    fn retained_session_prefix_len(&self, session_id: u64, prompt_tokens: &[u32]) -> Option<usize> {
        self.retained_sessions
            .lock()
            .unwrap()
            .get(&session_id)
            .map(|retained| {
                retained
                    .iter()
                    .zip(prompt_tokens)
                    .take_while(|(retained, prompt)| retained == prompt)
                    .count()
            })
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
                    "vocab": {
                        "[UNK]": 0, "token": 7,
                        "<": 10, ">": 11, "/": 12, "t": 13, "o": 14, "l": 15, "_": 16,
                        "c": 17, "a": 18, "\n": 19, "{": 20, "}": 21, "\"": 22, "n": 23,
                        "m": 24, "e": 25, ":": 26, " ": 27, ",": 28, "w": 29, "h": 30,
                        "r": 31, "s": 32, "g": 33, "u": 34, "y": 35, "R": 36, "i": 37,
                        "b": 38, "d": 39, "f": 40, "p": 41
                    },
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

pub struct EngineRetentionClaim {
    simple: Option<higgs_engine::simple::RetainedReservationClaim>,
    owner_id: u64,
}

impl EngineRetentionClaim {
    pub const fn owner_id(&self) -> u64 {
        self.owner_id
    }
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

    /// Seed retained-session tokens on a stub engine (no-op on real engines).
    #[cfg(test)]
    pub fn test_retain_session_tokens(&self, session_id: u64, tokens: Vec<u32>) {
        if let Self::Stub(stub) = self {
            stub.retain_session_tokens(session_id, tokens);
        }
    }

    /// Seed the prompt tokens a stub engine reports for chat-prompt
    /// preparation (no-op on real engines).
    #[cfg(test)]
    pub fn test_set_chat_prompt_tokens(&self, tokens: Vec<u32>) {
        if let Self::Stub(stub) = self {
            stub.set_chat_prompt_tokens(tokens);
        }
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
                    || stub.name() == "session-prefill-render-spy"
                    || stub.name() == "prompt-limit-mutation-spy"
                    || stub.name() == "capacity-interrupted"
                    || stub.name().starts_with("required-stream-") =>
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

    /// Client-facing tool protocol for the loaded model. The engine keeps the
    /// actual parser private; clients only need to choose native schemas or
    /// the textual fallback.
    pub fn tool_call_mode(&self) -> &'static str {
        match self {
            Self::Simple(e) => e.tool_call_mode(),
            Self::Batch(e) => e.tool_call_mode(),
            #[cfg(test)]
            Self::Stub(_) => "textual",
        }
    }

    /// Client-facing thinking mode for the loaded model.
    pub fn thinking_mode(&self) -> &'static str {
        match self {
            Self::Simple(e) => e.thinking_mode(),
            Self::Batch(e) => e.thinking_mode(),
            #[cfg(test)]
            Self::Stub(_) => "disabled",
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
            Self::Stub(stub) if !stub.chat_prompt_tokens().is_empty() => {
                Ok((stub.chat_prompt_tokens(), PFlashPromptPolicy::default()))
            }
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
                if matches!(stub.name(), "zero-prefix-accept" | "seed-binding") {
                    return stub.drop_session(session_id);
                }
                if stub.name() == "prompt-limit-mutation-spy" {
                    stub.record_mutation();
                }
                false
            }
        }
    }

    /// Best-effort reclamation: an active session remains owned by generation.
    pub fn try_drop_retained_session(&self, session_id: u64) -> bool {
        match self {
            Self::Simple(engine) => engine.try_drop_retained_session(session_id),
            Self::Batch(_) => false,
            #[cfg(test)]
            Self::Stub(_) => false,
        }
    }

    /// Common retained/prompt prefix length — the race-safe fact capacity
    /// admission charges as already cached. `None` means no retained state is
    /// known (callers then charge the full prompt).
    pub fn retained_session_prefix_len(
        &self,
        session_id: u64,
        prompt_tokens: &[u32],
    ) -> Option<usize> {
        match self {
            Self::Simple(e) => e.retained_session_prefix_len(session_id, prompt_tokens),
            Self::Batch(_) => None,
            #[cfg(test)]
            Self::Stub(stub) => stub.retained_session_prefix_len(session_id, prompt_tokens),
        }
    }

    pub fn retained_session_receipt(&self, session_id: u64) -> Option<(usize, usize)> {
        match self {
            Self::Simple(engine) => engine.retained_session_receipt(session_id),
            Self::Batch(_) => None,
            #[cfg(test)]
            Self::Stub(_) => None,
        }
    }

    pub fn reserve_retained_session(&self, session_id: u64) -> Option<EngineRetentionClaim> {
        self.reserve_retained_session_replacing(session_id, &[])
    }

    pub fn reserve_retained_session_replacing(
        &self,
        session_id: u64,
        retired_session_ids: &[u64],
    ) -> Option<EngineRetentionClaim> {
        match self {
            Self::Simple(engine) => engine
                .reserve_retained_session_replacing(session_id, retired_session_ids)
                .map(|simple| EngineRetentionClaim {
                    owner_id: simple.owner_id(),
                    simple: Some(simple),
                }),
            Self::Batch(_) => None,
            #[cfg(test)]
            Self::Stub(stub)
                if matches!(stub.name(), "seed-binding" | "rotated-seed" | "drop-target") =>
            {
                static NEXT_TEST_OWNER: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(1);
                Some(EngineRetentionClaim {
                    simple: None,
                    owner_id: NEXT_TEST_OWNER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                })
            }
            #[cfg(test)]
            Self::Stub(_) => None,
        }
    }

    pub fn release_retained_reservation(&self, claim: EngineRetentionClaim) -> bool {
        match (self, claim.simple) {
            (Self::Simple(engine), Some(simple)) => engine.release_retained_reservation(simple),
            #[cfg(test)]
            (Self::Stub(_), None) => true,
            _ => false,
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
                if matches!(stub.name(), "zero-prefix-accept" | "seed-binding") {
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
            Self::Stub(stub) => matches!(stub.name(), "raw-accept-worker-reject" | "seed-binding"),
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
        assumed_retained_prefix_tokens: u64,
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
                assumed_retained_prefix_tokens,
            ),
            Self::Batch(_) => Err(EngineError::Generation(
                "session_id (session-routed generation) is only supported by the Simple engine"
                    .to_owned(),
            )),
            #[cfg(test)]
            Self::Stub(stub) => {
                let retained_prefix = stub.retained_session_prefix_len(session_id, prompt_tokens);
                // Required continuation depends on retained-state existence,
                // independently of the fixed context admission calculation.
                if (continuation_policy == SessionContinuationPolicy::RequireContinuation
                    && retained_prefix.is_none())
                    || usize::try_from(assumed_retained_prefix_tokens).unwrap_or(usize::MAX)
                        > retained_prefix.unwrap_or(0)
                {
                    return Err(EngineError::RetainedSessionUnavailable(session_id));
                }
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
        assumed_retained_prefix_tokens: u64,
    ) -> Result<(), EngineError> {
        let _gpu = gpu_gate();
        // A request can disconnect while waiting for the serialized GPU worker.
        if sender.is_closed() {
            return Err(EngineError::Cancelled);
        }
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
                assumed_retained_prefix_tokens,
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
            Self::Stub(stub) => {
                let retained_prefix = stub.retained_session_prefix_len(session_id, prompt_tokens);
                // Required continuation depends on retained-state existence,
                // independently of the fixed context admission calculation.
                if (continuation_policy == SessionContinuationPolicy::RequireContinuation
                    && retained_prefix.is_none())
                    || usize::try_from(assumed_retained_prefix_tokens).unwrap_or(usize::MAX)
                        > retained_prefix.unwrap_or(0)
                {
                    if let Some(acceptance) = acceptance.take() {
                        let _ = acceptance.send(Err(session_id));
                    }
                    return Err(EngineError::RetainedSessionUnavailable(session_id));
                }
                if stub.name().starts_with("required-stream-script-") {
                    stub.route_session(session_id);
                    if let Some(acceptance_sender) = acceptance.take() {
                        let _ = acceptance_sender.send(Ok(()));
                    }
                    return streaming_fixtures::emit(stub.name(), sender, prompt_tokens.len());
                }
                if stub.name() == "session-prefill-render-spy" {
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
                } else if stub.name() == "zero-prefix-materialization-fail" {
                    Err(EngineError::Generation(
                        "injected retained-state materialization failure".to_owned(),
                    ))
                } else if stub.name() == "zero-prefix-accept" {
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
                } else {
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
        // A request can disconnect while waiting for the serialized GPU worker.
        if sender.is_closed() {
            return Err(EngineError::Cancelled);
        }
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
            Self::Stub(stub) if stub.name().starts_with("required-stream-script-") => {
                streaming_fixtures::emit(stub.name(), sender, prompt_tokens.len())
            }
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
        // A request can disconnect while waiting for the serialized GPU worker.
        if sender.is_closed() {
            return Err(EngineError::Cancelled);
        }
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
            Self::Stub(stub) if stub.name().starts_with("required-stream-script-") => {
                streaming_fixtures::emit(stub.name(), sender, prompt_tokens.len())
            }
            #[cfg(test)]
            Self::Stub(stub) if stub.name() == "capacity-interrupted" => {
                Err(EngineError::CapacityInterrupted {
                    boot_id: "boot-route-test".to_owned(),
                    generation: 4,
                })
            }
            #[cfg(test)]
            Self::Stub(stub) if stub.name() == "required-stream-valid-call" => {
                // One complete, well-formed tool call: the required/named
                // postcondition must pass and emit exactly one delta.
                sender
                    .blocking_send(StreamingOutput {
                        new_text:
                            "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>"
                                .to_owned(),
                        finished: true,
                        finish_reason: Some("stop".to_owned()),
                        prompt_tokens: u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX),
                        completion_tokens: 13,
                        token_logprob: None,
                        prefill_progress: None,
                    })
                    .map_err(|_| EngineError::Cancelled)
            }
            #[cfg(test)]
            Self::Stub(stub) if stub.name() == "required-stream-leaky-text" => {
                // Visible text leaks outside the tool_call envelope: the
                // required/named postcondition must fail the stream closed.
                sender
                    .blocking_send(StreamingOutput {
                        new_text:
                            "Sure, calling <tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call> hope that helps!"
                                .to_owned(),
                        finished: true,
                        finish_reason: Some("stop".to_owned()),
                        prompt_tokens: u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX),
                        completion_tokens: 21,
                        token_logprob: None,
                        prefill_progress: None,
                    })
                    .map_err(|_| EngineError::Cancelled)
            }
            #[cfg(test)]
            Self::Stub(stub) if stub.name() == "required-stream-capacity-partial-call" => {
                // Partial `<tool_call>` bytes reach the route trackers, then
                // the engine dies with a capacity interruption: the typed
                // terminal must end the required stream with nothing
                // parser-visible after it.
                sender
                    .blocking_send(StreamingOutput {
                        new_text: "<tool_call>\n{\"name\": \"weat".to_owned(),
                        // A pending finish marker must not survive the
                        // capacity terminal on a required stream.
                        finished: true,
                        finish_reason: Some("stop".to_owned()),
                        prompt_tokens: u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX),
                        completion_tokens: 7,
                        token_logprob: None,
                        prefill_progress: None,
                    })
                    .map_err(|_| EngineError::Cancelled)?;
                Err(EngineError::CapacityInterrupted {
                    boot_id: "boot-route-test".to_owned(),
                    generation: 4,
                })
            }
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
                engine
                    .apply_capacity_cache_limits(
                        retained_limit,
                        prefix_limit,
                        if pressure == crate::capacity::MemoryPressure::Critical {
                            higgs_engine::simple::CachePressurePolicy::Critical
                        } else {
                            higgs_engine::simple::CachePressurePolicy::Normal
                        },
                    )
                    .map_err(|floor| {
                        EngineError::Generation(format!(
                            "retained allocation {retained_limit} is below guaranteed floor {floor}"
                        ))
                    })?;
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
    let disk_cache_config = model_cfg.disk_prefix_cache_config(resolved)?;
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
            disk_cache_config,
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
    capacity: &Arc<CapacityRegistry>,
) -> Result<(String, Engine, ModelCapacityFacts), String> {
    if model_cfg.max_context_tokens == 0 {
        return Err("max_context_tokens must be greater than zero".to_owned());
    }
    let identity = fingerprint_model_artifacts(resolved)
        .map_err(|error| format!("failed to fingerprint model artifacts: {error}"))?;
    let (name, engine) = with_serialized_mlx_load(|| {
        // Optional sidecars follow explicit model configuration. Memory samples
        // are diagnostics; only real loader failures reject a load.
        let _optional_load_policy_guard = higgs_models::progress::install_optional_load_policy(
            higgs_models::progress::OptionalLoadPolicy::Allow,
        );
        let (name, engine) = run_model_load_attempt(
            || build_engine(resolved, model_cfg, &config.local),
            || {
                higgs_engine::simple::maybe_clear_mlx_cache(true, "failed model load");
                let memory = higgs_engine::MlxMemorySnapshot::measure().map_err(|error| {
                    format!("failed to remeasure MLX after model load failure: {error}")
                })?;
                capacity.refresh_memory(memory);
                Ok(())
            },
        )?;
        // Release freed conversion buffers from MLX's allocator cache. The
        // successfully loaded model arrays remain referenced and resident.
        higgs_engine::simple::maybe_clear_mlx_cache(
            true,
            "model loaded: release conversion transients",
        );
        // A missing diagnostic sample must not reject a successfully loaded model.
        match higgs_engine::MlxMemorySnapshot::measure() {
            Ok(memory) => {
                capacity.refresh_memory(memory);
            }
            Err(error) => tracing::warn!(%error, "failed to measure MLX after model load"),
        }
        Ok::<_, String>((name, engine))
    })?;
    let cache_capabilities = engine.cache_capabilities();
    let facts = match build_capacity_facts_from_measurements(
        &name,
        resolved,
        model_cfg,
        config,
        identity,
        cache_capabilities,
    ) {
        Ok(facts) => facts,
        Err(error) => {
            let cleanup = release_failed_engine_final(engine, capacity, "failed model publication");
            return match cleanup {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(format!("{error}; cleanup failed: {cleanup_error}")),
            };
        }
    };
    Ok((name, engine, facts))
}

fn run_model_load_attempt<T>(
    load: impl FnOnce() -> Result<T, String>,
    cleanup: impl FnOnce() -> Result<(), String>,
) -> Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(load)) {
        Ok(Ok(loaded)) => Ok(loaded),
        Ok(Err(error)) => match cleanup() {
            Ok(()) => Err(error),
            Err(cleanup_error) => Err(format!("{error}; cleanup failed: {cleanup_error}")),
        },
        Err(_) => match cleanup() {
            Ok(()) => Err("model load panicked after partial allocation".to_owned()),
            Err(cleanup_error) => Err(format!(
                "model load panicked after partial allocation; cleanup failed: {cleanup_error}"
            )),
        },
    }
}

#[doc(hidden)]
pub fn release_failed_engine(engine: Engine, capacity: &CapacityRegistry) {
    #[cfg(test)]
    if matches!(engine, Engine::Stub(_)) {
        drop(engine);
        return;
    }
    if let Err(error) = release_failed_engine_final(engine, capacity, "failed model publication") {
        tracing::error!(%error, "failed engine cleanup was not measurable and final");
    }
}

fn release_failed_engine_final(
    engine: Engine,
    capacity: &CapacityRegistry,
    reason: &'static str,
) -> Result<(), String> {
    with_serialized_mlx_load(|| {
        let shutdown_error = engine.shutdown().err();
        higgs_engine::simple::maybe_clear_mlx_cache(true, reason);
        let memory = higgs_engine::MlxMemorySnapshot::measure()
            .map_err(|error| format!("failed to measure MLX after engine drop: {error}"))?;
        capacity.refresh_memory(memory);
        if let Some(error) = shutdown_error {
            Err(format!(
                "failed to join engine during failed publication cleanup: {error}"
            ))
        } else {
            Ok(())
        }
    })
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

#[cfg(not(test))]
fn refresh_after_engine_drop_locked(
    capacity: &CapacityRegistry,
    reason: &'static str,
) -> Option<crate::capacity::PublishedMemoryMeasurement> {
    let memory = measure_after_engine_drop_locked(reason)?;
    Some(capacity.refresh_memory(memory))
}

#[cfg(not(test))]
fn measure_after_engine_drop_locked(
    reason: &'static str,
) -> Option<higgs_engine::MlxMemorySnapshot> {
    higgs_engine::simple::maybe_clear_mlx_cache(true, reason);
    higgs_engine::MlxMemorySnapshot::measure().ok()
}

fn build_capacity_facts_from_measurements(
    name: &str,
    resolved: &Path,
    model_cfg: &ModelConfig,
    config: &HiggsConfig,
    identity: ModelContentIdentity,
    mut cache_capabilities: crate::capacity::CacheCapabilities,
) -> Result<ModelCapacityFacts, String> {
    let model_json = std::fs::read(resolved.join("config.json"))
        .map_err(|error| format!("failed to read model config: {error}"))?;
    let model_json: serde_json::Value = serde_json::from_slice(&model_json)
        .map_err(|error| format!("failed to parse model config: {error}"))?;
    let architectural_max_tokens = config_u64(&model_json, "max_position_embeddings")
        .filter(|tokens| *tokens > 0)
        .ok_or_else(|| "model config lacks a positive max_position_embeddings".to_owned())?;
    if model_cfg.max_context_tokens == 0 {
        return Err("max_context_tokens must be greater than zero".to_owned());
    }
    let legacy_retained_tokens = u64::try_from(model_cfg.kv_max_session_tokens)
        .map_err(|_| "retained-session token ceiling overflows u64".to_owned())?;
    let output_reserve = u64::from(config.server.max_tokens);
    let retained_session_tokens = if legacy_retained_tokens == 0 {
        0
    } else if let Some(tokens) = legacy_retained_tokens.checked_sub(output_reserve) {
        tokens
    } else {
        cache_capabilities.retained_sessions = false;
        0
    };
    if legacy_retained_tokens > 0 && retained_session_tokens == 0 {
        cache_capabilities.retained_sessions = false;
    }
    let (retained_fixed_bytes_per_session, retained_bytes_per_token) =
        retained_session_geometry(resolved, model_cfg, cache_capabilities)?;
    if retained_bytes_per_token == 0 {
        cache_capabilities.retained_sessions = false;
    }
    Ok(ModelCapacityFacts {
        model: name.to_owned(),
        model_fingerprint: identity.fingerprint,
        architectural_max_tokens,
        retained_session_tokens,
        retained_bytes_ceiling: u64::try_from(model_cfg.kv_max_retained_bytes)
            .map_err(|_| "retained byte ceiling overflows u64".to_owned())?,
        guaranteed_retained_sessions: if cache_capabilities.retained_sessions {
            1
        } else {
            0
        },
        retained_fixed_bytes_per_session,
        retained_bytes_per_token,
        prefix_cache_bytes_ceiling: u64::try_from(model_cfg.kv_cache_config().kv_cache_bytes)
            .map_err(|_| "prefix-cache byte ceiling overflows u64".to_owned())?,
        cache_capabilities,
        configured_total_token_ceiling: Some(u64::from(model_cfg.max_context_tokens)),
        configured_output_token_ceiling: Some(u64::from(config.server.max_tokens)),
    })
}

fn retained_session_geometry(
    model: &Path,
    model_cfg: &ModelConfig,
    cache_capabilities: crate::capacity::CacheCapabilities,
) -> Result<(u64, u64), String> {
    if !cache_capabilities.retained_sessions {
        return Ok((0, 0));
    }
    let transient = higgs_engine::TransientPrefillEstimate {
        base_bytes: 0,
        bytes_per_prompt_token: 0,
        bytes_per_chunk_token: 0,
        max_prompt_tokens: u64::MAX,
        max_chunk_tokens: u64::MAX,
    };
    let draft_path = model_cfg
        .draft_model
        .as_deref()
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HIGGS_DFLASH_PATH").map(PathBuf::from));
    let Some(cost) = higgs_engine::EngineCostDescription::runtime_pair_from_model_dirs(
        model,
        draft_path.as_deref(),
        transient,
    ) else {
        return Ok((0, 0));
    };
    Ok((
        cost.fixed_live_session_bytes,
        cost.persistent_bytes_per_token,
    ))
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
    retention_bindings:
        std::sync::Mutex<std::collections::HashMap<(String, u64), (String, u64, bool, u64)>>,
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
            retention_bindings: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn claim_retention_seed(
        &self,
        model: &str,
        session_id: u64,
        revision: &str,
        epoch: u64,
    ) -> bool {
        self.claim_retention_seed_replacing(model, session_id, revision, epoch, &[])
    }

    pub fn claim_retention_seed_replacing(
        &self,
        model: &str,
        session_id: u64,
        revision: &str,
        epoch: u64,
        retired_session_ids: &[u64],
    ) -> bool {
        let mut bindings = self
            .retention_bindings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for retired_session_id in retired_session_ids {
            bindings.remove(&(model.to_owned(), *retired_session_id));
        }
        let key = (model.to_owned(), session_id);
        if bindings.contains_key(&key)
            || bindings.keys().any(|(bound_model, _)| bound_model == model)
        {
            return false;
        }
        bindings.insert(key, (revision.to_owned(), epoch, false, 0));
        true
    }

    pub fn claim_exclusively_reserved_seed(
        &self,
        model: &str,
        session_id: u64,
        revision: &str,
        epoch: u64,
        _reservation: &EngineRetentionClaim,
    ) -> bool {
        let mut bindings = self
            .retention_bindings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        bindings.retain(|(bound_model, _), _| bound_model != model);
        bindings.insert(
            (model.to_owned(), session_id),
            (revision.to_owned(), epoch, false, _reservation.owner_id()),
        );
        true
    }

    pub fn publish_retention_seed(&self, model: &str, session_id: u64) -> bool {
        self.publish_owned_retention_seed(model, session_id, 0)
    }

    pub fn publish_owned_retention_seed(
        &self,
        model: &str,
        session_id: u64,
        owner_id: u64,
    ) -> bool {
        let mut bindings = self
            .retention_bindings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(binding) = bindings.get_mut(&(model.to_owned(), session_id)) else {
            return false;
        };
        if binding.3 != owner_id {
            return false;
        }
        binding.2 = true;
        true
    }

    pub fn abort_retention_seed(&self, model: &str, session_id: u64) {
        self.abort_owned_retention_seed(model, session_id, 0);
    }

    pub fn abort_owned_retention_seed(&self, model: &str, session_id: u64, owner_id: u64) {
        let key = (model.to_owned(), session_id);
        let mut bindings = self
            .retention_bindings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if bindings
            .get(&key)
            .is_some_and(|binding| binding.3 == owner_id)
        {
            bindings.remove(&key);
        }
    }

    pub fn retention_binding_matches(
        &self,
        model: &str,
        session_id: u64,
        revision: &str,
        epoch: u64,
    ) -> bool {
        self.retention_bindings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&(model.to_owned(), session_id))
            .is_some_and(|(bound_revision, bound_epoch, published, _)| {
                *published && bound_revision == revision && *bound_epoch == epoch
            })
    }

    pub fn drop_retention_bindings(&self, model: &str, session_ids: &[u64]) {
        let mut bindings = self
            .retention_bindings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for session_id in session_ids {
            bindings.remove(&(model.to_owned(), *session_id));
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
    fn loader_unwind_drops_partial_state_before_cleanup() {
        use std::cell::RefCell;
        use std::rc::Rc;

        struct Partial(Rc<RefCell<Vec<&'static str>>>);
        impl Drop for Partial {
            fn drop(&mut self) {
                self.0.borrow_mut().push("drop");
            }
        }

        let events = Rc::new(RefCell::new(Vec::new()));
        let load_events = Rc::clone(&events);
        let cleanup_events = Rc::clone(&events);
        let capacity = CapacityRegistry::new(Vec::<String>::new());
        let cleanup_capacity = Arc::clone(&capacity);
        let outcome: Result<(), String> = run_model_load_attempt(
            move || {
                let _partial = Partial(load_events);
                panic!("synthetic loader panic");
            },
            move || {
                cleanup_events.borrow_mut().push("cache-clear");
                cleanup_events.borrow_mut().push("measure");
                let published = cleanup_capacity.refresh_memory(higgs_engine::MlxMemorySnapshot {
                    active_bytes: 17,
                    ..higgs_engine::MlxMemorySnapshot::default()
                });
                assert_eq!(published.revision(), 1);
                cleanup_events.borrow_mut().push("publish");
                Ok(())
            },
        );
        assert!(outcome.unwrap_err().contains("panicked"));
        assert_eq!(
            *events.borrow(),
            vec!["drop", "cache-clear", "measure", "publish"]
        );
    }

    #[test]
    fn loader_cleanup_failure_is_not_hidden_by_original_error() {
        let outcome: Result<(), String> = run_model_load_attempt(
            || Err("synthetic load failure".to_owned()),
            || Err("synthetic remeasure failure".to_owned()),
        );
        assert_eq!(
            outcome.unwrap_err(),
            "synthetic load failure; cleanup failed: synthetic remeasure failure"
        );
    }

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
        let model = tempfile::tempdir().unwrap();
        // An engine-supported architecture needs no KV cost-estimator geometry.
        std::fs::write(
            model.path().join("config.json"),
            br#"{"max_position_embeddings":8192,"num_hidden_layers":8,"num_key_value_heads":4,"hidden_size":1024,"num_attention_heads":8}"#,
        )
        .unwrap();
        let model_config = ModelConfig::default();
        let facts = build_capacity_facts_from_measurements(
            "model",
            model.path(),
            &model_config,
            &HiggsConfig::default(),
            ModelContentIdentity {
                fingerprint: "sha256:exact".to_owned(),
                artifact_bytes: 1,
            },
            crate::capacity::CacheCapabilities::SIMPLE,
        )
        .unwrap();
        assert_eq!(facts.model_fingerprint, "sha256:exact");
        assert_eq!(facts.architectural_max_tokens, 8192);
        assert_eq!(facts.configured_total_token_ceiling, Some(65_536));
        assert_eq!(facts.prefix_cache_bytes_ceiling, 1_073_741_824);
        assert_eq!(facts.retained_bytes_ceiling, 2_147_483_648);
        assert_eq!(facts.guaranteed_retained_sessions, 1);
        assert_eq!(facts.retained_session_tokens, 28_672);
        assert!(facts.retained_bytes_per_token > 0);
    }

    #[test]
    fn unknown_retained_geometry_disables_only_retention_capability() {
        let model = tempfile::tempdir().unwrap();
        std::fs::write(
            model.path().join("config.json"),
            br#"{"max_position_embeddings":8192,"num_hidden_layers":8}"#,
        )
        .unwrap();
        let facts = build_capacity_facts_from_measurements(
            "stateless-model",
            model.path(),
            &ModelConfig::default(),
            &HiggsConfig::default(),
            ModelContentIdentity {
                fingerprint: "sha256:unknown-geometry".to_owned(),
                artifact_bytes: 1,
            },
            crate::capacity::CacheCapabilities::SIMPLE,
        )
        .expect("unsupported retained geometry must not fail stateless model registration");
        assert!(!facts.cache_capabilities.retained_sessions);
        assert_eq!(facts.guaranteed_retained_sessions, 0);
        assert_eq!(facts.retained_bytes_per_token, 0);
    }

    #[test]
    fn legacy_token_cap_consumed_by_output_reserve_disables_retention() {
        let model = tempfile::tempdir().unwrap();
        std::fs::write(
            model.path().join("config.json"),
            br#"{"max_position_embeddings":8192,"num_hidden_layers":8,"num_key_value_heads":4,"hidden_size":1024,"num_attention_heads":8}"#,
        )
        .unwrap();
        let mut model_config = ModelConfig::default();
        model_config.kv_max_session_tokens = 1024;
        let mut config = HiggsConfig::default();
        config.server.max_tokens = 1024;

        let facts = build_capacity_facts_from_measurements(
            "model",
            model.path(),
            &model_config,
            &config,
            ModelContentIdentity {
                fingerprint: "sha256:legacy-cap".to_owned(),
                artifact_bytes: 1,
            },
            crate::capacity::CacheCapabilities::SIMPLE,
        )
        .unwrap();

        assert!(!facts.cache_capabilities.retained_sessions);
        assert_eq!(facts.guaranteed_retained_sessions, 0);
        assert_eq!(facts.retained_session_tokens, 0);
    }

    #[test]
    fn exclusive_engine_claim_replaces_binding_left_by_idle_expiry() {
        let (state, _) = crate::capacity::retained_contract_route_test_state(
            "model",
            4 * 1024 * 1024 * 1024,
            128 * 1024,
        );
        assert!(state.claim_retention_seed("model", 7, "old", 1));
        assert!(state.publish_retention_seed("model", 7));
        let claim = EngineRetentionClaim {
            simple: None,
            owner_id: 7,
        };
        assert!(state.claim_exclusively_reserved_seed("model", 8, "new", 2, &claim));
        assert!(!state.retention_binding_matches("model", 7, "old", 1));
        assert!(!state.retention_binding_matches("model", 8, "new", 2));
        state.abort_owned_retention_seed("model", 8, 6);
        assert!(state.publish_owned_retention_seed("model", 8, 7));
        assert!(state.retention_binding_matches("model", 8, "new", 2));
    }
}
