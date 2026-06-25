use std::path::Path;
use std::sync::Arc;

use higgs_engine::batch_engine::BatchEngine;
use higgs_engine::chat_template::ChatMessage;
use higgs_engine::engine::{GenerationOutput, StreamingOutput};
use higgs_engine::error::EngineError;
use higgs_engine::mlx_tuning::MlxRuntimeTuning;
use higgs_engine::simple::{CacheStats, SessionGeneration, SimpleEngine};
use higgs_engine::tokenizers::Tokenizer;
use higgs_models::SamplingParams;
use higgs_models::turboquant::KvCacheConfig;
use mlx_rs::Array;

use crate::config::HiggsConfig;
use crate::metrics::MetricsStore;
use crate::router::Router;

/// Unified engine interface wrapping either the simple (serialized) or batch
/// (interleaved) engine. Route handlers interact with this enum exclusively.
pub enum Engine {
    Simple(Box<SimpleEngine>),
    Batch(Box<BatchEngine>),
    #[cfg(test)]
    Stub(String),
}

impl Engine {
    pub fn load_simple<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        tuning: MlxRuntimeTuning,
        raise_wired_limit: bool,
    ) -> Result<Self, EngineError> {
        SimpleEngine::load(dir, kv_cache_config, tuning, raise_wired_limit)
            .map(|e| Self::Simple(Box::new(e)))
    }

    pub fn load_batch<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        raise_wired_limit: bool,
    ) -> Result<Self, EngineError> {
        BatchEngine::load(dir, kv_cache_config, raise_wired_limit).map(|e| Self::Batch(Box::new(e)))
    }

    #[cfg(test)]
    pub fn test_stub(name: &str) -> Self {
        Self::Stub(name.to_owned())
    }

    pub fn model_name(&self) -> &str {
        match self {
            Self::Simple(e) => e.model_name(),
            Self::Batch(e) => e.model_name(),
            #[cfg(test)]
            Self::Stub(name) => name,
        }
    }

    #[cfg_attr(test, allow(clippy::panic))]
    pub fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Simple(e) => e.tokenizer(),
            Self::Batch(e) => e.tokenizer(),
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
            Self::Batch(_) => false,
            #[cfg(test)]
            Self::Stub(_) => false,
        }
    }

    pub fn vlm_image_size(&self) -> Option<i32> {
        match self {
            Self::Simple(e) => e.vlm_image_size(),
            Self::Batch(_) => None,
            #[cfg(test)]
            Self::Stub(_) => None,
        }
    }

    pub fn replace_image_tokens(&self, tokens: &mut [u32]) {
        match self {
            Self::Simple(e) => e.replace_image_tokens(tokens),
            Self::Batch(_) => {}
            #[cfg(test)]
            Self::Stub(_) => {}
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

    /// Render the chat template to its prompt STRING (the exact text
    /// [`Self::prepare_chat_prompt_with_thinking`] tokenizes). Only the Simple
    /// engine, which owns retained session caches, needs this — it lets the
    /// continuation path compute a text-anchored delta against the retained
    /// tokens' own detokenization. Other variants have no retained cache, so
    /// this is unreachable for them.
    pub fn render_chat_prompt_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> Result<String, EngineError> {
        match self {
            Self::Simple(e) => e.render_chat_prompt_with_thinking(messages, tools, enable_thinking),
            Self::Batch(_) => Err(EngineError::Generation(
                "render_chat_prompt_with_thinking is only used by the Simple engine".to_owned(),
            )),
            #[cfg(test)]
            Self::Stub(_) => Ok(String::new()),
        }
    }

    /// The exact token sequence a retained session cache currently holds
    /// (prompt + previously generated tokens), or `None` when no live cache
    /// exists for this `session_id`. Only the Simple engine retains caches.
    pub fn retained_session_tokens(&self, session_id: u64) -> Option<Vec<u32>> {
        match self {
            Self::Simple(e) => e.retained_session_tokens(session_id),
            Self::Batch(_) => None,
            #[cfg(test)]
            Self::Stub(_) => None,
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
        match self {
            Self::Simple(e) => e.generate_continued(session_id, prompt_tokens, max_tokens, params),
            Self::Batch(_) => Err(EngineError::Generation(
                "session_id (continued generation) is only supported by the Simple engine"
                    .to_owned(),
            )),
            #[cfg(test)]
            Self::Stub(_) => Err(EngineError::Generation("test stub".to_owned())),
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
        pixel_values: Option<Array>,
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
            pixel_values,
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
        pixel_values: Option<Array>,
    ) -> Result<GenerationOutput, EngineError> {
        match self {
            Self::Simple(e) => e.generate_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                enable_thinking,
                constraint,
                pixel_values,
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
                pixel_values,
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
        pixel_values: Option<Array>,
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
            pixel_values,
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
        pixel_values: Option<Array>,
    ) -> Result<(), EngineError> {
        match self {
            Self::Simple(e) => e.generate_streaming_with_thinking(
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
                pixel_values,
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
                pixel_values,
            ),
            #[cfg(test)]
            Self::Stub(_) => Err(EngineError::Generation("test stub".to_owned())),
        }
    }

    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        match self {
            Self::Simple(e) => e.embed(token_ids),
            Self::Batch(e) => e.embed(token_ids),
            #[cfg(test)]
            Self::Stub(_) => Ok(Vec::new()),
        }
    }
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
