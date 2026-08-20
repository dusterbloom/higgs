use serde::{Deserialize, Serialize};

/// POST /v1/chat/completions request body.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    /// Optional Higgs extension controlling prefix-cache participation.
    /// `"bypass"` still runs inference but neither reads nor writes the
    /// stateless prefix cache.
    #[serde(default)]
    pub cache_mode: Option<String>,
    /// Maximum number of tokens to generate.
    ///
    /// Accepts `max_completion_tokens` and `max_output_tokens` aliases.
    #[serde(default, alias = "max_completion_tokens", alias = "max_output_tokens")]
    pub max_tokens: Option<u32>,
    /// Reject when the fully rendered prompt exceeds this exact token count.
    #[serde(default)]
    pub max_prompt_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// llama.cpp/Ollama alias for [`Self::repetition_penalty`]. Accepted as a
    /// separate field (never an `alias`) so clients that send both names — e.g.
    /// some local backends emit `repeat_penalty` alongside a vLLM-style
    /// `repetition_penalty` — don't get a "duplicate field" 400. Merged at
    /// sampling-param build time, with `repetition_penalty` taking precedence.
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Per-request speculative-decoding method: `auto` (default), `dflash`,
    /// `mtp`, or `none`. `auto` uses the `DFlash` drafter when one is loaded
    /// (including while streaming), else the built-in MTP head.
    #[serde(default)]
    pub speculation: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// Optional OpenAI-style reasoning controls.
    ///
    /// When omitted, Higgs chooses a model-specific default. Set `effort` to
    /// a non-empty value such as `"low"` to explicitly enable reasoning.
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    /// When true, streaming responses include `prompt_progress` chunks during
    /// chunked prefill (llama.cpp-compatible shape: `{total, cache,
    /// processed, time_ms}`). Ignored for non-streaming requests.
    #[serde(default)]
    pub return_progress: Option<bool>,
    /// Optional Higgs extension naming a disk prefix-cache checkpoint to load/store.
    #[serde(default)]
    pub checkpoint_id: Option<String>,
    /// Max `<think>` tokens before `</think>` is force-closed (de-facto local
    /// extension; sent by clients like nanobot's `/thinking N`). `None` falls
    /// back to the engine default budget.
    #[serde(default)]
    pub reasoning_budget: Option<u32>,
    /// Jinja chat-template kwargs (vLLM/Qwen convention). Only
    /// `enable_thinking` is honored: it overrides per-request whether the model
    /// reasons.
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Top-level alias for `chat_template_kwargs.enable_thinking`, accepted
    /// because many OpenAI-compatible clients send the toggle here. When both
    /// are present, `chat_template_kwargs.enable_thinking` wins; otherwise this
    /// value is used.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// Opt-in multi-turn KV-cache reuse. When set (non-streaming, Simple engine
    /// only) the conversation's KV cache is retained across turns so that a
    /// continued turn prefills only the new suffix instead of the full history.
    /// Omitted by default — behavior is unchanged when absent.
    #[serde(default)]
    pub session_id: Option<u64>,
    /// Best-effort idle-eviction lease for an already retained session.
    #[serde(default)]
    pub session_lease: Option<SessionLease>,
    /// Whether a missing retained continuation may cold-prefill.
    #[serde(default)]
    pub session_cache_policy: Option<SessionCachePolicy>,
    /// Optional Higgs extension: drop a retained per-session KV cache before
    /// serving this request. This is for logical session resets; it does not
    /// clear exact radix/disk prefix caches.
    #[serde(default)]
    pub drop_session_id: Option<u64>,
    /// Optional Higgs extension: drop multiple retained per-session KV caches
    /// before serving this request. This is the batched form of
    /// `drop_session_id`; both fields may be supplied and are de-duplicated by
    /// the route.
    #[serde(default)]
    pub drop_session_ids: Option<Vec<u64>>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct SessionLease {
    pub session_id: u64,
    pub ttl_seconds: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionCachePolicy {
    BestEffort,
    RequireContinuation,
}

/// Subset of `chat_template_kwargs` that Higgs acts on.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatTemplateKwargs {
    /// Per-request override for the model's reasoning mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
}

/// Optional request-level controls for streaming responses.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

/// Optional reasoning controls accepted on chat completion requests.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReasoningConfig {
    /// Requested reasoning effort level.
    ///
    /// Higgs currently treats any non-empty value as an explicit opt-in to the
    /// model's reasoning / thinking mode and preserves the original string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

/// Response format specification.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseFormat {
    pub r#type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
}

/// Message content: either a plain string or an array of content parts (for multimodal).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract the concatenated text from all text parts.
    pub fn text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Extract image URLs from content parts (base64 data URIs or HTTP URLs).
    pub fn image_urls(&self) -> Vec<&str> {
        match self {
            Self::Text(_) => vec![],
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.as_str()),
                    ContentPart::Text { .. } => None,
                })
                .collect(),
        }
    }

    /// Whether this content contains any images.
    pub fn has_images(&self) -> bool {
        matches!(self, Self::Parts(parts) if parts.iter().any(|p| matches!(p, ContentPart::ImageUrl { .. })))
    }
}

/// A content part in a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// An image URL reference (base64 data URI or HTTP URL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    /// `OpenAI` `detail` resolution control (`auto` / `low` / `high`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<higgs_models::vision::ImageDetail>,
}

/// A message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool call in the `OpenAI` format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ToolCallFunction,
}

/// The function details of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// Stop sequence: single string or array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequence {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v,
        }
    }

    /// Extract stop sequences from an optional value, returning empty vec if None.
    pub fn extract(stop: Option<Self>) -> Vec<String> {
        stop.map_or_else(Vec::new, Self::into_vec)
    }
}

/// POST /v1/chat/completions response (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: CompletionUsage,
}

/// A choice in a chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Logprob data for a completion choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprob>,
}

/// Logprob information for a single generated token.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub top_logprobs: Vec<TopLogprob>,
}

/// A top-logprob entry (one of the most likely tokens at a given position).
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
}

/// Streaming chunk for /v1/chat/completions.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    /// Optional token usage summary attached to the terminal stream chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<CompletionUsage>,
}

/// A choice in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// A tool call delta for streaming.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

/// Function delta in a streaming tool call.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// POST /v1/completions request body.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    /// Maximum number of tokens to generate.
    ///
    /// Accepts `max_completion_tokens` and `max_output_tokens` aliases.
    #[serde(default, alias = "max_completion_tokens", alias = "max_output_tokens")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// llama.cpp/Ollama alias for [`Self::repetition_penalty`]. Accepted as a
    /// separate field (never an `alias`) so clients that send both names don't
    /// get a "duplicate field" 400. Merged at sampling-param build time via
    /// [`merge_repetition_penalty`], taking the stronger (higher) control so a
    /// weaker default can't defeat a repetition-loop safeguard.
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// Optional Higgs extension naming a disk prefix-cache checkpoint to load/store.
    #[serde(default)]
    pub checkpoint_id: Option<String>,
}

/// Merge an OpenAI/vLLM `repetition_penalty` with the llama.cpp/Ollama
/// `repeat_penalty` alias. Some clients (e.g. nanobot) send both on the same
/// request — `repetition_penalty` from a model-config default and
/// `repeat_penalty` as a per-model-class loop safeguard. We must accept both
/// without a "duplicate field" 400, and we take the stronger control (higher
/// value, since repetition penalties above 1.0 suppress loops) so a weaker
/// default can never silently disable the safeguard.
pub fn merge_repetition_penalty(repetition: Option<f32>, repeat: Option<f32>) -> Option<f32> {
    match (repetition, repeat) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// POST /v1/completions response (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
}

/// A choice in a completions response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Streaming chunk for /v1/completions.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

/// A choice in a completions streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunkChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Breakdown of the prompt token count (OpenAI `prompt_tokens_details`).
///
/// Only `cached_tokens` is populated: the number of prompt tokens served from
/// reused KV state (session continuation or radix prefix cache) instead of being
/// re-prefilled this turn. Clients read this as `usage.prompt_tokens_details.cached_tokens`.
#[derive(Debug, Clone, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// OpenAI-shape prompt breakdown. Omitted from the wire when no prompt
    /// tokens were served from cache, so `cached_tokens: 0` never masquerades as
    /// a measured zero for paths that don't track reuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Higgs extension emitted as `1` only after a lease is confirmed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub higgs_session_lease_active: Option<u32>,
}

impl CompletionUsage {
    /// Build a usage block. `cached_tokens` is the count of prompt tokens
    /// served from reused KV; when it is 0 the `prompt_tokens_details` field is
    /// omitted entirely (OpenAI clients treat a missing block as "no reuse").
    #[must_use]
    pub fn new(prompt_tokens: u32, completion_tokens: u32, cached_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: (cached_tokens > 0)
                .then_some(PromptTokensDetails { cached_tokens }),
            higgs_session_lease_active: None,
        }
    }

    #[must_use]
    pub fn with_session_lease_active(mut self, active: bool) -> Self {
        self.higgs_session_lease_active = active.then_some(1);
        self
    }
}

/// GET /v1/models response.
#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
    /// higgs extension (additive, `OpenAI` clients ignore unknown keys): whether
    /// runtime model load/switch is enabled (`local.allow_runtime_model_load`).
    pub runtime_model_load: bool,
}

/// A model in the models list.
#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
    /// higgs extension (additive): whether this model accepts image input (VLM).
    pub vision: bool,
}

/// POST /v1/embeddings request body.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    #[allow(dead_code)] // Required for API deserialization compatibility
    pub encoding_format: Option<String>,
}

/// Embedding input: single string or array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

/// POST /v1/embeddings response.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// A single embedding result.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: u32,
}

/// Usage for embeddings.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_reports_cached_tokens_only_when_nonzero() {
        // Reuse happened: OpenAI-shape `prompt_tokens_details.cached_tokens`.
        let reused = serde_json::to_value(CompletionUsage::new(100, 20, 80)).unwrap();
        assert_eq!(reused["prompt_tokens"], 100);
        assert_eq!(reused["total_tokens"], 120);
        assert_eq!(reused["prompt_tokens_details"]["cached_tokens"], 80);

        // Cold prefill: the block is omitted so a client never reads a
        // fabricated `cached_tokens: 0`.
        let cold = serde_json::to_value(CompletionUsage::new(100, 20, 0)).unwrap();
        assert!(cold.get("prompt_tokens_details").is_none());
    }

    /// Deserialize a chat completion request from JSON with a single user message
    /// and one extra field merged in (e.g., `"max_tokens": 0`).
    fn chat_request_with(extra_field: &str) -> ChatCompletionRequest {
        let json = format!(
            r#"{{"model": "m", "messages": [{{"role": "user", "content": "hi"}}], {extra_field}}}"#,
        );
        serde_json::from_str(&json).unwrap()
    }

    fn make_chat_chunk(
        id: &str,
        delta: ChatCompletionDelta,
        finish_reason: Option<String>,
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: id.to_owned(),
            object: "chat.completion.chunk",
            created: 1_700_000_000,
            model: "test".to_owned(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta,
                finish_reason,
                logprobs: None,
            }],
            usage: None,
        }
    }

    fn make_empty_delta_chunk(id: &str, finish_reason: Option<String>) -> ChatCompletionChunk {
        make_chat_chunk(
            id,
            ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason,
        )
    }

    fn make_usage(prompt: u32, completion: u32) -> CompletionUsage {
        CompletionUsage::new(prompt, completion, 0)
    }

    #[test]
    fn test_chat_request_minimal_deserialization() {
        let json = r#"{"model": "test", "messages": [{"role": "user", "content": "hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.messages.len(), 1);
        assert!(req.stream.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.reasoning.is_none());
        assert!(req.cache_mode.is_none());
        assert!(req.max_prompt_tokens.is_none());
        assert!(req.session_lease.is_none());
        assert!(req.session_cache_policy.is_none());
    }

    #[test]
    fn chat_request_parses_prefill_and_session_lease_controls() {
        let req = chat_request_with(
            r#""max_prompt_tokens": 32768,
                "session_lease": {"session_id": 41, "ttl_seconds": 300},
                "session_cache_policy": "require_continuation""#,
        );

        assert_eq!(req.max_prompt_tokens, Some(32_768));
        let lease = req.session_lease.expect("session lease");
        assert_eq!(lease.session_id, 41);
        assert_eq!(lease.ttl_seconds, 300);
        assert_eq!(
            req.session_cache_policy,
            Some(SessionCachePolicy::RequireContinuation)
        );
    }

    #[test]
    fn chat_request_rejects_unknown_session_cache_policy() {
        let json = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "session_cache_policy": "cold_fallback"
        }"#;
        assert!(serde_json::from_str::<ChatCompletionRequest>(json).is_err());
    }

    #[test]
    fn usage_emits_confirmed_session_lease_only() {
        let inactive = serde_json::to_value(CompletionUsage::new(8, 0, 0)).unwrap();
        assert!(inactive.get("higgs_session_lease_active").is_none());

        let active =
            serde_json::to_value(CompletionUsage::new(8, 0, 0).with_session_lease_active(true))
                .unwrap();
        assert_eq!(active["higgs_session_lease_active"], 1);
    }

    #[test]
    fn test_chat_request_cache_bypass_deserialization() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "."}],
            "cache_mode": "bypass"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.cache_mode.as_deref(), Some("bypass"));
    }

    #[test]
    fn test_chat_request_full_deserialization() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": true,
            "stop": ["\n", "END"]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.stream, Some(true));
        assert!(req.reasoning.is_none());
    }

    #[test]
    fn test_chat_request_reasoning_deserialization() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "none"}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.reasoning.and_then(|reasoning| reasoning.effort),
            Some("none".to_owned())
        );
    }

    #[test]
    fn test_chat_request_drop_session_id_deserialization() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "session_id": 9,
            "drop_session_id": 8,
            "drop_session_ids": [7, 8]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.session_id, Some(9));
        assert_eq!(req.drop_session_id, Some(8));
        assert_eq!(req.drop_session_ids, Some(vec![7, 8]));
    }

    #[test]
    fn test_stop_sequence_single() {
        let json = r#"{"model": "m", "messages": [], "stop": "END"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.stop, Some(StopSequence::Single(_))));
    }

    #[test]
    fn test_chat_response_serialization() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_owned(),
            object: "chat.completion",
            created: 1_234_567_890,
            model: "test".to_owned(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_owned(),
                    content: Some(MessageContent::Text("Hello!".to_owned())),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".to_owned(),
                logprobs: None,
            }],
            usage: make_usage(5, 1),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("chat.completion"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_completion_request_deserialization() {
        let json = r#"{"model": "test", "prompt": "Once upon a time"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "Once upon a time");
    }

    #[test]
    fn test_model_list_serialization() {
        let list = ModelList {
            object: "list",
            data: vec![ModelObject {
                id: "test-model".to_owned(),
                object: "model",
                created: 1_234_567_890,
                owned_by: "local".to_owned(),
                vision: false,
            }],
            runtime_model_load: false,
        };
        let json = serde_json::to_string(&list).unwrap();
        assert!(json.contains("test-model"));
    }

    #[test]
    fn test_response_format_json_mode() {
        let req = chat_request_with(r#""response_format": {"type": "json_object"}"#);
        let fmt = req.response_format.unwrap();
        assert_eq!(fmt.r#type, "json_object");
        assert!(fmt.json_schema.is_none());
    }

    #[test]
    fn test_response_format_json_schema() {
        let req = chat_request_with(
            r#""response_format": {"type": "json_schema", "json_schema": {"type": "object", "properties": {"name": {"type": "string"}}}}"#,
        );
        let fmt = req.response_format.unwrap();
        assert_eq!(fmt.r#type, "json_schema");
        assert!(fmt.json_schema.is_some());
    }

    #[test]
    fn test_tool_call_serialization() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_owned(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_123".to_owned(),
                r#type: "function".to_owned(),
                function: ToolCallFunction {
                    name: "get_weather".to_owned(),
                    arguments: r#"{"city":"London"}"#.to_owned(),
                },
            }]),
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("call_123"));
        assert!(json.contains("get_weather"));
        assert!(!json.contains("\"content\""));
    }

    #[test]
    fn test_tool_call_message_deserialization() {
        let json = r#"{
            "role": "tool",
            "content": "72 degrees",
            "tool_call_id": "call_123"
        }"#;
        let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id, Some("call_123".to_owned()));
    }

    #[test]
    fn test_embedding_request_single() {
        let json = r#"{"model": "test", "input": "Hello world"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Single(_)));
    }

    #[test]
    fn test_embedding_request_multiple() {
        let json = r#"{"model": "test", "input": ["Hello", "World"]}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Multiple(_)));
    }

    #[test]
    fn test_embedding_response_serialization() {
        let resp = EmbeddingResponse {
            object: "list",
            data: vec![EmbeddingObject {
                object: "embedding",
                embedding: vec![0.1, 0.2, 0.3],
                index: 0,
            }],
            model: "test".to_owned(),
            usage: EmbeddingUsage {
                prompt_tokens: 3,
                total_tokens: 3,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("embedding"));
        assert!(json.contains("0.1"));
    }

    #[test]
    fn test_stop_sequence_into_vec() {
        let single = StopSequence::Single("END".to_owned());
        assert_eq!(single.into_vec(), vec!["END"]);

        let multiple = StopSequence::Multiple(vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]);
        assert_eq!(multiple.into_vec(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_stop_sequence_extract() {
        assert!(StopSequence::extract(None).is_empty());

        let single = StopSequence::extract(Some(StopSequence::Single("x".to_owned())));
        assert_eq!(single, vec!["x"]);
    }

    #[test]
    fn test_stop_sequence_multiple_deserialization() {
        let json = r#"{"model": "m", "messages": [], "stop": ["a", "b"]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.stop, Some(StopSequence::Multiple(_))));
    }

    #[test]
    fn test_chat_completion_chunk_serialization() {
        let chunk = make_chat_chunk(
            "chatcmpl-123",
            ChatCompletionDelta {
                role: Some("assistant".to_owned()),
                content: Some("Hi".to_owned()),
                reasoning_content: None,
                tool_calls: None,
            },
            None,
        );
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("chat.completion.chunk"));
    }

    #[test]
    fn test_completion_chunk_serialization() {
        let chunk = CompletionChunk {
            id: "cmpl-123".to_owned(),
            object: "text_completion",
            created: 1_234_567_890,
            model: "test".to_owned(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text: "hello".to_owned(),
                finish_reason: None,
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("text_completion"));
    }

    #[test]
    fn test_chat_completion_delta_skips_none_fields() {
        let delta = ChatCompletionDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: None,
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_chat_request_with_max_tokens_zero() {
        let req = chat_request_with(r#""max_tokens": 0"#);
        assert_eq!(req.max_tokens, Some(0));
    }

    #[test]
    fn test_chat_request_accepts_max_completion_tokens_alias() {
        let req = chat_request_with(r#""max_completion_tokens": 100"#);
        assert_eq!(req.max_tokens, Some(100));
    }

    #[test]
    fn test_chat_request_accepts_max_output_tokens_alias() {
        let req = chat_request_with(r#""max_output_tokens": 100"#);
        assert_eq!(req.max_tokens, Some(100));
    }

    #[test]
    fn test_chat_request_with_temperature_zero() {
        let req = chat_request_with(r#""temperature": 0.0"#);
        assert!((req.temperature.unwrap()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chat_request_with_top_p_zero() {
        let req = chat_request_with(r#""top_p": 0.0"#);
        assert!((req.top_p.unwrap()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chat_request_with_top_p_one() {
        let req = chat_request_with(r#""top_p": 1.0"#);
        assert!((req.top_p.unwrap() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_response_format_with_json_schema_type() {
        let json = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
                }
            }
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let fmt = req.response_format.unwrap();
        assert_eq!(fmt.r#type, "json_schema");
        assert!(fmt.json_schema.is_some());
    }

    #[test]
    fn test_embedding_input_single_vs_array() {
        let single_json = r#"{"model": "m", "input": "hello"}"#;
        let single: EmbeddingRequest = serde_json::from_str(single_json).unwrap();
        assert!(matches!(single.input, EmbeddingInput::Single(_)));

        let array_json = r#"{"model": "m", "input": ["hello", "world"]}"#;
        let array: EmbeddingRequest = serde_json::from_str(array_json).unwrap();
        assert!(matches!(array.input, EmbeddingInput::Multiple(_)));
    }

    #[test]
    fn test_completion_request_all_optional_fields_missing() {
        let json = r#"{"model": "m", "prompt": "test"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.max_tokens.is_none());
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.stream.is_none());
        assert!(req.stop.is_none());
    }

    #[test]
    fn test_completion_request_all_optional_fields_present() {
        let json = r#"{
            "model": "m",
            "prompt": "test",
            "max_tokens": 512,
            "temperature": 0.5,
            "top_p": 0.8,
            "stream": false,
            "stop": ["END"]
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, Some(512));
        assert!((req.temperature.unwrap() - 0.5).abs() < f32::EPSILON);
        assert!((req.top_p.unwrap() - 0.8).abs() < f32::EPSILON);
        assert_eq!(req.stream, Some(false));
        assert!(req.stop.is_some());
    }

    #[test]
    fn test_completion_request_accepts_max_completion_tokens_alias() {
        let json = r#"{
            "model": "m",
            "prompt": "test",
            "max_completion_tokens": 100
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, Some(100));
    }

    #[test]
    fn test_completion_request_accepts_repeat_penalty_field() {
        // llama.cpp/Ollama clients send `repeat_penalty`; higgs reads it as a
        // dedicated field and merges it into `repetition_penalty` at sampling
        // build time, so local repetition guards are not silently dropped.
        let json = r#"{
            "model": "m",
            "prompt": "test",
            "repeat_penalty": 1.1
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.repeat_penalty, Some(1.1));
    }

    #[test]
    fn test_chat_request_accepts_repeat_penalty_field() {
        let json = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "repeat_penalty": 1.15
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.repeat_penalty, Some(1.15));
    }

    #[test]
    fn test_chat_request_accepts_both_repetition_and_repeat_penalty() {
        // Some clients emit both names in one body. With `repeat_penalty` as a
        // serde alias this 400s with "duplicate field repetition_penalty"; as a
        // dedicated field it must parse cleanly, and `merge_repetition_penalty`
        // must take the stronger (higher) control so a weaker default can't
        // defeat a loop safeguard.
        let json = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "repetition_penalty": 1.0,
            "repeat_penalty": 1.15
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.repetition_penalty, Some(1.0));
        assert_eq!(req.repeat_penalty, Some(1.15));
        assert_eq!(
            merge_repetition_penalty(req.repetition_penalty, req.repeat_penalty),
            Some(1.15)
        );
    }

    #[test]
    fn test_merge_repetition_penalty_takes_max() {
        // repetition_penalty is the weaker config default; repeat_penalty is the
        // per-model-class safeguard. The safeguard (higher) must win.
        assert_eq!(merge_repetition_penalty(Some(1.0), Some(1.1)), Some(1.1));
        assert_eq!(merge_repetition_penalty(Some(1.3), Some(1.1)), Some(1.3));
        assert_eq!(merge_repetition_penalty(Some(1.1), None), Some(1.1));
        assert_eq!(merge_repetition_penalty(None, Some(1.1)), Some(1.1));
        assert_eq!(merge_repetition_penalty(None, None), None);
    }

    #[test]
    fn test_completion_request_accepts_max_output_tokens_alias() {
        let json = r#"{
            "model": "m",
            "prompt": "test",
            "max_output_tokens": 100
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, Some(100));
    }

    #[test]
    fn test_extra_unknown_fields_silently_ignored() {
        let json = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "unknown_field_xyz": 42,
            "another_unknown": "hello"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "m");
    }

    #[test]
    fn test_chat_message_with_null_content() {
        let json = r#"{"role": "assistant", "content": null}"#;
        let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
    }

    #[test]
    fn test_chat_message_without_content_field() {
        let json = r#"{"role": "assistant"}"#;
        let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
        assert!(msg.content.is_none());
    }

    #[test]
    fn test_completion_request_with_stop_sequences_as_array() {
        let json = r#"{"model": "m", "prompt": "test", "stop": ["END", "\n", "DONE"]}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        match req.stop.unwrap() {
            StopSequence::Multiple(v) => assert_eq!(v.len(), 3),
            StopSequence::Single(_) => panic!("expected Multiple variant"),
        }
    }

    #[test]
    fn test_chat_response_with_tool_calls_serialization() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-tools".to_owned(),
            object: "chat.completion",
            created: 1_700_000_000,
            model: "test".to_owned(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_owned(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".to_owned(),
                        r#type: "function".to_owned(),
                        function: ToolCallFunction {
                            name: "search".to_owned(),
                            arguments: r#"{"query":"rust"}"#.to_owned(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: "tool_calls".to_owned(),
                logprobs: None,
            }],
            usage: make_usage(10, 5),
        };
        let json_val: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(json_val["choices"][0]["finish_reason"], "tool_calls");
        assert!(json_val["choices"][0]["message"].get("content").is_none());
        assert!(json_val["choices"][0]["message"]["tool_calls"].is_array());
    }

    #[test]
    fn test_streaming_chunk_with_finish_reason() {
        let chunk = make_empty_delta_chunk("chatcmpl-fin", Some("stop".to_owned()));
        let json_val: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json_val["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_message_content_string_deserialization() {
        let json = r#"{"role": "user", "content": "hello"}"#;
        let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg.content, Some(MessageContent::Text(ref s)) if s == "hello"));
    }

    #[test]
    fn test_message_content_parts_deserialization() {
        let json = r#"{"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}}
        ]}"#;
        let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
        match &msg.content {
            Some(MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                assert!(
                    matches!(&parts[0], ContentPart::Text { text } if text == "What is in this image?")
                );
                assert!(
                    matches!(&parts[1], ContentPart::ImageUrl { image_url } if image_url.url.starts_with("data:"))
                );
            }
            other => panic!("expected Parts, got {other:?}"),
        }
    }

    #[test]
    fn test_message_content_text_method() {
        let text_content = MessageContent::Text("hello".to_owned());
        assert_eq!(text_content.text(), "hello");

        let parts_content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "What is ".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,abc".to_owned(),
                    detail: None,
                },
            },
            ContentPart::Text {
                text: "this?".to_owned(),
            },
        ]);
        assert_eq!(parts_content.text(), "What is this?");
    }

    #[test]
    fn test_message_content_image_urls() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,abc".to_owned(),
                    detail: None,
                },
            },
        ]);
        let urls = content.image_urls();
        assert_eq!(urls.len(), 1);
        assert!(urls[0].starts_with("data:"));
    }

    #[test]
    fn test_message_content_has_images() {
        let text = MessageContent::Text("no images".to_owned());
        assert!(!text.has_images());

        let text_parts = MessageContent::Parts(vec![ContentPart::Text {
            text: "no images".to_owned(),
        }]);
        assert!(!text_parts.has_images());

        let with_image = MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "data:image/png;base64,abc".to_owned(),
                detail: None,
            },
        }]);
        assert!(with_image.has_images());
    }

    #[test]
    fn test_message_content_text_serializes_as_string() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_owned(),
            content: Some(MessageContent::Text("hello".to_owned())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""content":"hello""#));
    }
}
