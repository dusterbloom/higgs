#![allow(
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::manual_let_else
)]

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use higgs_models::{
    AnyCache, AnyModel, LogprobArrays, SamplingParams, apply_penalties,
    dflash::{DFlashDrafter, accept_prefix, crop_drafter_cache},
    sample,
    turboquant::KvCacheConfig,
};
use mlx_rs::{
    Array, Dtype, Stream,
    ops::indexing::{IndexOp, NewAxis},
    transforms::{async_eval, eval},
    with_new_default_stream,
};
use tokenizers::Tokenizer;

use crate::{
    cache::PagedKvCache,
    chat_template::{ChatMessage, ChatTemplateRenderer},
    engine::{GenerationOutput, StreamingOutput},
    error::EngineError,
    mlx_tuning::MlxRuntimeTuning,
    model_loader,
    paged_prefix_cache::{DEFAULT_BLOCK_SIZE, PagedPrefixCache},
    scheduler::RoundRobinScheduler,
};

/// Default maximum number of cached prefixes.
const DEFAULT_PREFIX_CACHE_SIZE: usize = 8;
const DEFAULT_PAGED_KV_BLOCK_SIZE: usize = 64;

/// Acquire a `Mutex` lock, recovering from poison by reusing the inner data.
/// Used in this crate to keep session-management methods infallible while
/// still satisfying `clippy::unwrap_used`.
fn lock_or_recover<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn parse_enabled_flag(raw: Option<&str>) -> Option<bool> {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "true" | "on" | "yes") => Some(true),
        Some("0" | "false" | "off" | "no") => Some(false),
        _ => None,
    }
}

fn experimental_paged_kv_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_EXPERIMENTAL_PAGED_KV").ok().as_deref())
        .unwrap_or(false)
}

fn prompt_lookup_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_PROMPT_LOOKUP").ok().as_deref()).unwrap_or(false)
}

fn unchecked_prompt_lookup_enabled() -> bool {
    parse_enabled_flag(
        std::env::var("HIGGS_PROMPT_LOOKUP_UNCHECKED")
            .ok()
            .as_deref(),
    )
    .unwrap_or(false)
}

fn mtp_adaptive_draft_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_MTP_ADAPTIVE_DRAFT").ok().as_deref()).unwrap_or(false)
}

fn mtp_prompt_lookup_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_MTP_PROMPT_LOOKUP").ok().as_deref()).unwrap_or(false)
}

fn adaptive_draft_depth_for_cap(configured_max: usize) -> crate::mtp::AdaptiveDraftDepth {
    crate::mtp::AdaptiveDraftDepth::new(configured_max, configured_max)
}

fn mtp_prefill_priming_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_MTP_PRIME_PREFILL").ok().as_deref()).unwrap_or(true)
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn prompt_lookup_config() -> crate::mtp::PromptLookupConfig {
    let defaults = crate::mtp::PromptLookupConfig::default();
    crate::mtp::PromptLookupConfig {
        max_drafts: parse_env_usize("HIGGS_PROMPT_LOOKUP_DRAFT_N_MAX", defaults.max_drafts),
        max_ngram: parse_env_usize("HIGGS_PROMPT_LOOKUP_NGRAM_MAX", defaults.max_ngram),
        max_window: parse_env_usize("HIGGS_PROMPT_LOOKUP_WINDOW", defaults.max_window),
    }
}

fn estimate_paged_kv_blocks(
    target_bytes: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> usize {
    let bytes_per_token = num_kv_heads
        .saturating_mul(head_dim)
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<half::f16>())
        .max(1);
    let bytes_per_block = bytes_per_token.saturating_mul(block_size).max(1);
    (target_bytes / bytes_per_block).clamp(256, 4096)
}

#[allow(unsafe_code)]
pub(crate) fn should_clear_mlx_cache_after_prefill() -> bool {
    parse_enabled_flag(
        std::env::var("HIGGS_CLEAR_CACHE_AFTER_PREFILL")
            .ok()
            .as_deref(),
    )
    .unwrap_or(false)
}

#[allow(unsafe_code)]
pub(crate) fn maybe_clear_mlx_cache(enabled: bool, reason: &str) {
    if !enabled {
        return;
    }

    let rc = unsafe { mlx_sys::mlx_clear_cache() };
    if rc == 0 {
        tracing::debug!(reason, "Cleared MLX allocator cache");
    } else {
        tracing::warn!(reason, rc, "Failed to clear MLX allocator cache");
    }
}

/// Configure MLX's large-model working set on Apple Silicon.
///
/// By default we mirror upstream `mlx-lm` and raise MLX's wired limit to the
/// device's `max_recommended_working_set_size`. Set
/// `HIGGS_WIRED_LIMIT_MODE=legacy` to restore the older conservative
/// `memory_limit/cache_limit` behavior, or `HIGGS_NO_MEM_LIMIT=1` to skip both.
#[allow(unsafe_code)]
pub(crate) fn set_wired_limit_to_max(enabled: bool) {
    if !enabled {
        tracing::info!("MLX wired-limit escalation disabled by config");
        return;
    }
    unsafe {
        let mut info = mlx_sys::mlx_device_info_new();
        let mut dev = mlx_sys::mlx_device_new();
        mlx_sys::mlx_get_default_device(&raw mut dev);
        if mlx_sys::mlx_device_info_get(&raw mut info, dev) == 0 {
            let mut max_rec: usize = 0;
            let key = c"max_recommended_working_set_size";
            if mlx_sys::mlx_device_info_get_size(&raw mut max_rec, info, key.as_ptr()) == 0
                && max_rec > 0
            {
                let wired_mode = std::env::var("HIGGS_WIRED_LIMIT_MODE").ok();
                let use_legacy_limits =
                    matches!(wired_mode.as_deref(), Some("legacy" | "safe" | "caps"));
                let mut prev_mem: usize = 0;
                let mut prev_cache: usize = 0;
                let mut prev_wired: usize = 0;

                let limits_enabled = std::env::var("HIGGS_NO_MEM_LIMIT").is_err();

                if limits_enabled {
                    if use_legacy_limits {
                        let mem_limit = max_rec * 3 / 4;
                        let cache_limit = max_rec / 2;
                        mlx_sys::mlx_set_memory_limit(&raw mut prev_mem, mem_limit);
                        mlx_sys::mlx_set_cache_limit(&raw mut prev_cache, cache_limit);
                        tracing::info!(
                            mode = "legacy",
                            max_recommended_mb = max_rec / (1024 * 1024),
                            memory_limit_mb = mem_limit / (1024 * 1024),
                            cache_limit_mb = cache_limit / (1024 * 1024),
                            prev_mem_mb = prev_mem / (1024 * 1024),
                            prev_cache_mb = prev_cache / (1024 * 1024),
                            "Configured MLX legacy memory/cache caps",
                        );
                    } else {
                        mlx_sys::mlx_set_wired_limit(&raw mut prev_wired, max_rec);
                        tracing::info!(
                            mode = "mlx_wired_limit",
                            max_recommended_mb = max_rec / (1024 * 1024),
                            wired_limit_mb = max_rec / (1024 * 1024),
                            prev_wired_mb = prev_wired / (1024 * 1024),
                            "Configured MLX wired limit",
                        );
                    }
                } else {
                    tracing::info!(
                        mode = if use_legacy_limits {
                            "legacy"
                        } else {
                            "mlx_wired_limit"
                        },
                        max_recommended_mb = max_rec / (1024 * 1024),
                        "Skipped MLX memory-limit configuration",
                    );
                }
            }
        }
        mlx_sys::mlx_device_info_free(info);
        mlx_sys::mlx_device_free(dev);
    }
}

/// Session state for batched generation.
#[derive(Debug, Clone)]
pub struct Session {
    /// Stable session identifier used by the scheduler and paged KV cache.
    pub id: u64,
    /// Prompt plus generated token IDs accumulated so far for this session.
    pub tokens: Vec<u32>,
    /// Whether generation for this session has already terminated.
    pub finished: bool,
    /// Maximum number of completion tokens allowed for this session.
    pub max_tokens: usize,
}

/// Simple single-request inference engine with paged KV caching.
///
/// Uses paged KV cache for efficient memory management during single-request
/// generation. Session-based batched stepping APIs are not wired to real decode
/// yet and return explicit errors instead of placeholder output.
/// `DFlash` block-diffusion speculative decoding state, held alongside the
/// target model when a drafter is loaded.
struct DFlashState {
    drafter: Mutex<DFlashDrafter>,
    tap_layers: Vec<usize>,
    block_size: i32,
    mask_token_id: i32,
}

pub struct SimpleEngine {
    model: Mutex<AnyModel>,
    prefix_cache: Mutex<PagedPrefixCache>,
    /// Paged KV cache for session-based generation
    paged_cache: Option<Mutex<PagedKvCache>>,
    /// Session scheduler for continuous batching
    scheduler: Mutex<RoundRobinScheduler>,
    /// Active sessions
    sessions: Mutex<std::collections::HashMap<u64, Session>>,
    tokenizer: Tokenizer,
    template: Option<ChatTemplateRenderer>,
    model_name: String,
    eos_token_ids: Vec<u32>,
    /// Control tokens stripped from decoded output (EOS + `<|…|>` chat
    /// delimiters + classic sentinels), while content-bearing special tokens
    /// (tool-call markup, `<think>`) are preserved. See [`Self::decode_tokens`].
    /// Wrapped in `Arc` so each request's streaming [`IncrementalDetok`] shares
    /// the same set and strips identically to `decode_tokens`.
    decode_skip_ids: std::sync::Arc<std::collections::HashSet<u32>>,
    /// Whether to enable thinking mode (Qwen3.5 `<think>` tags).
    enable_thinking: bool,
    /// Token ID for `</think>`, resolved from the tokenizer at load time.
    /// `None` if the tokenizer doesn't know this token (thinking will be disabled).
    think_close_token: Option<u32>,
    /// Number of trailing tokens added by `add_generation_prompt=true`.
    /// Stripped from the prefix cache key so that multi-turn conversations
    /// share the same token prefix (the generation prompt changes between turns).
    gen_prompt_suffix_len: usize,
    kv_cache_config: KvCacheConfig,
    tuning: MlxRuntimeTuning,
    /// Optional `DFlash` block-diffusion speculative decoding state, enabled
    /// when `HIGGS_DFLASH_PATH` points at a drafter checkpoint.
    dflash: Option<DFlashState>,
}

/// Intermediate state after prefix cache lookup and model locking.
struct PreparedGeneration<'a> {
    model: MutexGuard<'a, AnyModel>,
    cache: AnyCache,
    actual_prompt_tokens: Vec<u32>,
    prompt_array: Array,
    prompt_len: u32,
    pixel_values: Option<Array>,
}

impl SimpleEngine {
    /// Load a model and tokenizer from a directory.
    pub fn load<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        tuning: MlxRuntimeTuning,
        raise_wired_limit: bool,
    ) -> Result<Self, EngineError> {
        Self::load_with_dflash(dir, kv_cache_config, tuning, raise_wired_limit, None)
    }

    /// Load a model with an optional `DFlash` speculative-decoding drafter.
    ///
    /// The drafter path is taken from `dflash_path` when `Some`, otherwise from
    /// the `HIGGS_DFLASH_PATH` env var. When a drafter is present, `generate`
    /// dispatches to the block-diffusion draft-verify loop.
    #[allow(clippy::too_many_lines)]
    pub fn load_with_dflash<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        tuning: MlxRuntimeTuning,
        raise_wired_limit: bool,
        dflash_path: Option<&Path>,
    ) -> Result<Self, EngineError> {
        let model_dir = dir.as_ref();
        let model_name = derive_model_name(model_dir);

        tracing::info!(model_dir = %model_dir.display(), "Loading model");

        let model = model_loader::load_model(model_dir)?;
        let _ = model
            .make_cache_with_config(kv_cache_config)
            .map_err(EngineError::Mlx)?;
        let tokenizer = model_loader::load_tokenizer(model_dir)?;
        let template = ChatTemplateRenderer::try_from_model_dir(model_dir)?;
        if template.is_none() {
            tracing::warn!("No chat template found; /v1/chat/completions will be unavailable");
        }

        let eos_token_ids = extract_eos_tokens(model_dir);

        // Control tokens that must never surface in decoded text (EOS + the
        // `<|…|>` chat-control delimiters + classic sentinels). Shared via `Arc`
        // with each request's streaming `IncrementalDetok` so streaming strips
        // identically to `decode_tokens`.
        let decode_skip_ids =
            std::sync::Arc::new(content_preserving_skip_ids(&tokenizer, &eos_token_ids));

        // Auto-detect thinking mode: Qwen3.5 models support <think> tags.
        // Override with HIGGS_ENABLE_THINKING=0/1, off/true, yes/no etc.
        let mut enable_thinking = std::env::var("HIGGS_ENABLE_THINKING")
            .ok()
            .as_deref()
            .map_or_else(
                || detect_thinking_support(model_dir),
                |v| {
                    parse_enabled_flag(Some(v))
                        .unwrap_or_else(|| detect_thinking_support(model_dir))
                },
            );

        // Resolve </think> token ID from the tokenizer. If the tokenizer
        // doesn't know this token, disable thinking to avoid injecting
        // out-of-vocab IDs into the embedding lookup.
        let think_close_token = tokenizer.encode("</think>", false).ok().and_then(|enc| {
            let ids = enc.get_ids();
            // Must encode to exactly one token to be usable as a forced stop.
            if let [single] = ids {
                Some(*single)
            } else {
                None
            }
        });
        if enable_thinking && think_close_token.is_none() {
            tracing::warn!("Tokenizer has no single </think> token; disabling thinking mode");
            enable_thinking = false;
        }
        if enable_thinking {
            tracing::info!(think_close_token, "Thinking mode enabled");
        }

        set_wired_limit_to_max(raise_wired_limit);

        // Compute the generation prompt suffix length: tokens added by
        // `add_generation_prompt=true` (e.g., `<|im_start|>assistant\n<think>\n`).
        // We strip these from the prefix cache key so multi-turn conversations
        // share their common history prefix.
        let gen_prompt_suffix_len = template
            .as_ref()
            .and_then(|tmpl| {
                let test_msg = vec![crate::chat_template::ChatMessage {
                    role: "user".to_owned(),
                    content: "x".to_owned(),
                    tool_calls: None,
                }];
                let with_gen = tmpl
                    .apply_with_thinking(&test_msg, None, true, enable_thinking)
                    .ok()?;
                let without_gen = tmpl
                    .apply_with_thinking(&test_msg, None, false, enable_thinking)
                    .ok()?;
                let toks_with = tokenizer.encode(with_gen.as_str(), false).ok()?;
                let toks_without = tokenizer.encode(without_gen.as_str(), false).ok()?;
                let suffix = toks_with
                    .get_ids()
                    .len()
                    .saturating_sub(toks_without.get_ids().len());
                tracing::info!(
                    gen_prompt_suffix_len = suffix,
                    "Computed generation prompt suffix length for prefix cache"
                );
                Some(suffix)
            })
            .unwrap_or(0);

        let experimental_paged_kv = experimental_paged_kv_enabled();
        if experimental_paged_kv {
            tracing::info!(
                model_name = %model_name,
                eos_tokens = ?eos_token_ids,
                requested_mlx_profile = tuning.requested_profile().as_str(),
                effective_mlx_profile = tuning.resolved_profile().as_str(),
                chunked_prefill_threshold = tuning.chunked_prefill_threshold(),
                chunked_prefill_chunk_size = tuning.chunked_prefill_chunk_size(),
                clear_cache_after_prefill = tuning.clear_cache_after_prefill(),
                mtp_enabled = tuning.enable_mtp(),
                mtp_draft_n_max = tuning.mtp_draft_n_max(),
                paged_kv_target_mb = tuning.paged_kv_target_bytes() / (1024 * 1024),
                "Engine ready"
            );
        } else {
            tracing::info!(
                model_name = %model_name,
                eos_tokens = ?eos_token_ids,
                requested_mlx_profile = tuning.requested_profile().as_str(),
                effective_mlx_profile = tuning.resolved_profile().as_str(),
                chunked_prefill_threshold = tuning.chunked_prefill_threshold(),
                chunked_prefill_chunk_size = tuning.chunked_prefill_chunk_size(),
                clear_cache_after_prefill = tuning.clear_cache_after_prefill(),
                mtp_enabled = tuning.enable_mtp(),
                mtp_draft_n_max = tuning.mtp_draft_n_max(),
                "Engine ready"
            );
            tracing::debug!("Experimental paged KV disabled; session cache allocation skipped");
        }

        let (raw_num_kv_heads, raw_head_dim) = model
            .kv_cache_geometry()
            .map_err(|e| EngineError::Generation(format!("paged cache geometry: {e}")))?;
        let num_kv_heads = usize::try_from(raw_num_kv_heads)
            .map_err(|_| EngineError::Generation("paged cache num_kv_heads overflow".to_owned()))?;
        let head_dim = usize::try_from(raw_head_dim)
            .map_err(|_| EngineError::Generation("paged cache head_dim overflow".to_owned()))?;
        let paged_cache = experimental_paged_kv
            .then(|| {
                let num_blocks = estimate_paged_kv_blocks(
                    tuning.paged_kv_target_bytes(),
                    num_kv_heads,
                    head_dim,
                    DEFAULT_PAGED_KV_BLOCK_SIZE,
                );
                PagedKvCache::new(
                    num_blocks,
                    DEFAULT_PAGED_KV_BLOCK_SIZE,
                    num_kv_heads,
                    head_dim,
                )
                .map_err(|e| EngineError::Generation(format!("paged cache init: {e}")))
            })
            .transpose()?;

        // DFlash speculative decoding: load the drafter from the explicit path
        // or HIGGS_DFLASH_PATH. generate_inner then dispatches to the
        // draft-verify loop for unconstrained text generation.
        let dflash = dflash_path
            .map(Path::to_path_buf)
            .or_else(|| {
                std::env::var("HIGGS_DFLASH_PATH")
                    .ok()
                    .map(std::path::PathBuf::from)
            })
            .map(|dp| -> Result<DFlashState, EngineError> {
                tracing::info!(drafter = %dp.display(), "Loading DFlash drafter");
                let drafter = model_loader::load_dflash_drafter(&dp)?;
                let tap_layers = drafter.config.target_layer_ids().to_vec();
                // Decode block size: HIGGS_DFLASH_BLOCK_SIZE overrides the
                // trained block_size, clamped to [1, trained]. Smaller blocks
                // amortize the per-round verify + lm_head cost better when the
                // accept length plateaus below the trained block.
                let trained_block = drafter.config.block_size;
                let block_size = std::env::var("HIGGS_DFLASH_BLOCK_SIZE")
                    .ok()
                    .and_then(|v| v.parse::<i32>().ok())
                    .filter(|&v| v >= 1)
                    .map_or(trained_block, |v| v.min(trained_block));
                let mask_token_id = drafter.config.mask_token_id();
                tracing::info!(
                    tap_layers = ?tap_layers,
                    block_size,
                    mask_token_id,
                    "DFlash drafter loaded — speculative decoding enabled"
                );
                Ok(DFlashState {
                    drafter: Mutex::new(drafter),
                    tap_layers,
                    block_size,
                    mask_token_id,
                })
            })
            .transpose()?;

        Ok(Self {
            model: Mutex::new(model),
            prefix_cache: Mutex::new(PagedPrefixCache::new(
                DEFAULT_PREFIX_CACHE_SIZE,
                DEFAULT_BLOCK_SIZE,
            )),
            paged_cache: paged_cache.map(Mutex::new),
            scheduler: Mutex::new(RoundRobinScheduler::new()),
            sessions: Mutex::new(std::collections::HashMap::new()),
            tokenizer,
            template,
            model_name,
            eos_token_ids,
            decode_skip_ids,
            enable_thinking,
            think_close_token,
            gen_prompt_suffix_len,
            kv_cache_config,
            tuning,
            dflash,
        })
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get a reference to the tokenizer.
    pub const fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the model's EOS token IDs.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Whether the engine has thinking mode enabled.
    pub const fn enable_thinking(&self) -> bool {
        self.enable_thinking
    }

    /// Apply chat template and tokenize messages with explicit thinking control.
    pub fn prepare_chat_prompt_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> Result<Vec<u32>, EngineError> {
        let renderer = self.template.as_ref().ok_or_else(|| {
            EngineError::Template(
                "This model has no chat template; use /v1/completions instead".to_owned(),
            )
        })?;
        let prompt = renderer.apply_with_thinking(messages, tools, true, enable_thinking)?;
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| EngineError::Tokenization(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Apply chat template and tokenize messages.
    pub fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>, EngineError> {
        self.prepare_chat_prompt_with_thinking(messages, tools, self.enable_thinking)
    }

    /// Whether the loaded model is a vision-language model.
    pub fn is_vlm(&self) -> bool {
        let model = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        model.is_vlm()
    }

    /// The expected image size for the VLM's vision encoder, or `None`.
    pub fn vlm_image_size(&self) -> Option<i32> {
        let model = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        model.image_size()
    }

    /// Replace image placeholder tokens with `IMAGE_TOKEN_INDEX` in the token
    /// sequence. The `<image>` token ID is looked up from the tokenizer.
    #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
    pub fn replace_image_tokens(&self, tokens: &mut [u32]) {
        let Some(image_token_id) = self.tokenizer.token_to_id("<image>") else {
            return;
        };
        let image_token_u32 = higgs_models::llava_qwen2::IMAGE_TOKEN_INDEX as u32;
        for token in tokens.iter_mut() {
            if *token == image_token_id {
                *token = image_token_u32;
            }
        }
    }

    /// Convert prompt length to u32, returning a descriptive error on overflow.
    fn prompt_len(prompt_tokens: &[u32]) -> Result<u32, EngineError> {
        prompt_tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))
    }

    /// Look up the prefix cache, lock the model, and resolve the actual tokens
    /// to feed into the forward pass.
    fn prepare_generation(
        &self,
        prompt_tokens: &[u32],
        pixel_values: Option<Array>,
    ) -> Result<PreparedGeneration<'_>, EngineError> {
        let prompt_len = Self::prompt_len(prompt_tokens)?;
        let has_images = pixel_values.is_some();

        // Skip prefix caching for multimodal requests: different images
        // produce different KV states even with identical token sequences.
        let prefix_match = if has_images {
            None
        } else {
            let mut pc = self
                .prefix_cache
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            pc.find_longest_prefix(prompt_tokens)
        };

        let model = self
            .model
            .lock()
            .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;

        let (actual_prompt_tokens, cache) = if let Some(matched) = prefix_match {
            tracing::debug!(
                prefix_len = matched.prefix_len,
                total_len = prompt_tokens.len(),
                suffix_len = prompt_tokens.len() - matched.prefix_len,
                "Prefix cache hit — reusing cached prefix"
            );
            let suffix = prompt_tokens.get(matched.prefix_len..).unwrap_or_default();
            if suffix.is_empty() {
                tracing::debug!(
                    prompt_len = prompt_tokens.len(),
                    "Full prefix hit requires cached logits; falling back to fresh prefill"
                );
                (
                    prompt_tokens.to_vec(),
                    model
                        .make_cache_with_config(self.kv_cache_config)
                        .map_err(EngineError::Mlx)?,
                )
            } else {
                (suffix.to_vec(), matched.cache)
            }
        } else {
            (
                prompt_tokens.to_vec(),
                model
                    .make_cache_with_config(self.kv_cache_config)
                    .map_err(EngineError::Mlx)?,
            )
        };

        let prompt_array = Array::from(actual_prompt_tokens.as_slice()).index(NewAxis);

        Ok(PreparedGeneration {
            model,
            cache,
            actual_prompt_tokens,
            prompt_array,
            prompt_len,
            pixel_values,
        })
    }

    /// Run the prefill forward pass and sample the first token. Stores the
    /// post-prefill KV state back into the prefix cache (skipped for multimodal).
    /// Optionally computes logprobs for the first token.
    fn run_prefill(
        &self,
        prompt_tokens: &[u32],
        prepared: &mut PreparedGeneration<'_>,
        params: &SamplingParams,
        logprob_top_n: Option<u32>,
        constraint: Option<&crate::constrained::ConstrainedGenerator>,
        capture_hidden: bool,
    ) -> Result<(Array, Option<LogprobArrays>, Option<Array>), EngineError> {
        let mut prefill_hidden = None;
        let logits = if let Some(ref pixel_values) = prepared.pixel_values {
            // Multimodal path: full forward (VLMs need all tokens for vision)
            prepared
                .model
                .forward_multimodal(&prepared.prompt_array, pixel_values, &mut prepared.cache)
                .map_err(EngineError::Mlx)?
        } else {
            // Text-only prefill: use chunked prefill for long sequences to bound
            // peak memory, otherwise single-pass with last-token-only LM head.
            let seq_len = prepared.prompt_array.shape().get(1).copied().unwrap_or(0);
            let chunked_threshold = self.tuning.chunked_prefill_threshold();
            let chunked_size = self.tuning.chunked_prefill_chunk_size();
            if capture_hidden && seq_len <= chunked_threshold {
                let (hidden, logits) = prepared
                    .model
                    .forward_with_hidden(&prepared.prompt_array, None, &mut prepared.cache)
                    .map_err(EngineError::Mlx)?;
                prefill_hidden = Some(hidden);
                logits
            } else if seq_len > chunked_threshold {
                prepared
                    .model
                    .forward_chunked(&prepared.prompt_array, &mut prepared.cache, chunked_size)
                    .map_err(EngineError::Mlx)?
            } else {
                prepared
                    .model
                    .forward_last_token(&prepared.prompt_array, None, &mut prepared.cache)
                    .map_err(EngineError::Mlx)?
            }
        };
        let last_logits = logits.index((.., -1, ..));

        let constrained_logits = if let Some(cg) = constraint {
            cg.apply_mask(&last_logits).map_err(EngineError::Mlx)?
        } else {
            last_logits
        };

        let current_token = sample(&constrained_logits, params).map_err(EngineError::Mlx)?;

        let logprob_data = if let Some(top_n) = logprob_top_n {
            let scaled = if params.temperature <= f32::EPSILON {
                constrained_logits
            } else {
                constrained_logits
                    .multiply(Array::from_f32(1.0 / params.temperature))
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
            if let Some(ref lp) = logprob_data {
                eval_targets.extend(lp.eval_targets());
            }
            if let Some(ref hidden) = prefill_hidden {
                eval_targets.push(hidden);
            }
            eval(eval_targets).map_err(EngineError::Mlx)?;
        }

        // Skip prefix cache for multimodal (image-specific KV states)
        if prepared.pixel_values.is_none() {
            let mut pc = self
                .prefix_cache
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            // Strip generation prompt suffix so multi-turn conversations
            // share their common history prefix. The suffix tokens
            // (`<|im_start|>assistant\n<think>\n`) change between turns.
            let cache_key = prompt_tokens
                .get(
                    ..prompt_tokens
                        .len()
                        .saturating_sub(self.gen_prompt_suffix_len),
                )
                .unwrap_or(prompt_tokens);
            pc.store(cache_key, &prepared.cache);
        }
        maybe_clear_mlx_cache(
            self.tuning.clear_cache_after_prefill(),
            "simple_post_prefill",
        );

        Ok((current_token, logprob_data, prefill_hidden))
    }

    /// Decode a single step: forward pass on the current token, apply penalties
    /// and optional constraint mask, then sample. Returns `(next_token, Option<LogprobArrays>)`.
    fn decode_step(
        current_token: &Array,
        model: &mut AnyModel,
        cache: &mut AnyCache,
        params: &SamplingParams,
        generated_tokens: &[u32],
        logprob_top_n: Option<u32>,
        constraint: Option<&crate::constrained::ConstrainedGenerator>,
    ) -> Result<(Array, Option<LogprobArrays>), EngineError> {
        let decode_input = current_token.index((.., NewAxis));
        let logits = model
            .forward(&decode_input, None, cache)
            .map_err(EngineError::Mlx)?;
        let sliced = logits.index((.., -1, ..));

        let penalized =
            apply_penalties(&sliced, generated_tokens, params).map_err(EngineError::Mlx)?;

        // Apply constraint mask if structured output is requested
        let constrained = if let Some(cg) = constraint {
            cg.apply_mask(&penalized).map_err(EngineError::Mlx)?
        } else {
            penalized
        };

        let next_token = sample(&constrained, params).map_err(EngineError::Mlx)?;

        let logprob_data = if let Some(top_n) = logprob_top_n {
            // Compute logprobs from the same distribution we sampled from.
            // Temperature is already accounted for inside `sample`, so we
            // replicate the scaling here for the logprob computation.
            let scaled = if params.temperature <= f32::EPSILON {
                constrained
            } else {
                constrained
                    .multiply(mlx_rs::array!(1.0 / params.temperature))
                    .map_err(EngineError::Mlx)?
            };
            Some(
                LogprobArrays::compute(&scaled, &next_token, Some(top_n))
                    .map_err(EngineError::Mlx)?,
            )
        } else {
            None
        };

        Ok((next_token, logprob_data))
    }

    /// Decode the token buffer and return the text, mapping tokenizer errors.
    ///
    /// Decodes WITHOUT skipping special tokens so content-bearing markup
    /// survives — notably models (e.g. `MiniCPM5`) that encode their tool-call
    /// structure (`<function>`, `<param>`, …) as special tokens, which the
    /// tool parser needs to see. Control tokens (EOS) are filtered out first so
    /// they never leak into visible text. Plain text contains no special
    /// tokens and decodes identically either way, so normal responses are
    /// unaffected.
    fn decode_tokens(&self, tokens: &[u32]) -> Result<String, EngineError> {
        let decode = |ids: &[u32]| {
            self.tokenizer
                .decode(ids, false)
                .map_err(|e| EngineError::Tokenization(e.to_string()))
        };
        // Fast path: no control token present, decode the slice as-is.
        if !tokens.iter().any(|id| self.decode_skip_ids.contains(id)) {
            return decode(tokens);
        }
        let filtered: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|id| !self.decode_skip_ids.contains(id))
            .collect();
        decode(&filtered)
    }

    /// The model's hidden dimension (embedding output size).
    pub fn hidden_size(&self) -> i32 {
        let model = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        model.hidden_size()
    }

    /// Compute embeddings for a sequence of token IDs.
    ///
    /// Runs a single forward pass through the model to get hidden states,
    /// mean-pools across the sequence dimension, and L2-normalizes.
    #[allow(clippy::significant_drop_tightening)]
    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::Generation("Input is empty".to_owned()));
        }

        with_new_default_stream(Stream::new(), || {
            let input = Array::from(token_ids).index(NewAxis);
            let mut model = self
                .model
                .lock()
                .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;
            let mut cache = model
                .make_cache_with_config(self.kv_cache_config)
                .map_err(EngineError::Mlx)?;

            // Forward pass to get hidden states [1, seq_len, hidden_size]
            let hidden = model
                .forward_hidden(&input, None, &mut cache)
                .map_err(EngineError::Mlx)?;

            // Mean-pool across seq_len (axis 1), producing [1, hidden_size]
            let pooled = hidden.mean_axes(&[1], false).map_err(EngineError::Mlx)?;

            // Cast to f32 before extracting values (model may use bfloat16)
            let pooled_f32 = pooled.as_dtype(Dtype::Float32).map_err(EngineError::Mlx)?;
            eval([&pooled_f32]).map_err(EngineError::Mlx)?;

            // L2-normalize on CPU
            let values = pooled_f32.as_slice::<f32>().to_vec();
            let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                Ok(values.iter().map(|x| x / norm).collect())
            } else {
                Ok(values)
            }
        })
    }

    /// Convert a token count to u32, with an overflow error.
    fn completion_len(tokens: &[u32]) -> Result<u32, EngineError> {
        tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Too many tokens generated".to_owned()))
    }

    fn hidden_row_from_sequence(hidden: &Array, row_index: usize) -> Result<Array, EngineError> {
        let row_i32 = i32::try_from(row_index)
            .map_err(|_| EngineError::Generation("hidden row index too large".to_owned()))?;
        Ok(hidden.index((.., row_i32..row_i32 + 1, ..)))
    }

    // =========================================================================
    // Session Management (Batched Generation)
    // =========================================================================

    /// Create a new session for batched generation.
    ///
    /// Returns the session ID.
    pub fn create_session(
        &self,
        _prompt_tokens: &[u32],
        _max_tokens: usize,
    ) -> Result<u64, EngineError> {
        Err(EngineError::Generation(
            "session generation is not implemented for SimpleEngine yet".to_owned(),
        ))
    }

    /// Get session state.
    pub fn get_session(&self, session_id: u64) -> Option<Session> {
        let sessions = lock_or_recover(&self.sessions);
        sessions.get(&session_id).cloned()
    }

    /// Remove a session and free its resources.
    pub fn remove_session(&self, session_id: u64) -> Result<(), EngineError> {
        let mut scheduler = lock_or_recover(&self.scheduler);
        let mut sessions = lock_or_recover(&self.sessions);
        sessions.remove(&session_id);
        scheduler.remove(session_id);
        if let Some(paged_cache_mutex) = &self.paged_cache {
            let mut paged_cache = lock_or_recover(paged_cache_mutex);
            paged_cache
                .remove_session(session_id)
                .map_err(|e| EngineError::Generation(format!("Failed to remove session: {e}")))?;
        }

        Ok(())
    }

    /// Check if session is finished.
    pub fn is_session_finished(&self, session_id: u64) -> bool {
        let sessions = lock_or_recover(&self.sessions);
        sessions.get(&session_id).is_none_or(|s| s.finished)
    }

    /// Step one token for all active sessions (batched generation).
    ///
    /// Returns outputs for each session that produced a token.
    ///
    /// Note: Current implementation processes sessions sequentially.
    /// True batched generation (parallel decode across sessions) is TODO.
    pub fn step(
        &self,
        _params: &SamplingParams,
    ) -> Result<Vec<(u64, GenerationOutput)>, EngineError> {
        Err(EngineError::Generation(
            "session stepping is not implemented for SimpleEngine yet".to_owned(),
        ))
    }

    /// Generate a complete response for a session (batched mode).
    ///
    /// This is a helper for batched generation that generates all tokens
    /// for a session in one call, using the paged cache.
    pub fn generate_session(
        &self,
        session_id: u64,
        _params: &SamplingParams,
        _stop_sequences: &[String],
        _logprobs: bool,
        _top_logprobs: Option<u32>,
    ) -> Result<GenerationOutput, EngineError> {
        Err(EngineError::Generation(format!(
            "session generation is not implemented for SimpleEngine yet (session {session_id})"
        )))
    }

    /// Generate a complete response from a token prompt.
    ///
    /// For multimodal requests, pass `pixel_values` with preprocessed image
    /// data and ensure `prompt_tokens` contains `IMAGE_TOKEN_INDEX` at image
    /// positions.
    #[allow(clippy::significant_drop_tightening, clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<GenerationOutput, EngineError> {
        self.generate_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            self.enable_thinking,
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
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<GenerationOutput, EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }
        if max_tokens == 0 {
            return Ok(GenerationOutput {
                text: String::new(),
                finish_reason: "length".to_owned(),
                prompt_tokens: Self::prompt_len(prompt_tokens)?,
                completion_tokens: 0,
                token_logprobs: None,
            });
        }

        // Set a task-local default stream so every MLX operation reuses it
        // instead of creating a new Stream (5 FFI calls) per operation.
        with_new_default_stream(Stream::new(), || {
            self.generate_inner(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                enable_thinking,
                constraint,
                pixel_values,
            )
        })
    }

    #[allow(
        clippy::significant_drop_tightening,
        clippy::too_many_lines,
        clippy::too_many_arguments
    )]
    fn generate_inner(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        enable_thinking: bool,
        mut constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<GenerationOutput, EngineError> {
        // DFlash speculative decoding: use the draft-verify loop when a drafter
        // is loaded, no constraints active, and no multimodal input.
        if self.dflash.is_some() && constraint.is_none() && pixel_values.is_none() {
            return self.generate_dflash_inner(prompt_tokens, max_tokens, params, stop_sequences);
        }

        let logprob_top_n = logprobs.then(|| top_logprobs.unwrap_or(0));

        let mut prepared = self.prepare_generation(prompt_tokens, pixel_values)?;
        let prompt_len = prepared.prompt_len;
        #[allow(clippy::float_cmp)]
        let capture_mtp_prefill = mtp_prefill_priming_enabled()
            && self.tuning.enable_mtp()
            && prepared.model.has_mtp()
            && prepared.pixel_values.is_none()
            && constraint.is_none()
            && !logprobs
            && params.temperature == 0.0;

        let (current_token, first_logprob_data, prefill_hidden) = self.run_prefill(
            prompt_tokens,
            &mut prepared,
            params,
            logprob_top_n,
            constraint.as_ref(),
            capture_mtp_prefill,
        )?;

        // Capture T1 (already eval'd inside run_prefill).
        let first_token_id: u32 = current_token.item();
        // Advance the constraint past the first sampled token before decode.
        if let Some(ref mut cg) = constraint {
            cg.advance(first_token_id);
        }
        let mut tokens: Vec<u32> = vec![first_token_id];
        let mut all_logprobs: Option<Vec<higgs_models::TokenLogprobInfo>> = logprobs.then(Vec::new);
        if let (Some(all_lp), Some(lp_data)) = (&mut all_logprobs, &first_logprob_data) {
            all_lp.push(lp_data.materialize(first_token_id));
        }
        let has_stop_sequences = !stop_sequences.is_empty();

        // Handle T1 termination before entering the pipeline.
        if self.eos_token_ids.contains(&first_token_id) {
            return Ok(GenerationOutput {
                text: self.decode_tokens(&tokens)?,
                finish_reason: "stop".to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprobs: all_logprobs,
            });
        }
        if has_stop_sequences {
            let text = self.decode_tokens(&tokens)?;
            if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
                return Ok(GenerationOutput {
                    text: truncated,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: 1,
                    token_logprobs: all_logprobs,
                });
            }
        }
        if max_tokens <= 1 {
            return Ok(GenerationOutput {
                text: self.decode_tokens(&tokens)?,
                finish_reason: "length".to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprobs: all_logprobs,
            });
        }

        // Architecture-neutral speculative decode: prompt-lookup drafting plus
        // batched verifier logits. Explicitly opt-in while benchmark data is
        // collected because the normal path has a pipelined single-token loop.
        #[allow(clippy::float_cmp)]
        if prompt_lookup_enabled() && constraint.is_none() && !logprobs && params.temperature == 0.0
        {
            return self.prompt_lookup_generate(
                &mut prepared.model,
                &mut prepared.cache,
                first_token_id,
                max_tokens,
                prompt_len,
                &mut tokens,
                stop_sequences,
                enable_thinking,
            );
        }

        // MTP speculative decode: enabled by the resolved MLX runtime tuning.
        // Only for greedy (temperature == 0), no constraints, no logprobs.
        #[allow(clippy::float_cmp)]
        if self.tuning.enable_mtp()
            && prepared.model.has_mtp()
            && constraint.is_none()
            && !logprobs
            && params.temperature == 0.0
        {
            let actual_prompt_tokens = prepared.actual_prompt_tokens.clone();
            return self.mtp_generate(
                &mut prepared.model,
                &mut prepared.cache,
                &actual_prompt_tokens,
                prefill_hidden.as_ref(),
                first_token_id,
                max_tokens,
                prompt_len,
                &mut tokens,
                stop_sequences,
                enable_thinking,
            );
        }

        // Pipelined decode: build step N+2's graph while GPU computes step N+1.
        // When constrained generation is active, pipelining would apply the FSM mask
        // one step behind (since we need the sampled token value to advance the FSM
        // before constraining the next step). Fall back to sequential decode instead.
        let (mut next_token, mut next_logprob_data) = Self::decode_step(
            &current_token,
            &mut prepared.model,
            &mut prepared.cache,
            params,
            &tokens,
            logprob_top_n,
            constraint.as_ref(),
        )?;
        {
            let mut eval_targets: Vec<&Array> = vec![&next_token];
            if let Some(ref lp) = next_logprob_data {
                eval_targets.extend(lp.eval_targets());
            }
            if constraint.is_some() {
                eval(eval_targets).map_err(EngineError::Mlx)?;
            } else {
                async_eval(eval_targets).map_err(EngineError::Mlx)?;
            }
        }

        let mut total_forward_ns: u128 = 0;
        let mut total_eval_ns: u128 = 0;
        let mut total_item_ns: u128 = 0;
        let mut total_other_ns: u128 = 0;
        let mut step_count: u32 = 0;

        // Thinking budget: force </think> after N tokens if model hasn't closed it.
        const THINKING_BUDGET: u32 = 256;
        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        // Seed thinking state from the first token (already emitted above).
        let mut thinking_tokens: u32 = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        loop {
            let t0 = std::time::Instant::now();

            // When constrained, extract the sampled token and advance the FSM
            // before building the next step, so the mask is always applied at the
            // correct FSM state.
            let constrained_token_id: Option<u32> = constraint.is_some().then(|| {
                let id: u32 = next_token.item();
                if let Some(ref mut cg) = constraint {
                    cg.advance(id);
                }
                id
            });

            let (following, following_logprob_data) = Self::decode_step(
                &next_token,
                &mut prepared.model,
                &mut prepared.cache,
                params,
                &tokens,
                logprob_top_n,
                constraint.as_ref(),
            )?;
            let t1 = std::time::Instant::now();
            {
                let mut eval_targets: Vec<&Array> = vec![&following];
                if let Some(ref lp) = following_logprob_data {
                    eval_targets.extend(lp.eval_targets());
                }
                if constraint.is_some() {
                    eval(eval_targets).map_err(EngineError::Mlx)?;
                } else {
                    async_eval(eval_targets).map_err(EngineError::Mlx)?;
                }
            }
            let t2 = std::time::Instant::now();

            // In the unconstrained pipeline, extract the token here (after building following).
            let mut token_id: u32 = constrained_token_id.unwrap_or_else(|| next_token.item());

            // Thinking budget: force </think> after N tokens if model hasn't closed it.
            // NOTE: when the budget fires, token_id is overwritten but the KV cache
            // already reflects the originally-sampled token.  The next forward pass
            // feeds close_id as input while the cache holds a different token at this
            // position — a one-entry discontinuity.  Re-running forward to fix the
            // cache is expensive for negligible quality impact after 256+ tokens.
            if let Some(close_id) = think_close_token {
                if !seen_think_close {
                    if token_id == close_id {
                        seen_think_close = true;
                    } else {
                        thinking_tokens += 1;
                        if thinking_tokens >= THINKING_BUDGET {
                            token_id = close_id;
                            seen_think_close = true;
                            tracing::info!(
                                budget = THINKING_BUDGET,
                                "Thinking budget reached, forcing </think>"
                            );
                        }
                    }
                }
            }

            // Materialize logprobs for the token we just extracted
            if let (Some(all_lp), Some(lp_data)) = (&mut all_logprobs, &next_logprob_data) {
                all_lp.push(lp_data.materialize(token_id));
            }

            let t3 = std::time::Instant::now();

            tokens.push(token_id);
            let completion_len = Self::completion_len(&tokens)?;
            let t4 = std::time::Instant::now();

            total_forward_ns += (t1 - t0).as_nanos();
            total_eval_ns += (t2 - t1).as_nanos();
            total_item_ns += (t3 - t2).as_nanos();
            total_other_ns += (t4 - t3).as_nanos();
            step_count += 1;

            // Check if constraint is in final state
            if constraint
                .as_ref()
                .is_some_and(crate::constrained::ConstrainedGenerator::is_finished)
            {
                Self::log_decode_timing(
                    step_count,
                    total_forward_ns,
                    total_eval_ns,
                    total_item_ns,
                    total_other_ns,
                );
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: all_logprobs,
                });
            }

            if self.eos_token_ids.contains(&token_id) {
                Self::log_decode_timing(
                    step_count,
                    total_forward_ns,
                    total_eval_ns,
                    total_item_ns,
                    total_other_ns,
                );
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: all_logprobs,
                });
            }

            if has_stop_sequences {
                let text = self.decode_tokens(&tokens)?;
                if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
                    Self::log_decode_timing(
                        step_count,
                        total_forward_ns,
                        total_eval_ns,
                        total_item_ns,
                        total_other_ns,
                    );
                    return Ok(GenerationOutput {
                        text: truncated,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: completion_len,
                        token_logprobs: all_logprobs,
                    });
                }
            }

            if completion_len >= max_tokens {
                Self::log_decode_timing(
                    step_count,
                    total_forward_ns,
                    total_eval_ns,
                    total_item_ns,
                    total_other_ns,
                );
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: all_logprobs,
                });
            }

            // If thinking budget was just reached, override the pipelined token
            // so the next decode step gets </think> as input.
            if seen_think_close && thinking_tokens == THINKING_BUDGET {
                if let Some(close_id) = think_close_token {
                    next_token = Array::from_slice(&[close_id], &[1]);
                }
                thinking_tokens += 1; // prevent re-triggering
            } else {
                next_token = following;
            }
            next_logprob_data = following_logprob_data;
        }
    }

    /// `DFlash` block-diffusion speculative decode loop.
    ///
    /// Each round: drafter proposes a `block_size` block from the target's tap
    /// hidden states, the target verifies the block in a single tape-recording
    /// forward, `accept_prefix` takes the longest greedy-matching prefix, and a
    /// GDN tape replay rolls partial-accept state back bit-exactly. Headline
    /// metric for tuning is `accept_len` (mean accepted tokens per round).
    #[allow(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        clippy::too_many_lines
    )]
    fn generate_dflash_inner(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
    ) -> Result<GenerationOutput, EngineError> {
        let dflash = self
            .dflash
            .as_ref()
            .ok_or_else(|| EngineError::Generation("DFlash state missing".to_owned()))?;
        let prompt_len = Self::prompt_len(prompt_tokens)?;
        let has_stop_sequences = !stop_sequences.is_empty();

        let mut model = lock_or_recover(&self.model);
        let mut drafter = lock_or_recover(&dflash.drafter);

        let mut cache = model
            .make_cache_with_config(self.kv_cache_config)
            .map_err(EngineError::Mlx)?;

        // Prefill with taps.
        let prompt_array = Array::from(prompt_tokens).index(NewAxis);
        let (prefill_logits, taps) = model
            .forward_with_taps(&prompt_array, None, &mut cache, &dflash.tap_layers)
            .map_err(EngineError::Mlx)?;
        eval([&prefill_logits]).map_err(EngineError::Mlx)?;

        // Sample first token.
        let last_logits = prefill_logits.index((.., -1, ..));
        let first_token = sample(&last_logits, params).map_err(EngineError::Mlx)?;
        eval([&first_token]).map_err(EngineError::Mlx)?;

        let first_token_id: u32 = first_token.item();
        let mut tokens: Vec<u32> = vec![first_token_id];

        if self.eos_token_ids.contains(&first_token_id) || max_tokens <= 1 {
            let finish_reason = if self.eos_token_ids.contains(&first_token_id) {
                "stop"
            } else {
                "length"
            };
            return Ok(GenerationOutput {
                text: self.decode_tokens(&tokens)?,
                finish_reason: finish_reason.to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprobs: None,
            });
        }

        let mut current_taps = taps;
        let mut draft_cache = drafter.make_cache();
        let mut last_token = i32::try_from(first_token_id)
            .map_err(|_| EngineError::Generation("first_token_id overflow for i32".to_owned()))?;
        let mut start = i32::try_from(prompt_len)
            .map_err(|_| EngineError::Generation("prompt_len overflow for i32".to_owned()))?;
        // Adaptive (entropy-gated) block size. Acceptance length is the entropy
        // proxy: predictable/low-entropy regions accept long blocks (big win,
        // byte-exact), uncertain/high-entropy regions reject them. So grow the
        // block toward `block_max` when a block is fully accepted and shrink
        // toward `block_min` (≈ plain decode) when drafts are rejected — this
        // both recovers throughput and keeps the S>1 verify off high-entropy
        // near-ties (where it would flip argmax). Disable with
        // HIGGS_DFLASH_ADAPTIVE=0; floor via HIGGS_DFLASH_MIN_BLOCK.
        let block_max = dflash.block_size;
        let block_min = std::env::var("HIGGS_DFLASH_MIN_BLOCK")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|&v| v >= 1)
            .map_or(2, |v| v.min(block_max));
        let adaptive = std::env::var("HIGGS_DFLASH_ADAPTIVE").map_or(true, |v| v != "0");
        let mut block_size = block_max;
        // Smoothed utilization: a single hard token inside otherwise-predictable
        // text (code dip ~0.38 ≈ prose plateau ~0.36) shouldn't collapse the
        // block — only a *persistently* low utilization should. EMA separates
        // "occasional dip" from "sustained high entropy".
        let mut ema_util = 1.0_f64;
        let mask_id = dflash.mask_token_id;
        let t_start = std::time::Instant::now();
        // Acceptance-length aggregation: mean accepted tokens per round is the
        // headline metric for DFlash tuning (see P5).
        let mut total_accepted: u64 = 0;
        let mut rounds: u64 = 0;

        // Realized-speedup gate (auto-calibrated, no per-model knob). Measure
        // T_ar (wall per AR token) live; each spec round compute the realized
        // speedup ratio = n_accepted * T_ar / step_wall. If the EMA of that
        // ratio falls below 1.0, spec is actually slower than plain AR for this
        // text (MoE's fast AR raises the bar; a fixed entropy threshold can't
        // see that), so floor to AR and re-probe periodically. Flooring also
        // keeps the S>1 verify off high-entropy near-ties (where it both slows
        // down and flips argmax), so AR regions stay byte-exact. Start in AR for
        // one step to calibrate T_ar, then enter spec. Disable: HIGGS_DFLASH_GATE=0.
        let gate = std::env::var("HIGGS_DFLASH_GATE").map_or(true, |v| v != "0");
        let mut t_ar_ema: Option<f64> = None;
        let mut ratio_ema = 2.0_f64;
        let mut in_ar = true;
        let mut ar_run: u32 = 0;
        // Re-probe spec sparsely with exponential backoff. In a sustained
        // high-entropy region every probe is a wasted slow spec round (~4
        // decode-units for ~1-2 tokens), so a fixed cadence taxed the floor by
        // ~probe_cost/cadence (≈11% at 32 — the measured 0.89× prose gap; the
        // per-step taps were free, see dflash_floor_tapless_vs_taps_per_step_cost).
        // Each losing probe doubles the interval (amortizing the cost toward 0 on
        // sustained prose); a winning probe resets to base so a regime shift back
        // to spec-friendly text is caught within `AR_PROBE_BASE` tokens.
        const AR_PROBE_BASE: u32 = 32;
        // ponytail: cap the backoff at 512 — residual probe tax <1% (≈0.99×),
        // and recovery never lags a regime change by more than ~512 tokens.
        const AR_PROBE_MAX: u32 = 512;
        let mut probe_every = AR_PROBE_BASE;

        // The first decode after prefill is kernel-cold on Metal (~8× steady
        // state: measured t_ar=187ms vs ~23ms warm). Seeding the gate's AR
        // reference from it froze t_ar high, so the realized-speedup ratio was
        // always >1 and the gate NEVER floored — a dead no-op. Fix: spend a short
        // warm-up window in AR, discard the cold sample, seed t_ar from the warm
        // steps, then hand off to spec. These are real greedy tokens, not waste.
        // ponytail: no periodic re-calibration — t_ar only updates on floored
        // steps, so during a spec-winning streak it holds its last warm value
        // (can't drift high to mask a loss); a real degradation re-floors and
        // refreshes it. Add periodic recal only if AR-baseline thermal drift
        // during long spec streaks ever proves to matter.
        const T_AR_CALIB: u32 = 3;
        let mut calibrated = false;

        // Spec-side cold-start grace (symmetric to T_AR_CALIB). The first spec
        // rounds after every (re-)entry are cold on two axes the warm t_ar is
        // not: the verify/drafter/lm_head Metal kernels are kernel-cold
        // (step_wall ~2x inflated until they JIT-warm) and the drafter KV cache
        // is empty (accept length climbs from ~3 to its steady ~6 as it fills).
        // Judging those rounds floored winning workloads (9B code: 1.53x->0.83x)
        // and then death-spiralled (floored -> spec re-cools -> re-probe also
        // looks bad). So don't update ratio_ema or floor until the burst warms.
        const SPEC_WARMUP: u32 = 3;
        let mut spec_warmup_left: u32 = 0;

        loop {
            let round_t0 = std::time::Instant::now();
            if gate && in_ar {
                // AR floor: plain S=1 decode (byte-exact). Measures T_ar live.
                //
                // Taps are only consumed by the drafter when we re-enter spec next
                // round, i.e. on the step that hands control back ("leaving"): the
                // end of the initial calibration window, or a probe cooldown. Use
                // the optimized tap-less `forward` for every other floored step.
                // The two paths share `forward_raw_hidden` (identical layer loop,
                // mask, and KV/GDN cache mutation) and the same `project_logits`,
                // so logits + cache are byte-identical; the only thing dropped is a
                // tap clone the next floored step would overwrite unused.
                let calibrating = !calibrated;
                let window = if calibrating { T_AR_CALIB } else { probe_every };
                let leaving = ar_run + 1 >= window;
                let need_taps = leaving;
                let single = Array::from_slice(&[last_token], &[1, 1]);
                let ar_logits = if need_taps {
                    let (logits, ar_taps) = model
                        .forward_with_taps(&single, None, &mut cache, &dflash.tap_layers)
                        .map_err(EngineError::Mlx)?;
                    current_taps = ar_taps;
                    logits
                } else {
                    model
                        .forward(&single, None, &mut cache)
                        .map_err(EngineError::Mlx)?
                };
                let ar_next =
                    sample(&ar_logits.index((.., -1, ..)), params).map_err(EngineError::Mlx)?;
                eval([&ar_next]).map_err(EngineError::Mlx)?;
                let dt = round_t0.elapsed().as_secs_f64();
                // Skip the kernel-cold first decode (ar_run == 0 of the initial
                // calibration window); every later floored step is warm.
                if !(calibrating && ar_run == 0) {
                    t_ar_ema = Some(t_ar_ema.map_or(dt, |e| 0.7f64.mul_add(e, 0.3 * dt)));
                }
                let ar_id: u32 = ar_next.item();
                tokens.push(ar_id);
                last_token = i32::try_from(ar_id)
                    .map_err(|_| EngineError::Generation("ar token overflow".to_owned()))?;
                start += 1;
                ar_run += 1;
                // Hand back to spec once the warm calibration window completes, or
                // after a probe cooldown re-tests whether spec is worthwhile again.
                if leaving {
                    in_ar = false;
                    ar_run = 0;
                    calibrated = true;
                    // Grant the fresh spec burst its cold-start grace and re-seed
                    // the EMA to the optimistic prior. Re-seeding (not blending)
                    // is load-bearing: otherwise a re-probe inherits the stale
                    // sub-1.0 ratio_ema and re-floors in one warm round, undoing
                    // the grace. Mirrors the t_ar_ema map_or seed.
                    spec_warmup_left = SPEC_WARMUP;
                    ratio_ema = 2.0;
                }
            } else {
                // a. Build block: [anchor, mask, mask, ...]
                let block_size_us = usize::try_from(block_size).map_err(|_| {
                    EngineError::Generation("block_size overflow for usize".to_owned())
                })?;
                let mut block_tokens = vec![mask_id; block_size_us];
                if let Some(slot) = block_tokens.get_mut(0) {
                    *slot = last_token;
                }
                let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);

                // b. Embed through target's embedding layer.
                let noise_embedding = model
                    .embed_token_ids(&block_ids)
                    .map_err(EngineError::Mlx)?;

                // c. Drafter forward.
                let draft_hidden = drafter
                    .forward(&noise_embedding, &current_taps, &mut draft_cache)
                    .map_err(EngineError::Mlx)?;
                crop_drafter_cache(&mut draft_cache, start);

                // d. Target lm_head on sliced hidden -> argmax draft tokens.
                let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
                let draft_logits = model
                    .forward_all_logits_from_hidden(&draft_hidden_sliced)
                    .map_err(EngineError::Mlx)?;
                let draft_token_arr =
                    mlx_rs::argmax_axis!(draft_logits, -1).map_err(EngineError::Mlx)?;
                eval([&draft_token_arr]).map_err(EngineError::Mlx)?;
                let draft_u32: Vec<u32> = draft_token_arr
                    .reshape(&[-1])
                    .map_err(EngineError::Mlx)?
                    .as_slice::<u32>()
                    .to_vec();
                let draft_i32: Vec<i32> = draft_u32
                    .iter()
                    .map(|&x| {
                        i32::try_from(x).map_err(|_| {
                            EngineError::Generation("draft token overflow i32".to_owned())
                        })
                    })
                    .collect::<Result<_, _>>()?;

                // e. Build verify input: [anchor, draft_0..draft_{N-2}].
                let mut verify_tokens = vec![last_token];
                verify_tokens.extend_from_slice(&draft_i32);
                let verify_len = i32::try_from(verify_tokens.len())
                    .map_err(|_| EngineError::Generation("verify_len overflow".to_owned()))?;
                let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

                // f. Tape-recording verify: forward with taps + GDN innovation tapes.
                //    The tape kernel's recurrence is bit-exact with sequential AR
                //    steps, fixing the S>1 vs S=1 numerical drift that a full-rerun
                //    verify exhibits, and lets partial accept replay only the SSM
                //    recurrence for accepted positions instead of a full rerun.
                let (verify_logits, verify_taps, layer_tapes) = model
                    .forward_with_taps_tape(&verify_input, None, &mut cache, &dflash.tap_layers)
                    .map_err(EngineError::Mlx)?;

                // g. Accept prefix. One host barrier per round: eval'ing the argmax
                //    pulls verify_logits + the whole verify forward through the
                //    graph, so a separate eval([&verify_logits]) is redundant.
                let verify_argmax =
                    mlx_rs::argmax_axis!(verify_logits, -1).map_err(EngineError::Mlx)?;
                eval([&verify_argmax]).map_err(EngineError::Mlx)?;
                let verify_flat: Vec<u32> = verify_argmax
                    .reshape(&[-1])
                    .map_err(EngineError::Mlx)?
                    .as_slice::<u32>()
                    .to_vec();
                let accepted = accept_prefix(&draft_u32, &verify_flat);
                let n_accepted = i32::try_from(accepted.len())
                    .map_err(|_| EngineError::Generation("n_accepted overflow".to_owned()))?;
                rounds += 1;
                total_accepted += accepted.len() as u64;

                if std::env::var("HIGGS_DFLASH_TRACE").is_ok() && tokens.len() <= 32 {
                    tracing::info!(
                        drafts = ?draft_u32,
                        verify_argmax = ?verify_flat,
                        n_accepted,
                        accepted = ?accepted,
                        "DFlash iter trace"
                    );
                }

                // h. Partial accept — GDN-only replay from tape. The verify
                //    advanced state for ALL positions; on partial rejection we
                //    restore each GDN layer's snapshot, replay the SSM kernel for
                //    the n_accepted positions, and trim KV layers by the rejected
                //    count. Issued lazily (no eval) so it folds into the next
                //    verify's host barrier.
                if n_accepted < block_size {
                    let kv_rollback = verify_len - n_accepted;
                    model
                        .replay_tape_rollback(&layer_tapes, &mut cache, n_accepted, kv_rollback)
                        .map_err(EngineError::Mlx)?;
                }
                // Slice verify taps to accepted positions (valid for both full and
                // partial accept — earlier positions' hidden states are causally
                // independent of later ones).
                current_taps = verify_taps
                    .into_iter()
                    .map(|tap| tap.index((.., ..n_accepted, ..)))
                    .collect();

                // i. Update state.
                for &tok in &accepted {
                    tokens.push(tok);
                }
                last_token = i32::try_from(*accepted.last().ok_or_else(|| {
                    EngineError::Generation("accept_prefix returned empty vec".to_owned())
                })?)
                .map_err(|_| EngineError::Generation("accepted token overflow i32".to_owned()))?;
                start += n_accepted;

                // Adapt the next round's block size by block UTILIZATION
                // (n_accepted / block_size) — the entropy proxy. On low-entropy text
                // acceptance scales with the block (util stays high → grow toward
                // block_max for the biggest win); on high-entropy text acceptance
                // plateaus (util drops → shrink so the verify isn't wasted, and the
                // S>1 verify stays off near-ties). Gating on util, not raw accept,
                // keeps big blocks where they pay off. (all block_size uses above.)
                if adaptive {
                    let util = f64::from(n_accepted) / f64::from(block_size);
                    ema_util = 0.5f64.mul_add(ema_util, 0.5 * util);
                    block_size = if ema_util >= 0.75 {
                        (block_size * 2).min(block_max) // sustained high → grow
                    } else if ema_util <= 0.5 {
                        (block_size / 2).max(block_min) // sustained low → shrink
                    } else {
                        block_size
                    };
                }

                // Realized-speedup gate: floor to AR next round if spec has become
                // slower than plain AR (or to calibrate T_ar if we have no sample).
                if gate {
                    let step_wall = round_t0.elapsed().as_secs_f64().max(1e-9);
                    if spec_warmup_left > 0 {
                        // Cold spec warm-up: stay in spec, don't judge or floor,
                        // don't poison ratio_ema with the cold step_wall / low
                        // cold-cache accept. The kernels warm and the drafter
                        // cache fills over these rounds.
                        spec_warmup_left -= 1;
                        if std::env::var("HIGGS_DFLASH_TRACE").is_ok() && rounds <= 8 {
                            tracing::info!(
                                round = rounds,
                                n_accepted,
                                step_wall_ms = format!("{:.1}", step_wall * 1e3),
                                warmup_left = spec_warmup_left,
                                "GATE warmup (discarded)"
                            );
                        }
                    } else if let Some(t_ar) = t_ar_ema {
                        let ratio = f64::from(n_accepted) * t_ar / step_wall;
                        ratio_ema = 0.6f64.mul_add(ratio_ema, 0.4 * ratio);
                        if std::env::var("HIGGS_DFLASH_TRACE").is_ok() && rounds <= 8 {
                            tracing::info!(
                                round = rounds,
                                n_accepted,
                                t_ar_ms = format!("{:.1}", t_ar * 1e3),
                                step_wall_ms = format!("{:.1}", step_wall * 1e3),
                                ratio = format!("{ratio:.2}"),
                                ratio_ema = format!("{ratio_ema:.2}"),
                                "GATE"
                            );
                        }
                        if ratio_ema < 1.0 {
                            // Losing probe: floor and back off so the next probe
                            // amortizes over a longer AR run (sustained prose).
                            in_ar = true;
                            ar_run = 0;
                            probe_every = probe_every.saturating_mul(2).min(AR_PROBE_MAX);
                        } else {
                            // Winning: stay in spec, but reset the cadence so a
                            // later degradation is re-probed promptly, not lazily.
                            probe_every = AR_PROBE_BASE;
                        }
                    } else {
                        // No T_ar sample yet (initial calibration) — floor at base
                        // cadence; this isn't a losing probe, so don't back off.
                        in_ar = true;
                        ar_run = 0;
                        probe_every = AR_PROBE_BASE;
                    }
                }
            } // end spec-round branch

            // j. Check termination.
            let completion_len = Self::completion_len(&tokens)?;
            let accept_len = if rounds > 0 {
                total_accepted as f64 / rounds as f64
            } else {
                0.0
            };

            if tokens.iter().any(|t| self.eos_token_ids.contains(t)) {
                let secs = t_start.elapsed().as_secs_f64();
                let tok_per_sec = if secs > 0.0 {
                    tokens.len() as f64 / secs
                } else {
                    0.0
                };
                tracing::info!(
                    tokens = tokens.len(),
                    accept_len = format!("{accept_len:.2}"),
                    tok_per_sec = format!("{tok_per_sec:.1}"),
                    "DFlash generation complete"
                );
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: None,
                });
            }

            if has_stop_sequences {
                let text = self.decode_tokens(&tokens)?;
                if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
                    return Ok(GenerationOutput {
                        text: truncated,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: completion_len,
                        token_logprobs: None,
                    });
                }
            }

            if completion_len >= max_tokens {
                let secs = t_start.elapsed().as_secs_f64();
                let tok_per_sec = if secs > 0.0 {
                    tokens.len() as f64 / secs
                } else {
                    0.0
                };
                tracing::info!(
                    tokens = tokens.len(),
                    accept_len = format!("{accept_len:.2}"),
                    spec_rounds = rounds,
                    probe_every = probe_every,
                    tok_per_sec = format!("{tok_per_sec:.1}"),
                    "DFlash generation complete (length limit)"
                );
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: None,
                });
            }
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    fn log_decode_timing(
        steps: u32,
        forward_ns: u128,
        eval_ns: u128,
        item_ns: u128,
        other_ns: u128,
    ) {
        if steps > 0 {
            let s = f64::from(steps);
            tracing::info!(
                steps,
                forward_ms = format!("{:.2}", forward_ns as f64 / s / 1e6),
                eval_ms = format!("{:.2}", eval_ns as f64 / s / 1e6),
                item_ms = format!("{:.2}", item_ns as f64 / s / 1e6),
                other_ms = format!("{:.2}", other_ns as f64 / s / 1e6),
                total_ms = format!(
                    "{:.2}",
                    (forward_ns + eval_ns + item_ns + other_ns) as f64 / s / 1e6
                ),
                "Decode loop timing (per step avg)"
            );
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn log_mtp_decode_stats(
        stats: &crate::mtp::MtpStats,
        elapsed: std::time::Duration,
        reason: &str,
    ) {
        tracing::info!(
            reason,
            cycles = stats.cycles(),
            drafted = stats.drafted(),
            accepted_drafts = stats.accepted_drafts(),
            emitted = stats.emitted(),
            accept_rate = format!("{:.1}%", stats.acceptance_rate_percent()),
            tok_per_s = format!("{:.1}", f64::from(stats.emitted()) / elapsed.as_secs_f64()),
            "MTP decode complete"
        );
    }

    #[allow(clippy::cast_precision_loss)]
    fn log_prompt_lookup_decode_stats(
        stats: &crate::mtp::MtpStats,
        elapsed: std::time::Duration,
        reason: &str,
    ) {
        tracing::info!(
            reason,
            cycles = stats.cycles(),
            drafted = stats.drafted(),
            accepted_drafts = stats.accepted_drafts(),
            emitted = stats.emitted(),
            accept_rate = format!("{:.1}%", stats.acceptance_rate_percent()),
            tok_per_s = format!("{:.1}", f64::from(stats.emitted()) / elapsed.as_secs_f64()),
            "Prompt-lookup decode complete"
        );
    }

    /// Architecture-neutral prompt-lookup speculative decode loop.
    ///
    /// The draft provider copies likely next tokens from repeated prompt/history
    /// spans, then verifies the whole candidate window in one model pass.
    #[allow(
        clippy::too_many_arguments,
        clippy::as_conversions,
        clippy::cast_precision_loss
    )]
    fn prompt_lookup_generate(
        &self,
        model: &mut higgs_models::AnyModel,
        cache: &mut higgs_models::AnyCache,
        first_token_id: u32,
        max_tokens: u32,
        prompt_len: u32,
        tokens: &mut Vec<u32>,
        stop_sequences: &[String],
        enable_thinking: bool,
    ) -> Result<GenerationOutput, EngineError> {
        let has_stop_sequences = !stop_sequences.is_empty();
        let unchecked_lookup = unchecked_prompt_lookup_enabled();
        let first_token_i32 = i32::try_from(first_token_id)
            .map_err(|_| EngineError::Generation("token id exceeds i32 range".to_owned()))?;
        let first_input = Array::from_slice(&[first_token_i32], &[1, 1]);
        let logits = model
            .forward_all_logits(&first_input, None, cache)
            .map_err(EngineError::Mlx)?;
        let next_arr =
            mlx_rs::argmax_axis!(&logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
        eval([&next_arr]).map_err(EngineError::Mlx)?;

        let mut confirmed_token_id: u32 = next_arr.item();
        let base_config = prompt_lookup_config();
        let mut stats = crate::mtp::MtpStats::default();
        let t_start = std::time::Instant::now();

        const THINKING_BUDGET: u32 = 256;
        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        let mut thinking_tokens: u32 = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        loop {
            let completion_len = Self::completion_len(tokens)?;
            if completion_len >= max_tokens {
                let elapsed = t_start.elapsed();
                Self::log_prompt_lookup_decode_stats(&stats, elapsed, "length");
                return Ok(GenerationOutput {
                    text: self.decode_tokens(tokens)?,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: None,
                });
            }

            let remaining = usize::try_from(max_tokens.saturating_sub(completion_len))
                .map_err(|_| EngineError::Generation("max_tokens overflow".to_owned()))?;
            let config = crate::mtp::PromptLookupConfig {
                max_drafts: base_config.max_drafts.min(remaining.saturating_sub(1)),
                ..base_config
            };

            let result = if unchecked_lookup {
                crate::mtp::unchecked_prompt_lookup_cycle(
                    model,
                    cache,
                    tokens,
                    confirmed_token_id,
                    config,
                )?
            } else {
                crate::mtp::prompt_lookup_cycle(model, cache, tokens, confirmed_token_id, config)?
            };
            stats.record_cycle(result.drafted, result.tokens.len(), result.accepted_drafts);

            for &tok in &result.tokens {
                if let Some(close_id) = think_close_token {
                    if !seen_think_close {
                        if tok == close_id {
                            seen_think_close = true;
                        } else {
                            thinking_tokens += 1;
                            if thinking_tokens >= THINKING_BUDGET {
                                tokens.push(close_id);
                                seen_think_close = true;
                                tracing::info!(
                                    budget = THINKING_BUDGET,
                                    "Prompt lookup: thinking budget reached, forcing </think>"
                                );
                                if self.eos_token_ids.contains(&close_id) {
                                    let elapsed = t_start.elapsed();
                                    Self::log_prompt_lookup_decode_stats(&stats, elapsed, "stop");
                                    return Ok(GenerationOutput {
                                        text: self.decode_tokens(tokens)?,
                                        finish_reason: "stop".to_owned(),
                                        prompt_tokens: prompt_len,
                                        completion_tokens: Self::completion_len(tokens)?,
                                        token_logprobs: None,
                                    });
                                }

                                if has_stop_sequences {
                                    let text = self.decode_tokens(tokens)?;
                                    if let Some(truncated) =
                                        check_stop_sequences(&text, stop_sequences)
                                    {
                                        let elapsed = t_start.elapsed();
                                        Self::log_prompt_lookup_decode_stats(
                                            &stats, elapsed, "stop",
                                        );
                                        return Ok(GenerationOutput {
                                            text: truncated,
                                            finish_reason: "stop".to_owned(),
                                            prompt_tokens: prompt_len,
                                            completion_tokens: Self::completion_len(tokens)?,
                                            token_logprobs: None,
                                        });
                                    }
                                }

                                let forced_completion_len = Self::completion_len(tokens)?;
                                if forced_completion_len >= max_tokens {
                                    let elapsed = t_start.elapsed();
                                    Self::log_prompt_lookup_decode_stats(&stats, elapsed, "length");
                                    return Ok(GenerationOutput {
                                        text: self.decode_tokens(tokens)?,
                                        finish_reason: "length".to_owned(),
                                        prompt_tokens: prompt_len,
                                        completion_tokens: forced_completion_len,
                                        token_logprobs: None,
                                    });
                                }
                                break;
                            }
                        }
                    }
                }

                tokens.push(tok);

                if self.eos_token_ids.contains(&tok) {
                    let elapsed = t_start.elapsed();
                    Self::log_prompt_lookup_decode_stats(&stats, elapsed, "stop");
                    return Ok(GenerationOutput {
                        text: self.decode_tokens(tokens)?,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: Self::completion_len(tokens)?,
                        token_logprobs: None,
                    });
                }
            }

            if has_stop_sequences {
                let text = self.decode_tokens(tokens)?;
                if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
                    let elapsed = t_start.elapsed();
                    Self::log_prompt_lookup_decode_stats(&stats, elapsed, "stop");
                    return Ok(GenerationOutput {
                        text: truncated,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: Self::completion_len(tokens)?,
                        token_logprobs: None,
                    });
                }
            }

            let final_completion_len = Self::completion_len(tokens)?;
            if final_completion_len >= max_tokens {
                let elapsed = t_start.elapsed();
                Self::log_prompt_lookup_decode_stats(&stats, elapsed, "length");
                return Ok(GenerationOutput {
                    text: self.decode_tokens(tokens)?,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: final_completion_len,
                    token_logprobs: None,
                });
            }

            confirmed_token_id = result.next_token_id;
        }
    }

    /// MTP speculative decode loop.
    ///
    /// Runs the backbone to get the initial hidden state, then loops calling
    /// `mtp_cycle()` which drafts multiple tokens per cycle for speculative speedup.
    #[allow(
        clippy::too_many_arguments,
        clippy::as_conversions,
        clippy::cast_precision_loss
    )]
    fn mtp_generate(
        &self,
        model: &mut higgs_models::AnyModel,
        cache: &mut higgs_models::AnyCache,
        actual_prompt_tokens: &[u32],
        prefill_hidden: Option<&Array>,
        first_token_id: u32,
        max_tokens: u32,
        prompt_len: u32,
        tokens: &mut Vec<u32>,
        stop_sequences: &[String],
        enable_thinking: bool,
    ) -> Result<GenerationOutput, EngineError> {
        let has_stop_sequences = !stop_sequences.is_empty();

        // Create MTP cache for the MTP head's attention layer(s).
        let mut mtp_cache = model
            .make_mtp_cache()
            .ok_or_else(|| EngineError::Generation("MTP cache creation failed".into()))?;
        if let Some(hidden) = prefill_hidden {
            crate::mtp::prime_mtp_cache(model, &mut mtp_cache, actual_prompt_tokens, hidden)?;
        }

        let first_input = Array::from_slice(&[first_token_id as i32], &[1, 1]);
        let (hidden, logits) = model
            .forward_with_hidden(&first_input, None, cache)
            .map_err(EngineError::Mlx)?;
        let next_arr =
            mlx_rs::argmax_axis!(&logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
        let h = hidden.index((.., -1.., ..));
        if let Some(previous_hidden) = prefill_hidden
            .filter(|_| !actual_prompt_tokens.is_empty())
            .map(|prefill| Self::hidden_row_from_sequence(prefill, actual_prompt_tokens.len() - 1))
            .transpose()?
        {
            crate::mtp::mirror_mtp_token(model, &mut mtp_cache, &previous_hidden, first_token_id)?;
        }
        eval([&next_arr, &h]).map_err(EngineError::Mlx)?;

        let mut current_hidden = h;
        let mut confirmed_token_id: u32 = next_arr.item();
        let mut mtp_stats = crate::mtp::MtpStats::default();
        let mut adaptive_depth = mtp_adaptive_draft_enabled()
            .then(|| adaptive_draft_depth_for_cap(self.tuning.mtp_draft_n_max()));
        let hybrid_prompt_lookup = mtp_prompt_lookup_enabled();
        let hybrid_prompt_lookup_config = prompt_lookup_config();
        let t_start = std::time::Instant::now();

        // Thinking budget: force </think> after N tokens if model hasn't closed it.
        const THINKING_BUDGET: u32 = 256;
        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        // Seed thinking state from the first token (already emitted by caller).
        let mut thinking_tokens: u32 = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        loop {
            let cycle_completion_len = Self::completion_len(tokens)?;
            let remaining = usize::try_from(max_tokens.saturating_sub(cycle_completion_len))
                .map_err(|_| EngineError::Generation("max_tokens overflow".to_owned()))?;
            let draft_depth = adaptive_depth
                .as_ref()
                .map_or_else(
                    || self.tuning.mtp_draft_n_max(),
                    crate::mtp::AdaptiveDraftDepth::current,
                )
                .min(remaining.saturating_sub(1).max(1));
            let prompt_config = crate::mtp::PromptLookupConfig {
                max_drafts: hybrid_prompt_lookup_config
                    .max_drafts
                    .min(remaining.saturating_sub(1)),
                ..hybrid_prompt_lookup_config
            };
            let result = if hybrid_prompt_lookup && prompt_config.max_drafts > 0 {
                crate::mtp::mtp_prompt_lookup_cycle(
                    model,
                    cache,
                    &mut mtp_cache,
                    &current_hidden,
                    tokens,
                    confirmed_token_id,
                    prompt_config,
                )?
                .map_or_else(
                    || {
                        crate::mtp::mtp_cycle(
                            model,
                            cache,
                            &mut mtp_cache,
                            &current_hidden,
                            confirmed_token_id,
                            draft_depth,
                        )
                    },
                    Ok,
                )?
            } else {
                crate::mtp::mtp_cycle(
                    model,
                    cache,
                    &mut mtp_cache,
                    &current_hidden,
                    confirmed_token_id,
                    draft_depth,
                )?
            };

            mtp_stats.record_cycle(result.drafted, result.tokens.len(), result.accepted_drafts);
            if let Some(depth) = &mut adaptive_depth {
                depth.observe(result.accepted_drafts, result.drafted);
            }

            for &tok in &result.tokens {
                // Thinking budget enforcement
                if let Some(close_id) = think_close_token {
                    if !seen_think_close {
                        if tok == close_id {
                            seen_think_close = true;
                        } else {
                            thinking_tokens += 1;
                            if thinking_tokens >= THINKING_BUDGET {
                                tokens.push(close_id);
                                seen_think_close = true;
                                tracing::info!(
                                    budget = THINKING_BUDGET,
                                    "MTP: thinking budget reached, forcing </think>"
                                );
                                if self.eos_token_ids.contains(&close_id) {
                                    let elapsed = t_start.elapsed();
                                    Self::log_mtp_decode_stats(&mtp_stats, elapsed, "stop");
                                    return Ok(GenerationOutput {
                                        text: self.decode_tokens(tokens)?,
                                        finish_reason: "stop".to_owned(),
                                        prompt_tokens: prompt_len,
                                        completion_tokens: Self::completion_len(tokens)?,
                                        token_logprobs: None,
                                    });
                                }

                                if has_stop_sequences {
                                    let text = self.decode_tokens(tokens)?;
                                    if let Some(truncated) =
                                        check_stop_sequences(&text, stop_sequences)
                                    {
                                        let elapsed = t_start.elapsed();
                                        Self::log_mtp_decode_stats(&mtp_stats, elapsed, "stop");
                                        return Ok(GenerationOutput {
                                            text: truncated,
                                            finish_reason: "stop".to_owned(),
                                            prompt_tokens: prompt_len,
                                            completion_tokens: Self::completion_len(tokens)?,
                                            token_logprobs: None,
                                        });
                                    }
                                }

                                let completion_len = Self::completion_len(tokens)?;
                                if completion_len >= max_tokens {
                                    let elapsed = t_start.elapsed();
                                    Self::log_mtp_decode_stats(&mtp_stats, elapsed, "length");
                                    return Ok(GenerationOutput {
                                        text: self.decode_tokens(tokens)?,
                                        finish_reason: "length".to_owned(),
                                        prompt_tokens: prompt_len,
                                        completion_tokens: completion_len,
                                        token_logprobs: None,
                                    });
                                }

                                // Skip remaining tokens from this cycle
                                break;
                            }
                        }
                    }
                }

                tokens.push(tok);

                if self.eos_token_ids.contains(&tok) {
                    let elapsed = t_start.elapsed();
                    Self::log_mtp_decode_stats(&mtp_stats, elapsed, "stop");
                    return Ok(GenerationOutput {
                        text: self.decode_tokens(tokens)?,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: Self::completion_len(tokens)?,
                        token_logprobs: None,
                    });
                }
            }

            if has_stop_sequences {
                let text = self.decode_tokens(tokens)?;
                if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
                    let elapsed = t_start.elapsed();
                    Self::log_mtp_decode_stats(&mtp_stats, elapsed, "stop");
                    return Ok(GenerationOutput {
                        text: truncated,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: Self::completion_len(tokens)?,
                        token_logprobs: None,
                    });
                }
            }

            let completion_len = Self::completion_len(tokens)?;
            if completion_len >= max_tokens {
                let elapsed = t_start.elapsed();
                Self::log_mtp_decode_stats(&mtp_stats, elapsed, "length");
                return Ok(GenerationOutput {
                    text: self.decode_tokens(tokens)?,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: None,
                });
            }

            current_hidden = result.hidden;
            confirmed_token_id = result.next_token_id;
        }
    }

    /// MTP speculative decode loop — streaming variant.
    ///
    /// Same logic as `mtp_generate`, but sends each accepted token (or pair)
    /// via the streaming channel instead of accumulating into a buffer.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::as_conversions,
        clippy::cast_precision_loss
    )]
    fn mtp_generate_streaming(
        &self,
        model: &mut higgs_models::AnyModel,
        cache: &mut higgs_models::AnyCache,
        actual_prompt_tokens: &[u32],
        prefill_hidden: Option<&Array>,
        first_token_id: u32,
        max_tokens: u32,
        prompt_len: u32,
        tokens: &mut Vec<u32>,
        stop_sequences: &[String],
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        mut detok: IncrementalDetok,
        enable_thinking: bool,
    ) -> Result<(), EngineError> {
        let has_stop_sequences = !stop_sequences.is_empty();

        let mut mtp_cache = model
            .make_mtp_cache()
            .ok_or_else(|| EngineError::Generation("MTP cache creation failed".into()))?;
        if let Some(hidden) = prefill_hidden {
            crate::mtp::prime_mtp_cache(model, &mut mtp_cache, actual_prompt_tokens, hidden)?;
        }

        let first_input = Array::from_slice(&[first_token_id as i32], &[1, 1]);
        let (hidden, logits) = model
            .forward_with_hidden(&first_input, None, cache)
            .map_err(EngineError::Mlx)?;
        let next_arr =
            mlx_rs::argmax_axis!(&logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
        let h = hidden.index((.., -1.., ..));
        if let Some(previous_hidden) = prefill_hidden
            .filter(|_| !actual_prompt_tokens.is_empty())
            .map(|prefill| Self::hidden_row_from_sequence(prefill, actual_prompt_tokens.len() - 1))
            .transpose()?
        {
            crate::mtp::mirror_mtp_token(model, &mut mtp_cache, &previous_hidden, first_token_id)?;
        }
        eval([&next_arr, &h]).map_err(EngineError::Mlx)?;

        let mut current_hidden = h;
        let mut confirmed_token_id: u32 = next_arr.item();
        let mut mtp_stats = crate::mtp::MtpStats::default();
        let mut adaptive_depth = mtp_adaptive_draft_enabled()
            .then(|| adaptive_draft_depth_for_cap(self.tuning.mtp_draft_n_max()));
        let hybrid_prompt_lookup = mtp_prompt_lookup_enabled();
        let hybrid_prompt_lookup_config = prompt_lookup_config();
        let t_start = std::time::Instant::now();

        const THINKING_BUDGET: u32 = 256;
        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        let mut thinking_tokens: u32 = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        loop {
            let cycle_completion_len = Self::completion_len(tokens)?;
            let remaining = usize::try_from(max_tokens.saturating_sub(cycle_completion_len))
                .map_err(|_| EngineError::Generation("max_tokens overflow".to_owned()))?;
            let draft_depth = adaptive_depth
                .as_ref()
                .map_or_else(
                    || self.tuning.mtp_draft_n_max(),
                    crate::mtp::AdaptiveDraftDepth::current,
                )
                .min(remaining.saturating_sub(1).max(1));
            let prompt_config = crate::mtp::PromptLookupConfig {
                max_drafts: hybrid_prompt_lookup_config
                    .max_drafts
                    .min(remaining.saturating_sub(1)),
                ..hybrid_prompt_lookup_config
            };
            let result = if hybrid_prompt_lookup && prompt_config.max_drafts > 0 {
                crate::mtp::mtp_prompt_lookup_cycle(
                    model,
                    cache,
                    &mut mtp_cache,
                    &current_hidden,
                    tokens,
                    confirmed_token_id,
                    prompt_config,
                )?
                .map_or_else(
                    || {
                        crate::mtp::mtp_cycle(
                            model,
                            cache,
                            &mut mtp_cache,
                            &current_hidden,
                            confirmed_token_id,
                            draft_depth,
                        )
                    },
                    Ok,
                )?
            } else {
                crate::mtp::mtp_cycle(
                    model,
                    cache,
                    &mut mtp_cache,
                    &current_hidden,
                    confirmed_token_id,
                    draft_depth,
                )?
            };

            mtp_stats.record_cycle(result.drafted, result.tokens.len(), result.accepted_drafts);
            if let Some(depth) = &mut adaptive_depth {
                depth.observe(result.accepted_drafts, result.drafted);
            }

            for &tok in &result.tokens {
                // Thinking budget enforcement
                if let Some(close_id) = think_close_token {
                    if !seen_think_close {
                        if tok == close_id {
                            seen_think_close = true;
                        } else {
                            thinking_tokens += 1;
                            if thinking_tokens >= THINKING_BUDGET {
                                tokens.push(close_id);
                                seen_think_close = true;
                                tracing::info!(
                                    budget = THINKING_BUDGET,
                                    "MTP streaming: thinking budget reached, forcing </think>"
                                );

                                let is_eos = self.eos_token_ids.contains(&close_id);
                                let completion_len = Self::completion_len(tokens)?;
                                let is_max = completion_len >= max_tokens;

                                let new_text = detok.append(&self.tokenizer, tokens)?;
                                let emitted_before = detok.text.len() - new_text.len();
                                let (mut final_new_text, hit_stop_seq) = if has_stop_sequences {
                                    find_stop_in_tail(&detok.text, new_text.len(), stop_sequences)
                                        .map_or((new_text, false), |pos| {
                                            let emit = detok
                                                .text
                                                .get(emitted_before..pos)
                                                .unwrap_or_default()
                                                .to_owned();
                                            (emit, true)
                                        })
                                } else {
                                    (new_text, false)
                                };
                                let step_finished = is_eos || is_max || hit_stop_seq;
                                if step_finished && !hit_stop_seq {
                                    final_new_text.push_str(&detok.flush(&self.tokenizer, tokens)?);
                                }
                                let finish_reason = if is_eos || hit_stop_seq {
                                    Some("stop".to_owned())
                                } else if is_max {
                                    Some("length".to_owned())
                                } else {
                                    None
                                };

                                if step_finished {
                                    let elapsed = t_start.elapsed();
                                    Self::log_mtp_decode_stats(
                                        &mtp_stats,
                                        elapsed,
                                        finish_reason.as_deref().unwrap_or("client"),
                                    );
                                }

                                if sender
                                    .blocking_send(StreamingOutput {
                                        new_text: final_new_text,
                                        finished: step_finished,
                                        finish_reason,
                                        prompt_tokens: prompt_len,
                                        completion_tokens: completion_len,
                                        token_logprob: None,
                                        prefill_progress: None,
                                    })
                                    .is_err()
                                {
                                    return Ok(());
                                }

                                if step_finished {
                                    return Ok(());
                                }

                                break;
                            }
                        }
                    }
                }

                tokens.push(tok);

                let is_eos = self.eos_token_ids.contains(&tok);
                let completion_len = Self::completion_len(tokens)?;
                let is_max = completion_len >= max_tokens;

                let new_text = detok.append(&self.tokenizer, tokens)?;
                let emitted_before = detok.text.len() - new_text.len();

                let (mut final_new_text, hit_stop_seq) = if has_stop_sequences {
                    find_stop_in_tail(&detok.text, new_text.len(), stop_sequences).map_or(
                        (new_text, false),
                        |pos| {
                            let emit = detok
                                .text
                                .get(emitted_before..pos)
                                .unwrap_or_default()
                                .to_owned();
                            (emit, true)
                        },
                    )
                } else {
                    (new_text, false)
                };

                let step_finished = is_eos || is_max || hit_stop_seq;
                if step_finished && !hit_stop_seq {
                    final_new_text.push_str(&detok.flush(&self.tokenizer, tokens)?);
                }
                let finish_reason = if is_eos || hit_stop_seq {
                    Some("stop".to_owned())
                } else if is_max {
                    Some("length".to_owned())
                } else {
                    None
                };

                if step_finished {
                    let elapsed = t_start.elapsed();
                    Self::log_mtp_decode_stats(
                        &mtp_stats,
                        elapsed,
                        finish_reason.as_deref().unwrap_or("client"),
                    );
                }

                if sender
                    .blocking_send(StreamingOutput {
                        new_text: final_new_text,
                        finished: step_finished,
                        finish_reason,
                        prompt_tokens: prompt_len,
                        completion_tokens: completion_len,
                        token_logprob: None,
                        prefill_progress: None,
                    })
                    .is_err()
                {
                    return Ok(());
                }

                if step_finished {
                    return Ok(());
                }
            }

            current_hidden = result.hidden;
            confirmed_token_id = result.next_token_id;
        }
    }

    /// Generate tokens one at a time, sending each via the provided channel.
    ///
    /// If the receiver is dropped (client disconnected), generation stops early.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::significant_drop_tightening
    )]
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
            self.enable_thinking,
            // Non-thinking convenience entry (used by /v1/completions) never
            // streams prefill progress.
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
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<(), EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }
        if max_tokens == 0 {
            let prompt_len = Self::prompt_len(prompt_tokens)?;
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

        with_new_default_stream(Stream::new(), || {
            self.generate_streaming_inner(
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
            )
        })
    }

    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::significant_drop_tightening
    )]
    fn generate_streaming_inner(
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
        mut constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<(), EngineError> {
        let logprob_top_n = logprobs.then(|| top_logprobs.unwrap_or(0));

        let mut prepared = self.prepare_generation(prompt_tokens, pixel_values)?;
        let prompt_len = prepared.prompt_len;
        #[allow(clippy::float_cmp)]
        let capture_mtp_prefill = mtp_prefill_priming_enabled()
            && self.tuning.enable_mtp()
            && prepared.model.has_mtp()
            && prepared.pixel_values.is_none()
            && constraint.is_none()
            && !logprobs
            && params.temperature == 0.0;

        // Stream prefill progress only when the caller opted in (OpenAI
        // `return_progress: true`). Direct higgs-engine callers and BatchEngine
        // keep the progress-free streaming contract: no sink, no extra events.
        //
        // The chunked-prefill loops report suffix-relative completions through a
        // thread-local sink; map them to absolute prompt position by adding the
        // prefix-cache hit. Events ride the normal streaming channel as
        // progress-only outputs. try_send: never stall prefill on a slow
        // consumer — dropped progress events are harmless. The returned guard is
        // held for the prefill's duration and uninstalls the sink on drop.
        let prefill_sink = return_progress.then(|| {
            let actual_tokens =
                u32::try_from(prepared.actual_prompt_tokens.len()).unwrap_or(u32::MAX);
            let cached = prompt_len.saturating_sub(actual_tokens);
            let make_progress_output = move |suffix_done: u32| StreamingOutput {
                new_text: String::new(),
                finished: false,
                finish_reason: None,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
                token_logprob: None,
                prefill_progress: Some(crate::engine::PrefillProgress {
                    processed: (cached + suffix_done).min(prompt_len),
                    cached,
                    total: prompt_len,
                }),
            };
            // Initial event: tells the client the total (and cache hit) before
            // the first ~1024-token chunk completes.
            let _ = sender.try_send(make_progress_output(0));
            let progress_sender = sender.clone();
            higgs_models::progress::install_prefill_progress_sink(Box::new(move |done, _total| {
                let _ = progress_sender.try_send(make_progress_output(
                    u32::try_from(done.max(0)).unwrap_or(0),
                ));
            }))
        });

        let (current_token, first_logprob_data, prefill_hidden) = self.run_prefill(
            prompt_tokens,
            &mut prepared,
            params,
            logprob_top_n,
            constraint.as_ref(),
            capture_mtp_prefill,
        )?;
        // Prefill done — decode must not report progress. Dropping the Option
        // uninstalls the sink when present; a no-op when progress was off.
        drop(prefill_sink);

        let mut all_tokens: Vec<u32> = Vec::new();
        let first_token_id: u32 = current_token.item();
        // Advance the constraint past the first sampled token before decode.
        if let Some(ref mut cg) = constraint {
            cg.advance(first_token_id);
        }
        all_tokens.push(first_token_id);

        let mut detok = IncrementalDetok::new(
            String::new(),
            0,
            std::sync::Arc::clone(&self.decode_skip_ids),
        );
        let first_new_text = detok.append(&self.tokenizer, &all_tokens)?;
        let first_emitted_before = detok.text.len() - first_new_text.len();
        let (mut first_text, first_hit_stop) = if stop_sequences.is_empty() {
            (first_new_text, false)
        } else {
            find_stop_in_tail(&detok.text, first_new_text.len(), stop_sequences).map_or(
                (first_new_text, false),
                |pos| {
                    let emit = detok
                        .text
                        .get(first_emitted_before..pos)
                        .unwrap_or_default()
                        .to_owned();
                    (emit, true)
                },
            )
        };

        let first_is_eos = self.eos_token_ids.contains(&first_token_id);
        let finished = first_is_eos || first_hit_stop || 1 >= max_tokens;

        if finished && !first_hit_stop {
            first_text.push_str(&detok.flush(&self.tokenizer, &all_tokens)?);
        }

        let first_logprob = first_logprob_data
            .as_ref()
            .map(|lp| lp.materialize(first_token_id));

        if sender
            .blocking_send(StreamingOutput {
                new_text: first_text,
                finished,
                finish_reason: if first_is_eos || first_hit_stop {
                    Some("stop".to_owned())
                } else if 1 >= max_tokens {
                    Some("length".to_owned())
                } else {
                    None
                },
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprob: first_logprob,
                prefill_progress: None,
            })
            .is_err()
        {
            return Ok(());
        }

        if finished {
            return Ok(());
        }

        // MTP speculative decode (streaming): greedy, no constraints, no logprobs.
        #[allow(clippy::float_cmp)]
        if self.tuning.enable_mtp()
            && prepared.model.has_mtp()
            && constraint.is_none()
            && !logprobs
            && params.temperature == 0.0
        {
            let actual_prompt_tokens = prepared.actual_prompt_tokens.clone();
            return self.mtp_generate_streaming(
                &mut prepared.model,
                &mut prepared.cache,
                &actual_prompt_tokens,
                prefill_hidden.as_ref(),
                first_token_id,
                max_tokens,
                prompt_len,
                &mut all_tokens,
                stop_sequences,
                sender,
                detok,
                enable_thinking,
            );
        }

        // Thinking budget (streaming): force </think> after N tokens.
        const THINKING_BUDGET: u32 = 256;
        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        // Seed thinking state from the first token (already emitted above).
        let mut thinking_tokens: u32 = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        // Pipelined decode loop: build step N+2 while GPU computes step N+1
        let (mut next_token, mut next_logprob_data) = Self::decode_step(
            &current_token,
            &mut prepared.model,
            &mut prepared.cache,
            params,
            &all_tokens,
            logprob_top_n,
            constraint.as_ref(),
        )?;
        {
            let mut eval_targets: Vec<&Array> = vec![&next_token];
            if let Some(ref lp) = next_logprob_data {
                eval_targets.extend(lp.eval_targets());
            }
            async_eval(eval_targets).map_err(EngineError::Mlx)?;
        }

        loop {
            // When constrained, extract the sampled token and advance the FSM
            // before decode_step, so the mask is always applied at the correct
            // FSM state (mirrors the non-streaming pattern in generate_inner).
            let constrained_token_id: Option<u32> = constraint.is_some().then(|| {
                let id: u32 = next_token.item();
                if let Some(ref mut cg) = constraint {
                    cg.advance(id);
                }
                id
            });

            let (following, following_logprob_data) = Self::decode_step(
                &next_token,
                &mut prepared.model,
                &mut prepared.cache,
                params,
                &all_tokens,
                logprob_top_n,
                constraint.as_ref(),
            )?;
            {
                let mut eval_targets: Vec<&Array> = vec![&following];
                if let Some(ref lp) = following_logprob_data {
                    eval_targets.extend(lp.eval_targets());
                }
                if constraint.is_some() {
                    eval(eval_targets).map_err(EngineError::Mlx)?;
                } else {
                    async_eval(eval_targets).map_err(EngineError::Mlx)?;
                }
            }

            let mut token_id: u32 = constrained_token_id.unwrap_or_else(|| next_token.item());

            // Thinking budget: force </think> after N tokens if model hasn't closed it.
            // NOTE: same KV-cache discontinuity caveat as the non-streaming path.
            if let Some(close_id) = think_close_token {
                if !seen_think_close {
                    if token_id == close_id {
                        seen_think_close = true;
                    } else {
                        thinking_tokens += 1;
                        if thinking_tokens >= THINKING_BUDGET {
                            token_id = close_id;
                            seen_think_close = true;
                            tracing::info!(
                                budget = THINKING_BUDGET,
                                "Thinking budget reached, forcing </think>"
                            );
                        }
                    }
                }
            }

            let token_logprob = next_logprob_data
                .as_ref()
                .map(|lp_data| lp_data.materialize(token_id));

            all_tokens.push(token_id);

            let completion_len = Self::completion_len(&all_tokens)?;

            let new_text = detok.append(&self.tokenizer, &all_tokens)?;
            let emitted_before = detok.text.len() - new_text.len();

            let (mut final_new_text, hit_stop_seq) = if stop_sequences.is_empty() {
                (new_text, false)
            } else {
                find_stop_in_tail(&detok.text, new_text.len(), stop_sequences).map_or(
                    (new_text, false),
                    |pos| {
                        let emit = detok
                            .text
                            .get(emitted_before..pos)
                            .unwrap_or_default()
                            .to_owned();
                        (emit, true)
                    },
                )
            };

            let is_eos = self.eos_token_ids.contains(&token_id);
            let is_max = completion_len >= max_tokens;
            let constraint_done = constraint
                .as_ref()
                .is_some_and(crate::constrained::ConstrainedGenerator::is_finished);
            let step_finished = is_eos || is_max || hit_stop_seq || constraint_done;

            if step_finished && !hit_stop_seq {
                final_new_text.push_str(&detok.flush(&self.tokenizer, &all_tokens)?);
            }

            let finish_reason = if is_eos || hit_stop_seq || constraint_done {
                Some("stop".to_owned())
            } else if is_max {
                Some("length".to_owned())
            } else {
                None
            };

            if sender
                .blocking_send(StreamingOutput {
                    new_text: final_new_text,
                    finished: step_finished,
                    finish_reason,
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprob,
                    prefill_progress: None,
                })
                .is_err()
            {
                return Ok(());
            }

            if step_finished {
                break;
            }

            // If thinking budget was just reached, override the pipelined token
            // so the next decode step gets </think> as input.
            if seen_think_close && thinking_tokens == THINKING_BUDGET {
                if let Some(close_id) = think_close_token {
                    next_token = Array::from_slice(&[close_id], &[1]);
                }
                thinking_tokens += 1; // prevent re-triggering
            } else {
                next_token = following;
            }
            next_logprob_data = following_logprob_data;
        }

        Ok(())
    }
}

/// Cap on the trailing token window re-decoded per streaming step.
/// Generous for any multi-token UTF-8 sequence; prevents a stream of
/// undecodable tokens from growing the window without bound.
const MAX_DETOK_WINDOW: usize = 64;

/// Incremental streaming detokenizer.
///
/// Re-decoding the full completion on every generated token is O(n^2) in
/// completion length. Instead, decode only `tokens[prefix_offset..]` and emit
/// the difference against `tokens[prefix_offset..read_offset]`; both windows
/// start at the same token, so tokenizer boundary effects cancel out. Text
/// ending in an incomplete UTF-8 sequence (trailing replacement char) is held
/// back until a later token completes it.
/// Token IDs that must never surface in decoded output text.
///
/// Seeded with the model's EOS ids, plus every added *control* special token:
/// the `<|…|>` chat-control delimiters and the classic `<s>`/`</s>`/`<pad>`/
/// `<unk>`/`<mask>` sentinels. Content-bearing special tokens (`<think>`,
/// `<tool_call>`, `MiniCPM`'s `<function>`/`<param>`) deliberately do NOT match,
/// so they survive decoding and reach the tool-call / reasoning parsers. Shared
/// by the non-streaming decode path ([`SimpleEngine::decode_tokens`]) and the
/// streaming detokenizer ([`IncrementalDetok`]) so the two strip identically.
pub(crate) fn content_preserving_skip_ids(
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
) -> std::collections::HashSet<u32> {
    let mut ids: std::collections::HashSet<u32> = eos_token_ids.iter().copied().collect();
    for (id, added) in tokenizer.get_added_tokens_decoder() {
        let content = added.content.as_str();
        let is_control = (content.starts_with("<|") && content.ends_with("|>"))
            || matches!(content, "<s>" | "</s>" | "<pad>" | "<unk>" | "<mask>");
        if is_control {
            ids.insert(id);
        }
    }
    ids
}

pub(crate) struct IncrementalDetok {
    /// Start of the decode window; advanced whenever text is emitted.
    prefix_offset: usize,
    /// Number of leading tokens already represented in `text`.
    read_offset: usize,
    /// All text decoded so far (streamed to the client incrementally).
    pub(crate) text: String,
    /// Control-token IDs filtered out before decoding so content-bearing
    /// special tokens (tool-call markup) survive streaming. See
    /// [`content_preserving_skip_ids`].
    skip_ids: std::sync::Arc<std::collections::HashSet<u32>>,
}

impl IncrementalDetok {
    /// Start from text already decoded for the first `token_count` tokens.
    pub(crate) const fn new(
        text: String,
        token_count: usize,
        skip_ids: std::sync::Arc<std::collections::HashSet<u32>>,
    ) -> Self {
        Self {
            prefix_offset: 0,
            read_offset: token_count,
            text,
            skip_ids,
        }
    }

    /// Decode `tokens`, dropping only the control tokens in `skip_ids` while
    /// preserving content-bearing special tokens. Mirrors
    /// [`SimpleEngine::decode_tokens`] so streamed and non-streamed text match.
    fn decode(&self, tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String, EngineError> {
        let run = |ids: &[u32]| {
            tokenizer
                .decode(ids, false)
                .map_err(|e| EngineError::Tokenization(e.to_string()))
        };
        if !tokens.iter().any(|id| self.skip_ids.contains(id)) {
            return run(tokens);
        }
        let filtered: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|id| !self.skip_ids.contains(id))
            .collect();
        run(&filtered)
    }

    /// Decode the trailing window of `tokens`, appending newly stable text to
    /// `self.text` and returning it.
    pub(crate) fn append(
        &mut self,
        tokenizer: &Tokenizer,
        tokens: &[u32],
    ) -> Result<String, EngineError> {
        let prefix_tokens = tokens
            .get(self.prefix_offset..self.read_offset)
            .unwrap_or_default();
        let window_tokens = tokens.get(self.prefix_offset..).unwrap_or_default();
        let prefix_text = self.decode(tokenizer, prefix_tokens)?;
        let window_text = self.decode(tokenizer, window_tokens)?;

        let over_window = window_tokens.len() > MAX_DETOK_WINDOW;
        if window_text.len() > prefix_text.len()
            && (!window_text.ends_with('\u{FFFD}') || over_window)
        {
            let new_text = window_text
                .get(prefix_text.len()..)
                .unwrap_or_default()
                .to_owned();
            self.prefix_offset = self.read_offset;
            self.read_offset = tokens.len();
            self.text.push_str(&new_text);
            return Ok(new_text);
        }
        if over_window && window_text.len() == prefix_text.len() {
            // The pending tokens decode to nothing (e.g. skipped special
            // tokens); drop them so the window stays bounded.
            self.prefix_offset = tokens.len();
            self.read_offset = tokens.len();
        }
        Ok(String::new())
    }

    /// Emit any text still held back by `append` (a trailing incomplete UTF-8
    /// sequence). Called when generation finishes so the total streamed text
    /// matches a full decode of the token buffer.
    pub(crate) fn flush(
        &mut self,
        tokenizer: &Tokenizer,
        tokens: &[u32],
    ) -> Result<String, EngineError> {
        if self.read_offset >= tokens.len() {
            return Ok(String::new());
        }
        let prefix_tokens = tokens
            .get(self.prefix_offset..self.read_offset)
            .unwrap_or_default();
        let window_tokens = tokens.get(self.prefix_offset..).unwrap_or_default();
        let prefix_text = self.decode(tokenizer, prefix_tokens)?;
        let window_text = self.decode(tokenizer, window_tokens)?;
        let new_text = window_text
            .get(prefix_text.len()..)
            .unwrap_or_default()
            .to_owned();
        self.prefix_offset = self.read_offset;
        self.read_offset = tokens.len();
        self.text.push_str(&new_text);
        Ok(new_text)
    }
}

/// Find the earliest stop-sequence occurrence that could involve the newly appended text.
///
/// Scans only the tail of `text` rather than the whole buffer.
/// Returns the absolute byte position where the match starts.
pub(crate) fn find_stop_in_tail(
    text: &str,
    new_len: usize,
    stop_sequences: &[String],
) -> Option<usize> {
    let max_stop = stop_sequences.iter().map(String::len).max().unwrap_or(0);
    if max_stop == 0 {
        return None;
    }
    let mut start = text.len().saturating_sub(new_len + max_stop - 1);
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    let tail = text.get(start..)?;
    let mut earliest: Option<usize> = None;
    for seq in stop_sequences {
        if let Some(pos) = tail.find(seq.as_str()) {
            earliest = Some(earliest.map_or(pos, |prev| prev.min(pos)));
        }
    }
    earliest.map(|pos| start + pos)
}

/// Check if any stop sequence appears in the generated text.
/// Returns `Some(truncated_text)` if a stop sequence was found, None otherwise.
fn check_stop_sequences(text: &str, stop_sequences: &[String]) -> Option<String> {
    let mut earliest: Option<usize> = None;
    for seq in stop_sequences {
        if let Some(pos) = text.find(seq.as_str()) {
            earliest = Some(earliest.map_or(pos, |prev| prev.min(pos)));
        }
    }
    earliest.map(|pos| text.get(..pos).unwrap_or_default().to_owned())
}

/// Derive a human-readable model name from a directory path.
///
/// Detects `HuggingFace` cache paths (`models--<org>--<name>/snapshots/<hash>`)
/// and extracts `<org>/<name>` instead of using the hash as the name.
/// Falls back to the directory's file name.
pub(crate) fn derive_model_name(model_dir: &Path) -> String {
    // HuggingFace cache: .../models--<org>--<name>/snapshots/<hash>
    if let (Some(leaf), Some(parent)) = (model_dir.file_name(), model_dir.parent()) {
        let leaf_str = leaf.to_string_lossy();
        if let (Some(snapshots), Some(grandparent)) = (parent.file_name(), parent.parent()) {
            if snapshots.to_string_lossy() == "snapshots" {
                let gp_name = grandparent
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default();
                if let Some(rest) = gp_name.strip_prefix("models--") {
                    // "org--model-name" -> "org/model-name"
                    if let Some(sep) = rest.find("--") {
                        let org = &rest[..sep];
                        let model = &rest[sep + 2..];
                        return format!("{org}/{model}");
                    }
                    return rest.to_owned();
                }
            }
        }
        // Not an HF cache path -- use the leaf directory name
        if !leaf_str.is_empty() {
            return leaf_str.to_string();
        }
    }
    "unknown".to_owned()
}

/// Extract EOS token IDs from config.json.
pub(crate) fn extract_eos_tokens(model_dir: &Path) -> Vec<u32> {
    let config_path = model_dir.join("config.json");
    let config_str = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(path = %config_path.display(), error = %e, "Could not read config.json for EOS tokens");
            return vec![];
        }
    };

    let config: serde_json::Value = match serde_json::from_str(&config_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "Could not parse config.json for EOS tokens");
            return vec![];
        }
    };

    // Check top-level first, then text_config (VLM/Qwen3.5 nested config).
    // Filter null so explicit `"eos_token_id": null` falls through to text_config.
    let eos_value = config
        .get("eos_token_id")
        .filter(|v| !v.is_null())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("eos_token_id"))
        });

    let mut ids = eos_ids_from_value(eos_value);

    // Union eos_token_id from generation_config.json — HF applies it at inference
    // and it often lists turn-end tokens the base config.json omits.
    if let Ok(gen_str) = std::fs::read_to_string(model_dir.join("generation_config.json"))
        && let Ok(gen_cfg) = serde_json::from_str::<serde_json::Value>(&gen_str)
    {
        for id in eos_ids_from_value(gen_cfg.get("eos_token_id")) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }

    // Gemma chat models end assistant turns with <end_of_turn>, which their
    // config.json typically leaves out of eos_token_id. Without it generation runs
    // past the answer into filler. Add it for the Gemma family.
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("model_type"))
                .and_then(|v| v.as_str())
        });
    if model_type.is_some_and(|t| t.starts_with("gemma"))
        && let Some(id) = special_token_id(model_dir, "<end_of_turn>")
        && !ids.contains(&id)
    {
        ids.push(id);
    }

    if ids.is_empty() {
        tracing::warn!("No eos_token_id found in config.json, generation will rely on max_tokens");
    }
    ids
}

/// Parse an `eos_token_id` JSON value (a single number or an array of numbers).
fn eos_ids_from_value(value: Option<&serde_json::Value>) -> Vec<u32> {
    match value {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .map_or_else(Vec::new, |id| vec![id]),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_u64().and_then(|val| u32::try_from(val).ok()))
            .collect(),
        _ => vec![],
    }
}

/// Resolve the id of an added special token by its `content` string, reading
/// `tokenizer_config.json`'s `added_tokens_decoder` map.
fn special_token_id(model_dir: &Path, content: &str) -> Option<u32> {
    let text = std::fs::read_to_string(model_dir.join("tokenizer_config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    let decoder = config.get("added_tokens_decoder")?.as_object()?;
    decoder.iter().find_map(|(id, info)| {
        (info.get("content").and_then(serde_json::Value::as_str) == Some(content))
            .then(|| id.parse::<u32>().ok())
            .flatten()
    })
}

/// Detect whether a model supports thinking mode based on `model_type`.
/// Whether the model *supports* a thinking toggle (capability, not default).
///
/// The per-request default — e.g. Qwen3.6 reasons off unless asked — is decided
/// separately by `model_defaults_to_non_thinking` in the router; this only
/// answers "can it think at all".
///
/// Signals, in order:
/// 1. the chat template exposes an `enable_thinking` switch — the model
///    author's own marker, which covers Qwen3.5/3.6, `MiniCPM5`, and future
///    reasoning models without hardcoding model types; or
/// 2. a known reasoning `model_type`.
///
/// The caller additionally requires a single-token `</think>`, so a stray
/// mention can't enable thinking for a model that wasn't trained for it.
fn detect_thinking_support(model_dir: &Path) -> bool {
    if chat_template_mentions_enable_thinking(model_dir) {
        return true;
    }
    let Ok(config_str) = std::fs::read_to_string(model_dir.join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) else {
        return false;
    };
    // Check both top-level and nested text_config (VLM wrappers).
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("model_type"))
                .and_then(|v| v.as_str())
        });
    matches!(model_type, Some("qwen3_5" | "qwen3_5_moe"))
}

/// Whether the model's chat template references the `enable_thinking` toggle,
/// read from `chat_template.jinja` or `tokenizer_config.json`'s `chat_template`
/// (a string, or a `{name, template}` array).
fn chat_template_mentions_enable_thinking(model_dir: &Path) -> bool {
    const MARKER: &str = "enable_thinking";
    if let Ok(jinja) = std::fs::read_to_string(model_dir.join("chat_template.jinja")) {
        return jinja.contains(MARKER);
    }
    let Ok(cfg_str) = std::fs::read_to_string(model_dir.join("tokenizer_config.json")) else {
        return false;
    };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&cfg_str) else {
        return false;
    };
    let template = cfg.get("chat_template");
    if let Some(s) = template.and_then(|v| v.as_str()) {
        return s.contains(MARKER);
    }
    if let Some(arr) = template.and_then(|v| v.as_array()) {
        return arr.iter().any(|entry| {
            entry
                .get("template")
                .and_then(|t| t.as_str())
                .is_some_and(|t| t.contains(MARKER))
        });
    }
    false
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::{
        IncrementalDetok, Tokenizer, adaptive_draft_depth_for_cap, check_stop_sequences,
        derive_model_name, detect_thinking_support, estimate_paged_kv_blocks, extract_eos_tokens,
        find_stop_in_tail, parse_enabled_flag,
    };
    use std::path::Path;

    /// Write a config.json file into the given directory with the provided JSON content.
    fn write_config(dir: &std::path::Path, json: &str) {
        std::fs::write(dir.join("config.json"), json).unwrap();
    }

    /// P5: `DFlash` speculative decode must be byte-identical to plain AR greedy
    /// decode (T=0) and reports acceptance length + tok/s. Loads the real 35B
    /// target twice (AR then `DFlash`, sequentially to bound memory). Manual:
    ///
    /// ```text
    /// HIGGS_DFLASH_TARGET_DIR=<target dir> \
    /// HIGGS_DFLASH_DRAFTER_DIR=<modal drafter snapshot dir> \
    /// cargo test -p higgs-engine dflash_matches_ar_greedy -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "p5: loads real 35B target; set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"]
    fn dflash_matches_ar_greedy_and_reports_accept_len() {
        use super::SimpleEngine;
        use crate::chat_template::ChatMessage;
        use crate::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
        use higgs_models::{SamplingParams, turboquant::KvCacheConfig};

        // Surface the engine's accept_len / per-iter trace logs.
        let _ = tracing_subscriber::fmt()
            .with_env_filter("info")
            .with_test_writer()
            .try_init();

        let target = std::env::var("HIGGS_DFLASH_TARGET_DIR")
            .expect("set HIGGS_DFLASH_TARGET_DIR to the 35B target model dir");
        let drafter = std::env::var("HIGGS_DFLASH_DRAFTER_DIR")
            .expect("set HIGGS_DFLASH_DRAFTER_DIR to the Modal drafter snapshot dir");

        // Match the reference benchmark workload: chat template + thinking
        // enabled, a code/reasoning prompt (DFlash is trained/measured on
        // GSM8K/HumanEval/MT-Bench, not raw factual completions). Keep
        // max_tokens under the 256-token thinking budget so the AR (MTP) path
        // never force-closes </think> while DFlash wouldn't.
        let user_prompt = std::env::var("HIGGS_TEST_PROMPT").unwrap_or_else(|_| {
            "Write a Python function that returns the n-th Fibonacci number iteratively.".to_owned()
        });
        let user_prompt = user_prompt.as_str();
        // Toggle with HIGGS_TEST_THINKING=0 to measure the thinking-disabled
        // workload (default: enabled, matching the reference benchmark).
        let enable_thinking = std::env::var("HIGGS_TEST_THINKING")
            .map(|v| v != "0")
            .unwrap_or(true);
        let max_tokens = 200u32;
        let greedy = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };

        let run = |with_drafter: bool| -> (String, f64) {
            let drafter_path = with_drafter.then(|| Path::new(&drafter));
            let tuning =
                MlxRuntimeTuning::from_model_dir(Path::new(&target), RequestedMlxProfile::Auto);
            let engine = SimpleEngine::load_with_dflash(
                &target,
                KvCacheConfig::default(),
                tuning,
                false,
                drafter_path,
            )
            .expect("load target");
            let messages = [ChatMessage {
                role: "user".to_owned(),
                content: user_prompt.to_owned(),
                tool_calls: None,
            }];
            let toks = engine
                .prepare_chat_prompt_with_thinking(&messages, None, enable_thinking)
                .expect("chat prompt");
            let t = std::time::Instant::now();
            let out = engine
                .generate_with_thinking(
                    &toks,
                    max_tokens,
                    &greedy,
                    &[],
                    false,
                    None,
                    enable_thinking,
                    None,
                    None,
                )
                .expect("generate");
            let tps = f64::from(out.completion_tokens) / t.elapsed().as_secs_f64();
            (out.text, tps)
        };

        let (ar_text, ar_tps) = run(false);
        let (df_text, df_tps) = run(true);

        println!("AR     {ar_tps:.1} tok/s: {ar_text:?}");
        println!("DFLASH {df_tps:.1} tok/s: {df_text:?}");
        println!("speedup vs AR: {:.2}x", df_tps / ar_tps);

        // Correctness gate: greedy outputs identical up to a block-overshoot tail
        // (DFlash commits whole blocks, so it may emit a few extra trailing
        // tokens past max_tokens — the shorter must be a prefix of the longer).
        let (short, long) = if ar_text.len() <= df_text.len() {
            (ar_text.as_str(), df_text.as_str())
        } else {
            (df_text.as_str(), ar_text.as_str())
        };
        assert!(!short.is_empty(), "empty generation");
        assert!(
            long.starts_with(short),
            "DFlash diverged from AR greedy:\n AR={ar_text:?}\n DF={df_text:?}"
        );
    }

    /// Isolates the per-step cost the gate's AR floor pays for taps. Floored
    /// steps now call `forward` (tap-less) instead of `forward_with_taps`; this
    /// interleaves the two over many single-token decodes so thermal drift
    /// averages out (the end-to-end ratio is hopelessly thermally confounded on
    /// this serial-load harness) and prints mean µs/token for each. If the delta
    /// is ~0 the tap clones are lazy/dropped and the fast-decode path is a no-op;
    /// a real delta is the prose-parity win. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "p5: loads real 35B target; set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn dflash_floor_tapless_vs_taps_per_step_cost() {
        use super::{SimpleEngine, lock_or_recover};
        use crate::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
        use higgs_models::turboquant::KvCacheConfig;
        use mlx_rs::{Array, ops::indexing::IndexOp, transforms::eval};

        let Ok(target) = std::env::var("HIGGS_DFLASH_TARGET_DIR") else {
            eprintln!("skip: set HIGGS_DFLASH_TARGET_DIR");
            return;
        };
        let drafter =
            std::env::var("HIGGS_DFLASH_DRAFTER_DIR").expect("set HIGGS_DFLASH_DRAFTER_DIR");
        let tuning =
            MlxRuntimeTuning::from_model_dir(Path::new(&target), RequestedMlxProfile::Auto);
        let engine = SimpleEngine::load_with_dflash(
            &target,
            KvCacheConfig::default(),
            tuning,
            false,
            Some(Path::new(&drafter)),
        )
        .expect("load target + drafter");
        let tap_layers = engine.dflash.as_ref().expect("dflash").tap_layers.clone();
        let cfg = engine.kv_cache_config;
        let mut model = lock_or_recover(&engine.model);
        let mut cache = model.make_cache_with_config(cfg).expect("cache");

        // Prefill so decode steps run at a realistic cache offset.
        let prompt: Vec<i32> = (1..=32).collect();
        let parr = Array::from_slice(&prompt, &[1, prompt.len() as i32]);
        let (logits, _t) = model
            .forward_with_taps(&parr, None, &mut cache, &tap_layers)
            .expect("prefill");
        let am = mlx_rs::argmax_axis!(logits.index((.., -1, ..)), -1).expect("argmax");
        eval([&am]).expect("eval");
        let mut next: i32 = i32::try_from(am.item::<u32>()).expect("tok");

        let (warm, iters) = (8u32, 160u32);
        let (mut t_taps, mut n_taps, mut t_plain, mut n_plain) = (0f64, 0u32, 0f64, 0u32);
        for i in 0..(warm + iters) {
            let single = Array::from_slice(&[next], &[1, 1]);
            let use_taps = i % 2 == 0;
            let start = std::time::Instant::now();
            let logits = if use_taps {
                model
                    .forward_with_taps(&single, None, &mut cache, &tap_layers)
                    .expect("taps")
                    .0
            } else {
                model.forward(&single, None, &mut cache).expect("plain")
            };
            let am = mlx_rs::argmax_axis!(logits.index((.., -1, ..)), -1).expect("argmax");
            eval([&am]).expect("eval");
            let dt = start.elapsed().as_secs_f64();
            next = i32::try_from(am.item::<u32>()).expect("tok");
            if i >= warm {
                if use_taps {
                    t_taps += dt;
                    n_taps += 1;
                } else {
                    t_plain += dt;
                    n_plain += 1;
                }
            }
        }
        let us_taps = t_taps / f64::from(n_taps) * 1e6;
        let us_plain = t_plain / f64::from(n_plain) * 1e6;
        eprintln!(
            "floor step cost: forward_with_taps {us_taps:.0} µs/tok | forward(tap-less) {us_plain:.0} µs/tok | taps overhead {:.1}% ({n_taps} taps / {n_plain} plain)",
            (us_taps - us_plain) / us_taps * 100.0
        );
    }

    /// Localizes the pre-existing DFlash-vs-greedy prose divergence. The spec
    /// round commits exactly the verify forward's per-position argmax
    /// (`accept_prefix` is textbook-correct), so any divergence from AR greedy is
    /// `verify_argmax[j] != ar_argmax[j]` — the S>1 batched verify flips an
    /// argmax vs S=1 sequential AR. The GDN tape makes the SSM layers bit-exact,
    /// but the full-attention layers run batched in verify and drift on near-ties.
    /// Feeds the AR-correct tokens through the real verify path and reports which
    /// positions flip + the top-2 logit gap there (small gap == near-tie ==
    /// numerical, not a logic bug). Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "p5: loads real 35B target; set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn dflash_verify_vs_ar_argmax_divergence() {
        use super::{SimpleEngine, lock_or_recover};
        use crate::chat_template::ChatMessage;
        use crate::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
        use higgs_models::turboquant::KvCacheConfig;
        use mlx_rs::{Array, Dtype, ops::indexing::IndexOp, transforms::eval};

        let Ok(target) = std::env::var("HIGGS_DFLASH_TARGET_DIR") else {
            eprintln!("skip: set HIGGS_DFLASH_TARGET_DIR");
            return;
        };
        let drafter =
            std::env::var("HIGGS_DFLASH_DRAFTER_DIR").expect("set HIGGS_DFLASH_DRAFTER_DIR");
        let tuning =
            MlxRuntimeTuning::from_model_dir(Path::new(&target), RequestedMlxProfile::Auto);
        let engine = SimpleEngine::load_with_dflash(
            &target,
            KvCacheConfig::default(),
            tuning,
            false,
            Some(Path::new(&drafter)),
        )
        .expect("load target + drafter");

        let prompt = std::env::var("HIGGS_TEST_PROMPT").unwrap_or_else(|_| {
            "Write several paragraphs about the history and cultural significance of tea across different civilizations.".to_owned()
        });
        let messages = [ChatMessage {
            role: "user".to_owned(),
            content: prompt,
            tool_calls: None,
        }];
        let prompt_ids = engine
            .prepare_chat_prompt_with_thinking(&messages, None, false)
            .expect("chat prompt");
        let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&u| u as i32).collect();

        let tap_layers = engine.dflash.as_ref().expect("dflash").tap_layers.clone();
        let cfg = engine.kv_cache_config;
        let mut model = lock_or_recover(&engine.model);

        // top-1 token + (top1 - top2) logit gap from a [.., 1, vocab] tensor.
        let argmax_gap = |logits: &Array| -> (u32, f32) {
            let row = logits
                .index((.., -1, ..))
                .reshape(&[-1])
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            eval([&row]).unwrap();
            let h: Vec<f32> = row.as_slice::<f32>().to_vec();
            let (mut i1, mut v1) = (0usize, f32::MIN);
            for (i, &v) in h.iter().enumerate() {
                if v > v1 {
                    v1 = v;
                    i1 = i;
                }
            }
            let mut v2 = f32::MIN;
            for (i, &v) in h.iter().enumerate() {
                if i != i1 && v > v2 {
                    v2 = v;
                }
            }
            (i1 as u32, v1 - v2)
        };

        let k = 48usize;
        let parr = Array::from_slice(&prompt_i32, &[1, prompt_i32.len() as i32]);

        // Ground-truth sequential AR greedy: ar_seq[0..=k]; ar_gap[i] is the
        // top-2 gap of the forward that predicts ar_seq[i+1].
        let mut cache_ar = model.make_cache_with_config(cfg).expect("cache");
        let l0 = model
            .forward(&parr, None, &mut cache_ar)
            .expect("prefill ar");
        let (mut tok, _g0) = argmax_gap(&l0);
        let mut ar_seq = vec![tok];
        let mut ar_gap = Vec::with_capacity(k);
        for _ in 0..k {
            let single = Array::from_slice(&[tok as i32], &[1, 1]);
            let l = model
                .forward(&single, None, &mut cache_ar)
                .expect("ar step");
            let (t, g) = argmax_gap(&l);
            ar_seq.push(t);
            ar_gap.push(g);
            tok = t;
        }

        // Verify path on a fresh cache: prefill the prompt, then ONE batched
        // verify forward over the AR-correct tokens (anchor + ar_seq[0..k-1]),
        // exactly as a fully-accepted spec round would see them.
        let mut cache_v = model.make_cache_with_config(cfg).expect("cache");
        let _ = model.forward(&parr, None, &mut cache_v).expect("prefill v");
        let verify_in: Vec<i32> = ar_seq[..k].iter().map(|&u| u as i32).collect();
        let vin = Array::from_slice(&verify_in, &[1, k as i32]);
        let (vlogits, _taps, _tapes) = model
            .forward_with_taps_tape(&vin, None, &mut cache_v, &tap_layers)
            .expect("verify");
        let vargmax = mlx_rs::argmax_axis!(vlogits, -1).expect("argmax");
        eval([&vargmax]).expect("eval");
        let vflat: Vec<u32> = vargmax
            .reshape(&[-1])
            .expect("reshape")
            .as_slice::<u32>()
            .to_vec();

        // verify position i predicts the token after ar_seq[i] -> compare ar_seq[i+1].
        let mut nmis = 0u32;
        let mut first = None;
        let (mut min_gap_mis, mut min_gap_match) = (f32::MAX, f32::MAX);
        for i in 0..k {
            let ar_next = ar_seq[i + 1];
            if vflat[i] == ar_next {
                min_gap_match = min_gap_match.min(ar_gap[i]);
            } else {
                nmis += 1;
                if first.is_none() {
                    first = Some(i);
                }
                min_gap_mis = min_gap_mis.min(ar_gap[i]);
                eprintln!(
                    "MISMATCH pos {i}: verify={} ar={ar_next} ar_top2_gap={:.4}",
                    vflat[i], ar_gap[i]
                );
            }
        }
        eprintln!(
            "verify-vs-AR over {k} positions: {nmis} mismatches, first at {first:?}; min top2-gap at mismatch={min_gap_mis:.4} vs match={min_gap_match:.4}"
        );
    }

    /// Gemma chat models stop on `<end_of_turn>`, which their `config.json` often
    /// omits from `eos_token_id`. It must be resolved from the tokenizer and added.
    #[test]
    fn extract_eos_tokens_adds_gemma_end_of_turn() {
        let dir = tempfile::tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{"model_type":"gemma3_text","eos_token_id":1}"#,
        );
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder":{"106":{"content":"<end_of_turn>"}}}"#,
        )
        .unwrap();
        let ids = extract_eos_tokens(dir.path());
        assert!(ids.contains(&1), "base eos retained: {ids:?}");
        assert!(ids.contains(&106), "gemma <end_of_turn> added: {ids:?}");
    }

    /// `generation_config.json`'s `eos_token_id` (used by HF at inference) is unioned in.
    #[test]
    fn extract_eos_tokens_unions_generation_config() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type":"llama","eos_token_id":2}"#);
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"eos_token_id":[2,7]}"#,
        )
        .unwrap();
        let ids = extract_eos_tokens(dir.path());
        assert!(
            ids.contains(&2) && ids.contains(&7),
            "union with generation_config: {ids:?}"
        );
    }

    /// Non-Gemma models must not pull in `<end_of_turn>`.
    #[test]
    fn extract_eos_tokens_non_gemma_ignores_end_of_turn() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type":"llama","eos_token_id":2}"#);
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder":{"106":{"content":"<end_of_turn>"}}}"#,
        )
        .unwrap();
        let ids = extract_eos_tokens(dir.path());
        assert_eq!(ids, vec![2], "non-gemma must not add end_of_turn: {ids:?}");
    }

    // --- detect_thinking_support tests ---

    /// MiniCPM5-style: a non-reasoning `model_type` (llama) but a chat template
    /// that exposes the `enable_thinking` switch ⇒ thinking-capable.
    #[test]
    fn detect_thinking_from_template_marker() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type": "llama"}"#);
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{%- if enable_thinking %}<think>\n{%- endif %}",
        )
        .unwrap();
        assert!(detect_thinking_support(dir.path()));
    }

    /// Qwen3.5 reasoning `model_type` is detected even without a template file.
    #[test]
    fn detect_thinking_from_model_type() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type": "qwen3_5_moe"}"#);
        assert!(detect_thinking_support(dir.path()));
    }

    /// A plain Llama (no reasoning `model_type`, no `enable_thinking` in the
    /// template) must NOT be treated as a thinking model.
    #[test]
    fn no_thinking_for_plain_llama() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type": "llama"}"#);
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{%- for m in messages %}{{ m.content }}{%- endfor %}",
        )
        .unwrap();
        assert!(!detect_thinking_support(dir.path()));
    }

    // --- derive_model_name tests ---

    #[test]
    fn test_derive_model_name_plain_directory() {
        let name = derive_model_name(Path::new("/home/user/models/Llama-3.2-1B"));
        assert_eq!(name, "Llama-3.2-1B");
    }

    #[test]
    fn test_derive_model_name_hf_cache_path() {
        let path = "/Users/me/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5";
        let name = derive_model_name(Path::new(path));
        assert_eq!(name, "mlx-community/Qwen3-Coder-Next-4bit");
    }

    #[test]
    fn test_derive_model_name_hf_cache_no_org() {
        let path = "/cache/models--MyModel/snapshots/abc123";
        let name = derive_model_name(Path::new(path));
        assert_eq!(name, "MyModel");
    }

    #[test]
    fn test_derive_model_name_relative_path() {
        let name = derive_model_name(Path::new("./my-model"));
        assert_eq!(name, "my-model");
    }

    #[test]
    fn adaptive_draft_depth_respects_configured_cap() {
        let mut depth = adaptive_draft_depth_for_cap(1);

        depth.observe(1, 1);

        assert_eq!(depth.current(), 1);
    }

    /// Create a temp dir, write config.json with the given content, and return
    /// the result of `extract_eos_tokens`.
    fn eos_from_config(json: &str) -> Vec<u32> {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), json);
        super::extract_eos_tokens(dir.path())
    }

    #[test]
    fn test_single_stop_sequence_found() {
        let result = check_stop_sequences("Hello world, goodbye!", &["goodbye".to_owned()]);
        assert_eq!(result, Some("Hello world, ".to_owned()));
    }

    #[test]
    fn test_no_stop_sequence_match() {
        let stops = vec!["goodbye".to_owned(), "farewell".to_owned()];
        assert!(check_stop_sequences("Hello world", &stops).is_none());
    }

    #[test]
    fn test_empty_stop_sequences_list() {
        assert!(check_stop_sequences("Hello world", &[]).is_none());
    }

    #[test]
    fn test_empty_text() {
        assert!(check_stop_sequences("", &["hello".to_owned()]).is_none());
    }

    #[test]
    fn test_stop_sequence_at_beginning() {
        let result = check_stop_sequences("STOP rest of text", &["STOP".to_owned()]);
        assert_eq!(result, Some(String::new()));
    }

    #[test]
    fn test_stop_sequence_at_end() {
        let result = check_stop_sequences("Hello world END", &["END".to_owned()]);
        assert_eq!(result, Some("Hello world ".to_owned()));
    }

    fn assert_stop_sequence(text: &str, stops: &[&str], expected: &str) {
        let owned_stops: Vec<String> = stops.iter().map(|s| (*s).to_owned()).collect();
        let result = check_stop_sequences(text, &owned_stops);
        assert_eq!(result, Some(expected.to_owned()));
    }

    #[test]
    fn test_multiple_stop_sequences_earliest_wins() {
        assert_stop_sequence("aaa bbb ccc ddd", &["ccc", "bbb"], "aaa ");
    }

    #[test]
    fn test_multiple_stop_sequences_earliest_wins_reverse_order() {
        assert_stop_sequence("aaa bbb ccc ddd", &["bbb", "ccc"], "aaa ");
    }

    #[test]
    fn test_overlapping_stop_sequences_prefix() {
        // "ab" is a prefix of "abc". "ab" appears first at position 0.
        let stops = vec!["abc".to_owned(), "ab".to_owned()];
        assert_eq!(check_stop_sequences("abc def", &stops), Some(String::new()));
    }

    #[test]
    fn test_stop_sequence_appears_multiple_times() {
        let result = check_stop_sequences("before stop middle stop after", &["stop".to_owned()]);
        assert_eq!(result, Some("before ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_is_entire_text() {
        assert_eq!(
            check_stop_sequences("STOP", &["STOP".to_owned()]),
            Some(String::new())
        );
    }

    #[test]
    fn test_stop_sequence_with_newlines() {
        let result = check_stop_sequences("line one\nline two\nline three", &["\n".to_owned()]);
        assert_eq!(result, Some("line one".to_owned()));
    }

    #[test]
    fn test_extract_eos_tokens_single_number() {
        assert_eq!(
            eos_from_config(r#"{"eos_token_id": 151643}"#),
            vec![151_643]
        );
    }

    #[test]
    fn test_extract_eos_tokens_array() {
        assert_eq!(
            eos_from_config(r#"{"eos_token_id": [151643, 151645]}"#),
            vec![151_643, 151_645]
        );
    }

    #[test]
    fn test_extract_eos_tokens_missing_field() {
        assert!(eos_from_config(r#"{"model_type": "qwen2"}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_unexpected_type() {
        assert!(eos_from_config(r#"{"eos_token_id": "string"}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_missing_config_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(super::extract_eos_tokens(dir.path()).is_empty());
    }

    // -- Additional check_stop_sequences edge cases --

    #[test]
    fn test_stop_sequence_substring_of_another() {
        assert_stop_sequence("Hello stop_now world", &["stop_now", "stop"], "Hello ");
    }

    #[test]
    fn test_stop_sequence_unicode() {
        let stops = vec!["\u{1F600}".to_owned()];
        assert!(check_stop_sequences("Hello world, a]b stop here", &stops).is_none());

        let result = check_stop_sequences("Hello \u{1F600} world", &stops);
        assert_eq!(result, Some("Hello ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_unicode_multibyte() {
        let stops = vec!["arr\u{00EA}t".to_owned()];
        let result = check_stop_sequences("Bonjour le monde, arr\u{00EA}t ici", &stops);
        assert_eq!(result, Some("Bonjour le monde, ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_very_long_text_short_stop() {
        let long_text = format!("{}STOP{}", "a".repeat(10_000), "b".repeat(5_000));
        let result = check_stop_sequences(&long_text, &["STOP".to_owned()]);
        assert_eq!(result, Some("a".repeat(10_000)));
    }

    // -- Additional extract_eos_tokens edge cases --

    #[test]
    fn test_extract_eos_tokens_float_value() {
        // serde_json parses 151643.0 as a float, and as_u64() returns None for floats
        assert!(eos_from_config(r#"{"eos_token_id": 151643.0}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_string_value() {
        assert!(eos_from_config(r#"{"eos_token_id": "not_a_number"}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_nested_array() {
        // Inner arrays are not numbers, so as_u64() returns None for them
        assert!(eos_from_config(r#"{"eos_token_id": [[1, 2], [3, 4]]}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_negative_number() {
        // as_u64() returns None for negative numbers
        assert!(eos_from_config(r#"{"eos_token_id": -1}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_very_large_number() {
        // u32::MAX is 4294967295; as_u64() succeeds but u32::try_from fails
        assert!(eos_from_config(r#"{"eos_token_id": 4294967296}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_empty_array() {
        assert!(eos_from_config(r#"{"eos_token_id": []}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_mixed_types_in_array() {
        // Only numeric entries are extracted; "two" is skipped
        assert_eq!(
            eos_from_config(r#"{"eos_token_id": [1, "two", 3]}"#),
            vec![1, 3]
        );
    }

    #[test]
    fn test_estimate_paged_kv_blocks_clamps_to_minimum() {
        assert_eq!(estimate_paged_kv_blocks(512, 512, 512, 64), 256);
    }

    #[test]
    fn test_estimate_paged_kv_blocks_clamps_to_maximum() {
        assert_eq!(estimate_paged_kv_blocks(usize::MAX, 1, 1, 1), 4096);
    }

    #[test]
    fn test_estimate_paged_kv_blocks_scales_with_geometry() {
        assert_eq!(
            estimate_paged_kv_blocks(512 * 1024 * 1024, 8, 128, 64),
            2048
        );
    }

    #[test]
    fn test_parse_enabled_flag_accepts_common_truthy_values() {
        assert_eq!(parse_enabled_flag(Some("1")), Some(true));
        assert_eq!(parse_enabled_flag(Some("true")), Some(true));
        assert_eq!(parse_enabled_flag(Some("On")), Some(true));
        assert_eq!(parse_enabled_flag(Some("yes")), Some(true));
        assert_eq!(parse_enabled_flag(None), None);
        assert_eq!(parse_enabled_flag(Some("0")), Some(false));
        assert_eq!(parse_enabled_flag(Some("false")), Some(false));
        assert_eq!(parse_enabled_flag(Some("off")), Some(false));
        assert_eq!(parse_enabled_flag(Some("no")), Some(false));
        assert_eq!(parse_enabled_flag(Some("unexpected")), None);
    }

    // -----------------------------------------------------------------------
    // find_stop_in_tail
    // -----------------------------------------------------------------------

    #[test]
    fn find_stop_in_tail_empty_stops_returns_none() {
        assert_eq!(find_stop_in_tail("hello world", 5, &[]), None);
    }

    #[test]
    fn find_stop_in_tail_finds_stop_in_new_text() {
        let stops = vec!["</s>".to_owned()];
        // "hello</s>" with the last 4 bytes new
        assert_eq!(find_stop_in_tail("hello</s>", 4, &stops), Some(5));
    }

    #[test]
    fn find_stop_in_tail_finds_stop_spanning_boundary() {
        let stops = vec!["STOP".to_owned()];
        // "abcSTOP" where only "OP" is new; "ST" was emitted previously
        assert_eq!(find_stop_in_tail("abcSTOP", 2, &stops), Some(3));
    }

    #[test]
    fn find_stop_in_tail_ignores_stop_fully_in_old_text() {
        let stops = vec!["XY".to_owned()];
        // "XYabcdefgh" with 2 new bytes: the scan window covers the last
        // 2 + (2 - 1) = 3 bytes only, so the old "XY" is not rescanned.
        assert_eq!(find_stop_in_tail("XYabcdefgh", 2, &stops), None);
    }

    #[test]
    fn find_stop_in_tail_earliest_of_multiple() {
        let stops = vec!["BBB".to_owned(), "AAA".to_owned()];
        assert_eq!(find_stop_in_tail("xAAAyBBBz", 9, &stops), Some(1));
    }

    #[test]
    fn find_stop_in_tail_handles_multibyte_boundary() {
        let stops = vec!["端".to_owned()];
        // The tail start can land mid-codepoint; it must back up to a
        // char boundary instead of panicking. "日本語端" = 4 chars, 12 bytes.
        assert_eq!(find_stop_in_tail("日本語端", 3, &stops), Some(9));
    }

    // -----------------------------------------------------------------------
    // IncrementalDetok
    // -----------------------------------------------------------------------

    /// Minimal word-level tokenizer with a byte-level decoder for detok tests.
    fn word_tokenizer() -> Tokenizer {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "WordLevel",
                "vocab": {"Hello": 0, "Ġworld": 1, "!": 2, "ðŁĺ": 3, "Ģ": 4},
                "unk_token": "Hello"
            }
        }"#;
        Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    /// Empty skip-id set for detok tests that exercise plain (non-special)
    /// tokens, where `skip_special_tokens` makes no difference.
    fn no_skip() -> std::sync::Arc<std::collections::HashSet<u32>> {
        std::sync::Arc::new(std::collections::HashSet::new())
    }

    /// `content_preserving_skip_ids` strips control tokens (chat delimiters,
    /// sentinels, the explicit EOS list) but keeps content-bearing special
    /// tokens so tool-call markup reaches the parser.
    #[test]
    fn skip_ids_strip_control_but_keep_content_special_tokens() {
        let mut tok = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let _ = tok.add_special_tokens([
            tokenizers::AddedToken::from("<|im_end|>", true),
            tokenizers::AddedToken::from("</s>", true),
            tokenizers::AddedToken::from("<function>", true),
            tokenizers::AddedToken::from("<tool_call>", true),
        ]);
        let im_end = tok.token_to_id("<|im_end|>").unwrap();
        let skip = super::content_preserving_skip_ids(&tok, &[im_end, 7]);

        // Stripped: <|…|> delimiter, the </s> sentinel, and the explicit eos id.
        assert!(skip.contains(&im_end), "<|…|> delimiter is a control token");
        assert!(
            skip.contains(&tok.token_to_id("</s>").unwrap()),
            "</s> sentinel stripped"
        );
        assert!(skip.contains(&7), "explicit eos id retained");
        // Preserved: content-bearing tool-call markup the parser depends on.
        assert!(
            !skip.contains(&tok.token_to_id("<function>").unwrap()),
            "<function> preserved for the tool-call parser"
        );
        assert!(
            !skip.contains(&tok.token_to_id("<tool_call>").unwrap()),
            "<tool_call> preserved"
        );
    }

    #[test]
    fn incremental_detok_emits_per_token_diffs() {
        let tokenizer = word_tokenizer();
        let mut tokens: Vec<u32> = vec![0];
        let first = tokenizer.decode(&tokens, true).unwrap();
        let mut detok = IncrementalDetok::new(first.clone(), tokens.len(), no_skip());

        tokens.push(1);
        let second = detok.append(&tokenizer, &tokens).unwrap();
        tokens.push(2);
        let third = detok.append(&tokenizer, &tokens).unwrap();

        let full = tokenizer.decode(&tokens, true).unwrap();
        assert_eq!(format!("{first}{second}{third}"), full);
        assert_eq!(detok.text, full);
    }

    #[test]
    fn incremental_detok_flush_without_pending_is_empty() {
        let tokenizer = word_tokenizer();
        let mut tokens: Vec<u32> = vec![0];
        let first = tokenizer.decode(&tokens, true).unwrap();
        let mut detok = IncrementalDetok::new(first, tokens.len(), no_skip());

        tokens.push(1);
        detok.append(&tokenizer, &tokens).unwrap();
        assert_eq!(detok.flush(&tokenizer, &tokens).unwrap(), "");
    }

    // Tokens 3 and 4 are the byte-level pieces of 😀 (U+1F600).
    // Appending only token 3 must hold back the incomplete UTF-8 sequence
    // (return ""), and appending both tokens must emit the full emoji.
    #[test]
    fn incremental_detok_first_token_partial_utf8_held_back() {
        let tokenizer = word_tokenizer();
        let mut detok = IncrementalDetok::new(String::new(), 0, no_skip());

        // First partial piece: held back, no replacement char emitted
        let held = detok.append(&tokenizer, &[3]).unwrap();
        assert_eq!(held, "", "partial UTF-8 token must be held back");

        // Completing the sequence emits the full emoji
        let emitted = detok.append(&tokenizer, &[3, 4]).unwrap();
        assert_eq!(emitted, "😀", "completing UTF-8 should emit the emoji");

        let full = tokenizer.decode(&[3u32, 4], true).unwrap();
        assert_eq!(detok.text, full, "detok.text must equal full decode");
    }

    // flush() must emit any bytes held back by append(), and a second flush
    // on the same (now-fully-drained) detok must return "".
    #[test]
    fn incremental_detok_flush_emits_pending() {
        let tokenizer = word_tokenizer();
        let mut detok = IncrementalDetok::new(String::new(), 0, no_skip());

        // Partial piece held back
        let held = detok.append(&tokenizer, &[3]).unwrap();
        assert_eq!(held, "", "partial UTF-8 must be held back before flush");

        // flush must emit something (a replacement char is acceptable here)
        let flushed = detok.flush(&tokenizer, &[3]).unwrap();
        assert!(!flushed.is_empty(), "flush must emit held-back text");

        // A second flush on an already-drained detok returns ""
        let flushed2 = detok.flush(&tokenizer, &[3]).unwrap();
        assert_eq!(flushed2, "", "second flush must return empty string");
    }

    // find_stop_in_tail must identify a stop whose prefix was already emitted,
    // returning the byte position that lets the caller emit the prefix ("hi")
    // before the stop ("STOP") in the same new-text window.
    #[test]
    fn find_stop_in_tail_first_token_prefix_and_stop() {
        let stops = vec!["STOP".to_owned()];
        // "hiSTOP": all 6 bytes are new, stop starts at byte 2
        assert_eq!(
            find_stop_in_tail("hiSTOP", 6, &stops),
            Some(2),
            "stop at pos 2 so 'hi' can be emitted as prefix"
        );
    }
}
