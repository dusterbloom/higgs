use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use higgs_models::{
    AnyCache, AnyModel, LogprobArrays, SamplingParams, apply_penalties, sample,
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
    chat_template::{ChatMessage, ChatTemplateRenderer},
    engine::{GenerationOutput, StreamingOutput},
    error::EngineError,
    model_loader,
    paged_prefix_cache::{DEFAULT_BLOCK_SIZE, PagedPrefixCache},
};

/// Default maximum number of cached prefixes.
const DEFAULT_PREFIX_CACHE_SIZE: usize = 8;

/// Sequences longer than this trigger chunked prefill to bound peak memory.
const CHUNKED_PREFILL_THRESHOLD: i32 = 512;

/// Number of tokens per chunk during chunked prefill.
const CHUNKED_PREFILL_CHUNK_SIZE: i32 = 512;

/// Cap MLX memory allocations to avoid Metal OOM on constrained GPUs.
///
/// Sets `mlx_set_memory_limit` to 75% of `max_recommended_working_set_size`
/// and `mlx_set_cache_limit` to 50%. This lets MLX manage memory within
/// bounds instead of the OS killing the process when Metal command buffers
/// exceed GPU capacity (the root cause of 35B crashes on 32GB machines).
#[allow(unsafe_code)]
pub(crate) fn set_wired_limit_to_max() {
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
                let mem_limit = max_rec * 3 / 4; // 75% of GPU max
                let cache_limit = max_rec / 2; // 50% of GPU max

                let mut prev_mem: usize = 0;
                let mut prev_cache: usize = 0;

                // Check env var to optionally disable memory limits for testing
                let limits_enabled = std::env::var("HIGGS_NO_MEM_LIMIT").is_err();

                if limits_enabled {
                    mlx_sys::mlx_set_memory_limit(&raw mut prev_mem, mem_limit);
                    mlx_sys::mlx_set_cache_limit(&raw mut prev_cache, cache_limit);
                }

                tracing::info!(
                    max_recommended_mb = max_rec / (1024 * 1024),
                    memory_limit_mb = mem_limit / (1024 * 1024),
                    cache_limit_mb = cache_limit / (1024 * 1024),
                    limits_enabled,
                    "MLX memory limits {} (prev mem={}MB, cache={}MB)",
                    if limits_enabled { "set" } else { "skipped" },
                    prev_mem / (1024 * 1024),
                    prev_cache / (1024 * 1024),
                );
            }
        }
        mlx_sys::mlx_device_info_free(info);
        mlx_sys::mlx_device_free(dev);
    }
}

/// Simple single-request inference engine with prefix KV caching.
///
/// Serializes requests with a mutex (same pattern as vllm-mlx's `SimpleEngine`).
/// Reuses cached KV states for shared prompt prefixes (e.g., system prompts).
pub struct SimpleEngine {
    model: Mutex<AnyModel>,
    prefix_cache: Mutex<PagedPrefixCache>,
    tokenizer: Tokenizer,
    template: Option<ChatTemplateRenderer>,
    model_name: String,
    eos_token_ids: Vec<u32>,
    /// Whether to enable thinking mode (Qwen3.5 `<think>` tags).
    enable_thinking: bool,
    /// Token ID for `</think>`, resolved from the tokenizer at load time.
    /// `None` if the tokenizer doesn't know this token (thinking will be disabled).
    think_close_token: Option<u32>,
    kv_cache_config: KvCacheConfig,
}

/// Intermediate state after prefix cache lookup and model locking.
struct PreparedGeneration<'a> {
    model: MutexGuard<'a, AnyModel>,
    cache: AnyCache,
    prompt_array: Array,
    prompt_len: u32,
    pixel_values: Option<Array>,
}

impl SimpleEngine {
    /// Load a model and tokenizer from a directory.
    pub fn load<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
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

        // Auto-detect thinking mode: Qwen3.5 models support <think> tags.
        // Override with HIGGS_ENABLE_THINKING=0 or HIGGS_ENABLE_THINKING=1.
        let mut enable_thinking = match std::env::var("HIGGS_ENABLE_THINKING").ok().as_deref() {
            Some("0" | "false") => false,
            Some("1" | "true") => true,
            _ => detect_thinking_support(model_dir),
        };

        // Resolve </think> token ID from the tokenizer. If the tokenizer
        // doesn't know this token, disable thinking to avoid injecting
        // out-of-vocab IDs into the embedding lookup.
        let think_close_token = tokenizer.encode("</think>", false).ok().and_then(|enc| {
            let ids = enc.get_ids();
            // Must encode to exactly one token to be usable as a forced stop.
            if ids.len() == 1 { Some(ids[0]) } else { None }
        });
        if enable_thinking && think_close_token.is_none() {
            tracing::warn!("Tokenizer has no single </think> token; disabling thinking mode");
            enable_thinking = false;
        }
        if enable_thinking {
            tracing::info!(
                think_close_token,
                "Thinking mode enabled (Qwen3.5 model detected)"
            );
        }

        set_wired_limit_to_max();

        tracing::info!(
            model_name = %model_name,
            eos_tokens = ?eos_token_ids,
            "Engine ready"
        );

        Ok(Self {
            model: Mutex::new(model),
            prefix_cache: Mutex::new(PagedPrefixCache::new(
                DEFAULT_PREFIX_CACHE_SIZE,
                DEFAULT_BLOCK_SIZE,
            )),
            tokenizer,
            template,
            model_name,
            eos_token_ids,
            enable_thinking,
            think_close_token,
            kv_cache_config,
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

    /// Apply chat template and tokenize messages.
    pub fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>, EngineError> {
        let renderer = self.template.as_ref().ok_or_else(|| {
            EngineError::Template(
                "This model has no chat template; use /v1/completions instead".to_owned(),
            )
        })?;
        let prompt = renderer.apply_with_thinking(messages, tools, true, self.enable_thinking)?;
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| EngineError::Tokenization(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
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
                "Reusing cached prefix"
            );
            let suffix = prompt_tokens.get(matched.prefix_len..).unwrap_or_default();
            if suffix.is_empty() {
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
            prompt_array,
            prompt_len,
            pixel_values,
        })
    }

    /// Run the prefill forward pass and sample the first token. Stores the
    /// post-prefill KV state back into the prefix cache (skipped for multimodal).
    /// Run prefill forward pass and sample the first token.
    /// Always computes chosen-token logprob for surprise scoring.
    fn run_prefill(
        &self,
        prompt_tokens: &[u32],
        prepared: &mut PreparedGeneration<'_>,
        params: &SamplingParams,
        logprob_top_n: Option<u32>,
        constraint: Option<&crate::constrained::ConstrainedGenerator>,
    ) -> Result<(Array, LogprobArrays), EngineError> {
        let logits = if let Some(ref pixel_values) = prepared.pixel_values {
            prepared
                .model
                .forward_multimodal(&prepared.prompt_array, pixel_values, &mut prepared.cache)
                .map_err(EngineError::Mlx)?
        } else {
            let seq_len = prepared.prompt_array.shape().get(1).copied().unwrap_or(0);
            if seq_len > CHUNKED_PREFILL_THRESHOLD {
                prepared
                    .model
                    .forward_chunked(
                        &prepared.prompt_array,
                        &mut prepared.cache,
                        CHUNKED_PREFILL_CHUNK_SIZE,
                    )
                    .map_err(EngineError::Mlx)?
            } else {
                prepared
                    .model
                    .forward(&prepared.prompt_array, None, &mut prepared.cache)
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

        // Always compute chosen-token logprob (for surprise scoring).
        // top-N argsort only runs when the user requests logprobs.
        let scaled = if params.temperature <= f32::EPSILON {
            constrained_logits
        } else {
            constrained_logits
                .multiply(Array::from_f32(1.0 / params.temperature))
                .map_err(EngineError::Mlx)?
        };
        let logprob_data =
            LogprobArrays::compute(&scaled, &current_token, logprob_top_n)
                .map_err(EngineError::Mlx)?;

        {
            let mut eval_targets: Vec<&Array> = vec![&current_token];
            eval_targets.extend(logprob_data.eval_targets());
            eval(eval_targets).map_err(EngineError::Mlx)?;
        }

        // Skip prefix cache for multimodal (image-specific KV states)
        if prepared.pixel_values.is_none() {
            let mut pc = self
                .prefix_cache
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            pc.store(prompt_tokens, &prepared.cache);
        }

        Ok((current_token, logprob_data))
    }

    /// Decode a single step: forward pass on the current token, apply penalties
    /// and optional constraint mask, then sample. Always computes chosen-token
    /// logprob for surprise scoring; top-N argsort only when `logprob_top_n` is `Some`.
    fn decode_step(
        current_token: &Array,
        model: &mut AnyModel,
        cache: &mut AnyCache,
        params: &SamplingParams,
        generated_tokens: &[u32],
        logprob_top_n: Option<u32>,
        constraint: Option<&crate::constrained::ConstrainedGenerator>,
    ) -> Result<(Array, LogprobArrays), EngineError> {
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

        // Always compute chosen-token logprob (for surprise scoring).
        // Temperature scaling replicates what `sample` sees internally.
        let scaled = if params.temperature <= f32::EPSILON {
            constrained
        } else {
            constrained
                .multiply(mlx_rs::array!(1.0 / params.temperature))
                .map_err(EngineError::Mlx)?
        };
        let logprob_data =
            LogprobArrays::compute(&scaled, &next_token, logprob_top_n)
                .map_err(EngineError::Mlx)?;

        Ok((next_token, logprob_data))
    }

    /// Decode the token buffer and return the text, mapping tokenizer errors.
    fn decode_tokens(&self, tokens: &[u32]) -> Result<String, EngineError> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| EngineError::Tokenization(e.to_string()))
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

    /// Run EGGROLL self-training on the loaded model.
    ///
    /// Locks the model mutex (blocking inference) for the duration of training.
    /// Only supported for Qwen3Next models.
    pub fn train_eggroll(
        &self,
        config: higgs_models::diffusion_eggroll::EggrollConfig,
        tokens: &[u32],
        prompt_len: usize,
        merge_interval: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let mut model = self
            .model
            .lock()
            .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;
        model
            .train_eggroll(config, tokens, prompt_len, merge_interval)
            .map_err(EngineError::Mlx)
    }

    /// Run PCAST gradient-based self-training on the loaded model.
    ///
    /// Locks the model mutex (blocking inference) for the duration of training.
    /// Only supported for Qwen3Next models.
    pub fn train_gradient(
        &self,
        config: higgs_models::diffusion_eggroll::EggrollConfig,
        tokens: &[u32],
        prompt_len: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let mut model = self
            .model
            .lock()
            .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;
        model
            .train_gradient(config, tokens, prompt_len)
            .map_err(EngineError::Mlx)
    }

    /// Convert a token count to u32, with an overflow error.
    fn completion_len(tokens: &[u32]) -> Result<u32, EngineError> {
        tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Too many tokens generated".to_owned()))
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
                surprise: None,
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
        mut constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<GenerationOutput, EngineError> {
        let logprob_top_n = logprobs.then(|| top_logprobs.unwrap_or(0));

        let mut prepared = self.prepare_generation(prompt_tokens, pixel_values)?;
        let prompt_len = prepared.prompt_len;
        let (current_token, first_logprob_data) = self.run_prefill(
            prompt_tokens,
            &mut prepared,
            params,
            logprob_top_n,
            constraint.as_ref(),
        )?;

        // Capture T1 (already eval'd inside run_prefill).
        let first_token_id: u32 = current_token.item();
        // Advance the constraint past the first sampled token before decode.
        if let Some(ref mut cg) = constraint {
            cg.advance(first_token_id);
        }
        let mut tokens: Vec<u32> = vec![first_token_id];
        let mut all_logprobs: Option<Vec<higgs_models::TokenLogprobInfo>> = logprobs.then(Vec::new);

        // Surprise accumulator: sum of -logprob for each completion token.
        let first_surprise: f32 = -first_logprob_data.token_logprob.item::<f32>();
        let mut surprise_sum: f32 = first_surprise;

        if let Some(all_lp) = &mut all_logprobs {
            all_lp.push(first_logprob_data.materialize(first_token_id));
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
                surprise: Some(surprise_sum),
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
                    surprise: Some(surprise_sum),
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
                surprise: Some(surprise_sum),
            });
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
            eval_targets.extend(next_logprob_data.eval_targets());
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
        let think_close_token = if self.enable_thinking {
            self.think_close_token
        } else {
            None
        };
        let mut thinking_tokens: u32 = 0;
        let mut seen_think_close = false;

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
                eval_targets.extend(following_logprob_data.eval_targets());
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

            // Accumulate surprise from the chosen-token logprob.
            surprise_sum += -next_logprob_data.token_logprob.item::<f32>();

            // Materialize full logprobs for response only if user requested them.
            if let Some(all_lp) = &mut all_logprobs {
                all_lp.push(next_logprob_data.materialize(token_id));
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
                let mean_surprise = surprise_sum / completion_len as f32;
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: all_logprobs,
                    surprise: Some(mean_surprise),
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
                let mean_surprise = surprise_sum / completion_len as f32;
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: all_logprobs,
                    surprise: Some(mean_surprise),
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
                    let mean_surprise = surprise_sum / completion_len as f32;
                    return Ok(GenerationOutput {
                        text: truncated,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: completion_len,
                        token_logprobs: all_logprobs,
                        surprise: Some(mean_surprise),
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
                let mean_surprise = surprise_sum / completion_len as f32;
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: all_logprobs,
                    surprise: Some(mean_surprise),
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
        mut constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<(), EngineError> {
        let logprob_top_n = logprobs.then(|| top_logprobs.unwrap_or(0));

        let mut prepared = self.prepare_generation(prompt_tokens, pixel_values)?;
        let prompt_len = prepared.prompt_len;
        let (current_token, first_logprob_data) = self.run_prefill(
            prompt_tokens,
            &mut prepared,
            params,
            logprob_top_n,
            constraint.as_ref(),
        )?;

        let mut all_tokens: Vec<u32> = Vec::new();
        let first_token_id: u32 = current_token.item();
        // Advance the constraint past the first sampled token before decode.
        if let Some(ref mut cg) = constraint {
            cg.advance(first_token_id);
        }
        all_tokens.push(first_token_id);

        let first_decoded = self.decode_tokens(&all_tokens)?;
        let (first_text, first_hit_stop) = if stop_sequences.is_empty() {
            (first_decoded.clone(), false)
        } else {
            check_stop_sequences(&first_decoded, stop_sequences).map_or_else(
                || (first_decoded.clone(), false),
                |truncated| (truncated, true),
            )
        };
        let mut prev_decoded_len = first_decoded.len();

        let first_is_eos = self.eos_token_ids.contains(&first_token_id);
        let finished = first_is_eos || first_hit_stop || 1 >= max_tokens;

        let first_logprob = if logprobs {
            Some(first_logprob_data.materialize(first_token_id))
        } else {
            None
        };

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
            })
            .is_err()
        {
            return Ok(());
        }

        if finished {
            return Ok(());
        }

        // Thinking budget (streaming): force </think> after N tokens.
        const THINKING_BUDGET: u32 = 256;
        let think_close_token = if self.enable_thinking {
            self.think_close_token
        } else {
            None
        };
        let mut thinking_tokens: u32 = 0;
        let mut seen_think_close = false;

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
            eval_targets.extend(next_logprob_data.eval_targets());
            async_eval(eval_targets).map_err(EngineError::Mlx)?;
        }

        loop {
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
                eval_targets.extend(following_logprob_data.eval_targets());
                async_eval(eval_targets).map_err(EngineError::Mlx)?;
            }

            let mut token_id: u32 = next_token.item();

            // Thinking budget: force </think> after N tokens if model hasn't closed it.
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

            // Advance constrained generator state
            if let Some(ref mut cg) = constraint {
                cg.advance(token_id);
            }

            let token_logprob = if logprobs {
                Some(next_logprob_data.materialize(token_id))
            } else {
                None
            };

            all_tokens.push(token_id);

            let completion_len = Self::completion_len(&all_tokens)?;

            let full_text = self.decode_tokens(&all_tokens)?;
            let new_text = full_text
                .get(prev_decoded_len..)
                .unwrap_or_default()
                .to_owned();
            let old_decoded_len = prev_decoded_len;
            prev_decoded_len = full_text.len();

            let (final_new_text, hit_stop_seq) = if stop_sequences.is_empty() {
                (new_text, false)
            } else {
                check_stop_sequences(&full_text, stop_sequences).map_or(
                    (new_text, false),
                    |truncated| {
                        let emit = truncated
                            .get(old_decoded_len..)
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

    // Check top-level first, then text_config (VLM/Qwen3.5 nested config)
    let eos_value = config.get("eos_token_id").or_else(|| {
        config
            .get("text_config")
            .and_then(|tc| tc.get("eos_token_id"))
    });

    match eos_value {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .map_or_else(Vec::new, |id| vec![id]),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_u64().and_then(|val| u32::try_from(val).ok()))
            .collect(),
        Some(other) => {
            tracing::warn!(value = ?other, "Unexpected eos_token_id type in config.json");
            vec![]
        }
        None => {
            tracing::warn!(
                "No eos_token_id found in config.json, generation will rely on max_tokens"
            );
            vec![]
        }
    }
}

/// Detect whether a model supports thinking mode based on model_type.
fn detect_thinking_support(model_dir: &Path) -> bool {
    let config_path = model_dir.join("config.json");
    let config_str = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let config: serde_json::Value = match serde_json::from_str(&config_str) {
        Ok(v) => v,
        Err(_) => return false,
    };
    // Qwen3.5 models (qwen3_5, qwen3_5_moe) support <think> tags.
    // Check both top-level and nested text_config for VLM wrappers.
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

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::{check_stop_sequences, derive_model_name};
    use std::path::Path;

    /// Write a config.json file into the given directory with the provided JSON content.
    fn write_config(dir: &std::path::Path, json: &str) {
        std::fs::write(dir.join("config.json"), json).unwrap();
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
}
