use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use higgs_models::{
    AnyCache, AnyModel, LogprobArrays, SamplingParams, apply_penalties, compute_probs,
    diffusion::{accept_prefix, accept_prefix_rs},
    sample, sample_from_probs,
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

/// Log GPU limits. No cap by default — `e5c47264` removed the cap because it
/// cost 5× throughput on 35B MoE decode. Use [`set_mlx_memory_cap`] when
/// targeting 27B+DFlash where the tape-verify allocator SIGKILLs otherwise.
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
                tracing::info!(
                    max_recommended_mb = max_rec / (1024 * 1024),
                    "MLX memory: letting framework manage limits (GPU max {}MB)",
                    max_rec / (1024 * 1024),
                );
            }
        }
        mlx_sys::mlx_device_info_free(info);
        mlx_sys::mlx_device_free(dev);
    }
}

/// Cap MLX memory at `cap_fraction * max_recommended_working_set_size`.
/// Required for 27B/35B-dense + DFlash to avoid Metal silent SIGKILL on
/// verify-tape allocations. Only call on dense large models with DFlash.
#[allow(unsafe_code)]
pub(crate) fn set_mlx_memory_cap(cap_fraction: f64) {
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
                #[allow(
                    clippy::cast_precision_loss,
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss
                )]
                let cap = ((max_rec as f64) * cap_fraction) as usize;
                let mut prev: usize = 0;
                let _ = mlx_sys::mlx_set_memory_limit(&raw mut prev, cap);
                tracing::info!(
                    cap_mb = cap / (1024 * 1024),
                    max_recommended_mb = max_rec / (1024 * 1024),
                    cap_fraction,
                    "MLX memory: cap enabled for DFlash large-dense stability"
                );
            }
        }
        mlx_sys::mlx_device_info_free(info);
        mlx_sys::mlx_device_free(dev);
    }
}

/// DFlash block-diffusion speculative decoding state.
///
/// When present, `generate_inner` uses the DFlash draft-verify loop
/// instead of autoregressive decode: draft 16 tokens per round via the
/// small drafter, verify with the target model, accept prefix + correction.
struct DFlashState {
    /// GPU drafter — uses MLX ops, no CPU↔GPU transfer needed.
    /// Wrapped in Mutex for interior mutability (drafter.forward is &mut).
    gpu_drafter: std::sync::Mutex<higgs_models::dflash::DFlashDrafter>,
    /// CPU BLAS engine for off-GPU drafter forward (Arc for pipeline thread sharing).
    /// `None` when the CPU engine is not needed at runtime — i.e. ANE is active
    /// (the ANE worker owns its own copy) AND pipelined drafting is disabled.
    /// Avoids a ~3 GB heap tax on 27B when `HIGGS_DFLASH_DISABLE_ANE=1` with
    /// pipeline off.
    cpu_engine: Option<Arc<higgs_models::dflash_cpu::DFlashCpuEngine>>,
    /// ANE+CPU hybrid executor — pinned to a dedicated worker thread so its
    /// `!Send` IOSurface handles never cross threads. Handle is `Send + Sync`.
    /// When present, preferred over cpu_engine.
    #[cfg(feature = "ane")]
    ane_worker: Option<higgs_models::dflash_ane::DFlashAneWorkerHandle>,
    tap_layers: Vec<usize>,
    block_size: i32,
    mask_token_id: i32,
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
    /// Optional DFlash speculative decoding drafter.
    dflash: Option<DFlashState>,
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
    /// Load a model and tokenizer from a directory, with optional DFlash drafter.
    pub fn load<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
    ) -> Result<Self, EngineError> {
        Self::load_with_dflash(dir, kv_cache_config, None)
    }

    /// Load a model with an optional DFlash speculative decoding drafter.
    pub fn load_with_dflash<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        dflash_path: Option<&Path>,
    ) -> Result<Self, EngineError> {
        let model_dir = dir.as_ref();
        let model_name = derive_model_name(model_dir);

        tracing::info!(model_dir = %model_dir.display(), "Loading model");

        let (mut model, pending_ane_gdn, pending_ane_lm_head, pending_ane_mlp_int8) =
            model_loader::load_model(model_dir)?;
        // P0.8 Stage 2: SimpleEngine has no dedicated inference thread (model
        // lives in Mutex shared across Tokio threads). Finalize immediately on
        // the load thread — for proper thread-pinning use HIGGS_ANE_GDN_WORKER=1
        // which keeps the legacy mpsc handle path.
        #[cfg(feature = "ane")]
        {
            if let Some(pending) = pending_ane_gdn {
                if let AnyModel::Qwen3Next(qwen) = &mut model {
                    if let Err(e) = qwen.finalize_ane_gdn_inline(pending.weights, pending.seq_len) {
                        tracing::error!(
                            error = %e,
                            "SimpleEngine: finalize_ane_gdn_inline failed — \
                             falling back to Metal (use HIGGS_ANE_GDN_WORKER=1 for legacy path)"
                        );
                    }
                }
            }
            if let Some(pending) = pending_ane_lm_head {
                if let AnyModel::Qwen3Next(qwen) = &mut model {
                    if let Err(e) = qwen.finalize_ane_lm_head_inline(
                        pending.weights,
                        pending.hidden,
                        pending.vocab,
                        pending.seq_len,
                    ) {
                        tracing::error!(
                            error = %e,
                            "SimpleEngine: finalize_ane_lm_head_inline failed — \
                             falling back to Metal"
                        );
                    }
                }
            }
            if let Some(pending) = pending_ane_mlp_int8 {
                if let AnyModel::Qwen3Next(qwen) = &mut model {
                    if let Err(e) = qwen.finalize_ane_mlp_layer0_int8_inline(
                        pending.gate_f32,
                        pending.up_f32,
                        pending.down_f32,
                        pending.hidden,
                        pending.inter,
                        pending.seq_len,
                    ) {
                        tracing::error!(
                            error = %e,
                            "SimpleEngine: finalize_ane_mlp_layer0_int8_inline failed — \
                             falling back to Metal"
                        );
                    }
                }
            }
        }
        #[cfg(not(feature = "ane"))]
        {
            let _ = pending_ane_gdn;
            let _ = pending_ane_lm_head;
            let _ = pending_ane_mlp_int8;
        }
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

        // Load DFlash drafter if path provided or HIGGS_DFLASH_PATH env var set.
        // Reads safetensors directly into CPU f32 vecs — no MLX arrays, no GPU
        // memory used for drafter weights.
        let dflash_resolved = dflash_path.map(|p| p.to_path_buf()).or_else(|| {
            std::env::var("HIGGS_DFLASH_PATH")
                .ok()
                .map(std::path::PathBuf::from)
        });
        let dflash = if let Some(ref dp) = dflash_resolved {
            // Load GPU drafter (MLX ops, no CPU↔GPU transfer)
            tracing::info!(drafter = %dp.display(), "Loading DFlash drafter (GPU)");
            let t0 = std::time::Instant::now();
            let mut gpu_drafter =
                higgs_models::dflash::load_dflash_drafter(dp).map_err(EngineError::Model)?;
            let cfg = gpu_drafter.config.clone();
            let tap_layers = cfg.target_layer_ids().to_vec();
            let block_size = std::env::var("HIGGS_DFLASH_BLOCK_SIZE")
                .ok()
                .and_then(|s| s.parse::<i32>().ok())
                .filter(|&n| n >= 2)
                .unwrap_or(higgs_models::dflash::DEFAULT_DECODE_BLOCK_SIZE);
            if block_size as usize != gpu_drafter.config.block_size as usize {
                tracing::info!(
                    trained = gpu_drafter.config.block_size,
                    runtime = block_size,
                    "DFlash decode block_size differs from drafter's trained value \
                     (override HIGGS_DFLASH_BLOCK_SIZE to tune)"
                );
                gpu_drafter.config.block_size = block_size;
            }
            let mask_token_id = cfg.mask_token_id();
            tracing::info!(
                elapsed_ms = t0.elapsed().as_millis(),
                "DFlash GPU drafter loaded from safetensors"
            );

            // Decide whether to load the CPU engine at all. It's needed only when:
            //   - ANE is active (for one-time compile into the ANE worker), OR
            //   - HIGGS_DFLASH_PIPELINE=1 (pipelined drafter uses it at runtime).
            // Loading eats ~3 GB heap on 27B (bf16 weight copies); skip when
            // neither path will touch it.
            let pipeline_enabled = std::env::var("HIGGS_DFLASH_PIPELINE")
                .map(|v| v == "1")
                .unwrap_or(false);
            #[cfg(feature = "ane")]
            let ane_wanted = std::env::var_os("HIGGS_DFLASH_DISABLE_ANE").is_none();
            #[cfg(not(feature = "ane"))]
            let ane_wanted = false;
            let needs_cpu_engine = ane_wanted || pipeline_enabled;

            let mut cpu_engine_loaded = if needs_cpu_engine {
                tracing::info!(drafter = %dp.display(), "Loading DFlash CPU engine (for ANE compile / pipeline drafter)");
                let t1 = std::time::Instant::now();
                let (mut cpu_engine, _cfg) =
                    higgs_models::dflash_cpu::load_dflash_cpu_engine_from_safetensors(dp)
                        .map_err(EngineError::Model)?;
                if block_size as usize != cpu_engine.config.block_size {
                    cpu_engine.config.block_size = block_size as usize;
                }
                tracing::info!(
                    elapsed_ms = t1.elapsed().as_millis(),
                    "DFlash CPU engine loaded from safetensors"
                );
                Some(cpu_engine)
            } else {
                tracing::info!(
                    "Skipping DFlash CPU engine load \
                     (ANE disabled + HIGGS_DFLASH_PIPELINE!=1) — saves ~3 GB heap"
                );
                None
            };

            // Compile ANE+CPU hybrid executor if available.
            // Spawn a dedicated worker thread that owns the executor — the
            // returned handle is `Send + Sync` so it can live in `AppState`.
            #[cfg(feature = "ane")]
            let ane_worker = if std::env::var_os("HIGGS_DFLASH_DISABLE_ANE").is_some() {
                tracing::warn!("HIGGS_DFLASH_DISABLE_ANE set — forcing CPU BLAS fallback");
                None
            } else if cpu_engine_loaded.is_some() {
                let t0 = std::time::Instant::now();
                // When pipeline is off, move the engine into the ANE worker
                // instead of cloning — the worker's copy is sufficient, and
                // holding a redundant copy on the engine side raises peak RSS
                // by ~3 GB during ANE compile (jetsam fires ~23.5 GB on 27B).
                // See .planning/phase1-ane-memory-surgery-plan.md.
                let engine_for_worker = if pipeline_enabled {
                    cpu_engine_loaded.as_ref().unwrap().clone()
                } else {
                    cpu_engine_loaded.take().unwrap()
                };
                match higgs_models::dflash_ane::spawn_ane_worker(engine_for_worker) {
                    Ok(handle) => {
                        tracing::info!(
                            elapsed_ms = t0.elapsed().as_millis(),
                            moved_engine = !pipeline_enabled,
                            "DFlash ANE worker spawned (executor compiled on worker thread)"
                        );
                        Some(handle)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "ANE worker init failed, falling back to CPU BLAS");
                        None
                    }
                }
            } else {
                None
            };

            // Retain the CPU engine only when pipelined drafting will use it at
            // runtime. When ANE is active and pipeline is off, the ANE worker
            // already owns its own copy — drop ours to reclaim ~3 GB.
            let cpu_engine_retained = if pipeline_enabled {
                cpu_engine_loaded.map(Arc::new)
            } else {
                drop(cpu_engine_loaded);
                None
            };

            #[cfg(feature = "ane")]
            let backend = if ane_worker.is_some() {
                "ANE+CPU"
            } else {
                "CPU BLAS"
            };
            #[cfg(not(feature = "ane"))]
            let backend = "CPU BLAS";
            tracing::info!(
                tap_layers = ?tap_layers,
                block_size,
                mask_token_id,
                backend,
                "DFlash drafter loaded — speculative decoding enabled"
            );
            Some(DFlashState {
                gpu_drafter: std::sync::Mutex::new(gpu_drafter),
                cpu_engine: cpu_engine_retained,
                #[cfg(feature = "ane")]
                ane_worker,
                tap_layers,
                block_size,
                mask_token_id,
            })
        } else {
            None
        };

        // Cap MLX memory for large-dense + DFlash to prevent Metal silent
        // SIGKILL on verify-tape allocations at long contexts (crash fix per
        // .planning/next-session-27b-dflash-crash.md). Override with
        // HIGGS_MLX_CAP_FRACTION=0 (disable) or e.g. 0.80 (tighter).
        if dflash.is_some() && model.num_layers() > 32 {
            let cap_frac = std::env::var("HIGGS_MLX_CAP_FRACTION")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|f| (0.0..=1.0).contains(f))
                .unwrap_or(0.88);
            if cap_frac > 0.0 {
                set_mlx_memory_cap(cap_frac);
            }
        }

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
    /// Optionally computes logprobs for the first token.
    fn run_prefill(
        &self,
        prompt_tokens: &[u32],
        prepared: &mut PreparedGeneration<'_>,
        params: &SamplingParams,
        logprob_top_n: Option<u32>,
        constraint: Option<&crate::constrained::ConstrainedGenerator>,
    ) -> Result<(Array, Option<LogprobArrays>), EngineError> {
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
        // DFlash speculative decoding: use draft-verify loop when available,
        // no constraints active, and no multimodal input.
        if self.dflash.is_some() && constraint.is_none() && pixel_values.is_none() {
            return self.generate_dflash_inner(prompt_tokens, max_tokens, params, stop_sequences);
        }

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
                // Decode only a suffix window for stop detection. Full
                // decode_tokens(&tokens) every step is O(gen_len) on a growing
                // vec → O(gen_len²) cumulative, serialized against the GPU
                // pipeline. Stop sequences are local string patterns; 64 tokens
                // (~200-500 chars) covers any realistic stop. On detection,
                // re-decode full tokens once to produce correctly truncated
                // return text.
                const STOP_CHECK_WINDOW: usize = 64;
                let tail_start = tokens.len().saturating_sub(STOP_CHECK_WINDOW);
                let tail = self.decode_tokens(&tokens[tail_start..])?;
                if check_stop_sequences(&tail, stop_sequences).is_some() {
                    let full = self.decode_tokens(&tokens)?;
                    if let Some(truncated) = check_stop_sequences(&full, stop_sequences) {
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

    /// DFlash speculative decode: draft 16 tokens per round, verify, accept prefix.
    ///
    /// Mirrors the proven `test_dflash_27b_full_loop` logic adapted for the
    /// engine's sampling, stop-sequence, and EOS handling.
    #[allow(
        clippy::too_many_lines,
        clippy::significant_drop_tightening,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn generate_dflash_inner(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
    ) -> Result<GenerationOutput, EngineError> {
        let dflash = self.dflash.as_ref().expect("DFlash state must be Some");
        let prompt_len = Self::prompt_len(prompt_tokens)?;
        let has_stop_sequences = !stop_sequences.is_empty();

        let mut model = self
            .model
            .lock()
            .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;

        let mut cache = model
            .make_cache_with_config(self.kv_cache_config)
            .map_err(EngineError::Mlx)?;

        // Prefill with taps
        let prompt_array = Array::from(prompt_tokens).index(NewAxis);
        let (prefill_logits, taps) = model
            .forward_with_taps(&prompt_array, None, &mut cache, &dflash.tap_layers)
            .map_err(EngineError::Mlx)?;
        eval([&prefill_logits]).map_err(EngineError::Mlx)?;

        // Sample first token
        let last_logits = prefill_logits.index((.., -1, ..));
        let first_token = sample(&last_logits, params).map_err(EngineError::Mlx)?;
        eval([&first_token]).map_err(EngineError::Mlx)?;

        let first_token_id: u32 = first_token.item();
        let mut tokens: Vec<u32> = vec![first_token_id];

        if self.eos_token_ids.contains(&first_token_id) || max_tokens <= 1 {
            return Ok(GenerationOutput {
                text: self.decode_tokens(&tokens)?,
                finish_reason: if self.eos_token_ids.contains(&first_token_id) {
                    "stop"
                } else {
                    "length"
                }
                .to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprobs: None,
            });
        }

        let mut current_taps = taps;
        let mut drafter_cache = dflash.gpu_drafter.lock().unwrap().make_cache();
        // Pipelined drafting is the only path that needs a CPU cache. When
        // pipeline is off, `dflash.cpu_engine` is `None`; we never touch the
        // CPU cache below that branch.
        let mut cpu_cache: Option<higgs_models::dflash_cpu::DFlashCpuCache> =
            dflash.cpu_engine.as_ref().map(|e| e.make_cache());
        let mut last_token = first_token_id as i32;
        let mut start = prompt_len as i32;
        let block_size = dflash.block_size;
        let mask_id = dflash.mask_token_id;
        let hidden_dim = dflash.gpu_drafter.lock().unwrap().config.hidden_size;
        let t_start = Instant::now();
        let trace = std::env::var("HIGGS_DFLASH_TRACE").map_or(false, |v| v == "1");
        let kv_debug = std::env::var("HIGGS_DFLASH_DEBUG_KV").map_or(false, |v| v == "1");
        // Pipelining the CPU/ANE drafter against the target verify was attempted and
        // REGRESSED throughput (422ms round_total vs 311ms, accept 5.5→4.0, 13→8.7 tok/s):
        // the pipeline path uses the CPU/ANE drafter which (a) is slower than the lazy
        // GPU drafter the non-pipeline path uses implicitly, (b) contends with the
        // target's ANE GDN verify for the ANE worker queue, and (c) produces lower-
        // quality drafts than the 9B GPU drafter. The implicit overlap between the
        // lazy GPU drafter graph (evaluated during the draft token eval) and the
        // synchronous ANE GDN dispatches in verify is the design that actually works.
        // Keep the pipeline code path in case a future config (smaller ANE drafter,
        // dim-matched 4B target) makes it win.
        //
        // 2026-04-16: opt-in via HIGGS_DFLASH_PIPELINE=1 so we can re-evaluate
        // under topology B (GDN on GPU, ANE queue free) without a rebuild.
        let pipeline = std::env::var("HIGGS_DFLASH_PIPELINE")
            .map(|v| v == "1")
            .unwrap_or(false);
        let verify_layer_chunk: usize = std::env::var("HIGGS_DFLASH_LAYER_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if model.num_layers() > 32 { 16 } else { 0 });
        let mut round_idx: u32 = 0;
        let mut total_accepted: u64 = 0;

        // Pipelined drafter: Option holding background thread handle + result receiver.
        // When pipelining, the CPU drafter for round N+1 runs in background while
        // GPU verify for N+1 runs on the main thread.
        type DraftResult = (Vec<f32>, higgs_models::dflash_cpu::DFlashCpuCache);
        let mut pending_draft: Option<std::sync::mpsc::Receiver<DraftResult>> = None;

        loop {
            let t_round = Instant::now();
            let t_loop_start = Instant::now();

            // c. Drafter forward — receive from pipeline or run synchronously
            let t_draft = Instant::now();
            let loop_overhead_ms = t_draft.elapsed().as_secs_f64() * 1000.0;
            let (draft_hidden, embed_ms) = if let Some(rx) = pending_draft.take() {
                // Pipelined: drafter was already running in background (embed done at spawn time)
                let (out_f32, returned_cache) = rx.recv().expect("drafter thread panicked");
                cpu_cache = Some(returned_cache);
                // Context-only cache: no crop needed — only ctx positions were persisted.
                if kv_debug {
                    eprintln!(
                        "KV_ROUND round={} path=pipelined start={} cache_len={}",
                        round_idx + 1,
                        start,
                        cpu_cache.as_ref().map_or(0, |c| c.len),
                    );
                }
                (
                    Array::from_slice(&out_f32, &[1, block_size, hidden_dim]),
                    0.0,
                )
            } else {
                // First round or pipeline disabled: GPU drafter forward (no CPU transfer)
                // a. Build block: [anchor, mask, mask, ...]
                let mut block_tokens = vec![mask_id; block_size as usize];
                block_tokens[0] = last_token;
                let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);

                // b. Embed through target's embedding layer (lazy, no eval yet)
                let t_embed = Instant::now();
                let noise_embedding = model
                    .embed_token_ids(&block_ids)
                    .map_err(EngineError::Mlx)?;
                // No eval here — chain into drafter graph for single GPU submit

                // GPU drafter forward — all MLX ops, no CPU↔GPU transfer
                // Depends on noise_embedding + taps (all lazy, chained into one graph)
                let draft_hidden = dflash
                    .gpu_drafter
                    .lock()
                    .unwrap()
                    .forward(&noise_embedding, &current_taps, &mut drafter_cache)
                    .map_err(EngineError::Mlx)?;
                let em_ms = t_embed.elapsed().as_secs_f64() * 1000.0;

                // Context-only cache: no crop needed.
                if kv_debug {
                    eprintln!(
                        "KV_ROUND round={} path=gpu_drafter start={} cache_len={}",
                        round_idx + 1,
                        start,
                        drafter_cache
                            .first()
                            .and_then(|c| c.as_ref())
                            .map_or(0, |(k, _)| k.shape()[2]),
                    );
                }

                (draft_hidden, em_ms)
            };
            let draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;

            // e. Target lm_head on sliced hidden → sample draft tokens
            //    (greedy argmax when temperature=0, rejection sampling otherwise)
            let t_lm_draft = Instant::now();
            let t_slice = Instant::now();
            let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
            let slice_ms = t_slice.elapsed().as_secs_f64() * 1000.0;
            let t_logits = Instant::now();
            let draft_logits = model
                .forward_all_logits_from_hidden(&draft_hidden_sliced)
                .map_err(EngineError::Mlx)?;
            let logits_ms = t_logits.elapsed().as_secs_f64() * 1000.0;

            let (draft_u32, draft_i32, rs_state, argmax_build_ms, eval_block_ms, draft_transfer_ms) =
                if params.temperature == 0.0 {
                    let t_argmax = Instant::now();
                    let draft_token_arr =
                        mlx_rs::argmax_axis!(draft_logits, -1).map_err(EngineError::Mlx)?;
                    let argmax_build_ms = t_argmax.elapsed().as_secs_f64() * 1000.0;
                    let t_eval_done = Instant::now();
                    eval([&draft_token_arr]).map_err(EngineError::Mlx)?;
                    let eval_block_ms = t_eval_done.elapsed().as_secs_f64() * 1000.0;
                    let t_draft_transfer = Instant::now();
                    let du32: Vec<u32> = draft_token_arr
                        .reshape(&[-1])
                        .map_err(EngineError::Mlx)?
                        .as_slice::<u32>()
                        .to_vec();
                    let di32: Vec<i32> = du32.iter().map(|&x| x as i32).collect();
                    let draft_transfer_ms = t_draft_transfer.elapsed().as_secs_f64() * 1000.0;
                    (
                        du32,
                        di32,
                        None,
                        argmax_build_ms,
                        eval_block_ms,
                        draft_transfer_ms,
                    )
                } else {
                    let t_rs = Instant::now();
                    let (du32, di32, arr, q) = rs_draft_sample(&draft_logits, params)?;
                    let rs_ms = t_rs.elapsed().as_secs_f64() * 1000.0;
                    (du32, di32, Some((arr, q)), 0.0, 0.0, rs_ms)
                };
            let _ = eval_block_ms; // kept for trace format compatibility
            let lm_draft_ms = t_lm_draft.elapsed().as_secs_f64() * 1000.0;

            // f. Build verify input: [anchor, draft_0..draft_14]
            let t_gap = Instant::now();
            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_i32);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            // Drop draft intermediates before verify to free Metal buffers
            drop(draft_hidden);
            let gap_ms = t_gap.elapsed().as_secs_f64() * 1000.0;

            // Diagnostic: HIGGS_DFLASH_NO_TAPE=1 runs a measurement-only
            // forward_with_taps (no tape recording) on the same input first,
            // then restores GDN/KV state so the real verify starts unaffected.
            // Lets us attribute the cost the tape recording adds to forward.
            // Diagnostic only — adds one full forward per round.
            let no_tape_fwd_ms = if std::env::var("HIGGS_DFLASH_NO_TAPE").as_deref() == Ok("1") {
                // Snapshot GDN state so we can restore after the probe forward.
                let probe_snapshots: Vec<(Option<Array>, Option<Array>, i32)> = cache
                    .as_hybrid()
                    .iter()
                    .map(|lc| match lc {
                        Some(higgs_models::qwen3_next::LayerCache::Arrays(ac)) => {
                            ac.eval_arrays().expect("eval_arrays");
                            (ac.conv_state.clone(), ac.ssm_state.clone(), ac.offset)
                        }
                        _ => (None, None, 0),
                    })
                    .collect();
                let t = Instant::now();
                let (logits_nt, _taps_nt) = model
                    .forward_with_taps(&verify_input, None, &mut cache, &dflash.tap_layers)
                    .map_err(EngineError::Mlx)?;
                eval([&logits_nt]).map_err(EngineError::Mlx)?;
                let probe_ms = t.elapsed().as_secs_f64() * 1000.0;
                // Restore GDN snapshots and rollback KV.
                for (lc, (snap_conv, snap_ssm, snap_offset)) in
                    cache.as_hybrid_mut().iter_mut().zip(probe_snapshots.iter())
                {
                    match lc {
                        Some(higgs_models::qwen3_next::LayerCache::Arrays(ac)) => {
                            ac.conv_state = snap_conv.clone();
                            ac.ssm_state = snap_ssm.clone();
                            ac.offset = *snap_offset;
                        }
                        Some(higgs_models::qwen3_next::LayerCache::KV(kv)) => {
                            kv.rollback(verify_len);
                        }
                        _ => {}
                    }
                }
                Some(probe_ms)
            } else {
                None
            };

            // g+h. Tape-recording verify (chunked for large models to bound GPU peak memory)
            //
            // SNAPSHOT PATH NOTE: Array::clone() in mlx-rs is a refcount on the lazy
            // graph node (mlx_array_set), not a data copy — so we can clone un-evaluated
            // arrays freely. The previous eval-per-layer loop forced 48 serial GPU syncs
            // (24 layers × {conv, ssm}) which dominated the round (~220ms at 4B).
            // We now clone handles only; if partial-accept triggers rollback, the
            // `eval(replay_states)` at the end of that branch will materialize the
            // restored state lazily, walking back through the original graph.
            //
            // IMPORTANT: When HIGGS_TARGET_ANE_GDN=1 is active, forward_with_taps_tape
            // is NOT lazy — each GDN layer makes 3 blocking ANE dispatch calls
            // (GdnAneWorkerHandle::dispatch: eval input, send to worker, recv result).
            // For a 32-layer model with 24 GDN layers this is 72 synchronous round-trips
            // per verify, accounting for ~228ms/round that does not appear in any sub-timer.
            // The verify_build_ms field in the trace captures this hidden cost.
            let t_verify_build = Instant::now();
            let (verify_logits, verify_taps, layer_tapes) = if verify_layer_chunk > 0 {
                model
                    .forward_with_taps_tape_chunked(
                        &verify_input,
                        None,
                        &mut cache,
                        &dflash.tap_layers,
                        verify_layer_chunk,
                    )
                    .map_err(EngineError::Mlx)?
            } else {
                model
                    .forward_with_taps_tape(&verify_input, None, &mut cache, &dflash.tap_layers)
                    .map_err(EngineError::Mlx)?
            };
            let verify_build_ms = t_verify_build.elapsed().as_secs_f64() * 1000.0;

            // Force the deferred forward pass first so we can attribute its
            // cost independently of the argmax + final sync. MLX is lazy:
            // without this, the argmax eval below would be charged for the
            // entire forward.
            let t_verify_fwd = Instant::now();
            eval([&verify_logits]).map_err(EngineError::Mlx)?;
            let verify_fwd_ms = t_verify_fwd.elapsed().as_secs_f64() * 1000.0;

            // i. Accept: greedy argmax+accept_prefix when temp=0, else rejection sampling
            let (verify_argmax_ms, accept_ms, accepted) = match rs_state {
                None => {
                    let t_verify_argmax = Instant::now();
                    let verify_argmax =
                        mlx_rs::argmax_axis!(verify_logits, -1).map_err(EngineError::Mlx)?;
                    eval([&verify_argmax]).map_err(EngineError::Mlx)?;
                    let v_ms = t_verify_argmax.elapsed().as_secs_f64() * 1000.0;
                    let t_accept = Instant::now();
                    let verify_flat: Vec<u32> = verify_argmax
                        .reshape(&[-1])
                        .map_err(EngineError::Mlx)?
                        .as_slice::<u32>()
                        .to_vec();
                    if std::env::var("HIGGS_DFLASH_FORENSICS").as_deref() == Ok("1")
                        && round_idx == 0
                    {
                        tracing::info!(
                            "dflash_forensics round=1 last_token={} draft_tokens={:?} verify_argmax={:?}",
                            last_token,
                            draft_u32,
                            verify_flat,
                        );
                    }
                    let acc = accept_prefix(&draft_u32, &verify_flat);
                    let a_ms = t_accept.elapsed().as_secs_f64() * 1000.0;
                    (v_ms, a_ms, acc)
                }
                Some((draft_arr, q_probs)) => {
                    let t_accept = Instant::now();
                    let acc = rs_verify_accept(
                        &draft_u32,
                        &draft_arr,
                        &q_probs,
                        &verify_logits,
                        params,
                        block_size,
                    )?;
                    let a_ms = t_accept.elapsed().as_secs_f64() * 1000.0;
                    (0.0, a_ms, acc)
                }
            };
            let verify_ms = verify_fwd_ms + verify_argmax_ms;
            let n_accepted = accepted.len() as i32;

            // j. Partial accept — GDN-only replay from tape (no full rerun)
            //    Tape-recording verify already advanced state for ALL positions.
            //    On partial rejection: restore GDN snapshots, replay only accepted
            //    positions through the cheap tape kernel, rollback KV for rejected.
            let t_replay = Instant::now();
            if n_accepted < block_size {
                let kv_rollback = verify_len - n_accepted;
                model
                    .replay_tape_rollback(&layer_tapes, &mut cache, n_accepted, kv_rollback)
                    .map_err(EngineError::Mlx)?;
                // Batch-eval all replayed GDN states in one call
                let replay_states: Vec<&Array> = cache
                    .as_hybrid()
                    .iter()
                    .filter_map(|lc| match lc {
                        Some(higgs_models::qwen3_next::LayerCache::Arrays(ac)) => {
                            ac.ssm_state.as_ref()
                        }
                        _ => None,
                    })
                    .collect();
                if !replay_states.is_empty() {
                    eval(replay_states).map_err(EngineError::Mlx)?;
                }
            }
            let replay_ms = t_replay.elapsed().as_secs_f64() * 1000.0;

            // Taps: slice verify taps to accepted positions (valid for both
            // full and partial accept — causal model, earlier positions don't
            // depend on later ones)
            let t_tap_slice = Instant::now();
            current_taps = verify_taps
                .into_iter()
                .map(|tap| tap.index((.., ..n_accepted, ..)))
                .collect();
            let tap_slice_ms = t_tap_slice.elapsed().as_secs_f64() * 1000.0;

            // k. Update state
            let t_end = Instant::now();
            for &tok in &accepted {
                tokens.push(tok);
            }
            last_token = *accepted.last().expect("accept_prefix always returns >= 1") as i32;
            start += n_accepted;
            let end_ms = t_end.elapsed().as_secs_f64() * 1000.0;
            if kv_debug {
                eprintln!(
                    "KV_ROUND round={} n_accepted={}/{} post_accept_start={} cpu_cache_len={}",
                    round_idx + 1,
                    n_accepted,
                    block_size,
                    start,
                    cpu_cache.as_ref().map_or(0, |c| c.len),
                );
            }

            // Pipeline: spawn next round's CPU drafter in background.
            // The drafter runs entirely on CPU (BLAS/ANE), so it overlaps with
            // the GPU snapshot+verify in the next iteration, saving ~47ms/round.
            if pipeline {
                // Build next round's block tokens
                let mut next_block = vec![mask_id; block_size as usize];
                next_block[0] = last_token;
                let next_block_ids = Array::from_slice(&next_block, &[1, block_size]);

                // Embed (GPU, fast <1ms) + eval taps
                let next_noise = model
                    .embed_token_ids(&next_block_ids)
                    .map_err(EngineError::Mlx)?;
                let mut to_eval: Vec<&Array> = vec![&next_noise];
                to_eval.extend(current_taps.iter());
                eval(to_eval).map_err(EngineError::Mlx)?;

                // Convert to f32 for CPU drafter
                let noise_f32: Vec<f32> = next_noise
                    .as_dtype(Dtype::Float32)
                    .map_err(EngineError::Mlx)?
                    .reshape(&[-1])
                    .map_err(EngineError::Mlx)?
                    .as_slice::<f32>()
                    .to_vec();
                let taps_f32: Vec<Vec<f32>> = current_taps
                    .iter()
                    .map(|t| {
                        t.as_dtype(Dtype::Float32)
                            .unwrap()
                            .reshape(&[-1])
                            .unwrap()
                            .as_slice::<f32>()
                            .to_vec()
                    })
                    .collect();
                let ctx_len = current_taps[0].shape()[1] as usize;
                // Pipeline=1 requires cpu_engine to have been loaded.
                let pipe_engine = dflash
                    .cpu_engine
                    .as_ref()
                    .expect("HIGGS_DFLASH_PIPELINE=1 but CPU engine was not loaded");
                let fresh_cache = pipe_engine.make_cache();
                let mut draft_cache = cpu_cache
                    .replace(fresh_cache)
                    .unwrap_or_else(|| pipe_engine.make_cache());

                let (tx, rx) = std::sync::mpsc::channel();

                // Spawn drafter on a background thread so it overlaps with the
                // next round's GPU snapshot+verify (~47 ms/round saved).
                //
                // ANE path: the worker handle is `Send + Sync`, so we can move
                // a clone into the spawned thread; the handle forwards the
                // request to the pinned ANE worker thread (which owns the
                // `!Send` executor) and returns the cache alongside the output.
                //
                // CPU path: move an `Arc<DFlashCpuEngine>` clone into the
                // spawned thread as before.
                #[cfg(feature = "ane")]
                let ane_worker = dflash.ane_worker.clone();
                let cpu_engine = Arc::clone(pipe_engine);

                std::thread::spawn(move || {
                    let out_f32_and_cache = {
                        #[cfg(feature = "ane")]
                        {
                            if let Some(worker) = ane_worker {
                                let (out, cache_back) =
                                    worker.forward(noise_f32, taps_f32, ctx_len, draft_cache);
                                (out, cache_back)
                            } else {
                                let tap_slices: Vec<&[f32]> =
                                    taps_f32.iter().map(Vec::as_slice).collect();
                                let out = cpu_engine.forward(
                                    &noise_f32,
                                    &tap_slices,
                                    ctx_len,
                                    &mut draft_cache,
                                );
                                (out, draft_cache)
                            }
                        }
                        #[cfg(not(feature = "ane"))]
                        {
                            let tap_slices: Vec<&[f32]> =
                                taps_f32.iter().map(Vec::as_slice).collect();
                            let out = cpu_engine.forward(
                                &noise_f32,
                                &tap_slices,
                                ctx_len,
                                &mut draft_cache,
                            );
                            (out, draft_cache)
                        }
                    };
                    let _ = tx.send(out_f32_and_cache);
                });
                pending_draft = Some(rx);
            }

            // Trace logging
            round_idx += 1;
            total_accepted += n_accepted as u64;
            if trace {
                let round_ms = t_round.elapsed().as_secs_f64() * 1000.0;
                let avg_accept = total_accepted as f64 / f64::from(round_idx);
                let elapsed_s = t_start.elapsed().as_secs_f64();
                let eff_tps = if elapsed_s > 0.0 {
                    tokens.len() as f64 / elapsed_s
                } else {
                    0.0
                };
                tracing::info!(
                    "dflash_trace round={} embed={:.1}ms draft={:.1}ms lm_draft={:.1}ms(slice={:.1}ms logits={:.1}ms argmax_build={:.1}ms transfer={:.1}ms) \
                     gap={:.1}ms verify_build={:.1}ms verify={:.1}ms verify_fwd={:.1}ms verify_argmax={:.1}ms \
                     no_tape_fwd={:.1}ms accept={:.1}ms replay={:.1}ms tap_slice={:.1}ms end={:.1}ms \
                     round_total={:.1}ms accepted={} avg_accept={:.1} eff_tps={:.1}",
                    round_idx,
                    embed_ms,
                    draft_ms,
                    lm_draft_ms,
                    slice_ms,
                    logits_ms,
                    argmax_build_ms,
                    draft_transfer_ms,
                    gap_ms,
                    verify_build_ms,
                    verify_ms,
                    verify_fwd_ms,
                    verify_argmax_ms,
                    no_tape_fwd_ms.unwrap_or(0.0),
                    accept_ms,
                    replay_ms,
                    tap_slice_ms,
                    end_ms,
                    round_ms,
                    n_accepted,
                    avg_accept,
                    eff_tps,
                );
            }

            // l. Check termination
            let completion_len = Self::completion_len(&tokens)?;

            if tokens.iter().any(|t| self.eos_token_ids.contains(t)) {
                let elapsed = t_start.elapsed();
                tracing::info!(
                    tokens = tokens.len(),
                    rounds = tokens.len(), // approximate
                    tok_per_sec = format!("{:.1}", tokens.len() as f64 / elapsed.as_secs_f64()),
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
                // See sync-path comment above — windowed tail decode for
                // O(1)-per-step stop detection, full decode only on hit.
                const STOP_CHECK_WINDOW: usize = 64;
                let tail_start = tokens.len().saturating_sub(STOP_CHECK_WINDOW);
                let tail = self.decode_tokens(&tokens[tail_start..])?;
                if check_stop_sequences(&tail, stop_sequences).is_some() {
                    let full = self.decode_tokens(&tokens)?;
                    if let Some(truncated) = check_stop_sequences(&full, stop_sequences) {
                        return Ok(GenerationOutput {
                            text: truncated,
                            finish_reason: "stop".to_owned(),
                            prompt_tokens: prompt_len,
                            completion_tokens: completion_len,
                            token_logprobs: None,
                        });
                    }
                }
            }

            if completion_len >= max_tokens {
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

    /// DFlash speculative decode — streaming variant.
    /// Same draft-verify loop as `generate_dflash_inner` but streams accepted
    /// tokens through the `sender` channel for SSE responses.
    #[allow(
        clippy::too_many_lines,
        clippy::significant_drop_tightening,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn generate_dflash_streaming_inner(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
    ) -> Result<(), EngineError> {
        let dflash = self.dflash.as_ref().expect("DFlash state must be Some");
        let prompt_len = Self::prompt_len(prompt_tokens)?;
        let has_stop_sequences = !stop_sequences.is_empty();

        let mut model = self
            .model
            .lock()
            .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;

        let mut cache = model
            .make_cache_with_config(self.kv_cache_config)
            .map_err(EngineError::Mlx)?;

        // Prefill with taps
        let prompt_array = Array::from(prompt_tokens).index(NewAxis);
        let (prefill_logits, taps) = model
            .forward_with_taps(&prompt_array, None, &mut cache, &dflash.tap_layers)
            .map_err(EngineError::Mlx)?;
        eval([&prefill_logits]).map_err(EngineError::Mlx)?;

        // Sample first token
        let last_logits = prefill_logits.index((.., -1, ..));
        let first_token = sample(&last_logits, params).map_err(EngineError::Mlx)?;
        eval([&first_token]).map_err(EngineError::Mlx)?;

        let first_token_id: u32 = first_token.item();
        let mut tokens: Vec<u32> = vec![first_token_id];
        let mut prev_decoded_len: usize = 0;

        // Stream first token
        let full_text = self.decode_tokens(&tokens)?;
        let new_text = full_text[prev_decoded_len..].to_owned();
        prev_decoded_len = full_text.len();

        let is_eos = self.eos_token_ids.contains(&first_token_id);
        let is_done = is_eos || max_tokens <= 1;
        let finish_reason = if is_eos {
            Some("stop".to_owned())
        } else if max_tokens <= 1 {
            Some("length".to_owned())
        } else {
            None
        };

        if sender
            .blocking_send(StreamingOutput {
                new_text,
                finished: is_done,
                finish_reason,
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprob: None,
            })
            .is_err()
        {
            return Ok(());
        }

        if is_done {
            return Ok(());
        }

        let mut current_taps = taps;
        let mut drafter_cache = dflash.gpu_drafter.lock().unwrap().make_cache();
        let mut last_token = first_token_id as i32;
        let mut start = prompt_len as i32;
        let block_size = dflash.block_size;
        let mask_id = dflash.mask_token_id;
        let verify_layer_chunk: usize = std::env::var("HIGGS_DFLASH_LAYER_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if model.num_layers() > 32 { 16 } else { 0 });
        let trace = std::env::var("HIGGS_DFLASH_TRACE").is_ok_and(|v| v == "1");
        let t_start = Instant::now();
        let mut round_idx: u32 = 0;
        let mut total_accepted: u64 = 0;

        loop {
            let t_round = Instant::now();
            let t_embed = Instant::now();
            // Build block + embed + draft (synchronous, no pipeline for streaming simplicity)
            let mut block_tokens = vec![mask_id; block_size as usize];
            block_tokens[0] = last_token;
            let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);

            let noise_embedding = model
                .embed_token_ids(&block_ids)
                .map_err(EngineError::Mlx)?;
            let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;

            let t_draft = Instant::now();
            // GPU drafter forward — all MLX ops, no CPU↔GPU transfer.
            // Mirrors generate_dflash_inner's non-pipeline branch.
            let draft_hidden = dflash
                .gpu_drafter
                .lock()
                .unwrap()
                .forward(&noise_embedding, &current_taps, &mut drafter_cache)
                .map_err(EngineError::Mlx)?;
            let draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;

            let t_lm_draft = Instant::now();
            // Draft lm_head: greedy argmax (temp=0) or rejection sampling (temp>0)
            let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
            let draft_logits = model
                .forward_all_logits_from_hidden(&draft_hidden_sliced)
                .map_err(EngineError::Mlx)?;
            let (draft_u32, draft_i32, rs_state) = if params.temperature == 0.0 {
                let draft_token_arr =
                    mlx_rs::argmax_axis!(draft_logits, -1).map_err(EngineError::Mlx)?;
                eval([&draft_token_arr]).map_err(EngineError::Mlx)?;
                let du32: Vec<u32> = draft_token_arr
                    .reshape(&[-1])
                    .map_err(EngineError::Mlx)?
                    .as_slice::<u32>()
                    .to_vec();
                let di32: Vec<i32> = du32.iter().map(|&x| x as i32).collect();
                (du32, di32, None)
            } else {
                let (du32, di32, arr, q) = rs_draft_sample(&draft_logits, params)?;
                (du32, di32, Some((arr, q)))
            };

            let lm_draft_ms = t_lm_draft.elapsed().as_secs_f64() * 1000.0;

            let t_verify = Instant::now();
            // Verify
            let mut verify_tokens = vec![last_token];
            verify_tokens.extend_from_slice(&draft_i32);
            let verify_len = verify_tokens.len() as i32;
            let verify_input = Array::from_slice(&verify_tokens, &[1, verify_len]);

            // Drop draft intermediates before verify to free Metal buffers
            drop(draft_hidden);

            let (verify_logits, verify_taps, layer_tapes) = if verify_layer_chunk > 0 {
                model
                    .forward_with_taps_tape_chunked(
                        &verify_input,
                        None,
                        &mut cache,
                        &dflash.tap_layers,
                        verify_layer_chunk,
                    )
                    .map_err(EngineError::Mlx)?
            } else {
                model
                    .forward_with_taps_tape(&verify_input, None, &mut cache, &dflash.tap_layers)
                    .map_err(EngineError::Mlx)?
            };
            let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;

            let t_accept = Instant::now();
            // Accept: greedy argmax+accept_prefix (temp=0) or rejection sampling (temp>0)
            let accepted = match rs_state {
                None => {
                    let verify_argmax =
                        mlx_rs::argmax_axis!(verify_logits, -1).map_err(EngineError::Mlx)?;
                    eval([&verify_argmax]).map_err(EngineError::Mlx)?;
                    let verify_flat: Vec<u32> = verify_argmax
                        .reshape(&[-1])
                        .map_err(EngineError::Mlx)?
                        .as_slice::<u32>()
                        .to_vec();
                    accept_prefix(&draft_u32, &verify_flat)
                }
                Some((draft_arr, q_probs)) => rs_verify_accept(
                    &draft_u32,
                    &draft_arr,
                    &q_probs,
                    &verify_logits,
                    params,
                    block_size,
                )?,
            };
            let n_accepted = accepted.len() as i32;
            let accept_ms = t_accept.elapsed().as_secs_f64() * 1000.0;

            let t_replay = Instant::now();
            // Tape replay on partial rejection
            if n_accepted < block_size {
                let kv_rollback = verify_len - n_accepted;
                model
                    .replay_tape_rollback(&layer_tapes, &mut cache, n_accepted, kv_rollback)
                    .map_err(EngineError::Mlx)?;
                let replay_states: Vec<&Array> = cache
                    .as_hybrid()
                    .iter()
                    .filter_map(|lc| match lc {
                        Some(higgs_models::qwen3_next::LayerCache::Arrays(ac)) => {
                            ac.ssm_state.as_ref()
                        }
                        _ => None,
                    })
                    .collect();
                if !replay_states.is_empty() {
                    eval(replay_states).map_err(EngineError::Mlx)?;
                }
            }

            let replay_ms = t_replay.elapsed().as_secs_f64() * 1000.0;

            current_taps = verify_taps
                .into_iter()
                .map(|tap| tap.index((.., ..n_accepted, ..)))
                .collect();

            // Stream accepted tokens
            for &tok in &accepted {
                tokens.push(tok);
            }
            last_token = *accepted.last().expect("accept_prefix always returns >= 1") as i32;
            start += n_accepted;

            round_idx += 1;
            total_accepted += n_accepted as u64;
            if trace {
                let round_ms = t_round.elapsed().as_secs_f64() * 1000.0;
                let avg_accept = total_accepted as f64 / f64::from(round_idx);
                let elapsed_s = t_start.elapsed().as_secs_f64();
                let eff_tps = if elapsed_s > 0.0 {
                    tokens.len() as f64 / elapsed_s
                } else {
                    0.0
                };
                tracing::info!(
                    "dflash_stream_trace round={round_idx} embed={embed_ms:.1}ms \
                     draft={draft_ms:.1}ms lm_draft={lm_draft_ms:.1}ms \
                     verify={verify_ms:.1}ms accept={accept_ms:.1}ms \
                     replay={replay_ms:.1}ms round_total={round_ms:.1}ms \
                     accepted={n_accepted} avg_accept={avg_accept:.2} eff_tps={eff_tps:.1}"
                );
            }

            let completion_len = Self::completion_len(&tokens)?;
            let full_text = self.decode_tokens(&tokens)?;
            let new_text = full_text[prev_decoded_len..].to_owned();
            let old_decoded_len = prev_decoded_len;
            prev_decoded_len = full_text.len();

            let (final_new_text, hit_stop_seq) = if !has_stop_sequences {
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

            let is_eos = tokens.iter().any(|t| self.eos_token_ids.contains(t));
            let is_max = completion_len >= max_tokens;
            let step_finished = is_eos || is_max || hit_stop_seq;

            let finish_reason = if is_eos || hit_stop_seq {
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
                    token_logprob: None,
                })
                .is_err()
            {
                return Ok(());
            }

            if step_finished {
                break;
            }
        }
        Ok(())
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
                graph_build_ms = format!("{:.2}", forward_ns as f64 / s / 1e6),
                device_eval_ms = format!("{:.2}", eval_ns as f64 / s / 1e6),
                forward_ms = format!("{:.2}", forward_ns as f64 / s / 1e6),
                eval_ms = format!("{:.2}", eval_ns as f64 / s / 1e6),
                item_ms = format!("{:.2}", item_ns as f64 / s / 1e6),
                other_ms = format!("{:.2}", other_ns as f64 / s / 1e6),
                total_ms = format!(
                    "{:.2}",
                    (forward_ns + eval_ns + item_ns + other_ns) as f64 / s / 1e6
                ),
                "Decode loop timing (per step avg; MLX forward builds a lazy graph, eval runs it)"
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
        // DFlash speculative decoding: use draft-verify streaming when available.
        if self.dflash.is_some() && constraint.is_none() && pixel_values.is_none() {
            return self.generate_dflash_streaming_inner(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                sender,
            );
        }

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
            if let Some(ref lp) = next_logprob_data {
                eval_targets.extend(lp.eval_targets());
            }
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
                if let Some(ref lp) = following_logprob_data {
                    eval_targets.extend(lp.eval_targets());
                }
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

            let token_logprob = next_logprob_data
                .as_ref()
                .map(|lp_data| lp_data.materialize(token_id));

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

/// Stochastic draft sampling for DFlash speculative decoding.
///
/// Computes the drafter's probability distribution `q` from logits and samples
/// one token per position. Returns host copies (for building verify_input) and
/// the on-device arrays needed by the rejection-sampling accept step.
///
/// Used in both DFlash loops when `params.temperature > 0`. For temperature=0
/// callers keep the greedy argmax path.
fn rs_draft_sample(
    draft_logits: &Array,
    params: &SamplingParams,
) -> Result<(Vec<u32>, Vec<i32>, Array, Array), EngineError> {
    // draft_logits: [1, K-1, V] → q_probs: [1, K-1, V] (normalized, vocab order)
    let q_probs = compute_probs(draft_logits, params).map_err(EngineError::Mlx)?;
    let draft_tokens_arr = sample_from_probs(&q_probs).map_err(EngineError::Mlx)?; // [1, K-1]

    // Only draft tokens need host transfer now (to build verify_input).
    // q_probs + draft_tokens_arr stay on device for the accept phase.
    eval([&draft_tokens_arr]).map_err(EngineError::Mlx)?;

    let draft_u32: Vec<u32> = draft_tokens_arr
        .reshape(&[-1])
        .map_err(EngineError::Mlx)?
        .as_slice::<u32>()
        .to_vec();
    let draft_i32: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

    Ok((draft_u32, draft_i32, draft_tokens_arr, q_probs))
}

/// Rejection-sampling accept for DFlash (Leviathan et al. 2023, Algorithm 2).
///
/// Given the drafter's stochastic draft + its `q` distribution and the target's
/// verify logits, decides how many tokens to accept via min(1, p/q) and emits
/// either a residual sample (on first rejection) or a bonus sample (on full
/// acceptance). Returns the accepted prefix (length 1..=block_size).
///
/// Called after the verify forward pass. The greedy path (temp=0) should use
/// `accept_prefix` on the verify argmax instead.
fn rs_verify_accept(
    draft_u32: &[u32],
    draft_tokens_arr: &Array,
    q_probs: &Array,
    verify_logits: &Array,
    params: &SamplingParams,
    block_size: i32,
) -> Result<Vec<u32>, EngineError> {
    use mlx_rs::ops::maximum;
    use rand::Rng;

    let k_minus_1 = block_size - 1;

    // Target's probability distribution: [1, K, V]
    let verify_probs = compute_probs(verify_logits, params).map_err(EngineError::Mlx)?;

    // Align first K-1 verify positions with the K-1 draft positions.
    // .index() returns bare Array (not Result).
    let verify_probs_for_drafts = verify_probs.index((.., 0..k_minus_1, ..)); // [1, K-1, V]

    // Build gather index: [1, K-1, 1]. expand_dims takes a single i32.
    let draft_idx = draft_tokens_arr.expand_dims(-1).map_err(EngineError::Mlx)?;

    // Gather q(drafted_i) at drafted positions — [1, K-1, 1] → squeeze → [1, K-1].
    let q_at_draft = q_probs
        .take_along_axis(&draft_idx, -1)
        .map_err(EngineError::Mlx)?
        .squeeze_axes(&[-1])
        .map_err(EngineError::Mlx)?;

    // Gather p(drafted_i) similarly
    let p_at_draft = verify_probs_for_drafts
        .take_along_axis(&draft_idx, -1)
        .map_err(EngineError::Mlx)?
        .squeeze_axes(&[-1])
        .map_err(EngineError::Mlx)?;

    // Residual distribution: (p - q)_+ normalized per row
    let diff = verify_probs_for_drafts
        .subtract(q_probs)
        .map_err(EngineError::Mlx)?;
    let zero = Array::from_slice(&[0.0_f32], &[1]);
    let residual_raw = maximum(&diff, &zero).map_err(EngineError::Mlx)?; // [1, K-1, V]

    let residual_sum = residual_raw
        .sum_axes(&[-1], true)
        .map_err(EngineError::Mlx)?; // [1, K-1, 1]
    let eps = Array::from_slice(&[1e-10_f32], &[1]);
    let residual_sum_safe = residual_sum.add(&eps).map_err(EngineError::Mlx)?;
    let residual_normalized = residual_raw
        .divide(&residual_sum_safe)
        .map_err(EngineError::Mlx)?; // broadcasts [1, K-1, 1] → [1, K-1, V]

    // One residual token per draft position
    let residual_tokens = sample_from_probs(&residual_normalized).map_err(EngineError::Mlx)?; // [1, K-1]

    // Bonus: sample from verify distribution at position K-1
    let bonus_slice = verify_probs.index((.., k_minus_1..k_minus_1 + 1, ..)); // [1, 1, V]
    let bonus_token_arr = sample_from_probs(&bonus_slice).map_err(EngineError::Mlx)?; // [1, 1]

    // Single eval — batch all host-bound tensors
    eval([&q_at_draft, &p_at_draft, &residual_tokens, &bonus_token_arr])
        .map_err(EngineError::Mlx)?;

    // Host transfer
    let q_f32: Vec<f32> = q_at_draft
        .reshape(&[-1])
        .map_err(EngineError::Mlx)?
        .as_slice::<f32>()
        .to_vec();
    let p_f32: Vec<f32> = p_at_draft
        .reshape(&[-1])
        .map_err(EngineError::Mlx)?
        .as_slice::<f32>()
        .to_vec();
    let residual_u32: Vec<u32> = residual_tokens
        .reshape(&[-1])
        .map_err(EngineError::Mlx)?
        .as_slice::<u32>()
        .to_vec();
    let bonus_flat = bonus_token_arr.reshape(&[-1]).map_err(EngineError::Mlx)?;
    let bonus_u32: u32 = bonus_flat.as_slice::<u32>()[0];

    // CPU uniforms — one per draft slot. Cheap vs. device RNG round-trip.
    let mut rng = rand::rng();
    let rand_uniform: Vec<f32> = (0..draft_u32.len()).map(|_| rng.random::<f32>()).collect();

    Ok(accept_prefix_rs(
        draft_u32,
        &q_f32,
        &p_f32,
        &residual_u32,
        bonus_u32,
        &rand_uniform,
    ))
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
