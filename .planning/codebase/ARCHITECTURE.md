# Architecture

**Analysis Date:** 2026-04-03

## Pattern Overview

**Overall:** Unified inference gateway with modular inference engine and request router.

**Key Characteristics:**
- Multi-engine dispatch pattern: routes requests to local MLX-based engines or remote providers
- Layered architecture: CLI/config → HTTP API (Axum) → router → inference engines → model implementations
- Dual engine modes: SimpleEngine (serialized, single-request) and BatchEngine (interleaved, continuous batching)
- Abstraction over model architectures: AnyModel enum dispatches across 11+ model types
- Paged KV cache with continuous batching scheduler for efficient memory management

## Layers

**Configuration & CLI:**
- Purpose: Parse arguments, validate config, manage daemon lifecycle
- Location: `crates/higgs/src/config.rs`, `crates/higgs/src/daemon.rs`, `crates/higgs/src/doctor.rs`
- Contains: Config schema, CLI parsers (clap), profile management, validation rules
- Depends on: std, clap, figment (config merging)
- Used by: main.rs → cmd_serve/cmd_start/cmd_stop/cmd_init

**HTTP API Layer:**
- Purpose: Accept requests in OpenAI/Anthropic formats, translate and route them
- Location: `crates/higgs/src/routes/` (chat.rs, completions.rs, anthropic.rs, embeddings.rs)
- Contains: HTTP handlers, request/response types, streaming endpoints
- Depends on: Axum, tokio, serde_json, Engine trait
- Used by: build_router in lib.rs

**Router (Request Dispatch):**
- Purpose: Map model names to destination (local engine, remote provider, or auto-route)
- Location: `crates/higgs/src/router.rs`
- Contains: RoutingMethod enum, ResolvedRoute variants, regex pattern matching, auto-router AI classification
- Depends on: Local engine map, provider config, regex patterns
- Used by: Route handlers access via AppState

**Inference Engine Interface:**
- Purpose: Unify SimpleEngine and BatchEngine behind a single enum
- Location: `crates/higgs/src/state.rs` (Engine enum)
- Contains: Engine::Simple/Batch wrapper, delegation to concrete implementations
- Depends on: higgs_engine crate (SimpleEngine, BatchEngine)
- Used by: All route handlers

**SimpleEngine:**
- Purpose: Single-request inference with paged KV cache and optional prefix cache
- Location: `crates/higgs-engine/src/simple.rs`
- Contains: Model loading, tokenization, generation loop, streaming, session management
- Depends on: higgs_models (AnyModel), mlx_rs, Tokenizer, PagedKvCache, RoundRobinScheduler
- Key subsystems:
  - PrefixCache: Legacy prefix caching (30-46x TTFT speedup on cache hit)
  - PagedKvCache: Paged memory allocation for KV states across sessions
  - RoundRobinScheduler: Continuous batching scheduler for multi-session workflows
  - SpecPrefillEngine: Sparse prefill optimization for large prompt tokens

**BatchEngine:**
- Purpose: Experimental batched inference (currently placeholder)
- Location: `crates/higgs-engine/src/batch_engine.rs`
- Contains: Batch scheduling, request queueing
- Depends on: AnyModel, similar to SimpleEngine
- Used by: Alternate Engine variant when batch=true in config

**Model Abstraction Layer:**
- Purpose: Dispatch to correct architecture (Qwen3.5, Llama, etc.) via AnyModel enum
- Location: `crates/higgs-models/src/lib.rs` (AnyModel and AnyCache enums)
- Contains:
  - Transformer (standard: Llama, Mistral, Qwen2, Qwen3)
  - Qwen3Next (hybrid SSM+attention with MoE)
  - Qwen3Moe (sparse MoE)
  - Gemma2, Phi3, Starcoder2, etc.
- Depends on: MLX-RS compute kernels, quantization schemes
- Used by: SimpleEngine.forward()/forward_streaming()

**Model Implementations:**
- Purpose: Architecture-specific forward passes, KV cache handling
- Location: `crates/higgs-models/src/{transformer, qwen3_next, qwen3_moe, gemma2, phi3, starcoder2}.rs`
- Contains: Layer stacking, attention/MoE dispatch, quantized matmul, activation functions
- Depends on: MLX-RS ops (matmul, gather, softmax), custom GEMV kernels
- Key pattern: Each model exposes forward() → Array and forward_lm_head() → logits

**Cache Subsystem:**
- Purpose: Efficient KV state management with paging and block allocation
- Location: `crates/higgs-engine/src/cache/` (paged.rs, allocator.rs, pagetable.rs, storage.rs)
- Contains:
  - PagedKvCache: Manages sessions, allocates/frees blocks, handles gather/write operations
  - BlockAllocator: Free list tracking for paged blocks
  - PageTable: Maps logical session positions to physical block IDs
  - CpuKvStorage: CPU-side KV buffer management
- Depends on: mlx_rs Array for GPU gather/write ops
- Used by: SimpleEngine.run_prefill() and decode loop

**Scheduler:**
- Purpose: Round-robin scheduling across active sessions for continuous batching
- Location: `crates/higgs-engine/src/scheduler/round_robin.rs`
- Contains: Session ordering, batch composition, idle tracking
- Depends on: Session struct (id, tokens, finished flag, max_tokens)
- Used by: SimpleEngine.step() for each generation token

**Chat Template Renderer:**
- Purpose: Format messages as prompt tokens according to model-specific template
- Location: `crates/higgs-engine/src/chat_template.rs`
- Contains: Jinja2 template loading/rendering, message formatting
- Depends on: minijinja (Jinja2 implementation)
- Used by: Route handlers to convert /chat/completions to token stream

**Constrained Generation:**
- Purpose: Enforce structured output (JSON, regex) via guided sampling
- Location: `crates/higgs-engine/src/constrained.rs`
- Contains: Regex/JSON constraint compilation, token filtering
- Depends on: regex crate
- Used by: SimpleEngine.generate() when constraint specified

**Model Loader:**
- Purpose: Load model weights and tokenizer from MLX safetensors format
- Location: `crates/higgs-engine/src/model_loader.rs`
- Contains: Config.json parsing, weight loading, tokenizer instantiation
- Depends on: mlx_rs, serde_json, tokenizers crate
- Used by: SimpleEngine::load() and BatchEngine::load()

## Data Flow

**Single-Request Generation (http request → response):**

1. HTTP handler receives POST /v1/chat/completions (AppState has Router)
2. Router resolves model name to Engine (local or Remote provider)
3. Route handler calls Engine.prepare_chat_prompt() → token IDs
4. Route handler calls Engine.generate(tokens, max_tokens, sampling_params, ...)
5. SimpleEngine.generate() acquires lock on AnyModel
6. Tokenizer processes any prefix cache hits
7. Model.forward(prompt_tokens) → logits for prefill phase
8. Sampling loop: sample token → apply penalties → check stop conditions
9. Model.forward(single_token) → logits for decode phase (repeat until max_tokens or stop)
10. Collect final text and return GenerationOutput

**Streaming Generation (http request → stream of Server-Sent Events):**

1-4. Same as single-request
5. Route handler calls Engine.generate_streaming(tx) passing mpsc channel
6. SimpleEngine spawns generation task, sends StreamingOutput to channel for each token
7. HTTP handler yields chunks as they arrive

**Continuous Batching (multiple concurrent requests):**

1. Multiple http requests arrive for same or different models
2. Each maps to a Session (id, tokens, finished flag)
3. Scheduler (RoundRobinScheduler) orders sessions for next token
4. SimpleEngine.step() processes one token for active sessions
5. PagedKvCache allocates blocks per-session, gathers appropriate KV slices
6. Model.forward(batched_tokens) → batched logits
7. Per-session sampling → per-session tokens → update sessions
8. Return batch of new tokens to clients via streaming channels

**Request Routing:**

1. Router.resolve(model_name) checks in order:
2. If model=="auto": use auto_router AI classification (if enabled)
3. Pattern matching: iterate compiled_routes, return first regex match
4. Direct lookup in local_engines HashMap
5. Default provider fallback (Remote or error)

**State Management:**

- AppState (Arc-wrapped): shared across all http handlers
  - router: model name → engine map
  - config: all server configuration
  - http_client: for proxying to remote providers
  - metrics: optional performance tracking

- SimpleEngine (Arc-wrapped per engine):
  - model: Mutex<AnyModel> to serialize forward passes
  - paged_cache: Mutex<PagedKvCache> for multi-session KV management
  - scheduler: Mutex<RoundRobinScheduler> for token ordering
  - sessions: HashMap of active generation sessions
  - prefix_cache: Legacy single-request cache (backward compatible)

## Key Abstractions

**Engine Trait (unified interface):**
- Purpose: Abstract over SimpleEngine vs BatchEngine implementations
- Examples: `crates/higgs/src/state.rs` Engine enum (lines 19-213)
- Pattern: Match on enum variant, delegate to concrete impl

**AnyModel Enum (architecture dispatch):**
- Purpose: Dispatch to correct model forward() based on loaded weights
- Examples: `crates/higgs-models/src/lib.rs` (lines 89-101)
- Pattern: Detect model_type from config.json, instantiate correct variant
- Usage: model.forward(tokens, cache) → Array of logits

**AnyCache Enum (cache variant):**
- Purpose: Hold cache state appropriate for model architecture
- Examples: `crates/higgs-models/src/lib.rs` (lines 81-86)
- Pattern: KV variant for standard transformers, Hybrid variant for Qwen3Next (SSM+attn)

**ResolvedRoute Enum (routing outcome):**
- Purpose: Represent local vs remote destination
- Examples: `crates/higgs/src/router.rs` (lines 24-42)
- Pattern: Higgs variant contains Arc<Engine>, Remote variant contains provider URL

**Session Struct (continuous batching):**
- Purpose: Track generation state per concurrent request
- Examples: `crates/higgs-engine/src/simple.rs` (lines 77-83)
- Pattern: Created per /v1/chat/completions request, deleted when finished

## Entry Points

**CLI Main:**
- Location: `crates/higgs/src/main.rs` (tokio::main)
- Triggers: Binary execution
- Responsibilities: Parse CLI, dispatch to command handlers, init tracing

**HTTP Server Start:**
- Location: `crates/higgs/src/main.rs` cmd_serve() (lines 120-206)
- Triggers: `higgs serve` or `higgs start` commands
- Responsibilities: Load config, instantiate engines, build Axum router, bind TCP listener

**Route Handlers:**
- Location: `crates/higgs/src/routes/{chat, completions, anthropic, embeddings}.rs`
- Triggers: POST /v1/chat/completions, /v1/messages, etc.
- Responsibilities: Parse request, call router to resolve model, invoke engine, format response

**Daemon Process:**
- Location: `crates/higgs/src/daemon.rs`
- Triggers: `higgs start` command
- Responsibilities: Fork background process, write PID file, detach stdio

## Error Handling

**Strategy:** Custom error enums with thiserror derive, propagate via Result<T, E>

**Patterns:**
- EngineError (higgs_engine): Generation, ModelLoad, Cache variants with context
- ModelError (higgs_models): UnsupportedModel, InvalidWeights, Computation
- ConfigError (implied from main.rs error handling): Invalid config values, missing files
- CacheError (cache module): OutOfBlocks, InvalidBlockId, WriteOutOfBounds for paged cache

**HTTP Error Mapping:**
- EngineError::Generation → 500 Internal Server Error + error message in response
- Model not found → 404 Not Found
- Rate limit hit → 429 Too Many Requests
- Timeout → 504 Gateway Timeout

## Cross-Cutting Concerns

**Logging:**
- Framework: tracing crate with tracing_subscriber
- Patterns: info!() for major events (model load, server start), debug!() for detailed flow, warn!() for recoverable issues
- Example: `crates/higgs-engine/src/simple.rs` line 132 logs model_dir on load

**Validation:**
- Approach: doctor command validates entire config before server start
- Location: `crates/higgs/src/doctor.rs`
- Pattern: Check model paths exist, probe remote providers, verify sampler config

**Authentication:**
- Bearer token validation via tower-http ValidateRequestHeaderLayer if api_key set in config
- Rate limiting via governor crate (per IP address, configurable RPM)

**Metrics:**
- Optional collection when config mode (not simple mode)
- Location: `crates/higgs/src/metrics.rs`
- Tracks: requests per endpoint, tokens per minute, error rates, generation latency

---

*Architecture analysis: 2026-04-03*
