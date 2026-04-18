# Architecture

**Analysis Date:** 2026-04-18

## Pattern Overview

**Overall:** Unified inference gateway with request routing to local MLX models and remote API providers. Implements a three-layer modular architecture:

1. **Gateway layer** (`higgs`) - HTTP API, routing decisions, metrics
2. **Engine layer** (`higgs-engine`) - Inference execution, tokenization, caching
3. **Models layer** (`higgs-models`) - Model architectures, computations, quantization

**Key Characteristics:**
- Multimodel inference: multiple local models or remote provider routes via single HTTP endpoint
- Dual execution modes: `SimpleEngine` (serialized requests, lower latency) and `BatchEngine` (concurrent request interleaving)
- Speculative decoding: optional DFlash drafter for faster token generation
- Hardware acceleration: MLX (Apple Silicon optimized) with optional ANE (Apple Neural Engine) offload for specific layers
- Format-agnostic routing: OpenAI-compatible API but can proxy to Anthropic, other providers

## Layers

**Gateway (`crates/higgs/`):**
- Purpose: HTTP request handling, model routing, metrics collection, daemon/TUI management
- Location: `crates/higgs/src/`
- Contains: HTTP routes, config parsing, model discovery, CLI interface
- Depends on: `higgs-engine`, `higgs-models`, Axum/Tokio for HTTP, MLX for memory info
- Used by: Clients making HTTP requests (OpenAI-compatible, Anthropic-compatible)

**Engine (`crates/higgs-engine/`):**
- Purpose: Inference loop orchestration, tokenization, KV cache management, streaming output
- Location: `crates/higgs-engine/src/`
- Contains: SimpleEngine (mutex-serialized), BatchEngine (dedicated background thread), chat templates, prefix cache, token streaming
- Depends on: `higgs-models` for model implementations, MLX for computation, tokenizers for encoding/decoding
- Used by: Gateway routes for all inference operations

**Models (`crates/higgs-models/`):**
- Purpose: Model architecture definitions, weight loading, inference kernels, sampling logic
- Location: `crates/higgs-models/src/`
- Contains: Qwen3.5, Qwen2, LLaMA, Mistral, Gemma, specialized layers (DFlash, diffusion, ANE bridges)
- Depends on: MLX for tensor ops, safetensors for weight loading
- Used by: Engine for forward passes and sampling

## Data Flow

**Chat Completion Request (Streaming):**

1. HTTP POST to `/v1/chat/completions` arrives at `chat_completions()` in `crates/higgs/src/routes/chat.rs`
2. Request parsed as `ChatCompletionRequest` (OpenAI format)
3. Router resolves model name to `ResolvedRoute::Higgs` or `ResolvedRoute::Remote` via `state.router.resolve()` in `crates/higgs/src/router.rs`
4. For local (Higgs):
   - Request forwarded to selected `Engine` (Simple or Batch)
   - Engine's `prepare_chat_prompt()` tokenizes messages using `ChatTemplateRenderer` (handles chat format like `<|im_start|>`)
   - Engine's `generate_streaming()` called with prompt tokens
   - Streaming output channels tokens back to handler
   - Handler wraps tokens in OpenAI SSE events
5. For remote (provider proxy):
   - Request forwarded to `crate::proxy::proxy_request()` with model rewrite if needed
   - Response streamed directly from provider

**State Management:**
- `SharedState` (Arc-wrapped `AppState`) holds router, config, metrics, HTTP client
- Available to all route handlers via Axum extractor `State(state)`
- Each model loaded as `Engine` (boxed SimpleEngine or BatchEngine) in hashmap keyed by model name
- Inference mutex/queue ensures serialization (Simple) or orderly interleaving (Batch)

## Key Abstractions

**Engine (Dual Mode):**
- Purpose: Abstracts serialized vs. concurrent inference, unified public API
- Examples: `crates/higgs/src/state.rs` enum wraps `SimpleEngine` or `BatchEngine`
- Pattern: Enum-based polymorphism with match arms for delegation to concrete impl

**ResolvedRoute:**
- Purpose: Result of model name lookup, distinguishes local vs. remote
- Examples: `crates/higgs/src/router.rs`
- Pattern: Enum with variant payloads (engine + metadata for local, provider URL for remote)

**Router (Model Routing Table):**
- Purpose: Maps model names to local engines or remote providers via regex patterns or AI classification
- Examples: `crates/higgs/src/router.rs` - compiles route config into regex and lookup tables
- Pattern: Config-driven routing (direct, pattern, auto-router, default fallback)

**ChatTemplateRenderer:**
- Purpose: Formats messages in model-specific chat syntax (e.g., Qwen's `<|im_start|>user\n...`)
- Examples: `crates/higgs-engine/src/chat_template.rs`
- Pattern: Loads Jinja2-style templates from model config, renders dynamically

**KV Cache:**
- Purpose: Reuses attention key/value arrays across decode steps (within a request) and prefix cache (across requests)
- Examples: `SimpleEngine` uses `PagedPrefixCache` in `crates/higgs-engine/src/paged_prefix_cache.rs`
- Pattern: Mutex-guarded cache, lookups by prefix hash, memory-mapped storage

**DFlash (Speculative Decoding):**
- Purpose: Accelerate token generation by running small drafter model in parallel with verifier
- Examples: `DFlashState` in `SimpleEngine`, wired in `crates/higgs-engine/src/simple.rs:DFlashState`
- Pattern: Optional `dflash_path` at load time, uses GPU drafter or CPU/ANE worker

## Entry Points

**CLI Entry (`main`):**
- Location: `crates/higgs/src/main.rs`
- Triggers: `higgs serve`, `higgs init`, `higgs doctor`, daemon control
- Responsibilities: Parse CLI args, load config, spawn HTTP server or daemon

**HTTP Router:**
- Location: `crates/higgs/src/lib.rs` - `build_router()` constructs Axum app
- Triggers: Incoming HTTP requests (via Tokio listener in main)
- Responsibilities: Dispatch to route handlers, apply middleware (auth, rate limiting, timeouts, CORS)

**Route Handlers:**
- Location: `crates/higgs/src/routes/*.rs` - `chat_completions`, `completions`, `embeddings`, `models`, `anthropic`
- Triggers: HTTP POST to specific paths (`/v1/chat/completions`, `/v1/completions`, etc.)
- Responsibilities: Parse request, resolve route, invoke engine or proxy, serialize response

**Model Loading:**
- Location: `crates/higgs/src/main.rs` - `load_engines()` during startup
- Triggers: Server initialization from config
- Responsibilities: Resolve model paths (local or HF download), instantiate engine with KV cache config

## Error Handling

**Strategy:** Typed error enums at each layer (`ServerError`, `EngineError`, `ModelError`) with conversion to HTTP status codes and JSON error responses.

**Patterns:**
- `ServerError` in `crates/higgs/src/error.rs` - converts to HTTP (400/401/404/500)
- `EngineError` in `crates/higgs-engine/src/error.rs` - generation, caching, constraint failures
- `ModelError` in `crates/higgs-models/src/error.rs` - architecture mismatch, weight loading failures
- Route handlers use `?` operator to propagate; Axum's `IntoResponse` renders as JSON error with appropriate status code

## Cross-Cutting Concerns

**Logging:** `tracing` crate with `env-filter` for level control. Initialized in `init_tracing()` (main.rs). Configured via `RUST_LOG` env var or `--verbose` CLI flag.

**Validation:** Configuration validated at startup by `doctor` command (`crates/higgs/src/doctor.rs`). Model paths resolved, providers probed, weight files checked. Crashes must not happen at runtime due to lint rules (`unwrap_used = "deny"`).

**Authentication:** Bearer token check via `tower-http::ValidateRequestHeaderLayer` applied conditionally if `api_key` set in config. Optional per-request auth.

**Rate Limiting:** Per-IP minute-based rate limiter (governor crate) applied conditionally if `rate_limit` > 0. Keyed by client IP from `ConnectInfo`.

**Metrics (Config Mode):** Request records logged to disk via `MetricsStore` (present only in config mode, not simple mode). Includes wallclock time, token counts, provider, routing method. Eviction task periodically removes old files.

**Crash Diagnostics:** Optional signal handlers + atexit hook via `install_crash_diagnostics()` (main.rs). Writes to `/tmp/higgs_crash_<pid>.log` if `HIGGS_CRASH_DIAG=1`.

---

*Architecture analysis: 2026-04-18*
