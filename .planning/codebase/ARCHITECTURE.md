# Architecture

**Analysis Date:** 2026-03-31

## Pattern Overview

**Overall:** Layered monolith with a unified AI gateway pattern

**Key Characteristics:**
- Three-crate workspace with strict dependency direction: `higgs` → `higgs-engine` → `higgs-models`
- Single-binary CLI/server with subcommands (serve, start, stop, attach, init, doctor, config, exec)
- OpenAI-compatible and Anthropic-compatible HTTP API on top of local MLX inference
- Request routing layer that dispatches to local MLX engines or remote provider proxies
- Cross-format translation between OpenAI and Anthropic API protocols
- AI-powered auto-routing using a local classifier model
- Daemon mode with PID file management, log capture, and TUI dashboard

## Layers

**Server Layer (higgs crate):**
- Purpose: HTTP API, request routing, proxying, daemon management, CLI
- Location: `crates/higgs/src/`
- Contains: Axum route handlers, request/response translation, config loading, daemon process management, TUI dashboard
- Depends on: `higgs-engine`, `higgs-models`, `axum`, `reqwest`, `clap`, `figment`
- Used by: End users via CLI binary

**Engine Layer (higgs-engine crate):**
- Purpose: Tokenization, chat templating, generation loop, prompt caching, constrained decoding
- Location: `crates/higgs-engine/src/`
- Contains: `SimpleEngine` (serialized), `BatchEngine` (interleaved), chat template rendering, prefix caching, reasoning parser, tool call parser
- Depends on: `higgs-models`, `mlx-rs`, `tokenizers`, `minijinja`, `outlines-core`
- Used by: `higgs` server layer

**Model Layer (higgs-models crate):**
- Purpose: Model architectures, weight loading, forward passes, KV caching, sampling
- Location: `crates/higgs-models/src/`
- Contains: `AnyModel` dispatch enum, transformer implementations, cache types, TurboQuant KV quantization, sampling, safetensors loading
- Depends on: `mlx-rs`, `mlx-sys`, `tokenizers`, `safetensors`
- Used by: `higgs-engine`

## Data Flow

**Local Inference (chat completions):**

1. HTTP request arrives at `POST /v1/chat/completions` (`crates/higgs/src/routes/chat.rs`)
2. Request body deserialized as `ChatCompletionRequest` (`crates/higgs/src/types/openai.rs`)
3. `Router::resolve()` dispatches to local engine or remote provider (`crates/higgs/src/router.rs`)
4. For local: messages converted to engine format, chat template applied (`crates/higgs-engine/src/chat_template.rs`)
5. Tokens fed to `Engine::generate()` or `Engine::generate_streaming()` via `tokio::task::spawn_blocking` (`crates/higgs/src/state.rs`)
6. Inside engine: prompt tokens processed through `AnyModel::forward()` on MLX Metal GPU (`crates/higgs-models/src/lib.rs`)
7. KV cache updated per-layer, TurboQuant quantization applied if configured (`crates/higgs-models/src/cache.rs`)
8. Logits sampled via temperature/top-k/top-p/min-p filtering (`crates/higgs-models/src/lib.rs`)
9. Generated tokens returned as `GenerationOutput` or streamed via `mpsc::channel` as `StreamingOutput`
10. Response serialized as OpenAI-compatible JSON (non-streaming) or SSE events (streaming)

**Remote Proxy Flow:**

1. Request resolved to `ResolvedRoute::Remote` by router
2. If provider format differs from request format, translation applied (`crates/higgs/src/translate.rs`)
3. `proxy::proxy_request()` or `proxy::send_to_provider()` forwards to upstream (`crates/higgs/src/proxy.rs`)
4. Response headers filtered, body streamed back (or translated for cross-format)
5. Usage metrics extracted and recorded

**Auto-Routing Flow:**

1. Request with `model="auto"` or `auto_router.force=true`
2. `Router::try_auto_route()` calls `auto_router::classify_local()` via `spawn_blocking` with timeout
3. Local classifier model generates JSON `{"route": "name"}` from route descriptions + conversation
4. Parsed route name matched against configured route targets
5. Falls through to default provider if classification fails or times out

**State Management:**
- `AppState` shared via `Arc<AppState>` across all Axum handlers
- Contains `Router`, `HiggsConfig`, `reqwest::Client`, and optional `MetricsStore`
- `Engine` instances (Simple or Batch) held in `Router::local_engines` as `Arc<Engine>`
- SimpleEngine uses `Mutex<Model>` for serialized access; BatchEngine uses a dedicated background thread with `mpsc` channel

## Key Abstractions

**Engine enum:**
- Purpose: Unified interface over Simple and Batch inference engines
- Examples: `crates/higgs/src/state.rs`
- Pattern: Enum dispatch — `Engine::Simple(Box<SimpleEngine>)` or `Engine::Batch(Box<BatchEngine>)` with `#[cfg(test)]` `Engine::Stub` variant

**AnyModel enum:**
- Purpose: Unified dispatch across all supported model architectures
- Examples: `crates/higgs-models/src/lib.rs`
- Pattern: Enum with variants for each architecture (Transformer, Qwen3Next, Qwen3Moe, Gemma2, Phi3, Starcoder2, LlavaQwen2, DeepSeekV2, Rwkv7) — matched against `AnyCache` variants for forward pass dispatch

**AnyCache enum:**
- Purpose: Architecture-appropriate cache storage
- Examples: `crates/higgs-models/src/cache.rs`, `crates/higgs-models/src/lib.rs`
- Pattern: `KV(Vec<SteppingKeyValueCache>)` for transformers, `Hybrid(Vec<LayerCache>)` for Qwen3Next, `Rwkv7(Vec<Rwkv7LayerState>)` for recurrent models

**ResolvedRoute enum:**
- Purpose: Outcome of routing a model name
- Examples: `crates/higgs/src/router.rs`
- Pattern: `Higgs { engine, model_name, routing_method }` or `Remote { provider_name, provider_url, provider_format, ... }`

**Router struct:**
- Purpose: Routes model names to local engines or remote providers via pattern matching, direct lookup, auto-routing, or default fallback
- Examples: `crates/higgs/src/router.rs`
- Pattern: Resolution order: auto-route → regex pattern match → direct engine lookup → default provider

## Entry Points

**CLI binary:**
- Location: `crates/higgs/src/main.rs`
- Triggers: `higgs serve`, `higgs start`, `higgs stop`, `higgs attach`, `higgs init`, `higgs doctor`, `higgs config`, `higgs exec`, `higgs shellenv`
- Responsibilities: CLI argument parsing via clap, config loading, engine initialization, Axum server startup, daemon management

**HTTP API routes:**
- Location: `crates/higgs/src/routes/`
- Triggers: HTTP requests to the running server
- Responsibilities:
  - `routes/chat.rs`: `POST /v1/chat/completions` (OpenAI chat format)
  - `routes/completions.rs`: `POST /v1/completions` (OpenAI completions format)
  - `routes/embeddings.rs`: `POST /v1/embeddings`
  - `routes/models.rs`: `GET /v1/models`
  - `routes/anthropic.rs`: `POST /v1/messages`, `POST /v1/messages/count_tokens` (Anthropic format)
  - `routes/health.rs`: `GET /health`

**TUI dashboard:**
- Location: `crates/higgs/src/tui/mod.rs`
- Triggers: `higgs attach` command
- Responsibilities: Real-time metrics dashboard using ratatui/crossterm, displays request history, model info, routing stats

**Daemon lifecycle:**
- Location: `crates/higgs/src/daemon.rs`
- Triggers: `higgs start` (fork), `higgs stop` (SIGTERM), `higgs attach` (TUI)
- Responsibilities: Process detachment via `setsid`, PID file management, log file capture, graceful shutdown via SIGINT/SIGTERM

## Error Handling

**Strategy:** Layered error enums with `thiserror`, converted to HTTP responses via `IntoResponse`

**Patterns:**
- `ModelError` (`crates/higgs-models/src/error.rs`): Weight loading, unsupported architectures, quantization errors
- `EngineError` (`crates/higgs-engine/src/error.rs`): Tokenization, generation, template rendering — wraps `ModelError`
- `ServerError` (`crates/higgs/src/error.rs`): HTTP-layer errors — wraps `EngineError`, adds BadRequest/ModelNotFound/ProxyError variants
- `ServerError::IntoResponse` maps to OpenAI-compatible JSON error responses with appropriate HTTP status codes
- Internal errors (500) mask details to prevent information leakage; proxy errors (502) include upstream details

## Cross-Cutting Concerns

**Logging:** `tracing` crate with `tracing-subscriber`, configurable via `RUST_LOG` or `--verbose` flag. Structured fields on log statements (e.g., `tracing::info!(model = %name, "Model loaded")`).

**Validation:** Config validation in `crates/higgs/src/config.rs::validate_config()` — checks model paths, route references, provider existence, timeout bounds. Request validation in route handlers (non-empty messages, valid JSON).

**Authentication:** Optional Bearer token auth via `tower-http::ValidateRequestHeaderLayer` configured per-server. Rate limiting via `governor::RateLimiter` keyed by client IP.

**Metrics:** `MetricsStore` (`crates/higgs/src/metrics.rs`) records request latency, token counts, routing method, provider. Optional JSONL file logging via `MetricsLogger` (`crates/higgs/src/metrics_log.rs`) with size-based rotation.

**Observability:** Request tracing via `tower-http::TraceLayer`. Request IDs generated as `chatcmpl-<uuid>`. Metrics available in TUI dashboard via `higgs attach`.

---

*Architecture analysis: 2026-03-31*
