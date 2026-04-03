# External Integrations

**Analysis Date:** 2026-04-03

## APIs & External Services

**Remote LLM Providers:**
- OpenAI (e.g., GPT-4, GPT-4o)
  - Endpoint: `https://api.openai.com` (default, configurable)
  - Format: `ApiFormat::OpenAi`
  - Auth: `Authorization: Bearer sk-...` (passed through, or rewritten via config `api_key`)
  - Implementation: `crates/higgs/src/routes/openai.rs`, `crates/higgs/src/types/openai.rs`

- Anthropic (e.g., Claude, Claude Sonnet)
  - Endpoint: `https://api.anthropic.com` (default, configurable)
  - Format: `ApiFormat::Anthropic`
  - Auth: `x-api-key: sk-ant-...` header (passed through, or rewritten via config)
  - Implementation: `crates/higgs/src/routes/anthropic.rs`, `crates/higgs/src/types/anthropic.rs`

- Ollama (local/remote self-hosted)
  - Endpoint: Configurable (default: `http://localhost:11434`)
  - Format: `ApiFormat::OpenAi` (Ollama speaks OpenAI format)
  - Auth: Optional (local deployments often have no auth)
  - `strip_auth`: Recommended `true` to remove client headers

- Custom OpenAI-compatible APIs
  - Any URL accepting OpenAI format requests
  - Can be proxied with optional format translation (OpenAI ↔ Anthropic)

**Model Repository:**
- HuggingFace Hub
  - Used for: Model discovery and download
  - Model format: MLX safetensors (org/name format, e.g., `mlx-community/Llama-3.2-1B-Instruct-4bit`)
  - Integration: External `huggingface-cli` command for download
  - Cache location: `~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{hash}/`
  - Files resolved in: `crates/higgs/src/model_resolver.rs`, `crates/higgs/src/model_download.rs`
  - No credentials required for public models (mlx-community)

## Data Storage

**Databases:**
- None - Higgs is stateless
- No persistent datastore (no PostgreSQL, Redis, MongoDB, etc.)
- State is ephemeral: in-memory request routing, inference caches

**File Storage:**
- Local filesystem only
  - Model weights: User's HuggingFace cache or custom path
  - Config files: `~/.config/higgs/config.toml` (or `config.<profile>.toml`)
  - Metrics logs: `~/.config/higgs/logs/metrics.jsonl` (JSONL format, rotated)
  - PID files: `~/.config/higgs/higgs.pid` (or `higgs.<profile>.pid`)
  - Log files: Stderr for main logs, metrics.jsonl for structured metrics

**Caching:**
- In-process KV cache (MLX managed memory):
  - Prompt prefix cache (radix tree) for reused context
  - Paged KV cache for long-context requests
  - No external cache service (Redis, Memcached)

## Authentication & Identity

**Auth Provider:**
- Custom bearer token (optional)
  - Config: `[server] api_key = "sk-..."`
  - Implementation: Middleware in `crates/higgs/src/lib.rs:build_router()`
  - Validates `Authorization: Bearer <token>` on every request
  - If `api_key` unset, no auth enforced

**Upstream Auth:**
- Transparent passthrough:
  - Client's `Authorization` header forwarded to upstream provider
  - Can be overridden: `[provider.{name}] api_key` injects `x-api-key` header
  - Can be stripped: `[provider.{name}] strip_auth = true` removes auth from proxied request

## Monitoring & Observability

**Error Tracking:**
- None - No external error reporting (Sentry, Rollbar, etc.)
- Errors logged to stderr via `tracing` at `ERROR` level

**Logs:**
- `tracing` framework with structured output
  - Default: Human-readable to stderr
  - `--verbose` flag: Debug-level logging
  - `RUST_LOG` env var: Fine-grained filter (e.g., `higgs=debug,axum=info`)
  - JSON output available via `tracing-subscriber` (structured logging)

**Metrics:**
- In-process metrics dashboard (TUI)
  - Displayed via `higgs attach` command
  - Stored in JSONL format: `~/.config/higgs/logs/metrics.jsonl`
  - Per-request metrics: latency, token throughput, error rates, request counts
  - Retention: Configurable (default 60 minutes, retention-enabled by default)
  - Implementation: `crates/higgs/src/metrics.rs`

**Health Endpoint:**
- `/health` - GET - Returns 200 OK (no probe details)
- Implementation: `crates/higgs/src/routes/health.rs`

## CI/CD & Deployment

**Hosting:**
- Self-hosted on user's Apple Silicon Mac
- No cloud deployment (AWS, GCP, Azure, etc.)
- Single static binary for distribution

**CI Pipeline:**
- GitHub Actions (`.github/workflows/ci.yml`)
  - Runs: `cargo test`, `cargo clippy`, `cargo fmt --check`
  - Test mode: Single-threaded (`--test-threads=1`) due to shared port bindings
  - Release binary building and distribution

**Homebrew Distribution:**
- Formula: `panbanda/brews/higgs`
- Installation: `brew install panbanda/brews/higgs`

## Environment Configuration

**Required env vars (optional at runtime):**
- None strictly required; all config via TOML or CLI flags
- Optional overrides:
  - `HIGGS_HOST` - Server bind address (default: `0.0.0.0`)
  - `HIGGS_PORT` - Server port (default: `8000`)
  - `HIGGS_MODELS` - Model paths (semicolon-separated, simple mode)
  - `HIGGS_API_KEY` - Bearer token for auth
  - `HIGGS_RATE_LIMIT` - Requests/minute per client
  - `HIGGS_TIMEOUT` - Request timeout in seconds
  - `HIGGS_MAX_TOKENS` - Max generation length
  - `RUST_LOG` - Tracing filter (e.g., `higgs=debug,tokio=info`)

**Secrets location:**
- API keys stored in `~/.config/higgs/config.toml` under `[server] api_key` or `[provider.{name}] api_key`
- No `.env` file support (config is TOML only)
- Secrets not committed to git (ignored via `.gitignore`)

**Config file locations:**
- Primary: `~/.config/higgs/config.toml`
- Profiles: `~/.config/higgs/config.<name>.toml` (e.g., `config.dev.toml`, `config.prod.toml`)
- Custom: `--config /path/to/config.toml` flag

## Webhooks & Callbacks

**Incoming:**
- None - Higgs is a request/response gateway only, no webhook handlers

**Outgoing:**
- None - No callbacks to external systems
- Proxying is transparent (client calls Higgs, Higgs proxies to upstream)

## Request/Response Flow

**OpenAI Format Endpoints:**
- `POST /v1/chat/completions` - Chat completion (local or proxied)
- `POST /v1/completions` - Legacy text completion (local or proxied)
- `POST /v1/embeddings` - Embeddings (local or proxied)
- `GET /v1/models` - List loaded models

**Anthropic Format Endpoints:**
- `POST /v1/messages` - Message creation (local or proxied)
- `POST /v1/messages/count_tokens` - Token counting (local or proxied with stub support)

**Format Translation:**
- Request translation: OpenAI → Anthropic or Anthropic → OpenAI (based on route config)
- Response translation: Matches request format or target format (streaming SSE + JSON)
- Implementation: `crates/higgs/src/translate.rs`

**Proxy Routing Logic:**
- Requests matched against `[[routes]]` patterns (regex on `model` field)
- First match wins
- Default route: `[default] provider = "higgs"` (local) or provider name (forward unmatched)
- Auto-router (optional): Classify request with local LLM to pick provider
- Implementation: `crates/higgs/src/router.rs`

**Provider Options (config):**
- `url` - Upstream API base URL
- `format` - `"openai"` or `"anthropic"`
- `api_key` - Optional API key to inject (overrides client auth)
- `strip_auth` - Remove `Authorization` header before proxying (for local/open APIs)
- `stub_count_tokens` - Return stub for `/v1/messages/count_tokens` if upstream doesn't support

## Model Integration

**Local Model Loading:**
- Resolved from HuggingFace cache or local path
- Formats: MLX safetensors (quantized or full precision)
- Supported architectures: LLaMA, Mistral, Qwen2/3, Qwen3-MoE, Qwen3-Next, Gemma 2, Phi-3, Starcoder2, DeepSeek-V2, LLaVA-Qwen2
- Implementation: `crates/higgs-models/src/` (per-architecture loaders)
- Engine: `crates/higgs-engine/src/` (inference + batching + prompt caching)

**Remote Model Proxying:**
- No model sync or caching of remote models
- Requests forwarded with optional model name rewriting
- Model rewriting: `[[routes]] model = "actual-upstream-name"` maps alias to upstream

---

*Integration audit: 2026-04-03*
