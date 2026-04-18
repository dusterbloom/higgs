# External Integrations

**Analysis Date:** 2026-04-18

## APIs & External Services

**AI Model Providers (Proxy Support):**
- OpenAI API - Passthrough proxy support (format: `openai`)
  - SDK/Client: `reqwest`
  - Route handler: `crates/higgs/src/routes/openai.rs`
  - Type definitions: `crates/higgs/src/types/openai.rs`
  - Translation support: Request/response shape compatibility in `crates/higgs/src/translate.rs`

- Anthropic API - Passthrough proxy support (format: `anthropic`)
  - SDK/Client: `reqwest`
  - Route handler: `crates/higgs/src/routes/anthropic.rs`
  - Type definitions: `crates/higgs/src/types/anthropic.rs`
  - Adapter: `crates/higgs/src/anthropic_adapter.rs`
  - Translation support: Cross-format request/response conversion

- Ollama - Compatible with OpenAI format (local inference)
  - SDK/Client: `reqwest`
  - Supported as OpenAI-compatible endpoint

**HuggingFace Integration:**
- Model repository integration via HuggingFace model IDs
- Default cache location: `~/.cache/huggingface/hub/models--org--name/snapshots/<hash>`
- Model resolution: `crates/higgs/src/model_resolver.rs`
- Model download helper: `crates/higgs/src/model_download.rs` (wraps `huggingface-cli`)
- Supported models: LLaMA, Mistral, Qwen2/3 architectures

## Data Storage

**Databases:**
- None - Stateless inference server
- All state is in-memory or configuration-based

**File Storage:**
- Local filesystem only (no cloud storage integration)
- Model weights stored locally (HuggingFace cache or custom paths)
- Configuration files: TOML in `~/.config/higgs/`
- Logs: JSONL metrics logs in `~/.config/higgs/logs/` or custom path
- Process management: PID files in config directory

**Caching:**
- In-memory metrics store with sliding window (configurable, default 60 minutes)
- Prompt caching (built into inference engine at `crates/higgs-engine/src/prompt_cache.rs`)
- KV cache quantization options: off or TurboQuant (3-4 bit quantization)

## Authentication & Identity

**Auth Provider:**
- Custom (none required for local inference)
- Optional Bearer token API key support (configured via `server.api_key` or `--api-key`)
- Auth headers passed through to upstream providers when configured
- Strippable auth headers for provider translation (config: `strip_auth`)

**Request Authentication:**
- Bearer token validation: `x-api-key` or `authorization` header
- Configured per-provider or globally
- Optional per route

## Monitoring & Observability

**Error Tracking:**
- None (no external error tracking service)
- Errors logged to stdout via `tracing-subscriber`
- Local metrics with configurable JSON line logging

**Logs:**
- Structured JSON logging via `tracing-subscriber` with `json` feature
- Metrics logged to JSONL file (default: `~/.config/higgs/logs/metrics.jsonl`)
- Log level controlled by `RUST_LOG` env var or configured via config
- Metrics include: request ID, model, provider, routing method, status, tokens, duration
- Configurable retention: keep metrics for sliding time window (default 60 minutes)
- File rotation: max size per file (50 MB default), max number of files (5 default)

**Distributed Tracing:**
- Tracing instrumentation via `tracing` crate (supports OpenTelemetry propagation)
- No collector configured out of the box
- Ready for integration with external trace collectors via tracing-opentelemetry

## CI/CD & Deployment

**Hosting:**
- Standalone binary (no container/cloud platform requirement)
- Deploy as: daemon process, systemd service, or foreground process
- Process management: `higgs start`/`higgs stop` commands
- PID file tracking: `~/.config/higgs/higgs.pid`
- Log file: `~/.config/higgs/higgs.log`

**CI Pipeline:**
- GitHub repository: `https://github.com/panbanda/higgs`
- No external CI service configured in codebase
- Build: `cargo build --release`
- Test: `cargo test -p higgs -- --test-threads=1`
- Format check: `cargo fmt -- --check`
- Lint: `cargo clippy -p higgs` (pedantic + nursery)

## Environment Configuration

**Required env vars:**
- None strictly required (all have sensible defaults)

**Optional env vars (prefix: `HIGGS_`):**
- `HIGGS_CONFIG_DIR` - Override default config directory
- `HIGGS_*` - Overlay any config value (e.g., `HIGGS_SERVER__PORT=9000`)
- `RUST_LOG` - Control log level (tracing)
- `HF_TOKEN` - For authenticated HuggingFace downloads (used by huggingface-cli)

**Secrets location:**
- Config file: `~/.config/higgs/config.toml` (contains api_key if configured)
- Environment variables: not committed to repo
- `.env` file: not used, direct env var recommended

## Webhooks & Callbacks

**Incoming:**
- None (not a webhook receiver)

**Outgoing:**
- None (not a webhook sender)

## Request/Response Formats

**Supported API Formats:**

**OpenAI Compatible:**
- Endpoint: `/v1/chat/completions`
- Request: `{ "model": "...", "messages": [...], ... }`
- Response: `{ "choices": [...], "usage": {...}, ... }`
- Stream support: Yes (SSE format)
- Implementation: `crates/higgs/src/types/openai.rs`

**Anthropic Compatible:**
- Endpoint: `/v1/messages`
- Request: `{ "model": "...", "messages": [...], ... }`
- Response: `{ "id": "...", "content": [...], "usage": {...}, ... }`
- Stream support: Yes (custom event stream format)
- Implementation: `crates/higgs/src/types/anthropic.rs`

**Model Routing:**
- Pattern-based routing: regex match on model name to provider
- Automatic routing: Smart router (optional, Arch-Router-1.5B default)
- Default routing: Single fallback provider for unmatched models
- Per-request model override via provider translation

## Cross-Format Translation

**OpenAI → Anthropic:**
- Message format conversion: `crates/higgs/src/translate.rs`
- Role mapping: user/assistant/system
- Tool use adaptation

**Anthropic → OpenAI:**
- Message format conversion
- Response shape compatibility

## Provider Configuration

**Provider Definition:**
```toml
[provider.{name}]
url = "https://api.provider.com"      # Required: upstream URL
format = "openai" or "anthropic"      # Optional: API format (default: openai)
api_key = "sk-..."                     # Optional: provider API key
strip_auth = false                     # Optional: strip client auth before forwarding
stub_count_tokens = false              # Optional: stub /count_tokens responses
```

**Route Definition:**
```toml
[[routes]]
pattern = "claude-.*"                  # Regex pattern for model names
provider = "anthropic"                 # Target provider name
model = "claude-3-opus"                # Optional: override model name
```

## Rate Limiting & Traffic Control

**Rate Limiting:**
- Per-client IP rate limiting (configurable via `server.rate_limit`)
- Requests per minute enforcement via `governor` crate
- Disabled by default (rate_limit = 0)

**Request Timeouts:**
- Global timeout: `server.timeout` (default 300 seconds)
- Applied per request via Tower middleware

**Body Size Limits:**
- `server.max_body_size` (default 10 MB)

## Auto-Routing Intelligence

**Arch-Router Integration:**
- Smart model routing model: `katanemo/Arch-Router-1.5B` (default)
- Classifies requests to best provider
- Optional, disabled by default
- Config: `auto_router.enabled`, `auto_router.model`, `auto_router.timeout_ms`
- Implementation: `crates/higgs/src/auto_router.rs`

---

*Integration audit: 2026-04-18*
