# External Integrations

**Analysis Date:** 2026-03-31

## APIs & External Services

**LLM Provider Proxies (outgoing):**
- Anthropic API (`https://api.anthropic.com`) — Proxy target for Claude models
  - SDK/Client: `reqwest` (raw HTTP)
  - Auth: Provider API key in config `[provider.<name>].api_key` or `HIGGS__PROVIDER__<NAME>__API_KEY`
  - Format: `anthropic` — Higgs translates between OpenAI ↔ Anthropic request/response formats automatically
  - Routes: `/v1/messages`, `/v1/messages/count_tokens`

- OpenAI API (any compatible endpoint) — Proxy target for GPT/o1 models
  - SDK/Client: `reqwest` (raw HTTP)
  - Auth: Provider API key in config or env var
  - Format: `openai` (default format for all providers)
  - Routes: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`

- Ollama (`http://localhost:11434`) — Local proxy target
  - SDK/Client: `reqwest` (raw HTTP)
  - Auth: Optional; supports `strip_auth = true` to remove auth headers
  - Format: `openai` (Ollama exposes OpenAI-compatible API)
  - Supports `stub_count_tokens = true` for providers without token counting

- Any OpenAI-compatible or Anthropic-compatible endpoint — Generic proxy support
  - Config: `[provider.<name>].url`, `[provider.<name>].format`
  - Supports model rewrite (`[routes].model`), auth stripping, and stub count_tokens

**HuggingFace (model downloads):**
- HuggingFace Model Hub — Model download and caching
  - Client: `huggingface-cli` (external CLI, not an SDK dependency)
  - Download: Invoked via `std::process::Command` in `crates/higgs/src/main.rs`
  - Install hint: `brew install huggingface-cli`
  - Cache: Uses HuggingFace default cache directory (`~/.cache/huggingface/`)
  - Tokenizer loading: `tokenizers` crate can fetch tokenizer.json via HTTP (`http` feature enabled)

**Tokenizers Library (HuggingFace):**
- HuggingFace Tokenizers — Rust tokenizer library
  - SDK: `tokenizers` 0.22 crate
  - Usage: Loads `tokenizer.json` from local model directories
  - HTTP feature enabled for potential remote tokenizer fetching

## Data Storage

**Model Weights:**
- Local filesystem — Models stored as safetensors files
  - Format: `model.safetensors` (single file) or `model.safetensors.index.json` + sharded files
  - Client: `safetensors` 0.4 crate + MLX `Array::load_safetensors`
  - Locations: `~/.cache/huggingface/` (HF cache), `~/.cache/lm-studio/models/`, or arbitrary paths

**Configuration:**
- TOML files — Server configuration
  - Default: `~/.config/higgs/config.toml`
  - Profiles: `~/.config/higgs/config.<NAME>.toml`
  - Client: `figment` 0.10 (TOML + env + serialized layers)

**Metrics Log:**
- JSONL files — Request metrics
  - Default: `~/.config/higgs/logs/metrics.jsonl`
  - Rotation: Configurable `max_size_mb` (default 50) and `max_files` (default 5)
  - Implementation: `crates/higgs/src/metrics_log.rs`

**PID/Log Files:**
- PID file: `~/.config/higgs/higgs.pid` (or `higgs.<profile>.pid`)
- Log file: `~/.config/higgs/higgs.log` (or `higgs.<profile>.log`)

**File Storage:**
- No cloud file storage; all model weights and configs are local filesystem only

**Caching:**
- KV Cache — In-memory key-value cache for transformer attention during inference
  - Standard dense cache: `SteppingKeyValueCache` in `crates/higgs-models/src/cache.rs`
  - TurboQuant compressed cache: Quantized KV cache for memory efficiency in `crates/higgs-models/src/turboquant.rs`
  - Paged prefix cache: `PagedPrefixCache` for multi-turn conversations in `crates/higgs-engine/src/paged_prefix_cache.rs`
  - Prompt cache: Prefix-level caching in `crates/higgs-engine/src/prompt_cache.rs`

## Authentication & Identity

**Auth Provider:**
- Custom Bearer token authentication
  - Implementation: `tower-http` `ValidateRequestHeaderLayer::bearer(key)` in `crates/higgs/src/lib.rs`
  - Config: `server.api_key` in TOML, `HIGGS_SERVER_API_KEY` env var, or `--api-key` CLI arg
  - Behavior: If unset, no authentication required (open access)
  - Applied to: All `/v1/*` routes; `/health` endpoint is unauthenticated

**Proxy Auth:**
- Per-provider API keys for upstream providers
  - Config: `[provider.<name>].api_key`
  - Forwarded as `x-api-key` header to upstream
  - `strip_auth = true` removes incoming Authorization/x-api-key before forwarding (for local providers like Ollama)

## Monitoring & Observability

**Error Tracking:**
- None (no external error tracking service)

**Logs:**
- Structured logging via `tracing` + `tracing-subscriber`
  - Config: `RUST_LOG` env var (or `HIGGS_LOG` filtered by `tracing_subscriber::EnvFilter`)
  - Default filter: `info` (or `higgs=debug` with `--verbose` flag)
  - JSON output supported via `tracing-subscriber` json feature
  - Daemon mode writes to `~/.config/higgs/higgs.log`

**Metrics:**
- Built-in request metrics store
  - Implementation: `crates/higgs/src/metrics.rs`
  - Tracked: request count, latency, token usage, status codes, routing method, per-minute throughput
  - TUI dashboard: `crates/higgs/src/tui/` — Real-time terminal UI showing metrics
  - Retention: Configurable time window (default 60 minutes)
  - Persistence: JSONL log files with rotation

## CI/CD & Deployment

**Hosting:**
- GitHub repository: `panbanda/higgs`
- Binary distribution: GitHub Releases (tar.gz + SHA256 checksums)
- Homebrew tap: `panbanda/homebrew-brews`

**CI Pipeline:**
- GitHub Actions (`.github/workflows/ci.yml`)
  - Test job: `cargo test --all-features -- --test-threads=1` on `macos-latest`
  - Lint job: `cargo fmt --check` + `cargo clippy --all-targets --all-features`
  - MSRV job: `cargo check --all-features` with Rust 1.87.0
  - Coverage job: `cargo llvm-cov` with 70% line coverage threshold (excluding experimental modules)
  - Omen analysis: Code quality scoring on pull requests (`panbanda/omen@omen-v4.21.2`)

**Release Pipeline:**
- GitHub Actions (`.github/workflows/release.yml`)
  - Trigger: Push to `main`
  - Release Please: Automated versioning and changelog (`googleapis/release-please-action`)
  - Build: `cargo build --release --target aarch64-apple-darwin`
  - Publish: `cargo publish` to crates.io for all 3 crates
  - Homebrew: Auto-updates formula in `panbanda/homebrew-brews`
  - Artifacts: `higgs` binary + `mlx.metallib` Metal shader

**Security Scanning:**
- OpenSSF Scorecard (`.github/workflows/scorecard.yml`)
  - Weekly scheduled analysis + push/branch_protection triggers
  - Results published to GitHub Security tab

## Environment Configuration

**Required env vars:**
- None required for basic operation (all defaults are usable)

**Optional env vars:**
- `HIGGS_SERVER_HOST` — Bind address (default: `0.0.0.0`)
- `HIGGS_SERVER_PORT` — Bind port (default: `8000`)
- `HIGGS_SERVER_API_KEY` — Bearer token for authentication
- `HIGGS_SERVER_MAX_TOKENS` — Default max generation tokens
- `HIGGS_SERVER_RATE_LIMIT` — Requests per minute per client
- `HIGGS_SERVER_TIMEOUT` — Request timeout in seconds
- `HIGGS_CONFIG_DIR` — Override config directory
- `RUST_LOG` — Tracing log filter
- `HIGGS_ENABLE_THINKING` — Used in benchmark scripts (not in Rust source)

**Nested env vars (config mode):**
- `HIGGS__PROVIDER__<NAME>__URL` — Provider endpoint URL
- `HIGGS__PROVIDER__<NAME>__API_KEY` — Provider API key
- `HIGGS__PROVIDER__<NAME>__FORMAT` — `openai` or `anthropic`

**Secrets location:**
- API keys stored in TOML config files or passed via environment variables
- GitHub Actions secrets: `GITHUB_TOKEN`, `CARGO_REGISTRY_TOKEN`, `HOMEBREW_TAP_GITHUB_TOKEN`

## Webhooks & Callbacks

**Incoming:**
- None — Higgs does not receive webhooks

**Outgoing:**
- None — Higgs does not send webhooks

## Apple Platform Integrations

**Metal GPU:**
- MLX framework — Apple's machine learning framework backed by Metal Performance Shaders
  - Bindings: `mlx-rs` (Rust) + `mlx-sys` (FFI to MLX C API)
  - Shader: `mlx.metallib` compiled Metal compute shader (bundled with binary)
  - Required: Apple Silicon GPU

**Apple Neural Engine (ANE):**
- Optional ANE inference path (behind `ane` feature flag)
  - Implementation: `crates/higgs-models/bridge/ane/` — Objective-C bridge (`ane_bridge.h`, `ane_bridge.m`)
  - Build: `cc` crate compiles Objective-C bridge
  - Files: `ane_bridge.rs`, `ane_extract.rs`, `ane_forward.rs`, `ane_mil.rs`, `diffusion_ane.rs`
  - Makefile: `crates/higgs-models/bridge/ane/Makefile` for native compilation

## Supported Model Architectures

**Text-only:**
- Qwen2, Qwen2.5, Qwen3, Qwen3.5, Qwen3-Next (hybrid SSM/attention), Qwen3-MoE
- Llama, Llama 2, Llama 3.x
- Mistral, Mixtral
- Gemma 2
- Phi-3
- StarCoder 2
- DeepSeek V2 (MoE)
- RWKV-7 (recurrent, no attention)

**Vision-Language:**
- LLaVA-Qwen2 (nanoLLaVA architecture with SigLIP vision encoder)

**Diffusion:**
- Diffusion models (standard + ANE-accelerated)

---

*Integration audit: 2026-03-31*
