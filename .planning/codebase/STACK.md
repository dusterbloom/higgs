# Technology Stack

**Analysis Date:** 2026-03-31

## Languages

**Primary:**
- Rust (Edition 2024, MSRV 1.87.0) — All production code across three workspace crates

**Secondary:**
- Python 3.12 — Benchmarking scripts in `benchmarks/` and utility/test scripts in `scripts/`
- Objective-C — Apple Neural Engine (ANE) bridge in `crates/higgs-models/bridge/ane/ane_bridge.m`

**Build (C/C++):**
- C — ANE bridge compilation via `cc` build dependency (behind `ane` feature flag)

## Runtime

**Environment:**
- macOS (Apple Silicon / ARM64 required) — MLX framework and Metal GPU compute only run on Apple Silicon
- No Linux or Windows support (CI runs on `macos-latest`)

**Package Manager:**
- Cargo — Rust package manager
- Lockfile: `Cargo.lock` present

## Frameworks

**Core:**
- MLX (via `mlx-rs` v0.1.0, git dep `oxideai/mlx-rs` rev `af21d79`) — Apple's ML framework for Metal GPU-accelerated inference
- Axum 0.8 — HTTP server framework for the inference API
- Tokio 1 (full features) — Async runtime
- tower-http 0.6 — HTTP middleware (CORS, tracing, timeout, auth, request-id)

**LLM Inference:**
- `tokenizers` 0.22 (HuggingFace) — Fast tokenizer library with HTTP feature for remote tokenizer loading
- `safetensors` 0.4 — Safe tensor format weight loading
- `outlines-core` 0.2.14 — Constrained/structured generation
- `minijinja` 2 + `minijinja-contrib` 2 — Jinja2-compatible chat template rendering

**Serialization:**
- `serde` 1 (derive) — Serialization framework
- `serde_json` 1 — JSON serialization
- `json5` 0.4 — JSON5 config parsing

**CLI:**
- `clap` 4 (derive) — Command-line argument parsing
- `figment` 0.10 (env, toml) — Layered configuration (TOML + env vars)
- `ratatui` 0.29 + `crossterm` 0.28 — Terminal UI for the daemon attach mode
- `directories` 6 — XDG/config directory resolution

**Proxy/Networking:**
- `reqwest` 0.12 (stream, json) — HTTP client for proxying to remote providers
- `governor` 0.8 — Rate limiting middleware
- `regex` 1 — Regex-based route pattern matching
- `async-stream` 0.3 — Async stream utilities for SSE translation
- `futures` 0.3 — Async utilities

**Daemon:**
- `nix` 0.29 (signal, process) — POSIX process management for daemon mode
- `ctrlc` 3 — Graceful shutdown signal handling

**Observability:**
- `tracing` 0.1 — Structured logging/instrumentation
- `tracing-subscriber` 0.3 (env-filter, json) — Log formatting and filtering

**Error Handling:**
- `thiserror` 2 — Derive macro for error types

**Image Processing:**
- `image` 0.25 (jpeg, png) — Image decoding for vision-language models (LLaVA)

**Utilities:**
- `uuid` 1 (v4) — Unique ID generation
- `chrono` 0.4 (serde) — Timestamp handling
- `rand` 0.9 — Random number generation
- `base64` 0.22 — Base64 encoding
- `bytes` 1 — Byte buffer utilities
- `toml_edit` 0.22 — TOML file read/write for `higgs config set`

## Key Dependencies

**Critical:**
- `mlx-rs` — Core MLX bindings; pinned to a specific git revision (`af21d79`). Any update requires testing against Metal GPU behavior.
- `tokenizers` — HuggingFace tokenizer; used for all model tokenization. Critical for prompt preparation.
- `safetensors` — Weight format; all model weights loaded through this.
- `axum` + `tokio` — Server infrastructure; all API endpoints depend on this stack.

**Infrastructure:**
- `reqwest` — Proxying requests to remote LLM providers (Anthropic, OpenAI, Ollama, etc.)
- `figment` — Configuration layering: TOML file → `HIGGS_*` env vars → CLI args
- `governor` — Per-client rate limiting
- `tower-http` — CORS, timeouts, Bearer auth middleware

## Configuration

**Environment:**
- Configuration layered via `figment`: TOML file (`~/.config/higgs/config.toml`) → `HIGGS_*` env vars (with `__` nesting) → CLI args
- Named profiles: `~/.config/higgs/config.<NAME>.toml`
- Env var `HIGGS_CONFIG_DIR` overrides config directory
- `RUSTFLAGS=-Dwarnings` enforced in CI (all warnings are errors)
- `HIGGS_ENABLE_THINKING` env var used in benchmark scripts (not in Rust code)

**Build:**
- `Cargo.toml` — Workspace root with shared dependencies and strict lint configuration
- `rustfmt.toml` — `max_width = 100`, `use_field_init_shorthand = true`
- `lefthook.yml` — Git hooks: pre-commit (`cargo fmt`), pre-push (`cargo fmt --check` + `cargo clippy`)
- `omen.toml` — Code quality analysis thresholds (Omen tool)

**Release:**
- `release-please-config.json` — Automated release with linked versions across 3 crates
- `release-please-manifest.json` — Current versions: all at 0.1.20

## Platform Requirements

**Development:**
- macOS on Apple Silicon (M1/M2/M3/M4)
- Rust toolchain >= 1.87.0
- `huggingface-cli` installed (for model downloads)
- Xcode Command Line Tools (for Metal/ANE compilation)
- Python 3.12+ with venv (for benchmark scripts)

**Production:**
- macOS on Apple Silicon only
- Binary distributed as `higgs` + `mlx.metallib` (Metal compute shader)
- Supports background daemon mode with TUI attach
- Homebrew tap available (`panbanda/homebrew-brews`)

---

*Stack analysis: 2026-03-31*
