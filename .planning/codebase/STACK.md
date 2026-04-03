# Technology Stack

**Analysis Date:** 2026-04-03

## Languages

**Primary:**
- Rust 1.87.0+ - All server code, inference engine, model loading, and CLI
- TOML - Configuration files (`config.toml`, `Cargo.toml`)

**Secondary:**
- JSON - Request/response payloads, metrics logging, model metadata
- JSON5 - Flexible config file format (fallback/alt config parser)

## Runtime

**Environment:**
- Apple Silicon (M-series Macs) - MLX is ARM-native only
- macOS (Sonoma+) - Metal GPU acceleration via MLX
- Xcode CLI Tools required for building

**Package Manager:**
- Cargo (Rust) - All dependencies managed via `Cargo.toml`
- Lockfile: `Cargo.lock` present
- Workspace structure: root `Cargo.toml` with 3 member crates in `crates/`

## Frameworks

**Core:**
- `axum` 0.8 - HTTP server framework with macros feature
- `tokio` 1.x (full features) - Async runtime with multitasking, signals, process management
- `mlx-rs` (git rev af21d79) - MLX array framework bindings
- `mlx-sys` (git rev af21d79) - MLX C FFI bindings (patched to v0.4.0 Metal kernels)

**Request Handling:**
- `tower-http` 0.6 - Middleware: CORS, tracing, timeouts, request IDs, auth
- `reqwest` 0.12 - HTTP client for proxying to external providers
- `bytes` 1 - Efficient byte buffer handling
- `http` 1 - HTTP primitives (headers, status codes)
- `http-body-util` 0.1 - Streaming body utilities
- `futures` 0.3 - Async stream combinators

**Templating & Prompts:**
- `minijinja` 2.x - Jinja2-compatible prompt templates with loader
- `minijinja-contrib` 2.x - Python compatibility helpers for templates

**Tokenization:**
- `tokenizers` 0.22 - HuggingFace tokenizer library with HTTP support

**Serialization:**
- `serde` 1.x (derive) - Serialization/deserialization framework
- `serde_json` 1.x - JSON encoding/decoding
- `json5` 0.4 - JSON5 parser for flexible configs

**Configuration:**
- `figment` 0.10 - Config composition (env vars, TOML, merging)
- `clap` 4 (derive) - CLI argument parsing
- `toml_edit` 0.22 - TOML manipulation for config set/get
- `directories` 6 - Platform-specific config/cache paths (`~/.config/higgs/`, `~/.cache/huggingface/`)

**Daemon & TUI:**
- `nix` 0.29 - Unix signal handling, process management (SIGHUP, SIGTERM)
- `ctrlc` 3 - Graceful Ctrl+C shutdown
- `ratatui` 0.29 - Terminal UI framework for metrics dashboard
- `crossterm` 0.28 - Cross-platform terminal control

**Observability:**
- `tracing` 0.1 - Structured logging framework
- `tracing-subscriber` 0.3 (with env-filter, json features) - Log output formatting, filtering, JSON serialization

**Error Handling:**
- `thiserror` 2.x - Derive-based error type macros

**Utilities:**
- `uuid` 1.x (v4) - Random request/session IDs
- `chrono` 0.4 (with serde) - Timestamps in metrics logs
- `async-stream` 0.3 - Streaming macro helper
- `base64` 0.22 - Base64 encoding for image data URIs
- `image` 0.25 (jpeg, png only) - Image loading for vision models (LLaVA)
- `regex` 1.x - Route pattern matching

## Testing & Development

**Test Framework:**
- Built-in Rust test harness (no external test runner)
- `wiremock` 0.6 - HTTP mocking for proxy tests
- `tempfile` 3.25 - Temporary directories for test fixtures

**Build Profile:**
- Release: thin LTO with single codegen unit (optimized binary size + speed)

## Key Dependencies by Purpose

**MLX Integration:**
- `mlx-rs` / `mlx-sys` (git af21d79) - Core GPU compute bindings
  - Patched to MLX-C v0.4.0 for fast Metal kernels (60 tok/s vs 37 tok/s at crates.io)
  - Fetch from: `https://github.com/oxideai/mlx-rs.git` at specific revision
  - Custom Metal kernels: Conv1d fast path for sparse prefill, MLP gate+up fusion

**Model Loading:**
- `directories` 6 - Resolve `~/.cache/huggingface/hub/` cache paths
- Integrated HuggingFace model ID resolution (org/name format)

**Gateway/Proxy:**
- `reqwest` 0.12 - Proxies to OpenAI, Anthropic, Ollama, custom OpenAI-compatible APIs
- Streaming support for SSE responses

**Inference Engine:**
- `minijinja` 2 - Prompt template compilation
- `tokenizers` 0.22 - Fast BPE/WordPiece tokenization from HF model cards

## Configuration

**Environment:**
- `HIGGS_*` env vars override config file settings (host, port, models, api_key, rate_limit, timeout)
- Config loader: Figment with Env + TOML providers in order
- `RUST_LOG` / `RUST_LK_SPAN` - Tracing env filter (default: "info", "--verbose" sets to "debug")

**Build:**
- Workspace-wide: `rust-version = "1.87.0"` (MSRV enforced)
- Edition: "2024" (as specified in Cargo.toml; though 2024 is future - likely 2021 intent)
- Lints: Workspace-wide clippy pedantic + nursery (warn), deny on unsafe/panic/todo/unwrap/dbg/indexing

## Platform Requirements

**Development:**
- Rust 1.87.0+
- Xcode Command Line Tools (for Clang/LLVM to compile MLX-C)
- ~10GB for model caches (varies by model)
- Apple Silicon required (ARM/Metal only)

**Runtime:**
- macOS Sonoma+
- Apple Silicon (M1/M2/M3/M4 series)
- Minimum 8GB unified memory; 32GB+ recommended for larger models
- Single static binary (no runtime dependencies beyond macOS system libraries)

**Deployment:**
- Self-contained Rust binary (no Python, no containerization required)
- Listens on configurable TCP socket (default `0.0.0.0:8000`)
- Binaries available via Homebrew: `brew install panbanda/brews/higgs`
- Or build from source: `cargo build --release`

## External Dependencies at Risk

**MLX-RS Git Pin (af21d79):**
- If upstream `oxideai/mlx-rs` breaks at this revision, build fails
- Patch strategy: Cargo.toml `[patch.crates-io]` section overrides crates.io version
- Critical: This revision includes MLX-C v0.4.0 upgrade (62% perf vs crates.io v0.25.3)

**HuggingFace Hub Integration:**
- Model resolution via `huggingface-cli` external command
- Requires `huggingface-cli` installed (Homebrew: `brew install huggingface-cli`)
- Falls back to manual download if CLI unavailable or stdin not TTY
- No API key required for public models (mlx-community org)

---

*Stack analysis: 2026-04-03*
