# Technology Stack

**Analysis Date:** 2026-04-18

## Languages

**Primary:**
- Rust 1.87.0+ - Entire codebase (inference engine, models, server)

**Secondary:**
- Shell/Bash - Build scripts, benchmarking infrastructure

## Runtime

**Environment:**
- Native binary compilation via Cargo (Rust toolchain)
- Target: macOS with Apple Silicon primary architecture
- MLX framework (Rust bindings) for accelerated inference

**Package Manager:**
- Cargo 1.x (Rust package manager)
- Lockfile: `Cargo.lock` present

## Frameworks

**Core HTTP/Async:**
- Axum 0.8 - Web framework with macros
- Tokio 1.x - Async runtime with full feature set
- Tower-HTTP 0.6 - HTTP middleware (CORS, tracing, timeouts, request-id, auth)

**ML/Inference:**
- MLX-RS - Rust bindings to MLX framework (commit af21d79)
- MLX-SYS - Low-level MLX C bindings

**Configuration & CLI:**
- Clap 4.x - Command-line argument parsing with derive macros
- Figment 0.10 - Configuration management (env, TOML sources)
- Directories 6.x - Platform-aware config/cache paths

**Serialization:**
- Serde 1.x - Serialization framework
- serde_json 1.x - JSON support
- json5 0.4 - JSON5 parsing for relaxed JSON configs
- toml_edit 0.22 - TOML file manipulation and updates

**Model Loading & Tokenization:**
- Tokenizers 0.22 (with http feature) - HuggingFace tokenizers
- Minijinja 2.x (with loader) - Jinja2 template rendering for chat templates
- Minijinja-contrib 2.x (with pycompat) - Python-compatible template functions
- Safetensors 0.4 - Loading model weights from safetensors format
- Memmap2 0.9 - Memory-mapped file access for large model files

**HTTP Client & Proxying:**
- Reqwest 0.12 (stream, json features) - HTTP client with streaming support
- Bytes 1.x - Zero-copy byte buffer handling
- HTTP 1.x - HTTP primitives (status, headers, methods)
- http-body-util 0.1 - HTTP body utilities
- Futures 0.3 - Future combinators and streams

**Observability & Logging:**
- Tracing 0.1 - Distributed tracing instrumentation
- Tracing-subscriber 0.3 (env-filter, json features) - Structured logging with JSON output
- Chrono 0.4 (serde feature) - Datetime handling with serialization

**Daemon & TUI:**
- Nix 0.29 (signal, process) - Low-level OS abstractions (signals, process management)
- Ctrlc 3.x - Graceful shutdown handling (Ctrl+C)
- Ratatui 0.29 - Terminal UI framework
- Crossterm 0.28 - Terminal event handling

**Utility Libraries:**
- Thiserror 2.x - Error type derive macros
- UUID 1.x (v4) - UUID generation
- Rand 0.9 - Random number generation
- Base64 0.22 - Base64 encoding/decoding
- Regex 1.x - Regular expression support
- Image 0.25 (jpeg, png features) - Image format support
- Async-stream 0.3 - Async stream utilities
- Governor 0.8 - Rate limiting

**Data Processing & ML:**
- Half 2.4 - f16/bf16 half-precision floating point
- Outlines-core 0.2.14 - Structured generation/guidance (optional, for sampling)

**Error Handling:**
- Thiserror 2.x - Ergonomic error enums

**Testing:**
- Wiremock 0.6 - HTTP mocking for integration tests
- Tower 0.5 (util) - Tower service abstractions
- Hyper 1.x - Low-level HTTP library
- Tempfile 3.25 - Temporary file/directory creation

## Local Dependencies

**Workspace members:**
- `higgs-engine` - Core inference engine (tokenization, generation loop, prompt caching)
- `higgs-models` - Model architectures (LLaMA, Mistral, Qwen2/3)
- `higgs` - Main binary (server, router, CLI, daemon)

**Internal crate features:**
- `ane` - Apple Neural Engine backend support (optional, for DFlash drafter acceleration)

## Configuration

**Environment:**
- TOML configuration file format (`~/.config/higgs/config.toml` default location)
- Environment variable overlays with `HIGGS_*` prefix
- CLI argument overrides in priority order: CLI args > HIGGS_* env vars > TOML config > defaults
- Configuration via Figment multi-source merging

**Key Configs Required:**
- `HIGGS_CONFIG_DIR` - Override default config directory
- Provider API keys (optional, via env or config)
- Server host/port (defaults: 0.0.0.0:8000)
- Model paths (HuggingFace model IDs or local directories)

**Build Configuration:**
- `Cargo.toml` workspace root with lints workspace-wide
- `rustfmt.toml` - Code formatting rules
- `omen.toml` - Project metadata/omen configuration
- Workspace lints: deny unsafe_code, forbid non-ascii identifiers
- Clippy: warn pedantic/nursery (with project-specific allow list)
- Restriction lints enforced: unwrap_used, expect_used, panic, todo, unimplemented (all denied)

## Platform Requirements

**Development:**
- Rust 1.87.0+ (MSRV enforced)
- Apple Silicon Mac (target platform)
- Cargo package manager

**Production:**
- macOS with Apple Silicon (primary target)
- MLX framework installed or bundled
- Optional: `huggingface-cli` tool for model management

## Model Loading

**HuggingFace Integration:**
- Direct model ID support: `org/model-name` resolves to `~/.cache/huggingface/hub/models--org--model/snapshots/<hash>`
- Automatic cache detection and fallback to local filesystem paths
- Manual download via `huggingface-cli download org/model` if not cached

**Supported Model Formats:**
- Safetensors weight format
- MLP model architecture
- Tokenizer loading from model directory

---

*Stack analysis: 2026-04-18*
