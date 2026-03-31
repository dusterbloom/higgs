# Codebase Structure

**Analysis Date:** 2026-03-31

## Directory Layout

```
higgs/
├── Cargo.toml              # Workspace root — defines all 3 crate members + shared deps
├── Cargo.lock              # Dependency lockfile
├── rustfmt.toml            # Rust formatting config
├── lefthook.yml            # Git hooks config (lefthook)
├── omen.toml               # Omen config
├── CLAUDE.md               # Claude Code user preferences
├── AGENTS.md               # GitNexus agent instructions
├── crates/
│   ├── higgs/              # Server crate — CLI, HTTP API, routing, proxy, daemon, TUI
│   │   ├── Cargo.toml
│   │   ├── build.rs        # Build script
│   │   ├── src/
│   │   │   ├── main.rs     # CLI entry point (tokio runtime)
│   │   │   ├── lib.rs      # Public API — build_router(), rate_limit_middleware()
│   │   │   ├── config.rs   # CLI args, TOML config, env overlay, validation
│   │   │   ├── state.rs    # Engine enum (Simple/Batch/Stub), AppState
│   │   │   ├── router.rs   # Model name → engine/provider routing
│   │   │   ├── proxy.rs    # HTTP proxy to remote providers
│   │   │   ├── translate.rs # OpenAI ↔ Anthropic format translation
│   │   │   ├── error.rs    # ServerError → JSON HTTP responses
│   │   │   ├── auto_router.rs # AI classifier for route selection
│   │   │   ├── daemon.rs   # Process detach, PID mgmt, exec, shellenv
│   │   │   ├── doctor.rs   # Config/model validation diagnostics
│   │   │   ├── model_resolver.rs # HuggingFace cache / local path resolution
│   │   │   ├── model_download.rs # Interactive download prompt
│   │   │   ├── metrics.rs  # Request metrics store (in-memory + JSONL)
│   │   │   ├── metrics_log.rs # JSONL file writer with rotation
│   │   │   ├── attach.rs   # Load metrics history for TUI
│   │   │   ├── cli_config.rs # CLI get/set/path config operations
│   │   │   ├── anthropic_adapter.rs # Anthropic message conversion helpers
│   │   │   ├── routes/     # Axum route handlers
│   │   │   │   ├── mod.rs
│   │   │   │   ├── chat.rs        # POST /v1/chat/completions
│   │   │   │   ├── completions.rs # POST /v1/completions
│   │   │   │   ├── embeddings.rs  # POST /v1/embeddings
│   │   │   │   ├── models.rs      # GET /v1/models
│   │   │   │   ├── anthropic.rs   # POST /v1/messages, count_tokens
│   │   │   │   └── health.rs      # GET /health
│   │   │   ├── types/      # Request/response type definitions
│   │   │   │   ├── mod.rs
│   │   │   │   ├── openai.rs     # OpenAI API types
│   │   │   │   └── anthropic.rs  # Anthropic API types
│   │   │   └── tui/        # Terminal dashboard
│   │   │       ├── mod.rs        # TUI app loop, TuiConfig
│   │   │       └── views/
│   │   │           ├── mod.rs
│   │   │           ├── overview.rs
│   │   │           ├── models.rs
│   │   │           ├── providers.rs
│   │   │           ├── routing.rs
│   │   │           └── errors.rs
│   │   └── tests/
│   │       ├── integration_tests.rs  # Integration test entry
│   │       └── integration/
│   │           ├── mod.rs
│   │           ├── cli_exec.rs
│   │           ├── error_contract.rs
│   │           ├── proxy_e2e.rs
│   │           ├── request_validation.rs
│   │           ├── response_contract.rs
│   │           └── router.rs
│   ├── higgs-engine/       # Engine crate — tokenization, generation, caching
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs              # Module re-exports
│   │       ├── engine.rs           # GenerationOutput, StreamingOutput types
│   │       ├── simple.rs           # SimpleEngine — serialized request processing
│   │       ├── batch_engine.rs     # BatchEngine — interleaved token-level batching
│   │       ├── chat_template.rs    # Jinja2 chat template rendering
│   │       ├── model_loader.rs     # Unified model loading from directory
│   │       ├── prompt_cache.rs     # PrefixCache for SimpleEngine
│   │       ├── paged_prefix_cache.rs # PagedPrefixCache for SimpleEngine
│   │       ├── reasoning_parser.rs # Think tag parsing (reasoning content extraction)
│   │       ├── tool_parser.rs      # Tool call parsing from generated text
│   │       ├── constrained.rs      # JSON schema constrained generation (outlines)
│   │       ├── mtp.rs              # Multi-token prediction speculative decode
│   │       └── error.rs            # EngineError enum
│   └── higgs-models/       # Model crate — architectures, weights, caching, sampling
│       ├── Cargo.toml
│       ├── build.rs        # Build script
│       ├── bridge/         # ANE (Apple Neural Engine) bridge code
│       │   └── ane/        # C/Objective-C ANE bindings (feature-gated)
│       └── src/
│           ├── lib.rs              # AnyModel, AnyCache, sampling, weight loading
│           ├── registry.rs         # Model type detection from config.json
│           ├── transformer.rs      # Unified Qwen2/Llama/Mistral transformer
│           ├── qwen3_next.rs       # Qwen3-Next hybrid SSM/attention + MoE
│           ├── qwen3_moe.rs        # Qwen3-MoE sparse MoE
│           ├── gemma2.rs           # Gemma 2 architecture
│           ├── phi3.rs             # Phi-3 architecture
│           ├── starcoder2.rs       # Starcoder2 architecture
│           ├── deepseek_v2.rs      # DeepSeek-V2 MLA + MoE
│           ├── rwkv7.rs            # RWKV-7 recurrent architecture
│           ├── llava_qwen2.rs      # LLaVA-Qwen2 VLM (vision-language)
│           ├── diffusion.rs        # Diffusion model support
│           ├── diffusion_ane.rs    # ANE-accelerated diffusion (feature-gated)
│           ├── ane_bridge.rs       # ANE bridge FFI (feature-gated)
│           ├── ane_extract.rs      # ANE weight extraction (feature-gated)
│           ├── ane_forward.rs      # ANE forward pass (feature-gated)
│           ├── ane_mil.rs          # ANE MIL program generation (feature-gated)
│           ├── llada_moe.rs        # LLADA MoE architecture
│           ├── siglip.rs           # SigLIP vision encoder for VLMs
│           ├── cache.rs            # KV cache, TurboQuant cache, PagedPrefixCache
│           ├── turboquant.rs       # TurboQuant KV quantization
│           ├── utils.rs            # RoPE, attention, masking utilities
│           └── error.rs            # ModelError enum
├── scripts/                # Python test/debug scripts for ML model experiments
│   ├── test_full_model.py
│   ├── test_rwkv7_forward.py
│   ├── test_stateless.py
│   ├── test_fp16_convert.py
│   └── ...
├── benchmarks/             # Python benchmarking scripts
│   ├── bench_all.py
│   ├── bench_h2h.py
│   ├── bench_prefix_cache.py
│   ├── bench_turboquant_full.py
│   └── ...
├── docs/                   # Design documents and research notes
│   ├── chunked-prefill-design.md
│   ├── mlx_rs_capabilities.md
│   ├── qwen3_next_architecture.md
│   └── paged-attention-research/
├── .github/workflows/      # CI/CD configuration
└── memory/                 # Agent memory/context files
```

## Directory Purposes

**`crates/higgs/`:**
- Purpose: Top-level server binary and HTTP API
- Contains: CLI parsing, Axum routes, request routing, proxy layer, format translation, daemon management, TUI dashboard, metrics
- Key files: `src/main.rs`, `src/lib.rs`, `src/router.rs`, `src/state.rs`, `src/routes/chat.rs`

**`crates/higgs-engine/`:**
- Purpose: Inference engine abstraction layer
- Contains: Tokenization, chat template rendering, generation loops (simple + batch), prompt caching, constrained generation, reasoning/tool parsing
- Key files: `src/simple.rs`, `src/batch_engine.rs`, `src/chat_template.rs`, `src/model_loader.rs`

**`crates/higgs-models/`:**
- Purpose: Model architectures and ML computation
- Contains: Neural network architectures, forward passes, KV caching, TurboQuant quantization, sampling, safetensors weight loading, vision encoder
- Key files: `src/lib.rs`, `src/transformer.rs`, `src/qwen3_next.rs`, `src/cache.rs`, `src/registry.rs`

**`scripts/`:**
- Purpose: Python scripts for testing ML model behavior, weight conversion, and debugging
- Contains: Standalone test scripts (not part of the Rust build)
- Key files: `test_full_model.py`, `test_rwkv7_forward.py`, `test_fp16_convert.py`

**`benchmarks/`:**
- Purpose: Performance benchmarking of inference speed, TTFT, perplexity
- Contains: Python benchmark scripts and result files
- Key files: `bench_all.py`, `bench_h2h.py`, `bench_prefix_cache.py`, `bench_turboquant_full.py`

**`docs/`:**
- Purpose: Architecture design documents and research notes
- Contains: Design docs for chunked prefill, paged attention, MLX capabilities, Qwen3-Next architecture
- Generated: No (manually authored)
- Committed: Yes

## Key File Locations

**Entry Points:**
- `crates/higgs/src/main.rs`: CLI binary entry point — command dispatch, server startup
- `crates/higgs/src/lib.rs`: Public API — `build_router()` function used by main.rs

**Configuration:**
- `crates/higgs/src/config.rs`: All config types (`HiggsConfig`, `ServerSection`, `ModelConfig`, `ProviderConfig`, `RouteConfig`), CLI args, config loading/validation
- `Cargo.toml`: Workspace-level dependency declarations and lint configuration
- `crates/higgs/Cargo.toml`, `crates/higgs-engine/Cargo.toml`, `crates/higgs-models/Cargo.toml`: Per-crate dependencies

**Core Logic:**
- `crates/higgs/src/router.rs`: Request routing logic (pattern, direct, auto, default)
- `crates/higgs/src/proxy.rs`: HTTP proxy to remote providers
- `crates/higgs/src/translate.rs`: OpenAI ↔ Anthropic format translation
- `crates/higgs/src/state.rs`: `Engine` enum wrapping Simple/Batch, `AppState`
- `crates/higgs-engine/src/simple.rs`: Serialized inference engine
- `crates/higgs-engine/src/batch_engine.rs`: Interleaved batch inference engine
- `crates/higgs-engine/src/chat_template.rs`: Chat template rendering
- `crates/higgs-engine/src/model_loader.rs`: Model loading orchestration
- `crates/higgs-models/src/lib.rs`: `AnyModel` dispatch, sampling, weight loading
- `crates/higgs-models/src/transformer.rs`: Main transformer architecture
- `crates/higgs-models/src/cache.rs`: KV cache implementations (dense + TurboQuant)
- `crates/higgs-models/src/registry.rs`: Model type detection

**Type Definitions:**
- `crates/higgs/src/types/openai.rs`: OpenAI API request/response types
- `crates/higgs/src/types/anthropic.rs`: Anthropic API request/response types
- `crates/higgs-engine/src/engine.rs`: `GenerationOutput`, `StreamingOutput`
- `crates/higgs-models/src/lib.rs`: `SamplingParams`, `AnyModel`, `AnyCache`

**Testing:**
- `crates/higgs/tests/integration/`: Integration tests (proxy E2E, error contracts, router, request validation, CLI exec)
- Unit tests are inline in each source file under `#[cfg(test)]` modules

## Naming Conventions

**Files:**
- snake_case: `chat_template.rs`, `batch_engine.rs`, `model_loader.rs`
- Module files match the public type/concept they contain

**Directories:**
- snake_case: `higgs-models/`, `higgs-engine/`
- Subdirectories for grouped functionality: `routes/`, `types/`, `tui/views/`

**Crate names:**
- Kebab-case in Cargo.toml: `higgs-engine`, `higgs-models`
- snake_case in Rust code: `higgs_engine`, `higgs_models`

**Types:**
- PascalCase: `AppState`, `HiggsConfig`, `SimpleEngine`, `BatchEngine`, `AnyModel`, `AnyCache`
- Enum variants: PascalCase (`ResolvedRoute::Higgs`, `AnyModel::Transformer`)
- Config structs: PascalCase with `Section`/`Config` suffix: `ServerSection`, `ModelConfig`, `RouteConfig`

**Functions:**
- snake_case: `build_router()`, `chat_completions()`, `load_engines()`, `resolve()`

## Where to Add New Code

**New HTTP endpoint:**
- Route handler: `crates/higgs/src/routes/<name>.rs`
- Register route in `crates/higgs/src/lib.rs::build_router()`
- Add module to `crates/higgs/src/routes/mod.rs`
- Request/response types in `crates/higgs/src/types/openai.rs` or new file in `types/`

**New model architecture:**
- Implementation: `crates/higgs-models/src/<arch_name>.rs`
- Add variant to `AnyModel` enum in `crates/higgs-models/src/lib.rs`
- Add variant to `AnyCache` if needed (e.g., new state type)
- Add to `is_supported()` in `crates/higgs-models/src/registry.rs`
- Update `model_loader.rs` in `crates/higgs-engine/src/` to instantiate

**New engine feature (e.g., new caching strategy):**
- Implementation in `crates/higgs-engine/src/` or `crates/higgs-models/src/cache.rs`
- Expose through `Engine` enum in `crates/higgs/src/state.rs`

**New remote provider format:**
- Add variant to `ApiFormat` in `crates/higgs/src/config.rs`
- Add translation functions in `crates/higgs/src/translate.rs`
- Handle in route handlers in `crates/higgs/src/routes/chat.rs`

**New CLI subcommand:**
- Add variant to `Commands` enum in `crates/higgs/src/config.rs`
- Add handler in `crates/higgs/src/main.rs`

**Utilities:**
- Shared helpers: `crates/higgs-models/src/utils.rs` (ML utilities)
- Server helpers: inline in relevant `crates/higgs/src/` modules

## Special Directories

**`crates/higgs-models/bridge/ane/`:**
- Purpose: C/Objective-C source files for Apple Neural Engine integration
- Generated: No (handwritten FFI bindings)
- Committed: Yes
- Feature-gated: Only compiled with `--features ane`

**`target/`:**
- Purpose: Rust build artifacts
- Generated: Yes (by cargo)
- Committed: No (in .gitignore)

**`.venv/`:**
- Purpose: Python virtual environment for scripts/benchmarks
- Generated: Yes
- Committed: No

**`scripts/` and `benchmarks/`:**
- Purpose: Standalone Python scripts for ML experimentation
- Generated: No
- Committed: Yes
- Independent from Rust build system

**`docs/`:**
- Purpose: Architecture research and design documentation
- Generated: No
- Committed: Yes

---

*Structure analysis: 2026-03-31*
