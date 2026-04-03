# Codebase Structure

**Analysis Date:** 2026-04-03

## Directory Layout

```
/Users/peppi/Dev/higgs/
├── crates/                      # Rust workspace with 3 member crates
│   ├── higgs/                   # Main binary crate (CLI, HTTP API, routing)
│   ├── higgs-engine/            # Inference engine (execution, caching, scheduling)
│   └── higgs-models/            # Model architectures (Qwen, Llama, Phi, etc.)
├── docs/                        # User documentation
├── target/                      # Build artifacts (ignored)
├── vendor/                      # Vendored dependencies
├── .planning/                   # GSD analysis documents
├── Cargo.toml                   # Workspace manifest
├── Cargo.lock                   # Dependency lock file
├── README.md                    # Project overview
└── CLAUDE.md                    # Project development rules
```

## Directory Purposes

**crates/higgs/**
- Purpose: Main binary and server logic
- Contains: CLI parsing, HTTP server (Axum), routing, request translation, daemon management
- Key files: main.rs (entry point), lib.rs (public API), config.rs (configuration schema)

**crates/higgs/src/routes/**
- Purpose: HTTP route handlers for OpenAI and Anthropic API endpoints
- Contains: /v1/chat/completions, /v1/messages, /v1/completions, /v1/embeddings
- Files: chat.rs (chat completion handler), anthropic.rs (Claude API), completions.rs (raw text), embeddings.rs

**crates/higgs/src/types/**
- Purpose: Request/response type definitions for different API formats
- Contains: OpenAI and Anthropic protocol types (request/response DTOs)
- Files: openai.rs, anthropic.rs

**crates/higgs/src/tui/**
- Purpose: Terminal UI for daemon monitoring (higgs attach command)
- Contains: Dashboard views, status display, routing table visualization
- Files: mod.rs (TUI entry), views/ directory with individual screens

**crates/higgs-engine/**
- Purpose: Core inference execution and session management
- Contains: SimpleEngine, BatchEngine, KV cache, continuous batching scheduler
- Key modules: simple.rs (main engine), batch_engine.rs (batched variant)

**crates/higgs-engine/src/cache/**
- Purpose: Paged KV cache implementation
- Contains: BlockAllocator (free list), PageTable (logical→physical mapping), CPU storage
- Files:
  - mod.rs: public interface (PagedKvCache, BlockAllocator exports)
  - paged.rs: PagedKvCache struct with allocate/free/gather/write methods
  - allocator.rs: BlockAllocator free list management
  - pagetable.rs: Logical position → physical block ID mapping
  - storage.rs: CPU-side KV buffer management

**crates/higgs-engine/src/scheduler/**
- Purpose: Round-robin scheduling for continuous batching
- Contains: Session ordering logic, batch composition
- Files: mod.rs (exports), round_robin.rs (RoundRobinScheduler implementation)

**crates/higgs-models/**
- Purpose: Model architecture implementations
- Contains: 11+ model types (Qwen3.5, Llama, Mistral, Gemma2, Phi3, etc.)
- Key files: lib.rs (AnyModel enum), registry.rs (model detection)

**crates/higgs-models/src/spec_prefill/**
- Purpose: Speculative prefill optimization for large prompts
- Contains: Sparse attention (only high-probability tokens), RoPE rotation, scoring
- Files: mod.rs, prefill.rs, draft.rs, rope.rs, scoring.rs, scoring_attention.rs

## Key File Locations

**Entry Points:**

- `crates/higgs/src/main.rs`: Binary entry point
  - tokio::main() → CLI parsing → command dispatch
  - cmd_serve(): load config, instantiate engines, start Axum server
  - load_engines(): resolve model paths, call Engine::load_simple/load_batch

- `crates/higgs/src/lib.rs`: Library exports
  - build_router(): construct Axum Router with all routes and middleware
  - Public modules: config, routes, state, router, daemon

**Configuration:**

- `crates/higgs/src/config.rs`: Configuration schema and parsing
  - HiggsConfig: root config structure
  - ServeArgs: CLI arguments for serve/start commands
  - figment-based config merging (TOML + env vars)
  - load_config_file(), build_simple_config(), validate_profile_name()

- `crates/higgs/src/daemon.rs`: Daemon lifecycle management
  - cmd_init(): create default config template
  - detach(): fork background process, write PID
  - write_pid_file(), remove_pid_file(), await_shutdown_signal()

- `crates/higgs/src/doctor.rs`: Configuration validation
  - run_doctor(): check model paths, probe providers, validate sampling config
  - Called before server start or with `higgs doctor` command

**Core Logic:**

- `crates/higgs/src/router.rs`: Model name → engine/provider routing
  - Router struct with from_config()
  - Resolution order: auto-route (if enabled) → pattern matching → direct lookup → default
  - ResolvedRoute enum: Higgs (local) or Remote (proxy)

- `crates/higgs/src/state.rs`: Shared application state
  - Engine enum: SimpleEngine | BatchEngine | Stub (test)
  - AppState: router + config + http_client + metrics (Arc-wrapped)
  - Delegation methods: model_name(), tokenizer(), generate(), generate_streaming()

- `crates/higgs-engine/src/simple.rs`: Main inference engine
  - SimpleEngine struct with load(), generate(), generate_streaming()
  - Internal: PreparedGeneration for prefix cache lookup + model locking
  - Session and scheduler management for continuous batching
  - Includes paged cache, prefix cache, spec prefill, thinking mode support

- `crates/higgs-engine/src/batch_engine.rs`: Experimental batch inference
  - BatchEngine struct (placeholder; not feature-complete)

**Testing:**

- `crates/higgs/tests/integration/`: Integration test suite
  - mod.rs: test utilities, shared fixtures
  - error_contract.rs: error response schema tests
  - response_contract.rs: OpenAI response format validation
  - router.rs: routing resolution tests
  - proxy_e2e.rs: end-to-end proxy tests to remote providers
  - request_validation.rs: request parsing validation
  - cli_exec.rs: CLI command execution tests

- Unit tests are co-located with source files (after tests module in each file)

## Naming Conventions

**Files:**

- Module files: snake_case.rs (e.g., `simple.rs`, `chat_template.rs`)
- Submodule directories: snake_case/ with mod.rs inside (e.g., `cache/mod.rs`)
- Binary entry: `main.rs`
- Library entry: `lib.rs`
- Integration tests: `tests/` directory

**Directories:**

- Feature modules: lowercase plural when grouping related files
  - `routes/` for HTTP handlers
  - `types/` for request/response types
  - `cache/` for cache implementations
  - `scheduler/` for scheduling logic
  - `views/` for TUI components
  - `spec_prefill/` for speculative prefill variants

**Functions/Types:**

- Types: PascalCase (e.g., SimpleEngine, GenerationOutput, PagedKvCache)
- Functions: snake_case (e.g., generate(), prepare_chat_prompt(), detect_model_type())
- Methods: snake_case (e.g., engine.model_name(), router.resolve())
- Constants: SCREAMING_SNAKE_CASE (e.g., DEFAULT_PREFIX_CACHE_SIZE = 8)
- Trait implementations: Direct impl blocks without suffix (e.g., `impl Engine { ... }`)

**Modules:**

- Public API exports: explicit pub use statements in mod.rs
- Private modules: pub mod submodule { ... } without exporting internals
- Error types: suffixed with Error or ErrorCode (e.g., EngineError, ModelError)

## Where to Add New Code

**New Feature (e.g., new API endpoint):**

- Route handler: `crates/higgs/src/routes/{feature}.rs`
- Request type: `crates/higgs/src/types/openai.rs` or `types/anthropic.rs`
- Response type: same location as request
- Integration test: `crates/higgs/tests/integration/{feature}.rs`
- Export: add `pub mod {feature}` in `routes/mod.rs` and add route in `lib.rs:build_router()`

**New Model Architecture:**

- Implementation: `crates/higgs-models/src/{arch_name}.rs`
  - Struct with forward() → Array, forward_lm_head() → logits
  - Implement ModelTrait if shared behavior needed
- Registry: Add to `registry.rs:is_supported()` match statement
- Variant: Add variant to AnyModel enum in `lib.rs`
- Instantiation: Add case to model_loader.rs or matching logic in AnyModel constructor
- Tests: Co-locate unit tests in implementation file, integration tests in `crates/higgs/tests/`

**New Cache/Scheduling Feature:**

- Cache implementation: `crates/higgs-engine/src/cache/{feature}.rs`
- Public interface: export from `cache/mod.rs`
- Scheduler variant: `crates/higgs-engine/src/scheduler/{variant}.rs`
- Integration: wire into SimpleEngine if needed (update fields, initialization)

**Utility/Helper Functions:**

- Shared helpers: `crates/higgs-engine/src/utils.rs` or `higgs-models/src/utils.rs`
- Model-specific utils: `crates/higgs-models/src/{arch_name}/utils.rs` (submodule)
- Server utils: `crates/higgs/src/` (create new file or use existing module)

**Tests:**

- Unit tests: inline in source file with `#[cfg(test)] mod tests { ... }`
- Integration tests: `crates/higgs/tests/integration/{feature_name}.rs`
- Test fixtures: `crates/higgs/tests/fixtures/` (create if needed)
- Shared test utils: `crates/higgs/tests/integration/mod.rs`

## Special Directories

**target/**
- Purpose: Cargo build artifacts
- Generated: Yes (by `cargo build`)
- Committed: No (in .gitignore)

**vendor/**
- Purpose: Vendored Rust dependencies
- Generated: No (checked in)
- Committed: Yes (for reproducible builds)

**.planning/codebase/**
- Purpose: GSD analysis documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (by /gsd:map-codebase)
- Committed: Yes

**docs/**
- Purpose: User documentation (guides, API reference)
- Generated: No
- Committed: Yes

## Import Organization

**Crate-wide pattern (from lib.rs and routes):**

```rust
// 1. Standard library
use std::path::Path;
use std::sync::Arc;

// 2. External crates
use axum::{Router, routing::get};
use mlx_rs::Array;
use serde_json::{json, Value};

// 3. Internal crate modules
use crate::config::HiggsConfig;
use crate::state::Engine;
use crate::routes;
```

**Path aliases (if used):**

Check for `#[path = ...]` or workspace.resolver version - currently using standard module organization without aliases.

**Barrel files:**

- `routes/mod.rs`: exports all handlers
- `types/mod.rs`: exports all protocol types
- `cache/mod.rs`: exports BlockAllocator, PagedKvCache, PageTable
- `scheduler/mod.rs`: exports RoundRobinScheduler

No re-exports of external crates at crate root except explicit `pub use tokenizers` in higgs-engine lib.rs.

## Building and Testing

**Build variants:**

```bash
cargo build --release              # Optimized binary
cargo build -p higgs              # Main crate only
cargo build -p higgs-engine       # Engine crate only
```

**Test execution:**

```bash
cargo test -p higgs -- --test-threads=1    # Main crate (required for port binding)
cargo test -p higgs-engine                 # Engine crate
cargo test -p higgs-models                 # Models crate
cargo test                                 # All crates
```

**Linting/formatting:**

```bash
cargo clippy -p higgs              # Check lints (nursery + pedantic enabled)
cargo fmt -p higgs -- --check     # Verify formatting
```

**Key lint rules enforced:**

- unwrap_used, expect_used, panic, todo, unimplemented: DENY
- indexing_slicing: DENY (use indexing guards)
- as_conversions: DENY (no unchecked casting)
- unsafe_code: DENY at workspace level
- Pedantic and Nursery lints: WARN (not enforced to fail)

---

*Structure analysis: 2026-04-03*
