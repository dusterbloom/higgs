# Coding Conventions

**Analysis Date:** 2026-03-31

## Language & Edition

- **Language:** Rust, edition 2024
- **Rust version:** 1.87.0 (minimum)
- **MSRV policy:** Pinned; MSRV-compatibility lints are allowed where features are post-1.87.0

## Naming Patterns

**Files:**
- `snake_case.rs` for all source files (e.g., `model_loader.rs`, `chat_template.rs`, `reasoning_parser.rs`)
- Model architecture files: `{architecture_name}.rs` (e.g., `transformer.rs`, `gemma2.rs`, `phi3.rs`, `qwen3_moe.rs`, `rwkv7.rs`)
- Module files: `mod.rs` for directory modules (e.g., `crates/higgs/src/routes/mod.rs`, `crates/higgs/src/tui/mod.rs`)

**Directories:**
- `kebab-case` for crate names (e.g., `higgs-engine`, `higgs-models`)
- `snake_case` for subdirectories (e.g., `routes/`, `types/`, `tui/views/`)

**Functions:**
- `snake_case` (e.g., `load_tokenizer`, `build_forwarding_headers`, `extract_usage`)
- Test helper functions: `snake_case` with descriptive names (e.g., `assert_masked_500`, `response_status_and_body`)

**Types:**
- `PascalCase` for structs and enums (e.g., `SamplingParams`, `AnyCache`, `ResolvedRoute`, `ServerError`)
- `PascalCase` for type aliases (e.g., `SharedRateLimiter`, `SharedState`)
- `PascalCase` for enum variants (e.g., `AnyModel::Transformer`, `AnyModel::Qwen3Next`, `ResolvedRoute::Remote`)

**Constants:**
- `SCREAMING_SNAKE_CASE` for constants (e.g., `MAX_UNMATCHED_WARNS`)
- `snake_case` for local constants in tests (e.g., `production_toml()`)

**Variables:**
- `snake_case` (e.g., `model_path`, `kv_cache_config`, `top_logprobs`)

## Code Style

**Formatter:**
- Tool: `rustfmt`
- Config: `rustfmt.toml`
  - `max_width = 100`
  - `use_field_init_shorthand = true`
- Enforced via lefthook pre-commit hook: `cargo fmt --all` (auto-fixes) + pre-push check: `cargo fmt --all -- --check`

**Linting:**
- Tool: Clippy
- Severity: Warnings treated as errors (`RUSTFLAGS=-Dwarnings cargo clippy`)
- Workspace-level lint configuration in `Cargo.toml`:
  - `pedantic = "warn"` and `nursery = "warn"` enabled
  - Many restriction lints set to `deny` (see "Restriction Lints" below)

**Restriction Lints (deny):**
The following Clippy restriction lints are enforced at workspace level:
- `unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented`, `unreachable` — **no panicking in production code**
- `indexing_slicing`, `as_conversions` — use safe alternatives
- `dbg_macro`, `print_stdout`, `print_stderr` — use `tracing` instead
- `shadow_reuse`, `shadow_same`, `shadow_unrelated` — no variable shadowing
- `clone_on_ref_ptr`, `rc_buffer`, `rc_mutex` — no reference-counted pointers
- `str_to_string`, `string_add` — use `.to_owned()` and `format!`
- `tests_outside_test_module` — tests must live in `#[cfg(test)] mod tests` blocks
- `self_named_module_files` — module files must not be named `mod.rs` (except for directory modules)
- `float_cmp_const` — use approximate comparisons for floats
- `impl_trait_in_params` — prefer explicit type parameters

**Test Code Exceptions:**
Test modules allow restricted lints via `#[allow(...)]`:
```rust
#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests { ... }
```

## Import Organization

**Order:**
1. `std` imports
2. External crate imports (alphabetical within group)
3. Internal crate imports (using `crate::`)

**Pattern observed:**
```rust
use std::collections::HashMap;
use std::sync::Arc;

use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

use crate::error::ServerError;
use crate::state::AppState;
```

**Wildcard imports:** Not used in production code. Only used in test modules for convenience.

**Path Aliases:**
- No path aliases configured. Uses `crate::` for intra-crate references.
- Workspace crate references use `higgs_engine::`, `higgs_models::` from the `higgs` crate.

## Error Handling

**Pattern:** Layered error enums using `thiserror`

Each crate has its own error enum:
- `higgs-models/src/error.rs` → `ModelError` (MLX, IO, JSON, model-specific)
- `higgs-engine/src/error.rs` → `EngineError` (wraps `ModelError`, MLX, tokenization, template, generation)
- `higgs/src/error.rs` → `ServerError` (wraps `EngineError`, bad request, model not found, proxy)

**Error chain:** `ModelError` → `EngineError` → `ServerError` (each wraps the lower level via `#[from]`)

**Error conversion pattern:**
```rust
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Engine error: {0}")]
    Engine(#[from] higgs_engine::error::EngineError),
    #[error("Bad request: {0}")]
    BadRequest(String),
    // ...
}
```

**Server error responses:**
- Internal errors are **masked** — clients see "Internal server error" with no details
- Bad requests forward the actual message to the client
- All errors produce OpenAI-compatible JSON: `{"error": {"message": "...", "type": "...", "code": null}}`

**Error propagation in routes:**
```rust
pub async fn handler() -> Result<Response, ServerError> {
    let data = something.map_err(|e| ServerError::BadRequest(format!("...: {e}")))?;
    // ...
}
```

**No `unwrap()`/`expect()` in production code:** Enforced by Clippy deny lints. Only permitted in `#[cfg(test)]` blocks.

## Logging

**Framework:** `tracing` (not `log` or `slog`)

**Pattern:**
```rust
tracing::info!(model = %model_path, resolved = %resolved.display(), "Loading model");
tracing::error!(error = %e, "Engine error");
tracing::debug!(url = %url, body_bytes = body.len(), "sending to provider");
tracing::warn!("api_key contains invalid header characters, skipping");
```

**Structured fields:** Use named key-value pairs (not format strings) for structured data:
```rust
// Correct:
tracing::info!(status = %status, url = %url, "provider responded");
// Avoid:
tracing::info!("provider responded with status {}", status);
```

**Verbosity:** Controlled by `--verbose` CLI flag, which sets the tracing filter to `higgs=debug` vs `info`.

**Configuration:** `EnvFilter` reads `RUST_LOG` env var, falling back to the CLI flag.

## Comments & Documentation

**Doc comments:** `///` on all public items (types, functions, methods, structs)
```rust
/// Load a tokenizer from a model directory.
pub fn load_tokenizer<P: AsRef<Path>>(model_dir: P) -> Result<Tokenizer, ModelError> { ... }
```

**Module-level docs:** `//!` on modules with substantive content (e.g., `crates/higgs/src/translate.rs`, integration test files)
```rust
//! Cross-format translation between OpenAI and Anthropic API formats.
```

**Inline comments:** Used sparingly for non-obvious logic:
```rust
// Eval hidden output + all cache states between chunks.
// Without eval, MLX's lazy graph accumulates and OOMs on long sequences.
eval(targets)?;
```

**Section separators:** `// --- Section name ---` or `// -- Subsection ---` used to divide long files into logical sections (seen in `crates/higgs/src/config.rs`, `crates/higgs/src/proxy.rs`)

**`#[allow(...)]` annotations:** Always include a brief justification:
```rust
#[allow(clippy::print_stderr)]
// tower-http deprecated this as "too basic", but it's fine for a local inference server
let auth_layer = ValidateRequestHeaderLayer::bearer(key);
```

## Function Design

**Size:** No explicit limit enforced, but functions tend to stay under ~100 lines. Long functions use `#[allow(clippy::too_many_lines)]`.

**Parameters:**
- Use structs for 3+ related parameters (e.g., `SamplingParams`, `KvCacheConfig`)
- Use `impl AsRef<Path>` for path parameters
- Use generics with trait bounds for cache types: `fn cached_sdp<C: KeyValueCache>(...)`

**Return Values:**
- `Result<T, ErrorType>` for fallible operations
- `Option<T>` for genuinely optional values
- Never return `()` from fallible operations — always use `Result`

**Async:**
- Route handlers are `async fn` returning `Result<Response, ServerError>`
- CPU-bound work wrapped in `tokio::task::spawn_blocking` (e.g., auto-router classification)

## Module Design

**Exports:** Public types/functions explicitly listed in `lib.rs`:
```rust
pub mod config;
pub mod error;
pub mod router;
// ...
```

**Re-exports:** Used for commonly accessed types:
```rust
pub use qwen3_next::{LayerCache, MtpHead, Qwen3NextCausalLM};
pub use transformer::{Model, ModelArgs};
```

**Test stubs:** Test-only variants using `#[cfg(test)]`:
```rust
pub enum Engine {
    Simple(Box<SimpleEngine>),
    Batch(Box<BatchEngine>),
    #[cfg(test)]
    Stub(String),
}
```

**Feature flags:** Conditional compilation for platform-specific code:
```rust
#[cfg(feature = "ane")]
pub mod diffusion_ane;
```

## Serialization

**Serde pattern:**
- `#[derive(Debug, Clone, Serialize, Deserialize)]` for config types
- `#[serde(default)]` for optional fields with sensible defaults
- `#[serde(skip_serializing_if = "Option::is_none")]` to omit null fields from JSON output
- `#[serde(rename_all = "lowercase")]` for enum serialization
- `#[serde(rename = "field_name")]` for field name mapping

**Config defaults:** Use dedicated `fn default_field() -> Type` functions:
```rust
fn default_host() -> String { "0.0.0.0".to_owned() }
const fn default_port() -> u16 { 8000 }
```

## Build & CI Hooks

**Pre-commit (lefthook):**
- `cargo fmt --all` — auto-formats, stages fixed files

**Pre-push (lefthook):**
- `cargo fmt --all -- --check` — verifies formatting
- `RUSTFLAGS=-Dwarnings cargo clippy --all-targets --all-features` — all warnings are errors

---

*Convention analysis: 2026-03-31*
