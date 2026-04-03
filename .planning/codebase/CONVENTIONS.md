# Coding Conventions

**Analysis Date:** 2026-04-03

## Naming Patterns

**Files:**
- Snake_case for module files: `chat_template.rs`, `prompt_cache.rs`, `round_robin.rs`
- Test files co-located in source: `#[cfg(test)] mod tests` at file end
- Subdirectories for logical grouping: `src/cache/`, `src/scheduler/`, `src/routes/`

**Functions:**
- Snake_case for all function names: `test_apply_without_generation_prompt`, `create_session`, `append_token`
- Test functions prefixed with `test_`: `test_round_robin_rotates`, `test_scheduler_remove_works`
- Public functions clearly named for intent: `from_model_dir`, `try_from_model_dir`, `apply_with_thinking`

**Variables:**
- Snake_case for bindings: `session_id`, `kv_dim`, `current_tokens`, `block_size`
- Descriptive names for loop variables: `(token_idx, _)` not single letters
- Mutable state explicitly marked: `mut scheduler`, `mut cache`, `mut child`

**Types:**
- PascalCase for structs and enums: `ChatTemplateRenderer`, `PagedKvCache`, `RoundRobinScheduler`, `ServerError`
- Generic type parameters short and clear: `T`, `S`, `E` for error types
- Newtype patterns for semantic clarity: `SessionId` alias for `u64`

**Constants:**
- SCREAMING_SNAKE_CASE for module constants: `DEFAULT_PREFIX_CACHE_SIZE`, `MIN_PREFIX_LEN`
- Const comments explaining intent: `const MIN_PREFIX_LEN: usize = 16; // Minimum prefix length (in tokens)`

## Code Style

**Formatting:**
- Tool: `rustfmt` with custom config
- Max line width: 100 characters (`max_width = 100`)
- Field init shorthand enabled (`use_field_init_shorthand = true`)
- Run via: `cargo fmt -p <crate>`

**Linting:**
- Tool: `clippy` with strict rules
- Workspace lints enforced in `Cargo.toml`: `[workspace.lints]`
- Pedantic and nursery lints enabled at warn level
- Restriction lints denied (strict enforcement):
  - `unwrap_used`, `expect_used`, `panic` - **DENIED** use Result/Option methods instead
  - `todo`, `unimplemented`, `unreachable` - **DENIED** handle all cases
  - `indexing_slicing` - **DENIED** use safe indexing with `.get()`
  - `as_conversions` - **DENIED** use `.try_from()` or explicit conversion helpers
  - `dbg_macro`, `print_stdout`, `print_stderr` - **DENIED** use `tracing`
  - `shadow_reuse`, `shadow_same`, `shadow_unrelated` - **DENIED** no variable shadowing
  - `tests_outside_test_module` - **DENIED** all tests in `#[cfg(test)]`
- Run via: `cargo clippy -p <crate> -- -D warnings`

**Comment Style:**
- Document public APIs with doc comments: `/// [description]`
- Explain "why" not "what" in comments: Non-obvious logic, assumptions, constraints
- Test modules may allow `#![allow(...)]` for test-specific lints
- SPDX license headers on kernel files: `// SPDX-License-Identifier: Apache-2.0`

## Import Organization

**Order:**
1. Standard library: `use std::*`
2. External crates: `use crate_name::*`, `use serde::*`
3. Local modules: `use crate::*`

**Example from `simple.rs`:**
```rust
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use higgs_models::{AnyCache, AnyModel, LogprobArrays, SamplingParams, apply_penalties, sample};
use mlx_rs::{Array, Dtype, Stream, ops::indexing::{IndexOp, NewAxis}, transforms::{async_eval, eval}};
use tokenizers::Tokenizer;

use crate::{
    cache::PagedKvCache,
    chat_template::{ChatMessage, ChatTemplateRenderer},
    engine::{GenerationOutput, StreamingOutput},
    scheduler::RoundRobinScheduler,
    spec_prefill::{SpecPrefillConfig, SpecPrefillEngine},
    error::EngineError,
    model_loader,
    prompt_cache::PrefixCache,
};
```

**Patterns:**
- Group related imports on same line: `use std::sync::{Mutex, MutexGuard}`
- Nested paths with `::`: `use mlx_rs::ops::indexing::{IndexOp, NewAxis}`
- Crate-local imports grouped in `use crate::{...}` block
- No bare glob imports in production code

## Error Handling

**Pattern: Result-based errors with thiserror:**

Use `thiserror::Error` enum for public error types (`crate/src/error.rs`):
```rust
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Engine error: {0}")]
    Engine(#[from] higgs_engine::error::EngineError),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}
```

**Conversion to HTTP responses via IntoResponse:**
```rust
impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            Self::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                msg.clone(),
            ),
            Self::InternalError(msg) => {
                tracing::error!(error = %msg, "Internal error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "Internal server error".to_owned(),
                )
            }
        };
        // ... build response
    }
}
```

**Key principles:**
- Internal errors masked to client: "Internal server error" returned instead of details
- Serde errors on deserialization trigger 400 Bad Request before handler runs
- Engine errors trigger 500 Internal Server Error with masked message
- Result types preferred: `Result<T, E>` over panics
- Error messages propagated via context, not swallowed

## Logging

**Framework:** `tracing` crate (structured logging)

**Patterns:**
- Span/level: `tracing::error!()`, `tracing::warn!()`, `tracing::info!()`
- Structured fields: `tracing::info!(field = %value, "message")`
- Error context: `%err` interpolation for error Display

**Examples from codebase:**
```rust
tracing::error!(error = %e, "Engine error");
tracing::error!(error = %msg, "Internal error");
tracing::warn!("chat_template field present but not a string or valid array");
tracing::info!(
    max_recommended_mb = max_rec / (1024 * 1024),
    memory_limit_mb = mem_limit / (1024 * 1024),
    "MLX memory limits {} (prev mem={}MB, cache={}MB)",
    if limits_enabled { "set" } else { "skipped" },
    prev_mem / (1024 * 1024),
    prev_cache / (1024 * 1024),
);
```

**No console output in production code:**
- Denied by clippy: `print_stdout`, `print_stderr`
- Use `tracing` instead
- Test code allowed via `#![allow(clippy::print_stderr)]`

## Comments

**When to comment:**
- Non-obvious algorithms or performance decisions
- Safety invariants for unsafe code blocks
- Constraints or assumptions
- Links to external documentation or bugs

**Example (interior mutability):**
```rust
/// Interior mutability so `find_longest_prefix` can update access time
/// through a shared reference during tree traversal.
last_accessed: Cell<Instant>,
```

**Example (unsafe rationale):**
```rust
/// Cap MLX memory allocations to avoid Metal OOM on constrained GPUs.
/// Sets `mlx_set_memory_limit` to 75% of `max_recommended_working_set_size`
/// and `mlx_set_cache_limit` to 50%.
#[allow(unsafe_code)]
pub(crate) fn set_wired_limit_to_max() {
    unsafe {
        // ... implementation
    }
}
```

**Doc comments (///)**
- Required on public functions and types
- Explain what and why, not how
- Include examples for complex behaviors
- Link related types/functions with backticks

## Function Design

**Size:** Small, single-responsibility functions

**Parameters:**
- Prefer ownership transfer or borrowed references
- Slices `&[T]` over vectors for input
- Result type for fallible operations: `Result<T, E>`
- No unwrap/expect in public APIs

**Return values:**
- Result for fallible ops: `fn create(id: u64) -> Result<(), Error>`
- Options for nullable: `fn get_block(idx: u32) -> Option<u32>`
- Direct values for simple getters: `fn num_tokens(&self) -> usize`

**Example from `PagedKvCache`:**
```rust
/// Create a new session.
pub fn create_session(&mut self, session_id: u64) -> Result<(), CacheError> {
    if self.page_table.has_session(session_id) {
        return Err(CacheError::SessionNotFound(session_id));
    }
    self.session_tokens.insert(session_id, 0);
    Ok(())
}

/// Get number of tokens in this session.
pub fn num_tokens(&self) -> usize {
    self.num_tokens
}

/// Get current block ID for a token position (0-indexed).
pub fn block_for_token(&self, token_idx: usize) -> u32 {
    let block_size = self.layout.block_size;
    let block_idx = token_idx / block_size;
    self.blocks[block_idx]  // Safe: blocks allocated for num_tokens
}
```

## Module Design

**Exports:**
- Explicit pub items only (no glob re-exports in production)
- Re-export convenience types at crate root: `pub use qwen3_next::{LayerCache, Qwen3NextCausalLM}`
- Private implementation details behind pub interface

**Example from `crates/higgs-engine/src/lib.rs`:**
```rust
pub mod batch_engine;
pub mod cache;
pub mod chat_template;
pub mod engine;
pub mod error;
pub mod model_loader;
pub mod simple;
pub mod spec_prefill;
pub mod tool_parser;

pub use tokenizers;  // Re-export for consumers
```

**Barrel files:**
- Not used for glob imports
- Used for explicit re-exports from public modules
- Consolidate related types at crate boundary

**Submodule organization:**
- `mod tests` blocks at end of file for unit tests
- `#[cfg(test)]` guards all test code
- No test functions outside `#[cfg(test)]` (clippy enforces)

---

*Convention analysis: 2026-04-03*
