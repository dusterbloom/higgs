# Conventions

Coding conventions, style, and patterns used across the Higgs codebase.

## Language & Toolchain

- **Rust edition:** 2021
- **Toolchain:** Rust 1.87.0
- **Formatter:** `rustfmt` — max line width 100
- **Linter:** `clippy` with `pedantic` + `nursery` lints enabled
- Enforced via `cargo fmt -p higgs -- --check` and `cargo clippy -p higgs`

## Naming

| Kind | Convention | Example |
|------|------------|---------|
| Functions / variables / modules | `snake_case` | `load_model`, `router_state` |
| Types / structs / enums / traits | `PascalCase` | `ChatTemplate`, `EngineError` |
| Constants / statics | `SCREAMING_SNAKE_CASE` | `DEFAULT_PORT` |
| Lifetimes | short lowercase | `'a`, `'ctx` |

## Forbidden Patterns

Clippy lints actively deny these in non-test code:

- `unwrap()`
- `expect()`
- `panic!()`
- `todo!()` / `unimplemented!()`
- `dbg!()`
- `print!` / `println!` / `eprint!` / `eprintln!` macros

Tests opt out with file-level `#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]`.

## Error Handling

- Custom error enums per crate using `thiserror`.
- `#[from]` conversions for automatic error propagation.
- Prefer `Result<T, CrateError>` return types over panicking.
- Example: `crates/higgs-engine/src/error.rs` defines `EngineError` with `#[from]` for `std::io::Error`, serde errors, etc.

```rust
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("template error: {0}")]
    Template(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

## Logging

- Uses the `tracing` crate — never `println!` for diagnostic output.
- Structured key-value logging:

```rust
tracing::info!(model = %name, tokens = tok_count, "prefill complete");
tracing::warn!(error = %e, "falling back to cpu path");
```

- Levels: `error!`, `warn!`, `info!`, `debug!`, `trace!`.

## Documentation

- `///` doc comments required on public structs, enums, and functions.
- First line is a one-sentence summary; further paragraphs explain invariants and safety.
- Keep one-line summaries for obvious getters.

## Style Preferences

- Field-init shorthand: `Foo { bar, baz }` not `Foo { bar: bar, baz: baz }`.
- `async fn` with `.await` at call sites — no manual `Future` impls unless necessary.
- Prefer iterator chains over manual indexing loops.
- Use `?` for error propagation.
- Prefer borrowing (`&str`, `&[T]`) over owned parameters unless the function needs ownership.

## Module Organization

- `mod.rs` not used — modules declared in their file (`foo.rs` with `mod bar;` inside).
- Public re-exports at crate root (`lib.rs`) keep the public surface flat.
- Internal helpers kept `pub(crate)` or `pub(super)`.

## Concurrency

- Tokio is the async runtime.
- Shared state wrapped in `Arc<T>` with interior mutability via `tokio::sync::{Mutex, RwLock}` or `parking_lot` for non-async contexts.
- Channels: `tokio::sync::mpsc` / `oneshot` for task communication.

## Config & Doctor Rule

Project rule (from `CLAUDE.md`): when adding or changing config fields, update `crates/higgs/src/doctor.rs` to validate the new field. Doctor must catch misconfiguration before the server starts.

## Documentation Rule

When changing user-facing behavior (config fields, CLI flags, API surface), update:

1. `README.md` — config examples and reference tables.
2. `crates/higgs/src/daemon.rs` — the `higgs init` config template.
3. Doc comments on public structs/fields.

## Summary

The codebase enforces a strict, explicit style: no panics in production code, structured tracing, thiserror for errors, clippy pedantic+nursery as gates. Tests are the only place where relaxed lints apply.
