# Testing

Test organization, frameworks, and conventions for the Higgs codebase.

## Frameworks & Dependencies

| Purpose | Crate |
|---------|-------|
| Test runner | Built-in `cargo test` |
| Async tests | `tokio` with `#[tokio::test]` attribute |
| HTTP mocking | `wiremock` |
| Temp filesystems | `tempfile` |
| HTTP test layer | `tower` (Service / Layer composition) |

## Test Layout

**Unit tests** — inline in source files inside `#[cfg(test)] mod tests { ... }` blocks.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_template() { /* ... */ }
}
```

**Integration tests** — under `crates/higgs/tests/integration/`, organized by concern:

- `crates/higgs/tests/integration/error_contract.rs` — error response shape
- `crates/higgs/tests/integration/request_validation.rs` — input validation (~50 tests)
- `crates/higgs/tests/integration/response_contract.rs` — response serialization
- `crates/higgs/tests/integration/router.rs` — router layer behavior
- `crates/higgs/tests/integration/proxy_e2e.rs` — end-to-end proxy flow using `wiremock`

Integration tests use a relaxed clippy profile via file-level:

```rust
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
```

## Async Tests

```rust
#[tokio::test]
async fn returns_503_when_model_missing() {
    let state = build_test_state().await;
    let res = send_request(&state, msg("hi")).await;
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
}
```

## Test Helpers

No mocking framework. Helpers are plain functions named for intent:

- `msg(body: &str) -> Request` — build a request
- `extract_response(res) -> Value` — parse JSON body
- `build_test_state() -> RouterState` — assemble app state for a test

Kept near the tests that use them (often in the same integration file or a shared `tests/integration/common.rs`).

## HTTP Mocking

External LLM / proxy targets stubbed with `wiremock`:

```rust
let mock = MockServer::start().await;
Mock::given(method("POST"))
    .and(path("/v1/chat/completions"))
    .respond_with(ResponseTemplate::new(200).set_body_json(fixture))
    .mount(&mock)
    .await;
```

## Running Tests

Project rule: tests must run with `--test-threads=1` due to shared port bindings.

```bash
cargo test -p higgs -- --test-threads=1
```

Other crates run with default parallelism:

```bash
cargo test -p higgs-engine
cargo test -p higgs-models
```

## Notable Test Files

| File | What it covers |
|------|----------------|
| `crates/higgs-engine/src/chat_template.rs` | 58+ unit tests for chat template parsing |
| `crates/higgs-engine/src/error.rs` | Error type construction and conversions |
| `crates/higgs/tests/integration/request_validation.rs` | ~50 validation cases |
| `crates/higgs/tests/integration/proxy_e2e.rs` | End-to-end proxy with `wiremock` |
| `crates/higgs/tests/integration/router.rs` | Router layering |

## Coverage Philosophy

From `CLAUDE.md` user rules: "keep the count of new tests to a minimal yet their quality and coverage the highest" and "we write tests to prove general logic not custom made assertions."

- Prefer one well-targeted test over several redundant ones.
- Test contracts (JSON shape, error codes) rather than internal plumbing.
- E2E tests reserved for proxy flows; most logic covered by unit + integration.

## What's Missing

- No benchmarks under `#[bench]` (inference benchmarks live in `benchmarks/` as standalone scripts, not cargo bench harness).
- No property-based testing (no `proptest` / `quickcheck`).
- No snapshot testing (no `insta`).
- Coverage tooling not wired into CI (no `cargo-tarpaulin` / `llvm-cov` config visible).
