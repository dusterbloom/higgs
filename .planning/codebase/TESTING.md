# Testing Patterns

**Analysis Date:** 2026-04-03

## Test Framework

**Runner:**
- `cargo test` (built-in Rust test framework)
- Single-threaded execution required: `cargo test -p higgs -- --test-threads=1` (due to shared port bindings in integration tests)
- Async tests use `#[tokio::test]` macro for async function execution

**Test organization:**
- Unit tests: `#[cfg(test)] mod tests { ... }` at file end
- Integration tests: `crates/higgs/tests/integration/` directory
- Test modules included in binary when testing

**Run commands:**
```bash
# Unit + integration tests (higgs crate, single-threaded)
cargo test -p higgs -- --test-threads=1

# Specific crate
cargo test -p higgs-engine

# Specific test
cargo test -p higgs test_name

# Verbose output
cargo test -p higgs -- --nocapture --test-threads=1
```

## Test File Organization

**Location:**
- Co-located: Unit tests in same file as implementation (preferred for engine, models crates)
- Separate: Integration tests in `crates/higgs/tests/integration/` directory
- Pattern: `#[cfg(test)]` guards all test code

**Naming:**
- Test functions: `#[test] fn test_<what_is_being_tested>()`
- Async tests: `#[tokio::test] async fn <name>()`
- Test modules: `#[cfg(test)] mod tests { ... }`

**Structure (integration tests):**
```
tests/
├── integration_tests.rs         # Entrypoint (mod integration;)
├── integration/
│   ├── mod.rs                  # Module exports
│   ├── error_contract.rs       # Error handling contract tests
│   ├── request_validation.rs   # Request deserialization tests
│   ├── response_contract.rs    # Response format tests
│   ├── router.rs               # HTTP routing tests
│   ├── cli_exec.rs             # CLI integration tests
│   └── proxy_e2e.rs            # End-to-end proxy tests
```

## Test Structure

**Suite organization (unit tests in `#[cfg(test)]` block):**
```rust
#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module
)]
mod tests {
    use super::*;

    #[test]
    fn test_specific_behavior() {
        // Arrange
        let cache = PagedKvCache::new(1024, 64, 2, 128);

        // Act
        cache.create_session(1u64).unwrap();

        // Assert
        assert_eq!(cache.session_token_count(1u64), Some(0));
    }
}
```

**Integration tests:**
```rust
//! Module-level documentation explaining what's being tested.
//!
//! These tests verify that X works correctly under Y conditions.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module,
    clippy::needless_pass_by_value
)]

use higgs::types::openai::ChatCompletionRequest;

#[test]
fn test_chat_request_minimal_valid() {
    let json = r#"{"model": "test-model", "messages": [...]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "test-model");
}
```

**Patterns:**
- Arrange-Act-Assert structure (even if implicit)
- Comments explaining non-obvious test setup
- Direct assertions on behavior, not implementation
- Dedicated helper functions for repeated setup

## Test Structure Examples

**Helper function pattern (from `router.rs`):**
```rust
/// Build a minimal router with just the health endpoint for testing.
fn build_health_only_router() -> axum::Router {
    use axum::routing::get;
    axum::Router::new().route("/health", get(higgs::routes::health::health))
}

/// Send a request to the health-only router and return the response.
async fn send_request(
    method: &str,
    uri: &str,
    headers: &[(&str, &str)],
) -> axum::http::Response<axum::body::Body> {
    let app = build_health_only_router();
    let mut builder = Request::builder().method(method).uri(uri);
    for (key, value) in headers {
        builder = builder.header(*key, *value);
    }
    app.oneshot(builder.body(axum::body::Body::empty()).unwrap())
        .await
        .unwrap()
}

#[tokio::test]
async fn health_returns_200_with_json() {
    let response = send_request("GET", "/health", &[]).await;
    assert_eq!(response.status(), StatusCode::OK);
}
```

**Test data generation pattern (from `cli_exec.rs`):**
```rust
/// Write a minimal valid config.toml pointing at the given port.
fn write_test_config(port: u16) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    let mut f = std::fs::File::create(&config_path).unwrap();
    write!(
        f,
        "[server]\nhost = \"127.0.0.1\"\nport = {port}\n\n\
         [provider.dummy]\nurl = \"http://127.0.0.1:1\"\n"
    )
    .unwrap();
    dir
}
```

**Async async test pattern:**
```rust
#[tokio::test]
async fn bad_request_returns_400() {
    let error = ServerError::BadRequest("field 'model' is required".to_owned());
    let (status, body, content_type) = extract_response(error).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(content_type.contains("application/json"));
    assert_eq!(body["error"]["message"], "field 'model' is required");
}

async fn extract_response(error: ServerError) -> (StatusCode, serde_json::Value, String) {
    let response = error.into_response();
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default();
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    (status, body, content_type)
}
```

## Mocking

**Framework:** `wiremock` for HTTP mocking (in `Cargo.toml` dev-dependencies)

**When to mock:**
- External services/APIs (Anthropic, OpenAI providers)
- HTTP endpoints that are tested separately
- Avoid mocking internal functions

**When NOT to mock:**
- Core inference logic (tested with real models or fixtures)
- Request/response serialization (test actual serde behavior)
- Error handling paths (test actual error types)

**Example integration test setup (from `proxy_e2e.rs`):**
- Uses `wiremock::MockServer` to stand up fake upstream providers
- Tests request forwarding, response proxying, error handling
- Does not test actual model inference

## Fixtures and Factories

**Test data patterns:**

**Simple helper functions:**
```rust
fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_owned(),
        content: content.to_owned(),
        tool_calls: None,
    }
}

#[test]
fn test_apply_with_assistant_role() {
    let messages = vec![msg("user", "What is 2+2?"), msg("assistant", "4")];
    // ... test logic
}
```

**Temporary file fixtures:**
```rust
#[test]
fn test_from_model_dir_standalone_jinja_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("chat_template.jinja"),
        r"{%- for message in messages %}{{ message.content }}{%- endfor %}",
    ).unwrap();
    let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
    // ... test logic
}
```

**Location:**
- Inline in test functions for clarity
- Extracted to module-level helper functions only if reused across multiple tests
- Use `tempfile` crate for temporary directories (automatically cleaned up)

## Coverage

**Requirements:** No enforced coverage target

**View coverage:**
- Run tests and check which lines were executed: `cargo test`
- No specific coverage reporting tool configured
- Code review relies on inspecting test behavior

## Test Types

**Unit Tests (co-located in modules):**
- Scope: Single function or tight component group
- Approach: Fast, deterministic, exhaustive path coverage
- Location: `#[cfg(test)]` mod tests` at file end
- Example: `test_paged_cache_session_lifecycle`, `test_round_robin_rotates`
- Dependencies: May depend on real types but avoid external services

**Integration Tests (in `tests/integration/` directory):**
- Scope: Multiple modules working together (routing, error handling, request/response contracts)
- Approach: Exercise public APIs, verify contracts, test without real engine when possible
- Location: `crates/higgs/tests/integration/`
- Example: `test_chat_request_minimal_valid`, `health_returns_200_with_json`
- Dependencies: May use `tower::ServiceExt::oneshot()` to test without full server

**E2E Tests (marked #[ignore]):**
- Scope: Full system with real engine
- Approach: Require actual model weights, slow, only run when explicitly requested
- Location: Integration test files with `#[ignore]` attribute
- Comment: Explain why they're ignored (e.g., "requires model weights, slow")
- Run via: `cargo test -- --ignored`

## Common Patterns

**Testing error cases (from `error_contract.rs`):**
```rust
/// Asserts that the given error produces a masked 500 response
/// and the leaked detail does not appear in the client message.
async fn assert_masked_500(error: ServerError, leaked_detail: &str) {
    let (status, body, _) = extract_response(error).await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let message = body["error"]["message"].as_str().unwrap();
    assert_eq!(message, "Internal server error");
    assert!(
        !message.contains(leaked_detail),
        "Internal error detail leaked to client: {leaked_detail}"
    );
}

#[tokio::test]
async fn internal_error_returns_500_with_masked_message() {
    assert_masked_500(
        ServerError::InternalError("database unreachable".to_owned()),
        "database unreachable",
    ).await;
}
```

**Testing CLI commands (from `cli_exec.rs`):**
```rust
#[test]
fn exec_forwards_child_exit_code() {
    let (_listener, dir) = fake_server_env();

    let output = Command::new(higgs_bin())
        .args(["exec", "--", "sh", "-c", "exit 42"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(42),
        "expected exit code 42, got {:?}",
        output.status.code()
    );
}
```

**Testing state transitions (from `paged.rs`):**
```rust
#[test]
fn test_paged_cache_session_lifecycle() {
    let mut cache = PagedKvCache::new(1024, 64, 2, 128);
    let session_id = 1u64;

    // Create
    cache.create_session(session_id).unwrap();

    // Modify
    let k_data = vec![f16::ZERO; 256];
    let v_data = vec![f16::ZERO; 256];
    cache.append_token(session_id, &k_data, &v_data).unwrap();

    // Verify
    assert_eq!(cache.session_token_count(session_id), Some(1));

    // Cleanup
    cache.remove_session(session_id).unwrap();
    assert!(cache.get_session_view(session_id).is_none());
}
```

**Testing error conditions:**
```rust
#[test]
fn test_paged_cache_out_of_blocks() {
    let mut cache = PagedKvCache::new(2, 4, 2, 128);  // Only 2 blocks
    cache.create_session(1u64).unwrap();

    // Append 8 tokens (fits in 2 blocks with block_size=4)
    for _ in 0..8 {
        cache.append_token(1u64, &k_data, &v_data).unwrap();
    }

    // 9th token requires 3 blocks, should error
    let err = cache
        .append_token(1u64, &k_data, &v_data)
        .unwrap_err();
    assert!(matches!(err, CacheError::OutOfBlocks { .. }));
}
```

## Lint Configuration for Tests

**Test module lints (applied via `#![allow(...)]`):**

```rust
#[cfg(test)]
#[allow(
    clippy::panic,           // assert! allowed in tests
    clippy::unwrap_used,     // unwrap() allowed for test setup
    clippy::indexing_slicing,  // Direct indexing allowed for simple test data
    clippy::tests_outside_test_module  // Kept below file-level when needed
)]
mod tests {
    // ...
}
```

**Integration test lint header (from integration test files):**
```rust
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module,
    clippy::needless_pass_by_value  // function params don't need to be borrows in tests
)]
```

---

*Testing analysis: 2026-04-03*
