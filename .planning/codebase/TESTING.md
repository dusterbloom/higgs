# Testing Patterns

**Analysis Date:** 2026-03-31

## Test Framework

**Runner:**
- Rust built-in test harness (`cargo test`)
- Async runtime: `tokio` (via `#[tokio::test]`)

**Assertion Library:**
- Standard `assert!`, `assert_eq!`, `assert!()` macros
- No external assertion crates

**Run Commands:**
```bash
cargo test                          # Run all tests across all crates
cargo test -p higgs                 # Run tests for the higgs crate only
cargo test -p higgs-engine          # Run tests for the engine crate only
cargo test -p higgs-models          # Run tests for the models crate only
cargo test -- --nocapture           # Run with stdout output visible
cargo test -- --test-threads=1      # Run tests sequentially
```

**Test Counts:**
- ~1003 total `#[test]` functions
- ~57 `#[tokio::test]` async tests
- Integration tests: ~35 tests across 6 modules

## Test File Organization

**Location:** Tests are co-located with source code using inline `#[cfg(test)] mod tests` blocks within each `.rs` file. Integration tests live in a separate `tests/` directory.

**Structure:**
```
crates/
├── higgs/
│   ├── src/
│   │   ├── error.rs              # Contains mod tests { ... }
│   │   ├── config.rs             # Contains mod tests { ... }
│   │   ├── proxy.rs              # Contains mod tests { ... }
│   │   ├── router.rs             # Contains mod tests { ... }
│   │   ├── translate.rs          # Contains mod tests { ... }
│   │   └── ...
│   └── tests/
│       ├── integration_tests.rs  # Entry point: mod integration;
│       └── integration/
│           ├── mod.rs            # Submodule declarations
│           ├── cli_exec.rs       # CLI integration tests
│           ├── error_contract.rs # Error response contract tests
│           ├── proxy_e2e.rs      # Wiremock-based proxy E2E tests
│           ├── request_validation.rs  # Request deserialization tests
│           ├── response_contract.rs   # Response serialization tests
│           └── router.rs         # Router resolution tests
├── higgs-engine/
│   └── src/
│       ├── error.rs              # Contains mod tests { ... }
│       ├── chat_template.rs      # Contains mod tests { ... }
│       └── ...
└── higgs-models/
    └── src/
        ├── lib.rs                # Contains mod tests { ... }
        ├── utils.rs              # Contains mod tests { ... }
        ├── error.rs              # Contains mod tests { ... }
        └── ...
```

**Naming:**
- Test functions: `snake_case` descriptive names (e.g., `test_engine_error_returns_500_with_masked_message`, `test_cached_scaled_dot_product_attention_matches_dense_turbo_adapter`)
- No prefix convention — some use `test_` prefix, others don't
- Test helper functions: descriptive `snake_case` (e.g., `assert_masked_500`, `response_status_and_body`, `config_from_toml`, `router_from_toml`)

## Test Structure

**Suite Organization:**
Every source file with tests follows this pattern:
```rust
#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    // Helper functions (private to the test module)
    fn helper_function() -> ... { ... }

    // Individual test functions
    #[test]
    fn descriptive_name() { ... }

    #[tokio::test]
    async fn async_descriptive_name() { ... }
}
```

**Test Module Attributes:** Always include `#[allow(...)]` to permit restricted operations:
```rust
#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests { ... }
```

Integration test files add additional allows:
```rust
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module
)]
```

**Setup Pattern:** No global setup/teardown. Each test constructs its own fixtures inline or via helper functions.

**Teardown Pattern:** None required. Tests use `tempfile::tempdir()` for filesystem resources, which auto-clean on drop.

## Mocking

**Framework:** No mocking framework. The codebase uses:
1. **Test stubs** — `#[cfg(test)]` enum variants (e.g., `Engine::Stub(String)`)
2. **Wiremock** — HTTP mock server for integration tests (`wiremock = "0.6"` in dev-dependencies)
3. **Real lightweight construction** — create real objects with minimal data (e.g., small config structs, simple arrays)

**Test Stubs:**
```rust
// In crates/higgs/src/state.rs
pub enum Engine {
    Simple(Box<SimpleEngine>),
    Batch(Box<BatchEngine>),
    #[cfg(test)]
    Stub(String),  // Only compiled in test mode
}

#[cfg(test)]
pub fn test_stub(name: &str) -> Self {
    Self::Stub(name.to_owned())
}
```

**Wiremock (Integration):**
```rust
// In crates/higgs/tests/integration/proxy_e2e.rs
let mock_server = MockServer::start().await;
Mock::given(method("POST"))
    .and(path("/v1/chat/completions"))
    .respond_with(ResponseTemplate::new(200).set_body_json(&response))
    .mount(&mock_server)
    .await;
```

**What to Mock:**
- HTTP upstream providers (wiremock)
- Local engine loading (test stubs or `Engine::Stub`)
- Filesystem config files (`tempfile::tempdir()` + write TOML)

**What NOT to Mock:**
- Internal business logic (test real code paths)
- Error types (test real error enums)
- Serialization (test real serde behavior)

## Fixtures and Factories

**Test Data:** Inline construction, no fixture files:
```rust
fn small_qwen3_moe_args() -> qwen3_moe::Qwen3MoeModelArgs {
    qwen3_moe::Qwen3MoeModelArgs {
        model_type: "qwen3_moe".to_owned(),
        hidden_size: 32,
        num_hidden_layers: 2,
        // ... minimal valid config
    }
}
```

**Config Fixtures:** TOML strings written to temp files:
```rust
fn config_from_toml(toml: &str) -> HiggsConfig {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(&path, toml).unwrap();
    load_config_file(&path, None).unwrap()
}

fn router_from_toml(toml: &str) -> Router {
    let config = config_from_toml(toml);
    Router::from_config(&config, HashMap::new()).unwrap()
}
```

**Shared Test Data Functions:** Helper functions that construct common test objects:
```rust
fn make_chat_chunk(id: &str, delta: ChatCompletionDelta, finish_reason: Option<String>) -> ChatCompletionChunk { ... }
fn make_tool_call(id: &str, name: &str, arguments: &str) -> ToolCall { ... }
fn openai_chat_response() -> serde_json::Value { ... }
```

**Location:** Fixtures live within the same test module or integration test file that uses them.

## Coverage

**Requirements:** No formal coverage threshold enforced.

**View Coverage:** No coverage tooling configured. Run:
```bash
cargo install cargo-tarpaulin
cargo tarpaulin --out Html
```

## Test Types

**Unit Tests:**
- Co-located in `#[cfg(test)] mod tests` blocks
- Test individual functions, types, error conversions
- No external dependencies beyond `tempfile`
- Examples: error display tests, sampling tests, causal mask creation, config validation, key remapping

**Integration Tests:**
- Located in `crates/higgs/tests/integration/`
- Six modules covering different concerns:
  - `error_contract.rs` — Server error HTTP contract (status codes, JSON shape, masking)
  - `request_validation.rs` — Serde deserialization of OpenAI/Anthropic request types
  - `response_contract.rs` — Response type serialization to OpenAI/Anthropic JSON
  - `router.rs` — Router resolution logic (pattern matching, defaults, auto-router)
  - `proxy_e2e.rs` — Full proxy E2E with wiremock (passthrough, cross-format translation, metrics)
  - `cli_exec.rs` — CLI binary integration tests (exit codes, signal handling)

**E2E Tests:**
- `proxy_e2e.rs` uses `tower::ServiceExt::oneshot` to send requests through the real axum router
- `cli_exec.rs` spawns the actual `higgs` binary as a child process

## Common Patterns

**Error Testing:**
```rust
#[test]
fn error_describes_correctly() {
    let err = ModelError::UnsupportedModel("gpt5".to_owned());
    assert_eq!(err.to_string(), "Unsupported model type: gpt5");
}

#[test]
fn from_io_error_converts() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
    let model_err: ModelError = io_err.into();
    assert!(matches!(model_err, ModelError::Io(_)));
}
```

**Async Testing:**
```rust
#[tokio::test]
async fn remote_pattern_resolves_to_anthropic() {
    let router = router_from_toml(production_toml());
    let route = router.resolve("claude-opus-4-6", None).await.unwrap();
    match route {
        ResolvedRoute::Remote { provider_name, .. } => {
            assert_eq!(provider_name, "anthropic");
        }
        ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
    }
}
```

**Validation/Rejection Testing:**
```rust
#[test]
fn test_simple_mode_no_models_rejected() {
    let args = ServeArgs { models: vec![], ... };
    assert!(build_simple_config(&args).is_err());
}

#[test]
fn chat_request_missing_model_fails() {
    let json = r#"{"messages": [{"role": "user", "content": "hi"}]}"#;
    let result = serde_json::from_str::<ChatCompletionRequest>(json);
    assert!(result.is_err());
}
```

**Equivalence Testing (quantized vs unquantized):**
```rust
#[test]
fn test_turboquant_attention_vs_unquantized_ground_truth() {
    // ... setup quantized and dense caches ...
    let dense_out = scaled_dot_product_attention(...).unwrap();
    let turbo_out = cached_scaled_dot_product_attention(...).unwrap();
    // Compare via cosine similarity
    assert!(cos > 0.85, "TurboQuant vs dense: cos={cos:.6} (need > 0.85)");
}
```

**HTTP Contract Testing:**
```rust
#[tokio::test]
async fn bad_request_returns_400() {
    let error = ServerError::BadRequest("field 'model' is required".to_owned());
    let (status, body, content_type) = extract_response(error).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(content_type.contains("application/json"));
    assert_eq!(body["error"]["type"], "invalid_request_error");
}
```

**Wiremock E2E Pattern:**
```rust
#[tokio::test]
async fn proxy_openai_passthrough() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock_server)
        .await;

    let state = build_test_state(&mock_server.uri(), ApiFormat::OpenAi);
    let app = build_app(state);

    let response = app.oneshot(post_json("/v1/chat/completions", &body)).await.unwrap();
    assert_eq!(response.status(), 200);
}
```

## Dev Dependencies

| Crate | Version | Crate | Purpose |
|-------|---------|-------|---------|
| `tempfile` | 3.x | higgs, higgs-engine, higgs-models | Temporary directories for config/model files |
| `wiremock` | 0.6 | higgs | HTTP mock server for proxy integration tests |
| `http-body-util` | workspace | higgs | Collecting response bodies in tests |
| `tower` | 0.5 | higgs | `ServiceExt::oneshot` for axum router testing |
| `hyper` | 1 | higgs | HTTP types for test requests |

---

*Testing analysis: 2026-03-31*
