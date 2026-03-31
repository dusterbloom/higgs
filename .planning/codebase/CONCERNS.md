# Codebase Concerns

**Analysis Date:** 2026-03-31

## Tech Debt

### Giant `qwen3_next.rs` — 13,256 lines in a single file

- **Issue:** `crates/higgs-models/src/qwen3_next.rs` is 13,256 lines, making up ~22.5% of the entire crate's source. It contains the full Qwen3-Coder-Next hybrid SSM/attention model implementation including FFI bindings, Metal kernel management, forward passes, chunked prefill, MTP speculative decode, and all associated helper structs.
- **Files:** `crates/higgs-models/src/qwen3_next.rs`
- **Impact:** Extremely difficult to navigate, review, or modify. Any change risks unintended side effects in unrelated functionality. The file contains 65 `unsafe` blocks (FFI bindings to mlx-sys), `CachedMetalKernel` with manual `Send`/`Sync` impls, and multiple `#[allow(unsafe_code)]` annotations scattered across 10+ locations.
- **Fix approach:** Split into sub-modules (e.g., `qwen3_next/attention.rs`, `qwen3_next/ssm.rs`, `qwen3_next/mtp.rs`, `qwen3_next/ffi.rs`, `qwen3_next/forward.rs`). The FFI and unsafe code should be isolated into the smallest possible module with clear safety invariants documented.

### Pervasive `#[allow(clippy::too_many_lines)]` — 38 instances

- **Issue:** There are 38 `#[allow(clippy::too_many_lines)]` suppressions across the codebase, indicating many functions that exceed reasonable length thresholds. This is a symptom of functions doing too much.
- **Files:** Concentrated in `crates/higgs/src/translate.rs` (4), `crates/higgs/src/routes/chat.rs` (3), `crates/higgs-engine/src/simple.rs` (5+), `crates/higgs/src/routes/anthropic.rs` (2), `crates/higgs-models/src/turboquant.rs`, `crates/higgs/src/daemon.rs`, `crates/higgs/src/routes/completions.rs`, `crates/higgs/src/routes/embeddings.rs`, `crates/higgs/src/tui/views/models.rs`, `crates/higgs-models/src/gemma2.rs`, `crates/higgs-models/src/transformer.rs`.
- **Impact:** Functions with too many lines are harder to test, harder to reason about, and more prone to bugs. The route handlers in particular mix request parsing, model routing, proxying, format translation, and metrics recording in single functions.
- **Fix approach:** Extract helper functions from route handlers. For example, `chat_completions()` in `crates/higgs/src/routes/chat.rs` (lines 33–248) should be split into separate local-handler and remote-handler paths. The translate module's functions should extract JSON construction helpers.

### `#[allow(clippy::too_many_arguments)]` — 20+ instances

- **Issue:** Functions with excessive parameter counts, especially in `Engine::generate()`, `Engine::generate_streaming()`, and model forward methods.
- **Files:** `crates/higgs/src/state.rs` (lines 134, 172), `crates/higgs-engine/src/simple.rs` (5+ locations), `crates/higgs-engine/src/batch_engine.rs` (2 locations), `crates/higgs-models/src/qwen3_next.rs` (3 locations).
- **Impact:** Prone to argument reordering bugs. Makes adding new parameters error-prone.
- **Fix approach:** Introduce parameter structs (e.g., `GenerateRequest`, `StreamRequest`) that bundle related arguments.

### `ServerConfig` type alias is misleading legacy

- **Issue:** `crates/higgs/src/config.rs` line 771: `pub type ServerConfig = ServerSection;` exists solely for backward compatibility with code that references `state.config.max_tokens`. The comment says "Backward-compatible alias for existing route handler code."
- **Files:** `crates/higgs/src/config.rs` (line 771)
- **Impact:** Creates confusion — two names for the same type. New code may use either, reducing consistency.
- **Fix approach:** Migrate all usages to `ServerSection` and remove the alias.

### Duplicate `ModelConfig` construction in config.rs

- **Issue:** The `ModelConfig` struct is constructed identically in both `build_simple_config()` (lines 453–468) and `load_config_file()` (lines 536–552). This is a copy-paste pattern.
- **Files:** `crates/higgs/src/config.rs` (lines 453–468, 536–552)
- **Impact:** Adding a new field to `ModelConfig` requires updating both sites. Already happened once with `kv_adaptive_dense_layers` and `kv_seed`.
- **Fix approach:** Extract a `fn build_model_config(path: &str, args: &ServeArgs) -> ModelConfig` helper.

## Security Considerations

### `CorsLayer::permissive()` allows any origin

- **Risk:** The API server uses `CorsLayer::permissive()` which allows requests from any origin. Combined with the default `0.0.0.0` bind address, this means any website can make requests to the local server if the user has it running.
- **Files:** `crates/higgs/src/lib.rs` (line 90)
- **Current mitigation:** The server is intended for local development (an inference server). The API key authentication (`--api-key`) provides auth when enabled. The bind address defaults to `0.0.0.0` which listens on all interfaces.
- **Recommendations:** 
  - Default `CorsLayer` to allow only `localhost` origins instead of permissive.
  - Consider defaulting the bind address to `127.0.0.1` instead of `0.0.0.0` for the `serve` command (keep `0.0.0.0` as an explicit opt-in).
  - Document the security implications of `--host 0.0.0.0` in the config file generated by `higgs init`.

### API keys stored in plaintext config files

- **Risk:** Provider API keys (e.g., Anthropic, OpenAI) are stored as plaintext strings in `config.toml`. The `HIGGS_*` environment variables also pass keys as plaintext.
- **Files:** `crates/higgs/src/config.rs` (ProviderConfig.api_key, line 279)
- **Current mitigation:** Config files are in `~/.config/higgs/` with default filesystem permissions. `.gitignore` should exclude config files.
- **Recommendations:** Document that config files should not be committed. Consider supporting keychain integration or encrypted config for production use.

### No request body size limit enforcement on proxy paths

- **Risk:** While the server config has `max_body_size` (default 10MB), the proxy forwarding code in `crates/higgs/src/proxy.rs` does not explicitly enforce body size limits before forwarding to upstream providers. The `Content-Length` header is set from the original body but there's no explicit rejection of oversized proxy requests.
- **Files:** `crates/higgs/src/proxy.rs` (functions `send_and_read`, `send_to_provider`, `proxy_request`)
- **Recommendations:** Add explicit body size validation in the proxy layer, rejecting requests exceeding `max_body_size`.

### Auto-router runs local model inference on every unmatched request when `force = true`

- **Risk:** When `auto_router.force = true`, every request triggers a local model inference call for classification, even if the model is not suitable. This could be abused to consume GPU resources.
- **Files:** `crates/higgs/src/router.rs` (line 188), `crates/higgs/src/auto_router.rs` (function `classify_local`)
- **Current mitigation:** Rate limiting (`--rate-limit`) applies before auto-routing. The auto-router has a configurable timeout (default 2000ms).
- **Recommendations:** Ensure rate limiting is applied before auto-routing in all code paths (currently it is). Document the GPU cost implications of `force = true`.

## Performance Bottlenecks

### Auto-router spawns blocking task for every classification

- **Problem:** `Router::try_auto_route()` in `crates/higgs/src/router.rs` (line 240) spawns a `tokio::task::spawn_blocking` for every auto-route classification. This moves the work to the blocking thread pool but creates task overhead.
- **Files:** `crates/higgs/src/router.rs` (lines 240–256)
- **Cause:** Necessary because MLX inference is CPU/GPU-bound and cannot run on the async runtime.
- **Improvement path:** This is architecturally correct. The main concern is that when `force = true`, every request pays this cost. Consider caching recent classification results for identical message hashes.

### Metrics store uses unbounded `Vec` with periodic eviction

- **Problem:** `MetricsStore` in `crates/higgs/src/metrics.rs` stores all records in a `Vec<RequestRecord>` with periodic eviction (every 60 seconds). Between eviction cycles, the vector can grow unboundedly under high request volume.
- **Files:** `crates/higgs/src/metrics.rs` (lines 55–61, 167–181)
- **Cause:** `records` Vec only gets evicted when `evict_expired()` is called. High request rates between eviction cycles cause memory growth.
- **Improvement path:** Add a maximum record count cap. Evict on insertion when the cap is reached, or use a ring buffer for the time window.

### `translate.rs` allocates extensively per-request

- **Problem:** The format translation functions (`openai_to_anthropic_request`, `anthropic_to_openai_request`, etc.) create many intermediate `serde_json::Value` objects, `String` allocations, and `Vec` allocations per request. The streaming translators create new `String` and `serde_json::json!()` allocations per SSE event.
- **Files:** `crates/higgs/src/translate.rs` (1341 lines total)
- **Cause:** JSON manipulation with `serde_json::Value` is inherently allocation-heavy. This is acceptable for correctness but suboptimal for high-throughput proxying.
- **Improvement path:** For high-throughput proxying scenarios, consider pass-through when source and target formats match (same provider format). Only translate when formats differ.

### Proxy does not support connection pooling reuse hints

- **Problem:** The `reqwest::Client` is created once (`crates/higgs/src/main.rs` line 180) and shared, which is good. However, there is no connection keep-alive configuration or idle timeout tuning for the HTTP client.
- **Files:** `crates/higgs/src/main.rs` (line 180)
- **Improvement path:** Configure `reqwest::ClientBuilder` with explicit pool limits and keep-alive settings.

## Fragile Areas

### Format translation layer — semantic drift risk

- **Files:** `crates/higgs/src/translate.rs` (1341 lines), `crates/higgs/src/anthropic_adapter.rs` (238 lines)
- **Why fragile:** The translation layer manually maps between OpenAI and Anthropic API formats using `serde_json::Value` manipulation. Both APIs evolve independently — new fields, event types, or format changes in either API will silently pass through or be incorrectly translated.
- **Safe modification:** Always test both translation directions. Add tests for any new fields. The existing test coverage for translation is good (50+ tests) but only covers current field sets.
- **Test coverage:** Good unit test coverage for request/response translation. No integration tests that verify end-to-end format fidelity against real provider APIs.

### Engine enum dispatch — every new model requires updates in multiple locations

- **Files:** `crates/higgs-models/src/lib.rs` (AnyModel enum, ~514 lines), `crates/higgs/src/state.rs` (Engine enum, ~235 lines)
- **Why fragile:** Adding a new model architecture requires:
  1. Adding a variant to `AnyModel` in `lib.rs`
  2. Adding match arms in `forward()`, `forward_hidden()`, `forward_chunked()`, `forward_batched()`, `make_cache_with_config()`, `hidden_size()`, `is_vlm()`, `image_size()`, `forward_multimodal()`, and all MTP methods
  3. Adding a variant to `Engine` in `state.rs` with match arms in every method
- **Safe modification:** Use the existing model implementations as templates. Ensure all dispatch methods are updated together.
- **Test coverage:** Cache type mismatch tests exist (`any_model_qwen3_moe_forward_with_hybrid_cache_errors`). No exhaustive dispatch tests that verify all model variants handle all operations.

### SSE streaming state machines — easy to desynchronize

- **Files:** `crates/higgs/src/routes/chat.rs` (lines 499–694), `crates/higgs/src/routes/anthropic.rs` (lines 366–491), `crates/higgs/src/translate.rs` (SseReader at lines 865–936, stream translators at lines 532–858)
- **Why fragile:** The streaming implementations maintain state (reasoning tracker, tool call index, block indices) across async stream iterations. The `SseReader` is a hand-rolled SSE parser that handles `\n\n` and `\r\n\r\n` separators but may not handle edge cases like partial event data across chunks.
- **Safe modification:** Test with streaming providers that produce varied event sequences. The `[DONE]` sentinel handling differs between OpenAI and Anthropic formats.
- **Test coverage:** Unit tests exist for the translate functions but no integration tests for streaming end-to-end. The `SseReader` has no tests.

### `SseReader` hand-rolled parser has no tests

- **Files:** `crates/higgs/src/translate.rs` (lines 865–936)
- **Why fragile:** The SSE parser handles event separation, field extraction (`event:`, `data:`), and multi-line data joining. Edge cases like empty events, events without data, or events with only `event:` type but no data are handled by returning `None`, but this behavior is untested.
- **Test coverage:** Zero tests for `SseReader`.

## Scaling Limits

### Single-threaded generation per model

- **Current capacity:** Each `SimpleEngine` uses a `Mutex<...>` around model state (`crates/higgs-engine/src/simple.rs`). Only one generation can run at a time per model.
- **Limit:** Concurrent requests to the same model serialize behind the mutex. The `BatchEngine` exists (`crates/higgs-engine/src/batch_engine.rs`) for interleaved batching but requires explicit `batch = true` in config.
- **Scaling path:** Batch engine already exists. Document its use for high-concurrency scenarios. Consider auto-enabling batching when concurrent request count exceeds a threshold.

### Metrics stored in-memory with no persistence guarantees

- **Current capacity:** Metrics are stored in memory with configurable retention (default 60 minutes). Written to JSONL log files for persistence.
- **Limit:** The `MetricsStore` `snapshot()` method clones all records in the time window on every call. For the TUI dashboard, this happens periodically. Under high request volume with long retention windows, this causes heap pressure.
- **Scaling path:** The `clone()` in `snapshot()` is the main bottleneck. Consider returning an `Arc<[RequestRecord]>` or using a read-optimized data structure.

## Dependencies at Risk

### `mlx-rs` pinned to specific git revision

- **Risk:** Both `mlx-rs` and `mlx-sys` are pinned to `rev = "af21d79"` from `https://github.com/oxideai/mlx-rs.git`. This is a third-party fork/wrapper around Apple's MLX framework. If the repository is deleted, moved, or the revision is garbage-collected, the project cannot build.
- **Impact:** Complete build failure. This is the core ML dependency for all model inference.
- **Migration plan:** Pin to a tagged release if/when available. Consider vendoring critical FFI bindings. Monitor the upstream repository health.

### `tokenizers` with `http` feature

- **Risk:** `crates/higgs/Cargo.toml` line 98: `tokenizers = { version = "0.22", features = ["http"] }`. The `http` feature enables downloading tokenizers from URLs at runtime.
- **Impact:** Minor — increases attack surface for supply chain. Only used for HuggingFace tokenizer downloads.
- **Migration plan:** Remove the `http` feature if tokenizer downloading is handled externally (via `huggingface-cli`).

## Missing Critical Features

### No streaming tool calls support

- **Problem:** Streaming responses in `crates/higgs/src/routes/chat.rs` explicitly ignore tools (line 412: "Tool-calling responses are not supported in streaming mode"). The `Engine::Stub` variant in `state.rs` has no tokenizer, preventing TUI testing of tool call flows.
- **Blocks:** Clients that require streaming tool calls (e.g., Claude Code in streaming mode) cannot use tool-calling features with local models via streaming.

### No image URL fetching for VLMs

- **Problem:** `extract_images()` in `crates/higgs/src/routes/chat.rs` (line 744) only supports base64 data URIs. HTTP/HTTPS image URLs are silently ignored with a comment "could be fetched in the future."
- **Blocks:** Multi-modal requests from clients that provide image URLs (common in Anthropic API usage) fail silently.

### No request cancellation support

- **Problem:** When a client disconnects mid-generation, the `spawn_blocking` task continues running until completion. There is no cancellation token or abort mechanism.
- **Files:** `crates/higgs/src/routes/chat.rs` (lines 298–312, 482–497), `crates/higgs/src/routes/anthropic.rs` (lines 243–257, 332–347)
- **Blocks:** Wasted GPU resources on cancelled requests.

### `higgs start` daemon lacks systemd/launchd integration

- **Problem:** The daemon management (`start`/`stop`/`attach`) uses PID files and manual signal handling. No integration with macOS `launchd` or Linux `systemd` for proper service management.
- **Files:** `crates/higgs/src/daemon.rs`
- **Blocks:** The daemon won't auto-restart on crashes, won't start at boot, and has no proper service lifecycle management.

## Test Coverage Gaps

### `SseReader` — zero tests

- **What's not tested:** The SSE event parser in `crates/higgs/src/translate.rs` (lines 865–936). No tests for event separation, multi-line data joining, event type extraction, or edge cases.
- **Files:** `crates/higgs/src/translate.rs` (struct `SseReader`)
- **Risk:** Silent data corruption in streaming proxy responses if the parser mishandles edge cases.
- **Priority:** High — affects all streaming proxy functionality.

### Streaming route handlers — no integration tests

- **What's not tested:** End-to-end streaming for both `chat_completions` and `create_message` routes. The streaming code paths in `chat.rs` and `anthropic.rs` are only tested via unit tests on helper functions, not via HTTP request/response cycles.
- **Files:** `crates/higgs/src/routes/chat.rs` (function `chat_completions_stream`), `crates/higgs/src/routes/anthropic.rs` (function `create_message_stream`)
- **Risk:** SSE framing errors, missing `[DONE]` sentinels, incorrect event ordering, or reasoning tracker desynchronization.
- **Priority:** High — streaming is a primary use case.

### `proxy.rs` — no tests for `send_and_read` or `send_to_provider`

- **What's not tested:** The actual HTTP proxy functions (`send_and_read`, `send_to_provider`, `proxy_request`) have no unit or integration tests. Only the helper functions (`build_forwarding_headers`, `rewrite_model_in_body`, etc.) are tested.
- **Files:** `crates/higgs/src/proxy.rs` (lines 111–232)
- **Risk:** Header forwarding bugs, incorrect status code handling, or response header filtering issues. The `wiremock` dev-dependency is available but unused for proxy testing.
- **Priority:** Medium — proxy is well-isolated and the helper functions are tested, but the core forwarding logic is untested.

### Auto-router end-to-end — no integration tests

- **What's not tested:** The full auto-router flow (request → classification → route selection → generation) is not tested end-to-end. Only unit tests for `parse_route_name` and `build_prompt` exist.
- **Files:** `crates/higgs/src/auto_router.rs`
- **Risk:** Auto-routing failures in production (wrong route selection, timeout handling, engine errors) would only be caught by manual testing.
- **Priority:** Medium — auto-router is an opt-in feature.

### `Engine::Batch` variant — no tests beyond type dispatch

- **What's not tested:** The `BatchEngine` code path is used when `batch = true` but has no integration tests that verify batched decode produces correct outputs.
- **Files:** `crates/higgs-engine/src/batch_engine.rs`
- **Risk:** Batch scheduling bugs, incorrect cache management, or output interleaving issues.
- **Priority:** Medium — batch mode is explicitly opted-in.

### No negative/boundary tests for configuration validation

- **What's not tested:** Config validation covers common cases but doesn't test boundary values like extremely large timeout values, zero port, very long model names, or malformed regex patterns that compile but match nothing.
- **Files:** `crates/higgs/src/config.rs` (function `validate_config`)
- **Risk:** Subtle misconfigurations that pass validation but cause runtime issues.
- **Priority:** Low — existing validation is comprehensive for common cases.

---

*Concerns audit: 2026-03-31*
