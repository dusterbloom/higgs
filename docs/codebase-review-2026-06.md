# Codebase Review — June 2026

A full review of the workspace (`higgs`, `higgs-engine`, `higgs-models`, `higgs-bench`)
covering security, performance, correctness, cleanup, and an assessment of Apple's
Core AI framework. File/line references are against the tree at the time of review;
items marked **[fixed]** were addressed on the branch that introduced this document.

## Overall assessment

The codebase is in good shape. The strict lint policy in `Cargo.toml` is genuinely
enforced — production code is free of `unwrap`/`panic` (all occurrences are
`#[cfg(test)]`-guarded), custom Metal kernels are cached correctly in `OnceLock`
statics, README and docs match actual behavior, and CI covers fmt, clippy
(pedantic + nursery), tests, MSRV, and a 70% coverage floor.

The significant findings cluster in three areas:

1. An O(n²) streaming decode path in the engine.
2. Insecure-by-default network posture and world-readable secrets.
3. Duplication: two prefix-cache implementations, duplicated prefill logic, and
   near-duplicate route handlers.

## Apple Core AI assessment

[Core AI](https://developer.apple.com/documentation/coreai/) is Apple's WWDC 2026
on-device inference framework — the runtime behind Apple Intelligence, now public.
It targets CPU, GPU, and the **Apple Neural Engine (ANE)**, with ahead-of-time model
compilation for instant load, stateful (KV-cache-style) execution, zero-copy data
paths, and PyTorch conversion tooling.

**It is Swift-only today.** There is no documented C/C++ API surface, and therefore
no path to call it from Rust without a hand-rolled Swift shim. Recommendation:

- **Keep MLX (mlx-rs) as the backbone.** It is the right tool for a Rust inference
  server on Apple Silicon, and higgs already exploits it well (custom Metal kernels
  for GatedDeltaNet and TurboQuant, `mlx_rs::fast::*` SDPA/RoPE paths).
- **Future opportunity — ANE offload for auxiliary models.** A small Swift
  shim/XPC sidecar could run compact models on the ANE while the GPU stays
  dedicated to the main model. Best candidates: speculative-decode draft models
  (`higgs-engine/src/mtp.rs`), embedding models, and the SigLIP vision encoder
  (`higgs-models/src/siglip.rs`). This is R&D, not a near-term refactor.
- **Cold start.** Core AI's AOT-compiled model assets could eventually reduce model
  load times if a bridge materializes.
- **Watch for** a C API or community Rust bindings before committing to anything.

## P0 — Security

1. **Insecure network defaults.** The server defaulted to `0.0.0.0`
   (`crates/higgs/src/config.rs`, `default_host()`), with no API key required and
   `CorsLayer::permissive()` unconditionally applied (`crates/higgs/src/lib.rs`).
   Any host on the LAN could use the server. **[fixed]** Default bind is now
   `127.0.0.1`, CORS is configurable via `server.cors_origins` (disabled unless
   set; `"*"` opts into permissive), and `higgs doctor` warns when binding a
   non-loopback address without an API key.
2. **World-readable secrets.** Config files containing provider API keys were
   written with default permissions (0o644) by `higgs init`
   (`crates/higgs/src/daemon.rs`) and `higgs config set`
   (`crates/higgs/src/cli_config.rs`). **[fixed]** Config files are now written
   0o600 on Unix, and doctor warns when an existing config containing an
   `api_key` is group/world-readable.
3. **Upstream error bodies leaked into errors/metrics.**
   `crates/higgs/src/proxy.rs` embedded the entire upstream error body in
   `ProxyError`, which flows into logs and the metrics store; upstream bodies can
   echo request headers/keys. **[fixed]** Bodies are truncated to a bounded length.
4. **Chat template execution surface.** `higgs-engine/src/chat_template.rs`
   registers all of `minijinja_contrib` plus the pycompat method callback for
   templates loaded from model directories (`chat_template.jinja` /
   `tokenizer_config.json`). A malicious model repo therefore gets a fairly rich
   template language. minijinja has no filesystem/process access so this is
   bounded, but unbounded loops are possible. **[fixed]** Template execution is
   now fuel-limited via `Environment::set_fuel`. Follow-up (not done): trim the
   contrib filter set to what HF templates actually use.
5. **Missing resource bounds** (recommendation, not implemented):
   - No configurable prompt-length limit before tokenization/prefill.
   - Constrained decoding compiles schema-derived structures without a size cap
     (`higgs-engine/src/constrained.rs`); a pathological JSON schema is a DoS vector.
   - The engine `sessions` map (`higgs-engine/src/simple.rs`) has no cap or TTL.
   Suggested: `max_prompt_tokens` and `max_schema_bytes` config fields plus a
   session cap, all validated by doctor.

## P1 — Performance

6. **O(n²) streaming decode.** Each generated token triggered
   `decode_tokens(tokens)` over the *entire* completion so far, then sliced off
   the new suffix; with stop sequences enabled, `check_stop_sequences` also
   rescanned the full text every step (`higgs-engine/src/simple.rs`,
   streaming loop). Cost grows quadratically with completion length — measurable
   on long generations. The batch engine had the identical pattern in
   `batch_engine.rs`. **[fixed]** Both engines now share an incremental
   detokenizer that decodes a bounded trailing window of tokens per step
   (UTF-8 sequences split across tokens are held back until complete, which
   also fixes replacement-char corruption in streamed output), and stop
   sequences are scanned over only the new tail plus the maximum
   stop-sequence overlap. The non-streaming path still re-decodes per token
   when stop sequences are configured — same recipe applies if it shows up
   in profiles.
7. **Prefix cache clones — corrected, no change needed.** An initial pass
   flagged `PrefixCache::find_longest_prefix` for cloning the entire cached
   `AnyCache` per lookup (`higgs-engine/src/prompt_cache.rs`). Inspection of
   mlx-rs shows `Array::clone` is a refcounted handle copy (`mlx_array_set`),
   so an `AnyCache` clone is O(layers) handle bumps, not a tensor copy — and
   consumers genuinely need an owned copy because the forward pass mutates it.
   The remaining open question (macOS-only to verify): a cached handle keeps
   the underlying buffers alive, which can prevent MLX buffer donation and
   force copy-on-write during subsequent decode steps. Worth profiling before
   any restructuring.
8. **O(n) LRU eviction.** Eviction walks the whole radix tree to find the oldest
   entry (`prompt_cache.rs::evict_lru`; similar in `paged_prefix_cache.rs`).
   Acceptable at current cache sizes now that per-lookup clones are gone; revisit
   with an access-ordered index if `max_entries` grows.
9. **TurboQuant dtype conversions — corrected, no change needed.** The
   unconditional `as_dtype(Dtype::Float32)` calls in
   `higgs-models/src/turboquant.rs` looked like redundant materializations,
   but MLX core's `astype` short-circuits to a same-handle return when the
   dtype already matches (`mlx/ops.cpp: if (dtype == a.dtype()) return a;`),
   so these are no-ops beyond an FFI call.
10. Lower priority (not implemented): per-request `messages.clone()` in route
    handlers (`crates/higgs/src/routes/chat.rs`); `Mutex` rather than `RwLock`
    for read-heavy engine state (`simple.rs`); metrics store uses two `RwLock`s
    per request (fine at local-server request rates).

## P2 — Correctness

11. **Two prefix-cache implementations.** `prompt_cache.rs` (~670 LOC,
    clone-based radix tree) and `paged_prefix_cache.rs` (~1,070 LOC, block-paged)
    coexist; the simple engine uses the paged one while the batch engine uses the
    old one. Converge on the paged cache and delete the old implementation.
12. **Client disconnect cleanup.** The batch engine only notices a disconnected
    client when `blocking_send` fails (`batch_engine.rs`), leaving in-flight
    request state to be cleaned up late; on the HTTP side, a stream that ends
    early can leave the metrics record without final token counts
    (`routes/chat.rs` streaming finalization). Add cancellation-aware cleanup and
    mark such records as cancelled.
13. **Error swallowing.** `cache/paged.rs::remove_session` logs only the first
    block-free error; weight loading warns (rather than errors) on unmatched
    keys (`higgs-models/src/lib.rs`), which is lenient enough to hide a corrupt
    or mismatched checkpoint. Consider a strict-loading flag.
14. **PID-file TOCTTOU.** `read_pid → pid_is_alive → kill` in `daemon.rs` is
    non-atomic. Low impact (stale-PID cleanup exists); noted for completeness.
15. **Unimplemented session APIs.** `step()`/`generate_session()` in `simple.rs`
    return "not implemented" — either finish batched session generation or remove
    the scaffolding so the API surface matches reality.

## P3 — Cleanup / maintainability

- **Duplicated prefill/decode-graph logic** between `simple.rs` and
  `batch_engine.rs` (~100 LOC overlap each for prefill and decode-graph
  construction). Extract shared helpers.
- **`generate_inner()` is ~1,100 lines** (`simple.rs`) covering standard decode,
  MTP, prompt-lookup, and thinking-budget paths. Split per strategy.
- **Route handler duplication.** `routes/chat.rs` and `routes/anthropic.rs`
  share the parse → route → stream/non-stream → metrics skeleton; several
  handlers run 300-950 lines. A shared streaming-response helper would remove
  most of it.
- **`qwen3_next.rs` is ~15k lines** — inline Metal kernel codegen, GDN layers,
  MoE, and tests in one file. Split into submodules.
- **Attention/mask logic duplicated** across `gemma2.rs`, `phi3.rs`,
  `starcoder2.rs`, `deepseek_v2.rs` despite shared helpers existing in
  `utils.rs` (which `transformer.rs` already uses for Llama/Qwen2/Mistral).
- **Hardcoded `THINKING_BUDGET = 256`** (`simple.rs`) — should be a config field
  (with doctor validation and README/init-template updates per project rules).
- **mlx-rs git pin** blocks crates.io publishing (release workflow skips publish
  when git-pinned). Track upstream releases.
- **Ignored test due to global MLX state** (`higgs-models/src/yarn.rs`) — known
  Metal/RNG state contamination across tests in one process; documented, but
  worth revisiting if more tests start interfering.

## Test gaps

- Streaming SSE end-to-end over HTTP (chunk framing, `[DONE]`).
- Client-disconnect/cancellation paths (engine cleanup, metrics finalization).
- Cache exhaustion: prefix-cache eviction under pressure, paged-cache block
  exhaustion, multi-session contention.
- Adversarial inputs: hostile chat templates, pathological constrained-decoding
  schemas, very long prompts.
- Session lifecycle (create → generate → remove) and leak detection.
- Concurrent-request stress (rate limiter, lock contention).
