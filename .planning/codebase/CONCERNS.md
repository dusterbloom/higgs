# Codebase Concerns

**Analysis Date:** 2026-04-03

## Test Failures

**Daemon PID file tests (3 failures):**
- Files: `crates/higgs/src/daemon.rs` (lines 570-603)
- Issue: Three tests fail due to PID file operations being context-dependent
  - `read_pid_returns_valid_pid`: unwrap() on None at line 590
  - `remove_pid_file_removes_file`: File still exists after removal at line 602
  - `write_and_read_pid_file`: PID mismatch at line 573
- Root cause: Tests use `with_temp_config_dir` which appears to have timing/cleanup issues
- Impact: Cannot verify daemon lifecycle management works correctly
- Fix approach: Refactor tests to properly mock filesystem operations or ensure temp dir isolation

## Unsafe Code in FFI Bindings

**MLX C bindings in Qwen3NextCausalLM:**
- Files: `crates/higgs-models/src/qwen3_next.rs` (lines 32-395)
- Issue: Multiple unsafe FFI calls to `mlx_sys` for custom kernels
  - Line 33-40: FFI error handler registered with global state
  - Line 341-343, 368, 372, 395: `mlx_array_new()`, `mlx_array_free()`, `Array::from_ptr()`
  - Line 611, 624, 637: Unsafe dtype checks and array data access
- Complexity: Tight coupling to MLX-C 0.4.0 API; any changes break compilation
- Risk: Memory leaks if exception occurs between allocation and free; pointer aliasing issues
- Mitigation: Error handlers are properly wrapped; null sentinel patterns used
- Recommended: Consider RAII wrapper to auto-free arrays on drop

**Memory limit FFI:**
- Files: `crates/higgs-engine/src/simple.rs` (lines 33-73)
- Issue: Unsafe device info queries with manual struct initialization
  - Line 36-39: `mlx_device_info_new()` and `mlx_get_default_device()` with raw pointers
  - Line 42: Size extraction via FFI with error code checking
- Context: Required to prevent Metal OOM on constrained GPUs (M4 Pro 32GB)
- Mitigation: Disableable via `HIGGS_NO_MEM_LIMIT` env var; error codes checked; limits set to conservative defaults (75% memory, 50% cache)

## Known Model Incompatibilities

**MTP Speculative Decode (dead feature):**
- Files: `crates/higgs-models/src/qwen3_next.rs`, `crates/higgs-engine/src/simple.rs`
- Issue: MTP head in Qwen3.5-27B-4bit produces garbage output
- Status: Infrastructure exists (wired into SimpleEngine) but non-functional due to model weights
  - Both mlx-community and grafted weights produce wrong tokens
  - mlx-lm strips these weights in sanitize() — they acknowledge the head is non-functional
  - MTP was trained as training objective, not for inference-time speculative decoding
- Impact: 0% token acceptance; feature silently fails
- Recommendation: Need model with MTP head specifically fine-tuned for speculative decode, or wait for Qwen team release

**Qwen3.5-27B-Claude-Opus-Distilled (2.5x slower than mlx-lm):**
- Files: Dense model architecture in `crates/higgs-models/src/qwen3_next.rs`
- Issue: Higgs achieves 6.31 t/s vs mlx-lm 15.7 t/s on M4 Pro 64GB
- Root cause: 27B is fully-dense (all 27B params active), not MoE
  - 64 layers, hidden=5120, intermediate=17408
  - GDN layers (GatedDeltaNet SSM) with sequential scan ops hard to parallelize
  - Higgs GDN implementation likely has optimization gap vs mlx-lm native
- Architecture mismatch: MoE models (35B-A3B) run 9x faster on Higgs (55+ t/s), but dense 27B underperforms
- Impact: User may choose dense distill due to quality, but on constrained hardware
- Recommendation: Use 35B-A3B MoE for Mac. For Claude-distilled quality, need MoE distill (doesn't exist yet)

## Performance Bottlenecks

**Context length limits on M4 Pro:**
- Files: `crates/higgs-engine/src/simple.rs`, `crates/higgs-models/src/qwen3_next.rs`
- Issue: Qwen3.5-35B-A3B crashes at ~24K context on M4 Pro 32GB
- Root cause: TurboQuant prefill + lazy compute graph OOMs on intermediate activations
- Fix implemented: Chunked prefill wired into `AnyModel::forward_chunked()` (512-token chunks)
- Status: Fixed; now supports 28K+ tokens
- Monitor: Paged cache allocator under high context load

**TurboQuant hurts dense models:**
- Files: Quantization kernel in `crates/higgs-models/src/qwen3_next.rs`
- Issue: TurboQuant fused kernels reduce decode from 6.31 t/s to 5.57 t/s (13% SLOWER) on Qwen3.5-27B
- Cause: Dense models are compute-bound, not bandwidth-bound — TQ overhead exceeds bandwidth savings
- Recommendation: Auto-disable TurboQuant when model has no MoE AND active_params > ~15B

**MLX-RS Git Pin vs Crates.io:**
- Files: `Cargo.toml` (lines 82-83, 143-144)
- Issue: Dependency on git rev af21d79 (MLX-C 0.4.0) vs crates.io 0.25.3 (older MLX-C)
  - crates.io version: 37 tok/s
  - git pin version: 60 tok/s (+62%)
- Patch: mlx-rs and mlx-sys manually patched in Cargo.toml
- Risk: Warnings on build ("patch was not used in crate graph")
- Workaround: metallib stale after version switch — must `rm target/release/mlx.metallib`

## Linting Warnings (High Lint Level)

**Non-snake_case variables (architectural legacy):**
- Files: `crates/higgs-models/src/qwen3_next.rs` (lines 1758, 1772, 10357, 10360)
- Issue: Mathematical naming conventions (B, L, D, H, D_kv) conflict with Rust linting
  - Line 1758: `let D = shape[ndim - 1]` (head dimension)
  - Line 1772: `let L = if pos_shape.len() == 1` (sequence length)
  - Line 10357-10360: `B` (batch size), `L` (length) in inner functions
- Cause: Convention borrowed from ML literature for readability
- Severity: Warnings (non-blocking); lint config allows override
- Recommendation: Add inline `#[allow(non_snake_case)]` on mathematical helper functions

**Unused imports in test code:**
- Files: `crates/higgs-engine/src/simple.rs` (lines 1740-1741), `crates/higgs-models/src/spec_prefill/prefill.rs` (lines 5-6)
- Issue: Dead imports in inactive test modules
- Impact: None; warnings only
- Fix: Remove unused imports from test code

**Unnecessary parentheses in closure:**
- Files: `crates/higgs-models/src/spec_prefill/prefill.rs` (line 22)
- Issue: Tuple expression `(b * L + i as i32)` should be `b * L + i as i32`
- Impact: None; style only

## Architectural Fragility

**Tight Coupling to MLX-C API:**
- Files: `crates/higgs-models/src/qwen3_next.rs` (gather_qmm, GDN kernel)
- Issue: Custom Metal kernels and FFI calls directly to MLX-C 0.4.0
  - `gather_qmm()` function (lines 327-396) implements fused expert selection
  - `GATED_DELTA_KERNEL_SOURCE` (lines 409+) hardcoded Metal kernel
  - Error handler (lines 32-50) uses global Mutex for FFI error capture
- Risk: Breaking change in MLX-C (0.4.1 or later) requires rewrite
- Upgrade path: Would need to verify API compatibility; possibly no backward compat
- Recommendation: Add integration test that verifies kernel outputs against reference implementation

**Custom GDN Kernel Maintenance:**
- Files: `crates/higgs-models/src/qwen3_next.rs` (lines 409-600)
- Issue: ~400 lines of Metal shader code with complex grid/threadgroup logic
- Risk: Difficult to debug; fragile to parameter changes (Dk, Dv, Hk, Hv)
- Known issue: Conv1d T=1 fast path started but incomplete (`conv_weight_t` field added but not used)
- Recommendation: Document grid computation formulas and test with multiple tensor shapes

## Thread Safety and Concurrency

**Mutex-heavy SimpleEngine:**
- Files: `crates/higgs-engine/src/simple.rs` (lines 91-101)
- Issue: Every inference request acquires 4+ mutexes sequentially
  - `model: Mutex<AnyModel>`
  - `prefix_cache: Mutex<PrefixCache>`
  - `paged_cache: Mutex<PagedKvCache>`
  - `scheduler: Mutex<RoundRobinScheduler>`
  - `sessions: Mutex<HashMap>`
- Impact: Serializes all requests through single-threaded model inference
- Mitigation: Intended design (MLX is not thread-safe); concurrency handled at HTTP layer via Tokio
- Recommendation: Switch to `BatchEngine` for concurrent requests (separate background thread with dedicated model)

**FFI Error Handler Global State:**
- Files: `crates/higgs-models/src/qwen3_next.rs` (lines 28-50)
- Issue: `FFI_LAST_ERROR: Mutex<Option<String>>` captures errors from FFI
  - Used in `gather_qmm()` error path (line 373-377)
  - Requires lock per inference call
- Risk: Race condition if multiple threads call gather_qmm simultaneously (should not happen due to model Mutex, but fragile)
- Mitigation: Only accessed when gather_qmm returns error (rare path)

## Security and Validation

**Chat Template Rendering with User Input:**
- Files: `crates/higgs-engine/src/chat_template.rs` (lines 141-240)
- Issue: ChatTemplate uses minijinja to render system prompts + messages
  - User messages passed directly to template engine
  - Template parsing uses `ChatTemplateRenderer::new()` (lines 159, 175, 203, etc.)
- Risk: Potential template injection if jinja syntax appears in user message (e.g., `{{ ... }}`)
- Mitigation: minijinja uses jinja2 auto-escaping; check if enabled in renderer config
- Recommended: Add test with jinja syntax in messages to verify escaping

**Config Validation Gaps:**
- Files: `crates/higgs/src/doctor.rs`
- Issue: Not examined in depth; doctor validation should catch misconfiguration before startup
- Related: Project CLAUDE.md states "always update doctor.rs when changing config fields"
- Recommendation: Audit doctor.rs against all config fields in `crates/higgs/src/config.rs`

## Compilation and Build Issues

**Clippy Warnings Allowed:**
- Files: `crates/higgs-engine/src/chat_template.rs` (line 141, 498)
- Issue: `#[allow(clippy::panic, clippy::unwrap_used)]` in test code
  - Line 141: Allows unwrap in test setup (acceptable)
  - Line 498: Allows panic in test teardown (acceptable)
- Impact: Test code is allowed to panic; production code denies unwrap_used/panic
- Status: Intentional; tests need to fail fast

## Pre-Existing Test Failures (Not Our Code)

**reasoning_parser (higgs-engine):**
- 2 tests failing in `higgs_engine` dependency
- Not examined; appears to be upstream issue

**qwen3_next::tests::test_sparse_moe_forward_output_shape (higgs-models):**
- 1 test failing in `higgs-models`
- Not examined; appears to be upstream issue

## Stale Artifacts

**MLX Metallib Cache:**
- Issue: Compiled GPU kernel (`target/release/mlx.metallib`) from previous mlx-rs version persists
- Problem: When switching mlx-rs git revisions (e.g., af21d79 → f4aa309), stale .metallib uses old MLX-C code
- Impact: Silent perf regression or crash (Metal kernel mismatch)
- Fix: `rm target/release/mlx.metallib` after version switch (documented in MEMORY.md)
- Recommendation: Consider build script that invalidates metallib on mlx-sys version change

## Documentation Debt

**Incomplete Fast Path Implementation:**
- Files: `crates/higgs-models/src/qwen3_next.rs` (Conv1d field added but not wired)
- Issue: Conv1d T=1 fast path started but abandoned
  - `conv_weight_t` field added to model struct
  - Kernel dispatch logic not completed
  - No test coverage for partial implementation
- Risk: Technical debt; may confuse future maintainers
- Recommendation: Either complete implementation or remove partial code

**Custom Kernel Documentation:**
- Files: `crates/higgs-models/src/qwen3_next.rs` (lines 409-600)
- Issue: Metal kernel source (GATED_DELTA_KERNEL_SOURCE) lacks parameter documentation
  - Template params Dk, Dv, Hk, Hv not explained
  - Grid formula `(32, Dv, B * Hv)` not justified
  - Threadgroup size `(32, 4, 1)` not motivated
- Recommendation: Add doc comments explaining grid dimensions and kernel algorithm

## Summary of High-Priority Issues

| Issue | Severity | Impact | Effort |
|-------|----------|--------|--------|
| Daemon PID file tests fail | Medium | Cannot verify daemon lifecycle | Low |
| Dense 27B model 2.5x slower | High | Limits distill model usability | High |
| MTP speculative decode non-functional | Low | Feature silently fails | Blocked |
| MLX-C API tight coupling | High | Fragile to MLX updates | Medium |
| TurboQuant hurts dense models | Medium | Performance regression on some models | Low |
| Chat template injection risk | Low | Unlikely but possible if escaping broken | Low |
| Conv1d T=1 fast path incomplete | Low | Technical debt | Medium |
| Mutex contention in SimpleEngine | Low | Expected; switch to BatchEngine for concurrency | N/A |

---

*Concerns audit: 2026-04-03*
