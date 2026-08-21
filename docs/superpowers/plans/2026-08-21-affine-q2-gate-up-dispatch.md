# Affine Q2 Gate/Up Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing opt-in affine-Q2 SIMD gate/up experiment observable, regression-tested for Escha's group-64 layout, and measurable only when the exact stateless radix cache is warm.

**Architecture:** Keep the production Q2 decision unchanged: only an explicit `HIGGS_BONSAI_Q2_SIMD=1` request can use the existing Metal QMV kernel. Extract its shape boundary into a pure helper, use a disabled-by-default one-shot trace for each selected route, and validate both the Metal kernel and dense SwiGLU equivalence at Escha Q2/G64. Extend the existing HTTP decode benchmark to parse and require exact radix-prefix reuse; it must never use the lossy `session_id` continuation path.

**Tech Stack:** Rust, MLX/Metal through `mlx-rs`, `tracing`, Clap, Reqwest/SSE, Higgs radix prefix cache.

## Global Constraints

- Escha W2 affine-Q2 is `bits=2`, `group_size=64`; no Bonsai fixture or dispatch gate may assume group size 128.
- Preserve the present unset production behavior and the explicit `HIGGS_BONSAI_Q2_SIMD=1` opt-in contract.
- Do not make `HIGGS_BONSAI_Q2_SIMD` a process-global default and do not change `HIGGS_DENSE_FFN_GATE_UP` policy.
- The automatic fast route remains decode-only future work; this plan must not alter any prefill routing.
- Diagnostics are disabled unless `HIGGS_TRACE_Q2_DISPATCH=1`; when disabled, there is no logging and no additional allocation.
- A trace emits at most once per process for stock-MLX Q2 and once per process for SIMD Q2, with `route`, `row_count`, `n_rows`, `k_dim`, and `group_size` fields.
- Full-model experiments use stateless requests with no `session_id`; normal radix-prefix reuse is exact while retained-session continuation can be lossy.
- The user is conserving disk space: do not create another worktree or a new `target/` directory. Do not run a Cargo build, test, server, or GPU benchmark from this worktree.

---

## File Structure

- `crates/higgs-models/src/qwen3_next.rs`: owns affine Q2 dispatch, dense SwiGLU test fixtures, and the tests that can directly invoke the existing Metal QMV kernel.
- `crates/higgs-bench/src/bin/bench_decode.rs`: owns stateless streaming benchmark request construction and parsing of terminal SSE usage.
- `docs/superpowers/specs/2026-08-21-affine-q2-gate-up-dispatch-design.md`: existing experiment matrix and promotion gate; do not modify it in this plan.

### Task 1: Make the affine-Q2 dispatch boundary testable and observable

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next.rs:469-520`
- Test: `crates/higgs-models/src/qwen3_next.rs` test module near `dense_hidden_fused_matches_separate_path`

**Interfaces:**
- Consumes: `quantized_forward`, `bonsai_q2_qmv_simd`, `HIGGS_BONSAI_Q2_SIMD`.
- Produces: `fn affine_q2_simd_eligible(row_count: i32, weight_shape: &[i32]) -> bool` and a one-shot trace inside `affine_q2_simd_forward`.

- [ ] **Step 1: Write the failing pure dispatch-boundary test**

Add this test before defining `affine_q2_simd_eligible`:

```rust
#[test]
fn affine_q2_simd_eligibility_requires_single_row_and_exact_packed_shape() {
    assert!(affine_q2_simd_eligible(1, &[17_408, 320]));
    assert!(!affine_q2_simd_eligible(2, &[17_408, 320]));
    assert!(!affine_q2_simd_eligible(1, &[34_816, 320]));
    assert!(!affine_q2_simd_eligible(1, &[17_408, 319]));
    assert!(!affine_q2_simd_eligible(1, &[17_408]));
}
```

- [ ] **Step 2: Verify the test fails for the intended reason**

Run only after the implementation is copied into the existing active build workspace, never from this clean worktree:

```bash
cargo test -p higgs-models --lib affine_q2_simd_eligibility_requires_single_row_and_exact_packed_shape -- --exact
```

Expected: compilation fails because `affine_q2_simd_eligible` does not exist.

- [ ] **Step 3: Extract the smallest pure predicate and preserve the opt-in**

Implement exactly this shape predicate near `affine_q2_simd_forward`:

```rust
fn affine_q2_simd_eligible(row_count: i32, weight_shape: &[i32]) -> bool {
    row_count == 1 && matches!(weight_shape, [17_408, 320])
}
```

Set `use_simd` only when both the existing environment request equals `"1"` and this helper returns true. Do not inspect `group_size` in this predicate: the existing Metal kernel accepts its runtime group size, and Escha's actual Q2 conversion uses G64.

Add `OnceLock<bool>` for `HIGGS_TRACE_Q2_DISPATCH` and two `AtomicBool` values, one for each selected route. When tracing is requested, use `swap(true, Ordering::Relaxed)` to emit at most one `tracing::info!` line for stock MLX and one for SIMD. The trace must include `route`, `row_count`, `n_rows`, `k_dim`, and `group_size`; derive `n_rows` and `k_dim` safely from `weight.shape()` without panicking on malformed shapes. Do not change errors or fallback behavior.

- [ ] **Step 4: Verify the focused test passes and format only the changed file**

Run from the existing active build workspace after applying the reviewed patch:

```bash
cargo test -p higgs-models --lib affine_q2_simd_eligibility_requires_single_row_and_exact_packed_shape -- --exact
cargo fmt --check
```

Expected: the focused test passes; formatting makes no changes after `cargo fmt --all` if required.

- [ ] **Step 5: Commit the focused change without unrelated worktree files**

```bash
git add crates/higgs-models/src/qwen3_next.rs
git commit -m "test(escha): cover affine Q2 SIMD dispatch"
```

### Task 2: Add Escha Q2/G64 numerical and dense-SwiGLU coverage

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next.rs:27289-27326`
- Test: `crates/higgs-models/src/qwen3_next.rs` test module

**Interfaces:**
- Consumes: `ops::quantize`, `ops::quantized_matmul`, `crate::metal_kernel::bonsai_q2_qmv_simd`, `FfnBlock::dense_hidden_fused`, and `FfnBlock::dense_hidden_separate`.
- Produces: an evaluated affine-Q2/G64 Metal-vs-MLX parity test and Q2/G64 coverage in the existing fused/separate test.

- [ ] **Step 1: Write the Q2/G64 Metal-vs-MLX parity test**

Add this test before changing the existing Q4-only dense fixture:

```rust
#[test]
fn affine_q2_qmv_simd_matches_mlx_stock_for_escha_g64() {
    use mlx_rs::Dtype;

    let (n, k, group_size) = (128, 256, 64);
    let x = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 1, k], None)
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let dense = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[n, k], None).unwrap();
    let (weight, scales, biases) = ops::quantize(&dense, group_size, 2).unwrap();
    let stock = ops::quantized_matmul(&x, &weight, &scales, &biases, true, group_size, 2).unwrap();
    let simd = crate::metal_kernel::bonsai_q2_qmv_simd(&x, &weight, &scales, &biases, group_size).unwrap();
    mlx_rs::transforms::eval([&stock, &simd]).unwrap();

    assert_eq!(stock.shape(), simd.shape());
    let stock = stock.as_dtype(Dtype::Float32).unwrap();
    let simd = simd.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::transforms::eval([&stock, &simd]).unwrap();
    for (index, (&expected, &actual)) in stock.as_slice::<f32>().iter().zip(simd.as_slice::<f32>()).enumerate() {
        assert!((expected - actual).abs() < 0.5, "Q2/G64 mismatch at {index}: stock={expected}, simd={actual}");
    }
}
```

- [ ] **Step 2: Verify the new kernel test initially runs against the existing route**

Run from the existing active build workspace:

```bash
cargo test -p higgs-models --lib affine_q2_qmv_simd_matches_mlx_stock_for_escha_g64 -- --exact --test-threads=1
```

Expected: this is a real Metal/MLX numeric comparison, not a mock or a source-text assertion. It may pass before Task 1 because it validates the already present kernel; preserve that result in the task report.

- [ ] **Step 3: Generalize the existing dense fixture to cover Q4/G32 and Q2/G64**

Change its assignment helper to receive quantization parameters:

```rust
fn assign_qlinear(layer: &mut QLinear, out_dim: i32, in_dim: i32, group_size: i32, bits: i32) {
    let raw = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[out_dim, in_dim], None)
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let (w, s, b) = ops::quantize(&raw, group_size, bits).unwrap();
    layer.weight = Param::new(w);
    layer.scales = Param::new(s);
    layer.biases = Param::new(b);
    layer.group_size = group_size;
    layer.bits = bits;
}
```

Run the existing fused/separate construction and evaluated max-difference assertion once for `(32, 4, "q4_g32")` and once for `(64, 2, "q2_g64")`. Keep the existing `max_diff < 1e-3` tolerance, and include the label in its failure message. Do not set process environment variables in these tests because Rust tests run in parallel.

- [ ] **Step 4: Verify all focused model tests pass**

Run from the existing active build workspace:

```bash
cargo test -p higgs-models --lib affine_q2_simd_eligibility_requires_single_row_and_exact_packed_shape -- --exact --test-threads=1
cargo test -p higgs-models --lib affine_q2_qmv_simd_matches_mlx_stock_for_escha_g64 -- --exact --test-threads=1
cargo test -p higgs-models --lib dense_hidden_fused_matches_separate_path -- --exact --test-threads=1
```

Expected: all three pass. The Q2/G64 test must force evaluation before comparing values.

- [ ] **Step 5: Commit the coverage as a single focused change**

```bash
git add crates/higgs-models/src/qwen3_next.rs
git commit -m "test(escha): verify Q2 G64 dense dispatch"
```

### Task 3: Refuse to benchmark a cold or lossy cache path

**Files:**
- Modify: `crates/higgs-bench/src/bin/bench_decode.rs:34-99, 92-126, 280-470`
- Test: `crates/higgs-bench/src/bin/bench_decode.rs` new `#[cfg(test)]` module

**Interfaces:**
- Consumes: streaming terminal `usage.prompt_tokens_details.cached_tokens` from Higgs's stateless chat endpoint.
- Produces: `--require-prefix-cache`, `TrialResult.cached_prompt_tokens`, and a pure usage parser test.

- [ ] **Step 1: Write failing parser tests for cached prompt tokens**

Add a unit test module with the following cases before extracting the parser:

```rust
#[test]
fn usage_counters_capture_radix_cached_tokens() {
    let usage = serde_json::json!({
        "prompt_tokens": 1580,
        "completion_tokens": 128,
        "prompt_tokens_details": { "cached_tokens": 1572 }
    });
    let mut counters = UsageCounters::default();
    update_usage_counters(&usage, &mut counters);
    assert_eq!(counters.prompt_tokens, Some(1580));
    assert_eq!(counters.completion_tokens, Some(128));
    assert_eq!(counters.cached_prompt_tokens, Some(1572));
}

#[test]
fn usage_counters_leave_cache_unknown_when_detail_is_absent() {
    let usage = serde_json::json!({ "prompt_tokens": 1580, "completion_tokens": 128 });
    let mut counters = UsageCounters::default();
    update_usage_counters(&usage, &mut counters);
    assert_eq!(counters.cached_prompt_tokens, None);
}
```

- [ ] **Step 2: Verify the tests fail because the parser does not exist**

Run from the existing active build workspace:

```bash
cargo test -p higgs-bench --bin bench_decode usage_counters_capture_radix_cached_tokens -- --exact
cargo test -p higgs-bench --bin bench_decode usage_counters_leave_cache_unknown_when_detail_is_absent -- --exact
```

Expected: compilation fails because `UsageCounters` and `update_usage_counters` do not exist.

- [ ] **Step 3: Add cache-aware benchmark accounting and fail closed**

Add this private accounting type and helper near `handle_sse_line`:

```rust
#[derive(Default)]
struct UsageCounters {
    completion_tokens: Option<u32>,
    prompt_tokens: Option<u32>,
    cached_prompt_tokens: Option<u32>,
}

fn update_usage_counters(usage: &serde_json::Value, counters: &mut UsageCounters) {
    counters.completion_tokens = usage.get("completion_tokens").and_then(serde_json::Value::as_u64).map_or(counters.completion_tokens, |v| Some(u32::try_from(v).unwrap_or(u32::MAX)));
    counters.prompt_tokens = usage.get("prompt_tokens").and_then(serde_json::Value::as_u64).map_or(counters.prompt_tokens, |v| Some(u32::try_from(v).unwrap_or(u32::MAX)));
    counters.cached_prompt_tokens = usage.get("prompt_tokens_details").and_then(|d| d.get("cached_tokens")).and_then(serde_json::Value::as_u64).map_or(counters.cached_prompt_tokens, |v| Some(u32::try_from(v).unwrap_or(u32::MAX)));
}
```

Route terminal `usage` through this helper. Add `cached_prompt_tokens: Option<u32>` to `TrialResult`, add a default-false `--require-prefix-cache` Clap flag and include it in `Params`. After every measured trial, return an error when the flag is enabled and `cached_prompt_tokens.unwrap_or(0) == 0`. Do not add `session_id`, `session_cache_policy`, or `cache_mode: "bypass"` to the request body. Preserve the existing `stream_options.include_usage` request.

- [ ] **Step 4: Verify parser tests and the existing benchmark compile path**

Run from the existing active build workspace:

```bash
cargo test -p higgs-bench --bin bench_decode usage_counters_capture_radix_cached_tokens -- --exact
cargo test -p higgs-bench --bin bench_decode usage_counters_leave_cache_unknown_when_detail_is_absent -- --exact
cargo check -p higgs-bench --bin bench_decode
cargo fmt --check
```

Expected: both parser tests pass; the binary type-checks without a running server.

- [ ] **Step 5: Commit the benchmark safeguard**

```bash
git add crates/higgs-bench/src/bin/bench_decode.rs
git commit -m "feat(bench): require exact radix cache hits"
```

## End-to-End Execution Gate

After the reviewed patch is transplanted into the active Escha workspace, run the P/A/B/C/D matrix in the design document from five fresh server processes using `HIGGS_ESCHA_AFFINE_BITS=2`, greedy decoding, a prompt long enough to produce cached tokens, and `bench_decode --require-prefix-cache --warmup 1 --trials 1 --max-tokens 512`. For P-versus-D, run three ABBA pairs, discarding 32 warmup tokens in each server process before the 512-token measured decode. Record the six individual samples, median, worst sample, model-load time, prompt tokens/s, decode tokens/s, `cached_prompt_tokens`, and the two one-shot Q2 dispatch trace lines.

Do not use `session_id` for the parity or speed gate. This measurement patch can compare exact streamed bytes from each greedy run, but the current HTTP API does not expose generated token IDs; never claim token-ID parity from a re-tokenized response. Before any runtime-policy promotion, add a direct-engine token-ID collector and use it to compare the 128-token greedy continuations across cells. A candidate may be promoted only if that collector agrees for every run, D's median is at least 3% faster than actual production P, and D's worst valid sample is no more than 1% slower than P's worst valid sample.
