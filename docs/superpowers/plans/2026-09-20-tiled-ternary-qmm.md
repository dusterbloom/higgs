# Tiled Ternary QMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a default-off Metal SIMD-group matrix kernel for Prism ternary Q2 prefill, then integrate only projections that beat stock MLX and pass quality and memory gates.

**Architecture:** A new `BM=8`, `BN=32`, `BK=128` Metal kernel consumes the existing canonical packed weights and uses four 8 by 8 SIMD-group matrix accumulators with FP32 accumulation. A strict Rust wrapper owns shape and dtype validation. Model dispatch remains stock until isolated real-shape measurements pass; guarded integration then covers both `QLinear::forward` and the dense fused gate/up bypass.

**Tech Stack:** Rust, `mlx-rs`/`mlx-sys` 0.31 compatibility branch, embedded Metal Shading Language, Apple `simdgroup_matrix`, Xcode 16.4, existing Rust CPU oracle and Python end-to-end benchmark harness.

## Global Constraints

- Work directly in `/Users/peppi/Dev/higgs` on branch `nightly`; do not create or use worktrees.
- Do not modify `/Users/peppi/Dev/higgs/.omen/search.db` or `target-build-xcode164/`.
- Run GitNexus `impact` before editing each named symbol. If risk remains `UNKNOWN` because the large source file is skipped or the DB version mismatches, record that result and confirm exact call sites with `rg` before editing.
- Run GitNexus `detect-changes --scope all` before every implementation commit. A partial or truncated result is not clean.
- Keep `HIGGS_TERNARY_QMM` default off. Do not change existing M=1 QMV or M<=8 verifier behavior.
- Do not create a second persistent packed layout, dense weight cache, checkpoint conversion, or model-wide materialization.
- Workers must not run GPU tests, stop the server, or start another model process. The coordinator alone owns GPU execution.
- Before GPU work, stop the single port 9000 server. Restore the committed binary and port 9000 server after measurements.
- Use at most two kernel tile variants. Stop when the operator, quality, memory, or end-to-end gate fails.
- Gate/up and down projections are promoted independently.

---

### Task 1: Freeze the Candidate Contract and Add Red Oracle Coverage

**Owner:** Luna B

**Files:**
- Modify: `crates/higgs-models/src/bonsai_q2.rs:746-1095`

**Interfaces:**
- Consumes: existing `PackedQ2Linear`, `upload_to_mlx`, `dense_matvec_reference`, and stock `quantized_matmul` test helpers.
- Produces: calls to `crate::metal_kernel::bonsai_q2_qmm_tiled_ternary(&Array, &Array, &Array, i32) -> Result<Array, Exception>` and ignored benchmark `q2_qmm_tiled_ternary_go_no_go`.

- [ ] **Step 1: Record graph and text impact before editing**

Run:

```bash
node .gitnexus/run.cjs impact bonsai_q2_qmm_ternary --direction upstream --repo higgs
rg -n "bonsai_q2_qmm_ternary|q2_qmm_ternary_go_no_go|dense_matvec_reference" crates/higgs-models/src
```

Expected: the graph may report `UNKNOWN`; text search shows the scalar wrapper is used only by tests/benchmarks.

- [ ] **Step 2: Add a zero-safe comparison helper**

Add a helper in the existing test module that evaluates both arrays as FP32 and returns maximum absolute error and normalized RMS error:

```rust
fn qmm_error(actual: &Array, reference: &[f32]) -> (f32, f32) {
    let actual = actual.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    actual.eval().unwrap();
    let got = actual.as_slice::<f32>();
    assert_eq!(got.len(), reference.len());
    let mut max_abs = 0.0_f32;
    let mut error_sq = 0.0_f32;
    let mut reference_sq = 0.0_f32;
    for (&value, &want) in got.iter().zip(reference) {
        let error = value - want;
        max_abs = max_abs.max(error.abs());
        error_sq += error * error;
        reference_sq += want * want;
        assert!(value.is_finite());
    }
    (max_abs, (error_sq / reference_sq.max(1.0e-12)).sqrt())
}
```

- [ ] **Step 3: Add the failing candidate oracle test**

Create `q2_qmm_tiled_ternary_matches_cpu_reference` using ternary codes and `bias=-scale`. Cover these exact `(N, K, M, dtype)` cases:

```rust
[
    (96, 256, 9, Dtype::Float16),
    (130, 4096, 17, Dtype::Bfloat16),
    (5120, 5120, 31, Dtype::Bfloat16),
    (96, 17408, 33, Dtype::Float16),
]
```

For every activation row, append the existing CPU `dense_matvec_reference` result to one flat FP32 reference vector. Call native stock, cast-matched stock, and the candidate before inspecting candidate metrics. Record all three. Assert candidate shape `[M, N]`, original output dtype, finite values, normalized RMS error at most `1.0e-2`, and maximum absolute error at most `0.25`. The fixed-prompt model gate remains authoritative for errors that can change behavior.

- [ ] **Step 4: Add rank and contract cases**

Add a rank-3 input case shaped `[1, 9, K]` and verify `[1, 9, N]`. Add invalid group size, trailing K mismatch, scale shape mismatch, and `M<=8` cases; each must return an error whose message names the failed contract rather than reaching raw pointer arithmetic.

- [ ] **Step 5: Add the ignored operator benchmark**

Add `q2_qmm_tiled_ternary_go_no_go` comparing stock MLX, the existing scalar ternary QMM, and the candidate. Warm each path, force `eval`, alternate measurement order, and print median microseconds and peak memory for:

```text
gate_up: N=34816 K=5120 M=16,32,64,512,1024
down:    N=5120  K=17408 M=16,32,64,512,1024
```

The scalar result is informational. Stock MLX is the pass/fail baseline.

- [ ] **Step 6: Confirm the test is red before kernel implementation**

Run CPU compilation only:

```bash
DEVELOPER_DIR=/Applications/Xcode-16.4.0.app/Contents/Developer cargo test -p higgs-models q2_qmm_tiled_ternary_matches_cpu_reference --no-run
```

Expected: compilation fails because `bonsai_q2_qmm_tiled_ternary` does not exist. Do not run the test binary.

- [ ] **Step 7: Commit the red tests**

Run required graph change analysis, stage only `bonsai_q2.rs`, and commit:

```bash
git add crates/higgs-models/src/bonsai_q2.rs
git commit -m "test(models): specify tiled ternary qmm"
```

---

### Task 2: Implement the First Tiled Metal Kernel

**Owner:** Luna A

**Files:**
- Modify: `crates/higgs-models/src/metal_kernel.rs:3127-3514`

**Interfaces:**
- Consumes: canonical `u32 [N,K/16]` weights, `[N,K/128]` scales, input shaped `[...,K]`, and task-local MLX stream.
- Produces: `pub fn bonsai_q2_qmm_tiled_ternary(x: &Array, weight: &Array, scales: &Array, group_size: i32) -> Result<Array, Exception>`.

- [ ] **Step 1: Record graph and text impact before editing**

Run:

```bash
node .gitnexus/run.cjs impact bonsai_q2_qmm_ternary --direction upstream --repo higgs
node .gitnexus/run.cjs impact bonsai_q2_mma_m8 --direction upstream --repo higgs
rg -n "bonsai_q2_qmm_ternary|bonsai_q2_mma_m8|simdgroup_matrix" crates/higgs-models/src
```

Warn the coordinator if GitNexus reports HIGH or CRITICAL risk. Record `UNKNOWN` and text-confirmed callers if the index cannot resolve the large file.

- [ ] **Step 2: Add the versioned MSL source and cached handle**

Add `Q2_TILED_TERNARY_QMM_V1_SOURCE`, `Q2_TILED_TERNARY_QMM_V1`, and `create_q2_tiled_ternary_qmm_v1_kernel`. Use the versioned host name `higgs_bonsai_q2_tiled_ternary_qmm_v1` and this header:

```rust
let header = CString::new(
    "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n",
)
.unwrap_or_default();
```

- [ ] **Step 3: Stage one quantization group per loop**

Use these compile-time constants and scratch layouts:

```metal
constexpr int BM = 8;
constexpr int BN = 32;
constexpr int BK = 128;
threadgroup bfloat16_t x_sh[BM * BK];
threadgroup bfloat16_t w_sh[BK * BN];
threadgroup float out_sh[BM * BN];
```

The 128 threads cooperatively copy eight activation values each. They decode the 256 packed u32 words for the `32 x 128` weight tile exactly once, writing K-major `w_sh[k * BN + n]`. Invalid M/N loads write zero. All threads execute identical barriers.

- [ ] **Step 4: Accumulate four independent matrix fragments**

Map one SIMD group to each 8-column N fragment. Keep one `simdgroup_matrix<float, 8, 8>` accumulator per SIMD group across every K group. For each group, load sixteen A/B fragments and call:

```metal
simdgroup_multiply_accumulate(acc, a, b, acc);
```

Decode trits as `int(code) - 1`. Convert activation and scaled weights to BF16 explicitly in scratch while preserving FP32 accumulation. Do not claim arithmetic identity with the stock FP32-Hadamard path.

- [ ] **Step 5: Store tails safely**

Store each SIMD-group fragment into its disjoint 8 by 8 region in `out_sh`, execute one uniform barrier, then have all 128 threads copy valid `(m,n)` cells to the output. Do not use an unpredicated matrix store directly against an unpadded user output.

- [ ] **Step 6: Implement strict wrapper validation**

Before FFI application, reject:

```text
group_size != 128
M <= 8
weight rank != 2 or weight dtype != uint32
K <= 0 or K % 128 != 0
input trailing dimension != K
scale shape != [N, K/128]
unsupported input or scale dtype
```

Flatten leading input dimensions to `[M,K]`, allocate `[M,N]`, and reshape the result back to `[...,N]`. Use `Stream::task_local_or_default()` and free every FFI vector/config on both success and immediate failure.

- [ ] **Step 7: Run CPU compilation checks only**

Run:

```bash
DEVELOPER_DIR=/Applications/Xcode-16.4.0.app/Contents/Developer cargo check -p higgs-models --release --lib
DEVELOPER_DIR=/Applications/Xcode-16.4.0.app/Contents/Developer cargo test -p higgs-models q2_qmm_tiled_ternary_matches_cpu_reference --no-run
```

Expected: Rust and test binaries compile. Runtime MSL is not yet considered validated.

- [ ] **Step 8: Commit the kernel implementation**

Run required graph change analysis, stage only `metal_kernel.rs`, and commit:

```bash
git add crates/higgs-models/src/metal_kernel.rs
git commit -m "perf(models): add tiled ternary qmm probe"
```

---

### Task 3: Static Kernel and Fixture Gate

**Owner:** Astra low reviewer

**Files:**
- Review: `crates/higgs-models/src/metal_kernel.rs`
- Review: `crates/higgs-models/src/bonsai_q2.rs`

**Interfaces:**
- Consumes: Task 1 and Task 2 commits.
- Produces: written PASS/HOLD review covering contract, indexing, precision, FFI ownership, tests, and benchmark fairness.

- [ ] **Step 1: Review the exact diff and call graph evidence**

Inspect both commits and the recorded GitNexus/text results. Treat unresolved graph risk as unresolved and identify the exact text-confirmed callers.

- [ ] **Step 2: Review Metal synchronization and bounds**

Check that every thread reaches every barrier, scratch regions do not overlap, packed-word indexing stays within `[N,K/16]`, scale indexing stays within `[N,K/128]`, and only valid outputs leave scratch.

- [ ] **Step 3: Review numerical semantics**

Verify trit mapping, scale placement, FP32 accumulation, activation conversion, scale conversion, output dtype, and accumulation order. Reject any description that calls BF16 staging exact relative to native stock.

- [ ] **Step 4: Review benchmark fairness and memory scope**

Ensure stock and candidate include necessary casts/allocations, compile warmup is outside timing, evaluation is forced, real shapes are present, and no full dense matrix or second persistent pack exists.

- [ ] **Step 5: Issue PASS or HOLD before GPU execution**

PASS authorizes only the isolated JIT/correctness run. HOLD lists concrete required fixes and returns ownership to the relevant Luna.

---

### Task 4: Run Isolated JIT, Correctness, and Operator Gates

**Owner:** Coordinator

**Files:**
- No source edits unless Task 3 returns a required fix.
- Save evidence under `target/bench-results/bench_prefill/<timestamp>__tiled-qmm-operator/`.

**Interfaces:**
- Consumes: Astra-approved kernel and fixtures.
- Produces: runtime MSL result, correctness metrics, operator medians/spread, and peak memory for the promotion decision.

- [ ] **Step 1: Record and stop the serving process**

Run `pgrep -fal higgs`, record the tmux command/config, and stop only the existing Higgs server. Confirm port 9000 is free. Do not start another GPU process.

- [ ] **Step 2: Run the smallest runtime JIT case**

Run only `q2_qmm_tiled_ternary_matches_cpu_reference` with `--nocapture --test-threads=1`. Confirm the Metal source compiles and the M=9 tail case executes without MLX error.

- [ ] **Step 3: Run all candidate oracle cases**

Capture maximum absolute error, normalized RMS error, dtype, shapes, and finite checks. Compare native stock, cast-matched stock, and candidate without changing frozen tolerances after seeing candidate output.

- [ ] **Step 4: Run the ignored operator benchmark**

Run one benchmark process with `--test-threads=1`. Persist raw stdout/stderr and machine/toolchain metadata. Do not run the full model yet.

- [ ] **Step 5: Apply the operator and memory gate**

Pass gate/up only if candidate/stock is at least 1.2x at both M=512 and M=1024. Pass down independently under the same rule. Reject any projection that causes spilling, unsafe peak allocation, or a chunk-size reduction.

- [ ] **Step 6: Decide whether the one allowed tile revision is justified**

If the first tile is correct but below threshold, use counters or timing evidence to choose exactly one change such as weight staging order or one alternate BN. Do not add split K, persistent repacking, and double buffering together.

- [ ] **Step 7: Restore the port 9000 server**

Restart the same committed binary and config in `tmux` and verify `/health`. The candidate is still not routed into the model.

---

### Task 5: Add Default-Off Model Dispatch for Passing Projections

**Owner:** Luna B, only after Task 4 passes at least one projection

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next.rs:1175-1340`
- Modify: `crates/higgs-models/src/qwen3_next.rs:1886-1919`
- Modify: `crates/higgs-models/src/qwen3_next.rs:7804-7940`
- Test: `crates/higgs-models/src/qwen3_next.rs` existing test module

**Interfaces:**
- Consumes: `bonsai_q2_qmm_tiled_ternary` and the per-projection Task 4 decision.
- Produces: `ternary_qmm_enabled()`, `ternary_qmm_eligible(...)`, QLinear dispatch, dense fused gate/up dispatch, and default-off unit tests.

- [ ] **Step 1: Record graph and text impact before editing**

Run impact for `QLinear::forward`, `dense_hidden_fused`, and `dense_fused_ternary_qmv_eligible`, then confirm their callers and the fused bypass with `rg` when graph risk is `UNKNOWN`.

- [ ] **Step 2: Write the failing eligibility test**

Add a pure CPU test asserting that the new path accepts only:

```rust
bits == 2
group_size == 128
has_hadamard == true
enabled == true
row_count > 8
projection_kind passed its Task 4 gate
```

Assert false for M=1, M=8, non-Hadamard Q2, Q1/Q4, group 64, disabled switch, and any projection that failed the operator gate.

- [ ] **Step 3: Add the default-off switch**

Implement:

```rust
fn ternary_qmm_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HIGGS_TERNARY_QMM").is_ok_and(|value| value == "1")
    })
}
```

- [ ] **Step 4: Route eligible standalone QLinear projections**

After the existing M=1 and M<=8 checks, call the tiled wrapper only for an operator-approved projection shape. Leave stock `quantized_forward_with_q2_simd_policy` as the disabled and unsupported path.

- [ ] **Step 5: Route the dense fused gate/up bypass**

In the affine branch of `dense_hidden_fused`, after computing the already-rotated `x` and `row_count`, call the tiled wrapper against `fw` and `fs` only when fused gate/up passed Task 4 and eligibility is true. Preserve split and SwiGLU behavior.

- [ ] **Step 6: Run CPU checks**

Run the eligibility unit test and `cargo check -p higgs-models --release --lib`. Do not run model/GPU tests.

- [ ] **Step 7: Commit guarded integration**

Run required graph change analysis, stage only `qwen3_next.rs`, and commit:

```bash
git add crates/higgs-models/src/qwen3_next.rs
git commit -m "perf(models): gate tiled ternary prefill"
```

---

### Task 6: Validate the Guarded Full Model

**Owner:** Coordinator

**Files:**
- Reuse: `/tmp/higgs_prefill_bench.py`
- Save: `target/bench-results/bench_prefill/<timestamp>__tiled-qmm-e2e/`

**Interfaces:**
- Consumes: guarded integration and exact root-checkout release build.
- Produces: quality, 512/4096 prefill, memory, decode, and server smoke evidence.

- [ ] **Step 1: Build from the root checkout with Xcode 16.4**

Use `CARGO_TARGET_DIR=/Users/peppi/Dev/higgs/target-build-xcode164` and the same explicit Xcode 16.4 compiler/linker environment used by the current installed binary. Record binary and metallib hashes.

- [ ] **Step 2: Stop the current server and run fixed quality prompts**

Run native stock and `HIGGS_TERNARY_QMM=1` against the same prompt/tokenization. Compare logits where available and deterministic greedy output. Review every changed token.

- [ ] **Step 3: Run exact unprofiled prefill A/B**

For stock and candidate, run the exact 512-token and 4096-token prompt artifacts with cache bypass, speculation disabled, `max_tokens=1`, identical chunk size, and three or more alternating samples. Report median, CV, prompt tok/s, TTFT, and peak memory.

- [ ] **Step 4: Apply the end-to-end gate**

Require at least 10 percent lower unprofiled prefill latency at both lengths beyond observed noise, no chunk-size reduction, and passing quality. Keep the switch default off when the gate fails.

- [ ] **Step 5: Verify decode and DFlash did not regress**

Run one short standard decode and one multi-token DFlash smoke. Confirm no MLX error and compare decode throughput with the committed baseline. This is a regression check, not a decode tuning task.

- [ ] **Step 6: Restore the installed root-checkout server**

Install only after all gates pass. Start the single tmux server on port 9000 with the existing 26-model config, API key, and DFlash model. Verify `/health`, `/v1/models`, and one authenticated chat request.

---

### Task 7: Final Graph Review, Astra Gate, and Delivery

**Owner:** Coordinator and Astra low reviewer

**Files:**
- Review every implementation diff and persisted result artifact.

**Interfaces:**
- Consumes: all implementation commits and Task 6 evidence.
- Produces: final PASS/HOLD, one clean implementation commit series, pushed `fork/nightly`, and a concise performance report.

- [ ] **Step 1: Run final GitNexus change analysis**

Run `detect-changes --scope all` with the compatible CLI. Re-run any partial or truncated result. Report all HIGH/CRITICAL risks and unresolved UNKNOWN areas with text-confirmed callers.

- [ ] **Step 2: Have Astra review the complete diff and evidence**

Astra checks spec coverage, numerical behavior, dispatch completeness, lazy-error claims, memory, benchmark fairness, and whether each promoted projection independently passed.

- [ ] **Step 3: Run final verification**

Run formatting, `git diff --check`, release library check, release Higgs build, candidate oracle test, fixed quality prompts, and the final smoke sequence. Preserve raw outputs.

- [ ] **Step 4: Commit any review fixes atomically**

Stage only owned source/test files. Do not stage `.omen/search.db` or the build directory. Use a concise Conventional Commit message describing the actual shipped behavior.

- [ ] **Step 5: Push the reviewed branch**

Push root-checkout `nightly` to `fork/nightly` after the final PASS. Confirm local HEAD equals `fork/nightly`.

- [ ] **Step 6: Report measured outcome**

Report stock and candidate operator speedups, exact 512/4096 prefill medians, memory, quality result, decode/DFlash smoke, installed hashes, server status, commit hashes, and any disabled or rejected projection.
