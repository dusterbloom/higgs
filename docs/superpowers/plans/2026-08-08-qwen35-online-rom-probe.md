# Qwen3.5 Online Residual ROM Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a test-only Qwen3.5 decode probe that measures oracle and causal rank-32/64/128 activation energy at every layer and decides the agreed rank-64, 99% go/no-go gate.

**Architecture:** A `cfg(test)` one-shot hook captures pre-attention, pre-MLP, and post-layer activations without changing any public model API or production build. A separate test module validates trajectories, performs batched Gram-matrix/eigendecomposition analysis with MLX, aggregates energy-weighted domain metrics, and owns the ignored model-backed experiment.

**Tech Stack:** Rust 2024 workspace, `mlx-rs` arrays/linear algebra, `tokenizers`, `serde`/`serde_json`, Cargo unit and ignored tests.

## Global Constraints

- Use `/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit` as the first canonical checkpoint.
- The rank-64 gate requires at least 0.99 causal retained energy for code, prose, and reasoning at both `attention_in` and `mlp_in`.
- Report rank 32, 64, and 128, oracle spectra, per-layer distributions, and worst cases.
- Do not alter or disable Bonsai Q1 row4, Ternary Q2 row2, dSpark/DFlash, MTP, ANE, or Escha native fast-path flags.
- Do not implement reduced weights, approximate matmuls, basis refresh, or runtime speedup claims in this phase.
- The capture hook must compile only under `cfg(test)` and preserve all existing forward return types.
- `forward_raw_hidden_with_taps` has CRITICAL upstream impact: two direct callers, ten indexed processes, and decode/tap/sparse-prefill reachability. Keep its change branch-only and verify logits with capture enabled and disabled.
- Preserve all unrelated dirty-worktree changes; stage only files named by each task.

---

## File Structure

- Modify `crates/higgs-models/src/qwen3_next.rs`: test-only capture types, one-shot request/take storage, three guarded capture calls in the dense layer loop, and registration of the external test module.
- Create `crates/higgs-models/src/qwen3_next_rom_probe.rs`: trajectory validation, MLX Gram analysis, report/gate types, fixed prompts, unit tests, and the ignored Qwen3.5-9B probe.
- Create `/private/tmp/higgs-rom-qwen35-9b-20260808/report.json` at experiment time: canonical machine-readable result; do not commit generated activation data.

---

### Task 1: One-Shot Test-Only Activation Capture

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next.rs:7706-7734`
- Modify: `crates/higgs-models/src/qwen3_next.rs:8299-8566`
- Create: `crates/higgs-models/src/qwen3_next_rom_probe.rs`

**Interfaces:**
- Produces: `RomActivationSite`, `DiagRomActivation`, `DiagRomCapture`, `diag_request_rom_capture()`, `diag_take_rom_capture()`, and private `diag_begin_rom_capture()`.
- Preserves: `DiagLayer`, `diag_request_hidden_capture()`, `diag_take_hidden_capture()`, and every `Qwen3NextCausalLM` forward signature.

- [ ] **Step 1: Register the external test module and write failing one-shot tests**

Add this immediately before the existing inline `mod tests` in `qwen3_next.rs`:

```rust
#[cfg(test)]
#[path = "qwen3_next_rom_probe.rs"]
mod rom_probe_tests;
```

Create `qwen3_next_rom_probe.rs` with:

```rust
use super::*;

#[test]
fn rom_capture_request_is_consumed_once() {
    diag_clear_rom_capture_for_test();
    diag_request_rom_capture();
    assert!(diag_begin_rom_capture());
    assert!(!diag_begin_rom_capture());
    assert!(diag_take_rom_capture().is_none());
}

#[test]
fn rom_capture_slot_does_not_touch_hidden_capture_slot() {
    diag_clear_rom_capture_for_test();
    DIAG_CAPTURED.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
    diag_store_rom_capture_for_test(Vec::new());

    assert_eq!(diag_take_rom_capture(), Some(Vec::new()));
    assert_eq!(diag_take_hidden_capture(), Some(Vec::new()));
}
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cargo test -p higgs-models --lib qwen3_next::rom_probe_tests::rom_capture -- --nocapture
```

Expected: compilation fails because the ROM capture types and functions do not exist.

- [ ] **Step 3: Add the minimal capture state and types**

Near the existing hidden diagnostic storage in `qwen3_next.rs`, add test-only definitions with these exact shapes:

```rust
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RomActivationSite {
    AttentionIn,
    MlpIn,
    PostLayer,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DiagRomActivation {
    pub layer_idx: usize,
    pub site: RomActivationSite,
    pub values: Vec<f32>,
    pub hidden_dim: usize,
}

#[cfg(test)]
pub(crate) type DiagRomCapture = Vec<DiagRomActivation>;

#[cfg(test)]
thread_local! {
    static DIAG_ROM_CAPTURE_REQ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static DIAG_ROM_CAPTURED: std::cell::RefCell<Option<DiagRomCapture>> = const { std::cell::RefCell::new(None) };
}
```

Implement `diag_request_rom_capture()` so it clears any prior ROM result and requests exactly the next forward. Implement `diag_begin_rom_capture()` with `Cell::replace(false)`, `diag_take_rom_capture()` with `Option::take`, plus the two `#[cfg(test)]` state helpers named by the tests. Do not re-export these symbols from `lib.rs`.

- [ ] **Step 4: Run the one-shot tests and verify GREEN**

Run the command from Step 2.

Expected: both tests pass and no model files are loaded.

- [ ] **Step 5: Write a failing array-materialization test**

Append:

```rust
#[test]
fn rom_capture_materializes_owned_f32_values() {
    let x = Array::from_slice(
        &[
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
        ],
        &[1, 1, 4],
    );
    let mut capture = Vec::new();
    capture_rom_activation(&mut capture, 3, RomActivationSite::MlpIn, &x).unwrap();

    assert_eq!(capture.len(), 1);
    assert_eq!(capture[0].layer_idx, 3);
    assert_eq!(capture[0].site, RomActivationSite::MlpIn);
    assert_eq!(capture[0].hidden_dim, 4);
    assert_eq!(capture[0].values, vec![1.0, 2.0, 3.0, 4.0]);
}
```

- [ ] **Step 6: Verify RED, implement materialization, then verify GREEN**

Run the Task 1 test filter. Confirm failure is `capture_rom_activation` missing. Implement the helper in `qwen3_next.rs` by converting to `Dtype::Float32`, evaluating the converted array, checking the final dimension, copying `as_slice::<f32>()`, rejecting non-finite values, and appending one `DiagRomActivation`. Run the filter again and expect PASS.

- [ ] **Step 7: Wire the guarded hook into the critical layer loop**

In `forward_raw_hidden_with_taps`:

```rust
#[cfg(test)]
let do_rom_capture = diag_begin_rom_capture();
#[cfg(test)]
let mut rom_capture = Vec::with_capacity(if do_rom_capture {
    self.model.layers.len() * 3
} else {
    0
});
```

Under `if do_rom_capture`, capture `normed` as `AttentionIn` immediately after `input_layernorm`, capture `normed_post` as `MlpIn` immediately after `post_attention_layernorm`, and capture updated `h` as `PostLayer` after the MLP residual add. Store the completed vector only after all layers finish successfully. Keep each declaration and branch under `#[cfg(test)]` so non-test builds contain no hook.

- [ ] **Step 8: Format, test, and commit Task 1**

Run:

```bash
cargo fmt --all -- --check
cargo test -p higgs-models --lib qwen3_next::rom_probe_tests::rom_capture -- --nocapture
```

Expected: formatting and all Task 1 tests pass.

Before the commit, run:

```bash
node .gitnexus/run.cjs detect-changes --repo higgs
```

Confirm the reported critical reachability is limited to the known Qwen3Next forward family, then stage only the two Task 1 files and commit:

```bash
git add crates/higgs-models/src/qwen3_next.rs crates/higgs-models/src/qwen3_next_rom_probe.rs
git commit -m "test(rom): capture Qwen3.5 activations"
```

---

### Task 2: Strict Trajectory Assembly

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next_rom_probe.rs`

**Interfaces:**
- Consumes: one complete `DiagRomCapture` per decode token.
- Produces: `TrajectorySet::new(layer_count, hidden_dim)`, `TrajectorySet::push(capture)`, and `TrajectorySet::site_tensor(site)` returning row-major `[layers, tokens, hidden]` values.

- [ ] **Step 1: Write failing validation tests**

Add this fixture and the four validation tests:

```rust
fn synthetic_capture(layer_count: usize, hidden_dim: usize, token: usize) -> DiagRomCapture {
    let mut capture = Vec::new();
    for layer_idx in 0..layer_count {
        for site in [
            RomActivationSite::AttentionIn,
            RomActivationSite::MlpIn,
            RomActivationSite::PostLayer,
        ] {
            let site_offset = match site {
                RomActivationSite::AttentionIn => 0.0,
                RomActivationSite::MlpIn => 10.0,
                RomActivationSite::PostLayer => 20.0,
            };
            let value = layer_idx as f32 * 100.0 + site_offset + token as f32;
            capture.push(DiagRomActivation {
                layer_idx,
                site,
                values: vec![value; hidden_dim],
                hidden_dim,
            });
        }
    }
    capture
}

#[test]
fn trajectory_assembly_orders_layers_sites_and_tokens() {
    let mut set = TrajectorySet::new(2, 2);
    set.push(synthetic_capture(2, 2, 0)).unwrap();
    set.push(synthetic_capture(2, 2, 1)).unwrap();

    let tensor = set.site_tensor(RomActivationSite::AttentionIn).unwrap();
    assert_eq!(tensor.shape(), &[2, 2, 2]);
    assert_eq!(
        tensor.as_slice::<f32>(),
        &[0.0, 0.0, 1.0, 1.0, 100.0, 100.0, 101.0, 101.0]
    );
}

#[test]
fn trajectory_assembly_rejects_missing_layer_site() {
    let mut set = TrajectorySet::new(2, 2);
    let mut capture = synthetic_capture(2, 2, 0);
    capture.retain(|entry| {
        !(entry.layer_idx == 1 && entry.site == RomActivationSite::MlpIn)
    });
    assert!(set.push(capture).unwrap_err().contains("missing"));
}

#[test]
fn trajectory_assembly_rejects_duplicate_layer_site() {
    let mut set = TrajectorySet::new(2, 2);
    let mut capture = synthetic_capture(2, 2, 0);
    capture.push(capture[0].clone());
    assert!(set.push(capture).unwrap_err().contains("duplicate"));
}

#[test]
fn trajectory_assembly_rejects_wrong_dimension_and_non_finite_values() {
    let mut wrong_width = synthetic_capture(2, 2, 0);
    wrong_width[0].values.pop();
    assert!(
        TrajectorySet::new(2, 2)
            .push(wrong_width)
            .unwrap_err()
            .contains("dimension")
    );

    let mut non_finite = synthetic_capture(2, 2, 0);
    non_finite[0].values[0] = f32::NAN;
    assert!(
        TrajectorySet::new(2, 2)
            .push(non_finite)
            .unwrap_err()
            .contains("non-finite")
    );
}
```

The ordering assertion must verify that `site_tensor(AttentionIn)` returns all tokens for layer 0 before all tokens for layer 1, matching shape `[2, 2, hidden_dim]`.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cargo test -p higgs-models --lib qwen3_next::rom_probe_tests::trajectory_assembly -- --nocapture
```

Expected: compilation fails because `TrajectorySet` is missing.

- [ ] **Step 3: Implement the minimal strict accumulator**

Use this storage contract:

```rust
#[derive(Debug)]
struct TrajectorySet {
    layer_count: usize,
    hidden_dim: usize,
    tokens: usize,
    values: std::collections::BTreeMap<(RomActivationSite, usize), Vec<f32>>,
}

impl TrajectorySet {
    fn new(layer_count: usize, hidden_dim: usize) -> Self;
    fn push(&mut self, capture: DiagRomCapture) -> Result<(), String>;
    fn site_tensor(&self, site: RomActivationSite) -> Result<Array, String>;
}
```

`push` must require exactly `layer_count * 3` records, reject duplicate keys before mutation, validate every width and value, verify every expected key, then append atomically. `site_tensor` must return shape `[layer_count, tokens, hidden_dim]`.

- [ ] **Step 4: Verify GREEN and commit Task 2**

Run the Task 2 test filter, `cargo fmt --all -- --check`, and the Task 1 filter. Expect all PASS. Run GitNexus change detection, stage only `qwen3_next_rom_probe.rs`, and commit:

```bash
git add crates/higgs-models/src/qwen3_next_rom_probe.rs
git commit -m "test(rom): validate activation trajectories"
```

---

### Task 3: Oracle and Causal Gram Analysis

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next_rom_probe.rs`

**Interfaces:**
- Consumes: an activation tensor `[layers, 256, hidden]`, warmup length 128, and ranks `[32, 64, 128]`.
- Produces: one `LayerSpectrum` per layer with total/captured energies and oracle/causal retention for each rank.

- [ ] **Step 1: Write failing numerical tests**

Add this helper and the four numerical tests:

```rust
fn trajectory_array(values: &[f32], layers: i32, tokens: i32, hidden: i32) -> Array {
    Array::from_slice(values, &[layers, tokens, hidden])
}

#[test]
fn exact_rank_one_trajectory_has_unit_rank_one_retention() {
    let x = trajectory_array(
        &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        1,
        4,
        2,
    );
    let result = analyze_trajectory(&x, 2, &[1]).unwrap();
    assert!((result[0].oracle[0].retained - 1.0).abs() < 1e-6);
    assert!((result[0].causal[0].retained - 1.0).abs() < 1e-6);
}

#[test]
fn causal_basis_does_not_include_held_out_direction() {
    let x = trajectory_array(
        &[1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 2.0],
        1,
        4,
        2,
    );
    let result = analyze_trajectory(&x, 2, &[1]).unwrap();
    assert!(result[0].causal[0].retained.abs() < 1e-6);
    assert!((result[0].oracle[0].retained - 0.5).abs() < 1e-6);
}

#[test]
fn retained_energy_is_monotonic_in_rank() {
    let x = trajectory_array(
        &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        1,
        8,
        4,
    );
    let result = analyze_trajectory(&x, 4, &[1, 2, 4]).unwrap();
    let causal = &result[0].causal;
    assert!(causal[0].retained <= causal[1].retained);
    assert!(causal[1].retained <= causal[2].retained);
    assert!((causal[2].retained - 1.0).abs() < 1e-6);
}

#[test]
fn zero_eigenvalues_do_not_produce_non_finite_metrics() {
    let x = trajectory_array(
        &[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        1,
        4,
        2,
    );
    let result = analyze_trajectory(&x, 2, &[1, 2]).unwrap();
    assert!(result[0].oracle.iter().all(|metric| metric.retained.is_finite()));
    assert!(result[0].causal.iter().all(|metric| metric.retained.is_finite()));
}
```

Use dimensions no larger than `[2 layers, 8 tokens, 4 hidden]` and ranks `[1, 2, 4]` so these tests finish quickly.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cargo test -p higgs-models --lib exact_rank_one_trajectory_has_unit_rank_one_retention -- --nocapture
cargo test -p higgs-models --lib causal_basis_does_not_include_held_out_direction -- --nocapture
cargo test -p higgs-models --lib retained_energy_is_monotonic_in_rank -- --nocapture
cargo test -p higgs-models --lib zero_eigenvalues_do_not_produce_non_finite_metrics -- --nocapture
```

Expected: compilation fails because the analysis types/functions are absent.

- [ ] **Step 3: Implement batched Gram decomposition**

Add serializable metric types:

```rust
#[derive(Debug, Clone, serde::Serialize)]
struct RankEnergy {
    rank: usize,
    retained: f64,
    captured_energy: f64,
    total_energy: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct LayerSpectrum {
    layer: usize,
    oracle: Vec<RankEnergy>,
    causal: Vec<RankEnergy>,
    effective_rank_95: usize,
    effective_rank_99: usize,
    effective_rank_999: usize,
}
```

Implement:

```rust
fn analyze_trajectory(
    all: &Array,
    warmup_tokens: usize,
    ranks: &[usize],
) -> Result<Vec<LayerSpectrum>, String>;
```

Validate shape `[L, T, D]`, `T > warmup_tokens`, nonempty sorted unique ranks, and `rank <= warmup_tokens`. Form batched Gram matrices with `ops::matmul(X, X.transpose_axes(&[0, 2, 1])?)`. Use `mlx_rs::linalg::eigh_device(..., StreamOrDevice::cpu())`; MLX returns eigenvalues in ascending order, so select retained columns from the end.

For causal energy, compute `C = Y X^T`, then `C V_r diag(lambda_r^-1/2)`. Sum its squared entries for captured energy and divide by `sum(Y^2)`. Discard eigenvalues `lambda <= max(lambda) * 1e-7`; do not divide them. For oracle energy, eigendecompose the full-trajectory Gram matrix and divide the sum of the largest `r` non-negative eigenvalues by their total. Clamp reported retention to `[0, 1]` only after recording finite numerator and denominator.

- [ ] **Step 4: Verify GREEN and commit Task 3**

Run all `qwen3_next::rom_probe_tests` non-ignored tests and formatting. Expected: the exact-rank, causal-boundary, monotonicity, and zero-eigen tests pass. Run GitNexus change detection, stage the probe module only, and commit:

```bash
git add crates/higgs-models/src/qwen3_next_rom_probe.rs
git commit -m "test(rom): analyze causal activation rank"
```

---

### Task 4: Energy-Weighted Report and Gate

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next_rom_probe.rs`

**Interfaces:**
- Consumes: prompt/domain/site `LayerSpectrum` results.
- Produces: `RomProbeReport`, compact console summary, pretty JSON, and authoritative `GateResult`.

- [ ] **Step 1: Write failing aggregation tests**

Add this fixture and the four aggregation tests:

```rust
fn required_aggregate_fixture(retained: f64) -> Vec<AggregateReport> {
    let mut cells = Vec::new();
    for domain in [Domain::Code, Domain::Prose, Domain::Reasoning] {
        for site in [RomActivationSite::AttentionIn, RomActivationSite::MlpIn] {
            cells.push(AggregateReport {
                domain,
                site,
                rank: 64,
                captured_energy: retained * 100.0,
                total_energy: 100.0,
                retained,
                median: retained,
                p05: retained,
                worst_retained: retained,
                worst_prompt: "fixture".to_owned(),
                worst_layer: 0,
            });
        }
    }
    cells
}

#[test]
fn aggregation_weights_retention_by_total_energy() {
    let metrics = [
        RankEnergy { rank: 64, retained: 0.99, captured_energy: 99.0, total_energy: 100.0 },
        RankEnergy { rank: 64, retained: 0.001, captured_energy: 1.0, total_energy: 1000.0 },
    ];
    let aggregate = aggregate_rank_energy(&metrics).unwrap();
    assert!((aggregate.retained - (100.0 / 1100.0)).abs() < 1e-12);
}

#[test]
fn gate_requires_every_domain_and_both_linear_input_sites() {
    let complete = required_aggregate_fixture(0.995);
    assert!(evaluate_gate(&complete, &[]).unwrap().passed);

    let mut missing = complete.clone();
    missing.pop();
    assert!(!evaluate_gate(&missing, &[]).unwrap().passed);

    let mut failed = complete;
    failed[0].retained = 0.989;
    assert!(!evaluate_gate(&failed, &[]).unwrap().passed);
}

#[test]
fn post_layer_cannot_rescue_a_failed_linear_input_site() {
    let mut aggregates = required_aggregate_fixture(0.995);
    let mlp = aggregates
        .iter_mut()
        .find(|cell| cell.domain == Domain::Code && cell.site == RomActivationSite::MlpIn)
        .unwrap();
    mlp.retained = 0.98;
    aggregates.push(AggregateReport {
        domain: Domain::Code,
        site: RomActivationSite::PostLayer,
        rank: 64,
        captured_energy: 100.0,
        total_energy: 100.0,
        retained: 1.0,
        median: 1.0,
        p05: 1.0,
        worst_retained: 1.0,
        worst_prompt: "fixture".to_owned(),
        worst_layer: 0,
    });
    assert!(!evaluate_gate(&aggregates, &[]).unwrap().passed);
}

#[test]
fn worst_layer_and_p05_are_reported_from_layer_metrics() {
    let summary = summarize_retentions(&[
        (4, "p4", 0.9),
        (1, "p1", 0.5),
        (3, "p3", 0.8),
        (2, "p2", 0.7),
        (5, "p5", 1.0),
    ])
    .unwrap();
    assert!((summary.median - 0.8).abs() < 1e-12);
    assert!((summary.p05 - 0.5).abs() < 1e-12);
    assert_eq!(summary.worst_layer, 1);
    assert_eq!(summary.worst_prompt, "p1");
}
```

- [ ] **Step 2: Run and verify RED**

Run the four exact tests. Expected: compilation fails because aggregation/report types are absent.

- [ ] **Step 3: Implement serializable report types and the exact gate**

Define:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Domain { Code, Prose, Reasoning }

#[derive(Debug, Clone, serde::Serialize)]
struct AggregateReport {
    domain: Domain,
    site: RomActivationSite,
    rank: usize,
    captured_energy: f64,
    total_energy: f64,
    retained: f64,
    median: f64,
    p05: f64,
    worst_retained: f64,
    worst_prompt: String,
    worst_layer: usize,
}

#[derive(Debug, Clone)]
struct DistributionSummary {
    median: f64,
    p05: f64,
    worst_retained: f64,
    worst_prompt: String,
    worst_layer: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct SiteReport {
    site: RomActivationSite,
    layers: Vec<LayerSpectrum>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct PromptReport {
    label: String,
    domain: Domain,
    eos_positions: Vec<usize>,
    sites: Vec<SiteReport>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct LayerFailure {
    domain: Domain,
    site: RomActivationSite,
    prompt: String,
    layer: usize,
    retained: f64,
}

#[derive(Debug, serde::Serialize)]
struct GateCell {
    domain: Domain,
    site: RomActivationSite,
    rank: usize,
    retained: f64,
    passed: bool,
}

#[derive(Debug, serde::Serialize)]
struct GateResult {
    threshold: f64,
    passed: bool,
    cells: Vec<GateCell>,
    layers_below_095: Vec<LayerFailure>,
}

#[derive(Debug, serde::Serialize)]
struct RomProbeReport {
    schema_version: u32,
    canonical_run: bool,
    model_path: String,
    git_commit: String,
    num_hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    warmup_tokens: usize,
    measured_tokens: usize,
    ranks: Vec<usize>,
    prompts: Vec<PromptReport>,
    aggregates: Vec<AggregateReport>,
    gate: GateResult,
}
```

Aggregate with `sum(captured_energy) / sum(total_energy)`. The gate examines exactly rank 64 and the six `(domain, AttentionIn|MlpIn)` cells. Use threshold `0.99`. Record but do not fail solely on layers below `0.95`. Add `write_report(path, &report)` using `serde_json::to_writer_pretty` and a summary printer with one row per domain/site/rank.

- [ ] **Step 4: Verify GREEN and commit Task 4**

Run all non-ignored ROM tests and formatting. Run GitNexus change detection, stage only the probe module, and commit:

```bash
git add crates/higgs-models/src/qwen3_next_rom_probe.rs
git commit -m "test(rom): report the rank-64 gate"
```

---

### Task 5: Deterministic Qwen3.5-9B Model Probe

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next_rom_probe.rs`

**Interfaces:**
- Consumes: `HIGGS_MODEL_PATH`, `HIGGS_ROM_OUTPUT_DIR`, optional smoke-test overrides, and the checkpoint tokenizer.
- Produces: `/private/tmp/higgs-rom-qwen35-9b-20260808/report.json` and console summary.

- [ ] **Step 1: Write failing workload/config tests**

Add these tests for `ProbeConfig::canonical()` and `prompt_suite()`:

```rust
#[test]
fn probe_config_canonical_shape_is_fixed() {
    let config = ProbeConfig::canonical();
    assert_eq!(config.warmup_tokens, 128);
    assert_eq!(config.measured_tokens, 128);
    assert_eq!(config.ranks, vec![32, 64, 128]);
    assert_eq!(config.prompt_limit, 9);
    assert!(config.is_canonical());
}

#[test]
fn probe_config_accepts_smoke_shape_and_rejects_rank_above_warmup() {
    let smoke = ProbeConfig::from_values(16, 8, vec![4, 8, 16], 1).unwrap();
    assert_eq!(smoke.warmup_tokens, 16);
    assert!(!smoke.is_canonical());
    assert!(ProbeConfig::from_values(16, 8, vec![4, 17], 1).is_err());
}

#[test]
fn prompt_suite_has_three_unique_prompts_per_domain() {
    let prompts = prompt_suite();
    assert_eq!(prompts.len(), 9);
    let labels = prompts
        .iter()
        .map(|prompt| prompt.label)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(labels.len(), 9);
    for domain in [Domain::Code, Domain::Prose, Domain::Reasoning] {
        assert_eq!(prompts.iter().filter(|prompt| prompt.domain == domain).count(), 3);
    }
}
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cargo test -p higgs-models --lib qwen3_next::rom_probe_tests::probe_config -- --nocapture
```

Expected: compilation fails because `ProbeConfig` and `prompt_suite` are absent.

- [ ] **Step 3: Implement configuration and the fixed prompt suite**

Implement this configuration type with `canonical`, `from_values`,
`from_env`, and `is_canonical` methods:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProbeConfig {
    warmup_tokens: usize,
    measured_tokens: usize,
    ranks: Vec<usize>,
    prompt_limit: usize,
}
```

`from_env` starts from canonical defaults and parses only these overrides:

```text
HIGGS_ROM_WARMUP_TOKENS
HIGGS_ROM_MEASURE_TOKENS
HIGGS_ROM_RANKS          comma-separated positive integers
HIGGS_ROM_PROMPT_LIMIT   1..=9
```

Define `PromptSpec { label: &'static str, domain: Domain, text: &'static str }`
and return these exact nine entries from `prompt_suite()`:

```rust
[
    PromptSpec {
        label: "code_rust_parser",
        domain: Domain::Code,
        text: "Write a complete Rust implementation of an incremental JSON parser that accepts arbitrarily split byte chunks, preserves UTF-8 boundaries, reports byte offsets for syntax errors, and emits SAX-style events. Explain the state machine only after the code. Include enums, structs, error types, unit tests, and at least two property-oriented invariants. Continue for at least 400 tokens and do not abbreviate any implementation.",
    },
    PromptSpec {
        label: "code_python_scheduler",
        domain: Domain::Code,
        text: "Implement a deterministic cooperative task scheduler in Python using generators, a monotonic logical clock, cancellation tokens, deadlines, and a priority queue. Include the full implementation, type hints, tests for cancellation races and equal-deadline ordering, and a short complexity analysis. Continue for at least 400 tokens without placeholders or omitted helper functions.",
    },
    PromptSpec {
        label: "code_sql_btree",
        domain: Domain::Code,
        text: "Design and implement the core of a small B+ tree storage engine in clear pseudocode and SQL-oriented data structures. Cover page layout, search, insertion, splitting, deletion, sibling redistribution, root contraction, write-ahead log records, and crash recovery. Include executable-style procedures and invariants. Continue for at least 400 tokens and spell out every important branch.",
    },
    PromptSpec {
        label: "prose_storage_engines",
        domain: Domain::Prose,
        text: "Write a cohesive technical essay comparing B-tree, LSM-tree, and append-only log storage engines. Explain write amplification, read amplification, compaction, caching, recovery, range scans, and workload-dependent tradeoffs through one running example. Continue for at least 400 tokens in connected prose rather than bullet fragments.",
    },
    PromptSpec {
        label: "prose_renaissance_trade",
        domain: Domain::Prose,
        text: "Write a historically grounded essay about how Mediterranean trade networks shaped Renaissance cities, institutions, accounting practices, shipbuilding, and artistic patronage. Trace causes and consequences across several regions while distinguishing evidence from inference. Continue for at least 400 tokens in polished narrative prose.",
    },
    PromptSpec {
        label: "prose_ecosystem_recovery",
        domain: Domain::Prose,
        text: "Explain how a damaged coastal wetland can recover over several decades after tidal flow is restored. Follow the succession of microbes, plants, invertebrates, fish, and birds; discuss sediment, salinity, carbon, feedback loops, and uncertainty. Continue for at least 400 tokens as a coherent scientific narrative.",
    },
    PromptSpec {
        label: "reasoning_probability",
        domain: Domain::Reasoning,
        text: "Solve this problem step by step: two factories produce components with different defect rates, inspections have factory-dependent false-positive and false-negative rates, and shipments are mixed by an unknown source selected from a stated prior. Construct a concrete numerical instance, compute the posterior source and defect probabilities after three mixed inspection outcomes, check the result two ways, and discuss sensitivity. Continue for at least 400 tokens.",
    },
    PromptSpec {
        label: "reasoning_graph_invariant",
        domain: Domain::Reasoning,
        text: "Develop a rigorous proof that every finite connected graph with exactly two vertices of odd degree has an Euler trail between those vertices. Begin from first principles, prove the parity invariant, give a constructive algorithm, handle bridges carefully, and verify the argument on a nontrivial example. Continue for at least 400 tokens and expose every logical dependency.",
    },
    PromptSpec {
        label: "reasoning_allocation_puzzle",
        domain: Domain::Reasoning,
        text: "Create and solve a constrained allocation puzzle with five researchers, four projects, limited GPU days, precedence constraints, incompatible assignments, and a fairness objective. State all numerical constraints, derive the feasible set systematically, prove the optimum, test at least one tempting alternative, and explain which constraint is binding. Continue for at least 400 tokens.",
    },
]
```

- [ ] **Step 4: Add the ignored model-backed probe**

Add:

```rust
#[test]
#[ignore = "requires Qwen3.5 model files and performs a long decode"]
fn probe_qwen35_online_rom() {
    run_qwen35_online_rom_probe().unwrap();
}
```

`run_qwen35_online_rom_probe` must:

1. Require `HIGGS_MODEL_PATH` and `HIGGS_ROM_OUTPUT_DIR`.
2. Load `tokenizer.json`, `load_qwen3_5_model`, and one model instance.
3. Wrap each prompt as
   `"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"`,
   create a fresh model cache, tokenize with special-token recognition enabled,
   prefill it, and greedily select the first decode token.
4. Before every one-token decode forward, call `diag_request_rom_capture()`; evaluate logits/cache state, take the capture, and push it into `TrajectorySet`.
5. Record generated EOS positions but continue until warmup plus measured count.
6. Analyze each site immediately after the prompt, append report metrics, and release its trajectory buffers before starting the next prompt.
7. Write `report.json` only after all configured prompts complete.
8. Refuse to call the canonical pass/fail gate unless all nine prompts and the canonical token/rank settings are active; smoke runs write `gate.passed = false` with a `canonical_run = false` field.

- [ ] **Step 5: Add capture-off versus capture-on logits parity inside the ignored probe**

Before collecting the first prompt, run the same short token sequence with two fresh caches. Request ROM capture only for the second forward, evaluate both logits, and assert `all_close` with `rtol=1e-5`, `atol=1e-5`. Also assert that the captured record count is `num_hidden_layers * 3`. This is mandatory because GitNexus classifies the edited forward function as CRITICAL.

- [ ] **Step 6: Verify unit tests and compile the ignored probe**

Run:

```bash
cargo fmt --all -- --check
cargo test -p higgs-models --lib qwen3_next::rom_probe_tests --no-run
cargo test -p higgs-models --lib qwen3_next::rom_probe_tests -- --skip probe_qwen35_online_rom
```

Expected: compile succeeds and every non-ignored ROM test passes.

- [ ] **Step 7: Run the reduced smoke probe**

Run:

```bash
HIGGS_MODEL_PATH=/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit \
HIGGS_ROM_OUTPUT_DIR=/private/tmp/higgs-rom-qwen35-9b-smoke \
HIGGS_ROM_WARMUP_TOKENS=16 \
HIGGS_ROM_MEASURE_TOKENS=8 \
HIGGS_ROM_RANKS=4,8,16 \
HIGGS_ROM_PROMPT_LIMIT=1 \
cargo test -p higgs-models --release qwen3_next::rom_probe_tests::probe_qwen35_online_rom -- --ignored --exact --nocapture --test-threads=1
```

Expected: one prompt completes; parity passes; the JSON parses; shapes are 32 layers by 24 tokens by 4096 hidden; `canonical_run` is false.

- [ ] **Step 8: Commit Task 5**

Run GitNexus change detection. Stage only `qwen3_next_rom_probe.rs` and commit:

```bash
git add crates/higgs-models/src/qwen3_next_rom_probe.rs
git commit -m "test(rom): add Qwen3.5 rank probe"
```

---

### Task 6: Regression Verification and Canonical Experiment

**Files:**
- Verify: `crates/higgs-models/src/qwen3_next.rs`
- Verify: `crates/higgs-models/src/qwen3_next_rom_probe.rs`
- Create at runtime: `/private/tmp/higgs-rom-qwen35-9b-20260808/report.json`

**Interfaces:**
- Consumes: the completed probe implementation and cached 9B model.
- Produces: verified test results and the canonical ROM go/no-go report.

- [ ] **Step 1: Run focused and package-level regression tests**

Run:

```bash
cargo fmt --all -- --check
cargo test -p higgs-models --lib qwen3_next::rom_probe_tests -- --skip probe_qwen35_online_rom
cargo test -p higgs-models --lib qwen3_next::tests
cargo check -p higgs-models
```

Expected: all commands exit 0. Existing ignored model tests remain ignored.

- [ ] **Step 2: Re-run the existing capture-disabled dense decode benchmark**

Run with no `HIGGS_ROM_*` variables:

```bash
HIGGS_MODEL_PATH=/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit \
BENCH_PROMPT_LEN=128 \
BENCH_DECODE_STEPS=32 \
cargo test -p higgs-models --release qwen3_next::tests::bench_actual_qwen3_5_dense_decode -- --ignored --exact --nocapture --test-threads=1
```

Expected: 32 decode steps complete without a ROM capture request or capture output. Record the warm-step average for context only; do not treat this harness number as the production throughput headline.

- [ ] **Step 3: Run the canonical nine-prompt experiment**

Run:

```bash
HIGGS_MODEL_PATH=/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit \
HIGGS_ROM_OUTPUT_DIR=/private/tmp/higgs-rom-qwen35-9b-20260808 \
cargo test -p higgs-models --release qwen3_next::rom_probe_tests::probe_qwen35_online_rom -- --ignored --exact --nocapture --test-threads=1
```

Expected: all nine prompts complete and `report.json` contains `canonical_run: true`, 864 prompt/site/layer result groups, six rank-64 gate cells, and an explicit overall pass/fail value.

- [ ] **Step 4: Validate the report mechanically**

Run:

```bash
jq -e '.schema_version == 1 and .canonical_run == true and (.prompts | length) == 9 and (.gate.cells | length) == 6' /private/tmp/higgs-rom-qwen35-9b-20260808/report.json
jq '{gate: .gate, aggregates: [.aggregates[] | select(.rank == 64)]}' /private/tmp/higgs-rom-qwen35-9b-20260808/report.json
```

Expected: the first command exits 0; the second prints the authoritative rank-64 result.

- [ ] **Step 5: Run final structural review**

Run:

```bash
node .gitnexus/run.cjs detect-changes --repo higgs
git diff --check HEAD~5..HEAD
git status --short
```

Verify only the two planned source files and the already-committed design/plan documents belong to the ROM work. Do not stage, restore, or modify the user's unrelated parser/streaming files.

- [ ] **Step 6: Report the measured decision**

Return:

- whether each of the six domain/site rank-64 cells passed 0.99;
- aggregate, p05, and worst-layer causal retention at ranks 32/64/128;
- oracle-versus-causal gaps;
- every layer below 0.95;
- the clickable report path;
- whether the result authorizes the `o_proj`/`down_proj` follow-up probe.

Do not describe the work as a runtime speedup. A passing result proves only that the first two linear-input trajectory families support the next measurement phase.
