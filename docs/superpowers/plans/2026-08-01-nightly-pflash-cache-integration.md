# Nightly and PFlash Cache Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge all 49 commits unique to `feat/pflash-bonsai-q2-dspark-cache` into the NanBeige-capable `nightly` line without losing either branch's behavior or history.

**Architecture:** Perform one non-fast-forward merge on `integration/nightly-pflash-cache`, using the pflash side as the cache/runtime baseline and semantically reapplying NanBeige's generic-transformer extensions in the six overlapping files. Preserve documentation that the pflash branch deleted during its in-flight commit, then verify both ancestry and behavior from narrow NanBeige/cache tests through workspace-wide checks.

**Tech Stack:** Git, Rust 2024 workspace, Cargo, MLX/Metal model runtime, shell smoke harnesses.

## Global Constraints

- Source scope is exactly `8be7d7445..feat/pflash-bonsai-q2-dspark-cache`; unrelated divergent branches remain out of scope.
- Preserve both `f92694b76` (NanBeige) and `e64369170` (latest cache fix) as ancestors of the integration tip.
- Preserve merge history with `--no-ff`; do not squash or replay the 49 commits.
- Keep the pflash implementation for cache, session, radix, PFlash, dSpark, metrics, and request-routing behavior.
- Keep NanBeige loader, registry, shared-weight loop, direct-quantization, logical KV-layer, non-batched, documentation, and smoke behavior.
- Do not accept the broad documentation deletions carried by `9cdd5e5ed` as feature work.
- Do not move the existing `nightly` ref during implementation.
- Do not require real model weights for the default verification suite; hardware/model-path smoke tests remain ignored unless their inputs are already available.

---

## File Structure and Responsibilities

- `crates/higgs-models/src/transformer.rs`: generic dense transformer, including NanBeige shared physical layers and logical loop passes.
- `crates/higgs-models/src/lib.rs`: `AnyModel` dispatch and cache construction across dense, hybrid, MoE, and Bonsai variants.
- `crates/higgs-engine/src/model_loader.rs`: model-type routing into the generic transformer loader.
- `crates/higgs-models/src/registry.rs`: public supported-model registry.
- `crates/higgs/src/config.rs`: cache/PFlash model configuration and batch eligibility.
- `crates/higgs/src/doctor.rs`: configuration diagnostics for merged cache and model behavior.
- `crates/higgs/src/state.rs`: request defaults, engine construction, and merged runtime state.
- `README.md`, `docs/models.md`, `docs/configuration.md`: user-facing union of cache and NanBeige documentation.
- `scripts/release_smoke_cached_models.sh`: cached-model smoke matrix with NanBeige opt-in.
- `crates/higgs-engine/src/cache/`, `paged_prefix_cache.rs`, `simple.rs`, and route modules: source-branch cache stack, expected to merge without semantic hand-editing outside conflicts.

---

### Task 1: Establish a Reproducible Baseline

**Files:**
- Verify: repository and test state only

**Interfaces:**
- Consumes: `integration/nightly-pflash-cache` at design commit `f1bc364e8`.
- Produces: recorded clean status, source/target commit identities, and passing NanBeige baseline tests before the merge.

- [ ] **Step 1: Verify branch and worktree state**

Run:

```bash
git status --short --branch
git branch --show-current
git rev-parse integration/nightly-pflash-cache nightly feat/pflash-bonsai-q2-dspark-cache
git merge-base nightly feat/pflash-bonsai-q2-dspark-cache
git rev-list --left-right --count nightly...feat/pflash-bonsai-q2-dspark-cache
```

Expected: clean `integration/nightly-pflash-cache`; merge base `8be7d7445c88110f253061da20b795a679d5964c`; branch distance `1 49` before accounting for the design commit.

- [ ] **Step 2: Prove the NanBeige baseline compiles and passes**

Run:

```bash
cargo test -p higgs-models nanbeige --lib
cargo test -p higgs-engine model_config_from_dir_nanbeige --lib
cargo test -p higgs test_batch_support_model_types_excludes_nanbeige --lib
```

Expected: all selected tests pass. If this baseline fails, stop and report it before merging so a pre-existing failure is not attributed to the integration.

- [ ] **Step 3: Record source-side cache test inventory**

Run:

```bash
git grep -n '#\[test\]' feat/pflash-bonsai-q2-dspark-cache -- crates/higgs-engine/src/cache crates/higgs-engine/src/paged_prefix_cache.rs crates/higgs-engine/src/simple.rs
git ls-tree -r --name-only feat/pflash-bonsai-q2-dspark-cache crates/higgs-engine/tests
```

Expected: the source exposes unit coverage plus `cache_memory_bounds`, `concurrent_session`, `cross_turn_hybrid_reuse`, `dspark_radix_cache`, `dspark_session_cache`, `golden_cache_equivalence`, and `session_prefill` integration targets.

---

### Task 2: Merge the PFlash History and Resolve Non-Model Conflicts

**Files:**
- Modify: `README.md`
- Modify: `crates/higgs/src/config.rs`
- Modify: `crates/higgs/src/doctor.rs`
- Modify: `crates/higgs/src/state.rs`
- Preserve: `docs/BONSAI_Q1.md`
- Preserve: `docs/DSPARK_MLX_DESIGN.md`
- Preserve: `docs/benchmarking.md`
- Preserve: `docs/codebase-review-2026-06.md`
- Preserve: `docs/configuration.md`
- Preserve: `docs/higgs-header.jpg`
- Preserve: `docs/kv-prune-eval-harness.codex.md`
- Preserve: `docs/mlx_rs_capabilities.md`
- Preserve: `docs/models.md`
- Preserve: `docs/qwen3_next_architecture.md`
- Merge without hand edits: all source-only cache/model/test/benchmark additions

**Interfaces:**
- Consumes: pflash configuration fields and `nightly`'s NanBeige batch/model handling.
- Produces: an unresolved merge containing only the two model-core conflicts reserved for Task 3.

- [ ] **Step 1: Start the explicit merge without committing**

Run:

```bash
git merge --no-ff --no-commit feat/pflash-bonsai-q2-dspark-cache
git status --short
git diff --name-only --diff-filter=U
```

Expected: content conflicts in `README.md`, `crates/higgs-models/src/lib.rs`, `crates/higgs-models/src/transformer.rs`, `crates/higgs/src/config.rs`, `crates/higgs/src/doctor.rs`, and `crates/higgs/src/state.rs`, plus modify/delete conflicts for retained documentation.

- [ ] **Step 2: Preserve documentation deleted only by the source in-flight commit**

Run:

```bash
git restore --ours -- docs/BONSAI_Q1.md docs/DSPARK_MLX_DESIGN.md docs/benchmarking.md docs/codebase-review-2026-06.md docs/configuration.md docs/higgs-header.jpg docs/kv-prune-eval-harness.codex.md docs/mlx_rs_capabilities.md docs/models.md docs/qwen3_next_architecture.md
git add docs/BONSAI_Q1.md docs/DSPARK_MLX_DESIGN.md docs/benchmarking.md docs/codebase-review-2026-06.md docs/configuration.md docs/higgs-header.jpg docs/kv-prune-eval-harness.codex.md docs/mlx_rs_capabilities.md docs/models.md docs/qwen3_next_architecture.md
```

Expected: those paths remain present and leave the unmerged list.

- [ ] **Step 3: Resolve `config.rs` as the union of both branches**

Keep the pflash-side `PFlashConfig`, prefill plan cache controls, cache retention controls, `ModelConfig` fields, CLI mapping, and cache-related tests. Reintroduce NanBeige's non-batch rule with this exact predicate:

```rust
fn supports_batch_model_type(model_type: &str) -> bool {
    matches!(model_type, "qwen2" | "qwen3" | "llama" | "mistral")
}
```

Retain the test:

```rust
#[test]
fn test_batch_support_model_types_excludes_nanbeige() {
    assert!(supports_batch_model_type("qwen3"));
    assert!(!supports_batch_model_type("nanbeige"));
}
```

Run:

```bash
rg -n 'PFlashConfig|prefill_plan_cache|kv_max_sessions|supports_batch_model_type|nanbeige' crates/higgs/src/config.rs
git add crates/higgs/src/config.rs
```

Expected: all five concepts are present and no conflict markers remain in the file.

- [ ] **Step 4: Resolve `doctor.rs`, `state.rs`, and `README.md` semantically**

For `doctor.rs`, retain pflash diagnostics for cache retention/PFlash while keeping NanBeige-compatible model validation. For `state.rs`, retain pflash request/cache/PFlash defaults and all engine-construction fields while keeping `nightly`'s NanBeige-compatible loader call shape. In `README.md`, retain current NanBeige support/model references and add the source branch's cache/PFlash usage text without duplicating headings.

Run:

```bash
rg -n 'kv_max_session_tokens|kv_retained_idle_secs|prefill|PFlash|nanbeige|Nanbeige' crates/higgs/src/doctor.rs crates/higgs/src/state.rs README.md
rg -n '^(<<<<<<<|=======|>>>>>>>)' README.md crates/higgs/src/config.rs crates/higgs/src/doctor.rs crates/higgs/src/state.rs
git add README.md crates/higgs/src/config.rs crates/higgs/src/doctor.rs crates/higgs/src/state.rs
```

Expected: cache and NanBeige concepts coexist; the conflict-marker search returns no matches.

- [ ] **Step 5: Confirm only model-core conflicts remain**

Run:

```bash
git diff --name-only --diff-filter=U
```

Expected exactly:

```text
crates/higgs-models/src/lib.rs
crates/higgs-models/src/transformer.rs
```

---

### Task 3: Integrate NanBeige into the PFlash Model Core

**Files:**
- Modify: `crates/higgs-models/src/transformer.rs`
- Modify: `crates/higgs-models/src/lib.rs`
- Verify merged source files: `crates/higgs-engine/src/model_loader.rs`
- Verify merged source files: `crates/higgs-models/src/registry.rs`

**Interfaces:**
- Consumes: pflash's expanded `AnyModel`, Bonsai Q2, PFlash, and cache dispatch plus NanBeige changes from `f92694b76`.
- Produces: a generic transformer that supports NanBeige loops/direct quantization and an `AnyModel` cache path compatible with both dense NanBeige and source-side cache variants.

- [ ] **Step 1: Use the pflash model files as the structural baseline**

Run:

```bash
git restore --theirs -- crates/higgs-models/src/lib.rs crates/higgs-models/src/transformer.rs
git show f92694b76 -- crates/higgs-models/src/lib.rs crates/higgs-models/src/transformer.rs > /tmp/nightly-nanbeige-model.patch
```

Expected: the working files contain the newest pflash model structure; `/tmp/nightly-nanbeige-model.patch` is the authoritative NanBeige delta to port semantically.

- [ ] **Step 2: Port NanBeige configuration and validation into `transformer.rs`**

Add or preserve these exact interfaces from `f92694b76` within the pflash structure:

```rust
const fn default_num_loops() -> i32 { 1 }

impl ModelArgs {
    pub fn num_cache_layers(&self) -> Result<i32, ModelError>;
    pub fn supports_batched_decode(&self) -> bool;
    fn direct_quantization(&self) -> Option<&QuantizationConfig>;
    fn uses_direct_quantization(&self) -> bool;
}
```

`ModelArgs` must deserialize `num_loops` with default `1` and `skip_loop_final_norm` with its NanBeige default. `load_model_args` must call `validate_nanbeige_config`, rejecting unsupported NanBeige options, invalid loop metadata, `pretraining_tp > 1`, non-`silu` activations, and unsupported ngram modes exactly as the source commit does.

- [ ] **Step 3: Port NanBeige shared-weight execution and direct quantization**

Retain pflash's transformer implementation, but ensure model construction passes `args.direct_quantization()` through attention/MLP/output projection construction. Both `forward` and hidden/all-logit variants must execute every physical decoder layer for each logical loop, assign distinct cache slots per `(loop_index, layer_index)`, and apply final normalization according to `skip_loop_final_norm`.

Run:

```bash
rg -n 'num_loops|skip_loop_final_norm|num_cache_layers|direct_quantization|validate_nanbeige_config|supports_batched_decode' crates/higgs-models/src/transformer.rs
```

Expected: each interface and execution control appears in production code and its tests.

- [ ] **Step 4: Port NanBeige cache and batch dispatch into `lib.rs`**

Ensure `AnyModel::Transformer` delegates `supports_batched_decode()` to the transformer, `make_cache_with_config` uses `m.num_cache_layers()` rather than physical layer count, and both dense and TurboQuant cache construction receive that logical count. Preserve all pflash-side `BonsaiQ2`, hybrid, PFlash, and dSpark match arms.

Retain the regression:

```rust
#[test]
fn any_model_nanbeige_make_cache_uses_logical_loop_layers() {
    let mut args = small_transformer_args("nanbeige");
    args.num_loops = 2;
    let model = transformer::Model::new(args).unwrap();
    let any = AnyModel::Transformer(model);
    let cache = any.make_cache().unwrap();
    match &cache {
        AnyCache::KV(layers) => assert_eq!(layers.len(), 4),
        AnyCache::Hybrid(_) => panic!("Expected KV cache for Nanbeige"),
    }
    assert!(!any.supports_batched_decode());
}
```

- [ ] **Step 5: Verify loader and registry routes survived the merge**

Run:

```bash
git grep -n 'nanbeige' -- crates/higgs-engine/src/model_loader.rs crates/higgs-models/src/registry.rs
```

Expected: `model_loader` routes `nanbeige` through `transformer::load_model`, and `registry::is_supported` includes `nanbeige`, with their unit tests.

- [ ] **Step 6: Stage model resolution and prove all conflicts are gone**

Run:

```bash
git add crates/higgs-models/src/lib.rs crates/higgs-models/src/transformer.rs
git diff --name-only --diff-filter=U
rg -n '^(<<<<<<<|=======|>>>>>>>)' --glob '*.rs' --glob '*.md' --glob '*.sh' .
git diff --check
```

Expected: both conflict searches and `git diff --check` report no problems.

---

### Task 4: Restore the Full NanBeige User and Smoke Surface

**Files:**
- Modify: `docs/models.md`
- Modify: `docs/configuration.md`
- Modify: `scripts/release_smoke_cached_models.sh`
- Verify: `README.md`

**Interfaces:**
- Consumes: merged NanBeige runtime and source-side cache/PFlash documentation.
- Produces: documented, opt-in cached NanBeige smoke coverage without deleting pflash configuration guidance.

- [ ] **Step 1: Compare the post-merge documentation against both parents**

Run:

```bash
git diff nightly -- README.md docs/models.md docs/configuration.md scripts/release_smoke_cached_models.sh
git diff feat/pflash-bonsai-q2-dspark-cache -- README.md docs/models.md docs/configuration.md scripts/release_smoke_cached_models.sh
```

Expected: the comparisons expose the intended union rather than wholesale loss from either side.

- [ ] **Step 2: Preserve the NanBeige model contract in documentation**

Ensure `docs/models.md` states:

```text
Nanbeige uses repeated shared-weight decoder loops with loop-aware KV cache slots, so it is not included in true batched decode support.
```

Ensure the supported-model table includes `nanbeige` / `Nanbeige4.2` and the tested-model table includes `MercuriusDream/Nanbeige4.2-3B-mlx-6bit`. Retain pflash/cache sections from the source branch wherever they exist in surviving docs.

- [ ] **Step 3: Preserve opt-in NanBeige cached smoke coverage**

Ensure the smoke script contains:

```bash
NANBEIGE_MODEL="${HIGGS_SMOKE_NANBEIGE_MODEL:-MercuriusDream/Nanbeige4.2-3B-mlx-6bit}"
NANBEIGE_EXPECTED_NAME="${HIGGS_SMOKE_NANBEIGE_NAME:-MercuriusDream/Nanbeige4.2-3B-mlx-6bit}"
```

and includes the model only when `HIGGS_SMOKE_INCLUDE_NANBEIGE=1`.

- [ ] **Step 4: Check and stage the documentation/smoke union**

Run:

```bash
bash -n scripts/release_smoke_cached_models.sh
rg -n 'Nanbeige|nanbeige|HIGGS_SMOKE_INCLUDE_NANBEIGE|PFlash|prefix cache' README.md docs/models.md docs/configuration.md scripts/release_smoke_cached_models.sh
git add README.md docs/models.md docs/configuration.md scripts/release_smoke_cached_models.sh
git diff --cached --stat
```

Expected: shell syntax passes, NanBeige and cache behavior are documented, and no broad doc deletion appears in the staged stat.

---

### Task 5: Run the Integration Test Ladder and Commit the Merge

**Files:**
- Verify: entire staged merge
- Commit: merge result

**Interfaces:**
- Consumes: fully resolved and staged merge.
- Produces: one verified merge commit with both required histories.

- [ ] **Step 1: Format and compile before tests**

Run:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
```

Expected: both commands exit `0`.

- [ ] **Step 2: Run NanBeige regression tests**

Run:

```bash
cargo test -p higgs-models nanbeige --lib
cargo test -p higgs-engine model_config_from_dir_nanbeige --lib
cargo test -p higgs test_batch_support_model_types_excludes_nanbeige --lib
```

Expected: all selected tests pass; the real-checkpoint test remains ignored unless explicitly requested with a model path.

- [ ] **Step 3: Run cache unit suites**

Run:

```bash
cargo test -p higgs-engine --lib cache
cargo test -p higgs-engine --lib radix
cargo test -p higgs-engine --lib session
cargo test -p higgs-engine --lib hybrid
cargo test -p higgs-models --lib cache
```

Expected: all selected unit tests pass.

- [ ] **Step 4: Run cache integration targets**

Run:

```bash
cargo test -p higgs-engine --test cache_memory_bounds
cargo test -p higgs-engine --test concurrent_session
cargo test -p higgs-engine --test cross_turn_hybrid_reuse
cargo test -p higgs-engine --test dspark_radix_cache
cargo test -p higgs-engine --test dspark_session_cache
cargo test -p higgs-engine --test golden_cache_equivalence
cargo test -p higgs-engine --test session_prefill
```

Expected: all non-ignored integration tests pass. Tests requiring external checkpoints may report ignored, not failed.

- [ ] **Step 5: Run workspace-wide verification**

Run:

```bash
cargo test --workspace
```

Expected: exit `0`; record ignored hardware/model tests separately.

- [ ] **Step 6: Audit the staged merge before committing**

Run:

```bash
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached --diff-filter=D --name-status
git grep -n -E '^(<<<<<<<|=======|>>>>>>>)' -- ':!Cargo.lock'
```

Expected: no unmerged paths or conflict markers; deleted paths are intentional source replacements rather than the protected documentation set.

- [ ] **Step 7: Create the merge commit**

Run:

```bash
git commit -m "merge: integrate complete pflash cache stack"
```

Expected: Git records a two-parent merge commit.

- [ ] **Step 8: Prove ancestry and final repository state**

Run:

```bash
git merge-base --is-ancestor f92694b76 HEAD
git merge-base --is-ancestor e64369170 HEAD
git show -s --format='%H%n%P%n%s' HEAD
git status --short --branch
```

Expected: both ancestry checks exit `0`; `HEAD` has two parents; the worktree is clean on `integration/nightly-pflash-cache`.

---

### Task 6: Review the Union Against Both Parents

**Files:**
- Review: all files changed by the merge

**Interfaces:**
- Consumes: verified merge commit.
- Produces: a review report identifying any lost feature, accidental deletion, or follow-up required before moving `nightly`.

- [ ] **Step 1: Review the first-parent delta**

Run:

```bash
git diff --stat HEAD^1..HEAD
git diff --name-status HEAD^1..HEAD
```

Expected: all source-side cache/model additions appear, with protected documentation retained.

- [ ] **Step 2: Review the second-parent NanBeige delta**

Run:

```bash
git diff --stat HEAD^2..HEAD
git diff HEAD^2..HEAD -- crates/higgs-engine/src/model_loader.rs crates/higgs-models/src/registry.rs crates/higgs-models/src/lib.rs crates/higgs-models/src/transformer.rs crates/higgs/src/config.rs docs/models.md scripts/release_smoke_cached_models.sh
```

Expected: the difference from pflash consists of the NanBeige port, retained documentation, and the design/plan artifacts—not removal of cache behavior.

- [ ] **Step 3: Report delivery state without moving `nightly`**

Report the integration branch name, merge commit, both parent hashes, exact verification commands/results, ignored tests, and any residual risks. Do not update, force, or push `nightly` without a separate explicit instruction.
