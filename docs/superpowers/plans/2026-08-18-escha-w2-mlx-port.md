# Escha-W2 MLX/K3 Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the nightly Escha-W2 native MLX path against short-route and invalid-input failures, then establish exactness gates before any prefill QGEMM promotion.

**Architecture:** Keep resident trellis experts and direct Metal reads. Move QMM-only padding away from the native route, add dimension and device-side expert-ID safety guards, and expand tests around the QMV/QGEMM boundary without changing the default route or arithmetic contract. Runtime model benchmarks remain a separate, fresh-process evaluation after the implementation tasks.

**Tech Stack:** Rust workspace, `higgs-models`, MLX Rust bindings, runtime-jitted Metal kernels, `cargo fmt`, and focused `cargo test` commands when the GPU is available.

## Global Constraints

- Work only in `/private/tmp/higgs-escha-w2-mlx-roadmap` on `codex/escha-w2-mlx-roadmap`, based on `nightly`.
- Do not modify the root checkout, PR #275 worktree, MLX dependency pins, or model files.
- Do not port CUDA-only selected-slab uploads, pinned host staging, CPU/GPU layer placement, D3 approximation, or MTP/DSpark speculation.
- Preserve `HIGGS_ESCHA_NATIVE=0` and `HIGGS_ESCHA_TRELLIS_GEMM=1` as rollback/experiment controls; do not change either default.
- Preserve each kernel's arithmetic contract; tolerance-only comparisons cannot justify a claimed faithful port.
- Do not run model loads, Metal benchmarks, or GPU-heavy tests while another Codex is using the GPU; use formatting and source review until the window is clear.
- Before editing any existing function, class, or method, run GitNexus upstream impact analysis and report the blast radius; before each commit, run GitNexus change detection.
- Keep each task independently reviewable and commit it separately.

---

## File Map

- `crates/higgs-models/src/qwen3_next.rs`: global expert sort, native Escha dispatch, and routing regression tests.
- `crates/higgs-models/src/metal_kernel.rs`: shared gather input contract, QMV/QGEMM Metal source, and kernel-boundary tests.
- `crates/higgs-models/src/eschamoe.rs`: native projection exactness/row-boundary tests and the existing model fixture gate.
- `docs/benchmarking.md`: only if the implementation adds a new reproducible parity/digest command or environment variable.

## Task 1: Keep QMM-only padding out of native Escha routing

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next.rs:5120-5305` (`SwitchMlpWeights::forward_gather_global_sort`)
- Test: `crates/higgs-models/src/qwen3_next.rs` existing `test_forward_gather_global_sort_*` module

**Interfaces:**
- Preserve `SwitchMlpWeights::forward_gather_global_sort(&self, x: &Array, indices: &Array) -> Result<Array, Exception>`.
- Preserve the existing sorted `x_sorted`, `idx_sorted`, and `inv_order` semantics.
- The native Escha branch must consume exactly `b * l * top_k` sorted activation rows and the same number of expert IDs; only the affine `gather_qmm` branch may append zero rows assigned to the last expert.

- [ ] **Step 1: Run GitNexus impact analysis**

  Analyze `SwitchMlpWeights::forward_gather_global_sort` upstream and record its callers and risk before editing it. The implementation must not proceed if GitNexus reports HIGH or CRITICAL risk without surfacing that warning.

- [ ] **Step 2: Add a regression test for a short native domain**

  Extend the existing Qwen test module with a deterministic native Escha fixture using a small valid `EschaSpec` and resident code arrays. Exercise `B=1`, `L=1`, `top_k=2` with at least two experts, assert the native call returns `[1, 1, 2, hidden]`, and assert the native gather receives no padded rows. Also cover a native domain of 31, 32, and 33 flattened routed rows so the QMV/QGEMM dispatch boundary is exercised without changing output order.

  The regression must fail before the fix because the old code pads the sorted IDs but reshapes the activations using the unpadded row count.

- [ ] **Step 3: Move the native branch before affine-only padding**

  Keep sorting and inverse permutation shared. Branch to native Escha immediately after the unpadded sorted arrays are created. Move the `num_experts * 4` padding calculation and concatenation below that return path, so `gather_qmm` retains its MLX workaround while native trellis kernels see matched arrays.

  The resulting control flow must have this shape:

  ```rust
  let x_sorted = x_flat.take_axis(&token_idx, 0)?;
  let idx_sorted = idx_flat.take_axis(&order, 0)?;

  if let Some(escha) = &self.escha {
      // Use only the unpadded sorted domain and restore original order.
      // Existing native gate/up, SwiGLU, down, and reshape logic stays here.
  }

  // Padding below this point is exclusively for affine gather_qmm.
  let (x_sorted, idx_sorted) = pad_for_gather_qmm(...)?;
  ```

  Do not duplicate the native path or alter its dtype conversions.

- [ ] **Step 4: Run the focused static checks**

  Run `cargo fmt -p higgs-models -- --check` and `git diff --check`. If the other Codex has cleared the GPU, run the focused test by exact name; otherwise record it as deferred rather than starting MLX execution.

- [ ] **Step 5: Run GitNexus change detection and commit**

  Run `gitnexus detect-changes --repo higgs`, inspect the diff, then commit:

  ```bash
  git add crates/higgs-models/src/qwen3_next.rs
  git commit -m "fix(escha): keep native routes unpadded"
  ```

## Task 2: Make native gather contracts fail closed without host ID synchronization

**Files:**
- Modify: `crates/higgs-models/src/metal_kernel.rs:547-1108` (`check_gather_inputs`, QMV/QGEMM configuration, and the two Metal source strings)
- Test: `crates/higgs-models/src/metal_kernel.rs` private `tests` module

**Interfaces:**
- Preserve `eschamoe_gather_qmv` and `eschamoe_gather_qgemm` signatures and output shapes.
- Preserve the no-host-read QGEMM route; invalid expert IDs must be guarded in the Metal kernel rather than synchronously copied to Rust for normal execution.
- Keep the existing `EschaSpec` K range and `xh` float32 contract.

- [ ] **Step 1: Run GitNexus impact analysis**

  Analyze `check_gather_inputs`, `eschamoe_gather_qmv`, and `eschamoe_gather_qgemm` upstream. Record direct callers and execution-flow risk before changing any of the three symbols.

- [ ] **Step 2: Add shape/count validation tests first**

  Add focused tests that construct small host arrays and assert errors for a code tensor whose leading expert dimension differs from `spec.num_experts`, for a code tensor with the wrong trailing tile dimensions, and for expert IDs with the wrong dtype or row count. Keep the valid control case unchanged.

  The essential assertions are:

  ```rust
  assert!(check_gather_inputs(&xh, &wrong_expert_count, &ids, &spec, "test").is_err());
  assert!(check_gather_inputs(&xh, &code, &ids.as_dtype(Dtype::Int32).unwrap(), &spec, "test").is_err());
  assert!(check_gather_inputs(&xh, &code, &short_ids, &spec, "test").is_err());
  ```

- [ ] **Step 3: Validate the expert axis and guard device IDs**

  In `check_gather_inputs`, require `code.shape()[0] == spec.num_experts` in addition to the existing trailing-dimension check. Add an `E` template argument to both QMV and QGEMM configurations. In the QMV source, read the row's ID and write zero output for invalid IDs before forming the expert base pointer. In QGEMM, treat an invalid expert selected by the block's expert walk as inactive with zero weights, never forming `code + expert * stride` for that ID.

  The guard must be before pointer arithmetic and must not add `expert_ids.as_slice()` or `eval()` to the normal Rust path. Document that router-produced valid IDs retain the fast path while malformed device IDs fail closed to zero output.

- [ ] **Step 4: Add invalid-ID kernel coverage**

  Add a small QMV test with an ID equal to `spec.num_experts` and assert the call completes without an out-of-bounds access and returns an all-zero row. Add the corresponding QGEMM case to the existing 33+ row synthetic coverage. Retain the current valid sorted, unsorted, singleton, and boundary cases.

- [ ] **Step 5: Run static checks, change detection, and commit**

  Run `cargo fmt -p higgs-models -- --check`, `git diff --check`, and `gitnexus detect-changes --repo higgs`. Run the focused kernel tests only when GPU use is cleared, then commit:

  ```bash
  git add crates/higgs-models/src/metal_kernel.rs
  git commit -m "fix(escha): guard native gather inputs"
  ```

## Task 3: Add exactness-oriented Escha row and trajectory gates

**Files:**
- Modify: `crates/higgs-models/src/eschamoe.rs` native gather tests around `eschamoe_proj_gather_forward_matches_oracle` and `eschamoe_gather_qgemm_matches_scratch_matmul`
- Modify: `crates/higgs-models/src/qwen3_next.rs` ignored `escha_native_fixture` only if the digest gate can be added without changing model loading behavior
- Modify: `docs/benchmarking.md` only if a new digest environment variable or command is added

**Interfaces:**
- Keep tolerance comparisons against decoded CPU/scratch oracles as diagnostic tests; do not relabel them as bit-exact.
- A faithful same-kernel test compares `f32::to_bits()` row outputs, while cross-backend tests remain tolerance/quality gates.
- The real-checkpoint fixture remains ignored when the local Escha model is absent.

- [ ] **Step 1: Run GitNexus impact analysis**

  Analyze `EschaProj::gather_forward`, `eschamoe_gather_qgemm_matches_scratch_matmul`, and `escha_native_fixture` before editing any of them. Record the affected test flows and risk.

- [ ] **Step 2: Add same-kernel row-independence tests**

  Extend the synthetic native tests with one deterministic transformed activation row and one expert ID. Repeat that identical logical row at row counts 1, 31, 32, and 33, and compare the repeated output row bit-for-bit against the single-row result for the path that owns each boundary. Add sorted/unsorted permutations and a 32-distinct-expert QGEMM block, while retaining tolerance-only comparisons to scratch.

  Use bit comparisons like:

  ```rust
  let want_bits: Vec<u32> = single
      .as_slice::<f32>()
      .iter()
      .map(|value| value.to_bits())
      .collect();
  for got_row in repeated.as_slice::<f32>().chunks(out_f as usize) {
      let got_bits: Vec<u32> = got_row.iter().map(|value| value.to_bits()).collect();
      assert_eq!(got_bits, want_bits);
  }
  ```

  Adapt the slice shapes to the existing MLX test helpers; the test must compare corresponding rows, not a CPU matmul with a different reduction order.

- [ ] **Step 3: Strengthen the ignored real-checkpoint gate**

  Reuse the existing fixture model/cache setup to add an opt-in fixed prompt trajectory mode. Record the first token and 128 subsequent greedy token IDs under a new `HIGGS_ESCHA_TOKEN_DIGEST_OUT`/`HIGGS_ESCHA_TOKEN_DIGEST_REF` pair, and assert the complete digest whenever the reference variable is supplied. Do not change model loading or introduce a second inference implementation; if the existing fixture cannot expose one-token steps without changing those interfaces, report that concrete blocker and keep the current fixture diagnostic.

- [ ] **Step 4: Run static checks, change detection, and commit**

  Run `cargo fmt -p higgs-models -- --check`, `git diff --check`, and `gitnexus detect-changes --repo higgs`. Run the focused synthetic/fixture tests only when GPU use is cleared, then commit:

  ```bash
  git add crates/higgs-models/src/eschamoe.rs crates/higgs-models/src/qwen3_next.rs docs/benchmarking.md
  git commit -m "test(escha): add exactness gates"
  ```

## Task 4: Whole-branch verification and benchmark handoff

**Files:**
- Modify: none unless a reviewer identifies a required test/doc correction.
- Read: all Task 1–3 reports, the design spec, and `docs/benchmarking.md`.

- [ ] **Step 1: Check the complete branch diff and ledger**

  Confirm only the planned Escha files and the committed spec/plan changed. Run `git diff --check`, `git status --short`, and `git log --oneline nightly..HEAD`.

- [ ] **Step 2: Run non-GPU verification**

  Run `cargo fmt -p higgs-models -- --check` and the repository's static checks that do not initialize MLX/Metal. Do not run `cargo clean`, full `cargo build`, model loading, or benchmark commands in the other Codex's GPU window.

- [ ] **Step 3: Prepare the deferred fresh-process benchmark matrix**

  When the other Codex clears the machine, run alternating fresh processes for scratch versus `HIGGS_ESCHA_TRELLIS_GEMM=1` at 64/1K/4K prefill and a fixed 1K decode window. Record median/tail latency, TTFT, true causal decode rate, load time, active/peak MLX memory, greedy token digest, and sorted/ragged routing. Do not promote the default from a synthetic or even-32-only gain.

- [ ] **Step 4: Run GitNexus detection and final review**

  Run `gitnexus detect-changes --repo higgs`, obtain a whole-branch review package from the subagent-driven-development workflow, and resolve all Critical/Important findings before declaring the branch ready for an upstream PR.
