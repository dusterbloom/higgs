# Qwen3.8 Minimal Upstream Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port only the code required to reproduce the validated Qwen3.8 27B crossrow performance result on current upstream `main`.

**Architecture:** Keep upstream's existing Qwen3-Next model and quantization path. Add the MLX 0.31 fork dependency required by the fast Metal-kernel ABI, a self-contained crossrow affine-4/group-64 QMV kernel, and a narrowly gated QLinear dispatch with conservative M4/M7 fallback boundaries. Exclude unrelated nightly subsystems and benchmark-only scaffolding.

**Tech Stack:** Rust, MLX Rust bindings, MLX fast Metal kernels, Cargo, exactness regression tests.

## Global Constraints

- Base the contribution on the latest fetched `upstream/main`.
- Preserve bit-exact output relative to the stock QMV path.
- Enable the optimization only for affine 4-bit group-64 MTP verify shapes with 2–9 rows.
- Fall back to stock QMV for known divergent M4 and M7 shapes.
- Do not include cache, DFlash, Bonsai, Escha, MTP benchmark harness, or session-stack changes.

### Task 1: Create the clean upstream branch

**Files:** Git refs only.

- [ ] Switch the isolated worktree to a new branch based on `upstream/main`.
- [ ] Confirm the worktree has no unrelated tracked changes.

### Task 2: Port the crossrow kernel and dispatch

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `crates/higgs-models/src/lib.rs`
- Create: `crates/higgs-models/src/crossrow_qmv.rs`
- Modify: `crates/higgs-models/src/qwen3_next.rs`

- [ ] Add the pinned MLX fork dependency needed for `mlx_fast_metal_kernel`.
- [ ] Add the crossrow affine-4/group-64 kernel and its schedule helpers.
- [ ] Wire QLinear's existing affine quantized path to the kernel only for eligible MTP verify shapes.
- [ ] Preserve the stock path for unsupported shapes, disabled opt-out, kernel errors, M4, and M7.
- [ ] Add or retain focused schedule and exactness tests for the new path.

### Task 3: Validate and publish

**Files:** None beyond Task 2.

- [ ] Run formatting, focused model tests, and the release build.
- [ ] Run the local Qwen3.8 benchmark/control needed to verify the optimization is active and exact.
- [ ] Run `omen diff` against `upstream/main` and inspect the final file/LOC risk.
- [ ] Commit only the focused implementation and push a new draft PR.
