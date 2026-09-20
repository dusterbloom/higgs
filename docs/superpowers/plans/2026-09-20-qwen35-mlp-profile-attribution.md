# Qwen3.5 MLP Profile Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `HIGGS_PROFILE=1` with synchronized stage timings for the dense Qwen3.5 MLP serving path.

**Architecture:** Keep the diagnostic private to `qwen3_next.rs`. The existing outer profiler activates a thread-local accumulator only for its representative three-GDN-plus-one-full-attention layer sample. `FfnBlock` synchronizes and records input readiness, the Prism Hadamard rotation, fused gate/up projection, split plus SwiGLU, down projection, and residual. Normal execution remains unchanged when profiling is disabled.

**Tech Stack:** Rust, MLX arrays/evaluation barriers, tracing, GitNexus.

## Global Constraints

- Modify only `crates/higgs-models/src/qwen3_next.rs` plus this plan.
- Add no dependency or public API.
- Reuse `HIGGS_PROFILE=1`; add no runtime option.
- Limit additional `eval` barriers to the four layers already sampled by the outer profiler.
- Keep the normal non-profiled path byte-for-byte equivalent in operation ordering.
- GitNexus cannot index `qwen3_next.rs` at its configured size limit, so symbol risk is `UNKNOWN`; text tracing confirms `DecoderLayer::mlp -> FfnBlock::forward` at the serving call sites.

---

### Task 1: Add synchronized MLP stage attribution

**Files:**
- Modify: `crates/higgs-models/src/qwen3_next.rs:3951-3970,7718-7838,7911-8021,9408-9465`

**Interfaces:**
- Consumes: existing `HIGGS_PROFILE=1`, `mlx_rs::transforms::eval`, and the sampled-layer condition in `forward_raw_hidden_with_taps_from_hidden`.
- Produces: one `PROFILE-MLP` tracing event per profiled chunk with per-sampled-layer milliseconds for `input_ready`, `hadamard`, `gate_up`, `swiglu`, `down`, and `residual`.

- [ ] **Step 1: Add private thread-local state**

Add a defaultable counter struct, an active flag, a scoped helper that restores the previous flag even when the closure returns an error, short stage-recording helpers, and a take/reset helper. Store nanoseconds as `u128`, samples as `u32`, and fused-path samples separately.

- [ ] **Step 2: Instrument the fused Prism path**

In `dense_hidden_fused`, start timing only after the one-time fused parameter concatenation. When the profile flag is active, evaluate and record the Hadamard output, fused gate/up output, and split-plus-SwiGLU output before advancing to the next stage.

- [ ] **Step 3: Instrument entry and down projection**

In the dense branch of `FfnBlock::forward`, evaluate the input and record `input_ready` before dispatch. Evaluate the down-projection output and record `down`. Leave all fast-path selection unchanged.

- [ ] **Step 4: Scope sampling and report**

Wrap only the existing sampled `layer.mlp.forward` call with the active-profile helper. Time the residual separately after the now-materialized MLP output. Take and reset the counters once per chunk and emit `PROFILE-MLP`, including stage sample counts so unsupported paths cannot masquerade as zero-cost stages.

- [ ] **Step 5: Verify without GPU**

Run:

```bash
cargo fmt --all -- --check
cargo check --release -p higgs-models -p higgs-engine -p higgs
git diff --check
```

Expected: all commands exit 0; existing unrelated warnings are allowed.

- [ ] **Step 6: Verify on the exact model**

Build the release server, restart port 9000 with `HIGGS_PROFILE=1`, and run distinct exact 512-token and 4096-token prompts. Confirm every chunk emits `PROFILE-MLP`, all sampled Prism layers report the fused stages, and the stage sum explains the existing outer MLP attribution. Restart the normal server afterward.

- [ ] **Step 7: Analyze, commit, and publish**

Run GitNexus `detect-changes --scope all`, rejecting partial/truncated results, then commit the plan and source change. Fast-forward local `nightly` and push `fix/mlx-031-compat` to `fork` without rewriting the diverged remote `nightly`.
