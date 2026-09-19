# Qwen3.8 Performance Port to Higgs Design

## Objective

Port the relevant performance ideas from the latest promoted Qwen3.8 challenge
result into Higgs in the most correct Rust/MLX-native way, so the resulting
changes can be benchmarked and then upstreamed to `origin/nightly`.

The target is equivalent model behavior with measurable decode or MTP
verification improvement. Challenge-specific scoring, manifests, and Swift
implementation details are not integration requirements.

## Existing Baseline

`nightly` already contains a Rust-native cross-row affine4/group64 QMV path
with direct nibble extraction, shared weight reads, and exactness coverage for
two through nine rows. It is wired into `QLinear::forward` and falls back to
the stock quantized matmul path when its shape or quantization preconditions do
not hold.

The current cross-row schedule pairs rows (`2 + 2 + ...`, with a single-row
tail). The promoted challenge result instead uses wider groups for the hot
widths, notably `3 + 3 + 2` for eight rows and `3 + 3 + 3` for nine rows. That
grouping is the clearest missing optimization that maps onto the existing Higgs
architecture. Higgs also already has compiled GDN/QMV paths and an adaptive MTP
controller, so those mechanisms should be extended only where their existing
interfaces and correctness invariants support it.

## Scope

### In scope

- Establish a reproducible `nightly` baseline for the relevant Higgs model and
  MTP verification paths.
- Extend the cross-row QMV kernel with an explicit, compile-time-dispatched
  row-group schedule for the measured hot widths while preserving the current
  arithmetic, output types, and fallback behavior.
- Keep exactness tests for every supported row count and add coverage for
  grouped layouts and adversarial signed/zero nibble patterns.
- Measure the existing adaptive MTP policy before changing it. If telemetry
  shows a safe opportunity, tune or extend the model-agnostic controller with
  focused tests and an environment-gated rollout.
- Evaluate additional fused GDN, MLP, attention, or top-2 operations only when
  there is a direct Rust/MLX equivalent and an isolated benchmark can show a
  benefit without changing token selection.
- Keep changes separated into reviewable commits so each optimization can be
  reverted or upstreamed independently.

### Out of scope

- Copying Swift/Metal source literally where Higgs already has a different
  kernel or tensor API.
- Importing challenge manifests, submission metadata, challenge-only model
  heads, or scoring logic.
- Changing the model's greedy token trajectory, quantization semantics, cache
  ownership, or public API merely to match the challenge implementation.
- Modifying the unrelated PR #187 worktree or its untracked files.

## Integration Strategy

1. Work from a clean branch based on `nightly` in an isolated worktree.
2. Capture narrow baseline correctness and performance measurements before
   editing production code.
3. Implement row grouping behind the existing cross-row eligibility checks.
   Use one source of truth for group layout so dispatch, indexing, and tests
   cannot diverge. Preserve the existing direct nibble arithmetic and shared
   weight-load behavior.
4. Verify exact outputs against the stock per-row quantized matmul for all
   supported row counts and representative K/N shapes before comparing speed.
5. Benchmark M8/M9 and neighboring widths separately. Keep the grouping change
   only if it improves the hot widths without an unacceptable regression on
   other eligible widths.
6. Inspect the MTP policy and existing hidden-state interfaces. A post-norm
   reuse optimization is considered only if Higgs can expose the equivalent
   state without duplicating or reordering model work; otherwise it remains a
   documented follow-up rather than a speculative port.
7. Review any fused-kernel candidate as a separate experiment with an exact
   fallback and token-by-token verification.

## Correctness Invariants

- Cross-row output is bit-exact with the stock quantized matmul within the
  existing test contract for every supported M, including odd tails.
- Grouping changes only work decomposition; it does not change nibble order,
  affine zero/scale handling, BF16 conversion, accumulation order where the
  exactness contract depends on it, or output dtype.
- Unsupported quantization, shape, row count, disabled-feature environment
  settings, and non-eligible paths continue to use the existing fallback.
- MTP draft and verification token IDs remain unchanged for the same input and
  cache state.
- Adaptive depth remains bounded and cannot increase work after poor
  acceptance; serial fallback remains available.
- Model cache and hidden-state ownership remain unchanged.

## Verification

Verification proceeds from narrow to broad:

1. Inspect and record the clean `nightly` baseline.
2. Run focused cross-row exactness tests and MTP controller tests before and
   after each relevant change.
3. Run formatting, compile checks, and the affected package tests.
4. Run the Qwen3.8 performance benchmark with stable settings and compare
   eligible widths, fallback widths, and disabled-path behavior.
5. Check the final diff for challenge-specific leakage, accidental fallback
   removal, unresolved markers, and unrelated worktree changes.
6. Request an independent code review before considering the branch ready for
   publication.

Success means the port is exact, model-agnostic where it claims to be, and
shows a measured path toward closing or exceeding the relevant promoted
performance on `nightly`.

## Delivery

The work remains on `codex/qwen38-higgs-port` until verification and review are
complete. Only then should it be pushed and proposed against `origin/nightly`,
with the optimization commits and benchmark evidence included in the handoff.
