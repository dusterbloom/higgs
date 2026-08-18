# Qwen3.8 Low-Power Performance Optimization Design

## Objective

Push the local Qwen3.8 27B Higgs path toward the physical throughput limit on
this Apple M4 while preserving exact greedy output and production cache
semantics. Treat Low Power Mode as a first-class target: a change is more
valuable when it improves both normal AC and Low Power Mode throughput, or when
it materially improves Low Power Mode without sacrificing normal throughput.

The current reference is the grouped cross-row verifier path at MTP draft depth
8. Its three-trial median was 3.67 tok/s versus 2.90 tok/s for stock cross-row
dispatch, with matching measured and whole-trajectory token digests.

## Constraints and invariants

- Target model: `/Users/peppi/AI-Models/qwen38-higgs` (local Qwen3.8 27B).
- Work remains isolated on `codex/qwen38-higgs-port`.
- Draft and verifier token IDs, measured suffix digest, whole-trajectory digest,
  cache offsets, and accepted-token counts must remain unchanged for paired
  normal/optimized runs.
- Unsupported shapes, quantization, row counts, or disabled feature settings
  must retain the existing stock fallback.
- No challenge-specific implementation, model-policy change, or public API is
  admitted without an independent correctness justification.
- Power-state changes are reversible and benchmark-only; do not leave Low Power
  Mode or other system settings modified after the experiment.

## Measurement protocol

### Power-state stabilization

Before comparing power modes, confirm the charger is genuinely connected and
the battery is not discharging. Record `pmset -g ps`, `pmset -g batt`, and
`pmset -g custom` immediately before each trial. Runs showing continued battery
discharge or thermal pressure are annotated or rejected rather than mixed into
the comparison.

Normal and Low Power Mode trials alternate in fresh processes:

`Normal → Low Power → Normal → Low Power → Normal → Low Power`.

Keep display state, model, prompt, decode length, environment, and machine
activity fixed. Use one-second power/thermal telemetry when available; it is
used to identify throttled runs, not to claim cross-device energy efficiency.

### Benchmark matrix

Start with the production MTP benchmark and the local model using:

- `BENCH_PROMPT_LEN=256`;
- `BENCH_DECODE_STEPS=32` for the first depth sweep;
- `HIGGS_MTP_ADAPTIVE_DRAFT=0`;
- `HIGGS_CROSSROW_QMV=1` for grouped and `0` for stock;
- `HIGGS_MTP_DRAFT_N_MAX=1..8`;
- identical warm-up and digest accounting in every run.

The winning depths are then retested with a longer decode window (target 128
or 256 tokens) and representative prompt lengths. Each candidate records
median tok/s, average cycle time, acceptance rate, verifier rows, measured
count/digest, whole count/digest, and thermal/power annotations.

### Retention rule

A candidate must pass all exactness and focused-test gates. Prefer it when it
improves the target workload by at least 3% in paired median throughput with
the same trajectory. A Low Power Mode win is especially valuable when normal
mode does not regress by more than 1%; noisy or one-off gains remain diagnostic
only.

## Optimization ladder

### 1. Dispatch and configuration experiments

Measure the existing cross-row path against the available QGEMM verifier path
for `T=2..9`, representative `N,K` shapes, and the production MTP cycle. Test
whether caching immutable Metal configuration objects reduces host launch
overhead without changing kernel behavior.

### 2. Cross-row kernel experiments

First test broadcasting repeated scale/bias metadata within SIMD quartets.
Preserve each row's arithmetic and accumulation order, and require the
existing bit-exact cross-row suite before timing. Then test alternative row
groupings only for the measured hot widths (especially triple groups and the
`M=5` schedule), keeping the single source of truth for dispatch and layout.

### 3. Trace-guided deeper work

Capture warmed Metal System Traces for the best grouped and stock paths. Use
the trace to distinguish GPU kernel time, host/FFI launch overhead, and the
serial GPU-to-CPU dependency in MTP drafting. Consider fused MLP/GDN work only
if it is a material share of the trace and has an isolated exact fallback.

Device-resident drafting is a last resort: the recurrence is semantically
serial and changing it risks token/cache divergence, so it requires a separate
design and proof rather than an overnight opportunistic patch.

## Verification and delivery

For every code candidate:

1. Run focused exactness tests for all supported cross-row row counts and
   fallback boundaries.
2. Run MTP controller, digest, and production-harness tests.
3. Run release compile/check and targeted rustfmt/diff checks.
4. Repeat normal/Low Power paired real-model measurements.
5. Request independent review of only that candidate's diff.

Keep benchmark instrumentation and each production optimization in separate,
revertible commits. The final handoff includes raw paired results, exact
digests, power/thermal caveats, and a clean diff suitable for upstream review.
