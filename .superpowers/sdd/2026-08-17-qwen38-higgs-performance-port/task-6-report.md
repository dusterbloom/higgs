# Task 6 benchmark gap report

## Status

Implemented and compile-verified an ignored real-model benchmark that exercises
the production MTP cycle. On 2026-08-18, the benchmark was run as a three-trial
grouped-ON/stock-OFF AB against the local Qwen3.8 27B checkpoint. The measured
suffix matched by emitted-token count and digest in all six runs; the final
review fix also makes subsequent runs report a comparable whole-trajectory
count and digest that include the unmeasured warm-up.

## Changed files

- `crates/higgs-engine/src/mtp.rs`
  - Added ignored test `bench_production_mtp_cycle_real_model`.
  - No production MTP policy, cache semantics, model code, challenge code, or public API changed.
- `.superpowers/sdd/2026-08-17-qwen38-higgs-performance-port/task-6-report.md`
  - This report.

`crates/higgs-models/src/lib.rs` was briefly changed by workspace-wide rustfmt, then restored byte-for-byte from `HEAD`; it is not part of the final diff.

## Design choice

The benchmark is colocated with the existing `mtp.rs` unit tests so it can call the production engine path directly without exposing new APIs. It loads the real model through `model_loader::load_model`, creates production backbone and MTP caches, and reproduces the production `mtp_generate` bootstrap sequence:

1. Prefill with raw hidden-state capture and last-token logits.
2. Prime the MTP cache with `prime_mtp_cache`.
3. Greedily select and forward the first token.
4. Mirror that token with `mirror_mtp_token`.
5. Repeatedly call production `mtp_cycle` at the runtime-configured draft depth.

Greedy token selection uses the existing MLX `mlx_rs::argmax_axis!(..., -1)` macro convention. Draft depth is resolved by `MlxRuntimeTuning::from_model_dir(..., RequestedMlxProfile::Auto)`, so `HIGGS_MTP_DRAFT_N_MAX` follows the production parser, default, and 1..=8 clamp. `HIGGS_MODEL_PATH`, `BENCH_PROMPT_LEN`, and `BENCH_DECODE_STEPS` follow the existing ignored benchmark conventions; prompt and decode defaults are 256 and 32.

Each cycle prints configured draft depth, verifier batch rows (`T = drafted + 1`), drafted count, accepted count/rate, emitted count, cycle time, and cycle tok/s. The summary prints cycle count, total/min/max/average verifier rows, total drafted/accepted counts, measured emitted count/digest, whole-trajectory emitted count/digest, aggregate acceptance rate, total/average cycle time, and aggregate tok/s. The whole trajectory is the warm-up result followed by the measured suffix; timing statistics remain measured-only. A configured depth of 1..=8 exercises verifier row counts T=2..=9.

A benchmark-only token-accounting helper keeps the measured suffix and whole
trajectory separate. Deterministic unit coverage verifies that warm-up tokens
affect only the whole-trajectory count/digest. The harness remains ignored and
normal tests do not load model files.

## Commands and outputs

### Formatting and diff checks

~~~text
$ rustfmt --edition 2024 --check crates/higgs-engine/src/mtp.rs
exit 0

$ git diff --check
exit 0
~~~

## Task 1 low-power depth sweep (2026-08-18)

Ran three fresh-process trials for grouped (`HIGGS_CROSSROW_QMV=1`) and stock
(`HIGGS_CROSSROW_QMV=0`) at each `HIGGS_MTP_DRAFT_N_MAX` depth 1..8, with
`BENCH_PROMPT_LEN=256`, `BENCH_DECODE_STEPS=64`,
`HIGGS_MTP_ADAPTIVE_DRAFT=0`, and `HIGGS_MTP_MIRROR_VERIFY=0`. Complete raw
logs are under `/private/tmp/higgs-qwen38-sweep/`.

Power metadata: initial `pmset -g ps` was AC Power, battery 48% charging;
final was AC Power, battery 66% charging. `pmset -g custom` reported
`lowpowermode 0` both times. `pmset -g therm` returned macOS status errors
(`0xe00002bc`) for thermal/performance/CPU power status; no visible thermal
pressure or battery drain occurred.

The first sandboxed process aborted in MLX Metal initialization with
`NSRangeException` from `mlx::core::metal::DeviceC2Ev` (empty Metal device
array). `system_profiler SPDisplaysDataType` reported an Apple M4 with Metal
supported. The exact trial was rerun outside the sandbox, then the full 48-run
matrix completed with exit 0. This was an execution-environment blocker, not a
model or harness failure.

Raw three-run medians:

| Depth | Grouped total ms | Grouped avg ms | Grouped tok/s | Stock total ms | Stock avg ms | Stock tok/s | Pair |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 9877.375 | 282.211 | 6.58 | 9082.451 | 259.499 | 7.16 | eligible |
| 2 | 8578.653 | 343.146 | 7.58 | 8052.787 | 322.111 | 8.07 | eligible |
| 3 | 9216.686 | 418.940 | 7.16 | 8843.216 | 421.106 | 7.46 | excluded |
| 4 | 9581.873 | 504.309 | 6.78 | 10049.022 | 528.896 | 6.47 | eligible |
| 5 | 8361.479 | 522.592 | 7.89 | 9239.099 | 577.444 | 7.14 | eligible |
| 6 | 9852.398 | 656.827 | 6.80 | 9541.908 | 681.565 | 7.02 | excluded |
| 7 | 9717.756 | 694.125 | 7.00 | 11769.862 | 840.704 | 5.78 | eligible |
| 8 | 11121.603 | 794.400 | 6.20 | 14362.995 | 1025.928 | 4.80 | eligible |

Exact same-depth grouped/stock parity fields (grouped / stock):

| Depth | Verifier rows | Drafted | Accepted | Measured count/digest | Whole count/digest | Result |
| ---: | :---: | :---: | :---: | :--- | :--- | :--- |
| 1 | 70 / 70 | 35 / 35 | 30 / 30 | 65 / `d9f7648e69fb6545` | 66 / `b1a92999f97f9409` | equal |
| 2 | 75 / 75 | 50 / 50 | 40 / 40 | 65 / `d9f7648e69fb6545` | 66 / `b1a92999f97f9409` | equal |
| 3 | 88 / 84 | 66 / 63 | 44 / 45 | 66 / `627c449dc3ba967e` | 67 / `791ce5685e0782f2` | excluded |
| 4 | 95 / 95 | 76 / 76 | 46 / 46 | 65 / `d9f7648e69fb6545` | 66 / `b1a92999f97f9409` | equal |
| 5 | 96 / 96 | 80 / 80 | 50 / 50 | 66 / `627c449dc3ba967e` | 67 / `791ce5685e0782f2` | equal |
| 6 | 105 / 98 | 90 / 84 | 52 / 53 | 67 / `726e7a647ed849c2` | 68 / `0ea78be42d60840e` | excluded |
| 7 | 112 / 112 | 98 / 98 | 54 / 54 | 68 / `fd0928623b9f2e8f` | 69 / `fc6d89c2f6e52f83` | equal |
| 8 | 126 / 126 | 112 / 112 | 55 / 55 | 69 / `c2925335c643d359` | 70 / `26d2263a1dcf7215` | equal |

Depths 3 and 6 were excluded before ranking because their verifier-row totals
and drafted/accepted counts differed. Selected depth: **5**, the highest
grouped median throughput among eligible exact-parity pairs (7.89 tok/s).

### Whole-trajectory parity rerun after final-review fix

Using the exact grouped-ON and stock-OFF commands below with the same model,
prompt, depth, and decode settings:

| Path | Measured tok/s | Measured digest | Whole count | Whole digest | Total cycle ms |
| --- | ---: | --- | ---: | --- | ---: |
| grouped ON | 3.51 | `03a5ca3d6f61a958` | 34 | `b9e078dd8d30e814` | 9400.844 |
| stock OFF | 3.00 | `03a5ca3d6f61a958` | 34 | `b9e078dd8d30e814` | 11012.790 |

The warm-up-inclusive trajectory now matches exactly in both modes; the final
review parity finding is resolved. The original three-trial timing medians
remain the performance comparison because this rerun exists to validate the
instrumentation change.

## Follow-up production-cycle benchmark (2026-08-18)

The local Qwen3.8 27B checkpoint was available at
`/Users/peppi/AI-Models/qwen38-higgs`. The new engine-level benchmark exercised
the production `mtp_cycle` with `HIGGS_MTP_DRAFT_N_MAX=8`,
`BENCH_PROMPT_LEN=256`, `BENCH_DECODE_STEPS=32`, one unmeasured warm-up cycle,
and `HIGGS_MTP_ADAPTIVE_DRAFT=0`. Grouped runs set `HIGGS_CROSSROW_QMV=1`;
stock runs set `HIGGS_CROSSROW_QMV=0`. Each run used verifier rows `T=9`,
drafted 56 rows, accepted 26 (46.4%), emitted 33 measured tokens, and reported
the same measured-suffix digest `03a5ca3d6f61a958`.

Raw measured summaries:

| Path | Total cycle ms | Avg cycle ms | tok/s |
| --- | ---: | ---: | ---: |
| grouped ON 1 | 10931.201 | 1561.600 | 3.02 |
| stock OFF 1 | 11535.084 | 1647.869 | 2.86 |
| stock OFF 2 | 11379.003 | 1625.572 | 2.90 |
| grouped ON 2 | 9000.226 | 1285.747 | 3.67 |
| grouped ON 3 | 8985.552 | 1283.650 | 3.67 |
| stock OFF 3 | 11312.717 | 1616.102 | 2.92 |

Three-run medians are **3.67 tok/s grouped** versus **2.90 tok/s stock**
(approximately **26.6% faster**), with a median total-cycle reduction from
11379.003 ms to 9000.226 ms. The grouped path is noisier on the first trial,
so the raw samples remain recorded rather than presenting only the median.

This AB established exact parity for the measured suffix. The benchmark at the
time did not include the warm-up tokens in its digest; the final-review fix now
reports a whole-trajectory count/digest so future stock/grouped trials can
compare the warm-up and measured suffix together. The previous one-draft model
microbenchmark remains useful only as a narrow diagnostic.

### Initial focused compile

~~~text
$ cargo test -p higgs-engine --lib mtp --no-run
error[E0423]: expected function, found macro `argmax_axis`
  --> crates/higgs-engine/src/mtp.rs:1164
error[E0423]: expected function, found macro `argmax_axis`
  --> crates/higgs-engine/src/mtp.rs:1176
warning: unused import: `argmax_axis`
~~~

The attempted import was removed; both call sites use the existing
`mlx_rs::argmax_axis!(..., -1)` macro convention.

### Focused debug compile after correction

~~~text
$ cargo test -p higgs-engine --lib mtp --no-run
Finished `test` profile [unoptimized + debuginfo] target(s) in 7.57s
Executable unittests src/lib.rs (target/debug/deps/higgs_engine-eabe5787942cedce)
~~~

### Required focused release compile

~~~text
$ cargo test -p higgs-engine --release --lib mtp --no-run
Finished `release` profile [optimized] target(s) in 1m 11s
Executable unittests src/lib.rs (target/release/deps/higgs_engine-6ae3dd2051b08dd8)
~~~

### Focused release tests

~~~text
$ cargo test -p higgs-engine --release --lib mtp -- --nocapture
running 16 tests
test mtp::tests::bench_production_mtp_cycle_real_model ... ignored, requires model files on disk
test result: ok. 15 passed; 0 failed; 1 ignored; 0 measured; 529 filtered out; finished in 0.00s
~~~

## Benchmark invocation

The 2026-08-18 AB used the following grouped-ON command three times:

~~~bash
HIGGS_MODEL_PATH=/Users/peppi/AI-Models/qwen38-higgs \
BENCH_PROMPT_LEN=256 \
BENCH_DECODE_STEPS=32 \
HIGGS_MTP_DRAFT_N_MAX=8 \
HIGGS_MTP_ADAPTIVE_DRAFT=0 \
HIGGS_CROSSROW_QMV=1 \
cargo test -p higgs-engine --release --lib \
  mtp::tests::bench_production_mtp_cycle_real_model -- \
  --ignored --exact --nocapture
~~~

The stock-OFF command was identical except for the explicit cross-row opt-out:

~~~bash
HIGGS_MODEL_PATH=/Users/peppi/AI-Models/qwen38-higgs \
BENCH_PROMPT_LEN=256 \
BENCH_DECODE_STEPS=32 \
HIGGS_MTP_DRAFT_N_MAX=8 \
HIGGS_MTP_ADAPTIVE_DRAFT=0 \
HIGGS_CROSSROW_QMV=0 \
cargo test -p higgs-engine --release --lib \
  mtp::tests::bench_production_mtp_cycle_real_model -- \
  --ignored --exact --nocapture
~~~

## Concerns

The six-run AB captured measured-suffix parity, not full-trajectory parity,
because its digest accumulator started after warm-up. The final-review fix
closes that instrumentation gap for future runs without changing the recorded
2026-08-18 timings or production MTP behavior. No other bounded-harness concern
is known.

## Task 3 bounded Metal-trace update (2026-08-18)

Task 3 selected the Task 2 grouped/stock pair at depth 5 (`T=6`) and attempted
a 90-second grouped Metal System Trace with the local Qwen3.8 checkpoint,
prompt 256, decode 128, adaptive draft disabled, and mirror verification
disabled. Instruments refused to start because
`/private/tmp/higgs-metal/qwen38-grouped.trace` already existed:

```text
Trace file already exists at path: /private/tmp/higgs-metal/qwen38-grouped.trace.
Specify append-run option to append a run to it.
```

The attempt was not retried and no stock trace was started. The existing trace
is only 96 bytes; its accompanying prior benchmark log ran unrelated ignored
tests and is not usable evidence for the requested pair. TOC export in the
sandbox additionally failed because Instruments could not create
`~/Library/Caches/com.apple.dt.InstrumentsCLI/path_manager` (Cocoa error 513 /
POSIX `Operation not permitted`).

Consequently the workload remains **indeterminate**: Task 2's exact external
benchmark establishes a 19.95% grouped end-to-end throughput improvement over
stock, but it cannot distinguish kernel duration, CPU launch gaps, or draft
synchronization. No Tasks 4--6 optimization should claim a GPU/kernel cause
until valid grouped and stock traces are captured and exported. Full evidence
is in `.superpowers/sdd/2026-08-18-qwen38-low-power-optimization/task-3-report.md`.

## Task 2 verifier dispatch comparison (2026-08-18)

Task 1 selected `HIGGS_MTP_DRAFT_N_MAX=5`, which produces verifier width
`T=6`. Three fresh-process trials were run for each condition against
`/Users/peppi/AI-Models/qwen38-higgs` with `BENCH_PROMPT_LEN=256`,
`BENCH_DECODE_STEPS=128`, `HIGGS_MTP_ADAPTIVE_DRAFT=0`, and
`HIGGS_MTP_MIRROR_VERIFY=0`. Grouped used `HIGGS_CROSSROW_QMV=1`; stock used
`HIGGS_CROSSROW_QMV=0`; QGEMM used `HIGGS_CROSSROW_QMV=0` and
`HIGGS_QGEMM_VERIFY=1`. All nine external processes exited 0.

Complete logs are retained at `/private/tmp/higgs-qwen38-sweep/` as
`dispatch-{grouped,stock,qgemm}-run-{1,2,3}.log`. The first sandboxed grouped
attempt is preserved separately as `dispatch-grouped-sandbox-abort.log`; it
aborted during MLX Metal initialization with
`NSRangeException` from an empty Metal device array. The external run was
required for Metal access, as in Task 1.

Raw summaries:

| Condition | Run | Cycles | Rows | Drafted | Accepted | Measured count/digest | Whole count/digest | Total ms | Avg ms | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| grouped | 1 | 29 | 174 | 145 | 103 | 132 / `35c68639d86d7d4e` | 133 / `0fd2400ecb5dfe42` | 13721.673 | 473.161 | 9.62 |
| grouped | 2 | 29 | 174 | 145 | 103 | 132 / `35c68639d86d7d4e` | 133 / `0fd2400ecb5dfe42` | 13751.803 | 474.200 | 9.60 |
| grouped | 3 | 29 | 174 | 145 | 103 | 132 / `35c68639d86d7d4e` | 133 / `0fd2400ecb5dfe42` | 13701.233 | 472.456 | 9.63 |
| stock | 1 | 29 | 174 | 145 | 103 | 132 / `35c68639d86d7d4e` | 133 / `0fd2400ecb5dfe42` | 16336.401 | 563.324 | 8.08 |
| stock | 2 | 29 | 174 | 145 | 103 | 132 / `35c68639d86d7d4e` | 133 / `0fd2400ecb5dfe42` | 16460.322 | 567.597 | 8.02 |
| stock | 3 | 29 | 174 | 145 | 103 | 132 / `35c68639d86d7d4e` | 133 / `0fd2400ecb5dfe42` | 16517.959 | 569.585 | 7.99 |
| QGEMM | 1 | 28 | 168 | 140 | 104 | 132 / `35c68639d86d7d4e` | 133 / `0fd2400ecb5dfe42` | 26349.971 | 941.070 | 5.01 |
| QGEMM | 2 | 28 | 168 | 140 | 104 | 132 / `35c68639d86d7d4e` | 133 / `0fd2400ecb5dfe42` | 26044.451 | 930.159 | 5.07 |
| QGEMM | 3 | 28 | 168 | 140 | 104 | 132 / `35c68639d86d7d4e` | 133 / `0fd2400ecb5dfe42` | 26159.381 | 934.264 | 5.05 |

Exactness was equal across all three conditions in every trial: measured
`132 / 35c68639d86d7d4e`, whole trajectory `133 / 0fd2400ecb5dfe42`,
verifier rows/drafted/accepted `174/145/103` for grouped and stock, and
`168/140/104` for QGEMM. The QGEMM gate was shape-eligible at `T=6`, but
dispatch status is indeterminate: the implementation silently falls through to
stock when the kernel call returns an error, and the logs provide no dispatch
telemetry. The distinct, slower timing is retained only as an observed exact
QGEMM condition, not as evidence of kernel acceptance or a fallback speed.

Three-run medians were grouped `13721.673 ms / 473.161 ms / 9.62 tok/s`,
stock `16460.322 ms / 567.597 ms / 8.02 tok/s`, and QGEMM `26159.381 ms /
934.264 ms / 5.05 tok/s`. Grouped was 19.95% faster than stock by median
throughput and 16.63% lower in median total cycle time. QGEMM is rejected
pending dispatch telemetry and remains out of the trace candidates; its
observed timing was 37.0% slower than stock but is not treated as a kernel
speed result.

Trace-condition selection: retain the exact grouped/stock pair at selected
depth 5 (`T=6`) for Metal tracing, with grouped as the faster condition. Do not
add QGEMM as a third trace condition. The Task 1 depth matrix remains the
dispatch guard for `T=2..9`: eligible exact grouped/stock pairs were depths
1, 2, 4, 5, 7, and 8; depths 3 and 6 were excluded for count/trajectory
divergence before speed ranking.

## Fix worker update (2026-08-18)

- Replaced the benchmark's two callable `argmax_axis` expressions with the
  existing `mlx_rs::argmax_axis!(..., -1)` convention and removed the function
  import. No production MTP behavior or grouped cross-row code changed.
- Verification: `cargo test -p higgs-engine --release --lib mtp --no-run` and
  `git diff --check` exited successfully. The requested
  `cargo fmt --check crates/higgs-engine/src/mtp.rs` exited 2 because this Cargo
  version rejects a positional path; the file-scoped equivalent
  `rustfmt --edition 2024 --check crates/higgs-engine/src/mtp.rs` exited 0.
- Amended benchmark commit: `e79459a48` (`test(mtp): benchmark production cycles`).

## Fix round 1 (2026-08-18)

- Added one fixed, deterministic, unmeasured production `mtp_cycle` warm-up. The
  benchmark reports `warmup_cycles=1` and the warm-up cycle's configured draft
  depth, verifier rows, drafted, accepted, and emitted counts. Backbone cache,
  MTP cache, and returned hidden state are force-evaluated before the warm-up
  completes and inside every measured timer, so deferred cycle-end work is not
  shifted out of the measured cycle.
- Accumulated every measured emitted token ID and added a stable FNV-1a 64-bit
  digest to the summary as `emitted_digest_fnv1a64`, alongside the existing
  emitted count. Stock and grouped runs can now compare exact greedy output by
  count and digest without changing decode semantics.
- Split explicit and default model-path handling. A missing path supplied by
  `HIGGS_MODEL_PATH` now fails with a clear error; an absent environment variable
  still permits the conventional missing-default-path skip for this ignored
  benchmark.
- Added deterministic unit coverage for explicit/default path resolution and
  the emitted-token digest. No production MTP behavior or grouped cross-row code
  changed. The real-model benchmark was subsequently run in the 2026-08-18 AB
  recorded above.

### Fix round 1 verification

~~~text
$ cargo test -p higgs-engine --lib mtp::tests:: -- --nocapture
test result: ok. 14 passed; 0 failed; 1 ignored; 0 measured; 534 filtered out

$ cargo test -p higgs-engine --release --lib mtp --no-run
Finished `release` profile [optimized] target(s) in 1m 12s
Executable unittests src/lib.rs (target/release/deps/higgs_engine-6ae3dd2051b08dd8)

$ rustfmt --edition 2024 --check crates/higgs-engine/src/mtp.rs
exit 0

$ git diff --check
exit 0
~~~

## Final-review fix wave (2026-08-18)

- Added benchmark-only token accounting that records warm-up `result.tokens`
  before the measured loop, then records measured tokens in both the measured
  suffix and whole trajectory. The summary now reports
  `measured_emitted`/`measured_digest_fnv1a64` and
  `whole_trajectory_emitted`/`whole_trajectory_digest_fnv1a64` separately.
  Cycle elapsed time is still captured before measured token accounting, so the
  existing timing boundary and production MTP semantics are unchanged.
- Added deterministic coverage proving that warm-up tokens affect the
  whole-trajectory count/digest but not the measured count/digest. The focused
  MTP test run passed 15 tests with the real-model benchmark still ignored.
- Updated this report to record the actual 2026-08-18 AB, its exact grouped-ON
  and stock-OFF environment settings, and the original measured-suffix scope of
  its parity evidence. Removed duplicate unchecked Tasks 4 and 5 from the local
  SDD progress ledger.

### Final-review verification

~~~text
$ cargo test -p higgs-engine --lib mtp::tests:: -- --nocapture
test result: ok. 15 passed; 0 failed; 1 ignored; 0 measured; 534 filtered out

$ cargo test -p higgs-engine --release --lib mtp --no-run
Finished `release` profile [optimized] target(s) in 1m 11s
Executable unittests src/lib.rs (target/release/deps/higgs_engine-6ae3dd2051b08dd8)

$ rustfmt --edition 2024 --check crates/higgs-engine/src/mtp.rs
exit 0

$ git diff --check
exit 0
~~~
