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

Greedy token selection uses the existing MLX `ops::indexing::argmax_axis` convention. Draft depth is resolved by `MlxRuntimeTuning::from_model_dir(..., RequestedMlxProfile::Auto)`, so `HIGGS_MTP_DRAFT_N_MAX` follows the production parser, default, and 1..=8 clamp. `HIGGS_MODEL_PATH`, `BENCH_PROMPT_LEN`, and `BENCH_DECODE_STEPS` follow the existing ignored benchmark conventions; prompt and decode defaults are 256 and 32.

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

### Whole-trajectory parity rerun after final-review fix

Using the exact grouped-ON and stock-OFF commands above with the same model,
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

The import was corrected from the root macro to the existing function convention, `mlx_rs::ops::indexing::argmax_axis`.

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
