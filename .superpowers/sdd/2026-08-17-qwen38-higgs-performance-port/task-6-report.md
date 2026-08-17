# Task 6 benchmark gap report

## Status

Implemented and compile-verified an ignored real-model benchmark that exercises the production MTP cycle. The real-model benchmark was not run, per the task follow-up instruction.

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

Each cycle prints configured draft depth, verifier batch rows (`T = drafted + 1`), drafted count, accepted count/rate, emitted count, cycle time, and cycle tok/s. The summary prints cycle count, total/min/max/average verifier rows, total drafted/accepted/emitted counts, aggregate acceptance rate, total/average cycle time, and aggregate tok/s. A configured depth of 1..=8 exercises verifier row counts T=2..=9.

No parser or accounting helper was added, so there was no new deterministic helper suitable for a separate TDD unit test. The harness is ignored and normal tests do not load model files.

## Commands and outputs

### Formatting and diff checks

~~~text
$ rustfmt --edition 2024 --check crates/higgs-engine/src/mtp.rs
exit 0

$ git diff --check
exit 0
~~~

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

The benchmark was intentionally not run during this task. Stock and grouped cross-row builds can be measured with identical values using:

~~~bash
HIGGS_MODEL_PATH=/absolute/path/to/local/qwen3.8-model \
BENCH_PROMPT_LEN=256 \
BENCH_DECODE_STEPS=32 \
HIGGS_MTP_DRAFT_N_MAX=8 \
cargo test -p higgs-engine --release --lib \
  mtp::tests::bench_production_mtp_cycle_real_model -- \
  --ignored --exact --nocapture
~~~

Use the same model, prompt length, decode steps, draft depth, and warm process state for both stock and grouped cross-row trials.

## Concerns

None for the bounded harness implementation. Runtime performance numbers and stock-versus-grouped token/timing comparison remain deliberately uncollected because the follow-up explicitly prohibited running the real-model benchmark.

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
  changed. The real-model benchmark was not run.

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
