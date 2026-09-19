# Escha Performance Scorecard Design

## Goal

Make speculative-decode measurements for a local Qwen3.8/Escha checkpoint reproducible and reject a result when greedy visible output differs from the AR baseline.

## Scope

Extend `bench_speculative` only. It already launches fresh servers for each mode, accepts `--model-path`, forces greedy sampling, and collects MTP completion telemetry. The benchmark will retain each trial's complete visible output in memory, compare it with the baseline trial of the same repeat, and persist an explicit parity result with the existing timing and telemetry. It accepts an optional `--api-key` that it passes only to its fresh server and HTTP client; it never appears in the persisted parameters.

No inference kernel, scheduler, ANE backend, model conversion, benchmark manifest entry, or new dependency is part of this change.

## Contract

- The baseline remains the `baseline_mtp_off` trial.
- Every non-baseline trial with the same repeat index must have exactly the same visible response content and completion-token count as its baseline.
- A trial with `HIGGS_MTP=1` must emit the MTP completion log record; a checkpoint without MTP weights is not a speculative result.
- A mismatch makes the benchmark return an error after persisting no misleading comparison result.
- The JSON output includes a stable FNV-1a digest and `parity_with_baseline` for each successful trial, so reviewers can verify parity without storing full completions.

## Verification

Unit tests cover a matching response, visible-content and completion-token mismatches, benchmark authentication, and missing MTP completion telemetry. The focused test and release build run without starting Higgs. A later live run, only when the GPU is free, uses `baseline,mtp_default,mtp_adaptive` plus `bench_frontier` and a Metal System Trace.
