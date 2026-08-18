# Task 2 implementation report: Qwen3.8 verifier dispatch comparison

## Status

Completed. No production code was modified. Task 1 selected draft depth 5;
this task compared grouped cross-row, stock, and the existing QGEMM verifier
gate at verifier width `T=6`.

## Benchmark matrix

Model: `/Users/peppi/AI-Models/qwen38-higgs`
`BENCH_PROMPT_LEN=256`
`BENCH_DECODE_STEPS=128`
`HIGGS_MTP_DRAFT_N_MAX=5`
`HIGGS_MTP_ADAPTIVE_DRAFT=0`
`HIGGS_MTP_MIRROR_VERIFY=0`

Each condition used three fresh processes:

| Condition | Environment delta |
| --- | --- |
| grouped | `HIGGS_CROSSROW_QMV=1` |
| stock | `HIGGS_CROSSROW_QMV=0` |
| QGEMM | `HIGGS_CROSSROW_QMV=0 HIGGS_QGEMM_VERIFY=1` |

All nine external benchmark processes exited 0. Complete logs:
`/private/tmp/higgs-qwen38-sweep/dispatch-grouped-run-{1,2,3}.log`,
`dispatch-stock-run-{1,2,3}.log`, and `dispatch-qgemm-run-{1,2,3}.log`.

The initial sandboxed grouped process aborted before model execution during MLX
Metal initialization:

```text
NSRangeException: -[__NSArray0 objectAtIndex:]: index 0 beyond bounds for empty array
mlx::core::metal::DeviceC2Ev
signal: 6, SIGABRT
```

That failed attempt is preserved separately in
`dispatch-grouped-sandbox-abort.log`; `dispatch-grouped-run-1.log` is the
successful external rerun. External execution was used for the full matrix, as
required by the Metal environment.

## Raw summaries

| Condition | Run | Cycles | Verifier rows | Drafted | Accepted | Measured count/digest | Whole count/digest | Total ms | Avg ms | Tok/s |
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

## Exact comparisons and decision

Grouped and stock matched exactly on every digest/count field in every trial:

- verifier rows / drafted / accepted: `174 / 145 / 103`;
- measured trajectory: `132 / 35c68639d86d7d4e`;
- whole trajectory: `133 / 0fd2400ecb5dfe42`.

QGEMM also matched the exact measured and whole digests/counts, with
`168 / 140 / 104` verifier rows/drafted/accepted in each run. Its gate was
shape-eligible, but dispatch status is indeterminate: the code silently falls
through to stock if the kernel call errors, and no dispatch telemetry is
emitted. The QGEMM rows below therefore record only the observed exact output
and timing condition; they do not establish kernel acceptance or a fallback
speed.

Median summaries:

| Condition | Median total ms | Median avg ms | Median tok/s |
| --- | ---: | ---: | ---: |
| grouped | 13721.673 | 473.161 | 9.62 |
| stock | 16460.322 | 567.597 | 8.02 |
| QGEMM | 26159.381 | 934.264 | 5.05 |

Grouped is the fastest exact grouped/stock condition: 19.95% higher median
throughput than stock. QGEMM is rejected pending dispatch telemetry and remains
out of the trace candidates. Its observed timing was 37.0% slower than stock,
but that is not treated as a kernel speed result. The trace pair is grouped and stock at depth 5
(`T=6`), with grouped selected as the faster condition. Prior Task 1 results
remain the dispatch guard across `T=2..9`: depths 1, 2, 4, 5, 7, and 8 were
exact eligible pairs; depths 3 and 6 were excluded before ranking.

## Concerns

The MLX Metal sandbox cannot initialize the device, so external execution is
required for reproducible GPU timings. QGEMM has silent error fallback in the
implementation and no explicit dispatch telemetry; these runs had no error
output, exact parity, and a distinct slower timing profile, but the logs alone
cannot independently prove kernel-level acceptance. QGEMM is therefore
indeterminate/rejected pending telemetry.

## Fix round 1 (2026-08-18)

Relabeled QGEMM dispatch status as indeterminate/rejected pending telemetry.
The exact QGEMM output/count fields, observed timings, grouped/stock arithmetic,
and grouped/stock trace selection are unchanged. No production code was
modified.

Verification:

```text
$ git diff --check
exit 0

$ rg -n 'indeterminate|rejected pending telemetry' task-2-report.md task-6-report.md
task-6-report.md:264:dispatch status is indeterminate: the implementation silently falls through to
task-2-report.md:68:shape-eligible, but dispatch status is indeterminate: the code silently falls
task-2-report.md:85:throughput than stock. QGEMM is rejected pending dispatch telemetry and remains
task-2-report.md:97:indeterminate/rejected pending telemetry.
```
