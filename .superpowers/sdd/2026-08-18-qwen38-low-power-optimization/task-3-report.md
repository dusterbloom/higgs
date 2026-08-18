# Task 3 report: bounded Metal trace attempt

## Status

**Blocked / no trace classification.** No production code was modified. The
requested Metal System Trace pair could not be captured in this bounded attempt,
so this task does not classify the verifier as kernel-bound,
host/configuration-bound, or draft-synchronization-bound. Tasks 4--6 must not
treat this report as evidence for a GPU/kernel optimization.

## Intended trace pair

The exact Task 2 comparison was selected:

| Setting | Value |
| --- | --- |
| model | `/Users/peppi/AI-Models/qwen38-higgs` |
| prompt length | `BENCH_PROMPT_LEN=256` |
| decode steps | `BENCH_DECODE_STEPS=128` |
| draft depth | `HIGGS_MTP_DRAFT_N_MAX=5` (verifier `T=6`) |
| adaptive draft | `HIGGS_MTP_ADAPTIVE_DRAFT=0` |
| mirror verify | `HIGGS_MTP_MIRROR_VERIFY=0` |
| grouped | `HIGGS_CROSSROW_QMV=1` |
| stock | `HIGGS_CROSSROW_QMV=0` |

The existing release test executable was verified to contain
`mtp::tests::bench_production_mtp_cycle_real_model`; no build was performed.

## Capture attempt and artifacts

The grouped capture was launched outside the sandbox using the Metal System
Trace template, the pinned grouped environment above, and `--time-limit 10m`.
The supplied log,
`/private/tmp/higgs-metal/qwen38-grouped-20260818T005804Z.log`, records the
single exact production-MTP benchmark output (one passed test) and ends with
the successful benchmark run output. It does not contain the existing-output-
path error. That error belongs only to
`/private/tmp/higgs-metal/qwen38-grouped.log`, which records that Instruments
did not start a recording because the requested output path already existed:

```text
Trace file already exists at path: /private/tmp/higgs-metal/qwen38-grouped.trace.
Specify append-run option to append a run to it.
```

Per the bounded-attempt rule, the command was not retried and the stock capture
was not started.

The requested pre-existing output path
`/private/tmp/higgs-metal/qwen38-grouped.trace` is a 96-byte file. The attempt
also left `/private/tmp/higgs-metal/qwen38-grouped-20260818T005804Z.trace`,
whose 320-byte directory entry corresponds to a ~307 MiB bundle:
`du -sh /private/tmp/higgs-metal/qwen38-grouped-20260818T005804Z.trace` reports
`307M`. It is not a 320-byte bundle. Neither artifact is a valid Metal trace
and neither is used for classification. The earlier `qwen38-grouped-benchmark.log`
ran five ignored tests and was still running after 60 seconds; it is unrelated
to the single exact production-MTP output in the timestamped benchmark log.

An attempt to export the pre-existing trace table of contents also failed in
the sandbox before trace analysis:

```text
Cannot create temporary directory for Instruments Analysis Core: Error
Domain=NSCocoaErrorDomain Code=513 "You don’t have permission to save the file
“path_manager” in the folder “com.apple.dt.InstrumentsCLI”."
NSFilePath=/Users/peppi/Library/Caches/com.apple.dt.InstrumentsCLI/path_manager
NSUnderlyingError=NSPOSIXErrorDomain Code=1 "Operation not permitted"
```

No export retry was made because the trace is incomplete and the capture start
had already failed. No `.xml` table of contents exists.

## Conservative result

Task 2 remains valid wall-time evidence only: for three fresh external runs,
the exact grouped and stock trajectories matched, while grouped had median
`13,721.673 ms`, `473.161 ms` per cycle, and `9.62 tok/s`; stock had median
`16,460.322 ms`, `567.597 ms` per cycle, and `8.02 tok/s`. That is a 19.95%
throughput improvement for grouped.

These data show an end-to-end improvement but do **not** expose encoder
durations, CPU launch gaps, queue overlap, GPU counters, or per-kernel time.
Therefore the dominant-cost class is **indeterminate (no valid Metal trace)**;
there is no evidence to select a kernel, host/configuration, or draft-sync
optimization.

## Concerns and next capture requirements

- Preserve or choose fresh, unique output paths before invoking `xctrace`; do
  not overwrite the existing artifact without explicit authorization.
- Pass the test filter directly to the binary:
  `mtp::tests::bench_production_mtp_cycle_real_model --ignored --exact
  --nocapture`. The pre-existing benchmark log demonstrates that omitting the
  filter runs unrelated ignored tests.
- Run `xctrace export` where Instruments may write its cache directory, then
  export both trace TOCs before drawing GPU or CPU-launch conclusions.

## Fix verification

- Rechecked this report against the Task 3 brief and
  `/private/tmp/higgs-metal/qwen38-grouped-20260818T005804Z.log`: the attempted
  command used `--time-limit 10m`, with `BENCH_PROMPT_LEN=256`,
  `BENCH_DECODE_STEPS=128`, `HIGGS_MTP_DRAFT_N_MAX=5`,
  `HIGGS_MTP_ADAPTIVE_DRAFT=0`, `HIGGS_MTP_MIRROR_VERIFY=0`, and grouped
  `HIGGS_CROSSROW_QMV=1` (stock comparison `HIGGS_CROSSROW_QMV=0`).
- Restored the unrelated performance-port Task 6 report exactly to
  `1dd909aab`; Task 6 will aggregate Task 3 evidence later.
- `git diff --check` exited 0.

## Fix verification (round 2)

- Confirmed the timestamped trace is documented as a ~307 MiB bundle, with its
  320-byte directory entry distinguished from the `du -sh` result of `307M`.
- Confirmed the existing-output-path error is attributed only to
  `/private/tmp/higgs-metal/qwen38-grouped.log`; the timestamped benchmark log
  is documented as ending with the successful run output.
- Preserved the exact `10m`/environment names, the no-trace/no-classification
  conclusion, and the Task 6 restoration.
- `git diff --check` exited 0.
