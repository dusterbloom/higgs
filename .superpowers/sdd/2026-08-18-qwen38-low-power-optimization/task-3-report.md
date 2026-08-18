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
| prompt / decode | `256` / `128` |
| draft depth | `HIGGS_MTP_DRAFT_N_MAX=5` (verifier `T=6`) |
| adaptive draft | `HIGGS_MTP_ADAPTIVE_DRAFT=0` |
| mirror verify | `HIGGS_MTP_MIRROR_VERIFY=0` |
| grouped | `HIGGS_CROSSROW_QMV=1` |
| stock | `HIGGS_CROSSROW_QMV=0` |

The existing release test executable was verified to contain
`mtp::tests::bench_production_mtp_cycle_real_model`; no build was performed.

## Capture attempt and artifacts

The grouped capture was launched outside the sandbox using the Metal System
Trace template, the pinned grouped environment above, and a `90s` time limit.
Instruments did not start a recording because the requested output path already
existed:

```text
Trace file already exists at path: /private/tmp/higgs-metal/qwen38-grouped.trace.
Specify append-run option to append a run to it.
```

That exact message is preserved in
`/private/tmp/higgs-metal/qwen38-grouped.log`. Per the bounded-attempt rule,
the command was not retried and the stock capture was not started.

The pre-existing `/private/tmp/higgs-metal/qwen38-grouped.trace` was 96 bytes,
so it is not a usable trace artifact. Its accompanying pre-existing
`qwen38-grouped-benchmark.log` shows that an earlier invocation ran five
ignored tests and was still running after 60 seconds; it did not establish the
single exact production-MTP benchmark. It is retained but excluded from
analysis.

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
