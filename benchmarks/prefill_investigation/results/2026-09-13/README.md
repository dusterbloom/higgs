# Evidence index

Files larger than 8 KiB are gzip-compressed without content changes. The
artifact manifest records uncompressed bytes and SHA256. Local paths in runners
refer to the measurement machine; read each private config before reproducing.

- `profile45/`: instrumented 45K server, leading-cycle attribution, VM/footprint,
  exact payload/stream, Xcode trace failure. Not a speed candidate.
- `capture-heterogeneous/`, `capture-inputs/`: 16K real QKV capture provenance,
  request, server outcomes and SHA256 manifest for nine local safetensors.
- `sparse-evaluation/`: all nine strata and seven alpha values. Alpha-zero is
  internal NumPy dense-limit parity, not a comparison with MLX output traces.
- `attention/`: scalar traversal versus 128-query tiled dense controls. Reduced
  density is a compute ceiling, without selection/correction or quality parity.
  The 8K control differs from the serving dispatch's >=16K tiling guard.
- `tuning/`: four uninstrumented 32K arms, including failed pre-load attempts.
  The identical 1024 controls drift by 19%; neither candidate establishes a
  speedup. New swapouts during loading settled before the timed intervals.
- `gdn-corrected-bf16/`, `gdn-corrected-fp32/`: fresh paired runs with actual
  threadgroups and source hashes, explicit TG4 oracle, parity assertions and
  cumulative swapout endpoints. These supersede the invalid paired rows below.
- `gdn/`, `gdn-verified/`, `gdn-fp32/`: original sequential screens and parity.
  **All `paired_tg4_vs_candidate` rows in these three directories are invalid
  for launch comparisons: both labels actually used TG8.** The reviewer found
  the shared-kwargs defect after collection. Preserve these attempts as failures;
  corrected runs belong in separate directories. `gdn/` also lacks cumulative
  swapout endpoints; flat swap usage is not a passing swapout gate.

The full model runs import `measure.py` from `benchmarks/ane_prefill` and the
machine's existing `memory_probe.py` from
`/Users/peppi/Dev/nanobot-rs/experiments/context-recovery`. That sampler uses
native physical-footprint accounting; substituting RSS changes the metric.
The large trace bundle, binaries, Metal library and full external article are
not archived in Git. Reproduce captures using the documented environment and
compare the saved manifest rather than assuming local trace files are portable.
