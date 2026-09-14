# GDN prefill launch probe

Measured campaign: [M4 prefill investigation](../prefill_investigation/RESULTS.md).

This benchmark isolates the exact fused GDN recurrence from projections,
convolution, normalization, output gating, output projection, and MoE. It uses
the Qwen3.5 35B production geometry (`B=1, Hk=16, Hv=32, Dk=Dv=128`), BF16
Q/K/V/a/b inputs, and FP32 recurrent state. The Metal arithmetic and serial
time loop are copied from `qwen3_next.rs` at commit
`8c6b5f66ba985b0c823ccca82af6f32d73da5ea4`.

The four variants change only threadgroup Y (`32x1`, `32x2`, `32x4`, `32x8`).
They test launch grouping/occupancy without changing arithmetic. They are not a
parallel scan or a prediction of its speedup.

The temporal-tile candidate is opt-in. It adds a compiler unroll hint to the
existing serial time loop while preserving the arithmetic and state layout.
Set `HIGGS_BENCH_GDN_TILED_T4=1` only when exercising the Rust model path; the
standalone probe reports plain and tiled variants together.

```bash
python3 -m unittest discover -s benchmarks/gdn_prefill -p 'test_*.py'

benchmarks/prefill_investigation/run_safe.sh \
  /opt/homebrew/bin/python3 \
  benchmarks/gdn_prefill/benchmark.py --check
```

Use `--dtype float32` when a capture shows FP32 recurrence inputs; the default
remains BF16 to reproduce the original kernel campaign.

Corrected paired reruns (after obtaining the serialized hardware slot):

```bash
benchmarks/prefill_investigation/run_safe.sh \
  /opt/homebrew/bin/python3 benchmarks/gdn_prefill/benchmark.py \
  --dtype bfloat16 --lengths 1024 --check --repeats 9 --paired-repeats 15
benchmarks/prefill_investigation/run_safe.sh \
  /opt/homebrew/bin/python3 benchmarks/gdn_prefill/benchmark.py \
  --dtype float32 --lengths 1024 --check --repeats 9 --paired-repeats 15
```

Each paired row records both its variant label and `actual_threadgroup`. Results
created before this field existed have invalid paired timings; their sequential
screens and output/state parity remain usable.

The JSON output reports synchronized wall latency. Run on the same powered,
thermally stable machine. `--check` compares an eight-token prefix against a
NumPy sequential reference, including every output and final FP32 state.
The recorded campaign provenance and raw-result caveats live in
`target/prefill-investigation/gdn/PROVENANCE.md`. Flat `vm.swapusage` alone is
not evidence of zero new swapouts; a promotion run must capture cumulative
`vm_stat` swapout counters at both endpoints. The `run_safe.sh` wrapper records
those endpoints and refuses to start below 30% free memory or 700,000
reclaimable VM pages (`HIGGS_BENCH_MIN_FREE_PCT` and
`HIGGS_BENCH_MIN_RECLAIMABLE_PAGES` change the thresholds). The Python probe
records the same preflight in its JSON output. Set
`HIGGS_BENCH_ALLOW_LOW_MEMORY=1` only for an intentional, supervised override.

Projection cost remains a separate measurement: reuse
`benchmarks/ane_prefill/gpu_projection.py` for the QKVZ projection. This probe
only bounds the recurrence share of the observed whole-GDN layer latency.

ReplaySSM feasibility probe
---------------------------

`replayssm_probe.py` checks the exact affine form of the GDN state update before
attempting a Metal implementation. It compares the serial recurrence with a
dense composition of `S_t = A_t S_{t-1} + c_t` at the production geometry:

```bash
benchmarks/prefill_investigation/run_safe.sh \
  python3 benchmarks/gdn_prefill/replayssm_probe.py --lengths 1 2 4 --repeats 5
```

The L=1 case is bit-identical. At L=2 and L=4, dense composition is roughly
73--83x slower than serial and produces state differences up to `2.2e-4` in
float32, so it cannot replace exact tape replay. A low-rank/WY formulation
would need to avoid materializing the 128x128 transition matrices and preserve
the serial operation order near token-acceptance ties. No production path is
enabled by this probe.

The existing guarded Metal tape benchmark (`accept=10/16`) measured replay at
`1.4247 ms` versus `0.7025 ms` for an SSM-only forward (`2.03x` slower). Its
whole-model value comes from skipping projections and other forward work, not
from the replay kernel itself.
