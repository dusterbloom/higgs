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

```bash
python3 -m unittest discover -s benchmarks/gdn_prefill -p 'test_*.py'

/opt/homebrew/bin/python3 \
  benchmarks/gdn_prefill/benchmark.py --check
```

Use `--dtype float32` when a capture shows FP32 recurrence inputs; the default
remains BF16 to reproduce the original kernel campaign.

Corrected paired reruns (after obtaining the serialized hardware slot):

```bash
/opt/homebrew/bin/python3 benchmarks/gdn_prefill/benchmark.py \
  --dtype bfloat16 --lengths 1024 --check --repeats 9 --paired-repeats 15
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
`vm_stat` swapout counters at both endpoints.

Projection cost remains a separate measurement: reuse
`benchmarks/ane_prefill/gpu_projection.py` for the QKVZ projection. This probe
only bounds the recurrence share of the observed whole-GDN layer latency.
