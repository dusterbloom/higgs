# GDN prefill microbenchmark provenance

- Kernel source: `GDN_RECURRENCE_METAL_PREAMBLE` and the plain, non-tape body
  of `GATED_DELTA_FORWARD_KERNEL_SOURCE` in
  `crates/higgs-models/src/qwen3_next.rs`.
- Source commit: `8c6b5f66ba985b0c823ccca82af6f32d73da5ea4`.
- Checkpoint geometry source:
  `/Users/peppi/.cache/lm-studio/models/EschaLabs/Qwen3.6-35B-A3B-Escha-W2/config.json`.
- Runtime: MLX 0.30.6 through `/opt/homebrew/bin/python3`.
- Final screen: `results.json`, with raw sample arrays and alternating paired
  order retained. First successful screen: `attempt-3-success/results.json`.
- Failed launches: `attempt-1-wrong-python/` and `attempt-2-bf16-numpy/`.
- Power endpoints: battery power, 92% before and after.
- Swap evidence: `vm.swapusage` was 2560.94 MiB before and after. New swapout
  status is **unknown** because cumulative `vm_stat` swapout counters were not
  captured at both endpoints. This does not pass a no-new-swapouts gate.
