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

The follow-up matrix-free low-rank composition avoids dense matrices, but at
this geometry it remains `4.1--5.8x` slower than serial for L=1, 2, and 4 in
the CPU probe. It has the same float32 drift for L>1, so it is not an exact
rollback replacement. A useful Metal implementation would need a genuinely
parallel triangular solve and an explicit policy for near-tie token changes.

For replay specifically, the fixed innovation tape reduces the update to
`s_t = g_t s_{t-1} + k_t delta_t`. The probe's prefix-product scan reproduces
serial replay within `9.3e-10` at L=1, 2, and 4. Its NumPy implementation is
`8--11x` slower because it is not parallelized, but it preserves the exact
replay algebra and is the candidate for the next guarded Metal scan prototype.

The guarded Metal prototype is in `replay_scan_benchmark.py` and uses both a
128x1 lane layout and the production-style packed 32x4 layout. The packed scan
was `0.93x` serial at BF16 L=2 but `1.38x` at L=4; FP32 was `1.08x` and `1.00x`
respectively. State error stayed below `3.8e-9`, and the input checkpoint was
unchanged, but there is no consistent speed win. The isolated kernel has no
LM-head logits, so accepted-token decisions are explicitly not measured here;
the exact production tape path remains the fallback.

The follow-up end-to-end reconstruction run tested BF16 L=4 and L=8 with the
packed 32x4 layout. Scan/serial ratios were `1.02x` and `1.11x`; the obvious
32x2 layout iteration was `1.12x` and `1.13x`. These timings include synchronized
input/state traffic, kernel dispatch, and output materialization. State error
remained below `3.8e-9`, but the scan has no repeatable latency win, so it is
not wired into the linear MTP path.

The replay kernel also has a safe decomposition win: its gate is constant for
all lanes in a `(batch, head, timestep)` threadgroup. Sharing that scalar with
threadgroup memory preserves bitwise state output and reduced the packed BF16
replay probe to `0.923x` of the old layout at L=4 and `0.929x` at L=8 (about
7--8% faster). The production replay kernel now uses this form. These are
isolated rollback-kernel timings; whole-model MTP speedup still needs a paired
partial-rejection trace.

Hybrid rollback checkpoint
--------------------------

`AnyCache::checkpoint_for_rollback_light` copies only GDN recurrent state and
records KV offsets; rollback trims KV storage in place and restores the copied
SSM state. At the Qwen3.6-35B-A3B geometry (30 GDN layers, 10 full-attention
layers, BF16 KV, FP32 SSM), the guarded synthetic benchmark measured:

* 2,048 resident tokens: `106.3 MB` full clone versus `64.4 MB` recurrent copy;
  `26.0 ms` versus `16.2 ms`.
* 8,192 resident tokens: `232.2 MB` full clone versus `64.4 MB` recurrent copy;
  `34.5 ms` versus `18.7 ms`.

The checkpoint path is covered by an exact restore test and is now used by the
generic MTP and prompt-lookup rollback helpers. It is separate from the
DFlash `replay_tape_rollback` path, which already restores GDN state from its
per-layer transaction tape. These numbers are cache-transaction measurements,
not whole-model decode speedups.

The generic path uses the lightweight checkpoint by default. Set
`HIGGS_MTP_FULL_CHECKPOINT=1` to force the previous full-clone transaction for
compatibility comparisons.

Generic MTP also has a guarded `HIGGS_MTP_TAPE_VERIFY=1` seam that returns the
first verify's hidden rows, logits, taps, and GDN transaction data together.
On partial rejection it repairs the Hybrid cache from that transaction and
slices the already-computed outputs instead of launching a second backbone
verify; unset the flag to retain the original full-reverify fallback. The seam
is covered by a deterministic Qwen3Next model-level parity test. A real MTP
cycle comparison still needs a loader-supported MTP fixture.

A recorded decay/gate tape was also prototyped in the standalone packed replay
kernel. Loading one precomputed gate per threadgroup was `0.826x` the shared
recompute time in one BF16 L=8 run, but the host-side gate differed by up to
`2e-6` because it did not reuse the forward kernel's exact operation sequence.
Shipping that variant would require extending the forward tape to emit the
Metal-computed gate and adding a second replay-kernel ABI; the exact shared-gate
kernel remains the production path until that contract is implemented.
