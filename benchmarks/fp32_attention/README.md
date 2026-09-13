# Exact FP32/D256 Steel attention experiment

This standalone harness reuses Apple's MLX Steel attention implementation from the
local build tree. It does not edit vendor code or change serving behavior. Inputs,
SIMDgroup matrix operands, accumulation, and output are FP32. The only arithmetic
transformation changes `fast::exp2` to `metal::exp2`; online softmax and normal
floating-point reassociation remain. Exact here means no deliberate precision
reduction or pruning, not bitwise equivalence.

The extracted headers include Apple's MIT license and copyright comments. Every
source input is recorded by SHA256 in each result. `--mlx-source` defaults to the
specific local MLX source path in the script; it must point to a compatible MLX tree.

```sh
python3 benchmarks/fp32_attention/test_steel.py
python3 benchmarks/fp32_attention/steel.py --mode prepare
python3 benchmarks/fp32_attention/steel.py --mode correctness --output target/fp32-attention/steel-correctness.json
python3 benchmarks/fp32_attention/steel.py --mode traces --output target/fp32-attention/steel-traces.json
python3 benchmarks/fp32_attention/steel.py --mode timing --key-len 8192 --repeats 8 --output target/fp32-attention/steel-8k.json
python3 benchmarks/fp32_attention/steel.py --mode timing --key-len 16384 --repeats 8 --output target/fp32-attention/steel-16k.json
```

Only `prepare` and CPU tests are safe to run alongside GPU experiments. Other modes
require a serialized GPU slot and unsandboxed Metal access on this machine.

- `q8k16`: BQ8/BK16/BD256/WM1/WN1, 32 threads, 28,800 bytes threadgroup storage.
- `q16k8`: BQ16/BK8/BD256/WM2/WN1, 64 threads, 28,928 bytes threadgroup storage.
- Synthetic checks cover positive causal offsets, GQA, partial Q/K tiles, sampled
  nonconsecutive positions, and finite high-magnitude logits. Both layouts are checked.
- Real captures use their absolute query positions and an explicit Boolean mask;
  sampled queries are never relabeled as consecutive. Captured FP32 arrays may
  originate from lower-precision model activations; this harness does not restore
  information lost before capture.
- Noncontiguous source views are constructed in MLX by transpose/materialize/transpose.
  The kernel consumes explicit contiguous arrays. Kernel-only and fresh-copy-inclusive
  arms are separated, and materialization is measured independently.
- Each timing candidate (including full SDPA) is paired with the evaluated 128-query
  Boolean-mask dense control. Even repeats alternate AB/BA, candidate order is seeded,
  each call constructs a fresh output, and raw records include pair IDs and order.
- CPU correctness uses a float64 row-wise oracle. Large timing checks compare to dense128.
  Gates are max absolute error and max row relative L2, both defaulting to 2e-4.
- Results record actual launch configuration, MLX version, device, source SHA256,
  power source, thermal status, swap counters, and per-arm samples. New case swapouts
  stop the run. Exceptions retain top-level endpoints and failure text; abrupt native
  aborts still require the shell log. Timings are exploratory until independent review,
  paired drift analysis, and native/serving validation.

The opt-in second revision removes Q threadgroup staging and caches each lane's
64 FP32 query scalars in registers before the KV loop. `qreg32k16` uses BQ32/WM4
(128 threads), and `qreg64k16` uses BQ64/WM8 (256 threads); both use BK16 and
20,480 bytes KV threadgroup storage. Existing output accumulation already requires
64 FP32 scalars/lane, so register pressure/spilling is an explicit performance risk.
Select with `--variants qreg32k16 qreg64k16`. Original variants remain the defaults.
The transformation refuses changed upstream snippets, keeps the MMA and softmax
operation order, and safe-loads query tails directly from device memory. Synthetic
checks also include Q65/K83 consecutive and sampled positions to cross both new
query tile boundaries. Preparation and CPU tests pass; hardware status is recorded
in the accompanying experiment report, not inferred from these static calculations.
