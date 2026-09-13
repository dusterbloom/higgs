# Prefill investigation

See [measured results and next experiments](RESULTS.md).

This directory contains the CPU oracle for the row-local, indexless mean
correction described in `docs/superpowers/specs/2026-09-02-indexless-metal-prefill-design.md`.
It does not alter serving policy. This is a mean-correction screening tool,
not the design's full approximation comparison or end-to-end quality gate.

## Capture format

Set `HIGGS_PREFILL_TRACE_DIR` while running a chunked Qwen3-Next prefill. The
diagnostic writes at most one file for each selected `(layer, query_offset)`.
For a 40-layer checkpoint with full attention every four layers, the selected
layers are 3, 19, and 39. The default offsets are 1024 and 15360; override them
with a comma-separated `HIGGS_PREFILL_TRACE_OFFSETS` value. A requested offset
must equal the start of a prefill chunk.

Each safetensors file contains:

- `q`: 16 or fewer sampled post-normalization/post-RoPE queries `[R,Hq,D]`;
- `k`, `v`: the full causally available cache prefix `[Hkv,K,D]`;
- `query_positions`: absolute zero-based positions `[R]`;
- `layer`, `query_offset`, and `scale`: scalar tensors;
- `source_dtype_codes`: Q/K/V source types before FP32 trace conversion, where
  1 is BF16, 2 is FP16, 3 is FP32, and 0 is another type.
- `source_shapes_qkv`: original Q, cache-K, and cache-V four-dimensional shapes
  concatenated before sampling/conversion.

Rows include both sides of 128-token boundaries near the beginning, middle,
and tail of the chunk. Existing files are not overwritten. With the environment
variable absent, no arrays are materialized or written.

Start the normal server with the diagnostic environment after the active
hardware campaign permits model work, then submit the fixed heterogeneous
long-context payload used by the baseline campaign:

```bash
mkdir -p /tmp/higgs-prefill-traces
HIGGS_PREFILL_TRACE_DIR=/tmp/higgs-prefill-traces \
HIGGS_PREFILL_TRACE_OFFSETS=1024,15360 \
target/release/higgs --config /path/to/isolated-config.toml serve
```

Build the release binary first. Use a private localhost configuration pointing
to the intended checkpoint, with `mlx_profile = "throughput"`, disabled disk
prefix caching and `[local] raise_wired_limit = true`, as in the corrected
[baseline](../ane_prefill/BASELINE.md). Run no other inference workload alongside
the capture. Do not time this diagnostic path as a performance candidate.

The request must exceed 15,360 tokens and use prefill chunks aligned to both
requested offsets. Verify that the server produced six files before running
the evaluator. The trace records attention geometry only; quality conclusions
must use a natural, heterogeneous prompt rather than generated token IDs.

Run the CPU correctness test and evaluator with:

```bash
PYTHONPATH=benchmarks/prefill_investigation \
python3 -m unittest benchmarks/prefill_investigation/test_evaluator.py

python3 benchmarks/prefill_investigation/evaluator.py \
  /tmp/higgs-prefill-traces --alpha 0,0.01,0.03,0.1,0.3,0.5
```

The evaluator runs the dense causal parity control before emitting any requested
alpha, including when `--alpha` omits zero or lists it last. Other rows report
selected-token density, selected block count, relative-L2 percentiles/max,
minimum cosine, per-query/head selected-density p50/p95/p99/max, and the worst
row/head coordinate, already stratified by true model layer and absolute query
offset.
