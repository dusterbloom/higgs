# Attention prefill ceiling probe

Measured campaign: [M4 prefill investigation](../prefill_investigation/RESULTS.md).

This standalone MLX benchmark compares the unchanged feasibility probe from
commit `8b4cdff3ac22aa54241b02e0f0113a0668d37611` with a tiled dense
control: Q is tiled by 128 rows, each tile gets its explicit causal mask and is
evaluated before concatenation. This matches the fallback shape at 16K; at 8K,
the serving path does not activate its >=16K tiling guard, so this is not an
exact 8K serving-dispatch comparison.

Ready hardware commands:

```bash
mkdir -p target/prefill-investigation/attention
/opt/homebrew/bin/python3 benchmarks/attention_prefill/benchmark.py --key-len 8192 \
  > target/prefill-investigation/attention/k8192-fp32.json
/opt/homebrew/bin/python3 benchmarks/attention_prefill/benchmark.py --key-len 16384 \
  > target/prefill-investigation/attention/k16384-fp32.json
```

The 100% density case must satisfy both max absolute error and maximum per-row
relative L2 error at `2e-3` before timing. Lower densities are optimistic speed
ceilings only; they do not implement or measure approximation quality.
