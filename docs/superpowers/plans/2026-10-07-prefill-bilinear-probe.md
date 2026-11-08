# Bonsai prefill bilinear feasibility probe

User authorization: “go for prefill first” (2026-10-07).

Use the committed September 20 base-M4/32GB baseline: stock Q2, 1024-token
chunks, 4096-token TTFT 81.774s. Gate/up and down were attributed 59.6% of
prefill time. The old tiled kernel lost; do not repeat that experiment.

The October 2 nine-fourths manuscript does not specify a finite rank
decomposition to implement. One/two-level Strassen is an executable control
for the costs of bilinear arithmetic savings, **not** the new algorithm.

1. Add one standalone example; leave production dispatch unchanged. GitNexus
   impact on the shared `quantized_matmul` wrapper is CRITICAL (32 symbols,
   seven processes). Impact on the new example is UNKNOWN/not found;
   source search confirms it does not yet exist or have callers. The index
   is eight commits stale. No existing functions are edited.
2. First verify the control against an independent small f64 CPU oracle.
3. Extract layer-0 gate/up and down weights from the already cached Bonsai
   snapshot, without loading the entire checkpoint or changing it.
4. Compare stock native FP32 QMM, FP16-cast stock QMM, transient FP16 dense
   GEMM, and transient one/two-level Strassen at M=512/1024. Include input
   cast, weight dequantization, sums, allocations, and forced evaluation.
   Prepared dense-weight paths are optimistic controls only; report their
   persistent memory separately. Inputs are deterministic synthetic
   activations; weights and scales are real. This is not model quality proof.
5. Warm each path, rotate order over seven rounds, save every timing,
   median/range, MLX active/peak memory, and error against native stock.
   Run timing only without the currently active Metal training process.
6. Use the existing 1.2x operator gate at both M values to decide whether
   model integration is worth considering. Cast error and arithmetic error
   must be reported separately. No model integration on an operator result
   alone: end-to-end TTFT, real logits/behavior, and loaded-model memory are
   additional gates.
7. Save a reproducible result report, including environment and limitations.
   No production change or commit is required for this feasibility probe.
