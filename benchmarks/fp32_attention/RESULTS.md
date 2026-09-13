# Exact FP32/D256 attention: measured outcome

**No serving speedup from this experiment.** Four fused Steel variants passed
numerical checks but all lost to their paired dense controls. The native dense
tile sweep found a modest operator gain at the cost of larger temporary matrices.
Serving code and defaults remain unchanged.

This continues the [prefill bottleneck investigation](../prefill_investigation/RESULTS.md)
on an Apple M4 with 32 GiB memory. That investigation traced native Escha attention
to FP32 Q/K/V with 16 query heads, two KV heads, and head dimension 256. The pinned
MLX full-attention fused dispatcher does not enable D256. These experiments test
that gap without pruning or deliberately reducing arithmetic precision.

## Native dense query tiles

The Rust example uses Q[1,16,1024,256], K/V[1,2,K,256], deterministic nonconstant
FP32 inputs, and a preallocated lower-right causal Boolean mask. Each candidate
has eight adjacent comparisons with tile128, alternating AB/BA and rotating
candidate order. Each call evaluates fresh tile outputs and concatenation.

Median **paired candidate/control latency ratio**, lower is better:

| Query tile | K8192 | K16384 | Nominal one-score-matrix storage at K16384 |
|---:|---:|---:|---:|
| 64 | 1.105 | 0.982 | 64 MiB |
| 128 control | 1.000 | 1.000 | 128 MiB |
| 256 | 1.001 | 1.028 | 256 MiB |
| 512 | 0.951 | 1.000 | 512 MiB |
| 1024 | 0.948 | 0.924 | 1 GiB |

Tile1024 reduced measured operator latency by 5.24% and 7.58%. Its single score
matrix is eight times larger; these calculated sizes are **not measured peak
allocation or physical footprint**. Additional intermediates can coexist. This
does not establish a whole-request gain or justify changing the memory safeguard.
In particular, the existing native path already uses whole-query SDPA below 16K.

All four candidates passed full-output parity against tile128 at both lengths
(absolute and maximum row relative L2 gates 2e-4), plus a hand-computed offset,
GQA mapping and partial-tile fixture. Both runs exited successfully with zero
new swapouts. The final example defaults to eight repeats and rejects odd counts;
the measured source snapshot had a default of nine but both runs explicitly used
eight. The numerical and timing code is unchanged by this final argument guard.

## Fused FP32 Steel variants

The standalone Python harness specializes the pinned local Apple MLX Steel
implementation with FP32 operands, accumulation and output. It replaces
`fast::exp2` with `metal::exp2`, retains Apple's license and hashes every source
input. Normal FP32 reassociation is permitted; bitwise equivalence is not claimed.

| Variant | Query/key tile | Threads | Shared bytes | 8K paired ratio | 16K paired ratio |
|---|---|---:|---:|---:|---:|
| q8k16 | 8/16 | 32 | 28,800 | 6.321 | 5.536 |
| q16k8 | 16/8 | 64 | 28,928 | 2.225 | 1.817 |
| qreg32k16 | 32/16 | 128 | 20,480 | 2.049 | 1.949 |
| qreg64k16 | 64/16 | 256 | 20,480 | 1.401 | 1.467 |

Ratios above use contiguous inputs and eight matched pairs per candidate. Caching
Q fragments in thread-local storage allowed larger query tiles within the shared
memory limit. The best revised kernel still took 40–47% longer than its control.
For actual noncontiguous source views, fresh-copy-inclusive ratios were 1.442/1.430;
pre-copied view ratios were 1.493/1.480. Copy-only medians were about 1.8–3.6 ms.
Copy-inclusive candidates are compared with pre-copied dense controls, so this is
a conservative cost comparison, not a full native strided-layout benchmark.

Original variants passed 10 synthetic cases; revised variants passed 14, including
query/key tails, offsets, GQA, sampled nonconsecutive positions and amplitude-12
finite logits. Each revision passed nine actual model captures in two layouts,
covering layers 3/19/39 and valid key lengths 2048/8192/16320. Worst real-trace
absolute error was 1.3444e-4 and maximum row relative L2 was 1.8929e-5 against an
independent float64 row-wise oracle, within the fixed 2e-4 gates. All timed shapes
also passed parity checks. This is attention-output validation, not a new
whole-model generation-quality evaluation.

All trace and timing runs recorded zero new swapouts. Battery power and substantial
control drift limit small-effect claims: one original view run's dense controls
spanned approximately 72–221 ms. Paired ratios support rejecting these large
regressions; absolute medians across different runs should not be compared as
controlled speedups. The original synthetic run has incomplete swap coverage,
documented in the [detailed Steel report](STEEL_RESULTS.md).

## What the data supports next

1. **Measure the fused kernel's hardware limits before another rewrite.** Register
   allocation/spills, occupancy, barrier frequency and matrix throughput are still
   hypotheses. The shared-memory footprint is known; actual register use is not.
   The next bounded experiment should capture compiler/GPU evidence for dense128,
   q16k8 and qreg64k16 at one matched 16K shape. The earlier System Trace crash means
   this may require a small standalone frame capture rather than a serving trace.
2. **Treat larger dense query tiles as a memory-budgeted experiment.** Repeat the
   small gain with measured peak allocation and then paired whole-request timing
   before considering a runtime shape plan. At long contexts, a 1 GiB score matrix
   alone can erase the benefit through memory pressure.
3. **Keep persistent buffers in proportion to their measured cost.** Eliminating
   these particular copies would recover only a few milliseconds in this operator
   screen, far less than the fused kernel deficit. This does not measure every
   serving allocation or rule out scheduling improvements elsewhere.

The previous instrumented profile assigned about 42% of extrapolated component
time to the full-attention group, including projections, cache and residual work.
It used evaluation barriers; this is not isolated kernel time or an uninstrumented
prefill share. If that share held end to end, halving the entire group would yield
about 1.27× overall speedup by Amdahl's law. This is an illustration under that
assumption, not a forecast for the attention kernel. A much larger exact gain also
requires progress in MLP/GDN, or separately validated approximate attention.

## Reproduction and evidence

See [README](README.md) for Steel commands. Native runs:

```sh
cargo build --release -p higgs-models --example prefill_dense_tuning
cp target/release/mlx.metallib target/release/examples/mlx.metallib
target/release/examples/prefill_dense_tuning 8192 8
target/release/examples/prefill_dense_tuning 16384 8
```

Serialize Metal workloads, use the same library build, and record swap/power
endpoints. Results, logs, source snapshots and hashes are in
[results/2026-09-13](results/2026-09-13/); larger text artifacts use gzip. The
manifest hashes both archived bytes and original bytes. Large QKV tensors remain
under `target/prefill-investigation/capture-heterogeneous/traces`; their provenance
manifest is archived here. Initial sandbox device-discovery and native build
failures are retained alongside successful runs.

Final checks: four CPU contract tests, release example build, rustfmt, independent
measurement review and complete GitNexus all/staged change analysis. Staged
risk was medium with three internal benchmark flows; whole-worktree risk was
critical including pre-existing serving edits. The archived graph snapshots
precede adding the graph evidence files themselves. No production
integration was attempted because the fused candidates did not clear the timing
gate. Existing user changes were preserved byte for byte.
