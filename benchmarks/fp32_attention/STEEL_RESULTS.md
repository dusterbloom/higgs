# Exact FP32/D256 Steel attention experiment

All four Steel configurations passed the required numerical checks. All four lost
to dense128 in matched timing. The best configuration, qreg64k16, took about
1.40–1.47 times the paired control latency for contiguous and explicit view-copy
arms (up to 1.49 times for pre-copied view arms).
No native integration or whole-request acceleration claim is justified.

## Source and hardware

Apple M4, 32 GiB memory, Python 3.14, MLX 0.30.6. All measured runs were on battery
power. The wrapper extracts Apple's float SIMDgroup MMA/online-softmax implementation
from the local MLX source tree without editing vendor files. It replaces
`fast::exp2` with `metal::exp2` and specializes parameters. All inputs, operands,
accumulators, and outputs are FP32; no pruning, half, TF32, or deliberate approximation
is introduced. Numerical parity permits normal floating-point reassociation.

The generated headers retain Apple's MIT license and copyright comments. Every
input source has a SHA256 in each result; `steel-library-provenance.json` additionally
records the Python extension, libmlx.dylib, Python version, and harness hashes.
`steel-source-v1.py` preserves the original trace/timing harness; `steel-source-v2.py`
preserves the Q-register revision. Q-register results record their harness SHA256.

| Variant | BQ/BK/BD | WM/WN | Threads | Threadgroup bytes | Q cache scalars/lane |
|---|---|---|---:|---:|---:|
| q8k16 | 8/16/256 | 1/1 | 32 | 28,800 | 0 |
| q16k8 | 16/8/256 | 2/1 | 64 | 28,928 | 0 |
| qreg32k16 | 32/16/256 | 4/1 | 128 | 20,480 | 64 |
| qreg64k16 | 64/16/256 | 8/1 | 256 | 20,480 | 64 |

For Q-register variants, a complete Q MMATile is safe-loaded from device before the
KV loop. Each cached fragment replaces the original Qtile load in unchanged dd
order. Removing Q_smem permits larger BQ, while TQ remains 1 and BQ=8*WM*WN. Each
lane logically holds 64 additional FP32 Q scalars beside 64 output scalars. Actual
register allocation, spills, and occupancy were not measured.

## Correctness

Gates were fixed at max absolute error <=2e-4 and max row relative L2 <=2e-4 against
an independent float64 row-wise oracle. Large timing fixtures also checked outputs
against dense128 before recording samples.

| Suite | Original variants | Q-register variants |
|---|---:|---:|
| Synthetic fixtures × layouts | 5×2 = 10 | 7×2 = 14 |
| Worst synthetic absolute error | 2.681974e-6 | 2.681974e-6 |
| Worst synthetic row relative L2 | 1.214950e-6 | 1.417746e-6 |
| Real captures × layouts | 9×2 = 18 | 9×2 = 18 |
| Worst real absolute error | 1.344389e-4 | 1.344389e-4 |
| Worst real row relative L2 | 1.892936e-5 | 1.892936e-5 |

Synthetic fixtures use nonconstant random FP32 Q/K/V, four query heads/two KV heads,
positive causal offsets, partial Q/K tiles, nonconsecutive absolute positions, and
amplitude-12 finite logits. Q-register checks add Q65/K83 consecutive and sampled
cases to cross full and partial BQ32/BQ64 blocks. Both contiguous arrays and actual
MLX noncontiguous views are checked, with explicit materialization.

Real captures cover layers 3/19/39 at K2048/8192/16320, with 16 sampled queries and
16 query heads/two KV heads. Their absolute query positions drive Boolean masks;
sampled rows are never relabeled consecutive. Captures are FP32 arrays but may
originate from lower-precision activations; lost upstream precision is not recovered.

All trace cases, Q-register synthetic cases, and timed cases recorded zero new
swapouts. Original synthetic counters bracketed only the post-check region, so they
do not establish whole-run swap behavior. The original sandbox attempt failed at
Metal device discovery; the authorized unsandboxed run succeeded. Both logs remain.

## Timing method and results

Shapes: Q[1,16,1024,256], K/V[1,2,K,256], lower-right causal attention. Controls are
128-query native SDPA tiles with preallocated Boolean masks and full native SDPA.
Each candidate has 8 paired comparisons with dense128, alternating AB/BA with seeded
candidate order. Every sample constructs and evaluates a fresh output; pair IDs and
actual order remain in raw records. Per-case and top-level power/thermal/swap
endpoints are preserved. Exceptions retain failure text and endpoints.

The view cases construct actual noncontiguous MLX inputs. Kernel-only arms use
explicitly materialized inputs; copy-inclusive arms construct fresh contiguous
copies inside timing. Copy-only arms measure materialization independently.

| K | Layout | dense128 ms | full SDPA ms | q8k16 ms | q16k8 ms |
|---:|---|---:|---:|---:|---:|
| 8192 | contiguous | 79.160 | 86.453 | 494.979 | 166.681 |
| 8192 | view, pre-copied | 127.525 | 112.789 | 669.998 | 184.069 |
| 16384 | contiguous | 190.228 | 194.320 | 921.140 | 347.443 |
| 16384 | view, pre-copied | 250.797 | 243.194 | 1499.714 | 333.476 |

| K | Layout | dense128 ms | full SDPA ms | qreg32k16 ms | qreg64k16 ms |
|---:|---|---:|---:|---:|---:|
| 8192 | contiguous | 101.222 | 95.858 | 211.791 | 138.066 |
| 8192 | view, pre-copied | 110.816 | 114.905 | 233.464 | 164.004 |
| 16384 | contiguous | 233.714 | 234.638 | 479.292 | 316.210 |
| 16384 | view, pre-copied | 220.888 | 198.474 | 455.230 | 311.772 |

| K | Original copy-only ms | q16k8 incl. copy ms | Q-register copy-only ms | qreg64k16 incl. copy ms |
|---:|---:|---:|---:|---:|
| 8192 | 1.862 | 175.873 | 1.818 | 158.747 |
| 16384 | 3.193 | 329.449 | 3.555 | 336.040 |

Median paired candidate/dense128 ratios (lower is better; 1 means parity):

| Candidate | 8K contiguous | 16K contiguous | 8K incl. copy | 16K incl. copy |
|---|---:|---:|---:|---:|
| q8k16 | 6.321 | 5.536 | 5.131 | 5.449 |
| q16k8 | 2.225 | 1.817 | 1.405 | 1.376 |
| qreg32k16 | 2.049 | 1.949 | 2.292 | 2.053 |
| qreg64k16 | 1.401 | 1.467 | 1.442 | 1.430 |

Controls drift substantially, including within runs: original 8K view controls span
roughly 72–221 ms across pairs. Therefore separate medians must not be subtracted
to estimate copy cost, apparent copy-inclusive improvements are not real gains,
and small full-SDPA differences are not promoted. Cross-run absolute comparisons
are also confounded. Larger BQ improves the observed control-normalized loss versus
the original BK16/BQ8 configuration, but the best measured revision remains slower.

## Decision and remaining evidence

Keep serving defaults unchanged. This bounded reuse of the Steel kernel establishes
correct FP32/D256 GQA attention across offsets, tails, sampled positions, and real
activations; it does not establish acceleration. No further algorithm rewrite was
attempted after the two reviewed revisions.

Possible follow-up profiling questions are whether register pressure, loop/barrier
frequency, matrix instruction throughput, or occupancy limits dominate. These are
hypotheses, not diagnosed causes; spill/occupancy claims require compiler or GPU
profiling evidence. Any further revision needs a separate bounded design and matched
validation before native/serving integration.

Four CPU contract tests passed after observed red→green implementation, covering
sampled-position/GQA behavior, partial launch coverage, paired scheduling, and the
register-variant resource calculation. CPU preparation, all GPU correctness suites,
and all four paired timing runs completed. Independent review and repository graph
checks are coordinated by the parent task before a scoped commit.
