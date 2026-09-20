# Ternary Bonsai prefill experiments

Date: 2026-09-20

Model: `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`

Hardware: base Apple M4 MacBook Pro, 32 GB unified memory. The chunk sweep ran
on AC power with macOS reporting `lowpowermode=0`.

## Tiled ternary QMM

The BF16-staged Metal prototype passed all frozen correctness fixtures. The
final `BM=32, BN=32, BK=128` tile improved the first `BM=8` candidate by
1.28x-1.40x, but remained substantially slower than MLX stock QMM.

| Projection | M | Stock (ms) | Tiled (ms) | Tiled throughput / stock |
|---|---:|---:|---:|---:|
| gate/up | 512 | 74.72 | 180.32 | 0.414x |
| gate/up | 1024 | 161.44 | 402.68 | 0.401x |
| down | 512 | 38.34 | 85.81 | 0.447x |
| down | 1024 | 79.62 | 191.77 | 0.415x |

The 1.2x promotion gate failed for both projections. No production dispatch
was added.

## 4K prefill chunk sweep

Each condition used the same 4,096-token prompt, cache bypass, prefix cache
off, MTP/speculation off, and `max_tokens=1`. Each process ran one warmup and
three measured requests. Every request generated exactly `0`, reported 4,096
prompt tokens and one completion token, had zero cached tokens, and completed
prompt progress.

| Chunk | Median TTFT (s) | Prompt tok/s | CV | Peak RSS (GB) | Minimum live available memory (GB) |
|---:|---:|---:|---:|---:|---:|
| 512 | 80.697 | 50.76 | 6.823% | 5.64 | 5.60 |
| 768 | 81.639 | 50.17 | 0.542% | 6.43 | 4.90 |
| 1024 | 81.774 | 50.09 | 0.232% | 5.32 | 4.96 |
| 1536 | 81.643 | 50.17 | 0.033% | 6.65 | 4.20 |
| 2048 | 83.205 | 49.23 | 1.876% | 6.05 | 3.55 |

The apparent 1.3% advantage at chunk 512 is below its 6.8% run-to-run
variation. Chunks 768-1536 are effectively tied. Chunk 2048 is slower and
leaves less memory headroom. Retain the default chunk size of 1024.

## Packed ternary sparsity scan

The streaming CPU scan covered 402 packed `uint32` tensors and
26,869,760,000 two-bit codes. Peak scanner RSS was 177.8 MiB. Codes were
decoded in row-major `[output,input]` layout; structured groups and blocks
were aligned along the input dimension within each output row.

Overall code distribution:

| Value | Fraction |
|---:|---:|
| -1 | 33.654246% |
| 0 | 32.761538% |
| +1 | 33.584216% |
| invalid code 3 | 0% |

MLP structure:

| Projection | Zero fraction | Exact 2:4 | Exact 4:8 | Zero 8x8 blocks | Zero 16x16 blocks |
|---|---:|---:|---:|---:|---:|
| gate | 33.638823% | 29.987662% | 17.420963% | 0 | 0 |
| up | 33.617719% | 29.947384% | 17.396257% | 0 | 0 |
| down | 33.618647% | 29.972363% | 17.401968% | 0 | 0 |

The zeros are consistent with unstructured one-third ternary sparsity. There
are no naturally empty GPU-sized blocks, so block skipping is not available
without changing or retraining the model. Even perfect zero skipping has only
about a `1 / (1 - 0.336) = 1.51x` projection-level ceiling before overhead.

## Decision

- Keep MLX stock QMM for prefill.
- Keep prefill chunk size 1024.
- Do not pursue block skipping for this checkpoint.
- Preserve the tiled kernel as an isolated, default-disconnected experiment.

Raw local artifacts are under `target/bench-results/bench_prefill/`.
