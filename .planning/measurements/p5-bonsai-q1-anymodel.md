# Bonsai-Q1 through AnyModel — full-matrix bench

Run unix: 1777053570
Path: AnyModel::BonsaiQ1 → BonsaiQ1Gpu::forward (P5 integration)
Workload: synthetic deterministic token IDs, argmax decode.

## Bonsai-1.7B

- layers: 28, vocab: 151669, packed resident: 256.5 MB
- load: 52 ms, to_gpu: 53 ms

### Prefill (per-shape kernel warmed, fresh KV cache)

| L (tokens) | ms | tok/s |
|---:|---:|---:|
| 1 | 11.3 | 88.4 |
| 16 | 71.3 | 224.3 |
| 128 | 136.1 | 940.6 |
| 512 | 541.1 | 946.2 |
| 2048 | 2453.6 | 834.7 |

### Sustained decode after 16-token prefill (autoregressive argmax)

| decode steps | total ms | ms/tok | tok/s |
|---:|---:|---:|---:|
| 252 (after 4 warmup) | 2919.8 | 11.59 | 86.3 |

## Bonsai-8B

- layers: 36, vocab: 151669, packed resident: 1220.7 MB
- load: 263 ms, to_gpu: 150 ms

### Prefill (per-shape kernel warmed, fresh KV cache)

| L (tokens) | ms | tok/s |
|---:|---:|---:|
| 1 | 43.5 | 23.0 |
| 16 | 164.8 | 97.1 |
| 128 | 630.0 | 203.2 |
| 512 | 2525.6 | 202.7 |

### Sustained decode after 16-token prefill (autoregressive argmax)

| decode steps | total ms | ms/tok | tok/s |
|---:|---:|---:|---:|
| 124 (after 4 warmup) | 5293.9 | 42.69 | 23.4 |
