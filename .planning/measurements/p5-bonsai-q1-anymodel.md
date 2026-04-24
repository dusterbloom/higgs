# Bonsai-Q1 through AnyModel — full-matrix bench

Run unix: 1777071863
Path: AnyModel::BonsaiQ1 → BonsaiQ1Gpu::forward (P5 integration)
Workload: synthetic deterministic token IDs, argmax decode.

## Bonsai-1.7B

- layers: 28, vocab: 151669, packed resident: 256.5 MB
- load: 52 ms, to_gpu: 48 ms

### Prefill (per-shape kernel warmed, fresh KV cache)

| L (tokens) | ms | tok/s |
|---:|---:|---:|
| 1 | 5.6 | 178.6 |
| 16 | 31.5 | 507.7 |
| 128 | 113.0 | 1132.8 |
| 512 | 444.5 | 1151.9 |
| 2048 | 1993.5 | 1027.3 |

### Sustained decode after 16-token prefill (autoregressive argmax)

| decode steps | total ms | ms/tok | tok/s |
|---:|---:|---:|---:|
| 252 (after 4 warmup, uncompiled) | 1372.7 | 5.45 | 183.6 |

### Sustained decode (compile_with_state wrap)

| decode steps | total ms | ms/tok | tok/s | speedup |
|---:|---:|---:|---:|---:|
| 252 (after 4 warmup, compiled) | 1847.1 | 7.33 | 136.4 | 0.74x |

## Bonsai-8B

- layers: 36, vocab: 151669, packed resident: 1220.7 MB
- load: 265 ms, to_gpu: 133 ms

### Prefill (per-shape kernel warmed, fresh KV cache)

| L (tokens) | ms | tok/s |
|---:|---:|---:|
| 1 | 15.5 | 64.4 |
| 16 | 137.2 | 116.7 |
| 128 | 519.2 | 246.5 |
| 512 | 2089.5 | 245.0 |

### Sustained decode after 16-token prefill (autoregressive argmax)

| decode steps | total ms | ms/tok | tok/s |
|---:|---:|---:|---:|
| 124 (after 4 warmup, uncompiled) | 1945.8 | 15.69 | 63.7 |

### Sustained decode (compile_with_state wrap)

| decode steps | total ms | ms/tok | tok/s | speedup |
|---:|---:|---:|---:|---:|
| 124 (after 4 warmup, compiled) | 2237.5 | 18.04 | 55.4 | 0.87x |
