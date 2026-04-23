# Phase 2 Probe v2 — salvage parse (Mac crashed mid-sweep)

Date parsed: 2026-04-23T10:27:28Z
Target: Qwen3.6-27B-4bit  Drafter: Qwen3.5-27B-DFlash  Chunk: 32 (all)

Crash context: bs16 launched at 10:15:51Z, Mac went down during model load.
Only cap094 and capunset produced full completions.

| config | BS | cap | vbuild_med | vfwd_med | rtotal_med | eff_tps_last | avg_accept_frac | peak_rss_mb | cap_mb | backend |
|--------|----|-----|------------|----------|------------|--------------|-----------------|-------------|--------|---------|
| baseline | 12 | 0.88 | - | - | - | - | - | 5363 | - | CPU BLAS (forced) |
| cap094 | 12 | 0.94 | 267.05 | 319.65 | 678.80 | 3.1 | 0.18 | - | - | CPU BLAS (forced) |
| cap100 | 12 | 1.0 | - | - | - | - | - | 7841 | - | CPU BLAS (forced) |
| capunset | 12 | unset | 313.25 | 337.60 | 754.85 | 2.8 | 0.18 | - | - | CPU BLAS (forced) |
| bs16 | 16 | 0.88 | - | - | - | - | - | - | - | unknown |

## Completion excerpts (first 80 chars)
- baseline: (empty/missing)
- cap094: Autumn arrives not with a shout, but with a whisper, a subtle shift in the air t
- cap100: (empty/missing)
- capunset: Autumn arrives not with a shout, but with a whisper, a subtle shift in the air t
- bs16: (empty/missing)

## Verdict

Stop-criterion: verify_build ≤ 200 ms AND eff_tps ≥ 10 AND avg_accept ≥ 0.30.
