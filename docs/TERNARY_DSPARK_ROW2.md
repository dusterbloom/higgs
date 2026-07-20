# Ternary dSpark verifier optimization notes

Date: 2026-07-20

This note records the evidence trail for the Ternary-Bonsai-27B dSpark verifier work. The goal was to close the gap between the ternary dSpark speedup and the earlier Bonsai Q1 row4/TG-LUT result.

## Baseline context

Target model:

```text
/Users/peppi/.cache/lm-studio/models/prism-ml/Ternary-Bonsai-27B-mlx-2bit
```

Drafter sidecar:

```text
/Users/peppi/models/ternary-bonsai-27b-dspark-mlx
```

Reference dSpark flags:

```bash
HIGGS_DFLASH_VERIFY_MODE=block \
HIGGS_DFLASH_GATE=0 \
HIGGS_DSPARK_DRAFT_CAP=4 \
HIGGS_DSPARK_TARGET_HEAD=0
```

Earlier long Fibonacci reference:

```text
client decode: 13.87 tok/s
server decode: ~12.0 tok/s
accept_len: 4.23
spec_rounds: 30
```

The AR Fibonacci reference was about `13.51 tok/s`, so the old ternary speculative path was only a small win.

## What did not explain the gap

MLX version was not the blocker. Newer MLX/mlx-c experiments either failed API compatibility or regressed ternary decode.

Gate/up fusion was not enough:

```text
gate+up only: 1.061x
full MLP with fused gate/up: 1.026x
```

Tensor rank was not the blocker. `[1,5,K]` was comparable to `[5,K]` for MLX Q2 matmuls.

Native dSpark verifier batching is not safe as a default. It can help regular prompts, but on prose it dropped acceptance to `2.89`, increased rounds to `44`, and regressed throughput to about `8.28 tok/s`.

Zero-elision is not promising for this checkpoint. Real tensor scans showed zero-code density around `28-30%`, with no sampled groups near the `50-60%` threshold where sparse masks become plausible.

## Checkpoint structure

The actual ternary checkpoint has strict ternary affine structure:

```text
codes: q in {0, 1, 2}; code 3 unused
bias / scale: exactly -1.0
weight: scale * (q - 1)
```

This made a strict ternary kernel viable. It can drop the bias buffer and avoid generic affine Q2 math.

## Head argmax result

A verifier-only Q2 `lm_head -> argmax` candidate kernel avoids materializing full `[5, vocab]` logits.

Isolated kill gate:

```text
MLX qmm + argmax: 15686.5 us
candidate kernel: 11300.2 us
speedup: 1.388x
argmax parity: matched
```

Runtime impact was real but small:

```text
head argmax only: 14.03 tok/s
```

The head path is available behind:

```bash
HIGGS_DSPARK_Q2_HEAD_ARGMAX=1
```

## Winning row2 MLP path

The breakthrough was a strict ternary row2 M=5 kernel for verifier MLP projections:

```text
bonsai_q2_row2_m5_ternary_direct
```

It uses row2-transposed Q2 weights and strict ternary math:

```text
output = scale * sum((q - 1) * x)
```

Projection microbench:

```text
gate/up M=5:
  MLX stock:    1394.2 us
  ternary row2:  935.8 us
  speedup:      1.49x

down M=5:
  MLX stock:    1366.2 us
  ternary row2: 1308.7 us
  speedup:      1.04x
```

Full MLP microbench:

```text
MLX stock full MLP:    3557.4 us
ternary row2 full MLP: 2557.2 us
speedup:              1.391x
```

Runtime result with:

```bash
HIGGS_DSPARK_Q2_ROW2_MLP=1 \
HIGGS_DSPARK_Q2_HEAD_ARGMAX=1
```

Long Fibonacci, 128 tokens:

```text
previous long Fibonacci: 13.87 tok/s
row2 MLP + head argmax:  17.49 tok/s
```

Server telemetry:

```text
accept_len: 4.23
spec_rounds: 30
server decode before: ~12.0-12.1 tok/s
server decode after:  14.9-15.5 tok/s
```

This is about `1.29x` versus the AR Fibonacci baseline, still short of the `1.5x` target but now materially close.

## Runtime controls

The row2 MLP path is opt-in:

```bash
HIGGS_DSPARK_Q2_ROW2_MLP=1
```

The head argmax path is opt-in:

```bash
HIGGS_DSPARK_Q2_HEAD_ARGMAX=1
```

Native dSpark verifier remains opt-in because it is prompt-sensitive:

```bash
HIGGS_DSPARK_NATIVE_VERIFY=1
```

## Next targets

The remaining gap is likely in `down_proj`, `lm_head`, and verifier overhead outside the MLP.

Recommended next moves:

1. Improve `down_proj` row2 M=5, where current speedup is only `1.04x`.
2. Replace the head candidate path with a row2/radix-3 head argmax kernel to recover more of the `lm_head` cost.
3. Explore radix-3 prepack only after proving it beats direct row2 on gate/up and down in isolation.
4. Keep native verifier scheduling as an explicit probe/flag until exactness and acceptance are understood across prose.
