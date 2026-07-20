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

The first runtime impact was real but small:

```text
head argmax only: 14.03 tok/s
```

The follow-up moved the final candidate reduction from CPU to a tiny GPU kernel returning `[1,5]` `uint32` ids directly. Isolated head timing was neutral because the projection dominates:

```text
MLX qmm + argmax:       15724.8 us
candidate + CPU reduce: 11284.3 us
candidate + GPU reduce: 11285.1 us
GPU vs CPU:             1.000x
argmax parity:          matched
```

End-to-end runtime did improve because the verifier no longer syncs/copies the candidate arrays to CPU:

```text
row2 MLP + CPU head reduce:         17.49 tok/s
row2 MLP + GPU head reduce:         18.55 tok/s
row2 MLP + GPU head + split-K down: 19.29 tok/s
```

The next head probe specialized the candidate kernel to the checkpoint's strict ternary affine structure (`bias = -scale`, `weight = scale * (q - 1)`). The generic synthetic benchmark initially failed parity with arbitrary affine biases, then matched once the benchmark used strict ternary biases:

```text
MLX qmm + argmax:            15494.7 us
affine candidate + GPU reduce: 11218.7 us
ternary candidate + GPU reduce: 8369.6 us
ternary vs affine candidate: 1.340x
ternary vs MLX qmm+argmax:  1.851x
argmax parity:              matched under strict ternary affine
```

Runtime with strict ternary head:

```text
row2 MLP + GPU head + split-K down + ternary head: 19.68 tok/s
```

Split-K variants for the strict ternary head were then tested. They were parity-correct but did not materially beat the direct strict ternary candidate:

```text
MLX qmm + argmax:              15594.7 us
affine candidate + GPU reduce: 11307.1 us
ternary direct + GPU reduce:    8495.6 us
ternary split-K2 + GPU reduce:  8476.4 us
ternary split-K4 + GPU reduce:  8655.0 us
split-K2 vs direct:             1.002x
split-K4 vs direct:             0.982x
argmax parity:                  matched
```

Conclusion: split-K head is below the wiring threshold. The extra partial-sum traffic cancels the small parallelism gain, unlike `down_proj`.

Server telemetry stayed behavior-identical:

```text
accept_len: 4.23
spec_rounds: 30
server decode after warmup before split-K: ~16.0 tok/s
server decode after warmup with split-K:   ~16.5-16.8 tok/s
server decode with ternary head:           ~17.0 tok/s
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

Split-K was tested for `down_proj`, where `K=17408` and the direct row2 kernel exposes less row parallelism:

```text
gate/up M=5:
  MLX stock:     1356.9 us
  ternary row2:   929.0 us
  split-K2:       941.1 us
  split-K4:       935.0 us

down M=5:
  MLX stock:     1366.0 us
  ternary row2:  1295.6 us
  split-K2:      1164.8 us
  split-K4:      1144.8 us
```

Conclusion: split-K hurts or is neutral for gate/up, but wins for down. Runtime dispatch now uses split-K4 only for the Bonsai ternary `down_proj` shape (`N=5120`, `K=17408`) under the row2 MLP flag.

Full MLP microbench:

```text
MLX stock full MLP:    3557.4 us
ternary row2 full MLP: 2557.2 us
speedup:              1.391x
```

Hybrid row2 plus stock MLX down was tested after the row2 win:

```text
MLX stock full MLP:                 3518.8 us
ternary row2 full MLP:              2542.8 us
row2 gate/up + MLX stock down:      2642.2 us
row2 full MLP speedup:              1.384x
hybrid row2/MLX-down speedup:       1.332x
```

Conclusion: keeping `down_proj` on stock MLX does not recover throughput. The all-row2 MLP remains faster, so the remaining gap is not explained by a bad row2 down-projection dispatch alone.

Fusing row2 gate/up into a single Metal launch was also tested:

```text
MLX stock full MLP:                 3500.9 us
ternary row2 full MLP:              2525.7 us
fused row2 gate/up + row2 down:     2594.8 us
row2 full MLP speedup:              1.386x
fused row2 gate/up speedup:         1.349x
```

Conclusion: one fewer launch is not enough to win here. The fused gate/up kernel increases per-thread register/weight work enough that it loses to two independent row2 projection launches.

Power-of-two M-scaling was tested with the legal dSpark cap:

```bash
HIGGS_DSPARK_DRAFT_CAP=3
```

This gives verifier `M=4` (`anchor + 3 draft`) because the published dSpark artifact has `block_size=4`; attempts to set `HIGGS_DSPARK_DRAFT_CAP=7` were clamped back to `draft_cap=4`.

Long Fibonacci, 128 tokens:

```text
client decode mean: 13.68 tok/s
accept_len:         3.63
spec_rounds:        35
server decode:      ~12.0-12.4 tok/s
```

Conclusion: the feasible power-of-two verifier tile loses. The lower draft cap increases rounds and drops acceptance enough that any M=4 scaling benefit is erased. Testing M=8 would require a dSpark artifact or verifier schedule that can actually propose seven draft positions.

## Verifier overhead outside target forward

Phase tracing showed that accepted-id resolution and commit bookkeeping are not the remaining bottleneck:

```text
steady resolve_ms:       ~0.1 ms
steady draft_commit_ms:  ~0.0 ms
steady target_commit_ms: ~0.1 ms
```

The measurable non-target overhead is drafter-side:

```text
sidecar output proposal:
  steady stage_ms:   ~13 ms
  steady propose_ms: ~20 ms
  steady total_ms:   ~251-253 ms

target-head proposal:
  steady stage_ms:   ~13 ms
  steady propose_ms: ~17 ms
  steady total_ms:   ~246-248 ms
```

However, full 128-token Fibonacci throughput with target-head proposal did not beat the sidecar-output best:

```text
sidecar output proposal best: 19.68 tok/s
target-head proposal:        19.61 tok/s
```

Conclusion: `HIGGS_DSPARK_TARGET_HEAD=1` reduces proposal timing in traces, but it is not a net throughput win on the apples-to-apples run. The next useful work should target the dSpark proposal implementation itself, especially the sequential Markov resampler and sidecar Q4 output path.

Proposal-detail tracing (`HIGGS_DSPARK_PROPOSE_TRACE=1`) then split the sidecar proposal:

```text
steady sidecar output projection: ~16-17 ms
steady Markov step 0:             ~1.1 ms
steady Markov step 1:             ~1.1 ms
steady Markov step 2:             ~1.1 ms
steady Markov step 3:             ~1.1 ms
steady concat:                    ~0.1-0.2 ms
```

Conclusion: the sidecar Q4 output projection is the dominant proposal cost. A fused `base logits + Markov bias -> argmax` kernel can only attack roughly `4-5 ms` per round; the larger lever is the sidecar output head projection itself.

Stage-detail tracing (`HIGGS_DSPARK_STAGE_TRACE=1`) split the dSpark trunk:

```text
steady context append/project: ~1.5-2.3 ms
steady log-SNR add:           ~0.1-0.2 ms
steady layer 0:               ~2.1 ms
steady layer 1:               ~2.1 ms
steady layer 2:               ~2.1 ms
steady layer 3:               ~2.1 ms
steady layer 4:               ~2.1 ms
steady layer 5:               ~2.1 ms
steady final norm:            ~0.2 ms
steady stage total:           ~15 ms with trace barriers
```

Conclusion: the trunk is evenly distributed across six layers; no single stage subcomponent is an obvious quick win. The output head remains the most concentrated target.

Runtime result with:

```bash
HIGGS_DSPARK_Q2_ROW2_MLP=1 \
HIGGS_DSPARK_Q2_HEAD_ARGMAX=1
```

Long Fibonacci, 128 tokens:

```text
previous long Fibonacci: 13.87 tok/s
row2 MLP + head argmax:  19.68 tok/s
```

Server telemetry:

```text
accept_len: 4.23
spec_rounds: 30
server decode before: ~12.0-12.1 tok/s
server decode after:  ~17.0 tok/s
```

This is about `1.46x` versus the AR Fibonacci baseline, still short of the `1.5x` target but now within roughly another three percent.

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

The remaining gap is likely in the dSpark drafter proposal/stage path and smaller verifier scheduling overheads.

Recommended next moves:

1. Probe a faster sidecar Q4 output head path for `[1,4,5120] -> [1,4,vocab]`, ideally argmax-oriented so it avoids materializing all logits if Markov bias can be folded later.
2. Keep fused Markov `base logits + low-rank bias -> argmax` as a secondary probe; it can save at most the measured ~4-5 ms per round unless paired with output-head changes.
3. If the output-head path stalls, inspect one dSpark trunk layer internally; stage-level tracing shows each layer costs about the same.
4. Test M=8 only if a dSpark artifact or schedule can produce seven draft positions; the current artifact clamps at four.
5. Keep native verifier scheduling as an explicit probe/flag until exactness and acceptance are understood across prose.

## Sidecar Q4 output-head argmax probe

The next bottleneck after the verifier target-head and row2 MLP work is the dSpark sidecar proposal path. Proposal tracing showed the frozen sidecar Q4 output projection dominates steady-state draft generation:

```text
sidecar Q4 output projection/eval: ~16-17 ms
four Markov steps total:           ~4-5 ms
concat:                            ~0.1-0.2 ms
```

A Q4 M=4 output-head argmax candidate kernel was tested against MLX `qmm + argmax` for the sidecar output-head shape (`248320 x 5120`). This is an argmax-only kill gate, not yet a drop-in runtime path, because the dSpark Markov resampler needs exact `base_logits + markov_bias` argmax per sequential token.

```text
Q4_M4_OUTPUT_HEAD_ARGMAX_PARITY ref=[139620, 13434, 93037, 51939] cand=[139620, 13434, 93037, 51939]
MLX qmm + argmax:            11249.7 us
candidate + GPU reduction:    9453.0 us
speedup:                       1.190x
argmax parity:                 matched
```

Conclusion: the sidecar Q4 output head still has measurable room. The current custom argmax path is not directly wireable because it only finds `argmax(base_logits)`, while dSpark needs Markov-biased argmax. The next scientifically useful probe is an exact fused one-step kernel for `output_q4(hidden) + markov_q4(A(prev_token)) -> argmax`, then benchmark four sequential fused steps against the current `output_q4(M=4) + Markov` path.

## Exact fused Markov-biased Q4 argmax probe

The direct follow-up was an exact one-step kernel for the actual sidecar shape:

```text
output_head: 248320 x 5120, Q4 group size 32
markov_b:    248320 x 256,  Q4 group size 32
```

The kernel computes `argmax(output_q4(hidden_row) + markov_q4(markov_embedding_row))` without materializing either logits vector. Four sequential M=1 calls were compared against the current proposal math: one MLX M=4 output-head projection plus four Markov-bias projections/adds/argmaxes.

```text
Q4_M1_MARKOV_ARGMAX_PARITY ref=[51939, 76445, 93037, 20293] fused=[51939, 76445, 93037, 20293]
current MLX M=4 output + Markov: 14218.7 us
fused exact M=1 x4:              40718.0 us
speedup:                         0.349x
argmax parity:                   matched
```

Conclusion: exact four-step M=1 fusion is a dead end. It repeats the 5120-wide output dot four times and loses the M=4 batching advantage by more than the materialization savings can recover. The output-head path needs either an M=4 fused Markov-aware kernel that preserves output batching, or a different strategy that avoids recomputing the output dot per proposed token.

## Markov-only fused Q4 argmax probe

The safer exact variant preserved MLX's batched M=4 sidecar output projection and fused only the sequential Markov tail:

```text
base_logits = output_q4(hidden[4])
argmax(base_logits[row] + markov_q4(markov_embedding[row]))
```

This avoids recomputing the expensive `5120 -> vocab` output dot four times. It only streams the `256 -> vocab` Markov-B projection and reduces against the already materialized base row.

```text
Q4_M1_MARKOV_ARGMAX_PARITY ref=[51939, 76445, 93037, 20293] fused=[51939, 76445, 93037, 20293] markov_only=[51939, 76445, 93037, 20293]
current MLX M=4 output + Markov: 14062.9 us
full fused M=1 x4:               41893.7 us
Markov-only fused:               13696.9 us
full fused speedup:              0.336x
Markov-only speedup:             1.027x
argmax parity:                   matched
```

Conclusion: preserving M=4 output batching was the right direction, but the exact Markov-only fusion only trims about `2.7%` in this benchmark. It is not enough by itself to explain a path from `19.68 tok/s` to a robust `1.5x`. The next higher-upside question is whether full-vocab Markov-biased argmax is necessary at all: measure base-topK containment of `argmax(base + markov_bias)` on real proposal states, then consider topK plus exact fallback only if containment is high.

## Base-topK containment probe

The next probe measured whether the exact dSpark token from `argmax(base_logits + markov_bias)` was already present in the base-output head's top-K candidates. This was instrumented behind:

```bash
HIGGS_DSPARK_TOPK_TRACE=1
```

The trace is non-invasive: it computes the current exact Markov-biased argmax, then ranks that chosen token under the base logits alone. It is intentionally not a throughput benchmark because it copies the base row to CPU for rank measurement.

Long Fibonacci, 128 tokens, one warmup and one traced trial:

```text
samples:        234 proposal positions
mean base rank: 1.20
max base rank:  13
hit@16:         100.0%
hit@32:         100.0%
hit@64:         100.0%
hit@128:        100.0%
hit@256:        100.0%
hit@512:        100.0%
accept_len:     4.23
spec_rounds:    30
```

Representative final cumulative line:

```text
samples=234 mean_rank="1.20" max_rank=13 hit16_rate="1.000" hit32_rate="1.000" hit64_rate="1.000" hit128_rate="1.000" hit256_rate="1.000" hit512_rate="1.000"
```

Conclusion: for the Fibonacci workload, the Markov bias almost never changes the candidate outside the base head's tiny shortlist. This is the first evidence with enough upside for a larger proposal-path speedup. The next implementation probe should test a top-16 or top-32 candidate path with exact fallback: compute base topK, evaluate Markov-B only for those candidate rows, and fall back to the full current Markov path whenever the exactness guard cannot prove the shortlist winner.

## Top-16 Markov shortlist compare probe

A first implementation probe was added behind:

```bash
HIGGS_DSPARK_TOPK_COMPARE=16
```

This mode is still exact-generation: it computes a shortlist candidate, then computes the existing full-vocab Markov-biased argmax and logs whether they match. The shortlist path uses MLX base top-K selection, gathers only those Markov-B output rows, evaluates `markov_q4(A(prev))` only for the shortlist, adds the base top-K scores, and picks the best candidate.

Long Fibonacci, 128 tokens, one warmup and one traced trial:

```text
samples:       234 proposal positions
K:             16
matches:       234 / 234
hit rate:      100.0%
accept_len:    4.23
spec_rounds:   30
```

Representative final cumulative line:

```text
position=0 k=16 exact=18 shortlist=18 matched=true samples=234 hit_rate="1.000"
```

Conclusion: the top-16 shortlist is exact on this Fibonacci run in compare mode. The current compare path is not a throughput result because it deliberately runs both shortlist and full exact computation. The next probe should add a runtime fast-path flag that uses the shortlist token directly, with a separate compare flag retained for correctness audits on prose.

## Top-16 Markov shortlist fast-path probe

The compare-only shortlist was then promoted behind:

```bash
HIGGS_DSPARK_TOPK_FAST=16
```

This mode uses the top-16 shortlist token directly and skips the full-vocab Markov-biased argmax. The exact full path remains available, and `HIGGS_DSPARK_TOPK_COMPARE=16` can still be used for audits.

Long Fibonacci, 128 tokens, one warmup and three measured trials:

```text
client decode mean: 18.77 tok/s
trials:             18.69, 18.49, 19.12 tok/s
accept_len:         4.23
spec_rounds:        30
server decode:      ~16.0-16.5 tok/s after warmup
```

Current best without the shortlist fast path remains:

```text
client decode mean: 19.68 tok/s
```

Conclusion: the top-16 shortlist is behavior-correct on Fibonacci but slower in this implementation. The likely culprit is MLX `argpartition + gather + tiny quantized_matmul` overhead: selecting and gathering 16 rows costs more than the full Markov tail it replaces. This kills the stock-MLX shortlist fast path. If shortlist acceleration is revisited, it needs a custom streaming top-K/Markov kernel or a cheaper candidate source than MLX argpartition.

Clean rerun after closing LM Studio and with no cargo build running:

```text
client decode mean: 18.96 tok/s
trials:             19.36, 18.90, 18.61 tok/s
accept_len:         4.23
spec_rounds:        30
server decode:      ~16.1-16.8 tok/s after warmup
```

This improves over the contaminated `18.77 tok/s` run but remains below the current best `19.68 tok/s`. The top-K scaffold should be kept: correctness evidence is strong, but the stock-MLX implementation is overhead-limited. The next move is not deletion; it is replacing `argpartition + gather + tiny qmm` with a custom base top-16 extraction kernel, then a fused topK Markov scorer if needed.
