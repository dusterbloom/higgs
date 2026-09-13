# M4 prefill investigation — 13 September 2026

Stages 1–3 are complete. No stable whole-request speedup was demonstrated.
Chunk-size timing was confounded by 19% control drift; corrected GDN launch
changes were neutral, and the existing scalar sparse-attention kernel lost.
The 512-token chunk used less memory in this screen. Full attention
becomes the largest sampled component late in a long prompt, making an exact,
tiled attention kernel for the observed FP32/D256 geometry a strong next
experiment. Sparse selection alone cannot rescue the tested scalar kernel.

## Scope and controls

Machine: base Apple M4, Mac16,1, 32 GiB unified memory, battery power. Checkpoint:
`EschaLabs/Qwen3.6-35B-A3B-Escha-W2`, a native Escha MoE checkpoint. These results
do not measure dense Qwen3.8-27B or predict its performance.

The existing [corrected baseline](../ane_prefill/BASELINE.md) measured TTFT of
327.289 s at 32,005 input tokens and 418.301 s at 45,001. Those are separate
single observations, not paired controls for this campaign. All serving runs
use a private localhost server, the throughput profile, dense KV, disabled disk
prefix caching, `raise_wired_limit=true`, a 256 MiB allocator cache, a short
warmup, and a fresh long-prompt cache. A watchdog rejects any new cumulative
system swapouts. Request latency excludes loading and warmup. No terminal
prefill timing was returned by SSE, so we report **TTFT**, not pure prefill time.

The uninstrumented executable was pinned as `target/prefill-investigation/higgs-control`
with SHA256 `8c4c9fa407e1b89964d80accb434f5d1b7cd442d74618e13d28bc287af3640a0`.
It includes the existing worktree changes recorded with the baseline. The
capture binary is separate, SHA256
`cd4424a187db959311a522e9c095f410d04f2485161d933c43c6bad6b1649c57`.
Python microbenchmarks use MLX 0.30.6; they do not share the complete Rust serving
execution graph. Hardware workloads were serialized. Battery operation and
visible latency drift limit small-effect conclusions.

## 1. Profile the long-context serving path

The 45,001-token diagnostic request completed with both retrieval facts,
zero new swapouts, 502.378 s TTFT, 528.282 s total request time, and 16.509 GiB
peak sampled physical footprint. Instrumentation inserts evaluation barriers,
so this timing is **not a performance candidate**.

Existing `HIGGS_PROFILE` samples the leading three GDN layers and first
full-attention layer in each chunk. Multiplying their mean component times by
30 GDN and 10 full-attention layers gives this coarse attribution:

| Component group | Extrapolated sampled share |
| --- | ---: |
| Full-attention group | 41.99% |
| MLP/MoE across both layer types | 37.26% |
| GDN group | 20.76% |

The full-attention share rises from 20.98% in the first third to 41.01% in the
middle and 53.86% in the last third. Each attention group includes its
projections, cache/recurrent work and other surrounding operations; these are
not isolated SDPA or recurrence measurements, nor uninstrumented wall-time
shares. Later layers may differ from sampled layers.

As an illustrative Amdahl calculation on these shares, doubling the **entire**
full-attention group gives about 1.27× overall; deleting it gives an optimistic
1.72× ceiling. Doubling the entire GDN group gives about 1.12×. A kernel that
covers only part of either group has a smaller bound. Under this illustrative decomposition, a 3× whole-request gain
needs improvements across several substantial components or a change
in how much model work is performed.

Xcode 27 Metal System Trace recorded briefly but crashed while saving with an
`XRMTLResidencySetModeler` assertion. The trace bundle is unusable; no shader
counters or GPU occupancy conclusions are claimed. The failure log is retained.

## 2. Exact GPU controls

### GDN recurrence launch geometry

The standalone probe copies the current serial-token Metal recurrence with
`B=1, Hk=16, Hv=32, Dk=Dv=128`, changing only threadgroup Y from 4 to 1, 2 or 8.
It checks every output and final recurrent state, including a short sequential
NumPy oracle. At T=1024 all launch variants are bitwise equal to the existing
TG4 schedule in both BF16 and FP32 campaigns.

Early sequential screens suggested a possible gain, but review found that the
original paired loop reused TG8 launch arguments for both labels. **All original
paired GDN timings are invalid for comparing launch geometries.** The corrected
harness records actual threadgroups, preserves each T1024 kernel/argument set,
uses an explicit TG4 scalar oracle, and has a mocked launch-order regression.

Fresh, corrected 15-pair runs gave:

| Inputs | Screen-selected candidate | TG4 median | Candidate median | Median within-pair candidate / TG4 |
| --- | --- | ---: | ---: | ---: |
| BF16 | TG2 | 13.481 ms | 13.499 ms | 1.0005× |
| FP32 | TG1 | 14.242 ms | 14.227 ms | 0.9879× |

Both runs passed output/final-state parity, actual-launch assertions and zero
new cumulative swapouts. Pair ratios ranged from 0.934–1.188 in BF16 and
0.913–1.126 in FP32. These small median differences with overlapping samples
do not justify replacing TG4. The earlier BF16 run recorded only swap usage,
so its swapout gate remains unknown.

Actual recurrent-input dtype was not captured; BF16 and FP32 probes bracket the
relevant kernel types. Standalone times cannot be directly subtracted from Rust
whole-layer timings. This rejects the tested launch-only promotion; it does not
test a tiled or parallel recurrence algorithm.

### Whole-model chunk sweep

The four arms are 1024 → 512 → 2048 → 1024 tokens per chunk, using identical
32K request bytes and the pinned uninstrumented executable. Each arm starts a
fresh server, warms it, waits for stable swap counters, and records memory,
power, cache counts and both retrieval facts.

| Arm | Chunk tokens | TTFT | Whole request | Peak physical footprint | New timed swapouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control A | 1024 | 338.092 s | 340.187 s | 15.604 GiB | 0 |
| Candidate | 512 | 325.606 s | 327.633 s | 14.820 GiB | 0 |
| Candidate | 2048 | 297.945 s | 299.923 s | 17.370 GiB | 0 |
| Control B | 1024 | 273.793 s | 275.765 s | 15.652 GiB | 0 |

Every arm returned 19 output tokens, both retrieval facts and zero cached
prompt tokens. The identical controls differ by 64.299 s (19.02% lower TTFT in
Control B). Both candidate latencies lie between them. Chunk 2048 appears
11.87% faster against Control A but 8.82% slower against Control B: **there is
no established chunk-size speedup**. The 512-token arm used roughly 0.78–0.83
GiB less peak footprint than the controls; 2048 used roughly 1.72–1.77 GiB more.
These are single observations, not a stable distribution.

No 45K candidate follow-up was launched because no timing candidate survived
the bracketing controls. Prepared 45K runner files remain local and unexecuted.
Before promoting a small whole-model gain, repeat balanced/randomized pairs
under more stable power/thermal conditions and distinguish compilation,
OS page-cache and background-load effects. This campaign did not isolate the
cause of the control drift. New swapouts during process loading were allowed
to settle before each timed request; the zero-swapout claim applies to the
timed interval, not the entire server lifetime.

Initial attempts failed before loading because the
copied executable lacked its adjacent `mlx.metallib`; those failures are
retained and the required library was bundled before retrying.

## 3. Sparse attention: real activations and a hardware gate

### Observed geometry and approximation error

A separate heterogeneous 16,345-token request assembled from repository prose
and code produced nine actual Q/K/V captures: true full-attention layers 3, 19
and 39 at offsets 1024, 7168 and 15360. All observed Q/K/V sources are FP32;
Q has 16 heads, KV has 2 heads, and head dimension is 256. Each file contains
up to 16 query rows around 128-token boundaries and the full visible KV prefix.
The request finished with both retrieval facts and zero new swapouts. Its
capture-enabled timing is diagnostic only.

The FP32 source is upstream: native `EschaProj` casts inputs and its folded
scales to FP32 and returns FP32; adding its MoE output promotes the residual
stream. Subsequent Q/K/V projections preserve that dtype. Q/K normalization
and RoPE are not an accidental new promotion: V already arrives as FP32, and
RoPE casts rotated halves back to its input dtype. See
[Escha projection](../../crates/higgs-models/src/eschamoe.rs#L613) and
[attention projection](../../crates/higgs-models/src/qwen3_next.rs#L4834).
Downcasting the residual stream or Q/K/V to BF16 would change arithmetic;
BF16 KV storage would also affect later decode. Treat those as separate
quality-tested candidates, not exact dtype-preserving optimizations.

The CPU evaluator compares dense causal GQA with B128 row-local mean correction,
keeping sink/local/boundary regions exact. Alpha zero passed dense NumPy oracle
parity at 1e-6 across all nine files before positive-alpha results were emitted.
This validates the evaluator's dense limit; it is not a comparison against
captured MLX attention outputs or end-to-end model logits.

At the late 16K offset:

| Alpha | Layer | Mean selected-token density | p99 row/head density | p99 relative L2 error | Max relative L2 error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 3 | 65.3% | 100.0% | 6.1% | 7.2% |
| 0.01 | 19 | 31.7% | 86.5% | 9.7% | 12.9% |
| 0.01 | 39 | 66.0% | 98.8% | 2.1% | 4.2% |
| 0.10 | 3 | 23.6% | 89.7% | 28.7% | 70.3% |
| 0.10 | 19 | 10.8% | 33.7% | 36.7% | 44.1% |
| 0.10 | 39 | 20.8% | 64.0% | 8.7% | 13.8% |

There is no single attractive global alpha from this screen: small alpha often
retains most tokens in the tail, while aggressive alpha creates large,
layer-dependent errors. Mean density alone also hides nearly dense heads.
These are attention-output errors on one prompt, not downstream behavioral
scores. No approximation was enabled in serving, and neither KVA nor PFlash
was tested.

### Metal traversal ceiling

The existing scalar row4/head4 feasibility kernel was measured at the observed
FP32/D256/GQA geometry, Q=1024, K=8192 or 16384. The dense control uses
128-query tiling with preallocated Boolean causal masks. This matches the
serving fallback shape at 16K; at 8K, serving does not activate its >=16K
tiling guard, so the 8K control is not the exact serving dispatch. The
100%-density probe passed maximum absolute and per-row relative-L2 error gates.
Timing uses alternating dense/candidate runs; raw samples preserve clock drift.

Even at **nominal 12.5% block density**, without paying selection or correction:

| KV length | Schedule | Scalar candidate median | Dense control median | Candidate / dense |
| --- | --- | ---: | ---: | ---: |
| 8192 | row4 | 174.8 ms | 88.4 ms | 1.98× |
| 8192 | head4 | 206.1 ms | 100.3 ms | 2.06× |
| 16384 | row4 | 776.4 ms | 428.7 ms | 1.81× |
| 16384 | head4 | 446.6 ms | 215.8 ms | 2.07× |

Rounding and causal-boundary inclusion produce per-row token densities of
11.12–14.04% at 8K and 11.82–13.22% at 16K. Ratios above divide arm medians; median within-pair ratios are 2.03×, 2.06×,
1.65× and 2.07× respectively. The 16K row4 case has substantial drift, but
every individual pair still loses. At full density it is roughly 19–23× slower. All hardware cases had zero new
cumulative swapouts. Inputs are synthetic contiguous tensors; trace-view
copying and integration costs are not included. This is an optimistic reduced
work timing, not a quality-matched sparse implementation.

**Decision:** reject this scalar kernel design for integration. The result does
not rule out sparse attention using tiled matrix operations and shared GQA KV
loads. Such a design must beat the existing dense path at a density supported
by real quality evidence, including selection, correction and layout costs.

## What deserves the next engineering effort

1. **Exact tiled FP32/D256 attention.** Start with a dense kernel at real Q/K/V
   strides and causal offsets, testing SIMD-group matrix reuse and GQA reuse.
   Beat the current 128-query fallback before adding sparse selection. This
   targets the growing long-context component and supplies a better foundation
   for a later approximate path.
2. **Tiled GDN recurrence plus surrounding-operation attribution.** The current
   recurrence remains serial in T. Test a real algorithm change with output
   and final-state parity; separately measure projection, convolution,
   normalization and output costs so a kernel gain is weighted correctly.
3. **Native Escha fusion/layout work where a trace proves repeated traffic.**
   The existing native route already globally sorts experts and uses fused
   gate/up trellis GEMM, SwiGLU, down projection and unsorting. The generic
   affine gate/up switch is not a new optimization for this checkpoint.
   Hadamard transforms and surrounding intermediate traffic remain hypotheses,
   not measured wins. A full dequantized expert cache cannot be assumed to fit.
4. **Approximation only after the hardware and quality gates.** Investigate
   layer/head-dependent policies or separate KVA/PFlash branches, with logits,
   retrieval/reasoning and deployed-generation tests on heterogeneous prompts.

The [Strix Halo journey](https://pwilkin.github.io/strix-halo/journey.html)
strengthens the tiled-GDN hypothesis, but its AMD-specific aggregate improvement
is not portable. Its final ablation has approximately neutral performance when
sparse attention is fully disabled, while disabling only its chosen sparse
kernel leaves selection overhead and is much worse. Those are different
comparisons. ROCm matrix instructions, Linux memory mapping and the model's
large lookup table cannot be counted as available M4 gains.

## Evidence and reproduction

Small raw artifacts, failures, scripts, exact payloads, power/swap endpoints,
source manifests and derived summaries are archived under
`results/2026-09-13/`. The nine large safetensors remain at
`target/prefill-investigation/capture-heterogeneous/traces/`, identified by the
archived SHA256 manifest; they are not embedded in Git. The broken native trace
bundle and the full external article are also excluded.

See [capture/evaluator instructions](README.md),
[GDN probe](../gdn_prefill/README.md), and
[attention probe](../attention_prefill/README.md). Reproducing the full model
runs requires the same checkpoint and native memory sampler used by the
[baseline harness](../ane_prefill/README.md). Runtime code changes are confined
to opt-in attention capture; no inference default or attention math is changed.

Verification: 21 CPU tests passed across the profile/measurement, sparse oracle,
attention and GDN suites. The release build containing the opt-in capture
succeeded and produced the nine real traces. Independent review found and
corrected the GDN paired-launch defect; old data remains explicitly invalidated.
The hardware timing campaign ended before final indexing and graph checks.

The final GitNexus index covers the large model file. Full-worktree and staged
change analyses returned without partial/truncated/error flags. The staged
scope reports 118 changed symbols and 23 affected flows with **critical** risk;
the full worktree also includes preexisting changes. This classification is
retained, not treated as an all-clear. Shared-forward capture edits were warned
about before editing and received independent review. No unrelated changes
are included in the experiment commit.
