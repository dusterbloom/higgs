# Qwen3.5 Online Residual ROM Probe Design

## Goal

Determine whether the activation trajectories that feed the expensive dense
linear projections during batch-one Qwen3.5 decode admit a useful causal
rank-64 representation. The first experiment is a measurement-only go/no-go
gate. It does not replace any matrix multiplication or change generated model
outputs.

The experiment passes only if a basis constructed exclusively from preceding
decode tokens retains at least 99% of the measured activation energy at rank 64
in each of the code, prose, and reasoning domains. Rank-32 and rank-128 results,
per-layer distributions, and worst cases are reported even when the aggregate
gate passes.

## Scope

The first target is the locally cached Qwen3.5-9B MLX 4-bit checkpoint:

`/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit`

The probe covers the dense Qwen3.5/Qwen3Next autoregressive path. It does not
alter, disable, or benchmark against the Bonsai Q1 row4, Ternary Q2 row2,
dSpark/DFlash, MTP, ANE, or Escha native trellis paths. Those paths remain the
production comparison targets for later phases.

This phase does not build reduced weights, a streaming SVD updater, or an
approximate forward pass. It also does not make a runtime speedup claim from
spectral measurements alone.

## Why the Probe Measures Linear Inputs

A low-rank residual stream is useful for inference only when it induces a
low-rank trajectory at the inputs of expensive linear operations. Qwen3.5
applies learned RMS normalization and contains nonlinear attention and MLP
intermediates, so capturing only the final hidden state of each layer would be
an incomplete test.

For every decoder layer and measured token, the probe captures:

1. `attention_in`: the normalized residual consumed by the layer's attention
   or GDN input projections.
2. `mlp_in`: the post-attention normalized residual shared by the MLP gate and
   up projections.
3. `post_layer`: the post-MLP residual, retained to test the original residual
   trajectory hypothesis and to connect adjacent layers.

The first two sites directly cover shared inputs to a large fraction of dense
weight traffic. If they pass, a follow-up probe must capture the attention
output-projection input and MLP down-projection input before any end-to-end
speed prototype is authorized. Nonlinear intermediate trajectories cannot be
assumed to share the residual stream's rank.

## Activation-Capture Boundary

Capture is diagnostic and explicitly requested for one forward pass through a
thread-local request/take interface, following the existing
`diag_request_hidden_capture` pattern. When no request is active, the layer
loop performs one predictable false branch and allocates no capture buffers.

When requested, the next Qwen3Next forward materializes the three activation
sites as owned `f32` vectors after each layer. The benchmark takes the completed
capture immediately. A missing capture, an unexpected layer count, a mismatched
hidden dimension, duplicate sites, or non-finite activation aborts the probe
with a descriptive error. Partial trajectories are never analyzed.

The existing hidden-difference diagnostics keep their current tuple format and
semantics. The ROM probe uses separate types and storage so it cannot silently
change warm-cache debugging behavior in `higgs-engine`.

## Workload

The ignored model-backed probe uses the checkpoint's `tokenizer.json` and nine
fixed prompts: three code prompts, three prose prompts, and three multi-step
reasoning prompts. Prompts request sufficiently long continuations and include
stable labels in the report. Decode is greedy and deterministic.

Each prompt has two phases:

- 128 warmup decode tokens used to construct the first causal basis.
- 128 measured decode tokens used only after their predictions are made.

The prompt prefill is not included in the trajectory. The probe continues to
the configured token limit even if an end-of-sequence token is generated so
every prompt produces the same matrix dimensions; the report records generated
EOS positions. Environment variables may reduce prompt count or token counts
for smoke testing, but the canonical pass/fail report requires all nine prompts
and the 128+128 schedule.

## Causal and Oracle Analysis

For a layer/site pair, let `X` be the preceding activation window with shape
`[W, D]` and `Y` the subsequent held-out activations. The canonical window is
`W = 128`.

The analyzer does not form a `D x D` covariance matrix. It computes the smaller
Gram matrix `G = X X^T`, performs a symmetric eigendecomposition, and derives
ranked energy from its non-negative eigenvalues. Held-out projection energy is
computed without materializing a `D x r` basis:

`C = Y X^T`, followed by projection through the eigenvectors of `G` and inverse
square roots of the retained eigenvalues.

This calculation is batched across layers with MLX. Eigenvalues smaller than a
scale-relative numerical floor are discarded rather than inverted. All energy
ratios are clamped only for floating-point roundoff and the unclamped extrema
are retained in diagnostic output.

The report contains two views:

1. **Oracle window spectrum:** cumulative rank-32/64/128 energy over the full
   256-token trajectory. This describes intrinsic trajectory dimension but is
   not sufficient to pass the experiment because it uses future samples.
2. **Causal held-out retention:** a basis built from the 128 warmup tokens is
   evaluated on the next 128 tokens without incorporating them. This is the
   authoritative go/no-go measurement.

A later refresh-frequency sweep may rebuild the basis every 8, 16, or 32
tokens. It is deliberately excluded from the first gate so a frequently
refreshed basis cannot hide poor temporal generalization.

## Metrics and Decision Rule

For ranks 32, 64, and 128, the JSON report records:

- retained Frobenius energy for each prompt, activation site, and layer;
- relative reconstruction energy `1 - retained_energy`;
- domain-wide weighted aggregate retention;
- median, p05, and worst-layer retention by domain and activation site;
- the layer and prompt responsible for every worst case;
- oracle effective ranks needed for 95%, 99%, and 99.9% energy;
- the model path, model architecture dimensions, prompt labels, token counts,
  window size, git commit, and active ROM-probe configuration.

The rank-64 gate passes when causal held-out retained energy is at least 0.99
for every domain-wide aggregate at both `attention_in` and `mlp_in`.
`post_layer` is reported but is not allowed to compensate for a failing linear
input site. The result also highlights any individual layer below 0.95; such a
layer does not silently fail the aggregate gate, but it becomes a mandatory
full-precision correction candidate in a later design.

Console output is a compact summary table. The canonical machine-readable
artifact is JSON written beneath a user-selected output directory. The probe
does not write multi-gigabyte raw activation dumps by default.

## Implementation Boundaries

The feature is split into three units:

1. A small Qwen3Next diagnostic capture type and request/take API located near
   the existing hidden-state diagnostics.
2. Pure trajectory validation and metric aggregation helpers that can be tested
   on small synthetic matrices without loading model weights.
3. An ignored Qwen3.5-9B probe that tokenizes prompts, performs deterministic
   decode, batches the Gram analysis, prints the summary, and writes JSON.

No command-line server surface or persistent engine configuration is added.
The model-backed test is invoked explicitly with `HIGGS_MODEL_PATH` and an
output directory. Production builds retain no active file writer or background
trace worker.

## Testing

Implementation follows test-first development.

Unit tests establish that:

- requested captures are one-shot and do not overwrite the existing hidden
  diagnostic slot;
- trajectory assembly rejects missing layers, duplicate sites, inconsistent
  dimensions, short windows, and non-finite values;
- a synthetic exact-rank matrix reports the expected effective rank and energy;
- causal analysis never includes the first held-out sample in its basis;
- rank metrics are monotonic and zero-eigenvalue directions remain finite;
- domain aggregation is weighted by activation energy rather than averaging
  percentages from unequal trajectories;
- the pass/fail gate requires all three domains and both linear-input sites.

The ignored integration probe first runs in a reduced smoke configuration, then
in the canonical nine-prompt configuration. Existing Qwen3Next unit tests and
the dense decode benchmark are rerun to verify that inactive capture has not
changed logits or decode behavior.

## Follow-Up Gate

Passing this experiment authorizes a second measurement phase for
`o_proj`/output-projection and `down_proj` inputs. Only if those trajectories
also show useful causal rank does the project design a reduced-weight cache
`W U` and compare it against:

- Qwen3.5-9B Q4 autoregressive decode;
- Qwen3.5-4B Q4 as the principal small-model result;
- Ternary-Bonsai Q2 with row2 M=5 verification and head argmax enabled;
- Bonsai Q1 row4 dSpark and Escha native W2 as strong production controls.

Any speed claim must include basis-update time, cache memory, correction
forwards, acceptance behavior when speculation is active, quality loss, and
end-to-end tokens per second.
