# Model-aware fast defaults

## Goal

Nightly should give a user the best validated experience for a supported model
after selecting its path or Hugging Face ID. Normal use must not require
Escha-, MLX-, or thinking-related environment variables.

## Decisions

### Thinking policy

An omitted request defaults to non-thinking for every model. An explicit
request remains authoritative: `chat_template_kwargs.enable_thinking = true`
or a non-`none` reasoning effort enables thinking when the loaded engine
supports it. An explicit `false` or `reasoning.effort = "none"` disables it.

The implementation must distinguish a model's capability to use thinking from
the default selected for an omitted request. It must not use an engine-level
hard-disable as the default, because that would make an explicit opt-in fail.

### Escha policy

The resolver selects the fastest validated representation from the checkpoint
identity and structural configuration:

- `Qwen3.6-35B-A3B-Escha-W2` and compatible Escha MoE checkpoints keep their
  experts in native trellis Metal form. This is already the default; the
  affine path remains an explicit diagnostic fallback.
- The exact dense Qwen3.8-27B Escha structural profile converts trellis
  projections to affine Q2 by default. Its existing shape-gated SIMD decode
  and separate gate/up layout then select automatically.
- Other dense Escha checkpoints retain affine Q4 unless a future validated
  structural profile is added.

`HIGGS_ESCHA_NATIVE`, `HIGGS_ESCHA_AFFINE_BITS`, and
`HIGGS_BONSAI_Q2_SIMD` remain diagnostic and benchmark overrides. They do not
belong in normal user instructions.

### Runtime policy

The existing automatic MLX profile remains model-size-aware and resolves to
throughput for large and huge models. Users may override it in config or the
CLI. `HIGGS_ESCHA_TRELLIS_GEMM=1` remains experimental because it changes only
large-prefill expert execution and lacks the required full-model promotion
evidence; it is not made automatic in this change.

## Precedence

1. Explicit request reasoning controls thinking when supported.
2. Explicit model `generation_defaults.enable_thinking` controls an omitted
   request.
3. Otherwise, the nightly model policy chooses non-thinking.

Checkpoint structural matching is exact and conservative. A near match never
inherits the Qwen3.8 Q2 default.

## Verification

- Add red/green unit coverage for the non-thinking default, explicit thinking
  opt-in, and explicit configuration precedence.
- Add red/green unit coverage that the exact dense Qwen3.8 Escha profile
  defaults to Q2, while all other Escha profiles stay Q4.
- Run focused route/config/model tests, formatting, Clippy, and a release
  model-load/generation check for both local Escha checkpoints without Escha
  environment variables.
- Update the README to describe automatic defaults and reserve environment
  variables for diagnostics.
