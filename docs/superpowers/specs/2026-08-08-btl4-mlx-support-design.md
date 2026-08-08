# BTL-4 Native Higgs Support Design

## Goal

Serve `badtheorylabs/BTL-4` through Higgs' native MLX inference engine while
reusing the existing Qwen3.5/Qwen3.6 MoE implementation. The supported artifact
is a text-only MLX safetensors conversion, initially using affine 4-bit weights
with group size 64. Direct loading of the source Transformers checkpoint and
native loading of the GGUF Compact edition are outside this first phase.

Success means that the converted model loads through the existing
`qwen3_5_moe` route, produces token-compatible results with MLX-LM on the same
converted artifact, preserves BTL-4's reasoning and XML tool-call behavior,
and completes deterministic multi-turn agent smokes without accumulated
reasoning causing repeated turns.

## Decision

BTL-4 is the lowest-risk native Higgs target among the evaluated models because
its decoder architecture is already implemented. Its `text_config` describes
the same hybrid Qwen3.5-MoE family Higgs uses for Qwen3.5-35B-A3B and
Qwen3.6-35B-A3B: 40 decoder layers, three linear-attention layers followed by
one full-attention layer, 256 routed experts, eight selected experts per token,
and a shared expert.

The source repository is not directly consumable by Higgs. It contains roughly
70 GB of BF16 Transformers-layout safetensors, including a vision tower,
individually named expert tensors, and wrapper prefixes that differ from the
MLX parameter tree. Higgs requires MLX safetensors. The design therefore puts a
conversion boundary before the runtime rather than adding Transformers and
GGUF compatibility to the model loader.

## Scope

The first phase includes:

1. A deterministic Transformers-to-MLX conversion recipe for BTL-4's text
   decoder.
2. A small synthetic compatibility fixture that proves every BTL-specific
   tensor rewrite without downloading the full checkpoint.
3. Loading the converted artifact through Higgs' existing `qwen3_5_moe` model
   path.
4. Correctness tests against MLX-LM using the identical converted weights.
5. Reasoning, tool-call, multi-turn, memory, and throughput validation.
6. Documentation of the validated model path and operational limits.

The first phase excludes vision input, raw Transformers weights at runtime,
GGUF/IQ2_XXS support, MTP, continuous batching, and claims about the full
262,144-token context window. Those capabilities require independent evidence
and are not prerequisites for useful text and agent serving.

## Artifact Conversion Boundary

The preferred converter is MLX-LM's supported Qwen3.5 conversion path. The
canonical starting recipe uses affine 4-bit quantization with group size 64 and
preserves the tokenizer, generation configuration, and BTL chat template.

Before the full conversion is attempted, a miniature checkpoint fixture must
cover the transformations that are easy to miss:

- discard `model.visual.*` and other vision-only tensors;
- rewrite `model.language_model.*` into the MLX language-model namespace;
- retain the untied top-level `lm_head`;
- stack individually stored expert gate, up, and down projections into the
  expert arrays expected by the MLX and Higgs MoE implementations;
- transpose unsanitized causal-convolution weights into MLX layout;
- apply the Qwen3.5 RMSNorm offset convention exactly once;
- retain float32 GDN state tensors such as `A_log` where required;
- exclude absent or unusable MTP weights without leaving initialized
  placeholders in the active forward path.

If the current MLX-LM converter does not perform one of these transformations
for BTL-4's precise checkpoint layout, the fallback is a narrowly scoped
conversion adapter. The fallback changes the offline artifact only. It does
not add BTL-specific branches to autoregressive inference.

The converted config retains top-level `model_type = "qwen3_5_moe"`, nested
`text_config`, rope parameters, special-token IDs, and the original chat
template. It adds the MLX quantization description emitted by the converter.

## Runtime Integration

The converted model follows the existing runtime data flow:

1. Higgs resolves the Hugging Face model ID or local directory.
2. The registry reads `model_type = "qwen3_5_moe"`.
3. `higgs-engine` routes the directory to
   `load_qwen3_5_moe_model`.
4. The Qwen3.5 loader flattens `text_config`, builds hybrid GDN/full-attention
   caches, constructs sparse MoE layers, loads the converted tensors, and fuses
   compatible GDN projections.
5. The normal `AnyModel::Qwen3Next` forward and cache paths serve generation.
6. The chat layer renders BTL-4's embedded template, the reasoning parser
   separates `<think>` content, and the tool parser converts the model's
   `<function=...><parameter=...>` XML into structured API tool calls.

No new model enum variant, architecture module, cache type, Metal kernel, or
batch-engine implementation is introduced. Any runtime edit must be justified
by a failing compatibility test that cannot be corrected at the conversion
boundary.

## Reasoning and Tool-Call Contract

BTL-4 requires prior reasoning to remain separate from visible assistant
content. The template can remove reasoning from older turns only when the API
and engine preserve it as reasoning content. A response that folds `<think>`
text into ordinary content may repeat prior work indefinitely during agent
loops.

Validation covers both complete and streaming responses. XML tool calls use
the form embedded in BTL-4's template:

```text
<tool_call>
<function=tool_name>
<parameter=argument_name>
argument value
</parameter>
</function>
</tool_call>
```

The parser must preserve multiline string values and use the request's tool
schema when coercing numbers, booleans, arrays, and objects. Tool results must
round-trip through the template without reintroducing old reasoning.

## Failure Handling

The supported boundary is explicit:

- A directory containing only Transformers BF16 weights fails with a message
  directing the user to the BTL-4 MLX conversion recipe.
- A GGUF-only directory fails as an unsupported artifact rather than being
  mistaken for a missing safetensors download.
- Missing expert stacks, unmatched active parameters, invalid GDN shapes, or a
  missing untied output head abort model loading before the server becomes
  ready.
- `higgs doctor` reports the converted artifact's estimated resident size and
  warns when the configured context or model approaches available memory.
- Unsupported continuous batching or MTP settings fail during configuration or
  startup, not during the first request.

The loader must not silently initialize unmatched parameters, silently use
four-bit defaults for dense source tensors, or ignore a partially converted
expert layer.

## Verification

Implementation follows test-first development. Unit and fixture tests verify:

- every source-to-MLX tensor-name rewrite;
- complete expert stacking and deterministic expert order;
- convolution orientation and RMSNorm adjustment;
- preservation of dense state tensors and output-head tying semantics;
- model registry and loader routing;
- rejection of raw Transformers and GGUF-only artifacts with actionable
  errors;
- complete and chunked XML tool-call parsing;
- reasoning separation across multiple assistant/tool turns.

Model-backed validation proceeds in gates:

1. MLX-LM loads the converted model and completes greedy generation.
2. Higgs loads the same artifact with no unmatched active parameters.
3. For fixed prompts and a fixed cache state, Higgs and MLX-LM agree on greedy
   token IDs and have bounded logit error appropriate to identical quantized
   weights.
4. A deterministic text suite covers factual answers, code generation, long
   reasoning, XML tool calls, tool responses, and at least one recovery turn
   after a failed tool result.
5. Context smokes run first at 8K and 32K. Larger contexts are enabled only
   after memory and cache growth are measured.
6. The final report records download size, converted size, peak resident
   memory, startup time, prompt throughput, decode throughput, and the exact
   model and converter revisions.

The feature is accepted only when the converted checkpoint passes all gates
without runtime BTL-specific model math. If converter defects prevent a valid
artifact, they are fixed offline before reconsidering the runtime boundary.

## Operational Rollout

The first documented configuration references a local converted directory.
Publishing a derived Hugging Face checkpoint is a separate release decision
that must preserve the source license, model card, attribution, conversion
parameters, hashes, and measured validation results.

After the local artifact is validated, it may be added to the cached-model
smoke matrix as a text-only `qwen3_5_moe` example. The documentation must state
that the full Transformers repository and BTL-4-Compact GGUF are not direct
Higgs inputs.

## Deferred Alternatives

`Akahsizrr/fuse-1-Lite-MLX` remains an independent candidate because it is
already a compact MLX artifact, but it requires a new `fuse3` architecture and
correct sparse expert augmentation on top of LFM2. It is not a fallback inside
this design.

`badtheorylabs/BTL-4-Compact` can run today through llama.cpp. Treating a
llama.cpp process as an external provider may be useful operationally, but it
is not native Higgs model support and does not justify adding GGUF and IQ2_XXS
to this phase.
