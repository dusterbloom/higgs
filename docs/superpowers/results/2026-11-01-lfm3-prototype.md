# Native LFM3 prototype validation

Date: 2026-11-01. Base: `91d4a5e0398c0ac12fb873f1e2f0a074fa44b5c5`.

## Result

Experimental native inference works with the official LiquidAI LFM3-1.7B MLX
4-bit checkpoint: all 594 tensors load, direct greedy generation is coherent,
OpenAI-compatible chat returns a correct Rome answer, and Italian streaming
returns coherent text followed by `[DONE]`. No running user service or saved
configuration was changed. The temporary smoke server was stopped.

The model was already present in
`~/.cache/lm-studio/models/LiquidAI/LFM3-1.7B-MLX-4bit`; the initial cache search
missed that location. A separate temporary download was made for this work.

Checkpoint revision: `d0ca73dcfa70ce18733754a492cdc1fb0123262d`.
Weight SHA-256: `1be5bf672013cd91aa62148f77c5eea8d2784a95845d5fe826e08c7b84823815`.
Weight bytes: 973,083,386.

## Correctness evidence

- Synthetic model tests cover prefill/decode equivalence across the local
  attention boundary, prefixed weights, finite logits, malformed dimensions,
  missing/wrong-shaped tensors, unsupported quantization, untied embeddings,
  and explicit-mask rejection before cache mutation.
- Adapter tests verify conservative capabilities and already-resolved config
  use. Engine tests verify config inspection and dedicated loader routing.
- Byte-golden template checks cover plain chat, tools, multi-turn history and
  agent messages against HF `apply_chat_template`. A streamed tool-parser
  fixture splits XML markup across token boundaries.
- Independent CPU PyTorch equations dequantize the official affine-q4 tensors;
  unpacking was checked against MLX dequantization. Float32 native one-shot
  and stepped execution match all tested top-1 predictions, with maximum mean
  KL 7.92e-8 on three probes of 10, 28 and 19 tokens. This is a small smoke
  corpus, not a quality benchmark or long-context certification.
- BF16 parity does **not** fully pass. Eleven of twelve schedule-aware gates
  pass; multilingual stepped execution has KL 1.379e-3 versus CPU BF16 drift
  6.234e-4, exceeding the unchanged 1.5x gate despite 100% top-1 agreement.
- RoPE BF16 trig-table rounding was corrected to match upstream semantics.
  RMSNorm microtests on real weights matched HF rounding exactly. Remaining
  schedule drift is dominated by MLX q4 QMM/QMV shape rounding: identical
  BF16 inputs produce mean absolute differences 1.31e-3–1.55e-3 between
  multi-row matmul and repeated single-row matvec. CPU dequantized GEMM is
  shape-invariant; fixed-QKV MLX SDPA drift is much smaller (1.21e-5).

See [parity output](2026-11-01-lfm3-parity.txt),
[MLX primitive output](2026-11-01-lfm3-kernel-mlx.txt), and
[CPU primitive output](2026-11-01-lfm3-kernel-torch.txt).

## Performance

Apple M4 (10 CPU cores), 32 GB unified memory, macOS 27.0, release build,
official 4-bit weights. Two uncontended runs, each with per-shape warmup,
exact-length tokenizer chat prompts, and 128 synchronized greedy decode
steps. Every hybrid-cache offset was checked. Earlier server-contended
measurements were discarded.

| Prompt tokens | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 128 | 660–903 | 62–79 |
| 512 | 971–993 | 52–68 |
| 2048 | 782–904 | 41–71 |

The benchmark uses repeated `x ` as prompt filler and reports direct-model
throughput, not representative task accuracy or API overhead. Significant
run variance is retained. See [raw benchmark output](2026-11-01-lfm3-benchmarks.txt).

## Reproduction

Use the official modeling source whose SHA is recorded in the parity script;
the script requires an explicit source path and verifies its bytes alongside
the checkpoint, config and chat template. Python reference dependencies are
PyTorch, safetensors, NumPy and Transformers; the reference runs on CPU.

```sh
python scripts/lfm3_parity.py ref /path/to/model /path/to/modeling_lfm3.py /tmp/lfm3-parity
python scripts/lfm3_parity.py ref-bf16 /path/to/model /path/to/modeling_lfm3.py /tmp/lfm3-parity
cargo run -p higgs-models --example lfm3_logits -- /path/to/model /tmp/lfm3-parity/ids.json /tmp/lfm3-parity/higgs_f32 f32
cargo run -p higgs-models --example lfm3_logits -- /path/to/model /tmp/lfm3-parity/ids.json /tmp/lfm3-parity/higgs
python scripts/lfm3_parity.py compare /tmp/lfm3-parity
python scripts/lfm3_kernel_rounding.py mlx /path/to/model /tmp/lfm3-kernels
python scripts/lfm3_kernel_rounding.py torch /path/to/model /tmp/lfm3-kernels
cargo run --release -p higgs-models --example lfm3_bench -- /path/to/model
HIGGS_LFM3_DIR=/path/to/model HIGGS_LFM3_PARITY=/tmp/lfm3-parity cargo test -p higgs-engine --test lfm3_template -- --include-ignored
```

GPU checks require host Metal access; the execution sandbox aborts during
device initialization. A release benchmark link in this environment needed
Xcode's `libclang_rt.osx.a` passed as a linker argument; ordinary release
Higgs builds succeeded without that workaround.

## Scope and broad checks

No batch, MTP, speculative, vision or TurboQuant support. Explicit masks and
packed/padded inputs fail closed. Local attention retains full KV storage.
256K context and retained-session semantics are unvalidated. End-to-end tool
execution is not certified by template/parser fixtures.

Broad regressions were compared with a clean checkout of the same base:

| Check | Prototype | Clean baseline |
| --- | --- | --- |
| Higgs library suite | 747 pass, 4 fail, 1 ignored | Same counts and same four failures |
| Engine library suite | 724 pass, 2 fail, 9 ignored | 722 pass, same two failures, 9 ignored |
| Clippy before new-module cleanup | 179 errors | 170 errors |
| New model Clippy after cleanup | No `lfm3.rs` diagnostics | Not applicable |
| `cargo fmt -p higgs -- --check` | Existing diffs in untouched Higgs files | Files match base; no LFM3 edits there |

The four baseline Higgs failures are capacity contract, canonical resident
identity, cancelled-load cleanup timeout, and post-load residency facts. The
two baseline engine failures are MLX execution-gate assertions. Model library
unit tests also have pre-existing MLX API compilation errors. These are not
claimed fixed by the LFM3 work. Focused new/related tests are run separately.

GitNexus rates shared registry/loader changes HIGH/CRITICAL. UNKNOWN verdicts
for new/unindexed helpers were followed by text confirmation. No commit or
remote publication is part of this prototype.
