# Bonsai-Q1

Higgs supports MLX affine 1-bit checkpoints with `quantization.bits = 1` and
`quantization.group_size = 128` on the pinned upstream `oxideai/mlx-rs`
revision. Upstream MLX does not ship the required 1-bit affine kernels, so Higgs
provides runtime JIT Metal kernels for packed matvec and dequantization.

Two layouts are supported:

- Qwen3-shaped Bonsai checkpoints use the dedicated packed engine in
  `crates/higgs-models/src/bonsai_q1.rs`.
- Qwen3.5 hybrid checkpoints, including Bonsai-27B, use the existing
  `qwen3_next` architecture with its affine 1-bit operations dispatched to the
  same Higgs Metal kernels.

Single-token decode and narrow multi-token forwards stay packed. For Qwen3.5,
the packed Metal path covers up to 8 flattened rows by default, including the
small verifier batches used by speculative decoding. Wider prefill inputs
dequantize the selected matrix to the input dtype before using regular MLX
matmul. Set `HIGGS_BONSAI_QMM_MAX_ROWS=0` to disable the narrow packed path, or
raise it up to 64 for A/B testing.

For Qwen3.5 Q1 checkpoints, the loader validates every affine scale/bias pair.
When a tensor is exactly symmetric (`bias = -scale / 2`), Higgs releases its
bias array and derives the bias in the Metal kernel. Any non-symmetric tensor
keeps the general affine path. Set `HIGGS_BONSAI_SYMMETRIC_Q1=0` to retain all
bias tensors for A/B debugging.

Qwen3.5 checkpoints packaged as multimodal models currently load the text
backbone only. Their vision tower is not exposed by Higgs, so image input remains
unsupported for those checkpoints.

## Bonsai-27B dSpark

Higgs can run Prism's public
[`Bonsai-27B-dspark-Q4_1.gguf`](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
through the DFlash recurrent tape verifier.

A ready-to-run target-head conversion is published as
[`peppi314/Bonsai-27B-dSpark-MLX-4bit`](https://huggingface.co/peppi314/Bonsai-27B-dSpark-MLX-4bit).
To reproduce it from the source GGUF:

```bash
python scripts/convert_dspark_gguf.py \
  Bonsai-27B-dspark-Q4_1.gguf Bonsai-27B-dspark-mlx \
  --reuse-target-head
```

The converter losslessly repacks GGUF Q4_1 blocks into MLX affine Q4/group-32.
It omits the duplicate token embedding because the published dSpark and target
Q1 embeddings are bit-identical. `--reuse-target-head` also omits dSpark's
frozen Q4 output copy and uses the resident Bonsai Q1 head for proposals. This
reduces the sidecar tensors from about 1534 MiB to 776 MiB. The two output heads
are not numerically identical, so omit this flag for Prism-head fidelity;
target verification keeps generated tokens distribution-exact in either mode.

Load the converted directory as the configured `draft_model`, or set
`HIGGS_DFLASH_PATH`. dSpark always runs its trained four-position non-causal
trunk: generic DFlash adaptive sizing, wall-clock flooring, and early-exit
verification are disabled for it. The runtime validates target hidden size,
vocabulary size, and tap-layer indices at load.

Performance knobs:

- `HIGGS_DSPARK_DRAFT_CAP=1..4` caps vocabulary-head and verify positions while
  retaining the full trained trunk. The target-head conversion defaults to 4;
  the full Prism-head conversion defaults to 3 on Apple.
- `HIGGS_DSPARK_TARGET_HEAD=1` lets a full conversion use the target Q1 head.
  A sidecar converted with `--reuse-target-head` always uses it.

In two final back-to-back local 64-token greedy, thinking-enabled code checks,
the 776 MiB target-head sidecar committed 3.94 tokens per round, produced
byte-identical output at the exact length limit, and measured 1.03x to 1.08x
end-to-end versus plain Bonsai-27B autoregressive decoding. An earlier
thinking-disabled run committed 4.50 tokens per round and reached 1.18x; it is
not an apples-to-apples acceptance comparison. Fresh-engine prefill, lazy graph
materialization, workload, and laptop thermals all move short-run tok/s, so use
the paired harness in `docs/benchmarking.md` with identical thinking settings
and warmed repetitions before treating these as headline numbers.
