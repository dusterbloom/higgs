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

Single-token decode stays packed. Embedding lookup and multi-token prefill
dequantize the selected matrix to the input dtype before using regular MLX
matmul.

Qwen3.5 checkpoints packaged as multimodal models currently load the text
backbone only. Their vision tower is not exposed by Higgs, so image input remains
unsupported for those checkpoints.
