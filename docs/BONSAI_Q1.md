# Bonsai-Q1

Bonsai-Q1 checkpoints are Qwen3-shaped models with MLX 1-bit affine
quantization metadata:

- `model_type = "qwen3"`
- `quantization.bits = 1`
- `quantization.group_size = 128`

The Higgs workspace stays on the pinned upstream `oxideai/mlx-rs` dependency.
That upstream revision does not yet include the MLX bits=1 affine Metal kernels,
so `higgs-engine` detects Bonsai-Q1 configs and returns an explicit unsupported
model error instead of routing them into the regular Qwen3 transformer loader.

The packed loader and engine live in `crates/higgs-models/src/bonsai_q1.rs` so
the Rust-side code can be reviewed independently. Runtime enablement should wait
until bits=1 affine quantization support lands upstream in the MLX dependency
chain.
