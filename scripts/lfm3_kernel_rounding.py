#!/usr/bin/env python3
"""Reproduce MLX BF16 q4/SDPA shape-dependent rounding on real LFM3 weights.

Run the two stages with environments that provide the corresponding backend::

    python scripts/lfm3_kernel_rounding.py mlx <model_dir> <work_dir>
    python scripts/lfm3_kernel_rounding.py torch <model_dir> <work_dir>

The MLX stage writes fixed inputs and operator outputs to ``work_dir``. The
Torch stage independently unpacks the affine-q4 weights, runs CPU GEMMs and
compares both schedules. Neither stage executes Higgs model code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOKENS, HIDDEN = 19, 2048
PROJECTIONS = {
    "q_proj": "model.layers.2.self_attn.q_proj",
    "conv_in": "model.layers.0.conv.in_proj",
    "ffn_w1": "model.layers.0.feed_forward.w1",
}


def settings(model_dir: Path) -> tuple[int, int]:
    config = json.loads((model_dir / "config.json").read_text())
    quant = config.get("quantization", config.get("quantization_config", {}))
    group_size, bits = int(quant.get("group_size", 0)), int(quant.get("bits", 0))
    if (group_size, bits, quant.get("mode")) != (64, 4, "affine"):
        raise ValueError("expected the official affine q4/group-64 checkpoint")
    return group_size, bits


def mlx_metrics(mx, label: str, left, right) -> None:
    left, right = left.astype(mx.float32), right.astype(mx.float32)
    difference = mx.abs(left - right)
    cosine = mx.sum(left * right) / mx.sqrt(mx.sum(left * left) * mx.sum(right * right))
    mx.eval(difference, cosine)
    print(
        f"{label}: unequal={int(mx.sum(left != right).item())}/{left.size} "
        f"max_abs={float(mx.max(difference).item()):.8g} "
        f"mean_abs={float(mx.mean(difference).item()):.8g} "
        f"cosine={float(cosine.item()):.10f}"
    )


def run_mlx(model_dir: Path, work_dir: Path) -> None:
    import mlx.core as mx

    group_size, bits = settings(model_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "lfm3_kernel_rounding.safetensors"
    weights = mx.load(str(model_dir / "model.safetensors"))
    prefix = "language_model." if any(key.startswith("language_model.") for key in weights) else ""
    values = mx.sin(mx.arange(TOKENS * HIDDEN, dtype=mx.float32) * 0.0137)
    values = (values.reshape(1, TOKENS, HIDDEN) * 0.25).astype(mx.bfloat16)
    saved = {"input": values}

    def qmm(name: str, inputs):
        base = prefix + name
        return mx.quantized_matmul(
            inputs,
            weights[base + ".weight"],
            weights[base + ".scales"],
            weights[base + ".biases"],
            group_size=group_size,
            bits=bits,
        )

    for label, name in PROJECTIONS.items():
        one = qmm(name, values)
        stepped = mx.concatenate(
            [qmm(name, values[:, index : index + 1]) for index in range(TOKENS)],
            axis=1,
        )
        mx.eval(one, stepped)
        mlx_metrics(mx, f"mlx qmm {label} one-vs-step", one, stepped)
        saved[label + "_mlx_one"] = one
        saved[label + "_mlx_step"] = stepped

    q = qmm("model.layers.2.self_attn.q_proj", values).reshape(1, TOKENS, 16, 128)
    k = qmm("model.layers.2.self_attn.k_proj", values).reshape(1, TOKENS, 4, 128)
    v = qmm("model.layers.2.self_attn.v_proj", values).reshape(1, TOKENS, 4, 128)
    q, k, v = (tensor.transpose(0, 2, 1, 3) for tensor in (q, k, v))
    one = mx.fast.scaled_dot_product_attention(q, k, v, scale=128**-0.5, mask="causal")
    stepped = mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[:, :, index : index + 1],
                k[:, :, : index + 1],
                v[:, :, : index + 1],
                scale=128**-0.5,
            )
            for index in range(TOKENS)
        ],
        axis=2,
    )
    mx.eval(q, k, v, one, stepped)
    mlx_metrics(mx, "mlx sdpa causal-one-vs-cache-step", one, stepped)
    saved.update(
        {"q": q, "k": k, "v": v, "sdpa_mlx_one": one, "sdpa_mlx_step": stepped}
    )
    mx.save_safetensors(str(output), saved)
    print(f"saved {output}")


def torch_metrics(torch, label: str, left, right) -> None:
    left, right = left.float(), right.float()
    difference = (left - right).abs()
    cosine = torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0)
    print(
        f"{label}: unequal={int((left != right).sum())}/{left.numel()} "
        f"max_abs={difference.max().item():.8g} mean_abs={difference.mean().item():.8g} "
        f"cosine={cosine.item():.10f}"
    )


def dequant(torch, reader, prefix: str, name: str, group_size: int):
    base = prefix + name
    packed = reader.get_tensor(base + ".weight").to(torch.int64)
    scales = reader.get_tensor(base + ".scales").float()
    biases = reader.get_tensor(base + ".biases").float()
    shifts = torch.arange(8, dtype=torch.int64) * 4
    quantized = ((packed[..., None] >> shifts) & 15).flatten(-2).float()
    return (
        quantized * scales.repeat_interleave(group_size, dim=-1)
        + biases.repeat_interleave(group_size, dim=-1)
    ).bfloat16()


def run_torch(model_dir: Path, work_dir: Path) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    group_size, _ = settings(model_dir)
    data = load_file(work_dir / "lfm3_kernel_rounding.safetensors")
    inputs = data["input"]
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        prefix = "language_model." if any(key.startswith("language_model.") for key in keys) else ""
        for label, name in PROJECTIONS.items():
            weight = dequant(torch, reader, prefix, name, group_size)
            one = torch.nn.functional.linear(inputs, weight)
            stepped = torch.cat(
                [
                    torch.nn.functional.linear(inputs[:, index : index + 1], weight)
                    for index in range(TOKENS)
                ],
                dim=1,
            )
            torch_metrics(torch, f"torch qmm {label} one-vs-step", one, stepped)
            torch_metrics(torch, f"torch-vs-mlx {label} one", one, data[label + "_mlx_one"])
            torch_metrics(
                torch, f"torch-vs-mlx {label} step", stepped, data[label + "_mlx_step"]
            )

    q, k, v = data["q"], data["k"], data["v"]
    k_heads, v_heads = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
    scores = torch.matmul(q, k_heads.transpose(2, 3)) * (128**-0.5)
    causal = torch.arange(TOKENS)[None, :] <= torch.arange(TOKENS)[:, None]
    probabilities = torch.softmax(
        scores.masked_fill(~causal[None, None], float("-inf")), dim=-1, dtype=torch.float32
    ).bfloat16()
    one = torch.matmul(probabilities, v_heads)
    parts = []
    for index in range(TOKENS):
        step_scores = torch.matmul(
            q[:, :, index : index + 1], k_heads[:, :, : index + 1].transpose(2, 3)
        ) * (128**-0.5)
        step_probabilities = torch.softmax(
            step_scores, dim=-1, dtype=torch.float32
        ).bfloat16()
        parts.append(torch.matmul(step_probabilities, v_heads[:, :, : index + 1]))
    stepped = torch.cat(parts, dim=2)
    torch_metrics(torch, "torch sdpa causal-one-vs-cache-step", one, stepped)
    torch_metrics(torch, "torch-vs-mlx sdpa one", one, data["sdpa_mlx_one"])
    torch_metrics(torch, "torch-vs-mlx sdpa step", stepped, data["sdpa_mlx_step"])


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["mlx", model_dir, work_dir]:
            run_mlx(Path(model_dir), Path(work_dir))
        case ["torch", model_dir, work_dir]:
            run_torch(Path(model_dir), Path(work_dir))
        case _:
            raise SystemExit(__doc__)
