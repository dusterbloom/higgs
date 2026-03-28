#!/usr/bin/env python3
"""Graft MTP weights from trevon/Qwen3.5-27B-MLX-MTP into a local MLX model.

Downloads the standalone mtp.safetensors (bf16), quantizes large projection
tensors to 4-bit/group_size=64/affine using MLX, and adds them as a new shard
to the target model directory.
"""

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.numpy import save_file

# --- Config ---
MTP_REPO = "trevon/Qwen3.5-27B-MLX-MTP"
MTP_FILE = "mtp.safetensors"
TARGET_DIR = Path(
    "/Users/peppi/.cache/lm-studio/models/mlx-community/"
    "Qwen3.5-27B-4bit"
)
GROUP_SIZE = 64
BITS = 4
KEY_PREFIX = "language_model."

# Tensors that are too small / shouldn't be quantized (norms, biases, etc.)
# mtp.fc.weight MUST stay in fp16 — quantizing it destroys MTP prediction quality.
# mlx-lm's quant_predicate explicitly excludes "mtp.fc" from quantization.
SKIP_QUANTIZE = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}


def load_bf16_tensors(path: str) -> dict[str, np.ndarray]:
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            tensors[key] = t.to(torch.float16).numpy()
    return tensors


def quantize_tensor(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize a 2D bf16/fp32 tensor using MLX affine 4-bit."""
    x = mx.array(arr.astype(np.float16))
    w_q, scales, biases = mx.quantize(x, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    mx.eval(w_q, scales, biases)
    return np.array(w_q), np.array(scales), np.array(biases)


def main():
    # 1. Download MTP tensors
    print(f"Downloading {MTP_FILE} from {MTP_REPO}...")
    mtp_path = hf_hub_download(MTP_REPO, MTP_FILE)
    print(f"  -> {mtp_path}")

    # 2. Load bf16 tensors
    raw = load_bf16_tensors(mtp_path)
    print(f"Loaded {len(raw)} MTP tensors")

    # 3. Quantize and build output dict
    out: dict[str, np.ndarray] = {}
    for key, arr in sorted(raw.items()):
        prefixed = KEY_PREFIX + key
        if key in SKIP_QUANTIZE:
            out[prefixed] = arr.astype(np.float16)
            print(f"  {prefixed}: {arr.shape} float16 (kept)")
        else:
            stem = prefixed.rsplit(".weight", 1)[0]
            w_q, scales, biases = quantize_tensor(arr)
            out[stem + ".weight"] = w_q
            out[stem + ".scales"] = scales
            out[stem + ".biases"] = biases
            print(
                f"  {stem}.weight: {arr.shape} -> q{BITS} "
                f"[{w_q.shape}, {scales.shape}, {biases.shape}]"
            )

    # 4. Save as new shard
    index_path = TARGET_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    existing_shards = {v for v in index["weight_map"].values()}
    n_shards = len(existing_shards)
    new_shard = f"model-{n_shards + 1:05d}-of-{n_shards + 1:05d}.safetensors"

    # Rename existing shards to reflect new total
    old_to_new = {}
    for i in range(1, n_shards + 1):
        old_name = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{n_shards + 1:05d}.safetensors"
        old_to_new[old_name] = new_name

    # Rename files on disk
    for old, new in old_to_new.items():
        src = TARGET_DIR / old
        dst = TARGET_DIR / new
        if src.exists() and old != new:
            print(f"  Renaming {old} -> {new}")
            shutil.move(str(src), str(dst))

    # Update index weight_map with renamed shards
    new_weight_map = {}
    for k, v in index["weight_map"].items():
        new_weight_map[k] = old_to_new.get(v, v)

    # Add MTP keys
    for k in out:
        new_weight_map[k] = new_shard

    # Save new shard
    shard_path = TARGET_DIR / new_shard
    print(f"Saving {len(out)} tensors to {shard_path}...")
    save_file(out, str(shard_path))

    # Update index
    index["weight_map"] = dict(sorted(new_weight_map.items()))
    # Update metadata total size
    total_size = sum(
        arr.nbytes for arr in out.values()
    )
    if "metadata" in index and "total_size" in index["metadata"]:
        index["metadata"]["total_size"] += total_size

    # Backup old index
    backup = TARGET_DIR / "model.safetensors.index.json.bak"
    shutil.copy2(str(index_path), str(backup))
    print(f"  Backed up index to {backup.name}")

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    print(f"\nDone! Grafted {len(raw)} MTP tensors ({len(out)} quantized keys) into {TARGET_DIR.name}")
    print(f"New shard: {new_shard}")

    # Verify
    with open(index_path) as f:
        idx = json.load(f)
    mtp_keys = [k for k in idx["weight_map"] if "mtp" in k]
    print(f"Verification: {len(mtp_keys)} MTP keys in index")


if __name__ == "__main__":
    main()
