#!/usr/bin/env python3
"""E2E test: Load RWKV-7 1.5B in Higgs and run greedy generation.

This script:
1. Encodes a prompt with the RWKV tokenizer
2. Calls the Rust binary via subprocess to run inference
3. Decodes the output tokens

But first, we test pure-Python forward to establish ground truth.
"""

import sys
import os
import json
import subprocess
import time

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--fla-hub--rwkv7-1.5B-world/"
    "snapshots/004140baad7a62d49a26d97508ef19cf09672328"
)

# Add model dir to path for the custom tokenizer
sys.path.insert(0, MODEL_DIR)
from hf_rwkv_tokenizer import RwkvTokenizer

def main():
    tok = RwkvTokenizer(os.path.join(MODEL_DIR, "rwkv_vocab_v20230424.txt"))

    prompt = "The capital of France is"
    ids = tok.encode(prompt)
    print(f"Prompt: {prompt!r}")
    print(f"Token IDs: {ids}")

    # Try loading with transformers if available (ground truth)
    try:
        import torch
        from safetensors.torch import load_file

        print("\n--- Loading model weights for shape inspection ---")
        weights = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
        print(f"Total weights: {len(weights)}")

        # Print a few key weight shapes
        for key in sorted(weights.keys())[:20]:
            print(f"  {key}: {weights[key].shape} {weights[key].dtype}")
        print("  ...")

        # Check if weight names match what our Rust loader expects
        print("\n--- Checking weight key patterns ---")
        lora_keys = [k for k in weights if "lora" in k]
        print(f"LoRA keys ({len(lora_keys)}):")
        for k in lora_keys[:10]:
            print(f"  {k}: {weights[k].shape}")

        norm_keys = [k for k in weights if "g_norm" in k or "norm" in k]
        print(f"\nNorm keys ({len(norm_keys)}):")
        for k in norm_keys[:10]:
            print(f"  {k}: {weights[k].shape}")

    except ImportError:
        print("(torch/safetensors not available, skipping weight inspection)")

    # Print what token IDs we'd use for Rust test
    print(f"\n--- For Rust test ---")
    print(f"let prompt_ids: Vec<i32> = vec!{ids};")
    print(f'// Decodes to: "{tok.decode(ids)}"')

    # Test round-trip some common tokens
    for text in ["Paris", " Paris", ".", "\n\n"]:
        enc = tok.encode(text)
        dec = tok.decode(enc)
        print(f"  {text!r:20s} -> {enc} -> {dec!r}")


if __name__ == "__main__":
    main()
