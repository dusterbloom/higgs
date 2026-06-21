#!/usr/bin/env python3
"""Generate the Bonsai-Q1 reference golden for the higgs 1-bit quality test.

The higgs oracle tests prove the GPU kernels match a CPU dequant of the *same*
1-bit weights. They do not prove the full forward is numerically faithful to the
upstream reference. This dumps reference logits + greedy continuations from the
PrismML mlx fork (the only mlx that runs bits=1) so the Rust test can assert the
higgs forward tracks it on fixed prompts.

Requires the PrismML mlx fork venv (mlx with 1-bit affine quantization):
    /Users/peppi/Dev/diffusion_bonsai/.venv  (mlx 0.31.x dev + mlx_lm)

Usage:
    <fork-venv>/bin/python3 gen_bonsai_q1_golden.py <model_dir> > \
        ../src/testdata/bonsai_q1_golden.json
"""
import json
import sys

import mlx.core as mx
import mlx_lm

PROMPTS = [
    "The capital of France is",
    "The opposite of hot is",
    "Water is made of hydrogen and",
    "The quick brown fox jumps over the lazy",
]
TOPK = 8
GREEDY_STEPS = 6


def last_logits(model, ids):
    out = model(mx.array([ids]))
    mx.eval(out)
    return out[0, -1]


def main():
    model_dir = sys.argv[1]
    model, tok = mlx_lm.load(model_dir)

    cases = []
    for prompt in PROMPTS:
        ids = list(tok.encode(prompt))
        logits = last_logits(model, ids)
        order = mx.argsort(-logits)[:TOPK].tolist()
        topk = [[int(i), float(logits[i].item())] for i in order]

        # Teacher-free greedy continuation (argmax, re-feed full sequence).
        cont, seq = [], list(ids)
        for _ in range(GREEDY_STEPS):
            nxt = int(mx.argmax(last_logits(model, seq)).item())
            cont.append(nxt)
            seq.append(nxt)

        cases.append(
            {
                "prompt": prompt,
                "input_ids": ids,
                "topk": topk,            # [[token_id, logit], ...] at last position
                "greedy": cont,          # next GREEDY_STEPS argmax token ids
                "greedy_text": tok.decode(cont),
            }
        )

    print(
        json.dumps(
            {
                "model": model_dir.rstrip("/").split("/")[-3]
                if "snapshots" in model_dir
                else model_dir,
                "mlx_version": mx.__version__,
                "topk": TOPK,
                "cases": cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
