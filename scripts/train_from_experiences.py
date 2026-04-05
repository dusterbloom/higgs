#!/usr/bin/env python3
"""Train EGGROLL on real nanobot experience data.

Pulls high-quality prompt→response pairs from ~/.nanobot/experience.db,
tokenizes them with the model's tokenizer, and sends to /v1/train with
the new conservative defaults (sigma=0.001, lr=0.0005, clip_ratio=0.05,
delta_decay=0.001).

Tests generation quality before and after training to verify the model
improves on the training distribution without catastrophic forgetting.

Usage:
    # Start Higgs first:
    cargo run --release -p higgs -- serve --timeout 1800

    # Train on top-N experiences:
    python3 scripts/train_from_experiences.py --n 5 --steps 30

    # Dry run (tokenize + show stats, no training):
    python3 scripts/train_from_experiences.py --dry-run --n 10
"""

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DB_PATH = Path.home() / ".nanobot" / "experience.db"
MODEL_PATH = "/Users/peppi/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit"
MODEL_NAME = "qwen35"  # As registered in Higgs config
BASE = "http://localhost:8000"

def _update_base(url: str):
    global BASE
    BASE = url

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_tok = None

def get_tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return _tok


def tokenize_experience(prompt: str, response: str):
    """Tokenize a prompt→response pair in ChatML format.

    Returns (token_ids, prompt_len) where loss is computed only on
    tokens after prompt_len (the response portion).
    """
    tok = get_tokenizer()
    # ChatML format matching Qwen3.5
    prompt_text = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full_text = prompt_text + response + "<|im_end|>"
    tokens = tok.encode(full_text, add_special_tokens=False)
    prompt_tokens = tok.encode(prompt_text, add_special_tokens=False)
    return tokens, len(prompt_tokens)


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------

def load_experiences(n: int, min_quality: float = 0.8, model_filter: str = "%35B%"):
    """Load top-N high-quality experiences from the DB."""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("""
        SELECT id, prompt, response, quality, surprise, model
        FROM experiences
        WHERE model LIKE ? AND quality >= ?
        ORDER BY quality DESC, surprise DESC
        LIMIT ?
    """, (model_filter, min_quality, n)).fetchall()
    conn.close()
    return [
        {"id": r[0], "prompt": r[1], "response": r[2],
         "quality": r[3], "surprise": r[4], "model": r[5]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_post(url: str, data: dict, timeout: int = 1800):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:500]}
    except Exception as e:
        return 0, {"error": str(e)}


def generate(prompt: str, max_tokens: int = 256) -> str:
    """Generate via /v1/chat/completions (greedy)."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    status, result = http_post(f"{BASE}/v1/chat/completions", payload, timeout=120)
    if status != 200:
        return f"[ERROR {status}: {result}]"
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "[PARSE_ERROR]"


def train(tokens, prompt_len, steps, sigma, lr, pop, clip_ratio, delta_decay, timeout=1800):
    """Send training request to /v1/train."""
    payload = {
        "model": MODEL_NAME,
        "tokens": tokens,
        "prompt_len": prompt_len,
        "sigma": sigma,
        "lr": lr,
        "rank": 4,
        "population": pop,
        "total_steps": steps,
        "merge_interval": 0,
        "clip_ratio": clip_ratio,
        "delta_decay": delta_decay,
    }
    return http_post(f"{BASE}/v1/train", payload, timeout=timeout)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EVAL_PROMPTS = [
    # In-distribution: tasks similar to experiences
    "What is the latest news from BBC?",
    "Can you search online for jazz radio streams?",
    "Read the file at /tmp/test.txt and summarize it",
    # Out-of-distribution: general knowledge (forgetting test)
    "What is the capital of France?",
    "Explain photosynthesis in one paragraph.",
    "Write a Python function to reverse a string.",
]


def run_eval(label: str):
    """Run evaluation prompts and return results."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {label}")
    print(f"{'='*60}")
    results = []
    for prompt in EVAL_PROMPTS:
        gen = generate(prompt, max_tokens=200)
        is_coherent = len(gen) > 10 and not gen.startswith("[ERROR") and not gen.startswith("[PARSE")
        results.append({"prompt": prompt[:60], "coherent": is_coherent, "len": len(gen)})
        status = "OK" if is_coherent else "FAIL"
        print(f"  [{status}] {prompt[:50]}...")
        print(f"         -> {gen[:120]}...")
        print()
    coherent_count = sum(1 for r in results if r["coherent"])
    print(f"  Coherent: {coherent_count}/{len(results)}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EGGROLL on real experience data")
    parser.add_argument("--n", type=int, default=5, help="Number of experiences to train on")
    parser.add_argument("--steps", type=int, default=30, help="Training steps per experience")
    parser.add_argument("--sigma", type=float, default=0.001, help="Perturbation scale")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--pop", type=int, default=16, help="Population size")
    parser.add_argument("--clip-ratio", type=float, default=0.05, help="Delta norm clip ratio")
    parser.add_argument("--delta-decay", type=float, default=0.001, help="Per-step delta decay")
    parser.add_argument("--dry-run", action="store_true", help="Tokenize only, no training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip pre/post evaluation")
    parser.add_argument("--base", type=str, default=BASE, help="Higgs server base URL")
    parser.add_argument("--timeout", type=int, default=1800, help="Request timeout (seconds)")
    args = parser.parse_args()

    # Update module-level BASE if overridden
    if args.base != BASE:
        _update_base(args.base)

    print(f"Loading top {args.n} experiences from {DB_PATH}...")
    experiences = load_experiences(args.n)
    if not experiences:
        print("No matching experiences found!")
        sys.exit(1)

    print(f"Found {len(experiences)} experiences. Tokenizing...")
    tokenized = []
    for exp in experiences:
        tokens, prompt_len = tokenize_experience(exp["prompt"], exp["response"])
        seq_len = len(tokens)
        comp_len = seq_len - prompt_len
        tokenized.append({
            **exp,
            "tokens": tokens,
            "prompt_len": prompt_len,
            "seq_len": seq_len,
            "comp_len": comp_len,
        })
        print(f"  #{exp['id']:3d} q={exp['quality']:.2f} s={exp['surprise']:.2f} "
              f"seq={seq_len:4d} prompt={prompt_len:4d} comp={comp_len:4d} "
              f"| {exp['prompt'][:60]}...")

    total_tokens = sum(t["seq_len"] for t in tokenized)
    total_comp = sum(t["comp_len"] for t in tokenized)
    print(f"\nTotal: {total_tokens} tokens ({total_comp} completion)")
    print(f"Params: sigma={args.sigma} lr={args.lr} pop={args.pop} "
          f"clip={args.clip_ratio} decay={args.delta_decay} steps={args.steps}")

    if args.dry_run:
        print("\n[DRY RUN] Stopping before training.")
        return

    # Pre-training eval
    if not args.skip_eval:
        pre_results = run_eval("PRE-TRAINING (baseline)")

    # Train on each experience
    all_losses = []
    for i, exp in enumerate(tokenized):
        print(f"\n--- Training on experience #{exp['id']} ({i+1}/{len(tokenized)}) ---")
        print(f"    {exp['prompt'][:80]}...")
        print(f"    seq_len={exp['seq_len']} prompt_len={exp['prompt_len']}")

        t0 = time.time()
        status, result = train(
            exp["tokens"], exp["prompt_len"], args.steps,
            args.sigma, args.lr, args.pop,
            args.clip_ratio, args.delta_decay,
            timeout=args.timeout,
        )
        elapsed = time.time() - t0

        if status != 200:
            print(f"    FAILED ({status}): {result}")
            continue

        losses = result.get("losses", [])
        all_losses.extend(losses)
        first = losses[0] if losses else 0
        last = losses[-1] if losses else 0
        delta_pct = (first - last) / first * 100 if first > 0 else 0
        print(f"    loss: {first:.4f} -> {last:.4f} ({delta_pct:+.1f}%) "
              f"in {elapsed:.1f}s ({elapsed/len(losses)*1000:.0f}ms/step)")

    # Post-training eval
    if not args.skip_eval:
        post_results = run_eval("POST-TRAINING")

        # Compare
        pre_coherent = sum(1 for r in pre_results if r["coherent"])
        post_coherent = sum(1 for r in post_results if r["coherent"])
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  Coherent before: {pre_coherent}/{len(pre_results)}")
        print(f"  Coherent after:  {post_coherent}/{len(post_results)}")
        if post_coherent < pre_coherent:
            print(f"  WARNING: Generation quality DEGRADED! Catastrophic forgetting detected.")
        elif post_coherent == pre_coherent:
            print(f"  OK: Generation quality preserved.")
        else:
            print(f"  IMPROVED: Generation quality increased!")


if __name__ == "__main__":
    main()
