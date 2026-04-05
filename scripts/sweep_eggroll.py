#!/usr/bin/env python3
"""EGGROLL hyperparameter sweep on real models.

Sweeps sigma, lr, rank, population across multiple tasks and models.
Optionally evaluates generation quality after each training run.
Outputs a TSV results file for analysis.

Usage:
    # Start Higgs: cargo run --release -p higgs -- serve --timeout 1800
    python3 scripts/sweep_eggroll.py [--steps 20] [--timeout 1800] [--out results.tsv]

    # With generation quality checks:
    python3 scripts/sweep_eggroll.py --quick --gen-check --steps 20

    # Fine LR sweep around known-good sigma=0.01:
    python3 scripts/sweep_eggroll.py --lr-sweep --gen-check --steps 50
"""

import argparse
import itertools
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Models — add local model paths here
# ---------------------------------------------------------------------------
MODELS = {
    "qwen35": "/Users/peppi/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit",
}

# ---------------------------------------------------------------------------
# Tasks — (system, user, assistant_completion) tuples
# ---------------------------------------------------------------------------
TASKS = {
    "fibonacci": (
        "You are a helpful coding assistant.",
        "Write a Python function to compute the nth Fibonacci number efficiently.",
        (
            "Here's an efficient O(n) implementation using iteration:\n\n"
            "```python\n"
            "def fibonacci(n: int) -> int:\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
            "```\n\n"
            "This avoids exponential time complexity of naive recursion."
        ),
    ),
    "fizzbuzz": (
        "You are a helpful coding assistant.",
        "Write a Python function that prints FizzBuzz for numbers 1 to n.",
        (
            "```python\n"
            "def fizzbuzz(n: int) -> None:\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 15 == 0:\n"
            "            print('FizzBuzz')\n"
            "        elif i % 3 == 0:\n"
            "            print('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            print('Buzz')\n"
            "        else:\n"
            "            print(i)\n"
            "```"
        ),
    ),
    "math_gcd": (
        "You are a helpful math assistant.",
        "Explain and implement the Euclidean algorithm for GCD.",
        (
            "The Euclidean algorithm finds the greatest common divisor by "
            "repeatedly replacing the larger number with the remainder of "
            "dividing the larger by the smaller.\n\n"
            "```python\n"
            "def gcd(a: int, b: int) -> int:\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return a\n"
            "```\n\n"
            "Time complexity: O(log(min(a, b)))."
        ),
    ),
    "reasoning": (
        "You are a helpful assistant.",
        "A farmer has 17 sheep. All but 9 die. How many are left?",
        "9 sheep are left. The phrase 'all but 9' means 9 survive.",
    ),
}

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
SIGMA_VALUES = [0.005, 0.01, 0.02, 0.05]
LR_VALUES = [0.0005, 0.001, 0.005, 0.01]
RANK_VALUES = [2, 4, 8]
POP_VALUES = [2, 4]

# Fine LR sweep: sigma=0.01 confirmed best, zoom into LR neighborhood
LR_SWEEP_SIGMAS = [0.01]
LR_SWEEP_LRS = [0.002, 0.003, 0.005, 0.007, 0.01]
LR_SWEEP_RANKS = [4]
LR_SWEEP_POPS = [4]

# ---------------------------------------------------------------------------
# Generation quality evaluation
# ---------------------------------------------------------------------------

# Keywords per task that a good answer should contain
TASK_KEYWORDS = {
    "fibonacci": ["def fibonacci", "a, b = 0, 1", "a, b = b, a + b", "return b"],
    "fizzbuzz": ["def fizzbuzz", "FizzBuzz", "% 15", "% 3", "% 5"],
    "math_gcd": ["def gcd", "while b", "a % b", "return a"],
    "reasoning": ["9"],
}


def generate_completion(model_name: str, system: str, user: str, max_tokens: int = 256, timeout: int = 120) -> str:
    """Generate a completion via /v1/chat/completions (greedy, no streaming)."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    status, result = http_post(f"{BASE}/v1/chat/completions", payload, timeout=timeout)
    if status != 200:
        return f"[GEN_ERROR: {status}]"
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "[GEN_PARSE_ERROR]"


def score_generation(text: str, task_name: str, expected: str) -> dict:
    """Score generated text against expected completion.

    Returns dict with:
      - keyword_hits: fraction of task keywords found in output
      - token_overlap: fraction of expected words found in output (BLEU-1 proxy)
      - length_ratio: len(output) / len(expected), clamped to [0, 2]
    """
    text_lower = text.lower()
    keywords = TASK_KEYWORDS.get(task_name, [])

    # Keyword hit rate
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    keyword_score = hits / len(keywords) if keywords else 0.0

    # Token overlap (unigram precision — cheap BLEU-1)
    expected_words = set(re.findall(r'\w+', expected.lower()))
    output_words = set(re.findall(r'\w+', text_lower))
    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
    else:
        overlap = 0.0

    # Length ratio
    length_ratio = min(len(text) / max(len(expected), 1), 2.0)

    return {
        "keyword_score": keyword_score,
        "token_overlap": overlap,
        "length_ratio": length_ratio,
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_tokenizers = {}

def get_tokenizer(model_path: str):
    if model_path not in _tokenizers:
        from transformers import AutoTokenizer
        _tokenizers[model_path] = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
    return _tokenizers[model_path]


def tokenize_task(model_path: str, system: str, user: str, completion: str):
    tok = get_tokenizer(model_path)
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    full = prompt + completion + "<|im_end|>"
    tokens = tok.encode(full, add_special_tokens=False)
    prompt_tokens = tok.encode(prompt, add_special_tokens=False)
    return tokens, len(prompt_tokens)


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


def run_one(model_name, tokens, prompt_len, sigma, lr, rank, pop, steps, merge_interval, timeout):
    payload = {
        "model": model_name,
        "tokens": tokens,
        "prompt_len": prompt_len,
        "sigma": sigma,
        "lr": lr,
        "rank": rank,
        "population": pop,
        "total_steps": steps,
        "merge_interval": merge_interval,
    }
    t0 = time.time()
    status, result = http_post(f"{BASE}/v1/train", payload, timeout=timeout)
    elapsed = time.time() - t0

    if status != 200:
        return {
            "status": "error",
            "http_code": status,
            "error": str(result.get("error", ""))[:200],
            "elapsed": elapsed,
        }

    losses = result.get("losses", [])
    if not losses:
        return {"status": "no_losses", "elapsed": elapsed}

    first = losses[0]
    last = losses[-1]
    best = min(losses)
    # Convergence: average of last 25% vs first loss
    window = max(len(losses) // 4, 1)
    tail_avg = sum(losses[-window:]) / window

    return {
        "status": "ok",
        "first_loss": first,
        "last_loss": last,
        "best_loss": best,
        "tail_avg": tail_avg,
        "delta_pct": (first - tail_avg) / first * 100 if first > 0 else 0,
        "n_steps": len(losses),
        "elapsed": elapsed,
        "losses": losses,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EGGROLL hyperparameter sweep")
    parser.add_argument("--steps", type=int, default=20, help="Training steps per run")
    parser.add_argument("--merge-interval", type=int, default=0, help="Merge interval (0=no merge)")
    parser.add_argument("--timeout", type=int, default=1800, help="HTTP timeout per run (seconds)")
    parser.add_argument("--out", type=str, default=None, help="Output TSV file")
    parser.add_argument("--tasks", type=str, nargs="*", default=None, help="Tasks to run (default: all)")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Models to run (default: all)")
    parser.add_argument("--quick", action="store_true", help="Quick sweep: fewer combos")
    parser.add_argument("--lr-sweep", action="store_true", help="Fine LR sweep around sigma=0.01")
    parser.add_argument("--gen-check", action="store_true", help="Generate text after each training run and evaluate quality")
    parser.add_argument("--gen-tokens", type=int, default=256, help="Max tokens for generation quality check")
    args = parser.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"benchmarks/sweep_eggroll_{ts}.tsv"

    # Check server
    try:
        req = urllib.request.Request(f"{BASE}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            models_data = json.loads(resp.read())
        available = [m["id"] for m in models_data.get("data", [])]
        print(f"Server up. Available models: {available}")
    except Exception as e:
        print(f"Server not reachable: {e}")
        print("Start Higgs: cargo run --release -p higgs -- serve")
        sys.exit(1)

    task_names = args.tasks or list(TASKS.keys())
    model_names = args.models or list(MODELS.keys())

    if args.lr_sweep:
        sigmas = LR_SWEEP_SIGMAS
        lrs = LR_SWEEP_LRS
        ranks = LR_SWEEP_RANKS
        pops = LR_SWEEP_POPS
    elif args.quick:
        sigmas = [0.01, 0.05]
        lrs = [0.001, 0.005]
        ranks = [4]
        pops = [4]
    else:
        sigmas = SIGMA_VALUES
        lrs = LR_VALUES
        ranks = RANK_VALUES
        pops = POP_VALUES

    combos = list(itertools.product(model_names, task_names, sigmas, lrs, ranks, pops))
    print(f"\nSweep: {len(combos)} configurations")
    print(f"  Models: {model_names}")
    print(f"  Tasks: {task_names}")
    print(f"  Sigma: {sigmas}")
    print(f"  LR: {lrs}")
    print(f"  Rank: {ranks}")
    print(f"  Pop: {pops}")
    print(f"  Steps: {args.steps}, Merge: {args.merge_interval}")
    print(f"  Output: {args.out}")
    print()

    # Ensure output dir exists
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Pre-tokenize all tasks for all models
    task_tokens = {}
    for model_name in model_names:
        model_path = MODELS[model_name]
        for task_name in task_names:
            system, user, completion = TASKS[task_name]
            tokens, prompt_len = tokenize_task(model_path, system, user, completion)
            task_tokens[(model_name, task_name)] = (tokens, prompt_len)
            print(f"  {model_name}/{task_name}: {len(tokens)} tokens, prompt={prompt_len}")

    # Baseline generation (before any training) for quality comparison
    baselines = {}
    if args.gen_check:
        print("\n--- Baseline generation (no training) ---")
        for task_name in task_names:
            system, user, completion = TASKS[task_name]
            # Use the first available model name (server resolves it)
            gen = generate_completion(model_names[0], system, user, args.gen_tokens)
            score = score_generation(gen, task_name, completion)
            baselines[task_name] = {"text": gen, "score": score}
            print(f"  {task_name}: kw={score['keyword_score']:.0%} overlap={score['token_overlap']:.0%} len_ratio={score['length_ratio']:.2f}")
            print(f"    output: {gen[:120]}...")

    # Header
    header = [
        "model", "task", "sigma", "lr", "rank", "pop", "steps",
        "status", "first_loss", "last_loss", "best_loss", "tail_avg",
        "delta_pct", "elapsed_s", "ms_per_step",
    ]
    if args.gen_check:
        header.extend(["gen_kw_score", "gen_overlap", "gen_len_ratio",
                        "base_kw_score", "base_overlap", "kw_delta", "overlap_delta"])

    results = []
    best_delta = -999
    best_config = None

    with open(args.out, "w") as f:
        f.write("\t".join(header) + "\n")

        for i, (model_name, task_name, sigma, lr, rank, pop) in enumerate(combos):
            tokens, prompt_len = task_tokens[(model_name, task_name)]
            label = f"[{i+1}/{len(combos)}] {model_name}/{task_name} s={sigma} lr={lr} r={rank} p={pop}"
            print(f"\n{label}")

            r = run_one(
                model_name, tokens, prompt_len,
                sigma, lr, rank, pop,
                args.steps, args.merge_interval, args.timeout,
            )

            if r["status"] == "ok":
                ms_step = (r["elapsed"] * 1000) / max(r["n_steps"], 1)
                row = [
                    model_name, task_name, sigma, lr, rank, pop, r["n_steps"],
                    "ok", f"{r['first_loss']:.4f}", f"{r['last_loss']:.4f}",
                    f"{r['best_loss']:.4f}", f"{r['tail_avg']:.4f}",
                    f"{r['delta_pct']:.2f}", f"{r['elapsed']:.1f}", f"{ms_step:.0f}",
                ]
                print(f"  Loss: {r['first_loss']:.4f} -> {r['tail_avg']:.4f} ({r['delta_pct']:+.2f}%) in {r['elapsed']:.0f}s")

                # Generation quality check (deltas are now active on the model)
                if args.gen_check:
                    system, user, completion = TASKS[task_name]
                    gen = generate_completion(model_name, system, user, args.gen_tokens)
                    score = score_generation(gen, task_name, completion)
                    base = baselines.get(task_name, {}).get("score", {})
                    base_kw = base.get("keyword_score", 0)
                    base_ov = base.get("token_overlap", 0)
                    kw_delta = score["keyword_score"] - base_kw
                    ov_delta = score["token_overlap"] - base_ov
                    row.extend([
                        f"{score['keyword_score']:.2f}",
                        f"{score['token_overlap']:.2f}",
                        f"{score['length_ratio']:.2f}",
                        f"{base_kw:.2f}",
                        f"{base_ov:.2f}",
                        f"{kw_delta:+.2f}",
                        f"{ov_delta:+.2f}",
                    ])
                    kw_arrow = "+" if kw_delta > 0 else ("-" if kw_delta < 0 else "=")
                    print(f"  Gen: kw={score['keyword_score']:.0%}({kw_arrow}) overlap={score['token_overlap']:.0%}")
                    print(f"    output: {gen[:120]}...")

                if r["delta_pct"] > best_delta:
                    best_delta = r["delta_pct"]
                    best_config = (model_name, task_name, sigma, lr, rank, pop)
                    print(f"  *** NEW BEST: {best_delta:+.2f}% ***")
            else:
                row = [
                    model_name, task_name, sigma, lr, rank, pop, 0,
                    r["status"], "", "", "", "",
                    "", f"{r['elapsed']:.1f}", "",
                ]
                if args.gen_check:
                    row.extend(["", "", "", "", "", "", ""])
                print(f"  {r['status']}: {r.get('error', '')[:100]}")

            f.write("\t".join(str(x) for x in row) + "\n")
            f.flush()
            results.append((label, r))

    # Summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)

    ok_results = [(l, r) for l, r in results if r["status"] == "ok"]
    err_results = [(l, r) for l, r in results if r["status"] != "ok"]

    print(f"  Successful: {len(ok_results)}/{len(results)}")
    print(f"  Failed: {len(err_results)}")

    if best_config:
        m, t, s, lr, r, p = best_config
        print(f"\n  BEST CONFIG: {best_delta:+.2f}% convergence")
        print(f"    Model: {m}, Task: {t}")
        print(f"    sigma={s}, lr={lr}, rank={r}, pop={p}")

    # Top 5 by loss reduction
    if ok_results:
        ranked = sorted(ok_results, key=lambda x: x[1]["delta_pct"], reverse=True)
        print(f"\n  Top 5 by loss reduction:")
        for label, r in ranked[:5]:
            print(f"    {r['delta_pct']:+6.2f}%  {label}")

    print(f"\n  Results: {args.out}")


if __name__ == "__main__":
    main()
