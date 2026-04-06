#!/usr/bin/env python3
"""
Demonstrate what Higgs adaptive memory training is good at — and what it isn't.

After the empty-completion fix, hard prompts that produce no response are
correctly skipped (no wasted buffer slots or training cycles). This script
shows both sides:

  1. Hard prompts → empty response → skipped (no buffer waste)
  2. Trainable prompts → real completions → positive feedback → learning
  3. Corrections → strongest signal (reward=1.5 + token replacement)

Usage:
  python scripts/training_demo.py [--base-url http://localhost:8000]
"""

import argparse
import json
import sys
import time
import requests

CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def section(title):
    print(f"\n{CYAN}{BOLD}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{RESET}\n")


def chat(base_url, model, prompt, max_tokens=120):
    """Send a chat completion, return (text, request_id)."""
    r = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }, timeout=120)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    rid = data.get("id", "?")
    return text, rid


def feedback(base_url, rid, signal, correction_text=None):
    """Send feedback. signal: positive, negative, or correction."""
    payload = {"request_id": rid, "signal": signal}
    if correction_text is not None:
        payload["correction"] = correction_text
    r = requests.post(f"{base_url}/v1/feedback", json=payload, timeout=10)
    return r.json() if r.status_code == 200 else r.text


def wait_bar(seconds, label=""):
    bar_width = 40
    for i in range(seconds):
        filled = int((i + 1) / seconds * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  {DIM}[{bar}] {i+1}/{seconds}s{RESET}", end="", flush=True)
        time.sleep(1)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--idle-wait", type=int, default=90,
                        help="Seconds to wait for idle trainer (default: 90)")
    args = parser.parse_args()
    base = args.base_url

    # Health + model
    r = requests.get(f"{base}/health", timeout=5)
    r.raise_for_status()
    models = requests.get(f"{base}/v1/models", timeout=5).json()["data"]
    model = next((m["id"] for m in models if m.get("owned_by") != "remote"), models[0]["id"])
    print(f"{GREEN}{BOLD}[MODEL]{RESET} {model}")

    # Verify memory is enabled
    fb = feedback(base, "probe", "positive")
    if isinstance(fb, str) and "not enabled" in fb.lower():
        print(f"{RED}Memory not enabled. Add [memory] enabled=true to config.{RESET}")
        sys.exit(1)

    # ── HARD PROMPTS: should be SKIPPED now ──────────────────────
    section("HARD PROMPTS (should be skipped — no buffer waste)")

    hard_prompts = [
        "What is 347 * 29 + 1583 - 467 * 3? Show only the final number.",
        "What was the mass in kilograms of the heaviest pumpkin ever grown as of 2024?",
    ]
    hard_results = []
    for i, p in enumerate(hard_prompts):
        text, rid = chat(base, model, p, max_tokens=80)
        empty = not text.strip()
        hard_results.append({"prompt": p, "text": text, "rid": rid, "empty": empty})
        status = f"{GREEN}SKIPPED (empty → no buffer waste){RESET}" if empty else f"{YELLOW}has content (will enter buffer){RESET}"
        print(f"  {BOLD}[HARD-{i+1}]{RESET} {status}")
        print(f"    {DIM}Q:{RESET} {p[:70]}")
        print(f"    {DIM}A:{RESET} {text[:80] if text.strip() else '(empty)'}")
        print()

    # ── TRAINABLE PROMPTS: what adaptive memory excels at ────────
    section("TRAINABLE PROMPTS (real completions → real gradients)")

    print(f"""  {BOLD}What rank-4 PCAST deltas are good at:{RESET}
  {DIM}• Style/format reinforcement (attention pattern shifts)
  • Instruction compliance (follow specific output formats)
  • Behavioral adaptation (response length, tone, structure)
  • Copy-transform tasks (reordering, reformatting input)
  • Correction learning (strongest signal: reward=1.5 + new tokens){RESET}
""")

    trainable_prompts = [
        # Style: specific output format
        "List exactly 3 benefits of Rust over C++. Use this format: '1) ... 2) ... 3) ...'",
        # Behavioral: concise factual answer
        "In one sentence, what is the capital of France and its population?",
        # Instruction compliance: structured response
        "Classify this sentiment as POSITIVE, NEGATIVE, or NEUTRAL and explain in exactly one sentence: 'The food was decent but the service was slow'",
        # Copy-transform: rearrange given data
        "Sort these words alphabetically and return only the sorted list: dog, cat, ant, bee, elk",
    ]

    baseline = []
    for i, p in enumerate(trainable_prompts):
        text, rid = chat(base, model, p, max_tokens=120)
        baseline.append({"prompt": p, "text": text, "rid": rid})
        print(f"  {BOLD}[BASE-{i+1}]{RESET} (id={rid})")
        print(f"    {DIM}Q:{RESET} {p[:80]}")
        print(f"    {DIM}A:{RESET} {text[:100]}")

        # Positive feedback — these are the kind of responses we want to reinforce
        fb = feedback(base, rid, "positive")
        print(f"    {DIM}feedback: positive → reward={fb.get('reward', '?') if isinstance(fb, dict) else '?'}{RESET}")
        print()

    # ── CORRECTION: strongest training signal ────────────────────
    section("CORRECTION FEEDBACK (reward=1.5 + token replacement)")

    correction_prompt = "What is the answer to life, the universe, and everything?"
    text, rid = chat(base, model, correction_prompt, max_tokens=80)
    print(f"  {BOLD}[CORRECT]{RESET} (id={rid})")
    print(f"    {DIM}Q:{RESET} {correction_prompt}")
    print(f"    {DIM}A:{RESET} {text[:100]}")

    # Send correction with the "right" answer
    correction_text = "42. As computed by Deep Thought over 7.5 million years."
    fb = feedback(base, rid, "correction", correction_text=correction_text)
    print(f"    {DIM}correction sent: '{correction_text}'{RESET}")
    print(f"    {DIM}feedback → {fb}{RESET}")
    print()

    # ── WAIT FOR TRAINING ────────────────────────────────────────
    section(f"TRAINING ({args.idle_wait}s idle wait ≈ {args.idle_wait // 5} training steps)")

    print(f"  {DIM}The idle trainer fires every 5s when no requests are in flight.{RESET}")
    print(f"  {DIM}Each step: pick highest-priority entry → PCAST fwd+bwd → Adam update{RESET}")
    print(f"  {DIM}Watch server logs for '[MEMORY] idle training step' messages.{RESET}")
    print()
    wait_bar(args.idle_wait)

    # ── POST-TRAINING: same prompts again ────────────────────────
    section("POST-TRAINING (same prompts)")

    post = []
    for i, p in enumerate(trainable_prompts):
        text, rid = chat(base, model, p, max_tokens=120)
        post.append({"prompt": p, "text": text, "rid": rid})
        print(f"  {BOLD}[POST-{i+1}]{RESET} (id={rid})")
        print(f"    {DIM}Q:{RESET} {p[:80]}")
        print(f"    {DIM}A:{RESET} {text[:100]}")
        print()

    # Also re-check the correction prompt
    corr_text2, corr_rid2 = chat(base, model, correction_prompt, max_tokens=80)
    print(f"  {BOLD}[POST-CORR]{RESET} (id={corr_rid2})")
    print(f"    {DIM}Q:{RESET} {correction_prompt}")
    print(f"    {DIM}A:{RESET} {corr_text2[:120]}")
    print()

    # ── RESULTS ──────────────────────────────────────────────────
    section("RESULTS")

    any_changed = False
    for i in range(len(trainable_prompts)):
        b = baseline[i]["text"]
        p = post[i]["text"]
        changed = b != p
        if changed:
            any_changed = True

        marker = f"{GREEN}{BOLD}CHANGED{RESET}" if changed else f"{DIM}same{RESET}"
        print(f"  {BOLD}Prompt {i+1}:{RESET} {marker}")
        if changed:
            print(f"    {RED}Before:{RESET} {b[:90]}")
            print(f"    {GREEN}After:{RESET}  {p[:90]}")
        else:
            print(f"    {DIM}{b[:90]}{RESET}")
        print()

    # Correction result
    corr_changed = text != corr_text2
    marker = f"{GREEN}{BOLD}CHANGED{RESET}" if corr_changed else f"{DIM}same{RESET}"
    print(f"  {BOLD}Correction:{RESET} {marker}")
    if corr_changed:
        print(f"    {RED}Before:{RESET} {text[:90]}")
        print(f"    {GREEN}After:{RESET}  {corr_text2[:90]}")
    else:
        print(f"    {DIM}{text[:90]}{RESET}")
    print()

    # Hard prompts summary
    print(f"  {BOLD}Hard prompts:{RESET}")
    skipped = sum(1 for r in hard_results if r["empty"])
    print(f"    {GREEN}{skipped}/{len(hard_results)} correctly skipped{RESET} (empty completion → no buffer waste)")
    entered = sum(1 for r in hard_results if not r["empty"])
    if entered:
        print(f"    {YELLOW}{entered}/{len(hard_results)} entered buffer{RESET} (model produced content — trainable!)")
    print()

    if any_changed or corr_changed:
        print(f"  {GREEN}{BOLD}Training produced visible response changes!{RESET}")
    else:
        print(f"  {YELLOW}No visible changes. Try --idle-wait 120 or check server logs for training activity.{RESET}")

    print(f"\n{CYAN}{BOLD}{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()
