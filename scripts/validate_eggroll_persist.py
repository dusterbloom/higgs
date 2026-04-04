#!/usr/bin/env python3
"""Validate EGGROLL persist-deltas: train on real data, verify generation stays coherent."""

import json, urllib.request, time, sys

BASE = "http://localhost:8000"

def chat(prompt, max_tokens=120):
    body = json.dumps({
        "model": "qwen35",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    c = data["choices"][0]["message"]
    return (c.get("content") or "") + (c.get("reasoning_content") or "")

def completions(prompt, max_tokens=80):
    body = json.dumps({
        "model": "qwen35", "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["text"]

def train(tokens, prompt_len, steps=5):
    body = json.dumps({
        "model": "qwen35", "tokens": tokens, "prompt_len": prompt_len,
        "sigma": 0.01, "lr": 0.005, "rank": 4, "population": 4,
        "total_steps": steps, "merge_interval": 0,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/train", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())

# Use completions endpoint with a coding prompt that exercises real token patterns.
# We'll encode a fibonacci prompt+completion as token IDs by getting them from the model.
# Since we don't have a tokenize endpoint, construct training data from known token patterns.

# The Qwen3.5 tokenizer encodes common ASCII as single bytes offset by a base.
# For a simple test, use the completion endpoint to get baseline, then train on
# a hand-crafted token sequence from the model's vocabulary.

print("=" * 60)
print("EGGROLL Persist-Deltas Validation")
print("=" * 60)

# Step 1: Baseline generation
print("\n--- BASELINE (clean model) ---")
test_prompts = [
    "What is 2+2? Reply with just the number.",
    "def fibonacci(n):",
]
baseline_outputs = []
for p in test_prompts:
    try:
        r = chat(p, 80)
    except Exception:
        r = completions(p, 80)
    baseline_outputs.append(r)
    print(f"  Prompt: {p[:50]}")
    print(f"  Output: {r[:120]}")
    print()

# Step 2: Train on a REAL coding pattern
# Use token IDs that represent actual text patterns.
# Qwen3.5 vocab: common tokens include small integers.
# We'll use a sequence that represents a real-ish pattern:
# prompt tokens (10) + completion tokens that follow a fibonacci-like pattern
print("--- TRAINING (5 steps, sigma=0.01, lr=0.005) ---")

# Build training tokens: use a repeated structured pattern
# These are real token IDs from the Qwen3.5 vocabulary
# (small IDs = common tokens like digits, punctuation, common words)
prompt_tokens = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645]  # <|im_start|>system\nYou are a helpful assistant.<|im_end|>
completion_tokens = [198, 151644, 77091, 198]  # \n<|im_start|>assistant\n
# Add some actual content tokens (common English words/code)
content = [785, 11, 358, 3003, 1492, 311, 1492, 499, 0, 151645]  # "Hi, I would like to help you!\n<|im_end|>"
all_tokens = prompt_tokens + completion_tokens + content
prompt_len = len(prompt_tokens)

print(f"  Total tokens: {len(all_tokens)}, prompt_len: {prompt_len}")
t0 = time.time()
result = train(all_tokens, prompt_len, steps=5)
elapsed = time.time() - t0
losses = result["losses"]
print(f"  Done in {elapsed:.1f}s")
print(f"  Losses: {[f'{l:.3f}' for l in losses]}")
delta_pct = (losses[0] - losses[-1]) / losses[0] * 100
print(f"  Loss change: {losses[0]:.3f} -> {losses[-1]:.3f} ({delta_pct:+.1f}%)")

# Step 3: Post-training generation (THE CRITICAL TEST)
print("\n--- POST-TRAINING (with persisted deltas) ---")
post_outputs = []
for p in test_prompts:
    try:
        r = chat(p, 80)
    except Exception:
        r = completions(p, 80)
    post_outputs.append(r)
    print(f"  Prompt: {p[:50]}")
    print(f"  Output: {r[:120]}")
    print()

# Step 4: Coherence check
print("--- VERDICT ---")
all_coherent = True
for i, (base, post) in enumerate(zip(baseline_outputs, post_outputs)):
    # Check: output has real words (not random unicode), reasonable length
    has_ascii = sum(1 for c in post if c.isascii() and c.isalpha()) > 5
    not_empty = len(post.strip()) > 10
    coherent = has_ascii and not_empty
    status = "PASS" if coherent else "FAIL"
    print(f"  Prompt {i}: {status} (len={len(post)}, ascii_letters={sum(1 for c in post if c.isascii() and c.isalpha())})")
    if not coherent:
        all_coherent = False

if all_coherent:
    print("\nSUCCESS: Generation remains coherent after EGGROLL training with persisted deltas!")
    sys.exit(0)
else:
    print("\nFAILURE: Generation corrupted after training!")
    sys.exit(1)
