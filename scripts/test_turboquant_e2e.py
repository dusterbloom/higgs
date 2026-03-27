#!/usr/bin/env python3
"""E2E TurboQuant test: start server, generate, compare outputs."""
import json
import subprocess
import sys
import time
import urllib.request
import signal
import os

MODEL = os.environ.get(
    "MODEL_PATH",
    os.path.expanduser("~/.cache/lm-studio/models/mlx-community/Qwen3-14B-4bit"),
)
HIGGS = "./target/release/higgs"
PORT = 8877
URL = f"http://localhost:{PORT}/v1/chat/completions"

PROMPT = "What is the capital of France? Answer in one sentence."
MAX_TOKENS = 100


def start_server(extra_args=None):
    cmd = [
        HIGGS, "serve",
        "--port", str(PORT),
        "--model", MODEL,
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    env["HIGGS_NO_CONFIG"] = "1"
    env["HIGGS_ENABLE_THINKING"] = "0"
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Wait for server to be ready
    for attempt in range(60):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health")
            return proc
        except Exception:
            # Check if process died
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode()
                print(f"  Server died with code {proc.returncode}")
                print(f"  stderr: {stderr[-500:]}")
                return None
    print("  Server failed to start after 60s")
    proc.kill()
    return None


def generate(runs=2):
    results = []
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    for i in range(runs):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        elapsed = time.perf_counter() - start

        if "error" in result:
            print(f"    Run {i+1}: ERROR: {result['error']}")
            continue

        usage = result.get("usage", {})
        comp = usage.get("completion_tokens", 0)
        tps = comp / elapsed if elapsed > 0 else 0
        text = result["choices"][0]["message"]["content"]
        results.append({"text": text, "tokens": comp, "elapsed": elapsed, "tps": tps})
        print(f"    Run {i+1}: {comp} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
    return results


def stop_server(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    # --- Normal decode (baseline) ---
    print(f"\n=== BASELINE (no TurboQuant) ===")
    print(f"  Model: {MODEL}")
    proc = start_server()
    if not proc:
        print("  FAILED to start baseline server")
        sys.exit(1)

    baseline = generate()
    stop_server(proc)
    time.sleep(2)

    if not baseline:
        print("  No baseline results")
        sys.exit(1)

    # --- TurboQuant 3-bit ---
    print(f"\n=== TURBOQUANT 3-bit ===")
    proc = start_server(["--kv-cache", "turboquant", "--kv-bits", "3"])
    if not proc:
        print("  FAILED to start TurboQuant server")
        sys.exit(1)

    turbo3 = generate()
    stop_server(proc)
    time.sleep(2)

    # --- TurboQuant 4-bit ---
    print(f"\n=== TURBOQUANT 4-bit ===")
    proc = start_server(["--kv-cache", "turboquant", "--kv-bits", "4"])
    if not proc:
        print("  FAILED to start TurboQuant 4-bit server")
        sys.exit(1)

    turbo4 = generate()
    stop_server(proc)

    # --- Compare ---
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    if baseline:
        b = baseline[0]
        print(f"\nBaseline output ({b['tps']:.1f} tok/s):")
        print(f"  {b['text'][:200]}")

    if turbo3:
        t = turbo3[0]
        print(f"\nTurboQuant 3-bit output ({t['tps']:.1f} tok/s):")
        print(f"  {t['text'][:200]}")

    if turbo4:
        t = turbo4[0]
        print(f"\nTurboQuant 4-bit output ({t['tps']:.1f} tok/s):")
        print(f"  {t['text'][:200]}")

    # Check text match
    if baseline and turbo4:
        b_text = baseline[0]["text"].strip()
        t4_text = turbo4[0]["text"].strip()
        match_4bit = b_text == t4_text
        print(f"\n4-bit exact match: {match_4bit}")
        if not match_4bit:
            # Show first divergence
            for i, (a, b) in enumerate(zip(b_text, t4_text)):
                if a != b:
                    print(f"  First divergence at char {i}: baseline='{b_text[max(0,i-10):i+10]}' vs turbo='{t4_text[max(0,i-10):i+10]}'")
                    break

    if baseline and turbo3:
        b_text = baseline[0]["text"].strip()
        t3_text = turbo3[0]["text"].strip()
        match_3bit = b_text == t3_text
        print(f"3-bit exact match: {match_3bit}")
        if not match_3bit:
            for i, (a, b) in enumerate(zip(b_text, t3_text)):
                if a != b:
                    print(f"  First divergence at char {i}: baseline='{b_text[max(0,i-10):i+10]}' vs turbo='{t3_text[max(0,i-10):i+10]}'")
                    break

    # Speed comparison
    if baseline and turbo3:
        speedup = turbo3[0]["tps"] / baseline[0]["tps"] if baseline[0]["tps"] > 0 else 0
        print(f"\n3-bit speed ratio: {speedup:.2f}x")
    if baseline and turbo4:
        speedup = turbo4[0]["tps"] / baseline[0]["tps"] if baseline[0]["tps"] > 0 else 0
        print(f"4-bit speed ratio: {speedup:.2f}x")


if __name__ == "__main__":
    main()
