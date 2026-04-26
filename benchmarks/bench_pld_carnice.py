#!/usr/bin/env python3
"""PLD A/B microbench for Carnice-9b.

Boots `higgs serve` once per config (--pld off, then --pld on), runs N trials
on a verbatim-repeat prompt (max PLD n-gram acceptance), reports decode tok/s.
"""

import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

HIGGS_BIN = "target/release/higgs"
MODEL = "/Users/peppi/.cache/lm-studio/models/jason-schulz/Carnice-9b-MLX"
MODEL_NAME = "Carnice-9b-MLX"
PORT = 8765
MAX_TOKENS = 384
RUNS = 3

# A passage the model should reproduce verbatim. Long enough that decode time
# dominates load/measurement noise; n-gram-rich so PLD has lots to match.
PASSAGE = (
    "B-tree storage engines split a page when an insert would overflow it; "
    "the median key is promoted into the parent and two siblings are written. "
    "Merge happens lazily when a delete drops occupancy under a threshold. "
    "Write-ahead logging records the intent of each modification before the "
    "page itself is mutated, so a crash mid-write is recoverable: replay the "
    "log forward and any committed transaction is durable, any uncommitted "
    "one is rolled back. TCP congestion control walks a state machine: slow "
    "start doubles the window every RTT until a loss, congestion avoidance "
    "linearly probes for more bandwidth, fast retransmit cuts the window in "
    "half when triplicate ACKs arrive, and fast recovery resumes additive "
    "increase without re-entering slow start. CUBIC replaces the linear "
    "probe with a cubic function around the previous max. BBR abandons "
    "loss-as-signal entirely and models the bottleneck pipe directly via "
    "delivery rate and minimum RTT. Out-of-order CPUs use Tomasulo-style "
    "reservation stations to dispatch ready instructions and a reorder "
    "buffer to retire them in program order. Branch prediction speculates "
    "across control flow, mis-predictions flush the pipeline. TLS 1.3 "
    "completes a key exchange in one round trip: ClientHello carries an "
    "ECDHE share, ServerHello completes the exchange, the certificate is "
    "signed under the negotiated key. 0-RTT resumes a session with a "
    "pre-shared key, sending early data before the handshake completes. "
    "Raft elects a leader by majority vote, replicates a log entry to a "
    "majority before considering it committed, and applies entries in log "
    "order to the state machine. Membership changes use a joint-consensus "
    "phase to avoid split-brain across the old and new configurations."
)

PROMPT = (
    "Repeat the following passage verbatim, exactly as written, surrounded by "
    "<<< and >>>. Do not paraphrase, summarize, or comment. Just reproduce it.\n\n"
    f"{PASSAGE}"
)


def start_server(extra_args, log_path):
    cmd = [HIGGS_BIN, "serve", "--model", MODEL, "--port", str(PORT), *extra_args]
    env = {**os.environ, "RUST_LOG": "info", "HIGGS_ENABLE_THINKING": "0"}
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid
        )
    base = f"http://127.0.0.1:{PORT}"
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=3) as r:
                if json.loads(r.read()).get("data"):
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            return None
        time.sleep(1)
    return None


def kill_server(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{PORT}"], text=True).strip()
        for pid in out.splitlines():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass
    except subprocess.CalledProcessError:
        pass
    time.sleep(2)


def stream_chat(base, prompt, max_tokens, timeout=600, capture_text=False):
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first_ts = last_ts = None
    prompt_tokens = completion_tokens = 0
    usage_seen = False
    text_buf = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        now = time.perf_counter()
                        if first_ts is None:
                            first_ts = now
                        last_ts = now
                        if capture_text:
                            text_buf.append(content)
                u = obj.get("usage")
                if u:
                    usage_seen = True
                    prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                    completion_tokens = u.get("completion_tokens", completion_tokens)
    if not usage_seen or completion_tokens < 2:
        return {"error": "no usage / too few tokens"}
    end = time.perf_counter()
    ttft = first_ts - t0 if first_ts else end - t0
    decode_s = (last_ts - first_ts) if (first_ts and last_ts and last_ts > first_ts) else 0
    wall_s = end - t0
    e2e_tps = completion_tokens / wall_s if wall_s > 0 else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft * 1000,
        "decode_s": decode_s,
        "wall_s": wall_s,
        "decode_tps": (completion_tokens - 1) / decode_s if decode_s > 0 else 0,
        "e2e_tps": e2e_tps,
        "text": "".join(text_buf) if capture_text else None,
    }


def run_config(label, extra_args, log_path):
    print(f"\n=== {label} ({' '.join(extra_args) or 'baseline'}) ===", flush=True)
    proc = start_server(extra_args, log_path)
    if proc is None:
        print(f"[{label}] server failed to start; tail of {log_path}:", file=sys.stderr)
        with open(log_path) as f:
            sys.stderr.write("".join(f.readlines()[-40:]))
        return None
    base = f"http://127.0.0.1:{PORT}"
    try:
        # Warmup: tiny request to JIT decode path.
        stream_chat(base, "Say hello.", 8, timeout=120)
        results = []
        for k in range(RUNS):
            r = stream_chat(base, PROMPT, MAX_TOKENS, capture_text=(k == 0))
            if "error" in r:
                print(f"[{label}] run {k+1} ERROR: {r['error']}", flush=True)
                continue
            results.append(r)
            print(
                f"[{label}] run {k+1}: prompt={r['prompt_tokens']} "
                f"completion={r['completion_tokens']} "
                f"ttft={r['ttft_ms']:.0f}ms "
                f"decode={r['decode_tps']:.2f} tok/s "
                f"e2e={r['e2e_tps']:.2f} tok/s "
                f"wall={r['wall_s']:.2f}s",
                flush=True,
            )
        return results
    finally:
        kill_server(proc)


def summarize(label, results):
    if not results:
        return None
    decode = [r["decode_tps"] for r in results]
    ttft = [r["ttft_ms"] for r in results]
    print(
        f"[{label}] median decode={statistics.median(decode):.2f} tok/s "
        f"(min={min(decode):.2f} max={max(decode):.2f})  "
        f"median ttft={statistics.median(ttft):.0f}ms",
        flush=True,
    )
    return statistics.median(decode)


def main():
    log_dir = "benchmarks/pld_carnice_20260426"
    os.makedirs(log_dir, exist_ok=True)

    base_log = os.path.join(log_dir, "baseline.server.log")
    pld_log = os.path.join(log_dir, "pld.server.log")

    base_runs = run_config("baseline", [], base_log)
    pld_runs = run_config("pld", ["--pld"], pld_log)

    print("\n=== summary ===")
    base_med = summarize("baseline", base_runs or [])
    pld_med = summarize("pld", pld_runs or [])
    if base_med and pld_med:
        speedup = pld_med / base_med
        print(f"pld decode tok/s vs baseline: {speedup:.2f}x")
        # Count PLD spec_decode cycles in the pld log to confirm path
        try:
            with open(pld_log) as f:
                cycles = sum(1 for line in f if "spec_decode: cycle" in line)
            accepted = 0
            k_hits = 0
            with open(pld_log) as f:
                for line in f:
                    if "spec_decode: cycle" in line:
                        # naive parse: look for accepted=N k=M
                        if "k=8" in line:
                            k_hits += 1
            print(f"pld cycles: {cycles}  k=8 hits: {k_hits}")
        except OSError:
            pass

    out = {
        "baseline": base_runs,
        "pld": pld_runs,
        "baseline_median": base_med,
        "pld_median": pld_med,
    }
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"results -> {log_dir}/results.json")


if __name__ == "__main__":
    main()
