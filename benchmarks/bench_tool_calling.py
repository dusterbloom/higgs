#!/usr/bin/env python3
"""Benchmark tool calling: streaming vs non-streaming, with and without tools.

Measures TTFT, decode tok/s, and correctness of tool call parsing
for streaming tool call support on Qwen3.5-35B-A3B.

Usage:
    python3 benchmarks/bench_tool_calling.py
    python3 benchmarks/bench_tool_calling.py --port 8899
    python3 benchmarks/bench_tool_calling.py --runs 5
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

PORT = 8899
MAX_TOKENS = 256
RUNS = 3

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    },
]

SCENARIOS = [
    {
        "name": "single_tool_call",
        "label": "Single tool call (weather)",
        "messages": [
            {"role": "user", "content": "What's the weather like in Tokyo right now?"},
        ],
        "expect_tool": "get_weather",
    },
    {
        "name": "multi_tool_call",
        "label": "Multiple tool calls",
        "messages": [
            {
                "role": "user",
                "content": "I need three things: 1) weather in London, 2) search for 'latest AI news', 3) calculate 42 * 17",
            },
        ],
        "expect_tool": None,  # multiple tools expected
    },
    {
        "name": "no_tool_needed",
        "label": "No tool needed (plain answer)",
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer in one word."},
        ],
        "expect_tool": None,
    },
    {
        "name": "tool_with_reasoning",
        "label": "Tool call with thinking",
        "messages": [
            {
                "role": "user",
                "content": "Think step by step about what information you need, then check the weather in both New York and San Francisco.",
            },
        ],
        "expect_tool": "get_weather",
    },
]


def make_request(port, messages, tools, stream, max_tokens):
    """Make a chat completion request and return timing + response data."""
    body = {
        "model": "Qwen3.5-35B-A3B-3bit",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0,
    }
    if tools:
        body["tools"] = tools

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    first_token_time = None
    total_text = ""
    tool_calls = []
    finish_reason = None
    has_reasoning = False
    completion_tokens = 0
    prompt_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if stream:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    # Usage chunk
                    if chunk.get("usage"):
                        completion_tokens = chunk["usage"].get("completion_tokens", 0)
                        prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr

                    # Content
                    content = delta.get("content")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        total_text += content

                    # Reasoning
                    if delta.get("reasoning_content"):
                        has_reasoning = True
                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                    # Tool calls
                    tc_deltas = delta.get("tool_calls")
                    if tc_deltas:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        for tc in tc_deltas:
                            tool_calls.append(tc)
            else:
                resp_data = json.loads(resp.read().decode())
                first_token_time = time.perf_counter()
                choice = resp_data["choices"][0]
                msg = choice["message"]
                total_text = msg.get("content") or ""
                finish_reason = choice.get("finish_reason", "stop")
                if msg.get("tool_calls"):
                    tool_calls = msg["tool_calls"]
                if msg.get("reasoning_content"):
                    has_reasoning = True
                usage = resp_data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)

    except Exception as e:
        return {"error": str(e)}

    t_end = time.perf_counter()
    ttft = (first_token_time - t0) if first_token_time else (t_end - t0)
    wall = t_end - t0

    # Estimate decode tok/s from completion_tokens and wall time minus TTFT
    decode_time = wall - ttft if wall > ttft else wall
    decode_tps = completion_tokens / decode_time if decode_time > 0 and completion_tokens > 0 else 0

    return {
        "ttft_ms": ttft * 1000,
        "wall_s": wall,
        "decode_tps": decode_tps,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish_reason,
        "tool_calls": tool_calls,
        "num_tool_calls": len(tool_calls),
        "text": total_text[:200],
        "has_reasoning": has_reasoning,
    }


def run_scenario(port, scenario, tools, stream, runs):
    """Run a scenario multiple times and return aggregated results."""
    results = []
    for i in range(runs):
        r = make_request(port, scenario["messages"], tools, stream, MAX_TOKENS)
        if "error" in r:
            print(f"  ERROR on run {i+1}: {r['error']}")
            continue
        results.append(r)
        # Small cooldown between runs
        time.sleep(0.5)

    if not results:
        return None

    avg = lambda key: sum(r[key] for r in results) / len(results)
    best = lambda key: min(r[key] for r in results)

    return {
        "ttft_ms_avg": avg("ttft_ms"),
        "ttft_ms_best": best("ttft_ms"),
        "decode_tps_avg": avg("decode_tps"),
        "wall_s_avg": avg("wall_s"),
        "completion_tokens_avg": avg("completion_tokens"),
        "prompt_tokens": results[0]["prompt_tokens"],
        "finish_reason": results[-1]["finish_reason"],
        "num_tool_calls": results[-1]["num_tool_calls"],
        "tool_calls": results[-1]["tool_calls"],
        "text_sample": results[-1]["text"][:120],
        "has_reasoning": any(r["has_reasoning"] for r in results),
        "runs": len(results),
    }


def format_tool_calls(tool_calls):
    """Format tool calls for display."""
    parts = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            args = fn.get("arguments", "{}")
            if isinstance(args, str) and len(args) > 60:
                args = args[:57] + "..."
            parts.append(f"{name}({args})")
    return "; ".join(parts) if parts else "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    # Verify server
    try:
        resp = urllib.request.urlopen(f"http://localhost:{args.port}/v1/models", timeout=5)
        models = json.loads(resp.read().decode())
        model_name = models["data"][0]["id"] if models.get("data") else "unknown"
        print(f"Server: localhost:{args.port}  Model: {model_name}")
    except Exception as e:
        print(f"Cannot reach server on port {args.port}: {e}")
        sys.exit(1)

    print(f"Runs per scenario: {args.runs}  Max tokens: {MAX_TOKENS}")
    print(f"Tools: {len(TOOLS)} ({', '.join(t['function']['name'] for t in TOOLS)})")
    print()

    # Warmup
    print("Warming up...")
    make_request(args.port, [{"role": "user", "content": "Hi"}], None, False, 10)
    time.sleep(1)

    all_results = []

    for scenario in SCENARIOS:
        print(f"{'='*70}")
        print(f"Scenario: {scenario['label']}")
        print(f"{'='*70}")

        configs = [
            ("Non-streaming + tools", False, TOOLS),
            ("Streaming + tools", True, TOOLS),
            ("Streaming NO tools", True, None),
        ]

        for label, stream, tools in configs:
            print(f"\n  {label} ({args.runs} runs)...")
            result = run_scenario(args.port, scenario, tools, stream, args.runs)
            if result is None:
                print("    FAILED - all runs errored")
                continue

            tc_str = format_tool_calls(result["tool_calls"])
            reasoning = " +reasoning" if result["has_reasoning"] else ""

            print(f"    TTFT:    {result['ttft_ms_avg']:7.0f}ms (best: {result['ttft_ms_best']:.0f}ms)")
            print(f"    Decode:  {result['decode_tps_avg']:7.1f} tok/s  ({result['completion_tokens_avg']:.0f} tokens)")
            print(f"    Wall:    {result['wall_s_avg']:7.2f}s")
            print(f"    Finish:  {result['finish_reason']}{reasoning}")
            print(f"    Tools:   {result['num_tool_calls']} calls — {tc_str}")
            if result["text_sample"].strip():
                sample = result["text_sample"].replace("\n", " ").strip()
                print(f"    Text:    {sample[:100]}...")

            all_results.append({
                "scenario": scenario["name"],
                "config": label,
                "stream": stream,
                "has_tools": tools is not None,
                **{k: v for k, v in result.items() if k != "tool_calls"},
            })

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<25} {'Config':<25} {'TTFT':>7} {'tok/s':>7} {'Tools':>5} {'Finish':>12}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['scenario']:<25} {r['config']:<25} "
            f"{r['ttft_ms_avg']:6.0f}ms {r['decode_tps_avg']:6.1f} "
            f"{r['num_tool_calls']:>5} {r['finish_reason'] or 'n/a':>12}"
        )

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"benchmarks/bench_tool_calling_{ts}.txt"
    with open(outfile, "w") as f:
        f.write(f"Tool Calling Benchmark — {datetime.now().isoformat()}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Runs: {args.runs}, Max tokens: {MAX_TOKENS}\n")
        f.write(f"Tools: {len(TOOLS)}\n\n")
        for r in all_results:
            f.write(f"{r['scenario']} | {r['config']} | "
                    f"TTFT={r['ttft_ms_avg']:.0f}ms | "
                    f"decode={r['decode_tps_avg']:.1f}t/s | "
                    f"tokens={r['completion_tokens_avg']:.0f} | "
                    f"tools={r['num_tool_calls']} | "
                    f"finish={r['finish_reason']} | "
                    f"reasoning={r['has_reasoning']}\n")
        f.write(f"\nFull JSON:\n{json.dumps(all_results, indent=2, default=str)}\n")
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
