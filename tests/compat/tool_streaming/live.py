"""Inert, serial cold/warm/multi-turn probe against a local Escha server."""
import argparse
import json
import time
from pathlib import Path

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("--origin", default="http://127.0.0.1:19092")
parser.add_argument("--output", default="/private/tmp/higgs-live-validation")
args = parser.parse_args()
Path(args.output).mkdir(parents=True, exist_ok=True)
client = httpx.Client(base_url=args.origin, headers={"Authorization": "Bearer streaming-local-validation"}, timeout=180)
model = "escha-35b-a3b"
session_id = time.time_ns() % (2**63 - 1)
last_disconnect = None
tool = {"type": "function", "function": {"name": "write", "description": "Inert validation sink. No file is written or code executed.",
    "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}}}
paragraph = "This is reference material for an isolated streaming test. A client receives structured events over an HTTP connection. The server reports actual generated arguments incrementally. Completed JSON can be inspected after the response succeeds. No external action is performed.\n"
system = {"role": "system", "content": "Use the provided write function to answer the final instruction. Output one call with a content string. Do not explain outside the call."}
instruction = "\nNow call write once. Its content must be a coherent explanation of streaming in approximately 120 words. This is an inert test; only produce the call."


def make_messages(repetitions):
    return [system, {"role": "user", "content": "Reference material:\n" + paragraph * repetitions + instruction}]


def count(repetitions):
    response = client.post("/v1/messages/count_tokens", json={"model": model, "system": system["content"],
        "messages": make_messages(repetitions)[1:]})
    response.raise_for_status()
    return response.json()["input_tokens"]


def generate(name, messages, *, cancel_after=None, request_session=None, cancel_delay=0, max_after_disconnect=None):
    global last_disconnect
    request = {"model": model, "messages": messages, "tools": [tool], "tool_choice": "auto",
        "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
        "enable_thinking": False, "speculation": "none", "max_tokens": 768, "session_id": session_id if request_session is None else request_session}
    started = time.monotonic()
    semantic_times, argument_times, comment_times = [], [], []
    calls = {}
    content = ""
    usage = None
    finish = None
    cancellation_triggered = False
    with client.stream("POST", "/v1/chat/completions", json=request) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            elapsed = time.monotonic() - started
            if max_after_disconnect is not None:
                assert last_disconnect is not None
                assert time.monotonic() - last_disconnect < max_after_disconnect, \
                    "worker remained blocked after cancellation"
            if line.startswith(":"):
                comment_times.append(elapsed)
                continue
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content += delta["content"]
                    semantic_times.append(elapsed)
                for call in delta.get("tool_calls") or []:
                    target = calls.setdefault(call["index"], {"id": None, "name": "", "arguments": ""})
                    function = call.get("function") or {}
                    if call.get("id"):
                        assert target["id"] is None, "identity was repeated"
                        target["id"] = call["id"]
                        target["name"] = function.get("name", "")
                        semantic_times.append(elapsed)
                    if function.get("arguments"):
                        target["arguments"] += function["arguments"]
                        argument_times.append(elapsed)
                        semantic_times.append(elapsed)
            if cancel_after is not None and len(argument_times) >= cancel_after:
                assert finish is None, "call completed before cancellation"
                if cancel_delay:
                    time.sleep(cancel_delay)
                cancellation_triggered = True
                break
    closed_at = time.monotonic()
    duration = closed_at - started
    if cancel_after is not None:
        assert cancellation_triggered, "stream ended before cancellation could be tested"
    gaps = [b - a for a, b in zip([0] + semantic_times, semantic_times + [duration])]
    result = {"label": args.label, "case": name, "duration_s": round(duration, 3),
        "first_semantic_s": semantic_times[0] if semantic_times else None,
        "first_argument_s": argument_times[0] if argument_times else None,
        "argument_events": len(argument_times), "transport_comments": len(comment_times),
        "max_silence_s": max(gaps),
        "max_inter_semantic_gap_s": max((b-a for a,b in zip(semantic_times, semantic_times[1:])), default=0),
        "max_argument_gap_s": max((b-a for a,b in zip(argument_times, argument_times[1:])), default=0),
        "input_tokens": (usage or {}).get("prompt_tokens"),
        "output_tokens": (usage or {}).get("completion_tokens"),
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "usage": usage, "finish": finish, "cancelled": cancellation_triggered,
        "first_semantic_after_disconnect_s": started + semantic_times[0] - last_disconnect
            if last_disconnect is not None and semantic_times else None}
    if cancellation_triggered:
        last_disconnect = closed_at
    print(json.dumps(result), flush=True)
    with (Path(args.output) / f"{args.label}-client.jsonl").open("a") as output:
        output.write(json.dumps(result) + "\n")
    if cancel_after is None:
        assert finish == "tool_calls" and len(calls) == 1, (finish, len(calls))
        assert isinstance(json.loads(calls[0]["arguments"])["content"], str)
    return content, list(calls.values())


def continuation(messages, content, calls, instruction):
    call = calls[0]
    return messages + [
        {"role": "assistant", "content": content or None, "tool_calls": [{"id": call["id"], "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]}}]},
        {"role": "tool", "tool_call_id": call["id"], "content": "Recorded by the validation client in memory only. No filesystem action occurred."},
        {"role": "user", "content": instruction},
    ]


repetitions = max(1, round(20_000 / (count(20) / 20)))
messages = make_messages(repetitions)
print(json.dumps({"label": args.label, "estimated_input_tokens": count(repetitions)}), flush=True)
content, calls = generate("cold", messages)
messages = continuation(messages, content, calls, "Call write again with a new explanation in approximately 120 words, focusing on client cancellation.")
content, calls = generate("warm-continuation", messages)
messages = continuation(messages, content, calls, "Call write again with approximately 120 words about preserving exact tool arguments.")
content, calls = generate("multi-turn", messages)
if args.label == "candidate":
    messages = continuation(messages, content, calls, "Call write with approximately 120 words about cancellation boundaries.")
    generate("cancel-during-arguments", messages, cancel_after=5)
    generate("after-cancellation", [system, {"role": "user", "content": "Call write with content exactly OK."}], request_session=session_id + 1, max_after_disconnect=30)
    cold_retry = [dict(system, content=system["content"] + " New initial-wait cancellation probe."), messages[1]]
    generate("cancel-during-initial-wait", cold_retry, cancel_after=0, cancel_delay=0.25, request_session=session_id + 2)
    generate("after-initial-wait-cancellation", [system, {"role": "user", "content": "Call write with content exactly OK."}], request_session=session_id + 3, max_after_disconnect=30)
client.close()
