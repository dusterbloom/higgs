"""Exercise official SDK argument events against the inert Rust HTTP fixture."""
import json
import os
import time

import anthropic
import openai

ORIGIN = os.environ.get("HIGGS_COMPAT_URL", "http://127.0.0.1:19091")
MODEL = "required-stream-script-long"
PARAMETERS = {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}
EXPECTED = {"content": "chunk " * (16 * 80)}


def verify(label, fragments, times, finish, expected_finish):
    assert finish == expected_finish, (label, finish)
    assert json.loads("".join(fragments)) == EXPECTED
    assert len(times) > 10, (label, "not incremental", len(times))
    assert times[-1] - times[0] > 1, (label, "fixture too short")
    max_gap = max(b - a for a, b in zip(times, times[1:]))
    assert max_gap < 1, (label, "semantic idle gap", max_gap)
    print(json.dumps({"client": label, "argument_events": len(times),
                      "max_gap_ms": round(max_gap * 1000), "result": "pass"}))


def check_openai():
    client = openai.OpenAI(base_url=f"{ORIGIN}/v1", api_key="fixture", max_retries=0)
    fragments, times, finish = [], [], None
    with client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "write fixture"}],
        tools=[{"type": "function", "function": {"name": "write", "parameters": PARAMETERS}}],
        stream=True, max_tokens=4096,
    ) as stream:
        for event in stream:
            for choice in event.choices:
                finish = choice.finish_reason or finish
                for call in choice.delta.tool_calls or []:
                    if call.function and call.function.arguments:
                        fragments.append(call.function.arguments)
                        times.append(time.monotonic())
    verify("openai-python", fragments, times, finish, "tool_calls")


def check_anthropic():
    client = anthropic.Anthropic(base_url=ORIGIN, api_key="fixture", max_retries=0)
    fragments, times, finish = [], [], None
    with client.messages.create(
        model=MODEL, messages=[{"role": "user", "content": "write fixture"}],
        tools=[{"name": "write", "description": "Inert fixture", "input_schema": PARAMETERS}],
        stream=True, max_tokens=4096,
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                if event.delta.partial_json:
                    fragments.append(event.delta.partial_json)
                    times.append(time.monotonic())
            if event.type == "message_delta":
                finish = event.delta.stop_reason
    verify("anthropic-python", fragments, times, finish, "tool_use")


if __name__ == "__main__":
    check_openai()
    check_anthropic()
