#!/usr/bin/env python3
"""Logit parity: higgs `lfm2_moe` vs mlx-lm (>=0.32) on the same 4-bit weights.

Two gates, both teacher-forced over the same token ids (prompt + mlx-lm greedy reply):

1. Exactness: with every float parameter cast to f32, higgs must reproduce the
   mlx-lm f32 run (the "oracle") — top-1 >= 99.9%, mean KL <= 1e-5. Any porting
   bug shows up here, free of bf16 noise.
2. bf16 equivalence: this model is very sensitive to bf16 activation rounding
   (mlx-lm bf16 itself flips ~5% of top-1 tokens vs the oracle), so in bf16 higgs
   must be no worse than mlx-lm, aggregated over all conversations:
   mean KL <= 1.25x mlx-lm's and top-1 >= mlx-lm's - 1pt.

Each step loads the 13.6 GB model, so run them one after another:

    python scripts/lfm2_parity.py ref <model_dir> <work>
    cargo run -p higgs-models --example lfm2_logits -- <model_dir> <work>/ids.json <work>/higgs
    cargo run -p higgs-models --example lfm2_logits -- <model_dir> <work>/ids.json <work>/higgs_f32 f32
    python scripts/lfm2_parity.py compare <work>      # exit 1 if a gate fails

`ref` also writes conversations.json + prompts.json (HF-rendered prompts) for the
higgs template golden test:

    HIGGS_LFM2_DIR=<model_dir> HIGGS_LFM2_PARITY=<work> \
        cargo test -p higgs-engine --test lfm2_template -- --ignored
"""

import json
import sys
from pathlib import Path

import mlx.core as mx

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
CONVERSATIONS = {
    "plain": {"messages": [{"role": "user", "content": "Explain in two short paragraphs why the sky is blue."}]},
    "tools": {
        "messages": [{"role": "user", "content": "What's the weather in Rome and in Oslo right now?"}],
        "tools": TOOLS,
    },
    "multi": {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Pick a prime number between 10 and 20."},
            {"role": "assistant", "reasoning_content": "Primes there: 11, 13, 17, 19. Pick 13.", "content": "13"},
            {"role": "user", "content": "Now write a haiku that uses that number."},
        ]
    },
    "agent": {
        "messages": [
            {"role": "user", "content": "What's the weather in Rome?"},
            {
                "role": "assistant",
                "reasoning_content": "The user wants Rome's weather, so call get_weather.",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 21, "sky": "clear"}'},
        ],
        "tools": TOOLS,
    },
    "code": {"messages": [{"role": "user", "content": "Write a Python function that merges overlapping intervals, with a short docstring."}]},
    "math": {"messages": [{"role": "user", "content": "A train leaves at 9:40 and arrives at 13:05. It stops twice for 7 minutes. What is its moving time?"}]},
    "multilingual": {"messages": [{"role": "user", "content": "Spiega in italiano, in tre frasi, cos'è la fotosintesi."}]},
}
GEN_TOKENS = 384
EXACT_TOP1, EXACT_KL = 0.999, 1e-5
KL_RATIO, TOP1_SLACK = 1.25, 0.01


def ref(model_dir: str, work: Path) -> None:
    from mlx.utils import tree_map
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    work.mkdir(parents=True, exist_ok=True)
    model, tok = load(model_dir)
    ids, prompts = {}, {}
    for name, conv in CONVERSATIONS.items():
        kw = {"tools": conv["tools"]} if "tools" in conv else {}
        prompts[name] = tok.apply_chat_template(conv["messages"], add_generation_prompt=True, tokenize=False, **kw)
        prompt = tok.apply_chat_template(conv["messages"], add_generation_prompt=True, **kw)
        gen = [r.token for r in stream_generate(model, tok, prompt, max_tokens=GEN_TOKENS, sampler=make_sampler(temp=0.0))]
        ids[name] = list(prompt) + gen
        mx.save_safetensors(str(work / f"ref_{name}.safetensors"), {"logits": model(mx.array(ids[name])[None])[0]})
        print(f"{name}: {len(prompt)} prompt + {len(gen)} generated tokens")
    # f32 oracle: same quantized weights, every float parameter (and so every activation) in f32.
    model.update(tree_map(lambda p: p.astype(mx.float32) if p.dtype == mx.bfloat16 else p, model.parameters()))
    for name, seq in ids.items():
        mx.save_safetensors(str(work / f"f32_{name}.safetensors"), {"logits": model(mx.array(seq)[None])[0]})
    (work / "ids.json").write_text(json.dumps(ids))
    (work / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=1))
    (work / "conversations.json").write_text(json.dumps(CONVERSATIONS, ensure_ascii=False, indent=1))


def kl_top1(oracle: mx.array, logits: mx.array) -> tuple[mx.array, mx.array]:
    """Per-position KL(oracle || logits) and top-1 agreement."""
    lp = oracle - mx.logsumexp(oracle, axis=-1, keepdims=True)
    lq = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    kl = (mx.exp(lp) * (lp - lq)).sum(-1)
    return kl, (oracle.argmax(-1) == logits.argmax(-1)).astype(mx.float32)


def compare(work: Path) -> bool:
    engines = ("mlx-lm", "higgs oneshot", "higgs stepped", "higgs-f32 oneshot", "higgs-f32 stepped")
    acc = {e: ([], []) for e in engines}
    print(f"{'conversation':<13}" + "".join(f"{e:>20}" for e in engines) + "   (mean KL vs f32 oracle)")
    for name in json.loads((work / "ids.json").read_text()):
        oracle = mx.load(str(work / f"f32_{name}.safetensors"))["logits"].astype(mx.float32)
        h = mx.load(str(work / "higgs" / f"{name}.safetensors"))
        h32 = mx.load(str(work / "higgs_f32" / f"{name}.safetensors"))
        logits = (mx.load(str(work / f"ref_{name}.safetensors"))["logits"], h["oneshot"], h["stepped"], h32["oneshot"], h32["stepped"])
        row = f"{name:<13}"
        for engine, x in zip(engines, logits):
            kl, top1 = kl_top1(oracle, x.astype(mx.float32))
            acc[engine][0].append(kl)
            acc[engine][1].append(top1)
            row += f"{kl.mean().item():20.2e}"
        print(row)

    total = {e: (mx.concatenate(k).mean().item(), mx.concatenate(t).mean().item()) for e, (k, t) in acc.items()}
    ref_kl, ref_top1 = total["mlx-lm"]
    ok = True
    print(f"\n{'engine':<18} {'meanKL':>9} {'top1':>8}  gate")
    for engine, (kl, top1) in total.items():
        if engine == "mlx-lm":
            gate, passed = "reference", True
        elif engine.startswith("higgs-f32"):
            gate, passed = f"exact: KL<={EXACT_KL:g}, top1>={EXACT_TOP1:.1%}", kl <= EXACT_KL and top1 >= EXACT_TOP1
        else:
            gate = f"KL<={KL_RATIO}x mlx-lm, top1>=mlx-lm-{TOP1_SLACK:.0%}"
            passed = kl <= KL_RATIO * ref_kl and top1 >= ref_top1 - TOP1_SLACK
        ok &= passed
        print(f"{engine:<18} {kl:9.2e} {top1:8.2%}  {gate} {'ok' if passed else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["ref", model_dir, work]:
            ref(model_dir, Path(work))
        case ["compare", work]:
            sys.exit(0 if compare(Path(work)) else 1)
        case _:
            sys.exit(__doc__)
