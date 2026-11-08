#!/usr/bin/env python3
"""Independent LFM3 parity oracle for the official MLX affine-q4 checkpoint.

The reference deliberately does not execute the Rust model or MLX model code:
it implements the reviewed upstream Transformers equations with CPU PyTorch and
dequantizes the packed MLX weights itself.  The q4 unpacking convention is the
one documented by MLX (eight low-to-high nibbles per uint32, groups of 64).

Typical use (the f32 run is the primary porting-bug gate)::

    python scripts/lfm3_parity.py ref <model_dir> <modeling_lfm3.py> <work>
    cargo run -p higgs-models --example lfm3_logits -- \
        <model_dir> <work>/ids.json <work>/higgs_f32 f32
    python scripts/lfm3_parity.py compare <work>

An optional bf16 comparison characterizes activation-rounding drift::

    python scripts/lfm3_parity.py ref-bf16 <model_dir> <modeling_lfm3.py> <work>
    cargo run -p higgs-models --example lfm3_logits -- \
        <model_dir> <work>/ids.json <work>/higgs
    python scripts/lfm3_parity.py compare <work>

Both reference commands also write HF-rendered ``conversations.json`` and
``prompts.json`` for ``crates/higgs-engine/tests/lfm3_template.rs``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


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

CONVERSATIONS: dict[str, dict[str, Any]] = {
    "plain": {
        "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    },
    "tools": {
        "messages": [{"role": "user", "content": "What's the weather in Rome?"}],
        "tools": TOOLS,
    },
    "multi": {
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Pick a prime from 10 to 20."},
            {
                "role": "assistant",
                "reasoning_content": "The primes are 11, 13, 17, and 19.",
                "content": "13",
            },
            {"role": "user", "content": "Double it."},
        ]
    },
    "agent": {
        "messages": [
            {"role": "user", "content": "What's the weather in Rome?"},
            {
                "role": "assistant",
                "reasoning_content": "I need current data, so I will call the tool.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Rome"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temp_c": 21, "sky": "clear"}',
            },
        ],
        "tools": TOOLS,
    },
}

# This stays intentionally short: a CPU oracle must unpack all 1.7B weights,
# and sequence length is not what this gate is trying to benchmark.
PROBE_TEXTS = {
    "probe": "Parity check: one two three four.",
    "positions": (
        "Position check: zero one two three four five six seven eight nine; "
        "then repeat nine eight seven six five four three two one zero."
    ),
    "multilingual": "Verifica numerica: il cielo è blu; 東京の空も青い。",
}

F32_KL_LIMIT = 1.0e-4
F32_TOP1_LIMIT = 0.99
BF16_KL_RATIO = 1.5
BF16_TOP1_SLACK = 0.02

CHECKPOINT_BYTES = 973_083_386
CHECKPOINT_SHA256 = "1be5bf672013cd91aa62148f77c5eea8d2784a95845d5fe826e08c7b84823815"
# SHA-256 of the checked-out source body.  The corresponding Hugging Face MLX
# tree Git blob recorded during inspection is 0ccffc7ed179c2e3aaa19ab93bba735abb1d124d.
MODELING_SHA256 = "e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab"
TEMPLATE_SHA256 = "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818"
CONFIG_SHA256 = "7df2ea819f9a31b770192a0a9dc1b7953a7dd8c41b816cc3905bbec99ef332bd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(model_dir: Path, modeling_source: Path) -> None:
    checkpoint = model_dir / "model.safetensors"
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("model.safetensors is not the pinned official LFM3-1.7B MLX 4-bit checkpoint")
    for name, expected in (
        ("config.json", CONFIG_SHA256),
        ("chat_template.jinja", TEMPLATE_SHA256),
    ):
        path = model_dir / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{name} does not match the reviewed upstream source")
    if not modeling_source.is_file() or sha256(modeling_source) != MODELING_SHA256:
        raise ValueError("modeling_lfm3.py does not match the reviewed upstream source")


class Q4Checkpoint:
    """Read an MLX affine-q4 safetensors file and dequantize lazily."""

    def __init__(self, model_dir: Path, dtype: "torch.dtype") -> None:
        import torch
        from safetensors import safe_open

        self.torch = torch
        self.dtype = dtype
        self.reader = safe_open(
            str(model_dir / "model.safetensors"), framework="pt", device="cpu"
        )
        self.prefix = "language_model."
        if not any(k.startswith(self.prefix) for k in self.reader.keys()):
            self.prefix = ""

    def tensor(self, name: str) -> "torch.Tensor":
        return self.reader.get_tensor(self.prefix + name)

    def _dequant_parts(
        self,
        packed: "torch.Tensor",
        scales: "torch.Tensor",
        biases: "torch.Tensor",
    ) -> "torch.Tensor":
        torch = self.torch
        shifts = torch.arange(8, dtype=torch.int64) * 4
        q = ((packed.to(torch.int64)[..., None] >> shifts) & 0xF).flatten(-2)
        # MLX computes the affine expression in float and rounds once to the
        # requested result dtype.  Doing the multiply itself in bf16 rounds
        # twice and does not match mx.dequantize.
        weight = (
            q.float() * scales.float().repeat_interleave(64, dim=-1)
            + biases.float().repeat_interleave(64, dim=-1)
        )
        return weight.to(self.dtype)

    def dequant(self, name: str) -> "torch.Tensor":
        return self._dequant_parts(
            self.tensor(name + ".weight"),
            self.tensor(name + ".scales"),
            self.tensor(name + ".biases"),
        )

    def linear(self, x: "torch.Tensor", name: str) -> "torch.Tensor":
        return self.torch.nn.functional.linear(x, self.dequant(name))

    def embedding(self, ids: "torch.Tensor") -> "torch.Tensor":
        name = "model.embed_tokens"
        return self._dequant_parts(
            self.tensor(name + ".weight")[ids],
            self.tensor(name + ".scales")[ids],
            self.tensor(name + ".biases")[ids],
        )

    def tied_logits(self, hidden: "torch.Tensor", chunk: int = 4096) -> "torch.Tensor":
        name = "model.embed_tokens"
        packed = self.tensor(name + ".weight")
        scales = self.tensor(name + ".scales")
        biases = self.tensor(name + ".biases")
        parts = []
        for start in range(0, packed.shape[0], chunk):
            stop = min(start + chunk, packed.shape[0])
            weight = self._dequant_parts(
                packed[start:stop], scales[start:stop], biases[start:stop]
            )
            parts.append(self.torch.nn.functional.linear(hidden, weight))
        return self.torch.cat(parts, dim=-1)


def rms_norm(x: "torch.Tensor", weight: "torch.Tensor", eps: float) -> "torch.Tensor":
    import torch

    dtype = x.dtype
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return weight.to(dtype) * normalized.to(dtype)


def rotate_half(x: "torch.Tensor") -> "torch.Tensor":
    import torch

    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class TorchLfm3:
    """Inference-only, cacheless transcription of upstream modeling_lfm3.py."""

    def __init__(self, model_dir: Path, dtype: "torch.dtype") -> None:
        self.model_dir = model_dir
        self.cfg = json.loads((model_dir / "config.json").read_text())
        self.dtype = dtype
        self.ckpt = Q4Checkpoint(model_dir, dtype)
        self.hidden = int(self.cfg["hidden_size"])
        self.head_dim = int(self.cfg["head_dim"])
        self.q_heads = int(self.cfg["num_attention_heads"])
        self.kv_heads = int(self.cfg["num_key_value_heads"])
        self.eps = float(self.cfg["norm_eps"])

    def norm(self, x: "torch.Tensor", name: str) -> "torch.Tensor":
        return rms_norm(x, self.ckpt.tensor(name + ".weight"), self.eps)

    def rope(
        self, q: "torch.Tensor", k: "torch.Tensor", layer_type: str, offset: int
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        params = self.cfg["rope_parameters"][layer_type]
        partial = float(params.get("partial_rotary_factor", 1.0))
        angles = int(partial * self.head_dim) // 2
        inv = 1.0 / (
            float(params["rope_theta"])
            ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32) / self.head_dim)
        )
        inv = torch.cat((inv, torch.zeros(self.head_dim // 2 - angles)))
        positions = torch.arange(offset, offset + q.shape[-2], dtype=torch.float32)
        freqs = positions[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        # Upstream returns cos/sin in the activation dtype, then the rotary
        # helper re-upcasts those already-rounded values for f32 arithmetic.
        cos = emb.cos()[None, None].to(self.dtype).float()
        sin = emb.sin()[None, None].to(self.dtype).float()
        q_out = q.float() * cos + rotate_half(q.float()) * sin
        k_out = k.float() * cos + rotate_half(k.float()) * sin
        return q_out.to(self.dtype), k_out.to(self.dtype)

    def attention(
        self,
        x: "torch.Tensor",
        base: str,
        layer_type: str,
        cache: dict[str, "torch.Tensor"] | None,
        offset: int,
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        import torch

        bsz, length, _ = x.shape
        q = self.ckpt.linear(x, base + ".q_proj").view(
            bsz, length, self.q_heads, self.head_dim
        )
        k = self.ckpt.linear(x, base + ".k_proj").view(
            bsz, length, self.kv_heads, self.head_dim
        )
        v = self.ckpt.linear(x, base + ".v_proj").view(
            bsz, length, self.kv_heads, self.head_dim
        )
        q = self.norm(q, base + ".q_norm").transpose(1, 2)
        k = self.norm(k, base + ".k_norm").transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope(q, k, layer_type, offset)

        if cache is not None:
            k = torch.cat((cache["k"], k), dim=2)
            v = torch.cat((cache["v"], v), dim=2)
        new_cache = {"k": k, "v": v}

        repeats = self.q_heads // self.kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q, k.transpose(2, 3)) * (self.head_dim**-0.5)
        row = torch.arange(offset, offset + length)[:, None]
        col = torch.arange(k.shape[2])[None, :]
        allowed = col <= row
        if layer_type == "sliding_attention":
            allowed &= col > row - int(self.cfg["sliding_window"])
        scores = scores.masked_fill(~allowed[None, None], float("-inf"))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)
        out = torch.matmul(probs, v).transpose(1, 2)
        gate = torch.sigmoid(self.ckpt.linear(x, base + ".gate_proj")).unsqueeze(-1)
        out = (out * gate).reshape(bsz, length, -1).contiguous()
        return self.ckpt.linear(out, base + ".o_proj"), new_cache

    def convolution(
        self,
        x: "torch.Tensor",
        base: str,
        cache: dict[str, "torch.Tensor"] | None,
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        import torch
        import torch.nn.functional as F

        projected = self.ckpt.linear(x, base + ".in_proj").transpose(-1, -2)
        b_gate, c_gate, values = projected.chunk(3, dim=-2)
        values = b_gate * values
        state_length = int(self.cfg["conv_L_cache"]) - 1
        state = (
            cache["state"]
            if cache is not None
            else torch.zeros(
                values.shape[0], values.shape[1], state_length, dtype=self.dtype
            )
        )
        padded = torch.cat((state, values), dim=-1)
        new_cache = {"state": padded[..., -state_length:]}
        weight = self.ckpt.tensor(base + ".conv.weight").transpose(1, 2).to(self.dtype)
        convolved = F.conv1d(padded, weight, groups=self.hidden)
        out = (c_gate * convolved).transpose(-1, -2).contiguous()
        return self.ckpt.linear(out, base + ".out_proj"), new_cache

    def mlp(self, x: "torch.Tensor", base: str) -> "torch.Tensor":
        import torch.nn.functional as F

        return self.ckpt.linear(
            F.silu(self.ckpt.linear(x, base + ".w1"))
            * self.ckpt.linear(x, base + ".w3"),
            base + ".w2",
        )

    def forward_chunk(
        self,
        ids: list[int],
        caches: list[dict[str, "torch.Tensor"] | None],
        offset: int,
        progress: str,
    ) -> tuple["torch.Tensor", list[dict[str, "torch.Tensor"] | None]]:
        import torch

        token_ids = torch.tensor(ids, dtype=torch.long)[None, :]
        hidden = self.ckpt.embedding(token_ids)
        started = time.monotonic()
        next_caches: list[dict[str, torch.Tensor] | None] = []
        with torch.inference_mode():
            for index, layer_type in enumerate(self.cfg["layer_types"]):
                base = f"model.layers.{index}"
                residual = hidden
                operator_in = self.norm(hidden, base + ".operator_norm")
                if layer_type == "conv":
                    branch, layer_cache = self.convolution(
                        operator_in, base + ".conv", caches[index]
                    )
                else:
                    branch, layer_cache = self.attention(
                        operator_in,
                        base + ".self_attn",
                        layer_type,
                        caches[index],
                        offset,
                    )
                next_caches.append(layer_cache)
                if self.cfg.get("use_sandwich_norm", False):
                    branch = self.norm(branch, base + ".post_operator_norm")
                hidden = residual + branch

                branch = self.mlp(
                    self.norm(hidden, base + ".ffn_norm"), base + ".feed_forward"
                )
                if self.cfg.get("use_sandwich_norm", False):
                    branch = self.norm(branch, base + ".post_ffn_norm")
                hidden = hidden + branch
                print(
                    f"{progress} layer {index + 1:02d}/{len(self.cfg['layer_types'])} "
                    f"{layer_type} ({time.monotonic() - started:.1f}s)",
                    flush=True,
                )
            hidden = self.norm(hidden, "model.embedding_norm")
            return self.ckpt.tied_logits(hidden)[0], next_caches

    def __call__(self, ids: list[int]) -> "torch.Tensor":
        caches = [None] * len(self.cfg["layer_types"])
        logits, _ = self.forward_chunk(ids, caches, 0, "oneshot")
        return logits

    def stepped(self, ids: list[int]) -> "torch.Tensor":
        import torch

        length = len(ids)
        decode_tail = min(4, length - 2)
        prefill_end = length - decode_tail
        split = max(prefill_end // 2, 1)
        ranges = [(0, split), (split, prefill_end)] + [
            (index, index + 1) for index in range(prefill_end, length)
        ]
        caches = [None] * len(self.cfg["layer_types"])
        parts = []
        for part, (start, stop) in enumerate(ranges):
            logits, caches = self.forward_chunk(
                ids[start:stop], caches, start, f"stepped[{part}]"
            )
            parts.append(logits)
        return torch.cat(parts, dim=0)


def write_fixtures(
    model_dir: Path, modeling_source: Path, work: Path
) -> dict[str, list[int]]:
    from transformers import PreTrainedTokenizerFast

    verify_inputs(model_dir, modeling_source)
    work.mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    prompts: dict[str, str] = {}
    for name, conversation in CONVERSATIONS.items():
        kwargs = {"tools": conversation["tools"]} if "tools" in conversation else {}
        prompts[name] = tokenizer.apply_chat_template(
            conversation["messages"],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
            **kwargs,
        )
    ids = {
        name: [int(tokenizer.bos_token_id)]
        + tokenizer.encode(text, add_special_tokens=False)
        for name, text in PROBE_TEXTS.items()
    }
    (work / "ids.json").write_text(json.dumps(ids, indent=1))
    (work / "prompts.json").write_text(
        json.dumps(prompts, ensure_ascii=False, indent=1)
    )
    (work / "conversations.json").write_text(
        json.dumps(CONVERSATIONS, ensure_ascii=False, indent=1)
    )
    for name, probe in ids.items():
        print(f"{name}: {len(probe)} tokens")
    return ids


def reference(model_dir: Path, modeling_source: Path, work: Path, bf16: bool) -> None:
    import torch
    from safetensors.torch import save_file

    ids = write_fixtures(model_dir, modeling_source, work)
    dtype = torch.bfloat16 if bf16 else torch.float32
    tag = "bf16" if bf16 else "f32"
    model = TorchLfm3(model_dir, dtype)
    for name, sequence in ids.items():
        logits = model(sequence).float().contiguous()
        save_file({"logits": logits}, str(work / f"ref_{tag}_{name}.safetensors"))
        print(f"saved {tag} {name}: {tuple(logits.shape)}")
        stepped = model.stepped(sequence).float().contiguous()
        save_file(
            {"logits": stepped},
            str(work / f"ref_{tag}_stepped_{name}.safetensors"),
        )
        print(f"saved {tag} stepped {name}: {tuple(stepped.shape)}")


def metrics(oracle: "torch.Tensor", actual: "torch.Tensor") -> tuple[float, float]:
    import torch

    lp = torch.log_softmax(oracle.float(), dim=-1)
    lq = torch.log_softmax(actual.float(), dim=-1)
    # Floating summation can produce a tiny negative value for effectively
    # identical distributions; mathematical KL is non-negative.
    kl = max(0.0, (lp.exp() * (lp - lq)).sum(-1).mean().item())
    top1 = (oracle.argmax(-1) == actual.argmax(-1)).float().mean().item()
    return kl, top1


def compare(work: Path) -> bool:
    from safetensors.torch import load_file

    names = json.loads((work / "ids.json").read_text()).keys()
    ok = True
    print(f"{'case':<12} {'engine':<18} {'mean KL':>12} {'top-1':>9}  gate")
    for name in names:
        f32_oracles = {
            "oneshot": load_file(work / f"ref_f32_{name}.safetensors")["logits"],
            "stepped": load_file(work / f"ref_f32_stepped_{name}.safetensors")[
                "logits"
            ],
        }
        f32_path = work / "higgs_f32" / f"{name}.safetensors"
        if not f32_path.exists():
            print(f"{name:<12} {'higgs f32':<18} missing")
            ok = False
            continue
        rust_f32 = load_file(f32_path)
        for key in ("oneshot", "stepped"):
            kl, top1 = metrics(f32_oracles[key], rust_f32[key])
            passed = kl <= F32_KL_LIMIT and top1 >= F32_TOP1_LIMIT
            ok &= passed
            print(
                f"{name:<12} {('higgs f32 ' + key):<18} {kl:12.3e} {top1:9.2%}  "
                f"KL<={F32_KL_LIMIT:g}, top1>={F32_TOP1_LIMIT:.0%} "
                f"{'ok' if passed else 'FAIL'}"
            )

        ref_bf16_path = work / f"ref_bf16_{name}.safetensors"
        ref_bf16_stepped_path = work / f"ref_bf16_stepped_{name}.safetensors"
        rust_bf16_path = work / "higgs" / f"{name}.safetensors"
        if (
            ref_bf16_path.exists()
            and ref_bf16_stepped_path.exists()
            and rust_bf16_path.exists()
        ):
            ref_bf16 = {
                "oneshot": load_file(ref_bf16_path)["logits"],
                "stepped": load_file(ref_bf16_stepped_path)["logits"],
            }
            rust_bf16 = load_file(rust_bf16_path)
            for key in ("oneshot", "stepped"):
                baseline_kl, baseline_top1 = metrics(
                    f32_oracles[key], ref_bf16[key]
                )
                kl_limit = max(F32_KL_LIMIT, BF16_KL_RATIO * baseline_kl)
                top1_limit = baseline_top1 - BF16_TOP1_SLACK
                kl, top1 = metrics(f32_oracles[key], rust_bf16[key])
                passed = kl <= kl_limit and top1 >= top1_limit
                ok &= passed
                print(
                    f"{name:<12} {('higgs bf16 ' + key):<18} {kl:12.3e} {top1:9.2%}  "
                    f"vs torch bf16 KL={baseline_kl:.3e}, top1={baseline_top1:.2%} "
                    f"{'ok' if passed else 'FAIL'}"
                )

            torch_schedule_kl, torch_schedule_top1 = metrics(
                ref_bf16["oneshot"], ref_bf16["stepped"]
            )
            rust_schedule_kl, rust_schedule_top1 = metrics(
                rust_bf16["oneshot"], rust_bf16["stepped"]
            )
            print(
                f"{name:<12} {'torch bf16 sched':<18} {torch_schedule_kl:12.3e} "
                f"{torch_schedule_top1:9.2%}  oneshot vs stepped (diagnostic)"
            )
            print(
                f"{name:<12} {'higgs bf16 sched':<18} {rust_schedule_kl:12.3e} "
                f"{rust_schedule_top1:9.2%}  oneshot vs stepped (diagnostic)"
            )
    print("PASS" if ok else "FAIL")
    return ok


def selftest() -> None:
    import torch

    # Literal nibble order and affine values, independent of the implementation.
    packed = torch.tensor([[0x76543210] + [0] * 7], dtype=torch.uint32)
    scales = torch.tensor([[0.5]], dtype=torch.bfloat16)
    biases = torch.tensor([[-1.0]], dtype=torch.bfloat16)
    obj = object.__new__(Q4Checkpoint)
    obj.torch, obj.dtype = torch, torch.float32
    got = obj._dequant_parts(packed, scales, biases)
    expected = torch.tensor(
        [[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5] + [-1.0] * 56]
    )
    torch.testing.assert_close(got, expected)
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    torch.testing.assert_close(rotate_half(x), torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))
    print("selftest PASS")


def input_selftest(model_dir: Path, modeling_source: Path) -> None:
    import tempfile

    verify_inputs(model_dir, modeling_source)
    with tempfile.TemporaryDirectory(prefix="lfm3-input-selftest-") as raw:
        changed = Path(raw)
        for name in ("model.safetensors", "chat_template.jinja"):
            source = model_dir / name
            if source.exists():
                (changed / name).symlink_to(source)
        (changed / "config.json").write_bytes(
            (model_dir / "config.json").read_bytes() + b"\n"
        )
        try:
            verify_inputs(changed, modeling_source)
        except ValueError as error:
            assert "config.json" in str(error), error
        else:
            raise AssertionError("modified config.json was not rejected")
        changed_source = changed / "modeling_lfm3.py"
        changed_source.write_bytes(modeling_source.read_bytes() + b"\n")
        try:
            verify_inputs(model_dir, changed_source)
        except ValueError as error:
            assert "modeling_lfm3.py" in str(error), error
        else:
            raise AssertionError("modified modeling_lfm3.py was not rejected")
    print(
        "input selftest PASS "
        "(official accepted, modified config/source rejected)"
    )


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["selftest"]:
            selftest()
        case ["selftest-inputs", model_dir, modeling_source]:
            input_selftest(Path(model_dir), Path(modeling_source))
        case ["fixtures", model_dir, modeling_source, work]:
            write_fixtures(Path(model_dir), Path(modeling_source), Path(work))
        case ["ref", model_dir, modeling_source, work]:
            reference(Path(model_dir), Path(modeling_source), Path(work), bf16=False)
        case ["ref-bf16", model_dir, modeling_source, work]:
            reference(Path(model_dir), Path(modeling_source), Path(work), bf16=True)
        case ["compare", work]:
            raise SystemExit(0 if compare(Path(work)) else 1)
        case _:
            raise SystemExit(__doc__)
