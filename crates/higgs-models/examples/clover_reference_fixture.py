#!/usr/bin/env python3
"""Build a Clover preview-forward parity fixture from DiffuSpace's reference code.

Writes two things:
  * ``<out>/slice``: the first ``--layers`` decoder layers of an MLX Clover pack
    (tensors byte-copied, config trimmed) that higgs loads like the full model.
  * ``<out>/fixture.safetensors`` + ``fixture.json``: logits from the HF
    reference ``CloverMoeModelLM`` run on the same weights (MLX affine
    dequantized), for prefill, preview and commit scenarios, plus the
    reference LowConfidence ``generate`` tokens (greedy, with/without penalties).

Needs torch, transformers==5.19.0 and safetensors, plus the reference
``modeling_clover_moe.py`` / ``configuration_clover_moe.py`` /
``sampling_clover_moe.py`` from ``Diffuspace/Clover-35B-A3B-Preview``::

    python clover_reference_fixture.py PACK_DIR REFERENCE_CODE_DIR OUT_DIR

The default 4-layer slice (3 GDN + 1 full attention) covers every preview op.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import GenerationConfig

PREFIX = "language_model."


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def collect(pack: Path, keep) -> dict[str, tuple[Path, dict, int]]:
    """key -> (file, header entry, data base offset) for keys passing ``keep``."""
    index = json.loads((pack / "model.safetensors.index.json").read_text())
    found = {}
    for file in sorted(set(index["weight_map"].values())):
        header, base = read_header(pack / file)
        for key, entry in header.items():
            if key != "__metadata__" and keep(key):
                found[key] = (pack / file, entry, base)
    return found


def raw_bytes(src: tuple[Path, dict, int]) -> bytes:
    path, entry, base = src
    start, end = entry["data_offsets"]
    with path.open("rb") as fh:
        fh.seek(base + start)
        return fh.read(end - start)


def write_slice(pack: Path, out: Path, layers: int, tensors: dict) -> None:
    """Byte-exact copy of the kept tensors plus a trimmed config."""
    out.mkdir(parents=True, exist_ok=True)
    header, offset, chunks = {}, 0, []
    for key in sorted(tensors):
        data = raw_bytes(tensors[key])
        entry = tensors[key][1]
        header[key] = {"dtype": entry["dtype"], "shape": entry["shape"],
                       "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with (out / "model.safetensors").open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for chunk in chunks:
            fh.write(chunk)

    config = json.loads((pack / "config.json").read_text())
    config["num_hidden_layers"] = layers
    config["layer_types"] = config["layer_types"][:layers]
    for name in ("quantization", "quantization_config"):
        q = config.get(name)
        if isinstance(q, dict):
            config[name] = {k: v for k, v in q.items()
                            if not isinstance(v, dict) or layer_of(k) is None or layer_of(k) < layers}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    for extra in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "generation_config.json"):
        if (pack / extra).exists():
            (out / extra).write_bytes((pack / extra).read_bytes())


def layer_of(key: str):
    parts = key.split(".")
    if "layers" in parts:
        return int(parts[parts.index("layers") + 1])
    return None


DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def tensor(src) -> torch.Tensor:
    entry = src[1]
    data = bytearray(raw_bytes(src))
    if entry["dtype"] == "U32":
        return torch.frombuffer(data, dtype=torch.int32).reshape(entry["shape"])
    return torch.frombuffer(data, dtype=DTYPES[entry["dtype"]]).reshape(entry["shape"])


def dequant(tensors: dict, stem: str, group_size: int, bits: int) -> torch.Tensor:
    """MLX affine dequantization: w = q * scale + bias per group, LSB-first packing."""
    q = tensor(tensors[stem + ".weight"]).view(torch.int32)
    scales = tensor(tensors[stem + ".scales"]).float()
    biases = tensor(tensors[stem + ".biases"]).float()
    per_word = 32 // bits
    shifts = torch.arange(per_word, dtype=torch.int32) * bits
    mask = (1 << bits) - 1
    values = (q.unsqueeze(-1) >> shifts) & mask
    values = values.reshape(*q.shape[:-1], q.shape[-1] * per_word).float()
    grouped = values.reshape(*values.shape[:-1], -1, group_size)
    w = grouped * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    return w.reshape(values.shape)


def load_reference(code_dir: Path):
    package = types.ModuleType("clover_ref")
    package.__path__ = [str(code_dir)]
    sys.modules["clover_ref"] = package
    modules = {}
    for name in ("configuration_clover_moe", "sampling_clover_moe", "modeling_clover_moe"):
        spec = importlib.util.spec_from_file_location(f"clover_ref.{name}", code_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"clover_ref.{name}"] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["configuration_clover_moe"], modules["modeling_clover_moe"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", type=Path)
    ap.add_argument("reference_code", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--new-tokens", type=int, default=64)
    args = ap.parse_args()
    torch.manual_seed(0)

    keep = lambda k: (layer_of(k) is None and k.startswith(PREFIX)) or (
        layer_of(k) is not None and layer_of(k) < args.layers)
    tensors = collect(args.pack, keep)
    write_slice(args.pack, args.out / "slice", args.layers, tensors)
    print(f"slice: {len(tensors)} tensors -> {args.out / 'slice'}")

    cfg_mod, model_mod = load_reference(args.reference_code)
    raw_cfg = json.loads((args.out / "slice" / "config.json").read_text())
    quant = raw_cfg["quantization"]
    config = cfg_mod.CloverMoeConfig(**{k: v for k, v in raw_cfg.items()
                                        if k not in ("quantization", "quantization_config",
                                                     "auto_map", "architectures")})
    with torch.device("meta"):
        model = model_mod.CloverMoeModelLM(config)

    def spec(stem):
        o = quant.get(stem, {})
        return o.get("group_size", quant["group_size"]), o.get("bits", quant["bits"])

    state = {}
    for key in tensors:
        name = key[len(PREFIX):]
        stem, _, leaf = name.rpartition(".")
        if leaf in ("scales", "biases"):
            continue
        full_stem = PREFIX + stem
        if full_stem + ".scales" in tensors:
            w = dequant(tensors, full_stem, *spec(full_stem))
            if ".switch_mlp." in name:
                continue  # handled below as fused experts
            state[stem + ".weight"] = w.to(torch.bfloat16)
        else:
            state[name] = tensor(tensors[key]).to(torch.bfloat16)
    for layer in range(args.layers):
        sw = f"{PREFIX}model.layers.{layer}.mlp.switch_mlp"
        gate, up, down = (dequant(tensors, f"{sw}.{p}", *spec(f"{sw}.{p}"))
                          for p in ("gate_proj", "up_proj", "down_proj"))
        state[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = torch.cat(
            [gate, up], dim=1).to(torch.bfloat16)
        state[f"model.layers.{layer}.mlp.experts.down_proj"] = down.to(torch.bfloat16)
        del gate, up, down
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    model.eval()
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not missing, f"missing keys: {missing[:5]}"

    mask = config.mask_token_id
    prompt = torch.tensor([[248045, 846, 198, 3838, 374, 279, 6722, 315, 9625, 30, 248046, 198,
                            248045, 74455, 198]])  # deterministic token ids; content irrelevant
    out = {}
    with torch.inference_mode():
        pre = model(prompt, use_cache=True)
        out["prefill_logits"] = pre.logits[0].float()
        cache = pre.past_key_values

        def preview(block, tag):
            res = model(torch.tensor([block]), past_key_values=cache, use_cache=True, preview=True)
            out[tag] = res.logits[0].float()
            return res.logits[0].float().argmax(-1).tolist()

        all_mask = [mask] * config.block_size
        first = preview(all_mask, "preview_mask32_logits")
        partial = list(all_mask)
        partial[:5] = first[:5]
        preview(partial, "preview_partial32_logits")
        preview([mask] * 8, "preview_mask8_logits")
        cache = model(torch.tensor([first]), past_key_values=cache, use_cache=True).past_key_values
        preview(all_mask, "preview_after_commit_logits")

        # Reference LowConfidence `generate` (greedy), with and without the
        # response penalties; "ramps" moves the length/think-end ramps into a
        # 64-token window so their bias paths are exercised too. Threshold 0
        # resolves each block in one pass (per-position argmax, high margin);
        # 0.95 is the trained setting, whose unmask order on this 4-layer
        # slice is decided by near-uniform (~1%) confidences.
        pack = json.loads((args.out / "slice" / "generation_config.json").read_text())
        penalties = {k: pack[k] for k in (
            "presence_penalty", "frequency_penalty", "repetition_penalty",
            "dllm_length_penalty", "dllm_length_penalty_start", "dllm_length_penalty_window",
            "dllm_think_end_bias", "dllm_think_end_start", "dllm_think_end_window",
            "dllm_think_end_token_id")}
        off = {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
               "dllm_length_penalty": 0.0, "dllm_think_end_bias": 0.0}
        ramps = dict(penalties, dllm_length_penalty_start=0, dllm_length_penalty_window=64,
                     dllm_think_end_start=0, dllm_think_end_window=32)
        generations = []
        for threshold in (0.0, 0.95):
            for name, extra in (("greedy", off), ("pack_penalties", penalties), ("ramps", ramps)):
                kwargs = dict(extra, max_new_tokens=args.new_tokens, do_sample=False,
                              eos_token_id=pack["eos_token_id"],
                              pad_token_id=pack["pad_token_id"],
                              block_size=config.block_size, threshold=threshold)
                seq = model.generate(prompt, generation_config=GenerationConfig(), **kwargs)
                tokens = seq[0, prompt.shape[1]:].tolist()
                name = f"{name}@{threshold}"
                generations.append({"name": name, "settings": kwargs, "tokens": tokens})
                print(f"generate {name}: {len(tokens)} tokens")

    args.out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(args.out / "fixture.safetensors"))
    (args.out / "fixture.json").write_text(json.dumps({
        "prompt": prompt[0].tolist(), "mask_token_id": mask, "block_size": config.block_size,
        "first_block": first, "partial_block": partial, "layers": args.layers,
        "generate": generations,
    }, indent=1))
    print("fixture:", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
