#!/usr/bin/env python3
"""Fused int8 projection mlpackage: N conv1x1 + constexpr_affine_dequantize
ops sharing one input `x`, producing N named outputs in a single MIL program.

Purpose: measure (via the Rust `dflash_ane_fusion_probe` test) whether the
per-dispatch overhead we bounded in `dflash_ane_dispatch_overhead_probe`
(~88 us per ANE-engaged predict call) amortizes across outputs when they
share a program, or whether ANE still serializes each op.

Sibling to `quantize_int8_proj.py` — same quantization, same conv1x1 layout,
same ANE iOS18 target. Only difference is N outputs instead of 1.

Inputs:
  --in-features INT           Shared input dim (C_IN for every conv).
  --seq-len INT               Compile-time W dim of [1, C_IN, 1, W] input.
  --proj N:OUT:WEIGHTS_BIN    Repeatable. `N` is an identifier (used in the
                              output name `y_<N>`); OUT is that projection's
                              C_OUT; WEIGHTS_BIN is f32 LE row-major of shape
                              [OUT, in_features].
  --out-dir PATH              Target directory; `model.mlmodelc/` written here.

Output (stdout JSON one-liner):
  {"mlmodelc": "<path>",
   "quant_ms": <int>, "compile_ms": <int>,
   "input_name": "x",
   "outputs": [{"name": "y_<N>", "out_features": <int>,
                "scale": <float>, "max_abs_w": <float>}, ...]}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


INPUT_NAME = "x"


def _load_weights(path: Path, out_features: int, in_features: int) -> np.ndarray:
    expected_bytes = out_features * in_features * 4
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"quantize_int8_fused: {path} size {actual_bytes} != "
            f"out*in*4 = {expected_bytes} (out={out_features}, in={in_features})"
        )
    arr = np.fromfile(str(path), dtype="<f4", count=out_features * in_features)
    return arr.reshape(out_features, in_features)


def _quantize_symmetric(w_f32: np.ndarray) -> tuple[np.ndarray, float, float]:
    max_abs = float(np.max(np.abs(w_f32)))
    scale = (max_abs / 127.0) if max_abs > 0.0 else (1.0 / 127.0)
    w_int8 = np.round(w_f32 / scale).clip(-127, 127).astype(np.int8)
    return w_int8, scale, max_abs


def _build_mlpackage(
    projs: list[dict],
    in_features: int,
    seq_len: int,
    out_path: Path,
) -> None:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, in_features, 1, seq_len), dtype=types.fp16)
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        outs = []
        for p in projs:
            w_conv = p["w_int8"].reshape(p["out_features"], in_features, 1, 1)
            w = mb.constexpr_affine_dequantize(
                quantized_data=w_conv,
                zero_point=np.int8(0),
                scale=np.float16(p["scale"]),
                axis=0,
            )
            y = mb.conv(
                x=x,
                weight=w,
                strides=[1, 1],
                pad_type="valid",
                dilations=[1, 1],
                name=p["output_name"],
            )
            outs.append(y)
        return tuple(outs)

    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    mlmodel.save(str(out_path))


def _compile(mlpackage_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(mlpackage_dir), str(out_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"quantize_int8_fused: coremlcompiler failed "
            f"(exit={result.returncode})\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    expected = out_dir / (mlpackage_dir.stem + ".mlmodelc")
    if not expected.is_dir():
        candidates = list(out_dir.glob("*.mlmodelc"))
        if len(candidates) != 1:
            raise SystemExit(
                f"quantize_int8_fused: expected one .mlmodelc in {out_dir}, "
                f"found {len(candidates)}"
            )
        expected = candidates[0]
    return expected


def _parse_proj(spec: str) -> tuple[str, int, Path]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(
            f"quantize_int8_fused: --proj expects NAME:OUT:WEIGHTS_BIN, got {spec!r}"
        )
    name, out_s, bin_s = parts
    if not name:
        raise SystemExit("quantize_int8_fused: --proj NAME must be non-empty")
    try:
        out_features = int(out_s)
    except ValueError as e:
        raise SystemExit(f"quantize_int8_fused: bad OUT in {spec!r}: {e}") from e
    return name, out_features, Path(bin_s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse N int8 projections into one CoreML mlpackage."
    )
    parser.add_argument("--in-features", required=True, type=int)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument(
        "--proj",
        required=True,
        action="append",
        help="NAME:OUT_FEATURES:WEIGHTS_BIN (repeatable, ≥1)",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.in_features <= 0 or args.seq_len <= 0:
        raise SystemExit("quantize_int8_fused: in-features / seq-len must be > 0")

    specs = [_parse_proj(s) for s in args.proj]
    if not specs:
        raise SystemExit("quantize_int8_fused: need ≥1 --proj")

    t0 = time.monotonic()
    projs: list[dict] = []
    for name, out_features, wpath in specs:
        if not wpath.is_file():
            raise SystemExit(f"quantize_int8_fused: weights not found: {wpath}")
        w_f32 = _load_weights(wpath, out_features, args.in_features)
        w_int8, scale, max_abs = _quantize_symmetric(w_f32)
        projs.append({
            "name": name,
            "output_name": f"y_{name}",
            "out_features": out_features,
            "w_int8": w_int8,
            "scale": scale,
            "max_abs_w": max_abs,
        })
    quant_ms = int((time.monotonic() - t0) * 1000)

    with tempfile.TemporaryDirectory(prefix="higgs-fused-int8-") as tmp:
        tmp_path = Path(tmp)
        mlpackage_dir = tmp_path / "model.mlpackage"
        _build_mlpackage(projs, args.in_features, args.seq_len, mlpackage_dir)

        t1 = time.monotonic()
        mlmodelc = _compile(mlpackage_dir, tmp_path)
        compile_ms = int((time.monotonic() - t1) * 1000)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        target = args.out_dir / "model.mlmodelc"
        if target.exists():
            raise SystemExit(f"quantize_int8_fused: {target} already exists")
        try:
            mlmodelc.rename(target)
        except OSError:
            import shutil
            shutil.copytree(str(mlmodelc), str(target))

    print(json.dumps({
        "mlmodelc": str(target.resolve()),
        "quant_ms": quant_ms,
        "compile_ms": compile_ms,
        "input_name": INPUT_NAME,
        "outputs": [
            {
                "name": p["name"],
                "output_name": p["output_name"],
                "out_features": p["out_features"],
                "scale": p["scale"],
                "max_abs_w": p["max_abs_w"],
            }
            for p in projs
        ],
    }))


if __name__ == "__main__":
    main()
