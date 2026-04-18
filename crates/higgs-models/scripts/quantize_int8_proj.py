#!/usr/bin/env python3
"""Symmetric per-tensor int8 quantization of a single projection weight,
emitted as a CoreML `.mlmodelc` bundle suitable for the Apple Neural Engine.

Sibling to `palettize_lm_head.py`. Where that one targets the LM head via
6-bit palettization, this one targets DFlash attention/MLP projections via
int8 `constexpr_affine_dequantize` — the only int8 weight path on ANE that
survives `ct.convert` + `coremlcompiler` (AB5/AB6, 2026-04-18).

Called by the Rust parity/latency tests in `ane_mlmodel.rs`, and will be
called by the DFlash forward path once parity lands.

Toolchain requirement: the project `.venv` is Python 3.14 and has broken
`libcoremlpython` (CLAIMS.md T1). Run this through a 3.13 sidecar, e.g.
`benchmarks/ane_int8_mlpackage_probe/.venv/bin/python`. The Rust side
passes the interpreter path via env var — see `ane_mlmodel.rs`.

Inputs:
  --weights-bin PATH    Row-major f32 LE bytes, shape [out_features, in_features].
  --out-features INT    Row count (output dim; C_OUT for conv1x1).
  --in-features INT     Column count (input dim; C_IN for conv1x1).
  --seq-len INT         Compile-time sequence length (the `W` of the [1,C,1,W]
                        conv1x1 input). Runtime seq must equal this (padded
                        by the caller if smaller).
  --out-dir PATH        Target directory. `model.mlmodelc/` is written here.

Output (stdout, one line of JSON):
  {"mlmodelc": "<abs-path>/model.mlmodelc",
   "quant_ms":   <int>,
   "compile_ms": <int>,
   "input_name": "x",
   "output_name": "y",
   "scale":      <float>,
   "max_abs_w":  <float>}

Errors go to stderr with non-zero exit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# ABI with `ane_mlmodel.rs` — passed verbatim to `predictionFromFeatures:`.
INPUT_NAME = "x"
OUTPUT_NAME = "y"


def _load_weights(path: Path, out_features: int, in_features: int) -> np.ndarray:
    expected_bytes = out_features * in_features * 4
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"quantize_int8_proj: --weights-bin size {actual_bytes} != "
            f"out*in*4 = {expected_bytes} "
            f"(out={out_features}, in={in_features})"
        )
    arr = np.fromfile(str(path), dtype="<f4", count=out_features * in_features)
    return arr.reshape(out_features, in_features)


def _quantize_symmetric(w_f32: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Symmetric per-tensor int8: scale = max(|w|) / 127, zp = 0.

    Mirrors `build_weight_blob_quantized` in `ane_bridge.rs` (the reference
    used by the raw-MIL path we're NOT extending).
    """
    max_abs = float(np.max(np.abs(w_f32)))
    if max_abs == 0.0:
        # Degenerate; keep everything zero so scale stays finite.
        scale = 1.0 / 127.0
    else:
        scale = max_abs / 127.0
    w_int8 = np.round(w_f32 / scale).clip(-127, 127).astype(np.int8)
    return w_int8, scale, max_abs


def _build_mlpackage(
    w_int8: np.ndarray,
    scale: float,
    in_features: int,
    out_features: int,
    seq_len: int,
    out_path: Path,
) -> None:
    """Build conv1x1 + constexpr_affine_dequantize via coremltools."""
    # Deferred imports — `--help` should not pay the ~3 s startup cost.
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    # Reshape weight to conv1x1 layout: [C_OUT, C_IN, 1, 1].
    w_conv = w_int8.reshape(out_features, in_features, 1, 1)

    scale_fp16 = np.float16(scale)
    zero_point = np.int8(0)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, in_features, 1, seq_len), dtype=types.fp16)
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        w = mb.constexpr_affine_dequantize(
            quantized_data=w_conv,
            zero_point=zero_point,
            scale=scale_fp16,
            axis=0,
        )
        y = mb.conv(
            x=x,
            weight=w,
            strides=[1, 1],
            pad_type="valid",
            dilations=[1, 1],
            name=OUTPUT_NAME,
        )
        return y

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
            f"quantize_int8_proj: coremlcompiler failed "
            f"(exit={result.returncode})\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    expected = out_dir / (mlpackage_dir.stem + ".mlmodelc")
    if not expected.is_dir():
        candidates = list(out_dir.glob("*.mlmodelc"))
        if len(candidates) != 1:
            raise SystemExit(
                f"quantize_int8_proj: expected exactly one .mlmodelc in "
                f"{out_dir}, found {len(candidates)}"
            )
        expected = candidates[0]
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a projection weight to int8 and compile to CoreML .mlmodelc"
    )
    parser.add_argument("--weights-bin", required=True, type=Path)
    parser.add_argument("--out-features", required=True, type=int)
    parser.add_argument("--in-features", required=True, type=int)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.out_features <= 0 or args.in_features <= 0 or args.seq_len <= 0:
        raise SystemExit("quantize_int8_proj: out-features/in-features/seq-len must be > 0")
    if not args.weights_bin.is_file():
        raise SystemExit(f"quantize_int8_proj: weights-bin not found: {args.weights_bin}")

    w_f32 = _load_weights(args.weights_bin, args.out_features, args.in_features)

    t0 = time.monotonic()
    w_int8, scale, max_abs = _quantize_symmetric(w_f32)
    quant_ms = int((time.monotonic() - t0) * 1000)

    with tempfile.TemporaryDirectory(prefix="higgs-quant-int8-") as tmp:
        tmp_path = Path(tmp)
        mlpackage_dir = tmp_path / "model.mlpackage"
        _build_mlpackage(
            w_int8, scale, args.in_features, args.out_features, args.seq_len, mlpackage_dir
        )

        t1 = time.monotonic()
        mlmodelc = _compile(mlpackage_dir, tmp_path)
        compile_ms = int((time.monotonic() - t1) * 1000)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        target = args.out_dir / "model.mlmodelc"
        if target.exists():
            raise SystemExit(f"quantize_int8_proj: {target} already exists")
        try:
            mlmodelc.rename(target)
        except OSError:
            import shutil
            shutil.copytree(str(mlmodelc), str(target))

    print(
        json.dumps(
            {
                "mlmodelc": str(target.resolve()),
                "quant_ms": quant_ms,
                "compile_ms": compile_ms,
                "input_name": INPUT_NAME,
                "output_name": OUTPUT_NAME,
                "scale": scale,
                "max_abs_w": max_abs,
            }
        )
    )


if __name__ == "__main__":
    main()
