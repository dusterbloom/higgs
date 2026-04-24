#!/usr/bin/env python3
"""
Palettize an `lm_head` weight matrix to 6-bit and compile to a CoreML
`.mlmodelc` bundle suitable for the Apple Neural Engine (ANE).

This script is the public-API half of the `HIGGS_TARGET_ANE_LM_HEAD=1`
path. The dense fp16 lm_head of Qwen3.5-4B ([vocab=248320, hidden=2560])
exceeds the ANE per-kernel microcode budget when shipped through the
private `_ANEInMemoryModel` path that `ane_bridge.m` uses for GDN
projections. Shipstuff's insight (github.com/shipstuff/mlx-ane-sd):
convert → `coremltools.optimize.coreml.palettize_weights` at 6 bits with
group_size=16, save as `.mlpackage`, compile with `coremlcompiler`, and
load through the public `MLModel` API with `cpuAndNeuralEngine`. That
pipeline goes through a different compile backend that accepts the
`constexpr_lut_to_dense` op the private path rejects (confirmed by the
`probe_lut6_constexpr` test in `qwen3_next_ane.rs`).

Called by `crates/higgs-models/src/qwen3_next_ane.rs::compile_proj_lut6`
on first-load cache miss. Subsequent loads skip this script entirely by
hitting the on-disk `.mlmodelc` cache.

Inputs:
  --weights-bin PATH    Row-major f32 LE bytes, shape [vocab, hidden].
                        Produced by `prepare_lm_head_weights` (qwen3_next.rs).
  --vocab INT           Row count (output dim).
  --hidden INT          Column count (input dim).
  --seq-len INT         Compile-time sequence length. Runtime seq must be <=.
  --out-dir PATH        Target directory. `model.mlmodelc/` is written here.

Output (stdout, one line of JSON):
  {"mlmodelc": "<abs-path>/model.mlmodelc",
   "palettize_ms": <int>,
   "compile_ms":   <int>}

Errors go to stderr with non-zero exit. No intermediate files are left
behind on success.
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


# --- MIL input/output names -------------------------------------------------
# These names are part of the Rust/Python ABI — `ane_bridge_mlmodel.m` passes
# them verbatim to `predictionFromFeatures:`. Keep in sync with
# `ane_mlmodel.rs::AneLmHeadLut6Kernel::dispatch`.
INPUT_NAME = "x"
OUTPUT_NAME = "logits"


def _load_weights(path: Path, vocab: int, hidden: int) -> np.ndarray:
    """Memory-map the weight file without copying until torch needs it."""
    expected_bytes = vocab * hidden * 4  # f32 = 4 bytes
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"palettize_lm_head: --weights-bin size {actual_bytes} != "
            f"vocab*hidden*4 = {expected_bytes} "
            f"(vocab={vocab}, hidden={hidden})"
        )
    arr = np.fromfile(str(path), dtype="<f4", count=vocab * hidden)
    return arr.reshape(vocab, hidden)


def _build_and_convert(w_f32: np.ndarray, seq_len: int):
    """Trace a bias-free nn.Linear and convert to a CoreML mlprogram."""
    # Imports are deferred so `--help` and argument parsing don't pay the
    # torch/coremltools startup cost (~3s combined).
    import torch
    import coremltools as ct

    vocab, hidden = w_f32.shape

    # nn.Linear stores weight as [out_features, in_features], matching the
    # shipstuff convention — rows are output channels, columns are inputs.
    linear = torch.nn.Linear(hidden, vocab, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(w_f32.astype(np.float32)))
    linear.eval()

    # Trace with an fp32 example; coremltools handles fp16 I/O at convert time.
    example = torch.zeros((1, seq_len, hidden), dtype=torch.float32)
    traced = torch.jit.trace(linear, example)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name=INPUT_NAME,
                shape=(1, seq_len, hidden),
                dtype=np.float16,
            )
        ],
        outputs=[ct.TensorType(name=OUTPUT_NAME, dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    return mlmodel


def _palettize(mlmodel):
    """6-bit per-grouped-channel kmeans palettization (shipstuff recipe)."""
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )

    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            nbits=6,
            mode="kmeans",
            granularity="per_grouped_channel",
            group_size=16,
        )
    )
    return palettize_weights(mlmodel, config)


def _compile(mlpackage_dir: Path, out_dir: Path) -> Path:
    """Invoke `xcrun coremlcompiler` and return the resulting .mlmodelc path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # `coremlcompiler compile` produces `<out_dir>/<pkg-stem>.mlmodelc`.
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(mlpackage_dir), str(out_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"palettize_lm_head: coremlcompiler failed "
            f"(exit={result.returncode})\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    expected = out_dir / (mlpackage_dir.stem + ".mlmodelc")
    if not expected.is_dir():
        # Fall back to scanning if Apple changes the naming convention.
        candidates = list(out_dir.glob("*.mlmodelc"))
        if len(candidates) != 1:
            raise SystemExit(
                f"palettize_lm_head: expected exactly one .mlmodelc in "
                f"{out_dir}, found {len(candidates)}: {candidates}"
            )
        expected = candidates[0]
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Palettize lm_head to 6-bit CoreML .mlmodelc")
    parser.add_argument("--weights-bin", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=int)
    parser.add_argument("--hidden", required=True, type=int)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.vocab <= 0 or args.hidden <= 0 or args.seq_len <= 0:
        raise SystemExit("palettize_lm_head: vocab/hidden/seq-len must be > 0")
    if not args.weights_bin.is_file():
        raise SystemExit(f"palettize_lm_head: weights-bin not found: {args.weights_bin}")

    w_f32 = _load_weights(args.weights_bin, args.vocab, args.hidden)

    with tempfile.TemporaryDirectory(prefix="higgs-palettize-") as tmp:
        tmp_path = Path(tmp)

        # Convert + palettize in the temp dir so a failure mid-way doesn't
        # leave stale artifacts in the target cache.
        mlmodel = _build_and_convert(w_f32, args.seq_len)

        t0 = time.monotonic()
        palettized = _palettize(mlmodel)
        palettize_ms = int((time.monotonic() - t0) * 1000)

        mlpackage_dir = tmp_path / "model.mlpackage"
        palettized.save(str(mlpackage_dir))

        t1 = time.monotonic()
        mlmodelc = _compile(mlpackage_dir, tmp_path)
        compile_ms = int((time.monotonic() - t1) * 1000)

        # Move the compiled bundle atomically into the caller's target
        # directory. The Rust side holds the parent cache dir and performs
        # a final `fs::rename` from --out-dir into the real cache slot — we
        # just need to stage something stable for it to pick up.
        args.out_dir.mkdir(parents=True, exist_ok=True)
        target = args.out_dir / "model.mlmodelc"
        if target.exists():
            # Caller is expected to hand us an empty directory; refuse
            # silently rather than corrupting anything live.
            raise SystemExit(f"palettize_lm_head: {target} already exists")
        # `shutil.move` would do the cross-device copy dance; Path.rename
        # stays on-disk within the same tmp mount + out-dir mount, but the
        # caller's out-dir is typically under ~/.nanobot which is the same
        # filesystem as /tmp on macOS. Fall back to copytree on EXDEV.
        try:
            mlmodelc.rename(target)
        except OSError:
            import shutil
            shutil.copytree(str(mlmodelc), str(target))

    print(
        json.dumps(
            {
                "mlmodelc": str(target.resolve()),
                "palettize_ms": palettize_ms,
                "compile_ms": compile_ms,
                "input_name": INPUT_NAME,
                "output_name": OUTPUT_NAME,
            }
        )
    )


if __name__ == "__main__":
    main()
