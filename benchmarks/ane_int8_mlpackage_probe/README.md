# ANE int8 mlpackage probe

Evidentiary probes for **AB5 / AB6 / AB7** (ANE int8 weight path, 2026-04-18).
See `.planning/research/ane-truth/CLAIMS.md` and `.planning/next-session-dflash-int8-weights.md`.

## What each script proves

| Script | Claim | Expected outcome |
|---|---|---|
| `build_int8_mlpackage.py` | AB6 — conv1x1 + `constexpr_affine_dequantize` compiles via `.mlpackage` (toy 64x64) | writes `int8_conv1x1.mlpackage` |
| `build_realistic.py` | AB6 — same path at DFlash-4B o_proj shape (3072x3072) | writes `int8_o_proj_4b.mlpackage` |
| `compute_plan.py` | AB7 — scheduler dispatch on toy | `preferred=MLCPUComputeDevice` (shape too small) |
| `plan_4b.py` | AB7 — scheduler dispatch on realistic shape | `preferred=MLNeuralEngineComputeDevice` |
| `run_and_compute_plan.py` | Parity CPU_AND_NE vs CPU_ONLY (toy) | currently partial — loader quirk on `MLModel` wrapper |

AB5 (the raw-MIL int8 path is dead) is still proven by the existing
`test_int8_conv1x1_nanobot_pattern` ignored test in `higgs-models`, not by these scripts.

## Setup — 3.13 sidecar venv required

Project `.venv` is 3.14, which has broken `libcoremlpython` (CLAIMS.md T1).
Create a sidecar:

```bash
uv venv --python 3.13 benchmarks/ane_int8_mlpackage_probe/.venv
source benchmarks/ane_int8_mlpackage_probe/.venv/bin/activate
pip install coremltools==9.0 numpy
python -c "from coremltools.libcoremlpython import _MLModelProxy; print('proxy OK')"
```

## Running

Scripts read `PROBE_OUT_DIR` env var (defaults to this directory). Each run drops
`.mlpackage` + `.mlmodelc` outputs next to the scripts.

```bash
export PROBE_OUT_DIR=benchmarks/ane_int8_mlpackage_probe
python benchmarks/ane_int8_mlpackage_probe/build_realistic.py
# compile to mlmodelc (macOS):
xcrun coremlcompiler compile \
  "$PROBE_OUT_DIR/int8_o_proj_4b.mlpackage" "$PROBE_OUT_DIR"
python benchmarks/ane_int8_mlpackage_probe/plan_4b.py
# expect: preferred=MLNeuralEngineComputeDevice
```

## Toolchain stamp (anchor for re-verification)

```
macOS:       26.3.1 (25D771280a)
Xcode:       26.0.1 (17A400)
SDK:         26.0
coremlc:     3505.4.1
MIL:         3510.2.1
coremltools: 9.0  (Python 3.13 sidecar)
```

If any move, re-run `build_realistic.py` + `plan_4b.py` before trusting AB6/AB7.

## Not in scope

- `run_and_compute_plan.py` has a loader quirk on the `MLModel` predict path;
  the dispatch preference (AB7) was verified via `MLComputePlan.load_from_path`
  on the compiled `.mlmodelc` directly. Fix the predict path in the bridge
  (ObjC `MLModel predictionFromFeatures:`), not here.
- Latency + effective bandwidth live in the engine benches, not here.
