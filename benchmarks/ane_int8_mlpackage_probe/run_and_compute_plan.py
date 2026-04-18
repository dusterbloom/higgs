"""Load the int8 mlpackage and (1) run on CPU_AND_NE, (2) introspect where ops landed."""

import os

import numpy as np
import coremltools as ct
from coremltools.models import MLModel
from coremltools.models.compute_plan import MLComputePlan
from coremltools.models.compute_device import MLNeuralEngineComputeDevice

OUT_DIR = os.environ.get(
    "PROBE_OUT_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)
PKG = os.path.join(OUT_DIR, "int8_conv1x1.mlpackage")
MLMODELC = os.path.join(OUT_DIR, "int8_conv1x1.mlmodelc")
C_IN = 64
SEQ = 16

print("=== runtime load (CPU_AND_NE) ===")
m_ne = MLModel(MLMODELC, compute_units=ct.ComputeUnit.CPU_AND_NE, is_temp_package=False)
x = np.sin(np.arange(C_IN * SEQ, dtype=np.float32) * 0.01).reshape(1, C_IN, 1, SEQ)
# coremltools' generic predict wants fp16 cast handled or accepts float ndarray for fp16 input
out_ne = m_ne.predict({"x": x.astype(np.float16)})
key = next(iter(out_ne.keys()))
y_ne = out_ne[key]
print(f"predict OK  key={key}  shape={y_ne.shape}  dtype={y_ne.dtype}")
print(f"  sample out: {y_ne.flatten()[:8]}")

print("\n=== runtime load (CPU_ONLY control) ===")
m_cpu = MLModel(MLMODELC, compute_units=ct.ComputeUnit.CPU_ONLY, is_temp_package=False)
out_cpu = m_cpu.predict({"x": x.astype(np.float16)})
y_cpu = out_cpu[key]
diff = np.abs(y_ne.astype(np.float32) - y_cpu.astype(np.float32))
print(
    f"max|ne-cpu|={diff.max():.6e}  mean|ne-cpu|={diff.mean():.6e}  "
    f"y_cpu[:4]={y_cpu.flatten()[:4]}"
)

print("\n=== compute plan (which device ran each op) ===")
plan = MLComputePlan.load_from_path(
    path=PKG,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
program = plan.model_structure.program
if program is None:
    print("ERROR: no program in plan")
    raise SystemExit(1)
fn = program.functions["main"]
for op in fn.block.operations:
    dev_usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    pref = (
        dev_usage.preferred_compute_device.__class__.__name__
        if dev_usage is not None
        else "NONE"
    )
    supported = (
        [d.__class__.__name__ for d in dev_usage.supported_compute_devices]
        if dev_usage is not None
        else []
    )
    print(f"  op={op.operator_name:<32} preferred={pref:<28} supported={supported}")
