"""Compute-plan introspection — which device each op maps to under CPU_AND_NE."""

import os

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

OUT_DIR = os.environ.get(
    "PROBE_OUT_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)
MLMODELC = os.path.join(OUT_DIR, "int8_conv1x1.mlmodelc")

plan = MLComputePlan.load_from_path(
    path=MLMODELC,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
program = plan.model_structure.program
assert program is not None, "no program in compute plan"

print("=== compute plan (CPU_AND_NE) ===")
fn = program.functions["main"]
for op in fn.block.operations:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    if usage is None:
        print(f"  op={op.operator_name:<32} usage=NONE")
        continue
    pref = type(usage.preferred_compute_device).__name__
    supported = [type(d).__name__ for d in usage.supported_compute_devices]
    est = plan.get_estimated_cost_for_mlprogram_operation(op)
    est_w = est.weight if est is not None else None
    print(
        f"  op={op.operator_name:<32} preferred={pref:<34} "
        f"supported={supported}  cost_weight={est_w}"
    )
