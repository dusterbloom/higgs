"""Compute plan for realistic DFlash-4B o_proj int8 conv."""

import os

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

OUT_DIR = os.environ.get(
    "PROBE_OUT_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)
plan = MLComputePlan.load_from_path(
    path=os.path.join(OUT_DIR, "int8_o_proj_4b.mlmodelc"),
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
fn = plan.model_structure.program.functions["main"]
for op in fn.block.operations:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    if usage is None:
        print(f"  op={op.operator_name:<32} NONE")
        continue
    pref = type(usage.preferred_compute_device).__name__
    sup = [type(d).__name__ for d in usage.supported_compute_devices]
    est = plan.get_estimated_cost_for_mlprogram_operation(op)
    w = est.weight if est is not None else None
    print(f"  op={op.operator_name:<34} preferred={pref:<32} supported={sup} cost={w}")
