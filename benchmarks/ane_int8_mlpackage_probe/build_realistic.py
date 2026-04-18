"""Repeat the probe at a realistic DFlash-4B o_proj shape.

DFlash-4B (Qwen3 0.6B drafter? — numbers cross-referenced from the
planning doc): hidden=3072, so o_proj is [hidden, hidden] =
[3072, 3072]. Conv1x1 lets us reuse the same nanobot pattern.
seq=16 (ctx=16 is the DFlash-drafter target).
"""

import os
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

C_IN = 3072
C_OUT = 3072
SEQ = 16
OUT_DIR = os.environ.get(
    "PROBE_OUT_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)

rng = np.random.default_rng(0)
w_int8 = rng.integers(-127, 127, size=(C_OUT, C_IN, 1, 1), dtype=np.int8)
scale = np.float16(0.01)
zero_point = np.int8(0)


@mb.program(
    input_specs=[mb.TensorSpec(shape=(1, C_IN, 1, SEQ), dtype=types.fp16)],
    opset_version=ct.target.iOS18,
)
def prog(x):
    w = mb.constexpr_affine_dequantize(
        quantized_data=w_int8,
        zero_point=zero_point,
        scale=scale,
        axis=0,
    )
    return mb.conv(x=x, weight=w, strides=[1, 1], pad_type="valid", dilations=[1, 1])


model = ct.convert(
    prog,
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)
pkg = os.path.join(OUT_DIR, "int8_o_proj_4b.mlpackage")
model.save(pkg)
print(f"WROTE {pkg}  ({C_IN}x{C_OUT} int8 = {C_IN*C_OUT} bytes)")
