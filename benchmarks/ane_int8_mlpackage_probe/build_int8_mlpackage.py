"""Minimal int8 conv1x1 mlpackage probe.

Builds the same conv1x1 + constexpr_affine_dequantize graph as the
raw-MIL kill-test (diffusion_ane.rs:test_int8_conv1x1_nanobot_pattern)
but through coremltools' typed mlprogram path instead of
_ANEDesc modelWithMILText:.

Exit 0 + an .mlpackage written = ct.convert accepted the op.
Actual ANE viability is decided by downstream coremlcompiler +
runtime dispatch.
"""

import os
import sys
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

C_IN = 64
C_OUT = 64
SEQ = 16
OUT_DIR = os.environ.get(
    "PROBE_OUT_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)

# int8 weights, mirroring the Rust test pattern:
# int8_data[i] = (i % 127) - 63
w_int8 = np.array(
    [(i % 127) - 63 for i in range(C_OUT * C_IN)], dtype=np.int8
).reshape(C_OUT, C_IN, 1, 1)
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
    y = mb.conv(x=x, weight=w, strides=[1, 1], pad_type="valid", dilations=[1, 1])
    return y


model = ct.convert(
    prog,
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
)

out_path = os.path.join(OUT_DIR, "int8_conv1x1.mlpackage")
model.save(out_path)
print(f"WROTE {out_path}")
print(f"  weights shape {w_int8.shape}, scale={scale}, zp={zero_point}")
