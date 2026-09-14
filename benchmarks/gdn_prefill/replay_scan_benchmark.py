#!/usr/bin/env python3
"""Guarded Metal benchmark for fixed-innovation GDN replay reconstruction."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from benchmark import DIMS, memory_preflight

SOURCE = r'''
#define GATE(a_value, dt_bias_value, a_log_value) \
  exp(-exp(a_log_value) * (fmax(static_cast<float>(a_value) + dt_bias_value, 0.0f) + \
  log1p(exp(-fabs(static_cast<float>(a_value) + dt_bias_value)))))

auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
auto dk_idx = thread_position_in_grid.x;
auto dv_idx = thread_position_in_grid.y;
auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
auto a_ = a + b_idx * T * Hv + hv_idx;
auto state_ptr = state_in + (n * Dv + dv_idx) * Dk + dk_idx;
auto out_ptr = state_out + (n * Dv + dv_idx) * Dk + dk_idx;
float s = static_cast<float>(*state_ptr);
float prefix = 1.0f;
float weighted = 0.0f;
for (int t = 0; t < T; ++t) {
  float gate = GATE(a_[0], dt_bias[hv_idx], a_log[hv_idx]);
  prefix *= gate;
  weighted += tape_[dv_idx] * k_[dk_idx] / prefix;
  tape_ += Hv * Dv;
  k_ += Hk * Dk;
  a_ += Hv;
}
*out_ptr = prefix * (s + weighted);
'''


def source(scan: bool) -> str:
    # The serial form uses the same per-lane decomposition but applies each
    # recurrence update directly; scan replaces it with prefix products.
    if scan:
        return SOURCE
    return SOURCE.replace(
        "float prefix = 1.0f;\nfloat weighted = 0.0f;",
        "float prefix = 1.0f;\nfloat weighted = 0.0f;",
    ).replace(
        "  prefix *= gate;\n  weighted += tape_[dv_idx] * k_[dk_idx] / prefix;",
        "  s = s * gate + tape_[dv_idx] * k_[dk_idx];",
    ).replace("*out_ptr = prefix * (s + weighted);", "*out_ptr = s;")


def packed_source(scan: bool) -> str:
    """Pack four Dk lanes per thread, matching the production 32x4 layout."""
    body = source(scan)
    body = body.replace("auto dk_idx = thread_position_in_grid.x;",
                        "auto dk_idx = thread_position_in_threadgroup.x;\nconstexpr int n_per_t = Dk / 32;")
    body = body.replace("auto state_ptr = state_in + (n * Dv + dv_idx) * Dk + dk_idx;",
                        "auto state_ptr = state_in + (n * Dv + dv_idx) * Dk;")
    body = body.replace("auto out_ptr = state_out + (n * Dv + dv_idx) * Dk + dk_idx;",
                        "auto out_ptr = state_out + (n * Dv + dv_idx) * Dk;")
    body = body.replace("auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;",
                        "auto tape_base = tape + b_idx * T * Hv * Dv + hv_idx * Dv;")
    body = body.replace("auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;",
                        "auto k_base = k + b_idx * T * Hk * Dk + hk_idx * Dk;")
    body = body.replace("auto a_ = a + b_idx * T * Hv + hv_idx;",
                        "auto a_base = a + b_idx * T * Hv + hv_idx;")
    body = body.replace("float s = static_cast<float>(*state_ptr);",
                        "for (int lane = 0; lane < n_per_t; ++lane) {\n"
                        "int s_idx = n_per_t * dk_idx + lane;\n"
                        "float s = static_cast<float>(state_ptr[s_idx]);\n"
                        "auto tape_ = tape_base; auto k_ = k_base; auto a_ = a_base;")
    body = body.replace("k_[dk_idx]", "k_[s_idx]")
    body = body.replace("*out_ptr = prefix * (s + weighted);", "out_ptr[s_idx] = prefix * (s + weighted);")
    body = body.replace("*out_ptr = s;", "out_ptr[s_idx] = s;")
    return body + "\n}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    memory = memory_preflight()

    import mlx.core as mx

    rng = np.random.default_rng(7)
    d = DIMS
    dtype = mx.float32 if args.dtype == "float32" else mx.bfloat16
    rows = []
    for steps in args.lengths:
        k = mx.array(rng.normal(0, .05, (1, steps, d["hk"], d["dk"])).astype(np.float32)).astype(dtype)
        tape = mx.array(rng.normal(0, .05, (1, steps, d["hv"], d["dv"])).astype(np.float32)).astype(dtype)
        a = mx.array(rng.normal(0, .1, (1, steps, d["hv"])).astype(np.float32)).astype(dtype)
        a_log = mx.zeros((d["hv"],), mx.float32)
        dt_bias = mx.zeros((d["hv"],), mx.float32)
        state = mx.array(rng.normal(0, .01, (1, d["hv"], d["dv"], d["dk"])).astype(np.float32))
        state_in_before = np.asarray(state)
        inputs = [tape, k, a, a_log, dt_bias, state, mx.array(steps, mx.int32)]
        kernels = {}
        kwargs = {}
        for name, is_scan, packed in (("serial", False, False), ("scan", True, False),
                                      ("serial32", False, True), ("scan32", True, True)):
            kernel = mx.fast.metal_kernel(
                name=f"higgs_replay_{name}_t{steps}",
                input_names=["tape", "k", "a", "a_log", "dt_bias", "state_in", "T"],
                output_names=["state_out"], source=packed_source(is_scan) if packed else source(is_scan),
                ensure_row_contiguous=True,
            )
            kernels[name] = kernel
            kwargs[name] = dict(
                inputs=inputs,
                template=[("InT", dtype), ("Dk", d["dk"]), ("Dv", d["dv"]),
                          ("Hk", d["hk"]), ("Hv", d["hv"])],
                grid=(32 if packed else d["dk"], d["dv"], d["hv"]),
                threadgroup=(32, 4, 1) if packed else (d["dk"], 1, 1),
                output_shapes=[(1, d["hv"], d["dv"], d["dk"])],
                output_dtypes=[mx.float32],
            )
            for _ in range(args.warmup):
                mx.eval(*kernel(**kwargs[name]))
        serial = kernels["serial32"](**kwargs["serial32"]); mx.eval(*serial)
        scan = kernels["scan32"](**kwargs["scan32"]); mx.eval(*scan)
        serial_np = np.asarray(serial[0]); scan_np = np.asarray(scan[0])
        serial_times, scan_times, serial32_times, scan32_times = [], [], [], []
        for round_i in range(args.repeats):
            order = (("serial", serial_times), ("scan", scan_times),
                     ("serial32", serial32_times), ("scan32", scan32_times))
            if round_i % 2:
                order = tuple(reversed(order))
            for name, out in order:
                start = time.perf_counter_ns()
                result = kernels[name](**kwargs[name]); mx.eval(*result)
                out.append((time.perf_counter_ns() - start) / 1e6)
        rows.append({"length": steps, "dtype": args.dtype,
                     "state_max_abs": float(np.max(np.abs(serial_np - scan_np))),
                     "state_bitwise_equal": bool(np.array_equal(serial_np, scan_np)),
                     "state_in_unchanged": bool(np.array_equal(np.asarray(state), state_in_before)),
                     "accepted_token_decision_parity": "not_measured_without_lm_head",
                     "serial_median_ms": float(np.median(serial_times)),
                     "scan_median_ms": float(np.median(scan_times)),
                     "scan_over_serial": float(np.median(scan_times) / np.median(serial_times)),
                     "serial32_median_ms": float(np.median(serial32_times)),
                     "scan32_median_ms": float(np.median(scan32_times)),
                     "scan32_over_serial32": float(np.median(scan32_times) / np.median(serial32_times)),
                     "scan32_state_max_abs": float(np.max(np.abs(serial_np - scan_np))),
                     "scan32_bitwise_equal": bool(np.array_equal(serial_np, scan_np))})
    print(json.dumps({"geometry": d, "memory_preflight": memory, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
