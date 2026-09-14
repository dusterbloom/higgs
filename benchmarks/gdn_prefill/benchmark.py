#!/usr/bin/env python3
"""Exact Qwen3.5 GDN recurrence launch probe; no model weights required."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import time

import numpy as np

DIMS = {"batch": 1, "hk": 16, "hv": 32, "dk": 128, "dv": 128}
VARIANTS = (1, 2, 4, 8)
LENGTHS = (128, 512, 1024)
DEFAULT_CONFIG = "/Users/peppi/.cache/lm-studio/models/EschaLabs/Qwen3.6-35B-A3B-Escha-W2/config.json"
DEFAULT_MIN_FREE_PCT = 30
DEFAULT_MIN_RECLAIMABLE_PAGES = 700_000

# Verbatim arithmetic/body from crates/higgs-models/src/qwen3_next.rs at
# 8c6b5f66ba985b0c823ccca82af6f32d73da5ea4 (GDN_RECURRENCE_METAL_PREAMBLE
# and GATED_DELTA_FORWARD_KERNEL_SOURCE). The generated launch-only variants
# differ solely in their host threadgroup Y dimension.
KERNEL_SOURCE = r"""
#define HIGGS_GDN_GATE(gate, a_value, dt_bias_value, a_log_value) \
  float x = static_cast<float>(a_value) + dt_bias_value; \
  float sp = fmax(x, 0.0f) + log1p(exp(-fabs(x))); \
  float gate = exp(-exp(a_log_value) * sp)
#define HIGGS_GDN_BETA(beta, b_value) \
  float beta = 1.0f / (1.0f + exp(-static_cast<float>(b_value)))
#define HIGGS_GDN_DECAY(state_value, gate) \
  state_value = state_value * gate
#define HIGGS_GDN_UPDATE(state_value, key_value, delta) \
  state_value = state_value + key_value * delta

auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;
auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;
auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;
auto i_state = reinterpret_cast<const device float*>(state_in) + (n * Dv + dv_idx) * Dk;
auto o_state = reinterpret_cast<device float*>(state_out) + (n * Dv + dv_idx) * Dk;
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}
float a_log_val = static_cast<float>(a_log[hv_idx]);
float dt_bias_val = static_cast<float>(dt_bias[hv_idx]);
auto a_ = a + b_idx * T * Hv;
auto b_ = b + b_idx * T * Hv;
for (int t = 0; t < T; ++t) {
  HIGGS_GDN_GATE(g_val, a_[hv_idx], dt_bias_val, a_log_val);
  HIGGS_GDN_BETA(beta_val, b_[hv_idx]);
  {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      HIGGS_GDN_DECAY(state[i], g_val);
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);
    auto delta = (v_[dv_idx] - kv_mem) * beta_val;
    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      HIGGS_GDN_UPDATE(state[i], k_[s_idx], delta);
      out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y[dv_idx] = static_cast<InT>(out);
    }
  }
  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  a_ += Hv;
  b_ += Hv;
}
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = state[i];
}
"""


def render_kernel(threadgroup_y: int, temporal_tile: int = 0) -> str:
    if threadgroup_y not in VARIANTS:
        raise ValueError(f"threadgroup_y must be one of {VARIANTS}")
    if temporal_tile not in (0, 4):
        raise ValueError("temporal_tile must be 0 or 4")
    marker = "for (int t = 0; t < T; ++t) {"
    pragma = "#pragma clang loop unroll_count(4)\n" if temporal_tile == 4 else ""
    source = KERNEL_SOURCE.replace(marker, pragma + marker, 1)
    return f"// threadgroup_y={threadgroup_y} temporal_tile={temporal_tile}\n{source}"


def recurrence_reference(q, k, v, a, b, a_log, dt_bias, state):
    """Sequential float32 oracle matching the Metal recurrence equations."""
    q, k, v, a, b = (np.asarray(x, dtype=np.float32) for x in (q, k, v, a, b))
    state = np.array(state, dtype=np.float32, copy=True)
    batch, steps, hk, _ = q.shape
    hv, dv = v.shape[2:]
    out = np.empty((batch, steps, hv, dv), dtype=np.float32)
    repeat = hv // hk
    for batch_i in range(batch):
        for t in range(steps):
            for hv_i in range(hv):
                hk_i = hv_i // repeat
                x = float(a[batch_i, t, hv_i] + dt_bias[hv_i])
                softplus = np.log1p(np.exp(-abs(x))) + max(x, 0.0)
                gate = np.exp(-np.exp(a_log[hv_i]) * softplus)
                beta = 1.0 / (1.0 + np.exp(-b[batch_i, t, hv_i]))
                s = state[batch_i, hv_i]
                s *= gate
                remembered = s @ k[batch_i, t, hk_i]
                delta = (v[batch_i, t, hv_i] - remembered) * beta
                s += delta[:, None] * k[batch_i, t, hk_i][None, :]
                out[batch_i, t, hv_i] = s @ q[batch_i, t, hk_i]
    return out, state


def run_paired(kernels, launch_kwargs, candidate, repeats, evaluate, clock_ns=time.perf_counter_ns):
    rows = []
    for round_i in range(repeats):
        order = (4, candidate) if round_i % 2 == 0 else (candidate, 4)
        for tg_y in order:
            kwargs = launch_kwargs[tg_y]
            start = clock_ns()
            output = kernels[tg_y](**kwargs)
            evaluate(output)
            rows.append({"round": round_i, "threadgroup_y": tg_y,
                         "actual_threadgroup": kwargs["threadgroup"],
                         "ms": (clock_ns() - start) / 1e6})
    return rows


def memory_preflight(min_free_pct=DEFAULT_MIN_FREE_PCT,
                     min_reclaimable_pages=DEFAULT_MIN_RECLAIMABLE_PAGES):
    """Refuse a run when macOS reports too little unified-memory headroom."""
    try:
        pressure = subprocess.run(
            ["memory_pressure", "-Q"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("memory preflight unavailable; refusing to start benchmark") from exc
    marker = "System-wide memory free percentage:"
    line = next((line for line in pressure.splitlines() if marker in line), None)
    if line is None:
        raise RuntimeError(f"memory preflight returned no free-percentage line: {pressure!r}")
    free_pct = int(line.split(":", 1)[1].strip().rstrip("%"))
    if free_pct < min_free_pct and os.environ.get("HIGGS_BENCH_ALLOW_LOW_MEMORY") != "1":
        raise RuntimeError(
            f"refusing benchmark: only {free_pct}% memory free; need {min_free_pct}% "
            "(set HIGGS_BENCH_ALLOW_LOW_MEMORY=1 only for an intentional override)"
        )
    vm_stat = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page_values = {}
    for line in vm_stat.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        try:
            page_values[label] = int(value.strip().rstrip("."))
        except ValueError:
            pass
    reclaimable_pages = sum(page_values.get(label, 0) for label in
                            ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    if (reclaimable_pages < min_reclaimable_pages
            and os.environ.get("HIGGS_BENCH_ALLOW_LOW_MEMORY") != "1"):
        raise RuntimeError(
            f"refusing benchmark: only {reclaimable_pages} reclaimable VM pages; "
            f"need {min_reclaimable_pages}"
        )
    swap_line = next((line for line in vm_stat.splitlines() if line.startswith("Swapouts:")), "")
    swapouts = int(swap_line.split(":", 1)[1].strip().rstrip(".")) if swap_line else None
    return {"free_pct": free_pct, "vm_stat": vm_stat, "swapouts": swapouts,
            "reclaimable_pages": reclaimable_pages,
            "min_free_pct": min_free_pct, "min_reclaimable_pages": min_reclaimable_pages}


def _run_gpu(args):
    memory = memory_preflight(args.min_free_pct, args.min_reclaimable_pages)
    import mlx.core as mx

    to_numpy = lambda value: np.asarray(value.astype(mx.float32))
    rng = np.random.default_rng(args.seed)
    d = DIMS
    rows = []
    with open(args.config, encoding="utf-8") as stream:
        raw_config = json.load(stream)
    config = raw_config.get("text_config", raw_config)
    observed = {
        "hidden_size": config.get("hidden_size"),
        "hk": config.get("linear_num_key_heads"),
        "hv": config.get("linear_num_value_heads"),
        "dk": config.get("linear_key_head_dim"),
        "dv": config.get("linear_value_head_dim"),
    }
    expected = {"hidden_size": 2048, "hk": d["hk"], "hv": d["hv"], "dk": d["dk"], "dv": d["dv"]}
    if observed != expected:
        raise ValueError(f"checkpoint GDN geometry mismatch: observed={observed}, expected={expected}")
    long_outputs = {}
    long_launch_kwargs = {}
    long_kernels = {}
    long_tiled_kernels = {}
    kernels = {}
    scalar_checks = []
    for steps in args.lengths:
        step_launch_kwargs = {}
        shapes = ((d["batch"], steps, d["hk"], d["dk"]),) * 2
        input_dtype = mx.float32 if args.dtype == "float32" else mx.bfloat16
        q, k = [mx.array(rng.normal(0, .05, s).astype(np.float32)).astype(input_dtype) for s in shapes]
        v = mx.array(rng.normal(0, .05, (1, steps, d["hv"], d["dv"])).astype(np.float32)).astype(input_dtype)
        a = mx.array(rng.normal(0, .1, (1, steps, d["hv"])).astype(np.float32)).astype(input_dtype)
        b = mx.array(rng.normal(0, .1, (1, steps, d["hv"])).astype(np.float32)).astype(input_dtype)
        a_log = mx.zeros((d["hv"],), mx.float32)
        dt_bias = mx.zeros((d["hv"],), mx.float32)
        state = mx.zeros((1, d["hv"], d["dv"], d["dk"]), mx.float32)
        t_scalar = mx.array(steps, mx.int32)
        inputs = [q, k, v, a_log, a, dt_bias, b, state, t_scalar]
        for tg_y in VARIANTS:
            kernel = mx.fast.metal_kernel(
                name=f"higgs_gdn_prefill_tg{tg_y}",
                input_names=["q", "k", "v", "a_log", "a", "dt_bias", "b", "state_in", "T"],
                output_names=["y", "state_out"], source=render_kernel(tg_y),
                ensure_row_contiguous=True,
            )
            kernels[tg_y] = kernel
            tiled_kernel = mx.fast.metal_kernel(
                name=f"higgs_gdn_prefill_t4_tg{tg_y}",
                input_names=["q", "k", "v", "a_log", "a", "dt_bias", "b", "state_in", "T"],
                output_names=["y", "state_out"], source=render_kernel(tg_y, temporal_tile=4),
                ensure_row_contiguous=True,
            )
            kwargs = dict(
                inputs=inputs,
                template=[("InT", q.dtype), ("Dk", d["dk"]), ("Dv", d["dv"]),
                          ("Hk", d["hk"]), ("Hv", d["hv"])],
                grid=(32, d["dv"], d["batch"] * d["hv"]),
                threadgroup=(32, tg_y, 1),
                output_shapes=[(1, steps, d["hv"], d["dv"]),
                               (1, d["hv"], d["dv"], d["dk"])],
                output_dtypes=[q.dtype, mx.float32],
            )
            step_launch_kwargs[tg_y] = kwargs
            for _ in range(args.warmup):
                mx.eval(*kernel(**kwargs))
                mx.eval(*tiled_kernel(**kwargs))
            samples = []
            last = None
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                last = kernel(**kwargs)
                mx.eval(*last)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            rows.append({"T": steps, "threadgroup_y": tg_y,
                         "median_ms": float(np.median(samples)), "samples_ms": samples})
            if steps == 1024:
                long_outputs[tg_y] = last
                long_launch_kwargs[tg_y] = kwargs
                long_kernels[tg_y] = kernel
                long_tiled_kernels[tg_y] = tiled_kernel
        # CPU oracle only on a small prefix; full T1024 NumPy oracle is intentionally optional.
        if args.check:
            check_t = min(steps, args.check_tokens)
            cpu = recurrence_reference(to_numpy(q[:, :check_t]), to_numpy(k[:, :check_t]),
                                       to_numpy(v[:, :check_t]), to_numpy(a[:, :check_t]),
                                       to_numpy(b[:, :check_t]), to_numpy(a_log),
                                       to_numpy(dt_bias), to_numpy(state))
            check_inputs = [q[:, :check_t], k[:, :check_t], v[:, :check_t], a_log,
                            a[:, :check_t], dt_bias, b[:, :check_t], state,
                            mx.array(check_t, mx.int32)]
            # Baseline launch validates both output sequence and final state.
            baseline_kwargs = step_launch_kwargs[4]
            check_kwargs = dict(baseline_kwargs, inputs=check_inputs,
                                output_shapes=[(1, check_t, d["hv"], d["dv"]),
                                               baseline_kwargs["output_shapes"][1]])
            got = kernels[4](**check_kwargs); mx.eval(*got)
            got_y, got_state = to_numpy(got[0]), to_numpy(got[1])
            np.testing.assert_allclose(got_y, cpu[0], rtol=2e-2, atol=2e-2)
            np.testing.assert_allclose(got_state, cpu[1], rtol=2e-4, atol=2e-4)
            scalar_checks.append({"source_T": steps, "checked_T": check_t,
                                  "threadgroup_y": 4,
                                  "actual_threadgroup": check_kwargs["threadgroup"],
                                  "output_max_abs": float(np.max(np.abs(got_y - cpu[0]))),
                                  "state_max_abs": float(np.max(np.abs(got_state - cpu[1])))})
    baseline = long_outputs[4]
    parity = {}
    for tg_y, output in long_outputs.items():
        parity[str(tg_y)] = {
            "output_bitwise_equal_tg4": bool(np.array_equal(to_numpy(output[0]), to_numpy(baseline[0]))),
            "state_bitwise_equal_tg4": bool(np.array_equal(to_numpy(output[1]), to_numpy(baseline[1]))),
        }
    tiled_outputs = {}
    tiled_parity = {}
    for tg_y, kwargs in long_launch_kwargs.items():
        tiled_outputs[tg_y] = long_tiled_kernels[tg_y](**kwargs)
        mx.eval(*tiled_outputs[tg_y])
        tiled_parity[str(tg_y)] = {
            "output_bitwise_equal_tg4": bool(np.array_equal(to_numpy(tiled_outputs[tg_y][0]), to_numpy(baseline[0]))),
            "state_bitwise_equal_tg4": bool(np.array_equal(to_numpy(tiled_outputs[tg_y][1]), to_numpy(baseline[1]))),
        }
    t1024 = [row for row in rows if row["T"] == 1024]
    candidate = min((row for row in t1024 if row["threadgroup_y"] != 4),
                    key=lambda row: row["median_ms"])["threadgroup_y"]
    paired = run_paired(long_kernels, long_launch_kwargs, candidate, args.paired_repeats,
                        lambda output: mx.eval(*output))
    tiled_paired = run_paired(
        {4: long_kernels[4], "tiled4": long_tiled_kernels[4]},
        {4: long_launch_kwargs[4], "tiled4": long_launch_kwargs[4]},
        "tiled4", args.paired_repeats, lambda output: mx.eval(*output))
    tiled_tg1_paired = run_paired(
        {4: long_kernels[4], "tiled1": long_tiled_kernels[1]},
        {4: long_launch_kwargs[4], "tiled1": long_launch_kwargs[1]},
        "tiled1", args.paired_repeats, lambda output: mx.eval(*output))
    print(json.dumps({"mlx_version": importlib.metadata.version("mlx"), "config": os.path.realpath(args.config),
                      "observed_geometry": observed, "dims": DIMS,
                      "input_dtype": args.dtype, "rows": rows,
                      "scalar_reference_checks": scalar_checks,
                      "t1024_bitwise_parity": parity,
                      "t1024_tiled4_bitwise_parity": tiled_parity,
                      "paired_candidate": candidate, "paired_tg4_vs_candidate": paired,
                      "paired_tg4_vs_tiled4": tiled_paired,
                      "paired_tg4_vs_tiled1": tiled_tg1_paired,
                      "memory_preflight": memory}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--paired-repeats", type=int, default=9)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--check", action="store_true")
    p.add_argument("--check-tokens", type=int, default=8)
    p.add_argument("--min-free-pct", type=int, default=DEFAULT_MIN_FREE_PCT)
    p.add_argument("--min-reclaimable-pages", type=int, default=DEFAULT_MIN_RECLAIMABLE_PAGES)
    _run_gpu(p.parse_args())


if __name__ == "__main__":
    main()
