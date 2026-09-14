#!/usr/bin/env python3
"""CPU feasibility probe for ReplaySSM-style affine GDN chain verification.

This is deliberately an upper-bound prototype, not a production kernel.  It
keeps the exact serial recurrence as the anchor and composes the equivalent
affine maps with dense matrices.  Dense composition exposes the arithmetic
cost a real low-rank/WY implementation must avoid.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def gates(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.exp(-np.log1p(np.exp(a)))
    beta = 1.0 / (1.0 + np.exp(-b))
    return g.astype(np.float32), beta.astype(np.float32)


def serial_chain(k, v, q, g, beta, state):
    state = np.array(state, dtype=np.float32, copy=True)
    outputs = []
    for t in range(k.shape[0]):
        kt = np.repeat(k[t], state.shape[0] // k.shape[1], axis=0)
        qt = np.repeat(q[t], state.shape[0] // q.shape[1], axis=0)
        delta = (v[t] - np.einsum("vd,vhd->vh", kt, state)) * beta[t, :, None]
        state = state * g[t, :, None, None] + delta[:, :, None] * kt[:, None, :]
        outputs.append(np.einsum("vd,vhd->vh", qt, state))
    return np.stack(outputs), state


def affine_chain(k, v, q, g, beta, state):
    """Compose exact token maps, retaining each prefix for output recovery."""
    d = k.shape[-1]
    identity = np.eye(d, dtype=np.float32)
    transform = np.broadcast_to(identity, (state.shape[0], d, d)).copy()
    offset = np.zeros((state.shape[0], state.shape[1], d), dtype=np.float32)
    outputs = []
    for t in range(k.shape[0]):
        kt = np.repeat(k[t], state.shape[0] // k.shape[1], axis=0)
        qt = np.repeat(q[t], state.shape[0] // q.shape[1], axis=0)
        token_transform = g[t, :, None, None] * (
            identity[None] - beta[t, :, None, None] * kt[:, :, None] * kt[:, None, :]
        )
        token_offset = beta[t, :, None, None] * v[t, :, :, None] * kt[:, None, :]
        transform = np.einsum("hij,hjk->hik", token_transform, transform)
        offset = np.einsum("hji,hvi->hvj", token_transform, offset) + token_offset
        state_t = np.einsum("hvi,hji->hvj", state, transform) + offset
        outputs.append(np.einsum("vd,vhd->vh", qt, state_t))
    final_state = np.einsum("hvi,hji->hvj", state, transform) + offset
    return np.stack(outputs), final_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hv, hk, dk, dv = 32, 16, 128, 128
    rows = []
    for length in args.lengths:
        k = rng.normal(0, 0.05, (length, hk, dk)).astype(np.float32)
        v = rng.normal(0, 0.05, (length, hv, dv)).astype(np.float32)
        q = rng.normal(0, 0.05, (length, hk, dk)).astype(np.float32)
        a = rng.normal(0, 0.1, (length, hv)).astype(np.float32)
        b = rng.normal(0, 0.1, (length, hv)).astype(np.float32)
        g, beta = gates(a, b)
        state = np.zeros((hv, dv, dk), dtype=np.float32)

        serial_out, serial_state = serial_chain(k, v, q, g, beta, state)
        affine_out, affine_state = affine_chain(k, v, q, g, beta, state)
        rows.append({
            "length": length,
            "output_max_abs": float(np.max(np.abs(serial_out - affine_out))),
            "state_max_abs": float(np.max(np.abs(serial_state - affine_state))),
            "output_allclose": bool(np.allclose(serial_out, affine_out, rtol=2e-5, atol=2e-5)),
            "state_allclose": bool(np.allclose(serial_state, affine_state, rtol=2e-5, atol=2e-5)),
        })

        serial_times = []
        affine_times = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            serial_chain(k, v, q, g, beta, state)
            serial_times.append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns()
            affine_chain(k, v, q, g, beta, state)
            affine_times.append((time.perf_counter_ns() - start) / 1e6)
        rows[-1].update({
            "serial_median_ms": float(np.median(serial_times)),
            "affine_median_ms": float(np.median(affine_times)),
            "affine_over_serial": float(np.median(affine_times) / np.median(serial_times)),
        })
    print(json.dumps({"geometry": {"Hk": hk, "Hv": hv, "Dk": dk, "Dv": dv}, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
