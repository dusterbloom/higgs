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


def forward_with_tape(k, v, g, beta, state):
    """Produce the fixed innovations consumed by replay verification."""
    state = np.array(state, dtype=np.float32, copy=True)
    deltas = []
    repeat = state.shape[0] // k.shape[1]
    for t in range(k.shape[0]):
        kt = np.repeat(k[t], repeat, axis=0)
        delta = (v[t] - np.einsum("vd,vhd->vh", kt, state)) * beta[t, :, None]
        state = state * g[t, :, None, None] + delta[:, :, None] * kt[:, None, :]
        deltas.append(delta)
    return np.stack(deltas), state


def replay_serial(k, delta, g, state):
    state = np.array(state, dtype=np.float32, copy=True)
    repeat = state.shape[0] // k.shape[1]
    for t in range(k.shape[0]):
        kt = np.repeat(k[t], repeat, axis=0)
        state = state * g[t, :, None, None] + delta[t, :, :, None] * kt[:, None, :]
    return state


def replay_scan(k, delta, g, state):
    """Prefix-product scan for fixed-innovation replay.

    For each head, p_t=prod(g[:t]); then s_t=p_t*(s0 + cumsum(k_t*delta_t/p_t)).
    """
    state = np.array(state, dtype=np.float32, copy=True)
    hv, dv, _ = state.shape
    repeat = hv // k.shape[1]
    kh = np.repeat(k, repeat, axis=1)
    prefix = np.cumprod(g, axis=0, dtype=np.float32)
    injected = delta[:, :, :, None] * kh[:, :, None, :]
    weighted = injected / prefix[:, :, None, None]
    summed = np.cumsum(weighted, axis=0, dtype=np.float32)
    return (prefix[:, :, None, None] * (state[None] + summed))[-1]


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


def low_rank_chain(k, v, q, g, beta, state):
    """Compose maps without dense matrices; rank grows by one per token."""
    hv, dv, d = state.shape
    factors_u = [np.empty((d, 0), dtype=np.float32) for _ in range(hv)]
    factors_v = [np.empty((d, 0), dtype=np.float32) for _ in range(hv)]
    scale = np.ones(hv, dtype=np.float32)
    offset = np.zeros((hv, dv, d), dtype=np.float32)
    initial = np.array(state, dtype=np.float32, copy=True)
    outputs = []
    repeat = hv // k.shape[1]
    for t in range(k.shape[0]):
        kt = np.repeat(k[t], repeat, axis=0)
        qt = np.repeat(q[t], repeat, axis=0)
        state_t = np.empty_like(initial)
        for h in range(hv):
            uh, vh = factors_u[h], factors_v[h]
            kh = kt[h]
            if uh.shape[1]:
                new_v = -beta[t, h] * (kh + vh @ (uh.T @ kh))
                transformed = initial[h] + (initial[h] @ vh) @ uh.T
            else:
                new_v = -beta[t, h] * kh[:, None]
                transformed = initial[h]
            factors_u[h] = np.column_stack((uh, kh))
            factors_v[h] = np.column_stack((vh, new_v))
            old_dot = offset[h] @ kh
            offset[h] = g[t, h] * (offset[h] - beta[t, h] * old_dot[:, None] * kh)
            offset[h] += beta[t, h] * v[t, h, :, None] * kh
            scale[h] *= g[t, h]
            state_t[h] = scale[h] * transformed + offset[h]
        outputs.append(np.einsum("vd,vhd->vh", qt, state_t))
    final_state = state_t
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
        delta, replay_state = forward_with_tape(k, v, g, beta, state)
        replay_state_serial = replay_serial(k, delta, g, state)
        replay_state_scan = replay_scan(k, delta, g, state)
        affine_out, affine_state = affine_chain(k, v, q, g, beta, state)
        lowrank_out, lowrank_state = low_rank_chain(k, v, q, g, beta, state)
        rows.append({
            "length": length,
            "output_max_abs": float(np.max(np.abs(serial_out - affine_out))),
            "state_max_abs": float(np.max(np.abs(serial_state - affine_state))),
            "output_allclose": bool(np.allclose(serial_out, affine_out, rtol=2e-5, atol=2e-5)),
            "state_allclose": bool(np.allclose(serial_state, affine_state, rtol=2e-5, atol=2e-5)),
            "replay_serial_state_max_abs": float(np.max(np.abs(replay_state - replay_state_serial))),
            "replay_scan_state_max_abs": float(np.max(np.abs(replay_state - replay_state_scan))),
            "replay_scan_state_allclose": bool(np.allclose(replay_state, replay_state_scan, rtol=2e-5, atol=2e-5)),
            "lowrank_output_max_abs": float(np.max(np.abs(serial_out - lowrank_out))),
            "lowrank_state_max_abs": float(np.max(np.abs(serial_state - lowrank_state))),
            "lowrank_output_allclose": bool(np.allclose(serial_out, lowrank_out, rtol=2e-5, atol=2e-5)),
            "lowrank_state_allclose": bool(np.allclose(serial_state, lowrank_state, rtol=2e-5, atol=2e-5)),
        })

        serial_times = []
        affine_times = []
        lowrank_times = []
        replay_serial_times = []
        replay_scan_times = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            serial_chain(k, v, q, g, beta, state)
            serial_times.append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns()
            affine_chain(k, v, q, g, beta, state)
            affine_times.append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns()
            low_rank_chain(k, v, q, g, beta, state)
            lowrank_times.append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns()
            replay_serial(k, delta, g, state)
            replay_serial_times.append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns()
            replay_scan(k, delta, g, state)
            replay_scan_times.append((time.perf_counter_ns() - start) / 1e6)
        rows[-1].update({
            "serial_median_ms": float(np.median(serial_times)),
            "affine_median_ms": float(np.median(affine_times)),
            "affine_over_serial": float(np.median(affine_times) / np.median(serial_times)),
            "lowrank_median_ms": float(np.median(lowrank_times)),
            "lowrank_over_serial": float(np.median(lowrank_times) / np.median(serial_times)),
            "replay_serial_median_ms": float(np.median(replay_serial_times)),
            "replay_scan_median_ms": float(np.median(replay_scan_times)),
            "replay_scan_over_serial": float(np.median(replay_scan_times) / np.median(replay_serial_times)),
        })
    print(json.dumps({"geometry": {"Hk": hk, "Hv": hv, "Dk": dk, "Dv": dv}, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
