#!/usr/bin/env python3
"""Dense/sparse attention feasibility ceiling at captured Qwen3.6 shapes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time

import numpy as np

DENSITIES = ((1, 1), (3, 8), (1, 4), (1, 8))
SCHEDULES = {"row4": 0, "head4": 1}
SOURCE_COMMIT = "8b4cdff3ac22aa54241b02e0f0113a0668d37611"

# Unmodified PREFILL_DENSE_PROBE_SOURCE from crates/higgs-models/src/metal_kernel.rs
# at SOURCE_COMMIT.
PREFILL_DENSE_PROBE_SOURCE = r"""
uint lane = thread_index_in_simdgroup;
uint simd = simdgroup_index_in_threadgroup;
uint3 group = threadgroup_position_in_grid;

int hq_value = Hq;
int hkv_value = Hkv;
int lq_value = Lq;
int capacity_value = Capacity;
int valid_len_value = ValidLen;
int query_offset_value = QueryOffset;
int density_num_value = DensityNum;
int density_den_value = DensityDen;
int pattern_value = Pattern;
int queries_per_kv = hq_value / hkv_value;

int q_head;
int q_row;
bool active;
if (Schedule == 0) {
    q_head = int(group.z);
    q_row = int(group.y) * 4 + int(simd);
    active = q_head < hq_value && q_row < lq_value;
} else {
    int q_head_subgroup = int(group.z) * 4 + int(simd);
    q_head = int(group.x) * queries_per_kv + q_head_subgroup;
    q_row = int(group.y);
    active = int(group.x) < hkv_value && q_head_subgroup < queries_per_kv
        && q_head < hq_value && q_row < lq_value;
}

if (active) {
    int kv_head = q_head / queries_per_kv;
    int visible_end = min(valid_len_value, query_offset_value + q_row + 1);
    int visible_blocks = (visible_end + 127) / 128;
    int selected_blocks = min(
        visible_blocks,
        (visible_blocks * density_num_value + density_den_value - 1) / density_den_value
    );
    float running_max = -INFINITY;
    float denominator = 0.0f;
    float out_fragment[D / 32];
    for (int slot_index = 0; slot_index < D / 32; ++slot_index) {
        out_fragment[slot_index] = 0.0f;
    }

    int q_base = (q_head * lq_value + q_row) * D;
    for (int block = 0; block < visible_blocks; ++block) {
        bool selected;
        if (pattern_value == 0) {
            selected = block >= visible_blocks - selected_blocks;
        } else if (selected_blocks == 1) {
            selected = block == visible_blocks - 1;
        } else {
            int slot = (block * (selected_blocks - 1) + visible_blocks - 2)
                / (visible_blocks - 1);
            selected = slot < selected_blocks
                && block == slot * (visible_blocks - 1) / (selected_blocks - 1);
        }
        selected = selected || block == visible_blocks - 1;
        if (!selected) {
            continue;
        }

        int key_end = min(visible_end, (block + 1) * 128);
        for (int key_row = block * 128; key_row < key_end; ++key_row) {
            int kv_base = (kv_head * capacity_value + key_row) * D;
            float partial = 0.0f;
            for (uint dim = lane; dim < uint(D); dim += 32) {
                partial += float(q[q_base + int(dim)]) * float(k[kv_base + int(dim)]);
            }
            float score = simd_sum(partial) * rsqrt(float(D));
            float next_max = max(running_max, score);
            float old_scale = exp(running_max - next_max);
            float weight = exp(score - next_max);
            denominator = denominator * old_scale + weight;
            for (uint dim = lane; dim < uint(D); dim += 32) {
                int slot_index = int(dim) / 32;
                out_fragment[slot_index] = out_fragment[slot_index] * old_scale
                    + weight * float(v[kv_base + int(dim)]);
            }
            running_max = next_max;
        }
    }

    for (uint dim = lane; dim < uint(D); dim += 32) {
        out[q_base + int(dim)] = T(out_fragment[int(dim) / 32] / denominator);
    }
}
"""


def causal_mask(query_start: int, rows: int, key_len: int, query_offset: int):
    q_positions = query_offset + np.arange(query_start, query_start + rows)
    k_positions = np.arange(key_len)
    return k_positions[None, :] <= q_positions[:, None]


def selected_block_ids(visible_end: int, density: tuple[int, int], pattern: str):
    visible = (visible_end + 127) // 128
    num, den = density
    selected = min(visible, (visible * num + den - 1) // den)
    if pattern == "tail":
        return list(range(visible - selected, visible))
    if selected == 1:
        return [visible - 1]
    blocks = []
    for block in range(visible):
        slot = (block * (selected - 1) + visible - 2) // (visible - 1)
        if slot < selected and block == slot * (visible - 1) // (selected - 1):
            blocks.append(block)
    if visible - 1 not in blocks:
        blocks.append(visible - 1)
    return blocks


def _run(args):
    import mlx.core as mx

    dtype = mx.float32 if args.dtype == "float32" else mx.bfloat16
    rng = np.random.default_rng(args.seed)
    shape = {"K": args.key_len, "Q": 1024, "Hq": 16, "Hkv": 2, "D": 256}
    q = mx.array(rng.normal(0, .05, (1, 16, 1024, 256)).astype(np.float32)).astype(dtype)
    k = mx.array(rng.normal(0, .05, (1, 2, args.key_len, 256)).astype(np.float32)).astype(dtype)
    v = mx.array(rng.normal(0, .05, (1, 2, args.key_len, 256)).astype(np.float32)).astype(dtype)
    mx.eval(q, k, v)
    offset = args.key_len - 1024
    scalar_controls = {
        density: [mx.array(x, mx.int32) for x in
                  (16, 2, 1024, args.key_len, args.key_len, offset,
                   density[0], density[1], 1)]
        for density in DENSITIES
    }
    mx.eval(*(scalar for controls in scalar_controls.values() for scalar in controls))
    full_mask = mx.array(causal_mask(0, 1024, args.key_len, offset))
    mx.eval(full_mask)
    kernel = mx.fast.metal_kernel(
        name="higgs_attention_prefill_ceiling", input_names=["q", "k", "v", "Hq", "Hkv", "Lq",
        "Capacity", "ValidLen", "QueryOffset", "DensityNum", "DensityDen", "Pattern"],
        output_names=["out"], source=PREFILL_DENSE_PROBE_SOURCE)

    def candidate(schedule, density):
        grid = (128, 256, 16) if schedule == "row4" else (256, 1024, 2)
        return kernel(inputs=[q, k, v, *scalar_controls[density]],
                      template=[("T", dtype), ("D", 256), ("Schedule", SCHEDULES[schedule])],
                      grid=grid, threadgroup=(128, 1, 1),
                      output_shapes=[(1, 16, 1024, 256)], output_dtypes=[dtype])[0]

    def dense():
        tiles = []
        for start in range(0, 1024, 128):
            tile = mx.fast.scaled_dot_product_attention(q[:, :, start:start + 128], k, v,
                                                        scale=256 ** -.5,
                                                        mask=full_mask[start:start + 128])
            mx.eval(tile)
            tiles.append(tile)
        out = mx.concatenate(tiles, axis=2); mx.eval(out)
        return out

    baseline = dense()
    checks = {}
    for schedule in SCHEDULES:
        out = candidate(schedule, (1, 1)); mx.eval(out)
        diff = np.asarray((out.astype(mx.float32) - baseline.astype(mx.float32)))
        ref = np.asarray(baseline.astype(mx.float32))
        row_rel = np.linalg.norm(diff, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-12)
        checks[schedule] = {"max_abs": float(np.max(np.abs(diff))), "max_row_rel_l2": float(np.max(row_rel))}
        if checks[schedule]["max_abs"] > 2e-3 or checks[schedule]["max_row_rel_l2"] > 2e-3:
            raise AssertionError(f"density-1 parity failed: {checks}")

    rows = []
    for schedule in SCHEDULES:
        for density in DENSITIES:
            warm_candidate = candidate(schedule, density); mx.eval(warm_candidate)
            dense()
            for repeat in range(args.repeats):
                order = ("dense", "candidate") if repeat % 2 == 0 else ("candidate", "dense")
                for arm in order:
                    start = time.perf_counter_ns()
                    out = dense() if arm == "dense" else candidate(schedule, density)
                    mx.eval(out)
                    rows.append({"schedule": schedule, "density": f"{density[0]}/{density[1]}",
                                 "repeat": repeat, "arm": arm,
                                 "ms": (time.perf_counter_ns() - start) / 1e6})
    print(json.dumps({"mlx_version": importlib.metadata.version("mlx"), "source_commit": SOURCE_COMMIT,
                      "shape": shape, "dtype": args.dtype, "query_offset": offset,
                      "query_tile": 128, "pattern": "interleaved",
                      "input_copy_materialization_ms": 0.0,
                      "input_copy_note": "q/k/v generated row-contiguous; no copy operation is present",
                      "density_1_parity": checks, "rows": rows}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-len", type=int, choices=(8192, 16384), required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
