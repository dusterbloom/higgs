#!/usr/bin/env python3
"""Offline oracle for bounded Qwen3-Next indexless-prefill traces."""

import argparse
import json
from pathlib import Path

import numpy as np


def _softmax(values):
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def _validate(q, k, v, query_positions, scale):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("expected q [R,Hq,D] and k/v [Hkv,K,D]")
    if k.shape != v.shape or q.shape[2] != k.shape[2]:
        raise ValueError("incompatible Q/K/V shapes")
    if len(query_positions) != q.shape[0]:
        raise ValueError("one query position is required per sampled row")
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")
    if q.shape[1] % k.shape[0]:
        raise ValueError("query heads must be divisible by KV heads")
    if np.any(query_positions < 0) or np.any(query_positions >= k.shape[1]):
        raise ValueError("query positions must address the captured K/V prefix")


def dense_attention(q, k, v, query_positions, scale):
    """Exact GQA attention for sampled absolute causal query positions."""
    _validate(q, k, v, query_positions, scale)
    rows, hq, _ = q.shape
    hkv = k.shape[0]
    gqa = hq // hkv
    out = np.empty_like(q, dtype=np.float32)
    for row in range(rows):
        visible = int(query_positions[row]) + 1
        for head in range(hq):
            kv_head = head // gqa
            logits = (k[kv_head, :visible] @ q[row, head]) * scale
            out[row, head] = _softmax(logits) @ v[kv_head, :visible]
    return out


def indexless_mean_attention(
    q, k, v, query_positions, scale, alpha, block_size=128, sink_tokens=128,
    local_tokens=512,
):
    """Row-local mean correction with an exact partial causal block."""
    _validate(q, k, v, query_positions, scale)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if block_size <= 0 or sink_tokens < 0 or local_tokens < 0:
        raise ValueError("block and forced-region sizes must be valid")
    rows, hq, _ = q.shape
    hkv = k.shape[0]
    gqa = hq // hkv
    out = np.empty_like(q, dtype=np.float32)
    selected_tokens = visible_tokens = selected_blocks = visible_blocks = 0
    row_head_densities = []

    for row in range(rows):
        visible = int(query_positions[row]) + 1
        if visible <= 0 or visible > k.shape[1]:
            raise ValueError(f"invalid query position {query_positions[row]}")
        # The block containing the causal boundary is always exact and never
        # contributes a pooled score, even when the row ends at its last token.
        complete = (visible - 1) // block_size
        partial_start = complete * block_size
        for head in range(hq):
            row_selected_tokens = 0
            row_visible_tokens = 0
            kv_head = head // gqa
            scores = np.array([
                (k[kv_head, b * block_size:(b + 1) * block_size].mean(axis=0)
                 @ q[row, head]) * scale
                for b in range(complete)
            ], dtype=np.float32)
            threshold = -np.inf if alpha <= 0 or not len(scores) else scores.max() + np.log(alpha)
            logits = []
            values = []
            for block, score in enumerate(scores):
                start = block * block_size
                end = start + block_size
                forced = start < sink_tokens or end > max(0, visible - local_tokens)
                selected = forced or score >= threshold
                visible_blocks += 1
                visible_tokens += block_size
                row_visible_tokens += block_size
                if selected:
                    selected_blocks += 1
                    selected_tokens += block_size
                    row_selected_tokens += block_size
                    logits.extend(((k[kv_head, start:end] @ q[row, head]) * scale).tolist())
                    values.extend(v[kv_head, start:end])
                else:
                    logits.append(float(score + np.log(block_size)))
                    values.append(v[kv_head, start:end].mean(axis=0))
            if partial_start < visible:
                tail = visible - partial_start
                selected_tokens += tail
                visible_tokens += tail
                row_selected_tokens += tail
                row_visible_tokens += tail
                logits.extend(((k[kv_head, partial_start:visible] @ q[row, head]) * scale).tolist())
                values.extend(v[kv_head, partial_start:visible])
            out[row, head] = _softmax(np.asarray(logits, np.float32)) @ np.asarray(values, np.float32)
            row_head_densities.append(row_selected_tokens / row_visible_tokens)

    density = np.asarray(row_head_densities, dtype=np.float64)
    return out, {
        "selected_tokens": selected_tokens,
        "visible_tokens": visible_tokens,
        "selected_blocks": selected_blocks,
        "visible_blocks": visible_blocks,
        "selected_density": selected_tokens / visible_tokens,
        "selected_density_p50": float(np.percentile(density, 50)),
        "selected_density_p95": float(np.percentile(density, 95)),
        "selected_density_p99": float(np.percentile(density, 99)),
        "selected_density_max": float(density.max()),
    }


def evaluate_trace(path, alphas, block_size, sink_tokens, local_tokens):
    from safetensors.numpy import load_file

    trace = load_file(path)
    q = trace["q"].astype(np.float32)
    k = trace["k"].astype(np.float32)
    v = trace["v"].astype(np.float32)
    positions = trace["query_positions"].astype(np.int32)
    scale = float(trace["scale"].reshape(-1)[0])
    exact = dense_attention(q, k, v, positions, scale)
    layer = int(trace["layer"].reshape(-1)[0])
    offset = int(trace["query_offset"].reshape(-1)[0])
    dtype_names = {0: "other", 1: "bfloat16", 2: "float16", 3: "float32"}
    source_dtypes = [dtype_names[int(code)] for code in trace["source_dtype_codes"]]
    source_shapes = trace["source_shapes_qkv"].astype(int).reshape(3, 4).tolist()
    dense_control, _ = indexless_mean_attention(
        q, k, v, positions, scale, 0.0, block_size, sink_tokens, local_tokens
    )
    if not np.allclose(dense_control, exact, rtol=1e-6, atol=1e-6):
        error = dense_control - exact
        relative = np.linalg.norm(error.reshape(-1, error.shape[-1]), axis=1) / np.maximum(
            np.linalg.norm(exact.reshape(-1, exact.shape[-1]), axis=1), 1e-12
        )
        raise AssertionError(
            f"mandatory dense alpha=0 parity failed for {path}: max rel-L2={relative.max()}"
        )
    for alpha in alphas:
        if alpha == 0:
            approx, stats = dense_control, indexless_mean_attention(
                q, k, v, positions, scale, 0.0, block_size, sink_tokens, local_tokens
            )[1]
        else:
            approx, stats = indexless_mean_attention(
                q, k, v, positions, scale, alpha, block_size, sink_tokens, local_tokens
            )
        flat_exact = exact.reshape(-1, exact.shape[-1])
        flat_error = (approx - exact).reshape(flat_exact.shape)
        denom = np.maximum(np.linalg.norm(flat_exact, axis=1), 1e-12)
        rel = np.linalg.norm(flat_error, axis=1) / denom
        cosine = np.sum(approx.reshape(flat_exact.shape) * flat_exact, axis=1) / np.maximum(
            np.linalg.norm(approx.reshape(flat_exact.shape), axis=1) * denom, 1e-12
        )
        worst = int(np.argmax(rel))
        yield {
            "trace": str(path), "layer": layer, "query_offset": offset,
            "alpha": alpha, "block_size": block_size,
            "sink_tokens": sink_tokens, "local_tokens": local_tokens,
            "q_shape": list(q.shape), "k_shape": list(k.shape), "v_shape": list(v.shape),
            "source_dtypes_qkv": source_dtypes, "source_shapes_qkv": source_shapes,
            "relative_l2_p50": float(np.percentile(rel, 50)),
            "relative_l2_p95": float(np.percentile(rel, 95)),
            "relative_l2_p99": float(np.percentile(rel, 99)),
            "relative_l2_max": float(rel.max()),
            "cosine_min": float(cosine.min()),
            "worst_sample_row": worst // q.shape[1],
            "worst_query_head": worst % q.shape[1],
            "worst_absolute_position": int(positions[worst // q.shape[1]]),
            **stats,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--alpha", default="0,0.01,0.03,0.1,0.3,0.5")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=128)
    parser.add_argument("--local-tokens", type=int, default=512)
    args = parser.parse_args()
    alphas = [float(value) for value in args.alpha.split(",")]
    paths = []
    for value in args.traces:
        path = Path(value)
        paths.extend(sorted(path.glob("*.safetensors")) if path.is_dir() else [path])
    for path in paths:
        for row in evaluate_trace(path, alphas, args.block_size, args.sink_tokens, args.local_tokens):
            print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
