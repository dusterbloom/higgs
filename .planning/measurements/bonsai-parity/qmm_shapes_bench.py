"""Isolated quantized_matmul kernel bench for Bonsai-Q1 decode shapes.

Mirror of crates/higgs-models/tests/qmm_shapes_bench.rs. Same shapes,
same protocol (warmup 50, then 1000 iters with mx.eval per call).

Run:
    source ~/Dev/diffusion_bonsai/.venv/bin/activate
    python .planning/measurements/bonsai-parity/qmm_shapes_bench.py
"""
from __future__ import annotations

import time

import mlx.core as mx

SHAPES = [
    # Bonsai-1.7B
    ("1p7b/q_or_o",  2048, 2048),
    ("1p7b/k_or_v",  2048, 1024),
    ("1p7b/gate_up", 2048, 6144),
    ("1p7b/down",    6144, 2048),
    ("1p7b/lm_head", 2048, 151669),
    # Bonsai-8B
    ("8b/q_or_o",    4096, 4096),
    ("8b/k_or_v",    4096, 1024),
    ("8b/gate_up",   4096, 12288),
    ("8b/down",      12288, 4096),
    ("8b/lm_head",   4096, 151669),
]

GROUP_SIZE = 128
BITS = 1
WARMUP = 50
ITERS = 1000


def main() -> None:
    print()
    print(f"qmm-isolated bench (Python mlx-py {mx.__version__}) "
          f"group={GROUP_SIZE} bits={BITS}")
    print(f"warmup={WARMUP} iters={ITERS} (default device)")
    print()
    print(f"{'shape':<18} {'K':>6} {'M':>6} {'ms/iter (sync)':>14}")

    for label, K, M in SHAPES:
        w_full = mx.random.normal(shape=(M, K)).astype(mx.float16)
        qw, scales, biases = mx.quantize(w_full, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(qw, scales, biases)

        x = mx.random.normal(shape=(1, 1, K)).astype(mx.float16)
        mx.eval(x)

        for _ in range(WARMUP):
            y = mx.quantized_matmul(
                x, qw, scales=scales, biases=biases,
                transpose=True, group_size=GROUP_SIZE, bits=BITS,
            )
            mx.eval(y)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            y = mx.quantized_matmul(
                x, qw, scales=scales, biases=biases,
                transpose=True, group_size=GROUP_SIZE, bits=BITS,
            )
            mx.eval(y)
        ms_sync = (time.perf_counter() - t0) * 1e3 / ITERS

        print(f"{label:<18} {K:>6} {M:>6} {ms_sync:>14.4f}")


if __name__ == "__main__":
    main()
