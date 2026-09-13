"""Diagnostic MLX Q8 fused-QKVZ marginal cost, not a Higgs forward benchmark.

Mirrors dequant_int8 + DENSE_TARGET(g64,b8) + build_qkvz_permutation.
Uses synthetic inputs and Python MLX; report version separately from Rust MLX.
Run only when the whole-model server has exited.
"""
import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
import numpy as np
from safetensors import safe_open


def run(model, out):
    import mlx.core as mx
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    weights = {}
    hashes = {}
    for projection in ('in_proj_qkv', 'in_proj_z'):
        arrays = []
        for suffix in ('weight_int8', 'weight_scale'):
            name = f'model.language_model.layers.0.linear_attn.{projection}.{suffix}'
            with safe_open(model / index[name], framework='np') as shard:
                v = shard.get_tensor(name)
            hashes[name] = hashlib.sha256(v.tobytes()).hexdigest()
            arrays.append(mx.array(v.astype(np.float32)))
        weights[projection] = arrays[0] * arrays[1][:, None]
    qkv, z = weights['in_proj_qkv'], weights['in_proj_z']
    assert qkv.shape == (8192, 2048) and z.shape == (4096, 2048)
    flat = mx.concatenate([qkv, z], axis=0)
    perm = []
    for head in range(16):
        for offset, width in ((head * 128, 128), (2048 + head * 128, 128),
                              (4096 + head * 256, 256), (8192 + head * 256, 256)):
            perm.extend(range(offset, offset + width))
    fused = flat[mx.array(perm)]
    packed = {name: mx.quantize(w, group_size=64, bits=8) for name, w in
              [('fused_qkvz', fused), ('qkv_only', qkv), ('z_only', z)]}
    mx.eval(packed)
    result = {'mlx_version': importlib.metadata.version('mlx'), 'source': hashes, 'samples': {}}
    rng = np.random.default_rng(20260913)
    for tokens in (1, 32, 1024):
        x = mx.array(rng.normal(0, 0.25, (1, tokens, 2048)).astype(np.float16))
        mx.eval(x)
        timings = {name: [] for name in packed}
        # Reverse alternating order to reveal drift, retain every sample.
        for iteration in range(18):
            order = list(packed) if iteration % 2 else list(reversed(packed))
            for name in order:
                start = time.perf_counter_ns()
                y = mx.quantized_matmul(x, *packed[name], transpose=True, group_size=64, bits=8)
                mx.eval(y)
                if iteration >= 4:
                    timings[name].append((time.perf_counter_ns() - start) / 1e6)
        result['samples'][str(tokens)] = timings
        print(tokens, {name: statistics.median(samples) for name, samples in timings.items()}, flush=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output exists')
    run(args.model, args.out)
