"""Validate saved native probe outputs and time CPU layout/copy separately."""
import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
from shapes import VARIANTS


def verify(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    tokens = manifest['tokens']
    x, w = np.load(directory / 'input.npy'), np.load(directory / 'weight.npy')
    reference = x.astype(np.float32) @ w.astype(np.float32).T
    report = {}
    original = None
    for variant in VARIANTS:
        base = directory / variant
        native = json.loads((base / 'result.json').read_text())
        pieces = [np.fromfile(base / (o['name'] + '.fp16'), dtype=np.float16)
                  .reshape(o['shape'][1], tokens).T for o in native['outputs']]
        actual = np.concatenate(pieces, axis=1)[:, :4096]
        if original is None:
            original = actual
        # Structural rewrites must agree bitwise on this fixed test fixture.
        np.testing.assert_array_equal(actual, original)
        error = actual.astype(np.float32) - reference
        relative_rmse = float(np.linalg.norm(error) / np.linalg.norm(reference))
        assert np.isfinite(actual).all() and relative_rmse < 0.02
        packed = np.empty((manifest['variants'][variant]['input_channels'], tokens), dtype=np.float16)
        input_backing = np.empty_like(packed)
        merged = np.empty((tokens, 4096), dtype=np.float16)
        phases = {'input_pack_ms': [], 'input_copy_ms': [], 'output_merge_ms': []}
        for iteration in range(16):
            start = time.perf_counter_ns()
            packed[:2048] = x.T
            packed[2048:] = 0
            packed_at = time.perf_counter_ns()
            np.copyto(input_backing, packed)
            copied_at = time.perf_counter_ns()
            offset = 0
            for piece in pieces:
                width = min(piece.shape[1], 4096 - offset)
                merged[:, offset:offset + width] = piece[:, :width]
                offset += width
            end = time.perf_counter_ns()
            if iteration >= 3:
                for key, duration in zip(phases, (packed_at-start, copied_at-packed_at, end-copied_at)):
                    phases[key].append(duration / 1e6)
        np.testing.assert_array_equal(merged, original)
        report[variant] = {'relative_rmse': relative_rmse, 'max_abs': float(np.max(np.abs(error))),
            'bitwise_equal_to_original': True, 'prediction_median_ms': statistics.median(native['prediction_ms']),
            'cpu_boundary_samples': phases, 'cpu_boundary_median_ms': {k: statistics.median(v) for k,v in phases.items()},
            'caveat': 'CPU boundary measured separately on NumPy buffers; no MLX synchronization or overlap measured.'}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
