"""Export exact dense-FP16 layer-0 z graph variants, never load the full model.

Source tensor formula follows experiment/ane-z-20260907 (4d7db8463, MIT).
Graph variants test the hypothesis in Anemll's KernelDMA gist; no private API.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from shapes import VARIANTS, rewrite, coefficient_bytes


def export_shapes(model, out, tokens):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.language_model.layers.0.linear_attn.in_proj_z.'
    source = {}
    arrays = []
    for suffix in ('weight_int8', 'weight_scale'):
        name = prefix + suffix
        with safe_open(model / index[name], framework='np') as shard:
            value = shard.get_tensor(name)
        source[name] = {'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                        'shape': list(value.shape), 'dtype': str(value.dtype)}
        arrays.append(value)
    quantized, scale = arrays
    assert quantized.shape == (4096, 2048) and quantized.dtype == np.int8
    assert scale.shape == (4096,) and scale.dtype == np.float16
    weight = (quantized.astype(np.float32) * scale.astype(np.float32)[:, None]).astype(np.float16)
    # Synthetic bounded nonzero inputs isolate shape changes. Not a model quality test.
    x = np.random.default_rng(20260913).normal(0, 0.25, (tokens, 2048)).astype(np.float16)
    np.save(out / 'weight.npy', weight)
    np.save(out / 'input.npy', x)
    manifest = {'source': source, 'tokens': tokens, 'coremltools': ct.__version__, 'variants': {}}
    for variant in VARIANTS:
        xp, weights = rewrite(x, weight, variant)
        destination = out / variant
        destination.mkdir()
        np.ascontiguousarray(xp.T).tofile(destination / 'input.fp16')
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, xp.shape[1], 1, tokens), dtype=types.fp16)],
                    opset_version=ct.target.iOS16)
        def program(x):
            return tuple(mb.conv(x=x, weight=np.ascontiguousarray(w[:, :, None, None]),
                                 pad_type='valid', name=f'z{i}') for i, w in enumerate(weights))
        package = destination / 'z.mlpackage'
        ct.convert(program, source='milinternal', convert_to='mlprogram',
                   minimum_deployment_target=ct.target.iOS16, compute_precision=ct.precision.FLOAT16,
                   package_dir=str(package), skip_model_load=True)
        manifest['variants'][variant] = {'input_channels': xp.shape[1],
            'output_channels': [w.shape[0] for w in weights],
            'nominal_coefficient_bytes_per_core': [coefficient_bytes(w.shape[1], w.shape[0]) for w in weights]}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokens', type=int, choices=(32, 1024), required=True)
    args = parser.parse_args()
    export_shapes(args.model, args.out, args.tokens)
