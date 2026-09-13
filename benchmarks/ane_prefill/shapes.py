"""Exact benchmark graph rewrites; coefficients are dense FP16 on 16 cores."""
import numpy as np

VARIANTS = ('original', 'pad_input', 'pad_output', 'split_output')

def rewrite(x, w, variant):
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError('expected token/channel input and output/channel weight')
    if variant == 'original':
        return x, (w,)
    if variant == 'pad_input':
        return np.pad(x, ((0, 0), (0, 32))), (np.pad(w, ((0, 0), (0, 32))),)
    if variant == 'pad_output':
        return x, (np.pad(w, ((0, 64), (0, 0))),)
    if variant == 'split_output' and w.shape[0] % 2 == 0:
        return x, tuple(np.split(w, 2, axis=0))
    raise ValueError(f'unsupported variant or odd output width: {variant}')

def coefficient_bytes(cin, cout, cores=16):
    if min(cin, cout, cores) <= 0 or cout % cores:
        raise ValueError('positive dimensions and integral channels/core required')
    return (cout // cores) * cin * 2
