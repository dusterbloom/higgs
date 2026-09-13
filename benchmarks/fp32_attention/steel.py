#!/usr/bin/env python3
"""Bounded FP32/D256 Steel experiment; imports MLX only for explicit GPU runs."""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import re
import subprocess
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MLX = ROOT / 'target/release/build/mlx-sys-60edabe453143de4/out/build/_deps/mlx-src'
VARIANTS = {'q8k16': (8, 16, 1, 1), 'q16k8': (16, 8, 2, 1),
            'qreg32k16': (32, 16, 4, 1), 'qreg64k16': (64, 16, 8, 1)}
PREFIX = 'mlx/backend/metal/kernels/steel/'


def position_mask(positions, key_len):
    positions = np.asarray(positions)
    if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer) or np.any(positions < 0) or np.any(positions >= key_len):
        raise ValueError('positions must be integer absolute key positions within valid K')
    return np.arange(key_len)[None, :] <= positions[:, None]


def reference(q, k, v, positions, scale):
    """Independent float64 oracle, row-wise to bound memory for real captures."""
    mask = position_mask(positions, k.shape[2])
    if q.shape[1] % k.shape[1] or q.shape[2] != len(positions) or k.shape != v.shape:
        raise ValueError('invalid GQA shapes or query positions')
    out = np.empty(q.shape, np.float64)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            kh = h // (q.shape[1] // k.shape[1])
            for row in range(q.shape[2]):
                visible = mask[row]
                scores = k[b, kh, visible].astype(np.float64) @ q[b, h, row].astype(np.float64) * scale
                weights = np.exp(scores - scores.max())
                out[b, h, row] = weights @ v[b, kh, visible].astype(np.float64) / weights.sum()
    return out


def launch_config(variant, shape):
    b, h, q, d = shape
    if d != 256 or min(b, h, q) < 1:
        raise ValueError('only nonempty D256 supported')
    bq, bk, wm, wn = VARIANTS[variant]
    threads = 32 * wm * wn
    return {'BQ': bq, 'BK': bk, 'BD': d, 'WM': wm, 'WN': wn,
            'grid': [((q + bq - 1) // bq) * threads, h, b],
            'threadgroup': [threads, 1, 1],
            'q_cache_fp32_scalars_per_thread': 64 if variant.startswith('qreg') else 0,
            'threadgroup_bytes': 4 * ((0 if variant.startswith('qreg') else bq * (d + 4)) + max((bk + 4) * d, bk * (d + 4)))}


def extract_source(root):
    """Inline the local Apple headers once; retain license and per-input SHA256."""
    root = pathlib.Path(root)
    hashes = {}
    def read(relative):
        data = (root / relative).read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        return data.decode()
    license_text = read('LICENSE')
    seen = set()
    def inline(relative):
        if relative in seen:
            return ''
        seen.add(relative)
        source = read(relative).replace('#pragma once', '')
        return re.sub(r'^#include "([^"]+)"\s*$', lambda m: inline(m[1]), source, flags=re.M)
    header = '/*\n' + license_text + '\n*/\n'
    header += '\n'.join(inline(PREFIX + file) for file in ['attn/loader.h', 'attn/mma.h', 'attn/params.h'])
    original = read(PREFIX + 'attn/kernels/steel_attention.h')
    ops = original[original.index('struct MaxOp'):original.index('// clang-format off')]
    # Precise exp2, retaining the online softmax algorithm and float MMA arithmetic.
    header += '\nusing namespace mlx::steel;\n' + ops.replace('fast::exp2', 'metal::exp2')
    body = original.split(') { // clang-format on', 1)[1].rsplit('}', 1)[0]
    body = body.replace('fast::exp2', 'metal::exp2')
    return header, body, hashes


def register_q_body(body):
    """Replace only Q staging/reloads; keep original MMA/softmax operation order."""
    replacements = [
        ('  threadgroup T Q_smem[BQ * (BD + padQ)];', ''),
        ('  threadgroup T* Qs = Q_smem;', ''),
        ('  QBlockLoader loader_q(\n      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);', ''),
        ('  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;',
         '  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Qcache;\n  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;'),
        ('  // Load Q blocks\n  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {\n    loader_q.load_safe(short2(BD, params->qL_rem));\n  } else {\n    loader_q.load_unsafe();\n  }',
         "  // Cache each lane's complete Q fragments once, with safe final-block rows.\n  const int rows_left = min(BQ, params->qL - int(tid.x) * BQ) - (tm + sm);\n  Qcache.template load_safe<T, 1, 1>(\n      Q + (tm + sm) * params->Q_strides[2] + sn,\n      params->Q_strides[2], short2(BD - sn, rows_left));"),
        ('      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(\n          &Qs[Qs_offset + dd * Qs_tile_stride]);',
         '      Qtile.frag_at(0, 0) = Qcache.frag_at(0, dd);'),
    ]
    for old, new in replacements:
        if body.count(old) != 1:
            raise ValueError('upstream Q staging changed; refusing ambiguous transformation')
        body = body.replace(old, new)
    return body


def make_candidate(mx, variant, qshape, kshape, scale, root, causal):
    launch = launch_config(variant, qshape)
    b, h, ql, d = qshape
    _, kh, kl, _ = kshape
    bq, bk, wm, wn = VARIANTS[variant]
    header, body, hashes = extract_source(root)
    if variant.startswith('qreg'):
        body = register_q_body(body)
    prelude = f'''
    using T = float; using AccumType = float; using MaskType = bool;
    constexpr int BQ = {bq}, BK = {bk}, BD = 256, WM = {wm}, WN = {wn};
    constexpr bool align_Q = {str(ql % bq == 0).lower()};
    constexpr bool align_K = {str(kl % bk == 0).lower()};
    constexpr bool has_mask = {str(not causal).lower()}, do_causal = {str(causal).lower()}, has_sinks = false;
    const device float* sinks = Q;
    AttnParams local_params = {{
        {b}, {h}, 256, {ql}, {kl}, {h // kh}, {scale:.17e}f,
        {(ql+bq-1)//bq}, {(kl+bk-1)//bk}, {ql//bq}, {kl//bk}, {ql%bq}, {kl%bk}, {kl-ql},
        {{{h*ql*d}, {ql*d}, {d}}}, {{{kh*kl*d}, {kl*d}, {d}}},
        {{{kh*kl*d}, {kl*d}, {d}}}, {{{h*ql*d}, {ql*d}, {d}}}
    }};
    const thread AttnParams* params = &local_params;
    AttnMaskParams local_mask = {{{{0, 0, {kl}}}}};
    const thread AttnMaskParams* mask_params = &local_mask;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;
    uint3 tid = threadgroup_position_in_grid;
    uint3 lid = thread_position_in_threadgroup;
    '''
    kernel = mx.fast.metal_kernel(name='higgs_steel_' + variant, input_names=['Q', 'K', 'V', 'mask'],
                                  output_names=['O'], source=prelude + body, header=header,
                                  ensure_row_contiguous=True)
    def run(q, k, v, mask):
        return kernel(inputs=[q, k, v, mask], grid=tuple(launch['grid']), threadgroup=tuple(launch['threadgroup']),
                      output_shapes=[qshape], output_dtypes=[mx.float32])[0]
    return run, launch, hashes


def counters():
    return {name: subprocess.check_output(command, text=True) for name, command in
            [('vm_stat', ['vm_stat']), ('thermal', ['pmset', '-g', 'therm']), ('power', ['pmset', '-g', 'batt']), ('swap', ['sysctl', 'vm.swapusage'])]}


def swapouts(snapshot):
    return int(re.search(r'Swapouts:\s+(\d+)', snapshot['vm_stat'])[1])


def errors(actual, expected):
    delta = np.asarray(actual).astype(np.float64) - expected
    return {'max_abs': float(np.max(np.abs(delta))),
            'max_row_rel_l2': float(np.max(np.linalg.norm(delta, axis=-1) / np.maximum(np.linalg.norm(expected, axis=-1), 1e-12)))}


def paired_schedule(candidates, repeats):
    if repeats < 2 or repeats % 2:
        raise ValueError('timing repeats must be positive and even')
    rng = np.random.default_rng(20260913)
    return [{'pair_id': f'{candidate}:{repeat}', 'candidate': str(candidate), 'repeat': repeat,
             'order': ['dense128', str(candidate)] if repeat % 2 == 0 else [str(candidate), 'dense128']}
            for repeat in range(repeats) for candidate in rng.permutation(candidates)]


def run_case(mx, arrays, positions, scale, args, name, timing=False, view=False):
    case_before = counters()
    qn, kn, vn = arrays
    if any(x.dtype != np.float32 for x in arrays):
        raise ValueError('FP32 inputs required')
    causal = np.array_equal(positions, np.arange(kn.shape[2] - qn.shape[2], kn.shape[2]))
    prepared = [mx.array(x) for x in arrays]
    if view:
        # Actual noncontiguous MLX views, with identical values; conversion is outside timing.
        prepared = [mx.contiguous(x.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3) for x in prepared]
    mx.eval(*prepared)
    start = time.perf_counter_ns()
    contiguous = [mx.contiguous(x) for x in prepared]
    mx.eval(*contiguous)
    materialize_ms = (time.perf_counter_ns() - start) / 1e6
    mask = mx.array(position_mask(positions, kn.shape[2]))
    mx.eval(mask)
    q, k, v = contiguous
    def dense(query, key, value, tiled):
        if not tiled:
            return mx.fast.scaled_dot_product_attention(query, key, value, scale=scale, mask=mask)
        parts = []
        for i in range(0, query.shape[2], 128):
            part = mx.fast.scaled_dot_product_attention(query[:, :, i:i+128], key, value, scale=scale, mask=mask[i:i+128])
            mx.eval(part)
            parts.append(part)
        return mx.concatenate(parts, axis=2)
    arms = {'dense128': lambda: dense(q, k, v, True), 'full_sdpa': lambda: dense(q, k, v, False)}
    launches = {}
    hashes = {}
    for variant in args.variants:
        candidate, launch, hashes = make_candidate(mx, variant, q.shape, k.shape, scale, args.mlx_source, causal)
        launches[variant] = launch
        arms[variant] = lambda f=candidate: f(q, k, v, mask)
        if view:
            def with_copy(f=candidate):
                copied = [mx.contiguous(x) for x in prepared]
                return f(*copied, mask)
            arms[variant + '_with_copy'] = with_copy
    # Repeated fresh materializations count the required view-to-contiguous costs.
    if view:
        arms['materialize_only'] = lambda: [mx.contiguous(x) for x in prepared]
    expected = reference(*arrays, positions, scale) if not timing else None
    checks = {}
    baseline = arms['dense128'](); mx.eval(baseline)
    if expected is None:
        expected = np.asarray(baseline).astype(np.float64)
    for arm, function in arms.items():
        output = function(); mx.eval(output)
        if arm == 'materialize_only':
            continue
        check = errors(output, expected)
        checks[arm] = check
        if not np.isfinite(check['max_abs']) or check['max_abs'] > args.atol or check['max_row_rel_l2'] > args.rtol:
            raise AssertionError(f'{name}/{arm}: {check}')
    samples = []
    before = counters()
    if timing:
        for pair in paired_schedule([arm for arm in arms if arm != 'dense128'], args.repeats):
            for arm in pair['order']:
                start = time.perf_counter_ns()
                output = arms[arm](); mx.eval(output)
                samples.append({**pair, 'arm': arm, 'ms': (time.perf_counter_ns() - start) / 1e6})
    after = counters()
    result = {'name': name, 'q_shape': list(q.shape), 'k_shape': list(k.shape), 'source_view': view,
              'causal_specialization': causal, 'positions': np.asarray(positions).tolist(), 'scale': scale,
              'materialize_once_ms': materialize_ms, 'checks': checks, 'launches': launches,
              'source_sha256': hashes, 'samples': samples, 'case_before': case_before, 'before': before, 'after': after,
              'timing_valid': swapouts(before) == swapouts(after),
              'case_swapouts_delta': swapouts(after) - swapouts(case_before)}
    if samples:
        result['medians_ms'] = {arm: float(np.median([s['ms'] for s in samples if s['arm'] == arm])) for arm in arms}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlx-source', type=pathlib.Path, default=DEFAULT_MLX)
    parser.add_argument('--output', type=pathlib.Path, default=ROOT / 'target/fp32-attention/steel-results.json')
    parser.add_argument('--mode', choices=['prepare', 'correctness', 'timing', 'traces'], default='prepare')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=['q8k16', 'q16k8'])
    parser.add_argument('--key-len', type=int, default=8192)
    parser.add_argument('--query-len', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--trace-dir', type=pathlib.Path, default=ROOT / 'target/prefill-investigation/capture-heterogeneous/traces')
    parser.add_argument('--atol', type=float, default=2e-4)
    parser.add_argument('--rtol', type=float, default=2e-4)
    args = parser.parse_args()
    header, body, hashes = extract_source(args.mlx_source)
    result = {'mode': args.mode, 'source_sha256': hashes, 'transformations': ['recursive header inlining', 'kernel body specialization in FP32', 'fast::exp2 to metal::exp2', 'thread-local constant parameters'],
              'mlx_version': importlib.metadata.version('mlx'), 'variants': args.variants,
              'harness_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), 'cases': []}
    if any(v.startswith('qreg') for v in args.variants):
        result['transformations'].append('qreg only: replace shared Q staging with safe full-Q register cache')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result['before'] = counters() if args.mode != 'prepare' else None
    args.output.write_text(json.dumps(result, indent=2))
    try:
        if args.mode == 'prepare':
            args.output.with_suffix('.metal').write_text(header + '\n' + body)
            for variant in args.variants:
                if variant.startswith('qreg'):
                    args.output.with_name(args.output.stem + '-' + variant + '.metal').write_text(header + '\n' + register_q_body(body))
        else:
            import mlx.core as mx
            result['device'] = str(mx.device_info())
            rng = np.random.default_rng(20260913)
            cases = []
            if args.mode == 'correctness':
                for ql, kl, sampled, amplitude in [(1, 1, False, 1), (7, 19, False, 1), (17, 37, False, 1), (17, 37, True, 1), (9, 19, False, 12), (65, 83, False, 1), (65, 83, True, 1)]:
                    arrays = [rng.normal(0, amplitude, shape).astype(np.float32) for shape in [(1, 4, ql, 256), (1, 2, kl, 256), (1, 2, kl, 256)]]
                    positions = np.sort(rng.choice(kl, ql, replace=False)) if sampled else np.arange(kl-ql, kl)
                    cases.append((arrays, positions, 1/16, f'q{ql}_k{kl}_sampled{sampled}_amp{amplitude}'))
            elif args.mode == 'timing':
                if args.key_len < args.query_len:
                    parser.error('key length must cover queries')
                arrays = [rng.normal(0, 1, shape).astype(np.float32) for shape in [(1, 16, args.query_len, 256), (1, 2, args.key_len, 256), (1, 2, args.key_len, 256)]]
                cases.append((arrays, np.arange(args.key_len-args.query_len, args.key_len), 1/16, f'K{args.key_len}'))
            else:
                from safetensors.numpy import load_file
                for path in sorted(args.trace_dir.glob('*.safetensors')):
                    trace = load_file(path)
                    arrays = [trace['q'].transpose(1, 0, 2)[None].astype(np.float32), trace['k'][None].astype(np.float32), trace['v'][None].astype(np.float32)]
                    cases.append((arrays, trace['query_positions'].astype(np.int32), float(trace['scale'].reshape(-1)[0]), path.name))
                if not cases:
                    raise ValueError('no trace files found')
            for arrays, positions, scale, name in cases:
                for view in [False, True]:
                    result['cases'].append(run_case(mx, arrays, positions, scale, args, name, args.mode == 'timing', view))
                    args.output.write_text(json.dumps(result, indent=2))
                    print(name, 'view=', view, result['cases'][-1]['checks'], flush=True)
                    if result['cases'][-1]['case_swapouts_delta']:
                        raise RuntimeError('new swapouts during case; stopping hardware run')
    except Exception as error:
        result['failure'] = repr(error)
        raise
    finally:
        result['after'] = counters() if args.mode != 'prepare' else None
        args.output.write_text(json.dumps(result, indent=2))
    print(args.output)

if __name__ == '__main__':
    main()
