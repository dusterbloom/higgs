"""Analyze Higgs's leading-cycle eval-barrier profile, not GPU wall-time shares."""
import argparse
import datetime
import json
import re
from pathlib import Path

COMPONENTS = ('gdn_attn_ms', 'gdn_mlp_ms', 'fa_attn_ms', 'fa_mlp_ms')


def parse_profile(logs):
    rows = []
    offset = 0
    for line in logs.splitlines():
        if 'PROFILE: per-layer avg' not in line:
            continue
        fields = dict(re.findall(r'(\w+)="?([0-9.]+)', line))
        required = ('query_tokens', 'gdn_layers', 'fa_layers', *COMPONENTS)
        if any(key not in fields for key in required):
            raise ValueError('incomplete PROFILE row')
        row = {key: float(fields[key]) for key in required}
        row['query_tokens'] = int(row['query_tokens'])
        if row['query_tokens'] <= 1:
            continue
        row['query_offset'] = offset
        offset += row['query_tokens']
        rows.append(row)
    return rows


def summarize_profile(rows):
    if not rows:
        raise ValueError('no prefill profile rows')
    components = {
        key: sum(row[key] * row['gdn_layers' if key.startswith('gdn') else 'fa_layers']
                 for row in rows)
        for key in COMPONENTS
    }
    total = sum(components.values())
    if total <= 0:
        raise ValueError('profile duration must be positive')
    fraction = components['fa_attn_ms'] / total
    return {
        'chunks': len(rows),
        'query_tokens': sum(row['query_tokens'] for row in rows),
        'extrapolated_component_ms': components,
        'component_fractions': {key: value / total for key, value in components.items()},
        'fa_component_fraction': fraction,
        'optimistic_fa_elimination_speedup': 1 / (1 - fraction) if fraction < 1 else None,
        'qualification': 'Leading attention-cycle timings extrapolated across layers; '
                         'eval barriers perturb execution. FA includes projections, cache, '
                         'attention and residual work, so eliminating all FA is more optimistic '
                         'than eliminating SDPA. These are not uninstrumented wall-time shares.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--footprint', type=Path, required=True,
                        help='Request-only timeline: excludes the process warmup')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = json.loads(args.footprint.read_text().splitlines()[0])['at']
    lines = []
    for line in args.log.read_text().splitlines():
        if 'PROFILE: per-layer avg' not in line:
            continue
        timestamp = datetime.datetime.fromisoformat(line.split()[0]).timestamp()
        if timestamp >= start:
            lines.append(line)
    rows = parse_profile('\n'.join(lines))
    summary = summarize_profile(rows)
    total_tokens = summary['query_tokens']
    summary['thirds'] = {
        str(band): summarize_profile([row for row in rows
                                     if min(2, int(3 * row['query_offset'] / total_tokens)) == band])
        for band in range(3)
    }
    summary['rows'] = rows
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}, indent=2))
