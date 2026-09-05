#!/usr/bin/env python3
"""Build an offline source-linked map; --check detects source or map drift."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import textwrap

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
manifest = root / 'docs/architecture.json'
data = json.loads(manifest.read_text())
fingerprint = hashlib.sha256(manifest.read_bytes())
nodes = {node['id']: node for node in data['nodes']}
assert len(nodes) == len(data['nodes']), 'duplicate node id'
for node in nodes.values():
    path = (root / node['path']).resolve()
    assert path.is_relative_to(root) and path.is_file(), f"Missing source: {node['path']}"
    fingerprint.update(node['path'].encode())
    fingerprint.update(path.read_bytes())
parts = []
for source, target in data['edges']:
    a, b = nodes[source], nodes[target]
    x1, y1 = 30 + a['column'] * 320, 30 + a['row'] * 230
    x2, y2 = 30 + b['column'] * 320, 30 + b['row'] * 230
    if y1 == y2:
        if x1 < x2: x1, x2 = x1 + 280, x2
        else: x1, x2 = x1, x2 + 280
        y1 += 55; y2 += 55
        line = f'M{x1},{y1} L{x2},{y2}'
    else:
        x1 += 140; x2 += 140
        if y1 < y2: y1 += 110
        else: y2 += 110
        mid = (y1 + y2) / 2
        line = f'M{x1},{y1} C{x1},{mid} {x2},{mid} {x2},{y2}'
    parts.append(f'<path d="{line}" class="edge" marker-end="url(#arrow)"/>')
for n in nodes.values():
    x, y = 30 + n['column'] * 320, 30 + n['row'] * 230
    label = html.escape(n['label']); path = html.escape(n['path'], quote=True)
    lines = textwrap.wrap(n['description'], 33)
    assert len(lines) <= 2, 'Keep card descriptions short'
    description = ''.join(f'<tspan x="{x+18}" dy="20">{html.escape(line)}</tspan>' for line in lines)
    parts.append(f'<a href="../{path}" aria-label="{label}: {path}"><title>{path}</title><rect x="{x}" y="{y}" width="280" height="110" rx="12"/><text x="{x+18}" y="{y+31}" class="label">{label}</text><text x="{x+18}" y="{y+42}" class="description">{description}</text><text x="{x+250}" y="{y+30}" class="link">↗</text></a>')
links = ''.join(f'<li><a href="../{html.escape(n["path"], quote=True)}">{html.escape(n["path"])}</a></li>' for n in nodes.values())
page = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · Architecture</title>
<style>
:root{color-scheme:light;--ink:#172c32;--muted:#536a70;--accent:#a14b21}*{box-sizing:border-box}body{margin:0;background:#f4f2eb;color:var(--ink);font:16px/1.6 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:48px 24px}nav{display:flex;justify-content:space-between;font-size:13px;letter-spacing:.08em;text-transform:uppercase}a{color:inherit;text-underline-offset:4px}a:focus-visible{outline:3px solid var(--accent);outline-offset:4px}h1{font-size:clamp(34px,6vw,64px);letter-spacing:-.055em;line-height:1.1;margin:40px 0 18px}p{max-width:720px;color:var(--muted)}.map{overflow:auto;border:1px solid #c9d1cc;border-radius:18px;background:#fffdf7;margin:30px 0}svg{display:block;width:100%;min-width:740px}rect{fill:#fffdf7;stroke:#9daea7;stroke-width:1.4}svg a:hover rect,svg a:focus rect{fill:#f4e7cb;stroke:var(--accent)}.edge{fill:none;stroke:#708980;stroke-width:2}.label{font:600 21px system-ui;fill:var(--ink)}.description{font:14px system-ui;fill:var(--muted)}.link{fill:var(--accent);font:20px system-ui}details{border-top:1px solid #c9d1cc;padding:14px 0}li{font:13px/1.8 ui-monospace,monospace;overflow-wrap:anywhere}footer{font-size:13px;color:var(--muted)}code{font-family:ui-monospace,monospace} @media print{main{padding:0}.map{overflow:visible}svg{min-width:0}details{display:none}}
</style>
<main><nav><a href="../README.md">← README</a><span>Source-linked architecture</span></nav>
<h1>__TITLE__</h1><p>__INTRO__</p>
<div class="map"><svg viewBox="0 0 1000 400" role="group" aria-label="Architecture flow; each component links to source"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#708980"/></marker></defs>__GRAPH__</svg></div>
<p>__NOTE__</p><details><summary>Source files</summary><ul>__LINKS__</ul></details>
<footer>Review relationships in <a href="architecture.json">architecture.json</a>. Regenerate with <code>python3 scripts/architecture.py</code>.<br>CI checks source paths and freshness. Source fingerprint: <code>__HASH__</code>.</footer></main></html>
'''
for key, value in {'TITLE': html.escape(data['title']), 'INTRO': html.escape(data['intro']), 'NOTE': html.escape(data['note']), 'GRAPH': ''.join(parts), 'LINKS': links, 'HASH': fingerprint.hexdigest()[:16]}.items():
    page = page.replace('__' + key + '__', value)
output = root / 'docs/architecture.html'
if args.check:
    if not output.exists() or output.read_text() != page:
        raise SystemExit('Architecture is stale. Review the map, then run python3 scripts/architecture.py')
    print('Architecture paths and generated output are current.')
else:
    output.write_text(page)
    print(output)
