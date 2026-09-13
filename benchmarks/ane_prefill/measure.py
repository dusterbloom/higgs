"""Measure a caller-supplied isolated Higgs endpoint, retaining all SSE events.

Does not start/stop servers, load credentials, or assume TTFT equals prefill.
Adapted from the local September 7 roofline measurement contract.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def processed_rate(tokens, milliseconds):
    if tokens < 0 or (milliseconds is not None and (not math.isfinite(milliseconds) or milliseconds < 0)):
        raise ValueError('negative count or invalid duration')
    return None if not milliseconds else tokens / (milliseconds / 1000)


def summarize_stream(records, done):
    result = dict(outcome='incomplete', prefill_ms=None, ttft_seconds=None,
                  usage=None, finish=None, answer='', progress=[])
    error = False
    for record in records:
        event = record['event']
        error |= bool(event.get('error'))
        if event.get('usage'):
            result['usage'] = event['usage']
        if event.get('prompt_progress'):
            p = event['prompt_progress']
            result['progress'].append(p)
            if p.get('processed', 0) > 0 and p.get('processed', 0) + p.get('cache', 0) >= p.get('total', float('inf')):
                result['prefill_ms'] = p.get('time_ms')
        for choice in event.get('choices', []):
            delta = choice.get('delta', {})
            content = delta.get('content') or delta.get('reasoning_content')
            if content:
                if result['ttft_seconds'] is None:
                    result['ttft_seconds'] = record.get('seconds')
                result['answer'] += content
            if choice.get('finish_reason'):
                result['finish'] = choice['finish_reason']
    if error:
        result['outcome'] = 'server_error'
    elif done and result['usage'] and result['finish'] in ('stop', 'length'):
        result['outcome'] = 'finished'
    return result


def machine_state():
    vm = subprocess.check_output(['vm_stat'], text=True)
    return {'power': subprocess.check_output(['pmset', '-g', 'batt'], text=True),
            'swapouts': int(re.search(r'^Swapouts:\s+(\d+)', vm, re.M)[1])}


def measure_request(endpoint, payload, output, pid):
    output.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(payload).encode()
    (output / 'request.json').write_bytes(encoded)
    before = machine_state()
    records, samples = [], []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                rss = subprocess.check_output(['ps', '-p', str(pid), '-o', 'rss='], text=True)
                samples.append({'at': time.time(), 'rss_bytes': int(rss) * 1024})
            except (ValueError, subprocess.CalledProcessError):
                pass
            stop.wait(1)
    thread = threading.Thread(target=monitor)
    thread.start()
    start = time.monotonic()
    done, failure = False, None
    try:
        request = urllib.request.Request(endpoint + '/v1/chat/completions', data=encoded,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=900) as response, (output / 'stream.jsonl').open('w') as stream:
            for raw in response:
                if not raw.startswith(b'data:'):
                    continue
                data = raw[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                record = {'seconds': time.monotonic() - start, 'event': json.loads(data)}
                records.append(record)
                stream.write(json.dumps(record) + '\n')
                stream.flush()
    except Exception as exc:
        failure = repr(exc)
    finally:
        elapsed = time.monotonic() - start
        stop.set()
        thread.join()
    after = machine_state()
    result = summarize_stream(records, done)
    result.update(request_seconds=elapsed, request_sha256=hashlib.sha256(encoded).hexdigest(),
                  before=before, after=after, swapouts_delta=after['swapouts'] - before['swapouts'],
                  peak_sampled_rss_bytes=max((s['rss_bytes'] for s in samples), default=None),
                  exception=failure)
    if failure:
        result['outcome'] = 'transport_error'
    (output / 'memory.json').write_text(json.dumps(samples))
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('answer', 'progress')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    result = measure_request(args.endpoint, json.loads(args.request.read_text()), args.out, args.pid)
    raise SystemExit(0 if result['outcome'] == 'finished' else 1)
