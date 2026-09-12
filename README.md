# Higgs nightly

Get more useful work from Escha on your MacBook—with inference built around Apple Silicon.

## TL;DR

Our ambition: the best Escha inference engine on this hardware, paired with nanobot for capable local agents. Native kernels, caching, and fixed context limits target speed that survives real sessions. Matched competitor evaluations must establish the claim.

This is the development branch of [dusterbloom/higgs](https://github.com/dusterbloom/higgs), a fork of [panbanda/higgs](https://github.com/panbanda/higgs). Build nightly from source; upstream packages may not include its changes. Higgs serves models and manages inference resources; clients such as nanobot own agent workflows.

With Rust and Xcode command-line tools installed:

```sh
git clone -b nightly https://github.com/dusterbloom/higgs
cd higgs
cargo build --release -p higgs
./target/release/higgs serve --model /path/to/model
```

Use `higgs init` followed by `higgs start` for configured daemon operation. See [configuration](docs/configuration.md), [supported models](docs/models.md), and [benchmarking](docs/benchmarking.md).

## Features

- Local MLX inference and remote-provider routing behind compatible HTTP APIs.
- Native Escha trellis execution and low-bit Metal kernels for supported models.
- Streaming responses and grammar-constrained required or named tool calls.
- Retained sessions, prefix caching, and model-bound disk cache support.
- Capacity admission informed by execution measurements, cache geometry, and memory-pressure evidence.
- Diagnostics for allocator usage, capacity transitions, cache retention, and runtime identity.
- Speculative decoding and vision support on supported model paths.

Native Escha production cache accounting reflects FP32 storage. FP16 attention remains an isolated candidate: long-context exact parity has not passed. Performance claims require matched measurements.

## Disk prefix cache

For the simple engine (`batch = false`), add these settings to a `[[models]]` entry:

```toml
kv_disk_dir = "/var/lib/higgs/prefix-kv"
kv_disk_space_mb = 4096
```

The directory enables the existing disk prefix cache. Files are separated by resolved
model path. The minimum budget is 64 MiB and counts the complete file, including
headers and all appended snapshots. When the next fitting snapshot would exceed
that ceiling, older cached entries are discarded; an individually oversized snapshot
is skipped. Reopening an oversized file also clears its old cache entries. This is
bounded append-log storage, not LRU eviction. Replacing model weights in place requires
clearing its disk cache.

Legacy `disk_cache_enabled = true` and optional `disk_cache_path` still work with
no file-byte ceiling. `kv_disk_dir` cannot be combined with `disk_cache_path`.
Run `higgs doctor` to check settings and directory access.

## Architecture

```text
HTTP client → API / routing → fixed context check → inference engine
                   ↓                                  ⇅
             remote provider                   model + Metal / MLX
                   └────────────── response ───────────┘
```

Open [the architecture companion](docs/architecture.html) locally to follow source links. Update `docs/architecture.json` and run `python3 scripts/architecture.py`; CI checks source paths and generated-file freshness. Relationships remain explicit and reviewable.

## Contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md). Target `nightly` for fork-specific changes. Run `cargo build --release -p higgs`, `cargo test --release -- --test-threads=1`, and `cargo fmt --all -- --check`. Include hardware, model, settings, and matched evidence for performance changes. Update reference docs and the architecture map when behavior or structure changes; describe limitations as well as results.

### Context and cache limits

Each `[[models]]` entry accepts `max_context_tokens = 32768`. This positive limit covers prompt plus requested output and is capped by the model architecture. Explicit output requests that exceed it are rejected before generation. When output length is omitted, the configured output default is capped by the context remaining after the prompt. Memory samples remain diagnostic; they do not shrink the context window, reject model loading, or cancel generation. Actual allocation and generation failures are returned with request cleanup.

Retained KV keeps its configured count, token, byte, and idle limits (2 GiB by default). `kv_cache_bytes = 0` selects a fixed 1 GiB in-memory prefix-cache bound; an explicit positive value overrides it. Disk prefix caching keeps its configured disk budget.
