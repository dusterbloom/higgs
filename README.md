# Higgs nightly

A Rust inference server for local MLX models on Apple Silicon and routed remote providers.

## TL;DR

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

## Architecture

```text
HTTP client → API / routing → capacity admission → inference engine
                   ↓                                  ⇅
             remote provider                   model + Metal / MLX
                   └────────────── response ───────────┘
```

Open [the architecture companion](docs/architecture.html) locally to follow source links. Update `docs/architecture.json` and run `python3 scripts/architecture.py`; CI checks source paths and generated-file freshness. Relationships remain explicit and reviewable.

## Contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md). Target `nightly` for fork-specific changes. Run `cargo build --release -p higgs`, `cargo test --release -- --test-threads=1`, and `cargo fmt --all -- --check`. Include hardware, model, settings, and matched evidence for performance changes. Update reference docs and the architecture map when behavior or structure changes; describe limitations as well as results.
