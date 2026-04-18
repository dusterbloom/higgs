# Structure

Directory layout, key locations, and naming conventions for the Higgs Rust workspace.

## Top-Level Layout

```
higgs/
├── crates/                    # Rust workspace members
│   ├── higgs/                 # Main binary (CLI, server, router, TUI, doctor)
│   ├── higgs-engine/          # Inference engine (prompt cache, batch engine, chat templates)
│   └── higgs-models/          # Model architectures + ANE / DFlash backends
├── benchmarks/                # Standalone perf scripts and logs
├── docs/                      # Design docs and architecture notes
├── scripts/                   # Python probes, validators, POCs
├── memory/                    # Persistent session notes
├── .planning/                 # GSD planning artifacts (this folder)
├── Cargo.toml                 # Workspace root manifest
├── CLAUDE.md                  # Project rules for Claude Code
├── AGENTS.md                  # Agent-specific guidance
└── README.md
```

## Workspace Crates

### `crates/higgs/` — main binary

Gateway layer: HTTP API, CLI, config, TUI, metrics, doctor.

| Path | Purpose |
|------|---------|
| `src/main.rs` | Binary entry point — CLI dispatch |
| `src/lib.rs` | HTTP server assembly (router, state) |
| `src/config.rs` | Config types + parsing |
| `src/cli_config.rs` | CLI flag → config merging |
| `src/daemon.rs` | `higgs init`, daemon lifecycle, config template |
| `src/doctor.rs` | Config validation before server start |
| `src/router.rs` | Request routing logic |
| `src/state.rs` | Shared app state (`RouterState`) |
| `src/proxy.rs` | Upstream proxy handling |
| `src/model_resolver.rs` | Route → model resolution |
| `src/auto_router.rs` | Automatic model selection |
| `src/model_download.rs` | Model fetch from HF / registry |
| `src/attach.rs` | Attach to running daemon |
| `src/translate.rs` | Cross-provider schema translation |
| `src/anthropic_adapter.rs` | Anthropic ↔ internal translation |
| `src/error.rs` | `HiggsError` enum (thiserror) |
| `src/metrics.rs`, `src/metrics_log.rs` | Metrics surface + persistence |
| `src/routes/` | HTTP endpoint handlers |
| `src/routes/chat.rs` | OpenAI `/v1/chat/completions` |
| `src/routes/completions.rs` | OpenAI `/v1/completions` |
| `src/routes/embeddings.rs` | OpenAI embeddings |
| `src/routes/anthropic.rs` | Anthropic `/v1/messages` |
| `src/routes/models.rs`, `health.rs` | Meta routes |
| `src/types/` | API schema types (`openai.rs`, `anthropic.rs`) |
| `src/tui/` | Terminal UI (`views/` subfolder) |
| `tests/integration/` | Integration test modules |

### `crates/higgs-engine/` — inference engine

| Path | Purpose |
|------|---------|
| `src/lib.rs` | Public re-exports |
| `src/engine.rs` | Engine trait + top-level driver |
| `src/batch_engine.rs` | Batched request execution |
| `src/simple.rs` | Simple (non-batched) executor |
| `src/model_loader.rs` | Model file loading + config discovery |
| `src/chat_template.rs` | Jinja chat template renderer (58+ unit tests) |
| `src/prompt_cache.rs` | Prompt-level cache |
| `src/paged_prefix_cache.rs` | Paged prefix cache (vLLM-style) |
| `src/constrained.rs` | Constrained / structured generation |
| `src/reasoning_parser.rs` | Reasoning-model output parsing |
| `src/tool_parser.rs` | Tool-call parsing |
| `src/error.rs` | `EngineError` enum |

### `crates/higgs-models/` — model architectures

Model implementations and acceleration backends (ANE, DFlash).

| Path | Purpose |
|------|---------|
| `src/lib.rs` | Model registry exports |
| `src/registry.rs` | Architecture registration |
| `src/transformer.rs` | Shared transformer primitives |
| `src/cache.rs` | KV cache |
| `src/error.rs` | Model-layer errors |
| `src/utils.rs` | Shared math / tensor helpers |
| **Architectures** | |
| `src/qwen3_next.rs` | Qwen3-Next hybrid SSM / attention / MoE (large file) |
| `src/qwen3_next_ane.rs`, `qwen3_next_ane_worker.rs` | Qwen3-Next ANE offload |
| `src/qwen3_moe.rs` | Qwen3 MoE |
| `src/deepseek_v2.rs` | DeepSeek V2 |
| `src/gemma2.rs` | Gemma 2 |
| `src/phi3.rs` | Phi-3 |
| `src/starcoder2.rs` | Starcoder 2 |
| `src/rwkv7.rs` | RWKV v7 |
| `src/llama_moe.rs` (llada_moe.rs) | LLaDA MoE |
| `src/llava_qwen2.rs`, `siglip.rs` | Multimodal (image encoder) |
| `src/diffusion.rs`, `diffusion_ane*.rs`, `diffusion_lora.rs`, `diffusion_train.rs` | Diffusion pipelines |
| **Backends** | |
| `src/ane_bridge.rs`, `ane_extract.rs`, `ane_forward.rs`, `ane_mil.rs`, `ane_mlmodel.rs` | Apple Neural Engine (CoreML) bridge |
| `src/dflash.rs`, `dflash_ane.rs`, `dflash_cpu.rs` | DFlash speculative decoding (ANE + CPU) |
| `src/turboquant.rs` | Quantization |

## Tests

```
crates/higgs/tests/
├── integration_tests.rs          # Entry point, wires modules
└── integration/
    ├── mod.rs                    # Shared helpers
    ├── cli_exec.rs               # CLI execution tests
    ├── router.rs                 # Router layer
    ├── proxy_e2e.rs              # E2E proxy with wiremock
    ├── request_validation.rs     # Input validation (~50 tests)
    ├── response_contract.rs      # Response shape
    └── error_contract.rs         # Error shape
```

Unit tests live inline under `#[cfg(test)] mod tests` in the source files.

## Ancillary Directories

| Path | What lives here |
|------|-----------------|
| `benchmarks/` | Shell + Python perf scripts, JSON/txt output logs (many timestamped) |
| `docs/` | Design docs — e.g. `ane-prefill-design.md`, `qwen3_next_architecture.md`, `paged-attention-research/` |
| `scripts/` | Python probes, POCs, validators (DFlash, EggRoll, ShadowKV, TurboQuant) |
| `memory/` | Long-running session notes |
| `.planning/` | GSD planning artifacts, phase handoffs, analysis docs |

## Naming Conventions

| Kind | Convention | Example |
|------|------------|---------|
| Source files | `snake_case.rs` | `chat_template.rs`, `model_loader.rs` |
| Modules | `snake_case` | `routes::chat` |
| Types / traits | `PascalCase` | `Engine`, `RouterState`, `ChatTemplate` |
| Functions / vars | `snake_case` | `load_model`, `prefill` |
| Constants | `SCREAMING_SNAKE_CASE` | `DEFAULT_PORT` |
| Test modules | `tests` inside `#[cfg(test)]` | |
| Integration test files | `<concern>.rs` | `proxy_e2e.rs`, `router.rs` |

## Entry Points

- **Binary:** `crates/higgs/src/main.rs` — parses CLI, dispatches to `daemon`, `attach`, `doctor`, etc.
- **Library root:** `crates/higgs/src/lib.rs` — assembles the axum app.
- **HTTP handlers:** `crates/higgs/src/routes/{chat,completions,embeddings,anthropic,models,health}.rs`.
- **Engine trait:** `crates/higgs-engine/src/engine.rs`.
- **Model registry:** `crates/higgs-models/src/registry.rs`.

## Notable Special Locations

- `crates/higgs-models/src/qwen3_next*.rs` — large hybrid SSM/attention/MoE + ANE offload (flagged in CONCERNS).
- `crates/higgs-models/src/dflash*.rs` — speculative decoding (ANE draft + CPU verify).
- `crates/higgs/src/tui/views/` — Ratatui views for the terminal UI.
- `crates/higgs/src/types/` — canonical API schema types, one file per provider.
