# Configuration

This document collects the full CLI, environment, and config-file reference for Higgs.

## Modes

- **Simple mode**: pass one or more `--model` flags to serve local MLX models without a config file.
- **Gateway mode**: run `higgs init` and edit `~/.config/higgs/config.toml` to combine local models, providers, routes, metrics, and daemon behavior.

## Simple Mode

### CLI flags and environment variables

| CLI Flag | Env Variable | Default | Description |
|---|---|---|---|
| `--model` | `HIGGS_MODELS` | *(required)* | Model path or HF ID (repeatable) |
| `--host` | `HIGGS_HOST` | `127.0.0.1` | Bind address (set `0.0.0.0` to expose on the network; pair with `--api-key`) |
| `--port` | `HIGGS_PORT` | `8000` | Bind port |
| `--max-tokens` | `HIGGS_MAX_TOKENS` | `32768` | Max generation tokens |
| `--api-key` | `HIGGS_API_KEY` | *(none)* | Bearer token for auth |
| `--rate-limit` | `HIGGS_RATE_LIMIT` | `0` | Requests/min per client |
| `--timeout` | `HIGGS_TIMEOUT` | `300` | Request timeout in seconds |
| `--mlx-profile` | `HIGGS_MLX_PROFILE` | `auto` | MLX tuning profile: `auto`, `latency`, `balanced`, or `throughput` |
| `--batch` | -- | `false` | Enable continuous batching |
| `--kv-cache` | -- | `off` | KV cache mode: `off` or `turboquant` |
| `--kv-bits` | -- | `3` | Default TurboQuant KV bit width |
| `--kv-key-bits` | -- | `kv-bits - 1` | Override TurboQuant key bit width |
| `--kv-value-bits` | -- | `kv-bits` | Override TurboQuant value bit width |
| `--kv-no-norm-correction` | -- | `false` | Disable TurboQuant norm correction |
| `--kv-adaptive-dense-layers` | -- | `0` | Keep the last N KV cache layers dense |
| `--kv-seed` | -- | `0` | TurboQuant seed |

`auto` resolves to `balanced` for small and medium models, and `throughput` for large and huge models.

### Additional environment toggles

- `HIGGS_ENABLE_THINKING=0|1` forces Qwen thinking on or off.
- `HIGGS_CHUNKED_PREFILL_THRESHOLD` enables chunked prefill above a token threshold.
- `HIGGS_CHUNKED_PREFILL_CHUNK_SIZE` controls chunk size during chunked prefill.
- `HIGGS_MTP=0|1` overrides the tuning profile's speculative decode choice when conditions allow.
- `HIGGS_MTP_DRAFT_N_MAX` controls the maximum MTP draft tokens per speculative cycle. The default is `2` for huge checkpoints and `1` otherwise, clamped to `1..=8`.
- `HIGGS_MTP_ADAPTIVE_DRAFT=1` lets the decode loop increase or decrease the MTP draft window based on recent verifier acceptance.
- `HIGGS_MTP_PROMPT_LOOKUP=1` enables a hybrid MTP loop that tries verified prompt-lookup drafts when the prompt/history has a repeated suffix, then keeps the MTP cache synchronized for later MTP-head cycles.
- Per request, the OpenAI and Anthropic bodies accept `"speculation": "auto" | "dflash" | "mtp" | "none"` to choose the speculative method for that request. `auto` (default) uses the DFlash drafter when one is loaded — including while streaming — and otherwise the MTP head; `mtp` forces the MTP head even when a drafter is loaded; `none` forces plain autoregressive decode. (DFlash requires `draft_model` and the simple engine; it is unavailable under `batch = true`.)
- `HIGGS_CLEAR_CACHE_AFTER_PREFILL` overrides the selected MLX profile behavior for cache clearing.
- `HIGGS_TURBOQUANT_MIN_TOKENS` overrides the TurboQuant activation threshold. The default is `2048`.
- `HIGGS_EXPERIMENTAL_PAGED_KV=1` enables the experimental paged-KV path.
- Qwen thinking budget is currently fixed at `256` tokens and is not currently configurable.

## Gateway Mode

Run `higgs init` to create `~/.config/higgs/config.toml`:

```toml
[server]
# Bind to loopback by default. To expose on the network, set host = "0.0.0.0"
# and set an api_key.
host = "127.0.0.1"
port = 8000
# max_tokens = 32768
# timeout = 300.0
# max_body_size = 10485760
# api_key = "sk-..."
# rate_limit = 0
# CORS origin allow-list for browser clients. Unset = no CORS headers;
# ["*"] allows any origin.
# cors_origins = ["https://app.example.com"]
# max_image_bytes = 20971520   # per-image decoded byte cap (default 20 MiB); keep below max_body_size
# image_fetch_timeout = 10.0   # remote image URL fetch timeout in seconds
# max_image_dimension = 4096   # long-edge pixel cap before family preprocessing

# --- Local defaults ---
[local]
mlx_profile = "auto"
raise_wired_limit = false

# --- Local models ---
[[models]]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# name = "llama"
# mlx_profile = "throughput"
# batch = false
# draft_model = "/path/to/dflash-drafter"   # enables DFlash speculative decoding (simple engine only)
# prefill_yield_tokens = 512 # 0 or omitted keeps synchronous prefill
# kv_cache = "turboquant"
# kv_bits = 3
# kv_key_bits = 2
# kv_value_bits = 3
# kv_norm_correction = true
# kv_adaptive_dense_layers = 0
# kv_seed = 0
# # Cache-resident multi-turn KV retention limits (bound resident KV memory):
# kv_max_sessions = 2                    # max retained conversations, LRU-evicted (>= 1)
# kv_max_session_tokens = 32768          # drop a conversation's KV past N tokens (0 = unlimited)
# kv_retained_idle_secs = 300            # evict KV idle longer than N seconds (0 = never)
# kv_max_suffix_prefill_tokens = 24576   # maximum exact suffix before degraded bootstrap
# kv_max_retained_bytes = 2147483648     # aggregate retained session KV byte limit
# disable_vision = true # force-disable vision for this model (escape hatch)

# --- Remote providers ---
[provider.anthropic]
url = "https://api.anthropic.com"
format = "anthropic"

[provider.openai]
url = "https://api.openai.com"
format = "openai"

[provider.ollama]
url = "http://localhost:11434"
strip_auth = true

# --- Routes ---
[[routes]]
pattern = "claude-.*"
provider = "anthropic"

[[routes]]
pattern = "gpt-.*"
provider = "openai"

# [[routes]]
# pattern = "my-alias"
# provider = "openai"
# model = "gpt-4o"

# --- Default route ---
[default]
provider = "higgs"

# --- Auto router ---
# [auto_router]
# enabled = true
# model = "llama"
# timeout_ms = 2000

# --- Metrics & dashboard ---
# [retention] controls how long request metrics are kept for the dashboard.
# It is NOT the KV cache retention — that is per-model (kv_retained_idle_secs above).
[retention]
enabled = true
minutes = 60

[logging.metrics]
enabled = true
# path = "~/.config/higgs/logs/metrics.jsonl"
# max_size_mb = 50
# max_files = 5
```

### Profile precedence for local models

Order of precedence:

1. `[[models]].mlx_profile`
2. `--mlx-profile`
3. `HIGGS_MLX_PROFILE`
4. `[local].mlx_profile`
5. built-in default `auto`

## Provider Options

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | string | *(required)* | Base URL of the upstream API |
| `format` | `"openai"` or `"anthropic"` | `"openai"` | API format the provider speaks |
| `api_key` | string | *(none)* | API key to inject into proxied requests |
| `strip_auth` | bool | `false` | Remove the client's `Authorization` header before proxying |
| `stub_count_tokens` | bool | `false` | Return a stub for `/v1/messages/count_tokens` |

## Route Options

| Field | Type | Description |
|---|---|---|
| `pattern` | regex | Match against the `model` field in requests |
| `provider` | string | Provider name to forward to |
| `model` | string | Rewrite the model field before forwarding |
| `name` | string | Human label used by the auto-router |
| `description` | string | Route description used for auto-router classification |

## Routing Behavior

Higgs resolves requests in this order:

1. Auto-router when `model == "auto"` or force mode is enabled
2. Direct local engine lookup by model name
3. Regex pattern routing, first match wins
4. Default route fallback

That means Higgs supports:

- direct local model selection
- pattern routing to local or remote targets
- model alias rewriting before forwarding
- auto-routing with a local classifier model
- a default target when nothing else matches

## Local Model Notes

- `batch=true` is only supported for standard transformer families with true batched decode support: `llama`, `mistral`, `qwen2`, and `qwen3`.
- `batch=true` is only supported for transformer families with true batched decode support: `llama`, `mistral`, `qwen2`, and `qwen3`, plus the vision families `llava-qwen2` and Qwen-VL (`qwen3_5_vl`, `qwen3_vl`, `qwen2_5_vl`).
- `higgs doctor` and server startup now reject unsupported `batch=true` combinations instead of silently degrading.
- **Vision requests**: images arrive as OpenAI `image_url` content parts (`data:` base64 URIs or `http(s)://` URLs) on `/v1/chat/completions`. A per-image decoded-byte cap (`server.max_image_bytes`, default 20 MiB) and an HTTP fetch timeout for remote URLs (`server.image_fetch_timeout`, default 10 s) apply; `server.max_image_dimension` (default 4096) is intended to cap the long edge before family preprocessing — it is currently validated by `higgs doctor` (must be within `64..=16384`), with enforcement pending. Malformed, oversize, unsupported, or unfetchable images — and images sent to a model without vision — return a strict 400. Anthropic-style image blocks are not yet processed on the local Anthropic endpoint.
- **Multimodal requests never use the prefix or disk cache**: image features are merged into the KV state, so a multimodal prompt would never match a text-only prefix; image requests neither read from nor populate the in-memory prefix cache or the disk prefix cache (`kv_disk_dir`). Image requests also disable MTP speculative decode, since draft logits at image positions are meaningless.
- `[[models]].disable_vision = true` is an escape hatch intended to force-disable vision processing for a model whose vision tower fails to load. Today the flag is parsed and validated by `higgs doctor` — on a checkpoint with no vision capability it is a no-op and the doctor warns — but runtime enforcement is not yet wired, so image requests are still gated solely by the loaded model's vision capability.
- `[local].raise_wired_limit` defaults to `false`. Turn it on only when you explicitly want MLX to raise the process wired-memory limit.
- Source builds on macOS require `mlx.metallib`. Higgs restores it from Cargo build output when possible and fails startup if it still cannot be resolved.
- The `session_id` chat-request field opts a conversation into cache-resident multi-turn reuse (prefill only the new turn instead of the whole history). It is a **best-effort latency optimization, not exact replay** — the retained KV is TurboQuant-compressed, so a continued turn's output may differ slightly from a stateless full prefill. Omit `session_id` for bit-identical output; the radix prefix cache on the normal path reuses dense KV exactly. Per-conversation KV is bounded by the `kv_max_sessions` / `kv_max_session_tokens` / `kv_retained_idle_secs` model settings above.

## Shell Integration

Export Higgs as the local OpenAI and Anthropic base URL:

```bash
eval "$(higgs shellenv)"
```

Run one command with those variables set:

```bash
higgs exec -- claude
higgs exec -- aider --model openai/gpt-4o
```

`higgs exec` verifies that the server is reachable, sets `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL`, then execs the command.
`higgs shellenv` uses the same strict config loading and reachability checks.

## CLI Overview

| Command | Description |
|---|---|
| `higgs serve` | Start the server in the foreground |
| `higgs start` | Start a background daemon from config or profile |
| `higgs stop` | Stop a running daemon (`--force` escalates to `SIGKILL`) |
| `higgs attach` | Open the daemon metrics dashboard |
| `higgs init` | Create the default config file |
| `higgs shellenv` | Print `export` lines for `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL` |
| `higgs exec -- <cmd>` | Set env vars and exec a command |
| `higgs config get <key>` | Read a config value |
| `higgs config set <key> <value>` | Write a config value |
| `higgs config path` | Print the resolved config file path |
| `higgs doctor` | Validate config, model paths, and providers |

### Global flags

| Flag | Description |
|---|---|
| `--config <FILE>` | Path to config file, conflicts with `--profile` |
| `--profile <NAME>` | Named profile, resolves to `config.<NAME>.toml`, conflicts with `--config` |
| `--verbose` | Enable debug logging |

## Migration Notes

- `higgs start` no longer accepts serve-style flags like `--model`, `--port`, or `--batch`.
- `higgs attach` now fails fast unless the daemon is alive, `/health` passes, and metrics logging is enabled.
- `/metrics` is available and `server.max_body_size` is enforced on API routes.
