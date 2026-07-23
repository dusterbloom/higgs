#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${HIGGS_BIN:-$ROOT/target/debug/higgs}"
TMP_DIR="${TMPDIR:-/tmp}/higgs-smoke.$$"
mkdir -p "$TMP_DIR"

DEFAULT_MODELS=(
  "mlx-community/Llama-3.2-1B-Instruct-4bit"
  "mlx-community/Qwen2.5-3B-Instruct-4bit"
  "mlx-community/Qwen3-1.7B-4bit"
  "mlx-community/Qwen3-Coder-Next-4bit"
)

OPTIONAL_MODELS=(
  "mlx-community/Qwen3.6-35B-A3B-4bit"
)

NANBEIGE_MODEL="${HIGGS_SMOKE_NANBEIGE_MODEL:-MercuriusDream/Nanbeige4.2-3B-mlx-6bit}"
NANBEIGE_EXPECTED_NAME="${HIGGS_SMOKE_NANBEIGE_NAME:-MercuriusDream/Nanbeige4.2-3B-mlx-6bit}"

MODELS=("${DEFAULT_MODELS[@]}")
if [[ "${HIGGS_SMOKE_INCLUDE_OPTIONAL_MODELS:-0}" == "1" ]]; then
  MODELS+=("${OPTIONAL_MODELS[@]}")
fi
if [[ "${HIGGS_SMOKE_INCLUDE_NANBEIGE:-0}" == "1" ]]; then
  MODELS+=("$NANBEIGE_MODEL")
fi

cleanup_pid=""
daemon_config_path=""
cleanup() {
  if [[ -n "${cleanup_pid}" ]] && kill -0 "$cleanup_pid" 2>/dev/null; then
    kill "$cleanup_pid" 2>/dev/null || true
    wait "$cleanup_pid" 2>/dev/null || true
  fi
  if [[ -n "${daemon_config_path}" ]]; then
    "$BIN" --config "$daemon_config_path" stop >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

require_cached_model() {
  local model_id="$1"
  if [[ -d "$model_id" ]]; then
    return 0
  fi
  if [[ "$model_id" == ~/* ]]; then
    local expanded="$HOME/${model_id#~/}"
    if [[ -d "$expanded" ]]; then
      return 0
    fi
  fi

  local cache_root="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-}}"
  if [[ -z "$cache_root" && -n "${HF_HOME:-}" ]]; then
    cache_root="$HF_HOME/hub"
  fi
  cache_root="${cache_root:-$HOME/.cache/huggingface/hub}"
  local cache_dir="$cache_root/models--${model_id//\//--}"
  if [[ ! -d "$cache_dir" ]]; then
    echo "missing cached model: $model_id ($cache_dir)" >&2
    exit 1
  fi
}

wait_for_health() {
  local port="$1"
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "server on port ${port} did not become healthy in time" >&2
  return 1
}

assert_models_contains() {
  local port="$1"
  local name="$2"
  curl -fsS "http://127.0.0.1:${port}/v1/models" | grep -F "\"id\":\"${name}\"" >/dev/null
}

run_chat_checks() {
  local port="$1"
  local model="$2"

  curl -fsS "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(cat <<EOF
{"model":"${model}","messages":[{"role":"user","content":"Answer with one short sentence about Higgs."}],"max_tokens":24}
EOF
)" | grep -F '"choices"' >/dev/null

  curl -fsS "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(cat <<EOF
{"model":"${model}","messages":[{"role":"user","content":"Count to three."}],"max_tokens":24,"stream":true}
EOF
)" | grep -F 'data:' >/dev/null
}

start_server() {
  local log_path="$1"
  shift
  "$BIN" "$@" >"$log_path" 2>&1 &
  cleanup_pid="$!"
}

stop_server() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  cleanup_pid=""
}

single_model_smoke() {
  local model="$1"
  local port="$2"
  local expected_name="${3:-$model}"
  local log_path
  log_path="$TMP_DIR/$(basename "$model").log"

  echo "single-model smoke: $model"
  start_server "$log_path" serve --host 127.0.0.1 --port "$port" --model "$model"
  local pid="$cleanup_pid"
  wait_for_health "$port"
  assert_models_contains "$port" "$expected_name"
  run_chat_checks "$port" "$expected_name"
  stop_server "$pid"
}

multi_model_smoke() {
  local port="$1"
  local config_path="$TMP_DIR/multi-model.toml"
  cat >"$config_path" <<EOF
[server]
host = "127.0.0.1"
port = ${port}

[local]
raise_wired_limit = false

[[models]]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
name = "llama"

[[models]]
path = "mlx-community/Qwen3-1.7B-4bit"
name = "qwen3"

[provider.blackhole]
url = "http://127.0.0.1:9"

[[routes]]
pattern = "llama"
provider = "blackhole"

[logging.metrics]
enabled = true
path = "${TMP_DIR}/multi.metrics.jsonl"
EOF

  echo "multi-model smoke"
  start_server "$TMP_DIR/multi-model.log" --config "$config_path" serve
  local pid="$cleanup_pid"
  wait_for_health "$port"
  assert_models_contains "$port" "llama"
  assert_models_contains "$port" "qwen3"
  run_chat_checks "$port" "llama"
  stop_server "$pid"
}

unsupported_batch_smoke() {
  local config_path="$TMP_DIR/qwen3-next-batch.toml"
  cat >"$config_path" <<EOF
[server]
host = "127.0.0.1"
port = 8998

[[models]]
path = "mlx-community/Qwen3-Coder-Next-4bit"
batch = true
EOF

  echo "batch guard smoke: mlx-community/Qwen3-Coder-Next-4bit"
  if "$BIN" --config "$config_path" doctor >"$TMP_DIR/qwen3-next-batch.out" 2>&1; then
    echo "expected doctor to reject unsupported batch=true for Qwen3-Next" >&2
    exit 1
  fi
  grep -F 'batch=true is only supported for standard transformer models' \
    "$TMP_DIR/qwen3-next-batch.out" >/dev/null
}

daemon_smoke() {
  local port="$1"
  local config_path="$TMP_DIR/daemon.toml"
  cat >"$config_path" <<EOF
[server]
host = "127.0.0.1"
port = ${port}

[local]
raise_wired_limit = false

[[models]]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
name = "llama"

[logging.metrics]
enabled = true
path = "${TMP_DIR}/daemon.metrics.jsonl"
EOF
  daemon_config_path="$config_path"

  echo "daemon/dashboard smoke"
  if "$BIN" --config "$config_path" attach >"$TMP_DIR/attach.fail.out" 2>&1; then
    echo "expected attach to fail before daemon start" >&2
    exit 1
  fi
  grep -F 'no running daemon' "$TMP_DIR/attach.fail.out" >/dev/null

  "$BIN" --config "$config_path" start
  wait_for_health "$port"

  script -q "$TMP_DIR/attach.typescript" "$BIN" --config "$config_path" attach >/dev/null 2>&1 &
  local attach_pid="$!"
  sleep 2
  if ! kill -0 "$attach_pid" 2>/dev/null; then
    echo "attach exited early; see $TMP_DIR/attach.typescript" >&2
    exit 1
  fi
  kill "$attach_pid" 2>/dev/null || true
  wait "$attach_pid" 2>/dev/null || true

  "$BIN" --config "$config_path" stop
  daemon_config_path=""
}

main() {
  require_cmd cargo
  require_cmd curl
  require_cmd grep
  require_cmd script

  if [[ "${HIGGS_SMOKE_INCLUDE_OPTIONAL_MODELS:-0}" == "1" ]]; then
    echo "including optional cached models in smoke matrix"
  fi
  if [[ "${HIGGS_SMOKE_INCLUDE_NANBEIGE:-0}" == "1" ]]; then
    echo "including Nanbeige cached model in smoke matrix"
  fi

  for model in "${MODELS[@]}"; do
    require_cached_model "$model"
  done

  echo "building higgs binary"
  cargo build -p higgs >/dev/null

  single_model_smoke "mlx-community/Llama-3.2-1B-Instruct-4bit" 8101
  single_model_smoke "mlx-community/Qwen2.5-3B-Instruct-4bit" 8102
  single_model_smoke "mlx-community/Qwen3-1.7B-4bit" 8103
  single_model_smoke "mlx-community/Qwen3-Coder-Next-4bit" 8104
  if [[ "${HIGGS_SMOKE_INCLUDE_OPTIONAL_MODELS:-0}" == "1" ]]; then
    single_model_smoke "mlx-community/Qwen3.6-35B-A3B-4bit" 8105
  fi
  if [[ "${HIGGS_SMOKE_INCLUDE_NANBEIGE:-0}" == "1" ]]; then
    single_model_smoke "$NANBEIGE_MODEL" 8106 "$NANBEIGE_EXPECTED_NAME"
  fi
  multi_model_smoke 8110
  unsupported_batch_smoke
  daemon_smoke 8115

  echo "cached-model smoke passed"
}

main "$@"
