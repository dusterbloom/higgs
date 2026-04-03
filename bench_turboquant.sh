#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <model_dir_or_hf_id>"
  echo "Environment:"
  echo "  HIGGS_BIN      Path to higgs binary (default: /Users/peppi/Dev/higgs/target/release/higgs)"
  echo "  PORT           Port to use (default: 8097)"
  echo "  MAX_TOKENS     Completion tokens to request (default: 256)"
  echo "  PROMPT_REPEATS Prompt repetition count for long-context request (default: 256)"
  echo "  TURBO_BITS     TurboQuant bits (default: 3)"
  echo "  TURBO_SEED     TurboQuant seed (default: 0)"
  exit 1
fi

MODEL_INPUT="$1"
HIGGS_BIN="${HIGGS_BIN:-/Users/peppi/Dev/higgs/target/release/higgs}"
PORT="${PORT:-8097}"
MAX_TOKENS="${MAX_TOKENS:-256}"
PROMPT_REPEATS="${PROMPT_REPEATS:-256}"
TURBO_BITS="${TURBO_BITS:-3}"
TURBO_SEED="${TURBO_SEED:-0}"

for cmd in curl python3 jq; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

if [ ! -x "$HIGGS_BIN" ]; then
  echo "higgs binary not found or not executable: $HIGGS_BIN" >&2
  exit 1
fi

WORKDIR="$(mktemp -d /tmp/higgs-turboquant.XXXXXX)"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

print_kv_estimate() {
  python3 - "$MODEL_INPUT" "$TURBO_BITS" <<'PY'
import json
import math
import os
import sys

model_input = sys.argv[1]
bits = int(sys.argv[2])

config_path = os.path.join(model_input, "config.json")
if not os.path.exists(config_path):
    print("KV estimate: skipped (model path does not have config.json)")
    raise SystemExit(0)

with open(config_path, "r", encoding="utf-8") as fh:
    config = json.load(fh)

model_type = config.get("model_type", "<unknown>")
supported = {"llama", "mistral", "qwen2", "qwen3", "phi3", "starcoder2", "gemma2"}
if model_type not in supported:
    print(f"KV estimate: skipped for model_type={model_type} (hybrid/non-standard path)")
    raise SystemExit(0)

layers = int(config["num_hidden_layers"])
kv_heads = int(config["num_key_value_heads"])
head_dim = int(config.get("head_dim") or (config["hidden_size"] // config["num_attention_heads"]))

dense_bytes_per_token = layers * kv_heads * head_dim * 4
key_bits = bits - 1
key_code_bytes = math.ceil(head_dim * key_bits / 8)
value_code_bytes = math.ceil(head_dim * bits / 8)
sign_bytes = math.ceil(head_dim / 8)
turbo_bytes_per_token = layers * kv_heads * (key_code_bytes + value_code_bytes + sign_bytes + 12)
ratio = turbo_bytes_per_token / dense_bytes_per_token

print(
    "KV estimate: "
    f"model_type={model_type} "
    f"dense_bytes/token={dense_bytes_per_token} "
    f"turbo_bytes/token={turbo_bytes_per_token} "
    f"ratio={ratio:.3f}"
)
PY
}

make_payload() {
  local model_id="$1"
  local payload_path="$2"
  python3 - "$model_id" "$MAX_TOKENS" "$PROMPT_REPEATS" "$payload_path" <<'PY'
import json
import pathlib
import sys

model_id = sys.argv[1]
max_tokens = int(sys.argv[2])
prompt_repeats = int(sys.argv[3])
payload_path = pathlib.Path(sys.argv[4])

prompt = (
    "Summarize the trade-offs between CPU caches, memory bandwidth, and branch prediction. "
    "Keep the answer coherent and grounded in systems details. "
) * prompt_repeats

payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0.0,
}

payload_path.write_text(json.dumps(payload), encoding="utf-8")
PY
}

wait_for_server() {
  local models_path="$1"
  for _ in $(seq 1 60); do
    if curl -s "http://127.0.0.1:${PORT}/v1/models" >"$models_path" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

run_case() {
  local mode="$1"
  local log_path="${WORKDIR}/${mode}.server.log"
  local time_path="${WORKDIR}/${mode}.time.log"
  local models_path="${WORKDIR}/${mode}.models.json"
  local payload_path="${WORKDIR}/${mode}.payload.json"
  local response_path="${WORKDIR}/${mode}.response.json"

  if [ "$mode" = "turboquant" ]; then
    /usr/bin/time -l -o "$time_path" \
      "$HIGGS_BIN" serve --model "$MODEL_INPUT" --port "$PORT" \
      --kv-cache turboquant --kv-bits "$TURBO_BITS" --kv-seed "$TURBO_SEED" \
      >"$log_path" 2>&1 &
  else
    /usr/bin/time -l -o "$time_path" \
      "$HIGGS_BIN" serve --model "$MODEL_INPUT" --port "$PORT" \
      >"$log_path" 2>&1 &
  fi
  SERVER_PID=$!

  if ! wait_for_server "$models_path"; then
    echo "[$mode] server failed to start"
    tail -40 "$log_path" || true
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
    SERVER_PID=""
    return 1
  fi

  local model_id
  model_id="$(jq -r '.data[0].id' "$models_path")"
  make_payload "$model_id" "$payload_path"

  local start_ts end_ts
  start_ts="$(python3 -c 'import time; print(time.time())')"
  if ! curl -s --max-time 600 \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data @"$payload_path" >"$response_path"; then
    echo "[$mode] request failed"
    tail -40 "$log_path" || true
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
    SERVER_PID=""
    return 1
  fi
  end_ts="$(python3 -c 'import time; print(time.time())')"

  kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
  SERVER_PID=""

  python3 - "$mode" "$response_path" "$time_path" "$log_path" "$start_ts" "$end_ts" <<'PY'
import json
import pathlib
import re
import sys

mode = sys.argv[1]
response_path = pathlib.Path(sys.argv[2])
time_path = pathlib.Path(sys.argv[3])
log_path = pathlib.Path(sys.argv[4])
start_ts = float(sys.argv[5])
end_ts = float(sys.argv[6])

response = json.loads(response_path.read_text(encoding="utf-8"))
elapsed = end_ts - start_ts

if "error" in response:
    error = response["error"]
    print(
        f"[{mode}] request_error="
        f"{error.get('type', 'unknown')}:{error.get('message', 'unknown error')}"
    )
    raise SystemExit(1)

usage = response.get("usage", {})
completion_tokens = usage.get("completion_tokens", 0)
finish_reason = response.get("choices", [{}])[0].get("finish_reason")
content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0

rss_kb = None
for line in time_path.read_text(encoding="utf-8").splitlines():
    if "maximum resident set size" in line:
        match = re.search(r"(\d+)", line)
        if match:
            rss_kb = int(match.group(1))
            break

decode_line = ""
for line in log_path.read_text(encoding="utf-8").splitlines():
    if "Decode loop timing" in line:
        decode_line = line.strip()

quality = "OK"
lower = content.lower()
if any(marker in lower for marker in ("meaning meaning", "sense sense", "the the the")):
    quality = "DEGRADED"
elif len(content.strip()) < 20 and finish_reason == "stop":
    quality = "MINIMAL"

print(f"[{mode}] finish={finish_reason} elapsed={elapsed:.2f}s completion_tokens={completion_tokens} tok/s={tok_s:.2f}")
if rss_kb is not None:
    print(f"[{mode}] peak_rss_mb={rss_kb / 1024:.2f}")
print(f"[{mode}] content_chars={len(content)} quality={quality}")
if decode_line:
    print(f"[{mode}] {decode_line}")
PY
}

echo "=============================="
echo "TURBOQUANT BENCH"
echo "Model: $MODEL_INPUT"
echo "Port: $PORT"
echo "Max tokens: $MAX_TOKENS"
echo "Prompt repeats: $PROMPT_REPEATS"
echo "Turbo bits: $TURBO_BITS"
echo "=============================="
print_kv_estimate
echo
run_case off
echo
run_case turboquant
