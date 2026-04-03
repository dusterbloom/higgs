#!/bin/bash
set -euo pipefail

HIGGS_BIN="${HIGGS_BIN:-/Users/peppi/Dev/higgs/target/release/higgs}"
PORT="${PORT:-8097}"
MAX_TOKENS="${MAX_TOKENS:-100}"
MODELS_BASE="/Users/peppi/.cache/lm-studio/models"

# Models to benchmark (path:label)
MODELS=(
  "mlx-community/Qwen3.5-0.8B-8bit:0.8B-8bit"
  "mlx-community/Qwen3.5-4B-MLX-4bit:4B-4bit"
  "mlx-community/Qwen3.5-9B-MLX-4bit:9B-4bit"
  "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit:27B-4bit"
  "NexVeridian/Qwen3.5-35B-A3B-3bit:35B-A3B-3bit"
)

# Prompts: short (~23 tokens), medium (~140 tokens)
SHORT_PROMPT="What is the capital of France? Answer in one sentence."
MEDIUM_PROMPT="Explain the key differences between TCP and UDP protocols. Cover reliability, ordering, connection setup, overhead, and typical use cases. Be thorough but concise, using technical terminology where appropriate. Include examples of applications that use each protocol."

WORKDIR="$(mktemp -d /tmp/higgs-ttft.XXXXXX)"
SERVER_PID=""
RESULTS_FILE="/Users/peppi/Dev/higgs/bench_ttft_results.txt"

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  for _ in $(seq 1 90); do
    if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

measure_ttft() {
  local model_id="$1"
  local prompt="$2"
  local label="$3"

  # Use streaming to measure TTFT (time to first SSE data event)
  python3 - "$model_id" "$prompt" "$MAX_TOKENS" "$PORT" "$label" <<'PY'
import json, sys, time, urllib.request

model_id = sys.argv[1]
prompt = sys.argv[2]
max_tokens = int(sys.argv[3])
port = int(sys.argv[4])
label = sys.argv[5]

payload = json.dumps({
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0.0,
    "stream": True,
}).encode()

req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)

t0 = time.perf_counter()
ttft = None
token_count = 0
last_chunk_time = t0

try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content and ttft is None:
                        ttft = time.perf_counter() - t0
                    if content:
                        token_count += 1
                        last_chunk_time = time.perf_counter()
except Exception as e:
    print(f"  {label}: ERROR - {e}")
    sys.exit(0)

total = last_chunk_time - t0
decode_time = total - (ttft or total)
decode_tps = (token_count - 1) / decode_time if decode_time > 0 and token_count > 1 else 0

print(f"  {label}: TTFT={ttft:.3f}s  decode={decode_tps:.1f} tok/s  tokens={token_count}  total={total:.2f}s")
PY
}

echo "=============================================="
echo "TTFT BENCHMARK — feat/prefill branch"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Max tokens: $MAX_TOKENS"
echo "HIGGS_ENABLE_THINKING: ${HIGGS_ENABLE_THINKING:-unset}"
echo "=============================================="

{
echo "TTFT BENCHMARK — feat/prefill — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Max tokens: $MAX_TOKENS"
echo ""

for entry in "${MODELS[@]}"; do
  model_path="${entry%%:*}"
  model_label="${entry##*:}"
  full_path="${MODELS_BASE}/${model_path}"

  if [ ! -f "${full_path}/config.json" ]; then
    echo "SKIP $model_label — not found at $full_path"
    continue
  fi

  echo ""
  echo "--- $model_label ---"

  # Start server
  "$HIGGS_BIN" serve --model "$full_path" --port "$PORT" >"${WORKDIR}/server.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server; then
    echo "  FAILED to start server"
    tail -5 "${WORKDIR}/server.log" || true
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
    SERVER_PID=""
    continue
  fi

  model_id="$(curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"

  measure_ttft "$model_id" "$SHORT_PROMPT" "short"
  measure_ttft "$model_id" "$MEDIUM_PROMPT" "medium"

  kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
  SERVER_PID=""
done

echo ""
echo "=============================================="
} 2>&1 | tee "$RESULTS_FILE"
