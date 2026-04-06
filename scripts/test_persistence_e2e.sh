#!/bin/bash
# E2E test: memory persistence across server restarts with real 35B model
set -euo pipefail

MODEL="$HOME/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit"
BINARY="./target/release/higgs"
PORT=9876
MEMORY_DIR="/tmp/higgs-persistence-test"
export HIGGS_CONFIG_DIR="$MEMORY_DIR/config"
CONFIG="$MEMORY_DIR/config/config.toml"

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    rm -rf "$MEMORY_DIR"
}
trap cleanup EXIT

rm -rf "$MEMORY_DIR"
mkdir -p "$MEMORY_DIR/config"

# Write config with memory enabled and short idle timeout
cat > "$CONFIG" <<'EOF'
[[models]]
path = "PLACEHOLDER"

[server]
port = 9876

[memory]
enabled = true
idle_timeout_secs = 10
surprise_threshold = 0.01
EOF
sed -i '' "s|PLACEHOLDER|$MODEL|" "$CONFIG"

export HIGGS_ENABLE_THINKING=0

echo "=== SESSION 1: Start server, generate, idle-train, shut down ==="
$BINARY serve --config "$CONFIG" --verbose 2>"$MEMORY_DIR/log1.txt" &
SERVER_PID=$!

for i in $(seq 1 90); do
    if curl -s http://localhost:$PORT/v1/health >/dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

echo "--- Request 1 ---"
RESP1=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-35B-A3B-3bit",
        "messages": [{"role": "user", "content": "Write a haiku about the ocean at sunset."}],
        "max_tokens": 200,
        "temperature": 0.7
    }')
TEXT1=$(echo "$RESP1" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR: $RESP1")
echo "Response: $TEXT1"

echo "--- Request 2 ---"
RESP2=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-35B-A3B-3bit",
        "messages": [{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
        "max_tokens": 200,
        "temperature": 0.7
    }')
TEXT2=$(echo "$RESP2" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")
echo "Response: $TEXT2"

echo "--- Waiting 20s for idle training (timeout=10s) ---"
sleep 20

echo "--- Shutting down (SIGTERM → save_state) ---"
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true

echo ""
echo "=== PERSISTENCE CHECK ==="
DELTA_FILE=$(find "$MEMORY_DIR" -name "deltas.safetensors" 2>/dev/null | head -1)
REPLAY_FILE=$(find "$MEMORY_DIR" -name "replay.json" 2>/dev/null | head -1)

if [ -n "$DELTA_FILE" ]; then
    DELTA_KB=$(( $(stat -f%z "$DELTA_FILE") / 1024 ))
    echo "deltas.safetensors: ${DELTA_KB} KB"
else
    echo "deltas.safetensors: NOT FOUND"
fi

if [ -n "$REPLAY_FILE" ]; then
    python3 -c "
import json
entries = json.load(open('$REPLAY_FILE'))
print(f'replay.json: {len(entries)} entries')
for e in entries:
    print(f'  {e[\"request_id\"]}: surprise={e[\"surprise\"]:.2f} reward={e[\"reward\"]:.2f} trained={e[\"train_count\"]}x')
"
else
    echo "replay.json: NOT FOUND"
fi

echo ""
echo "=== SESSION 1 FULL LOG (last 30 lines) ==="
tail -30 "$MEMORY_DIR/log1.txt" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g'
echo ""
echo "=== FILES IN MEMORY DIR ==="
find "$MEMORY_DIR" -type f 2>/dev/null || echo "(empty)"

echo ""
echo "=== SESSION 2: Restart, verify load ==="
$BINARY serve --config "$CONFIG" 2>"$MEMORY_DIR/log2.txt" &
SERVER_PID=$!

for i in $(seq 1 90); do
    if curl -s http://localhost:$PORT/v1/health >/dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

echo "--- Same request after restart ---"
RESP3=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-35B-A3B-3bit",
        "messages": [{"role": "user", "content": "Write a haiku about the ocean at sunset."}],
        "max_tokens": 200,
        "temperature": 0.7
    }')
TEXT3=$(echo "$RESP3" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")
echo "Response: $TEXT3"

kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null || true

echo ""
echo "=== SESSION 2 MEMORY LOGS ==="
grep "\[MEMORY\]" "$MEMORY_DIR/log2.txt" 2>/dev/null || echo "(none)"

echo ""
echo "=== RESULT ==="
echo "Session 1: $TEXT1"
echo "Session 2: $TEXT3"
[ -n "$DELTA_FILE" ] && echo "Deltas: PERSISTED (${DELTA_KB} KB)" || echo "Deltas: MISSING"
[ -n "$REPLAY_FILE" ] && echo "Replay: PERSISTED" || echo "Replay: MISSING"
