#!/bin/bash
# Benchmark all available Qwen models on Higgs
# Usage: ./bench_models.sh [model_dir_pattern]

PORT=8091
HIGGS="/Users/peppi/Dev/higgs/target/release/higgs"
PROMPT="Explain how a CPU cache works. Be detailed and thorough."
MAX_TOKENS=200

bench_model() {
  local model_dir="$1"
  local model_name=$(basename "$model_dir")
  local thinking_mode="${2:-auto}"

  echo ""
  echo "=============================================="
  echo "MODEL: $model_name (thinking=$thinking_mode)"
  echo "=============================================="

  # Kill any running higgs
  pkill -9 -f higgs 2>/dev/null
  sleep 2

  # Start higgs
  if [ "$thinking_mode" = "off" ]; then
    HIGGS_ENABLE_THINKING=0 RUST_LOG=info "$HIGGS" serve --model "$model_dir" --port $PORT > /tmp/higgs_bench.log 2>&1 &
  else
    RUST_LOG=info "$HIGGS" serve --model "$model_dir" --port $PORT > /tmp/higgs_bench.log 2>&1 &
  fi

  # Wait for server
  for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  # Check if server started
  if ! curl -s http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1; then
    echo "  FAILED TO START"
    tail -5 /tmp/higgs_bench.log
    return 1
  fi

  # Warm-up request
  curl -s --max-time 60 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model_name\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":10,\"temperature\":0.6}" > /dev/null 2>&1

  # Benchmark request
  local start=$(python3 -c "import time; print(time.time())")
  local result=$(curl -s --max-time 120 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model_name\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":0.6}")
  local end=$(python3 -c "import time; print(time.time())")

  python3 -c "
import json, sys
r = json.loads('''$result''')
elapsed = $end - $start
c = r['choices'][0]
u = r['usage']
tokens = u['completion_tokens']
tok_s = tokens / elapsed if elapsed > 0 else 0
msg = c['message']
content = msg.get('content', '')
reasoning = msg.get('reasoning_content', '')
print(f'  Tokens: {tokens}  |  Time: {elapsed:.1f}s  |  Speed: {tok_s:.1f} tok/s')
print(f'  Finish: {c[\"finish_reason\"]}')
print(f'  Content ({len(content)} chars): {content[:150]}...' if len(content) > 150 else f'  Content: {content}')
if reasoning:
  print(f'  Reasoning ({len(reasoning)} chars): {reasoning[:100]}...')
# Check quality
if any(w in content.lower() for w in ['repetition', 'meaning meaning', 'sense sense']):
  print('  QUALITY: DEGRADED (repetition detected)')
elif len(content) < 20 and c['finish_reason'] == 'stop':
  print('  QUALITY: MINIMAL OUTPUT')
else:
  print('  QUALITY: OK')
" 2>/dev/null || echo "  PARSE ERROR: $result"

  # Get decode timing from logs
  local decode_line=$(grep "Decode loop" /tmp/higgs_bench.log | tail -1)
  if [ -n "$decode_line" ]; then
    echo "  $decode_line" | sed 's/.*Decode loop/  Decode:/'
  fi

  pkill -9 -f higgs 2>/dev/null
  sleep 1
}

echo "=============================="
echo "HIGGS MODEL BENCHMARK SUITE"
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"
echo "=============================="

# Test models that have weights
MODELS=(
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3-4B-Instruct-2507-4bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3-14B-4bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit"
)

for model in "${MODELS[@]}"; do
  if [ -d "$model" ] && find "$model" -name "*.safetensors" -print -quit | grep -q .; then
    bench_model "$model" "off"
  fi
done

# Also test NexVeridian 3-bit if available
NEX="$HOME/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit"
if [ -d "$NEX" ] && find "$NEX" -name "*.safetensors" -print -quit | grep -q .; then
  bench_model "$NEX" "off"
  bench_model "$NEX" "auto"
fi

echo ""
echo "=============================="
echo "BENCHMARK COMPLETE"
echo "=============================="
