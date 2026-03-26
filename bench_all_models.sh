#!/usr/bin/env bash
set -euo pipefail

# Benchmark all local MLX models with higgs
# Measures: TTFT, prefill tok/s, decode tok/s at different context lengths

HIGGS="./target/release/higgs"
PORT=8899
BASE="http://localhost:$PORT"
RESULTS_FILE="bench_results_$(date +%Y%m%d_%H%M%S).txt"

export HIGGS_ENABLE_THINKING=0

# Models to benchmark (path : display_name)
MODELS=(
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3-0.6B-4bit|Qwen3-0.6B-4bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3-4B-Instruct-2507-4bit|Qwen3-4B-4bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3-8B-4bit|Qwen3-8B-4bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3-14B-4bit|Qwen3-14B-4bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit|Qwen3.5-0.8B-8bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit|Qwen3.5-2B-8bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit|Qwen3.5-4B-4bit"
  "$HOME/.cache/lm-studio/models/mlx-community/Qwen3.5-9B-MLX-4bit|Qwen3.5-9B-4bit"
  "$HOME/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit|Qwen3.5-35B-A3B-3bit"
  "$HOME/.cache/lm-studio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx|DeepSeek-Coder-V2-Lite-4bit"
)

# Context lengths to test (approximate prompt token counts)
# short=~50 tokens, medium=~500 tokens, long=~2000 tokens
SHORT_PROMPT="Explain what a neural network is in one paragraph."
MEDIUM_PROMPT="Write a detailed technical explanation of how transformer architectures work, covering attention mechanisms, positional encoding, layer normalization, feed-forward networks, and the differences between encoder and decoder architectures. Include discussion of multi-head attention, scaled dot-product attention, and how these components work together. Also explain the training process including backpropagation through the attention mechanism. Discuss the key innovations that made transformers superior to RNNs and LSTMs for sequence modeling tasks. Cover the evolution from the original Attention Is All You Need paper through modern variants like GPT, BERT, and their derivatives. Explain how context windows work and the computational complexity of self-attention. Discuss recent advances in efficient attention mechanisms like sparse attention, linear attention, and sliding window attention. Finally, explain how quantization and pruning can be used to make transformer models more efficient for deployment on edge devices and consumer hardware."
LONG_PROMPT="Write an extremely comprehensive and detailed technical guide covering the following topics in depth. For each topic, provide multiple paragraphs with specific technical details, examples, and explanations:

1. COMPILER DESIGN: Explain lexical analysis, parsing (LL, LR, LALR), abstract syntax trees, semantic analysis, intermediate representations (SSA form, three-address code), optimization passes (constant folding, dead code elimination, loop unrolling, register allocation via graph coloring), and code generation for modern CPU architectures. Discuss how LLVM works internally.

2. OPERATING SYSTEMS: Cover process scheduling algorithms (CFS, MLFQ, lottery scheduling), virtual memory management (page tables, TLB, huge pages, NUMA), file systems (ext4, btrfs, ZFS internals), I/O scheduling, interrupt handling, system calls, and the differences between monolithic and microkernel designs. Explain how modern kernels handle concurrency.

3. DISTRIBUTED SYSTEMS: Explain consensus protocols (Paxos, Raft, PBFT), distributed hash tables, vector clocks, CRDTs, the CAP theorem and its practical implications, leader election algorithms, distributed transactions (2PC, 3PC, saga pattern), and how systems like Spanner, CockroachDB, and TiKV achieve global consistency.

4. CRYPTOGRAPHY: Cover symmetric encryption (AES internals, modes of operation), asymmetric encryption (RSA, elliptic curves, key exchange), hash functions (SHA-256 internals), digital signatures, zero-knowledge proofs, homomorphic encryption, and post-quantum cryptography approaches. Explain TLS 1.3 handshake in detail.

5. DATABASE INTERNALS: Explain B-tree and LSM-tree storage engines, write-ahead logging, MVCC, query optimization (cost-based vs rule-based), join algorithms (nested loop, hash, sort-merge), buffer pool management, and how modern databases handle concurrent transactions with different isolation levels.

6. NETWORKING: Cover TCP congestion control (Reno, CUBIC, BBR), QUIC protocol design, BGP routing, DNS resolution chain, HTTP/3 internals, TLS certificate chains, NAT traversal techniques, and software-defined networking concepts.

7. MACHINE LEARNING SYSTEMS: Explain backpropagation mathematics, gradient descent variants (SGD, Adam, LAMB), mixed-precision training, data parallelism vs model parallelism vs pipeline parallelism, attention mechanism computation, KV-cache optimization, quantization methods (GPTQ, AWQ, GGML), and inference optimization techniques.

Be thorough and technical throughout. This should read like a graduate-level textbook."

MAX_TOKENS=200

log() {
  echo "$1" | tee -a "$RESULTS_FILE"
}

wait_for_server() {
  local max_wait=120
  local waited=0
  while ! curl -sf "$BASE/v1/models" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if [ $waited -ge $max_wait ]; then
      echo "TIMEOUT waiting for server" | tee -a "$RESULTS_FILE"
      return 1
    fi
  done
}

kill_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    unset SERVER_PID
    sleep 2
  fi
}

# Benchmark a single prompt. Streams response, computes TTFT and decode rate.
run_bench() {
  local model_name="$1"
  local prompt="$2"
  local label="$3"

  local tmpfile
  tmpfile=$(mktemp)

  local start_ns
  start_ns=$(python3 -c "import time; print(int(time.time_ns()))")

  # Stream the response, capture timing of first and last chunk
  local first_token_ns=0
  local token_count=0
  local prompt_tokens=0
  local completion_tokens=0

  # Use curl streaming to measure TTFT
  curl -sf --no-buffer -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(cat <<ENDJSON
{
  "model": "$model_name",
  "messages": [{"role": "user", "content": $(python3 -c "import json; print(json.dumps('$prompt'))" 2>/dev/null || echo "\"$prompt\"")}],
  "max_tokens": $MAX_TOKENS,
  "temperature": 0,
  "stream": true
}
ENDJSON
)" 2>/dev/null | while IFS= read -r line; do
    echo "$line" >> "$tmpfile"
  done

  local end_ns
  end_ns=$(python3 -c "import time; print(int(time.time_ns()))")

  # Parse SSE chunks from tmpfile
  local first_content_time=""
  local last_chunk_time=""
  local total_content=""
  local usage_prompt=0
  local usage_completion=0

  # Re-run with python for accurate timing
  python3 - "$BASE" "$model_name" "$prompt" "$MAX_TOKENS" "$label" "$RESULTS_FILE" <<'PYEOF'
import sys, json, time, urllib.request

base, model_name, prompt, max_tokens, label, results_file = sys.argv[1:]
max_tokens = int(max_tokens)

body = json.dumps({
    "model": model_name,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": True
}).encode()

req = urllib.request.Request(
    f"{base}/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"}
)

start = time.perf_counter()
first_token_time = None
token_count = 0
prompt_tokens = 0
completion_tokens = 0
full_text = ""

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
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except:
                    continue
                choices = obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content and first_token_time is None:
                        first_token_time = time.perf_counter()
                    if content:
                        token_count += 1
                        full_text += content
                usage = obj.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
except Exception as e:
    with open(results_file, "a") as f:
        f.write(f"  {label}: ERROR - {e}\n")
    print(f"  {label}: ERROR - {e}")
    sys.exit(0)

end = time.perf_counter()

total_s = end - start
ttft_ms = (first_token_time - start) * 1000 if first_token_time else -1
decode_s = end - first_token_time if first_token_time else 0
# Use server-reported completion_tokens if available, else count
actual_tokens = completion_tokens if completion_tokens > 0 else token_count
decode_tps = actual_tokens / decode_s if decode_s > 0 else 0
prefill_tps = prompt_tokens / (ttft_ms / 1000) if ttft_ms > 0 and prompt_tokens > 0 else 0

result = (
    f"  {label}: prompt={prompt_tokens}tok  TTFT={ttft_ms:.0f}ms  "
    f"prefill={prefill_tps:.1f}tok/s  decode={decode_tps:.1f}tok/s  "
    f"completion={actual_tokens}tok  total={total_s:.1f}s"
)
print(result)
with open(results_file, "a") as f:
    f.write(result + "\n")
PYEOF

  rm -f "$tmpfile"
}

trap kill_server EXIT

log "============================================"
log "HIGGS BENCHMARK - $(date)"
log "Binary: $HIGGS"
log "Max tokens: $MAX_TOKENS"
log "Thinking: disabled"
log "============================================"
log ""

for entry in "${MODELS[@]}"; do
  model_path="${entry%%|*}"
  model_name="${entry##*|}"

  if [ ! -d "$model_path" ]; then
    log "SKIP $model_name (not found: $model_path)"
    log ""
    continue
  fi

  kill_server

  log "--- $model_name ---"
  log "Path: $model_path"

  # Start server with this model
  $HIGGS serve --model "$model_path" --port $PORT &>/dev/null &
  SERVER_PID=$!

  if ! wait_for_server; then
    log "  FAILED to start server"
    log ""
    kill_server
    continue
  fi

  log "  Server ready (PID $SERVER_PID)"

  # Run benchmarks at each context length
  run_bench "$model_name" "$SHORT_PROMPT" "short"
  run_bench "$model_name" "$MEDIUM_PROMPT" "medium"
  run_bench "$model_name" "$LONG_PROMPT" "long"

  kill_server
  log ""
done

log "============================================"
log "DONE - results saved to $RESULTS_FILE"
log "============================================"
