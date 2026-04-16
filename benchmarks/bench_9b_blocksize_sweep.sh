#!/usr/bin/env bash
# 9B DFlash block_size sweep on Carnice-9b-MLX + GDN-ANE.
# Finds the block_size that matches actual acceptance rate.
set -euo pipefail

PORT="${PORT:-8911}"
MODEL="${MODEL:-$HOME/.cache/lm-studio/models/jason-schulz/Carnice-9b-MLX}"
DRAFTER="${DRAFTER:-$HOME/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/}"
HIGGS_BIN="${HIGGS_BIN:-./target/release/higgs}"
OUT="${OUT:-benchmarks/runs_$(date +%Y%m%d_%H%M%S)_9b_blocksize_sweep}"
PROMPT="${PROMPT:-Write a short technical paragraph about how MLX schedules Metal kernels for matmul on Apple Silicon. Include why dispatch latency matters when the model is bandwidth-bound.}"
MAX_TOKENS="${MAX_TOKENS:-100}"
BLOCK_SIZES="${BLOCK_SIZES:-4 6 8 12 16}"

mkdir -p "$OUT"

kill_server() {
    local pids
    pids=$(lsof -ti ":$PORT" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        kill -9 $pids 2>/dev/null || true
        sleep 2
    fi
}

run_one() {
    local bs="$1"
    local label="bs${bs}"
    local logfile="$OUT/${label}.log"
    local respfile="$OUT/${label}.resp.json"
    echo "=== [block_size=$bs] ==="
    kill_server

    env \
        HIGGS_ENABLE_THINKING=1 \
        HIGGS_TARGET_ANE_GDN=1 \
        HIGGS_TARGET_COMPILE=1 \
        HIGGS_DFLASH_PATH="$DRAFTER" \
        HIGGS_DFLASH_TRACE=1 \
        HIGGS_DFLASH_BLOCK_SIZE="$bs" \
        RUST_LOG=info \
        "$HIGGS_BIN" serve --model "$MODEL" --port "$PORT" \
        > "$logfile" 2>&1 &
    local server_pid=$!

    local deadline=$((SECONDS + 180))
    while (( SECONDS < deadline )); do
        if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    local model_id
    model_id=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id')

    local req
    req=$(jq -nc --arg p "$PROMPT" --arg m "$model_id" --argjson n "$MAX_TOKENS" '{
        model: $m,
        messages: [{role: "user", content: $p}],
        max_tokens: $n,
        temperature: 0,
        stream: false
    }')

    local start_ts=$(date +%s%N)
    curl -fsS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$req" > "$respfile"
    local end_ts=$(date +%s%N)

    local elapsed_ms=$(( (end_ts - start_ts) / 1000000 ))
    local completion_tokens
    completion_tokens=$(jq -r '.usage.completion_tokens' "$respfile")
    local tps
    tps=$(python3 -c "print(f'{$completion_tokens / ($elapsed_ms / 1000):.2f}')")

    kill -9 "$server_pid" 2>/dev/null || true
    kill_server

    # Steady-state average from trace (skip rounds 1-2 warmup)
    local trace_summary
    trace_summary=$(grep "dflash_trace" "$logfile" \
        | tr -d '\033[];m' \
        | awk 'NR>2' \
        | grep -oE '(draft|lm_draft|verify_build|verify_fwd|round_total|accepted|avg_accept|eff_tps)=[0-9.]+' \
        | awk -F= '{vals[$1]+=$2; n[$1]++} END {for (k in vals) printf "  %-14s avg=%7.1f\n", k, vals[k]/n[k]}' \
        | sort)

    echo "  tokens=$completion_tokens time=${elapsed_ms}ms tok/s=$tps"
    echo "$trace_summary"
    echo
}

echo "9B block_size sweep (target=Carnice-9b-MLX, drafter=Qwen3.5-9B-DFlash)"
echo "Block sizes: $BLOCK_SIZES"
echo

for bs in $BLOCK_SIZES; do
    run_one "$bs"
    sleep 2
done

echo "=== Summary (wall-clock tok/s vs block_size) ==="
for bs in $BLOCK_SIZES; do
    respfile="$OUT/bs${bs}.resp.json"
    logfile="$OUT/bs${bs}.log"
    tokens=$(jq -r '.usage.completion_tokens' "$respfile" 2>/dev/null || echo "?")
    final_tps=$(grep "dflash_trace" "$logfile" 2>/dev/null | tail -1 | grep -oE 'eff_tps=[0-9.]+' | cut -d= -f2)
    avg_accept=$(grep "dflash_trace" "$logfile" 2>/dev/null | tail -1 | grep -oE 'avg_accept=[0-9.]+' | cut -d= -f2)
    echo "  bs=$bs  tokens=$tokens  final_eff_tps=${final_tps:-?}  final_avg_accept=${avg_accept:-?}"
done
echo "Results in: $OUT"
