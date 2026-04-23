#!/bin/bash
# Parse-only companion to run_probe_v2.sh. Produces v2/RESULTS.md from
# whatever logs exist in v2/. Safe — no server launches.
set -u
cd /Users/peppi/Dev/higgs
OUT=.planning/measurements/phase2-verify-build/v2

CONFIGS=(
  "baseline|12|32|0.88"
  "cap094|12|32|0.94"
  "cap100|12|32|1.0"
  "capunset|12|32|"
  "bs16|16|32|0.88"
)

main_rounds() {
  local log="$1"
  awk '/dflash_trace round=1 embed=/ {last=NR; lines[NR]=$0; for(i=NR-1;i>=1;i--) delete lines[i]; next}
       {lines[NR]=$0}
       END {for (i=1; i<=NR; i++) if (i in lines && i >= last) print lines[i]}' "$log" 2>/dev/null
}

parse_field() {
  local log=$1 key=$2 mode=$3
  local vals
  vals=$(main_rounds "$log" | grep -oE "${key}=[0-9.]+" | sed "s/${key}=//")
  if [[ -z "$vals" ]]; then echo "-"; return; fi
  if [[ $mode == last ]]; then echo "$vals" | tail -1; return; fi
  # BSD awk has no asort → sort externally, then median in awk
  echo "$vals" | awk 'NR>3 {print}' | sort -n | awk '{a[NR]=$1} END{n=NR; if(n==0){print "-"; exit} if(n%2) print a[(n+1)/2]; else printf "%.2f\n",(a[n/2]+a[n/2+1])/2}'
}

accept_fraction() {
  local log=$1 block=$2
  main_rounds "$log" | grep -oE 'accepted=[0-9]+' | sed 's/accepted=//' \
    | awk -v b="$block" 'NR>3 {s+=$1; n++} END{if(n==0){print "-"; exit} printf "%.2f", s/(n*b)}'
}

backend_of() {
  if grep -q 'DFlash ANE worker spawned' "$1" 2>/dev/null; then echo "ANE+CPU"
  elif grep -q 'HIGGS_DFLASH_DISABLE_ANE' "$1" 2>/dev/null; then echo "CPU BLAS (forced)"
  elif grep -q 'CPU BLAS' "$1" 2>/dev/null; then echo "CPU BLAS"
  else echo "unknown"
  fi
}

rss_mb() {
  local bytes=$(grep 'maximum resident set size' "$1" 2>/dev/null | tail -1 | awk '{print $1}')
  if [[ -z "$bytes" ]]; then echo "-"; return; fi
  echo $(( bytes / 1024 / 1024 ))
}

cap_mb_of() {
  grep -oE 'cap_mb=[0-9]+' "$1" 2>/dev/null | tail -1 | sed 's/cap_mb=//'
}

completion_excerpt() {
  local f=$1
  if [[ ! -s "$f" ]]; then echo "(empty/missing)"; return; fi
  python3 -c "import json; d=json.load(open('$f'));
ch=d.get('choices',[{}])[0]; m=ch.get('message',{}) or {};
t=m.get('content') or ch.get('text') or '';
print(t[:80].replace(chr(10),' '))" 2>/dev/null || head -c 160 "$f"
}

{
  echo "# Phase 2 Probe v2 — salvage parse (Mac crashed mid-sweep)"
  echo ""
  echo "Date parsed: $(date -u +%FT%TZ)"
  echo "Target: Qwen3.6-27B-4bit  Drafter: Qwen3.5-27B-DFlash  Chunk: 32 (all)"
  echo ""
  echo "Crash context: bs16 launched at 10:15:51Z, Mac went down during model load."
  echo "Only cap094 and capunset produced full completions."
  echo ""
  echo "| config | BS | cap | vbuild_med | vfwd_med | rtotal_med | eff_tps_last | avg_accept_frac | peak_rss_mb | cap_mb | backend |"
  echo "|--------|----|-----|------------|----------|------------|--------------|-----------------|-------------|--------|---------|"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME BLOCK CHUNK CAP <<<"$cfg"
    LOG="$OUT/$NAME.log"
    if [[ ! -f "$LOG" ]]; then
      printf "| %s | %s | %s | (no log) |\n" "$NAME" "$BLOCK" "${CAP:-unset}"
      continue
    fi
    CAP_MB=$(cap_mb_of "$LOG"); [[ -z "$CAP_MB" ]] && CAP_MB="-"
    printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
      "$NAME" "$BLOCK" "${CAP:-unset}" \
      "$(parse_field "$LOG" verify_build median)" \
      "$(parse_field "$LOG" verify_fwd   median)" \
      "$(parse_field "$LOG" round_total  median)" \
      "$(parse_field "$LOG" eff_tps      last)" \
      "$(accept_fraction "$LOG" "$BLOCK")" \
      "$(rss_mb "$LOG")" \
      "$CAP_MB" \
      "$(backend_of "$LOG")"
  done
  echo ""
  echo "## Completion excerpts (first 80 chars)"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME BLOCK CHUNK CAP <<<"$cfg"
    echo "- $NAME: $(completion_excerpt "$OUT/$NAME.completion.json")"
  done
  echo ""
  echo "## Verdict"
  echo ""
  echo "Stop-criterion: verify_build ≤ 200 ms AND eff_tps ≥ 10 AND avg_accept ≥ 0.30."
} > "$OUT/RESULTS.md"

echo "wrote $OUT/RESULTS.md"
cat "$OUT/RESULTS.md"
