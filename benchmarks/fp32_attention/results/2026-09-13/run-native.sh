#!/bin/zsh
set -u
cd /Users/peppi/.codex/worktrees/4dbe/higgs
for keys in 8192 16384; do
  out="target/fp32-attention/native-k${keys}"
  mkdir -p "$out"
  shasum -a 256 target/release/examples/prefill_dense_tuning target/release/mlx.metallib crates/higgs-models/examples/prefill_dense_tuning.rs > "$out/source-sha.txt"
  pmset -g batt > "$out/power-before.txt"
  vm_stat > "$out/vm-before.txt"
  target/release/examples/prefill_dense_tuning "$keys" 8 > "$out/results.jsonl" 2> "$out/stderr.log"
  case_exit=$?
  pmset -g batt > "$out/power-after.txt"
  vm_stat > "$out/vm-after.txt"
  print -r -- "$case_exit" > "$out/run.exit"
  if [[ "$case_exit" -ne 0 ]]; then exit "$case_exit"; fi
done
