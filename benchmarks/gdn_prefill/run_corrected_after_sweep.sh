#!/bin/zsh
set -u

gate="${GDN_SWEEP_GATE:-target/prefill-investigation/tuning/control-b/run.exit}"
output_root="${GDN_OUTPUT_ROOT:-target/prefill-investigation}"
benchmark_python="${GDN_BENCHMARK_PYTHON:-/opt/homebrew/bin/python3}"
assert_python="${GDN_ASSERT_PYTHON:-/opt/homebrew/bin/python3}"
while [[ ! -f "$gate" ]]; do
  sleep 5
done

run_case() {
  suffix="$1"
  dtype="$2"
  out="${output_root}/gdn-corrected-${suffix}"
  mkdir -p "$out"
  {
    print -r -- "kernel_source_commit=8c6b5f66ba985b0c823ccca82af6f32d73da5ea4"
    print -r -- "benchmark_sha256=$(shasum -a 256 benchmarks/gdn_prefill/benchmark.py | awk '{print $1}')"
  } > "$out/source-sha.txt"
  pmset -g batt > "$out/pmset-before.txt"
  vm_stat > "$out/vm-stat-before.txt"
  "$benchmark_python" benchmarks/gdn_prefill/benchmark.py \
    --dtype "$dtype" --lengths 1024 --check --repeats 9 --paired-repeats 15 \
    > "$out/results.json" 2> "$out/stderr.log"
  bench_exit=$?
  pmset -g batt > "$out/pmset-after.txt"
  vm_stat > "$out/vm-stat-after.txt"
  if [[ "$bench_exit" -ne 0 ]]; then
    print -r -- "$bench_exit" > "$out/exit.code"
    return "$bench_exit"
  fi
  "$assert_python" - "$out/results.json" > "$out/assertions.txt" 2>> "$out/stderr.log" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"1": [32, 1, 1], "2": [32, 2, 1], "4": [32, 4, 1], "8": [32, 8, 1]}
assert all(row["output_bitwise_equal_tg4"] and row["state_bitwise_equal_tg4"]
           for row in result["t1024_bitwise_parity"].values())
assert result["scalar_reference_checks"][0]["actual_threadgroup"] == expected["4"]
for row in result["paired_tg4_vs_candidate"]:
    assert row["actual_threadgroup"] == expected[str(row["threadgroup_y"])]
print("parity=pass actual_threadgroups=pass scalar_oracle_tg4=pass")
PY
  assertion_exit=$?
  print -r -- "$assertion_exit" > "$out/exit.code"
  return "$assertion_exit"
}

run_case bf16 bfloat16 || exit $?
run_case fp32 float32
