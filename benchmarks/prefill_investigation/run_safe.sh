#!/bin/zsh
# Run one prefill experiment only when macOS has adequate unified-memory headroom.
set -euo pipefail

min_free_pct="${HIGGS_BENCH_MIN_FREE_PCT:-30}"
min_reclaimable_pages="${HIGGS_BENCH_MIN_RECLAIMABLE_PAGES:-700000}"
pressure="$(memory_pressure -Q)"
free_pct="$(printf '%s\n' "$pressure" | awk -F: '/System-wide memory free percentage:/ {gsub(/[^0-9]/, "", $2); print $2; exit}')"
if [[ -z "$free_pct" ]]; then
  print -u2 "memory preflight failed: could not parse memory_pressure -Q"
  exit 75
fi
if (( free_pct < min_free_pct )) && [[ "${HIGGS_BENCH_ALLOW_LOW_MEMORY:-0}" != 1 ]]; then
  print -u2 "refusing experiment: ${free_pct}% memory free; need ${min_free_pct}%"
  exit 75
fi
vm_before="$(vm_stat)"
reclaimable_pages="$(printf '%s\n' "$vm_before" | awk -F: '
  /Pages free:|Pages inactive:|Pages speculative:|Pages purgeable:/ {gsub(/[^0-9]/, "", $2); total += $2}
  END {print total+0}')"
if (( reclaimable_pages < min_reclaimable_pages )) && [[ "${HIGGS_BENCH_ALLOW_LOW_MEMORY:-0}" != 1 ]]; then
  print -u2 "refusing experiment: ${reclaimable_pages} reclaimable VM pages; need ${min_reclaimable_pages}"
  exit 75
fi

before="$vm_before"
before_swap="$(sysctl -n vm.swapusage 2>/dev/null || true)"
print -u2 "memory preflight: ${free_pct}% free (minimum ${min_free_pct}%)"
print -u2 "swap before: ${before_swap}"
set +e
"$@"
exit_code=$?
set -e
after="$(vm_stat)"
after_swap="$(sysctl -n vm.swapusage 2>/dev/null || true)"
print -u2 "swap after: ${after_swap}"
before_out="${HIGGS_BENCH_VMSTAT_BEFORE:-target/prefill-investigation/vm_stat.before}"
after_out="${HIGGS_BENCH_VMSTAT_AFTER:-target/prefill-investigation/vm_stat.after}"
mkdir -p "${before_out:h}" "${after_out:h}"
print -r -- "$before" > "$before_out"
print -r -- "$after" > "$after_out"
exit $exit_code
