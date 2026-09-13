#!/bin/zsh
cd /Users/peppi/.codex/worktrees/4dbe/higgs
vm_stat > target/prefill-investigation/gdn-verified/vm-before.txt
pmset -g batt > target/prefill-investigation/gdn-verified/power-before.txt
python3 benchmarks/gdn_prefill/benchmark.py --lengths 1024 --check --repeats 9 --paired-repeats 15 > target/prefill-investigation/gdn-verified/results.json 2> target/prefill-investigation/gdn-verified/stderr.log
echo $? > target/prefill-investigation/gdn-verified/run.exit
vm_stat > target/prefill-investigation/gdn-verified/vm-after.txt
pmset -g batt > target/prefill-investigation/gdn-verified/power-after.txt
