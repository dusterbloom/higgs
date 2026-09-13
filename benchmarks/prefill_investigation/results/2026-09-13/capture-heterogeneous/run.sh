#!/bin/zsh
cd /Users/peppi/.codex/worktrees/4dbe/higgs
while [[ ! -f target/prefill-investigation/gdn-verified/power-after.txt ]]; do sleep 2; done
python3 target/prefill-investigation/capture-heterogeneous/run.py > target/prefill-investigation/capture-heterogeneous/run.log 2>&1
echo $? > target/prefill-investigation/capture-heterogeneous/run.exit
