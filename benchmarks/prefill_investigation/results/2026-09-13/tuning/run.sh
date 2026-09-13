#!/bin/zsh
cd /Users/peppi/.codex/worktrees/4dbe/higgs
while [[ ! -f target/prefill-investigation/profile45/run.exit ]]; do sleep 5; done
while [[ ! -f target/prefill-investigation/start-tuning ]]; do sleep 5; done
for arm in control-a chunk512 chunk2048 control-b; do
  python3 target/prefill-investigation/tuning/$arm/run.py > target/prefill-investigation/tuning/$arm/run.log 2>&1
  echo $? > target/prefill-investigation/tuning/$arm/run.exit
done
