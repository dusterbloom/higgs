#!/bin/zsh
cd /Users/peppi/.codex/worktrees/4dbe/higgs
python3 target/prefill-investigation/profile45/run.py > target/prefill-investigation/profile45/run.log 2>&1
echo $? > target/prefill-investigation/profile45/run.exit
