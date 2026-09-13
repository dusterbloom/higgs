#!/bin/zsh
cd /Users/peppi/.codex/worktrees/4dbe/higgs
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -u benchmarks/prefill_investigation/evaluator.py target/prefill-investigation/capture-heterogeneous/traces --alpha 0,0.01,0.03,0.1,0.3,0.5,0.9 > target/prefill-investigation/sparse-evaluation/results.jsonl 2> target/prefill-investigation/sparse-evaluation/stderr.log
echo $? > target/prefill-investigation/sparse-evaluation/run.exit
