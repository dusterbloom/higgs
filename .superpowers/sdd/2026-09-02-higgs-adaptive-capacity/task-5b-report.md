# Task 5B report: bound model loading

## Outcome

Higgs now rejects unsafe model loads before irreversible process mutation and
rechecks capacity after acquiring the serialized GPU/load gate. A typed,
thread-scoped boundary sink covers every production shard, conversion, and
load-time evaluation path. Critical pressure aborts immediately; constrained
pressure suppresses DFlash and prefill-drafter sidecars without changing the
target loader or runtime prefill-plan cache.

The strict engine estimate follows the files selected by the real adapter,
validates index and safetensors metadata without allocating MLX objects, and
uses checked formulas for standard streaming, Qwen special, native/affine
Escha, Bonsai, VLM, Gemma, and unknown loaders. Failed loads drop partial state,
clear the allocator cache, remeasure, and publish a newer registry memory epoch.
A failed cleanup measurement is returned explicitly rather than hidden.

## Loader inventory and bound

| Production path | Boundary / conservative workspace |
| --- | --- |
| Three shared safetensor loaders | each selected shard; largest selected shard; final model eval |
| Qwen3.5 direct/fused/materialized | each shard plus GDN fusion, materialization, dense requantization, row4 promotion, final eval |
| Native Escha | retained raw artifact plus current real conversion group |
| Affine Escha / unknown | retained upper bound `max(artifact, 2 * artifact)` while sidecars load |
| DFlash / prefill drafter | typed optional begin/end identity; retained or discarded outcome |
| Qwen-VL / LLaVA / Gemma vision | full-artifact reread plus SigLIP assignment/eval |
| Gemma4 MoE | full-artifact second pass and per-expert reshape/eval |
| Bonsai Q1 | exact consolidated file selected by the runtime predicate, CPU read and GPU materialization |

The adapter inventory test exhaustively classifies all 14 built-in load kinds;
adding a new kind requires updating the match and changing the fixed inventory.

## TDD and mutation evidence

- Typed nested sink and unwind tests fail if the sink is not installed,
  propagated, or restored.
- The two-shard shared-loader fixtures reject shard 1 and prove no later
  allocation event. The indexless metadata scan has the same stop proof.
- A real tiny Escha checkpoint reaches two native conversion groups, rejects
  group 1, and observes no later group.
- Strict estimate tests cover largest-shard selection, VLM/Gemma full-artifact
  workspace, native and unknown formulas, Bonsai's exact consolidated-file
  routing predicate, unsafe index paths, mandatory typed index metadata,
  string-only safetensors metadata, corrupt/structurally invalid headers,
  HF file symlinks, and arithmetic overflow.
- Capacity tests cover critical abort, pre-load rejection before loader entry,
  native raw-plus-group charging, and preservation of affine target residency
  during optional loading.
- Optional tests cover initial/environment suppression, a fresh pre-construction
  gate, non-fallback capacity errors, and an exact-once warning latch.
- Cleanup probes prove unwind drops partial state before cache-clear, remeasure,
  and registry publication; publication advances the measurement revision.
  A remeasure failure is appended to the original load failure.
- Post-load facts tests prove the strict estimate is replaced by the serialized
  measured MLX delta and exact artifact/runtime identity.

The focused estimator initially exposed a non-loader-compatible multi-file
fixture; adding the real typed index made it green. Independent review then
found and drove corrections for sidecar residency, metadata strictness,
SigLIP/Qwen post-load evals, serialized precheck freshness, cleanup finality,
and Bonsai file-selection parity. All corrected cases are covered above.

## Independent review

The independent deep review is recorded in `task-5b-internal-review.md`.
After its findings were corrected and the direct-loader inventory rescanned,
the source verdict was **PASS** with no correctness, underbound, cleanup, or
publication blocker remaining.

## Release verification

- `cargo test --release -p higgs-models --lib`: **760 passed, 0 failed, 46 ignored**.
- `cargo test --release -p higgs-engine --lib`: **615 passed, 0 failed, 5 ignored**.
- `cargo test --release -p higgs-engine --test runtime_identity`: **3 passed, 0 failed**.
- `cargo test --release -p higgs --lib`: **736 passed, 0 failed**.
- `cargo test --release -p higgs --test integration_tests`: **107 passed, 0 failed, 10 ignored**.
- `cargo test --release -p higgs --bins`: **4 passed, 0 failed**.
- `cargo build --release -p higgs`: **passed**.
- `cargo fmt -p higgs-engine -p higgs -- --check`: **passed**.
- `git diff --check`: **passed**.

`cargo fmt --all --check` also reports pre-existing formatting debt in unrelated
Escha benchmark lines and `metal_kernel.rs`. Those lines are outside Task 5B's
diff and were preserved. The task-owned `higgs-models/src/lib.rs` formatting
finding from review was corrected; scoped checks and `git diff --check` are the
release formatting evidence.

## Impact and scope

GitNexus rated `build_engine_with_capacity` CRITICAL (7 upstream, 2 direct, 5
processes), `release_failed_engine` CRITICAL (14 upstream, 4 direct, 6
processes), and shared SigLIP loading CRITICAL (10 upstream, 3 direct). Common
optional-prefix loading and the state engine constructor were also CRITICAL;
prefix and Gemma expert loading were HIGH. Very large specialized loader files
were incompletely indexed and therefore treated as CRITICAL lower bounds.

The implementation stays within the loader-pressure seam, its strict estimator,
narrow registry measurement snapshot/revision evidence, focused tests, this
report, and the internal review. It adds no route, admission, generation, config,
or alternate loader behavior. Unrelated `AGENTS.md` and `CLAUDE.md` changes are
preserved and excluded from the commit.

Pre-commit staged GitNexus detection reports 22 task files, 187 mapped symbols,
79 affected processes, and CRITICAL risk, consistent with the disclosed loader
and lifecycle fanout. Compare-to-main is branch-wide (238 files, 4,728 mapped
symbols, 865 processes) because this worktree contains the preceding adaptive-
capacity tasks; it is not a Task 5B-only scope signal. The staged file list was
checked separately and excludes `AGENTS.md` and `CLAUDE.md`.
