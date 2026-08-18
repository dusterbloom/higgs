# Task 3 Report: Escha-W2 MLX exactness gates

## Outcome

Added a bounded synthetic same-kernel row exactness gate in
`crates/higgs-models/src/eschamoe.rs`. No production routing, kernel arithmetic,
kernel defaults, model-loading code, `metal_kernel.rs`, or Qwen production code
changed.

Commit: `0d5d54317debda17849950c25699cc4c229b1956`
(`test(escha): add exactness gates`).

## Pre-edit GitNexus impact

The shared GitNexus index predates some branch-local test symbols. Commands and
results were:

1. `npx gitnexus impact 'EschaProj::gather_forward' --direction upstream --repo higgs`
   - The qualified name was not found (`UNKNOWN`).
   - Resolved command:
     `npx gitnexus impact gather_forward --direction upstream --repo higgs --file crates/higgs-models/src/eschamoe.rs --kind Method --include-tests`
   - Risk: **LOW**.
   - Impacted count: 1; direct callers: 1; processes: 1; modules: 1.
   - Direct affected test flow: `escha_proj_gather_forward_matches_oracle`.

2. `npx gitnexus impact eschamoe_gather_qgemm_matches_scratch_matmul --direction upstream --repo higgs`
   - Risk: **LOW**.
   - Impacted count: 0; direct callers: 0; processes: 0; modules: 0.

3. `npx gitnexus impact escha_native_fixture --direction upstream --repo higgs --file crates/higgs-models/src/qwen3_next.rs --kind Function --include-tests`
   - The fixture was not present in the shared index.
   - Risk: **UNKNOWN**; impacted count reported as 0.
   - The source fixture exists, but it was not edited.

No HIGH or CRITICAL pre-edit risk was reported.

## Changed symbols and behavior covered

Source symbols added:

- `assert_rows_match_bits`: compares each corresponding f32 output row via
  `f32::to_bits()`.
- `eschamoe_gather_kernels_preserve_logical_row_bits`: covers:
  - one deterministic transformed activation row and expert;
  - repeated QMV rows at counts 1, 31, and 32;
  - repeated QGEMM rows at counts 1, 31, 32, and 33;
  - sorted and unsorted corresponding-row permutations;
  - one 32-distinct-expert QGEMM block;
  - the 32-row QMV / 33-row QGEMM `gather_forward_mode` dispatch boundary.

The pre-existing `eschamoe_gather_qgemm_matches_scratch_matmul` test remains a
tolerance diagnostic because scratch and QGEMM use different accumulation and
activation rounding. This change does not claim scratch equivalence is
bit-exact.

## Static verification

All commands below completed with exit status 0:

```text
rustfmt --edition 2024 crates/higgs-models/src/eschamoe.rs
rustfmt --edition 2024 --check crates/higgs-models/src/eschamoe.rs
git diff --check
npx gitnexus detect-changes --repo higgs
git diff --cached --check
```

The pre-commit `gitnexus detect-changes --repo higgs` result was:

```text
Changes: 1 files, 5 symbols
Affected processes: 4
Risk level: medium

Changed symbols:
  eschamoe_gather_qgemm_matches_scratch_matmul
  eschamoe_gather_qgemm_bench
  ROWS
  REPS
  tests

Affected flows:
  Eschamoe_gather_qgemm_bench -> Codebook_for_flag
  Eschamoe_gather_qgemm_bench -> Had_size
  Eschamoe_gather_qgemm_matches_scratch_matmul -> Codebook_for_flag
  Eschamoe_gather_qgemm_matches_scratch_matmul -> Had_size
```

The source diff itself contains only the two newly added test symbols; the
index attributed the insertion to adjacent older symbols because it does not
contain the new symbols yet.

After the commit, `git status --short` produced no output.

## Runtime verification deferred

No build, dependency fetch, GPU test, or runtime fixture was started. This was
required because disk is nearly full, no prepared test target was available,
and the final Task 3 instruction explicitly prohibited builds and runtime
tests.

Commands to run later when a prepared target and GPU clearance are available:

```bash
cargo test -p higgs-models --release -- eschamoe_gather_kernels_preserve_logical_row_bits --nocapture
cargo test -p higgs-models --release -- eschamoe_gather_qgemm_matches_scratch_matmul --nocapture
cargo test -p higgs-models --release -- escha_native_fixture --ignored --nocapture
```

## Token-digest limitation

The optional `HIGGS_ESCHA_TOKEN_DIGEST_OUT` / `HIGGS_ESCHA_TOKEN_DIGEST_REF`
trajectory gate was not added. There was no prepared real-checkpoint runtime in
which to validate a new 129-token artifact, and the final Task 3 scope explicitly
prohibited adding token-digest/model API work. The existing ignored fixture and
its model-loading semantics remain unchanged; `qwen3_next.rs` and
`docs/benchmarking.md` were not modified.

## Concerns

- The new GPU-backed synthetic gate has static verification only; its runtime
  result is deferred.
- The shared GitNexus index is stale for the ignored fixture and new test
  symbols, so its UNKNOWN/adjacent-symbol attribution should not be read as a
  source-level scope expansion.

# Fix-round 1

## Outcome

All four review findings are addressed. There are no open fix-round issues.
This section supersedes the earlier token-digest limitation.

## Addressed findings and exact locations

1. The ignored `escha_native_fixture` at
   `crates/higgs-models/src/qwen3_next.rs:25787` now handles
   `HIGGS_ESCHA_TOKEN_DIGEST_OUT` and `HIGGS_ESCHA_TOKEN_DIGEST_REF` beginning
   at line 25849. It records the initial greedy argmax plus the 128 IDs returned
   by the existing `decode_greedy`, writes 129 whitespace-separated IDs, and
   requires a parsed reference to contain and match all 129 IDs. Its fresh-
   process commands begin in the fixture doc comment at line 25769.
2. The 32/33 dispatch coverage in
   `eschamoe_gather_kernels_preserve_logical_row_bits` at
   `crates/higgs-models/src/eschamoe.rs:2824` uses distinct deterministic
   transformed rows. The 33-row direct QMV/QGEMM witness starts at line 2999;
   the 32-row QMV dispatch assertion starts at line 3042, and the 33-row QGEMM
   dispatch assertion starts at line 3053. Cross-backend scratch comparisons
   remain tolerance-only and unchanged.
3. `assert_rows_match_bits` at
   `crates/higgs-models/src/eschamoe.rs:2802` now takes the expected row count
   and requires the exact `[expected_rows, out_f]` shape. Every call passes its
   expected row count.
4. The 32-expert permutation at
   `crates/higgs-models/src/eschamoe.rs:2907` uses distinct deterministic
   activation rows and applies the same permutation to activation-row/expert-ID
   pairs. The reproducible fixture commands are also documented at
   `docs/benchmarking.md:30`.

No production arithmetic or model-loading behavior changed.

## Fresh static verification

The following commands were rerun during fix-round 1 and exited with status 0:

```text
rustfmt --edition 2024 --check crates/higgs-models/src/eschamoe.rs crates/higgs-models/src/qwen3_next.rs
# no output; exit 0

git diff --check
# no output; exit 0

npx gitnexus detect-changes --repo higgs
# exit 0
Changes: 3 files, 5 symbols
Affected processes: 4
Risk level: medium
```

GitNexus attributed the test insertion to adjacent symbols in its stale index,
as in the original Task 3 report. Source inspection confirms the implementation
changes are confined to the fixture/test surfaces and their documentation.

## Runtime verification deferred

No Cargo build or test was run in fix-round 1. Runtime Cargo tests remain
deferred because the MLX archive/build hit `No space left on device` and there
is no prepared target. This follows the explicit instruction not to run Cargo
builds or tests in this round.

## Open issues

None.

## Post-review runtime validation

After storage was reclaimed, the dedicated branch completed:

- `cargo build --release`: **PASS** (`Finished release profile [optimized]`).
- `eschamoe::tests::eschamoe_gather_kernels_preserve_logical_row_bits`: **PASS**.
- `eschamoe::tests::eschamoe_gather_qgemm_matches_scratch_matmul`: **PASS**; the
  reported relative gaps stayed below `4.0e-4` across K=2/K=3, sorted and
  unsorted fixtures.
- Gather contract validation: **5 PASS**.
- Invalid QMV/QGEMM expert-row guards: **2 PASS**.
- Native short-domain and 31/32/33 dispatch-order tests: **PASS** after commit
  `69fe38895` bounded the synthetic input magnitude. The original fixture
  reached about `43,218` after SwiGLU and overflowed the intentional f16
  scratch matmul at the down projection; production code was not involved.

The ignored real-checkpoint fixture was attempted against the complete local
`Qwen3.6-35B-A3B-Escha-W2` checkpoint in both affine and native modes. Both
processes aborted before loading weights with MLX's
`NSRangeException: __NSArray0 objectAtIndex: index 0 beyond bounds for empty
array` from `metal::Device` enumeration. No logits or 129-token digest was
produced. Focused fresh processes can still enumerate and execute Metal, so
this remains a heavyweight fixture/device-enumeration blocker rather than a
parity result.

## Production-path validation

The fixture was rerun with direct Metal access after confirming that the
ordinary sandbox cannot enumerate the host GPU. The native production path
(`HIGGS_ESCHA_NATIVE=1`, the default) now passes end-to-end:

```text
load: 6.4s
rss_after_load=5.78 GiB
mlx_active=11.16 GB
mlx_peak=11.64 GB
rss_after_forward=5.09 GiB
mlx_active=11.26 GB
mlx_peak=11.69 GB
129-token greedy trajectory written successfully
test result: 1 passed
```

This exercises the same `load_qwen3_5_moe_model` entry point used by
`crates/higgs-engine/src/model_loader.rs`, followed by a real forward and 129
greedy decode steps. The native path retains the trellis expert tensors and
fits the 32-GiB host.

The affine comparison (`HIGGS_ESCHA_NATIVE=0`) reached the real load path but
was terminated with `SIGKILL` during conversion. Its documented converted
footprint is approximately 21.7 GB before model/loader overhead, so this is a
host-memory limit rather than a routing or MLX correctness failure. The
production default remains native for this checkpoint.
