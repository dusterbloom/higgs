# Task 5B Independent Deep Review

**Verdict:** SOURCE PASS; RELEASE HOLD on the formatting/gate item below.

## Release-blocking finding

- **[P1] Step 7 formatting gate is not currently green.** `cargo fmt --all --check` reports formatting differences in task-owned `crates/higgs-models/src/lib.rs` and `crates/higgs-models/src/eschamoe.rs`. It also reports unrelated, pre-existing differences in `crates/higgs-models/src/metal_kernel.rs`, which must not be swept into this task. Format the task-owned edits (or run and record an equivalently scoped formatter check) before commit. `git diff --check` is clean.

## Correctness result

No unresolved correctness underbound, missed direct loader, dangling post-rejection event, cleanup-order defect, publication-order defect, or scope-expanding alternate loader path remains in the reviewed diff from `22e3e545c`.

The final rescan verified:

- The thread-local typed load sink propagates `ModelError`, nests correctly, and restores the prior sink on normal return and unwind. Optional-load policy/outcome state is likewise scoped by RAII.
- All three shared safetensor loaders report before/after shard events and separately checkpoint final model evaluation.
- Qwen direct/fused shard loading, full checkpoint materialization, GDN fusion, per-parameter materialization, final evaluation, dense-GDN requantization, and default row4 promotion now have rejection-capable boundaries before the allocation/evaluation work.
- Native and affine Escha retain the raw artifact, checkpoint each real conversion group before `take`/conversion/eval, and stop immediately on sink rejection. The actual-loop native regression rejects group 1 and observes no later group.
- DFlash, Qwen-VL, LLaVA, Gemma vision/SigLIP evaluation, Gemma4's second pass/expert reshape, and Bonsai's direct CPU-read/GPU-materialization path are bounded. Full-artifact VLM work remains charged while the all-weight map is consumed.
- Adapter inventory covers all 14 built-in load kinds and has no unclassified entry. The direct-load search found no unaccounted production safetensor read beyond the enumerated paths; remaining hits are tests or image input reads.
- The strict estimator validates the exact selected files and safetensor structure, rejects unsafe normalized paths, directory/dangling/non-file links, invalid mandatory index metadata, invalid string metadata, offset/shape/dtype corruption, empty artifacts, and checked-arithmetic overflow. HF-style regular-file symlinks remain supported.
- Standard loading uses artifact plus largest selected shard; Qwen/native/full-artifact use artifact plus one artifact workspace; affine/unknown use artifact plus checked `2 * artifact` workspace. Bonsai selection now matches the runtime's exact `qwen3`/1-bit/group-size predicate and counts the consolidated file it actually reads.
- A fresh capacity snapshot is re-enforced after acquiring the serialized GPU/load gate and before loader entry. Every boundary re-samples pressure/headroom without retaining registry locks across MLX work. Critical pressure aborts; constrained pressure suppresses optional sidecars while allowing a fitting target.
- The load ledger preserves the conservative retained upper bound for affine/unknown targets while optional models load, and preserves each retained optional model's `max(artifact, workspace)` residency before admitting another sidecar.
- Optional DFlash, including the environment-selected path, and the prefill drafter have typed begin/end identities. Dynamic constrained pressure discards a partially loaded optional object before allocator-cache clearing; capacity failures are not silently converted into fallback success.
- Pre-load rejection occurs before the irreversible wired-limit mutation. Failed loads unwind/drop partial state before cache clearing, remeasurement, and registry publication. Cleanup measurement failure is surfaced rather than hidden. Post-load facts use serialized measured MLX delta rather than the preflight estimate.
- Batch model/tokenizer/template loading completes before its unpublished worker handle is returned, and no route publication occurs in this load seam. Later publication failure shuts the engine down before cache clear/remeasure/epoch refresh.
- Changes outside the loader seam are limited to the capacity snapshot/read surface, typed errors/exports, and the registry revision accessor needed by cleanup verification. Unrelated `AGENTS.md` and `CLAUDE.md` changes were not touched or reviewed as Task 5B work.

## Evidence reviewed

- Owner-reported final focused evidence: strict estimator suite **14/14 GREEN**, real native conversion-stop test **GREEN**, updated `higgs-models` compilation **GREEN**.
- Independent static verification: `git diff --check` **GREEN**; direct-loader/read inventory re-scanned after the final fixes.
- `cargo fmt --all --check` **RED** as described above.
- The remaining serialized full release suites/build from Step 7 were not independently duplicated during the owner's active build pipeline; their exact results must be recorded in the Task 5B report before commit.

## Non-blocking review note

The preload ordering, exact-once suppression notice, and cleanup sequencing tests exercise small compositional helpers/fake closures rather than the full server constructor. The production call order was reviewed directly and is correct, but a future mutation-focused harness around the composed constructor would give stronger protection against call-site deletion without requiring real model pressure.
