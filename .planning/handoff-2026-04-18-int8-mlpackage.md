# Handoff — 2026-04-18 — DFlash int8 via mlpackage (AB5/AB6/AB7 established)

**Session goal (inherited):** ship int8 weight blobs to break the 18.5 ms DFlash-4B ANE floor per AB3.
**Session outcome:** planning doc falsified + rewritten; four docs updated with new Tier-E ground truth; **zero engine code changed**.

---

## TL;DR for next session

1. The old plan (`emit_blobfile_matmul_tiled_int8` in `ane_mil.rs`) was wrong. Do NOT write that.
2. int8 on ANE IS reachable — but through `.mlpackage` → `MLModel(.cpuAndNeuralEngine)`, a different CoreML entry point than the engine's current `_ANEDesc modelWithMILText:` bridge.
3. **Re-read `.planning/next-session-dflash-int8-weights.md`** — fully rewritten in this session with the correct bridge architecture, pitfalls, acceptance criteria, and first commands.
4. Before touching any ObjC, re-run `/tmp/higgs_int8_probe/plan_4b.py` to confirm AB6 still holds on whatever toolchain you're on (30 s check).

---

## What was proven this session

| Claim | Tier | Evidence |
|---|---|---|
| AB5: raw-MIL int8 path is dead | **E** | `cargo test -p higgs-models --lib --features ane test_int8_conv1x1_nanobot_pattern -- --ignored --nocapture --test-threads=1` → `InvalidMILProgram`. Toolchain: macOS 26.3.1, Xcode 26.0.1, coremlc 3510.2.1, coremltools 9.0. Same failure as 2026-04-03. |
| AB6: mlpackage int8 path is live | **E** | `coremlcompiler compile` of `ct.convert(..., opset=iOS18)` output with `constexpr_affine_dequantize` + conv1x1 succeeds; `MLComputePlan` reports `conv.supported_compute_devices` includes `MLNeuralEngineComputeDevice`. |
| AB7: scheduler threshold | **E** | 64×64 seq=16: `preferred=CPU` (cost 0.96). 3072×3072 seq=16: `preferred=NeuralEngine` (cost 0.54). Realistic DFlash-4B shape flips to ANE; toy probes silently don't. |

## What is still unproven

- Parity vs fp32 reference at realistic shape (predict API hit a loader quirk on the `MLModel` wrapper — compute-plan API worked, predict didn't).
- Wall-clock latency — compute-plan preference ≠ actual dispatch. Need `MLComputePlan` *after eval* or Instruments ANE signpost to confirm.
- Scalability to MLP chain (gate/up/down at `inter=9728`).
- Whether Python 3.14 will ever ship working coremltools (blocks `libcoremlpython`). Today requires a 3.13 sidecar venv.

## Probe artifacts (DO NOT LOSE)

Location: `/tmp/higgs_int8_probe/`

```
.venv/                          # Python 3.13 sidecar with working coremltools 9.0
build_int8_mlpackage.py         # 64x64 toy
build_realistic.py              # 3072x3072 DFlash-4B o_proj shape
compute_plan.py                 # MLComputePlan introspection (toy)
plan_4b.py                      # MLComputePlan introspection (realistic)
run_and_compute_plan.py         # (partial — predict() path not finished)
int8_conv1x1.mlpackage/         # toy build output
int8_conv1x1.mlmodelc/          # toy compiled output
int8_o_proj_4b.mlpackage/       # realistic build output
int8_o_proj_4b.mlmodelc/        # realistic compiled output
```

`/tmp` is ephemeral. **Step 0 of the new plan moves these into `benchmarks/ane_int8_mlpackage_probe/`**. Do this first.

## Docs touched (review these diffs before planning code)

| File | Change |
|---|---|
| `.planning/research/ane-truth/ARCHAEOLOGY.md` | AB4 marked superseded; full AB5/AB6 sections added with toolchain anchors |
| `.planning/research/ane-truth/CLAIMS.md` | New "ANE int8 Weight Path" section — AB1–AB8 in one table |
| `docs/ane-hardware-priors.md` | §7 rule rewritten; §8 table gains AB5/AB6/AB7 rows |
| `.planning/next-session-dflash-int8-weights.md` | **Completely rewritten.** Targets `.mlpackage` bridge, not MIL emitter |

None of these are committed — they show as `M` in `git status`.

## Known landmines for next session

1. **Python 3.14 `.venv` at repo root is broken for coremltools** (T1 / BlobWriter / libcoremlpython not loaded). Create/reuse a 3.13 sidecar via `uv venv --python 3.13 <path>`. Do NOT `pip install coremltools` into the main `.venv`.
2. **The existing `AneKernel` and all of `ane_mil.rs` is fp16-only and stays that way.** The new int8 path adds a sibling bridge (`AneKernel::from_mlpackage`), it does not replace. Two paths coexist.
3. **AB7 scheduler-threshold trap.** A parity test at a small shape will silently pass on CPU and look like success. The plan's step 1 mandates `MLComputePlan` verification before parity — honor this, don't shortcut.
4. **Scope discipline.** The rewritten plan is 4B-only. Do not drag 9B/27B in — 27B has compounding fp16 saturation at `inter=17408` + N17 SRAM cliff.
5. **Compute-plan preferred-device is a preference, not a guarantee.** After first eval on real weights, verify wall-clock matches the bandwidth expectation (1.5×+ fp16 GB/s). If not, ANE probably isn't engaged.

## Unresolved from the rewritten plan

Nothing planned-but-undone. The entire rewrite is the "what's next" deliverable. Engine code is untouched so no WIP to recover.

## Toolchain stamp (for drift checks)

```
macOS:       26.3.1 (25D771280a)
Xcode:       26.0.1 (17A400)
SDK:         26.0
coremlc:     3505.4.1
MIL:         3510.2.1
coremltools: 9.0  (Python 3.13 sidecar)
```

If any of these move, the AB5 kill-test and the AB6 `plan_4b.py` probe both need to re-run before trusting the rewritten plan.
