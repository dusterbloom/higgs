# Next session — can int8 mlpackage actually improve E2E decode?

**Source:** session 2026-04-18 (follow-up to `handoff-2026-04-18-int8-mlpackage.md`).
**Goal:** decide — with evidence, not extrapolation — whether the int8 `.mlpackage` bridge delivers a wall-clock decode win on the DFlash-4B drafter, or whether per-dispatch overhead nullifies it.

---

## What we know (Tier E, measured this session)

### Per-kernel int8 performance on ANE at DFlash-4B shapes

`cargo test -p higgs-models --features ane --lib dflash_ane_{o_proj,mlp_chain}_int8_latency -- --ignored --nocapture --test-threads=1`

With `HIGGS_CORETOOLS_PYTHON=/tmp/higgs_int8_probe/.venv/bin/python`:

| Shape            | seq | min_ms | int8 GB/s | fp16-eq GB/s | on_ane |
|------------------|-----|--------|-----------|--------------|--------|
| o_proj 3072×3072 | 16  | 4.390  | 2.15      | 4.30         | true   |
| o_proj 3072×3072 | 64  | 0.259  | 36.44     | 72.87        | true   |
| o_proj 3072×3072 | 256 | 0.478  | 19.74     | 39.49        | true   |
| gate 9728×3072   | 32  | 0.586  | 51.00     | 101.99       | true   |
| up   9728×3072   | 32  | 0.582  | 51.35     | 102.70       | true   |
| down 3072×9728   | 32  | 0.605  | 49.40     | 98.79        | true   |
| gate 9728×3072   | 64  | 0.624  | 47.89     | 95.78        | true   |
| up   9728×3072   | 64  | 0.618  | 48.36     | 96.71        | true   |
| down 3072×9728   | 64  | 0.633  | 47.21     | 94.42        | true   |
| gate 9728×3072   | 1024| 4.969  | 6.01      | 12.03        | true   |
| down 3072×9728   | 1024| 11.763 | 2.54      | 5.08         | true   |

### The alignment rule (refinement of plan §1 C4)

**seq % 32 == 0 is the binding constraint**, not seq % 16. 64-byte ANE innermost axis / 2 bytes per fp16 = 32 elements. seq ∈ {16, 48, 80} will cliff; seq ∈ {32, 64, 96} are aligned.

Bench evidence: seq=16 → 4.39 ms (~17× slower than seq=64). seq=32 → 0.59 ms, 51 GB/s — fully healthy.

### Compute-bound crossover around seq=128–256

seq ≤ 64: bandwidth-bound, ~48–51 GB/s int8 (~96–102 GB/s fp16-eq). Good ANE utilization (~80% of 60 GB/s ceiling).
seq ≥ 1024: compute-bound. gate@1024 = 12 TFLOPS (~32% of M4 ANE fp16 peak ~38 TFLOPS). down@1024 = 5.2 TFLOPS — anomalously low, scaling cliff at 9728-inner-dim + long seq. Needs investigation before using int8 for prefill.

### The 25% per-kernel win

In the bandwidth-bound regime (seq aligned, reasonable shapes):
- fp16 weights at same shape would need ~2× the bytes in the same wall-clock → 2× effective throughput
- Equivalent to saving ~50% of weight-bound time, or ~25% end-to-end **assuming weight-bound ops dominate the forward**

---

## The open question — why this handoff exists

**35 dispatches per drafter step.** 5 layers × (Q, K, V, O, gate, up, down) = 35 `AneMlPackageKernel::predict_fp16` calls. At 0.6 ms each = 21 ms per decode step — *worse* than today's 18.5 ms fp16 baseline.

Each `predict_fp16` crosses:
- Rust → ObjC FFI
- `MLMultiArray` alloc + `memcpy` input
- `MLDictionaryFeatureProvider init`
- `[MLModel predictionFromFeatures:]` (possibly with IOSurface or internal queuing)
- output `MLMultiArray` stride-walk + `memcpy` to caller

Some of that cost is independent of kernel size. It may be 50 µs (free) or 300 µs (lethal). We don't know.

**The raw-MIL fp16 path probably wins today because it fuses many projections into one `_ANEInMemoryModel` dispatch.** If that's right, int8-via-mlpackage cannot beat it projection-by-projection no matter how fast each kernel runs.

---

## What's dirty in git (not committed)

Branch `feat/magic-canvas`. Uncommitted changes from session 2026-04-18:

- `crates/higgs-models/bridge/ane/ane_bridge_mlmodel.{h,m}` — added `ane_mlmodel_verify_ane_dispatch` (MLComputePlan verifier)
- `crates/higgs-models/src/ane_mlmodel.rs` — added `verify_ane_dispatch()`, `AneMlPackageKernel` (generic shape-agnostic kernel), three ignored tests (`verify_o_proj_4b_prefers_ane`, `dflash_ane_o_proj_int8_parity`, `dflash_ane_o_proj_int8_latency`, `dflash_ane_mlp_chain_int8_latency`)
- `crates/higgs-models/scripts/quantize_int8_proj.py` — sibling to `palettize_lm_head.py`; symmetric per-tensor int8 via conv1x1 + `constexpr_affine_dequantize`
- `benchmarks/ane_int8_mlpackage_probe/` — moved from `/tmp/higgs_int8_probe/`, paths parameterized via `PROBE_OUT_DIR`, added `README.md` + `.gitignore`

All builds clean. All ignored tests pass on macOS 26.3.1 / Xcode 26.0.1 / coremltools 9.0 (3.13 sidecar at `/tmp/higgs_int8_probe/.venv`).

---

## Step 1 result (2026-04-18, measured)

`dflash_ane_dispatch_overhead_probe` in `ane_mlmodel.rs`. Two shapes at seq=32,
`HIGGS_CORETOOLS_PYTHON=/tmp/higgs_int8_probe/.venv/bin/python`:

| shape              | on_ane | min_us | p50_us | p99_us | compute_floor_us | overhead_us | 35× ms |
|--------------------|--------|--------|--------|--------|------------------|-------------|--------|
| tiny 64×64         | false  | 17.0   | 28.0   | 45.0   | 0.2              | 16.8        | 0.59   |
| ane 3072×3072      | true   | 252.0  | 293.0  | 324.0  | 163.8            | 88.2        | 3.09   |

(`compute_floor_us` uses an optimistic 60 GB/s ANE peak — err toward smaller overhead and a safer verdict.)

**Interpretation.** AB7 confirmed: tiny 64×64 silently falls to CPU; its 17 µs is the Rust→ObjC→MLModel floor alone (no coprocessor handoff). The 3072×3072 shape actually engages ANE, so `min_us − compute_floor ≈ 88 µs` is the real ANE-dispatch fixed cost, bandwidth-corrected.

**Verdict: WASH / FUSION-NEEDED.** At 35 dispatches per DFlash-4B drafter step the overhead alone is ~3 ms. The current raw-MIL fp16 path fuses all projections into a single dispatch (~1 of these 88 µs events, not 35). So fanning `AneMlPackageKernel` across projections *adds* ~3 ms; int8's bandwidth headroom only buys back ~1 ms. Net: slower than fp16 baseline.

Decision: **do NOT proceed to Step 2 or Step 4** as written. Routes forward are Step 3 (kernel fusion in mlpackage) or the Step 1-kill fallbacks (§"If Step 1 kills the approach"). User call.

---

## Step 3 result (2026-04-18, measured)

Fusion probe built: new multi-output bridge (`ane_mlmodel_predict_fp16_multi` in `ane_bridge_mlmodel.{h,m}`), Rust wrapper (`AneMlPackageKernel::predict_fp16_multi`), Python builder (`scripts/quantize_int8_fused.py`), and `dflash_ane_fusion_probe` test.

Probe: DFlash-4B QKV at seq=32. Q 3072→3072, K 3072→1024, V 3072→1024. Shared input x [1, 3072, 1, 32]. ANE engaged for both paths.

| path       | min_us | p50_us | floor_us | overhead_us | per-dispatch |
|------------|--------|--------|----------|-------------|--------------|
| individual | 507    | 587    | 277      | 229         | 76.5 µs/×3   |
| fused      | 355    | 376    | 271      | 84          | 84 µs/×1     |

**Fused overhead ≈ 84 µs — essentially equal to the single-dispatch 88 µs baseline. One ANE dispatch event for 3 outputs.** Saved 145 µs on QKV = 63% amortization.

### Projected DFlash-4B decode-step budget

Fusion topology respecting the layer data flow (activations that can actually share an input):

| block        | fusable group       | dispatches/layer |
|--------------|---------------------|------------------|
| Q+K+V        | shared RMSNorm(x)   | 1                |
| O            | takes attn_out      | 1                |
| gate+up      | shared attn_resid   | 1                |
| down         | takes gate⊙silu(up) | 1                |

5 layers × 4 = **20 dispatches/step** (down from 35).
Overhead budget: 20 × 88 µs = 1.76 ms/step.

Vs today's fp16 raw-MIL path (single fused dispatch per layer, ~5 events/step), this adds ~1.3 ms of overhead. The int8 bandwidth win across the MLP chain (747 MB → 375 MB) should save ~1–2 ms. Net looks shippable if the fp16 baseline's compute share is ≥10 ms of the 18.5 ms step.

Verdict: **FUSION AMORTIZES — ship layer-level fusion.** Proceed to an E2E wiring + bench.

---

## Full-layer probe result (2026-04-18, measured)

`dflash_ane_full_layer_int8_probe` in `ane_mlmodel.rs`. Simulates one DFlash-4B layer's weight-work: QKV fused + O single + gate+up fused + down single at seq=32, int8.

| metric             | value    |
|--------------------|----------|
| 1 layer min        | 2.22 ms  |
| 1 layer p50        | 2.48 ms  |
| 5 layers projected | 11.11 ms (min) / 12.39 ms (p50) |
| Effective BW/layer | ~61 GB/s (ANE peak — already optimal) |

**Indeterminate from this alone — but red flag.**

Baseline: fp16 decode step = 18.5 ms total; weights = 230 MB → implied 12 GB/s effective. Raw-MIL `_ANEInMemoryModel` is NOT bandwidth-bound; it caches microcode and streams weights more efficiently than public MLModel. fp16 weight-work portion of 18.5 ms is likely ~8–10 ms, not 14 ms.

If so: int8 via public-MLModel (11 ms weight-work) **loses to fp16 via raw-MIL** (~9 ms weight-work) by 1–3 ms. Per-kernel 2× bandwidth advantage is real but smaller than the raw-MIL fusion+microcode-cache win.

## Updated recommendation

Abandon the public-MLModel fanout → fusion path for DFlash decode. The int8 dispatch overhead (88 µs × 20 dispatches = 1.76 ms) and loss of microcode caching together exceed int8's 2× bandwidth gain on this drafter.

**Actual shipping route: fallback #3** — port `constexpr_affine_dequantize` + int8 conv1x1 into the raw-MIL bridge. AB5/AB6 said `compileWithQoS:` rejected those ops as of 2026-04-03/18; retest on current Xcode/macOS before committing. One-session probe: build a raw-MIL program that goes through `_ANEInMemoryModel::compileWithQoS:` and does not reject the int8 op; if accepted, parity test; if not, int8 is officially prefill-only on this drafter.

---

## Investigation plan — in strict order

### Step 1 — Measure per-dispatch overhead (30 min, the decision point) — DONE, see above

Build a pathological tiny int8 mlpackage (e.g. 64×64, seq=32 aligned — but the work is trivial). Time `predict_fp16` on it. Subtract the actual compute time (estimate from larger shapes' GB/s) to isolate overhead.

- If overhead ≤ 50 µs/call: 35 dispatches adds ~1.75 ms. Fine, fan out.
- If overhead 100–200 µs/call: 35 dispatches adds 3.5–7 ms. Fanout is a wash. Need fusion.
- If overhead ≥ 300 µs/call: 35 dispatches adds ≥10 ms. Fusion is mandatory.

Concrete test to add: `dflash_ane_dispatch_overhead_probe` in `ane_mlmodel.rs`. Build a 64×64 conv1x1 at seq=32, run 1000 iters, report min/median. Compare the 9728×3072 @ seq=32 min (0.586 ms) minus compute estimate (29 MB / 60 GB/s = 0.48 ms) → implied overhead ≈ 0.1 ms/call. But that's a single-point estimate; the 64×64 probe makes it robust.

### Step 2 — Only if Step 1 says "feasible" — parity at seq=32 for QKV shapes

`dflash_ane_qkv_int8_parity_seq32` — same pattern as `dflash_ane_o_proj_int8_parity` but for Q [3072, 3072], K [1024, 3072], V [1024, 3072] (GQA shapes for DFlash-4B). Verify parity ≤ 0.08, ANE dispatch, GB/s ≥ 40.

### Step 3 — Fusion probe (only if Step 1 says "fusion mandatory")

Build a single mlpackage that runs all 4 attention projections (Q, K, V, O) of one layer from one activation input. This is a significant coremltools change — you're stacking 4 constexpr_affine_dequantize + conv1x1 into one MIL program. Measure end-to-end for that layer, compare vs 4× individual.

If this works, it reframes the entire shipping story: one mlpackage per layer instead of 7. Cache implications, weight-packing implications, all different.

### Step 4 — End-to-end decode bench

Only after Steps 1–3 give a green light. Wire `AneMlPackageKernel` into `dflash_ane.rs` behind `HIGGS_DFLASH_ANE_INT8=1`. Run `benchmarks/bench_9b_blocksize_sweep.sh` 4B-equivalent. Compare tok/s at the environment flag off/on.

Acceptance per plan: ≥15% faster at int8=1 on DFlash drafter. If Step 1 measurement predicts <15% headroom after dispatch overhead, don't bother with Step 4 — tell the user and revisit the architecture.

---

## First commands

```bash
# 0. Sanity: probe still holds on current toolchain (30 s)
ls /tmp/higgs_int8_probe/.venv || echo "REBUILD: uv venv --python 3.13 /tmp/higgs_int8_probe/.venv && pip install coremltools==9.0 numpy"
PROBE_OUT_DIR=/tmp/higgs_int8_probe /tmp/higgs_int8_probe/.venv/bin/python benchmarks/ane_int8_mlpackage_probe/plan_4b.py
# expect: preferred=MLNeuralEngineComputeDevice

# 1. Start Step 1 of the plan — add the overhead probe test.
#    The existing `dflash_ane_o_proj_int8_latency` is the template.
#    Shape: 64×64, seq=32, iters=1000, warmup=50.
```

---

## Landmines (inherited + new)

Inherited from 2026-04-18 handoff — still apply:
- Python 3.14 venv is dead for coremltools; use the 3.13 sidecar.
- `compute_plan says ANE` ≠ ANE-grade wall clock. Always pair with latency.
- Do NOT extend `ane_mil.rs` — it is fp16-only on purpose.

New this session:
- Alignment is **seq % 32 == 0**, not 16. Any seq that's not a multiple of 32 is a ~15× cliff at this shape class.
- The `down` projection (3072×9728) shows scaling problems at seq=1024 (~half the throughput of gate/up). Do not use int8 for prefill without probing down separately at each prefill seq.
- Per-dispatch overhead is the unverified assumption gating everything. Measure it before any code that fans out `AneMlPackageKernel` across projections.

---

## If Step 1 kills the approach

Fallbacks in order of effort:
1. **Kernel fusion in mlpackage** (Step 3) — biggest change, biggest potential payoff.
2. **Shared `MLModel` with multiple outputs** — less invasive than fusion but needs different coremltools MIL structure.
3. **Abandon the public-MLModel path; port the int8 ops into the raw-MIL bridge.** AB5 says this doesn't compile today, but that was 2026-04-03 and 2026-04-18 on the specific ops we tried. New ops (e.g. `quantize_linear` + `dequantize_linear` vs `constexpr_affine_dequantize`) might land differently.
4. **Accept that ANE int8 is a prefill-only story, not a decode story.** Wire it only into paths that see seq ≥ 32 naturally (non-drafter inference), and keep the fp16 raw-MIL path for the DFlash drafter.

Each is a full session. Step 1 tells you which one to spend it on.
