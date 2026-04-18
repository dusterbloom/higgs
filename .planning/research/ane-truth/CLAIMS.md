# ANE & Hardware Ground Truth — Structured Claims

**Scope:** claims extracted from repo sources (measurements, code, docs, handoffs).
**Evidence tiers:** E = empirically measured / reproducible | D = documented in code/spec | I = inferred/analytical | S = speculation / unverified.
**Status date:** 2026-04-18 (incomplete — handoff narratives not yet read).

---

## Hardware — Apple M4 (base)

| # | Claim | Tier | Source |
|---|---|---|---|
| H1 | Memory bandwidth: 120 GB/s | D | `.planning/analysis/optimal-compute-routing-roofline.md:6` (spec) |
| H2 | GPU fp16 sustained: ~3.5-4.0 TFLOPs (use 3.8 conservative) | D | roofline.md:7 |
| H3 | GPU quantized matmul effective: 2.5-3.5 TFLOPs (dequant overhead) | I | roofline.md:299 |
| H4 | ANE int8 throughput scales strongly with M | E | ane_c1_sustained_tflops.md |
| H5 | ANE M=16 ≈ 0.25 GFLOPs (dispatch-dominated) | E | ane_bench_raw.json |
| H6 | ANE M=128 ≈ 2.9-3.8 TFLOPs (attn-proj, moe-gateup, moe-down) | E | ane_c1_sustained_tflops.md:13,17,21 |
| H7 | ANE M=1024 ≈ 11.3 TFLOPs (prefill-big shape) | E | ane_c1_sustained_tflops.md:25 |
| H8 | ANE warm dispatch round-trip: 0.032 ms | E | ane_g2_dispatch_roundtrip.md:13 |
| H9 | ANE cold first dispatch: 0.1 ms | E | ane_g2_dispatch_roundtrip.md:13 |
| H10 | ANE compile time: 16-120 ms depending on shape | E | ane_bench_raw.json (compile_ms) |
| H11 | Prior assumption of ≥18 TFLOPs ANE gate **FAILED** on M4 | E | ane_c1_sustained_tflops.md:32 |

## ANE Behavior / Routing

| # | Claim | Tier | Source |
|---|---|---|---|
| A1 | ANE silently falls back to GPU on unsupported shapes (e.g. lm_head N=152,000) | E | ane_c1_sustained_tflops.md:29 (1036 ms on `cpuAndNe`, slower than cpuOnly's 222 ms) |
| A2 | Detection heuristic: if `cpuAndNe` timings match `cpuAndGpu` within noise, ANE didn't run | D | benchmarks/ane_coreml_bench/README.md:79-82 |
| A3 | `cpuAndGpu` is slower than `cpuOnly` for matmuls (CPU+GPU coordination tax) | E | ane_c1_sustained_tflops.md:11-12 (675 vs 184 GFLOP/s on attn-proj) |
| A4 | ANE rows must be padded to multiples of 128 | D | benchmarks/ane_coreml_bench/README.md:27, roofline terminology |
| A5 | ANE requires int8 weights (2.37× more bytes per param vs 3-bit) | D | roofline.md:179 |
| A6 | Max 16-tile ANE compiler cap | D | `ane_mil.rs:138-142` via gdn-verify-dispatch-batch.md:76 |
| A7 | Tile budget: 16 MiB (`compute_blobfile_tile_plan`) | D | ane_mil.rs:137-163 |
| A8 | `AneKernel` is `!Send + !Sync` — IOSurface handles are thread-bound | D | `qwen3_next_ane_worker.rs:6-7` |
| A9 | Worker thread pattern exists because `!Send` kernel must co-live with `Send` model | D | `qwen3_next_ane_worker.rs:6-17` |
| A10 | One physical ANE → dispatches serialize at hardware even with multiple workers | I | gdn-verify-dispatch-batch.md:218-219 |
| A11 | Single-kernel realtime mode available (`begin_realtime()`) — saves Metal commit per dispatch | D | `qwen3_next_ane_worker.rs:567-572` |

## ANE Dispatch Cost (GDN case study, 9B)

| # | Claim | Tier | Source |
|---|---|---|---|
| D1 | 9B Carnice with `HIGGS_TARGET_ANE_GDN=1`: 24 linear layers × 3 projections = 72 blocking ANE dispatches per verify | E | gdn-verify-dispatch-batch.md:11-20 |
| D2 | Per-dispatch overhead: 3.17 ms measured (228 ms / 72) | E | gdn-verify-dispatch-batch.md:19-20 |
| D3 | Root cause of 3.17 ms: caller-side `x_f32.eval()` at `qwen3_next_ane_worker.rs:297` forces Metal sync mid-graph | I | gdn-verify-dispatch-batch.md:85-116 |
| D4 | Super-linear BS scaling evidence (209 ms @ BS=8, 232 @ BS=12, 346 @ BS=16) → confirms GPU sync stacking | E | gdn-verify-dispatch-batch.md:115 |
| D5 | GPU verify for same GDN layers: 144 ms (below ANE's 228 ms overhead) | E | gdn-verify-dispatch-batch.md:27-29 |
| D6 | **Topology B** (`HIGGS_TARGET_ANE_GDN=0`): 22.49 tok/s vs 18.08 tok/s ANE-on. GPU wins +24% | E | gdn-verify-dispatch-batch.md:172-175 |
| D7 | Fused qkvz+ba collapses 72→48 dispatches, estimated 228→~160 ms, still loses to GPU's 144 ms | I | gdn-verify-dispatch-batch.md:138 |
| D8 | Cross-layer pipelining is illegal (`qwen3_next.rs:4815` shows layer N+1 depends on N output) | D | gdn-verify-dispatch-batch.md:63-65 |
| D9 | NEON transpose overhead in `cpu_to_ane`/`ane_to_cpu` — ~10-20 ms across 72 dispatches | I | gdn-verify-dispatch-batch.md:118-121 |

## ANE Tooling / Toolchain

| # | Claim | Tier | Source |
|---|---|---|---|
| T1 | Python 3.14 lacks `libcoremlpython` C extension — `model.predict()` unusable | D | benchmarks/ane_coreml_bench/README.md:43-45 |
| T2 | coremltools `model.predict()` silently reshapes int8 → fp32 (destroys bench) | D | benchmarks/ane_coreml_bench/README.md:86-89 |
| T3 | pyobjc `MLMultiArray(dataType=Int8)` required for int8 inputs | D | benchmarks/ane_coreml_bench/README.md:45-47 |
| T4 | macOS 15+ required; iOS18 opset, spec v10+ | D | benchmarks/ane_coreml_bench/README.md:41-43 |
| T5 | MIL runtime uses `MLFeatureValue` inputs (not baked weights) in ccv-style sandwich | D | benchmarks/ane_coreml_bench/README.md:14-26 |
| T6 | ANE realtime mode (`begin_realtime`) is thread-local in bridge | D | `qwen3_next_ane_worker.rs:568-570` |

## MLX-rs Library State

| # | Claim | Tier | Source |
|---|---|---|---|
| M1 | MLX-rs pinned to git rev `af21d79` (oxideai/mlx-rs), NOT a release version | D | `Cargo.toml:83-84` |
| M2 | `docs/mlx_rs_capabilities.md` claims "mlx-rs 0.25.3" — **may be stale** vs current rev | S | `docs/mlx_rs_capabilities.md:1` |
| M3 | `gather_mm` (unquantized) NOT available → workaround = dequant + regular matmul per expert | D | mlx_rs_capabilities.md:33-34 |
| M4 | Custom Metal kernels NOT available → GDN uses `compile` for element-wise fusion only | D | mlx_rs_capabilities.md:35 |
| M5 | `SwitchLinear` NOT available → manual stacked weights + `gather_qmm` | D | mlx_rs_capabilities.md:36 |
| M6 | `gather_qmm` accessible via FFI wrapper to `mlx_sys::mlx_gather_qmm`, not native mlx-rs API | D | mlx_rs_capabilities.md:27 |
| M7 | Quantized weight key remapping required: flat `.weight` → nested `.inner.weight` | D | mlx_rs_capabilities.md:39-51, `higgs-models/src/lib.rs::remap_quantized_key` |
| M8 | Conv1d weight format: MLX `[out, kernel, in/groups]` vs PyTorch `[out, in/groups, kernel]`; `sanitize()` does `moveaxis(2,1)` | D | mlx_rs_capabilities.md:64-68 |

## Environment Flags (ACTUAL vs PROPOSED)

| # | Flag | Status | Behavior | Source |
|---|---|---|---|---|
| F1 | `HIGGS_TARGET_ANE_GDN` | **IMPLEMENTED** | "active — known to regress prefill ~3× (and decode ~38% on dFlash)" | `model_loader.rs:180` |
| F2 | `HIGGS_TARGET_ANE_LM_HEAD` | **IMPLEMENTED** (recent) | Dequantize lm_head to ANE; only if untied word embeddings | `model_loader.rs:257-291` |
| F3 | `HIGGS_ANE_GDN_WORKER` | IMPLEMENTED | Legacy worker fallback | `model_loader.rs:195` |
| F4 | `HIGGS_ANE_LM_HEAD_SEQ` | IMPLEMENTED | Seq bucket override for lm_head | `model_loader.rs:268` |
| F5 | `HIGGS_ANE_DUMP_MIL` | IMPLEMENTED | Diagnostic MIL dump | `dflash_ane.rs:278` |
| F6 | `HIGGS_DFLASH_PIPELINE` | IMPLEMENTED | Pipeline mode gate (reverted default=false) | commit `c1f85ade` |
| F7 | `HIGGS_TARGET_ANE_PREFILL` | **NOT IMPLEMENTED** | Proposed in roofline.md:549, no code exists | grep null |

## Roofline / Theoretical (Qwen3.5-35B-A3B-3bit on M4)

| # | Claim | Tier | Source |
|---|---|---|---|
| R1 | Decode (M=1) is bandwidth-bound | I | roofline.md:134-136 |
| R2 | Decode ceiling GPU-only: 77-87 tok/s; measured 55 tok/s (29-37% gap) | I | roofline.md:211-214 |
| R3 | Decode: ANE provides **ZERO value** — int8 reads 2.37× more bytes than 3-bit at M=1 | I | roofline.md:177-181, 585 |
| R4 | Prefill (M≥512) is compute-bound | I | roofline.md:232-254 |
| R5 | Prefill GPU ceiling: ~595 tok/s; measured 316 tok/s (48% gap) | I | roofline.md:305-308 |
| R6 | Hybrid ANE+GPU prefill ceiling: ~683 tok/s (+15% over GPU-only) | I | roofline.md:424 |
| R7 | ANE at M=1024 is 3.2-3.7× faster than GPU for dense projections **in isolation** | I | roofline.md:368-372, 377 |
| R8 | ANE cannot run MoE `gather_qmm` (no equivalent) | D | roofline.md:379, mlx_rs_capabilities.md:33 |
| R9 | Estimated realistic prefill gain from ANE: +20-40% tok/s (316→380-440) | I | roofline.md:591 |
| R10 | ANE sync overhead estimated 3-5 ms/layer × 40 layers = 160 ms at M=1024 | I | roofline.md:416-418 |

## ANE int8 Weight Path (re-measured 2026-04-18)

| # | Claim | Tier | Source |
|---|---|---|---|
| AB1 | DFlash-4B drafter fp16 forward: 1010 MB / 18.5 ms = 54.6 GB/s = 45% of M4 peak (bandwidth-bound) | E | ARCHAEOLOGY.md §AB + `memory/dflash-ane-projections-v2-handoff.md` |
| AB2 | v1→v2 scheduling tricks moved ctx=16 by only 3.4% — diminishing returns against the wall | E | ARCHAEOLOGY.md §AB |
| AB3 | int8 weight blobs (1010→505 MB) project ~9.2 ms ANE floor at same 54.6 GB/s → ~12-13 ms total | I | ARCHAEOLOGY.md §AB |
| AB4 | ~~`build_weight_blob_int8` + `constexpr_affine_dequantize` in `ane_mil.rs` emits the int8 path~~ **SUPERSEDED** by AB5/AB6 | D→X | see below |
| AB5 | Raw-MIL `_ANEDesc modelWithMILText:` (engine's current bridge) rejects `tensor<int8>` + `constexpr_affine_dequantize` with `InvalidMILProgram` | **E** | `diffusion_ane::tests::test_int8_conv1x1_nanobot_pattern` re-run 2026-04-18 · macOS 26.3.1 · coremlc 3510.2.1 |
| AB6 | `.mlpackage` → `xcrun coremlcompiler compile` → `MLModel(.cpuAndNeuralEngine)` path ACCEPTS the same op chain; `MLComputePlan` lists ANE in `supported_compute_devices` | **E** | `/tmp/higgs_int8_probe/compute_plan.py` · `plan_4b.py` |
| AB7 | CoreML scheduler picks ANE only above a shape threshold: toy 64×64 seq=16 → CPU (cost 0.96); DFlash-4B o_proj 3072×3072 seq=16 → ANE (cost 0.54) | **E** | `/tmp/higgs_int8_probe/plan_4b.py` |
| AB8 | int8 path requires a NEW bridge (`AneKernel::from_mlpackage`), not an edit to `ane_mil.rs`. The two paths coexist — raw-MIL emitter stays for fp16. | D | implication of AB5 + AB6 |
| AB9 | The *other* raw-MIL bridge (`compile_direct` → `_ANEClient compileModel:options:qos:` with `kANEFModelMIL`, documented "full op support") ALSO rejects `tensor<int8>` + `constexpr_affine_dequantize` with the identical `InvalidMILProgram` error. Both raw-MIL entry points fail; the mlpackage path works only because `xcrun coremlcompiler compile` runs an op-lowering pre-pass before producing `.mlmodelc`. | **E** | `diffusion_ane::tests::probe_int8_conv1x1_compile_direct` 2026-04-18 · macOS 26.3.1 · Xcode 26.0.1 · coremlc-MIL 3510.2.1 |
| AB10 | int8-mlpackage decode for DFlash-4B drafter LOSES to fp16 raw-MIL by 1-3 ms even with layer-fusion (Step 3 result): 20 dispatches × 88 µs overhead + 11.1 ms compute (already at 61 GB/s ANE peak) ≈ 12.9 ms vs ~9 ms fp16 baseline. Per-dispatch wrapper savings (hypothetical fallback #3.5) cannot close gap because compute floor is hardware-saturated. | **E** | `dflash_ane_full_layer_int8_probe` + roofline math, see `next-session-int8-e2e-decode.md` |
| AB11 | Shipping verdict: int8 mlpackage is a **prefill-only** story on this drafter. Decode stays on fp16 raw-MIL. Wire int8 into prefill paths that see seq ≥ 32 naturally (non-drafter inference). | D | AB10 + handoff fallback #4 |

## Contradictions Between Sources

| # | Topic | Source A | Source B | Status |
|---|---|---|---|---|
| X1 | ANE for prefill | roofline.md says "recommended: add `HIGGS_TARGET_ANE_PREFILL=1`" | `model_loader.rs:180` warns `HIGGS_TARGET_ANE_GDN=1` regresses prefill 3× | Roofline proposes a DIFFERENT flag not implemented; GDN path regresses; roofline's proposal never tested |
| X2 | MLX-rs version | `mlx_rs_capabilities.md:1` says "0.25.3" | `Cargo.toml:83-84` pins git rev `af21d79` | Doc is likely stale — verify rev maps to which tag |
| X3 | ANE multi-threading value | User intuition "multi-threading would help" | `qwen3_next_ane_worker.rs:6-7`+gdn-verify:218: single physical ANE, kernels are `!Send` | Hardware serialization — multi-threading unlikely to help; needs verification of CoreML API |
| X4 | "ANE helps prefill for dense" | User memory (prior) | `model_loader.rs:180` warning, gdn-verify.md D6 evidence | GDN path regresses. Is there a NON-GDN dense path that helps? Unknown — handoff narratives not yet read |

## Open Questions / Needs Verification

| # | Question | Method to resolve |
|---|---|---|
| Q1 | Does `HIGGS_TARGET_ANE_LM_HEAD=1` help prefill? | Run bench with flag on/off |
| Q2 | Does DFlash ANE draft (`dflash_ane.rs`) help prefill vs GPU-only DFlash? | Check existing benchmark logs |
| Q3 | Is MLX-rs rev `af21d79` current? Has `gather_mm` / `SwitchLinear` / custom Metal landed upstream? | Git log of oxideai/mlx-rs |
| Q4 | Can CoreML dispatch truly parallelize across `MLModel` instances on M4 ANE? | Apple docs + empirical test |
| Q5 | Python 3.14 coremltools status as of 2026-04 | Check coremltools release notes |
| Q6 | Was `HIGGS_TARGET_ANE_PREFILL` proposal ever prototyped / branched? | git log --all --oneline | grep -i prefill |
| Q7 | What was the 4B scaling story the user mentioned? | Read handoff narratives (skipped for context) |
