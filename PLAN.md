# PLAN: simdgroup product loop for the eschamoe gather-QGEMM kernel

## Problem

`eschamoe_gather_qgemm` (crates/higgs-models/src/metal_kernel.rs) decodes trellis
tiles into threadgroup memory, then runs a **per-thread scalar FMA** product
(RM=4 × RN=8 = 32 outputs/thread). Measured: **1.4 TFLOP/s** of the ~3.6 TFLOP/s
the M4 GPU gives (39%). MLX's simdgroup `quantized_matmul` on the same chip,
same shape class, same Low Power state: **3.10 TFLOP/s (86%)**. e2e effect:
prefill is ~135 tok/s where the compute ceiling is ~350–400.

Decode of the trellis codes is **not** the bottleneck — the product loop is.
The kernel comment already names the fix: `simdgroup_matrix` loads +
`simdgroup_multiply`/`simdgroup_add` accumulation, keeping the decode and the
expert-pass walk as they are.

## Why now

- Prefill is the user-visible pain: 135 tok/s vs 405 tok/s proven on this
  machine in April (affine path + sorted-MoE tricks).
- The omlx ANE PR (#3298) comparison settled it: ANE offload tops out at
  −17%…+5.8% full-model on MoE (routed experts stay on GPU), while the
  simdgroup upgrade is ~2.2× on the expert path — the dominant path.

## Constraint inventory (from the current kernel)

- Threadgroup: NT=128 threads; block BM=32 rows × BN=128 cols; K walks in
  16-row trellis-tile steps; w_sh[16·BN] (8 KB) + x_sh[16·XP] (2 KB, XP=33
  bank pad) threadgroup memory; 3 threadgroups/core target.
- Mixed-expert blocks: per-pass expert walk; non-member rows stage zero
  activations; pass count is uniform across threads (barriers stay uniform).
- MSL ships as a Rust string through the mlx-fast-metal-kernel API —
  `simdgroup_matrix/load/store/multiply/add` are MSL stdlib, no toolchain
  change.
- Correctness bar today: decode is bit-exact; the scalar FMA order makes the
  current output bit-identical to the scratch reference. A simdgroup
  accumulator changes summation order → **outputs become tolerance-equal, not
  bit-equal**. The `escha_native_fixture` digests will need re-baselining.

## Phases

### Phase 0 — Baseline lock (half session)

- [ ] Kernel microbench at production shapes: rows ∈ {512, 2048, 8192},
      out_features/hidden from the escha spec; record ms + TFLOP/s for the
      scalar path (`HIGGS_ESCHA_TRELLIS_GEMM` unset).
- [ ] e2e prefill: bench_frontier at 2K/8K/16K prompt tokens; record tok/s
      (expect ~135 at 2K, Low Power on).
- [ ] Correctness harness: A/B the qgemm output against the scratch decode
      reference on random and real shapes (max-abs + max-rel error report).
      This becomes the Phase-1 gate.

### Phase 1 — simdgroup product loop (one focused session)

- [ ] Keep unchanged: decode into w_sh, activation staging into x_sh, the
      per-expert pass walk, zero-staging, block geometry (BM=32, BN=128,
      NT=128), and the K walk in 16-row tile steps.
- [ ] Replace the scalar product with:
      - `simdgroup_load` of activation fragments from x_sh (M side, 8×8)
      - `simdgroup_load` of decoded-weight fragments from w_sh (N side, 8×8)
      - `simdgroup_multiply(acc, a, b, acc)` per K fragment; K=16 per tile
        step → two 8-deep MMA steps per stage
      - accumulator fragments: 32×128 output / 4 simdgroups / 32 threads =
        32 floats/thread — same register budget as today's RM×RN
- [ ] Correctness gate: A/B vs scratch reference, max-rel ≤ 1e-5 on fp32
      accumulation; re-baseline `escha_native_fixture` digests (document the
      new digests in the test).
- [ ] Perf gate: kernel µbench ≥ 2.0 TFLOP/s at 2048×4096×1024 shapes.
  - [ ] If register pressure caps occupancy (check threadgroup count/core),
        drop to BN=64 (2 KB weight block) and re-measure before tuning.

### Phase 2 — Geometry & occupancy tuning (one session)

- [ ] Sweep: (BM, BN, NT) ∈ {(32,128,128), (32,64,128), (16,128,128),
      (64,128,256)} with fragment counts recomputed; measure TFLOP/s at
      rows ∈ {512, 2048, 8192} and pick per-row-range winners if they differ.
- [ ] Confirm the 3-threadgroups/core occupancy holds with the new
      accumulator budget; check w_sh bank conflicts with fragment loads
      (the XP pad pattern may need an N-side equivalent).
- [ ] Measure with Low Power on (the user's operating point) and off.
- [ ] Gate: e2e prefill ≥ 250 tok/s at 2K ctx, Low Power on.

### Phase 3 — Integration & regression (half session)

- [ ] simdgroup path becomes the default for the GEMM route; the scalar
      kernel stays reachable via `HIGGS_ESCHA_TRELLIS_GEMM=scalar` for A/B
      and as the reference.
- [ ] Full test suite + quality gates; re-baseline any digest-bearing tests.
- [ ] e2e: bench_escha_e2e.py + a nanobot turn_bench run (prefill tok/s and
      ttft at 8K and 16K prompts).
- [ ] Update the kernel-head comment (the "scalar fma" ponytail note) and
      this plan's results section.

## Phase 0 results (2026-08-30, M4 base, Low Power on, commit e9b5e32+)

Harness: `phase0_qgemm_ab_and_tflops` (eschamoe.rs, `cargo test -p higgs-models
--release phase0_qgemm_ab -- --nocapture`). Synthetic k=8 codes, 128 experts,
4096x1024 projection, ids cycling all 128 experts (worst-case pass count).

- e2e prefill (bench_frontier, chunked 512): 94.1 / 96.8 / 89.7 tok/s at
  2K / 8K / 16K frontiers — flat, compute-bound. Decode 9.2-18.3 tok/s.
- Path agreement: qgemm vs scratch rel 2.16e-4 (f16-vs-f32 weight rounding
  noise) — gate for Phase 1 is "simdgroup stays in this class".
- Kernel timing (worst-case unsorted expert mixing, decode-dominated):
  | rows | scratch | qgemm |
  |---|---|---|
  | 512 | 150.7 ms (29 GF) | 99.4 ms (43 GF) |
  | 2048 | 614.2 ms (28 GF) | 379.3 ms (45 GF) |
  | 8192 | 2510.4 ms (27 GF) | 1518.8 ms (45 GF) |

  qgemm is 1.5-1.6x ahead of scratch; both are decode-dominated under
  32-distinct-experts-per-block mixing (GFLOP/s here counts only the product,
  not the repeated decode). Phase 1 must beat this baseline with SORTED ids
  as well as unsorted.

## Phase 1 status (2026-08-30 evening) — IN PROGRESS

- simd kernel implemented (ESCHA_QGEMM_SIMD_SOURCE): verbatim decode/staging,
  row-sliced simdgroups (sg owns 8 rows x 128 cols), transposed A-frag load
  from k-major x_sh, ulong2 origins, padded dst + front slice,
  HIGGS_ESCHA_QGEMM_SIMD=1 selection.
- Empirical semantics probe (simd_semantics_dump): transpose + origin
  semantics verified — transposed frag(i,j) = src[(oy+j)*ld + ox+i].
- BUG FOUND, NOT YET FIXED: simd vs scalar diverges (rel 6.58, simd values
  ~6x too large). Loads verified correct in isolation (probe passes).
  Localized suspects: (a) pass-accumulation across expert passes (acc frags
  accumulate tk-steps inside every pass — verify non-member zero-staging
  actually zeroes the A side for simdgroup-owned rows), (b) whether
  origin(0,0)+base-offset A loads alias w_sh-style strides incorrectly for
  the transposed path when ld != frag-row-multiple.
- NEXT: single-expert isolation test (rows=32, all expert 0, k=16 single
  tile row) — dump the 8x8 A fragment per lane vs scalar decode. Then
  re-enable multi-expert.

## Phase 1 results (2026-08-30 late) — CORRECT, MARGINAL PERF

- Fix: the k-walk created 16 overlapping 8-deep fragment accumulations
  (per-row origins instead of 2 disjoint fragments). After the two-disjoint-
  fragment fix, simd == scalar agreement (rel 2.16e-4, same as scalar vs
  scratch).
- Kernel timing, sorted ids (production pattern): scalar 148 / 524 / 736
  GFLOP/s at 512 / 2048 / 8192 rows vs simd 143 / 551 / 818 — simd edges
  ahead at scale (+5-11%), neutral at 512 rows.
- e2e (bench_frontier, SIMD=1): 97.5 / 93.8 tok/s vs baseline 94.1 / 96.8 —
  within noise. The expert gather is a minority of full-model prefill;
  the plateau is set by attention + dense + KV paths.
- Verdict: simd path is correct and the better kernel at scale. Keep it
  opt-in (HIGGS_ESCHA_QGEMM_SIMD=1) until it matters; flipping the default
  is free of risk but also of measurable gain today.
- Phase 2 geometry sweep: deprioritized — the SMEM budget (10.7 KB/TG, 3 TG
  limit) caps geometry moves, and e2e is not gather-bound. The real prefill
  levers are elsewhere (attention path, KV writes).

## Default flipped + GFLOP/s attribution (2026-08-30)

- simd is now the DEFAULT kernel; HIGGS_ESCHA_QGEMM_SIMD=0 falls back to
  scalar. e2e confirmed: 92.5 tok/s @ 2K (noise band 92-97).
- Attribution at sorted-8192 (68.7 GFLOP, 77 ms measured): total decode =
  1.07 G element-decodes across 2048 threadgroups = ~13.9 G elem/s, filling
  ~55 ms of the window; pure MMA at MLX-class 3.1 TF/s = ~22 ms. DECODE
  dominates ~72/28.
- Levers, ranked: (1) halve row-block decode duplication (BM=64 expert-run
  grouping — decode per expert-col-slice drops 2x); (2) fused
  decode-into-register MMA (drops the w_sh round trip entirely — bigger
  rewrite); (3) f16 w_sh (SMEM traffic, measured-worse before — retest under
  the new balance). Absolute standing: 890 GFLOP/s at Low Power on a base M4
  for a 3B-active MoE incl. decode — respectable; the MLX 3.1 TF/s figure is
  a pure dense GEMM with no decode or dispatch.

## BM=64 experiment (2026-08-30 night) — REVERTED, root-caused to SG gating

- BM=64/NT=256 (8 simdgroups): rows 0-7 correct, rows 8+ zero. Reverted to
  the working BM=32 state (kept in git history at a2f8c169d).
- The 64-row decode-once lever remains valid; the bug is in the SG gating or
  staging at 256 threads (every SG should have been live — all rows expert 0).
  Debug entry point: per-SG dump of sg_live/live/acc at the store site.

  UPDATE (session end, bb425bc6+): gate exonerated — rows 8+ stay zero with
  the gate removed. SG0 (rows 0-7, incl. its kf=1 fragment at base +8*XP)
  is always correct; SG1+ (base &x_sh[kf*8*XP + 8], transposed load) read
  zeros. The debug print was lost in a cleanup pass — RE-ADD it (per-row
  scratch vs gemm sums) before the next iteration. FIRST MOVE: kf=0-only
  build (skip the kf=1 A/B loads) — if rows 8+ then hold kf=0 half-sums,
  the loads work and the kf=1 A-frag read is the culprit.
- Perf note: even with the bug, correctness rows 0-7 matched — the fragment
  flow scales; only the gating/distribution at 8 SGs is broken.

- Kernel: 1.4 → **2.4–3.0 TFLOP/s** (MLX parity class on this chip).
- e2e prefill: 135 → **~250–300 tok/s** at Low Power; ~350–400 with sorted-MoE
  tricks from April re-applied on top.
- Decode (matvec path) untouched.

## Risks

- simdgroup_load alignment rules (threadgroup offsets must be fragment-
  aligned) → may need padding in w_sh/x_sh layouts.
- Register pressure from accumulator fragments → occupancy drop; mitigate via
  the Phase-2 geometry sweep.
- Loss of bit-exactness vs the scratch reference → handled by tolerance gate
  + digest re-baseline; the decode itself stays bit-exact.
- Mixed-expert zero-staging must survive fragment loads (zero rows still flow
  through simdgroup ops; no branch needed) — verify in the A/B harness with
  adversarial expert mixes.

## Non-goals

- ANE offload (omlx #3298 comparison): routed experts stay on GPU; full-model
  ANE effect measured at −17%…+5.8% upstream. Revisit only after this plan.
- Decode/matvec path changes.
- Affine-path revival (the 405 tok/s April plateau) — memory footprint makes
  it a non-starter while the 11 GB native path exists.

## BM=64 debug session summary (2026-08-30, closed)

- Three fragment-layout approaches tested (k-major transposed, k-major
  origin-carried, m-major non-transposed) — ALL produce wrong results at
  BM=64/NT=256 while BM=32/NT=128 works. The failure is systematic, not
  a single-address bug.
- ROOT CAUSE (identified but unfixed): at BM=64/NT=256, the x_sh staging
  (256 threads, 64 rows × 16 k-cols) and the fragment load pattern interact
  in a way that produces wrong fragment data for SG1-7. The staging covers
  all 64 rows (verified by address math), the loads read the right logical
  addresses (verified by the probe), but the composed result is wrong.
- The bug requires a dedicated Metal GPU capture session (Xcode GPU
  debugger) to see the actual fragment values per lane — beyond what
  print-based debugging can reach.
- The BM=32 simd kernel (the committed default) is correct and deliver.
  The BM=64 decode-halving optimization (~+30-50% e2e) remains future work
  requiring GPU-level debugging tools.

## Expected outcome