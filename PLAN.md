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

## Expected outcome

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
