# Session 27 — Bonsai-Q1 scaffolding gap isolated (kernel exonerated)

Date: 2026-04-25
Branch: `feat/magic-canvas`
Head at start: `a012c847 feat(bonsai-q1): B1 steps 1+2 — dynamic rope offset + KV pre-alloc`
Head at end:   (uncommitted; see "Diff")

## TL;DR

Decided to stop guessing and isolate the qmm kernel before touching code.
Two benches written + run. Result re-frames everything:

| Bonsai-8B build | ms/step | tok/s | what's included |
|---|---:|---:|---|
| qmm-only (stripped) | **2.14** | 467 | only the 7 qmm/layer + lm_head |
| mlx-lm Python (full) | 14.0 | 71.5 | full forward (PrismML reference) |
| higgs Rust (full)    | 44.5 | 22.5 | full forward |

**The kernel is already fast. The gap is entirely in scaffolding (rope,
rms_norm, sdpa, KV cache, residual adds).** Rust scaffolding costs
~42 ms/step on 8B. Python's costs ~12 ms. **3.5× heavier**, and that
3.5× IS the entire 3.2× decode gap.

## What was eliminated

- ❌ **MLX version skew** (suspect #2). Per-shape qmm bench: Rust mlx-rs
  (mlx 0.31.1+1bit cherry-picks) vs Python mlx-py 0.31.2.dev are within
  ±5% on every shape (table in `.planning/measurements/bonsai-parity/qmm-isolated-results.md`).
- ❌ **Unfused QKV / gate-up composition** (suspect #3). Verified
  `~/Dev/diffusion_bonsai/.venv/lib/python3.11/site-packages/mlx_lm/models/qwen3.py:44-46, 95-97` —
  mlx-lm has separate q/k/v and gate/up/down. Same 7 qmm/layer as us.
- ❌ **Per-call FFI overhead in mlx-rs** (suspect #1). qmm-only stripped
  decode dispatches 252 qmm/step on 8B in 2.14 ms — the binding is fine,
  MLX lazy graph pipelining works perfectly through Rust.
- ❌ **embed swap** (the original ask). mlx-rs `QuantizedEmbedding`
  uses `IndexOp::index(&x)` which lowers to the same `mlx_take_axis` C
  call as our `take_axis(&flat, 0)`. Identical op tree. (Not benched
  because the analytical answer was unambiguous, and embed is 0.2% of
  decode anyway.)

## What's left (the actual lever)

The ~30 ms/step Python-Rust gap on 8B (and ~7 ms on 1.7B) is somewhere
in these 5 scaffolding ops, in this order of suspicion:

1. **rope** — `apply_yarn_rope` in `crates/higgs-models/src/yarn.rs`,
   2 calls/layer × 36 = 72 calls/step. We allocate a fresh
   `Array::from_int(offset)` per call (`bonsai_q1.rs:515`); Python
   passes the offset directly. May cause CPU↔GPU stream crossings.
2. **KV cache update** — `SteppingKeyValueCache::update_dense` in
   `cache.rs:560-712`. Even though session-25 added pre-allocation,
   the steady-state path still does `slice_update` ops every step.
3. **rms_norm** — 4 calls/layer × 36 = 144/step. Should be fast-path
   (`mlx_rs::fast::rms_norm`) but worth confirming.
4. **scaled_dot_product_attention** — 1/layer × 36 = 36/step. Already
   uses `mlx_rs::fast::cached_scaled_dot_product_attention`.
5. **residual adds** — 2/layer × 36 = 72/step. Profiled at 180 μs/call
   in session-23 (mostly dispatch). Probably fine.

## Recommended next experiment (1 session)

Bisection bench. Modify `forward_trunk_free` (or write `forward_stripped_a`,
`forward_stripped_b` etc.) progressively re-enabling components:

| Variant | What's included | Expected ms/step on 8B |
|---|---|---:|
| stripped (already exists) | qmm only | 2.14 |
| +rope | + 2 rope/layer | ? |
| +rope+norms | + 4 rms_norm/layer | ? |
| +rope+norms+sdpa | + cached_sdpa | ? |
| +rope+norms+sdpa+cache | + KV update_dense | ? |
| +everything (= current) | + 2 residual/layer | 44.5 |

Each delta tells you how expensive that component is. The biggest jump
is the lever. Then compare *that* component to mlx-lm's equivalent
(probably has a Python micro-bench or inspect with MLX_TRACE).

Hot prediction: rope alone adds ≥10 ms/step on 8B. The
`Array::from_int(offset)` per call is the smell. Replacing it with a
pre-allocated rolling Array (or switching `apply_yarn_rope` to
`fast::rope_dynamic` correctly — session-25 did the call but maybe
still re-allocates the offset arg) should be ~free in code change.

## Files added this session (uncommitted)

```
A  crates/higgs-models/tests/qmm_shapes_bench.rs        (kernel parity bench)
A  crates/higgs-models/tests/qmm_only_decode.rs         (stripped-decode bench)
A  .planning/measurements/bonsai-parity/qmm_shapes_bench.py  (Python mirror)
A  .planning/measurements/bonsai-parity/qmm-isolated-results.md
A  .planning/RECAP-2026-04-25-session27-bonsai-scaffolding-gap-isolated.md
```

No production code touched. Both benches gated `#[ignore]`, run with
`cargo test --release -p higgs-models --test <name> <fn> -- --ignored --nocapture`.

## How to reproduce

```bash
# Rust kernel parity
cargo test --release -p higgs-models --test qmm_shapes_bench \
  qmm_shapes_isolated -- --ignored --nocapture

# Python kernel parity (PrismML venv)
source ~/Dev/diffusion_bonsai/.venv/bin/activate
python .planning/measurements/bonsai-parity/qmm_shapes_bench.py

# Stripped decode
cargo test --release -p higgs-models --test qmm_only_decode \
  qmm_only_decode_bench -- --ignored --nocapture
```

## Numbers to keep in mind for next session

- **8B target: 14 ms/step** (matches mlx-lm Python). Currently 44.5.
- **1.7B target: ~4.5 ms/step** (extrapolated, not measured). Currently 11.5.
- Stripped floors: 2.14 ms (8B), 1.66 ms (1.7B). Real target sits between
  those floors and Python's full-forward number.

## Things I did NOT do (next session knows)

- Did NOT verify whether `apply_yarn_rope` actually uses `rope_dynamic`
  or still bakes offset as compile-time constant. Session-25 said it
  was switched; needs eyes-on confirmation in `crates/higgs-models/src/yarn.rs`.
- Did NOT diff the actual op count per step between higgs and mlx-lm.
  The qmm-isolated bench was enough to exonerate the kernels; the next
  level (which non-qmm ops we issue extra of) needs MLX_TRACE-style
  instrumentation.
- Did NOT touch the live `forward_trunk_free`. Stripped variant is in
  `tests/qmm_only_decode.rs` only.
- Did NOT commit. All changes are pure additions; safe to commit as one
  bench-and-measurements bundle.

## Suggested commit message (if next session approves)

```
bench(bonsai-q1): kernel exonerated — gap is in scaffolding, not qmm

Adds tests/qmm_shapes_bench.rs (per-shape Rust kernel bench),
.planning/measurements/bonsai-parity/qmm_shapes_bench.py (Python
mirror on PrismML mlx 0.31.2.dev), and tests/qmm_only_decode.rs
(stripped-decode bench: 7 qmm/layer + lm_head only, no rope/norm/
sdpa/cache/residual).

Findings:
- qmm kernel times are within ±5% Rust vs Python at every Bonsai
  decode shape. MLX 0.31.1+cherry-picks performs identically to
  mlx-py 0.31.2.dev. Suspect "MLX version skew" eliminated.
- mlx-lm's Qwen3 has the same 7-unfused-qmm-per-layer composition.
  Suspect "fused QKV/gate-up" eliminated.
- Stripped decode hits 2.14 ms/step on 8B (467 tok/s). Per-call
  FFI in the binding is not the bottleneck.

Conclusion: the 44.5→14 ms/step gap vs mlx-lm Python is entirely
in scaffolding (rope, rms_norm, SDPA, KV update, residual).
Next session bisects which component dominates.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

## Open questions for next session

1. Does `apply_yarn_rope` use `fast::rope_dynamic` correctly, or still
   re-allocate the offset Array per call? `crates/higgs-models/src/yarn.rs`
   is where to look.
2. Is `SteppingKeyValueCache::update_dense` doing a `slice_update` op
   per step that could be replaced with a single in-place write?
3. What does mlx-lm's KVCache.update_and_fetch actually emit as MLX
   ops? (`~/Dev/diffusion_bonsai/.venv/lib/python3.11/site-packages/mlx_lm/models/cache.py`).
   Direct comparison may reveal a missing trick.
