# Session 18 — BonsaiQ1 packed loader landed + mlx-sys swap identified

Date: 2026-04-24
Branch: feat/magic-canvas
Base commit: `1cee9bd6` (session 17 close)

## TL;DR

P1 of the session-17 plan landed with **architectural simplifications that
collapse 3 original phases into a config edit**:

1. **P1 green**: `crates/higgs-models/src/bonsai_q1.rs` — `BonsaiQ1Engine`
   holds weights in MLX's packed `Q1_0_g128` affine layout (1.25 bpw). 8B
   loads at **1220.7 MB** (vs ~32 GB fp32 dequant). All 3 smoke tests pass.
2. **P2+P3 recap-original are obsolete**. mlx-rs already exposes
   `quantize`/`quantized_matmul` that's bits-parametric. The PrismML MLX
   fork enables `bits=1` on the upstream dispatch. No custom Metal kernel,
   no CPU reference needed — **the entire P2-P3 block collapses to a
   `CMakeLists.txt` one-line edit**.
3. Remaining architectural work (causal/incremental KV + spec-decode
   wiring + E2E bench) is unchanged.

## Session 17 directional decisions (user-confirmed)

1. **New parallel `BonsaiQ1Engine` type** (not an enum retrofit inside
   `DiffusionEngine`). Rationale: `AneBonsaiEngine::new_causal` (the
   working drafter path in `simple.rs:576`) feeds fp32 weights into ANE
   compilation before `drop_blas_layers()`. Packing those in place would
   break the drafter.
2. **Metal-first, skip CPU reference.** The recap's P2 (CPU fused 1-bit
   matmul) was a correctness oracle for the Metal kernel, not a perf
   target. Validate the Metal path against PrismML Python MLX on a fixed
   prompt.
3. **Drafter stays on ANE.** 1.7B at fp16-baked ANE is ~3 GB resident —
   fine. Only the 8B target needs packed storage + inline dequant.

## P1 — BonsaiQ1Engine packed loader (completed)

**Files:**
- `crates/higgs-models/src/bonsai_q1.rs` (new, 410 lines)
- `crates/higgs-models/src/lib.rs` (one-line module registration)

**Types:**
- `PackedQ1Linear { w_packed: Vec<u32>, scales: Vec<f16>, biases: Vec<f16>,
  out_features, in_features }` — matches MLX's safetensors layout
  (`weight: uint32[out, in/32]`, `scales/biases: fp16[out, in/128]`).
- `BonsaiQ1LayerWeights` — q/k/v/o/gate/up/down packed + q_norm/k_norm/
  input_norm/post_attn_norm fp16.
- `BonsaiQ1Config` — captures `tie_word_embeddings`, `rms_norm_eps`,
  `rope_yarn_factor`, `rope_original_max_seq` (Bonsai-8B has YARN
  factor=4.0, original=16384; 1.7B does not).
- `BonsaiQ1Engine` — layers + packed embed + optional packed lm_head
  (None for tied 1.7B; Some for untied 8B) + fp16 final_norm.

**Method `dequant_row_to_fp32`** is a reference oracle for P2 validation.
Not on the hot path; matches `diffusion::dequant_q1_g128` bit-for-bit.

**Smoke tests (all passing):**
- `test_load_bonsai_1_7b_packed` — 28L/hidden=2048/heads=16/8/inter=6144,
  tied embed, 256.5 MB resident, loads in ~2s.
- `test_load_bonsai_8b_packed` — 36L/hidden=4096/heads=32/8/inter=12288,
  untied lm_head, YARN factor=4.0, 1220.7 MB resident, loads in ~9.5s.
- `test_packed_row_matches_reference_dequant` — row 0 of layer 0 q_proj
  dequants with `max_err=0` vs existing `diffusion::dequant_q1_g128`.

**Run:**
```bash
cargo test -p higgs-models --lib bonsai_q1:: -- --nocapture --test-threads=1
```

## Major pivot: P2+P3 collapse to an mlx-sys vendor swap

### What we found

1. `mlx-rs/mlx-rs/src/ops/quantization.rs` already has
   `quantize_device(w, group_size, bits)`, `quantized_matmul_device(
   x, w, scales, biases, transpose, group_size, bits)`, and
   `dequantize_device(...)`. All three are **bits-parametric** — the
   Rust side is bits-agnostic. See mlx-rs commit `deaa56c4`.

2. PrismML's installed MLX Python at
   `~/Dev/diffusion_bonsai/.venv/lib/python3.11/site-packages/mlx/`
   accepts `bits=1` through the standard API. Verified end-to-end:
   ```python
   wq, s, b = mx.quantize(w, group_size=128, bits=1)
   # wq.shape=(4096, 128) dtype=uint32, s/b.shape=(4096, 32) dtype=float32
   y = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=128, bits=1)
   # Works.
   ```

3. **higgs's vendored MLX rejects bits=1.** `mlx-sys/src/mlx-c/
   CMakeLists.txt:37-38` pins `ml-explore/mlx v0.31.1`, which at
   `mlx/ops.cpp:4591` has:
   ```cpp
   if (bits < 2 || bits > 8 || bits == 7) {
     throw "[quantize] ... supported bits are 2, 3, 4, 5, 6 and 8.";
   }
   ```

### The edit (P2, new scope)

Swap `mlx-sys` vendor in `/Users/peppi/Dev/mlx-rs/mlx-sys/src/mlx-c/CMakeLists.txt`:

```diff
   FetchContent_Declare(
-    mlx
-    GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
-    GIT_TAG v0.31.1)
+    mlx
+    GIT_REPOSITORY "https://github.com/PrismML-Eng/mlx.git"
+    GIT_TAG 1bit-affine-quantization)  # commit b194cb9 (Apr 1, 2026)
```

Then rebuild (`cargo clean -p mlx-sys && cargo build -p higgs-models`).
~10 min cold compile.

### Validation gate for P2

Write `bonsai_q1::tests::test_mlx_quantized_matmul_bits1_matches_oracle`:

```rust
// Pick a small layer (say 1.7B layer 0 k_proj: out=1024, in=2048)
let layer = &engine.layers[0];
let k = &layer.k_proj;

// Build MLX arrays from the packed bytes (no dequant — feed bits=1 directly).
let w_mlx  = Array::from_slice(&k.w_packed, &[k.out_features as i32, (k.in_features/32) as i32]);
let s_f32: Vec<f32> = k.scales.iter().map(|f| f.to_f32()).collect();
let b_f32: Vec<f32> = k.biases.iter().map(|f| f.to_f32()).collect();
let s_mlx = Array::from_slice(&s_f32, &[k.out_features as i32, (k.in_features/128) as i32]);
let b_mlx = Array::from_slice(&b_f32, &[k.out_features as i32, (k.in_features/128) as i32]);

// x: arbitrary fp32 input [1, in_features]
let x = Array::from_slice(&x_f32, &[1, k.in_features as i32]);
let y_mlx = quantized_matmul(&x, &w_mlx, &s_mlx, Some(&b_mlx),
    /*transpose*/ true, 128, 1)?;

// Oracle: dequant_row_to_fp32 per row, then scalar dot with x.
let mut y_ref = vec![0.0f32; k.out_features];
let mut w_row = vec![0.0f32; k.in_features];
for row in 0..k.out_features {
    k.dequant_row_to_fp32(row, &mut w_row);
    y_ref[row] = w_row.iter().zip(x_f32.iter()).map(|(a, b)| a * b).sum();
}

// Compare to fp16 epsilon.
let y_mlx_vec: Vec<f32> = y_mlx.to_vec::<f32>();
let max_err = y_ref.iter().zip(y_mlx_vec.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f32, f32::max);
assert!(max_err < 1e-2, "max_err={max_err}");
```

If this passes, P2 is done and we have a production-ready 1-bit matmul.

### Risks to watch during the swap

1. **PrismML's mlx-c surface**: the C FFI layer (`mlx-sys/src/mlx-c/`)
   may need to match PrismML's C++ signatures. Check if PrismML's
   `quantize` has a 6th arg (`global_scale`) — our mlx-rs already
   passes one (`mlx_sys::mlx_array { ctx: null }`) so likely fine.
2. **API drift v0.31.1 → PrismML HEAD**: if PrismML is forked off
   newer MLX, other op signatures may have changed. Watch for compile
   errors in `mlx-sys/src/mlx-c/*.cpp` binding code.
3. **Sibling repo modification**: `/Users/peppi/Dev/mlx-rs` is a
   separately checked-out repo. Changes affect any other project using
   it. Either (a) branch-switch mlx-rs cleanly, or (b) pin higgs's
   `Cargo.toml` at an mlx-rs fork with the vendor change baked in.

### Why NOT Option B from the original recap (vendor `.metal` source)

The recap's "Option B" was "vendor PrismML's patches into mlx-sys". That's
what we're doing — just at the CMake fetch level instead of manual patch
application. Far cleaner.

## Revised phase map

| Phase | Original scope | Revised scope | Status |
|-------|----------------|---------------|--------|
| P1 | Packed weights in DiffusionEngine | **New BonsaiQ1Engine type** with packed storage | ✅ Completed |
| P2 | CPU fused 1-bit matmul (reference) | **Swap mlx-sys vendor to PrismML fork** | 🎯 Next |
| P3 | Metal fused 1-bit matmul kernel | Folded into P2 | — |
| P4 | Incremental + causal forward + KV | **Unchanged** — use mlx-rs `quantized_matmul` on the hot path, wire KV cache, causal mask | Pending |
| P5 | AnyModel::BonsaiQ1 target variant | **Simplified** — thin adapter in `simple.rs` (deferred full AnyModel integration; AnyModel is MLX-array-typed which now aligns naturally since BonsaiQ1Engine uses mlx-rs Arrays internally) | Pending |
| P6 | E2E bench 1.7B drafts → 8B | Unchanged | Pending |

The AnyModel question (open #1 from session 17) **resolves itself** once
we use mlx-rs `quantized_matmul` for the hot path: the engine naturally
works in `mlx_rs::Array` space, so `AnyModel::BonsaiQ1(BonsaiQ1Engine)`
slots in without the copy-tensor seam the session-17 recap worried about.

## Files to read first next session (cold start)

1. **This file**
2. `crates/higgs-models/src/bonsai_q1.rs` — the landed P1 engine
3. `/Users/peppi/Dev/mlx-rs/mlx-sys/src/mlx-c/CMakeLists.txt:35-38` —
   the swap target
4. `/Users/peppi/Dev/mlx-rs/mlx-rs/src/ops/quantization.rs:48-164` —
   the bits-parametric Rust API
5. `target/release/build/mlx-sys-*/out/build/_deps/mlx-src/mlx/ops.cpp:4591` —
   the current rejection (confirms the blocker)

## Reproduction (P1 green state)

```bash
cd /Users/peppi/Dev/higgs
cargo test -p higgs-models --lib bonsai_q1:: -- --nocapture --test-threads=1
# Expected: 3 passed. 1.7B=256.5MB, 8B=1220.7MB, oracle max_err=0.
```

## Commit landed this session

`feat(bonsai-q1): packed 1.25-bpw engine (1220 MB for 8B vs 32 GB fp32)`

Includes: new module `bonsai_q1.rs`, lib.rs registration, 3 passing tests.
Does NOT include: mlx-sys swap (requires a fresh context window due to
rebuild time).

## TaskList state

| # | Subject | Status |
|---|---|---|
| 1 | P1: BonsaiQ1Engine packed loader | ✅ completed |
| 2 | P2: Swap mlx-sys to PrismML fork for bits=1 | ⏳ pending (next) |
| 3 | P3: Incremental + causal forward with KV cache | ⏳ pending |
| 4 | P4: Spec-decode target wiring | ⏳ pending |
| 5 | P5: E2E bench 1.7B drafts → 8B verifies | ⏳ pending |

## Open questions for next session

- Should we branch `/Users/peppi/Dev/mlx-rs` cleanly (e.g., `prism-1bit`
  branch) before the CMake edit, so other projects aren't affected?
- Is PrismML's `1bit-affine-quantization` branch still at `b194cb9`, or
  has it moved? Check with `git ls-remote
  https://github.com/PrismML-Eng/mlx.git 1bit-affine-quantization`
  before editing.
- The 27B sparsity side-task (task #9 in session 17) is still open.
  Lower priority than the q1 target unblock.
