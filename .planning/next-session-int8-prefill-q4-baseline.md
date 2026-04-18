# Next session — int8 prefill needs MLX q4 baseline before wiring

**Source:** session 2026-04-18 (continuation of `next-session-int8-e2e-decode.md`).

## TL;DR

Decode int8 dead (AB9/AB10/AB11). Prefill probe at Qwen3-9B gate shape shows
ANE int8 beats MLX **f32** matmul 2.30×. But f32 isn't production — MLX q4
(group_size=64) is. Need q4 number before committing to the multi-session
prefill wiring.

## What landed this session

- **AB9** (commit `5d159425`): probe_int8_conv1x1_compile_direct proves
  `_ANEClient compileModel:` ALSO rejects int8 with InvalidMILProgram.
  Both raw-MIL bridges fail; mlpackage path works only via coremlc lowering
  pre-pass. Decode int8 confirmed dead.
- **Prefill probe** (commit `5720b35e`):
  `ane_mlmodel::tests::qwen3_9b_mlp_int8_vs_mlx_probe`. Synthetic gate_proj
  at Carnice-9B shape (12288×4096) seq=128:
  - ANE int8 mlpackage: 2.08 ms min, 24.2 GB/s int8
  - MLX f32 matmul:     4.80 ms min
  - Speedup vs f32:     **2.30×**, parity max_diff=0.0203

## The gap

ANE int8 = 24.2 GB/s. ANE peak ≈ 50–60 GB/s at this shape class (per AB
measurements). seq=128 already on the compute cliff for `gate@9728×3072`
(per existing `dflash_ane_mlp_chain` data). Either:
- The 12288×4096 shape is in the same compute-cliff regime and we're not at
  ANE peak — there may be headroom from picking a different seq bucket.
- Or seq=128 is just past the bandwidth-bound ceiling.

## Step 1 — extend the probe with MLX q4 baseline

Edit `qwen3_9b_mlp_int8_vs_mlx_probe` to add a third bench arm:
```rust
use mlx_rs::quantization::{QuantizedMatmul, quantize};
let (wq, scales, biases) = quantize(&w, group_size=64, bits=4)?;
// time: quantized_matmul(x, wq, scales, biases, transpose=true, group_size=64, bits=4)
```
Report q4_min_ms and `speedup_q4 = q4_min_ms / ane_min_ms`. **Decision rule:**
- speedup_q4 > 1.5×: GREEN — wire layer 0 MLP next session, scale plan after.
- 1.0–1.5×: YELLOW — marginal; consider whether engineering cost is worth it.
- < 1.0×: RED — prefill int8 also dead. Drop and pursue alternatives
  (LM head LUT6 expansion, hybrid GPU+ANE prefill split, etc.).

## Step 2 — if GREEN, also probe seq sweep

Run probe at seq ∈ {32, 64, 128, 256, 512}. Find the sweet spot. If ANE wins
big at seq=64 but loses at seq=512, we wire only for seq buckets where ANE
wins; long prefills stay on MLX.

## Step 3 — if GREEN, sketch the wiring

Pattern after `prepare_lm_head_weights` + `lm_head_ane` field on
`Qwen3NextCausalLM`:
- Add `gate_proj_ane`/`up_proj_ane`/`down_proj_ane` Option<Arc<AneMlPackageKernel>>
  on `Qwen3NextMLP` (or `Qwen3NextDecoderLayer`).
- New `prepare_ane_int8_mlp_layer0(model, seq_buckets)` builds 3 mlpackages,
  caches to `~/.cache/higgs/int8_mlpkgs/{model_hash}/{layer}/{proj}.mlmodelc`,
  loads kernels.
- New env flag `HIGGS_TARGET_ANE_INT8_MLP=1` (or `=0,4,8` for layer subset).
- `Qwen3NextMLP::forward`: if kernel set + seq matches bucket + prefill phase,
  dispatch ANE; else MLX. Convert fp16↔f32 at boundary.
- Doctor validation, README, parity test.

Realistic budget: 3 sessions for layer 0 only, +2 sessions to scale to all
32 layers with disk caching + invalidation. Bench is final session.

## What to NOT do

- Don't wire ANE for decode (AB10 — provably loses).
- Don't extend AB5/AB9 probes; the raw-MIL int8 question is fully closed.
- Don't pursue fallback #3.5 (mlmodelc-via-_ANEClient bridge); analysis
  shows it cannot beat fp16 raw-MIL even with optimistic dispatch savings.

## Toolchain & files

- macOS 26.3.1, Xcode 26.0.1, coremltools 9.0 (3.13 sidecar at
  `/tmp/higgs_int8_probe/.venv`)
- Probe: `crates/higgs-models/src/ane_mlmodel.rs::tests::qwen3_9b_mlp_int8_vs_mlx_probe`
- AB9 negative evidence: `crates/higgs-models/src/diffusion_ane.rs::tests::probe_int8_conv1x1_compile_direct`
- CLAIMS log: `.planning/research/ane-truth/CLAIMS.md` AB9–AB11
- Quant script: `crates/higgs-models/scripts/quantize_int8_proj.py`
