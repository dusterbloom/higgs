# Isolated qmm kernel bench — Rust mlx-rs vs Python mlx-py

Date: 2026-04-25. Run on the same Mac, default device (Metal).
Bench: `crates/higgs-models/tests/qmm_shapes_bench.rs` and
`.planning/measurements/bonsai-parity/qmm_shapes_bench.py`.

Protocol: warmup 50, then 1000 iters of `quantized_matmul(x, qw, scales,
biases, transpose=true, group=128, bits=1)` with `eval` per call. fp16 x
shape `[1, 1, K]`, fp16 W shape `[M, K]`. Mean ms/call.

| Shape         | K     | M      | Rust (ms) | Python (ms) | Δ    |
|---------------|------:|-------:|----------:|------------:|-----:|
| 1p7b/q_or_o   | 2048  | 2048   | 0.184     | 0.232       | -21% |
| 1p7b/k_or_v   | 2048  | 1024   | 0.178     | 0.204       | -13% |
| 1p7b/gate_up  | 2048  | 6144   | 0.223     | 0.223       |  0%  |
| 1p7b/down     | 6144  | 2048   | 0.226     | 0.222       | +2%  |
| 1p7b/lm_head  | 2048  | 151669 | 1.381     | 1.379       |  0%  |
| 8b/q_or_o     | 4096  | 4096   | 0.236     | 0.228       | +4%  |
| 8b/k_or_v     | 4096  | 1024   | 0.186     | 0.181       | +2%  |
| 8b/gate_up    | 4096  | 12288  | 0.306     | 0.296       | +3%  |
| 8b/down       | 12288 | 4096   | 0.306     | 0.304       | +1%  |
| 8b/lm_head    | 4096  | 151669 | 1.802     | 1.804       |  0%  |

mlx-py version: 0.31.2.dev20260423+72ec298f. mlx-sys (Rust) is on
v0.31.1 + PrismML 1-bit cherry-picks.

## Conclusion: kernel is identical

Within ±5% noise. **The qmm kernel is not the bottleneck.** Suspect #2
(MLX 0.31.1 vs 0.31.2.dev kernel-selection drift) is eliminated.

## Predicted vs measured ms/step

Sum of sync-per-call kernel times for one decode step on Bonsai-8B:
36 × (2·q_or_o + 2·k_or_v + 2·gate_up + down) + lm_head

| Engine       | Predicted sum | Measured | Overlap |
|--------------|--------------:|---------:|--------:|
| Rust higgs   | 65.2 ms       | 44.5 ms  | 1.47×   |
| Python mlx-lm| 63.5 ms       | 14.0 ms  | 4.54×   |

**Both engines run identical kernels at identical cost. Python harvests
~3× more overlap than Rust from MLX's lazy graph.** That ~3× *is* the
end-to-end gap.

## Where the gap must live

Not in the kernels. Not in the MLX version. The gap is in **how the
graph is built / when it's evaluated** in Rust vs Python.

Candidates, ranked by likely impact:

1. **Per-call FFI overhead in mlx-rs.** Each `ops::quantized_matmul`,
   `ops::reshape`, `ops::add`, `fast::rms_norm`, `fast::rope`,
   `fast::scaled_dot_product_attention` is a C call. Per layer that's
   ~25 FFI calls × 36 layers = 900 calls per step, plus ~8 in
   embed/lm_head. Python pays a similar pybind overhead but may amortize
   differently (GIL release pattern, async dispatch).

2. **Array drop forcing graph nodes to commit.** Each `let x = ...`
   that goes out of scope triggers `mlx_array_free`. In MLX C, this
   decrements an Rc and may finalize the underlying op. If finalization
   forces partial graph evaluation (vs leaving in pending), parallelism
   is lost.

3. ~~**Composition mismatch.**~~ **Eliminated 2026-04-25.** mlx-lm's
   Qwen3 (`mlx_lm/models/qwen3.py:44-46, 95-97`) uses separate
   `q_proj`/`k_proj`/`v_proj` and separate `gate_proj`/`up_proj`/
   `down_proj` — identical 7-qmm-per-layer composition as higgs.
   Fusion is not the answer.

4. **`Array::from_int(offset)` per rope call.** 72 small CPU-stream
   array creations per decode step. If they cause stream-crossing
   syncs, parallelism dies. (`apply_rope` in `bonsai_q1.rs:512`.)

## Recommended next experiments (cheapest first)

A. **Verify hypothesis #3.** Read mlx-lm's Qwen3 forward (`~/Dev/diffusion_bonsai/.venv/lib/python*/site-packages/mlx_lm/models/qwen3.py`).
   If it fuses Q/K/V and/or gate/up: this alone is ~30% off the gap.
   Implement fused-projection variant of `BonsaiQ1GpuLinear` (concat
   weights at load, single qmm at forward, slice). Cost: 1 session.

B. **Add an "everything lazy until end" version of forward_trunk_free**
   that skips any intermediate `eval`/`shape().to_vec()`/`dtype()` not
   strictly needed for op construction. Time it. If it closes the gap,
   we found a sync somewhere. Cost: ~1 day.

C. **MLX_TRACE comparison.** Set `MLX_METAL_DEBUG=1` (or equivalent)
   for both engines, dump the dispatched op sequence per step, diff.
   Will show op-count and op-shape divergence directly. Cost: 1 day if
   the env var works as expected; longer if we need to instrument.

D. **Strip down higgs forward to a single qmm-only loop** (no rope, no
   norm, no SDPA — just 7 qmm per layer × 36) and measure. If that hits
   ~14 ms, the fused-projection hypothesis is fully confirmed. If it's
   still 30+ ms, the issue is per-call dispatch overhead in mlx-rs and
   we'd need to look at batching the FFI itself (or fewer Array
   constructions in the hot path).

Suggest order: A → D → B → C.

## 2026-04-25 update — A eliminated

Verified mlx-lm's Qwen3: separate q/k/v and separate gate/up/down. No
fusion. The remaining suspects (in order of likely impact):

1. mlx-rs Array drop forcing partial graph commit — needs measurement
   via experiment B (lazy-only forward) or instrumentation in mlx-sys.
2. Per-call FFI overhead vs pybind's async dispatch.

Cheapest next move: experiment **D** (qmm-only no-frills loop). If we
can hit ≤14 ms with 7×36 qmm calls on Bonsai-8B in pure Rust, the
binding has no fundamental issue and the gap is in our forward's
non-qmm scaffolding (rope offset arrays, cache update, residual adds).
If we're still ≥30 ms with no scaffolding at all, the binding itself
is serializing graph nodes.
