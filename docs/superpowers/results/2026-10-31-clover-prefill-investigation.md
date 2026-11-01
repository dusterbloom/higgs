# Clover prefill investigation (M4 base, 32 GB)

Date: 2026-10-31. Branch `feat/clover-dllm` @ 401c3ce37. Investigation only:
no tracked source edited, nothing committed. Scratch bench:
`crates/higgs-models/examples/prefill_ops_bench.rs` (untracked); HTTP harness:
`$CLAUDE_JOB_DIR/tmp/ttft_bench.py`.

Model: `kapitsaprime/Clover-35B-A3B-Preview-MLX-4bit` (40 layers: 30 GDN + 10
full-attention, hidden 2048, 256 experts x top-8, I=512, GDN Hk16/Hv32/D128,
FA 16q/2kv heads x D256, 4-bit g64 affine; router 8-bit). Served by
`higgs serve --mlx-profile throughput` (chunked prefill threshold 1024, chunk
1024, `speculation = "self_spec"` with the DFlash2 drafter). MLX v0.31.1.

## 1. Headline: end-to-end prefill is ~200 tok/s, flat in prompt length

Measured against the live server over HTTP (unique nonce per prompt, streaming
`max_tokens=1`, `prompt_tokens` from `usage`; TTFT includes one decode step
which the 0.25 s full-hit repeat bounds from above). `cached_tokens` was never
reported (`None`), i.e. no prefix hits.

| prompt tokens | TTFT (s) | prefill tok/s |
|---:|---:|---:|
| 56-60 (x3) | 0.88 / 1.44 / 1.89 | 30-68 (fixed-overhead dominated, noisy) |
| 290 (x3, same length) | 1.66 / 1.69 / 1.37 | 171-211 |
| 336 | 2.08 | 162 |
| 686 (single pass, < chunk threshold) | 3.50 | 196 |
| 1438 | 6.98 | 206 |
| 2875 | 14.19 | 203 |
| 11549 | 56.78 | 203 |
| 16374 (after the 22:42 UTC restart, PID 19273, decode-only change) | 82.53 | 198 |
| 2875 repeated verbatim (prefix-cache hit) | 0.25 | - |

Shared-preamble probe (same 2.1k-token system prompt, two conversations with
different user messages): turn 0 2108 tokens 9.60 s (220 tok/s); turn 1
`cached_tokens=None`, i.e. **no reuse of the preamble across conversations**
(its 28.6 s TTFT is confounded by two concurrent deployment smoke requests in
`metrics.jsonl` at 22:43:00/22:43:16; the engine serializes requests, so the
2038-token re-check at 22:43:10 (14.97 s) is contended too and excluded).

Real-world corroboration from `~/.config/higgs/logs/metrics.jsonl` today
(current build, PID 13535 started 22:17 UTC): cold new conversations of
2274 / 2250 / 3596 tokens -> 12.6 / 13.8 / 16.9 s TTFT (163-212 tok/s);
11.3k-11.8k-token cold prompts earlier today -> 51-57 s (208-221 tok/s).
The user's reported 14-18 s cold TTFT on a few-thousand-token prompt is
exactly this: **~4.9-5.0 ms per prompt token, independent of length** (198
tok/s at 16k vs 203 at 2.9k: the super-linear attention term is only ~3% up
to 16k, so attention is not the bottleneck yet).

Two hypotheses tested and rejected on the way:

- "Per-length Metal JIT": the canonical conv kernel specializes `T` as a
  template arg (`configure_canonical_conv_kernel`, qwen3_next.rs:2324) so every
  new tail length is a new pipeline. Measured same-length A/B (290/290) vs new
  length (291): 1.66 / 1.69 / 1.42 s -> no per-length penalty visible.
- "Chunking overhead": 686 tokens (single pass) and 2875 (3 chunks) give the
  same 196-203 tok/s. Chunk boundaries and eval barriers are not the cost.

## 2. Where the time goes (kernel-level, synthetic weights at real shapes)

All numbers from `prefill_ops_bench` on this machine while the server held
~20 GB (GPU otherwise idle). Dense bf16 GEMM peak measured first: **3.5-3.65
TFLOPS** (1024x2048x2048 .. 2048^3). That is the compute roofline.

### Per 1024-token chunk, current production kernels

| component | kernel | ms / 1024-token chunk | share of 5.05 s measured (1024 x 4.93 ms) |
|---|---|---:|---:|
| MoE experts (40 layers) | `gather_qmm` unsorted -> **gather_qmv**, 1 row per matvec (`MoeGatherKernel::Qmv` default) | **2857** (71.4 ms/layer, 0.72 TFLOPS) | **58%** |
| Dense projections (qkvz/ba/out x30, q/k/v/o x10, shared expert + router x40) | MLX `quantized_matmul` 4-bit g64 at M=1024 | 912 (3.1-3.3 TFLOPS = 90% of dense peak) | 18% |
| GDN recurrence (30 layers) | fused sequential kernel, grid (32,128,32), 8.5 us/step regardless of T | 260 (8.67 ms/layer) | 5% |
| Full attention (10 layers), first chunk | MLX SDPA, D=256 is **unfused** (2 GEMMs + softmax); 6.4 ms/layer at KV=1k, +6 ms per extra 1k of context | 64 (chunk 1) .. 481 (KV=8k) | 1-10% |
| depthwise conv K=4 (30 layers) | MLX conv1d proxy | 14 | <1% |
| **modeled total** (attention averaged over a 3-chunk prompt, 123 ms) | | **~4.17 s** | 83% |
| unattributed (norms, routing softmax/argpartition/sort, 2x argsort + 2x take per MoE layer, swiglu, output gates, residuals, mask build, per-chunk evals, DFlash `prime_taps` ~20-40 ms est.) | | ~0.9 s | ~17% |

MoE kernel comparison at the same shapes (E=256, I=512, D=2048, top-8, uniform routing):

| routed tokens L | qmv (default) ms/layer | MMA (`HIGGS_MOE_GATHER_KERNEL=mma`) | native (MLX size switch) | MMA speedup |
|---:|---:|---:|---:|---:|
| 512 | 37.9 | 15.2 (1.70 TFLOPS) | 15.2 | 2.5x |
| 1024 | 71.4 | 25.7 (2.01 TFLOPS) | 26.0 | 2.8x |
| 2048 | 141.7 | 45.1 (2.28 TFLOPS) | 44.4 | 3.1x |

Fusing gate|up into one gather (`forward_fused`) changes nothing (71.5 vs
71.4; 25.6 vs 25.7): the cost is the kernel class, not dispatch count. The
all-expert weight byte floor is 3.8 ms/layer, so the MMA kernel is compute
bound too, just at 2x less efficiency than the dense qmm (2.0 vs 3.3 TFLOPS).

This is the same finding commit c93035808 made on the real model ("cold
prefill 230 -> 350-390 tok/s" with `mma`) and then set aside because the
mode is a process-global switch that also hits decode (AR 44 -> 24 tok/s).

GDN recurrence: 8.5 us per sequential step is latency-bound (one simd_sum
chain per step for all 4096 simdgroups). A chunked (FLA-style, chunk 64)
formulation does the same work as ~3 GFLOP of matmuls per layer, i.e. ~1-2 ms
instead of 8.7 ms (estimate, not measured).

Attention: at D=256 the fused steel kernel is not used (D=128 fused is 1.66 ms
vs 6.36 ms for D=256 at the same L/KV; 2x FLOPs would be 3.3 ms). But it still
runs the two GEMMs at 2.7-2.9 TFLOPS, so a fused kernel mainly saves the
masked half of the diagonal block and score materialization. Cost is linear in
context: for a 16k prompt the FA layers sum to ~8 s (10 of today's ~80 s, ~25%
of a post-fix ~30 s).

## 3. Ceiling

Per-token compute floor at the measured 3.3 TFLOPS 4-bit qmm rate:

| | per 1024 tokens | per token |
|---|---:|---:|
| dense projections (measured, already at 90% of peak) | 912 ms | 0.89 ms |
| MoE experts, ideal (51.5 GFLOP/layer at 3.3 TFLOPS) | 625 ms | 0.61 ms |
| MoE experts, best measured kernel (MMA at L=2048 rate) | 900 ms | 0.88 ms |
| GDN chunked (estimate) | ~50 ms | 0.05 ms |
| attention, first 4k of context, avg | ~150 ms | 0.15 ms |
| **floor (ideal kernels)** | **~1.7 s** | **1.7 ms -> ~600 tok/s** |
| **realistic (MMA MoE + chunked GDN + today's residual halved)** | **~2.4 s** | **2.4 ms -> ~430 tok/s** |
| today | 5.05 s | 4.93 ms -> 203 tok/s |

`current x best_case_speedup` = 203 x 3.0 = ~600 tok/s (hard, compute-bound
ceiling); 203 x 2.1 = ~430 tok/s is the realistic engineering target. So a
large win is physically available (2-3x on TTFT), but unlike decode this is
not a bandwidth problem with a 10x headroom: prefill is compute bound on a
3.6 TFLOPS GPU, and the dense half of the FLOPs is already at 90% of peak. The
only ways past ~600 tok/s are doing fewer tokens (prefix reuse) or fewer
FLOPs per token (sparse/compressed prefill, out of scope here).

For the user's scenario (3.6k-token cold prompt, ~1.8k of it a fixed tool
preamble): 16.9 s today -> ~8 s with MMA MoE -> ~5 s with the full kernel
plan -> **~2.5 s if the 1.8k preamble is also served from a warm prefix**.

## 4. Ranked candidates

| # | idea | expected gain (end-to-end) | evidence | effort | risk |
|---|---|---|---|---|---|
| 1 | **MoE gather on MMA kernels for prefill rows** (rows > 8 -> `gather_qmm_rhs`/steel tiles; rows <= 8 keep qmv/moe8) | 203 -> **330-390 tok/s** (1.7-1.9x; TTFT 16.9 s -> ~8-9 s at 3.6k) | measured 71.4 -> 25.7 ms/layer here; c93035808 measured 230 -> 350-390 tok/s on the real model | low: size-dependent policy in `MoeGatherLayout::new` / `moe_gather_kernel`, tests for tails | append-stability: prefill rows (MMA) vs decode rows (qmv/moe8) are not bit-identical. Note the production self_spec path already decodes with fast-rows kernels that are "not bit-identical to qmv" and only stores stock prefill state, so the stored-state contract (stored == cold prefill of the same tokens) still holds as long as every prefill size > 8 uses MMA (c93035808: MMA tiles give identical per-row bits regardless of tile height/slot/batch) |
| 2 | **Warm prefix for the shared preamble** (system prompt + tool defs): store a hybrid checkpoint at the end of the system message / at a configured warm prefix on startup, not only at the conversation-body end | removes ~1.8k of ~2.3-3.6k tokens from every new conversation: TTFT 12-17 s -> 3-9 s today, ~1.5-4 s stacked on #1 | shared-preamble probe: second conversation with an identical 2.1k system prompt got `cached_tokens=None`; metrics: every new 2.2-2.5k-token conversation today paid the full 12-14 s. Code: hybrid store keys only at `stored_clone.resident_len()` (body end floored to 32, `store_prefix_cache_after_prefill`), and `MatchEndpoint::PartialPaged` is never formed on a Hybrid edge (paged_prefix_cache.rs:306) so hybrid state is reusable only at a stored endpoint | medium: extra snapshot (~65 MB GDN state + small KV per checkpoint) at a message boundary, or a `warm_prefixes` config prefilled at startup | memory per checkpoint; matching only if the preamble is byte-identical across conversations (nanobot/nodecode must not embed timestamps/ids in the system prompt) |
| 3 | **Chunked-parallel GDN prefill kernel** (FLA `chunk_gated_delta_rule`-style, chunk 64, matmul form; decode keeps the recurrence) | -210 ms per 1024 tokens: ~5% today, ~9% after #1 (e.g. 330 -> 360 tok/s) | 8.67 ms/layer measured, 8.5 us/step latency-bound; matmul form is ~3 GFLOP/layer (~1-2 ms) [estimate] | high: new Metal kernel + exactness tests vs the recurrence | numerics: prefill state would no longer be bit-identical to the recurrence; f32 state keeps drift small but the project's append-stability tests would need an explicit tolerance |
| 4 | **Larger chunk (2048) once MoE is on MMA** | MMA at 2048 rows is 12% cheaper per row than at 1024 (45.1 vs 2x25.7 ms): ~-250 ms per 2048 tokens, ~5% [measured kernel, end-to-end is a guess] | table above; dense qmm flat vs M | trivial (`chunked_prefill_chunk_size`) | peak memory: unfused attention scores are 16 x 2048 x KV x 2 B (1 GB at KV=16k) on a machine with ~4 GB free; canonical conv config persistence is keyed to 1024 |
| 5 | **Attribute and fuse the ~17% residual** (routing sort/gather, norms, swiglu, output gates, weighted expert sum) | up to ~0.8 s per 1024 tokens if fully removed; realistic 0.3-0.4 s (~8%) [guess until profiled] | modeled vs measured gap; needs `HIGGS_PROFILE=1` on the real model (needs the server stopped, see below) | medium | low |
| 6 | Fused flash-style SDPA for D=256 | <3% below 4k context; ~10-20% only at 16k+ | 6.36 vs 1.66 ms (D=128 fused) per FA layer at KV=1k; linear in context | high (Metal kernel) | numerics vs unfused fp32-softmax path |
| 7 | Overlap CPU graph build with GPU across chunks | ~1% [guess]: per-chunk build is ~2000 lazy ops x a few us (op_overhead_bench) | GPU is compute-saturated during a chunk | low-medium | async eval interactions with cache snapshots |
| - | quantized-matmul kernel choice for the dense M>1 projections | none available: MLX qmm already at 3.1-3.3 of 3.6 TFLOPS | measured | - | - |
| - | fusing gate|up gathers, chunk-boundary tuning, dispatch-count reductions | none: measured 0% (fused) / flat (chunking) | measured | - | - |

Dead ends from prior work, not repeated: tiled/bilinear (Strassen) QMM probe
for Bonsai prefill (`2026-10-07-prefill-bilinear-feasibility.md`: GPU timings
never produced; Strassen was only a control), ternary tiled QMM (0.40-0.45x of
stock), chunk 768-1536 tie, block-sparsity skipping (no empty blocks).
`docs/suffix-cache-reuse.md` (SCR, mid-context edit reuse) addresses a
different workload (edits), not cold first turns.

## 5. Measurements that need the server stopped

1. Real-model per-layer attribution of the residual (item 5):
   `HIGGS_PROFILE=1 RUST_LOG=info ~/.local/bin/higgs serve --mlx-profile throughput`,
   then `python3 $CLAUDE_JOB_DIR/tmp/ttft_bench.py --exact 2048`; read the
   `PROFILE: per-layer avg` (gdn_attn_ms / gdn_mlp_ms / fa_attn_ms /
   fa_mlp_ms at query_tokens=1024) and `PROFILE-MLP` lines.
2. End-to-end A/B of item 1 without code changes (prefill only; decode will be
   slower under the global flag, which is why the real change must be
   size-dependent): `HIGGS_MOE_GATHER_KERNEL=mma ~/.local/bin/higgs serve
   --mlx-profile throughput` + `ttft_bench.py --exact 512 2048 8192`.
   Expected: 203 -> 330-390 tok/s.
3. Full-model chunk timing outside the server: extend
   `crates/higgs-models/examples/clover_pass_bench.rs` with a 1024-token
   `forward_chunked` loop (`cargo rustc --release -p higgs-models --example
   clover_pass_bench -- -C link-arg=<libclang_rt.osx.a>`; see section 6 for
   the link arg).

## 6. Reproduction

```bash
# kernel microbench (synthetic weights at Clover shapes; ~3 GB peak)
RT=$(fd -t f '^libclang_rt.osx.a$' /Applications/Xcode*.app/Contents/Developer/Toolchains | head -1)
cargo rustc --release -p higgs-models --example prefill_ops_bench -- -C link-arg="$RT"
B=target/release/examples/prefill_ops_bench
BENCH=dense ROWS=512,1024,2048 $B
BENCH=moe ROWS=512,1024,2048 MOE_FUSED=1 $B
HIGGS_MOE_GATHER_KERNEL=mma BENCH=moe ROWS=512,1024,2048 $B
HIGGS_MOE_GATHER_KERNEL=native BENCH=moe ROWS=512,1024,2048 $B
BENCH=sdpa KV=1024,2048,4096,8192 $B
BENCH=gdn ROWS=256,512,1024,2048 $B
BENCH=conv ROWS=512,1024,2048 $B

# end-to-end TTFT vs prompt length against the live server
python3 $CLAUDE_JOB_DIR/tmp/ttft_bench.py --exact 512 2048 8192 16384
python3 $CLAUDE_JOB_DIR/tmp/ttft_bench.py --shared-system 2048 2   # preamble reuse probe
```

Limitations: synthetic uniform routing (real routing is skewed; MMA tolerates
skew better than qmv, so the MMA gain may be understated or overstated by a
few percent); kernel numbers were taken with the server resident (20 GB) but
idle; the end-to-end numbers include one decode step (<= 0.25 s) and were
taken on a machine the user was intermittently using.
