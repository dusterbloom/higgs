# M4 ANE graph-notch screening — 2026-09-13

**Decision: do not integrate a production ANE path from this evidence.** The
public-CoreML graph workarounds help a small tile, but do not consistently help
the T1024 prefill tile. The fresh whole-model 32K control failed the no-swap gate;
45K and hybrid comparisons were therefore not run. No whole-model speedup,
zero-copy MLX path, or GPU/ANE overlap has been demonstrated or shipped.

## Provenance and earlier evidence

Mac16,1 (base M4), 32 GiB, 16-core ANE, Darwin 27.0.0, battery power (50→48%
during full-model work). Source: this worktree at `1904bc3def74de79ac155a7bb4351e171a0eb44b`
plus preexisting user edits. Isolated release binary SHA256:
`8c4c9fa407e1b89964d80accb434f5d1b7cd442d74618e13d28bc287af3640a0`.
No competing model/build was active during hardware runs. Desktop activity and
memory pressure were not controlled. No other user's process was stopped.

Read `4d7db8463` and `cd7b32f76` before implementing the probes. Their isolated
layer-0 result (6.172 ms GPU, 4.602 ms Rust/CoreML boundary) was not whole-model
performance. The later printed `speedup=15.14x` is inconsistent with its labeled
19.531 ms total and comes from misordered format arguments in `ane_z_bench.rs`:
the first positional value is GPU/total but is printed as `ane_predict`, while
the last value is conversion time but is printed as `speedup`. Do not use these
three mislabeled fields as phase evidence. The 19.531 ms total itself remains a
reported benchmark result, not a whole-serving measurement.

Retained September 7 whole-model evidence is stronger than the earlier 467 s
planning datum: the `roofline-20260907/final-default45k/result.json` run reported
45,003 prompt tokens, 290.135 s TTFT, 293.456 s request, 35 output tokens and all
three retrieval anchors. Its separate profile review found full-attention
per-layer cost growing from about 72 to 349 ms across a 32K run. Those historical
runs are context, not matched controls for this checkout or power/memory state.

## Fresh full-model control

Explicit private localhost config, throughput profile, no durable prefix cache,
no MTP sidecar, fixed 32-token output budget, temperature zero, thinking disabled.
Prompt text, native tokenizer IDs, request bytes and raw SSE are retained under
`results/2026-09-13/` (large repetitive payloads use gzip). The model's input
padding/template accounts for the difference between native and API tokens.

| Run | API prompt tokens | TTFT | Request | Outcome |
|---|---:|---:|---:|---|
| Short | 521 | 5.176 s | 6.161 s | Finished, 19 output tokens, both requested facts correct, zero new swapouts |
| Separate-server warmup | 521 | 4.905 s | 6.078 s | Finished, 19 output tokens, zero new swapouts |
| Long control | 32,005 | unavailable | 177.538 s until stop | Incomplete, invalid for latency comparison |
| 45K | not run | — | — | Preceding memory gate failed |

The long control reached 17,408 processed tokens. System swapouts increased by
479,252 pages (16 KiB pages, approximately 7.31 GiB); this is a system delta,
not bytes attributed exclusively to Higgs. The private server's capacity snapshot
reported constrained pressure, 15.76 GB MLX active, 16.51 GB MLX peak. Sampled
RSS peaked at 4.91 GB, demonstrating why RSS must not be called process footprint
on this platform. Physical footprint was not measured. `SIGINT` began graceful
shutdown but left active prefill running; the owned benchmark server was then
force-stopped. The harness recorded an incomplete result and exited nonzero.
This is not a validation of cooperative request cancellation.

No final prefill duration was emitted for the short runs; `prefill_ms` is null.
TTFT must not be relabeled prefill. The long run's progress durations are partial
progress only. There is no valid fresh 32K/45K speed or decode-regression estimate.

## Exact graph variants and public Core ML boundary

Real layer-0 `in_proj_z.weight_int8` and FP16 per-row scales, expanded to dense
FP16. T32 and T1024 synthetic bounded inputs (fixed seed), not a model-quality
corpus. Every public operation plan preferred `MLNeuralEngineComputeDevice` for
conv; this is a preferred placement plan, not a hardware execution counter.
FP16 input, feature provider and output backings are allocated before the timed
loop; returned output data pointers equal the supplied backing pointers in every
run. No per-tile process launch, FP32 output conversion or caller output allocation
occurs in that loop. Core ML internal allocation/transfers remain unobserved.

| Variant | Nominal dense coefficient bytes/core | T32 median (range), ms | T1024 median (range), ms |
|---|---:|---:|---:|
| Original 2048→4096 | 1,048,576 | 0.764 (0.747–0.779) | 1.948 (1.900–1.982) |
| Pad input to 2080, 32 zeros | 1,064,960 | 0.439 (0.435–0.449) | 2.212 (2.112–2.259) |
| Pad output to 4160, trim 64 rows | 1,064,960 | 0.444 (0.442–0.450) | 1.990 (1.972–2.004) |
| Two 2048-output convs, concatenate | 524,288 each | 0.440 (0.436–0.464) | 1.976 (1.926–2.001) |

Three serialized blocks per tile: original / pad input / pad output / split /
original. Each entry summarizes run medians, each from 13 predictions following
three warmups; original has six run medians. Initial screening runs are also
retained, separately, and were slower at T1024 (2.93–3.62 ms), so drift is real.
No timing samples were deleted. Model compile/load time is separately recorded
in every raw native result, outside prediction. CPU-only original medians were
0.323 ms at T32 and 12.045 ms at T1024; CPU results are operator controls, not a
production fallback validation.

All 30 repeated ANE graph outputs were bitwise identical to the original FP16
output after external trim/concatenation. Against FP32 matmul of the same FP16
weights, relative RMSE was 0.00112 and maximum absolute error about 0.00113.
This does not establish equivalence with Higgs's additional Q8(g64) repack or
model-level quality. No recurrent weights, experts, attention or cache state
were changed.

These results support a small-tile graph effect consistent with the linked
[KernelDMA note](https://gist.github.com/Anemll/39f657dc48b402747bdd96458edd415f),
not a measured native KernelDMA diagnosis on base M4. The T1024 tile does not
benefit consistently. No invented ANEC split flags or TD/vault patching were used.

## Boundary and marginal-cost limitations

Separate NumPy/preallocated-host measurements for T1024 original: 1.778 ms input
FP16 transpose/pack, 0.114 ms channel-major memcpy, 2.756 ms output transpose/merge.
These operations alone total about 4.65 ms, exceeding the roughly 1.95 ms public
prediction. They are not concurrent with prediction and do not include an MLX
producer fence, Core ML internal copies, MLX adoption or GPU merge. Do not sum
independent medians into a claimed end-to-end speedup. Keeping channel-major
output for a GPU consumer remains a possible future experiment, not implemented
zero-copy support. GPU/ANE overlap and synchronization were not measured.

`gpu_projection.py` mirrors source Q8(g64) conversion and fused row order, with
weights prepared before timing. Python MLX 0.30.6 T1024 medians: full fused QKVZ
52.007 ms, QKV alone 37.301 ms, isolated z 17.158 ms. This run followed the memory
incident; its timings are retained as low-confidence diagnostics. It uses a
separate Python MLX runtime and synthetic input, not the current Rust serving
seam. In particular, neither 17.158 ms nor the 14.706 ms difference establishes
removable whole-model cost. The old gather-row microbenchmark is not that cost
either. A production integration cannot be justified from these numbers.

## Verification and next gate

- `cargo build --release -p higgs --bin higgs`: passed, exact dirty checkout.
- `python3 -m unittest discover -s benchmarks/ane_prefill -p 'test_*.py'`: six
  tests passed, following observed red failures for missing behavior.
- `clang -O3 -Wall -Wextra -Werror -fobjc-arc -framework Foundation -framework
  CoreML benchmarks/ane_prefill/probe.m ...`: passed.
- `verify_shapes.py` passed both fixtures; 30 repeated output comparisons passed.
- Native Core ML CPU-only and CPU+ANE configurations ran. Non-Apple server builds
  were not run; the probes are standalone and change no Rust/build configuration.
- Git diff whitespace check passed. GitNexus results are recorded with the commit.

Before further integration work, restore enough memory headroom for a zero-swap
32K/45K GPU baseline, then measure fused-QKVZ marginal cost in the exact Rust
serving path. Only if the 15% whole-request gate remains plausible should a
preallocated FP16 GPU-consumable boundary and actual overlap be built. Cache,
session, capacity, cancellation and decode qualification remain mandatory. This
commit is measurement tooling and a no-integration decision, not a speedup release.

Graph checks: the full dirty checkout is critical (91 symbols, 133 affected
flows), including preexisting serving edits. The scoped benchmark stage is low
(29 symbols, zero affected flows). Both checks were non-partial and non-truncated;
the preexisting tracked diff remained byte-for-byte unchanged before staging.
