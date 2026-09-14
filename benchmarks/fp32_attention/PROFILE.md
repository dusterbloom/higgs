# Metal profile: qreg64k16 register pressure

This is a focused Apple M4 Metal System Trace of the standalone FP32/D256
attention harness at Q=1024, K=16384. The counter-enabled trace used Xcode beta
27.0 and the `Metal GPU Counters` instrument; it ran to process exit 0 in 8.808
seconds. An earlier no-counter trace of the same harness lasted 9.186 seconds.

The most concrete result is compiler spill evidence. The qreg64k16 trace contains
nine `graphics-compiler-spill-events` rows for the Python process. Each reports
1,104 spilled bytes, for 9,936 bytes across the captured submissions. The shader
list names the compiled target as
`custom_kernel_higgs_steel_qreg64k16`. A separate dense128-only control trace at
the same shape contains zero compiler spill-event rows. This establishes that
the q-register revision is spilling on this workload; it does not identify the
number of registers per thread or prove every spilled byte is on the critical
path.

The qreg trace's shader timeline contains 149 intervals associated with the
qreg64k16 shader. Five top-level intervals exceed 100 ms, with durations
168.76, 247.68, 258.68, 265.64 and 253.36 ms. These are several correctness and
timing invocations in one process and are retained as timeline evidence, not a
new paired latency estimate. The timeline exposes execution intervals and their
GPU-active percentages, but those percentages are not occupancy.

Occupancy counters were not available. The counter-enabled trace reports
`counter-profile=3` and shader-profiler mode, but its populated counter metadata
contains only `RT Unit Active`; the GPU counter profile and counter-interval tables
have no rows for this compute workload. Shader-profiler samples are also empty.
Therefore this run cannot support an occupancy percentage, active-warp count or
ALU/bandwidth limiter claim. The correct next action is to obtain a Metal compiler
artifact or Xcode GPU frame capture with a device-supported counter set; latency
alone must not be translated into occupancy.

Barrier cost is similarly bounded but not measured directly. The generated qreg
source contains six `threadgroup_barrier` call sites and seven `simdgroup_barrier`
call sites. These are static synchronization sites, some conditional by loop/path.
Metal System Trace records the qreg work as a shader timeline inside command
buffers and does not emit per-barrier timestamps. It can show command-buffer
serialization, but it cannot attribute the 40–49% slowdown to a particular
barrier. A barrier-removal experiment would change correctness semantics and was
not used as a performance proxy.

The dense control trace also has the same one-counter limitation, but its zero
spill-event result is a useful matched compiler comparison. Both traces were
captured on the same M4 host with the same Xcode/MLX environment; raw trace bundles,
exported tables and the dense driver are archived under
`results/2026-09-14/` with SHA256 metadata. The raw `.trace` bundles remain outside
Git when their native bundle size is not useful for source review.

This measurement changes the next kernel decision: do not increase the cached Q
tile further. The qreg64 design already keeps 64 additional FP32 scalars per lane,
and Metal has now recorded actual spills. Any future exact kernel should reduce
live state or use a different decomposition before trying a larger query tile.

## Live-state reduction and decomposition follow-up

The opt-in `qstream64k16` revision removes the full-D Q register cache while keeping BQ64/BK16 and reloads one 8-column Q MMA fragment per D chunk. It passed all 14 synthetic/layout correctness cases (maximum absolute error `2.70e-6`, maximum row relative L2 `1.46e-6`). In paired runs, median kernel latency was 88.4 ms versus 94.3 ms for dense128 at K=8192, and 190.8 ms versus 184.2 ms at K=16384. The earlier qreg64 comparison was 226.3 ms and 379.6 ms on the same harness invocation, so streaming removed most of the qreg penalty but did not create a large whole-attention speedup over dense128. Noncontiguous view timings tracked the contiguous results and all runs had zero swapout delta.

The qstream64 Metal trace still has nine compiler spill events, but each is 816 bytes (7,344 bytes total), down from qreg64's 1,104 bytes per event (9,936 bytes). The shader is `custom_kernel_higgs_steel_qstream64k16`. Occupancy, active warps, and per-barrier time remain unavailable: the exported shader profiler table is empty and this trace exposes no compute counter intervals.

To test decomposition after reducing live state, qstream16k16 and qstream32k16 use the same streamed Q body with 64 and 128 threads respectively. Both are correct, but slower: at K=8192 their contiguous medians were 142.5 ms and 93.3 ms (qstream64 was 88.4 ms); at K=16384 they were 309.7 ms and 198.6 ms (qstream64 was 190.8 ms). The smaller-row decompositions therefore do not improve this M4 workload. Keep all qstream variants benchmark-only until a serving-shaped trace validates a different decomposition.
