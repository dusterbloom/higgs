# Exact FP32/D256 attention experiment

**Goal:** Continue the authorized prefill investigation with measured exact
attention candidates after scalar sparse attention and GDN launch tuning failed.
**Architecture:** Keep existing serving defaults and checkpoint unchanged while
comparing native dense query-tile schedules and a specialization of the existing
MLX Steel fused kernel. Use full FP32 Q/K/V and float accumulation, causal GQA,
positive offsets, valid-length tails and actual sampled QKV activations.
**Stack:** Rust/MLX native oracle, Python MLX custom Metal kernel, NumPy references.

## Constraints

- Existing user edits and untracked probes are read-only inputs.
- Graph first; impact before edits, UNKNOWN requires text corroboration.
- Serialize GPU, build and indexing. No agent may use hardware until granted.
- First test correctness with nonconstant random inputs and boundary/tail cases.
- Exact means no deliberate precision reduction, pruning or approximate softmax;
  normal FP32 reassociation is tested numerically, not called bitwise parity.
- Compare fresh evaluated outputs in alternating order, preserve raw samples,
  actual launch parameters, source/library provenance, power and swap counters.
- Reject new timed swapouts and do not promote gains inside control drift.
- A microbenchmark gain needs native integration and matched serving validation
  before claiming whole-request acceleration. Failed candidates remain evidence.

## Tasks

- [x] Native dense tile screen: create a standalone Rust example for FP32
  Q[1,16,1024,256], KV[1,2,K,256] and lower-right causal mask at K=8192/16384.
  Compare query tiles64/128/256/512 and whole-query native SDPA. Keep QKV and
  Boolean masks preallocated; evaluate each tile then concatenate. Verify outputs
  against the128-row control. Rotate balanced order and record individual
  samples, not just medians. Own file examples/prefill_dense_tuning.rs.
- [x] Steel D256 candidate: create benchmarks/fp32_attention/ with Python wrapper
  extracting the pinned local MLX Steel header implementation, preserving Apple
  license/provenance. Specialize legal FP32 tiles such as BQ8/BK16/one SIMDgroup
  and BQ16/BK8/two SIMDgroups; exact FP32 only. Avoid touching vendor sources or
  existing untracked probes. Compare with the dense128 baseline and full SDPA;
  include offset causality, partial tiles, GQA mapping, noncontiguous source views,
  adversarial finite logits and sampled real traces. Actual hardware launch must
  agree with variant labels. Record copy/materialization costs explicitly.
- [x] Review both harnesses before hardware conclusions. Run serialized CPU/GPU
  correctness, then paired screen at8K/16K. If a kernel passes and wins clearly,
  integrate only an opt-in native experiment and validate whole-model behavior
  and latency; otherwise document why it fails and the next bounded revision.
- [x] Preserve results and failure provenance, complete independent review and
  graph checks, and make a scoped commit. No unsupported end-to-end speedup.

The user requested continuation after the prior report recommended exact tiled
FP32/D256 attention. This plan executes that direction without another approval
round. The lower-cost dense schedule and reused Steel implementation are tested
before a new TensorOps/FlashAttention implementation; existing TensorOps probes
are unfinished user work and will not be modified.


## Outcome

Completed two native dense sweeps and four paired Steel timing runs, with independent
review before and after hardware. The reviewed Q-register revision allowed BQ32/64
within 20 KiB shared memory, but every fused candidate remained slower. Therefore
the conditional native serving integration was not triggered. Native full-query
SDPA showed only exploratory 5–8% operator latency reductions with eight times the
nominal score matrix storage. See `benchmarks/fp32_attention/RESULTS.md` for evidence
and the next profiling questions. No serving defaults changed.
