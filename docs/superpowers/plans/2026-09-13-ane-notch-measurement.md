# M4 ANE notch and whole-prefill measurement plan

Authorized by the task dispatched on 2026-09-13. Supersedes the earlier
plan-only restriction, retains its performance/correctness gates.

Goal: decide whether exact graph rewrites and a public Core ML boundary can
justify an Escha whole-model prefill change on Mac16,1. Use existing Higgs
serving and MLX code. Keep benchmark artifacts separate from serving code.

- [x] Inspect the September 7 ANE commits and retained whole-request evidence.
- [x] Recover GitNexus 1.6.12 and index this worktree at 1904bc3 with a 2048 KiB
  file limit; default 512 KiB silently omitted qwen3_next.rs. Index graph has
  unresolved dynamic calls and capped process enumeration; empty results are
  not proof of independence.
- [x] Build this dirty checkout in an isolated cloned target directory. Record
  source diff and binary hash. Do not stage any preexisting changes.
- [ ] Run isolated-server 512/32K/45K baselines with fixed short output. Record
  exact prompts, stream, model/binary identity, power and swap counters. Run
  HIGGS_PROFILE separately for early/late attribution; barrier timings are
  diagnostic, not a sum equivalent to whole-request time.
- [x] Add test-first benchmark-only shape math for exact original, zero-padded
  input (2080), zero-padded output (4160 then trim), and two output projections
  (2048 each then concatenate). Never truncate actual input channels to 2016.
- [x] Export real layer-0 z FP16 weights with preserved math; compare T32 and
  T1024 through public Core ML. Record placement plan, load/warm prediction,
  preallocated output backing identity and output error. A Core ML host time
  cannot establish native KernelDMA notch presence or actual hardware usage.
- [x] Compare realistic removable z cost to whole-model time before integration.
  The old extracted-row GPU microbenchmark is not the production fused-qkvz
  marginal cost. Do not derive a release win from it.
- [ ] If plausible, measure preallocated FP16 layout/transfer/merge and independent
  GPU submission before ANE wait. Account remaining framework-internal copies;
  do not call merely shared memory zero-copy.
- [ ] Integrate only if measured gates pass: >=15% long-request median win at
  32K and 45K, >=5% every uncontaminated pair, decode <=3% regression, short
  <=5%, extra peak <=1 GiB, no new swapouts, same 45K capacity and all quality,
  cancellation, session/cache and fallback gates from the September 6 design.
  Otherwise publish a measured no-go or an explicitly blocked result.
- [ ] Run focused tests and GitNexus detect-changes (correct worktree, no
  partial/truncated result) before a scoped benchmark/evidence commit.

Use public Core ML; no compiler split flags, private runtime, or TD patching.
Runs/builds serialized in tmux. Initial state: battery 52%, no active Higgs,
Python inference or Rust compilation. Exploratory battery results are not
matched AC release evidence. No existing service is stopped or replaced.

## Final gate status

Fresh 521-token control completed. The 32,005-token control was stopped after
new swapouts; 45K and further full-model attribution were deferred. Repeated
public-CoreML workarounds helped T32 but did not consistently help T1024. The
preallocated FP16 probe reused public output backings, while separately measured
CPU packing/merge remained more expensive than prediction. No production path
was integrated; overlap and release qualification remain unmeasured. See
`benchmarks/ane_prefill/RESULTS.md` and its persisted raw evidence.
