# Batched independent cache copies

2026-11-14. Worktree: perf/ttft-cache-costs. Installed binary SHA-256: 79dcf3e1fec06fde7f2070e9a3c3fac8c0a8193ce18c379f3022d02340bb292d.

## Cause and change

Native sampling caught the incoming retained-cache fork in AnyCache::try_deep_clone -> try_eval_deep_clone -> MLX eval -> GPU event wait. The same serial copy helper is used for outgoing retained snapshots. Clover has about80 independently copied buffers; the old whole-cache path waited after each copy.

The whole-cache clone now builds independent fresh-allocation copy graphs, then evaluates all targets together under one MLX gate. Per-layer standalone clone APIs are unchanged. KV, recurrent, TurboQuant and MLA metadata are preserved. Snapshot phase diagnostics separately measure copying, validation/compression, final evaluation and publication.

## Measured comparison

Same M4/32GiB, full40-layer Clover plus MiniCPM, fresh isolated port19001 servers, same5500-word prompt and162-token follow-up suffixes. Four requests per binary: normal seed, normal continuation, background continuation, normal continuation. Outputs capped at4tokens. taskpolicy background/normal comparison is a deliberate stress test, not proof of the historical scheduling trigger. Normal policy restored after each run.

| Operation | Serial A | Batched B |
| --- | ---: | ---: |
| Normal continuation snapshot copy |1303.36ms|17.50ms|
| Background continuation snapshot copy |10408.91ms|31.10ms|
| Following normal snapshot copy |590.25ms|16.07ms|
| Background continuation TTFT |24.411s|2.897s|

Background TTFT improved8.43x in this single A/B pair. This is not a general8.43x model-speed claim. Cold TTFT remained substantial (30.08s vs24.60s); other scheduling, prefill and client-contract costs remain.

## Verification and limits

Release server build passed. Standalone cache_snapshot_check passed: strided KV/GDN numerical identity, distinct array handles, live-update isolation, sparse/empty layers, MLA, TurboQuant exact packed-bit equality before/after mutation, off-gate clone. Static independent review found no correctness blocker after strengthening TQ bit comparisons. git diff --check passed. GitNexus worktree change analysis:29symbols,23affected processes,3tracked files,risk critical; no partial/truncated flag reported. This includes prior simple.rs changes. New standalone example reviewed manually.

Full higgs-models unit-test compilation is blocked by83existing branch API errors (including AsSliceError::NotContiguous and Qwen test constructor mismatches); do not claim the full suite passed. Failed build recorded at /private/tmp/higgs-batch-copy-red.log. The attempted test-only evaluation counter was removed rather than shipped.

Huihui+MiniCPM restored on port9000 using candidate, normal scheduling. User config unchanged; startup config still controls future default model choice. Installed CLI binary atomically with rollback recorded in batched-cache-copies/install.json. Changes prepared on perf/ttft-cache-costs for commit and push.

## Project release build

`CARGO_TARGET_DIR=/Users/peppi/Dev/higgs/target cargo build --release -p higgs` passed (optimized, 8m40s). `target/release/higgs --version` reports higgs 1.8.0; Mach-O arm64. SHA-256: 4ffbc6d5e64b3866009cd607637e42fafeb71106404f22c4c9efabaff768ac67. The freshly built mlx.metallib is copied beside the executable because the CLI build script ran before native MLX compilation completed.
