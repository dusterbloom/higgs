# Corrected whole-model baseline — 2026-09-13

Both long-prompt controls completed with **zero new system swapouts**, normal
sampled capacity pressure, no cached prompt tokens, and correct retrieval of
`LANTERN-4729-QZ` and `Neri`. Each generated 19 tokens with a 32-token budget.

| API prompt tokens | Time to first content token | Whole request | Peak sampled physical footprint |
|---:|---:|---:|---:|
| 32,005 | 327.289 s | 328.937 s | 15.724 GiB |
| 45,001 | 418.301 s | 420.483 s | 16.228 GiB |

These are one valid observation per length, on battery power, not medians or a
speedup claim. The server did not emit final prefill duration through SSE, so
`prefill_ms` remains null. TTFT includes work through the first content event;
it must not be relabeled isolated prefill time. Whole-request time includes
completion and final stream handling. Existing swap usage was nonzero; **the
change in cumulative swapouts during each accepted request was zero**.

## Corrected setup

The earlier isolated config omitted `[local] raise_wired_limit = true`, which
this user's regular Higgs config enables. `LocalConfig` defaults that option to
false. Omitting it skipped both the MLX wired-limit setup and Higgs's 256 MiB
allocator-cache cap. The previous failed control therefore did not match the
user's existing allocator policy; it is not evidence that 32K cannot fit.

The corrected isolated config includes:

```toml
[local]
raise_wired_limit = true
```

Startup logs verify `wired_limit_mb=25559 cache_limit_mb=256`, with the previous
allocator cache limit reported as 31,129 MiB. This is an existing opt-in memory
policy, applied only to the benchmark config. No Rust code, user config, model
weights, chunk size or inference math was changed.

The binary remained SHA256
`8c4c9fa407e1b89964d80accb434f5d1b7cd442d74618e13d28bc287af3640a0`.
It is the same release build as the initial screen, from `1904bc3` plus the
preserved preexisting user edits. Current benchmark commit `b28225aa0` changed
no serving code. Source-diff identity and checkpoint/tokenizer hashes remain in
[the original manifest](results/2026-09-13/baseline-manifest.json) and
[checkpoint shard hashes](results/2026-09-13/checkpoint-shards.json).

Mac16,1, base M4, 32 GiB, Darwin 27.0.0. Power observations reported battery
power at 100%, falling to 99% at the end of the accepted 45K request. No other model, ANE probe, indexer or
build ran concurrently. Ordinary desktop applications remained running; none
were closed. GPU clock/thermal drift and sampler overhead were not quantified.

## Procedure and qualification

Each length used a new private localhost server on port 19093, explicit config,
throughput profile, T1024 prefill chunks, dense KV, disabled durable prefix cache,
and a 49,152-token context limit. Exact request bytes match the initial screen.
Temperature was zero and thinking was disabled. No draft model was configured;
the model loader reported disabling MTP because the checkpoint has no MTP weights.
A 521-token request warmed the process before each timed long request. This is
cold **prompt cache**, not cold OS page cache; it is not a full-length untimed
warmup. First-use costs for other shapes can remain in the measurement.

The 32K control completed on the first corrected attempt. The first corrected
45K attempt was automatically stopped after new system swapouts, at 123.150 s
and 16,384 last-reported processed tokens. Its 9,246-page delta is about
144.47 MiB at the measured 16 KiB page size. Pressure still reported normal;
normal pressure alone is therefore not a sufficient qualification rule. This
failed attempt remains in the evidence and was excluded by the predeclared
zero-new-swap requirement, not its speed.

The 45K retry used the same binary, config and request, with 30 seconds of stable
swap counters before timing (instead of 10). It completed without new swapouts.
No device clocks, system settings, or unrelated processes were changed to force
that result. The changing background state means these runs do not isolate a
causal performance delta from the previous attempt.

A separate monitor sampled `proc_pid_rusage(RUSAGE_INFO_V2).ri_phys_footprint`
and system VM counters about every two seconds, along with the private server's
capacity telemetry. There were 163 samples for 32K and 209 for the accepted 45K.
Every sampled pressure was normal and every sampled swapout counter matched
its request's initial counter. Samples are not guaranteed to catch an
instantaneous footprint peak. RSS was also retained, separately, and was not
used as a substitute for physical footprint. The task's monitor force-stops
only its owned server when the swap gate fails; this does not test cooperative
request cancellation.

## Reproduction and evidence

[Raw evidence](results/2026-09-13-wired-baseline/) contains:

- `summary.json`: accepted measurements and their qualification fields.
- `attempt-1/`: accepted 32K, failed 45K, both warmups, config and startup logs.
- `attempt-2/`: accepted 45K retry, warmup, config and startup logs.
- In each attempt: exact runner source (`runner.py.txt`), binary/config hashes,
  requests, SSE events, sampled RSS, and physical-footprint/capacity timelines.
- `attempt-1/memory-sampler-source.py.txt`: exact read-only macOS sampler reused
  from the local nanobot context-recovery experiments.

Large repetitive raw files are gzip-compressed with deterministic headers.
Token IDs and prompt text are unchanged from `results/2026-09-13/`.
The runner sources record the exact tmux-launched procedure and local imports.
For another machine, use its paths/unused port and preserve the measurement
policy explicitly; do not run this beside an existing inference workload.

Validation checked successful terminal outcome, expected anchors, zero cached
prompt tokens, zero new swapouts in endpoint and interval samples, normal
pressure in every sample, request SHA256 identity, agreement with final SSE
usage, and physical-footprint maxima against the raw timelines. The owned
servers exited after their runs. Preexisting user edits were preserved.

This establishes a usable whole-request control at both target lengths. It does
not establish ANE speedup, native KernelDMA attribution, zero-copy integration,
GPU/ANE overlap, or a repeated regression distribution.
