# Clover cold-prefill memory bounds

Date: 2026-11-13. Base commit: `a73e40dd8b6b04626f58c1983c5fef83ec388c87`.
This is a review candidate only. No Higgs binary or user configuration was
installed or restarted.

## Incident path and scope

The 28,832-token `ExactBootstrap` request enters
`generate_continued_self_spec_locked` without a retained continuation. Its
request-local radix match covers only about 1,358 tokens, leaving roughly
27.5k tokens for stock prefill. The throughput profile submits 1,024 rows per
chunk. After stock prefill, `retain_self_spec_prompt_state` deep-clones the
full prompt checkpoint while the live request cache remains resident. At this
geometry a 32,768-token Clover checkpoint is about 703 MiB, before weights,
allocator slack, and transient activations.

This evidence supports transient chunk width and duplicate live cache state as
material peak contributors. It does not prove that either one alone caused the
uncaught Metal exception, and the full production OOM was deliberately not
reproduced.

## Candidate

- Prefills at the 16K boundary cap a configured 1,024-row chunk at 512. The
  trigger starts one 32-token radix block early because hybrid body snapshots
  floor their split to that block size.
- When MLX reports less than 2 GiB between active bytes and its memory limit,
  the chunk is capped at 256 and the allocator cache is cleared.
- Low-headroom DFlash prefills force chunked execution below the normal
  chunking threshold even when the configured width is already 256 or 128.
- Prompts are not rejected or truncated. Cache identity and stock row
  semantics are unchanged; only execution width and pressure reclamation
  change.
- `HIGGS_DIAG_SESSION_TIMING=1` additionally reports the already-sampled MLX
  prefill/decode high-water values.

## Safe fixture evidence

Three fresh isolated servers loaded the same four-layer Clover fixture and
processed the same cold 18,040-token self-spec request. The only experimental
variable was the compile-time long-prefill cap. MLX's request sampler reset the
peak counter immediately before each prefill.

| Chunk rows | Active before | Active after | Prefill peak | Delta vs 1,024 |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 2,466,097,364 B | 2,598,593,236 B | 3,665,458,216 B | baseline |
| 512 | 2,466,080,980 B | 2,598,576,852 B | 3,267,093,560 B | -398,364,656 B (-10.9%, about 380 MiB) |
| 256 | 2,466,080,980 B | 2,598,576,852 B | 2,930,258,944 B | -735,199,272 B (-20.1%, about 701 MiB) |

All three runs reported 18,040 prompt tokens, four completion tokens, identical
output (`ouatecumulative`), and the same 18,016-token radix boundary on the
second request. Single-run TTFT was 5.81 s / 5.08 s / 5.07 s respectively;
that is recorded only as a gross-regression check, not a performance claim.

As a numerical invariant check, each artifact was normalized to the request
label, usage counts, output, progress counts, and cache-route deltas. The
1,024-, 512-, and 256-row normalizations compare byte-for-byte equal and have
the same SHA-256:
`0dc7c9f45101c7256e0bfcc8f5237e6c03ab146f7f876e6c63bc2480a1f4dee0`.

Request artifacts:

- `/private/tmp/higgs-prefill-memory/memory-1024.json`
- `/private/tmp/higgs-prefill-memory/memory-512.json`
- `/private/tmp/higgs-prefill-memory/memory-256.json`

Each server used `/private/tmp/higgs-prefill-memory/config.toml` and the request
command below, changing only the output filename and comparison binary:

```sh
HIGGS_CONFIG_DIR=/private/tmp/higgs-prefill-memory/state \
HIGGS_DIAG_SESSION_TIMING=1 RUST_LOG=info \
/private/tmp/higgs-ttft-binaries/memory-512 serve \
  -c /private/tmp/higgs-prefill-memory/config.toml --mlx-profile throughput

python3 crates/higgs-engine/examples/ttft_cache_routes.py \
  --base-url http://127.0.0.1:19002 \
  --output /private/tmp/higgs-prefill-memory/memory-512.json \
  --prefix-words 18000 --turns 0 --case-key memory-peak-18000
```

## Verification

The pure policy test was written first and observed failing before the helper
was implemented.

```sh
CARGO_TARGET_DIR=/private/tmp/higgs-ttft-target \
cargo test --offline --release -p higgs-engine \
  low_headroom_bounds_prefill_chunks_without_rejecting_the_prompt --lib
```

Result: 1 passed.

```sh
HIGGS_CLOVER_FIXTURE_DIR=/Users/peppi/AI-Models/clover-fixture \
CARGO_TARGET_DIR=/private/tmp/higgs-ttft-target \
cargo test --offline --release -p higgs-engine self_spec_cache_tests:: \
  --lib -- --ignored --test-threads=1
```

Result: 3 passed, including stock radix equivalence, shared-preamble reuse, and
exact retained-session continuation.

The final source built offline as `/private/tmp/higgs-ttft-target/release/higgs`
(SHA-256
`a046a791c6cc0aed06b31e67c630d1bb623bac8d5e097a9d784c7822d6984299`).
Its adjacent `mlx.metallib` has SHA-256
`96967765c23048d5330d69b76ccf0873c900a049909234b609fc56cb2aa068bc`.
Neither artifact was installed.

GitNexus classified every edited execution path as CRITICAL. Impact analyses
were run before edits for `self_spec_stock_prefill`, `self_spec_forward`,
`dflash_prefill_raw_with_taps`, `store_preamble_checkpoint`, and
`RequestMlxExecution::finish_phase`. This candidate therefore requires normal
review and a production-shaped canary before any deployment. The measured
reduction is fixture evidence, not proof that the production OOM is
eliminated.

## Local installation

After focused policy/cache tests, final offline release build, complete graph analysis and independent review, installed `/Users/peppi/.local/bin/higgs` atomically. Package version remains `1.8.0`; this is the local modified build, identified by SHA-256 `a046a791c6cc0aed06b31e67c630d1bb623bac8d5e097a9d784c7822d6984299`. Rollback executable: `/Users/peppi/.local/bin/higgs.before-prefill-memory-1794586661` (old SHA `aa99f17f30a3b71e6808e9eff8c5abfc7a2162ad63679119ed09c98856f801fc`). The existing Metal library is byte-identical to the build's and was not replaced. Binary hash and `--version` were verified after installation. No full40-layer OOM reproduction or claim of absolute memory safety was made.
