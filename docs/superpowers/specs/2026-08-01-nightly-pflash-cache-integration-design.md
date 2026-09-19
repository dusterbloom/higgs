# Nightly and PFlash Cache Integration Design

## Objective

Merge the complete history of `feat/pflash-bonsai-q2-dspark-cache` into
`nightly` while preserving the NanBeige4.2 support introduced by
`f92694b76`. The result must retain the branch history, all cache fixes, and
the working NanBeige model path.

## Scope

The source scope is the 49 commits on
`feat/pflash-bonsai-q2-dspark-cache` that are not in `nightly`, measured from
their shared base `8be7d7445`. Unrelated divergent repository branches are
out of scope.

The integration includes:

- Cache-resident session and radix-cache changes.
- Paired target/dSpark cache lifecycle changes.
- PFlash and Bonsai Q2 cache routing.
- Streaming cancellation and session-boundary fixes.
- Hybrid GDN/KV paging and boundary fixes.
- Cache accounting, metrics, request controls, and ChatML reconciliation.
- Every supporting model, route, configuration, test, and benchmark change
  carried by the source branch.
- Existing NanBeige loader, registry, transformer, cache-layer, configuration,
  documentation, and smoke-test support from `nightly`.

## Integration Strategy

Create `integration/nightly-pflash-cache` from `nightly` and merge
`feat/pflash-bonsai-q2-dspark-cache` with a non-fast-forward merge. This
preserves the source commit graph and makes the integration boundary explicit.

Resolve conflicts semantically:

1. Use the pflash branch as the baseline for cache, session, radix, PFlash,
   dSpark, metrics, and request-routing behavior.
2. Port NanBeige-specific changes into the resulting model and configuration
   code rather than selecting either conflicting file wholesale.
3. Preserve unrelated `nightly` content when the pflash side only deleted it
   as part of the acknowledged in-flight commit `9cdd5e5ed`.
4. Retain source-side documentation updates when they describe integrated
   behavior, but do not accept broad deletion of existing documentation as a
   cache feature.
5. Do not change public behavior beyond the union of the two branches unless a
   minimal compatibility fix is required to compile or preserve an invariant.

## Conflict-Sensitive Areas

The expected high-risk files are:

- `crates/higgs-models/src/transformer.rs`: preserve pflash changes and port
  NanBeige shared-weight loops, direct quantization, validation, and tests.
- `crates/higgs-models/src/lib.rs`: preserve new cache/model dispatch while
  keeping NanBeige logical cache-layer construction and non-batched behavior.
- `crates/higgs/src/config.rs`: retain pflash cache/request configuration and
  NanBeige batch exclusion plus model defaults.
- `crates/higgs/src/state.rs`, `doctor.rs`, and route modules: retain the new
  session/cache surface while preserving NanBeige-compatible model handling.
- Documentation and smoke scripts: combine behavior descriptions and keep the
  NanBeige smoke opt-in.

## Correctness Invariants

- `nanbeige` remains registered and loadable through the generic transformer
  loader.
- NanBeige allocates `num_hidden_layers * num_loops` logical KV-cache layers.
- NanBeige remains excluded from batched decode.
- Dense NanBeige KV caches remain compatible with normal radix-prefix reuse.
- Qwen3.6 hybrid caches use the source branch's exact-boundary and paged-hybrid
  logic.
- Session retention never publishes a cache under tokens it has not evaluated.
- Generated assistant turns and literal ChatML text do not cause false session
  boundaries.
- Explicit cache bypass, cache metrics, and accurate cached-token accounting
  remain available.
- Model unload continues to free model-owned caches and clear the MLX allocator.

## Verification

Verification proceeds from narrow to broad:

1. Confirm the merge graph contains both `f92694b76` and `e64369170`.
2. Run formatting checks.
3. Compile the workspace and all relevant test targets.
4. Run NanBeige registry, configuration, cache-layer, and transformer tests.
5. Run cache ownership, radix, session continuation, hybrid reuse, concurrent
   session, and dSpark cache tests.
6. Run the broadest practical workspace test suite, recording any hardware-only
   or ignored tests separately.
7. Inspect the final diff against both parents for lost NanBeige support,
   accidental documentation deletion, or unresolved conflict markers.

The integration is complete only when the branch contains both histories,
compiles, passes the applicable automated tests, and has no unintended broad
deletions.

## Delivery

The work remains on `integration/nightly-pflash-cache` for review. The existing
`nightly` ref is not moved until the integration result is reviewed and the
user explicitly chooses to update it.
