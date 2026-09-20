# Retained-Byte Fast Session Contract

**Status:** Approved 2026-09-20

## Objective

Make an interactive Nanobot session structurally unable to fall from retained
continuation into an unannounced large cold prefill. Users configure retained
memory in bytes. Higgs derives the model-specific prompt-token envelope that
can be retained within that memory, including target, draft, recurrent, output,
and safety headroom. Nanobot compacts before crossing that envelope. Higgs is
the final atomic authority and rejects unsafe exact continuations before doing
model work or mutating cache state.

## User Guarantees

1. A successful exact-continuation response publishes an exact reusable
   successor cache.
2. Required continuation never silently becomes a stateless cold prefill.
3. A request that cannot remain retained is rejected before prefill and decode
   with a typed `retention_compaction_required` error.
4. Nanobot automatically compacts, rotates the physical session, and retries
   once with bounded context.
5. One retained-byte budget produces model-specific real token limits. Models
   with different KV layouts behave through the same protocol.
6. Invalid, unknown, stale, or contradictory safety contracts fail closed for
   retained mode while ordinary stateless OpenAI-compatible requests remain
   available.

## Single Authority

The user-facing retention control is bytes:

```toml
kv_max_retained_bytes = 4294967296
```

`kv_max_session_tokens` is not an independent policy authority. During the
compatibility period it may remain accepted as a hard upper bound, but the
effective fast-session limit is always derived by Higgs and can only become
smaller. New setup UX persists bytes after resolving a desired-token request.

The logical model context remains a separate architectural ceiling. It never
claims that a prompt is cheaply retainable.

## Ownership

### Higgs

- Inspect the loaded model and all retained components.
- Account target, draft/DFlash, recurrent, auxiliary, response growth, and
  conservative transient headroom.
- Derive one `guaranteed_fast_prompt_tokens` value from the byte budget.
- Construct only internally consistent contracts.
- Perform pre-mutation hard admission for required continuations.
- Preserve the prior retained state on rejection.
- Return retention receipts on successful retained responses.
- Own model artifact inspection and retention planning.

### Nanobot

- Render and estimate the actual provider prompt including tool definitions.
- Use the server's absolute fast-prompt boundaries for LCM scheduling.
- Start background compaction at the soft boundary.
- Block and compact before the hard boundary.
- Preserve SQLite history and summary recovery pointers.
- Rotate session epoch, release the retired Higgs session, and retry once.
- Never reproduce KV geometry or bytes-per-token formulas.

## Protocol V2

The safety contract is versioned independently from ordinary model listing.
It is keyed by `boot_id`, `generation`, and `model_fingerprint`.

```json
{
  "schemaVersion": 2,
  "contractRevision": "<boot>:<generation>:<fingerprint>",
  "model": "ternary-bonsai2-27b-2bit",
  "maxContextTokens": 65536,
  "maxOutputTokens": 4096,
  "retainedBudgetBytes": 4294967296,
  "guaranteedFastPromptTokens": 24576,
  "softCompactionPromptTokens": 16384,
  "targetAfterCompactionTokens": 4096,
  "guaranteedSessions": 1
}
```

Required relationships:

- all numeric limits are positive;
- `guaranteed_fast_prompt_tokens + max_output_tokens <= max_context_tokens`;
- `target_after_compaction_tokens < soft_compaction_prompt_tokens`;
- `soft_compaction_prompt_tokens < guaranteed_fast_prompt_tokens`;
- the guaranteed prompt includes full reserved output growth and safety
  headroom;
- paired target/draft layouts are costed as one indivisible allocation;
- finite retained bytes can never produce an unlimited token guarantee.

Construction uses validated domain types. Serialization cannot expose a raw,
contradictory collection of independently assigned integers.

## Continuation Request

Retained behavior is tagged explicitly:

```json
{
  "retention": {
    "mode": "required",
    "sessionId": 123,
    "epoch": 7,
    "contractRevision": "<boot>:<generation>:<fingerprint>"
  }
}
```

The compatibility adapter may translate Nanobot's internal marker into these
fields, but the resulting request has one mode: `stateless` or
`continue_exact`. A required continuation cannot enter a best-effort cold path.

Typed terminal outcomes are:

- `retained_exact`;
- `retention_compaction_required`;
- `retained_session_unavailable`;
- `stale_retention_contract`.

`retention_compaction_required` and stale/unavailable errors occur before
prefill, decode, or retained-cache mutation.

## Retention Receipt

Successful retained responses return enough evidence to audit the guarantee:

```json
{
  "outcome": "retained_exact",
  "sessionId": 123,
  "epoch": 7,
  "retainedTokens": 18000,
  "retainedBytes": 2690416640,
  "contractRevision": "<boot>:<generation>:<fingerprint>"
}
```

Detailed target/draft/auxiliary byte breakdown remains available through Higgs
metrics and planning diagnostics. Nanobot does not use it for policy.

## Main Control Loop

1. Nanobot fetches and validates the contract when adopting a model and when
   boot, generation, or fingerprint changes.
2. Nanobot renders the real next prompt.
3. Below the soft boundary, it sends exact continuation normally.
4. At the soft boundary, it schedules background LCM.
5. If the prompt reaches the guaranteed boundary or the worst-case next turn
   cannot fit, it installs/executes blocking compaction before provider work.
6. Nanobot sends `continue_exact` with the expected contract revision.
7. Higgs tokenizes the actual request and atomically projects the successor
   retained allocation, including reserved output.
8. If the projection does not fit, Higgs returns
   `retention_compaction_required` without model work or cache mutation.
9. Nanobot compacts, rotates epoch, releases the retired session, refreshes a
   stale contract when necessary, and retries exactly once.
10. If the compacted immutable prefix/current transaction still cannot fit,
    Nanobot reports a bounded explicit error rather than looping or submitting
    the large prompt statelessly.

## Estimation and Headroom

Higgs may derive retained cost statically from model/cache geometry or from a
conservative observed upper bound. Observations may tighten safety, never
optimistically enlarge it. The estimate includes fixed and marginal costs and
must cover paired target/DFlash state.

The soft boundary is derived from the hard guarantee minus at least one
worst-case turn and maximum output growth. It is not a universal percentage.
The target-after-compaction is small enough that an unavoidable cold seed is a
bounded operation.

For more than one guaranteed retained session, Higgs reserves a per-session
share. Aggregate LRU without reservation cannot be advertised as a guarantee.
Active leased sessions are not ordinary LRU victims.

## Model Discovery and Human Setup

Higgs provides read-only discovery and planning commands:

```text
higgs models scan
higgs retention plan --model MODEL --bytes BYTES
higgs retention plan --model MODEL --tokens TOKENS
```

Discovery inspects configured roots and supported local model caches without
duplicating model-geometry logic in Nanobot. Planning reports model memory,
retained layout, requested budget, safe retained tokens, output reserve,
headroom, and the maximum safe alternative when a requested token target does
not fit. A desired-token selection resolves to and persists a byte budget.

## Compatibility

- V2 Nanobot with V2 Higgs enables guaranteed retained mode.
- V2 Nanobot with older Higgs retains stateless OpenAI compatibility but does
  not claim required retention.
- Older Nanobot remains able to make ordinary requests to V2 Higgs.
- Unknown V2-major or malformed contracts fail closed only for retained mode.
- Server restart and runtime model switch invalidate prior revisions and force
  session rotation.

## Test Strategy

All production behavior follows Red -> verified failure -> minimal Green ->
verified pass. Tests cover smart-constructor invariants, paired-state byte
accounting, pre-mutation rejection, prior-cache preservation, Nanobot soft and
hard compaction, bounded retry, stale contracts, cross-version matrices, and
same-byte/different-model token derivation.

The live acceptance test crosses soft and hard boundaries and asserts:

- no unexpected zero-cache turn;
- no cold prefill above the bounded compacted target;
- no oversized retention drop;
- no unchanged retry after compaction-required;
- retained bytes remain within budget;
- restart and model switch rotate contracts and sessions safely.

## Non-Goals

- Higgs does not semantically summarize conversations.
- Nanobot does not estimate KV tensor geometry.
- Paged allocation, disk offload, and attention eviction are not required for
  the safety invariant.
- V2 does not promise unlimited exact history on finite memory.
