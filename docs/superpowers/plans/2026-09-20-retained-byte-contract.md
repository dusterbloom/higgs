# Retained-Byte Fast Session Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make required Higgs continuations either publish an exact reusable successor cache or reject before inference, while Nanobot compacts against Higgs's model-specific retained-byte-derived prompt envelope.

**Architecture:** Higgs owns a validated V2 fast-session contract, conservative retained-cost derivation, and atomic hard admission. Nanobot consumes only the derived absolute prompt limits, schedules LCM, rotates physical sessions, and performs one bounded retry. Retained bytes are the sole user memory authority; logical context remains an independent architectural ceiling.

**Tech Stack:** Rust 2021, Axum, Tokio, serde/serde_json, MLX, SQLite, existing Higgs and Nanobot test harnesses.

## Global Constraints

- Follow `docs/superpowers/specs/2026-09-20-retained-byte-contract-design.md` verbatim.
- Production code follows strict Red -> observed failure -> minimal Green -> observed pass.
- Run GitNexus upstream impact analysis before modifying every existing symbol and report HIGH/CRITICAL risk before editing.
- Preserve all unrelated dirty-tree changes in both repositories.
- Nanobot must not calculate model KV geometry or bytes per token.
- Higgs must include target and draft/DFlash state in one retained allocation.
- Required continuation must never silently execute a stateless cold prefill.
- A compact-required rejection must occur before model work and must preserve the prior retained cache.
- Unknown or malformed safety contracts fail closed for retained mode while stateless OpenAI compatibility remains available.
- No new dependencies unless an existing standard-library or repository facility cannot implement the requirement.

---

### Task 1: Higgs V2 Fast-Session Contract

**Files:**
- Modify: `crates/higgs/src/capacity.rs`
- Modify: `crates/higgs/src/capacity/registry.rs`
- Modify: `crates/higgs/src/state.rs`
- Modify: `crates/higgs/src/routes/capacity.rs`
- Test: colocated unit tests in those modules

**Interfaces:**
- Produces validated `FastSessionContractV2` serialization with `contractRevision`, `retainedBudgetBytes`, `guaranteedFastPromptTokens`, `softCompactionPromptTokens`, `targetAfterCompactionTokens`, and `guaranteedSessions`.
- Consumes existing model fingerprint, boot ID, generation, logical context, output reserve, and retained-byte ceiling.

- [ ] **Step 1: Write failing constructor/serialization tests**

Add tests proving that contradictory numeric relationships cannot construct a V2 contract, that the current `retainedSessionTokens == safeTotalTokens > maxPromptTokens` incident is rejected internally rather than serialized, and that two model-cost fixtures under the same byte budget derive different guaranteed prompt limits.

- [ ] **Step 2: Run focused tests and record RED**

Run:

```bash
cargo test -p higgs capacity --release -- --nocapture
```

Expected: new V2 tests fail because the validated contract and derivation do not exist.

- [ ] **Step 3: Implement the minimal validated contract**

Use private fields and a constructor returning `Result`. Derive:

```text
guaranteed = min(
  model_context - output_reserve,
  legacy_token_upper_bound_if_nonzero,
  floor((per_session_retained_bytes - fixed_bytes - output_reserve_bytes - headroom_bytes)
        / conservative_bytes_per_token)
)
soft = guaranteed - one_worst_case_turn_and_output_growth
target = min(existing deterministic LCM target, soft - 1)
```

Use checked arithmetic and explicit typed construction errors. Preserve V1 only as a compatibility representation; V2 is authoritative for guaranteed retention.

- [ ] **Step 4: Run focused tests and record GREEN**

Run the Task 1 command again and require zero failures.

- [ ] **Step 5: Commit Task 1**

Commit only Task 1 files with `feat(capacity): publish retained-byte fast-session contract`.

### Task 2: Higgs Pre-Mutation Required-Retention Admission

**Files:**
- Modify: `crates/higgs/src/types/openai.rs`
- Modify: `crates/higgs/src/routes/chat.rs`
- Modify: `crates/higgs/src/routes/anthropic.rs`
- Modify: `crates/higgs-engine/src/simple.rs`
- Modify: `crates/higgs/src/error.rs`
- Test: `crates/higgs/tests/integration/retained_session_api.rs` or existing retained-session integration module

**Interfaces:**
- Consumes `contractRevision` and the Task 1 fast-session contract.
- Produces tagged `Stateless | ContinueExact` request behavior and typed `retention_compaction_required` / `stale_retention_contract` errors.

- [ ] **Step 1: Write failing engine and route tests**

Cover target-only under/over budget, paired target+DFlash where target alone fits but the pair does not, output reserve crossing the boundary, stale revision, missing retained session, and rejection preserving the prior cache. Add an execution counter/assertion proving rejected requests perform no prefill/decode.

- [ ] **Step 2: Run focused tests and record RED**

Run:

```bash
cargo test -p higgs-engine --release retained_admission -- --nocapture
cargo test -p higgs --release retained_session -- --nocapture
```

Expected: tests fail because admission currently occurs after retained state creation or best-effort fallback remains possible.

- [ ] **Step 3: Implement one admission path**

Parse the tagged retention request, validate contract identity, tokenize the real prompt, conservatively project the successor including maximum requested output, and reject before acquiring/mutating generation state when it cannot fit. Required mode must bypass every cold-bootstrap fallback. Preserve the old retained entry on every rejection.

- [ ] **Step 4: Add retention receipt**

Return `outcome`, session identity/epoch, retained tokens/bytes, and contract revision in the existing extension/usage surface without changing standard stateless response compatibility.

- [ ] **Step 5: Run focused tests and record GREEN**

Run both Task 2 commands and require zero failures.

- [ ] **Step 6: Commit Task 2**

Commit with `feat(cache): require atomic retained continuation`.

### Task 3: Nanobot V2 Contract and Fail-Closed Compatibility

**Files (Nanobot repository):**
- Modify: `src/agent/capacity.rs`
- Modify: `src/providers/openai_compat.rs`
- Modify: `src/providers/base.rs`
- Modify: `src/errors.rs`
- Test: colocated unit tests and provider contract tests

**Interfaces:**
- Consumes Higgs `FastSessionContractV2` JSON.
- Produces `EffectiveCapacity` containing absolute fast/soft/target prompt limits and typed retained outcomes.

- [ ] **Step 1: Write failing parser and compatibility tests**

Use golden V2 fixtures matching Task 1. Cover valid V2, malformed relationships, unknown major version, stale model/fingerprint/revision, V1 server, and ordinary stateless compatibility. Assert malformed/unknown contracts never fall back to a larger retained limit.

- [ ] **Step 2: Run focused tests and record RED**

Run:

```bash
cargo test --release agent::capacity providers::openai_compat -- --nocapture
```

Expected: V2 parsing and fail-closed retained-mode assertions fail.

- [ ] **Step 3: Implement minimal V2 consumption**

Make retained capability an explicit enum (`Unavailable`, `LegacyStatelessOnly`, `Guaranteed(V2)`). Retain standard stateless requests for older servers. Do not calculate bytes/token in Nanobot.

- [ ] **Step 4: Emit tagged required continuation**

Translate Nanobot session markers into the V2 tagged request with epoch and expected contract revision. Parse compact-required, stale-contract, and retained-session-unavailable errors structurally.

- [ ] **Step 5: Run focused tests and record GREEN**

Run Task 3 tests and require zero failures.

- [ ] **Step 6: Commit Task 3**

Commit in Nanobot with `feat(higgs): adopt fast-session retention contract`.

### Task 4: Nanobot Autonomous Compaction and Bounded Retry

**Files (Nanobot repository):**
- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/agent/lcm.rs`
- Modify: `src/agent/agent_core.rs`
- Modify: `src/agent/agent_loop/recovery_eval.rs`
- Test: `src/agent/agent_loop/tests.rs`
- Test: `tests/lcm_e2e_tests.rs`

**Interfaces:**
- Consumes Task 3 absolute `guaranteed`, `soft`, and `target` prompt limits.
- Produces background soft compaction, blocking hard compaction, epoch rotation, old-session release, and exactly one compacted retry.

- [ ] **Step 1: Write failing incident regression**

Model a 45,056 logical context with a 14,700-token fast-session boundary. Assert a rendered prompt crossing soft/hard retention pressure compacts before provider invocation even though it fits logical context. Assert there is no unchanged cold retry.

- [ ] **Step 2: Write failing recovery tests**

Cover pending background checkpoint installation, compaction failure preserving evidence, stale revision refresh, session rotation, retired-session release, retry count exactly one, and irreducible current transaction returning an explicit bounded error.

- [ ] **Step 3: Run focused tests and record RED**

Run:

```bash
cargo test --release retained_contract -- --nocapture
cargo test --release --test lcm_e2e_tests -- --nocapture
```

- [ ] **Step 4: Wire the fast-session wall into existing LCM**

Use exact rendered prompt tokens for hard admission and the server-provided absolute soft boundary for background scheduling. On typed compact-required: install/execute compaction, rotate epoch, eagerly release the retired ID, refresh stale contracts, and retry once. Never retry the unchanged prompt statelessly.

- [ ] **Step 5: Run focused tests and record GREEN**

Run both Task 4 commands and require zero failures.

- [ ] **Step 6: Commit Task 4**

Commit with `fix(lcm): compact before retained-cache cliff`.

### Task 5: Higgs Model Discovery and Retention Planner

**Files:**
- Modify: `crates/higgs/src/cli.rs` or the existing CLI command module
- Modify: `crates/higgs/src/model_resolver.rs`
- Create only if required by existing CLI layout: `crates/higgs/src/retention_plan.rs`
- Modify: `docs/configuration.md`
- Test: colocated CLI/planner tests

**Interfaces:**
- Produces `higgs models scan` and `higgs retention plan --model ... --bytes ...|--tokens ...`.
- Reuses the same Task 1 planner; it must not duplicate formulas.

- [ ] **Step 1: Write failing planner tests**

Fixture supported model directories with target-only and target+draft layouts. Assert same bytes produce different safe token recommendations, desired tokens resolve to required bytes, unsafe requests report a maximum, and persisted recommendations use bytes as the authority.

- [ ] **Step 2: Run focused tests and record RED**

Run:

```bash
cargo test -p higgs --release retention_plan -- --nocapture
```

- [ ] **Step 3: Implement scan and plan commands**

Reuse existing trusted runtime roots/cache resolution and adapter validation. Print model memory, retained layout, requested budget, safe token recommendation, output reserve, headroom, and rejection reason/max alternative. Do not mutate configuration without an explicit CLI write flag.

- [ ] **Step 4: Run focused tests and record GREEN**

Run Task 5 tests and require zero failures.

- [ ] **Step 5: Commit Task 5**

Commit with `feat(cli): plan retained memory for local models`.

### Task 6: Cross-Repository Protocol and Endurance Verification

**Files:**
- Modify/add the smallest existing integration fixtures in both repositories
- Add golden JSON fixtures under each repository's existing test-fixture convention
- Modify: `docs/configuration.md`
- Modify (Nanobot): relevant local-backend documentation

**Interfaces:**
- Consumes Tasks 1-5.
- Produces compatibility matrix and live acceptance evidence.

- [ ] **Step 1: Add golden cross-version matrix tests**

Cover V2/V2, V2/V1, V1/V2 stateless, malformed V2, restart, model switch, concurrent idle eviction, target-only, and paired target+DFlash. Assert no inference/cache mutation on invalid or compact-required requests.

- [ ] **Step 2: Run repository-focused suites**

Run:

```bash
cargo test -p higgs --release capacity -- --nocapture
cargo test -p higgs --release retained_session -- --nocapture
```

and in Nanobot:

```bash
cargo test --release retained_contract -- --nocapture
cargo test --release --test lcm_e2e_tests -- --nocapture
```

- [ ] **Step 3: Run full release verification**

Run `cargo test --release` and `cargo build --release` in both repositories. Run formatting/diff checks without rewriting unrelated user changes.

- [ ] **Step 4: Run isolated live acceptance**

Using a disposable config/session DB and an owned Higgs process, cross soft and hard boundaries and verify metrics show no unexpected zero-cache turn, no oversized retention drop, no unchanged retry, and retained bytes within budget. Restart Higgs and switch models to validate contract/session rotation. Stop every owned process.

- [ ] **Step 5: Detect scope and commit verification artifacts**

Run GitNexus detect-changes in both repositories and inspect `git diff --check`. Commit only intended tests/docs/evidence with `test: verify retained-session contract end to end`.

### Task 7: Independent Astra Review and Fix Gate

**Files:**
- No planned production files; findings determine minimal follow-up edits.

- [ ] **Step 1: Give Astra the spec, plan, commit ranges, and test reports**

Request medium-effort review of contract invariants, pre-mutation ordering, paired-state accounting, Nanobot retry/rotation behavior, compatibility, test validity, and user-facing simplicity.

- [ ] **Step 2: Resolve every Critical/Important finding**

Use Red→Green tests for fixes. Re-run scoped tests and request a focused re-review.

- [ ] **Step 3: Final verification and integration decision**

Run fresh full release tests/builds, GitNexus detect-changes, and diff checks. Do not install binaries or alter the user's live configuration unless separately authorized by the user.
