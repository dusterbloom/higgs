# Session 14 — Spec-decode K=12 measurement

Date: 2026-04-24
Branch: feat/magic-canvas
Binary: `./target/release/higgs` (cargo build --release -p higgs --features ane)
Host: Mac16,1 (M-series, 32 GB unified memory), macOS 26.3.1

## Objective

With the newly wired speculative decode path in `SimpleEngine`, time one
cycle (`draft.draft()` + `verify_fn`) for K=12 against the
NexVeridian Qwen3.6-27B-4bit target and a Bonsai drafter. Capture
`draft_ms` median / p95 to establish the K=12 baseline.

## Instrumentation

Added `tracing::info!(target: "spec_decode", k, accepted, draft_ms, verify_ms, "cycle")`
to both speculative paths in `crates/higgs-engine/src/simple.rs`:

- `speculative_generate` (non-streaming)
- `speculative_streaming` (streaming)

Timing scheme:
- Wrap `speculative::speculative_step(...)` with `Instant::now()` to get
  `total_ms`.
- Measure `verify_fn` body explicitly with a second `Instant` → `verify_ms`.
- `draft_ms = max(0, total_ms - verify_ms)` — the time `speculative_step`
  spends in `draft.draft()` plus negligible bookkeeping.

Build: clean (`Finished release profile in 37.34s`), only pre-existing
warnings.

## Test configuration

`/tmp/higgs-measure.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8765

[[models]]
name = "qwen3-27b"
path = "/Users/peppi/.cache/lm-studio/models/NexVeridian/Qwen3.6-27B-4bit"
draft_model = "<see per-attempt below>"
num_draft = 12

[default]
provider = "higgs"
```

Launch:

```bash
HIGGS_SPEC_ALLOW_TOKENIZER_MISMATCH=1 \
HIGGS_BONSAI_DRAFTER_SEQ_LEN=2048 \
RUST_LOG=spec_decode=info,higgs=warn \
./target/release/higgs serve --config /tmp/higgs-measure.toml
```

Probe request:

```bash
xh --ignore-stdin POST http://127.0.0.1:8765/v1/completions \
  model=qwen3-27b \
  prompt="Write a short limerick about a debugger." \
  max_tokens:=64 temperature:=0.0 stream:=false
```

## Result: **no K=12 measurement captured**

All three drafter candidates failed before a cycle probe could land.

### Attempt 1 — `./bonsai-bd3lm-merged-bf16/` (bf16, has `bd3lm_extras.safetensors`)

Panic at load:

```
thread 'main' panicked at crates/higgs-models/src/diffusion.rs:392:
Missing: model.embed_tokens.scales
```

Root cause: `SimpleEngine::build_bonsai_draft` (`crates/higgs-engine/src/simple.rs:573`)
unconditionally calls `DiffusionEngine::load_q1(...)`, which expects a
1-bit quantized checkpoint (reads `.weight`, `.scales`, `.biases`). The
merged drafter is bf16 (`model.safetensors`, `dtype: "bfloat16"`), so
the `.scales` tensor is absent. No bf16 drafter loader is wired today.

### Attempt 2 — `~/.cache/lm-studio/models/prism-ml/Bonsai-1.7B-mlx-1bit/`

Load succeeds:

```
DiffusionEngine::load_q1: 28L, hidden=2048, heads=16/8, vocab=151669,
  dequantized to 6561MB fp32
AneBonsaiEngine: compiling 28 causal attn + OC-tiled FFN (dim=2048, inter=6144, seq=2048)
  28 attn kernels in 17240ms
  FFN base tile kernels compile: 18460ms
  Patching 112 FFN tile kernels (28 layers × (2 gated + 2 down-partial tiles))...
AneBonsaiEngine: ready in 81381ms — 28 attn + 112 FFN tile kernels
```

First request arrives (14:30:55), engine emits the expected paged-cache
fallback warning, then the ANE XPC connection dies 43 s later and the
process exits with no stdout / stderr and no DiagnosticReport:

```
paged_prefix_cache: Failed to page cache, using clone fallback
  error="Cache too short for paging"
...
higgs[91708] (ANEServices) XPC_ERROR_CONNECTION_INTERRUPTED     14:31:38
```

No `jetsam` / `killing` entries in `log show` for pid 91708, no crash
report at `~/Library/Logs/DiagnosticReports/`. Net effect: the
first live ANE dispatch during `draft.draft()` torpedoes the ANE driver
and the host process exits. Zero `spec_decode cycle` lines emitted.

Memory budget at this moment: 27B-4bit target (~13–14 GB) plus the
1.7B drafter dequantized to 6.5 GB fp32 + kernels — tight but under 32
GB. Cause of the XPC drop unknown; needs its own investigation before
K=12 is measurable. (Candidates: K=12 stresses per-step ANE state
because `AneBonsaiDraftModel::draft` does `num_draft` serial
`forward_last` calls — 12 dispatches/cycle; tokenizer mismatch may
drive the drafter into cold vocab slots; 2048-seq compile + memory
pressure.)

### Attempt 3 — `~/.cache/lm-studio/models/prism-ml/Bonsai-8B-mlx-1bit/`

Dequantized footprint alone exceeds host RAM:

```
DiffusionEngine::load_q1: 36L, hidden=4096, heads=32/8, vocab=151669,
  dequantized to 28867MB fp32
AneBonsaiEngine: compiling 36 causal attn + OC-tiled FFN (dim=4096, inter=12288, seq=2048)
  L0 attn compile: 4529ms
```

Process disappeared during FFN kernel compile; 28.9 GB fp32 drafter +
4-bit 27B target cannot coexist on a 32 GB machine. Port never opened.
Not a spec-decode bug — infeasible hardware combination.

## Conclusions

- Instrumentation is wired and would emit one line per cycle as soon as
  a drafter survives `draft()`. Verified by inspection only; no live
  cycle observed this session.
- The "spec-decode engine wired" milestone still needs a
  load-compatible drafter checkpoint paired with this 27B target
  before K=12 (or any K) numbers exist. Two independent blockers:
  1. Only the q1 loader is reachable from the config path — a bf16
     drafter cannot be loaded regardless of how it was produced.
  2. The only q1 Bonsai checkpoints on this host (1.7B, 8B) either
     kill the ANE XPC connection on first dispatch (1.7B) or OOM the
     host during compile (8B).
- No `draft_ms` median / p95 produced.

## Next steps (proposed)

- Teach `build_bonsai_draft` to accept bf16 / fp16 / 4-bit Bonsai
  checkpoints (a `DiffusionEngine::load_bf16` sibling or a dispatch on
  the checkpoint's dtype in `config.json`). That unlocks the existing
  `bonsai-bd3lm-merged-bf16` tree, which is the checkpoint the
  milestone-14 plan actually targets.
- Independently, reproduce the 1.7B ANE XPC drop outside spec-decode —
  load `AneBonsaiEngine` for the 1.7B q1 tree and call
  `forward_last` K=12 times directly — to decide whether the issue is
  K-dependent or a general driver fault on this host.
- Only after one of the above clears should we retry the K=12
  measurement and populate the draft_ms median / p95 table below.

## Raw artifacts

- Config: `/tmp/higgs-measure.toml`
- Server log (latest attempt, 8B): `/tmp/higgs-measure/server.log`
- OS log snippets: see `log show --predicate 'process == "higgs"'`
- No `spec_decode=info cycle` lines to parse this session.

| K | cycles | accepted mean | draft_ms p50 | draft_ms p95 | verify_ms p50 | verify_ms p95 |
|---|--------|---------------|--------------|--------------|---------------|---------------|
| 12 | 0 | n/a | n/a | n/a | n/a | n/a |
