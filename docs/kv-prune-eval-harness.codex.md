# Codex task: KV-prune accuracy-sweep eval harness

Self-contained spec. You do **not** need the conversation that produced it.

## Background (why this exists)

`higgs` is adding TIM/TIMRUN-style **KV-cache pruning** to run long-horizon
reasoning on a stock (un-finetuned) Qwen3.6 MoE. The mechanism is already built
and proven in `crates/higgs-models/src/cache.rs`:

```rust
// On SteppingKeyValueCache (the dense decode cache):
pub fn prune_span(&mut self, a: i32, b: i32, rope: RopeShift) -> Result<(), Exception>
// RopeShift { base: f32, dims: i32, scale: f32, traditional: bool }
```

`prune_span(a, b, rope)` drops the half-open token span `[a, b)` from the cache,
compacts the survivors, and re-rotates the surviving suffix by `R(-(b-a))` so
positions stay dense. Proven bit-equivalent (f32 tol) to never inserting those
tokens — see test `prune_span_equiv_never_inserted`. Dense path only (errors on
TurboQuant).

The open question this harness answers: **how aggressively can we prune a stock
Qwen3.6 MoE's KV before reasoning accuracy degrades?** The paper's
quality-preserving regime is ~50–60% pruned. If our knee is near there, the
training-free thesis holds.

## What you own (the separable part)

A **grading + metrics + problem-set** module. You do NOT touch the model decode
loop or `prune_span` — those are wired on the higgs side against a clean
interface you define. Build a new crate-local module (suggest
`crates/higgs-bench/src/prune_eval.rs`) exposing:

```rust
/// One reasoning problem with a checkable final answer.
pub struct Problem { pub id: String, pub prompt: String, pub answer: String }

/// Curated set: ~50 items, mix of GSM8K-style arithmetic word problems and
/// MATH-style short-answer. Hard-code them (no network at run time). Answers
/// are exact strings (canonical numeric form, e.g. "42", "-3/4").
pub fn problem_set() -> Vec<Problem>;

/// Extract the model's final answer from free-form output and exact-match it
/// against `expected`. Handle: trailing "#### N" (GSM8K), "\boxed{...}" (MATH),
/// "The answer is X.", and bare-last-number fallback. Normalize whitespace,
/// commas in numbers, and trivial fraction forms.
pub fn grade(model_output: &str, expected: &str) -> bool;

/// One row of the sweep result.
pub struct SweepRow {
    pub prune_pct: u32,      // target prune rate this row was run at
    pub accuracy: f32,       // fraction graded correct
    pub mean_tok_per_s: f32,
    pub peak_resident_kv: u32, // max tokens resident across the run
    pub n: u32,
}

/// Render rows as a fixed-width table: prune% | acc | tok/s | peakKV | n,
/// plus a one-line summary naming the knee (highest prune_pct whose accuracy is
/// within `tol` of the prune=0 row).
pub fn render_table(rows: &[SweepRow], tol: f32) -> String;
```

Write a `grade` unit test proving the general logic (each extraction format +
one false case), not per-problem assertions. Keep the problem set small and
high quality.

## The interface boundary (higgs side wires this)

The higgs-side runner (built separately) will, per problem and per target
prune rate, drive the existing decode loop in
`crates/higgs-engine/src/simple.rs` (`generate_inner`) with a **prune policy**:

- Keep the first `S = 4` tokens always (attention sinks).
- Keep the most recent `W` tokens always.
- When resident length exceeds the budget implied by the target prune rate,
  call `prune_span(S, S + k, rope)` to evict the oldest `k` non-sink tokens,
  looping `for c in caches.iter_mut().flatten() { c.prune_span(..) }` across all
  layers. `rope = RopeShift { base: rope_theta, dims: head_dim, scale: 1.0,
  traditional: false }` (read from model config).

It collects `(model_output, tok_per_s, peak_resident_kv)` per problem and calls
your `grade` + `render_table`. **Token-age pruning only here — no Thread-2
schema yet** (that's a later phase); this isolates the mechanism's accuracy
curve.

## Deliverable

`cargo test -p higgs-bench prune_eval` passes (your grader tests). Module
compiles clean (`cargo clippy -p higgs-bench`, nursery lints) and `cargo fmt`.
Do not run the 35B sweep yourself — the higgs side does that once your grader +
table land.

## Sweep matrix (for the final run, FYI)

Target prune rates `{0, 25, 40, 55, 70}` %, N≈50 problems, model
`qwen3_moe` / `qwen3_5_moe` (Qwen3.6 35B-A3B). Headline = highest prune% whose
accuracy stays within noise of the 0% row, plus tok/s and peak-KV at that point.
