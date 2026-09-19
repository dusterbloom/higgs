# Model-aware fast defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make nightly select non-thinking and the fastest validated Escha execution profile automatically, with no normal-use environment flags.

**Architecture:** Request thinking resolution stays in crates/higgs/src/reasoning.rs: the loaded engine supplies capability while an omitted request resolves to non-thinking. Checkpoint performance policy stays in the Escha conversion target: only the exact Qwen3.8 dense structural profile defaults to affine Q2; all other checkpoints retain Q4. README/model documentation explains the automatic policy and reserves environment variables for diagnostics.

**Tech Stack:** Rust 2024, MLX/Metal, Cargo tests and Clippy, Markdown.

## Global Constraints

- Work in the existing nightly checkout; do not create another worktree or touch the user's unrelated AGENTS.md, CLAUDE.md, .omen/, or prior untracked planning files.
- Run GitNexus upstream impact analysis before editing each existing function and warn on HIGH or CRITICAL impact.
- Use test-first red/green cycles for production behavior changes.
- Preserve HIGGS_ESCHA_NATIVE, HIGGS_ESCHA_AFFINE_BITS, and HIGGS_BONSAI_Q2_SIMD as explicit diagnostic overrides.
- Do not default-enable HIGGS_ESCHA_TRELLIS_GEMM; it remains experimental.
- Run only one Higgs server at a time and stop it after each GPU E2E check.
- Run GitNexus detect-changes and git diff --check before each commit.

---

### Task 1: Make omitted requests non-thinking by default

**Files:**

- Modify: crates/higgs/src/reasoning.rs:3-43, tests in the same module

**Interfaces:**

- Consumes: effective_thinking_enabled(thinking_supported, model_names, reasoning, explicit) from OpenAI and Anthropic routes.
- Produces: false for an omitted request; explicit enable_thinking takes precedence over reasoning.effort; an engine without thinking support always returns false.

- [ ] **Step 1: Run impact analysis**

Run: /opt/homebrew/bin/gitnexus impact effective_thinking_enabled --direction upstream --repo higgs

Expected: record direct routes/tests and stop for a HIGH/CRITICAL warning before editing.

- [ ] **Step 2: Write failing behavior tests**

In reasoning.rs, replace positive omitted-default tests with:

~~~rust
#[test]
fn defaults_any_thinking_capable_model_off() {
    assert!(!effective_thinking_enabled(
        true,
        &["mlx-community/Qwen3.5-foo"],
        None,
        None,
    ));
}

#[test]
fn configured_default_can_enable_thinking() {
    assert!(effective_thinking_enabled(
        true,
        &["local-model"],
        None,
        Some(true),
    ));
}
~~~

- [ ] **Step 3: Verify the tests fail for the intended default**

Run: cargo test -p higgs reasoning::tests --lib

Expected: defaults_any_thinking_capable_model_off fails because the existing fallback enables Qwen3.5 thinking.

- [ ] **Step 4: Implement the minimal resolver change**

Delete the version-name heuristic. Rename the first parameter to thinking_supported, preserve its hard capability gate, preserve explicit toggle and reasoning.effort precedence, and make the omitted-request fallback false:

~~~rust
pub fn effective_thinking_enabled(
    thinking_supported: bool,
    _model_names: &[&str],
    reasoning: Option<&ReasoningConfig>,
    explicit: Option<bool>,
) -> bool {
    if !thinking_supported {
        return false;
    }
    if let Some(want) = explicit {
        return want;
    }
    match reasoning.and_then(|r| r.effort.as_deref()) {
        Some(effort) if effort.is_empty() || effort.eq_ignore_ascii_case("none") => false,
        Some(_) => true,
        None => false,
    }
}
~~~

- [ ] **Step 5: Verify focused behavior**

Run: cargo test -p higgs reasoning::tests --lib

Expected: all resolver tests pass, including explicit opt-in and unavailable-engine rejection.

- [ ] **Step 6: Commit the isolated behavior change**

Run:

~~~bash
git add crates/higgs/src/reasoning.rs
git diff --check
/opt/homebrew/bin/gitnexus detect-changes --repo higgs
git commit -m "feat(reasoning): default omitted requests off"
~~~

### Task 2: Default exact dense Escha Qwen3.8 to affine Q2

**Files:**

- Modify: crates/higgs-models/src/eschamoe.rs:989-1024, tests in the same module
- Verify: crates/higgs-models/src/qwen3_next.rs:13950-14025

**Interfaces:**

- Consumes: conversion_target_for_config(config, env_value) and the exact Qwen3.8 dense structural predicate.
- Produces: AffineTarget { group_size: 64, bits: 2 } when the exact profile has no override; HIGGS_ESCHA_AFFINE_BITS=2..8 wins; other Escha checkpoints receive CONVERSION_TARGET Q4.

- [ ] **Step 1: Run impact analysis**

Run: /opt/homebrew/bin/gitnexus impact conversion_target_for_config --direction upstream --repo higgs

Expected: record conversion call paths and stop for a HIGH/CRITICAL warning before editing.

- [ ] **Step 2: Write a failing structural-default test**

Add a test next to existing Escha conversion tests:

~~~rust
#[test]
fn qwen38_dense_defaults_to_q2_but_near_matches_stay_q4() {
    let exact = serde_json::json!({
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_experts": 0
        }
    });
    assert_eq!(conversion_target_for_config(&exact, None).bits, 2);
    assert_eq!(conversion_target_for_config(&exact, Some("4")).bits, 4);
    let mut near_match = exact;
    near_match["text_config"]["num_hidden_layers"] = serde_json::json!(63);
    assert_eq!(conversion_target_for_config(&near_match, None).bits, 4);
}
~~~

- [ ] **Step 3: Verify the exact-profile default fails**

Run: cargo test -p higgs-models qwen38_dense_defaults_to_q2_but_near_matches_stay_q4 --lib

Expected: the first assertion fails with Q4 (bits == 4).

- [ ] **Step 4: Implement the model-specific default**

Make conversion_target_from_env accept a fallback bit width. Call it with 2 from the exact Qwen3.8 branch and keep CONVERSION_TARGET.bits as the fallback for every other branch:

~~~rust
fn conversion_target_from_env(value: Option<&str>, default_bits: i32) -> AffineTarget {
    let bits = value
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|bits| (2..=8).contains(bits))
        .unwrap_or(default_bits);
    AffineTarget { bits, ..CONVERSION_TARGET }
}
~~~

- [ ] **Step 5: Verify conversion and decode-policy gates**

Run:

~~~bash
cargo test -p higgs-models qwen38_dense_defaults_to_q2_but_near_matches_stay_q4 --lib
cargo test -p higgs-models escha_qwen38_q2_defaults_are_structural_and_overridable --lib
cargo test -p higgs-models eschamoe --lib
~~~

Expected: all pass; the existing Q2 SIMD test still rejects non-Escha and non-exact shapes.

- [ ] **Step 6: Commit the isolated conversion-policy change**

Run:

~~~bash
git add crates/higgs-models/src/eschamoe.rs
git diff --check
/opt/homebrew/bin/gitnexus detect-changes --repo higgs
git commit -m "perf(escha): default exact dense 27B to Q2"
~~~

### Task 3: Document zero-flag fast defaults and verify real models

**Files:**

- Modify: README.md:133,175-184
- Modify: docs/models.md:167-195

**Interfaces:**

- Consumes: the automatic native MoE and dense-Q2 policies implemented above.
- Produces: normal-use examples that need only model selection and optional --mlx-profile throughput; environment variables are described as diagnostic overrides.

- [ ] **Step 1: Update user-facing wording**

Add these two statements to README's local-model section:

~~~markdown
- **Escha 35B MoE:** native trellis experts are selected automatically; no Escha environment flag is required.
- **Escha Qwen3.8-27B dense:** the exact released layout automatically uses affine Q2 and its matching SIMD decode path; set HIGGS_ESCHA_AFFINE_BITS=4..8 only to compare another conversion target.
~~~

State that omitted chat requests default to non-thinking, while enable_thinking: true or a non-none reasoning effort explicitly enables it when supported. Retain HIGGS_ESCHA_NATIVE=0 and HIGGS_ESCHA_TRELLIS_GEMM=1 only in a diagnostic/experimental paragraph.

- [ ] **Step 2: Format and run static verification**

Run:

~~~bash
cargo fmt --all -- --check
cargo clippy -p higgs --lib -- -D warnings
cargo clippy -p higgs-models -- -D warnings
cargo test -p higgs reasoning::tests --lib
cargo test -p higgs-models eschamoe --lib
~~~

Expected: all commands exit zero.

- [ ] **Step 3: Run sequential release E2E checks with no Escha environment variables**

For each model, first confirm pgrep -x higgs has no result; launch Higgs loopback-only with --mlx-profile throughput, send a non-thinking request that requires exactly PINEAPPLE, verify the response, and stop the server before the next model:

~~~bash
./target/release/higgs serve --model "$MODEL_PATH" --host 127.0.0.1 --port 9011 \
  --api-key e2e --mlx-profile throughput
~~~

Use Qwen3.6-35B-A3B-Escha-W2 first and assert its log includes Installed native trellis expert weights. Use Qwen3.8-27B-Escha-W2 second and assert its log includes the Q2/SIMD dispatch with HIGGS_TRACE_Q2_DISPATCH=1 only for this verification run. Do not set native/affine/GEMM performance flags.

- [ ] **Step 4: Commit documentation and final checks**

Run:

~~~bash
git add README.md docs/models.md
git diff --check
/opt/homebrew/bin/gitnexus detect-changes --repo higgs
git commit -m "docs: describe automatic Escha fast defaults"
~~~

