# Higgs

## Project Structure

- `crates/higgs/` -- main binary crate (config, router, doctor, daemon, CLI)
- `crates/higgs-engine/` -- inference engine
- `crates/higgs-models/` -- model architectures

## Development Rules

### Doctor Validation

When adding or changing config fields, always update `crates/higgs/src/doctor.rs` to validate the new field. The doctor should catch misconfiguration before the server starts. Run `cargo test -p higgs -- --test-threads=1` to verify.

### Documentation

When changing user-facing behavior (config fields, CLI flags, API surface), update:

1. `README.md` -- config examples and reference tables
2. `crates/higgs/src/daemon.rs` -- the `higgs init` config template
3. Doc comments on public structs/fields if applicable

### Testing

- `cargo clippy -p higgs` -- must be clean (nursery lints enabled)
- `cargo fmt -p higgs -- --check` -- must pass
- `cargo test -p higgs -- --test-threads=1` -- all tests must pass (thread limit required due to shared port bindings)

## Performance Engineering Rules (HARD)

These rules exist because past sessions violated them. Each maps to a real failure that wasted hours.

1. **Headline metric first.** Never report a sub-metric improvement (verify_fwd, prefill, kernel time, ANE dispatch) without stating end-to-end tok/s AND acceptance rate on the same line. If the headline regressed, lead with that — never bury it. (Trigger: "verify_fwd 3.5x" while eff_tps went 6.6→4.8.)

2. **Scope discipline.** If asked `cargo clippy -p X`, run exactly that. If asked to fix file F, fix only F. Never widen scope (workspace-wide commands, refactoring adjacent code, "while I'm here") without one explicit ask: "should I also touch Y?" (Trigger: "errors went from 5 to 144 WTF have you done.")

3. **Evidence before prediction.** Before saying "X probably won't load / Y is likely the cause", read the most recent 3 `.planning/RECAP-*.md`, run `git log --grep=<feature> --oneline -20`, and check `memory/` files. Cite what you read. Never predict feature state from architecture inspection alone. (Trigger: "WTF dont you see benches / memory / docs ??? plus why you ignore my remarks.")

4. **Ceiling calculation required.** Before any optimization plan spanning more than 1 session: compute `current_value × best_case_speedup` and confirm it exceeds the target. State the ceiling explicitly in the plan. If ceiling < target, do not pursue — propose a different approach. (Trigger: 6 sessions on Bonsai B1 compile-wrap with a 33 tok/s ceiling vs 71.5 target.)

5. **No hardware excuses.** Investigate the algorithm and code path first. Mentioning battery, thermal, or ANE warm-up requires evidence (`pmset -g batt`, `powermetrics`, trace data). Do not blame the machine before reading the code. (Trigger: "I am on battery WTF YOU TALKING ABOUT OPUS. I AM FURIOUS!")

## GitNexus (indexed as higgs, 6246 symbols)

Use these skills when needed:
- `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` — before editing any function
- `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` — renames/moves
- `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` — bug traces
- `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` — "how does X work"

A PostToolUse hook re-runs `npx gitnexus analyze` after commits, so the index stays fresh automatically.
