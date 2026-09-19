# Qwen3.8-27B Escha Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an evidence-backed Qwen3.8-27B Escha-W2 case study, link it from Higgs’ README, and produce reproducible 1K/4K/8K prefill and 128-token greedy decode measurements without a second 27B model load.

**Architecture:** Extend the existing server-driven `bench_decode` harness instead of using the in-process `bench_frontier` binary, because only the server applies the configured MLX wired-memory limit needed by the 27B model. The harness will send a unique-prefix, no-thinking, no-speculation request and persist server-reported prompt-token counts, TTFT, and post-first-token decode throughput. The static GitHub Pages route will consume those raw JSON artifacts and link the exact nightly commit that produced them.

**Tech Stack:** Rust workspace (`higgs-bench`), Clap, Reqwest/SSE, TOML model manifest, Markdown, static HTML/CSS/JS, GitHub Pages, `cargo test`, and a single loopback Higgs server on Apple Silicon.

## Global Constraints

- Base all Higgs work on `nightly` after creating an isolated worktree; do not alter the root checkout or its user-owned `AGENTS.md`, `CLAUDE.md`, and `.omen/` changes.
- Clone the separate public `dusterbloom/escha-mlx-evidence` repository into a disposable directory under `/private/tmp`; do not place it inside the Higgs worktree.
- Run exactly one Higgs process during model loading, smoke testing, or benchmarking. Abort when `pgrep -x higgs` finds an existing server and stop the benchmark server before loading anything else.
- Benchmark `EschaLabs/Qwen3.8-27B-Escha-W2` as a dense checkpoint. Do not claim 35B-A3B native-MoE residency, six-second loading, quality/fidelity, or a cross-runtime comparison.
- Use the normal `[local].raise_wired_limit = true` config path, a loopback bind, temperature `0`, `reasoning.effort = "none"`, and request `speculation = "none"`.
- Record only server-reported prompt/completion token counts. Do not publish an incomplete, cached, EOS-shortened, failed, or thermally unstable run.
- Each benchmark request starts with a distinct cache-buster before the shared prompt, so neither session nor global prefix reuse can supply prefill work.
- Before editing any existing function, class, or method, run GitNexus upstream impact analysis and report the blast radius; stop for HIGH or CRITICAL risk. Run GitNexus change detection before each commit.
- Do not stage, commit, push, or publish either repository without the user’s explicit authorization for that action.

---

## File Map

- `crates/higgs-bench/src/bin/bench_decode.rs`: server-driven request construction, generated long prompts, cache-buster, safe auth header, and persisted prompt-prefill metrics.
- `crates/higgs-bench/src/models.rs`: regression test that validates the repository model manifest and the new Escha entry.
- `benchmarks/models.toml`: public checkpoint metadata for the Qwen3.8-27B Escha-W2 benchmark key.
- `docs/benchmarking.md`: exact benchmark command and semantics for prompt tok/s, alias routing, cache busting, and environment-only API authentication.
- `README.md`: concise model-specific Escha support/evidence wording and both public evidence links.
- `dusterbloom/escha-mlx-evidence/index.html`: one non-navigation footer link to the new 27B route.
- `dusterbloom/escha-mlx-evidence/qwen3.8-27b/index.html`: 27B dense case-study page that reuses `../styles.css` and `../script.js`.
- `dusterbloom/escha-mlx-evidence/qwen3.8-27b/results.json`: selected raw `bench_decode` results and cold-load metadata used by the page.

### Task 1: Make `bench_decode` report cache-cold server prefill and strict greedy decode

**Files:**
- Modify: `crates/higgs-bench/src/bin/bench_decode.rs:31-425`
- Test: `crates/higgs-bench/src/bin/bench_decode.rs` new private `tests` module

**Interfaces:**
- Preserve `bench_decode --model <manifest-key>` and all existing request sampling flags.
- Add `--request-model <server-model-name>`; absent means the manifest `Model.path` remains the request model.
- Add paired `--tokenizer-dir <local-model-dir>` and `--prompt-tokens <count>`; the latter constructs ordinary deterministic text using the local tokenizer and is mutually exclusive with `--prompt`.
- Add `--cache-buster <true|false>`, defaulting to `true`. When enabled, `cache_busted_prompt(base, phase, ordinal)` prepends the distinct marker before the base prompt.
- Read an optional `HIGGS_BENCH_API_KEY` environment variable and set it only as an HTTP `Authorization: Bearer` header; never include its value in CLI args or persisted JSON.
- Extend each JSON `TrialResult` with `prompt_tokens: Option<u32>` and `prefill_tokps: Option<f64>`, where `prefill_tokps = prompt_tokens / (ttft_ms / 1000.0)` only when Higgs sent terminal usage.

- [ ] **Step 1: Run GitNexus impact analysis and record the risk**

  Run upstream impact for `run`, `run_trial`, and `handle_sse_line` in `bench_decode.rs`. Report direct callers and affected benchmark flows before touching the existing functions. Do not proceed without surfacing a HIGH or CRITICAL result.

- [ ] **Step 2: Write the failing request-contract tests**

  Add a private test module before implementing the helpers. The tests must define the desired public benchmark contract without initializing MLX or a network client:

  ```rust
  #[test]
  fn request_body_disables_thinking_and_speculation() {
      let body = request_body("escha-27b", "cache-key\nPrompt", 128, 0.0);
      assert_eq!(body["model"], "escha-27b");
      assert_eq!(body["reasoning"]["effort"], "none");
      assert_eq!(body["speculation"], "none");
      assert_eq!(body["stream_options"]["include_usage"], true);
  }

  #[test]
  fn cache_buster_is_first_and_unique() {
      let first = cache_busted_prompt("body", "trial", 1, true);
      let second = cache_busted_prompt("body", "trial", 2, true);
      assert!(first.starts_with("[higgs-bench cache-key trial-1]"));
      assert_ne!(first, second);
      assert_eq!(cache_busted_prompt("body", "trial", 1, false), "body");
  }

  #[test]
  fn usage_yields_prompt_prefill_rate() {
      let mut first = None;
      let mut chunks = 0;
      let mut completion = None;
      let mut prompt = None;
      handle_sse_line(
          b"data: {\"usage\":{\"prompt_tokens\":1024,\"completion_tokens\":128}}\n",
          &mut first,
          &mut chunks,
          &mut completion,
          &mut prompt,
      );
      assert_eq!(prompt, Some(1024));
      assert_eq!(completion, Some(128));
      assert_eq!(prefill_tokps(prompt, 5120.0), Some(200.0));
  }
  ```

- [ ] **Step 3: Run the new tests to demonstrate the missing contract**

  Run:

  ```bash
  cargo test -p higgs-bench --bin bench_decode
  ```

  Expected: compilation failure because `request_body`, `cache_busted_prompt`, and `prefill_tokps` do not yet exist.

- [ ] **Step 4: Implement minimal request and result helpers**

  Extract the inline JSON creation in `run_trial` into this pure helper, preserving existing optional sampling fields after construction:

  ```rust
  fn request_body(model: &str, prompt: &str, max_tokens: u32, temperature: f32) -> serde_json::Value {
      serde_json::json!({
          "model": model,
          "messages": [{"role": "user", "content": prompt}],
          "stream": true,
          "stream_options": {"include_usage": true},
          "reasoning": {"effort": "none"},
          "speculation": "none",
          "max_tokens": max_tokens,
          "temperature": temperature,
      })
  }

  fn cache_busted_prompt(base: &str, phase: &str, ordinal: u32, enabled: bool) -> String {
      if enabled {
          format!("[higgs-bench cache-key {phase}-{ordinal}]\n{base}")
      } else {
          base.to_owned()
      }
  }

  fn prefill_tokps(prompt_tokens: Option<u32>, ttft_ms: f64) -> Option<f64> {
      (ttft_ms > 0.0)
          .then(|| prompt_tokens.map(|tokens| f64::from(tokens) / (ttft_ms / 1_000.0)))
          .flatten()
  }
  ```

  Add `request_model`, `tokenizer_dir`, `prompt_tokens`, and `cache_buster` to `Args` and persist their non-secret settings in `Params`. Resolve the request model once in `run`. Resolve the base prompt by either retaining `--prompt`/the short default or by loading `tokenizer.json` from `--tokenizer-dir`, encoding a fixed ordinary-prose corpus, cycling exactly `--prompt-tokens` token IDs, decoding them to text, and appending this fixed instruction: `Write a continuous technical explanation of the preceding passage in at least 200 words. Do not conclude early.` Reject `--prompt-tokens` without `--tokenizer-dir` and reject it with `--prompt`.

  For every warmup and measured trial, create its distinct prompt before calling `run_trial`. Pass the optional environment key into `run_trial`; build the request with `request_body`, then conditionally append the bearer header. Keep the secret out of `RunMetadata`, `Params`, error messages, and command output.

  Keep `handle_sse_line` as the only parser of terminal usage. Place the parsed `prompt_tokens` and its `prefill_tokps` in `TrialResult`; retain `None` rather than inventing a count when a non-Higgs backend omits usage.

- [ ] **Step 5: Run the focused tests and static checks**

  Run:

  ```bash
  cargo test -p higgs-bench --bin bench_decode
  cargo fmt --all -- --check
  git diff --check
  ```

  Expected: all new pure unit tests pass; no GPU initialization occurs.

- [ ] **Step 6: Detect scope and commit when authorized**

  Run `npx gitnexus detect-changes --repo higgs`, inspect that only the benchmark request/result flow changed, then—only after explicit authorization—commit:

  ```bash
  git add crates/higgs-bench/src/bin/bench_decode.rs
  git commit -m "feat(bench): report cache-cold prefill"
  ```

### Task 2: Register and document the 27B Escha benchmark contract

**Files:**
- Modify: `benchmarks/models.toml`
- Modify: `crates/higgs-bench/src/models.rs` new private `tests` module
- Modify: `docs/benchmarking.md:13-28, 339-370`

**Interfaces:**
- Add one manifest entry with `key = "escha-qwen3.8-27b-w2"`, `path = "EschaLabs/Qwen3.8-27B-Escha-W2"`, `context = 32768`, and tags `large`, `dense`, and `escha`.
- Set `approx_size_gb` to the one-decimal disk size from `du -sk "$ESCHA_MODEL_DIR" | awk '{printf "%.1f\\n", $1 / 1048576}'`; the manifest must never contain a guessed size.
- The public checkpoint ID remains `ModelInfo.path`; `--request-model escha-27b` selects the operator’s configured server alias.

- [ ] **Step 1: Run GitNexus impact analysis and write the manifest regression**

  Run upstream impact for `load_manifest` before adding a test around its manifest contract. Add this failing test to `models.rs`:

  ```rust
  #[test]
  fn repository_manifest_registers_qwen38_escha_as_dense() -> anyhow::Result<()> {
      let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
          .join("../..").join("benchmarks/models.toml");
      let model = load_manifest(&path)?
          .find_by_key("escha-qwen3.8-27b-w2")
          .expect("Qwen3.8 Escha benchmark entry");
      assert_eq!(model.path, "EschaLabs/Qwen3.8-27B-Escha-W2");
      assert!(model.approx_size_gb > 0.0);
      assert!(model.tags.iter().any(|tag| tag == "dense"));
      assert!(model.tags.iter().any(|tag| tag == "escha"));
      Ok(())
  }
  ```

- [ ] **Step 2: Verify the test fails before the manifest entry exists**

  Run:

  ```bash
  cargo test -p higgs-bench repository_manifest_registers_qwen38_escha_as_dense
  ```

  Expected: FAIL with `Qwen3.8 Escha benchmark entry`.

- [ ] **Step 3: Add the measured manifest metadata and documentation**

  Measure the local checkpoint’s on-disk size using the exact command in this task’s Interfaces block, then add the entry using that value. Update `docs/benchmarking.md` so it states all of the following explicitly:

  - `prompt_tokens` is the terminal server usage count, and `prefill_tokps` is `prompt_tokens / TTFT` for a loopback request, not an in-process kernel-only timing.
  - `bench_decode` forces `reasoning.effort = "none"` and `speculation = "none"` for a baseline.
  - `--cache-buster true` prefixes every warmup/trial before the shared prompt; it is required for published cold-prefill measurements.
  - `HIGGS_BENCH_API_KEY` is the optional safe authentication mechanism, and it is never serialized.
  - `--request-model` is the server routing alias while the manifest path remains the public checkpoint identity.

  Include this complete reproducible 1K command, with the same command repeated only by changing `--prompt-tokens` to `4096` and `8192`:

  ```bash
  export ESCHA_MODEL_DIR="$HOME/.cache/lm-studio/models/EschaLabs/Qwen3.8-27B-Escha-W2"
  export HIGGS_BENCH_API_KEY='the-local-server-key-if-configured'
  cargo run --release -p higgs-bench --bin bench_decode -- \
    --host 127.0.0.1 --port 9011 --model escha-qwen3.8-27b-w2 \
    --request-model escha-27b --tokenizer-dir "$ESCHA_MODEL_DIR" \
    --prompt-tokens 1024 --cache-buster true --max-tokens 128 \
    --warmup 1 --trials 3 --temperature 0 --format json
  ```

- [ ] **Step 4: Run focused verification**

  Run:

  ```bash
  cargo test -p higgs-bench repository_manifest_registers_qwen38_escha_as_dense
  cargo test -p higgs-bench --bin bench_decode
  cargo fmt --all -- --check
  git diff --check
  ```

  Expected: the manifest is parsed, the entry has a nonzero measured size, and all benchmark unit tests pass without GPU work.

- [ ] **Step 5: Detect scope and commit when authorized**

  Run `npx gitnexus detect-changes --repo higgs`. After explicit authorization, commit:

  ```bash
  git add benchmarks/models.toml crates/higgs-bench/src/models.rs docs/benchmarking.md
  git commit -m "docs(bench): register qwen38 escha"
  ```

### Task 3: Collect a reproducible 27B result set with no GPU contention

**Files:**
- Create: `target/bench-results/bench_decode/*.json` generated artifacts; copy only the selected raw JSON into the evidence repository in Task 4.
- Read: `target/qwen38-escha-bench/higgs.log`, the three raw benchmark JSON files, and the exact `git rev-parse HEAD` used for the release build.

**Interfaces:**
- Server address is `127.0.0.1:9011`; benchmark request model is `escha-27b`; manifest model is `escha-qwen3.8-27b-w2`.
- Produce exactly three JSON artifacts: one context run each at requested `1024`, `4096`, and `8192` prompt tokens. Each run already contains both its prefill/TTFT and its 128-token decode measurements.
- Every published context run has one excluded warmup and three measured 128-token greedy trials with non-null `prompt_tokens` and `prefill_tokps`.

- [ ] **Step 1: Build the exact binary and establish an idle-GPU gate**

  From the isolated worktree, run:

  ```bash
  cargo build --release -p higgs -p higgs-bench
  if pgrep -x higgs >/dev/null; then echo "another Higgs process is active"; exit 1; fi
  git rev-parse HEAD
  ```

  Expected: release binaries build, no Higgs process exists, and the displayed SHA is saved alongside the result artifacts.

- [ ] **Step 2: Start one loopback server using the normal wired-memory configuration**

  Set `HIGGS_CONFIG` to the existing configuration containing the local 27B model named `escha-27b` and `local.raise_wired_limit = true`. Start the server as one background process and retain its PID:

  ```bash
  BENCH_LOG="$PWD/target/qwen38-escha-bench/higgs.log"
  mkdir -p "$PWD/target/qwen38-escha-bench"
  load_started="$(date +%s)"
  RUST_LOG=info ./target/release/higgs --config "$HIGGS_CONFIG" serve \
    --host 127.0.0.1 --port 9011 >"$BENCH_LOG" 2>&1 &
  HIGGS_PID=$!
  until rg -q 'Engine ready' "$BENCH_LOG"; do
    if ! kill -0 "$HIGGS_PID" 2>/dev/null; then tail -n 120 "$BENCH_LOG"; exit 1; fi
    sleep 1
  done
  load_finished="$(date +%s)"
  printf 'cold_load_seconds=%s\\n' "$((load_finished - load_started))" | tee "$PWD/target/qwen38-escha-bench/load.txt"
  ```

  Confirm the log reports the six-element dense Escha header and derived affine Q4/Q8 layout. Confirm no second server or model loader is started.

- [ ] **Step 3: Smoke-test a real non-empty completion before timing**

  Use one authenticated loopback chat request with `reasoning.effort = "none"`, `speculation = "none"`, temperature `0`, and a small `max_tokens` value. Require a successful response with non-empty assistant content, then retain only the status/content-length fact in the run notes—not the local API key.

- [ ] **Step 4: Run cache-cold context and decode measurements sequentially**

  Run the Task 2 command three times with `--prompt-tokens 1024`, `4096`, and `8192`, retaining each persisted JSON path. The generated long-prompt mode appends the fixed 200-word continuation instruction from Task 1 so every measured request has enough output work for a 128-token decode interval.

  Reject any artifact where a measured trial has `prompt_tokens: null`, `prefill_tokps: null`, fewer than 128 completion tokens, `decode_tokps <= 0`, an HTTP error, or a server log line showing a cache-resident reuse. Preserve the three per-trial values and medians; do not aggregate runs from differing power/thermal conditions.

- [ ] **Step 5: Stop the server and preserve raw evidence**

  Stop only the PID created in Step 2, then verify there is no process:

  ```bash
  kill "$HIGGS_PID"
  wait "$HIGGS_PID"
  if pgrep -x higgs >/dev/null; then echo "Higgs did not stop"; exit 1; fi
  ```

  Copy the selected three JSON files and `load.txt` into a dedicated review directory. Record the exact nightly SHA, macOS version, RAM, GPU description, low-power setting, trial count, and benchmark command line. Do not publish values unless each artifact passes Step 4’s rejection rules.

### Task 4: Publish the 27B evidence route and update Higgs’ README from raw measurements

**Files:**
- Modify: `README.md:40-44, 129-135, 182-184`
- Modify: cloned `dusterbloom/escha-mlx-evidence/index.html` footer only
- Create: cloned `dusterbloom/escha-mlx-evidence/qwen3.8-27b/index.html`
- Create: cloned `dusterbloom/escha-mlx-evidence/qwen3.8-27b/results.json`

**Interfaces:**
- The page imports `../styles.css` and `../script.js`; it introduces no stylesheet, script, external image, or font change.
- Its shared-asset and parent-page paths use `../` so the original 35B homepage continues to render unchanged.
- `results.json` is a redacted selected copy of Task 3’s raw results plus `cold_load_seconds`, `higgs_commit`, host/power notes, and the command contracts; it contains no local path, hostname, or API key.

- [ ] **Step 1: Prepare the two clean working directories and verify the public-page baseline**

  Use the worktree workflow to create a Higgs branch from the exact tested nightly SHA. Clone the evidence repository to `/private/tmp/escha-mlx-evidence-qwen38`. In the evidence clone, verify the original homepage remains a 35B-A3B MoE page, `styles.css` and `script.js` exist, and `git status --short` is empty before edits.

- [ ] **Step 2: Write the 27B companion page from measured data only**

  Create `qwen3.8-27b/index.html` with the existing semantic structure and classes (`masthead`, `hero`, `hero-panel`, `section reveal`, `section-head`, `proof-grid`, `chart-card`, `footer`). Its navigation targets only page-local anchors: `#compatibility`, `#compare`, `#results`, `#reproduce`, and `#limits`.

  Use this exact content contract:

  - Hero: identify `EschaLabs/Qwen3.8-27B-Escha-W2` as dense Qwen3.8 Escha-W2 on Apple Silicon; headline values are only the Task 3 cold-load time, measured 1K prompt tok/s, measured 128-token decode tok/s, and 32K context support.
  - Compatibility: state that the v2 six-element `escha_config` header is accepted; full int8 weight/scale pairs remain Q8; trellis code projections use the affine Q4 conversion layout. Link both the exact Higgs commit and the Hugging Face model card.
  - Comparison: table with `35B-A3B` = MoE / native trellis experts / seconds-scale native load, and `27B` = dense / affine Q4 expansion at load / measured Task 3 cold-load time. Do not copy the 35B’s 11.16 GB, 6.2 s, or fidelity claims.
  - Results: table with exact per-frontier median prompt tokens, TTFT, prompt tok/s, and decode tok/s from each context JSON. Include the three raw trial values in a details table or `<pre>` and link `results.json`.
  - Reproduce: show the Task 2 benchmark command, adding `HIGGS_BENCH_API_KEY` only as an environment-variable name. State that the server runs loopback-only with the operator’s existing `raise_wired_limit` configuration and that cache-buster prefixes begin every trial.
  - Limits: no quality evaluation, no cross-runtime comparison, one host/checkpoint/power state, and server-observed prompt tok/s rather than isolated kernel timing.

  Set the hero date to the actual measurement date, use exact `git rev-parse HEAD` in the nightly commit URL, and give the page a footer link back to `../`.

- [ ] **Step 3: Add reciprocal discovery and a redacted raw-results artifact**

  Add one footer link from the original `index.html` to `qwen3.8-27b/`; do not put a route link in the original `.nav`, because `script.js` treats every `.nav a` as a same-page CSS selector. Build `qwen3.8-27b/results.json` by copying only the three selected result envelopes and the `load.txt` measurement into an object whose local paths remain redacted and whose `host.hostname` remains `redacted`.

- [ ] **Step 4: Update the Higgs README with model-specific wording**

  Replace the first Escha bullet with two concise statements:

  ```markdown
  - **EschaLabs `eschamoe` MoE trellis checkpoints, read natively on Metal.**
    `Qwen3.6-35B-A3B-Escha-W2` retains native trellis experts; its measured
    compact residency, load, and throughput evidence is in
    [the 35B case study](https://dusterbloom.github.io/escha-mlx-evidence/).
  - **Qwen3.8-27B-Escha-W2 dense checkpoints.** Higgs accepts the v2 Escha
    layout and derives the Q4/Q8 runtime layout from checkpoint tensor pairs.
    Measured server prefill, TTFT, decode, and load evidence is in
    [the 27B case study](https://dusterbloom.github.io/escha-mlx-evidence/qwen3.8-27b/).
  ```

  Update the local-model support paragraph to distinguish the native 35B MoE path from the supported dense 27B affine conversion path. Keep the Apple Silicon notes’ native-trellis performance figures explicitly scoped to the 35B release; add no unmeasured 27B number outside the new evidence page.

- [ ] **Step 5: Verify page/link correctness and source scope**

  In the evidence clone, run `git diff --check` and inspect all `href`/`src` values with:

  ```bash
  rg -n 'href=|src=' index.html qwen3.8-27b/index.html
  ```

  Confirm the new page uses `../styles.css`, `../script.js`, `../`, the 27B model-card URL, the exact commit URL, and `results.json`; confirm the original `.nav` contains only fragment links. In Higgs, run:

  ```bash
  cargo test -p higgs-bench --bin bench_decode
  cargo test -p higgs-bench repository_manifest_registers_qwen38_escha_as_dense
  cargo fmt --all -- --check
  git diff --check
  npx gitnexus detect-changes --repo higgs
  ```

  Expected: no formatting errors, all non-GPU checks pass, and GitNexus reports only the benchmark/docs symbols and flows anticipated by Tasks 1–2.

- [ ] **Step 6: Commit and publish only when explicitly authorized**

  After user authorization, make independently reviewable commits:

  ```bash
  # Higgs worktree
  git add README.md
  git commit -m "docs: add qwen38 escha evidence"

  # Evidence repository clone
  git add index.html qwen3.8-27b/index.html qwen3.8-27b/results.json
  git commit -m "docs: add qwen38 escha evidence"
  ```

  Before each commit, run that repository’s `git status --short`, inspect the staged diff, and for Higgs re-run GitNexus change detection. Push only the specifically authorized branches, verify the Pages deployment returns HTTP success for both routes, and then re-check the README links.

## Plan Self-Review

- **Spec coverage:** Task 1 provides strict baseline metrics without unsafe direct loading; Task 2 registers and documents reproducibility; Task 3 supplies cold-load, 1K/4K/8K prefill/TTFT/decode raw evidence; Task 4 creates the 27B page, updates the existing evidence homepage and Higgs README, and validates links. The dense/MoE distinctions, no-fidelity claim, single-GPU rule, and no-cached-results rule are global constraints and page limits.
- **Placeholder scan:** No unresolved markers remain. The disk-size and measured performance values are deliberately generated by explicit commands because inventing them would violate the evidence requirement.
- **Interface consistency:** `bench_decode` owns `request_body`, `cache_busted_prompt`, and `prefill_tokps`; Task 2’s documented flags map directly to Task 1. Task 3’s four JSON envelopes feed Task 4’s `results.json` and tables without remeasurement or manual interpolation.
