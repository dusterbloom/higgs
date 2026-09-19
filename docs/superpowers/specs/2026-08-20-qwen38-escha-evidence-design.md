# Qwen3.8-27B Escha Evidence Design

## Goal

Publish an evidence-backed public case study for
`EschaLabs/Qwen3.8-27B-Escha-W2`, update Higgs' README to link it, and record
reproducible prefill and decode measurements on the local Apple Silicon host.

## Scope

- Add a Qwen3.8-27B case study at `/qwen3.8-27b/` in the existing public
  `dusterbloom/escha-mlx-evidence` GitHub Pages repository.
- Reuse that repository's stylesheet, script, evidence-page information
  hierarchy, navigation, provenance language, and footer links.
- Update Higgs' README with separate 35B MoE and 27B dense Escha-W2 evidence
  links and accurate model-specific support wording.
- Measure and publish only observed model-load, prefill, TTFT, and decode
  results for the current Higgs nightly build.

## Non-goals

- Do not describe Qwen3.8-27B as an MoE model.
- Do not claim the 35B model's native trellis residency or six-second load
  behavior for the 27B checkpoint. Its dense trellis projections are converted
  to affine tensors at load time.
- Do not claim weight-fidelity or quality results without a dedicated measured
  comparison against an appropriate base model.
- Do not change model loading, quantization, caching, or inference behavior.

## Public page design

The existing `escha-mlx-evidence` Pages repository remains the single public
home for Higgs' Escha evidence. The new `/qwen3.8-27b/` route has its own
`index.html` and uses the shared `../styles.css` and `../script.js` assets.

It presents a dense-model case study with four sections:

1. **Compatibility** — Higgs accepts the v2 six-element dense `escha_config`
   header, derives Q8 only from complete int8 pairs, and keeps all trellis
   projections on the affine Q4 conversion layout. The page links the exact
   Higgs nightly commit and model card.
2. **What differs from 35B** — a small comparison table makes the architecture
   and execution mode explicit: 35B-A3B is MoE with native trellis experts;
   27B is dense and currently expands to affine Q4 during CPU-bound load.
3. **Measured results** — a table gives cold-load wall time, 1K/4K/8K prefill
   throughput, TTFT, and 128-token greedy decode throughput. Each measurement
   declares trial count, thinking/speculation mode, cache policy, model name,
   Higgs commit, and host/power state.
4. **Reproduce and limits** — commands use a private loopback server and
   checked-in `higgs-bench` binaries. The page states that values are
   machine-local rather than cross-runtime comparisons.

## README design

Replace the 35B-only Escha wording with a compact model-specific statement:

- 35B-A3B: MoE, native trellis experts, compact resident state and fast load.
- Qwen3.8-27B: dense Escha-W2, supported through the Qwen3.5 dense adapter,
  with correctness covered by checkpoint-derived Q4/Q8 layout detection.

The README links the existing 35B evidence page and the new 27B route. It
contains only measured numbers that also appear in the public evidence page.

## Measurement protocol

1. Build the exact Higgs nightly revision in release mode.
2. Start only one Higgs process on a private loopback port, with the 27B model
   and normal `[local]` wired-memory configuration. Confirm no competing Higgs
   process uses the GPU.
3. Record cold load from process start to `Engine ready` in the server log.
4. Run `higgs-bench bench_frontier` at 1K, 4K, and 8K contexts to report
   incremental prefill tok/s and the one-token decode probe. Use unique prompt
   material or cache-off behavior so prefix-cache reuse cannot supply the
   result.
5. Run `higgs-bench bench_decode` with temperature 0, thinking disabled,
   speculation disabled, a 128-token request, one excluded warmup, and three
   measured trials. Publish median TTFT and median decode tok/s plus per-trial
   values.
6. Stop the private server before any other model is loaded. Do not publish a
   number if a run is short, hits an error, uses a cached prompt, or has an
   unstable thermal/power condition.

## Validation

- Build Higgs release binary.
- Keep the existing two Escha layout regression tests green.
- Verify the 27B server returns a non-empty completion before timing it.
- Inspect the generated public page locally and verify its 35B links, 27B
  model-card link, commit link, and reported values against raw benchmark JSON.
- Re-run the README link check after publishing the new public route.
