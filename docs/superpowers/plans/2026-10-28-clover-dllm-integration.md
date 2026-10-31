# Clover-35B-A3B-Preview (block-diffusion dLLM) integration — design

Status: design, not started. Authored 2026-10-28 (design by Fable, spot-checked against code).
Line numbers refer to `main` as of 2026-10-28.

## Handoff (2026-10-30) — start here

Branch `feat/clover-dllm` (pushed to `fork`), 7 commits on `main` + this doc update:
`1c4b4e9f9` P0 load/gather · `77286742a` conv layout · `0b85ee65c` P2 preview · `68578a964` docs ·
`5645a7b2a` P3 LowConfidence diffusion · `c93035808` opt-in append-stable MMA gather ·
`66d0efa14` opt-in native gather.

### Measured on M4 10-core / 32 GB (Low Power Mode OFF, no other Metal clients — both matter:
Low Power Mode cut decode ~3x; a LiquidAgent llama-server on :8180/:8181 skewed earlier runs)

| path | prefill tok/s @530/2026/8021 | AR decode tok/s | append-stable |
|---|---|---|---|
| default `gather_qmv` | 219 / 232 / 220 | 38-45 | yes |
| `HIGGS_MOE_GATHER_KERNEL=mma` | 311 / 373-392 / 347-356 | 22-24 | yes |
| `HIGGS_MOE_GATHER_KERNEL=native` (mlx-lm style) | 346 / 399 / 367 | 41-45 | NO (warm may differ from cold by rounding) |

Diffusion (opt-in `speculation: "diffusion"`, faithful to the reference, parity-tested): prose
4.7-11.5 tok/s, code 7.9-20, JSON 11.6-25, math 12-27, enumerative (count/repeat) 46-68 tok/s.

### Why diffusion does not beat AR here (two independent reviews, Fable + Codex, plus experiments)
- Tokens/pass is a model property: ~1 on prose, ~1.5-2 on code, 2.3-3 structured, 8-20 enumerative.
  AR greedy top-1 >0.95 on only 29-39% of prose tokens. Coreset tightening (top_p 0.5-0.7) does
  not cut pass time and corrupts output.
- Masked rows route independently: a 32-row pass touches ~160/256 experts (~12 GB), already ~84%
  of achievable bandwidth. AR runs at ~80 GB/s ≈ 92% of achievable (78 GB/s measured GEMV), so
  there is no idle bandwidth. MoE cost ≈ max(10 us/routed slot/layer, distinct experts × 1.77 MB
  / 78 GB/s) + ~43 ms/pass non-MoE (≈35 ms launch overhead vs AR).
- Parity needs tokens/pass ≥ 7.3 at bs8 today (≥ 4.3 with ideal kernels); measured bs8: prose
  1.05, code 2.05, math 3.0, count 7.6.
- Artifacts: `~/AI-Models/clover-fixture/research/` (diffreview/, space/ results, bench scripts,
  reference Python, test configs); fixture + 4-layer slice in `~/AI-Models/clover-fixture/`;
  failed qmv-tiled kernel attempt in `~/AI-Models/clover-fixture/kernel-attempt-qmv-tiled/`.

### Open decisions / next steps (ranked)
1. Decide the default gather mode (`qmv` stable vs `native` fast). Consider `native` as default
   with stable mode for retained sessions only.
2. Diffusion defaults: `clover_block_size = 8`, threshold 0.8-0.9 (2-2.7x over bs32/0.95).
3. Content-adaptive AR<->diffusion switching (only real win: enumerative spans 46-68 vs 46 AR).
4. Pass-cost cuts: fuse GDN preview taps / compile elementwise chains (-25-35 ms), freeze
   resolved rows, stream resolved left prefix per pass (latency).
5. ~~Cheapest decisive experiment~~ DONE 2026-10-30, gate PASSED (self-spec path stays open).
   Kapitsa 0.2.24 on this M4 (Low Power off, no other Metal clients), same pack, greedy, thinking
   off, 256 tok, warm rep (`research/kapitsa_spec.sh`, results in `research/kapitsa_spec/`):

   | prompt | `spec` tok/s (tok/cycle) | Kapitsa `diffusion` bs32/0.95 tok/s (tok/pass) | higgs AR | higgs diffusion |
   |---|---|---|---|---|
   | code (2) | 71.8 / 72.1 (5.31) | 22.9 (2.29) / 16.3 (1.57) | 41-45 | 7.9-20 |
   | prose (2) | 41.6 / 45.4 (3.04 / 3.43) | 9.6 (0.93) / 7.9 (0.77) | 41-45 | 4.7-11.5 |
   | count 1..200 | 102.2 (7.97) | 88.5 (10.67) | 41-45 | 46-68 |

   Per-cost: a self-spec cycle (8-row denoise draft + 8-row causal verify) is **74-78 ms**; a
   32-row denoising pass is **~96-100 ms** (higgs: ~255 ms). So the "no idle bandwidth" bound
   above is wrong for Kapitsa's kernels: it runs a 32-row pass ~2.6x faster on the same M4, and
   an 8+8 cycle at ~3.5x a higgs AR step.

   A/B (`research/kapitsa_ab.sh`, code1 prompt, greedy, 256 tok; ms/pass includes commit passes):

   | engine / mode | rows | coreset | passes | ms/pass (or /cycle) |
   |---|---|---|---|---|
   | Kapitsa spec | 8+8 | on / off | 48 / 44 cycles | 73.7 / 79.2 |
   | Kapitsa diffusion | 32 | on / off | 112 / 122 | 99.5 / 114.0 |
   | Kapitsa diffusion t0.95 | 8 | on / off | 117 / 112 | 48.6 / 52.6 |
   | higgs diffusion t0.95 | 8 | on | 113 | **106** (2 runs: 12.0 / 12.3 s) |
   | higgs AR | 1 | - | 255 steps | **21.4** (46.6 tok/s) |

   - The coreset is worth only 7-13% in Kapitsa. It is not the gap. higgs is ~2.1x slower per
     8-row pass. With `gather_qmv` each routed slot re-reads its expert's weights, so the coreset
     cannot save bytes in higgs. Kapitsa reads each distinct expert once per pass. That is option D.
   - higgs bs8 output (227 tok, 113 passes) matches Kapitsa with **coreset off** (227, 112), not
     coreset on (256, 117). The coreset call site looks wired (`forward_preview_block` ->
     `forward_routed(.., coreset_top_p)`, pack has `moe_block_coreset_enabled: true, top_p 0.9`),
     so this is either a reference-semantics difference or coincidence. Unresolved.
   - `HIGGS_PROFILE=1` halves AR decode (20-24 vs 46.6 tok/s). Never benchmark with it on.
   - **Ceiling for P4 self-spec in higgs.** Today: 106 ms draft + an 8-row verify of similar cost
     ≈ 200 ms/cycle. At 5.31 tok/cycle on code that is ~26 tok/s, *below* AR's 46.6. At Kapitsa's
     cycle cost (74 ms) it is ~72 tok/s (1.55x AR on code, ~0.9-1x on prose). So P4 is only worth
     building after a multi-row gather kernel (option D) brings an 8-row pass to ≲ 50 ms.
6. Deterministic small-tile MMA gather kernel (option D) only if prefill/commit gains justify it.

Pre-existing issues on `main` (not from this work): higgs-models unit tests don't compile (mlx-rs
drift); release-profile higgs-models test binaries fail to link; `logprobs: true` panics; clippy
errors in higgs-models/higgs; 4 failing higgs lib tests; 2 failing higgs-engine batch tests.
Uncommitted local edits to AGENTS.md / CLAUDE.md and the bilinear-probe files are the user's.

## Status (2026-10-29)

| Phase | State | Commit | Notes |
|---|---|---|---|
| P0 load + AR | done | `1c4b4e9f9` | Unmodified Kapitsa pack loads; AR 42-45 tok/s on M4 (was 5.3). Generic MoE gather now `gather_qmv` (append-stable, no padding); `HIGGS_MOE_GATHER_PAD=1` restores padded rhs. Norm rule uses median layer `input_layernorm` mean (final/q/k norms are not usable signals: raw Clover final norm 1.73, q/k 0.54). |
| P1 DFlash | gate FAILED | `77286742a` (conv layout fix) | Drafter acceptance 1.5-5.9/round; DFlash 7-20 tok/s vs AR 44. Root cause: multi-row MoE gather has no efficient deterministic path (`gather_qmv` re-reads weights per row; padded rhs pays 1024 rows). Ceiling with an ideal kernel ~1.2-1.4x AR on this M4. |
| P2 preview forward | done | `0b85ee65c` | `Qwen3NextCausalLM::forward_preview_block`; parity vs HF reference on 4-layer slice: top-1 100%, worst cosine 0.9955, cache bit-identical. |
| P3 LowConfidence diffusion | done | (this commit) | `clover::block_decode` + `speculation = "diffusion"` (request or `generation_defaults`; opt-in, never `auto`), pack penalties as diffusion-only defaults, `clover_block_size`/`clover_threshold`. Real Clover M4: 3.5-8.8 tok/s (1.6-2.5 positions/pass, ~255 ms per 32-row pass; same MoE multi-row cost as P1) vs AR ~35 tok/s; coherent on 3 prompts. Parity: block loop on reference logits exact (greedy, pack penalties, ramps); slice end-to-end exact for greedy@0, other cases flip on sub-ulp bf16 ties (see `clover_preview_parity.rs`). |
| MoE gather MMA family | opt-in (`HIGGS_MOE_GATHER_KERNEL=mma`) | (this commit) | Verified on MLX 0.31/M4: `gather_qmm_rhs` == steel `gather_qmm` tiles bit-for-bit (tile height 18/24/32/64, row position, row alone), so an MMA-only policy is append-stable. Not the default: Clover prefill 230 -> 350-390 tok/s but AR decode 44 -> 24 tok/s and diffusion 5-11 -> 3-11 tok/s (decode pays a 32-row tile per routed row: 0.61 vs 0.18 ms/layer). `forward_gather_fused` no longer passes `sorted=true` unconditionally (size switching under `HIGGS_MOE_FFN_GATE_UP`). A fast stable decode still needs option D. |
| Kernel (option D) | deferred by user | | Deterministic affine-4bit gather GEMM; prerequisite for fast P1/P3/P4. |
| Fast rows (8-row passes) | done | (this commit) | 8-row preview profile (M4): MoE 26.5 / GDN 22.6 / attention 15.7 / rest 15.3 ms. Dense projections ran at ~3x the one-row time (MLX qmv and `crossrow_qmv` re-read weights), and preview/commit RoPE padded every call to 1,024 rows. New `qmm8` (weight-once 8-row affine-4bit MMA kernel, Kapitsa's `verify8` technique) + plain RoPE inside `with_fast_rows` (previews and diffusion commits only; request-local cache). Preview 79.9 -> 56.5 ms, causal commit 80.0 -> 59.9 ms; bs8/t0.95 code 18.9 -> 25.6-26.4 tok/s at the same 112-113 passes. Parity divergences unchanged. Option D at 8 rows is worth only ~7 ms (MoE 26.6 vs ~19 floor); compiling the GDN mixer gained nothing. |
| P4 self-spec | done (opt-in) | (this commit) | `speculation = "self_spec"`: preview `[pending, mask...]` drafts, fast-rows tape verify on a request-local cache, exact acceptance vs the verify argmax, tape rollback; draft rows adapt 8 <-> 4 on the accepted fraction (hysteresis 0.5 / 0.8). Greedy without penalties only. M4, 256 tok greedy (AR 45.4): code 49.6 / 43.5 (5.93 / 5.10 tok/cycle), prose 26.5 / 29.0 (2.71 / 3.17), count 67.2 (7.97). Cycle cost ~120 ms (8 rows), ~100 ms (4 rows) vs Kapitsa 74 / 60 — the kernel gap. Output is not bit-identical to AR: the multi-row verify rounds differently (probe: an exact bf16 tie 21.125/21.125 in AR became 21.0/20.625 in verify), so near-ties flip after 5-100 tokens; `CanonicalS1` verify did not restore identity and was slower (code 37 tok/s). Kapitsa sweep on this M4 (`research/kapitsa_accept.sh`): exact acceptance code 58.5-76.9 at 8 rows, prose 49.4 at 4 rows (> AR), count 102.5; threshold acceptance similar; draft 6 is invalid (must divide 32). |
| 8-row MoE (`moe8`) | done | (this commit) | Weight-once distinct-expert kernels (GPU plan -> gate/up+SwiGLU -> down; Kapitsa `moe8_*` technique), fast rows only. Preview 55.2 -> 44.1 ms, fast causal 58.8 -> 56.5, cycle 114 -> 100.6 ms. Self-spec code 50.7/44.2 -> 58.1/51.8, prose 27.1/29.6 -> 30.4/31.4, count 68.4 -> 72.3 tok/s (AR 46.5, AR output byte-identical). Verify gains little: ~28-32 distinct experts/layer and the 8-row MMA is compute-bound (~82% ALU peak), crossover vs gather_qmv ~31 experts. Open: scalar path for experts used by <=2 rows (~2-3 ms/verify). |
| GDN fast-rows kernels (`gdn_rows`) | done | (this commit) | Fused preview mixer (conv+norms, block-softmax memory) and verify prep + fold rollback (Kapitsa `clover_gdn_*` / `gdn_*_verify8` technique), fast rows only, `HIGGS_GDN_ROWS=0` off switch. Preview 44.9 -> 41.3 ms, verify 58.1 -> 56.7, rollback 3.2 -> 1.9; cycle ~104 -> ~99 ms. Self-spec code 55.7 / 53.3, prose 32.3 / 31.1, count 75.2 tok/s (AR 46.2-46.4, byte-identical). Verify GDN is projection-bound (qmm8 ~68% of a layer), not dispatch-bound, so the planned -10 ms did not materialize. Next: qmm8 bandwidth (~66 vs ~90 GB/s, ~9 ms/cycle), then MoE. |

Fixture: `examples/clover_reference_fixture.py PACK REF_CODE_DIR OUT` (needs torch + transformers==5.19.0;
a throwaway venv works). Test: `HIGGS_CLOVER_FIXTURE_DIR=OUT cargo test -p higgs-models --test
clover_preview_parity -- --ignored` (dev profile: release-profile test binaries fail to link on main,
`___isPlatformVersionAtLeast`).

Pre-existing issues found (not caused by this work): `higgs-models` unit tests do not compile
(mlx-rs API drift, 83 errors) so new tests are integration tests via `#[doc(hidden)]` probe seams;
`logprobs: true` requests panic (`as_slice` on unevaluated array); `clippy -p higgs-models` has 156
errors; `clippy -p higgs` fails on `build.rs` shadowing; 4 `higgs` lib tests fail on main.

## Context and evidence (2026-10-28 session)

- Model: `Diffuspace/Clover-35B-A3B-Preview`, Qwen3.6-35B-A3B backbone fine-tuned for block
  diffusion (block 32, mask 151669, LowConfidence threshold 0.95, block-level expert coreset).
  Reference: `modeling_clover_moe.py`, `sampling_clover_moe.py` in the HF repo.
- Weights: `kapitsaprime/Clover-35B-A3B-Preview-MLX-4bit` (19.5 GB, sha256 verified), in
  `$HF_HOME/hub`. Flat `config.json`, `model_type: clover_moe`, raw zero-centered norms,
  PyTorch-layout conv1d, preview tensors (`conv_right_weight`, `bd_softmax_*`) in bf16.
- Kapitsa (M5 Max 128 GB): causal + Qwen3.6 DFlash2 319/332 tok/s; self-spec 214/210;
  block diffusion 46 tok/s.
- higgs today, causal path via `qwen3_5_moe` alias + nested config + norm +1 override shard
  (test dir `~/AI-Models/clover-35b-a3b-mlx4-qwenalias`): coherent output, **5.3-5.5 tok/s**
  on M4 32 GB vs 18.9 tok/s Escha-W2 (native trellis expert kernel). Not memory pressure
  (closing apps cut swap-ins 16x, no speedup). GPU eval ~175 ms/step.

## 0. Findings that shape the design (verified in code)

1. **Loader bug A is larger than first reported.** `resolve_runtime_model_args`
   (`qwen3_next.rs:11572-11597`) wraps a flat config as `{"text_config": config}` (11587);
   `load_qwen3_5_text_config_args_from_value` (12211-12488) then reads five things from the
   *outer* object that are lost for every flat config: `quantization` scalar merge (12228),
   per-path overrides (12240-12256), `tie_word_embeddings` (12265), `gate_quantization_override`
   (12324), `collect_quant_overrides` (12331), and the Hadamard `modules` manifest (12345).
2. **Clover-specific tensors are silently dropped** by `load_qwen3_5_moe_weights_fused`
   (13570-13827; unmatched keys skipped at 13648-13656). Conv weight [8192,1,4] is handled by
   `transposed_conv_weight` (6761-6784) but not by the `conv1d.forward` fallbacks (7039, 7366, 7612).
3. **Norm convention B.** Reference RMSNorm is zero-centered (`1+w`) for `input_layernorm`,
   `post_attention_layernorm`, `model.norm`, `q_norm`, `k_norm`; not for `linear_attn.norm`.
   higgs adds +1 only for dense-MTP norm keys (`qwen35_loaded_value` 13190-13203).
4. **A read-only GDN forward exists**: `GatedDeltaNet::forward_stateless` (7243-7431). Latent
   issue: it reads `cache.conv_state` as chronological (7312-7318), ignoring the decode ring
   cursor `conv_pos`; preview must use the chronological helper (6843-6881).
5. **Hybrid-cache rollback already exists**: `forward_with_taps_tape_scheduled` (10897-) records
   `GdnLayerTape`s; `replay_tape_rollback` (11209-11489) replays accepted innovations. Used by
   DFlash (`simple.rs:12305` verify, `:3058` rollback).
6. **Frozen-prefix block attention** needs a read-only K/V view; no `peek` exists yet
   (`update_and_view` mutates; cache.rs:1109-1201).
7. **MoE padding**: `forward_gather_global_sort` (6056-6192) pads every call with
   `rows < num_experts*4` to 1024 rows on the last expert (6119-6144); Escha's native path returns
   before padding. Comment cites MLX 0.30.6; workspace pins `mlx-rs` branch `mlx-0.31`, so the
   dispatch rule must be re-verified.
8. **Compatible DFlash drafter on disk**: `models--modal-labs--Qwen3.6-35B-A3B-DFlash`.
9. **Generation controls** (pack `generation_config.json`): presence 0.9, repetition 1.05,
   frequency 0, length penalty 10 @ 8192/8192, think-end bias 10 @ 4096/4096 on 248069,
   eos [248046, 248044], greedy default. Semantics in `sampling_clover_moe.py:63-153`.

## 1. Model-type integration

### 1.1 Reuse, not a new model module
Build inside `Qwen3NextCausalLM` (`qwen3_next.rs:8588-8600`). Decode algorithms go in a new
`crates/higgs-models/src/clover.rs`, mirroring `feat/nemotron-diffusion`'s
`nemotron_diffusion.rs` (`predict_block`/`commit_block`/`draft_and_verify`/`diffusion_generate`/
`linear_spec_generate`, `BlockCallback`).

Recognition touch points:
- `registry.rs:28-57` — add `"clover_moe"` to `is_supported` (+ test).
- `model_loader.rs:65-69` — `"qwen3_5_moe" | "clover_moe" => load_qwen3_5_moe_model(...)`.
- `adapter.rs` — `CLOVER_MOE` builtin (`LoadKind::CloverMoe`), before `QWEN35_MOE` in order
  (400-418); `is_exact` (425-450); `load` (608-650) same arm as `Qwen35Moe`. Test in
  `tests/adapter_registry.rs` next to line 246.
- `qwen3_next.rs:11582` — qwen3_5 branch gate must accept `clover_moe`.

### 1.2 Avoid scattered `qwen3_5_moe` string matches
Add `registry::runtime_family(model_type) -> &str` mapping `clover_moe -> qwen3_5_moe`; apply
at the two config-derivation choke points (`mlx_tuning.rs:778-786` `ModelMetadata::from_model_dir`,
`mlx_tuning.rs:341` `model_load_estimate`) plus `simple.rs:14598-14608`. Preview capability is
keyed on `AnyModel::supports_block_preview()` (`args.clover.is_some()`), not the string.

### 1.3 Durable fix for bug A (all flat configs)
In `resolve_runtime_model_args` (11581-11589), treat a flat config as its own wrapper:
```rust
let wrapped = { let mut w = config.clone(); w["text_config"] = config.clone(); w };
load_qwen3_5_text_config_args_from_value(model_dir, &wrapped)?
```
Regression test next to 27555: flat config with an 8-bit `...layers.0.mlp.gate` override ->
`resolve_gate_quant(...).bits == 8`, `shared_expert_gate` too.

### 1.4 Norm convention B, keyed on checkpoint evidence
Fused-loader pre-pass: if `model.norm.weight` and `layers.0.input_layernorm.weight` both have
`mean < 0.5`, the checkpoint stores raw zero-centered norms; add +1 for
`input_layernorm|post_attention_layernorm|model.norm|self_attn.q_norm|self_attn.k_norm`, never
`linear_attn.norm`. Log once with the measured means. Test: synthetic 1-layer checkpoint, means
0.0 -> w+1; means 1.0 -> unchanged; `linear_attn.norm` untouched.

### 1.5 Clover args and params
- `Qwen3NextModelArgs` (219-352): `clover: Option<CloverArgs>` {block_size, mask_token_id,
  coreset enabled/top_p, denoising threshold}, populated on evidence (`mask_token_id` +
  `block_size` present).
- `GatedDeltaNet` (6588-6624): optional params `conv_right_weight`, `bd_softmax_logit_scale`,
  `bd_softmax_history_bias`, `bd_softmax_history_c`; placeholders only when `clover.is_some()` so
  `ensure_all_model_params_loaded` (11891-11912) catches missing tensors. Fallback if the derive
  doesn't flatten `Param<Option<_>>`: nested `clover: Option<CloverPreviewParams>` + key remap in
  `qwen35_target_param_key` (13160).
- Normalize `conv1d.weight` to MLX `[C,K,1]` at load.

## 2. Preview forward

Entry: `Qwen3NextCausalLM::forward_preview_block(&mut self, block_ids, kv_cache: &[Option<LayerCache>]) -> logits [1,S,V]`
next to `forward_with_taps_stateless` (10739); immutable cache slice enforces non-mutation.
`AnyModel::forward_preview_block` in `lib.rs` near 1690. Layer loop = copy of 10788-10840 with:

### 2.1 GDN conv (`bd_gdn_conv1d`) — `GatedDeltaNet::forward_preview` after 7431
1. Projections as 7258-7303.
2. History via a new `chronological_conv_state_ref` (refactor of 6843-6881).
3. Causal taps: `canonical_conv_preactivation_ffi` when supported (2268-2286), else window loop
   (7343-7356); zero history for short prefixes.
4. Right conv: `for d in 1..K: out[:, :S-d] += mixed[:, d:] * conv_right_weight[:,0,d-1]` in fp32;
   SiLU; zero invalid rows.
5. q/k/v split (7370-7383); scaled q via existing `rms_norm(q, qk_norm_weight_q)`; unscaled
   `q_unit` for softmax logits; repeat Hk->Hv as 7089-7099.

### 2.2 GDN mixer (`bd_gdn_local_memory`) — plain fp32 MLX ops
`decay = compute_g_direct(...)` (6508-6513); `hist = q_scaled @ ssm_state^T * decay`;
`logits = q_unit @ k_unit^T * logit_scale`; history logit `bias + c*ln(valid/block_size)`;
softmax over concat; `y = w_hist*hist + w_local@v`; then `gdn_output_gate` (2043) + `out_proj`.
Compiled delta-rule kernels are not reusable here.

### 2.3 Full attention — `Qwen3NextAttention::forward_preview` after 5314
RoPE with `apply_qwen3_next_rope_scheduled` at `position_offset()` (8647-8690). New
`SteppingKeyValueCache::peek_view(&self)` (cache.rs after 826; dense slice / `turbo.view` 1569).
Concat prefix+block K/V, SDPA with no mask, `sigmoid_mul` gate (5307), `o_proj`.

### 2.4 MoE coreset — `FfnBlock::forward` MoE branch (8178-8203)
`coreset_allowed(logits, top_k, top_p)` per `coreset_torch`: softmax, top-k, aggregate via
one-hot matmul, normalize, sort desc, exclusive cumsum < top_p, force first; mask logits to -inf.

### 2.5 Padding interaction
A 32-row block (256 routed slots) is padded to 1024 like decode. Fix once in Phase 0.

## 3. Decode algorithms (`clover.rs`)

- **(c) Causal AR + DFlash drafter — first.** No new model code; `load_dflash_drafter_for_target`
  (`dflash.rs:2465`) + `dflash_decode` (`simple.rs:11402`). Risk: drafter taps trained on base
  Qwen3.6; measure acceptance.
- **(a) LowConfidence block diffusion — second** (validates preview vs reference). Prefill;
  per block: all-mask, up to block_size preview passes, mask-id -> -inf, `CloverPenaltyState`
  (committed-response counts; frequency -> presence -> repetition; length + think-end ramps),
  confidence transfer > threshold else argmax; emit block; stop at first EOS; commit only the
  returned tokens causally.
- **(b) Self-speculative — third.** S=8 (7 drafts): one preview pass for draft;
  `forward_with_taps_tape_scheduled` verify (lib.rs:1690); `dflash::accept_prefix` (2487);
  `replay_tape_rollback` (lib.rs:1720) for partial acceptance; `mtp::verify_targets` for sampled
  requests. Greedy output is AR-exact by construction.

## 4. Engine / API integration

- Extend `Speculation` (`lib.rs:156-184`) with `SelfSpec` and `Diffusion`; routes in
  `simple.rs:1207-1254`; `Auto` order DFlash -> SelfSpec -> MTP -> AR.
- Plug into `generate_inner` (10131, after 10184-10203) and `generate_streaming_inner` (13667)
  via siblings of `generate_dflash_inner`/`_streaming` (14814/14838) over `DflashSink`
  (14650-14678). Block-level streaming.
- Per-model config (`config.rs` near `draft_model` 525): `clover_block_size` (divides 32),
  `clover_threshold` (0..=1), `clover_draft_tokens` (draft+1 divides 32). Doctor validation
  (`doctor.rs` model loop ~573-590): ranges, no `batch = true`, warn for non-clover adapters.
  Docs: README, `docs/configuration.md`, `daemon.rs` template, `docs/models.md`.
- Prefix/radix caches: preview never writes; commits use prefill primitives -> append-stable.
  Retained-session integration deferred.

## 5. Correctness strategy (minimal tests)

Fixture generator `crates/higgs-models/examples/clover_reference_fixture.py`: 4-layer slice of
the HF reference (3 GDN + 1 FA, dequantized from the pack, CPU) dumping preview/commit hidden
states; plus a tiny random 2-layer fixture for CI.

1. `flat_config_keeps_per_path_quant_overrides`
2. `zero_centered_norms_offset_by_checkpoint_evidence`
3. `clover_preview_block_matches_reference_fixture`
4. `preview_never_mutates_hybrid_cache`
5. `coreset_allowed_matches_reference_semantics`
6. `selfspec_greedy_equals_causal_ar`
7. `diffusion_commits_only_remaining_and_stops_at_eos`
8. `#[ignore]` real-weights: diffusion coherence; self-spec == AR over 256 tokens.
Keep `append_stable_sorted_gather_matches_identical_short_tail` (17150) green.

## 6. Performance

Assumptions: M4 ~120 GB/s; affine 4-bit g64 = 0.5625 B/param.
- Active bytes/token ~1.65 GB -> floor 14 ms (70 tok/s), realistic 25-40. Measured 185 ms = ~13x
  floor.
- Padding estimate: 1016 padded rows x 3 matmuls (2048x512) -> 6.4 GFLOP/layer, 256 GFLOP/token,
  >= 90 ms/token at M4 fp16 peak. **Hypothesis, unmeasured.**
- Verify cheaply: env-gated `HIGGS_MOE_GATHER_PAD=0`, `HIGGS_PROFILE=1`, one Metal System Trace.
- Determinism-preserving fixes, in order: (A) read MLX 0.31 `GatherQMM::eval_gpu` rule;
  (B) pad each *present* expert to 4 rows (<=32 rows at decode); (C/D) custom affine-4bit sorted
  gather kernel with one reduction order.
- Mode ceilings on this M4: (a) diffusion 10-14 tok/s; (b) self-spec 50-60 ceiling, 25-35
  realistic; (c) DFlash ~110 ceiling, 40-60 realistic.

**Gates**: P0 loads unmodified pack, AR greedy unchanged, AR >= 20 tok/s after padding fix (else
stop and profile). P1 DFlash acceptance >= 4/round and >= 1.8x AR. P2 fixture parity (argmax >=
99%, bf16 tolerance), cache bit-unchanged. P3 tiny-fixture greedy == reference; real weights
coherent. P4 self-spec greedy == AR and >= 1.5x AR, else keep DFlash default.

## 7. Phases

- **P0 load + AR baseline (1-2 sessions)**: registry/adapter/loader fixes, CloverArgs/params,
  norm evidence, conv layout, padding measurement + fix. GitNexus impact on
  `forward_gather_global_sort` (HIGH, 8 callers) and `resolve_runtime_model_args` first.
- **P1 DFlash on Clover causal (0.5 session)**: config + measurement.
- **P2 preview forward (2 sessions)**: GDN/attention/coreset preview, `peek_view`,
  `forward_preview_block`, fixture script, tests 3-5.
- **P3 LowConfidence diffusion (1-2 sessions)**: `clover.rs`, `Speculation::Diffusion`, engine,
  config, doctor, docs, test 7.
- **P4 self-spec (1-2 sessions)**: test 6 + real-weights check.
- **P5 later**: retained sessions, batch>1, compiled preview, K/V concat avoidance, custom gather.

**Risks**: MLX 0.31 dispatch rule vs 0.30.6 comment; `Param<Option<Array>>` derive; bf16 vs fp32
conv parity; drafter acceptance on Clover; `history_c*ln(S/32)` only exercised at S<32; full
35B reference not runnable on 32 GB (use 4-layer slice).

**Open questions**: (1) `Speculation` variants vs separate `decode_mode`; (2) `auto` ordering;
(3) adopt pack penalties as Clover defaults (changes greedy output vs plain AR); (4) torch
available for fixtures; (5) appetite for a custom gather-QMM Metal kernel; (6) MLX 0.31 source
access; (7) diffusion-mode logprobs.
