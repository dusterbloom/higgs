# Bonsai prefill: baseline and bilinear probe

Date: 2026-10-07. Status: **probe built and CPU oracle passed; GPU timings pending**.

The Mac is running a separate `spotlight-memoru` MPS training sweep. Its
32-pair condition finished and its 64-pair condition started immediately.
Clean timings require waiting for that sweep or explicit authorization to
pause and resume it. No training process has been modified.

## Existing measurements

The September 20 results use the same base M4 MacBook Pro, 32GB, and
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`:

- 4096-token prefill, chunk 1024: **81.774s TTFT, 50.09 prompt tokens/s**,
  CV 0.232%, cache bypass and speculation disabled.
- Gate/up and down were attributed **59.6% of prefill time** in the approved
  tiled-QMM design. Their acceleration can materially change total latency.
- The previous tiled ternary kernel passed its correctness gate but delivered
  only 0.401–0.447x stock throughput. It is not a useful starting candidate.
- Chunks 768–1536 tied, and natural ternary sparsity had no empty 8x8 or 16x16
  blocks. Chunk tuning and simple block skipping are already screened.

Sources: [existing results](2026-09-20-ternary-prefill-experiments.md),
[prior design](../specs/2026-09-20-tiled-ternary-qmm-design.md).
The old raw benchmark directory is absent, so these are committed historical
measurements rather than newly reproduced results.

## What is ready

`crates/higgs-models/examples/prefill_bilinear_probe.rs` is a standalone
example. It compares stock FP32 Q2, FP16-cast stock Q2, transient FP32/FP16
dense GEMM, and one/two-level Strassen. A second phase retains one dequantized
FP16 projection as an optimistic control. Casts, dequantization, block sums,
output assembly and forced evaluation are timed for transient paths.

Weights and scales come from the cached model's real layer-0 gate/up and down;
activations are deterministic synthetic data. Shapes are gate/up
`N=34816,K=5120`, down `N=5120,K=17408`, at `M=512,1024`.

The checkpoint has FP32 normalization weights. MLX RMSNorm promotes the
input to the common input/weight dtype; Higgs and MLX Hadamard preserve that
dtype. Thus FP32 Q2 is the production prefill comparison. FP16 paths change
precision and require subsequent real-model validation.

The independent small f64 oracle first detected the missing implementation
(`got=0, want=2.71875`), then passed for recursion depths one and two on CPU.
Rust formatting, Python syntax and fixture offsets/dtype byte counts pass.
The executable builds using the repository's documented extra Xcode runtime
link argument. No production dispatch was changed.

The nine-fourths manuscript gives an asymptotic existence construction,
without a finite rank decomposition to execute here. **Strassen is a control,
not an implementation or empirical test of that new construction.** A loss
by this control would not disprove practical future uses of the theorem.

## All-fields follow-up

The [all-fields fork](https://github.com/selanavot/matrix-multiplication-all-fields)
exports the same 9/4 arithmetic exponent bound over every field, including
rationals and finite fields. Its
[cost theorem](https://github.com/selanavot/matrix-multiplication-all-fields/blob/main/lean/OAI/LinearAlgebra/MatrixMultiplication/AuxiliarySeparation/Main.lean)
remains existential; the README expressly supplies neither practical
constants nor an effective program generator. The
[descent proof](https://github.com/selanavot/matrix-multiplication-all-fields/blob/main/lean/OAI/LinearAlgebra/MatrixMultiplication/Arithmetic/FieldDescent.lean)
pays an extension-dimension-squared factor, fixed asymptotically but potentially
important at finite sizes. No new executable candidate is added to the probe.

Complex arithmetic already admits a fixed-cost simulation by real arithmetic,
so the earlier complex restriction was not an exponent-level obstruction to
real-valued prefill. The newly covered positive-characteristic fields offer a
research route for exact modular/fixed-point computation. Bonsai's ordinary
scaled dot products must not be replaced with arithmetic modulo three: a
normal sum of three becomes zero in that field. A modular inference experiment
would additionally need activation quantization, signed-range reconstruction,
scale handling, and model-quality validation. FP16/FP32 rounded arithmetic
does not satisfy the exact field identities assumed by the theorem.

## Artifacts and reproduction

Fixture: `target/bench-results/bench_prefill_bilinear/layer0.safetensors`,
75,203,656 bytes. Payload SHA256:
`8772fdbb3aa59fb952812a9f0e03e5d8041f406e4ca800b70dd42971b96bf8c8`.

Preparation metadata:
`target/bench-results/bench_prefill_bilinear/preparation.json`.
Cached snapshot: `3f926b415992eaa2ae9dd7b573706494d6bbf787`.

```bash
python3 crates/higgs-models/examples/extract_prefill_weights.py \
  /Users/peppi/AI-Models/shared/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/3f926b415992eaa2ae9dd7b573706494d6bbf787/model.safetensors \
  target/bench-results/bench_prefill_bilinear/layer0.safetensors

cargo rustc --offline --release -p higgs-models --example prefill_bilinear_probe -- \
  --test -C link-arg=/Applications/Xcode-16.4.0.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/lib/darwin/libclang_rt.osx.a
target/release/examples/prefill_bilinear_probe --test-threads=1

cargo rustc --offline --release -p higgs-models --example prefill_bilinear_probe -- \
  -C link-arg=/Applications/Xcode-16.4.0.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/lib/darwin/libclang_rt.osx.a
target/release/examples/prefill_bilinear_probe \
  target/bench-results/bench_prefill_bilinear/layer0.safetensors \
  target/bench-results/bench_prefill_bilinear/results.jsonl
```

Extraction and benchmark outputs intentionally require new output filenames;
reuse the existing fixture or choose a fresh filename on repeat runs. MLX
runtime execution requires Apple framework access outside the current sandbox.
Only run timings without another active GPU workload.

At preparation time macOS 27.0/Xcode 16.4 was active, and power switched to
battery with low-power mode enabled. Record power again immediately before
measurement. Battery measurements must not be presented as reproducing the
September AC-power baseline.

## Integration gate

Report seven rotated-order samples, medians/ranges, active/peak memory and
numerical error against FP32 stock and FP16-cast stock. A candidate must reach
the existing 1.2x operator gate at both M values before integration is worth
considering. Real-model logits/behavior, loaded-model headroom and unprofiled
end-to-end TTFT remain separate required evidence.

GitNexus rated the shared QMM wrapper CRITICAL (32 symbols, seven processes).
It is unchanged. The new example was absent from the stale index, giving
UNKNOWN; a source search confirmed the control's callers are confined to this
new example. Existing user changes were preserved; no commit was made.
