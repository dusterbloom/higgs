# ANE notch / whole-model prefill gate

Benchmark-only tools. No Higgs serving imports, policy changes or private ANE
APIs. See the September 13 plan and RESULTS.md for the qualification decision.

1. Build `cargo build --release -p higgs --bin higgs`. Pin the source diff,
   binary SHA256, checkpoint/tokenizer and a dedicated localhost config.
2. Launch that binary in tmux with an explicit config and unused port. Disable
   durable prefix caching and speculation. Never stop an unrelated server.
3. Prepare fixed request JSON (temperature zero, short fixed output budget,
   `return_progress`, `stream_options.include_usage`) and persist tokenizer IDs.
   Run `python3 benchmarks/ane_prefill/measure.py --endpoint http://127.0.0.1:PORT
   --request REQUEST.json --out NEW_RESULT_DIR --pid SERVER_PID`.
   Warm compiler/device separately. Fresh processes establish cold prompt cache.
   RSS is sampled process residency, **not** physical footprint or MLX allocation.
   Missing terminal progress timing is null; TTFT includes first-token work.
4. After the server exits, export with a Python environment containing NumPy,
   safetensors and CoreMLTools 9: `python export_shapes.py --model MODEL --out
   NEW_DIR --tokens 32` (also 1024). Runs serialize dense FP16 weights from the
   real layer-0 source tensor. Runtime Higgs uses an additional affine Q8 repack,
   so this is exact between graph variants, not bit-exact with Higgs.
5. `clang -O3 -fobjc-arc -framework Foundation -framework CoreML
   benchmarks/ane_prefill/probe.m -o PROBE`. For each variant run `PROBE
   VARIANT/z.mlpackage VARIANT/input.fp16 VARIANT ane > VARIANT/result.json`.
   Repeat `cpu` to test fallback. Calls reuse FP16 input/provider/output backings;
   returned-pointer identity is checked. Prediction still has Core ML internal
   dispatch/copy costs. This does **not** prove zero-copy MLX integration or actual
   ANE hardware residency. Placement is the public preferred-device plan.
6. Reassemble split outputs by output name (`z0`, `z1`); trim padded output to
   4096 channels. Validate against original on identical input. No real input
   channel may be discarded. Input padding adds 32 zeros; output adds 64 zeros.
7. Run `python3 benchmarks/ane_prefill/verify_shapes.py EXPORT_DIR` to validate
   outputs and measure separate CPU pack/copy/merge costs. Run `gpu_projection.py --model MODEL --out NEW_RESULT.json` under Python MLX
   after other hardware work ends. It compares fused QKVZ, QKV and isolated z
   Q8(g64) kernels, with prepacked weights and alternating timing order. This is
   a diagnostic estimate of marginal projection cost; only an in-serving paired
   comparison can prove a whole-request gain. Python MLX version is recorded.

`python3 -m unittest discover -s benchmarks/ane_prefill -p 'test_*.py'` checks
shape equivalence, lattice accounting and stream failure/timing handling.

Original notch reference: https://gist.github.com/Anemll/39f657dc48b402747bdd96458edd415f
The nominal product (Cout/16)*Cin*2 is 1 MiB at 2048→4096. Padding/splitting
changes graph geometry; the compiler may transform it again. Public prediction
latency cannot classify a native KernelDMA notch. No fabricated compiler flags
or hardware descriptor edits are used.
