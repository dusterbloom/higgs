# Self-Training on Mac: EGGROLL + Apple SSD

> Forward-pass-only training meets single-model self-distillation.
> A self-improving local AI that trains privately on Apple Silicon.

## Papers

### EGGROLL — Evolution Strategies at the Hyperscale

- **arXiv**: [2511.16652](https://arxiv.org/abs/2511.16652) (Sarkar, Fellows et al., Oxford/MILA/NVIDIA, Nov 2025 / rev Feb 2026)
- **Code**: [ESHyperscale/HyperscaleES](https://github.com/ESHyperscale/HyperscaleES) (JAX), [nano-egg](https://github.com/ESHyperscale/nano-egg), [d0rc/egg.c](https://github.com/d0rc/egg.c) (C + ARM NEON + GCD)
- **Project page**: [eshyperscale.github.io](https://eshyperscale.github.io/)

#### Core Algorithm

EGGROLL is a zeroth-order optimizer that replaces backpropagation with population-based forward passes. It needs **no gradients, no optimizer state, no activation storage**.

```
for each update step:
    for i in 1..N (population):
        sample A_i ~ N(0,I) ∈ R^{m×r}     # low-rank factor
        sample B_i ~ N(0,I) ∈ R^{n×r}     # low-rank factor
        E_i = A_i @ B_i^T                  # perturbation (rank r)
        W_perturbed = W + sigma * E_i
        fitness_i = forward_pass(W_perturbed, data)  # just inference!

    # ES gradient estimate (no backprop)
    g = (1 / N*sigma) * sum_i(fitness_i * score(E_i))
    W = W + lr * g
```

The key insight: each perturbation is rank-r (tiny memory), but the update summed across N population members is high-rank (expressive). Unlike LoRA which stays rank-r forever.

#### Memory Advantage

| Component | Backprop (Adam) | EGGROLL |
|-----------|----------------|---------|
| Optimizer state | 2× model size | **None** |
| Activations | O(layers × batch × seq × hidden) | **None** |
| Gradients | 1× model size | **None** |
| Perturbation | N/A | O(r(m+n)) per layer, on-demand via RNG |

For a 1024×4096 layer with r=4: EGGROLL stores 4×5120 = 20K floats vs 4M for full-rank. **200× reduction.**

At 14B scale, GRPO becomes infeasible due to Adam memory. EGGROLL runs fine.

#### Results

| Task | Model | EGGROLL | GRPO | Notes |
|------|-------|---------|------|-------|
| Countdown | RWKV-7 1.5B | **35% val** | 23% | Same wall-clock, 1 GPU |
| GSM8K | RWKV-7 7B | **Outperforms** | Baseline | 8 GPUs, pop=8192 |
| AIME24 | RWKV-7 14B | **30%** (from 13%) | OOM | 32 GPUs, 12h |
| AIME25 | RWKV-7 14B | **33%** (from 7%) | OOM | 32 GPUs, 12h |

Throughput: **91% of pure batched inference**. Training is nearly as fast as running the model.

#### egg.c — Apple Silicon Reference

The community [egg.c](https://github.com/d0rc/egg.c) implementation already targets Apple Silicon:
- ARM NEON intrinsics for vectorized int8 ops
- Grand Central Dispatch (GCD) for parallelism
- int8 weights + int32 CPU / int64 GPU accumulation
- ~300k tok/s with population 40K+ on single 4090
- Pure C, zero ML framework dependencies

---

### Apple SSD — Simple Self-Distillation

- **arXiv**: [2604.01193](https://arxiv.org/abs/2604.01193) (Zhang, Bai, Zheng et al., Apple, Apr 2026)
- **Code**: [apple/ml-ssd](https://github.com/apple/ml-ssd), also [mlx-ssd](https://pypi.org/project/mlx-ssd/) on PyPI
- **HuggingFace**: [papers/2604.01193](https://huggingface.co/papers/2604.01193)

#### Core Algorithm

SSD asks: can a model improve using only its own raw outputs? Yes.

```
Step 1 — Sample:   Generate N solutions from frozen model at temperature T > 1
                    (raw, unverified — no execution, no filtering, no correctness check)
Step 2 — Train:    Standard cross-entropy SFT on the model's own outputs
Step 3 — Decode:   Serve with separately tuned temperature
```

**No rewards. No verifier. No teacher. No RL. N=1 sample per prompt already works.**

#### Why It Works — The Precision-Exploration Conflict

Code generation has two types of positions:
- **Locks**: Only one correct token (e.g., closing a bracket). Need low T to suppress distractors.
- **Forks**: Multiple viable continuations (e.g., choosing an algorithm). Need high T for exploration.

A single global temperature cannot satisfy both. SSD resolves this by reshaping distributions context-dependently:
- At locks: support compression → peak becomes a spike, robust to T_eval
- At forks: within-support reshaping → top alternatives form plateaus, tail removed

After SSD, the viable decoding temperature band widens substantially.

#### Results

| Model | Base pass@1 | +SSD pass@1 | Gain |
|-------|------------|-------------|------|
| Qwen3-30B-A3B-Instruct | 42.4% | **55.3%** | +12.9pp |
| Qwen3-4B-Instruct | 35.7% | **43.2%** | +7.5pp |
| Llama-3.1-8B-Instruct | 31.9% | **35.4%** | +3.5pp |
| Qwen3-4B-Thinking | 37.2% | **40.5%** | +3.3pp |

Gains concentrate on **harder problems**: Qwen3-30B hard pass@1: 32.7% → 48.0% (+15.3pp).

**Stress test**: Even T=2.0 with no truncation (~62% gibberish, no code) still improves: +5.7pp pass@1, +10.5pp pass@5. The mechanism is distribution reshaping, not learning from correct solutions.

---

## The Composition: EGGROLL + SSD on Mac

### Why These Two Papers Compose

SSD's step 2 (fine-tune on own outputs) normally requires backpropagation. Replace it with EGGROLL and **the entire self-improvement loop becomes forward-pass only**:

1. **Sample** (GPU): Run model at T>1 to generate training data (standard inference)
2. **Score** (CPU): Evaluate fitness of samples (correctness, user feedback, self-eval)
3. **Evolve** (ANE): Run EGGROLL perturbations — N forward passes with perturbed weights
4. **Update**: ES gradient from fitness scores updates the weights
5. **Reload**: Merge new weights, model improves

The loop is entirely forward passes + scalar arithmetic. No backprop anywhere.

### Why ANE Is Perfect

ANE is dedicated neural network silicon on every Apple chip. It runs forward passes independently of the GPU.

| Metric | Value | Notes |
|--------|-------|-------|
| 0.6B causal forward | 19ms | Qwen3-0.6B on ANE (measured) |
| Pop=32 sequential | 608ms / update | 32 perturbations × 19ms |
| Pop=1000 sequential | 19s / update | Large population, still on ANE |
| GPU cost | **Zero** | ANE is separate silicon |

While ANE evolves the draft model, GPU is free to:
- Serve the main model (35B inference)
- Run speculative decode verification
- Generate SSD samples at T>1

**This turns Apple's "no CUDA" limitation into a moat.** CUDA users just run backprop. We can't — but we run 1000 forward passes on ANE for free while GPU serves. Nobody else has this.

### Architecture

```
┌──────────────────────────────────────────────────┐
│                   HIGGS ENGINE                    │
│                                                   │
│  ┌─────────────┐    ┌──────────────────────────┐ │
│  │  ANE ISLAND │    │     GPU ISLAND            │ │
│  │             │    │                            │ │
│  │ EGGROLL     │    │  Serving (35B inference)   │ │
│  │ Trainer:    │    │  Speculative verify        │ │
│  │ - 0.6B fwd  │    │  SSD sampling (T>1)        │ │
│  │ - N perturb │    │                            │ │
│  │ - Score     │    │                            │ │
│  │ - Update    │    │                            │ │
│  └──────┬──────┘    └───────────┬────────────────┘ │
│         │                       │                  │
│         └───── Weight Sync ─────┘                  │
│                                                    │
│  ┌────────────────────────────────────────────┐   │
│  │         Self-Improvement Loop               │   │
│  │                                             │   │
│  │  Phase A: Data Collection (background)      │   │
│  │    GPU generates samples at T>1 during      │   │
│  │    normal serving. Samples scored by         │   │
│  │    correctness / user feedback / self-eval.  │   │
│  │                                             │   │
│  │  Phase B: Evolution (background on ANE)     │   │
│  │    EGGROLL perturbs draft model weights,    │   │
│  │    runs forward passes, scores, updates.    │   │
│  │    Zero GPU cost.                           │   │
│  │                                             │   │
│  │  Phase C: Weight Merge + Reload             │   │
│  │    Updated weights merged to safetensors.   │   │
│  │    Model reloaded. Quality improves.        │   │
│  │                                             │   │
│  │  Repeat forever. Private. On-device.        │   │
│  └────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

---

## Existing Building Blocks

### In higgs (this repo)

| Component | File/Commit | Status |
|-----------|-------------|--------|
| ANE causal forward (0.6B, 19ms) | `diffusion.rs` | Working |
| `forward_last()` on all engines | `diffusion.rs` (a79b904c) | Working |
| ANE bridge (kernel compile, eval) | `ane_bridge.rs` | Working |
| DiffusionEngine fp32 weight access | `diffusion.rs` | Working |
| Speculative decode pipeline | `diffusion.rs` (de5d1552) | 12.7 tok/s |
| Adaptive K controller | `diffusion.rs` (2c5a44d2) | Working |
| MoE global sort + TurboQuant | multiple | Production |

### In nanobot-rs (sibling repo)

| Component | Commit | Status |
|-----------|--------|--------|
| REINFORCE router training E2E on 35B | `0def331` | Proven |
| Self-improvement loop on real 35B | `91faed4` | Proven |
| Outcome-reward training loop | `21be43d` | Working |
| ANE recurrence kernel | `b6ff893` | Production |
| Router merge to safetensors >2GB | `4df93ee` | mmap, working |
| Routing eval with file drain | `e136749` | Working |
| Hybrid ANE+Metal+CPU 35B decode | `c3ebd4f` | 1.4 tok/s |

### External references

| Component | Source | Notes |
|-----------|--------|-------|
| egg.c ARM NEON + GCD | [d0rc/egg.c](https://github.com/d0rc/egg.c) | C reference for Apple Silicon |
| HyperscaleES noiser | [ESHyperscale/HyperscaleES](https://github.com/ESHyperscale/HyperscaleES) | JAX reference implementation |
| nano-egg int8 training | [ESHyperscale/nano-egg](https://github.com/ESHyperscale/nano-egg) | Single-file minGRU demo |
| mlx-ssd | [PyPI mlx-ssd](https://pypi.org/project/mlx-ssd/) | MLX port of SSD (3rd party) |

---

## Implementation Plan

### Phase 1: EGGROLL Core for DiffusionEngine (MVP)

Implement the EGGROLL training loop for the 0.6B fp32 model we already have on ANE.

**Deliverables:**
- `EggrollTrainer` struct holding base weights + hyperparams (sigma, lr, rank r, pop size N)
- `LowRankPerturbation` — counter-based RNG generation of A[m,r] + B[n,r]
- `perturbed_forward()` — inject perturbation into DiffusionEngine weights, run forward
- `es_update()` — compute ES gradient from fitness scores, update base weights
- Test: train 0.6B on a small corpus, verify loss decreases over iterations

**Key design decisions:**
- Perturbation injection: modify weight buffers in-place before `sgemm_nt`, restore after. DiffusionEngine already stores weights as flat `Vec<f32>` — direct mutation.
- RNG: use `rand_chacha::ChaCha8Rng` with counter-based seeding for reproducibility
- Fitness: token-level cross-entropy (bits per byte), same as nano-egg

### Phase 2: SSD Sampling + Scoring Pipeline

**Deliverables:**
- Temperature-controlled sampling in higgs engine (already exists for serving)
- Sample collection: save (prompt, response) pairs to disk during normal serving
- Fitness scorer: compare perturbed model outputs against collected samples
- Integration: SSD samples become the fitness function for EGGROLL

### Phase 3: Concurrent ANE Training + GPU Serving

**Deliverables:**
- Background training thread: EGGROLL runs on ANE while GPU serves
- Weight checkpoint + atomic reload
- Telemetry: track improvement over time (loss, acceptance rate if used with spec decode)

### Phase 4: Full Self-Improvement Loop

**Deliverables:**
- Continuous loop: serve → collect → evolve → reload → serve (improved)
- User feedback integration: thumbs up/down as fitness signal
- Privacy guarantees: all data stays on device, no network calls

---

## Feasibility Analysis

### What's achievable (weeks)

- Port egg.c's NEON training loop to Rust (GCD → rayon, NEON → std::simd or raw intrinsics)
- Wire EGGROLL into DiffusionEngine (fp32 weights are already Vec<f32>, sgemm_nt is our own code)
- SSD sampling: higgs already does temperature-controlled generation
- Weight merge: nanobot-rs solved safetensors >2GB merging

### What's hard (months)

- EGGROLL for quantized 35B (perturbing 3-bit weights requires fp32 workspace or dequant/requant)
- Concurrent training + serving memory pressure on 32GB
- Convergence validation for SSD+EGGROLL combination (novel, unproven)
- Scaling population size within M4 base memory budget

### Honest risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Small population on 32GB | Medium | Sequential perturbations (pop=1000 at 19s/step is fine for background training) |
| SSD unproven beyond code | Medium | Start with code tasks, extend empirically |
| Slow convergence | Low | Background training — hours/days is acceptable for "your AI improves overnight" |
| Quantized model perturbation | High | Start with 0.6B fp32, defer 35B to Phase 3+ |
| Memory contention ANE+GPU | Low | Already measured: minimal bandwidth contention on M4 |

### The product story

"Your AI gets better the more you use it. Everything stays on your Mac. Nothing is sent anywhere. It learns from how you work, what you correct, what you approve. Overnight, it's a little smarter than yesterday."

This is not possible with any cloud-dependent system. It is uniquely possible on Apple Silicon with ANE + EGGROLL + SSD.
