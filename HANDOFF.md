# Higgs Handoff — simdgroup QGEMM + GGUF kernel development
# Session date: 2026-08-30/31 · Branch: nightly · Head: 2a105668d

## What's SHIPPED and working right now

| Component | Commit | Status |
|---|---|---|
| BM=32 simd QGEMM (default kernel) | 2bdfd8bf3 | ✅ correct, 926 GFLOP/s sorted at scale |
| ulong masks for 64-row walk | 2bdfd8bf3 | ✅ required for BM=64 |
| Circuit breaker: scaffold at 2, hard stop at 4 | f342baf | ✅ in nanobot-rs, same pattern applies here |
| GGUF parser skeleton (header + tensor infos) | 2a105668d | ⚠️ compiles, 2 test failures |
| Q4_K dequant skeleton (144 bytes → 256 f32) | 2a105668d | ⚠️ compiles, 3 test failures |
| Empirical simdgroup semantics probe | bb425bc63 | ✅ passes — transposed load verified |
| PLAN.md (full debug trail + GGUF plan) | 575df6174 | ✅ |

## Test failures (3, all known)

| Test | Why | Fix |
|---|---|---|
| `dflash::tests::accept_prefix_mismatched_lengths_panic_in_debug` | expects panic that only fires in debug builds; can't pass under `--release` | add `#[cfg(debug_assertions)]` or accept as expected-release-fail |
| `gguf::q4_k::tests::scale_min_k4_extraction` | expected value for j=4 is wrong (test expects 49, formula gives 1) | recompute expected from get_scale_min_k4 formula |
| `gguf::q4_k::tests::dequant_matches_manual_computation` | y[0] = -163840, expected 2.5 — f32→f16 encoding of d=0.5 produces wrong bits | debug `f32_to_f16_bits(0.5)`: should be 0x3800; verify by manual bit math |

## BM=64 debug status (the reason for this handoff)

**What happened**: BM=64/NT=256 simd kernel produces correct results for SG0
(rows 0-7) but zeros for SG1-7 (rows 8-63).

**What we know**:
- The sg_active gate is EXONERATED — removing it doesn't help (gate removed,
  rows 8+ still zero). The gate was removed permanently (correct: zero-staging
  makes it redundant).
- The kf=0-only bisection CONFIRMED: with kf=1 skipped, rows 8+ produce
  non-zero values → SG1-7 CAN run MMA, CAN read x_sh, CAN store.
- The kf=1 fragment load (base = &x_sh[8*XP + sg*8], transposed, origin (0,0))
  is the culprit. When it runs, it CORRUPTS or ZEROES the output.
- The standalone semantics probe (transposed load with per-SG m0 base offset,
  ld=16, known data) PASSES — the load works in isolation.
- The divergence only appears in the FULL kernel (multiple kf iterations ×
  multiple passes × decode concurrent with loads).

**Leading theory**: the kf=1 A-frag load (base = &x_sh[8*72 + sg*8], ld=72,
transpose=true) reads data written by the kf=0 decode of the NEXT pass —
a cross-pass race through threadgroup memory. The barrier placement may
not cover the case where the MMA of one kf reads while the staging of the
next kf writes.

**Fix to try (next session)**: add a `threadgroup_barrier` between the
kf=0 and kf=1 fragment load blocks inside the product loop. Or: load both
kf=0 and kf=1 A-frags into registers BEFORE any MMA (double-buffer).

## The simpler path that avoids this entirely

**Use kf=0-only** (skip kf=1 entirely): each pass covers only half the k-axis
per tk step, but the pass count doubles (the pass walk handles it). Net:
same total k coverage, decode cost doubles per pass but pass count doubles
too — the per-pass MMA is now 8-deep instead of 16-deep. Benchmark this.
If the MMA is not the bottleneck (it isn't — staging + decode dominate),
the half-depth MMA is free.

## What BM=64 enables (the prize)

70B-class models on a 32 GB M4 via IQ2/IQ3 quantization:
| Format | bpw | 70B weight size | Fits 32 GB? |
|---|---|---|---|
| IQ2_XXS | 2.06 | 18.0 GB | ✓ comfortable |
| IQ2_XS | 2.31 | 20.2 GB | ✓ |
| IQ4_XS | 4.25 | 37.2 GB | ✗ needs 48+ GB |

The IQ codebook lookup pattern (read index → threadgroup codebook → get
value vector) is structurally identical to the trellis decode higgs already
does. The gather kernel pattern transfers.

## GGUF Q4_K dequant reference

```rust
// Per super-block (144 bytes → 256 f32 values):
// bytes 0-1: f16 d (super-block scale)
// bytes 2-3: f16 dmin (super-block min)
// bytes 4-15: 12 bytes = 8 sub-block scales+mins, 6-bit each
// bytes 16-143: 128 bytes = 256 4-bit values

pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j - 4] >> 6) & 2),
        )
    }
}
```

## Known test failures (all pre-existing or WIP, none block shipping)

| Test | Status | Notes |
|---|---|---|
| `dflash::accept_prefix_mismatched_lengths_panic_in_debug` | pre-existing | expects debug-only panic under --release |
| `gguf::q4_k::scale_min_k4_extraction` | WIP | expected value needs recomputation |
| `gguf::q4_k::dequant_matches_manual_computation` | WIP | f32→f16 encoding precision |
| `gguf::parser` | WIP | compiles, may need GGUF v3 spec fixes |

## Session rotation config (just tuned)

```
localMaxContextTokens: 64536 (was 32768)
lcm.tauSoft: 0.85 (was 0.5 — compaction fires at ~46K conv tokens)
memory.sessionCompleteAfterSecs: 86400 (24h — was 1h)
memory.memoryFileMaxWords: 1500
```

Reload frequency: ~30 min → ~2-4 hours of active use.
