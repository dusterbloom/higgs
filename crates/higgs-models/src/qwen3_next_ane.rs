//! ANE-accelerated target verify path for Qwen3-Next.
//!
//! A compiled ANE kernel represents a single dense projection `y = x @ W^T`.
//! Weights are dequantized to f32 at compile time and baked into the ANE
//! BLOBFILE; each dispatch does
//! `ANE_in ← transpose(x); eval; x_out ← transpose(ANE_out)` at a compile-time
//! seq length (seq < compile seq is zero-padded; extra rows are sliced off).
//!
//! Phase 1 shipped `AneQkvzKernel` bound to `in_proj_qkvz`. Wave 1 generalises
//! to `AneProjKernel` so all three GDN projections (`in_proj_qkvz`,
//! `in_proj_ba`, `out_proj`) can run on ANE.
//!
//! Feature-gated behind `ane` — not compiled otherwise.
//!
//! Design references:
//! - `dflash_ane.rs` — drafter's ANE GDN path (same primitives, different shape)
//! - `ane_mil::gen_blobfile_matmul` — MIL generator for `y = x @ W`
//! - `.planning/dflash-forensics-and-ane-target-plan.md` — why we're doing this

#![allow(
    clippy::too_many_arguments,
    clippy::as_conversions,
    clippy::cast_possible_truncation
)]

use std::sync::Arc;

use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::{Array, Dtype};

use crate::ane_bridge::{self, AneKernel};
use crate::ane_mil::{self, FusedMil};
use crate::ane_mlmodel::AneLmHeadLut6Kernel;
use crate::dflash_ane::{ane_to_cpu, cpu_to_ane};

/// A compiled ANE kernel for a single dense projection `y = x @ W^T`.
///
/// `W` is `[out_dim, in_dim]` row-major f32 (already dequantized) at compile
/// time; inputs at dispatch have shape `[B, S, in_dim]` with `S <= seq_len`.
pub struct AneProjKernel {
    kernel: AneKernel,
    /// Tag for debug / MIL naming (e.g. `"qkvz"`, `"ba"`, `"out_proj"`).
    pub name: &'static str,
    /// Input channels.
    pub in_dim: usize,
    /// Output channels.
    pub out_dim: usize,
    /// Compile-time seq dim. Runtime seq must be `<= seq_len` (zero-padded).
    pub seq_len: usize,
    input_bytes: usize,
    output_bytes: usize,
}

// SAFETY: `AneKernel` is structurally `!Send + !Sync` (contains a raw IOSurface
// pointer that is thread-bound). The inline `Vec<Arc<GdnAneLayerKernels>>`
// field on `GatedDeltaNet` is populated only by `enable_ane_gdn*` methods,
// which are documented as main-thread-only and exercised exclusively by the
// Wave 1/2 parity tests. Production (`HIGGS_TARGET_ANE_GDN=1`) goes through
// `qwen3_next_ane_worker::GdnAneWorkerHandle`, whose mpsc-handle path never
// crosses an `AneKernel` between threads — the kernel is created and
// dropped on the same `qwen-gdn-ane-worker` thread.
//
// The unsafe impls below let `Qwen3NextCausalLM` stay `Send + Sync` so it can
// be moved into `batch_engine`'s worker thread (`batch_engine.rs:117`) even
// when the `ane` feature is on. Callers MUST NOT mutate / dispatch through
// `ane_kernels` from a thread other than where it was populated.
#[allow(unsafe_code)]
// SAFETY: see comment above.
unsafe impl Send for AneProjKernel {}
#[allow(unsafe_code)]
// SAFETY: see comment above.
unsafe impl Sync for AneProjKernel {}

impl std::fmt::Debug for AneProjKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AneProjKernel")
            .field("name", &self.name)
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .field("seq_len", &self.seq_len)
            .finish()
    }
}

/// Compile an ANE kernel for `y = x @ W^T` where `W` is `[out_dim, in_dim]`
/// row-major f32 (already dequantized).
///
/// `name` tags the MIL module and debug output — use a static string like
/// `"qkvz"`, `"ba"`, or `"out_proj"`.
pub fn compile_proj(
    w_f32: &[f32],
    in_dim: usize,
    out_dim: usize,
    seq_len: usize,
    name: &'static str,
) -> Result<AneProjKernel, String> {
    if w_f32.len() != out_dim * in_dim {
        return Err(format!(
            "{name} weight size mismatch: got {}, expected {out_dim}*{in_dim}={}",
            w_f32.len(),
            out_dim * in_dim
        ));
    }
    ane_bridge::ane_init()?;

    let mil: FusedMil = ane_mil::gen_blobfile_matmul(in_dim, out_dim, seq_len, name);

    // Tile `[out_dim, in_dim]` on the `oc` axis per the MIL generator's plan.
    // Tile boundaries, order, and transpose are identical to the drafter's
    // `build_tiled_weight_blobs` — kept inline here to avoid exposing a public
    // symbol that has no other caller.
    let plan = ane_mil::compute_blobfile_tile_plan(in_dim, out_dim);
    let mut tile_blobs: Vec<Vec<u8>> = Vec::with_capacity(plan.n_tiles);
    for t in 0..plan.n_tiles {
        let start = plan.tile_start(t);
        let this_oc = plan.actual_tile_size(t);
        let slice = &w_f32[start * in_dim..(start + this_oc) * in_dim];
        tile_blobs.push(ane_bridge::build_weight_blob_transposed(
            slice, this_oc, in_dim,
        ));
    }
    let blob_refs: Vec<&[u8]> = tile_blobs.iter().map(|v| v.as_slice()).collect();
    let name_refs: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

    let kernel = AneKernel::compile_multi_weights(
        &mil.mil_text,
        &name_refs,
        &blob_refs,
        &[mil.input_bytes],
        &[mil.output_bytes],
    )?;

    Ok(AneProjKernel {
        kernel,
        name,
        in_dim,
        out_dim,
        seq_len,
        input_bytes: mil.input_bytes,
        output_bytes: mil.output_bytes,
    })
}

/// Compile a new ANE projection kernel by patching weights into the donor's
/// already-compiled microcode (skips MIL compileWithQoS — only loadWithQoS).
///
/// The donor must have been compiled with the same `in_dim`, `out_dim`,
/// `seq_len`, and `name` as the new kernel will use — those parameters fully
/// determine the MIL text and tile layout, so the donor's microcode is
/// reusable. New weights `w_f32` are `[out_dim, in_dim]` row-major f32.
///
/// Used by Wave 2 to share microcode across all 24 GDN layers (one full
/// compile on layer 0; layers 1..23 patch in O(load) rather than O(compile)).
pub fn compile_proj_from_donor(
    donor: &AneProjKernel,
    w_f32: &[f32],
) -> Result<AneProjKernel, String> {
    let in_dim = donor.in_dim;
    let out_dim = donor.out_dim;
    let seq_len = donor.seq_len;
    let name = donor.name;
    if w_f32.len() != out_dim * in_dim {
        return Err(format!(
            "{name} donor patch weight size mismatch: got {}, expected {out_dim}*{in_dim}={}",
            w_f32.len(),
            out_dim * in_dim
        ));
    }

    // Regenerate the donor's MIL + tile plan deterministically from dims.
    let mil: FusedMil = ane_mil::gen_blobfile_matmul(in_dim, out_dim, seq_len, name);
    let plan = ane_mil::compute_blobfile_tile_plan(in_dim, out_dim);
    let mut tile_blobs: Vec<Vec<u8>> = Vec::with_capacity(plan.n_tiles);
    for t in 0..plan.n_tiles {
        let start = plan.tile_start(t);
        let this_oc = plan.actual_tile_size(t);
        let slice = &w_f32[start * in_dim..(start + this_oc) * in_dim];
        tile_blobs.push(ane_bridge::build_weight_blob_transposed(
            slice, this_oc, in_dim,
        ));
    }
    let blob_refs: Vec<&[u8]> = tile_blobs.iter().map(|v| v.as_slice()).collect();
    let name_refs: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

    let kernel = donor.kernel.patch_from_donor(
        &mil.mil_text,
        &name_refs,
        &blob_refs,
        &[mil.input_bytes],
        &[mil.output_bytes],
    )?;

    Ok(AneProjKernel {
        kernel,
        name,
        in_dim,
        out_dim,
        seq_len,
        input_bytes: mil.input_bytes,
        output_bytes: mil.output_bytes,
    })
}

impl AneProjKernel {
    /// Run `x @ W^T` on ANE.
    ///
    /// `x`: `[B, S, in_dim]`, any fp dtype. Requires `S <= seq_len`.
    /// Returns `[B, S, out_dim]` in the input's dtype.
    pub fn dispatch(&self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(Exception::custom(format!(
                "AneProjKernel({})::dispatch expects rank-3 input, got {:?}",
                self.name, shape
            )));
        }
        let b = shape[0] as usize;
        let s = shape[1] as usize;
        let h = shape[2] as usize;
        if h != self.in_dim {
            return Err(Exception::custom(format!(
                "{} in_dim mismatch: input {}, kernel {}",
                self.name, h, self.in_dim
            )));
        }
        if s > self.seq_len {
            return Err(Exception::custom(format!(
                "{} seq too long: input {}, kernel seq_len {}",
                self.name, s, self.seq_len
            )));
        }
        if b == 0 || s == 0 {
            return Err(Exception::custom(format!(
                "{} degenerate shape: B={}, S={}",
                self.name, b, s
            )));
        }

        // Materialise as contiguous f32 on CPU.
        let x_f32 = if x.dtype() == Dtype::Float32 {
            x.clone()
        } else {
            x.as_dtype(Dtype::Float32)?
        };
        x_f32.eval()?;
        let x_slice: &[f32] = x_f32.as_slice::<f32>();

        let oc = self.out_dim;
        let pad = self.seq_len;

        let mut out_all = vec![0.0f32; b * s * oc];
        let mut padded = vec![0.0f32; pad * h];
        let mut out_bytes = vec![0u8; self.output_bytes];

        for bi in 0..b {
            // Copy S rows into the first S rows of the [pad, h] buffer; rest are zero.
            let src = &x_slice[bi * s * h..(bi + 1) * s * h];
            padded[..s * h].copy_from_slice(src);
            for v in &mut padded[s * h..] {
                *v = 0.0;
            }

            let ane_in = cpu_to_ane(&padded, pad, h);
            debug_assert_eq!(ane_in.len(), self.input_bytes);
            self.kernel.write_input(0, &ane_in);
            // Prefer realtime eval (lower per-dispatch latency). Falls back to
            // standard eval if the caller hasn't entered realtime mode via
            // `AneKernel::begin_realtime()` — matches the dflash/diffusion
            // convention at `diffusion.rs:3027-3032`. When the GDN ANE worker
            // (or an inline inference-thread caller) has called
            // `begin_realtime`, this saves a Metal commit per dispatch.
            if self.kernel.eval_realtime().is_err() {
                self.kernel.eval().map_err(Exception::custom)?;
            }
            self.kernel.read_output(0, &mut out_bytes);

            // ANE output layout is [oc, pad]; transpose + slice back to [s, oc].
            let out_padded = ane_to_cpu(&out_bytes, pad, oc);
            out_all[bi * s * oc..(bi + 1) * s * oc].copy_from_slice(&out_padded[..s * oc]);
        }

        let out = Array::from_slice(&out_all, &[b as i32, s as i32, oc as i32]);
        if x.dtype() == Dtype::Float32 {
            Ok(out)
        } else {
            out.as_dtype(x.dtype())
        }
    }
}

/// Compiled ANE kernels for one GDN layer's three dense projections.
///
/// Populated by `enable_ane_gdn` on `GatedDeltaNet`. When present, the layer's
/// `forward_with_tape` dispatches `in_proj_qkvz`, `in_proj_ba`, and `out_proj`
/// on ANE instead of Metal matmul.
#[derive(Debug)]
pub struct GdnAneLayerKernels {
    pub qkvz: Arc<AneProjKernel>,
    pub ba: Arc<AneProjKernel>,
    pub out_proj: Arc<AneProjKernel>,
}

/// Dequantize a `QLinear`-style weight tensor to contiguous row-major f32.
///
/// Output layout is `[oc * ic]` f32 where the quantized weight was `[oc, ic]`
/// (i.e. PyTorch `nn.Linear` convention — `y = x @ W.T`).
///
/// For non-quantized (bf16/fp16) weights, just casts to f32.
pub(crate) fn dequantize_qlinear_to_f32(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Vec<f32>, Exception> {
    let w_f32 = if weight.dtype() == Dtype::Uint32 {
        let deq = ops::dequantize(weight, scales, biases, group_size, bits)?;
        deq.as_dtype(Dtype::Float32)?
    } else {
        weight.as_dtype(Dtype::Float32)?
    };
    w_f32.eval()?;
    Ok(w_f32.as_slice::<f32>().to_vec())
}

// ---------------------------------------------------------------------------
// LUT6 lm_head compile path (public CoreML MLModel + coremltools palettize)
// ---------------------------------------------------------------------------

/// Content hash for the on-disk `.mlmodelc` cache key.
///
/// Uses FNV-1a on the f32 weight bytes — collision-free enough for cache
/// invalidation (we're not signing anything). Returns a lowercase hex string.
fn lut6_weight_hash(w_f32: &[f32]) -> String {
    // FNV-1a 64-bit.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in w_f32 {
        for &b in &v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    format!("{h:016x}")
}

/// Root directory for compiled LUT6 `lm_head` bundles.
/// Mirrors `ane_bridge.m::ane_cache_dir` (`~/.nanobot/ane_cache`) but scoped
/// to a subdir so the dense microcode cache and the LUT6 bundle cache
/// don't collide.
fn lut6_cache_root() -> std::path::PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_else(|| std::ffi::OsString::from("/tmp"));
    let mut p = std::path::PathBuf::from(home);
    p.push(".nanobot");
    p.push("ane_cache");
    p.push("lm_head_lut6");
    p
}

/// Compile (or load from cache) a 6-bit palettized MLModel for `lm_head`.
///
/// Spawns `scripts/palettize_lm_head.py` on first use for the given content
/// hash; subsequent calls with identical weights hit the filesystem cache
/// under `~/.nanobot/ane_cache/lm_head_lut6/`.
///
/// `w_f32` is row-major `[out_dim * in_dim]` — same layout as
/// [`compile_proj`]. `seq_len` is the compile-time sequence bucket; runtime
/// seqs must be `<= seq_len`.
pub fn compile_proj_lut6(
    w_f32: &[f32],
    in_dim: usize,
    out_dim: usize,
    seq_len: usize,
    _name: &'static str,
) -> Result<AneLmHeadLut6Kernel, String> {
    use std::fs;
    if w_f32.len() != out_dim * in_dim {
        return Err(format!(
            "compile_proj_lut6: weight size {}, expected {out_dim}*{in_dim}={}",
            w_f32.len(),
            out_dim * in_dim
        ));
    }

    let hash = lut6_weight_hash(w_f32);
    let cache_slot = lut6_cache_root()
        .join(format!("{hash}_{out_dim}x{in_dim}_s{seq_len}"));
    let cache_mlmodelc = cache_slot.join("model.mlmodelc");

    if !cache_mlmodelc.is_dir() {
        fs::create_dir_all(&cache_slot)
            .map_err(|e| format!("compile_proj_lut6: create cache slot: {e}"))?;

        // Stage in a sibling temp dir, then atomically rename into place.
        let staging = cache_slot.with_extension("staging");
        let _ = fs::remove_dir_all(&staging);
        fs::create_dir_all(&staging)
            .map_err(|e| format!("compile_proj_lut6: create staging: {e}"))?;

        let weights_bin = staging.join("w.bin");
        {
            let mut bytes = Vec::with_capacity(w_f32.len() * 4);
            for v in w_f32 {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            fs::write(&weights_bin, &bytes)
                .map_err(|e| format!("compile_proj_lut6: write weights: {e}"))?;
        }

        let script = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/palettize_lm_head.py");
        let out = std::process::Command::new("python3")
            .arg(script)
            .arg("--weights-bin").arg(&weights_bin)
            .arg("--vocab").arg(out_dim.to_string())
            .arg("--hidden").arg(in_dim.to_string())
            .arg("--seq-len").arg(seq_len.to_string())
            .arg("--out-dir").arg(&staging)
            .output()
            .map_err(|e| format!("compile_proj_lut6: spawn python3: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "compile_proj_lut6: palettize_lm_head.py failed (exit={}): {}",
                out.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&out.stderr)
            ));
        }

        let staged = staging.join("model.mlmodelc");
        if !staged.is_dir() {
            return Err(format!(
                "compile_proj_lut6: script did not produce {}",
                staged.display()
            ));
        }
        // Clean up the weights file before sealing the cache slot.
        let _ = fs::remove_file(&weights_bin);
        fs::rename(&staged, &cache_mlmodelc)
            .map_err(|e| format!("compile_proj_lut6: rename to cache: {e}"))?;
        let _ = fs::remove_dir_all(&staging);
    }

    let path_str = cache_mlmodelc
        .to_str()
        .ok_or_else(|| "compile_proj_lut6: cache path not UTF-8".to_string())?;
    AneLmHeadLut6Kernel::load(path_str, out_dim, in_dim, seq_len)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::{ops::matmul, random};

    /// Synthetic parity: random f32 weights + random f32 input, ANE vs MLX matmul.
    ///
    /// Uses modest dimensions so the test runs quickly. Expanded 9B-sized weights
    /// are exercised by the `#[ignore]` end-to-end test in `qwen3_next.rs`.
    #[test]
    fn ane_proj_parity_synthetic() {
        // Small but non-trivial.
        let in_dim: usize = 512;
        let out_dim: usize = 1024;
        let s: usize = 17;
        let pad: usize = 32;

        let w = random::uniform::<f32, f32>(-0.05, 0.05, &[out_dim as i32, in_dim as i32], None)
            .expect("random w");
        w.eval().unwrap();
        let w_vec = w.as_slice::<f32>().to_vec();

        let x = random::uniform::<f32, f32>(-1.0, 1.0, &[1, s as i32, in_dim as i32], None)
            .expect("random x");
        x.eval().unwrap();

        // Reference: mlx matmul. W is [out_dim, in_dim]; we compute x @ W.T → [1, s, out_dim].
        let y_ref = matmul(&x, &w.t()).unwrap();
        y_ref.eval().unwrap();
        let y_ref_vec: Vec<f32> = y_ref.as_slice::<f32>().to_vec();

        let kernel = compile_proj(&w_vec, in_dim, out_dim, pad, "synthetic")
            .expect("compile_proj failed — ANE available?");
        let y_ane = kernel.dispatch(&x).expect("dispatch");
        y_ane.eval().unwrap();
        let y_ane_vec: Vec<f32> = y_ane.as_slice::<f32>().to_vec();

        assert_eq!(y_ane_vec.len(), y_ref_vec.len());
        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for (a, b) in y_ane_vec.iter().zip(y_ref_vec.iter()) {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
            let denom = b.abs().max(1e-4);
            let r = d / denom;
            if r > max_rel {
                max_rel = r;
            }
        }
        // Tolerance calibrated for fp16 ANE matmul of 512-wide GEMV.
        // Drafter GDN parity runs at max_diff ≈ 0.033 against CPU f32 at larger
        // dims; 512×1024 should comfortably stay under 0.05 absolute.
        assert!(
            max_diff < 0.05,
            "ANE proj parity: max_diff={max_diff} max_rel={max_rel} (budget 0.05)"
        );
    }

    /// Donor-patch parity: compile a donor with weights W1, patch with weights
    /// W2, and verify each kernel reproduces its own matmul. This validates the
    /// Wave 2 patch_from_donor path on synthetic data before the full 9B test.
    ///
    /// Also checks that `patch_from_donor` does NOT increment `compile_count()`
    /// — the whole point of donor patching is to skip MIL compilation.
    #[test]
    fn ane_proj_donor_patch_parity_synthetic() {
        let in_dim: usize = 512;
        let out_dim: usize = 1024;
        let s: usize = 17;
        let pad: usize = 32;

        let w1 = random::uniform::<f32, f32>(-0.05, 0.05, &[out_dim as i32, in_dim as i32], None)
            .expect("random w1");
        w1.eval().unwrap();
        let w1_vec = w1.as_slice::<f32>().to_vec();

        let w2 = random::uniform::<f32, f32>(-0.05, 0.05, &[out_dim as i32, in_dim as i32], None)
            .expect("random w2");
        w2.eval().unwrap();
        let w2_vec = w2.as_slice::<f32>().to_vec();

        let x = random::uniform::<f32, f32>(-1.0, 1.0, &[1, s as i32, in_dim as i32], None)
            .expect("random x");
        x.eval().unwrap();

        let donor = compile_proj(&w1_vec, in_dim, out_dim, pad, "donor")
            .expect("compile_proj donor failed");
        let compile_before = ane_bridge::compile_count();
        let patched =
            compile_proj_from_donor(&donor, &w2_vec).expect("compile_proj_from_donor failed");
        let compile_after = ane_bridge::compile_count();
        assert_eq!(
            compile_before, compile_after,
            "patch_from_donor must not trigger fresh MIL compile (before={compile_before}, \
             after={compile_after})"
        );

        // Donor reproduces W1 matmul.
        let y_donor_ane = donor.dispatch(&x).unwrap();
        y_donor_ane.eval().unwrap();
        let y_donor_ref = mlx_rs::ops::matmul(&x, &w1.t()).unwrap();
        y_donor_ref.eval().unwrap();
        let max_diff_donor: f32 = y_donor_ref
            .subtract(&y_donor_ane)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        assert!(
            max_diff_donor < 0.05,
            "donor kernel parity: max_diff={max_diff_donor}"
        );

        // Patched reproduces W2 matmul (NOT W1 — proves weights actually swapped).
        let y_patched_ane = patched.dispatch(&x).unwrap();
        y_patched_ane.eval().unwrap();
        let y_patched_ref = mlx_rs::ops::matmul(&x, &w2.t()).unwrap();
        y_patched_ref.eval().unwrap();
        let max_diff_patched: f32 = y_patched_ref
            .subtract(&y_patched_ane)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        assert!(
            max_diff_patched < 0.05,
            "patched kernel parity: max_diff={max_diff_patched}"
        );

        // And to be paranoid: patched output should differ materially from donor
        // matmul (otherwise the swap was a no-op).
        let cross = y_patched_ane
            .subtract(&y_donor_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            cross > 0.05,
            "patched output indistinguishable from donor weights — patch likely failed (max_diff={cross})"
        );
    }

    /// LUT6 probe: does ANE `compile_direct` accept `constexpr_lut_to_dense`?
    ///
    /// Minimal MIL program: 16×16 weight represented as uint8 indices into a
    /// 64-entry fp16 palette (scalar palettization, LUT shape [1,1,64,1]).
    /// If the op compiles and evals, the Rust-MIL LUT6 path is viable. If it
    /// fails with "op not supported" or parse error, we fall back to porting
    /// Shipstuff's Python .mlmodelc pipeline.
    ///
    /// The probe uses **inline const** uint8 indices and fp16 LUT values to
    /// avoid BLOBFILE format uncertainty — isolates the variable to "does the
    /// ANE compiler accept this MIL op?"
    ///
    /// Run:
    ///   cargo test -p higgs-models --features ane \
    ///     qwen3_next_ane::tests::probe_lut6_constexpr -- --nocapture
    #[test]
    #[ignore = "probe — run explicitly to decide LUT6 implementation path"]
    fn probe_lut6_constexpr() {
        use crate::ane_bridge::{self, AneKernel};
        use crate::ane_mil::MIL_HEADER;

        ane_bridge::ane_init().expect("ANE init");
        ane_bridge::set_quiet(false);

        // 16×16 weight, all-zero indices → W reconstructed to all lut[0]
        let oc = 16usize;
        let ic = 16usize;

        // Indices via BLOBFILE: 256 uint8 values, LSB-packed = 192 bytes of 0.
        // Blob layout follows the int8 builder: 64-byte header (magic DEADBEEF,
        // width marker at buf[10]) + packed data starting at offset 64.
        let idx_count = oc * ic; // 256
        let idx_bytes = idx_count; // uint8 = 1 byte per index
        let mut idx_blob = vec![0u8; 64 + idx_bytes];
        idx_blob[0] = 0xEF;
        idx_blob[1] = 0xBE;
        idx_blob[2] = 0xAD;
        idx_blob[3] = 0xDE;
        idx_blob[4] = 0x01;
        idx_blob[10] = 0x08; // 8-bit marker (uint8; mirrors int8 format)
        // packed data is already zero from vec![0u8; ...]

        // Inline fp16 LUT: 256 values for uint8 (NUM_PALETTES = 2^8), all 0.0.
        let lut_vals: String = (0..256).map(|_| "0x0000").collect::<Vec<_>>().join(",");

        let mil = format!(
            "{MIL_HEADER}    func main<ios18>(tensor<fp16, [{oc}, {ic}]> x) {{\n\
        tensor<uint8, [{oc},{ic}]> ind = const()[name=string(\"ind\"), val=tensor<uint8, [{oc},{ic}]>(BLOBFILE(path=string(\"@model_path/weights/ind.bin\"), offset=uint84(64)))];\n\
        tensor<fp16, [1,1,256,1]> lut = const()[name=string(\"lut\"), val=tensor<fp16, [1,1,256,1]>([{lut_vals}])];\n\
        tensor<fp16, [{oc},{ic}]> W = constexpr_lut_to_dense(indices=ind, lut=lut)[name=string(\"W\")];\n\
        tensor<fp16, [{oc},{ic}]> y = add(x=x, y=W)[name=string(\"y\")];\n\
    }} -> (y);\n}}\n"
        );

        eprintln!("--- LUT6 probe MIL (BLOBFILE indices) ---\n{mil}\n---");

        let names: Vec<&str> = vec!["@model_path/weights/ind.bin"];
        let blob_refs: Vec<&[u8]> = vec![&idx_blob];
        let bytes = oc * ic * 2; // fp16

        eprintln!("\n[probe] compile_direct (full op set)...");
        let direct_res =
            AneKernel::compile_direct(&mil, &names, &blob_refs, &[bytes], &[bytes]);
        let direct_ok = direct_res.is_ok();
        match &direct_res {
            Ok(_) => eprintln!("PASS: compile_direct accepts constexpr_lut_to_dense"),
            Err(e) => eprintln!("FAIL compile_direct: {e}"),
        }

        eprintln!("\n[probe] compile_multi_weights...");
        let multi_res = AneKernel::compile_multi_weights(
            &mil, &names, &blob_refs, &[bytes], &[bytes],
        );
        let multi_ok = multi_res.is_ok();
        match &multi_res {
            Ok(_) => eprintln!("PASS: compile_multi_weights accepts constexpr_lut_to_dense"),
            Err(e) => eprintln!("FAIL compile_multi_weights: {e}"),
        }

        if !direct_ok && !multi_ok {
            panic!(
                "constexpr_lut_to_dense rejected by BOTH compile paths (BLOBFILE uint8). \
                 Rust-MIL LUT6 path not viable — fall back to Python .mlmodelc pipeline."
            );
        }
        eprintln!("\n[probe] LUT6 VIABLE — at least one compile path accepted the op.");
    }
}
