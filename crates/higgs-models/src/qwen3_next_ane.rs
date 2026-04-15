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
}
