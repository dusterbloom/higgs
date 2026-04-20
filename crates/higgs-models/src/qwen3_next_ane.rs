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
use crate::ane_mlmodel::{AneLmHeadLut6Kernel, AneMlPackageKernel};
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

        for bi in 0..b {
            let src = &x_slice[bi * s * h..(bi + 1) * s * h];

            // Zero-copy strided write: transpose [s, h] directly into the
            // input IOSurface [1, h, 1, pad] via `get_input_base` + `dsb sy`,
            // skipping both the `Vec<u8>` allocation in `cpu_to_ane` and the
            // `IOSurfaceLock/Unlock` pair in `write_input`. Trailing pad
            // positions (s..pad) are left stale — the ANE channel-wise
            // matmul only reads position `p` to produce output position `p`,
            // so stale tails never leak into the first `s` output positions.
            if !self.kernel.write_input_strided_fp32(0, src, s, h, pad) {
                let mut padded = vec![0.0f32; pad * h];
                padded[..s * h].copy_from_slice(src);
                let ane_in = cpu_to_ane(&padded, pad, h);
                debug_assert_eq!(ane_in.len(), self.input_bytes);
                self.kernel.write_input(0, &ane_in);
            }
            // Prefer realtime eval (lower per-dispatch latency). Falls back to
            // standard eval if the caller hasn't entered realtime mode via
            // `AneKernel::begin_realtime()` — matches the dflash/diffusion
            // convention at `diffusion.rs:3027-3032`. When the GDN ANE worker
            // (or an inline inference-thread caller) has called
            // `begin_realtime`, this saves a Metal commit per dispatch.
            if self.kernel.eval_realtime().is_err() {
                self.kernel.eval().map_err(Exception::custom)?;
            }
            let out_slice = &mut out_all[bi * s * oc..(bi + 1) * s * oc];
            if !self.kernel.read_output_strided_fp32(0, out_slice, s, oc, pad) {
                let mut out_bytes = vec![0u8; self.output_bytes];
                self.kernel.read_output(0, &mut out_bytes);
                let out_padded = ane_to_cpu(&out_bytes, pad, oc);
                out_slice.copy_from_slice(&out_padded[..s * oc]);
            }
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
/// Populated by `enable_ane_gdn*` on `GatedDeltaNet`. Two layouts share this
/// struct:
///
/// * **Separate** (Wave 1/2, multi-bucket tests): `qkvz` and `ba` are
///   `Some(..)`; `qkvz_ba_fused` is `None`. Two separate dispatches per layer.
/// * **Fused** (P0.8 Stage 3, inline path): `qkvz_ba_fused` is `Some(..)`;
///   `qkvz` and `ba` are `None`. One dispatch per layer covering both
///   projections. Mirrors the worker-thread layout in
///   [`crate::qwen3_next_ane_worker::compile_all_layers`].
///
/// `out_proj` is always a separate kernel (it consumes a different input).
#[derive(Debug)]
pub struct GdnAneLayerKernels {
    /// Compile-time seq dim for this bucket — always equal to the underlying
    /// kernel's `seq_len` regardless of which variant is populated.
    pub seq_len: usize,
    pub qkvz: Option<Arc<AneProjKernel>>,
    pub ba: Option<Arc<AneProjKernel>>,
    /// Fused qkvz+ba kernel — preferred when present (one ANE dispatch vs two).
    pub qkvz_ba_fused: Option<Arc<FusedGdnProjKernel>>,
    pub out_proj: Arc<AneProjKernel>,
}

// ---------------------------------------------------------------------------
// Fused qkvz+ba projection kernel (Gate 0 — Espresso-style fusion)
// ---------------------------------------------------------------------------

/// A fused ANE kernel for `(qkvz, ba) = (x @ W_qkvz^T, x @ W_ba^T)` in one
/// dispatch. Both projections share the same input (normalized hidden state)
/// and are concatenated on the channel axis: `[1, qkvz_oc+ba_oc, 1, seq]`.
///
/// At Carnice 9B dims (ic=4096, qkvz_oc=12288, ba_oc=2064) this produces a
/// single ANE program with 8 BLOBFILEs instead of the current 2-dispatch path.
pub struct FusedGdnProjKernel {
    kernel: AneKernel,
    pub in_dim: usize,
    pub qkvz_oc: usize,
    pub ba_oc: usize,
    pub seq_len: usize,
    input_bytes: usize,
    output_bytes: usize,
    /// Matches `ane_mil::gdn_output_rowwise()` at compile time — determines
    /// whether dispatch uses a per-row memcpy (rowwise) or a NEON strided
    /// transpose (channel-major) on readback.
    output_rowwise: bool,
}

#[allow(unsafe_code)]
unsafe impl Send for FusedGdnProjKernel {}
#[allow(unsafe_code)]
unsafe impl Sync for FusedGdnProjKernel {}

impl std::fmt::Debug for FusedGdnProjKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedGdnProjKernel")
            .field("in_dim", &self.in_dim)
            .field("qkvz_oc", &self.qkvz_oc)
            .field("ba_oc", &self.ba_oc)
            .field("seq_len", &self.seq_len)
            .finish()
    }
}

/// Build all tiled weight blobs for a single projection weight matrix.
///
/// `w_f32` is `[oc, ic]` row-major. Returns a `Vec<Vec<u8>>` of transposed
/// blobs, one per tile. The tile plan matches `emit_blobfile_matmul_tiled`.
fn build_tiled_blobs(w_f32: &[f32], ic: usize, oc: usize) -> Vec<Vec<u8>> {
    let plan = ane_mil::compute_blobfile_tile_plan(ic, oc);
    let mut blobs = Vec::with_capacity(plan.n_tiles);
    for t in 0..plan.n_tiles {
        let start = plan.tile_start(t);
        let this_oc = plan.actual_tile_size(t);
        let slice = &w_f32[start * ic..(start + this_oc) * ic];
        blobs.push(ane_bridge::build_weight_blob_transposed(slice, this_oc, ic));
    }
    blobs
}

/// Compile a fused ANE kernel for `(qkvz, ba)` from two weight matrices.
///
/// `w_qkvz` is `[qkvz_oc, ic]` row-major f32. `w_ba` is `[ba_oc, ic]`.
pub fn compile_fused_gdn_proj(
    w_qkvz: &[f32],
    w_ba: &[f32],
    ic: usize,
    qkvz_oc: usize,
    ba_oc: usize,
    seq_len: usize,
) -> Result<FusedGdnProjKernel, String> {
    if w_qkvz.len() != qkvz_oc * ic {
        return Err(format!(
            "qkvz weight size mismatch: got {}, expected {}",
            w_qkvz.len(), qkvz_oc * ic
        ));
    }
    if w_ba.len() != ba_oc * ic {
        return Err(format!(
            "ba weight size mismatch: got {}, expected {}",
            w_ba.len(), ba_oc * ic
        ));
    }
    ane_bridge::ane_init()?;

    let mil = ane_mil::gen_fused_gdn_qkvz_ba_proj(ic, qkvz_oc, ba_oc, seq_len);

    let qkvz_blobs = build_tiled_blobs(w_qkvz, ic, qkvz_oc);
    let ba_blobs = build_tiled_blobs(w_ba, ic, ba_oc);
    let all_blobs: Vec<&[u8]> = qkvz_blobs.iter().chain(ba_blobs.iter())
        .map(|v| v.as_slice())
        .collect();
    let name_refs: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

    let kernel = AneKernel::compile_multi_weights(
        &mil.mil_text,
        &name_refs,
        &all_blobs,
        &[mil.input_bytes],
        &[mil.output_bytes],
    )?;

    Ok(FusedGdnProjKernel {
        kernel,
        in_dim: ic,
        qkvz_oc,
        ba_oc,
        seq_len,
        input_bytes: mil.input_bytes,
        output_bytes: mil.output_bytes,
        output_rowwise: ane_mil::gdn_output_rowwise(),
    })
}

/// Compile a fused kernel by patching weights into a donor's microcode.
pub fn compile_fused_gdn_proj_from_donor(
    donor: &FusedGdnProjKernel,
    w_qkvz: &[f32],
    w_ba: &[f32],
) -> Result<FusedGdnProjKernel, String> {
    let ic = donor.in_dim;
    let qkvz_oc = donor.qkvz_oc;
    let ba_oc = donor.ba_oc;
    let seq_len = donor.seq_len;
    if w_qkvz.len() != qkvz_oc * ic {
        return Err(format!(
            "fused donor patch qkvz size mismatch: got {}, expected {}",
            w_qkvz.len(), qkvz_oc * ic
        ));
    }
    if w_ba.len() != ba_oc * ic {
        return Err(format!(
            "fused donor patch ba size mismatch: got {}, expected {}",
            w_ba.len(), ba_oc * ic
        ));
    }

    let mil = ane_mil::gen_fused_gdn_qkvz_ba_proj(ic, qkvz_oc, ba_oc, seq_len);
    let qkvz_blobs = build_tiled_blobs(w_qkvz, ic, qkvz_oc);
    let ba_blobs = build_tiled_blobs(w_ba, ic, ba_oc);
    let all_blobs: Vec<&[u8]> = qkvz_blobs.iter().chain(ba_blobs.iter())
        .map(|v| v.as_slice())
        .collect();
    let name_refs: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

    let kernel = donor.kernel.patch_from_donor(
        &mil.mil_text,
        &name_refs,
        &all_blobs,
        &[mil.input_bytes],
        &[mil.output_bytes],
    )?;

    Ok(FusedGdnProjKernel {
        kernel,
        in_dim: ic,
        qkvz_oc,
        ba_oc,
        seq_len,
        input_bytes: mil.input_bytes,
        output_bytes: mil.output_bytes,
        output_rowwise: ane_mil::gdn_output_rowwise(),
    })
}

impl FusedGdnProjKernel {
    /// Run `(x @ W_qkvz^T, x @ W_ba^T)` on ANE in a single dispatch.
    ///
    /// `x`: `[B, S, in_dim]`, any fp dtype. Requires `S <= seq_len`.
    /// Returns `(qkvz, ba)` each in `[B, S, *_oc]` matching input dtype.
    pub fn dispatch(&self, x: &Array) -> Result<(Array, Array), Exception> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(Exception::custom(format!(
                "FusedGdnProjKernel::dispatch expects rank-3, got {:?}", shape
            )));
        }
        let b = shape[0] as usize;
        let s = shape[1] as usize;
        let h = shape[2] as usize;
        if h != self.in_dim {
            return Err(Exception::custom(format!(
                "fused_gdn in_dim mismatch: input {h}, kernel {}", self.in_dim
            )));
        }
        if s > self.seq_len {
            return Err(Exception::custom(format!(
                "fused_gdn seq too long: input {s}, kernel seq_len {}", self.seq_len
            )));
        }

        let prof = std::env::var("HIGGS_ANE_GDN_PROFILE").map(|v| v == "1").unwrap_or(false);
        let t_phase0 = prof.then(std::time::Instant::now);

        let x_f32 = if x.dtype() == Dtype::Float32 {
            x.clone()
        } else {
            x.as_dtype(Dtype::Float32)?
        };
        let t_cast = prof.then(std::time::Instant::now);
        // Async-eval so the MLX queue starts draining without blocking
        // this thread; the real fence happens at `as_slice::<f32>()` below.
        let _ = mlx_rs::transforms::async_eval([&x_f32]);
        let t_fence = prof.then(std::time::Instant::now);
        let x_slice: &[f32] = x_f32.as_slice::<f32>();
        let t_phase1 = prof.then(std::time::Instant::now);

        let total_oc = self.qkvz_oc + self.ba_oc;
        let pad = self.seq_len;

        let mut out_qkvz = vec![0.0f32; b * s * self.qkvz_oc];
        let mut out_ba = vec![0.0f32; b * s * self.ba_oc];

        let mut tw = 0u64;
        let mut ta = 0u64;
        let mut tr = 0u64;
        let mut fb_w = 0u64;
        let mut fb_r = 0u64;
        for bi in 0..b {
            let src = &x_slice[bi * s * h..(bi + 1) * s * h];

            let tw0 = prof.then(std::time::Instant::now);
            let write_ok = self.kernel.write_input_strided_fp32(0, src, s, h, pad);
            if !write_ok {
                fb_w += 1;
                let mut padded = vec![0.0f32; pad * h];
                padded[..s * h].copy_from_slice(src);
                let ane_in = cpu_to_ane(&padded, pad, h);
                debug_assert_eq!(ane_in.len(), self.input_bytes);
                self.kernel.write_input(0, &ane_in);
            }
            if let Some(t) = tw0 { tw += t.elapsed().as_nanos() as u64; }
            let ta0 = prof.then(std::time::Instant::now);
            if self.kernel.eval_realtime().is_err() {
                self.kernel.eval().map_err(Exception::custom)?;
            }
            if let Some(t) = ta0 { ta += t.elapsed().as_nanos() as u64; }

            let tr0 = prof.then(std::time::Instant::now);
            let out_q = &mut out_qkvz
                [bi * s * self.qkvz_oc..(bi + 1) * s * self.qkvz_oc];
            let out_b = &mut out_ba[bi * s * self.ba_oc..(bi + 1) * s * self.ba_oc];
            if self.output_rowwise {
                // MIL output is [1, 1, pad, total_oc] fp32 — row-major.
                // Per-row memcpy splits qkvz / ba without a NEON transpose.
                let base = self.kernel.get_output_base(0) as *const f32;
                if base.is_null() {
                    fb_r += 1;
                    let mut out_bytes = vec![0u8; self.output_bytes];
                    self.kernel.read_output(0, &mut out_bytes);
                    #[allow(unsafe_code)]
                    let all_f32: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            out_bytes.as_ptr() as *const f32,
                            out_bytes.len() / 4,
                        )
                    };
                    for si in 0..s {
                        let row_start = si * total_oc;
                        out_q[si * self.qkvz_oc..(si + 1) * self.qkvz_oc]
                            .copy_from_slice(&all_f32[row_start..row_start + self.qkvz_oc]);
                        out_b[si * self.ba_oc..(si + 1) * self.ba_oc]
                            .copy_from_slice(
                                &all_f32[row_start + self.qkvz_oc..row_start + total_oc],
                            );
                    }
                } else {
                    #[allow(unsafe_code)]
                    unsafe {
                        #[cfg(target_arch = "aarch64")]
                        std::arch::asm!("dsb sy", options(nostack, preserves_flags));
                        for si in 0..s {
                            let row_ptr = base.add(si * total_oc);
                            std::ptr::copy_nonoverlapping(
                                row_ptr,
                                out_q.as_mut_ptr().add(si * self.qkvz_oc),
                                self.qkvz_oc,
                            );
                            std::ptr::copy_nonoverlapping(
                                row_ptr.add(self.qkvz_oc),
                                out_b.as_mut_ptr().add(si * self.ba_oc),
                                self.ba_oc,
                            );
                        }
                    }
                }
            } else {
                let ok_q = self.kernel.read_output_strided_fp32_range(
                    0, out_q, s, 0, self.qkvz_oc, pad,
                );
                let ok_b = self.kernel.read_output_strided_fp32_range(
                    0, out_b, s, self.qkvz_oc, self.ba_oc, pad,
                );
                if !(ok_q && ok_b) {
                    fb_r += 1;
                    let mut out_bytes = vec![0u8; self.output_bytes];
                    self.kernel.read_output(0, &mut out_bytes);
                    let out_padded = ane_to_cpu(&out_bytes, pad, total_oc);
                    for si in 0..s {
                        let row_start = si * total_oc;
                        let dst_q = si * self.qkvz_oc;
                        out_q[dst_q..dst_q + self.qkvz_oc]
                            .copy_from_slice(&out_padded[row_start..row_start + self.qkvz_oc]);
                        let dst_b = si * self.ba_oc;
                        out_b[dst_b..dst_b + self.ba_oc].copy_from_slice(
                            &out_padded[row_start + self.qkvz_oc..row_start + total_oc],
                        );
                    }
                }
            }
            if prof { let _ = tr0.map(|t| tr += t.elapsed().as_nanos() as u64); }
        }
        if prof {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            static TE: AtomicU64 = AtomicU64::new(0);
            static TC: AtomicU64 = AtomicU64::new(0);
            static TF: AtomicU64 = AtomicU64::new(0);
            static TS: AtomicU64 = AtomicU64::new(0);
            static TW: AtomicU64 = AtomicU64::new(0);
            static TA: AtomicU64 = AtomicU64::new(0);
            static TR: AtomicU64 = AtomicU64::new(0);
            static FW: AtomicU64 = AtomicU64::new(0);
            static FR: AtomicU64 = AtomicU64::new(0);
            let te = t_phase0
                .zip(t_phase1)
                .map(|(a, b)| (b - a).as_nanos() as u64)
                .unwrap_or(0);
            let tc = t_phase0
                .zip(t_cast)
                .map(|(a, b)| (b - a).as_nanos() as u64)
                .unwrap_or(0);
            let tf = t_cast
                .zip(t_fence)
                .map(|(a, b)| (b - a).as_nanos() as u64)
                .unwrap_or(0);
            let ts = t_fence
                .zip(t_phase1)
                .map(|(a, b)| (b - a).as_nanos() as u64)
                .unwrap_or(0);
            let n = N.fetch_add(1, Ordering::Relaxed) + 1;
            TE.fetch_add(te, Ordering::Relaxed);
            TC.fetch_add(tc, Ordering::Relaxed);
            TF.fetch_add(tf, Ordering::Relaxed);
            TS.fetch_add(ts, Ordering::Relaxed);
            TW.fetch_add(tw, Ordering::Relaxed);
            TA.fetch_add(ta, Ordering::Relaxed);
            TR.fetch_add(tr, Ordering::Relaxed);
            FW.fetch_add(fb_w, Ordering::Relaxed);
            FR.fetch_add(fb_r, Ordering::Relaxed);
            if n % 200 == 0 {
                eprintln!(
                    "[gdn_prof] n={n} s={s} eval={:.1}us (cast={:.1} fence={:.1} slice={:.1}) wr={:.1}us ane={:.1}us rd={:.1}us fb_w={} fb_r={}",
                    TE.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0,
                    TC.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0,
                    TF.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0,
                    TS.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0,
                    TW.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0,
                    TA.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0,
                    TR.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0,
                    FW.load(Ordering::Relaxed),
                    FR.load(Ordering::Relaxed),
                );
            }
        }

        let qkvz = Array::from_slice(&out_qkvz, &[b as i32, s as i32, self.qkvz_oc as i32]);
        let ba = Array::from_slice(&out_ba, &[b as i32, s as i32, self.ba_oc as i32]);
        if x.dtype() == Dtype::Float32 {
            Ok((qkvz, ba))
        } else {
            Ok((qkvz.as_dtype(x.dtype())?, ba.as_dtype(x.dtype())?))
        }
    }
}

// ---------------------------------------------------------------------------

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
// int8 MLP projection compile path (symmetric per-tensor, conv1x1 mlpackage)
// ---------------------------------------------------------------------------

/// Root directory for compiled int8 MLP projection bundles.
/// Mirrors `lut6_cache_root`: same `~/.nanobot/ane_cache` base, separate
/// subdir so bundle kinds don't collide.
fn int8_mlpkg_cache_root() -> std::path::PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_else(|| std::ffi::OsString::from("/tmp"));
    let mut p = std::path::PathBuf::from(home);
    p.push(".nanobot");
    p.push("ane_cache");
    p.push("int8_mlp");
    p
}

/// Compile (or load from cache) an int8 mlpackage for a single MLP projection.
///
/// Spawns `scripts/quantize_int8_proj.py` on first use for the given content
/// hash; subsequent calls with identical weights hit the filesystem cache
/// under `~/.nanobot/ane_cache/int8_mlp/`.
///
/// `w_f32` is row-major `[out_dim * in_dim]`. The resulting kernel expects
/// input shape `[1, in_dim, 1, seq_len]` fp16 and emits `[1, out_dim, 1,
/// seq_len]` fp16 (conv1x1 layout — caller marshals the transpose).
///
/// Python interpreter: `HIGGS_CORETOOLS_PYTHON` env if set (must have
/// coremltools + torch), else `python3`.
pub fn compile_proj_int8_mlpkg(
    w_f32: &[f32],
    in_dim: usize,
    out_dim: usize,
    seq_len: usize,
    tag: &'static str,
) -> Result<AneMlPackageKernel, String> {
    use std::fs;
    if w_f32.len() != out_dim * in_dim {
        return Err(format!(
            "compile_proj_int8_mlpkg({tag}): weight size {}, expected {out_dim}*{in_dim}={}",
            w_f32.len(),
            out_dim * in_dim
        ));
    }

    let hash = lut6_weight_hash(w_f32);
    let cache_slot = int8_mlpkg_cache_root()
        .join(format!("{hash}_{out_dim}x{in_dim}_s{seq_len}"));
    let cache_mlmodelc = cache_slot.join("model.mlmodelc");

    if !cache_mlmodelc.is_dir() {
        fs::create_dir_all(&cache_slot)
            .map_err(|e| format!("compile_proj_int8_mlpkg({tag}): create cache slot: {e}"))?;

        let staging = cache_slot.with_extension("staging");
        let _ = fs::remove_dir_all(&staging);
        fs::create_dir_all(&staging)
            .map_err(|e| format!("compile_proj_int8_mlpkg({tag}): create staging: {e}"))?;

        let weights_bin = staging.join("w.bin");
        {
            let mut bytes = Vec::with_capacity(w_f32.len() * 4);
            for v in w_f32 {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            fs::write(&weights_bin, &bytes)
                .map_err(|e| format!("compile_proj_int8_mlpkg({tag}): write weights: {e}"))?;
        }

        // The script refuses to overwrite an existing `model.mlmodelc` in
        // --out-dir, so give it a dedicated subdir.
        let script_out = staging.join("out");
        fs::create_dir_all(&script_out)
            .map_err(|e| format!("compile_proj_int8_mlpkg({tag}): create script out: {e}"))?;

        let script = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/quantize_int8_proj.py");
        let py = std::env::var("HIGGS_CORETOOLS_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let out = std::process::Command::new(&py)
            .arg(script)
            .arg("--weights-bin").arg(&weights_bin)
            .arg("--out-features").arg(out_dim.to_string())
            .arg("--in-features").arg(in_dim.to_string())
            .arg("--seq-len").arg(seq_len.to_string())
            .arg("--out-dir").arg(&script_out)
            .output()
            .map_err(|e| format!("compile_proj_int8_mlpkg({tag}): spawn {py}: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "compile_proj_int8_mlpkg({tag}): quantize_int8_proj.py failed (exit={}): {}",
                out.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&out.stderr)
            ));
        }

        let staged = script_out.join("model.mlmodelc");
        if !staged.is_dir() {
            return Err(format!(
                "compile_proj_int8_mlpkg({tag}): script did not produce {}",
                staged.display()
            ));
        }
        let _ = fs::remove_file(&weights_bin);
        fs::rename(&staged, &cache_mlmodelc)
            .map_err(|e| format!("compile_proj_int8_mlpkg({tag}): rename to cache: {e}"))?;
        let _ = fs::remove_dir_all(&staging);
    }

    let path_str = cache_mlmodelc
        .to_str()
        .ok_or_else(|| format!("compile_proj_int8_mlpkg({tag}): cache path not UTF-8"))?;
    let input_shape = vec![1_i64, in_dim as i64, 1, seq_len as i64];
    let output_shape = vec![1_i64, out_dim as i64, 1, seq_len as i64];
    AneMlPackageKernel::load(path_str, "x", "y", input_shape, output_shape)
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

    // -----------------------------------------------------------------------
    // Gate 1 probe: can ANE handle elementwise at spatial=1?
    // -----------------------------------------------------------------------

    /// Probe whether ANE compiles + evals elementwise ops at spatial=1.
    ///
    /// The GDN recurrence operates at S=1 (single decode step). All existing
    /// ANE programs use spatial >= 16 (`ANE_MIN_SPATIAL`). If spatial=1 fails
    /// with status 0x1d, Gate 1 must use spatial=16 with zero-padded state,
    /// or fall back to per-head dispatch.
    ///
    /// Tests three programs of increasing complexity:
    ///   1. `y = sigmoid(x)` at `[1, 32, 1, 1]` — minimal elementwise
    ///   2. `y = sigmoid(x)` at `[1, 32, 1, 16]` — ANE_MIN_SPATIAL baseline
    ///   3. `y = a * b + c` at `[1, 128, 1, 1]` — multi-input elementwise
    ///
    /// Run:
    ///   cargo test -p higgs-models --features ane \
    ///     qwen3_next_ane::tests::probe_ane_elementwise_spatial_1 -- --nocapture
    #[test]
    #[ignore = "probe — run explicitly to gate the GDN recurrence ANE path"]
    fn probe_ane_elementwise_spatial_1() {
        use crate::ane_bridge::{self, AneKernel};
        use crate::ane_mil::MIL_HEADER;
        use half::f16;

        ane_bridge::ane_init().expect("ANE init");
        ane_bridge::set_quiet(false);

        // --- Probe 1: sigmoid at spatial=1 ---
        let ch = 32usize;
        let sp1_bytes = ch * 1 * 2; // fp16
        let mil_sp1 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            tensor<fp16, [1,{ch},1,1]> x) {{\n\
            \x20       tensor<fp16, [1,{ch},1,1]> y = sigmoid(x=x)\
            [name=string(\"y\")];\n\
            \x20   }} -> (y);\n}}\n"
        );

        eprintln!("\n--- Probe 1: sigmoid [1,{ch},1,1] (spatial=1) ---");
        eprintln!("{mil_sp1}");
        let res1 = AneKernel::compile(&mil_sp1, None, &[sp1_bytes], &[sp1_bytes]);
        let sp1_ok = match &res1 {
            Ok(k) => {
                // Write known input, eval, read output.
                let input: Vec<f16> = (0..ch).map(|i| {
                    f16::from_f32((i as f32 - 16.0) * 0.1)
                }).collect();
                let in_bytes: Vec<u8> = input.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                k.write_input(0, &in_bytes);
                match k.eval() {
                    Ok(()) => {
                        let mut out_buf = vec![0u8; sp1_bytes];
                        k.read_output(0, &mut out_buf);
                        let output: Vec<f16> = out_buf.chunks_exact(2)
                            .map(|c| f16::from_le_bytes([c[0], c[1]]))
                            .collect();
                        // Verify sigmoid: σ(0) = 0.5
                        let mid = output[16].to_f32();
                        eprintln!("PASS: sigmoid(0.0) = {mid:.4} (expected ~0.5)");
                        (mid - 0.5).abs() < 0.01
                    }
                    Err(e) => {
                        eprintln!("FAIL eval: {e}");
                        false
                    }
                }
            }
            Err(e) => {
                eprintln!("FAIL compile: {e}");
                false
            }
        };

        // --- Probe 2: sigmoid at spatial=16 (baseline — should always pass) ---
        let sp16_bytes = ch * 16 * 2;
        let mil_sp16 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            tensor<fp16, [1,{ch},1,16]> x) {{\n\
            \x20       tensor<fp16, [1,{ch},1,16]> y = sigmoid(x=x)\
            [name=string(\"y\")];\n\
            \x20   }} -> (y);\n}}\n"
        );

        eprintln!("\n--- Probe 2: sigmoid [1,{ch},1,16] (spatial=16 baseline) ---");
        let res2 = AneKernel::compile(&mil_sp16, None, &[sp16_bytes], &[sp16_bytes]);
        let sp16_ok = res2.is_ok();
        match &res2 {
            Ok(_) => eprintln!("PASS: spatial=16 compiles (baseline)"),
            Err(e) => eprintln!("FAIL: spatial=16 compile: {e}"),
        };

        // --- Probe 3: fused mul+add at spatial=1, larger channel dim ---
        let ch3 = 128usize;
        let io3_bytes = ch3 * 1 * 2;
        let mil_fma = format!(
            "{MIL_HEADER}    func main<ios18>(\
            tensor<fp16, [1,{ch3},1,1]> a, \
            tensor<fp16, [1,{ch3},1,1]> b, \
            tensor<fp16, [1,{ch3},1,1]> c) {{\n\
            \x20       tensor<fp16, [1,{ch3},1,1]> ab = mul(x=a,y=b)\
            [name=string(\"ab\")];\n\
            \x20       tensor<fp16, [1,{ch3},1,1]> y = add(x=ab,y=c)\
            [name=string(\"y\")];\n\
            \x20   }} -> (y);\n}}\n"
        );

        eprintln!("\n--- Probe 3: a*b+c [1,{ch3},1,1] (spatial=1, multi-input) ---");
        let res3 = AneKernel::compile(&mil_fma, None, &[io3_bytes * 3], &[io3_bytes]);
        let fma_ok = match &res3 {
            Ok(_) => { eprintln!("PASS: fma compiles at spatial=1"); true }
            Err(e) => { eprintln!("FAIL: fma compile: {e}"); false }
        };

        // --- Probe 3b: abs + real_div at spatial=1 ---
        // If these work, we can approximate sigmoid as 0.5 + 0.5*x/(1+|x|)
        // using only arithmetic ops, keeping everything on ANE.
        let ch_ab = 32usize;
        let ab_bytes = ch_ab * 1 * 2;
        let mil_abs_div = format!(
            "{MIL_HEADER}    func main<ios18>(\
            tensor<fp16, [1,{ch_ab},1,1]> x) {{\n\
            \x20       tensor<fp16, [1,{ch_ab},1,1]> ax = abs(x=x)[name=string(\"ax\")];\n\
            \x20       tensor<fp16, [1,{ch_ab},1,1]> one = const()[name=string(\"one\"), \
            val=tensor<fp16, [1,{ch_ab},1,1]>(1)];\n\
            \x20       tensor<fp16, [1,{ch_ab},1,1]> d = add(x=ax,y=one)[name=string(\"d\")];\n\
            \x20       tensor<fp16, [1,{ch_ab},1,1]> y = real_div(x=x,y=d)[name=string(\"y\")];\n\
            \x20   }} -> (y);\n}}\n"
        );
        eprintln!("\n--- Probe 3b: abs+div [1,{ch_ab},1,1] (spatial=1) ---");
        let res_ad = AneKernel::compile(&mil_abs_div, None, &[ab_bytes], &[ab_bytes]);
        let abs_div_ok = match &res_ad {
            Ok(k) => match k.eval() {
                Ok(()) => { eprintln!("PASS: abs+div compiles + evals"); true }
                Err(e) => { eprintln!("PASS compile, FAIL eval: {e}"); false }
            },
            Err(e) => { eprintln!("FAIL compile: {e}"); false }
        };

        // --- Probe 3c: exp at spatial=1 ---
        let mil_exp = format!(
            "{MIL_HEADER}    func main<ios18>(\
            tensor<fp16, [1,{ch_ab},1,1]> x) {{\n\
            \x20       tensor<fp16, [1,{ch_ab},1,1]> y = exp(x=x)[name=string(\"y\")];\n\
            \x20   }} -> (y);\n}}\n"
        );
        eprintln!("\n--- Probe 3c: exp [1,{ch_ab},1,1] (spatial=1) ---");
        let res_exp = AneKernel::compile(&mil_exp, None, &[ab_bytes], &[ab_bytes]);
        let exp_ok = match &res_exp {
            Ok(k) => match k.eval() {
                Ok(()) => { eprintln!("PASS: exp compiles + evals"); true }
                Err(e) => { eprintln!("PASS compile, FAIL eval: {e}"); false }
            },
            Err(e) => { eprintln!("FAIL compile: {e}"); false }
        };

        // --- Probe 4: reduce_sum over channel axis (critical for recurrence readout) ---
        // axes must be tensor<int32, [1]>, not scalar int32 — matches diffusion_ane.rs
        let dk = 128usize;
        let hv = 32usize;
        let red_in_bytes = dk * hv * 2;
        let red_out_bytes = hv * 2;
        let mil_reduce = format!(
            "{MIL_HEADER}    func main<ios18>(\
            tensor<fp16, [1,{dk},1,{hv}]> x) {{\n\
            \x20       tensor<int32, [1]> ax = const()[name=string(\"ax\"), \
            val=tensor<int32, [1]>([1])];\n\
            \x20       bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
            \x20       tensor<fp16, [1,1,1,{hv}]> y = reduce_sum(x=x,axes=ax,keep_dims=kd)\
            [name=string(\"y\")];\n\
            \x20   }} -> (y);\n}}\n"
        );

        eprintln!("\n--- Probe 4: reduce_sum [1,{dk},1,{hv}] -> [1,1,1,{hv}] ---");
        eprintln!("{mil_reduce}");
        let res4 = AneKernel::compile(&mil_reduce, None, &[red_in_bytes], &[red_out_bytes]);
        let reduce_ok = match &res4 {
            Ok(k) => {
                match k.eval() {
                    Ok(()) => { eprintln!("PASS: reduce_sum compiles + evals"); true }
                    Err(e) => { eprintln!("PASS compile, FAIL eval: {e}"); false }
                }
            }
            Err(e) => { eprintln!("FAIL: reduce_sum compile: {e}"); false }
        };

        // --- Summary ---
        eprintln!("\n=== Gate 1 Probe Summary ===");
        eprintln!("  sigmoid spatial=1:  {}", if sp1_ok { "PASS" } else { "FAIL" });
        eprintln!("  sigmoid spatial=16: {}", if sp16_ok { "PASS" } else { "FAIL" });
        eprintln!("  fma spatial=1:      {}", if fma_ok { "PASS" } else { "FAIL" });
        eprintln!("  abs+div spatial=1:  {}", if abs_div_ok { "PASS" } else { "FAIL" });
        eprintln!("  exp spatial=1:      {}", if exp_ok { "PASS" } else { "FAIL" });
        eprintln!("  reduce_sum:         {}", if reduce_ok { "PASS" } else { "FAIL" });

        if abs_div_ok && fma_ok && reduce_ok {
            eprintln!("\nGate 1 ALL-ANE VIABLE: abs+div works → polynomial sigmoid feasible.");
            if !exp_ok {
                eprintln!("  exp fails at spatial=1 → gate computation needs split or softplus approx.");
            }
            eprintln!("  State update [32,128,128] has natural spatial=128 — no padding needed.");
        } else if sp16_ok && !sp1_ok {
            eprintln!("\nGate 1 SPLIT REQUIRED: activations fail at spatial=1.");
            eprintln!("  Gate/beta on Metal, state update on ANE (spatial=128, no waste).");
        } else {
            eprintln!("\nGate 1 BLOCKED: elementwise ops failed on ANE.");
        }

        // Baseline must pass — if this fails the ANE is broken.
        assert!(sp16_ok, "spatial=16 sigmoid must compile — ANE sanity check");
    }

    /// Gate 1 parity: GDN recurrence state update on ANE vs f32 reference.
    ///
    /// Small dims (Hv=4, Dk=16, Dv=16) for fast test. Verifies the MIL
    /// program compiles, evals, and produces correct y and new_state.
    ///
    /// ANE layout: `[1, Dk, Hv, Dv]` — Dk in channels, Dv in spatial (W≥16).
    /// Inputs that conceptually lack a dimension are broadcast-expanded:
    ///   g[Hv], beta[Hv] → `[1,1,Hv,Dv]` (same value across Dv within a head)
    ///   k[Dk,Hv], q[Dk,Hv] → `[1,Dk,Hv,Dv]` (same value across Dv)
    ///   v[Hv,Dv] → `[1,1,Hv,Dv]` (natural layout, no expansion needed)
    #[test]
    fn gdn_recurrence_ane_parity_synthetic() {
        use crate::ane_bridge::{self, AneKernel};
        use crate::ane_mil::gen_gdn_recurrence_step;

        ane_bridge::ane_init().expect("ANE init");

        let hv = 4usize;
        let dk = 16usize;
        let dv = 16usize;

        let kerns = gen_gdn_recurrence_step(hv, dk, dv);
        eprintln!("STATE_UPDATE MIL:\n{}", kerns.state_update_mil);
        eprintln!("READOUT MIL:\n{}", kerns.readout_mil);

        let fw = kerns.flat_w;
        eprintln!("dk={dk}, hv={hv}, dv={dv}, flat_w={fw}");

        // Compile both kernels
        let k_state = AneKernel::compile(
            &kerns.state_update_mil, None, &kerns.state_input_sizes, &kerns.state_output_sizes,
        ).expect("state_update MIL compile failed");
        let k_readout = AneKernel::compile(
            &kerns.readout_mil, None, &kerns.readout_input_sizes, &kerns.readout_output_sizes,
        ).expect("readout MIL compile failed");

        // Logical per-head scalars
        let g_log: Vec<f32> = (0..hv).map(|i| 0.8 + (i as f32) * 0.02).collect();
        let beta_log: Vec<f32> = (0..hv).map(|i| 0.4 + (i as f32) * 0.05).collect();
        let k_log: Vec<f32> = (0..dk*hv).map(|i| ((i as f32) * 0.07).cos() * 0.1).collect();
        let v_log: Vec<f32> = (0..hv*dv).map(|i| ((i as f32) * 0.03).sin() * 0.1).collect();
        let q_log: Vec<f32> = (0..dk*hv).map(|i| ((i as f32) * 0.05).cos() * 0.1).collect();

        // Build flat IOSurface buffers — ALL [1, dk, 1, fw] (uniform channel dim).
        // ANE 0x1d when 3+ IOSurfaces mix C=1 and C=Dk.
        // g/beta/v are logically per-head but replicated across dk channels.
        let mut st_flat = vec![0.0f32; dk * fw];
        for c in 0..dk { for h in 0..hv { for d in 0..dv {
            st_flat[c * fw + h * dv + d] = (((c * hv * dv + h * dv + d) as f32 * 0.01) - 0.5).sin() * 0.1;
        }}}
        let mut g_flat = vec![0.0f32; dk * fw];
        for c in 0..dk { for h in 0..hv { for d in 0..dv {
            g_flat[c * fw + h * dv + d] = g_log[h];
        }}}
        let mut beta_flat = vec![0.0f32; dk * fw];
        for c in 0..dk { for h in 0..hv { for d in 0..dv {
            beta_flat[c * fw + h * dv + d] = beta_log[h];
        }}}
        let mut k_flat = vec![0.0f32; dk * fw];
        for c in 0..dk { for h in 0..hv { for d in 0..dv {
            k_flat[c * fw + h * dv + d] = k_log[c * hv + h];
        }}}
        let mut v_flat = vec![0.0f32; dk * fw];
        for c in 0..dk { for h in 0..hv { for d in 0..dv {
            v_flat[c * fw + h * dv + d] = v_log[h * dv + d];
        }}}
        let mut q_flat = vec![0.0f32; dk * fw];
        for c in 0..dk { for h in 0..hv { for d in 0..dv {
            q_flat[c * fw + h * dv + d] = q_log[c * hv + h];
        }}}

        fn to_f32_bytes(data: &[f32]) -> Vec<u8> {
            data.iter().flat_map(|v| v.to_le_bytes()).collect()
        }

        // ── Run kernel A: state_update ──
        // Input order matches MIL declaration: st, g, k, v, beta
        let st_bytes = to_f32_bytes(&st_flat);
        let g_bytes = to_f32_bytes(&g_flat);
        let k_bytes = to_f32_bytes(&k_flat);
        let v_bytes = to_f32_bytes(&v_flat);
        let beta_bytes = to_f32_bytes(&beta_flat);
        let q_bytes = to_f32_bytes(&q_flat);

        k_state.write_input(0, &st_bytes);
        k_state.write_input(1, &g_bytes);
        k_state.write_input(2, &k_bytes);
        k_state.write_input(3, &v_bytes);
        k_state.write_input(4, &beta_bytes);
        k_state.eval().expect("state_update ANE eval failed");

        // Read new_state [1, dk, 1, fw] fp32
        let mut ns_buf = vec![0u8; dk * fw * 4];
        k_state.read_output(0, &mut ns_buf);
        let ns_f32: Vec<f32> = ns_buf.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // ── Run kernel B: readout ──
        k_readout.write_input(0, &ns_buf);
        k_readout.write_input(1, &q_bytes);
        k_readout.eval().expect("readout ANE eval failed");

        // Read y [1, 1, 1, fw] fp32
        let mut y_buf = vec![0u8; fw * 4];
        k_readout.read_output(0, &mut y_buf);
        let y_f32: Vec<f32> = y_buf.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Extract logical dims (ignore padding)
        let mut ns_ane = vec![0.0f32; dk * hv * dv];
        for c in 0..dk { for h in 0..hv { for d in 0..dv {
            ns_ane[c * hv * dv + h * dv + d] = ns_f32[c * fw + h * dv + d];
        }}}
        let mut y_ane = vec![0.0f32; hv * dv];
        for h in 0..hv { for d in 0..dv {
            y_ane[h * dv + d] = y_f32[h * dv + d];
        }}

        // --- f32 reference ---
        let li = |c: usize, h: usize, d: usize| c * hv * dv + h * dv + d;

        let mut decay = vec![0.0f32; dk * hv * dv];
        for c in 0..dk { for h in 0..hv { for d in 0..dv {
            decay[li(c,h,d)] = st_flat[c * fw + h * dv + d] * g_log[h];
        }}}
        let mut kvm = vec![0.0f32; hv * dv];
        for h in 0..hv { for d in 0..dv {
            let mut s = 0.0f32;
            for c in 0..dk { s += decay[li(c,h,d)] * k_log[c * hv + h]; }
            kvm[h * dv + d] = s;
        }}
        let mut delta = vec![0.0f32; hv * dv];
        for h in 0..hv { for d in 0..dv {
            delta[h*dv+d] = (v_log[h*dv+d] - kvm[h*dv+d]) * beta_log[h];
        }}
        let mut ns_ref = vec![0.0f32; dk * hv * dv];
        for c in 0..dk { for h in 0..hv { for d in 0..dv {
            ns_ref[li(c,h,d)] = decay[li(c,h,d)] + k_log[c*hv+h] * delta[h*dv+d];
        }}}
        let mut y_ref = vec![0.0f32; hv * dv];
        for h in 0..hv { for d in 0..dv {
            let mut s = 0.0f32;
            for c in 0..dk { s += ns_ref[li(c,h,d)] * q_log[c*hv+h]; }
            y_ref[h*dv+d] = s;
        }}

        let max_diff_ns = ns_ane.iter().zip(ns_ref.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_diff_y = y_ane.iter().zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

        eprintln!("[recurrence parity] new_state max_diff={max_diff_ns:.6} y max_diff={max_diff_y:.6}");

        assert!(
            max_diff_ns < 0.01,
            "new_state parity: max_diff={max_diff_ns} (budget 0.01)"
        );
        assert!(
            max_diff_y < 0.05,
            "y parity: max_diff={max_diff_y} (budget 0.05 — fp16 reduce over {dk} channels)"
        );

        // ── Latency bench: 1000 iterations of the 2-kernel pipeline ──
        let warmup = 50;
        let iters = 1000;
        for _ in 0..warmup {
            k_state.write_input(0, &st_bytes);
            k_state.write_input(1, &g_bytes);
            k_state.write_input(2, &k_bytes);
            k_state.write_input(3, &v_bytes);
            k_state.write_input(4, &beta_bytes);
            k_state.eval().unwrap();
            k_state.read_output(0, &mut ns_buf);
            k_readout.write_input(0, &ns_buf);
            k_readout.write_input(1, &q_bytes);
            k_readout.eval().unwrap();
            k_readout.read_output(0, &mut y_buf);
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            k_state.write_input(0, &st_bytes);
            k_state.write_input(1, &g_bytes);
            k_state.write_input(2, &k_bytes);
            k_state.write_input(3, &v_bytes);
            k_state.write_input(4, &beta_bytes);
            k_state.eval().unwrap();
            k_state.read_output(0, &mut ns_buf);
            k_readout.write_input(0, &ns_buf);
            k_readout.write_input(1, &q_bytes);
            k_readout.eval().unwrap();
            k_readout.read_output(0, &mut y_buf);
        }
        let elapsed = t0.elapsed();
        let us_per = elapsed.as_micros() as f64 / iters as f64;
        eprintln!("[recurrence bench] {iters} iters in {:.1}ms, {us_per:.1}us/step (hv={hv} dk={dk} dv={dv})", elapsed.as_secs_f64() * 1000.0);
    }

    /// Bisect GDN recurrence InvalidMILProgram.
    ///
    /// Probes (simplest → full), stop at first failure:
    ///   A. 2-input mul at [1,16,16,4] — tests 2D spatial dims
    ///   B. add reduce_sum(axis=1) — tests channel reduce at 2D spatial
    ///   C. const(-1) fill syntax — tests negative const literal
    ///   D. 6 inputs with trivial mul — tests IOSurface count limit
    ///
    /// Run:
    ///   cargo test -p higgs-models --features ane \
    ///     qwen3_next_ane::tests::bisect_gdn_recurrence -- --nocapture
    #[test]
    fn bisect_gdn_recurrence() {
        use crate::ane_bridge::{self, AneKernel};
        use crate::ane_mil::MIL_HEADER;

        ane_bridge::ane_init().expect("ANE init");
        ane_bridge::set_quiet(false);

        // --- Probe A: 2-input mul at [1,16,16,4] ---
        let mil_a = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=a,y=b)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let sz_a = 16 * 16 * 4 * 2; // fp16
        let res_a = AneKernel::compile(&mil_a, None, &[sz_a, sz_a], &[sz_a]);
        let pass_a = res_a.is_ok();
        eprintln!("[bisect A] 2-input mul [1,16,16,4]: {}", if pass_a { "PASS" } else { "FAIL" });

        // --- Probe B: mul + reduce_sum(axis=1) at [1,16,16,4] ---
        let mil_b = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> p = mul(x=a,y=b)[name=string(\"p\")];\
            \n        tensor<fp16, [1,1,16,4]> y = reduce_sum(x=p,axes=ax,keep_dims=kd)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let sz_b_out = 1 * 16 * 4 * 2;
        let res_b = AneKernel::compile(&mil_b, None, &[sz_a, sz_a], &[sz_b_out]);
        let pass_b = res_b.is_ok();
        eprintln!("[bisect B] mul+reduce_sum [1,16,16,4]: {}", if pass_b { "PASS" } else { "FAIL" });

        // --- Probe C: const(-1) fill ---
        let mil_c = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,1,16,4]> a\
            \n    ) {{\
            \n        tensor<fp16, [1,1,16,4]> neg = const()[name=string(\"neg\"), val=tensor<fp16, [1,1,16,4]>(-1)];\
            \n        tensor<fp16, [1,1,16,4]> y = mul(x=a,y=neg)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let sz_c = 1 * 16 * 4 * 2;
        let res_c = AneKernel::compile(&mil_c, None, &[sz_c], &[sz_c]);
        let pass_c = res_c.is_ok();
        eprintln!("[bisect C] const(-1) fill: {}", if pass_c { "PASS" } else { "FAIL" });

        // --- Probe C2: scalar const(-1) + broadcast mul (fallback if C fails) ---
        if !pass_c {
            let mil_c2 = format!(
                "{MIL_HEADER}    func main<ios18>(\
                \n        tensor<fp16, [1,1,16,4]> a\
                \n    ) {{\
                \n        tensor<fp16, [1,1,1,1]> neg = const()[name=string(\"neg\"), val=tensor<fp16, [1,1,1,1]>(-1)];\
                \n        tensor<fp16, [1,1,16,4]> y = mul(x=a,y=neg)[name=string(\"y\")];\
                \n    }} -> (y);\n}}\n"
            );
            let res_c2 = AneKernel::compile(&mil_c2, None, &[sz_c], &[sz_c]);
            eprintln!("[bisect C2] scalar const(-1) + broadcast: {}", if res_c2.is_ok() { "PASS" } else { "FAIL" });
        }

        // --- Probe D: 6 inputs, trivial ops ---
        let mil_d = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> i0,\
            \n        tensor<fp16, [1,1,1,4]> i1,\
            \n        tensor<fp16, [1,1,1,4]> i2,\
            \n        tensor<fp16, [1,16,1,4]> i3,\
            \n        tensor<fp16, [1,1,16,4]> i4,\
            \n        tensor<fp16, [1,16,1,4]> i5\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> m1 = mul(x=i0,y=i1)[name=string(\"m1\")];\
            \n        tensor<fp16, [1,16,16,4]> m2 = mul(x=m1,y=i3)[name=string(\"m2\")];\
            \n        tensor<fp16, [1,16,16,4]> a1 = add(x=m2,y=i0)[name=string(\"a1\")];\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=a1,y=i5)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let sz_hv = 4 * 2;
        let sz_dk_hv = 16 * 4 * 2;
        let sz_dv_hv = 16 * 4 * 2;
        let res_d = AneKernel::compile(
            &mil_d, None,
            &[sz_a, sz_hv, sz_hv, sz_dk_hv, sz_dv_hv, sz_dk_hv],
            &[sz_a],
        );
        let pass_d = res_d.is_ok();
        eprintln!("[bisect D] 6 inputs, trivial ops: {}", if pass_d { "PASS" } else { "FAIL" });

        // Summary
        eprintln!("\n=== BISECT SUMMARY ===");
        eprintln!("  A (2D spatial mul):     {}", if pass_a { "OK" } else { "BLOCKED" });
        eprintln!("  B (reduce_sum):         {}", if pass_b { "OK" } else { "BLOCKED" });
        eprintln!("  C (const(-1)):          {}", if pass_c { "OK" } else { "BLOCKED" });
        eprintln!("  D (6 inputs):           {}", if pass_d { "OK" } else { "BLOCKED" });

        // --- Probe E: sub op at [1,1,16,4] ---
        let mil_e = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,1,16,4]> a,\
            \n        tensor<fp16, [1,1,16,4]> b\
            \n    ) {{\
            \n        tensor<fp16, [1,1,16,4]> y = sub(x=a,y=b)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_e = AneKernel::compile(&mil_e, None, &[sz_c, sz_c], &[sz_c]);
        let pass_e = res_e.is_ok();
        eprintln!("[bisect E] sub op [1,1,16,4]: {}", if pass_e { "PASS" } else { "FAIL" });

        // --- Probe F: const(0xBC00) = fp16 -1.0 via hex literal ---
        let mil_f = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,1,16,4]> a\
            \n    ) {{\
            \n        tensor<fp16, [1,1,16,4]> neg = const()[name=string(\"neg\"), val=tensor<fp16, [1,1,16,4]>(0xBC00)];\
            \n        tensor<fp16, [1,1,16,4]> y = mul(x=a,y=neg)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_f = AneKernel::compile(&mil_f, None, &[sz_c], &[sz_c]);
        let pass_f = res_f.is_ok();
        eprintln!("[bisect F] const(0xBC00) hex fill: {}", if pass_f { "PASS" } else { "FAIL" });

        // --- Probe G: const(0) then sub(zero, x) for negation ---
        let mil_g = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,1,16,4]> a,\
            \n        tensor<fp16, [1,1,16,4]> b\
            \n    ) {{\
            \n        tensor<fp16, [1,1,1,1]> z = const()[name=string(\"z\"), val=tensor<fp16, [1,1,1,1]>(0)];\
            \n        tensor<fp16, [1,1,16,4]> nb = sub(x=z,y=b)[name=string(\"nb\")];\
            \n        tensor<fp16, [1,1,16,4]> y = add(x=a,y=nb)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_g = AneKernel::compile(&mil_g, None, &[sz_c, sz_c], &[sz_c]);
        let pass_g = res_g.is_ok();
        eprintln!("[bisect G] sub(0,x) negation: {}", if pass_g { "PASS" } else { "FAIL" });

        // --- Probe H: direct sub(a, b) at full dims [1,16,16,4] ---
        let mil_h = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,1,16,4]> b\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> y = sub(x=a,y=b)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_h = AneKernel::compile(&mil_h, None, &[sz_a, sz_c], &[sz_a]);
        let pass_h = res_h.is_ok();
        eprintln!("[bisect H] sub(a,b) broadcast [1,16,16,4]-[1,1,16,4]: {}", if pass_h { "PASS" } else { "FAIL" });

        eprintln!("\n  E (sub op):             {}", if pass_e { "OK" } else { "BLOCKED" });
        eprintln!("  F (hex const):          {}", if pass_f { "OK" } else { "BLOCKED" });
        eprintln!("  G (sub(0,x)):           {}", if pass_g { "OK" } else { "BLOCKED" });
        eprintln!("  H (sub broadcast):      {}", if pass_h { "OK" } else { "BLOCKED" });

        // --- Probe I: concat on channel axis ---
        let mil_i = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,1,16,4]> b\
            \n    ) {{\
            \n        int32 ax = const()[name=string(\"ax\"), val=int32(1)];\
            \n        bool bF = const()[name=string(\"bF\"), val=bool(false)];\
            \n        tensor<fp16, [1,17,16,4]> y = concat(values=(a,b),axis=ax,interleave=bF)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let sz_i_out = 17 * 16 * 4 * 2;
        let res_i = AneKernel::compile(&mil_i, None, &[sz_a, sz_c], &[sz_i_out]);
        let pass_i = res_i.is_ok();
        eprintln!("[bisect I] concat ch axis [1,16+1,16,4]: {}", if pass_i { "PASS" } else { "FAIL" });

        // --- Probe J: reduce_sum with int32 const axes (matching actual program) ---
        let mil_j = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        int32 cat_ax = const()[name=string(\"catax\"), val=int32(1)];\
            \n        bool bF = const()[name=string(\"bF\"), val=bool(false)];\
            \n        tensor<fp16, [1,16,16,4]> p = mul(x=a,y=b)[name=string(\"p\")];\
            \n        tensor<fp16, [1,1,16,4]> rs = reduce_sum(x=p,axes=ax,keep_dims=kd)[name=string(\"rs\")];\
            \n        tensor<fp16, [1,1,16,4]> d = sub(x=a,y=rs)[name=string(\"d\")];\
            \n        tensor<fp16, [1,16,16,4]> ko = mul(x=b,y=d)[name=string(\"ko\")];\
            \n        tensor<fp16, [1,16,16,4]> ns = add(x=p,y=ko)[name=string(\"ns\")];\
            \n        tensor<fp16, [1,16,16,4]> sq = mul(x=ns,y=b)[name=string(\"sq\")];\
            \n        tensor<fp16, [1,1,16,4]> y = reduce_sum(x=sq,axes=ax,keep_dims=kd)[name=string(\"y\")];\
            \n        tensor<fp16, [1,17,16,4]> out = concat(values=(ns,y),axis=cat_ax,interleave=bF)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let res_j = AneKernel::compile(&mil_j, None, &[sz_a, sz_a], &[sz_i_out]);
        let pass_j = res_j.is_ok();
        eprintln!("[bisect J] full op chain (2 inputs): {}", if pass_j { "PASS" } else { "FAIL" });

        eprintln!("  I (concat):             {}", if pass_i { "OK" } else { "BLOCKED" });
        eprintln!("  J (full chain 2-in):    {}", if pass_j { "OK" } else { "BLOCKED" });

        // --- Probe J2: strip concat from J (just output ns) ---
        let mil_j2 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> p = mul(x=a,y=b)[name=string(\"p\")];\
            \n        tensor<fp16, [1,1,16,4]> rs = reduce_sum(x=p,axes=ax,keep_dims=kd)[name=string(\"rs\")];\
            \n        tensor<fp16, [1,1,16,4]> d = sub(x=a,y=rs)[name=string(\"d\")];\
            \n        tensor<fp16, [1,16,16,4]> ko = mul(x=b,y=d)[name=string(\"ko\")];\
            \n        tensor<fp16, [1,16,16,4]> ns = add(x=p,y=ko)[name=string(\"ns\")];\
            \n    }} -> (ns);\n}}\n"
        );
        let res_j2 = AneKernel::compile(&mil_j2, None, &[sz_a, sz_a], &[sz_a]);
        let pass_j2 = res_j2.is_ok();
        eprintln!("[bisect J2] chain no concat (7 ops): {}", if pass_j2 { "PASS" } else { "FAIL" });

        // --- Probe J3: just the sub(a,rs) where rs came from reduce_sum ---
        // (sub where one operand has broadcast dims from reduce)
        let mil_j3 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> p = mul(x=a,y=b)[name=string(\"p\")];\
            \n        tensor<fp16, [1,1,16,4]> rs = reduce_sum(x=p,axes=ax,keep_dims=kd)[name=string(\"rs\")];\
            \n        tensor<fp16, [1,16,16,4]> d = sub(x=a,y=rs)[name=string(\"d\")];\
            \n    }} -> (d);\n}}\n"
        );
        let res_j3 = AneKernel::compile(&mil_j3, None, &[sz_a, sz_a], &[sz_a]);
        let pass_j3 = res_j3.is_ok();
        eprintln!("[bisect J3] mul+reduce+sub(broadcast): {}", if pass_j3 { "PASS" } else { "FAIL" });

        // --- Probe J4: sub where input is [1,16,16,4] and b is [1,1,16,4] ---
        // This is sub(x=a, y=b) where a has C=16 but b has C=1. Broadcast on ch dim.
        let mil_j4 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,1,16,4]> b\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> d = sub(x=a,y=b)[name=string(\"d\")];\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=d,y=a)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_j4 = AneKernel::compile(&mil_j4, None, &[sz_a, sz_c], &[sz_a]);
        let pass_j4 = res_j4.is_ok();
        eprintln!("[bisect J4] sub(C=16, C=1) broadcast: {}", if pass_j4 { "PASS" } else { "FAIL" });

        // --- Probe J5: J3 (fixed) + mul(b, d) — d is [1,16,16,4] from broadcast sub ---
        let mil_j5 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> p = mul(x=a,y=b)[name=string(\"p\")];\
            \n        tensor<fp16, [1,1,16,4]> rs = reduce_sum(x=p,axes=ax,keep_dims=kd)[name=string(\"rs\")];\
            \n        tensor<fp16, [1,16,16,4]> d = sub(x=a,y=rs)[name=string(\"d\")];\
            \n        tensor<fp16, [1,16,16,4]> ko = mul(x=b,y=d)[name=string(\"ko\")];\
            \n    }} -> (ko);\n}}\n"
        );
        let res_j5 = AneKernel::compile(&mil_j5, None, &[sz_a, sz_a], &[sz_a]);
        let pass_j5 = res_j5.is_ok();
        eprintln!("[bisect J5] +outer_product (fixed d type): {}", if pass_j5 { "PASS" } else { "FAIL" });

        // --- Probe J6: J5 + add(p, ko) → ns ---
        let mil_j6 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> p = mul(x=a,y=b)[name=string(\"p\")];\
            \n        tensor<fp16, [1,1,16,4]> rs = reduce_sum(x=p,axes=ax,keep_dims=kd)[name=string(\"rs\")];\
            \n        tensor<fp16, [1,16,16,4]> d = sub(x=a,y=rs)[name=string(\"d\")];\
            \n        tensor<fp16, [1,16,16,4]> ko = mul(x=b,y=d)[name=string(\"ko\")];\
            \n        tensor<fp16, [1,16,16,4]> ns = add(x=p,y=ko)[name=string(\"ns\")];\
            \n    }} -> (ns);\n}}\n"
        );
        let res_j6 = AneKernel::compile(&mil_j6, None, &[sz_a, sz_a], &[sz_a]);
        let pass_j6 = res_j6.is_ok();
        eprintln!("[bisect J6] +add(p,ko)=ns: {}", if pass_j6 { "PASS" } else { "FAIL" });

        eprintln!("  J2 (no concat):         {}", if pass_j2 { "OK" } else { "BLOCKED" });
        eprintln!("  J3 (reduce+sub):        {}", if pass_j3 { "OK" } else { "BLOCKED" });
        eprintln!("  J4 (sub C broadcast):   {}", if pass_j4 { "OK" } else { "BLOCKED" });
        eprintln!("  J5 (+outer):            {}", if pass_j5 { "OK" } else { "BLOCKED" });
        eprintln!("  J6 (+add):              {}", if pass_j6 { "OK" } else { "BLOCKED" });

        // --- Probe K: mul with 2-axis broadcast [1,16,1,4] * [1,1,16,4] → [1,16,16,4] ---
        // This is the outer product pattern in the actual GDN program
        let mil_k = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,1,4]> a,\
            \n        tensor<fp16, [1,1,16,4]> b\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=a,y=b)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_k = AneKernel::compile(&mil_k, None, &[sz_dk_hv, sz_dv_hv], &[sz_a]);
        let pass_k = res_k.is_ok();
        eprintln!("[bisect K] 2-axis broadcast mul [16,1]*[1,16]→[16,16]: {}", if pass_k { "PASS" } else { "FAIL" });

        // --- Probe K2: mul with 1-axis broadcast [1,16,16,4] * [1,1,16,4] ---
        // This is the pattern used in J5 probes (but J5 had a type bug in d)
        let mil_k2 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,1,16,4]> b\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=a,y=b)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_k2 = AneKernel::compile(&mil_k2, None, &[sz_a, sz_dv_hv], &[sz_a]);
        let pass_k2 = res_k2.is_ok();
        eprintln!("[bisect K2] 1-axis broadcast mul [16,16]*[1,16]→[16,16]: {}", if pass_k2 { "PASS" } else { "FAIL" });

        eprintln!("  K (2-axis broadcast):   {}", if pass_k { "OK" } else { "BLOCKED" });
        eprintln!("  K2 (1-axis broadcast):  {}", if pass_k2 { "OK" } else { "BLOCKED" });

        // --- Probe L: J3 + one more mul (same shape, no broadcast) ---
        let mil_l = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> p = mul(x=a,y=b)[name=string(\"p\")];\
            \n        tensor<fp16, [1,1,16,4]> rs = reduce_sum(x=p,axes=ax,keep_dims=kd)[name=string(\"rs\")];\
            \n        tensor<fp16, [1,16,16,4]> d = sub(x=a,y=rs)[name=string(\"d\")];\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=d,y=d)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_l = AneKernel::compile(&mil_l, None, &[sz_a, sz_a], &[sz_a]);
        let pass_l = res_l.is_ok();
        eprintln!("[bisect L] J3 + mul(d,d) no broadcast: {}", if pass_l { "PASS" } else { "FAIL" });

        // --- Probe L2: multiple muls, no reduce, no sub ---
        let mil_l2 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> m1 = mul(x=a,y=b)[name=string(\"m1\")];\
            \n        tensor<fp16, [1,16,16,4]> m2 = mul(x=m1,y=a)[name=string(\"m2\")];\
            \n        tensor<fp16, [1,16,16,4]> m3 = mul(x=m2,y=b)[name=string(\"m3\")];\
            \n        tensor<fp16, [1,16,16,4]> m4 = mul(x=m3,y=a)[name=string(\"m4\")];\
            \n        tensor<fp16, [1,16,16,4]> m5 = mul(x=m4,y=b)[name=string(\"m5\")];\
            \n        tensor<fp16, [1,16,16,4]> m6 = mul(x=m5,y=a)[name=string(\"m6\")];\
            \n        tensor<fp16, [1,16,16,4]> m7 = mul(x=m6,y=b)[name=string(\"m7\")];\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=m7,y=a)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_l2 = AneKernel::compile(&mil_l2, None, &[sz_a, sz_a], &[sz_a]);
        let pass_l2 = res_l2.is_ok();
        eprintln!("[bisect L2] 8 chained muls: {}", if pass_l2 { "PASS" } else { "FAIL" });

        // --- Probe L3: sub(x,y) where y has broadcast from reduce (like actual program diff = sub(v, kvm)) ---
        // Both are [1,1,16,4] so no broadcast needed. Test sub in a chain.
        let mil_l3 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,1,16,4]> b\
            \n    ) {{\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> p = mul(x=a,y=b)[name=string(\"p\")];\
            \n        tensor<fp16, [1,1,16,4]> rs = reduce_sum(x=p,axes=ax,keep_dims=kd)[name=string(\"rs\")];\
            \n        tensor<fp16, [1,1,16,4]> d = sub(x=b,y=rs)[name=string(\"d\")];\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=a,y=d)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_l3 = AneKernel::compile(&mil_l3, None, &[sz_a, sz_dv_hv], &[sz_a]);
        let pass_l3 = res_l3.is_ok();
        eprintln!("[bisect L3] sub(same-shape) + mul chain: {}", if pass_l3 { "PASS" } else { "FAIL" });

        eprintln!("  L  (J3+mul same):       {}", if pass_l { "OK" } else { "BLOCKED" });
        eprintln!("  L2 (8 muls):            {}", if pass_l2 { "OK" } else { "BLOCKED" });
        eprintln!("  L3 (sub same+mul):      {}", if pass_l3 { "OK" } else { "BLOCKED" });

        // --- Probe N: exact copy of gen_gdn_recurrence_step output (6 inputs) ---
        let mil_n = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> state,\
            \n        tensor<fp16, [1,1,1,4]> g,\
            \n        tensor<fp16, [1,1,1,4]> beta,\
            \n        tensor<fp16, [1,16,1,4]> k,\
            \n        tensor<fp16, [1,1,16,4]> v,\
            \n        tensor<fp16, [1,16,1,4]> q\
            \n    ) {{\
            \n        tensor<int32, [1]> c_ax = const()[name=string(\"cax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        int32 cat_ax = const()[name=string(\"catax\"), val=int32(1)];\
            \n        bool bF = const()[name=string(\"bF\"), val=bool(false)];\
            \n        tensor<fp16, [1,16,16,4]> decay = mul(x=state,y=g)[name=string(\"dc\")];\
            \n        tensor<fp16, [1,16,16,4]> sk = mul(x=decay,y=k)[name=string(\"sk\")];\
            \n        tensor<fp16, [1,1,16,4]> kvm = reduce_sum(x=sk,axes=c_ax,keep_dims=kd)[name=string(\"kvm\")];\
            \n        tensor<fp16, [1,1,16,4]> diff = sub(x=v,y=kvm)[name=string(\"diff\")];\
            \n        tensor<fp16, [1,1,16,4]> delta = mul(x=diff,y=beta)[name=string(\"dl\")];\
            \n        tensor<fp16, [1,16,16,4]> kdo = mul(x=k,y=delta)[name=string(\"kdo\")];\
            \n        tensor<fp16, [1,16,16,4]> ns = add(x=decay,y=kdo)[name=string(\"ns\")];\
            \n        tensor<fp16, [1,16,16,4]> sq = mul(x=ns,y=q)[name=string(\"sq\")];\
            \n        tensor<fp16, [1,1,16,4]> y = reduce_sum(x=sq,axes=c_ax,keep_dims=kd)[name=string(\"y\")];\
            \n        tensor<fp16, [1,17,16,4]> out = concat(values=(ns,y),axis=cat_ax,interleave=bF)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let n_state = 16*16*4*2;
        let n_hv = 4*2;
        let n_dk_hv = 16*4*2;
        let n_dv_hv = 16*4*2;
        let n_out = 17*16*4*2;
        let res_n = AneKernel::compile(
            &mil_n, None,
            &[n_state, n_hv, n_hv, n_dk_hv, n_dv_hv, n_dk_hv],
            &[n_out],
        );
        let pass_n = res_n.is_ok();
        eprintln!("[bisect N] exact gen_gdn program: {}", if pass_n { "PASS" } else { "FAIL" });

        // --- Probe N2: same as N but without concat (2 separate outputs) ---
        let mil_n2 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> state,\
            \n        tensor<fp16, [1,1,1,4]> g,\
            \n        tensor<fp16, [1,1,1,4]> beta,\
            \n        tensor<fp16, [1,16,1,4]> k,\
            \n        tensor<fp16, [1,1,16,4]> v,\
            \n        tensor<fp16, [1,16,1,4]> q\
            \n    ) {{\
            \n        tensor<int32, [1]> c_ax = const()[name=string(\"cax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> decay = mul(x=state,y=g)[name=string(\"dc\")];\
            \n        tensor<fp16, [1,16,16,4]> sk = mul(x=decay,y=k)[name=string(\"sk\")];\
            \n        tensor<fp16, [1,1,16,4]> kvm = reduce_sum(x=sk,axes=c_ax,keep_dims=kd)[name=string(\"kvm\")];\
            \n        tensor<fp16, [1,1,16,4]> diff = sub(x=v,y=kvm)[name=string(\"diff\")];\
            \n        tensor<fp16, [1,1,16,4]> delta = mul(x=diff,y=beta)[name=string(\"dl\")];\
            \n        tensor<fp16, [1,16,16,4]> kdo = mul(x=k,y=delta)[name=string(\"kdo\")];\
            \n        tensor<fp16, [1,16,16,4]> ns = add(x=decay,y=kdo)[name=string(\"ns\")];\
            \n        tensor<fp16, [1,16,16,4]> sq = mul(x=ns,y=q)[name=string(\"sq\")];\
            \n        tensor<fp16, [1,1,16,4]> y = reduce_sum(x=sq,axes=c_ax,keep_dims=kd)[name=string(\"y\")];\
            \n    }} -> (ns, y);\n}}\n"
        );
        let res_n2 = AneKernel::compile(
            &mil_n2, None,
            &[n_state, n_hv, n_hv, n_dk_hv, n_dv_hv, n_dk_hv],
            &[n_state, n_dv_hv],
        );
        let pass_n2 = res_n2.is_ok();
        eprintln!("[bisect N2] 6-in, no concat, 2 outputs: {}", if pass_n2 { "PASS" } else { "FAIL" });

        // --- Probe N3: same as N but output only ns (drop y branch) ---
        let mil_n3 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> state,\
            \n        tensor<fp16, [1,1,1,4]> g,\
            \n        tensor<fp16, [1,1,1,4]> beta,\
            \n        tensor<fp16, [1,16,1,4]> k,\
            \n        tensor<fp16, [1,1,16,4]> v,\
            \n        tensor<fp16, [1,16,1,4]> q\
            \n    ) {{\
            \n        tensor<int32, [1]> c_ax = const()[name=string(\"cax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> decay = mul(x=state,y=g)[name=string(\"dc\")];\
            \n        tensor<fp16, [1,16,16,4]> sk = mul(x=decay,y=k)[name=string(\"sk\")];\
            \n        tensor<fp16, [1,1,16,4]> kvm = reduce_sum(x=sk,axes=c_ax,keep_dims=kd)[name=string(\"kvm\")];\
            \n        tensor<fp16, [1,1,16,4]> diff = sub(x=v,y=kvm)[name=string(\"diff\")];\
            \n        tensor<fp16, [1,1,16,4]> delta = mul(x=diff,y=beta)[name=string(\"dl\")];\
            \n        tensor<fp16, [1,16,16,4]> kdo = mul(x=k,y=delta)[name=string(\"kdo\")];\
            \n        tensor<fp16, [1,16,16,4]> ns = add(x=decay,y=kdo)[name=string(\"ns\")];\
            \n    }} -> (ns);\n}}\n"
        );
        let res_n3 = AneKernel::compile(
            &mil_n3, None,
            &[n_state, n_hv, n_hv, n_dk_hv, n_dv_hv, n_dk_hv],
            &[n_state],
        );
        let pass_n3 = res_n3.is_ok();
        eprintln!("[bisect N3] 6-in, ns output only: {}", if pass_n3 { "PASS" } else { "FAIL" });

        eprintln!("  N  (exact program):     {}", if pass_n { "OK" } else { "BLOCKED" });
        eprintln!("  N2 (2 outputs):         {}", if pass_n2 { "OK" } else { "BLOCKED" });
        eprintln!("  N3 (ns only):           {}", if pass_n3 { "OK" } else { "BLOCKED" });

        // --- Probe N4: N3 but strip unused inputs (only state, g, beta, k, v) ---
        let mil_n4 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> state,\
            \n        tensor<fp16, [1,1,1,4]> g,\
            \n        tensor<fp16, [1,1,1,4]> beta,\
            \n        tensor<fp16, [1,16,1,4]> k,\
            \n        tensor<fp16, [1,1,16,4]> v\
            \n    ) {{\
            \n        tensor<int32, [1]> c_ax = const()[name=string(\"cax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> decay = mul(x=state,y=g)[name=string(\"dc\")];\
            \n        tensor<fp16, [1,16,16,4]> sk = mul(x=decay,y=k)[name=string(\"sk\")];\
            \n        tensor<fp16, [1,1,16,4]> kvm = reduce_sum(x=sk,axes=c_ax,keep_dims=kd)[name=string(\"kvm\")];\
            \n        tensor<fp16, [1,1,16,4]> diff = sub(x=v,y=kvm)[name=string(\"diff\")];\
            \n        tensor<fp16, [1,1,16,4]> delta = mul(x=diff,y=beta)[name=string(\"dl\")];\
            \n        tensor<fp16, [1,16,16,4]> kdo = mul(x=k,y=delta)[name=string(\"kdo\")];\
            \n        tensor<fp16, [1,16,16,4]> ns = add(x=decay,y=kdo)[name=string(\"ns\")];\
            \n    }} -> (ns);\n}}\n"
        );
        let res_n4 = AneKernel::compile(
            &mil_n4, None,
            &[n_state, n_hv, n_hv, n_dk_hv, n_dv_hv],
            &[n_state],
        );
        let pass_n4 = res_n4.is_ok();
        eprintln!("[bisect N4] 5-in, ns only (no q): {}", if pass_n4 { "PASS" } else { "FAIL" });

        // --- Probe N5: just 3 inputs (state, g, k) — minimal to get decay+sk ---
        let mil_n5 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> state,\
            \n        tensor<fp16, [1,1,1,4]> g,\
            \n        tensor<fp16, [1,16,1,4]> k\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> decay = mul(x=state,y=g)[name=string(\"dc\")];\
            \n        tensor<fp16, [1,16,16,4]> sk = mul(x=decay,y=k)[name=string(\"sk\")];\
            \n    }} -> (sk);\n}}\n"
        );
        let res_n5 = AneKernel::compile(
            &mil_n5, None,
            &[n_state, n_hv, n_dk_hv],
            &[n_state],
        );
        let pass_n5 = res_n5.is_ok();
        eprintln!("[bisect N5] 3-in, 2 muls: {}", if pass_n5 { "PASS" } else { "FAIL" });

        // --- Probe N6: 3 inputs + reduce_sum ---
        let mil_n6 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> state,\
            \n        tensor<fp16, [1,1,1,4]> g,\
            \n        tensor<fp16, [1,16,1,4]> k\
            \n    ) {{\
            \n        tensor<int32, [1]> c_ax = const()[name=string(\"cax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> decay = mul(x=state,y=g)[name=string(\"dc\")];\
            \n        tensor<fp16, [1,16,16,4]> sk = mul(x=decay,y=k)[name=string(\"sk\")];\
            \n        tensor<fp16, [1,1,16,4]> kvm = reduce_sum(x=sk,axes=c_ax,keep_dims=kd)[name=string(\"kvm\")];\
            \n    }} -> (kvm);\n}}\n"
        );
        let res_n6 = AneKernel::compile(
            &mil_n6, None,
            &[n_state, n_hv, n_dk_hv],
            &[n_dv_hv],
        );
        let pass_n6 = res_n6.is_ok();
        eprintln!("[bisect N6] 3-in, 2 muls + reduce: {}", if pass_n6 { "PASS" } else { "FAIL" });

        eprintln!("  N4 (5-in ns):           {}", if pass_n4 { "OK" } else { "BLOCKED" });
        eprintln!("  N5 (3-in 2 muls):       {}", if pass_n5 { "OK" } else { "BLOCKED" });
        eprintln!("  N6 (3-in +reduce):      {}", if pass_n6 { "OK" } else { "BLOCKED" });

        // --- Probe N7: state[1,16,16,4] * g[1,1,1,4] (just one mul, 2 inputs, mixed shapes) ---
        let mil_n7 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> state,\
            \n        tensor<fp16, [1,1,1,4]> g\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=state,y=g)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_n7 = AneKernel::compile(&mil_n7, None, &[n_state, n_hv], &[n_state]);
        let pass_n7 = res_n7.is_ok();
        eprintln!("[bisect N7] state*g (2-in, tiny g): {}", if pass_n7 { "PASS" } else { "FAIL" });

        // --- Compare: same structure as D probe but with 3 inputs ---
        eprintln!("[info] D used sz_a={sz_a}, sz_hv={sz_hv}, sz_dk_hv={sz_dk_hv}");
        eprintln!("[info] N5 used n_state={n_state}, n_hv={n_hv}, n_dk_hv={n_dk_hv}");

        // --- Probe N8: exactly probe A's structure but with [1,1,1,4] input ---
        let mil_n8 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,1,1,4]> b\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=a,y=b)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_n8 = AneKernel::compile(&mil_n8, None, &[sz_a, n_hv], &[sz_a]);
        let pass_n8 = res_n8.is_ok();
        eprintln!("[bisect N8] mul(16x16x4, 1x1x4): {}", if pass_n8 { "PASS" } else { "FAIL" });

        eprintln!("  N7 (state*g):           {}", if pass_n7 { "OK" } else { "BLOCKED" });
        eprintln!("  N8 (mul 16x16x4*1x1x4):{}", if pass_n8 { "OK" } else { "BLOCKED" });

        // --- Probe P: exact same as N7 but rename 'state' → 'st' ---
        let mil_p = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> st,\
            \n        tensor<fp16, [1,1,1,4]> g\
            \n    ) {{\
            \n        tensor<fp16, [1,16,16,4]> y = mul(x=st,y=g)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_p = AneKernel::compile(&mil_p, None, &[n_state, n_hv], &[n_state]);
        let pass_p = res_p.is_ok();
        eprintln!("[bisect P] rename 'state'→'st': {}", if pass_p { "PASS" } else { "FAIL" });

        // --- Probe M: J6 + mul(ns,b) + reduce_sum → 2nd reduce (the actual J2 chain) ---
        let mil_m = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> p = mul(x=a,y=b)[name=string(\"p\")];\
            \n        tensor<fp16, [1,1,16,4]> rs = reduce_sum(x=p,axes=ax,keep_dims=kd)[name=string(\"rs\")];\
            \n        tensor<fp16, [1,16,16,4]> d = sub(x=a,y=rs)[name=string(\"d\")];\
            \n        tensor<fp16, [1,16,16,4]> ko = mul(x=b,y=d)[name=string(\"ko\")];\
            \n        tensor<fp16, [1,16,16,4]> ns = add(x=p,y=ko)[name=string(\"ns\")];\
            \n        tensor<fp16, [1,16,16,4]> sq = mul(x=ns,y=b)[name=string(\"sq\")];\
            \n        tensor<fp16, [1,1,16,4]> y = reduce_sum(x=sq,axes=ax,keep_dims=kd)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_m = AneKernel::compile(&mil_m, None, &[sz_a, sz_a], &[sz_b_out]);
        let pass_m = res_m.is_ok();
        eprintln!("[bisect M] J6 + mul+reduce (2nd reduce): {}", if pass_m { "PASS" } else { "FAIL" });

        // --- Probe M2: just 2 reduce_sums ---
        let mil_m2 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp16, [1,16,16,4]> a,\
            \n        tensor<fp16, [1,16,16,4]> b\
            \n    ) {{\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,16,4]> p = mul(x=a,y=b)[name=string(\"p\")];\
            \n        tensor<fp16, [1,1,16,4]> rs1 = reduce_sum(x=p,axes=ax,keep_dims=kd)[name=string(\"rs1\")];\
            \n        tensor<fp16, [1,16,16,4]> p2 = mul(x=a,y=a)[name=string(\"p2\")];\
            \n        tensor<fp16, [1,1,16,4]> rs2 = reduce_sum(x=p2,axes=ax,keep_dims=kd)[name=string(\"rs2\")];\
            \n        tensor<fp16, [1,1,16,4]> y = add(x=rs1,y=rs2)[name=string(\"y\")];\
            \n    }} -> (y);\n}}\n"
        );
        let res_m2 = AneKernel::compile(&mil_m2, None, &[sz_a, sz_a], &[sz_b_out]);
        let pass_m2 = res_m2.is_ok();
        eprintln!("[bisect M2] 2x reduce_sum (parallel): {}", if pass_m2 { "PASS" } else { "FAIL" });

        eprintln!("  M  (2nd reduce):        {}", if pass_m { "OK" } else { "BLOCKED" });
        eprintln!("  M2 (2x reduce):         {}", if pass_m2 { "OK" } else { "BLOCKED" });

        // At least probe A must pass for ANE to be viable at these dims
        assert!(pass_a, "Probe A failed — 2D spatial is rejected, need [1,C,1,W] layout");
    }

    /// Eval probe: fp32 IOSurface + cast + broadcast mul + cast back.
    /// Tests whether [1,16,1,64]*[1,1,1,64] broadcast works at eval time.
    /// Key findings: mixed-channel IOSurfaces cause 0x1d with 3+ inputs,
    /// and ANE reorders IOSurface bindings alphabetically by parameter name.
    #[test]
    #[ignore = "diagnostic probe — run explicitly"]
    fn probe_broadcast_eval() {
        use crate::ane_bridge::{self, AneKernel};
        use crate::ane_mil::MIL_HEADER;

        ane_bridge::ane_init().expect("ANE init");
        ane_bridge::set_quiet(false);

        // P1: same-shape mul (no broadcast) — fp32 in/out, fp16 compute
        let mil_p1 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp32, [1,16,1,64]> a_f,\
            \n        tensor<fp32, [1,16,1,64]> b_f\
            \n    ) {{\
            \n        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\
            \n        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\
            \n        tensor<fp16, [1,16,1,64]> a = cast(dtype=to16,x=a_f)[name=string(\"ca\")];\
            \n        tensor<fp16, [1,16,1,64]> b = cast(dtype=to16,x=b_f)[name=string(\"cb\")];\
            \n        tensor<fp16, [1,16,1,64]> y = mul(x=a,y=b)[name=string(\"y\")];\
            \n        tensor<fp32, [1,16,1,64]> out = cast(dtype=to32,x=y)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let sz16_64 = 16 * 64 * 4; // fp32
        let k_p1 = AneKernel::compile(&mil_p1, None, &[sz16_64, sz16_64], &[sz16_64]);
        let pass_compile_p1 = k_p1.is_ok();
        let pass_eval_p1 = k_p1.map(|k| { k.eval().is_ok() }).unwrap_or(false);
        eprintln!("[probe P1] same-shape mul fp32-IO: compile={pass_compile_p1} eval={pass_eval_p1}");

        // P2: broadcast mul [1,16,1,64]*[1,1,1,64] — fp32 in/out
        let mil_p2 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp32, [1,16,1,64]> a_f,\
            \n        tensor<fp32, [1,1,1,64]> b_f\
            \n    ) {{\
            \n        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\
            \n        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\
            \n        tensor<fp16, [1,16,1,64]> a = cast(dtype=to16,x=a_f)[name=string(\"ca\")];\
            \n        tensor<fp16, [1,1,1,64]> b = cast(dtype=to16,x=b_f)[name=string(\"cb\")];\
            \n        tensor<fp16, [1,16,1,64]> y = mul(x=a,y=b)[name=string(\"y\")];\
            \n        tensor<fp32, [1,16,1,64]> out = cast(dtype=to32,x=y)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let sz1_64 = 64 * 4; // fp32
        let k_p2 = AneKernel::compile(&mil_p2, None, &[sz16_64, sz1_64], &[sz16_64]);
        let pass_compile_p2 = k_p2.is_ok();
        let pass_eval_p2 = k_p2.map(|k| { k.eval().is_ok() }).unwrap_or(false);
        eprintln!("[probe P2] broadcast mul fp32-IO: compile={pass_compile_p2} eval={pass_eval_p2}");

        // P3: broadcast reduce_sum [1,16,1,64] → [1,1,1,64] — fp32 in/out
        let mil_p3 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp32, [1,16,1,64]> a_f\
            \n    ) {{\
            \n        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\
            \n        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\
            \n        tensor<fp16, [1,16,1,64]> a = cast(dtype=to16,x=a_f)[name=string(\"ca\")];\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,1,1,64]> y = reduce_sum(x=a,axes=ax,keep_dims=kd)[name=string(\"y\")];\
            \n        tensor<fp32, [1,1,1,64]> out = cast(dtype=to32,x=y)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let k_p3 = AneKernel::compile(&mil_p3, None, &[sz16_64], &[sz1_64]);
        let pass_compile_p3 = k_p3.is_ok();
        let pass_eval_p3 = k_p3.map(|k| { k.eval().is_ok() }).unwrap_or(false);
        eprintln!("[probe P3] reduce_sum fp32-IO: compile={pass_compile_p3} eval={pass_eval_p3}");

        // P4: bisect state_update — first 3 ops only: decay=st*g, sk=decay*k, output=decay
        // (2 inputs st+g, 1 broadcast mul, 1 same-shape mul, output decay)
        let mil_p4 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp32, [1,16,1,64]> st_f,\
            \n        tensor<fp32, [1,1,1,64]> g_f,\
            \n        tensor<fp32, [1,16,1,64]> k_f\
            \n    ) {{\
            \n        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\
            \n        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\
            \n        tensor<fp16, [1,16,1,64]> st = cast(dtype=to16,x=st_f)[name=string(\"c0\")];\
            \n        tensor<fp16, [1,1,1,64]> g = cast(dtype=to16,x=g_f)[name=string(\"c1\")];\
            \n        tensor<fp16, [1,16,1,64]> k = cast(dtype=to16,x=k_f)[name=string(\"c2\")];\
            \n        tensor<fp16, [1,16,1,64]> decay = mul(x=st,y=g)[name=string(\"dc\")];\
            \n        tensor<fp16, [1,16,1,64]> sk = mul(x=decay,y=k)[name=string(\"sk\")];\
            \n        tensor<fp32, [1,16,1,64]> out = cast(dtype=to32,x=sk)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let k_p4 = AneKernel::compile(&mil_p4, None, &[sz16_64, sz1_64, sz16_64], &[sz16_64]);
        let pass_compile_p4 = k_p4.is_ok();
        let pass_eval_p4 = k_p4.map(|k| { k.eval().is_ok() }).unwrap_or(false);
        eprintln!("[probe P4] 3-input decay+sk: compile={pass_compile_p4} eval={pass_eval_p4}");

        // P5: add reduce_sum — decay, sk, kvm=reduce_sum(sk)
        let mil_p5 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp32, [1,16,1,64]> st_f,\
            \n        tensor<fp32, [1,1,1,64]> g_f,\
            \n        tensor<fp32, [1,16,1,64]> k_f\
            \n    ) {{\
            \n        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\
            \n        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\
            \n        tensor<fp16, [1,16,1,64]> st = cast(dtype=to16,x=st_f)[name=string(\"c0\")];\
            \n        tensor<fp16, [1,1,1,64]> g = cast(dtype=to16,x=g_f)[name=string(\"c1\")];\
            \n        tensor<fp16, [1,16,1,64]> k = cast(dtype=to16,x=k_f)[name=string(\"c2\")];\
            \n        tensor<fp16, [1,16,1,64]> decay = mul(x=st,y=g)[name=string(\"dc\")];\
            \n        tensor<fp16, [1,16,1,64]> sk = mul(x=decay,y=k)[name=string(\"sk\")];\
            \n        tensor<int32, [1]> c_ax = const()[name=string(\"cax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,1,1,64]> kvm = reduce_sum(x=sk,axes=c_ax,keep_dims=kd)[name=string(\"kvm\")];\
            \n        tensor<fp32, [1,1,1,64]> out = cast(dtype=to32,x=kvm)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let k_p5 = AneKernel::compile(&mil_p5, None, &[sz16_64, sz1_64, sz16_64], &[sz1_64]);
        let pass_compile_p5 = k_p5.is_ok();
        let pass_eval_p5 = k_p5.map(|k| { k.eval().is_ok() }).unwrap_or(false);
        eprintln!("[probe P5] 3-input decay+sk+reduce: compile={pass_compile_p5} eval={pass_eval_p5}");

        // P6: add v input — decay, sk, kvm, diff=v-kvm, delta=diff*beta
        let mil_p6 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp32, [1,16,1,64]> st_f,\
            \n        tensor<fp32, [1,1,1,64]> g_f,\
            \n        tensor<fp32, [1,16,1,64]> k_f,\
            \n        tensor<fp32, [1,1,1,64]> v_f,\
            \n        tensor<fp32, [1,1,1,64]> beta_f\
            \n    ) {{\
            \n        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\
            \n        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\
            \n        tensor<fp16, [1,16,1,64]> st = cast(dtype=to16,x=st_f)[name=string(\"c0\")];\
            \n        tensor<fp16, [1,1,1,64]> g = cast(dtype=to16,x=g_f)[name=string(\"c1\")];\
            \n        tensor<fp16, [1,16,1,64]> k = cast(dtype=to16,x=k_f)[name=string(\"c2\")];\
            \n        tensor<fp16, [1,1,1,64]> v = cast(dtype=to16,x=v_f)[name=string(\"c3\")];\
            \n        tensor<fp16, [1,1,1,64]> beta = cast(dtype=to16,x=beta_f)[name=string(\"c4\")];\
            \n        tensor<int32, [1]> c_ax = const()[name=string(\"cax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,16,1,64]> decay = mul(x=st,y=g)[name=string(\"dc\")];\
            \n        tensor<fp16, [1,16,1,64]> sk = mul(x=decay,y=k)[name=string(\"sk\")];\
            \n        tensor<fp16, [1,1,1,64]> kvm = reduce_sum(x=sk,axes=c_ax,keep_dims=kd)[name=string(\"kvm\")];\
            \n        tensor<fp16, [1,1,1,64]> diff = sub(x=v,y=kvm)[name=string(\"diff\")];\
            \n        tensor<fp16, [1,1,1,64]> delta = mul(x=diff,y=beta)[name=string(\"dl\")];\
            \n        tensor<fp16, [1,16,1,64]> kdo = mul(x=k,y=delta)[name=string(\"kdo\")];\
            \n        tensor<fp16, [1,16,1,64]> ns = add(x=decay,y=kdo)[name=string(\"ns\")];\
            \n        tensor<fp32, [1,16,1,64]> out = cast(dtype=to32,x=ns)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let k_p6 = AneKernel::compile(&mil_p6, None,
            &[sz16_64, sz1_64, sz16_64, sz1_64, sz1_64], &[sz16_64]);
        let pass_compile_p6 = k_p6.is_ok();
        let pass_eval_p6 = k_p6.map(|k| { k.eval().is_ok() }).unwrap_or(false);
        eprintln!("[probe P6] 5-input full state_update: compile={pass_compile_p6} eval={pass_eval_p6}");

        // P7: 3 same-shape fp32 inputs, 2 muls — is it 3 inputs or the broadcast?
        let mil_p7 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp32, [1,16,1,64]> a_f,\
            \n        tensor<fp32, [1,16,1,64]> b_f,\
            \n        tensor<fp32, [1,16,1,64]> c_f\
            \n    ) {{\
            \n        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\
            \n        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\
            \n        tensor<fp16, [1,16,1,64]> a = cast(dtype=to16,x=a_f)[name=string(\"c0\")];\
            \n        tensor<fp16, [1,16,1,64]> b = cast(dtype=to16,x=b_f)[name=string(\"c1\")];\
            \n        tensor<fp16, [1,16,1,64]> c = cast(dtype=to16,x=c_f)[name=string(\"c2\")];\
            \n        tensor<fp16, [1,16,1,64]> ab = mul(x=a,y=b)[name=string(\"ab\")];\
            \n        tensor<fp16, [1,16,1,64]> y = mul(x=ab,y=c)[name=string(\"y\")];\
            \n        tensor<fp32, [1,16,1,64]> out = cast(dtype=to32,x=y)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let k_p7 = AneKernel::compile(&mil_p7, None, &[sz16_64, sz16_64, sz16_64], &[sz16_64]);
        let pass_compile_p7 = k_p7.is_ok();
        let pass_eval_p7 = k_p7.map(|k| { k.eval().is_ok() }).unwrap_or(false);
        eprintln!("[probe P7] 3-input same-shape 2-mul: compile={pass_compile_p7} eval={pass_eval_p7}");

        // P8: 2 inputs, chained mul+mul (same as P4 but without the small input)
        // st*st then result*k — is it the chain or the mixed shapes?
        let mil_p8 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp32, [1,16,1,64]> a_f,\
            \n        tensor<fp32, [1,1,1,64]> b_f\
            \n    ) {{\
            \n        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\
            \n        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\
            \n        tensor<fp16, [1,16,1,64]> a = cast(dtype=to16,x=a_f)[name=string(\"c0\")];\
            \n        tensor<fp16, [1,1,1,64]> b = cast(dtype=to16,x=b_f)[name=string(\"c1\")];\
            \n        tensor<fp16, [1,16,1,64]> ab = mul(x=a,y=b)[name=string(\"ab\")];\
            \n        tensor<fp16, [1,16,1,64]> y = mul(x=ab,y=a)[name=string(\"y\")];\
            \n        tensor<fp32, [1,16,1,64]> out = cast(dtype=to32,x=y)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let k_p8 = AneKernel::compile(&mil_p8, None, &[sz16_64, sz1_64], &[sz16_64]);
        let pass_compile_p8 = k_p8.is_ok();
        let pass_eval_p8 = k_p8.map(|k| { k.eval().is_ok() }).unwrap_or(false);
        eprintln!("[probe P8] 2-input broadcast+chain mul: compile={pass_compile_p8} eval={pass_eval_p8}");

        // P9a: reduce_sum then sub with internal broadcast
        // a=[1,16,1,64], b=[1,16,1,64]. r=reduce_sum(a,axis=1)=[1,1,1,64]. out=sub(b,r).
        // Tests if internal [1,1,1,64] broadcasts correctly to [1,16,1,64] in sub.
        let mil_p9a = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp32, [1,16,1,64]> a_f,\
            \n        tensor<fp32, [1,16,1,64]> b_f\
            \n    ) {{\
            \n        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\
            \n        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\
            \n        tensor<fp16, [1,16,1,64]> a = cast(dtype=to16,x=a_f)[name=string(\"ca\")];\
            \n        tensor<fp16, [1,16,1,64]> b = cast(dtype=to16,x=b_f)[name=string(\"cb\")];\
            \n        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\
            \n        bool kd = const()[name=string(\"kd\"), val=bool(true)];\
            \n        tensor<fp16, [1,1,1,64]> r = reduce_sum(x=a,axes=ax,keep_dims=kd)[name=string(\"r\")];\
            \n        tensor<fp16, [1,16,1,64]> d = sub(x=b,y=r)[name=string(\"d\")];\
            \n        tensor<fp32, [1,16,1,64]> out = cast(dtype=to32,x=d)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let k_p9a = AneKernel::compile(&mil_p9a, None, &[sz16_64, sz16_64], &[sz16_64]);
        if let Ok(k) = &k_p9a {
            // Fill a with 1.0 in all 16 channels, b with 10.0 everywhere
            // reduce_sum(a, axis=1) = 16.0 for each spatial element
            // sub(b, r) = 10.0 - 16.0 = -6.0 for each element of all 16 channels
            let a_data: Vec<u8> = vec![1.0f32; 16 * 64].iter().flat_map(|v| v.to_le_bytes()).collect();
            let b_data: Vec<u8> = vec![10.0f32; 16 * 64].iter().flat_map(|v| v.to_le_bytes()).collect();
            k.write_input(0, &a_data);
            k.write_input(1, &b_data);
            let eval_ok = k.eval().is_ok();
            if eval_ok {
                let mut out = vec![0u8; 16 * 64 * 4];
                k.read_output(0, &mut out);
                let vals: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
                eprintln!("[probe P9a] reduce+sub broadcast: eval=true, out[0..4]={:?} (expect -6.0)", &vals[..4]);
                // Check channel 0 vs channel 8
                eprintln!("[probe P9a] ch0[0]={} ch8[0]={} ch15[0]={}", vals[0], vals[8*64], vals[15*64]);
            } else {
                eprintln!("[probe P9a] reduce+sub broadcast: compile=true eval=false");
            }
        } else {
            eprintln!("[probe P9a] reduce+sub broadcast: compile=false");
        }

        // P9: P4 but all inputs [1,16,1,64] — g expanded to same shape by caller
        let mil_p9 = format!(
            "{MIL_HEADER}    func main<ios18>(\
            \n        tensor<fp32, [1,16,1,64]> st_f,\
            \n        tensor<fp32, [1,16,1,64]> g_f,\
            \n        tensor<fp32, [1,16,1,64]> k_f\
            \n    ) {{\
            \n        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\
            \n        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\
            \n        tensor<fp16, [1,16,1,64]> st = cast(dtype=to16,x=st_f)[name=string(\"c0\")];\
            \n        tensor<fp16, [1,16,1,64]> g = cast(dtype=to16,x=g_f)[name=string(\"c1\")];\
            \n        tensor<fp16, [1,16,1,64]> k = cast(dtype=to16,x=k_f)[name=string(\"c2\")];\
            \n        tensor<fp16, [1,16,1,64]> decay = mul(x=st,y=g)[name=string(\"dc\")];\
            \n        tensor<fp16, [1,16,1,64]> sk = mul(x=decay,y=k)[name=string(\"sk\")];\
            \n        tensor<fp32, [1,16,1,64]> out = cast(dtype=to32,x=sk)[name=string(\"out\")];\
            \n    }} -> (out);\n}}\n"
        );
        let k_p9 = AneKernel::compile(&mil_p9, None, &[sz16_64, sz16_64, sz16_64], &[sz16_64]);
        let pass_compile_p9 = k_p9.is_ok();
        let pass_eval_p9 = k_p9.map(|k| { k.eval().is_ok() }).unwrap_or(false);
        eprintln!("[probe P9] P4 with all [1,16,1,64]: compile={pass_compile_p9} eval={pass_eval_p9}");

        assert!(pass_eval_p1, "P1 eval failed — basic fp32 IO broken");
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

    // -----------------------------------------------------------------------
    // Fused qkvz+ba projection parity tests (Gate 0)
    // -----------------------------------------------------------------------

    /// Synthetic parity: fused (qkvz+ba) dispatch matches two separate matmuls.
    #[test]
    fn fused_gdn_proj_parity_synthetic() {
        let ic: usize = 256;
        let qkvz_oc: usize = 512;
        let ba_oc: usize = 128;
        let s: usize = 5;
        let pad: usize = 16;

        let w_qkvz = random::uniform::<f32, f32>(
            -0.05, 0.05, &[qkvz_oc as i32, ic as i32], None,
        ).unwrap();
        w_qkvz.eval().unwrap();
        let w_ba = random::uniform::<f32, f32>(
            -0.05, 0.05, &[ba_oc as i32, ic as i32], None,
        ).unwrap();
        w_ba.eval().unwrap();
        let x = random::uniform::<f32, f32>(
            -1.0, 1.0, &[1, s as i32, ic as i32], None,
        ).unwrap();
        x.eval().unwrap();

        // Reference: two separate MLX matmuls.
        let ref_qkvz = matmul(&x, &w_qkvz.t()).unwrap();
        ref_qkvz.eval().unwrap();
        let ref_ba = matmul(&x, &w_ba.t()).unwrap();
        ref_ba.eval().unwrap();

        // Fused ANE kernel.
        let kernel = compile_fused_gdn_proj(
            w_qkvz.as_slice::<f32>(),
            w_ba.as_slice::<f32>(),
            ic, qkvz_oc, ba_oc, pad,
        ).expect("compile_fused_gdn_proj failed");

        let (ane_qkvz, ane_ba) = kernel.dispatch(&x).expect("fused dispatch");
        ane_qkvz.eval().unwrap();
        ane_ba.eval().unwrap();

        let diff_qkvz: f32 = ref_qkvz
            .subtract(&ane_qkvz).unwrap().abs().unwrap().max(None).unwrap().item();
        let diff_ba: f32 = ref_ba
            .subtract(&ane_ba).unwrap().abs().unwrap().max(None).unwrap().item();

        eprintln!(
            "[fused parity] qkvz max_diff={diff_qkvz:.6} ba max_diff={diff_ba:.6}"
        );
        assert!(
            diff_qkvz < 0.05,
            "fused qkvz parity: max_diff={diff_qkvz} (budget 0.05)"
        );
        assert!(
            diff_ba < 0.05,
            "fused ba parity: max_diff={diff_ba} (budget 0.05)"
        );
    }

    /// Donor-patch parity for fused kernels: compile donor with W1, patch with
    /// W2, verify each reproduces its own matmul pair.
    #[test]
    fn fused_gdn_proj_donor_patch_parity() {
        let ic: usize = 256;
        let qkvz_oc: usize = 512;
        let ba_oc: usize = 128;
        let s: usize = 5;
        let pad: usize = 16;

        let w1_qkvz = random::uniform::<f32, f32>(
            -0.05, 0.05, &[qkvz_oc as i32, ic as i32], None,
        ).unwrap();
        w1_qkvz.eval().unwrap();
        let w1_ba = random::uniform::<f32, f32>(
            -0.05, 0.05, &[ba_oc as i32, ic as i32], None,
        ).unwrap();
        w1_ba.eval().unwrap();

        let w2_qkvz = random::uniform::<f32, f32>(
            -0.05, 0.05, &[qkvz_oc as i32, ic as i32], None,
        ).unwrap();
        w2_qkvz.eval().unwrap();
        let w2_ba = random::uniform::<f32, f32>(
            -0.05, 0.05, &[ba_oc as i32, ic as i32], None,
        ).unwrap();
        w2_ba.eval().unwrap();

        let x = random::uniform::<f32, f32>(
            -1.0, 1.0, &[1, s as i32, ic as i32], None,
        ).unwrap();
        x.eval().unwrap();

        // Compile donor with W1.
        let donor = compile_fused_gdn_proj(
            w1_qkvz.as_slice::<f32>(),
            w1_ba.as_slice::<f32>(),
            ic, qkvz_oc, ba_oc, pad,
        ).expect("donor compile");

        let compile_before = ane_bridge::compile_count();
        let patched = compile_fused_gdn_proj_from_donor(
            &donor,
            w2_qkvz.as_slice::<f32>(),
            w2_ba.as_slice::<f32>(),
        ).expect("donor patch");
        let compile_after = ane_bridge::compile_count();
        assert_eq!(
            compile_before, compile_after,
            "fused patch_from_donor must not trigger compile (before={compile_before}, after={compile_after})"
        );

        // Donor reproduces W1.
        let (d_qkvz, d_ba) = donor.dispatch(&x).unwrap();
        d_qkvz.eval().unwrap();
        d_ba.eval().unwrap();
        let ref_q1 = matmul(&x, &w1_qkvz.t()).unwrap();
        ref_q1.eval().unwrap();
        let ref_b1 = matmul(&x, &w1_ba.t()).unwrap();
        ref_b1.eval().unwrap();
        let d_diff_q: f32 = ref_q1.subtract(&d_qkvz).unwrap().abs().unwrap().max(None).unwrap().item();
        let d_diff_b: f32 = ref_b1.subtract(&d_ba).unwrap().abs().unwrap().max(None).unwrap().item();
        assert!(d_diff_q < 0.05, "donor qkvz parity: {d_diff_q}");
        assert!(d_diff_b < 0.05, "donor ba parity: {d_diff_b}");

        // Patched reproduces W2 (NOT W1).
        let (p_qkvz, p_ba) = patched.dispatch(&x).unwrap();
        p_qkvz.eval().unwrap();
        p_ba.eval().unwrap();
        let ref_q2 = matmul(&x, &w2_qkvz.t()).unwrap();
        ref_q2.eval().unwrap();
        let ref_b2 = matmul(&x, &w2_ba.t()).unwrap();
        ref_b2.eval().unwrap();
        let p_diff_q: f32 = ref_q2.subtract(&p_qkvz).unwrap().abs().unwrap().max(None).unwrap().item();
        let p_diff_b: f32 = ref_b2.subtract(&p_ba).unwrap().abs().unwrap().max(None).unwrap().item();
        assert!(p_diff_q < 0.05, "patched qkvz parity: {p_diff_q}");
        assert!(p_diff_b < 0.05, "patched ba parity: {p_diff_b}");

        // Cross-check: patched output should differ from donor's matmul.
        let cross_q: f32 = p_qkvz.subtract(&ref_q1).unwrap().abs().unwrap().max(None).unwrap().item();
        assert!(
            cross_q > 0.05,
            "patched qkvz indistinguishable from donor (max_diff={cross_q})"
        );
    }

    /// Benchmark: fused (1 dispatch) vs separate (2 dispatches) at Carnice 9B dims.
    #[test]
    #[ignore]
    fn bench_fused_vs_separate_gdn_proj() {
        let ic: usize = 4096;
        let qkvz_oc: usize = 12288;
        let ba_oc: usize = 2064;
        let s: usize = 1;
        let pad: usize = 16;
        let iters = 200;

        let w_qkvz = random::uniform::<f32, f32>(
            -0.02, 0.02, &[qkvz_oc as i32, ic as i32], None,
        ).unwrap();
        w_qkvz.eval().unwrap();
        let w_ba = random::uniform::<f32, f32>(
            -0.02, 0.02, &[ba_oc as i32, ic as i32], None,
        ).unwrap();
        w_ba.eval().unwrap();
        let x = random::uniform::<f32, f32>(
            -1.0, 1.0, &[1, s as i32, ic as i32], None,
        ).unwrap();
        x.eval().unwrap();

        // --- Separate path: 2 dispatches ---
        let sep_qkvz = compile_proj(
            w_qkvz.as_slice::<f32>(), ic, qkvz_oc, pad, "qkvz",
        ).expect("compile qkvz");
        let sep_ba = compile_proj(
            w_ba.as_slice::<f32>(), ic, ba_oc, pad, "ba",
        ).expect("compile ba");

        // Warmup
        for _ in 0..10 {
            let _ = sep_qkvz.dispatch(&x).unwrap();
            let _ = sep_ba.dispatch(&x).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = sep_qkvz.dispatch(&x).unwrap();
            let _ = sep_ba.dispatch(&x).unwrap();
        }
        let sep_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // --- Fused path: 1 dispatch ---
        let fused = compile_fused_gdn_proj(
            w_qkvz.as_slice::<f32>(),
            w_ba.as_slice::<f32>(),
            ic, qkvz_oc, ba_oc, pad,
        ).expect("compile fused");

        // Warmup
        for _ in 0..10 {
            let _ = fused.dispatch(&x).unwrap();
        }
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = fused.dispatch(&x).unwrap();
        }
        let fused_us = t1.elapsed().as_micros() as f64 / iters as f64;

        let speedup = sep_us / fused_us;
        let saved_pct = (1.0 - fused_us / sep_us) * 100.0;
        eprintln!("\n=== Gate 0 Benchmark (S=1, ic={ic}, qkvz_oc={qkvz_oc}, ba_oc={ba_oc}) ===");
        eprintln!("Separate (2 dispatches): {sep_us:.1} us/iter");
        eprintln!("Fused    (1 dispatch):   {fused_us:.1} us/iter");
        eprintln!("Speedup: {speedup:.2}x  ({saved_pct:.1}% faster)");
        eprintln!("Per-layer savings: {:.1} us", sep_us - fused_us);
        eprintln!("24-layer savings:  {:.1} us", (sep_us - fused_us) * 24.0);
        eprintln!("===");

        assert!(
            speedup > 1.0,
            "Fused kernel must be faster than separate (speedup={speedup:.2}x)"
        );
    }

    /// Benchmark GDN recurrence: ANE 2-kernel pipeline vs Metal fused kernel
    /// at Qwen3.5-9B dimensions (Dk=128, Dv=128, Hv=32).
    ///
    /// ANE side: pre-computed g/beta (production path — gate ops stay on Metal).
    /// Metal side: fused kernel (includes g/beta computation inside).
    ///
    /// Run: cargo test -p higgs-models --features ane -- bench_gdn_recurrence_ane_vs_metal_9b --ignored --test-threads=1 --nocapture
    #[test]
    #[ignore = "benchmark — run explicitly to compare ANE vs Metal recurrence at 9B dims"]
    fn bench_gdn_recurrence_ane_vs_metal_9b() {
        use crate::ane_bridge::AneKernel;
        use crate::ane_mil::gen_gdn_recurrence_step;
        use crate::qwen3_next::gated_delta_kernel_ffi_stateless;
        use mlx_rs::{Array, Dtype, ops, transforms::eval};
        use std::time::Instant;

        let hv = 32_usize;
        let hk = 16_usize;
        let dk = 128_usize;
        let dv = 128_usize;
        let warmup = 100;
        let iters = 500;

        // ── ANE side ──
        eprintln!("=== ANE: compiling recurrence kernels (Dk={dk}, Dv={dv}, Hv={hv}) ===");
        let kerns = gen_gdn_recurrence_step(hv, dk, dv);
        let fw = kerns.flat_w;
        let big = dk * fw * 4;
        let small = fw * 4;
        eprintln!("  flat_w={fw}, big_bytes={big}, small_bytes={small}");

        let k_state_r = AneKernel::compile(
            &kerns.state_update_mil, None,
            &kerns.state_input_sizes, &kerns.state_output_sizes,
        );
        if let Err(ref e) = k_state_r {
            eprintln!("  !! state_update compile FAILED at 9B dims: {e}");
            eprintln!("  Dk={dk} × flat_w={fw} = {} fp16 elems/tensor, {} bytes IOSurface",
                dk * fw, big);
            eprintln!("  Likely exceeds ANE op-count or SRAM limit at these dimensions.");
            eprintln!("\n  Sweeping to find max compilable Dk...");
            for test_dk in [64, 32, 16, 8_usize] {
                let test_kerns = gen_gdn_recurrence_step(hv, test_dk, dv);
                match AneKernel::compile(
                    &test_kerns.state_update_mil, None,
                    &test_kerns.state_input_sizes, &test_kerns.state_output_sizes,
                ) {
                    Ok(_) => {
                        eprintln!("  ✓ Dk={test_dk} compiles (flat_w={}, {}B/tensor)",
                            test_kerns.flat_w, test_dk * test_kerns.flat_w * 4);
                    }
                    Err(_) => {
                        eprintln!("  ✗ Dk={test_dk} also fails");
                    }
                }
            }
            eprintln!("\n  Falling back to largest compilable dims for ANE bench...");
        }

        // Find the largest Dk that compiles for ANE benchmarking
        let (k_state, k_readout, ane_dk, ane_fw, ane_big, ane_small) =
            if let Ok(ks) = k_state_r {
                let kr = AneKernel::compile(
                    &kerns.readout_mil, None,
                    &kerns.readout_input_sizes, &kerns.readout_output_sizes,
                ).expect("readout compile");
                (ks, kr, dk, fw, big, small)
            } else {
                // Try smaller dims — the bottleneck is flat_w = Hv*Dv
                let candidates: &[(usize, usize, usize)] = &[
                    (hv, dk, 64),    // reduce Dv
                    (hv, dk, 32),
                    (16, dk, 64),    // reduce Hv + Dv
                    (16, 64, 64),    // reduce all
                    (4, 16, 16),     // parity-test dims
                ];
                let mut found = None;
                for &(th, td, tv) in candidates {
                    let tk = gen_gdn_recurrence_step(th, td, tv);
                    eprintln!("  trying Hv={th} Dk={td} Dv={tv} fw={}...", tk.flat_w);
                    if let Ok(ks) = AneKernel::compile(
                        &tk.state_update_mil, None,
                        &tk.state_input_sizes, &tk.state_output_sizes,
                    ) {
                        if let Ok(kr) = AneKernel::compile(
                            &tk.readout_mil, None,
                            &tk.readout_input_sizes, &tk.readout_output_sizes,
                        ) {
                            let fb = td * tk.flat_w * 4;
                            let fs = tk.flat_w * 4;
                            eprintln!("  -> compiles at Hv={th} Dk={td} Dv={tv} fw={}", tk.flat_w);
                            found = Some((ks, kr, td, tk.flat_w, fb, fs));
                            break;
                        }
                    }
                }
                found.expect("No dims compile — ANE unavailable?")
            };

        // Random fp32 IOSurface buffers
        let rand_buf = |n: usize| -> Vec<u8> {
            let arr = mlx_rs::random::uniform::<f32, f32>(
                -0.1, 0.1, &[(n / 4) as i32], None,
            ).unwrap();
            arr.eval().unwrap();
            arr.as_slice::<f32>().iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        let st_b = rand_buf(ane_big);
        let g_b = rand_buf(ane_big);
        let k_b = rand_buf(ane_big);
        let v_b = rand_buf(ane_big);
        let beta_b = rand_buf(ane_big);
        let q_b = rand_buf(ane_big);
        let mut ns_buf = vec![0u8; ane_big];
        let mut y_buf = vec![0u8; ane_small];

        let rt = AneKernel::begin_realtime();

        // Warmup
        for _ in 0..warmup {
            k_state.write_input(0, &st_b);
            k_state.write_input(1, &g_b);
            k_state.write_input(2, &k_b);
            k_state.write_input(3, &v_b);
            k_state.write_input(4, &beta_b);
            k_state.eval().unwrap();
            k_state.read_output(0, &mut ns_buf);
            k_readout.write_input(0, &ns_buf);
            k_readout.write_input(1, &q_b);
            k_readout.eval().unwrap();
            k_readout.read_output(0, &mut y_buf);
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            k_state.write_input(0, &st_b);
            k_state.write_input(1, &g_b);
            k_state.write_input(2, &k_b);
            k_state.write_input(3, &v_b);
            k_state.write_input(4, &beta_b);
            k_state.eval().unwrap();
            k_state.read_output(0, &mut ns_buf);
            k_readout.write_input(0, &ns_buf);
            k_readout.write_input(1, &q_b);
            k_readout.eval().unwrap();
            k_readout.read_output(0, &mut y_buf);
        }
        let ane_us = t0.elapsed().as_micros() as f64 / iters as f64;

        if rt { AneKernel::end_realtime(); }

        // ── Metal side ──
        eprintln!("=== Metal: gated_delta_kernel_ffi_stateless (Hk={hk}, Dk={dk}, Hv={hv}, Dv={dv}) ===");
        let q = mlx_rs::random::normal::<f32>(&[1, 1, hk as i32, dk as i32], None, None, None)
            .unwrap().as_dtype(Dtype::Float16).unwrap();
        let k_arr = mlx_rs::random::normal::<f32>(&[1, 1, hk as i32, dk as i32], None, None, None)
            .unwrap().as_dtype(Dtype::Float16).unwrap();
        let v_arr = mlx_rs::random::normal::<f32>(&[1, 1, hv as i32, dv as i32], None, None, None)
            .unwrap().as_dtype(Dtype::Float16).unwrap();
        let state = ops::zeros_dtype(&[1, hv as i32, dv as i32, dk as i32], Dtype::Float16).unwrap();
        let a_log = mlx_rs::random::normal::<f32>(&[hv as i32], None, None, None)
            .unwrap().as_dtype(Dtype::Float16).unwrap();
        let dt_bias = mlx_rs::random::normal::<f32>(&[hv as i32], None, None, None)
            .unwrap().as_dtype(Dtype::Float16).unwrap();
        let a_proj = mlx_rs::random::normal::<f32>(&[1, 1, hv as i32], None, None, None)
            .unwrap().as_dtype(Dtype::Float16).unwrap();
        let b_proj = mlx_rs::random::normal::<f32>(&[1, 1, hv as i32], None, None, None)
            .unwrap().as_dtype(Dtype::Float16).unwrap();
        eval([&q, &k_arr, &v_arr, &state, &a_log, &dt_bias, &a_proj, &b_proj]).unwrap();

        // Warmup
        for _ in 0..warmup {
            let (y, _s) = gated_delta_kernel_ffi_stateless(
                &q, &k_arr, &v_arr, &a_log, &a_proj, &dt_bias, &b_proj,
                &state, 1, 1, hk as i32, dk as i32, hv as i32, dv as i32,
            ).unwrap();
            y.eval().unwrap();
        }

        let t1 = Instant::now();
        for _ in 0..iters {
            let (y, _s) = gated_delta_kernel_ffi_stateless(
                &q, &k_arr, &v_arr, &a_log, &a_proj, &dt_bias, &b_proj,
                &state, 1, 1, hk as i32, dk as i32, hv as i32, dv as i32,
            ).unwrap();
            y.eval().unwrap();
        }
        let metal_us = t1.elapsed().as_micros() as f64 / iters as f64;

        // ── Report ──
        let ratio = metal_us / ane_us;
        let dims_match = ane_dk == dk;
        eprintln!("\n╔══════════════════════════════════════════════════════════╗");
        eprintln!("║  GDN Recurrence: ANE vs Metal                           ║");
        eprintln!("╠══════════════════════════════════════════════════════════╣");
        eprintln!("║  Metal dims (9B):  Hk={hk:>3}  Dk={dk:>4}  Hv={hv:>3}  Dv={dv:>4}       ║");
        eprintln!("║  ANE dims:         Dk={ane_dk:>4}  fw={ane_fw:>5}  {:<23}║",
            if dims_match { "(full 9B)".to_string() } else { format!("(reduced — 9B Dk={dk} fails)") });
        eprintln!("║  ANE IOSurface: state_update 5×{:.1}MB, readout 2×{:.1}MB    ║",
            ane_big as f64 / 1e6, ane_big as f64 / 1e6);
        eprintln!("╠══════════════════════════════════════════════════════════╣");
        eprintln!("║  ANE  (2 kernels, pre-computed g/beta):  {ane_us:>8.1} µs      ║");
        eprintln!("║  Metal (fused, includes g/beta compute): {metal_us:>8.1} µs      ║");
        eprintln!("║  Ratio (Metal / ANE):                    {ratio:>8.2}x         ║");
        if ratio > 1.0 {
            eprintln!("║  -> ANE is {:.0}% faster                                  ║", (ratio - 1.0) * 100.0);
        } else {
            eprintln!("║  -> Metal is {:.0}% faster                                ║", (1.0 / ratio - 1.0) * 100.0);
        }
        eprintln!("╚══════════════════════════════════════════════════════════╝");
        if !dims_match {
            eprintln!("\nWARNING: ANE could not compile at full 9B Dk={dk}.");
            eprintln!("  ANE bench uses Dk={ane_dk} — NOT a direct comparison.");
            eprintln!("  The recurrence kernel needs splitting or tiling to fit 9B dims on ANE.");
        }
        eprintln!("\nNote: ANE excludes g/beta computation (pre-computed on Metal).");
        eprintln!("      Metal uses _stateless variant (slightly favors Metal).");
    }
}
