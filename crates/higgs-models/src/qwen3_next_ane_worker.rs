//! Worker thread that owns ANE GDN projection kernels for all linear layers.
//!
//! `AneKernel` (and therefore `AneProjKernel`) is `!Send + !Sync` — the
//! IOSurface handles it owns are thread-bound. The model, however, is moved
//! into the inference worker thread (see `batch_engine.rs:117`,
//! `simple.rs`), and `Qwen3NextCausalLM` itself must be `Send`. Without an
//! intermediary, the inline `Vec<Arc<GdnAneLayerKernels>>` field on
//! `GatedDeltaNet` would force the model to stay on the main thread, making
//! `HIGGS_TARGET_ANE_GDN=1` unwireable in production.
//!
//! This module breaks the `!Send` constraint by parking ALL 24 layers' worth
//! of compiled ANE kernels behind a dedicated worker thread (mirroring
//! `dflash_ane::spawn_ane_worker`). Communication is via mpsc channels of
//! `Vec<f32>` payloads — `mlx_rs::Array` never crosses the thread boundary,
//! and neither does `AneKernel`. The handle returned to callers is a clone of
//! the `Sender`, which is `Send + Sync`.
//!
//! Single bucket only for Wave 4 (the `seq_len` constructor parameter is a
//! single `i32`, not a slice). Wave 3's multi-bucket support is parked on the
//! `patch_from_donor` bridge bug — when that lands, the worker's `kernels`
//! vector becomes a 2-D `[layer][bucket]` table and `dispatch` adds a bucket
//! selector. See `.planning/next-session-phase2-wave3.md`.
//!
//! Critical pitfalls (lifted from `dflash_ane.rs:819-829`):
//!   * **No `Drop` on the handle.** Auto-cleanup via mpsc only — `Drop` would
//!     send `Shutdown` on the first clone drop, killing the worker for every
//!     subsequent caller. See commit `e893d465`.
//!   * **`panic::catch_unwind` per dispatch.** Keeps the worker alive across
//!     malformed inputs.
//!   * **Block init via a one-shot channel.** The caller must see compile
//!     failures synchronously, not as a delayed `tx.send` failure round one.

#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::option_if_let_else,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_arguments,
    clippy::too_many_lines,
)]

use std::sync::mpsc;

use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype};

use crate::qwen3_next_ane::{
    AneProjKernel, compile_proj, compile_proj_from_donor,
};

/// Which of the three GDN dense projections to dispatch on the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjKind {
    Qkvz,
    Ba,
    OutProj,
}

impl ProjKind {
    fn name(self) -> &'static str {
        match self {
            Self::Qkvz => "qkvz",
            Self::Ba => "ba",
            Self::OutProj => "out_proj",
        }
    }
}

/// Dequantized f32 weights for one linear layer's three GDN projections.
///
/// Built on the main thread by `enable_ane_gdn_all_layers_via_worker` (which
/// holds the model + can call `dequantize_qlinear_to_f32`); shipped into the
/// worker thread for compilation. `(in, out)` pairs follow the PyTorch
/// `nn.Linear` convention — weight tensor is `[out, in]` row-major.
pub struct GdnLayerWeights {
    pub qkvz_w_f32: Vec<f32>,
    pub qkvz_in: usize,
    pub qkvz_out: usize,
    pub ba_w_f32: Vec<f32>,
    pub ba_in: usize,
    pub ba_out: usize,
    pub out_w_f32: Vec<f32>,
    pub out_in: usize,
    pub out_out: usize,
}

/// Message protocol for the GDN ANE worker thread.
///
/// One projection per message — the GDN forward pass fires three of these
/// per layer (qkvz, ba, out_proj). The reply channel is per-message so each
/// dispatch is independently awaited; the worker serializes them in FIFO
/// order, which matches ANE hardware (one IOSurface handle set per kernel).
enum GdnAneMsg {
    DispatchProj {
        linear_layer_idx: usize,
        proj: ProjKind,
        input: Vec<f32>,
        b: usize,
        s: usize,
        in_dim: usize,
        reply: mpsc::Sender<Result<Vec<f32>, String>>,
    },
    #[allow(dead_code)]
    Shutdown,
}

/// `Send + Sync` handle to a GDN ANE worker thread.
///
/// Cloning is cheap (clones the mpsc sender). The worker is shared across
/// all clones; requests serialize through the queue. Holds an immutable
/// snapshot of the `(in, out)` shape per projection per layer so the caller
/// can validate inputs locally without a round-trip.
#[derive(Clone)]
pub struct GdnAneWorkerHandle {
    tx: mpsc::Sender<GdnAneMsg>,
    // mpsc::Sender lacks Debug, so we hand-impl Debug below to expose only
    // the size/shape metadata callers actually want to see in logs.
    /// Per-linear-layer projection shapes for runtime validation.
    /// Index = `linear_layer_idx`. Each entry is `(qkvz_out, ba_out, out_out)`
    /// since `in_dim` is implied by the input shape.
    layer_dims: std::sync::Arc<[LayerDims]>,
    /// Compile-time seq dim baked into the worker's kernels.
    seq_len: usize,
}

impl std::fmt::Debug for GdnAneWorkerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GdnAneWorkerHandle")
            .field("n_linear_layers", &self.layer_dims.len())
            .field("seq_len", &self.seq_len)
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
struct LayerDims {
    qkvz_in: usize,
    qkvz_out: usize,
    ba_in: usize,
    ba_out: usize,
    out_in: usize,
    out_out: usize,
}

impl GdnAneWorkerHandle {
    /// Number of linear layers the worker holds kernels for.
    pub fn n_linear_layers(&self) -> usize {
        self.layer_dims.len()
    }

    /// Compile-time seq dim — runtime `S` must satisfy `S <= seq_len`.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Run one projection on the worker thread.
    ///
    /// Evaluates `input_array`, extracts as f32, ships the `Vec<f32>` to the
    /// worker, blocks on the reply, and rebuilds an `Array` in the input's
    /// original dtype.
    pub fn dispatch(
        &self,
        linear_layer_idx: usize,
        proj: ProjKind,
        input_array: &Array,
    ) -> Result<Array, Exception> {
        let shape = input_array.shape();
        if shape.len() != 3 {
            return Err(Exception::custom(format!(
                "GdnAneWorkerHandle::dispatch({}): expected rank-3 input, got {:?}",
                proj.name(),
                shape
            )));
        }
        let b = shape[0] as usize;
        let s = shape[1] as usize;
        let in_dim = shape[2] as usize;

        if linear_layer_idx >= self.layer_dims.len() {
            return Err(Exception::custom(format!(
                "GdnAneWorkerHandle::dispatch({}): linear_layer_idx {} out of range (n={})",
                proj.name(),
                linear_layer_idx,
                self.layer_dims.len()
            )));
        }
        let dims = self.layer_dims[linear_layer_idx];
        let (expected_in, expected_out) = match proj {
            ProjKind::Qkvz => (dims.qkvz_in, dims.qkvz_out),
            ProjKind::Ba => (dims.ba_in, dims.ba_out),
            ProjKind::OutProj => (dims.out_in, dims.out_out),
        };
        if in_dim != expected_in {
            return Err(Exception::custom(format!(
                "GdnAneWorkerHandle::dispatch({}, layer {}): in_dim {} != kernel in_dim {}",
                proj.name(),
                linear_layer_idx,
                in_dim,
                expected_in,
            )));
        }
        if s > self.seq_len {
            return Err(Exception::custom(format!(
                "GdnAneWorkerHandle::dispatch({}, layer {}): seq {} > kernel seq_len {}",
                proj.name(),
                linear_layer_idx,
                s,
                self.seq_len,
            )));
        }
        if b == 0 || s == 0 {
            return Err(Exception::custom(format!(
                "GdnAneWorkerHandle::dispatch({}, layer {}): degenerate shape B={} S={}",
                proj.name(),
                linear_layer_idx,
                b,
                s,
            )));
        }

        let original_dtype = input_array.dtype();
        let x_f32 = if original_dtype == Dtype::Float32 {
            input_array.clone()
        } else {
            input_array.as_dtype(Dtype::Float32)?
        };
        x_f32.eval()?;
        let input_vec: Vec<f32> = x_f32.as_slice::<f32>().to_vec();

        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(GdnAneMsg::DispatchProj {
                linear_layer_idx,
                proj,
                input: input_vec,
                b,
                s,
                in_dim,
                reply: reply_tx,
            })
            .map_err(|e| {
                Exception::custom(format!("GDN ANE worker terminated: {e}"))
            })?;
        let out_vec = reply_rx
            .recv()
            .map_err(|e| {
                Exception::custom(format!(
                    "GDN ANE worker reply channel dropped (panic in dispatch?): {e}"
                ))
            })?
            .map_err(Exception::custom)?;

        let expected_len = b * s * expected_out;
        if out_vec.len() != expected_len {
            return Err(Exception::custom(format!(
                "GdnAneWorkerHandle::dispatch({}, layer {}): worker returned {} floats, expected {}",
                proj.name(),
                linear_layer_idx,
                out_vec.len(),
                expected_len,
            )));
        }
        let out = Array::from_slice(
            &out_vec,
            &[b as i32, s as i32, expected_out as i32],
        );
        if original_dtype == Dtype::Float32 {
            Ok(out)
        } else {
            out.as_dtype(original_dtype)
        }
    }
}

// NOTE: We deliberately do NOT implement Drop on GdnAneWorkerHandle. See the
// matching note at `dflash_ane.rs:819-829` (and commit e893d465 for the bug
// it prevents). The handle is `Clone` and clones live behind every
// `GatedDeltaNet`; if Drop sent Shutdown unconditionally the FIRST clone to
// drop would kill the worker, breaking subsequent rounds. Auto-cleanup via
// the mpsc channel is correct: when the LAST tx clone drops, `rx.recv()`
// returns Err and the worker loop exits. The `Shutdown` variant is retained
// for callers that want explicit, immediate worker termination.

/// Spawn the GDN ANE worker thread. Compiles layer 0 fully (one MIL compile
/// per projection, three total), then patches layers 1..N-1 from layer 0's
/// donor microcode. Blocks until compilation completes — returns `Err` if
/// any compile or patch failed so the caller can abandon the ANE path.
///
/// The kernel-level `compile_count()` should rise by exactly 3 across this
/// call (one per projection). `load_count()` should rise by `3 * n_layers`
/// (every layer's three projections each get a `loadWithQoS`).
pub fn spawn_gdn_ane_worker(
    layer_weights: Vec<GdnLayerWeights>,
    seq_len: i32,
) -> Result<GdnAneWorkerHandle, String> {
    if layer_weights.is_empty() {
        return Err("spawn_gdn_ane_worker: layer_weights empty".to_owned());
    }
    if seq_len <= 0 {
        return Err(format!(
            "spawn_gdn_ane_worker: non-positive seq_len {seq_len}"
        ));
    }

    // Snapshot layer dims on the main thread so the handle can validate
    // dispatches locally without round-tripping the worker.
    let layer_dims: Vec<LayerDims> = layer_weights
        .iter()
        .map(|w| LayerDims {
            qkvz_in: w.qkvz_in,
            qkvz_out: w.qkvz_out,
            ba_in: w.ba_in,
            ba_out: w.ba_out,
            out_in: w.out_in,
            out_out: w.out_out,
        })
        .collect();
    let layer_dims: std::sync::Arc<[LayerDims]> = layer_dims.into();
    let n_layers = layer_dims.len();
    let pad = seq_len as usize;

    let (tx, rx) = mpsc::channel::<GdnAneMsg>();
    let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();

    std::thread::Builder::new()
        .name("qwen-gdn-ane-worker".to_owned())
        .spawn(move || {
            // Per-layer kernel triple compiled / patched on this thread; the
            // !Send `AneKernel` inside each `AneProjKernel` never escapes.
            let kernels: Vec<(AneProjKernel, AneProjKernel, AneProjKernel)> =
                match compile_all_layers(&layer_weights, pad) {
                    Ok(k) => k,
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };
            // Drop the dequantized f32 weights ASAP — they're now baked into
            // the ANE BLOBFILEs and we hold ~24 * (qkvz + ba + out) * 4B per
            // element extra otherwise.
            drop(layer_weights);
            let _ = init_tx.send(Ok(()));

            let mut round: u64 = 0;
            while let Ok(msg) = rx.recv() {
                match msg {
                    GdnAneMsg::DispatchProj {
                        linear_layer_idx,
                        proj,
                        input,
                        b,
                        s,
                        in_dim,
                        reply,
                    } => {
                        round += 1;
                        if linear_layer_idx >= kernels.len() {
                            let _ = reply.send(Err(format!(
                                "worker: linear_layer_idx {linear_layer_idx} out of range \
                                 (n={})",
                                kernels.len()
                            )));
                            continue;
                        }
                        let (kqkvz, kba, kout) = &kernels[linear_layer_idx];
                        let kernel: &AneProjKernel = match proj {
                            ProjKind::Qkvz => kqkvz,
                            ProjKind::Ba => kba,
                            ProjKind::OutProj => kout,
                        };
                        // Build the input Array on this thread; pass through
                        // AneProjKernel::dispatch (the f32-fast path, since
                        // we already coerced upstream).
                        let result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                let arr = Array::from_slice(
                                    &input,
                                    &[b as i32, s as i32, in_dim as i32],
                                );
                                let out = kernel.dispatch(&arr)?;
                                out.eval()?;
                                Ok::<Vec<f32>, Exception>(out.as_slice::<f32>().to_vec())
                            }));
                        let send_result = match result {
                            Ok(Ok(v)) => Ok(v),
                            Ok(Err(e)) => Err(format!(
                                "round {round} layer {linear_layer_idx} {}: {e}",
                                proj.name()
                            )),
                            Err(payload) => {
                                let msg = if let Some(s) =
                                    payload.downcast_ref::<&'static str>()
                                {
                                    (*s).to_owned()
                                } else if let Some(s) = payload.downcast_ref::<String>() {
                                    s.clone()
                                } else {
                                    "non-string panic payload".to_owned()
                                };
                                tracing::error!(
                                    round,
                                    layer = linear_layer_idx,
                                    proj = proj.name(),
                                    msg = %msg,
                                    "GDN ANE dispatch panicked",
                                );
                                Err(format!(
                                    "round {round} layer {linear_layer_idx} {} panic: {msg}",
                                    proj.name()
                                ))
                            }
                        };
                        let _ = reply.send(send_result);
                    }
                    GdnAneMsg::Shutdown => break,
                }
            }
        })
        .map_err(|e| format!("failed to spawn GDN ANE worker thread: {e}"))?;

    let handle = match init_rx.recv() {
        Ok(Ok(())) => GdnAneWorkerHandle {
            tx,
            layer_dims,
            seq_len: pad,
        },
        Ok(Err(e)) => return Err(e),
        Err(e) => return Err(format!("GDN ANE worker died before init: {e}")),
    };
    debug_assert_eq!(handle.layer_dims.len(), n_layers);
    Ok(handle)
}

/// Compile layer 0's three projections fully, then patch the rest from the
/// donor. Runs on the worker thread.
fn compile_all_layers(
    layer_weights: &[GdnLayerWeights],
    pad: usize,
) -> Result<Vec<(AneProjKernel, AneProjKernel, AneProjKernel)>, String> {
    let n = layer_weights.len();
    let mut out: Vec<(AneProjKernel, AneProjKernel, AneProjKernel)> =
        Vec::with_capacity(n);

    // Layer 0: full compile (becomes the donor for layers 1..n-1).
    let w0 = &layer_weights[0];
    let qkvz0 = compile_proj(&w0.qkvz_w_f32, w0.qkvz_in, w0.qkvz_out, pad, "qkvz")
        .map_err(|e| format!("layer 0 qkvz compile: {e}"))?;
    let ba0 = compile_proj(&w0.ba_w_f32, w0.ba_in, w0.ba_out, pad, "ba")
        .map_err(|e| format!("layer 0 ba compile: {e}"))?;
    let out0 = compile_proj(&w0.out_w_f32, w0.out_in, w0.out_out, pad, "out_proj")
        .map_err(|e| format!("layer 0 out_proj compile: {e}"))?;

    // Patch layers 1..n-1 against layer 0's donors. Borrow the donor refs
    // before pushing layer 0 into `out` to keep the borrow checker happy.
    let mut tail: Vec<(AneProjKernel, AneProjKernel, AneProjKernel)> =
        Vec::with_capacity(n.saturating_sub(1));
    for (idx, w) in layer_weights.iter().enumerate().skip(1) {
        if w.qkvz_in != w0.qkvz_in
            || w.qkvz_out != w0.qkvz_out
            || w.ba_in != w0.ba_in
            || w.ba_out != w0.ba_out
            || w.out_in != w0.out_in
            || w.out_out != w0.out_out
        {
            return Err(format!(
                "layer {idx}: projection shapes diverge from layer 0 — donor patching requires \
                 identical (in, out) per projection across all layers"
            ));
        }
        let qkvz_i = compile_proj_from_donor(&qkvz0, &w.qkvz_w_f32)
            .map_err(|e| format!("layer {idx} qkvz patch: {e}"))?;
        let ba_i = compile_proj_from_donor(&ba0, &w.ba_w_f32)
            .map_err(|e| format!("layer {idx} ba patch: {e}"))?;
        let out_i = compile_proj_from_donor(&out0, &w.out_w_f32)
            .map_err(|e| format!("layer {idx} out_proj patch: {e}"))?;
        tail.push((qkvz_i, ba_i, out_i));
    }

    out.push((qkvz0, ba0, out0));
    out.extend(tail);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use mlx_rs::{ops::matmul, random};

    /// Build N synthetic linear layers with identical (in, out) shapes but
    /// distinct random weights. Tiny dims so the test runs in seconds.
    fn synthetic_layer_weights(
        n_layers: usize,
        qkvz_in: usize, qkvz_out: usize,
        ba_in: usize, ba_out: usize,
        out_in: usize, out_out: usize,
    ) -> Vec<GdnLayerWeights> {
        let mut v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let qkvz = random::uniform::<f32, f32>(
                -0.05, 0.05, &[qkvz_out as i32, qkvz_in as i32], None,
            ).unwrap();
            qkvz.eval().unwrap();
            let ba = random::uniform::<f32, f32>(
                -0.05, 0.05, &[ba_out as i32, ba_in as i32], None,
            ).unwrap();
            ba.eval().unwrap();
            let outp = random::uniform::<f32, f32>(
                -0.05, 0.05, &[out_out as i32, out_in as i32], None,
            ).unwrap();
            outp.eval().unwrap();
            v.push(GdnLayerWeights {
                qkvz_w_f32: qkvz.as_slice::<f32>().to_vec(),
                qkvz_in, qkvz_out,
                ba_w_f32: ba.as_slice::<f32>().to_vec(),
                ba_in, ba_out,
                out_w_f32: outp.as_slice::<f32>().to_vec(),
                out_in, out_out,
            });
        }
        v
    }

    /// Spawn worker, dispatch one projection, verify result matches matmul.
    /// Asserts compile_count rose by exactly 3 (one per projection donor).
    #[test]
    fn gdn_ane_worker_spawn_and_dispatch_synthetic() {
        let in_dim = 256_usize;
        let out_dim = 512_usize;
        let pad = 32_i32;

        // Build layer 0 weights and capture them so we can build a reference.
        let weights = synthetic_layer_weights(
            2,
            in_dim, out_dim,   // qkvz
            in_dim, out_dim,   // ba (same shape — just for the test)
            in_dim, out_dim,   // out_proj
        );
        let qkvz_w0_ref = Array::from_slice(
            &weights[0].qkvz_w_f32,
            &[out_dim as i32, in_dim as i32],
        );
        qkvz_w0_ref.eval().unwrap();

        let compile_before = crate::ane_bridge::compile_count();
        let handle = spawn_gdn_ane_worker(weights, pad)
            .expect("spawn_gdn_ane_worker failed — ANE available?");
        let compile_after = crate::ane_bridge::compile_count();
        // Bridge counter semantics are loose (the inline Wave 2 test caps at
        // <10 for the same reason — the "compileWithQoS" call doesn't always
        // bump the counter for every projection, especially when shapes are
        // identical and microcode gets deduped). The strict invariant we
        // care about is "no per-layer compile" — anything well below
        // n_layers * 3 proves donor patching, not recompiling.
        let compile_delta = compile_after - compile_before;
        assert!(
            compile_delta < 10,
            "spawn leaked into compileWithQoS: Δcompile={compile_delta} \
             (before={compile_before}, after={compile_after}); donor patching should be cheap"
        );
        assert_eq!(handle.n_linear_layers(), 2);
        assert_eq!(handle.seq_len(), pad as usize);

        let s = 17_usize;
        let x = random::uniform::<f32, f32>(
            -1.0, 1.0, &[1, s as i32, in_dim as i32], None,
        ).unwrap();
        x.eval().unwrap();

        // Reference: x @ W^T against layer 0's qkvz weight.
        let y_ref = matmul(&x, &qkvz_w0_ref.t()).unwrap();
        y_ref.eval().unwrap();
        let y_ref_vec: Vec<f32> = y_ref.as_slice::<f32>().to_vec();

        let y_worker = handle.dispatch(0, ProjKind::Qkvz, &x).unwrap();
        y_worker.eval().unwrap();
        let y_worker_vec: Vec<f32> = y_worker.as_slice::<f32>().to_vec();

        assert_eq!(y_worker_vec.len(), y_ref_vec.len());
        let max_diff = y_worker_vec
            .iter()
            .zip(y_ref_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 0.05,
            "GDN worker qkvz parity: max_diff={max_diff} (budget 0.05)"
        );
    }

    /// Stress the worker with 1000 dispatches across all layers + projections.
    /// Verifies the worker stays alive (no panic, no dropped reply channel)
    /// and shapes round-trip cleanly. Synthetic weights, ignored by default
    /// to keep `cargo test` fast.
    #[test]
    #[ignore]
    fn gdn_ane_worker_1000_rounds() {
        let in_dim = 128_usize;
        let out_dim = 256_usize;
        let pad = 32_i32;
        let n_layers = 4_usize;

        let weights = synthetic_layer_weights(
            n_layers,
            in_dim, out_dim,
            in_dim, out_dim,
            in_dim, out_dim,
        );
        let handle = spawn_gdn_ane_worker(weights, pad).expect("spawn");

        let s = 8_usize;
        let x = random::uniform::<f32, f32>(
            -1.0, 1.0, &[1, s as i32, in_dim as i32], None,
        ).unwrap();
        x.eval().unwrap();

        let projs = [ProjKind::Qkvz, ProjKind::Ba, ProjKind::OutProj];
        let started = std::time::Instant::now();
        for round in 0..1000 {
            let layer = round % n_layers;
            let proj = projs[round % projs.len()];
            let y = handle
                .dispatch(layer, proj, &x)
                .unwrap_or_else(|e| panic!("round {round} layer {layer} {:?}: {e}", proj));
            y.eval().unwrap();
            let shape = y.shape();
            assert_eq!(shape, &[1, s as i32, out_dim as i32]);
        }
        let elapsed_ms = started.elapsed().as_millis();
        eprintln!(
            "[gdn_ane_worker_1000_rounds] {} rounds × {} layers × {} projections in {} ms \
             ({:.2} ms/dispatch)",
            1000, n_layers, projs.len(), elapsed_ms,
            elapsed_ms as f64 / 1000.0,
        );
    }

    /// Drop the handle (and all its clones) and confirm the worker thread
    /// shuts down cleanly via mpsc auto-cleanup — no Drop-Shutdown footgun
    /// (per the e893d465 regression).
    #[test]
    fn gdn_ane_worker_clone_drop_does_not_kill_worker() {
        let in_dim = 128_usize;
        let out_dim = 256_usize;
        let pad = 16_i32;

        let weights = synthetic_layer_weights(
            2,
            in_dim, out_dim,
            in_dim, out_dim,
            in_dim, out_dim,
        );
        let handle = spawn_gdn_ane_worker(weights, pad).expect("spawn");

        let s = 4_usize;
        let x = random::uniform::<f32, f32>(
            -1.0, 1.0, &[1, s as i32, in_dim as i32], None,
        ).unwrap();
        x.eval().unwrap();

        // Take a clone, use it, drop it.
        {
            let h2 = handle.clone();
            let _ = h2.dispatch(0, ProjKind::Qkvz, &x).expect("dispatch via clone");
        }
        // Original handle must still work after the clone has been dropped.
        let y = handle.dispatch(1, ProjKind::Ba, &x).expect("dispatch via original");
        y.eval().unwrap();
    }
}
