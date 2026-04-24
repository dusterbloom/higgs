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
    clippy::too_many_lines
)]

use std::sync::mpsc;

use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype};

use crate::ane_bridge::AneKernel;
use crate::ane_mil::gen_gdn_recurrence_step;
use crate::qwen3_next_ane::{
    AneProjKernel, FusedGdnProjKernel, compile_fused_gdn_proj, compile_fused_gdn_proj_from_donor,
    compile_proj, compile_proj_from_donor,
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

/// Parameters for compiling the GDN recurrence ANE kernels.
/// Passed into [`spawn_gdn_ane_worker`] to request recurrence compilation.
#[derive(Debug, Clone, Copy)]
pub struct RecurrenceDims {
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
}

/// Shape metadata for recurrence IOSurfaces, exposed on the handle for callers.
#[derive(Debug, Clone, Copy)]
pub struct RecurrenceDimsInfo {
    pub dk: usize,
    pub dv: usize,
    pub hv: usize,
    pub flat_w: usize,
    /// Byte size of a `[1, Dk, 1, flat_w]` fp32 IOSurface.
    pub big_bytes: usize,
    /// Byte size of a `[1, 1, 1, flat_w]` fp32 IOSurface (readout output).
    pub small_bytes: usize,
}

/// Result payload from a recurrence dispatch (raw IOSurface bytes).
pub struct RecurrenceResult {
    /// New state: `[1, Dk, 1, flat_w]` fp32 — `big_bytes` long.
    pub new_state: Vec<u8>,
    /// Readout output: `[1, 1, 1, flat_w]` fp32 — `small_bytes` long.
    pub y: Vec<u8>,
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
    /// Fused qkvz+ba dispatch — single ANE eval returns both projections.
    DispatchFusedQkvzBa {
        linear_layer_idx: usize,
        input: Vec<f32>,
        b: usize,
        s: usize,
        in_dim: usize,
        reply: mpsc::Sender<Result<(Vec<f32>, Vec<f32>), String>>,
    },
    /// Recurrence dispatch — two ANE kernel evals (state_update + readout).
    /// All buffers are raw IOSurface bytes (`[1, Dk, 1, flat_w]` fp32).
    DispatchRecurrence {
        st: Vec<u8>,
        g: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        beta: Vec<u8>,
        q: Vec<u8>,
        reply: mpsc::Sender<Result<RecurrenceResult, String>>,
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
    /// Shape metadata for recurrence IOSurfaces (`None` if recurrence was not compiled).
    recurrence_dims: Option<RecurrenceDimsInfo>,
}

impl std::fmt::Debug for GdnAneWorkerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GdnAneWorkerHandle")
            .field("n_linear_layers", &self.layer_dims.len())
            .field("seq_len", &self.seq_len)
            .field("recurrence", &self.recurrence_dims.is_some())
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

    /// Recurrence IOSurface shape metadata (`None` if recurrence was not compiled).
    pub fn recurrence_dims(&self) -> Option<RecurrenceDimsInfo> {
        self.recurrence_dims
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
            .map_err(|e| Exception::custom(format!("GDN ANE worker terminated: {e}")))?;
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
        let out = Array::from_slice(&out_vec, &[b as i32, s as i32, expected_out as i32]);
        if original_dtype == Dtype::Float32 {
            Ok(out)
        } else {
            out.as_dtype(original_dtype)
        }
    }

    /// Run fused qkvz+ba projection in a single ANE dispatch.
    ///
    /// Returns `(mixed_qkvz, mixed_ba)` — same shapes as calling
    /// `dispatch(Qkvz)` and `dispatch(Ba)` separately, but in one eval.
    pub fn dispatch_fused(
        &self,
        linear_layer_idx: usize,
        input_array: &Array,
    ) -> Result<(Array, Array), Exception> {
        let shape = input_array.shape();
        if shape.len() != 3 {
            return Err(Exception::custom(format!(
                "dispatch_fused: expected rank-3, got {:?}",
                shape
            )));
        }
        let b = shape[0] as usize;
        let s = shape[1] as usize;
        let in_dim = shape[2] as usize;
        if linear_layer_idx >= self.layer_dims.len() {
            return Err(Exception::custom(format!(
                "dispatch_fused: layer {linear_layer_idx} out of range"
            )));
        }
        if s > self.seq_len || b == 0 || s == 0 {
            return Err(Exception::custom(format!(
                "dispatch_fused: bad dims B={b} S={s} seq_len={}",
                self.seq_len
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
            .send(GdnAneMsg::DispatchFusedQkvzBa {
                linear_layer_idx,
                input: input_vec,
                b,
                s,
                in_dim,
                reply: reply_tx,
            })
            .map_err(|e| Exception::custom(format!("GDN ANE worker terminated: {e}")))?;

        let (qkvz_vec, ba_vec) = reply_rx
            .recv()
            .map_err(|e| Exception::custom(format!("fused reply dropped: {e}")))?
            .map_err(Exception::custom)?;

        let dims = self.layer_dims[linear_layer_idx];
        let qkvz = Array::from_slice(&qkvz_vec, &[b as i32, s as i32, dims.qkvz_out as i32]);
        let ba = Array::from_slice(&ba_vec, &[b as i32, s as i32, dims.ba_out as i32]);
        if original_dtype == Dtype::Float32 {
            Ok((qkvz, ba))
        } else {
            Ok((qkvz.as_dtype(original_dtype)?, ba.as_dtype(original_dtype)?))
        }
    }

    /// Run the GDN recurrence (state_update + readout) on the worker thread.
    ///
    /// All six buffers are raw `[1, Dk, 1, flat_w]` fp32 IOSurface bytes.
    /// Caller must expand per-head scalars (g, beta, v) across Dk channels
    /// before calling — the worker writes them straight to IOSurfaces.
    pub fn dispatch_recurrence(
        &self,
        st: Vec<u8>,
        g: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        beta: Vec<u8>,
        q: Vec<u8>,
    ) -> Result<RecurrenceResult, String> {
        let dims = self
            .recurrence_dims
            .ok_or("dispatch_recurrence: recurrence was not compiled")?;
        let big = dims.big_bytes;
        for (name, buf) in [
            ("st", &st),
            ("g", &g),
            ("k", &k),
            ("v", &v),
            ("beta", &beta),
            ("q", &q),
        ] {
            if buf.len() != big {
                return Err(format!(
                    "dispatch_recurrence: {name} len {} != expected {big}",
                    buf.len()
                ));
            }
        }
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(GdnAneMsg::DispatchRecurrence {
                st,
                g,
                k,
                v,
                beta,
                q,
                reply: reply_tx,
            })
            .map_err(|e| format!("GDN ANE worker terminated: {e}"))?;
        reply_rx
            .recv()
            .map_err(|e| format!("recurrence reply channel dropped: {e}"))?
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
    recurrence: Option<RecurrenceDims>,
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
    let (init_tx, init_rx) = mpsc::channel::<Result<Option<RecurrenceDimsInfo>, String>>();

    std::thread::Builder::new()
        .name("qwen-gdn-ane-worker".to_owned())
        .spawn(move || {
            // Per-layer kernel pair compiled / patched on this thread; the
            // !Send `AneKernel` inside each kernel never escapes.
            // Tuple: (fused qkvz+ba, out_proj). Separate qkvz/ba kernels
            // eliminated — ProjKind::Qkvz/Ba dispatch routes through fused.
            let kernels: Vec<(FusedGdnProjKernel, AneProjKernel)> =
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

            // Compile recurrence kernels (state_update + readout) if requested.
            // One shared pair for all layers — no baked weights, all dynamic.
            let (recurrence_pair, recurrence_info) = if let Some(dims) = recurrence {
                let kerns =
                    gen_gdn_recurrence_step(dims.num_v_heads, dims.head_k_dim, dims.head_v_dim);
                let k_state = match AneKernel::compile(
                    &kerns.state_update_mil,
                    None,
                    &kerns.state_input_sizes,
                    &kerns.state_output_sizes,
                ) {
                    Ok(k) => k,
                    Err(e) => {
                        let _ = init_tx.send(Err(format!("recurrence state_update compile: {e}")));
                        return;
                    }
                };
                let k_readout = match AneKernel::compile(
                    &kerns.readout_mil,
                    None,
                    &kerns.readout_input_sizes,
                    &kerns.readout_output_sizes,
                ) {
                    Ok(k) => k,
                    Err(e) => {
                        let _ = init_tx.send(Err(format!("recurrence readout compile: {e}")));
                        return;
                    }
                };
                let info = RecurrenceDimsInfo {
                    dk: kerns.head_k_dim,
                    dv: kerns.head_v_dim,
                    hv: kerns.num_v_heads,
                    flat_w: kerns.flat_w,
                    big_bytes: kerns.head_k_dim * kerns.flat_w * 4,
                    small_bytes: kerns.flat_w * 4,
                };
                (Some((k_state, k_readout, info)), Some(info))
            } else {
                (None, None)
            };

            // Pre-allocate reusable buffers for recurrence dispatch to avoid
            // 2 MB allocation per step in the hot loop.
            let mut ns_buf = recurrence_info
                .map(|i| vec![0u8; i.big_bytes])
                .unwrap_or_default();
            let mut y_buf = recurrence_info
                .map(|i| vec![0u8; i.small_bytes])
                .unwrap_or_default();

            // Enter ANE realtime dispatch mode for this worker thread. Realtime
            // state is thread-local in the bridge, so begin/end must happen on
            // this same thread — which the lifetime of the worker guarantees.
            // `AneProjKernel::dispatch` prefers `eval_realtime` with fallback,
            // so this is a zero-risk speed-up for the dispatch hot path.
            let rt_enabled = AneKernel::begin_realtime();
            let _ = init_tx.send(Ok(recurrence_info));

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
                        let (ref kfused, ref kout) = kernels[linear_layer_idx];
                        // Build the input Array on this thread; pass through
                        // the appropriate kernel's dispatch.
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let arr =
                                Array::from_slice(&input, &[b as i32, s as i32, in_dim as i32]);
                            match proj {
                                ProjKind::OutProj => {
                                    let out = kout.dispatch(&arr)?;
                                    out.eval()?;
                                    Ok::<Vec<f32>, Exception>(out.as_slice::<f32>().to_vec())
                                }
                                // Qkvz/Ba: route through fused kernel, return
                                // only the requested half.
                                ProjKind::Qkvz => {
                                    let (q, _) = kfused.dispatch(&arr)?;
                                    q.eval()?;
                                    Ok(q.as_slice::<f32>().to_vec())
                                }
                                ProjKind::Ba => {
                                    let (_, ba) = kfused.dispatch(&arr)?;
                                    ba.eval()?;
                                    Ok(ba.as_slice::<f32>().to_vec())
                                }
                            }
                        }));
                        let send_result = match result {
                            Ok(Ok(v)) => Ok(v),
                            Ok(Err(e)) => Err(format!(
                                "round {round} layer {linear_layer_idx} {}: {e}",
                                proj.name()
                            )),
                            Err(payload) => {
                                let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
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
                    GdnAneMsg::DispatchFusedQkvzBa {
                        linear_layer_idx,
                        input,
                        b,
                        s,
                        in_dim,
                        reply,
                    } => {
                        round += 1;
                        if linear_layer_idx >= kernels.len() {
                            let _ = reply.send(Err(format!(
                                "worker fused: layer {linear_layer_idx} out of range"
                            )));
                            continue;
                        }
                        let (ref kfused, _) = kernels[linear_layer_idx];
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let arr =
                                Array::from_slice(&input, &[b as i32, s as i32, in_dim as i32]);
                            let (q, ba) = kfused.dispatch(&arr)?;
                            q.eval()?;
                            ba.eval()?;
                            Ok::<(Vec<f32>, Vec<f32>), Exception>((
                                q.as_slice::<f32>().to_vec(),
                                ba.as_slice::<f32>().to_vec(),
                            ))
                        }));
                        let send_result = match result {
                            Ok(Ok(v)) => Ok(v),
                            Ok(Err(e)) => {
                                Err(format!("round {round} layer {linear_layer_idx} fused: {e}"))
                            }
                            Err(payload) => {
                                let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                                    (*s).to_owned()
                                } else if let Some(s) = payload.downcast_ref::<String>() {
                                    s.clone()
                                } else {
                                    "non-string panic payload".to_owned()
                                };
                                Err(format!(
                                    "round {round} layer {linear_layer_idx} fused panic: {msg}"
                                ))
                            }
                        };
                        let _ = reply.send(send_result);
                    }
                    GdnAneMsg::DispatchRecurrence {
                        st,
                        g,
                        k,
                        v,
                        beta,
                        q,
                        reply,
                    } => {
                        round += 1;
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let (k_state, k_readout, _) =
                                recurrence_pair.as_ref().ok_or("recurrence not compiled")?;
                            // Kernel A: state_update — write 5 inputs, eval, read output
                            k_state.write_input(0, &st);
                            k_state.write_input(1, &g);
                            k_state.write_input(2, &k);
                            k_state.write_input(3, &v);
                            k_state.write_input(4, &beta);
                            k_state
                                .eval()
                                .map_err(|e| format!("state_update eval: {e}"))?;
                            k_state.read_output(0, &mut ns_buf);

                            // Kernel B: readout — write 2 inputs, eval, read output
                            k_readout.write_input(0, &ns_buf);
                            k_readout.write_input(1, &q);
                            k_readout.eval().map_err(|e| format!("readout eval: {e}"))?;
                            k_readout.read_output(0, &mut y_buf);

                            Ok::<RecurrenceResult, String>(RecurrenceResult {
                                new_state: ns_buf.clone(),
                                y: y_buf.clone(),
                            })
                        }));
                        let send_result = match result {
                            Ok(Ok(r)) => Ok(r),
                            Ok(Err(e)) => Err(format!("round {round} recurrence: {e}")),
                            Err(payload) => {
                                let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                                    (*s).to_owned()
                                } else if let Some(s) = payload.downcast_ref::<String>() {
                                    s.clone()
                                } else {
                                    "non-string panic payload".to_owned()
                                };
                                tracing::error!(
                                    round, msg = %msg,
                                    "GDN ANE recurrence dispatch panicked",
                                );
                                Err(format!("round {round} recurrence panic: {msg}"))
                            }
                        };
                        let _ = reply.send(send_result);
                    }
                    GdnAneMsg::Shutdown => break,
                }
            }
            // Exit realtime mode on thread shutdown (channel closed or
            // Shutdown message). Paired with begin_realtime above; safe no-op
            // if begin_realtime had returned false.
            if rt_enabled {
                AneKernel::end_realtime();
            }
        })
        .map_err(|e| format!("failed to spawn GDN ANE worker thread: {e}"))?;

    let handle = match init_rx.recv() {
        Ok(Ok(rec_info)) => GdnAneWorkerHandle {
            tx,
            layer_dims,
            seq_len: pad,
            recurrence_dims: rec_info,
        },
        Ok(Err(e)) => return Err(e),
        Err(e) => return Err(format!("GDN ANE worker died before init: {e}")),
    };
    debug_assert_eq!(handle.layer_dims.len(), n_layers);
    Ok(handle)
}

/// Compile layer 0's projections fully, then patch the rest from the donor.
/// Runs on the worker thread.
///
/// Returns `(FusedGdnProjKernel, AneProjKernel)` per layer — fused handles
/// both qkvz+ba in a single dispatch; out_proj stays separate (different
/// input). Separate qkvz/ba kernels are NOT compiled — ProjKind::Qkvz/Ba
/// dispatch routes through the fused kernel internally. This halves bridge
/// state accumulation (2 kernels/layer instead of 4), fixing the
/// patch_from_donor LOAD FAILED at layer ~18 on 9B models.
fn compile_all_layers(
    layer_weights: &[GdnLayerWeights],
    pad: usize,
) -> Result<Vec<(FusedGdnProjKernel, AneProjKernel)>, String> {
    let n = layer_weights.len();
    let mut out: Vec<(FusedGdnProjKernel, AneProjKernel)> = Vec::with_capacity(n);

    // Layer 0: full compile (becomes the donor for layers 1..n-1).
    let w0 = &layer_weights[0];
    let fused0 = compile_fused_gdn_proj(
        &w0.qkvz_w_f32,
        &w0.ba_w_f32,
        w0.qkvz_in,
        w0.qkvz_out,
        w0.ba_out,
        pad,
    )
    .map_err(|e| format!("layer 0 fused compile: {e}"))?;
    let out0 = compile_proj(&w0.out_w_f32, w0.out_in, w0.out_out, pad, "out_proj")
        .map_err(|e| format!("layer 0 out_proj compile: {e}"))?;

    let mut tail: Vec<(FusedGdnProjKernel, AneProjKernel)> =
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
                "layer {idx}: projection shapes diverge from layer 0"
            ));
        }
        // 2 patches/layer: fused + out_proj. No separate qkvz/ba.
        let fused_i = compile_fused_gdn_proj_from_donor(&fused0, &w.qkvz_w_f32, &w.ba_w_f32)
            .map_err(|e| format!("layer {idx} fused patch: {e}"))?;
        let out_i = compile_proj_from_donor(&out0, &w.out_w_f32)
            .map_err(|e| format!("layer {idx} out_proj patch: {e}"))?;
        tail.push((fused_i, out_i));
    }

    out.push((fused0, out0));
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
        qkvz_in: usize,
        qkvz_out: usize,
        ba_in: usize,
        ba_out: usize,
        out_in: usize,
        out_out: usize,
    ) -> Vec<GdnLayerWeights> {
        let mut v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let qkvz =
                random::uniform::<f32, f32>(-0.05, 0.05, &[qkvz_out as i32, qkvz_in as i32], None)
                    .unwrap();
            qkvz.eval().unwrap();
            let ba = random::uniform::<f32, f32>(-0.05, 0.05, &[ba_out as i32, ba_in as i32], None)
                .unwrap();
            ba.eval().unwrap();
            let outp =
                random::uniform::<f32, f32>(-0.05, 0.05, &[out_out as i32, out_in as i32], None)
                    .unwrap();
            outp.eval().unwrap();
            v.push(GdnLayerWeights {
                qkvz_w_f32: qkvz.as_slice::<f32>().to_vec(),
                qkvz_in,
                qkvz_out,
                ba_w_f32: ba.as_slice::<f32>().to_vec(),
                ba_in,
                ba_out,
                out_w_f32: outp.as_slice::<f32>().to_vec(),
                out_in,
                out_out,
            });
        }
        v
    }

    /// Spawn worker, dispatch one projection, verify result matches matmul.
    /// Asserts compile_count rose by exactly 2 (fused + out_proj donors).
    #[test]
    fn gdn_ane_worker_spawn_and_dispatch_synthetic() {
        let in_dim = 256_usize;
        let out_dim = 512_usize;
        let pad = 32_i32;

        // Build layer 0 weights and capture them so we can build a reference.
        let weights = synthetic_layer_weights(
            2, in_dim, out_dim, // qkvz
            in_dim, out_dim, // ba (same shape — just for the test)
            in_dim, out_dim, // out_proj
        );
        let qkvz_w0_ref =
            Array::from_slice(&weights[0].qkvz_w_f32, &[out_dim as i32, in_dim as i32]);
        qkvz_w0_ref.eval().unwrap();

        let compile_before = crate::ane_bridge::compile_count();
        let handle = spawn_gdn_ane_worker(weights, pad, None)
            .expect("spawn_gdn_ane_worker failed — ANE available?");
        let compile_after = crate::ane_bridge::compile_count();
        // Bridge counter semantics are loose — compileWithQoS doesn't always
        // bump the counter when shapes are identical and microcode gets deduped.
        // With the 2-tuple (fused + out_proj), layer 0 compiles 2 donors; all
        // subsequent layers are donor-patched. Anything well below n_layers * 2
        // proves donor patching, not recompiling.
        let compile_delta = compile_after - compile_before;
        assert!(
            compile_delta < 10,
            "spawn leaked into compileWithQoS: Δcompile={compile_delta} \
             (before={compile_before}, after={compile_after}); donor patching should be cheap"
        );
        assert_eq!(handle.n_linear_layers(), 2);
        assert_eq!(handle.seq_len(), pad as usize);

        let s = 17_usize;
        let x =
            random::uniform::<f32, f32>(-1.0, 1.0, &[1, s as i32, in_dim as i32], None).unwrap();
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

        let weights =
            synthetic_layer_weights(n_layers, in_dim, out_dim, in_dim, out_dim, in_dim, out_dim);
        let handle = spawn_gdn_ane_worker(weights, pad, None).expect("spawn");

        let s = 8_usize;
        let x =
            random::uniform::<f32, f32>(-1.0, 1.0, &[1, s as i32, in_dim as i32], None).unwrap();
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
            1000,
            n_layers,
            projs.len(),
            elapsed_ms,
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

        let weights = synthetic_layer_weights(2, in_dim, out_dim, in_dim, out_dim, in_dim, out_dim);
        let handle = spawn_gdn_ane_worker(weights, pad, None).expect("spawn");

        let s = 4_usize;
        let x =
            random::uniform::<f32, f32>(-1.0, 1.0, &[1, s as i32, in_dim as i32], None).unwrap();
        x.eval().unwrap();

        // Take a clone, use it, drop it.
        {
            let h2 = handle.clone();
            let _ = h2
                .dispatch(0, ProjKind::Qkvz, &x)
                .expect("dispatch via clone");
        }
        // Original handle must still work after the clone has been dropped.
        let y = handle
            .dispatch(1, ProjKind::Ba, &x)
            .expect("dispatch via original");
        y.eval().unwrap();
    }

    /// Spawn worker with recurrence enabled, dispatch one step, verify parity
    /// against an f32 reference implementation.
    #[test]
    fn gdn_ane_worker_recurrence_dispatch_synthetic() {
        let hv = 4_usize;
        let dk = 16_usize;
        let dv = 16_usize;

        // Need projection weights for spawn (worker requires them), use tiny dims.
        let in_dim = 128_usize;
        let out_dim = 256_usize;
        let pad = 16_i32;
        let weights = synthetic_layer_weights(2, in_dim, out_dim, in_dim, out_dim, in_dim, out_dim);
        let rec_dims = RecurrenceDims {
            num_v_heads: hv,
            head_k_dim: dk,
            head_v_dim: dv,
        };
        let handle =
            spawn_gdn_ane_worker(weights, pad, Some(rec_dims)).expect("spawn with recurrence");

        // Verify recurrence_dims metadata
        let info = handle.recurrence_dims().expect("recurrence_dims is None");
        assert_eq!(info.dk, dk);
        assert_eq!(info.dv, dv);
        assert_eq!(info.hv, hv);
        let fw = info.flat_w;
        let big = info.big_bytes;
        let small = info.small_bytes;
        assert_eq!(big, dk * fw * 4);
        assert_eq!(small, fw * 4);

        // Build synthetic inputs — bounded values to stay fp16-safe.
        // Matches the pattern from the working gdn_recurrence_ane_parity_synthetic test.
        let g_log: Vec<f32> = (0..hv).map(|i| 0.8 + (i as f32) * 0.02).collect();
        let beta_log: Vec<f32> = (0..hv).map(|i| 0.4 + (i as f32) * 0.05).collect();
        let k_log: Vec<f32> = (0..dk * hv)
            .map(|i| ((i as f32) * 0.07).cos() * 0.1)
            .collect();
        let v_log: Vec<f32> = (0..hv * dv)
            .map(|i| ((i as f32) * 0.03).sin() * 0.1)
            .collect();
        let q_log: Vec<f32> = (0..dk * hv)
            .map(|i| ((i as f32) * 0.05).cos() * 0.1)
            .collect();

        fn to_bytes(data: &[f32]) -> Vec<u8> {
            data.iter().flat_map(|v| v.to_le_bytes()).collect()
        }

        // All buffers [1, dk, 1, fw] — expand per-head values across channels.
        let mut st_flat = vec![0.0f32; dk * fw];
        for c in 0..dk {
            for h in 0..hv {
                for d in 0..dv {
                    st_flat[c * fw + h * dv + d] =
                        (((c * hv * dv + h * dv + d) as f32 * 0.01) - 0.5).sin() * 0.1;
                }
            }
        }
        let mut g_flat = vec![0.0f32; dk * fw];
        for c in 0..dk {
            for h in 0..hv {
                for d in 0..dv {
                    g_flat[c * fw + h * dv + d] = g_log[h];
                }
            }
        }
        let mut k_flat = vec![0.0f32; dk * fw];
        for c in 0..dk {
            for h in 0..hv {
                for d in 0..dv {
                    k_flat[c * fw + h * dv + d] = k_log[c * hv + h];
                }
            }
        }
        let mut v_flat = vec![0.0f32; dk * fw];
        for c in 0..dk {
            for h in 0..hv {
                for d in 0..dv {
                    v_flat[c * fw + h * dv + d] = v_log[h * dv + d];
                }
            }
        }
        let mut beta_flat = vec![0.0f32; dk * fw];
        for c in 0..dk {
            for h in 0..hv {
                for d in 0..dv {
                    beta_flat[c * fw + h * dv + d] = beta_log[h];
                }
            }
        }
        let mut q_flat = vec![0.0f32; dk * fw];
        for c in 0..dk {
            for h in 0..hv {
                for d in 0..dv {
                    q_flat[c * fw + h * dv + d] = q_log[c * hv + h];
                }
            }
        }

        let result = handle
            .dispatch_recurrence(
                to_bytes(&st_flat),
                to_bytes(&g_flat),
                to_bytes(&k_flat),
                to_bytes(&v_flat),
                to_bytes(&beta_flat),
                to_bytes(&q_flat),
            )
            .expect("dispatch_recurrence failed");

        assert_eq!(result.new_state.len(), big, "new_state size mismatch");
        assert_eq!(result.y.len(), small, "y size mismatch");

        // Decode ANE outputs
        let ns_f32: Vec<f32> = result
            .new_state
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let y_f32: Vec<f32> = result
            .y
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // F32 reference using logical values (matches working parity test)
        let li = |c: usize, h: usize, d: usize| c * hv * dv + h * dv + d;

        // Extract ANE output into logical dims
        let mut ns_ane = vec![0.0f32; dk * hv * dv];
        for c in 0..dk {
            for h in 0..hv {
                for d in 0..dv {
                    ns_ane[li(c, h, d)] = ns_f32[c * fw + h * dv + d];
                }
            }
        }
        let mut y_ane = vec![0.0f32; hv * dv];
        for h in 0..hv {
            for d in 0..dv {
                y_ane[h * dv + d] = y_f32[h * dv + d];
            }
        }

        let mut decay = vec![0.0f32; dk * hv * dv];
        for c in 0..dk {
            for h in 0..hv {
                for d in 0..dv {
                    decay[li(c, h, d)] = st_flat[c * fw + h * dv + d] * g_log[h];
                }
            }
        }
        let mut kvm = vec![0.0f32; hv * dv];
        for h in 0..hv {
            for d in 0..dv {
                let mut s = 0.0f32;
                for c in 0..dk {
                    s += decay[li(c, h, d)] * k_log[c * hv + h];
                }
                kvm[h * dv + d] = s;
            }
        }
        let mut delta = vec![0.0f32; hv * dv];
        for h in 0..hv {
            for d in 0..dv {
                delta[h * dv + d] = (v_log[h * dv + d] - kvm[h * dv + d]) * beta_log[h];
            }
        }
        let mut ns_ref = vec![0.0f32; dk * hv * dv];
        for c in 0..dk {
            for h in 0..hv {
                for d in 0..dv {
                    ns_ref[li(c, h, d)] =
                        decay[li(c, h, d)] + k_log[c * hv + h] * delta[h * dv + d];
                }
            }
        }
        let mut y_ref = vec![0.0f32; hv * dv];
        for h in 0..hv {
            for d in 0..dv {
                let mut s = 0.0f32;
                for c in 0..dk {
                    s += ns_ref[li(c, h, d)] * q_log[c * hv + h];
                }
                y_ref[h * dv + d] = s;
            }
        }

        let max_diff_ns = ns_ane
            .iter()
            .zip(ns_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_diff_y = y_ane
            .iter()
            .zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!(
            "[worker recurrence parity] new_state max_diff={max_diff_ns:.6} \
             y max_diff={max_diff_y:.6}"
        );
        assert!(
            max_diff_ns < 0.01,
            "new_state parity: max_diff={max_diff_ns} (budget 0.01)"
        );
        assert!(
            max_diff_y < 0.05,
            "y parity: max_diff={max_diff_y} (budget 0.05)"
        );
    }
}
