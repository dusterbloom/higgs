//! DFlash drafter — ANE+CPU hybrid forward pass.
//!
//! Moves the heavy matmuls (QKV, O, gate+up, down projections) to Apple Neural
//! Engine while keeping attention, norms, RoPE on CPU.  The fc projection stays
//! on CPU BLAS because its ctx_len varies per round.
//!
//! Target: <15ms drafter latency (vs 64ms CPU BLAS), fully overlapped with the
//! 95ms GPU verify step → 52 tok/s at accept=5.
//!
//! Feature-gated behind `ane`.

#![allow(
    clippy::too_many_arguments,
    unsafe_code,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::shadow_reuse,
    clippy::shadow_unrelated
)]

use crate::ane_bridge::{self, AneKernel};
use crate::ane_mil::{self, FusedMil};
use crate::dflash_cpu::{DFlashCpuCache, DFlashCpuConfig, DFlashCpuEngine};
use crate::diffusion::{
    apply_rope, rms_norm, rms_norm_slice, sgemm, sgemm_nt, sgemm_nt_scaled, softmax_inplace,
};

// ---------------------------------------------------------------------------
// CPU ↔ ANE data layout transpose
// ---------------------------------------------------------------------------
//
// CPU stores activations as [seq, channels] row-major (position-major).
// ANE IOSurface layout is [1, channels, 1, seq] = [channels * seq] (channel-major).
// Every ANE I/O boundary needs a transpose.

/// Transpose CPU [seq, ch] → ANE [ch, seq] and convert to raw bytes.
///
/// Uses NEON 4x4 block transpose for the aligned interior and scalar for edges.
fn cpu_to_ane(data: &[f32], seq: usize, ch: usize) -> Vec<u8> {
    debug_assert_eq!(data.len(), seq * ch);
    let mut out = vec![0u8; seq * ch * 4];
    let out_f32 =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, seq * ch) };
    transpose_rc_to_cr(data, out_f32, seq, ch);
    out
}

/// Transpose ANE [ch, seq] bytes → CPU [seq, ch] f32.
///
/// Uses NEON 4x4 block transpose for the aligned interior and scalar for edges.
fn ane_to_cpu(bytes: &[u8], seq: usize, ch: usize) -> Vec<f32> {
    debug_assert_eq!(bytes.len(), seq * ch * 4);
    let src =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, seq * ch) };
    let mut out = vec![0.0f32; seq * ch];
    // ANE [ch, seq] → CPU [seq, ch] is the inverse: rows=ch, cols=seq → rows=seq, cols=ch.
    transpose_rc_to_cr(src, &mut out, ch, seq);
    out
}

/// Read a split from ANE concatenated output [total_ch, seq] and transpose to CPU [seq, sub_ch].
/// Reads channels `ch_start..ch_start+sub_ch` from the output.
fn ane_split_to_cpu(bytes: &[u8], seq: usize, total_ch: usize, ch_start: usize, sub_ch: usize) -> Vec<f32> {
    debug_assert_eq!(bytes.len(), total_ch * seq * 4);
    let src =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, total_ch * seq) };
    let mut out = vec![0.0f32; seq * sub_ch];
    // This is a strided sub-transpose: read from [total_ch, seq] starting at row ch_start,
    // sub_ch rows, and transpose to [seq, sub_ch].
    // For sub_ch and seq both >= 4, use NEON blocks on the aligned interior.
    let rows_4 = sub_ch & !3;
    let cols_4 = seq & !3;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        for r in (0..rows_4).step_by(4) {
            let r_abs = ch_start + r;
            for c in (0..cols_4).step_by(4) {
                unsafe {
                    let r0 = vld1q_f32(src.as_ptr().add(( r_abs     ) * seq + c));
                    let r1 = vld1q_f32(src.as_ptr().add(( r_abs + 1 ) * seq + c));
                    let r2 = vld1q_f32(src.as_ptr().add(( r_abs + 2 ) * seq + c));
                    let r3 = vld1q_f32(src.as_ptr().add(( r_abs + 3 ) * seq + c));
                    let t01 = vtrnq_f32(r0, r1);
                    let t23 = vtrnq_f32(r2, r3);
                    let o0 = vcombine_f32(vget_low_f32(t01.0), vget_low_f32(t23.0));
                    let o1 = vcombine_f32(vget_low_f32(t01.1), vget_low_f32(t23.1));
                    let o2 = vcombine_f32(vget_high_f32(t01.0), vget_high_f32(t23.0));
                    let o3 = vcombine_f32(vget_high_f32(t01.1), vget_high_f32(t23.1));
                    vst1q_f32(out.as_mut_ptr().add(( c     ) * sub_ch + r), o0);
                    vst1q_f32(out.as_mut_ptr().add(( c + 1 ) * sub_ch + r), o1);
                    vst1q_f32(out.as_mut_ptr().add(( c + 2 ) * sub_ch + r), o2);
                    vst1q_f32(out.as_mut_ptr().add(( c + 3 ) * sub_ch + r), o3);
                }
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (rows_4, cols_4);
    }
    // Scalar edges: remaining columns (seq not div by 4)
    for r in 0..sub_ch.min(rows_4) {
        for c in cols_4..seq {
            out[c * sub_ch + r] = src[(ch_start + r) * seq + c];
        }
    }
    // Scalar edges: remaining rows (sub_ch not div by 4)
    for r in rows_4..sub_ch {
        for c in 0..seq {
            out[c * sub_ch + r] = src[(ch_start + r) * seq + c];
        }
    }
    out
}

/// NEON-accelerated transpose: src[rows, cols] → dst[cols, rows].
///
/// Processes 4x4 blocks with NEON intrinsics, scalar fallback for edges.
fn transpose_rc_to_cr(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);
    let rows_4 = rows & !3;
    let cols_4 = cols & !3;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        for r in (0..rows_4).step_by(4) {
            for c in (0..cols_4).step_by(4) {
                unsafe {
                    // Load 4 rows of 4 elements from src [rows, cols] layout.
                    let r0 = vld1q_f32(src.as_ptr().add(( r     ) * cols + c));
                    let r1 = vld1q_f32(src.as_ptr().add(( r + 1 ) * cols + c));
                    let r2 = vld1q_f32(src.as_ptr().add(( r + 2 ) * cols + c));
                    let r3 = vld1q_f32(src.as_ptr().add(( r + 3 ) * cols + c));
                    // 4x4 in-register transpose via trn + zip.
                    let t01 = vtrnq_f32(r0, r1);
                    let t23 = vtrnq_f32(r2, r3);
                    let o0 = vcombine_f32(vget_low_f32(t01.0), vget_low_f32(t23.0));
                    let o1 = vcombine_f32(vget_low_f32(t01.1), vget_low_f32(t23.1));
                    let o2 = vcombine_f32(vget_high_f32(t01.0), vget_high_f32(t23.0));
                    let o3 = vcombine_f32(vget_high_f32(t01.1), vget_high_f32(t23.1));
                    // Store to dst [cols, rows] layout.
                    vst1q_f32(dst.as_mut_ptr().add(( c     ) * rows + r), o0);
                    vst1q_f32(dst.as_mut_ptr().add(( c + 1 ) * rows + r), o1);
                    vst1q_f32(dst.as_mut_ptr().add(( c + 2 ) * rows + r), o2);
                    vst1q_f32(dst.as_mut_ptr().add(( c + 3 ) * rows + r), o3);
                }
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (rows_4, cols_4);
        for r in 0..rows {
            for c in 0..cols {
                dst[c * rows + r] = src[r * cols + c];
            }
        }
        return;
    }
    // Scalar edges: remaining columns (cols not divisible by 4).
    for r in 0..rows.min(rows_4) {
        for c in cols_4..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    // Scalar edges: remaining rows (rows not divisible by 4).
    for r in rows_4..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

// ---------------------------------------------------------------------------
// ANE kernel set per layer
// ---------------------------------------------------------------------------

struct DFlashAneLayerKernels {
    /// Fused Q|K_noise|V_noise projection from normed noise.
    qkv: AneKernel,
    qkv_out_bytes: usize,
    /// O projection (attn_out → hidden).
    o_proj: AneKernel,
    o_out_bytes: usize,
    /// Fused SiLU(gate)*up from post-attn normed hidden.
    silu_gate_up: AneKernel,
    silu_out_bytes: usize,
    /// Down projection (activated → hidden).
    down: AneKernel,
    down_out_bytes: usize,
    /// f32 shadow of k_proj for hot-path target K BLAS (parallel to ANE QKV).
    k_proj_f32: Vec<f32>,
    /// f32 shadow of v_proj for hot-path target V BLAS (parallel to ANE QKV).
    v_proj_f32: Vec<f32>,
}

/// Pre-quantization scale applied to down_proj weights before they go to ANE,
/// and inverted (multiply by 1/scale) on ANE output read-back.
/// See compile_dflash_ane and forward() call sites.
const DOWN_PROJ_ANE_SCALE: f32 = 0.5;
const DOWN_PROJ_ANE_UNSCALE: f32 = 1.0 / DOWN_PROJ_ANE_SCALE;

/// Convert bf16 raw bits to f32 (top 16 bits of f32 layout).
fn bf16_u16_to_f32_vec(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect()
}

// ---------------------------------------------------------------------------
// ANE executor
// ---------------------------------------------------------------------------

pub struct DFlashAneExecutor {
    layer_kernels: Vec<DFlashAneLayerKernels>,
    pub cpu_engine: DFlashCpuEngine,
    config: DFlashCpuConfig,
    /// f32 shadow of fc weight for hot-path CPU BLAS (once per forward).
    fc_f32: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

/// Compile an ANE executor from an existing CPU engine.
///
/// Extracts weights from the CPU engine, generates MIL programs, and compiles
/// ANE kernels.  Layer 0 gets a full compile; layers 1..N use `patch_from_donor`
/// (same microcode, different weights → no recompilation).
pub fn compile_dflash_ane(engine: DFlashCpuEngine) -> Result<DFlashAneExecutor, String> {
    ane_bridge::ane_init()?;

    let cfg = &engine.config;
    let h = cfg.hidden;
    let q_dim = cfg.heads * cfg.head_dim;
    let kv_dim = cfg.kv_heads * cfg.head_dim;
    let inter = cfg.inter;
    let block = cfg.block_size;

    // Generate MIL programs (same for all layers — dimensions are identical).
    let qkv_mil = ane_mil::gen_fused_qkv_proj(h, q_dim, kv_dim, block);
    let o_mil = ane_mil::gen_blobfile_matmul(q_dim, h, block, "o");
    let silu_mil = ane_mil::gen_fused_silu_gate_up_proj(h, inter, block);
    let down_mil = ane_mil::gen_blobfile_matmul(inter, h, block, "d");
    if std::env::var_os("HIGGS_ANE_DUMP_MIL").is_some() {
        eprintln!("=== QKV MIL ({} weights) ===\n{}", qkv_mil.weight_names.len(), qkv_mil.mil_text);
        eprintln!("=== O MIL ({} weights) ===\n{}", o_mil.weight_names.len(), o_mil.mil_text);
        eprintln!("=== SILU MIL ({} weights) ===\n{}", silu_mil.weight_names.len(), silu_mil.mil_text);
        eprintln!("=== DOWN MIL ({} weights) ===\n{}", down_mil.weight_names.len(), down_mil.mil_text);
    }

    let n_layers = cfg.layers;
    let mut layer_kernels = Vec::with_capacity(n_layers);

    for li in 0..n_layers {
        let lw = &engine.layers[li];

        // Dequantize bf16 weights to f32 once at compile time.
        // Used for both (a) the ANE bridge blob builder (expects &[f32]) and
        // (b) the f32 shadows kept on the layer kernel for the hot-path
        // target K/V BLAS that runs in parallel with ANE QKV eval.
        let q_f32 = bf16_u16_to_f32_vec(&lw.q_proj);
        let k_f32 = bf16_u16_to_f32_vec(&lw.k_proj);
        let v_f32 = bf16_u16_to_f32_vec(&lw.v_proj);
        let o_f32 = bf16_u16_to_f32_vec(&lw.o_proj);
        let gate_f32 = bf16_u16_to_f32_vec(&lw.gate_proj);
        let up_f32 = bf16_u16_to_f32_vec(&lw.up_proj);
        // down_proj: scale weights by 0.5 to halve ANE matmul output magnitude
        // so the fp16 store doesn't saturate to Inf on wide MLPs (9B inter=12288).
        // We multiply the post-ANE CPU output by 2.0 in forward() to restore value.
        // Rationale documented in .planning/next-session-ane-9b-parity.md.
        let down_scale_factor: f32 = DOWN_PROJ_ANE_SCALE;
        let down_f32: Vec<f32> = bf16_u16_to_f32_vec(&lw.down_proj)
            .into_iter()
            .map(|w| w * down_scale_factor)
            .collect();

        // Build weight blobs (PyTorch [out, in] → ANE [in, out] via transposed).
        // Each weight is split into N BLOBFILE tiles on the oc axis if it
        // exceeds the ANE safe per-BLOBFILE budget. The tile order here must
        // match the `weight_names` order emitted by the MIL generator.
        let bq_tiles = build_tiled_weight_blobs(&q_f32, q_dim, h);
        let bk_tiles = build_tiled_weight_blobs(&k_f32, kv_dim, h);
        let bv_tiles = build_tiled_weight_blobs(&v_f32, kv_dim, h);
        let bo_tiles = build_tiled_weight_blobs(&o_f32, h, q_dim);
        let bg_tiles = build_tiled_weight_blobs(&gate_f32, inter, h);
        let bu_tiles = build_tiled_weight_blobs(&up_f32, inter, h);
        let bd_tiles = build_tiled_weight_blobs(&down_f32, h, inter);

        // Flatten per-kernel: QKV takes q||k||v, silu_gate_up takes g||u.
        let qkv_blobs: Vec<&[u8]> = bq_tiles.iter().chain(bk_tiles.iter())
            .chain(bv_tiles.iter()).map(|v| v.as_slice()).collect();
        let o_blobs: Vec<&[u8]> = bo_tiles.iter().map(|v| v.as_slice()).collect();
        let silu_blobs: Vec<&[u8]> = bg_tiles.iter().chain(bu_tiles.iter())
            .map(|v| v.as_slice()).collect();
        let down_blobs: Vec<&[u8]> = bd_tiles.iter().map(|v| v.as_slice()).collect();

        if li == 0 {
            // Full compile for layer 0.
            let qkv = compile_kernel(&qkv_mil, &qkv_blobs)?;
            let o_proj = compile_kernel(&o_mil, &o_blobs)?;
            let silu_gate_up = compile_kernel(&silu_mil, &silu_blobs)?;
            let down = compile_kernel(&down_mil, &down_blobs)?;

            layer_kernels.push(DFlashAneLayerKernels {
                qkv,
                qkv_out_bytes: qkv_mil.output_bytes,
                o_proj,
                o_out_bytes: o_mil.output_bytes,
                silu_gate_up,
                silu_out_bytes: silu_mil.output_bytes,
                down,
                down_out_bytes: down_mil.output_bytes,
                k_proj_f32: k_f32,
                v_proj_f32: v_f32,
            });
        } else {
            // Patch from layer 0 donors (skip recompilation).
            let donor = &layer_kernels[0];
            let qkv = patch_kernel(&donor.qkv, &qkv_mil, &qkv_blobs)?;
            let o_proj = patch_kernel(&donor.o_proj, &o_mil, &o_blobs)?;
            let silu_gate_up = patch_kernel(&donor.silu_gate_up, &silu_mil, &silu_blobs)?;
            let down = patch_kernel(&donor.down, &down_mil, &down_blobs)?;

            layer_kernels.push(DFlashAneLayerKernels {
                qkv,
                qkv_out_bytes: qkv_mil.output_bytes,
                o_proj,
                o_out_bytes: o_mil.output_bytes,
                silu_gate_up,
                silu_out_bytes: silu_mil.output_bytes,
                down,
                down_out_bytes: down_mil.output_bytes,
                k_proj_f32: k_f32,
                v_proj_f32: v_f32,
            });
        }
    }

    // Wire silu_gate_up → down chain: share ANE IOSurface directly,
    // eliminating the CPU round-trip (read silu output + write to down input).
    for lk in &layer_kernels {
        lk.silu_gate_up.share_output_to(0, &lk.down, 0)
            .map_err(|e| format!("silu→down share_output_to failed: {e}"))?;
    }

    let config = engine.config.clone();
    // f32 shadow of the fc weight for the hot-path ctx BLAS at forward start.
    let fc_f32 = bf16_u16_to_f32_vec(&engine.fc);
    Ok(DFlashAneExecutor {
        layer_kernels,
        cpu_engine: engine,
        config,
        fc_f32,
    })
}

/// Split a row-major [oc, ic] f32 weight into N transposed ANE blobs tiled on
/// the oc axis, matching `ane_mil::compute_blobfile_tile_plan(ic, oc)`.
///
/// The MIL generator emits `weight_names` in the same oc-tile order; the
/// caller is responsible for flattening these into the `compile_kernel`
/// weight_datas slice in the declared order (e.g. Q|K|V for the fused QKV
/// kernel).
fn build_tiled_weight_blobs(w_f32: &[f32], oc: usize, ic: usize) -> Vec<Vec<u8>> {
    let plan = ane_mil::compute_blobfile_tile_plan(ic, oc);
    let mut out = Vec::with_capacity(plan.n_tiles);
    for t in 0..plan.n_tiles {
        let start = plan.tile_start(t);
        let this_oc = plan.actual_tile_size(t);
        let slice = &w_f32[start * ic..(start + this_oc) * ic];
        out.push(ane_bridge::build_weight_blob_transposed(slice, this_oc, ic));
    }
    out
}

fn compile_kernel(mil: &FusedMil, weight_datas: &[&[u8]]) -> Result<AneKernel, String> {
    let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
    AneKernel::compile_multi_weights(
        &mil.mil_text,
        &names,
        weight_datas,
        &[mil.input_bytes],
        &[mil.output_bytes],
    )
}

fn patch_kernel(
    donor: &AneKernel,
    mil: &FusedMil,
    weight_datas: &[&[u8]],
) -> Result<AneKernel, String> {
    let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
    donor.patch_from_donor(
        &mil.mil_text,
        &names,
        weight_datas,
        &[mil.input_bytes],
        &[mil.output_bytes],
    )
}

// ---------------------------------------------------------------------------
// Hybrid forward pass
// ---------------------------------------------------------------------------

impl DFlashAneExecutor {
    /// Create a fresh KV cache for this executor.
    pub fn make_cache(&self) -> DFlashCpuCache {
        self.cpu_engine.make_cache()
    }

    /// Run the DFlash drafter forward pass: ANE projections + CPU attention/norms.
    ///
    /// Signature identical to `DFlashCpuEngine::forward` for drop-in replacement.
    pub fn forward(
        &self,
        noise: &[f32],
        taps: &[&[f32]],
        ctx_len: usize,
        cache: &mut DFlashCpuCache,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden;
        let block = cfg.block_size;
        let hd = cfg.head_dim;
        let half_hd = hd / 2;
        let n_heads = cfg.heads;
        let n_kv = cfg.kv_heads;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let gqa_ratio = n_heads / n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let cache_offset = cache.len;

        assert_eq!(taps.len(), cfg.num_taps);
        assert_eq!(noise.len(), block * h);

        let trace = std::env::var_os("HIGGS_DFLASH_ANE_TRACE").is_some();
        let mut t_fc = 0u64;
        let mut t_ane_qkv = 0u64;
        let mut t_transpose = 0u64;
        let mut t_target_kv = 0u64;
        let mut t_qk_norm_rope = 0u64;
        let mut t_sdpa = 0u64;
        let mut t_ane_o = 0u64;
        let mut t_ane_mlp = 0u64;
        let mut t_norm_residual = 0u64;
        macro_rules! tick { () => { if trace { std::time::Instant::now() } else { std::time::Instant::now() } } }
        macro_rules! tock { ($acc:ident, $t0:expr) => { if trace { $acc += $t0.elapsed().as_nanos() as u64; } } }

        // Enter ANE realtime dispatch mode for lower per-dispatch latency.
        AneKernel::begin_realtime();

        // ── fc projection (CPU BLAS — variable ctx_len) ─────────────
        let t0 = tick!();
        let fc_in = cfg.num_taps * h;
        let mut target_cat = vec![0.0f32; ctx_len * fc_in];
        for s in 0..ctx_len {
            for (t, tap) in taps.iter().enumerate() {
                let src_off = s * h;
                let dst_off = s * fc_in + t * h;
                target_cat[dst_off..dst_off + h].copy_from_slice(&tap[src_off..src_off + h]);
            }
        }
        let mut target_hidden = vec![0.0f32; ctx_len * h];
        sgemm_nt(ctx_len, h, fc_in, &target_cat, &self.fc_f32, &mut target_hidden);

        // hidden_norm
        let mut target_normed = vec![0.0f32; ctx_len * h];
        rms_norm(&target_hidden, &self.cpu_engine.hidden_norm, &mut target_normed, ctx_len, h);
        let target_hidden = target_normed;
        tock!(t_fc, t0);

        // ── Layer loop ──────────────────────────────────────────────
        let mut hidden = noise.to_vec();

        // Scratch buffers (reused across layers)
        let mut normed = vec![0.0f32; block * h];
        let mut k_ctx_buf = vec![0.0f32; ctx_len * kv_dim];
        let mut v_ctx_buf = vec![0.0f32; ctx_len * kv_dim];
        let mut attn_out = vec![0.0f32; block * q_dim];

        for (li, lk) in self.layer_kernels.iter().enumerate() {
            let lw = &self.cpu_engine.layers[li];

            // ── Attention ───────────────────────────────────────────

            // 1. RMSNorm on noise hidden state
            let t0 = tick!();
            rms_norm(&hidden, &lw.input_norm, &mut normed, block, h);
            tock!(t_norm_residual, t0);

            // 2. ANE: fused QKV from normed noise — OVERLAPPED with target K/V BLAS.
            //
            // QKV eval on ANE doesn't depend on target K/V, and target K/V BLAS
            // doesn't depend on QKV output. Run them concurrently: background
            // thread does CPU BLAS while main thread dispatches ANE eval.
            let t0 = tick!();
            let normed_ane = cpu_to_ane(&normed, block, h);
            tock!(t_transpose, t0);

            let t0 = tick!();
            let mut qkv_out = vec![0u8; lk.qkv_out_bytes];
            // Bind f32 weight slices as locals so the spawn closure only captures
            // Send-safe slices, not &DFlashAneLayerKernels (which holds ANE kernel
            // raw pointers and is !Send).
            let k_proj_f32: &[f32] = &lk.k_proj_f32;
            let v_proj_f32: &[f32] = &lk.v_proj_f32;
            std::thread::scope(|s| {
                // 3. CPU: K/V from target context (background thread)
                let blas_job = s.spawn(|| {
                    sgemm_nt(ctx_len, kv_dim, h, &target_hidden, k_proj_f32, &mut k_ctx_buf);
                    sgemm_nt(ctx_len, kv_dim, h, &target_hidden, v_proj_f32, &mut v_ctx_buf);
                });
                // ANE QKV (main thread — blocks during eval)
                lk.qkv.write_input(0, &normed_ane);
                lk.qkv.eval_realtime().expect("ANE QKV eval failed");
                lk.qkv.read_output(0, &mut qkv_out);
                blas_job.join().unwrap();
            });
            tock!(t_ane_qkv, t0);

            let t0 = tick!();
            let total_qkv_ch = q_dim + kv_dim + kv_dim;
            let q_buf = ane_split_to_cpu(&qkv_out, block, total_qkv_ch, 0, q_dim);
            let mut q_buf = q_buf; // make mutable for in-place QK norm + RoPE
            let mut k_noise_buf = ane_split_to_cpu(&qkv_out, block, total_qkv_ch, q_dim, kv_dim);
            let v_noise_buf = ane_split_to_cpu(&qkv_out, block, total_qkv_ch, q_dim + kv_dim, kv_dim);
            tock!(t_transpose, t0);

            // 4. Per-head QK norm + 5. RoPE
            let t0 = tick!();
            for s in 0..block {
                for head in 0..n_heads {
                    let off = s * q_dim + head * hd;
                    rms_norm_slice(&mut q_buf[off..off + hd], &lw.q_norm);
                }
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    rms_norm_slice(&mut k_noise_buf[off..off + hd], &lw.k_norm);
                }
            }
            for s in 0..ctx_len {
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    rms_norm_slice(&mut k_ctx_buf[off..off + hd], &lw.k_norm);
                }
            }

            for s in 0..block {
                let pos = cache_offset + ctx_len + s;
                for head in 0..n_heads {
                    let off = s * q_dim + head * hd;
                    apply_rope(
                        &mut q_buf[off..off + hd],
                        pos, half_hd,
                        &self.cpu_engine.rope_cos, &self.cpu_engine.rope_sin,
                    );
                }
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    apply_rope(
                        &mut k_noise_buf[off..off + hd],
                        pos, half_hd,
                        &self.cpu_engine.rope_cos, &self.cpu_engine.rope_sin,
                    );
                }
            }
            for s in 0..ctx_len {
                let pos = cache_offset + s;
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    apply_rope(
                        &mut k_ctx_buf[off..off + hd],
                        pos, half_hd,
                        &self.cpu_engine.rope_cos, &self.cpu_engine.rope_sin,
                    );
                }
            }
            tock!(t_qk_norm_rope, t0);

            // 6. Append target context K/V then noise K/V to cache.
            // Order: [prior_cached | ctx_k | noise_k] — matches CPU engine and MLX drafter.
            cache.k[li].extend_from_slice(&k_ctx_buf[..ctx_len * kv_dim]);
            cache.v[li].extend_from_slice(&v_ctx_buf[..ctx_len * kv_dim]);
            cache.k[li].extend_from_slice(&k_noise_buf[..block * kv_dim]);
            cache.v[li].extend_from_slice(&v_noise_buf[..block * kv_dim]);

            // 7. SDPA: Q attends to full cache (prior + ctx + noise) — non-causal.
            // total_kv_len = prior_cached + ctx_len + block = new cache length.
            let t0 = tick!();
            let total_kv_len = cache.len + ctx_len + block;
            for kv_h in 0..n_kv {
                let mut k_full = vec![0.0f32; total_kv_len * hd];
                let mut v_full = vec![0.0f32; total_kv_len * hd];

                for s in 0..total_kv_len {
                    let src_off = s * kv_dim + kv_h * hd;
                    k_full[s * hd..(s + 1) * hd]
                        .copy_from_slice(&cache.k[li][src_off..src_off + hd]);
                    v_full[s * hd..(s + 1) * hd]
                        .copy_from_slice(&cache.v[li][src_off..src_off + hd]);
                }

                for g in 0..gqa_ratio {
                    let q_h = kv_h * gqa_ratio + g;
                    let mut q_head = vec![0.0f32; block * hd];
                    for s in 0..block {
                        let qo = s * q_dim + q_h * hd;
                        q_head[s * hd..(s + 1) * hd].copy_from_slice(&q_buf[qo..qo + hd]);
                    }

                    let mut scores = vec![0.0f32; block * total_kv_len];
                    sgemm_nt_scaled(block, total_kv_len, hd, &q_head, &k_full, &mut scores, scale);

                    for row in 0..block {
                        softmax_inplace(
                            &mut scores[row * total_kv_len..(row + 1) * total_kv_len],
                        );
                    }

                    let mut ctx = vec![0.0f32; block * hd];
                    sgemm(block, hd, total_kv_len, &scores, &v_full, &mut ctx);

                    for s in 0..block {
                        let ao = s * q_dim + q_h * hd;
                        attn_out[ao..ao + hd].copy_from_slice(&ctx[s * hd..(s + 1) * hd]);
                    }
                }
            }
            tock!(t_sdpa, t0);

            // 8. ANE: O projection
            let t0 = tick!();
            let attn_ane = cpu_to_ane(&attn_out, block, q_dim);
            tock!(t_transpose, t0);

            let t0 = tick!();
            lk.o_proj.write_input(0, &attn_ane);
            lk.o_proj.eval_realtime().expect("ANE O eval failed");
            let mut o_out = vec![0u8; lk.o_out_bytes];
            lk.o_proj.read_output(0, &mut o_out);
            tock!(t_ane_o, t0);

            let t0 = tick!();
            let o_cpu = ane_to_cpu(&o_out, block, h);

            if std::env::var_os("HIGGS_DFLASH_ANE_PROBE").is_some() {
                for b in 0..block {
                    let sl = &o_cpu[b * h..(b + 1) * h];
                    let ni = sl.iter().filter(|v| v.is_infinite()).count();
                    let nn = sl.iter().filter(|v| v.is_nan()).count();
                    let mx = sl.iter().fold(0.0f32, |m, v| if v.is_finite() { m.max(v.abs()) } else { m });
                    if ni > 0 || nn > 0 {
                        eprintln!("    [L{li}] o_cpu block {b}: inf={ni}, nan={nn}, max|finite|={mx}");
                    }
                }
            }

            // Residual add
            for i in 0..block * h {
                hidden[i] += o_cpu[i];
            }
            tock!(t_norm_residual, t0);

            // ── MLP ─────────────────────────────────────────────────

            // 9. Post-attention norm
            let t0 = tick!();
            rms_norm(&hidden, &lw.post_attn_norm, &mut normed, block, h);
            tock!(t_norm_residual, t0);

            // 10-11. ANE: fused SiLU(gate)*up → down via chained eval.
            // silu_gate_up output IOSurface is wired directly to down input
            // (share_output_to at compile time), so no CPU round-trip between them.
            let t0 = tick!();
            let normed_ane = cpu_to_ane(&normed, block, h);
            tock!(t_transpose, t0);

            let t0 = tick!();
            lk.silu_gate_up.write_input(0, &normed_ane);
            AneKernel::eval_chain_realtime(&[&lk.silu_gate_up, &lk.down])
                .expect("ANE silu→down chain eval failed");
            let mut down_out = vec![0u8; lk.down_out_bytes];
            lk.down.read_output(0, &mut down_out);
            tock!(t_ane_mlp, t0);

            let t0 = tick!();
            let down_cpu = ane_to_cpu(&down_out, block, h);

            if std::env::var_os("HIGGS_DFLASH_ANE_PROBE").is_some() {
                for b in 0..block {
                    let sl = &down_cpu[b * h..(b + 1) * h];
                    let ni = sl.iter().filter(|v| v.is_infinite()).count();
                    let nn = sl.iter().filter(|v| v.is_nan()).count();
                    let mx = sl.iter().fold(0.0f32, |m, v| if v.is_finite() { m.max(v.abs()) } else { m });
                    if ni > 0 || nn > 0 {
                        eprintln!("    [L{li}] down_cpu block {b}: inf={ni}, nan={nn}, max|finite|={mx}");
                    }
                }
            }

            // Residual add. down weights were pre-scaled by DOWN_PROJ_ANE_SCALE
            // so the ANE fp16 output stays in range; restore magnitude here in
            // fp32 before accumulating into the hidden state.
            for i in 0..block * h {
                hidden[i] += down_cpu[i] * DOWN_PROJ_ANE_UNSCALE;
            }
            tock!(t_norm_residual, t0);
        }

        // Exit ANE realtime dispatch mode.
        AneKernel::end_realtime();

        if trace {
            let us = |ns: u64| ns as f64 / 1000.0;
            eprintln!("  DFlash ANE trace (ctx={ctx_len}):");
            eprintln!("    fc_proj:       {:>8.0}us", us(t_fc));
            eprintln!("    ane_qkv:       {:>8.0}us (5 layers)", us(t_ane_qkv));
            eprintln!("    transpose:     {:>8.0}us", us(t_transpose));
            eprintln!("    target_kv:     {:>8.0}us (BLAS K/V)", us(t_target_kv));
            eprintln!("    qk_norm+rope:  {:>8.0}us", us(t_qk_norm_rope));
            eprintln!("    sdpa:          {:>8.0}us", us(t_sdpa));
            eprintln!("    ane_o:         {:>8.0}us", us(t_ane_o));
            eprintln!("    ane_mlp:       {:>8.0}us (silu+down chain)", us(t_ane_mlp));
            eprintln!("    norm+residual: {:>8.0}us", us(t_norm_residual));
            let total = t_fc + t_ane_qkv + t_transpose + t_target_kv + t_qk_norm_rope
                + t_sdpa + t_ane_o + t_ane_mlp + t_norm_residual;
            eprintln!("    TOTAL:         {:>8.0}us", us(total));
        }

        // Update cache length: each layer got ctx_len + block new entries this round
        // (target context K/V followed by noise K/V).
        cache.len += ctx_len + block;

        // Final RMSNorm
        let mut output = vec![0.0f32; block * h];
        rms_norm(&hidden, &self.cpu_engine.final_norm, &mut output, block, h);

        output
    }
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------
//
// `AneKernel` holds IOSurface handles that are thread-bound, so
// `DFlashAneExecutor` is `!Send + !Sync`.  That is incompatible with axum's
// `AppState` requiring `Send + Sync + 'static` and with `std::thread::spawn`
// moving the executor across threads for drafter pipelining.
//
// Solution: pin the executor to a single worker thread.  The executor is
// constructed *on* that thread (never crosses a thread boundary), and the
// outside world interacts with it through an mpsc channel.  The handle is a
// plain `Sender` clone, which is `Send + Sync`.

enum AneWorkerMsg {
    Forward {
        noise: Vec<f32>,
        taps: Vec<Vec<f32>>,
        ctx_len: usize,
        cache: DFlashCpuCache,
        reply: std::sync::mpsc::Sender<(Vec<f32>, DFlashCpuCache)>,
    },
    Shutdown,
}

/// `Send + Sync` handle to a DFlash ANE worker thread.
///
/// Cloning the handle is cheap (clones the mpsc sender); the worker thread is
/// shared across all clones.  Requests are serialized through the mpsc queue —
/// ANE dispatch is inherently single-threaded anyway (one IOSurface handle set
/// per executor), so FIFO serialization matches the hardware model.
#[derive(Clone)]
pub struct DFlashAneWorkerHandle {
    tx: std::sync::mpsc::Sender<AneWorkerMsg>,
    config: DFlashCpuConfig,
}

impl DFlashAneWorkerHandle {
    /// Run a forward pass on the worker thread.
    ///
    /// Takes ownership of `cache` and returns it alongside the output so the
    /// caller can swap it back into its local slot without interior mutability.
    pub fn forward(
        &self,
        noise: Vec<f32>,
        taps: Vec<Vec<f32>>,
        ctx_len: usize,
        cache: DFlashCpuCache,
    ) -> (Vec<f32>, DFlashCpuCache) {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.tx
            .send(AneWorkerMsg::Forward {
                noise,
                taps,
                ctx_len,
                cache,
                reply: reply_tx,
            })
            .expect("ANE worker thread terminated unexpectedly");
        reply_rx
            .recv()
            .expect("ANE worker reply channel dropped")
    }

    pub fn config(&self) -> &DFlashCpuConfig {
        &self.config
    }
}

impl Drop for DFlashAneWorkerHandle {
    fn drop(&mut self) {
        // Best-effort shutdown.  If this is the last sender clone the worker
        // will exit on channel close; the explicit Shutdown message just
        // short-circuits the `recv()` wait when other clones outlive the
        // request traffic.
        let _ = self.tx.send(AneWorkerMsg::Shutdown);
    }
}

/// Spawn a DFlash ANE worker thread that owns a freshly-compiled executor.
///
/// The executor is compiled *inside* the worker thread so the underlying
/// `AneKernel` IOSurface handles never cross a thread boundary.  The caller
/// passes the `DFlashCpuEngine` by value (it is `Send`); compilation happens
/// on the worker and the returned handle is `Send + Sync`.
///
/// Blocks until compilation completes — returns `Err` if the ANE compile
/// failed so the caller can fall back to CPU BLAS.
pub fn spawn_ane_worker(
    cpu_engine: DFlashCpuEngine,
) -> Result<DFlashAneWorkerHandle, String> {
    let config = cpu_engine.config.clone();
    let (tx, rx) = std::sync::mpsc::channel::<AneWorkerMsg>();
    let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    std::thread::Builder::new()
        .name("dflash-ane-worker".to_owned())
        .spawn(move || {
            // Compile on-thread — !Send AneKernel never leaves this stack.
            let executor = match compile_dflash_ane(cpu_engine) {
                Ok(e) => e,
                Err(e) => {
                    let _ = init_tx.send(Err(e));
                    return;
                }
            };
            let _ = init_tx.send(Ok(()));

            while let Ok(msg) = rx.recv() {
                match msg {
                    AneWorkerMsg::Forward {
                        noise,
                        taps,
                        ctx_len,
                        mut cache,
                        reply,
                    } => {
                        let tap_slices: Vec<&[f32]> =
                            taps.iter().map(Vec::as_slice).collect();
                        let out = executor.forward(&noise, &tap_slices, ctx_len, &mut cache);
                        let _ = reply.send((out, cache));
                    }
                    AneWorkerMsg::Shutdown => break,
                }
            }
        })
        .map_err(|e| format!("failed to spawn ANE worker thread: {e}"))?;

    match init_rx.recv() {
        Ok(Ok(())) => Ok(DFlashAneWorkerHandle { tx, config }),
        Ok(Err(e)) => Err(e),
        Err(e) => Err(format!("ANE worker died before init: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// Load a DFlash drafter and extract its CPU engine.
    ///
    /// Default: 4B drafter. Override with `DFLASH_ANE_DRAFTER_SNAP_DIR=<hub-dir>`
    /// to point at a different `models--…/snapshots` root (uses the first
    /// snapshot found).
    fn load_4b_engine() -> DFlashCpuEngine {
        let snap_root = std::env::var("DFLASH_ANE_DRAFTER_SNAP_DIR").unwrap_or_else(|_| {
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots"
                .to_string()
        });
        // If snap_root directly contains config.json, use it; otherwise treat
        // as a HF-style snapshots root and pick the first child.
        let root_pb = std::path::PathBuf::from(&snap_root);
        let snap_dir = if root_pb.join("config.json").exists() {
            root_pb
        } else {
            std::fs::read_dir(&snap_root)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path()
        };
        eprintln!("DFlash drafter snap: {}", snap_dir.display());
        let drafter = crate::dflash::load_dflash_drafter(&snap_dir).unwrap();
        crate::dflash_cpu::extract_dflash_cpu_engine(&drafter)
    }

    /// Parity test: ANE forward must match CPU forward within tolerance.
    ///
    /// `DFLASH_ANE_TEST_LAYERS=<n>` truncates the engine to the first `n`
    /// layers.  Used to sweep the ANE resource-exhaustion hypothesis — if
    /// failure is linear in layer count, some per-layer resource (DRAM,
    /// cache slots, IOSurfaces) is the ceiling.
    #[test]
    #[ignore] // requires ANE hardware + model weights
    fn test_dflash_ane_parity() {
        let mut cpu_engine = load_4b_engine();
        if let Ok(n) = std::env::var("DFLASH_ANE_TEST_LAYERS") {
            if let Ok(n) = n.parse::<usize>() {
                assert!(n > 0, "DFLASH_ANE_TEST_LAYERS must be > 0");
                let cap = n.min(cpu_engine.layers.len());
                cpu_engine.layers.truncate(cap);
                cpu_engine.config.layers = cap;
                eprintln!("DFlash ANE parity: truncated to {cap} layers via DFLASH_ANE_TEST_LAYERS");
            }
        }
        let cfg = &cpu_engine.config;
        let block = cfg.block_size;
        let h = cfg.hidden;
        let num_taps = cfg.num_taps;
        let ctx_len = 10;

        eprintln!(
            "DFlash ANE parity: hidden={h} layers={} block={block} taps={num_taps} ctx={ctx_len}",
            cfg.layers
        );

        // Deterministic test data
        let noise: Vec<f32> = (0..block * h)
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();
        let tap_data: Vec<Vec<f32>> = (0..num_taps)
            .map(|t| {
                (0..ctx_len * h)
                    .map(|i| ((i + t * 1000) as f32 * 0.001).cos() * 0.1)
                    .collect()
            })
            .collect();
        let tap_slices: Vec<&[f32]> = tap_data.iter().map(|t| t.as_slice()).collect();

        // CPU forward (reference)
        let mut cpu_cache = cpu_engine.make_cache();
        let cpu_out = cpu_engine.forward(&noise, &tap_slices, ctx_len, &mut cpu_cache);

        // ANE forward
        let t0 = std::time::Instant::now();
        let ane_executor = compile_dflash_ane(cpu_engine).expect("ANE compilation failed");
        let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Compilation: {compile_ms:.0}ms");

        let mut ane_cache = ane_executor.make_cache();
        let ane_out = ane_executor.forward(&noise, &tap_slices, ctx_len, &mut ane_cache);

        // Compare
        assert_eq!(cpu_out.len(), ane_out.len());
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        let mut n_nan = 0usize;
        let mut n_inf = 0usize;
        let mut first_nan_idx: Option<usize> = None;
        let mut worst_idx = 0usize;
        for (i, (a, b)) in cpu_out.iter().zip(ane_out.iter()).enumerate() {
            if b.is_nan() {
                n_nan += 1;
                if first_nan_idx.is_none() {
                    first_nan_idx = Some(i);
                }
                continue;
            }
            if b.is_infinite() {
                n_inf += 1;
                continue;
            }
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
                worst_idx = i;
            }
            sum_diff += diff;
        }
        let finite_count = cpu_out.len() - n_nan - n_inf;
        let mean_diff = sum_diff / finite_count.max(1) as f32;

        eprintln!(
            "  Parity: max_diff={max_diff:.6} @ idx={worst_idx}, mean_diff={mean_diff:.6}, n_nan={n_nan}, n_inf={n_inf}, first_nan@{first_nan_idx:?}, total={}",
            cpu_out.len()
        );
        eprintln!("  CPU  first 8: {:?}", &cpu_out[..8]);
        eprintln!("  ANE  first 8: {:?}", &ane_out[..8]);
        if let Some(idx) = first_nan_idx {
            let start = idx.saturating_sub(8);
            let end = (idx + 16).min(ane_out.len());
            eprintln!("  CPU  @{start}..{end}: {:?}", &cpu_out[start..end]);
            eprintln!("  ANE  @{start}..{end}: {:?}", &ane_out[start..end]);
        }
        // Count zeros by 4096-block.
        let hidden = 4096usize;
        let blocks = cpu_out.len() / hidden;
        for b in 0..blocks {
            let slice = &ane_out[b * hidden..(b + 1) * hidden];
            let zc = slice.iter().filter(|v| **v == 0.0 || **v == -0.0).count();
            let nc = slice.iter().filter(|v| v.is_nan()).count();
            let big = slice.iter().filter(|v| v.abs() > 100.0).count();
            if zc > 5 || nc > 0 {
                eprintln!("  block {b}: zeros={zc}, nans={nc}, |x|>100 count={big}");
            }
        }

        // ANE runs fp16 internally vs CPU fp32 — allow generous tolerance.
        assert!(
            max_diff < 0.15,
            "Max diff {max_diff} exceeds tolerance 0.15"
        );
    }

    /// Benchmark: ANE drafter latency at various context lengths.
    #[test]
    #[ignore]
    fn test_dflash_ane_latency() {
        let cpu_engine = load_4b_engine();
        let cfg = &cpu_engine.config;
        let block = cfg.block_size;
        let h = cfg.hidden;
        let num_taps = cfg.num_taps;
        let iters = 20;

        let ane_executor = compile_dflash_ane(cpu_engine).expect("ANE compilation failed");

        for ctx_len in [16, 64, 256] {
            let noise: Vec<f32> = (0..block * h).map(|i| (i as f32 * 0.001).sin()).collect();
            let tap_data: Vec<Vec<f32>> = (0..num_taps)
                .map(|t| {
                    (0..ctx_len * h)
                        .map(|i| ((i + t * 1000) as f32 * 0.001).cos())
                        .collect()
                })
                .collect();
            let tap_slices: Vec<&[f32]> = tap_data.iter().map(|t| t.as_slice()).collect();

            // Warmup
            let mut cache = ane_executor.make_cache();
            let _ = ane_executor.forward(&noise, &tap_slices, ctx_len, &mut cache);

            // Timed runs
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let mut cache = ane_executor.make_cache();
                let t0 = std::time::Instant::now();
                let _ = ane_executor.forward(&noise, &tap_slices, ctx_len, &mut cache);
                times.push(t0.elapsed());
            }
            times.sort();
            let median = times[iters / 2];
            let min = times[0];
            eprintln!(
                "ANE drafter ctx={ctx_len:>4}: median={:.2}ms  min={:.2}ms  ({block} tokens/round)",
                median.as_secs_f64() * 1000.0,
                min.as_secs_f64() * 1000.0,
            );
        }
    }

    /// Multi-round cache test: verify ANE and CPU produce identical results over
    /// multiple consecutive rounds with growing cache.
    #[test]
    #[ignore]
    fn test_dflash_ane_multi_round() {
        let cpu_engine = load_4b_engine();
        let cfg = &cpu_engine.config;
        let block = cfg.block_size;
        let h = cfg.hidden;
        let num_taps = cfg.num_taps;

        let ane_executor = compile_dflash_ane(cpu_engine).expect("ANE compilation failed");

        let mut cpu_cache = ane_executor.cpu_engine.make_cache();
        let mut ane_cache = ane_executor.make_cache();

        for round in 0..3 {
            let ctx_len = 10 + round * 5;
            let noise: Vec<f32> = (0..block * h)
                .map(|i| ((i + round * 10000) as f32 * 0.001).sin() * 0.1)
                .collect();
            let tap_data: Vec<Vec<f32>> = (0..num_taps)
                .map(|t| {
                    (0..ctx_len * h)
                        .map(|i| ((i + t * 1000 + round * 5000) as f32 * 0.001).cos() * 0.1)
                        .collect()
                })
                .collect();
            let tap_slices: Vec<&[f32]> = tap_data.iter().map(|t| t.as_slice()).collect();

            let cpu_out =
                ane_executor
                    .cpu_engine
                    .forward(&noise, &tap_slices, ctx_len, &mut cpu_cache);
            let ane_out = ane_executor.forward(&noise, &tap_slices, ctx_len, &mut ane_cache);

            let mut max_diff = 0.0f32;
            for (a, b) in cpu_out.iter().zip(ane_out.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }

            eprintln!("  Round {round} (ctx={ctx_len}): max_diff={max_diff:.6}");
            assert!(
                max_diff < 0.15,
                "Round {round} max diff {max_diff} exceeds tolerance"
            );
        }
    }

    /// Resolve a HuggingFace Hub cache dir to the directory containing config.json.
    /// Handles both `snapshots/<hash>/` layout and flat layout (config.json at root).
    fn resolve_drafter_dir(base: &std::path::Path) -> std::path::PathBuf {
        let snap_dir = base.join("snapshots");
        if snap_dir.is_dir() {
            std::fs::read_dir(&snap_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .next()
                .unwrap()
                .path()
        } else {
            base.to_path_buf()
        }
    }

    /// Run the full DFlash CPU-vs-ANE E2E benchmark for a given target + drafter pair.
    fn run_dflash_ane_e2e(target_path: &str, drafter_base: &str) {
        use crate::dflash::{load_dflash_drafter, GdnStateBackup};
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use mlx_rs::ops::indexing::IndexOp;
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let drafter_dir = resolve_drafter_dir(std::path::Path::new(drafter_base));

        // Load target model
        eprintln!("Loading target model from {target_path}...");
        let t0 = Instant::now();
        let mut target = load_qwen3_5_model(target_path).unwrap();
        eprintln!("Target loaded in {:.1}s", t0.elapsed().as_secs_f64());

        // Load drafter + extract CPU engine + compile ANE executor
        eprintln!("Loading DFlash drafter from {}...", drafter_dir.display());
        let drafter = load_dflash_drafter(&drafter_dir).unwrap();
        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();

        let cpu_engine = crate::dflash_cpu::extract_dflash_cpu_engine(&drafter);
        eprintln!("CPU engine extracted (hidden={})", cpu_engine.config.hidden);

        let t0 = Instant::now();
        let ane_executor = compile_dflash_ane(cpu_engine.clone()).expect("ANE compile failed");
        eprintln!("ANE executor compiled in {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);

        // Prompt tokens
        let prompt_tokens: Vec<i32> = vec![
            248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13,
            248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let eos_token: i32 = 248046;
        let prompt_len = prompt_tokens.len() as i32;
        let input_ids = mlx_rs::Array::from_slice(&prompt_tokens, &[1, prompt_len]);

        // Prefill with taps
        eprintln!("Prefilling...");
        let t0 = Instant::now();
        let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
        let (prefill_logits, taps) = target
            .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
            .unwrap();
        let mut eval_targets: Vec<&mlx_rs::Array> = vec![&prefill_logits];
        for t in &taps { eval_targets.push(t); }
        for lc in kv_cache.iter().flatten() {
            match lc {
                crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state { eval_targets.push(s); }
                    if let Some(ref c) = ac.conv_state { eval_targets.push(c); }
                }
            }
        }
        eval(eval_targets).unwrap();
        eprintln!("Prefill: {}ms", t0.elapsed().as_millis());

        let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
        let am_flat: Vec<u32> = prefill_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
        let first_token = *am_flat.last().unwrap() as i32;

        // Run E2E for a given forward function
        let mut run_e2e = |engine_name: &str,
                       forward_fn: &dyn Fn(&[f32], &[&[f32]], usize, &mut DFlashCpuCache) -> Vec<f32>,
                       hidden_dim: usize| -> (usize, u128, u128)
        {
            let mut kv = kv_cache.clone();
            let mut current_taps = taps.clone();
            let mut cpu_cache = cpu_engine.make_cache();
            let mut last_token = first_token;
            let mut start = prompt_len;
            let max_rounds = 10;
            let mut total_tokens = 0usize;
            let mut total_draft_ms = 0u128;
            let mut total_verify_ms = 0u128;

            for round in 0..max_rounds {
                let mut block_tokens = vec![mask_id; block_size as usize];
                block_tokens[0] = last_token;
                let block_ids = mlx_rs::Array::from_slice(&block_tokens, &[1, block_size]);
                let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

                // Draft
                let t0 = Instant::now();
                eval([&noise_embedding]).unwrap();
                let noise_f32: Vec<f32> = noise_embedding
                    .as_dtype(mlx_rs::Dtype::Float32).unwrap()
                    .reshape(&[-1]).unwrap()
                    .as_slice::<f32>().to_vec();
                eval(current_taps.iter().collect::<Vec<_>>()).unwrap();
                let taps_f32: Vec<Vec<f32>> = current_taps.iter().map(|t| {
                    t.as_dtype(mlx_rs::Dtype::Float32).unwrap()
                        .reshape(&[-1]).unwrap()
                        .as_slice::<f32>().to_vec()
                }).collect();
                let tap_slices: Vec<&[f32]> = taps_f32.iter().map(|t| t.as_slice()).collect();
                let ctx_len = current_taps[0].shape()[1] as usize;
                let out_f32 = forward_fn(&noise_f32, &tap_slices, ctx_len, &mut cpu_cache);
                cpu_cache.crop(start as usize);
                let draft_hidden = mlx_rs::Array::from_slice(
                    &out_f32, &[1, block_size, hidden_dim as i32]);
                let draft_ms = t0.elapsed().as_millis();
                total_draft_ms += draft_ms;

                let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
                let draft_logits = target.forward_all_logits_from_hidden(
                    &draft_hidden_sliced).unwrap();
                let draft_token_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
                eval([&draft_token_ids]).unwrap();
                let draft_u32: Vec<u32> = draft_token_ids.reshape(&[-1]).unwrap()
                    .as_slice::<u32>().to_vec();
                let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

                // Verify
                let mut verify_tokens = vec![last_token];
                verify_tokens.extend_from_slice(&draft_flat);
                let verify_len = verify_tokens.len() as i32;
                let verify_input = mlx_rs::Array::from_slice(
                    &verify_tokens, &[1, verify_len]);
                let gdn_backup = GdnStateBackup::save(&kv).unwrap();

                let t0 = Instant::now();
                let (verify_logits, verify_taps) = target
                    .forward_with_taps(&verify_input, None, &mut kv, &tap_layers).unwrap();
                eval([&verify_logits]).unwrap();
                let verify_ms = t0.elapsed().as_millis();
                total_verify_ms += verify_ms;

                let verify_flat: Vec<u32> = mlx_rs::argmax_axis!(verify_logits, -1).unwrap()
                    .reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
                let accepted = accept_prefix(&draft_u32, &verify_flat);
                let n_accepted = accepted.len();
                let tokens_this_round = n_accepted + 1;

                let keep = tokens_this_round as i32;
                let rollback = verify_len - keep;
                if rollback > 0 {
                    GdnStateBackup::restore_and_rollback(&gdn_backup, &mut kv, rollback);
                }
                current_taps = verify_taps.into_iter()
                    .map(|tap| tap.index((.., ..keep, ..))).collect();

                total_tokens += tokens_this_round;
                last_token = verify_flat[n_accepted] as i32;
                start += tokens_this_round as i32;

                eprintln!(
                    "  [{engine_name}] R{round}: draft={draft_ms}ms verify={verify_ms}ms accept={n_accepted}+1/{}",
                    block_size - 1
                );
                if last_token == eos_token { break; }
            }
            (total_tokens, total_draft_ms, total_verify_ms)
        };

        eprintln!("\n=== CPU BLAS E2E ===");
        let (cpu_tok, cpu_draft, cpu_verify) = run_e2e(
            "CPU",
            &|noise, taps, ctx, cache| cpu_engine.forward(noise, taps, ctx, cache),
            cpu_engine.config.hidden,
        );

        eprintln!("\n=== ANE+CPU E2E ===");
        let (ane_tok, ane_draft, ane_verify) = run_e2e(
            "ANE",
            &|noise, taps, ctx, cache| ane_executor.forward(noise, taps, ctx, cache),
            ane_executor.cpu_engine.config.hidden,
        );

        let cpu_tps = cpu_tok as f64 / ((cpu_draft + cpu_verify) as f64 / 1000.0);
        let ane_tps = ane_tok as f64 / ((ane_draft + ane_verify) as f64 / 1000.0);
        eprintln!("\n=== RESULTS ===");
        eprintln!("CPU BLAS: {cpu_tok} tok, draft={cpu_draft}ms, verify={cpu_verify}ms, {cpu_tps:.1} tok/s");
        eprintln!("ANE+CPU:  {ane_tok} tok, draft={ane_draft}ms, verify={ane_verify}ms, {ane_tps:.1} tok/s");
        eprintln!("Draft speedup: {:.2}x, Overall speedup: {:.2}x",
            cpu_draft as f64 / ane_draft.max(1) as f64,
            ane_tps / cpu_tps,
        );
    }

    /// Acceptance-rate investigation across diverse prompts.
    /// Tests ANE-only path with detailed per-round, per-position diagnostics.
    #[test]
    #[ignore]
    fn test_dflash_ane_acceptance_sweep() {
        use crate::dflash::{load_dflash_drafter, GdnStateBackup};
        use crate::diffusion::accept_prefix;
        use crate::qwen3_next::load_qwen3_5_model;
        use mlx_rs::ops::indexing::IndexOp;
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let target_path =
            "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-bf16";
        let drafter_dir = resolve_drafter_dir(std::path::Path::new(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        ));

        let mut target = load_qwen3_5_model(target_path).unwrap();
        let drafter = load_dflash_drafter(&drafter_dir).unwrap();
        let tap_layers = drafter.config.target_layer_ids().to_vec();
        let block_size = drafter.config.block_size;
        let mask_id = drafter.config.mask_token_id();
        let eos_token: i32 = 248046;

        let cpu_engine = crate::dflash_cpu::extract_dflash_cpu_engine(&drafter);
        let ane_executor = compile_dflash_ane(cpu_engine.clone()).expect("ANE compile failed");

        let prompts: Vec<(&str, Vec<i32>)> = vec![
            // Original: "Write a short paragraph about the history of computers" + think tags
            ("history+think", vec![
                248045, 846, 198, 7734, 264, 2716, 13901, 883, 279, 3712, 314, 17943, 13,
                248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
            ]),
            // Counting — very predictable
            ("counting", vec![248045, 846, 198, 2427, 494, 220, 16, 310, 220, 16, 15, 15, 248046, 198, 248045, 74455, 198]),
            // Factual — "What is the capital of France?"
            ("factual", vec![248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198]),
            // Code — "Write a Python fibonacci function"
            ("code", vec![248045, 846, 198, 7734, 264, 12654, 73111, 709, 248046, 198, 248045, 74455, 198]),
            // Repeat — "Repeat the word hello 50 times"
            ("repeat", vec![248045, 846, 198, 37436, 279, 3299, 23066, 220, 20, 15, 2942, 248046, 198, 248045, 74455, 198]),
            // Reasoning — harder
            ("reasoning", vec![248045, 846, 198, 814, 20139, 29144, 1157, 512, 954, 310, 264, 220, 20, 1007, 2235, 1608, 3487, 22672, 536, 248046, 198, 248045, 74455, 198]),
            // List — "List the days of the week"
            ("list", vec![248045, 846, 198, 826, 279, 2756, 314, 279, 1936, 248046, 198, 248045, 74455, 198]),
        ];

        eprintln!("\n{}", "=".repeat(80));
        eprintln!("  DFlash ANE Acceptance Sweep — 4B target + 4B drafter");
        eprintln!("  block_size={block_size}, taps={tap_layers:?}");
        eprintln!("{}\n", "=".repeat(80));

        let mut summary: Vec<(&str, usize, usize, usize, f64, f64)> = Vec::new();

        for (name, prompt_tokens) in &prompts {
            let prompt_len = prompt_tokens.len() as i32;
            let input_ids = mlx_rs::Array::from_slice(prompt_tokens, &[1, prompt_len]);

            // Prefill
            let mut kv_cache: Vec<Option<crate::qwen3_next::LayerCache>> = Vec::new();
            let (prefill_logits, taps) = target
                .forward_with_taps(&input_ids, None, &mut kv_cache, &tap_layers)
                .unwrap();
            let mut eval_targets: Vec<&mlx_rs::Array> = vec![&prefill_logits];
            for t in &taps { eval_targets.push(t); }
            for lc in kv_cache.iter().flatten() {
                match lc {
                    crate::qwen3_next::LayerCache::KV(kv) => eval_targets.extend(kv.eval_targets()),
                    crate::qwen3_next::LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state { eval_targets.push(s); }
                        if let Some(ref c) = ac.conv_state { eval_targets.push(c); }
                    }
                }
            }
            eval(eval_targets).unwrap();

            let prefill_am = mlx_rs::argmax_axis!(prefill_logits, -1).unwrap();
            let am_flat: Vec<u32> = prefill_am.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
            let first_token = *am_flat.last().unwrap() as i32;

            // Decode loop — ANE only, 15 rounds
            let mut current_taps = taps.clone();
            let mut cpu_cache = ane_executor.cpu_engine.make_cache();
            let mut last_token = first_token;
            let mut start = prompt_len;
            let max_rounds = 15;
            let mut total_tokens = 0usize;
            let mut total_accepted = 0usize;
            let mut total_rounds = 0usize;
            let mut total_draft_ms = 0u128;
            let mut total_verify_ms = 0u128;

            eprintln!("--- {name} (prompt_len={prompt_len}) ---");

            for round in 0..max_rounds {
                let mut block_tokens = vec![mask_id; block_size as usize];
                block_tokens[0] = last_token;
                let block_ids = mlx_rs::Array::from_slice(&block_tokens, &[1, block_size]);
                let noise_embedding = target.embed_token_ids(&block_ids).unwrap();

                let t0 = Instant::now();
                eval([&noise_embedding]).unwrap();
                let noise_f32: Vec<f32> = noise_embedding
                    .as_dtype(mlx_rs::Dtype::Float32).unwrap()
                    .reshape(&[-1]).unwrap()
                    .as_slice::<f32>().to_vec();
                eval(current_taps.iter().collect::<Vec<_>>()).unwrap();
                let taps_f32: Vec<Vec<f32>> = current_taps.iter().map(|t| {
                    t.as_dtype(mlx_rs::Dtype::Float32).unwrap()
                        .reshape(&[-1]).unwrap()
                        .as_slice::<f32>().to_vec()
                }).collect();
                let tap_slices: Vec<&[f32]> = taps_f32.iter().map(|t| t.as_slice()).collect();
                let ctx_len = current_taps[0].shape()[1] as usize;
                let out_f32 = ane_executor.forward(&noise_f32, &tap_slices, ctx_len, &mut cpu_cache);
                cpu_cache.crop(start as usize);
                let h = ane_executor.cpu_engine.config.hidden;
                let draft_hidden = mlx_rs::Array::from_slice(
                    &out_f32, &[1, block_size, h as i32]);
                let draft_ms = t0.elapsed().as_millis();
                total_draft_ms += draft_ms;

                let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
                let draft_logits = target.forward_all_logits_from_hidden(
                    &draft_hidden_sliced).unwrap();
                let draft_token_ids = mlx_rs::argmax_axis!(draft_logits, -1).unwrap();
                eval([&draft_token_ids]).unwrap();
                let draft_u32: Vec<u32> = draft_token_ids.reshape(&[-1]).unwrap()
                    .as_slice::<u32>().to_vec();
                let draft_flat: Vec<i32> = draft_u32.iter().map(|&x| x as i32).collect();

                // Verify
                let mut verify_tokens = vec![last_token];
                verify_tokens.extend_from_slice(&draft_flat);
                let verify_len = verify_tokens.len() as i32;
                let verify_input = mlx_rs::Array::from_slice(
                    &verify_tokens, &[1, verify_len]);
                let gdn_backup = GdnStateBackup::save(&kv_cache).unwrap();

                let t0 = Instant::now();
                let (verify_logits, verify_taps) = target
                    .forward_with_taps(&verify_input, None, &mut kv_cache, &tap_layers).unwrap();
                eval([&verify_logits]).unwrap();
                let verify_ms = t0.elapsed().as_millis();
                total_verify_ms += verify_ms;

                let verify_flat: Vec<u32> = mlx_rs::argmax_axis!(verify_logits, -1).unwrap()
                    .reshape(&[-1]).unwrap().as_slice::<u32>().to_vec();
                let accepted = accept_prefix(&draft_u32, &verify_flat);
                let n_accepted = accepted.len();
                let tokens_this_round = n_accepted + 1;

                let keep = tokens_this_round as i32;
                let rollback = verify_len - keep;
                if rollback > 0 {
                    GdnStateBackup::restore_and_rollback(&gdn_backup, &mut kv_cache, rollback);
                }
                current_taps = verify_taps.into_iter()
                    .map(|tap| tap.index((.., ..keep, ..))).collect();

                total_accepted += n_accepted;
                total_tokens += tokens_this_round;
                total_rounds += 1;
                last_token = verify_flat[n_accepted.min(verify_flat.len() - 1)] as i32;
                start += tokens_this_round as i32;

                // Show per-position match for first few rounds
                if round < 5 {
                    let mut match_str = String::new();
                    for i in 0..(block_size as usize - 1).min(draft_u32.len()).min(verify_flat.len()) {
                        if draft_u32[i] == verify_flat[i] {
                            match_str.push('=');
                        } else {
                            match_str.push('X');
                        }
                    }
                    eprintln!("  R{round:02}: accept={n_accepted:2}/{} [{match_str}] draft={draft_ms}ms verify={verify_ms}ms",
                        block_size - 1);
                }

                if last_token == eos_token { break; }
            }

            let accept_rate = total_accepted as f64 / (total_rounds as f64 * (block_size - 1) as f64) * 100.0;
            let tok_per_round = total_tokens as f64 / total_rounds as f64;
            let total_ms = total_draft_ms + total_verify_ms;
            let tps = total_tokens as f64 / (total_ms as f64 / 1000.0);

            eprintln!("  => {total_tokens} tok / {total_rounds} rounds, accept={accept_rate:.1}%, tok/round={tok_per_round:.1}, {tps:.1} tok/s (draft={total_draft_ms}ms verify={total_verify_ms}ms)\n");

            summary.push((name, total_tokens, total_rounds, total_accepted, accept_rate, tps));
        }

        eprintln!("\n{}", "=".repeat(80));
        eprintln!("  SUMMARY");
        eprintln!("{}", "=".repeat(80));
        eprintln!("{:<16} {:>5} {:>6} {:>8} {:>8} {:>8}", "Prompt", "Tok", "Rounds", "Accept%", "Tok/Rnd", "Tok/s");
        eprintln!("{:-<16} {:->5} {:->6} {:->8} {:->8} {:->8}", "", "", "", "", "", "");
        for (name, tok, rounds, _acc, rate, tps) in &summary {
            let tpr = *tok as f64 / *rounds as f64;
            eprintln!("{:<16} {:>5} {:>6} {:>7.1}% {:>8.1} {:>7.1}", name, tok, rounds, rate, tpr, tps);
        }
        eprintln!();
    }

    // -- 4B drafter + 4B target (bf16) -----------------------------------------
    #[test]
    #[ignore]
    fn test_dflash_ane_e2e_4b_bf16() {
        run_dflash_ane_e2e(
            "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-bf16",
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        );
    }

    // -- 4B drafter + 4B target (4-bit) ----------------------------------------
    #[test]
    #[ignore]
    fn test_dflash_ane_e2e_4b_4bit() {
        run_dflash_ane_e2e(
            "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-4B-MLX-4bit",
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash",
        );
    }

    // -- 9B drafter + 8B target (4-bit) ----------------------------------------
    // NOTE: skipped — Qwen3-8B uses flat config (Qwen3ForCausalLM), but
    // load_qwen3_5_model requires nested text_config (Qwen3_5). Need a
    // Qwen3.5-9B target or a Qwen3 loader to test this drafter.

    // -- 27B drafter + 27B target (4-bit) --------------------------------------
    #[test]
    #[ignore]
    fn test_dflash_ane_e2e_27b() {
        run_dflash_ane_e2e(
            "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit",
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash",
        );
    }

    // -- 35B-A3B drafter + 35B-A3B target (3-bit) ------------------------------
    #[test]
    #[ignore]
    fn test_dflash_ane_e2e_35b_a3b() {
        run_dflash_ane_e2e(
            "/Users/peppi/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit",
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash",
        );
    }
}
