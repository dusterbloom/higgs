//! Chunkwise-parallel gated delta rule (GDN prefill).
//!
//! The serial Metal kernel (`qwen3_next::GATED_DELTA_KERNEL_SOURCE`) walks the
//! sequence one token at a time inside each thread — ideal for decode (T=1),
//! but it caps prefill at the scan rate (~330 tok/s on Qwen3.6-35B) no matter
//! how the outer loop batches. This module computes the *same* recurrence in
//! chunk-parallel form (the UT/WY transform used by flash-linear-attention's
//! `chunk_gated_delta_rule`): all intra-chunk work is dense batched matmuls,
//! and only the running state crosses chunk boundaries — serial depth drops
//! from T to ceil(T/C).
//!
//! Recurrence (state `S: [Dv, Dk]`, per head):
//! ```text
//! S_t = g_t·S_{t-1} + β_t·(v_t − g_t·S_{t-1}·k_t)·k_tᵀ
//! y_t = S_t·q_t
//! ```
//!
//! Chunk form, per chunk of C tokens (writing L_i = Σ_{j≤i} log g_j,
//! γ_i = exp(L_i), and D[i,j] = exp(L_i − L_j)):
//! ```text
//! A[i,j] = β_i·(k_iᵀk_j)·D[i,j]            (strictly lower triangular)
//! U      = (I + A)⁻¹ · (β ⊙ (V − γ ⊙ K·S₀ᵀ))
//! Y      = γ ⊙ (Q·S₀ᵀ) + ((Q·Kᵀ ⊙ D) masked j≤i) · U
//! S_C    = γ_C·S₀ + (U ⊙ exp(L_C − L))ᵀ · K
//! ```
//! This is exact algebra, not an approximation: substituting U back yields the
//! token recurrence verbatim (tests compare against `gated_delta_step_ref`).
//!
//! All chunk math runs in f32 (the cache state is already f32 for the same
//! stability reason); inputs are cast in, outputs cast back.

use mlx_rs::error::Exception;
use mlx_rs::ops::{self, indexing::IndexOp};
use mlx_rs::Array;

/// Inner chunk length. 64 keeps the [C×C] solve tiny while cutting serial
/// depth by 64×; override with `HIGGS_GDN_CHUNK` for experiments.
pub(crate) fn chunk_len() -> i32 {
    std::env::var("HIGGS_GDN_CHUNK")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|&c| c > 0)
        .unwrap_or(64)
}

/// Floor for gate values before `log`: g is exp(−exp(A_log)·softplus(·)) > 0
/// mathematically, but can underflow to 0 in f32 for extreme gates. The
/// serial kernel multiplies by g directly (state → 0); clamping keeps the
/// log-space path finite and consistent with that behaviour.
const GATE_LOG_FLOOR: f32 = 1e-30;

/// Triangular masks + identity for a chunk of length `c`, shaped `[1,1,c,c]`
/// (f32, broadcast over batch and heads).
///
/// Returns `(strict_lower, incl_lower, identity)` where `strict_lower[i][j] =
/// 1 iff j < i`, `incl_lower[i][j] = 1 iff j ≤ i`.
fn chunk_masks(c: i32) -> Result<(Array, Array, Array), Exception> {
    let rows = ops::arange::<_, f32>(None, c, None)?.reshape(&[c, 1])?;
    let cols = ops::arange::<_, f32>(None, c, None)?.reshape(&[1, c])?;
    let strict = rows.gt(&cols)?.as_dtype(mlx_rs::Dtype::Float32)?;
    let incl = rows.ge(&cols)?.as_dtype(mlx_rs::Dtype::Float32)?;
    let eye = rows.eq(&cols)?.as_dtype(mlx_rs::Dtype::Float32)?;
    Ok((
        strict.reshape(&[1, 1, c, c])?,
        incl.reshape(&[1, 1, c, c])?,
        eye.reshape(&[1, 1, c, c])?,
    ))
}

/// Invert the unit-lower-triangular matrix `m = I + A` (A strictly lower)
/// by recursive block forward substitution:
/// `[[M11, 0], [M21, M22]]⁻¹ = [[X11, 0], [−X22·M21·X11, X22]]`.
///
/// Exact and numerically stable: only blocks of the *true inverse* appear.
/// (A Newton–Schulz iteration is exact in algebra via nilpotency but forms
/// high matrix powers — `A^64` overflows f32 long before the true inverse
/// does.) The delta-rule solve is well-conditioned because the model's
/// qk-norm keeps `|k_iᵀk_j| ≤ 1` and `β < 1`.
fn invert_unit_lower(m: &Array, c: i32) -> Result<Array, Exception> {
    if c <= 1 {
        // Unit diagonal: the 1×1 block of M already equals its inverse (=1).
        return Ok(m.clone());
    }
    let h = c / 2;
    let m11 = m.index((.., .., 0..h, 0..h));
    let m21 = m.index((.., .., h..c, 0..h));
    let m22 = m.index((.., .., h..c, h..c));
    let x11 = invert_unit_lower(&m11, h)?;
    let x22 = invert_unit_lower(&m22, c - h)?;
    let x21 = ops::matmul(&x22, &ops::matmul(&m21, &x11)?)?.negative()?;
    // The upper-right block of M is exactly zero — reuse it for the output.
    let zero = m.index((.., .., 0..h, h..c));
    let top = ops::concatenate_axis(&[x11, zero], -1)?;
    let bottom = ops::concatenate_axis(&[x21, x22], -1)?;
    ops::concatenate_axis(&[top, bottom], -2)
}

/// One chunk of the chunkwise gated delta rule.
///
/// Shapes (head-major, f32): `q,k: [B,Hv,C,Dk]`, `v: [B,Hv,C,Dv]`,
/// `log_g, beta: [B,Hv,C]`, `state: [B,Hv,Dv,Dk]`.
/// Returns `(y: [B,Hv,C,Dv], new_state)`.
fn process_chunk(
    q: &Array,
    k: &Array,
    v: &Array,
    log_g: &Array,
    beta: &Array,
    state: &Array,
    masks: &(Array, Array, Array),
) -> Result<(Array, Array), Exception> {
    let (strict, incl, eye) = masks;

    // Cumulative log-decay within the chunk: L_i = Σ_{j≤i} log g_j.
    let l = log_g.cumsum(-1, None, None)?; // [B,Hv,C]
    let gamma = l.exp()?.expand_dims(-1)?; // [B,Hv,C,1]
    let beta_col = beta.expand_dims(-1)?; // [B,Hv,C,1]

    // Decay ratios D[i,j] = exp(L_i − L_j), zeroed outside the causal mask
    // BEFORE exp would overflow: for j > i the exponent is ≥ 0, so mask the
    // exponent down to a huge negative instead (exp → 0, no NaN).
    let l_i = l.expand_dims(-1)?; // [B,Hv,C,1]
    let l_j = l.expand_dims(-2)?; // [B,Hv,1,C]
    let diff = l_i.subtract(&l_j)?; // [B,Hv,C,C]
    let neg_big = Array::from_f32(-1e30);
    let mask_exp = |mask: &Array| -> Result<Array, Exception> {
        let off = ops::ones_like(mask)?.subtract(mask)?.multiply(&neg_big)?;
        diff.multiply(mask)?.add(&off)?.exp()
    };
    let d_strict = mask_exp(strict)?; // j <  i
    let d_incl = mask_exp(incl)?; // j <= i

    let k_t = k.transpose_axes(&[0, 1, 3, 2])?; // [B,Hv,Dk,C]
    let state_t = state.transpose_axes(&[0, 1, 3, 2])?; // [B,Hv,Dk,Dv]

    // A[i,j] = β_i (k_iᵀ k_j) D[i,j], strictly lower.
    let kkt = ops::matmul(k, &k_t)?; // [B,Hv,C,C]
    let a = kkt.multiply(&d_strict)?.multiply(&beta_col)?;

    // U = (I + A)⁻¹ (β ⊙ (V − γ ⊙ K S₀ᵀ))
    let m = eye.add(&a)?;
    let t_mat = invert_unit_lower(&m, *a.shape().last().unwrap_or(&1))?;
    let k_s0 = ops::matmul(k, &state_t)?; // [B,Hv,C,Dv]
    let v_eff = v.subtract(&gamma.multiply(&k_s0)?)?.multiply(&beta_col)?;
    let u = ops::matmul(&t_mat, &v_eff)?; // [B,Hv,C,Dv]

    // Y = γ ⊙ (Q S₀ᵀ) + ((Q Kᵀ) ⊙ D_incl) U
    let q_s0 = ops::matmul(q, &state_t)?; // [B,Hv,C,Dv]
    let qkt = ops::matmul(q, &k_t)?; // [B,Hv,C,C]
    let y = gamma
        .multiply(&q_s0)?
        .add(&ops::matmul(&qkt.multiply(&d_incl)?, &u)?)?;

    // S_C = γ_C S₀ + (U ⊙ exp(L_C − L))ᵀ K
    let l_last = l.index((.., .., -1)).expand_dims(-1)?; // [B,Hv,1]
    let scale = l_last.subtract(&l)?.exp()?.expand_dims(-1)?; // [B,Hv,C,1]
    let w_t = u.multiply(&scale)?.transpose_axes(&[0, 1, 3, 2])?; // [B,Hv,Dv,C]
    let gamma_last = l_last.exp()?.expand_dims(-1)?; // [B,Hv,1,1]
    let new_state = state.multiply(&gamma_last)?.add(&ops::matmul(&w_t, k)?)?;

    Ok((y, new_state))
}

/// Chunkwise-parallel gated delta rule over a full prefill segment.
///
/// Inputs (token-major, any float dtype): `q,k: [B,T,Hv,Dk]` (already
/// GQA-repeated and **per-head normalised** — the model's qk-norm; the UT
/// solve's conditioning relies on `|k_iᵀk_j| ≲ 1`), `v: [B,T,Hv,Dv]`,
/// `g,beta: [B,T,Hv]` (decay and write gates, precomputed),
/// `state: [B,Hv,Dv,Dk]` f32.
///
/// Returns `(y: [B,T,Hv,Dv]` in the input dtype, `new_state` f32) — exactly
/// what `gated_delta_kernel_ffi` returns for the same inputs.
pub(crate) fn gated_delta_chunkwise(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    chunk: i32,
) -> Result<(Array, Array), Exception> {
    let in_dtype = v.dtype();
    let t = *q
        .shape()
        .get(1)
        .ok_or_else(|| Exception::custom("gdn_chunkwise: q must be [B,T,Hv,Dk]"))?;

    let f32_head_major = |x: &Array| -> Result<Array, Exception> {
        x.as_dtype(mlx_rs::Dtype::Float32)?
            .transpose_axes(&[0, 2, 1, 3])
    };
    let q = f32_head_major(q)?; // [B,Hv,T,Dk]
    let k = f32_head_major(k)?;
    let v = f32_head_major(v)?;
    // Gates: [B,T,Hv] → [B,Hv,T], log-space with underflow floor.
    let log_g = ops::maximum(
        &g.as_dtype(mlx_rs::Dtype::Float32)?,
        &Array::from_f32(GATE_LOG_FLOOR),
    )?
    .log()?
    .transpose_axes(&[0, 2, 1])?;
    let beta = beta
        .as_dtype(mlx_rs::Dtype::Float32)?
        .transpose_axes(&[0, 2, 1])?;

    let mut state = state.as_dtype(mlx_rs::Dtype::Float32)?;
    let mut outputs: Vec<Array> = Vec::new();
    // Masks depend only on the chunk length; at most two distinct lengths
    // occur per call (the full chunk and one tail), so cache by size.
    let mut mask_cache: Option<(i32, (Array, Array, Array))> = None;

    let mut t0 = 0i32;
    while t0 < t {
        let c_eff = chunk.min(t - t0);
        if mask_cache.as_ref().map(|(s, _)| *s) != Some(c_eff) {
            mask_cache = Some((c_eff, chunk_masks(c_eff)?));
        }
        let masks = match mask_cache.as_ref() {
            Some((_, m)) => m,
            None => return Err(Exception::custom("gdn_chunkwise: mask cache empty")),
        };

        let qc = q.index((.., .., t0..t0 + c_eff, ..));
        let kc = k.index((.., .., t0..t0 + c_eff, ..));
        let vc = v.index((.., .., t0..t0 + c_eff, ..));
        let gc = log_g.index((.., .., t0..t0 + c_eff));
        let bc = beta.index((.., .., t0..t0 + c_eff));

        let (y, new_state) = process_chunk(&qc, &kc, &vc, &gc, &bc, &state, masks)?;
        outputs.push(y);
        state = new_state;
        t0 += c_eff;
    }

    let y = ops::concatenate_axis(&outputs, 2)?
        .transpose_axes(&[0, 2, 1, 3])? // back to [B,T,Hv,Dv]
        .as_dtype(in_dtype)?;
    Ok((y, state))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Block forward substitution must satisfy (I+A)·X ≈ I directly, so a
    /// failure here pinpoints the inverter rather than the full chunk math.
    /// A is shaped like the real delta-rule solve: normalised keys (so
    /// `|k_iᵀk_j| ≤ 1`) scaled by β ∈ (0,1) — the conditioning regime the
    /// model's qk-norm guarantees.
    #[test]
    fn test_invert_unit_lower_exact() {
        let c = 64;
        let dk = 32;
        let (strict, _incl, eye) = chunk_masks(c).unwrap();
        let k = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[2, 3, c, dk], None).unwrap();
        let norm = k
            .multiply(&k)
            .unwrap()
            .sum_axes(&[-1], true)
            .unwrap()
            .sqrt()
            .unwrap()
            .add(&Array::from_f32(1e-6))
            .unwrap();
        let k = k.divide(&norm).unwrap();
        let beta =
            mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[2, 3, c, 1], None).unwrap();
        let kkt = ops::matmul(&k, &k.transpose_axes(&[0, 1, 3, 2]).unwrap()).unwrap();
        let a = kkt.multiply(&beta).unwrap().multiply(&strict).unwrap();
        let m = eye.add(&a).unwrap();
        let x = invert_unit_lower(&m, c).unwrap();
        let residual = ops::matmul(&m, &x)
            .unwrap()
            .subtract(&eye)
            .unwrap()
            .abs()
            .unwrap();
        let max: f32 = residual.max(None).unwrap().item();
        assert!(max < 1e-3, "inverse residual too large: {max}");
    }
}
