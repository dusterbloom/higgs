//! Multi-row Gated `DeltaNet` kernels for fast rows (Clover block preview and
//! the self-speculative verify), replacing ~60-90 small MLX dispatches per
//! layer with one or two.
//!
//! After Kapitsa's `clover_gdn_prep` / `clover_gdn_mem` (`clover.metal`) and
//! `gdn_prep_verify8` / `gdn_step_verify8s` (`verify8_fuse.metal`,
//! `verify8.metal`), Apache-2.0, nace-ai/kapitsa:
//! - `prep`: depthwise convolution over `[history | rows]` (plus Clover's
//!   right taps inside the block for the preview), `SiLU`, the q/k RMS norms
//!   with their fixed weights, and the decay `g` (preview); one threadgroup
//!   of `D` threads per 128-channel head chunk, one thread per channel. The
//!   verify variant also writes the next convolution window.
//! - `mem` (preview): per value head, a softmax over the block's rows and one
//!   history bucket (the decayed readout of the committed state); grid
//!   `(Hv, D / 32)` threadgroups of 128 threads, a simdgroup per 8 state rows.
//! - `fold` (partial accept): advances the committed state by the first `n`
//!   rows of a verify tape (`k`, `a`, `delta`) with the recurrence's own two
//!   statements, and the convolution window by the first `n` inputs; one
//!   thread per 4 state elements, no reductions.
//!
//! The verify's recurrence itself stays on
//! [`crate::qwen3_next::gated_delta_kernel_ffi_with_tape`]. Everything here
//! runs in fp32 from 16-bit inputs; it is within bf16 rounding of the MLX
//! chain, not bit-identical, so it runs only inside
//! [`crate::qwen3_next::with_fast_rows`].

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, error::Exception};

use crate::metal_kernel::{FastKernel, TemplateArg};

/// Head dimension the kernels are laid out for (`Dk == Dv == D`).
const D: i32 = 128;

/// Convolution + `SiLU` + q/k norms of `T` rows against `K - 1` history rows.
/// `{GATES}`, `{RIGHT_LOAD}`, `{RIGHT_TAPS}`, `{WINDOW}` select the variant.
const PREP: &str = r"
  threadgroup float scratch[D / 32];
  const uint tg = threadgroup_position_in_grid.x;
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
{GATES}
  const int c = int(tg) * D + int(tid);
  constexpr int QK = HK * D;
  const bool is_q = c < QK, is_k = c >= QK && c < 2 * QK;
  float wt[K];
  for (int j = 0; j < K; ++j) wt[j] = float(w[j * C + c]);
{RIGHT_LOAD}
  const float nq = float(wq[tid]), nk = float(wk[tid]);
  for (int t = 0; t < T; ++t) {
    // concat = [hist[0 .. K - 1), rows[0 .. T)]: output t reads concat[t .. t + K)
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
      const int i = t + j;
      const float x = i < K - 1 ? float(hist[i * C + c]) : float(rows[(i - (K - 1)) * C + c]);
      acc += wt[j] * x;
    }
{RIGHT_TAPS}
    // Rounded where the MLX chain rounds: the activation to the rows' dtype,
    // then `rms_norm`'s `w * InT(x * inv)`.
    const float o = float(InT(acc / (1.0f + exp(-acc))));
    if (is_q || is_k) {  // uniform: a threadgroup is one head's channels
      const float part = simd_sum(o * o);
      if (lane == 0) scratch[sg] = part;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float ss = 0.0f;
      for (int i = 0; i < D / 32; ++i) ss += scratch[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float unit = float(InT(o * metal::precise::rsqrt(ss / float(D) + 1e-6f)));
      if (is_q) {
        q[t * QK + c] = float(InT(unit * nq));
{QUNIT}
      } else {
        k[t * QK + (c - QK)] = float(InT(unit * nk));
      }
    } else {
      v[t * (C - 2 * QK) + (c - 2 * QK)] = o;
    }
  }
{WINDOW}
";

/// Preview: the last threadgroup writes `g = exp(-exp(A_log) softplus(a + dt_bias))`.
const GATES: &str = r"
  if (tg == uint(C / D)) {
    for (int h = int(tid); h < HV; h += D) {
      const float A = exp(float(a_log[h]));
      const float dtb = float(dt_bias[h]);
      for (int t = 0; t < T; ++t) {
        const float x = float(a[t * HV + h]) + dtb;
        const float sp = fmax(x, 0.0f) + log1p(exp(-fabs(x)));
        g[t * HV + h] = exp(-A * sp);
      }
    }
    return;
  }
";
/// Preview: the logits use `q` normalised like `k` (its own rounding).
const QUNIT: &str = r"
        qu[t * QK + c] = float(InT(unit * nk));
";
const RIGHT_LOAD: &str = r"
  float rt[K - 1];
  for (int d = 0; d < K - 1; ++d) rt[d] = float(r[c * (K - 1) + d]);
";
const RIGHT_TAPS: &str = r"
    for (int d = 1; d < K; ++d)
      if (t + d < T) acc += rt[d - 1] * float(rows[(t + d) * C + c]);
";
/// Verify: the next window is `concat[T .. T + K - 1)`.
const WINDOW: &str = r"
  for (int j = 0; j < K - 1; ++j) {
    const int i = T + j;
    win[j * C + c] = i < K - 1 ? hist[i * C + c] : rows[(i - (K - 1)) * C + c];
  }
";

/// Clover's block memory for value head `h = tg.x`, state rows `32 tg.y ..`:
/// `w = softmax([scale (qu_l . k_s) over s, bias + c log(T / 32)])`,
/// `out_l = sum_s w[l][s] v_s + w[l][T] g_l (S q_l)`; `qu` is the unit q,
/// `q` the unit q over `sqrt(D)`.
const MEM: &str = r"
  threadgroup float sw[32][33];
  const uint h = threadgroup_position_in_grid.x;
  const uint tid = thread_index_in_threadgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint hk = h / uint(HV / HK);
  const float ls = float(scale[h]);
  for (uint pr = tid; pr < uint(T * T); pr += 128) {
    const uint l = pr / uint(T), s = pr % uint(T);
    const device float4* q4 = reinterpret_cast<const device float4*>(qu + (l * uint(HK) + hk) * D);
    const device float4* k4 = reinterpret_cast<const device float4*>(k + (s * uint(HK) + hk) * D);
    float d = 0.0f;
    for (int j = 0; j < D / 4; ++j) d += dot(q4[j], k4[j]);
    sw[l][s] = d * ls;
  }
  if (tid < uint(T)) sw[tid][T] = float(hbias[h]) + float(hc[h]) * log(float(T) / 32.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < uint(T)) {
    float m = -INFINITY;
    for (int s = 0; s <= T; ++s) m = fmax(m, sw[tid][s]);
    float z = 0.0f;
    for (int s = 0; s <= T; ++s) {
      const float e = exp(sw[tid][s] - m);
      sw[tid][s] = e;
      z += e;
    }
    const float inv = 1.0f / z;
    for (int s = 0; s <= T; ++s) sw[tid][s] *= inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint dv0 = (threadgroup_position_in_grid.y * 4 + sg) * 8;
  const float gl = lane < uint(T) ? g[lane * uint(HV) + h] : 0.0f;
  for (int rr = 0; rr < 8; ++rr) {
    const uint dv = dv0 + uint(rr);
    const float4 s4 = *reinterpret_cast<const device float4*>(state + (h * D + dv) * D + lane * 4);
    float hist = 0.0f;
    for (int l = 0; l < T; ++l) {
      const device float4* ql = reinterpret_cast<const device float4*>(q + (uint(l) * uint(HK) + hk) * D + lane * 4);
      const float x = simd_sum(dot(s4, *ql));
      if (lane == uint(l)) hist = x;
    }
    if (lane < uint(T)) {
      float o = sw[lane][T] * hist * gl;
      for (int s = 0; s < T; ++s) o += sw[lane][s] * v[(uint(s) * uint(HV) + h) * D + dv];
      out[(lane * uint(HV) + h) * D + dv] = o;
    }
  }
";

/// `state_out = fold of tape rows 0 .. n` with the recurrence's statements
/// (`HIGGS_GDN_*`, the forward kernels' own), `win = concat[n .. n + K - 1)`.
const FOLD: &str = r"
  const uint i = thread_position_in_grid.x;
  constexpr uint NS = uint(HV) * uint(D) * uint(D / 4);
  const int N = n;
  if (i < NS) {
    const uint row = i / uint(D / 4);
    const uint j = i % uint(D / 4);
    const uint h = row / uint(D);
    const uint dv = row % uint(D);
    const uint hk = h / uint(HV / HK);
    float4 st = *reinterpret_cast<const device float4*>(state_in + row * D + j * 4);
    const float a_log_val = float(a_log[h]);
    const float dt_bias_val = float(dt_bias[h]);
    for (int t = 0; t < N; ++t) {
      HIGGS_GDN_GATE(g_val, a[uint(t) * uint(HV) + h], dt_bias_val, a_log_val);
      const float4 k4 = *reinterpret_cast<const device float4*>(k + (uint(t) * uint(HK) + hk) * D + j * 4);
      const float d = delta[(uint(t) * uint(HV) + h) * D + dv];
      HIGGS_GDN_DECAY(st, g_val);
      HIGGS_GDN_UPDATE(st, k4, d);
    }
    *reinterpret_cast<device float4*>(state_out + row * D + j * 4) = st;
  } else {
    const uint e = i - NS;
    const uint r = e / uint(C), c = e % uint(C);
    const int src = N + int(r);
    win[r * C + c] = src < K - 1 ? hist[uint(src) * C + c] : qkv[uint(src - (K - 1)) * C + c];
  }
";

fn source(parts: &[(&str, &str)]) -> std::ffi::CString {
    let mut body = PREP.to_owned();
    for (slot, text) in parts {
        body = body.replace(slot, text);
    }
    std::ffi::CString::new(body).unwrap_or_default()
}

fn prep_preview_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_gdn_rows_prep_preview",
            &[
                c"rows", c"hist", c"w", c"r", c"wq", c"wk", c"a", c"a_log", c"dt_bias",
            ],
            &[c"q", c"qu", c"k", c"v", c"g"],
            &source(&[
                ("{GATES}", GATES),
                ("{RIGHT_LOAD}", RIGHT_LOAD),
                ("{RIGHT_TAPS}", RIGHT_TAPS),
                ("{QUNIT}", QUNIT),
                ("{WINDOW}", ""),
            ]),
            c"",
        )
    })
}

fn prep_verify_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_gdn_rows_prep_verify",
            &[c"rows", c"hist", c"w", c"wq", c"wk"],
            &[c"q", c"k", c"v", c"win"],
            &source(&[
                ("{GATES}", ""),
                ("{RIGHT_LOAD}", ""),
                ("{RIGHT_TAPS}", ""),
                ("{QUNIT}", ""),
                ("{WINDOW}", WINDOW),
            ]),
            c"",
        )
    })
}

fn mem_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_gdn_rows_preview_mem",
            &[
                c"q", c"qu", c"k", c"v", c"g", c"state", c"scale", c"hbias", c"hc",
            ],
            &[c"out"],
            &std::ffi::CString::new(MEM).unwrap_or_default(),
            c"",
        )
    })
}

fn fold_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_gdn_rows_fold",
            &[
                c"state_in",
                c"k",
                c"a",
                c"a_log",
                c"dt_bias",
                c"delta",
                c"hist",
                c"qkv",
                c"n",
            ],
            &[c"state_out", c"win"],
            &std::ffi::CString::new(FOLD).unwrap_or_default(),
            &std::ffi::CString::new(crate::qwen3_next::GDN_RECURRENCE_METAL_PREAMBLE)
                .unwrap_or_default(),
        )
    })
}

/// One layer's geometry, read off the arrays: `rows` `[1, T, C]`, `state`
/// `[1, Hv, D, D]`, `conv_w` `[K, C]`.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub rows: i32,
    pub hk: i32,
    pub hv: i32,
    pub c: i32,
    pub k: i32,
}

impl Geometry {
    /// `InT`: the rows' dtype, where the MLX chain rounds.
    const fn template(self, dtype: Dtype) -> [TemplateArg<'static>; 7] {
        [
            TemplateArg::Dtype(c"InT", dtype),
            TemplateArg::Int(c"D", D),
            TemplateArg::Int(c"HK", self.hk),
            TemplateArg::Int(c"HV", self.hv),
            TemplateArg::Int(c"C", self.c),
            TemplateArg::Int(c"K", self.k),
            TemplateArg::Int(c"T", self.rows),
        ]
    }
}

/// `HIGGS_GDN_ROWS=0` keeps fast rows on the MLX chains (A/B switch).
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("HIGGS_GDN_ROWS").map_or(true, |v| v != "0"))
}

/// The shapes the kernels take, or `None`.
///
/// One sequence of `1..=max_rows` 16/32-bit float rows `[1, T, C]`, history
/// `[1, K - 1, C]` in the same dtype, fp32 state `[1, Hv, 128, 128]`,
/// `C = (2 Hk + Hv) 128`, `K >= 2`.
pub fn geometry(
    rows: &Array,
    hist: &Array,
    state: &Array,
    conv_w: &Array,
    max_rows: i32,
) -> Option<Geometry> {
    let ([b, t, c], [bs, hv, dv, dk], [k, cw]) = (rows.shape(), state.shape(), conv_w.shape())
    else {
        return None;
    };
    let hk = (c - hv * D) / (2 * D);
    let g = Geometry {
        rows: *t,
        hk,
        hv: *hv,
        c: *c,
        k: *k,
    };
    (*b == 1
        && *bs == 1
        && (1..=max_rows).contains(t)
        && *dv == D
        && *dk == D
        && *k >= 2
        && hk > 0
        && hv % hk == 0
        && *c == (2 * hk + hv) * D
        && cw == c
        && matches!(
            rows.dtype(),
            Dtype::Bfloat16 | Dtype::Float16 | Dtype::Float32
        )
        && hist.dtype() == rows.dtype()
        && hist.shape() == [1, k - 1, *c]
        && state.dtype() == Dtype::Float32)
        .then_some(g)
}

/// Inputs of one layer's Clover preview mixer.
///
/// The block's mixed `q|k|v` `[1, T, C]`, gate projection `a` `[1, T, Hv]`,
/// the committed convolution history (oldest first) and fp32 state, the
/// causal taps `[K, C]` (tap `K - 1` is the current row), the right taps
/// `[C, 1, K - 1]`, the q/k norm weights `[D]`, and the per-head decay and
/// softmax parameters `[Hv]`.
pub struct PreviewInputs<'a> {
    pub rows: &'a Array,
    pub a: &'a Array,
    pub history: &'a Array,
    pub state: &'a Array,
    pub conv_w: &'a Array,
    pub conv_r: &'a Array,
    pub norm_q: &'a Array,
    pub norm_k: &'a Array,
    pub a_log: &'a Array,
    pub dt_bias: &'a Array,
    pub scale: &'a Array,
    pub hist_bias: &'a Array,
    pub hist_c: &'a Array,
}

/// The preview mixer's output `[1, T, Hv, D]` fp32 (before the gated norm);
/// call only with a `geometry(.., 32)`.
pub(crate) fn preview_mixer(x: &PreviewInputs<'_>, g: Geometry) -> Result<Array, Exception> {
    let (t, hv) = (g.rows, g.hv);
    let mut prep = prep_preview_kernel().apply(
        &[
            x.rows, x.history, x.conv_w, x.conv_r, x.norm_q, x.norm_k, x.a, x.a_log, x.dt_bias,
        ],
        &g.template(x.rows.dtype()),
        [g.c + D, 1, 1],
        [D, 1, 1],
        &[
            (&[1, t, g.hk, D], Dtype::Float32),
            (&[1, t, g.hk, D], Dtype::Float32),
            (&[1, t, g.hk, D], Dtype::Float32),
            (&[1, t, hv, D], Dtype::Float32),
            (&[1, t, hv], Dtype::Float32),
        ],
    )?;
    let gate = prep.remove(4);
    let v = prep.remove(3);
    let k = prep.remove(2);
    let qu = prep.remove(1);
    let q = prep.remove(0);
    Ok(mem_kernel()
        .apply(
            &[
                &q,
                &qu,
                &k,
                &v,
                &gate,
                x.state,
                x.scale,
                x.hist_bias,
                x.hist_c,
            ],
            &g.template(x.rows.dtype()),
            [hv * 128, D / 32, 1],
            [128, 1, 1],
            &[(&[1, t, hv, D], Dtype::Float32)],
        )?
        .remove(0))
}

/// Verify prep: `(q, k, v, window)` with `q`, `k` `[1, T, Hk, D]` and `v`
/// `[1, T, Hv, D]` fp32, `window` `[1, K - 1, C]` the next convolution
/// history; call only with a `geometry(.., 8)`.
pub(crate) fn prep_verify(
    rows: &Array,
    history: &Array,
    conv_w: &Array,
    norm_q: &Array,
    norm_k: &Array,
    g: Geometry,
) -> Result<(Array, Array, Array, Array), Exception> {
    let t = g.rows;
    let mut out = prep_verify_kernel().apply(
        &[rows, history, conv_w, norm_q, norm_k],
        &g.template(rows.dtype()),
        [g.c, 1, 1],
        [D, 1, 1],
        &[
            (&[1, t, g.hk, D], Dtype::Float32),
            (&[1, t, g.hk, D], Dtype::Float32),
            (&[1, t, g.hv, D], Dtype::Float32),
            (&[1, g.k - 1, g.c], rows.dtype()),
        ],
    )?;
    let win = out.remove(3);
    let v = out.remove(2);
    let k = out.remove(1);
    Ok((out.remove(0), k, v, win))
}

/// Advance `state_in` by the first `n` tape rows (`k` `[1, T, Hk, D]` fp32,
/// `a` `[1, T, Hv]`, `delta` `[1, T, Hv, D]` fp32) and the window by the
/// first `n` of `qkv` `[1, T, C]`: `(state, window)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold(
    state_in: &Array,
    k: &Array,
    a: &Array,
    a_log: &Array,
    dt_bias: &Array,
    delta: &Array,
    history: &Array,
    qkv: &Array,
    n: i32,
    g: Geometry,
) -> Result<(Array, Array), Exception> {
    if !(0..=g.rows).contains(&n) {
        return Err(Exception::custom(format!(
            "GDN fold of {n} rows from a {}-row tape",
            g.rows
        )));
    }
    let mut out = fold_kernel().apply(
        &[
            state_in,
            k,
            a,
            a_log,
            dt_bias,
            delta,
            history,
            qkv,
            &Array::from_int(n),
        ],
        &g.template(qkv.dtype()),
        [g.hv * D * (D / 4) + (g.k - 1) * g.c, 1, 1],
        [256, 1, 1],
        &[
            (&[1, g.hv, D, D], Dtype::Float32),
            (&[1, g.k - 1, g.c], qkv.dtype()),
        ],
    )?;
    let win = out.remove(1);
    Ok((out.remove(0), win))
}

#[doc(hidden)]
pub mod probe {
    //! Test seam (higgs-models unit tests do not compile).
    use super::{Array, Exception, Geometry, PreviewInputs};

    /// The kernels' preview mixer output `[1, T, Hv, D]` fp32.
    pub fn preview_mixer(x: &PreviewInputs<'_>) -> Result<Array, Exception> {
        let g = super::geometry(x.rows, x.history, x.state, x.conv_w, 32)
            .ok_or_else(|| Exception::custom("preview geometry unsupported"))?;
        super::preview_mixer(x, g)
    }

    /// The MLX chain's preview mixer output `[1, T, Hv, D]` in the rows' dtype.
    pub fn preview_mixer_mlx(x: &PreviewInputs<'_>) -> Result<Array, Exception> {
        crate::qwen3_next::preview_mixer_mlx(x)
    }

    pub fn geometry(
        rows: &Array,
        hist: &Array,
        state: &Array,
        conv_w: &Array,
        max_rows: i32,
    ) -> Option<Geometry> {
        super::geometry(rows, hist, state, conv_w, max_rows)
    }

    pub fn prep_verify(
        rows: &Array,
        history: &Array,
        conv_w: &Array,
        norm_q: &Array,
        norm_k: &Array,
        g: Geometry,
    ) -> Result<(Array, Array, Array, Array), Exception> {
        super::prep_verify(rows, history, conv_w, norm_q, norm_k, g)
    }

    /// The verify recurrence: `(y, state_out, delta_tape)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_with_tape(
        q: &Array,
        k: &Array,
        v: &Array,
        a_log: &Array,
        a: &Array,
        dt_bias: &Array,
        b: &Array,
        state: &Array,
        g: Geometry,
    ) -> Result<(Array, Array, Array), Exception> {
        crate::qwen3_next::gated_delta_kernel_ffi_with_tape(
            q,
            k,
            v,
            a_log,
            a,
            dt_bias,
            b,
            state,
            1,
            g.rows,
            g.hk,
            super::D,
            g.hv,
            super::D,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fold(
        state_in: &Array,
        k: &Array,
        a: &Array,
        a_log: &Array,
        dt_bias: &Array,
        delta: &Array,
        history: &Array,
        qkv: &Array,
        n: i32,
        g: Geometry,
    ) -> Result<(Array, Array), Exception> {
        super::fold(state_in, k, a, a_log, dt_bias, delta, history, qkv, n, g)
    }
}
