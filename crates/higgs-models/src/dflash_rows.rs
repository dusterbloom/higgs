//! Fused `DFlash2` draft kernels for fast rows (the Clover self-speculative
//! drafter), replacing ~300 small MLX dispatches per draft with ~75.
//!
//! One dispatch each for a row RMS norm, a two-tap dynamic-convolution
//! prepare, the convolution finish + residual + next norm, the drafter
//! attention of one layer (q/k head norms, `RoPE`, context + block keys,
//! sliding mask, softmax, `P.V`, in the `o_proj` input layout), one layer's
//! context K/V commit (k norm, `RoPE`, window trim, append), and the `DFlash2`
//! selector (chunked top-16 + merge, then the lattice edges and the greedy
//! walk in one threadgroup).
//!
//! After Kapitsa's `spec_dflash.metal` (`dfl_conv_prep`, `dfl_conv_fin_norm`,
//! `dfl_qk_prep`, `dfl_attn`, `dfl_ctx_commit`, `dfl_top16` / `dfl_merge16`,
//! `dfl_lattice`, `dfl_walk`; Apache-2.0, nace-ai/kapitsa). The projections
//! themselves run on [`crate::qmm8`] (gate|up with the `SwiGLU` in its
//! epilogue). Everything here is fp32 arithmetic from
//! 16-bit inputs: within bf16 rounding of the MLX chain, not bit-identical,
//! so it runs only inside [`crate::qwen3_next::with_fast_rows`]; the drafts
//! only steer acceptance, the verify's greedy output is unchanged.

use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, error::Exception};

use crate::metal_kernel::{FastKernel, TemplateArg};

/// Head dimension the attention and commit kernels are laid out for.
pub(crate) const D: i32 = 128;
/// Channels per dynamic-convolution group.
pub(crate) const CONV_GROUP: i32 = 16;
/// Candidates per position of the selector.
pub(crate) const TOP_K: i32 = 16;
/// Elements per top-16 chunk (256 threads x 8).
const CHUNK: i32 = 2048;

/// `y[r] = BT(x[r] * rsqrt(mean(x[r]^2) + eps) * w)`; one threadgroup of 256
/// threads per row, `H / 256` consecutive channels per thread.
const NORM: &str = r"
  threadgroup float red[8];
  const uint r = threadgroup_position_in_grid.x;
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  constexpr int PER = H / 256;
  const float eps = as_type<float>(uint(EPS_BITS));
  float v[PER];
  float ss = 0.0f;
  for (int i = 0; i < PER; ++i) {
    v[i] = float(x[size_t(r) * H + tid * PER + i]);
    ss += v[i] * v[i];
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float inv = metal::precise::rsqrt(tot / float(H) + eps);
  for (int i = 0; i < PER; ++i) {
    const int c = int(tid) * PER + i;
    y[size_t(r) * H + c] = BT(v[i] * inv * float(w[c]));
  }
";

/// Prepare stage of the two-tap grouped dynamic convolution: `d` `[T, 4 G]`
/// holds `[stage][tap][group]`, `base` `[2][2][H]`;
/// `a[r] = (base[0][0] + d[r][0][0][g]) n[r] + (base[0][1] + d[r][0][1][g]) n[r - 1]`.
const CONV_PREP: &str = r"
  const uint r = threadgroup_position_in_grid.x;
  const uint tid = thread_position_in_threadgroup.x;
  constexpr int PER = H / 256;
  constexpr int G = H / GS;
  for (int i = 0; i < PER; ++i) {
    const int c = int(tid) * PER + i;
    const int g = c / GS;
    const float d0 = float(d[size_t(r) * 4 * G + g]);
    const float d1 = float(d[size_t(r) * 4 * G + G + g]);
    const float xc = float(n[size_t(r) * H + c]);
    const float xp = r > 0 ? float(n[size_t(r - 1) * H + c]) : 0.0f;
    a[size_t(r) * H + c] = BT((float(base[c]) + d0) * xc + (float(base[H + c]) + d1) * xp);
  }
";

/// Finish stage + residual + the next norm:
/// `h2[r] = h[r] + (base[1][0] + d[r][1][0][g]) u[r] + (base[1][1] + d[r][1][1][g]) u[r - 1]`,
/// `n = BT(h2 * rsqrt(mean(h2^2) + eps) * w)`; `h2` stays fp32.
const CONV_FIN_NORM: &str = r"
  threadgroup float red[8];
  const uint r = threadgroup_position_in_grid.x;
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  constexpr int PER = H / 256;
  constexpr int G = H / GS;
  const float eps = as_type<float>(uint(EPS_BITS));
  float v[PER];
  float ss = 0.0f;
  for (int i = 0; i < PER; ++i) {
    const int c = int(tid) * PER + i;
    const int g = c / GS;
    const float d0 = float(d[size_t(r) * 4 * G + 2 * G + g]);
    const float d1 = float(d[size_t(r) * 4 * G + 3 * G + g]);
    const float uc = float(u[size_t(r) * H + c]);
    const float up = r > 0 ? float(u[size_t(r - 1) * H + c]) : 0.0f;
    v[i] = float(h[size_t(r) * H + c]) + (float(base[2 * H + c]) + d0) * uc + (float(base[3 * H + c]) + d1) * up;
    h2[size_t(r) * H + c] = v[i];
    ss += v[i] * v[i];
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float inv = metal::precise::rsqrt(tot / float(H) + eps);
  for (int i = 0; i < PER; ++i) {
    const int c = int(tid) * PER + i;
    n[size_t(r) * H + c] = BT(v[i] * inv * float(w[c]));
  }
";

/// One layer's draft attention: threadgroup = one q head (`h`), thread = one
/// head dim; `T` block rows. Q and the block keys get their head RMS norm and
/// `RoPE` at `pos[0] + r` here; keys are the context `ck` `[1, HKV, L, D]`
/// (visible to row `r` iff `L + r - j < W`, the window on the concatenated
/// index as in the MLX chain) then the `T` block keys (always visible); fp32
/// scores in tiles of 128 keys with an online softmax; `o[r][h D + d]`.
const ATTN: &str = r"
  threadgroup float qs[T * D];
  threadgroup float kb[T * D];
  threadgroup float vb[T * D];
  threadgroup float sc[T * D];
  threadgroup float rowred[T * 8];
  threadgroup float stm[T], stl[T], sta[T];
  const uint h = threadgroup_position_in_grid.x;
  const uint d = thread_position_in_threadgroup.x;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint hk = h / uint(HQ / HKV);
  const int L = ck_shape[2];
  const int P = pos[0];
  const float eps = as_type<float>(uint(EPS_BITS));
  const float base = as_type<float>(uint(BASE_BITS));
  const float inv_freq = metal::exp2(-float(d & 63) / 64.0f * metal::log2(base));
  const float scale = metal::precise::rsqrt(float(D));
  constexpr int QLD = HQ * D, KLD = HKV * D;
  float xq[T], xk[T];
  for (int r = 0; r < T; ++r) {
    xq[r] = float(q[size_t(r) * QLD + h * D + d]);
    xk[r] = float(kn[size_t(r) * KLD + hk * D + d]);
    vb[r * D + d] = float(vn[size_t(r) * KLD + hk * D + d]);
  }
  for (int r = 0; r < T; ++r) {
    const float a = simd_sum(xq[r] * xq[r]);
    const float b = simd_sum(xk[r] * xk[r]);
    if (lane == 0) {
      rowred[r * 8 + sg] = a;
      rowred[r * 8 + 4 + sg] = b;
    }
  }
  if (d < uint(T)) {
    stm[d] = -INFINITY;
    stl[d] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = 0; r < T; ++r) {
    const float sq = (rowred[r * 8] + rowred[r * 8 + 1]) + (rowred[r * 8 + 2] + rowred[r * 8 + 3]);
    const float sk = (rowred[r * 8 + 4] + rowred[r * 8 + 5]) + (rowred[r * 8 + 6] + rowred[r * 8 + 7]);
    qs[r * D + d] = xq[r] * metal::precise::rsqrt(sq / float(D) + eps) * float(qw[d]);
    kb[r * D + d] = xk[r] * metal::precise::rsqrt(sk / float(D) + eps) * float(kw[d]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = 0; r < T; ++r) {
    const float ang = float(P + r) * inv_freq;
    const float c = metal::precise::cos(ang), s = metal::precise::sin(ang);
    const float q0 = qs[r * D + d], q1 = qs[r * D + (d ^ 64)];
    const float k0 = kb[r * D + d], k1 = kb[r * D + (d ^ 64)];
    xq[r] = (d < 64 ? q0 * c - q1 * s : q0 * c + q1 * s) * scale;
    xk[r] = d < 64 ? k0 * c - k1 * s : k0 * c + k1 * s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = 0; r < T; ++r) {
    qs[r * D + d] = xq[r];
    kb[r * D + d] = xk[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[T];
  for (int r = 0; r < T; ++r) acc[r] = 0.0f;
  const int NK = L + T;
  const device BT* ck_h = ck + size_t(hk) * size_t(L) * D;
  const device BT* cv_h = cv + size_t(hk) * size_t(L) * D;
  // Softmax statistics: thread (rr, part) owns 8 keys of row rr in each tile.
  const int rr = int(d) >> 4, part = int(d) & 15;
  for (int j0 = 0; j0 < NK; j0 += D) {
    const int j = j0 + int(d);
    float s[T];
    for (int r = 0; r < T; ++r) s[r] = -INFINITY;
    if (j < L) {
      const device BT* kr = ck_h + size_t(j) * D;
      for (int r = 0; r < T; ++r) s[r] = 0.0f;
      for (int i = 0; i < D; i += 4) {
        const uint2 w = *reinterpret_cast<const device uint2*>(kr + i);
        const float4 k4 = float4(as_type<float>(w.x << 16), as_type<float>(w.x & 0xFFFF0000u),
                                 as_type<float>(w.y << 16), as_type<float>(w.y & 0xFFFF0000u));
        for (int r = 0; r < T; ++r)
          s[r] += dot(*reinterpret_cast<const threadgroup float4*>(qs + r * D + i), k4);
      }
      for (int r = 0; r < T; ++r)
        if (L + r - j >= W) s[r] = -INFINITY;
    } else if (j < NK) {
      const int jb = j - L;
      for (int r = 0; r < T; ++r) {
        float a = 0.0f;
        for (int i = 0; i < D; i += 4)
          a += dot(*reinterpret_cast<const threadgroup float4*>(qs + r * D + i),
                   *reinterpret_cast<const threadgroup float4*>(kb + jb * D + i));
        s[r] = a;
      }
    }
    for (int r = 0; r < T; ++r) sc[r * D + d] = s[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tmax = -INFINITY;
    if (rr < T)
      for (int i = 0; i < 8; ++i) tmax = fmax(tmax, sc[rr * D + part * 8 + i]);
    tmax = fmax(tmax, simd_shuffle_xor(tmax, 8));
    tmax = fmax(tmax, simd_shuffle_xor(tmax, 4));
    tmax = fmax(tmax, simd_shuffle_xor(tmax, 2));
    tmax = fmax(tmax, simd_shuffle_xor(tmax, 1));
    const float mold = rr < T ? stm[rr] : -INFINITY;
    const float mnew = fmax(mold, tmax);
    const float mexp = mnew == -INFINITY ? 0.0f : mnew;
    float psum = 0.0f;
    if (rr < T) {
      for (int i = 0; i < 8; ++i) {
        const float sv = sc[rr * D + part * 8 + i];
        const float e = sv == -INFINITY ? 0.0f : metal::precise::exp(sv - mexp);
        sc[rr * D + part * 8 + i] = e;
        psum += e;
      }
    }
    psum += simd_shuffle_xor(psum, 8);
    psum += simd_shuffle_xor(psum, 4);
    psum += simd_shuffle_xor(psum, 2);
    psum += simd_shuffle_xor(psum, 1);
    if (rr < T && part == 0) {
      const float alpha = mold == -INFINITY ? 1.0f : metal::precise::exp(mold - mnew);
      stm[rr] = mnew;
      stl[rr] = stl[rr] * alpha + psum;
      sta[rr] = alpha;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = 0; r < T; ++r) acc[r] *= sta[r];
    const int nkeys = min(D, NK - j0);
    for (int jj = 0; jj < nkeys; ++jj) {
      const int jg = j0 + jj;
      const float vv = jg < L ? float(cv_h[size_t(jg) * D + d]) : vb[(jg - L) * D + d];
      for (int r = 0; r < T; ++r) acc[r] += sc[r * D + jj] * vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int r = 0; r < T; ++r) o[size_t(r) * QLD + h * D + d] = BT(acc[r] / stl[r]);
";

/// One layer's context commit: `kv` `[R, NL 2 HKV D]` (the stacked k|v
/// projections of all layers, layer `l` at `l 2 HKV D`, its v at `+ HKV D`),
/// `ck`/`cv` `[1, HKV, L, D]`; the outputs hold the last `L2 = min(L + R,
/// W - 1)` rows: the kept old rows then the `R` new ones (k: head RMS norm with
/// `kw`, `RoPE` at `pos[0] + r`). Threadgroup = 32 threads = one output row of
/// one head, 4 dims per thread.
const CTX_COMMIT: &str = r"
  const uint tg = threadgroup_position_in_grid.x;
  const uint lane = thread_index_in_simdgroup;
  const int L = ck_shape[2];
  const int L2 = min(L + R, W - 1);
  const uint hh = tg / uint(L2);
  const int i = int(tg % uint(L2));
  const int kept = L2 - R;
  const int skip = L + R - L2;
  const int d0 = int(lane) * 4;
  const size_t dst = (size_t(hh) * L2 + i) * D + d0;
  if (i < kept) {
    const size_t src = (size_t(hh) * L + (i + skip)) * D + d0;
    *reinterpret_cast<device uint2*>(ok + dst) = *reinterpret_cast<const device uint2*>(ck + src);
    *reinterpret_cast<device uint2*>(ov + dst) = *reinterpret_cast<const device uint2*>(cv + src);
    return;
  }
  const int r = i - kept;
  constexpr int KVLD = NL * 2 * HKV * D;
  const device BT* ks = kv + size_t(r) * KVLD + (LAYER * 2 * HKV + hh) * D + d0;
  const device BT* vs = ks + HKV * D;
  const float eps = as_type<float>(uint(EPS_BITS));
  const float base = as_type<float>(uint(BASE_BITS));
  float x[4];
  float ss = 0.0f;
  for (int c = 0; c < 4; ++c) {
    x[c] = float(ks[c]);
    ss += x[c] * x[c];
  }
  ss = simd_sum(ss);
  const float inv = metal::precise::rsqrt(ss / float(D) + eps);
  float t[4], p[4];
  for (int c = 0; c < 4; ++c) t[c] = x[c] * inv * float(kw[d0 + c]);
  // RoPE partner dim d ^ 64 lives in lane ^ 16.
  for (int c = 0; c < 4; ++c) p[c] = simd_shuffle_xor(t[c], 16);
  const float position = float(pos[0] + r);
  for (int c = 0; c < 4; ++c) {
    const int dd = d0 + c;
    const float ang = position * metal::exp2(-float(dd & 63) / 64.0f * metal::log2(base));
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    ok[dst + c] = BT(dd < 64 ? t[c] * cs - p[c] * sn : t[c] * cs + p[c] * sn);
    ov[dst + c] = vs[c];
  }
";

/// Kapitsa's ordered top-16 (value desc, id asc): every lane sorts a run of
/// 8, 16 rounds of (`simd_max`, `simd_min` of the ids at that max) pop the
/// simdgroup's best; the 8 simdgroups' lists merge the same way in simdgroup 0.
const TOP16_HEADER: &str = r#"
inline bool dfl_before(float va, uint ia, float vb, uint ib) { return va > vb || (va == vb && ia < ib); }
#define DFL_CS(a, b)                                     \
  if (dfl_before(v[b], id[b], v[a], id[a])) {             \
    const float tv = v[a]; v[a] = v[b]; v[b] = tv;        \
    const uint ti = id[a]; id[a] = id[b]; id[b] = ti;     \
  }
inline void dfl_sort8(thread float* v, thread uint* id) {
  DFL_CS(0, 1) DFL_CS(2, 3) DFL_CS(4, 5) DFL_CS(6, 7)
  DFL_CS(0, 2) DFL_CS(1, 3) DFL_CS(4, 6) DFL_CS(5, 7)
  DFL_CS(1, 2) DFL_CS(5, 6) DFL_CS(0, 4) DFL_CS(3, 7)
  DFL_CS(1, 5) DFL_CS(2, 6)
  DFL_CS(1, 4) DFL_CS(3, 6)
  DFL_CS(2, 4) DFL_CS(3, 5)
  DFL_CS(3, 4)
}
// Lane q < 16 receives the q-th largest (value, id) of the simdgroup's 256.
inline void dfl_simd_top16(thread float* v, thread uint* id, uint lane, thread float& ov, thread uint& oi) {
  ov = -INFINITY;
  oi = 0xffffffffu;
  for (uint q = 0; q < 16; ++q) {
    const float mv = simd_max(v[0]);
    const uint mi = simd_min(v[0] == mv ? id[0] : 0xffffffffu);
    if (lane == q) {
      ov = mv;
      oi = mi;
    }
    if (v[0] == mv && id[0] == mi) {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 7; ++j) {
        v[j] = v[j + 1];
        id[j] = id[j + 1];
      }
      v[7] = -INFINITY;
      id[7] = 0xffffffffu;
    }
  }
}
"#;

/// Grid `(NCH chunks of 2048, rows)` x 256: `out[(r NCH + chunk) 16 + q]` is
/// the chunk's ordered top-16 of `src[r LD + k]`, `k < N`, with ids `k`
/// (`HAS_ID` false) or `src_id[r LD + k]`.
const TOP16: &str = r"
  threadgroup float sv[8 * 16];
  threadgroup uint si[8 * 16];
  const uint chunk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  const uint tid = thread_position_in_threadgroup.x;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const device ST* s = src;
  const uint base = chunk * 2048 + tid * 8;
  float v[8];
  uint id[8];
  for (int j = 0; j < 8; ++j) {
    const uint k = base + uint(j);
    const bool ok = k < uint(N);
    v[j] = ok ? float(s[size_t(r) * LD + k]) : -INFINITY;
    id[j] = ok ? (HAS_ID ? src_id[size_t(r) * LD + k] : k) : 0xffffffffu;
  }
  dfl_sort8(v, id);
  float bv;
  uint bi;
  dfl_simd_top16(v, id, lane, bv, bi);
  if (lane < 16) {
    sv[sg * 16 + lane] = bv;
    si[sg * 16 + lane] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  for (int j = 0; j < 4; ++j) {
    v[j] = sv[lane * 4 + j];
    id[j] = si[lane * 4 + j];
    v[4 + j] = -INFINITY;
    id[4 + j] = 0xffffffffu;
  }
  dfl_simd_top16(v, id, lane, bv, bi);
  if (lane < 16) {
    const size_t o = (size_t(r) * NCH + chunk) * 16 + lane;
    ov[o] = bv;
    oi[o] = bi;
  }
";

/// The selector's edge lattice and greedy walk, one threadgroup of 256:
/// thread (candidate `k = tid / 16`, `part = tid % 16`) sums `RANK / 16` dims
/// of `A[pred] . sel[p] . B[cand[p][k]]` (8-bit g64 codebooks dequantized in
/// place); per position `S[k] = cv[p][k] + edge`, the first argmax is the
/// draft and the next position's predecessor (`anchor[0]` for `p = 0`).
const LATTICE_WALK: &str = r"
  threadgroup float S[16];
  threadgroup uint jsel;
  const uint tid = thread_position_in_threadgroup.x;
  const uint k = tid >> 4, part = tid & 15;
  constexpr int WPR = RANK / 4;
  constexpr int GPR = RANK / 64;
  constexpr int PER = RANK / 16;
  const int i0 = int(part) * PER;
  if (tid == 0) jsel = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int p = 0; p < K; ++p) {
    const uint pred = p == 0 ? uint(anchor[0]) : ci[(p - 1) * 16 + jsel];
    const uint succ = ci[p * 16 + k];
    float acc = 0.0f;
    for (int i = i0; i < i0 + PER; ++i) {
      const uint wa = a_w[size_t(pred) * WPR + i / 4];
      const uint wb = b_w[size_t(succ) * WPR + i / 4];
      const uint sh = 8 * uint(i & 3);
      const float a = float(a_s[size_t(pred) * GPR + i / 64]) * float((wa >> sh) & 255u) + float(a_b[size_t(pred) * GPR + i / 64]);
      const float b = float(b_s[size_t(succ) * GPR + i / 64]) * float((wb >> sh) & 255u) + float(b_b[size_t(succ) * GPR + i / 64]);
      acc += a * float(sel[size_t(p) * RANK + i]) * b;
    }
    acc += simd_shuffle_xor(acc, 8);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (part == 0) S[k] = cv[p * 16 + k] + acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint best = 0;
      float bv = S[0];
      for (uint kk = 1; kk < 16; ++kk) {
        if (S[kk] > bv) {
          bv = S[kk];
          best = kk;
        }
      }
      jsel = best;
      tok[p] = ci[p * 16 + best];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
";

fn cstr(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).unwrap_or_default()
}

fn norm_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_dflash_rows_norm",
            &[c"x", c"w"],
            &[c"y"],
            &cstr(NORM),
            c"",
        )
    })
}

fn conv_prep_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_dflash_rows_conv_prep",
            &[c"n", c"d", c"base"],
            &[c"a"],
            &cstr(CONV_PREP),
            c"",
        )
    })
}

fn conv_fin_norm_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_dflash_rows_conv_fin_norm",
            &[c"h", c"u", c"d", c"base", c"w"],
            &[c"h2", c"n"],
            &cstr(CONV_FIN_NORM),
            c"",
        )
    })
}

fn attn_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_dflash_rows_attn",
            &[c"q", c"kn", c"vn", c"qw", c"kw", c"ck", c"cv", c"pos"],
            &[c"o"],
            &cstr(ATTN),
            c"",
        )
    })
}

fn ctx_commit_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_dflash_rows_ctx_commit",
            &[c"kv", c"kw", c"pos", c"ck", c"cv"],
            &[c"ok", c"ov"],
            &cstr(CTX_COMMIT),
            c"",
        )
    })
}

fn top16_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_dflash_rows_top16",
            &[c"src", c"src_id"],
            &[c"ov", c"oi"],
            &cstr(TOP16),
            &cstr(TOP16_HEADER),
        )
    })
}

fn lattice_walk_kernel() -> &'static FastKernel {
    static KERNEL: OnceLock<FastKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        FastKernel::new(
            c"higgs_dflash_rows_lattice_walk",
            &[
                c"ci", c"cv", c"sel", c"a_w", c"a_s", c"a_b", c"b_w", c"b_s", c"b_b", c"anchor",
            ],
            &[c"tok"],
            &cstr(LATTICE_WALK),
            c"",
        )
    })
}

/// `HIGGS_DFLASH_ROWS=0` keeps fast-rows drafts on the MLX chain (A/B switch).
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("HIGGS_DFLASH_ROWS").map_or(true, |v| v != "0"))
}

/// A float as the i32 of its bits (template ints are the only scalars).
const fn bits(v: f32) -> i32 {
    i32::from_ne_bytes(v.to_bits().to_ne_bytes())
}

const BF16: TemplateArg<'static> = TemplateArg::Dtype(c"BT", Dtype::Bfloat16);

fn dims2(x: &Array) -> Result<(i32, i32), Exception> {
    match x.shape() {
        [t, h] => Ok((*t, *h)),
        other => Err(Exception::custom(format!(
            "dflash_rows: expected a 2-D array, got {other:?}"
        ))),
    }
}

fn one(out: Vec<Array>) -> Result<Array, Exception> {
    out.into_iter()
        .next()
        .ok_or_else(|| Exception::custom("dflash_rows: kernel returned no output"))
}

fn two(mut out: Vec<Array>) -> Result<(Array, Array), Exception> {
    let second = out
        .pop()
        .ok_or_else(|| Exception::custom("dflash_rows: kernel returned one output"))?;
    Ok((one(out)?, second))
}

/// Whether `H` is a hidden size the row kernels are laid out for.
pub(crate) const fn hidden_supported(h: i32) -> bool {
    h > 0 && h % 256 == 0 && h % CONV_GROUP == 0
}

/// `w * x[r] / rms(x[r])` for `x` `[T, H]` (any float dtype), bf16 out.
pub(crate) fn norm_rows(x: &Array, w: &Array, eps: f32) -> Result<Array, Exception> {
    let (t, h) = dims2(x)?;
    one(norm_kernel().apply(
        &[x, w],
        &[
            BF16,
            TemplateArg::Int(c"H", h),
            TemplateArg::Int(c"EPS_BITS", bits(eps)),
        ],
        [t * 256, 1, 1],
        [256, 1, 1],
        &[(&[t, h], Dtype::Bfloat16)],
    )?)
}

/// Prepare stage of the dynamic convolution: `n` `[T, H]`, `d` `[T, 4 H / 16]`
/// (`[stage][tap][group]`), `base` `[2, 2, H]`; bf16 `[T, H]`.
pub(crate) fn conv_prep(n: &Array, d: &Array, base: &Array) -> Result<Array, Exception> {
    let (t, h) = dims2(n)?;
    one(conv_prep_kernel().apply(
        &[n, d, base],
        &[
            BF16,
            TemplateArg::Int(c"H", h),
            TemplateArg::Int(c"GS", CONV_GROUP),
        ],
        [t * 256, 1, 1],
        [256, 1, 1],
        &[(&[t, h], Dtype::Bfloat16)],
    )?)
}

/// Finish stage + residual + the next RMS norm: `(h + conv_fin(u), norm(w))`
/// as (fp32 `[T, H]`, bf16 `[T, H]`).
pub(crate) fn conv_fin_res_norm(
    h: &Array,
    u: &Array,
    d: &Array,
    base: &Array,
    w: &Array,
    eps: f32,
) -> Result<(Array, Array), Exception> {
    let (t, hs) = dims2(h)?;
    two(conv_fin_norm_kernel().apply(
        &[h, u, d, base, w],
        &[
            BF16,
            TemplateArg::Dtype(c"HT", h.dtype()),
            TemplateArg::Int(c"H", hs),
            TemplateArg::Int(c"GS", CONV_GROUP),
            TemplateArg::Int(c"EPS_BITS", bits(eps)),
        ],
        [t * 256, 1, 1],
        [256, 1, 1],
        &[(&[t, hs], Dtype::Float32), (&[t, hs], Dtype::Bfloat16)],
    )?)
}

/// Geometry shared by the attention and commit kernels.
#[derive(Clone, Copy, Debug)]
pub struct Heads {
    pub hq: i32,
    pub hkv: i32,
    /// Sliding window (`W`); `W - 1` context rows are retained.
    pub window: i32,
    pub eps: f32,
    pub rope_base: f32,
}

/// One layer's draft attention: `q` `[T, HQ D]`, `kn`/`vn` `[T, HKV D]` (the
/// block's projections), `qw`/`kw` `[D]` head-norm weights, `ck`/`cv`
/// `[1, HKV, L, D]` context, `pos` i32 `[1]` the first block position; bf16
/// `[T, HQ D]`.
pub(crate) struct AttnInputs<'a> {
    pub q: &'a Array,
    pub kn: &'a Array,
    pub vn: &'a Array,
    pub qw: &'a Array,
    pub kw: &'a Array,
    pub ck: &'a Array,
    pub cv: &'a Array,
    pub pos: &'a Array,
    pub heads: Heads,
}

pub(crate) fn attention(x: &AttnInputs<'_>) -> Result<Array, Exception> {
    let (t, qld) = dims2(x.q)?;
    let h = x.heads;
    if qld != h.hq * D || t > 8 || h.hq % h.hkv != 0 {
        return Err(Exception::custom(format!(
            "dflash_rows attention: q [{t}, {qld}] with {} / {} heads",
            h.hq, h.hkv
        )));
    }
    one(attn_kernel().apply(
        &[x.q, x.kn, x.vn, x.qw, x.kw, x.ck, x.cv, x.pos],
        &[
            BF16,
            TemplateArg::Int(c"T", t),
            TemplateArg::Int(c"D", D),
            TemplateArg::Int(c"HQ", h.hq),
            TemplateArg::Int(c"HKV", h.hkv),
            TemplateArg::Int(c"W", h.window),
            TemplateArg::Int(c"EPS_BITS", bits(h.eps)),
            TemplateArg::Int(c"BASE_BITS", bits(h.rope_base)),
        ],
        [h.hq * D, 1, 1],
        [D, 1, 1],
        &[(&[t, qld], Dtype::Bfloat16)],
    )?)
}

/// Which of a drafter's layers a commit serves.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CommitLayer {
    pub index: i32,
    pub count: i32,
}

/// Commit the `R` rows of `kv` `[R, NL 2 HKV D]` for `layer.index` of
/// `layer.count` onto `ck`/`cv` `[1, HKV, L, D]`: the new caches `[1, HKV,
/// min(L + R, W - 1), D]`, k head-normed with `kw` and rotated at `pos[0] + r`.
pub(crate) fn ctx_commit(
    kv: &Array,
    kw: &Array,
    pos: &Array,
    ck: &Array,
    cv: &Array,
    layer: CommitLayer,
    heads: Heads,
) -> Result<(Array, Array), Exception> {
    let (r, kvld) = dims2(kv)?;
    let l = ck.dim(2);
    let CommitLayer { index, count } = layer;
    if kvld != count * 2 * heads.hkv * D || r < 1 || r >= heads.window {
        return Err(Exception::custom(format!(
            "dflash_rows commit: kv [{r}, {kvld}] for {count} layers of {} kv heads, window {}",
            heads.hkv, heads.window
        )));
    }
    let l2 = (l + r).min(heads.window - 1);
    let shape = [1, heads.hkv, l2, D];
    two(ctx_commit_kernel().apply(
        &[kv, kw, pos, ck, cv],
        &[
            BF16,
            TemplateArg::Int(c"R", r),
            TemplateArg::Int(c"D", D),
            TemplateArg::Int(c"HKV", heads.hkv),
            TemplateArg::Int(c"NL", count),
            TemplateArg::Int(c"LAYER", index),
            TemplateArg::Int(c"W", heads.window),
            TemplateArg::Int(c"EPS_BITS", bits(heads.eps)),
            TemplateArg::Int(c"BASE_BITS", bits(heads.rope_base)),
        ],
        [heads.hkv * l2 * 32, 1, 1],
        [32, 1, 1],
        &[(&shape, Dtype::Bfloat16), (&shape, Dtype::Bfloat16)],
    )?)
}

/// Per row of `src` `[K, N]` (16/32-bit float) the 16 largest values and
/// their indices, ordered (value desc, index asc): `(f32 [K, 16], u32 [K,
/// 16])`. `None` when `N` exceeds what one merge threadgroup covers.
pub(crate) fn top16(src: &Array) -> Result<Option<(Array, Array)>, Exception> {
    let (k, n) = dims2(src)?;
    let nch = (n + CHUNK - 1) / CHUNK;
    if n < 1 || nch * TOP_K > CHUNK {
        return Ok(None);
    }
    let dummy_ids = Array::from_slice(&[0_u32], &[1]);
    let pass = |values: &Array, ids: &Array, has_id: bool, width: i32, chunks: i32| {
        two(top16_kernel().apply(
            &[values, ids],
            &[
                TemplateArg::Dtype(c"ST", values.dtype()),
                TemplateArg::Int(c"N", width),
                TemplateArg::Int(c"LD", width),
                TemplateArg::Int(c"NCH", chunks),
                TemplateArg::Bool(c"HAS_ID", has_id),
            ],
            [chunks * 256, k, 1],
            [256, 1, 1],
            &[
                (&[k, chunks * TOP_K], Dtype::Float32),
                (&[k, chunks * TOP_K], Dtype::Uint32),
            ],
        )?)
    };
    let (pv, pi) = pass(src, &dummy_ids, false, n, nch)?;
    if nch == 1 {
        return Ok(Some((pv, pi)));
    }
    pass(&pv, &pi, true, nch * TOP_K, 1).map(Some)
}

/// An 8-bit group-64 affine codebook `(weight [V, RANK / 4] u32, scales [V,
/// RANK / 64], biases [V, RANK / 64])`.
pub(crate) type Codebook<'a> = (&'a Array, &'a Array, &'a Array);

/// The selector's greedy walk: `ci`/`cv` `[K, 16]` candidates (ids u32,
/// unary scores f32), `sel` `[K, RANK]` the projected hiddens, `anchor` i32
/// `[1]`; the `K` drafted ids as u32 `[K]`.
pub(crate) fn lattice_walk(
    ci: &Array,
    cv: &Array,
    sel: &Array,
    pred: Codebook<'_>,
    succ: Codebook<'_>,
    anchor: &Array,
) -> Result<Array, Exception> {
    let (k, rank) = dims2(sel)?;
    if rank % 64 != 0 || ci.shape() != [k, TOP_K] || cv.shape() != [k, TOP_K] {
        return Err(Exception::custom(format!(
            "dflash_rows lattice: sel [{k}, {rank}], candidates {:?}",
            ci.shape()
        )));
    }
    one(lattice_walk_kernel().apply(
        &[
            ci, cv, sel, pred.0, pred.1, pred.2, succ.0, succ.1, succ.2, anchor,
        ],
        &[TemplateArg::Int(c"K", k), TemplateArg::Int(c"RANK", rank)],
        [256, 1, 1],
        [256, 1, 1],
        &[(&[k], Dtype::Uint32)],
    )?)
}

#[doc(hidden)]
pub mod probe {
    //! Test seam (higgs-models unit tests do not compile).
    use super::{Array, Exception};

    pub use super::Heads;

    pub fn norm_rows(x: &Array, w: &Array, eps: f32) -> Result<Array, Exception> {
        super::norm_rows(x, w, eps)
    }

    pub fn conv_prep(n: &Array, d: &Array, base: &Array) -> Result<Array, Exception> {
        super::conv_prep(n, d, base)
    }

    pub fn conv_fin_res_norm(
        h: &Array,
        u: &Array,
        d: &Array,
        base: &Array,
        w: &Array,
        eps: f32,
    ) -> Result<(Array, Array), Exception> {
        super::conv_fin_res_norm(h, u, d, base, w, eps)
    }

    /// `(q, kn, vn, qw, kw, ck, cv, pos)`; see `attention`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        q: &Array,
        kn: &Array,
        vn: &Array,
        qw: &Array,
        kw: &Array,
        ck: &Array,
        cv: &Array,
        pos: &Array,
        heads: Heads,
    ) -> Result<Array, Exception> {
        super::attention(&super::AttnInputs {
            q,
            kn,
            vn,
            qw,
            kw,
            ck,
            cv,
            pos,
            heads,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ctx_commit(
        kv: &Array,
        kw: &Array,
        pos: &Array,
        ck: &Array,
        cv: &Array,
        layer: i32,
        nl: i32,
        heads: Heads,
    ) -> Result<(Array, Array), Exception> {
        let commit = super::CommitLayer {
            index: layer,
            count: nl,
        };
        super::ctx_commit(kv, kw, pos, ck, cv, commit, heads)
    }

    pub fn top16(src: &Array) -> Result<Option<(Array, Array)>, Exception> {
        super::top16(src)
    }

    pub fn lattice_walk(
        ci: &Array,
        cv: &Array,
        sel: &Array,
        pred: (&Array, &Array, &Array),
        succ: (&Array, &Array, &Array),
        anchor: &Array,
    ) -> Result<Array, Exception> {
        super::lattice_walk(ci, cv, sel, pred, succ, anchor)
    }
}
