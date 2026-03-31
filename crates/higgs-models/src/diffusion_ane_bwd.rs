//! ANE backward MIL generators for diffusion LM (Qwen3-0.6B MDLM) training.
//!
//! Ported from nanobot-rs ane_mil.rs, adapted for:
//! - **Bidirectional attention** (no causal mask)
//! - **QK normalization** (Qwen3-specific per-head RMSNorm after Q/K projections)
//! - **GQA 2:1** (16 Q heads, 8 KV heads)
//! - **Row-major [seq, dim]** training loop (converted to channel-first [1,C,1,S] at ANE boundary)
//!
//! Three backward kernels, each a single ANE dispatch:
//!   1. `gen_rmsnorm_bwd` — RMSNorm backward (used twice per layer: input_norm + post_attn_norm)
//!   2. `gen_fused_ffn_bwd` — W2^T + SiLU backward + W1^T + W3^T in one dispatch
//!   3. `gen_fused_attn_gqa_bwd` — Wo^T + SDPA backward + QK-norm backward + RoPE backward + QKV^T

#![cfg(feature = "ane")]

use std::fmt::Write;
use crate::ane_mil::{MIL_HEADER, ANE_MIN_SPATIAL, FusedMil};

// ---------------------------------------------------------------------------
// 1. RMSNorm backward
// ---------------------------------------------------------------------------

/// Generate ANE kernel for RMSNorm backward.
///
/// Math: `dx = rrms * (w * dy - x * mean(dy * x * w) * rrms^2)`
/// where `rrms = (mean(x^2) + eps)^(-0.5)`
///
/// Input:  `[1, 2*dim, 1, seq]` fp32 — dy[dim] | x[dim] concatenated on channel axis
/// Output: `[1, dim, 1, seq]` fp32 — dx
/// Weight: `rms_w.bin` `[1, dim, 1, 1]` fp16 — RMSNorm gamma
pub fn gen_rmsnorm_bwd(dim: usize, seq_len: usize, eps: f64) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let in_ch = 2 * dim;

    let mut m = String::with_capacity(4096);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{in_ch},1,{seq}]> input) {{");

    // Constants
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [1]> ch_ax = const()[name=string(\"chax\"), val=tensor<int32, [1]>([1])];");
    let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
    let _ = writeln!(m, "        fp16 eps_v = const()[name=string(\"epsv\"), val=fp16({eps})];");
    let _ = writeln!(m, "        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];");

    // RMSNorm weight
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,1]> w = const()[name=string(\"w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_w.bin\"), offset=uint64(64)))];");

    // Cast + slice dy and x
    let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> ih = cast(dtype=to16,x=input)[name=string(\"cin\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> dy_begin = const()[name=string(\"dyb\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> dy_end = const()[name=string(\"dye\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> x_begin = const()[name=string(\"xb\"), val=tensor<int32, [4]>([0,{dim},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> x_end = const()[name=string(\"xe\"), val=tensor<int32, [4]>([1,{in_ch},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dy = slice_by_index(x=ih,begin=dy_begin,end=dy_end)[name=string(\"dy\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> x = slice_by_index(x=ih,begin=x_begin,end=x_end)[name=string(\"x\")];");

    // Step 1: rrms = pow(reduce_mean(x*x, ch) + eps, -0.5)
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> sq = mul(x=x,y=x)[name=string(\"sq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> ms = reduce_mean(x=sq,axes=ch_ax,keep_dims=kd)[name=string(\"ms\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> me = add(x=ms,y=eps_v)[name=string(\"me\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rr = pow(x=me,y=nhalf)[name=string(\"rr\")];");

    // Step 2: dot = reduce_mean(dy*x*w, ch) * rr * rr
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxw = mul(x=dy,y=x)[name=string(\"dxw\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxww = mul(x=dxw,y=w)[name=string(\"dxww\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> dot_m = reduce_mean(x=dxww,axes=ch_ax,keep_dims=kd)[name=string(\"dotm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> dot_r1 = mul(x=dot_m,y=rr)[name=string(\"dr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> dot = mul(x=dot_r1,y=rr)[name=string(\"dot\")];");

    // Step 3: dx = rr * (w*dy - x*dot)
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> wdy = mul(x=w,y=dy)[name=string(\"wdy\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xdot = mul(x=x,y=dot)[name=string(\"xdot\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> diff = sub(x=wdy,y=xdot)[name=string(\"diff\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxh = mul(x=rr,y=diff)[name=string(\"dxh\")];");

    // Cast back to fp32
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> out = cast(dtype=to32,x=dxh)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec!["@model_path/weights/rms_w.bin".into()],
        input_bytes: in_ch * seq * 4,
        output_bytes: dim * seq * 4,
    }
}

// ---------------------------------------------------------------------------
// 2. Fused FFN backward
// ---------------------------------------------------------------------------

/// Generate ANE kernel for fused FFN backward: W2^T + SiLU backward + W1^T + W3^T.
///
/// Math:
///   dsilu = W2^T @ d_ffn
///   dh3 = dsilu * silu(h1)            (up branch gradient)
///   dh1 = dsilu * h3 * silu_deriv(h1) (gate branch gradient)
///   dx = W1^T @ dh1 + W3^T @ dh3     (input gradient for FFN block)
///
/// Input:  `[1, dim+2*inter, 1, seq]` fp32 — d_ffn[dim] | gate_pre_silu[inter] | up_out[inter]
/// Output: `[1, dim+inter, 1, seq]` fp32 — dx[dim] | dsilu[inter]
///
/// Weights (3 BLOBFILEs):
///   - w2t.bin: W2^T (down_proj transposed) `[1,1,inter,dim]` fp16
///   - w1t.bin: W1^T (gate_proj transposed) `[1,1,dim,inter]` fp16
///   - w3t.bin: W3^T (up_proj transposed)   `[1,1,dim,inter]` fp16
///
/// For Qwen3-0.6B: dim=1024, inter=3072. Total BLOBFILE ~21MB fp16, fits in ANE SRAM.
pub fn gen_fused_ffn_bwd(
    dim: usize,
    inter: usize,
    seq_len: usize,
) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let in_ch = dim + 2 * inter;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{in_ch},1,{seq}]> x) {{");

    // Constants
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        fp16 one_v = const()[name=string(\"onev\"), val=fp16(1.0)];");

    // Weight BLOBFILEs — W^T forms for backward pass
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{dim}]> W2t = const()[name=string(\"W2t\"), val=tensor<fp16, [1,1,{inter},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/w2t.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{inter}]> W1t = const()[name=string(\"W1t\"), val=tensor<fp16, [1,1,{dim},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/w1t.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{inter}]> W3t = const()[name=string(\"W3t\"), val=tensor<fp16, [1,1,{dim},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/w3t.bin\"), offset=uint64(64)))];");

    // Cast input + slice into 3 blocks
    let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // d_ffn [dim, seq]
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxf = slice_by_size(x=xh,begin=b0,size=sd)[name=string(\"dxf\")];");

    // gate_pre_silu (h1) [inter, seq]
    let _ = writeln!(m, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,{dim},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sh = const()[name=string(\"sh\"), val=tensor<int32, [4]>([1,{inter},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> h1 = slice_by_size(x=xh,begin=b1,size=sh)[name=string(\"h1\")];");

    // up_out (h3) [inter, seq]
    let off_h3 = dim + inter;
    let _ = writeln!(m, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,{off_h3},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> h3 = slice_by_size(x=xh,begin=b3,size=sh)[name=string(\"h3\")];");

    // Step 1: dsilu = W2^T @ d_ffn (matmul in [1,1,M,K] @ [1,1,K,N] form)
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},{seq},1]> dxf_t = transpose(perm=pm,x=dxf)[name=string(\"dxft\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rdx = const()[name=string(\"rdx\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dxfm = reshape(shape=rdx,x=dxf_t)[name=string(\"dxfm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{seq}]> dsm = matmul(transpose_x=bF,transpose_y=bF,x=W2t,y=dxfm)[name=string(\"dsm\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rds = const()[name=string(\"rds\"), val=tensor<int32, [4]>([1,{inter},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> dsilu = reshape(shape=rds,x=dsm)[name=string(\"dsilu\")];");

    // Step 2: SiLU backward
    // sig = sigmoid(h1), silu_val = h1 * sig
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> sig = sigmoid(x=h1)[name=string(\"sig\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> silu = mul(x=h1,y=sig)[name=string(\"silu\")];");
    // dh3 = dsilu * silu(h1)
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> dh3 = mul(x=dsilu,y=silu)[name=string(\"dh3\")];");
    // silu_deriv = sig * (1 + h1 * (1 - sig))
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> omsig = sub(x=one_v,y=sig)[name=string(\"oms\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> h1oms = mul(x=h1,y=omsig)[name=string(\"h1oms\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> opl = add(x=one_v,y=h1oms)[name=string(\"opl\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> sd1 = mul(x=sig,y=opl)[name=string(\"sd1\")];");
    // dh1 = dsilu * h3 * silu_deriv
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> dsh3 = mul(x=dsilu,y=h3)[name=string(\"dsh3\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> dh1 = mul(x=dsh3,y=sd1)[name=string(\"dh1\")];");

    // Step 3: dx = W1^T @ dh1 + W3^T @ dh3
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},{seq},1]> dh1_t = transpose(perm=pm,x=dh1)[name=string(\"dh1t\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rdh = const()[name=string(\"rdh\"), val=tensor<int32, [4]>([1,1,{inter},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{seq}]> dh1m = reshape(shape=rdh,x=dh1_t)[name=string(\"dh1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx1m = matmul(transpose_x=bF,transpose_y=bF,x=W1t,y=dh1m)[name=string(\"dx1m\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rdout = const()[name=string(\"rdout\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx1 = reshape(shape=rdout,x=dx1m)[name=string(\"dx1\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,{inter},{seq},1]> dh3_t = transpose(perm=pm,x=dh3)[name=string(\"dh3t\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{seq}]> dh3m = reshape(shape=rdh,x=dh3_t)[name=string(\"dh3m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx3m = matmul(transpose_x=bF,transpose_y=bF,x=W3t,y=dh3m)[name=string(\"dx3m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx3 = reshape(shape=rdout,x=dx3m)[name=string(\"dx3\")];");

    // Sum dx1 + dx3
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxs = add(x=dx1,y=dx3)[name=string(\"dxs\")];");

    // Output: dx[dim] | dsilu[inter] concatenated on channel axis
    let out_ch = dim + inter;
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> cat = concat(values=(dxs,dsilu),axis=cax,interleave=bF)[name=string(\"cat\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out = cast(dtype=to32,x=cat)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/w2t.bin".into(),
            "@model_path/weights/w1t.bin".into(),
            "@model_path/weights/w3t.bin".into(),
        ],
        input_bytes: in_ch * seq * 4,
        output_bytes: out_ch * seq * 4,
    }
}

// ---------------------------------------------------------------------------
// 3. Fused GQA attention backward (with QK-norm, bidirectional)
// ---------------------------------------------------------------------------

/// Generate ANE kernel for attention backward (Wo^T + SDPA + RoPE inverse).
///
/// Uses MHA-style SDPA (K/V already expanded to full head count by caller on CPU).
/// This avoids the GQA batch reshape that the ANE compiler rejects.
///
/// Outputs dQ_post_rope, dK_post_rope, dV for CPU to finish QK-norm backward
/// and GQA reduction, then pass to gen_qkvt_bwd for the projection backward.
///
/// Input:  `[1, in_ch, 1, seq]` fp32
///   where in_ch = dim + 3*attn_dim  (dx2 | Q_rot | K_exp | V_exp)
///   K_exp, V_exp are KV heads repeated hpg times → full attn_dim width.
///
/// Output: `[1, 3*attn_dim, 1, seq]` fp32 — dQ | dK_exp | dV_exp (all full-head-count)
///   Caller must reduce dK_exp/dV_exp back to kv_dim by summing groups of hpg heads.
///
/// Weights (3 BLOBFILEs): Wo^T, rope_cos, rope_sin
pub fn gen_fused_attn_gqa_bwd(
    dim: usize,
    heads: usize,
    _kv_heads: usize,
    hd: usize,
    seq_len: usize,
    _eps: f64,
) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let half_hd = hd / 2;
    let attn_dim = heads * hd;
    let sc = 1.0 / (hd as f64).sqrt();
    // Input: dx2[dim] | Q_rot[attn_dim] | K_exp[attn_dim] | V_exp[attn_dim]
    let in_ch = dim + 3 * attn_dim;

    let mut m = String::with_capacity(16384);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{in_ch},1,{seq}]> input) {{");

    // --- Constants ---
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
    let _ = writeln!(m, "        fp16 sc_v = const()[name=string(\"scv\"), val=fp16({sc})];");
    // ANE reduce_sum workaround
    let _ = writeln!(m, "        fp16 seq_v = const()[name=string(\"seqv\"), val=fp16({seq}.0)];");
    let _ = writeln!(m, "        tensor<int32, [1]> last_ax = const()[name=string(\"lax\"), val=tensor<int32, [1]>([3])];");

    // --- Weight BLOBFILEs (only Wo^T and RoPE) ---
    let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{dim}]> Wot = const()[name=string(\"Wot\"), val=tensor<fp16, [1,1,{attn_dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wot.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rcos\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rsin\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");

    // --- Cast + slice inputs ---
    let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> ih = cast(dtype=to16,x=input)[name=string(\"cin\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sdim = const()[name=string(\"sdim\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dx2 = slice_by_size(x=ih,begin=b0,size=sdim)[name=string(\"dx2\")];");

    let _ = writeln!(m, "        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,{dim},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sad = const()[name=string(\"sad\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> q_rot = slice_by_size(x=ih,begin=bq,size=sad)[name=string(\"qr\")];");

    let off_k = dim + attn_dim;
    let _ = writeln!(m, "        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,{off_k},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> k_exp = slice_by_size(x=ih,begin=bk,size=sad)[name=string(\"ke\")];");

    let off_v = dim + 2 * attn_dim;
    let _ = writeln!(m, "        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,{off_v},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> v_exp = slice_by_size(x=ih,begin=bv,size=sad)[name=string(\"ve\")];");

    // === Phase 1: Wo^T backward ===
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},{seq},1]> dx2_t = transpose(perm=pm,x=dx2)[name=string(\"dx2t\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rdd = const()[name=string(\"rdd\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> dx2m = reshape(shape=rdd,x=dx2_t)[name=string(\"dx2m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> dam = matmul(transpose_x=bF,transpose_y=bF,x=Wot,y=dx2m)[name=string(\"dam\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rad = const()[name=string(\"rad\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> d_attn = reshape(shape=rad,x=dam)[name=string(\"da\")];");

    // === Phase 2: Reshape to MHA head form [1, heads, seq, hd] ===
    let _ = writeln!(m, "        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");

    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> da4 = reshape(shape=rh,x=d_attn)[name=string(\"da4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dO = transpose(perm=pm,x=da4)[name=string(\"dO\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> qr4 = reshape(shape=rh,x=q_rot)[name=string(\"qr4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> Q = transpose(perm=pm,x=qr4)[name=string(\"Q\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ke4 = reshape(shape=rh,x=k_exp)[name=string(\"ke4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> K = transpose(perm=pm,x=ke4)[name=string(\"K\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> ve4 = reshape(shape=rh,x=v_exp)[name=string(\"ve4\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> V = transpose(perm=pm,x=ve4)[name=string(\"V\")];");

    // === Phase 3: SDPA backward (MHA, bidirectional, batch=1) ===
    // Recompute attention: aw = softmax(scale * Q @ K^T)
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> Qs = mul(x=Q,y=sc_v)[name=string(\"Qs\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> scores = matmul(transpose_x=bF,transpose_y=bT,x=Qs,y=K)[name=string(\"sc\")];");

    // Softmax (bidirectional — no mask)
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> s_max = reduce_max(x=scores,axes=last_ax,keep_dims=kd)[name=string(\"smax\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> s_shift = sub(x=scores,y=s_max)[name=string(\"ssh\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> s_exp = exp(x=s_shift)[name=string(\"sexp\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> s_mean = reduce_mean(x=s_exp,axes=last_ax,keep_dims=kd)[name=string(\"smean\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> s_sum = mul(x=s_mean,y=seq_v)[name=string(\"ssum\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> aw = real_div(x=s_exp,y=s_sum)[name=string(\"aw\")];");

    // dV = A^T @ dO → [1, heads, hd, seq] → transpose → [1, heads, seq, hd]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> at_do = matmul(transpose_x=bT,transpose_y=bF,x=aw,y=dO)[name=string(\"atdo\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dV = transpose(perm=pm,x=at_do)[name=string(\"dV\")];");

    // dP = dO @ V^T
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dP = matmul(transpose_x=bF,transpose_y=bT,x=dO,y=V)[name=string(\"dP\")];");

    // Softmax backward: dS = aw * (dP - rowsum(dP*aw))
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp_aw = mul(x=dP,y=aw)[name=string(\"dpaw\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> rs_mean = reduce_mean(x=dp_aw,axes=last_ax,keep_dims=kd)[name=string(\"rsmn\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> rs = mul(x=rs_mean,y=seq_v)[name=string(\"rs\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dp_sub = sub(x=dP,y=rs)[name=string(\"dpsb\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{seq}]> dS = mul(x=aw,y=dp_sub)[name=string(\"dS\")];");

    // dQ = scale * dS @ K
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dQ_raw = matmul(transpose_x=bF,transpose_y=bF,x=dS,y=K)[name=string(\"dQr\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dQ = mul(x=dQ_raw,y=sc_v)[name=string(\"dQ\")];");

    // dK = scale * dS^T @ Q
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dK_raw = matmul(transpose_x=bT,transpose_y=bF,x=dS,y=Q)[name=string(\"dKr\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dK = mul(x=dK_raw,y=sc_v)[name=string(\"dK\")];");

    // === Phase 4: RoPE backward (all heads, batch=1) ===
    let _ = writeln!(m, "        tensor<int32, [4]> h1b = const()[name=string(\"h1b\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> h1s = const()[name=string(\"h1s\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> h2b = const()[name=string(\"h2b\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
    let _ = writeln!(m, "        int32 last_d = const()[name=string(\"ld\"), val=int32(3)];");

    // RoPE inverse for dQ
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr1 = slice_by_size(x=dQ,begin=h1b,size=h1s)[name=string(\"dqr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dqr2 = slice_by_size(x=dQ,begin=h2b,size=h1s)[name=string(\"dqr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1c = mul(x=dqr1,y=rope_cos)[name=string(\"dq1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2s = mul(x=dqr2,y=rope_sin)[name=string(\"dq2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq_lo = add(x=dq1c,y=dq2s)[name=string(\"dqlo\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq2c = mul(x=dqr2,y=rope_cos)[name=string(\"dq2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq1s = mul(x=dqr1,y=rope_sin)[name=string(\"dq1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dq_hi = sub(x=dq2c,y=dq1s)[name=string(\"dqhi\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dq_pr = concat(values=(dq_lo,dq_hi),axis=last_d,interleave=bF)[name=string(\"dqpr\")];");

    // RoPE inverse for dK (also full heads since K was expanded)
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dkr1 = slice_by_size(x=dK,begin=h1b,size=h1s)[name=string(\"dkr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dkr2 = slice_by_size(x=dK,begin=h2b,size=h1s)[name=string(\"dkr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk1c = mul(x=dkr1,y=rope_cos)[name=string(\"dk1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk2s = mul(x=dkr2,y=rope_sin)[name=string(\"dk2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk_lo = add(x=dk1c,y=dk2s)[name=string(\"dklo\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk2c = mul(x=dkr2,y=rope_cos)[name=string(\"dk2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk1s = mul(x=dkr1,y=rope_sin)[name=string(\"dk1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> dk_hi = sub(x=dk2c,y=dk1s)[name=string(\"dkhi\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> dk_pr = concat(values=(dk_lo,dk_hi),axis=last_d,interleave=bF)[name=string(\"dkpr\")];");

    // === Output: flatten to channel-first [1, C, 1, S] and concat ===
    // dQ: [1, H, S, hd] → transpose → [1, H, hd, S] → reshape → [1, attn_dim, 1, S]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dq_ch = transpose(perm=pm,x=dq_pr)[name=string(\"dqch\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> raq = const()[name=string(\"raq\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> dq_flat = reshape(shape=raq,x=dq_ch)[name=string(\"dqfl\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dk_ch = transpose(perm=pm,x=dk_pr)[name=string(\"dkch\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> dk_flat = reshape(shape=raq,x=dk_ch)[name=string(\"dkfl\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> dv_ch = transpose(perm=pm,x=dV)[name=string(\"dvch\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> dv_flat = reshape(shape=raq,x=dv_ch)[name=string(\"dvfl\")];");

    let out_ch = 3 * attn_dim;
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> cat = concat(values=(dq_flat,dk_flat,dv_flat),axis=cax,interleave=bF)[name=string(\"cat\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out = cast(dtype=to32,x=cat)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/wot.bin".into(),
            "@model_path/weights/rope_cos.bin".into(),
            "@model_path/weights/rope_sin.bin".into(),
        ],
        input_bytes: in_ch * seq * 4,
        output_bytes: out_ch * seq * 4,
    }
}

// ---------------------------------------------------------------------------
// 4. QKV^T projection backward (second dispatch after QK-norm backward on CPU)
// ---------------------------------------------------------------------------

/// Generate ANE kernel for QKV^T projection backward.
///
/// Takes dQ, dK, dV (post QK-norm backward) and computes dx = Wq^T@dQ + Wk^T@dK + Wv^T@dV.
///
/// Input:  `[1, attn_dim + 2*kv_dim, 1, seq]` fp32 — dQ[attn_dim] | dK[kv_dim] | dV[kv_dim]
/// Output: `[1, dim, 1, seq]` fp32 — dx
///
/// Weights (3 BLOBFILEs): Wq^T, Wk^T, Wv^T
pub fn gen_qkvt_bwd(
    dim: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    seq_len: usize,
) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let attn_dim = heads * hd;
    let kv_dim = kv_heads * hd;
    let in_ch = attn_dim + 2 * kv_dim;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{in_ch},1,{seq}]> input) {{");

    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");

    // Weight BLOBFILEs
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{attn_dim}]> Wqt = const()[name=string(\"Wqt\"), val=tensor<fp16, [1,1,{dim},{attn_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wqt.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wkt = const()[name=string(\"Wkt\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wkt.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wvt = const()[name=string(\"Wvt\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wvt.bin\"), offset=uint64(64)))];");

    // Cast + slice
    let _ = writeln!(m, "        tensor<fp16, [1,{in_ch},1,{seq}]> ih = cast(dtype=to16,x=input)[name=string(\"cin\")];");

    // dQ [attn_dim, seq]
    let _ = writeln!(m, "        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sq = const()[name=string(\"sq\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> dq = slice_by_size(x=ih,begin=b0,size=sq)[name=string(\"dq\")];");

    // dK [kv_dim, seq]
    let _ = writeln!(m, "        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,{attn_dim},0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> sk = const()[name=string(\"sk\"), val=tensor<int32, [4]>([1,{kv_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> dk = slice_by_size(x=ih,begin=bk,size=sk)[name=string(\"dk\")];");

    // dV [kv_dim, seq]
    let off_v = attn_dim + kv_dim;
    let _ = writeln!(m, "        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,{off_v},0,0])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> dv = slice_by_size(x=ih,begin=bv,size=sk)[name=string(\"dv\")];");

    // Three matmuls: Wq^T @ dQ, Wk^T @ dK, Wv^T @ dV — all produce [1,1,dim,seq]
    // dQ → [1,1,attn_dim,seq]
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},{seq},1]> dq_t = transpose(perm=pm,x=dq)[name=string(\"dqt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rqa = const()[name=string(\"rqa\"), val=tensor<int32, [4]>([1,1,{attn_dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> dqm = reshape(shape=rqa,x=dq_t)[name=string(\"dqm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> xq = matmul(transpose_x=bF,transpose_y=bF,x=Wqt,y=dqm)[name=string(\"xq\")];");

    // dK → [1,1,kv_dim,seq]
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},{seq},1]> dk_t = transpose(perm=pm,x=dk)[name=string(\"dkt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rkva = const()[name=string(\"rkva\"), val=tensor<int32, [4]>([1,1,{kv_dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> dkm = reshape(shape=rkva,x=dk_t)[name=string(\"dkm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> xk = matmul(transpose_x=bF,transpose_y=bF,x=Wkt,y=dkm)[name=string(\"xk\")];");

    // dV → [1,1,kv_dim,seq]
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},{seq},1]> dv_t = transpose(perm=pm,x=dv)[name=string(\"dvt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> dvm = reshape(shape=rkva,x=dv_t)[name=string(\"dvm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> xv = matmul(transpose_x=bF,transpose_y=bF,x=Wvt,y=dvm)[name=string(\"xv\")];");

    // Sum: dx = xq + xk + xv
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> s1 = add(x=xq,y=xk)[name=string(\"s1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> s2 = add(x=s1,y=xv)[name=string(\"s2\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rout = const()[name=string(\"rout\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> dxh = reshape(shape=rout,x=s2)[name=string(\"dxh\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> out = cast(dtype=to32,x=dxh)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/wqt.bin".into(),
            "@model_path/weights/wkt.bin".into(),
            "@model_path/weights/wvt.bin".into(),
        ],
        input_bytes: in_ch * seq * 4,
        output_bytes: dim * seq * 4,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_bwd_mil_generation() {
        let mil = gen_rmsnorm_bwd(1024, 64, 1e-6);
        assert!(mil.mil_text.contains("func main<ios18>"));
        assert!(mil.mil_text.contains("rms_w.bin"));
        assert_eq!(mil.weight_names.len(), 1);
        assert_eq!(mil.input_bytes, 2 * 1024 * 64 * 4); // 2*dim*seq*f32
        assert_eq!(mil.output_bytes, 1024 * 64 * 4);
        eprintln!("RMSNorm bwd MIL: {} bytes", mil.mil_text.len());
    }

    #[test]
    fn test_ffn_bwd_mil_generation() {
        let mil = gen_fused_ffn_bwd(1024, 3072, 64);
        assert!(mil.mil_text.contains("func main<ios18>"));
        assert_eq!(mil.weight_names.len(), 3);
        let in_ch = 1024 + 2 * 3072;
        assert_eq!(mil.input_bytes, in_ch * 64 * 4);
        let out_ch = 1024 + 3072;
        assert_eq!(mil.output_bytes, out_ch * 64 * 4);
        eprintln!("FFN bwd MIL: {} bytes, 3 BLOBFILEs", mil.mil_text.len());
    }

    #[test]
    fn test_attn_gqa_bwd_mil_generation() {
        let mil = gen_fused_attn_gqa_bwd(1024, 16, 8, 128, 64, 1e-6);
        assert!(mil.mil_text.contains("func main<ios18>"));
        // New MHA-style: only Wo^T + rope_cos + rope_sin (QKV^T is a separate dispatch)
        assert_eq!(mil.weight_names.len(), 3);
        let attn_dim = 16 * 128; // = 2048
        // in_ch = dim + 3*attn_dim (dx2 | Q_rot | K_exp | V_exp, K/V expanded to full heads)
        let in_ch = 1024 + 3 * attn_dim;
        assert_eq!(mil.input_bytes, in_ch * 64 * 4);
        // out_ch = 3*attn_dim (dQ | dK_exp | dV_exp, all full heads)
        let out_ch = 3 * attn_dim;
        assert_eq!(mil.output_bytes, out_ch * 64 * 4);
        eprintln!("Attn GQA bwd MIL: {} bytes, 3 BLOBFILEs (MHA-style)", mil.mil_text.len());
    }

    #[test]
    fn test_qkvt_bwd_mil_generation() {
        let mil = gen_qkvt_bwd(1024, 16, 8, 128, 64);
        assert!(mil.mil_text.contains("func main<ios18>"));
        assert_eq!(mil.weight_names.len(), 3);
        let attn_dim = 16 * 128;
        let kv_dim = 8 * 128;
        let in_ch = attn_dim + 2 * kv_dim;
        assert_eq!(mil.input_bytes, in_ch * 64 * 4);
        assert_eq!(mil.output_bytes, 1024 * 64 * 4);
        eprintln!("QKV^T bwd MIL: {} bytes, 3 BLOBFILEs", mil.mil_text.len());
    }

    /// Compile and run RMSNorm backward on ANE. Requires `ane` feature.
    #[test]
    #[ignore]
    fn test_rmsnorm_bwd_ane_compile() {
        use crate::ane_bridge::{ane_init, build_weight_blob, AneKernel};
        ane_init();

        let dim = 1024;
        let seq = 64;
        let eps = 1e-6;

        let mil = gen_rmsnorm_bwd(dim, seq, eps);

        // Create dummy RMSNorm weight (all ones)
        let w = vec![1.0f32; dim];
        let w_blob = build_weight_blob(&w, 1, dim);

        let kernel = AneKernel::compile_multi_weights(
            &mil.mil_text,
            &["@model_path/weights/rms_w.bin"],
            &[&w_blob],
            &[mil.input_bytes],
            &[mil.output_bytes],
        ).expect("Failed to compile RMSNorm bwd");

        // Create test input: dy=1.0, x=random-ish
        let mut input = vec![0.0f32; 2 * dim * seq];
        for s in 0..seq {
            for d in 0..dim {
                input[s * 1 + d * seq] = 1.0; // dy = 1 (note: channel-first!)
                // Whoops — channel-first layout: [1, 2*dim, 1, seq]
                // So index is: ch * seq + s
                input[d * seq + s] = 1.0; // dy[d, s] = 1.0
                input[(dim + d) * seq + s] = ((d * 7 + s * 13) % 100) as f32 / 100.0; // x
            }
        }

        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &input_bytes);
        kernel.eval().expect("ANE eval failed");

        let mut output_bytes = vec![0u8; dim * seq * 4];
        kernel.read_output(0, &mut output_bytes);
        let output: Vec<f32> = output_bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Check output is finite and non-zero
        let nonzero = output.iter().filter(|v| v.abs() > 1e-6).count();
        eprintln!("RMSNorm bwd output: {nonzero}/{} non-zero values", output.len());
        assert!(nonzero > output.len() / 2, "Too many zero outputs");
        assert!(output.iter().all(|v| v.is_finite()), "Non-finite output");
        eprintln!("PASS: RMSNorm backward compiles and runs on ANE");
    }

    /// Compile and run FFN backward on ANE.
    #[test]
    #[ignore]
    fn test_ffn_bwd_ane_compile() {
        use crate::ane_bridge::{ane_init, build_weight_blob_transposed, AneKernel};
        ane_init();

        let dim = 1024;
        let inter = 3072;
        let seq = 64;

        let mil = gen_fused_ffn_bwd(dim, inter, seq);

        // Create dummy transposed weights
        let w2t = vec![0.01f32; inter * dim];
        let w1t = vec![0.01f32; dim * inter];
        let w3t = vec![0.01f32; dim * inter];
        let w2t_blob = build_weight_blob_transposed(&w2t, inter, dim);
        let w1t_blob = build_weight_blob_transposed(&w1t, dim, inter);
        let w3t_blob = build_weight_blob_transposed(&w3t, dim, inter);

        let kernel = AneKernel::compile_multi_weights(
            &mil.mil_text,
            &[
                "@model_path/weights/w2t.bin",
                "@model_path/weights/w1t.bin",
                "@model_path/weights/w3t.bin",
            ],
            &[&w2t_blob, &w1t_blob, &w3t_blob],
            &[mil.input_bytes],
            &[mil.output_bytes],
        ).expect("Failed to compile FFN bwd");

        // Input: d_ffn[dim] | gate_pre_silu[inter] | up_out[inter]
        let in_ch = dim + 2 * inter;
        let mut input = vec![0.0f32; in_ch * seq];
        for s in 0..seq {
            for d in 0..dim {
                input[d * seq + s] = 0.1; // d_ffn
            }
            for d in 0..inter {
                input[(dim + d) * seq + s] = ((d * 3 + s * 7) % 100) as f32 / 100.0; // gate
                input[(dim + inter + d) * seq + s] = 0.5; // up
            }
        }

        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &input_bytes);
        kernel.eval().expect("ANE eval failed");

        let out_ch = dim + inter;
        let mut output_bytes = vec![0u8; out_ch * seq * 4];
        kernel.read_output(0, &mut output_bytes);
        let output: Vec<f32> = output_bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let nonzero = output.iter().filter(|v| v.abs() > 1e-6).count();
        eprintln!("FFN bwd output: {nonzero}/{} non-zero values", output.len());
        assert!(nonzero > output.len() / 4, "Too many zero outputs");
        assert!(output.iter().all(|v| v.is_finite()), "Non-finite output");
        eprintln!("PASS: FFN backward compiles and runs on ANE");
    }

    /// Compile and run GQA attention backward on ANE.
    #[test]
    #[ignore]
    fn test_attn_gqa_bwd_ane_compile() {
        use crate::ane_bridge::{ane_init, build_weight_blob_transposed, build_weight_blob, AneKernel};
        ane_init();

        let dim = 1024;
        let heads = 16;
        let kv_heads = 8;
        let hd = 128;
        let half_hd = hd / 2;
        let seq = 64;
        let attn_dim = heads * hd;
        let kv_dim = kv_heads * hd;

        let mil = gen_fused_attn_gqa_bwd(dim, heads, kv_heads, hd, seq, 1e-6);

        // MHA-style: only Wo^T + rope tables (3 blobs). QKV^T is a separate dispatch.
        let wot_blob = build_weight_blob_transposed(&vec![0.01f32; attn_dim * dim], attn_dim, dim);

        // RoPE tables: shape [1,1,seq,half_hd] → rows=seq, cols=half_hd
        let mut cos_data = vec![0.0f32; seq * half_hd];
        let mut sin_data = vec![0.0f32; seq * half_hd];
        for pos in 0..seq {
            for d in 0..half_hd {
                let angle = pos as f32 * (1.0 / 1000000.0f32.powf(2.0 * d as f32 / hd as f32));
                cos_data[pos * half_hd + d] = angle.cos();
                sin_data[pos * half_hd + d] = angle.sin();
            }
        }
        let cos_blob = build_weight_blob(&cos_data, seq, half_hd);
        let sin_blob = build_weight_blob(&sin_data, seq, half_hd);

        let kernel = AneKernel::compile_multi_weights(
            &mil.mil_text,
            &[
                "@model_path/weights/wot.bin",
                "@model_path/weights/rope_cos.bin",
                "@model_path/weights/rope_sin.bin",
            ],
            &[&wot_blob, &cos_blob, &sin_blob],
            &[mil.input_bytes],
            &[mil.output_bytes],
        ).expect("Failed to compile attn GQA bwd");

        // Input: dx2[dim] | Q_rot[attn_dim] | K_exp[attn_dim] | V_exp[attn_dim]
        // K_exp/V_exp are KV heads repeated hpg (=heads/kv_heads=2) times → full attn_dim
        let in_ch = dim + 3 * attn_dim;
        let mut input = vec![0.0f32; in_ch * seq];
        for s in 0..seq {
            for d in 0..dim { input[d * seq + s] = 0.01; } // dx2
            for d in 0..attn_dim { input[(dim + d) * seq + s] = ((d + s) % 50) as f32 / 500.0; } // Q_rot
            for d in 0..attn_dim { input[(dim + attn_dim + d) * seq + s] = ((d * 3 + s) % 50) as f32 / 500.0; } // K_exp
            for d in 0..attn_dim { input[(dim + 2 * attn_dim + d) * seq + s] = 0.02; } // V_exp
        }

        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &input_bytes);
        kernel.eval().expect("ANE eval failed");

        let out_ch = 3 * attn_dim;
        let mut output_bytes = vec![0u8; out_ch * seq * 4];
        kernel.read_output(0, &mut output_bytes);
        let output: Vec<f32> = output_bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let nonzero = output.iter().filter(|v| v.abs() > 1e-6).count();
        let finite = output.iter().filter(|v| v.is_finite()).count();
        eprintln!("Attn GQA bwd output: {nonzero}/{} non-zero, {finite}/{} finite", output.len(), output.len());
        assert!(finite == output.len(), "Non-finite output");
        assert!(nonzero > output.len() / 4, "Too many zero outputs");
        eprintln!("PASS: Attention GQA backward compiles and runs on ANE");
    }

    /// Compile and run QKV^T projection backward on ANE.
    #[test]
    #[ignore]
    fn test_qkvt_bwd_ane_compile() {
        use crate::ane_bridge::{ane_init, build_weight_blob_transposed, AneKernel};
        ane_init();

        let dim = 1024;
        let heads = 16;
        let kv_heads = 8;
        let hd = 128;
        let seq = 64;
        let attn_dim = heads * hd;
        let kv_dim = kv_heads * hd;

        let mil = gen_qkvt_bwd(dim, heads, kv_heads, hd, seq);

        let wqt_blob = build_weight_blob_transposed(&vec![0.01f32; dim * attn_dim], dim, attn_dim);
        let wkt_blob = build_weight_blob_transposed(&vec![0.01f32; dim * kv_dim], dim, kv_dim);
        let wvt_blob = build_weight_blob_transposed(&vec![0.01f32; dim * kv_dim], dim, kv_dim);

        let kernel = AneKernel::compile_multi_weights(
            &mil.mil_text,
            &[
                "@model_path/weights/wqt.bin",
                "@model_path/weights/wkt.bin",
                "@model_path/weights/wvt.bin",
            ],
            &[&wqt_blob, &wkt_blob, &wvt_blob],
            &[mil.input_bytes],
            &[mil.output_bytes],
        ).expect("Failed to compile QKV^T bwd");

        let in_ch = attn_dim + 2 * kv_dim;
        let mut input = vec![0.0f32; in_ch * seq];
        for s in 0..seq {
            for d in 0..in_ch { input[d * seq + s] = 0.01; }
        }

        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &input_bytes);
        kernel.eval().expect("ANE eval failed");

        let mut output_bytes = vec![0u8; dim * seq * 4];
        kernel.read_output(0, &mut output_bytes);
        let output: Vec<f32> = output_bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let nonzero = output.iter().filter(|v| v.abs() > 1e-6).count();
        eprintln!("QKV^T bwd output: {nonzero}/{} non-zero values", output.len());
        assert!(nonzero > output.len() / 2, "Too many zero outputs");
        assert!(output.iter().all(|v| v.is_finite()), "Non-finite output");
        eprintln!("PASS: QKV^T backward compiles and runs on ANE");
    }
}
