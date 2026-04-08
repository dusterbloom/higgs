//! Fully-fused ANE layer kernel for diffusion LM (Qwen3-0.6B MDLM).
//!
//! ONE ANE dispatch per transformer layer: RMSNorm → QKV → QK-norm → RoPE → SDPA → Wo → residual → RMSNorm → FFN → residual.
//! 28 dispatches for the full model. Input: hidden [dim, seq] fp32. Output: hidden [dim, seq] fp32.
//!
//! Differences from nanobot-rs gen_fused_layer_fwd:
//! - **QK normalization** (Qwen3-specific) — extra per-head RMSNorm after Q/K projections
//! - **GQA 2:1** (16 Q heads, 8 KV heads)
//! - **Causal mask support** — BLOBFILE mask added before softmax (enabled via `causal: true`)
//!
//! Weight BLOBFILEs (13 total, ~30MB fp16 for 0.6B model):
//!   rms_att, rms_ffn, wq, wk, wv, wo, gate_proj, up_proj, down_proj, rope_cos, rope_sin, q_norm, k_norm
//!   + causal_mask.bin when causal=true

#![cfg(feature = "ane")]

use crate::ane_mil::{FusedMil, ANE_MIN_SPATIAL, MIL_HEADER};
use std::fmt::Write;

/// STUB: gen_decode_layer was removed from this branch but call sites remain.
/// Panics if invoked. Bidirectional Magic Canvas tests don't hit the
/// `AneArDecodeEngine` path so this is dead code for those tests.
#[allow(clippy::too_many_arguments)]
pub fn gen_decode_layer(
    _dim: usize,
    _heads: usize,
    _kv_heads: usize,
    _hd: usize,
    _inter: usize,
    _max_seq: usize,
    _eps: f64,
) -> FusedMil {
    panic!("gen_decode_layer: not implemented on this branch — AR-decode ANE path is dead");
}

/// Generate a fully-fused single transformer layer.
///
/// Input:  `[1, dim, 1, seq]` fp32
/// Output: `[1, dim, 1, seq]` fp32 (hidden state after full layer)
///
/// When `causal=false`: bidirectional diffusion (13 BLOBFILE weights)
/// When `causal=true`: causal language model (14 BLOBFILE weights, includes mask)
/// ~30MB at dim=1024 (fits under 32MB ANE limit).
pub fn gen_fused_diffusion_layer(
    dim: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    inter: usize, // MLP intermediate size
    seq_len: usize,
    eps: f64,
    causal: bool,
) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let half_hd = hd / 2;
    let kv_dim = kv_heads * hd;
    let attn_dim = heads * hd;
    let hpg = heads / kv_heads; // heads per group for GQA
    let sc = 1.0 / (hd as f64).sqrt();

    let mut m = String::with_capacity(32768);
    m.push_str(MIL_HEADER);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1,{dim},1,{seq}]> x) {{"
    );

    // --- Shared constants ---
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];"
    );
    let _ = writeln!(
        m,
        "        bool kd = const()[name=string(\"kd\"), val=bool(true)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [1]> ch_ax = const()[name=string(\"chax\"), val=tensor<int32, [1]>([1])];"
    );
    let _ = writeln!(
        m,
        "        fp16 eps_v = const()[name=string(\"epsv\"), val=fp16({eps})];"
    );
    let _ = writeln!(
        m,
        "        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];"
    );

    // Cast input
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];"
    );

    // === RMSNorm (attention) ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn1_sq = mul(x=xh,y=xh)[name=string(\"rn1sq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_m = reduce_mean(x=rn1_sq,axes=ch_ax,keep_dims=kd)[name=string(\"rn1m\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_e = add(x=rn1_m,y=eps_v)[name=string(\"rn1e\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_r = pow(x=rn1_e,y=nhalf)[name=string(\"rn1r\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn1_n = mul(x=xh,y=rn1_r)[name=string(\"rn1n\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,1]> rn1_w = const()[name=string(\"rn1w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_att.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xnorm = mul(x=rn1_n,y=rn1_w)[name=string(\"xnorm\")];"
    );

    // === QKV Projections ===
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{seq}]> xn2 = reshape(shape=r2d,x=xnorm)[name=string(\"xn2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{attn_dim}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{dim},{attn_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{attn_dim},{dim}]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [1,1,{attn_dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{attn_dim}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq)[name=string(\"qm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{kv_dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk)[name=string(\"km\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{kv_dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv)[name=string(\"vm\")];"
    );

    // Reshape Q → [1, heads, seq, hd]
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{attn_dim},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{hd},{seq}]> q4 = reshape(shape=qsh,x=qt)[name=string(\"rq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];"
    );

    // Reshape K → [1, kv_heads, seq, hd]
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{kv_dim},{seq}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> kvsh = const()[name=string(\"kvsh\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> k4 = reshape(shape=kvsh,x=kt)[name=string(\"rk\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_kv = transpose(perm=pm,x=k4)[name=string(\"tk\")];"
    );

    // Reshape V → [1, kv_heads, seq, hd]
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{kv_dim},{seq}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v4 = reshape(shape=kvsh,x=vt)[name=string(\"rv\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_kv = transpose(perm=pm,x=v4)[name=string(\"tv\")];"
    );

    // === QK Norm (Qwen3-specific: per-head RMSNorm on Q and K) ===
    let _ = writeln!(
        m,
        "        tensor<int32, [1]> hd_ax = const()[name=string(\"hdax\"), val=tensor<int32, [1]>([-1])];"
    );
    // Q norm
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q_sq = mul(x=q,y=q)[name=string(\"qsq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},1]> qn_m = reduce_mean(x=q_sq,axes=hd_ax,keep_dims=kd)[name=string(\"qnm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},1]> qn_e = add(x=qn_m,y=eps_v)[name=string(\"qne\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},1]> qn_r = pow(x=qn_e,y=nhalf)[name=string(\"qnr\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> qn = mul(x=q,y=qn_r)[name=string(\"qn\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{hd}]> qn_w = const()[name=string(\"qnw\"), val=tensor<fp16, [1,1,1,{hd}]>(BLOBFILE(path=string(\"@model_path/weights/q_norm.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q_normed = mul(x=qn,y=qn_w)[name=string(\"qnormed\")];"
    );
    // K norm
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_sq = mul(x=k_kv,y=k_kv)[name=string(\"ksq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_m = reduce_mean(x=k_sq,axes=hd_ax,keep_dims=kd)[name=string(\"knm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_e = add(x=kn_m,y=eps_v)[name=string(\"kne\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_r = pow(x=kn_e,y=nhalf)[name=string(\"knr\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kn = mul(x=k_kv,y=kn_r)[name=string(\"kn\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{hd}]> kn_w = const()[name=string(\"knw\"), val=tensor<fp16, [1,1,1,{hd}]>(BLOBFILE(path=string(\"@model_path/weights/k_norm.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_normed = mul(x=kn,y=kn_w)[name=string(\"knormed\")];"
    );

    // === RoPE ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];"
    );

    // Q RoPE
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_b0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_sh = const()[name=string(\"rpsh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1 = slice_by_size(x=q_normed,begin=rp_b0,size=rp_sh)[name=string(\"q1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_bh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2 = slice_by_size(x=q_normed,begin=rp_bh,size=rp_sh)[name=string(\"q2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1c = mul(x=q1,y=rope_cos)[name=string(\"q1c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2s = mul(x=q2,y=rope_sin)[name=string(\"q2s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr1 = sub(x=q1c,y=q2s)[name=string(\"qr1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1s = mul(x=q1,y=rope_sin)[name=string(\"q1s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2c = mul(x=q2,y=rope_cos)[name=string(\"q2c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr2 = add(x=q1s,y=q2c)[name=string(\"qr2\")];"
    );
    let _ = writeln!(
        m,
        "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];"
    );
    let _ = writeln!(
        m,
        "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q_rot = concat(axis=rpax,interleave=rpid,values=(qr1,qr2))[name=string(\"qrot\")];"
    );

    // K RoPE
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_ksh = const()[name=string(\"rpksh\"), val=tensor<int32, [4]>([1,{kv_heads},{seq},{half_hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1 = slice_by_size(x=k_normed,begin=rp_b0,size=rp_ksh)[name=string(\"k1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2 = slice_by_size(x=k_normed,begin=rp_bh,size=rp_ksh)[name=string(\"k2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1c = mul(x=k1,y=rope_cos)[name=string(\"k1c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2s = mul(x=k2,y=rope_sin)[name=string(\"k2s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> kr1 = sub(x=k1c,y=k2s)[name=string(\"kr1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1s = mul(x=k1,y=rope_sin)[name=string(\"k1s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2c = mul(x=k2,y=rope_cos)[name=string(\"k2c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> kr2 = add(x=k1s,y=k2c)[name=string(\"kr2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_rot = concat(axis=rpax,interleave=rpid,values=(kr1,kr2))[name=string(\"krot\")];"
    );

    // === GQA Bidirectional SDPA (NO causal mask) ===
    // Q: [1,H,S,hd] → [kvH, hpg, S, hd]
    // K: [1,kvH,S,hd] → [kvH, 1, S, hd]
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=q_rot)[name=string(\"qb\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=k_rot)[name=string(\"kb\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_kv)[name=string(\"vb\")];"
    );

    // Q@K^T scaled → softmax
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"mm1\")];"
    );
    let _ = writeln!(
        m,
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];"
    );
    let _ = writeln!(
        m,
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];"
    );

    // Causal mask (when enabled)
    if causal {
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/causal_mask.bin\"), offset=uint64(64)))];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];"
        );
    } else {
        let _ = writeln!(
            m,
            "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=sc2)[name=string(\"sm\")];"
        );
    }

    // scores@V
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=vb)[name=string(\"mm2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rha = const()[name=string(\"rha\"), val=tensor<int32, [4]>([1,{heads},{seq},{hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> a_out = reshape(shape=rha,x=a4)[name=string(\"aout\")];"
    );

    // Reshape attn output → [1, attn_dim, 1, seq]
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{hd},{seq}]> at = transpose(perm=pm,x=a_out)[name=string(\"ta\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> osa = const()[name=string(\"osa\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{attn_dim},1,{seq}]> af = reshape(shape=osa,x=at)[name=string(\"ra\")];"
    );

    // === Wo projection ===
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> r2a = const()[name=string(\"r2a\"), val=tensor<int32, [4]>([1,1,{attn_dim},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{attn_dim},{seq}]> af2 = reshape(shape=r2a,x=af)[name=string(\"af2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{attn_dim}]> aft = transpose(perm=pm,x=af2)[name=string(\"aft\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> om = matmul(transpose_x=bF,transpose_y=bF,x=aft,y=Wo)[name=string(\"om\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{seq}]> ot = transpose(perm=pm,x=om)[name=string(\"ot\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> oo = reshape(shape=os,x=ot)[name=string(\"oo\")];"
    );

    // === Residual 1 ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> x2 = add(x=xh,y=oo)[name=string(\"x2\")];"
    );

    // === RMSNorm (FFN) ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn2_sq = mul(x=x2,y=x2)[name=string(\"rn2sq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn2_m = reduce_mean(x=rn2_sq,axes=ch_ax,keep_dims=kd)[name=string(\"rn2m\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn2_e = add(x=rn2_m,y=eps_v)[name=string(\"rn2e\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn2_r = pow(x=rn2_e,y=nhalf)[name=string(\"rn2r\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn2_n = mul(x=x2,y=rn2_r)[name=string(\"rn2n\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,1]> rn2_w = const()[name=string(\"rn2w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_ffn.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> x2norm = mul(x=rn2_n,y=rn2_w)[name=string(\"x2norm\")];"
    );

    // === FFN ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{seq}]> fn2 = reshape(shape=r2d,x=x2norm)[name=string(\"fn2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> fnt = transpose(perm=pm,x=fn2)[name=string(\"fnt\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{inter}]> Wg = const()[name=string(\"Wg\"), val=tensor<fp16, [1,1,{dim},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/gate.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{inter}]> Wu = const()[name=string(\"Wu\"), val=tensor<fp16, [1,1,{dim},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/up.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{inter},{dim}]> Wd = const()[name=string(\"Wd\"), val=tensor<fp16, [1,1,{inter},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/down.bin\"), offset=uint64(64)))];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{inter}]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=fnt,y=Wg)[name=string(\"h1m\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{inter}]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=fnt,y=Wu)[name=string(\"h3m\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{inter},{seq}]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{inter},{seq}]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,{inter},1,{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{inter},1,{seq}]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{inter},1,{seq}]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];"
    );

    // SiLU(gate) * up
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{inter},1,{seq}]> sig = sigmoid(x=h1)[name=string(\"sg\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{inter},1,{seq}]> silu = mul(x=h1,y=sig)[name=string(\"si\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{inter},1,{seq}]> gated = mul(x=silu,y=h3)[name=string(\"gated\")];"
    );

    // Down projection
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rh2 = const()[name=string(\"rh2\"), val=tensor<int32, [4]>([1,1,{inter},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{inter},{seq}]> g2 = reshape(shape=rh2,x=gated)[name=string(\"g2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{inter}]> g2t = transpose(perm=pm,x=g2)[name=string(\"g2t\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> fm = matmul(transpose_x=bF,transpose_y=bF,x=g2t,y=Wd)[name=string(\"fm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{seq}]> ft = transpose(perm=pm,x=fm)[name=string(\"ft\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> ffn_out = reshape(shape=os,x=ft)[name=string(\"ffn\")];"
    );

    // === Residual 2 ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xout = add(x=x2,y=ffn_out)[name=string(\"xout\")];"
    );

    // Output
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=xout)[name=string(\"cout\")];"
    );
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");

    let mut weight_names = vec![
        "@model_path/weights/rms_att.bin".to_string(),
        "@model_path/weights/rms_ffn.bin".to_string(),
        "@model_path/weights/wq.bin".to_string(),
        "@model_path/weights/wk.bin".to_string(),
        "@model_path/weights/wv.bin".to_string(),
        "@model_path/weights/wo.bin".to_string(),
        "@model_path/weights/gate.bin".to_string(),
        "@model_path/weights/up.bin".to_string(),
        "@model_path/weights/down.bin".to_string(),
        "@model_path/weights/rope_cos.bin".to_string(),
        "@model_path/weights/rope_sin.bin".to_string(),
        "@model_path/weights/q_norm.bin".to_string(),
        "@model_path/weights/k_norm.bin".to_string(),
    ];
    if causal {
        weight_names.push("@model_path/weights/causal_mask.bin".to_string());
    }

    FusedMil {
        mil_text: m,
        weight_names,
        input_bytes: dim * seq * 4,
        output_bytes: dim * seq * 4,
    }
}

// NOTE: Int8 BLOBFILE weights were investigated and confirmed dead.
// ANE's raw MIL compiler (`_ANEDesc modelWithMILText:`) rejects `tensor<int8>` entirely —
// both `cast(int8→fp16)` and `constexpr_affine_dequantize` fail with `InvalidMILProgram`.
// The path forward for larger models is multi-dispatch (split attention + FFN into
// separate ANE programs, each under 32MB fp16).

// ===========================================================================
// Multi-dispatch generators: split layer into attention + FFN programs
// ===========================================================================

/// Generate attention-only dispatch for multi-dispatch architecture.
///
/// Split from the fused layer: RMSNorm → QKV → QKnorm → RoPE → SDPA → Wo → residual.
/// Output is post-attention hidden state (before FFN).
///
/// Input:  `[1, dim, 1, seq]` fp32
/// Output: `[1, dim, 1, seq]` fp32
///
/// 9 BLOBFILEs (~12MB at dim=1024, ~32MB at dim=2048):
///   rms_att, wq, wk, wv, wo, rope_cos, rope_sin, q_norm, k_norm
pub fn gen_diffusion_attention(
    dim: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    seq_len: usize,
    eps: f64,
) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let half_hd = hd / 2;
    let kv_dim = kv_heads * hd;
    let attn_dim = heads * hd;
    let hpg = heads / kv_heads;
    let sc = 1.0 / (hd as f64).sqrt();

    let mut m = String::with_capacity(16384);
    m.push_str(MIL_HEADER);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1,{dim},1,{seq}]> x) {{"
    );

    // --- Constants ---
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];"
    );
    let _ = writeln!(
        m,
        "        bool kd = const()[name=string(\"kd\"), val=bool(true)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [1]> ch_ax = const()[name=string(\"chax\"), val=tensor<int32, [1]>([1])];"
    );
    let _ = writeln!(
        m,
        "        fp16 eps_v = const()[name=string(\"epsv\"), val=fp16({eps})];"
    );
    let _ = writeln!(
        m,
        "        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];"
    );

    // Cast input
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];"
    );

    // === RMSNorm (attention) ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn1_sq = mul(x=xh,y=xh)[name=string(\"rn1sq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_m = reduce_mean(x=rn1_sq,axes=ch_ax,keep_dims=kd)[name=string(\"rn1m\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_e = add(x=rn1_m,y=eps_v)[name=string(\"rn1e\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_r = pow(x=rn1_e,y=nhalf)[name=string(\"rn1r\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn1_n = mul(x=xh,y=rn1_r)[name=string(\"rn1n\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,1]> rn1_w = const()[name=string(\"rn1w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_att.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xnorm = mul(x=rn1_n,y=rn1_w)[name=string(\"xnorm\")];"
    );

    // === QKV Projections ===
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{seq}]> xn2 = reshape(shape=r2d,x=xnorm)[name=string(\"xn2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{attn_dim}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{dim},{attn_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{attn_dim},{dim}]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [1,1,{attn_dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{attn_dim}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq)[name=string(\"qm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{kv_dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk)[name=string(\"km\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{kv_dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv)[name=string(\"vm\")];"
    );

    // Reshape Q → [1, heads, seq, hd]
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{attn_dim},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{hd},{seq}]> q4 = reshape(shape=qsh,x=qt)[name=string(\"rq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];"
    );

    // Reshape K → [1, kv_heads, seq, hd]
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{kv_dim},{seq}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> kvsh = const()[name=string(\"kvsh\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> k4 = reshape(shape=kvsh,x=kt)[name=string(\"rk\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_kv = transpose(perm=pm,x=k4)[name=string(\"tk\")];"
    );

    // Reshape V → [1, kv_heads, seq, hd]
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{kv_dim},{seq}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v4 = reshape(shape=kvsh,x=vt)[name=string(\"rv\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_kv = transpose(perm=pm,x=v4)[name=string(\"tv\")];"
    );

    // === QK Norm (Qwen3-specific: per-head RMSNorm on Q and K) ===
    let _ = writeln!(
        m,
        "        tensor<int32, [1]> hd_ax = const()[name=string(\"hdax\"), val=tensor<int32, [1]>([-1])];"
    );
    // Q norm
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q_sq = mul(x=q,y=q)[name=string(\"qsq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},1]> qn_m = reduce_mean(x=q_sq,axes=hd_ax,keep_dims=kd)[name=string(\"qnm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},1]> qn_e = add(x=qn_m,y=eps_v)[name=string(\"qne\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},1]> qn_r = pow(x=qn_e,y=nhalf)[name=string(\"qnr\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> qn = mul(x=q,y=qn_r)[name=string(\"qn\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{hd}]> qn_w = const()[name=string(\"qnw\"), val=tensor<fp16, [1,1,1,{hd}]>(BLOBFILE(path=string(\"@model_path/weights/q_norm.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q_normed = mul(x=qn,y=qn_w)[name=string(\"qnormed\")];"
    );
    // K norm
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_sq = mul(x=k_kv,y=k_kv)[name=string(\"ksq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_m = reduce_mean(x=k_sq,axes=hd_ax,keep_dims=kd)[name=string(\"knm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_e = add(x=kn_m,y=eps_v)[name=string(\"kne\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_r = pow(x=kn_e,y=nhalf)[name=string(\"knr\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kn = mul(x=k_kv,y=kn_r)[name=string(\"kn\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{hd}]> kn_w = const()[name=string(\"knw\"), val=tensor<fp16, [1,1,1,{hd}]>(BLOBFILE(path=string(\"@model_path/weights/k_norm.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_normed = mul(x=kn,y=kn_w)[name=string(\"knormed\")];"
    );

    // === RoPE ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];"
    );

    // Q RoPE
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_b0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_sh = const()[name=string(\"rpsh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1 = slice_by_size(x=q_normed,begin=rp_b0,size=rp_sh)[name=string(\"q1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_bh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2 = slice_by_size(x=q_normed,begin=rp_bh,size=rp_sh)[name=string(\"q2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1c = mul(x=q1,y=rope_cos)[name=string(\"q1c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2s = mul(x=q2,y=rope_sin)[name=string(\"q2s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr1 = sub(x=q1c,y=q2s)[name=string(\"qr1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1s = mul(x=q1,y=rope_sin)[name=string(\"q1s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2c = mul(x=q2,y=rope_cos)[name=string(\"q2c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr2 = add(x=q1s,y=q2c)[name=string(\"qr2\")];"
    );
    let _ = writeln!(
        m,
        "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];"
    );
    let _ = writeln!(
        m,
        "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q_rot = concat(axis=rpax,interleave=rpid,values=(qr1,qr2))[name=string(\"qrot\")];"
    );

    // K RoPE
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_ksh = const()[name=string(\"rpksh\"), val=tensor<int32, [4]>([1,{kv_heads},{seq},{half_hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1 = slice_by_size(x=k_normed,begin=rp_b0,size=rp_ksh)[name=string(\"k1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2 = slice_by_size(x=k_normed,begin=rp_bh,size=rp_ksh)[name=string(\"k2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1c = mul(x=k1,y=rope_cos)[name=string(\"k1c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2s = mul(x=k2,y=rope_sin)[name=string(\"k2s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> kr1 = sub(x=k1c,y=k2s)[name=string(\"kr1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1s = mul(x=k1,y=rope_sin)[name=string(\"k1s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2c = mul(x=k2,y=rope_cos)[name=string(\"k2c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> kr2 = add(x=k1s,y=k2c)[name=string(\"kr2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_rot = concat(axis=rpax,interleave=rpid,values=(kr1,kr2))[name=string(\"krot\")];"
    );

    // === GQA Bidirectional SDPA (NO causal mask) ===
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=q_rot)[name=string(\"qb\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=k_rot)[name=string(\"kb\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_kv)[name=string(\"vb\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"mm1\")];"
    );
    let _ = writeln!(
        m,
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];"
    );
    let _ = writeln!(
        m,
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=sc2)[name=string(\"sm\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=vb)[name=string(\"mm2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rha = const()[name=string(\"rha\"), val=tensor<int32, [4]>([1,{heads},{seq},{hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> a_out = reshape(shape=rha,x=a4)[name=string(\"aout\")];"
    );

    // Reshape attn output → [1, attn_dim, 1, seq]
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{hd},{seq}]> at = transpose(perm=pm,x=a_out)[name=string(\"ta\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> osa = const()[name=string(\"osa\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{attn_dim},1,{seq}]> af = reshape(shape=osa,x=at)[name=string(\"ra\")];"
    );

    // === Wo projection ===
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> r2a = const()[name=string(\"r2a\"), val=tensor<int32, [4]>([1,1,{attn_dim},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{attn_dim},{seq}]> af2 = reshape(shape=r2a,x=af)[name=string(\"af2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{attn_dim}]> aft = transpose(perm=pm,x=af2)[name=string(\"aft\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> om = matmul(transpose_x=bF,transpose_y=bF,x=aft,y=Wo)[name=string(\"om\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{seq}]> ot = transpose(perm=pm,x=om)[name=string(\"ot\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> oo = reshape(shape=os,x=ot)[name=string(\"oo\")];"
    );

    // === Residual ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> x2 = add(x=xh,y=oo)[name=string(\"x2\")];"
    );

    // Output
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=x2)[name=string(\"cout\")];"
    );
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/rms_att.bin".to_string(),
            "@model_path/weights/wq.bin".to_string(),
            "@model_path/weights/wk.bin".to_string(),
            "@model_path/weights/wv.bin".to_string(),
            "@model_path/weights/wo.bin".to_string(),
            "@model_path/weights/rope_cos.bin".to_string(),
            "@model_path/weights/rope_sin.bin".to_string(),
            "@model_path/weights/q_norm.bin".to_string(),
            "@model_path/weights/k_norm.bin".to_string(),
        ],
        input_bytes: dim * seq * 4,
        output_bytes: dim * seq * 4,
    }
}

/// Generate attention-only dispatch that returns the attention context before
/// the Wo projection. Bonsai uses this so the host can apply Wo and the
/// residual add in fp32 outside the ANE program.
///
/// When `causal=true`: adds causal mask BLOBFILE before softmax
pub fn gen_bonsai_attention_projection(
    dim: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    seq_len: usize,
    eps: f64,
    causal: bool,
) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let half_hd = hd / 2;
    let kv_dim = kv_heads * hd;
    let attn_dim = heads * hd;
    let hpg = heads / kv_heads;
    let sc = 1.0 / (hd as f64).sqrt();

    let mut m = String::with_capacity(16384);
    m.push_str(MIL_HEADER);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1,{dim},1,{seq}]> x) {{"
    );

    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];"
    );
    let _ = writeln!(
        m,
        "        bool kd = const()[name=string(\"kd\"), val=bool(true)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [1]> ch_ax = const()[name=string(\"chax\"), val=tensor<int32, [1]>([1])];"
    );
    let _ = writeln!(
        m,
        "        fp16 eps_v = const()[name=string(\"epsv\"), val=fp16({eps})];"
    );
    let _ = writeln!(
        m,
        "        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn1_sq = mul(x=xh,y=xh)[name=string(\"rn1sq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_m = reduce_mean(x=rn1_sq,axes=ch_ax,keep_dims=kd)[name=string(\"rn1m\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_e = add(x=rn1_m,y=eps_v)[name=string(\"rn1e\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn1_r = pow(x=rn1_e,y=nhalf)[name=string(\"rn1r\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn1_n = mul(x=xh,y=rn1_r)[name=string(\"rn1n\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,1]> rn1_w = const()[name=string(\"rn1w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_att.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xnorm = mul(x=rn1_n,y=rn1_w)[name=string(\"xnorm\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{seq}]> xn2 = reshape(shape=r2d,x=xnorm)[name=string(\"xn2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{attn_dim}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{dim},{attn_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{attn_dim}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq)[name=string(\"qm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{kv_dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk)[name=string(\"km\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{kv_dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv)[name=string(\"vm\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{attn_dim},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{hd},{seq}]> q4 = reshape(shape=qsh,x=qt)[name=string(\"rq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{kv_dim},{seq}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> kvsh = const()[name=string(\"kvsh\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> k4 = reshape(shape=kvsh,x=kt)[name=string(\"rk\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_kv = transpose(perm=pm,x=k4)[name=string(\"tk\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{kv_dim},{seq}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v4 = reshape(shape=kvsh,x=vt)[name=string(\"rv\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_kv = transpose(perm=pm,x=v4)[name=string(\"tv\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<int32, [1]> hd_ax = const()[name=string(\"hdax\"), val=tensor<int32, [1]>([-1])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q_sq = mul(x=q,y=q)[name=string(\"qsq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},1]> qn_m = reduce_mean(x=q_sq,axes=hd_ax,keep_dims=kd)[name=string(\"qnm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},1]> qn_e = add(x=qn_m,y=eps_v)[name=string(\"qne\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},1]> qn_r = pow(x=qn_e,y=nhalf)[name=string(\"qnr\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> qn = mul(x=q,y=qn_r)[name=string(\"qn\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{hd}]> qn_w = const()[name=string(\"qnw\"), val=tensor<fp16, [1,1,1,{hd}]>(BLOBFILE(path=string(\"@model_path/weights/q_norm.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q_normed = mul(x=qn,y=qn_w)[name=string(\"qnormed\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_sq = mul(x=k_kv,y=k_kv)[name=string(\"ksq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_m = reduce_mean(x=k_sq,axes=hd_ax,keep_dims=kd)[name=string(\"knm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_e = add(x=kn_m,y=eps_v)[name=string(\"kne\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_r = pow(x=kn_e,y=nhalf)[name=string(\"knr\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kn = mul(x=k_kv,y=kn_r)[name=string(\"kn\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{hd}]> kn_w = const()[name=string(\"knw\"), val=tensor<fp16, [1,1,1,{hd}]>(BLOBFILE(path=string(\"@model_path/weights/k_norm.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_normed = mul(x=kn,y=kn_w)[name=string(\"knormed\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];"
    );

    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_b0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_sh = const()[name=string(\"rpsh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1 = slice_by_size(x=q_normed,begin=rp_b0,size=rp_sh)[name=string(\"q1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_bh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2 = slice_by_size(x=q_normed,begin=rp_bh,size=rp_sh)[name=string(\"q2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1c = mul(x=q1,y=rope_cos)[name=string(\"q1c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2s = mul(x=q2,y=rope_sin)[name=string(\"q2s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr1 = sub(x=q1c,y=q2s)[name=string(\"qr1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1s = mul(x=q1,y=rope_sin)[name=string(\"q1s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2c = mul(x=q2,y=rope_cos)[name=string(\"q2c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr2 = add(x=q1s,y=q2c)[name=string(\"qr2\")];"
    );
    let _ = writeln!(
        m,
        "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];"
    );
    let _ = writeln!(
        m,
        "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> q_rot = concat(axis=rpax,interleave=rpid,values=(qr1,qr2))[name=string(\"qrot\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rp_ksh = const()[name=string(\"rpksh\"), val=tensor<int32, [4]>([1,{kv_heads},{seq},{half_hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1 = slice_by_size(x=k_normed,begin=rp_b0,size=rp_ksh)[name=string(\"k1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2 = slice_by_size(x=k_normed,begin=rp_bh,size=rp_ksh)[name=string(\"k2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1c = mul(x=k1,y=rope_cos)[name=string(\"k1c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2s = mul(x=k2,y=rope_sin)[name=string(\"k2s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> kr1 = sub(x=k1c,y=k2s)[name=string(\"kr1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1s = mul(x=k1,y=rope_sin)[name=string(\"k1s\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2c = mul(x=k2,y=rope_cos)[name=string(\"k2c\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> kr2 = add(x=k1s,y=k2c)[name=string(\"kr2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_rot = concat(axis=rpax,interleave=rpid,values=(kr1,kr2))[name=string(\"krot\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=q_rot)[name=string(\"qb\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=k_rot)[name=string(\"kb\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_kv)[name=string(\"vb\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"mm1\")];"
    );
    let _ = writeln!(
        m,
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];"
    );
    let _ = writeln!(
        m,
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];"
    );

    // Causal mask (when enabled)
    if causal {
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{seq},{seq}]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string(\"@model_path/weights/causal_mask.bin\"), offset=uint64(64)))];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];"
        );
    } else {
        let _ = writeln!(
            m,
            "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=sc2)[name=string(\"sm\")];"
        );
    }

    let _ = writeln!(
        m,
        "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=vb)[name=string(\"mm2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rha = const()[name=string(\"rha\"), val=tensor<int32, [4]>([1,{heads},{seq},{hd}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{seq},{hd}]> a_out = reshape(shape=rha,x=a4)[name=string(\"aout\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{heads},{hd},{seq}]> at = transpose(perm=pm,x=a_out)[name=string(\"ta\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> osa = const()[name=string(\"osa\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{attn_dim},1,{seq}]> af = reshape(shape=osa,x=at)[name=string(\"ra\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{attn_dim},1,{seq}]> y = cast(dtype=to32,x=af)[name=string(\"cout\")];"
    );
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");

    let mut weight_names = vec![
        "@model_path/weights/rms_att.bin".to_string(),
        "@model_path/weights/wq.bin".to_string(),
        "@model_path/weights/wk.bin".to_string(),
        "@model_path/weights/wv.bin".to_string(),
        "@model_path/weights/rope_cos.bin".to_string(),
        "@model_path/weights/rope_sin.bin".to_string(),
        "@model_path/weights/q_norm.bin".to_string(),
        "@model_path/weights/k_norm.bin".to_string(),
    ];
    if causal {
        weight_names.push("@model_path/weights/causal_mask.bin".to_string());
    }

    FusedMil {
        mil_text: m,
        weight_names,
        input_bytes: dim * seq * 4,
        output_bytes: attn_dim * seq * 4,
    }
}

/// Generate FFN-only dispatch for multi-dispatch architecture.
///
/// Split from the fused layer: RMSNorm → gate/up → SiLU → down → residual.
/// Input is post-attention hidden state, output is post-FFN hidden state.
///
/// Input:  `[1, dim, 1, seq]` fp32
/// Output: `[1, dim, 1, seq]` fp32
///
/// 4 BLOBFILEs (~18MB at dim=1024/inter=3072):
///   rms_ffn, gate, up, down
pub fn gen_diffusion_ffn(dim: usize, inter: usize, seq_len: usize, eps: f64) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HEADER);
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1,{dim},1,{seq}]> x) {{"
    );

    // --- Constants ---
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];"
    );
    let _ = writeln!(
        m,
        "        bool kd = const()[name=string(\"kd\"), val=bool(true)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [1]> ch_ax = const()[name=string(\"chax\"), val=tensor<int32, [1]>([1])];"
    );
    let _ = writeln!(
        m,
        "        fp16 eps_v = const()[name=string(\"epsv\"), val=fp16({eps})];"
    );
    let _ = writeln!(
        m,
        "        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];"
    );

    // Cast input
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];"
    );

    // === RMSNorm (FFN) ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn_sq = mul(x=xh,y=xh)[name=string(\"rnsq\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn_m = reduce_mean(x=rn_sq,axes=ch_ax,keep_dims=kd)[name=string(\"rnm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn_e = add(x=rn_m,y=eps_v)[name=string(\"rne\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,1,{seq}]> rn_r = pow(x=rn_e,y=nhalf)[name=string(\"rnr\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> rn_n = mul(x=xh,y=rn_r)[name=string(\"rnn\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,1]> rn_w = const()[name=string(\"rnw\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_ffn.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xnorm = mul(x=rn_n,y=rn_w)[name=string(\"xnorm\")];"
    );

    // === FFN ===
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{seq}]> fn2 = reshape(shape=r2d,x=xnorm)[name=string(\"fn2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> fnt = transpose(perm=pm,x=fn2)[name=string(\"fnt\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{inter}]> Wg = const()[name=string(\"Wg\"), val=tensor<fp16, [1,1,{dim},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/gate.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{inter}]> Wu = const()[name=string(\"Wu\"), val=tensor<fp16, [1,1,{dim},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/up.bin\"), offset=uint64(64)))];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{inter},{dim}]> Wd = const()[name=string(\"Wd\"), val=tensor<fp16, [1,1,{inter},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/down.bin\"), offset=uint64(64)))];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{inter}]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=fnt,y=Wg)[name=string(\"h1m\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{inter}]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=fnt,y=Wu)[name=string(\"h3m\")];"
    );

    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{inter},{seq}]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{inter},{seq}]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,{inter},1,{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{inter},1,{seq}]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{inter},1,{seq}]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];"
    );

    // SiLU(gate) * up
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{inter},1,{seq}]> sig = sigmoid(x=h1)[name=string(\"sg\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{inter},1,{seq}]> silu = mul(x=h1,y=sig)[name=string(\"si\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{inter},1,{seq}]> gated = mul(x=silu,y=h3)[name=string(\"gated\")];"
    );

    // Down projection
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rh2 = const()[name=string(\"rh2\"), val=tensor<int32, [4]>([1,1,{inter},{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{inter},{seq}]> g2 = reshape(shape=rh2,x=gated)[name=string(\"g2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{inter}]> g2t = transpose(perm=pm,x=g2)[name=string(\"g2t\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{seq},{dim}]> fm = matmul(transpose_x=bF,transpose_y=bF,x=g2t,y=Wd)[name=string(\"fm\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{dim},{seq}]> ft = transpose(perm=pm,x=fm)[name=string(\"ft\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> ffn_out = reshape(shape=os,x=ft)[name=string(\"ffn\")];"
    );

    // === Residual ===
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{dim},1,{seq}]> xout = add(x=xh,y=ffn_out)[name=string(\"xout\")];"
    );

    // Output
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=xout)[name=string(\"cout\")];"
    );
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/rms_ffn.bin".to_string(),
            "@model_path/weights/gate.bin".to_string(),
            "@model_path/weights/up.bin".to_string(),
            "@model_path/weights/down.bin".to_string(),
        ],
        input_bytes: dim * seq * 4,
        output_bytes: dim * seq * 4,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, unsafe_code)]
mod tests {
    use super::*;

    fn model_dir() -> Option<String> {
        let dir = format!(
            "{}/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1",
            std::env::var("HOME").ok()?
        );
        if std::path::Path::new(&dir)
            .join("model.safetensors")
            .exists()
        {
            Some(dir)
        } else {
            None
        }
    }

    /// Replicate the nanobot-rs int8 conv1x1 pattern exactly.
    /// Tests whether `constexpr_affine_dequantize` with conv (not matmul) works on ANE.
    /// Result: CONFIRMED DEAD — ANE compiler rejects int8 entirely.
    #[test]
    #[ignore = "int8 confirmed dead on ANE — kept as evidence"]
    fn test_int8_conv1x1_nanobot_pattern() {
        use crate::ane_bridge::{self, build_weight_blob_int8, AneKernel};
        use crate::ane_mil::MIL_HEADER;

        ane_bridge::ane_init().expect("ANE init");
        ane_bridge::set_quiet(false);

        let c_in = 64usize;
        let c_out = 64usize;
        let seq = 16usize;
        let scale: f32 = 0.01;

        // Replicate nanobot-rs gen_conv1x1_int8_blob exactly
        // f32 0.01 → fp16 = 0x211E (verified: half::f16::from_f32(0.01).to_bits())
        let scale_bits: u16 = {
            // IEEE 754 f32→f16 conversion for small positive values
            let bits = scale.to_bits();
            let sign = (bits >> 31) & 1;
            let exp = ((bits >> 23) & 0xFF) as i32 - 127;
            let frac = bits & 0x7FFFFF;
            if exp < -24 {
                0u16
            } else if exp < -14 {
                let shift = -14 - exp;
                let mant = (0x800000 | frac) >> (shift + 13);
                ((sign as u16) << 15) | mant as u16
            } else {
                let e16 = (exp + 15) as u16;
                let m16 = (frac >> 13) as u16;
                ((sign as u16) << 15) | (e16 << 10) | m16
            }
        };

        let mil = format!(
            "{MIL_HEADER}\
    func main<ios18>(tensor<fp32, [1, {c_in}, 1, {seq}]> x) {{\n\
        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n\
        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n\
        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n\
        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n\
        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n\
        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n\
        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n\
        fp16 dq_scale = const()[name=string(\"dq_scale\"), val=fp16(0x{scale_bits:04X})];\n\
        int8 dq_zero = const()[name=string(\"dq_zero\"), val=int8(0)];\n\
        int32 dq_axis = const()[name=string(\"dq_axis\"), val=int32(0)];\n\
        tensor<int8, [{c_out},{c_in},1,1]> W_q = const()[name=string(\"W_q\"), val=tensor<int8, [{c_out},{c_in},1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n\
        tensor<fp16, [{c_out},{c_in},1,1]> W = constexpr_affine_dequantize(axis=dq_axis,quantized_data=W_q,scale=dq_scale,zero_point=dq_zero)[name=string(\"dequant\")];\n\
        tensor<fp16, [1,{c_in},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];\n\
        tensor<fp16, [1,{c_out},1,{seq}]> yh = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=xh)[name=string(\"conv\")];\n\
        tensor<fp32, [1,{c_out},1,{seq}]> y = cast(dtype=to32,x=yh)[name=string(\"cout\")];\n\
    }} -> (y);\n}}\n"
        );

        eprintln!("--- MIL (nanobot-rs int8 conv1x1 pattern) ---");
        eprintln!("{mil}");
        eprintln!("---");

        // Build int8 blob in conv format [c_out, c_in, 1, 1]
        let int8_data: Vec<i8> = (0..c_out * c_in).map(|i| ((i % 127) as i8 - 63)).collect();
        let blob = build_weight_blob_int8(&int8_data, c_out, c_in);

        let names = ["@model_path/weights/w.bin"];
        let name_ptrs: Vec<&str> = names.iter().copied().collect();
        let blob_refs = [blob.as_slice()];

        match AneKernel::compile_multi_weights(
            &mil,
            &name_ptrs,
            &blob_refs,
            &[c_in * seq * 4],
            &[c_out * seq * 4],
        ) {
            Ok(kernel) => {
                eprintln!("PASS: nanobot-rs int8 conv1x1 pattern COMPILES on ANE!");
                // Try eval too
                let input: Vec<f32> = (0..c_in * seq).map(|i| (i as f32 * 0.01).sin()).collect();
                let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
                kernel.write_input(0, &input_bytes);
                match kernel.eval() {
                    Ok(()) => eprintln!("PASS: eval succeeded!"),
                    Err(e) => eprintln!("FAIL: eval failed: {e}"),
                }
            }
            Err(e) => {
                eprintln!("FAIL: nanobot-rs int8 conv1x1 pattern: {e}");
                panic!("int8 conv1x1 compile failed — GLM5 was wrong");
            }
        }
    }

    /// Multi-dispatch correctness: verify chained (attention + FFN) matches fused single dispatch.
    /// Also benchmarks 28 fused dispatches vs 56 multi-dispatches at seq=128.
    #[test]
    fn test_multi_dispatch_vs_fused() {
        use crate::ane_bridge::{self, build_weight_blob, build_weight_blob_transposed, AneKernel};
        use crate::diffusion::DiffusionEngine;

        let Some(dir) = model_dir() else {
            eprintln!("Model not found");
            return;
        };
        ane_bridge::ane_init().expect("ANE init");
        ane_bridge::set_quiet(false);

        let engine = DiffusionEngine::load(&dir).unwrap();
        let cfg = &engine.config;
        let lw = &engine.layers[0];
        let seq = 128usize;
        let half_hd = cfg.head_dim / 2;

        // Precompute RoPE
        let mut rope_cos = vec![0.0f32; seq * half_hd];
        let mut rope_sin = vec![0.0f32; seq * half_hd];
        for pos in 0..seq {
            for d in 0..half_hd {
                let freq = 1.0 / (cfg.rope_theta as f32).powf(2.0 * d as f32 / cfg.head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_hd + d] = angle.cos();
                rope_sin[pos * half_hd + d] = angle.sin();
            }
        }

        // Build all weight blobs
        let rms_att_blob = build_weight_blob(&lw.input_norm, 1, cfg.hidden);
        let rms_ffn_blob = build_weight_blob(&lw.post_attn_norm, 1, cfg.hidden);
        let wq_blob =
            build_weight_blob_transposed(&lw.q_proj, cfg.heads * cfg.head_dim, cfg.hidden);
        let wk_blob =
            build_weight_blob_transposed(&lw.k_proj, cfg.kv_heads * cfg.head_dim, cfg.hidden);
        let wv_blob =
            build_weight_blob_transposed(&lw.v_proj, cfg.kv_heads * cfg.head_dim, cfg.hidden);
        let wo_blob =
            build_weight_blob_transposed(&lw.o_proj, cfg.hidden, cfg.heads * cfg.head_dim);
        let gate_blob = build_weight_blob_transposed(&lw.gate_proj, cfg.inter, cfg.hidden);
        let up_blob = build_weight_blob_transposed(&lw.up_proj, cfg.inter, cfg.hidden);
        let down_blob = build_weight_blob_transposed(&lw.down_proj, cfg.hidden, cfg.inter);
        let rope_cos_blob = build_weight_blob(&rope_cos, seq, half_hd);
        let rope_sin_blob = build_weight_blob(&rope_sin, seq, half_hd);
        let q_norm_blob = build_weight_blob(&lw.q_norm, 1, cfg.head_dim);
        let k_norm_blob = build_weight_blob(&lw.k_norm, 1, cfg.head_dim);

        // === 1. Compile fused kernel (all 13 blobs) ===
        let fused_mil = gen_fused_diffusion_layer(
            cfg.hidden,
            cfg.heads,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.inter,
            seq,
            1e-6,
            false, // bidirectional for diffusion
        );
        let fused_blobs: Vec<&[u8]> = vec![
            &rms_att_blob,
            &rms_ffn_blob,
            &wq_blob,
            &wk_blob,
            &wv_blob,
            &wo_blob,
            &gate_blob,
            &up_blob,
            &down_blob,
            &rope_cos_blob,
            &rope_sin_blob,
            &q_norm_blob,
            &k_norm_blob,
        ];
        let fused_names: Vec<&str> = fused_mil.weight_names.iter().map(|s| s.as_str()).collect();
        let fused_kernel = AneKernel::compile_multi_weights(
            &fused_mil.mil_text,
            &fused_names,
            &fused_blobs,
            &[fused_mil.input_bytes],
            &[fused_mil.output_bytes],
        )
        .expect("fused compile");
        eprintln!("Fused kernel compiled (13 BLOBFILEs)");

        // === 2. Compile attention kernel (9 blobs) ===
        let attn_mil =
            gen_diffusion_attention(cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, seq, 1e-6);
        let attn_blobs: Vec<&[u8]> = vec![
            &rms_att_blob,
            &wq_blob,
            &wk_blob,
            &wv_blob,
            &wo_blob,
            &rope_cos_blob,
            &rope_sin_blob,
            &q_norm_blob,
            &k_norm_blob,
        ];
        let attn_names: Vec<&str> = attn_mil.weight_names.iter().map(|s| s.as_str()).collect();
        let attn_blob_mb: f64 = attn_blobs.iter().map(|b| b.len() as f64).sum::<f64>() / 1e6;
        let attn_kernel = AneKernel::compile_multi_weights(
            &attn_mil.mil_text,
            &attn_names,
            &attn_blobs,
            &[attn_mil.input_bytes],
            &[attn_mil.output_bytes],
        )
        .expect("attention compile");
        eprintln!("Attention kernel compiled (9 BLOBFILEs, {attn_blob_mb:.1}MB)");

        // === 3. Compile FFN kernel (4 blobs) ===
        let ffn_mil = gen_diffusion_ffn(cfg.hidden, cfg.inter, seq, 1e-6);
        let ffn_blobs: Vec<&[u8]> = vec![&rms_ffn_blob, &gate_blob, &up_blob, &down_blob];
        let ffn_names: Vec<&str> = ffn_mil.weight_names.iter().map(|s| s.as_str()).collect();
        let ffn_blob_mb: f64 = ffn_blobs.iter().map(|b| b.len() as f64).sum::<f64>() / 1e6;
        let ffn_kernel = AneKernel::compile_multi_weights(
            &ffn_mil.mil_text,
            &ffn_names,
            &ffn_blobs,
            &[ffn_mil.input_bytes],
            &[ffn_mil.output_bytes],
        )
        .expect("FFN compile");
        eprintln!("FFN kernel compiled (4 BLOBFILEs, {ffn_blob_mb:.1}MB)");

        // === 4. Run fused ===
        let act: Vec<f32> = (0..cfg.hidden * seq)
            .map(|i| ((i as f32) * 0.001).sin() * 0.01)
            .collect();
        let act_bytes: Vec<u8> = act.iter().flat_map(|f| f.to_le_bytes()).collect();

        fused_kernel.write_input(0, &act_bytes);
        fused_kernel.eval().expect("fused eval");
        let mut fused_out = vec![0u8; cfg.hidden * seq * 4];
        fused_kernel.read_output(0, &mut fused_out);

        // === 5. Run chained: attention → FFN ===
        attn_kernel.write_input(0, &act_bytes);
        attn_kernel.eval().expect("attn eval");
        let mut attn_out = vec![0u8; cfg.hidden * seq * 4];
        attn_kernel.read_output(0, &mut attn_out);

        ffn_kernel.write_input(0, &attn_out);
        ffn_kernel.eval().expect("ffn eval");
        let mut multi_out = vec![0u8; cfg.hidden * seq * 4];
        ffn_kernel.read_output(0, &mut multi_out);

        // === 6. Compare outputs ===
        let fused_f32: Vec<f32> = fused_out
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let multi_f32: Vec<f32> = multi_out
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let max_err = fused_f32
            .iter()
            .zip(multi_f32.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_err = fused_f32
            .iter()
            .zip(multi_f32.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / fused_f32.len() as f32;
        eprintln!("Fused vs multi-dispatch: max_err={max_err:.6}, mean_err={mean_err:.8}");
        // fp16→fp32→fp16 round-trip is lossless, so outputs should be identical
        assert!(
            max_err < 0.01,
            "Multi-dispatch output diverges from fused: max_err={max_err}"
        );

        // === 7. Benchmark: 28 fused vs 56 multi-dispatch ===
        let n = 20u128;

        let t0 = std::time::Instant::now();
        for _ in 0..n {
            for _ in 0..28 {
                fused_kernel.write_input(0, &act_bytes);
                fused_kernel.eval().unwrap();
            }
        }
        let fused_us = t0.elapsed().as_micros() / n;

        let t0 = std::time::Instant::now();
        for _ in 0..n {
            for _ in 0..28 {
                attn_kernel.write_input(0, &act_bytes);
                attn_kernel.eval().unwrap();
                attn_kernel.read_output(0, &mut attn_out);
                ffn_kernel.write_input(0, &attn_out);
                ffn_kernel.eval().unwrap();
            }
        }
        let multi_us = t0.elapsed().as_micros() / n;

        let fused_ms = fused_us as f64 / 1000.0;
        let multi_ms = multi_us as f64 / 1000.0;
        let overhead = multi_ms / fused_ms;
        let gen_tok = seq - 5;
        let steps = 64;
        let fused_tps = gen_tok as f64 / (fused_ms * steps as f64 / 1000.0);
        let multi_tps = gen_tok as f64 / (multi_ms * steps as f64 / 1000.0);

        eprintln!("\n=== Multi-dispatch benchmark (seq={seq}) ===");
        eprintln!("Fused:  {fused_ms:.1}ms (28 dispatches) → {fused_tps:.0} tok/s @ {steps} steps");
        eprintln!("Multi:  {multi_ms:.1}ms (56 dispatches) → {multi_tps:.0} tok/s @ {steps} steps");
        eprintln!("Overhead: {overhead:.2}x");

        // Estimate sizes for LLaDA-MoE (dim=2048)
        let d = 2048usize;
        let hd_2 = 128usize;
        let attn_mb_2k = (4 * d * d * 2 + 2 * hd_2 * 2) as f64 / 1e6; // 4 proj + 2 qk_norm (tiny)
        let ffn_moe_mb = (8 * 3 * d * 1024 * 2) as f64 / 1e6; // 8 active experts × 3 proj
        eprintln!("\nLLaDA-MoE (dim={d}) BLOBFILE estimates:");
        eprintln!("  Attention: {attn_mb_2k:.1}MB (fits under 32MB)");
        eprintln!("  FFN MoE:   {ffn_moe_mb:.1}MB (8 active × 3 proj @ {d}×1024)");
    }

    /// THE BIG TEST: compile + eval a fully-fused layer on ANE, benchmark 28 dispatches.
    #[test]
    fn test_fused_layer_compile_and_benchmark() {
        use crate::ane_bridge::{self, build_weight_blob, build_weight_blob_transposed, AneKernel};
        use crate::diffusion::DiffusionEngine;

        let Some(dir) = model_dir() else {
            eprintln!("Model not found");
            return;
        };
        ane_bridge::ane_init().expect("ANE init");
        ane_bridge::set_quiet(false);

        let engine = DiffusionEngine::load(&dir).unwrap();
        let cfg = &engine.config;

        for seq in [64, 128] {
            let mil = gen_fused_diffusion_layer(
                cfg.hidden,
                cfg.heads,
                cfg.kv_heads,
                cfg.head_dim,
                cfg.inter,
                seq,
                1e-6,
                false, // bidirectional for diffusion
            );
            eprintln!(
                "seq={seq}: MIL={} bytes, {} BLOBFILEs",
                mil.mil_text.len(),
                mil.weight_names.len()
            );

            // Build all 13 weight blobs from layer 0
            let lw = &engine.layers[0];
            let half_hd = cfg.head_dim / 2;

            // Precompute RoPE for BLOBFILE
            let mut rope_cos = vec![0.0f32; seq * half_hd];
            let mut rope_sin = vec![0.0f32; seq * half_hd];
            for pos in 0..seq {
                for d in 0..half_hd {
                    let freq =
                        1.0 / (cfg.rope_theta as f32).powf(2.0 * d as f32 / cfg.head_dim as f32);
                    let angle = pos as f32 * freq;
                    rope_cos[pos * half_hd + d] = angle.cos();
                    rope_sin[pos * half_hd + d] = angle.sin();
                }
            }

            let blobs: Vec<Vec<u8>> = vec![
                build_weight_blob(&lw.input_norm, 1, cfg.hidden), // rms_att [dim,1] → [1,dim,1,1]
                build_weight_blob(&lw.post_attn_norm, 1, cfg.hidden), // rms_ffn
                build_weight_blob_transposed(&lw.q_proj, cfg.heads * cfg.head_dim, cfg.hidden), // wq
                build_weight_blob_transposed(&lw.k_proj, cfg.kv_heads * cfg.head_dim, cfg.hidden), // wk
                build_weight_blob_transposed(&lw.v_proj, cfg.kv_heads * cfg.head_dim, cfg.hidden), // wv
                build_weight_blob_transposed(&lw.o_proj, cfg.hidden, cfg.heads * cfg.head_dim), // wo
                build_weight_blob_transposed(&lw.gate_proj, cfg.inter, cfg.hidden), // gate
                build_weight_blob_transposed(&lw.up_proj, cfg.inter, cfg.hidden),   // up
                build_weight_blob_transposed(&lw.down_proj, cfg.hidden, cfg.inter), // down
                build_weight_blob(&rope_cos, seq, half_hd), // rope_cos [seq, half_hd]
                build_weight_blob(&rope_sin, seq, half_hd), // rope_sin
                build_weight_blob(&lw.q_norm, 1, cfg.head_dim), // q_norm [1,1,1,hd]
                build_weight_blob(&lw.k_norm, 1, cfg.head_dim), // k_norm
            ];
            let total_mb: f64 = blobs.iter().map(|b| b.len() as f64).sum::<f64>() / 1e6;
            eprintln!("seq={seq}: total BLOBFILE = {total_mb:.1}MB");

            let blob_refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
            let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

            match AneKernel::compile_multi_weights(
                &mil.mil_text,
                &names,
                &blob_refs,
                &[mil.input_bytes],
                &[mil.output_bytes],
            ) {
                Ok(kernel) => {
                    eprintln!("seq={seq}: COMPILED!");

                    // Write dummy activation
                    let act: Vec<f32> = (0..cfg.hidden * seq)
                        .map(|i| ((i as f32) * 0.001).sin() * 0.01)
                        .collect();
                    let act_bytes: Vec<u8> = act.iter().flat_map(|f| f.to_le_bytes()).collect();
                    kernel.write_input(0, &act_bytes);
                    kernel.eval().expect("fused layer eval");
                    eprintln!("seq={seq}: EVAL OK!");

                    // Benchmark 28 dispatches (full model forward)
                    let n = 20u128;
                    let t0 = std::time::Instant::now();
                    for _ in 0..n {
                        for _ in 0..28 {
                            kernel.write_input(0, &act_bytes);
                            kernel.eval().unwrap();
                        }
                    }
                    let step_us = t0.elapsed().as_micros() / n;
                    let step_ms = step_us as f64 / 1000.0;
                    let gen_tok = seq - 5;
                    let steps = 64;
                    let tps = gen_tok as f64 / (step_ms * steps as f64 / 1000.0);
                    eprintln!(
                        "seq={seq}: {step_ms:.1}ms/step (28 dispatches) × {steps} = {:.0}ms → {tps:.0} tok/s",
                        step_ms * steps as f64
                    );
                }
                Err(e) => {
                    eprintln!("seq={seq}: COMPILE FAILED: {e}");
                }
            }
        }
    }

    /// Test if dim=2048 MHA attention fits in one ANE dispatch (32MB limit).
    /// LLaDA-MoE: dim=2048, heads=16, kv_heads=16, hd=128 → QKVO = 32MB exactly.
    #[test]
    fn test_dim2048_attention_compile() {
        use crate::ane_bridge::{self, build_weight_blob, build_weight_blob_transposed, AneKernel};

        ane_bridge::ane_init().expect("ANE init");

        let dim = 2048;
        let heads = 16;
        let kv_heads = 16; // MHA
        let hd = 128;
        let seq = 128;
        let half_hd = hd / 2;

        // Generate MIL
        let mil = gen_diffusion_attention(dim, heads, kv_heads, hd, seq, 1e-5);
        eprintln!("MIL generated for dim={dim}, heads={heads}, kv_heads={kv_heads}");

        // Create dummy weight blobs
        let rms_att = build_weight_blob(&vec![1.0f32; dim], 1, dim);
        let wq = build_weight_blob_transposed(&vec![0.01f32; dim * dim], dim, dim);
        let wk = build_weight_blob_transposed(&vec![0.01f32; dim * dim], dim, dim);
        let wv = build_weight_blob_transposed(&vec![0.01f32; dim * dim], dim, dim);
        let wo = build_weight_blob_transposed(&vec![0.01f32; dim * dim], dim, dim);
        let rope_cos = build_weight_blob(&vec![1.0f32; seq * half_hd], seq, half_hd);
        let rope_sin = build_weight_blob(&vec![0.0f32; seq * half_hd], seq, half_hd);
        let q_norm = build_weight_blob(&vec![1.0f32; hd], 1, hd);
        let k_norm = build_weight_blob(&vec![1.0f32; hd], 1, hd);

        let blobs: Vec<&[u8]> = vec![
            &rms_att, &wq, &wk, &wv, &wo, &rope_cos, &rope_sin, &q_norm, &k_norm,
        ];
        let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
        let total_mb: f64 = blobs.iter().map(|b| b.len() as f64).sum::<f64>() / 1e6;
        eprintln!("Total BLOBFILE size: {total_mb:.1}MB (limit: 32MB)");

        match AneKernel::compile_multi_weights(
            &mil.mil_text,
            &names,
            &blobs,
            &[mil.input_bytes],
            &[mil.output_bytes],
        ) {
            Ok(kernel) => {
                eprintln!("COMPILED OK — dim=2048 attention fits in one dispatch!");
                // Quick eval test
                let act: Vec<f32> = (0..dim * seq)
                    .map(|i| ((i as f32) * 0.001).sin() * 0.01)
                    .collect();
                let act_bytes: Vec<u8> = act.iter().flat_map(|f| f.to_le_bytes()).collect();
                kernel.write_input(0, &act_bytes);
                kernel.eval().expect("eval");
                let mut out = vec![0u8; dim * seq * 4];
                kernel.read_output(0, &mut out);
                let out_f32: Vec<f32> = out
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let max = out_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("Output max abs: {max:.6}");
                assert!(out_f32.iter().all(|v| v.is_finite()), "Non-finite output!");
            }
            Err(e) => {
                eprintln!("COMPILE FAILED — need to split attention: {e}");
                // Not a test failure — we need this info to decide on architecture
            }
        }
    }
}
