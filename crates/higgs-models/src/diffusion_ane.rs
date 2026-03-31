//! Fully-fused ANE layer kernel for diffusion LM (Qwen3-0.6B MDLM).
//!
//! ONE ANE dispatch per transformer layer: RMSNorm → QKV → QK-norm → RoPE → SDPA → Wo → residual → RMSNorm → FFN → residual.
//! 28 dispatches for the full model. Input: hidden [dim, seq] fp32. Output: hidden [dim, seq] fp32.
//!
//! Differences from nanobot-rs gen_fused_layer_fwd:
//! - **Bidirectional attention** (no causal mask) — simpler, no mask BLOBFILE
//! - **QK normalization** (Qwen3-specific) — extra per-head RMSNorm after Q/K projections
//! - **GQA 2:1** (16 Q heads, 8 KV heads)
//!
//! Weight BLOBFILEs (13 total, ~30MB fp16 for 0.6B model):
//!   rms_att, rms_ffn, wq, wk, wv, wo, gate_proj, up_proj, down_proj, rope_cos, rope_sin, q_norm, k_norm

#![cfg(feature = "ane")]

use std::fmt::Write;
use crate::ane_mil::{MIL_HEADER, ANE_MIN_SPATIAL, FusedMil};

/// Generate a fully-fused single transformer layer for bidirectional diffusion.
///
/// Input:  `[1, dim, 1, seq]` fp32
/// Output: `[1, dim, 1, seq]` fp32 (hidden state after full layer)
///
/// 13 BLOBFILE weights. ~30MB at dim=1024 (fits under 32MB ANE limit).
pub fn gen_fused_diffusion_layer(
    dim: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    inter: usize, // MLP intermediate size
    seq_len: usize,
    eps: f64,
) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let half_hd = hd / 2;
    let kv_dim = kv_heads * hd;
    let attn_dim = heads * hd;
    let hpg = heads / kv_heads; // heads per group for GQA
    let sc = 1.0 / (hd as f64).sqrt();

    let mut m = String::with_capacity(32768);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{dim},1,{seq}]> x) {{");

    // --- Shared constants ---
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        bool bT = const()[name=string(\"bT\"), val=bool(true)];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool kd = const()[name=string(\"kd\"), val=bool(true)];");
    let _ = writeln!(m, "        tensor<int32, [1]> ch_ax = const()[name=string(\"chax\"), val=tensor<int32, [1]>([1])];");
    let _ = writeln!(m, "        fp16 eps_v = const()[name=string(\"epsv\"), val=fp16({eps})];");
    let _ = writeln!(m, "        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];");

    // Cast input
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");

    // === RMSNorm (attention) ===
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> rn1_sq = mul(x=xh,y=xh)[name=string(\"rn1sq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rn1_m = reduce_mean(x=rn1_sq,axes=ch_ax,keep_dims=kd)[name=string(\"rn1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rn1_e = add(x=rn1_m,y=eps_v)[name=string(\"rn1e\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rn1_r = pow(x=rn1_e,y=nhalf)[name=string(\"rn1r\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> rn1_n = mul(x=xh,y=rn1_r)[name=string(\"rn1n\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,1]> rn1_w = const()[name=string(\"rn1w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_att.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xnorm = mul(x=rn1_n,y=rn1_w)[name=string(\"xnorm\")];");

    // === QKV Projections ===
    let _ = writeln!(m, "        tensor<int32, [4]> r2d = const()[name=string(\"r2d\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> xn2 = reshape(shape=r2d,x=xnorm)[name=string(\"xn2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{attn_dim}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{dim},{attn_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{kv_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{dim},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{dim}]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [1,1,{attn_dim},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq)[name=string(\"qm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk)[name=string(\"km\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv)[name=string(\"vm\")];");

    // Reshape Q → [1, heads, seq, hd]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,{heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> q4 = reshape(shape=qsh,x=qt)[name=string(\"rq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];");

    // Reshape K → [1, kv_heads, seq, hd]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> kvsh = const()[name=string(\"kvsh\"), val=tensor<int32, [4]>([1,{kv_heads},{hd},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> k4 = reshape(shape=kvsh,x=kt)[name=string(\"rk\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_kv = transpose(perm=pm,x=k4)[name=string(\"tk\")];");

    // Reshape V → [1, kv_heads, seq, hd]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{hd},{seq}]> v4 = reshape(shape=kvsh,x=vt)[name=string(\"rv\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> v_kv = transpose(perm=pm,x=v4)[name=string(\"tv\")];");

    // === QK Norm (Qwen3-specific: per-head RMSNorm on Q and K) ===
    let _ = writeln!(m, "        tensor<int32, [1]> hd_ax = const()[name=string(\"hdax\"), val=tensor<int32, [1]>([-1])];");
    // Q norm
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q_sq = mul(x=q,y=q)[name=string(\"qsq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> qn_m = reduce_mean(x=q_sq,axes=hd_ax,keep_dims=kd)[name=string(\"qnm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> qn_e = add(x=qn_m,y=eps_v)[name=string(\"qne\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},1]> qn_r = pow(x=qn_e,y=nhalf)[name=string(\"qnr\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> qn = mul(x=q,y=qn_r)[name=string(\"qn\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{hd}]> qn_w = const()[name=string(\"qnw\"), val=tensor<fp16, [1,1,1,{hd}]>(BLOBFILE(path=string(\"@model_path/weights/q_norm.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q_normed = mul(x=qn,y=qn_w)[name=string(\"qnormed\")];");
    // K norm
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_sq = mul(x=k_kv,y=k_kv)[name=string(\"ksq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_m = reduce_mean(x=k_sq,axes=hd_ax,keep_dims=kd)[name=string(\"knm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_e = add(x=kn_m,y=eps_v)[name=string(\"kne\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},1]> kn_r = pow(x=kn_e,y=nhalf)[name=string(\"knr\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> kn = mul(x=k_kv,y=kn_r)[name=string(\"kn\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{hd}]> kn_w = const()[name=string(\"knw\"), val=tensor<fp16, [1,1,1,{hd}]>(BLOBFILE(path=string(\"@model_path/weights/k_norm.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_normed = mul(x=kn,y=kn_w)[name=string(\"knormed\")];");

    // === RoPE ===
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{half_hd}]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,{seq},{half_hd}]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];");

    // Q RoPE
    let _ = writeln!(m, "        tensor<int32, [4]> rp_b0 = const()[name=string(\"rpb0\"), val=tensor<int32, [4]>([0,0,0,0])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rp_sh = const()[name=string(\"rpsh\"), val=tensor<int32, [4]>([1,{heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1 = slice_by_size(x=q_normed,begin=rp_b0,size=rp_sh)[name=string(\"q1\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rp_bh = const()[name=string(\"rpbh\"), val=tensor<int32, [4]>([0,0,0,{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2 = slice_by_size(x=q_normed,begin=rp_bh,size=rp_sh)[name=string(\"q2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1c = mul(x=q1,y=rope_cos)[name=string(\"q1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2s = mul(x=q2,y=rope_sin)[name=string(\"q2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr1 = sub(x=q1c,y=q2s)[name=string(\"qr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q1s = mul(x=q1,y=rope_sin)[name=string(\"q1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> q2c = mul(x=q2,y=rope_cos)[name=string(\"q2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{half_hd}]> qr2 = add(x=q1s,y=q2c)[name=string(\"qr2\")];");
    let _ = writeln!(m, "        int32 rpax = const()[name=string(\"rpax\"), val=int32(-1)];");
    let _ = writeln!(m, "        bool rpid = const()[name=string(\"rpid\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> q_rot = concat(axis=rpax,interleave=rpid,values=(qr1,qr2))[name=string(\"qrot\")];");

    // K RoPE
    let _ = writeln!(m, "        tensor<int32, [4]> rp_ksh = const()[name=string(\"rpksh\"), val=tensor<int32, [4]>([1,{kv_heads},{seq},{half_hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1 = slice_by_size(x=k_normed,begin=rp_b0,size=rp_ksh)[name=string(\"k1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2 = slice_by_size(x=k_normed,begin=rp_bh,size=rp_ksh)[name=string(\"k2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1c = mul(x=k1,y=rope_cos)[name=string(\"k1c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2s = mul(x=k2,y=rope_sin)[name=string(\"k2s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> kr1 = sub(x=k1c,y=k2s)[name=string(\"kr1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k1s = mul(x=k1,y=rope_sin)[name=string(\"k1s\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> k2c = mul(x=k2,y=rope_cos)[name=string(\"k2c\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{half_hd}]> kr2 = add(x=k1s,y=k2c)[name=string(\"kr2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_heads},{seq},{hd}]> k_rot = concat(axis=rpax,interleave=rpid,values=(kr1,kr2))[name=string(\"krot\")];");

    // === GQA Bidirectional SDPA (NO causal mask) ===
    // Q: [1,H,S,hd] → [kvH, hpg, S, hd]
    // K: [1,kvH,S,hd] → [kvH, 1, S, hd]
    let _ = writeln!(m, "        tensor<int32, [4]> rqb = const()[name=string(\"rqb\"), val=tensor<int32, [4]>([{kv_heads},{hpg},{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> qb = reshape(shape=rqb,x=q_rot)[name=string(\"qb\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rkb = const()[name=string(\"rkb\"), val=tensor<int32, [4]>([{kv_heads},1,{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> kb = reshape(shape=rkb,x=k_rot)[name=string(\"kb\")];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},1,{seq},{hd}]> vb = reshape(shape=rkb,x=v_kv)[name=string(\"vb\")];");

    // Q@K^T scaled → softmax (BIDIRECTIONAL — no mask!)
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=qb,y=kb)[name=string(\"mm1\")];");
    let _ = writeln!(m, "        fp16 scv = const()[name=string(\"scv\"), val=fp16({sc})];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];");
    let _ = writeln!(m, "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];");
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{seq}]> aw = softmax(axis=sax,x=sc2)[name=string(\"sm\")];");

    // scores@V
    let _ = writeln!(m, "        tensor<fp16, [{kv_heads},{hpg},{seq},{hd}]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=vb)[name=string(\"mm2\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rha = const()[name=string(\"rha\"), val=tensor<int32, [4]>([1,{heads},{seq},{hd}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{seq},{hd}]> a_out = reshape(shape=rha,x=a4)[name=string(\"aout\")];");

    // Reshape attn output → [1, attn_dim, 1, seq]
    let _ = writeln!(m, "        tensor<fp16, [1,{heads},{hd},{seq}]> at = transpose(perm=pm,x=a_out)[name=string(\"ta\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> osa = const()[name=string(\"osa\"), val=tensor<int32, [4]>([1,{attn_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{attn_dim},1,{seq}]> af = reshape(shape=osa,x=at)[name=string(\"ra\")];");

    // === Wo projection ===
    let _ = writeln!(m, "        tensor<int32, [4]> r2a = const()[name=string(\"r2a\"), val=tensor<int32, [4]>([1,1,{attn_dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{attn_dim},{seq}]> af2 = reshape(shape=r2a,x=af)[name=string(\"af2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{attn_dim}]> aft = transpose(perm=pm,x=af2)[name=string(\"aft\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> om = matmul(transpose_x=bF,transpose_y=bF,x=aft,y=Wo)[name=string(\"om\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> ot = transpose(perm=pm,x=om)[name=string(\"ot\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> oo = reshape(shape=os,x=ot)[name=string(\"oo\")];");

    // === Residual 1 ===
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> x2 = add(x=xh,y=oo)[name=string(\"x2\")];");

    // === RMSNorm (FFN) ===
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> rn2_sq = mul(x=x2,y=x2)[name=string(\"rn2sq\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rn2_m = reduce_mean(x=rn2_sq,axes=ch_ax,keep_dims=kd)[name=string(\"rn2m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rn2_e = add(x=rn2_m,y=eps_v)[name=string(\"rn2e\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,1,{seq}]> rn2_r = pow(x=rn2_e,y=nhalf)[name=string(\"rn2r\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> rn2_n = mul(x=x2,y=rn2_r)[name=string(\"rn2n\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,1]> rn2_w = const()[name=string(\"rn2w\"), val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_ffn.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> x2norm = mul(x=rn2_n,y=rn2_w)[name=string(\"x2norm\")];");

    // === FFN ===
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> fn2 = reshape(shape=r2d,x=x2norm)[name=string(\"fn2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> fnt = transpose(perm=pm,x=fn2)[name=string(\"fnt\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{inter}]> Wg = const()[name=string(\"Wg\"), val=tensor<fp16, [1,1,{dim},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/gate.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{inter}]> Wu = const()[name=string(\"Wu\"), val=tensor<fp16, [1,1,{dim},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/up.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{dim}]> Wd = const()[name=string(\"Wd\"), val=tensor<fp16, [1,1,{inter},{dim}]>(BLOBFILE(path=string(\"@model_path/weights/down.bin\"), offset=uint64(64)))];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{inter}]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=fnt,y=Wg)[name=string(\"h1m\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{inter}]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=fnt,y=Wu)[name=string(\"h3m\")];");

    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{seq}]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{seq}]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,{inter},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];");

    // SiLU(gate) * up
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> sig = sigmoid(x=h1)[name=string(\"sg\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> silu = mul(x=h1,y=sig)[name=string(\"si\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> gated = mul(x=silu,y=h3)[name=string(\"gated\")];");

    // Down projection
    let _ = writeln!(m, "        tensor<int32, [4]> rh2 = const()[name=string(\"rh2\"), val=tensor<int32, [4]>([1,1,{inter},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{seq}]> g2 = reshape(shape=rh2,x=gated)[name=string(\"g2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{inter}]> g2t = transpose(perm=pm,x=g2)[name=string(\"g2t\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> fm = matmul(transpose_x=bF,transpose_y=bF,x=g2t,y=Wd)[name=string(\"fm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> ft = transpose(perm=pm,x=fm)[name=string(\"ft\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> ffn_out = reshape(shape=os,x=ft)[name=string(\"ffn\")];");

    // === Residual 2 ===
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xout = add(x=x2,y=ffn_out)[name=string(\"xout\")];");

    // Output
    let _ = writeln!(m, "        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=xout)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec![
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
        if std::path::Path::new(&dir).join("model.safetensors").exists() { Some(dir) } else { None }
    }

    /// THE BIG TEST: compile + eval a fully-fused layer on ANE, benchmark 28 dispatches.
    #[test]
    fn test_fused_layer_compile_and_benchmark() {
        use crate::ane_bridge::{self, AneKernel, build_weight_blob, build_weight_blob_transposed};
        use crate::diffusion::DiffusionEngine;

        let Some(dir) = model_dir() else { eprintln!("Model not found"); return; };
        ane_bridge::ane_init().expect("ANE init");
        ane_bridge::set_quiet(false);

        let engine = DiffusionEngine::load(&dir).unwrap();
        let cfg = &engine.config;

        for seq in [64, 128] {
            let mil = gen_fused_diffusion_layer(
                cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.inter, seq, 1e-6,
            );
            eprintln!("seq={seq}: MIL={} bytes, {} BLOBFILEs", mil.mil_text.len(), mil.weight_names.len());

            // Build all 13 weight blobs from layer 0
            let lw = &engine.layers[0];
            let half_hd = cfg.head_dim / 2;

            // Precompute RoPE for BLOBFILE
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

            let blobs: Vec<Vec<u8>> = vec![
                build_weight_blob(&lw.input_norm, 1, cfg.hidden),           // rms_att [dim,1] → [1,dim,1,1]
                build_weight_blob(&lw.post_attn_norm, 1, cfg.hidden),       // rms_ffn
                build_weight_blob_transposed(&lw.q_proj, cfg.heads * cfg.head_dim, cfg.hidden),  // wq
                build_weight_blob_transposed(&lw.k_proj, cfg.kv_heads * cfg.head_dim, cfg.hidden), // wk
                build_weight_blob_transposed(&lw.v_proj, cfg.kv_heads * cfg.head_dim, cfg.hidden), // wv
                build_weight_blob_transposed(&lw.o_proj, cfg.hidden, cfg.heads * cfg.head_dim),    // wo
                build_weight_blob_transposed(&lw.gate_proj, cfg.inter, cfg.hidden),  // gate
                build_weight_blob_transposed(&lw.up_proj, cfg.inter, cfg.hidden),    // up
                build_weight_blob_transposed(&lw.down_proj, cfg.hidden, cfg.inter),  // down
                build_weight_blob(&rope_cos, seq, half_hd),                 // rope_cos [seq, half_hd]
                build_weight_blob(&rope_sin, seq, half_hd),                 // rope_sin
                build_weight_blob(&lw.q_norm, 1, cfg.head_dim),            // q_norm [1,1,1,hd]
                build_weight_blob(&lw.k_norm, 1, cfg.head_dim),            // k_norm
            ];
            let total_mb: f64 = blobs.iter().map(|b| b.len() as f64).sum::<f64>() / 1e6;
            eprintln!("seq={seq}: total BLOBFILE = {total_mb:.1}MB");

            let blob_refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
            let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

            match AneKernel::compile_multi_weights(
                &mil.mil_text, &names, &blob_refs,
                &[mil.input_bytes], &[mil.output_bytes],
            ) {
                Ok(kernel) => {
                    eprintln!("seq={seq}: COMPILED!");

                    // Write dummy activation
                    let act: Vec<f32> = (0..cfg.hidden * seq).map(|i| ((i as f32) * 0.001).sin() * 0.01).collect();
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
                    eprintln!("seq={seq}: {step_ms:.1}ms/step (28 dispatches) × {steps} = {:.0}ms → {tps:.0} tok/s",
                        step_ms * steps as f64);
                }
                Err(e) => {
                    eprintln!("seq={seq}: COMPILE FAILED: {e}");
                }
            }
        }
    }
}
