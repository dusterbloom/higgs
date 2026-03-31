//! MIL (Model Intermediate Language) program generators for ANE.
//!
//! Generates MIL program text strings that get compiled to ANE binaries.
//! Only contains generators needed for RWKV-7 inference.
//!
//! Feature-gated behind `ane`.

#![cfg(feature = "ane")]

use std::fmt::Write;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// ANE SRAM capacity in fp16 elements (~28 MB).
/// When a single kernel's working set exceeds this, throughput drops ~30%.
const ANE_SRAM_FP16_ELEMS: usize = 14_000_000;

/// ANE hardware minimum spatial dimension. Programs with spatial < 16
/// compile OK but fail at eval with `status=0x1d` (Program Inference error).
/// All BLOBFILE-weighted ops (conv, matmul) require spatial >= 16.
pub const ANE_MIN_SPATIAL: usize = 16;

/// MIL program header (version 1.3, matching CoreML toolchain on macOS 15+).
pub const MIL_HEADER: &str = concat!(
    "program(1.3)\n",
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
    "{\"coremlc-version\", \"3505.4.1\"}, ",
    "{\"coremltools-component-milinternal\", \"\"}, ",
    "{\"coremltools-version\", \"9.0\"}})]\n",
    "{\n",
);

// ---------------------------------------------------------------------------
// SRAM tiling
// ---------------------------------------------------------------------------

/// Plan for tiling a large matmul to fit in ANE SRAM.
#[derive(Debug, Clone)]
pub struct TilePlan {
    pub tile_size: usize,
    pub n_tiles: usize,
    pub last_actual: usize,
    pub full_size: usize,
}

impl TilePlan {
    pub fn needs_tiling(&self) -> bool {
        self.n_tiles > 1
    }
}

/// Compute output-channel tiling so each tile fits in SRAM.
///
/// Layout: `[1, channels, 1, spatial]` where `spatial = seq_len + tile_oc`.
/// Constraint: `ic * (seq_len + tile_oc) <= ANE_SRAM_FP16_ELEMS`.
pub fn compute_oc_tile_plan(ic: usize, oc: usize, seq_len: usize) -> TilePlan {
    let budget = ANE_SRAM_FP16_ELEMS / ic;
    if budget <= seq_len {
        // Even a single column doesn't fit; fall back to full size.
        return TilePlan {
            tile_size: oc,
            n_tiles: 1,
            last_actual: oc,
            full_size: oc,
        };
    }
    let max_tile = budget - seq_len;
    // Round down to multiple of 128 for ANE alignment.
    let tile = (max_tile / 128) * 128;
    let tile = tile.max(128).min(oc);

    let n_tiles = (oc + tile - 1) / tile;
    let last = oc - (n_tiles - 1) * tile;

    TilePlan {
        tile_size: tile,
        n_tiles,
        last_actual: last,
        full_size: oc,
    }
}

/// Compute input-channel (reduction dimension) tiling.
pub fn compute_ic_tile_plan(ic: usize, oc: usize, seq_len: usize) -> TilePlan {
    let budget = ANE_SRAM_FP16_ELEMS / (seq_len + oc);
    let tile = (budget / 128) * 128;
    let tile = tile.max(128).min(ic);

    let n_tiles = (ic + tile - 1) / tile;
    let last = ic - (n_tiles - 1) * tile;

    TilePlan {
        tile_size: tile,
        n_tiles,
        last_actual: last,
        full_size: ic,
    }
}

// ---------------------------------------------------------------------------
// MIL config
// ---------------------------------------------------------------------------

/// Configuration for generating RWKV-7 MIL programs.
#[derive(Debug, Clone)]
pub struct MilConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub seq_len: usize,
}

// ---------------------------------------------------------------------------
// MIL generators
// ---------------------------------------------------------------------------

/// Generate a split-input dynamic matmul MIL: `y = act @ W`.
///
/// Two separate IOSurfaces (ported from nanobot-rs `gen_dyn_matmul_split_mil`):
/// - Input 0: `[1, ic, 1, seq]` fp32 — activations (zero-padded to ANE_MIN_SPATIAL)
/// - Input 1: `[1, ic, 1, oc]` fp32 — weights (written once, reused every step)
/// - Output:  `[1, oc, 1, seq]` fp32 (only position 0 is valid for decode)
///
/// `seq` is padded up to `ANE_MIN_SPATIAL` (16) if smaller — ANE hardware
/// rejects spatial < 16 at eval time (0x1d).
///
/// Returns `(mil_text, act_bytes, weight_bytes, output_bytes)`.
pub fn gen_matmul_program(ic: usize, oc: usize, seq_len: usize) -> (String, usize, usize, usize) {
    let seq = seq_len.max(ANE_MIN_SPATIAL); // pad to ANE minimum
    let act_bytes = ic * seq * 4; // fp32
    let weight_bytes = ic * oc * 4; // fp32
    let output_bytes = oc * seq * 4; // fp32

    let mut m = String::with_capacity(4096);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{ic},1,{seq}]> act, tensor<fp32, [1,{ic},1,{oc}]> w) {{");
    // Cast to fp16
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{ic},1,{seq}]> ah = cast(dtype=to16,x=act)[name=string(\"cin_a\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{ic},1,{oc}]> wh = cast(dtype=to16,x=w)[name=string(\"cin_w\")];");
    // Reshape act: [1,ic,1,seq] → [1,1,ic,seq]
    let _ = writeln!(m, "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra,x=ah)[name=string(\"a2\")];");
    // Transpose act: [1,1,ic,seq] → [1,1,seq,ic]
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];");
    // Reshape weight: [1,ic,1,oc] → [1,1,ic,oc]
    let _ = writeln!(m, "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw,x=wh)[name=string(\"W\")];");
    // matmul: [1,1,seq,ic] @ [1,1,ic,oc] → [1,1,seq,oc]
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];");
    // Transpose back: [1,1,seq,oc] → [1,1,oc,seq]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];");
    // Reshape: [1,1,oc,seq] → [1,oc,1,seq]
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{oc},1,{seq}]> yf = reshape(shape=ro,x=yt)[name=string(\"yf\")];");
    // Cast back to fp32
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{oc},1,{seq}]> y = cast(dtype=to32,x=yf)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (y);");
    m.push_str("}\n");

    (m, act_bytes, weight_bytes, output_bytes)
}

/// Generate a MIL program for element-wise sigmoid: `y = sigmoid(x)`.
pub fn gen_sigmoid_program(dim: usize, seq_len: usize) -> (String, usize, usize) {
    let bytes = dim * seq_len * 2;

    let mut mil = String::with_capacity(1024);
    write!(
        mil,
        r#"{MIL_HEADER}    %input: tensor<[1, {dim}, 1, {seq_len}], fp16>
  ) {{
    %out = sigmoid(x = %input);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, bytes, bytes)
}

/// Generate a MIL program for element-wise multiply: `y = a * b`.
pub fn gen_multiply_program(dim: usize, seq_len: usize) -> (String, usize, usize) {
    let bytes = dim * seq_len * 2;

    let mut mil = String::with_capacity(1024);
    write!(
        mil,
        r#"{MIL_HEADER}    %a: tensor<[1, {dim}, 1, {seq_len}], fp16>,
    %b: tensor<[1, {dim}, 1, {seq_len}], fp16>
  ) {{
    %out = mul(x = %a, y = %b);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    // Two inputs, one output.
    (mil, bytes * 2, bytes)
}

/// Generate a MIL program for element-wise add: `y = a + b`.
pub fn gen_add_program(dim: usize, seq_len: usize) -> (String, usize, usize) {
    let bytes = dim * seq_len * 2;

    let mut mil = String::with_capacity(1024);
    write!(
        mil,
        r#"{MIL_HEADER}    %a: tensor<[1, {dim}, 1, {seq_len}], fp16>,
    %b: tensor<[1, {dim}, 1, {seq_len}], fp16>
  ) {{
    %out = add(x = %a, y = %b);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, bytes * 2, bytes)
}

/// Generate a MIL program for SiLU (SiGLU without gate): `y = x * sigmoid(x)`.
pub fn gen_silu_program(dim: usize, seq_len: usize) -> (String, usize, usize) {
    let bytes = dim * seq_len * 2;

    let mut mil = String::with_capacity(1024);
    write!(
        mil,
        r#"{MIL_HEADER}    %input: tensor<[1, {dim}, 1, {seq_len}], fp16>
  ) {{
    %sig = sigmoid(x = %input);
    %out = mul(x = %input, y = %sig);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, bytes, bytes)
}

/// Generate a MIL program for squared ReLU: `y = relu(x)^2`.
pub fn gen_sqrelu_program(dim: usize, seq_len: usize) -> (String, usize, usize) {
    let bytes = dim * seq_len * 2;

    let mut mil = String::with_capacity(1024);
    write!(
        mil,
        r#"{MIL_HEADER}    %input: tensor<[1, {dim}, 1, {seq_len}], fp16>
  ) {{
    %relu = relu(x = %input);
    %out = mul(x = %relu, y = %relu);
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, bytes, bytes)
}

/// Generate a MIL program for LayerNorm.
///
/// Uses ANE's built-in `layer_norm` op with weight/bias as constexpr.
/// Weight and bias are packed into the IOSurface alongside activations.
pub fn gen_layer_norm_program(dim: usize, seq_len: usize, eps: f32) -> (String, usize, usize) {
    // Input layout: [1, dim, 1, seq_len + 2*dim] where:
    //   [0:seq_len] = activations
    //   [seq_len:seq_len+dim] = gamma (weight)
    //   [seq_len+dim:seq_len+2*dim] = beta (bias)
    let spatial = seq_len + 2 * dim;
    let input_bytes = dim * spatial * 2;
    let output_bytes = dim * seq_len * 2;

    let gamma_start = seq_len;
    let beta_start = seq_len + dim;

    let mut mil = String::with_capacity(2048);
    write!(
        mil,
        r#"{MIL_HEADER}    %input: tensor<[1, {dim}, 1, {spatial}], fp16>
  ) {{
    %act = slice_by_size(x = %input, begin = [0, 0, 0, 0], size = [1, {dim}, 1, {seq_len}]);
    %gamma_raw = slice_by_size(x = %input, begin = [0, 0, 0, {gamma_start}], size = [1, {dim}, 1, 1]);
    %beta_raw = slice_by_size(x = %input, begin = [0, 0, 0, {beta_start}], size = [1, {dim}, 1, 1]);
    %gamma = reshape(x = %gamma_raw, shape = [{dim}]);
    %beta = reshape(x = %beta_raw, shape = [{dim}]);
    %out = layer_norm(x = %act, axes = [1], gamma = %gamma, beta = %beta, epsilon = {eps});
  }} -> (%out)
}}
"#
    )
    .unwrap();

    (mil, input_bytes, output_bytes)
}

// ---------------------------------------------------------------------------
// BLOBFILE-weighted MIL generators (weights baked into compiled kernel)
// ---------------------------------------------------------------------------

/// Result of a fused MIL generation — carries the MIL text + weight file names.
pub struct FusedMil {
    pub mil_text: String,
    pub weight_names: Vec<String>,
    pub input_bytes: usize,
    pub output_bytes: usize,
}

/// Generate a BLOBFILE-weighted matmul: `y = x @ W` where W is a compiled constant.
///
/// - Input:  `[1, ic, 1, seq]` fp32
/// - Output: `[1, oc, 1, seq]` fp32
/// - Weight: `[1, 1, ic, oc]` fp16 BLOBFILE (compiled into kernel, zero runtime I/O)
///
/// `seq` is padded to `ANE_MIN_SPATIAL` if smaller.
pub fn gen_blobfile_matmul(ic: usize, oc: usize, seq_len: usize, name: &str) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let weight_file = format!("@model_path/weights/{name}.bin");

    let mut m = String::with_capacity(4096);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{ic},1,{seq}]> x) {{");
    // Constants
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    // BLOBFILE weight
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{oc}]> W = const()[name=string(\"W\"), val=tensor<fp16, [1,1,{ic},{oc}]>(BLOBFILE(path=string(\"{weight_file}\"), offset=uint64(64)))];");
    // Cast + reshape for matmul
    let _ = writeln!(m, "        tensor<fp16, [1,{ic},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ri = const()[name=string(\"ri\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{seq}]> x2 = reshape(shape=ri,x=xh)[name=string(\"x2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{ic}]> xt = transpose(perm=pm,x=x2)[name=string(\"xt\")];");
    // matmul: [1,1,seq,ic] @ [1,1,ic,oc] → [1,1,seq,oc]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=W)[name=string(\"yh\")];");
    // Transpose + reshape back to [1,oc,1,seq]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{oc},1,{seq}]> yf = reshape(shape=ro,x=yt)[name=string(\"yf\")];");
    // Cast back to fp32
    let _ = writeln!(m, "        tensor<fp32, [1,{oc},1,{seq}]> out = cast(dtype=to32,x=yf)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec![weight_file],
        input_bytes: ic * seq * 4,
        output_bytes: oc * seq * 4,
    }
}

/// Generate a fused r+k+v BLOBFILE projection: 3 matmuls in one ANE dispatch.
///
/// - Input:  `[1, dim, 1, seq]` fp32 (normed hidden state)
/// - Output: `[1, 3*dim, 1, seq]` fp32 (r|k|v concatenated on channel axis)
/// - Weights: 3 BLOBFILEs `[1,1,ic,oc]` fp16 each
pub fn gen_fused_rkv_proj(dim: usize, key_dim: usize, seq_len: usize) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let out_ch = 3 * key_dim;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{dim},1,{seq}]> x) {{");
    // Constants
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    // 3 BLOBFILE weights
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{key_dim}]> Wr = const()[name=string(\"Wr\"), val=tensor<fp16, [1,1,{dim},{key_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wr.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{key_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{dim},{key_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{key_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{dim},{key_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];");
    // Cast + reshape input
    let _ = writeln!(m, "        tensor<fp16, [1,{dim},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ri = const()[name=string(\"ri\"), val=tensor<int32, [4]>([1,1,{dim},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{dim},{seq}]> x2 = reshape(shape=ri,x=xh)[name=string(\"x2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{dim}]> xt = transpose(perm=pm,x=x2)[name=string(\"xt\")];");
    // 3 matmuls
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{key_dim}]> rm = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wr)[name=string(\"rm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{key_dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wk)[name=string(\"km\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{key_dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wv)[name=string(\"vm\")];");
    // Transpose back + reshape each to [1,key_dim,1,seq]
    let _ = writeln!(m, "        tensor<fp16, [1,1,{key_dim},{seq}]> rt = transpose(perm=pm,x=rm)[name=string(\"rt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{key_dim},{seq}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{key_dim},{seq}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{key_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{key_dim},1,{seq}]> rf = reshape(shape=ro,x=rt)[name=string(\"rf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{key_dim},1,{seq}]> kf = reshape(shape=ro,x=kt)[name=string(\"kf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{key_dim},1,{seq}]> vf = reshape(shape=ro,x=vt)[name=string(\"vf\")];");
    // Concatenate r|k|v on channel axis → [1, 3*key_dim, 1, seq]
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> cat = concat(values=(rf,kf,vf),axis=cax,interleave=bF)[name=string(\"cat\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out = cast(dtype=to32,x=cat)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/wr.bin".to_string(),
            "@model_path/weights/wk.bin".to_string(),
            "@model_path/weights/wv.bin".to_string(),
        ],
        input_bytes: dim * seq * 4,
        output_bytes: out_ch * seq * 4,
    }
}

/// Generate a fused QKV BLOBFILE projection: 3 matmuls in one ANE dispatch.
///
/// Input:  `[1, ic, 1, seq]` fp32
/// Output: `[1, q_dim+kv_dim+kv_dim, 1, seq]` fp32 (Q|K|V concatenated on channel axis)
/// Weights: 3 BLOBFILEs
pub fn gen_fused_qkv_proj(ic: usize, q_dim: usize, kv_dim: usize, seq_len: usize) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let out_ch = q_dim + kv_dim + kv_dim;

    let mut m = String::with_capacity(8192);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{ic},1,{seq}]> x) {{");
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    // 3 BLOBFILE weights
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{q_dim}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{ic},{q_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{kv_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{ic},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{kv_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{ic},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];");
    // Cast + reshape
    let _ = writeln!(m, "        tensor<fp16, [1,{ic},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ri = const()[name=string(\"ri\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{seq}]> x2 = reshape(shape=ri,x=xh)[name=string(\"x2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{ic}]> xt = transpose(perm=pm,x=x2)[name=string(\"xt\")];");
    // 3 matmuls
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{q_dim}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wq)[name=string(\"qm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wk)[name=string(\"km\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wv)[name=string(\"vm\")];");
    // Transpose back + reshape to channel-first
    let _ = writeln!(m, "        tensor<fp16, [1,1,{q_dim},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rq = const()[name=string(\"rq\"), val=tensor<int32, [4]>([1,{q_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rk = const()[name=string(\"rk\"), val=tensor<int32, [4]>([1,{kv_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{q_dim},1,{seq}]> qf = reshape(shape=rq,x=qt)[name=string(\"qf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> kf = reshape(shape=rk,x=kt)[name=string(\"kf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> vf = reshape(shape=rk,x=vt)[name=string(\"vf\")];");
    // Concat Q|K|V on channel axis
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> cat = concat(values=(qf,kf,vf),axis=cax,interleave=bF)[name=string(\"cat\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out = cast(dtype=to32,x=cat)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/wq.bin".to_string(),
            "@model_path/weights/wk.bin".to_string(),
            "@model_path/weights/wv.bin".to_string(),
        ],
        input_bytes: ic * seq * 4,
        output_bytes: out_ch * seq * 4,
    }
}

/// Generate a fused Gate+Up BLOBFILE projection: 2 matmuls in one ANE dispatch.
///
/// Input:  `[1, ic, 1, seq]` fp32
/// Output: `[1, 2*inter, 1, seq]` fp32 (gate|up concatenated)
pub fn gen_fused_gate_up_proj(ic: usize, inter: usize, seq_len: usize) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    let out_ch = 2 * inter;

    let mut m = String::with_capacity(4096);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{ic},1,{seq}]> x) {{");
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{inter}]> Wg = const()[name=string(\"Wg\"), val=tensor<fp16, [1,1,{ic},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/wg.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{inter}]> Wu = const()[name=string(\"Wu\"), val=tensor<fp16, [1,1,{ic},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/wu.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,{ic},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ri = const()[name=string(\"ri\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{ic},{seq}]> x2 = reshape(shape=ri,x=xh)[name=string(\"x2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{ic}]> xt = transpose(perm=pm,x=x2)[name=string(\"xt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{inter}]> gm = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wg)[name=string(\"gm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{inter}]> um = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wu)[name=string(\"um\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{seq}]> gt = transpose(perm=pm,x=gm)[name=string(\"gt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{seq}]> ut = transpose(perm=pm,x=um)[name=string(\"ut\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{inter},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> gf = reshape(shape=ro,x=gt)[name=string(\"gf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> uf = reshape(shape=ro,x=ut)[name=string(\"uf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> cat = concat(values=(gf,uf),axis=cax,interleave=bF)[name=string(\"cat\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out = cast(dtype=to32,x=cat)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    FusedMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/wg.bin".to_string(),
            "@model_path/weights/wu.bin".to_string(),
        ],
        input_bytes: ic * seq * 4,
        output_bytes: out_ch * seq * 4,
    }
}

/// Generate a fully-fused single-layer kernel: QKV + O + Gate+Up + Down.
/// 7 matmuls in ONE ANE dispatch. All weights as BLOBFILE constants (~30MB fp16 for 0.6B).
///
/// Input:  `[1, hidden, 1, seq]` fp32
/// Output: `[1, out_ch, 1, seq]` fp32 — concatenated: Q|K|V | O_placeholder | gate|up | down_placeholder
///
/// For the diffusion forward, we split the output and apply attention/SiLU on CPU between dispatches.
/// BUT — this test verifies we can compile all 7 weights into one kernel.
///
/// Actually, for max dispatch reduction: we fuse QKV (share input) and Gate+Up (share input).
/// O and Down need different inputs, so they must be separate dispatches.
/// Total: QKV_fused + O + GateUp_fused + Down = 4 dispatches/layer.
///
/// To get to 1 dispatch/layer, we'd need to put attention inside the kernel.
/// Testing: can we compile a 7-BLOBFILE kernel that does all projections?
pub fn gen_full_layer_projections(hidden: usize, q_dim: usize, kv_dim: usize, inter: usize, seq_len: usize) -> FusedMil {
    let seq = seq_len.max(ANE_MIN_SPATIAL);
    // Output: q|k|v|gate|up = q_dim + kv_dim + kv_dim + inter + inter
    let out_ch = q_dim + 2 * kv_dim + 2 * inter;

    let mut m = String::with_capacity(16384);
    m.push_str(MIL_HEADER);
    let _ = writeln!(m, "    func main<ios18>(tensor<fp32, [1,{hidden},1,{seq}]> x) {{");
    // Constants
    let _ = writeln!(m, "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];");
    let _ = writeln!(m, "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];");
    let _ = writeln!(m, "        bool bF = const()[name=string(\"bF\"), val=bool(false)];");
    let _ = writeln!(m, "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];");
    // 7 BLOBFILE weights
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{q_dim}]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [1,1,{hidden},{q_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{kv_dim}]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [1,1,{hidden},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{kv_dim}]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1,1,{hidden},{kv_dim}]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{inter}]> Wg = const()[name=string(\"Wg\"), val=tensor<fp16, [1,1,{hidden},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/wg.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{inter}]> Wu = const()[name=string(\"Wu\"), val=tensor<fp16, [1,1,{hidden},{inter}]>(BLOBFILE(path=string(\"@model_path/weights/wu.bin\"), offset=uint64(64)))];");
    // O and Down have different input dims but we include them for the BLOBFILE size test
    let _ = writeln!(m, "        tensor<fp16, [1,1,{q_dim},{hidden}]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [1,1,{q_dim},{hidden}]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{hidden}]> Wd = const()[name=string(\"Wd\"), val=tensor<fp16, [1,1,{inter},{hidden}]>(BLOBFILE(path=string(\"@model_path/weights/wd.bin\"), offset=uint64(64)))];");
    // Cast + reshape input
    let _ = writeln!(m, "        tensor<fp16, [1,{hidden},1,{seq}]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> ri = const()[name=string(\"ri\"), val=tensor<int32, [4]>([1,1,{hidden},{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{hidden},{seq}]> x2 = reshape(shape=ri,x=xh)[name=string(\"x2\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{hidden}]> xt = transpose(perm=pm,x=x2)[name=string(\"xt\")];");
    // 5 matmuls from same input (QKV + gate + up)
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{q_dim}]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wq)[name=string(\"qm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> km = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wk)[name=string(\"km\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{kv_dim}]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wv)[name=string(\"vm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{inter}]> gm = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wg)[name=string(\"gm\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{seq},{inter}]> um = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wu)[name=string(\"um\")];");
    // Transpose + reshape each to channel-first
    let _ = writeln!(m, "        tensor<fp16, [1,1,{q_dim},{seq}]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{kv_dim},{seq}]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{seq}]> gt = transpose(perm=pm,x=gm)[name=string(\"gt\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,1,{inter},{seq}]> ut = transpose(perm=pm,x=um)[name=string(\"ut\")];");
    let _ = writeln!(m, "        tensor<int32, [4]> rq = const()[name=string(\"rq\"), val=tensor<int32, [4]>([1,{q_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rk = const()[name=string(\"rk\"), val=tensor<int32, [4]>([1,{kv_dim},1,{seq}])];");
    let _ = writeln!(m, "        tensor<int32, [4]> rg = const()[name=string(\"rg\"), val=tensor<int32, [4]>([1,{inter},1,{seq}])];");
    let _ = writeln!(m, "        tensor<fp16, [1,{q_dim},1,{seq}]> qf = reshape(shape=rq,x=qt)[name=string(\"qf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> kf = reshape(shape=rk,x=kt)[name=string(\"kf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{kv_dim},1,{seq}]> vf = reshape(shape=rk,x=vt)[name=string(\"vf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> gf = reshape(shape=rg,x=gt)[name=string(\"gf\")];");
    let _ = writeln!(m, "        tensor<fp16, [1,{inter},1,{seq}]> uf = reshape(shape=rg,x=ut)[name=string(\"uf\")];");
    // Concat all on channel axis
    let _ = writeln!(m, "        tensor<fp16, [1,{out_ch},1,{seq}]> cat = concat(values=(qf,kf,vf,gf,uf),axis=cax,interleave=bF)[name=string(\"cat\")];");
    let _ = writeln!(m, "        tensor<fp32, [1,{out_ch},1,{seq}]> out = cast(dtype=to32,x=cat)[name=string(\"cout\")];");
    let _ = writeln!(m, "    }} -> (out);");
    m.push_str("}\n");

    // Total BLOBFILE: q(4MB) + k(2MB) + v(2MB) + o(4MB) + gate(6MB) + up(6MB) + down(6MB) = 30MB at fp16
    FusedMil {
        mil_text: m,
        weight_names: vec![
            "@model_path/weights/wq.bin".to_string(),
            "@model_path/weights/wk.bin".to_string(),
            "@model_path/weights/wv.bin".to_string(),
            "@model_path/weights/wg.bin".to_string(),
            "@model_path/weights/wu.bin".to_string(),
            "@model_path/weights/wo.bin".to_string(),
            "@model_path/weights/wd.bin".to_string(),
        ],
        input_bytes: hidden * seq * 4,
        output_bytes: out_ch * seq * 4,
    }
}

/// Check whether a matmul with given dimensions fits in ANE SRAM.
pub fn fits_in_sram(ic: usize, oc: usize, seq_len: usize) -> bool {
    ic * (seq_len + oc) <= ANE_SRAM_FP16_ELEMS
}

/// Compute the total IOSurface bytes for a DynMatmul kernel.
pub fn dyn_matmul_input_bytes(ic: usize, oc: usize, seq_len: usize) -> usize {
    ic * (seq_len + oc) * 2
}

/// Compute the output bytes for a matmul kernel.
pub fn matmul_output_bytes(oc: usize, seq_len: usize) -> usize {
    oc * seq_len * 2
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, unsafe_code)]
mod tests {
    use super::*;

    /// Test split-input matmul compiles + evals on ANE.
    #[test]
    fn test_split_matmul_compiles_on_ane() {
        use crate::ane_bridge::{self, AneKernel};

        ane_bridge::ane_init().expect("ANE init failed");
        ane_bridge::set_quiet(false);

        // 64x64 matmul (ANE needs dim >= ~16 for channels too)
        let ic = 64usize;
        let oc = 64usize;
        let (mil, act_bytes, weight_bytes, out_bytes) = gen_matmul_program(ic, oc, 1);

        let kernel = AneKernel::compile(
            &mil, None, &[act_bytes, weight_bytes], &[out_bytes],
        ).expect("64x64 compile failed");

        // Identity weight
        let mut w = vec![0.0f32; ic * oc];
        for i in 0..ic { w[i * oc + i] = 1.0; }
        kernel.write_input(1, &w.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>());

        let ps = ANE_MIN_SPATIAL;
        let mut act = vec![0.0f32; ic * ps];
        for c in 0..ic { act[c * ps] = (c + 1) as f32; }
        kernel.write_input(0, &act.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>());

        kernel.eval().expect("64x64 eval failed");

        let mut buf = vec![0u8; out_bytes];
        kernel.read_output(0, &mut buf);
        let out_all: Vec<f32> = buf.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let out: Vec<f32> = (0..oc).map(|c| out_all[c * ps]).collect();
        let max_err = (0..oc).map(|i| ((i+1) as f32 - out[i]).abs()).fold(0.0f32, f32::max);
        eprintln!("Split 64x64 identity max_err: {max_err}");
        assert!(max_err < 1.0);
    }

    /// Test with nanobot-rs dimensions (ic=64, oc=32, seq=16) to match known-working config.
    #[test]
    fn test_nanobot_dims_matmul_eval() {
        use crate::ane_bridge::{self, AneKernel};

        ane_bridge::ane_init().expect("ANE init failed");
        ane_bridge::set_quiet(false);

        let ic = 64usize;
        let oc = 32usize;
        let seq = 16usize;
        let (mil, act_bytes, wt_bytes, out_bytes) = gen_matmul_program(ic, oc, seq);

        let kernel = AneKernel::compile(
            &mil, None, &[act_bytes, wt_bytes], &[out_bytes],
        ).expect("64x32 compile failed");
        eprintln!("64x32x16 compiled OK!");

        // Random-ish data
        let act: Vec<f32> = (0..ic * seq).map(|i| (i as f32 * 0.01).sin()).collect();
        let w: Vec<f32> = (0..ic * oc).map(|i| (i as f32 * 0.007).cos() * 0.1).collect();
        let a_bytes: Vec<u8> = act.iter().flat_map(|f| f.to_le_bytes()).collect();
        let w_bytes: Vec<u8> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &a_bytes);
        kernel.write_input(1, &w_bytes);

        kernel.eval().expect("64x32x16 eval failed");
        eprintln!("64x32x16 eval OK!");

        let mut out_buf = vec![0u8; out_bytes];
        kernel.read_output(0, &mut out_buf);
        let out: Vec<f32> = out_buf.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        eprintln!("First 5 output: {:?}", &out[..5]);
    }

    /// Test 2048x2048 matmul compiles AND evals (the actual model projection size).
    #[test]
    fn test_2048x2048_matmul_compiles_on_ane() {
        use crate::ane_bridge::{self, AneKernel};

        ane_bridge::ane_init().expect("ANE init failed");
        ane_bridge::set_quiet(false);

        let ic = 2048usize;
        let oc = 2048usize;
        let (mil, act_bytes, weight_bytes, out_bytes) = gen_matmul_program(ic, oc, 1);
        eprintln!("act={act_bytes}, wt={weight_bytes}, out={out_bytes}");

        let kernel = AneKernel::compile(
            &mil, None, &[act_bytes, weight_bytes], &[out_bytes],
        ).expect("compile failed");
        eprintln!("2048x2048 matmul compiled OK!");

        // Write identity-ish weights and a test activation
        let mut w = vec![0.0f32; ic * oc];
        // Set diagonal to 1.0 (identity matrix)
        for i in 0..ic.min(oc) {
            w[i * oc + i] = 1.0;
        }
        let w_bytes: Vec<u8> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(1, &w_bytes);

        // Activation: [1.0, 2.0, 3.0, ...]
        let act: Vec<f32> = (0..ic).map(|i| (i + 1) as f32).collect();
        let act_bytes: Vec<u8> = act.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &act_bytes);

        kernel.eval().expect("eval failed");
        eprintln!("2048x2048 eval OK!");

        let mut out_buf = vec![0u8; out_bytes];
        kernel.read_output(0, &mut out_buf);
        let out: Vec<f32> = out_buf.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // With identity weights, output should ≈ input (fp16 rounding)
        let max_err = act.iter().zip(out.iter())
            .map(|(a, o)| (a - o).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Identity matmul max error: {max_err}");
        eprintln!("First 5 out: {:?}", &out[..5]);
        assert!(max_err < 2.0, "Identity matmul error too large: {max_err}");
    }

    // === TDD: BLOBFILE fused projection tests ===

    /// RED→GREEN: Single BLOBFILE matmul — weight baked into compiled kernel.
    #[test]
    fn test_blobfile_matmul_identity() {
        use crate::ane_bridge::{self, AneKernel, build_weight_blob};

        ane_bridge::ane_init().expect("ANE init failed");
        ane_bridge::set_quiet(false);

        let dim = 64usize;
        let seq = ANE_MIN_SPATIAL;

        // Identity weight [dim, dim] — transposed for MIL (row=ic=dim, col=oc=dim)
        let mut w = vec![0.0f32; dim * dim];
        for i in 0..dim { w[i * dim + i] = 1.0; }
        let blob = build_weight_blob(&w, dim, dim);

        let fused = gen_blobfile_matmul(dim, dim, seq, "w0");
        eprintln!("MIL ({} bytes):\n{}", fused.mil_text.len(), &fused.mil_text[..200.min(fused.mil_text.len())]);

        let names: Vec<&str> = fused.weight_names.iter().map(|s| s.as_str()).collect();
        let kernel = AneKernel::compile_multi_weights(
            &fused.mil_text, &names, &[&blob],
            &[fused.input_bytes], &[fused.output_bytes],
        ).expect("BLOBFILE matmul compile failed");

        // Activation: [1.0, 2.0, ..., dim] padded to seq positions
        let mut act = vec![0.0f32; dim * seq];
        for c in 0..dim { act[c * seq] = (c + 1) as f32; }
        let act_bytes: Vec<u8> = act.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &act_bytes);

        kernel.eval().expect("BLOBFILE eval failed");

        let mut out_buf = vec![0u8; fused.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let out_all: Vec<f32> = out_buf.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        // Extract position 0 from each output channel
        let out: Vec<f32> = (0..dim).map(|c| out_all[c * seq]).collect();

        let max_err = (0..dim).map(|i| ((i + 1) as f32 - out[i]).abs()).fold(0.0f32, f32::max);
        eprintln!("BLOBFILE identity max_err: {max_err}");
        eprintln!("First 5: {:?}", &out[..5]);
        assert!(max_err < 1.0, "BLOBFILE identity error too large: {max_err}");
    }

    /// RED→GREEN: Fused r+k+v BLOBFILE projection — 3 matmuls in one ANE dispatch.
    #[test]
    fn test_fused_rkv_blobfile() {
        use crate::ane_bridge::{self, AneKernel, build_weight_blob};

        ane_bridge::ane_init().expect("ANE init failed");
        ane_bridge::set_quiet(false);

        let dim = 64usize;
        let seq = ANE_MIN_SPATIAL;

        // 3 identity weight matrices
        let mut w_id = vec![0.0f32; dim * dim];
        for i in 0..dim { w_id[i * dim + i] = 1.0; }
        let blob_r = build_weight_blob(&w_id, dim, dim);
        let blob_k = build_weight_blob(&w_id, dim, dim);
        let blob_v = build_weight_blob(&w_id, dim, dim);

        let fused = gen_fused_rkv_proj(dim, dim, seq);

        let names: Vec<&str> = fused.weight_names.iter().map(|s| s.as_str()).collect();
        let kernel = AneKernel::compile_multi_weights(
            &fused.mil_text, &names, &[&blob_r, &blob_k, &blob_v],
            &[fused.input_bytes], &[fused.output_bytes],
        ).expect("Fused r+k+v compile failed");

        // Activation: [1..dim] at position 0
        let mut act = vec![0.0f32; dim * seq];
        for c in 0..dim { act[c * seq] = (c + 1) as f32; }
        kernel.write_input(0, &act.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>());

        kernel.eval().expect("Fused r+k+v eval failed");

        let mut out_buf = vec![0u8; fused.output_bytes];
        kernel.read_output(0, &mut out_buf);
        let out_all: Vec<f32> = out_buf.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Output is [1, 3*dim, 1, seq]. Extract r/k/v at position 0.
        let r: Vec<f32> = (0..dim).map(|c| out_all[c * seq]).collect();
        let k: Vec<f32> = (dim..2*dim).map(|c| out_all[c * seq]).collect();
        let v: Vec<f32> = (2*dim..3*dim).map(|c| out_all[c * seq]).collect();

        // All should equal input (identity weights)
        let max_err_r = (0..dim).map(|i| ((i+1) as f32 - r[i]).abs()).fold(0.0f32, f32::max);
        let max_err_k = (0..dim).map(|i| ((i+1) as f32 - k[i]).abs()).fold(0.0f32, f32::max);
        let max_err_v = (0..dim).map(|i| ((i+1) as f32 - v[i]).abs()).fold(0.0f32, f32::max);
        eprintln!("Fused r+k+v identity: r_err={max_err_r}, k_err={max_err_k}, v_err={max_err_v}");
        assert!(max_err_r < 1.0 && max_err_k < 1.0 && max_err_v < 1.0);
    }

    /// Benchmark fused QKV + Gate+Up + individual O + Down at diffusion model dimensions.
    /// This simulates the 4-dispatch-per-layer approach (112 total, under 119 limit).
    #[test]
    fn test_fused_diffusion_layer_benchmark() {
        use crate::ane_bridge::{self, AneKernel, build_weight_blob};

        ane_bridge::ane_init().expect("ANE init failed");

        let hidden = 1024usize;
        let q_dim = 2048usize;
        let kv_dim = 1024usize;
        let inter = 3072usize;
        let n_iters = 50u128;

        eprintln!("\n  Fused diffusion layer: 4 dispatches/layer, Qwen3-0.6B dims");

        for seq in [64, 128] {
            // Compile fused QKV
            let qkv_mil = gen_fused_qkv_proj(hidden, q_dim, kv_dim, seq);
            let wq: Vec<f32> = (0..hidden*q_dim).map(|i| ((i as f32)*1e-5).sin()*0.01).collect();
            let wk: Vec<f32> = (0..hidden*kv_dim).map(|i| ((i as f32)*1e-5).cos()*0.01).collect();
            let wv = wk.clone();
            let bq = build_weight_blob(&wq, q_dim, hidden);
            let bk = build_weight_blob(&wk, kv_dim, hidden);
            let bv = build_weight_blob(&wv, kv_dim, hidden);
            let qkv_names: Vec<&str> = qkv_mil.weight_names.iter().map(|s| s.as_str()).collect();
            let qkv_kern = AneKernel::compile_multi_weights(
                &qkv_mil.mil_text, &qkv_names, &[&bq, &bk, &bv],
                &[qkv_mil.input_bytes], &[qkv_mil.output_bytes],
            ).expect("QKV compile failed");

            // Compile fused Gate+Up
            let gu_mil = gen_fused_gate_up_proj(hidden, inter, seq);
            let wg: Vec<f32> = (0..hidden*inter).map(|i| ((i as f32)*1e-5).sin()*0.01).collect();
            let wu = wg.clone();
            let bg = build_weight_blob(&wg, inter, hidden);
            let bu = build_weight_blob(&wu, inter, hidden);
            let gu_names: Vec<&str> = gu_mil.weight_names.iter().map(|s| s.as_str()).collect();
            let gu_kern = AneKernel::compile_multi_weights(
                &gu_mil.mil_text, &gu_names, &[&bg, &bu],
                &[gu_mil.input_bytes], &[gu_mil.output_bytes],
            ).expect("Gate+Up compile failed");

            // Compile individual O and Down
            let o_mil = gen_blobfile_matmul(q_dim, hidden, seq, "o");
            let wo: Vec<f32> = (0..hidden*q_dim).map(|i| ((i as f32)*1e-5).sin()*0.01).collect();
            let bo = build_weight_blob(&wo, hidden, q_dim);
            let o_names: Vec<&str> = o_mil.weight_names.iter().map(|s| s.as_str()).collect();
            let o_kern = AneKernel::compile_multi_weights(
                &o_mil.mil_text, &o_names, &[&bo],
                &[o_mil.input_bytes], &[o_mil.output_bytes],
            ).expect("O compile failed");

            let d_mil = gen_blobfile_matmul(inter, hidden, seq, "d");
            let wd: Vec<f32> = (0..hidden*inter).map(|i| ((i as f32)*1e-5).cos()*0.01).collect();
            let bd = build_weight_blob(&wd, hidden, inter);
            let d_names: Vec<&str> = d_mil.weight_names.iter().map(|s| s.as_str()).collect();
            let d_kern = AneKernel::compile_multi_weights(
                &d_mil.mil_text, &d_names, &[&bd],
                &[d_mil.input_bytes], &[d_mil.output_bytes],
            ).expect("Down compile failed");

            // Prepare dummy activation
            let act: Vec<f32> = (0..hidden * seq).map(|i| ((i as f32)*0.001).sin()).collect();
            let act_bytes: Vec<u8> = act.iter().flat_map(|f| f.to_le_bytes()).collect();
            // For O: q_dim input
            let act_q: Vec<f32> = (0..q_dim * seq).map(|i| ((i as f32)*0.001).sin()).collect();
            let act_q_bytes: Vec<u8> = act_q.iter().flat_map(|f| f.to_le_bytes()).collect();
            // For Down: inter input
            let act_i: Vec<f32> = (0..inter * seq).map(|i| ((i as f32)*0.001).sin()).collect();
            let act_i_bytes: Vec<u8> = act_i.iter().flat_map(|f| f.to_le_bytes()).collect();

            // Warmup
            qkv_kern.write_input(0, &act_bytes); qkv_kern.eval().unwrap();
            gu_kern.write_input(0, &act_bytes); gu_kern.eval().unwrap();
            o_kern.write_input(0, &act_q_bytes); o_kern.eval().unwrap();
            d_kern.write_input(0, &act_i_bytes); d_kern.eval().unwrap();

            // Benchmark: simulate 28 layers
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                for _layer in 0..28 {
                    qkv_kern.write_input(0, &act_bytes); qkv_kern.eval().unwrap();
                    o_kern.write_input(0, &act_q_bytes); o_kern.eval().unwrap();
                    gu_kern.write_input(0, &act_bytes); gu_kern.eval().unwrap();
                    d_kern.write_input(0, &act_i_bytes); d_kern.eval().unwrap();
                }
            }
            let total_us = t0.elapsed().as_micros() / n_iters;
            let per_step_ms = total_us as f64 / 1000.0;
            let gen_tok = seq - 5;
            let steps = 64;
            let tok_per_sec = gen_tok as f64 / (per_step_ms * steps as f64 / 1000.0);

            eprintln!(
                "  seq={seq}: {per_step_ms:.1}ms/step (112 dispatches) × {steps} = {:.0}ms → {tok_per_sec:.0} tok/s",
                per_step_ms * steps as f64
            );

            // Test eval_chain: batch all 4 dispatches per layer into one chain call
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                for _layer in 0..28 {
                    qkv_kern.write_input(0, &act_bytes);
                    o_kern.write_input(0, &act_q_bytes);
                    gu_kern.write_input(0, &act_bytes);
                    d_kern.write_input(0, &act_i_bytes);
                    AneKernel::eval_chain(&[&qkv_kern, &o_kern, &gu_kern, &d_kern]).unwrap();
                }
            }
            let chain_us = t0.elapsed().as_micros() / n_iters;
            let chain_ms = chain_us as f64 / 1000.0;
            let chain_tps = gen_tok as f64 / (chain_ms * steps as f64 / 1000.0);
            eprintln!(
                "  seq={seq}: {chain_ms:.1}ms/step (28 chains) × {steps} = {:.0}ms → {chain_tps:.0} tok/s [eval_chain]",
                chain_ms * steps as f64
            );
        }
    }

    /// THE BIG ONE: compile a mega-fused 7-weight kernel and benchmark 28 dispatches/step.
    /// This is the path to 280 tok/s.
    #[test]
    fn test_mega_fused_layer_benchmark() {
        use crate::ane_bridge::{self, AneKernel, build_weight_blob};

        ane_bridge::ane_init().expect("ANE init failed");
        ane_bridge::set_quiet(false);

        let hidden = 1024usize;
        let q_dim = 2048usize;
        let kv_dim = 1024usize;
        let inter = 3072usize;

        for seq in [64, 128] {
            let mil = gen_full_layer_projections(hidden, q_dim, kv_dim, inter, seq);
            eprintln!("MIL size: {} bytes, {} weight files", mil.mil_text.len(), mil.weight_names.len());

            // Build all 7 weight blobs
            let mk = |rows: usize, cols: usize| -> Vec<f32> {
                (0..rows*cols).map(|i| ((i as f32)*1e-5).sin()*0.01).collect()
            };
            let blobs: Vec<Vec<u8>> = vec![
                build_weight_blob(&mk(q_dim, hidden), q_dim, hidden),    // wq
                build_weight_blob(&mk(kv_dim, hidden), kv_dim, hidden),  // wk
                build_weight_blob(&mk(kv_dim, hidden), kv_dim, hidden),  // wv
                build_weight_blob(&mk(inter, hidden), inter, hidden),    // wg
                build_weight_blob(&mk(inter, hidden), inter, hidden),    // wu
                build_weight_blob(&mk(hidden, q_dim), hidden, q_dim),    // wo
                build_weight_blob(&mk(hidden, inter), hidden, inter),    // wd
            ];
            let blob_refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
            let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();

            let total_blob_mb: f64 = blobs.iter().map(|b| b.len() as f64).sum::<f64>() / 1e6;
            eprintln!("Total BLOBFILE: {total_blob_mb:.1}MB (limit: ~32MB)");

            match AneKernel::compile_multi_weights(
                &mil.mil_text, &names, &blob_refs,
                &[mil.input_bytes], &[mil.output_bytes],
            ) {
                Ok(kernel) => {
                    eprintln!("seq={seq}: COMPILED! Testing eval...");

                    let act: Vec<f32> = (0..hidden * seq).map(|i| ((i as f32)*0.001).sin()).collect();
                    let act_bytes: Vec<u8> = act.iter().flat_map(|f| f.to_le_bytes()).collect();
                    kernel.write_input(0, &act_bytes);
                    kernel.eval().expect("mega fused eval failed");
                    eprintln!("seq={seq}: EVAL OK!");

                    // Benchmark: 28 dispatches = one full model forward
                    // Use zero-copy write for minimum overhead
                    let n = 50u128;
                    let base = kernel.get_input_base(0);
                    let t0 = std::time::Instant::now();
                    for _ in 0..n {
                        for _ in 0..28 {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    act_bytes.as_ptr(), base, act_bytes.len());
                                // dsb sy — ensure write is visible to ANE DMA
                                #[cfg(target_arch = "aarch64")]
                                std::arch::asm!("dsb sy");
                            }
                            kernel.eval().unwrap();
                        }
                    }
                    let step_us = t0.elapsed().as_micros() / n;
                    let step_ms = step_us as f64 / 1000.0;
                    let n_gen = seq - 5;
                    let steps = 64;
                    let tps = n_gen as f64 / (step_ms * steps as f64 / 1000.0);
                    eprintln!("seq={seq}: {step_ms:.1}ms/step (28 dispatches) × {steps} = {:.0}ms → {tps:.0} tok/s",
                        step_ms * steps as f64);
                }
                Err(e) => {
                    eprintln!("seq={seq}: COMPILE FAILED: {e}");
                    eprintln!("  (Total BLOBFILE {total_blob_mb:.1}MB may exceed 32MB limit)");
                }
            }
        }
    }

    /// Full sweep: BLAS vs ANE vs GPU (MLX Metal) across seq_len for 2048×2048 matmul.
    /// Realistic decode simulation: includes write_input + eval + read_output.
    #[test]
    fn test_blas_vs_ane_blobfile_latency() -> Result<(), Box<dyn std::error::Error>> {
        use crate::ane_bridge::{self, AneKernel, build_weight_blob};

        // cblas_sgemm FFI — Accelerate framework (already linked)
        unsafe extern "C" {
            unsafe fn cblas_sgemm(
                order: i32, transa: i32, transb: i32,
                m: i32, n: i32, k: i32,
                alpha: f32, a: *const f32, lda: i32,
                b: *const f32, ldb: i32,
                beta: f32, c: *mut f32, ldc: i32,
            );
        }

        ane_bridge::ane_init().expect("ANE init failed");

        let n_iters = 100u128;

        for (label, ic, oc) in [
            ("attn proj (2048x2048)", 2048usize, 2048usize),
            ("ffn_key   (2048x8192)", 2048, 8192),
            ("ffn_value (8192x2048)", 8192, 2048),
        ] {
        eprintln!("\n======================================================================");
        eprintln!("  {label}  |  {n_iters} iters, release");
        eprintln!("======================================================================");
        eprintln!("   seq |  BLAS(CPU)           |  ANE(DMA)            |  GPU(Metal)          | best");
        eprintln!("-------+----------------------+----------------------+----------------------+------");

        // Shared weight data [oc, ic] (PyTorch layout)
        let w_f32: Vec<f32> = (0..ic * oc).map(|i| ((i as f32) * 0.00001).sin() * 0.01).collect();

        // Transpose for BLAS: [ic, oc] for gemm(CblasTrans on A)
        let mut w_t = vec![0.0f32; ic * oc];
        for r in 0..oc { for c in 0..ic { w_t[c * oc + r] = w_f32[r * ic + c]; } }

        for seq in [1, 16, 32, 64, 128, 256, 512] {
            let actual_seq = seq.max(ANE_MIN_SPATIAL); // ANE needs ≥16

            // --- BLAS: y[oc, seq] = W^T[oc,ic] @ act[ic, seq] via sgemm ---
            let act: Vec<f32> = (0..ic * seq).map(|i| ((i as f32) * 0.001).sin()).collect();
            let mut out_blas = vec![0.0f32; oc * seq];

            // sgemm: C[M,N] = alpha * A^T[M,K] @ B[K,N] + beta*C
            // A = w_t[ic,oc] (transposed → [oc,ic]), B = act[ic,seq], C = out[oc,seq]
            unsafe { cblas_sgemm(101, 112, 111, oc as i32, seq as i32, ic as i32,
                1.0, w_t.as_ptr(), oc as i32, act.as_ptr(), seq as i32,
                0.0, out_blas.as_mut_ptr(), seq as i32); }

            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                unsafe { cblas_sgemm(101, 112, 111, oc as i32, seq as i32, ic as i32,
                    1.0, w_t.as_ptr(), oc as i32, act.as_ptr(), seq as i32,
                    0.0, out_blas.as_mut_ptr(), seq as i32); }
            }
            let blas_us = t0.elapsed().as_micros() / n_iters;

            // --- ANE BLOBFILE ---
            let blob = build_weight_blob(&w_f32, oc, ic);
            let fused = gen_blobfile_matmul(ic, oc, actual_seq, "sweep");
            let names: Vec<&str> = fused.weight_names.iter().map(|s| s.as_str()).collect();
            let kernel = AneKernel::compile_multi_weights(
                &fused.mil_text, &names, &[&blob],
                &[fused.input_bytes], &[fused.output_bytes],
            ).expect("compile failed");

            // Pad activation to actual_seq
            let mut act_padded = vec![0.0f32; ic * actual_seq];
            for c in 0..ic {
                for s in 0..seq.min(actual_seq) {
                    act_padded[c * actual_seq + s] = act[c * seq + s];
                }
            }
            let act_bytes: Vec<u8> = act_padded.iter().flat_map(|f| f.to_le_bytes()).collect();
            kernel.write_input(0, &act_bytes);
            kernel.eval().unwrap(); // warmup

            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                kernel.write_input(0, &act_bytes);
                kernel.eval().unwrap();
            }
            let ane_us = t0.elapsed().as_micros() / n_iters;

            // --- GPU (MLX Metal) ---
            // W is [oc, ic] f32, act is [seq, ic] f32, out = act @ W^T = [seq, oc]
            use mlx_rs::{Array as MlxArray, ops, transforms::eval as mlx_eval};
            let w_mlx = MlxArray::from_slice(&w_f32, &[oc as i32, ic as i32]);
            let act_mlx = MlxArray::from_slice(&act, &[seq as i32, ic as i32]);
            // Warmup
            let out_gpu = ops::matmul(&act_mlx, &w_mlx.t())?;
            mlx_eval([&out_gpu])?;

            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                let out_gpu = ops::matmul(&act_mlx, &w_mlx.t())?;
                mlx_eval([&out_gpu])?;
            }
            let gpu_us = t0.elapsed().as_micros() / n_iters;

            // Pick winner
            let min_us = blas_us.min(ane_us).min(gpu_us);
            let winner = if min_us == blas_us { "BLAS" }
                else if min_us == ane_us { "ANE" }
                else { "GPU" };
            eprintln!(
                "{seq:>6} | {blas_us:>7}µs {blas_24:>6}ms | {ane_us:>7}µs {ane_24:>6}ms | {gpu_us:>7}µs {gpu_24:>6}ms | {winner:>5}",
                blas_24 = blas_us * 24 / 1000,
                ane_24 = ane_us * 24 / 1000,
                gpu_24 = gpu_us * 24 / 1000,
            );
        }
        } // end matrix size loop

        Ok(())
    }
}
