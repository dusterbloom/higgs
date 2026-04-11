//! ANE-accelerated GDN (GatedDeltaNet) projections for prefill.
//!
//! Multi-dispatch: 2 kernels per bucket (qkvz + ba), matching the model's
//! combined projection layout. Output goes through `fix_query_key_value_ordering`
//! for correct per-head splitting — same path as GPU forward.
//!
//! Uses `delta_reload` to swap per-layer weights (~100µs per swap).
//! Feature-gated behind `ane`.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::error::Exception;
use mlx_rs::module::ModuleParameters;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::ane_bridge::{self, build_weight_blob_transposed, AneKernel};
use crate::ane_mil::{gen_blobfile_matmul, ANE_MIN_SPATIAL};
use crate::qwen3_next::Qwen3NextCausalLM;

// ---------------------------------------------------------------------------
// GDN dimensions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct GdnDims {
    pub hidden: usize,
    pub qkvz_dim: usize, // combined Q+K+V+Z output dim
    pub ba_dim: usize,   // combined B+A output dim
}

impl GdnDims {
    pub fn from_model(model: &Qwen3NextCausalLM) -> Self {
        let a = &model.args;
        let key_dim = (a.linear_num_key_heads * a.linear_key_head_dim) as usize;
        let value_dim = (a.linear_num_value_heads * a.linear_value_head_dim) as usize;
        let hidden = a.hidden_size as usize;
        // Combined dims match in_proj_qkvz and in_proj_ba.
        let qkvz_dim = key_dim * 2 + value_dim * 2; // Q + K + V + Z
        let ba_dim = a.linear_num_value_heads as usize * 2; // B + A
        Self { hidden, qkvz_dim, ba_dim }
    }
}

// ---------------------------------------------------------------------------
// Per-layer weight blobs
// ---------------------------------------------------------------------------

pub struct GdnLayerWeights {
    pub w_qkvz: Vec<u8>, // BLOBFILE for in_proj_qkvz
    pub w_ba: Vec<u8>,   // BLOBFILE for in_proj_ba
}

// ---------------------------------------------------------------------------
// Weight extraction
// ---------------------------------------------------------------------------

type ParamMap<'a> = HashMap<Rc<str>, &'a Array>;

fn dequantize_to_f32(
    params: &ParamMap<'_>, prefix: &str, gs: i32, bits: i32,
    out_feat: usize, in_feat: usize,
) -> Vec<f32> {
    let w = *params.get(format!("{prefix}.weight").as_str())
        .unwrap_or_else(|| panic!("Missing: {prefix}.weight"));
    let s = *params.get(format!("{prefix}.scales").as_str())
        .unwrap_or_else(|| panic!("Missing: {prefix}.scales"));
    let b = *params.get(format!("{prefix}.biases").as_str())
        .unwrap_or_else(|| panic!("Missing: {prefix}.biases"));
    let deq = mlx_rs::ops::dequantize(w, s, b, gs, bits).expect("dequantize");
    let f32_arr = if deq.dtype() != Dtype::Float32 {
        deq.as_dtype(Dtype::Float32).expect("cast")
    } else { deq };
    eval([&f32_arr]).expect("eval");
    let data = f32_arr.as_slice::<f32>();
    assert_eq!(data.len(), out_feat * in_feat,
        "{prefix}: expected {}×{}={}, got {}", out_feat, in_feat, out_feat * in_feat, data.len());
    data.to_vec()
}

fn extract_all_weights(
    model: &Qwen3NextCausalLM, dims: &GdnDims,
) -> Vec<GdnLayerWeights> {
    let params_nested = model.parameters();
    let params = params_nested.flatten();
    let all_arrays: Vec<&Array> = params.values().copied().collect();
    eval(all_arrays).expect("eval params");

    let gs = model.args.quantization.as_ref().map_or(64, |q| q.group_size);
    let bits = model.args.quantization.as_ref().map_or(4, |q| q.bits);
    let n_layers = model.args.num_hidden_layers as usize;
    let interval = model.args.full_attention_interval as usize;

    let mut weights = Vec::new();
    for i in 0..n_layers {
        if (i + 1) % interval == 0 { continue; } // FA layer
        let pfx = format!("model.layers.{i}.linear_attn");

        // Dequantize the combined projections directly — no row splitting needed.
        let d_qkvz = dequantize_to_f32(&params, &format!("{pfx}.in_proj_qkvz"),
            gs, bits, dims.qkvz_dim, dims.hidden);
        let d_ba = dequantize_to_f32(&params, &format!("{pfx}.in_proj_ba"),
            gs, bits, dims.ba_dim, dims.hidden);

        // Dequantized data is [oc, ic] (PyTorch convention).
        // build_weight_blob_transposed(data[oc,ic], oc, ic) → blob in [ic,oc] for MIL.
        weights.push(GdnLayerWeights {
            w_qkvz: build_weight_blob_transposed(&d_qkvz, dims.qkvz_dim, dims.hidden),
            w_ba: build_weight_blob_transposed(&d_ba, dims.ba_dim, dims.hidden),
        });
    }
    weights
}

// ---------------------------------------------------------------------------
// GDN Prefill Engine
// ---------------------------------------------------------------------------

pub const SEQ_BUCKETS: &[usize] = &[512, 1024, 2048];

fn select_bucket(seq_len: usize) -> usize {
    for &b in SEQ_BUCKETS {
        if seq_len <= b { return b; }
    }
    *SEQ_BUCKETS.last().unwrap()
}

struct ProjKernel {
    kernel: AneKernel,
    oc: usize,
    padded_seq: usize,
}

/// Per-bucket: 2 kernels (qkvz + ba).
struct BucketKernels {
    seq: usize,
    qkvz: ProjKernel,
    ba: ProjKernel,
}

pub struct GdnPrefillEngine {
    buckets: Vec<BucketKernels>,
    layer_weights: Vec<GdnLayerWeights>,
    gdn_layer_indices: Vec<usize>,
    dims: GdnDims,
}

impl GdnPrefillEngine {
    pub fn new(model: &Qwen3NextCausalLM) -> Result<Self, String> {
        ane_bridge::ane_init().map_err(|e| format!("ANE init: {e}"))?;

        let dims = GdnDims::from_model(model);
        let n_layers = model.args.num_hidden_layers as usize;
        let interval = model.args.full_attention_interval as usize;
        let gdn_layer_indices: Vec<usize> = (0..n_layers)
            .filter(|i| (i + 1) % interval != 0).collect();

        tracing::info!("ANE GDN prefill: {} layers, hidden={} qkvz={} ba={}",
            gdn_layer_indices.len(), dims.hidden, dims.qkvz_dim, dims.ba_dim);

        let t0 = std::time::Instant::now();
        let layer_weights = extract_all_weights(model, &dims);
        tracing::info!("ANE GDN prefill: weight extraction {}ms", t0.elapsed().as_millis());
        assert_eq!(layer_weights.len(), gdn_layer_indices.len());

        let compile_proj = |ic: usize, oc: usize, seq: usize, name: &str, blob: &[u8]|
            -> Result<ProjKernel, String>
        {
            let mil = gen_blobfile_matmul(ic, oc, seq, name);
            let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
            let kernel = AneKernel::compile_multi_weights(
                &mil.mil_text, &names, &[blob],
                &[mil.input_bytes], &[mil.output_bytes],
            ).map_err(|e| format!("compile {name} seq={seq}: {e}"))?;
            Ok(ProjKernel { kernel, oc, padded_seq: seq.max(ANE_MIN_SPATIAL) })
        };

        let mut buckets = Vec::with_capacity(SEQ_BUCKETS.len());
        for &seq in SEQ_BUCKETS {
            let t0 = std::time::Instant::now();
            let qkvz = compile_proj(dims.hidden, dims.qkvz_dim, seq, "qkvz",
                &layer_weights[0].w_qkvz)?;
            let ba = compile_proj(dims.hidden, dims.ba_dim, seq, "ba",
                &layer_weights[0].w_ba)?;
            tracing::info!("ANE GDN prefill: bucket seq={seq} compiled {}ms",
                t0.elapsed().as_millis());
            buckets.push(BucketKernels { seq, qkvz, ba });
        }

        Ok(Self { buckets, layer_weights, gdn_layer_indices, dims })
    }

    pub fn gdn_local_index(&self, global_layer_idx: usize) -> Option<usize> {
        self.gdn_layer_indices.iter().position(|&i| i == global_layer_idx)
    }

    /// Run combined projections on ANE. Returns `(qkvz_output, ba_output)` in
    /// the same format as `in_proj_qkvz.forward()` / `in_proj_ba.forward()`.
    /// Caller should pass these to `fix_query_key_value_ordering`.
    pub fn forward_combined_projections(
        &self, gdn_idx: usize, normed_hidden: &Array,
    ) -> Result<(Array, Array), Exception> {
        let shape = normed_hidden.shape();
        let seq = *shape.get(1)
            .ok_or_else(|| Exception::custom("expected [1,seq,hidden]"))? as usize;

        let bucket_seq = select_bucket(seq);
        let bk = self.buckets.iter().find(|b| b.seq == bucket_seq)
            .ok_or_else(|| Exception::custom(format!("no bucket {bucket_seq}")))?;

        let lw = &self.layer_weights[gdn_idx];
        let hidden = self.dims.hidden;
        let padded_seq = bucket_seq.max(ANE_MIN_SPATIAL);

        // Convert MLX [1, seq, hidden] row-major → ANE [hidden, padded_seq] channel-major.
        // Read from original row-major storage: x_data[s * hidden + ch].
        eval([normed_hidden])?;
        let x_data = normed_hidden
            .reshape(&[seq as i32, hidden as i32])?;
        eval([&x_data])?;
        let x_flat = x_data.as_slice::<f32>();

        let mut input_buf = vec![0.0f32; hidden * padded_seq];
        for s in 0..seq {
            for ch in 0..hidden {
                input_buf[ch * padded_seq + s] = x_flat[s * hidden + ch];
            }
        }
        let input_bytes: Vec<u8> = input_buf.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Run both projections.
        let run_proj = |pk: &ProjKernel, blob: &[u8]| -> Result<Array, Exception> {
            pk.kernel.delta_reload(&[blob])
                .or_else(|_| pk.kernel.reload_weights(&[blob]))
                .map_err(|e| Exception::custom(format!("reload: {e}")))?;
            pk.kernel.write_input(0, &input_bytes);
            pk.kernel.eval().map_err(|e| Exception::custom(format!("eval: {e}")))?;

            let mut out_bytes = vec![0u8; pk.oc * padded_seq * 4];
            pk.kernel.read_output(0, &mut out_bytes);
            let out_f32: Vec<f32> = out_bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

            // Transpose from channel-major [oc, padded_seq] → [1, seq, oc].
            let mut result = vec![0.0f32; seq * pk.oc];
            for ch in 0..pk.oc {
                for s in 0..seq {
                    result[s * pk.oc + ch] = out_f32[ch * padded_seq + s];
                }
            }
            Ok(Array::from_slice(&result, &[1, seq as i32, pk.oc as i32]))
        };

        let qkvz = run_proj(&bk.qkvz, &lw.w_qkvz)?;
        let ba = run_proj(&bk.ba, &lw.w_ba)?;
        Ok((qkvz, ba))
    }

    pub fn num_gdn_layers(&self) -> usize { self.layer_weights.len() }
    pub fn bucket_sizes(&self) -> Vec<usize> { self.buckets.iter().map(|b| b.seq).collect() }
    pub fn dims(&self) -> &GdnDims { &self.dims }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, unsafe_code)]
mod tests {
    use super::*;

    #[test]
    fn test_select_bucket() {
        assert_eq!(select_bucket(1), 512);
        assert_eq!(select_bucket(512), 512);
        assert_eq!(select_bucket(513), 1024);
        assert_eq!(select_bucket(2048), 2048);
        assert_eq!(select_bucket(4096), 2048);
    }

    /// Verify build_weight_blob_transposed produces same result as manual transpose + build_weight_blob.
    #[test]
    fn test_blob_transposed_parity() {
        use crate::ane_bridge::{self, build_weight_blob, build_weight_blob_transposed, AneKernel};
        use crate::ane_mil::gen_blobfile_matmul;

        ane_bridge::ane_init().expect("ANE init");

        let ic = 64usize;
        let oc = 128usize;
        let seq = 32usize;

        // Weight in PyTorch [oc, ic] layout.
        let w_pytorch: Vec<f32> = (0..oc * ic).map(|i| ((i as f32) * 1e-3).sin()).collect();

        // Method A: manual transpose to [ic, oc], then build_weight_blob.
        let mut w_transposed = vec![0.0f32; ic * oc];
        for r in 0..oc {
            for c in 0..ic {
                w_transposed[c * oc + r] = w_pytorch[r * ic + c];
            }
        }
        let blob_a = build_weight_blob(&w_transposed, ic, oc);

        // Method B: build_weight_blob_transposed directly.
        let blob_b = build_weight_blob_transposed(&w_pytorch, oc, ic);

        assert_eq!(blob_a.len(), blob_b.len(), "blob size mismatch");
        assert_eq!(blob_a, blob_b, "blobs differ!");

        // Now verify both produce correct matmul output.
        let mil = gen_blobfile_matmul(ic, oc, seq, "test");
        let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
        let ka = AneKernel::compile_multi_weights(&mil.mil_text, &names, &[&blob_a], &[mil.input_bytes], &[mil.output_bytes]).unwrap();
        let kb = AneKernel::compile_multi_weights(&mil.mil_text, &names, &[&blob_b], &[mil.input_bytes], &[mil.output_bytes]).unwrap();

        let act: Vec<f32> = (0..ic * seq).map(|i| ((i as f32) * 0.01).cos()).collect();
        let act_bytes: Vec<u8> = act.iter().flat_map(|f| f.to_le_bytes()).collect();

        ka.write_input(0, &act_bytes);
        ka.eval().unwrap();
        let mut out_a = vec![0u8; mil.output_bytes];
        ka.read_output(0, &mut out_a);

        kb.write_input(0, &act_bytes);
        kb.eval().unwrap();
        let mut out_b = vec![0u8; mil.output_bytes];
        kb.read_output(0, &mut out_b);

        assert_eq!(out_a, out_b, "outputs differ between manual transpose and build_weight_blob_transposed");
        eprintln!("blob_transposed parity: OK (blobs identical, outputs identical)");
    }

    /// End-to-end: identity weight should reproduce input through MLX→ANE→MLX round-trip.
    #[test]
    fn test_identity_roundtrip() {
        use crate::ane_bridge::{self, build_weight_blob, AneKernel};
        use crate::ane_mil::gen_blobfile_matmul;

        ane_bridge::ane_init().expect("ANE init");

        let dim = 64usize;
        let seq = 32usize;
        let padded_seq = seq; // 32 is already aligned

        // Identity weight [dim, dim] in [ic, oc] layout for build_weight_blob.
        let mut w = vec![0.0f32; dim * dim];
        for i in 0..dim { w[i * dim + i] = 1.0; }
        let blob = build_weight_blob(&w, dim, dim);

        let mil = gen_blobfile_matmul(dim, dim, seq, "id");
        let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
        let kernel = AneKernel::compile_multi_weights(
            &mil.mil_text, &names, &[&blob], &[mil.input_bytes], &[mil.output_bytes],
        ).unwrap();

        // MLX-like input: row-major [seq, dim] = [[1,2,..dim], [dim+1,...], ...]
        let mlx_input: Vec<f32> = (0..seq * dim).map(|i| (i + 1) as f32).collect();

        // Convert to ANE channel-major [dim, padded_seq]: data[ch * padded_seq + s]
        let mut ane_input = vec![0.0f32; dim * padded_seq];
        for ch in 0..dim {
            for s in 0..seq {
                ane_input[ch * padded_seq + s] = mlx_input[s * dim + ch];
            }
        }
        let input_bytes: Vec<u8> = ane_input.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &input_bytes);
        kernel.eval().unwrap();

        let mut out_bytes = vec![0u8; dim * padded_seq * 4];
        kernel.read_output(0, &mut out_bytes);
        let out_f32: Vec<f32> = out_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        // Convert ANE output back to row-major [seq, dim].
        let mut mlx_output = vec![0.0f32; seq * dim];
        for ch in 0..dim {
            for s in 0..seq {
                mlx_output[s * dim + ch] = out_f32[ch * padded_seq + s];
            }
        }

        let max_err = mlx_input.iter().zip(mlx_output.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("Identity roundtrip: max_err={max_err}, first 5 in={:?} out={:?}",
            &mlx_input[..5], &mlx_output[..5]);
        assert!(max_err < 0.1, "Identity roundtrip error too large: {max_err}");
    }

    /// Identity roundtrip at real dims: 2048→64 (small oc, real ic).
    #[test]
    fn test_identity_real_ic() {
        use crate::ane_bridge::{self, build_weight_blob_transposed, AneKernel};
        use crate::ane_mil::gen_blobfile_matmul;

        ane_bridge::ane_init().expect("ANE init");

        let ic = 2048usize;
        let oc = 64usize;
        let seq = 32usize;

        // Identity-like: first 64 outputs = first 64 inputs.
        // Data in [oc=64, ic=2048] PyTorch layout, transpose to [ic,oc] for MIL.
        let mut w = vec![0.0f32; oc * ic];
        for i in 0..oc { w[i * ic + i] = 1.0; }
        let blob = build_weight_blob_transposed(&w, oc, ic);

        let mil = gen_blobfile_matmul(ic, oc, seq, "test");
        let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
        let kernel = AneKernel::compile_multi_weights(
            &mil.mil_text, &names, &[&blob], &[mil.input_bytes], &[mil.output_bytes],
        ).unwrap();

        // Input: channel-major [ic, seq], each channel c has value c+1 at pos 0.
        let mut ane_input = vec![0.0f32; ic * seq];
        for ch in 0..ic { ane_input[ch * seq] = (ch + 1) as f32; }
        let bytes: Vec<u8> = ane_input.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &bytes);
        kernel.eval().unwrap();

        let mut out_bytes = vec![0u8; oc * seq * 4];
        kernel.read_output(0, &mut out_bytes);
        let out: Vec<f32> = out_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        // Extract position 0 from each output channel.
        let result: Vec<f32> = (0..oc).map(|c| out[c * seq]).collect();
        let max_err = (0..oc).map(|i| ((i + 1) as f32 - result[i]).abs()).fold(0.0f32, f32::max);
        eprintln!("2048→64 identity: max_err={max_err}, first 5={:?}", &result[..5]);
        assert!(max_err < 1.0, "Identity error: {max_err}");
    }

    /// Compile + eval a single projection at real 35B dimensions.
    #[test]
    fn test_compile_real_dims() {
        use crate::ane_bridge::{self, build_weight_blob, AneKernel};
        use crate::ane_mil::gen_blobfile_matmul;

        ane_bridge::ane_init().expect("ANE init");

        let ic = 2048usize;
        let oc = 12352usize; // qkvz_dim = 2048+2048+4096+4096
        let seq = 512usize;
        let mil = gen_blobfile_matmul(ic, oc, seq, "qkvz");

        // Random weight [oc, ic] — transposed to [ic, oc] for MIL.
        let data: Vec<f32> = (0..oc * ic).map(|i| ((i as f32) * 1e-5).sin() * 0.01).collect();
        let blob = build_weight_blob_transposed(&data, oc, ic);

        let names: Vec<&str> = mil.weight_names.iter().map(|s| s.as_str()).collect();
        let kernel = AneKernel::compile_multi_weights(
            &mil.mil_text, &names, &[blob.as_slice()],
            &[mil.input_bytes], &[mil.output_bytes],
        ).expect("compile 2048→12352 failed");

        let input: Vec<f32> = (0..ic * seq).map(|i| ((i as f32) * 0.001).sin()).collect();
        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &input_bytes);
        kernel.eval().expect("eval failed");

        let mut out_bytes = vec![0u8; oc * seq * 4];
        kernel.read_output(0, &mut out_bytes);
        let out_f32: Vec<f32> = out_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let nonzero = out_f32.iter().filter(|&&v| v.abs() > 1e-10).count();
        assert!(nonzero > oc * seq / 2);
        eprintln!("2048→12352 @ seq=512: OK ({nonzero}/{} non-zero)", oc * seq);
    }
}
