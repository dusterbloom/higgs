//! DFlash drafter — CPU BLAS forward pass (no GPU dependency).
//!
//! Runs the DFlash block-diffusion drafter entirely on CPU via Accelerate BLAS,
//! freeing the GPU for the target model's verify step (speculative overlap).
//!
//! Architecture: identical to `dflash.rs` (8-layer dual-stream transformer),
//! but using flat `Vec<f32>` weights and `cblas_sgemm` instead of MLX arrays.
//!
//! Weight extraction from the MLX model uses `ModuleParameters` reflection.

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

use crate::diffusion::{
    apply_rope, rms_norm, rms_norm_slice, sgemm, sgemm_nt, sgemm_nt_scaled, softmax_inplace,
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DFlashCpuConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub num_taps: usize,
    pub block_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

// ---------------------------------------------------------------------------
// Per-layer weights
// ---------------------------------------------------------------------------

/// Per-layer weight data. Large projection matrices stored as bf16 (`Vec<u16>`)
/// to halve memory; converted to f32 into a scratch buffer during forward.
/// Small norm weights stay f32.
#[derive(Clone)]
pub struct DFlashCpuLayerWeights {
    // Attention projections [out, in] — bf16 raw bits
    pub q_proj: Vec<u16>,         // [heads*head_dim, hidden]
    pub k_proj: Vec<u16>,         // [kv_heads*head_dim, hidden]
    pub v_proj: Vec<u16>,         // [kv_heads*head_dim, hidden]
    pub o_proj: Vec<u16>,         // [hidden, heads*head_dim]
    // Per-head QK norms — f32 (tiny)
    pub q_norm: Vec<f32>,         // [head_dim]
    pub k_norm: Vec<f32>,         // [head_dim]
    // Layer norms — f32 (tiny)
    pub input_norm: Vec<f32>,     // [hidden]
    pub post_attn_norm: Vec<f32>, // [hidden]
    // SwiGLU MLP — bf16 raw bits
    pub gate_proj: Vec<u16>,      // [inter, hidden]
    pub up_proj: Vec<u16>,        // [inter, hidden]
    pub down_proj: Vec<u16>,      // [hidden, inter]
}

/// Convert bf16 raw bits to f32 in a pre-allocated buffer.
#[inline]
fn bf16_to_f32(src: &[u16], dst: &mut [f32]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = half::bf16::from_bits(*s).to_f32();
    }
}

/// SGEMM with bf16 weight matrix B: convert B to f32 in scratch, then call sgemm_nt.
fn sgemm_nt_bf16(m: usize, n: usize, k: usize, a: &[f32], b: &[u16], c: &mut [f32], scratch: &mut Vec<f32>) {
    let len = b.len();
    if scratch.len() < len {
        scratch.resize(len, 0.0);
    }
    bf16_to_f32(b, &mut scratch[..len]);
    sgemm_nt(m, n, k, a, &scratch[..len], c);
}

// ---------------------------------------------------------------------------
// Engine (all weights + precomputed RoPE)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DFlashCpuEngine {
    /// Input projection: [hidden, num_taps * hidden] row-major — bf16 raw bits.
    pub fc: Vec<u16>,
    /// Post-fc RMSNorm weight: [hidden].
    pub hidden_norm: Vec<f32>,
    /// Final output RMSNorm weight: [hidden].
    pub final_norm: Vec<f32>,
    /// Per-layer weights.
    pub layers: Vec<DFlashCpuLayerWeights>,
    pub config: DFlashCpuConfig,
    /// Precomputed RoPE cos table: [max_seq, head_dim/2].
    pub rope_cos: Vec<f32>,
    /// Precomputed RoPE sin table: [max_seq, head_dim/2].
    pub rope_sin: Vec<f32>,
}

// ---------------------------------------------------------------------------
// KV cache (CPU-side, stores target context K/V AND noise K/V)
// ---------------------------------------------------------------------------

/// CPU-side KV cache for the DFlash drafter.
///
/// Stores accumulated target context K/V AND noise K/V (post-QK-norm, post-RoPE)
/// across rounds. Both are needed so the drafter can attend to its own previous
/// predictions ("dual-stream attention" memory).
///
/// Layout per layer: flat `[cached_len * kv_dim]` where `kv_dim = kv_heads * head_dim`.
/// Each round appends only `ctx_len` positions of target-context K/V (derived
/// from taps of the target's verified hidden states). Noise K/V are never
/// persisted — they exist only inside a single round's forward to serve SDPA,
/// then are discarded. `cache.len` therefore equals the absolute sequence
/// position where the next round's context will be written, keeping RoPE
/// offsets in lockstep with the real token positions. This mirrors
/// `ContextOnlyDraftKVCache` in the dflash-mlx reference.
pub struct DFlashCpuCache {
    /// Per-layer K cache (target-context positions only; no noise).
    pub(crate) k: Vec<Vec<f32>>,
    /// Per-layer V cache (target-context positions only; no noise).
    pub(crate) v: Vec<Vec<f32>>,
    /// Number of cached sequence positions (same for all layers).
    /// Advances by `ctx_len` per round.
    pub len: usize,
    kv_dim: usize,
}

impl DFlashCpuCache {
    pub fn new(n_layers: usize, kv_dim: usize) -> Self {
        Self {
            k: vec![Vec::new(); n_layers],
            v: vec![Vec::new(); n_layers],
            len: 0,
            kv_dim,
        }
    }

    /// Crop cache to `keep_len` positions (discard everything after).
    pub fn crop(&mut self, keep_len: usize) {
        if keep_len < self.len {
            for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
                k.truncate(keep_len * self.kv_dim);
                v.truncate(keep_len * self.kv_dim);
            }
            self.len = keep_len;
        }
    }
}

// ---------------------------------------------------------------------------
// Weight extraction from MLX DFlashDrafter
// ---------------------------------------------------------------------------

use mlx_rs::module::ModuleParameters;
use mlx_rs::transforms::eval;
use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::Array;

type ParamMap<'a> = HashMap<Rc<str>, &'a Array>;

/// Extract a parameter as bf16 raw bits (u16), converting from any dtype.
fn get_bf16_mlx(params: &ParamMap<'_>, key: &str) -> Vec<u16> {
    let f32_vec = get_f32_mlx(params, key);
    f32_vec.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect()
}

/// Extract a single parameter as f32 vec, converting dtype if needed.
fn get_f32_mlx(params: &ParamMap<'_>, key: &str) -> Vec<f32> {
    let arr = params
        .get(key)
        .unwrap_or_else(|| panic!("Missing weight: {key}"));
    if arr.dtype() == mlx_rs::Dtype::Float32 {
        arr.as_slice::<f32>().to_vec()
    } else {
        let converted = arr
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap_or_else(|e| panic!("Failed to convert {key} to f32: {e}"));
        eval([&converted]).unwrap_or_else(|e| panic!("Failed to eval {key}: {e}"));
        converted.as_slice::<f32>().to_vec()
    }
}

/// Extract all weights from a loaded MLX `DFlashDrafter` into a CPU engine.
pub fn extract_dflash_cpu_engine(drafter: &crate::dflash::DFlashDrafter) -> DFlashCpuEngine {
    let cfg = &drafter.config;

    let params_nested = drafter.parameters();
    let params = params_nested.flatten();

    // Eval all parameters so as_slice works.
    let all_arrays: Vec<&Array> = params.values().copied().collect();
    eval(all_arrays).expect("Failed to eval DFlash parameters");

    let hidden = cfg.hidden_size as usize;
    let layers = cfg.num_hidden_layers as usize;
    let heads = cfg.num_attention_heads as usize;
    let kv_heads = cfg.num_key_value_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let inter = cfg.intermediate_size as usize;
    let num_taps = cfg.num_taps();

    // Top-level weights
    let fc = get_bf16_mlx(&params, "fc.weight");
    let hidden_norm = get_f32_mlx(&params, "hidden_norm.weight");
    let final_norm = get_f32_mlx(&params, "norm.weight");

    // Per-layer weights
    let mut layer_weights = Vec::with_capacity(layers);
    for i in 0..layers {
        let prefix = format!("layers.{i}");
        layer_weights.push(DFlashCpuLayerWeights {
            q_proj: get_bf16_mlx(&params, &format!("{prefix}.self_attn.q_proj.weight")),
            k_proj: get_bf16_mlx(&params, &format!("{prefix}.self_attn.k_proj.weight")),
            v_proj: get_bf16_mlx(&params, &format!("{prefix}.self_attn.v_proj.weight")),
            o_proj: get_bf16_mlx(&params, &format!("{prefix}.self_attn.o_proj.weight")),
            q_norm: get_f32_mlx(&params, &format!("{prefix}.self_attn.q_norm.weight")),
            k_norm: get_f32_mlx(&params, &format!("{prefix}.self_attn.k_norm.weight")),
            input_norm: get_f32_mlx(&params, &format!("{prefix}.input_layernorm.weight")),
            post_attn_norm: get_f32_mlx(
                &params,
                &format!("{prefix}.post_attention_layernorm.weight"),
            ),
            gate_proj: get_bf16_mlx(&params, &format!("{prefix}.mlp.gate_proj.weight")),
            up_proj: get_bf16_mlx(&params, &format!("{prefix}.mlp.up_proj.weight")),
            down_proj: get_bf16_mlx(&params, &format!("{prefix}.mlp.down_proj.weight")),
        });
    }

    // Precompute RoPE tables (same formula as diffusion.rs)
    let max_seq = 4096; // generous upper bound
    let half_dim = head_dim / 2;
    let mut rope_cos = vec![0.0f32; max_seq * half_dim];
    let mut rope_sin = vec![0.0f32; max_seq * half_dim];
    for pos in 0..max_seq {
        for d in 0..half_dim {
            let freq = 1.0 / cfg.rope_theta.powf(2.0 * d as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            rope_cos[pos * half_dim + d] = angle.cos();
            rope_sin[pos * half_dim + d] = angle.sin();
        }
    }

    let config = DFlashCpuConfig {
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        inter,
        num_taps,
        block_size: cfg.block_size as usize,
        rope_theta: cfg.rope_theta,
        rms_norm_eps: cfg.rms_norm_eps,
    };

    DFlashCpuEngine {
        fc,
        hidden_norm,
        final_norm,
        layers: layer_weights,
        config,
        rope_cos,
        rope_sin,
    }
}

// ---------------------------------------------------------------------------
// Direct safetensors loading (bypasses MLX entirely)
// ---------------------------------------------------------------------------

/// Read a tensor as raw bf16 bits (u16). Converts f32/f16 sources to bf16.
fn tensor_to_bf16(safetensors: &safetensors::SafeTensors<'_>, name: &str) -> Vec<u16> {
    let view = safetensors
        .tensor(name)
        .unwrap_or_else(|_| panic!("Missing weight: {name}"));
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::BF16 => {
            let count = data.len() / 2;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                out.push(u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]));
            }
            out
        }
        safetensors::Dtype::F32 => {
            let count = data.len() / 4;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let v = f32::from_le_bytes([data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]]);
                out.push(half::bf16::from_f32(v).to_bits());
            }
            out
        }
        safetensors::Dtype::F16 => {
            let count = data.len() / 2;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                let v = half::f16::from_bits(bits).to_f32();
                out.push(half::bf16::from_f32(v).to_bits());
            }
            out
        }
        other => panic!("Unsupported dtype {other:?} for weight {name}"),
    }
}

/// Read a tensor and convert to f32. Used for small norm weights.
fn tensor_to_f32(safetensors: &safetensors::SafeTensors<'_>, name: &str) -> Vec<f32> {
    let view = safetensors
        .tensor(name)
        .unwrap_or_else(|_| panic!("Missing weight: {name}"));
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::BF16 => {
            let count = data.len() / 2;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                out.push(half::bf16::from_bits(bits).to_f32());
            }
            out
        }
        safetensors::Dtype::F32 => {
            let count = data.len() / 4;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                out.push(f32::from_le_bytes([data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]]));
            }
            out
        }
        other => panic!("Unsupported dtype {other:?} for weight {name}"),
    }
}

/// Load a `DFlashCpuEngine` directly from safetensors files on disk.
///
/// Reads `config.json` for architecture parameters, then memory-maps the
/// safetensors file(s) and converts each weight from BF16/F16 → f32 on CPU.
/// This completely bypasses MLX — no GPU memory is touched, no `DFlashDrafter`
/// is constructed, and no `ModuleParameters` reflection is needed.
pub fn load_dflash_cpu_engine_from_safetensors(
    model_path: &std::path::Path,
) -> Result<(DFlashCpuEngine, crate::dflash::DFlashConfig), crate::error::ModelError> {
    use crate::error::ModelError;

    // 1. Parse config.json
    let config_str = std::fs::read_to_string(model_path.join("config.json"))
        .map_err(|e| ModelError::Io(std::io::Error::other(format!("reading config.json: {e}"))))?;
    let cfg: crate::dflash::DFlashConfig = serde_json::from_str(&config_str)
        .map_err(|e| ModelError::Io(std::io::Error::other(format!("parsing config.json: {e}"))))?;

    let hidden = cfg.hidden_size as usize;
    let layers = cfg.num_hidden_layers as usize;
    let heads = cfg.num_attention_heads as usize;
    let kv_heads = cfg.num_key_value_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let inter = cfg.intermediate_size as usize;
    let num_taps = cfg.num_taps();

    // 2. Collect and mmap safetensors files (zero-copy, no heap allocation)
    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mmaps: Vec<memmap2::Mmap> = safetensors_files
        .iter()
        .map(|p| {
            let file = std::fs::File::open(p)
                .unwrap_or_else(|e| panic!("Failed to open {}: {e}", p.display()));
            unsafe { memmap2::Mmap::map(&file) }
                .unwrap_or_else(|e| panic!("Failed to mmap {}: {e}", p.display()))
        })
        .collect();
    let tensors: Vec<safetensors::SafeTensors<'_>> = mmaps
        .iter()
        .map(|m| safetensors::SafeTensors::deserialize(m).expect("Invalid safetensors"))
        .collect();

    // 3. Extract weights (projections as bf16, norms as f32)
    let get_bf16 = |name: &str| -> Vec<u16> {
        for st in &tensors {
            if st.tensor(name).is_ok() {
                return tensor_to_bf16(st, name);
            }
        }
        panic!("Weight {name} not found in any safetensors shard");
    };
    let get_f32 = |name: &str| -> Vec<f32> {
        for st in &tensors {
            if st.tensor(name).is_ok() {
                return tensor_to_f32(st, name);
            }
        }
        panic!("Weight {name} not found in any safetensors shard");
    };

    let fc = get_bf16("fc.weight");
    let hidden_norm = get_f32("hidden_norm.weight");
    let final_norm = get_f32("norm.weight");

    let mut layer_weights = Vec::with_capacity(layers);
    for i in 0..layers {
        let pb = |suffix: &str| -> Vec<u16> { get_bf16(&format!("layers.{i}.{suffix}")) };
        let pf = |suffix: &str| -> Vec<f32> { get_f32(&format!("layers.{i}.{suffix}")) };
        layer_weights.push(DFlashCpuLayerWeights {
            q_proj: pb("self_attn.q_proj.weight"),
            k_proj: pb("self_attn.k_proj.weight"),
            v_proj: pb("self_attn.v_proj.weight"),
            o_proj: pb("self_attn.o_proj.weight"),
            q_norm: pf("self_attn.q_norm.weight"),
            k_norm: pf("self_attn.k_norm.weight"),
            input_norm: pf("input_layernorm.weight"),
            post_attn_norm: pf("post_attention_layernorm.weight"),
            gate_proj: pb("mlp.gate_proj.weight"),
            up_proj: pb("mlp.up_proj.weight"),
            down_proj: pb("mlp.down_proj.weight"),
        });
    }

    // 4. Precompute RoPE tables
    let max_seq = 4096;
    let half_dim = head_dim / 2;
    let mut rope_cos = vec![0.0f32; max_seq * half_dim];
    let mut rope_sin = vec![0.0f32; max_seq * half_dim];
    for pos in 0..max_seq {
        for d in 0..half_dim {
            let freq = 1.0 / cfg.rope_theta.powf(2.0 * d as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            rope_cos[pos * half_dim + d] = angle.cos();
            rope_sin[pos * half_dim + d] = angle.sin();
        }
    }

    let config = DFlashCpuConfig {
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        inter,
        num_taps,
        block_size: cfg.block_size as usize,
        rope_theta: cfg.rope_theta,
        rms_norm_eps: cfg.rms_norm_eps,
    };

    let engine = DFlashCpuEngine {
        fc,
        hidden_norm,
        final_norm,
        layers: layer_weights,
        config,
        rope_cos,
        rope_sin,
    };

    Ok((engine, cfg))
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

impl DFlashCpuEngine {
    /// Create a fresh KV cache for this engine.
    pub fn make_cache(&self) -> DFlashCpuCache {
        DFlashCpuCache::new(
            self.config.layers,
            self.config.kv_heads * self.config.head_dim,
        )
    }

    /// Run the DFlash drafter forward pass on CPU.
    ///
    /// - `noise`: `[block_size * hidden]` flat row-major — embedded block tokens.
    /// - `taps`: slice of `num_taps` target hidden states, each `[ctx_len * hidden]` flat.
    /// - `ctx_len`: number of context positions in each tap.
    /// - `cache`: CPU KV cache (grows each round with target context).
    ///
    /// Returns `[block_size * hidden]` flat — pass to target's lm_head for logits.
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

        // --- Concatenate taps and project ---
        // taps: num_taps × [ctx_len, hidden] → target_cat: [ctx_len, num_taps * hidden]
        let fc_in = cfg.num_taps * h;
        let mut target_cat = vec![0.0f32; ctx_len * fc_in];
        for s in 0..ctx_len {
            for (t, tap) in taps.iter().enumerate() {
                let src_off = s * h;
                let dst_off = s * fc_in + t * h;
                target_cat[dst_off..dst_off + h].copy_from_slice(&tap[src_off..src_off + h]);
            }
        }

        // fc projection: target_cat[ctx_len, fc_in] @ fc^T → target_hidden[ctx_len, hidden]
        let mut target_hidden = vec![0.0f32; ctx_len * h];
        // Scratch buffer for bf16→f32 weight conversion (sized for largest weight)
        let mut w_scratch: Vec<f32> = Vec::new();
        sgemm_nt_bf16(ctx_len, h, fc_in, &target_cat, &self.fc, &mut target_hidden, &mut w_scratch);

        // hidden_norm: RMSNorm on target_hidden
        let mut target_normed = vec![0.0f32; ctx_len * h];
        rms_norm(
            &target_hidden,
            &self.hidden_norm,
            &mut target_normed,
            ctx_len,
            h,
        );
        // Use the normed version as target_hidden going forward
        let target_hidden = target_normed;

        // --- Layer loop ---
        let mut hidden = noise.to_vec(); // [block, h]

        // Scratch buffers (reused across layers)
        let mut normed = vec![0.0f32; block * h];
        let mut q_buf = vec![0.0f32; block * q_dim];
        let mut k_ctx_buf = vec![0.0f32; ctx_len * kv_dim];
        let mut v_ctx_buf = vec![0.0f32; ctx_len * kv_dim];
        let mut k_noise_buf = vec![0.0f32; block * kv_dim];
        let mut v_noise_buf = vec![0.0f32; block * kv_dim];
        let mut attn_out = vec![0.0f32; block * q_dim];
        let mut o_buf = vec![0.0f32; block * h];
        let mut gate_buf = vec![0.0f32; block * cfg.inter];
        let mut up_buf = vec![0.0f32; block * cfg.inter];

        for (li, layer) in self.layers.iter().enumerate() {
            // --- Attention ---
            // RMSNorm on noise (hidden state)
            rms_norm(&hidden, &layer.input_norm, &mut normed, block, h);

            // Q from noise only
            sgemm_nt_bf16(block, q_dim, h, &normed, &layer.q_proj, &mut q_buf, &mut w_scratch);

            // K/V from target context — convert k_proj/v_proj once, reuse for both ctx and noise
            bf16_to_f32(&layer.k_proj, { if w_scratch.len() < layer.k_proj.len() { w_scratch.resize(layer.k_proj.len(), 0.0); } &mut w_scratch[..layer.k_proj.len()] });
            sgemm_nt(ctx_len, kv_dim, h, &target_hidden, &w_scratch[..layer.k_proj.len()], &mut k_ctx_buf);
            sgemm_nt(block, kv_dim, h, &normed, &w_scratch[..layer.k_proj.len()], &mut k_noise_buf);

            bf16_to_f32(&layer.v_proj, { if w_scratch.len() < layer.v_proj.len() { w_scratch.resize(layer.v_proj.len(), 0.0); } &mut w_scratch[..layer.v_proj.len()] });
            sgemm_nt(ctx_len, kv_dim, h, &target_hidden, &w_scratch[..layer.v_proj.len()], &mut v_ctx_buf);
            sgemm_nt(block, kv_dim, h, &normed, &w_scratch[..layer.v_proj.len()], &mut v_noise_buf);

            // Per-head QK norm (RMSNorm over head_dim)
            for s in 0..block {
                for head in 0..n_heads {
                    let off = s * q_dim + head * hd;
                    rms_norm_slice(&mut q_buf[off..off + hd], &layer.q_norm);
                }
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    rms_norm_slice(&mut k_noise_buf[off..off + hd], &layer.k_norm);
                }
            }
            for s in 0..ctx_len {
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    rms_norm_slice(&mut k_ctx_buf[off..off + hd], &layer.k_norm);
                }
            }

            // RoPE with absolute positions
            // Q and noise K: positions [cache_offset + ctx_len .. cache_offset + ctx_len + block]
            for s in 0..block {
                let pos = cache_offset + ctx_len + s;
                for head in 0..n_heads {
                    let off = s * q_dim + head * hd;
                    apply_rope(
                        &mut q_buf[off..off + hd],
                        pos,
                        half_hd,
                        &self.rope_cos,
                        &self.rope_sin,
                    );
                }
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    apply_rope(
                        &mut k_noise_buf[off..off + hd],
                        pos,
                        half_hd,
                        &self.rope_cos,
                        &self.rope_sin,
                    );
                }
            }
            // Context K: positions [cache_offset .. cache_offset + ctx_len]
            for s in 0..ctx_len {
                let pos = cache_offset + s;
                for head in 0..n_kv {
                    let off = s * kv_dim + head * hd;
                    apply_rope(
                        &mut k_ctx_buf[off..off + hd],
                        pos,
                        half_hd,
                        &self.rope_cos,
                        &self.rope_sin,
                    );
                }
            }

            // Append ONLY target context K/V to the persistent cache.
            // Noise K/V are used locally this round via k_noise_buf/v_noise_buf
            // and intentionally discarded after the round — persisting them
            // would poison RoPE offsets in subsequent rounds (see struct doc).
            cache.k[li].extend_from_slice(&k_ctx_buf[..ctx_len * kv_dim]);
            cache.v[li].extend_from_slice(&v_ctx_buf[..ctx_len * kv_dim]);

            // SDPA: Q attends to [prior_cached | this-round ctx | this-round noise].
            // The first two now live in cache (cache.len + ctx_len entries);
            // the noise K/V live in the local _noise_buf and are concat'd on the fly.
            let kv_cached_len = cache.len + ctx_len;
            let total_kv_len = kv_cached_len + block;
            for kv_h in 0..n_kv {
                // Build K_full and V_full for this KV head: [total_kv_len, hd].
                // [0..kv_cached_len) from persistent cache; [kv_cached_len..total] from local noise.
                let mut k_full = vec![0.0f32; total_kv_len * hd];
                let mut v_full = vec![0.0f32; total_kv_len * hd];

                for s in 0..kv_cached_len {
                    let src_off = s * kv_dim + kv_h * hd;
                    k_full[s * hd..(s + 1) * hd]
                        .copy_from_slice(&cache.k[li][src_off..src_off + hd]);
                    v_full[s * hd..(s + 1) * hd]
                        .copy_from_slice(&cache.v[li][src_off..src_off + hd]);
                }
                for s in 0..block {
                    let src_off = s * kv_dim + kv_h * hd;
                    let dst = kv_cached_len + s;
                    k_full[dst * hd..(dst + 1) * hd]
                        .copy_from_slice(&k_noise_buf[src_off..src_off + hd]);
                    v_full[dst * hd..(dst + 1) * hd]
                        .copy_from_slice(&v_noise_buf[src_off..src_off + hd]);
                }

                // For each Q head in this GQA group
                for g in 0..gqa_ratio {
                    let q_h = kv_h * gqa_ratio + g;
                    let mut q_head = vec![0.0f32; block * hd];
                    for s in 0..block {
                        let qo = s * q_dim + q_h * hd;
                        q_head[s * hd..(s + 1) * hd].copy_from_slice(&q_buf[qo..qo + hd]);
                    }

                    // scores = Q[block, hd] @ K^T[hd, total_kv_len] → [block, total_kv_len]
                    let mut scores = vec![0.0f32; block * total_kv_len];
                    sgemm_nt_scaled(block, total_kv_len, hd, &q_head, &k_full, &mut scores, scale);

                    // Softmax — non-causal (no mask)
                    for row in 0..block {
                        softmax_inplace(
                            &mut scores[row * total_kv_len..(row + 1) * total_kv_len],
                        );
                    }

                    // context = scores[block, total_kv_len] @ V[total_kv_len, hd] → [block, hd]
                    let mut ctx = vec![0.0f32; block * hd];
                    sgemm(block, hd, total_kv_len, &scores, &v_full, &mut ctx);

                    // Write back to attn_out
                    for s in 0..block {
                        let ao = s * q_dim + q_h * hd;
                        attn_out[ao..ao + hd].copy_from_slice(&ctx[s * hd..(s + 1) * hd]);
                    }
                }
            }

            // O projection
            sgemm_nt_bf16(block, h, q_dim, &attn_out, &layer.o_proj, &mut o_buf, &mut w_scratch);

            // Residual add
            for i in 0..block * h {
                hidden[i] += o_buf[i];
            }

            // --- MLP ---
            rms_norm(&hidden, &layer.post_attn_norm, &mut normed, block, h);

            // Gate + SiLU: gate = normed @ gate_proj^T, then gate *= sigmoid(gate)
            sgemm_nt_bf16(block, cfg.inter, h, &normed, &layer.gate_proj, &mut gate_buf, &mut w_scratch);
            for v in gate_buf.iter_mut() {
                let sig = 1.0 / (1.0 + (-*v).exp());
                *v *= sig;
            }

            // Up
            sgemm_nt_bf16(block, cfg.inter, h, &normed, &layer.up_proj, &mut up_buf, &mut w_scratch);

            // gate * up
            for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()) {
                *g *= u;
            }

            // Down
            sgemm_nt_bf16(block, h, cfg.inter, &gate_buf, &layer.down_proj, &mut o_buf, &mut w_scratch);

            // Residual add
            for i in 0..block * h {
                hidden[i] += o_buf[i];
            }
        }

        // Update cache length: each layer got ctx_len new entries this round
        // (target context K/V only — noise is not persisted).
        cache.len += ctx_len;

        // Final RMSNorm
        let mut output = vec![0.0f32; block * h];
        rms_norm(&hidden, &self.final_norm, &mut output, block, h);

        output
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// Smoke test: extract weights and run CPU forward on the 4B DFlash drafter.
    /// Compares output to the MLX forward pass for numerical parity.
    #[test]
    #[ignore] // requires model weights on disk
    fn test_dflash_cpu_parity() {
        use mlx_rs::ops;
        use mlx_rs::transforms::eval;

        // Load MLX drafter
        let snap_dir = std::path::PathBuf::from(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots",
        );
        let snap_dir = std::fs::read_dir(&snap_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let mut mlx_drafter = crate::dflash::load_dflash_drafter(&snap_dir).unwrap();

        // Extract CPU engine
        let cpu_engine = extract_dflash_cpu_engine(&mlx_drafter);

        let block_size = mlx_drafter.config.block_size;
        let hidden = mlx_drafter.config.hidden_size;
        let num_taps = mlx_drafter.config.num_taps();
        let ctx_len = 10;

        // Create deterministic test data
        let _ = mlx_rs::random::seed(42);
        let noise_mlx =
            mlx_rs::random::normal::<f32>(&[1, block_size, hidden], None, None, None).unwrap();
        let taps_mlx: Vec<Array> = (0..num_taps)
            .map(|_| {
                mlx_rs::random::normal::<f32>(&[1, ctx_len as i32, hidden], None, None, None)
                    .unwrap()
            })
            .collect();
        eval(
            std::iter::once(&noise_mlx)
                .chain(taps_mlx.iter())
                .collect::<Vec<_>>(),
        )
        .unwrap();

        // --- MLX forward ---
        let mut mlx_cache = mlx_drafter.make_cache();
        let tap_refs: Vec<Array> = taps_mlx.clone();
        let mlx_out = mlx_drafter
            .forward(&noise_mlx, &tap_refs, &mut mlx_cache)
            .unwrap();
        eval([&mlx_out]).unwrap();

        // Squeeze batch dim: [1, block, hidden] → [block * hidden]
        let mlx_flat: Vec<f32> = mlx_out
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .reshape(&[-1])
            .unwrap()
            .as_slice::<f32>()
            .to_vec();

        // --- CPU forward ---
        // Convert inputs to f32 flat
        let noise_f32: Vec<f32> = noise_mlx
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .reshape(&[-1])
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let taps_f32: Vec<Vec<f32>> = taps_mlx
            .iter()
            .map(|t| {
                t.as_dtype(mlx_rs::Dtype::Float32)
                    .unwrap()
                    .reshape(&[-1])
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec()
            })
            .collect();
        let tap_slices: Vec<&[f32]> = taps_f32.iter().map(|t| t.as_slice()).collect();

        let mut cpu_cache = cpu_engine.make_cache();
        let cpu_out = cpu_engine.forward(&noise_f32, &tap_slices, ctx_len as usize, &mut cpu_cache);

        // Compare
        assert_eq!(mlx_flat.len(), cpu_out.len());
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        for (a, b) in mlx_flat.iter().zip(cpu_out.iter()) {
            let diff = (a - b).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
        }
        let mean_diff = sum_diff / mlx_flat.len() as f32;

        eprintln!("Parity check: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}");
        eprintln!(
            "  MLX  first 8: {:?}",
            &mlx_flat[..8]
        );
        eprintln!(
            "  CPU  first 8: {:?}",
            &cpu_out[..8]
        );

        // MLX runs in float16 by default, CPU in float32.
        // Allow generous tolerance for fp16 accumulation differences.
        assert!(
            max_diff < 0.05,
            "Max diff {max_diff} exceeds tolerance 0.05"
        );
    }

    /// Benchmark: CPU BLAS drafter latency at various context lengths.
    #[test]
    #[ignore]
    fn test_dflash_cpu_latency() {
        use mlx_rs::transforms::eval;

        let snap_dir = std::path::PathBuf::from(
            "/Users/peppi/AI-Models/shared/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots",
        );
        let snap_dir = std::fs::read_dir(&snap_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let drafter = crate::dflash::load_dflash_drafter(&snap_dir).unwrap();
        let cpu_engine = extract_dflash_cpu_engine(&drafter);

        let block = cpu_engine.config.block_size;
        let h = cpu_engine.config.hidden;
        let num_taps = cpu_engine.config.num_taps;
        let iters = 20;

        for ctx_len in [16, 64, 256] {
            // Generate random f32 data
            let noise: Vec<f32> = (0..block * h).map(|i| (i as f32 * 0.001).sin()).collect();
            let tap_data: Vec<Vec<f32>> = (0..num_taps)
                .map(|t| (0..ctx_len * h).map(|i| ((i + t * 1000) as f32 * 0.001).cos()).collect())
                .collect();
            let tap_slices: Vec<&[f32]> = tap_data.iter().map(|t| t.as_slice()).collect();

            // Warmup
            let mut cache = cpu_engine.make_cache();
            let _ = cpu_engine.forward(&noise, &tap_slices, ctx_len, &mut cache);

            // Timed runs
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let mut cache = cpu_engine.make_cache();
                let t0 = std::time::Instant::now();
                let _ = cpu_engine.forward(&noise, &tap_slices, ctx_len, &mut cache);
                times.push(t0.elapsed());
            }
            times.sort();
            let median = times[iters / 2];
            let min = times[0];
            eprintln!(
                "CPU drafter ctx={ctx_len:>4}: median={:.2}ms  min={:.2}ms  ({block} tokens/round)",
                median.as_secs_f64() * 1000.0,
                min.as_secs_f64() * 1000.0,
            );
        }
    }
}
