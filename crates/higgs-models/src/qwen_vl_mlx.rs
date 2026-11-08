//! mlx-vlm dialect Qwen-VL vision tower (Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL
//! MLX exports).
//!
//! These checkpoints name their SigLIP-shaped tower in mlx-vlm style —
//! `vision_tower.patch_embed.*`, `vision_tower.blocks.{i}.{attn.qkv,attn.proj,
//! mlp.linear_fc1,mlp.linear_fc2,norm1,norm2}`, a learned
//! `vision_tower.pos_embed` table, and a `vision_tower.merger.{norm,
//! linear_fc1,linear_fc2}` projector — with a fused QKV projection per block
//! and a 5-D `[O, T, H, W, C]` (temporal) patch-conv weight.
//!
//! Static-image serving folds the temporal dimension of the conv weight
//! (identical duplicated frames make `W0·x + W1·x == (W0+W1)·x`) and splits
//! the fused QKV on the *activation* side, mirroring the Phi-3 combined-QKV
//! path. MLX exports ship the tower unquantized, so plain `Param` loading
//! suffices everywhere.

use std::collections::HashMap;

use mlx_rs::error::Exception;
use mlx_rs::{builder::Builder, module::{Module, Param}, nn, ops, ops::indexing::IndexOp, Array};

use crate::error::ModelError;
use crate::siglip::SigLipVisionConfig;

/// One mlx-vlm vision block: post-norm attention (fused QKV) + MLP.
struct MlxVisionBlock {
    norm1: nn::LayerNorm,
    qkv_proj: nn::Linear,
    attn_proj: nn::Linear,
    norm2: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
    num_heads: i32,
    head_dim: i32,
    hidden: i32,
}

/// mlx-vlm dialect Qwen-VL tower + merger projector.
pub(crate) struct MlxQwenVisionTower {
    patch_embed: nn::Conv2d,
    pos_embed: nn::Embedding,
    blocks: Vec<MlxVisionBlock>,
    merger_norm: nn::LayerNorm,
    merger_fc1: nn::Linear,
    merger_fc2: nn::Linear,
    hidden: i32,
    num_patches: i32,
    patch_size: i32,
}

impl MlxQwenVisionTower {
    /// Build the tower from a `weights` map with mlx-vlm `vision_tower.*` keys.
    #[allow(clippy::similar_names)]
    pub(crate) fn from_weights(
        config: &SigLipVisionConfig,
        merge_size: i32,
        lm_hidden: i32,
        weights: &HashMap<String, Array>,
    ) -> Result<Self, ModelError> {
        let get = |name: &str| -> Result<Array, ModelError> {
            weights
                .get(name)
                .cloned()
                .ok_or_else(|| ModelError::MissingWeight(format!("Missing weight: {name}")))
        };
        let opt = |name: &str| weights.get(name).cloned();
        let required = |name: &str| -> Result<Array, ModelError> {
            get(&format!("{name}.weight"))
        };

        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads.max(1);
        if hidden % num_heads != 0 {
            return Err(ModelError::ShapeMismatch(format!(
                "vision hidden_size {hidden} not divisible by num_heads {num_heads}"
            )));
        }
        let head_dim = hidden / num_heads;

        // Patch conv: fold the temporal dimension [O, T, H, W, C] -> [O, H, W, C].
        let conv5 = get("vision_tower.patch_embed.proj.weight")?;
        let conv_weight = fold_temporal_conv(&conv5, config.num_channels).map_err(ModelError::Mlx)?;
        let conv_bias = opt("vision_tower.patch_embed.proj.bias");
        let mut patch_embed = nn::Conv2dBuilder::new(config.num_channels, hidden, config.patch_size)
            .stride(config.patch_size)
            .bias(conv_bias.is_some())
            .build()?;
        patch_embed.weight = Param::new(conv_weight);
        if let Some(b) = conv_bias {
            patch_embed.bias = Param::new(Some(b));
        }

        // Learned position table [num_patches, hidden].
        let pos_weight = get("vision_tower.pos_embed.weight")?;
        let num_patches = i32::try_from(pos_weight.shape().first().copied().unwrap_or(0))
            .map_err(|_| ModelError::UnsupportedModel("pos_embed row count overflow".into()))?;
        let mut pos_embed = nn::Embedding::new(num_patches, hidden)?;
        pos_embed.weight = Param::new(pos_weight);

        let n_layers = usize::try_from(config.num_hidden_layers).unwrap_or(0);
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("vision_tower.blocks.{i}");
            let mut norm1 = nn::LayerNormBuilder::new(hidden)
                .eps(config.layer_norm_eps)
                .build()?;
            norm1.weight = Param::new(Some(required(&format!("{p}.norm1"))?));
            if let Some(b) = opt(&format!("{p}.norm1.bias")) {
                norm1.bias = Param::new(Some(b));
            }
            let mut norm2 = nn::LayerNormBuilder::new(hidden)
                .eps(config.layer_norm_eps)
                .build()?;
            norm2.weight = Param::new(Some(required(&format!("{p}.norm2"))?));
            if let Some(b) = opt(&format!("{p}.norm2.bias")) {
                norm2.bias = Param::new(Some(b));
            }
            let mut qkv_proj = nn::LinearBuilder::new(hidden, 3 * hidden).build()?;
            qkv_proj.weight = Param::new(get(&format!("{p}.attn.qkv.weight"))?);
            if let Some(b) = opt(&format!("{p}.attn.qkv.bias")) {
                qkv_proj.bias = Param::new(Some(b));
            }
            let mut attn_proj = nn::LinearBuilder::new(hidden, hidden).build()?;
            attn_proj.weight = Param::new(get(&format!("{p}.attn.proj.weight"))?);
            if let Some(b) = opt(&format!("{p}.attn.proj.bias")) {
                attn_proj.bias = Param::new(Some(b));
            }
            let mut fc1 = nn::LinearBuilder::new(hidden, config.intermediate_size).build()?;
            fc1.weight = Param::new(get(&format!("{p}.mlp.linear_fc1.weight"))?);
            if let Some(b) = opt(&format!("{p}.mlp.linear_fc1.bias")) {
                fc1.bias = Param::new(Some(b));
            }
            let mut fc2 = nn::LinearBuilder::new(config.intermediate_size, hidden).build()?;
            fc2.weight = Param::new(get(&format!("{p}.mlp.linear_fc2.weight"))?);
            if let Some(b) = opt(&format!("{p}.mlp.linear_fc2.bias")) {
                fc2.bias = Param::new(Some(b));
            }
            blocks.push(MlxVisionBlock {
                norm1,
                qkv_proj,
                attn_proj,
                norm2,
                fc1,
                fc2,
                num_heads,
                head_dim,
                hidden,
            });
        }

        // Merger projector: LayerNorm over the per-patch features (hidden),
        // then the 2x2 spatial merge concats to hidden * merge^2 for
        // fc1 -> gelu(tanh) -> fc2 -> LM hidden. (The Qwen2.5/3.5-VL merger
        // norms BEFORE the merge concat — the checkpoint's merger.norm is
        // hidden-wide, and its fc1 input is hidden * merge^2.)
        let merged_in = hidden
            .checked_mul(merge_size)
            .and_then(|v| v.checked_mul(merge_size))
            .ok_or_else(|| ModelError::ShapeMismatch("merger input width overflow".into()))?;
        let mut merger_norm = nn::LayerNormBuilder::new(hidden)
            .eps(config.layer_norm_eps)
            .build()?;
        merger_norm.weight = Param::new(Some(get("vision_tower.merger.norm.weight")?));
        if let Some(b) = opt("vision_tower.merger.norm.bias") {
            merger_norm.bias = Param::new(Some(b));
        }
        let mut merger_fc1 = nn::LinearBuilder::new(merged_in, config.intermediate_size).build()?;
        merger_fc1.weight = Param::new(get("vision_tower.merger.linear_fc1.weight")?);
        if let Some(b) = opt("vision_tower.merger.linear_fc1.bias") {
            merger_fc1.bias = Param::new(Some(b));
        }
        let mut merger_fc2 = nn::LinearBuilder::new(config.intermediate_size, lm_hidden).build()?;
        merger_fc2.weight = Param::new(get("vision_tower.merger.linear_fc2.weight")?);
        if let Some(b) = opt("vision_tower.merger.linear_fc2.bias") {
            merger_fc2.bias = Param::new(Some(b));
        }

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            merger_norm,
            merger_fc1,
            merger_fc2,
            hidden,
            num_patches,
            patch_size: config.patch_size,
        })
    }

    /// Fixed square processing side (px) implied by the learned pos table.
    pub(crate) fn target_side(&self) -> i32 {
        let side = (f64::from(self.num_patches).sqrt().round() as i32).max(1);
        side * self.patch_size
    }

    /// Nominal patch-grid capacity of the learned position table.
    pub(crate) const fn num_patches(&self) -> i32 {
        self.num_patches
    }

    /// Encode one image canvas `[1, H, W, 3]` -> `[1, gh*gw, hidden]`.
    pub(crate) fn forward(&mut self, pixel_values: &Array) -> Result<Array, Exception> {
        let patch_embeds = self.patch_embed.forward(pixel_values)?; // [1, gh, gw, hidden]
        let &[1, gh, gw, _] = patch_embeds.shape() else {
            return Err(Exception::custom(
                "MlxQwenVisionTower: expected [1, gh, gw, hidden] conv output",
            ));
        };
        let feats = patch_embeds.reshape(&[1, gh * gw, self.hidden])?;
        // Position lookup over the flat grid: the conv output's row-major
        // order matches the learned table's layout.
        let idx: Vec<i32> = (0..gh * gw).collect();
        let pos = self
            .pos_embed
            .forward(&Array::from_slice(&idx, &[gh * gw]))?
            .reshape(&[1, gh * gw, self.hidden])?;
        let mut x = feats.add(&pos)?;
        for block in &mut self.blocks {
            let normed = block.norm1.forward(&x)?;
            let attn_out = block.attention(&normed)?;
            let after_attn = x.add(&attn_out)?;
            let normed2 = block.norm2.forward(&after_attn)?;
            let mlp_out = block.mlp(&normed2)?;
            x = after_attn.add(&mlp_out)?;
        }
        Ok(x)
    }

    /// Project raw tower features `[1, gh*gw, hidden]` -> `[1, (gh/m)*(gw/m),
    /// lm_hidden]`: LayerNorm over patch features, 2x2 spatial merge concat,
    /// then fc1 -> gelu(tanh) -> fc2.
    pub(crate) fn project(
        &mut self,
        feats: &Array,
        gh: i32,
        gw: i32,
        merge: i32,
    ) -> Result<Array, Exception> {
        let normed = self.merger_norm.forward(feats)?; // [1, gh*gw, hidden]
        let m = merge.max(1);
        let hidden = self.hidden_size();
        let merged = normed
            .reshape(&[1, gh / m, m, gw / m, m, hidden])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[1, (gh / m) * (gw / m), hidden * m * m])?;
        let h = self.merger_fc1.forward(&merged)?;
        let act = nn::gelu_approximate(&h)?;
        Ok(self.merger_fc2.forward(&act)?.index(0)) // [tokens, lm_hidden]
    }

    /// Per-patch feature width of the tower.
    pub(crate) const fn hidden_size(&self) -> i32 {
        self.hidden
    }
}

impl MlxVisionBlock {
    /// Fused-QKV attention: split on the activation side (Phi-3 pattern).
    fn attention(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        let &[b, seq_len, _] = hidden_states.shape() else {
            return Err(Exception::custom("MlxVisionBlock: expected 3D input"));
        };
        let qkv = self.qkv_proj.forward(hidden_states)?; // [b, s, 3*hidden]
        let hd = self.head_dim;
        let heads = self.num_heads;
        let hidden = self.hidden;
        let queries = qkv
            .index((.., .., 0..hidden))
            .reshape(&[b, seq_len, heads, hd])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = qkv
            .index((.., .., hidden..2 * hidden))
            .reshape(&[b, seq_len, heads, hd])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = qkv
            .index((.., .., 2 * hidden..3 * hidden))
            .reshape(&[b, seq_len, heads, hd])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let scale = f64::from(hd).powf(-0.5);
        let attn_output = sdpa(&queries, &keys, &values, scale as f32, b, seq_len, heads, hd)?;
        self.attn_proj.forward(&attn_output)
    }

    fn mlp(&mut self, input: &Array) -> Result<Array, Exception> {
        let h = self.fc1.forward(input)?;
        let act = nn::gelu_approximate(&h)?;
        self.fc2.forward(&act)
    }
}

/// Scaled dot-product attention on pre-split [b, h, s, d] tensors, mirroring
/// `SigLipAttention::forward`.
fn sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    b: i32,
    seq_len: i32,
    heads: i32,
    head_dim: i32,
) -> Result<Array, Exception> {
    let scores_raw = ops::matmul(queries, &keys.transpose_axes(&[0, 1, 3, 2])?)?;
    let scale_arr = Array::from_f32(scale).as_dtype(scores_raw.dtype())?;
    let scores = scores_raw.multiply(&scale_arr)?;
    let weights = ops::softmax_axis(&scores, -1, None)?;
    Ok(ops::matmul(&weights, values)?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, seq_len, heads * head_dim])?)
}

/// Fold a 5-D temporal patch-conv weight into a 4-D conv weight by summing
/// over the temporal axis — exact for static-image serving where the T frames
/// the reference implementation feeds are identical.
///
/// Two 5-D layouts ship in the wild, told apart by which axis carries the
/// input channels (`num_channels`):
/// - MLX conv3d export `[O, T, H, W, C]` (mlx-vlm dialect), channels last;
/// - PyTorch Conv3d `[O, C, T, H, W]` (native HF multimodal releases),
///   channels second.
/// Output is the MLX conv2d layout `[O, H, W, C]`.
fn fold_temporal_conv(conv5: &Array, num_channels: i32) -> Result<Array, Exception> {
    let shape = conv5.shape();
    let &[d0, d1, d2, d3, d4] = shape else {
        return Err(Exception::custom(
            "expected 5-D [O, T, H, W, C] / [O, C, T, H, W] patch-conv weight",
        ));
    };
    let (mlx_layout, t) = if d4 == num_channels {
        (true, d1)
    } else if d1 == num_channels {
        (false, d2)
    } else {
        return Err(Exception::custom(format!(
            "patch-conv weight {shape:?} matches neither MLX [O,T,H,W,C] nor \
             PyTorch [O,C,T,H,W] layout with {num_channels} input channels"
        )));
    };
    // Normalize PyTorch [O, C, T, H, W] -> [O, T, H, W, C].
    let normalized = if mlx_layout {
        conv5.clone()
    } else {
        conv5.transpose_axes(&[0, 2, 3, 4, 1])?
    };
    // MLX layout: [O, T, H, W, C] = (d0..d4) as shipped.
    // PyTorch layout: [O, C, T, H, W] as shipped -> normalized [O, T, H, W, C].
    let (o, hh, ww, c) = if mlx_layout {
        (d0, d2, d3, d4)
    } else {
        (d0, d3, d4, d1)
    };
    let target = [o, hh, ww, c];
    if t == 1 {
        return normalized.reshape(&target);
    }
    let mut acc: Option<Array> = None;
    for ti in 0..t {
        let slice = normalized
            .index((.., ti..ti + 1, .., .., ..))
            .reshape(&target)?;
        acc = Some(match acc {
            None => slice,
            Some(prev) => prev.add(&slice)?,
        });
    }
    acc.ok_or_else(|| Exception::custom("empty temporal conv weight"))
}
