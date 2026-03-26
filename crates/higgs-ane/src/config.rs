//! Model configuration parsed from config.json.

use std::path::Path;

/// Model configuration (dimensions, heads, RoPE, etc.).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub rms_eps: f32,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub group_size: usize,
    pub bits: usize,
    pub attn_output_gate: bool,
}

impl ModelConfig {
    /// Parse model config from a directory's `config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("read {}: {e}", config_path.display()))?;
        let root: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("parse {}: {e}", config_path.display()))?;

        let tc = root.get("text_config").unwrap_or(&root);

        let read_usize = |field: &str| -> Result<usize, String> {
            tc.get(field)
                .or_else(|| root.get(field))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json missing field: {field}"))
        };

        let dim = read_usize("hidden_size")?;
        let hidden_dim = tc
            .get("intermediate_size")
            .or_else(|| tc.get("moe_intermediate_size"))
            .and_then(|v| v.as_u64())
            .ok_or("missing intermediate_size")?
            as usize;
        let n_heads = read_usize("num_attention_heads")?;
        let n_kv_heads = tc
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(n_heads as u64) as usize;
        let head_dim = tc
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or((dim / n_heads) as u64) as usize;
        let rope_theta = tc
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| tc.get("rope_theta").and_then(|v| v.as_f64()))
            .or_else(|| root.get("rope_theta").and_then(|v| v.as_f64()))
            .unwrap_or(1_000_000.0);
        let rms_eps = tc
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;
        let attn_output_gate = tc
            .get("attn_output_gate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let quant = root
            .get("quantization")
            .or_else(|| root.get("quantization_config"));
        let group_size = quant
            .and_then(|q| q.get("group_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let bits = quant
            .and_then(|q| q.get("bits"))
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;

        Ok(Self {
            dim,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
            rms_eps,
            n_layers: read_usize("num_hidden_layers")?,
            vocab_size: read_usize("vocab_size")?,
            group_size,
            bits,
            attn_output_gate,
        })
    }

    /// Number of Q heads per KV group (for GQA).
    pub fn heads_per_group(&self) -> usize {
        if self.n_kv_heads == 0 {
            1
        } else {
            self.n_heads / self.n_kv_heads
        }
    }

    /// KV dimension: n_kv_heads * head_dim.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Q projection dimension (includes gate if attn_output_gate).
    pub fn q_proj_dim(&self) -> usize {
        let attn_dim = self.n_heads * self.head_dim;
        if self.attn_output_gate {
            attn_dim * 2
        } else {
            attn_dim
        }
    }
}
