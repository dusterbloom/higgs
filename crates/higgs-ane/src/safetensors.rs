//! Memory-mapped safetensors store for zero-copy weight access.
//!
//! Ported from nanobot-rs ane_weights.rs.

use std::collections::HashMap;
use std::io;
use std::path::Path;

/// Memory-mapped safetensors store for on-demand tensor access.
pub(crate) struct MmapTensorStore {
    _mmaps: Vec<memmap2::Mmap>,
    offsets: HashMap<String, (usize, usize, usize)>,
    meta: HashMap<String, (String, Vec<usize>)>,
}

impl MmapTensorStore {
    /// Open all safetensors files in `dir`, building an mmap-backed index.
    pub fn open(dir: &Path) -> io::Result<Self> {
        let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        st_files.sort();

        if st_files.is_empty() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no safetensors files"));
        }

        let mut mmaps = Vec::with_capacity(st_files.len());
        let mut offsets = HashMap::new();
        let mut meta = HashMap::new();

        for st_path in &st_files {
            let file = std::fs::File::open(st_path)?;
            // SAFETY: file is read-only, mmap lifetime is tied to MmapTensorStore.
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let mmap_idx = mmaps.len();

            if mmap.len() < 8 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
            }
            let hdr_size =
                u64::from_le_bytes(mmap[..8].try_into().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "header size")
                })?) as usize;
            let hdr_json: serde_json::Value =
                serde_json::from_slice(&mmap[8..8 + hdr_size]).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("bad header: {e}"))
                })?;
            let data_start = 8 + hdr_size;

            if let serde_json::Value::Object(map) = hdr_json {
                for (name, m) in &map {
                    if name == "__metadata__" {
                        continue;
                    }
                    // Skip MoE expert tensors (draft models should be dense)
                    if name.contains(".experts.") {
                        continue;
                    }
                    let dtype = m["dtype"].as_str().unwrap_or("").to_string();
                    let shape: Vec<usize> = m["shape"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                .collect()
                        })
                        .unwrap_or_default();
                    let data_offsets = m["data_offsets"]
                        .as_array()
                        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing data_offsets"))?;
                    let start = data_offsets[0].as_u64().unwrap_or(0) as usize;
                    let end = data_offsets[1].as_u64().unwrap_or(0) as usize;
                    offsets.insert(name.clone(), (mmap_idx, data_start + start, data_start + end));
                    meta.insert(name.clone(), (dtype, shape));
                }
            }

            mmaps.push(mmap);
        }

        tracing::debug!(
            "mmap tensor store: indexed {} tensors from {} files",
            offsets.len(),
            mmaps.len()
        );

        Ok(Self { _mmaps: mmaps, offsets, meta })
    }

    /// Get raw bytes for a tensor (zero-copy slice into mmap).
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        let &(idx, start, end) = self.offsets.get(name)?;
        Some(&self._mmaps[idx][start..end])
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.offsets.contains_key(name)
    }

    pub fn meta(&self, name: &str) -> Option<&(String, Vec<usize>)> {
        self.meta.get(name)
    }

    /// Resolve weight base name, trying `language_model.` prefix fallback.
    pub fn resolve_weight_base(&self, base: &str) -> String {
        let direct = format!("{base}.weight");
        if self.contains_key(&direct) {
            return base.to_string();
        }
        let prefixed = format!("language_model.{base}");
        let prefixed_weight = format!("{prefixed}.weight");
        if self.contains_key(&prefixed_weight) {
            prefixed
        } else {
            base.to_string()
        }
    }

    /// Resolve tensor name, trying `language_model.` prefix fallback.
    pub fn resolve_tensor_name(&self, name: &str) -> String {
        if self.contains_key(name) {
            return name.to_string();
        }
        let prefixed = format!("language_model.{name}");
        if self.contains_key(&prefixed) {
            prefixed
        } else {
            name.to_string()
        }
    }
}

/// BF16 raw bytes → f32 vec.
pub(crate) fn bf16_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// Dequantize N-bit packed weights (MLX format).
///
/// MLX stores quantized values as u32 words in little-endian byte order,
/// with values packed LSB-first within each word.
pub(crate) fn dequant_nbit(
    weight: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    bits: usize,
) -> Vec<f32> {
    let n_groups = cols / group_size;
    let mask = (1u32 << bits) - 1;
    let packed_cols = (cols * bits + 31) / 32;
    let mut out = vec![0.0f32; rows * cols];
    let spans_words = (32 % bits) != 0;

    for r in 0..rows {
        let row_byte_offset = r * packed_cols * 4;
        for c in 0..cols {
            let qval = if spans_words {
                let bit_offset = c * bits;
                let word_idx = bit_offset / 32;
                let bit_within_word = bit_offset % 32;
                let byte_off = row_byte_offset + word_idx * 4;
                let lo_word = u32::from_le_bytes([
                    weight[byte_off],
                    weight[byte_off + 1],
                    weight[byte_off + 2],
                    weight[byte_off + 3],
                ]);
                if bit_within_word + bits <= 32 {
                    ((lo_word >> bit_within_word) & mask) as f32
                } else {
                    let lo_bits = 32 - bit_within_word;
                    let hi_byte_off = byte_off + 4;
                    let hi_word = u32::from_le_bytes([
                        weight[hi_byte_off],
                        weight[hi_byte_off + 1],
                        weight[hi_byte_off + 2],
                        weight[hi_byte_off + 3],
                    ]);
                    let lo = lo_word >> bit_within_word;
                    let hi = hi_word & ((1u32 << (bits - lo_bits)) - 1);
                    (lo | (hi << lo_bits)) as f32
                }
            } else {
                let elems_per_u32 = 32 / bits;
                let word_idx = c / elems_per_u32;
                let elem_idx = c % elems_per_u32;
                let byte_off = row_byte_offset + word_idx * 4;
                let u32_val = u32::from_le_bytes([
                    weight[byte_off],
                    weight[byte_off + 1],
                    weight[byte_off + 2],
                    weight[byte_off + 3],
                ]);
                ((u32_val >> (elem_idx * bits)) & mask) as f32
            };
            let g = c / group_size;
            let s = scales[r * n_groups + g];
            let b = biases[r * n_groups + g];
            out[r * cols + c] = s * qval + b;
        }
    }
    out
}
