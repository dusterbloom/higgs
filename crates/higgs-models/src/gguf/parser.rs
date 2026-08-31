//! Minimal GGUF v3 parser: magic, header, tensor infos, kv pairs.
//!
//! Only extracts what higgs needs: model architecture, tensor data offsets,
//! and tokenizer info. Weight tensors are returned as (name, dtype, data)
//! triples for dequantization by the format-specific modules (q4_k, etc).
//!
//! Value types follow the GGUF spec (all 13); `TensorInfo.offset` is
//! relative to the data section, whose absolute start is [`GgufFile::data_start`].

use std::collections::HashMap;

pub struct GgufFile {
    pub version: u32,
    pub tensors: HashMap<String, TensorInfo>,
    pub metadata: HashMap<String, GgufValue>,
    pub data: Vec<u8>, // full file bytes (tensor data is at data_start + offset)
    /// Absolute byte offset of the tensor data section (aligned).
    pub data_start: usize,
}

pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    /// GGUF dimension order: dims[0] is the fastest-varying (row length).
    pub dims: Vec<u64>,
    pub dtype: u32, // GGMLQuantizationType: 0=F32, 1=F16, 8=Q8_0, 12=Q4_K, ...
    pub offset: u64, // offset into the data section
}

#[derive(Debug)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

impl GgufFile {
    pub fn parse(data: Vec<u8>) -> Result<GgufFile, String> {
        if data.len() < 4 || &data[0..4] != GGUF_MAGIC {
            return Err("not a GGUF file (bad magic)".into());
        }
        let mut pos = 4usize;
        let read_u32 = |p: &mut usize| {
            let v = u32::from_le_bytes(data[*p..*p + 4].try_into().unwrap());
            *p += 4;
            v
        };
        let read_u64 = |p: &mut usize| {
            let v = u64::from_le_bytes(data[*p..*p + 8].try_into().unwrap());
            *p += 8;
            v
        };

        let version = read_u32(&mut pos);
        if !(2..=3).contains(&version) {
            return Err(format!("unsupported GGUF version {version}"));
        }
        // Counts are u64 per the spec.
        let n_tensors = read_u64(&mut pos) as usize;
        let n_kv = read_u64(&mut pos) as usize;

        // Read kv pairs (metadata). `general.alignment` is read before the
        // data section is aligned, so two passes over the map: parse all,
        // then look the alignment up.
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            // Keys are gguf_strings: u64 length prefix.
            let klen = read_u64(&mut pos) as usize;
            let key = String::from_utf8_lossy(&data[pos..pos + klen]).to_string();
            pos += klen;
            let vtype = read_u32(&mut pos);
            let (val, new_pos) = read_gguf_value(&data, pos, vtype)?;
            pos = new_pos;
            metadata.insert(key, val);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) if *a >= 8 => *a as u64,
            _ => DEFAULT_ALIGNMENT,
        };

        // Read tensor infos (names are gguf_strings: u64 length prefix)
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let nlen = read_u64(&mut pos) as usize;
            let name = String::from_utf8_lossy(&data[pos..pos + nlen]).to_string();
            pos += nlen;
            let n_dims = read_u32(&mut pos);
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut pos));
            }
            let dtype = read_u32(&mut pos);
            let offset = read_u64(&mut pos);
            tensors.insert(name.clone(), TensorInfo { name, n_dims, dims, dtype, offset });
        }

        // Align to the declared alignment
        let align_mask = alignment - 1;
        let data_start = ((pos as u64 + align_mask) & !align_mask) as usize;
        if data_start > data.len() {
            return Err("GGUF data section starts past end of file".into());
        }

        Ok(GgufFile { version, tensors, metadata, data, data_start })
    }

    /// The raw bytes of one tensor.
    pub fn tensor_bytes(&self, name: &str) -> Option<Result<&[u8], String>> {
        let info = self.tensors.get(name)?;
        let start = self.data_start + info.offset as usize;
        let end = start + tensor_nbytes(info)?;
        if end > self.data.len() {
            return Some(Err(format!("tensor {name} extends past end of file")));
        }
        Some(Ok(&self.data[start..end]))
    }
}

/// Byte length of one tensor from its dims and quant type.
pub fn tensor_nbytes(info: &TensorInfo) -> Option<usize> {
    let n_elems: u64 = info.dims.iter().product();
    // (type, bytes per block, values per block); f32/f16 are per-element.
    let (per_block, block_len): (u64, u64) = match info.dtype {
        0 => (1, 4),   // F32
        1 => (1, 2),   // F16
        24 => (1, 1),  // BF16
        2 => (32, 18), // Q4_0
        3 => (32, 20), // Q4_1
        6 => (32, 21), // Q5_0
        7 => (32, 22), // Q5_1
        8 => (32, 34), // Q8_0
        12 => (256, 144), // Q4_K
        13 => (256, 210), // Q5_K
        14 => (256, 84),  // Q6_K
        16 => (32, 11),   // IQ2_XXS? no: 16 = IQ1_S
        _ => return None,
    };
    Some((n_elems.div_ceil(per_block) * block_len) as usize)
}

fn read_gguf_value(data: &[u8], mut pos: usize, vtype: u32) -> Result<(GgufValue, usize), String> {
    let rd_u8 = |p: &mut usize| {
        let v = data[*p];
        *p += 1;
        v
    };
    let rd_i8 = |p: &mut usize| {
        let v = i8::from_le_bytes([data[*p]]);
        *p += 1;
        v
    };
    let rd_u16 = |p: &mut usize| {
        let v = u16::from_le_bytes(data[*p..*p + 2].try_into().unwrap());
        *p += 2;
        v
    };
    let rd_i16 = |p: &mut usize| {
        let v = i16::from_le_bytes(data[*p..*p + 2].try_into().unwrap());
        *p += 2;
        v
    };
    let rd_u32 = |p: &mut usize| {
        let v = u32::from_le_bytes(data[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let rd_i32 = |p: &mut usize| {
        let v = i32::from_le_bytes(data[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let rd_u64 = |p: &mut usize| {
        let v = u64::from_le_bytes(data[*p..*p + 8].try_into().unwrap());
        *p += 8;
        v
    };
    let rd_i64 = |p: &mut usize| {
        let v = i64::from_le_bytes(data[*p..*p + 8].try_into().unwrap());
        *p += 8;
        v
    };
    let rd_f32 = |p: &mut usize| {
        let v = f32::from_le_bytes(data[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let rd_f64 = |p: &mut usize| {
        let v = f64::from_le_bytes(data[*p..*p + 8].try_into().unwrap());
        *p += 8;
        v
    };
    let rd_str = |p: &mut usize| {
        let len = u64::from_le_bytes(data[*p..*p + 8].try_into().unwrap()) as usize;
        *p += 8;
        let s = String::from_utf8_lossy(&data[*p..*p + len]).to_string();
        *p += len;
        s
    };
    match vtype {
        0 => Ok((GgufValue::U8(rd_u8(&mut pos)), pos)),
        1 => Ok((GgufValue::I8(rd_i8(&mut pos)), pos)),
        2 => Ok((GgufValue::U16(rd_u16(&mut pos)), pos)),
        3 => Ok((GgufValue::I16(rd_i16(&mut pos)), pos)),
        4 => Ok((GgufValue::U32(rd_u32(&mut pos)), pos)),
        5 => Ok((GgufValue::I32(rd_i32(&mut pos)), pos)),
        6 => Ok((GgufValue::F32(rd_f32(&mut pos)), pos)),
        7 => Ok((GgufValue::Bool(rd_u8(&mut pos) != 0), pos)),
        8 => Ok((GgufValue::String(rd_str(&mut pos)), pos)),
        9 => {
            let elem_type = rd_u32(&mut pos);
            let count = rd_u64(&mut pos) as usize;
            let mut vals = Vec::with_capacity(count.min(1 << 20));
            for _ in 0..count {
                let (v, new_pos) = read_gguf_value(data, pos, elem_type)?;
                vals.push(v);
                pos = new_pos;
            }
            Ok((GgufValue::Array(vals), pos))
        }
        10 => Ok((GgufValue::U64(rd_u64(&mut pos)), pos)),
        11 => Ok((GgufValue::I64(rd_i64(&mut pos)), pos)),
        12 => Ok((GgufValue::F64(rd_f64(&mut pos)), pos)),
        _ => Err(format!("unsupported GGUF value type {vtype}")),
    }
}
