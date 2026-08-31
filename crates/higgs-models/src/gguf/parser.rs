//! Minimal GGUF v3 parser: magic, header, tensor infos, kv pairs.
//!
//! Only extracts what higgs needs: model architecture, tensor data offsets,
//! and tokenizer info. Weight tensors are returned as (name, dtype, data)
//! triples for dequantization by the format-specific modules (q4_k, etc).

use std::collections::HashMap;

pub struct GgufFile {
    pub version: u32,
    pub tensors: HashMap<String, TensorInfo>,
    pub metadata: HashMap<String, GgufValue>,
    pub data: Vec<u8>,  // full file bytes (tensor data is at tensor.offset)
}

pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub dtype: u32,     // GGUF quant type (0=f32, 1=f16, 8=Q8_0, 12=Q4_K, etc.)
    pub offset: u64,    // offset into data section
}

pub enum GgufValue {
    U8(u8), I16(i16), U32(u32), I32(i32), F32(f32), Bool(bool),
    String(String), Array(Vec<GgufValue>), U64(u64), I64(i64), F64(f64),
}

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

impl GgufFile {
    pub fn parse(data: Vec<u8>) -> Result<GgufFile, String> {
        if data.len() < 4 || &data[0..4] != GGUF_MAGIC {
            return Err("not a GGUF file (bad magic)".into());
        }
        let mut pos = 4usize;
        let read_u32 = |p: &mut usize| { let v = u32::from_le_bytes(data[*p..*p+4].try_into().unwrap()); *p += 4; v };
        let read_u64 = |p: &mut usize| { let v = u64::from_le_bytes(data[*p..*p+8].try_into().unwrap()); *p += 8; v };

        let version = read_u32(&mut pos);
        let n_tensors = read_u32(&mut pos) as usize;
        let n_kv = read_u32(&mut pos) as usize;

        // Read kv pairs (metadata)
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let klen = read_u32(&mut pos) as usize;
            let key = String::from_utf8_lossy(&data[pos..pos+klen]).to_string();
            pos += klen;
            let vtype = read_u32(&mut pos);
            // Skip value (types vary — read/skip based on type)
            let (val, skip) = read_gguf_value(&data, pos, vtype)?;
            pos = skip;
            metadata.insert(key, val);
        }

        // Read tensor infos
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let nlen = read_u32(&mut pos) as usize;
            let name = String::from_utf8_lossy(&data[pos..pos+nlen]).to_string();
            pos += nlen;
            let n_dims = read_u32(&mut pos);
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims { dims.push(read_u64(&mut pos)); }
            let dtype = read_u32(&mut pos);
            let offset = read_u64(&mut pos);
            tensors.insert(name.clone(), TensorInfo { name, n_dims, dims, dtype, offset });
        }

        // Align to 32 bytes
        pos = (pos + 31) & !31;

        Ok(GgufFile { version, tensors, metadata, data })
    }
}

fn read_gguf_value(data: &[u8], mut pos: usize, vtype: u32) -> Result<(GgufValue, usize), String> {
    let rd_u32 = |p: &mut usize| { let v = u32::from_le_bytes(data[*p..*p+4].try_into().unwrap()); *p += 4; v };
    let rd_i32 = |p: &mut usize| { let v = i32::from_le_bytes(data[*p..*p+4].try_into().unwrap()); *p += 4; v };
    let rd_str = |p: &mut usize| {
        let len = u64::from_le_bytes(data[*p..*p+8].try_into().unwrap()) as usize;
        *p += 8;
        let s = String::from_utf8_lossy(&data[*p..*p+len]).to_string();
        *p += len;
        s
    };
    match vtype {
        0 => { let v = data[pos]; pos += 1; Ok((GgufValue::U8(v), pos)) }
        2 => { let v = rd_u32(&mut pos); Ok((GgufValue::U32(v), pos)) }
        4 => { let v = rd_i32(&mut pos); Ok((GgufValue::I32(v), pos)) }
        6 => { let v = f32::from_le_bytes(data[pos..pos+4].try_into().unwrap()); pos += 4; Ok((GgufValue::F32(v), pos)) }
        8 => { let s = rd_str(&mut pos); Ok((GgufValue::String(s), pos)) }
        9 => { // array
            let elem_type = rd_u32(&mut pos);
            let count = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap()) as usize;
            pos += 8;
            let mut vals = Vec::new();
            for _ in 0..count {
                let (v, new_pos) = read_gguf_value(data, pos, elem_type)?;
                vals.push(v);
                pos = new_pos;
            }
            Ok((GgufValue::Array(vals), pos))
        }
        _ => Err(format!("unsupported GGUF value type {vtype}")),
    }
}
