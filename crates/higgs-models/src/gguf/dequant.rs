//! Dequant dispatch for GGML quantization types found in GGUF files.
//!
//! Each decoder mirrors the gguf-py reference (`gguf/quants.py`) — the same
//! source the Q4_K real-file validation used — so per-type oracle checks
//! transfer directly.

use super::q4_k;

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q5_0: u32 = 6;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q6_K: u32 = 14;
pub const BF16: u32 = 30;

/// Dequantize one tensor's raw bytes to f32 row-major values.
pub fn dequant_tensor(dtype: u32, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        F32 => Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()),
        F16 => Ok(bytes.chunks_exact(2).map(|b| q4_k::f16_to_f32(b)).collect()),
        BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16))
            .collect()),
        Q8_0 => dequant_q8_0(bytes),
        Q5_0 => dequant_q5_0(bytes),
        Q4_K => {
            if bytes.len() % 144 != 0 {
                return Err(format!(
                    "Q4_K byte count {} not a multiple of 144",
                    bytes.len()
                ));
            }
            Ok(bytes
                .chunks_exact(144)
                .flat_map(q4_k::dequant_super_block)
                .collect())
        }
        Q6_K => dequant_q6_k(bytes),
        other => Err(format!("unsupported tensor dtype {other}")),
    }
}

/// Q8_0: 34-byte blocks — f16 d, 32 × i8. y = d * q.
fn dequant_q8_0(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 34 != 0 {
        return Err(format!(
            "Q8_0 byte count {} not a multiple of 34",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(34)
        .flat_map(|block| {
            let d = q4_k::f16_to_f32(&block[0..2]);
            block[2..34].iter().map(move |b| d * (*b as i8) as f32)
        })
        .collect())
}

/// Q5_0: 22-byte blocks — f16 d, u32 qh (1 high bit per value), 16 bytes of
/// nibbles (16 lows then 16 highs). y = d * (nibble | bit<<4 - 16).
fn dequant_q5_0(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 22 != 0 {
        return Err(format!(
            "Q5_0 byte count {} not a multiple of 22",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(22)
        .flat_map(|block| {
            let d = q4_k::f16_to_f32(&block[0..2]);
            let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
            let qs = &block[6..22];
            (0..32).map(move |j| {
                let nibble = if j < 16 {
                    (qs[j] & 0xF) as u32
                } else {
                    (qs[j - 16] >> 4) as u32
                };
                let bit = (qh >> j) & 1;
                let q = ((nibble | (bit << 4)) as u8 as i8).wrapping_sub(16);
                d * q as f32
            })
        })
        .collect())
}

/// Q6_K: 210-byte blocks — 128 B low nibbles, 64 B high 2-bit pairs,
/// 16 × i8 scales, f16 d. 16-value groups, y = d * scale[g] * q,
/// q = nibble | bits<<4 - 32.
///
/// Flat element order (mirroring gguf-py's reshapes): element i of each
/// 128-value half h draws its low nibble from ql[h*64 + i%64] shifted by
/// (i/64)*4, and its 2 high bits from qh[h*32 + i%32] shifted by (i/32)*2.
fn dequant_q6_k(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 210 != 0 {
        return Err(format!(
            "Q6_K byte count {} not a multiple of 210",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(210)
        .flat_map(|block| {
            let d = q4_k::f16_to_f32(&block[208..210]);
            let ql = &block[0..128];
            let qh = &block[128..192];
            (0..256).map(move |i| {
                let half = i / 128;
                let within = i % 128;
                let nib = within / 64;
                let b = within % 64;
                let low = (ql[half * 64 + b] >> (nib * 4)) & 0xF;
                let hb = within % 32;
                let shift = (within / 32) * 2;
                let high = (qh[half * 32 + hb] >> shift) & 0x3;
                let q = (low | (high << 4)) as i8 - 32;
                let scale = block[192 + (i / 16)] as i8;
                d * scale as f32 * q as f32
            })
        })
        .collect())
}
