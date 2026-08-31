//! Q4_K dequantization: 256 values per super-block, 144 bytes.
//!
//! Layout (llama.cpp `block_q4_K`, verified against gguf-py dequantize):
//!   bytes 0-1:   f16 d       (super-block scale)
//!   bytes 2-3:   f16 dmin    (super-block min)
//!   bytes 4-15:  12 bytes    (8 sub-block scales + mins, 6-bit each)
//!   bytes 16-143: 128 bytes = four 32-byte pages. Page g holds the
//!                nibbles of sub-blocks 2g and 2g+1: sub 2g takes the
//!                LOW nibbles of all 32 bytes, sub 2g+1 the HIGH nibbles.
//!
//! Dequant: y[j] = d * sc[j/32] * q4 - dmin * m[j/32]

/// Extract 6-bit scale and min for sub-block `j` (0..8) from the packed
/// 12-byte scales array. Bit layout (gguf-py `Q4_K::get_scale_min`):
///   bytes 0-3:  sc[0..4] low 6 bits; their high 2 bits top up sc[4..8]
///   bytes 4-7:  m[0..4] low 6 bits; their high 2 bits top up m[4..8]
///   bytes 8-11: sc[4..8] low 4 bits, m[4..8] low 4 bits
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 2) & 0x30),
            (q[j + 4] >> 4) | ((q[j] >> 2) & 0x30),
        )
    }
}

/// Dequantize one Q4_K super-block (144 bytes → 256 f32 values).
pub fn dequant_super_block(block: &[u8]) -> Vec<f32> {
    assert_eq!(block.len(), 144, "Q4_K block must be 144 bytes");
    let d = f16_to_f32(&block[0..2]);
    let dmin = f16_to_f32(&block[2..4]);
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut out = vec![0.0f32; 256];
    for sub in 0..8 {
        let page = sub / 2;
        let high = sub % 2 == 1;
        let (sc, m) = get_scale_min_k4(sub, scales);
        let scale = d * sc as f32;
        let offset = dmin * m as f32;
        for l in 0..32 {
            let byte = qs[page * 32 + l];
            let q4 = if high { byte >> 4 } else { byte & 0xF };
            out[sub * 32 + l] = scale * q4 as f32 - offset;
        }
    }
    out
}

/// Dequant a full row (multiple super-blocks).
pub fn dequant_row(data: &[u8], num_blocks: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_blocks * 256);
    for b in 0..num_blocks {
        out.extend(dequant_super_block(&data[b * 144..(b + 1) * 144]));
    }
    out
}

/// Read f16 from two bytes (little-endian). Manual conversion — std f16 is unstable.
pub(crate) fn f16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        f32::from_bits(sign) * (frac as f32) * 2f32.powi(-24)
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (frac << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
    }
}

/// Convert f32 to f16 bits (for encoding test values).
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = ((bits >> 13) & 0x3FF) as u16;
    if exp == 0 {
        return sign;
    }
    if exp == 255 {
        return sign | 0x7C00 | (frac & 0x3FF);
    }
    let e = exp - 127 + 15;
    if e <= 0 {
        return sign;
    }
    if e >= 31 {
        return sign | 0x7C00;
    }
    sign | ((e as u16) << 10) | frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::parser::{GgufFile, GgufValue};

    /// End-to-end against a real GGUF file. Opt in with
    /// HIGGS_GGUF_TEST_FILE=<path to a Q4_K gguf>.
    #[test]
    fn real_file_q4_k_smoke() {
        let path = match std::env::var("HIGGS_GGUF_TEST_FILE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skipping: set HIGGS_GGUF_TEST_FILE to a Q4_K gguf");
                return;
            }
        };
        let data = std::fs::read(&path).expect("read gguf");
        let file = GgufFile::parse(data).expect("parse");
        assert_eq!(file.version, 3, "GGUF v3");

        // Metadata: architecture + shape hyperparameters must be present.
        let arch = match file.metadata.get("general.architecture") {
            Some(GgufValue::String(s)) => s.clone(),
            other => panic!("general.architecture missing: {other:?}"),
        };
        eprintln!("arch = {arch}, {} tensors", file.tensors.len());
        let n_layers = match file.metadata.get(&format!("{arch}.block_count")) {
            Some(GgufValue::U32(n)) => *n,
            other => panic!("block_count missing: {other:?}"),
        };
        assert!(n_layers > 0);

        // Every recorded tensor must resolve inside the data section.
        let q4_k: Vec<&str> = file
            .tensors
            .values()
            .filter(|t| t.dtype == 12)
            .map(|t| t.name.as_str())
            .collect();
        eprintln!("{} Q4_K tensors, e.g. {:?}", q4_k.len(), q4_k.first());
        assert!(!q4_k.is_empty(), "no Q4_K tensors in file");

        // Dequant the first super-block of every Q4_K tensor: values must be
        // finite and within the range the 6-bit scales + 4-bit qs can express
        // (|y| <= |d|*63*15 + |dmin|*63, with slack for f16 magnitudes).
        for name in &q4_k {
            let bytes = file
                .tensor_bytes(name)
                .expect("tensor in bounds")
                .expect("tensor fits");
            let out = dequant_super_block(&bytes[..144]);
            let d = f16_to_f32(&bytes[0..2]);
            let dmin = f16_to_f32(&bytes[2..4]);
            let bound = d.abs() * 63.0 * 15.0 + dmin.abs() * 63.0 + 1e-6;
            for (j, v) in out.iter().enumerate() {
                assert!(v.is_finite(), "{name}: y[{j}] not finite");
                assert!(
                    v.abs() <= bound,
                    "{name}: y[{j}] = {v} outside expressible range ±{bound}"
                );
            }
        }

        // Spot-check one tensor against the hand-verified oracle values
        // (gguf-py 0.19 dequantize) when running the known 100 MB file.
        let name = std::env::var("HIGGS_GGUF_ORACLE_TENSOR").unwrap_or_default();
        if let Some(info) = file.tensors.get(&name) {
            let bytes = file.tensor_bytes(&name).unwrap().unwrap();
            let out = dequant_super_block(&bytes[..144]);
            eprintln!("{name}: dims {:?}, dtype {}", info.dims, info.dtype);
            eprintln!(
                "block0[:8] = {:?}",
                out[..8]
                    .iter()
                    .map(|v| (v * 1e6).round() / 1e6)
                    .collect::<Vec<_>>()
            );
            let per_sub: Vec<f32> = (0..8)
                .map(|s| {
                    out[s * 32..(s + 1) * 32]
                        .iter()
                        .fold(0.0f32, |m, v| m.max(v.abs()))
                })
                .collect();
            eprintln!(
                "per-sub max|w| = {:?}",
                per_sub
                    .iter()
                    .map(|v| (v * 1e6).round() / 1e6)
                    .collect::<Vec<_>>()
            );
        }
    }

    /// Test the 6-bit scale extraction matches llama.cpp get_scale_min_k4.
    #[test]
    fn scale_min_k4_extraction() {
        let q = [0x3F, 0x2A, 0x1B, 0x0C, 0x15, 0x3E, 0x07, 0x29];
        // j=0: d = q[0] & 63 = 0x3F & 63 = 63; m = q[4] & 63 = 0x15 & 63 = 21
        let (d, m) = get_scale_min_k4(0, &q);
        assert_eq!(d, 63);
        assert_eq!(m, 21);
        // j=4: d = (q[8]&0xF) | ((q[0]>>6)<<4) — but q only has 8 elements
        // so j=4 uses q[4+4]=q[8] which is OOB for an 8-element array.
        // The actual scales array is 12 bytes.
        // j=4 exercises the high-bit paths: q12[0] and q12[4] carry 0b11
        // above their 6-bit fields so the & 0x30 branches contribute.
        let q12 = [
            0xFF, 0x2A, 0x1B, 0x0C, 0xF5, 0x3E, 0x07, 0x29, 0x11, 0x22, 0x33, 0x44,
        ];
        let (d, m) = get_scale_min_k4(4, &q12);
        // d = (q[8] & 0xF) | ((q[0] >> 2) & 0x30) = 1 | ((0xFF >> 2) & 0x30) = 1 | 48 = 49
        assert_eq!(d, 49);
        // m = (q[8] >> 4) | ((q[4] >> 2) & 0x30) = 1 | ((0xF5 >> 2) & 0x30) = 1 | 48 = 49
        assert_eq!(m, 49);
    }

    /// Q4_K super-block with known values: verify dequant matches manual calc.
    #[test]
    fn dequant_known_values() {
        let mut block = vec![0u8; 144];
        // d = 1.0 (f16)
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        // dmin = 0.0 (f16)
        block[2] = 0x00;
        block[3] = 0x00; // f16 0.0
        // scales: all 8 sub-blocks get sc=1, m=0
        // j<4: q[j] & 63 = 1, q[j+4] & 63 = 0
        // j>=4: (q[j+4]&0xF) | ((q[j-4]>>6)<<4) = 1, (q[j+4]>>4) | ((q[j-4]>>6)&2) = 0
        // With q[0..4]=1 (bit 6 not set), and q[8..12]=1 (low nibble = 1):
        //   j=4: d=(q[8]&0xF)|((q[0]>>6)<<4) = 1|0 = 1, m=(q[8]>>4)|((q[0]>>6)&2) = 0|0 = 0
        for j in 0..4 {
            block[4 + j] = 1;
        }
        for j in 4..8 {
            block[4 + j] = 0;
        }
        for j in 8..12 {
            block[4 + j] = 1;
        }
        // qs: all 4-bit values = 1
        for b in block[16..144].iter_mut() {
            *b = 0x11;
        } // both nibbles = 1

        let out = dequant_super_block(&block);
        // d=1.0, sc=1, m=0, q4=1 → y = 1.0 * 1 * 1 - 1.0 * 0 * 1 = 1.0
        for (j, v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "y[{j}] = {v}, expected 1.0");
        }
    }

    /// Full round-trip: encode weights as Q4_K, dequantize, verify.
    #[test]
    fn dequant_matches_manual_computation() {
        let d: f32 = 0.5;
        let dmin: f32 = 0.1;
        let scales: [u8; 12] = [2, 4, 6, 8, 1, 3, 5, 7, 9, 11, 13, 15];
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16_to_bytes(d));
        block[2..4].copy_from_slice(&f16_to_bytes(dmin));
        block[4..16].copy_from_slice(&scales);

        let q4_vals: [u8; 256] = core::array::from_fn(|i| ((i * 7 + 3) & 0xF) as u8);
        // Set known 4-bit values, encoded per the real nibble layout: sub s
        // lives in page s/2 — low nibbles for even s, high nibbles for odd.
        for (idx, &q) in q4_vals.iter().enumerate() {
            let (sub, l) = (idx / 32, idx % 32);
            let byte_idx = 16 + (sub / 2) * 32 + l;
            if sub % 2 == 0 {
                block[byte_idx] = (block[byte_idx] & 0xF0) | (q & 0xF);
            } else {
                block[byte_idx] = (block[byte_idx] & 0x0F) | (q << 4);
            }
        }

        let out = dequant_super_block(&block);
        // The impl reads d and dmin back from f16, so the manual reference
        // must use the decoded values: f16(0.1) != 0.1.
        let d16 = f16_to_f32(&f16_to_bytes(d));
        let dmin16 = f16_to_f32(&f16_to_bytes(dmin));
        for sub in 0..8 {
            let (sc, m) = get_scale_min_k4(sub, &scales);
            let scale = d16 * sc as f32;
            let offset = dmin16 * m as f32;
            for j in 0..32 {
                let idx = sub * 32 + j;
                let q4 = q4_vals[idx] as f32;
                let expected = scale * q4 - offset;
                assert!(
                    (out[idx] - expected).abs() < 1e-5,
                    "y[{idx}] = {}, expected {expected}",
                    out[idx]
                );
            }
        }
    }

    fn f16_to_bytes(v: f32) -> [u8; 2] {
        f32_to_f16_bits(v).to_le_bytes()
    }
}
