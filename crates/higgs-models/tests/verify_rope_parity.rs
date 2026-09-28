#![allow(clippy::tests_outside_test_module, clippy::unwrap_used)]
//! Speculative verify rows must be rotated exactly as decode rotates each
//! token, bit for bit, for every row, dtype and absolute position, including
//! rows that straddle the 1,024-row blocks of the prefill rotation.

use higgs_models::verify_rope_probe::{decode, verify};
use mlx_rs::{Array, Dtype, ops, ops::indexing::IndexOp};

fn bits(array: &Array) -> Vec<u32> {
    let values = array.as_dtype(Dtype::Float32).unwrap();
    values.eval().unwrap();
    values
        .as_slice::<f32>()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn verify_rows_take_decode_rope_bit_for_bit() {
    let _exec = higgs_models::mlx_exec::acquire();
    for rows in [2, 8, 16] {
        // Query heads and key/value heads of the full-attention layers.
        for heads in [24, 4] {
            for dtype in [Dtype::Float32, Dtype::Bfloat16] {
                let x = mlx_rs::random::normal::<f32>(&[1, heads, rows, 256], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                for offset in [0, 300, 1016, 40_000] {
                    let rows_decoded: Vec<Array> = (0..rows)
                        .map(|row| decode(&x.index((.., .., row..=row, ..)), offset + row).unwrap())
                        .collect();
                    let decoded =
                        ops::concatenate_axis(&rows_decoded.iter().collect::<Vec<_>>(), 2).unwrap();
                    assert!(
                        bits(&verify(&x, offset).unwrap()) == bits(&decoded),
                        "rows={rows} heads={heads} {dtype:?} offset={offset}: verify RoPE differs from decode"
                    );
                }
            }
        }
    }
}
