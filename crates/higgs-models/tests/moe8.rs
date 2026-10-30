//! `moe8` (weight-once 8-row `MoE` expert passes) agrees with the generic
//! switch-MLP gather within bf16 rounding, for partial and full row tiles and
//! uneven expert sharing (one expert used by every row, others by 1..3 rows).
#![allow(clippy::unwrap_used)]

use higgs_models::moe8::probe;
use higgs_models::qwen3_next::switch_mlp_gather_probe::SwitchMlpGather;
use mlx_rs::{Array, Dtype, ops, random};

fn quantized(shape: &[i32], seed: u64) -> (Array, Array, Array) {
    let key = random::key(seed).unwrap();
    let w = random::uniform::<f32, f32>(-0.05, 0.05, shape, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    ops::quantize(&w, 64, 4).unwrap()
}

const fn parts((w, s, b): &(Array, Array, Array)) -> (&Array, &Array, &Array) {
    (w, s, b)
}

#[test]
fn moe8_matches_switch_mlp_gather() {
    // 20 experts (not a multiple of the 32-thread plan simdgroup), top-4.
    let (experts, hidden, inter, top_k) = (20, 256, 256, 4);
    let gate = quantized(&[experts, inter, hidden], 1);
    let up = quantized(&[experts, inter, hidden], 2);
    let down = quantized(&[experts, hidden, inter], 3);
    let block = SwitchMlpGather::new(64, 4, [gate.clone(), up.clone(), down.clone()]).unwrap();
    for rows in [1_i32, 3, 8] {
        let key = random::key(10 + u64::from(rows.cast_unsigned())).unwrap();
        let x = random::normal::<f32>(&[1, rows, hidden], None, None, Some(&key))
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        // Row t: expert 0 (every row), 1 + t % 3 (3 rows each), 5 + 2t % 7
        // (1-2 rows), 13 + t % 3; ascending and distinct within the row.
        let routing: Vec<u32> = (0..rows.cast_unsigned())
            .flat_map(|t| [0, 1 + t % 3, 5 + (2 * t) % 7, 13 + t % 3])
            .collect();
        let indices = Array::from_slice(&routing, &[1, rows, top_k]);
        let want = block
            .forward(&x, &indices)
            .unwrap()
            .reshape(&[rows, top_k, hidden])
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();

        let x2 = x.reshape(&[rows, hidden]).unwrap();
        let idx = indices.reshape(&[rows, top_k]).unwrap();
        assert!(probe::supported(&x2, &idx, &gate.0, &up.0, &down.0));
        let out = probe::moe8(&x2, &idx, parts(&gate), parts(&up), parts(&down)).unwrap();
        assert_eq!(out.shape(), &[rows, top_k, hidden]);
        assert_eq!(out.dtype(), Dtype::Bfloat16);
        let got = out.as_dtype(Dtype::Float32).unwrap();
        let err: f32 = ops::abs(got.subtract(&want).unwrap())
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        let mag: f32 = ops::abs(&want).unwrap().max(None).unwrap().item();
        // Both paths round h to 16 bits (bf16 there, half here) before the
        // down projection; a wrong expert, slot or row is an O(1) error.
        assert!(
            err <= 0.02 * mag,
            "rows={rows}: max err {err} vs magnitude {mag}"
        );
    }
}
