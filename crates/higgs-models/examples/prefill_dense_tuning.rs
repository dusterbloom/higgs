//! Exact FP32/D256 dense attention query-tile screen. No serving defaults change.
#![allow(clippy::unwrap_used, clippy::print_stdout)]
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, fast, ops, transforms::eval};
use serde_json::json;
use std::time::Instant;

fn dense(q: &Array, k: &Array, v: &Array, mask: &Array, tile: i32) -> Array {
    let mut blocks = Vec::new();
    for start in (0..q.shape()[2]).step_by(tile as usize) {
        let end = (start + tile).min(q.shape()[2]);
        let block = fast::scaled_dot_product_attention(
            q.index((.., .., start..end, ..)),
            k,
            v,
            0.0625,
            Some(fast::ScaledDotProductAttentionMask::Array(
                &mask.index((start..end, ..)),
            )),
            None::<&Array>,
        )
        .unwrap();
        eval([&block]).unwrap();
        blocks.push(block);
    }
    let out = ops::concatenate_axis(&blocks.iter().collect::<Vec<_>>(), 2).unwrap();
    eval([&out]).unwrap();
    out
}

fn mask(rows: i32, keys: i32) -> Array {
    let values: Vec<bool> = (0..rows)
        .flat_map(|q| (0..keys).map(move |k| k <= keys - rows + q))
        .collect();
    Array::from_slice(&values, &[rows, keys])
}

fn check_offset_gqa_tail() {
    let q = Array::zeros::<f32>(&[1, 4, 3, 256]).unwrap();
    let k = Array::zeros::<f32>(&[1, 2, 5, 256]).unwrap();
    let values: Vec<f32> = (0..2)
        .flat_map(|h| (0..5).flat_map(move |k| std::iter::repeat_n((100 * h + k) as f32, 256)))
        .collect();
    let v = Array::from_slice(&values, &[1, 2, 5, 256]);
    let m = mask(3, 5);
    eval([&q, &k, &v, &m]).unwrap();
    let out = dense(&q, &k, &v, &m, 2);
    let data = out.as_slice::<f32>();
    for h in 0..4 {
        for row in 0..3 {
            let expected = (100 * (h / 2)) as f32 + (2 + row) as f32 / 2.0;
            for d in 0..256 {
                assert!((data[(h * 3 + row) * 256 + d] - expected).abs() < 2e-5);
            }
        }
    }
    println!(
        "{}",
        json!({"check":"offset_gqa_partial_tile", "pass":true})
    );
}

fn random_array(shape: &[i32], state: &mut u64) -> Array {
    let count: usize = shape.iter().map(|x| *x as usize).product();
    let data: Vec<f32> = (0..count)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            ((*state >> 40) as f32 / 16777216.0 - 0.5) * 2.0
        })
        .collect();
    Array::from_slice(&data, shape)
}

fn main() {
    let key_len: i32 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "16384".into())
        .parse()
        .unwrap();
    let repeats: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "8".into())
        .parse()
        .unwrap();
    assert!(key_len >= 1024 && repeats >= 2 && repeats % 2 == 0);
    check_offset_gqa_tail();
    let mut rng = 41_u64;
    let q = random_array(&[1, 16, 1024, 256], &mut rng);
    let k = random_array(&[1, 2, key_len, 256], &mut rng);
    let v = random_array(&[1, 2, key_len, 256], &mut rng);
    let m = mask(1024, key_len);
    eval([&q, &k, &v, &m]).unwrap();
    let control = dense(&q, &k, &v, &m, 128);
    let reference = control.as_slice::<f32>();
    for tile in [64, 256, 512, 1024] {
        let out = dense(&q, &k, &v, &m, tile);
        let data = out.as_slice::<f32>();
        let mut max_abs = 0.0_f32;
        let mut max_row_rel = 0.0_f64;
        for (got, want) in data.chunks_exact(256).zip(reference.chunks_exact(256)) {
            let mut error = 0.0_f64;
            let mut norm = 0.0_f64;
            for (&g, &w) in got.iter().zip(want) {
                assert!(g.is_finite() && w.is_finite());
                max_abs = max_abs.max((g - w).abs());
                error += f64::from(g - w).powi(2);
                norm += f64::from(w).powi(2);
            }
            max_row_rel = max_row_rel.max((error / norm.max(1e-24)).sqrt());
        }
        assert!(max_abs < 2e-4 && max_row_rel < 2e-4);
        println!(
            "{}",
            json!({"check":"tile_parity", "tile":tile, "max_abs":max_abs, "max_row_rel_l2":max_row_rel})
        );
    }
    // Alternate each candidate with its own control, rotate candidate order.
    let candidates = [64, 256, 512, 1024];
    for round in 0..repeats {
        for j in 0..4 {
            let candidate = candidates[(round + j) % 4];
            let order = if round % 2 == 0 {
                [128, candidate]
            } else {
                [candidate, 128]
            };
            for tile in order {
                let start = Instant::now();
                let out = dense(&q, &k, &v, &m, tile);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                std::hint::black_box(out);
                println!(
                    "{}",
                    json!({"round":round,"candidate":candidate,"actual_tile":tile,"key_len":key_len,"query_len":1024,"dtype":"float32","ms":ms})
                );
            }
        }
    }
}
