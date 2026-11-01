//! Append-stability contract of the generic switch-MLP gather: routed rows fed
//! as a short tail must produce bit-identical outputs to the same rows inside a
//! longer sequence (retained-session continuation == cold prefill). The long
//! domain crosses MLX's `gather_qmm_rhs` threshold (rows >= 4 * experts), so
//! the steel-tile and `gather_qmm_rhs` kernels must agree bit for bit.
use higgs_models::qwen3_next::switch_mlp_gather_probe::{SwitchMlpGather, gather_qmm};
use mlx_rs::{Array, Dtype, ops, ops::indexing::IndexOp};

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .subtract(&b.as_dtype(Dtype::Float32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item()
}

fn quantized(shape: &[i32], seed: u64) -> (Array, Array, Array) {
    let key = mlx_rs::random::key(seed).unwrap();
    let w = mlx_rs::random::uniform::<f32, f32>(-0.05, 0.05, shape, Some(&key)).unwrap();
    ops::quantize(&w.as_dtype(Dtype::Bfloat16).unwrap(), 64, 4).unwrap()
}

/// The kernel mode is read once per process, so each mode runs
/// [`append_stability_contract`] in its own child process.
#[test]
fn short_tails_match_same_rows_inside_long_sequence() {
    for (var, value) in [
        ("HIGGS_MOE_GATHER_KERNEL", "auto"),
        ("HIGGS_MOE_GATHER_KERNEL", "qmv"),
        ("HIGGS_MOE_GATHER_KERNEL", "mma"),
        ("HIGGS_MOE_GATHER_PAD", "1"),
    ] {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "append_stability_contract"])
            .env_remove("HIGGS_MOE_GATHER_KERNEL")
            .env_remove("HIGGS_MOE_GATHER_PAD")
            .env(var, value)
            .status()
            .unwrap();
        assert!(status.success(), "{var}={value}");
    }
}

#[test]
#[ignore = "run once per kernel mode by short_tails_match_same_rows_inside_long_sequence"]
fn append_stability_contract() {
    // 64 experts. Under `mma` the 1- and 3-token tails (2 and 6 rows) take
    // one steel tile per row, the 9- to 100-token tails `gather_qmm_rhs`
    // padded to 4 * E = 256 rows, and the 160-token domain (320 rows)
    // unpadded `gather_qmm_rhs`. `auto` (the default) runs MMA from 18 tokens
    // up and qmv below, so only its tails from 18 tokens must match the MMA
    // rows of the long domain, like MLX's own dense qmv -> qmm switch.
    let auto = std::env::var("HIGGS_MOE_GATHER_KERNEL").is_ok_and(|v| v == "auto");
    let (num_experts, hidden, top_k, long_len) = (64_u32, 64, 2, 160);
    let e = num_experts as i32;
    let mut block = SwitchMlpGather::new(
        64,
        4,
        [
            quantized(&[e, hidden, hidden], 1),
            quantized(&[e, hidden, hidden], 2),
            quantized(&[e, hidden, hidden], 3),
        ],
    )
    .unwrap();

    let values: Vec<f32> = (0..long_len * hidden)
        .map(|i| ((i % 97) as f32 - 48.0) / 31.0)
        .collect();
    let long = Array::from_slice(&values, &[1, long_len, hidden])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let indices: Vec<u32> = (0..long_len as u32)
        .flat_map(|t| [0, 1 + (t * 7) % (num_experts - 1)])
        .collect();
    let long_indices = Array::from_slice(&indices, &[1, long_len, top_k]);
    let long_out = block.forward(&long, &long_indices).unwrap();
    let long_fused = block.forward_fused(&long, &long_indices).unwrap();

    for tail_len in [1, 3, 9, 17, 18, 20, 33, 57, 100] {
        if auto && tail_len < 18 {
            continue;
        }
        let start = long_len - tail_len;
        let tail_out = block
            .forward(
                &long.index((.., start.., ..)),
                &long_indices.index((.., start.., ..)),
            )
            .unwrap();
        let diff = max_abs_diff(&tail_out, &long_out.index((.., start.., .., ..)));
        assert_eq!(diff, 0.0, "{tail_len}-token tail must be append-stable");
        let tail_fused = block
            .forward_fused(
                &long.index((.., start.., ..)),
                &long_indices.index((.., start.., ..)),
            )
            .unwrap();
        let diff = max_abs_diff(&tail_fused, &long_fused.index((.., start.., .., ..)));
        assert_eq!(
            diff, 0.0,
            "{tail_len}-token fused tail must be append-stable"
        );
    }
}

/// MLX 0.31 premise of the MMA policy at Clover's expert shapes (E = 256,
/// 2048 -> 512 and 512 -> 2048): `gather_qmm_rhs` over a sorted domain equals
/// steel `gather_qmm` over per-expert tiles, for any tile height the steel
/// kernel accepts, and a row alone in a tile equals the same row in the batch.
#[test]
fn rhs_and_steel_tiles_agree_bitwise_at_clover_shapes() {
    let (experts, rows) = (256_u32, 2048_usize);
    let mut state = 11_u64;
    let mut routed: Vec<u32> = (0..rows)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as u32) % experts
        })
        .collect();
    routed.sort_unstable();
    let rhs_indices = Array::from_slice(&routed, &[rows as i32]);

    for (k, n) in [(2048, 512), (512, 2048)] {
        let (w, scales, biases) = quantized(&[experts as i32, n, k], 7);
        let weights = (&w, &scales, &biases);
        let key = mlx_rs::random::key(3).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[rows as i32, 1, k], None, None, Some(&key))
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let rhs = gather_qmm(&x, weights, &rhs_indices, 64, 4, true)
            .unwrap()
            .reshape(&[rows as i32, n])
            .unwrap();

        // 18 is the M4 qmv batch limit for these shapes; 32 is MLX's BM.
        for tile in [18_usize, 32] {
            let (mut source, mut tile_experts, mut dest) = (Vec::new(), Vec::new(), vec![0; rows]);
            let mut row = 0;
            while row < rows {
                let end = (row + tile).min(rows);
                let end = row
                    + routed[row..end]
                        .iter()
                        .take_while(|&&e| e == routed[row])
                        .count();
                tile_experts.push(routed[row]);
                for (slot, r) in (row..end).enumerate() {
                    dest[r] = (source.len() + slot) as u32;
                }
                let base = source.len();
                source.extend((row..end).map(|r| r as u32));
                source.resize(base + tile, rows as u32); // zero row
                row = end;
            }
            let x_ext = ops::concatenate_axis(
                &[
                    &x.reshape(&[rows as i32, k]).unwrap(),
                    &ops::zeros_dtype(&[1, k], x.dtype()).unwrap(),
                ],
                0,
            )
            .unwrap();
            let tiles = tile_experts.len() as i32;
            let tiled_x = x_ext
                .take_axis(&Array::from_slice(&source, &[source.len() as i32]), 0)
                .unwrap()
                .reshape(&[tiles, tile as i32, k])
                .unwrap();
            let tiled = gather_qmm(
                &tiled_x,
                weights,
                &Array::from_slice(&tile_experts, &[tiles]),
                64,
                4,
                false,
            )
            .unwrap()
            .reshape(&[tiles * tile as i32, n])
            .unwrap()
            .take_axis(&Array::from_slice(&dest, &[rows as i32]), 0)
            .unwrap();
            assert_eq!(max_abs_diff(&rhs, &tiled), 0.0, "K={k} N={n} tile={tile}");
        }

        // One row alone at the bottom of an otherwise-zero tile.
        let row = 777;
        let alone = ops::concatenate_axis(
            &[
                &ops::zeros_dtype(&[31, k], x.dtype()).unwrap(),
                &x.index(row as i32).reshape(&[1, k]).unwrap(),
            ],
            0,
        )
        .unwrap()
        .reshape(&[1, 32, k])
        .unwrap();
        let alone_out = gather_qmm(
            &alone,
            weights,
            &Array::from_slice(&routed[row..=row], &[1]),
            64,
            4,
            false,
        )
        .unwrap()
        .reshape(&[32, n])
        .unwrap()
        .index(31);
        assert_eq!(
            max_abs_diff(&alone_out, &rhs.index(row as i32)),
            0.0,
            "K={k} N={n}: lone row"
        );
    }
}
