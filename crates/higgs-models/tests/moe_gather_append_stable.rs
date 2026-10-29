//! Append-stability contract of the generic switch-MLP gather: routed rows fed
//! as a short tail must produce bit-identical outputs to the same rows inside a
//! longer sequence (retained-session continuation == cold prefill). The long
//! domain crosses MLX's `gather_qmm_rhs` threshold (rows >= 4 * experts).
use higgs_models::qwen3_next::switch_mlp_gather_probe::SwitchMlpGather;
use mlx_rs::{Array, Dtype, ops, ops::indexing::IndexOp};

#[test]
fn short_tail_matches_same_rows_inside_long_sequence() {
    let (num_experts, hidden, top_k, prefix_len, tail_len) = (8, 64, 2, 16, 4);
    let projection = || {
        let w =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
                .unwrap();
        ops::quantize(&w, 64, 4).unwrap()
    };
    let block = SwitchMlpGather::new(64, 4, [projection(), projection(), projection()]).unwrap();

    let rows = prefix_len + tail_len;
    let values: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i % 97) as f32 - 48.0) / 31.0)
        .collect();
    let long = Array::from_slice(&values, &[1, rows, hidden])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let indices: Vec<u32> = (0..rows * top_k)
        .map(|i| (i % num_experts) as u32)
        .collect();
    let long_indices = Array::from_slice(&indices, &[1, rows, top_k]);

    let short_out = block
        .forward(
            &long.index((.., prefix_len.., ..)),
            &long_indices.index((.., prefix_len.., ..)),
        )
        .unwrap();
    let long_out = block
        .forward(&long, &long_indices)
        .unwrap()
        .index((.., prefix_len.., .., ..));

    let max_diff: f32 = short_out
        .subtract(&long_out)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item();
    assert_eq!(
        max_diff, 0.0,
        "identical logical rows must be append-stable"
    );
}
