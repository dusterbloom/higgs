use half::f16;
use higgs_models::bonsai_q2::{GROUP_SIZE, PackedQ2Linear};
use higgs_models::q2_row2_bench::BonsaiQ2Row2Bench;
use higgs_models::quant_mode::QuantMode;
use higgs_models::quant_mode::quantized_matmul;
use mlx_rs::{self, Array};

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

fn make_packed_q2(out_features: usize, in_features: usize, seed: u64) -> PackedQ2Linear {
    let packed_cols = in_features / higgs_models::bonsai_q2::WEIGHTS_PER_WORD;
    let n_groups = in_features / GROUP_SIZE;
    let mut st = seed;
    let w_packed: Vec<u32> = (0..out_features * packed_cols)
        .map(|_| lcg(&mut st))
        .collect();
    let scales: Vec<f16> = (0..out_features * n_groups)
        .map(|i| f16::from_f32(0.05 + 0.013 * ((i % 7) as f32)))
        .collect();
    let biases: Vec<f16> = (0..out_features * n_groups)
        .map(|i| f16::from_f32(-0.03 + 0.011 * ((i % 5) as f32)))
        .collect();
    PackedQ2Linear {
        w_packed,
        scales,
        biases,
        out_features,
        in_features,
    }
}

fn upload_to_mlx(p: &PackedQ2Linear) -> (Array, Array, Array) {
    let packed_cols = p.packed_cols();
    let n_groups = p.n_groups();
    let w = Array::from_slice(&p.w_packed, &[p.out_features as i32, packed_cols as i32]);
    let s = Array::from_slice(&p.scales, &[p.out_features as i32, n_groups as i32]);
    let b = Array::from_slice(&p.biases, &[p.out_features as i32, n_groups as i32]);
    (w, s, b)
}

fn main() {
    let _exec = higgs_models::mlx_exec::acquire();

    let shapes: &[(usize, usize, &str)] = &[
        (17_408, 5_120, "gate_up (N=17408, K=5120)"),
        (5_120, 17_408, "down_proj (N=5120, K=17408)"),
    ];

    for &(out_f, in_f, label) in shapes {
        let p = make_packed_q2(out_f, in_f, 0xBEEF_BEEF);
        let (w_canon, s_canon, b_canon) = upload_to_mlx(&p);
        let packed = BonsaiQ2Row2Bench::from_row_major(&w_canon, &s_canon).expect("pack row2");

        let mut st = 0x1234_5678_u64;
        let x_f32: Vec<f32> = (0..(6 * in_f))
            .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
            .collect();
        let x = Array::from_slice(&x_f32, &[6, in_f as i32])
            .as_dtype(mlx_rs::Dtype::Float16)
            .expect("x dtype float16");

        for _ in 0..5 {
            let _ = quantized_matmul(
                &x,
                &w_canon,
                &s_canon,
                Some(&b_canon),
                true,
                128,
                2,
                QuantMode::Affine,
            )
            .expect("stock baseline warmup")
            .eval()
            .expect("stock warmup eval");
            let _ = packed
                .m5_contract(&x, &b_canon)
                .expect("row2 warmup")
                .eval()
                .expect("row2 warmup eval");
        }

        let n_iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            let y = quantized_matmul(
                &x,
                &w_canon,
                &s_canon,
                Some(&b_canon),
                true,
                128,
                2,
                QuantMode::Affine,
            )
            .expect("stock run");
            y.eval().expect("stock eval");
        }
        let stock_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..n_iters {
            let y = packed.m5_contract(&x, &b_canon).expect("row2 run");
            y.eval().expect("row2 eval");
        }
        let row2_us = t1.elapsed().as_micros() as f64 / n_iters as f64;

        let ratio = stock_us / row2_us;
        println!("MICROBENCH {label}: stock={stock_us:.1}us row2={row2_us:.1}us ratio={ratio:.3}x");
    }
}
