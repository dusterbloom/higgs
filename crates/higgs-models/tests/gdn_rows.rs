//! The fast-rows GDN kernels (`gdn_rows`) agree with the MLX chains they
//! replace within bf16 rounding: the Clover preview mixer at block 8 and 32
//! with and without committed history, and the verify prep whose tape, folded
//! by `n` rows, leaves the state and window of a plain forward of those rows.
#![allow(clippy::unwrap_used)]

use higgs_models::gdn_rows::{PreviewInputs, probe};
use mlx_rs::{Array, Dtype, fast, nn, ops, ops::indexing::IndexOp, random};

const D: i32 = 128;
const HK: i32 = 2;
const HV: i32 = 4;
const C: i32 = (2 * HK + HV) * D;
const K: i32 = 4;

fn normal(shape: &[i32], scale: f32, seed: u64, dtype: Dtype) -> Array {
    let key = random::key(seed).unwrap();
    random::normal::<f32>(shape, None, Some(scale), Some(&key))
        .unwrap()
        .as_dtype(dtype)
        .unwrap()
}

fn uniform(lo: f32, hi: f32, shape: &[i32], seed: u64, dtype: Dtype) -> Array {
    let key = random::key(seed).unwrap();
    random::uniform::<f32, f32>(lo, hi, shape, Some(&key))
        .unwrap()
        .as_dtype(dtype)
        .unwrap()
}

fn max_abs(x: &Array) -> f32 {
    ops::abs(x).unwrap().max(None).unwrap().item()
}

/// `got` within `tol * max|want|` of `want` elementwise (both any float dtype).
fn assert_close(tag: &str, got: &Array, want: &Array, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "{tag}: shape");
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    let err = max_abs(&got.subtract(&want).unwrap());
    let mag = max_abs(&want);
    println!("{tag}: max err {err:.3e} vs magnitude {mag:.3e}");
    assert!(err <= tol * mag, "{tag}: max err {err} vs magnitude {mag}");
}

struct Layer {
    conv_w: Array,
    conv_r: Array,
    norm_q: Array,
    norm_k: Array,
    a_log: Array,
    dt_bias: Array,
    scale: Array,
    hist_bias: Array,
    hist_c: Array,
}

impl Layer {
    fn new(dtype: Dtype) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let d = D as f32;
        Self {
            conv_w: uniform(-0.5, 0.5, &[K, C], 1, dtype),
            conv_r: uniform(-0.5, 0.5, &[C, 1, K - 1], 2, dtype),
            norm_q: Array::from_slice(&vec![d.sqrt().recip().powi(2); D as usize], &[D])
                .as_dtype(dtype)
                .unwrap(),
            norm_k: Array::from_slice(&vec![d.sqrt().recip(); D as usize], &[D])
                .as_dtype(dtype)
                .unwrap(),
            a_log: normal(&[HV], 0.5, 3, Dtype::Float32),
            dt_bias: normal(&[HV], 0.5, 4, Dtype::Float32),
            scale: uniform(2.0, 6.0, &[HV], 5, Dtype::Float32),
            hist_bias: normal(&[HV], 1.0, 6, Dtype::Float32),
            hist_c: normal(&[HV], 1.0, 7, Dtype::Float32),
        }
    }
}

#[test]
fn preview_mixer_matches_mlx_chain() {
    let _gate = higgs_models::mlx_exec::acquire();
    let dtype = Dtype::Bfloat16;
    let layer = Layer::new(dtype);
    let state = normal(&[1, HV, D, D], 0.2, 8, Dtype::Float32);
    for (rows_n, with_history) in [(8, true), (32, true), (8, false), (3, true)] {
        let rows = normal(
            &[1, rows_n, C],
            1.0,
            10 + u64::from(rows_n.cast_unsigned()),
            dtype,
        );
        let a = normal(
            &[1, rows_n, HV],
            1.0,
            20 + u64::from(rows_n.cast_unsigned()),
            dtype,
        );
        let history = if with_history {
            normal(&[1, K - 1, C], 1.0, 30, dtype)
        } else {
            ops::zeros_dtype(&[1, K - 1, C], dtype).unwrap()
        };
        let x = PreviewInputs {
            rows: &rows,
            a: &a,
            history: &history,
            state: &state,
            conv_w: &layer.conv_w,
            conv_r: &layer.conv_r,
            norm_q: &layer.norm_q,
            norm_k: &layer.norm_k,
            a_log: &layer.a_log,
            dt_bias: &layer.dt_bias,
            scale: &layer.scale,
            hist_bias: &layer.hist_bias,
            hist_c: &layer.hist_c,
        };
        let want = probe::preview_mixer_mlx(&x).unwrap();
        let got = probe::preview_mixer(&x).unwrap();
        assert_eq!(got.dtype(), Dtype::Float32);
        // The MLX chain rounds the convolution, the norms and the output to
        // bf16; a wrong tap, row, head or softmax slot is an O(1) error.
        assert_close(
            &format!("preview rows={rows_n} history={with_history}"),
            &got,
            &want,
            0.03,
        );
    }
}

/// Production's convolution + `SiLU` + norms for the verify (`nn::Conv1d` on
/// `[history | rows]`, bf16), as `(q, k, v)` per-head.
fn prep_reference(rows: &Array, history: &Array, layer: &Layer) -> (Array, Array, Array) {
    let t = rows.dim(1);
    let input = ops::concatenate_axis(&[history, rows], 1).unwrap();
    let weight = layer.conv_w.transpose().unwrap().expand_dims(-1).unwrap(); // [C, K, 1]
    let conv = nn::silu(&ops::conv1d(&input, &weight, 1, 0, 1, C).unwrap()).unwrap();
    let parts = conv.split_axis(&[HK * D, 2 * HK * D], Some(-1)).unwrap();
    let q = fast::rms_norm(
        &parts[0].reshape(&[1, t, HK, D]).unwrap(),
        &layer.norm_q,
        1e-6,
    )
    .unwrap();
    let k = fast::rms_norm(
        &parts[1].reshape(&[1, t, HK, D]).unwrap(),
        &layer.norm_k,
        1e-6,
    )
    .unwrap();
    (q, k, parts[2].reshape(&[1, t, HV, D]).unwrap())
}

#[test]
fn verify_prep_and_fold_match_plain_forward_of_kept_rows() {
    let _gate = higgs_models::mlx_exec::acquire();
    let dtype = Dtype::Bfloat16;
    let layer = Layer::new(dtype);
    let rows_n = 8;
    let rows = normal(&[1, rows_n, C], 1.0, 40, dtype);
    let history = normal(&[1, K - 1, C], 1.0, 41, dtype);
    let a = normal(&[1, rows_n, HV], 1.0, 42, dtype);
    let b = normal(&[1, rows_n, HV], 1.0, 43, dtype);
    let state = normal(&[1, HV, D, D], 0.2, 44, Dtype::Float32);
    let g = probe::geometry(&rows, &history, &state, &layer.conv_w, 8).unwrap();

    let (q, k, v, window) = probe::prep_verify(
        &rows,
        &history,
        &layer.conv_w,
        &layer.norm_q,
        &layer.norm_k,
        g,
    )
    .unwrap();
    let (q_ref, k_ref, v_ref) = prep_reference(&rows, &history, &layer);
    assert_close("verify q", &q, &q_ref, 0.01);
    assert_close("verify k", &k, &k_ref, 0.01);
    assert_close("verify v", &v, &v_ref, 0.01);
    let concat = ops::concatenate_axis(&[&history, &rows], 1).unwrap();
    assert_close(
        "verify window",
        &window,
        &concat.index((.., rows_n..rows_n + K - 1, ..)),
        0.0,
    );

    let (_, state_all, delta) =
        probe::gated_delta_with_tape(&q, &k, &v, &layer.a_log, &a, &layer.dt_bias, &b, &state, g)
            .unwrap();
    for n in [1, 5, 8] {
        let (folded, folded_window) = probe::fold(
            &state,
            &k,
            &a,
            &layer.a_log,
            &layer.dt_bias,
            &delta,
            &history,
            &rows,
            n,
            g,
        )
        .unwrap();
        // The committed state after keeping n rows is the state a forward of
        // exactly those rows produces (bitwise: the fold replays the
        // recurrence's own statements on the recorded innovations).
        let want = if n == rows_n {
            state_all.clone()
        } else {
            let mut gn = g;
            gn.rows = n;
            probe::gated_delta_with_tape(
                &q.index((.., ..n, .., ..)),
                &k.index((.., ..n, .., ..)),
                &v.index((.., ..n, .., ..)),
                &layer.a_log,
                &a.index((.., ..n, ..)),
                &layer.dt_bias,
                &b.index((.., ..n, ..)),
                &state,
                gn,
            )
            .unwrap()
            .1
        };
        assert_close(&format!("fold n={n} state"), &folded, &want, 0.0);
        assert_close(
            &format!("fold n={n} window"),
            &folded_window,
            &concat.index((.., n..n + K - 1, ..)),
            0.0,
        );
    }
}

/// On the Clover slice: the fast-rows verify agrees with the MLX verify, and
/// rolling its tape back to `n` rows leaves every GDN layer's state, window
/// and offset equal to a fresh fast-rows verify of those `n` rows.
#[test]
#[ignore = "needs HIGGS_CLOVER_FIXTURE_DIR (examples/clover_reference_fixture.py)"]
fn fixture_verify_rollback_equals_plain_forward() {
    use higgs_models::cache::KeyValueCache;
    use higgs_models::qwen3_next::{
        DFlashRowSchedule, LayerCache, load_qwen3_5_moe_model, with_fast_rows,
    };
    let Ok(dir) = std::env::var("HIGGS_CLOVER_FIXTURE_DIR") else {
        return;
    };
    let _gate = higgs_models::mlx_exec::acquire();
    let dir = std::path::PathBuf::from(dir);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("fixture.json")).unwrap()).unwrap();
    let ids = |key: &str| -> Vec<u32> {
        meta[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
            .collect()
    };
    let (prompt, block) = (ids("prompt"), ids("first_block")[..8].to_vec());
    let mut model = load_qwen3_5_moe_model(dir.join("slice")).unwrap();
    let row = |t: &[u32]| Array::from_slice(t, &[1, i32::try_from(t.len()).unwrap()]);
    let prefilled = |model: &mut higgs_models::qwen3_next::Qwen3NextCausalLM| {
        let mut cache = model.make_cache();
        model
            .forward(&row(&prompt), None, &mut cache)
            .unwrap()
            .eval()
            .unwrap();
        cache
    };
    let verify = |model: &mut higgs_models::qwen3_next::Qwen3NextCausalLM,
                  cache: &mut Vec<Option<LayerCache>>,
                  tokens: &[u32],
                  fast: bool| {
        let run = |model: &mut higgs_models::qwen3_next::Qwen3NextCausalLM,
                   cache: &mut Vec<Option<LayerCache>>| {
            model.forward_with_taps_tape_scheduled(
                &row(tokens),
                None,
                cache,
                &[],
                None,
                DFlashRowSchedule::NativeBatch,
            )
        };
        let (logits, _, tapes) = if fast {
            with_fast_rows(|| run(model, cache)).unwrap()
        } else {
            run(model, cache).unwrap()
        };
        logits.eval().unwrap();
        (logits.reshape(&[8, -1]).unwrap(), tapes)
    };

    let mut cache = prefilled(&mut model);
    let (want, _) = verify(&mut model, &mut cache, &block, false);
    let mut cache = prefilled(&mut model);
    let (got, tapes) = verify(&mut model, &mut cache, &block, true);
    // `HIGGS_GDN_ROWS=0`: the same comparison on the MLX chains (A/B).
    assert!(
        tapes.iter().flatten().all(|t| t.fused)
            || std::env::var("HIGGS_GDN_ROWS").as_deref() == Ok("0"),
        "fast rows took the MLX tape path"
    );
    let (got, want) = (
        got.as_dtype(Dtype::Float32).unwrap(),
        want.as_dtype(Dtype::Float32).unwrap(),
    );
    for r in 0..8 {
        let (g, w) = (got.index(r), want.index(r));
        let g_arg: u32 = ops::indexing::argmax(&g, None).unwrap().item();
        let w_arg: u32 = ops::indexing::argmax(&w, None).unwrap().item();
        let cos: f32 = g.multiply(&w).unwrap().sum(None).unwrap().item::<f32>()
            / (max_abs(&g.square().unwrap().sum(None).unwrap()).sqrt()
                * max_abs(&w.square().unwrap().sum(None).unwrap()).sqrt());
        println!("verify row {r}: top1 {g_arg} vs {w_arg}, cosine {cos:.5}");
        assert!(cos > 0.995, "verify row {r}: cosine {cos}");
    }

    let kept = 3;
    model
        .replay_tape_rollback(&tapes, &mut cache, kept, 8 - kept)
        .unwrap();
    let mut plain = prefilled(&mut model);
    let _ = verify(
        &mut model,
        &mut plain,
        &block[..usize::try_from(kept).unwrap()],
        true,
    );
    for (index, (a, b)) in cache.iter().zip(&plain).enumerate() {
        match (a.as_ref().unwrap(), b.as_ref().unwrap()) {
            (LayerCache::Arrays(a), LayerCache::Arrays(b)) => {
                assert_eq!(
                    (a.offset, a.conv_pos),
                    (b.offset, b.conv_pos),
                    "layer {index}"
                );
                assert_close(
                    &format!("layer {index} ssm_state"),
                    a.ssm_state.as_ref().unwrap(),
                    b.ssm_state.as_ref().unwrap(),
                    0.0,
                );
                assert_close(
                    &format!("layer {index} conv_state"),
                    a.conv_state.as_ref().unwrap(),
                    b.conv_state.as_ref().unwrap(),
                    0.0,
                );
            }
            (LayerCache::KV(a), LayerCache::KV(b)) => {
                assert_eq!(a.offset(), b.offset(), "layer {index} KV offset");
            }
            _ => panic!("layer {index}: cache variants differ"),
        }
    }
}
