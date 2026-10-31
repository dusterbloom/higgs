//! The fused fast-rows `DFlash2` draft (`dflash_rows`) agrees with the MLX
//! chain it replaces within bf16 rounding on a tiny random drafter: the
//! trunk's hidden output and the committed K/V context (window trim included,
//! with context rows the sliding mask hides from the last block rows), and the
//! selector picks the tokens of its MLX chain; its pieces (ordered top-16 over
//! chunks, the lattice walk) match exact references.
#![allow(clippy::unwrap_used)]

use higgs_models::dflash::probe::{self as drafter, BLOCK, HIDDEN, RANK, TAPS, VOCAB, WINDOW};
use higgs_models::dflash_rows::probe;
use mlx_rs::{Array, Dtype, ops, ops::indexing::IndexOp, random};

fn normal(shape: &[i32], scale: f32, seed: u64, dtype: Dtype) -> Array {
    let key = random::key(seed).unwrap();
    random::normal::<f32>(shape, None, Some(scale), Some(&key))
        .unwrap()
        .as_dtype(dtype)
        .unwrap()
}

fn max_abs(x: &Array) -> f32 {
    ops::abs(x).unwrap().max(None).unwrap().item()
}

/// `got` within `tol * max|want|` of `want` elementwise.
fn assert_close(tag: &str, got: &Array, want: &Array, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "{tag}: shape");
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    let err = max_abs(&got.subtract(&want).unwrap());
    let mag = max_abs(&want);
    println!("{tag}: max err {err:.3e} vs magnitude {mag:.3e}");
    assert!(err <= tol * mag, "{tag}: max err {err} vs magnitude {mag}");
}

fn taps(rows: i32, seed: u64) -> Vec<Array> {
    (0..TAPS)
        .map(|i| {
            normal(
                &[1, rows, HIDDEN],
                1.0,
                seed + u64::try_from(i).unwrap(),
                Dtype::Bfloat16,
            )
        })
        .collect()
}

/// One draft after a 32-row prime (the MLX chain projects whole tiles; the
/// window keeps 15 rows) with 3 new context rows: the fused trunk, the
/// trimmed and appended K/V, and the mask over the oldest keys all match.
#[test]
fn fused_draft_matches_the_mlx_chain() {
    let _gate = higgs_models::mlx_exec::acquire();
    let mut d = drafter::tiny_dflash2(3).unwrap();
    let mut cache = d.make_cache();
    d.prime_taps(&taps(32, 100), &mut cache).unwrap();
    assert_eq!(cache.position(), 32);
    let noise = normal(&[1, BLOCK, HIDDEN], 1.0, 200, Dtype::Bfloat16);
    let new_taps = taps(3, 300);

    let want = d.stage_forward(&noise, &new_taps, &cache).unwrap();
    let got = probe_or_fail(drafter::fused_stage_forward(&d, &noise, &new_taps, &cache));
    assert_eq!(got.position().unwrap(), want.position().unwrap());
    assert_close("hidden", got.hidden(), want.hidden(), 0.05);
    assert_eq!(got.layers().len(), want.layers().len());
    for (layer, (g, w)) in got.layers().iter().zip(want.layers()).enumerate() {
        let (gk, gv) = g.as_ref().unwrap();
        let (wk, wv) = w.as_ref().unwrap();
        assert_eq!(gk.shape(), &[1, drafter::KV_HEADS, WINDOW - 1, 128]);
        assert_close(&format!("layer {layer} K"), gk, wk, 0.02);
        assert_close(&format!("layer {layer} V"), gv, wv, 0.02);
    }
    // With 15 context rows and a window of 16, `15 + r - j >= 16` hides keys
    // `j <= r - 1` from block row `r`: the sliding mask is exercised.
}

fn probe_or_fail<T>(result: Result<Option<T>, mlx_rs::error::Exception>) -> T {
    result
        .unwrap()
        .expect("the fused path must take this geometry (gate declined)")
}

/// The fused selector returns the MLX selector's tokens for random logits
/// and hiddens whose lattice edges are of the logits' order.
#[test]
fn fused_selector_matches_the_mlx_chain() {
    let _gate = higgs_models::mlx_exec::acquire();
    let d = drafter::tiny_dflash2(5).unwrap();
    let k = 5;
    let hidden = normal(&[1, k, HIDDEN], 1.0, 400, Dtype::Bfloat16);
    let logits = normal(&[1, k, VOCAB], 3.0, 401, Dtype::Bfloat16);
    let want = d
        .propose_dflash2_tokens(&hidden, &logits, 17)
        .unwrap()
        .unwrap();
    let got = probe_or_fail(drafter::fused_select(&d, &hidden, &logits, 17));
    assert_eq!(got.shape(), &[1, k]);
    let want: Vec<u32> = want.as_dtype(Dtype::Uint32).unwrap().as_slice().to_vec();
    let got: Vec<u32> = got.as_slice().to_vec();
    assert_eq!(got, want);
}

/// Ordered top-16 over a row longer than one chunk (a partial last chunk,
/// then the merge): the argpartition's set, in descending order.
#[test]
fn top16_matches_argpartition() {
    let _gate = higgs_models::mlx_exec::acquire();
    let n = 5000;
    let src = normal(&[3, n], 1.0, 500, Dtype::Bfloat16);
    let (vals, ids) = probe::top16(&src).unwrap().unwrap();
    assert_eq!(vals.shape(), &[3, 16]);
    let part = ops::argpartition_axis(&src.negative().unwrap(), 15, -1).unwrap();
    for r in 0..3 {
        let mut want: Vec<u32> = part.index((r, 0..16)).as_slice().to_vec();
        want.sort_unstable();
        let got_ids: Vec<u32> = ids.index(r).as_slice().to_vec();
        let mut sorted = got_ids.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, want, "row {r}: id set");
        let got_vals: Vec<f32> = vals.index(r).as_slice().to_vec();
        assert!(
            got_vals.windows(2).all(|w| w[0] >= w[1]),
            "row {r}: order {got_vals:?}"
        );
        let src_f32: Vec<f32> = src
            .index(r)
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice()
            .to_vec();
        for (v, &i) in got_vals.iter().zip(&got_ids) {
            assert_eq!(
                *v,
                src_f32[usize::try_from(i).unwrap()],
                "row {r}: value of id {i}"
            );
        }
    }
}

/// The lattice walk against a host fp32 reference of the same math on the
/// same shortlist: edges from the dequantized codebooks, first argmax,
/// chained predecessor.
#[test]
fn lattice_walk_matches_a_host_reference() {
    let _gate = higgs_models::mlx_exec::acquire();
    let d = drafter::tiny_dflash2(7).unwrap();
    let [pred, succ] = drafter::codebooks(&d).unwrap();
    let k = 6;
    let anchor = 123;
    let key = random::key(600).unwrap();
    let ids = random::randint::<_, u32>(0, VOCAB, &[k, 16], Some(&key)).unwrap();
    let vals = normal(&[k, 16], 1.0, 601, Dtype::Float32);
    let sel = normal(&[k, RANK], 1.0, 602, Dtype::Bfloat16);
    let anchor_arr = Array::from_slice(&[anchor], &[1]);
    let got: Vec<u32> = probe::lattice_walk(&ids, &vals, &sel, pred, succ, &anchor_arr)
        .unwrap()
        .as_slice()
        .to_vec();

    let dense = |(w, s, b): (&Array, &Array, &Array)| -> Vec<f32> {
        ops::dequantize(w, s, b, 64, 8)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice()
            .to_vec()
    };
    let (a, b) = (dense(pred), dense(succ));
    let ids_h: Vec<u32> = ids.as_slice().to_vec();
    let vals_h: Vec<f32> = vals.as_slice().to_vec();
    let sel_h: Vec<f32> = sel.as_dtype(Dtype::Float32).unwrap().as_slice().to_vec();
    let rank = usize::try_from(RANK).unwrap();
    let mut predecessor = u32::try_from(anchor).unwrap();
    let mut want = Vec::new();
    for p in 0..usize::try_from(k).unwrap() {
        let mut best = (f32::NEG_INFINITY, 0);
        for c in 0..16 {
            let cand = ids_h[p * 16 + c];
            let edge: f32 = (0..rank)
                .map(|r| {
                    a[usize::try_from(predecessor).unwrap() * rank + r]
                        * sel_h[p * rank + r]
                        * b[usize::try_from(cand).unwrap() * rank + r]
                })
                .sum();
            let score = vals_h[p * 16 + c] + edge;
            if score > best.0 {
                best = (score, c);
            }
        }
        predecessor = ids_h[p * 16 + best.1];
        want.push(predecessor);
    }
    assert_eq!(got, want);
}
