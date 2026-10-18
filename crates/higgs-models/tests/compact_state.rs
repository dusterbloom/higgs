//! A cache kept across requests may keep alive only what `estimated_bytes`
//! counts. Recurrent state is routinely a slice of a prompt-sized buffer, and
//! an MLX slice pins its whole parent; `compact_state` must release it.
#![allow(clippy::unwrap_used, unsafe_code)]

use higgs_models::cache::SteppingKeyValueCache;
use higgs_models::qwen3_next::ArraysCache;
use higgs_models::{AnyCache, LayerCache};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

fn active_bytes() -> i64 {
    let mut bytes = 0usize;
    assert_eq!(unsafe { mlx_sys::mlx_get_active_memory(&raw mut bytes) }, 0);
    i64::try_from(bytes).unwrap()
}

/// Real (non-broadcast) data so the array owns a full-size buffer.
fn filled(shape: &[i32]) -> Array {
    let n: i32 = shape.iter().product();
    let a = mlx_rs::ops::arange::<_, f32>(0, n, None)
        .unwrap()
        .reshape(shape)
        .unwrap();
    a.eval().unwrap();
    a
}

#[test]
fn compacted_state_retains_no_more_than_it_counts() {
    let before = active_bytes();
    // Like a prefill: conv rows are the tail of concat(state, 4096-token
    // chunk); the SSM state is one layer of a stacked all-layer output.
    let layers = (0..4)
        .map(|_| {
            let chunk = filled(&[1, 4098, 1024]);
            let stacked = filled(&[8, 1, 16, 64, 64]);
            Some(LayerCache::Arrays(ArraysCache {
                conv_state: Some(chunk.index((.., 4096.., ..))),
                ssm_state: Some(stacked.index(3)),
                conv_pos: 0,
                offset: 4096,
            }))
        })
        .chain(std::iter::once(Some(LayerCache::KV(
            SteppingKeyValueCache::from_arrays(filled(&[1, 2, 64, 8]), filled(&[1, 2, 64, 8]))
                .unwrap(),
        ))))
        .collect();
    let mut cache = AnyCache::Hybrid(layers);
    cache.compact_state().unwrap();

    let counted = i64::try_from(cache.estimated_bytes()).unwrap();
    // Metal releases dropped parents from its completion handler, shortly
    // after eval returns; wait for it rather than racing it.
    let slack = 1 << 20;
    let mut retained = active_bytes() - before;
    for _ in 0..200 {
        if retained <= counted + slack {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        retained = active_bytes() - before;
    }
    assert!(
        retained <= counted + slack,
        "cache retains {retained} B but counts {counted} B: state still pins its parents"
    );
}
