//! Cache-only TTFT costs at Clover geometry; no model weights are loaded.
//! cargo run --release -p higgs-engine --example ttft_cache_costs -- 2048 8192
#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing
)]
use higgs_engine::paged_prefix_cache::{DEFAULT_BLOCK_SIZE, PagedPrefixCache};
use higgs_models::cache::SteppingKeyValueCache;
use higgs_models::mlx_exec::{acquire, eval};
use higgs_models::qwen3_next::ArraysCache;
use higgs_models::{AnyCache, LayerCache};
use mlx_rs::{Dtype, ops};
use std::time::Instant;

fn fixture(tokens: i32) -> AnyCache {
    let layers = (0..40)
        .map(|i| {
            Some(if i % 4 == 3 {
                let k = ops::ones_dtype(&[1, 2, tokens, 256], Dtype::Float16).unwrap();
                let v = ops::ones_dtype(&[1, 2, tokens, 256], Dtype::Float16).unwrap();
                LayerCache::KV(SteppingKeyValueCache::from_arrays(k, v).unwrap())
            } else {
                LayerCache::Arrays(ArraysCache {
                    conv_state: Some(ops::ones_dtype(&[1, 3, 8192], Dtype::Float32).unwrap()),
                    ssm_state: Some(ops::ones_dtype(&[1, 32, 128, 128], Dtype::Float32).unwrap()),
                    conv_pos: 0,
                    offset: tokens,
                })
            })
        })
        .collect();
    let cache = AnyCache::Hybrid(layers);
    eval(cache.eval_targets()).unwrap();
    cache
}

fn main() {
    let _gate = acquire();
    let sizes: Vec<i32> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let sizes = if sizes.is_empty() {
        vec![2048, 8192]
    } else {
        sizes
    };
    for size in sizes {
        assert!(size > 0 && size % DEFAULT_BLOCK_SIZE as i32 == 0);
        let original = fixture(size);
        let tokens: Vec<u32> = (0..size as u32).collect();
        let mut radix = PagedPrefixCache::new(8, DEFAULT_BLOCK_SIZE);
        for iteration in 0..6 {
            let t = Instant::now();
            let snapshot = original.try_deep_clone().unwrap();
            eval(snapshot.eval_targets()).unwrap();
            let fork_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            radix.store(&tokens, &snapshot);
            let store_ms = t.elapsed().as_secs_f64() * 1000.0;
            drop(snapshot);
            let t = Instant::now();
            let restored = radix
                .find_longest_prefix(&tokens)
                .expect("exact prefix must hit");
            eval(restored.cache.eval_targets()).unwrap();
            let restore_ms = t.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(restored.prefix_len, size as usize);
            assert_eq!(restored.cache.estimated_bytes(), original.estimated_bytes());
            println!(
                "{{\"tokens\":{size},\"iteration\":{iteration},\"cache_bytes\":{},\"radix_resident_bytes\":{},\"fork_ms\":{fork_ms:.3},\"radix_store_ms\":{store_ms:.3},\"radix_restore_ms\":{restore_ms:.3}}}",
                original.estimated_bytes(),
                radix.radix_byte_stats().resident_bytes
            );
        }
    }
}
