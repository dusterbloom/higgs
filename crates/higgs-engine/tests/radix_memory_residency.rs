//! The radix prefix cache's byte ledger must describe the device memory it
//! actually keeps alive.
//!
//! A multi-turn conversation stores a growing prefix every turn, each from a
//! fresh prefill buffer. Blocks that are MLX slice views pin their whole
//! source buffer, so live memory would grow with the sum of every turn's
//! buffer while `resident_bytes` counts only the deduplicated blocks. Own test
//! binary so no concurrent test perturbs the process-wide MLX counters.
#![allow(clippy::unwrap_used, clippy::tests_outside_test_module, unsafe_code)]

use higgs_engine::paged_prefix_cache::PagedPrefixCache;
use higgs_models::AnyCache;
use higgs_models::cache::SteppingKeyValueCache;
use mlx_rs::{Array, transforms::eval};

fn active_bytes() -> usize {
    let mut bytes = 0usize;
    assert_eq!(unsafe { mlx_sys::mlx_get_active_memory(&raw mut bytes) }, 0);
    bytes
}

/// One evaluated, independently allocated KV buffer per layer, like a prefill.
fn prefill_cache(layers: usize, seq_len: i32) -> AnyCache {
    AnyCache::KV(
        (0..layers)
            .map(|_| {
                let keys = Array::ones::<f32>(&[1, 8, seq_len, 128]).unwrap();
                let values = Array::ones::<f32>(&[1, 8, seq_len, 128]).unwrap();
                eval([&keys, &values]).unwrap();
                Some(SteppingKeyValueCache::from_arrays(keys, values).unwrap())
            })
            .collect(),
    )
}

#[test]
fn radix_live_memory_matches_resident_ledger_across_turns() {
    let block = 32;
    let mut radix = PagedPrefixCache::new(8, block);
    let baseline = active_bytes();

    for turn in 1..=12 {
        let len = turn * 4 * block;
        let tokens: Vec<u32> = (0..u32::try_from(len).unwrap()).collect();
        let cache = prefill_cache(2, i32::try_from(len).unwrap());
        radix.store(&tokens, &cache);
    } // every turn's prefill cache is dropped here, as after a real request

    let live = active_bytes().saturating_sub(baseline);
    let resident = radix.radix_byte_stats().resident_bytes;
    assert!(resident > 0, "the radix must hold the conversation");
    assert!(
        live <= resident + resident / 8,
        "radix keeps {live} live bytes but accounts {resident}: dropped prefill buffers are pinned"
    );
}

#[test]
fn radix_live_memory_matches_resident_ledger_after_splits_and_evictions() {
    let block = 32;
    let mut radix = PagedPrefixCache::new(3, block);
    let baseline = active_bytes();

    // Conversations share a 40-block trunk, branch at varying depths (each
    // branch splits an existing edge), and the 3-entry cap keeps evicting.
    for branch in 0u32..10 {
        let shared = usize::try_from(8 + branch * 3).unwrap() * block;
        let len = shared + 16 * block;
        let tokens: Vec<u32> = (0..u32::try_from(len).unwrap())
            .map(|t| {
                if (t as usize) < shared {
                    t
                } else {
                    100_000 * (branch + 1) + t
                }
            })
            .collect();
        let cache = prefill_cache(2, i32::try_from(len).unwrap());
        radix.store(&tokens, &cache);
    }

    let live = active_bytes().saturating_sub(baseline);
    let resident = radix.radix_byte_stats().resident_bytes;
    assert!(resident > 0, "the radix must hold the surviving branches");
    assert!(
        live <= resident + resident / 8,
        "radix keeps {live} live bytes but accounts {resident} after splits and evictions"
    );
}
