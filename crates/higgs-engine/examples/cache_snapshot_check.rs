//! Standalone snapshot safety checks, runnable without the unrelated model unit-test build.
use higgs_models::{
    AnyCache, LayerCache,
    cache::{KeyValueCache, SteppingKeyValueCache},
    mlx_exec,
    qwen3_next::ArraysCache,
};
use mlx_rs::{Array, Dtype};
fn normalized(a: &Array) -> Array {
    a.flatten(None, None).unwrap().reshape(a.shape()).unwrap()
}
fn exact_bits(a: &Array) -> Vec<u32> {
    match a.dtype() {
        Dtype::Uint32 => a.as_slice::<u32>().to_vec(),
        Dtype::Float32 => a.as_slice::<f32>().iter().map(|v| v.to_bits()).collect(),
        Dtype::Float16 => a.as_slice::<half::f16>().iter().map(|v| u32::from(v.to_bits())).collect(),
        other => panic!("unexpected dtype {other:?}"),
    }
}
fn check(original: &AnyCache) -> AnyCache {
    let copy = original.try_deep_clone().unwrap();
    assert_eq!(copy.estimated_bytes(), original.estimated_bytes());
    for (a, b) in original.eval_targets().into_iter().zip(copy.eval_targets()) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.dtype(), b.dtype());
        let (a, b) = (normalized(a), normalized(b));
        mlx_exec::eval([&a, &b]).unwrap();
        match a.dtype() {
            Dtype::Float32 => assert_eq!(a.as_slice::<f32>(), b.as_slice::<f32>()),
            Dtype::Float16 => assert_eq!(a.as_slice::<half::f16>(), b.as_slice::<half::f16>()),
            Dtype::Uint32 => assert_eq!(a.as_slice::<u32>(), b.as_slice::<u32>()),
            other => panic!("unexpected dtype {other:?}"),
        }
    }
    copy
}
fn main() {
    let gate = mlx_exec::acquire();
    let backing = Array::from_slice(
        &(0..24).map(|x| x as f32).collect::<Vec<_>>(),
        &[1, 3, 2, 4],
    );
    let kv = SteppingKeyValueCache::from_arrays(
        backing.transpose_axes(&[0, 2, 1, 3]).unwrap(),
        backing.transpose_axes(&[0, 2, 1, 3]).unwrap(),
    )
    .unwrap();
    let mut original = AnyCache::Hybrid(vec![
        Some(LayerCache::KV(kv)),
        None,
        Some(LayerCache::Arrays(ArraysCache {
            conv_state: Some(backing.transpose_axes(&[0, 2, 1, 3]).unwrap()),
            ssm_state: Some(Array::from_slice(&[5_f32, 6., 7., 8.], &[1, 2, 2])),
            conv_pos: 3,
            offset: 3,
        })),
    ]);
    let copy = check(&original);
    for (a, b) in original.eval_targets().into_iter().zip(copy.eval_targets()) {
        assert_ne!(a.as_ptr().ctx, b.as_ptr().ctx);
    }
    let before: Vec<Vec<f32>> = copy
        .eval_targets()
        .into_iter()
        .map(|a| a.as_slice::<f32>().to_vec())
        .collect();
    if let AnyCache::Hybrid(layers) = &mut original {
        if let Some(LayerCache::KV(kv)) = &mut layers[0] {
            kv.update_and_fetch(
                Array::ones::<f32>(&[1, 2, 1, 4]).unwrap(),
                Array::ones::<f32>(&[1, 2, 1, 4]).unwrap(),
            )
            .unwrap();
        }
        if let Some(LayerCache::Arrays(state)) = &mut layers[2] {
            state.ssm_state = Some(Array::zeros::<f32>(&[1, 2, 2]).unwrap());
        }
    }
    original.eval().unwrap();
    for (a, expected) in copy.eval_targets().into_iter().zip(before) {
        assert_eq!(a.as_slice::<f32>(), expected);
    }
    if let AnyCache::Hybrid(layers) = &copy {
        assert!(layers[1].is_none());
        if let Some(LayerCache::Arrays(state)) = &layers[2] {
            assert_eq!((state.conv_pos, state.offset), (3, 3));
        }
    }
    let empty = AnyCache::KV(vec![None, Some(SteppingKeyValueCache::new())]);
    check(&empty);
    let mla = AnyCache::KV(vec![Some(
        SteppingKeyValueCache::from_latent_array(Array::ones::<f32>(&[1, 1, 3, 8]).unwrap(), 4, 4)
            .unwrap(),
    )]);
    check(&mla);
    let cfg = higgs_models::turboquant::KvCacheConfig {
        mode: higgs_models::turboquant::KvCacheMode::Turboquant,
        bits: 3,
        seed: 17,
        ..Default::default()
    };
    let mut tq = SteppingKeyValueCache::new();
    let rows = Array::ones::<f32>(&[1, 2, 4, 8]).unwrap();
    tq.update_and_fetch(rows.clone(), rows).unwrap();
    assert!(tq.quantize_for_retention(cfg).unwrap());
    let mut tq = AnyCache::KV(vec![Some(tq)]);
    let tq_copy = check(&tq);
    let saved: Vec<Vec<u32>> = tq_copy.eval_targets().into_iter().map(exact_bits).collect();
    if let AnyCache::KV(layers) = &mut tq {
        let next = Array::zeros::<f32>(&[1, 2, 1, 8]).unwrap();
        layers[0]
            .as_mut()
            .unwrap()
            .update_and_view(next.clone(), next)
            .unwrap();
    }
    tq.eval().unwrap();
    for (a, expected) in tq_copy.eval_targets().into_iter().zip(saved) {
        assert_eq!(exact_bits(a), expected);
    }
    original.eval().unwrap();
    drop(gate);
    // Whole-cache clone also acquires its own execution gate when called off-gate.
    original.try_deep_clone().unwrap();
    println!(
        "PASS: strided KV/recurrent values, independent allocations, live-update isolation, sparse/empty layers, MLA, TurboQuant, off-gate clone"
    );
}
