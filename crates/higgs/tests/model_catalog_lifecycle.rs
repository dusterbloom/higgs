#![allow(clippy::unwrap_used)]

use std::{
    collections::HashSet,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use higgs::{
    config::HiggsConfig, model_resolver::load_after_model_path_preflight, router::Router,
    state::ModelCatalogCache,
};

#[test]
fn duplicate_startup_path_is_rejected_before_loader_runs() {
    let mut claimed = HashSet::new();
    let path = PathBuf::from("/canonical/model");
    let calls = AtomicUsize::new(0);

    load_after_model_path_preflight(&mut claimed, path.clone(), || {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok::<_, String>(())
    })
    .unwrap();
    let duplicate = load_after_model_path_preflight(&mut claimed, path, || {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok::<_, String>(())
    });

    assert!(duplicate.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn duplicate_runtime_path_is_rejected_before_loader_runs() {
    let router = Router::from_config(&HiggsConfig::default(), Default::default()).unwrap();
    let path = PathBuf::from("/canonical/runtime-model");
    let _first_load = router.reserve_model_path(path.clone()).unwrap();
    let calls = AtomicUsize::new(0);

    let duplicate = router.reserve_model_path(path).map(|_reservation| {
        calls.fetch_add(1, Ordering::SeqCst);
    });

    assert!(duplicate.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cancelled_catalog_request_does_not_start_a_second_scan() {
    let cache = ModelCatalogCache::default();
    let scans = Arc::new(AtomicUsize::new(0));
    let roots = vec![PathBuf::from("/catalog")];
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let first_cache = cache.clone();
    let first_scans = Arc::clone(&scans);
    let first_roots = roots.clone();
    let first = tokio::spawn(async move {
        first_cache
            .get_or_scan(first_roots, move |_| {
                first_scans.fetch_add(1, Ordering::SeqCst);
                let _ = started_tx.send(());
                std::thread::sleep(Duration::from_millis(100));
                Vec::new()
            })
            .await
    });
    started_rx.await.unwrap();
    first.abort();

    let second_scans = Arc::clone(&scans);
    cache
        .get_or_scan(roots, move |_| {
            second_scans.fetch_add(1, Ordering::SeqCst);
            Vec::new()
        })
        .await;

    assert_eq!(scans.load(Ordering::SeqCst), 1);
}
