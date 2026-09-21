#![allow(clippy::unwrap_used)]

use std::{collections::HashMap, sync::Arc};

use axum::body::Body;
use higgs::{build_router, router::Router, state::AppState};
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

fn write_model(path: &std::path::Path, name: &str) {
    std::fs::create_dir_all(path).unwrap();
    std::fs::write(
        path.join("config.json"),
        serde_json::json!({"model_type": "llama", "_name_or_path": name}).to_string(),
    )
    .unwrap();
    std::fs::write(path.join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(path.join("model.safetensors"), b"weights").unwrap();
}

fn state_with_catalog_root(root: &std::path::Path) -> Arc<AppState> {
    let config_dir = tempfile::tempdir().unwrap();
    let config_path = config_dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        format!(
            r#"
            [local]
            allow_runtime_model_load = true

            [[models]]
            path = "{}"
            name = "catalog-root"

            [provider.mock]
            url = "http://127.0.0.1:1"

            [default]
            provider = "mock"
            "#,
            root.display(),
        ),
    )
    .unwrap();
    let config = higgs::config::load_config_file(&config_path, None).unwrap();
    let router = Router::from_config(&config, HashMap::new()).unwrap();
    Arc::new(AppState::new(router, config, reqwest::Client::new(), None))
}

#[tokio::test]
async fn available_models_endpoint_returns_typed_catalog() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("Nanbeige4.1-3B");
    write_model(&model, "Nanbeige/Nanbeige4.1-3B");
    let app = build_router(
        state_with_catalog_root(root.path()),
        30.0,
        None,
        0,
        1024,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/available")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["runtime_model_load"], true);
    let canonical = model.canonicalize().unwrap();
    let record = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|record| record["path"] == canonical.to_string_lossy().as_ref())
        .unwrap();
    assert_eq!(record["id"], "Nanbeige/Nanbeige4.1-3B");
    assert_eq!(record["model_type"], "llama");
    assert_eq!(record["adapter"], "transformer-dense");
    assert_eq!(record["loaded"], false);
}

#[cfg(unix)]
#[test]
fn available_models_marks_only_the_exact_configured_artifact_loaded() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let resident = root.path().join("resident-artifact");
    let decoy = root.path().join("same-name-decoy");
    let alias = root.path().join("resident-alias");
    write_model(&resident, "Publisher/MetadataName");
    write_model(&decoy, "Publisher/Resident");
    symlink(&resident, &alias).unwrap();
    let config_dir = tempfile::tempdir().unwrap();
    let config_path = config_dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        format!(
            r#"
            [local]
            allow_runtime_model_load = true

            [[models]]
            path = "{}"
            name = "Publisher/Resident"

            [[models]]
            path = "{}"
            name = "catalog-root"
            "#,
            alias.display(),
            root.path().display(),
        ),
    )
    .unwrap();
    let config = higgs::config::load_config_file(&config_path, None).unwrap();
    let models = higgs::retention_plan::scan_models(&[root.path().to_path_buf()]);
    let catalog = higgs::routes::models::available_model_catalog(
        config.local.allow_runtime_model_load,
        &models,
        &std::collections::HashSet::from([resident.canonicalize().unwrap()]),
    );
    let resident_path = resident.canonicalize().unwrap();
    let resident_records = catalog
        .data
        .iter()
        .filter(|record| record.path == resident_path)
        .collect::<Vec<_>>();

    assert_eq!(resident_records.len(), 1);
    assert_eq!(resident_records[0].id, "Publisher/MetadataName");
    assert!(resident_records[0].loaded);
    let decoy = catalog
        .data
        .iter()
        .find(|record| record.path == decoy.canonicalize().unwrap())
        .unwrap();
    assert_eq!(decoy.id, "Publisher/Resident");
    assert!(!decoy.loaded);
}

#[tokio::test]
async fn available_models_endpoint_uses_api_bearer_authentication() {
    let root = tempfile::tempdir().unwrap();
    let app = build_router(
        state_with_catalog_root(root.path()),
        30.0,
        Some("secret".to_owned()),
        0,
        1024,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/available")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[test]
fn loaded_state_never_overrides_the_stable_scanner_name() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("runtime-model");
    write_model(&model, "Publisher/Metadata");
    let canonical = model.canonicalize().unwrap();
    let models = higgs::retention_plan::scan_models(&[root.path().to_path_buf()]);
    let loaded = higgs::routes::models::available_model_catalog(
        true,
        &models,
        &std::collections::HashSet::from([canonical.clone()]),
    );
    assert!(
        loaded
            .data
            .iter()
            .find(|record| record.path == canonical)
            .unwrap()
            .loaded
    );
    assert_eq!(
        loaded
            .data
            .iter()
            .find(|record| record.path == canonical)
            .unwrap()
            .id,
        "Publisher/Metadata"
    );
}
