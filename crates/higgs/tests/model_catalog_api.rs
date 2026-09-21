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
        &HashMap::from([(
            resident.canonicalize().unwrap(),
            "Publisher/ActiveAlias".to_owned(),
        )]),
        &HashMap::from([(
            resident.canonicalize().unwrap(),
            "Publisher/ConfiguredAlias".to_owned(),
        )]),
    );
    let resident_path = resident.canonicalize().unwrap();
    let resident_records = catalog
        .data
        .iter()
        .filter(|record| record.path == resident_path)
        .collect::<Vec<_>>();

    assert_eq!(resident_records.len(), 1);
    assert_eq!(resident_records[0].id, "Publisher/ActiveAlias");
    assert_eq!(resident_records[0].stable_id, "Publisher/MetadataName");
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
fn active_alias_is_selectable_while_stable_scanner_identity_is_preserved() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("runtime-model");
    write_model(&model, "Publisher/Metadata");
    let canonical = model.canonicalize().unwrap();
    let models = higgs::retention_plan::scan_models(&[root.path().to_path_buf()]);
    let loaded = higgs::routes::models::available_model_catalog(
        true,
        &models,
        &HashMap::from([(canonical.clone(), "local:active".to_owned())]),
        &HashMap::new(),
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
        "local:active"
    );
    assert_eq!(
        loaded
            .data
            .iter()
            .find(|record| record.path == canonical)
            .unwrap()
            .stable_id,
        "Publisher/Metadata"
    );
}

#[test]
fn configured_alias_is_selectable_when_artifact_is_not_loaded() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("configured-model");
    write_model(&model, "Publisher/Metadata");
    let canonical = model.canonicalize().unwrap();
    let models = higgs::retention_plan::scan_models(&[root.path().to_path_buf()]);

    let catalog = higgs::routes::models::available_model_catalog(
        true,
        &models,
        &HashMap::new(),
        &HashMap::from([(canonical.clone(), "local:configured".to_owned())]),
    );
    let record = catalog
        .data
        .iter()
        .find(|record| record.path == canonical)
        .unwrap();

    assert_eq!(record.id, "local:configured");
    assert_eq!(record.stable_id, "Publisher/Metadata");
    assert!(!record.loaded);
}

#[test]
fn publication_and_unload_snapshots_change_selection_without_duplicates() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("lifecycle-model");
    write_model(&model, "Publisher/Stable");
    let canonical = model.canonicalize().unwrap();
    let models = higgs::retention_plan::scan_models(&[root.path().to_path_buf()]);
    let configured = HashMap::from([(canonical.clone(), "local:configured".to_owned())]);

    let published = higgs::routes::models::available_model_catalog(
        true,
        &models,
        &HashMap::from([(canonical.clone(), "local:active".to_owned())]),
        &configured,
    );
    assert_eq!(published.data.len(), 1);
    assert_eq!(published.data[0].id, "local:active");
    assert!(published.data[0].loaded);

    let unloaded = higgs::routes::models::available_model_catalog(
        true,
        &models,
        &HashMap::new(),
        &configured,
    );
    assert_eq!(unloaded.data.len(), 1);
    assert_eq!(unloaded.data[0].id, "local:configured");
    assert!(!unloaded.data[0].loaded);
    assert_eq!(unloaded.data[0].stable_id, "Publisher/Stable");
}
