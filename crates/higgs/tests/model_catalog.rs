#![allow(clippy::unwrap_used)]

use std::path::Path;

use higgs::retention_plan::scan_models;

fn write_metadata(path: &Path, config: &[u8]) {
    std::fs::create_dir_all(path).unwrap();
    std::fs::write(path.join("config.json"), config).unwrap();
}

fn write_valid_model(path: &Path, config: &str) {
    write_metadata(path, config.as_bytes());
    std::fs::write(path.join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(path.join("model.safetensors"), b"weights").unwrap();
}

#[test]
fn catalog_accepts_valid_model() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    write_valid_model(&model, r#"{"model_type":"llama"}"#);

    let available = scan_models(&[root.path().to_path_buf()]);

    assert_eq!(available.len(), 1);
    assert_eq!(available[0].path, model.canonicalize().unwrap());
    assert_eq!(available[0].adapter, "transformer-dense");
}

#[cfg(unix)]
#[test]
fn catalog_deduplicates_canonical_path() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    let alias = root.path().join("alias");
    write_valid_model(&model, r#"{"model_type":"llama"}"#);
    symlink(&model, &alias).unwrap();

    let available = scan_models(&[model, alias]);

    assert_eq!(available.len(), 1);
}

#[test]
fn catalog_rejects_malformed_metadata() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    write_metadata(&model, b"not json");
    std::fs::write(model.join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(model.join("model.safetensors"), b"weights").unwrap();

    assert!(scan_models(&[root.path().to_path_buf()]).is_empty());
}

#[test]
fn catalog_rejects_tokenizer_config_without_runtime_tokenizer() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    write_metadata(&model, br#"{"model_type":"llama"}"#);
    std::fs::write(model.join("tokenizer_config.json"), b"{}").unwrap();
    std::fs::write(model.join("model.safetensors"), b"weights").unwrap();

    assert!(scan_models(&[root.path().to_path_buf()]).is_empty());
}

#[test]
fn catalog_rejects_mtp_only_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    write_metadata(&model, br#"{"model_type":"llama"}"#);
    std::fs::write(model.join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(model.join("mtp.safetensors"), b"weights").unwrap();

    assert!(scan_models(&[root.path().to_path_buf()]).is_empty());
}

#[test]
fn catalog_rejects_incomplete_indexed_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    write_metadata(&model, br#"{"model_type":"llama"}"#);
    std::fs::write(model.join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(
        model.join("model.safetensors.index.json"),
        br#"{"metadata":{},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
    )
    .unwrap();
    std::fs::write(model.join("model-00001-of-00002.safetensors"), b"weights").unwrap();

    assert!(scan_models(&[root.path().to_path_buf()]).is_empty());
}

#[test]
fn catalog_uses_metadata_name_then_directory_fallback() {
    let root = tempfile::tempdir().unwrap();
    let named = root.path().join("named-leaf");
    let fallback = root.path().join("fallback-leaf");
    write_valid_model(
        &named,
        r#"{"model_type":"llama","_name_or_path":"Publisher/StableName"}"#,
    );
    write_valid_model(&fallback, r#"{"model_type":"llama"}"#);

    let available = scan_models(&[root.path().to_path_buf()]);
    let names = available
        .iter()
        .map(|model| model.id.as_str())
        .collect::<Vec<_>>();

    assert!(names.contains(&"Publisher/StableName"));
    assert!(names.contains(&"fallback-leaf"));
}
