#![allow(clippy::unwrap_used)]

#[test]
fn authoritative_custom_hf_cache_discovers_unconfigured_model() {
    let cache = tempfile::tempdir().unwrap();
    let model = cache
        .path()
        .join("models--Nanbeige--Nanbeige4.1-3B/snapshots/deadbeef");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(
        model.join("config.json"),
        r#"{"model_type":"llama","_name_or_path":"Nanbeige/Nanbeige4.1-3B"}"#,
    )
    .unwrap();
    std::fs::write(model.join("tokenizer.json"), "{}").unwrap();
    std::fs::write(model.join("model.safetensors"), "weights").unwrap();

    let roots = higgs::model_resolver::local_model_roots_from(
        Some(cache.path().to_path_buf()),
        None,
    );
    let catalog = higgs::retention_plan::scan_models(&roots);

    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0].id, "Nanbeige/Nanbeige4.1-3B");
    assert_eq!(catalog[0].path, model.canonicalize().unwrap());
}
