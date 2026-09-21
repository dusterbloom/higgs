use higgs::retention_plan::{BudgetRequest, plan_from_geometry, scan_roots};

#[test]
fn byte_authority_and_model_cost_drive_safe_tokens() {
    let cheap = plan_from_geometry(
        "cheap",
        65_536,
        4_096,
        0,
        64 << 10,
        1,
        BudgetRequest::Bytes(4 << 30),
    )
    .unwrap();
    let paired = plan_from_geometry(
        "paired",
        65_536,
        4_096,
        0,
        128 << 10,
        1,
        BudgetRequest::Bytes(4 << 30),
    )
    .unwrap();
    assert!(cheap.safe_prompt_tokens > paired.safe_prompt_tokens);
    assert_eq!(paired.persisted_authority, "retained_bytes");
}

#[test]
fn token_request_resolves_bytes_and_impossible_target_reports_maximum() {
    let plan = plan_from_geometry(
        "model",
        65_536,
        4_096,
        0,
        128 << 10,
        1,
        BudgetRequest::Tokens(20_000),
    )
    .unwrap();
    assert!(plan.safe_prompt_tokens >= 20_000 && plan.retained_budget_bytes > 0);
    assert!(
        plan_from_geometry(
            "model",
            8_192,
            4_096,
            0,
            128 << 10,
            1,
            BudgetRequest::Tokens(8_000)
        )
        .unwrap_err()
        .contains("maximum safe")
    );
}

#[test]
fn scan_discovers_supported_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("models--org--name/snapshots/hash");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    assert_eq!(scan_roots(&[root.path().into()]), vec![model]);
}

#[test]
fn planner_uses_complete_target_and_draft_geometry() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("target");
    let draft = root.path().join("draft");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::create_dir_all(&draft).unwrap();
    let target_config = r#"{"max_position_embeddings":65536,"num_hidden_layers":8,"num_key_value_heads":4,"head_dim":128,"hidden_size":1024,"num_attention_heads":8}"#;
    let draft_config = r#"{"max_position_embeddings":65536,"num_hidden_layers":8,"num_key_value_heads":4,"head_dim":128,"hidden_size":1024,"num_attention_heads":8,"intermediate_size":2048,"vocab_size":32000,"dflash_config":{"target_layer_ids":[1,6]}}"#;
    std::fs::write(target.join("config.json"), target_config).unwrap();
    std::fs::write(draft.join("config.json"), draft_config).unwrap();
    let solo =
        higgs::retention_plan::plan_model(&target, BudgetRequest::Bytes(1 << 30), 1, 4096).unwrap();
    let paired = higgs::retention_plan::plan_model_with_draft(
        &target,
        Some(&draft),
        BudgetRequest::Bytes(1 << 30),
        1,
        4096,
    )
    .unwrap();
    assert!(paired.safe_prompt_tokens < solo.safe_prompt_tokens);
}
