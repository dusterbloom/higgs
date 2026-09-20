use higgs::retention_plan::{BudgetRequest, plan_from_geometry, scan_roots};

#[test]
fn byte_authority_and_model_cost_drive_safe_tokens() {
    let cheap = plan_from_geometry("cheap", 65_536, 4_096, 0, 64 << 10, 1, BudgetRequest::Bytes(4 << 30)).unwrap();
    let paired = plan_from_geometry("paired", 65_536, 4_096, 0, 128 << 10, 1, BudgetRequest::Bytes(4 << 30)).unwrap();
    assert!(cheap.safe_prompt_tokens > paired.safe_prompt_tokens);
    assert_eq!(paired.persisted_authority, "retained_bytes");
}

#[test]
fn token_request_resolves_bytes_and_impossible_target_reports_maximum() {
    let plan = plan_from_geometry("model", 65_536, 4_096, 0, 128 << 10, 1, BudgetRequest::Tokens(20_000)).unwrap();
    assert!(plan.safe_prompt_tokens >= 20_000 && plan.retained_budget_bytes > 0);
    assert!(plan_from_geometry("model", 8_192, 4_096, 0, 128 << 10, 1, BudgetRequest::Tokens(8_000)).unwrap_err().contains("maximum safe"));
}

#[test]
fn scan_discovers_supported_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("models--org--name/snapshots/hash");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    assert_eq!(scan_roots(&[root.path().into()]), vec![model]);
}
