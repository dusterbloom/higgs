use higgs::capacity::{
    FastSessionContractInputs, FastSessionContractV2, RequiredRetentionAdmissionError,
};
use higgs::types::openai::{ChatCompletionRequest, RetentionMode};
use higgs::types::openai::{CompletionUsage, RetentionReceipt};

fn inputs(bytes_per_token: u64) -> FastSessionContractInputs {
    FastSessionContractInputs {
        contract_revision: "boot:3:sha256:model".to_owned(),
        model: "model".to_owned(),
        max_context_tokens: 65_536,
        max_output_tokens: 4_096,
        retained_budget_bytes: 4 * 1024 * 1024 * 1024,
        guaranteed_sessions: 1,
        legacy_token_upper_bound: 96_576,
        fixed_bytes_per_session: 0,
        conservative_bytes_per_token: bytes_per_token,
        worst_case_turn_tokens: 4_096,
        target_after_compaction_tokens: 4_096,
    }
}

#[test]
fn paired_cache_cost_reduces_the_same_byte_budgets_guarantee() {
    let target_only = FastSessionContractV2::new(inputs(64 * 1024)).unwrap();
    let target_and_draft = FastSessionContractV2::new(inputs(128 * 1024)).unwrap();

    assert!(
        target_only.guaranteed_fast_prompt_tokens()
            > target_and_draft.guaranteed_fast_prompt_tokens()
    );
}

#[test]
fn parses_the_tagged_required_retention_wire_shape() {
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "retention": {
            "mode": "required",
            "sessionId": 42,
            "epoch": 7,
            "contractRevision": "boot:3:sha256:model"
        }
    }))
    .unwrap();
    let retention = request.retention.unwrap();

    assert_eq!(retention.mode, RetentionMode::Required);
    assert_eq!(retention.session_id, 42);
    assert_eq!(retention.epoch, 7);
    assert_eq!(retention.contract_revision, "boot:3:sha256:model");
}

#[test]
fn contract_admission_rejects_stale_and_oversized_requests() {
    let contract = FastSessionContractV2::new(inputs(128 * 1024)).unwrap();
    assert_eq!(
        contract.admit_required("stale", 1, 1),
        Err(RequiredRetentionAdmissionError::StaleContract)
    );
    assert_eq!(
        contract.admit_required(
            contract.contract_revision(),
            contract.guaranteed_fast_prompt_tokens() + 1,
            contract.max_output_tokens(),
        ),
        Err(RequiredRetentionAdmissionError::CompactionRequired)
    );
    assert!(
        contract
            .admit_required(
                contract.contract_revision(),
                contract.guaranteed_fast_prompt_tokens(),
                contract.max_output_tokens(),
            )
            .is_ok()
    );
}

#[test]
fn retention_receipt_is_additive_and_absent_for_stateless_usage() {
    let stateless = serde_json::to_value(CompletionUsage::new(8, 2, 0)).unwrap();
    assert!(stateless.get("higgs_retention").is_none());

    let retained = CompletionUsage::new(8, 2, 8).with_retention_receipt(Some(RetentionReceipt {
        outcome: "seeded",
        session_id: 42,
        epoch: 7,
        retained_tokens: 10,
        retained_bytes: 1_310_720,
        contract_revision: "boot:3:sha256:model".to_owned(),
    }));
    let json = serde_json::to_value(retained).unwrap();
    assert_eq!(json["higgs_retention"]["outcome"], "seeded");
    assert_eq!(json["higgs_retention"]["retainedBytes"], 1_310_720);
    assert_eq!(
        json["higgs_retention"]["contractRevision"],
        "boot:3:sha256:model"
    );
}

#[test]
fn seed_is_bounded_by_compaction_target_while_required_uses_fast_wall() {
    let contract = FastSessionContractV2::new(inputs(128 * 1024)).unwrap();
    assert!(
        contract
            .admit_seed(
                contract.contract_revision(),
                contract.target_after_compaction_tokens(),
                contract.max_output_tokens()
            )
            .is_ok()
    );
    assert_eq!(
        contract.admit_seed(
            contract.contract_revision(),
            contract.target_after_compaction_tokens() + 1,
            1
        ),
        Err(RequiredRetentionAdmissionError::CompactionRequired)
    );

    let seed: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model":"model","messages":[],"retention":{"mode":"seed","sessionId":99,"epoch":8,"contractRevision":"boot:3:sha256:model"}
    })).unwrap();
    assert_eq!(seed.retention.unwrap().mode, RetentionMode::Seed);
}
