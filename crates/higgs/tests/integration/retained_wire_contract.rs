use axum::response::IntoResponse;
use higgs::capacity::{FastSessionContractInputs, FastSessionContractV2};
use higgs::error::{RetentionErrorContext, ServerError};
use higgs::types::openai::ChatCompletionRequest;
use http_body_util::BodyExt;

async fn json(error: ServerError) -> serde_json::Value {
    let response = error.into_response();
    assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
    serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
}

fn context(revision: &str) -> RetentionErrorContext {
    RetentionErrorContext {
        contract_revision: revision.to_owned(),
        session_id: 42,
        epoch: 7,
    }
}

#[tokio::test]
async fn typed_retention_409s_match_shared_golden_envelopes() {
    let cases = [
        (
            ServerError::RetentionCompactionRequired(context("boot:7:sha256:model")),
            include_str!("../fixtures/retention_compaction_required.json"),
        ),
        (
            ServerError::StaleRetentionContract(context("stale:revision")),
            include_str!("../fixtures/stale_retention_contract.json"),
        ),
        (
            ServerError::RequiredRetentionUnavailable(context("boot:7:sha256:model")),
            include_str!("../fixtures/retained_session_unavailable.json"),
        ),
    ];
    for (error, expected) in cases {
        assert_eq!(
            json(error).await,
            serde_json::from_str::<serde_json::Value>(expected).unwrap()
        );
    }
}

#[tokio::test]
async fn stateless_error_shape_remains_openai_compatible() {
    let body = json(ServerError::Conflict("busy".into())).await;
    assert_eq!(body["error"]["type"], "conflict");
    assert!(body["error"].get("contractRevision").is_none());
    assert!(body["error"].get("sessionId").is_none());
    assert!(body["error"].get("epoch").is_none());
}

#[test]
fn v2_contract_and_malformed_request_complete_the_wire_matrix() {
    let contract = FastSessionContractV2::new(FastSessionContractInputs {
        contract_revision: "boot:7:sha256:model".into(),
        model: "model".into(),
        max_context_tokens: 65_536,
        max_output_tokens: 4_096,
        retained_budget_bytes: 4 * 1024 * 1024 * 1024,
        guaranteed_sessions: 1,
        legacy_token_upper_bound: 96_576,
        fixed_bytes_per_session: 0,
        conservative_bytes_per_token: 128 * 1024,
        worst_case_turn_tokens: 4_096,
        target_after_compaction_tokens: 4_096,
    })
    .unwrap();
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/fast_session_contract_v2.json")).unwrap();
    assert_eq!(serde_json::to_value(contract).unwrap(), expected);

    let malformed = serde_json::json!({"model":"model","messages":[],"retention":{"mode":"required","sessionId":42,"contractRevision":"revision"}});
    assert!(serde_json::from_value::<ChatCompletionRequest>(malformed).is_err());
    let stateless = serde_json::json!({"model":"model","messages":[]});
    assert!(serde_json::from_value::<ChatCompletionRequest>(stateless).is_ok());
}
