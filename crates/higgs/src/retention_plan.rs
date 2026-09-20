use std::path::{Path, PathBuf};

use crate::capacity::{FastSessionContractInputs, FastSessionContractV2};

#[derive(Clone, Copy, Debug)]
pub enum BudgetRequest {
    Bytes(u64),
    Tokens(u64),
}

#[derive(Clone, Debug)]
pub struct RetentionPlan {
    pub model: String,
    pub retained_budget_bytes: u64,
    pub safe_prompt_tokens: u64,
    pub output_reserve_tokens: u64,
    pub target_after_compaction_tokens: u64,
    pub persisted_authority: &'static str,
}

pub fn plan_from_geometry(
    model: &str,
    context: u64,
    output: u64,
    fixed: u64,
    per_token: u64,
    sessions: u64,
    request: BudgetRequest,
) -> Result<RetentionPlan, String> {
    let build = |bytes| {
        FastSessionContractV2::new(FastSessionContractInputs {
            contract_revision: "planner".into(),
            model: model.into(),
            max_context_tokens: context,
            max_output_tokens: output,
            retained_budget_bytes: bytes,
            guaranteed_sessions: sessions,
            legacy_token_upper_bound: 0,
            fixed_bytes_per_session: fixed,
            conservative_bytes_per_token: per_token,
            worst_case_turn_tokens: output,
            target_after_compaction_tokens: output,
        })
    };
    let bytes = match request {
        BudgetRequest::Bytes(bytes) => bytes,
        BudgetRequest::Tokens(wanted) => {
            let architectural_max = context.saturating_sub(output);
            if wanted > architectural_max {
                return Err(format!(
                    "requested {wanted} tokens exceed maximum safe {architectural_max}"
                ));
            }
            let mut low = 1_u64;
            let mut high = u64::MAX;
            while low < high {
                let mid = low + (high - low) / 2;
                let fits = build(mid).is_ok_and(|c| c.guaranteed_fast_prompt_tokens() >= wanted);
                if fits {
                    high = mid;
                } else {
                    low = mid.saturating_add(1);
                }
            }
            low
        }
    };
    let contract = build(bytes).map_err(|e| e.to_string())?;
    Ok(RetentionPlan {
        model: model.into(),
        retained_budget_bytes: bytes,
        safe_prompt_tokens: contract.guaranteed_fast_prompt_tokens(),
        output_reserve_tokens: output,
        target_after_compaction_tokens: contract.target_after_compaction_tokens(),
        persisted_authority: "retained_bytes",
    })
}

pub fn scan_roots(roots: &[PathBuf]) -> Vec<PathBuf> {
    fn walk(path: &Path, out: &mut Vec<PathBuf>, depth: usize) {
        if depth > 5 {
            return;
        }
        if path.join("config.json").is_file() {
            out.push(path.to_path_buf());
            return;
        }
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                walk(&entry.path(), out, depth + 1);
            }
        }
    }
    let mut found = Vec::new();
    for root in roots {
        walk(root, &mut found, 0);
    }
    found.sort();
    found.dedup();
    found
}

pub fn plan_model(
    path: &Path,
    request: BudgetRequest,
    sessions: u64,
    output: u64,
) -> Result<RetentionPlan, String> {
    let raw = std::fs::read_to_string(path.join("config.json")).map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let context = json
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(32_768);
    let transient = higgs_engine::TransientPrefillEstimate {
        base_bytes: 0,
        bytes_per_prompt_token: 0,
        bytes_per_chunk_token: 0,
        max_prompt_tokens: u64::MAX,
        max_chunk_tokens: u64::MAX,
    };
    let cost = higgs_engine::EngineCostDescription::runtime_from_model_dir(path, 0, 0, transient)
        .ok_or("model lacks supported retained-cache geometry")?;
    plan_from_geometry(
        &path.display().to_string(),
        context,
        output,
        cost.fixed_live_session_bytes,
        cost.persistent_bytes_per_token,
        sessions,
        request,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_budget_produces_model_specific_safe_tokens() {
        let cheap = plan_from_geometry(
            "cheap",
            65_536,
            4_096,
            0,
            64 * 1024,
            1,
            BudgetRequest::Bytes(4 << 30),
        )
        .unwrap();
        let paired = plan_from_geometry(
            "paired",
            65_536,
            4_096,
            0,
            128 * 1024,
            1,
            BudgetRequest::Bytes(4 << 30),
        )
        .unwrap();
        assert!(cheap.safe_prompt_tokens > paired.safe_prompt_tokens);
    }

    #[test]
    fn desired_tokens_resolve_to_bytes_and_unsafe_targets_report_maximum() {
        let plan = plan_from_geometry(
            "model",
            65_536,
            4_096,
            0,
            128 * 1024,
            1,
            BudgetRequest::Tokens(20_000),
        )
        .unwrap();
        assert!(plan.retained_budget_bytes > 0);
        assert!(plan.safe_prompt_tokens >= 20_000);
        assert_eq!(plan.persisted_authority, "retained_bytes");

        let err = plan_from_geometry(
            "model",
            8_192,
            4_096,
            0,
            128 * 1024,
            1,
            BudgetRequest::Tokens(8_000),
        )
        .unwrap_err();
        assert!(err.contains("maximum safe"));
    }

    #[test]
    fn scan_finds_supported_model_snapshots() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("models--org--name/snapshots/hash");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("config.json"), br#"{"num_hidden_layers":8,"num_key_value_heads":4,"hidden_size":1024,"num_attention_heads":8}"#).unwrap();
        assert_eq!(scan_roots(&[root.path().to_path_buf()]), vec![model]);
    }
}
