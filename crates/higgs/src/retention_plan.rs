use std::path::{Path, PathBuf};

use crate::capacity::{FastSessionContractInputs, FastSessionContractV2};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct AvailableModel {
    pub id: String,
    pub path: PathBuf,
    pub model_type: String,
    pub adapter: String,
}

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

pub fn scan_models(roots: &[PathBuf]) -> Vec<AvailableModel> {
    let mut found = scan_roots(roots)
        .into_iter()
        .filter_map(|path| {
            let path = path.canonicalize().ok()?;
            if !path.join("tokenizer.json").is_file()
                || !higgs_models::collect_safetensors_files(&path)
                    .is_ok_and(|files| files.iter().all(|file| file.is_file()))
            {
                return None;
            }
            let detected = higgs_models::adapter::detect(&path).ok()?;
            let adapter = higgs_models::adapter::resolve(&detected).ok()?;
            Some(AvailableModel {
                id: model_name(&path, &detected.raw),
                path,
                model_type: detected.model_type,
                adapter: adapter.id().to_owned(),
            })
        })
        .collect::<Vec<_>>();
    found.sort_by(|left, right| left.path.cmp(&right.path));
    found.dedup_by(|left, right| left.path == right.path);
    found
}

fn model_name(path: &Path, config: &serde_json::Value) -> String {
    if let Some(name) = config
        .get("_name_or_path")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|name| name.contains('/') && !Path::new(name).is_absolute())
    {
        return name.to_owned();
    }

    for component in path
        .components()
        .filter_map(|part| part.as_os_str().to_str())
    {
        if let Some(repository) = component.strip_prefix("models--") {
            return repository.replace("--", "/");
        }
    }

    let components = path
        .components()
        .filter_map(|part| part.as_os_str().to_str())
        .collect::<Vec<_>>();
    if let Some(index) = components
        .windows(2)
        .position(|parts| parts == ["lm-studio", "models"])
    {
        return components[index + 2..].join("/");
    }

    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_owned()
}

pub fn plan_model(
    path: &Path,
    request: BudgetRequest,
    sessions: u64,
    output: u64,
) -> Result<RetentionPlan, String> {
    plan_model_with_draft(path, None, request, sessions, output)
}

pub fn plan_model_with_draft(
    path: &Path,
    draft: Option<&Path>,
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
    let cost =
        higgs_engine::EngineCostDescription::runtime_pair_from_model_dirs(path, draft, transient)
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

    fn write_catalog_model(path: &Path, config: serde_json::Value) {
        std::fs::create_dir_all(path).unwrap();
        std::fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(path.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(path.join("model.safetensors"), b"weights").unwrap();
    }

    #[test]
    fn catalog_accepts_metadata_compatible_nanbeige() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("Nanbeige4.1-3B");
        write_catalog_model(
            &model,
            serde_json::json!({
                "model_type": "llama",
                "_name_or_path": "Nanbeige/Nanbeige4.1-3B"
            }),
        );

        let available = scan_models(&[root.path().to_path_buf()]);

        assert_eq!(available.len(), 1);
        assert_eq!(available[0].id, "Nanbeige/Nanbeige4.1-3B");
        assert_eq!(available[0].path, model.canonicalize().unwrap());
        assert_eq!(available[0].model_type, "llama");
        assert_eq!(available[0].adapter, "transformer-dense");
    }

    #[test]
    fn catalog_rejects_unsupported_malformed_and_incomplete_artifacts() {
        let root = tempfile::tempdir().unwrap();
        let unsupported = root.path().join("unsupported");
        write_catalog_model(&unsupported, serde_json::json!({"model_type": "diffusion"}));
        let malformed = root.path().join("malformed");
        std::fs::create_dir_all(&malformed).unwrap();
        std::fs::write(malformed.join("config.json"), b"not json").unwrap();
        std::fs::write(malformed.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(malformed.join("model.safetensors"), b"weights").unwrap();
        let no_tokenizer = root.path().join("no-tokenizer");
        write_catalog_model(&no_tokenizer, serde_json::json!({"model_type": "llama"}));
        std::fs::remove_file(no_tokenizer.join("tokenizer.json")).unwrap();
        let no_weights = root.path().join("no-weights");
        write_catalog_model(&no_weights, serde_json::json!({"model_type": "llama"}));
        std::fs::remove_file(no_weights.join("model.safetensors")).unwrap();

        assert!(scan_models(&[root.path().to_path_buf()]).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn catalog_deduplicates_canonical_paths() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model");
        let alias = root.path().join("alias");
        write_catalog_model(&model, serde_json::json!({"model_type": "llama"}));
        symlink(&model, &alias).unwrap();

        let available = scan_models(&[model.clone(), alias]);

        assert_eq!(available.len(), 1);
        assert_eq!(available[0].path, model.canonicalize().unwrap());
    }

    #[test]
    fn catalog_names_hugging_face_snapshots_from_cache_identity() {
        let root = tempfile::tempdir().unwrap();
        let model = root
            .path()
            .join("models--LiquidAI--LFM2.5-2.6B-MLX/snapshots/deadbeef");
        write_catalog_model(&model, serde_json::json!({"model_type": "llama"}));

        let available = scan_models(&[root.path().to_path_buf()]);

        assert_eq!(available[0].id, "LiquidAI/LFM2.5-2.6B-MLX");
    }

    #[test]
    fn catalog_preserves_nested_lm_studio_variant_names() {
        let root = tempfile::tempdir().unwrap();
        let models_root = root.path().join(".cache/lm-studio/models");
        let model = models_root.join("LiquidAI/LFM2.5-2.6B-MLX/8bit");
        write_catalog_model(&model, serde_json::json!({"model_type": "llama"}));

        let available = scan_models(&[models_root]);

        assert_eq!(available[0].id, "LiquidAI/LFM2.5-2.6B-MLX/8bit");
    }

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
