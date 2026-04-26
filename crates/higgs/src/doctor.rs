use std::collections::HashSet;
use std::time::Instant;

use crate::config::HiggsConfig;
use crate::model_resolver;

pub struct DoctorResult {
    pub passes: u32,
    pub warnings: u32,
    pub failures: u32,
}

#[allow(clippy::print_stderr)]
fn pass(msg: &str, result: &mut DoctorResult) {
    eprintln!("\x1b[32m[PASS]\x1b[0m {msg}");
    result.passes += 1;
}

#[allow(clippy::print_stderr)]
fn warn(msg: &str, result: &mut DoctorResult) {
    eprintln!("\x1b[33m[WARN]\x1b[0m {msg}");
    result.warnings += 1;
}

#[allow(clippy::print_stderr)]
fn fail(msg: &str, result: &mut DoctorResult) {
    eprintln!("\x1b[31m[FAIL]\x1b[0m {msg}");
    result.failures += 1;
}

#[allow(clippy::print_stderr)]
pub async fn run_doctor(config: &HiggsConfig) -> DoctorResult {
    let mut result = DoctorResult {
        passes: 0,
        warnings: 0,
        failures: 0,
    };

    eprintln!("\x1b[1mhiggs doctor\x1b[0m\n");

    check_config_valid(&mut result);
    check_models(config, &mut result);
    check_draft_models(config, &mut result);
    check_pld(config, &mut result);
    check_duplicate_models(config, &mut result);
    check_providers(config, &mut result).await;
    check_route_consistency(config, &mut result);
    check_default_provider(config, &mut result);
    check_auto_router(config, &mut result);
    check_port_availability(config, &mut result);
    check_orphaned_providers(config, &mut result);
    check_ane_int8_mlp(&mut result);

    eprintln!(
        "\n{} passed, {} warnings, {} failures",
        result.passes, result.warnings, result.failures
    );

    result
}

fn check_config_valid(result: &mut DoctorResult) {
    // If we got this far, the config parsed and validated successfully.
    pass("config file is valid", result);
}

fn model_label(model: &crate::config::ModelConfig) -> String {
    model.name.as_ref().map_or_else(
        || model.path.clone(),
        |name| format!("\"{name}\" ({})", model.path),
    )
}

fn check_models(config: &HiggsConfig, result: &mut DoctorResult) {
    for model in &config.models {
        let label = model_label(model);
        match model_resolver::resolve(&model.path) {
            Ok(_) => pass(&format!("model {label} resolvable"), result),
            Err(err) => fail(&format!("model {label} not found: {err}"), result),
        }
        if let Some(ref dflash_path) = model.dflash {
            match model_resolver::resolve(dflash_path) {
                Ok(p) => {
                    if p.join("config.json").exists() {
                        pass(&format!("dflash drafter for {label} resolvable"), result);
                    } else {
                        fail(
                            &format!(
                                "dflash drafter for {label}: no config.json in {}",
                                p.display()
                            ),
                            result,
                        );
                    }
                }
                Err(err) => fail(
                    &format!("dflash drafter for {label} not found: {err}"),
                    result,
                ),
            }
        }
        if let Some(ref ar_spec_path) = model.ar_spec {
            match model_resolver::resolve(ar_spec_path) {
                Ok(p) => {
                    if p.join("config.json").exists() {
                        pass(&format!("ar_spec drafter for {label} resolvable"), result);
                    } else {
                        fail(
                            &format!(
                                "ar_spec drafter for {label}: no config.json in {}",
                                p.display()
                            ),
                            result,
                        );
                    }
                }
                Err(err) => fail(
                    &format!("ar_spec drafter for {label} not found: {err}"),
                    result,
                ),
            }
            if model.dflash.is_some() {
                fail(
                    &format!(
                        "{label}: both dflash and ar_spec are set — they are mutually exclusive (ar_spec wins at runtime)"
                    ),
                    result,
                );
            }
        }
        if let Ok(ref model_dir) = model_resolver::resolve(&model.path) {
            check_bd3lm_config(model_dir, &label, result);
        }
    }
}

fn check_bd3lm_config(model_dir: &std::path::Path, label: &str, result: &mut DoctorResult) {
    let config_path = model_dir.join("config.json");
    let Ok(f) = std::fs::File::open(&config_path) else {
        return;
    };
    let Ok(v) = serde_json::from_reader::<_, serde_json::Value>(f) else {
        return;
    };
    if v.get("model_type").and_then(|t| t.as_str()) != Some("bd3lm_qwen3") {
        return;
    }
    // Require bd3lm_config.json
    let bd3lm_cfg_path = model_dir.join("bd3lm_config.json");
    if !bd3lm_cfg_path.exists() {
        fail(
            &format!("model {label}: bd3lm_qwen3 requires bd3lm_config.json"),
            result,
        );
        return;
    }
    let Ok(f2) = std::fs::File::open(&bd3lm_cfg_path) else {
        fail(
            &format!("model {label}: cannot open bd3lm_config.json"),
            result,
        );
        return;
    };
    let Ok(cfg) = serde_json::from_reader::<_, serde_json::Value>(f2) else {
        fail(
            &format!("model {label}: bd3lm_config.json is not valid JSON"),
            result,
        );
        return;
    };
    let block_size = cfg.get("block_size").and_then(|v| v.as_i64()).unwrap_or(64);
    if ![16, 32, 64, 128].contains(&block_size) {
        fail(
            &format!("model {label}: bd3lm block_size={block_size} must be one of 16, 32, 64, 128"),
            result,
        );
    } else {
        pass(
            &format!("model {label}: bd3lm block_size={block_size} is valid"),
            result,
        );
    }
    let num_steps = cfg
        .get("num_denoising_steps")
        .and_then(|v| v.as_i64())
        .unwrap_or(8);
    if num_steps <= 0 || block_size % num_steps != 0 {
        fail(
            &format!(
                "model {label}: bd3lm num_denoising_steps={num_steps} must be >0 and divide block_size={block_size}"
            ),
            result,
        );
    } else {
        pass(
            &format!("model {label}: bd3lm num_denoising_steps={num_steps} valid"),
            result,
        );
    }
    // Require bd3lm_extras.safetensors
    if !model_dir.join("bd3lm_extras.safetensors").exists() {
        fail(
            &format!("model {label}: bd3lm_qwen3 requires bd3lm_extras.safetensors"),
            result,
        );
    } else {
        pass(
            &format!("model {label}: bd3lm_extras.safetensors present"),
            result,
        );
    }
}

fn check_draft_models(config: &HiggsConfig, result: &mut DoctorResult) {
    for model in &config.models {
        let Some(ref draft_path) = model.draft_model else {
            continue;
        };
        let label = model_label(model);
        match model_resolver::resolve(draft_path) {
            Ok(_) => pass(&format!("draft model for {label} resolvable"), result),
            Err(err) => fail(
                &format!("draft model \"{draft_path}\" for {label} not found: {err}"),
                result,
            ),
        }
        if model.batch {
            warn(
                &format!(
                    "{label} has draft_model but batch=true; speculative decoding is only supported with SimpleEngine"
                ),
                result,
            );
        }
    }
}

fn check_pld(config: &HiggsConfig, result: &mut DoctorResult) {
    for model in &config.models {
        if !model.pld {
            continue;
        }
        let label = model_label(model);
        let mut conflicts = Vec::new();
        if model.draft_model.is_some() {
            conflicts.push("draft_model");
        }
        if model.dflash.is_some() {
            conflicts.push("dflash");
        }
        if model.ar_spec.is_some() {
            conflicts.push("ar_spec");
        }
        if !conflicts.is_empty() {
            fail(
                &format!(
                    "{label}: pld=true conflicts with {} — choose one speculative path",
                    conflicts.join(", ")
                ),
                result,
            );
            continue;
        }
        if model.batch {
            warn(
                &format!("{label}: pld=true with batch=true; PLD only runs in SimpleEngine"),
                result,
            );
        }
        if model.pld_min_ngram < 1 {
            fail(
                &format!(
                    "{label}: pld_min_ngram={} must be >= 1",
                    model.pld_min_ngram
                ),
                result,
            );
            continue;
        }
        if model.pld_max_ngram < model.pld_min_ngram {
            fail(
                &format!(
                    "{label}: pld_max_ngram={} must be >= pld_min_ngram={}",
                    model.pld_max_ngram, model.pld_min_ngram
                ),
                result,
            );
            continue;
        }
        if model.num_draft == 0 {
            fail(
                &format!("{label}: pld=true requires num_draft >= 1 (got 0)"),
                result,
            );
            continue;
        }
        pass(
            &format!(
                "PLD enabled for {label} (n-gram {}..={}, num_draft={})",
                model.pld_min_ngram, model.pld_max_ngram, model.num_draft
            ),
            result,
        );
    }
}

fn check_duplicate_models(config: &HiggsConfig, result: &mut DoctorResult) {
    let mut seen_paths = HashSet::new();
    let mut seen_names = HashSet::new();
    let mut duplicates = Vec::new();
    for model in &config.models {
        if !seen_paths.insert(&model.path) {
            duplicates.push(format!("path: {}", model.path));
        }
        if let Some(ref name) = model.name {
            if !seen_names.insert(name) {
                duplicates.push(format!("name: {name}"));
            }
        }
    }
    if duplicates.is_empty() {
        if config.models.len() > 1 {
            pass("no duplicate model paths or names", result);
        }
    } else {
        for dup in &duplicates {
            warn(&format!("duplicate model {dup}"), result);
        }
    }
}

async fn check_providers(config: &HiggsConfig, result: &mut DoctorResult) {
    let http_client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(err) => {
            warn(&format!("could not create HTTP client: {err}"), result);
            return;
        }
    };

    for (name, provider) in &config.providers {
        let start = Instant::now();
        match http_client.head(&provider.url).send().await {
            Ok(response) => {
                let elapsed = start.elapsed();
                pass(
                    &format!(
                        "provider {name} reachable ({} {}ms)",
                        response.status(),
                        elapsed.as_millis()
                    ),
                    result,
                );
            }
            Err(err) => {
                warn(&format!("provider {name} unreachable: {err}"), result);
            }
        }
    }
}

fn check_route_consistency(config: &HiggsConfig, result: &mut DoctorResult) {
    let mut all_valid = true;
    for route in &config.routes {
        if route.provider == "higgs" {
            if config.models.is_empty() {
                warn(
                    &format!(
                        "route {:?} targets \"higgs\" but no models are loaded",
                        route
                            .name
                            .as_deref()
                            .or(route.pattern.as_deref())
                            .unwrap_or("(unnamed)")
                    ),
                    result,
                );
                all_valid = false;
            }
        } else if !config.providers.contains_key(&route.provider) {
            fail(
                &format!(
                    "route {:?} references unknown provider \"{}\"",
                    route
                        .name
                        .as_deref()
                        .or(route.pattern.as_deref())
                        .unwrap_or("(unnamed)"),
                    route.provider
                ),
                result,
            );
            all_valid = false;
        }
    }
    if all_valid && !config.routes.is_empty() {
        pass("all route providers exist", result);
    }
}

fn check_default_provider(config: &HiggsConfig, result: &mut DoctorResult) {
    let provider = &config.default.provider;
    if provider == "higgs" {
        if config.models.is_empty() {
            warn(
                "default provider is \"higgs\" but no models are loaded",
                result,
            );
        } else {
            pass(&format!("default provider \"{provider}\" exists"), result);
        }
    } else if config.providers.contains_key(provider) {
        pass(&format!("default provider \"{provider}\" exists"), result);
    } else {
        fail(
            &format!("default provider \"{provider}\" not found in providers"),
            result,
        );
    }
}

fn check_auto_router(config: &HiggsConfig, result: &mut DoctorResult) {
    if !config.auto_router.enabled {
        return;
    }

    let model_ref = &config.auto_router.model;
    if model_ref.is_empty() {
        fail("auto_router enabled but no model specified", result);
        return;
    }

    // Match by name or path
    let matched = config
        .models
        .iter()
        .find(|m| m.path == *model_ref || m.name.as_deref() == Some(model_ref));

    if let Some(matched_model) = matched {
        let label = model_label(matched_model);
        pass(
            &format!("auto_router model {label} found in models"),
            result,
        );
        match model_resolver::resolve(&matched_model.path) {
            Ok(_) => pass(&format!("auto_router model {label} downloaded"), result),
            Err(err) => fail(
                &format!("auto_router model {label} not downloaded: {err}"),
                result,
            ),
        }
    } else {
        fail(
            &format!("auto_router model \"{model_ref}\" not found in models"),
            result,
        );
    }

    let routes_with_descriptions = config
        .routes
        .iter()
        .filter(|r| r.description.is_some())
        .count();
    if routes_with_descriptions == 0 && !config.routes.is_empty() {
        warn(
            "auto_router enabled but no routes have descriptions",
            result,
        );
    }
}

fn check_port_availability(config: &HiggsConfig, result: &mut DoctorResult) {
    let addr = format!("{}:{}", config.server.host, config.server.port);
    match std::net::TcpListener::bind(&addr) {
        Ok(_) => pass(&format!("port {} available", config.server.port), result),
        Err(err) => warn(
            &format!("port {} unavailable: {err}", config.server.port),
            result,
        ),
    }
}

/// Validate env flags that gate the experimental ANE int8 MLP layer-0 prefill path.
///
/// - `HIGGS_TARGET_ANE_INT8_MLP=1` enables compiling MLP layer 0 as int8 mlpackage
///   kernels and dispatching prefill through the ANE. Requires the `ane` feature.
/// - `HIGGS_ANE_INT8_MLP_SEQ=<int>` sets the seq bucket (default 128). Runtime seqs
///   outside `(1, bucket]` fall back to the MLX q4 path.
fn check_ane_int8_mlp(result: &mut DoctorResult) {
    let target = std::env::var("HIGGS_TARGET_ANE_INT8_MLP").ok();
    let seq = std::env::var("HIGGS_ANE_INT8_MLP_SEQ").ok();

    let target_enabled = target.as_deref() == Some("1");

    if target_enabled {
        #[cfg(feature = "ane")]
        pass("HIGGS_TARGET_ANE_INT8_MLP=1 (ANE feature enabled)", result);
        #[cfg(not(feature = "ane"))]
        warn(
            "HIGGS_TARGET_ANE_INT8_MLP=1 set but binary built without the `ane` feature \
             — flag will be a no-op",
            result,
        );
    }

    if let Some(ref raw) = seq {
        if !target_enabled {
            warn(
                &format!(
                    "HIGGS_ANE_INT8_MLP_SEQ={raw} set but HIGGS_TARGET_ANE_INT8_MLP != 1 \
                     — flag will be a no-op"
                ),
                result,
            );
        }
        match raw.parse::<i32>() {
            Ok(n) if n > 1 => {
                if target_enabled {
                    pass(
                        &format!("HIGGS_ANE_INT8_MLP_SEQ={n} (valid bucket)"),
                        result,
                    );
                }
            }
            Ok(n) => warn(
                &format!("HIGGS_ANE_INT8_MLP_SEQ={n} is not > 1 (must be a prefill bucket)"),
                result,
            ),
            Err(_) => warn(
                &format!("HIGGS_ANE_INT8_MLP_SEQ={raw} is not a valid integer"),
                result,
            ),
        }
    }
}

fn check_orphaned_providers(config: &HiggsConfig, result: &mut DoctorResult) {
    let mut referenced: HashSet<&str> = HashSet::new();

    if config.default.provider != "higgs" {
        referenced.insert(&config.default.provider);
    }

    for route in &config.routes {
        if route.provider != "higgs" {
            referenced.insert(&route.provider);
        }
    }

    for name in config.providers.keys() {
        if !referenced.contains(name.as_str()) {
            warn(
                &format!("provider \"{name}\" defined but not used by any route"),
                result,
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::config::{
        AutoRouterConfig, DefaultRoute, HiggsConfig, ModelConfig, ProviderConfig, RouteConfig,
        ServerSection,
    };
    use std::collections::HashMap;

    fn empty_result() -> DoctorResult {
        DoctorResult {
            passes: 0,
            warnings: 0,
            failures: 0,
        }
    }

    // -- Helper function counter tests --

    #[test]
    fn test_pass_increments_counter() {
        let mut result = empty_result();
        pass("test", &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_warn_increments_counter() {
        let mut result = empty_result();
        warn("test", &mut result);
        assert_eq!(result.passes, 0);
        assert_eq!(result.warnings, 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_fail_increments_counter() {
        let mut result = empty_result();
        fail("test", &mut result);
        assert_eq!(result.passes, 0);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 1);
    }

    // -- Duplicate model detection --

    #[test]
    fn test_no_duplicates_passes() {
        let config = HiggsConfig {
            models: vec![
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    name: None,
                    batch: false,
                    draft_model: None,
                    num_draft: 8,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    dflash: None,
                    ar_spec: None,
                    bd3lm: None,
                    pld: false,
                    pld_max_ngram: 3,
                    pld_min_ngram: 1,
                },
                ModelConfig {
                    path: "org/model-b".to_owned(),
                    name: None,
                    batch: false,
                    draft_model: None,
                    num_draft: 8,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    dflash: None,
                    ar_spec: None,
                    bd3lm: None,
                    pld: false,
                    pld_max_ngram: 3,
                    pld_min_ngram: 1,
                },
            ],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_duplicate_models(&config, &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_duplicate_models_warns() {
        let config = HiggsConfig {
            models: vec![
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    name: None,
                    batch: false,
                    draft_model: None,
                    num_draft: 8,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    dflash: None,
                    ar_spec: None,
                    bd3lm: None,
                    pld: false,
                    pld_max_ngram: 3,
                    pld_min_ngram: 1,
                },
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    name: None,
                    batch: false,
                    draft_model: None,
                    num_draft: 8,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    dflash: None,
                    ar_spec: None,
                    bd3lm: None,
                    pld: false,
                    pld_max_ngram: 3,
                    pld_min_ngram: 1,
                },
            ],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_duplicate_models(&config, &mut result);
        assert_eq!(result.warnings, 1);
    }

    // -- Orphaned provider detection --

    #[test]
    fn test_orphaned_provider_warns() {
        let mut providers = HashMap::new();
        providers.insert(
            "openai".to_owned(),
            ProviderConfig {
                url: "https://api.openai.com".to_owned(),
                format: crate::config::ApiFormat::OpenAi,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let config = HiggsConfig {
            providers,
            default: DefaultRoute {
                provider: "higgs".to_owned(),
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_orphaned_providers(&config, &mut result);
        assert_eq!(result.warnings, 1);
    }

    #[test]
    fn test_referenced_provider_not_orphaned() {
        let mut providers = HashMap::new();
        providers.insert(
            "anthropic".to_owned(),
            ProviderConfig {
                url: "https://api.anthropic.com".to_owned(),
                format: crate::config::ApiFormat::Anthropic,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let config = HiggsConfig {
            providers,
            default: DefaultRoute {
                provider: "anthropic".to_owned(),
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_orphaned_providers(&config, &mut result);
        assert_eq!(result.warnings, 0);
    }

    // -- Route consistency --

    #[test]
    fn test_route_unknown_provider_fails() {
        let config = HiggsConfig {
            routes: vec![RouteConfig {
                name: Some("test".to_owned()),
                description: None,
                pattern: None,
                provider: "nonexistent".to_owned(),
                model: None,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_route_consistency(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_route_higgs_no_models_warns() {
        let config = HiggsConfig {
            routes: vec![RouteConfig {
                name: Some("local".to_owned()),
                description: None,
                pattern: None,
                provider: "higgs".to_owned(),
                model: None,
            }],
            models: vec![],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_route_consistency(&config, &mut result);
        assert_eq!(result.warnings, 1);
    }

    #[test]
    fn test_route_valid_provider_passes() {
        let mut providers = HashMap::new();
        providers.insert(
            "anthropic".to_owned(),
            ProviderConfig {
                url: "https://api.anthropic.com".to_owned(),
                format: crate::config::ApiFormat::Anthropic,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let config = HiggsConfig {
            providers,
            routes: vec![RouteConfig {
                name: Some("claude".to_owned()),
                description: None,
                pattern: Some("claude-.*".to_owned()),
                provider: "anthropic".to_owned(),
                model: None,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_route_consistency(&config, &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.failures, 0);
    }

    // -- Draft model validation --

    #[test]
    fn test_draft_model_not_found_fails() {
        let config = HiggsConfig {
            models: vec![ModelConfig {
                path: "org/target-model".to_owned(),
                name: None,
                batch: false,
                draft_model: Some("org/nonexistent-draft".to_owned()),
                num_draft: 8,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                dflash: None,
                ar_spec: None,
                bd3lm: None,
                pld: false,
                pld_max_ngram: 3,
                pld_min_ngram: 1,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_draft_models(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_draft_model_with_batch_warns() {
        let config = HiggsConfig {
            models: vec![ModelConfig {
                path: "org/target-model".to_owned(),
                name: None,
                batch: true,
                draft_model: Some("org/some-draft".to_owned()),
                num_draft: 8,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                dflash: None,
                ar_spec: None,
                bd3lm: None,
                pld: false,
                pld_max_ngram: 3,
                pld_min_ngram: 1,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_draft_models(&config, &mut result);
        // Fails for unresolvable path + warns for batch incompatibility
        assert!(result.failures >= 1);
        assert_eq!(result.warnings, 1);
    }

    #[test]
    fn test_no_draft_model_skips() {
        let config = HiggsConfig {
            models: vec![ModelConfig {
                path: "org/model".to_owned(),
                name: None,
                batch: false,
                draft_model: None,
                num_draft: 8,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                dflash: None,
                ar_spec: None,
                bd3lm: None,
                pld: false,
                pld_max_ngram: 3,
                pld_min_ngram: 1,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_draft_models(&config, &mut result);
        assert_eq!(result.passes, 0);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    // -- Default provider --

    #[test]
    fn test_default_provider_exists() {
        let mut providers = HashMap::new();
        providers.insert(
            "anthropic".to_owned(),
            ProviderConfig {
                url: "https://api.anthropic.com".to_owned(),
                format: crate::config::ApiFormat::Anthropic,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let config = HiggsConfig {
            providers,
            default: DefaultRoute {
                provider: "anthropic".to_owned(),
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_default_provider(&config, &mut result);
        assert_eq!(result.passes, 1);
    }

    #[test]
    fn test_default_provider_missing_fails() {
        let config = HiggsConfig {
            default: DefaultRoute {
                provider: "nonexistent".to_owned(),
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_default_provider(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_default_higgs_no_models_warns() {
        let config = HiggsConfig {
            default: DefaultRoute {
                provider: "higgs".to_owned(),
            },
            models: vec![],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_default_provider(&config, &mut result);
        assert_eq!(result.warnings, 1);
    }

    // -- Port availability --

    #[test]
    fn test_port_zero_available() {
        let config = HiggsConfig {
            server: ServerSection {
                host: "127.0.0.1".to_owned(),
                port: 0,
                ..ServerSection::default()
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_port_availability(&config, &mut result);
        assert_eq!(result.passes, 1);
    }

    // -- Auto router --

    #[test]
    fn test_auto_router_disabled_skips() {
        let config = HiggsConfig {
            auto_router: AutoRouterConfig {
                enabled: false,
                force: false,
                model: String::new(),
                timeout_ms: 2000,
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        assert_eq!(result.passes, 0);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_auto_router_empty_model_fails() {
        let config = HiggsConfig {
            auto_router: AutoRouterConfig {
                enabled: true,
                force: false,
                model: String::new(),
                timeout_ms: 2000,
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_auto_router_unknown_model_fails() {
        let config = HiggsConfig {
            auto_router: AutoRouterConfig {
                enabled: true,
                force: false,
                model: "nonexistent/model".to_owned(),
                timeout_ms: 2000,
            },
            models: vec![ModelConfig {
                path: "org/other-model".to_owned(),
                name: None,
                batch: false,
                draft_model: None,
                num_draft: 8,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                dflash: None,
                ar_spec: None,
                bd3lm: None,
                pld: false,
                pld_max_ngram: 3,
                pld_min_ngram: 1,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        // Fails once: not in [[models]] (download check skipped)
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_auto_router_model_not_downloaded_fails() {
        let config = HiggsConfig {
            auto_router: AutoRouterConfig {
                enabled: true,
                force: false,
                model: "org/router-model".to_owned(),
                timeout_ms: 2000,
            },
            models: vec![ModelConfig {
                path: "org/router-model".to_owned(),
                name: None,
                batch: false,
                draft_model: None,
                num_draft: 8,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                dflash: None,
                ar_spec: None,
                bd3lm: None,
                pld: false,
                pld_max_ngram: 3,
                pld_min_ngram: 1,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        // Model is in [[models]] (pass), but not downloaded (fail)
        assert_eq!(result.passes, 1);
        assert_eq!(result.failures, 1);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_auto_router_no_descriptions_warns() {
        let config = HiggsConfig {
            auto_router: AutoRouterConfig {
                enabled: true,
                force: false,
                model: "org/router-model".to_owned(),
                timeout_ms: 2000,
            },
            models: vec![ModelConfig {
                path: "org/router-model".to_owned(),
                name: None,
                batch: false,
                draft_model: None,
                num_draft: 8,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                dflash: None,
                ar_spec: None,
                bd3lm: None,
                pld: false,
                pld_max_ngram: 3,
                pld_min_ngram: 1,
            }],
            routes: vec![RouteConfig {
                name: Some("test".to_owned()),
                description: None,
                pattern: None,
                provider: "higgs".to_owned(),
                model: None,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        // Should pass for model found, but warn for no descriptions
        assert_eq!(result.passes, 1);
        assert_eq!(result.warnings, 1);
    }

    // -- Provider reachability --

    #[tokio::test]
    async fn test_unreachable_provider_warns() {
        let mut providers = HashMap::new();
        providers.insert(
            "bad".to_owned(),
            ProviderConfig {
                url: "http://127.0.0.1:1".to_owned(),
                format: crate::config::ApiFormat::OpenAi,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let config = HiggsConfig {
            providers,
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_providers(&config, &mut result).await;
        assert_eq!(result.warnings, 1);
        assert_eq!(result.passes, 0);
    }

    // -- DFlash drafter validation --

    #[test]
    fn test_dflash_resolve_with_config_json_passes() {
        let model_dir = tempfile::tempdir().unwrap();
        let dflash_dir = tempfile::tempdir().unwrap();
        std::fs::write(dflash_dir.path().join("config.json"), "{}").unwrap();

        let config = HiggsConfig {
            models: vec![ModelConfig {
                path: model_dir.path().to_str().unwrap().to_owned(),
                name: None,
                batch: false,
                draft_model: None,
                num_draft: 8,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                dflash: Some(dflash_dir.path().to_str().unwrap().to_owned()),
                ar_spec: None,
                bd3lm: None,
                pld: false,
                pld_max_ngram: 3,
                pld_min_ngram: 1,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_models(&config, &mut result);
        assert_eq!(result.passes, 2);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_dflash_resolve_without_config_json_fails() {
        let model_dir = tempfile::tempdir().unwrap();
        let dflash_dir = tempfile::tempdir().unwrap();
        // dflash_dir exists but has no config.json

        let config = HiggsConfig {
            models: vec![ModelConfig {
                path: model_dir.path().to_str().unwrap().to_owned(),
                name: None,
                batch: false,
                draft_model: None,
                num_draft: 8,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                dflash: Some(dflash_dir.path().to_str().unwrap().to_owned()),
                ar_spec: None,
                bd3lm: None,
                pld: false,
                pld_max_ngram: 3,
                pld_min_ngram: 1,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_models(&config, &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_dflash_resolve_path_not_found_fails() {
        let model_dir = tempfile::tempdir().unwrap();

        let config = HiggsConfig {
            models: vec![ModelConfig {
                path: model_dir.path().to_str().unwrap().to_owned(),
                name: None,
                batch: false,
                draft_model: None,
                num_draft: 8,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                dflash: Some("/nonexistent/dflash/drafter".to_owned()),
                ar_spec: None,
                bd3lm: None,
                pld: false,
                pld_max_ngram: 3,
                pld_min_ngram: 1,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_models(&config, &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_dflash_none_no_extra_checks() {
        let model_dir = tempfile::tempdir().unwrap();

        let config = HiggsConfig {
            models: vec![ModelConfig {
                path: model_dir.path().to_str().unwrap().to_owned(),
                name: None,
                batch: false,
                draft_model: None,
                num_draft: 8,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                dflash: None,
                ar_spec: None,
                bd3lm: None,
                pld: false,
                pld_max_ngram: 3,
                pld_min_ngram: 1,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_models(&config, &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_check_bd3lm_config_valid() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "bd3lm_qwen3"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("bd3lm_config.json"),
            r#"{"block_size": 64, "num_denoising_steps": 8, "denoise_hidden": 4096}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("bd3lm_extras.safetensors"), b"").unwrap();
        let mut result = empty_result();
        check_bd3lm_config(dir.path(), "test-model", &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.passes, 3);
    }

    fn pld_model(path: &str) -> ModelConfig {
        ModelConfig {
            path: path.to_owned(),
            name: None,
            batch: false,
            draft_model: None,
            num_draft: 8,
            kv_cache: higgs_models::turboquant::KvCacheMode::Off,
            kv_bits: 3,
            kv_seed: 0,
            dflash: None,
            ar_spec: None,
            bd3lm: None,
            pld: true,
            pld_max_ngram: 3,
            pld_min_ngram: 1,
        }
    }

    #[test]
    fn test_pld_disabled_skips() {
        let mut model = pld_model("org/m");
        model.pld = false;
        let config = HiggsConfig {
            models: vec![model],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_pld(&config, &mut result);
        assert_eq!(result.passes, 0);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_pld_enabled_passes() {
        let config = HiggsConfig {
            models: vec![pld_model("org/m")],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_pld(&config, &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_pld_conflicts_with_draft_model() {
        let mut model = pld_model("org/m");
        model.draft_model = Some("org/d".to_owned());
        let config = HiggsConfig {
            models: vec![model],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_pld(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_pld_conflicts_with_dflash() {
        let mut model = pld_model("org/m");
        model.dflash = Some("org/d".to_owned());
        let config = HiggsConfig {
            models: vec![model],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_pld(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_pld_conflicts_with_ar_spec() {
        let mut model = pld_model("org/m");
        model.ar_spec = Some("org/a".to_owned());
        let config = HiggsConfig {
            models: vec![model],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_pld(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_pld_min_ngram_zero_fails() {
        let mut model = pld_model("org/m");
        model.pld_min_ngram = 0;
        let config = HiggsConfig {
            models: vec![model],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_pld(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_pld_max_lt_min_fails() {
        let mut model = pld_model("org/m");
        model.pld_min_ngram = 4;
        model.pld_max_ngram = 2;
        let config = HiggsConfig {
            models: vec![model],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_pld(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_pld_zero_num_draft_fails() {
        let mut model = pld_model("org/m");
        model.num_draft = 0;
        let config = HiggsConfig {
            models: vec![model],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_pld(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_pld_with_batch_warns() {
        let mut model = pld_model("org/m");
        model.batch = true;
        let config = HiggsConfig {
            models: vec![model],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_pld(&config, &mut result);
        assert_eq!(result.warnings, 1);
        assert_eq!(result.passes, 1);
    }

    #[test]
    fn test_check_bd3lm_config_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "bd3lm_qwen3"}"#,
        )
        .unwrap();
        // no bd3lm_config.json, no extras
        let mut result = empty_result();
        check_bd3lm_config(dir.path(), "test-model", &mut result);
        assert_eq!(result.failures, 1); // missing bd3lm_config.json
    }
}
