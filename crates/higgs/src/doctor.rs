use std::collections::HashSet;
use std::time::Instant;

use crate::config::HiggsConfig;
use crate::model_resolver;
use higgs_engine::mlx_tuning::resolve_effective_mlx_profile;

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
pub async fn run_doctor(
    config: &HiggsConfig,
    config_path: Option<&std::path::Path>,
) -> DoctorResult {
    let mut result = DoctorResult {
        passes: 0,
        warnings: 0,
        failures: 0,
    };

    eprintln!("\x1b[1mhiggs doctor\x1b[0m\n");

    check_config_valid(&mut result);
    check_config_file_permissions(config, config_path, &mut result);
    check_misplaced_local_keys(config_path, &mut result);
    check_server_section(config, &mut result);
    check_models(config, &mut result);
    check_duplicate_models(config, &mut result);
    check_providers(config, &mut result).await;
    check_route_consistency(config, &mut result);
    check_default_provider(config, &mut result);
    check_auto_router(config, &mut result);
    check_runtime_model_load(config, &mut result);
    check_port_availability(config, &mut result);
    check_orphaned_providers(config, &mut result);

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

#[allow(clippy::too_many_lines)]
fn check_server_section(config: &crate::config::HiggsConfig, result: &mut DoctorResult) {
    let server = &config.server;

    if server.max_tokens == 0 {
        fail(
            "server.max_tokens=0 produces empty completions; set a positive value",
            result,
        );
    } else {
        pass(&format!("server.max_tokens={}", server.max_tokens), result);
    }

    if !server.timeout.is_finite() || server.timeout <= 0.0 {
        fail(
            &format!(
                "server.timeout={} must be a positive finite number of seconds",
                server.timeout
            ),
            result,
        );
    } else if server.timeout > 600.0 {
        warn(
            &format!(
                "server.timeout={}s is unusually high (>10 min); check intent",
                server.timeout
            ),
            result,
        );
    } else {
        pass(&format!("server.timeout={}s", server.timeout), result);
    }

    if server.max_body_size == 0 {
        fail("server.max_body_size=0 rejects all request bodies", result);
    } else if server.max_body_size > 1 << 30 {
        warn(
            &format!(
                "server.max_body_size={} bytes (>1 GiB); check intent",
                server.max_body_size
            ),
            result,
        );
    } else {
        pass(
            &format!("server.max_body_size={} bytes", server.max_body_size),
            result,
        );
    }

    if server.max_image_bytes > server.max_body_size {
        warn(
            "server.max_image_bytes > server.max_body_size: images can never arrive within the body cap",
            result,
        );
    } else {
        pass(
            &format!("server.max_image_bytes={} bytes", server.max_image_bytes),
            result,
        );
    }

    if (64..=16384).contains(&server.max_image_dimension) {
        pass(
            &format!("server.max_image_dimension={}", server.max_image_dimension),
            result,
        );
    } else {
        fail(
            "server.max_image_dimension must be within 64..=16384",
            result,
        );
    }

    if !server.image_fetch_timeout.is_finite() || server.image_fetch_timeout <= 0.0 {
        fail(
            "server.image_fetch_timeout must be a positive finite number of seconds",
            result,
        );
    } else {
        pass(
            &format!("server.image_fetch_timeout={}s", server.image_fetch_timeout),
            result,
        );
    }

    if server.host.parse::<std::net::IpAddr>().is_ok() || server.host == "localhost" {
        pass(&format!("server.host=\"{}\"", server.host), result);
    } else {
        warn(
            &format!(
                "server.host=\"{}\" is not an IP address or \"localhost\"; bind may fail at runtime",
                server.host
            ),
            result,
        );
    }

    if server.api_key.is_some() {
        pass("server.api_key set; API key auth enabled", result);
    } else {
        pass(
            "server.api_key unset; no auth enforced (server is open)",
            result,
        );
    }

    let non_loopback = matches!(
        server.host.parse::<std::net::IpAddr>(),
        Ok(ip) if !ip.is_loopback()
    );
    if non_loopback && server.api_key.is_none() {
        warn(
            &format!(
                "server.host=\"{}\" is reachable from the network but server.api_key is unset; \
                 anyone on the network can use this server",
                server.host
            ),
            result,
        );
    }

    check_cors_origins(server, non_loopback, result);

    if server.rate_limit == 0 {
        pass("server.rate_limit=0 (disabled)", result);
    } else {
        pass(
            &format!("server.rate_limit={} req/min/client", server.rate_limit),
            result,
        );
    }
}

fn check_cors_origins(
    server: &crate::config::ServerSection,
    non_loopback: bool,
    result: &mut DoctorResult,
) {
    match &server.cors_origins {
        None => pass("server.cors_origins unset; no CORS headers sent", result),
        Some(origins) if origins.iter().any(|o| o == "*") => {
            if non_loopback {
                warn(
                    "server.cors_origins allows any origin (\"*\") on a network-reachable host; \
                     consider an explicit origin list",
                    result,
                );
            } else {
                pass("server.cors_origins=[\"*\"] (permissive)", result);
            }
        }
        Some(origins) => {
            let mut all_valid = true;
            for origin in origins {
                let parses = origin.parse::<http::HeaderValue>().is_ok();
                if !parses || !(origin.starts_with("http://") || origin.starts_with("https://")) {
                    fail(
                        &format!(
                            "server.cors_origins entry \"{origin}\" is not a valid origin \
                             (expected e.g. \"https://example.com\")"
                        ),
                        result,
                    );
                    all_valid = false;
                }
            }
            if all_valid {
                pass(
                    &format!("server.cors_origins lists {} origin(s)", origins.len()),
                    result,
                );
            }
        }
    }
}

/// Warn when the config file holding API keys is readable by other users.
#[cfg(unix)]
fn check_config_file_permissions(
    config: &HiggsConfig,
    config_path: Option<&std::path::Path>,
    result: &mut DoctorResult,
) {
    use std::os::unix::fs::PermissionsExt as _;

    let Some(path) = config_path else { return };
    let Ok(metadata) = std::fs::metadata(path) else {
        return;
    };
    let mode = metadata.permissions().mode() & 0o777;
    let has_secrets =
        config.server.api_key.is_some() || config.providers.values().any(|p| p.api_key.is_some());
    if mode.trailing_zeros() >= 6 {
        pass(
            &format!("config file permissions are owner-only ({mode:03o})"),
            result,
        );
    } else if has_secrets {
        warn(
            &format!(
                "config file {} is group/world-accessible (mode {mode:03o}) and contains API \
                 keys; run: chmod 600 {}",
                path.display(),
                path.display()
            ),
            result,
        );
    } else {
        pass(
            &format!("config file permissions {mode:03o} (no API keys present)"),
            result,
        );
    }
}

#[cfg(not(unix))]
fn check_config_file_permissions(
    _config: &HiggsConfig,
    _config_path: Option<&std::path::Path>,
    _result: &mut DoctorResult,
) {
}

fn model_label(model: &crate::config::ModelConfig) -> String {
    model.name.as_ref().map_or_else(
        || model.path.clone(),
        |name| format!("\"{name}\" ({})", model.path),
    )
}

/// Adapter-level inspection of a resolved model directory.
///
/// The merged `higgs-engine::model_loader::ModelConfig` only exposes
/// `model_dir`/`model_type`; capability, adapter, and version metadata lives on
/// the merged adapter registry (`higgs-models::adapter`), so the doctor resolves
/// it here.
struct InspectedAdapter {
    capabilities: higgs_models::adapter::Capabilities,
    adapter_id: &'static str,
    family: String,
    version: Option<higgs_models::adapter::ModelVersion>,
}

fn inspect_adapter(resolved: &std::path::Path) -> Result<InspectedAdapter, String> {
    let detected = higgs_models::adapter::detect(resolved)
        .map_err(|e| format!("architecture detection failed: {e}"))?;
    let adapter = higgs_models::adapter::resolve(&detected)
        .map_err(|e| format!("adapter resolution failed: {e}"))?;
    let info = adapter.describe();
    Ok(InspectedAdapter {
        capabilities: info.capabilities,
        adapter_id: info.id,
        family: info.family.to_string(),
        version: detected.version,
    })
}

/// Whether a checkpoint declares vision capability from its own `config.json`:
/// a `vision_config` key or a `*_vl` model type (wrapper or effective).
///
/// This is the checkpoint-side signal; it is independent of whether the
/// resolved adapter implements vision ([`higgs_models::adapter::Capabilities`]'s
/// `vision`).
fn checkpoint_declares_vision(detected: &higgs_models::adapter::DetectedModel) -> bool {
    detected.raw.get("vision_config").is_some()
        || detected.model_type.contains("_vl")
        || detected
            .wrapper_model_type
            .as_deref()
            .is_some_and(|model_type| model_type.contains("_vl"))
}

/// The vision column for the capability report.
///
/// The report is checkpoint-driven so a text-only `gemma3_text` / `gemma4_text`
/// checkpoint never claims vision even though it shares an adapter with the
/// multimodal `gemma3` / `gemma4` checkpoints:
/// `supported` when the resolved adapter implements vision **and** the
/// checkpoint declares vision weights, `tower-ignored` when the checkpoint
/// declares vision weights that the resolved text-only adapter skips,
/// `disabled` when the config's `disable_vision` escape hatch forces a
/// vision-capable checkpoint to load text-only, and `none` otherwise. The
/// parenthetical names the checkpoint's declared model type (wrapper when
/// present), matching the family names a loaded model would report.
fn vision_status(
    inspected: &InspectedAdapter,
    detected: Option<&higgs_models::adapter::DetectedModel>,
    disable_vision: bool,
) -> String {
    let Some(detected_config) = detected else {
        return "vision: none".to_owned();
    };
    let family = detected_config
        .wrapper_model_type
        .as_deref()
        .unwrap_or(detected_config.model_type.as_str());
    if disable_vision && checkpoint_declares_vision(detected_config) {
        format!("vision: disabled (escape hatch; {family})")
    } else if inspected.capabilities.vision && checkpoint_declares_vision(detected_config) {
        format!("vision: supported ({family})")
    } else if checkpoint_declares_vision(detected_config) {
        format!("vision: tower-ignored ({family})")
    } else {
        "vision: none".to_owned()
    }
}

fn check_prefill_yield_tokens(
    label: &str,
    prefill_yield_tokens: Option<u32>,
    result: &mut DoctorResult,
) -> bool {
    let Some(tokens) = prefill_yield_tokens else {
        return true;
    };
    if tokens != 0 && tokens < 128 {
        fail(
            &format!("model {label} prefill_yield_tokens={tokens} must be 0 or at least 128"),
            result,
        );
        return false;
    }
    if tokens != 0 && tokens < 512 {
        warn(
            &format!("model {label} prefill_yield_tokens={tokens} is below the recommended 512"),
            result,
        );
    }
    true
}

/// Warn (not fail) when the *resolved* MLA decision is enabled for an adapter
/// that does not advertise MLA latent-cache support. The flag is a no-op for
/// those adapters at runtime, so this is advisory rather than a hard failure.
///
/// Uses [`higgs_models::cache::resolve_mla_latent_cache`] rather than the raw
/// `model.mla_latent_cache` field, so this matches runtime behavior: e.g.
/// `HIGGS_MLA_LATENT_CACHE=1` with `mla_latent_cache` unset in config still
/// warns (the flag is effectively on), and `HIGGS_MLA_LATENT_CACHE=0` with
/// `mla_latent_cache=true` in config does not warn (the flag is effectively
/// off).
#[cfg(test)]
fn check_mla_latent_cache_architecture(
    label: &str,
    model: &crate::config::ModelConfig,
    resolved: &std::path::Path,
    result: &mut DoctorResult,
) {
    if !higgs_models::cache::resolve_mla_latent_cache(model.kv_cache_config().mla_latent) {
        return;
    }
    match inspect_adapter(resolved) {
        Ok(inspected) => check_mla_latent_cache_adapter(label, &inspected, result),
        Err(err) => {
            warn(
                &format!(
                    "model {label} enables mla_latent_cache=true but its architecture could not be determined: {err}"
                ),
                result,
            );
        }
    }
}

fn check_mla_latent_cache_adapter(
    label: &str,
    inspected: &InspectedAdapter,
    result: &mut DoctorResult,
) {
    if inspected.capabilities.mla_latent_cache {
        pass(
            &format!(
                "model {label} mla_latent_cache=true (adapter {})",
                inspected.adapter_id
            ),
            result,
        );
    } else {
        warn(
            &format!(
                "model {label} enables mla_latent_cache=true but adapter '{}' does not support MLA latent cache; the flag is a no-op at runtime",
                inspected.adapter_id
            ),
            result,
        );
    }
}

#[allow(clippy::too_many_lines)]
fn check_models(config: &HiggsConfig, result: &mut DoctorResult) {
    for model in &config.models {
        let label = model_label(model);
        if let Err(error) = model.validate_disk_prefix_store() {
            fail(&format!("model {label} disk prefix store: {error}"), result);
        } else if model.kv_disk_dir.is_some() {
            pass(
                &format!("model {label} disk prefix store is writable"),
                result,
            );
        }
        if !check_prefill_yield_tokens(&label, model.prefill_yield_tokens, result) {
            continue;
        }
        let kv_cache_config = model.kv_cache_config();
        match kv_cache_config.validate() {
            Ok(()) => {
                // `validate()` only rejects the MLA/TurboQuant combination
                // using the *resolved* decision (env-aware), so a
                // config-declared conflict that HIGGS_MLA_LATENT_CACHE
                // overrides away passes silently there. Surface that as an
                // advisory warning rather than staying silent.
                if model.mla_latent_cache == Some(true)
                    && kv_cache_config.is_turboquant()
                    && !higgs_models::cache::resolve_mla_latent_cache(kv_cache_config.mla_latent)
                {
                    warn(
                        &format!(
                            "model {label} sets mla_latent_cache=true with kv_cache=turboquant, but HIGGS_MLA_LATENT_CACHE overrides MLA off; the conflict is masked at runtime"
                        ),
                        result,
                    );
                }
            }
            Err(err) => {
                fail(
                    &format!("model {label} has invalid KV cache config: {err}"),
                    result,
                );
                continue;
            }
        }
        // Cache-resident retention limits: catch values that "work" but quietly
        // defeat the cache (a too-low cap drops it almost every turn → constant
        // full-prefill). 0 = disabled, which is fine.
        let kv = model.kv_cache_config();
        if kv.max_session_tokens > 0 && kv.max_session_tokens < 512 {
            warn(
                &format!(
                    "model {label} kv_max_session_tokens={} is very low — most turns will drop the retained cache and full-prefill (use 0 to disable the cap, or a larger value)",
                    kv.max_session_tokens
                ),
                result,
            );
        }
        if kv.retained_idle_secs > 0 && kv.retained_idle_secs < 30 {
            warn(
                &format!(
                    "model {label} kv_retained_idle_secs={} is very low — retained caches will be evicted almost immediately (use 0 to disable idle eviction, or a larger value)",
                    kv.retained_idle_secs
                ),
                result,
            );
        }
        // A byte budget smaller than a single typical prefix defeats the cache
        // (every store evicts immediately). Warn but don't fail — 0 disables it.
        if kv.kv_cache_bytes > 0 && kv.kv_cache_bytes < 1 << 20 {
            warn(
                &format!(
                    "model {label} kv_cache_bytes={} is very low (< 1 MiB) — the prefix cache will likely evict every entry and provide no reuse (use 0 to disable the budget, or a larger value)",
                    kv.kv_cache_bytes
                ),
                result,
            );
        }
        if model.batch && kv_cache_config.is_turboquant() {
            fail(
                &format!(
                    "model {label} enables unsupported combination: TurboQuant with batch=true"
                ),
                result,
            );
            continue;
        }
        if model.kv_cache_config().is_turboquant() {
            let tq_default = higgs_models::cache::DEFAULT_TURBOQUANT_ACTIVATE_AT;
            warn(
                &format!(
                    "model {label} sets kv_cache=turboquant: its custom Metal decode kernels are \
                     ~20-25% slower than dense SDPA until the activation threshold (default \
                     {tq_default} tokens; override with HIGGS_TURBOQUANT_MIN_TOKENS), plus a \
                     first-token stall when prefilled KV is bulk-quantized. Dense KV is only \
                     ~10 KB/token — prefer it unless very long context threatens memory."
                ),
                result,
            );
        }
        if let Some(ref drafter) = model.draft_model {
            if !std::path::Path::new(drafter).exists() {
                fail(
                    &format!("model {label} draft_model path does not exist: {drafter}"),
                    result,
                );
                continue;
            }
            if model.batch {
                fail(
                    &format!(
                        "model {label} sets draft_model but DFlash is simple-engine only (batch=true)"
                    ),
                    result,
                );
                continue;
            }
        }
        // PFlash compressive prefill (docs/RESEARCH-pflash-prior-art.md).
        if let Some(ref pd) = model.prefill_drafter {
            if !std::path::Path::new(pd).exists() {
                fail(
                    &format!("model {label} prefill_drafter path does not exist: {pd}"),
                    result,
                );
                continue;
            }
            if model.batch {
                fail(
                    &format!(
                        "model {label} sets prefill_drafter but PFlash is simple-engine only (batch=true)"
                    ),
                    result,
                );
                continue;
            }
        }
        if model.prefill_compression != crate::config::PrefillCompressionMode::Off
            && model.prefill_drafter.is_none()
        {
            fail(
                &format!(
                    "model {label} sets prefill_compression={:?} but no prefill_drafter is configured",
                    model.prefill_compression
                ),
                result,
            );
            continue;
        }
        if model.prefill_drafter.is_some()
            && model.prefill_compression != crate::config::PrefillCompressionMode::Off
        {
            warn(
                &format!(
                    "model {label} enables PFlash (mode={:?}, keep_ratio={}, threshold={}): \
                     max_auto_prefill_ratio={}; \
                     compressed output is NOT byte-identical to uncompressed. \
                     Validate on your workload before relying on it.",
                    model.prefill_compression,
                    model.prefill_keep_ratio,
                    model.prefill_threshold,
                    model.prefill_max_auto_prefill_ratio
                ),
                result,
            );
        }
        if let Err(error) = crate::config::validate_pflash_settings(model) {
            fail(&format!("model {label} {error}"), result);
            continue;
        }
        if model.prefill_suffix_identity_threshold > 512 {
            warn(
                &format!(
                    "model {label} prefill_suffix_identity_threshold={} is high; exact suffix prefill can dominate TTFT on slow targets",
                    model.prefill_suffix_identity_threshold
                ),
                result,
            );
        }
        if model.prefill_drafter.is_some() {
            if model.prefill_threshold < 1024 {
                warn(
                    &format!(
                        "model {label} prefill_threshold={} is very low; compressing short prompts costs more than it saves",
                        model.prefill_threshold
                    ),
                    result,
                );
            }
        }
        match model_resolver::resolve(&model.path) {
            Ok(resolved) => {
                let inspected = match inspect_adapter(&resolved) {
                    Ok(inspected) => inspected,
                    Err(err) => {
                        fail(
                            &format!("model {label} architecture validation failed: {err}"),
                            result,
                        );
                        continue;
                    }
                };
                let detected = higgs_models::adapter::detect(&resolved).ok();
                if model.disable_vision
                    && !detected.as_ref().is_some_and(checkpoint_declares_vision)
                {
                    warn(
                        &format!(
                            "model {label} sets disable_vision=true but the checkpoint has no vision weights; the flag is a no-op"
                        ),
                        result,
                    );
                }
                if !inspected.capabilities.vision
                    && detected.as_ref().is_some_and(checkpoint_declares_vision)
                {
                    warn(
                        &format!(
                            "model {label} checkpoint contains vision weights that Higgs will ignore (adapter '{}' does not implement vision)",
                            inspected.adapter_id
                        ),
                        result,
                    );
                }
                if model.batch && !inspected.capabilities.batch_engine {
                    fail(
                        &format!(
                            "model {label} enables unsupported batch=true; adapter '{}' does not support true batched decode",
                            inspected.adapter_id
                        ),
                        result,
                    );
                    continue;
                }
                if higgs_models::cache::resolve_mla_latent_cache(kv_cache_config.mla_latent) {
                    check_mla_latent_cache_adapter(&label, &inspected, result);
                }
                let requested_profile = model.requested_mlx_profile(&config.local);
                let profile_msg = if model.batch {
                    "batch=true; batched decode supported".to_owned()
                } else {
                    let effective_profile =
                        resolve_effective_mlx_profile(&resolved, requested_profile);
                    if effective_profile.as_str() == requested_profile.as_str() {
                        format!("mlx_profile={}", effective_profile.as_str())
                    } else {
                        format!(
                            "mlx_profile={} (requested {})",
                            effective_profile.as_str(),
                            requested_profile.as_str()
                        )
                    }
                };
                let version = inspected
                    .version
                    .map_or_else(|| "unknown".to_owned(), |version| version.to_string());
                let vision = vision_status(&inspected, detected.as_ref(), model.disable_vision);
                pass(
                    &format!(
                        "model {label} resolvable (adapter={}, family={}, version={version}; {vision}; {profile_msg})",
                        inspected.adapter_id, inspected.family
                    ),
                    result,
                );
                check_eschamoe_checkpoint(&resolved, &label, result);
            }
            Err(err) => fail(&format!("model {label} not found: {err}"), result),
        }
    }
}

// -- Eschamoe checkpoint checks --

/// Check an eschamoe checkpoint before the server starts.
///
/// The check validates the quantization declaration, estimates the resident
/// size after conversion, and warns about the slow CPU-bound load.
fn check_eschamoe_checkpoint(model_dir: &std::path::Path, label: &str, result: &mut DoctorResult) {
    check_eschamoe_checkpoint_with_ram(
        model_dir,
        label,
        total_system_ram_bytes(),
        higgs_models::eschamoe::native_mode(),
        result,
    );
}

fn check_eschamoe_checkpoint_with_ram(
    model_dir: &std::path::Path,
    label: &str,
    ram_bytes: Option<u64>,
    native: bool,
    result: &mut DoctorResult,
) {
    let is_eschamoe = match higgs_models::eschamoe::is_eschamoe_checkpoint(model_dir) {
        Ok(flag) => flag,
        Err(err) => {
            fail(
                &format!(
                    "model {label} has a malformed quantization declaration in \
                     quantize_config.json or config.json (field quant_method): {err}"
                ),
                result,
            );
            return;
        }
    };

    if !check_quant_method_declarations(model_dir, label, result) {
        return;
    }
    if !is_eschamoe {
        return;
    }

    let target = higgs_models::eschamoe::CONVERSION_TARGET;
    if native {
        pass(
            &format!(
                "model {label} is an eschamoe trellis checkpoint; higgs keeps the experts in the \
                 trellis form and reads them with the Metal kernel. The other weights become MLX \
                 affine {}-bit (group size {}).",
                target.bits, target.group_size
            ),
            result,
        );
    } else {
        pass(
            &format!(
                "model {label} is an eschamoe trellis checkpoint; HIGGS_ESCHA_NATIVE=0 selects \
                 the affine path, so higgs decodes every expert to MLX affine {}-bit (group size \
                 {}) in memory at load",
                target.bits, target.group_size
            ),
            result,
        );
        warn(
            &format!(
                "model {label} is an eschamoe checkpoint on the affine path: the trellis decode \
                 runs on the CPU at load (~140s for the 35B checkpoint) and the result is about \
                 twice the resident size of the native path; a long first start is expected, not \
                 a hang. Unset HIGGS_ESCHA_NATIVE for the native path."
            ),
            result,
        );
    }
    match eschamoe_resident_estimate_bytes(model_dir, native) {
        Some(estimate) => check_eschamoe_memory(estimate, ram_bytes, label, result),
        None => warn(
            &format!(
                "model {label} config.json lacks size fields (num_hidden_layers, num_experts, \
                 moe_intermediate_size, hidden_size, vocab_size); cannot estimate resident \
                 memory after eschamoe conversion"
            ),
            result,
        ),
    }
}

/// Compare the `quant_method` declarations of the two config files.
///
/// A conflict or a bad field type is an error. Return `false` on error.
fn check_quant_method_declarations(
    model_dir: &std::path::Path,
    label: &str,
    result: &mut DoctorResult,
) -> bool {
    let mut methods: Vec<(&str, String)> = Vec::new();
    for file in ["quantize_config.json", "config.json"] {
        let Ok(raw) = std::fs::read_to_string(model_dir.join(file)) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
            // `is_eschamoe_checkpoint` already reports files that are not JSON.
            continue;
        };
        if let Some(quant_config) = value.get("quantization_config") {
            if !quant_config.is_object() {
                fail(
                    &format!("model {label} {file} field quantization_config is not a JSON object"),
                    result,
                );
                return false;
            }
        }
        let method = value.get("quant_method").or_else(|| {
            value
                .get("quantization_config")
                .and_then(|qc| qc.get("quant_method"))
        });
        match method {
            Some(serde_json::Value::String(name)) => methods.push((file, name.clone())),
            Some(_) => {
                fail(
                    &format!("model {label} {file} field quant_method is not a string"),
                    result,
                );
                return false;
            }
            None => {}
        }
    }
    if let [(file_a, method_a), (file_b, method_b)] = methods.as_slice() {
        if method_a != method_b {
            fail(
                &format!(
                    "model {label} {file_a} quant_method=\"{method_a}\" conflicts with {file_b} \
                     quant_method=\"{method_b}\"; the loader obeys {file_a}"
                ),
                result,
            );
            return false;
        }
    }
    true
}

/// Estimate the resident bytes of an eschamoe model after conversion.
///
/// The estimate counts the expert weights and the vocabulary weights. These
/// weights dominate the total.
///
/// The vocabulary weights always take the affine layout: packed values plus
/// one fp16 scale and one fp16 bias per group. The expert weights depend on
/// the mode. The affine path gives them the same layout. The native path
/// keeps the trellis codes, which hold `K` bits for each weight, and `K`
/// differs per projection: the 35B release uses 2 bits for `gate_up_proj` and
/// 3 for `down_proj`. Thus the native estimate reads the rate of each
/// projection from `quantization_config.layer_meta` and falls back to the
/// affine estimate when that block is absent.
fn eschamoe_resident_estimate_bytes(model_dir: &std::path::Path, native: bool) -> Option<u64> {
    let raw = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&raw).ok()?;
    // A checkpoint with a vision tower nests the text fields under
    // `text_config`. The released 35B escha checkpoint has that shape, so a
    // top-level read alone finds nothing and the estimate never runs.
    let field = |key: &str| -> Option<u64> {
        config
            .get(key)
            .or_else(|| config.get("text_config").and_then(|text| text.get(key)))?
            .as_u64()
    };
    let layers = field("num_hidden_layers")?;
    let experts = field("num_experts")?;
    let moe_intermediate = field("moe_intermediate_size")?;
    let hidden = field("hidden_size")?;
    let vocab = field("vocab_size")?;

    let target = higgs_models::eschamoe::CONVERSION_TARGET;
    let bits = u64::try_from(target.bits).ok()?;
    let group_size = u64::try_from(target.group_size).ok()?;
    let affine_bytes = |params: u64| params * bits / 8 + params * 4 / group_size;

    // Three projections per expert: gate, up, and down.
    let expert_params = layers * 3 * experts * moe_intermediate * hidden;
    let expert_bytes = if native {
        trellis_expert_bytes(&config).unwrap_or_else(|| affine_bytes(expert_params))
    } else {
        affine_bytes(expert_params)
    };

    // The embedding table and the output head.
    Some(expert_bytes + affine_bytes(2 * vocab * hidden))
}

/// Sum the trellis code bytes of every expert projection.
///
/// Each entry of `quantization_config.layer_meta` gives the expert count, the
/// two feature lengths, and the rate `K`. The code holds `K` bits for each
/// weight. Give `None` when the block is absent or an entry lacks a field, so
/// the caller can fall back.
fn trellis_expert_bytes(config: &serde_json::Value) -> Option<u64> {
    let meta = config
        .get("quantization_config")?
        .get("layer_meta")?
        .as_object()?;
    if meta.is_empty() {
        return None;
    }
    let mut bits = 0u64;
    for entry in meta.values() {
        let field = |key: &str| entry.get(key)?.as_u64();
        bits +=
            field("num_experts")? * field("in_features")? * field("out_features")? * field("K")?;
    }
    Some(bits / 8)
}

/// Warn when the resident estimate crowds the memory of the machine.
///
/// No public accessor gives the Metal working-set limit to the doctor. Thus
/// the check uses 75% of the total system RAM as a stand-in and says so.
fn check_eschamoe_memory(
    estimate: u64,
    ram_bytes: Option<u64>,
    label: &str,
    result: &mut DoctorResult,
) {
    let Some(ram) = ram_bytes else { return };
    let threshold = ram / 4 * 3;
    if estimate > threshold {
        warn(
            &format!(
                "model {label} needs an estimated {:.1} GiB resident after eschamoe conversion — \
                 far more than the on-disk size suggests (the 35B release is 12.3 GB on disk but \
                 ~20 GB resident). That exceeds 75% of system RAM ({:.1} GiB). No Metal \
                 working-set accessor is available to doctor, so total RAM is the reference; the \
                 real Metal limit is lower. Expect memory pressure or an OOM kill",
                gib(estimate),
                gib(ram)
            ),
            result,
        );
    } else {
        pass(
            &format!(
                "model {label} estimated resident size after eschamoe conversion (~{:.1} GiB \
                 weights) fits in system RAM ({:.1} GiB)",
                gib(estimate),
                gib(ram)
            ),
            result,
        );
    }
}

/// Convert bytes to GiB for display.
#[allow(clippy::cast_precision_loss, clippy::as_conversions)]
const fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Read the total system RAM in bytes.
///
/// The engine reads the Metal working-set limit through a private helper.
/// The doctor cannot call it, so the doctor reads the total RAM instead.
#[cfg(target_os = "macos")]
fn total_system_ram_bytes() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}

#[cfg(not(target_os = "macos"))]
const fn total_system_ram_bytes() -> Option<u64> {
    None
}

/// Catch `[local]` keys mistakenly written under `[server]`.
///
/// `local.allow_runtime_model_load = true` placed below a `[server]` header is
/// parsed by TOML as `server.local.*` -- an unknown key serde silently drops, so
/// the setting never takes effect. This is only visible in the raw file; the
/// parsed [`HiggsConfig`] no longer carries the stray key.
fn check_misplaced_local_keys(config_path: Option<&std::path::Path>, result: &mut DoctorResult) {
    let Some(path) = config_path else { return };
    let Ok(raw) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(doc) = raw.parse::<toml_edit::DocumentMut>() else {
        return;
    };
    let Some(server) = doc.get("server").and_then(|item| item.as_table()) else {
        return;
    };

    let stray: Vec<&str> = [
        "local",
        "allow_runtime_model_load",
        "raise_wired_limit",
        "mlx_profile",
    ]
    .into_iter()
    .filter(|key| server.contains_key(key))
    .collect();
    if stray.is_empty() {
        pass("no misplaced [local] keys under [server]", result);
    } else {
        warn(
            &format!(
                "{stray:?} found under [server]: TOML reads these as server.* and silently ignores them, so the setting never applies. Move them to a top-level [local] table."
            ),
            result,
        );
    }
}

fn check_runtime_model_load(config: &HiggsConfig, result: &mut DoctorResult) {
    if config.local.allow_runtime_model_load {
        warn(
            "local.allow_runtime_model_load is enabled: POST/DELETE /v1/models can load and unload models at runtime; ensure server.api_key restricts this to trusted operators",
            result,
        );
    } else {
        pass("runtime model load/unload disabled", result);
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
    let host = &config.server.host;
    let port = config.server.port;
    let addr = host.parse::<std::net::IpAddr>().map_or_else(
        |_| format!("{host}:{port}"),
        |ip| std::net::SocketAddr::new(ip, port).to_string(),
    );
    match std::net::TcpListener::bind(&addr) {
        Ok(_) => pass(&format!("port {} available", config.server.port), result),
        Err(err) => warn(
            &format!("port {} unavailable: {err}", config.server.port),
            result,
        ),
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

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
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

    // -- Misplaced [local] keys under [server] --

    fn write_config(toml: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();
        dir
    }

    #[test]
    fn test_local_key_under_server_warns() {
        // The exact footgun: dotted `local.*` written below [server] parses as
        // server.local.* and is silently dropped.
        let dir =
            write_config("[server]\nhost = \"0.0.0.0\"\nlocal.allow_runtime_model_load = true\n");
        let mut result = empty_result();
        check_misplaced_local_keys(Some(&dir.path().join("config.toml")), &mut result);
        assert_eq!(result.warnings, 1);
        assert_eq!(result.passes, 0);
    }

    #[test]
    fn test_correct_local_table_passes() {
        let dir = write_config(
            "[server]\nhost = \"0.0.0.0\"\n\n[local]\nallow_runtime_model_load = true\n",
        );
        let mut result = empty_result();
        check_misplaced_local_keys(Some(&dir.path().join("config.toml")), &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn prefill_yield_tokens_rejects_small_nonzero_values() {
        let mut result = empty_result();
        assert!(!check_prefill_yield_tokens("test", Some(127), &mut result));
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn prefill_yield_tokens_warns_below_recommended_quantum() {
        let mut result = empty_result();
        assert!(check_prefill_yield_tokens("test", Some(128), &mut result));
        assert_eq!(result.warnings, 1);
    }

    #[test]
    fn prefill_yield_tokens_accepts_disabled_quantum() {
        let mut result = empty_result();
        assert!(check_prefill_yield_tokens("test", Some(0), &mut result));
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    // -- Duplicate model detection --

    #[test]
    fn test_no_duplicates_passes() {
        let config = HiggsConfig {
            models: vec![
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    ..Default::default()
                },
                ModelConfig {
                    path: "org/model-b".to_owned(),
                    ..Default::default()
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
                    ..Default::default()
                },
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    ..Default::default()
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

    // -- Server section validation --

    fn server_with(modify: impl FnOnce(&mut ServerSection)) -> HiggsConfig {
        let mut server = ServerSection::default();
        modify(&mut server);
        // Keep the image cap within the body cap so the default advisory (the
        // 20 MiB image cap exceeds the 10 MiB body cap) doesn't fire in tests
        // isolating other server checks. Tests exercising the advisory set
        // `max_image_bytes` themselves and build the config directly.
        if server.max_image_bytes > server.max_body_size {
            server.max_image_bytes = server.max_body_size;
        }
        HiggsConfig {
            server,
            ..HiggsConfig::default()
        }
    }

    #[test]
    fn test_server_default_passes() {
        let config = HiggsConfig::default();
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        // Advisory only: the default image cap (20 MiB) exceeds the default
        // body cap (10 MiB), which the doctor flags on an untouched config.
        assert_eq!(result.warnings, 1);
    }

    #[test]
    fn test_max_tokens_zero_fails() {
        let config = server_with(|s| s.max_tokens = 0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_timeout_zero_fails() {
        let config = server_with(|s| s.timeout = 0.0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_timeout_negative_fails() {
        let config = server_with(|s| s.timeout = -1.0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_timeout_nan_fails() {
        let config = server_with(|s| s.timeout = f64::NAN);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_timeout_infinite_fails() {
        let config = server_with(|s| s.timeout = f64::INFINITY);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_timeout_unusually_high_warns() {
        let config = server_with(|s| s.timeout = 3600.0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_max_body_size_zero_fails() {
        let config = server_with(|s| s.max_body_size = 0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_max_body_size_huge_warns() {
        let config = server_with(|s| s.max_body_size = (1 << 30) + 1);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
        assert_eq!(result.failures, 0);
    }

    // -- Vision server-section validation --

    #[test]
    fn doctor_warns_when_image_cap_exceeds_body_cap() {
        let mut server = ServerSection::default();
        server.max_image_bytes = server.max_body_size + 1;
        let config = HiggsConfig {
            server,
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert!(result.warnings >= 1);
    }

    #[test]
    fn test_max_image_dimension_below_range_fails() {
        let config = server_with(|s| s.max_image_dimension = 63);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_max_image_dimension_above_range_fails() {
        let config = server_with(|s| s.max_image_dimension = 16385);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_max_image_dimension_in_range_passes() {
        let config = server_with(|s| s.max_image_dimension = 4096);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_image_fetch_timeout_zero_fails() {
        let config = server_with(|s| s.image_fetch_timeout = 0.0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_image_fetch_timeout_negative_fails() {
        let config = server_with(|s| s.image_fetch_timeout = -1.0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_image_fetch_timeout_nan_fails() {
        let config = server_with(|s| s.image_fetch_timeout = f64::NAN);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_image_fetch_timeout_positive_passes() {
        let config = server_with(|s| s.image_fetch_timeout = 10.0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_host_localhost_passes() {
        let config = server_with(|s| s.host = "localhost".to_owned());
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_host_ipv4_passes() {
        let config = server_with(|s| s.host = "127.0.0.1".to_owned());
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_host_ipv6_passes() {
        let config = server_with(|s| s.host = "::1".to_owned());
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_host_garbage_warns() {
        let config = server_with(|s| s.host = "not a valid host!!".to_owned());
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
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

    #[test]
    fn test_port_available_ipv6_localhost() {
        let config = HiggsConfig {
            server: ServerSection {
                host: "::1".to_owned(),
                port: 0,
                ..ServerSection::default()
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_port_availability(&config, &mut result);
        assert_eq!(result.passes, 1);
    }

    #[test]
    fn test_api_key_set_passes() {
        let config = server_with(|s| s.api_key = Some("secret".to_owned()));
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_rate_limit_nonzero_passes() {
        let config = server_with(|s| s.rate_limit = 120);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_non_loopback_host_without_api_key_warns() {
        let config = server_with(|s| s.host = "0.0.0.0".to_owned());
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_non_loopback_host_with_api_key_no_warning() {
        let config = server_with(|s| {
            s.host = "0.0.0.0".to_owned();
            s.api_key = Some("sk-test".to_owned());
        });
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_cors_wildcard_on_loopback_passes() {
        let config = server_with(|s| s.cors_origins = Some(vec!["*".to_owned()]));
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_cors_wildcard_on_network_host_warns() {
        let config = server_with(|s| {
            s.host = "0.0.0.0".to_owned();
            s.api_key = Some("sk-test".to_owned());
            s.cors_origins = Some(vec!["*".to_owned()]);
        });
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_cors_valid_origin_list_passes() {
        let config = server_with(|s| {
            s.cors_origins = Some(vec![
                "https://example.com".to_owned(),
                "http://localhost:3000".to_owned(),
            ]);
        });
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_cors_invalid_origin_fails() {
        let config = server_with(|s| s.cors_origins = Some(vec!["not a url".to_owned()]));
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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

    // -- Eschamoe checkpoint checks --

    fn write_model_dir(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, content) in files {
            std::fs::write(dir.path().join(name), content).unwrap();
        }
        dir
    }

    /// A small eschamoe model dir. The formula gives 2304 estimate bytes:
    /// params = 2*3*4*8*16 + 2*32*16 = 4096; 4096*4/8 + 4096*4/64 = 2304.
    fn eschamoe_model_dir() -> tempfile::TempDir {
        write_model_dir(&[
            ("quantize_config.json", r#"{"quant_method":"eschamoe"}"#),
            (
                "config.json",
                r#"{"model_type":"qwen3_5_moe","num_hidden_layers":2,"num_experts":4,
                    "moe_intermediate_size":8,"hidden_size":16,"vocab_size":32}"#,
            ),
        ])
    }

    #[test]
    fn test_eschamoe_conflicting_quant_method_fails() {
        let dir = write_model_dir(&[
            ("quantize_config.json", r#"{"quant_method":"awq"}"#),
            (
                "config.json",
                r#"{"quantization_config":{"quant_method":"eschamoe"}}"#,
            ),
        ]);
        let mut result = empty_result();
        check_eschamoe_checkpoint_with_ram(dir.path(), "test", Some(1 << 40), true, &mut result);
        assert_eq!(result.failures, 1);
        assert_eq!(result.passes, 0);
    }

    #[test]
    fn test_eschamoe_malformed_quantize_config_fails() {
        let dir = write_model_dir(&[("quantize_config.json", "not json")]);
        let mut result = empty_result();
        check_eschamoe_checkpoint_with_ram(dir.path(), "test", Some(1 << 40), true, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_eschamoe_exceeds_ram_warns_memory() {
        let dir = eschamoe_model_dir();
        // 2304 estimate bytes exceed 75% of 1024 bytes of injected RAM.
        let mut native = empty_result();
        check_eschamoe_checkpoint_with_ram(dir.path(), "test", Some(1024), true, &mut native);
        assert_eq!(native.failures, 0);
        assert_eq!(native.passes, 1);
        // The native path warns about the memory estimate and nothing else.
        assert_eq!(native.warnings, 1);

        let mut affine = empty_result();
        check_eschamoe_checkpoint_with_ram(dir.path(), "test", Some(1024), false, &mut affine);
        assert_eq!(affine.failures, 0);
        assert_eq!(affine.passes, 1);
        // The affine path adds the slow CPU load warning.
        assert_eq!(affine.warnings, 2);
    }

    #[test]
    fn test_eschamoe_fits_ram_passes_memory_check() {
        let dir = eschamoe_model_dir();
        let mut native = empty_result();
        check_eschamoe_checkpoint_with_ram(dir.path(), "test", Some(1 << 40), true, &mut native);
        assert_eq!(native.failures, 0);
        // Detection pass plus memory-fit pass, and no warning.
        assert_eq!(native.passes, 2);
        assert_eq!(native.warnings, 0);

        let mut affine = empty_result();
        check_eschamoe_checkpoint_with_ram(dir.path(), "test", Some(1 << 40), false, &mut affine);
        assert_eq!(affine.passes, 2);
        assert_eq!(affine.warnings, 1);
    }

    /// Test that the size fields are found under `text_config` too. A
    /// checkpoint with a vision tower nests them there, and the released 35B
    /// escha checkpoint does. The numbers match the flat fixture, so the
    /// estimate must agree with it.
    #[test]
    fn test_eschamoe_estimate_reads_nested_text_config() {
        let dir = write_model_dir(&[
            ("quantize_config.json", r#"{"quant_method":"eschamoe"}"#),
            (
                "config.json",
                r#"{"model_type":"qwen3_5_moe","text_config":{"num_hidden_layers":2,
                    "num_experts":4,"moe_intermediate_size":8,"hidden_size":16,
                    "vocab_size":32}}"#,
            ),
        ]);
        assert_eq!(
            eschamoe_resident_estimate_bytes(dir.path(), false),
            Some(2304)
        );
    }

    /// Test the estimate in both modes. The fixture has no `layer_meta`, so
    /// the native estimate falls back to the affine one.
    #[test]
    fn test_eschamoe_resident_estimate_formula() {
        let dir = eschamoe_model_dir();
        assert_eq!(
            eschamoe_resident_estimate_bytes(dir.path(), false),
            Some(2304)
        );
        assert_eq!(
            eschamoe_resident_estimate_bytes(dir.path(), true),
            Some(2304)
        );
    }

    /// Test that the native estimate reads the trellis rate of each
    /// projection. The two entries hold 4*8*16*2 and 4*16*8*3 bits, so the
    /// codes take (1024 + 1536) / 8 = 320 bytes. The vocabulary adds
    /// 2*32*16 = 1024 params, which take 1024*4/8 + 1024*4/64 = 576 bytes.
    #[test]
    fn test_eschamoe_native_estimate_uses_the_trellis_rate() {
        let dir = write_model_dir(&[
            ("quantize_config.json", r#"{"quant_method":"eschamoe"}"#),
            (
                "config.json",
                r#"{"model_type":"qwen3_5_moe","num_hidden_layers":2,"num_experts":4,
                    "moe_intermediate_size":8,"hidden_size":16,"vocab_size":32,
                    "quantization_config":{"quant_method":"eschamoe","layer_meta":{
                      "layers.0.mlp.experts.gate_up_proj":{"K":2,"num_experts":4,
                        "in_features":8,"out_features":16},
                      "layers.0.mlp.experts.down_proj":{"K":3,"num_experts":4,
                        "in_features":16,"out_features":8}}}}"#,
            ),
        ]);
        assert_eq!(
            eschamoe_resident_estimate_bytes(dir.path(), true),
            Some(320 + 576)
        );
        // The affine path ignores `layer_meta` and keeps the old formula.
        assert_eq!(
            eschamoe_resident_estimate_bytes(dir.path(), false),
            Some(2304)
        );
    }

    #[test]
    fn test_non_eschamoe_dir_is_silent() {
        let dir = write_model_dir(&[("config.json", r#"{"model_type":"llama"}"#)]);
        let mut result = empty_result();
        check_eschamoe_checkpoint_with_ram(dir.path(), "test", Some(1 << 40), true, &mut result);
        assert_eq!(result.passes, 0);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
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

    // -- mla_latent_cache --

    fn model_with_path(path: String) -> ModelConfig {
        ModelConfig {
            path,
            ..Default::default()
        }
    }

    fn write_model_config_json(dir: &std::path::Path, model_type: &str) {
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"model_type": "{model_type}"}}"#),
        )
        .unwrap();
    }

    /// Write a config.json that declares vision capability (`vision_config`)
    /// alongside the given `model_type`.
    fn write_model_config_with_vision(dir: &std::path::Path, model_type: &str) {
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"model_type": "{model_type}", "vision_config": {{"hidden_size": 768}}}}"#),
        )
        .unwrap();
    }

    /// Run `f` with `HIGGS_MLA_LATENT_CACHE` set to `env_value` (or unset,
    /// for `None`), restoring the prior value afterward. Serialized via
    /// `crate::test_env_lock()` since this mutates process-global state;
    /// combined with `--test-threads=1` (the repo's mandated test-runner
    /// flag for this crate) there is no interleaving risk, but the lock
    /// keeps the guarantee explicit and independent of that flag.
    #[allow(unsafe_code)]
    fn with_mla_env<R>(env_value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _guard = crate::test_env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = std::env::var("HIGGS_MLA_LATENT_CACHE").ok();
        match env_value {
            Some(v) => unsafe { std::env::set_var("HIGGS_MLA_LATENT_CACHE", v) },
            None => unsafe { std::env::remove_var("HIGGS_MLA_LATENT_CACHE") },
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

        match previous.as_deref() {
            Some(v) => unsafe { std::env::set_var("HIGGS_MLA_LATENT_CACHE", v) },
            None => unsafe { std::env::remove_var("HIGGS_MLA_LATENT_CACHE") },
        }
        result.unwrap_or_else(|payload| std::panic::resume_unwind(payload))
    }

    #[test]
    fn test_mla_latent_cache_turboquant_conflict_fails_in_check_models() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "deepseek_v2");
            let mut model = model_with_path(dir.path().to_str().unwrap().to_owned());
            model.kv_cache = higgs_models::turboquant::KvCacheMode::Turboquant;
            model.mla_latent_cache = Some(true);
            let config = HiggsConfig {
                models: vec![model],
                ..HiggsConfig::default()
            };
            let mut result = empty_result();
            check_models(&config, &mut result);
            assert_eq!(result.failures, 1);
            assert_eq!(result.warnings, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_env_off_masks_turboquant_conflict_as_warning() {
        with_mla_env(Some("0"), || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "deepseek_v2");
            let mut model = model_with_path(dir.path().to_str().unwrap().to_owned());
            model.kv_cache = higgs_models::turboquant::KvCacheMode::Turboquant;
            model.mla_latent_cache = Some(true);
            let config = HiggsConfig {
                models: vec![model],
                ..HiggsConfig::default()
            };
            let mut result = empty_result();
            check_models(&config, &mut result);
            assert_eq!(
                result.failures, 0,
                "HIGGS_MLA_LATENT_CACHE=0 should resolve the conflict away, not fail"
            );
            assert!(
                result.warnings >= 1,
                "the masked conflict should still surface as a warning"
            );
        });
    }

    #[test]
    fn test_mla_latent_cache_env_on_triggers_turboquant_conflict() {
        with_mla_env(Some("1"), || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "deepseek_v2");
            let mut model = model_with_path(dir.path().to_str().unwrap().to_owned());
            model.kv_cache = higgs_models::turboquant::KvCacheMode::Turboquant;
            // mla_latent_cache left unset in config -- the env var alone
            // must be enough to trigger the conflict.
            let config = HiggsConfig {
                models: vec![model],
                ..HiggsConfig::default()
            };
            let mut result = empty_result();
            check_models(&config, &mut result);
            assert_eq!(result.failures, 1);
            assert_eq!(result.warnings, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_architecture_passes_for_deepseek_v2() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "deepseek_v2");
            let model = ModelConfig {
                mla_latent_cache: Some(true),
                ..model_with_path(dir.path().to_str().unwrap().to_owned())
            };
            let mut result = empty_result();
            check_mla_latent_cache_architecture("test-model", &model, dir.path(), &mut result);
            assert_eq!(result.passes, 1);
            assert_eq!(result.warnings, 0);
            assert_eq!(result.failures, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_architecture_warns_for_non_deepseek() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "qwen2");
            let model = ModelConfig {
                mla_latent_cache: Some(true),
                ..model_with_path(dir.path().to_str().unwrap().to_owned())
            };
            let mut result = empty_result();
            check_mla_latent_cache_architecture("test-model", &model, dir.path(), &mut result);
            assert_eq!(result.passes, 0);
            assert_eq!(result.warnings, 1);
            assert_eq!(result.failures, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_architecture_env_on_warns_for_non_deepseek() {
        with_mla_env(Some("1"), || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "qwen2");
            // config leaves mla_latent_cache unset -- the env override alone
            // must be enough to trigger the architecture warning.
            let model = model_with_path(dir.path().to_str().unwrap().to_owned());
            let mut result = empty_result();
            check_mla_latent_cache_architecture("test-model", &model, dir.path(), &mut result);
            assert_eq!(result.passes, 0);
            assert_eq!(result.warnings, 1);
            assert_eq!(result.failures, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_architecture_env_off_suppresses_warning() {
        with_mla_env(Some("0"), || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "qwen2");
            let model = ModelConfig {
                mla_latent_cache: Some(true),
                ..model_with_path(dir.path().to_str().unwrap().to_owned())
            };
            let mut result = empty_result();
            check_mla_latent_cache_architecture("test-model", &model, dir.path(), &mut result);
            assert_eq!(result.passes, 0);
            assert_eq!(
                result.warnings, 0,
                "env forcing MLA off should suppress the architecture warning"
            );
            assert_eq!(result.failures, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_architecture_noop_when_unset() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "qwen2");
            let model = model_with_path(dir.path().to_str().unwrap().to_owned());
            let mut result = empty_result();
            check_mla_latent_cache_architecture("test-model", &model, dir.path(), &mut result);
            assert_eq!(result.passes, 0);
            assert_eq!(result.warnings, 0);
            assert_eq!(result.failures, 0);
        });
    }

    // -- Vision capability report --

    #[test]
    fn test_vision_status_supported_for_vision_adapter() {
        let dir = tempfile::tempdir().unwrap();
        write_model_config_with_vision(dir.path(), "llava-qwen2");
        let inspected = inspect_adapter(dir.path()).unwrap();
        let detected = higgs_models::adapter::detect(dir.path()).unwrap();
        assert_eq!(
            vision_status(&inspected, Some(&detected), false),
            "vision: supported (llava-qwen2)"
        );
    }

    #[test]
    fn test_vision_status_supported_for_multimodal_gemma() {
        // The `gemma3` / `gemma4` adapters run `load_gemma_vision_tower`, so a
        // multimodal checkpoint (vision weights declared) reports `supported`.
        for model_type in ["gemma3", "gemma4"] {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_with_vision(dir.path(), model_type);
            let inspected = inspect_adapter(dir.path()).unwrap();
            let detected = higgs_models::adapter::detect(dir.path()).unwrap();
            assert!(
                inspected.capabilities.vision,
                "{model_type} must advertise vision"
            );
            assert_eq!(
                vision_status(&inspected, Some(&detected), false),
                format!("vision: supported ({model_type})")
            );
        }
    }

    #[test]
    fn test_vision_status_none_for_text_only_gemma() {
        // `gemma3_text` / `gemma4_text` checkpoints carry no vision weights and
        // report `none` (the shared adapter only loads a tower when the
        // checkpoint actually has one).
        for model_type in ["gemma3_text", "gemma4_text"] {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), model_type);
            let inspected = inspect_adapter(dir.path()).unwrap();
            let detected = higgs_models::adapter::detect(dir.path()).unwrap();
            assert_eq!(
                vision_status(&inspected, Some(&detected), false),
                "vision: none",
                "{model_type} must not report vision"
            );
        }
    }

    #[test]
    fn test_vision_status_tower_ignored_for_text_adapter() {
        // A text-family adapter that truly implements no vision still reports
        // `tower-ignored` when the checkpoint declares vision weights.
        let dir = tempfile::tempdir().unwrap();
        write_model_config_with_vision(dir.path(), "phi3");
        let inspected = inspect_adapter(dir.path()).unwrap();
        let detected = higgs_models::adapter::detect(dir.path()).unwrap();
        assert!(!inspected.capabilities.vision);
        assert_eq!(
            vision_status(&inspected, Some(&detected), false),
            "vision: tower-ignored (phi3)"
        );
    }

    #[test]
    fn test_vision_status_disabled_by_escape_hatch() {
        let dir = tempfile::tempdir().unwrap();
        write_model_config_with_vision(dir.path(), "llava-qwen2");
        let inspected = inspect_adapter(dir.path()).unwrap();
        let detected = higgs_models::adapter::detect(dir.path()).unwrap();
        assert_eq!(
            vision_status(&inspected, Some(&detected), true),
            "vision: disabled (escape hatch; llava-qwen2)"
        );
    }

    #[test]
    fn test_vision_status_none_for_text_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        write_model_config_json(dir.path(), "qwen2");
        let inspected = inspect_adapter(dir.path()).unwrap();
        let detected = higgs_models::adapter::detect(dir.path()).unwrap();
        assert_eq!(
            vision_status(&inspected, Some(&detected), false),
            "vision: none"
        );
    }

    #[test]
    fn test_doctor_warns_tower_ignored_for_multimodal_checkpoint() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            // `phi3` has no vision adapter; a vision-declaring checkpoint still
            // warns that its tower would be ignored.
            write_model_config_with_vision(dir.path(), "phi3");
            let model = model_with_path(dir.path().to_str().unwrap().to_owned());
            let config = HiggsConfig {
                models: vec![model],
                ..HiggsConfig::default()
            };
            let mut result = empty_result();
            check_models(&config, &mut result);
            assert_eq!(result.failures, 0);
            assert_eq!(result.warnings, 1);
        });
    }

    #[test]
    fn test_doctor_no_longer_warns_tower_ignored_for_multimodal_gemma() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            // gemma3/gemma4 adapters load towers: no "will ignore" warning.
            write_model_config_with_vision(dir.path(), "gemma3");
            let model = model_with_path(dir.path().to_str().unwrap().to_owned());
            let config = HiggsConfig {
                models: vec![model],
                ..HiggsConfig::default()
            };
            let mut result = empty_result();
            check_models(&config, &mut result);
            assert_eq!(result.failures, 0);
            assert_eq!(result.warnings, 0);
        });
    }

    #[test]
    fn test_doctor_disable_vision_noop_warns_for_text_checkpoint() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "qwen2");
            let mut model = model_with_path(dir.path().to_str().unwrap().to_owned());
            model.disable_vision = true;
            let config = HiggsConfig {
                models: vec![model],
                ..HiggsConfig::default()
            };
            let mut result = empty_result();
            check_models(&config, &mut result);
            assert_eq!(result.failures, 0);
            assert_eq!(result.warnings, 1);
        });
    }

    #[test]
    fn test_doctor_disable_vision_noop_not_flagged_for_vlm() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_with_vision(dir.path(), "llava-qwen2");
            let mut model = model_with_path(dir.path().to_str().unwrap().to_owned());
            model.disable_vision = true;
            let config = HiggsConfig {
                models: vec![model],
                ..HiggsConfig::default()
            };
            let mut result = empty_result();
            check_models(&config, &mut result);
            assert_eq!(result.failures, 0);
            assert_eq!(result.warnings, 0);
        });
    }
}
