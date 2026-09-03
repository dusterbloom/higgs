//! Canonical identity for environment-selected inference behavior.
//!
//! Capacity profiles are reusable only when the choices that affect physical
//! layout, resident memory, or execution kernels are the same. Keep that list
//! here, next to the engine/model code that consumes the choices; frontends
//! must hash this value instead of maintaining their own partial inventory.

use serde::Serialize;
use serde_json::{Value, json};

/// Fully resolved set of process-wide choices that affect profile reuse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ResolvedRuntimeIdentity(Value);

/// Resolve the process environment into the canonical capacity-profile identity.
#[must_use]
pub fn resolved_runtime_identity(
    is_eschamoe: bool,
    mla_latent_cache: bool,
) -> ResolvedRuntimeIdentity {
    resolved_runtime_identity_with_selection(is_eschamoe, mla_latent_cache, None, |name| {
        std::env::var(name).ok()
    })
}

/// Resolve identity with the exact model/platform choices used by Qwen block
/// construction. Capacity callers with parsed model args must prefer this over
/// the model-agnostic compatibility entry point.
#[must_use]
pub fn resolved_runtime_identity_for_qwen(
    is_eschamoe: bool,
    mla_latent_cache: bool,
    args: &higgs_models::qwen3_next::Qwen3NextModelArgs,
) -> ResolvedRuntimeIdentity {
    let selection = higgs_models::qwen3_next::resolved_dense_runtime_selections(args, is_eschamoe);
    resolved_runtime_identity_with_selection(
        is_eschamoe,
        mla_latent_cache,
        Some(selection),
        |name| std::env::var(name).ok(),
    )
}

/// Resolver-injected form used by deterministic mutation tests.
#[doc(hidden)]
#[must_use]
pub fn resolved_runtime_identity_with(
    is_eschamoe: bool,
    mla_latent_cache: bool,
    env: impl Fn(&str) -> Option<String>,
) -> ResolvedRuntimeIdentity {
    resolved_runtime_identity_with_selection(is_eschamoe, mla_latent_cache, None, env)
}

fn resolved_runtime_identity_with_selection(
    is_eschamoe: bool,
    mla_latent_cache: bool,
    dense_selection: Option<higgs_models::qwen3_next::ResolvedDenseRuntimeSelections>,
    env: impl Fn(&str) -> Option<String>,
) -> ResolvedRuntimeIdentity {
    let flag = |name: &str, default: bool| {
        env(name)
            .as_deref()
            .and_then(parse_enabled_flag)
            .unwrap_or(default)
    };
    let nonzero = |name: &str, default: bool| env(name).map_or(default, |value| value != "0");
    let truthy = |name: &str| env(name).as_deref().is_some_and(is_truthy);
    let present = |name: &str| env(name).is_some();
    let optional_usize =
        |name: &str| env(name).and_then(|value| value.trim().parse::<usize>().ok());
    let optional_i32 = |name: &str| env(name).and_then(|value| value.parse::<i32>().ok());
    let restricted_i32 =
        |name: &str, allowed: &[i32]| optional_i32(name).filter(|value| allowed.contains(value));

    let prompt_defaults = crate::mtp::PromptLookupConfig::default();
    let dflash_verify_mode = env("HIGGS_DFLASH_VERIFY_MODE")
        .map(|value| value.trim().to_ascii_lowercase())
        .map_or("canonical_s1", |value| match value.as_str() {
            "block" | "batched" | "batched-tape" => "batched_tape",
            _ => "canonical_s1",
        });
    let dflash_confidence_bits = env("HIGGS_DFLASH_CONF_TRUNC")
        .map_or(Some(0.5_f32), |value| {
            value
                .trim()
                .parse::<f32>()
                .ok()
                .filter(|threshold| *threshold > 0.0 && *threshold <= 1.0)
        })
        .map(f32::to_bits);
    let mla_latent_cache = env("HIGGS_MLA_LATENT_CACHE")
        .as_deref()
        .and_then(parse_enabled_flag)
        .unwrap_or(mla_latent_cache);
    let turboquant_activate_at = env("HIGGS_TURBOQUANT_MIN_TOKENS")
        .or_else(|| env("HIGGS_TURBOQUANT_ACTIVATE_AT"))
        .and_then(|value| value.parse::<i32>().ok())
        .map_or(
            higgs_models::cache::DEFAULT_TURBOQUANT_ACTIVATE_AT,
            |value| value.max(0),
        );
    let qgemv_ffn_mode = env("HIGGS_QGEMV_FFN_MODE")
        .map(|value| value.trim().to_ascii_lowercase())
        .map_or("both", |value| match value.as_str() {
            "fused" | "fused_only" => "fused_only",
            "down" | "down_only" => "down_only",
            "off" | "none" => "off",
            _ => "both",
        });
    let dense_ffn_gate_up = dense_selection.map_or_else(
        || {
            env("HIGGS_DENSE_FFN_GATE_UP")
                .map(|value| value.trim().to_ascii_lowercase())
                .map_or("fused", |value| match value.as_str() {
                    "separate" | "split" | "0" | "false" | "off" => "separate",
                    _ => "fused",
                })
        },
        |selection| {
            if selection.gate_up_fused {
                "fused"
            } else {
                "separate"
            }
        },
    );
    let bonsai_q2_simd = dense_selection.map_or_else(
        || {
            env("HIGGS_BONSAI_Q2_SIMD")
                .map(|value| value.trim().to_owned())
                .map_or("disabled", |value| match value.as_str() {
                    "1" => "enabled",
                    "0" => "disabled",
                    _ => "disabled",
                })
        },
        |selection| match selection.q2_simd_decode_policy {
            higgs_models::qwen3_next::Q2SimdDecodePolicy::Stock => "disabled",
            higgs_models::qwen3_next::Q2SimdDecodePolicy::EschaQwen38 => "escha_qwen38",
            higgs_models::qwen3_next::Q2SimdDecodePolicy::ForceOn => "enabled",
        },
    );
    let escha_affine_bits =
        optional_i32("HIGGS_ESCHA_AFFINE_BITS").filter(|bits| (2..=8).contains(bits));

    let Value::Object(mut identity) = json!({
        "eschamoeNative": is_eschamoe && nonzero("HIGGS_ESCHA_NATIVE", true),
        "eschamoeAffineBitsOverride": escha_affine_bits,
        "eschamoeTrellisGemm": is_eschamoe && env("HIGGS_ESCHA_TRELLIS_GEMM").as_deref() == Some("1"),
        "eschamoeQgemmSimd": is_eschamoe && nonzero("HIGGS_ESCHA_QGEMM_SIMD", true),
        "eschamoeQgemmBlockRows": if env("HIGGS_ESCHA_QGEMM_BM").as_deref() == Some("64") { 64 } else { 32 },
        "mlaLatentCache": mla_latent_cache,
        "turboquantActivateAt": turboquant_activate_at,
        "prefixCache": flag("HIGGS_PREFIX_CACHE", true),
        "experimentalPagedKv": flag("HIGGS_EXPERIMENTAL_PAGED_KV", false),
        "promptLookup": flag("HIGGS_PROMPT_LOOKUP", false),
        "promptLookupUnchecked": flag("HIGGS_PROMPT_LOOKUP_UNCHECKED", false),
        "promptLookupDraftMax": optional_usize("HIGGS_PROMPT_LOOKUP_DRAFT_N_MAX").unwrap_or(prompt_defaults.max_drafts),
        "promptLookupNgramMax": optional_usize("HIGGS_PROMPT_LOOKUP_NGRAM_MAX").unwrap_or(prompt_defaults.max_ngram),
        "promptLookupWindow": optional_usize("HIGGS_PROMPT_LOOKUP_WINDOW").unwrap_or(prompt_defaults.max_window),
        "pflashFullScoreMaxTokens": crate::simple::resolve_pflash_full_score_max_tokens(env("HIGGS_PREFLASH_FULL_SCORE_MAX_TOKENS").as_deref()),
        "pflashMinimumFreeMb": crate::simple::resolve_pflash_min_free_memory_mb(env("HIGGS_PREFLASH_MIN_FREE_MB").as_deref()),
        "mtpAdaptiveDraft": flag("HIGGS_MTP_ADAPTIVE_DRAFT", false),
        "mtpPromptLookup": flag("HIGGS_MTP_PROMPT_LOOKUP", false),
        "mtpPrimePrefill": flag("HIGGS_MTP_PRIME_PREFILL", true),
        "dflashBlockSize": crate::simple::resolve_dflash_block_size_override(env("HIGGS_DFLASH_BLOCK_SIZE").as_deref()),
        "dsparkDraftCapOverride": crate::simple::resolve_dspark_draft_cap_override(env("HIGGS_DSPARK_DRAFT_CAP").as_deref()),
        "dflashVerifyMode": dflash_verify_mode,
        "dflashTargetHead": nonzero("HIGGS_DSPARK_TARGET_HEAD", false),
        "dflashAdaptive": nonzero("HIGGS_DFLASH_ADAPTIVE", true),
        "dflashGate": nonzero("HIGGS_DFLASH_GATE", true),
        "dflashMinimumBlock": crate::simple::resolve_dflash_min_block_override(env("HIGGS_DFLASH_MIN_BLOCK").as_deref()),
        "dflashConfidenceBits": dflash_confidence_bits,
        "dflashFusedConvolution": flag("HIGGS_DFLASH_FUSED_CONV", false),
        "dflashGdnConfigCache": flag("HIGGS_DFLASH_GDN_CONFIG_CACHE", false),
    }) else {
        unreachable!("JSON object literal must remain an object")
    };
    let Value::Object(model_runtime) = json!({
        "denseGdnRequant8Bit": present("HIGGS_DENSE_REQUANT_8BIT"),
        "separateGdnProjections": present("HIGGS_SEPARATE_GDN_PROJ"),
        "chunkedLoadEvaluation": nonzero("HIGGS_LOAD_EVAL_CHUNKED", true),
        "compiledGating": flag("HIGGS_COMPILED_GATING", true),
        "compiledGdnDecode": truthy("HIGGS_COMPILED_GDN_DECODE"),
        "asyncLayerStateEvaluation": flag("HIGGS_ASYNC_LAYER_STATE_EVAL", true),
        "bonsaiSymmetricQ1": flag("HIGGS_BONSAI_SYMMETRIC_Q1", true),
        "bonsaiTgLut4": nonzero("HIGGS_BONSAI_TG_LUT4", true),
        "bonsaiTgLut4FusedMlp": env("HIGGS_BONSAI_TG_LUT4_FUSED_MLP").as_deref() == Some("1"),
        "bonsaiTgLut4M5Workgroup": restricted_i32("HIGGS_BONSAI_TG_LUT4_M5_WG", &[128, 160, 192, 224, 256]).unwrap_or(256),
        "bonsaiQmvKernel": if env("HIGGS_BONSAI_QMV_KERNEL").is_some_and(|value| value.eq_ignore_ascii_case("legacy")) { "legacy" } else { "fast" },
        "bonsaiQmvNsgOverride": restricted_i32("HIGGS_BONSAI_QMV_NSG", &[4, 8, 16, 32]),
        "bonsaiFastNsgOverride": restricted_i32("HIGGS_BONSAI_FAST_NSG", &[1, 2, 4, 8]),
        "bonsaiAlignedFastQmv": env("HIGGS_BONSAI_ALIGNED_FAST_QMV").as_deref() == Some("1"),
        "bonsaiWideQmm": env("HIGGS_BONSAI_QMM_WIDE").as_deref() == Some("1"),
        "bonsaiQmmMaxRows": optional_i32("HIGGS_BONSAI_QMM_MAX_ROWS").filter(|rows| (0..=64).contains(rows)).unwrap_or(8),
        "bonsaiQ2Simd": bonsai_q2_simd,
        "crossrowQmv": nonzero("HIGGS_CROSSROW_QMV", true),
        "selectedDecodeGemv": present("HIGGS_ENABLE_SELECTED_DECODE_GEMV"),
        "qgemvFfnMode": qgemv_ffn_mode,
        "qgemvNsgOverride": restricted_i32("HIGGS_QGEMV_NSG", &[4, 8, 16, 32]),
        "qgemvConfigCache": truthy("HIGGS_CACHE_QGEMV_CONFIGS"),
        "gatedDeltaConfigCache": flag("HIGGS_CACHE_GATED_DELTA_CONFIGS", true),
        "denseFfnGateUp": dense_ffn_gate_up,
        "moeFfnGateUp": truthy("HIGGS_MOE_FFN_GATE_UP"),
        "qgemmMxfp4": truthy("HIGGS_QGEMM_MXFP4"),
    }) else {
        unreachable!("JSON object literal must remain an object")
    };
    let Value::Object(dspark_runtime) = json!({
        "dsparkNativeVerify": env("HIGGS_DSPARK_NATIVE_VERIFY").as_deref() == Some("1"),
        "dsparkQ2Row2Mlp": nonzero("HIGGS_DSPARK_Q2_ROW2_MLP", true),
        "dsparkQ2HeadArgmax": nonzero("HIGGS_DSPARK_Q2_HEAD_ARGMAX", true),
        "dsparkTopkFast": optional_i32("HIGGS_DSPARK_TOPK_FAST").filter(|value| *value > 0),
        "dsparkTopkCompare": optional_i32("HIGGS_DSPARK_TOPK_COMPARE").filter(|value| *value > 0),
        "dsparkTopkBaseKernel": env("HIGGS_DSPARK_TOPK_BASE_KERNEL").as_deref() == Some("1"),
        "dsparkTopkMarkovKernel": env("HIGGS_DSPARK_TOPK_MARKOV_KERNEL").as_deref() == Some("1"),
    }) else {
        unreachable!("JSON object literal must remain an object")
    };
    identity.extend(model_runtime);
    identity.extend(dspark_runtime);
    ResolvedRuntimeIdentity(Value::Object(identity))
}

fn parse_enabled_flag(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

fn is_truthy(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "on" | "yes"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_serializes_the_typed_selection_without_auto_placeholders() {
        let selection = higgs_models::qwen3_next::ResolvedDenseRuntimeSelections {
            q2_simd_decode_policy: higgs_models::qwen3_next::Q2SimdDecodePolicy::EschaQwen38,
            gate_up_fused: false,
        };
        let identity =
            resolved_runtime_identity_with_selection(true, false, Some(selection), |_| None);
        let value = serde_json::to_value(identity).unwrap();
        assert_eq!(value["bonsaiQ2Simd"], "escha_qwen38");
        assert_eq!(value["denseFfnGateUp"], "separate");
    }
}
