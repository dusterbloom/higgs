use std::collections::HashMap;

use higgs_engine::runtime_identity::{ResolvedRuntimeIdentity, resolved_runtime_identity_with};

fn identity_with(overrides: &[(&str, &str)]) -> ResolvedRuntimeIdentity {
    let values = overrides.iter().copied().collect::<HashMap<_, _>>();
    resolved_runtime_identity_with(true, false, |name| {
        values.get(name).map(|value| (*value).to_owned())
    })
}

#[test]
fn every_runtime_or_layout_mutation_changes_the_capacity_identity() {
    let baseline = identity_with(&[]);
    let mutations = [
        ("HIGGS_PREFIX_CACHE", "0"),
        ("HIGGS_EXPERIMENTAL_PAGED_KV", "1"),
        ("HIGGS_PROMPT_LOOKUP", "1"),
        ("HIGGS_PROMPT_LOOKUP_UNCHECKED", "1"),
        ("HIGGS_PROMPT_LOOKUP_DRAFT_N_MAX", "97"),
        ("HIGGS_PROMPT_LOOKUP_NGRAM_MAX", "97"),
        ("HIGGS_PROMPT_LOOKUP_WINDOW", "97"),
        ("HIGGS_PREFLASH_FULL_SCORE_MAX_TOKENS", "4096"),
        ("HIGGS_PREFLASH_MIN_FREE_MB", "1024"),
        ("HIGGS_MTP_ADAPTIVE_DRAFT", "1"),
        ("HIGGS_MTP_PROMPT_LOOKUP", "1"),
        ("HIGGS_MTP_PRIME_PREFILL", "0"),
        ("HIGGS_DFLASH_BLOCK_SIZE", "4"),
        ("HIGGS_DSPARK_DRAFT_CAP", "3"),
        ("HIGGS_DFLASH_VERIFY_MODE", "block"),
        ("HIGGS_DSPARK_TARGET_HEAD", "1"),
        ("HIGGS_DFLASH_ADAPTIVE", "0"),
        ("HIGGS_DFLASH_GATE", "0"),
        ("HIGGS_DFLASH_MIN_BLOCK", "4"),
        ("HIGGS_DFLASH_CONF_TRUNC", "0.75"),
        ("HIGGS_DFLASH_FUSED_CONV", "1"),
        ("HIGGS_DFLASH_GDN_CONFIG_CACHE", "1"),
        ("HIGGS_DENSE_REQUANT_8BIT", "1"),
        ("HIGGS_SEPARATE_GDN_PROJ", "1"),
        ("HIGGS_LOAD_EVAL_CHUNKED", "0"),
        ("HIGGS_MLA_LATENT_CACHE", "1"),
        ("HIGGS_TURBOQUANT_MIN_TOKENS", "0"),
        ("HIGGS_COMPILED_GATING", "0"),
        ("HIGGS_COMPILED_GDN_DECODE", "1"),
        ("HIGGS_ASYNC_LAYER_STATE_EVAL", "0"),
        ("HIGGS_BONSAI_SYMMETRIC_Q1", "0"),
        ("HIGGS_BONSAI_TG_LUT4", "0"),
        ("HIGGS_BONSAI_TG_LUT4_FUSED_MLP", "1"),
        ("HIGGS_BONSAI_TG_LUT4_M5_WG", "224"),
        ("HIGGS_BONSAI_QMV_KERNEL", "legacy"),
        ("HIGGS_BONSAI_QMV_NSG", "16"),
        ("HIGGS_BONSAI_FAST_NSG", "4"),
        ("HIGGS_BONSAI_ALIGNED_FAST_QMV", "1"),
        ("HIGGS_BONSAI_QMM_WIDE", "1"),
        ("HIGGS_BONSAI_QMM_MAX_ROWS", "16"),
        ("HIGGS_BONSAI_Q2_SIMD", "1"),
        ("HIGGS_CROSSROW_QMV", "0"),
        ("HIGGS_ENABLE_SELECTED_DECODE_GEMV", "1"),
        ("HIGGS_QGEMV_FFN_MODE", "off"),
        ("HIGGS_QGEMV_NSG", "16"),
        ("HIGGS_CACHE_QGEMV_CONFIGS", "1"),
        ("HIGGS_CACHE_GATED_DELTA_CONFIGS", "0"),
        ("HIGGS_DENSE_FFN_GATE_UP", "separate"),
        ("HIGGS_MOE_FFN_GATE_UP", "1"),
        ("HIGGS_QGEMM_MXFP4", "1"),
        ("HIGGS_DSPARK_NATIVE_VERIFY", "1"),
        ("HIGGS_DSPARK_Q2_ROW2_MLP", "0"),
        ("HIGGS_DSPARK_Q2_HEAD_ARGMAX", "0"),
        ("HIGGS_DSPARK_TOPK_FAST", "32"),
        ("HIGGS_DSPARK_TOPK_COMPARE", "16"),
        ("HIGGS_DSPARK_TOPK_BASE_KERNEL", "1"),
        ("HIGGS_DSPARK_TOPK_MARKOV_KERNEL", "1"),
        ("HIGGS_ESCHA_NATIVE", "0"),
        ("HIGGS_ESCHA_AFFINE_BITS", "3"),
        ("HIGGS_ESCHA_TRELLIS_GEMM", "1"),
        ("HIGGS_ESCHA_QGEMM_SIMD", "0"),
        ("HIGGS_ESCHA_QGEMM_BM", "64"),
        ("HIGGS_NO_MEM_LIMIT", "1"),
        ("HIGGS_WIRED_LIMIT_MODE", "legacy"),
    ];

    for (name, value) in mutations {
        assert_ne!(
            identity_with(&[(name, value)]),
            baseline,
            "{name} must participate in the canonical runtime identity"
        );
    }
}

#[test]
fn runtime_identity_normalizes_equivalent_inputs_and_model_context() {
    assert_eq!(
        identity_with(&[("HIGGS_COMPILED_GATING", " off ")]),
        identity_with(&[("HIGGS_COMPILED_GATING", "0")])
    );
    assert_eq!(
        identity_with(&[("HIGGS_DFLASH_VERIFY_MODE", "sequential")]),
        identity_with(&[("HIGGS_DFLASH_VERIFY_MODE", "canonical")])
    );
    assert_eq!(
        identity_with(&[("HIGGS_NO_MEM_LIMIT", "0")]),
        identity_with(&[("HIGGS_NO_MEM_LIMIT", "1")]),
        "allocator limits are disabled by variable presence, independent of its value"
    );
    assert_eq!(
        identity_with(&[("HIGGS_WIRED_LIMIT_MODE", "safe")]),
        identity_with(&[("HIGGS_WIRED_LIMIT_MODE", "legacy")])
    );
    assert_eq!(
        identity_with(&[("HIGGS_WIRED_LIMIT_MODE", "caps")]),
        identity_with(&[("HIGGS_WIRED_LIMIT_MODE", "legacy")])
    );
    assert_eq!(
        identity_with(&[("HIGGS_WIRED_LIMIT_MODE", "unknown")]),
        identity_with(&[]),
        "unknown wired modes resolve to the default MLX wired-limit policy"
    );
    assert_eq!(
        identity_with(&[
            ("HIGGS_NO_MEM_LIMIT", "1"),
            ("HIGGS_WIRED_LIMIT_MODE", "legacy"),
        ]),
        identity_with(&[("HIGGS_NO_MEM_LIMIT", "1")]),
        "disabled allocator limits override the legacy/wired selection"
    );

    let regular = resolved_runtime_identity_with(false, false, |_| None);
    let escha = resolved_runtime_identity_with(true, false, |_| None);
    let mla = resolved_runtime_identity_with(false, true, |_| None);
    assert_ne!(regular, escha, "model family changes executable layout");
    assert_ne!(regular, mla, "MLA cache representation changes memory cost");
}

#[test]
fn malformed_numeric_values_resolve_to_runtime_defaults() {
    let baseline = identity_with(&[]);
    for name in [
        "HIGGS_DFLASH_BLOCK_SIZE",
        "HIGGS_DSPARK_DRAFT_CAP",
        "HIGGS_DFLASH_MIN_BLOCK",
        "HIGGS_QGEMV_NSG",
        "HIGGS_BONSAI_QMV_NSG",
    ] {
        assert_eq!(
            identity_with(&[(name, " 4096 ")]),
            baseline,
            "{name} rejects values that its runtime resolver rejects"
        );
    }
}
