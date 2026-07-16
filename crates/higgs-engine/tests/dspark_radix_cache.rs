//! Real-model release gate for radix-paired dSpark caching.
//!
//! Manual:
//! ```text
//! HIGGS_DFLASH_TARGET_DIR=/path/to/Bonsai-27B-mlx-1bit \
//! HIGGS_DFLASH_DRAFTER_DIR=/path/to/dSpark-MLX \
//! cargo test -p higgs-engine --release --test dspark_radix_cache \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! MLX/Metal tests must run serially. The test intentionally uses only public
//! engine observability: radix entry count, hit/saved-token counters, dSpark
//! acceptance telemetry, and cache clear.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::tests_outside_test_module
)]

use std::path::Path;

use higgs_engine::{
    chat_template::{ChatMessage, ChatTemplateRenderer},
    mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile},
    simple::SimpleEngine,
};
use higgs_models::{SamplingParams, Speculation, turboquant::KvCacheConfig};

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl ScopedEnvVar {
    #[allow(unsafe_code)]
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: This ignored real-model gate is documented and asserted to
        // run alone (`--test-threads=1`). The guard restores the process-global
        // setting even if the test unwinds.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: See `ScopedEnvVar::set`; restoration happens in the same
        // serial ignored test that performed the mutation.
        unsafe {
            if let Some(previous) = self.previous.take() {
                std::env::set_var(self.key, previous);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct DflashAcceptance {
    matched: u64,
    drafted: u64,
}

impl DflashAcceptance {
    fn rate(self) -> f64 {
        self.matched as f64 / self.drafted as f64
    }
}

fn dflash_acceptance(engine: &SimpleEngine, label: &str) -> DflashAcceptance {
    let matches = engine.last_dflash_draft_matches();
    let draft_counts = engine.last_dflash_draft_counts();
    assert_eq!(
        matches.len(),
        draft_counts.len(),
        "{label}: every speculative round must report both matched and drafted counts"
    );
    let acceptance = DflashAcceptance {
        matched: matches.into_iter().map(u64::from).sum(),
        drafted: draft_counts.into_iter().map(u64::from).sum(),
    };
    assert!(
        acceptance.drafted > 0,
        "{label}: the acceptance gate requires at least one drafted token"
    );
    assert!(
        acceptance.matched <= acceptance.drafted,
        "{label}: matched tokens cannot exceed drafted tokens"
    );
    acceptance
}

fn assert_acceptance_within(
    candidate_label: &str,
    candidate: DflashAcceptance,
    baseline: DflashAcceptance,
) {
    const MAX_REGRESSION: f64 = 0.03;
    assert!(
        candidate.rate() + MAX_REGRESSION >= baseline.rate(),
        "{candidate_label} aggregate dFlash acceptance regressed by more than \
         3 percentage points: candidate={:.2}% ({}/{}) baseline={:.2}% ({}/{})",
        candidate.rate() * 100.0,
        candidate.matched,
        candidate.drafted,
        baseline.rate() * 100.0,
        baseline.matched,
        baseline.drafted
    );
}

fn user(content: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_owned(),
        content: content.to_owned(),
        tool_calls: None,
    }
}

fn assistant(content: String) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_owned(),
        content,
        tool_calls: None,
    }
}

fn greedy_dflash() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        speculation: Speculation::DFlash,
        ..SamplingParams::default()
    }
}

fn greedy_ar() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        speculation: Speculation::None,
        ..SamplingParams::default()
    }
}

/// Mirror `SimpleEngine`'s load-time exact generation-suffix proof.
fn generation_suffix(engine: &SimpleEngine, renderer: &ChatTemplateRenderer) -> Vec<u32> {
    let probe = [user("x")];
    let with_generation = renderer
        .apply_with_thinking(&probe, None, true, false)
        .expect("render suffix probe with generation prompt");
    let without_generation = renderer
        .apply_with_thinking(&probe, None, false, false)
        .expect("render suffix probe without generation prompt");
    let with_tokens = engine
        .tokenizer()
        .encode(with_generation, false)
        .expect("tokenize suffix probe with generation prompt");
    let without_tokens = engine
        .tokenizer()
        .encode(without_generation, false)
        .expect("tokenize suffix probe without generation prompt");
    with_tokens
        .get_ids()
        .strip_prefix(without_tokens.get_ids())
        .expect("generation prompt must be an exact token suffix")
        .to_vec()
}

#[test]
#[ignore = "loads real Bonsai target + dSpark drafter; set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"]
fn bonsai_radix_pair_reuses_only_conversation_body_and_clear_restores_cold() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();
    let _prefix_cache_enabled = ScopedEnvVar::set("HIGGS_PREFIX_CACHE", "1");
    let target = std::env::var("HIGGS_DFLASH_TARGET_DIR")
        .expect("set HIGGS_DFLASH_TARGET_DIR to the Bonsai target model");
    let drafter = std::env::var("HIGGS_DFLASH_DRAFTER_DIR")
        .expect("set HIGGS_DFLASH_DRAFTER_DIR to the MLX dSpark drafter");
    let target_path = Path::new(&target);
    let renderer =
        ChatTemplateRenderer::from_model_dir(target_path).expect("load target chat template");
    let tuning = MlxRuntimeTuning::from_model_dir(target_path, RequestedMlxProfile::Auto);

    eprintln!("dspark-radix checkpoint: loading target + drafter");
    let engine = SimpleEngine::load_with_dflash(
        target_path,
        KvCacheConfig::default(),
        tuning,
        false,
        Some(Path::new(&drafter)),
        None,
    )
    .expect("load paired dSpark engine");
    let params = greedy_dflash();
    let generation_suffix = generation_suffix(&engine, &renderer);
    assert!(
        !generation_suffix.is_empty(),
        "this gate requires a chat template with a non-empty generation suffix"
    );

    engine.clear_prefix_cache();
    assert_eq!(engine.prefix_cache_len(), 0);
    let before_first = engine.cache_stats();
    let first_messages = [user(
        "Print the integers from 1 upward as comma-separated values. \
         Output only the sequence and continue for many terms.",
    )];
    let first_prompt = engine
        .prepare_chat_prompt_with_thinking(&first_messages, None, false)
        .expect("render first no-thinking prompt");
    let first_body_len = first_prompt
        .strip_suffix(generation_suffix.as_slice())
        .expect("first prompt must end in the proven generation suffix")
        .len();
    assert!(first_body_len > 0);

    let first_started = std::time::Instant::now();
    let first = engine
        .generate_with_thinking(
            &first_prompt,
            32,
            &params,
            &[],
            false,
            None,
            false,
            None,
            None,
            None,
        )
        .expect("cold paired dSpark request");
    let first_wall = first_started.elapsed();
    let after_first = engine.cache_stats();
    assert!(
        after_first.radix_entries > 0,
        "the cold request must publish a reusable paired radix endpoint"
    );
    assert_eq!(
        after_first.paired_radix_entries, 1,
        "the first request must publish exactly one atomic target+dSpark endpoint"
    );
    assert!(
        after_first.paired_radix_target_bytes > 0 && after_first.paired_radix_dflash_bytes > 0,
        "paired accounting must include both frozen cache halves"
    );
    assert_eq!(
        after_first.radix_lookups - before_first.radix_lookups,
        1,
        "the cold request must perform exactly one paired radix lookup"
    );
    assert_eq!(
        after_first.paired_radix_lookups - before_first.paired_radix_lookups,
        1,
        "the cold request must perform exactly one paired-capability lookup"
    );
    assert_eq!(
        after_first.radix_hits, before_first.radix_hits,
        "an empty radix cannot report a paired hit"
    );
    assert_eq!(
        after_first.paired_radix_hits, before_first.paired_radix_hits,
        "an empty radix cannot report a paired-capability hit"
    );
    assert_eq!(
        after_first.prefill_saved_tokens, before_first.prefill_saved_tokens,
        "the first request must prefill cold"
    );

    let second_messages = [
        first_messages[0].clone(),
        assistant(first.text.clone()),
        user("Continue the same sequence. Output only comma-separated integers."),
    ];
    let second_prompt = engine
        .prepare_chat_prompt_with_thinking(&second_messages, None, false)
        .expect("render related second no-thinking prompt");
    assert!(
        second_prompt
            .strip_suffix(generation_suffix.as_slice())
            .is_some(),
        "turn two must end in the exact no-thinking generation suffix"
    );
    assert_eq!(
        second_prompt.get(..first_body_len),
        first_prompt.get(..first_body_len),
        "the first conversation body must be an exact token prefix of turn two"
    );
    let expected_second_prefill = second_prompt
        .len()
        .checked_sub(first_body_len)
        .expect("paired body cannot exceed the second prompt");

    let before_warm = engine.cache_stats();
    let warm_started = std::time::Instant::now();
    let warm = engine
        .generate_with_thinking(
            &second_prompt,
            48,
            &params,
            &[],
            false,
            None,
            false,
            None,
            None,
            None,
        )
        .expect("warm paired dSpark request");
    let warm_wall = warm_started.elapsed();
    let warm_accepts = engine.last_dflash_accepts();
    let warm_acceptance = dflash_acceptance(&engine, "warm paired radix");
    let after_warm = engine.cache_stats();
    let saved = after_warm.prefill_saved_tokens - before_warm.prefill_saved_tokens;
    eprintln!(
        "dspark-radix warm: prompt={} body_reused={} prefilled={} wall={warm_wall:.2?} accepts={warm_accepts:?}",
        second_prompt.len(),
        saved,
        expected_second_prefill
    );
    assert_eq!(
        after_warm.radix_hits - before_warm.radix_hits,
        1,
        "turn two must reuse one exact paired radix endpoint"
    );
    assert_eq!(
        after_warm.paired_radix_lookups - before_warm.paired_radix_lookups,
        1,
        "turn two must perform one paired-capability lookup"
    );
    assert_eq!(
        after_warm.paired_radix_hits - before_warm.paired_radix_hits,
        1,
        "turn two must materialize one complete target+dSpark pair"
    );
    assert_eq!(
        saved,
        u64::try_from(first_body_len).expect("body length fits u64"),
        "paired reuse must stop before the old generation-prompt suffix"
    );
    assert_eq!(
        second_prompt.len() - usize::try_from(saved).expect("saved count fits usize"),
        expected_second_prefill,
        "turn two must prefill only the conversation remainder plus its generation suffix"
    );
    assert!(
        !warm_accepts.is_empty(),
        "the reused dSpark branch must enter speculative rounds"
    );

    engine.clear_prefix_cache();
    assert_eq!(
        engine.prefix_cache_len(),
        0,
        "clear must atomically remove target and dSpark radix state"
    );
    let after_clear = engine.cache_stats();
    assert_eq!(after_clear.paired_radix_entries, 0);
    assert_eq!(after_clear.paired_radix_target_bytes, 0);
    assert_eq!(after_clear.paired_radix_dflash_bytes, 0);
    let before_cold = engine.cache_stats();
    let cold_started = std::time::Instant::now();
    let cold = engine
        .generate_with_thinking(
            &second_prompt,
            48,
            &params,
            &[],
            false,
            None,
            false,
            None,
            None,
            None,
        )
        .expect("cold-after-clear paired dSpark request");
    let cold_wall = cold_started.elapsed();
    let cold_accepts = engine.last_dflash_accepts();
    let cold_acceptance = dflash_acceptance(&engine, "cold paired split");
    let after_cold = engine.cache_stats();
    assert_eq!(
        after_cold.radix_hits, before_cold.radix_hits,
        "clear must restore a genuine cold lookup"
    );
    assert_eq!(
        after_cold.paired_radix_hits, before_cold.paired_radix_hits,
        "cold-after-clear cannot report a paired-capability hit"
    );
    assert_eq!(
        after_cold.prefill_saved_tokens, before_cold.prefill_saved_tokens,
        "cold-after-clear must not claim saved prefill tokens"
    );
    assert!(
        after_cold.radix_entries > 0,
        "the cold-after-clear request must republish its paired body"
    );
    assert_eq!(
        after_cold.paired_radix_entries, 1,
        "cold-after-clear must republish one complete target+dSpark endpoint"
    );
    assert!(
        !cold_accepts.is_empty(),
        "the cold reference must still exercise dSpark"
    );
    assert_eq!(
        warm.text, cold.text,
        "greedy no-thinking paired reuse must match the identical cold prompt"
    );
    assert_eq!(warm.completion_tokens, cold.completion_tokens);
    assert_eq!(warm.finish_reason, cold.finish_reason);

    engine.clear_prefix_cache();
    assert_eq!(engine.prefix_cache_len(), 0);
    let before_legacy = engine.cache_stats();
    let (legacy, legacy_wall, legacy_acceptance) = {
        let _prefix_cache_disabled = ScopedEnvVar::set("HIGGS_PREFIX_CACHE", "0");
        let legacy_started = std::time::Instant::now();
        let legacy = engine
            .generate_with_thinking(
                &second_prompt,
                48,
                &params,
                &[],
                false,
                None,
                false,
                None,
                None,
                None,
            )
            .expect("legacy one-shot dSpark request with paired cache disabled");
        let legacy_wall = legacy_started.elapsed();
        let legacy_acceptance = dflash_acceptance(&engine, "legacy one-shot dSpark");
        (legacy, legacy_wall, legacy_acceptance)
    };
    let after_legacy = engine.cache_stats();
    assert_eq!(
        engine.prefix_cache_len(),
        0,
        "the cache-disabled legacy request must not publish a radix endpoint"
    );
    assert_eq!(
        after_legacy.radix_lookups, before_legacy.radix_lookups,
        "the cache-disabled legacy request must bypass radix lookup"
    );
    assert_eq!(
        after_legacy.paired_radix_lookups, before_legacy.paired_radix_lookups,
        "the cache-disabled legacy request must bypass paired-capability lookup"
    );
    assert_eq!(
        cold.text, legacy.text,
        "the cache-disabled one-shot dSpark path must preserve exact greedy output"
    );
    assert_eq!(cold.completion_tokens, legacy.completion_tokens);
    assert_eq!(cold.finish_reason, legacy.finish_reason);
    assert_acceptance_within("warm paired radix", warm_acceptance, legacy_acceptance);
    assert_acceptance_within("cold paired split", cold_acceptance, legacy_acceptance);
    eprintln!(
        "dspark-radix acceptance: warm={:.2}% ({}/{}) cold={:.2}% ({}/{}) legacy={:.2}% ({}/{})",
        warm_acceptance.rate() * 100.0,
        warm_acceptance.matched,
        warm_acceptance.drafted,
        cold_acceptance.rate() * 100.0,
        cold_acceptance.matched,
        cold_acceptance.drafted,
        legacy_acceptance.rate() * 100.0,
        legacy_acceptance.matched,
        legacy_acceptance.drafted
    );

    engine.clear_prefix_cache();
    let ar_started = std::time::Instant::now();
    let ar = engine
        .generate_with_thinking(
            &second_prompt,
            48,
            &greedy_ar(),
            &[],
            false,
            None,
            false,
            None,
            None,
            None,
        )
        .expect("greedy autoregressive reference");
    let ar_wall = ar_started.elapsed();
    assert_eq!(
        cold.text, ar.text,
        "cold dSpark verification must preserve greedy target token decisions"
    );
    assert_eq!(cold.completion_tokens, ar.completion_tokens);
    assert_eq!(cold.finish_reason, ar.finish_reason);
    eprintln!(
        "dspark-radix wall: initial={first_wall:.2?} warm={warm_wall:.2?} \
         cold_after_clear={cold_wall:.2?} legacy_one_shot={legacy_wall:.2?} ar={ar_wall:.2?}"
    );
}
