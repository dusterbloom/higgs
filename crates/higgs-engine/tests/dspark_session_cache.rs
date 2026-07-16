//! Real-model release gate for session-paired dSpark caching.
//!
//! Manual:
//! ```text
//! HIGGS_DFLASH_TARGET_DIR=/path/to/Bonsai-27B-mlx-1bit \
//! HIGGS_DFLASH_DRAFTER_DIR=/path/to/dSpark-MLX \
//! cargo test -p higgs-engine --test dspark_session_cache -- --ignored --nocapture
//! ```

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::tests_outside_test_module,
    clippy::unwrap_used
)]

mod support;

use std::path::Path;

use higgs_engine::{
    chat_template::ChatMessage,
    mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile},
    simple::SimpleEngine,
};
use higgs_models::{SamplingParams, Speculation, turboquant::KvCacheConfig};
use support::{
    ReferenceDsparkEnv, assert_acceptance_within, assert_decode_tps_within, dflash_acceptance,
    dflash_decode_tps,
};

fn greedy(speculation: Speculation) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        speculation,
        ..SamplingParams::default()
    }
}

fn append_suffix(engine: &SimpleEngine, prefix: &[u32], text: &str) -> (Vec<u32>, usize) {
    let suffix = engine
        .tokenizer()
        .encode(text, false)
        .expect("encode suffix")
        .get_ids()
        .to_vec();
    let suffix_len = suffix.len();
    let mut extended = prefix.to_vec();
    extended.extend_from_slice(&suffix);
    (extended, suffix_len)
}

#[test]
#[ignore = "loads real Bonsai target + dSpark drafter; set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"]
fn bonsai_session_pair_resumes_suffix_only_and_demotes_atomically() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();
    let _reference_dspark = ReferenceDsparkEnv::install();
    let target = std::env::var("HIGGS_DFLASH_TARGET_DIR")
        .expect("set HIGGS_DFLASH_TARGET_DIR to the Bonsai target model");
    let drafter = std::env::var("HIGGS_DFLASH_DRAFTER_DIR")
        .expect("set HIGGS_DFLASH_DRAFTER_DIR to the MLX dSpark drafter");
    eprintln!("dspark-session checkpoint: loading target + drafter");
    let tuning = MlxRuntimeTuning::from_model_dir(Path::new(&target), RequestedMlxProfile::Auto);
    let engine = SimpleEngine::load_with_dflash(
        &target,
        KvCacheConfig::default(),
        tuning,
        false,
        Some(Path::new(&drafter)),
        None,
    )
    .expect("load paired dSpark engine");
    eprintln!("dspark-session checkpoint: engine loaded");

    let prompt = engine
        .prepare_chat_prompt_with_thinking(
            &[ChatMessage {
                role: "user".to_owned(),
                content: "Count upward from 1. Print only comma-separated integers.".to_owned(),
                tool_calls: None,
            }],
            None,
            false,
        )
        .expect("render no-thinking prompt");
    eprintln!(
        "dspark-session checkpoint: turn1 prompt rendered ({} tokens)",
        prompt.len()
    );

    const SID: u64 = 0xD5A4_0001;
    let first = engine
        .generate_continued_with_thinking(SID, &prompt, 1, &greedy(Speculation::DFlash), false)
        .expect("one-token paired turn");
    eprintln!("dspark-session checkpoint: turn1 complete");
    assert!(!first.continued);
    assert_eq!(first.prefilled_tokens as usize, prompt.len());
    assert_eq!(first.completion_tokens, 1);
    assert!(
        engine.last_dflash_accepts().is_empty(),
        "max_tokens=1 must perform no speculative rounds"
    );
    assert_eq!(
        engine.cache_stats().retained_paired_sessions,
        1,
        "the one-token cache-only forward must publish one complete pair"
    );

    let retained = engine
        .retained_session_tokens(SID)
        .expect("one-token turn must seal a retained pair");
    assert_eq!(
        retained.len(),
        prompt.len() + 1,
        "max_tokens=1 must cache-forward the visible non-EOS token before sealing"
    );

    let (second_prompt, second_suffix_len) =
        append_suffix(&engine, &retained, ", 2, 3, 4, 5, 6, 7, 8, 9, ");
    let second = engine
        .generate_continued_with_thinking(
            SID,
            &second_prompt,
            32,
            &greedy(Speculation::DFlash),
            false,
        )
        .expect("resume paired dSpark session");
    let second_accepts = engine.last_dflash_accepts();
    let second_acceptance = dflash_acceptance(&engine, "warm paired session");
    let second_decode_tps = dflash_decode_tps(&engine, "warm paired session");
    eprintln!(
        "dspark-session checkpoint: turn2 generated={} decode={second_decode_tps:.2} tok/s \
         accepts={second_accepts:?}",
        second.completion_tokens,
    );
    assert!(
        second.continued,
        "turn two must move-reuse both cache halves"
    );
    assert_eq!(
        second.prefilled_tokens as usize, second_suffix_len,
        "paired continuation must prefill only the appended suffix"
    );
    assert!(
        !second_accepts.is_empty(),
        "direct session dSpark must run without an MTP checkpoint"
    );
    assert_eq!(
        engine.cache_stats().retained_paired_sessions,
        1,
        "a resumed dSpark turn must replace the session with one sealed pair"
    );

    // Decode-only release gate against the identical session decoder with no
    // retained pair. `drop_retained_session` makes the comparator cold while
    // preserving the same prompt, sampling domain, and dSpark implementation.
    const COLD_SID: u64 = SID + 1;
    engine.drop_retained_session(COLD_SID);
    let cold = engine
        .generate_continued_with_thinking(
            COLD_SID,
            &second_prompt,
            32,
            &greedy(Speculation::DFlash),
            false,
        )
        .expect("cold session dSpark reference");
    let cold_acceptance = dflash_acceptance(&engine, "uncached session baseline");
    let cold_decode_tps = dflash_decode_tps(&engine, "uncached session baseline");
    assert!(
        !cold.continued,
        "the session performance baseline must begin without retained state"
    );
    assert_eq!(
        cold.prefilled_tokens, cold.prompt_tokens,
        "the session performance baseline must prefill its complete prompt"
    );
    assert_eq!(
        second.text, cold.text,
        "paired session reuse must preserve the cold greedy output"
    );
    assert_eq!(second.completion_tokens, cold.completion_tokens);
    assert_acceptance_within("warm paired session", second_acceptance, cold_acceptance);
    assert_decode_tps_within("warm paired session", second_decode_tps, cold_decode_tps);
    eprintln!(
        "dspark-session release gate: warm_decode={second_decode_tps:.2} \
         uncached_decode={cold_decode_tps:.2} tok/s warm_acceptance={:.2}% ({}/{}) \
         uncached_acceptance={:.2}% ({}/{})",
        second_acceptance.rate() * 100.0,
        second_acceptance.matched,
        second_acceptance.drafted,
        cold_acceptance.rate() * 100.0,
        cold_acceptance.matched,
        cold_acceptance.drafted,
    );
    engine.drop_retained_session(COLD_SID);

    let paired_tokens = engine
        .retained_session_tokens(SID)
        .expect("second turn must retain its pair");
    let (none_prompt, none_suffix_len) =
        append_suffix(&engine, &paired_tokens, "\nNow answer with one integer.");
    let none = engine
        .generate_continued_with_thinking(SID, &none_prompt, 1, &greedy(Speculation::None), false)
        .expect("explicit autoregressive continuation");
    assert!(none.continued, "none may reuse the target half by demotion");
    assert_eq!(none.prefilled_tokens as usize, none_suffix_len);
    assert_eq!(
        engine.cache_stats().retained_paired_sessions,
        0,
        "an execution path without taps must atomically discard the dSpark sidecar"
    );
    assert_eq!(
        engine.cache_stats().retained_sessions,
        1,
        "target-only continuity must remain retained after sidecar demotion"
    );

    let target_only_tokens = engine
        .retained_session_tokens(SID)
        .expect("autoregressive turn must retain target-only state");
    let (third_prompt, _) = append_suffix(
        &engine,
        &target_only_tokens,
        "\nContinue the sequence again.",
    );
    let third = engine
        .generate_continued_with_thinking(
            SID,
            &third_prompt,
            2,
            &greedy(Speculation::DFlash),
            false,
        )
        .expect("dSpark after target-only demotion");
    assert!(
        !third.continued,
        "a target-only cache cannot be combined with an independently reconstructed drafter"
    );
    assert_eq!(
        third.prefilled_tokens, third.prompt_tokens,
        "dSpark must cold-prefill after the sidecar was discarded"
    );
    assert_eq!(
        engine.cache_stats().retained_paired_sessions,
        1,
        "the cold dSpark retry must restore one complete retained pair"
    );
}
