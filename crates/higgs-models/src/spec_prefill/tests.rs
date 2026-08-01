//! Tests for the model-free PFlash selection half (steps 4-6).
//!
//! These prove the selection logic without loading any model — they cannot
//! OOM and need no GPU. The scorer half (step 1-3) gets its own gated tests
//! once implemented (DESIGN §5.4 asserts the ~75 MB memory bound).

#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::shadow_unrelated,
    clippy::shadow_reuse
)]

use super::*;
use mlx_rs::{Dtype, random};

fn needle_importance(s: usize, needle_pos: usize) -> Vec<f32> {
    // Flat low-importance background; one clearly salient needle block.
    let mut v = vec![0.01_f32; s];
    for i in needle_pos..(needle_pos + 16).min(s) {
        v[i] = 1.0;
    }
    v
}

#[test]
fn smooth_importance_is_length_preserving_and_peaks_at_needle() {
    let imp = needle_importance(512, 256);
    let sm = smooth_importance(&imp, 13).unwrap();
    assert_eq!(sm.len(), imp.len());
    // The smoothed peak is still at / next to the needle center.
    let peak = sm.iter().copied().fold(0.0_f32, f32::max);
    let peak_idx = sm.iter().position(|x| (*x - peak).abs() < 1e-6).unwrap();
    assert!(
        (240..=272).contains(&peak_idx),
        "peak at {peak_idx}, expected near needle (256)"
    );
}

#[test]
fn smooth_importance_rejects_even_kernel() {
    let imp = vec![0.0_f32; 8];
    assert!(smooth_importance(&imp, 12).is_err());
    assert!(smooth_importance(&imp, 0).is_err());
    assert!(smooth_importance(&imp, 1).is_ok());
}

#[test]
fn select_survivors_keeps_needle_block_at_aggressive_keep_ratio() {
    // The whole point of SpecPrefill-Full-LAH: a salient needle must be in the
    // survival mask even at keep_ratio = 0.10. (Our naive per-token scorer
    // failed exactly this — see RESEARCH §5.3. The block-max selection here is
    // what makes the needle survive once the scorer ranks it above background.)
    let s: usize = 4096;
    let chunk = 32;
    let needle_pos: usize = 2048; // mid-prompt
    let tokens: Vec<u32> = (0..s).map(|i| i as u32 % 1000).collect();
    let imp = needle_importance(s, needle_pos);
    let cfg = PrefillScoreConfig {
        keep_ratio: 0.10,
        chunk,
        avgpool: 13,
        lookahead: 8,
    };
    let plan = select_survivors(&tokens, &imp, &cfg).unwrap();
    assert_eq!(plan.source_token_count, s);
    assert_eq!(plan.metadata.version, PrefillPlanMetadata::VERSION);
    assert_eq!(plan.metadata.score_mode, PrefillScoreMode::Full);
    assert_eq!(plan.metadata.exit_layer, None);
    let kept_positions: std::collections::HashSet<i32> =
        plan.original_positions.iter().copied().collect();
    // Every needle token survives.
    for i in needle_pos..(needle_pos + 16) {
        assert!(
            kept_positions.contains(&(i as i32)),
            "needle token {i} dropped at keep=0.10"
        );
    }
    // ~10% keep ratio: plan length is near keep_ratio * s (plus the two forced blocks).
    let expected = (0.10 * s as f32) as usize;
    assert!(
        plan.len() <= expected + 2 * chunk,
        "plan kept {} tokens, expected ~{expected} (+2 forced blocks)",
        plan.len()
    );
}

#[test]
fn select_survivors_always_keeps_sink_and_final_token_blocks() {
    let s: usize = 1024;
    let tokens: Vec<u32> = (0..s).map(|i| i as u32).collect();
    // Importance concentrated in the middle — sink and tail would otherwise lose.
    let mut imp = vec![0.0_f32; s];
    for i in 400..500 {
        imp[i] = 1.0;
    }
    let cfg = PrefillScoreConfig {
        keep_ratio: 0.10,
        chunk: 32,
        avgpool: 13,
        lookahead: 8,
    };
    let plan = select_survivors(&tokens, &imp, &cfg).unwrap();
    let kept: std::collections::HashSet<i32> = plan.original_positions.iter().copied().collect();
    // First token (BOS / system-prompt anchor) and last token (sampled logits).
    assert!(kept.contains(&0), "sink token 0 dropped");
    assert!(kept.contains(&((s - 1) as i32)), "final token dropped");
}

#[test]
fn select_survivors_hard_keeps_exact_tokens_without_chunk_bloat() {
    let s: usize = 512;
    let tokens: Vec<u32> = (0..s).map(|i| i as u32).collect();
    let imp = vec![0.001_f32; s];
    let cfg = PrefillScoreConfig {
        keep_ratio: 0.02,
        chunk: 64,
        avgpool: 1,
        lookahead: 4,
    };

    let plan = select_survivors_with_hard_keep(&tokens, &imp, &cfg, &[HardKeepSpan::new(201, 204)])
        .unwrap();
    let kept: std::collections::HashSet<i32> = plan.original_positions.iter().copied().collect();

    assert!(kept.contains(&201));
    assert!(kept.contains(&202));
    assert!(kept.contains(&203));
    assert!(
        !kept.contains(&200),
        "hard-keeping a narrow span should not force the whole chunk"
    );
    assert!(
        !kept.contains(&204),
        "hard-keeping a narrow span should remain half-open"
    );
}

#[test]
fn select_survivors_preserves_original_order_and_positions() {
    let s: usize = 256;
    let tokens: Vec<u32> = (1000..(1000 + s)).map(|x| x as u32).collect();
    let imp = needle_importance(s, 128);
    let plan = select_survivors(&tokens, &imp, &PrefillScoreConfig::default()).unwrap();
    assert_eq!(plan.source_token_count, s);
    assert!(!plan.is_contiguous_identity());
    // Positions strictly increasing; token ids match tokens[position].
    assert!(plan.original_positions.windows(2).all(|w| w[0] < w[1]));
    for (tok, pos) in plan.token_ids.iter().zip(plan.original_positions.iter()) {
        assert_eq!(*tok, tokens[*pos as usize]);
    }
}

#[test]
fn adaptive_keep_ratio_tracks_importance_entropy() {
    let sharp = {
        let mut scores = vec![0.001_f32; 128];
        scores[64] = 1.0;
        scores
    };
    let diffuse = vec![1.0_f32; 128];

    let sharp_keep = adaptive_keep_ratio_from_importance(&sharp, 0.10, 0.75);
    let diffuse_keep = adaptive_keep_ratio_from_importance(&diffuse, 0.10, 0.75);

    assert!(
        sharp_keep < 0.35,
        "sharp importance should compress aggressively, got {sharp_keep}"
    );
    assert!(
        diffuse_keep > 0.70,
        "diffuse importance should keep near ceiling, got {diffuse_keep}"
    );
}

#[test]
fn target_sparse_prefill_plan_tracks_logical_source_position() {
    let plan = SurvivalPlan {
        token_ids: vec![10, 11, 99],
        original_positions: vec![0, 16, 63],
        source_token_count: 64,
        metadata: PrefillPlanMetadata::from_config(&PrefillScoreConfig::default()),
    }
    .with_scorer(PrefillScoreMode::L7, Some(7));

    let target = plan.target_sparse_prefill_plan().unwrap();
    assert_eq!(target.token_ids, &[10, 11, 99]);
    assert_eq!(target.original_positions, &[0, 16, 63]);
    assert_eq!(target.logical_next_pos, 64);
    assert!(!target.is_contiguous_identity());
}

#[test]
fn target_sparse_prefill_plan_accepts_contiguous_identity() {
    let plan = SurvivalPlan {
        token_ids: vec![1, 2, 3, 4],
        original_positions: vec![0, 1, 2, 3],
        source_token_count: 4,
        metadata: PrefillPlanMetadata::from_config(&PrefillScoreConfig::default()),
    };

    let target = plan.target_sparse_prefill_plan().unwrap();
    assert_eq!(target.logical_next_pos, 4);
    assert!(target.is_contiguous_identity());
}

#[test]
fn survival_plan_identity_keeps_every_suffix_token() {
    let metadata = PrefillPlanMetadata::from_config(&PrefillScoreConfig::default());
    let plan = SurvivalPlan::identity(&[7, 8, 9], metadata).unwrap();

    assert_eq!(plan.token_ids, &[7, 8, 9]);
    assert_eq!(plan.original_positions, &[0, 1, 2]);
    assert_eq!(plan.source_token_count, 3);
    assert!(
        plan.target_sparse_prefill_plan()
            .unwrap()
            .is_contiguous_identity()
    );
}

#[test]
fn survival_plan_append_suffix_offsets_positions() {
    let metadata = PrefillPlanMetadata::from_config(&PrefillScoreConfig::default());
    let prefix = SurvivalPlan {
        token_ids: vec![10, 99],
        original_positions: vec![0, 7],
        source_token_count: 8,
        metadata: metadata.clone(),
    };
    let suffix = SurvivalPlan::identity(&[20, 21, 22], metadata.clone()).unwrap();

    let combined = prefix.append_suffix(&suffix, metadata).unwrap();
    assert_eq!(combined.token_ids, &[10, 99, 20, 21, 22]);
    assert_eq!(combined.original_positions, &[0, 7, 8, 9, 10]);
    assert_eq!(combined.source_token_count, 11);
    let target = combined.target_sparse_prefill_plan().unwrap();
    assert_eq!(target.logical_next_pos, 11);
}

#[test]
fn target_sparse_prefill_plan_rejects_invalid_positions() {
    let base = SurvivalPlan {
        token_ids: vec![1, 2, 3],
        original_positions: vec![0, 2, 4],
        source_token_count: 5,
        metadata: PrefillPlanMetadata::from_config(&PrefillScoreConfig::default()),
    };
    assert!(base.target_sparse_prefill_plan().is_ok());

    let mut mismatch = base.clone();
    mismatch.original_positions.pop();
    assert!(mismatch.target_sparse_prefill_plan().is_err());

    let mut unsorted = base.clone();
    unsorted.original_positions = vec![0, 4, 2];
    assert!(unsorted.target_sparse_prefill_plan().is_err());

    let mut negative = base.clone();
    negative.original_positions = vec![0, -1, 4];
    assert!(negative.target_sparse_prefill_plan().is_err());

    let mut out_of_bounds = base.clone();
    out_of_bounds.original_positions = vec![0, 2, 5];
    assert!(out_of_bounds.target_sparse_prefill_plan().is_err());

    let mut drops_sink = base.clone();
    drops_sink.original_positions = vec![1, 2, 4];
    assert!(drops_sink.target_sparse_prefill_plan().is_err());

    let mut drops_final_token = base;
    drops_final_token.original_positions = vec![0, 2, 3];
    assert!(drops_final_token.target_sparse_prefill_plan().is_err());
}

#[test]
fn target_sparse_prefill_plan_rejects_logical_position_overflow() {
    let plan = SurvivalPlan {
        token_ids: Vec::new(),
        original_positions: Vec::new(),
        source_token_count: i32::MAX as usize + 1,
        metadata: PrefillPlanMetadata::from_config(&PrefillScoreConfig::default()),
    };

    assert!(plan.target_sparse_prefill_plan().is_err());
}

#[test]
fn select_survivors_rejects_bad_inputs() {
    let t = vec![0_u32; 4];
    let i = vec![0.0_f32; 4];
    // length mismatch
    assert!(select_survivors(&t, &i[..3], &PrefillScoreConfig::default()).is_err());
    // keep_ratio out of range
    let bad = PrefillScoreConfig {
        keep_ratio: 0.96,
        ..PrefillScoreConfig::default()
    };
    assert!(select_survivors(&t, &i, &bad).is_err());
    // chunk = 0
    let bad_chunk = PrefillScoreConfig {
        chunk: 0,
        ..PrefillScoreConfig::default()
    };
    assert!(select_survivors(&t, &i, &bad_chunk).is_err());
}

#[test]
fn layer_importance_shape_and_range() {
    // Memory-safety smoke: the scorer must produce [S] from [H, lah, d] x
    // [Hkv, S, d] without materializing [H, S, S]. S=2048 is enough to OOM the
    // naive form; this runs in microseconds.
    let n_heads = 16;
    let n_kv_heads = 8;
    let head_dim = 128;
    let lah = 9; // lookahead + final prompt token
    let s = 2048;
    let q = random::uniform::<f32, f32>(0.0, 1.0, &[n_heads, lah, head_dim], None).unwrap();
    let k = random::uniform::<f32, f32>(0.0, 1.0, &[n_kv_heads, s, head_dim], None).unwrap();
    let imp = super::layer_importance(&q, &k, n_heads, n_kv_heads, head_dim, 0.0884).unwrap();
    assert_eq!(
        imp.shape(),
        &[s],
        "importance must be [S], got {:?}",
        imp.shape()
    );
    // softmax outputs are in [0, 1]; the mean-over-lah of max-over-heads stays so.
    let vals = imp.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::transforms::eval([&vals]).unwrap();
    let slice = vals.as_slice::<f32>();
    assert!(
        slice.iter().all(|x| *x >= 0.0 && *x <= 1.0),
        "importance out of [0,1]"
    );
}
