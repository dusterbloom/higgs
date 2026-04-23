//! Threaded variant of [`crate::diffusion::speculative_generate_next`].
//!
//! Structural refactor: drafter lives on a dedicated scoped thread, communicating
//! with the main (verifier) thread via mpsc. Algorithmically identical to the
//! serial implementation — same accept-longest-prefix, same fast/slow cache
//! advance, same adaptive K. The payoff is latent: on GPU-only both threads
//! serialize on the MLX command stream (no win). When the drafter runs on ANE
//! and the ANE worker's per-dispatch GPU sync is reduced, drafter work overlaps
//! with verifier GPU work and draft latency hides behind verify latency.
//!
//! Gated by `HIGGS_SPEC_DECODE_PIPELINE=1`. Default is off → caller uses the
//! serial implementation unchanged.

use std::sync::mpsc::{Receiver, Sender};

use mlx_rs::ops::indexing::{self as ix, IndexOp};

use crate::diffusion::{accept_prefix, AdaptiveKController, QwenNextCausalDrafter};
use crate::qwen3_next::LayerCache;

enum DraftReq {
    Prefill {
        prompt: Vec<u32>,
        k: usize,
    },
    AdvanceAndDraft {
        accepted: Vec<u32>,
        prev_drafts: Vec<u32>,
        k: usize,
    },
}

enum DraftResp {
    Drafts(Vec<u32>),
    Err(String),
}

fn produce_drafts(
    drafter: &mut QwenNextCausalDrafter,
    cache: &mut Vec<Option<LayerCache>>,
    saved_logits: &mlx_rs::Array,
    k: usize,
    max_seq: usize,
    context_len: usize,
) -> Result<(Vec<u32>, Vec<Option<LayerCache>>), String> {
    let draft_0 = ix::argmax_axis(saved_logits, -1, false)
        .map_err(|e| format!("argmax saved: {e}"))?
        .index((0, 0))
        .item::<i32>() as u32;
    let snapshot = cache.clone();
    let mut drafts = vec![draft_0];
    let mut last = draft_0;
    for _ in 1..k {
        if context_len + drafts.len() >= max_seq {
            break;
        }
        let next = drafter.step(last, cache).map_err(|e| format!("step: {e}"))?;
        drafts.push(next);
        last = next;
    }
    Ok((drafts, snapshot))
}

fn drafter_thread_loop(
    drafter: &mut QwenNextCausalDrafter,
    rx: &Receiver<DraftReq>,
    tx: &Sender<DraftResp>,
) {
    let max_seq = drafter.max_seq;
    let mut d_cache: Vec<Option<LayerCache>> = drafter.model.make_cache();
    let mut saved_logits: Option<mlx_rs::Array> = None;
    let mut d_snapshot: Option<Vec<Option<LayerCache>>> = None;
    let mut context_len: usize = 0;

    while let Ok(req) = rx.recv() {
        match req {
            DraftReq::Prefill { prompt, k } => {
                let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
                let prompt_arr =
                    mlx_rs::Array::from_slice(&prompt_i32, &[1, prompt_i32.len() as i32]);
                d_cache = drafter.model.make_cache();
                let logits = match drafter.model.forward(&prompt_arr, None, &mut d_cache) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.send(DraftResp::Err(format!("drafter prefill: {e}")));
                        return;
                    }
                };
                if let Err(e) = mlx_rs::transforms::eval([&logits]) {
                    let _ = tx.send(DraftResp::Err(format!("eval prefill: {e}")));
                    return;
                }
                if let Err(e) = drafter.eval_cache_with(&d_cache, &[&logits]) {
                    let _ = tx.send(DraftResp::Err(format!("eval prefill cache: {e}")));
                    return;
                }
                context_len = prompt.len();
                saved_logits = Some(logits);
                let logits_ref = saved_logits.as_ref().expect("just set");
                match produce_drafts(drafter, &mut d_cache, logits_ref, k, max_seq, context_len) {
                    Ok((drafts, snap)) => {
                        d_snapshot = Some(snap);
                        if tx.send(DraftResp::Drafts(drafts)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(DraftResp::Err(e));
                        return;
                    }
                }
            }
            DraftReq::AdvanceAndDraft {
                accepted,
                prev_drafts,
                k,
            } => {
                let accepted_len = accepted.len();
                let actual_k = prev_drafts.len();
                let was_full_accept = accepted_len == actual_k + 1
                    && accepted.len() >= actual_k
                    && accepted[..actual_k] == prev_drafts[..];

                let adv_arr = if was_full_accept {
                    let bonus = accepted[accepted_len - 1];
                    let adv = [prev_drafts[actual_k - 1] as i32, bonus as i32];
                    mlx_rs::Array::from_slice(&adv, &[1, 2])
                } else {
                    // Rewind to snapshot (pre-draft state), then feed accepted prefix.
                    if let Some(snap) = d_snapshot.as_ref() {
                        d_cache = snap.clone();
                    }
                    let adv_input: Vec<i32> = accepted.iter().map(|&t| t as i32).collect();
                    mlx_rs::Array::from_slice(&adv_input, &[1, adv_input.len() as i32])
                };

                let logits = match drafter.model.forward(&adv_arr, None, &mut d_cache) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.send(DraftResp::Err(format!("drafter advance: {e}")));
                        return;
                    }
                };
                if let Err(e) = mlx_rs::transforms::eval([&logits]) {
                    let _ = tx.send(DraftResp::Err(format!("eval advance: {e}")));
                    return;
                }
                if let Err(e) = drafter.eval_cache_with(&d_cache, &[&logits]) {
                    let _ = tx.send(DraftResp::Err(format!("eval advance cache: {e}")));
                    return;
                }
                context_len += accepted_len;
                saved_logits = Some(logits);
                let logits_ref = saved_logits.as_ref().expect("just set");
                match produce_drafts(drafter, &mut d_cache, logits_ref, k, max_seq, context_len) {
                    Ok((drafts, snap)) => {
                        d_snapshot = Some(snap);
                        if tx.send(DraftResp::Drafts(drafts)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(DraftResp::Err(e));
                        return;
                    }
                }
            }
        }
    }
}

/// Threaded variant of `speculative_generate_next`. See module docs.
///
/// Signature matches the serial version so callers dispatch based on env var
/// without restructuring.
#[allow(clippy::as_conversions, clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn speculative_generate_next_threaded(
    drafter: &mut QwenNextCausalDrafter,
    verifier: &mut crate::AnyModel,
    prompt: &[u32],
    max_tokens: usize,
    k_low: usize,
    k_high: usize,
    eos_token_ids: &[u32],
) -> Vec<u32> {
    use std::sync::mpsc;

    let mut context: Vec<u32> = prompt.to_vec();
    let mut generated: Vec<u32> = Vec::new();
    let mut k_ctrl = AdaptiveKController::new(k_low, k_high, 3);

    std::thread::scope(|s| {
        let (req_tx, req_rx) = mpsc::channel::<DraftReq>();
        let (resp_tx, resp_rx) = mpsc::channel::<DraftResp>();

        let _handle = s.spawn(move || {
            drafter_thread_loop(drafter, &req_rx, &resp_tx);
        });

        // --- Verifier bootstrap (main thread) ---
        let t_boot = std::time::Instant::now();
        let prompt_i32: Vec<i32> = context.iter().map(|&t| t as i32).collect();
        let prompt_arr = mlx_rs::Array::from_slice(&prompt_i32, &[1, prompt_i32.len() as i32]);

        let mut verify_cache = verifier.make_cache();
        let mut saved_verify_logits = verifier
            .forward(&prompt_arr, None, &mut verify_cache)
            .expect("verifier bootstrap prefill");
        mlx_rs::transforms::eval([&saved_verify_logits]).expect("eval verify prefill logits");
        verify_cache
            .eval_for_clone()
            .expect("eval verify prefill cache");

        // Kick off drafter prefill + first draft round in parallel with any
        // downstream setup (today: nothing; placeholder for future overlap).
        let k0 = k_ctrl.current_k().min(max_tokens.max(1));
        req_tx
            .send(DraftReq::Prefill {
                prompt: context.clone(),
                k: k0,
            })
            .expect("prefill send");

        let boot_ms = t_boot.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Bootstrap (verifier prefill; drafter in flight): {boot_ms:.0}ms");

        let mut total_drafted = 0usize;
        let mut total_accepted = 0usize;
        let mut draft_wait_ms = 0.0f64;
        let mut verify_ms = 0.0f64;
        let mut advance_ms = 0.0f64;
        let mut round: usize = 0;

        while generated.len() < max_tokens {
            // Wait for drafter's response (prefill or advance).
            let t0 = std::time::Instant::now();
            let draft_tokens = match resp_rx.recv() {
                Ok(DraftResp::Drafts(d)) => d,
                Ok(DraftResp::Err(e)) => {
                    eprintln!("drafter error: {e}");
                    break;
                }
                Err(_) => break,
            };
            let actual_k = draft_tokens.len();
            draft_wait_ms += t0.elapsed().as_secs_f64() * 1000.0;

            if actual_k == 0 {
                break;
            }

            // --- Verify ---
            let t1 = std::time::Instant::now();
            let verify_snapshot = verify_cache.clone();

            let draft_input: Vec<i32> = draft_tokens.iter().map(|&d| d as i32).collect();
            let draft_arr = mlx_rs::Array::from_slice(&draft_input, &[1, draft_input.len() as i32]);
            let all_logits = verifier
                .forward_all_logits(&draft_arr, None, &mut verify_cache)
                .expect("verifier forward_all_logits");
            mlx_rs::transforms::eval([&all_logits]).expect("eval all_logits");

            let new_2d = all_logits.squeeze_axes(&[0]).expect("squeeze");
            let new_preds = ix::argmax_axis(&new_2d, -1, false).expect("argmax new");
            mlx_rs::transforms::eval([&new_preds]).expect("eval new argmax");

            let first_pred = {
                let p = ix::argmax_axis(&saved_verify_logits, -1, false).expect("argmax saved");
                mlx_rs::transforms::eval([&p]).expect("eval saved argmax");
                p.index((0, 0)).item::<i32>() as u32
            };

            let mut verify_argmax: Vec<u32> = Vec::with_capacity(actual_k + 1);
            verify_argmax.push(first_pred);
            for i in 0..actual_k {
                verify_argmax.push(new_preds.index(i as i32).item::<i32>() as u32);
            }
            verify_ms += t1.elapsed().as_secs_f64() * 1000.0;

            // --- Accept ---
            let mut accepted_tokens = accept_prefix(&draft_tokens, &verify_argmax);
            let remaining = max_tokens - generated.len();
            if accepted_tokens.len() > remaining {
                accepted_tokens.truncate(remaining);
            }
            let eos_hit = if eos_token_ids.is_empty() {
                None
            } else {
                accepted_tokens
                    .iter()
                    .position(|t| eos_token_ids.contains(t))
            };
            if let Some(idx) = eos_hit {
                accepted_tokens.truncate(idx + 1);
            }
            let accepted = accepted_tokens.len().saturating_sub(1);
            total_accepted += accepted;
            total_drafted += actual_k;
            k_ctrl.record(accepted, actual_k);

            generated.extend_from_slice(&accepted_tokens);
            context.extend_from_slice(&accepted_tokens);

            round += 1;
            let round_verify_ms = t1.elapsed().as_secs_f64() * 1000.0;
            let round_draft_ms = t0.elapsed().as_secs_f64() * 1000.0 - round_verify_ms;
            eprintln!(
                "  R{round}: accepted {accepted}/{actual_k} (+{} new) | K={} | draft_wait={round_draft_ms:.0}ms verify={round_verify_ms:.0}ms",
                accepted_tokens.len(),
                k_ctrl.current_k(),
            );

            if eos_hit.is_some() || generated.len() >= max_tokens {
                break;
            }

            // --- Advance verify cache on main thread ---
            let t_adv = std::time::Instant::now();
            if accepted == actual_k {
                let bonus = *accepted_tokens.last().expect("accepted has bonus");
                let bonus_arr = mlx_rs::Array::from_slice(&[bonus as i32], &[1, 1]);
                saved_verify_logits = verifier
                    .forward(&bonus_arr, None, &mut verify_cache)
                    .expect("verifier bonus forward");
                mlx_rs::transforms::eval([&saved_verify_logits]).expect("eval bonus logits");
                verify_cache
                    .eval_for_clone()
                    .expect("eval verify cache fast");
            } else {
                verify_cache = verify_snapshot;
                let advance_input: Vec<i32> = accepted_tokens.iter().map(|&t| t as i32).collect();
                let advance_arr =
                    mlx_rs::Array::from_slice(&advance_input, &[1, advance_input.len() as i32]);
                saved_verify_logits = verifier
                    .forward(&advance_arr, None, &mut verify_cache)
                    .expect("verifier advance forward");
                mlx_rs::transforms::eval([&saved_verify_logits]).expect("eval advance logits");
                verify_cache
                    .eval_for_clone()
                    .expect("eval verify cache slow");
            }
            advance_ms += t_adv.elapsed().as_secs_f64() * 1000.0;

            // --- Ask drafter to advance and produce next round drafts ---
            let k_next = k_ctrl.current_k().min(max_tokens - generated.len());
            if k_next == 0 {
                break;
            }
            req_tx
                .send(DraftReq::AdvanceAndDraft {
                    accepted: accepted_tokens.clone(),
                    prev_drafts: draft_tokens,
                    k: k_next,
                })
                .expect("advance send");
        }

        // Drop req_tx → drafter's recv returns Err → drafter thread exits;
        // scope joins on exit.
        drop(req_tx);

        let total_ms = draft_wait_ms + verify_ms + advance_ms;
        let tps = if total_ms > 0.0 {
            generated.len() as f64 / total_ms * 1000.0
        } else {
            0.0
        };
        let acc_rate = if total_drafted > 0 {
            total_accepted as f64 / total_drafted as f64 * 100.0
        } else {
            0.0
        };
        eprintln!("  --- Totals (threaded) ---");
        eprintln!(
            "  {draft_wait_ms:.0}ms draft_wait + {verify_ms:.0}ms verify + {advance_ms:.0}ms advance = {total_ms:.0}ms",
        );
        eprintln!("  Acceptance: {total_accepted}/{total_drafted} ({acc_rate:.1}%)");
        eprintln!("  Throughput: {tps:.1} tok/s");
    });

    generated
}
