//! Speculative decoding with exact acceptance reproduces the causal greedy
//! sequence whatever the drafter proposes (and however many tokens it
//! drafts), keeps only verified rows in the state, and stops at EOS /
//! `max_new_tokens`. The drafter-context bookkeeping hands the drafter
//! exactly the rows the verify kept.
use higgs_models::clover::{
    CloverPenalties, DraftTaps, Penalize, SelfSpecConfig, SpecModel, self_spec_decode,
};
use mlx_rs::{Array, error::Exception};

const EOS: u32 = 14;

/// Causal greedy model whose next token after position `p` is `seq[p + 1]`
/// (a context that left `seq` predicts token 0). Drafts are `seq` with every
/// `wrong_every`-th drafted token replaced by a wrong one, `short_by` fewer
/// than asked.
struct Oracle {
    seq: Vec<u32>,
    /// Committed rows: `seq[..len]` is in the state.
    len: usize,
    verified: Vec<u32>,
    wrong_every: usize,
    short_by: usize,
    drafts: usize,
}

impl SpecModel for Oracle {
    fn draft(&mut self, pending: u32, max: usize) -> Result<Array, Exception> {
        // The pending token is the greedy token after the state.
        assert_eq!(pending, self.seq[self.len]);
        let k = max.saturating_sub(self.short_by);
        let tokens: Vec<u32> = (1..=k)
            .map(|i| {
                self.drafts += 1;
                let truth = self.seq.get(self.len + i).copied().unwrap_or(0);
                if self.drafts % self.wrong_every == 0 {
                    (truth + 1) % 14
                } else {
                    truth
                }
            })
            .collect();
        Ok(Array::from_slice(&tokens, &[k as i32]))
    }

    fn verify(&mut self, tokens: &Array, penalize: &Penalize<'_>) -> Result<Vec<u32>, Exception> {
        let tokens = tokens.as_slice::<u32>().to_vec();
        let mut on_track = true;
        let next: Vec<u32> = (0..tokens.len())
            .map(|i| {
                on_track &= self.seq.get(self.len + i) == Some(&tokens[i]);
                if on_track {
                    self.seq.get(self.len + i + 1).copied().unwrap_or(0)
                } else {
                    0
                }
            })
            .collect();
        self.verified = tokens;
        // One-hot logits: penalties are off, so `penalize` must keep the argmax.
        let logits: Vec<f32> = next
            .iter()
            .flat_map(|&t| (0..15).map(move |v| if v == t { 10.0 } else { 0.0 }))
            .collect();
        argmax_rows(&penalize(Array::from_slice(
            &logits,
            &[next.len() as i32, 15],
        ))?)
    }

    fn keep(&mut self, n: usize) -> Result<(), Exception> {
        // Only causal-greedy tokens may ever enter the state.
        assert_eq!(&self.verified[..n], &self.seq[self.len..self.len + n]);
        self.len += n;
        Ok(())
    }
}

fn run(
    seq: &[u32],
    wrong_every: usize,
    short_by: usize,
    max_new: usize,
    rows: &[usize],
) -> Vec<u32> {
    let _gate = higgs_models::mlx_exec::acquire();
    // seq[0] is the prompt's last token, already in the state; seq[1] is `first`.
    let mut model = Oracle {
        seq: seq.to_vec(),
        len: 1,
        verified: Vec::new(),
        wrong_every,
        short_by,
        drafts: 0,
    };
    let cfg = SelfSpecConfig {
        max_new_tokens: max_new,
        eos_token_ids: vec![EOS],
        rows: rows.to_vec(),
        cycle_base_rows: 8.2,
        penalties: CloverPenalties::default(),
        vocab: 15,
        pad_token_id: None,
    };
    let mut streamed = Vec::new();
    let out = self_spec_decode(&cfg, seq[1], &mut model, |new| {
        streamed.extend_from_slice(new);
        false
    })
    .unwrap();
    assert_eq!(streamed, out.tokens);
    out.tokens
}

#[test]
fn output_is_the_causal_greedy_sequence_for_any_drafter() {
    let seq: Vec<u32> = (0..60).map(|i| (i * 7 % 13) as u32).collect();
    for wrong_every in [1, 2, 3, 5, 1000] {
        for rows in [&[4, 8][..], &[3, 6, 8], &[4], &[8], &[2]] {
            for short_by in [0, 1, 7] {
                assert_eq!(
                    run(&seq, wrong_every, short_by, 40, rows),
                    seq[1..41],
                    "wrong_every {wrong_every} rows {rows:?} short_by {short_by}"
                );
            }
        }
    }
}

#[test]
fn stops_at_first_eos_and_at_max_new_tokens() {
    let mut seq: Vec<u32> = (0..40).map(|i| (i * 5 % 13) as u32).collect();
    seq[23] = EOS;
    assert_eq!(run(&seq, 1000, 0, 100, &[4, 8]), seq[1..24]);
    assert_eq!(run(&seq, 3, 0, 7, &[8]), seq[1..8]);
}

/// `[1, rows, 2]` tap whose row `r` holds `base + r`.
fn tap(base: u32, rows: usize) -> Array {
    let values: Vec<f32> = (0..rows)
        .flat_map(|r| [(base + r as u32) as f32; 2])
        .collect();
    Array::from_slice(&values, &[1, rows as i32, 2])
}

fn rows_of(tap: &Array) -> Vec<u32> {
    tap.as_slice::<f32>()
        .chunks(2)
        .map(|row| row[0] as u32)
        .collect()
}

#[test]
fn drafter_context_receives_exactly_the_kept_rows() {
    let _gate = higgs_models::mlx_exec::acquire();
    // Prefill: the last chunk's taps (two tap layers) wait for the first draft.
    let mut taps = DraftTaps::new(vec![tap(0, 5), tap(100, 5)]);
    assert_eq!(taps.pending_rows().unwrap(), 5);
    taps.consume();
    assert_eq!(taps.pending_rows().unwrap(), 0);

    // Verify 8 rows, keep 3: the drafter must see rows 0..3 of each layer.
    taps.record(vec![tap(10, 8), tap(110, 8)]);
    taps.keep(3).unwrap();
    assert_eq!(taps.pending_rows().unwrap(), 3);
    assert_eq!(rows_of(&taps.pending()[0]), [10, 11, 12]);
    assert_eq!(rows_of(&taps.pending()[1]), [110, 111, 112]);

    // Keeping before the drafter consumed the pending rows would skip them.
    taps.record(vec![tap(20, 4), tap(120, 4)]);
    assert!(taps.keep(2).is_err());
    taps.consume();
    // A verify keeps at least its first row and at most all of them.
    assert!(taps.keep(0).is_err());
    assert!(taps.keep(5).is_err());
    taps.keep(4).unwrap();
    assert_eq!(rows_of(&taps.pending()[0]), [20, 21, 22, 23]);
    assert_eq!(rows_of(&taps.pending()[1]), [120, 121, 122, 123]);

    // One keep per verify.
    taps.consume();
    assert!(taps.keep(1).is_err());
}

fn argmax_rows(logits: &Array) -> Result<Vec<u32>, Exception> {
    let next =
        mlx_rs::ops::indexing::argmax_axis(logits, -1, None)?.as_dtype(mlx_rs::Dtype::Uint32)?;
    next.eval()?;
    Ok(next.as_slice::<u32>().to_vec())
}

const V: usize = 16;
const STOP: u32 = 15;
const CLOSE: u32 = 7;

/// A causal LM whose next-token logits depend on the last token only (a
/// fixed pseudo-random table, values in `[-3, 3)`): greedy decoding falls
/// into short loops, which is what the penalties exist to break. `CLOSE` and
/// `STOP` sit at -8, so only their ramps can emit them.
fn table() -> Vec<[f32; V]> {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    (0..V)
        .map(|_| {
            std::array::from_fn(|v| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if v as u32 == CLOSE || v as u32 == STOP {
                    -8.0
                } else {
                    ((x >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0
                }
            })
        })
        .collect()
}

fn ramp(position: usize, start: u32, window: u32, strength: f32) -> f32 {
    ((position as f32 - start as f32) / window as f32).clamp(0.0, 1.0) * strength
}

/// Sequential penalised greedy decoding, written out independently of the
/// library: frequency, presence, repetition, length ramp, think-end ramp.
fn sequential(t: &[[f32; V]], p: &CloverPenalties, last: u32, max_new: usize) -> Vec<u32> {
    let argmax = |row: &[f32]| {
        (0..V)
            .max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a)))
            .unwrap() as u32
    };
    let mut out = vec![argmax(&t[last as usize])];
    while out.len() < max_new && *out.last().unwrap() != STOP {
        let mut row = t[*out.last().unwrap() as usize];
        let position = out.len() + 1;
        let closed = out.contains(&CLOSE);
        for (v, logit) in row.iter_mut().enumerate() {
            let count = out.iter().filter(|&&o| o as usize == v).count() as f32;
            *logit -= count * p.frequency;
            if count > 0.0 {
                *logit -= p.presence;
                *logit = if *logit < 0.0 {
                    *logit * p.repetition
                } else {
                    *logit / p.repetition
                };
            }
            if v as u32 == STOP {
                *logit += ramp(position, p.length_start, p.length_window, p.length_penalty);
            }
            if v as u32 == CLOSE && !closed {
                *logit += ramp(
                    position,
                    p.think_end_start,
                    p.think_end_window,
                    p.think_end_bias,
                );
            }
        }
        out.push(argmax(&row));
    }
    out
}

/// [`SpecModel`] over the table; drafts follow `truth` with every
/// `wrong_every`-th drafted token corrupted.
struct Markov {
    table: Vec<[f32; V]>,
    truth: Vec<u32>,
    /// Response tokens in the state (the pending one is not).
    len: usize,
    verified: Vec<u32>,
    wrong_every: usize,
    drafts: usize,
}

impl SpecModel for Markov {
    fn draft(&mut self, _pending: u32, max: usize) -> Result<Array, Exception> {
        let tokens: Vec<u32> = (1..=max)
            .map(|i| {
                self.drafts += 1;
                let truth = self.truth.get(self.len + i).copied().unwrap_or(0);
                if self.drafts % self.wrong_every == 0 {
                    (truth + 1) % V as u32
                } else {
                    truth
                }
            })
            .collect();
        Ok(Array::from_slice(&tokens, &[max as i32]))
    }

    fn verify(&mut self, tokens: &Array, penalize: &Penalize<'_>) -> Result<Vec<u32>, Exception> {
        self.verified = tokens.as_slice::<u32>().to_vec();
        let logits: Vec<f32> = self
            .verified
            .iter()
            .flat_map(|&t| self.table[t as usize])
            .collect();
        let rows = self.verified.len() as i32;
        argmax_rows(&penalize(Array::from_slice(&logits, &[rows, V as i32]))?)
    }

    fn keep(&mut self, n: usize) -> Result<(), Exception> {
        assert_eq!(&self.verified[..n], &self.truth[self.len..self.len + n]);
        self.len += n;
        Ok(())
    }
}

#[test]
fn penalised_output_is_sequential_penalised_greedy_for_any_drafter() {
    let _gate = higgs_models::mlx_exec::acquire();
    let t = table();
    let p = CloverPenalties {
        presence: 0.9,
        frequency: 0.3,
        repetition: 1.05,
        length_penalty: 20.0,
        length_start: 30,
        length_window: 10,
        think_end_bias: 20.0,
        think_end_start: 4,
        think_end_window: 8,
        think_end_token: Some(CLOSE),
    };
    let max_new = 60;
    let truth = sequential(&t, &p, 3, max_new);
    // The penalties must matter here, else the test proves nothing.
    assert_ne!(
        truth,
        sequential(&t, &CloverPenalties::default(), 3, max_new)
    );
    // Both ramps fire: one `CLOSE` past the think-end start, `STOP` past the
    // length start.
    let close_at = truth.iter().position(|&x| x == CLOSE).unwrap();
    assert!(close_at > 4 && truth.iter().filter(|&&x| x == CLOSE).count() == 1);
    assert!(truth.len() > 30 && truth.last() == Some(&STOP));
    for wrong_every in [1, 2, 3, 1000] {
        for rows in [&[3, 6, 8][..], &[2], &[8]] {
            let cfg = SelfSpecConfig {
                max_new_tokens: max_new,
                eos_token_ids: vec![STOP],
                rows: rows.to_vec(),
                cycle_base_rows: 8.6,
                penalties: p.clone(),
                vocab: V as i32,
                pad_token_id: None,
            };
            let mut model = Markov {
                table: t.clone(),
                truth: truth.clone(),
                len: 0,
                verified: Vec::new(),
                wrong_every,
                drafts: 0,
            };
            let out = self_spec_decode(&cfg, truth[0], &mut model, |_| false).unwrap();
            assert_eq!(out.tokens, truth, "wrong_every {wrong_every} rows {rows:?}");
        }
    }
}
