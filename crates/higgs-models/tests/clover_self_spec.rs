//! Speculative decoding with exact acceptance reproduces the causal greedy
//! sequence whatever the drafter proposes (and however many tokens it
//! drafts), keeps only verified rows in the state, and stops at EOS /
//! `max_new_tokens`. The drafter-context bookkeeping hands the drafter
//! exactly the rows the verify kept.
use higgs_models::clover::{DraftTaps, SelfSpecConfig, SpecModel, self_spec_decode};
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

    fn verify(&mut self, tokens: &Array) -> Result<Vec<u32>, Exception> {
        let tokens = tokens.as_slice::<u32>().to_vec();
        let mut on_track = true;
        let next = (0..tokens.len())
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
        Ok(next)
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
