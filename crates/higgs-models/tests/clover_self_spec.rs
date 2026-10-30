//! Self-speculation with exact acceptance reproduces the causal greedy
//! sequence whatever the drafter proposes, keeps only verified rows in the
//! state, and stops at EOS / `max_new_tokens`.
use higgs_models::clover::{BlockModel, SelfSpecConfig, SpecModel, self_spec_decode};
use mlx_rs::{Array, error::Exception};

const VOCAB: i32 = 16;
const MASK: u32 = 15;
const EOS: u32 = 14;

/// Causal greedy model whose next token after position `p` is `seq[p + 1]`
/// (a context that left `seq` predicts token 0). Drafts are `seq` with every
/// `wrong_every`-th drafted position replaced by a wrong token.
struct Oracle {
    seq: Vec<u32>,
    /// Committed rows: `seq[..len]` is in the state.
    len: usize,
    verified: Vec<u32>,
    wrong_every: usize,
    drafts: usize,
}

impl BlockModel for Oracle {
    fn preview(&mut self, block: &[u32]) -> Result<Array, Exception> {
        let rows = block.len();
        let mut flat = vec![0.0_f32; rows * VOCAB as usize];
        for i in 0..rows {
            self.drafts += 1;
            let truth = self.seq.get(self.len + i).copied().unwrap_or(0);
            let token = if self.drafts % self.wrong_every == 0 {
                (truth + 1) % 14
            } else {
                truth
            };
            flat[i * VOCAB as usize + token as usize] = 1.0;
            flat[i * VOCAB as usize + MASK as usize] = 9.0; // must never be drafted
        }
        Ok(Array::from_slice(&flat, &[1, rows as i32, VOCAB]))
    }

    fn commit(&mut self, _tokens: &[u32]) -> Result<(), Exception> {
        unreachable!("self-spec commits through verify + keep")
    }
}

impl SpecModel for Oracle {
    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<u32>, Exception> {
        self.verified = tokens.to_vec();
        let mut on_track = true;
        Ok((0..tokens.len())
            .map(|i| {
                on_track &= self.seq.get(self.len + i) == Some(&tokens[i]);
                if on_track {
                    self.seq.get(self.len + i + 1).copied().unwrap_or(0)
                } else {
                    0
                }
            })
            .collect())
    }

    fn keep(&mut self, n: usize) -> Result<(), Exception> {
        // Only causal-greedy tokens may ever enter the state.
        assert_eq!(&self.verified[..n], &self.seq[self.len..self.len + n]);
        self.len += n;
        Ok(())
    }
}

fn run(seq: &[u32], wrong_every: usize, max_new: usize, rows: Option<usize>) -> Vec<u32> {
    let _gate = higgs_models::mlx_exec::acquire();
    // seq[0] is the prompt's last token, already in the state; seq[1] is `first`.
    let mut model = Oracle {
        seq: seq.to_vec(),
        len: 1,
        verified: Vec::new(),
        wrong_every,
        drafts: 0,
    };
    let cfg = SelfSpecConfig {
        mask_token_id: MASK,
        max_new_tokens: max_new,
        eos_token_ids: vec![EOS],
        draft_rows: rows,
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
        for rows in [None, Some(4), Some(8), Some(2)] {
            assert_eq!(
                run(&seq, wrong_every, 40, rows),
                seq[1..41],
                "wrong_every {wrong_every} rows {rows:?}"
            );
        }
    }
}

#[test]
fn stops_at_first_eos_and_at_max_new_tokens() {
    let mut seq: Vec<u32> = (0..40).map(|i| (i * 5 % 13) as u32).collect();
    seq[23] = EOS;
    assert_eq!(run(&seq, 1000, 100, None), seq[1..24]);
    assert_eq!(run(&seq, 3, 7, Some(8)), seq[1..8]);
}
