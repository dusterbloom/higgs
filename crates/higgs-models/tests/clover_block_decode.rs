//! `LowConfidence` block-loop invariants on a synthetic model (no weights):
//! only `remaining` tokens are returned, generation stops at the first EOS in
//! positional order even when it resolves first, and unresolved/uncommitted
//! block tokens never penalise themselves.
use higgs_models::{
    SamplingParams,
    clover::{BlockDecodeConfig, BlockModel, CloverPenalties, block_decode},
};
use mlx_rs::{Array, error::Exception};

const VOCAB: i32 = 8;
const MASK: u32 = 7;
const EOS: u32 = 6;
const BLOCK: usize = 4;

/// Fixed per-position logits (`favourite token -> score`), independent of the block.
struct Fake {
    rows: Vec<Vec<(u32, f32)>>,
    commits: Vec<Vec<u32>>,
}

impl BlockModel for Fake {
    fn preview(&mut self, block: &[u32]) -> Result<Array, Exception> {
        assert_eq!(block.len(), BLOCK);
        let mut flat = vec![0.0_f32; BLOCK * 8];
        for (i, row) in self.rows.iter().enumerate() {
            for &(token, score) in row {
                flat[i * 8 + token as usize] = score;
            }
        }
        Ok(Array::from_slice(&flat, &[1, 4, VOCAB]))
    }

    fn commit(&mut self, tokens: &[u32]) -> Result<(), Exception> {
        self.commits.push(tokens.to_vec());
        Ok(())
    }
}

fn config(max_new_tokens: usize, threshold: f32, penalties: CloverPenalties) -> BlockDecodeConfig {
    BlockDecodeConfig {
        block_size: BLOCK,
        threshold,
        mask_token_id: MASK,
        max_new_tokens,
        eos_token_ids: vec![EOS],
        pad_token_id: None,
        penalties,
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
    }
}

fn run(
    cfg: &BlockDecodeConfig,
    rows: Vec<Vec<(u32, f32)>>,
) -> (Vec<u32>, Vec<usize>, bool, Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let _gate = higgs_models::mlx_exec::acquire();
    let mut fake = Fake {
        rows,
        commits: Vec::new(),
    };
    let mut blocks = Vec::new();
    let out = block_decode(cfg, VOCAB, &mut fake, |emitted| {
        blocks.push(emitted.to_vec());
        false
    })
    .unwrap();
    (
        out.tokens,
        out.passes_per_block,
        out.hit_eos,
        fake.commits,
        blocks,
    )
}

#[test]
fn returns_only_remaining_and_commits_only_continued_blocks() {
    // Later positions are more confident, so each block resolves back to front
    // one position per pass (threshold 1 is never exceeded).
    let rows = (0..4u32).map(|i| vec![(i % 3, 5.0 + i as f32)]).collect();
    let (tokens, passes, hit_eos, commits, blocks) =
        run(&config(6, 1.0, CloverPenalties::default()), rows);
    assert_eq!(tokens, [0, 1, 2, 0, 0, 1]);
    assert_eq!(passes, [4, 4]);
    assert!(!hit_eos);
    assert_eq!(blocks, [vec![0, 1, 2, 0], vec![0, 1]]);
    // The final, partially returned block is never committed.
    assert_eq!(commits, [vec![0, 1, 2, 0]]);
}

#[test]
fn stops_at_first_eos_in_positional_order() {
    // EOS at position 2 is the most confident (resolved first); the earlier
    // positions resolved after it are still returned, the later one is not.
    let rows = vec![
        vec![(0, 5.0)],
        vec![(1, 6.0)],
        vec![(EOS, 9.0)],
        vec![(2, 7.0)],
    ];
    let (tokens, passes, hit_eos, commits, _) =
        run(&config(64, 1.0, CloverPenalties::default()), rows);
    assert_eq!(tokens, [0, 1, EOS]);
    assert_eq!(passes, [4]);
    assert!(hit_eos);
    assert!(commits.is_empty());
}

#[test]
fn block_tokens_do_not_penalise_themselves_until_committed() {
    // Every slot prefers 3 over 4 by less than the presence penalty. Within a
    // block, already-resolved 3s must not penalise the remaining slots; once
    // the block is committed, the next block switches to 4.
    let rows = vec![vec![(3, 5.0), (4, 3.0)]; 4];
    let penalties = CloverPenalties {
        presence: 3.0,
        ..CloverPenalties::default()
    };
    let (tokens, _, _, commits, _) = run(&config(8, 1.0, penalties), rows);
    assert_eq!(tokens, [3, 3, 3, 3, 4, 4, 4, 4]);
    assert_eq!(commits, [vec![3, 3, 3, 3]]);
}
