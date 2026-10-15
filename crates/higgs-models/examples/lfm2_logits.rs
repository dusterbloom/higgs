//! Dump LFM2-MoE logits for the mlx-lm parity check (`scripts/lfm2_parity.py`).
//!
//! Usage: `lfm2_logits <model_dir> <ids.json> <out_dir> [f32]`
//!
//! `f32` casts every bf16 parameter to f32 first (quantized weights untouched),
//! so all activations run in f32 — compared against the mlx-lm f32 oracle this
//! isolates porting bugs from bf16 rounding noise.
//!
//! `ids.json` maps a name to token ids. Writes `<out_dir>/<name>.safetensors`
//! with two `[T, V]` logit tensors:
//! - `oneshot`: one forward pass over the whole sequence (prefill path);
//! - `stepped`: two prefill chunks, then one token at a time for the last
//!   `TAIL` positions (decode path: conv state, KV offsets, L=1 expert gather).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use std::collections::{BTreeMap, HashMap};

use higgs_models::lfm2_moe::load_lfm2_moe_model;
use mlx_rs::{Array, Dtype, module::ModuleParameters, ops, ops::indexing::IndexOp};

const TAIL: i32 = 64;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (model_dir, ids_path, out_dir, f32) = match args.as_slice() {
        [_, m, i, o] => (m, i, o, false),
        [_, m, i, o, mode] if mode == "f32" => (m, i, o, true),
        _ => panic!("usage: lfm2_logits <model_dir> <ids.json> <out_dir> [f32]"),
    };
    let seqs: BTreeMap<String, Vec<i32>> =
        serde_json::from_str(&std::fs::read_to_string(ids_path).unwrap()).unwrap();
    std::fs::create_dir_all(out_dir).unwrap();
    let mut model = load_lfm2_moe_model(model_dir).unwrap();
    if f32 {
        for p in model.parameters_mut().flatten().into_values() {
            if p.dtype() == Dtype::Bfloat16 {
                *p = p.as_dtype(Dtype::Float32).unwrap();
            }
        }
    }

    for (name, ids) in seqs {
        let t = i32::try_from(ids.len()).unwrap();
        let tokens = Array::from_slice(&ids, &[1, t]);
        let mut fwd = |range: std::ops::Range<i32>, cache: &mut Vec<_>| {
            let logits = model
                .forward_all_logits(&tokens.index((.., range)), None, cache)
                .unwrap();
            logits.eval().unwrap();
            logits
        };

        let oneshot = fwd(0..t, &mut Vec::new());
        let split = (t - TAIL) / 2;
        let mut cache = Vec::new();
        let mut parts = vec![fwd(0..split, &mut cache), fwd(split..t - TAIL, &mut cache)];
        parts.extend((t - TAIL..t).map(|i| fwd(i..i + 1, &mut cache)));
        let stepped = ops::concatenate_axis(&parts, 1).unwrap();

        let out = HashMap::from([
            ("oneshot".to_owned(), oneshot.index(0)),
            ("stepped".to_owned(), stepped.index(0)),
        ]);
        Array::save_safetensors(&out, None, format!("{out_dir}/{name}.safetensors")).unwrap();
        println!("{name}: {t} tokens");
    }
}
