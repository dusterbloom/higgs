//! Dump LFM3 logits for the independent `PyTorch` parity oracle.
//!
//! Usage: `lfm3_logits <model_dir> <ids.json> <out_dir> [f32]`
//!
//! Each output safetensors file contains `[T, V]` `oneshot` and `stepped`
//! tensors.  `stepped` uses two prefill chunks followed by single-token decode,
//! exercising both convolution and KV cache state.  `f32` casts every bf16
//! parameter (including q4 scales/biases) to f32 before execution.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout
)]

use std::collections::{BTreeMap, HashMap};

use higgs_models::lfm3::load_lfm3_model;
use mlx_rs::{Array, Dtype, module::ModuleParameters, ops, ops::indexing::IndexOp};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (model_dir, ids_path, out_dir, f32) = match args.as_slice() {
        [_, model, ids, out] => (model, ids, out, false),
        [_, model, ids, out, mode] if mode == "f32" => (model, ids, out, true),
        _ => panic!("usage: lfm3_logits <model_dir> <ids.json> <out_dir> [f32]"),
    };

    let sequences: BTreeMap<String, Vec<i32>> =
        serde_json::from_str(&std::fs::read_to_string(ids_path).unwrap()).unwrap();
    std::fs::create_dir_all(out_dir).unwrap();
    let mut model = load_lfm3_model(model_dir).unwrap();
    if f32 {
        for parameter in model.parameters_mut().flatten().into_values() {
            if parameter.dtype() == Dtype::Bfloat16 {
                *parameter = parameter.as_dtype(Dtype::Float32).unwrap();
            }
        }
    }

    for (name, ids) in sequences {
        let length = i32::try_from(ids.len()).unwrap();
        assert!(
            length >= 3,
            "parity sequence must contain at least three tokens"
        );
        let tokens = Array::from_slice(&ids, &[1, length]);
        let mut forward = |range: std::ops::Range<i32>, cache: &mut Vec<_>| {
            let logits = model
                .forward_all_logits(&tokens.index((.., range)), None, cache)
                .unwrap();
            logits.eval().unwrap();
            logits
        };

        let oneshot = forward(0..length, &mut Vec::new());
        let decode_tail = 4.min(length - 2);
        let prefill_end = length - decode_tail;
        let split = (prefill_end / 2).max(1);
        let mut cache = Vec::new();
        let mut parts = vec![
            forward(0..split, &mut cache),
            forward(split..prefill_end, &mut cache),
        ];
        parts.extend((prefill_end..length).map(|i| forward(i..i + 1, &mut cache)));
        let stepped = ops::concatenate_axis(&parts, 1).unwrap();

        let output = HashMap::from([
            ("oneshot".to_owned(), oneshot.index(0)),
            ("stepped".to_owned(), stepped.index(0)),
        ]);
        Array::save_safetensors(&output, None, format!("{out_dir}/{name}.safetensors")).unwrap();
        println!("{name}: {length} tokens ({prefill_end} prefill, {decode_tail} decode)");
    }
}
