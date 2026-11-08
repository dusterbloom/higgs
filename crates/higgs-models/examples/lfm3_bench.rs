//! Real-checkpoint LFM3 prefill/decode benchmark.
//!
//! Usage: `lfm3_bench <model_dir>`
//!
//! The benchmark uses complete plain-chat prompts tokenized by the checkpoint's
//! own tokenizer, warms the model once, then reports synchronized wall-clock
//! prefill and 128-step greedy decode throughput. Every logits graph and every
//! argmax is evaluated before the timer advances.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout
)]

use std::{path::Path, time::Instant};

use higgs_models::{
    LayerCache,
    cache::KeyValueCache,
    lfm3::{Lfm3CausalLM, load_lfm3_model},
};
use mlx_rs::{Array, ops, ops::indexing::IndexOp};
use tokenizers::Tokenizer;

const CHAT_PREFIX: &str = "<|startoftext|><|im_start|>user\n";
const CHAT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>";
const DECODE_STEPS: usize = 128;

#[derive(Debug)]
struct BenchResult {
    prompt_tokens: usize,
    prefill_seconds: f64,
    decode_seconds: f64,
    generated: Vec<u32>,
}

fn exact_chat_prompt(tokenizer: &Tokenizer, target: usize) -> (String, Vec<i32>) {
    // For this tokenizer each additional `x ` contributes exactly one token,
    // but use measured feedback rather than baking that property into the
    // result. This keeps the prompt a complete user/assistant chat message.
    let mut repeats = target.saturating_sub(11).max(1);
    for _ in 0..16 {
        let content = "x ".repeat(repeats);
        let prompt = format!("{CHAT_PREFIX}{content}{CHAT_SUFFIX}");
        let encoding = tokenizer.encode(prompt.clone(), false).unwrap();
        let length = encoding.len();
        if length == target {
            return (
                prompt,
                encoding
                    .get_ids()
                    .iter()
                    .map(|&id| i32::try_from(id).unwrap())
                    .collect(),
            );
        }
        if length < target {
            repeats += target - length;
        } else {
            repeats = repeats.saturating_sub(length - target).max(1);
        }
    }
    panic!("could not construct an exact {target}-token chat prompt");
}

fn cache_offset(cache: &[Option<LayerCache>]) -> i32 {
    let mut offsets = cache.iter().filter_map(|slot| match slot.as_ref()? {
        LayerCache::KV(kv) => Some(KeyValueCache::offset(kv)),
        LayerCache::Arrays(arrays) => Some(arrays.offset),
    });
    let first = offsets.next().unwrap();
    assert!(
        offsets.all(|offset| offset == first),
        "cache layer offsets diverged"
    );
    first
}

fn greedy(logits: &Array) -> u32 {
    // Evaluate the full logits graph, then evaluate and synchronize the argmax.
    // Calling `item` only after eval avoids timing a queued graph as completed.
    logits.eval().unwrap();
    let token = ops::indexing::argmax_axis(logits.index((.., -1, ..)), -1, None).unwrap();
    token.eval().unwrap();
    token.item()
}

fn run_case(
    model: &mut Lfm3CausalLM,
    tokenizer: &Tokenizer,
    prompt_tokens: usize,
    decode_steps: usize,
) -> BenchResult {
    let (_prompt, ids) = exact_chat_prompt(tokenizer, prompt_tokens);
    assert_eq!(ids.len(), prompt_tokens);
    let input = Array::from_slice(&ids, &[1, i32::try_from(ids.len()).unwrap()]);
    let mut cache = Vec::new();

    let prefill_start = Instant::now();
    let logits = model.forward(&input, None, &mut cache).unwrap();
    logits.eval().unwrap();
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    assert_eq!(cache_offset(&cache), i32::try_from(prompt_tokens).unwrap());

    // Sampling the first token is outside the decode timer. The measured loop
    // then performs exactly `decode_steps` synchronized single-token forwards.
    let mut next = greedy(&logits);
    let mut generated = Vec::with_capacity(decode_steps);
    let decode_start = Instant::now();
    for step in 0..decode_steps {
        generated.push(next);
        let token = Array::from_slice(&[i32::try_from(next).unwrap()], &[1, 1]);
        let decode_logits = model.forward(&token, None, &mut cache).unwrap();
        next = greedy(&decode_logits);
        assert_eq!(
            cache_offset(&cache),
            i32::try_from(prompt_tokens + step + 1).unwrap(),
            "cache position after decode step {step}"
        );
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    BenchResult {
        prompt_tokens,
        prefill_seconds,
        decode_seconds,
        generated,
    }
}

fn print_result(tokenizer: &Tokenizer, result: &BenchResult) {
    let prefill_tps =
        f64::from(u32::try_from(result.prompt_tokens).unwrap()) / result.prefill_seconds;
    let decode_tps =
        f64::from(u32::try_from(result.generated.len()).unwrap()) / result.decode_seconds;
    let text = tokenizer.decode(&result.generated, true).unwrap();
    println!(
        "RESULT prompt_tokens={} prefill_seconds={:.6} prefill_tok_s={:.3} \
         decode_steps={} decode_seconds={:.6} decode_tok_s={:.3}",
        result.prompt_tokens,
        result.prefill_seconds,
        prefill_tps,
        result.generated.len(),
        result.decode_seconds,
        decode_tps,
    );
    println!(
        "GENERATED prompt_tokens={} text={text:?}",
        result.prompt_tokens
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, model_dir] = args.as_slice() else {
        panic!("usage: lfm3_bench <model_dir>");
    };
    let model_path = Path::new(model_dir);
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).unwrap();
    let load_start = Instant::now();
    let mut model = load_lfm3_model(model_path).unwrap();
    println!("LOAD seconds={:.6}", load_start.elapsed().as_secs_f64());

    println!("WARMUP prompt_tokens=32 decode_steps=8");
    let warmup = run_case(&mut model, &tokenizer, 32, 8);
    println!(
        "WARMUP_DONE prefill_seconds={:.6} decode_seconds={:.6}",
        warmup.prefill_seconds, warmup.decode_seconds
    );

    for prompt_tokens in [128, 512, 2048] {
        // Prefill and attention shapes differ by context length. Warm each full
        // measured shape (including the 128 cache lengths reached by decode)
        // so shader compilation cannot masquerade as model throughput.
        println!("CASE_WARMUP prompt_tokens={prompt_tokens} decode_steps={DECODE_STEPS}");
        let case_warmup = run_case(&mut model, &tokenizer, prompt_tokens, DECODE_STEPS);
        println!(
            "CASE_WARMUP_DONE prompt_tokens={} prefill_seconds={:.6} decode_seconds={:.6}",
            prompt_tokens, case_warmup.prefill_seconds, case_warmup.decode_seconds
        );
        print_result(
            &tokenizer,
            &run_case(&mut model, &tokenizer, prompt_tokens, DECODE_STEPS),
        );
    }
}
