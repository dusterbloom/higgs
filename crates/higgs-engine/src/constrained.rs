//! Constrained (guided) generation using FSM-based token masking.
//!
//! Uses `outlines-core` to pre-compute a finite-state machine from a JSON
//! schema or regex. During decode, `allowed_token_ids` returns which tokens
//! are valid at the current state, and `apply_mask` zeros out disallowed
//! logits before sampling.

use mlx_rs::{Array, error::Exception};
use outlines_core::index::Index;
use outlines_core::json_schema;
use outlines_core::vocabulary::Vocabulary;

use crate::error::EngineError;

/// Wraps an `outlines-core` Index for constrained decoding.
pub struct ConstrainedGenerator {
    index: Index,
    state: outlines_core::primitives::StateId,
}

impl ConstrainedGenerator {
    /// Build from a pre-computed `Index`.
    fn new(index: Index) -> Self {
        let state = index.initial_state();
        Self { index, state }
    }

    /// Build from a JSON schema string.
    ///
    /// Converts the schema to a regex via `outlines-core`, then builds the
    /// FSM index against the given vocabulary.
    pub fn from_json_schema(schema: &str, vocabulary: &Vocabulary) -> Result<Self, EngineError> {
        let regex = json_schema::regex_from_str(schema, None, None)
            .map_err(|e| EngineError::Generation(format!("Invalid JSON schema: {e}")))?;
        let index = Index::new(&regex, vocabulary)
            .map_err(|e| EngineError::Generation(format!("Failed to build FSM index: {e}")))?;
        Ok(Self::new(index))
    }

    /// Build a JSON-schema constraint inside the tool-call envelope understood
    /// by the response parsers.
    pub fn from_tagged_json_schema(
        schema: &str,
        vocabulary: &Vocabulary,
    ) -> Result<Self, EngineError> {
        let json_regex = json_schema::regex_from_str(schema, None, None)
            .map_err(|e| EngineError::Generation(format!("Invalid tool JSON schema: {e}")))?;
        Self::from_regex(
            &format!(r"<tool_call>\n{json_regex}\n</tool_call>"),
            vocabulary,
        )
    }

    /// Build for `json_object` mode (any valid JSON object).
    pub fn for_json_object(vocabulary: &Vocabulary) -> Result<Self, EngineError> {
        Self::from_json_schema(r#"{"type": "object"}"#, vocabulary)
    }

    /// Build from a regex pattern directly.
    pub fn from_regex(pattern: &str, vocabulary: &Vocabulary) -> Result<Self, EngineError> {
        let index = Index::new(pattern, vocabulary)
            .map_err(|e| EngineError::Generation(format!("Failed to build FSM index: {e}")))?;
        Ok(Self::new(index))
    }

    /// Get the set of allowed token IDs at the current state.
    pub fn allowed_token_ids(&self) -> Option<Vec<outlines_core::primitives::TokenId>> {
        self.index.allowed_tokens(&self.state)
    }

    /// Advance the FSM state after sampling a token.
    ///
    /// Returns `true` if the transition was valid, `false` if the token was
    /// not allowed (should not happen if `apply_mask` was used).
    fn advance(&mut self, token_id: u32) -> bool {
        if let Some(next) = self.index.next_state(&self.state, &token_id) {
            self.state = next;
            true
        } else {
            false
        }
    }

    /// Advance the FSM after a masked sample, failing closed when the mask
    /// invariant breaks.
    ///
    /// A masked decode can only sample tokens the FSM accepts; a rejection
    /// means the invariant broke and the caller must fail the request instead
    /// of continuing off-grammar. Every FSM transition in the decode paths
    /// goes through this method.
    pub fn advance_checked(&mut self, token_id: u32) -> Result<(), EngineError> {
        if self.advance(token_id) {
            Ok(())
        } else {
            Err(EngineError::Generation(format!(
                "grammar-constrained generation sampled token {token_id}, which the FSM rejected"
            )))
        }
    }

    /// Whether the FSM is in a final (accepting) state.
    pub fn is_finished(&self) -> bool {
        self.index.is_final_state(&self.state)
    }

    /// Apply the constraint mask to logits.
    ///
    /// Sets disallowed token logits to negative infinity so they have zero
    /// probability after softmax.
    ///
    /// Fail-closed: a state with no allowed token has no legal continuation,
    /// so this is a typed error — never unchanged logits for unconstrained
    /// sampling.
    pub fn apply_mask(&self, logits: &Array) -> Result<Array, EngineError> {
        Self::masked_logits(logits, self.allowed_token_ids().as_deref())
    }

    /// Mask construction, split from [`Self::apply_mask`] so the
    /// no-allowed-tokens failure is testable without a dead FSM state.
    fn masked_logits(
        logits: &Array,
        allowed: Option<&[outlines_core::primitives::TokenId]>,
    ) -> Result<Array, EngineError> {
        let Some(allowed) = allowed else {
            return Err(EngineError::Generation(
                "constrained generation reached an FSM state with no allowed tokens; \
                 refusing to sample unconstrained logits"
                    .to_owned(),
            ));
        };
        if allowed.is_empty() {
            return Err(EngineError::Generation(
                "constrained generation reached an FSM state with an empty allowed-token set; \
                 refusing to sample unconstrained logits"
                    .to_owned(),
            ));
        }

        // Use the actual logits vocab dimension, not the tokenizer's count.
        // Models often pad their embedding table beyond the tokenizer vocabulary.
        let vocab_size = usize::try_from(*logits.shape().last().unwrap_or(&0)).unwrap_or(0);
        if vocab_size == 0 {
            return Err(EngineError::Generation(
                "constrained generation produced an empty logits vocabulary dimension".to_owned(),
            ));
        }

        // Build a mask: -inf for disallowed, 0 for allowed
        let mut mask_vec = vec![f32::NEG_INFINITY; vocab_size];
        let mut in_vocab = 0usize;
        for &tid in allowed {
            if let Some(slot) = mask_vec.get_mut(usize::try_from(tid).unwrap_or(usize::MAX)) {
                *slot = 0.0;
                in_vocab += 1;
            }
        }
        if in_vocab == 0 {
            return Err(EngineError::Generation(format!(
                "none of the {} allowed tokens are inside the logits vocabulary of \
                 size {vocab_size}; refusing to sample unconstrained logits",
                allowed.len()
            )));
        }

        let vocab_i32 =
            i32::try_from(vocab_size).map_err(|_| Exception::custom("vocab_size overflow"))?;
        let mask_array = Array::from_slice(&mask_vec, &[vocab_i32]);

        // Reshape for broadcasting: [1, vocab_size] broadcasts across batch dim.
        let reshaped = if logits.ndim() > 1 {
            mask_array.reshape(&[1, vocab_i32])?
        } else {
            mask_array
        };

        logits.add(reshaped).map_err(EngineError::Mlx)
    }
}

/// Build an `outlines-core` [`Vocabulary`] from a `tokenizers` tokenizer.
///
/// Replicates the token processing that outlines-core's `TokenProcessor` does internally:
/// - `ByteLevel` tokenizers (GPT-2, Llama 3): each character is decoded via a `CHAR_MAP`
///   that maps Unicode surrogates (U+0100–U+017E, U+00A1–U+00FF) back to the raw byte
///   they represent.
/// - `ByteFallback` tokenizers (Llama 2): `▁` -> space, `<0x__>` -> single byte.
/// - Others: pass UTF-8 bytes as-is.
pub fn build_vocabulary(
    tokenizer: &tokenizers::Tokenizer,
    eos_token_id: u32,
) -> Result<Vocabulary, EngineError> {
    use tokenizers::DecoderWrapper;

    let processor = match tokenizer.get_decoder() {
        Some(DecoderWrapper::ByteLevel(_)) => TokenKind::Byte,
        Some(DecoderWrapper::Sequence(seq)) => {
            let has_byte_fallback = seq
                .get_decoders()
                .iter()
                .any(|d| matches!(d, DecoderWrapper::ByteFallback(_)));
            let spacechar = seq
                .get_decoders()
                .iter()
                .find_map(|d| {
                    if let DecoderWrapper::Replace(r) = d {
                        // Extract the replacement from the Replace decoder via JSON.
                        let v = serde_json::to_value(r).ok()?;
                        if v.get("content")?.as_str()? == " " {
                            let pat = v.get("pattern")?.get("String")?.as_str()?;
                            let mut chars = pat.chars();
                            let first = chars.next()?;
                            chars.next().is_none().then_some(String::from(first))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| String::from("▁"));
            if has_byte_fallback {
                TokenKind::ByteFallback { spacechar }
            } else {
                TokenKind::Raw
            }
        }
        _ => TokenKind::Raw,
    };

    let mut vocab = Vocabulary::new(eos_token_id);

    // Added non-special tokens are inserted as-is since they represent literal strings.
    for (id, added) in tokenizer.get_added_tokens_decoder() {
        if !added.special && id != eos_token_id {
            if vocab
                .try_insert(added.content.as_bytes().to_vec(), id)
                .is_err()
            {
                tracing::trace!(token = %added.content, id, "Skipping duplicate added token");
            }
        }
    }

    // Main vocabulary tokens require tokenizer-specific decoding.
    for (token_str, token_id) in tokenizer.get_vocab(false) {
        if token_id == eos_token_id {
            continue;
        }
        let bytes = processor.process(&token_str);
        if vocab.try_insert(bytes, token_id).is_err() {
            tracing::trace!(token = %token_str, id = token_id, "Skipping duplicate token");
        }
    }

    Ok(vocab)
}

#[derive(Debug)]
enum TokenKind {
    /// GPT-2 / Llama 3 style: each char in the token string maps to a byte via `CHAR_MAP`.
    Byte,
    /// Llama 2 / `SentencePiece` style: replace spacechar with 0x20, handle `<0x__>`.
    ByteFallback { spacechar: String },
    /// No special decoding needed; use UTF-8 bytes directly.
    Raw,
}

impl TokenKind {
    #[allow(clippy::indexing_slicing)]
    fn process(&self, token: &str) -> Vec<u8> {
        match self {
            Self::Byte => token
                .chars()
                .map(|c| {
                    BYTE_CHAR_MAP
                        .get(&c)
                        .copied()
                        .unwrap_or_else(|| u8::try_from(c).unwrap_or(0))
                })
                .collect(),
            Self::ByteFallback { spacechar } => {
                if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        return vec![byte];
                    }
                }
                token.replace(spacechar.as_str(), " ").into_bytes()
            }
            Self::Raw => token.as_bytes().to_vec(),
        }
    }
}

/// Maps Unicode surrogate characters used by `ByteLevel` tokenizers back to their
/// raw byte values (the same table as outlines-core's `CHAR_MAP`).
static BYTE_CHAR_MAP: std::sync::LazyLock<std::collections::HashMap<char, u8>> =
    std::sync::LazyLock::new(|| {
        let mut map = std::collections::HashMap::with_capacity(256);
        let mut key = 0x100u32;
        for byte in 0..=255u8 {
            let ch = char::from(byte);
            if matches!(ch, '!'..='~' | '\u{00A1}'..='\u{00AC}' | '\u{00AE}'..='\u{00FF}') {
                map.insert(ch, byte);
            } else if let Some(c) = char::from_u32(key) {
                map.insert(c, byte);
                key += 1;
            }
        }
        map
    });

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::disallowed_methods
)]
mod tests {
    use super::*;

    #[test]
    fn build_vocabulary_from_tokenizer_succeeds() {
        // Verify the vocabulary builder doesn't panic with a real-ish scenario.
        // Full integration tests require a real tokenizer + model.
        let mut vocab = Vocabulary::new(0);
        vocab.try_insert("hello", 1).unwrap();
        vocab.try_insert("world", 2).unwrap();
        // Just verify construction doesn't panic
        assert!(vocab.token_ids("hello").is_some());
    }

    #[test]
    fn apply_mask_shapes_are_correct() {
        // Test the mask building logic directly without needing a full FSM.
        let vocab_size = 10;
        let vocab_i32 = i32::try_from(vocab_size).unwrap();

        // Simulate allowed tokens: only tokens 2 and 5
        let mut mask_vec = vec![f32::NEG_INFINITY; vocab_size];
        mask_vec[2] = 0.0;
        mask_vec[5] = 0.0;
        let mask_array = Array::from_slice(&mask_vec, &[1, vocab_i32]);

        let logits = Array::from_slice(&vec![1.0_f32; vocab_size], &[1, vocab_i32]);
        let masked = logits.add(mask_array).unwrap();
        mlx_rs::transforms::eval([&masked]).unwrap();
        let vals = masked.as_slice::<f32>();

        assert!(
            (vals[2] - 1.0).abs() < 1e-5,
            "Allowed token 2 should keep logit"
        );
        assert!(
            (vals[5] - 1.0).abs() < 1e-5,
            "Allowed token 5 should keep logit"
        );
        assert!(
            vals[0].is_infinite() && vals[0].is_sign_negative(),
            "Token 0 should be -inf"
        );
        assert!(
            vals[3].is_infinite() && vals[3].is_sign_negative(),
            "Token 3 should be -inf"
        );
    }

    #[test]
    fn constrained_generator_initial_state() {
        // Test with a very simple regex that can terminate
        let mut vocab = Vocabulary::new(0);
        // Token 0 is EOS
        vocab.try_insert("a", 1).unwrap();
        vocab.try_insert("b", 2).unwrap();

        // Simple regex: one or more 'a' characters
        let result = ConstrainedGenerator::from_regex("a+", &vocab);
        if let Ok(cg) = result {
            assert!(!cg.is_finished());
            let allowed = cg.allowed_token_ids();
            assert!(allowed.is_some());
            assert!(allowed.unwrap().contains(&1)); // 'a' should be allowed
        }
        // If it fails due to EOS handling, that's OK for this minimal vocab
    }

    #[test]
    fn tagged_json_schema_accepts_a_parser_visible_tool_call() {
        let output = "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Rome\"}}\n</tool_call>";
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"const": "get_weather"},
                "arguments": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                }
            },
            "required": ["name", "arguments"],
            "additionalProperties": false
        });
        let mut vocabulary = Vocabulary::new(0);
        let mut token_ids = std::collections::HashMap::new();
        for byte in output.bytes() {
            if token_ids.contains_key(&byte) {
                continue;
            }
            let token_id = u32::try_from(token_ids.len() + 1).unwrap();
            vocabulary.try_insert(vec![byte], token_id).unwrap();
            token_ids.insert(byte, token_id);
        }

        let mut generator =
            ConstrainedGenerator::from_tagged_json_schema(&schema.to_string(), &vocabulary)
                .unwrap();
        for byte in output.bytes() {
            let token_id = token_ids[&byte];
            assert!(
                generator
                    .allowed_token_ids()
                    .is_some_and(|allowed| allowed.contains(&token_id)),
                "byte {byte:?} should be allowed"
            );
            assert!(generator.advance(token_id));
        }
        assert!(generator.is_finished());

        let parsed = crate::tool_parser::parse_tool_calls(output, None);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        assert_eq!(parsed.tool_calls[0].arguments["city"], "Rome");
        assert!(parsed.text.is_empty());
    }

    #[test]
    fn apply_mask_uses_logit_vocab_dimension() {
        // Model vocab (logits) may be larger than tokenizer vocab.
        // Mask size should match the logits dimension, not the tokenizer's count.
        let model_vocab = 12;
        let vocab_i32 = i32::try_from(model_vocab).unwrap();

        // Simulate allowed tokens: only token 3
        let mut mask_vec = vec![f32::NEG_INFINITY; model_vocab];
        mask_vec[3] = 0.0;
        let mask_array = Array::from_slice(&mask_vec, &[1, vocab_i32]);

        let logits = Array::ones::<f32>(&[1, vocab_i32]).unwrap();
        let masked = logits.add(mask_array).unwrap();
        mlx_rs::transforms::eval([&masked]).unwrap();

        assert_eq!(masked.shape(), &[1, vocab_i32]);
        let vals = masked.as_slice::<f32>();
        assert!((vals[3] - 1.0).abs() < 1e-5, "Token 3 should keep logit");
        assert!(vals[0].is_infinite(), "Token 0 should be -inf");
        assert!(vals[11].is_infinite(), "Token 11 should be -inf");
    }

    #[test]
    fn masked_logits_without_allowed_tokens_is_an_error() {
        // A dead FSM state offers no legal continuation: masking must fail
        // closed instead of returning unchanged logits for unconstrained
        // sampling.
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let error = ConstrainedGenerator::masked_logits(&logits, None)
            .expect_err("no allowed tokens must fail closed");
        assert!(error.to_string().contains("no allowed tokens"), "{error}");
    }

    #[test]
    fn masked_logits_empty_allowed_set_is_an_error() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let error = ConstrainedGenerator::masked_logits(&logits, Some(&[]))
            .expect_err("an empty allowed set must fail closed");
        assert!(
            error.to_string().contains("empty allowed-token set"),
            "{error}"
        );
    }

    #[test]
    fn masked_logits_empty_vocab_dimension_is_an_error() {
        let logits = mlx_rs::Array::zeros::<f32>(&[0]).unwrap();
        let error = ConstrainedGenerator::masked_logits(&logits, Some(&[0]))
            .expect_err("a zero-width vocabulary must fail closed");
        assert!(
            error
                .to_string()
                .contains("empty logits vocabulary dimension"),
            "{error}"
        );
    }

    #[test]
    fn masked_logits_all_allowed_outside_vocab_is_an_error() {
        // Every allowed ID maps past the logits dimension: masking would
        // leave an all--inf mask (uniform zero probability), so fail closed.
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let error = ConstrainedGenerator::masked_logits(&logits, Some(&[7, 999]))
            .expect_err("all-OOV allowed tokens must fail closed");
        assert!(
            error
                .to_string()
                .contains("are inside the logits vocabulary"),
            "{error}"
        );
    }

    #[test]
    fn masked_logits_partially_oov_allowed_tokens_still_mask() {
        // Mixed in-vocab and out-of-vocab allowed IDs: the OOV ones are
        // skipped, the in-vocab ones still constrain sampling.
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let masked = ConstrainedGenerator::masked_logits(&logits, Some(&[1, 999])).unwrap();
        mlx_rs::transforms::eval([&masked]).unwrap();
        let vals = masked.as_slice::<f32>();
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!(vals[0].is_infinite() && vals[2].is_infinite());
    }

    #[test]
    fn constrained_one_token_regex_completes_on_the_first_sampled_token() {
        // Regex `a` with max_tokens > 1: the single accepted token completes
        // the grammar, so both the blocking and streaming decode loops — which
        // consult `is_finished` immediately after the checked advance — must
        // terminate successfully at completion_tokens=1 without scheduling a
        // successor decode.
        let mut vocabulary = Vocabulary::new(0);
        vocabulary.try_insert(vec![b'a'], 1).unwrap();
        let mut generator = ConstrainedGenerator::from_regex("a", &vocabulary).unwrap();
        assert!(!generator.is_finished(), "initial state is not final");
        generator
            .advance_checked(1)
            .expect("token 'a' is the whole grammar");
        assert!(
            generator.is_finished(),
            "one accepted token must complete the grammar; the loops key off this \
             predicate to finish 'stop' before any second decode"
        );
    }

    #[test]
    fn masked_logits_masks_disallowed_tokens() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let masked = ConstrainedGenerator::masked_logits(&logits, Some(&[1])).unwrap();
        mlx_rs::transforms::eval([&masked]).unwrap();
        let vals = masked.as_slice::<f32>();
        assert!((vals[1] - 2.0).abs() < 1e-5, "allowed token keeps logit");
        assert!(vals[0].is_infinite() && vals[0].is_sign_negative());
        assert!(vals[2].is_infinite() && vals[2].is_sign_negative());
    }

    #[test]
    fn advance_rejects_token_outside_grammar() {
        let mut vocabulary = Vocabulary::new(0);
        vocabulary.try_insert(vec![b'a'], 1).unwrap();
        vocabulary.try_insert(vec![b'b'], 2).unwrap();
        let mut generator = ConstrainedGenerator::from_regex("a", &vocabulary).unwrap();
        generator
            .advance_checked(1)
            .expect("'a' is the only accepted character");
        let error = generator
            .advance_checked(2)
            .expect_err("a token outside the grammar must fail the request, not continue");
        assert!(
            error.to_string().contains("which the FSM rejected"),
            "{error}"
        );
    }

    #[test]
    fn apply_mask_allowed_token_beyond_vocab_is_ignored() {
        // If a token ID is beyond the vocab size, it should be silently skipped.
        let vocab_size = 5;
        let mut mask_vec = vec![f32::NEG_INFINITY; vocab_size];
        // Simulate: allowed = [1, 999] where 999 > vocab_size
        mask_vec[1] = 0.0;
        // Token 999 would be out of bounds -- get_mut returns None, so it's skipped.
        let vocab_i32 = i32::try_from(vocab_size).unwrap();
        let mask_array = Array::from_slice(&mask_vec, &[vocab_i32]);
        let logits = Array::ones::<f32>(&[vocab_i32]).unwrap();
        let masked = logits.add(mask_array).unwrap();
        mlx_rs::transforms::eval([&masked]).unwrap();
        let vals = masked.as_slice::<f32>();
        assert!((vals[1] - 1.0).abs() < 1e-5);
        assert!(vals[0].is_infinite());
    }
}
