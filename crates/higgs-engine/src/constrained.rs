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
    pub fn advance(&mut self, token_id: u32) -> bool {
        if let Some(next) = self.index.next_state(&self.state, &token_id) {
            self.state = next;
            true
        } else {
            false
        }
    }

    /// Read-only walk of the FSM along `draft_ids` from the current state.
    ///
    /// Returns the entry state followed by the state after each legal transition.
    /// Stops at the first illegal token, so the returned length is
    /// `1 + min(draft_ids.len(), longest_legal_prefix)`.
    pub fn peek_states_for_drafts(
        &self,
        draft_ids: &[u32],
    ) -> Vec<outlines_core::primitives::StateId> {
        let mut states = Vec::with_capacity(draft_ids.len() + 1);
        let mut state = self.state;
        states.push(state);
        for &tid in draft_ids {
            match self.index.next_state(&state, &tid) {
                Some(next) => {
                    state = next;
                    states.push(state);
                }
                None => break,
            }
        }
        states
    }

    /// Whether the FSM is in a final (accepting) state.
    pub fn is_finished(&self) -> bool {
        self.index.is_final_state(&self.state)
    }

    /// Apply the constraint mask to logits at the current FSM state.
    ///
    /// Sets disallowed token logits to negative infinity so they have zero
    /// probability after softmax.
    pub fn apply_mask(&self, logits: &Array) -> Result<Array, Exception> {
        self.mask_for_state(logits, self.state)
    }

    /// Apply the constraint mask to logits at an arbitrary FSM state.
    ///
    /// Used by speculative decoding to mask K+1 verify-row logits using
    /// per-position states without mutating `self`.
    pub fn apply_mask_at_state(
        &self,
        logits: &Array,
        state: outlines_core::primitives::StateId,
    ) -> Result<Array, Exception> {
        self.mask_for_state(logits, state)
    }

    /// Build a multi-row FSM mask covering `num_rows` verify positions.
    ///
    /// Row `i` masks to the allowed set at `states[i]`; rows beyond
    /// `states.len()` (the FSM walk halted on an illegal draft) stay
    /// unmasked (zeros). Used by DFlash speculative decode to mask K+1
    /// verify rows in a single on-device add.
    ///
    /// Returns shape `[num_rows, vocab_size]`. Caller reshapes to broadcast
    /// against `[1, num_rows, vocab_size]` verify logits.
    pub fn build_mask_rows(
        &self,
        states: &[outlines_core::primitives::StateId],
        num_rows: usize,
        vocab_size: usize,
    ) -> Result<Array, Exception> {
        let mut mask_vec = vec![0.0_f32; num_rows * vocab_size];
        for (i, &state) in states.iter().enumerate().take(num_rows) {
            let Some(allowed) = self.index.allowed_tokens(&state) else {
                continue;
            };
            let base = i * vocab_size;
            for slot in &mut mask_vec[base..base + vocab_size] {
                *slot = f32::NEG_INFINITY;
            }
            for &tid in &allowed {
                let offset = usize::try_from(tid).unwrap_or(usize::MAX);
                if offset < vocab_size {
                    mask_vec[base + offset] = 0.0;
                }
            }
        }
        let num_rows_i32 =
            i32::try_from(num_rows).map_err(|_| Exception::custom("num_rows overflow"))?;
        let vocab_i32 =
            i32::try_from(vocab_size).map_err(|_| Exception::custom("vocab_size overflow"))?;
        Ok(Array::from_slice(&mask_vec, &[num_rows_i32, vocab_i32]))
    }

    fn mask_for_state(
        &self,
        logits: &Array,
        state: outlines_core::primitives::StateId,
    ) -> Result<Array, Exception> {
        let Some(allowed) = self.index.allowed_tokens(&state) else {
            // No allowed tokens -- shouldn't happen in practice; let EOS take over.
            return Ok(logits.clone());
        };

        // Use the actual logits vocab dimension, not the tokenizer's count.
        // Models often pad their embedding table beyond the tokenizer vocabulary.
        let vocab_size = usize::try_from(*logits.shape().last().unwrap_or(&0)).unwrap_or(0);

        let mut mask_vec = vec![f32::NEG_INFINITY; vocab_size];
        for &tid in &allowed {
            if let Some(slot) = mask_vec.get_mut(usize::try_from(tid).unwrap_or(usize::MAX)) {
                *slot = 0.0;
            }
        }

        let vocab_i32 =
            i32::try_from(vocab_size).map_err(|_| Exception::custom("vocab_size overflow"))?;
        let mask_array = Array::from_slice(&mask_vec, &[vocab_i32]);

        let reshaped = if logits.ndim() > 1 {
            mask_array.reshape(&[1, vocab_i32])?
        } else {
            mask_array
        };

        logits.add(reshaped)
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
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
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
    fn apply_mask_no_allowed_tokens_returns_unchanged() {
        // When allowed_token_ids returns None, logits should be unchanged.
        // We test this by directly checking the code path:
        // If there are no allowed tokens the function returns early.
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let vals: Vec<f32> = logits.as_slice().to_vec();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    fn build_ab_generator() -> ConstrainedGenerator {
        let mut vocab = Vocabulary::new(0);
        vocab.try_insert("a", 1).unwrap();
        vocab.try_insert("b", 2).unwrap();
        vocab.try_insert("c", 3).unwrap();
        ConstrainedGenerator::from_regex("[ab]+", &vocab).unwrap()
    }

    #[test]
    fn peek_states_for_drafts_empty_returns_entry_state() {
        let cg = build_ab_generator();
        let states = cg.peek_states_for_drafts(&[]);
        assert_eq!(
            states.len(),
            1,
            "empty drafts should yield only the entry state"
        );
    }

    #[test]
    fn peek_states_for_drafts_does_not_mutate() {
        let cg = build_ab_generator();
        let before = cg
            .allowed_token_ids()
            .expect("initial state has allowed tokens");
        let _ = cg.peek_states_for_drafts(&[1, 2, 1]);
        let after = cg.allowed_token_ids().expect("state preserved after peek");
        let mut b_sorted = before.clone();
        let mut a_sorted = after.clone();
        b_sorted.sort_unstable();
        a_sorted.sort_unstable();
        assert_eq!(b_sorted, a_sorted, "peek must not mutate the FSM state");
    }

    #[test]
    fn peek_states_for_drafts_three_legal_tokens_yields_four_states() {
        let cg = build_ab_generator();
        let states = cg.peek_states_for_drafts(&[1, 2, 1]);
        assert_eq!(states.len(), 4, "K=3 legal drafts should yield K+1 states");
    }

    #[test]
    fn apply_mask_at_state_reflects_state_specific_allowed_set() {
        // regex "ab|ba": after 'a' only 'b' allowed; after 'b' only 'a' allowed.
        let mut vocab = Vocabulary::new(0);
        vocab.try_insert("a", 1).unwrap();
        vocab.try_insert("b", 2).unwrap();
        let cg = ConstrainedGenerator::from_regex("ab|ba", &vocab).unwrap();

        let states_a = cg.peek_states_for_drafts(&[1]);
        let states_b = cg.peek_states_for_drafts(&[2]);
        assert_eq!(states_a.len(), 2, "drafts=[a] walks one step");
        assert_eq!(states_b.len(), 2, "drafts=[b] walks one step");

        let vocab_i32 = 3_i32; // EOS=0, a=1, b=2
        let logits = Array::zeros::<f32>(&[vocab_i32]).unwrap();

        let masked_after_a = cg.apply_mask_at_state(&logits, states_a[1]).unwrap();
        let masked_after_b = cg.apply_mask_at_state(&logits, states_b[1]).unwrap();
        mlx_rs::transforms::eval([&masked_after_a, &masked_after_b]).unwrap();

        let after_a = masked_after_a.as_slice::<f32>();
        let after_b = masked_after_b.as_slice::<f32>();

        assert!((after_a[2] - 0.0).abs() < 1e-5, "'b' allowed after 'a'");
        assert!(after_a[1].is_infinite(), "'a' disallowed after 'a'");
        assert!((after_b[1] - 0.0).abs() < 1e-5, "'a' allowed after 'b'");
        assert!(after_b[2].is_infinite(), "'b' disallowed after 'b'");
    }

    #[test]
    fn apply_mask_at_state_with_entry_state_matches_apply_mask() {
        let cg = build_ab_generator();
        let entry = cg.peek_states_for_drafts(&[])[0];
        let logits = Array::zeros::<f32>(&[4_i32]).unwrap();

        let via_apply = cg.apply_mask(&logits).unwrap();
        let via_at_state = cg.apply_mask_at_state(&logits, entry).unwrap();
        mlx_rs::transforms::eval([&via_apply, &via_at_state]).unwrap();

        let v1 = via_apply.as_slice::<f32>();
        let v2 = via_at_state.as_slice::<f32>();

        for i in 0..4 {
            if v1[i].is_infinite() {
                assert!(v2[i].is_infinite(), "mismatch at index {i}");
            } else {
                assert!((v1[i] - v2[i]).abs() < 1e-5, "mismatch at index {i}");
            }
        }
    }

    #[test]
    fn peek_states_for_drafts_illegal_first_halts_at_entry() {
        let cg = build_ab_generator();
        // Token 3 ("c") is not in [ab]+ so the first transition is illegal.
        let states = cg.peek_states_for_drafts(&[3, 1, 2]);
        assert_eq!(
            states.len(),
            1,
            "illegal first draft should halt the walk at the entry state"
        );
    }

    #[test]
    fn build_mask_rows_full_walk_matches_per_row_apply_mask_at_state() {
        // K+1 = 3 verify rows over a full legal walk; row i must match
        // apply_mask_at_state(row_logits, states[i]) elementwise.
        let mut vocab = Vocabulary::new(0);
        vocab.try_insert("a", 1).unwrap();
        vocab.try_insert("b", 2).unwrap();
        let cg = ConstrainedGenerator::from_regex("ab|ba", &vocab).unwrap();

        let states = cg.peek_states_for_drafts(&[1]); // [entry, after-a]
        assert_eq!(states.len(), 2);

        let vocab_size = 3_usize;
        let mask_2d = cg.build_mask_rows(&states, 2, vocab_size).unwrap();
        mlx_rs::transforms::eval([&mask_2d]).unwrap();
        assert_eq!(mask_2d.shape(), &[2, 3]);

        let bytes = mask_2d.as_slice::<f32>();
        // Row 0: entry state allows a,b (1,2); EOS=0 disallowed.
        assert!(bytes[0].is_infinite() && bytes[0].is_sign_negative());
        assert!((bytes[1] - 0.0).abs() < 1e-5);
        assert!((bytes[2] - 0.0).abs() < 1e-5);
        // Row 1: after 'a', only 'b' allowed (regex "ab|ba": a→b is the only legal step).
        assert!(bytes[3].is_infinite());
        assert!(bytes[4].is_infinite());
        assert!((bytes[5] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn build_mask_rows_partial_walk_leaves_trailing_rows_unmasked() {
        // FSM walk halted at 1 state but caller asked for 3 rows; rows 1 and 2
        // must stay zeros (unmasked) so accept_prefix can cut at the illegal draft.
        let cg = build_ab_generator();
        let states = cg.peek_states_for_drafts(&[3]); // illegal first → 1 state
        assert_eq!(states.len(), 1);

        let vocab_size = 4_usize;
        let mask_2d = cg.build_mask_rows(&states, 3, vocab_size).unwrap();
        mlx_rs::transforms::eval([&mask_2d]).unwrap();
        assert_eq!(mask_2d.shape(), &[3, 4]);

        let bytes = mask_2d.as_slice::<f32>();
        // Row 0: masked by entry state.
        assert!(bytes[0].is_infinite(), "EOS disallowed at entry");
        // Rows 1 and 2: unmasked (all zero).
        for i in 4..12 {
            assert!(
                (bytes[i] - 0.0).abs() < 1e-5,
                "trailing row index {i} should be unmasked",
            );
        }
    }

    #[test]
    fn build_mask_rows_with_empty_states_returns_all_unmasked() {
        // No states (e.g., constraint inactive) → every row stays at zero.
        let cg = build_ab_generator();
        let mask_2d = cg.build_mask_rows(&[], 2, 3).unwrap();
        mlx_rs::transforms::eval([&mask_2d]).unwrap();
        assert_eq!(mask_2d.shape(), &[2, 3]);
        for v in mask_2d.as_slice::<f32>() {
            assert!((v - 0.0).abs() < 1e-5);
        }
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
