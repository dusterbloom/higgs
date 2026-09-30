//! Keep chat structure out of the text channel.
//!
//! A chat template flattens messages into one string that is then tokenized
//! with special-token parsing on. Content that merely *quotes* a special
//! token (source code containing `"<|im_start|>"`) therefore became the real
//! control token: the model saw a fake message boundary, and the retained
//! session splice — which finds message boundaries by those markers — gave
//! up and forced a full cold re-prefill.
//!
//! Fix: special-token literals inside message content are swapped for
//! placeholders before rendering ([`SpecialLiterals::escape`]).
//! [`SpecialLiterals::encode`] turns each placeholder into that literal's
//! plain-text token ids, and [`SpecialLiterals::decode`] maps exactly those
//! id runs back to placeholders, so retained tokens and fresh renders meet in
//! the same "escaped" text where every marker is structural.
//!
//! Content without special literals (nearly every request) takes a
//! zero-copy fast path and tokenizes byte-identically to before.

use std::borrow::Cow;
use std::collections::HashSet;

use tokenizers::Tokenizer;

use crate::error::EngineError;

// Unicode noncharacters: never valid in interchanged text, so they cannot
// collide with real content in practice.
// ponytail: content that itself contains `\u{FDD0}<n>\u{FDD1}` is read as
// placeholder n and reaches the model as that literal's *plain* text —
// harmless (never a control token); reject such content if it ever matters.
const OPEN: char = '\u{FDD0}';
const CLOSE: char = '\u{FDD1}';

struct Literal {
    text: String,
    plain_ids: Vec<u32>,
}

pub struct SpecialLiterals {
    /// Longest text first, so overlapping literals match maximally.
    literals: Vec<Literal>,
    special_ids: HashSet<u32>,
    /// Marker the server itself inserts into content (vision image marker);
    /// it must stay structural.
    exempt: Option<&'static str>,
}

impl SpecialLiterals {
    /// Build from the tokenizer's special added tokens. `exempt` is the
    /// model's image marker text, if any.
    pub fn from_tokenizer(
        tokenizer: &Tokenizer,
        exempt: Option<&'static str>,
    ) -> Result<Self, EngineError> {
        let mut plain = tokenizer.clone();
        plain.set_encode_special_tokens(true);
        let mut special_ids = HashSet::new();
        let mut literals = Vec::new();
        for (id, token) in tokenizer.get_added_tokens_decoder() {
            if !token.special || token.content.is_empty() {
                continue;
            }
            special_ids.insert(id);
            let plain_ids = plain
                .encode(token.content.as_str(), false)
                .map_err(|e| EngineError::Tokenization(e.to_string()))?
                .get_ids()
                .to_vec();
            literals.push(Literal {
                text: token.content,
                plain_ids,
            });
        }
        literals.sort_by(|a, b| b.text.len().cmp(&a.text.len()).then(a.text.cmp(&b.text)));
        Ok(Self {
            literals,
            special_ids,
            exempt,
        })
    }

    /// Replace special-token literals in untrusted content with placeholders.
    pub fn escape<'a>(&self, content: &'a str) -> Cow<'a, str> {
        // ponytail: O(content × literals) scan; fine for chat-model special
        // sets (~14-40). Swap in aho-corasick if a large special vocab shows
        // up in profiles.
        if !self.literals.iter().any(|l| content.contains(&l.text)) {
            return Cow::Borrowed(content);
        }
        let mut out = String::with_capacity(content.len());
        let Some(exempt) = self.exempt.filter(|m| content.contains(m)) else {
            self.escape_into(content, &mut out);
            return Cow::Owned(out);
        };
        for (i, piece) in content.split(exempt).enumerate() {
            if i > 0 {
                out.push_str(exempt);
            }
            self.escape_into(piece, &mut out);
        }
        Cow::Owned(out)
    }

    fn escape_into(&self, mut rest: &str, out: &mut String) {
        while !rest.is_empty() {
            if let Some((index, literal)) = self
                .literals
                .iter()
                .enumerate()
                .find(|(_, l)| rest.starts_with(&l.text))
            {
                push_placeholder(out, index);
                rest = &rest[literal.text.len()..];
                continue;
            }
            let Some(ch) = rest.chars().next() else { break };
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }

    /// Tokenize rendered (escaped) text: placeholders become the literal's
    /// plain ids; everything else tokenizes exactly as before.
    pub fn encode(&self, tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, EngineError> {
        let mut ids = Vec::new();
        let mut rest = text;
        while let Some((before, index, after)) = self.next_placeholder(rest) {
            encode_into(tokenizer, before, &mut ids)?;
            ids.extend_from_slice(&self.literals[index].plain_ids);
            rest = after;
        }
        encode_into(tokenizer, rest, &mut ids)?;
        Ok(ids)
    }

    /// Detokenize to escaped text: runs equal to a literal's plain ids come
    /// back as its placeholder, so the result compares 1:1 with a render.
    pub fn decode(&self, tokenizer: &Tokenizer, ids: &[u32]) -> Result<String, EngineError> {
        let mut out = String::new();
        let mut run_start = 0;
        let mut pos = 0;
        while pos < ids.len() {
            let hit = self.literals.iter().enumerate().find(|(_, l)| {
                !l.plain_ids.is_empty()
                    && !self.special_ids.contains(&ids[pos])
                    && ids[pos..].starts_with(&l.plain_ids)
            });
            let Some((index, literal)) = hit else {
                pos += 1;
                continue;
            };
            decode_into(tokenizer, &ids[run_start..pos], &mut out)?;
            push_placeholder(&mut out, index);
            pos += literal.plain_ids.len();
            run_start = pos;
        }
        decode_into(tokenizer, &ids[run_start..], &mut out)?;
        Ok(out)
    }

    fn next_placeholder<'a>(&self, text: &'a str) -> Option<(&'a str, usize, &'a str)> {
        let mut search = 0;
        loop {
            let open = search + text[search..].find(OPEN)?;
            let body = &text[open + OPEN.len_utf8()..];
            let parsed = body.find(CLOSE).and_then(|close| {
                let index = body[..close].parse::<usize>().ok()?;
                (index < self.literals.len()).then_some((index, close))
            });
            if let Some((index, close)) = parsed {
                return Some((&text[..open], index, &body[close + CLOSE.len_utf8()..]));
            }
            // Stray OPEN that is not one of ours: keep it as text.
            search = open + OPEN.len_utf8();
        }
    }
}

fn push_placeholder(out: &mut String, index: usize) {
    out.push(OPEN);
    out.push_str(&index.to_string());
    out.push(CLOSE);
}

fn encode_into(tokenizer: &Tokenizer, text: &str, ids: &mut Vec<u32>) -> Result<(), EngineError> {
    if text.is_empty() {
        return Ok(());
    }
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| EngineError::Tokenization(e.to_string()))?;
    ids.extend_from_slice(encoding.get_ids());
    Ok(())
}

fn decode_into(tokenizer: &Tokenizer, ids: &[u32], out: &mut String) -> Result<(), EngineError> {
    if ids.is_empty() {
        return Ok(());
    }
    let text = tokenizer
        .decode(ids, false)
        .map_err(|e| EngineError::Tokenization(e.to_string()))?;
    out.push_str(&text);
    Ok(())
}

#[cfg(test)]
/// Byte-level BPE with no merges (one token per byte) plus real special
/// tokens — the same shape as Qwen/ChatML tokenizers, but hermetic.
pub(crate) fn chatml_test_tokenizer() -> Tokenizer {
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::AddedToken;

    let vocab: tokenizers::models::bpe::Vocab = ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(id, ch)| (ch.to_string(), u32::try_from(id).unwrap()))
        .collect();
    let bpe = BPE::builder().vocab_and_merges(vocab, vec![]).build().unwrap();
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tok.with_decoder(Some(ByteLevelDecoder::default().add_prefix_space(false)));
    tok.add_special_tokens([
        AddedToken::from("<|im_start|>", true),
        AddedToken::from("<|im_end|>", true),
        AddedToken::from("<|image_pad|>", true),
    ]);
    tok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_markers_reach_the_model_as_text_and_round_trip() {
        let tok = chatml_test_tokenizer();
        let lits = SpecialLiterals::from_tokenizer(&tok, None).unwrap();
        let im_start = tok.token_to_id("<|im_start|>").unwrap();
        let quoted = r#"const S: &str = "<|im_start|>";"#;

        let rendered = format!("<|im_start|>user\n{}<|im_end|>\n", lits.escape(quoted));
        let ids = lits.encode(&tok, &rendered).unwrap();

        // Exactly one structural <|im_start|>: the quoted one is plain text.
        assert_eq!(ids.iter().filter(|&&id| id == im_start).count(), 1);
        // Decoding lands back on the same escaped text a fresh render gives.
        assert_eq!(lits.decode(&tok, &ids).unwrap(), rendered);
    }

    #[test]
    fn plain_content_is_untouched_and_tokenizes_as_before() {
        let tok = chatml_test_tokenizer();
        let lits = SpecialLiterals::from_tokenizer(&tok, None).unwrap();
        let rendered = "<|im_start|>user\nhello <b>world</b><|im_end|>\n";

        assert!(matches!(lits.escape("hello <b>world</b>"), Cow::Borrowed(_)));
        assert_eq!(
            lits.encode(&tok, rendered).unwrap(),
            tok.encode(rendered, false).unwrap().get_ids()
        );
    }

    #[test]
    fn server_inserted_image_marker_stays_structural() {
        let tok = chatml_test_tokenizer();
        let lits = SpecialLiterals::from_tokenizer(&tok, Some("<|image_pad|>")).unwrap();
        let pad = tok.token_to_id("<|image_pad|>").unwrap();
        let im_end = tok.token_to_id("<|im_end|>").unwrap();

        let content = lits.escape("see <|image_pad|> not <|im_end|>");
        let ids = lits.encode(&tok, &content).unwrap();

        assert_eq!(ids.iter().filter(|&&id| id == pad).count(), 1);
        assert!(!ids.contains(&im_end));
    }
}
