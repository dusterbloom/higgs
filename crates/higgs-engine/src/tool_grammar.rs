//! Builds a regex that constrains generation to a well-formed tool call:
//! `<tool_call>{"name":"<tool>","arguments":<args>}</tool_call>`.
//!
//! The `<args>` sub-regex is derived per tool from that tool's JSON-Schema via
//! `outlines-core`. The union over all tools is assembled here as a regex
//! alternation, so we do not depend on outlines' `oneOf` support. The envelope
//! and the `{"name":…,"arguments":…}` key order match exactly what
//! [`crate::tool_parser`] expects and what Qwen-family templates emit, so a
//! constrained call is guaranteed to round-trip through the existing parser.
//!
//! Used by the server route to apply a from-token-0 constraint when a request
//! forces a tool call (`tool_choice: "required"`).

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use outlines_core::index::Index;
use outlines_core::json_schema;
use outlines_core::vocabulary::Vocabulary;

use crate::constrained::{ConstrainedGenerator, build_vocabulary};
use crate::error::EngineError;

/// Flexible JSON whitespace between structural tokens.
const WS: &str = "[ \\t\\n\\r]*";

/// Permissive "any JSON object" fallback regex, used when a tool's parameter
/// schema is absent or cannot be compiled by outlines-core.
const PERMISSIVE_OBJECT: &str = "\\{[^{}]*\\}";

/// Build a regex matching a single `<tool_call>…</tool_call>` block whose body
/// is one of `tools`, with `arguments` constrained to that tool's parameter
/// schema.
///
/// Falls back to a permissive JSON object for any tool whose schema
/// outlines-core cannot compile, so one exotic schema never disables the
/// constraint for the rest. Tools without a usable `name` are skipped.
pub fn tool_call_regex(tools: &[serde_json::Value]) -> String {
    let mut branches: Vec<String> = Vec::new();
    for tool in tools {
        // OpenAI tool defs nest under `function`; tolerate a bare def too.
        let func = tool.get("function").unwrap_or(tool);
        let Some(name) = func.get("name").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let args = args_value_regex(func.get("parameters"));
        branches.push(tool_branch_regex(name, &args));
    }

    if branches.is_empty() {
        // No usable tools: a never-completing pattern. Callers should avoid
        // applying a tool constraint when there are no tools, but this keeps
        // the function total.
        return String::from("<tool_call>(?!)</tool_call>");
    }

    let mut out = String::from("<tool_call>");
    out.push_str(WS);
    out.push_str("(?:");
    out.push_str(&branches.join("|"));
    out.push(')');
    out.push_str(WS);
    out.push_str("</tool_call>");
    out
}

/// `\{ "name" : "<name>" , "arguments" : <args> \}` with flexible whitespace.
fn tool_branch_regex(name: &str, args_regex: &str) -> String {
    let mut b = String::new();
    b.push_str("\\{");
    b.push_str(WS);
    b.push_str("\"name\"");
    b.push_str(WS);
    b.push(':');
    b.push_str(WS);
    b.push('"');
    b.push_str(&escape_regex_literal(name));
    b.push('"');
    b.push_str(WS);
    b.push(',');
    b.push_str(WS);
    b.push_str("\"arguments\"");
    b.push_str(WS);
    b.push(':');
    b.push_str(WS);
    b.push_str(args_regex);
    b.push_str(WS);
    b.push_str("\\}");
    b
}

/// Regex for the `arguments` value: the tool's parameter schema compiled by
/// outlines-core, or a permissive JSON object when absent/uncompilable.
fn args_value_regex(params: Option<&serde_json::Value>) -> String {
    let schema_str = match params {
        Some(p) if p.is_object() => p.to_string(),
        _ => String::from(r#"{"type":"object"}"#),
    };
    json_schema::regex_from_str(&schema_str, None, None)
        .or_else(|_| json_schema::regex_from_str(r#"{"type":"object"}"#, None, None))
        .unwrap_or_else(|_| String::from(PERMISSIVE_OBJECT))
}

/// Process-wide cache of built constraint FSMs, mirroring the `GPU_GATE` static
/// idiom in `state.rs`. Building an `Index` over the full model vocab costs
/// ~0.3s (measured), and a forced-tool path re-sends the same tool set every
/// turn. The vocab is cached per model; each `Index` is cached by
/// `(model, regex)` and shared via `Arc`.
static TOOL_CONSTRAINT_CACHE: LazyLock<ConstraintCache> = LazyLock::new(ConstraintCache::new);

/// Return a constraint for `regex`, reusing a cached FSM when one exists for
/// this `(model_name, regex)`. Builds (and caches) the vocab + index on a miss.
pub fn cached_tool_constraint(
    model_name: &str,
    tokenizer: &tokenizers::Tokenizer,
    eos_token_id: u32,
    regex: &str,
) -> Result<ConstrainedGenerator, EngineError> {
    TOOL_CONSTRAINT_CACHE.tool_constraint(model_name, tokenizer, eos_token_id, regex)
}

#[derive(Default)]
struct ConstraintCache {
    vocab: Mutex<HashMap<String, Arc<Vocabulary>>>,
    index: Mutex<HashMap<u64, Arc<Index>>>,
}

impl ConstraintCache {
    fn new() -> Self {
        Self::default()
    }

    fn tool_constraint(
        &self,
        model_name: &str,
        tokenizer: &tokenizers::Tokenizer,
        eos_token_id: u32,
        regex: &str,
    ) -> Result<ConstrainedGenerator, EngineError> {
        let key = cache_key(model_name, regex);
        if let Some(index) = lock(&self.index).get(&key) {
            return Ok(ConstrainedGenerator::from_shared(Arc::clone(index)));
        }
        let vocab = self.vocab_for(model_name, tokenizer, eos_token_id)?;
        let index = Arc::new(
            Index::new(regex, vocab.as_ref())
                .map_err(|e| EngineError::Generation(format!("Failed to build FSM index: {e}")))?,
        );
        lock(&self.index).insert(key, Arc::clone(&index));
        Ok(ConstrainedGenerator::from_shared(index))
    }

    fn vocab_for(
        &self,
        model_name: &str,
        tokenizer: &tokenizers::Tokenizer,
        eos_token_id: u32,
    ) -> Result<Arc<Vocabulary>, EngineError> {
        if let Some(v) = lock(&self.vocab).get(model_name) {
            return Ok(Arc::clone(v));
        }
        let v = Arc::new(build_vocabulary(tokenizer, eos_token_id)?);
        lock(&self.vocab).insert(model_name.to_owned(), Arc::clone(&v));
        Ok(v)
    }
}

/// Lock a cache mutex, recovering from poisoning so one panicked build cannot
/// permanently wedge the cache.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn cache_key(model_name: &str, regex: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model_name.hash(&mut h);
    regex.hash(&mut h);
    h.finish()
}

/// Escape regex metacharacters so a literal string matches itself.
fn escape_regex_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(
            c,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$'
        ) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use outlines_core::index::Index;
    use outlines_core::vocabulary::Vocabulary;
    use serde_json::json;

    /// A vocab covering printable ASCII as single-byte tokens, so `Index::new`
    /// has real transitions and exercises the assembled regex end to end.
    fn ascii_vocab() -> Vocabulary {
        let mut v = Vocabulary::new(0);
        let mut id = 1u32;
        for byte in 0x20u8..=0x7e {
            let _ = v.try_insert(vec![byte], id);
            id += 1;
        }
        v
    }

    fn tool(name: &str, parameters: serde_json::Value) -> serde_json::Value {
        json!({"type": "function", "function": {"name": name, "parameters": parameters}})
    }

    /// The assembled regex must be accepted by outlines-core's FSM compiler.
    fn compiles(regex: &str) -> bool {
        Index::new(regex, &ascii_vocab()).is_ok()
    }

    #[test]
    fn envelope_and_keys_present() {
        let r = tool_call_regex(&[tool(
            "write_file",
            json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
        )]);
        assert!(r.starts_with("<tool_call>"), "regex: {r}");
        assert!(r.ends_with("</tool_call>"), "regex: {r}");
        assert!(r.contains("\"name\""));
        assert!(r.contains("\"arguments\""));
        assert!(r.contains("write_file"));
        // name must precede arguments (matches Qwen emission order).
        assert!(r.find("write_file").unwrap() < r.find("\"arguments\"").unwrap());
    }

    #[test]
    fn compiles_simple_schema() {
        let r = tool_call_regex(&[tool(
            "write_file",
            json!({"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
        )]);
        assert!(compiles(&r), "did not compile: {r}");
    }

    #[test]
    fn compiles_enum_schema() {
        let r = tool_call_regex(&[tool(
            "spawn",
            json!({"type": "object", "properties": {"action": {"type": "string", "enum": ["spawn", "list", "check"]}, "task": {"type": "string"}}}),
        )]);
        assert!(compiles(&r), "did not compile: {r}");
    }

    #[test]
    fn compiles_nested_array_schema() {
        let r = tool_call_regex(&[tool(
            "pipeline",
            json!({
                "type": "object",
                "properties": {
                    "steps": {"type": "array", "items": {"type": "object",
                        "properties": {"task": {"type": "string"}, "tools": {"type": "array", "items": {"type": "string"}}},
                        "required": ["task"]}}
                },
                "required": ["steps"]
            }),
        )]);
        assert!(compiles(&r), "did not compile: {r}");
    }

    #[test]
    fn compiles_union_of_tools() {
        let r = tool_call_regex(&[
            tool("read_file", json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})),
            tool("write_file", json!({"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]})),
        ]);
        assert!(r.contains('|'), "union should contain alternation: {r}");
        assert!(r.contains("read_file") && r.contains("write_file"));
        assert!(compiles(&r), "did not compile: {r}");
    }

    #[test]
    fn missing_parameters_falls_back_and_compiles() {
        // A tool with no `parameters` key still yields a compilable regex.
        let r = tool_call_regex(&[json!({"type": "function", "function": {"name": "ping"}})]);
        assert!(r.contains("ping"));
        assert!(compiles(&r), "did not compile: {r}");
    }

    #[test]
    fn no_tools_yields_guarded_pattern() {
        let r = tool_call_regex(&[]);
        assert!(r.contains("<tool_call>"));
    }
}
