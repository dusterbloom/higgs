//! Parse tool calls from model-generated text.
//!
//! Qwen models wrap tool calls in `<tool_call>…</tool_call>` tags, but the
//! payload *inside* the tags comes in two shapes depending on the model
//! generation:
//!
//! Legacy JSON (Qwen2.5 / Qwen3):
//! ```text
//! <tool_call>
//! {"name": "function_name", "arguments": {"arg1": "value1"}}
//! </tool_call>
//! ```
//!
//! XML function/parameter (Qwen3.5 / Qwen3.6 — what their
//! `chat_template.jinja` instructs the model to emit):
//! ```text
//! <tool_call>
//! <function=function_name>
//! <parameter=arg1>
//! value1
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! This module extracts structured tool calls from either shape. The XML form
//! emits every value as a raw string, so values are coerced to JSON types
//! using the request's declared tool schema ([`ToolSchema`]) when available,
//! falling back to best-effort parsing otherwise.

/// A parsed tool call extracted from model output.
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of parsing model output for tool calls.
#[derive(Debug, Clone)]
pub struct ToolParseResult {
    /// Text content before/outside any tool calls.
    pub text: String,
    /// Extracted tool calls (empty if none found).
    pub tool_calls: Vec<ParsedToolCall>,
}

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

const LFM2_TOOL_CALL_OPEN: &str = "<|tool_call_start|>";
const LFM2_TOOL_CALL_CLOSE: &str = "<|tool_call_end|>";

/// Hard cap on bytes buffered while inside an unclosed `<tool_call>`.
///
/// Without a cap, a model that emits `<tool_call>` and never closes the tag
/// would grow `buffer` until OOM — flagged CRITICAL on the closed upstream
/// PR #63. On overflow the tracker abandons the parse, emits `<tool_call>`
/// plus the buffered bytes as visible content (preserving the "never
/// silently drop tokens" invariant), and resets so subsequent well-formed
/// tool calls in the same stream still parse.
const MAX_INSIDE_TOOL_CALL_BYTES: usize = 1024 * 1024;

/// Parse model output text for Qwen-format tool calls.
///
/// `schema` carries the request's declared tool parameter types so XML-format
/// values can be coerced; pass `None` for best-effort coercion.
///
/// Returns the non-tool-call text and any extracted tool calls.
pub fn parse_tool_calls(text: &str, schema: Option<&ToolSchema>) -> ToolParseResult {
    // LFM2 / Macaw format: `<|tool_call_start|>[func(args), ...]<|tool_call_end|>`.
    // Run LFM2 extraction alongside the existing pipeline rather than
    // short-circuiting, so mixed-format output (e.g. `<tool_call>` blocks
    // from previous turns) still parses.
    let mut lfm2_result: Option<ToolParseResult> = None;
    if text.contains(LFM2_TOOL_CALL_OPEN) {
        lfm2_result = Some(parse_lfm2_tool_calls(text, schema));
    }

    // MiniCPM5 emits bare `<function name=…>…</function>` with no `<tool_call>`
    // wrapper. When there's no wrapper but a function opener is present, take
    // that path; otherwise fall through to the `<tool_call>` scanner (which
    // covers both the JSON and Qwen `<function=` XML inner forms).
    if !text.contains(TOOL_CALL_OPEN)
        && !text.contains(LFM2_TOOL_CALL_OPEN)
        && text.contains(MINICPM_FUNCTION_OPEN)
    {
        return parse_minicpm_tool_calls(text, schema);
    }

    let mut result_text = String::new();
    let mut tool_calls = Vec::new();
    let mut remaining = text;

    loop {
        if let Some(start_pos) = remaining.find(TOOL_CALL_OPEN) {
            result_text.push_str(remaining.get(..start_pos).unwrap_or_default());

            let after_open = remaining
                .get(start_pos + TOOL_CALL_OPEN.len()..)
                .unwrap_or_default();

            if let Some(end_pos) = after_open.find(TOOL_CALL_CLOSE) {
                let raw_block = after_open.get(..end_pos).unwrap_or_default();
                let call_content = raw_block.trim();

                if let Some(parsed) = parse_tool_call_block(call_content, schema) {
                    tool_calls.push(parsed);
                } else {
                    result_text.push_str(TOOL_CALL_OPEN);
                    result_text.push_str(raw_block);
                    result_text.push_str(TOOL_CALL_CLOSE);
                }

                remaining = after_open
                    .get(end_pos + TOOL_CALL_CLOSE.len()..)
                    .unwrap_or_default();
            } else {
                result_text.push_str(remaining.get(start_pos..).unwrap_or_default());
                break;
            }
        } else {
            result_text.push_str(remaining);
            break;
        }
    }

    // Merge LFM2 results if we have both Qwen and LFM2 calls.
    if let Some(lfm2) = lfm2_result {
        // If the Qwen scan found nothing useful, just use the LFM2 result.
        if tool_calls.is_empty() {
            return lfm2;
        }
        // Both formats produced calls — keep the Qwen-format text and
        // append LFM2 calls (text from LFM2 blocks was preserved verbatim
        // inside the LFM2 parse, so both sets of tool calls are correct).
        tool_calls.extend(lfm2.tool_calls);
    }

    let mut result = ToolParseResult {
        text: result_text.trim().to_owned(),
        tool_calls,
    };

    // Fallback: bare `[func(args), ...]` at end of text for LFM models
    // that emit tool calls without `<|tool_call_start|>` / `<|tool_call_end|>`.
    // Gated behind LFM-specific markers to avoid false positives.
    if result.tool_calls.is_empty() {
        if let Some((prefix, bare_calls)) = extract_bare_lfm2_tool_calls(&result.text, schema) {
            result.text = prefix.trim().to_owned();
            result.tool_calls = bare_calls;
        }
    }

    result
}

/// Try to parse a single tool call JSON block.
fn try_parse_tool_call(content: &str) -> Option<ParsedToolCall> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    let obj = value.as_object()?;

    let name = obj.get("name").and_then(|v| v.as_str())?.to_owned();

    let arguments = obj
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

    Some(ParsedToolCall { name, arguments })
}

const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

// MiniCPM5-style tool calls: `<function name="NAME"><param name="KEY">VALUE</param></function>`
// with no `<tool_call>` wrapper and optional `<![CDATA[…]]>`-wrapped values.
// `FUNCTION_CLOSE` (`</function>`) is shared with the Qwen XML form above.
const MINICPM_FUNCTION_OPEN: &str = "<function ";
const MINICPM_PARAM_OPEN: &str = "<param name=\"";
const MINICPM_PARAM_CLOSE: &str = "</param>";
const NAME_ATTR: &str = "name=\"";
const CDATA_OPEN: &str = "<![CDATA[";
const CDATA_CLOSE: &str = "]]>";

/// Declared JSON-schema type for a single tool parameter, used to coerce the
/// raw string values that the Qwen XML tool-call format emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamType {
    Str,
    Integer,
    Number,
    Boolean,
    Object,
    Array,
}

impl ParamType {
    fn from_schema_str(s: &str) -> Option<Self> {
        match s {
            "string" => Some(Self::Str),
            "integer" => Some(Self::Integer),
            "number" => Some(Self::Number),
            "boolean" => Some(Self::Boolean),
            "object" => Some(Self::Object),
            "array" => Some(Self::Array),
            _ => None,
        }
    }
}

/// Per-request tool parameter types, keyed by `function name → parameter
/// name → declared type`.
///
/// Built from the `OpenAI` `tools` array so the XML tool-call parser can
/// coerce raw string parameter values to the JSON types the client declared.
pub struct ToolSchema {
    params: std::collections::HashMap<String, std::collections::HashMap<String, ParamType>>,
}

impl ToolSchema {
    /// Build a [`ToolSchema`] from the request's `OpenAI` tool definitions.
    ///
    /// Each tool is either `{"type":"function","function":{...}}` or a bare
    /// function object. Returns `None` when no function declares a typed
    /// `parameters.properties` map — callers then use best-effort coercion.
    #[must_use]
    pub fn from_tools(tools: Option<&[serde_json::Value]>) -> Option<Self> {
        let tool_list = tools?;
        let mut params: std::collections::HashMap<
            String,
            std::collections::HashMap<String, ParamType>,
        > = std::collections::HashMap::new();

        for tool in tool_list {
            let function = tool.get("function").unwrap_or(tool);
            let Some(name) = function.get("name").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let Some(properties) = function
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(serde_json::Value::as_object)
            else {
                continue;
            };

            let param_types: std::collections::HashMap<String, ParamType> = properties
                .iter()
                .filter_map(|(param, spec)| {
                    let ty = spec
                        .get("type")
                        .and_then(serde_json::Value::as_str)
                        .and_then(ParamType::from_schema_str)?;
                    Some((param.clone(), ty))
                })
                .collect();

            if !param_types.is_empty() {
                params.insert(name.to_owned(), param_types);
            }
        }

        if params.is_empty() {
            return None;
        }
        Some(Self { params })
    }

    fn param_type(&self, function: &str, param: &str) -> Option<ParamType> {
        self.params.get(function)?.get(param).copied()
    }

    /// Return the parameter name at `index` for `function`, respecting
    /// the insertion order of the schema's `properties` map.
    fn param_at(&self, function: &str, index: usize) -> Option<&str> {
        self.params.get(function)?.keys().nth(index).map(String::as_str)
    }
}

/// Coerce a raw XML parameter string into a JSON value using its declared
/// schema type, falling back to best-effort JSON parsing when the type is
/// unknown or absent.
fn coerce_param_value(raw: &str, declared: Option<ParamType>) -> serde_json::Value {
    use serde_json::Value;
    let as_string = || Value::String(raw.to_owned());
    let parsed_if = |pred: fn(&Value) -> bool| {
        serde_json::from_str::<Value>(raw)
            .ok()
            .filter(pred)
            .unwrap_or_else(|| Value::String(raw.to_owned()))
    };
    match declared {
        Some(ParamType::Str) => as_string(),
        // `integer` must reject fractional values — `is_number` accepts floats.
        Some(ParamType::Integer) => parsed_if(|v| v.is_i64() || v.is_u64()),
        Some(ParamType::Number) => parsed_if(Value::is_number),
        Some(ParamType::Boolean) => match raw.trim() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => as_string(),
        },
        Some(ParamType::Object) => parsed_if(Value::is_object),
        Some(ParamType::Array) => parsed_if(Value::is_array),
        // No schema for this parameter: parse if it's valid JSON (so `42`
        // becomes a number), otherwise keep the raw string (so `London`
        // stays a string).
        None => serde_json::from_str::<Value>(raw).unwrap_or_else(|_| as_string()),
    }
}

/// Strip a single leading and trailing newline — the wrapping the template
/// adds around `<parameter>` values — preserving any intentional inner or
/// edge whitespace.
fn strip_one_wrapping_newline(s: &str) -> &str {
    let without_lead = s
        .strip_prefix("\r\n")
        .or_else(|| s.strip_prefix('\n'))
        .unwrap_or(s);
    without_lead
        .strip_suffix("\r\n")
        .or_else(|| without_lead.strip_suffix('\n'))
        .unwrap_or(without_lead)
}

/// Parse the Qwen XML tool-call body (the text between `<tool_call>` and
/// `</tool_call>`): a single `<function=NAME>…</function>` block containing
/// zero or more `<parameter=KEY>…</parameter>` entries.
///
/// Returns `None` when no well-formed `<function=…>` opener is present so the
/// caller can fall back to JSON parsing / verbatim preservation. The template
/// never nests more than one function per `<tool_call>`, so only the first is
/// parsed.
fn parse_xml_tool_call(content: &str, schema: Option<&ToolSchema>) -> Option<ParsedToolCall> {
    let open = content.find(FUNCTION_OPEN)?;
    let after_open = content.get(open + FUNCTION_OPEN.len()..)?;
    let name_end = after_open.find('>')?;
    let name = after_open.get(..name_end)?.trim().to_owned();
    if name.is_empty() {
        return None;
    }

    // Body between the `>` of `<function=NAME>` and the matching
    // `</function>` (or end of content if the closer is absent).
    let body_all = after_open.get(name_end + 1..).unwrap_or_default();
    let body = body_all
        .find(FUNCTION_CLOSE)
        .and_then(|i| body_all.get(..i))
        .unwrap_or(body_all);

    let mut map = serde_json::Map::new();
    let mut rest = body;
    while let Some(p_open) = rest.find(PARAM_OPEN) {
        let after_p = rest.get(p_open + PARAM_OPEN.len()..).unwrap_or_default();
        let Some(key_end) = after_p.find('>') else {
            break;
        };
        let key = after_p.get(..key_end).unwrap_or_default().trim().to_owned();
        let value_region = after_p.get(key_end + 1..).unwrap_or_default();
        let (raw_value, consumed) = value_region.find(PARAM_CLOSE).map_or_else(
            || (value_region, value_region.len()),
            |close| {
                (
                    value_region.get(..close).unwrap_or_default(),
                    close + PARAM_CLOSE.len(),
                )
            },
        );

        if !key.is_empty() {
            let value = strip_one_wrapping_newline(raw_value);
            let declared = schema.and_then(|s| s.param_type(&name, &key));
            map.insert(key, coerce_param_value(value, declared));
        }

        // Advance past this whole `<parameter=…>…</parameter>` entry.
        let advance = p_open + PARAM_OPEN.len() + key_end + 1 + consumed;
        rest = rest.get(advance..).unwrap_or_default();
    }

    Some(ParsedToolCall {
        name,
        arguments: serde_json::Value::Object(map),
    })
}

/// Parse one `<tool_call>` block body, dispatching on shape: the Qwen XML
/// `<function=…>` form vs the legacy JSON-object form.
fn parse_tool_call_block(content: &str, schema: Option<&ToolSchema>) -> Option<ParsedToolCall> {
    if content.trim_start().starts_with(FUNCTION_OPEN) {
        parse_xml_tool_call(content, schema)
    } else {
        try_parse_tool_call(content)
    }
}

/// Byte offset of the `</function>` that closes a `MiniCPM` function block in
/// `s`, skipping any `<![CDATA[ … ]]>` spans whose content may itself contain
/// a literal `</function>`.
///
/// Returns `None` when the block is not yet terminated: either no closer has
/// arrived, or scanning is parked inside an unclosed CDATA span (the caller
/// should wait for more input).
fn minicpm_function_end(s: &str) -> Option<usize> {
    let mut i = 0;
    loop {
        let rest = s.get(i..)?;
        let next_close = rest.find(FUNCTION_CLOSE);
        // A CDATA span that opens before the next close tag must be skipped
        // whole, otherwise a `</function>` inside it would close early.
        if let Some(d) = rest.find(CDATA_OPEN) {
            if next_close.is_none_or(|c| d < c) {
                let after_open = d + CDATA_OPEN.len();
                let close = rest.get(after_open..)?.find(CDATA_CLOSE)?;
                i += after_open + close + CDATA_CLOSE.len();
                continue;
            }
        }
        return next_close.map(|c| i + c);
    }
}

/// Extract one `MiniCPM` `<param>` value from `vr` — the text immediately after
/// the param tag's `>`. Returns `(value, rest_after_</param>)`. A
/// `<![CDATA[…]]>` wrapper yields its verbatim content; otherwise the value is
/// the text up to `</param>`. Both returned slices borrow `vr`.
fn extract_param_value(vr: &str) -> (&str, &str) {
    if let Some(stripped) = vr.strip_prefix(CDATA_OPEN) {
        if let Some(close) = stripped.find(CDATA_CLOSE) {
            let value = stripped.get(..close).unwrap_or_default();
            let tail = stripped
                .get(close + CDATA_CLOSE.len()..)
                .unwrap_or_default();
            let after = tail
                .find(MINICPM_PARAM_CLOSE)
                .and_then(|i| tail.get(i + MINICPM_PARAM_CLOSE.len()..))
                .unwrap_or_default();
            return (value, after);
        }
        return (stripped, "");
    }
    vr.find(MINICPM_PARAM_CLOSE).map_or((vr, ""), |i| {
        (
            vr.get(..i).unwrap_or_default(),
            vr.get(i + MINICPM_PARAM_CLOSE.len()..).unwrap_or_default(),
        )
    })
}

/// Parse a single `MiniCPM` function block (`<function name="…">…` up to, but
/// not including, the closing `</function>`).
///
/// Returns `None` when no `name="…"` attribute is present so the caller can
/// preserve the text verbatim.
fn parse_minicpm_function(block: &str, schema: Option<&ToolSchema>) -> Option<ParsedToolCall> {
    // Read `name="…"` only from the opening `<function …>` tag (before its
    // closing `>`). Scanning the whole block would let a malformed payload
    // like `<function><param name="x">…` be parsed as a tool call named `x`
    // instead of being preserved verbatim.
    let tag_close = block.find('>')?;
    let open_tag = block.get(..tag_close)?;
    let name_attr = open_tag.find(NAME_ATTR)?;
    let after_attr = open_tag.get(name_attr + NAME_ATTR.len()..)?;
    let name_end = after_attr.find('"')?;
    let name = after_attr.get(..name_end)?.to_owned();
    if name.is_empty() {
        return None;
    }
    // Params start after the `>` that closes the `<function …>` open tag.
    let mut rest = block.get(tag_close + 1..).unwrap_or_default();

    let mut map = serde_json::Map::new();
    while let Some(p_open) = rest.find(MINICPM_PARAM_OPEN) {
        let after_p = rest
            .get(p_open + MINICPM_PARAM_OPEN.len()..)
            .unwrap_or_default();
        let Some(key_end) = after_p.find('"') else {
            break;
        };
        let key = after_p.get(..key_end).unwrap_or_default().to_owned();
        let after_key = after_p.get(key_end + 1..).unwrap_or_default();
        let Some(gt) = after_key.find('>') else {
            break;
        };
        let value_region = after_key.get(gt + 1..).unwrap_or_default();
        let (raw_value, after) = extract_param_value(value_region);
        if !key.is_empty() {
            let declared = schema.and_then(|s| s.param_type(&name, &key));
            map.insert(key, coerce_param_value(raw_value, declared));
        }
        rest = after;
    }

    Some(ParsedToolCall {
        name,
        arguments: serde_json::Value::Object(map),
    })
}

// ---------------------------------------------------------------------------
// LFM2 / Macaw tool-call parser
// ---------------------------------------------------------------------------

/// Parse a single Python-style function call of the form
/// `func_name(key1='val1', key2=42)` into a [`ParsedToolCall`].
///
/// Returns `None` if the content is not a well-formed function call.
/// When `schema` is provided, values are coerced to declared types.
fn parse_python_style_call(
    content: &str,
    schema: Option<&ToolSchema>,
) -> Option<ParsedToolCall> {
    let content = content.trim();
    // Must be `func_name(...)` — find the opening paren.
    let open_paren = content.find('(')?;

    let name = content[..open_paren].trim().to_owned();
    if name.is_empty() {
        return None;
    }

    // Everything between `(` and the last `)` is the argument list.
    // Use a depth-aware scan so a `)` inside a quoted string or nested
    // call (e.g. `func(msg='good)bye')`) does not fool the parser.
    let after_open = &content[open_paren + 1..];
    let close_paren = find_closing_paren(after_open)?;
    let args_str = after_open[..close_paren].trim();

    let mut map = serde_json::Map::new();
    let mut positional_idx: usize = 0;

    if !args_str.is_empty() {
        // Split by top-level commas, respecting string quoting AND
        // bracket/paren/brace depth (so nested calls like `f(a=g(1,2))`
        // and lists like `f(paths=['a','b'])` do not split mid-argument).
        let arg_parts = split_top_level(args_str, ',');
        for part in arg_parts {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            // Try `key=value` first; positional args fall through.
            if let Some(eq_pos) = find_top_level_char(part, '=') {
                let key = part[..eq_pos].trim().to_owned();
                let value_str = part[eq_pos + 1..].trim();
                if !key.is_empty() {
                    let declared = schema.and_then(|s| s.param_type(&name, &key));
                    map.insert(key, parse_python_literal(value_str, declared));
                    continue;
                }
            }
            // Positional argument — map by index if schema provides the name,
            // otherwise use `argN` as a fallback key.
            if let Some(param_name) = schema.and_then(|s| s.param_at(&name, positional_idx)) {
                let declared = schema.and_then(|s| s.param_type(&name, param_name));
                map.insert(param_name.to_owned(), parse_python_literal(part, declared));
            } else {
                let key = format!("arg{positional_idx}");
                map.insert(key, parse_python_literal(part, None));
            }
            positional_idx += 1;
        }
    }

    Some(ParsedToolCall {
        name,
        arguments: serde_json::Value::Object(map),
    })
}

/// Split `s` by top-level commas, respecting:
/// - Paired quote delimiters (`'…'`, `"…"`) with backslash escaping
/// - Bracket/paren/brace depth: `(…)`, `[…]`, `{…}`
///
/// Commas inside a quoted string or nested bracket are never splitters.
fn split_top_level<'a>(s: &'a str, delim: char) -> Vec<&'a str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut i = 0;
    let mut depth: i32 = 0; // bracket/paren/brace nesting
    let bytes = s.as_bytes();
    while i < bytes.len() {
        match bytes[i] {
            b'\'' | b'"' => {
                let quote = bytes[i];
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == quote {
                        break;
                    }
                    i += 1;
                }
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = (depth - 1).max(0),
            c if c == delim as u8 && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    parts.push(&s[start..]);
    parts
}

/// Find the first occurrence of `c` at depth 0 (outside strings, brackets,
/// parens, and braces). Returns `None` when `c` only appears nested.
fn find_top_level_char(s: &str, c: char) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut depth: i32 = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' | b'"' => {
                let quote = bytes[i];
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == quote {
                        break;
                    }
                    i += 1;
                }
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = (depth - 1).max(0),
            ch if ch == c as u8 && depth == 0 => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

/// Unescape a Python-style quoted-string body.
///
/// Handles: `\\` → `\`, `\'` → `'`, `\"` → `"`, `\n` → newline,
/// `\r` → CR, `\t` → tab. Unknown escapes keep the backslash.
/// Find the matching closing `)` for an opening `(` at position 0,
/// respecting quoted strings and nested bracket/paren/brace depth.
fn find_closing_paren(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth: i32 = 1; // we start just after the opening `(`, so depth=1
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' | b'"' => {
                let quote = bytes[i];
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == quote {
                        break;
                    }
                    i += 1;
                }
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            b']' | b'}' => depth = (depth - 1).max(0),
            _ => {}
        }
        i += 1;
    }
    None
}

/// Unescape a Python-style quoted-string body.
///
/// Handles: `\\` → `\`, `\'` → `'`, `\"` → `"`, `\n` → newline,
/// `\r` → CR, `\t` → tab. Unknown escapes keep the backslash.
fn unescape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'\\' => { out.push('\\'); i += 2; }
                b'\'' => { out.push('\''); i += 2; }
                b'"'  => { out.push('"');  i += 2; }
                b'n'  => { out.push('\n'); i += 2; }
                b'r'  => { out.push('\r'); i += 2; }
                b't'  => { out.push('\t'); i += 2; }
                _     => { out.push('\\'); i += 1; }
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// Parse a Python-style literal value into a JSON value.
///
/// Handles:
/// - `True` / `False` → boolean
/// - `None` → null
/// - Integer literals (`42`, `-5`)
/// - Float literals (`3.14`, `-2.5`, `1e10`)
/// - Single-quoted strings (`'hello'`, `'it\\'s'`) — escapes stripped
/// - Double-quoted strings (`"world"`) — escapes stripped
/// - Lists (`['a', 2, True]`) — elements parsed recursively
/// - Dicts (`{'key': 'val'}`) — values parsed recursively
/// - Falls back to string for anything unrecognised.
///
/// When `declared` is provided, values are coerced to match (e.g. a
/// quoted `'5'` for an integer parameter becomes `5`).
fn parse_python_literal(s: &str, declared: Option<ParamType>) -> serde_json::Value {
    let s = s.trim();

    // Boolean
    if s == "True" {
        return serde_json::Value::Bool(true);
    }
    if s == "False" {
        return serde_json::Value::Bool(false);
    }
    // None / null
    if s == "None" {
        return serde_json::Value::Null;
    }

    // Quoted strings — unescape the body.
    if (s.starts_with('\'') && s.ends_with('\'')) || (s.starts_with('"') && s.ends_with('"')) {
        let inner = &s[1..s.len() - 1];
        let unescaped = unescape_string(inner);
        // Coerce to declared type if requested (e.g. `'5'` for integer → 5).
        if let Some(dt) = declared {
            return coerce_param_value(&unescaped, Some(dt));
        }
        return serde_json::Value::String(unescaped);
    }

    // List literal: `['a', 2, True]`
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let elements: Vec<serde_json::Value> = if inner.trim().is_empty() {
            Vec::new()
        } else {
            split_top_level(inner, ',')
                .into_iter()
                .map(|e| parse_python_literal(e.trim(), None))
                .collect()
        };
        return serde_json::Value::Array(elements);
    }

    // Dict literal: `{'key': 'val', 'n': 1}`
    if s.starts_with('{') && s.ends_with('}') {
        let inner = &s[1..s.len() - 1];
        let mut map = serde_json::Map::new();
        if !inner.trim().is_empty() {
            for pair in split_top_level(inner, ',') {
                let pair = pair.trim();
                if let Some(colon) = find_top_level_char(pair, ':') {
                    let key_raw = pair[..colon].trim();
                    let val_raw = pair[colon + 1..].trim();
                    // Key is always a string (strip quotes if present).
                    let key = if (key_raw.starts_with('\'') && key_raw.ends_with('\''))
                        || (key_raw.starts_with('"') && key_raw.ends_with('"'))
                    {
                        unescape_string(&key_raw[1..key_raw.len() - 1])
                    } else {
                        key_raw.to_owned()
                    };
                    map.insert(key, parse_python_literal(val_raw, None));
                }
            }
        }
        return serde_json::Value::Object(map);
    }

    // Try integer (accept negative too)
    if let Ok(n) = s.parse::<i64>() {
        return serde_json::Value::Number(serde_json::Number::from(n));
    }

    // Try float
    if let Ok(n) = s.parse::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(n) {
            return serde_json::Value::Number(num);
        }
    }

    // Fallback: treat as raw string.
    serde_json::Value::String(s.to_owned())
}

/// Try to find and extract bare `[func(args), ...]` tool calls from the end
/// of text. LFM models sometimes emit tool calls without the
/// `<|tool_call_start|>` / `<|tool_call_end|>` delimiters, appending
/// `[func(args)]` directly after the visible text.
///
/// Gated: only activates when the text contains LFM-specific markers
/// (`<|im_start|>`, `<|tool_list_start|>`, `<|tool_response_start|>`, or
/// `<|tool_call_start|>`) so non-LFM models do not get false positives
/// from bracket-enclosed expressions in their output.
///
/// Returns `(prefix_text, tool_calls)` where `prefix_text` is everything
/// before the bare bracket list, or `None` if no valid bracket-enclosed
/// function calls were found at the end.
fn extract_bare_lfm2_tool_calls<'a>(
    text: &'a str,
    schema: Option<&ToolSchema>,
) -> Option<(&'a str, Vec<ParsedToolCall>)> {
    // Gate: require at least one LFM-specific marker in the text.
    if !text.contains("<|im_start|>")
        && !text.contains("<|tool_list_start|>")
        && !text.contains("<|tool_response_start|>")
        && !text.contains(LFM2_TOOL_CALL_OPEN)
    {
        return None;
    }

    let trimmed = text.trim_end();
    // Must end with `]`
    if !trimmed.ends_with(']') {
        return None;
    }

    // Find the matching `[` — scan backwards from the end, but only
    // within the last "paragraph" (stop at a newline). This avoids the
    // backward-quote ambiguity and keeps the scan local to the final
    // segment where tool calls typically appear.
    let last_newline = trimmed.rfind('\n').unwrap_or(0);
    let tail = &trimmed[last_newline..];
    let tail_start_in_trimmed = last_newline;

        // Now scan forward inside `tail` to find a top-level `[...]` at the end.
    let mut bracket_start: Option<usize> = None;
    let mut depth: i32 = 0;
    let tail_bytes = tail.as_bytes();
    for (i, &b) in tail_bytes.iter().enumerate() {
        match b {
            b'\'' | b'"' => {
                // Peek ahead for a closing quote so brackets inside
                // quoted strings are not mis-counted.  The scan index
                // `i` is not advanced — the for-loop still visits every
                // byte so lone quotes (e.g. apostrophes) don't cause
                // the scanner to skip over real brackets.
                let quote = b;
                let mut j = i + 1;
                while j < tail_bytes.len() {
                    if tail_bytes[j] == b'\\' {
                        j += 2;
                        continue;
                    }
                    if tail_bytes[j] == quote {
                        break;
                    }
                    j += 1;
                }
            }
            b'[' => {
                if depth == 0 {
                    bracket_start = Some(i);
                }
                depth += 1;
            }
            b']' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            _ => {}
        }
    }
    let start_in_tail = bracket_start?;
    let open = tail_start_in_trimmed + start_in_tail;

    // Verify the bracket list extends to the end (modulo trailing whitespace).
    let after_open = &trimmed[open..];
    if !after_open.ends_with(']') {
        return None;
    }

    let bracket_content = &trimmed[open + 1..trimmed.len() - 1];

    // Quick heuristic: must contain at least one `(` and `)` to look like
    // function calls. Avoid false positives on bare lists like `[1, 2, 3]`.
    if !bracket_content.contains('(') || !bracket_content.contains(')') {
        return None;
    }

    // Try to parse as function calls.
    let calls = parse_lfm2_block(&trimmed[open..], schema)?;

    if calls.is_empty() {
        return None;
    }

    let prefix = trimmed[..open].trim_end();
    Some((prefix, calls))
}

/// Parse the content of an LFM2 block — the text between
/// `<|tool_call_start|>` and `<|tool_call_end|>` — which must be a
/// bracket-enclosed list of Python-style function calls:
/// `[func1(arg='val'), func2()]`.
fn parse_lfm2_block(
    content: &str,
    schema: Option<&ToolSchema>,
) -> Option<Vec<ParsedToolCall>> {
    let content = content.trim();

    // Must be a bracket-enclosed list.
    if !content.starts_with('[') || !content.ends_with(']') {
        return None;
    }

    let inner = &content[1..content.len() - 1];
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }

    // Split the top-level comma-separated list to find individual calls.
    // Each call is `func_name(args)` — we split on `),` boundaries at depth 0.
    let mut calls = Vec::new();
    let mut depth: i32 = 0;
    let mut start = 0;
    let bytes = inner.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        match bytes[i] {
            b'\'' | b'"' => {
                // Skip quoted string
                let quote = bytes[i];
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == quote {
                        break;
                    }
                    i += 1;
                }
            }
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                // At depth 0, the next comma (or end) marks a call boundary.
                if depth == 0 {
                    // Find the next comma at depth 0
                    let mut j = i + 1;
                    while j < bytes.len() {
                        if bytes[j] == b'\'' || bytes[j] == b'"' {
                            let q = bytes[j];
                            j += 1;
                            while j < bytes.len() {
                                if bytes[j] == b'\\' {
                                    j += 2;
                                    continue;
                                }
                                if bytes[j] == q {
                                    break;
                                }
                                j += 1;
                            }
                        } else if bytes[j] == b',' {
                            break;
                        }
                        j += 1;
                    }
                    let call_str = &inner[start..j].trim();
                    if !call_str.is_empty() {
                        if let Some(tc) = parse_python_style_call(call_str, schema) {
                            calls.push(tc);
                        }
                    }
                    start = j + 1; // skip past the comma
                    i = j;
                }
            }
            _ => {}
        }
        i += 1;
    }

    // Last call (or only call if no commas)
    if start < inner.len() {
        let call_str = inner[start..].trim();
        if !call_str.is_empty() {
            if let Some(tc) = parse_python_style_call(call_str, schema) {
                calls.push(tc);
            }
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Scan text for one or more `<|tool_call_start|>…<|tool_call_end|>` LFM2
/// blocks. Text outside the blocks is preserved as visible content;
/// unparseable or unterminated blocks are preserved verbatim.
fn parse_lfm2_tool_calls(text: &str, schema: Option<&ToolSchema>) -> ToolParseResult {
    let mut result_text = String::new();
    let mut tool_calls = Vec::new();
    let mut remaining = text;

    loop {
        let Some(start) = remaining.find(LFM2_TOOL_CALL_OPEN) else {
            result_text.push_str(remaining);
            break;
        };
        result_text.push_str(remaining.get(..start).unwrap_or_default());
        let after_open = remaining
            .get(start + LFM2_TOOL_CALL_OPEN.len()..)
            .unwrap_or_default();

        if let Some(end_pos) = after_open.find(LFM2_TOOL_CALL_CLOSE) {
            let raw_block = after_open.get(..end_pos).unwrap_or_default();
            let call_content = raw_block.trim();

            if let Some(parsed) = parse_lfm2_block(call_content, schema) {
                tool_calls.extend(parsed);
            } else {
                result_text.push_str(LFM2_TOOL_CALL_OPEN);
                result_text.push_str(raw_block);
                result_text.push_str(LFM2_TOOL_CALL_CLOSE);
            }

            remaining = after_open
                .get(end_pos + LFM2_TOOL_CALL_CLOSE.len()..)
                .unwrap_or_default();
        } else {
            result_text.push_str(remaining.get(start..).unwrap_or_default());
            break;
        }
    }

    ToolParseResult {
        text: result_text.trim().to_owned(),
        tool_calls,
    }
}

/// Scan text for one or more bare `MiniCPM` `<function …>…</function>` blocks
/// (no `<tool_call>` wrapper). Text outside the blocks is preserved as visible
/// content; unparseable or unterminated blocks are preserved verbatim.
fn parse_minicpm_tool_calls(text: &str, schema: Option<&ToolSchema>) -> ToolParseResult {
    let mut result_text = String::new();
    let mut tool_calls = Vec::new();
    let mut remaining = text;

    loop {
        let Some(start) = remaining.find(MINICPM_FUNCTION_OPEN) else {
            result_text.push_str(remaining);
            break;
        };
        result_text.push_str(remaining.get(..start).unwrap_or_default());
        let block_region = remaining.get(start..).unwrap_or_default();

        let Some(end) = minicpm_function_end(block_region) else {
            result_text.push_str(block_region);
            break;
        };

        let block = block_region.get(..end).unwrap_or_default();
        if let Some(parsed) = parse_minicpm_function(block, schema) {
            tool_calls.push(parsed);
        } else {
            result_text.push_str(block);
            result_text.push_str(FUNCTION_CLOSE);
        }
        remaining = block_region
            .get(end + FUNCTION_CLOSE.len()..)
            .unwrap_or_default();
    }

    ToolParseResult {
        text: result_text.trim().to_owned(),
        tool_calls,
    }
}

/// One chunk of streaming output from [`StreamingToolCallTracker::process`]
/// or [`StreamingToolCallTracker::flush`].
///
/// `visible` is the text that should be forwarded to the client as a normal
/// content delta. `new_tool_calls` are any tool calls that became complete
/// during this chunk — the route layer turns them into `ToolCallDelta` SSE
/// events.
#[derive(Debug, Default)]
pub struct StreamingToolOutput {
    /// Text to forward to the client as a normal content delta.
    pub visible: String,
    /// Tool calls that became complete during this chunk; the route layer
    /// emits each as a `tool_calls` SSE delta.
    pub new_tool_calls: Vec<ParsedToolCall>,
}

/// Longest opener token. In the scanning state the tracker keeps this many
/// bytes at the buffer tail so a `<tool_call>` or `<function ` opener split
/// across a chunk boundary is still detected next chunk.
const MAX_OPENER_LEN: usize = {
    let mut max = TOOL_CALL_OPEN.len();
    if MINICPM_FUNCTION_OPEN.len() > max {
        max = MINICPM_FUNCTION_OPEN.len();
    }
    if LFM2_TOOL_CALL_OPEN.len() > max {
        max = LFM2_TOOL_CALL_OPEN.len();
    }
    max
};

/// Which kind of tool-call block the tracker is currently inside.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Inside {
    /// Scanning for the next opener.
    None,
    /// Inside a `<tool_call>…</tool_call>` block (JSON or Qwen `<function=` XML).
    ToolCall,
    /// Inside a bare `MiniCPM` `<function …>…</function>` block.
    Function,
    /// Inside a `<|tool_call_start|>…<|tool_call_end|>` LFM2 / Macaw block.
    Lfm2ToolCall,
}

/// State machine that buffers streaming text chunks and extracts tool-call
/// blocks on the fly — `<tool_call>…</tool_call>` (JSON or Qwen `<function=`
/// XML) and bare `MiniCPM` `<function …>…</function>`.
///
/// Designed to be cheap: when `active = false` (no tools in the request),
/// `process` is a single allocation per chunk and `flush` is a no-op.
///
/// When active, it retains a small tail so an opener can't straddle a chunk
/// boundary; once a complete block is buffered it is parsed and emitted as a
/// [`ParsedToolCall`]. Text before/after blocks streams out verbatim.
///
/// Invariants:
/// - **Never silently drops tokens.** Unclosed tags at `flush` are re-emitted
///   as visible content rather than discarded.
/// - **UTF-8 safe.** Tail-flushes walk back to the previous char boundary
///   so a partial multi-byte sequence is never split.
/// - **Pure passthrough when inactive.** Zero parsing cost on requests
///   that did not pass `tools` to the chat route.
pub struct StreamingToolCallTracker {
    buffer: String,
    inside: Inside,
    completed_count: usize,
    active: bool,
    schema: Option<ToolSchema>,
}

impl StreamingToolCallTracker {
    /// `schema` carries the request's declared tool parameter types so
    /// XML-format values can be coerced; pass `None` for best-effort.
    pub const fn new(active: bool, schema: Option<ToolSchema>) -> Self {
        Self {
            buffer: String::new(),
            inside: Inside::None,
            completed_count: 0,
            active,
            schema,
        }
    }

    pub const fn completed_count(&self) -> usize {
        self.completed_count
    }

    pub const fn has_tool_calls(&self) -> bool {
        self.completed_count > 0
    }

    /// In the scanning state, advance to the next opener — entering
    /// `ToolCall`/`Function` — or flush all-but-tail and signal "wait".
    /// Returns `true` to keep looping, `false` to break (need more input).
    fn scan_for_opener(&mut self, out: &mut StreamingToolOutput) -> bool {
        let tc = self.buffer.find(TOOL_CALL_OPEN);
        let fc = self.buffer.find(MINICPM_FUNCTION_OPEN);
        let lc = self.buffer.find(LFM2_TOOL_CALL_OPEN);
        // Enter whichever opener appears first.
        // Pick: (pos, kind) where kind: 0 = ToolCall, 1 = Function, 2 = Lfm2ToolCall
        let pick: Option<(usize, u8)> = {
            let mut best: Option<(usize, u8)> = None;
            if let Some(p) = tc {
                best = Some((p, 0));
            }
            if let Some(p) = fc {
                if best.is_none_or(|b| p < b.0) {
                    best = Some((p, 1));
                }
            }
            if let Some(p) = lc {
                if best.is_none_or(|b| p < b.0) {
                    best = Some((p, 2));
                }
            }
            best
        };
        let Some((pos, kind)) = pick else {
            // No opener yet — flush all but a tail large enough to hold a
            // split opener, walking back to a UTF-8 char boundary.
            if self.buffer.len() > MAX_OPENER_LEN {
                let target_len = self.buffer.len() - MAX_OPENER_LEN;
                let mut safe_len = target_len;
                while safe_len > 0 && !self.buffer.is_char_boundary(safe_len) {
                    safe_len -= 1;
                }
                out.visible
                    .push_str(self.buffer.get(..safe_len).unwrap_or_default());
                self.buffer = self.buffer.get(safe_len..).unwrap_or_default().to_owned();
            }
            return false;
        };
        out.visible
            .push_str(self.buffer.get(..pos).unwrap_or_default());
        match kind {
            0 => {
                // `<tool_call>` opener: strip it; inner body parsed at closer.
                self.buffer = self
                    .buffer
                    .get(pos + TOOL_CALL_OPEN.len()..)
                    .unwrap_or_default()
                    .to_owned();
                self.inside = Inside::ToolCall;
            }
            1 => {
                // `<function …` opener: keep it for the block parser.
                self.buffer = self.buffer.get(pos..).unwrap_or_default().to_owned();
                self.inside = Inside::Function;
            }
            2 => {
                // `<|tool_call_start|>` opener: strip it; inner body parsed at closer.
                self.buffer = self
                    .buffer
                    .get(pos + LFM2_TOOL_CALL_OPEN.len()..)
                    .unwrap_or_default()
                    .to_owned();
                self.inside = Inside::Lfm2ToolCall;
            }
            _ => unreachable!(),
        }
        true
    }

    /// Feed a chunk of streamed text. Returns visible text + any tool calls
    /// that became complete in this chunk.
    pub fn process(&mut self, text: &str) -> StreamingToolOutput {
        if !self.active {
            return StreamingToolOutput {
                visible: text.to_owned(),
                new_tool_calls: Vec::new(),
            };
        }

        self.buffer.push_str(text);
        let mut out = StreamingToolOutput::default();

        loop {
            match self.inside {
                Inside::ToolCall => {
                    // Seek `</tool_call>`; once seen, parse the inner block
                    // (JSON or Qwen `<function=` XML) and keep scanning.
                    if let Some(end) = self.buffer.find(TOOL_CALL_CLOSE) {
                        let raw_block = self.buffer.get(..end).unwrap_or_default();
                        let call_content = raw_block.trim();
                        if let Some(parsed) =
                            parse_tool_call_block(call_content, self.schema.as_ref())
                        {
                            out.new_tool_calls.push(parsed);
                            self.completed_count += 1;
                        } else {
                            // Unparseable inner — preserve verbatim so the
                            // client/operator sees what the model emitted.
                            out.visible.push_str(TOOL_CALL_OPEN);
                            out.visible.push_str(raw_block);
                            out.visible.push_str(TOOL_CALL_CLOSE);
                        }
                        self.buffer = self
                            .buffer
                            .get(end + TOOL_CALL_CLOSE.len()..)
                            .unwrap_or_default()
                            .to_owned();
                        self.inside = Inside::None;
                    } else if self.buffer.len() > MAX_INSIDE_TOOL_CALL_BYTES {
                        // Overflow guard: opener seen, closer never arrived.
                        let leftover = std::mem::take(&mut self.buffer);
                        out.visible.push_str(TOOL_CALL_OPEN);
                        out.visible.push_str(&leftover);
                        self.inside = Inside::None;
                        break;
                    } else {
                        break;
                    }
                }
                Inside::Function => {
                    // The `<function …` opener is kept in the buffer so the
                    // block parser can read the `name="…"` attribute. Seek a
                    // CDATA-aware `</function>`.
                    if let Some(end) = minicpm_function_end(&self.buffer) {
                        let block = self.buffer.get(..end).unwrap_or_default();
                        if let Some(parsed) = parse_minicpm_function(block, self.schema.as_ref()) {
                            out.new_tool_calls.push(parsed);
                            self.completed_count += 1;
                        } else {
                            out.visible.push_str(block);
                            out.visible.push_str(FUNCTION_CLOSE);
                        }
                        self.buffer = self
                            .buffer
                            .get(end + FUNCTION_CLOSE.len()..)
                            .unwrap_or_default()
                            .to_owned();
                        self.inside = Inside::None;
                    } else if self.buffer.len() > MAX_INSIDE_TOOL_CALL_BYTES {
                        // Overflow guard: `<function …` opened, never closed.
                        let leftover = std::mem::take(&mut self.buffer);
                        out.visible.push_str(&leftover);
                        self.inside = Inside::None;
                        break;
                    } else {
                        break;
                    }
                }
                Inside::Lfm2ToolCall => {
                    // Seek `<|tool_call_end|>`; once seen, parse the inner
                    // Python-style bracket list.
                    if let Some(end) = self.buffer.find(LFM2_TOOL_CALL_CLOSE) {
                        let raw_block = self.buffer.get(..end).unwrap_or_default();
                        let call_content = raw_block.trim();
                        if let Some(parsed) = parse_lfm2_block(call_content, self.schema.as_ref()) {
                            for tc in parsed {
                                out.new_tool_calls.push(tc);
                                self.completed_count += 1;
                            }
                        } else {
                            out.visible.push_str(LFM2_TOOL_CALL_OPEN);
                            out.visible.push_str(raw_block);
                            out.visible.push_str(LFM2_TOOL_CALL_CLOSE);
                        }
                        self.buffer = self
                            .buffer
                            .get(end + LFM2_TOOL_CALL_CLOSE.len()..)
                            .unwrap_or_default()
                            .to_owned();
                        self.inside = Inside::None;
                    } else if self.buffer.len() > MAX_INSIDE_TOOL_CALL_BYTES {
                        // Overflow guard: opener seen, closer never arrived.
                        let leftover = std::mem::take(&mut self.buffer);
                        out.visible.push_str(LFM2_TOOL_CALL_OPEN);
                        out.visible.push_str(&leftover);
                        self.inside = Inside::None;
                        break;
                    } else {
                        break;
                    }
                }
                Inside::None => {
                    if !self.scan_for_opener(&mut out) {
                        break;
                    }
                }
            }
        }

        out
    }

    /// Drain everything still buffered. Call this when the model stream
    /// ends. Any unclosed `<tool_call>` block is emitted as visible content
    /// (with its opener prepended) so no tokens silently vanish.
    pub fn flush(&mut self) -> StreamingToolOutput {
        let leftover = std::mem::take(&mut self.buffer);
        let inside = self.inside;
        self.inside = Inside::None;

        let visible = match inside {
            // The `<tool_call>` opener was stripped on entry, so re-prepend it.
            Inside::ToolCall => {
                let mut v = String::with_capacity(TOOL_CALL_OPEN.len() + leftover.len());
                v.push_str(TOOL_CALL_OPEN);
                v.push_str(&leftover);
                v
            }
            // The `<|tool_call_start|>` opener was stripped on entry, re-prepend.
            Inside::Lfm2ToolCall => {
                let mut v =
                    String::with_capacity(LFM2_TOOL_CALL_OPEN.len() + leftover.len());
                v.push_str(LFM2_TOOL_CALL_OPEN);
                v.push_str(&leftover);
                v
            }
            // `Function` keeps its `<function …` opener in the buffer, and
            // `None` is plain text — both emit the leftover verbatim.
            Inside::Function | Inside::None => leftover,
        };

        // Bare `[func(args)]` fallback (mirrors non-streaming path).
        let mut tool_calls = Vec::new();
        let visible = if !self.active {
            visible
        } else if let Some((raw_prefix, bare_calls)) =
            extract_bare_lfm2_tool_calls(&visible, self.schema.as_ref())
        {
            tool_calls = bare_calls;
            // During Lfm2ToolCall flush the opener was re-prepended;
            // strip it so the control token never reaches the user.
            raw_prefix
                .strip_prefix(LFM2_TOOL_CALL_OPEN)
                .unwrap_or(raw_prefix)
                .to_owned()
        } else {
            visible
        };

        StreamingToolOutput {
            visible,
            new_tool_calls: tool_calls,
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// Parse input and assert expected tool call count and optional text fragment.
    fn assert_parse(
        input: &str,
        expected_tools: usize,
        text_contains: Option<&str>,
    ) -> ToolParseResult {
        let result = parse_tool_calls(input, None);
        assert_eq!(
            result.tool_calls.len(),
            expected_tools,
            "expected {expected_tools} tool calls, got {}",
            result.tool_calls.len()
        );
        if let Some(fragment) = text_contains {
            assert!(
                result.text.contains(fragment),
                "expected text to contain {fragment:?}, got {:?}",
                result.text
            );
        }
        result
    }

    /// Assert the parsed result has no tool calls and preserves the raw tags in text.
    fn assert_raw_preserved(input: &str) {
        let result = assert_parse(input, 0, Some("<tool_call>"));
        assert!(result.text.contains("</tool_call>"));
    }

    /// Get the name of the first parsed tool call.
    fn first_tool_name(result: &ToolParseResult) -> &str {
        &result.tool_calls.first().unwrap().name
    }

    #[test]
    fn test_no_tool_calls() {
        let result = assert_parse(
            "Hello, how can I help you?",
            0,
            Some("Hello, how can I help you?"),
        );
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_single_tool_call() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "London"}}
</tool_call>"#;
        let result = assert_parse(input, 1, None);
        assert!(!result.text.contains("[ping()]"));
        assert_eq!(first_tool_name(&result), "get_weather");
    }

    #[test]
    fn test_tool_call_with_surrounding_text() {
        let input = r#"Let me check the weather for you.
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
I've requested the weather."#;
        let result = assert_parse(input, 1, Some("Let me check"));
        assert!(result.text.contains("I've requested"));
    }

    #[test]
    fn test_multiple_tool_calls() {
        let input = r#"<tool_call>
{"name": "search", "arguments": {"query": "rust"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"expression": "2+2"}}
</tool_call>"#;
        let result = assert_parse(input, 2, None);
        assert_eq!(first_tool_name(&result), "search");
        assert_eq!(result.tool_calls.get(1).unwrap().name, "calculate");
    }

    #[test]
    fn test_invalid_json_in_tool_call() {
        assert_parse(
            "<tool_call>\nnot valid json\n</tool_call>",
            0,
            Some("not valid json"),
        );
    }

    #[test]
    fn test_unclosed_tool_call_tag() {
        assert_parse(
            "Text before <tool_call>\n{\"name\": \"test\"}",
            0,
            Some("<tool_call>"),
        );
    }

    #[test]
    fn test_tool_call_missing_arguments() {
        let input = r#"<tool_call>
{"name": "no_args_tool"}
</tool_call>"#;
        let result = assert_parse(input, 1, None);
        assert_eq!(first_tool_name(&result), "no_args_tool");
        assert!(result.tool_calls.first().unwrap().arguments.is_object());
    }

    #[test]
    fn test_tool_call_missing_name() {
        let input = r#"<tool_call>
{"arguments": {"key": "value"}}
</tool_call>"#;
        assert_parse(input, 0, None);
    }

    #[test]
    fn test_empty_text() {
        let result = assert_parse("", 0, None);
        assert!(!result.text.contains("[ping()]"));
    }

    #[test]
    fn test_invalid_json_preserves_original_tags() {
        let input = "<tool_call>\nnot valid json\n</tool_call>";
        let result = assert_parse(input, 0, Some("<tool_call>"));
        assert!(result.text.contains("</tool_call>"));
        assert!(result.text.contains("not valid json"));
    }

    #[test]
    fn test_mix_of_valid_and_invalid_tool_calls() {
        let input = r#"<tool_call>
{"name": "good_tool", "arguments": {"key": "value"}}
</tool_call>
<tool_call>
this is not json
</tool_call>
<tool_call>
{"name": "another_good", "arguments": {}}
</tool_call>"#;
        let result = assert_parse(input, 2, Some("this is not json"));
        assert_eq!(first_tool_name(&result), "good_tool");
        assert_eq!(result.tool_calls.get(1).unwrap().name, "another_good");
    }

    #[test]
    fn test_valid_json_but_missing_name_preserved_as_raw() {
        let input = r#"<tool_call>
{"arguments": {"key": "value"}, "description": "no name field"}
</tool_call>"#;
        assert_raw_preserved(input);
        let result = parse_tool_calls(input, None);
        assert!(result.text.contains("no name field"));
    }

    #[test]
    fn test_valid_json_array_not_object_preserved_as_raw() {
        let input = "<tool_call>\n[1, 2, 3]\n</tool_call>";
        assert_raw_preserved(input);
        let result = parse_tool_calls(input, None);
        assert!(result.text.contains("[1, 2, 3]"));
    }

    #[test]
    fn test_valid_json_name_is_not_string_preserved_as_raw() {
        let input = r#"<tool_call>
{"name": 42, "arguments": {}}
</tool_call>"#;
        assert_raw_preserved(input);
    }

    #[test]
    fn test_text_between_multiple_tool_calls() {
        let input = r#"Before first.
<tool_call>
{"name": "tool_a", "arguments": {}}
</tool_call>
Middle text.
<tool_call>
{"name": "tool_b", "arguments": {}}
</tool_call>
After last."#;
        let result = assert_parse(input, 2, Some("Before first."));
        assert!(result.text.contains("Middle text."));
        assert!(result.text.contains("After last."));
    }

    #[test]
    fn test_nested_tool_call_tags() {
        // A <tool_call> tag nested inside another -- the inner one becomes
        // part of the content between the first open and first close.
        let input = r#"<tool_call>
<tool_call>
{"name": "inner", "arguments": {}}
</tool_call>
</tool_call>"#;
        let result = parse_tool_calls(input, None);
        // The parser finds the first <tool_call>, then looks for first </tool_call>.
        // Content between them: "\n<tool_call>\n{\"name\": \"inner\", \"arguments\": {}}\n"
        // This is not valid JSON (starts with <tool_call>), so it's preserved as raw text.
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("<tool_call>"));
    }

    #[test]
    fn test_arguments_as_json_array() {
        let input = r#"<tool_call>
{"name": "batch_op", "arguments": [1, 2, 3]}
</tool_call>"#;
        let result = assert_parse(input, 1, None);
        assert_eq!(first_tool_name(&result), "batch_op");
        let first = result.tool_calls.first().unwrap();
        assert!(first.arguments.is_array());
        assert_eq!(first.arguments, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_arguments_with_special_chars_and_unicode() {
        let input = r#"<tool_call>
{"name": "translate", "arguments": {"text": "Caf\u00e9 \"quotes\" \\backslash", "emoji": "\ud83d\ude00"}}
</tool_call>"#;
        let result = assert_parse(input, 1, None);
        assert_eq!(first_tool_name(&result), "translate");
        let text_val = result
            .tool_calls
            .first()
            .unwrap()
            .arguments
            .get("text")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(text_val.contains("Caf\u{00e9}"));
        assert!(text_val.contains("\"quotes\""));
        assert!(text_val.contains("\\backslash"));
    }

    #[test]
    fn test_whitespace_only_content_between_tags() {
        let input = "<tool_call>\n   \n  \t  \n</tool_call>";
        assert_parse(input, 0, Some("<tool_call>"));
    }

    // ============================================================
    // StreamingToolCallTracker tests
    //
    // The tracker is a state machine fed text chunks. It buffers
    // until it sees `<tool_call>…</tool_call>` boundaries, returning
    // (visible_text, completed_tool_calls) on every chunk.
    //
    // Invariants tested:
    // 1. inactive=false → pure passthrough, zero overhead
    // 2. complete tag in one chunk → tool call emitted, no visible
    // 3. tag split across chunks → tracker reassembles
    // 4. text before/after tag → both visible, tool extracted
    // 5. invalid JSON inside tag → preserved as visible
    // 6. unclosed tag at flush → buffered prefix emitted as visible
    // 7. multi-byte UTF-8 boundary at buffer-tail → no panic
    // 8. has_tool_calls / completed_count track state correctly
    // ============================================================

    fn drain_visible_and_calls(
        tracker: &mut StreamingToolCallTracker,
        chunks: &[&str],
    ) -> (String, Vec<ParsedToolCall>) {
        let mut visible = String::new();
        let mut calls = Vec::new();
        for chunk in chunks {
            let out = tracker.process(chunk);
            visible.push_str(&out.visible);
            calls.extend(out.new_tool_calls);
        }
        let final_out = tracker.flush();
        visible.push_str(&final_out.visible);
        calls.extend(final_out.new_tool_calls);
        (visible, calls)
    }

    #[test]
    fn streaming_inactive_is_passthrough() {
        let mut t = StreamingToolCallTracker::new(false, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[
                "hello ",
                "<tool_call>",
                "{\"name\":\"x\"}",
                "</tool_call>",
                " world",
            ],
        );
        assert_eq!(
            vis, "hello <tool_call>{\"name\":\"x\"}</tool_call> world",
            "inactive tracker must pass every chunk through verbatim",
        );
        assert!(calls.is_empty());
        assert!(!t.has_tool_calls());
        assert_eq!(t.completed_count(), 0);
    }

    #[test]
    fn streaming_single_call_one_chunk() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[r#"<tool_call>{"name":"get_weather","arguments":{"city":"London"}}</tool_call>"#],
        );
        assert!(
            vis.trim().is_empty(),
            "tool-only input should yield no visible text, got {vis:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(t.has_tool_calls());
        assert_eq!(t.completed_count(), 1);
    }

    #[test]
    fn streaming_tag_split_across_chunks() {
        // Open tag arrives in pieces; close tag also chunk-split. Tracker must reassemble.
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[
                "<tool",
                "_call>",
                r#"{"name":"search","#,
                r#""arguments":{"q":"rust"}}"#,
                "</tool",
                "_call>",
            ],
        );
        assert!(
            vis.trim().is_empty(),
            "split tags must not leak into visible, got {vis:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn streaming_text_before_and_after() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[
                "Let me check. ",
                r#"<tool_call>{"name":"lookup","arguments":{}}</tool_call>"#,
                " Done.",
            ],
        );
        assert!(vis.contains("Let me check."));
        assert!(vis.contains("Done."));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
    }

    #[test]
    fn streaming_invalid_json_preserved_as_visible() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) =
            drain_visible_and_calls(&mut t, &["<tool_call>not json</tool_call> after"]);
        assert!(vis.contains("<tool_call>"));
        assert!(vis.contains("not json"));
        assert!(vis.contains("</tool_call>"));
        assert!(vis.contains("after"));
        assert!(calls.is_empty());
        assert_eq!(t.completed_count(), 0);
    }

    #[test]
    fn streaming_unclosed_tag_flushed_as_visible() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(&mut t, &["<tool_call>{\"name\":\"partial\""]);
        // No closing tag ever arrives — at flush, the buffered prefix MUST be
        // emitted as visible (otherwise tokens vanish silently).
        assert!(vis.contains("<tool_call>"));
        assert!(vis.contains("partial"));
        assert!(calls.is_empty());
    }

    #[test]
    fn streaming_utf8_char_boundary_safety() {
        // The tracker's tail-flush logic must respect UTF-8 char boundaries,
        // otherwise it can panic when slicing inside a multi-byte sequence.
        let mut t = StreamingToolCallTracker::new(true, None);
        // Buffer ends just before the `é` byte sequence; next chunk completes it.
        let (vis, calls) =
            drain_visible_and_calls(&mut t, &["caf", "\u{00e9}", " and more text here"]);
        assert!(vis.contains("caf\u{00e9}"));
        assert!(vis.contains("more text"));
        assert!(calls.is_empty());
    }

    #[test]
    fn streaming_unbounded_buffer_capped_and_recovers() {
        // CRITICAL guard (closed upstream PR #63 finding): a model that
        // opens `<tool_call>` and never closes must not grow `buffer` past
        // `MAX_INSIDE_TOOL_CALL_BYTES`. On overflow we drop the parse,
        // flush the buffered bytes as visible, and reset so a later valid
        // tool call in the same stream still parses.
        let mut t = StreamingToolCallTracker::new(true, None);
        let huge = "x".repeat(MAX_INSIDE_TOOL_CALL_BYTES + 1);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[
                "<tool_call>",
                huge.as_str(),
                // Same stream, after the overflow — a well-formed call
                // arrives. The reset state must let it through.
                r#"<tool_call>{"name":"after","arguments":{}}</tool_call>"#,
            ],
        );
        assert!(
            vis.contains("<tool_call>"),
            "overflow must surface opener as visible, not silently swallow",
        );
        assert!(
            vis.contains(huge.as_str()),
            "overflow must surface buffered bytes as visible",
        );
        assert_eq!(calls.len(), 1, "post-overflow valid call still parses");
        assert_eq!(calls[0].name, "after");
        assert_eq!(t.completed_count(), 1);
    }

    #[test]
    fn streaming_multiple_calls_with_text_between() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[
                "first ",
                r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
                " middle ",
                r#"<tool_call>{"name":"b","arguments":{}}</tool_call>"#,
                " last",
            ],
        );
        assert!(vis.contains("first"));
        assert!(vis.contains("middle"));
        assert!(vis.contains("last"));
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_eq!(t.completed_count(), 2);
        assert!(t.has_tool_calls());
    }

    // ============================================================
    // Qwen XML tool-call format: <function=NAME><parameter=KEY>…
    // ============================================================

    /// The canonical XML shape Qwen3.5/3.6 emit: one string parameter,
    /// values wrapped in newlines by the template.
    #[test]
    fn xml_single_call_one_param() {
        let input = "<tool_call>\n<function=get_weather>\n<parameter=city>\nLondon\n</parameter>\n</function>\n</tool_call>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        let tc = result.tool_calls.first().unwrap();
        assert_eq!(tc.name, "get_weather");
        assert_eq!(tc.arguments, serde_json::json!({ "city": "London" }));
        assert!(!result.text.contains("[ping()]"));
    }

    /// Multiple parameters, and a multi-line value: only the single wrapping
    /// newline is stripped, internal newlines are preserved.
    #[test]
    fn xml_multi_param_multiline_value() {
        let input = "<tool_call>\n<function=write_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n<parameter=content>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(
            result.tool_calls.first().unwrap().arguments,
            serde_json::json!({ "path": "src/main.rs", "content": "line one\nline two" })
        );
    }

    /// With a declared schema, values are coerced to their JSON types — and
    /// crucially a `string`-typed `"123"` stays a string (schema beats the
    /// best-effort number guess).
    #[test]
    fn xml_schema_driven_coercion() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "configure",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": { "type": "integer" },
                        "enabled": { "type": "boolean" },
                        "opts": { "type": "object" },
                        "label": { "type": "string" }
                    }
                }
            }
        })];
        let schema = ToolSchema::from_tools(Some(tools.as_slice()));
        let input = "<tool_call>\n<function=configure>\n<parameter=count>\n42\n</parameter>\n<parameter=enabled>\ntrue\n</parameter>\n<parameter=opts>\n{\"a\": 1}\n</parameter>\n<parameter=label>\n123\n</parameter>\n</function>\n</tool_call>";
        let result = parse_tool_calls(input, schema.as_ref());
        assert_eq!(
            result.tool_calls.first().unwrap().arguments,
            serde_json::json!({ "count": 42, "enabled": true, "opts": { "a": 1 }, "label": "123" })
        );
    }

    /// An `integer`-typed parameter must reject fractional input (kept as a
    /// string) but accept whole numbers — `is_number` alone would wrongly
    /// accept `3.14`.
    #[test]
    fn xml_integer_rejects_fractional() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "f",
                "parameters": { "type": "object", "properties": { "n": { "type": "integer" } } }
            }
        })];
        let schema = ToolSchema::from_tools(Some(tools.as_slice()));
        let frac = "<tool_call>\n<function=f>\n<parameter=n>\n3.14\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(
            parse_tool_calls(frac, schema.as_ref())
                .tool_calls
                .first()
                .unwrap()
                .arguments,
            serde_json::json!({ "n": "3.14" })
        );
        let whole =
            "<tool_call>\n<function=f>\n<parameter=n>\n42\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(
            parse_tool_calls(whole, schema.as_ref())
                .tool_calls
                .first()
                .unwrap()
                .arguments,
            serde_json::json!({ "n": 42 })
        );
    }

    /// Without a schema, coercion is best-effort: valid-JSON scalars parse
    /// (`42` → number) while bare words stay strings (`London`).
    #[test]
    fn xml_no_schema_best_effort_coercion() {
        let input = "<tool_call>\n<function=f>\n<parameter=n>\n42\n</parameter>\n<parameter=city>\nLondon\n</parameter>\n</function>\n</tool_call>";
        let result = parse_tool_calls(input, None);
        assert_eq!(
            result.tool_calls.first().unwrap().arguments,
            serde_json::json!({ "n": 42, "city": "London" })
        );
    }

    /// Backward-compat guard: a JSON `<tool_call>` and an XML `<tool_call>`
    /// in the same text both parse (dispatch on shape, not on the model).
    #[test]
    fn mixed_json_and_xml_tool_calls_both_parse() {
        let input = concat!(
            "<tool_call>\n{\"name\": \"json_call\", \"arguments\": {\"x\": 1}}\n</tool_call>\n",
            "<tool_call>\n<function=xml_call>\n<parameter=y>\nhi\n</parameter>\n</function>\n</tool_call>"
        );
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "json_call");
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({ "x": 1 })
        );
        assert_eq!(result.tool_calls[1].name, "xml_call");
        assert_eq!(
            result.tool_calls[1].arguments,
            serde_json::json!({ "y": "hi" })
        );
    }

    /// The streaming tracker must reassemble an XML tool call split across
    /// chunk boundaries (inside the `<function=…>` opener and the value) and
    /// not leak any of it to visible content.
    #[test]
    fn streaming_xml_split_across_chunks() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[
                "<tool_call>\n<func",
                "tion=get_weather>\n<parameter=city>\nLon",
                "don\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert!(
            vis.trim().is_empty(),
            "split XML must not leak to visible, got {vis:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, serde_json::json!({ "city": "London" }));
        assert_eq!(t.completed_count(), 1);
    }

    /// Thinking-model + tool-call interaction. Qwen3.6 reasons first: in
    /// thinking mode the chat template opens `<think>`, so generation starts
    /// inside the think block and the tool call is emitted AFTER `</think>`.
    /// The chat route prepends `<think>`, splits reasoning via
    /// [`crate::reasoning_parser::parse_reasoning`], then runs
    /// [`parse_tool_calls`] on the remainder. A parser that scanned the whole
    /// output (or only the reasoning) would drop the call. This guards that
    /// composition — the most common thinking+tools failure mode.
    #[test]
    fn xml_tool_call_after_think_block_is_extracted() {
        // What the model generates after the template's opening `<think>`:
        let generated = "The user wants the weather. I'll call the tool.</think>\n\
            <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        // chat.rs composition: prepend `<think>` so the reasoning parser can
        // find the matching `</think>` and split reasoning from visible text.
        let reasoning = crate::reasoning_parser::parse_reasoning(&format!("<think>{generated}"));
        assert!(
            reasoning.reasoning.is_some(),
            "the `<think>` block must be split off as reasoning"
        );

        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } }
                }
            }
        })];
        let schema = ToolSchema::from_tools(Some(tools.as_slice()));
        let result = parse_tool_calls(&reasoning.text, schema.as_ref());

        assert_eq!(
            result.tool_calls.len(),
            1,
            "a tool call emitted after </think> must still be extracted, got {:?}",
            result.tool_calls
        );
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({ "city": "Paris" })
        );
    }

    // ============================================================
    // MiniCPM5 tool-call format: <function name="…"><param name="…">…
    // (no <tool_call> wrapper, attribute-named, optional CDATA values)
    // ============================================================

    /// Canonical `MiniCPM` shape: bare `<function name=…>` with one param.
    #[test]
    fn minicpm_single_call_one_param() {
        let input = r#"<function name="get_weather"><param name="city">London</param></function>"#;
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        let tc = result.tool_calls.first().unwrap();
        assert_eq!(tc.name, "get_weather");
        assert_eq!(tc.arguments, serde_json::json!({ "city": "London" }));
        assert!(!result.text.contains("[ping()]"));
    }

    /// Multiple consecutive blocks, with text before/between them preserved.
    #[test]
    fn minicpm_multiple_calls_with_text() {
        let input = concat!(
            "Sure.",
            r#"<function name="a"><param name="x">1</param></function>"#,
            " then ",
            r#"<function name="b"><param name="y">two</param></function>"#,
        );
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "a");
        // No schema → best-effort: "1" parses to a number.
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({ "x": 1 })
        );
        assert_eq!(result.tool_calls[1].name, "b");
        assert_eq!(
            result.tool_calls[1].arguments,
            serde_json::json!({ "y": "two" })
        );
        assert!(result.text.contains("Sure."));
        assert!(result.text.contains("then"));
    }

    /// A CDATA value containing both a newline and a literal `</function>`
    /// must be captured verbatim and must NOT close the block early.
    #[test]
    fn minicpm_cdata_value_with_literal_close_tag() {
        let input = "<function name=\"write\"><param name=\"code\"><![CDATA[fn main() {\n  // </function> not a real close\n}]]></param></function>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        let tc = result.tool_calls.first().unwrap();
        assert_eq!(tc.name, "write");
        let code = tc.arguments.get("code").unwrap().as_str().unwrap();
        assert!(code.contains("fn main()"));
        assert!(code.contains("</function> not a real close"));
        assert!(code.contains('\n'));
        assert!(!result.text.contains("[ping()]"));
    }

    /// Declared schema coerces `MiniCPM` param values; a `string`-typed `"123"`
    /// stays a string (schema beats the best-effort number guess).
    #[test]
    fn minicpm_schema_driven_coercion() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "cfg",
                "parameters": { "type": "object", "properties": {
                    "count": { "type": "integer" },
                    "on": { "type": "boolean" },
                    "label": { "type": "string" }
                }}
            }
        })];
        let schema = ToolSchema::from_tools(Some(tools.as_slice()));
        let input = r#"<function name="cfg"><param name="count">7</param><param name="on">true</param><param name="label">123</param></function>"#;
        let result = parse_tool_calls(input, schema.as_ref());
        assert_eq!(
            result.tool_calls.first().unwrap().arguments,
            serde_json::json!({ "count": 7, "on": true, "label": "123" })
        );
    }

    /// A function with no params yields empty arguments, not a failure.
    #[test]
    fn minicpm_no_param_function() {
        let input = r#"<function name="ping"></function>"#;
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls.first().unwrap().name, "ping");
        assert_eq!(
            result.tool_calls.first().unwrap().arguments,
            serde_json::json!({})
        );
    }

    /// A `<function>` opener with no `name="…"` attribute must NOT borrow the
    /// `name` from a nested `<param>` — the block is preserved verbatim rather
    /// than routed into the tool-execution path as a call named "city".
    #[test]
    fn minicpm_function_without_name_is_not_parsed() {
        let input = "<function ><param name=\"city\">Paris</param></function>";
        let result = parse_tool_calls(input, None);
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("<param name=\"city\">"));
    }

    /// Streaming: the tracker reassembles a `MiniCPM` call split inside the
    /// `<function` opener AND inside a CDATA value, with no leak to visible.
    #[test]
    fn streaming_minicpm_split_across_chunks() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[
                "<func",
                "tion name=\"run\"><param name=\"cmd\">",
                "<![CDATA[echo ",
                "hi]]></param></function>",
            ],
        );
        assert!(
            vis.trim().is_empty(),
            "split MiniCPM must not leak to visible, got {vis:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "run");
        assert_eq!(calls[0].arguments, serde_json::json!({ "cmd": "echo hi" }));
        assert_eq!(t.completed_count(), 1);
    }

    // ============================================================

    // ============================================================
    // Bare LFM2 tool calls: [func(args)] without delimiters
    //
    // LFM models sometimes emit bare bracket-enclosed function
    // calls at the end of output without the
    // <|tool_call_start|> / <|tool_call_end|> wrappers.
    // ============================================================

    /// Bare LFM2-style call at end of text — no delimiters.
    #[test]
    fn bare_lfm2_single_call_at_end() {
        let input = "<|im_start|>assistant\nHere is the answer.[get_weather(city='London')]";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({"city": "London"})
        );
        assert!(result.text.contains("Here is the answer."));
    }

    /// Bare LFM2 call with multiple comma-separated functions.
    #[test]
    fn bare_lfm2_multiple_calls() {
        let input = "<|im_start|>assistant\nI'll look that up.[search(query='rust'), get_time()]";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[1].name, "get_time");
        assert!(result.text.contains("look that up"));
    }

    /// Bare LFM2 call at the very end with no prefix text.
    #[test]
    fn bare_lfm2_only_call_no_text() {
        let input = "<|im_start|>assistant\n[ping()]";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "ping");
        assert!(!result.text.contains("[ping()]"));
    }

    /// A bracketed list that is NOT a function call (e.g. data list)
    /// must NOT be parsed as a tool call.
    #[test]
    fn bare_list_without_func_not_parsed() {
        let input = "Here's a list: [1, 2, 3]";
        let result = parse_tool_calls(input, None);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.text, "Here's a list: [1, 2, 3]");
    }

    /// Bracketed list mid-text (not at end) is not a tool call.
    #[test]
    fn bare_brackets_mid_text_not_parsed() {
        let input = "Start [get_weather(city='Paris')] and more text.";
        let result = parse_tool_calls(input, None);
        // Brackets not at end — should not be treated as tool call
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.text, input);
    }

    /// Bare LFM2 with mixed argument types.
    #[test]
    fn bare_lfm2_mixed_arg_types() {
        let input = "<|im_start|>assistant\nConfiguring.[set(count=5, enabled=True, name='test')]";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({"count": 5, "enabled": true, "name": "test"})
        );
        assert!(result.text.contains("Configuring"));
    }

    /// Bare LFM2 with thinking/rambling text prefix — the full scenario
    /// from the user's screenshot where thinking leaks into context.
    #[test]
    fn bare_lfm2_with_thinking_prefix() {
        let input = r#"<|im_start|>assistant\nThe user is asking me to introduce myself. I should provide information about my identity as LFM (Liquid Foundation Model) by Liquid AI. I can use the get_skills tool or just answer based on my system instructions. Let me first check what skills/memory I have access to, and also recall any background information. I'll start by reading my own profile/system info to give an accurate introduction. But actually, the prompt already tells me who I am - "You are an LFM (Liquid Foundation Model) by Liquid AI." So I should present this information accurately. Let me provide a clear introduction based on the canonical facts: I'm LFM, built by Liquid AI, part of a family of models, my architecture is hybrid, designed for efficient on-device inference, available in various sizes. I should be warm and confident, not recite a biography. Let me give a natural introduction.Hi there! I'm LFM (Liquid Foundation Model), built by Liquid AI. I'm part of a family of models designed for efficient, on-device intelligence—think phones, laptops, cars, robots, and edge hardware. My architecture is a hybrid: most layers use gated short convolutions (fast and memory-efficient), with a minority of grouped-query attention layers for long-range context. This mix was chosen through hardware-aware search to run quickly on devices while still handling complex tasks. I come in a range of sizes—from small models that fit on a phone up to larger mixture-of-experts variants—and I support text, vision, and audio inputs. Key points: Openly available weights — you can run me locally. Fast on-device inference — thanks to the convolution-heavy design. Built by Liquid AI (spun out of MIT CSAIL, Cambridge, MA). I'm happy to help with coding, reasoning, creativity, explanation, or just chatting. What can I assist you with today?[read_file(path='/home/user'), get_skills()]"#;
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "read_file");
        assert_eq!(result.tool_calls[1].name, "get_skills");
        // Prefix text should end before the brackets
        assert!(result.text.ends_with("today?"));
        assert!(!result.text.contains("[read_file"));
    }

    // LFM2 / Macaw tool-call format:
    //   <|tool_call_start|>[func_name(arg1='value1', arg2=42), func2()]<|tool_call_end|>
    //
    // Key differences from Qwen:
    //   - Delimiters: <|tool_call_start|> / <|tool_call_end|>
    //   - Content: Python-style function calls [func(args)]
    //   - Multiple calls in one block: comma-separated list inside [...]
    // ============================================================

    /// Canonical single LFM2 call with string arguments.
    #[test]
    fn lfm2_single_call_string_args() {
        let input = "<|tool_call_start|>[get_weather(city='London')]<|tool_call_end|>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        let tc = result.tool_calls.first().unwrap();
        assert_eq!(tc.name, "get_weather");
        assert_eq!(tc.arguments, serde_json::json!({ "city": "London" }));
        assert!(!result.text.contains("[ping()]"));
    }

    /// LFM2 call with mixed argument types: string, integer, float,
    /// boolean (Python-style True/False), and None.
    #[test]
    fn lfm2_mixed_arg_types() {
        let input = "<|tool_call_start|>[configure(count=42, enabled=True, ratio=3.14, label='hello', nothing=None)]<|tool_call_end|>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        let tc = result.tool_calls.first().unwrap();
        assert_eq!(tc.name, "configure");
        assert_eq!(
            tc.arguments,
            serde_json::json!({
                "count": 42,
                "enabled": true,
                "ratio": 3.14,
                "label": "hello",
                "nothing": null
            })
        );
    }

    /// LFM2 call with no arguments.
    #[test]
    fn lfm2_no_args() {
        let input = "<|tool_call_start|>[ping()]<|tool_call_end|>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        let tc = result.tool_calls.first().unwrap();
        assert_eq!(tc.name, "ping");
        assert_eq!(tc.arguments, serde_json::json!({}));
    }

    /// LFM2 block with multiple comma-separated calls inside one bracket list.
    #[test]
    fn lfm2_multiple_calls_in_block() {
        let input = "<|tool_call_start|>[get_weather(city='London'), search(query='rust', limit=10)]<|tool_call_end|>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({ "city": "London" })
        );
        assert_eq!(result.tool_calls[1].name, "search");
        assert_eq!(
            result.tool_calls[1].arguments,
            serde_json::json!({ "query": "rust", "limit": 10 })
        );
    }

    /// LFM2 call with surrounding text preserved.
    #[test]
    fn lfm2_with_surrounding_text() {
        let input = "Let me check.\n<|tool_call_start|>[lookup(key='weather')]<|tool_call_end|>\nDone.";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.text.contains("Let me check."));
        assert!(result.text.contains("Done."));
        assert_eq!(result.tool_calls[0].name, "lookup");
    }

    /// Multiple LFM2 blocks in the same text.
    #[test]
    fn lfm2_multiple_blocks() {
        let input = "<|tool_call_start|>[a(x=1)]<|tool_call_end|> mid <|tool_call_start|>[b(y='two')]<|tool_call_end|>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "a");
        assert_eq!(result.tool_calls[1].name, "b");
        assert!(result.text.contains("mid"));
    }

    /// LFM2 with double-quoted string values.
    #[test]
    fn lfm2_double_quoted_strings() {
        let input = r#"<|tool_call_start|>[echo(msg="hello world")]<|tool_call_end|>"#;
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({ "msg": "hello world" })
        );
    }

    /// LFM2 with negative numbers.
    #[test]
    fn lfm2_negative_numbers() {
        let input = "<|tool_call_start|>[adjust(offset=-5, factor=-2.5)]<|tool_call_end|>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({ "offset": -5, "factor": -2.5 })
        );
    }

    /// LFM2 with False boolean.
    #[test]
    fn lfm2_bool_false() {
        let input = "<|tool_call_start|>[toggle(active=False)]<|tool_call_end|>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({ "active": false })
        );
    }

    /// Unclosed LFM2 block is preserved as visible text.
    #[test]
    fn lfm2_unclosed_tag_preserved() {
        let input = "prefix <|tool_call_start|>[incomplete(";
        let result = parse_tool_calls(input, None);
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("<|tool_call_start|>"));
        assert!(result.text.contains("[incomplete("));
    }

    /// Invalid LFM2 content (not bracketed) preserved as raw.
    #[test]
    fn lfm2_invalid_content_preserved() {
        let input = "<|tool_call_start|>not a bracketed list<|tool_call_end|>";
        let result = parse_tool_calls(input, None);
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("<|tool_call_start|>"));
        assert!(result.text.contains("not a bracketed list"));
    }

    /// LFM2 call with empty string value.
    #[test]
    fn lfm2_empty_string_value() {
        let input = "<|tool_call_start|>[log(msg='')]<|tool_call_end|>";
        let result = parse_tool_calls(input, None);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({ "msg": "" })
        );
    }

    /// Streaming: LFM2 call in a single chunk.
    #[test]
    fn streaming_lfm2_single_chunk() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &["<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>"],
        );
        assert!(
            vis.trim().is_empty(),
            "pure LFM2 call should yield no visible text, got {vis:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, serde_json::json!({ "city": "Paris" }));
        assert_eq!(t.completed_count(), 1);
    }

    /// Streaming: LFM2 tag split across chunk boundaries.
    #[test]
    fn streaming_lfm2_split_across_chunks() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[
                "<|tool_call",
                "_start|>",
                "[search(query='rust', ",
                "limit=5)]",
                "<|tool_call_",
                "end|>",
            ],
        );
        assert!(
            vis.trim().is_empty(),
            "split LFM2 must not leak to visible, got {vis:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(
            calls[0].arguments,
            serde_json::json!({ "query": "rust", "limit": 5 })
        );
        assert_eq!(t.completed_count(), 1);
    }

    /// Streaming: LFM2 with surrounding text.
    #[test]
    fn streaming_lfm2_with_text() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[
                "Okay, ",
                "<|tool_call_start|>[do_it()]<|tool_call_end|>",
                " there.",
            ],
        );
        assert!(vis.contains("Okay,"));
        assert!(vis.contains("there."));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "do_it");
    }

    /// Streaming: unclosed LFM2 tag flushed as visible.
    #[test]
    fn streaming_lfm2_unclosed_tag_flushed() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &["<|tool_call_start|>[partial(stuck='yes'"],
        );
        assert!(vis.contains("<|tool_call_start|>"));
        assert!(vis.contains("partial"));
        assert!(vis.contains("stuck"));
        assert!(calls.is_empty());
    }

    /// Streaming: LFM2 invalid content preserved.
    #[test]
    fn streaming_lfm2_invalid_content_preserved() {
        let mut t = StreamingToolCallTracker::new(true, None);
        let (vis, calls) = drain_visible_and_calls(
            &mut t,
            &[
                "<|tool_call_start|>bad stuff<|tool_call_end|> after",
            ],
        );
        assert!(vis.contains("<|tool_call_start|>"));
        assert!(vis.contains("bad stuff"));
        assert!(vis.contains("after"));
        assert!(calls.is_empty());
    }


}
