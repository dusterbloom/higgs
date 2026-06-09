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

    ToolParseResult {
        text: result_text.trim().to_owned(),
        tool_calls,
    }
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

/// State machine that buffers streaming text chunks and extracts
/// `<tool_call>{json}</tool_call>` blocks on the fly.
///
/// Designed to be cheap: when `active = false` (no tools in the request),
/// `process` is a single allocation per chunk and `flush` is a no-op.
///
/// When active, it accumulates a tail buffer just large enough that the
/// `<tool_call>` opener can't straddle a chunk boundary. As soon as a
/// complete `<tool_call>…</tool_call>` block is in the buffer, the JSON
/// between the tags is parsed and emitted as a [`ParsedToolCall`]. Text
/// before/after tags streams out verbatim.
///
/// Invariants:
/// - **Never silently drops tokens.** Unclosed tool-call tags at `flush`
///   are re-emitted (with the `<tool_call>` opener prepended) as visible
///   content rather than discarded.
/// - **UTF-8 safe.** Tail-flushes walk back to the previous char boundary
///   so a partial multi-byte sequence is never split.
/// - **Pure passthrough when inactive.** Zero parsing cost on requests
///   that did not pass `tools` to the chat route.
pub struct StreamingToolCallTracker {
    buffer: String,
    inside_tool_call: bool,
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
            inside_tool_call: false,
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
            if self.inside_tool_call {
                // Look for closing tag — once seen, parse the JSON body
                // and continue scanning (another tool call may follow in
                // the same chunk).
                if let Some(end) = self.buffer.find(TOOL_CALL_CLOSE) {
                    let raw_block = self.buffer.get(..end).unwrap_or_default();
                    let call_content = raw_block.trim();
                    if let Some(parsed) = parse_tool_call_block(call_content, self.schema.as_ref())
                    {
                        out.new_tool_calls.push(parsed);
                        self.completed_count += 1;
                    } else {
                        // Invalid JSON inside the tag — preserve verbatim
                        // so the client/operator can see what the model
                        // emitted instead of silent loss.
                        out.visible.push_str(TOOL_CALL_OPEN);
                        out.visible.push_str(raw_block);
                        out.visible.push_str(TOOL_CALL_CLOSE);
                    }
                    self.buffer = self
                        .buffer
                        .get(end + TOOL_CALL_CLOSE.len()..)
                        .unwrap_or_default()
                        .to_owned();
                    self.inside_tool_call = false;
                } else if self.buffer.len() > MAX_INSIDE_TOOL_CALL_BYTES {
                    // Overflow guard: `<tool_call>` opened but the close
                    // tag never arrived within the cap. Abandon the parse
                    // and emit verbatim so tokens still aren't dropped.
                    let leftover = std::mem::take(&mut self.buffer);
                    out.visible.push_str(TOOL_CALL_OPEN);
                    out.visible.push_str(&leftover);
                    self.inside_tool_call = false;
                    break;
                } else {
                    // Still accumulating — wait for more.
                    break;
                }
            } else if let Some(start) = self.buffer.find(TOOL_CALL_OPEN) {
                // Everything before the open tag flushes out as visible.
                out.visible
                    .push_str(self.buffer.get(..start).unwrap_or_default());
                self.buffer = self
                    .buffer
                    .get(start + TOOL_CALL_OPEN.len()..)
                    .unwrap_or_default()
                    .to_owned();
                self.inside_tool_call = true;
            } else if self.buffer.len() > TOOL_CALL_OPEN.len() {
                // No open tag yet, but the buffer is bigger than the open
                // tag — safe to flush all-but-tail. Keep
                // `TOOL_CALL_OPEN.len()` bytes at the tail so a partial
                // `<tool_call>` straddling the next chunk is still
                // detectable.
                let target_len = self.buffer.len() - TOOL_CALL_OPEN.len();
                // Walk back to the previous UTF-8 char boundary so a
                // multi-byte sequence is never split.
                let mut safe_len = target_len;
                while safe_len > 0 && !self.buffer.is_char_boundary(safe_len) {
                    safe_len -= 1;
                }
                out.visible
                    .push_str(self.buffer.get(..safe_len).unwrap_or_default());
                self.buffer = self.buffer.get(safe_len..).unwrap_or_default().to_owned();
                break;
            } else {
                // Buffer smaller than the open tag — keep waiting.
                break;
            }
        }

        out
    }

    /// Drain everything still buffered. Call this when the model stream
    /// ends. Any unclosed `<tool_call>` block is emitted as visible content
    /// (with its opener prepended) so no tokens silently vanish.
    pub fn flush(&mut self) -> StreamingToolOutput {
        let leftover = std::mem::take(&mut self.buffer);
        let was_inside = self.inside_tool_call;
        self.inside_tool_call = false;

        if was_inside {
            let mut visible = String::with_capacity(TOOL_CALL_OPEN.len() + leftover.len());
            visible.push_str(TOOL_CALL_OPEN);
            visible.push_str(&leftover);
            StreamingToolOutput {
                visible,
                new_tool_calls: Vec::new(),
            }
        } else {
            StreamingToolOutput {
                visible: leftover,
                new_tool_calls: Vec::new(),
            }
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
        assert!(result.text.is_empty());
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
        assert!(result.text.is_empty());
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
        assert!(result.text.is_empty());
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
}
