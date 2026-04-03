//! Parse tool calls from model-generated text.
//!
//! Qwen models emit tool calls in a specific XML-like format:
//! ```text
//! <tool_call>
//! {"name": "function_name", "arguments": {"arg1": "value1"}}
//! </tool_call>
//! ```
//!
//! This module extracts those structured tool calls from the raw text.

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

/// Parse model output text for Qwen-format tool calls.
///
/// Returns the non-tool-call text and any extracted tool calls.
pub fn parse_tool_calls(text: &str) -> ToolParseResult {
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

                if let Some(parsed) = try_parse_tool_call(call_content) {
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

/// Output from the streaming tool call tracker.
#[derive(Debug, Clone)]
pub struct StreamingToolOutput {
    /// Text that should be emitted as visible content.
    pub visible: String,
    /// Completed tool calls parsed in this chunk.
    pub new_tool_calls: Vec<ParsedToolCall>,
}

/// Streaming tool call state tracker.
///
/// Buffers incoming text token-by-token, detecting `<tool_call>...</tool_call>`
/// boundaries. Tool call content is held back (not emitted as visible) until
/// the closing tag arrives, at which point the JSON is parsed and returned
/// as a structured `ParsedToolCall`.
///
/// When `active` is false (no tools in the request), all text passes through
/// as visible content with zero overhead.
pub struct StreamingToolCallTracker {
    buffer: String,
    inside_tool_call: bool,
    completed_count: usize,
    active: bool,
}

impl StreamingToolCallTracker {
    pub fn new(active: bool) -> Self {
        Self {
            buffer: String::new(),
            inside_tool_call: false,
            completed_count: 0,
            active,
        }
    }

    /// Process a new text chunk. Returns visible text and any completed tool calls.
    pub fn process(&mut self, text: &str) -> StreamingToolOutput {
        if !self.active {
            return StreamingToolOutput {
                visible: text.to_owned(),
                new_tool_calls: vec![],
            };
        }

        self.buffer.push_str(text);

        let mut visible = String::new();
        let mut new_tool_calls = Vec::new();

        loop {
            if self.inside_tool_call {
                if let Some(end_pos) = self.buffer.find(TOOL_CALL_CLOSE) {
                    let raw_block = self.buffer.get(..end_pos).unwrap_or_default();
                    let call_content = raw_block.trim();

                    if let Some(parsed) = try_parse_tool_call(call_content) {
                        new_tool_calls.push(parsed);
                        self.completed_count += 1;
                    } else {
                        // Invalid JSON — emit raw tags as visible text
                        visible.push_str(TOOL_CALL_OPEN);
                        visible.push_str(raw_block);
                        visible.push_str(TOOL_CALL_CLOSE);
                    }

                    self.buffer = self
                        .buffer
                        .get(end_pos + TOOL_CALL_CLOSE.len()..)
                        .unwrap_or_default()
                        .to_owned();
                    self.inside_tool_call = false;
                } else {
                    // Still accumulating tool call content — don't flush
                    break;
                }
            } else if let Some(start_pos) = self.buffer.find(TOOL_CALL_OPEN) {
                visible.push_str(self.buffer.get(..start_pos).unwrap_or_default());
                self.buffer = self
                    .buffer
                    .get(start_pos + TOOL_CALL_OPEN.len()..)
                    .unwrap_or_default()
                    .to_owned();
                self.inside_tool_call = true;
            } else if self.buffer.len() > TOOL_CALL_CLOSE.len() {
                // Flush all but the tail (could be a partial <tool_call> tag)
                let mut safe_len = self.buffer.len() - TOOL_CALL_CLOSE.len();
                while safe_len > 0 && !self.buffer.is_char_boundary(safe_len) {
                    safe_len -= 1;
                }
                visible.push_str(&self.buffer[..safe_len]);
                self.buffer = self.buffer[safe_len..].to_owned();
                break;
            } else {
                break;
            }
        }

        StreamingToolOutput {
            visible,
            new_tool_calls,
        }
    }

    /// Flush remaining buffer. Call when generation is complete.
    pub fn flush(&mut self) -> StreamingToolOutput {
        let buf = std::mem::take(&mut self.buffer);
        let was_inside = self.inside_tool_call;
        self.inside_tool_call = false;

        if was_inside {
            // Unclosed tool call — emit raw tag + content as visible
            let mut visible = String::from(TOOL_CALL_OPEN);
            visible.push_str(&buf);
            StreamingToolOutput {
                visible,
                new_tool_calls: vec![],
            }
        } else {
            StreamingToolOutput {
                visible: buf,
                new_tool_calls: vec![],
            }
        }
    }

    /// Whether any tool calls have been successfully parsed.
    pub const fn has_tool_calls(&self) -> bool {
        self.completed_count > 0
    }

    /// Number of tool calls parsed so far (used for OpenAI `index` field).
    pub const fn completed_count(&self) -> usize {
        self.completed_count
    }
}

/// Try to parse a single tool call block (JSON or Hermes XML format).
fn try_parse_tool_call(content: &str) -> Option<ParsedToolCall> {
    // Try JSON format first: {"name": "func", "arguments": {...}}
    if let Some(parsed) = try_parse_json_tool_call(content) {
        return Some(parsed);
    }
    // Fall back to Hermes XML format (qwen3_coder / Qwen3.5):
    // <function=NAME><parameter=KEY>VALUE</parameter></function>
    try_parse_hermes_tool_call(content)
}

/// Parse JSON-format tool call: `{"name": "func", "arguments": {...}}`
fn try_parse_json_tool_call(content: &str) -> Option<ParsedToolCall> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    let obj = value.as_object()?;

    let name = obj.get("name").and_then(|v| v.as_str())?.to_owned();

    let arguments = obj
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

    Some(ParsedToolCall { name, arguments })
}

const FUNC_OPEN: &str = "<function=";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

/// Parse Hermes XML-format tool call:
/// ```text
/// <function=get_weather>
/// <parameter=location>
/// Tokyo
/// </parameter>
/// </function>
/// ```
fn try_parse_hermes_tool_call(content: &str) -> Option<ParsedToolCall> {
    let func_start = content.find(FUNC_OPEN)?;
    let after_prefix = content.get(func_start + FUNC_OPEN.len()..)?;
    let name_end = after_prefix.find('>')?;
    let name = after_prefix.get(..name_end)?.trim().to_owned();
    if name.is_empty() {
        return None;
    }

    let func_body = after_prefix.get(name_end + 1..)?;
    let mut arguments = serde_json::Map::new();
    let mut remaining = func_body;

    while let Some(param_start) = remaining.find(PARAM_OPEN) {
        let after_param = remaining.get(param_start + PARAM_OPEN.len()..)?;
        let key_end = after_param.find('>')?;
        let key = after_param.get(..key_end)?.trim();
        if key.is_empty() {
            remaining = after_param.get(key_end + 1..)?;
            continue;
        }

        let value_start = after_param.get(key_end + 1..)?;
        let value_end = value_start.find(PARAM_CLOSE)?;
        let value = value_start.get(..value_end)?.trim();

        arguments.insert(
            key.to_owned(),
            serde_json::Value::String(value.to_owned()),
        );
        remaining = value_start.get(value_end + PARAM_CLOSE.len()..)?;
    }

    Some(ParsedToolCall {
        name,
        arguments: serde_json::Value::Object(arguments),
    })
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Parse input and assert expected tool call count and optional text fragment.
    fn assert_parse(
        input: &str,
        expected_tools: usize,
        text_contains: Option<&str>,
    ) -> ToolParseResult {
        let result = parse_tool_calls(input);
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
        let result = parse_tool_calls(input);
        assert!(result.text.contains("no name field"));
    }

    #[test]
    fn test_valid_json_array_not_object_preserved_as_raw() {
        let input = "<tool_call>\n[1, 2, 3]\n</tool_call>";
        assert_raw_preserved(input);
        let result = parse_tool_calls(input);
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
        let result = parse_tool_calls(input);
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

    // --- Hermes XML format tests ---

    #[test]
    fn test_hermes_single_param() {
        let input = "<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n</function>\n</tool_call>";
        let result = assert_parse(input, 1, None);
        assert_eq!(first_tool_name(&result), "get_weather");
        let args = &result.tool_calls.first().unwrap().arguments;
        assert_eq!(args.get("location").unwrap().as_str().unwrap(), "Tokyo");
    }

    #[test]
    fn test_hermes_multi_param() {
        let input = "<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n</tool_call>";
        let result = assert_parse(input, 1, None);
        let args = &result.tool_calls.first().unwrap().arguments;
        assert_eq!(args.get("location").unwrap().as_str().unwrap(), "Tokyo");
        assert_eq!(args.get("unit").unwrap().as_str().unwrap(), "fahrenheit");
    }

    #[test]
    fn test_hermes_multiple_tool_calls() {
        let input = "<tool_call>\n<function=get_weather>\n<parameter=location>\nLondon\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=calculate>\n<parameter=expression>\n42*17\n</parameter>\n</function>\n</tool_call>";
        let result = assert_parse(input, 2, None);
        assert_eq!(first_tool_name(&result), "get_weather");
        assert_eq!(result.tool_calls.get(1).unwrap().name, "calculate");
        assert_eq!(
            result.tool_calls.get(1).unwrap().arguments.get("expression").unwrap().as_str().unwrap(),
            "42*17"
        );
    }

    #[test]
    fn test_hermes_with_surrounding_text() {
        let input = "Let me check.\n<tool_call>\n<function=search>\n<parameter=query>\nrust programming\n</parameter>\n</function>\n</tool_call>\nDone.";
        let result = assert_parse(input, 1, Some("Let me check."));
        assert!(result.text.contains("Done."));
        assert_eq!(first_tool_name(&result), "search");
    }

    #[test]
    fn test_hermes_no_params() {
        let input = "<tool_call>\n<function=list_all>\n</function>\n</tool_call>";
        let result = assert_parse(input, 1, None);
        assert_eq!(first_tool_name(&result), "list_all");
        assert!(result.tool_calls.first().unwrap().arguments.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_hermes_multiline_value() {
        let input = "<tool_call>\n<function=write>\n<parameter=content>\nline one\nline two\nline three\n</parameter>\n</function>\n</tool_call>";
        let result = assert_parse(input, 1, None);
        let val = result.tool_calls.first().unwrap().arguments.get("content").unwrap().as_str().unwrap();
        assert!(val.contains("line one\nline two\nline three"));
    }

    #[test]
    fn streaming_hermes_tool_call() {
        let (visible, tool_calls) = collect_streaming(
            &[
                "<tool_call>\n<function=get_wea",
                "ther>\n<parameter=location>\nTokyo\n</param",
                "eter>\n</function>\n</tool_call>",
            ],
            true,
        );
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls.first().unwrap().name, "get_weather");
        assert!(visible.trim().is_empty());
    }

    // --- Streaming tool call tracker tests ---

    /// Collect all visible text and tool calls from a series of process() calls + flush().
    fn collect_streaming(chunks: &[&str], active: bool) -> (String, Vec<ParsedToolCall>) {
        let mut tracker = StreamingToolCallTracker::new(active);
        let mut visible = String::new();
        let mut tool_calls = Vec::new();
        for chunk in chunks {
            let out = tracker.process(chunk);
            visible.push_str(&out.visible);
            tool_calls.extend(out.new_tool_calls);
        }
        let out = tracker.flush();
        visible.push_str(&out.visible);
        tool_calls.extend(out.new_tool_calls);
        (visible, tool_calls)
    }

    #[test]
    fn streaming_inactive_passthrough() {
        let (visible, tool_calls) = collect_streaming(
            &["Hello <tool_call>\n{\"name\":\"test\"}\n</tool_call>"],
            false,
        );
        assert!(tool_calls.is_empty());
        assert!(visible.contains("<tool_call>"));
    }

    #[test]
    fn streaming_no_tool_calls() {
        let (visible, tool_calls) =
            collect_streaming(&["Hello", " world", ", no tools here."], true);
        assert!(tool_calls.is_empty());
        assert!(visible.contains("Hello"));
        assert!(visible.contains("world"));
        assert!(visible.contains("no tools here."));
    }

    #[test]
    fn streaming_single_tool_call_one_chunk() {
        let input =
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}\n</tool_call>";
        let (visible, tool_calls) = collect_streaming(&[input], true);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls.first().unwrap().name, "get_weather");
        assert!(visible.trim().is_empty());
    }

    #[test]
    fn streaming_tool_call_split_across_chunks() {
        let (visible, tool_calls) = collect_streaming(
            &[
                "Let me check. <tool_",
                "call>\n{\"name\": \"search\", \"arguments\": {\"q\"",
                ": \"rust\"}}\n</tool_call>",
                " Done.",
            ],
            true,
        );
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls.first().unwrap().name, "search");
        assert!(visible.contains("Let me check."));
        assert!(visible.contains("Done."));
        assert!(!visible.contains("<tool_call>"));
    }

    #[test]
    fn streaming_multiple_tool_calls() {
        let (visible, tool_calls) = collect_streaming(
            &[
                "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>",
                "<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>",
            ],
            true,
        );
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls.first().unwrap().name, "a");
        assert_eq!(tool_calls.get(1).unwrap().name, "b");
        assert!(visible.trim().is_empty());
    }

    #[test]
    fn streaming_invalid_json_emitted_as_visible() {
        let (visible, tool_calls) = collect_streaming(
            &["<tool_call>\nnot valid json\n</tool_call>"],
            true,
        );
        assert!(tool_calls.is_empty());
        assert!(visible.contains("not valid json"));
        assert!(visible.contains("<tool_call>"));
        assert!(visible.contains("</tool_call>"));
    }

    #[test]
    fn streaming_flush_inside_unclosed_tool_call() {
        let mut tracker = StreamingToolCallTracker::new(true);
        tracker.process("<tool_call>\n{\"name\": \"test\"");
        let out = tracker.flush();
        assert!(out.new_tool_calls.is_empty());
        assert!(out.visible.contains("<tool_call>"));
        assert!(out.visible.contains("\"name\": \"test\""));
    }

    #[test]
    fn streaming_completed_count() {
        let mut tracker = StreamingToolCallTracker::new(true);
        assert_eq!(tracker.completed_count(), 0);
        assert!(!tracker.has_tool_calls());

        tracker.process(
            "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>",
        );
        assert_eq!(tracker.completed_count(), 1);
        assert!(tracker.has_tool_calls());

        tracker.process(
            "<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>",
        );
        assert_eq!(tracker.completed_count(), 2);
    }

    #[test]
    fn streaming_text_between_tool_calls() {
        let (visible, tool_calls) = collect_streaming(
            &[
                "Before. ",
                "<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>",
                " Middle. ",
                "<tool_call>\n{\"name\": \"y\", \"arguments\": {}}\n</tool_call>",
                " After.",
            ],
            true,
        );
        assert_eq!(tool_calls.len(), 2);
        assert!(visible.contains("Before."));
        assert!(visible.contains("Middle."));
        assert!(visible.contains("After."));
    }

    #[test]
    fn streaming_partial_open_tag_at_boundary() {
        // The open tag "<tool_call>" gets split right at a chunk boundary.
        let (visible, tool_calls) = collect_streaming(
            &[
                "Hello <tool",
                "_call>\n{\"name\": \"split\", \"arguments\": {}}\n</tool_call>",
            ],
            true,
        );
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls.first().unwrap().name, "split");
        assert!(visible.contains("Hello"));
        assert!(!visible.contains("<tool"));
    }

    #[test]
    fn streaming_partial_close_tag_at_boundary() {
        let (visible, tool_calls) = collect_streaming(
            &[
                "<tool_call>\n{\"name\": \"split\", \"arguments\": {}}\n</tool",
                "_call> Done.",
            ],
            true,
        );
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls.first().unwrap().name, "split");
        assert!(visible.contains("Done."));
    }
}
