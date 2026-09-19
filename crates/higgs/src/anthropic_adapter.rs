use crate::types::anthropic::{AnthropicContent, AnthropicMessage, ContentBlock, SystemPrompt};

/// Map an `OpenAI` `finish_reason` to an Anthropic `stop_reason`.
pub fn openai_finish_to_anthropic_stop(finish_reason: &str) -> String {
    match finish_reason {
        "stop" => "end_turn".to_owned(),
        "length" => "max_tokens".to_owned(),
        "tool_calls" => "tool_use".to_owned(),
        other => other.to_owned(),
    }
}

/// Rebuild Anthropic message content with the family image marker spliced at
/// each image block's true position, mirroring `routes::chat::render_markers`.
///
/// Text blocks keep their relative order; each top-level `image` block (and
/// each image nested inside `tool_result` content — the extractor collects
/// those at the enclosing block's position) becomes one marker run. Every
/// other block type contributes nothing, exactly as
/// [`anthropic_messages_to_engine`] handles it today, so the marker count
/// always matches the image count the extractor produces.
pub fn render_anthropic_markers(
    messages: &[AnthropicMessage],
    marker: Option<&'static str>,
) -> Vec<AnthropicMessage> {
    let marker_text = marker.unwrap_or("<image>");
    messages
        .iter()
        .map(|m| {
            let AnthropicContent::Blocks(blocks) = &m.content else {
                return m.clone();
            };
            let mut out = String::new();
            for block in blocks {
                match block {
                    ContentBlock::Text { text } => out.push_str(text),
                    ContentBlock::Image { .. } => out.push_str(marker_text),
                    ContentBlock::ToolResult { content, .. } => {
                        if let AnthropicContent::Blocks(inner) = content {
                            for inner_block in inner {
                                if matches!(inner_block, ContentBlock::Image { .. }) {
                                    out.push_str(marker_text);
                                }
                            }
                        }
                    }
                    ContentBlock::ToolUse { .. }
                    | ContentBlock::Thinking { .. }
                    | ContentBlock::RedactedThinking { .. }
                    | ContentBlock::Document { .. }
                    | ContentBlock::ServerToolUse { .. }
                    | ContentBlock::WebSearchToolResult { .. }
                    | ContentBlock::CodeExecutionToolResult { .. }
                    | ContentBlock::Other => {}
                }
            }
            AnthropicMessage {
                role: m.role.clone(),
                content: AnthropicContent::Text(out),
            }
        })
        .collect()
}

/// Convert Anthropic messages to the engine's `ChatMessage` format.
pub fn anthropic_messages_to_engine(
    messages: &[AnthropicMessage],
    system: Option<&SystemPrompt>,
) -> Vec<higgs_engine::chat_template::ChatMessage> {
    let mut result = Vec::new();

    if let Some(sys) = system {
        result.push(higgs_engine::chat_template::ChatMessage {
            role: "system".to_owned(),
            content: sys.to_text(),
            tool_calls: None,
        });
    }

    for msg in messages {
        let content = match &msg.content {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    crate::types::anthropic::ContentBlock::Text { text } => Some(text.as_str()),
                    crate::types::anthropic::ContentBlock::ToolUse { .. }
                    | crate::types::anthropic::ContentBlock::ToolResult { .. }
                    | crate::types::anthropic::ContentBlock::Thinking { .. }
                    | crate::types::anthropic::ContentBlock::RedactedThinking { .. }
                    | crate::types::anthropic::ContentBlock::Image { .. }
                    | crate::types::anthropic::ContentBlock::Document { .. }
                    | crate::types::anthropic::ContentBlock::ServerToolUse { .. }
                    | crate::types::anthropic::ContentBlock::WebSearchToolResult { .. }
                    | crate::types::anthropic::ContentBlock::CodeExecutionToolResult { .. }
                    | crate::types::anthropic::ContentBlock::Other => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        };

        result.push(higgs_engine::chat_template::ChatMessage {
            role: msg.role.clone(),
            content,
            tool_calls: None,
        });
    }

    result
}

#[allow(clippy::indexing_slicing, clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::anthropic::ContentBlock;

    fn text_message(role: &str, content: &str) -> AnthropicMessage {
        AnthropicMessage {
            role: role.to_owned(),
            content: AnthropicContent::Text(content.to_owned()),
        }
    }

    fn blocks_message(role: &str, blocks: Vec<ContentBlock>) -> AnthropicMessage {
        AnthropicMessage {
            role: role.to_owned(),
            content: AnthropicContent::Blocks(blocks),
        }
    }

    #[test]
    fn test_finish_reason_mapping() {
        let cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("other", "other"),
            ("", ""),
            ("content_filter", "content_filter"),
            ("something_new", "something_new"),
        ];
        for (input, expected) in cases {
            assert_eq!(openai_finish_to_anthropic_stop(input), expected);
        }
    }

    #[test]
    fn test_anthropic_messages_to_engine_with_system() {
        let messages = vec![text_message("user", "Hello")];
        let system = SystemPrompt::Text("Be helpful".to_owned());

        let result = anthropic_messages_to_engine(&messages, Some(&system));
        assert_eq!(result.len(), 2);
        assert_eq!(result.first().map(|m| m.role.as_str()), Some("system"));
        assert_eq!(
            result.first().map(|m| m.content.as_str()),
            Some("Be helpful")
        );
    }

    #[test]
    fn test_anthropic_messages_to_engine_without_system() {
        let messages = vec![text_message("user", "Hello")];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result.first().map(|m| m.role.as_str()), Some("user"));
    }

    #[test]
    fn test_anthropic_messages_to_engine_content_blocks() {
        let messages = vec![blocks_message(
            "user",
            vec![
                ContentBlock::Text {
                    text: "Hello ".to_owned(),
                },
                ContentBlock::Text {
                    text: "World".to_owned(),
                },
            ],
        )];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result.first().map(|m| m.content.as_str()),
            Some("Hello World")
        );
    }

    #[test]
    fn test_anthropic_messages_to_engine_mixed_blocks_filters_non_text() {
        let messages = vec![blocks_message(
            "user",
            vec![
                ContentBlock::Text {
                    text: "Hello".to_owned(),
                },
                ContentBlock::ToolUse {
                    id: "tu_1".to_owned(),
                    name: "get_weather".to_owned(),
                    input: serde_json::json!({}),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tu_1".to_owned(),
                    content: AnthropicContent::Text("72 degrees".to_owned()),
                },
            ],
        )];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result.first().map(|m| m.content.as_str()), Some("Hello"));
    }

    #[test]
    fn test_anthropic_messages_to_engine_empty_messages() {
        let result = anthropic_messages_to_engine(&[], None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_anthropic_messages_to_engine_multiple_messages() {
        let messages = vec![
            text_message("user", "First"),
            text_message("assistant", "Second"),
            text_message("user", "Third"),
        ];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 3);
        assert_eq!(result.first().map(|m| m.content.as_str()), Some("First"));
        assert_eq!(result.get(1).map(|m| m.content.as_str()), Some("Second"));
        assert_eq!(result.get(2).map(|m| m.content.as_str()), Some("Third"));
    }

    #[test]
    fn test_system_as_empty_string() {
        let messages = vec![text_message("user", "Hello")];
        let system = SystemPrompt::Text(String::new());

        let result = anthropic_messages_to_engine(&messages, Some(&system));
        assert_eq!(result.len(), 2);
        assert_eq!(result.first().map(|m| m.role.as_str()), Some("system"));
        assert_eq!(result.first().map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_only_non_text_content_blocks_results_in_empty_content() {
        let messages = vec![blocks_message(
            "assistant",
            vec![
                ContentBlock::ToolUse {
                    id: "tu_1".to_owned(),
                    name: "get_weather".to_owned(),
                    input: serde_json::json!({"city": "NYC"}),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tu_1".to_owned(),
                    content: AnthropicContent::Text("72 degrees".to_owned()),
                },
            ],
        )];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result.first().map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_unicode_content() {
        let messages = vec![text_message(
            "user",
            "Hej! Jag talar svenska. \u{1F600} \u{4F60}\u{597D}",
        )];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert!(
            result
                .first()
                .is_some_and(|m| m.content.contains("svenska"))
        );
    }

    #[test]
    fn test_very_long_content() {
        let long_content = "a".repeat(100_000);
        let messages = vec![text_message("user", &long_content)];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result.first().map(|m| m.content.len()), Some(100_000));
    }

    // -- render_anthropic_markers --

    #[test]
    fn test_render_anthropic_markers_splices_marker_at_image_positions() {
        let messages = vec![blocks_message(
            "user",
            vec![
                ContentBlock::Text {
                    text: "what is ".to_owned(),
                },
                ContentBlock::Image {
                    source: serde_json::json!({"type": "base64"}),
                },
                ContentBlock::Text {
                    text: " in this photo".to_owned(),
                },
            ],
        )];
        let rendered = render_anthropic_markers(&messages, Some("<image>"));
        let engine = anthropic_messages_to_engine(&rendered, None);
        assert_eq!(engine[0].content, "what is <image> in this photo");
    }

    #[test]
    fn test_render_anthropic_markers_defaults_to_image_marker() {
        let messages = vec![blocks_message(
            "user",
            vec![ContentBlock::Image {
                source: serde_json::json!({"type": "base64"}),
            }],
        )];
        let rendered = render_anthropic_markers(&messages, None);
        let engine = anthropic_messages_to_engine(&rendered, None);
        assert_eq!(engine[0].content, "<image>");
    }

    #[test]
    fn test_render_anthropic_markers_nested_tool_result_images() {
        // Each image nested in tool_result content splices one marker (the
        // extractor collects them at the enclosing block's position).
        let messages = vec![blocks_message(
            "user",
            vec![
                ContentBlock::Text {
                    text: "result: ".to_owned(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tu_1".to_owned(),
                    content: AnthropicContent::Blocks(vec![
                        ContentBlock::Image {
                            source: serde_json::json!({"type": "base64"}),
                        },
                        ContentBlock::Image {
                            source: serde_json::json!({"type": "base64"}),
                        },
                    ]),
                },
            ],
        )];
        let rendered = render_anthropic_markers(&messages, Some("<image>"));
        let engine = anthropic_messages_to_engine(&rendered, None);
        assert_eq!(engine[0].content, "result: <image><image>");
    }

    #[test]
    fn test_render_anthropic_markers_passes_plain_text_through() {
        let messages = vec![text_message("user", "hello")];
        let rendered = render_anthropic_markers(&messages, Some("<image>"));
        assert_eq!(rendered[0].content.to_text(), "hello");
    }

    #[test]
    fn test_render_anthropic_markers_keeps_tool_blocks_dropped() {
        let messages = vec![blocks_message(
            "assistant",
            vec![
                ContentBlock::Text {
                    text: "thinking".to_owned(),
                },
                ContentBlock::ToolUse {
                    id: "t1".to_owned(),
                    name: "calc".to_owned(),
                    input: serde_json::json!({}),
                },
            ],
        )];
        let rendered = render_anthropic_markers(&messages, Some("<image>"));
        let engine = anthropic_messages_to_engine(&rendered, None);
        assert_eq!(engine[0].content, "thinking");
    }
}
