//! LFM3 chat-template and streamed tool-call compatibility checks.
//!
//! Golden prompts are produced by the official tokenizer through
//! `scripts/lfm3_parity.py fixtures`, rather than by this Rust renderer.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::tests_outside_test_module
)]

use std::collections::BTreeMap;

use higgs_engine::{
    chat_template::{ChatMessage, ChatTemplateRenderer, normalize_tool_call_for_template},
    tool_parser::{StreamingToolCallTracker, ToolSchema},
};
use serde_json::Value;

#[test]
#[ignore = "needs HIGGS_LFM3_DIR and HIGGS_LFM3_PARITY (lfm3_parity.py fixtures output)"]
fn renders_byte_identically_to_hf_apply_chat_template() {
    let model_dir = std::env::var("HIGGS_LFM3_DIR").unwrap();
    let fixture_dir = std::path::PathBuf::from(std::env::var("HIGGS_LFM3_PARITY").unwrap());
    let read_map = |file: &str| -> BTreeMap<String, Value> {
        serde_json::from_str(&std::fs::read_to_string(fixture_dir.join(file)).unwrap()).unwrap()
    };
    let conversations = read_map("conversations.json");
    let expected = read_map("prompts.json");
    let renderer = ChatTemplateRenderer::try_from_model_dir(model_dir.as_ref())
        .unwrap()
        .unwrap();

    for (name, conversation) in &conversations {
        // This mirrors the OpenAI route's conversion boundary.  In particular,
        // tool-call arguments arrive as JSON strings at that boundary and are
        // normalized back to mappings for the official template.
        let messages: Vec<ChatMessage> = conversation["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|message| ChatMessage {
                role: message["role"].as_str().unwrap().to_owned(),
                content: message["content"].as_str().unwrap_or_default().to_owned(),
                reasoning_content: message["reasoning_content"].as_str().map(str::to_owned),
                tool_calls: message["tool_calls"].as_array().map(|calls| {
                    calls
                        .iter()
                        .cloned()
                        .map(|mut call| {
                            normalize_tool_call_for_template(&mut call);
                            call
                        })
                        .collect()
                }),
            })
            .collect();
        let tools = conversation["tools"].as_array().map(Vec::as_slice);
        let rendered = renderer
            .apply_with_thinking(&messages, tools, true, true)
            .unwrap();
        assert_eq!(rendered, expected[name].as_str().unwrap(), "{name}");
    }
}

#[test]
fn streamed_lfm3_xml_tool_call_survives_token_boundaries() {
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }
        }
    })];
    let schema = ToolSchema::from_tools(Some(tools.as_slice()));
    let mut tracker = StreamingToolCallTracker::new(true, schema);
    let chunks = [
        "<tool_",
        "call>\n<function=get_",
        "weather>\n<parameter=ci",
        "ty>\nRome\n</parameter>\n</function>\n</tool_",
        "call>",
    ];
    let mut visible = String::new();
    let mut calls = Vec::new();
    for chunk in chunks {
        let output = tracker.process(chunk);
        visible.push_str(&output.visible);
        calls.extend(output.new_tool_calls);
    }
    let tail = tracker.flush();
    visible.push_str(&tail.visible);
    calls.extend(tail.new_tool_calls);

    assert!(
        visible.is_empty(),
        "tool markup leaked into content: {visible:?}"
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments, serde_json::json!({ "city": "Rome" }));
    assert_eq!(tracker.completed_count(), 1);
}
