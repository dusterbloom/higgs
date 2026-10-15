//! Golden check: higgs renders the real LFM2.5 chat template byte-identically to
//! HF `apply_chat_template`, including reasoning kept on post-user assistant
//! turns and Pythonic tool calls. Fixtures come from `scripts/lfm2_parity.py ref`.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use std::collections::BTreeMap;

use higgs_engine::chat_template::{
    ChatMessage, ChatTemplateRenderer, normalize_tool_call_for_template,
};
use serde_json::Value;

#[test]
#[ignore = "needs HIGGS_LFM2_DIR (model) and HIGGS_LFM2_PARITY (lfm2_parity.py ref output)"]
fn renders_like_hf_apply_chat_template() {
    let model = std::env::var("HIGGS_LFM2_DIR").unwrap();
    let work = std::path::PathBuf::from(std::env::var("HIGGS_LFM2_PARITY").unwrap());
    let read = |f: &str| -> BTreeMap<String, Value> {
        serde_json::from_str(&std::fs::read_to_string(work.join(f)).unwrap()).unwrap()
    };
    let (conversations, expected) = (read("conversations.json"), read("prompts.json"));
    let renderer = ChatTemplateRenderer::try_from_model_dir(model.as_ref())
        .unwrap()
        .unwrap();

    for (name, conv) in &conversations {
        // Same mapping as the OpenAI chat route's `convert_messages`.
        let messages: Vec<ChatMessage> = conv["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| ChatMessage {
                role: m["role"].as_str().unwrap().to_owned(),
                content: m["content"].as_str().unwrap_or_default().to_owned(),
                reasoning_content: m["reasoning_content"].as_str().map(str::to_owned),
                tool_calls: m["tool_calls"].as_array().map(|calls| {
                    calls
                        .iter()
                        .cloned()
                        .map(|mut tc| {
                            normalize_tool_call_for_template(&mut tc);
                            tc
                        })
                        .collect()
                }),
            })
            .collect();
        let tools = conv["tools"].as_array().map(Vec::as_slice);
        let rendered = renderer
            .apply_with_thinking(&messages, tools, true, true)
            .unwrap();
        assert_eq!(rendered, expected[name].as_str().unwrap(), "{name}");
    }
}
