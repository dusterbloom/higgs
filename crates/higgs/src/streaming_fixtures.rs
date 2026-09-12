//! Scripted worker output for route/SDK tests. Never linked into production.
use higgs_engine::{engine::StreamingOutput, error::EngineError};
use tokio::sync::mpsc::Sender;

pub fn emit(
    name: &str,
    sender: &Sender<StreamingOutput>,
    prompt_len: usize,
) -> Result<(), EngineError> {
    let mut chunks: Vec<String> = Vec::new();
    let trailing_leak = name.ends_with("required-leaky");
    let required = name.ends_with("required-json") || trailing_leak;
    if !required {
        chunks.push("Preparing.\n".to_owned());
    }
    if name.ends_with("json") || trailing_leak {
        if required {
            chunks.extend(
                [
                    "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"",
                    "Rome",
                    "\"}}</tool_call>",
                ]
                .map(str::to_owned),
            );
        } else {
            chunks.extend(
                [
                    "<tool_call>{\"name\":\"write\",\"arguments\":{\"content\":\"",
                    "hello\\n",
                    "world",
                    "\"}}</tool_call>",
                ]
                .map(str::to_owned),
            );
        }
    } else if name.ends_with("minicpm") {
        chunks.extend(
            [
                "<function name=\"write\"><param name=\"content\"><![CDATA[",
                "hello\n",
                "world",
                "]]></param></function>",
            ]
            .map(str::to_owned),
        );
    } else {
        chunks.push("<tool_call><function=write><parameter=content>".to_owned());
        if name.ends_with("long") || name.ends_with("idle") {
            chunks.extend((0..80).map(|_| "chunk ".repeat(16)));
        } else {
            chunks.extend(["hello\n", "world"].map(str::to_owned));
        }
        if !name.ends_with("malformed") && !name.ends_with("capacity") {
            chunks.push("</parameter></function></tool_call>".to_owned());
        }
    }
    if trailing_leak {
        chunks.push("invalid trailing prose".to_owned());
    }
    let len = chunks.len();
    for (index, text) in chunks.into_iter().enumerate() {
        if sender.is_closed() {
            return Err(EngineError::Cancelled);
        }
        if name.ends_with("idle") && index == 2 {
            // Bounded cooperative idle fixture: continue observing disconnects.
            for _ in 0..200 {
                if sender.is_closed() {
                    return Err(EngineError::Cancelled);
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
        }
        let last = index + 1 == len;
        sender
            .blocking_send(StreamingOutput {
                new_text: text,
                finished: last && !name.ends_with("capacity"),
                finish_reason: (last && !name.ends_with("capacity")).then(|| "stop".to_owned()),
                prompt_tokens: u32::try_from(prompt_len).unwrap_or(u32::MAX),
                completion_tokens: u32::try_from(index + 1).unwrap_or(u32::MAX),
                token_logprob: None,
                prefill_progress: None,
            })
            .map_err(|_| EngineError::Cancelled)?;
        if name.ends_with("long") {
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    }
    if name.ends_with("capacity") {
        return Err(EngineError::CapacityInterrupted {
            boot_id: "boot-script".to_owned(),
            generation: 4,
        });
    }
    Ok(())
}
