//! POST /v1/feedback — explicit user feedback on model responses.

use axum::{Json, extract::State, response::IntoResponse};
use serde::{Deserialize, Serialize};

use crate::{error::ServerError, state::SharedState};

#[derive(Debug, Deserialize)]
pub struct FeedbackRequest {
    /// The request_id from the chat completion response (e.g. "chatcmpl-...").
    pub request_id: String,
    /// Signal type: "positive", "negative", or "correction".
    pub signal: String,
    /// For "correction": the text the user actually wanted.
    pub correction: Option<String>,
    /// If true, pin this entry (exempt from eviction).
    #[serde(default)]
    pub pin: bool,
}

#[derive(Debug, Serialize)]
pub struct FeedbackResponse {
    pub request_id: String,
    pub status: String,
    pub reward: f32,
}

pub async fn feedback(
    State(state): State<SharedState>,
    Json(req): Json<FeedbackRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let memory = state.memory.as_ref().ok_or_else(|| {
        ServerError::BadRequest("Adaptive memory is not enabled".to_owned())
    })?;

    let reward_delta = match req.signal.as_str() {
        "positive" => 1.0_f32,
        "negative" => -1.0,
        "correction" => {
            // For corrections, replace the completion tokens and give high reward
            if let Some(ref correction_text) = req.correction {
                // Tokenize the correction using the first available engine
                let resolved = state
                    .router
                    .resolve_first_local()
                    .await
                    .ok_or_else(|| ServerError::BadRequest("No local engine available".to_owned()))?;

                let tokenizer = resolved.tokenizer();
                let encoding = tokenizer
                    .encode(correction_text.as_str(), false)
                    .map_err(|e| ServerError::InternalError(format!("Tokenization failed: {e}")))?;
                let correction_tokens: Vec<u32> = encoding.get_ids().to_vec();

                // Get the original entry's prompt tokens
                let mut buf = memory.replay_buffer.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(entry) = buf.get(&req.request_id) {
                    let prompt_len = entry.prompt_len;
                    let mut new_tokens = entry.tokens[..prompt_len].to_vec();
                    new_tokens.extend_from_slice(&correction_tokens);
                    buf.replace_completion(&req.request_id, new_tokens, prompt_len);
                }
            }
            1.5
        }
        other => {
            return Err(ServerError::BadRequest(format!(
                "Unknown signal type: {other}. Expected: positive, negative, correction"
            )));
        }
    };

    memory.apply_feedback(&[(req.request_id.clone(), reward_delta)]);

    if req.pin {
        memory.replay_buffer.lock().unwrap_or_else(|e| e.into_inner()).pin(&req.request_id);
    }

    let reward = memory
        .replay_buffer
        .lock().unwrap_or_else(|e| e.into_inner())
        .get(&req.request_id)
        .map_or(reward_delta, |e| e.reward);

    Ok(Json(FeedbackResponse {
        request_id: req.request_id,
        status: "ok".to_owned(),
        reward,
    }))
}
