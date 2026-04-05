use axum::{Json, extract::State, response::IntoResponse};
use serde::{Deserialize, Serialize};

use crate::{error::ServerError, state::SharedState};

#[derive(Debug, Deserialize)]
pub struct TrainRequest {
    /// Model name (must resolve to a local Qwen3Next engine).
    pub model: String,
    /// Full token sequence (prompt + completion).
    pub tokens: Vec<u32>,
    /// Number of prompt tokens (loss computed only on completion).
    pub prompt_len: usize,
    /// EGGROLL hyperparameters.
    #[serde(default = "default_sigma")]
    pub sigma: f32,
    #[serde(default = "default_lr")]
    pub lr: f32,
    #[serde(default = "default_rank")]
    pub rank: usize,
    #[serde(default = "default_population")]
    pub population: usize,
    #[serde(default = "default_total_steps")]
    pub total_steps: usize,
    #[serde(default = "default_merge_interval")]
    pub merge_interval: usize,
    /// Max delta norm as fraction of base weight norm (0 = no clipping).
    #[serde(default = "default_clip_ratio")]
    pub clip_ratio: f32,
    /// Per-step decay applied to accumulated deltas (0 = no decay).
    #[serde(default = "default_delta_decay")]
    pub delta_decay: f32,
}

fn default_sigma() -> f32 { 0.001 }
fn default_lr() -> f32 { 0.0005 }
fn default_rank() -> usize { 4 }
fn default_population() -> usize { 16 }
fn default_total_steps() -> usize { 50 }
fn default_merge_interval() -> usize { 0 }
fn default_clip_ratio() -> f32 { 0.05 }
fn default_delta_decay() -> f32 { 0.001 }

#[derive(Debug, Serialize)]
pub struct TrainResponse {
    pub model: String,
    pub steps: usize,
    pub losses: Vec<f32>,
    pub final_loss: f32,
}

pub async fn train(
    State(state): State<SharedState>,
    Json(req): Json<TrainRequest>,
) -> Result<impl IntoResponse, ServerError> {
    // Input validation — catch misconfig before locking the model.
    if req.tokens.is_empty() {
        return Err(ServerError::BadRequest("tokens must be non-empty".into()));
    }
    if req.prompt_len == 0 {
        return Err(ServerError::BadRequest("prompt_len must be >= 1".into()));
    }
    if req.prompt_len >= req.tokens.len() {
        return Err(ServerError::BadRequest(format!(
            "prompt_len ({}) must be < tokens.len() ({})",
            req.prompt_len,
            req.tokens.len()
        )));
    }
    if req.sigma <= 0.0 {
        return Err(ServerError::BadRequest("sigma must be > 0".into()));
    }
    if req.lr <= 0.0 {
        return Err(ServerError::BadRequest("lr must be > 0".into()));
    }
    if req.rank == 0 {
        return Err(ServerError::BadRequest("rank must be >= 1".into()));
    }
    if req.population == 0 {
        return Err(ServerError::BadRequest("population must be >= 1".into()));
    }
    if req.total_steps == 0 {
        return Err(ServerError::BadRequest("total_steps must be >= 1".into()));
    }

    let resolved = state
        .router
        .resolve(&req.model, None)
        .await
        .map_err(ServerError::ModelNotFound)?;

    match resolved {
        crate::router::ResolvedRoute::Higgs {
            engine,
            model_name,
            ..
        } => {
            let config = higgs_models::diffusion_eggroll::EggrollConfig {
                sigma: req.sigma,
                lr: req.lr,
                rank: req.rank,
                population: req.population,
                total_steps: req.total_steps,
                warmup_steps: req.total_steps / 10,
                min_lr_frac: 0.1,
                log_interval: 10,
                base_seed: 42,
                clip_ratio: req.clip_ratio,
                delta_decay: req.delta_decay,
            };

            // Run training synchronously (blocks the model mutex).
            let losses = tokio::task::spawn_blocking(move || {
                engine.train_eggroll(config, &req.tokens, req.prompt_len, req.merge_interval)
            })
            .await
            .map_err(|e| ServerError::InternalError(format!("Training task panicked: {e}")))?
            .map_err(|e| ServerError::InternalError(format!("EGGROLL training failed: {e}")))?;

            let final_loss = losses.last().copied().unwrap_or(0.0);
            Ok(Json(TrainResponse {
                model: model_name,
                steps: losses.len(),
                losses,
                final_loss,
            }))
        }
        crate::router::ResolvedRoute::Remote { .. } => Err(ServerError::BadRequest(
            "EGGROLL training only supported on local models".to_owned(),
        )),
    }
}
