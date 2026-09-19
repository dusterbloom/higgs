use axum::{
    Json,
    extract::{Path, State},
};
use serde::Serialize;

use crate::error::ServerError;
use crate::state::SharedState;

#[derive(Debug, Serialize)]
pub struct DropRetainedSessionResponse {
    pub session_id: u64,
    pub dropped: usize,
}

/// Drop retained per-session KV for every loaded local engine.
///
/// This is a logical-session reset hook. It deliberately leaves exact
/// content-addressed radix/disk prefix caches alive.
pub async fn drop_retained_session(
    State(state): State<SharedState>,
    Path(session_id): Path<u64>,
) -> Result<Json<DropRetainedSessionResponse>, ServerError> {
    let engines = state.router.local_engines();

    let dropped = tokio::task::spawn_blocking(move || {
        engines
            .into_iter()
            .filter(|engine| engine.drop_retained_session(session_id))
            .count()
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?;

    tracing::info!(
        session_id,
        dropped_engines = dropped,
        "retained session drop requested by API"
    );

    Ok(Json(DropRetainedSessionResponse {
        session_id,
        dropped,
    }))
}
