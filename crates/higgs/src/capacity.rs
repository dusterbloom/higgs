use serde::{Deserialize, Serialize};

pub const CAPACITY_SCHEMA_VERSION: u32 = 1;
pub const CAPACITY_RETRY_AFTER_MS: u64 = 5_000;

fn deserialize_schema_version<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let schema_version = u32::deserialize(deserializer)?;
    if schema_version != CAPACITY_SCHEMA_VERSION {
        return Err(<D::Error as serde::de::Error>::custom(format_args!(
            "unsupported capacity schemaVersion {schema_version}; expected {CAPACITY_SCHEMA_VERSION}"
        )));
    }
    Ok(schema_version)
}

/// Whether the requested model can currently accept its minimum working request.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityAvailability {
    Available,
    Unavailable,
}

/// Process memory pressure used to derive the published capacity envelope.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPressure {
    Normal,
    Constrained,
    Critical,
}

/// Evidence backing the current capacity envelope.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityBasis {
    Conservative,
    Learned,
}

/// Versioned capacity advertised for one model by this Higgs process.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CapacitySnapshot {
    #[serde(deserialize_with = "deserialize_schema_version")]
    pub schema_version: u32,
    pub model: String,
    pub model_fingerprint: String,
    pub boot_id: String,
    pub generation: u64,
    pub availability: CapacityAvailability,
    pub pressure: MemoryPressure,
    pub safe_total_tokens: u64,
    pub recommended_output_tokens: u64,
    pub max_prompt_tokens: u64,
    pub retained_session_tokens: u64,
    pub retained_bytes: u64,
    pub prefix_cache_bytes: u64,
    pub basis: CapacityBasis,
}

impl CapacitySnapshot {
    /// Revisions are process-local: a restarted server may reuse a generation number.
    #[must_use]
    pub fn is_same_revision(&self, other: &Self) -> bool {
        self.boot_id == other.boot_id && self.generation == other.generation
    }
}

/// OpenAI-compatible outer error object shared by HTTP errors and terminal SSE errors.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CapacityErrorEnvelope<T> {
    error: T,
}

impl<T> CapacityErrorEnvelope<T> {
    #[must_use]
    pub fn new(error: T) -> Self {
        Self { error }
    }
}

/// Request-specific limits returned before inference when a request is too large.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityExceededError {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    safe_prompt_tokens: u64,
    safe_total_tokens: u64,
    boot_id: String,
    generation: u64,
}

impl CapacityExceededError {
    #[must_use]
    pub fn new(
        safe_prompt_tokens: u64,
        safe_total_tokens: u64,
        boot_id: String,
        generation: u64,
    ) -> Self {
        Self {
            error_type: "higgs_capacity_exceeded",
            code: "compact_and_retry",
            safe_prompt_tokens,
            safe_total_tokens,
            boot_id,
            generation,
        }
    }
}

/// Temporary inability to fit even the minimum working request.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityUnavailableError {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    boot_id: String,
    generation: u64,
    retry_after_ms: u64,
}

impl CapacityUnavailableError {
    #[must_use]
    pub fn new(boot_id: String, generation: u64) -> Self {
        Self {
            error_type: "higgs_capacity_unavailable",
            code: "capacity_unavailable",
            boot_id,
            generation,
            retry_after_ms: CAPACITY_RETRY_AFTER_MS,
        }
    }
}

/// Terminal stream event emitted when pressure interrupts active generation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityInterruptedError {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    boot_id: String,
    generation: u64,
    partial_output_tokens: u64,
}

impl CapacityInterruptedError {
    #[must_use]
    pub fn new(boot_id: String, generation: u64, partial_output_tokens: u64) -> Self {
        Self {
            error_type: "higgs_capacity_interrupted",
            code: "capacity_interrupted",
            boot_id,
            generation,
            partial_output_tokens,
        }
    }
}

/// Typed unknown-model response for the capacity extension route.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapacityModelNotFoundError {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
    model: String,
}

impl CapacityModelNotFoundError {
    #[must_use]
    pub fn new(model: String) -> Self {
        Self {
            error_type: "higgs_capacity_model_not_found",
            code: "model_not_found",
            model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MODEL: &str = "escha-35b-a3b";
    const FINGERPRINT: &str =
        "sha256:7b2f5c8ae91a5b1d83f1364c2023e5e53b5530d0461a4193cf9bd37f4e70d821";
    const BOOT_ID: &str = "01993654-8af2-7b31-a420-c52ebc349287";

    fn available_snapshot() -> CapacitySnapshot {
        CapacitySnapshot {
            schema_version: 1,
            model: MODEL.to_owned(),
            model_fingerprint: FINGERPRINT.to_owned(),
            boot_id: BOOT_ID.to_owned(),
            generation: 7,
            availability: CapacityAvailability::Available,
            pressure: MemoryPressure::Normal,
            safe_total_tokens: 53_248,
            recommended_output_tokens: 4_096,
            max_prompt_tokens: 49_152,
            retained_session_tokens: 49_152,
            retained_bytes: 2_147_483_648,
            prefix_cache_bytes: 1_073_741_824,
            basis: CapacityBasis::Learned,
        }
    }

    #[test]
    fn capacity_snapshot_matches_schema_v1_json() {
        assert_eq!(
            serde_json::to_value(available_snapshot()).unwrap(),
            json!({
                "schemaVersion": 1,
                "model": MODEL,
                "modelFingerprint": FINGERPRINT,
                "bootId": BOOT_ID,
                "generation": 7,
                "availability": "available",
                "pressure": "normal",
                "safeTotalTokens": 53_248,
                "recommendedOutputTokens": 4_096,
                "maxPromptTokens": 49_152,
                "retainedSessionTokens": 49_152,
                "retainedBytes": 2_147_483_648_u64,
                "prefixCacheBytes": 1_073_741_824_u64,
                "basis": "learned"
            })
        );
    }

    #[test]
    fn known_but_unloaded_snapshot_is_unavailable_with_zero_token_fields() {
        let snapshot = CapacitySnapshot {
            availability: CapacityAvailability::Unavailable,
            safe_total_tokens: 0,
            recommended_output_tokens: 0,
            max_prompt_tokens: 0,
            retained_session_tokens: 0,
            retained_bytes: 0,
            prefix_cache_bytes: 0,
            basis: CapacityBasis::Conservative,
            ..available_snapshot()
        };

        let value = serde_json::to_value(snapshot).unwrap();
        assert_eq!(
            value,
            json!({
                "schemaVersion": 1,
                "model": MODEL,
                "modelFingerprint": FINGERPRINT,
                "bootId": BOOT_ID,
                "generation": 7,
                "availability": "unavailable",
                "pressure": "normal",
                "safeTotalTokens": 0,
                "recommendedOutputTokens": 0,
                "maxPromptTokens": 0,
                "retainedSessionTokens": 0,
                "retainedBytes": 0,
                "prefixCacheBytes": 0,
                "basis": "conservative"
            })
        );
    }

    #[test]
    fn capacity_enums_reject_unknown_values() {
        assert!(serde_json::from_str::<CapacityAvailability>("\"loading\"").is_err());
        assert!(serde_json::from_str::<MemoryPressure>("\"warning\"").is_err());
        assert!(serde_json::from_str::<CapacityBasis>("\"measured\"").is_err());
    }

    #[test]
    fn capacity_snapshot_rejects_non_v1_schema_at_deserialization() {
        let mut value = serde_json::to_value(available_snapshot()).unwrap();
        value["schemaVersion"] = json!(2);

        let error = serde_json::from_value::<CapacitySnapshot>(value).unwrap_err();
        assert_eq!(
            error.to_string(),
            "unsupported capacity schemaVersion 2; expected 1"
        );
    }

    #[test]
    fn generation_is_comparable_only_within_one_boot() {
        let current = available_snapshot();
        let same = available_snapshot();
        let restarted = CapacitySnapshot {
            boot_id: "01993654-8af2-7b31-a420-c52ebc349288".to_owned(),
            ..available_snapshot()
        };
        let advanced = CapacitySnapshot {
            generation: 8,
            ..available_snapshot()
        };

        assert!(current.is_same_revision(&same));
        assert!(!current.is_same_revision(&restarted));
        assert!(!current.is_same_revision(&advanced));
    }

    #[test]
    fn capacity_exceeded_matches_openai_shaped_413_body() {
        let body = CapacityErrorEnvelope::new(CapacityExceededError::new(
            36_864,
            40_960,
            BOOT_ID.to_owned(),
            8,
        ));

        assert_eq!(
            serde_json::to_value(body).unwrap(),
            json!({
                "error": {
                    "type": "higgs_capacity_exceeded",
                    "code": "compact_and_retry",
                    "safePromptTokens": 36_864,
                    "safeTotalTokens": 40_960,
                    "bootId": BOOT_ID,
                    "generation": 8
                }
            })
        );
    }

    #[test]
    fn capacity_unavailable_matches_openai_shaped_503_body() {
        let body = CapacityErrorEnvelope::new(CapacityUnavailableError::new(BOOT_ID.to_owned(), 9));

        assert_eq!(
            serde_json::to_value(body).unwrap(),
            json!({
                "error": {
                    "type": "higgs_capacity_unavailable",
                    "code": "capacity_unavailable",
                    "bootId": BOOT_ID,
                    "generation": 9,
                    "retryAfterMs": 5000
                }
            })
        );
    }

    #[test]
    fn capacity_interrupted_matches_terminal_sse_body() {
        let body =
            CapacityErrorEnvelope::new(CapacityInterruptedError::new(BOOT_ID.to_owned(), 10, 317));

        assert_eq!(
            serde_json::to_value(body).unwrap(),
            json!({
                "error": {
                    "type": "higgs_capacity_interrupted",
                    "code": "capacity_interrupted",
                    "bootId": BOOT_ID,
                    "generation": 10,
                    "partialOutputTokens": 317
                }
            })
        );
    }

    #[tokio::test]
    async fn typed_unknown_model_is_distinct_from_axum_legacy_route_absence() {
        use axum::{
            Router,
            body::Body,
            http::{Request, StatusCode},
        };
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let typed = serde_json::to_value(CapacityErrorEnvelope::new(
            CapacityModelNotFoundError::new("missing-model".to_owned()),
        ))
        .unwrap();
        let legacy = Router::new()
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity?model=missing-model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(legacy.status(), StatusCode::NOT_FOUND);
        assert!(legacy.headers().get("content-type").is_none());
        let legacy_body = legacy.into_body().collect().await.unwrap().to_bytes();
        assert!(legacy_body.is_empty());

        assert_eq!(
            typed,
            json!({
                "error": {
                    "type": "higgs_capacity_model_not_found",
                    "code": "model_not_found",
                    "model": "missing-model"
                }
            })
        );
        assert_ne!(typed.to_string().as_bytes(), legacy_body.as_ref());
    }
}
