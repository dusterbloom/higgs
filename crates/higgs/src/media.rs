//! Media extraction and validation for vision requests.
//!
//! One shared pipeline for `OpenAI` parts and `Anthropic` blocks. Produces
//! position-preserving [`MediaItem`]s in client order; the engine renders
//! family markers at those positions.

use base64::Engine as _;
use reqwest::Client;

use crate::error::ServerError;
use crate::types::anthropic::{AnthropicContent, AnthropicMessage, ContentBlock, SystemPrompt};
use crate::types::openai::{ChatCompletionMessage, ContentPart, MessageContent};

/// Default decoded-byte cap per image (20 MiB).
pub const MAX_IMAGE_BYTES_DEFAULT: usize = 20 << 20;
/// Default HTTP fetch timeout for remote image URLs (seconds).
pub const IMAGE_FETCH_TIMEOUT_DEFAULT: f64 = 10.0;
/// Default long-edge pixel cap applied before family preprocessing.
pub const MAX_IMAGE_DIMENSION_DEFAULT: u32 = 4096;

/// A validated image from a request, in client order.
#[derive(Debug, Clone)]
pub struct MediaItem {
    pub position: usize,
    pub message_index: usize,
    pub bytes: Vec<u8>,
    pub media_type: String,
    pub detail: higgs_models::vision::ImageDetail,
    pub max_dims: Option<(u32, u32)>,
}

impl From<MediaItem> for higgs_models::vision::ImageInput {
    fn from(m: MediaItem) -> Self {
        Self {
            position: m.position,
            message_index: m.message_index,
            bytes: m.bytes,
            media_type: m.media_type,
            detail: m.detail,
            max_dims: m.max_dims,
        }
    }
}

/// Extracts and validates media items from API requests.
pub struct MediaExtractor {
    pub max_image_bytes: usize,
    pub fetch_timeout: std::time::Duration,
    pub max_image_dimension: u32,
    pub http_client: Client,
}

impl MediaExtractor {
    /// Build an extractor with an HTTP client whose timeout matches
    /// `fetch_timeout_secs` (floored at 0.1s so `from_secs_f64` never panics).
    pub fn new(
        max_image_bytes: usize,
        fetch_timeout_secs: f64,
        max_image_dimension: u32,
    ) -> Result<Self, ServerError> {
        let fetch_timeout = std::time::Duration::from_secs_f64(fetch_timeout_secs.max(0.1));
        let http_client = Client::builder()
            .timeout(fetch_timeout)
            .build()
            .map_err(|e| ServerError::InternalError(format!("HTTP client build failed: {e}")))?;
        Ok(Self {
            max_image_bytes,
            fetch_timeout,
            max_image_dimension,
            http_client,
        })
    }

    /// Extract media from OpenAI-style chat messages, preserving part order.
    ///
    /// `position` is the part index within the message; text parts advance the
    /// index but produce no item.
    pub async fn extract_openai(
        &self,
        messages: &[ChatCompletionMessage],
    ) -> Result<Vec<MediaItem>, ServerError> {
        let mut items = Vec::new();
        for (mi, msg) in messages.iter().enumerate() {
            let Some(content) = &msg.content else {
                continue;
            };
            match content {
                MessageContent::Text(_) => {}
                MessageContent::Parts(parts) => {
                    for (pi, part) in parts.iter().enumerate() {
                        if let ContentPart::ImageUrl { image_url } = part {
                            let (media_type, bytes) =
                                self.resolve_url(&image_url.url, pi, mi).await?;
                            self.validate_media_type(&media_type).map_err(|_| {
                                ServerError::BadRequest(format!(
                                    "unsupported media type: {media_type} at part {pi} \
                                     of message {mi}"
                                ))
                            })?;
                            items.push(MediaItem {
                                position: pi,
                                message_index: mi,
                                bytes,
                                media_type,
                                detail: image_url.detail.unwrap_or_default(),
                                max_dims: None,
                            });
                        }
                    }
                }
            }
        }
        Ok(items)
    }

    /// Extract media from Anthropic-style messages and blocks.
    ///
    /// The `system` prompt cannot carry images: `SystemBlock` only accepts
    /// text, so a system image fails request deserialization with a 400 before
    /// extraction runs. Images nested in `tool_result` content are collected
    /// with the block index of their enclosing `tool_result`.
    pub async fn extract_anthropic(
        &self,
        messages: &[AnthropicMessage],
        _system: Option<&SystemPrompt>,
    ) -> Result<Vec<MediaItem>, ServerError> {
        let mut items = Vec::new();
        for (mi, msg) in messages.iter().enumerate() {
            let AnthropicContent::Blocks(blocks) = &msg.content else {
                continue;
            };
            for (bi, block) in blocks.iter().enumerate() {
                match block {
                    ContentBlock::Image { source } => {
                        self.push_image(source, bi, mi, &mut items).await?;
                    }
                    ContentBlock::ToolResult { content, .. } => {
                        let AnthropicContent::Blocks(inner) = content else {
                            continue;
                        };
                        for inner_block in inner {
                            let ContentBlock::Image { source } = inner_block else {
                                continue;
                            };
                            self.push_image(source, bi, mi, &mut items).await?;
                        }
                    }
                    ContentBlock::Text { .. }
                    | ContentBlock::ToolUse { .. }
                    | ContentBlock::Thinking { .. }
                    | ContentBlock::RedactedThinking { .. }
                    | ContentBlock::Document { .. }
                    | ContentBlock::ServerToolUse { .. }
                    | ContentBlock::WebSearchToolResult { .. }
                    | ContentBlock::CodeExecutionToolResult { .. }
                    | ContentBlock::Other => {}
                }
            }
        }
        Ok(items)
    }

    /// Resolve a `data:` URI or `http(s)://` URL to decoded bytes and its
    /// media type. Applies [`Self::max_image_bytes`] after decoding.
    pub async fn resolve_url(
        &self,
        url: &str,
        position: usize,
        message_index: usize,
    ) -> Result<(String, Vec<u8>), ServerError> {
        if let Some(data) = url.strip_prefix("data:") {
            let media_type = data.split(';').next().unwrap_or("image/png").to_owned();
            let b64 = data.split_once(";base64,").map(|(_, b)| b).ok_or_else(|| {
                ServerError::BadRequest(format!(
                    "malformed data URI at part {position} of message {message_index}"
                ))
            })?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| {
                    ServerError::BadRequest(format!(
                        "invalid base64 image at part {position} of message {message_index}: {e}"
                    ))
                })?;
            self.check_size(&bytes, position, message_index)?;
            Ok((media_type, bytes))
        } else if url.starts_with("http://") || url.starts_with("https://") {
            // Non-2xx responses (404, 5xx, ...) are fetch failures, not images.
            let resp = self
                .http_client
                .get(url)
                .send()
                .await
                .and_then(reqwest::Response::error_for_status)
                .map_err(|e| {
                    ServerError::BadRequest(format!(
                        "failed to fetch image URL at part {position} of message {message_index}: {e}"
                    ))
                })?;
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("image/png")
                .to_owned();
            let bytes = resp.bytes().await.map_err(|e| {
                ServerError::BadRequest(format!(
                    "failed to read image response at part {position} of message {message_index}: {e}"
                ))
            })?;
            self.check_size(&bytes, position, message_index)?;
            Ok((content_type, bytes.to_vec()))
        } else {
            Err(ServerError::BadRequest(format!(
                "unsupported image URL scheme at part {position} of message {message_index}"
            )))
        }
    }

    /// Resolve a `data:` URI or `http(s)://` URL to decoded bytes, discarding
    /// the media type.
    pub async fn resolve_bytes(
        &self,
        url: &str,
        position: usize,
        message_index: usize,
    ) -> Result<Vec<u8>, ServerError> {
        let (_, bytes) = self.resolve_url(url, position, message_index).await?;
        Ok(bytes)
    }

    /// Validate a media type string against the shared supported-image set.
    ///
    /// Kept as a `&self` method per the extraction interface even though the
    /// supported set is currently global; a config-driven allow-list may use
    /// `self` later.
    #[allow(clippy::unused_self)]
    fn validate_media_type(&self, media_type: &str) -> Result<(), ServerError> {
        let base = media_type.split(';').next().unwrap_or(media_type).trim();
        if higgs_models::vision::is_supported_image_media_type(base) {
            Ok(())
        } else {
            Err(ServerError::BadRequest(format!(
                "unsupported media type: {media_type}"
            )))
        }
    }

    /// Validate one Anthropic `image` source object and push its [`MediaItem`].
    ///
    /// Handles `base64` (inline `data`) and `url` sources; `max_width` /
    /// `max_height` become `max_dims` when both are present.
    async fn push_image(
        &self,
        source: &serde_json::Value,
        position: usize,
        message_index: usize,
        items: &mut Vec<MediaItem>,
    ) -> Result<(), ServerError> {
        let obj = source.as_object().ok_or_else(|| {
            ServerError::BadRequest(format!(
                "image source at block {position} of message {message_index} is not an object"
            ))
        })?;
        let media_type = obj
            .get("media_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("image/png")
            .to_owned();
        self.validate_media_type(&media_type).map_err(|_| {
            ServerError::BadRequest(format!(
                "unsupported media type: {media_type} at block {position} of message \
                 {message_index}"
            ))
        })?;
        let max_dims = match (
            obj.get("max_width").and_then(serde_json::Value::as_u64),
            obj.get("max_height").and_then(serde_json::Value::as_u64),
        ) {
            (Some(w), Some(h)) => Some((
                u32::try_from(w).unwrap_or(u32::MAX),
                u32::try_from(h).unwrap_or(u32::MAX),
            )),
            _ => None,
        };
        let src_type = obj.get("type").and_then(serde_json::Value::as_str);
        let data_field = obj.get("data").and_then(serde_json::Value::as_str);
        let url_field = obj.get("url").and_then(serde_json::Value::as_str);
        let bytes = match (src_type, data_field, url_field) {
            (Some("base64"), Some(b64), _) => {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| {
                        ServerError::BadRequest(format!(
                            "invalid base64 image at block {position} of message \
                             {message_index}: {e}"
                        ))
                    })?;
                self.check_size(&bytes, position, message_index)?;
                bytes
            }
            (Some("url"), _, Some(image_url)) => {
                let (_, bytes) = self.resolve_url(image_url, position, message_index).await?;
                bytes
            }
            _ => {
                return Err(ServerError::BadRequest(format!(
                    "unsupported image source type at block {position} of message \
                     {message_index}"
                )));
            }
        };
        items.push(MediaItem {
            position,
            message_index,
            bytes,
            media_type,
            detail: higgs_models::vision::ImageDetail::Auto,
            max_dims,
        });
        Ok(())
    }

    /// Reject images over the configured decoded-byte cap.
    fn check_size(
        &self,
        bytes: &[u8],
        position: usize,
        message_index: usize,
    ) -> Result<(), ServerError> {
        if bytes.len() > self.max_image_bytes {
            return Err(ServerError::BadRequest(format!(
                "image at part {position} of message {message_index} is {} bytes, exceeding the {}-byte cap",
                bytes.len(),
                self.max_image_bytes
            )));
        }
        Ok(())
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::ImageUrl;

    fn extractor() -> MediaExtractor {
        MediaExtractor {
            max_image_bytes: 1 << 20,
            fetch_timeout: std::time::Duration::from_secs(1),
            max_image_dimension: 4096,
            http_client: reqwest::Client::new(),
        }
    }

    fn base64_png() -> &'static str {
        // 1x1 red PNG
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    }

    fn base64_png_url() -> String {
        format!("data:image/png;base64,{}", base64_png())
    }

    #[tokio::test]
    async fn decodes_base64_data_uri() {
        let e = extractor();
        let bytes = e.resolve_bytes(&base64_png_url(), 0, 0).await.unwrap();
        assert!(!bytes.is_empty());
    }

    #[tokio::test]
    async fn rejects_oversize_image() {
        let e = extractor(); // 1 MiB cap
        let big = format!("data:image/png;base64,{}", "A".repeat(2 << 20));
        let err = e.resolve_bytes(&big, 0, 0).await.unwrap_err();
        assert!(err.to_string().contains("cap"));
    }

    #[test]
    fn rejects_non_image_media_type() {
        let e = extractor();
        let err = e.validate_media_type("audio/mp3").unwrap_err();
        assert!(err.to_string().contains("unsupported media type"));
    }

    #[tokio::test]
    async fn preserves_position_order_across_parts() {
        // Build a message: text, image, text, image -> positions 1 and 3
        let msg = ChatCompletionMessage {
            role: "user".to_owned(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "a".to_owned(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: base64_png_url(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "b".to_owned(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: base64_png_url(),
                        detail: None,
                    },
                },
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let items = extractor().extract_openai(&[msg]).await.unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].position, 1);
        assert_eq!(items[1].position, 3);
    }

    #[tokio::test]
    async fn preserves_openai_detail() {
        let msg = ChatCompletionMessage {
            role: "user".to_owned(),
            content: Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: base64_png_url(),
                    detail: Some(higgs_models::vision::ImageDetail::Low),
                },
            }])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let items = extractor().extract_openai(&[msg]).await.unwrap();
        assert_eq!(items[0].detail, higgs_models::vision::ImageDetail::Low);
    }

    #[tokio::test]
    async fn extracts_anthropic_image_block() {
        let msg = AnthropicMessage {
            role: "user".to_owned(),
            content: AnthropicContent::Blocks(vec![
                ContentBlock::Text {
                    text: "what is this".to_owned(),
                },
                ContentBlock::Image {
                    source: serde_json::json!({
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_png(),
                    }),
                },
            ]),
        };
        let items = extractor().extract_anthropic(&[msg], None).await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].position, 1);
        assert_eq!(items[0].message_index, 0);
        assert_eq!(items[0].media_type, "image/png");
        assert!(!items[0].bytes.is_empty());
    }

    #[tokio::test]
    async fn extracts_anthropic_tool_result_image() {
        let msg = AnthropicMessage {
            role: "user".to_owned(),
            content: AnthropicContent::Blocks(vec![
                ContentBlock::Text {
                    text: "tool output".to_owned(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tu_1".to_owned(),
                    content: AnthropicContent::Blocks(vec![
                        ContentBlock::Text {
                            text: "result".to_owned(),
                        },
                        ContentBlock::Image {
                            source: serde_json::json!({
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_png(),
                            }),
                        },
                    ]),
                },
            ]),
        };
        let items = extractor().extract_anthropic(&[msg], None).await.unwrap();
        assert_eq!(items.len(), 1);
        // The nested image takes the enclosing tool_result's block index.
        assert_eq!(items[0].position, 1);
        assert_eq!(items[0].media_type, "image/jpeg");
        assert!(!items[0].bytes.is_empty());
    }

    #[tokio::test]
    async fn extracts_anthropic_url_source_image() {
        let msg = AnthropicMessage {
            role: "user".to_owned(),
            content: AnthropicContent::Blocks(vec![ContentBlock::Image {
                source: serde_json::json!({
                    "type": "url",
                    "media_type": "image/png",
                    "url": base64_png_url(),
                }),
            }]),
        };
        let items = extractor().extract_anthropic(&[msg], None).await.unwrap();
        assert_eq!(items.len(), 1);
        assert!(!items[0].bytes.is_empty());
    }

    #[tokio::test]
    async fn extracts_anthropic_max_dims() {
        let msg = AnthropicMessage {
            role: "user".to_owned(),
            content: AnthropicContent::Blocks(vec![ContentBlock::Image {
                source: serde_json::json!({
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_png(),
                    "max_width": 512,
                    "max_height": 384,
                }),
            }]),
        };
        let items = extractor().extract_anthropic(&[msg], None).await.unwrap();
        assert_eq!(items[0].max_dims, Some((512, 384)));
    }

    #[tokio::test]
    async fn rejects_unsupported_anthropic_image_source() {
        let msg = AnthropicMessage {
            role: "user".to_owned(),
            content: AnthropicContent::Blocks(vec![ContentBlock::Image {
                source: serde_json::json!({"type": "file", "media_type": "image/png"}),
            }]),
        };
        let err = extractor()
            .extract_anthropic(&[msg], None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unsupported image source type"));
    }
}
