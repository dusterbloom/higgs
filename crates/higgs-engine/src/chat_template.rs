use minijinja::{Environment, Value};
use serde::Serialize;

use crate::error::EngineError;

/// A chat message for template rendering.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// Renders chat messages using a Jinja2 template (`HuggingFace` format).
pub struct ChatTemplateRenderer {
    env: Environment<'static>,
    /// Special tokens loaded from `tokenizer_config.json` for template rendering.
    bos_token: String,
    eos_token: String,
}

impl ChatTemplateRenderer {
    /// Create a renderer from a Jinja2 template string.
    pub fn new<S: Into<String>>(template_source: S) -> Result<Self, EngineError> {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson_filter);
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat".to_owned(), template_source.into())
            .map_err(|e| EngineError::Template(e.to_string()))?;
        Ok(Self {
            env,
            bos_token: String::new(),
            eos_token: String::new(),
        })
    }

    /// Load template from a model directory (`chat_template.jinja` or `tokenizer_config.json`).
    pub fn from_model_dir(model_dir: &std::path::Path) -> Result<Self, EngineError> {
        Self::try_from_model_dir(model_dir)?.ok_or_else(|| {
            EngineError::Template("No chat template found in model directory".to_owned())
        })
    }

    /// Like [`Self::from_model_dir`] but returns `Ok(None)` when no template is present,
    /// rather than an error. Parse/IO failures still propagate as `Err`.
    pub fn try_from_model_dir(model_dir: &std::path::Path) -> Result<Option<Self>, EngineError> {
        // Load tokenizer_config.json for special tokens (needed by both paths)
        let config_path = model_dir.join("tokenizer_config.json");
        let config: Option<serde_json::Value> = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| EngineError::Template(format!("Failed to read config: {e}")))?;
            Some(
                serde_json::from_str(&config_str)
                    .map_err(|e| EngineError::Template(format!("Invalid JSON: {e}")))?,
            )
        } else {
            None
        };

        let extract_token = |cfg: &serde_json::Value, key: &str| -> String {
            cfg.get(key)
                .and_then(|v| {
                    // Token can be a string or {"content": "..."} object
                    v.as_str().map(ToOwned::to_owned).or_else(|| {
                        v.get("content")
                            .and_then(|c| c.as_str())
                            .map(ToOwned::to_owned)
                    })
                })
                .unwrap_or_default()
        };

        let set_tokens = |renderer: &mut Self| {
            if let Some(ref cfg) = config {
                renderer.bos_token = extract_token(cfg, "bos_token");
                renderer.eos_token = extract_token(cfg, "eos_token");
            }
        };

        // Prefer standalone chat_template.jinja
        let jinja_path = model_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            let template = std::fs::read_to_string(&jinja_path)
                .map_err(|e| EngineError::Template(format!("Failed to read template: {e}")))?;
            let mut renderer = Self::new(&template)?;
            set_tokens(&mut renderer);
            return Ok(Some(renderer));
        }

        // Fall back to tokenizer_config.json
        if let Some(ref cfg) = config {
            if let Some(ct) = cfg.get("chat_template") {
                // String template
                if let Some(template) = ct.as_str() {
                    let mut renderer = Self::new(template)?;
                    set_tokens(&mut renderer);
                    return Ok(Some(renderer));
                }
                // Array of {name, template} objects -- use "default" or first entry
                if let Some(arr) = ct.as_array() {
                    let found = arr
                        .iter()
                        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("default"))
                        .or_else(|| arr.first())
                        .and_then(|v| v.get("template"))
                        .and_then(|v| v.as_str());
                    if let Some(template) = found {
                        let mut renderer = Self::new(template)?;
                        set_tokens(&mut renderer);
                        return Ok(Some(renderer));
                    }
                }
                tracing::warn!("chat_template field present but not a string or valid array");
            }
        }

        Ok(None)
    }

    /// Apply the chat template to messages, returning the formatted prompt string.
    pub fn apply(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<String, EngineError> {
        self.apply_with_thinking(messages, tools, add_generation_prompt, false)
    }

    /// Apply the chat template with explicit `enable_thinking` control.
    pub fn apply_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, EngineError> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| EngineError::Template(e.to_string()))?;

        let bos = &self.bos_token;
        let eos = &self.eos_token;
        let context = tools.map_or_else(
            || {
                minijinja::context! {
                    messages => messages,
                    add_generation_prompt => add_generation_prompt,
                    enable_thinking => enable_thinking,
                    bos_token => bos,
                    eos_token => eos,
                }
            },
            |tool_list| {
                minijinja::context! {
                    messages => messages,
                    tools => tool_list,
                    add_generation_prompt => add_generation_prompt,
                    enable_thinking => enable_thinking,
                    bos_token => bos,
                    eos_token => eos,
                }
            },
        );

        tmpl.render(context)
            .map_err(|e| EngineError::Template(e.to_string()))
    }
}

/// Custom tojson filter for minijinja (used by HF chat templates).
/// Normalise a tool-call JSON object so Qwen-Hermes-style chat templates
/// can render it without crashing on `tool_call.arguments|items`.
///
/// Two transformations are applied in place:
///
/// 1. **Flatten `function.{name,arguments}` to top level.** The `OpenAI`
///    request shape nests them under `function`; Qwen's
///    `chat_template.jinja` references `tool_call.name` and
///    `tool_call.arguments` directly. After this call, both shapes are
///    accessible.
/// 2. **Parse string-encoded arguments to a JSON value.** `OpenAI` says
///    `function.arguments` is a JSON-encoded string. Qwen's template
///    iterates it via `|items` (mapping pairs), which makes minijinja
///    raise `cannot convert value into pairs` when it sees a string.
///    Any string that successfully parses as JSON becomes a `Value`;
///    strings that fail to parse are left untouched so the template can
///    decide how to handle them.
///
/// Other fields (`id`, `type`, …) are preserved unchanged. Callers that
/// already supply the flat shape pay only the cost of a `serde_json::Value`
/// match.
pub fn normalize_tool_call_for_template(tc: &mut serde_json::Value) {
    let Some(obj) = tc.as_object_mut() else {
        return;
    };

    // Promote `function.name` / `function.arguments` to the top level.
    if let Some(function) = obj.get("function").cloned() {
        if let Some(func_obj) = function.as_object() {
            if !obj.contains_key("name") {
                if let Some(name) = func_obj.get("name") {
                    obj.insert("name".to_owned(), name.clone());
                }
            }
            if !obj.contains_key("arguments") {
                if let Some(arguments) = func_obj.get("arguments") {
                    obj.insert("arguments".to_owned(), arguments.clone());
                }
            }
        }
    }

    // String-encoded arguments → parsed JSON value (if it parses).
    if let Some(args) = obj.get_mut("arguments") {
        if let Some(s) = args.as_str() {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                *args = parsed;
            }
        }
    }
}

#[allow(clippy::needless_pass_by_value)]
fn tojson_filter(value: Value) -> Result<String, minijinja::Error> {
    let serialized = serde_json::to_string(&value).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "JSON serialization failed",
        )
        .with_source(e)
    })?;
    Ok(serialized)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_owned(),
            content: content.to_owned(),
            tool_calls: None,
        }
    }

    /// Create a minijinja environment with the tojson filter and return the
    /// compiled template for `{{ value | tojson }}`.
    fn tojson_env(template_source: &str) -> minijinja::Environment<'static> {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson_filter);
        env.add_template_owned("test".to_owned(), template_source.to_owned())
            .unwrap();
        env
    }

    const CHATML_TEMPLATE: &str = r"{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}";

    const TOJSON_TEMPLATE: &str = r"{{ value | tojson }}";

    #[test]
    fn test_simple_chatml_template() {
        let renderer = ChatTemplateRenderer::new(CHATML_TEMPLATE).unwrap();
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello!")];

        let result = renderer.apply(&messages, None, true).unwrap();
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_tojson_filter() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => "hello" })
            .unwrap();
        assert_eq!(result, r#""hello""#);
    }

    #[test]
    fn test_invalid_template_syntax_returns_error() {
        assert!(ChatTemplateRenderer::new("{%- invalid syntax %}}}").is_err());
    }

    #[test]
    fn test_apply_without_generation_prompt() {
        let renderer = ChatTemplateRenderer::new(CHATML_TEMPLATE).unwrap();
        let result = renderer
            .apply(&[msg("user", "Hello!")], None, false)
            .unwrap();
        assert!(!result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_apply_empty_messages() {
        let template = r"{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let result = renderer.apply(&[], None, false).unwrap();
        assert!(!result.contains("<|im_start|>"));
    }

    #[test]
    fn test_apply_with_tools() {
        let template = r"{%- for message in messages %}
{{ message.content }}
{%- endfor %}
{%- if tools %}
TOOLS:{{ tools | length }}
{%- endif %}";

        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let tools = vec![serde_json::json!({"type": "function", "function": {"name": "test"}})];
        let result = renderer
            .apply(&[msg("user", "Hi")], Some(&tools), false)
            .unwrap();
        assert!(result.contains("TOOLS:1"));
    }

    #[test]
    fn test_from_model_dir_no_template_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(ChatTemplateRenderer::from_model_dir(dir.path()).is_err());
    }

    #[test]
    fn test_from_model_dir_tokenizer_config_no_chat_template_field() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_type": "qwen2"}"#,
        )
        .unwrap();
        assert!(ChatTemplateRenderer::from_model_dir(dir.path()).is_err());
    }

    #[test]
    fn test_from_model_dir_standalone_jinja_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            r"{%- for message in messages %}{{ message.content }}{%- endfor %}",
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer
            .apply(&[msg("user", "hello")], None, false)
            .unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_from_model_dir_jinja_takes_priority_over_tokenizer_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "JINJA:{{ messages[0].content }}",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "CONFIG:{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer.apply(&[msg("user", "test")], None, false).unwrap();
        assert!(result.starts_with("JINJA:"));
    }

    #[test]
    fn test_from_model_dir_fallback_to_tokenizer_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "{%- for message in messages %}{{ message.content }}{%- endfor %}"}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer
            .apply(&[msg("user", "from_config")], None, false)
            .unwrap();
        assert_eq!(result, "from_config");
    }

    #[test]
    fn test_from_model_dir_malformed_tokenizer_config_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            "this is not valid json {{{",
        )
        .unwrap();
        match ChatTemplateRenderer::from_model_dir(dir.path()) {
            Err(e) => assert!(e.to_string().contains("Invalid JSON")),
            Ok(_) => panic!("Expected error for malformed JSON"),
        }
    }

    #[test]
    fn test_apply_with_assistant_role() {
        let template = r"{%- for message in messages %}<|{{ message.role }}|>{{ message.content }}{%- endfor %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let messages = vec![msg("user", "What is 2+2?"), msg("assistant", "4")];
        let result = renderer.apply(&messages, None, false).unwrap();
        assert!(result.contains("<|assistant|>4"));
    }

    #[test]
    fn test_apply_with_tool_calls_field() {
        let template = r"{%- for message in messages %}{{ message.role }}:{{ message.content }}{%- if message.tool_calls %} [tools]{%- endif %}{%- endfor %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let messages = vec![ChatMessage {
            role: "assistant".to_owned(),
            content: "calling tool".to_owned(),
            tool_calls: Some(vec![serde_json::json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
            })]),
        }];
        let result = renderer.apply(&messages, None, false).unwrap();
        assert!(result.contains("[tools]"));
    }

    #[test]
    fn test_tojson_filter_with_nested_objects() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let nested = serde_json::json!({"a": {"b": [1, 2, 3]}});
        let result = tmpl
            .render(minijinja::context! { value => nested })
            .unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(
            reparsed.get("a").unwrap().get("b").unwrap(),
            &serde_json::json!([1, 2, 3])
        );
    }

    #[test]
    fn test_tojson_filter_with_arrays() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => vec![1, 2, 3] })
            .unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(reparsed, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_tojson_filter_with_special_characters() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => "quotes: \"hello\" and backslash: \\" })
            .unwrap();
        let reparsed: String = serde_json::from_str(&result).unwrap();
        assert!(reparsed.contains("quotes: \"hello\""));
        assert!(reparsed.contains("backslash: \\"));
    }

    #[test]
    fn test_from_model_dir_array_of_templates_uses_default() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": [
                {"name": "rag", "template": "RAG:{{ messages[0].content }}"},
                {"name": "default", "template": "DEFAULT:{{ messages[0].content }}"}
            ]}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer.apply(&[msg("user", "hi")], None, false).unwrap();
        assert!(
            result.starts_with("DEFAULT:"),
            "Expected default template, got: {result}"
        );
    }

    #[test]
    fn test_from_model_dir_array_of_templates_falls_back_to_first() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": [
                {"name": "rag", "template": "RAG:{{ messages[0].content }}"},
                {"name": "tool_use", "template": "TOOL:{{ messages[0].content }}"}
            ]}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer.apply(&[msg("user", "hi")], None, false).unwrap();
        assert!(
            result.starts_with("RAG:"),
            "Expected first template, got: {result}"
        );
    }

    #[test]
    fn test_from_model_dir_array_template_empty_array_errors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": []}"#,
        )
        .unwrap();
        assert!(ChatTemplateRenderer::from_model_dir(dir.path()).is_err());
    }

    #[test]
    fn test_template_rendering_error_undefined_variable() {
        let renderer = ChatTemplateRenderer::new(r"{{ undefined_variable.nested_field }}").unwrap();
        assert!(renderer.apply(&[msg("user", "hi")], None, false).is_err());
    }

    // -----------------------------------------------------------------------
    // try_from_model_dir
    // -----------------------------------------------------------------------

    #[test]
    fn try_from_model_dir_empty_directory_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let result = ChatTemplateRenderer::try_from_model_dir(dir.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn try_from_model_dir_config_without_template_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_type": "starcoder2"}"#,
        )
        .unwrap();
        let result = ChatTemplateRenderer::try_from_model_dir(dir.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn try_from_model_dir_with_jinja_returns_some() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            r"{{ messages[0].content }}",
        )
        .unwrap();
        let result = ChatTemplateRenderer::try_from_model_dir(dir.path()).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn try_from_model_dir_with_config_template_returns_some() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let result = ChatTemplateRenderer::try_from_model_dir(dir.path()).unwrap();
        assert!(result.is_some());
    }

    // -----------------------------------------------------------------------
    // enable_thinking context passing
    // -----------------------------------------------------------------------

    /// Template that uses the `enable_thinking` variable.
    const THINKING_TEMPLATE: &str = r"{%- for message in messages %}{{ message.content }}{%- endfor %}{%- if enable_thinking %}<think>{%- endif %}";

    #[test]
    fn apply_with_thinking_false_omits_think_tag() {
        let renderer = ChatTemplateRenderer::new(THINKING_TEMPLATE).unwrap();
        let result = renderer
            .apply_with_thinking(&[msg("user", "hello")], None, false, false)
            .unwrap();
        assert!(
            !result.contains("<think>"),
            "should not contain <think> when disabled"
        );
    }

    #[test]
    fn apply_with_thinking_true_emits_think_tag() {
        let renderer = ChatTemplateRenderer::new(THINKING_TEMPLATE).unwrap();
        let result = renderer
            .apply_with_thinking(&[msg("user", "hello")], None, false, true)
            .unwrap();
        assert!(
            result.contains("<think>"),
            "should contain <think> when enabled"
        );
    }

    #[test]
    fn apply_delegates_to_apply_with_thinking_false() {
        let renderer = ChatTemplateRenderer::new(THINKING_TEMPLATE).unwrap();
        let via_apply = renderer.apply(&[msg("user", "hi")], None, false).unwrap();
        let via_explicit = renderer
            .apply_with_thinking(&[msg("user", "hi")], None, false, false)
            .unwrap();
        assert_eq!(
            via_apply, via_explicit,
            "apply() should delegate with enable_thinking=false"
        );
    }

    #[test]
    fn apply_with_thinking_and_tools() {
        let template = r"{%- for message in messages %}{{ message.content }}{%- endfor %}{%- if tools %}[TOOLS]{%- endif %}{%- if enable_thinking %}<think>{%- endif %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let tools = vec![serde_json::json!({"type": "function"})];
        let result = renderer
            .apply_with_thinking(&[msg("user", "hi")], Some(&tools), false, true)
            .unwrap();
        assert!(result.contains("[TOOLS]"), "tools should be rendered");
        assert!(result.contains("<think>"), "thinking tag should be present");
    }

    #[test]
    fn try_from_model_dir_malformed_json_returns_err() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            "not valid json {{{",
        )
        .unwrap();
        assert!(ChatTemplateRenderer::try_from_model_dir(dir.path()).is_err());
    }

    // -----------------------------------------------------------------
    // normalize_tool_call_for_template
    // -----------------------------------------------------------------
    //
    // Invariants asserted, one test per shape we observed in production:
    //
    // 1. `OpenAI` shape (name/arguments nested under `function`,
    //    arguments as JSON-encoded STRING) → after normalize, top-level
    //    name and arguments-as-mapping. This is the case that crashed
    //    Qwen's `chat_template.jinja:120` with "cannot convert value
    //    into pairs".
    // 2. Qwen-flat shape (top-level name/arguments, arguments already
    //    an object) → no-op, identity.
    // 3. Non-JSON string in `function.arguments` → flattened but kept
    //    as string (template can decide what to do).
    // 4. Non-object input (string, null, array) → no-op, can't panic.

    fn parsed(s: &str) -> serde_json::Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn normalize_openai_shape_to_qwen_flat() {
        let mut tc = parsed(
            r#"{
                "id": "call_0",
                "type": "function",
                "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
            }"#,
        );
        normalize_tool_call_for_template(&mut tc);

        // Top-level name and arguments are present.
        assert_eq!(tc.get("name").and_then(|v| v.as_str()), Some("get_weather"));
        // arguments is now an OBJECT, not a string.
        let args = tc.get("arguments").unwrap();
        assert!(args.is_object(), "expected arguments to be an object, got {args:?}");
        assert_eq!(args.get("city").and_then(|v| v.as_str()), Some("Paris"));
        // id and type preserved.
        assert_eq!(tc.get("id").and_then(|v| v.as_str()), Some("call_0"));
        assert_eq!(tc.get("type").and_then(|v| v.as_str()), Some("function"));
    }

    #[test]
    fn normalize_qwen_flat_shape_is_noop() {
        let original = parsed(
            r#"{ "name": "search", "arguments": { "q": "rust" } }"#,
        );
        let mut tc = original.clone();
        normalize_tool_call_for_template(&mut tc);
        assert_eq!(tc, original, "already-flat shape must be a no-op");
    }

    #[test]
    fn normalize_unparseable_string_arguments_kept_as_string() {
        let mut tc = parsed(
            r#"{
                "function": { "name": "f", "arguments": "this is not json" }
            }"#,
        );
        normalize_tool_call_for_template(&mut tc);
        // name flattened.
        assert_eq!(tc.get("name").and_then(|v| v.as_str()), Some("f"));
        // arguments preserved as string (the template / model side can
        // choose how to surface it).
        assert_eq!(
            tc.get("arguments").and_then(|v| v.as_str()),
            Some("this is not json")
        );
    }

    #[test]
    fn normalize_non_object_is_noop() {
        let mut s = parsed(r#""not a tool call""#);
        normalize_tool_call_for_template(&mut s);
        assert_eq!(s, parsed(r#""not a tool call""#));

        let mut n = parsed("null");
        normalize_tool_call_for_template(&mut n);
        assert_eq!(n, parsed("null"));

        let mut a = parsed("[1, 2, 3]");
        normalize_tool_call_for_template(&mut a);
        assert_eq!(a, parsed("[1, 2, 3]"));
    }
}
