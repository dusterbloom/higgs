use crate::types::openai::ReasoningConfig;

pub fn effective_thinking_enabled(
    thinking_supported: bool,
    _model_names: &[&str],
    reasoning: Option<&ReasoningConfig>,
    explicit: Option<bool>,
) -> bool {
    if !thinking_supported {
        return false;
    }

    // An explicit per-request toggle (`chat_template_kwargs.enable_thinking`,
    // e.g. nanobot's `/thinking on|off`) wins over OpenAI `reasoning.effort`.
    if let Some(want) = explicit {
        return want;
    }

    match reasoning.and_then(|r| r.effort.as_deref()) {
        Some(effort) if effort.is_empty() || effort.eq_ignore_ascii_case("none") => false,
        Some(_) => true,
        None => false,
    }
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_any_thinking_capable_model_off() {
        assert!(!effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.5-foo"],
            None,
            None,
        ));
    }

    #[test]
    fn defaults_qwen36_off_from_route_name() {
        assert!(!effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.6-35B-A3B-4bit"],
            None,
            None,
        ));
    }

    #[test]
    fn defaults_qwen36_off_from_engine_name_even_when_aliased() {
        assert!(!effective_thinking_enabled(
            true,
            &["qwen", "mlx-community/Qwen3.6-35B-A3B-4bit"],
            None,
            None,
        ));
    }

    #[test]
    fn defaults_unrecognized_thinking_model_off() {
        assert!(!effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.65-35B-A3B-4bit"],
            None,
            None,
        ));
    }

    #[test]
    fn honors_reasoning_none() {
        assert!(!effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.5-foo"],
            Some(&ReasoningConfig {
                effort: Some("none".to_owned()),
            }),
            None,
        ));
    }

    #[test]
    fn honors_empty_reasoning_as_not_explicit() {
        assert!(!effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.5-foo"],
            Some(&ReasoningConfig {
                effort: Some(String::new()),
            }),
            None,
        ));
    }

    #[test]
    fn honors_explicit_reasoning_request() {
        assert!(effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.6-35B-A3B-4bit"],
            Some(&ReasoningConfig {
                effort: Some("low".to_owned()),
            }),
            None,
        ));
    }

    #[test]
    fn engine_default_off_overrides_explicit_request() {
        assert!(!effective_thinking_enabled(
            false,
            &["mlx-community/Qwen3.5-foo"],
            Some(&ReasoningConfig {
                effort: Some("low".to_owned()),
            }),
            None,
        ));
    }

    #[test]
    fn configured_default_can_enable_thinking() {
        // The config-derived explicit default remains an opt-in for a model
        // whose omitted request defaults to non-thinking.
        assert!(effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.6-35B-A3B-4bit"],
            None,
            Some(true),
        ));
    }

    #[test]
    fn explicit_enable_thinking_false_overrides_reasoning_effort() {
        // /thinking off wins even when reasoning.effort asked for thinking.
        assert!(!effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.5-foo"],
            Some(&ReasoningConfig {
                effort: Some("high".to_owned()),
            }),
            Some(false),
        ));
    }

    #[test]
    fn explicit_true_cannot_force_a_non_thinking_engine() {
        assert!(!effective_thinking_enabled(
            false,
            &["mlx-community/Qwen3.5-foo"],
            None,
            Some(true),
        ));
    }
}
