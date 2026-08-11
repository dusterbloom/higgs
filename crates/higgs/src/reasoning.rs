use crate::types::openai::ReasoningConfig;

fn model_defaults_to_non_thinking(model_names: &[&str]) -> bool {
    model_names.iter().any(|model_name| {
        let normalized = model_name.to_ascii_lowercase();
        normalized.match_indices("qwen3.6").any(|(idx, _)| {
            let after = idx + "qwen3.6".len();
            let before_is_boundary = idx == 0
                || normalized
                    .as_bytes()
                    .get(idx - 1)
                    .is_some_and(|b| !b.is_ascii_alphanumeric());
            let after_is_boundary = after == normalized.len()
                || normalized
                    .as_bytes()
                    .get(after)
                    .is_some_and(|b| !b.is_ascii_digit());
            before_is_boundary && after_is_boundary
        })
    })
}

/// LFM2.5 unconditionally injects `<think>` at the generation prompt
/// (its template ignores `enable_thinking`), so the reasoning tracker
/// MUST start inside a think block — otherwise the model's output is
/// misclassified as visible text.
fn model_always_thinks(model_names: &[&str]) -> bool {
    model_names.iter().any(|model_name| {
        let normalized = model_name.to_ascii_lowercase();
        // Match "lfm2.5" or "lfm2_5" but not "lfm2" (the non-thinking base).
        // Use boundary checks so "lfm2.5" matches but "lfm25" does not.
        for (idx, _) in normalized.match_indices("lfm2.5") {
            let before_ok = idx == 0
                || normalized.as_bytes().get(idx - 1)
                    .is_some_and(|b| !b.is_ascii_alphanumeric());
            let after_ok = normalized.as_bytes().get(idx + "lfm2.5".len())
                .is_none_or(|b| !b.is_ascii_alphanumeric());
            if before_ok && after_ok {
                return true;
            }
        }
        for (idx, _) in normalized.match_indices("lfm2_5") {
            let before_ok = idx == 0
                || normalized.as_bytes().get(idx - 1)
                    .is_some_and(|b| !b.is_ascii_alphanumeric());
            let after_ok = normalized.as_bytes().get(idx + "lfm2_5".len())
                .is_none_or(|b| !b.is_ascii_alphanumeric());
            if before_ok && after_ok {
                return true;
            }
        }
        false
    })
}

pub fn effective_thinking_enabled(
    engine_default: bool,
    model_names: &[&str],
    reasoning: Option<&ReasoningConfig>,
    explicit: Option<bool>,
) -> bool {
    if !engine_default {
        return false;
    }

    // Models whose template unconditionally injects `<think>` (LFM2.5)
    // always need the tracker in sync — explicit toggles can't override.
    if model_always_thinks(model_names) {
        return true;
    }

    // An explicit per-request toggle (`chat_template_kwargs.enable_thinking`,
    // e.g. nanobot's `/thinking on|off`) wins over OpenAI `reasoning.effort`.
    if let Some(want) = explicit {
        return want;
    }

    match reasoning.and_then(|r| r.effort.as_deref()) {
        Some(effort) if effort.is_empty() || effort.eq_ignore_ascii_case("none") => false,
        Some(_) => true,
        None => !model_defaults_to_non_thinking(model_names),
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_qwen35_on() {
        assert!(effective_thinking_enabled(
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
    fn qwen365_does_not_use_qwen36_default() {
        assert!(effective_thinking_enabled(
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
    fn explicit_enable_thinking_true_overrides_qwen36_default() {
        // chat_template_kwargs.enable_thinking=true turns reasoning on even for
        // a model that defaults off.
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

    #[test]
    fn lfm25_always_thinks_ignores_explicit_false() {
        // LFM2.5 injects <think> unconditionally — explicit /thinking off
        // must not break the tracker/template sync.
        assert!(effective_thinking_enabled(
            true,
            &["local:lfm2.5-2.6b-8bit"],
            None,
            Some(false),
        ));
    }

    #[test]
    fn lfm25_always_thinks_even_with_reasoning_none() {
        assert!(effective_thinking_enabled(
            true,
            &["lfm2.5-foo"],
            Some(&ReasoningConfig {
                effort: Some("none".to_owned()),
            }),
            None,
        ));
    }

    #[test]
    fn lfm2_base_does_not_always_think() {
        // Plain LFM2 (not 2.5) has a different template without <think>.
        assert!(!model_always_thinks(&["lfm2-2.6b"]));
    }

    #[test]
    fn lfm25_underscore_variant() {
        assert!(model_always_thinks(&["lfm2_5-3b"]));
    }

    #[test]
    fn lfm25_mid_path() {
        assert!(model_always_thinks(&["mlx-community/lfm2.5-2.6b-4bit"]));
    }
}
