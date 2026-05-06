/// Projection for one speculative depth.
///
/// `depth` is the number of draft positions verified per cycle. The expected
/// token count includes the target correction/bonus token, so depth 0 would be
/// 1.0 emitted token per ordinary decode step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeculationProjection {
    pub depth: usize,
    pub expected_emitted_tokens: f64,
    pub cycle_cost_multiplier: f64,
    pub speedup: f64,
}

/// Static description of one GDN value head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateHead {
    pub layer: usize,
    pub head: usize,
    pub a_log: f64,
    pub dt_bias: f64,
}

/// Static score derived from the nominal GDN decay gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateHeadScore {
    pub layer: usize,
    pub head: usize,
    pub retention: f64,
    pub time_constant_steps: f64,
}

/// Per-head precision decision for head-selective ternary experiments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadPrecision {
    PreserveBf16,
    QuantizeTernary,
}

/// Controls how many long-horizon heads stay in higher precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProtectionConfig {
    pub protected_fraction: f64,
    pub min_protected: usize,
    pub max_protected: Option<usize>,
    pub max_refresh_interval: usize,
}

/// Concrete offline plan for one GDN value head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadExecutionPlan {
    pub layer: usize,
    pub head: usize,
    pub retention: f64,
    pub time_constant_steps: f64,
    pub precision: HeadPrecision,
    pub refresh_every_steps: usize,
}

/// Expected emitted tokens for conditional per-position acceptance rates.
///
/// For K draft positions, speculative decoding always emits at least one
/// target token. If position acceptances are conditional on all previous draft
/// positions being accepted, the expectation is:
/// `1 + a0 + a0*a1 + ... + a0*...*aK`.
pub fn expected_emitted_tokens_conditional(position_acceptance: &[f64]) -> f64 {
    let mut prefix_probability = 1.0;
    let mut emitted = 1.0;

    for &acceptance in position_acceptance {
        prefix_probability *= acceptance.clamp(0.0, 1.0);
        emitted += prefix_probability;
    }

    emitted
}

/// Project speedup for depths 1..=N using conditional acceptance rates.
///
/// `cycle_cost_multipliers[i]` is the relative cost of verifying depth `i + 1`.
/// It should include any MTP-head or drafter overhead. A multiplier of 1.10
/// means one speculative cycle costs 10% more than a baseline decode cycle.
pub fn project_conditional_depths(
    position_acceptance: &[f64],
    cycle_cost_multipliers: &[f64],
) -> Result<Vec<SpeculationProjection>, &'static str> {
    if cycle_cost_multipliers.len() > position_acceptance.len() {
        return Err("cycle_cost_multipliers cannot be longer than position_acceptance");
    }

    let mut projections = Vec::with_capacity(cycle_cost_multipliers.len());
    for (idx, &cycle_cost_multiplier) in cycle_cost_multipliers.iter().enumerate() {
        if !cycle_cost_multiplier.is_finite() || cycle_cost_multiplier <= 0.0 {
            return Err("cycle cost multipliers must be finite and positive");
        }

        let depth = idx + 1;
        let expected_emitted_tokens =
            expected_emitted_tokens_conditional_prefix(position_acceptance, depth);
        projections.push(SpeculationProjection {
            depth,
            expected_emitted_tokens,
            cycle_cost_multiplier,
            speedup: expected_emitted_tokens / cycle_cost_multiplier,
        });
    }

    Ok(projections)
}

/// Pick the highest-speedup projection, preferring shallower depths on ties.
pub fn best_projected_depth(
    projections: &[SpeculationProjection],
) -> Option<SpeculationProjection> {
    projections.iter().copied().max_by(|a, b| {
        a.speedup
            .total_cmp(&b.speedup)
            .then_with(|| b.depth.cmp(&a.depth))
    })
}

/// Project end-to-end tok/s from a baseline decode rate and depth projection.
pub fn projected_tps(
    baseline_tps: f64,
    projection: SpeculationProjection,
) -> Result<f64, &'static str> {
    if !baseline_tps.is_finite() || baseline_tps <= 0.0 {
        return Err("baseline_tps must be finite and positive");
    }
    if !projection.speedup.is_finite() || projection.speedup <= 0.0 {
        return Err("projection speedup must be finite and positive");
    }

    Ok(baseline_tps * projection.speedup)
}

/// Whether a projection clears a target tok/s threshold.
pub fn meets_tps_target(
    baseline_tps: f64,
    projection: SpeculationProjection,
    target_tps: f64,
) -> Result<bool, &'static str> {
    if !target_tps.is_finite() || target_tps <= 0.0 {
        return Err("target_tps must be finite and positive");
    }

    Ok(projected_tps(baseline_tps, projection)? >= target_tps)
}

/// Nominal one-step GDN retention from static gate parameters.
///
/// This evaluates the recurrent decay at `a = 0`:
/// `g = exp(-exp(A_log) * softplus(dt_bias))`.
pub fn nominal_gdn_retention(a_log: f64, dt_bias: f64) -> f64 {
    let rate = a_log.exp() * softplus(dt_bias);
    (-rate).exp().clamp(0.0, 1.0)
}

/// Time constant, in decode steps, implied by a one-step retention value.
pub fn refresh_interval_steps(retention: f64, max_interval: f64) -> f64 {
    if !max_interval.is_finite() || max_interval <= 0.0 {
        return 0.0;
    }
    if !retention.is_finite() || retention <= 0.0 {
        return 0.0;
    }
    if retention >= 1.0 {
        return max_interval;
    }

    let interval = -1.0 / retention.ln();
    if interval.is_finite() {
        interval.min(max_interval)
    } else {
        max_interval
    }
}

/// Integer refresh cadence derived from the retention time constant.
///
/// This floors the continuous time constant so fast-forgetting heads stay on a
/// one-step cadence. Long-horizon heads are capped by `max_interval`.
pub fn refresh_every_steps(retention: f64, max_interval: usize) -> usize {
    if max_interval == 0 {
        return 0;
    }

    let interval = refresh_interval_steps(retention, max_interval as f64);
    if interval < 1.0 {
        1
    } else {
        interval.floor() as usize
    }
}

/// Score heads by static GDN retention.
pub fn score_gdn_heads(heads: &[GateHead]) -> Vec<GateHeadScore> {
    heads
        .iter()
        .map(|head| {
            let retention = nominal_gdn_retention(head.a_log, head.dt_bias);
            GateHeadScore {
                layer: head.layer,
                head: head.head,
                retention,
                time_constant_steps: refresh_interval_steps(retention, f64::INFINITY),
            }
        })
        .collect()
}

/// Return the longest-horizon heads first for selective protection.
pub fn select_long_horizon_heads(heads: &[GateHead], protect_count: usize) -> Vec<GateHeadScore> {
    let mut scores = score_gdn_heads(heads);
    scores.sort_by(|a, b| {
        b.retention
            .total_cmp(&a.retention)
            .then_with(|| a.layer.cmp(&b.layer))
            .then_with(|| a.head.cmp(&b.head))
    });
    scores.truncate(protect_count.min(scores.len()));
    scores
}

/// Build a BF16-protection and temporal-refresh plan for static GDN heads.
pub fn plan_head_selective_ternary(
    heads: &[GateHead],
    config: ProtectionConfig,
) -> Result<Vec<HeadExecutionPlan>, &'static str> {
    if !config.protected_fraction.is_finite()
        || config.protected_fraction < 0.0
        || config.protected_fraction > 1.0
    {
        return Err("protected_fraction must be finite and within 0..=1");
    }
    if config.max_refresh_interval == 0 {
        return Err("max_refresh_interval must be at least 1");
    }

    let raw_count = (heads.len() as f64 * config.protected_fraction).ceil() as usize;
    let mut protect_count = raw_count.max(config.min_protected).min(heads.len());
    if let Some(max_protected) = config.max_protected {
        protect_count = protect_count.min(max_protected);
    }

    let protected = select_long_horizon_heads(heads, protect_count);
    let protected_keys: std::collections::BTreeSet<(usize, usize)> = protected
        .iter()
        .map(|head| (head.layer, head.head))
        .collect();

    let mut scores = score_gdn_heads(heads);
    scores.sort_by(|a, b| a.layer.cmp(&b.layer).then_with(|| a.head.cmp(&b.head)));

    Ok(scores
        .into_iter()
        .map(|score| {
            let precision = if protected_keys.contains(&(score.layer, score.head)) {
                HeadPrecision::PreserveBf16
            } else {
                HeadPrecision::QuantizeTernary
            };
            HeadExecutionPlan {
                layer: score.layer,
                head: score.head,
                retention: score.retention,
                time_constant_steps: score.time_constant_steps,
                precision,
                refresh_every_steps: refresh_every_steps(
                    score.retention,
                    config.max_refresh_interval,
                ),
            }
        })
        .collect())
}

/// Extract static GDN gate heads from flattened model parameter names.
///
/// Accepts names ending in `model.layers.N.linear_attn.A_log` and
/// `model.layers.N.linear_attn.dt_bias`, including longer prefixes such as
/// `language_model.model.layers.N...`. Non-GDN tensors are ignored.
pub fn extract_gate_heads_from_tensors<'a, I>(tensors: I) -> Result<Vec<GateHead>, String>
where
    I: IntoIterator<Item = (&'a str, &'a [f64])>,
{
    let mut by_layer: std::collections::BTreeMap<usize, StaticGatePair<'a>> =
        std::collections::BTreeMap::new();

    for (name, values) in tensors {
        let Some((layer, kind)) = parse_static_gate_name(name) else {
            continue;
        };
        let entry = by_layer.entry(layer).or_default();
        match kind {
            StaticGateKind::ALog => entry.a_log = Some(values),
            StaticGateKind::DtBias => entry.dt_bias = Some(values),
        }
    }

    let mut heads = Vec::new();
    for (layer, pair) in by_layer {
        let a_log = pair
            .a_log
            .ok_or_else(|| format!("layer {layer}: missing A_log"))?;
        let dt_bias = pair
            .dt_bias
            .ok_or_else(|| format!("layer {layer}: missing dt_bias"))?;

        if a_log.len() != dt_bias.len() {
            return Err(format!(
                "layer {layer}: mismatched A_log ({}) and dt_bias ({}) lengths",
                a_log.len(),
                dt_bias.len()
            ));
        }

        for (head, (&a_log, &dt_bias)) in a_log.iter().zip(dt_bias.iter()).enumerate() {
            if !a_log.is_finite() || !dt_bias.is_finite() {
                return Err(format!("layer {layer} head {head}: non-finite gate value"));
            }
            heads.push(GateHead {
                layer,
                head,
                a_log,
                dt_bias,
            });
        }
    }

    Ok(heads)
}

#[derive(Default)]
struct StaticGatePair<'a> {
    a_log: Option<&'a [f64]>,
    dt_bias: Option<&'a [f64]>,
}

#[derive(Clone, Copy)]
enum StaticGateKind {
    ALog,
    DtBias,
}

fn parse_static_gate_name(name: &str) -> Option<(usize, StaticGateKind)> {
    let (_, after_layers) = name.split_once(".layers.")?;
    let (layer, after_layer) = after_layers.split_once('.')?;
    let layer = layer.parse::<usize>().ok()?;
    let suffix = after_layer.strip_prefix("linear_attn.")?;
    match suffix {
        "A_log" => Some((layer, StaticGateKind::ALog)),
        "dt_bias" => Some((layer, StaticGateKind::DtBias)),
        _ => None,
    }
}

fn expected_emitted_tokens_conditional_prefix(position_acceptance: &[f64], depth: usize) -> f64 {
    let mut prefix_probability = 1.0;
    let mut emitted = 1.0;

    for &acceptance in position_acceptance.iter().take(depth) {
        prefix_probability *= acceptance.clamp(0.0, 1.0);
        emitted += prefix_probability;
    }

    emitted
}

fn softplus(x: f64) -> f64 {
    if x > 40.0 {
        x
    } else if x < -40.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} tolerance={tolerance}"
        );
    }

    #[test]
    fn conditional_acceptance_matches_geometric_mtp_model() {
        let emitted = expected_emitted_tokens_conditional(&[0.6, 0.6, 0.6]);
        assert_close(emitted, 2.176, 1e-12);
    }

    #[test]
    fn decayed_position_acceptance_can_select_mtp2_over_mtp3() {
        let projections =
            project_conditional_depths(&[0.847, 0.587, 0.493], &[1.03, 1.10, 1.35]).unwrap();

        assert_close(projections[1].expected_emitted_tokens, 2.344_189, 1e-6);
        let best = best_projected_depth(&projections).unwrap();
        assert_eq!(best.depth, 2);
    }

    #[test]
    fn static_gate_scores_rank_long_horizon_heads_first() {
        let heads = [
            GateHead {
                layer: 2,
                head: 0,
                a_log: 0.0,
                dt_bias: 0.0,
            },
            GateHead {
                layer: 0,
                head: 8,
                a_log: -10.0,
                dt_bias: -10.0,
            },
            GateHead {
                layer: 1,
                head: 3,
                a_log: 2.0,
                dt_bias: 1.0,
            },
        ];

        let protected = select_long_horizon_heads(&heads, 2);
        assert_eq!(protected[0].layer, 0);
        assert_eq!(protected[0].head, 8);
        assert!(protected[0].retention > 0.999_999_99);
        assert_eq!(protected[1].layer, 2);
        assert_eq!(protected[1].head, 0);
    }

    #[test]
    fn refresh_interval_tracks_retention_time_constant() {
        assert_close(
            refresh_interval_steps(0.5, 512.0),
            1.442_695_040_888_963_4,
            1e-12,
        );
        assert_close(
            refresh_interval_steps(0.99, 512.0),
            99.499_162_473_422_07,
            1e-10,
        );
        assert_close(refresh_interval_steps(0.999_999, 128.0), 128.0, 1e-12);
    }

    #[test]
    fn head_selective_ternary_plan_protects_configured_fraction() {
        let heads = [
            GateHead {
                layer: 0,
                head: 0,
                a_log: -10.0,
                dt_bias: -10.0,
            },
            GateHead {
                layer: 0,
                head: 1,
                a_log: 0.0,
                dt_bias: 0.0,
            },
            GateHead {
                layer: 0,
                head: 2,
                a_log: 2.0,
                dt_bias: 1.0,
            },
            GateHead {
                layer: 0,
                head: 3,
                a_log: 3.0,
                dt_bias: 2.0,
            },
        ];

        let plan = plan_head_selective_ternary(
            &heads,
            ProtectionConfig {
                protected_fraction: 0.25,
                min_protected: 1,
                max_protected: None,
                max_refresh_interval: 128,
            },
        )
        .unwrap();

        assert_eq!(plan.len(), 4);
        assert_eq!(plan[0].precision, HeadPrecision::PreserveBf16);
        assert_eq!(plan[0].refresh_every_steps, 128);
        assert!(
            plan[1..]
                .iter()
                .all(|entry| entry.precision == HeadPrecision::QuantizeTernary)
        );
    }

    #[test]
    fn temporal_schedule_is_conservative_for_fast_forgetting_heads() {
        assert_eq!(refresh_every_steps(0.5, 512), 1);
        assert_eq!(refresh_every_steps(0.99, 512), 99);
        assert_eq!(refresh_every_steps(0.999_999, 128), 128);
    }

    #[test]
    fn extracts_gate_heads_from_flattened_qwen_parameter_names() {
        let a0 = [-10.0, 0.0];
        let dt0 = [-10.0, 0.0];
        let a1 = [2.0];
        let dt1 = [1.0];
        let ignored = [123.0];
        let heads = extract_gate_heads_from_tensors([
            ("model.layers.0.linear_attn.A_log", a0.as_slice()),
            ("model.layers.0.linear_attn.dt_bias", dt0.as_slice()),
            (
                "language_model.model.layers.1.linear_attn.A_log",
                a1.as_slice(),
            ),
            (
                "language_model.model.layers.1.linear_attn.dt_bias",
                dt1.as_slice(),
            ),
            ("model.layers.3.self_attn.q_proj.weight", ignored.as_slice()),
        ])
        .unwrap();

        assert_eq!(heads.len(), 3);
        assert_eq!(heads[0].layer, 0);
        assert_eq!(heads[0].head, 0);
        assert_close(heads[0].a_log, -10.0, 0.0);
        assert_close(heads[0].dt_bias, -10.0, 0.0);
        assert_eq!(heads[2].layer, 1);
        assert_eq!(heads[2].head, 0);
    }

    #[test]
    fn extractor_rejects_mismatched_static_gate_shapes() {
        let a0 = [0.0, 1.0];
        let dt0 = [0.0];
        let err = extract_gate_heads_from_tensors([
            ("model.layers.0.linear_attn.A_log", a0.as_slice()),
            ("model.layers.0.linear_attn.dt_bias", dt0.as_slice()),
        ])
        .unwrap_err();

        assert!(err.contains("mismatched"));
    }

    #[test]
    fn throughput_projection_confirms_mtp2_clears_20_tps_target() {
        let projections =
            project_conditional_depths(&[0.847, 0.587, 0.493], &[1.03, 1.10, 1.35]).unwrap();
        let best = best_projected_depth(&projections).unwrap();

        assert_eq!(best.depth, 2);
        assert_close(projected_tps(12.0, best).unwrap(), 25.572_970_909, 1e-9);
        assert!(meets_tps_target(12.0, best, 20.0).unwrap());
    }
}
