use std::ffi::OsString;

use higgs_engine::simple::SimpleEngine;

pub(crate) const MAX_DSPARK_RELATIVE_REGRESSION: f64 = 0.03;

pub(crate) struct ScopedEnvVar {
    key: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvVar {
    #[allow(unsafe_code)]
    pub(crate) fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: The ignored real-model gates are documented and invoked with
        // `--test-threads=1`. Each guard restores its process-global setting.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: See `ScopedEnvVar::set`; restoration happens in the same
        // serial ignored test that performed the mutation.
        unsafe {
            if let Some(previous) = self.previous.take() {
                std::env::set_var(self.key, previous);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}

/// Pin the full-Q4, row4 reference path used by the promotable dSpark gates.
///
/// Keeping this in shared test support prevents the session and radix gates
/// from silently exercising different verifier, head, or kernel settings.
pub(crate) struct ReferenceDsparkEnv {
    _guards: Vec<ScopedEnvVar>,
}

impl ReferenceDsparkEnv {
    pub(crate) fn install() -> Self {
        Self {
            _guards: vec![
                ScopedEnvVar::set("HIGGS_DFLASH_VERIFY_MODE", "block"),
                ScopedEnvVar::set("HIGGS_DFLASH_GATE", "0"),
                ScopedEnvVar::set("HIGGS_DFLASH_ADAPTIVE", "0"),
                ScopedEnvVar::set("HIGGS_DSPARK_DRAFT_CAP", "4"),
                ScopedEnvVar::set("HIGGS_DSPARK_TARGET_HEAD", "0"),
                ScopedEnvVar::set("HIGGS_BONSAI_TG_LUT4", "1"),
                ScopedEnvVar::set("HIGGS_BONSAI_TG_LUT4_FUSED_MLP", "0"),
                ScopedEnvVar::set("HIGGS_BONSAI_TG_LUT4_M5_WG", "256"),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DflashAcceptance {
    pub(crate) matched: u64,
    pub(crate) drafted: u64,
}

impl DflashAcceptance {
    pub(crate) fn rate(self) -> f64 {
        self.matched as f64 / self.drafted as f64
    }
}

pub(crate) fn dflash_acceptance(engine: &SimpleEngine, label: &str) -> DflashAcceptance {
    let matches = engine.last_dflash_draft_matches();
    let draft_counts = engine.last_dflash_draft_counts();
    assert_eq!(
        matches.len(),
        draft_counts.len(),
        "{label}: every speculative round must report both matched and drafted counts"
    );
    let acceptance = DflashAcceptance {
        matched: matches.into_iter().map(u64::from).sum(),
        drafted: draft_counts.into_iter().map(u64::from).sum(),
    };
    assert!(
        acceptance.drafted > 0,
        "{label}: the acceptance gate requires at least one drafted token"
    );
    assert!(
        acceptance.matched <= acceptance.drafted,
        "{label}: matched tokens cannot exceed drafted tokens"
    );
    acceptance
}

pub(crate) fn dflash_decode_tps(engine: &SimpleEngine, label: &str) -> f64 {
    let rate = engine
        .last_dflash_decode_tokens_per_second()
        .unwrap_or_else(|| panic!("{label}: dSpark request must publish positive decode timing"));
    assert!(
        rate.is_finite() && rate > 0.0,
        "{label}: dSpark decode rate must be finite and positive, got {rate}"
    );
    rate
}

fn relative_floor(baseline: f64, max_regression: f64) -> f64 {
    baseline * (1.0 - max_regression)
}

pub(crate) fn assert_acceptance_within(
    candidate_label: &str,
    candidate: DflashAcceptance,
    baseline: DflashAcceptance,
) {
    let floor = relative_floor(baseline.rate(), MAX_DSPARK_RELATIVE_REGRESSION);
    assert!(
        candidate.rate() >= floor,
        "{candidate_label} aggregate dSpark acceptance is more than 3% below the uncached \
         baseline: candidate={:.2}% ({}/{}) floor={:.2}% baseline={:.2}% ({}/{})",
        candidate.rate() * 100.0,
        candidate.matched,
        candidate.drafted,
        floor * 100.0,
        baseline.rate() * 100.0,
        baseline.matched,
        baseline.drafted
    );
}

pub(crate) fn assert_decode_tps_within(
    candidate_label: &str,
    candidate_tps: f64,
    baseline_tps: f64,
) {
    let floor = relative_floor(baseline_tps, MAX_DSPARK_RELATIVE_REGRESSION);
    assert!(
        candidate_tps >= floor,
        "{candidate_label} decode throughput is more than 3% below the uncached dSpark \
         baseline: candidate={candidate_tps:.2} tok/s floor={floor:.2} tok/s \
         baseline={baseline_tps:.2} tok/s"
    );
}

#[cfg(test)]
mod tests {
    use super::relative_floor;

    #[test]
    fn relative_floor_is_multiplicative() {
        let floor = relative_floor(100.0, 0.03);
        assert!((floor - 97.0).abs() < f64::EPSILON);
    }
}
