use std::ffi::OsString;

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
