use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const LEARNED_PROFILE_SCHEMA_VERSION: u32 = 1;

/// Exact persisted-profile identity. Every field participates in equality.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LearnedProfileKey {
    pub hardware_identifier: String,
    pub physical_memory_bytes: u64,
    pub os_version: String,
    pub os_build: String,
    pub backend_authority_bytes: u64,
    pub higgs_build: String,
    pub model_fingerprint: String,
    pub quantization: String,
    pub execution_mode: String,
    pub kv_representation: String,
    pub prefill_model_identity: Option<String>,
    pub execution_cache_fingerprint: String,
    pub drafter_identity: Option<String>,
}

impl LearnedProfileKey {
    fn is_complete(&self) -> bool {
        self.physical_memory_bytes > 0
            && self.backend_authority_bytes > 0
            && !self.hardware_identifier.is_empty()
            && !self.os_version.is_empty()
            && !self.os_build.is_empty()
            && !self.higgs_build.is_empty()
            && !self.model_fingerprint.is_empty()
            && !self.quantization.is_empty()
            && !self.execution_mode.is_empty()
            && !self.kv_representation.is_empty()
            && !self.execution_cache_fingerprint.is_empty()
            && self
                .prefill_model_identity
                .as_ref()
                .is_none_or(|identity| !identity.is_empty())
            && self
                .drafter_identity
                .as_ref()
                .is_none_or(|identity| !identity.is_empty())
    }
}

/// Conservative per-band evidence; no live admission state is persisted.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LearnedBandEvidence {
    pub prompt_band: u64,
    pub cold_high_water_bytes: u64,
    pub cold_replacement_qualified: bool,
    pub retained_high_water_bytes: u64,
    pub suffix_high_water_bytes: u64,
}

/// Versioned evidence file reused only with exact identity and startup headroom.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LearnedProfile {
    schema_version: u32,
    key: LearnedProfileKey,
    startup_headroom_bytes: u64,
    evidence: Vec<LearnedBandEvidence>,
}

impl LearnedProfile {
    #[must_use]
    pub fn new(
        key: LearnedProfileKey,
        startup_headroom_bytes: u64,
        evidence: Vec<LearnedBandEvidence>,
    ) -> Self {
        Self {
            schema_version: LEARNED_PROFILE_SCHEMA_VERSION,
            key,
            startup_headroom_bytes,
            evidence,
        }
    }

    #[must_use]
    pub fn key(&self) -> &LearnedProfileKey {
        &self.key
    }

    #[must_use]
    pub const fn startup_headroom_bytes(&self) -> u64 {
        self.startup_headroom_bytes
    }

    #[must_use]
    pub fn evidence(&self) -> &[LearnedBandEvidence] {
        &self.evidence
    }

    pub(super) fn is_complete(&self) -> bool {
        let unique_bands = self
            .evidence
            .iter()
            .map(|band| band.prompt_band)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == self.evidence.len();
        self.schema_version == LEARNED_PROFILE_SCHEMA_VERSION
            && self.key.is_complete()
            && self.startup_headroom_bytes > 0
            && !self.evidence.is_empty()
            && unique_bands
            && self.evidence.iter().all(|band| {
                band.prompt_band.is_power_of_two()
                    && (!band.cold_replacement_qualified || band.cold_high_water_bytes > 0)
                    && (band.cold_high_water_bytes > 0
                        || band.retained_high_water_bytes > 0
                        || band.suffix_high_water_bytes > 0)
            })
    }

    pub(super) fn is_compatible(
        &self,
        expected_key: &LearnedProfileKey,
        current_startup_headroom_bytes: u64,
    ) -> bool {
        self.is_complete()
            && self.key == *expected_key
            && current_startup_headroom_bytes >= self.startup_headroom_bytes
    }
}

#[derive(Debug)]
pub struct LearnedProfileStore {
    path: PathBuf,
}

impl LearnedProfileStore {
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn load(
        &self,
        expected_key: &LearnedProfileKey,
        current_startup_headroom_bytes: u64,
    ) -> io::Result<Option<LearnedProfile>> {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let Ok(profile) = serde_json::from_slice::<LearnedProfile>(&bytes) else {
            return Ok(None);
        };
        if !profile.is_compatible(expected_key, current_startup_headroom_bytes) {
            return Ok(None);
        }
        Ok(Some(profile))
    }

    pub fn save(&self, profile: &LearnedProfile) -> io::Result<()> {
        if !profile.is_complete() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "learned capacity profile is incomplete",
            ));
        }
        let parent = self.path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let file_name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("capacity.json");
        let temporary = parent.join(format!(".{file_name}.{}.tmp", uuid::Uuid::new_v4()));
        let bytes = serde_json::to_vec(profile).map_err(io::Error::other)?;
        let mut file = File::create(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        if let Err(error) = fs::rename(&temporary, &self.path) {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        File::open(parent)?.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn profile_key() -> LearnedProfileKey {
        LearnedProfileKey {
            hardware_identifier: "Mac15,9".into(),
            physical_memory_bytes: 64 * GIB,
            os_version: "15.6".into(),
            os_build: "24G90".into(),
            backend_authority_bytes: 48 * GIB,
            higgs_build: "abc123".into(),
            model_fingerprint: "sha256:model".into(),
            quantization: "3bit".into(),
            execution_mode: "native".into(),
            kv_representation: "fp16-hybrid".into(),
            prefill_model_identity: Some("prefill-v1".into()),
            execution_cache_fingerprint: "native;kv=fp16;radix=v1".into(),
            drafter_identity: None,
        }
    }

    #[test]
    fn learned_profile_is_atomic_evidence_only_and_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("capacity.json");
        let store = LearnedProfileStore::new(path.clone());
        let profile = LearnedProfile::new(
            profile_key(),
            12 * GIB,
            vec![LearnedBandEvidence {
                prompt_band: 65_536,
                cold_high_water_bytes: 5 * GIB,
                cold_replacement_qualified: true,
                retained_high_water_bytes: GIB,
                suffix_high_water_bytes: GIB / 2,
            }],
        );
        store.save(&profile).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json.get("bootId").is_none());
        assert!(json.get("capacity").is_none());
        assert_eq!(
            std::fs::read_dir(directory.path()).unwrap().count(),
            1,
            "atomic replacement must not leave a temporary file"
        );
        assert_eq!(
            store.load(&profile_key(), 12 * GIB).unwrap(),
            Some(profile.clone())
        );
        assert_eq!(store.load(&profile_key(), 12 * GIB - 1).unwrap(), None);

        let mut mismatch = profile_key();
        mismatch.model_fingerprint = "sha256:different".into();
        assert_eq!(store.load(&mismatch, 12 * GIB).unwrap(), None);
        let mut prefill_mismatch = profile_key();
        prefill_mismatch.prefill_model_identity = Some("prefill-v2".into());
        assert_eq!(store.load(&prefill_mismatch, 12 * GIB).unwrap(), None);
        let mut settings_mismatch = profile_key();
        settings_mismatch.execution_cache_fingerprint = "affine;kv=q8".into();
        assert_eq!(store.load(&settings_mismatch, 12 * GIB).unwrap(), None);

        let mut wrong_schema = json;
        wrong_schema["schemaVersion"] = 2.into();
        std::fs::write(&path, serde_json::to_vec(&wrong_schema).unwrap()).unwrap();
        assert_eq!(store.load(&profile_key(), 12 * GIB).unwrap(), None);

        let invalid_qualified = LearnedProfile::new(
            profile_key(),
            12 * GIB,
            vec![LearnedBandEvidence {
                prompt_band: 65_536,
                cold_high_water_bytes: 0,
                cold_replacement_qualified: true,
                retained_high_water_bytes: GIB,
                suffix_high_water_bytes: 0,
            }],
        );
        assert!(!invalid_qualified.is_complete());
        std::fs::write(&path, b"not-json").unwrap();
        assert_eq!(store.load(&profile_key(), 12 * GIB).unwrap(), None);
    }
}
