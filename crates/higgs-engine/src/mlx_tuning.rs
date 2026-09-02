use std::path::Path;

/// Conservative default for chunked prefill on 27B-class models.
const DEFAULT_CHUNKED_PREFILL_THRESHOLD: i32 = 512;

/// Conservative default chunk size for long-prefill chunking.
const DEFAULT_CHUNKED_PREFILL_CHUNK_SIZE: i32 = 512;

const DEFAULT_PAGED_KV_TARGET_BYTES: usize = 512 * 1024 * 1024;
const MIN_PAGED_KV_TARGET_BYTES: usize = 256 * 1024 * 1024;
const MAX_PAGED_KV_TARGET_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_MTP_DRAFT_N_MAX: usize = 8;

/// Immutable process-wide allocator and device working-set facts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MlxMemorySnapshot {
    /// Absolute process-wide MLX active allocation at measurement time.
    pub active_bytes: u64,
    pub peak_bytes: u64,
    /// `None` means MLX reports no configured allocator limit.
    pub memory_limit_bytes: Option<u64>,
    /// `None` means the backend does not expose Metal's recommended limit.
    pub metal_recommended_working_set_bytes: Option<u64>,
}

/// A content-free allocator observation for one bounded inference phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestMemoryHighWater {
    pub phase: MemoryPhase,
    pub active_before_bytes: u64,
    pub active_after_bytes: u64,
    pub peak_bytes: u64,
    pub active_growth_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxMemoryProbeError {
    QueryFailed(&'static str),
    ResetPeakFailed,
}

impl std::fmt::Display for MlxMemoryProbeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueryFailed(operation) => {
                write!(formatter, "MLX memory query failed: {operation}")
            }
            Self::ResetPeakFailed => formatter.write_str("MLX peak-memory reset failed"),
        }
    }
}

impl std::error::Error for MlxMemoryProbeError {}

trait MlxMemoryProbe {
    fn active_memory_bytes(&self) -> Result<u128, MlxMemoryProbeError>;
    fn peak_memory_bytes(&self) -> Result<u128, MlxMemoryProbeError>;
    fn memory_limit_bytes(&self) -> Result<u128, MlxMemoryProbeError>;
    fn metal_recommended_working_set_bytes(&self) -> Result<Option<u128>, MlxMemoryProbeError>;
    fn reset_peak_memory(&self) -> Result<(), MlxMemoryProbeError>;
}

struct SystemMlxMemoryProbe;

#[allow(unsafe_code)]
impl MlxMemoryProbe for SystemMlxMemoryProbe {
    fn active_memory_bytes(&self) -> Result<u128, MlxMemoryProbeError> {
        let mut active_bytes = 0_usize;
        unsafe {
            if mlx_sys::mlx_get_active_memory(&raw mut active_bytes) != 0 {
                return Err(MlxMemoryProbeError::QueryFailed("active memory"));
            }
        }
        Ok(active_bytes as u128)
    }

    fn peak_memory_bytes(&self) -> Result<u128, MlxMemoryProbeError> {
        let mut peak_bytes = 0_usize;
        unsafe {
            if mlx_sys::mlx_get_peak_memory(&raw mut peak_bytes) != 0 {
                return Err(MlxMemoryProbeError::QueryFailed("peak memory"));
            }
        }
        Ok(peak_bytes as u128)
    }

    fn memory_limit_bytes(&self) -> Result<u128, MlxMemoryProbeError> {
        let mut memory_limit_bytes = 0_usize;
        unsafe {
            if mlx_sys::mlx_get_memory_limit(&raw mut memory_limit_bytes) != 0 {
                return Err(MlxMemoryProbeError::QueryFailed("memory limit"));
            }
        }
        Ok(memory_limit_bytes as u128)
    }

    fn metal_recommended_working_set_bytes(&self) -> Result<Option<u128>, MlxMemoryProbeError> {
        Ok(metal_recommended_working_set_bytes().map(u128::from))
    }

    fn reset_peak_memory(&self) -> Result<(), MlxMemoryProbeError> {
        let result = unsafe { mlx_sys::mlx_reset_peak_memory() };
        if result == 0 {
            Ok(())
        } else {
            Err(MlxMemoryProbeError::ResetPeakFailed)
        }
    }
}

#[allow(unsafe_code)]
pub(crate) fn metal_recommended_working_set_bytes() -> Option<u64> {
    unsafe {
        let mut info = mlx_sys::mlx_device_info_new();
        let mut device = mlx_sys::mlx_device_new();
        mlx_sys::mlx_get_default_device(&raw mut device);
        let value = if mlx_sys::mlx_device_info_get(&raw mut info, device) == 0 {
            let mut bytes = 0_usize;
            let key = c"max_recommended_working_set_size";
            (mlx_sys::mlx_device_info_get_size(&raw mut bytes, info, key.as_ptr()) == 0
                && bytes > 0)
                .then(|| u64::try_from(bytes).ok())
                .flatten()
        } else {
            None
        };
        mlx_sys::mlx_device_info_free(info);
        mlx_sys::mlx_device_free(device);
        value
    }
}

fn measure_mlx_memory(
    probe: &impl MlxMemoryProbe,
) -> Result<MlxMemorySnapshot, MlxMemoryProbeError> {
    let active_bytes = probe.active_memory_bytes()?;
    let peak_bytes = probe.peak_memory_bytes()?;
    let memory_limit_bytes = probe.memory_limit_bytes()?;
    let metal_recommended_working_set_bytes = probe.metal_recommended_working_set_bytes()?;

    Ok(MlxMemorySnapshot {
        active_bytes: checked_raw_memory_bytes(active_bytes, "active memory conversion")?,
        peak_bytes: checked_raw_memory_bytes(peak_bytes, "peak memory conversion")?,
        memory_limit_bytes: (memory_limit_bytes > 0)
            .then(|| checked_raw_memory_bytes(memory_limit_bytes, "memory limit conversion"))
            .transpose()?,
        metal_recommended_working_set_bytes: metal_recommended_working_set_bytes
            .map(|bytes| checked_raw_memory_bytes(bytes, "Metal working-set conversion"))
            .transpose()?,
    })
}

fn checked_raw_memory_bytes(
    bytes: u128,
    operation: &'static str,
) -> Result<u64, MlxMemoryProbeError> {
    u64::try_from(bytes).map_err(|_| MlxMemoryProbeError::QueryFailed(operation))
}

impl MlxMemorySnapshot {
    /// Measure the current MLX allocator and backend working-set facts.
    pub fn measure() -> Result<Self, MlxMemoryProbeError> {
        measure_mlx_memory(&SystemMlxMemoryProbe)
    }
}

/// Before/after sampler for a bounded prefill or decode phase.
///
/// Starting a sample resets MLX's process-global peak counter. Callers must
/// therefore hold Higgs's process-global MLX execution gate until `finish`.
/// The mutable borrow prevents nested samplers from resetting the same global
/// counter while an earlier phase is still being measured.
///
/// ```compile_fail
/// use higgs_engine::{MemoryPhase, RequestMemorySampler};
///
/// let mut gate = higgs_models::mlx_exec::acquire();
/// let prefill = RequestMemorySampler::start(MemoryPhase::Prefill, &mut gate).unwrap();
/// let decode = RequestMemorySampler::start(MemoryPhase::Decode, &mut gate).unwrap();
/// drop(prefill);
/// drop(decode);
/// ```
pub struct RequestMemorySampler<'gate> {
    phase: MemoryPhase,
    before: MlxMemorySnapshot,
    _mlx_gate: &'gate mut higgs_models::mlx_exec::MlxExecToken,
}

impl<'gate> RequestMemorySampler<'gate> {
    pub fn start(
        phase: MemoryPhase,
        mlx_gate: &'gate mut higgs_models::mlx_exec::MlxExecToken,
    ) -> Result<Self, MlxMemoryProbeError> {
        Self::start_with_probe(phase, mlx_gate, &SystemMlxMemoryProbe)
    }

    fn start_with_probe(
        phase: MemoryPhase,
        mlx_gate: &'gate mut higgs_models::mlx_exec::MlxExecToken,
        probe: &impl MlxMemoryProbe,
    ) -> Result<Self, MlxMemoryProbeError> {
        let before = measure_mlx_memory(probe)?;
        probe.reset_peak_memory()?;
        Ok(Self {
            phase,
            before,
            _mlx_gate: mlx_gate,
        })
    }

    pub fn finish(self) -> Result<RequestMemoryHighWater, MlxMemoryProbeError> {
        self.finish_with_probe(&SystemMlxMemoryProbe)
    }

    fn finish_with_probe(
        self,
        probe: &impl MlxMemoryProbe,
    ) -> Result<RequestMemoryHighWater, MlxMemoryProbeError> {
        let after = measure_mlx_memory(probe)?;
        Ok(RequestMemoryHighWater {
            phase: self.phase,
            active_before_bytes: self.before.active_bytes,
            active_after_bytes: after.active_bytes,
            peak_bytes: after
                .peak_bytes
                .max(self.before.active_bytes)
                .max(after.active_bytes),
            active_growth_bytes: after.active_bytes.saturating_sub(self.before.active_bytes),
        })
    }
}

/// Artifact and measured allocator residency attributable to one model load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFootprint {
    pub artifact_weight_bytes: u64,
    /// Checked `after - before` delta from a serialized model-load window.
    ///
    /// This is load-attributable growth, not current absolute process state;
    /// use [`MlxMemorySnapshot::active_bytes`] for an absolute measurement.
    pub loaded_mlx_bytes: u64,
}

impl ModelFootprint {
    /// Describe one model load measured without concurrent MLX allocations.
    pub fn from_load_measurements(
        model_dir: &Path,
        before: MlxMemorySnapshot,
        after: MlxMemorySnapshot,
    ) -> Option<Self> {
        Some(Self {
            artifact_weight_bytes: strict_model_artifact_bytes(model_dir)?,
            loaded_mlx_bytes: after.active_bytes.checked_sub(before.active_bytes)?,
        })
    }
}

/// Conservative transient-prefill function over a declared input domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransientPrefillEstimate {
    pub base_bytes: u64,
    pub bytes_per_prompt_token: u64,
    pub bytes_per_chunk_token: u64,
    pub max_prompt_tokens: u64,
    pub max_chunk_tokens: u64,
}

impl TransientPrefillEstimate {
    pub fn estimate_bytes(&self, prompt_tokens: u64, chunk_tokens: u64) -> Option<u64> {
        if prompt_tokens > self.max_prompt_tokens || chunk_tokens > self.max_chunk_tokens {
            return None;
        }
        self.base_bytes
            .checked_add(self.bytes_per_prompt_token.checked_mul(prompt_tokens)?)?
            .checked_add(self.bytes_per_chunk_token.checked_mul(chunk_tokens)?)
    }
}

/// Immutable byte-domain costs required by the adaptive capacity controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineCostDescription {
    pub fixed_live_session_bytes: u64,
    pub persistent_bytes_per_token: u64,
    pub decode_workspace_bytes: u64,
    pub transient_prefill: TransientPrefillEstimate,
}

impl EngineCostDescription {
    /// Derive the fp16 dense-KV term from model metadata without exposing the
    /// loader's private model representation.
    pub fn fp16_from_model_dir(
        model_dir: &Path,
        fixed_live_session_bytes: u64,
        decode_workspace_bytes: u64,
        transient_prefill: TransientPrefillEstimate,
    ) -> Option<Self> {
        let metadata = ModelMetadata::from_model_dir(model_dir);
        let layers = u64::try_from(metadata.num_hidden_layers?).ok()?;
        let kv_heads = u64::try_from(metadata.num_key_value_heads?).ok()?;
        let head_dim = u64::try_from(metadata.head_dim?).ok()?;
        if layers == 0 || kv_heads == 0 || head_dim == 0 {
            return None;
        }
        let dense_layers = if metadata.is_hybrid_attention() {
            let interval = u64::try_from(metadata.full_attention_interval?).ok()?;
            if interval == 0 {
                return None;
            }
            layers / interval
        } else {
            layers
        };
        if dense_layers == 0 {
            return None;
        }
        let persistent_bytes_per_token = dense_layers
            .checked_mul(2)?
            .checked_mul(kv_heads)?
            .checked_mul(head_dim)?
            .checked_mul(2)?;

        Some(Self {
            fixed_live_session_bytes,
            persistent_bytes_per_token,
            decode_workspace_bytes,
            transient_prefill,
        })
    }

    pub fn persistent_bytes(&self, tokens: u64) -> Option<u64> {
        self.persistent_bytes_per_token.checked_mul(tokens)
    }
}

fn parse_positive_chunked_prefill_value(raw: Option<&str>, default: i32) -> i32 {
    raw.and_then(|s| s.parse::<i32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn parse_mtp_draft_n_max(raw: Option<&str>, default: usize) -> usize {
    raw.and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .map_or(default, |v| v.min(MAX_MTP_DRAFT_N_MAX))
}

fn parse_enabled_flag(raw: Option<&str>) -> Option<bool> {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "true" | "on" | "yes") => Some(true),
        Some("0" | "false" | "off" | "no") => Some(false),
        _ => None,
    }
}

const fn default_mtp_draft_n_max(size_class: ModelSizeClass) -> usize {
    match size_class {
        ModelSizeClass::Huge => 2,
        ModelSizeClass::Small | ModelSizeClass::Medium | ModelSizeClass::Large => 1,
    }
}

/// User-requested MLX profile before auto-resolution.
///
/// `RequestedMlxProfile` is used by CLI/config/env precedence and is resolved to
/// a concrete `ResolvedMlxProfile` before runtime settings are built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestedMlxProfile {
    /// Use conservative baseline runtime settings.
    ///
    /// This value is only exposed via env aliases (`baseline`, `default`, `off`).
    Baseline,
    /// Resolve per model via size-class heuristics.
    #[default]
    Auto,
    /// Tune for lower time-to-first-token at potential throughput cost.
    Latency,
    /// Favor a mixed latency/throughput balance.
    Balanced,
    /// Tune for sustained throughput.
    Throughput,
}

impl RequestedMlxProfile {
    /// Parse `HIGGS_MLX_PROFILE`, including legacy aliases.
    ///
    /// Supported values:
    /// - canonical: `auto`, `latency`, `balanced`, `throughput`
    /// - aliases: `baseline` (same as `default`/`off`), `auto` (`mlx`)
    /// - legacy benchmark aliases: `ttft`, `tps`
    pub fn from_env_raw(raw: Option<&str>) -> Result<Option<Self>, String> {
        match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            None => Ok(None),
            Some("baseline" | "default" | "off") => Ok(Some(Self::Baseline)),
            Some("auto" | "mlx") => Ok(Some(Self::Auto)),
            Some("latency" | "ttft") => Ok(Some(Self::Latency)),
            Some("balanced") => Ok(Some(Self::Balanced)),
            Some("throughput" | "tps") => Ok(Some(Self::Throughput)),
            Some(other) => Err(format!(
                "invalid HIGGS_MLX_PROFILE '{other}'; expected auto, latency, balanced, throughput, or legacy aliases baseline/default/off, ttft, tps, mlx"
            )),
        }
    }

    /// Canonical profile token used for diagnostics and user-facing logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Auto => "auto",
            Self::Latency => "latency",
            Self::Balanced => "balanced",
            Self::Throughput => "throughput",
        }
    }
}

/// MLX profile after resolving `auto` against model metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedMlxProfile {
    /// Conservative baseline runtime settings.
    Baseline,
    /// Latency-biased runtime settings.
    Latency,
    /// Balanced runtime settings.
    Balanced,
    /// Throughput-biased runtime settings.
    Throughput,
}

impl ResolvedMlxProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Latency => "latency",
            Self::Balanced => "balanced",
            Self::Throughput => "throughput",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelSizeClass {
    Small,
    Medium,
    Large,
    Huge,
}

#[derive(Debug, Clone, Default)]
struct ModelMetadata {
    model_type: Option<String>,
    quantization_bits: Option<u8>,
    num_hidden_layers: Option<usize>,
    hidden_size: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    full_attention_interval: Option<usize>,
    max_position_embeddings: Option<usize>,
    weight_bytes: Option<u64>,
}

impl ModelMetadata {
    fn from_model_dir(model_dir: &Path) -> Self {
        let config_path = model_dir.join("config.json");
        let config = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .unwrap_or(serde_json::Value::Null);

        Self {
            model_type: config_lookup_str(&config, "model_type").map(str::to_owned),
            quantization_bits: config
                .get("quantization")
                .and_then(|quantization| quantization.get("bits"))
                .and_then(serde_json::Value::as_u64)
                .and_then(|bits| u8::try_from(bits).ok()),
            num_hidden_layers: config_lookup_u64(&config, "num_hidden_layers")
                .and_then(|v| usize::try_from(v).ok()),
            hidden_size: config_lookup_u64(&config, "hidden_size")
                .and_then(|v| usize::try_from(v).ok()),
            num_key_value_heads: config_lookup_u64(&config, "num_key_value_heads")
                .and_then(|v| usize::try_from(v).ok()),
            head_dim: config_lookup_u64(&config, "head_dim").and_then(|v| usize::try_from(v).ok()),
            full_attention_interval: config_lookup_u64(&config, "full_attention_interval")
                .and_then(|v| usize::try_from(v).ok()),
            max_position_embeddings: config_lookup_u64(&config, "max_position_embeddings")
                .and_then(|v| usize::try_from(v).ok()),
            weight_bytes: model_weight_bytes(model_dir),
        }
    }

    fn size_class(&self) -> ModelSizeClass {
        const SMALL_BYTES: u64 = 4 * 1024 * 1024 * 1024;
        const MEDIUM_BYTES: u64 = 8 * 1024 * 1024 * 1024;
        const LARGE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

        self.weight_bytes.map_or_else(
            || match (
                self.num_hidden_layers.unwrap_or_default(),
                self.hidden_size.unwrap_or_default(),
            ) {
                (layers, hidden) if layers >= 80 || hidden >= 8192 => ModelSizeClass::Huge,
                (layers, hidden) if layers >= 48 || hidden >= 5120 => ModelSizeClass::Large,
                (layers, hidden) if layers >= 32 || hidden >= 3072 => ModelSizeClass::Medium,
                _ => ModelSizeClass::Small,
            },
            |weight_bytes| {
                if weight_bytes >= LARGE_BYTES {
                    ModelSizeClass::Huge
                } else if weight_bytes >= MEDIUM_BYTES {
                    ModelSizeClass::Large
                } else if weight_bytes >= SMALL_BYTES {
                    ModelSizeClass::Medium
                } else {
                    ModelSizeClass::Small
                }
            },
        )
    }

    fn is_moe(&self) -> bool {
        matches!(
            self.model_type.as_deref(),
            Some("qwen3_moe" | "qwen3_5_moe" | "deepseek_v2")
        )
    }

    fn is_long_context(&self) -> bool {
        self.max_position_embeddings.unwrap_or_default() >= 65_536
    }

    fn is_hybrid_attention(&self) -> bool {
        matches!(
            self.model_type.as_deref(),
            Some("qwen3_next" | "qwen3_5" | "qwen3_5_moe")
        )
    }

    fn should_clear_cache_after_prefill(&self) -> bool {
        match self.model_type.as_deref() {
            Some("qwen3_5") => {
                matches!(self.quantization_bits, Some(bits) if (1..=4).contains(&bits))
            }
            Some("qwen3_5_moe") => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
/// Tunable MLX runtime settings derived from a requested profile and model metadata.
///
/// These settings are consumed only by the local simple-engine path and are resolved
/// before model load. Advanced `HIGGS_*` env overrides may further mutate these
/// settings in `from_model_dir`.
pub struct MlxRuntimeTuning {
    requested_profile: RequestedMlxProfile,
    resolved_profile: ResolvedMlxProfile,
    chunked_prefill_threshold: i32,
    chunked_prefill_chunk_size: i32,
    clear_cache_after_prefill: bool,
    enable_mtp: bool,
    /// Maximum MTP draft tokens per speculative cycle.
    ///
    /// Controlled by `HIGGS_MTP_DRAFT_N_MAX`; defaults to 2 for huge
    /// checkpoints and 1 otherwise, clamped to 1..=8.
    mtp_draft_n_max: usize,
    paged_kv_target_bytes: usize,
}

impl MlxRuntimeTuning {
    /// Resolve runtime tuning settings from model metadata.
    /// `requested_profile` drives both explicit profile selection and auto-resolution.
    pub fn from_model_dir(model_dir: &Path, requested_profile: RequestedMlxProfile) -> Self {
        let metadata = ModelMetadata::from_model_dir(model_dir);
        let resolved_profile = resolve_profile_from_metadata(requested_profile, &metadata);
        let mut tuning = Self::from_profile(requested_profile, resolved_profile, &metadata);

        tuning.chunked_prefill_threshold = parse_positive_chunked_prefill_value(
            std::env::var("HIGGS_CHUNKED_PREFILL_THRESHOLD")
                .ok()
                .as_deref(),
            tuning.chunked_prefill_threshold,
        );
        tuning.chunked_prefill_chunk_size = parse_positive_chunked_prefill_value(
            std::env::var("HIGGS_CHUNKED_PREFILL_CHUNK_SIZE")
                .ok()
                .as_deref(),
            tuning.chunked_prefill_chunk_size,
        );
        tuning.clear_cache_after_prefill = parse_enabled_flag(
            std::env::var("HIGGS_CLEAR_CACHE_AFTER_PREFILL")
                .ok()
                .as_deref(),
        )
        .unwrap_or(tuning.clear_cache_after_prefill);
        tuning.enable_mtp = parse_enabled_flag(std::env::var("HIGGS_MTP").ok().as_deref())
            .unwrap_or(tuning.enable_mtp);
        tuning.mtp_draft_n_max = parse_mtp_draft_n_max(
            std::env::var("HIGGS_MTP_DRAFT_N_MAX").ok().as_deref(),
            tuning.mtp_draft_n_max,
        );

        tuning
    }

    fn from_profile(
        requested_profile: RequestedMlxProfile,
        resolved_profile: ResolvedMlxProfile,
        metadata: &ModelMetadata,
    ) -> Self {
        let size_class = metadata.size_class();
        let is_long_context = metadata.is_long_context();
        let is_moe = metadata.is_moe();
        let (balanced_threshold, balanced_chunk) =
            balanced_chunked_prefill(size_class, is_long_context, is_moe);
        let balanced_paged_kv = heuristic_paged_kv_target_bytes(metadata, size_class, is_moe);
        let default_mtp_draft_n_max = default_mtp_draft_n_max(size_class);
        let clear_cache_after_prefill = metadata.should_clear_cache_after_prefill();

        match resolved_profile {
            ResolvedMlxProfile::Baseline => Self {
                requested_profile,
                resolved_profile,
                chunked_prefill_threshold: DEFAULT_CHUNKED_PREFILL_THRESHOLD,
                chunked_prefill_chunk_size: DEFAULT_CHUNKED_PREFILL_CHUNK_SIZE,
                clear_cache_after_prefill,
                enable_mtp: false,
                mtp_draft_n_max: 1,
                paged_kv_target_bytes: DEFAULT_PAGED_KV_TARGET_BYTES,
            },
            ResolvedMlxProfile::Latency => Self {
                requested_profile,
                resolved_profile,
                chunked_prefill_threshold: (balanced_threshold.saturating_mul(2)).min(4096),
                chunked_prefill_chunk_size: balanced_chunk.max(768),
                clear_cache_after_prefill,
                enable_mtp: true,
                mtp_draft_n_max: default_mtp_draft_n_max,
                paged_kv_target_bytes: clamp_paged_kv_target_bytes(
                    balanced_paged_kv.saturating_mul(9) / 8,
                ),
            },
            ResolvedMlxProfile::Balanced => Self {
                requested_profile,
                resolved_profile,
                chunked_prefill_threshold: balanced_threshold,
                chunked_prefill_chunk_size: balanced_chunk,
                clear_cache_after_prefill,
                enable_mtp: true,
                mtp_draft_n_max: default_mtp_draft_n_max,
                paged_kv_target_bytes: balanced_paged_kv,
            },
            ResolvedMlxProfile::Throughput => Self {
                requested_profile,
                resolved_profile,
                chunked_prefill_threshold: balanced_threshold.max(1024),
                chunked_prefill_chunk_size: balanced_chunk.max(1024),
                clear_cache_after_prefill,
                enable_mtp: true,
                mtp_draft_n_max: default_mtp_draft_n_max,
                paged_kv_target_bytes: clamp_paged_kv_target_bytes(
                    balanced_paged_kv.saturating_mul(5) / 4,
                ),
            },
        }
    }

    /// Requested profile before auto-resolution or env/default mutation.
    pub const fn requested_profile(&self) -> RequestedMlxProfile {
        self.requested_profile
    }

    /// Effective profile after resolution.
    pub const fn resolved_profile(&self) -> ResolvedMlxProfile {
        self.resolved_profile
    }

    /// Chunked prefill threshold used by `simple` generation.
    pub const fn chunked_prefill_threshold(&self) -> i32 {
        self.chunked_prefill_threshold
    }

    pub const fn chunked_prefill_chunk_size(&self) -> i32 {
        self.chunked_prefill_chunk_size
    }

    pub const fn clear_cache_after_prefill(&self) -> bool {
        self.clear_cache_after_prefill
    }

    pub const fn enable_mtp(&self) -> bool {
        self.enable_mtp
    }

    /// Maximum MTP draft tokens per speculative cycle.
    ///
    /// Configurable via `HIGGS_MTP_DRAFT_N_MAX`.
    pub const fn mtp_draft_n_max(&self) -> usize {
        self.mtp_draft_n_max
    }

    pub const fn paged_kv_target_bytes(&self) -> usize {
        self.paged_kv_target_bytes
    }
}

/// Resolve the full `MlxRuntimeTuning` object for runtime use.
pub fn resolve_runtime_tuning(
    model_dir: &Path,
    requested_profile: RequestedMlxProfile,
) -> MlxRuntimeTuning {
    MlxRuntimeTuning::from_model_dir(model_dir, requested_profile)
}

/// Resolve only the effective runtime profile (used for diagnostics and reporting).
pub fn resolve_effective_mlx_profile(
    model_dir: &Path,
    requested_profile: RequestedMlxProfile,
) -> ResolvedMlxProfile {
    let metadata = ModelMetadata::from_model_dir(model_dir);
    resolve_profile_from_metadata(requested_profile, &metadata)
}

fn resolve_profile_from_metadata(
    requested_profile: RequestedMlxProfile,
    metadata: &ModelMetadata,
) -> ResolvedMlxProfile {
    match requested_profile {
        RequestedMlxProfile::Baseline => ResolvedMlxProfile::Baseline,
        RequestedMlxProfile::Latency => ResolvedMlxProfile::Latency,
        RequestedMlxProfile::Balanced => ResolvedMlxProfile::Balanced,
        RequestedMlxProfile::Throughput => ResolvedMlxProfile::Throughput,
        RequestedMlxProfile::Auto => match metadata.size_class() {
            ModelSizeClass::Small | ModelSizeClass::Medium => ResolvedMlxProfile::Balanced,
            ModelSizeClass::Large | ModelSizeClass::Huge => ResolvedMlxProfile::Throughput,
        },
    }
}

fn balanced_chunked_prefill(
    size_class: ModelSizeClass,
    is_long_context: bool,
    is_moe: bool,
) -> (i32, i32) {
    let (base_threshold, base_chunk) = match size_class {
        ModelSizeClass::Small => (2048, 1024),
        ModelSizeClass::Medium => (1536, 768),
        ModelSizeClass::Large => (1024, 512),
        ModelSizeClass::Huge => (768, 384),
    };

    let tuned_threshold = if is_long_context {
        base_threshold.min(1024)
    } else {
        base_threshold
    };
    let tuned_chunk = if is_moe {
        base_chunk.min(512)
    } else {
        base_chunk
    };
    (tuned_threshold, tuned_chunk)
}

fn heuristic_paged_kv_target_bytes(
    metadata: &ModelMetadata,
    size_class: ModelSizeClass,
    is_moe: bool,
) -> usize {
    heuristic_paged_kv_target_bytes_with_max(
        metadata,
        size_class,
        is_moe,
        configured_max_working_set_bytes(),
    )
}

fn heuristic_paged_kv_target_bytes_with_max(
    metadata: &ModelMetadata,
    size_class: ModelSizeClass,
    is_moe: bool,
    max_working_set_bytes: Option<usize>,
) -> usize {
    let Some(max_recommended) = max_working_set_bytes else {
        return DEFAULT_PAGED_KV_TARGET_BYTES;
    };

    let available = metadata
        .weight_bytes
        .and_then(|bytes| usize::try_from(bytes).ok())
        .map_or(max_recommended, |weight_bytes| {
            max_recommended.saturating_sub(weight_bytes)
        });

    if available == 0 {
        return DEFAULT_PAGED_KV_TARGET_BYTES.min(max_recommended);
    }

    let divisor = if is_moe {
        8
    } else {
        match size_class {
            ModelSizeClass::Small => 4,
            ModelSizeClass::Medium => 5,
            ModelSizeClass::Large => 6,
            ModelSizeClass::Huge => 8,
        }
    };

    clamp_paged_kv_target_bytes(available / divisor).min(max_recommended)
}

fn configured_max_working_set_bytes() -> Option<usize> {
    std::env::var("HIGGS_MLX_MAX_WORKING_SET_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn clamp_paged_kv_target_bytes(bytes: usize) -> usize {
    bytes.clamp(MIN_PAGED_KV_TARGET_BYTES, MAX_PAGED_KV_TARGET_BYTES)
}

fn config_lookup<'a>(config: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    config
        .get(key)
        .filter(|v| !v.is_null())
        .or_else(|| config.get("text_config").and_then(|tc| tc.get(key)))
}

fn config_lookup_str<'a>(config: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    config_lookup(config, key).and_then(serde_json::Value::as_str)
}

fn config_lookup_u64(config: &serde_json::Value, key: &str) -> Option<u64> {
    config_lookup(config, key).and_then(serde_json::Value::as_u64)
}

fn checked_artifact_total(total: u64, bytes: u64) -> Option<u64> {
    total.checked_add(bytes)
}

fn strict_model_artifact_bytes(model_dir: &Path) -> Option<u64> {
    let mut total = 0_u64;
    let mut stack = vec![model_dir.to_path_buf()];

    while let Some(dir) = stack.pop() {
        for entry_result in std::fs::read_dir(&dir).ok()? {
            let entry = entry_result.ok()?;
            let path = entry.path();
            let file_type = entry.file_type().ok()?;
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if (file_type.is_file() || file_type.is_symlink())
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
            {
                let metadata = if file_type.is_symlink() {
                    std::fs::metadata(&path)
                } else {
                    entry.metadata()
                }
                .ok()?;
                if !metadata.is_file() {
                    return None;
                }
                total = checked_artifact_total(total, metadata.len())?;
            }
        }
    }

    (total > 0).then_some(total)
}

fn model_weight_bytes(model_dir: &Path) -> Option<u64> {
    let mut total: u64 = 0;
    let mut stack = vec![model_dir.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(err) => {
                tracing::warn!(
                    model_dir = %dir.display(),
                    error = %err,
                    "Failed to read MLX model directory while estimating weight; continuing"
                );
                continue;
            }
        };
        for entry_result in entries {
            let dir_entry = match entry_result {
                Ok(entry) => entry,
                Err(err) => {
                    tracing::warn!(error = %err, "Failed to read MLX model directory entry; skipping");
                    continue;
                }
            };
            let path = dir_entry.path();
            let file_type = match dir_entry.file_type() {
                Ok(file_type) => file_type,
                Err(err) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %err,
                        "Failed to read MLX directory entry type; skipping"
                    );
                    continue;
                }
            };
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if (file_type.is_file() || file_type.is_symlink())
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
            {
                let metadata = if file_type.is_symlink() {
                    std::fs::metadata(&path)
                } else {
                    dir_entry.metadata()
                };
                match metadata {
                    Ok(meta) if meta.is_file() => {
                        total = total.saturating_add(meta.len());
                    }
                    Ok(_) => {}
                    Err(err) => {
                        tracing::warn!(
                            path = %path.display(),
                            error = %err,
                            "Failed to stat MLX model file while estimating weight; skipping"
                        );
                    }
                }
            }
        }
    }

    Some(total).filter(|sum| *sum > 0)
}

#[cfg(test)]
mod tests {
    use super::{
        EngineCostDescription, MemoryPhase, MlxMemoryProbe, MlxMemoryProbeError, MlxMemorySnapshot,
        MlxRuntimeTuning, ModelFootprint, ModelMetadata, ModelSizeClass, RequestMemorySampler,
        RequestedMlxProfile, ResolvedMlxProfile, TransientPrefillEstimate, checked_artifact_total,
        default_mtp_draft_n_max, heuristic_paged_kv_target_bytes_with_max, measure_mlx_memory,
        model_weight_bytes, parse_enabled_flag, parse_mtp_draft_n_max,
        parse_positive_chunked_prefill_value, resolve_effective_mlx_profile,
        resolve_profile_from_metadata, resolve_runtime_tuning,
    };
    use std::collections::VecDeque;
    use std::fs;
    use std::sync::Mutex;
    use tempfile::TempDir;

    #[derive(Clone, Copy)]
    struct RawMemoryMeasurement {
        active_bytes: Result<u128, MlxMemoryProbeError>,
        peak_bytes: Result<u128, MlxMemoryProbeError>,
        memory_limit_bytes: Result<u128, MlxMemoryProbeError>,
        metal_recommended_working_set_bytes: Result<Option<u128>, MlxMemoryProbeError>,
    }

    impl RawMemoryMeasurement {
        fn successful(
            active_bytes: u128,
            peak_bytes: u128,
            memory_limit_bytes: u128,
            metal_recommended_working_set_bytes: Option<u128>,
        ) -> Self {
            Self {
                active_bytes: Ok(active_bytes),
                peak_bytes: Ok(peak_bytes),
                memory_limit_bytes: Ok(memory_limit_bytes),
                metal_recommended_working_set_bytes: Ok(metal_recommended_working_set_bytes),
            }
        }
    }

    struct FakeMemoryProbe {
        measurements: Mutex<VecDeque<RawMemoryMeasurement>>,
        reset_result: Result<(), MlxMemoryProbeError>,
        resets: Mutex<usize>,
    }

    impl FakeMemoryProbe {
        fn new(snapshots: impl IntoIterator<Item = MlxMemorySnapshot>) -> Self {
            Self::from_raw(snapshots.into_iter().map(|snapshot| {
                RawMemoryMeasurement::successful(
                    u128::from(snapshot.active_bytes),
                    u128::from(snapshot.peak_bytes),
                    u128::from(snapshot.memory_limit_bytes.unwrap_or(0)),
                    snapshot.metal_recommended_working_set_bytes.map(u128::from),
                )
            }))
        }

        fn from_raw(measurements: impl IntoIterator<Item = RawMemoryMeasurement>) -> Self {
            Self {
                measurements: Mutex::new(measurements.into_iter().collect()),
                reset_result: Ok(()),
                resets: Mutex::new(0),
            }
        }

        fn with_reset_result(mut self, result: Result<(), MlxMemoryProbeError>) -> Self {
            self.reset_result = result;
            self
        }

        fn current(
            &self,
            exhausted: &'static str,
        ) -> Result<RawMemoryMeasurement, MlxMemoryProbeError> {
            self.measurements
                .lock()
                .map_err(|_| MlxMemoryProbeError::QueryFailed("test probe lock"))?
                .front()
                .copied()
                .ok_or(MlxMemoryProbeError::QueryFailed(exhausted))
        }

        fn take(&self) -> Result<RawMemoryMeasurement, MlxMemoryProbeError> {
            self.measurements
                .lock()
                .map_err(|_| MlxMemoryProbeError::QueryFailed("test probe lock"))?
                .pop_front()
                .ok_or(MlxMemoryProbeError::QueryFailed(
                    "test measurement exhausted",
                ))
        }
    }

    impl MlxMemoryProbe for FakeMemoryProbe {
        fn active_memory_bytes(&self) -> Result<u128, MlxMemoryProbeError> {
            self.current("test active memory exhausted")?.active_bytes
        }

        fn peak_memory_bytes(&self) -> Result<u128, MlxMemoryProbeError> {
            self.current("test peak memory exhausted")?.peak_bytes
        }

        fn memory_limit_bytes(&self) -> Result<u128, MlxMemoryProbeError> {
            self.current("test memory limit exhausted")?
                .memory_limit_bytes
        }

        fn metal_recommended_working_set_bytes(&self) -> Result<Option<u128>, MlxMemoryProbeError> {
            self.take()?.metal_recommended_working_set_bytes
        }

        fn reset_peak_memory(&self) -> Result<(), MlxMemoryProbeError> {
            self.reset_result?;
            let mut resets = self
                .resets
                .lock()
                .map_err(|_| MlxMemoryProbeError::QueryFailed("test reset lock"))?;
            *resets += 1;
            Ok(())
        }
    }

    #[test]
    fn memory_snapshot_preserves_allocator_and_working_set_measurements() {
        let probe =
            FakeMemoryProbe::from_raw([RawMemoryMeasurement::successful(11, 17, 23, Some(29))]);

        assert_eq!(
            measure_mlx_memory(&probe),
            Ok(MlxMemorySnapshot {
                active_bytes: 11,
                peak_bytes: 17,
                memory_limit_bytes: Some(23),
                metal_recommended_working_set_bytes: Some(29),
            })
        );
    }

    #[test]
    fn memory_snapshot_normalizes_zero_mlx_limit() {
        let probe = FakeMemoryProbe::from_raw([RawMemoryMeasurement::successful(11, 17, 0, None)]);

        assert_eq!(
            measure_mlx_memory(&probe),
            Ok(MlxMemorySnapshot {
                active_bytes: 11,
                peak_bytes: 17,
                memory_limit_bytes: None,
                metal_recommended_working_set_bytes: None,
            })
        );
    }

    #[test]
    fn memory_snapshot_propagates_each_raw_query_failure() {
        let error = MlxMemoryProbeError::QueryFailed("injected query failure");
        let mut cases = [
            RawMemoryMeasurement::successful(1, 2, 3, Some(4)),
            RawMemoryMeasurement::successful(1, 2, 3, Some(4)),
            RawMemoryMeasurement::successful(1, 2, 3, Some(4)),
            RawMemoryMeasurement::successful(1, 2, 3, Some(4)),
        ];
        cases[0].active_bytes = Err(error);
        cases[1].peak_bytes = Err(error);
        cases[2].memory_limit_bytes = Err(error);
        cases[3].metal_recommended_working_set_bytes = Err(error);

        for measurement in cases {
            assert_eq!(
                measure_mlx_memory(&FakeMemoryProbe::from_raw([measurement])),
                Err(error)
            );
        }
    }

    #[test]
    fn memory_snapshot_rejects_each_u64_conversion_overflow() {
        let too_large = u128::from(u64::MAX) + 1;
        let cases = [
            (
                RawMemoryMeasurement::successful(too_large, 2, 3, Some(4)),
                MlxMemoryProbeError::QueryFailed("active memory conversion"),
            ),
            (
                RawMemoryMeasurement::successful(1, too_large, 3, Some(4)),
                MlxMemoryProbeError::QueryFailed("peak memory conversion"),
            ),
            (
                RawMemoryMeasurement::successful(1, 2, too_large, Some(4)),
                MlxMemoryProbeError::QueryFailed("memory limit conversion"),
            ),
            (
                RawMemoryMeasurement::successful(1, 2, 3, Some(too_large)),
                MlxMemoryProbeError::QueryFailed("Metal working-set conversion"),
            ),
        ];

        for (measurement, expected) in cases {
            assert_eq!(
                measure_mlx_memory(&FakeMemoryProbe::from_raw([measurement])),
                Err(expected)
            );
        }
    }

    #[test]
    fn request_memory_sampler_propagates_reset_failure() {
        let probe = FakeMemoryProbe::new([MlxMemorySnapshot::default()])
            .with_reset_result(Err(MlxMemoryProbeError::ResetPeakFailed));
        let mut mlx_gate = higgs_models::mlx_exec::acquire();

        assert!(matches!(
            RequestMemorySampler::start_with_probe(MemoryPhase::Prefill, &mut mlx_gate, &probe),
            Err(MlxMemoryProbeError::ResetPeakFailed)
        ));
    }

    #[test]
    fn request_memory_sampler_records_content_free_prefill_high_water() {
        let before = MlxMemorySnapshot {
            active_bytes: 100,
            peak_bytes: 900,
            memory_limit_bytes: Some(1_000),
            metal_recommended_working_set_bytes: Some(800),
        };
        let after = MlxMemorySnapshot {
            active_bytes: 140,
            peak_bytes: 220,
            memory_limit_bytes: Some(1_000),
            metal_recommended_working_set_bytes: Some(800),
        };
        let probe = FakeMemoryProbe::new([before, after]);
        let mut mlx_gate = higgs_models::mlx_exec::acquire();

        let sampler =
            RequestMemorySampler::start_with_probe(MemoryPhase::Prefill, &mut mlx_gate, &probe)
                .unwrap();
        let sample = sampler.finish_with_probe(&probe).unwrap();

        assert_eq!(sample.phase, MemoryPhase::Prefill);
        assert_eq!(sample.active_before_bytes, 100);
        assert_eq!(sample.active_after_bytes, 140);
        assert_eq!(sample.peak_bytes, 220);
        assert_eq!(sample.active_growth_bytes, 40);
        assert_eq!(*probe.resets.lock().unwrap(), 1);
    }

    #[test]
    fn request_memory_sampler_records_decode_and_never_understates_baseline() {
        let before = MlxMemorySnapshot {
            active_bytes: 300,
            peak_bytes: 700,
            memory_limit_bytes: None,
            metal_recommended_working_set_bytes: None,
        };
        let after = MlxMemorySnapshot {
            active_bytes: 280,
            peak_bytes: 250,
            memory_limit_bytes: None,
            metal_recommended_working_set_bytes: None,
        };
        let probe = FakeMemoryProbe::new([before, after]);
        let mut mlx_gate = higgs_models::mlx_exec::acquire();

        let sampler =
            RequestMemorySampler::start_with_probe(MemoryPhase::Decode, &mut mlx_gate, &probe)
                .unwrap();
        let sample = sampler.finish_with_probe(&probe).unwrap();

        assert_eq!(sample.phase, MemoryPhase::Decode);
        assert_eq!(sample.peak_bytes, 300);
        assert_eq!(sample.active_growth_bytes, 0);
    }

    #[test]
    fn request_memory_sampler_allows_sequential_phases_on_one_gate() {
        let snapshot = MlxMemorySnapshot::default();
        let probe = FakeMemoryProbe::new([snapshot; 4]);
        let mut mlx_gate = higgs_models::mlx_exec::acquire();

        let prefill =
            RequestMemorySampler::start_with_probe(MemoryPhase::Prefill, &mut mlx_gate, &probe)
                .unwrap()
                .finish_with_probe(&probe)
                .unwrap();
        let decode =
            RequestMemorySampler::start_with_probe(MemoryPhase::Decode, &mut mlx_gate, &probe)
                .unwrap()
                .finish_with_probe(&probe)
                .unwrap();

        assert_eq!(prefill.phase, MemoryPhase::Prefill);
        assert_eq!(decode.phase, MemoryPhase::Decode);
        assert_eq!(*probe.resets.lock().unwrap(), 2);
    }

    #[test]
    fn model_memory_footprint_uses_artifact_bytes_and_checked_active_delta() -> std::io::Result<()>
    {
        let temp = TempDir::new().map_err(std::io::Error::other)?;
        fs::write(temp.path().join("model.safetensors"), [0_u8; 13])?;
        let before = MlxMemorySnapshot {
            active_bytes: 80,
            ..MlxMemorySnapshot::default()
        };
        let after = MlxMemorySnapshot {
            active_bytes: 180,
            ..MlxMemorySnapshot::default()
        };

        assert_eq!(
            ModelFootprint::from_load_measurements(temp.path(), before, after),
            Some(ModelFootprint {
                artifact_weight_bytes: 13,
                loaded_mlx_bytes: 100,
            })
        );
        assert!(ModelFootprint::from_load_measurements(temp.path(), after, before).is_none());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn model_memory_footprint_fails_closed_on_artifact_io_errors() -> std::io::Result<()> {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().map_err(std::io::Error::other)?;
        fs::write(temp.path().join("model.safetensors"), [0_u8; 13])?;
        symlink(
            temp.path().join("missing-target"),
            temp.path().join("broken.safetensors"),
        )?;
        let before = MlxMemorySnapshot::default();
        let after = MlxMemorySnapshot {
            active_bytes: 100,
            ..MlxMemorySnapshot::default()
        };

        assert_eq!(
            ModelFootprint::from_load_measurements(temp.path(), before, after),
            None,
            "a partial artifact count is unsafe when a safetensors stat fails"
        );

        let missing = temp.path().join("missing-directory");
        assert_eq!(
            ModelFootprint::from_load_measurements(&missing, before, after),
            None
        );

        let not_directory = temp.path().join("plain-file");
        fs::write(&not_directory, b"not a model directory")?;
        assert_eq!(
            ModelFootprint::from_load_measurements(&not_directory, before, after),
            None
        );
        Ok(())
    }

    #[test]
    fn model_memory_footprint_rejects_artifact_sum_overflow() {
        assert_eq!(checked_artifact_total(u64::MAX, 1), None);
    }

    #[test]
    fn escha_memory_geometry_is_20480_dense_kv_bytes_per_token() -> std::io::Result<()> {
        let temp = TempDir::new().map_err(std::io::Error::other)?;
        write_json(
            &temp.path().join("config.json"),
            &serde_json::json!({
                "model_type": "qwen3_5_moe",
                "num_hidden_layers": 40,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "full_attention_interval": 4
            }),
        )?;
        let transient = TransientPrefillEstimate {
            base_bytes: 0,
            bytes_per_prompt_token: 0,
            bytes_per_chunk_token: 0,
            max_prompt_tokens: 131_072,
            max_chunk_tokens: 4_096,
        };

        let cost = EngineCostDescription::fp16_from_model_dir(temp.path(), 0, 0, transient)
            .expect("complete Escha cache geometry");

        assert_eq!(cost.persistent_bytes_per_token, 20_480);
        assert_eq!(cost.persistent_bytes(49_152), Some(960 * 1024 * 1024));
        Ok(())
    }

    #[test]
    fn engine_memory_cost_rejects_zero_cache_geometry() -> std::io::Result<()> {
        let transient = TransientPrefillEstimate {
            base_bytes: 0,
            bytes_per_prompt_token: 0,
            bytes_per_chunk_token: 0,
            max_prompt_tokens: 1,
            max_chunk_tokens: 1,
        };

        for (layers, kv_heads, head_dim) in [(0, 2, 256), (40, 0, 256), (40, 2, 0)] {
            let temp = TempDir::new().map_err(std::io::Error::other)?;
            write_json(
                &temp.path().join("config.json"),
                &serde_json::json!({
                    "model_type": "qwen3_5_moe",
                    "num_hidden_layers": layers,
                    "num_key_value_heads": kv_heads,
                    "head_dim": head_dim,
                    "full_attention_interval": 4
                }),
            )?;

            assert_eq!(
                EngineCostDescription::fp16_from_model_dir(temp.path(), 0, 0, transient),
                None,
                "zero geometry must fail closed: layers={layers}, kv_heads={kv_heads}, head_dim={head_dim}"
            );
        }
        Ok(())
    }

    #[test]
    fn engine_memory_cost_rejects_overflow_and_out_of_domain_prefill() {
        let transient = TransientPrefillEstimate {
            base_bytes: u64::MAX,
            bytes_per_prompt_token: 1,
            bytes_per_chunk_token: 1,
            max_prompt_tokens: 8,
            max_chunk_tokens: 4,
        };
        let cost = EngineCostDescription {
            fixed_live_session_bytes: 1,
            persistent_bytes_per_token: u64::MAX,
            decode_workspace_bytes: 1,
            transient_prefill: transient,
        };

        assert_eq!(cost.persistent_bytes(2), None);
        assert_eq!(transient.estimate_bytes(1, 1), None);
        assert_eq!(transient.estimate_bytes(9, 1), None);
        assert_eq!(transient.estimate_bytes(1, 5), None);
    }

    #[test]
    fn test_requested_profile_env_aliases_parse() {
        assert_eq!(
            RequestedMlxProfile::from_env_raw(Some("ttft")),
            Ok(Some(RequestedMlxProfile::Latency))
        );
        assert_eq!(
            RequestedMlxProfile::from_env_raw(Some("tps")),
            Ok(Some(RequestedMlxProfile::Throughput))
        );
        assert_eq!(
            RequestedMlxProfile::from_env_raw(Some("baseline")),
            Ok(Some(RequestedMlxProfile::Baseline))
        );
        assert_eq!(RequestedMlxProfile::from_env_raw(None), Ok(None));
    }

    #[test]
    fn test_requested_profile_invalid_env_fails() {
        assert!(RequestedMlxProfile::from_env_raw(Some("unknown")).is_err());
    }

    #[test]
    fn test_auto_profile_resolves_to_balanced_for_small_models() {
        let metadata = ModelMetadata {
            hidden_size: Some(2048),
            num_hidden_layers: Some(24),
            ..ModelMetadata::default()
        };
        assert_eq!(
            resolve_profile_from_metadata(RequestedMlxProfile::Auto, &metadata),
            ResolvedMlxProfile::Balanced
        );
    }

    #[test]
    fn test_auto_profile_resolves_to_throughput_for_large_models() {
        let metadata = ModelMetadata {
            weight_bytes: Some(20 * 1024 * 1024 * 1024),
            ..ModelMetadata::default()
        };
        assert_eq!(
            resolve_profile_from_metadata(RequestedMlxProfile::Auto, &metadata),
            ResolvedMlxProfile::Throughput
        );
    }

    #[test]
    fn test_qwen35_low_bit_affine_clears_prefill_allocator_cache_by_default() {
        let metadata = ModelMetadata {
            model_type: Some("qwen3_5".to_owned()),
            quantization_bits: Some(1),
            ..ModelMetadata::default()
        };
        let tuning = MlxRuntimeTuning::from_profile(
            RequestedMlxProfile::Latency,
            ResolvedMlxProfile::Latency,
            &metadata,
        );
        assert!(tuning.clear_cache_after_prefill());

        // Q1 through Q4 all dequantize large projections during prefill and need
        // the allocator cache cleared to avoid multi-GB post-prefill residency.
        for bits in [1, 2, 3, 4] {
            let low_bit = ModelMetadata {
                quantization_bits: Some(bits),
                ..metadata.clone()
            };
            assert!(
                low_bit.should_clear_cache_after_prefill(),
                "bits={bits} should clear cache after prefill"
            );
        }

        // Dense (bits=0) and 8-bit re-quant paths do not materialize oversized
        // fp16 dequant projections and therefore should not trigger.
        for bits in [0, 8] {
            let dense_or_8bit = ModelMetadata {
                quantization_bits: Some(bits),
                ..metadata.clone()
            };
            assert!(
                !dense_or_8bit.should_clear_cache_after_prefill(),
                "bits={bits} should not clear cache after prefill"
            );
        }

        // Non-Qwen3.5 architectures are out of domain regardless of bits.
        let non_qwen35 = ModelMetadata {
            model_type: Some("qwen3".to_owned()),
            quantization_bits: Some(1),
            ..metadata.clone()
        };
        assert!(!non_qwen35.should_clear_cache_after_prefill());

        // Qwen3.5-MoE (eschamoe native trellis path) always clears,
        // even without traditional quantization bits, because the
        // trellis decode creates transient weight intermediates.
        let moe = ModelMetadata {
            model_type: Some("qwen3_5_moe".to_owned()),
            quantization_bits: None,
            ..metadata.clone()
        };
        assert!(moe.should_clear_cache_after_prefill());
    }

    fn write_json(path: &std::path::Path, value: &serde_json::Value) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
            std::io::Error::other(format!("failed to serialize JSON fixture: {error}"))
        })?;
        fs::write(path, bytes)
    }

    #[test]
    fn test_from_model_dir_reads_config_and_resolves_large_auto_to_throughput()
    -> std::io::Result<()> {
        let temp = TempDir::new().map_err(std::io::Error::other)?;
        let config_path = temp.path().join("config.json");
        write_json(
            &config_path,
            &serde_json::json!({
                "num_hidden_layers": 72,
                "hidden_size": 4096,
            }),
        )?;
        let tuning = MlxRuntimeTuning::from_model_dir(temp.path(), RequestedMlxProfile::Auto);
        assert_eq!(tuning.requested_profile(), RequestedMlxProfile::Auto);
        assert_eq!(tuning.resolved_profile(), ResolvedMlxProfile::Throughput);
        assert!(tuning.chunked_prefill_threshold() >= 1024);
        assert!(tuning.chunked_prefill_chunk_size() >= 1024);
        assert!(tuning.enable_mtp());
        Ok(())
    }

    #[test]
    fn test_resolve_effective_profile_auto_classification_via_model_fixture() -> std::io::Result<()>
    {
        let small = TempDir::new().map_err(std::io::Error::other)?;
        let large = TempDir::new().map_err(std::io::Error::other)?;
        let small_config = small.path().join("config.json");
        write_json(
            &small_config,
            &serde_json::json!({
                "num_hidden_layers": 20,
                "hidden_size": 2048,
            }),
        )?;
        let large_config = large.path().join("config.json");
        write_json(
            &large_config,
            &serde_json::json!({
                "num_hidden_layers": 56,
                "hidden_size": 8192,
            }),
        )?;

        assert_eq!(
            resolve_effective_mlx_profile(small.path(), RequestedMlxProfile::Auto),
            ResolvedMlxProfile::Balanced
        );
        assert_eq!(
            resolve_effective_mlx_profile(large.path(), RequestedMlxProfile::Auto),
            ResolvedMlxProfile::Throughput
        );
        Ok(())
    }

    #[test]
    fn test_resolve_runtime_tuning_applies_requested_latency_profile() -> std::io::Result<()> {
        let temp = TempDir::new().map_err(std::io::Error::other)?;
        write_json(
            &temp.path().join("config.json"),
            &serde_json::json!({
                "num_hidden_layers": 32,
                "hidden_size": 3072,
            }),
        )?;

        let tuning = resolve_runtime_tuning(temp.path(), RequestedMlxProfile::Latency);
        assert_eq!(tuning.requested_profile(), RequestedMlxProfile::Latency);
        assert_eq!(tuning.resolved_profile(), ResolvedMlxProfile::Latency);
        assert_eq!(tuning.chunked_prefill_threshold(), 3072);
        assert_eq!(tuning.chunked_prefill_chunk_size(), 768);
        assert!(!tuning.clear_cache_after_prefill());
        assert!(tuning.enable_mtp());
        Ok(())
    }

    #[test]
    fn test_parse_positive_chunked_prefill_value_defaults_on_invalid_input() {
        assert_eq!(parse_positive_chunked_prefill_value(Some("4096"), 32), 4096);
        assert_eq!(parse_positive_chunked_prefill_value(Some("0"), 32), 32);
        assert_eq!(parse_positive_chunked_prefill_value(Some("-8"), 32), 32);
        assert_eq!(parse_positive_chunked_prefill_value(Some("bad"), 32), 32);
        assert_eq!(parse_positive_chunked_prefill_value(None, 32), 32);
    }

    #[test]
    fn test_parse_mtp_draft_n_max_clamps_invalid_and_large_values() {
        assert_eq!(parse_mtp_draft_n_max(Some("1"), 3), 1);
        assert_eq!(parse_mtp_draft_n_max(Some("3"), 1), 3);
        assert_eq!(parse_mtp_draft_n_max(Some("0"), 3), 3);
        assert_eq!(parse_mtp_draft_n_max(Some("bad"), 3), 3);
        assert_eq!(parse_mtp_draft_n_max(Some("99"), 3), 8);
        assert_eq!(parse_mtp_draft_n_max(None, 3), 3);
    }

    #[test]
    fn test_default_mtp_draft_depth_uses_two_only_for_huge_models() {
        assert_eq!(default_mtp_draft_n_max(ModelSizeClass::Small), 1);
        assert_eq!(default_mtp_draft_n_max(ModelSizeClass::Medium), 1);
        assert_eq!(default_mtp_draft_n_max(ModelSizeClass::Large), 1);
        assert_eq!(default_mtp_draft_n_max(ModelSizeClass::Huge), 2);
    }

    #[cfg(unix)]
    #[test]
    fn test_model_weight_bytes_follows_hf_snapshot_symlinks() -> std::io::Result<()> {
        let temp = TempDir::new().map_err(std::io::Error::other)?;
        let blob = temp.path().join("blob");
        fs::write(&blob, [0u8; 13])?;
        std::os::unix::fs::symlink(&blob, temp.path().join("model.safetensors"))?;

        assert_eq!(model_weight_bytes(temp.path()), Some(13));
        Ok(())
    }

    #[test]
    fn test_parse_enabled_flag_ignores_unknown_values() {
        assert_eq!(parse_enabled_flag(Some("TRUE")), Some(true));
        assert_eq!(parse_enabled_flag(Some("0")), Some(false));
        assert_eq!(parse_enabled_flag(Some("maybe")), None);
        assert_eq!(parse_enabled_flag(Some("3")), None);
        assert_eq!(parse_enabled_flag(None), None);
    }

    #[test]
    fn test_paged_kv_target_respects_configured_working_set_cap() {
        let metadata = ModelMetadata {
            weight_bytes: Some(134_217_728),
            ..ModelMetadata::default()
        };

        let target = heuristic_paged_kv_target_bytes_with_max(
            &metadata,
            ModelSizeClass::Small,
            false,
            Some(134_217_728),
        );

        assert_eq!(target, 134_217_728);
    }

    #[test]
    fn test_from_model_dir_defaults_when_config_and_weights_missing() -> std::io::Result<()> {
        let temp = TempDir::new().map_err(std::io::Error::other)?;
        let missing = temp.path().join("missing-dir");
        let tuning = resolve_runtime_tuning(&missing, RequestedMlxProfile::Auto);
        assert_eq!(tuning.requested_profile(), RequestedMlxProfile::Auto);
        assert_eq!(tuning.resolved_profile(), ResolvedMlxProfile::Balanced);
        assert_eq!(tuning.chunked_prefill_threshold(), 2048);
        assert_eq!(tuning.chunked_prefill_chunk_size(), 1024);
        assert!(tuning.enable_mtp());
        Ok(())
    }

    #[test]
    fn test_from_model_dir_handles_unreadable_config_gracefully() -> std::io::Result<()> {
        let temp = TempDir::new().map_err(std::io::Error::other)?;
        fs::write(temp.path().join("config.json"), b"{not valid json")?;
        let tuning = resolve_runtime_tuning(temp.path(), RequestedMlxProfile::Throughput);
        assert_eq!(tuning.requested_profile(), RequestedMlxProfile::Throughput);
        assert_eq!(tuning.resolved_profile(), ResolvedMlxProfile::Throughput);
        assert_eq!(tuning.chunked_prefill_threshold(), 2048);
        assert!(tuning.enable_mtp());
        Ok(())
    }
}
