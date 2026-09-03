//! Engine crate for Higgs local inference, including MLX-backed MTP decoding.

pub mod batch_engine;
pub mod cache;
pub mod chat_template;
pub mod constrained;
pub(crate) mod decode;
pub mod disk_prefix_store;
pub mod engine;
pub mod error;
pub mod mlx_tuning;
pub mod model_loader;
pub mod mtp;
pub mod paged_prefix_cache;
pub mod prompt_cache;
pub mod prune;
pub mod prune_eval;
pub mod reasoning_parser;
pub mod runtime_identity;
pub mod scheduler;
pub mod simple;
pub mod tool_parser;

pub use mlx_tuning::{
    EngineCostDescription, LoaderWorkspaceKind, MemoryPhase, MlxMemoryProbeError,
    MlxMemorySnapshot, ModelFootprint, ModelLoadEstimate, ModelLoadEstimateError,
    RequestMemoryHighWater, RequestMemorySampler, TransientPrefillEstimate, model_load_estimate,
};
pub use simple::CacheResidency;
pub use tokenizers;
