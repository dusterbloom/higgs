// SPDX-License-Identifier: Apache-2.0
//! SpecPrefill: Attention-based sparse prefill for MLX.
//!
//! Reduces TTFT on long prompts by using a small draft model to identify
//! important tokens, then prefilling only those tokens on the target model
//! while preserving original positional encoding via manual RoPE.
//!
//! Based on arxiv.org/abs/2502.02789 and oMLX implementation.

pub mod draft;
pub mod prefill;
pub mod rope;
pub mod scoring;
pub mod scoring_attention;

pub use scoring::{compute_keep_rate, score_tokens_uniform, TokenImportance};

pub use rope::{manual_rope, manual_rope_with_freqs, OffsetAdjustedRoPE, PositionMappedRoPE};

pub use prefill::{cleanup_sparse_prefill, sparse_prefill, SparsePrefillState};

pub use draft::{auto_select_draft_model, score_with_draft, DraftModel, DraftModelConfig};

/// Compute keep rate threshold for enabling SpecPrefill.
pub const SPEC_PREFILL_THRESHOLD: usize = 8192;

/// Maximum context length for SpecPrefill (beyond this, overhead outweighs benefits).
pub const SPEC_PREFILL_MAX_TOKENS: usize = 65536;

/// Default chunk size for token selection.
pub const DEFAULT_CHUNK_SIZE: usize = 32;

/// Keep rate presets for different context lengths.
pub const KEEP_RATE_PRESETS: &[(usize, &str)] = &[
    (8192, "No pruning (<8k)"),
    (16384, "Aggressive (~3x, 30%)"),
    (32768, "Balanced (~4x, 25%)"),
    (65536, "Conservative (~5x, 20%)"),
];
