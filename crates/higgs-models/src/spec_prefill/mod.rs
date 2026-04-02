// SPDX-License-Identifier: Apache-2.0
//! SpecPrefill: Attention-based sparse prefill for MLX.
//!
//! Reduces TTFT on long prompts by using a small draft model to identify
//! important tokens, then prefilling only those tokens on the target model
//! while preserving original positional encoding via manual RoPE.
//!
//! Based on arxiv.org/abs/2502.02789 and oMLX implementation.

pub mod scoring;
// pub mod selection;  // TODO
// pub mod rope;       // TODO
// pub mod prefill;    // TODO

pub use scoring::{compute_keep_rate, score_tokens_uniform, ScoringConfig, TokenImportance};

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
