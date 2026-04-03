// SPDX-License-Identifier: Apache-2.0
//! SpecPrefill: Attention-based sparse prefill for MLX.

pub mod scoring;
pub mod scoring_attention;
pub mod rope;
pub mod prefill;
pub mod draft;

pub use scoring::{TokenImportance, score_tokens_uniform, compute_keep_rate};
pub use scoring_attention::score_tokens_with_attention;
pub use draft::{DraftModel, DraftModelConfig, score_with_draft, auto_select_draft_model};
pub use prefill::{sparse_prefill, cleanup_sparse_prefill, SparsePrefillState};

pub const SPEC_PREFILL_THRESHOLD: usize = 8192;
pub const SPEC_PREFILL_MAX_TOKENS: usize = 65536;
pub const DEFAULT_CHUNK_SIZE: usize = 32;
