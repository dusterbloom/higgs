//! CoreML bridge for running a compiled `.mlpackage` on Apple Neural Engine.
//!
//! Provides `CoreMlDraftModel` for speculative decoding. Supports two modes:
//!
//! - **Stateful** (MLState, macOS 15+): KV cache lives on-device, no bus copies.
//! - **Stateless** (explicit KV cache): fp16 KV buffers passed per predict.
//!   With MAX_SEQ=32 the KV cache is only 3.5 MB (~7ms copy), enabling
//!   CPU_AND_NE compute units for true ANE utilization without GPU contention.
//!   When positions are exhausted, auto-refills from recent token history.
//!
//! Rollback for speculative decoding: stateful resets position (causal mask
//! hides stale entries); stateless restores the KV buffer snapshot.

use std::path::Path;

// ── FFI bindings to bridge/coreml/coreml_bridge.{h,m} ──────────────────

#[allow(non_camel_case_types)]
enum CoreMLHandle {}

#[allow(non_camel_case_types)]
enum CoreMLState {}

unsafe extern "C" {
    fn coreml_load(path: *const std::ffi::c_char) -> *mut CoreMLHandle;
    fn coreml_get_vocab_size(handle: *mut CoreMLHandle) -> i32;
    fn coreml_get_kv_cache_size(handle: *mut CoreMLHandle) -> i32;
    fn coreml_get_max_seq(handle: *mut CoreMLHandle) -> i32;
    fn coreml_is_stateful(handle: *mut CoreMLHandle) -> i32;
    fn coreml_free(handle: *mut CoreMLHandle);

    // Stateful API
    fn coreml_new_state(handle: *mut CoreMLHandle) -> *mut CoreMLState;
    fn coreml_free_state(state: *mut CoreMLState);
    fn coreml_predict_stateful(
        handle: *mut CoreMLHandle,
        state: *mut CoreMLState,
        input_ids: *const i32,
        position: *const i32,
        logits_out: *mut u8,
    ) -> i32;

    // Legacy stateless API
    fn coreml_predict(
        handle: *mut CoreMLHandle,
        input_ids: *const i32,
        position: *const i32,
        kv_cache_in: *const u8,
        kv_cache_out: *mut u8,
        logits_out: *mut u8,
    ) -> i32;
}

// ── Safe wrapper ────────────────────────────────────────────────────────

/// A loaded CoreML model for single-token inference.
pub struct CoreMlModel {
    handle: *mut CoreMLHandle,
    vocab_size: usize,
    kv_cache_bytes: usize,
    max_seq: usize,
    is_stateful: bool,
}

#[allow(unsafe_code)]
unsafe impl Send for CoreMlModel {}

impl CoreMlModel {
    /// Load a `.mlpackage` from disk. Compiles to `.mlmodelc` on first load.
    #[allow(unsafe_code)]
    pub fn load(path: &Path) -> Result<Self, String> {
        let c_path = std::ffi::CString::new(path.to_str().ok_or("invalid path")?)
            .map_err(|e| format!("CString: {e}"))?;
        let handle = unsafe { coreml_load(c_path.as_ptr()) };
        if handle.is_null() {
            return Err(format!("coreml_load failed for {}", path.display()));
        }
        let vocab_size = unsafe { coreml_get_vocab_size(handle) };
        let kv_cache_bytes = unsafe { coreml_get_kv_cache_size(handle) };
        let max_seq = unsafe { coreml_get_max_seq(handle) };
        let is_stateful = unsafe { coreml_is_stateful(handle) } != 0;
        if vocab_size < 0 {
            unsafe { coreml_free(handle) };
            return Err("failed to read model dimensions".into());
        }
        tracing::info!(
            vocab_size,
            is_stateful,
            kv_cache_bytes,
            max_seq,
            "CoreML model loaded"
        );
        Ok(Self {
            handle,
            vocab_size: vocab_size as usize,
            kv_cache_bytes: kv_cache_bytes as usize,
            max_seq: max_seq as usize,
            is_stateful,
        })
    }

    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub const fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub const fn is_stateful(&self) -> bool {
        self.is_stateful
    }
}

impl Drop for CoreMlModel {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { coreml_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

// ── MLState wrapper ─────────────────────────────────────────────────────

struct MlState {
    ptr: *mut CoreMLState,
}

#[allow(unsafe_code)]
unsafe impl Send for MlState {}

impl MlState {
    #[allow(unsafe_code)]
    fn new(handle: *mut CoreMLHandle) -> Result<Self, String> {
        let ptr = unsafe { coreml_new_state(handle) };
        if ptr.is_null() {
            return Err("coreml_new_state failed".into());
        }
        Ok(Self { ptr })
    }
}

impl Drop for MlState {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { coreml_free_state(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

// ── Draft model for speculative decoding ────────────────────────────────

/// CoreML-backed draft model for speculative decoding.
///
/// If the model is stateful (MLState), the KV cache lives on-device and
/// rollback is handled by position tracking + causal mask (no buffer copy).
///
/// Falls back to explicit KV cache buffers for non-stateful models.
pub struct CoreMlDraftModel {
    model: CoreMlModel,
    /// Current MLState (stateful models only).
    state: Option<MlState>,
    /// Current sequence position (within KV cache, 0..max_seq).
    seq_pos: usize,
    /// Max sequence length for the KV cache (stateless models only, 0 = unlimited).
    max_seq: usize,
    /// All confirmed tokens so far (for re-prefill context).
    token_history: Vec<u32>,
    /// The last_token_id from the current draft (committed on advance, discarded on rollback).
    pending_token: Option<u32>,
    /// Tokens from the last `draft()` call (for pushing to history on `advance`).
    last_draft_tokens: Vec<u32>,
    /// How many context tokens to use on re-prefill.
    refill_context: usize,
    /// Saved position for rollback.
    checkpoint_pos: Option<usize>,
    /// Reusable buffer for logits output (fp16).
    logits_buf: Vec<u8>,

    // Fields for non-stateful models
    kv_cache: Vec<u8>,
    kv_checkpoint: Option<Vec<u8>>,
    kv_out_buf: Vec<u8>,
}

impl CoreMlDraftModel {
    /// Load a CoreML draft model from a `.mlpackage` directory.
    pub fn load(path: &Path) -> Result<Self, String> {
        let model = CoreMlModel::load(path)?;
        let vocab = model.vocab_size();
        let is_stateful = model.is_stateful();
        let max_seq = model.max_seq();

        let state = if is_stateful {
            Some(MlState::new(model.handle)?)
        } else {
            None
        };

        let kv_bytes = model.kv_cache_bytes;
        // For re-prefill: keep enough context to maintain quality.
        // With MAX_SEQ=32, use half for context leaves half for drafting.
        let refill_context = if max_seq > 0 { max_seq / 2 } else { 0 };

        Ok(Self {
            logits_buf: vec![0u8; vocab * 2], // fp16
            seq_pos: 0,
            max_seq,
            token_history: Vec::new(),
            pending_token: None,
            last_draft_tokens: Vec::new(),
            refill_context,
            checkpoint_pos: None,
            state,
            kv_cache: vec![0u8; kv_bytes],
            kv_checkpoint: None,
            kv_out_buf: vec![0u8; kv_bytes],
            model,
        })
    }

    /// Sample argmax from fp16 logits buffer.
    fn argmax_fp16(logits: &[u8], vocab_size: usize) -> u32 {
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..vocab_size {
            let offset = i * 2;
            let bits = u16::from_le_bytes([logits[offset], logits[offset + 1]]);
            let val = half::f16::from_bits(bits).to_f32();
            if val > best_val {
                best_val = val;
                best_idx = i as u32;
            }
        }
        best_idx
    }

    /// Run a single predict step (dispatches to stateful or legacy).
    #[allow(unsafe_code)]
    fn predict_one(&mut self, token_id: i32, position: i32) -> Result<(), String> {
        if let Some(ref state) = self.state {
            let ret = unsafe {
                coreml_predict_stateful(
                    self.model.handle,
                    state.ptr,
                    &token_id,
                    &position,
                    self.logits_buf.as_mut_ptr(),
                )
            };
            if ret != 0 {
                return Err("coreml_predict_stateful failed".into());
            }
        } else {
            let ret = unsafe {
                coreml_predict(
                    self.model.handle,
                    &token_id,
                    &position,
                    self.kv_cache.as_ptr(),
                    self.kv_out_buf.as_mut_ptr(),
                    self.logits_buf.as_mut_ptr(),
                )
            };
            if ret != 0 {
                return Err("coreml_predict failed".into());
            }
            std::mem::swap(&mut self.kv_cache, &mut self.kv_out_buf);
        }
        Ok(())
    }
}

impl CoreMlDraftModel {
    /// Prefill — process all prompt tokens through the model.
    ///
    /// For stateless models with limited MAX_SEQ, only feeds the last
    /// `max_seq` tokens (at most) to stay within KV cache bounds.
    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), String> {
        if self.model.is_stateful() {
            self.state = Some(MlState::new(self.model.handle)?);
        } else {
            self.kv_cache.fill(0);
        }
        self.kv_checkpoint = None;
        self.checkpoint_pos = None;
        self.pending_token = None;
        self.last_draft_tokens.clear();
        self.seq_pos = 0;
        self.token_history.clear();

        // For stateless models with small MAX_SEQ, only seed with the tail
        let tokens = if self.max_seq > 0 && prompt_tokens.len() >= self.max_seq {
            let start = prompt_tokens.len().saturating_sub(self.max_seq);
            &prompt_tokens[start..]
        } else {
            prompt_tokens
        };

        for &token in tokens {
            self.predict_one(token as i32, self.seq_pos as i32)?;
            self.seq_pos += 1;
            self.token_history.push(token);
        }
        Ok(())
    }

    /// Re-prefill the model when position limit is reached.
    /// Resets KV cache and feeds the last `refill_context` tokens from history.
    fn refill(&mut self) -> Result<(), String> {
        self.kv_cache.fill(0);
        self.seq_pos = 0;
        self.kv_checkpoint = None;
        self.checkpoint_pos = None;

        let ctx = self.refill_context.min(self.token_history.len());
        let start = self.token_history.len().saturating_sub(ctx);
        let context_tokens: Vec<u32> = self.token_history[start..].to_vec();

        for &token in &context_tokens {
            self.predict_one(token as i32, self.seq_pos as i32)?;
            self.seq_pos += 1;
        }

        tracing::debug!(
            seq_pos = self.seq_pos,
            max_seq = self.max_seq,
            history_len = self.token_history.len(),
            "re-prefilled draft model"
        );
        Ok(())
    }

    /// Draft `num_draft` tokens greedily starting from `last_token_id`.
    ///
    /// Checkpoint is saved BEFORE processing `last_token_id` so rollback
    /// correctly undoes it. If the position limit would be exceeded,
    /// a re-prefill is triggered first.
    pub fn draft(&mut self, last_token_id: u32, num_draft: usize) -> Result<Vec<u32>, String> {
        self.pending_token = Some(last_token_id);

        // Check if we need to re-prefill (need room for last_token_id + num_draft)
        if self.max_seq > 0 && self.seq_pos + 1 + num_draft > self.max_seq {
            self.refill()?;
        }

        // Save checkpoint BEFORE processing last_token_id (matches DraftModel contract)
        self.checkpoint_pos = Some(self.seq_pos);
        if !self.model.is_stateful() {
            self.kv_checkpoint = Some(self.kv_cache.clone());
        }

        // Process last_token_id + draft tokens in the same loop (original pattern)
        let mut tokens = Vec::with_capacity(num_draft);
        let mut current = last_token_id;

        for _ in 0..num_draft {
            if self.max_seq > 0 && self.seq_pos >= self.max_seq {
                break;
            }
            self.predict_one(current as i32, self.seq_pos as i32)?;
            self.seq_pos += 1;

            let next = Self::argmax_fp16(&self.logits_buf, self.model.vocab_size());
            tokens.push(next);
            current = next;
        }

        self.last_draft_tokens = tokens.clone();
        Ok(tokens)
    }

    /// Advance — keep the first `n` drafted tokens in the cache.
    ///
    /// Commits `last_token_id` + first `n` draft tokens to history for
    /// future re-prefill context.
    pub fn advance(&mut self, n: usize) -> Result<(), String> {
        // Commit the verified token
        if let Some(tok) = self.pending_token.take() {
            self.token_history.push(tok);
        }
        // Commit accepted draft tokens
        let accepted = n.min(self.last_draft_tokens.len());
        self.token_history
            .extend_from_slice(&self.last_draft_tokens[..accepted]);
        self.last_draft_tokens.clear();

        if let Some(pre_pos) = self.checkpoint_pos.take() {
            self.seq_pos = pre_pos + n;
        }
        self.kv_checkpoint = None;
        Ok(())
    }

    /// Rollback — restore to state before the last `draft()` call.
    pub fn rollback(&mut self) -> Result<(), String> {
        self.pending_token = None;
        self.last_draft_tokens.clear();
        if let Some(pos) = self.checkpoint_pos.take() {
            self.seq_pos = pos;
        }
        if let Some(checkpoint) = self.kv_checkpoint.take() {
            self.kv_cache = checkpoint;
        }
        Ok(())
    }
}
