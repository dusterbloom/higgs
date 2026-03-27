//! CoreML bridge for running a compiled `.mlpackage` on Apple Neural Engine.
//!
//! Provides `CoreMlDraftModel` implementing the speculative decode `DraftModel`
//! trait from `higgs-engine`. The model runs single-token decode steps through
//! CoreML, which dispatches to ANE/GPU/CPU as the runtime sees fit.
//!
//! KV cache is managed as explicit fp16 buffers passed to/from the bridge
//! on each prediction call.

use std::path::Path;

// ── FFI bindings to bridge/coreml/coreml_bridge.{h,m} ──────────────────

#[allow(non_camel_case_types)]
enum CoreMLHandle {}

unsafe extern "C" {
    fn coreml_load(path: *const std::ffi::c_char) -> *mut CoreMLHandle;
    fn coreml_predict(
        handle: *mut CoreMLHandle,
        input_ids: *const i32,
        position: *const i32,
        kv_cache_in: *const u8,
        kv_cache_out: *mut u8,
        logits_out: *mut u8,
    ) -> i32;
    fn coreml_get_vocab_size(handle: *mut CoreMLHandle) -> i32;
    fn coreml_get_kv_cache_size(handle: *mut CoreMLHandle) -> i32;
    fn coreml_free(handle: *mut CoreMLHandle);
}

// ── Safe wrapper ────────────────────────────────────────────────────────

/// A loaded CoreML model for single-token inference.
pub struct CoreMlModel {
    handle: *mut CoreMLHandle,
    vocab_size: usize,
    kv_cache_bytes: usize,
}

// CoreML's prediction is thread-safe (the framework handles internal locking).
#[allow(unsafe_code)]
unsafe impl Send for CoreMlModel {}

impl CoreMlModel {
    /// Load a `.mlpackage` from disk. Compiles to `.mlmodelc` on first load
    /// (cached for subsequent loads).
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
        if vocab_size < 0 || kv_cache_bytes < 0 {
            unsafe { coreml_free(handle) };
            return Err("failed to read model dimensions".into());
        }
        tracing::info!(
            vocab_size,
            kv_cache_mb = kv_cache_bytes / (1024 * 1024),
            "CoreML model loaded"
        );
        Ok(Self {
            handle,
            vocab_size: vocab_size as usize,
            kv_cache_bytes: kv_cache_bytes as usize,
        })
    }

    /// Run a single decode step. Returns fp16 logits buffer.
    #[allow(unsafe_code)]
    pub fn predict(
        &self,
        token_id: i32,
        position: i32,
        kv_cache: &[u8],
        kv_cache_out: &mut [u8],
        logits_out: &mut [u8],
    ) -> Result<(), String> {
        assert_eq!(kv_cache.len(), self.kv_cache_bytes, "kv_cache size mismatch");
        assert_eq!(kv_cache_out.len(), self.kv_cache_bytes, "kv_cache_out size mismatch");
        assert_eq!(
            logits_out.len(),
            self.vocab_size * 2,
            "logits_out size mismatch (expected {} fp16 elements)",
            self.vocab_size
        );

        let ret = unsafe {
            coreml_predict(
                self.handle,
                &token_id,
                &position,
                kv_cache.as_ptr(),
                kv_cache_out.as_mut_ptr(),
                logits_out.as_mut_ptr(),
            )
        };
        if ret != 0 {
            return Err("coreml_predict failed".into());
        }
        Ok(())
    }

    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub const fn kv_cache_bytes(&self) -> usize {
        self.kv_cache_bytes
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

// ── Draft model for speculative decoding ────────────────────────────────

/// CoreML-backed draft model for speculative decoding.
///
/// Runs single-token decode on ANE via CoreML, with explicit KV cache
/// management (prefill, advance, rollback).
pub struct CoreMlDraftModel {
    model: CoreMlModel,
    /// Current KV cache state (fp16 buffer).
    kv_cache: Vec<u8>,
    /// Snapshot saved before each `draft()` call, restored on `rollback()`.
    kv_checkpoint: Option<Vec<u8>>,
    /// Current sequence position (how many tokens have been processed).
    seq_pos: usize,
    /// Saved position for rollback.
    checkpoint_pos: Option<usize>,
    /// Reusable buffer for logits output (fp16).
    logits_buf: Vec<u8>,
    /// Reusable buffer for kv_cache output.
    kv_out_buf: Vec<u8>,
}

impl CoreMlDraftModel {
    /// Load a CoreML draft model from a `.mlpackage` directory.
    pub fn load(path: &Path) -> Result<Self, String> {
        let model = CoreMlModel::load(path)?;
        let kv_bytes = model.kv_cache_bytes();
        let vocab = model.vocab_size();
        Ok(Self {
            kv_cache: vec![0u8; kv_bytes],
            kv_checkpoint: None,
            seq_pos: 0,
            checkpoint_pos: None,
            logits_buf: vec![0u8; vocab * 2], // fp16
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
}

impl CoreMlDraftModel {
    /// Prefill — process all prompt tokens through the model.
    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), String> {
        // Reset state
        self.kv_cache.fill(0);
        self.kv_checkpoint = None;
        self.checkpoint_pos = None;
        self.seq_pos = 0;

        // Process each prompt token
        for &token in prompt_tokens {
            self.model.predict(
                token as i32,
                self.seq_pos as i32,
                &self.kv_cache,
                &mut self.kv_out_buf,
                &mut self.logits_buf,
            )?;
            // Swap kv buffers (the output becomes the new state)
            std::mem::swap(&mut self.kv_cache, &mut self.kv_out_buf);
            self.seq_pos += 1;
        }
        Ok(())
    }

    /// Draft `num_draft` tokens greedily starting from `last_token_id`.
    pub fn draft(&mut self, last_token_id: u32, num_draft: usize) -> Result<Vec<u32>, String> {
        // Save checkpoint
        self.kv_checkpoint = Some(self.kv_cache.clone());
        self.checkpoint_pos = Some(self.seq_pos);

        let mut tokens = Vec::with_capacity(num_draft);
        let mut current = last_token_id;

        for _ in 0..num_draft {
            self.model.predict(
                current as i32,
                self.seq_pos as i32,
                &self.kv_cache,
                &mut self.kv_out_buf,
                &mut self.logits_buf,
            )?;
            std::mem::swap(&mut self.kv_cache, &mut self.kv_out_buf);
            self.seq_pos += 1;

            let next = Self::argmax_fp16(&self.logits_buf, self.model.vocab_size());
            tokens.push(next);
            current = next;
        }
        Ok(tokens)
    }

    /// Advance — keep the first `n` drafted tokens in the cache.
    pub fn advance(&mut self, n: usize) -> Result<(), String> {
        if let Some(pre_pos) = self.checkpoint_pos.take() {
            // Roll back to checkpoint, then advance by n
            if let Some(checkpoint) = self.kv_checkpoint.take() {
                // We need to re-run n tokens from the checkpoint state.
                // But we already have the cache for pos = pre_pos + num_drafted.
                // For advance(n), the correct state is at pre_pos + n.
                // Since we ran sequentially, cache at pre_pos + n is valid only
                // if n == num_drafted. Otherwise we need to rollback and replay.
                //
                // Optimization: if n tokens were accepted, the cache is already
                // correct up to pre_pos + n. Just trim seq_pos.
                self.seq_pos = pre_pos + n;
                // The kv_cache already contains valid entries for all positions
                // up to the full draft length. Positions > seq_pos will be
                // overwritten on next predict, so it's safe to leave them.
                drop(checkpoint);
            }
        }
        self.kv_checkpoint = None;
        Ok(())
    }

    /// Rollback — restore to state before the last `draft()` call.
    pub fn rollback(&mut self) -> Result<(), String> {
        if let Some(checkpoint) = self.kv_checkpoint.take() {
            self.kv_cache = checkpoint;
        }
        if let Some(pos) = self.checkpoint_pos.take() {
            self.seq_pos = pos;
        }
        Ok(())
    }
}
