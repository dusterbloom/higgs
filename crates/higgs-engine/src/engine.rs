pub use higgs_models::{TokenLogprobInfo, TopLogprobEntry};

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::mlx_tuning::{MemoryPhase, RequestMemoryHighWater};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestExecutionPath {
    Cold,
    RetainedSuffix,
    RadixHit,
}

/// Content-free allocator and cache facts completed by one inference worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestAllocationReceipt {
    prefill: RequestMemoryHighWater,
    decode: Option<RequestMemoryHighWater>,
    retained_bytes: u64,
    radix_growth_bytes: u64,
    execution_path: RequestExecutionPath,
    full_prompt_tokens: u64,
    suffix_tokens: u64,
}

impl RequestAllocationReceipt {
    #[must_use]
    pub const fn new(
        prefill: RequestMemoryHighWater,
        decode: Option<RequestMemoryHighWater>,
        retained_bytes: u64,
        radix_growth_bytes: u64,
        execution_path: RequestExecutionPath,
        full_prompt_tokens: u64,
        suffix_tokens: u64,
    ) -> Self {
        Self {
            prefill,
            decode,
            retained_bytes,
            radix_growth_bytes,
            execution_path,
            full_prompt_tokens,
            suffix_tokens,
        }
    }

    #[must_use]
    pub const fn prefill(self) -> RequestMemoryHighWater {
        self.prefill
    }

    #[must_use]
    pub const fn decode(self) -> Option<RequestMemoryHighWater> {
        self.decode
    }

    #[must_use]
    pub const fn execution_path(self) -> RequestExecutionPath {
        self.execution_path
    }

    #[must_use]
    pub const fn full_prompt_tokens(self) -> u64 {
        self.full_prompt_tokens
    }

    #[must_use]
    pub const fn suffix_tokens(self) -> u64 {
        self.suffix_tokens
    }

    #[must_use]
    pub fn observed_peak_bytes(self) -> u64 {
        self.decode.map_or(self.prefill.peak_bytes, |decode| {
            decode.peak_bytes.max(self.prefill.peak_bytes)
        })
    }

    #[must_use]
    pub fn observed_retained_bytes(self) -> u64 {
        self.retained_bytes.saturating_add(self.radix_growth_bytes)
    }

    #[must_use]
    pub fn allocation_bearing(self) -> bool {
        self.prefill.peak_bytes > self.prefill.active_before_bytes
            || self.prefill.active_growth_bytes > 0
            || self.decode.is_some_and(|decode| {
                decode.peak_bytes > decode.active_before_bytes || decode.active_growth_bytes > 0
            })
    }
}

struct RequestAllocationState {
    prefill: Option<RequestMemoryHighWater>,
    decode: Option<RequestMemoryHighWater>,
    retained_bytes: Option<u64>,
    radix_growth_bytes: u64,
    execution: Option<(RequestExecutionPath, u64, u64)>,
    valid: bool,
}

impl Default for RequestAllocationState {
    fn default() -> Self {
        Self {
            prefill: None,
            decode: None,
            retained_bytes: None,
            radix_growth_bytes: 0,
            execution: None,
            valid: true,
        }
    }
}

thread_local! {
    static REQUEST_ALLOCATION_CAPTURE: RefCell<Option<Rc<RefCell<RequestAllocationState>>>> = const { RefCell::new(None) };
}

/// Worker-thread capture installed beside the reservation and stop guards.
/// Dropping it always restores the prior slot, preventing stale receipts when
/// Tokio reuses a blocking worker thread.
pub struct RequestAllocationCapture {
    previous: Option<Weak<RefCell<RequestAllocationState>>>,
    state: Rc<RefCell<RequestAllocationState>>,
    restored: bool,
}

impl RequestAllocationCapture {
    #[must_use]
    pub fn start() -> Self {
        let state = Rc::new(RefCell::new(RequestAllocationState::default()));
        let previous = REQUEST_ALLOCATION_CAPTURE.with(|slot| {
            slot.replace(Some(Rc::clone(&state)))
                .map(|previous| Rc::downgrade(&previous))
        });
        Self {
            previous,
            state,
            restored: false,
        }
    }

    /// Return a receipt only when a complete prefill phase was measured.
    /// Routes call this only after successful engine completion.
    #[must_use]
    pub fn finish(mut self) -> Option<RequestAllocationReceipt> {
        self.restore();
        let state = self.state.borrow();
        if !state.valid {
            return None;
        }
        let (execution_path, full_prompt_tokens, suffix_tokens) = state.execution?;
        Some(RequestAllocationReceipt::new(
            state.prefill?,
            state.decode,
            state.retained_bytes.unwrap_or(0),
            state.radix_growth_bytes,
            execution_path,
            full_prompt_tokens,
            suffix_tokens,
        ))
    }

    fn restore(&mut self) {
        if !self.restored {
            REQUEST_ALLOCATION_CAPTURE.with(|slot| {
                let owns_slot = slot
                    .borrow()
                    .as_ref()
                    .is_some_and(|current| Rc::ptr_eq(current, &self.state));
                if owns_slot {
                    slot.replace(self.previous.take().and_then(|previous| previous.upgrade()));
                }
            });
            self.restored = true;
        }
    }
}

impl Drop for RequestAllocationCapture {
    fn drop(&mut self) {
        self.restore();
    }
}

pub(crate) fn record_request_memory_high_water(sample: RequestMemoryHighWater) {
    REQUEST_ALLOCATION_CAPTURE.with(|slot| {
        if let Some(state) = slot.borrow().as_ref() {
            let mut state = state.borrow_mut();
            match sample.phase {
                MemoryPhase::Prefill => state.prefill = Some(sample),
                MemoryPhase::Decode => state.decode = Some(sample),
            }
        }
    });
}

pub(crate) fn record_request_retained_bytes(bytes: u64) {
    REQUEST_ALLOCATION_CAPTURE.with(|slot| {
        if let Some(state) = slot.borrow().as_ref() {
            state.borrow_mut().retained_bytes = Some(bytes);
        }
    });
}

pub(crate) fn record_request_radix_growth_bytes(bytes: u64) {
    REQUEST_ALLOCATION_CAPTURE.with(|slot| {
        if let Some(state) = slot.borrow().as_ref() {
            let mut state = state.borrow_mut();
            state.radix_growth_bytes = state.radix_growth_bytes.saturating_add(bytes);
        }
    });
}

pub(crate) fn record_request_execution(
    path: RequestExecutionPath,
    full_prompt_tokens: u64,
    suffix_tokens: u64,
) {
    REQUEST_ALLOCATION_CAPTURE.with(|slot| {
        if let Some(state) = slot.borrow().as_ref() {
            state.borrow_mut().execution = Some((path, full_prompt_tokens, suffix_tokens));
        }
    });
}

pub(crate) fn invalidate_request_allocation_capture() {
    REQUEST_ALLOCATION_CAPTURE.with(|slot| {
        if let Some(state) = slot.borrow().as_ref() {
            state.borrow_mut().valid = false;
        }
    });
}

/// Output from a generation request.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub finish_reason: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub token_logprobs: Option<Vec<TokenLogprobInfo>>,
    /// Reasoning content, split from `text` at the `</think>` token boundary
    /// when thinking mode is enabled. `None` when thinking is off or the split
    /// was not performed by the engine (callers may fall back to string parsing).
    ///
    /// The boundary token is excluded from both fields so it never surfaces in
    /// streamed/returned text. See `SimpleEngine::generate_inner`.
    pub reasoning_content: Option<String>,
    /// Prompt tokens served from the radix prefix cache, mirroring
    /// `PrefillProgress.cached` on the streaming path. `0` when the route
    /// doesn't track prefix-cache reuse (e.g. `DFlash`, batch engine).
    pub cached_prompt_tokens: u32,
}

/// Prefill progress for one streaming request, in absolute prompt tokens.
///
/// `processed` counts cached + prefilled tokens, so `processed / total` is
/// directly displayable; `cached` exposes the prefix-cache hit separately.
/// Matches the semantics of llama.cpp's `prompt_progress` SSE field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillProgress {
    pub processed: u32,
    pub cached: u32,
    pub total: u32,
}

/// Output from a streaming generation step.
#[derive(Debug, Clone)]
pub struct StreamingOutput {
    pub new_text: String,
    pub finished: bool,
    pub finish_reason: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub token_logprob: Option<TokenLogprobInfo>,
    /// Set on progress-only events emitted during chunked prefill
    /// (`new_text` is empty, `completion_tokens` is 0). `None` on token
    /// outputs.
    pub prefill_progress: Option<PrefillProgress>,
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    fn memory_sample(
        phase: crate::mlx_tuning::MemoryPhase,
        peak_bytes: u64,
    ) -> crate::mlx_tuning::RequestMemoryHighWater {
        crate::mlx_tuning::RequestMemoryHighWater {
            phase,
            active_before_bytes: 100,
            active_after_bytes: 120,
            peak_bytes,
            active_growth_bytes: 20,
        }
    }

    #[test]
    fn request_allocation_capture_is_cleared_before_worker_thread_reuse() {
        let first = RequestAllocationCapture::start();
        record_request_execution(RequestExecutionPath::Cold, 10, 10);
        record_request_memory_high_water(memory_sample(
            crate::mlx_tuning::MemoryPhase::Prefill,
            200,
        ));
        let receipt = first.finish().expect("complete prefill receipt");
        assert_eq!(receipt.observed_peak_bytes(), 200);

        let reused_thread = RequestAllocationCapture::start();
        assert!(reused_thread.finish().is_none());
    }

    #[test]
    fn out_of_order_nested_capture_drop_cannot_restore_stale_state() {
        let outer = RequestAllocationCapture::start();
        let inner = RequestAllocationCapture::start();
        drop(outer);
        drop(inner);

        record_request_memory_high_water(memory_sample(
            crate::mlx_tuning::MemoryPhase::Prefill,
            999,
        ));
        let reused_thread = RequestAllocationCapture::start();
        assert!(reused_thread.finish().is_none());
    }

    #[test]
    fn request_allocation_capture_requires_complete_prefill_and_uses_actual_cache_bytes() {
        let capture = RequestAllocationCapture::start();
        record_request_execution(RequestExecutionPath::RadixHit, 10, 2);
        record_request_memory_high_water(memory_sample(
            crate::mlx_tuning::MemoryPhase::Prefill,
            200,
        ));
        record_request_memory_high_water(memory_sample(
            crate::mlx_tuning::MemoryPhase::Decode,
            250,
        ));
        record_request_retained_bytes(512);
        record_request_radix_growth_bytes(128);

        let receipt = capture.finish().expect("complete request receipt");
        assert_eq!(receipt.observed_peak_bytes(), 250);
        assert_eq!(receipt.observed_retained_bytes(), 640);
        assert_eq!(receipt.execution_path(), RequestExecutionPath::RadixHit);
        assert_eq!(receipt.full_prompt_tokens(), 10);
        assert_eq!(receipt.suffix_tokens(), 2);
        assert!(receipt.allocation_bearing());
    }

    #[test]
    fn generation_output_construction_and_field_access() {
        let output = GenerationOutput {
            text: "Hello world".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 10,
            completion_tokens: 5,
            token_logprobs: None,
            reasoning_content: None,
            cached_prompt_tokens: 0,
        };
        assert_eq!(output.text, "Hello world");
        assert_eq!(output.finish_reason, "stop");
        assert_eq!(output.prompt_tokens, 10);
        assert_eq!(output.completion_tokens, 5);
    }

    #[test]
    fn generation_output_empty_defaults() {
        let output = GenerationOutput {
            text: String::new(),
            finish_reason: "length".to_owned(),
            prompt_tokens: 0,
            completion_tokens: 0,
            token_logprobs: None,
            reasoning_content: None,
            cached_prompt_tokens: 0,
        };
        assert!(output.text.is_empty());
        assert_eq!(output.prompt_tokens, 0);
        assert_eq!(output.completion_tokens, 0);
    }

    #[test]
    fn streaming_output_finished_true() {
        let output = StreamingOutput {
            new_text: "done".to_owned(),
            finished: true,
            finish_reason: Some("stop".to_owned()),
            prompt_tokens: 20,
            completion_tokens: 15,
            token_logprob: None,
            prefill_progress: None,
        };
        assert!(output.finished);
        assert_eq!(output.finish_reason.as_deref(), Some("stop"));
        assert_eq!(output.new_text, "done");
    }

    #[test]
    fn streaming_output_finished_false() {
        let output = StreamingOutput {
            new_text: "partial".to_owned(),
            finished: false,
            finish_reason: None,
            prompt_tokens: 20,
            completion_tokens: 3,
            token_logprob: None,
            prefill_progress: None,
        };
        assert!(!output.finished);
        assert!(output.finish_reason.is_none());
    }

    #[test]
    fn streaming_output_empty_text_zero_tokens() {
        let output = StreamingOutput {
            new_text: String::new(),
            finished: true,
            finish_reason: Some("length".to_owned()),
            prompt_tokens: 0,
            completion_tokens: 0,
            token_logprob: None,
            prefill_progress: None,
        };
        assert!(output.new_text.is_empty());
        assert_eq!(output.prompt_tokens, 0);
        assert_eq!(output.completion_tokens, 0);
    }

    #[test]
    fn generation_output_clone() {
        let output = GenerationOutput {
            text: "test".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 5,
            completion_tokens: 3,
            token_logprobs: None,
            reasoning_content: None,
            cached_prompt_tokens: 0,
        };
        let cloned = output.clone();
        assert_eq!(cloned.text, output.text);
        assert_eq!(cloned.finish_reason, output.finish_reason);
    }

    #[test]
    fn streaming_output_clone() {
        let output = StreamingOutput {
            new_text: "stream".to_owned(),
            finished: false,
            finish_reason: None,
            prompt_tokens: 10,
            completion_tokens: 2,
            token_logprob: None,
            prefill_progress: None,
        };
        let cloned = output.clone();
        assert_eq!(cloned.new_text, output.new_text);
        assert_eq!(cloned.finished, output.finished);
        assert_eq!(cloned.finish_reason, output.finish_reason);
    }

    #[test]
    fn generation_output_debug_format() {
        let output = GenerationOutput {
            text: "hi".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            token_logprobs: None,
            reasoning_content: None,
            cached_prompt_tokens: 0,
        };
        let debug_str = format!("{output:?}");
        assert!(debug_str.contains("GenerationOutput"));
        assert!(debug_str.contains("hi"));
    }

    #[test]
    fn streaming_output_debug_format() {
        let output = StreamingOutput {
            new_text: "token".to_owned(),
            finished: true,
            finish_reason: Some("stop".to_owned()),
            prompt_tokens: 5,
            completion_tokens: 10,
            token_logprob: None,
            prefill_progress: None,
        };
        let debug_str = format!("{output:?}");
        assert!(debug_str.contains("StreamingOutput"));
        assert!(debug_str.contains("token"));
    }

    #[test]
    fn generation_output_with_logprobs() {
        let output = GenerationOutput {
            text: "hello".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 5,
            completion_tokens: 1,
            token_logprobs: Some(vec![TokenLogprobInfo {
                token_id: 42,
                logprob: -0.5,
                top_logprobs: vec![
                    TopLogprobEntry {
                        token_id: 42,
                        logprob: -0.5,
                    },
                    TopLogprobEntry {
                        token_id: 99,
                        logprob: -1.2,
                    },
                ],
            }]),
            reasoning_content: None,
            cached_prompt_tokens: 0,
        };
        let lps = output.token_logprobs.unwrap();
        assert_eq!(lps.len(), 1);
        assert_eq!(lps[0].token_id, 42);
        assert_eq!(lps[0].top_logprobs.len(), 2);
    }
}
