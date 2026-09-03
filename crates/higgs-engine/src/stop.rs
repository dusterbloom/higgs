//! Per-generation stop signal checked at bounded allocation boundaries.
//!
//! The plan contract: a reservation's bytes are freed only after the worker
//! acknowledges cancellation at a safe boundary and stops allocating. This
//! type is that acknowledgement channel. It is deliberately cheap — one
//! atomic flag, a first-reason slot, and one timestamp — because it is probed
//! before every prefill chunk and decode step.
//!
//! Conditions that stop a generation:
//!
//! - client disconnect (observed by the streaming forwarder or the HTTP task;
//!   an HTTP-layer request timeout manifests the same way once the stream is
//!   dropped);
//! - the no-progress watchdog, using the configured request timeout as the
//!   window since the last prefill chunk / decode token;
//! - critical memory pressure (marked process-wide by the capacity registry);
//! - model drain (unload cancels and joins workers before releasing weights).
//!
//! Engines observe it either directly (batch requests carry one per request)
//! or through the thread-local installed by the HTTP worker next to its
//! reservation, which the thread-local prefill progress sink also reads.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Why an in-flight generation must stop allocating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// The HTTP client went away; the stream is dead.
    ClientDisconnect,
    /// No prefill chunk or decode token progressed within the configured
    /// request-timeout window.
    NoProgressWatchdog,
    /// Critical memory pressure: the registry interrupted live reservations.
    /// Carries the boot ID and generation for the typed terminal SSE.
    CriticalPressure { boot_id: String, generation: u64 },
    /// The model is being unloaded; workers must stop and join.
    ModelDrain,
}

#[derive(Debug)]
struct Shared {
    stopped: AtomicBool,
    reason: Mutex<Option<StopReason>>,
    /// Watchdog window in milliseconds; zero disables the watchdog.
    watchdog_ms: AtomicU64,
    last_progress: Mutex<Instant>,
}

/// Cheap shared stop signal for one in-flight generation.
#[derive(Clone, Debug)]
pub struct GenerationStop {
    shared: Arc<Shared>,
}

impl Default for GenerationStop {
    fn default() -> Self {
        Self::new(None)
    }
}

impl GenerationStop {
    #[must_use]
    pub fn new(watchdog: Option<Duration>) -> Self {
        Self {
            shared: Arc::new(Shared {
                stopped: AtomicBool::new(false),
                reason: Mutex::new(None),
                watchdog_ms: AtomicU64::new(watchdog.map_or(0, |w| {
                    u64::try_from(w.as_millis()).unwrap_or(u64::MAX).max(1)
                })),
                last_progress: Mutex::new(Instant::now()),
            }),
        }
    }

    /// Record an external stop (disconnect, pressure, drain). The first
    /// reason wins; later calls are no-ops so the typed terminal event stays
    /// stable.
    pub fn stop(&self, reason: StopReason) {
        let mut slot = self.shared.reason.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(reason);
            self.shared.stopped.store(true, Ordering::Release);
        }
    }

    /// The first recorded stop reason, if any.
    #[must_use]
    pub fn reason(&self) -> Option<StopReason> {
        self.shared
            .reason
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Set or clear the watchdog window. The registry creates stops without
    /// one; the route worker arms it from the configured request timeout
    /// before entering the engine call.
    pub fn set_watchdog(&self, watchdog: Option<std::time::Duration>) {
        self.shared.watchdog_ms.store(
            watchdog.map_or(0, |w| {
                u64::try_from(w.as_millis()).unwrap_or(u64::MAX).max(1)
            }),
            Ordering::Relaxed,
        );
        *self
            .shared
            .last_progress
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = std::time::Instant::now();
    }

    /// Note allocation progress (a prefill chunk or a decode token) so the
    /// no-progress watchdog distinguishes a slow-but-alive generation from a
    /// stalled one.
    pub fn note_progress(&self) {
        *self
            .shared
            .last_progress
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Instant::now();
    }

    /// Boundary check: returns the reason allocation must stop, evaluating
    /// the watchdog lazily (and recording it if it fires). `None` means
    /// continue.
    pub fn check(&self) -> Option<StopReason> {
        if let Some(reason) = self.reason() {
            return Some(reason);
        }
        let watchdog_ms = self.shared.watchdog_ms.load(Ordering::Relaxed);
        if watchdog_ms == 0 {
            return None;
        }
        let stalled_ms = u64::try_from(
            self.shared
                .last_progress
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .elapsed()
                .as_millis(),
        )
        .unwrap_or(u64::MAX);
        if stalled_ms >= watchdog_ms {
            self.stop(StopReason::NoProgressWatchdog);
            return Some(StopReason::NoProgressWatchdog);
        }
        None
    }

    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.shared.stopped.load(Ordering::Acquire)
    }
}

thread_local! {
    static GENERATION_STOP: Mutex<Option<GenerationStop>> = const { Mutex::new(None) };
}

/// Install `stop` as the current thread's generation stop for the guard's
/// lifetime. The HTTP worker does this next to its reservation before
/// entering an engine call; engine loops read it back with
/// [`generation_stop`].
pub fn install_generation_stop(stop: GenerationStop) -> GenerationStopGuard {
    GENERATION_STOP.with(|s| {
        *s.lock().unwrap_or_else(|e| e.into_inner()) = Some(stop);
    });
    GenerationStopGuard
}

/// RAII uninstall, mirroring the prefill sink guard.
pub struct GenerationStopGuard;

impl Drop for GenerationStopGuard {
    fn drop(&mut self) {
        GENERATION_STOP.with(|s| {
            *s.lock().unwrap_or_else(|e| e.into_inner()) = None;
        });
    }
}

/// The current thread's stop signal, if the calling worker installed one.
/// Clone is cheap (one `Arc`).
#[must_use]
pub fn generation_stop() -> Option<GenerationStop> {
    GENERATION_STOP.with(|s| s.lock().unwrap_or_else(|e| e.into_inner()).clone())
}

/// Convenience: the current thread's boundary check. `None` when no stop is
/// installed (engines keep their pre-Task-7 behavior) or when allocation may
/// continue.
#[must_use]
pub fn generation_stop_check() -> Option<StopReason> {
    generation_stop().and_then(|stop| stop.check())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn first_external_reason_wins() {
        let stop = GenerationStop::new(None);
        stop.stop(StopReason::ClientDisconnect);
        stop.stop(StopReason::ModelDrain);
        assert_eq!(stop.check(), Some(StopReason::ClientDisconnect));
        assert_eq!(stop.reason(), Some(StopReason::ClientDisconnect));
        assert!(stop.is_stopped());
    }

    #[test]
    fn watchdog_trips_only_without_progress() {
        let stalled = GenerationStop::new(Some(Duration::from_millis(20)));
        thread::sleep(Duration::from_millis(30));
        assert_eq!(stalled.check(), Some(StopReason::NoProgressWatchdog));

        // Steady progress inside a much larger window stays alive; the
        // watchdog measures since the last chunk/token, not since start.
        let alive = GenerationStop::new(Some(Duration::from_secs(2)));
        thread::sleep(Duration::from_millis(20));
        alive.note_progress();
        assert_eq!(alive.check(), None);
    }

    #[test]
    fn zero_watchdog_disables_the_check() {
        let stop = GenerationStop::new(None);
        thread::sleep(Duration::from_millis(5));
        assert_eq!(stop.check(), None);
    }

    #[test]
    fn thread_local_is_scoped_by_the_guard() {
        assert!(generation_stop().is_none());
        let stop = GenerationStop::new(None);
        {
            let _guard = install_generation_stop(stop.clone());
            assert!(generation_stop().is_some());
            assert_eq!(generation_stop_check(), None);
            stop.stop(StopReason::ModelDrain);
            assert_eq!(generation_stop_check(), Some(StopReason::ModelDrain));
        }
        assert!(generation_stop().is_none());
    }

    #[test]
    fn stop_reason_is_visible_across_threads() {
        let stop = GenerationStop::new(None);
        let worker = stop.clone();
        let handle = thread::spawn(move || {
            worker.stop(StopReason::CriticalPressure {
                boot_id: "boot".to_owned(),
                generation: 7,
            });
        });
        handle.join().unwrap();
        assert_eq!(
            stop.check(),
            Some(StopReason::CriticalPressure {
                boot_id: "boot".to_owned(),
                generation: 7
            })
        );
    }
}
