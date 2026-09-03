use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use tokio::{
    sync::{Mutex, mpsc, oneshot},
    task::JoinHandle,
};

use super::{
    CapacityController, CapacityPressureCoordinator, CapacityRegistry, Clock, MemoryPressure,
    PressureObservation,
};

const VM_COUNTER_SAMPLE_PERIOD: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct VmCounters {
    pub(crate) swap_outs: u64,
    pub(crate) compressions: u64,
}

pub(crate) trait CounterSampler: Send + 'static {
    fn sample(&mut self) -> Option<VmCounters>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ObserverEvent {
    Pressure(MemoryPressure),
    Sample,
}

pub(crate) trait ProducerHandle: Send {
    /// Cancellation itself happens synchronously; the future acknowledges that
    /// callbacks/tasks have stopped and released their sender clone.
    fn cancel(self: Box<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>>;
}

pub(crate) trait PressureEventSource: Send + 'static {
    fn start(
        self,
        sender: mpsc::UnboundedSender<ObserverEvent>,
    ) -> Result<Box<dyn ProducerHandle>, PressureObserverError>;
}

#[cfg(any(test, not(target_os = "macos")))]
pub(crate) struct NoopPressureSource;

#[cfg(any(test, not(target_os = "macos")))]
struct NoopPressureHandle;

#[cfg(any(test, not(target_os = "macos")))]
impl ProducerHandle for NoopPressureHandle {
    fn cancel(self: Box<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(async {})
    }
}

#[cfg(any(test, not(target_os = "macos")))]
impl PressureEventSource for NoopPressureSource {
    fn start(
        self,
        _sender: mpsc::UnboundedSender<ObserverEvent>,
    ) -> Result<Box<dyn ProducerHandle>, PressureObserverError> {
        Ok(Box::new(NoopPressureHandle))
    }
}

pub(crate) trait PressureObservationSink: Send + Sync + 'static {
    fn apply<'a>(
        &'a self,
        observation: PressureObservation,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

impl<C: Clock + Send + 'static> PressureObservationSink for Mutex<CapacityController<C>> {
    fn apply<'a>(
        &'a self,
        observation: PressureObservation,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            self.lock().await.apply_pressure_observation(observation);
        })
    }
}

impl PressureObservationSink for CapacityRegistry {
    fn apply<'a>(
        &'a self,
        observation: PressureObservation,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            self.apply_pressure_observation(observation);
        })
    }
}

impl PressureObservationSink for CapacityPressureCoordinator {
    fn apply<'a>(
        &'a self,
        observation: PressureObservation,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            self.capacity.apply_pressure_observation(observation);
            let state = self
                .state
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_ref()
                .and_then(std::sync::Weak::upgrade);
            if let Some(state) = state
                && let Err(error) = state
                    .router
                    .apply_capacity_cache_allocations(&self.capacity)
                    .await
            {
                tracing::warn!(%error, "failed to apply pressure-adjusted cache allocations");
            }
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum PressureObserverError {
    #[error("memory-pressure observation is unavailable on this platform")]
    UnsupportedPlatform,
    #[error("memory-pressure consumer failed: {0}")]
    ConsumerJoin(#[from] tokio::task::JoinError),
}

/// Owned, single-use observer configuration. Starting consumes every producer,
/// so the same dispatch source or cadence cannot be started twice.
pub(crate) struct PressureObserverConfig<P, S, T> {
    pressure_source: P,
    counters: S,
    cadence: T,
}

impl<P, S, T> PressureObserverConfig<P, S, T> {
    pub(crate) const fn new(pressure_source: P, counters: S, cadence: T) -> Self {
        Self {
            pressure_source,
            counters,
            cadence,
        }
    }
}

impl<P, S, T> PressureObserverConfig<P, S, T>
where
    P: PressureEventSource,
    S: CounterSampler,
    T: PressureEventSource,
{
    pub(crate) async fn start<Sink: PressureObservationSink>(
        self,
        sink: Arc<Sink>,
    ) -> Result<PressureObserverHandle, PressureObserverError> {
        let Self {
            pressure_source,
            mut counters,
            cadence,
        } = self;
        // Seed before producers start. Historical cumulative totals therefore
        // establish a baseline and can never masquerade as new activity.
        let mut prior = counters.sample();
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let pressure = pressure_source.start(sender.clone())?;
        let cadence = match cadence.start(sender.clone()) {
            Ok(handle) => handle,
            Err(error) => {
                pressure.cancel().await;
                return Err(error);
            }
        };
        let consumer = tokio::spawn(async move {
            let mut reported_pressure = MemoryPressure::Normal;
            while let Some(event) = receiver.recv().await {
                if let ObserverEvent::Pressure(pressure) = event {
                    reported_pressure = pressure;
                }
                let current = counters.sample();
                let (swap_out_delta, compressor_delta) = match (prior, current) {
                    (Some(previous), Some(current)) => (
                        current.swap_outs.saturating_sub(previous.swap_outs),
                        current.compressions.saturating_sub(previous.compressions),
                    ),
                    _ => (0, 0),
                };
                if current.is_some() {
                    prior = current;
                }
                sink.apply(PressureObservation {
                    pressure: reported_pressure,
                    swap_out_delta,
                    compressor_delta,
                })
                .await;
            }
        });
        Ok(PressureObserverHandle {
            pressure: Some(pressure),
            cadence: Some(cadence),
            sender: Some(sender),
            consumer: Some(consumer),
        })
    }
}

/// Explicitly stopped by Task 5 during server shutdown. Drop is only a safety
/// net: it cancels producers and aborts because synchronous Drop cannot join.
#[must_use = "a live pressure observer must be stopped and joined"]
pub(crate) struct PressureObserverHandle {
    pressure: Option<Box<dyn ProducerHandle>>,
    cadence: Option<Box<dyn ProducerHandle>>,
    sender: Option<mpsc::UnboundedSender<ObserverEvent>>,
    consumer: Option<JoinHandle<()>>,
}

impl PressureObserverHandle {
    pub(crate) async fn stop(mut self) -> Result<(), PressureObserverError> {
        if let Some(pressure) = self.pressure.take() {
            pressure.cancel().await;
        }
        if let Some(cadence) = self.cadence.take() {
            cadence.cancel().await;
        }
        self.sender.take();
        if let Some(consumer) = self.consumer.take() {
            consumer.await?;
        }
        Ok(())
    }
}

impl Drop for PressureObserverHandle {
    fn drop(&mut self) {
        if let Some(pressure) = self.pressure.take() {
            drop(pressure.cancel());
        }
        if let Some(cadence) = self.cadence.take() {
            drop(cadence.cancel());
        }
        self.sender.take();
        if let Some(consumer) = self.consumer.take() {
            consumer.abort();
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TokioCadence {
    period: Duration,
}

impl Default for TokioCadence {
    fn default() -> Self {
        Self {
            period: VM_COUNTER_SAMPLE_PERIOD,
        }
    }
}

struct TokioCadenceHandle {
    cancel: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl ProducerHandle for TokioCadenceHandle {
    fn cancel(mut self: Box<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
        let task = self.task;
        Box::pin(async move {
            let _ = task.await;
        })
    }
}

impl PressureEventSource for TokioCadence {
    fn start(
        self,
        sender: mpsc::UnboundedSender<ObserverEvent>,
    ) -> Result<Box<dyn ProducerHandle>, PressureObserverError> {
        let (cancel, mut cancelled) = oneshot::channel();
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.period);
            // The first Tokio interval tick is immediate; consume it because the
            // counter baseline was already sampled synchronously at start.
            interval.tick().await;
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if sender.send(ObserverEvent::Sample).is_err() {
                            break;
                        }
                    }
                    _ = &mut cancelled => break,
                }
            }
        });
        Ok(Box::new(TokioCadenceHandle {
            cancel: Some(cancel),
            task,
        }))
    }
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod platform {
    use std::sync::{Arc, Mutex as StdMutex};

    use block2::RcBlock;
    use dispatch2::{
        _dispatch_source_type_memorypressure, DispatchObject, DispatchQueue, DispatchQueueAttr,
        DispatchRetained, DispatchSource, dispatch_source_memorypressure_flags_t,
    };

    use super::{
        CounterSampler, MemoryPressure, ObserverEvent, PressureEventSource, PressureObserverError,
        ProducerHandle, VmCounters, mpsc, oneshot,
    };

    pub(crate) struct SystemPressureSource;

    struct SystemPressureHandle {
        source: DispatchRetained<DispatchSource>,
        cancelled: oneshot::Receiver<()>,
    }

    impl ProducerHandle for SystemPressureHandle {
        fn cancel(self: Box<Self>) -> super::Pin<Box<dyn super::Future<Output = ()> + Send>> {
            self.source.cancel();
            let source = self.source;
            let cancelled = self.cancelled;
            Box::pin(async move {
                let _ = cancelled.await;
                // Keep the activated source alive through cancellation acknowledgement.
                drop(source);
            })
        }
    }

    impl PressureEventSource for SystemPressureSource {
        #[allow(unsafe_code)]
        fn start(
            self,
            sender: mpsc::UnboundedSender<ObserverEvent>,
        ) -> Result<Box<dyn ProducerHandle>, PressureObserverError> {
            let queue = DispatchQueue::new(
                "dev.higgs.capacity.memory-pressure",
                DispatchQueueAttr::SERIAL,
            );
            let mask = usize::try_from(
                dispatch_source_memorypressure_flags_t::DISPATCH_MEMORYPRESSURE_NORMAL.0
                    | dispatch_source_memorypressure_flags_t::DISPATCH_MEMORYPRESSURE_WARN.0
                    | dispatch_source_memorypressure_flags_t::DISPATCH_MEMORYPRESSURE_CRITICAL.0,
            )
            .map_err(|_| PressureObserverError::UnsupportedPlatform)?;
            let warning_flag = usize::try_from(
                dispatch_source_memorypressure_flags_t::DISPATCH_MEMORYPRESSURE_WARN.0,
            )
            .map_err(|_| PressureObserverError::UnsupportedPlatform)?;
            let critical_flag = usize::try_from(
                dispatch_source_memorypressure_flags_t::DISPATCH_MEMORYPRESSURE_CRITICAL.0,
            )
            .map_err(|_| PressureObserverError::UnsupportedPlatform)?;
            // SAFETY: This is Apple's documented memory-pressure source type;
            // it takes handle 0 and the three declared memory-pressure flags.
            let source = unsafe {
                DispatchSource::new(
                    std::ptr::from_ref(&_dispatch_source_type_memorypressure).cast_mut(),
                    0,
                    mask,
                    Some(&queue),
                )
            };
            let event_source = source.clone();
            let event_handler = RcBlock::new(move || {
                let flags = event_source.data();
                let pressure = if flags & critical_flag != 0 {
                    MemoryPressure::Critical
                } else if flags & warning_flag != 0 {
                    MemoryPressure::Constrained
                } else {
                    MemoryPressure::Normal
                };
                // GCD callbacks only enqueue a tiny copy. A stopped receiver is normal.
                let _ = sender.send(ObserverEvent::Pressure(pressure));
            });
            // SAFETY: Dispatch copies the heap block and releases it after cancellation.
            unsafe { source.set_event_handler_with_block(RcBlock::as_ptr(&event_handler)) };

            let (cancelled_sender, cancelled) = oneshot::channel();
            let cancelled_sender = Arc::new(StdMutex::new(Some(cancelled_sender)));
            let cancellation_handler = RcBlock::new(move || {
                if let Ok(mut sender) = cancelled_sender.lock() {
                    if let Some(sender) = sender.take() {
                        let _ = sender.send(());
                    }
                }
            });
            // SAFETY: Dispatch copies this heap block and calls it after event delivery stops.
            unsafe {
                source.set_cancel_handler_with_block(RcBlock::as_ptr(&cancellation_handler));
            }
            source.activate();
            Ok(Box::new(SystemPressureHandle { source, cancelled }))
        }
    }

    type PortReleaser = Box<dyn FnOnce(libc::mach_port_t) + Send>;

    /// Owns the send right returned by mach_host_self and balances it exactly
    /// once when the process-wide sampler shuts down.
    struct OwnedHostPort {
        raw: libc::mach_port_t,
        releaser: Option<PortReleaser>,
    }

    impl OwnedHostPort {
        #[allow(deprecated)]
        fn acquire() -> Self {
            unsafe extern "C" {
                fn mach_port_deallocate(
                    task: libc::mach_port_t,
                    name: libc::mach_port_t,
                ) -> libc::kern_return_t;
            }

            // SAFETY: mach_host_self returns one owned send right for this call.
            let raw = unsafe { libc::mach_host_self() };
            Self::with_releaser(
                raw,
                Box::new(move |host| {
                    // SAFETY: `host` is the still-owned right returned above;
                    // mach_task_self is borrowed and must not itself be released.
                    let _ = unsafe { mach_port_deallocate(libc::mach_task_self(), host) };
                }),
            )
        }

        fn with_releaser(raw: libc::mach_port_t, releaser: PortReleaser) -> Self {
            Self {
                raw,
                releaser: Some(releaser),
            }
        }
    }

    impl Drop for OwnedHostPort {
        fn drop(&mut self) {
            if let Some(releaser) = self.releaser.take() {
                releaser(self.raw);
            }
        }
    }

    fn field_is_covered<T>(returned_count: libc::mach_msg_type_number_t, offset: usize) -> bool {
        usize::try_from(returned_count)
            .ok()
            .and_then(|count| count.checked_mul(std::mem::size_of::<libc::integer_t>()))
            .zip(offset.checked_add(std::mem::size_of::<T>()))
            .is_some_and(|(returned_bytes, field_end)| returned_bytes >= field_end)
    }

    fn decode_vm_counters(
        result: libc::kern_return_t,
        returned_count: libc::mach_msg_type_number_t,
        statistics: &libc::vm_statistics64,
    ) -> Option<VmCounters> {
        if result != libc::KERN_SUCCESS
            || !field_is_covered::<u64>(
                returned_count,
                std::mem::offset_of!(libc::vm_statistics64, compressions),
            )
            || !field_is_covered::<u64>(
                returned_count,
                std::mem::offset_of!(libc::vm_statistics64, swapouts),
            )
        {
            return None;
        }
        Some(VmCounters {
            swap_outs: statistics.swapouts,
            compressions: statistics.compressions,
        })
    }

    pub(crate) struct SystemVmCounters {
        host: OwnedHostPort,
    }

    impl SystemVmCounters {
        pub(super) fn new() -> Self {
            Self {
                host: OwnedHostPort::acquire(),
            }
        }
    }

    impl CounterSampler for SystemVmCounters {
        fn sample(&mut self) -> Option<VmCounters> {
            // Zero initialization makes the whole buffer valid even when Darwin
            // returns an error or writes fewer integer_t slots than requested.
            let mut statistics = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
            let mut count = libc::HOST_VM_INFO64_COUNT;
            // SAFETY: The output buffer is exactly vm_statistics64 and count is
            // initialized to the Darwin-declared HOST_VM_INFO64_COUNT.
            let result = unsafe {
                libc::host_statistics64(
                    self.host.raw,
                    libc::HOST_VM_INFO64,
                    statistics.as_mut_ptr().cast(),
                    &raw mut count,
                )
            };
            // SAFETY: The allocation started fully zeroed, so all fields remain
            // initialized even if Darwin wrote only a prefix or returned failure.
            let statistics = unsafe { statistics.assume_init() };
            decode_vm_counters(result, count, &statistics)
        }
    }

    #[cfg(test)]
    mod tests {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        use super::{OwnedHostPort, decode_vm_counters, field_is_covered};

        #[allow(unsafe_code)]
        fn statistics_with_counters(compressions: u64, swapouts: u64) -> libc::vm_statistics64 {
            // SAFETY: Every field in vm_statistics64 is an integer, for which
            // the all-zero bit pattern is valid.
            let mut statistics =
                unsafe { std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed().assume_init() };
            statistics.compressions = compressions;
            statistics.swapouts = swapouts;
            statistics
        }

        #[test]
        fn decoder_requires_both_counter_fields_and_a_successful_call() {
            let statistics = statistics_with_counters(7, 11);

            assert_eq!(
                decode_vm_counters(libc::KERN_SUCCESS, 27, &statistics),
                None
            );
            assert_eq!(
                decode_vm_counters(libc::KERN_SUCCESS, 31, &statistics),
                None
            );
            assert_eq!(
                decode_vm_counters(libc::KERN_SUCCESS, 32, &statistics),
                Some(super::VmCounters {
                    swap_outs: 11,
                    compressions: 7,
                })
            );
            assert_eq!(decode_vm_counters(5, 32, &statistics), None);
        }

        #[test]
        fn returned_count_is_interpreted_as_integer_slots_for_each_field() {
            let compressions = std::mem::offset_of!(libc::vm_statistics64, compressions);
            let swapouts = std::mem::offset_of!(libc::vm_statistics64, swapouts);

            assert!(!field_is_covered::<u64>(27, compressions));
            assert!(field_is_covered::<u64>(28, compressions));
            assert!(!field_is_covered::<u64>(31, swapouts));
            assert!(field_is_covered::<u64>(32, swapouts));
        }

        #[test]
        fn owned_host_port_releases_exactly_once() {
            let releases = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&releases);
            {
                let _port = OwnedHostPort::with_releaser(
                    41,
                    Box::new(move |port| {
                        assert_eq!(port, 41);
                        observed.fetch_add(1, Ordering::Relaxed);
                    }),
                );
            }
            assert_eq!(releases.load(Ordering::Relaxed), 1);
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::{CounterSampler, VmCounters};

    pub(crate) use super::NoopPressureSource as SystemPressureSource;

    pub(crate) struct SystemVmCounters;

    impl SystemVmCounters {
        pub(super) fn new() -> Self {
            Self
        }
    }

    impl CounterSampler for SystemVmCounters {
        fn sample(&mut self) -> Option<VmCounters> {
            None
        }
    }
}

pub(crate) use platform::{SystemPressureSource, SystemVmCounters};

pub(crate) fn system_observer_config()
-> PressureObserverConfig<SystemPressureSource, SystemVmCounters, TokioCadence> {
    PressureObserverConfig::new(
        SystemPressureSource,
        SystemVmCounters::new(),
        TokioCadence::default(),
    )
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        future::{Future, ready},
        pin::Pin,
        sync::{
            Arc, Mutex as StdMutex,
            atomic::{AtomicU64, AtomicUsize, Ordering},
        },
    };

    use higgs_engine::{EngineCostDescription, MlxMemorySnapshot, TransientPrefillEstimate};
    use tokio::sync::oneshot;

    use super::super::{
        CapacityAvailability, CapacityController, CapacityInputs, Clock, MemoryPressure,
        PressureObservation, floor_1024,
    };
    use super::{
        CounterSampler, NoopPressureSource, ObserverEvent, PressureEventSource,
        PressureObserverConfig, ProducerHandle, VmCounters,
    };

    const GIB: u64 = 1024 * 1024 * 1024;

    #[tokio::test]
    async fn portable_noop_pressure_source_owns_a_stoppable_handle() {
        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        let handle = NoopPressureSource.start(sender).unwrap();

        handle.cancel().await;
    }

    #[cfg(not(target_os = "macos"))]
    #[tokio::test]
    async fn system_observer_starts_and_stops_on_non_macos() {
        let controller = Arc::new(tokio::sync::Mutex::new(CapacityController::with_clock(
            controller_inputs(),
            TestClock::new(),
        )));

        let handle = super::system_observer_config()
            .start(controller)
            .await
            .unwrap();
        handle.stop().await.unwrap();
    }

    #[derive(Clone)]
    struct TestClock(Arc<AtomicU64>);

    impl TestClock {
        fn new() -> Self {
            Self(Arc::new(AtomicU64::new(0)))
        }

        fn set_seconds(&self, seconds: u64) {
            self.0.store(seconds * 1_000, Ordering::Relaxed);
        }
    }

    impl Clock for TestClock {
        fn now_millis(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    fn controller_inputs() -> CapacityInputs {
        CapacityInputs {
            memory: MlxMemorySnapshot {
                active_bytes: 11 * GIB,
                peak_bytes: 11 * GIB,
                memory_limit_bytes: Some(48 * GIB),
                metal_recommended_working_set_bytes: Some(48 * GIB),
            },
            costs: EngineCostDescription {
                fixed_live_session_bytes: 256 * 1024 * 1024,
                persistent_bytes_per_token: 20_480,
                decode_workspace_bytes: 256 * 1024 * 1024,
                transient_prefill: TransientPrefillEstimate {
                    base_bytes: 2 * GIB,
                    bytes_per_prompt_token: 0,
                    bytes_per_chunk_token: 4 * 1024 * 1024,
                    max_prompt_tokens: 131_072,
                    max_chunk_tokens: 512,
                },
            },
            loaded_model_bytes: 11 * GIB,
            architectural_max_tokens: 131_072,
            prefill_chunk_tokens: 512,
            retained_bytes: 256 * 1024 * 1024,
            prefix_cache_bytes: 256 * 1024 * 1024,
            active_reservation_bytes: 0,
            configured_total_token_ceiling: None,
            configured_output_token_ceiling: None,
            pressure: MemoryPressure::Normal,
        }
    }

    #[test]
    fn constrained_observation_downshifts_once() {
        let clock = TestClock::new();
        let mut controller = CapacityController::with_clock(controller_inputs(), clock);
        let initial = controller.decision().safe_total_tokens;
        let observation = PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        };

        let first = controller.apply_pressure_observation(observation);
        let second = controller.apply_pressure_observation(observation);

        assert_eq!(first.safe_total_tokens, floor_1024(initial * 75 / 100));
        assert_eq!(second, first);
        assert_eq!(controller.pressure(), MemoryPressure::Constrained);
        assert_eq!(first.availability, CapacityAvailability::Available);
    }

    #[test]
    fn critical_observation_downshifts_once_and_rejects_new_admission() {
        let clock = TestClock::new();
        let mut controller = CapacityController::with_clock(controller_inputs(), clock);
        let initial = controller.decision().safe_total_tokens;
        let observation = PressureObservation {
            pressure: MemoryPressure::Critical,
            swap_out_delta: 0,
            compressor_delta: 0,
        };

        let first = controller.apply_pressure_observation(observation);
        let second = controller.apply_pressure_observation(observation);

        assert_eq!(first.safe_total_tokens, floor_1024(initial * 50 / 100));
        assert_eq!(second, first);
        assert_eq!(first.availability, CapacityAvailability::Unavailable);
    }

    #[test]
    fn critical_to_constrained_to_normal_recovers_without_another_downshift() {
        let clock = TestClock::new();
        let mut controller = CapacityController::with_clock(controller_inputs(), clock);
        let critical = controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Critical,
            swap_out_delta: 0,
            compressor_delta: 0,
        });

        let constrained = controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(constrained.safe_total_tokens, critical.safe_total_tokens);
        assert_eq!(constrained.availability, CapacityAvailability::Available);

        let normal = controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(normal.safe_total_tokens, critical.safe_total_tokens);
        assert_eq!(normal.availability, CapacityAvailability::Available);
    }

    #[test]
    fn warning_critical_oscillation_does_not_ratchet_the_envelope() {
        let clock = TestClock::new();
        let mut controller = CapacityController::with_clock(controller_inputs(), clock);
        controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        let first_critical = controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Critical,
            swap_out_delta: 0,
            compressor_delta: 0,
        });

        for _ in 0..3 {
            let constrained = controller.apply_pressure_observation(PressureObservation {
                pressure: MemoryPressure::Constrained,
                swap_out_delta: 0,
                compressor_delta: 0,
            });
            assert_eq!(
                constrained.safe_total_tokens,
                first_critical.safe_total_tokens
            );
            let critical = controller.apply_pressure_observation(PressureObservation {
                pressure: MemoryPressure::Critical,
                swap_out_delta: 0,
                compressor_delta: 0,
            });
            assert_eq!(critical.safe_total_tokens, first_critical.safe_total_tokens);
            assert_eq!(critical.availability, CapacityAvailability::Unavailable);
        }
    }

    #[test]
    fn normal_recovery_changes_state_without_raising_capacity() {
        let clock = TestClock::new();
        let mut controller = CapacityController::with_clock(controller_inputs(), clock);
        controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Constrained,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        let constrained = controller.decision().safe_total_tokens;

        let recovered = controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });

        assert_eq!(controller.pressure(), MemoryPressure::Normal);
        assert_eq!(recovered.safe_total_tokens, constrained);
        assert_eq!(recovered.availability, CapacityAvailability::Available);
    }

    #[test]
    fn swap_out_is_critical_until_sixty_clean_seconds() {
        let clock = TestClock::new();
        let mut controller = CapacityController::with_clock(controller_inputs(), clock.clone());
        controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 1,
            compressor_delta: 0,
        });
        assert_eq!(controller.pressure(), MemoryPressure::Critical);

        clock.set_seconds(59);
        controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(controller.pressure(), MemoryPressure::Critical);

        clock.set_seconds(60);
        controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(controller.pressure(), MemoryPressure::Normal);
    }

    #[test]
    fn new_swap_out_restarts_the_clean_minute_without_repeated_downshift() {
        let clock = TestClock::new();
        let mut controller = CapacityController::with_clock(controller_inputs(), clock.clone());
        controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 1,
            compressor_delta: 0,
        });
        let first_downshift = controller.decision();

        clock.set_seconds(30);
        let repeated = controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 2,
            compressor_delta: 0,
        });
        assert_eq!(repeated, first_downshift);

        clock.set_seconds(89);
        controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(controller.pressure(), MemoryPressure::Critical);

        clock.set_seconds(90);
        controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 0,
        });
        assert_eq!(controller.pressure(), MemoryPressure::Normal);
    }

    #[test]
    fn compressor_growth_constrains_without_an_invented_rate_threshold() {
        let clock = TestClock::new();
        let mut controller = CapacityController::with_clock(controller_inputs(), clock);
        let initial = controller.decision().safe_total_tokens;

        let constrained = controller.apply_pressure_observation(PressureObservation {
            pressure: MemoryPressure::Normal,
            swap_out_delta: 0,
            compressor_delta: 1,
        });

        assert_eq!(controller.pressure(), MemoryPressure::Constrained);
        assert_eq!(
            constrained.safe_total_tokens,
            floor_1024(initial * 75 / 100)
        );
    }

    #[derive(Clone)]
    struct FakeEmitter {
        sender: Arc<StdMutex<Option<tokio::sync::mpsc::UnboundedSender<ObserverEvent>>>>,
    }

    impl FakeEmitter {
        fn emit(&self, event: ObserverEvent) {
            if let Ok(sender) = self.sender.lock() {
                if let Some(sender) = sender.as_ref() {
                    let _ = sender.send(event);
                }
            }
        }
    }

    struct FakeProducer {
        emitter: FakeEmitter,
        starts: Arc<AtomicUsize>,
        cancels: Arc<AtomicUsize>,
    }

    impl FakeProducer {
        fn new() -> (Self, FakeEmitter, Arc<AtomicUsize>, Arc<AtomicUsize>) {
            let emitter = FakeEmitter {
                sender: Arc::new(StdMutex::new(None)),
            };
            let starts = Arc::new(AtomicUsize::new(0));
            let cancels = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    emitter: emitter.clone(),
                    starts: Arc::clone(&starts),
                    cancels: Arc::clone(&cancels),
                },
                emitter,
                starts,
                cancels,
            )
        }
    }

    struct FakeProducerHandle {
        emitter: FakeEmitter,
        cancels: Arc<AtomicUsize>,
    }

    impl ProducerHandle for FakeProducerHandle {
        fn cancel(self: Box<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
            self.cancels.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut sender) = self.emitter.sender.lock() {
                sender.take();
            }
            Box::pin(ready(()))
        }
    }

    impl PressureEventSource for FakeProducer {
        fn start(
            self,
            sender: tokio::sync::mpsc::UnboundedSender<ObserverEvent>,
        ) -> Result<Box<dyn ProducerHandle>, super::PressureObserverError> {
            self.starts.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut slot) = self.emitter.sender.lock() {
                *slot = Some(sender);
            }
            Ok(Box::new(FakeProducerHandle {
                emitter: self.emitter,
                cancels: self.cancels,
            }))
        }
    }

    struct FailingProducer;

    impl PressureEventSource for FailingProducer {
        fn start(
            self,
            _sender: tokio::sync::mpsc::UnboundedSender<ObserverEvent>,
        ) -> Result<Box<dyn ProducerHandle>, super::PressureObserverError> {
            Err(super::PressureObserverError::UnsupportedPlatform)
        }
    }

    #[derive(Clone)]
    struct FakeCounters {
        samples: Arc<StdMutex<VecDeque<Option<VmCounters>>>>,
        calls: Arc<AtomicUsize>,
    }

    impl FakeCounters {
        fn new(samples: impl IntoIterator<Item = VmCounters>) -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    samples: Arc::new(StdMutex::new(samples.into_iter().map(Some).collect())),
                    calls: Arc::clone(&calls),
                },
                calls,
            )
        }
    }

    impl CounterSampler for FakeCounters {
        fn sample(&mut self) -> Option<VmCounters> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.samples.lock().ok()?.pop_front().flatten()
        }
    }

    async fn wait_for_samples(calls: &AtomicUsize, expected: usize) {
        for _ in 0..1_000 {
            if calls.load(Ordering::Relaxed) >= expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(calls.load(Ordering::Relaxed), expected);
    }

    #[tokio::test]
    async fn seeded_baseline_ignores_historical_nonzero_totals() {
        let clock = TestClock::new();
        let controller = Arc::new(tokio::sync::Mutex::new(CapacityController::with_clock(
            controller_inputs(),
            clock,
        )));
        let initial = controller.lock().await.decision();
        let (pressure_source, pressure, starts, cancels) = FakeProducer::new();
        let (cadence_source, cadence, cadence_starts, cadence_cancels) = FakeProducer::new();
        let (counters, calls) = FakeCounters::new([
            VmCounters {
                swap_outs: 900,
                compressions: 4_000,
            },
            VmCounters {
                swap_outs: 900,
                compressions: 4_000,
            },
        ]);
        let handle = PressureObserverConfig::new(pressure_source, counters, cadence_source)
            .start(Arc::clone(&controller))
            .await
            .unwrap();

        cadence.emit(ObserverEvent::Sample);
        wait_for_samples(&calls, 2).await;

        assert_eq!(controller.lock().await.decision(), initial);
        assert_eq!(starts.load(Ordering::Relaxed), 1);
        assert_eq!(cadence_starts.load(Ordering::Relaxed), 1);
        handle.stop().await.unwrap();
        assert_eq!(cancels.load(Ordering::Relaxed), 1);
        assert_eq!(cadence_cancels.load(Ordering::Relaxed), 1);
        pressure.emit(ObserverEvent::Pressure(MemoryPressure::Critical));
        assert_eq!(controller.lock().await.decision(), initial);
    }

    #[tokio::test]
    async fn failed_observer_start_cancels_the_started_source_and_returns_no_handle() {
        let registry = crate::capacity::CapacityRegistry::new(["escha".to_owned()]);
        let (pressure_source, _pressure, starts, cancels) = FakeProducer::new();
        let (counters, _calls) = FakeCounters::new([VmCounters::default()]);

        let result = PressureObserverConfig::new(pressure_source, counters, FailingProducer)
            .start(registry)
            .await;

        assert!(matches!(
            result,
            Err(super::PressureObserverError::UnsupportedPlatform)
        ));
        assert_eq!(starts.load(Ordering::Relaxed), 1);
        assert_eq!(cancels.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn one_observer_feeds_the_process_registry_sink() {
        let registry = crate::capacity::CapacityRegistry::new(["escha".to_owned()]);
        let (pressure_source, pressure, starts, cancels) = FakeProducer::new();
        let (cadence_source, _cadence, cadence_starts, cadence_cancels) = FakeProducer::new();
        let (counters, calls) = FakeCounters::new([VmCounters::default(), VmCounters::default()]);

        let handle = PressureObserverConfig::new(pressure_source, counters, cadence_source)
            .start(Arc::clone(&registry))
            .await
            .unwrap();
        pressure.emit(ObserverEvent::Pressure(MemoryPressure::Constrained));
        wait_for_samples(&calls, 2).await;

        assert_eq!(
            registry.snapshot("escha").unwrap().pressure,
            MemoryPressure::Constrained
        );
        assert_eq!(starts.load(Ordering::Relaxed), 1);
        assert_eq!(cadence_starts.load(Ordering::Relaxed), 1);
        handle.stop().await.unwrap();
        assert_eq!(cancels.load(Ordering::Relaxed), 1);
        assert_eq!(cadence_cancels.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn periodic_counter_sample_detects_swap_without_pressure_transition() {
        let clock = TestClock::new();
        let controller = Arc::new(tokio::sync::Mutex::new(CapacityController::with_clock(
            controller_inputs(),
            clock,
        )));
        let (pressure_source, _pressure, _starts, _cancels) = FakeProducer::new();
        let (cadence_source, cadence, _starts, _cancels) = FakeProducer::new();
        let (counters, calls) = FakeCounters::new([
            VmCounters {
                swap_outs: 12,
                compressions: 30,
            },
            VmCounters {
                swap_outs: 13,
                compressions: 30,
            },
        ]);
        let handle = PressureObserverConfig::new(pressure_source, counters, cadence_source)
            .start(Arc::clone(&controller))
            .await
            .unwrap();

        cadence.emit(ObserverEvent::Sample);
        wait_for_samples(&calls, 2).await;

        let controller = controller.lock().await;
        assert_eq!(controller.pressure(), MemoryPressure::Critical);
        assert_eq!(
            controller.decision().availability,
            CapacityAvailability::Unavailable
        );
        drop(controller);
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn periodic_counter_sample_detects_compression_without_pressure_transition() {
        let clock = TestClock::new();
        let controller = Arc::new(tokio::sync::Mutex::new(CapacityController::with_clock(
            controller_inputs(),
            clock,
        )));
        let (pressure_source, _pressure, _starts, _cancels) = FakeProducer::new();
        let (cadence_source, cadence, _starts, _cancels) = FakeProducer::new();
        let (counters, calls) = FakeCounters::new([
            VmCounters {
                swap_outs: 12,
                compressions: 30,
            },
            VmCounters {
                swap_outs: 12,
                compressions: 31,
            },
        ]);
        let handle = PressureObserverConfig::new(pressure_source, counters, cadence_source)
            .start(Arc::clone(&controller))
            .await
            .unwrap();

        cadence.emit(ObserverEvent::Sample);
        wait_for_samples(&calls, 2).await;

        let controller = controller.lock().await;
        assert_eq!(controller.pressure(), MemoryPressure::Constrained);
        assert_eq!(
            controller.decision().availability,
            CapacityAvailability::Available
        );
        drop(controller);
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn counter_rollback_is_zero_delta_and_becomes_the_new_baseline() {
        let clock = TestClock::new();
        let controller = Arc::new(tokio::sync::Mutex::new(CapacityController::with_clock(
            controller_inputs(),
            clock,
        )));
        let (pressure_source, _pressure, _starts, _cancels) = FakeProducer::new();
        let (cadence_source, cadence, _starts, _cancels) = FakeProducer::new();
        let (counters, calls) = FakeCounters::new([
            VmCounters {
                swap_outs: 100,
                compressions: 100,
            },
            VmCounters {
                swap_outs: 10,
                compressions: 10,
            },
            VmCounters {
                swap_outs: 11,
                compressions: 10,
            },
        ]);
        let handle = PressureObserverConfig::new(pressure_source, counters, cadence_source)
            .start(Arc::clone(&controller))
            .await
            .unwrap();

        cadence.emit(ObserverEvent::Sample);
        wait_for_samples(&calls, 2).await;
        assert_eq!(controller.lock().await.pressure(), MemoryPressure::Normal);

        cadence.emit(ObserverEvent::Sample);
        wait_for_samples(&calls, 3).await;
        assert_eq!(controller.lock().await.pressure(), MemoryPressure::Critical);
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn pressure_events_are_sampled_and_applied_exactly_once() {
        let clock = TestClock::new();
        let controller = Arc::new(tokio::sync::Mutex::new(CapacityController::with_clock(
            controller_inputs(),
            clock,
        )));
        let initial = controller.lock().await.decision().safe_total_tokens;
        let (pressure_source, pressure, _starts, _cancels) = FakeProducer::new();
        let (cadence_source, _cadence, _starts, _cancels) = FakeProducer::new();
        let (counters, calls) = FakeCounters::new([
            VmCounters::default(),
            VmCounters::default(),
            VmCounters::default(),
            VmCounters::default(),
        ]);
        let handle = PressureObserverConfig::new(pressure_source, counters, cadence_source)
            .start(Arc::clone(&controller))
            .await
            .unwrap();

        pressure.emit(ObserverEvent::Pressure(MemoryPressure::Constrained));
        wait_for_samples(&calls, 2).await;
        assert_eq!(
            controller.lock().await.decision().safe_total_tokens,
            floor_1024(initial * 75 / 100)
        );

        pressure.emit(ObserverEvent::Pressure(MemoryPressure::Critical));
        wait_for_samples(&calls, 3).await;
        assert_eq!(controller.lock().await.pressure(), MemoryPressure::Critical);

        pressure.emit(ObserverEvent::Pressure(MemoryPressure::Normal));
        wait_for_samples(&calls, 4).await;
        assert_eq!(controller.lock().await.pressure(), MemoryPressure::Normal);
        assert_eq!(calls.load(Ordering::Relaxed), 4);
        handle.stop().await.unwrap();
    }

    struct GatedProducer {
        emitter: FakeEmitter,
        cancels: Arc<AtomicUsize>,
        acknowledgement: Arc<StdMutex<Option<oneshot::Receiver<()>>>>,
    }

    struct GatedProducerHandle {
        emitter: FakeEmitter,
        cancels: Arc<AtomicUsize>,
        acknowledgement: oneshot::Receiver<()>,
    }

    impl ProducerHandle for GatedProducerHandle {
        fn cancel(self: Box<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
            self.cancels.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut sender) = self.emitter.sender.lock() {
                sender.take();
            }
            let acknowledgement = self.acknowledgement;
            Box::pin(async move {
                let _ = acknowledgement.await;
            })
        }
    }

    impl PressureEventSource for GatedProducer {
        fn start(
            self,
            sender: tokio::sync::mpsc::UnboundedSender<ObserverEvent>,
        ) -> Result<Box<dyn ProducerHandle>, super::PressureObserverError> {
            if let Ok(mut slot) = self.emitter.sender.lock() {
                *slot = Some(sender);
            }
            let acknowledgement = self
                .acknowledgement
                .lock()
                .ok()
                .and_then(|mut acknowledgement| acknowledgement.take())
                .ok_or(super::PressureObserverError::UnsupportedPlatform)?;
            Ok(Box::new(GatedProducerHandle {
                emitter: self.emitter,
                cancels: self.cancels,
                acknowledgement,
            }))
        }
    }

    #[tokio::test]
    async fn stop_awaits_pressure_cancellation_before_cadence_and_join() {
        let clock = TestClock::new();
        let controller = Arc::new(tokio::sync::Mutex::new(CapacityController::with_clock(
            controller_inputs(),
            clock,
        )));
        let emitter = FakeEmitter {
            sender: Arc::new(StdMutex::new(None)),
        };
        let cancels = Arc::new(AtomicUsize::new(0));
        let (acknowledge, acknowledged) = oneshot::channel();
        let gated = GatedProducer {
            emitter: emitter.clone(),
            cancels: Arc::clone(&cancels),
            acknowledgement: Arc::new(StdMutex::new(Some(acknowledged))),
        };
        let (cadence_source, _cadence, _starts, cadence_cancels) = FakeProducer::new();
        let (counters, _calls) = FakeCounters::new([VmCounters::default()]);
        let handle = PressureObserverConfig::new(gated, counters, cadence_source)
            .start(Arc::clone(&controller))
            .await
            .unwrap();

        let stopping = tokio::spawn(handle.stop());
        wait_for_samples(&cancels, 1).await;
        assert!(!stopping.is_finished());
        assert_eq!(cadence_cancels.load(Ordering::Relaxed), 0);
        emitter.emit(ObserverEvent::Pressure(MemoryPressure::Critical));
        assert_eq!(controller.lock().await.pressure(), MemoryPressure::Normal);

        acknowledge.send(()).unwrap();
        stopping.await.unwrap().unwrap();
        assert_eq!(cadence_cancels.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn drop_fallback_cancels_producers_and_aborts_the_consumer() {
        let clock = TestClock::new();
        let controller = Arc::new(tokio::sync::Mutex::new(CapacityController::with_clock(
            controller_inputs(),
            clock,
        )));
        let (pressure_source, pressure, _starts, pressure_cancels) = FakeProducer::new();
        let (cadence_source, cadence, _starts, cadence_cancels) = FakeProducer::new();
        let (counters, _calls) = FakeCounters::new([VmCounters::default()]);
        let handle = PressureObserverConfig::new(pressure_source, counters, cadence_source)
            .start(Arc::clone(&controller))
            .await
            .unwrap();

        drop(handle);
        assert_eq!(pressure_cancels.load(Ordering::Relaxed), 1);
        assert_eq!(cadence_cancels.load(Ordering::Relaxed), 1);
        pressure.emit(ObserverEvent::Pressure(MemoryPressure::Critical));
        cadence.emit(ObserverEvent::Sample);
        tokio::task::yield_now().await;
        assert_eq!(controller.lock().await.pressure(), MemoryPressure::Normal);
    }
}
