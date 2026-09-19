use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

use super::{
    CapacityPressureCoordinator, CapacityRegistry, MemoryPressure, PressureObservation,
    PressureTelemetry,
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
        telemetry: PressureTelemetry,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

impl PressureObservationSink for CapacityRegistry {
    fn apply<'a>(
        &'a self,
        observation: PressureObservation,
        telemetry: PressureTelemetry,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            self.apply_pressure_observation_with_telemetry(observation, telemetry);
        })
    }
}

impl PressureObservationSink for CapacityPressureCoordinator {
    fn apply<'a>(
        &'a self,
        observation: PressureObservation,
        telemetry: PressureTelemetry,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            self.capacity
                .apply_pressure_observation_with_telemetry(observation, telemetry);
        })
    }
}

fn unix_time_millis() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
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
        let mut sample_epoch = 0_u64;
        let mut latest_sample = prior.and_then(|sample| {
            let sampled_at_unix_ms = unix_time_millis()?;
            sample_epoch = 1;
            Some((sample, sampled_at_unix_ms, sample_epoch))
        });
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
            let mut non_normal_pressure_event_epoch = 0_u64;
            while let Some(event) = receiver.recv().await {
                if let ObserverEvent::Pressure(pressure) = event {
                    reported_pressure = pressure;
                    if pressure != MemoryPressure::Normal {
                        non_normal_pressure_event_epoch =
                            non_normal_pressure_event_epoch.saturating_add(1);
                    }
                }
                let current = counters.sample();
                let (swap_out_delta, compressor_delta) = match (prior, current) {
                    (Some(previous), Some(current)) => (
                        current.swap_outs.saturating_sub(previous.swap_outs),
                        current.compressions.saturating_sub(previous.compressions),
                    ),
                    _ => (0, 0),
                };
                if let Some(current) = current {
                    prior = Some(current);
                    if let Some(sampled_at_unix_ms) = unix_time_millis() {
                        sample_epoch = sample_epoch.saturating_add(1);
                        latest_sample = Some((current, sampled_at_unix_ms, sample_epoch));
                    }
                }
                let telemetry = latest_sample.map_or_else(
                    || PressureTelemetry {
                        raw_pressure: reported_pressure,
                        non_normal_pressure_event_epoch,
                        ..PressureTelemetry::default()
                    },
                    |(sample, sampled_at_unix_ms, sample_epoch)| PressureTelemetry {
                        raw_pressure: reported_pressure,
                        swap_outs_total: Some(sample.swap_outs),
                        compressions_total: Some(sample.compressions),
                        sampled_at_unix_ms: Some(sampled_at_unix_ms),
                        sample_epoch: Some(sample_epoch),
                        non_normal_pressure_event_epoch,
                    },
                );
                sink.apply(
                    PressureObservation {
                        pressure: reported_pressure,
                        swap_out_delta,
                        compressor_delta,
                    },
                    telemetry,
                )
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
