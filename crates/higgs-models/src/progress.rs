//! Prefill progress reporting via a thread-scoped sink.
//!
//! The chunked-prefill loops live deep in the model crate with no access to
//! the engine's streaming channel; threading a callback through every
//! `forward_chunked` signature (the `AnyModel` zoo plus per-model overrides
//! and their test callers) would churn a dozen call sites for one optional
//! observer. Engines run generation on a dedicated blocking thread, so a
//! thread-local sink installed for the duration of one prefill is exact and
//! invisible to every other code path.

use std::cell::{Cell, RefCell};
use std::path::Path;

use crate::error::ModelError;

type Sink = Box<dyn FnMut(i32, i32) -> Result<(), ModelError>>;
type LoadSink = Box<dyn FnMut(LoadBoundary) -> Result<(), ModelError>>;

/// The kind of allocation-producing conversion about to run while loading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConversionKind {
    FinalModelEval,
    QwenMaterialization,
    NativeEscha,
    AffineEscha,
    GemmaExpertReshape,
    FullArtifact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptionalModelKind {
    DFlash,
    PrefillDrafter,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptionalModelDisposition {
    Retained,
    Discarded,
}

/// A capacity checkpoint at a concrete model-loader allocation boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadBoundary {
    BeforeShard {
        index: usize,
        bytes: u64,
    },
    AfterShard {
        index: usize,
    },
    BeforeConversion {
        index: usize,
        bytes: u64,
        kind: ConversionKind,
    },
    AfterConversion {
        index: usize,
        kind: ConversionKind,
    },
    BeforeOptionalModel {
        artifact_bytes: u64,
        workspace_bytes: u64,
        kind: OptionalModelKind,
    },
    AfterOptionalModel {
        artifact_bytes: u64,
        kind: OptionalModelKind,
        disposition: OptionalModelDisposition,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptionalLoadPolicy {
    Allow,
    Suppress,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OptionalLoadOutcome {
    pub dflash_loaded: bool,
    pub prefill_drafter_loaded: bool,
}

thread_local! {
    static PREFILL_SINK: RefCell<Option<Sink>> = const { RefCell::new(None) };
    static LOAD_SINK: RefCell<Option<LoadSink>> = const { RefCell::new(None) };
    static OPTIONAL_LOAD_POLICY: Cell<OptionalLoadPolicy> = const { Cell::new(OptionalLoadPolicy::Allow) };
    static OPTIONAL_LOAD_OUTCOME: Cell<OptionalLoadOutcome> = const { Cell::new(OptionalLoadOutcome { dflash_loaded: false, prefill_drafter_loaded: false }) };
}

pub struct OptionalLoadPolicyGuard {
    previous: OptionalLoadPolicy,
    previous_outcome: OptionalLoadOutcome,
}

impl Drop for OptionalLoadPolicyGuard {
    fn drop(&mut self) {
        OPTIONAL_LOAD_POLICY.with(|policy| policy.set(self.previous));
        OPTIONAL_LOAD_OUTCOME.with(|outcome| outcome.set(self.previous_outcome));
    }
}

pub fn install_optional_load_policy(policy: OptionalLoadPolicy) -> OptionalLoadPolicyGuard {
    let previous = OPTIONAL_LOAD_POLICY.with(|current| current.replace(policy));
    let previous_outcome =
        OPTIONAL_LOAD_OUTCOME.with(|current| current.replace(OptionalLoadOutcome::default()));
    OptionalLoadPolicyGuard {
        previous,
        previous_outcome,
    }
}

pub fn optional_load_policy() -> OptionalLoadPolicy {
    OPTIONAL_LOAD_POLICY.with(Cell::get)
}

pub fn suppress_optional_loads() {
    OPTIONAL_LOAD_POLICY.with(|policy| policy.set(OptionalLoadPolicy::Suppress));
}

pub fn record_dflash_loaded() {
    OPTIONAL_LOAD_OUTCOME.with(|outcome| {
        let mut current = outcome.get();
        current.dflash_loaded = true;
        outcome.set(current);
    });
}

pub fn record_prefill_drafter_loaded() {
    OPTIONAL_LOAD_OUTCOME.with(|outcome| {
        let mut current = outcome.get();
        current.prefill_drafter_loaded = true;
        outcome.set(current);
    });
}

pub fn optional_load_outcome() -> OptionalLoadOutcome {
    OPTIONAL_LOAD_OUTCOME.with(Cell::get)
}

/// Restores the thread's previous loader sink, including across unwind.
pub struct LoadBoundarySinkGuard {
    previous: Option<LoadSink>,
}

impl Drop for LoadBoundarySinkGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        LOAD_SINK.with(|sink| *sink.borrow_mut() = previous);
    }
}

/// Install one typed capacity sink for the duration of a model load.
pub fn install_load_boundary_sink(sink: LoadSink) -> LoadBoundarySinkGuard {
    let previous = LOAD_SINK.with(|current| current.borrow_mut().replace(sink));
    LoadBoundarySinkGuard { previous }
}

/// Report a loader boundary. The no-sink path is deliberately fallible-shaped
/// so allocation loops can propagate policy rejection with `?`.
pub fn report_load_boundary(boundary: LoadBoundary) -> Result<(), ModelError> {
    LOAD_SINK.with(|sink| match sink.borrow_mut().as_mut() {
        Some(sink) => sink(boundary),
        None => Ok(()),
    })
}

/// Report a file-backed shard before opening it. Metadata failure is load
/// failure rather than a zero-byte estimate, because zero would bypass policy.
pub fn report_before_shard(index: usize, path: &Path) -> Result<u64, ModelError> {
    let metadata = std::fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(ModelError::Io(std::io::Error::other(format!(
            "model shard is not a regular file: {}",
            path.display()
        ))));
    }
    let bytes = metadata.len();
    report_load_boundary(LoadBoundary::BeforeShard { index, bytes })?;
    Ok(bytes)
}

pub fn report_after_shard(index: usize) -> Result<(), ModelError> {
    report_load_boundary(LoadBoundary::AfterShard { index })
}

/// RAII guard that removes the thread's prefill sink on drop, keeping
/// installs scoped to a single prefill call.
pub struct PrefillSinkGuard;

impl Drop for PrefillSinkGuard {
    fn drop(&mut self) {
        PREFILL_SINK.with(|s| *s.borrow_mut() = None);
    }
}

/// Install a prefill-progress sink for the current thread.
///
/// The sink receives `(processed, total)` after each completed prefill chunk.
/// `processed` is the cumulative number of tokens forwarded so far in *this*
/// prefill — i.e. relative to the suffix that survived prefix-cache reuse, not
/// a per-chunk delta and not an absolute prompt offset. Callers that want an
/// absolute prompt position add the cached-prefix length themselves. Hold the
/// returned guard for the duration of the prefill; dropping it uninstalls the
/// sink.
///
/// The sink must not re-enter the progress machinery: calling
/// [`report_prefill_progress`] or installing another sink from inside the sink
/// callback panics, because the thread-local is `borrow_mut`-held while the
/// sink runs.
pub fn install_prefill_progress_sink(sink: Sink) -> PrefillSinkGuard {
    PREFILL_SINK.with(|s| *s.borrow_mut() = Some(sink));
    PrefillSinkGuard
}

/// Report chunked-prefill progress: `processed` of `total` tokens done.
/// No-op when no sink is installed (the common path: one `thread_local`
/// lookup + `Option` check per ~1024-token chunk).
///
/// The sink is invoked while the thread-local is `borrow_mut`-held, so the
/// sink must not call back into this function or reinstall the sink.
pub fn report_prefill_progress(processed: i32, total: i32) -> Result<(), ModelError> {
    PREFILL_SINK.with(|s| match s.borrow_mut().as_mut() {
        Some(f) => f(processed, total),
        None => Ok(()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Reports reach the sink only while the guard is alive; the no-sink
    /// path is a silent no-op (the invariant every model forward relies on).
    #[test]
    fn test_sink_scoped_by_guard() {
        let seen: Rc<RefCell<Vec<(i32, i32)>>> = Rc::new(RefCell::new(Vec::new()));

        // No sink installed — must not panic, must not record.
        report_prefill_progress(512, 4096).unwrap();
        assert!(seen.borrow().is_empty());

        let sink_seen = Rc::clone(&seen);
        let guard = install_prefill_progress_sink(Box::new(move |p, t| {
            sink_seen.borrow_mut().push((p, t));
            Ok(())
        }));
        report_prefill_progress(1024, 4096).unwrap();
        report_prefill_progress(2048, 4096).unwrap();
        drop(guard);

        // After the guard drops, reports are no-ops again.
        report_prefill_progress(3072, 4096).unwrap();
        assert_eq!(*seen.borrow(), vec![(1024, 4096), (2048, 4096)]);
    }

    /// A sink that observes cancellation returns `Err` and the report
    /// propagates it, so chunked prefill loops abort at the next chunk
    /// boundary instead of continuing to allocate.
    #[test]
    fn test_sink_abort_propagates() {
        let guard =
            install_prefill_progress_sink(Box::new(|_p, _t| Err(ModelError::PrefillCancelled)));
        assert!(matches!(
            report_prefill_progress(1024, 4096),
            Err(ModelError::PrefillCancelled)
        ));
        drop(guard);
        assert!(report_prefill_progress(2048, 4096).is_ok());
    }

    /// Removing the load-boundary reporter or failing to restore a nested sink
    /// would either lose ordering or leak policy into the next model load.
    #[test]
    fn load_boundary_sink_is_typed_nested_and_restored() {
        let outer = Rc::new(RefCell::new(Vec::new()));
        let outer_seen = Rc::clone(&outer);
        let _outer_guard = install_load_boundary_sink(Box::new(move |event| {
            outer_seen.borrow_mut().push(event);
            Ok(())
        }));

        report_load_boundary(LoadBoundary::BeforeShard {
            index: 0,
            bytes: 13,
        })
        .unwrap();
        {
            let inner = Rc::new(RefCell::new(Vec::new()));
            let inner_seen = Rc::clone(&inner);
            let _inner_guard = install_load_boundary_sink(Box::new(move |event| {
                inner_seen.borrow_mut().push(event);
                Err(crate::error::ModelError::LoadCapacity("stop".to_owned()))
            }));
            let error = report_load_boundary(LoadBoundary::BeforeConversion {
                index: 1,
                bytes: 29,
                kind: ConversionKind::NativeEscha,
            })
            .unwrap_err();
            assert!(
                matches!(error, crate::error::ModelError::LoadCapacity(message) if message == "stop")
            );
            assert_eq!(inner.borrow().len(), 1);
        }
        report_load_boundary(LoadBoundary::AfterShard { index: 0 }).unwrap();
        assert_eq!(
            *outer.borrow(),
            vec![
                LoadBoundary::BeforeShard {
                    index: 0,
                    bytes: 13,
                },
                LoadBoundary::AfterShard { index: 0 },
            ]
        );
    }

    /// A panicking load must not leave its capacity callback installed on the
    /// thread and poison a later independent load.
    #[test]
    fn load_boundary_sink_restores_after_unwind() {
        let calls = Rc::new(RefCell::new(0usize));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let calls_for_sink = Rc::clone(&calls);
            let _guard = install_load_boundary_sink(Box::new(move |_| {
                *calls_for_sink.borrow_mut() += 1;
                Ok(())
            }));
            report_load_boundary(LoadBoundary::AfterConversion {
                index: 3,
                kind: ConversionKind::FinalModelEval,
            })
            .unwrap();
            panic!("synthetic loader panic");
        }));
        assert!(caught.is_err());
        report_load_boundary(LoadBoundary::AfterConversion {
            index: 4,
            kind: ConversionKind::FinalModelEval,
        })
        .unwrap();
        assert_eq!(*calls.borrow(), 1);
    }

    /// Warning-pressure policy is scoped to one load and cannot leak into the
    /// next normal model load, including through panic unwind.
    #[test]
    fn optional_load_policy_is_scoped_and_restored() {
        assert_eq!(optional_load_policy(), OptionalLoadPolicy::Allow);
        let caught = std::panic::catch_unwind(|| {
            let _guard = install_optional_load_policy(OptionalLoadPolicy::Suppress);
            assert_eq!(optional_load_policy(), OptionalLoadPolicy::Suppress);
            panic!("synthetic load panic");
        });
        assert!(caught.is_err());
        assert_eq!(optional_load_policy(), OptionalLoadPolicy::Allow);
    }
}
