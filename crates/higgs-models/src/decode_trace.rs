//! Per-token, per-layer decode trace emitter.
//!
//! Activated by `HIGGS_DECODE_TRACE=/path/to/trace.jsonl`. When unset, all
//! entry points compile down to a single atomic load + branch — zero cost
//! on the decode hot path.
//!
//! Each call to [`begin_forward`] bumps the token counter. Each layer of
//! the forward pass calls [`record_layer`] with the routing decision.
//! Output is one JSON object per line, flushed per token.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

struct Tracer {
    writer: Mutex<BufWriter<File>>,
    tok: AtomicU64,
}

static TRACER: OnceLock<Option<Tracer>> = OnceLock::new();

fn init() -> Option<Tracer> {
    let path = std::env::var("HIGGS_DECODE_TRACE").ok()?;
    if path.is_empty() {
        return None;
    }
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .inspect_err(|e| tracing::error!("HIGGS_DECODE_TRACE: cannot open {path}: {e}"))
        .ok()?;
    tracing::info!("decode-trace active → {path}");
    Some(Tracer {
        writer: Mutex::new(BufWriter::new(file)),
        tok: AtomicU64::new(0),
    })
}

fn tracer() -> Option<&'static Tracer> {
    TRACER.get_or_init(init).as_ref()
}

/// True if `HIGGS_DECODE_TRACE` is set and the file opened.
#[inline]
pub fn is_active() -> bool {
    tracer().is_some()
}

/// Bump the token counter and return the new value. Call once per forward
/// pass before the layer loop. Returns 0 when tracing is off.
pub fn begin_forward(seq_len: usize) -> u64 {
    let Some(t) = tracer() else {
        return 0;
    };
    let tok = t.tok.fetch_add(1, Ordering::Relaxed);
    let ts_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);
    let mut buf = format!(r#"{{"evt":"forward","tok":{tok},"seq_len":{seq_len},"ts_us":{ts_us}}}"#);
    buf.push('\n');
    if let Ok(mut w) = t.writer.lock() {
        let _ = w.write_all(buf.as_bytes());
    }
    tok
}

/// Emit one row describing a single layer's routing decision.
///
/// `kind`: "attn_full" or "attn_linear"
/// `mlp_path`: "moe", "ane_int8", "fp16_dense", "quantized_fused"
#[allow(clippy::too_many_arguments)]
pub fn record_layer(
    tok: u64,
    layer: usize,
    kind: &'static str,
    mlp_path: &'static str,
    seq_len: usize,
    hidden: usize,
    ns: u64,
) {
    let Some(t) = tracer() else { return };
    let mut buf = format!(
        r#"{{"evt":"layer","tok":{tok},"layer":{layer},"kind":"{kind}","mlp":"{mlp_path}","seq_len":{seq_len},"hidden":{hidden},"ns":{ns}}}"#
    );
    buf.push('\n');
    if let Ok(mut w) = t.writer.lock() {
        let _ = w.write_all(buf.as_bytes());
    }
}

/// Flush the underlying BufWriter. Call after a forward pass completes if
/// you want rows visible before process exit. Cheap when tracing is off.
pub fn flush() {
    if let Some(t) = tracer()
        && let Ok(mut w) = t.writer.lock()
    {
        let _ = w.flush();
    }
}
