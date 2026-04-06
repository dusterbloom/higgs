//! Adaptive memory manager — orchestrates replay buffer, implicit feedback,
//! idle training, and delta persistence.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::config::MemoryConfig;
use crate::state::Engine;
use higgs_engine::replay_buffer::{ReplayBuffer, ReplayEntry};
use higgs_models::delta_persistence::{
    compress_deltas, load_deltas, load_replay_metadata, save_deltas, save_replay_metadata,
    ReplayMeta,
};

/// Central orchestrator for the adaptive memory system.
pub struct AdaptiveMemoryManager {
    pub replay_buffer: Mutex<ReplayBuffer>,
    /// Unix millis of the last inference request.
    pub last_request_time: AtomicU64,
    pub config: MemoryConfig,
    /// Cached system prompt tokens (captured from first request).
    system_prompt_tokens: Mutex<Option<Vec<u32>>>,
    /// Total idle training steps performed.
    idle_train_steps: AtomicU64,
    /// Recent request hashes for re-prompt detection.
    recent_hashes: Mutex<Vec<(u64, String)>>,
}

impl AdaptiveMemoryManager {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            replay_buffer: Mutex::new(ReplayBuffer::new(
                config.replay_buffer_size,
                config.surprise_threshold,
            )),
            last_request_time: AtomicU64::new(now_millis()),
            config,
            system_prompt_tokens: Mutex::new(None),
            idle_train_steps: AtomicU64::new(0),
            recent_hashes: Mutex::new(Vec::new()),
        }
    }

    /// Record that a request just started (updates idle timer).
    pub fn touch(&self) {
        self.last_request_time.store(now_millis(), Ordering::Relaxed);
    }

    /// Milliseconds since last request.
    pub fn idle_ms(&self) -> u64 {
        now_millis().saturating_sub(self.last_request_time.load(Ordering::Relaxed))
    }

    /// Total idle training steps performed.
    pub fn total_train_steps(&self) -> u64 {
        self.idle_train_steps.load(Ordering::Relaxed)
    }

    /// Cache the system prompt tokens (called once from first request).
    pub fn set_system_prompt(&self, tokens: Vec<u32>) {
        let mut sp = self.system_prompt_tokens.lock().unwrap_or_else(|e| e.into_inner());
        if sp.is_none() {
            *sp = Some(tokens);
        }
    }

    /// Get cached system prompt tokens.
    pub fn system_prompt_tokens(&self) -> Option<Vec<u32>> {
        self.system_prompt_tokens.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    // -----------------------------------------------------------------------
    // Implicit feedback
    // -----------------------------------------------------------------------

    /// Detect implicit feedback signals from an incoming request.
    ///
    /// Returns `Vec<(request_id, reward_delta)>` for entries that should be updated.
    pub fn detect_implicit_feedback(
        &self,
        prompt_tokens: &[u32],
        has_tool_result: bool,
        user_message_text: Option<&str>,
    ) -> Vec<(String, f32)> {
        let mut updates = Vec::new();
        let buf = self.replay_buffer.lock().unwrap_or_else(|e| e.into_inner());

        // Continuation detection: if the prompt prefix matches a stored entry's
        // full token sequence, the user continued the conversation → positive signal.
        // Skip entries with empty tokens (restored metadata without token data).
        for entry in buf.iter() {
            if !entry.tokens.is_empty()
                && prompt_tokens.len() > entry.tokens.len()
                && prompt_tokens[..entry.tokens.len()] == entry.tokens[..]
            {
                updates.push((entry.request_id.clone(), 0.5));
            }
        }

        // Tool result: if the request contains a tool result message, the previous
        // tool call was executed → positive signal for the entry that generated it.
        if has_tool_result {
            // Find the most recent entry (last pushed)
            if let Some(last) = buf.iter().last() {
                updates.push((last.request_id.clone(), 0.3));
            }
        }

        drop(buf);

        // Re-prompt detection: same user message hash as a recent request → negative signal.
        if let Some(text) = user_message_text {
            let hash = hash_str(text);
            let mut recent = self.recent_hashes.lock().unwrap_or_else(|e| e.into_inner());

            for (stored_hash, req_id) in recent.iter() {
                if *stored_hash == hash {
                    updates.push((req_id.clone(), -0.5));
                }
            }

            // We'll associate this hash with a request_id later (after generation).
            // For now just check for matches.
            // Keep bounded: last 32 requests
            if recent.len() >= 32 {
                recent.remove(0);
            }
        }

        updates
    }

    /// Record a request hash for future re-prompt detection.
    pub fn record_request_hash(&self, user_text: &str, request_id: &str) {
        let hash = hash_str(user_text);
        let mut recent = self.recent_hashes.lock().unwrap_or_else(|e| e.into_inner());
        if recent.len() >= 32 {
            recent.remove(0);
        }
        recent.push((hash, request_id.to_owned()));
    }

    /// Apply reward updates from implicit/explicit feedback.
    pub fn apply_feedback(&self, updates: &[(String, f32)]) {
        let mut buf = self.replay_buffer.lock().unwrap_or_else(|e| e.into_inner());
        for (req_id, delta) in updates {
            buf.update_reward(req_id, *delta);
        }
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Save deltas and replay metadata to disk. Never panics.
    pub fn save_state(&self, engine: &Engine) {
        let model_name = engine.model_name();
        let dir = memory_data_dir(model_name);

        // Save deltas
        match engine.get_deltas() {
            Ok(Some(mut deltas)) => {
                let budget = self.config.delta_budget_mb as usize * 1024 * 1024;
                if let Err(e) = compress_deltas(&mut deltas, budget) {
                    tracing::warn!(error = %e, "[MEMORY] delta compression failed");
                }
                let path = dir.join("deltas.safetensors");
                match save_deltas(&deltas, &path) {
                    Ok(()) => {
                        tracing::info!(
                            path = %path.display(),
                            n_tensors = deltas.len(),
                            "[MEMORY] deltas saved"
                        );
                    }
                    Err(e) => tracing::warn!(error = %e, "[MEMORY] failed to save deltas"),
                }
            }
            Ok(None) => {
                tracing::debug!("[MEMORY] no deltas to save");
            }
            Err(e) => {
                tracing::warn!(error = %e, "[MEMORY] failed to get deltas from engine");
            }
        }

        // Save replay metadata
        let buf = self.replay_buffer.lock().unwrap_or_else(|e| e.into_inner());
        let metas: Vec<ReplayMeta> = buf
            .iter()
            .map(|e| ReplayMeta {
                request_id: e.request_id.clone(),
                surprise: e.surprise,
                reward: e.reward,
                pinned: e.pinned,
                train_count: e.train_count,
            })
            .collect();
        drop(buf);

        if !metas.is_empty() {
            let path = dir.join("replay.json");
            match save_replay_metadata(&metas, &path) {
                Ok(()) => {
                    tracing::info!(
                        path = %path.display(),
                        n_entries = metas.len(),
                        "[MEMORY] replay metadata saved"
                    );
                }
                Err(e) => tracing::warn!(error = %e, "[MEMORY] failed to save replay metadata"),
            }
        }
    }

    /// Load deltas and replay metadata from disk. Missing files = cold start.
    pub fn load_state(&self, engine: &Engine) {
        let model_name = engine.model_name();
        let dir = memory_data_dir(model_name);

        // Load deltas (missing file = cold start, not an error)
        let deltas_path = dir.join("deltas.safetensors");
        match load_deltas(&deltas_path) {
            Ok(deltas) => {
                let n = deltas.len();
                match engine.set_deltas(deltas) {
                    Ok(()) => {
                        tracing::info!(
                            path = %deltas_path.display(),
                            n_tensors = n,
                            "[MEMORY] deltas loaded"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "[MEMORY] failed to set deltas on engine");
                    }
                }
            }
            Err(e) if e.contains("Failed to read") => {
                tracing::debug!("[MEMORY] no persisted deltas found (cold start)");
            }
            Err(e) => {
                tracing::warn!(error = %e, "[MEMORY] failed to load deltas");
            }
        }

        // Load replay metadata (missing file = cold start)
        let replay_path = dir.join("replay.json");
        match load_replay_metadata(&replay_path) {
            Ok(metas) => {
                let n = metas.len();
                let mut buf = self.replay_buffer.lock().unwrap_or_else(|e| e.into_inner());
                let mut dropped = 0usize;
                for meta in metas {
                    let ok = buf.push_unchecked(ReplayEntry {
                        request_id: meta.request_id,
                        tokens: Vec::new(), // tokens not persisted
                        prompt_len: 0,
                        surprise: meta.surprise,
                        reward: meta.reward,
                        created: std::time::Instant::now(),
                        pinned: meta.pinned,
                        train_count: meta.train_count,
                    });
                    if !ok { dropped += 1; }
                }
                if dropped > 0 {
                    tracing::warn!(dropped, "[MEMORY] restored entries dropped (buffer full)");
                }
                tracing::info!(
                    path = %replay_path.display(),
                    n_entries = n - dropped,
                    "[MEMORY] replay metadata loaded"
                );
            }
            Err(e) if e.contains("Failed to read") => {
                tracing::debug!("[MEMORY] no persisted replay metadata found (cold start)");
            }
            Err(e) => {
                tracing::warn!(error = %e, "[MEMORY] failed to load replay metadata");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Idle training
    // -----------------------------------------------------------------------

    /// Run a single idle training step if conditions are met.
    ///
    /// Returns `Some(loss)` if training happened, `None` if skipped.
    pub fn train_one_step(&self, engine: &Engine) -> Option<f32> {
        // Double-check we're still idle
        let idle_ms = self.idle_ms();
        if idle_ms < self.config.idle_timeout_secs * 1000 {
            return None;
        }

        // Pick highest-priority entry
        let (tokens, prompt_len, request_id) = {
            let buf = self.replay_buffer.lock().unwrap_or_else(|e| e.into_inner());
            let entry = buf.pick_for_training(self.config.max_trains_per_entry)?;
            (
                entry.tokens.clone(),
                entry.prompt_len,
                entry.request_id.clone(),
            )
        };

        // Check idle again before locking the model
        if self.idle_ms() < self.config.idle_timeout_secs * 1000 {
            return None;
        }

        let eggroll_config = higgs_models::diffusion_eggroll::EggrollConfig {
            sigma: 0.001,
            lr: self.config.lr,
            rank: self.config.rank,
            population: 1,
            total_steps: 1,
            warmup_steps: 0,
            min_lr_frac: 0.1,
            log_interval: 0,
            base_seed: 42,
            clip_ratio: 0.05,
            delta_decay: 0.001,
        };

        tracing::info!(
            request_id = %request_id,
            seq_len = tokens.len(),
            "[MEMORY] idle training step"
        );

        match engine.train_gradient(eggroll_config, &tokens, prompt_len) {
            Ok(losses) => {
                let loss = losses.last().copied().unwrap_or(0.0);
                self.replay_buffer.lock().unwrap_or_else(|e| e.into_inner()).mark_trained(&request_id);
                self.idle_train_steps.fetch_add(1, Ordering::Relaxed);

                tracing::info!(
                    loss = format!("{loss:.4}"),
                    total_steps = self.total_train_steps(),
                    "[MEMORY] idle train step complete"
                );

                Some(loss)
            }
            Err(e) => {
                tracing::warn!(error = %e, "[MEMORY] idle training failed");
                None
            }
        }
    }
}

/// Spawn the idle training loop as a background tokio task.
pub fn spawn_idle_trainer(memory: Arc<AdaptiveMemoryManager>, engine: Arc<Engine>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            let idle_ms = memory.idle_ms();
            if idle_ms >= memory.config.idle_timeout_secs * 1000 {
                let mem = Arc::clone(&memory);
                let eng = Arc::clone(&engine);
                tokio::task::spawn_blocking(move || {
                    if mem.train_one_step(&eng).is_some()
                        && mem.total_train_steps() % 10 == 0
                    {
                        mem.save_state(&eng);
                    }
                })
                .await
                .ok();
            }
        }
    });
}

/// Build the persistence directory for a given model name.
fn memory_data_dir(model_name: &str) -> PathBuf {
    let sanitized = model_name.replace('/', "--");
    crate::config::config_dir().join("memory").join(sanitized)
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn hash_str(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, unsafe_code)]
mod tests {
    use super::*;

    fn test_config() -> MemoryConfig {
        MemoryConfig {
            enabled: true,
            replay_buffer_size: 10,
            idle_timeout_secs: 1,
            rank: 4,
            lr: 0.0005,
            surprise_threshold: 0.5,
            delta_budget_mb: 32,
            max_trains_per_entry: 10,
            anchor_kl_threshold: 0.1,
        }
    }

    #[test]
    fn manager_touch_updates_idle() {
        let mgr = AdaptiveMemoryManager::new(test_config());
        mgr.touch();
        assert!(mgr.idle_ms() < 100); // should be nearly zero
    }

    #[test]
    fn reprompt_detection() {
        let mgr = AdaptiveMemoryManager::new(test_config());

        // First request — no re-prompt
        mgr.record_request_hash("hello world", "req-1");
        let feedback = mgr.detect_implicit_feedback(&[], false, Some("hello world"));
        // Should detect re-prompt for req-1
        assert!(feedback.iter().any(|(id, r)| id == "req-1" && *r < 0.0));
    }

    #[test]
    fn tool_result_positive_feedback() {
        let mgr = AdaptiveMemoryManager::new(test_config());
        {
            let mut buf = mgr.replay_buffer.lock().unwrap_or_else(|e| e.into_inner());
            buf.push(ReplayEntry {
                request_id: "req-tool".to_owned(),
                tokens: vec![1, 2, 3],
                prompt_len: 1,
                surprise: 2.0,
                reward: 0.0,
                created: std::time::Instant::now(),
                pinned: false,
                train_count: 0,
            });
        }
        let feedback = mgr.detect_implicit_feedback(&[], true, None);
        assert!(feedback.iter().any(|(id, r)| id == "req-tool" && *r > 0.0));
    }

    #[test]
    fn system_prompt_cached_once() {
        let mgr = AdaptiveMemoryManager::new(test_config());
        mgr.set_system_prompt(vec![10, 20, 30]);
        mgr.set_system_prompt(vec![99, 99]); // should not overwrite
        assert_eq!(mgr.system_prompt_tokens(), Some(vec![10, 20, 30]));
    }

    /// E2E: simulates two server sessions. Session 1 trains (sets deltas on
    /// engine) and accumulates replay entries, then shuts down (save_state).
    /// Session 2 boots fresh (new engine + new manager), calls load_state,
    /// then verifies the engine holds bit-identical delta tensors AND the
    /// replay buffer contains the correct metadata.
    #[test]
    fn e2e_shutdown_save_startup_load() {
        use higgs_models::qwen3_next::DeltaMap;
        use mlx_rs::Array;

        let tmp = tempfile::tempdir().unwrap();
        // SAFETY: test runs with --test-threads=1, no concurrent env mutation.
        unsafe { std::env::set_var("HIGGS_CONFIG_DIR", tmp.path()) };

        // === Session 1: train + accumulate + shutdown ===
        let engine1 = Engine::test_stub("test-org/my-model");

        // Simulate training: set deltas on the engine
        let mut deltas = DeltaMap::new();
        deltas.insert(
            "layers.0.mlp.delta_A".to_owned(),
            Array::from_slice(&[0.1_f32, -0.2, 0.3, 0.0], &[2, 2]),
        );
        deltas.insert(
            "layers.5.attn.delta_B".to_owned(),
            Array::from_slice(&[1.0_f32, 2.0, 3.0], &[1, 3]),
        );
        engine1.set_deltas(deltas).unwrap();

        // Verify engine actually holds them
        assert!(engine1.get_deltas().unwrap().is_some(), "engine must hold deltas after set");

        let mgr1 = AdaptiveMemoryManager::new(test_config());
        {
            let mut buf = mgr1.replay_buffer.lock().unwrap();
            buf.push(ReplayEntry {
                request_id: "high-surprise".to_owned(),
                tokens: vec![10, 20, 30, 40, 50],
                prompt_len: 2,
                surprise: 5.0,
                reward: 1.5,
                created: std::time::Instant::now(),
                pinned: true,
                train_count: 3,
            });
            buf.push(ReplayEntry {
                request_id: "normal".to_owned(),
                tokens: vec![1, 2, 3],
                prompt_len: 1,
                surprise: 2.0,
                reward: -0.2,
                created: std::time::Instant::now(),
                pinned: false,
                train_count: 0,
            });
        }

        // Shutdown: save everything
        mgr1.save_state(&engine1);

        // Verify files on disk
        let data_dir = tmp.path().join("memory").join("test-org--my-model");
        assert!(data_dir.join("deltas.safetensors").exists());
        assert!(data_dir.join("replay.json").exists());

        // === Session 2: fresh boot ===
        let engine2 = Engine::test_stub("test-org/my-model");
        assert!(engine2.get_deltas().unwrap().is_none(), "fresh engine has no deltas");

        let mgr2 = AdaptiveMemoryManager::new(test_config());
        assert!(mgr2.replay_buffer.lock().unwrap().is_empty());

        // Startup: load persisted state
        mgr2.load_state(&engine2);

        // --- Verify deltas are bit-identical ---
        let loaded = engine2.get_deltas().unwrap().expect("engine must have deltas after load");
        assert_eq!(loaded.len(), 2);

        let a = &loaded["layers.0.mlp.delta_A"];
        assert_eq!(a.shape(), &[2, 2]);
        let a_vals = a.as_slice::<f32>();
        assert_eq!(a_vals[0].to_bits(), 0.1_f32.to_bits());
        assert_eq!(a_vals[1].to_bits(), (-0.2_f32).to_bits());
        assert_eq!(a_vals[2].to_bits(), 0.3_f32.to_bits());
        assert_eq!(a_vals[3].to_bits(), 0.0_f32.to_bits());

        let b = &loaded["layers.5.attn.delta_B"];
        assert_eq!(b.shape(), &[1, 3]);
        assert_eq!(b.as_slice::<f32>(), &[1.0, 2.0, 3.0]);

        // --- Verify replay metadata ---
        let buf = mgr2.replay_buffer.lock().unwrap();
        assert_eq!(buf.len(), 2);

        let high = buf.get("high-surprise").unwrap();
        assert!((high.surprise - 5.0).abs() < f32::EPSILON);
        assert!((high.reward - 1.5).abs() < f32::EPSILON);
        assert!(high.pinned);
        assert_eq!(high.train_count, 3);
        assert!(high.tokens.is_empty(), "tokens not persisted");

        let normal = buf.get("normal").unwrap();
        assert!((normal.surprise - 2.0).abs() < f32::EPSILON);
        assert!((normal.reward - (-0.2)).abs() < f32::EPSILON);
        assert!(!normal.pinned);
        assert_eq!(normal.train_count, 0);

        // SAFETY: test runs with --test-threads=1, no concurrent env mutation.
        unsafe { std::env::remove_var("HIGGS_CONFIG_DIR") };
    }

    /// Restored entries (empty tokens) must be skipped by pick_for_training.
    /// Proves idle trainer won't crash on tokenless metadata from a previous session.
    #[test]
    fn restored_entries_skipped_by_trainer() {
        let mut buf = ReplayBuffer::new(10, 0.0);

        // Restored entry: empty tokens, highest priority score
        buf.push_unchecked(ReplayEntry {
            request_id: "restored".to_owned(),
            tokens: Vec::new(),
            prompt_len: 0,
            surprise: 10.0,
            reward: 5.0,
            created: std::time::Instant::now(),
            pinned: false,
            train_count: 0,
        });

        // Live entry: has tokens, lower priority
        buf.push(ReplayEntry {
            request_id: "live".to_owned(),
            tokens: vec![1, 2, 3, 4, 5],
            prompt_len: 2,
            surprise: 1.0,
            reward: 0.1,
            created: std::time::Instant::now(),
            pinned: false,
            train_count: 0,
        });

        let picked = buf.pick_for_training(10).unwrap();
        assert_eq!(picked.request_id, "live");

        buf.remove("live");
        assert!(buf.pick_for_training(10).is_none());
    }

    /// push_unchecked bypasses surprise threshold. Required because restored
    /// entries already passed the threshold in a previous session.
    #[test]
    fn push_unchecked_bypasses_threshold() {
        let mut buf = ReplayBuffer::new(10, 5.0);

        let rejected = buf.push(ReplayEntry {
            request_id: "low".to_owned(),
            tokens: vec![1],
            prompt_len: 0,
            surprise: 1.0,
            reward: 0.0,
            created: std::time::Instant::now(),
            pinned: false,
            train_count: 0,
        });
        assert!(!rejected);
        assert!(buf.is_empty());

        buf.push_unchecked(ReplayEntry {
            request_id: "low".to_owned(),
            tokens: vec![1],
            prompt_len: 0,
            surprise: 1.0,
            reward: 0.0,
            created: std::time::Instant::now(),
            pinned: false,
            train_count: 0,
        });
        assert_eq!(buf.len(), 1);
        assert!(buf.get("low").is_some());
    }

    /// Cold start: load_state on nonexistent dir is a no-op. Manager still works.
    #[test]
    fn cold_start_no_persisted_state() {
        let tmp = tempfile::tempdir().unwrap();
        // SAFETY: test runs with --test-threads=1, no concurrent env mutation.
        unsafe { std::env::set_var("HIGGS_CONFIG_DIR", &tmp.path().join("nope")) };

        let mgr = AdaptiveMemoryManager::new(test_config());
        let engine = Engine::test_stub("ghost-model");
        mgr.load_state(&engine);

        assert!(mgr.replay_buffer.lock().unwrap().is_empty());
        assert!(engine.get_deltas().unwrap().is_none());
        assert_eq!(mgr.total_train_steps(), 0);

        // SAFETY: test runs with --test-threads=1, no concurrent env mutation.
        unsafe { std::env::remove_var("HIGGS_CONFIG_DIR") };
    }

    /// Regression: restored entries (empty tokens) must NOT trigger false
    /// continuation detection. Before the fix, `prompt[..0] == []` matched
    /// every request → spurious +0.5 reward on all restored entries.
    #[test]
    fn restored_entries_no_false_continuation_signal() {
        let mgr = AdaptiveMemoryManager::new(test_config());
        {
            let mut buf = mgr.replay_buffer.lock().unwrap();
            buf.push_unchecked(ReplayEntry {
                request_id: "restored".to_owned(),
                tokens: Vec::new(),
                prompt_len: 0,
                surprise: 5.0,
                reward: 0.0,
                created: std::time::Instant::now(),
                pinned: false,
                train_count: 2,
            });
            buf.push(ReplayEntry {
                request_id: "live".to_owned(),
                tokens: vec![1, 2, 3],
                prompt_len: 1,
                surprise: 2.0,
                reward: 0.0,
                created: std::time::Instant::now(),
                pinned: false,
                train_count: 0,
            });
        }

        let prompt = &[1, 2, 3, 4, 5];
        let feedback = mgr.detect_implicit_feedback(prompt, false, None);

        assert!(
            feedback.iter().any(|(id, _)| id == "live"),
            "live entry must get continuation signal"
        );
        assert!(
            feedback.iter().all(|(id, _)| id != "restored"),
            "restored entry must not get false continuation signal"
        );
    }

    /// Benchmark: measures save/load latency with production-sized deltas.
    ///
    /// Simulates a 35B-A3B model's PCAST output: ~200 targets, each delta
    /// is a full-rank [out, in] matrix. Total ~10 MB of f32 tensor data.
    /// Also measures replay metadata I/O with a full 256-entry buffer.
    #[test]
    fn bench_persistence_latency() {
        use higgs_models::qwen3_next::DeltaMap;
        use mlx_rs::Array;

        let tmp = tempfile::tempdir().unwrap();
        // SAFETY: test runs with --test-threads=1, no concurrent env mutation.
        unsafe { std::env::set_var("HIGGS_CONFIG_DIR", tmp.path()) };

        let engine = Engine::test_stub("bench-model");

        // Build production-sized deltas: ~200 targets matching 35B-A3B dimensions.
        // MoE model: hidden=2048, intermediate=1408, 40 layers, ~5 targets/layer.
        let mut deltas = DeltaMap::new();
        let mut total_bytes: usize = 0;
        for layer in 0..40 {
            // Attention: q/k/v/o projections — [2048, 2048] each but rank-expanded
            // PCAST output: full-rank delta = lora_a @ lora_b.T
            // With rank=4: [2048, 4] @ [4, 2048] → [2048, 2048]... but that's huge.
            // In practice deltas are rank-compressed: stored as [out, rank] * [rank, in].
            // The actual DeltaMap stores the multiplied-out result for forward pass.
            // At rank=4 with 200 targets, ~9.7 MB total means ~48 KB per target avg.
            // That's ~12K floats → e.g., [96, 128] or similar small residuals.

            // Match real measured size: 9.7 MB / 200 targets ≈ 48 KB each ≈ 12288 floats
            for suffix in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.gate_proj", "mlp.up_proj"] {
                let name = format!("model.layers.{layer}.{suffix}");
                // [96, 128] = 12288 floats = 48 KB — matches real PCAST output
                let data: Vec<f32> = (0..12288).map(|i| (i as f32) * 1e-5).collect();
                let arr = Array::from_slice(&data, &[96, 128]);
                total_bytes += 12288 * 4;
                deltas.insert(name, arr);
            }
        }
        let n_deltas = deltas.len();
        eprintln!("[BENCH] {n_deltas} deltas, {:.1} MB", total_bytes as f64 / 1e6);

        engine.set_deltas(deltas).unwrap();

        // Build full replay buffer (256 entries)
        let mut cfg = test_config();
        cfg.replay_buffer_size = 256;
        cfg.surprise_threshold = 0.0;
        let mgr = AdaptiveMemoryManager::new(cfg);
        {
            let mut buf = mgr.replay_buffer.lock().unwrap();
            for i in 0..256 {
                buf.push(ReplayEntry {
                    request_id: format!("req-{i}"),
                    tokens: vec![1; 512], // typical request length
                    prompt_len: 128,
                    surprise: 2.0 + (i as f32) * 0.01,
                    reward: 0.1 * (i as f32 % 10.0),
                    created: std::time::Instant::now(),
                    pinned: i % 20 == 0,
                    train_count: (i % 5) as u32,
                });
            }
        }

        // --- Benchmark save ---
        let t0 = std::time::Instant::now();
        mgr.save_state(&engine);
        let save_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Verify files and sizes
        let data_dir = tmp.path().join("memory").join("bench-model");
        let delta_size = std::fs::metadata(data_dir.join("deltas.safetensors")).unwrap().len();
        let replay_size = std::fs::metadata(data_dir.join("replay.json")).unwrap().len();

        eprintln!("[BENCH] save: {save_ms:.1}ms (deltas: {:.1} MB, replay: {:.1} KB)",
            delta_size as f64 / 1e6, replay_size as f64 / 1e3);

        // --- Benchmark load ---
        let engine2 = Engine::test_stub("bench-model");
        let mut cfg2 = test_config();
        cfg2.replay_buffer_size = 256;
        cfg2.surprise_threshold = 0.0;
        let mgr2 = AdaptiveMemoryManager::new(cfg2);

        let t1 = std::time::Instant::now();
        mgr2.load_state(&engine2);
        let load_ms = t1.elapsed().as_secs_f64() * 1000.0;

        eprintln!("[BENCH] load: {load_ms:.1}ms");

        // Verify data survived
        let loaded = engine2.get_deltas().unwrap().unwrap();
        assert_eq!(loaded.len(), n_deltas);
        assert_eq!(mgr2.replay_buffer.lock().unwrap().len(), 256);

        // --- Overhead context ---
        // Compare to server startup time (~2-5s for model load) and
        // shutdown time (~0ms, just signal handling).
        eprintln!("[BENCH] overhead: save adds {save_ms:.1}ms to shutdown, load adds {load_ms:.1}ms to startup");
        eprintln!("[BENCH] for context: model load is ~2000-5000ms, so this is {:.1}% of startup",
            load_ms / 3000.0 * 100.0);

        // SAFETY: test runs with --test-threads=1, no concurrent env mutation.
        unsafe { std::env::remove_var("HIGGS_CONFIG_DIR") };
    }
}
