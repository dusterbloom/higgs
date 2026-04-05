//! Adaptive memory manager — orchestrates replay buffer, implicit feedback,
//! idle training, and delta persistence.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::config::MemoryConfig;
use crate::state::Engine;
use higgs_engine::replay_buffer::{ReplayBuffer, ReplayEntry};

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
        for entry in buf.iter() {
            if prompt_tokens.len() > entry.tokens.len()
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
                    mem.train_one_step(&eng);
                })
                .await
                .ok();
            }
        }
    });
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
#[allow(clippy::unwrap_used)]
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
}
