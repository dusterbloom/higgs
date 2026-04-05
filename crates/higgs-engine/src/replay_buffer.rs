//! Replay buffer for adaptive memory — stores training signals from inference.
//!
//! Each entry records a (prompt + completion) token sequence, its surprise score,
//! and a reward signal that accumulates from implicit/explicit feedback. The idle
//! trainer samples high-priority entries for gradient steps.

use std::collections::HashMap;
use std::time::Instant;

/// A single replay entry: one request's token sequence + metadata.
#[derive(Debug, Clone)]
pub struct ReplayEntry {
    pub request_id: String,
    /// Full token sequence (prompt + completion).
    pub tokens: Vec<u32>,
    /// Number of prompt tokens (loss computed only on completion portion).
    pub prompt_len: usize,
    /// Mean negative log-probability of the completion tokens.
    pub surprise: f32,
    /// Accumulated reward from implicit/explicit feedback.
    pub reward: f32,
    /// When this entry was created.
    pub created: Instant,
    /// If true, this entry is exempt from eviction.
    pub pinned: bool,
    /// How many times this entry has been used for training.
    pub train_count: u32,
}

/// Bounded replay buffer with priority-based eviction.
pub struct ReplayBuffer {
    entries: Vec<ReplayEntry>,
    capacity: usize,
    id_index: HashMap<String, usize>,
    /// Minimum surprise to admit a new entry.
    pub surprise_threshold: f32,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, surprise_threshold: f32) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            id_index: HashMap::with_capacity(capacity),
            surprise_threshold,
        }
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Try to insert a new entry. Returns false if surprise is below threshold.
    pub fn push(&mut self, entry: ReplayEntry) -> bool {
        if entry.surprise < self.surprise_threshold {
            return false;
        }

        // Evict if at capacity
        if self.entries.len() >= self.capacity {
            if !self.evict_one() {
                // All entries are pinned and buffer is full — reject
                return false;
            }
        }

        let idx = self.entries.len();
        self.id_index.insert(entry.request_id.clone(), idx);
        self.entries.push(entry);
        true
    }

    /// Look up an entry by request_id.
    pub fn get(&self, request_id: &str) -> Option<&ReplayEntry> {
        self.id_index.get(request_id).map(|&i| &self.entries[i])
    }

    /// Mutably look up an entry by request_id.
    pub fn get_mut(&mut self, request_id: &str) -> Option<&mut ReplayEntry> {
        self.id_index
            .get(request_id)
            .copied()
            .map(|i| &mut self.entries[i])
    }

    /// Update reward for a given request_id. Returns true if found.
    pub fn update_reward(&mut self, request_id: &str, delta: f32) -> bool {
        if let Some(entry) = self.get_mut(request_id) {
            entry.reward += delta;
            // Auto-evict heavily penalized entries
            if entry.reward < -1.0 && !entry.pinned {
                self.remove(request_id);
            }
            true
        } else {
            false
        }
    }

    /// Pin an entry (exempt from eviction).
    pub fn pin(&mut self, request_id: &str) -> bool {
        if let Some(entry) = self.get_mut(request_id) {
            entry.pinned = true;
            true
        } else {
            false
        }
    }

    /// Pick the highest-priority entry for training.
    ///
    /// Priority = `reward * surprise` where `train_count < max_trains`.
    pub fn pick_for_training(&self, max_trains: u32) -> Option<&ReplayEntry> {
        self.entries
            .iter()
            .filter(|e| e.train_count < max_trains)
            .max_by(|a, b| {
                let score_a = a.reward * a.surprise + a.surprise * 0.5;
                let score_b = b.reward * b.surprise + b.surprise * 0.5;
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Increment the train count for an entry.
    pub fn mark_trained(&mut self, request_id: &str) {
        if let Some(entry) = self.get_mut(request_id) {
            entry.train_count += 1;
        }
    }

    /// Remove an entry by request_id.
    pub fn remove(&mut self, request_id: &str) -> Option<ReplayEntry> {
        let idx = self.id_index.remove(request_id)?;
        let entry = self.entries.swap_remove(idx);
        // Fix the index of the entry that was swapped in
        if idx < self.entries.len() {
            let swapped_id = self.entries[idx].request_id.clone();
            self.id_index.insert(swapped_id, idx);
        }
        Some(entry)
    }

    /// Replace the completion tokens in an entry (for corrections).
    pub fn replace_completion(&mut self, request_id: &str, new_tokens: Vec<u32>, prompt_len: usize) {
        if let Some(entry) = self.get_mut(request_id) {
            entry.tokens = new_tokens;
            entry.prompt_len = prompt_len;
            entry.train_count = 0; // Reset so it gets retrained
        }
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &ReplayEntry> {
        self.entries.iter()
    }

    /// Evict the lowest-scoring non-pinned entry. Returns true if an entry was evicted.
    fn evict_one(&mut self) -> bool {
        let now = Instant::now();
        let victim = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.pinned)
            .min_by(|(_, a), (_, b)| {
                let score_a = Self::eviction_score(a, now);
                let score_b = Self::eviction_score(b, now);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        if let Some(idx) = victim {
            let id = self.entries[idx].request_id.clone();
            self.remove(&id);
            true
        } else {
            false
        }
    }

    /// Score for eviction: lower = more likely to be evicted.
    fn eviction_score(entry: &ReplayEntry, now: Instant) -> f32 {
        let age_hours = now.duration_since(entry.created).as_secs_f32() / 3600.0;
        entry.reward + entry.surprise * 0.5 - age_hours * 0.1
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn make_entry(id: &str, surprise: f32, reward: f32) -> ReplayEntry {
        ReplayEntry {
            request_id: id.to_owned(),
            tokens: vec![1, 2, 3, 4, 5],
            prompt_len: 2,
            surprise,
            reward,
            created: Instant::now(),
            pinned: false,
            train_count: 0,
        }
    }

    #[test]
    fn push_and_retrieve() {
        let mut buf = ReplayBuffer::new(10, 0.5);
        let entry = make_entry("req-1", 2.0, 0.0);
        assert!(buf.push(entry));
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.get("req-1").unwrap().surprise, 2.0);
    }

    #[test]
    fn reject_below_threshold() {
        let mut buf = ReplayBuffer::new(10, 1.0);
        let entry = make_entry("req-low", 0.5, 0.0);
        assert!(!buf.push(entry));
        assert!(buf.is_empty());
    }

    #[test]
    fn eviction_at_capacity() {
        let mut buf = ReplayBuffer::new(2, 0.0);
        buf.push(make_entry("a", 1.0, 0.0));
        buf.push(make_entry("b", 3.0, 1.0));
        // Buffer full, pushing c should evict the lowest-scoring entry
        buf.push(make_entry("c", 2.0, 0.5));
        assert_eq!(buf.len(), 2);
        // "a" had lowest score (0 + 1.0*0.5 - age ~= 0.5) so it should be evicted
        assert!(buf.get("a").is_none());
        assert!(buf.get("b").is_some());
        assert!(buf.get("c").is_some());
    }

    #[test]
    fn pinned_entries_survive_eviction() {
        let mut buf = ReplayBuffer::new(2, 0.0);
        buf.push(make_entry("pinned", 0.1, -0.5));
        buf.pin("pinned");
        buf.push(make_entry("normal", 5.0, 1.0));
        // Attempting to push a third — only "normal" can be evicted (pinned is exempt)
        // but "normal" has higher score, so it would be the victim since it's the only
        // non-pinned candidate
        buf.push(make_entry("new", 3.0, 0.0));
        assert_eq!(buf.len(), 2);
        assert!(buf.get("pinned").is_some());
    }

    #[test]
    fn reward_update() {
        let mut buf = ReplayBuffer::new(10, 0.0);
        buf.push(make_entry("req-1", 2.0, 0.0));
        buf.update_reward("req-1", 1.0);
        assert_eq!(buf.get("req-1").unwrap().reward, 1.0);
        buf.update_reward("req-1", -2.5);
        // reward = 1.0 - 2.5 = -1.5, below -1.0 and not pinned → auto-evicted
        assert!(buf.get("req-1").is_none());
    }

    #[test]
    fn pick_for_training_respects_max_trains() {
        let mut buf = ReplayBuffer::new(10, 0.0);
        let mut e = make_entry("trained", 5.0, 2.0);
        e.train_count = 10;
        buf.push(e);
        buf.push(make_entry("fresh", 1.0, 0.5));
        let picked = buf.pick_for_training(10).unwrap();
        assert_eq!(picked.request_id, "fresh");
    }

    #[test]
    fn mark_trained_increments() {
        let mut buf = ReplayBuffer::new(10, 0.0);
        buf.push(make_entry("req-1", 2.0, 0.0));
        buf.mark_trained("req-1");
        buf.mark_trained("req-1");
        assert_eq!(buf.get("req-1").unwrap().train_count, 2);
    }

    #[test]
    fn replace_completion() {
        let mut buf = ReplayBuffer::new(10, 0.0);
        buf.push(make_entry("req-1", 2.0, 0.0));
        buf.replace_completion("req-1", vec![10, 20, 30, 40], 1);
        let e = buf.get("req-1").unwrap();
        assert_eq!(e.tokens, vec![10, 20, 30, 40]);
        assert_eq!(e.prompt_len, 1);
        assert_eq!(e.train_count, 0);
    }

    #[test]
    fn remove_fixes_indices() {
        let mut buf = ReplayBuffer::new(10, 0.0);
        buf.push(make_entry("a", 1.0, 0.0));
        buf.push(make_entry("b", 2.0, 0.0));
        buf.push(make_entry("c", 3.0, 0.0));
        buf.remove("a");
        // After swap_remove, "c" should now be at index 0
        assert!(buf.get("b").is_some());
        assert!(buf.get("c").is_some());
        assert_eq!(buf.len(), 2);
    }
}
