// SPDX-License-Identifier: Apache-2.0
//! Disk-backed wrapper around the in-memory paged prefix cache.

use std::path::PathBuf;

use half::f16;
use higgs_models::AnyCache;
use higgs_models::cache::{KeyValueCache, SteppingKeyValueCache, slice_axis2};
use mlx_rs::error::Exception;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use crate::cache::disk_storage::{
    DiskCacheBlock, DiskCacheError, DiskCacheFileHeader, DiskCacheLayer, DiskCacheSnapshot,
    DiskStorage,
};
use crate::paged_prefix_cache::{PagedPrefixCache, PagedPrefixMatch};

pub const DEFAULT_MIN_TOKENS_TO_PERSIST: usize = 512;
pub const DEFAULT_MAX_DISK_BLOCKS: usize = 4096;

#[derive(Debug, Clone)]
pub struct DiskPrefixCacheConfig {
    pub disk_path: PathBuf,
    pub max_disk_blocks: usize,
    pub min_tokens_to_persist: usize,
}

/// Prefix cache that mirrors durable dense KV snapshots to disk.
pub struct DiskPrefixCache {
    memory: PagedPrefixCache,
    storage: Option<DiskStorage>,
    block_size: usize,
    min_tokens_to_persist: usize,
}

impl DiskPrefixCache {
    pub fn memory_only(max_entries: usize, block_size: usize) -> Self {
        Self {
            memory: PagedPrefixCache::new(max_entries, block_size),
            storage: None,
            block_size,
            min_tokens_to_persist: usize::MAX,
        }
    }

    pub fn new(
        max_entries: usize,
        block_size: usize,
        config: DiskPrefixCacheConfig,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, DiskCacheError> {
        let DiskPrefixCacheConfig {
            disk_path,
            max_disk_blocks,
            min_tokens_to_persist,
        } = config;
        let storage = DiskStorage::open(
            &disk_path,
            block_size,
            max_disk_blocks,
            num_kv_heads,
            head_dim,
        )?;
        Ok(Self {
            memory: PagedPrefixCache::new(max_entries, block_size),
            storage: Some(storage),
            block_size,
            min_tokens_to_persist,
        })
    }

    /// Find the longest matching prefix. Disk snapshots are consulted before
    /// memory, then the longer of the two hits is returned.
    pub fn find_longest_prefix(
        &mut self,
        tokens: &[u32],
        checkpoint_id: Option<&str>,
    ) -> Option<PagedPrefixMatch> {
        let disk_match = self.find_disk_prefix(tokens, checkpoint_id);
        let memory_match = self.memory.find_longest_prefix(tokens);
        match (disk_match, memory_match) {
            (Some(disk), Some(memory)) if memory.prefix_len > disk.prefix_len => Some(memory),
            (Some(disk), _) => Some(disk),
            (None, memory) => memory,
        }
    }

    /// Store in memory and, when large enough, append a dense f16 snapshot to
    /// disk. Unsupported cache shapes remain memory-only.
    pub fn store(&mut self, prefix_tokens: &[u32], cache: &AnyCache, checkpoint_id: Option<&str>) {
        self.memory.store(prefix_tokens, cache);

        if self.storage.is_none() || prefix_tokens.len() < self.min_tokens_to_persist {
            return;
        }

        let stored_len = prefix_tokens.len() / self.block_size * self.block_size;
        if stored_len < self.min_tokens_to_persist {
            return;
        }
        let Some(tokens_to_store) = prefix_tokens.get(..stored_len) else {
            return;
        };
        let token_hash = hash_tokens(tokens_to_store);

        let Some(storage) = self.storage.as_mut() else {
            return;
        };
        let header = storage.header().clone();
        let layers = match snapshot_layers(cache, &header, stored_len) {
            Ok(layers) => layers,
            Err(DiskCacheError::Unsupported(reason)) => {
                tracing::debug!(reason, "Skipping disk prefix cache store");
                return;
            }
            Err(error) => {
                tracing::warn!(error = %error, "Failed to build disk prefix cache snapshot");
                return;
            }
        };

        let token_session_id = hash_tokens_for_session(tokens_to_store);
        if let Err(error) = storage.save_blocks(token_session_id, token_hash, stored_len, &layers) {
            tracing::warn!(error = %error, "Failed to persist disk prefix cache snapshot");
        }
        if let Some(checkpoint) = checkpoint_id {
            let checkpoint_session_id = hash_checkpoint_id(checkpoint);
            if checkpoint_session_id != token_session_id {
                if let Err(error) =
                    storage.save_blocks(checkpoint_session_id, token_hash, stored_len, &layers)
                {
                    tracing::warn!(
                        checkpoint_id = checkpoint,
                        error = %error,
                        "Failed to persist named disk prefix cache checkpoint"
                    );
                }
            }
        }
    }

    pub const fn len(&self) -> usize {
        self.memory.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    pub fn clear(&mut self) {
        self.memory.clear();
    }

    fn find_disk_prefix(
        &mut self,
        tokens: &[u32],
        checkpoint_id: Option<&str>,
    ) -> Option<PagedPrefixMatch> {
        if let Some(checkpoint) = checkpoint_id {
            let checkpoint_session_id = hash_checkpoint_id(checkpoint);
            if let Some(found) = self.load_snapshot_match(tokens, checkpoint_session_id) {
                return Some(found);
            }
        }

        let mut candidate_len = tokens.len() / self.block_size * self.block_size;
        while candidate_len >= self.block_size {
            let Some(candidate_tokens) = tokens.get(..candidate_len) else {
                break;
            };
            let session_id = hash_tokens_for_session(candidate_tokens);
            if let Some(found) = self.load_snapshot_match(tokens, session_id) {
                return Some(found);
            }
            let Some(next_candidate) = candidate_len.checked_sub(self.block_size) else {
                break;
            };
            candidate_len = next_candidate;
        }
        None
    }

    fn load_snapshot_match(&mut self, tokens: &[u32], session_id: u64) -> Option<PagedPrefixMatch> {
        let storage = self.storage.as_ref()?;
        let snapshot = match storage.load_blocks(session_id) {
            Ok(Some(snapshot)) => snapshot,
            Ok(None) => return None,
            Err(error) => {
                tracing::warn!(error = %error, "Failed to load disk prefix cache snapshot");
                return None;
            }
        };
        if snapshot.token_count > tokens.len() {
            return None;
        }
        let prefix_tokens = tokens.get(..snapshot.token_count)?;
        if hash_tokens(prefix_tokens) != snapshot.token_hash {
            tracing::debug!("Skipping disk prefix cache snapshot with mismatched token hash");
            return None;
        }
        let header = storage.header().clone();
        let cache = match materialize_snapshot(&snapshot, &header) {
            Ok(cache) => cache,
            Err(error) => {
                tracing::warn!(error = %error, "Failed to materialize disk prefix cache snapshot");
                return None;
            }
        };
        self.memory.store(prefix_tokens, &cache);
        Some(PagedPrefixMatch {
            prefix_len: snapshot.token_count,
            cache,
        })
    }
}

pub fn hash_tokens(tokens: &[u32]) -> u64 {
    let mut hash = FNV_OFFSET;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

fn hash_tokens_for_session(tokens: &[u32]) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = fnv_byte(hash, b't');
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash = fnv_byte(hash, byte);
        }
    }
    hash
}

fn hash_checkpoint_id(checkpoint_id: &str) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = fnv_byte(hash, b'c');
    for byte in checkpoint_id.as_bytes() {
        hash = fnv_byte(hash, *byte);
    }
    hash
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn fnv_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
}

fn snapshot_layers(
    cache: &AnyCache,
    header: &DiskCacheFileHeader,
    stored_len: usize,
) -> Result<Vec<DiskCacheLayer>, DiskCacheError> {
    let AnyCache::KV(layers) = cache else {
        return Err(DiskCacheError::Unsupported("hybrid caches are memory-only"));
    };
    let block_count = stored_len / header.block_size;
    let block_size_i32 =
        i32::try_from(header.block_size).map_err(|_| DiskCacheError::Overflow("block_size"))?;
    let block_elems = header
        .block_size
        .checked_mul(header.num_kv_heads)
        .and_then(|value| value.checked_mul(header.head_dim))
        .ok_or(DiskCacheError::Overflow("block elements"))?;

    let mut disk_layers = Vec::with_capacity(layers.len());
    for layer in layers {
        let Some(kv) = layer.as_ref() else {
            disk_layers.push(DiskCacheLayer { blocks: Vec::new() });
            continue;
        };
        if kv.is_quantized() {
            return Err(DiskCacheError::Unsupported(
                "TurboQuant caches are memory-only",
            ));
        }
        let (Some(keys), Some(values)) = (kv.keys(), kv.values()) else {
            disk_layers.push(DiskCacheLayer { blocks: Vec::new() });
            continue;
        };
        validate_array_layout(keys, header, stored_len)?;
        validate_array_layout(values, header, stored_len)?;

        let mut blocks = Vec::with_capacity(block_count);
        for block_index in 0..block_count {
            let start_usize = block_index
                .checked_mul(header.block_size)
                .ok_or(DiskCacheError::Overflow("block start"))?;
            let start =
                i32::try_from(start_usize).map_err(|_| DiskCacheError::Overflow("block start"))?;
            let end = start
                .checked_add(block_size_i32)
                .ok_or(DiskCacheError::Overflow("block end"))?;
            let k_block = array_block_to_f16(keys, start, end, block_elems)?;
            let v_block = array_block_to_f16(values, start, end, block_elems)?;
            blocks.push(DiskCacheBlock {
                k: k_block,
                v: v_block,
            });
        }
        disk_layers.push(DiskCacheLayer { blocks });
    }
    Ok(disk_layers)
}

fn validate_array_layout(
    array: &Array,
    header: &DiskCacheFileHeader,
    stored_len: usize,
) -> Result<(), DiskCacheError> {
    let shape = array.shape();
    let heads = shape_dim(shape, 1, "num_kv_heads")?;
    let tokens = shape_dim(shape, 2, "tokens")?;
    let head_dim = shape_dim(shape, 3, "head_dim")?;
    if heads != header.num_kv_heads || head_dim != header.head_dim || tokens < stored_len {
        return Err(DiskCacheError::Format(format!(
            "array layout mismatch: shape={shape:?}, expected heads={} head_dim={} tokens>={stored_len}",
            header.num_kv_heads, header.head_dim
        )));
    }
    Ok(())
}

fn shape_dim(shape: &[i32], index: usize, label: &'static str) -> Result<usize, DiskCacheError> {
    let value = shape
        .get(index)
        .copied()
        .ok_or_else(|| DiskCacheError::Format(format!("array missing {label} dimension")))?;
    usize::try_from(value).map_err(|_| DiskCacheError::Format(format!("invalid {label} dimension")))
}

fn array_block_to_f16(
    array: &Array,
    start: i32,
    end: i32,
    block_elems: usize,
) -> Result<Vec<f16>, DiskCacheError> {
    let block =
        slice_axis2(array, start, end).map_err(|error| DiskCacheError::Mlx(format!("{error}")))?;
    let block_f16 = block
        .as_dtype(Dtype::Float16)
        .map_err(|error| DiskCacheError::Mlx(format!("{error}")))?;
    let data = block_f16.as_slice::<f16>().to_vec();
    if data.len() != block_elems {
        return Err(DiskCacheError::Format(
            "sliced block element count does not match layout".to_owned(),
        ));
    }
    Ok(data)
}

fn materialize_snapshot(
    snapshot: &DiskCacheSnapshot,
    header: &DiskCacheFileHeader,
) -> Result<AnyCache, DiskCacheError> {
    let layers: Result<Vec<_>, _> = snapshot
        .layers
        .iter()
        .map(|layer| {
            if layer.blocks.is_empty() {
                return Ok(Some(SteppingKeyValueCache::new()));
            }
            materialize_layer(layer, header).map(Some)
        })
        .collect();
    Ok(AnyCache::KV(layers?))
}

fn materialize_layer(
    layer: &DiskCacheLayer,
    header: &DiskCacheFileHeader,
) -> Result<SteppingKeyValueCache, DiskCacheError> {
    let shape = [
        1,
        i32::try_from(header.num_kv_heads).map_err(|_| DiskCacheError::Overflow("num_kv_heads"))?,
        i32::try_from(header.block_size).map_err(|_| DiskCacheError::Overflow("block_size"))?,
        i32::try_from(header.head_dim).map_err(|_| DiskCacheError::Overflow("head_dim"))?,
    ];
    let key_arrays: Vec<Array> = layer
        .blocks
        .iter()
        .map(|block| Array::from_slice(&block.k, &shape))
        .collect();
    let value_arrays: Vec<Array> = layer
        .blocks
        .iter()
        .map(|block| Array::from_slice(&block.v, &shape))
        .collect();
    let keys =
        concat_blocks(key_arrays).map_err(|error| DiskCacheError::Mlx(format!("{error}")))?;
    let values =
        concat_blocks(value_arrays).map_err(|error| DiskCacheError::Mlx(format!("{error}")))?;
    SteppingKeyValueCache::from_arrays(keys, values)
        .map_err(|error| DiskCacheError::Mlx(format!("{error}")))
}

fn concat_blocks(mut arrays: Vec<Array>) -> Result<Array, Exception> {
    if arrays.len() == 1 {
        return arrays
            .pop()
            .ok_or_else(|| Exception::custom("missing disk cache block"));
    }
    concatenate_axis(&arrays, 2)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use higgs_models::cache::KeyValueCache;

    fn make_kv_cache(num_layers: usize, seq_len: i32) -> AnyCache {
        let layers: Vec<Option<SteppingKeyValueCache>> = (0..num_layers)
            .map(|_| {
                let elem_count = usize::try_from(2 * seq_len * 4).unwrap();
                let values = vec![1.0_f32; elem_count];
                let keys = Array::from_slice(&values, &[1, 2, seq_len, 4]);
                let vals = Array::from_slice(&values, &[1, 2, seq_len, 4]);
                Some(SteppingKeyValueCache::from_arrays(keys, vals).unwrap())
            })
            .collect();
        AnyCache::KV(layers)
    }

    #[test]
    fn disk_cache_restores_into_empty_memory_cache() {
        let dir = tempfile::tempdir().unwrap();
        let config = DiskPrefixCacheConfig {
            disk_path: dir.path().join("prefix.bin"),
            max_disk_blocks: 16,
            min_tokens_to_persist: 32,
        };
        let tokens: Vec<u32> = (0..64).collect();
        let cache = make_kv_cache(2, 64);

        let mut writer = DiskPrefixCache::new(8, 32, config.clone(), 2, 4).unwrap();
        writer.store(&tokens, &cache, Some("checkpoint-a"));
        drop(writer);

        let mut reader = DiskPrefixCache::new(8, 32, config, 2, 4).unwrap();
        let mut query = tokens.clone();
        query.push(999);
        let matched = reader.find_longest_prefix(&query, None).unwrap();
        assert_eq!(matched.prefix_len, 64);
        match matched.cache {
            AnyCache::KV(layers) => {
                assert_eq!(layers.len(), 2);
                let kv = layers[0].as_ref().unwrap();
                assert_eq!(KeyValueCache::offset(kv), 64);
                assert_eq!(kv.keys().unwrap().shape(), &[1, 2, 64, 4]);
            }
            AnyCache::Hybrid(_) => panic!("expected KV cache"),
        }
    }

    #[test]
    fn named_checkpoint_validates_prompt_hash() {
        let dir = tempfile::tempdir().unwrap();
        let config = DiskPrefixCacheConfig {
            disk_path: dir.path().join("prefix.bin"),
            max_disk_blocks: 16,
            min_tokens_to_persist: 32,
        };
        let tokens: Vec<u32> = (0..64).collect();
        let cache = make_kv_cache(1, 64);

        let mut writer = DiskPrefixCache::new(8, 32, config.clone(), 2, 4).unwrap();
        writer.store(&tokens, &cache, Some("checkpoint-a"));
        drop(writer);

        let mut reader = DiskPrefixCache::new(8, 32, config, 2, 4).unwrap();
        let wrong_tokens: Vec<u32> = (1000..1064).collect();
        assert!(
            reader
                .find_longest_prefix(&wrong_tokens, Some("checkpoint-a"))
                .is_none()
        );
    }
}
