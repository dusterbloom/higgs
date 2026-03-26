# SSD Spill Tier Design for Higgs PagedAttention KV Cache

## Executive Summary

SSD spill tier sits below the in-RAM prefix cache. Target: Qwen3.5-35B-A3B-3bit on Apple M4 32GB. On Apple Silicon, "GPU memory" and "RAM" are identical unified memory -- no PCIe transfer penalty. The bottleneck for spill is NVMe SSD (~7 GB/s sequential).

---

## 1. Model Parameters: Qwen3.5-35B-A3B

| Parameter | Value |
|---|---|
| `num_hidden_layers` | 94 |
| `decoder_sparse_step` | 1 (every other layer is GDN) |
| KV attention layers | ~47 (non-GDN, standard attention) |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| Dtype | bf16 (2 bytes) |
| GDN layers | ~47 (no KV cache -- recurrent state only) |

---

## 2. Block Format Specification

### 2.1 Block Size Calculation

A **block** is 64 tokens of KV cache across all attention layers.

```
block_bytes = n_kv_layers x 2 (K+V) x n_tokens x n_kv_heads x head_dim x dtype_bytes
            = 47 x 2 x 64 x 8 x 128 x 2
            = 6,160,384 bytes
            ~ 5.88 MB per block
```

For 10-layer variant: `10 x 2 x 64 x 8 x 128 x 2 = 2.5 MB per block`

### 2.2 On-disk Block File Layout

Raw binary -- no safetensors overhead, no JSON metadata per block. Metadata in process index.

```
Offset  Size     Content
------  ------   -------
0       8 bytes  Magic: b"HIGSBLK\x01"
8       4 bytes  u32le: n_kv_layers
12      4 bytes  u32le: n_kv_heads
16      4 bytes  u32le: tokens_in_block (<= 64)
20      4 bytes  u32le: head_dim
24      4 bytes  u32le: dtype (0 = bf16, 1 = f16, 2 = f32)
28      4 bytes  u32le: reserved (zero)
32      32 bytes SHA-256 of data section (integrity check)
64      8 bytes  u64le: content_hash (xxhash3 of token sequence for cache key lookup)
72      8 bytes  u64le: block_id (unique per session, for orphan cleanup)
80      8 bytes  u64le: written_at_unix_ms
88      40 bytes reserved/padding to reach 128-byte header
128     N bytes  data: K tensors [n_kv_layers, n_kv_heads, tokens, head_dim] bf16, then V same shape
```

K and V stored as separate contiguous blobs in layer-major order for sequential read speed.

### 2.3 Slab File (Recommended over individual files)

- Single pre-allocated file: `higgs_kvcache.slab`
- Size set at startup (e.g. `slab_size_gb = 64`)
- Block slot index: `slot_id -> byte_offset = slot_id * BLOCK_SLOT_SIZE` (aligned to 4096)
- Free-list managed in memory
- On macOS: `fcntl(F_PREALLOCATE)` + `ftruncate`

---

## 3. Capacity Calculations

### 3.1 RAM Budget (47-layer model)

```
Total unified memory:        32 GB
Model weights (3-bit quant): ~13 GB
Runtime overhead:            ~1 GB
Prompt cache (in-RAM):       ~16 GB remaining

Blocks per RAM budget = 16 * 1024^3 / 6,160,384 ~ 2,795 blocks
Tokens in RAM cache   = 2,795 x 64 ~ 178,880 tokens ~ 175K tokens
```

### 3.2 SSD Budget

| SSD size | Slab budget | Blocks | Tokens |
|---|---|---|---|
| 256 GB | 200 GB | 34,130 | ~2.18M tokens |
| 512 GB | 440 GB | 75,000 | ~4.8M tokens |
| 1 TB | 900 GB | 153,600 | ~9.8M tokens |

### 3.3 Restore vs Recompute

```
Restore time for 1 block (5.88 MB) = 5.88 MB / 7000 MB/s = 0.84 ms
Recompute time for 64 tokens prefill on 35B model ~ 406 ms
```

**Restoring from SSD is ~480x faster than recomputing.** Even at 100 MB/s worst-case random reads: 58.8 ms, still ~7x faster.

---

## 4. I/O Performance Budget

### 4.1 Throughput Model

Apple M4 NVMe:
- Sequential read: ~7 GB/s
- Random 4K read: ~1M IOPS = ~4 GB/s effective
- Sequential write: ~5 GB/s

```
Max restore rate:  7,000 MB/s / 5.88 MB = 1,190 blocks/s
Max evict rate:    5,000 MB/s / 5.88 MB = 850 blocks/s
```

### 4.2 Practical Latency Budget

For 128K token context (2,000 blocks), 90% RAM hit rate:
- RAM misses: 200 blocks
- At 1,190 blocks/s sequential: 168 ms total restore time
- Acceptable as background prefetch

### 4.3 Async I/O Strategy (macOS)

Use `tokio::task::spawn_blocking` wrapping `pread`/`pwrite` on the slab file.

- `io_uring` is Linux-only
- `tokio::fs` wraps `spawn_blocking` internally on macOS anyway
- Use `fcntl(F_NOCACHE, 1)` on slab fd to bypass page cache (macOS equivalent of `O_DIRECT`)

**Pre-fetch pipeline:** When block B is `Cold`, speculatively fetch B+1..B+7 in background.

---

## 5. Eviction Policy: Clock-Pro (simplified ARC)

### 5.1 Algorithm

- Two lists: `T1` (recently-loaded, touched once) and `T2` (frequently-used, touched >= 2)
- Each block has `reference_bit` and `frequency_count`
- Clock hand sweeps T1 first; blocks with `reference_bit = 1` -> T2; `reference_bit = 0` -> evict
- T2 blocks get one extra pass before eviction

### 5.2 Reference Counting for Prefix Sharing

- Each block has `Arc<AtomicU32>` reference count
- Blocks referenced by live `PrefixMatch` have `refcount > 0` -- MUST NOT be evicted
- `refcount == 0` blocks are eligible for eviction
- `BlockPin` RAII guard increments on create, decrements on drop

### 5.3 Watermarks

| Watermark | Default | Behavior |
|---|---|---|
| `high_water` | 90% RAM budget | Start background eviction |
| `low_water` | 75% RAM budget | Stop eviction |
| `ssd_high_water` | 85% SSD slab | Start deleting oldest SSD blocks |

### 5.4 State Machine

```
           fill RAM                    evict to SSD (LRU, unpinned)
  Free ----------------> Hot ------------------------------------> Cold (on SSD)
   ^                      |                                          |
   |                      | pin acquired                             | restore request
   |                      v                                          v
   |                   Pinned <-------------------------------- Restoring
   |                      |
   |                      | pin released
   +----------------------+
                  (returns to Hot pool)

  Cold -> Evicted: triggered when SSD watermark exceeded
```

No "Warm" tier on Apple Silicon -- Hot and Warm are the same unified memory.

---

## 6. Rust Implementation Sketch

### 6.1 Core Types

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockKey(pub u64);  // xxhash3 of token slice

#[derive(Clone, Copy, Debug)]
pub struct SlabSlot(pub u32);

#[derive(Debug)]
pub enum BlockState {
    Hot { ram_slot: u32, refcount: Arc<AtomicU32> },
    Spilling { ram_slot: u32, refcount: Arc<AtomicU32> },
    Cold { slab_slot: SlabSlot },
    Restoring { slab_slot: SlabSlot, notify: Arc<tokio::sync::Notify> },
}

pub struct BlockMeta {
    pub key: BlockKey,
    pub state: BlockState,
    pub last_access_epoch: u64,
    pub token_count: u16,
    pub layer_range: std::ops::Range<u16>,
}

/// RAII pin guard
pub struct BlockPin {
    refcount: Arc<AtomicU32>,
}

impl Drop for BlockPin {
    fn drop(&mut self) {
        self.refcount.fetch_sub(1, Ordering::Release);
    }
}

pub struct BlockCache {
    index: Mutex<HashMap<BlockKey, BlockMeta>>,
    ram_slab: RamSlab,
    ssd_slab: SsdSlab,
    clock_hand: AtomicU32,
    epoch: AtomicU64,
    config: BlockCacheConfig,
}

pub struct BlockCacheConfig {
    pub ram_budget_bytes: u64,
    pub ssd_slab_path: PathBuf,
    pub ssd_slab_bytes: u64,
    pub block_size_tokens: u16,       // default: 64
    pub high_water_fraction: f32,      // default: 0.90
    pub low_water_fraction: f32,       // default: 0.75
    pub prefetch_lookahead: u8,        // default: 8 blocks
}
```

### 6.2 Slab I/O

```rust
pub struct SsdSlab {
    fd: std::os::fd::OwnedFd,
    slot_size: u64,
    free_slots: Mutex<Vec<SlabSlot>>,
    total_slots: u32,
}

impl SsdSlab {
    pub async fn write_block(&self, slot: SlabSlot, data: &[u8]) -> io::Result<()> {
        // spawn_blocking + pwrite at slot offset
    }
    pub async fn read_block(&self, slot: SlabSlot, buf: &mut Vec<u8>) -> io::Result<()> {
        // spawn_blocking + pread at slot offset
    }
}
```

### 6.3 Integration with PrefixCache

1. **Eviction hook:** When `PrefixCache::evict_lru()` evicts an `AnyCache` >= 1 block, serialize and spill to SSD
2. **Lookup extension:** `BlockCache::find_longest_prefix()` called first; on cold hit, issues restore + returns `RestoringFuture`

### 6.4 Key Trait

```rust
#[async_trait::async_trait]
pub trait TieredPrefixCache {
    fn find_hot(&mut self, tokens: &[u32]) -> Option<PrefixMatch>;
    async fn find_with_restore(&mut self, tokens: &[u32]) -> Option<PrefixMatch>;
    async fn store(&mut self, prefix_tokens: &[u32], cache: AnyCache);
}
```

---

## 7. Risk Analysis

### 7.1 AnyCache Serialization

MLX arrays are opaque C++ objects. Serialize via `array.eval()` + `as_slice::<u8>()` (already used in TurboQuant). Must call `eval()` on inference thread before handing to I/O thread.

### 7.2 Hybrid Cache Complexity

Only KV attention layers should be spilled; GDN states must remain in RAM. Block file `layer_range` field identifies stored layers.

### 7.3 macOS File Cache Pollution

Use `fcntl(F_NOCACHE, 1)` on slab fd to bypass page cache.

### 7.4 Crash Safety

Write data first, then header (with SHA-256) as commit record. Partial writes with no header treated as free slots on restart.

### 7.5 Prototyping Priority

1. AnyCache -> bytes round-trip correctness
2. Slab I/O throughput on M4 with `F_NOCACHE`
3. Clock-Pro eviction with shared prefix tree
4. Integration latency (evict -> spill -> restore -> re-pin)

---

## 8. Summary

| Decision | Recommendation | Rationale |
|---|---|---|
| Format | Raw binary slab, 128-byte header | Zero parsing overhead |
| I/O | `spawn_blocking` + `pread`/`pwrite` + `F_NOCACHE` | macOS-native |
| Eviction | Clock-Pro (simplified ARC) | Scan-resistance for shared prefixes |
| Block size | 64 tokens x all KV layers | Matches cache chunks |
| RAM capacity | ~2,800 blocks (~175K tokens) on 16GB budget | After model weights |
| SSD capacity (256GB) | ~34,000 blocks (~2.18M tokens) | 200GB of 256GB SSD |
| Restore vs recompute | Always restore -- 480x faster | 0.84 ms vs 406 ms |
| GDN | RAM only, not spilled | Not block-addressable |
