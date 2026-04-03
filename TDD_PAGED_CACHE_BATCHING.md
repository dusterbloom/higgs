# Higgs Paged Cache + Continuous Batching Implementation Plan

## Goal

Achieve **400+ tok/s prefill** on Qwen3.5-35B-A3B-3bit (M4, 32GB) by implementing:
1. **Paged KV Cache** - Block-based memory management
2. **Continuous Batching** - Round-robin session scheduling

**Inspired by:** [cellm](https://github.com/jeffasante/cellm) (Apache-2.0)

**Timeline:** 4 weeks (measured approach)

---

## Architecture Overview

### Current Higgs (Single Request)

```
┌─────────────────────────────────────────┐
│  SimpleEngine                           │
│  ┌─────────────────────────────────┐   │
│  │  PrefixCache (in-memory)        │   │
│  │  - Stores full KV for prefixes  │   │
│  │  - No eviction                  │   │
│  │  - No batching                  │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Single request → Single generation     │
└─────────────────────────────────────────┘
```

### Target Higgs (Batched + Paged)

```
┌─────────────────────────────────────────┐
│  SimpleEngine                           │
│  ┌─────────────────────────────────┐   │
│  │  PagedKvCache                   │   │
│  │  - BlockAllocator (free list)   │   │
│  │  - PageTable (session→blocks)   │   │
│  │  - CpuKvStorage (f16 buffers)   │   │
│  │  - LRU eviction                 │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  BatchScheduler                 │   │
│  │  - Round-robin session queue    │   │
│  │  - Add/remove sessions          │   │
│  │  - Token budget tracking        │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Multiple requests → Batched decode     │
└─────────────────────────────────────────┘
```

---

## Phase 1: Paged KV Cache (Week 1-2)

### 1.1 BlockAllocator

**Source:** `cellm/crates/cellm-cache/src/allocator.rs`

**Purpose:** Manage block IDs (not bytes). Blocks are fixed-size KV storage units.

**TDD Plan:**

#### Test 1: `test_alloc_free_roundtrip`
```rust
#[test]
fn test_alloc_free_roundtrip() {
    let mut allocator = BlockAllocator::new(1024);
    
    // Allocate a block
    let block_id = allocator.alloc().unwrap();
    assert!(block_id < 1024);
    assert_eq!(allocator.free_count(), 1023);
    assert_eq!(allocator.in_use_count(), 1);
    
    // Free the block
    allocator.free(block_id).unwrap();
    assert_eq!(allocator.free_count(), 1024);
    assert_eq!(allocator.in_use_count(), 0);
}
```

**RED:** Write test, watch fail (BlockAllocator doesn't exist)
**GREEN:** Implement:
```rust
pub struct BlockAllocator {
    total: u32,
    in_use: u32,
    free_list: VecDeque<u32>,
    is_free: Vec<bool>,
}

impl BlockAllocator {
    pub fn new(total_blocks: usize) -> Self { ... }
    pub fn alloc(&mut self) -> Option<u32> { ... }
    pub fn free(&mut self, block_id: u32) -> Result<(), CacheError> { ... }
    pub fn free_count(&self) -> usize { ... }
    pub fn in_use_count(&self) -> usize { ... }
}
```
**REFACTOR:** Extract error type `CacheError::OutOfBlocks`, `CacheError::DoubleFree`

---

#### Test 2: `test_alloc_exhaustion`
```rust
#[test]
fn test_alloc_exhaustion() {
    let mut allocator = BlockAllocator::new(2);
    assert!(allocator.alloc().is_some());
    assert!(allocator.alloc().is_some());
    assert!(allocator.alloc().is_none()); // Exhausted
}
```

**RED:** Write test
**GREEN:** Ensure `alloc()` returns `None` when free_list is empty
**REFACTOR:** ---

---

#### Test 3: `test_alloc_n_atomic`
```rust
#[test]
fn test_alloc_n_atomic() {
    let mut allocator = BlockAllocator::new(5);
    
    // Request more than available - should fail atomically
    let result = allocator.alloc_n(10);
    assert!(result.is_err());
    assert_eq!(allocator.free_count(), 5); // No blocks allocated
    
    // Request within limit - should succeed
    let blocks = allocator.alloc_n(3).unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(allocator.free_count(), 2);
}
```

**RED:** Write test
**GREEN:** Implement atomic `alloc_n()`:
```rust
pub fn alloc_n(&mut self, n: usize) -> Result<Vec<u32>, CacheError> {
    if n > self.free_count() {
        return Err(CacheError::OutOfBlocks { requested: n, free: self.free_count() });
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(self.alloc().expect("checked free_count"));
    }
    Ok(out)
}
```
**REFACTOR:** ---

---

#### Test 4: `test_double_free_errors`
```rust
#[test]
fn test_double_free_errors() {
    let mut allocator = BlockAllocator::new(1);
    let block_id = allocator.alloc().unwrap();
    allocator.free(block_id).unwrap();
    
    let err = allocator.free(block_id).unwrap_err();
    assert!(matches!(err, CacheError::DoubleFree(_)));
}
```

**RED:** Write test
**GREEN:** Track `is_free` state, return error on double-free
**REFACTOR:** ---

---

### 1.2 PageTable

**Purpose:** Map session IDs to allocated blocks.

**TDD Plan:**

#### Test 5: `test_pagetable_session_blocks`
```rust
#[test]
fn test_pagetable_session_blocks() {
    let mut table = PageTable::new();
    
    // Assign blocks to session
    let session_id = 1u64;
    let blocks = vec![10u32, 11, 12];
    table.assign_blocks(session_id, &blocks).unwrap();
    
    // Retrieve blocks
    let retrieved = table.get_blocks(session_id).unwrap();
    assert_eq!(retrieved, blocks);
    
    // Remove session
    table.remove_session(session_id).unwrap();
    assert!(table.get_blocks(session_id).is_none());
}
```

**RED:** Write test
**GREEN:** Implement:
```rust
pub struct PageTable {
    sessions: HashMap<u64, Vec<u32>>,
}

impl PageTable {
    pub fn new() -> Self { ... }
    pub fn assign_blocks(&mut self, session_id: u64, blocks: &[u32]) -> Result<(), CacheError> { ... }
    pub fn get_blocks(&self, session_id: u64) -> Option<&[u32]> { ... }
    pub fn remove_session(&mut self, session_id: u64) -> Result<Vec<u32>, CacheError> { ... }
}
```
**REFACTOR:** ---

---

### 1.3 CpuKvStorage

**Source:** `cellm/crates/cellm-cache/src/kvcache.rs`

**Purpose:** Store actual KV bytes in f16 buffers.

**TDD Plan:**

#### Test 6: `test_kv_storage_write_read`
```rust
#[test]
fn test_kv_storage_write_read() {
    let num_blocks = 1024;
    let block_size = 64; // tokens per block
    let num_kv_heads = 2;
    let head_dim = 128;
    
    let mut storage = CpuKvStorage::new(num_blocks, block_size, num_kv_heads, head_dim);
    
    // Write token at position 0
    let k_data: Vec<f16> = (0..(num_kv_heads * head_dim)).map(|i| f16::from_f32(i as f32)).collect();
    let v_data: Vec<f16> = (0..(num_kv_heads * head_dim)).map(|i| f16::from_f32(i as f32 + 1000.0)).collect();
    
    storage.write_token_f16(0, &k_data, &v_data).unwrap();
    
    // Read back
    let mut k_out = vec![f16::ZERO; num_kv_heads * head_dim];
    let mut v_out = vec![f16::ZERO; num_kv_heads * head_dim];
    storage.read_token_f16(0, &mut k_out, &mut v_out).unwrap();
    
    assert_eq!(k_out, k_data);
    assert_eq!(v_out, v_data);
}
```

**RED:** Write test
**GREEN:** Implement:
```rust
pub struct CpuKvStorage {
    k: Vec<f16>,
    v: Vec<f16>,
    layout: KvCacheLayout,
}

pub struct KvCacheLayout {
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl CpuKvStorage {
    pub fn new(num_blocks: usize, block_size: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let total_elems = num_blocks * block_size * num_kv_heads * head_dim;
        Self {
            k: vec![f16::ZERO; total_elems],
            v: vec![f16::ZERO; total_elems],
            layout: KvCacheLayout { num_blocks, block_size, num_kv_heads, head_dim },
        }
    }
    
    pub fn write_token_f16(&mut self, base: usize, k_src: &[f16], v_src: &[f16]) -> Result<(), CacheError> { ... }
    pub fn read_token_f16(&self, base: usize, k_out: &mut [f16], v_out: &mut [f16]) -> Result<(), CacheError> { ... }
}
```
**REFACTOR:** Add bounds checking, error types

---

#### Test 7: `test_kv_storage_gather`
```rust
#[test]
fn test_kv_storage_gather() {
    // Test gathering tokens from non-contiguous positions
    let mut storage = CpuKvStorage::new(1024, 64, 2, 128);
    
    // Write tokens at positions 0, 10, 25
    for (pos, val) in [(0, 1.0), (10, 2.0), (25, 3.0)] {
        let k_data = vec![f16::from_f32(val); 256]; // 2 heads * 128 dim
        let v_data = vec![f16::from_f32(val + 100.0); 256];
        storage.write_token_f16(pos * 256, &k_data, &v_data).unwrap();
    }
    
    // Gather from positions [0, 25]
    let bases = vec![0, 25 * 256];
    let mut k_out = vec![f16::ZERO; 2 * 256];
    let mut v_out = vec![f16::ZERO; 2 * 256];
    
    storage.gather_tokens_f16(&bases, 256, &mut k_out, &mut v_out).unwrap();
    
    // Verify gathered values
    assert_eq!(k_out[0], f16::from_f32(1.0)); // Position 0
    assert_eq!(k_out[256], f16::from_f32(3.0)); // Position 25
}
```

**RED:** Write test
**GREEN:** Implement `gather_tokens_f16()` for sparse attention
**REFACTOR:** Optimize with rayon parallelism

---

### 1.4 PagedKvCache Integration

**Purpose:** Combine BlockAllocator + PageTable + CpuKvStorage.

**TDD Plan:**

#### Test 8: `test_paged_cache_session_lifecycle`
```rust
#[test]
fn test_paged_cache_session_lifecycle() {
    let mut cache = PagedKvCache::new(1024, 64, 2, 128);
    
    // Create session
    let session_id = 1u64;
    cache.create_session(session_id).unwrap();
    
    // Append tokens
    let k_data = vec![f16::ZERO; 256];
    let v_data = vec![f16::ZERO; 256];
    cache.append_token(session_id, &k_data, &v_data).unwrap();
    cache.append_token(session_id, &k_data, &v_data).unwrap();
    
    // Read tokens
    let view = cache.get_session_view(session_id).unwrap();
    assert_eq!(view.num_tokens(), 2);
    
    // Remove session
    cache.remove_session(session_id).unwrap();
    assert!(cache.get_session_view(session_id).is_none());
}
```

**RED:** Write test
**GREEN:** Implement `PagedKvCache`:
```rust
pub struct PagedKvCache {
    allocator: BlockAllocator,
    page_table: PageTable,
    storage: CpuKvStorage,
    block_size: usize,
}

impl PagedKvCache {
    pub fn new(num_blocks: usize, block_size: usize, num_kv_heads: usize, head_dim: usize) -> Self { ... }
    pub fn create_session(&mut self, session_id: u64) -> Result<(), CacheError> { ... }
    pub fn append_token(&mut self, session_id: u64, k: &[f16], v: &[f16]) -> Result<(), CacheError> { ... }
    pub fn get_session_view(&self, session_id: u64) -> Option<KvCacheView> { ... }
    pub fn remove_session(&mut self, session_id: u64) -> Result<(), CacheError> { ... }
}
```
**REFACTOR:** Add LRU eviction when blocks exhausted

---

## Phase 2: Continuous Batching (Week 3)

### 2.1 RoundRobinScheduler

**Source:** `cellm/crates/cellm-scheduler/src/rr.rs`

**Purpose:** Schedule decode steps across multiple sessions.

**TDD Plan:**

#### Test 9: `test_round_robin_rotates`
```rust
#[test]
fn test_round_robin_rotates() {
    let mut scheduler = RoundRobinScheduler::new();
    
    scheduler.add(1);
    scheduler.add(2);
    scheduler.add(3);
    
    assert_eq!(scheduler.next(), Some(1));
    assert_eq!(scheduler.next(), Some(2));
    assert_eq!(scheduler.next(), Some(3));
    assert_eq!(scheduler.next(), Some(1)); // Wraps around
}
```

**RED:** Write test
**GREEN:** Implement:
```rust
pub type SessionId = u64;

pub struct RoundRobinScheduler {
    q: VecDeque<SessionId>,
}

impl RoundRobinScheduler {
    pub fn new() -> Self { ... }
    pub fn add(&mut self, id: SessionId) { ... }
    pub fn remove(&mut self, id: SessionId) { ... }
    pub fn next(&mut self) -> Option<SessionId> {
        let id = self.q.pop_front()?;
        self.q.push_back(id); // Rotate to back
        Some(id)
    }
}
```
**REFACTOR:** ---

---

#### Test 10: `test_scheduler_remove_works`
```rust
#[test]
fn test_scheduler_remove_works() {
    let mut scheduler = RoundRobinScheduler::new();
    scheduler.add(1);
    scheduler.add(2);
    scheduler.remove(1);
    
    assert_eq!(scheduler.next(), Some(2));
    assert_eq!(scheduler.next(), Some(2)); // Only session 2 remains
}
```

**RED:** Write test
**GREEN:** Implement `remove()` that removes session from queue
**REFACTOR:** ---

---

### 2.2 BatchedEngine Integration

**Purpose:** Integrate scheduler with SimpleEngine.

**TDD Plan:**

#### Test 11: `test_batched_generate`
```rust
#[test]
fn test_batched_generate() {
    let mut engine = BatchedEngine::new(model_path, max_batch_size: 4);
    
    // Submit multiple requests
    let req1 = engine.generate("Hello", max_tokens: 10);
    let req2 = engine.generate("World", max_tokens: 10);
    let req3 = engine.generate("Test", max_tokens: 10);
    
    // Process batch
    let outputs = engine.step().unwrap();
    assert_eq!(outputs.len(), 3);
    
    // Verify all requests made progress
    for output in outputs {
        assert!(!output.text.is_empty());
    }
}
```

**RED:** Write test
**GREEN:** Implement `BatchedEngine` wrapper around SimpleEngine
**REFACTOR:** Add token budget tracking, early stopping

---

## Phase 3: Integration & Benchmarking (Week 4)

### 3.1 SimpleEngine Integration

**Changes to `higgs-engine/src/simple.rs`:**

1. Add `PagedKvCache` field
2. Replace `PrefixCache` with session-based caching
3. Add `BatchScheduler` field
4. Modify `generate()` to support batching

### 3.2 Benchmarking

**Benchmarks to run:**

```bash
# Baseline (current)
./bench_prefill.py --model Qwen3.5-35B-A3B-3bit --ctx 2000
# Expected: ~150 tok/s

# With paged cache
./bench_prefill.py --model Qwen3.5-35B-A3B-3bit --ctx 2000 --paged-cache
# Expected: ~250-300 tok/s (reduced memory pressure)

# With batching (4 concurrent)
./bench_prefill.py --model Qwen3.5-35B-A3B-3bit --ctx 2000 --batch-size 4
# Expected: ~400+ tok/s (amortized overhead)
```

### 3.3 Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Prefill 2k | 153 tok/s | 400+ tok/s | ⏳ Pending |
| Decode | 55 tok/s | 50+ tok/s | ✅ Met |
| Memory usage | High | Reduced (paged) | ⏳ Pending |
| Concurrent sessions | 1 | 4+ | ⏳ Pending |

---

## File Structure

```
crates/higgs-engine/src/
├── cache/
│   ├── mod.rs              # Module exports
│   ├── allocator.rs        # BlockAllocator
│   ├── pagetable.rs        # PageTable
│   ├── storage.rs          # CpuKvStorage
│   └── paged.rs            # PagedKvCache (integration)
├── scheduler/
│   ├── mod.rs              # Module exports
│   └── round_robin.rs      # RoundRobinScheduler
├── simple.rs               # SimpleEngine (updated)
└── error.rs                # CacheError additions
```

---

## Credit & Attribution

**Inline comments:**
```rust
/// Fixed-size block allocator for paged KV cache.
///
/// Implementation inspired by cellm's BlockAllocator:
/// https://github.com/jeffasante/cellm/blob/main/crates/cellm-cache/src/allocator.rs
```

**README section:**
```markdown
## Acknowledgments

The paged KV cache and continuous batching implementations are inspired by
[cellm](https://github.com/jeffasante/cellm), a mobile-native LLM serving
engine in Rust. We adapted their BlockAllocator and RoundRobinScheduler
designs for Higgs's architecture.
```

**Cargo.toml:**
```toml
[package.metadata.inspired-by]
cellm = "https://github.com/jeffasante/cellm"
```

---

## Future: Turboquant Contribution to cellm

**What we can contribute back:**

1. **SpecPrefill integration** - Token importance scoring
2. **Manual RoPE** - Per-token position RoPE
3. **Qwen3.5 optimizations** - Fused GDN projections

**Design for extractability:**
- Keep SpecPrefill in separate module (`higgs-engine/src/spec_prefill/`)
- Minimal dependencies on Higgs internals
- Clear API boundaries for reuse

---

## Timeline

| Week | Milestone | Tests | Deliverable |
|------|-----------|-------|-------------|
| 1 | BlockAllocator + PageTable | 5 tests | `higgs-engine/src/cache/allocator.rs`, `pagetable.rs` |
| 2 | CpuKvStorage + PagedKvCache | 4 tests | `higgs-engine/src/cache/storage.rs`, `paged.rs` |
| 3 | RoundRobinScheduler + BatchedEngine | 3 tests | `higgs-engine/src/scheduler/`, batched engine |
| 4 | Integration + Benchmarks | 1 integration test | Full integration, 400+ tok/s |

**Total:** 12 unit tests + 1 integration test + benchmarks

---

## Next Steps

1. **Create directory structure**
2. **Start with Test 1** (`test_alloc_free_roundtrip`)
3. **Follow TDD cycle** (RED → GREEN → REFACTOR)
4. **Weekly check-ins** on progress

**Ready to begin?**
