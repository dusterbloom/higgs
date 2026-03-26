# mistral.rs Paged Attention API Surface

## Architecture Overview

mistral.rs splits paged attention across two crates:
- **`mistralrs-core/src/paged_attention/`** -- High-level block management, scheduling, prefix caching
- **`mistralrs-paged-attn/`** -- Low-level kernels (CUDA + Metal) and Rust bindings

Built on candle-core tensors. Uses `candle-metal-kernels`, `objc2-metal`, `objc2-foundation`, `dispatch2` for Metal path.

---

## 1. Block Hash System (`block_hash.rs`)

Content-addressable hashing for prefix cache reuse. Chain hashing ensures position-awareness.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(u64);  // DefaultHasher (u64)

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHashWithGroupId {
    pub block_hash: BlockHash,
    pub group_id: u32,  // disambiguates full-attn vs sliding-window blocks
}

pub fn hash_block_tokens(
    parent_hash: Option<BlockHash>,  // chain: includes parent hash
    block_tokens: &[u32],
    extra_keys: Option<&[ExtraHashKey]>,  // multimodal, LoRA, cache salt
) -> BlockHash;

pub fn compute_block_hashes(
    tokens: &[u32],
    block_size: usize,
    mm_features: &[MultiModalFeature],
    extra_keys_base: &[ExtraHashKey],
) -> Vec<BlockHash>;  // one per FULL block (partial blocks not hashed)

pub fn compute_new_block_hashes(
    tokens: &[u32],
    block_size: usize,
    existing_hashes: &[BlockHash],
    mm_features: &[MultiModalFeature],
    extra_keys_base: &[ExtraHashKey],
) -> Vec<BlockHash>;  // incremental: only newly-full blocks
```

**Key design:** Hash chain = `hash(parent_hash || tokens || extra_keys)`. Seed for first block is `0`. Two blocks with same tokens but different history produce different hashes.

---

## 2. Block Pool (`block_pool.rs`)

Flat block pool with doubly-linked LRU free list. vLLM v1 design.

```rust
pub struct KVCacheBlock {
    pub block_id: usize,
    pub ref_cnt: u32,
    pub block_hash: Option<BlockHashWithGroupId>,  // retained after free for reuse
    prev_free: usize,  // doubly-linked free list
    next_free: usize,
    pub is_null: bool,
}
```

**Free list:** `FreeKVCacheBlockQueue` -- O(1) popleft (evict LRU), O(1) remove (cache hit), O(1) append (free).

**Eviction order:** Head = least recently used -> evict first. Tail = most recently freed -> evict last.

**Critical property:** Freed blocks RETAIN their hash. Only cleared when reallocated. This enables "lazy" prefix cache -- a freed block can still be reused if a future request matches its hash before it's reallocated.

```rust
pub struct BlockHashToBlockMap {
    cache: HashMap<BlockHashWithGroupId, CachedBlocks>,
}

enum CachedBlocks {
    Single(usize),           // common case: one block per hash
    Multiple(HashMap<usize, usize>),  // collision/duplicate case
}
```

---

## 3. KV Cache Manager (`kv_cache_manager.rs`)

High-level API for block allocation and prefix cache lookups.

```rust
pub struct KVCacheManager {
    block_pool: BlockPool,
    block_size: usize,
    enable_caching: bool,
    kv_cache_group_ids: Vec<u32>,  // most models: [0]
    req_to_blocks: HashMap<usize, RequestBlocks>,
}

pub struct ComputedBlocks {
    pub block_ids: Vec<usize>,
    pub num_computed_tokens: usize,  // always multiple of block_size
}

impl KVCacheManager {
    pub fn new(num_gpu_blocks, block_size, enable_caching, kv_cache_group_ids) -> Self;

    // Find longest prefix cache hit
    pub fn get_computed_blocks(&self, block_hashes: &[BlockHash], num_tokens: usize) -> ComputedBlocks;

    // Allocate blocks for request (handles both new and running requests)
    pub fn allocate_slots(&mut self, request_id, num_tokens, computed_blocks: &[usize]) -> Option<Vec<usize>>;

    // Free blocks when request completes
    pub fn free(&mut self, request_id: usize);

    // Cache newly-full blocks after computation
    pub fn cache_blocks(&mut self, request_id, block_hashes: &[BlockHash]);

    pub fn usage(&self) -> f64;
    pub fn num_free_blocks(&self) -> usize;
}
```

**Key behavior:** `get_computed_blocks` walks the hash chain and stops at first miss (chain property -- can't skip). Returns at most `num_tokens - 1` tokens (must recompute last token for logits).

**Allocation flow:**
1. `get_computed_blocks()` -- find prefix cache hit
2. `allocate_slots(req_id, num_tokens, computed_blocks)` -- alloc new blocks + touch cached blocks
3. Touch = increment ref_cnt + remove from free list (prevents eviction)
4. On completion: `free(req_id)` -- decrement ref_cnt, add to free list tail

---

## 4. Cache Engine (`cache_engine.rs`)

Physical GPU tensor allocation for paged KV cache.

```rust
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub cache_type: PagedCacheType,  // Auto | F8E4M3
}

pub type KVCache = (Tensor, Tensor);  // (key_cache, value_cache) per layer

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,  // one (K, V) pair per layer
}
```

**Key cache layout:** `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
- `x` = dtype size in bytes (packing factor for coalesced access)
- For bf16: x = 2, so head_size/x groups elements for vectorized loads

**Value cache layout:** `[num_blocks, num_kv_heads, head_size, block_size]`
- Different from key cache -- no x-packing

**Metal-specific:** Uses `dev.new_private_buffer()` for GPU-private memory (not shared/managed). This avoids CPU-visible overhead.

---

## 5. Config (`config.rs`)

```rust
pub trait ModelConfigLike {
    fn max_seq_len(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn k_head_dim(&self) -> usize;
    fn v_head_dim(&self) -> usize;
    fn kv_cache_layout(&self) -> KvCacheLayout;  // Standard | Mla
}

pub enum KvCacheLayout {
    Standard,
    Mla { kv_lora_rank: usize, kpe_head_dim: usize },
}

pub const DEFAULT_PAGED_ATTENTION_BLOCK_SIZE: usize = 32;
const SUPPORTED_BLOCK_SIZE: &[usize] = &[8, 16, 32];
```

---

## 6. Paged Attention Layer (`layers/paged_attention.rs`)

The actual attention computation using paged KV caches.

```rust
pub struct PagedAttention {
    alibi_slopes: Option<Tensor>,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
}

impl PagedAttention {
    pub fn forward(
        &self,
        query: &Tensor,          // [batch_size, seq_len, num_heads * head_size]
        key: &Tensor,            // [batch_size, seq_len, num_kv_heads * head_size]
        value: &Tensor,          // [batch_size, num_kv_heads * head_size]
        attention_mask: Option<&Tensor>,
        key_cache: Option<Tensor>,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
        value_cache: Option<Tensor>, // [num_blocks, num_kv_heads, head_size, block_size]
        input_metadata: &PagedAttentionInputMetadata,
        sdpa_params: &SdpaParams,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor>;
}
```

**Two execution paths:**

### Path 1: Prefill with prefix cache hit
When `input_metadata.num_cached_tokens.is_some()`:
1. Write new tokens to cache via `reshape_and_cache()` (slot-mapped write)
2. **Gather** all K/V from paged cache into contiguous tensors via `gather_kv_cache()`
3. Run standard SDPA (flash attention) on the gathered contiguous K/V
4. Uses `cu_seqlens_kv` for packed sequence boundaries

### Path 2: Decode (single token generation)
When no attention mask (decode step):
1. Write K/V to cache via `reshape_and_cache()`
2. Call Metal `paged_attention` kernel directly (v1 or v2)
3. v1: single pass, used when `max_partitions == 1` or many sequences
4. v2: two-pass with intermediate exp_sums/max_logits, for long contexts

**Key insight for Higgs:** The prefill path gathers paged blocks into contiguous tensors, then uses normal SDPA. Only the decode path uses the actual paged attention kernel with block table indirection. This is simpler than expected.

---

## 7. Public API Summary

### From `mistralrs-paged-attn` (kernel crate)
```rust
pub fn paged_attention(q, k_scale, v_scale, key_cache, value_cache,
                       block_tables, context_lens, alibi_slopes,
                       max_context_len, softmax_scale, softcapping, sinks) -> Result<Tensor>;

pub fn reshape_and_cache(key, value, k_scale, v_scale,
                         key_cache, value_cache, slot_mapping) -> Result<()>;

pub fn gather_kv_cache(key_cache, value_cache, k_scale, v_scale,
                       block_tables, cu_seq_lens, out_dtype) -> Result<(Tensor, Tensor)>;
```

### From `mistralrs-core/src/paged_attention/`
```rust
pub use cache_engine::{CacheConfig, CacheEngine, PagedCacheType};
pub use config::{KvCacheLayout, ModelConfigLike, ModelConfigMetadata};
pub use kv_cache_manager::KVCacheManager;
pub use layers::PagedAttention;
pub use scheduler::{PagedAttentionScheduler, PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput};

pub const DEFAULT_PAGED_ATTENTION_BLOCK_SIZE: usize = 32;
pub const _PAD_SLOT_ID: i64 = -1;
```

---

## 8. Implications for Higgs

### What we can reuse conceptually
- Block hash chain design (parent_hash + tokens -> hash)
- Block pool with doubly-linked free list (O(1) alloc/free/evict)
- KV cache manager flow (get_computed -> allocate -> generate -> cache -> free)
- The "gather then SDPA" approach for prefill (avoids custom prefill kernel)

### What we CANNOT directly reuse
- Candle tensor types (we use MLX arrays)
- Metal kernel code (our KV layout is `[B, H, T, D]` not `[num_blocks, H, D/x, block_size, x]`)
- Scheduler (different engine architecture)

### Key architectural decisions to adopt
1. **Block size 32** (default) -- matches their supported sizes `[8, 16, 32]`
2. **Gather-then-SDPA for prefill** -- avoids writing a custom paged prefill kernel
3. **Direct paged attention kernel for decode** -- the real perf win (avoids gathering for single-token decode)
4. **Lazy prefix cache** -- freed blocks retain hash, only cleared on realloc
5. **Slot mapping** for cache writes -- `slot_id = block_id * block_size + offset_in_block`
