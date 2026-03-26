# KV Cache Touchpoint Inventory for PagedAttention Migration

## 1. Trait Hierarchy Diagram

```
KeyValueCache (trait)  [cache.rs:168]
├── impl for &mut T where T: KeyValueCache  [cache.rs:203]  (blanket ref-mut delegation)
├── ConcatKeyValueCache  [cache.rs:242]     (simple grow-by-concat, unbounded)
└── SteppingKeyValueCache  [cache.rs:305]   (pre-alloc 256-slot slab, slice_update writes)
      └── has Option<TurboQuantStorage>     (quantized path, Rust Vec backed)

KvCacheView (enum)  [cache.rs:11]
├── Dense { keys: Array, values: Array }    (contiguous [B, H, T, D] MLX arrays)
└── TurboQuant(TurboQuantKvView)            (quantized codes + norms, not [B,H,T,D])

TurboQuantKvView  [cache.rs:34]
  ↳ decode_scores()  [cache.rs:112]
  ↳ decode_values()  [cache.rs:143]
  ↳ materialize_dense() → [1, H, T, D]  [cache.rs:46]

LayerCache (enum, qwen3_next only)  [qwen3_next.rs:1660]
├── KV(SteppingKeyValueCache)
└── Arrays(ArraysCache)

ArraysCache  [qwen3_next.rs:1025]
├── conv_state: Option<Array>   (GDN convolution state)
└── ssm_state:  Option<Array>   (GDN SSM/recurrence state)

AnyCache (enum, lib.rs:82)
├── KV(Vec<Option<SteppingKeyValueCache>>)     (all models except Qwen3Next)
└── Hybrid(Vec<Option<LayerCache>>)            (Qwen3Next only)
```

---

## 2. Per-File Inventory

### `crates/higgs-models/src/cache.rs`

| Symbol | Line | Role |
|--------|------|------|
| `KvCacheView` enum | 11 | Return type of every cache update; two variants: Dense and TurboQuant |
| `KvCacheView::into_dense()` | 17 | Materializes quantized -> contiguous `[B,H,T,D]` |
| `TurboQuantKvView` struct | 34 | Quantized view; holds raw code/norm Vecs wrapped as Arrays |
| `TurboQuantKvView::materialize_dense()` | 46 | Rebuilds full `[1, H, T, D]` from codes; iterates head x pos |
| `TurboQuantKvView::decode_scores()` | 112 | Asserts shape `[1, H, 1, D]` for queries |
| `TurboQuantKvView::decode_values()` | 143 | Returns `[1, H, 1, D]` output |
| `KeyValueCache` trait | 168 | Core trait; `update_and_view`, `update_and_fetch`, `offset`, `max_size` |
| `KeyValueCache::update_and_view()` | 191 | Primary mutation: append keys/values, return view |
| `KeyValueCache::update_and_fetch()` | 194 | Default impl: calls `update_and_view` then `into_dense()` |
| `ConcatKeyValueCache` struct | 242 | Simple unbounded cache; concatenates on axis -2 every step |
| `ConcatKeyValueCache::update_and_view()` | 263 | `concatenate_axis([existing, new], -2)` -- explicit axis-2 dependence |
| `SteppingKeyValueCache` struct | 305 | Main production cache; pre-allocates in 256-slot slabs |
| `SteppingKeyValueCache::new_turbo()` | 344 | Constructs TurboQuant variant |
| `SteppingKeyValueCache::update_dense()` | 368 | Grows slab, calls `slice_update_axis2`, reads `shape()[2]` |
| `SteppingKeyValueCache::eval_targets()` | 444 | Returns `&Array` refs for batch eval between chunks |
| `TurboQuantStorage::append()` | 495 | Quantizes incoming `[B,H,T,D]` keys/values, writes to Vec storage |
| `validate_turboquant_shapes()` | 677 | Hard assertion: `key_shape.len() == 4`, `shape[0]==1`, `shape[1]==H`, `shape[3]==D` |
| `slice_axis2()` | 754 | Low-level: slices on axis 2 specifically (sequence dimension) |
| `slice_update_axis2()` | 786 | Low-level: writes update into axis 2 slot range |

**All contiguity assumptions in this file:**
- `update_dense` reads `keys.shape()[2]` (line 370) as token count -- assumes axis 2 is sequence
- `slice_axis2` / `slice_update_axis2` are hardcoded to axis 2
- `TurboQuantStorage::append` reads `keys.shape()[2]` (line 504) as new token count
- `materialize_dense` returns shape `[1, H, T, D]` (line 101-108) -- fully contiguous

---

### `crates/higgs-models/src/utils.rs`

| Symbol | Line | Role |
|--------|------|------|
| `cached_scaled_dot_product_attention()` | ~65 | Core SDPA wrapper; calls `kv_cache.update_and_view(keys, values)` at line 75; dispatches Dense vs TurboQuant |
| `create_causal_mask()` | 223 | Creates `[T, T+offset]` causal mask; offset comes from `cache.offset()` |
| `create_attention_mask()` | ~242 | Called by all model `forward()` methods; reads `kv_cache.offset()` at line 254; computes mask shape from sequence length + offset |

**Key contiguity assumption:** `create_causal_mask(N, Some(offset))` bakes the assumption that all cached tokens are contiguous from 0 to offset. PagedAttention with non-contiguous pages cannot use this mask directly.

---

### `crates/higgs-models/src/transformer.rs`

| Symbol | Line | Role |
|--------|------|------|
| `Attention::forward()` | 246 | Reshapes Q/K/V to `[B, L, n_heads, -1]`, calls `kv_cache.offset()` for RoPE, then `update_and_fetch` |
| `Attention (quantized path)::forward()` | 413 | Alternative attention forward; same pattern, reads `cache.update_and_fetch(k_rope, v_i)` at line 795; reads `full_k.shape()[2]` as seq_len at line 798 |
| `TransformerModel::forward()` | 599 | Public entry; calls `create_attention_mask(T, kv_cache)` |
| `TransformerModel::forward_hidden()` | 585 | Same pattern, returns hidden states |
| `TransformerModel::forward_from_embeddings()` | 627 | Used by LLaVA path |
| `TransformerModel::forward_batched()` | 673 | Batch path; reads `caches.offset()` |

---

### `crates/higgs-models/src/qwen3_next.rs`

| Symbol | Line | Role |
|--------|------|------|
| `LayerCache` enum | 1660 | Hybrid: `KV(SteppingKeyValueCache)` or `Arrays(ArraysCache)` |
| `ArraysCache` struct | 1025 | GDN state: `conv_state` + `ssm_state` as raw Arrays |
| `ArraysCache::eval_targets()` | 1050 | Returns refs to conv/ssm arrays for batch eval |
| `Qwen3NextAttention::forward()` | 737 | Dense attention layer: calls `cache.update_and_fetch(keys, values)` at line 784; reads returned keys/values shape |
| `GdnLayer::forward()` | 1176 | GDN/SSM layer: reads `cache.conv_state` and `cache.ssm_state` inline; updates in-place via `take()` / re-assignment |
| `GdnBlock::forward()` | 1620 | Dispatches to `LayerCache::Arrays` or `LayerCache::KV` |
| `Qwen3NextCausalLM::make_cache()` | 1740 | Creates per-layer `LayerCache::Arrays` (GDN) or `LayerCache::KV` (attn) |
| `Qwen3NextCausalLM::forward_hidden()` | 1756 | Prefill: resets cache if needed; reads first KV layer's `offset()` at line 1791 for mask |
| `Qwen3NextCausalLM::forward()` | 1853 | Wraps `forward_hidden` + lm_head |
| `Qwen3NextCausalLM::forward_chunked()` | 1875 | Chunked prefill: calls `eval_targets()` on each `LayerCache::KV` (line 1911) and on `ArraysCache` (lines 1912-1916) between chunks |

**GDN forward details (line 1241-1265, 1296-1321):**
- `conv_state` is `[batch, kernel_width, features]`; prepended with `concatenate_axis` along axis 1, then sliced to keep a rolling window
- `ssm_state` is `[batch, d_state, features]`; updated via matrix recurrence: `new_state = A * old_state + B * input`
- Both are fully sequential: each token step depends on the previous state

---

### `crates/higgs-models/src/qwen3_moe.rs`

| Symbol | Line | Role |
|--------|------|------|
| `Qwen3MoeAttention::forward()` | 159 | Reshapes to `[B,L,n_heads,-1]`; calls `kv_cache.offset()` for RoPE at line 198-199; calls `kv_cache.update_and_fetch(keys, values)` at line 204 |
| `Qwen3MoeModel::forward()` | 415 | Layer loop; passes each `cache[layer]` by `&mut Option<SteppingKeyValueCache>` |
| `Qwen3MoeCausalLM::forward()` | 544 | Public entry; calls `create_attention_mask` |

---

### `crates/higgs-models/src/gemma2.rs`

| Symbol | Line | Role |
|--------|------|------|
| `Gemma2Attention::forward()` | 221 | Reshapes to `[B,L,n_heads,-1]`; calls `kv_cache.offset()` at lines 249-250 for RoPE; calls `kv_cache.update_and_view(keys, values)` at line 252 |
| TurboQuant decode path | ~355-450 | Reshapes for grouped-query: `[B, n_kv_heads, n_rep, L, D]` at lines 361-363; reads returned shape for SDPA |
| `Gemma2CausalLM::forward()` | 802 | Public; calls `create_attention_mask` |

---

### `crates/higgs-models/src/phi3.rs`

| Symbol | Line | Role |
|--------|------|------|
| `Phi3Attention::forward()` | 170 | Reshapes to `[B,L,n_heads,-1]`; `kv_cache.offset()` for RoPE; `update_and_fetch` |
| `Phi3CausalLM::forward()` | 505 | Public entry |

---

### `crates/higgs-models/src/deepseek_v2.rs`

| Symbol | Line | Role |
|--------|------|------|
| `DeepSeekV2Attention::forward()` | 376 | MLA attention; calls `kv_cache.offset()` for RoPE; `update_and_fetch` for compressed KV |
| `DeepSeekV2CausalLM::forward()` | 828 | Public |

---

### `crates/higgs-models/src/starcoder2.rs`

| Symbol | Line | Role |
|--------|------|------|
| `Starcoder2Attention::forward()` | 223 | Reshapes to `[B,L,n_heads,-1]`; RoPE from `offset()`; `update_and_fetch` |
| `create_sliding_causal_mask()` | 118 | Sliding window mask; creates `[T_q, T_kv]` mask using absolute positions |
| `Starcoder2Model::forward()` | 463 | Chooses `create_sliding_causal_mask` or `create_causal_mask` at lines 490/495 |

---

### `crates/higgs-models/src/lib.rs`

| Symbol | Line | Role |
|--------|------|------|
| `AnyCache` enum | 82 | Top-level cache type; `KV(Vec<Option<SteppingKeyValueCache>>)` or `Hybrid(Vec<Option<LayerCache>>)` |
| `make_kv_cache()` | 121 | Creates `AnyCache::KV` with one `SteppingKeyValueCache` per layer |
| `AnyModel::forward()` | 161 | Dispatch table; matches `(model_variant, AnyCache::KV(c))` or `Hybrid(c)` |
| `AnyModel::make_cache_with_config()` | 265 | Routes to `make_kv_cache_with_config` or `m.make_cache()` for Qwen3Next |

---

### `crates/higgs-engine/src/simple.rs`

| Symbol | Line | Role |
|--------|------|------|
| `SimpleEngine::prepare_generation()` | 245 | Cache lifecycle: clones cached prefix or calls `model.make_cache_with_config()` |
| `PreparedGeneration` struct | 87 | Holds `cache: AnyCache` -- one cache per request |

---

### `crates/higgs-engine/src/prompt_cache.rs`

| Symbol | Line | Role |
|--------|------|------|
| `PrefixCache` struct | 41 | Radix-tree LRU prefix cache; stores and clones full `AnyCache` objects |
| `PrefixCache::find_longest_prefix()` | 173 | Returns `PrefixMatch { cache: AnyCache }` -- clones the entire KV state |
| `PrefixCache::store()` | 186 | Inserts `AnyCache` into radix tree |

**Key implication:** The prefix cache clones entire `AnyCache` objects. With PagedAttention, cloning a cache means copying page table mappings, not tensor data.

---

## 3. Contiguity Assumptions -- Every [B, H, T, D] Assumption

| Location | What Assumes Contiguous Layout | Migration Impact |
|----------|-------------------------------|-----------------|
| `cache.rs:370` | `keys.shape()[2]` is token count | Must change for paged blocks |
| `cache.rs:375` | `k.shape()[2]` buffer fullness check on axis 2 | Must change |
| `cache.rs:386-387` | `zeros_dtype(&[b, n_kv_heads, new_slots, head_dim])` allocates contiguous slab | Core of paging change |
| `cache.rs:396-397,402-403` | `concatenate_axis(..., 2)` for slab growth | Replace with page allocation |
| `cache.rs:415-416` | `slice_update_axis2(k, &keys, prev, n)` writes into slot on axis 2 | Replace with page-slot write |
| `cache.rs:422-434` | `slice_axis2(keys, 0, self.offset)` returns prefix of contiguous buffer | Replace with page-gather |
| `transformer.rs:798` | `full_k.shape()[2]` as `seq_len` for SDPA | Must use total paged seq len |
| `utils.rs:254` | `kv_cache.offset()` to build causal mask dimensions | Offset concept must survive paging |
| `utils.rs:257` | `create_causal_mask(T, Some(offset))` assumes all prior tokens contiguous from 0 | Must handle non-contiguous pages |
| `starcoder2.rs:490` | `kv_len = offset + T` for sliding mask | Same |
| `gemma2.rs:361-363` | GQA reshape `[B, n_kv_heads, n_rep, kv_s, D]` | Must reflect actual gathered length |
| All model attention `forward()` | `apply_rope(keys, rope, kv_cache.offset())` -- RoPE position = integer offset | Must become absolute position; paged blocks track absolute token position |

---

## 4. Migration Risk Assessment

### Easy to Change (isolated, low blast radius)
- `KeyValueCache` trait methods -- only 2 concrete impls; new `PagedKeyValueCache` is additive
- `AnyCache::KV(Vec<Option<PagedKVCache>>)` -- enum variant change
- `make_kv_cache()` / `make_kv_cache_with_config()` -- factory functions
- `ConcatKeyValueCache` -- not used in production
- `TurboQuantStorage` -- already non-contiguous; orthogonal to paging

### Moderate Risk (needs coordinated changes)
- `SteppingKeyValueCache::update_dense()` -- core hot path
- `slice_axis2` / `slice_update_axis2` -- internal helpers
- `KvCacheView::Dense { keys, values }` -- consumers expect contiguous tensor
- `eval_targets()` on `SteppingKeyValueCache` -- used in chunked prefill
- `PrefixCache` clone semantics -- must become page-table clone + ref-count bump

### Hard to Change (deep in attention hot path, many callers)
- **`update_and_fetch` return type** -- every attention `forward()` in 7 model files uses shape
- **`create_causal_mask` / mask creation** -- MLX SDPA doesn't support position tables
- **RoPE offset** -- `apply_rope(keys, rope, kv_cache.offset())` in every attention impl
- **`AnyModel::forward_batched()`** -- panics on Hybrid; batch paging requires coordinated page tables
- **`Gemma2` TurboQuant decode path** -- GQA reshape requires knowing total sequence length

---

## 5. GDN/SSM Exclusions -- What Cannot Be Paged

**`conv_state`** (`qwen3_next.rs:1241-1265`):
- Shape: `[batch, kernel_width, features]` -- rolling convolution window (3 steps wide)
- Cannot be "paged" -- no per-token structure, single fused state

**`ssm_state`** (`qwen3_next.rs:1296-1321`):
- Shape: `[batch, d_state, features]` -- SSM recurrence state
- Fundamentally sequential: `state[t]` depends on `state[t-1]`

**Decision:** GDN state must remain as a single per-sequence tensor in GPU memory. Not a candidate for paging, eviction, or prefix sharing.

---

## 6. Minimum Viable PagedAttention Migration Surface

1. **`cache.rs`**: Introduce `PagedKeyValueCache` implementing `KeyValueCache`. Internals: page table + block pool reference.
2. **`cache.rs` / `utils.rs`**: Update `cached_scaled_dot_product_attention()` for paged view (gather before SDPA or paged kernel).
3. **`lib.rs`**: `AnyCache::KV` to accept `PagedKeyValueCache`; update factory.
4. **`utils.rs` `create_causal_mask`**: Accept absolute position indices, not just contiguous offset.
5. **All 7 model `attention::forward()` impls**: RoPE must use absolute token positions per block.
6. **`prompt_cache.rs` `PrefixCache`**: Replace full `AnyCache` clone with page-table clone + shared block pool ref counting.
7. **GDN layers** (`qwen3_next.rs`): No changes needed.
