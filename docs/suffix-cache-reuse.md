# Suffix Cache Reuse (SCR) — Deep Dive + Higgs Feasibility

## What SCR Is (from the CLM Paper, Appendix B)

**The problem it solves:** In standard prefix caching (like SGLang/vLLM), when you edit text *in the middle* of a context `[A, B, C] → [A, B', C]`, only `A` matches the cached radix tree. Both `B'` and `C` get re-prefilled from scratch — even though `C` is **unchanged text**. This wastes compute on tokens that survived the edit.

**How SCR works for full attention layers:**
1. After an edit replaces `B` with `B'`, SCR diffs the new prompt against the previous session prompt to find **surviving spans** (e.g., `C`)
2. It relocates up to **K=6** surviving spans, largest first
3. For each relocated span like `C`:
   - **Reuses the cached K/V entries** from before the edit
   - **Re-rotates the rotary position encodings** to their new positions (this is key — RoPE depends on absolute position, so reused KV must be rotated)
   - **Splices them in** after `B'`
4. Only `B'` and newly appended tokens are actually prefilled
5. The relocated entries live in the same KV cache buffer; no extra memory allocation needed beyond what was already reserved

**Why this matters:** For long-context conversations where users frequently edit earlier messages while keeping later ones intact, SCR can dramatically reduce redundant computation. Instead of re-prefilling thousands of tokens that haven't changed, you reuse their cached representations with just a RoPE rotation.

---

## How Higgs Currently Handles Sessions

### Current Architecture Overview

Higgs uses a **suffix-based LRU eviction policy** rather than traditional prefix caching:

1. **Session Management**: Each conversation has a unique session ID stored in `SessionManager`. When a message arrives, the engine checks if the session exists and retrieves its state.

2. **KV Cache Structure**: The KV cache is organized as a contiguous array per layer (`kv_cache: Vec<Tensor>`). Tokens are stored sequentially by position index within each layer's cache.

3. **Eviction Policy**: Uses a suffix-based LRU approach — when the cache fills up, the oldest tokens at the *beginning* of the sequence are evicted, not the end. This preserves recent context which is typically more important.

4. **Prefill Strategy**: During prefill, tokens are processed in chunks. The engine tracks `prefill_end_pos` to know how far into the cache the current prefill has progressed. New tokens are appended after existing ones.

5. **Token Position Tracking**: The engine maintains `current_token_position` which increments as tokens are processed. This determines where new tokens go in the KV cache.

### Key Code Paths

**Session Prefill Logic** (`generate_continued_impl_locked`):
```rust
// Check if we have enough history to continue
if self.session_manager.get_session(session_id).is_some() {
    // Session exists — retrieve and process
} else {
    // New session — initialize
}
```

**KV Cache Insertion** (`insert_kv`):
- Takes `(layer_idx, token_positions, kv_tensors)`
- Places tensors at specific positions in the cache
- Respects the `max_tokens_per_layer` limit

**RoPE Rotation**:
- Applied during both prefill and decode phases
- Uses sinusoidal position embeddings based on absolute token positions
- Critical for maintaining positional awareness across cache operations

---

## Feasibility Analysis: Adding SCR to Higgs

### What Would Need to Change

#### 1. Prompt Diffing Engine
Need a mechanism to compare the new prompt against the previous session's prompt and identify surviving spans. This could use:
- Longest Common Subsequence (LCS) algorithm
- Or simpler: line-level diff since most edits happen at message boundaries

#### 2. KV Cache Relocation Support
Current KV cache is a flat array indexed by position. To support SCR:
- Need to mark certain KV entries as "relocated" vs "original"
- Need to handle RoPE re-rotation for relocated entries
- Option A: Keep a separate "scratch" region for relocated entries
- Option B: Allow in-place relocation with proper position tracking

#### 3. RoPE Re-Rotation
This is the trickiest part. When a KV entry moves from position `p_old` to `p_new`:
- Its Q/K vectors need to be re-rotated using the new position's RoPE angles
- Must maintain correctness of attention scores
- Implementation: Apply inverse RoPE at old position, then apply forward RoPE at new position

#### 4. Span Selection Heuristic
Limit to K=6 largest surviving spans (per CLM paper). Need:
- Sort surviving spans by length (descending)
- Select top-K
- Ensure selected spans don't overlap with each other or with the edited region

#### 5. Cache Eviction Interaction
When SCR relocates entries, they may displace newer entries. Need to coordinate with the LRU eviction policy:
- Relocated entries should be treated as "recently used"
- May need to update LRU timestamps for relocated entries

### Implementation Phases

**Phase 1: Prompt Diffing**
- Add a `diff_prompts(old_prompt, new_prompt)` function
- Returns list of surviving spans: `(start_pos, end_pos, original_start_pos)` tuples
- Test with various edit patterns (middle insertion, deletion, replacement)

**Phase 2: Basic KV Relocation**
- Extend `insert_kv` to accept a "relocation" flag
- When relocating, skip the actual tensor copy (already in cache) but update position metadata
- Verify attention scores remain correct

**Phase 3: RoPE Re-Rotation**
- Implement `rotate_kv_at_position(kv_tensor, old_pos, new_pos)`
- Integrate into the relocation path
- Benchmark accuracy against full prefill

**Phase 4: Span Selection & Integration**
- Implement top-K selection heuristic
- Wire into the main generation pipeline
- Add configuration options (enable/disable, max spans K)

**Phase 5: Optimization**
- Profile performance gains on realistic workloads
- Consider batched RoPE rotation for efficiency
- Explore adaptive K based on context length

---

## Potential Challenges

### 1. Correctness Verification
Must ensure that relocated KV entries produce identical attention scores to a full prefill. Any numerical drift would corrupt outputs.

### 2. Memory Layout
Flat KV cache arrays may not efficiently support random access for relocation. Could benefit from a more flexible data structure (e.g., linked segments or sparse representation).

### 3. Concurrency
If multiple sessions are active simultaneously, relocation operations must not interfere with each other's cache regions.

### 4. Edge Cases
- What if a surviving span overlaps with another relocated span?
- What if the entire context is replaced (no surviving spans)?
- What if RoPE angles overflow for very large position shifts?

### 5. Performance Trade-offs
Diffing prompts adds overhead. For simple append-only scenarios, SCR provides no benefit and adds cost. Need smart gating: only activate SCR when edits are detected.

---

## Estimated Impact

For typical chatbot usage patterns:
- **Edit-heavy sessions** (user revises earlier messages): High impact — potentially 50-80% reduction in redundant prefill compute
- **Append-only sessions**: No impact — SCR simply doesn't activate
- **Mixed sessions**: Moderate impact — benefits scale with frequency of mid-context edits

Given that Higgs already handles sessions and KV caching, adding SCR would primarily require:
1. A diffing module (~200 lines of Rust)
2. KV relocation logic integrated into existing insert paths (~150 lines)
3. RoPE re-rotation utility (~100 lines)

Total estimated addition: ~450 lines of new code, mostly in `higgs-engine/src/simple.rs` and possibly a new module `higgs-engine/src/scr.rs`.

---

## References

- CLM Paper (Continuous Language Models), Appendix B: Suffix Cache Reuse
- SGLang / vLLM prefix caching implementations
- Higgs source code: `crates/higgs-engine/src/simple.rs`
- Rotary Position Embedding (RoPE) literature
