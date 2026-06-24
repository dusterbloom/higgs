use std::cell::Cell;
use std::collections::HashMap;
use std::time::Instant;

use std::sync::Arc;

use higgs_models::cache::{KeyValueCache, SteppingKeyValueCache, slice_axis1, slice_axis2};
use higgs_models::qwen3_next::ArraysCache;
use higgs_models::turboquant::TurboQuantContext;
use higgs_models::{AnyCache, LayerCache};
use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::ops::concatenate_axis;

/// Default block size in tokens for paged caching.
pub const DEFAULT_BLOCK_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Block data structures
// ---------------------------------------------------------------------------

/// Per-layer, per-block: K and V array slices with shape `[1, H, block_size, D]`.
///
/// MLX arrays use internal ref-counting, so cloning blocks shares the
/// underlying data without copying.
#[derive(Debug, Clone)]
struct KvBlock {
    keys: Array,
    values: Array,
}

impl KvBlock {
    /// Build a block whose K/V arrays are fully EVALUATED. Radix blocks are
    /// `Arc`-shared and reconstructed from different server threads, so a block
    /// must hold a concrete, immutable MLX buffer — never a pending lazy slice
    /// graph. Evaluating here is the soundness invariant the `unsafe impl Sync`
    /// below relies on; it costs nothing net (the slice would materialize on
    /// first use anyway) and removes the cross-thread data race on shared graphs.
    fn new(keys: Array, values: Array) -> Result<Arc<Self>, Exception> {
        // Construction-time soundness eval (eval-before-share). Always reached
        // via `store()` → `run_prefill` under the MLX gate in production; uses
        // the raw transform directly so the radix unit tests (single-threaded,
        // no gate) can construct blocks without tripping the gate's debug_assert.
        mlx_rs::transforms::eval([&keys, &values])?;
        Ok(Arc::new(Self { keys, values }))
    }
}

/// GDN state snapshot at a block boundary (Hybrid models only).
#[derive(Debug, Clone)]
struct GdnSnapshot {
    conv_state: Option<Array>,
    ssm_state: Option<Array>,
    conv_pos: i32,
    offset: i32,
}

/// Per-layer block for `TurboQuant` KV cache.
///
/// Each block holds the 5 quantized arrays for `block_size` tokens:
/// key/value codes (packed u32), norms, gammas.
#[derive(Debug, Clone)]
struct TqBlock {
    key_codes: Array,
    key_norms: Array,
    key_gammas: Array,
    value_codes: Array,
    value_norms: Array,
}

impl TqBlock {
    /// Build a `TurboQuant` block with all 5 arrays EVALUATED — same soundness
    /// invariant as [`KvBlock::new`]: shared across threads, so never lazy.
    fn new(
        key_codes: Array,
        key_norms: Array,
        key_gammas: Array,
        value_codes: Array,
        value_norms: Array,
    ) -> Result<Arc<Self>, Exception> {
        // Construction-time soundness eval; see KvBlock::new for why this uses
        // the raw transform rather than the gated wrapper.
        mlx_rs::transforms::eval([
            &key_codes,
            &key_norms,
            &key_gammas,
            &value_codes,
            &value_norms,
        ])?;
        Ok(Arc::new(Self {
            key_codes,
            key_norms,
            key_gammas,
            value_codes,
            value_norms,
        }))
    }
}

// MLX `Array` is `Send` but `!Sync` (it holds a `*mut c_void` into the MLX
// runtime). Radix blocks are `Arc`-shared and reconstructed from DIFFERENT
// server threads (each chat request runs in its own `spawn_blocking`), so the
// pointee must be `Send + Sync`. SAFETY rests on one invariant: a block holds
// ONLY fully-evaluated, immutable MLX buffers — never a pending lazy graph.
// That is enforced by construction — `KvBlock::new` / `TqBlock::new` `eval`
// their arrays before the block exists — so concurrent reconstruction touches
// no mutable MLX graph state, only read-only buffers. A *lazy* block would be
// unsound: concurrent build+eval of shared graphs is a data race (SIGSEGV) — the
// regression is pinned by `radix_blocks_reconstruct_safely_across_threads`.
#[allow(unsafe_code)]
unsafe impl Sync for KvBlock {}
#[allow(unsafe_code)]
unsafe impl Sync for TqBlock {}

/// Per-layer cached data covering a single radix edge's token run.
///
/// Blocks are wrapped in `Arc` so that, after an edge split, the shared leading
/// blocks live on a single parent edge and are physically referenced (not
/// copied) by every descendant path. `Arc::strong_count` therefore reflects how
/// many stored prefixes share a given block.
#[allow(dead_code)]
#[derive(Debug, Clone)]
enum CachedLayerData {
    /// Attention layer: sequence of dense K/V blocks.
    Kv(Vec<Arc<KvBlock>>),
    /// Attention layer: sequence of `TurboQuant` blocks.
    TurboQuantKv(Vec<Arc<TqBlock>>),
    /// GDN/SSM layer: state snapshot at block boundary.
    Gdn(GdnSnapshot),
    /// Layer had no cache data.
    Empty,
}

// ---------------------------------------------------------------------------
// Cache entry stored in radix trie
// ---------------------------------------------------------------------------

/// Per-edge block payload for a paged prefix.
///
/// One `EdgeBlocks` describes exactly the tokens spanned by the radix edge it
/// sits on; the full cache for a node is the concatenation of the `EdgeBlocks`
/// of every edge on the root -> node path (see `gather_path`). Because the
/// vectors hold `Arc`-wrapped blocks, splitting an edge moves the shared leading
/// blocks onto the parent edge and both children reference them through the
/// same `Arc`s -- block storage is deduplicated across overlapping prefixes.
struct EdgeBlocks {
    layers: Vec<CachedLayerData>,
    tokens: usize,
    /// `TurboQuant` context when these blocks are quantized; `None` for dense.
    /// Carried on the edge so a block-aligned match that lands *inside* an edge
    /// (not on a stored endpoint) can still reconstruct correctly.
    context: Option<Arc<TurboQuantContext>>,
}

/// Block payload carried by a radix edge.
enum EdgeData {
    /// Dense / `TurboQuant` paged blocks for this edge's tokens.
    Paged(EdgeBlocks),
    /// No paged payload (edges that only carry tokens for a `Cloned` endpoint,
    /// or purely structural internal edges).
    None,
}

/// Endpoint metadata for a stored prefix.
///
/// The actual KV blocks live on the path's edges (`EdgeData::Paged`); this only
/// records how to interpret them and any non-paged fallback.
enum CachedData {
    /// Block-paged cache (dense KV). Blocks are reconstructed from the path.
    Paged { is_hybrid: bool },
    /// Block-paged `TurboQuant` cache with shared quantization context.
    TurboQuantPaged {
        context: Arc<TurboQuantContext>,
        is_hybrid: bool,
    },
    /// Full clone fallback (cache too short for paging).
    Cloned(AnyCache),
}

struct CachedState {
    data: CachedData,
    last_accessed: Cell<Instant>,
}

// ---------------------------------------------------------------------------
// Radix trie
// ---------------------------------------------------------------------------

struct RadixNode {
    edge: Vec<u32>,
    /// Blocks covering exactly `edge`'s tokens. Shared across descendant paths.
    edge_blocks: EdgeData,
    cached: Option<CachedState>,
    children: HashMap<u32, Self>,
}

/// Endpoint kind for a lookup match.
enum MatchEndpoint<'a> {
    /// A stored trie endpoint (full metadata available).
    Stored(&'a CachedData),
    /// A block-aligned position inside an edge (no stored endpoint). Paged
    /// caches are never hybrid; only the optional TQ context is needed.
    PartialPaged {
        context: Option<Arc<TurboQuantContext>>,
    },
}

/// A candidate lookup match: how deep it reaches, how to interpret its blocks,
/// the path of full edges, and an optional partially-matched final edge.
struct MatchResult<'a> {
    prefix_len: usize,
    kind: MatchEndpoint<'a>,
    full_path: Vec<&'a EdgeBlocks>,
    partial_tail: Option<EdgeBlocks>,
    touch: Option<&'a Cell<Instant>>,
}

/// Pick the deeper of two candidate matches.
fn deeper_of<'a>(
    a: Option<MatchResult<'a>>,
    b: Option<MatchResult<'a>>,
) -> Option<MatchResult<'a>> {
    match (a, b) {
        (Some(am), Some(bm)) => Some(if bm.prefix_len > am.prefix_len {
            bm
        } else {
            am
        }),
        (left, right) => left.or(right),
    }
}

/// Result of a paged prefix cache lookup.
pub struct PagedPrefixMatch {
    /// Number of tokens from the beginning that matched the cached prefix.
    pub prefix_len: usize,
    /// Materialized cache state for the matched prefix.
    pub cache: AnyCache,
}

/// Paged prefix cache with block-level storage and LRU eviction.
///
/// Instead of cloning entire `AnyCache` objects (which pins a full KV slab per
/// layer per entry), this cache stores block-sized array slices. MLX arrays use
/// internal ref-counting, so blocks from shared prefixes only store data once.
/// On lookup, blocks are gathered into a contiguous cache via
/// `concatenate_axis` (one-time cost per request).
pub struct PagedPrefixCache {
    root: RadixNode,
    num_cached: usize,
    max_cached: usize,
    block_size: usize,
}

// ---------------------------------------------------------------------------
// RadixNode impl (mirrors prompt_cache.rs but stores CachedState)
// ---------------------------------------------------------------------------

impl RadixNode {
    fn empty() -> Self {
        Self {
            edge: Vec::new(),
            edge_blocks: EdgeData::None,
            cached: None,
            children: HashMap::new(),
        }
    }

    fn leaf(edge: Vec<u32>, edge_blocks: EdgeData, data: CachedData) -> Self {
        Self {
            edge,
            edge_blocks,
            cached: Some(CachedState {
                data,
                last_accessed: Cell::new(Instant::now()),
            }),
            children: HashMap::new(),
        }
    }

    /// Walk the trie matching `tokens`, accumulating the path's edge blocks, and
    /// return the DEEPEST valid match.
    ///
    /// A match is valid at:
    /// - a stored endpoint (`cached`) reached at this `depth`, or
    /// - a block-aligned position *inside* a partially-matched child edge (true
    ///   `RadixAttention` sub-prefix sharing): if the query and an edge share the
    ///   first `k` whole blocks but then diverge, those `k` blocks form a valid
    ///   reusable prefix even though no endpoint was stored there.
    ///
    /// `full_path` references the `EdgeBlocks` of every fully-traversed edge from
    /// the root to the matched node; `partial_tail` (owned, cheap `Arc` clones)
    /// holds the leading whole blocks of a partially-matched final edge. The
    /// caller concatenates `full_path` then `partial_tail` to rebuild a
    /// byte-identical KV cache for the matched prefix.
    fn find_deepest_match<'a>(
        &'a self,
        tokens: &[u32],
        depth: usize,
        min_prefix: usize,
        block_size: usize,
        path: &mut Vec<&'a EdgeBlocks>,
    ) -> Option<MatchResult<'a>> {
        // Record this edge's blocks on the running path (root edge is empty).
        if let EdgeData::Paged(blocks) = &self.edge_blocks {
            path.push(blocks);
        }

        // Block-token depth reachable here from the path's paged edges. May be
        // less than the edge-token `depth` after a non-block-aligned ancestor
        // split; it is exactly the reconstructable prefix length.
        let block_depth: usize = path.iter().map(|e| e.tokens).sum();

        // Candidate 1: a stored endpoint at this node (gives a `touch` handle and
        // handles the Cloned fallback).
        let mut deepest: Option<MatchResult<'a>> = self
            .cached
            .as_ref()
            .filter(|cs| match &cs.data {
                CachedData::Cloned(_) => depth > 0,
                CachedData::Paged { .. } | CachedData::TurboQuantPaged { .. } => {
                    depth >= min_prefix
                }
            })
            .map(|cs| MatchResult {
                prefix_len: match &cs.data {
                    CachedData::Cloned(_) => depth,
                    CachedData::Paged { .. } | CachedData::TurboQuantPaged { .. } => block_depth,
                },
                kind: MatchEndpoint::Stored(&cs.data),
                full_path: path.clone(),
                partial_tail: None,
                touch: Some(&cs.last_accessed),
            });

        // Candidate 2: this node itself sits at a reconstructable block-aligned
        // prefix (e.g. a shared split node with no stored endpoint). The full
        // path's blocks reconstruct it exactly -- true RadixAttention prefix
        // sharing even when no endpoint was stored at this boundary.
        if block_depth >= min_prefix && matches!(&self.edge_blocks, EdgeData::Paged(_)) {
            let node_match = MatchResult {
                prefix_len: block_depth,
                kind: MatchEndpoint::PartialPaged {
                    context: path.last().and_then(|e| e.context.clone()),
                },
                full_path: path.clone(),
                partial_tail: None,
                touch: None,
            };
            deepest = deeper_of(deepest, Some(node_match));
        }

        if let Some(&next_token) = tokens.get(depth) {
            if let Some(child) = self.children.get(&next_token) {
                let remaining = tokens.get(depth..).unwrap_or_default();
                let common = child
                    .edge
                    .iter()
                    .zip(remaining.iter())
                    .take_while(|(a, b)| a == b)
                    .count();

                if common == child.edge.len() {
                    // Whole edge matched: descend.
                    if let Some(found) = child.find_deepest_match(
                        tokens,
                        depth + common,
                        min_prefix,
                        block_size,
                        path,
                    ) {
                        deepest = deeper_of(deepest, Some(found));
                    }
                } else {
                    // Partial edge match: reuse the leading whole blocks that the
                    // query shares with this child edge (RadixAttention sub-prefix).
                    if let Some(found) =
                        child.partial_edge_match(common, min_prefix, block_size, path.as_slice())
                    {
                        deepest = deeper_of(deepest, Some(found));
                    }
                }
            }
        }

        // Pop this edge's blocks so siblings explored by the caller don't inherit them.
        if matches!(&self.edge_blocks, EdgeData::Paged(_)) {
            path.pop();
        }

        deepest
    }

    /// Build a match from the leading whole blocks of THIS edge that the query
    /// shares (`common` tokens matched before divergence). Returns `None` when
    /// fewer than one block is shared or the edge carries no paged blocks.
    fn partial_edge_match<'a>(
        &'a self,
        common: usize,
        min_prefix: usize,
        block_size: usize,
        path: &[&'a EdgeBlocks],
    ) -> Option<MatchResult<'a>> {
        let EdgeData::Paged(blocks) = &self.edge_blocks else {
            return None;
        };
        let n_blocks = common / block_size;
        if n_blocks == 0 {
            return None;
        }
        let matched_tokens = n_blocks * block_size;
        // Derive the reachable prefix length from the ACTUAL block-token sum of
        // the path so far (not the edge-token depth, which can exceed the block
        // sum after a non-block-aligned ancestor split). This keeps `prefix_len`
        // exactly equal to the reconstructed cache's token count.
        let base: usize = path.iter().map(|e| e.tokens).sum();
        let prefix_len = base + matched_tokens;
        if prefix_len < min_prefix {
            return None;
        }
        // Take the leading `n_blocks` of every layer on this edge (Arc clones).
        let tail_layers: Vec<CachedLayerData> = blocks
            .layers
            .iter()
            .map(|l| l.split_at_blocks(n_blocks).0)
            .collect();
        let partial = EdgeBlocks {
            layers: tail_layers,
            tokens: matched_tokens,
            context: blocks.context.clone(),
        };
        Some(MatchResult {
            prefix_len,
            kind: MatchEndpoint::PartialPaged {
                context: blocks.context.clone(),
            },
            // The path up to (but excluding) this edge -- this edge's blocks are
            // not on `path` yet (the parent pushes its own edge; this child edge
            // is represented by `partial_tail`).
            full_path: path.to_vec(),
            partial_tail: Some(partial),
            touch: None,
        })
    }

    fn oldest_cached_time(&self) -> Option<Instant> {
        let mut oldest: Option<Instant> = self.cached.as_ref().map(|cs| cs.last_accessed.get());

        for child in self.children.values() {
            if let Some(child_time) = child.oldest_cached_time() {
                oldest = Some(oldest.map_or(child_time, |o| o.min(child_time)));
            }
        }

        oldest
    }

    fn remove_cached_with_time(&mut self, target: Instant) -> bool {
        if self
            .cached
            .as_ref()
            .is_some_and(|cs| cs.last_accessed.get() == target)
        {
            self.cached = None;
            return true;
        }

        for child in self.children.values_mut() {
            if child.remove_cached_with_time(target) {
                return true;
            }
        }

        false
    }

    fn prune(&mut self) {
        for child in self.children.values_mut() {
            child.prune();
        }
        self.children
            .retain(|_, child| child.cached.is_some() || !child.children.is_empty());

        if self.cached.is_none() && self.children.len() == 1 && !self.edge.is_empty() {
            let Some(key) = self.children.keys().next().copied() else {
                return;
            };
            let Some(mut only_child) = self.children.remove(&key) else {
                return;
            };
            self.edge.append(&mut only_child.edge);
            self.edge_blocks = EdgeData::merge(
                std::mem::replace(&mut self.edge_blocks, EdgeData::None),
                only_child.edge_blocks,
            );
            self.cached = only_child.cached;
            self.children = only_child.children;
        }
    }
}

impl EdgeData {
    /// Concatenate two consecutive edges' block payloads into one. Used when
    /// `prune` collapses a node into its sole child after eviction.
    fn merge(parent: Self, child: Self) -> Self {
        match (parent, child) {
            (Self::Paged(mut p), Self::Paged(c)) => {
                p.tokens += c.tokens;
                p.context = p.context.or(c.context);
                for (pl, cl) in p.layers.iter_mut().zip(c.layers) {
                    pl.append_from(cl);
                }
                Self::Paged(p)
            }
            // A paged edge followed by a non-paged one (or vice versa) only
            // happens around Cloned endpoints, which carry no path blocks; the
            // paged side (if any) is structurally irrelevant once merged.
            (Self::Paged(p), Self::None) => Self::Paged(p),
            (Self::None, other) => other,
        }
    }
}

impl CachedLayerData {
    /// Append another edge segment's blocks for the same layer onto `self`.
    fn append_from(&mut self, other: Self) {
        match (self, other) {
            (Self::Kv(a), Self::Kv(b)) => a.extend(b),
            (Self::TurboQuantKv(a), Self::TurboQuantKv(b)) => a.extend(b),
            // Empty/Gdn layers carry no per-block run; keep the existing one.
            // (Gdn snapshots are endpoint-level, not split across edges, and
            // only ever appear on a single terminal edge.)
            _ => {}
        }
    }

    /// Split this layer's block run at `n` blocks, returning `(head, tail)`.
    /// `Arc` clones are reference bumps -- no array data is copied.
    fn split_at_blocks(&self, n: usize) -> (Self, Self) {
        match self {
            Self::Kv(blocks) => {
                let head = blocks.iter().take(n).map(Arc::clone).collect();
                let tail = blocks.iter().skip(n).map(Arc::clone).collect();
                (Self::Kv(head), Self::Kv(tail))
            }
            Self::TurboQuantKv(blocks) => {
                let head = blocks.iter().take(n).map(Arc::clone).collect();
                let tail = blocks.iter().skip(n).map(Arc::clone).collect();
                (Self::TurboQuantKv(head), Self::TurboQuantKv(tail))
            }
            Self::Gdn(snap) => (Self::Gdn(snap.clone()), Self::Empty),
            Self::Empty => (Self::Empty, Self::Empty),
        }
    }

    /// Drop the leading `n` blocks, returning the remaining tail.
    fn drop_leading(&self, n: usize) -> Self {
        self.split_at_blocks(n).1
    }

    /// Keep only the last `n` blocks of this layer's run.
    fn take_last_blocks(&self, n: usize) -> Self {
        match self {
            Self::Kv(blocks) => {
                let start = blocks.len().saturating_sub(n);
                Self::Kv(blocks.iter().skip(start).map(Arc::clone).collect())
            }
            Self::TurboQuantKv(blocks) => {
                let start = blocks.len().saturating_sub(n);
                Self::TurboQuantKv(blocks.iter().skip(start).map(Arc::clone).collect())
            }
            Self::Gdn(snap) => Self::Gdn(snap.clone()),
            Self::Empty => Self::Empty,
        }
    }
}

/// Shared insert/lookup parameters threaded through the recursion.
struct Ctx {
    block_size: usize,
    context: Option<Arc<TurboQuantContext>>,
}

/// Build an `EdgeData::Paged` from a per-layer block run spanning `tokens`,
/// or `EdgeData::None` when there are no blocks (e.g. the remainder of a
/// `Cloned` insert). `context` is the shared `TurboQuant` context (dense: `None`).
fn edge_blocks_from(
    blocks: Option<Vec<CachedLayerData>>,
    context: Option<Arc<TurboQuantContext>>,
) -> EdgeData {
    blocks.map_or(EdgeData::None, |layers| {
        let tokens = layer_run_tokens(&layers);
        EdgeData::Paged(EdgeBlocks {
            layers,
            tokens,
            context,
        })
    })
}

/// Number of tokens a per-layer block run covers (block count x block size,
/// inferred from a non-`Empty` layer's block dimension).
fn layer_run_tokens(layers: &[CachedLayerData]) -> usize {
    for layer in layers {
        match layer {
            CachedLayerData::Kv(blocks) => {
                if let Some(b) = blocks.first() {
                    let per_block =
                        usize::try_from(b.keys.shape().get(2).copied().unwrap_or(0)).unwrap_or(0);
                    return blocks.len() * per_block;
                }
            }
            CachedLayerData::TurboQuantKv(blocks) => {
                if let Some(b) = blocks.first() {
                    let per_block =
                        usize::try_from(b.key_norms.shape().get(1).copied().unwrap_or(0))
                            .unwrap_or(0);
                    return blocks.len() * per_block;
                }
            }
            CachedLayerData::Gdn(_) | CachedLayerData::Empty => {}
        }
    }
    0
}

/// Drop the leading `n_tokens` worth of blocks (`n_tokens / block_size` blocks)
/// from every layer of an incoming block run. Used to discard the incoming
/// duplicates of blocks that already live on the trie's shared edges.
fn drop_leading_blocks(
    blocks: Option<Vec<CachedLayerData>>,
    n_tokens: usize,
    block_size: usize,
) -> Option<Vec<CachedLayerData>> {
    let n_blocks = n_tokens / block_size;
    blocks.map(|layers| layers.iter().map(|l| l.drop_leading(n_blocks)).collect())
}

/// Split an existing edge's blocks at `n_tokens` (`n_tokens / block_size`
/// blocks), returning `(shared_head, leftover_tail)` as `EdgeData`s.
fn split_edge_blocks(edge: EdgeData, n_tokens: usize, block_size: usize) -> (EdgeData, EdgeData) {
    match edge {
        EdgeData::None => (EdgeData::None, EdgeData::None),
        EdgeData::Paged(blocks) => {
            let n_blocks = n_tokens / block_size;
            let head_tokens = n_blocks * block_size;
            let tail_tokens = blocks.tokens.saturating_sub(head_tokens);
            let mut head_layers = Vec::with_capacity(blocks.layers.len());
            let mut tail_layers = Vec::with_capacity(blocks.layers.len());
            for layer in &blocks.layers {
                let (h, t) = layer.split_at_blocks(n_blocks);
                head_layers.push(h);
                tail_layers.push(t);
            }
            (
                EdgeData::Paged(EdgeBlocks {
                    layers: head_layers,
                    tokens: head_tokens,
                    context: blocks.context.clone(),
                }),
                EdgeData::Paged(EdgeBlocks {
                    layers: tail_layers,
                    tokens: tail_tokens,
                    context: blocks.context,
                }),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// PagedPrefixCache impl
// ---------------------------------------------------------------------------

impl PagedPrefixCache {
    pub fn new(max_entries: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "PagedPrefixCache block_size must be > 0");
        Self {
            root: RadixNode::empty(),
            num_cached: 0,
            max_cached: max_entries,
            block_size,
        }
    }

    /// Find the longest cached prefix that matches the beginning of `tokens`.
    ///
    /// Returns `None` if no prefix matches or if the match is shorter than one
    /// block. On hit, blocks along the matched path are gathered into a
    /// contiguous `AnyCache`.
    pub fn find_longest_prefix(&mut self, tokens: &[u32]) -> Option<PagedPrefixMatch> {
        let mut scratch: Vec<&EdgeBlocks> = Vec::new();
        let m = self.root.find_deepest_match(
            tokens,
            0,
            self.block_size,
            self.block_size,
            &mut scratch,
        )?;
        let prefix_len = m.prefix_len;
        match materialize(&m) {
            Ok(cache) => {
                tracing::debug!(prefix_len, "Prefix cache hit");
                if let Some(touch) = m.touch {
                    touch.set(Instant::now());
                }
                Some(PagedPrefixMatch { prefix_len, cache })
            }
            Err(e) => {
                tracing::warn!(error = %e, "Prefix cache materialize failed");
                None
            }
        }
    }

    /// Store a prefix and its cache state as paged blocks.
    ///
    /// For dense KV caches, the K/V arrays are sliced into block-sized views
    /// (lazy, nearly free) and inserted into the radix trie one block per edge
    /// segment. Where the new sequence shares a leading run of blocks with an
    /// existing entry, the shared blocks already live on the trie's edges and
    /// are reused -- the incoming duplicates are dropped, so storage is
    /// deduplicated. For `TurboQuant` caches with deferred quantization a full
    /// clone fallback is used. Only block-aligned tokens are stored in the trie.
    pub fn store(&mut self, prefix_tokens: &[u32], cache: &AnyCache) {
        if self.max_cached == 0 {
            return;
        }

        // TurboQuant caches with deferred quantization are stored as dense
        // blocks until TQ activates. Full TQ block paging is implemented in the
        // CachedData::TurboQuantPaged variant but requires the TQ arrays to be
        // populated (post-activation). For now, use clone fallback when TQ config
        // is set but arrays aren't yet quantized to avoid cache corruption.
        let prepared = match slice_into_blocks(cache, self.block_size, prefix_tokens.len()) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to page cache, using clone fallback");
                match kv_offset(cache).and_then(|offset| usize::try_from(offset).ok()) {
                    Some(offset) if offset > prefix_tokens.len() => {
                        tracing::warn!(
                            cache_tokens = offset,
                            key_tokens = prefix_tokens.len(),
                            "Skipping prefix cache store: cache longer than key"
                        );
                        return;
                    }
                    _ => PreparedStore {
                        blocks: None,
                        context: None,
                        total_tokens: prefix_tokens.len(),
                        endpoint: CachedData::Cloned(cache.clone()),
                    },
                }
            }
        };

        let stored_len = prepared.total_tokens;
        let Some(tokens_to_store) = prefix_tokens.get(..stored_len) else {
            tracing::warn!(
                key_tokens = prefix_tokens.len(),
                stored_len,
                "Skipping prefix cache store: cache longer than key"
            );
            return;
        };

        let ctx = Ctx {
            block_size: self.block_size,
            context: prepared.context,
        };
        let added = Self::insert(
            &mut self.root,
            tokens_to_store,
            0,
            prepared.blocks.clone(),
            prepared.blocks.as_ref(),
            prepared.endpoint,
            &ctx,
        );

        if added {
            self.num_cached += 1;
            while self.num_cached > self.max_cached {
                self.evict_lru();
            }
        }
    }

    /// Insert `tokens` (with optional per-layer block run `blocks` spanning all
    /// of `tokens`) marking the terminal node with `endpoint`.
    ///
    /// `blocks` covers exactly the tokens still being placed (`[pos, len)`).
    /// As the trie is descended, the block run is sliced so that each edge
    /// segment carries precisely the blocks for the tokens on that edge. When
    /// the descent reaches an already-stored edge whose tokens match, the
    /// existing edge blocks are reused and the incoming ones for that span are
    /// discarded -- this is where shared prefixes deduplicate.
    fn insert(
        node: &mut RadixNode,
        tokens: &[u32],
        pos: usize,
        blocks: Option<Vec<CachedLayerData>>,
        full_blocks: Option<&Vec<CachedLayerData>>,
        endpoint: CachedData,
        ctx: &Ctx,
    ) -> bool {
        let block_size = ctx.block_size;
        if pos >= tokens.len() {
            let is_new = node.cached.is_none();
            // Overwrite: refresh this terminal edge's blocks from the full
            // incoming run so a re-store with changed KV replaces the stale
            // blocks. (For identical tokens the model produces identical KV, so
            // this is a no-op in production; it keeps re-stores correct.) The
            // terminal edge owns the LAST `edge.len()/block_size` blocks of the
            // sequence; slice those from the full run.
            if !is_new {
                if let Some(full) = full_blocks {
                    let edge_blocks = node.edge.len() / block_size;
                    let refreshed: Vec<CachedLayerData> = full
                        .iter()
                        .map(|l| l.take_last_blocks(edge_blocks))
                        .collect();
                    node.edge_blocks = edge_blocks_from(Some(refreshed), ctx.context.clone());
                }
            }
            node.cached = Some(CachedState {
                data: endpoint,
                last_accessed: Cell::new(Instant::now()),
            });
            return is_new;
        }

        let Some(&next_token) = tokens.get(pos) else {
            return false;
        };

        if node.children.contains_key(&next_token) {
            let Some(child) = node.children.get(&next_token) else {
                return false;
            };

            let remaining = tokens.get(pos..).unwrap_or_default();
            let common = child
                .edge
                .iter()
                .zip(remaining.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common == child.edge.len() {
                // Whole child edge matched: its blocks are reused as-is. Drop
                // the incoming blocks for this span (they are byte-identical KV)
                // and recurse with the remainder.
                let remainder = drop_leading_blocks(blocks, common, block_size);
                let Some(child_mut) = node.children.get_mut(&next_token) else {
                    return false;
                };
                return Self::insert(
                    child_mut,
                    tokens,
                    pos + common,
                    remainder,
                    full_blocks,
                    endpoint,
                    ctx,
                );
            }

            // Partial match -- split the child edge at `common`.
            let Some(mut old_child) = node.children.remove(&next_token) else {
                return false;
            };

            let common_edge = old_child.edge.get(..common).unwrap_or_default().to_vec();
            let leftover_edge = old_child.edge.get(common..).unwrap_or_default().to_vec();

            let Some(&leftover_key) = leftover_edge.first() else {
                return false;
            };
            old_child.edge = leftover_edge;

            // Split the existing edge's blocks at the same token boundary so the
            // shared leading blocks live on the new `split` parent (referenced by
            // both children's paths) and the rest stay on `old_child`.
            let (shared_blocks, leftover_blocks) =
                split_edge_blocks(old_child.edge_blocks, common, block_size);
            old_child.edge_blocks = leftover_blocks;

            let mut split = RadixNode {
                edge: common_edge,
                edge_blocks: shared_blocks,
                cached: None,
                children: HashMap::new(),
            };
            split.children.insert(leftover_key, old_child);

            // The incoming blocks for `[pos, pos+common)` are duplicates of the
            // shared blocks now on `split`; drop them and keep the remainder.
            let remainder = drop_leading_blocks(blocks, common, block_size);

            if pos + common >= tokens.len() {
                split.cached = Some(CachedState {
                    data: endpoint,
                    last_accessed: Cell::new(Instant::now()),
                });
                node.children.insert(next_token, split);
                return true;
            }

            let new_edge = tokens.get(pos + common..).unwrap_or_default().to_vec();
            let Some(&new_key) = new_edge.first() else {
                node.children.insert(next_token, split);
                return false;
            };
            let new_leaf = RadixNode::leaf(
                new_edge,
                edge_blocks_from(remainder, ctx.context.clone()),
                endpoint,
            );
            split.children.insert(new_key, new_leaf);

            node.children.insert(next_token, split);
            return true;
        }

        // No matching child -- create a new leaf carrying all remaining blocks.
        let new_edge = tokens.get(pos..).unwrap_or_default().to_vec();
        let new_leaf = RadixNode::leaf(
            new_edge,
            edge_blocks_from(blocks, ctx.context.clone()),
            endpoint,
        );
        node.children.insert(next_token, new_leaf);
        true
    }

    fn evict_lru(&mut self) {
        if let Some(oldest) = self.root.oldest_cached_time() {
            if self.root.remove_cached_with_time(oldest) {
                self.num_cached -= 1;
                self.root.prune();
            }
        }
    }

    pub const fn len(&self) -> usize {
        self.num_cached
    }

    pub const fn is_empty(&self) -> bool {
        self.num_cached == 0
    }

    pub fn clear(&mut self) {
        self.root = RadixNode::empty();
        self.num_cached = 0;
    }

    /// Test-only: collect, per layer-0 dense block, its `Arc` pointer identity
    /// and `strong_count` across every edge in the trie. Distinct pointers ==
    /// distinct stored blocks (no duplication); a `strong_count > 1` means the
    /// block is physically shared by multiple prefixes' paths.
    #[cfg(test)]
    #[allow(clippy::as_conversions)]
    fn layer0_block_stats(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        self.root.collect_layer0_blocks(&mut out);
        out
    }
}

#[cfg(test)]
#[allow(clippy::as_conversions)]
impl RadixNode {
    /// Walk the trie, appending `(arc_ptr_as_usize, strong_count)` for the
    /// layer-0 dense KV blocks on every edge.
    fn collect_layer0_blocks(&self, out: &mut Vec<(usize, usize)>) {
        if let EdgeData::Paged(blocks) = &self.edge_blocks {
            if let Some(CachedLayerData::Kv(layer0)) = blocks.layers.first() {
                for b in layer0 {
                    out.push((Arc::as_ptr(b) as usize, Arc::strong_count(b)));
                }
            }
        }
        for child in self.children.values() {
            child.collect_layer0_blocks(out);
        }
    }
}

// ---------------------------------------------------------------------------
// Slice & materialize helpers
// ---------------------------------------------------------------------------

/// Check if any layer in the cache uses `TurboQuant`.
#[allow(dead_code)]
fn is_turboquant(cache: &AnyCache) -> bool {
    match cache {
        AnyCache::KV(layers) => layers.iter().any(|l| {
            l.as_ref()
                .is_some_and(|c| c.kv_cache_config().is_turboquant())
        }),
        AnyCache::Hybrid(layers) => layers
            .iter()
            .any(|l| matches!(l, Some(LayerCache::KV(c)) if c.kv_cache_config().is_turboquant())),
    }
}

/// Get the KV offset from the first non-empty KV layer.
fn kv_offset(cache: &AnyCache) -> Option<i32> {
    match cache {
        AnyCache::KV(layers) => layers
            .iter()
            .find_map(|l| l.as_ref())
            .map(KeyValueCache::offset),
        AnyCache::Hybrid(layers) => layers.iter().find_map(|l| match l {
            Some(LayerCache::KV(c)) => Some(KeyValueCache::offset(c)),
            _ => None,
        }),
    }
}

/// Outcome of preparing a cache for storage: the per-layer block run (if paged),
/// how many tokens it covers, and the endpoint metadata for the trie node.
struct PreparedStore {
    /// `None` for `Cloned` endpoints (no path blocks).
    blocks: Option<Vec<CachedLayerData>>,
    /// Shared `TurboQuant` context for paged-TQ blocks; `None` for dense/clone.
    context: Option<Arc<TurboQuantContext>>,
    total_tokens: usize,
    endpoint: CachedData,
}

/// Slice a cache into block-aligned paged data.
fn slice_into_blocks(
    cache: &AnyCache,
    block_size: usize,
    max_tokens: usize,
) -> Result<PreparedStore, Exception> {
    // Hybrid caches (GDN+KV) can't be block-paged because GDN sequential state
    // doesn't align to block boundaries. The KV offset would mismatch the GDN
    // offset after materialization, producing corrupt attention. Use clone instead.
    let AnyCache::KV(kv_layers) = cache else {
        return Ok(PreparedStore {
            blocks: None,
            context: None,
            total_tokens: max_tokens,
            endpoint: CachedData::Cloned(cache.clone()),
        });
    };

    let offset = kv_offset(cache).unwrap_or(0);
    let offset_usize = usize::try_from(offset).unwrap_or(0);
    let num_blocks = offset_usize.min(max_tokens) / block_size;
    if num_blocks == 0 {
        return Err(Exception::custom("Cache too short for paging"));
    }
    let total_tokens = num_blocks * block_size;
    let block_size_i32 =
        i32::try_from(block_size).map_err(|_| Exception::custom("block_size overflow"))?;

    // Slice KV layers as TQ blocks when actually quantized, dense otherwise.
    let mut tq_context: Option<Arc<TurboQuantContext>> = None;
    let layers: Vec<CachedLayerData> = kv_layers
        .iter()
        .map(|layer_opt| {
            let Some(kv) = layer_opt.as_ref() else {
                return Ok(CachedLayerData::Empty);
            };
            if kv.is_quantized() {
                if tq_context.is_none() {
                    tq_context = kv.turbo_arrays().map(|(c, ..)| Arc::clone(c));
                }
                slice_tq_layer(kv, num_blocks, block_size_i32)
            } else {
                slice_kv_layer(Some(kv), num_blocks, block_size_i32)
            }
        })
        .collect::<Result<_, _>>()?;

    let endpoint = tq_context
        .as_ref()
        .map_or(CachedData::Paged { is_hybrid: false }, |context| {
            CachedData::TurboQuantPaged {
                context: Arc::clone(context),
                is_hybrid: false,
            }
        });

    Ok(PreparedStore {
        blocks: Some(layers),
        context: tq_context,
        total_tokens,
        endpoint,
    })
}

/// Slice a single `TurboQuant` KV layer into blocks along axis 1.
fn slice_tq_layer(
    kv: &SteppingKeyValueCache,
    num_blocks: usize,
    block_size: i32,
) -> Result<CachedLayerData, Exception> {
    let Some((_ctx, key_codes, key_norms, key_gammas, value_codes, value_norms)) =
        kv.turbo_arrays()
    else {
        return Ok(CachedLayerData::Empty);
    };

    let mut blocks = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let start = i32::try_from(i)
            .map_err(|_| Exception::custom("block index overflow"))?
            .checked_mul(block_size)
            .ok_or_else(|| Exception::custom("block start overflow"))?;
        let end = start
            .checked_add(block_size)
            .ok_or_else(|| Exception::custom("block end overflow"))?;
        blocks.push(TqBlock::new(
            slice_axis1(key_codes, start, end)?,
            slice_axis1(key_norms, start, end)?,
            slice_axis1(key_gammas, start, end)?,
            slice_axis1(value_codes, start, end)?,
            slice_axis1(value_norms, start, end)?,
        )?);
    }

    Ok(CachedLayerData::TurboQuantKv(blocks))
}

/// Slice a single KV layer into blocks.
fn slice_kv_layer(
    kv_opt: Option<&SteppingKeyValueCache>,
    num_blocks: usize,
    block_size: i32,
) -> Result<CachedLayerData, Exception> {
    let Some(kv) = kv_opt else {
        return Ok(CachedLayerData::Empty);
    };
    let (Some(keys), Some(values)) = (kv.keys(), kv.values()) else {
        return Ok(CachedLayerData::Empty);
    };

    let mut blocks = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let start = i32::try_from(i)
            .map_err(|_| Exception::custom("block index overflow"))?
            .checked_mul(block_size)
            .ok_or_else(|| Exception::custom("block start overflow"))?;
        let end = start
            .checked_add(block_size)
            .ok_or_else(|| Exception::custom("block end overflow"))?;
        let k = slice_axis2(keys, start, end)?;
        let v = slice_axis2(values, start, end)?;
        blocks.push(KvBlock::new(k, v)?);
    }

    Ok(CachedLayerData::Kv(blocks))
}

/// Flatten the per-edge block runs along a root -> node path (plus an optional
/// partially-matched final edge) into a single per-layer block run, in order.
///
/// Each `EdgeBlocks` has the same layer layout, so layer `l`'s full run is the
/// in-order concatenation of every edge's `layers[l]`.
fn flatten_path_layers(
    full_path: &[&EdgeBlocks],
    partial_tail: Option<&EdgeBlocks>,
) -> Vec<CachedLayerData> {
    let mut out: Vec<CachedLayerData> = Vec::new();
    let segments = full_path.iter().copied().chain(partial_tail);
    for edge in segments {
        if out.is_empty() {
            out.clone_from(&edge.layers);
        } else {
            for (acc, seg) in out.iter_mut().zip(edge.layers.iter()) {
                acc.append_from(seg.clone());
            }
        }
    }
    out
}

/// Materialize a matched prefix into a contiguous `AnyCache`.
///
/// Blocks are gathered from every edge along the match's `full_path` (root ->
/// matched node) plus any `partial_tail` (leading whole blocks of a partially
/// matched final edge). Shared leading edges contribute their blocks exactly
/// once, so the reconstruction is byte-identical to the originally stored KV
/// for the matched span.
fn materialize(m: &MatchResult) -> Result<AnyCache, Exception> {
    match &m.kind {
        MatchEndpoint::Stored(CachedData::Cloned(cache)) => Ok(cache.clone()),
        MatchEndpoint::Stored(CachedData::Paged { is_hybrid, .. }) => {
            let layers = flatten_path_layers(&m.full_path, m.partial_tail.as_ref());
            if *is_hybrid {
                materialize_hybrid(&layers)
            } else {
                materialize_kv(&layers)
            }
        }
        MatchEndpoint::Stored(CachedData::TurboQuantPaged {
            context, is_hybrid, ..
        }) => {
            let layers = flatten_path_layers(&m.full_path, m.partial_tail.as_ref());
            if *is_hybrid {
                materialize_tq_hybrid(&layers, context)
            } else {
                materialize_tq_kv(&layers, context)
            }
        }
        MatchEndpoint::PartialPaged { context } => {
            let layers = flatten_path_layers(&m.full_path, m.partial_tail.as_ref());
            context.as_ref().map_or_else(
                || materialize_kv(&layers),
                |ctx| materialize_tq_kv(&layers, ctx),
            )
        }
    }
}

fn materialize_kv(layers: &[CachedLayerData]) -> Result<AnyCache, Exception> {
    let kv_layers: Result<Vec<_>, _> = layers
        .iter()
        .map(|layer| match layer {
            CachedLayerData::Kv(blocks) => gather_blocks(blocks).map(Some),
            CachedLayerData::TurboQuantKv(_) => {
                Err(Exception::custom("TQ layer in non-TQ materialize"))
            }
            CachedLayerData::Empty => Ok(Some(SteppingKeyValueCache::new())),
            CachedLayerData::Gdn(_) => Err(Exception::custom("Unexpected GDN layer in KV cache")),
        })
        .collect();
    Ok(AnyCache::KV(kv_layers?))
}

fn materialize_tq_kv(
    layers: &[CachedLayerData],
    context: &Arc<TurboQuantContext>,
) -> Result<AnyCache, Exception> {
    let kv_layers: Result<Vec<_>, _> = layers
        .iter()
        .map(|layer| match layer {
            CachedLayerData::TurboQuantKv(blocks) => gather_tq_blocks(blocks, context).map(Some),
            CachedLayerData::Kv(blocks) => gather_blocks(blocks).map(Some),
            CachedLayerData::Empty => Ok(Some(SteppingKeyValueCache::new())),
            CachedLayerData::Gdn(_) => {
                Err(Exception::custom("Unexpected GDN layer in TQ KV cache"))
            }
        })
        .collect();
    Ok(AnyCache::KV(kv_layers?))
}

fn materialize_hybrid(layers: &[CachedLayerData]) -> Result<AnyCache, Exception> {
    let hybrid_layers: Result<Vec<_>, _> = layers
        .iter()
        .map(|layer| match layer {
            CachedLayerData::Kv(blocks) => gather_blocks(blocks).map(|kv| Some(LayerCache::KV(kv))),
            CachedLayerData::TurboQuantKv(_) => {
                Err(Exception::custom("TQ layer in non-TQ hybrid materialize"))
            }
            CachedLayerData::Gdn(snap) => Ok(Some(LayerCache::Arrays(ArraysCache {
                conv_state: snap.conv_state.clone(),
                ssm_state: snap.ssm_state.clone(),
                conv_pos: snap.conv_pos,
                offset: snap.offset,
            }))),
            CachedLayerData::Empty => Ok(None),
        })
        .collect();
    Ok(AnyCache::Hybrid(hybrid_layers?))
}

fn materialize_tq_hybrid(
    layers: &[CachedLayerData],
    context: &Arc<TurboQuantContext>,
) -> Result<AnyCache, Exception> {
    let hybrid_layers: Result<Vec<_>, _> = layers
        .iter()
        .map(|layer| match layer {
            CachedLayerData::TurboQuantKv(blocks) => {
                gather_tq_blocks(blocks, context).map(|kv| Some(LayerCache::KV(kv)))
            }
            CachedLayerData::Kv(blocks) => gather_blocks(blocks).map(|kv| Some(LayerCache::KV(kv))),
            CachedLayerData::Gdn(snap) => Ok(Some(LayerCache::Arrays(ArraysCache {
                conv_state: snap.conv_state.clone(),
                ssm_state: snap.ssm_state.clone(),
                conv_pos: snap.conv_pos,
                offset: snap.offset,
            }))),
            CachedLayerData::Empty => Ok(None),
        })
        .collect();
    Ok(AnyCache::Hybrid(hybrid_layers?))
}

/// Gather KV blocks into a single contiguous `SteppingKeyValueCache`.
fn gather_blocks(blocks: &[Arc<KvBlock>]) -> Result<SteppingKeyValueCache, Exception> {
    let Some(first) = blocks.first() else {
        return Ok(SteppingKeyValueCache::new());
    };

    if blocks.len() == 1 {
        return SteppingKeyValueCache::from_arrays(first.keys.clone(), first.values.clone());
    }

    let key_arrays: Vec<Array> = blocks.iter().map(|b| b.keys.clone()).collect();
    let value_arrays: Vec<Array> = blocks.iter().map(|b| b.values.clone()).collect();
    let keys = concatenate_axis(&key_arrays, 2)?;
    let values = concatenate_axis(&value_arrays, 2)?;

    SteppingKeyValueCache::from_arrays(keys, values)
}

/// Gather TQ blocks into a single `SteppingKeyValueCache` with TQ storage.
fn gather_tq_blocks(
    blocks: &[Arc<TqBlock>],
    context: &Arc<TurboQuantContext>,
) -> Result<SteppingKeyValueCache, Exception> {
    if blocks.is_empty() {
        return Ok(SteppingKeyValueCache::new());
    }

    // Concatenate all block arrays along axis 1 (the sequence dimension).
    let concat1 = |arrays: Vec<Array>| -> Result<Array, Exception> {
        match arrays.len() {
            0 => Err(Exception::custom("empty TQ block array")),
            1 => arrays
                .into_iter()
                .next()
                .ok_or_else(|| Exception::custom("empty TQ block array")),
            _ => concatenate_axis(&arrays, 1),
        }
    };

    let key_codes = concat1(blocks.iter().map(|b| b.key_codes.clone()).collect())?;
    let key_norms = concat1(blocks.iter().map(|b| b.key_norms.clone()).collect())?;
    let key_gammas = concat1(blocks.iter().map(|b| b.key_gammas.clone()).collect())?;
    let value_codes = concat1(blocks.iter().map(|b| b.value_codes.clone()).collect())?;
    let value_norms = concat1(blocks.iter().map(|b| b.value_norms.clone()).collect())?;

    // Total tokens = sum of block sizes along axis 1.
    let total = key_norms.shape().get(1).copied().unwrap_or(0);

    SteppingKeyValueCache::from_turbo_arrays(
        Arc::clone(context),
        key_codes,
        key_norms,
        key_gammas,
        value_codes,
        value_norms,
        total,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    clippy::as_conversions,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::identity_op,
    clippy::suboptimal_flops,
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use higgs_models::cache::KeyValueCache;

    /// Create a KV cache with `num_layers` layers, each containing `seq_len`
    /// tokens of shape `[1, 2, seq_len, 8]`.
    fn make_kv_cache(num_layers: usize, seq_len: i32) -> AnyCache {
        let layers: Vec<Option<SteppingKeyValueCache>> = (0..num_layers)
            .map(|_| {
                let keys = Array::zeros::<f32>(&[1, 2, seq_len, 8]).unwrap();
                let values = Array::zeros::<f32>(&[1, 2, seq_len, 8]).unwrap();
                Some(SteppingKeyValueCache::from_arrays(keys, values).unwrap())
            })
            .collect();
        AnyCache::KV(layers)
    }

    /// Create a Hybrid cache with interleaved KV and GDN layers.
    fn make_hybrid_cache(num_layers: usize, seq_len: i32) -> AnyCache {
        let layers: Vec<Option<LayerCache>> = (0..num_layers)
            .map(|i| {
                if i % 4 == 0 {
                    Some(LayerCache::Arrays(ArraysCache {
                        conv_state: Some(Array::zeros::<f32>(&[1, 4, 4]).unwrap()),
                        ssm_state: Some(Array::zeros::<f32>(&[1, 16]).unwrap()),
                        conv_pos: 3,
                        offset: seq_len,
                    }))
                } else {
                    let keys = Array::zeros::<f32>(&[1, 2, seq_len, 8]).unwrap();
                    let values = Array::zeros::<f32>(&[1, 2, seq_len, 8]).unwrap();
                    Some(LayerCache::KV(
                        SteppingKeyValueCache::from_arrays(keys, values).unwrap(),
                    ))
                }
            })
            .collect();
        AnyCache::Hybrid(layers)
    }

    fn kv_layer_count(cache: &AnyCache) -> usize {
        match cache {
            AnyCache::KV(v) => v.len(),
            AnyCache::Hybrid(v) => v.len(),
        }
    }

    fn kv_cache_offset(cache: &AnyCache) -> i32 {
        match cache {
            AnyCache::KV(layers) => layers
                .iter()
                .find_map(|l| l.as_ref())
                .map_or(0, KeyValueCache::offset),
            AnyCache::Hybrid(layers) => layers
                .iter()
                .find_map(|l| match l {
                    Some(LayerCache::KV(c)) => Some(KeyValueCache::offset(c)),
                    _ => None,
                })
                .unwrap_or(0),
        }
    }

    #[test]
    fn test_empty_cache_returns_none() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        assert!(cache.find_longest_prefix(&[1, 2, 3]).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_store_and_find_exact_match() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..64).collect();
        let kv = make_kv_cache(4, 64);

        cache.store(&prefix, &kv);
        assert_eq!(cache.len(), 1);

        let mut query: Vec<u32> = prefix;
        query.extend_from_slice(&[100, 101, 102]);

        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());
        let matched = result.unwrap();
        assert_eq!(matched.prefix_len, 64);
        assert_eq!(kv_layer_count(&matched.cache), 4);
    }

    #[test]
    fn test_block_aligned_prefix_len() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        // Store 50 tokens of data with 50 token prefix
        let prefix: Vec<u32> = (0..50).collect();
        let kv = make_kv_cache(4, 50);
        cache.store(&prefix, &kv);
        assert_eq!(cache.len(), 1);

        // Query with all 50 tokens + extra
        let mut query: Vec<u32> = (0..50).collect();
        query.push(999);
        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());

        let matched = result.unwrap();
        // Should be block-aligned: floor(50/32)*32 = 32
        assert_eq!(matched.prefix_len, 32);
        assert_eq!(kv_cache_offset(&matched.cache), 32);
    }

    #[test]
    fn test_materialize_correct_shapes() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..96).collect();
        let kv = make_kv_cache(4, 96);
        cache.store(&prefix, &kv);

        let mut query: Vec<u32> = prefix;
        query.push(999);
        let matched = cache.find_longest_prefix(&query).unwrap();

        // 96 tokens / 32 block_size = 3 blocks, materialized to 96 tokens
        assert_eq!(matched.prefix_len, 96);

        match &matched.cache {
            AnyCache::KV(layers) => {
                assert_eq!(layers.len(), 4);
                for layer in layers {
                    let kv = layer.as_ref().unwrap();
                    assert_eq!(KeyValueCache::offset(kv), 96);
                    assert_eq!(kv.keys().unwrap().shape(), &[1, 2, 96, 8]);
                    assert_eq!(kv.values().unwrap().shape(), &[1, 2, 96, 8]);
                }
            }
            AnyCache::Hybrid(_) => panic!("Expected KV cache"),
        }
    }

    #[test]
    fn test_hybrid_cache_roundtrip() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..64).collect();
        let hybrid = make_hybrid_cache(8, 64);
        cache.store(&prefix, &hybrid);
        assert_eq!(cache.len(), 1);

        let mut query: Vec<u32> = prefix;
        query.push(999);
        let matched = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(matched.prefix_len, 64);

        match &matched.cache {
            AnyCache::Hybrid(layers) => {
                assert_eq!(layers.len(), 8);
                for (i, layer) in layers.iter().enumerate() {
                    match layer.as_ref().unwrap() {
                        LayerCache::KV(kv) => {
                            assert_ne!(i % 4, 0, "Layer {i} should be KV");
                            assert_eq!(KeyValueCache::offset(kv), 64);
                        }
                        LayerCache::Arrays(ac) => {
                            assert_eq!(i % 4, 0, "Layer {i} should be GDN");
                            assert_eq!(ac.offset, 64);
                            assert_eq!(ac.conv_pos, 3);
                            assert!(ac.conv_state.is_some());
                            assert!(ac.ssm_state.is_some());
                        }
                    }
                }
            }
            AnyCache::KV(_) => panic!("Expected Hybrid cache"),
        }
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = PagedPrefixCache::new(2, DEFAULT_BLOCK_SIZE);

        let prefix_a: Vec<u32> = (0..64).collect();
        let prefix_b: Vec<u32> = (100..164).collect();
        let prefix_c: Vec<u32> = (200..264).collect();

        cache.store(&prefix_a, &make_kv_cache(4, 64));
        cache.store(&prefix_b, &make_kv_cache(4, 64));
        assert_eq!(cache.len(), 2);

        cache.store(&prefix_c, &make_kv_cache(4, 64));
        assert_eq!(cache.len(), 2);

        let mut query_c: Vec<u32> = prefix_c;
        query_c.push(999);
        assert!(cache.find_longest_prefix(&query_c).is_some());
    }

    #[test]
    fn test_zero_capacity_never_stores() {
        let mut cache = PagedPrefixCache::new(0, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..64).collect();
        cache.store(&prefix, &make_kv_cache(4, 64));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_longest_prefix_wins() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        let short_prefix: Vec<u32> = (0..32).collect();
        cache.store(&short_prefix, &make_kv_cache(4, 32));

        let long_prefix: Vec<u32> = (0..96).collect();
        cache.store(&long_prefix, &make_kv_cache(4, 96));

        let query: Vec<u32> = (0..128).collect();
        let result = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(result.prefix_len, 96);
    }

    #[test]
    fn test_prefix_shorter_than_block_ignored() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..16).collect();
        let kv = make_kv_cache(4, 16);
        cache.store(&prefix, &kv);
        // Stored via clone fallback since too short for block paging.
        // Clone fallback still makes the prefix findable.
        let mut query: Vec<u32> = prefix;
        query.push(999);
        let matched = cache
            .find_longest_prefix(&query)
            .expect("clone fallback should be findable");
        assert_eq!(matched.prefix_len, 16);
    }

    #[test]
    fn test_clear() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..64).collect();
        cache.store(&prefix, &make_kv_cache(4, 64));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_overwrite_same_prefix() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..64).collect();

        cache.store(&prefix, &make_kv_cache(2, 64));
        assert_eq!(cache.len(), 1);

        cache.store(&prefix, &make_kv_cache(8, 64));
        assert_eq!(cache.len(), 1);

        let mut query = prefix;
        query.push(999);
        let result = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(kv_layer_count(&result.cache), 8);
    }

    #[test]
    fn test_shared_prefix_partial_match() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        let system_prefix: Vec<u32> = (0..64).collect();
        cache.store(&system_prefix, &make_kv_cache(2, 64));

        let full_prompt: Vec<u32> = (0..128).collect();
        cache.store(&full_prompt, &make_kv_cache(4, 128));
        assert_eq!(cache.len(), 2);

        // Query with same system prefix but different user message
        let mut different_suffix: Vec<u32> = (0..64).collect();
        different_suffix.extend(500..564);
        let result = cache.find_longest_prefix(&different_suffix).unwrap();
        assert_eq!(result.prefix_len, 64);
        assert_eq!(kv_layer_count(&result.cache), 2);
    }

    #[test]
    fn test_from_arrays_enables_decode() {
        // Verify that from_arrays produces a cache that can accept new tokens.
        let keys = Array::ones::<f32>(&[1, 2, 32, 8]).unwrap();
        let values = Array::ones::<f32>(&[1, 2, 32, 8]).unwrap();
        let mut kv = SteppingKeyValueCache::from_arrays(keys, values).unwrap();
        assert_eq!(KeyValueCache::offset(&kv), 32);

        // Simulate a decode step
        let new_k = Array::zeros::<f32>(&[1, 2, 1, 8]).unwrap();
        let new_v = Array::zeros::<f32>(&[1, 2, 1, 8]).unwrap();
        let (rk, rv) = kv.update_and_fetch(new_k, new_v).unwrap();
        assert_eq!(rk.shape(), &[1, 2, 33, 8]);
        assert_eq!(rv.shape(), &[1, 2, 33, 8]);
        assert_eq!(KeyValueCache::offset(&kv), 33);
    }

    // -- Radix-tree block sharing tests --------------------------------------

    /// KV cache whose K/V values are a deterministic function of absolute token
    /// position, so block content is position-distinct and reconstruction can be
    /// verified byte-for-byte. Element at (token `t`, head `h`, dim `d`) =
    /// `base + t*1000 + h*100 + d`.
    fn make_kv_cache_content(num_layers: usize, seq_len: i32, base: f32) -> AnyCache {
        let s = seq_len as usize;
        let layers: Vec<Option<SteppingKeyValueCache>> = (0..num_layers)
            .map(|layer| {
                let mut data = vec![0.0_f32; 1 * 2 * s * 8];
                for h in 0..2 {
                    for t in 0..s {
                        for d in 0..8 {
                            let idx = ((h * s) + t) * 8 + d;
                            data[idx] = base
                                + (layer as f32) * 1_000_000.0
                                + (t as f32) * 1000.0
                                + (h as f32) * 100.0
                                + d as f32;
                        }
                    }
                }
                let keys = Array::from_slice(&data, &[1, 2, seq_len, 8]);
                let values = Array::from_slice(&data, &[1, 2, seq_len, 8]);
                Some(SteppingKeyValueCache::from_arrays(keys, values).unwrap())
            })
            .collect();
        AnyCache::KV(layers)
    }

    /// Layer-0 keys array of a KV cache.
    fn cache_keys(cache: &AnyCache, layer: usize) -> Array {
        match cache {
            AnyCache::KV(layers) => layers[layer].as_ref().unwrap().keys().unwrap().clone(),
            AnyCache::Hybrid(_) => panic!("expected KV"),
        }
    }

    /// Assert the first `n` tokens (axis 2) of two `[1, H, *, 8]` key arrays are
    /// byte-identical. Uses MLX `array_eq` + `all` so strided slice views are
    /// compared by VALUE (raw `as_slice` would read the contiguous backing
    /// buffer and ignore strides).
    fn assert_keys_eq_first_n(got: &Array, expected: &Array, n: i32) {
        let g = slice_axis2(got, 0, n).unwrap();
        let e = slice_axis2(expected, 0, n).unwrap();
        let eq = g.array_eq(&e, None).unwrap();
        let all = eq.all(None).unwrap();
        assert!(
            all.item::<bool>(),
            "reconstructed keys differ from stored KV over first {n} tokens"
        );
    }

    /// (a) Two inserts sharing a leading prefix must PHYSICALLY share the
    /// overlapping blocks: one stored `Arc` per shared block (not 2x), with a
    /// strong_count reflecting the shared reference.
    #[test]
    fn test_shared_prefix_dedups_blocks() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        // Both sequences share the first 64 tokens (2 blocks), diverge after.
        let mut seq_a: Vec<u32> = (0..64).collect();
        seq_a.extend(1000..1064);
        let mut seq_b: Vec<u32> = (0..64).collect();
        seq_b.extend(2000..2064);

        cache.store(&seq_a, &make_kv_cache(1, 128));
        cache.store(&seq_b, &make_kv_cache(1, 128));
        assert_eq!(cache.len(), 2);

        // Total distinct layer-0 blocks: 2 shared + 2 (a-only) + 2 (b-only) = 6.
        // Without sharing (storing each prefix's blocks independently) it would
        // be 2*4 = 8. Distinct-Arc count == 6 IS the dedup proof: the shared
        // leading blocks are stored once on the common parent edge, not copied
        // into each prefix's storage.
        let stats = cache.layer0_block_stats();
        assert_eq!(stats.len(), 6, "expected 6 distinct blocks, got {stats:?}");

        // Both prefixes still reconstruct correctly.
        let mut q_a = seq_a.clone();
        q_a.push(9);
        let mut q_b = seq_b.clone();
        q_b.push(9);
        assert_eq!(cache.find_longest_prefix(&q_a).unwrap().prefix_len, 128);
        assert_eq!(cache.find_longest_prefix(&q_b).unwrap().prefix_len, 128);
    }

    /// Stronger sharing proof: store a short prefix, then a longer one extending
    /// it. The short prefix's blocks are REUSED by the long prefix's path (same
    /// Arc), so storing the extension adds no duplicate of the shared blocks.
    #[test]
    fn test_extension_reuses_shared_block_arcs() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        let short: Vec<u32> = (0..64).collect(); // 2 blocks
        cache.store(&short, &make_kv_cache(1, 64));
        let before = cache.layer0_block_stats();
        assert_eq!(before.len(), 2);

        // Extend: shares the first 64 tokens, adds 64 more (2 new blocks).
        let long: Vec<u32> = (0..128).collect(); // 4 blocks
        cache.store(&long, &make_kv_cache(1, 128));

        let after = cache.layer0_block_stats();
        // 4 distinct blocks total (2 shared reused + 2 new) -- not 2 + 4 = 6.
        assert_eq!(
            after.len(),
            4,
            "extension must reuse shared blocks: {after:?}"
        );

        // The two original block Arcs are still present by pointer identity.
        let before_ptrs: std::collections::HashSet<usize> =
            before.iter().map(|(p, _)| *p).collect();
        let after_ptrs: std::collections::HashSet<usize> = after.iter().map(|(p, _)| *p).collect();
        assert!(
            before_ptrs.is_subset(&after_ptrs),
            "original shared block Arcs must survive the extension"
        );
    }

    /// (b) `find_longest_prefix` returns the DEEPEST shared match for a query
    /// that overlaps the stored prefix only partially -- including divergence
    /// in the MIDDLE of a block, where only whole shared blocks may be reused.
    #[test]
    fn test_deepest_match_mid_block_divergence() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        // Store 96 tokens (3 blocks) of content-distinct KV.
        let stored: Vec<u32> = (0..96).collect();
        let kv = make_kv_cache_content(2, 96, 7.0);
        let expected_keys = cache_keys(&kv, 0);
        cache.store(&stored, &kv);

        // Query shares 40 tokens then diverges mid-block-2 (block boundary 32).
        // No endpoint is stored at depth 32 -- this exercises the RadixAttention
        // intra-edge block-boundary match.
        let mut query: Vec<u32> = (0..40).collect();
        query.extend(5000..5060);
        let result = cache.find_longest_prefix(&query).unwrap();

        // Deepest block-aligned match below 40 tokens is 32 (1 block).
        assert_eq!(result.prefix_len, 32);
        assert_eq!(kv_cache_offset(&result.cache), 32);

        // Reconstruction must be byte-identical to the first 32 tokens of the
        // originally stored KV.
        assert_keys_eq_first_n(&cache_keys(&result.cache, 0), &expected_keys, 32);
    }

    /// Byte-identical reconstruction across a SHARED block boundary: a query
    /// reusing a fully-shared 2-block prefix rebuilds the exact stored KV.
    #[test]
    fn test_shared_block_reconstruction_byte_identical() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        let mut seq_a: Vec<u32> = (0..64).collect();
        seq_a.extend(1000..1032);
        let kv_a = make_kv_cache_content(1, 96, 11.0);
        let expected = cache_keys(&kv_a, 0);
        cache.store(&seq_a, &kv_a);

        // seq_b shares the first 64 tokens, then diverges. After the split the
        // first two blocks live on the shared parent edge.
        let mut seq_b: Vec<u32> = (0..64).collect();
        seq_b.extend(3000..3032);
        cache.store(&seq_b, &make_kv_cache_content(1, 96, 22.0));

        // Query reusing the shared 64-token (2-block) prefix. The match lands on
        // the shared parent edge (no stored endpoint there) via an intra-edge
        // block-boundary match.
        let mut query: Vec<u32> = (0..64).collect();
        query.push(9);
        let result = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(result.prefix_len, 64);

        // The shared blocks came from seq_a (stored first); reconstruction of
        // the first 64 tokens must byte-match seq_a's stored KV.
        assert_keys_eq_first_n(&cache_keys(&result.cache, 0), &expected, 64);
    }

    /// Concurrency contract for the radix block cache. Blocks are `Arc`-shared
    /// and, in the server, reconstructed from DIFFERENT tokio blocking-pool
    /// threads across turns (each request runs in its own `spawn_blocking`). Two
    /// invariants must hold:
    /// 1. A block is genuinely `Send + Sync` — safe to hand to and read from
    ///    another thread — which requires it to hold only fully-EVALUATED MLX
    ///    buffers (enforced by `KvBlock::new`), never a pending lazy slice graph.
    /// 2. MLX eval must be serialized: MLX's Metal command buffer is process-
    ///    global and aborts (SIGABRT in `concatenate_gpu` → command encoder) on
    ///    concurrent eval. The engine serializes via the model `Mutex`; this test
    ///    mirrors that with a shared lock around reconstruction.
    ///
    /// It reconstructs from many threads (under the shared lock) and checks each
    /// thread's result is well-formed. Without evaluated blocks (invariant 1) the
    /// shared lazy graphs would race even under the lock; without the lock
    /// (invariant 2) the Metal command buffer aborts.
    #[test]
    fn radix_blocks_reconstruct_serialized_across_threads() {
        let cache = make_kv_cache_content(1, 128, 7.0);
        let AnyCache::KV(layers) = &cache else {
            panic!("expected KV cache");
        };
        let kv = layers[0].as_ref().expect("layer 0 present");
        let CachedLayerData::Kv(blocks) = slice_kv_layer(Some(kv), 4, 32).unwrap() else {
            panic!("expected Kv blocks");
        };
        let shared = Arc::new(blocks);
        let mlx_lock = Arc::new(std::sync::Mutex::new(()));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let s = Arc::clone(&shared);
                let lock = Arc::clone(&mlx_lock);
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        let guard = lock.lock().unwrap();
                        let c = gather_blocks(&s).unwrap();
                        let k = c.keys().expect("keys").clone();
                        let v = c.values().expect("values").clone();
                        mlx_rs::transforms::eval([&k, &v]).unwrap();
                        drop(guard);
                        assert_eq!(k.shape(), [1, 2, 128, 8].as_slice());
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("reconstruction thread panicked");
        }
    }

    /// (c) Inserting then evicting frees only UNSHARED blocks while shared
    /// blocks stay alive as long as another prefix references them.
    #[test]
    fn test_eviction_frees_only_unshared_blocks() {
        let mut cache = PagedPrefixCache::new(2, DEFAULT_BLOCK_SIZE);

        // A and B share the first 64 tokens (2 blocks), diverge after.
        let mut seq_a: Vec<u32> = (0..64).collect();
        seq_a.extend(1000..1064);
        let mut seq_b: Vec<u32> = (0..64).collect();
        seq_b.extend(2000..2064);

        cache.store(&seq_a, &make_kv_cache(1, 128));
        cache.store(&seq_b, &make_kv_cache(1, 128));
        assert_eq!(cache.len(), 2);
        // 6 distinct blocks: 2 shared + 2 (a) + 2 (b).
        assert_eq!(cache.layer0_block_stats().len(), 6);

        // Touch A so it is the most-recently-used; B becomes the LRU victim
        // when C arrives.
        let mut q_a = seq_a.clone();
        q_a.push(9);
        assert!(cache.find_longest_prefix(&q_a).is_some());

        // Insert C (disjoint) -> evicts the LRU (B). B's UNSHARED blocks (the 2
        // after the shared prefix) are freed; the 2 SHARED blocks remain because
        // A still references them.
        let mut seq_c: Vec<u32> = (5000..5064).collect();
        seq_c.extend(6000..6064);
        cache.store(&seq_c, &make_kv_cache(1, 128));
        assert_eq!(cache.len(), 2);

        // A must still reconstruct fully (its shared blocks were NOT freed).
        let result_a = cache.find_longest_prefix(&q_a).unwrap();
        assert_eq!(result_a.prefix_len, 128);

        // Distinct blocks now: A keeps 4 (2 shared + 2 a-only), C has 4.
        // B's 2 unshared blocks are gone; the 2 formerly-shared blocks survive
        // (now referenced only by A). Total distinct = 8.
        let stats = cache.layer0_block_stats();
        assert_eq!(
            stats.len(),
            8,
            "B's unshared blocks should be freed, shared+A+C kept: {stats:?}"
        );

        // B is gone: a B-only query no longer reaches depth 128.
        let mut q_b = seq_b.clone();
        q_b.push(9);
        match cache.find_longest_prefix(&q_b) {
            None => {}
            Some(m) => assert!(
                m.prefix_len <= 64,
                "evicted B must not yield its full 128-token prefix, got {}",
                m.prefix_len
            ),
        }
    }
}
