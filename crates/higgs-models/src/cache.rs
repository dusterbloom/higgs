use mlx_rs::{Array, Stream, error::Exception, ops, ops::concatenate_axis};

/// Trait for key-value caches used in autoregressive generation.
pub trait KeyValueCache {
    /// Whether the cache stores quantized KV pairs.
    fn is_quantized(&self) -> bool {
        false
    }

    /// Group size for quantized cache. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Bit width for quantized cache. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

    /// Current sequence offset (number of tokens already cached).
    ///
    /// Returns the offset as a concrete `i32`. During non-compiled execution
    /// this extracts from the underlying Array instantly. Callers inside a
    /// compiled trace should use `offset_array()` instead.
    fn offset(&self) -> i32;

    /// Current sequence offset as an `Array` (for compiled trace paths).
    ///
    /// Default implementation wraps `offset()` in a scalar Array.
    fn offset_array(&self) -> Array {
        Array::from_int(self.offset())
    }

    /// Maximum cache size, if bounded.
    fn max_size(&self) -> Option<i32>;

    /// Append new key/value tensors and return the full cached key/value.
    fn update_and_fetch(&mut self, keys: Array, values: Array)
    -> Result<(Array, Array), Exception>;
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn offset_array(&self) -> Array {
        T::offset_array(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }
}

/// Simple KV cache that concatenates new keys/values with existing ones.
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl ConcatKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        if let (Some(existing_keys), Some(existing_values)) = (self.keys.take(), self.values.take())
        {
            self.keys = Some(concatenate_axis(&[existing_keys, keys], -2)?);
            self.values = Some(concatenate_axis(&[existing_values, values], -2)?);
        } else {
            self.keys = Some(keys);
            self.values = Some(values);
        }

        let key_shape = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("Keys cannot be None after update"))?
            .shape();
        let seq_dim_index = key_shape.len().wrapping_sub(2);
        self.offset = *key_shape
            .get(seq_dim_index)
            .ok_or_else(|| Exception::custom("Key shape has fewer than 2 dimensions"))?;

        let result_keys = self
            .keys
            .clone()
            .ok_or_else(|| Exception::custom("Keys cannot be None after update"))?;
        let result_values = self
            .values
            .clone()
            .ok_or_else(|| Exception::custom("Values cannot be None after update"))?;

        Ok((result_keys, result_values))
    }
}

/// Pre-allocated KV cache that grows in chunks, avoiding per-token allocation.
///
/// Matches Python `mlx_lm`'s `KVCache`: pre-allocates 256 slots at a time and
/// uses `mlx_slice_update` for writes instead of concatenation every token.
/// Keys/values have shape `[B, n_heads, seq_len, head_dim]` with sequence on axis 2.
///
/// `offset` is stored as an `Array` (scalar i32) so that `FlatCache` can
/// round-trip it without calling `.item()`, which panics during `mx.compile` trace.
#[derive(Debug, Clone)]
pub struct SteppingKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: Array,
    step: i32,
}

impl Default for SteppingKeyValueCache {
    fn default() -> Self {
        Self {
            keys: None,
            values: None,
            offset: Array::from_int(0),
            step: 256,
        }
    }
}

impl SteppingKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reference to cached keys (for FlatCache packing).
    pub fn keys_ref(&self) -> Option<&Array> {
        self.keys.as_ref()
    }

    /// Reference to cached values (for FlatCache packing).
    pub fn values_ref(&self) -> Option<&Array> {
        self.values.as_ref()
    }

    /// Set keys directly (for FlatCache unpacking).
    pub fn set_keys(&mut self, keys: Array) {
        self.keys = Some(keys);
    }

    /// Set values directly (for FlatCache unpacking).
    pub fn set_values(&mut self, values: Array) {
        self.values = Some(values);
    }

    /// Set offset directly from a concrete i32 (wraps in a scalar Array).
    pub fn set_offset(&mut self, offset: i32) {
        self.offset = Array::from_int(offset);
    }

    /// Set offset directly from an Array (for FlatCache unpacking in compiled path).
    pub fn set_offset_array(&mut self, offset: Array) {
        self.offset = offset;
    }

    /// Get the offset as an Array reference (for FlatCache packing).
    pub fn offset_array_ref(&self) -> &Array {
        &self.offset
    }
}

/// Slice an array along axis 2: `arr[..., start:end, ...]`
#[allow(unsafe_code, clippy::indexing_slicing)]
fn slice_axis2(arr: &Array, start: i32, end: i32) -> Result<Array, Exception> {
    let ndim = arr.ndim();
    debug_assert!(ndim >= 3, "slice_axis2 requires ndim >= 3, got {ndim}");
    let mut starts = vec![0i32; ndim];
    let mut ends: Vec<i32> = arr.shape().to_vec();
    let strides = vec![1i32; ndim];
    starts[2] = start;
    ends[2] = end;

    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_slice(
            &raw mut result,
            arr.as_ptr(),
            starts.as_ptr(),
            starts.len(),
            ends.as_ptr(),
            ends.len(),
            strides.as_ptr(),
            strides.len(),
            Stream::task_local_or_default().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(Exception::custom("mlx_slice failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Write `update` into `target` at `[..., start:start+n, ...]` on axis 2.
#[allow(unsafe_code, clippy::indexing_slicing)]
fn slice_update_axis2(
    target: &Array,
    update: &Array,
    start: i32,
    n: i32,
) -> Result<Array, Exception> {
    let ndim = target.ndim();
    debug_assert!(
        ndim >= 3,
        "slice_update_axis2 requires ndim >= 3, got {ndim}"
    );
    let mut starts = vec![0i32; ndim];
    let mut ends: Vec<i32> = target.shape().to_vec();
    let strides = vec![1i32; ndim];
    starts[2] = start;
    ends[2] = start + n;

    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_slice_update(
            &raw mut result,
            target.as_ptr(),
            update.as_ptr(),
            starts.as_ptr(),
            starts.len(),
            ends.as_ptr(),
            ends.len(),
            strides.as_ptr(),
            strides.len(),
            Stream::task_local_or_default().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(Exception::custom("mlx_slice_update failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

impl SteppingKeyValueCache {
    /// Evaluate cached keys/values/offset to prevent graph accumulation.
    pub fn eval_state(&self) {
        let mut targets: Vec<&Array> = vec![&self.offset];
        if let Some(ref k) = self.keys { targets.push(k); }
        if let Some(ref v) = self.values { targets.push(v); }
        let _ = mlx_rs::transforms::eval(targets);
    }
}

impl KeyValueCache for SteppingKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset.item::<i32>()
    }

    fn offset_array(&self) -> Array {
        self.offset.clone()
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    #[allow(clippy::indexing_slicing)]
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let new_tokens = keys.shape()[2];

        // Growth check needs concrete offset. During compiled decode (seq=1),
        // the buffer is pre-grown during prefill so this path isn't hit.
        // For prefill (non-compiled), .item() is safe on a concrete array.
        let prev_concrete = self.offset.item::<i32>();

        let need_grow = self
            .keys
            .as_ref()
            .is_none_or(|k| (prev_concrete + new_tokens) > k.shape()[2]);

        if need_grow {
            let b = keys.shape()[0];
            let n_kv_heads = keys.shape()[1];
            let k_head_dim = keys.shape()[3];
            let v_head_dim = values.shape()[3];

            let n_steps = (self.step + new_tokens - 1) / self.step;
            let new_slots = n_steps * self.step;

            let new_k = ops::zeros_dtype(&[b, n_kv_heads, new_slots, k_head_dim], keys.dtype())?;
            let new_v = ops::zeros_dtype(&[b, n_kv_heads, new_slots, v_head_dim], values.dtype())?;

            let (grown_k, grown_v) = match (self.keys.as_ref(), self.values.as_ref()) {
                (Some(old_k), Some(old_v)) => {
                    let (trimmed_k, trimmed_v) = if prev_concrete % self.step != 0 {
                        (
                            slice_axis2(old_k, 0, prev_concrete)?,
                            slice_axis2(old_v, 0, prev_concrete)?,
                        )
                    } else {
                        (old_k.clone(), old_v.clone())
                    };
                    let cat_k = concatenate_axis(&[trimmed_k, new_k], 2)?;
                    let cat_v = concatenate_axis(&[trimmed_v, new_v], 2)?;
                    (cat_k, cat_v)
                }
                _ => (new_k, new_v),
            };
            self.keys = Some(grown_k);
            self.values = Some(grown_v);
        }

        let k = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("Keys cannot be None after grow"))?;
        let v = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("Values cannot be None after grow"))?;

        // Write new K/V at the current offset position.
        let updated_k = slice_update_axis2(k, &keys, prev_concrete, new_tokens)?;
        let updated_v = slice_update_axis2(v, &values, prev_concrete, new_tokens)?;
        self.keys = Some(updated_k);
        self.values = Some(updated_v);

        let new_offset = prev_concrete + new_tokens;
        self.offset = Array::from_int(new_offset);

        // Read: slice up to the new offset.
        let result_k = slice_axis2(
            self.keys
                .as_ref()
                .ok_or_else(|| Exception::custom("Keys cannot be None after update"))?,
            0,
            new_offset,
        )?;
        let result_v = slice_axis2(
            self.values
                .as_ref()
                .ok_or_else(|| Exception::custom("Values cannot be None after update"))?,
            0,
            new_offset,
        )?;

        Ok((result_k, result_v))
    }
}

impl SteppingKeyValueCache {
    /// Compiled-mode update: concatenation-based (fully traceable by mx.compile).
    ///
    /// Instead of slice_update into a pre-allocated buffer (which needs `.item()`
    /// for concrete indices), this simply concatenates the new KV onto the existing
    /// cache. No padding, no masking. Shape grows by `new_tokens` each step —
    /// `shapeless=true` in mx.compile handles this.
    ///
    /// Call only after prefill has populated `self.keys`/`self.values`.
    pub fn update_and_fetch_compiled(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        // Number of new tokens from input shape (concrete during trace)
        let new_tokens = keys.shape()[2];

        let (result_k, result_v) = match (self.keys.take(), self.values.take()) {
            (Some(old_k), Some(old_v)) => {
                let new_k = concatenate_axis(&[old_k, keys], 2)?;
                let new_v = concatenate_axis(&[old_v, values], 2)?;
                (new_k, new_v)
            }
            _ => (keys, values),
        };

        self.keys = Some(result_k.clone());
        self.values = Some(result_v.clone());
        self.offset = self.offset.add(Array::from_int(new_tokens))?;

        Ok((result_k, result_v))
    }
}

// ---------------------------------------------------------------------------
// FlatCache: flattened cache for compiled decode
// ---------------------------------------------------------------------------

/// Describes which flat-array indices belong to each layer's cache.
#[derive(Debug, Clone)]
pub enum FlatCacheSlot {
    /// GDN layer: conv_state + ssm_state + offset_scalar
    Gdn {
        conv_idx: usize,
        ssm_idx: usize,
        offset_idx: usize,
    },
    /// MHA layer: keys + values + offset_scalar
    Kv {
        keys_idx: usize,
        values_idx: usize,
        offset_idx: usize,
    },
}

/// All layer cache state flattened into a single `Vec<Array>`.
///
/// This enables `mlx_compile` by making cache state explicit inputs/outputs
/// instead of mutable side effects. The compiled closure receives and returns
/// these arrays, and the evaluator can trace through them.
#[derive(Debug, Clone)]
pub struct FlatCache {
    /// Flat array of all cache state. Layout described by `slots`.
    pub arrays: Vec<Array>,
    /// Per-layer metadata: which indices in `arrays` belong to each layer.
    pub slots: Vec<FlatCacheSlot>,
}

impl FlatCache {
    /// Pack a hybrid cache (Vec<Option<LayerCache>>) into a flat array vec.
    ///
    /// Any `None` arrays (uninitialized cache) become zero-filled sentinels
    /// with shapes that will be correct for the first decode step.
    /// Offsets are stored as scalar i32 arrays so compile can trace them.
    pub fn pack(
        layers: &[Option<crate::qwen3_next::LayerCache>],
    ) -> Result<Self, Exception> {
        let mut arrays = Vec::new();
        let mut slots = Vec::new();

        for layer in layers {
            let layer = layer
                .as_ref()
                .ok_or_else(|| Exception::custom("FlatCache::pack: layer cache is None"))?;
            match layer {
                crate::qwen3_next::LayerCache::Arrays(ac) => {
                    let conv_idx = arrays.len();
                    arrays.push(
                        ac.conv_state
                            .clone()
                            .unwrap_or_else(|| Array::from_f32(0.0)),
                    );
                    let ssm_idx = arrays.len();
                    arrays.push(
                        ac.ssm_state
                            .clone()
                            .unwrap_or_else(|| Array::from_f32(0.0)),
                    );
                    let offset_idx = arrays.len();
                    arrays.push(ac.offset.clone());
                    slots.push(FlatCacheSlot::Gdn {
                        conv_idx,
                        ssm_idx,
                        offset_idx,
                    });
                }
                crate::qwen3_next::LayerCache::KV(kv) => {
                    let keys_idx = arrays.len();
                    arrays.push(
                        kv.keys_ref()
                            .cloned()
                            .unwrap_or_else(|| Array::from_f32(0.0)),
                    );
                    let values_idx = arrays.len();
                    arrays.push(
                        kv.values_ref()
                            .cloned()
                            .unwrap_or_else(|| Array::from_f32(0.0)),
                    );
                    let offset_idx = arrays.len();
                    arrays.push(kv.offset_array_ref().clone());
                    slots.push(FlatCacheSlot::Kv {
                        keys_idx,
                        values_idx,
                        offset_idx,
                    });
                }
            }
        }

        Ok(Self { arrays, slots })
    }

    /// Unpack flat arrays back into the structured cache.
    pub fn unpack(
        &self,
        layers: &mut [Option<crate::qwen3_next::LayerCache>],
    ) -> Result<(), Exception> {
        for (i, slot) in self.slots.iter().enumerate() {
            let layer = layers
                .get_mut(i)
                .and_then(|l| l.as_mut())
                .ok_or_else(|| Exception::custom("FlatCache::unpack: layer index out of bounds"))?;
            match (slot, layer) {
                (
                    FlatCacheSlot::Gdn {
                        conv_idx,
                        ssm_idx,
                        offset_idx,
                    },
                    crate::qwen3_next::LayerCache::Arrays(ac),
                ) => {
                    ac.conv_state = Some(self.arrays[*conv_idx].clone());
                    ac.ssm_state = Some(self.arrays[*ssm_idx].clone());
                    ac.offset = self.arrays[*offset_idx].clone();
                }
                (
                    FlatCacheSlot::Kv {
                        keys_idx,
                        values_idx,
                        offset_idx,
                    },
                    crate::qwen3_next::LayerCache::KV(kv),
                ) => {
                    kv.set_keys(self.arrays[*keys_idx].clone());
                    kv.set_values(self.arrays[*values_idx].clone());
                    kv.set_offset_array(self.arrays[*offset_idx].clone());
                }
                _ => {
                    return Err(Exception::custom(
                        "FlatCache::unpack: slot type mismatch",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Number of arrays in the flat cache.
    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    /// Whether the flat cache is empty.
    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    /// Create a zero-filled KV pair with shape `[1, n_heads, seq_len, head_dim]`.
    fn make_kv_pair(seq_len: i32, head_dim: i32) -> (Array, Array) {
        let shape = [1, 2, seq_len, head_dim];
        (
            Array::zeros::<f32>(&shape).unwrap(),
            Array::zeros::<f32>(&shape).unwrap(),
        )
    }

    #[test]
    fn test_concat_cache_initial_update() {
        let mut cache = ConcatKeyValueCache::new();
        assert_eq!(cache.offset(), 0);
        assert!(cache.max_size().is_none());
        assert!(!cache.is_quantized());

        let (keys, values) = make_kv_pair(4, 8);
        let (result_keys, result_values) = cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(result_keys.shape(), &[1, 2, 4, 8]);
        assert_eq!(result_values.shape(), &[1, 2, 4, 8]);
        assert_eq!(cache.offset(), 4);
    }

    #[test]
    fn test_concat_cache_sequential_updates() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys1, values1) = make_kv_pair(4, 8);
        cache.update_and_fetch(keys1, values1).unwrap();
        assert_eq!(cache.offset(), 4);

        let (keys2, values2) = make_kv_pair(1, 8);
        let (result_keys, result_values) = cache.update_and_fetch(keys2, values2).unwrap();
        assert_eq!(result_keys.shape(), &[1, 2, 5, 8]);
        assert_eq!(result_values.shape(), &[1, 2, 5, 8]);
        assert_eq!(cache.offset(), 5);
    }

    #[test]
    fn test_concat_cache_many_sequential_updates() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys, values) = make_kv_pair(3, 8);
        cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(cache.offset(), 3);

        for i in 0..5 {
            let (k, v) = make_kv_pair(1, 8);
            let (rk, rv) = cache.update_and_fetch(k, v).unwrap();
            let expected_seq = 3 + i + 1;
            assert_eq!(cache.offset(), expected_seq);
            assert_eq!(rk.shape(), &[1, 2, expected_seq, 8]);
            assert_eq!(rv.shape(), &[1, 2, expected_seq, 8]);
        }

        assert_eq!(cache.offset(), 8);
    }

    #[test]
    fn test_concat_cache_default_values() {
        let cache = ConcatKeyValueCache::default();
        assert_eq!(cache.offset(), 0);
        assert!(cache.max_size().is_none());
        assert!(!cache.is_quantized());
        assert!(cache.group_size().is_none());
        assert!(cache.bits().is_none());
    }

    #[test]
    fn test_concat_cache_mismatched_shapes_error() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys1, values1) = make_kv_pair(4, 8);
        cache.update_and_fetch(keys1, values1).unwrap();

        // Mismatched head_dim (16 instead of 8)
        let (keys2, values2) = make_kv_pair(1, 16);
        let result = cache.update_and_fetch(keys2, values2);
        assert!(
            result.is_err(),
            "Mismatched head_dim should fail concatenation"
        );
    }

    #[test]
    fn test_concat_cache_1d_keys_error() {
        let mut cache = ConcatKeyValueCache::new();
        let keys = Array::zeros::<f32>(&[4]).unwrap();
        let values = Array::zeros::<f32>(&[4]).unwrap();
        let result = cache.update_and_fetch(keys, values);
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_cache_ref_mut_delegation() {
        let mut cache = ConcatKeyValueCache::new();
        let cache_ref: &mut ConcatKeyValueCache = &mut cache;

        assert_eq!(KeyValueCache::offset(&cache_ref), 0);
        assert!(KeyValueCache::max_size(&cache_ref).is_none());
        assert!(!KeyValueCache::is_quantized(&cache_ref));
        assert!(KeyValueCache::group_size(&cache_ref).is_none());
        assert!(KeyValueCache::bits(&cache_ref).is_none());

        let (keys, values) = make_kv_pair(3, 8);
        let (rk, rv) = cache_ref.update_and_fetch(keys, values).unwrap();
        assert_eq!(rk.shape(), &[1, 2, 3, 8]);
        assert_eq!(rv.shape(), &[1, 2, 3, 8]);
        assert_eq!(KeyValueCache::offset(&cache_ref), 3);
    }

    // --- SteppingKeyValueCache tests ---

    #[test]
    fn test_stepping_cache_initial_update() {
        let mut cache = SteppingKeyValueCache::new();
        assert_eq!(cache.offset(), 0);

        let (keys, values) = make_kv_pair(4, 8);
        let (rk, rv) = cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(rk.shape(), &[1, 2, 4, 8]);
        assert_eq!(rv.shape(), &[1, 2, 4, 8]);
        assert_eq!(cache.offset(), 4);
        // Internal buffer should be 256 slots
        assert_eq!(cache.keys.as_ref().unwrap().shape()[2], 256);
    }

    #[test]
    fn test_stepping_cache_sequential_decode() {
        let mut cache = SteppingKeyValueCache::new();

        // Prefill with 4 tokens
        let (keys, values) = make_kv_pair(4, 8);
        cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(cache.offset(), 4);

        // Decode 5 single tokens
        for i in 0..5 {
            let (k, v) = make_kv_pair(1, 8);
            let (rk, rv) = cache.update_and_fetch(k, v).unwrap();
            let expected_seq = 4 + i + 1;
            assert_eq!(cache.offset(), expected_seq);
            assert_eq!(rk.shape(), &[1, 2, expected_seq, 8]);
            assert_eq!(rv.shape(), &[1, 2, expected_seq, 8]);
        }
        // Should still be using the initial 256-slot buffer (no regrowth)
        assert_eq!(cache.keys.as_ref().unwrap().shape()[2], 256);
    }

    #[test]
    fn test_stepping_cache_values_preserved() {
        let mut cache = SteppingKeyValueCache::new();

        // Write ones
        let ones_k = Array::ones::<f32>(&[1, 1, 2, 4]).unwrap();
        let ones_v = Array::ones::<f32>(&[1, 1, 2, 4]).unwrap();
        cache.update_and_fetch(ones_k, ones_v).unwrap();

        // Write twos
        let two = Array::from_f32(2.0);
        let twos_k = Array::full::<f32>(&[1, 1, 1, 4], &two).unwrap();
        let twos_v = Array::full::<f32>(&[1, 1, 1, 4], &two).unwrap();
        let (rk, rv) = cache.update_and_fetch(twos_k, twos_v).unwrap();

        rk.eval().unwrap();
        rv.eval().unwrap();

        assert_eq!(rk.shape(), &[1, 1, 3, 4]);
        // First 2 tokens should be 1.0, third should be 2.0
        let k_data: Vec<f32> = rk.as_slice().to_vec();
        assert!((k_data[0] - 1.0).abs() < 1e-6);
        assert!((k_data[4] - 1.0).abs() < 1e-6);
        assert!((k_data[8] - 2.0).abs() < 1e-6);
    }
}
