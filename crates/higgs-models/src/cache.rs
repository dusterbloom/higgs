use std::sync::Arc;

use mlx_rs::{Array, Dtype, Stream, error::Exception, ops, ops::concatenate_axis};

use crate::turboquant::{
    KvCacheConfig, KvCacheMode, QuantizedKey, QuantizedValue, TurboQuantContext,
};

/// View over a KV cache after appending new tokens.
#[derive(Debug, Clone)]
pub enum KvCacheView {
    Dense { keys: Array, values: Array },
    TurboQuant(TurboQuantKvView),
}

impl KvCacheView {
    pub fn into_dense(self) -> Result<(Array, Array), Exception> {
        match self {
            Self::Dense { keys, values } => Ok((keys, values)),
            Self::TurboQuant(view) => view.materialize_dense(),
        }
    }

    pub fn turboquant(&self) -> Option<&TurboQuantKvView> {
        match self {
            Self::Dense { .. } => None,
            Self::TurboQuant(view) => Some(view),
        }
    }
}

/// Quantized cache view used by the TurboQuant decode path.
#[derive(Debug, Clone)]
pub struct TurboQuantKvView {
    pub context: Arc<TurboQuantContext>,
    pub key_codes: Array,
    pub key_norms: Array,
    pub key_qjl_signs: Array,
    pub key_gammas: Array,
    pub value_codes: Array,
    pub value_norms: Array,
    pub seq_len: i32,
}

impl TurboQuantKvView {
    pub fn materialize_dense(&self) -> Result<(Array, Array), Exception> {
        let num_kv_heads = usize_from_i32(self.context.num_kv_heads, "num_kv_heads")?;
        let head_dim = usize_from_i32(self.context.head_dim, "head_dim")?;
        let seq_len = usize_from_i32(self.seq_len, "seq_len")?;
        let key_code_bytes = usize_from_i32(self.context.key_code_bytes, "key_code_bytes")?;
        let value_code_bytes = usize_from_i32(self.context.value_code_bytes, "value_code_bytes")?;
        let sign_bytes = usize_from_i32(self.context.sign_bytes, "sign_bytes")?;

        let key_codes = self.key_codes.as_slice::<u8>();
        let key_norms = self.key_norms.as_slice::<f32>();
        let key_qjl_signs = self.key_qjl_signs.as_slice::<u8>();
        let key_gammas = self.key_gammas.as_slice::<f32>();
        let value_codes = self.value_codes.as_slice::<u8>();
        let value_norms = self.value_norms.as_slice::<f32>();

        let total_values = checked_mul(num_kv_heads, seq_len, "cache size")?;
        let total_dense = checked_mul(total_values, head_dim, "dense cache size")?;
        let mut dense_keys = Vec::with_capacity(total_dense);
        let mut dense_values = Vec::with_capacity(total_dense);

        for head in 0..num_kv_heads {
            for pos in 0..seq_len {
                let scalar_index = checked_add(
                    checked_mul(head, seq_len, "scalar index")?,
                    pos,
                    "scalar index",
                )?;
                let key_code_start = checked_mul(scalar_index, key_code_bytes, "key code index")?;
                let key_code_end = checked_add(key_code_start, key_code_bytes, "key code range")?;
                let sign_start = checked_mul(scalar_index, sign_bytes, "key sign index")?;
                let sign_end = checked_add(sign_start, sign_bytes, "key sign range")?;
                let value_code_start =
                    checked_mul(scalar_index, value_code_bytes, "value code index")?;
                let value_code_end = checked_add(
                    value_code_start,
                    value_code_bytes,
                    "value code range",
                )?;

                let key = QuantizedKey {
                    norm: key_norms[scalar_index],
                    gamma: key_gammas[scalar_index],
                    codes: key_codes[key_code_start..key_code_end].to_vec(),
                    qjl_signs: key_qjl_signs[sign_start..sign_end].to_vec(),
                };
                let value = QuantizedValue {
                    norm: value_norms[scalar_index],
                    codes: value_codes[value_code_start..value_code_end].to_vec(),
                };

                dense_keys.extend(self.context.dequantize_key(&key)?);
                dense_values.extend(self.context.dequantize_value(&value)?);
            }
        }

        let shape = [
            1,
            self.context.num_kv_heads,
            self.seq_len,
            self.context.head_dim,
        ];
        let keys = Array::from_slice(&dense_keys, &shape);
        let values = Array::from_slice(&dense_values, &shape);
        Ok((keys, values))
    }

    pub fn decode_scores(&self, queries: &Array, num_heads: i32) -> Result<Array, Exception> {
        let query_shape = queries.shape();
        if query_shape != [1, num_heads, 1, self.context.head_dim] {
            return Err(Exception::custom("TurboQuant decode expects [1, H, 1, D] queries"));
        }

        let queries = queries
            .as_dtype(Dtype::Float32)?
            .reshape(&[num_heads, self.context.head_dim])?;
        let q_rot = self.context.rotate_queries(&queries)?;
        let q_qjl = self.context.project_queries_qjl(&queries)?;

        crate::turboquant::decode_scores(
            &q_rot,
            &q_qjl,
            &self.key_codes,
            &self.key_norms,
            &self.key_qjl_signs,
            &self.key_gammas,
            &self.context.key_centroids_array()?,
            num_heads,
            self.context.num_kv_heads,
            self.context.head_dim,
            self.seq_len,
            self.seq_len,
            self.context.config.key_bits(),
            self.context.key_code_bytes,
            self.context.sign_bytes,
        )
    }

    pub fn decode_values(&self, weights: &Array, num_heads: i32) -> Result<Array, Exception> {
        let weights = weights
            .as_dtype(Dtype::Float32)?
            .reshape(&[num_heads, self.seq_len])?;
        let out_rot = crate::turboquant::decode_weighted_values(
            &weights,
            &self.value_codes,
            &self.value_norms,
            &self.context.value_centroids_array()?,
            num_heads,
            self.context.num_kv_heads,
            self.context.head_dim,
            self.seq_len,
            self.seq_len,
            self.context.config.bits,
            self.context.value_code_bytes,
        )?;

        out_rot
            .matmul(&self.context.rotation_array()?)?
            .reshape(&[1, num_heads, 1, self.context.head_dim])
    }
}

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
    fn offset(&self) -> i32;

    /// Maximum cache size, if bounded.
    fn max_size(&self) -> Option<i32>;

    /// Append new key/value tensors and return a cache view.
    fn update_and_view(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception>;

    /// Append new key/value tensors and return the full cached key/value.
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        self.update_and_view(keys, values)?.into_dense()
    }
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

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_view(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception> {
        T::update_and_view(self, keys, values)
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

    fn update_and_view(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception> {
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

        Ok(KvCacheView::Dense {
            keys: result_keys,
            values: result_values,
        })
    }
}

/// Pre-allocated KV cache that grows in chunks, avoiding per-token allocation.
///
/// Matches Python `mlx_lm`'s `KVCache`: pre-allocates 256 slots at a time and
/// uses `mlx_slice_update` for writes instead of concatenation every token.
/// Keys/values have shape `[B, n_heads, seq_len, head_dim]` with sequence on axis 2.
#[derive(Debug, Clone)]
pub struct SteppingKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    turbo: Option<TurboQuantStorage>,
    config: KvCacheConfig,
    offset: i32,
    step: i32,
}

#[derive(Debug, Clone)]
struct TurboQuantStorage {
    context: Arc<TurboQuantContext>,
    key_codes: Vec<u8>,
    key_norms: Vec<f32>,
    key_qjl_signs: Vec<u8>,
    key_gammas: Vec<f32>,
    value_codes: Vec<u8>,
    value_norms: Vec<f32>,
    capacity: i32,
}

impl Default for SteppingKeyValueCache {
    fn default() -> Self {
        Self {
            keys: None,
            values: None,
            turbo: None,
            config: KvCacheConfig::default(),
            offset: 0,
            step: 256,
        }
    }
}

impl SteppingKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_turbo(
        config: KvCacheConfig,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self, Exception> {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            ..config
        };
        let context = Arc::new(TurboQuantContext::new(config, head_dim, num_kv_heads)?);
        Ok(Self {
            keys: None,
            values: None,
            turbo: Some(TurboQuantStorage::new(context)),
            config,
            offset: 0,
            step: 256,
        })
    }

    pub const fn kv_cache_config(&self) -> KvCacheConfig {
        self.config
    }

    fn update_dense(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception> {
        let prev = self.offset;
        let new_tokens = keys.shape()[2];

        let need_grow = self
            .keys
            .as_ref()
            .is_none_or(|k| (prev + new_tokens) > k.shape()[2]);

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
                    let (trimmed_k, trimmed_v) = if prev % self.step != 0 {
                        (slice_axis2(old_k, 0, prev)?, slice_axis2(old_v, 0, prev)?)
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

        let updated_k = slice_update_axis2(k, &keys, prev, new_tokens)?;
        let updated_v = slice_update_axis2(v, &values, prev, new_tokens)?;
        self.keys = Some(updated_k);
        self.values = Some(updated_v);

        self.offset = prev + new_tokens;

        let result_k = slice_axis2(
            self.keys
                .as_ref()
                .ok_or_else(|| Exception::custom("Keys cannot be None after update"))?,
            0,
            self.offset,
        )?;
        let result_v = slice_axis2(
            self.values
                .as_ref()
                .ok_or_else(|| Exception::custom("Values cannot be None after update"))?,
            0,
            self.offset,
        )?;

        Ok(KvCacheView::Dense {
            keys: result_k,
            values: result_v,
        })
    }
}

impl TurboQuantStorage {
    fn new(context: Arc<TurboQuantContext>) -> Self {
        Self {
            context,
            key_codes: vec![],
            key_norms: vec![],
            key_qjl_signs: vec![],
            key_gammas: vec![],
            value_codes: vec![],
            value_norms: vec![],
            capacity: 0,
        }
    }

    fn ensure_capacity(&mut self, required: i32, step: i32) -> Result<(), Exception> {
        if required <= self.capacity {
            return Ok(());
        }
        let new_capacity = ((required + step - 1) / step) * step;
        let heads = usize_from_i32(self.context.num_kv_heads, "num_kv_heads")?;
        let capacity = usize_from_i32(new_capacity, "new_capacity")?;
        let key_code_bytes = usize_from_i32(self.context.key_code_bytes, "key_code_bytes")?;
        let value_code_bytes = usize_from_i32(self.context.value_code_bytes, "value_code_bytes")?;
        let sign_bytes = usize_from_i32(self.context.sign_bytes, "sign_bytes")?;
        let scalar_len = checked_mul(heads, capacity, "TurboQuant scalar capacity")?;
        let key_byte_len = checked_mul(scalar_len, key_code_bytes, "TurboQuant key bytes")?;
        let value_byte_len = checked_mul(scalar_len, value_code_bytes, "TurboQuant value bytes")?;
        let sign_len = checked_mul(scalar_len, sign_bytes, "TurboQuant sign bytes")?;

        self.key_codes.resize(key_byte_len, 0);
        self.key_norms.resize(scalar_len, 0.0);
        self.key_qjl_signs.resize(sign_len, 0);
        self.key_gammas.resize(scalar_len, 0.0);
        self.value_codes.resize(value_byte_len, 0);
        self.value_norms.resize(scalar_len, 0.0);
        self.capacity = new_capacity;
        Ok(())
    }

    fn append(
        &mut self,
        keys: Array,
        values: Array,
        prev: i32,
        step: i32,
    ) -> Result<TurboQuantKvView, Exception> {
        validate_turboquant_shapes(&keys, &values, &self.context)?;

        let new_tokens = keys.shape()[2];
        self.ensure_capacity(prev + new_tokens, step)?;

        let keys = keys.as_dtype(Dtype::Float32)?;
        let values = values.as_dtype(Dtype::Float32)?;
        keys.eval()?;
        values.eval()?;

        let num_kv_heads = usize_from_i32(self.context.num_kv_heads, "num_kv_heads")?;
        let head_dim = usize_from_i32(self.context.head_dim, "head_dim")?;
        let new_tokens_usize = usize_from_i32(new_tokens, "new_tokens")?;
        let prev_usize = usize_from_i32(prev, "prev offset")?;
        let key_stride = checked_mul(new_tokens_usize, head_dim, "key stride")?;
        let value_stride = checked_mul(new_tokens_usize, head_dim, "value stride")?;
        let keys_slice = keys.as_slice::<f32>();
        let values_slice = values.as_slice::<f32>();

        for head in 0..num_kv_heads {
            for token in 0..new_tokens_usize {
                let start = checked_add(
                    checked_mul(head, key_stride, "key offset")?,
                    checked_mul(token, head_dim, "key token offset")?,
                    "key start",
                )?;
                let end = checked_add(start, head_dim, "key end")?;
                let quantized_key = self.context.quantize_key(&keys_slice[start..end])?;
                self.write_key(head, prev_usize + token, &quantized_key)?;

                let value_start = checked_add(
                    checked_mul(head, value_stride, "value offset")?,
                    checked_mul(token, head_dim, "value token offset")?,
                    "value start",
                )?;
                let value_end = checked_add(value_start, head_dim, "value end")?;
                let quantized_value = self.context.quantize_value(&values_slice[value_start..value_end])?;
                self.write_value(head, prev_usize + token, &quantized_value)?;
            }
        }

        self.view(prev + new_tokens)
    }

    fn view(&self, seq_len: i32) -> Result<TurboQuantKvView, Exception> {
        let heads = usize_from_i32(self.context.num_kv_heads, "num_kv_heads")?;
        let capacity = usize_from_i32(self.capacity, "capacity")?;
        let seq_len_usize = usize_from_i32(seq_len, "seq_len")?;
        let key_code_bytes = usize_from_i32(self.context.key_code_bytes, "key_code_bytes")?;
        let value_code_bytes = usize_from_i32(self.context.value_code_bytes, "value_code_bytes")?;
        let sign_bytes = usize_from_i32(self.context.sign_bytes, "sign_bytes")?;

        let key_codes = collect_prefix_bytes(
            &self.key_codes,
            heads,
            capacity,
            seq_len_usize,
            key_code_bytes,
        )?;
        let value_codes = collect_prefix_bytes(
            &self.value_codes,
            heads,
            capacity,
            seq_len_usize,
            value_code_bytes,
        )?;
        let key_qjl_signs = collect_prefix_bytes(
            &self.key_qjl_signs,
            heads,
            capacity,
            seq_len_usize,
            sign_bytes,
        )?;
        let key_norms = collect_prefix_scalars(&self.key_norms, heads, capacity, seq_len_usize)?;
        let key_gammas = collect_prefix_scalars(&self.key_gammas, heads, capacity, seq_len_usize)?;
        let value_norms =
            collect_prefix_scalars(&self.value_norms, heads, capacity, seq_len_usize)?;

        Ok(TurboQuantKvView {
            context: Arc::clone(&self.context),
            key_codes: Array::from_slice(
                &key_codes,
                &[self.context.num_kv_heads, seq_len, self.context.key_code_bytes],
            ),
            key_norms: Array::from_slice(&key_norms, &[self.context.num_kv_heads, seq_len]),
            key_qjl_signs: Array::from_slice(
                &key_qjl_signs,
                &[self.context.num_kv_heads, seq_len, self.context.sign_bytes],
            ),
            key_gammas: Array::from_slice(&key_gammas, &[self.context.num_kv_heads, seq_len]),
            value_codes: Array::from_slice(
                &value_codes,
                &[self.context.num_kv_heads, seq_len, self.context.value_code_bytes],
            ),
            value_norms: Array::from_slice(&value_norms, &[self.context.num_kv_heads, seq_len]),
            seq_len,
        })
    }

    fn write_key(&mut self, head: usize, pos: usize, key: &QuantizedKey) -> Result<(), Exception> {
        let capacity = usize_from_i32(self.capacity, "capacity")?;
        let key_code_bytes = usize_from_i32(self.context.key_code_bytes, "key_code_bytes")?;
        let sign_bytes = usize_from_i32(self.context.sign_bytes, "sign_bytes")?;
        if key.codes.len() != key_code_bytes || key.qjl_signs.len() != sign_bytes {
            return Err(Exception::custom("TurboQuant key payload length mismatch"));
        }

        let scalar_index = checked_add(checked_mul(head, capacity, "scalar head stride")?, pos, "scalar index")?;
        let key_code_start =
            checked_mul(scalar_index, key_code_bytes, "key code start")?;
        let key_code_end = checked_add(key_code_start, key_code_bytes, "key code end")?;
        let sign_start = checked_mul(scalar_index, sign_bytes, "key sign start")?;
        let sign_end = checked_add(sign_start, sign_bytes, "key sign end")?;

        self.key_codes[key_code_start..key_code_end].copy_from_slice(&key.codes);
        self.key_qjl_signs[sign_start..sign_end].copy_from_slice(&key.qjl_signs);
        self.key_norms[scalar_index] = key.norm;
        self.key_gammas[scalar_index] = key.gamma;
        Ok(())
    }

    fn write_value(
        &mut self,
        head: usize,
        pos: usize,
        value: &QuantizedValue,
    ) -> Result<(), Exception> {
        let capacity = usize_from_i32(self.capacity, "capacity")?;
        let value_code_bytes = usize_from_i32(self.context.value_code_bytes, "value_code_bytes")?;
        if value.codes.len() != value_code_bytes {
            return Err(Exception::custom("TurboQuant value payload length mismatch"));
        }

        let scalar_index = checked_add(checked_mul(head, capacity, "scalar head stride")?, pos, "scalar index")?;
        let value_code_start =
            checked_mul(scalar_index, value_code_bytes, "value code start")?;
        let value_code_end = checked_add(value_code_start, value_code_bytes, "value code end")?;

        self.value_codes[value_code_start..value_code_end].copy_from_slice(&value.codes);
        self.value_norms[scalar_index] = value.norm;
        Ok(())
    }
}

impl KeyValueCache for SteppingKeyValueCache {
    fn is_quantized(&self) -> bool {
        self.config.is_turboquant()
    }

    fn bits(&self) -> Option<i32> {
        self.config.is_turboquant().then_some(i32::from(self.config.bits))
    }

    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_view(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception> {
        let view = if let Some(turbo) = self.turbo.as_mut() {
            KvCacheView::TurboQuant(turbo.append(keys, values, self.offset, self.step)?)
        } else {
            self.update_dense(keys, values)?
        };
        self.offset = match &view {
            KvCacheView::Dense { keys, .. } => keys.shape()[2],
            KvCacheView::TurboQuant(view) => view.seq_len,
        };
        Ok(view)
    }
}

fn validate_turboquant_shapes(
    keys: &Array,
    values: &Array,
    context: &TurboQuantContext,
) -> Result<(), Exception> {
    let key_shape = keys.shape();
    let value_shape = values.shape();
    if key_shape.len() != 4 || value_shape.len() != 4 {
        return Err(Exception::custom("TurboQuant cache expects 4D [B, H, T, D] tensors"));
    }
    if key_shape[0] != 1 || value_shape[0] != 1 {
        return Err(Exception::custom("TurboQuant cache currently only supports batch size 1"));
    }
    if key_shape[1] != context.num_kv_heads || value_shape[1] != context.num_kv_heads {
        return Err(Exception::custom("TurboQuant KV head count mismatch"));
    }
    if key_shape[2] != value_shape[2] {
        return Err(Exception::custom("TurboQuant keys/values token count mismatch"));
    }
    if key_shape[3] != context.head_dim || value_shape[3] != context.head_dim {
        return Err(Exception::custom("TurboQuant head_dim mismatch"));
    }
    Ok(())
}

fn collect_prefix_bytes(
    data: &[u8],
    heads: usize,
    capacity: usize,
    seq_len: usize,
    bytes_per_slot: usize,
) -> Result<Vec<u8>, Exception> {
    let total_slots = checked_mul(heads, seq_len, "TurboQuant byte prefix slots")?;
    let total_bytes = checked_mul(total_slots, bytes_per_slot, "TurboQuant byte prefix size")?;
    let mut out = Vec::with_capacity(total_bytes);
    let head_stride = checked_mul(capacity, bytes_per_slot, "TurboQuant head byte stride")?;
    let prefix_len = checked_mul(seq_len, bytes_per_slot, "TurboQuant byte prefix len")?;
    for head in 0..heads {
        let head_start = checked_mul(head, head_stride, "TurboQuant byte head start")?;
        let head_end = checked_add(head_start, prefix_len, "TurboQuant byte head end")?;
        out.extend_from_slice(&data[head_start..head_end]);
    }
    Ok(out)
}

fn collect_prefix_scalars(
    data: &[f32],
    heads: usize,
    capacity: usize,
    seq_len: usize,
) -> Result<Vec<f32>, Exception> {
    let total_scalars = checked_mul(heads, seq_len, "TurboQuant scalar prefix size")?;
    let mut out = Vec::with_capacity(total_scalars);
    for head in 0..heads {
        let head_start = checked_mul(head, capacity, "TurboQuant scalar head start")?;
        let head_end = checked_add(head_start, seq_len, "TurboQuant scalar head end")?;
        out.extend_from_slice(&data[head_start..head_end]);
    }
    Ok(out)
}

fn usize_from_i32(value: i32, label: &str) -> Result<usize, Exception> {
    usize::try_from(value).map_err(|_| Exception::custom(format!("{label} conversion overflow")))
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize, Exception> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Exception::custom(format!("{label} overflow")))
}

fn checked_add(lhs: usize, rhs: usize, label: &str) -> Result<usize, Exception> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Exception::custom(format!("{label} overflow")))
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

    #[test]
    fn test_turboquant_cache_round_trips_dense_fetch() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 7,
        };
        let mut cache = SteppingKeyValueCache::new_turbo(config, 2, 8).unwrap();
        let (keys, values) = make_kv_pair(3, 8);
        let (dense_keys, dense_values) = cache.update_and_fetch(keys, values).unwrap();

        assert_eq!(dense_keys.shape(), &[1, 2, 3, 8]);
        assert_eq!(dense_values.shape(), &[1, 2, 3, 8]);
        assert!(cache.is_quantized());
        assert_eq!(cache.bits(), Some(3));
    }

    #[test]
    fn test_turboquant_cache_returns_quantized_view() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 11,
        };
        let mut cache = SteppingKeyValueCache::new_turbo(config, 2, 8).unwrap();
        let (keys, values) = make_kv_pair(2, 8);
        let view = cache.update_and_view(keys, values).unwrap();

        let turbo = view.turboquant().unwrap();
        assert_eq!(turbo.seq_len, 2);
        assert_eq!(turbo.key_codes.shape(), &[2, 2, 2]);
        assert_eq!(turbo.value_codes.shape(), &[2, 2, 3]);
    }
}
