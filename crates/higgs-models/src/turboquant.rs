#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::{Mutex, OnceLock};

use mlx_rs::{Array, Dtype, Stream, error::Exception};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

const ONE_BIT_CENTROIDS: [f32; 2] = [-0.797_884_6, 0.797_884_6];
const TWO_BIT_CENTROIDS: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
const THREE_BIT_CENTROIDS: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];
const FOUR_BIT_CENTROIDS: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881, 0.6568,
    0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
];
const SQRT_PI_OVER_TWO: f32 = 1.253_314_1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum KvCacheMode {
    #[default]
    Off,
    Turboquant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheConfig {
    #[serde(default)]
    pub mode: KvCacheMode,
    #[serde(default = "default_bits")]
    pub bits: u8,
    #[serde(default)]
    pub seed: u64,
}

const fn default_bits() -> u8 {
    3
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            mode: KvCacheMode::Off,
            bits: default_bits(),
            seed: 0,
        }
    }
}

impl KvCacheConfig {
    pub const fn is_turboquant(self) -> bool {
        matches!(self.mode, KvCacheMode::Turboquant)
    }

    pub fn validate(self) -> Result<(), Exception> {
        if self.is_turboquant() && !(2..=4).contains(&self.bits) {
            return Err(Exception::custom(format!(
                "TurboQuant bits must be in [2, 4], got {}",
                self.bits
            )));
        }
        Ok(())
    }

    pub const fn key_bits(self) -> u8 {
        self.bits - 1
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedValue {
    pub norm: f32,
    pub codes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct QuantizedKey {
    pub norm: f32,
    pub gamma: f32,
    pub codes: Vec<u8>,
    pub qjl_signs: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct TurboQuantContext {
    pub config: KvCacheConfig,
    pub head_dim: i32,
    pub num_kv_heads: i32,
    pub rotation: Vec<f32>,
    pub rotation_t: Vec<f32>,
    pub qjl: Vec<f32>,
    pub qjl_t: Vec<f32>,
    pub key_centroids: Vec<f32>,
    pub value_centroids: Vec<f32>,
    pub key_code_bytes: i32,
    pub value_code_bytes: i32,
    pub sign_bytes: i32,
}

impl TurboQuantContext {
    pub fn new(config: KvCacheConfig, head_dim: i32, num_kv_heads: i32) -> Result<Self, Exception> {
        config.validate()?;
        if head_dim <= 0 || num_kv_heads <= 0 {
            return Err(Exception::custom(
                "TurboQuant head_dim and num_kv_heads must be > 0",
            ));
        }
        let dim = usize::try_from(head_dim).map_err(|_| Exception::custom("head_dim overflow"))?;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let rotation = generate_orthogonal_matrix(dim, config.seed);
        let rotation_t = transpose(&rotation, dim);
        let qjl = generate_gaussian_matrix(dim, config.seed.wrapping_add(1));
        let qjl_t = transpose(&qjl, dim);
        let key_centroids = scaled_centroids(config.key_bits(), scale)?;
        let value_centroids = scaled_centroids(config.bits, scale)?;

        let key_code_bytes = packed_bytes(dim, config.key_bits())?;
        let value_code_bytes = packed_bytes(dim, config.bits)?;
        let sign_bytes = packed_sign_bytes(dim)?;

        Ok(Self {
            config,
            head_dim,
            num_kv_heads,
            rotation,
            rotation_t,
            qjl,
            qjl_t,
            key_centroids,
            value_centroids,
            key_code_bytes,
            value_code_bytes,
            sign_bytes,
        })
    }

    pub fn quantize_value(&self, values: &[f32]) -> Result<QuantizedValue, Exception> {
        let dim =
            usize::try_from(self.head_dim).map_err(|_| Exception::custom("head_dim overflow"))?;
        if values.len() != dim {
            return Err(Exception::custom("TurboQuant value length mismatch"));
        }

        let norm = l2_norm(values);
        if norm <= f32::EPSILON {
            return Ok(QuantizedValue {
                norm: 0.0,
                codes: vec![0; usize::try_from(self.value_code_bytes).unwrap_or(0)],
            });
        }

        let normalized: Vec<f32> = values.iter().map(|value| *value / norm).collect();
        let rotated = mat_vec(&self.rotation, dim, &normalized);
        let indices = quantize_rotated(&rotated, &self.value_centroids);

        Ok(QuantizedValue {
            norm,
            codes: pack_indices(&indices, self.config.bits),
        })
    }

    pub fn quantize_key(&self, keys: &[f32]) -> Result<QuantizedKey, Exception> {
        let dim =
            usize::try_from(self.head_dim).map_err(|_| Exception::custom("head_dim overflow"))?;
        if keys.len() != dim {
            return Err(Exception::custom("TurboQuant key length mismatch"));
        }

        let norm = l2_norm(keys);
        let key_code_len = usize::try_from(self.key_code_bytes).unwrap_or(0);
        let sign_len = usize::try_from(self.sign_bytes).unwrap_or(0);
        if norm <= f32::EPSILON {
            return Ok(QuantizedKey {
                norm: 0.0,
                gamma: 0.0,
                codes: vec![0; key_code_len],
                qjl_signs: vec![0; sign_len],
            });
        }

        let normalized: Vec<f32> = keys.iter().map(|value| *value / norm).collect();
        let rotated = mat_vec(&self.rotation, dim, &normalized);
        let mse_indices = quantize_rotated(&rotated, &self.key_centroids);
        let rotated_approx = dequantize_rotated(
            &pack_indices(&mse_indices, self.config.key_bits()),
            self.config.key_bits(),
            &self.key_centroids,
            dim,
            1.0,
        );
        let approx = mat_vec(&self.rotation_t, dim, &rotated_approx);

        let residual: Vec<f32> = normalized
            .iter()
            .zip(approx.iter())
            .map(|(lhs, rhs)| *lhs - *rhs)
            .collect();
        let gamma = l2_norm(&residual) * norm;
        let qjl_proj = mat_vec(&self.qjl, dim, &residual);
        let qjl_signs = pack_signs(qjl_proj.iter().map(|value| *value >= 0.0));

        Ok(QuantizedKey {
            norm,
            gamma,
            codes: pack_indices(&mse_indices, self.config.key_bits()),
            qjl_signs,
        })
    }

    pub fn dequantize_value(&self, value: &QuantizedValue) -> Result<Vec<f32>, Exception> {
        let dim =
            usize::try_from(self.head_dim).map_err(|_| Exception::custom("head_dim overflow"))?;
        let rotated = dequantize_rotated(
            &value.codes,
            self.config.bits,
            &self.value_centroids,
            dim,
            value.norm,
        );
        Ok(mat_vec(&self.rotation_t, dim, &rotated))
    }

    pub fn dequantize_key(&self, key: &QuantizedKey) -> Result<Vec<f32>, Exception> {
        let dim =
            usize::try_from(self.head_dim).map_err(|_| Exception::custom("head_dim overflow"))?;
        let rotated = dequantize_rotated(
            &key.codes,
            self.config.key_bits(),
            &self.key_centroids,
            dim,
            key.norm,
        );
        let approx = mat_vec(&self.rotation_t, dim, &rotated);
        let residual = dequantize_qjl(&key.qjl_signs, key.gamma, &self.qjl_t, dim);
        Ok(approx
            .iter()
            .zip(residual.iter())
            .map(|(lhs, rhs)| *lhs + *rhs)
            .collect())
    }

    pub fn rotate_queries(&self, queries: &Array) -> Result<Array, Exception> {
        queries
            .as_dtype(Dtype::Float32)?
            .matmul(&self.rotation_t_array()?)
    }

    pub fn project_queries_qjl(&self, queries: &Array) -> Result<Array, Exception> {
        queries
            .as_dtype(Dtype::Float32)?
            .matmul(&self.qjl_t_array()?)
    }

    pub fn rotation_array(&self) -> Result<Array, Exception> {
        Ok(Array::from_slice(
            &self.rotation,
            &[self.head_dim, self.head_dim],
        ))
    }

    pub fn rotation_t_array(&self) -> Result<Array, Exception> {
        Ok(Array::from_slice(
            &self.rotation_t,
            &[self.head_dim, self.head_dim],
        ))
    }

    pub fn qjl_t_array(&self) -> Result<Array, Exception> {
        Ok(Array::from_slice(
            &self.qjl_t,
            &[self.head_dim, self.head_dim],
        ))
    }

    pub fn key_centroids_array(&self) -> Result<Array, Exception> {
        Ok(Array::from_slice(
            &self.key_centroids,
            &[i32::try_from(self.key_centroids.len())
                .map_err(|_| Exception::custom("key centroid len overflow"))?],
        ))
    }

    pub fn value_centroids_array(&self) -> Result<Array, Exception> {
        Ok(Array::from_slice(
            &self.value_centroids,
            &[i32::try_from(self.value_centroids.len())
                .map_err(|_| Exception::custom("value centroid len overflow"))?],
        ))
    }
}

#[allow(unsafe_code)]
pub(crate) fn decode_scores(
    q_rot: &Array,
    q_qjl: &Array,
    key_codes: &Array,
    key_norms: &Array,
    key_qjl: &Array,
    key_gammas: &Array,
    key_centroids: &Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    capacity: i32,
    seq_len: i32,
    key_bits: u8,
    key_code_bytes: i32,
    sign_bytes: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let kernel = SCORE_KERNEL.get_or_init(|| CachedMetalKernel(create_scores_kernel()));
    let config = configure_scores_kernel(
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        key_bits,
        key_code_bytes,
        sign_bytes,
    );

    let capacity_scalar = unsafe { mlx_sys::mlx_array_new_int(capacity) };
    let seq_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        q_rot.as_ptr(),
        q_qjl.as_ptr(),
        key_codes.as_ptr(),
        key_norms.as_ptr(),
        key_qjl.as_ptr(),
        key_gammas.as_ptr(),
        key_centroids.as_ptr(),
        capacity_scalar,
        seq_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            kernel.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = extract_single_output(status, outputs_vec, "turboquant_scores");

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(capacity_scalar);
        mlx_sys::mlx_array_free(seq_scalar);
    }

    result
}

#[allow(unsafe_code)]
pub(crate) fn decode_weighted_values(
    weights: &Array,
    value_codes: &Array,
    value_norms: &Array,
    value_centroids: &Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    capacity: i32,
    seq_len: i32,
    value_bits: u8,
    value_code_bytes: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let kernel = VALUE_KERNEL.get_or_init(|| CachedMetalKernel(create_values_kernel()));
    let config = configure_values_kernel(
        num_heads,
        num_kv_heads,
        head_dim,
        value_bits,
        value_code_bytes,
    );

    let capacity_scalar = unsafe { mlx_sys::mlx_array_new_int(capacity) };
    let seq_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        weights.as_ptr(),
        value_codes.as_ptr(),
        value_norms.as_ptr(),
        value_centroids.as_ptr(),
        capacity_scalar,
        seq_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            kernel.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = extract_single_output(status, outputs_vec, "turboquant_values");

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(capacity_scalar);
        mlx_sys::mlx_array_free(seq_scalar);
    }

    result
}

fn scaled_centroids(bits: u8, scale: f32) -> Result<Vec<f32>, Exception> {
    let base: &[f32] = match bits {
        1 => &ONE_BIT_CENTROIDS,
        2 => &TWO_BIT_CENTROIDS,
        3 => &THREE_BIT_CENTROIDS,
        4 => &FOUR_BIT_CENTROIDS,
        _ => {
            return Err(Exception::custom(format!(
                "TurboQuant centroids only implemented for 1-4 bits, got {}",
                bits
            )));
        }
    };
    Ok(base.iter().map(|value| *value * scale).collect())
}

fn packed_bytes(dim: usize, bits: u8) -> Result<i32, Exception> {
    let total_bits = dim
        .checked_mul(usize::from(bits))
        .ok_or_else(|| Exception::custom("TurboQuant packed byte overflow"))?;
    i32::try_from(total_bits.div_ceil(8))
        .map_err(|_| Exception::custom("TurboQuant packed byte overflow"))
}

fn packed_sign_bytes(dim: usize) -> Result<i32, Exception> {
    i32::try_from(dim.div_ceil(8)).map_err(|_| Exception::custom("TurboQuant sign byte overflow"))
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn dequantize_qjl(signs: &[u8], gamma: f32, qjl_t: &[f32], dim: usize) -> Vec<f32> {
    let mut z = vec![0.0_f32; dim];
    for index in 0..dim {
        z[index] = if unpack_sign(signs, index) { 1.0 } else { -1.0 };
    }
    let projection = mat_vec(qjl_t, dim, &z);
    let scale = gamma * SQRT_PI_OVER_TWO / dim as f32;
    projection.into_iter().map(|value| value * scale).collect()
}

fn quantize_rotated(values: &[f32], centroids: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|value| nearest_centroid(*value, centroids))
        .collect()
}

fn nearest_centroid(value: f32, centroids: &[f32]) -> u8 {
    let mut best_index = 0usize;
    let mut best_distance = f32::INFINITY;
    for (index, centroid) in centroids.iter().enumerate() {
        let distance = (value - *centroid).abs();
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }
    u8::try_from(best_index).unwrap_or(0)
}

fn dequantize_rotated(
    codes: &[u8],
    bits: u8,
    centroids: &[f32],
    dim: usize,
    norm: f32,
) -> Vec<f32> {
    (0..dim)
        .map(|index| centroids[usize::from(unpack_index(codes, index, bits))] * norm)
        .collect()
}

fn pack_indices(indices: &[u8], bits: u8) -> Vec<u8> {
    let total_bits = indices.len() * usize::from(bits);
    let mut packed = vec![0_u8; total_bits.div_ceil(8)];
    let mask = (1_u16 << bits) - 1;
    for (index, code) in indices.iter().enumerate() {
        let bit_index = index * usize::from(bits);
        let byte_index = bit_index / 8;
        let shift = bit_index % 8;
        let value = u16::from(*code) & mask;
        packed[byte_index] |= u8::try_from(value << shift).unwrap_or(0);
        if shift + usize::from(bits) > 8 {
            packed[byte_index + 1] |= u8::try_from(value >> (8 - shift)).unwrap_or(0);
        }
    }
    packed
}

fn unpack_index(data: &[u8], index: usize, bits: u8) -> u8 {
    let bit_index = index * usize::from(bits);
    let byte_index = bit_index / 8;
    let shift = bit_index % 8;
    let mask = (1_u16 << bits) - 1;
    let mut value = u16::from(data[byte_index] >> shift);
    if shift + usize::from(bits) > 8 {
        value |= u16::from(data[byte_index + 1]) << (8 - shift);
    }
    u8::try_from(value & mask).unwrap_or(0)
}

fn pack_signs(signs: impl IntoIterator<Item = bool>) -> Vec<u8> {
    let values: Vec<bool> = signs.into_iter().collect();
    let mut packed = vec![0_u8; values.len().div_ceil(8)];
    for (index, sign) in values.iter().enumerate() {
        if *sign {
            packed[index / 8] |= 1_u8 << (index % 8);
        }
    }
    packed
}

fn unpack_sign(data: &[u8], index: usize) -> bool {
    ((data[index / 8] >> (index % 8)) & 1_u8) == 1
}

fn mat_vec(matrix: &[f32], dim: usize, vector: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; dim];
    for row in 0..dim {
        let mut acc = 0.0_f32;
        for col in 0..dim {
            acc += matrix[row * dim + col] * vector[col];
        }
        out[row] = acc;
    }
    out
}

fn transpose(matrix: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            out[col * dim + row] = matrix[row * dim + col];
        }
    }
    out
}

fn generate_gaussian_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut matrix = vec![0.0_f32; dim * dim];
    for value in &mut matrix {
        *value = random_normal(&mut rng);
    }
    matrix
}

fn generate_orthogonal_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut cols: Vec<Vec<f32>> = (0..dim)
        .map(|_| (0..dim).map(|_| random_normal(&mut rng)).collect())
        .collect();

    for index in 0..dim {
        for prev in 0..index {
            let dot = dot(&cols[index], &cols[prev]);
            for dim_index in 0..dim {
                cols[index][dim_index] -= dot * cols[prev][dim_index];
            }
        }
        let norm = l2_norm(&cols[index]);
        if norm > f32::EPSILON {
            for value in &mut cols[index] {
                *value /= norm;
            }
        }
    }

    let mut matrix = vec![0.0_f32; dim * dim];
    for col in 0..dim {
        for row in 0..dim {
            matrix[row * dim + col] = cols[col][row];
        }
    }
    matrix
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
}

fn random_normal<R: Rng>(rng: &mut R) -> f32 {
    let u1 = rng.random::<f32>().max(1.0e-7);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Metal kernel plumbing.
// ---------------------------------------------------------------------------

static FFI_LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

#[allow(unsafe_code)]
unsafe extern "C" fn ffi_error_handler(msg: *const c_char, _data: *mut c_void) {
    let message = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    if let Ok(mut guard) = FFI_LAST_ERROR.lock() {
        *guard = Some(message);
    }
}

fn ensure_ffi_error_handler() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_set_error_handler(Some(ffi_error_handler), std::ptr::null_mut(), None);
        }
    });
}

struct CachedMetalKernel(mlx_sys::mlx_fast_metal_kernel);

#[allow(unsafe_code)]
unsafe impl Send for CachedMetalKernel {}
#[allow(unsafe_code)]
unsafe impl Sync for CachedMetalKernel {}

impl Drop for CachedMetalKernel {
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.0);
        }
    }
}

static SCORE_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static VALUE_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

const TURBOQUANT_SCORES_KERNEL_SOURCE: &str = r"
constexpr float kQjlScale = 1.2533141373155001f / float(D);
auto pos = thread_position_in_grid.x;
auto head = thread_position_in_grid.y;
if (pos >= T) { return; }
auto kv_head = head / (H / HKV);

auto q_rot_ptr = q_rot + head * D;
auto q_qjl_ptr = q_qjl + head * D;

auto packed_index = kv_head * Capacity + pos;
auto code_ptr = k_codes + packed_index * KBytes;
auto sign_ptr = k_qjl + packed_index * QBytes;

float mse = 0.0f;
for (int j = 0; j < D; ++j) {
  auto bit_index = j * KBits;
  auto byte_index = bit_index / 8;
  auto shift = bit_index % 8;
  uint code = uint(code_ptr[byte_index]) >> shift;
  if (shift + KBits > 8) {
    code |= uint(code_ptr[byte_index + 1]) << (8 - shift);
  }
  code &= ((1u << KBits) - 1u);
  mse += float(q_rot_ptr[j]) * key_centroids[code];
}
float score = mse * float(k_norms[packed_index]);

float residual = 0.0f;
for (int j = 0; j < D; ++j) {
  auto byte_index = j / 8;
  auto shift = j % 8;
  uint sign = (uint(sign_ptr[byte_index]) >> shift) & 1u;
  residual += (sign == 1u ? 1.0f : -1.0f) * float(q_qjl_ptr[j]);
}

scores[head * T + pos] = score + float(k_gammas[packed_index]) * kQjlScale * residual;
";

const TURBOQUANT_VALUES_KERNEL_SOURCE: &str = r"
auto dim = thread_position_in_grid.x;
auto head = thread_position_in_grid.y;
if (dim >= D) { return; }
auto kv_head = head / (H / HKV);

float acc = 0.0f;
for (int pos = 0; pos < T; ++pos) {
  auto packed_index = kv_head * Capacity + pos;
  auto bit_index = dim * VBits;
  auto byte_index = bit_index / 8;
  auto shift = bit_index % 8;
  auto code_ptr = v_codes + packed_index * VBytes;
  uint code = uint(code_ptr[byte_index]) >> shift;
  if (shift + VBits > 8) {
    code |= uint(code_ptr[byte_index + 1]) << (8 - shift);
  }
  code &= ((1u << VBits) - 1u);
  acc += float(weights[head * T + pos]) * value_centroids[code] * float(v_norms[packed_index]);
}
out_rot[head * D + dim] = acc;
";

#[allow(unsafe_code)]
fn create_scores_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 9] = [
        c"q_rot",
        c"q_qjl",
        c"k_codes",
        c"k_norms",
        c"k_qjl",
        c"k_gammas",
        c"key_centroids",
        c"Capacity",
        c"T",
    ];
    let output_names: [&std::ffi::CStr; 1] = [c"scores"];
    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|name| name.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|name| name.as_ptr()).collect();
    let source =
        CString::new(TURBOQUANT_SCORES_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"turboquant_scores".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn create_values_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 6] = [
        c"weights",
        c"v_codes",
        c"v_norms",
        c"value_centroids",
        c"Capacity",
        c"T",
    ];
    let output_names: [&std::ffi::CStr; 1] = [c"out_rot"];
    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|name| name.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|name| name.as_ptr()).collect();
    let source =
        CString::new(TURBOQUANT_VALUES_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"turboquant_values".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_scores_kernel(
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    seq_len: i32,
    key_bits: u8,
    key_code_bytes: i32,
    sign_bytes: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"D".as_ptr(), head_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"H".as_ptr(),
            num_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"HKV".as_ptr(),
            num_kv_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KBits".as_ptr(),
            i32::from(key_bits),
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KBytes".as_ptr(),
            key_code_bytes,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"QBytes".as_ptr(),
            sign_bytes,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, seq_len, num_heads, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);
        let shape = [num_heads, seq_len];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            shape.as_ptr(),
            shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        config
    }
}

#[allow(unsafe_code)]
fn configure_values_kernel(
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    value_bits: u8,
    value_code_bytes: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"D".as_ptr(), head_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"H".as_ptr(),
            num_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"HKV".as_ptr(),
            num_kv_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"VBits".as_ptr(),
            i32::from(value_bits),
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"VBytes".as_ptr(),
            value_code_bytes,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, head_dim, num_heads, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);
        let shape = [num_heads, head_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            shape.as_ptr(),
            shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        config
    }
}

#[allow(unsafe_code)]
fn extract_single_output(
    status: i32,
    outputs_vec: mlx_sys::mlx_vector_array,
    label: &str,
) -> Result<Array, Exception> {
    if status != 0 {
        let message = FFI_LAST_ERROR
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
            .unwrap_or_default();
        if message.is_empty() {
            return Err(Exception::custom(format!("{label} kernel failed")));
        }
        return Err(Exception::custom(format!(
            "{label} kernel failed: {message}"
        )));
    }

    let mut out_ptr = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_vector_array_get(&raw mut out_ptr, outputs_vec, 0);
        Ok(Array::from_ptr(out_ptr))
    }
}
