//! Runtime JIT Metal kernels for Bonsai-Q1 (1-bit affine quantization).
//!
//! Upstream `oxideai/mlx-rs` ships no `bits=1` affine kernels (MLX gates affine
//! quant to `bits >= 2`), so `ops::quantized_matmul`/`ops::dequantize` with
//! `bits=1` fail at runtime with `Unable to load kernel affine_dequantize_*_b_1`.
//!
//! Rather than fork mlx-rs (which forces a full from-source mlx-c rebuild), we
//! add the missing kernels *from this crate* using the runtime JIT facility that
//! mlx-c already exposes (`mlx_fast_metal_kernel_*`) and that `mlx-sys` compiles
//! in. The kernels below are JIT-compiled by Metal at first use and cached by
//! MLX internally per template instantiation. This keeps us on the stock
//! `oxideai/mlx-rs` pin with no extra native recompile.
//!
//! The FFI plumbing (kernel handle wrapper, `Array` <-> `mlx_array`, vector
//! construction, error capture) mirrors the proven `qgemv_4bit` path in
//! [`crate::qwen3_next`]; the kernel math mirrors
//! [`crate::bonsai_q1::PackedQ1Linear::dequant_row_to_fp32`]:
//! `W[r,c] = scale[r, c/G] * bit + bias[r, c/G]`, `bit = (w[r, c/32] >> (c%32)) & 1`.
//! Checkpoints whose affine metadata is symmetric use an empty bias sentinel;
//! their kernels derive `bias = -scale / 2` and never read a bias buffer.

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, Stream, error::Exception};

// ---------------------------------------------------------------------------
// FFI error capture (per-thread, mirrors qwen3_next).
// ---------------------------------------------------------------------------

thread_local! {
    static FFI_LAST_ERROR: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Error handler registered once with MLX to capture error messages on the
/// calling thread.
#[allow(unsafe_code)]
unsafe extern "C" fn ffi_error_handler(msg: *const c_char, _data: *mut c_void) {
    let s = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    FFI_LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(s));
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

fn take_last_error() -> String {
    FFI_LAST_ERROR
        .with(|cell| cell.borrow_mut().take())
        .unwrap_or_else(|| "(no MLX error message captured)".to_owned())
}

// ---------------------------------------------------------------------------
// Cached kernel handle.
// ---------------------------------------------------------------------------

/// Wraps a compiled `mlx_fast_metal_kernel`, freed on drop.
struct CachedMetalKernel(mlx_sys::mlx_fast_metal_kernel);

// SAFETY: the handle is created once and only ever read (passed by value to
// `mlx_fast_metal_kernel_apply`); no interior mutability is shared across threads.
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

/// Number of simdgroups per threadgroup for the fused matvec. More simdgroups
/// help large-K layers (fewer chunk barriers). Overridable for tuning.
fn qmv_nsg(k_dim: i32) -> i32 {
    static OVERRIDE: OnceLock<Option<i32>> = OnceLock::new();
    let ovr = *OVERRIDE.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_QMV_NSG")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|n| matches!(n, 4 | 8 | 16 | 32))
    });
    ovr.unwrap_or(if k_dim > 8192 { 16 } else { 8 })
}

/// Build the vector-of-strings that names kernel inputs/outputs.
#[allow(unsafe_code)]
fn cstr_vec(names: &[&CStr]) -> mlx_sys::mlx_vector_string {
    let ptrs: Vec<*const c_char> = names.iter().map(|s| s.as_ptr()).collect();
    unsafe { mlx_sys::mlx_vector_string_new_data(ptrs.as_ptr().cast_mut(), ptrs.len()) }
}

// ---------------------------------------------------------------------------
// Fused 1-bit quantized matvec (decode hot path).
//
// y = x @ dequant(W).T  for a single token (M = 1).
// Mirrors qgemv_4bit but unpacks 32 1-bit weights per uint32 word.
// One simdgroup per output row; x staged in threadgroup memory; simd_sum reduce.
// ---------------------------------------------------------------------------

const QMV_KERNEL_SOURCE: &str = r"
constexpr int CHUNK = (K <= 8192) ? K : 8192;

threadgroup OutT x_sh[CHUNK];

auto tg = threadgroup_position_in_grid.x;
auto sg = simdgroup_index_in_threadgroup;
auto lane = thread_index_in_simdgroup;
auto tid = thread_index_in_threadgroup;
auto n_sg = simdgroups_per_threadgroup;
uint tg_sz = n_sg * 32u;

int row = tg * int(n_sg) + int(sg);
bool valid = (row < n_param);

float acc = 0.0f;

for (int k_off = 0; k_off < K; k_off += CHUNK) {
    int k_end = min(k_off + CHUNK, K);
    int k_len = k_end - k_off;

    for (uint i = tid; i < uint(k_len); i += tg_sz) {
        x_sh[i] = x[k_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        int wp_off = k_off / 32;
        int wp_end = k_end / 32;
        auto w_row = w + row * KPacked;

        for (int idx = wp_off + int(lane); idx < wp_end; idx += 32) {
            uint packed = w_row[idx];
            int kl = (idx - wp_off) * 32;

            float dot_val = 0.0f;
            float sum_x = 0.0f;
            for (uint j = 0u; j < 32u; ++j) {
                float xv = float(x_sh[kl + int(j)]);
                float bit = float((packed >> j) & 1u);
                dot_val += bit * xv;
                sum_x += xv;
            }

            int g = idx * 32 / GroupSize;
            float s_val = float(sc[row * NumGroups + g]);
            float b_val = Symmetric ? (-0.5f * s_val) : float(bi[row * NumGroups + g]);
            acc += s_val * dot_val + b_val * sum_x;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (valid) {
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = OutT(acc);
    }
}
";

#[allow(unsafe_code)]
fn create_qmv_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi", c"x", c"n_param"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(QMV_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q1_qmv".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,  // raw pointer arithmetic requires row-contiguous inputs
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_qmv_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
    symmetric: bool,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"OutT".as_ptr(),
            out_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"K".as_ptr(), k_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"GroupSize".as_ptr(),
            group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KPacked".as_ptr(),
            k_dim / 32,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Symmetric".as_ptr(),
            i32::from(symmetric),
        );

        let nsg = qmv_nsg(k_dim);
        let n_tgs = (n_rows + nsg - 1) / nsg;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tgs * 32, nsg, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1);

        let y_shape = [1, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            out_dtype,
        );
        config
    }
}

/// Original per-row 1-bit matvec: one simdgroup computes one output row, with
/// `x` staged in threadgroup memory. Kept as the A/B baseline (selected when
/// `HIGGS_BONSAI_QMV_KERNEL=legacy`). See [`bonsai_q1_qmv`] for the dispatcher.
#[allow(unsafe_code)]
pub fn bonsai_q1_qmv_legacy(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let x_shape = x.shape();
    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_qmv: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_qmv: weight has no columns"))?;
    let k_dim = k_packed * 32; // 32 one-bit weights per uint32 word

    let x_flat = x.reshape(&[k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let symmetric = biases.size() == 0;
    // FastMetal still binds the affine input signature. Reuse the scale array
    // as a harmless dummy; the `Symmetric` template constant removes the bias
    // load from the compiled kernel.
    let b_flat = if symmetric {
        s_flat.clone()
    } else {
        biases.flatten(None, None)?
    };

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = QMV_KERNEL.get_or_init(|| CachedMetalKernel(create_qmv_kernel()));
    let config = configure_qmv_kernel(out_dtype, n_rows, k_dim, group_size, symmetric);

    let n_scalar = unsafe { mlx_sys::mlx_array_new_int(n_rows) };
    let input_ptrs = [
        w_flat.as_ptr(),
        s_flat.as_ptr(),
        b_flat.as_ptr(),
        x_flat.as_ptr(),
        n_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        Err(Exception::custom(format!(
            "bonsai_q1_qmv failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        let y = unsafe { Array::from_ptr(y_ptr) };
        let trim_to = x_shape.len().saturating_sub(1);
        let mut out_shape = x_shape
            .get(..trim_to)
            .ok_or_else(|| Exception::custom("bonsai_q1_qmv: x_shape too small"))?
            .to_vec();
        out_shape.push(n_rows);
        y.reshape(&out_shape)
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }
    result
}

static QMV_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static FAST_QMV_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static TG_LUT4_CONTRACT_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static TG_LUT4_CONTRACT_M5_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

/// Simdgroups per threadgroup for the `qmv_fast`-class kernel. Each simdgroup
/// computes `RESULTS_PER_SIMDGROUP` (= 4) output rows. Tunable via
/// `HIGGS_BONSAI_FAST_NSG` (Phase-2 sweep); MLX's reference uses 2.
fn fast_qmv_nsg() -> i32 {
    static OVERRIDE: OnceLock<i32> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_FAST_NSG")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|n| matches!(n, 1 | 2 | 4 | 8))
            .unwrap_or(2)
    })
}

/// Whether the fast QMV kernel may specialize away output-row bounds checks.
///
/// This is deliberately opt-in while the specialization is benchmarked on
/// real Bonsai verifier shapes. The unaligned kernel remains the fallback for
/// shapes that do not fill a complete threadgroup's output-row tile.
fn use_aligned_fast_qmv() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_ALIGNED_FAST_QMV").is_ok_and(|value| value == "1")
    })
}

const fn fast_qmv_has_aligned_rows(n_rows: i32, nsg: i32, prefer_aligned: bool) -> bool {
    const RESULTS_PER_SIMDGROUP: i32 = 4;
    prefer_aligned && n_rows > 0 && n_rows % (nsg * RESULTS_PER_SIMDGROUP) == 0
}

/// Whether to route the decode matvec through the `qmv_fast`-class kernel.
/// It is the **default** (measured 2.3× faster on Bonsai-8B decode and bit-exact
/// vs the CPU reference); opt back to the original per-row kernel with
/// `HIGGS_BONSAI_QMV_KERNEL=legacy`.
fn use_fast_qmv() -> bool {
    static FAST: OnceLock<bool> = OnceLock::new();
    *FAST.get_or_init(|| {
        !std::env::var("HIGGS_BONSAI_QMV_KERNEL").is_ok_and(|v| v.eq_ignore_ascii_case("legacy"))
    })
}

/// Fused 1-bit quantized matvec: `y = x @ dequant(weight).T` for a single token.
///
/// `x` must hold exactly `in_features` elements (M = 1). `weight` is the packed
/// `[out_features, in_features/32]` uint32 matrix; `scales`/`biases` are
/// `[out_features, in_features/group_size]`. Output dtype matches `x`.
///
/// Dispatches to the `qmv_fast`-class kernel ([`bonsai_q1_qmv_fast`]) by
/// default; set `HIGGS_BONSAI_QMV_KERNEL=legacy` to force the per-row kernel.
pub fn bonsai_q1_qmv(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    if use_fast_qmv() {
        bonsai_q1_qmv_fast(x, weight, scales, biases, group_size)
    } else {
        bonsai_q1_qmv_legacy(x, weight, scales, biases, group_size)
    }
}

// ---------------------------------------------------------------------------
// `qmv_fast`-class 1-bit narrow matrix multiply (decode / verify hot path).
//
// Ports MLX/PrismML `qmv_fast` tiling onto our uint32 packing: each simdgroup
// computes RESULTS_PER_SIMDGROUP (4) output rows for one input row; the grid's
// z dimension covers narrow M > 1 verifier batches without materializing the
// dense weight matrix. Each lane holds VPT (32) input values in registers and
// reuses them across all 4 output rows. Keeping one packed word per lane
// reduces register pressure and raises occupancy for 1-bit weights. The bits=1
// affine math is identical to the legacy kernel —
// `scale * sum(bit*x) + bias * sum(x)` — only the data movement differs.
// Group scales/biases are per-lane (a lane's 32 values lie in one 128-wide
// group); per-row partials are simd_sum-reduced.
// ---------------------------------------------------------------------------

const FAST_QMV_KERNEL_SOURCE: &str = r"
constexpr int VPT = 32;          // values_per_thread (one packed word per lane)
constexpr int RPS = 4;           // results_per_simdgroup
constexpr int WPT = VPT / 32;    // packed uint32 words per thread (1)
constexpr int BLK = VPT * 32;    // block_size = 1024

uint tgx = threadgroup_position_in_grid.x;
uint sg  = simdgroup_index_in_threadgroup;
uint lid = thread_index_in_simdgroup;
uint nsg = simdgroups_per_threadgroup;
uint batch = threadgroup_position_in_grid.z;

int out_row = int(tgx) * (int(nsg) * RPS) + int(sg) * RPS;
auto x_row = x + int(batch) * K;

float xt[VPT];
float result[RPS];
for (int r = 0; r < RPS; ++r) { result[r] = 0.0f; }

int aligned_end = (K / BLK) * BLK;

// Main loop: full 1024-element blocks (covers every real Bonsai layer, since
// all K are multiples of 2048).
for (int k = 0; k < aligned_end; k += BLK) {
    int xbase = k + int(lid) * VPT;
    float sum = 0.0f;
    for (int i = 0; i < VPT; ++i) { float v = float(x_row[xbase + i]); xt[i] = v; sum += v; }

    int wcol = (k / 32) + int(lid) * WPT;
    int g = xbase / GroupSize;   // all VPT values fall in one group

    for (int r = 0; r < RPS; ++r) {
        int row = out_row + r;
        if constexpr (!AlignedN) {
            if (row >= n_param) { continue; }
        }
        float accum = 0.0f;
        for (int wp = 0; wp < WPT; ++wp) {
            uint packed = w[row * KPacked + wcol + wp];
            int xo = wp * 32;
            for (int bk = 0; bk < 4; ++bk) {
                uint wb = (packed >> (uint(bk) * 8u)) & 0xFFu;
                int b = xo + bk * 8;
                accum += select(0.0f, xt[b + 0], (wb & 0x01u) != 0u);
                accum += select(0.0f, xt[b + 1], (wb & 0x02u) != 0u);
                accum += select(0.0f, xt[b + 2], (wb & 0x04u) != 0u);
                accum += select(0.0f, xt[b + 3], (wb & 0x08u) != 0u);
                accum += select(0.0f, xt[b + 4], (wb & 0x10u) != 0u);
                accum += select(0.0f, xt[b + 5], (wb & 0x20u) != 0u);
                accum += select(0.0f, xt[b + 6], (wb & 0x40u) != 0u);
                accum += select(0.0f, xt[b + 7], (wb & 0x80u) != 0u);
            }
        }
        float s_val = float(sc[row * NumGroups + g]);
        float b_val = Symmetric ? (-0.5f * s_val) : float(bi[row * NumGroups + g]);
        result[r] += s_val * accum + b_val * sum;
    }
}

// Tail: only exercised by tests with K < 2048 or K % 2048 != 0.
if (aligned_end < K) {
    int xbase = aligned_end + int(lid) * VPT;
    bool in_bounds = xbase < K;
    float sum = 0.0f;
    for (int i = 0; i < VPT; ++i) {
        float v = (in_bounds && (xbase + i) < K) ? float(x_row[xbase + i]) : 0.0f;
        xt[i] = v;
        sum += v;
    }
    int wcol = (aligned_end / 32) + int(lid) * WPT;
    int g = in_bounds ? (xbase / GroupSize) : 0;
    for (int r = 0; r < RPS; ++r) {
        int row = out_row + r;
        if constexpr (AlignedN) {
            if (!in_bounds) { continue; }
        } else {
            if (row >= n_param || !in_bounds) { continue; }
        }
        float accum = 0.0f;
        for (int wp = 0; wp < WPT; ++wp) {
            int widx = wcol + wp;
            if (widx >= KPacked) { continue; }
            uint packed = w[row * KPacked + widx];
            int xo = wp * 32;
            for (int bk = 0; bk < 4; ++bk) {
                uint wb = (packed >> (uint(bk) * 8u)) & 0xFFu;
                int b = xo + bk * 8;
                accum += select(0.0f, xt[b + 0], (wb & 0x01u) != 0u);
                accum += select(0.0f, xt[b + 1], (wb & 0x02u) != 0u);
                accum += select(0.0f, xt[b + 2], (wb & 0x04u) != 0u);
                accum += select(0.0f, xt[b + 3], (wb & 0x08u) != 0u);
                accum += select(0.0f, xt[b + 4], (wb & 0x10u) != 0u);
                accum += select(0.0f, xt[b + 5], (wb & 0x20u) != 0u);
                accum += select(0.0f, xt[b + 6], (wb & 0x40u) != 0u);
                accum += select(0.0f, xt[b + 7], (wb & 0x80u) != 0u);
            }
        }
        float s_val = float(sc[row * NumGroups + g]);
        float b_val = Symmetric ? (-0.5f * s_val) : float(bi[row * NumGroups + g]);
        result[r] += s_val * accum + b_val * sum;
    }
}

for (int r = 0; r < RPS; ++r) {
    int row = out_row + r;
    float v = simd_sum(result[r]);
    if (lid == 0u) {
        if constexpr (AlignedN) {
            y[int(batch) * n_param + row] = OutT(v);
        } else if (row < n_param) {
            y[int(batch) * n_param + row] = OutT(v);
        }
    }
}
";

#[allow(unsafe_code)]
fn create_fast_qmv_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi", c"x", c"n_param"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(FAST_QMV_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q1_qmv_fast".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,  // raw pointer arithmetic requires row-contiguous inputs
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_fast_qmv_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    m_rows: i32,
    k_dim: i32,
    group_size: i32,
    symmetric: bool,
    prefer_aligned: bool,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"OutT".as_ptr(),
            out_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"K".as_ptr(), k_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"GroupSize".as_ptr(),
            group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KPacked".as_ptr(),
            k_dim / 32,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Symmetric".as_ptr(),
            i32::from(symmetric),
        );

        // Each simdgroup computes 4 rows; nsg simdgroups per threadgroup.
        let nsg = fast_qmv_nsg();
        let rows_per_tg = nsg * 4;
        let aligned_n = fast_qmv_has_aligned_rows(n_rows, nsg, prefer_aligned);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"AlignedN".as_ptr(),
            i32::from(aligned_n),
        );
        let n_tgs = (n_rows + rows_per_tg - 1) / rows_per_tg;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tgs * 32, nsg, m_rows);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1);

        let y_shape = [m_rows, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            out_dtype,
        );
        config
    }
}

/// `qmv_fast`-class variant of [`bonsai_q1_qmv_legacy`]. Same inputs/outputs and
/// bit-exact result; faster tiling. Set `HIGGS_BONSAI_ALIGNED_FAST_QMV=1` to
/// specialize away row bounds checks when N fills complete threadgroup tiles.
/// Unaligned shapes retain the guarded kernel. See [`bonsai_q1_qmv`] for
/// dispatch.
#[allow(unsafe_code)]
pub fn bonsai_q1_qmv_fast(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    bonsai_q1_qmv_fast_impl(
        x,
        weight,
        scales,
        biases,
        group_size,
        use_aligned_fast_qmv(),
    )
}

#[allow(unsafe_code)]
fn bonsai_q1_qmv_fast_impl(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    prefer_aligned: bool,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let x_shape = x.shape();
    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_qmv_fast: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_qmv_fast: weight has no columns"))?;
    let k_dim = k_packed * 32;
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();

    let x_flat = x.reshape(&[m_rows, k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let symmetric = biases.size() == 0;
    let b_flat = if symmetric {
        s_flat.clone()
    } else {
        biases.flatten(None, None)?
    };

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = FAST_QMV_KERNEL.get_or_init(|| CachedMetalKernel(create_fast_qmv_kernel()));
    let config = configure_fast_qmv_kernel(
        out_dtype,
        n_rows,
        m_rows,
        k_dim,
        group_size,
        symmetric,
        prefer_aligned,
    );

    let n_scalar = unsafe { mlx_sys::mlx_array_new_int(n_rows) };
    let input_ptrs = [
        w_flat.as_ptr(),
        s_flat.as_ptr(),
        b_flat.as_ptr(),
        x_flat.as_ptr(),
        n_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        Err(Exception::custom(format!(
            "bonsai_q1_qmv_fast failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        let y = unsafe { Array::from_ptr(y_ptr) };
        let trim_to = x_shape.len().saturating_sub(1);
        let mut out_shape = x_shape
            .get(..trim_to)
            .ok_or_else(|| Exception::custom("bonsai_q1_qmv_fast: x_shape too small"))?
            .to_vec();
        out_shape.push(n_rows);
        y.reshape(&out_shape)
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }
    result
}

/// Packed affine Q1 matrix multiply for narrow verifier batches.
///
/// This shares the decode-optimized kernel with [`bonsai_q1_qmv_fast`] but
/// dispatches one grid slice per flattened input row. It intentionally targets
/// small sequence lengths: weights stay packed and resident, avoiding the very
/// large temporary produced by full dequantization.
pub fn bonsai_q1_qmm(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    bonsai_q1_qmv_fast(x, weight, scales, biases, group_size)
}

// ---------------------------------------------------------------------------
// Experimental symmetric-Q1 threadgroup-local LUT4 path.
//
// This is deliberately exposed through a typed row4 container instead of raw
// arrays. The kernel's pointer arithmetic requires the physical order encoded
// by the shapes below; accepting a flat array merely because its byte count
// matches would silently reinterpret ordinary row-major checkpoint weights.
// ---------------------------------------------------------------------------

/// One-time row4 materialization consumed by the TG-LUT4 kernels.
///
/// Physical shapes are `[N/4, K/128, 4 words, 4 output lanes]` for packed Q1
/// bits and `[N/4, K/128, 4 output lanes]` for scales. Fields stay private so
/// callers cannot construct this contract from ambiguous flat buffers.
#[derive(Debug, Clone)]
pub(crate) struct BonsaiQ1Row4 {
    weights: Array,
    scales: Array,
    n_rows: i32,
    k_dim: i32,
    cached_bytes: usize,
}

impl BonsaiQ1Row4 {
    /// Transform canonical checkpoint arrays `[N,K/32]` and `[N,K/128]` into
    /// the row4 layout entirely through MLX. `mlx_contiguous(..., false)` is
    /// essential: allowing column-major storage could preserve the transposed
    /// view and make the kernel's flattened indexing incorrect.
    pub(crate) fn from_row_major(weight: &Array, scales: &Array) -> Result<Self, Exception> {
        let [n_rows, k_packed] = *weight.shape() else {
            return Err(Exception::custom(
                "BonsaiQ1Row4: canonical weight must have shape [N,K/32]",
            ));
        };
        let [scale_rows, groups] = *scales.shape() else {
            return Err(Exception::custom(
                "BonsaiQ1Row4: canonical scales must have shape [N,K/128]",
            ));
        };
        if weight.dtype() != Dtype::Uint32
            || !matches!(scales.dtype(), Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err(Exception::custom(format!(
                "BonsaiQ1Row4: expected Uint32 bits and Float16/Bfloat16 scales, got {:?}/{:?}",
                weight.dtype(),
                scales.dtype()
            )));
        }
        let k_dim = k_packed
            .checked_mul(32)
            .ok_or_else(|| Exception::custom("BonsaiQ1Row4: K overflow"))?;
        if n_rows <= 0
            || k_dim <= 0
            || n_rows % 4 != 0
            || k_dim % 128 != 0
            || scale_rows != n_rows
            || groups != k_dim / 128
        {
            return Err(Exception::custom(format!(
                "BonsaiQ1Row4: incompatible canonical shapes {:?}/{:?}; require N%4=0 and K%128=0",
                weight.shape(),
                scales.shape()
            )));
        }

        let weights_reshaped = weight.reshape(&[n_rows / 4, 4, groups, 4])?;
        let weights_view = weights_reshaped.transpose_axes(&[0, 2, 3, 1])?;
        let scales_reshaped = scales.reshape(&[n_rows / 4, 4, groups])?;
        let scales_view = scales_reshaped.transpose_axes(&[0, 2, 1])?;
        let weights = row_contiguous_copy(&weights_view)?;
        let packed_scales = row_contiguous_copy(&scales_view)?;
        // Force the two device copies now. Keeping only lazy transpose graphs
        // would repeat or defer the multi-GiB model-wide materialization into
        // timed decode.
        crate::mlx_exec::eval([&weights, &packed_scales])?;
        Self::from_packed_parts(weights, packed_scales, n_rows, k_dim)
    }

    fn from_packed_parts(
        weights: Array,
        scales: Array,
        n_rows: i32,
        k_dim: i32,
    ) -> Result<Self, Exception> {
        let expected_weights = [n_rows / 4, k_dim / 128, 4, 4];
        let expected_scales = [n_rows / 4, k_dim / 128, 4];
        if n_rows <= 0
            || k_dim <= 0
            || n_rows % 4 != 0
            || k_dim % 128 != 0
            || weights.shape() != expected_weights
            || scales.shape() != expected_scales
            || weights.dtype() != Dtype::Uint32
            || !matches!(scales.dtype(), Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err(Exception::custom(format!(
                "BonsaiQ1Row4: invalid packed contract bits={:?}/{:?} scales={:?}/{:?}; expected {:?} Uint32 and {:?} Float16/Bfloat16",
                weights.shape(),
                weights.dtype(),
                scales.shape(),
                scales.dtype(),
                expected_weights,
                expected_scales
            )));
        }
        if !array_is_row_contiguous(&weights)? || !array_is_row_contiguous(&scales)? {
            return Err(Exception::custom(
                "BonsaiQ1Row4: packed arrays must be physically row-contiguous",
            ));
        }
        let cached_bytes = weights.nbytes().saturating_add(scales.nbytes());
        Ok(Self {
            weights,
            scales,
            n_rows,
            k_dim,
            cached_bytes,
        })
    }

    pub(crate) const fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    pub(crate) fn accepts_input(&self, input: &Array) -> bool {
        if !matches!(input.dtype(), Dtype::Float16 | Dtype::Bfloat16)
            || input.shape().last().copied() != Some(self.k_dim)
        {
            return false;
        }
        let rows: i32 = input
            .shape()
            .iter()
            .take(input.shape().len().saturating_sub(1))
            .product();
        (1..=5).contains(&rows)
    }
}

#[allow(unsafe_code)]
fn row_contiguous_copy(array: &Array) -> Result<Array, Exception> {
    ensure_ffi_error_handler();
    let stream = Stream::task_local_or_default();
    let mut output = unsafe { mlx_sys::mlx_array_new() };
    let status = unsafe {
        mlx_sys::mlx_contiguous(
            &raw mut output,
            array.as_ptr(),
            false, // never preserve a column-major transposed view
            stream.as_ptr(),
        )
    };
    if status != 0 {
        unsafe { mlx_sys::mlx_array_free(output) };
        Err(Exception::custom(format!(
            "mlx_contiguous row-major copy failed: {}",
            take_last_error()
        )))
    } else {
        Ok(unsafe { Array::from_ptr(output) })
    }
}

#[allow(unsafe_code)]
fn array_is_row_contiguous(array: &Array) -> Result<bool, Exception> {
    ensure_ffi_error_handler();
    let mut result = false;
    let status = unsafe { mlx_sys::_mlx_array_is_row_contiguous(&raw mut result, array.as_ptr()) };
    if status == 0 {
        Ok(result)
    } else {
        Err(Exception::custom(format!(
            "MLX row-contiguous query failed: {}",
            take_last_error()
        )))
    }
}

const TG_LUT4_CONTRACT_KERNEL_SOURCE: &str = r"
constexpr int WG = 256;
constexpr int NTILE = 256;
constexpr int MTILE = 4;
threadgroup half lut[2048];

uint tid = thread_index_in_threadgroup;
uint n = threadgroup_position_in_grid.x * uint(NTILE) + tid;
uint mbase = threadgroup_position_in_grid.z * uint(MTILE);
float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;

for (int g = 0; g < NumGroups; ++g) {
    if (tid < 128u) {
        int mlocal = int(tid) / 32;
        int nibble = int(tid) & 31;
        int m = int(mbase) + mlocal;
        int kbase = g * 128 + nibble * 4;
        float x0 = 0.0f;
        float x1 = 0.0f;
        float x2 = 0.0f;
        float x3 = 0.0f;
        if (m < MRows) {
            int xb = m * K + kbase;
            x0 = float(x[xb + 0]); x1 = float(x[xb + 1]);
            x2 = float(x[xb + 2]); x3 = float(x[xb + 3]);
        }
        float xy = x0 + x1;
        float xz = x0 + x2;
        float yz = x1 + x2;
        float xyz = xy + x2;
        float c = 0.5f * (x0 + x1 + x2 + x3);
        int base = (mlocal * 32 + nibble) * 16;
        lut[base + 0] = half(-c);
        lut[base + 1] = half(x0 - c);
        lut[base + 2] = half(x1 - c);
        lut[base + 3] = half(xy - c);
        lut[base + 4] = half(x2 - c);
        lut[base + 5] = half(xz - c);
        lut[base + 6] = half(yz - c);
        lut[base + 7] = half(xyz - c);
        lut[base + 8] = half(x3 - c);
        lut[base + 9] = half(x0 + x3 - c);
        lut[base + 10] = half(x1 + x3 - c);
        lut[base + 11] = half(xy + x3 - c);
        lut[base + 12] = half(x2 + x3 - c);
        lut[base + 13] = half(xz + x3 - c);
        lut[base + 14] = half(yz + x3 - c);
        lut[base + 15] = half(xyz + x3 - c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n < uint(NRows)) {
        int row_tile = int(n) / 4;
        int row_lane = int(n) & 3;
        int group_base = (row_tile * NumGroups + g) * 4;
        float qa0 = 0.0f;
        float qa1 = 0.0f;
        float qa2 = 0.0f;
        float qa3 = 0.0f;
#pragma clang loop unroll(full)
        for (int word = 0; word < 4; ++word) {
            uint packed = w[(group_base + word) * 4 + row_lane];
#pragma clang loop unroll(full)
            for (int ni = 0; ni < 8; ++ni) {
                uint mask = (packed >> (uint(ni) * 4u)) & 0xFu;
                int li = (word * 8 + ni) * 16 + int(mask);
                qa0 += float(lut[li]);
                qa1 += float(lut[512 + li]);
                qa2 += float(lut[1024 + li]);
                qa3 += float(lut[1536 + li]);
            }
        }
        float scale = float(sc[group_base + row_lane]);
        acc0 += scale * qa0;
        acc1 += scale * qa1;
        acc2 += scale * qa2;
        acc3 += scale * qa3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (n < uint(NRows)) {
    if (mbase + 0u < uint(MRows)) { y[int(mbase + 0u) * NRows + int(n)] = OutT(acc0); }
    if (mbase + 1u < uint(MRows)) { y[int(mbase + 1u) * NRows + int(n)] = OutT(acc1); }
    if (mbase + 2u < uint(MRows)) { y[int(mbase + 2u) * NRows + int(n)] = OutT(acc2); }
    if (mbase + 3u < uint(MRows)) { y[int(mbase + 3u) * NRows + int(n)] = OutT(acc3); }
}
";

// The five activation rows are independent of the four adjacent output rows
// encoded by row4 packing. One scale and one packed-weight pass feed all five
// accumulators.
const TG_LUT4_CONTRACT_M5_KERNEL_SOURCE: &str = r"
threadgroup half lut[2560];

uint tid = thread_index_in_threadgroup;
uint n = threadgroup_position_in_grid.x * uint(NTILE) + tid;
float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;
float acc4 = 0.0f;

for (int g = 0; g < NumGroups; ++g) {
    for (uint build = tid; build < 160u; build += uint(WG)) {
        int mlocal = int(build) / 32;
        int nibble = int(build) & 31;
        int kbase = g * 128 + nibble * 4;
        int xb = mlocal * K + kbase;
        float x0 = float(x[xb + 0]);
        float x1 = float(x[xb + 1]);
        float x2 = float(x[xb + 2]);
        float x3 = float(x[xb + 3]);
        float xy = x0 + x1;
        float xz = x0 + x2;
        float yz = x1 + x2;
        float xyz = xy + x2;
        float c = 0.5f * (x0 + x1 + x2 + x3);
        int base = (mlocal * 32 + nibble) * 16;
        lut[base + 0] = half(-c);
        lut[base + 1] = half(x0 - c);
        lut[base + 2] = half(x1 - c);
        lut[base + 3] = half(xy - c);
        lut[base + 4] = half(x2 - c);
        lut[base + 5] = half(xz - c);
        lut[base + 6] = half(yz - c);
        lut[base + 7] = half(xyz - c);
        lut[base + 8] = half(x3 - c);
        lut[base + 9] = half(x0 + x3 - c);
        lut[base + 10] = half(x1 + x3 - c);
        lut[base + 11] = half(xy + x3 - c);
        lut[base + 12] = half(x2 + x3 - c);
        lut[base + 13] = half(xz + x3 - c);
        lut[base + 14] = half(yz + x3 - c);
        lut[base + 15] = half(xyz + x3 - c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n < uint(NRows)) {
        int row_tile = int(n) / 4;
        int row_lane = int(n) & 3;
        int group_base = (row_tile * NumGroups + g) * 4;
        float qa0 = 0.0f;
        float qa1 = 0.0f;
        float qa2 = 0.0f;
        float qa3 = 0.0f;
        float qa4 = 0.0f;
#pragma clang loop unroll(full)
        for (int word = 0; word < 4; ++word) {
            uint packed = w[(group_base + word) * 4 + row_lane];
#pragma clang loop unroll(full)
            for (int ni = 0; ni < 8; ++ni) {
                uint mask = (packed >> (uint(ni) * 4u)) & 0xFu;
                int li = (word * 8 + ni) * 16 + int(mask);
                qa0 += float(lut[li]);
                qa1 += float(lut[512 + li]);
                qa2 += float(lut[1024 + li]);
                qa3 += float(lut[1536 + li]);
                qa4 += float(lut[2048 + li]);
            }
        }
        float scale = float(sc[group_base + row_lane]);
        acc0 += scale * qa0;
        acc1 += scale * qa1;
        acc2 += scale * qa2;
        acc3 += scale * qa3;
        acc4 += scale * qa4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (n < uint(NRows)) {
    y[int(n)] = OutT(acc0);
    y[NRows + int(n)] = OutT(acc1);
    y[2 * NRows + int(n)] = OutT(acc2);
    y[3 * NRows + int(n)] = OutT(acc3);
    y[4 * NRows + int(n)] = OutT(acc4);
}
";

#[allow(unsafe_code)]
fn create_tg_lut4_kernel(native_m5: bool) -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"x"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(if native_m5 {
        TG_LUT4_CONTRACT_M5_KERNEL_SOURCE
    } else {
        TG_LUT4_CONTRACT_KERNEL_SOURCE
    })
    .unwrap_or_default();
    let name = if native_m5 {
        c"higgs_bonsai_q1_tg_lut4_contract_m5"
    } else {
        c"higgs_bonsai_q1_tg_lut4_contract"
    };
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
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

const TG_LUT4_NATIVE_M5_WGS: [i32; 5] = [128, 160, 192, 224, 256];

fn tg_lut4_native_m5_wg() -> i32 {
    static WG: OnceLock<i32> = OnceLock::new();
    *WG.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_TG_LUT4_M5_WG")
            .ok()
            .and_then(|raw| raw.parse::<i32>().ok())
            .filter(|wg| TG_LUT4_NATIVE_M5_WGS.contains(wg))
            .unwrap_or(256)
    })
}

#[allow(unsafe_code)]
fn configure_tg_lut4_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    packed: &BonsaiQ1Row4,
    m_rows: i32,
    native_m5_wg: Option<i32>,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"OutT".as_ptr(),
            out_dtype,
        );
        for (name, value) in [
            (c"NRows", packed.n_rows),
            (c"MRows", m_rows),
            (c"K", packed.k_dim),
            (c"NumGroups", packed.k_dim / 128),
        ] {
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config,
                name.as_ptr(),
                value,
            );
        }
        let n_tile = native_m5_wg.unwrap_or(256);
        if let Some(wg) = native_m5_wg {
            for name in [c"WG", c"NTILE"] {
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    config,
                    name.as_ptr(),
                    wg,
                );
            }
        }
        let n_tiles = (packed.n_rows + n_tile - 1) / n_tile;
        let m_tiles = if native_m5_wg.is_some() {
            1
        } else {
            (m_rows + 3) / 4
        };
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tiles * n_tile, 1, m_tiles);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, n_tile, 1, 1);
        let output_shape = [m_rows, packed.n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            output_shape.as_ptr(),
            output_shape.len(),
            out_dtype,
        );
        config
    }
}

/// Apply the faithful F16-LUT/F32-accumulation plan. M=1..4 use the scalar
/// contract kernel; exactly M=5 uses the native five-accumulator kernel.
#[allow(unsafe_code)]
pub(crate) fn bonsai_q1_tg_lut4_qmm(x: &Array, packed: &BonsaiQ1Row4) -> Result<Array, Exception> {
    ensure_ffi_error_handler();
    if !matches!(x.dtype(), Dtype::Float16 | Dtype::Bfloat16) {
        return Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_qmm: input must be Float16/Bfloat16, got {:?}",
            x.dtype()
        )));
    }
    let input_shape = x.shape();
    if input_shape.last().copied() != Some(packed.k_dim) {
        return Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_qmm: input last dim {:?}, expected {}",
            input_shape.last(),
            packed.k_dim
        )));
    }
    let m_rows: i32 = input_shape
        .iter()
        .take(input_shape.len().saturating_sub(1))
        .product();
    if !(1..=5).contains(&m_rows) {
        return Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_qmm: requires 1..=5 flattened rows, got {m_rows}"
        )));
    }
    let native_m5_wg = (m_rows == 5).then(tg_lut4_native_m5_wg);
    let cached = if native_m5_wg.is_some() {
        TG_LUT4_CONTRACT_M5_KERNEL.get_or_init(|| CachedMetalKernel(create_tg_lut4_kernel(true)))
    } else {
        TG_LUT4_CONTRACT_KERNEL.get_or_init(|| CachedMetalKernel(create_tg_lut4_kernel(false)))
    };
    let config = configure_tg_lut4_kernel(
        unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) },
        packed,
        m_rows,
        native_m5_wg,
    );
    let x_flat = x.reshape(&[m_rows * packed.k_dim])?;
    let input_ptrs = [
        packed.weights.as_ptr(),
        packed.scales.as_ptr(),
        x_flat.as_ptr(),
    ];
    let inputs =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };
    let mut outputs = unsafe { mlx_sys::mlx_vector_array_new() };
    let stream = Stream::task_local_or_default();
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs,
            cached.0,
            inputs,
            config,
            stream.as_ptr(),
        )
    };
    let result = if status != 0 {
        Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_qmm failed: {}",
            take_last_error()
        )))
    } else {
        let mut output = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut output, outputs, 0) };
        let output = unsafe { Array::from_ptr(output) };
        let mut output_shape = input_shape
            .get(..input_shape.len().saturating_sub(1))
            .ok_or_else(|| Exception::custom("bonsai_q1_tg_lut4_qmm: invalid input shape"))?
            .to_vec();
        output_shape.push(packed.n_rows);
        output.reshape(&output_shape)
    };
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
    }
    result
}

// ---------------------------------------------------------------------------
// 1-bit dequantize to dense (embedding gather + prefill matmul path).
//
// wd[n, c] = scales[n, c/G] * bit(w[n, c/32], c%32) + biases[n, c/G].
// One thread per packed uint32 word (writes 32 dense outputs).
// ---------------------------------------------------------------------------

const DEQUANT_KERNEL_SOURCE: &str = r"
uint gid = thread_position_in_grid.x;
if (gid >= uint(NWords)) { return; }

uint n = gid / uint(KPacked);
uint idx = gid % uint(KPacked);
uint packed = w[gid];

int g = int(idx) * 32 / GroupSize;
float s_val = float(sc[n * uint(NumGroups) + uint(g)]);
float b_val = Symmetric ? (-0.5f * s_val) : float(bi[n * uint(NumGroups) + uint(g)]);

uint base = n * uint(K) + idx * 32u;
for (uint j = 0u; j < 32u; ++j) {
    float bit = float((packed >> j) & 1u);
    wd[base + j] = OutT(s_val * bit + b_val);
}
";

#[allow(unsafe_code)]
fn create_dequant_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi"]);
    let out_vec = cstr_vec(&[c"wd"]);
    let source = CString::new(DEQUANT_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q1_dequant".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            false,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_dequant_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
    symmetric: bool,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    let k_packed = k_dim / 32;
    let n_words = n_rows * k_packed;
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"OutT".as_ptr(),
            out_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"K".as_ptr(), k_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KPacked".as_ptr(),
            k_packed,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"GroupSize".as_ptr(),
            group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NWords".as_ptr(),
            n_words,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Symmetric".as_ptr(),
            i32::from(symmetric),
        );

        let tg: i32 = 256;
        let grid = ((n_words + tg - 1) / tg) * tg;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, grid, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1);

        let wd_shape = [n_rows, k_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            wd_shape.as_ptr(),
            wd_shape.len(),
            out_dtype,
        );
        config
    }
}

/// Dequantize a packed 1-bit matrix to a dense `[out_features, in_features]`
/// array (dtype matches `scales`). Used for embedding gather and the prefill
/// (M > 1) matmul path.
#[allow(unsafe_code)]
pub fn bonsai_q1_dequant(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_dequant: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_dequant: weight has no columns"))?;
    let k_dim = k_packed * 32;

    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let symmetric = biases.size() == 0;
    let b_flat = if symmetric {
        s_flat.clone()
    } else {
        biases.flatten(None, None)?
    };

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(scales.as_ptr()) };

    let cached = DEQUANT_KERNEL.get_or_init(|| CachedMetalKernel(create_dequant_kernel()));
    let config = configure_dequant_kernel(out_dtype, n_rows, k_dim, group_size, symmetric);

    let input_ptrs = [w_flat.as_ptr(), s_flat.as_ptr(), b_flat.as_ptr()];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        Err(Exception::custom(format!(
            "bonsai_q1_dequant failed: {}",
            take_last_error()
        )))
    } else {
        let mut wd_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut wd_ptr, outputs_vec, 0) };
        Ok(unsafe { Array::from_ptr(wd_ptr) })
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
    }
    result
}

static DEQUANT_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::print_stdout,
    clippy::shadow_reuse,
    clippy::too_many_lines
)]
mod tests {
    use super::*;
    use crate::mlx_exec::eval;
    use mlx_rs::Dtype;

    const GROUP_SIZE: i32 = 128;

    fn patterned_weights(n: i32, k: i32, dtype: Dtype, symmetric: bool) -> (Array, Array, Array) {
        assert_eq!(k % GROUP_SIZE, 0);
        let packed = (0..n * k / 32)
            .map(|index| {
                let shift = u32::try_from((index * 7 + 3).rem_euclid(31)).unwrap();
                0x963C_A5F0_u32.rotate_left(shift)
            })
            .collect::<Vec<_>>();
        let scale_values = (0..n * k / GROUP_SIZE)
            .map(|index| ((index % 7) as f32).mul_add(0.031_25, 0.125))
            .collect::<Vec<_>>();
        let bias_values = (0..n * k / GROUP_SIZE)
            .map(|index| ((index % 5) as f32).mul_add(0.015_625, -0.093_75))
            .collect::<Vec<_>>();

        let weight = Array::from_slice(&packed, &[n, k / 32]);
        let scales = Array::from_slice(&scale_values, &[n, k / GROUP_SIZE])
            .as_dtype(dtype)
            .unwrap();
        let biases = if symmetric {
            let empty = Vec::<f32>::new();
            Array::from_slice(&empty, &[0])
        } else {
            Array::from_slice(&bias_values, &[n, k / GROUP_SIZE])
                .as_dtype(dtype)
                .unwrap()
        };
        (weight, scales, biases)
    }

    fn patterned_input(m: i32, k: i32, dtype: Dtype) -> Array {
        Array::from_slice(
            &(0..m * k)
                .map(|index| ((index * 11 + 9).rem_euclid(53) - 26) as f32 * 0.007_812_5)
                .collect::<Vec<_>>(),
            &[m, k],
        )
        .as_dtype(dtype)
        .unwrap()
    }

    fn assert_array_exact(label: &str, actual: &Array, expected: &Array) {
        assert_eq!(actual.shape(), expected.shape(), "{label} shape");
        assert_eq!(actual.dtype(), expected.dtype(), "{label} dtype");
        let actual_f32 = actual.as_dtype(Dtype::Float32).unwrap();
        let expected_f32 = expected.as_dtype(Dtype::Float32).unwrap();
        eval([&actual_f32, &expected_f32]).unwrap();
        for (index, (got, want)) in actual_f32
            .as_slice::<f32>()
            .iter()
            .zip(expected_f32.as_slice::<f32>())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label}[{index}] differs: {got:?} != {want:?}"
            );
        }
    }

    #[test]
    fn aligned_fast_qmv_matches_guarded_kernel_for_aligned_and_unaligned_n() {
        const M: i32 = 5;
        const K: i32 = 1024;
        let _exec = crate::mlx_exec::acquire();
        let nsg = fast_qmv_nsg();

        for &(n, symmetric) in &[(64_i32, true), (65_i32, false)] {
            assert_eq!(
                fast_qmv_has_aligned_rows(n, nsg, true),
                n == 64,
                "test shape must select the intended specialization"
            );
            let (weight, scales, biases) = patterned_weights(n, K, Dtype::Bfloat16, symmetric);
            let x = patterned_input(M, K, Dtype::Bfloat16);
            let guarded =
                bonsai_q1_qmv_fast_impl(&x, &weight, &scales, &biases, GROUP_SIZE, false).unwrap();
            let candidate =
                bonsai_q1_qmv_fast_impl(&x, &weight, &scales, &biases, GROUP_SIZE, true).unwrap();
            let public = bonsai_q1_qmv_fast(&x, &weight, &scales, &biases, GROUP_SIZE).unwrap();
            assert_array_exact(
                if n == 64 {
                    "aligned-N specialized QMV"
                } else {
                    "unaligned-N guarded fallback"
                },
                &candidate,
                &guarded,
            );
            assert_array_exact("public aligned-QMV dispatch", &public, &guarded);
        }
    }

    #[test]
    fn tg_lut4_row4_transform_matches_index_oracle_and_rejects_flat_layout() {
        const N: i32 = 8;
        const K: i32 = 256;
        let _exec = crate::mlx_exec::acquire();
        let groups = K / GROUP_SIZE;
        let k_packed = K / 32;
        let bits = (0..N * k_packed)
            .map(|index| u32::try_from(index).unwrap())
            .collect::<Vec<_>>();
        let scale_values = (0..N * groups)
            .map(|index| index as f32 + 0.25)
            .collect::<Vec<_>>();
        let weight = Array::from_slice(&bits, &[N, k_packed]);
        let scales = Array::from_slice(&scale_values, &[N, groups])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let packed = BonsaiQ1Row4::from_row_major(&weight, &scales).unwrap();
        eval([&packed.weights, &packed.scales]).unwrap();

        let packed_bits = packed.weights.as_slice::<u32>();
        let packed_scales_f32 = packed.scales.as_dtype(Dtype::Float32).unwrap();
        eval([&packed_scales_f32]).unwrap();
        let packed_scales = packed_scales_f32.as_slice::<f32>();
        for tile in 0..N / 4 {
            for group in 0..groups {
                for word in 0..4 {
                    for lane in 0..4 {
                        let src_row = tile * 4 + lane;
                        let src = (src_row * k_packed + group * 4 + word) as usize;
                        let dst = ((((tile * groups + group) * 4 + word) * 4) + lane) as usize;
                        assert_eq!(packed_bits[dst], bits[src]);
                    }
                }
                for lane in 0..4 {
                    let src = ((tile * 4 + lane) * groups + group) as usize;
                    let dst = ((tile * groups + group) * 4 + lane) as usize;
                    let expected = half::bf16::from_f32(scale_values[src]).to_f32();
                    assert_eq!(packed_scales[dst].to_bits(), expected.to_bits());
                }
            }
        }

        let flat_bits = Array::from_slice(&bits, &[bits.len() as i32]);
        let flat_scales = scales.reshape(&[-1]).unwrap();
        let error = BonsaiQ1Row4::from_packed_parts(flat_bits, flat_scales, N, K).unwrap_err();
        assert!(error.to_string().contains("invalid packed contract"));
    }

    #[test]
    fn tg_lut4_fp16_bf16_scales_preserve_leading_shape_and_m1_row_plan() {
        const N: i32 = 64;
        const K: i32 = 1024;
        const MAX_M: i32 = 5;
        let _exec = crate::mlx_exec::acquire();
        let values = (0..MAX_M * K)
            .map(|index| {
                let row = index / K;
                let col = index % K;
                ((row * 29 + col * 11 + 7).rem_euclid(61) - 30) as f32 * 0.007_812_5
            })
            .collect::<Vec<_>>();

        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let (weight, scales, _) = patterned_weights(N, K, dtype, true);
            let packed = BonsaiQ1Row4::from_row_major(&weight, &scales).unwrap();
            assert_eq!(packed.scales.dtype(), dtype);

            for m in 1..=MAX_M {
                let input = Array::from_slice(&values[..(m * K) as usize], &[1, m, K])
                    .as_dtype(dtype)
                    .unwrap();
                let stacked = bonsai_q1_tg_lut4_qmm(&input, &packed).unwrap();
                assert_eq!(stacked.shape(), &[1, m, N]);
                let stacked_f32 = stacked.as_dtype(Dtype::Float32).unwrap();
                eval([&stacked_f32]).unwrap();

                for row in 0..m {
                    let start = (row * K) as usize;
                    let single = Array::from_slice(&values[start..start + K as usize], &[1, 1, K])
                        .as_dtype(dtype)
                        .unwrap();
                    let single = bonsai_q1_tg_lut4_qmm(&single, &packed)
                        .unwrap()
                        .as_dtype(Dtype::Float32)
                        .unwrap();
                    eval([&single]).unwrap();
                    for column in 0..N as usize {
                        let got = stacked_f32.as_slice::<f32>()[(row * N) as usize + column];
                        let expected = single.as_slice::<f32>()[column];
                        assert_eq!(
                            got.to_bits(),
                            expected.to_bits(),
                            "dtype={dtype:?} M={m} row={row} column={column}"
                        );
                    }
                }
            }
        }
    }

    /// Compare the stock guarded QMV curve with the opt-in aligned-N
    /// specialization on Bonsai-27B's dominant gate/up projection shape.
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- \
    ///   bench_aligned_fast_qmv_m_sweep --ignored --nocapture --exact
    /// ```
    #[test]
    #[ignore = "microbenchmark, requires Apple Metal GPU"]
    fn bench_aligned_fast_qmv_m_sweep() {
        use std::time::Instant;

        const N: i32 = 17_408;
        const K: i32 = 5_120;
        const M_VALUES: [i32; 6] = [1, 2, 3, 4, 5, 8];
        const WARMUP_ITERS: usize = 8;
        const DEFAULT_SAMPLES: usize = 51;

        let _exec = crate::mlx_exec::acquire();
        assert!(fast_qmv_has_aligned_rows(N, fast_qmv_nsg(), true));
        let (weight, scales, biases) = patterned_weights(N, K, Dtype::Bfloat16, true);
        let inputs = M_VALUES
            .iter()
            .map(|&m| (m, patterned_input(m, K, Dtype::Bfloat16)))
            .collect::<Vec<_>>();
        let mut resident = vec![&weight, &scales, &biases];
        resident.extend(inputs.iter().map(|(_, input)| input));
        eval(resident).unwrap();

        let samples = std::env::var("HIGGS_BONSAI_ALIGNED_QMV_BENCH_SAMPLES")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|count| *count > 0)
            .unwrap_or(DEFAULT_SAMPLES);

        let measure = |input: &Array, prefer_aligned: bool| -> f64 {
            let start = Instant::now();
            let output = bonsai_q1_qmv_fast_impl(
                input,
                &weight,
                &scales,
                &biases,
                GROUP_SIZE,
                prefer_aligned,
            )
            .unwrap();
            eval([&output]).unwrap();
            let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
            std::hint::black_box(output);
            elapsed_us
        };

        let summarize = |values: &mut [f64]| -> (f64, f64) {
            values.sort_by(f64::total_cmp);
            let median = values[values.len() / 2];
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            (median, mean)
        };

        let mut rows = Vec::with_capacity(M_VALUES.len());
        for (m, input) in &inputs {
            let guarded_check =
                bonsai_q1_qmv_fast_impl(input, &weight, &scales, &biases, GROUP_SIZE, false)
                    .unwrap();
            let aligned_check =
                bonsai_q1_qmv_fast_impl(input, &weight, &scales, &biases, GROUP_SIZE, true)
                    .unwrap();
            assert_array_exact(
                &format!("aligned-N benchmark M={m}"),
                &aligned_check,
                &guarded_check,
            );

            for iteration in 0..WARMUP_ITERS {
                if iteration % 2 == 0 {
                    std::hint::black_box(measure(input, false));
                    std::hint::black_box(measure(input, true));
                } else {
                    std::hint::black_box(measure(input, true));
                    std::hint::black_box(measure(input, false));
                }
            }

            let mut guarded_us = Vec::with_capacity(samples);
            let mut aligned_us = Vec::with_capacity(samples);
            for sample in 0..samples {
                if sample % 2 == 0 {
                    guarded_us.push(measure(input, false));
                    aligned_us.push(measure(input, true));
                } else {
                    aligned_us.push(measure(input, true));
                    guarded_us.push(measure(input, false));
                }
            }
            let (guarded_median, guarded_mean) = summarize(&mut guarded_us);
            let (aligned_median, aligned_mean) = summarize(&mut aligned_us);
            rows.push((
                *m,
                guarded_median,
                guarded_mean,
                aligned_median,
                aligned_mean,
            ));
        }

        let (_, guarded_m1_median, guarded_m1_mean, aligned_m1_median, aligned_m1_mean) = rows[0];
        println!("Bonsai Q1 BF16 N={N} K={K}, samples={samples}");
        println!(
            " M | OFF median  mean  med/M1 mean/M1 | ON median  mean  med/M1 mean/M1 | ON speedup"
        );
        for (m, guarded_median, guarded_mean, aligned_median, aligned_mean) in rows {
            println!(
                "{m:>2} | {guarded_median:>9.1} {guarded_mean:>7.1} {guarded_median_norm:>6.2}x {guarded_mean_norm:>7.2}x | \
                 {aligned_median:>9.1} {aligned_mean:>7.1} {aligned_median_norm:>6.2}x {aligned_mean_norm:>7.2}x | \
                 {speedup:>8.3}x",
                guarded_median_norm = guarded_median / guarded_m1_median,
                guarded_mean_norm = guarded_mean / guarded_m1_mean,
                aligned_median_norm = aligned_median / aligned_m1_median,
                aligned_mean_norm = aligned_mean / aligned_m1_mean,
                speedup = guarded_median / aligned_median,
            );
        }
    }
}
