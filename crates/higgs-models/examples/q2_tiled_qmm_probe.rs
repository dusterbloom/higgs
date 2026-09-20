use std::time::Instant;

use half::{bf16, f16};
use higgs_models::bonsai_q2::{GROUP_SIZE, PackedQ2Linear, WEIGHTS_PER_WORD};
use higgs_models::quant_mode::{QuantMode, quantized_matmul};
use higgs_models::q2_tiled_qmm_bench;
use mlx_rs::{Array, Dtype, error::Exception};

type ProbeResult<T> = Result<T, String>;
const REPETITIONS: usize = 7;

#[derive(Clone, Copy)]
enum ActivationDtype {
    F16,
    Bf16,
}

impl ActivationDtype {
    const fn mlx(self) -> Dtype {
        match self {
            Self::F16 => Dtype::Float16,
            Self::Bf16 => Dtype::Bfloat16,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    fn round_to_dtype(self, value: f32) -> f32 {
        match self {
            Self::F16 => f16::from_f32(value).to_f32(),
            Self::Bf16 => bf16::from_f32(value).to_f32(),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct MemoryUsage {
    active_before_bytes: usize,
    absolute_peak_bytes: usize,
    incremental_peak_bytes: usize,
}

fn main() {
    if let Err(error) = entry() {
        eprintln!("Q2_TILED_QMM_ERROR {error}");
        std::process::exit(1);
    }
}

fn entry() -> ProbeResult<()> {
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .ok_or_else(|| "usage: q2_tiled_qmm_probe <correctness|bench>".to_owned())?;
    if args.next().is_some() {
        return Err("expected exactly one positional mode".to_owned());
    }

    let _exec = higgs_models::mlx_exec::acquire();
    match mode.as_str() {
        "correctness" => correctness(),
        "bench" => benchmark(),
        _ => Err(format!("unknown mode {mode:?}; expected correctness or bench")),
    }
}

fn mlx<T>(result: Result<T, Exception>) -> ProbeResult<T> {
    result.map_err(|error| error.to_string())
}

fn eval(array: &Array) -> ProbeResult<()> {
    mlx(array.eval())
}

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

fn make_ternary(n: usize, k: usize, seed: u64) -> PackedQ2Linear {
    let packed_cols = k / WEIGHTS_PER_WORD;
    let n_groups = k / GROUP_SIZE;
    let mut state = seed;
    let w_packed = (0..n * packed_cols)
        .map(|_| {
            let mut word = 0_u32;
            for field in 0..WEIGHTS_PER_WORD {
                word |= (lcg(&mut state) % 3) << (2 * field);
            }
            word
        })
        .collect();
    let scales: Vec<f16> = (0..n * n_groups)
        .map(|i| f16::from_f32(0.05 + 0.013 * (i % 7) as f32))
        .collect();
    let biases = scales
        .iter()
        .map(|&scale| f16::from_f32(-scale.to_f32()))
        .collect();
    PackedQ2Linear {
        w_packed,
        scales,
        biases,
        out_features: n,
        in_features: k,
    }
}

fn upload(p: &PackedQ2Linear) -> (Array, Array, Array) {
    let w = Array::from_slice(
        &p.w_packed,
        &[p.out_features as i32, p.packed_cols() as i32],
    );
    let s = Array::from_slice(
        &p.scales,
        &[p.out_features as i32, p.n_groups() as i32],
    );
    let b = Array::from_slice(
        &p.biases,
        &[p.out_features as i32, p.n_groups() as i32],
    );
    (w, s, b)
}

fn dense_matvec_reference(p: &PackedQ2Linear, x: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; p.out_features];
    let mut weights = vec![0.0_f32; p.in_features];
    for (row, result) in output.iter_mut().enumerate() {
        p.dequant_row_to_fp32(row, &mut weights);
        *result = x
            .iter()
            .zip(&weights)
            .map(|(&activation, &weight)| activation * weight)
            .sum();
    }
    output
}

#[derive(Clone, Copy)]
struct ErrorMetrics {
    max_abs: f32,
    normalized_rms: f32,
}

fn error_metrics(actual: &Array, reference: &[f32], path: &str) -> ProbeResult<ErrorMetrics> {
    let actual_f32 = mlx(actual.as_dtype(Dtype::Float32))?;
    eval(&actual_f32)?;
    let got = actual_f32.as_slice::<f32>();
    if got.len() != reference.len() {
        return Err(format!(
            "{path} output length {} != reference length {}",
            got.len(),
            reference.len()
        ));
    }
    let mut max_abs = 0.0_f32;
    let mut error_sq = 0.0_f32;
    let mut reference_sq = 0.0_f32;
    for (&value, &want) in got.iter().zip(reference) {
        if !value.is_finite() {
            return Err(format!("{path} output contains a non-finite value"));
        }
        let error = value - want;
        max_abs = max_abs.max(error.abs());
        error_sq += error * error;
        reference_sq += want * want;
    }
    Ok(ErrorMetrics {
        max_abs,
        normalized_rms: (error_sq / reference_sq.max(1.0e-12)).sqrt(),
    })
}

fn correctness() -> ProbeResult<()> {
    let cases = [
        (96usize, 256usize, 9usize, ActivationDtype::F16, 0x1234_5678_u64),
        (130, 4096, 17, ActivationDtype::Bf16, 0x5EED_5EED),
        (96, 17_408, 33, ActivationDtype::F16, 0xCAFE_BABE),
    ];

    for &(n, k, m, dtype, seed) in &cases {
        let p = make_ternary(n, k, seed);
        let (weight, scales, biases) = upload(&p);
        let mut state = 0xABCD_EF01_u64;
        let input_f32: Vec<f32> = (0..m * k)
            .map(|_| (lcg(&mut state) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
            .collect();
        let x = mlx(
            Array::from_slice(&input_f32, &[m as i32, k as i32]).as_dtype(dtype.mlx()),
        )?;
        let input_reference: Vec<f32> = input_f32
            .iter()
            .map(|&value| dtype.round_to_dtype(value))
            .collect();
        let mut reference = Vec::with_capacity(m * n);
        for row in input_reference.chunks_exact(k) {
            reference.extend(dense_matvec_reference(&p, row));
        }

        let round_through_bf16_to_f16 = |array: &Array| -> ProbeResult<Array> {
            let bf16 = mlx(array.as_dtype(Dtype::Bfloat16))?;
            mlx(bf16.as_dtype(Dtype::Float16))
        };
        let (matched_x, matched_scales, matched_biases) = match dtype {
            ActivationDtype::F16 => (
                round_through_bf16_to_f16(&x)?,
                round_through_bf16_to_f16(&scales)?,
                round_through_bf16_to_f16(&biases)?,
            ),
            ActivationDtype::Bf16 => (
                x.clone(),
                mlx(scales.as_dtype(Dtype::Bfloat16))?,
                mlx(biases.as_dtype(Dtype::Bfloat16))?,
            ),
        };
        let native_stock = mlx(quantized_matmul(
            &x,
            &weight,
            &scales,
            Some(&biases),
            true,
            GROUP_SIZE as i32,
            2,
            QuantMode::Affine,
        ))?;
        let cast_matched_stock = mlx(quantized_matmul(
            &matched_x,
            &weight,
            &matched_scales,
            Some(&matched_biases),
            true,
            GROUP_SIZE as i32,
            2,
            QuantMode::Affine,
        ))?;
        let scalar = mlx(q2_tiled_qmm_bench::scalar(
            &x,
            &weight,
            &scales,
            GROUP_SIZE as i32,
        ))?;
        let tiled = mlx(q2_tiled_qmm_bench::tiled(
            &x,
            &weight,
            &scales,
            GROUP_SIZE as i32,
        ))?;
        for output in [&native_stock, &cast_matched_stock, &scalar, &tiled] {
            eval(output)?;
        }
        if tiled.shape() != [m as i32, n as i32] {
            return Err(format!(
                "tiled shape {:?} != [{m}, {n}] for dtype {}",
                tiled.shape(),
                dtype.label()
            ));
        }
        if tiled.dtype() != x.dtype() {
            return Err(format!(
                "tiled dtype {:?} != input dtype {:?}",
                tiled.dtype(),
                x.dtype()
            ));
        }

        let native_metrics = error_metrics(&native_stock, &reference, "native_stock")?;
        let cast_metrics = error_metrics(&cast_matched_stock, &reference, "cast_matched_stock")?;
        let scalar_metrics = error_metrics(&scalar, &reference, "scalar")?;
        let tiled_metrics = error_metrics(&tiled, &reference, "tiled")?;
        println!(
            "Q2_TILED_QMM mode=correctness N={n} K={k} M={m} dtype={} native_stock_max_abs={:.6} native_stock_nrmse={:.6} cast_stock_max_abs={:.6} cast_stock_nrmse={:.6} scalar_max_abs={:.6} scalar_nrmse={:.6} tiled_max_abs={:.6} tiled_nrmse={:.6}",
            dtype.label(),
            native_metrics.max_abs,
            native_metrics.normalized_rms,
            cast_metrics.max_abs,
            cast_metrics.normalized_rms,
            scalar_metrics.max_abs,
            scalar_metrics.normalized_rms,
            tiled_metrics.max_abs,
            tiled_metrics.normalized_rms,
        );
        if tiled_metrics.normalized_rms > 1.0e-2 || tiled_metrics.max_abs > 0.25 {
            return Err(format!(
                "tiled correctness gate failed for N={n} K={k} M={m}: nrmse={} max_abs={}",
                tiled_metrics.normalized_rms, tiled_metrics.max_abs
            ));
        }
    }
    Ok(())
}

#[allow(unsafe_code)]
fn active_memory_bytes() -> ProbeResult<usize> {
    let mut bytes = 0_usize;
    let status = unsafe { mlx_sys::mlx_get_active_memory(&mut bytes) };
    if status == 0 {
        Ok(bytes)
    } else {
        Err(format!("mlx_get_active_memory failed: status {status}"))
    }
}

#[allow(unsafe_code)]
fn reset_peak_memory() -> ProbeResult<()> {
    let status = unsafe { mlx_sys::mlx_reset_peak_memory() };
    if status == 0 {
        Ok(())
    } else {
        Err(format!("mlx_reset_peak_memory failed: status {status}"))
    }
}

#[allow(unsafe_code)]
fn peak_memory_bytes() -> ProbeResult<usize> {
    let mut bytes = 0_usize;
    let status = unsafe { mlx_sys::mlx_get_peak_memory(&mut bytes) };
    if status == 0 {
        Ok(bytes)
    } else {
        Err(format!("mlx_get_peak_memory failed: status {status}"))
    }
}

fn measure_one(operation: &mut dyn FnMut() -> ProbeResult<Array>) -> ProbeResult<(f64, MemoryUsage)> {
    let active_before_bytes = active_memory_bytes()?;
    reset_peak_memory()?;
    let started = Instant::now();
    let output = operation()?;
    eval(&output)?;
    let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0;
    drop(output);
    let absolute_peak_bytes = peak_memory_bytes()?;
    Ok((
        elapsed_us,
        MemoryUsage {
            active_before_bytes,
            absolute_peak_bytes,
            incremental_peak_bytes: absolute_peak_bytes.saturating_sub(active_before_bytes),
        },
    ))
}

fn warm(operation: &mut dyn FnMut() -> ProbeResult<Array>) -> ProbeResult<()> {
    let output = operation()?;
    eval(&output)
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn benchmark() -> ProbeResult<()> {
    let mut state = 0x7654_3210_u64;
    let mut missed_gates = Vec::new();

    for &(n, k, label, seed) in &[
        (34_816usize, 5_120usize, "gate_up", 0x1234_5678_u64),
        (5_120, 17_408, "down", 0x8765_4321),
    ] {
        let p = make_ternary(n, k, seed);
        let (weight, scales, biases) = upload(&p);

        for m in [16usize, 32, 64, 512, 1024] {
            let input_values: Vec<f16> = (0..m * k)
                .map(|_| {
                    f16::from_f32(
                        (lcg(&mut state) as f32 / u32::MAX as f32).mul_add(0.2, -0.1),
                    )
                })
                .collect();
            let x = Array::from_slice(&input_values, &[m as i32, k as i32]);
            let mut stock = || {
                mlx(quantized_matmul(
                    &x,
                    &weight,
                    &scales,
                    Some(&biases),
                    true,
                    GROUP_SIZE as i32,
                    2,
                    QuantMode::Affine,
                ))
            };
            let mut scalar = || {
                mlx(q2_tiled_qmm_bench::scalar(
                    &x,
                    &weight,
                    &scales,
                    GROUP_SIZE as i32,
                ))
            };
            let mut tiled = || {
                mlx(q2_tiled_qmm_bench::tiled(
                    &x,
                    &weight,
                    &scales,
                    GROUP_SIZE as i32,
                ))
            };

            warm(&mut stock)?;
            warm(&mut scalar)?;
            warm(&mut tiled)?;
            let mut samples: [Vec<f64>; 3] =
                std::array::from_fn(|_| Vec::with_capacity(REPETITIONS));
            let mut memory = [MemoryUsage::default(); 3];
            let mut saw_memory = [false; 3];
            for repetition in 0..REPETITIONS {
                let order = match repetition % 3 {
                    0 => [0, 1, 2],
                    1 => [1, 2, 0],
                    _ => [2, 0, 1],
                };
                for path in order {
                    let (elapsed_us, usage) = match path {
                        0 => measure_one(&mut stock)?,
                        1 => measure_one(&mut scalar)?,
                        _ => measure_one(&mut tiled)?,
                    };
                    samples[path].push(elapsed_us);
                    if !saw_memory[path]
                        || usage.incremental_peak_bytes > memory[path].incremental_peak_bytes
                    {
                        memory[path] = usage;
                        saw_memory[path] = true;
                    }
                }
            }
            let medians = [
                median(&mut samples[0]),
                median(&mut samples[1]),
                median(&mut samples[2]),
            ];
            let tiled_over_stock = medians[0] / medians[2];
            println!(
                "Q2_TILED_QMM mode=bench projection={label} N={n} K={k} M={m} repetitions={REPETITIONS} stock_us={:.2} scalar_us={:.2} tiled_us={:.2} tiled_over_stock_speedup={:.3} stock_active_before_bytes={} stock_peak_bytes={} stock_incremental_peak_bytes={} scalar_active_before_bytes={} scalar_peak_bytes={} scalar_incremental_peak_bytes={} tiled_active_before_bytes={} tiled_peak_bytes={} tiled_incremental_peak_bytes={}",
                medians[0],
                medians[1],
                medians[2],
                tiled_over_stock,
                memory[0].active_before_bytes,
                memory[0].absolute_peak_bytes,
                memory[0].incremental_peak_bytes,
                memory[1].active_before_bytes,
                memory[1].absolute_peak_bytes,
                memory[1].incremental_peak_bytes,
                memory[2].active_before_bytes,
                memory[2].absolute_peak_bytes,
                memory[2].incremental_peak_bytes,
            );
            if m >= 512 && tiled_over_stock < 1.2 {
                missed_gates.push(format!("{label}:M={m}:{tiled_over_stock:.3}x"));
            }
        }
        drop((p, weight, scales, biases));
    }

    if missed_gates.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "1.2x high-M gate failed after full sweep: {}",
            missed_gates.join(", ")
        ))
    }
}
