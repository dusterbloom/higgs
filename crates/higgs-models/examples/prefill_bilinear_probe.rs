//! Standalone feasibility control, not an implementation of the 9/4 paper.
//! Real packed weights, synthetic FP32 activations; no production dispatch.
use higgs_models::quant_mode::{QuantMode, dequantize, quantized_matmul};
use mlx_rs::{
    Array, Dtype,
    error::Exception,
    ops::{concatenate_axis, indexing::IndexOp},
};
use serde_json::json;
use std::{collections::HashMap, fs::File, io::Write, time::Instant};

fn main() {
    if let Err(error) = run() {
        eprintln!("PREFILL_BILINEAR_ERROR {error}");
        std::process::exit(1);
    }
}

// `wt` stores B transposed, preserving the checkpoint's [N,K] layout.
fn bilinear(x: &Array, wt: &Array, depth: usize) -> Result<Array, Exception> {
    if x.ndim() != 2 || wt.ndim() != 2 || x.shape()[1] != wt.shape()[1] {
        return Err(Exception::custom("expected x[M,K], wt[N,K]"));
    }
    if depth == 0 {
        return x.matmul(wt.transpose()?);
    }
    if x.shape().iter().chain(wt.shape()).any(|v| v % 2 != 0) {
        return Err(Exception::custom(
            "bilinear control requires even dimensions at each level",
        ));
    }
    let (m, k, n) = (x.shape()[0] / 2, x.shape()[1] / 2, wt.shape()[0] / 2);
    let (a, b, c, d) = (
        x.index((..m, ..k)),
        x.index((..m, k..)),
        x.index((m.., ..k)),
        x.index((m.., k..)),
    );
    let (e, f, g, h) = (
        wt.index((..n, ..k)),
        wt.index((n.., ..k)),
        wt.index((..n, k..)),
        wt.index((n.., k..)),
    );
    let p1 = bilinear(&a.add(&d)?, &e.add(&h)?, depth - 1)?;
    let p2 = bilinear(&c.add(&d)?, &e, depth - 1)?;
    let p3 = bilinear(&a, &f.subtract(&h)?, depth - 1)?;
    let p4 = bilinear(&d, &g.subtract(&e)?, depth - 1)?;
    let p5 = bilinear(&a.add(&b)?, &h, depth - 1)?;
    let p6 = bilinear(&c.subtract(&a)?, &e.add(&f)?, depth - 1)?;
    let p7 = bilinear(&b.subtract(&d)?, &g.add(&h)?, depth - 1)?;
    let c11 = p1.add(&p4)?.subtract(&p5)?.add(&p7)?;
    let c12 = p3.add(&p5)?;
    let c21 = p2.add(&p4)?;
    let c22 = p1.subtract(&p2)?.add(&p3)?.add(&p6)?;
    let top = concatenate_axis(&[c11, c12], 1)?;
    let bottom = concatenate_axis(&[c21, c22], 1)?;
    concatenate_axis(&[top, bottom], 0)
}

#[derive(Clone, Copy, Debug)]
enum Path {
    StockF32,
    StockF16,
    DenseF32,
    DenseF16,
    Strassen1,
    Strassen2,
}

fn operation(
    path: Path,
    x: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    prepared: Option<&Array>,
) -> Result<Array, Exception> {
    let xf = match path {
        Path::StockF32 | Path::DenseF32 => x.clone(),
        _ => x.as_dtype(Dtype::Float16)?,
    };
    if matches!(path, Path::StockF32 | Path::StockF16) {
        return quantized_matmul(&xf, w, s, Some(b), true, 128, 2, QuantMode::Affine);
    }
    let dense = match prepared {
        Some(value) => value.clone(),
        None => dequantize(w, s, Some(b), 128, 2, QuantMode::Affine)?,
    };
    let dense = if matches!(path, Path::DenseF32) {
        dense.as_dtype(Dtype::Float32)?
    } else {
        dense
    };
    let depth = match path {
        Path::Strassen1 => 1,
        Path::Strassen2 => 2,
        _ => 0,
    };
    bilinear(&xf, &dense, depth)
}

#[allow(unsafe_code)]
fn memory(peak: bool) -> Result<usize, Exception> {
    let mut bytes = 0;
    let status = unsafe {
        if peak {
            mlx_sys::mlx_get_peak_memory(&mut bytes)
        } else {
            mlx_sys::mlx_get_active_memory(&mut bytes)
        }
    };
    if status != 0 {
        return Err(Exception::custom("MLX memory counter failed"));
    }
    Ok(bytes)
}

#[allow(unsafe_code)]
fn reset_peak() -> Result<(), Exception> {
    if unsafe { mlx_sys::mlx_reset_peak_memory() } != 0 {
        return Err(Exception::custom("MLX peak reset failed"));
    }
    Ok(())
}

fn values(a: &Array) -> Result<Vec<f32>, Exception> {
    let f = a.as_dtype(Dtype::Float32)?;
    f.eval()?;
    Ok(f.as_slice::<f32>().to_vec())
}

fn errors(actual: &[f32], reference: &[f32]) -> Result<serde_json::Value, Exception> {
    if actual.len() != reference.len() || actual.iter().any(|v| !v.is_finite()) {
        return Err(Exception::custom(
            "invalid output length or non-finite output",
        ));
    }
    let (mut max_abs, mut sum, mut denominator) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&v, &r) in actual.iter().zip(reference) {
        let delta = f64::from(v) - f64::from(r);
        max_abs = max_abs.max(delta.abs());
        sum += delta * delta;
        denominator += f64::from(r) * f64::from(r);
    }
    Ok(json!({"max_abs": max_abs, "normalized_rms": (sum/denominator.max(1e-30)).sqrt()}))
}

fn tensor<'a>(weights: &'a HashMap<String, Array>, name: &str) -> Result<&'a Array, Exception> {
    weights
        .get(name)
        .ok_or_else(|| Exception::custom(format!("missing tensor {name}")))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: prefill_bilinear_probe FIXTURE.safetensors OUTPUT.jsonl".into());
    }
    let _exec = higgs_models::mlx_exec::acquire();
    let weights = Array::load_safetensors(&args[0])?;
    let mut output = File::options()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    writeln!(
        output,
        "{}",
        json!({"kind":"metadata", "fixture":args[0],
        "activations":"deterministic synthetic FP32, uniform [-0.1,0.1]",
        "repetitions":7, "control":"Strassen, not the nine-fourths construction"})
    )?;
    for (label, w, s, b) in [
        (
            "gate_up",
            concatenate_axis(
                &[
                    tensor(&weights, "gate_proj.weight")?,
                    tensor(&weights, "up_proj.weight")?,
                ],
                0,
            )?,
            concatenate_axis(
                &[
                    tensor(&weights, "gate_proj.scales")?,
                    tensor(&weights, "up_proj.scales")?,
                ],
                0,
            )?,
            concatenate_axis(
                &[
                    tensor(&weights, "gate_proj.biases")?,
                    tensor(&weights, "up_proj.biases")?,
                ],
                0,
            )?,
        ),
        (
            "down",
            tensor(&weights, "down_proj.weight")?.clone(),
            tensor(&weights, "down_proj.scales")?.clone(),
            tensor(&weights, "down_proj.biases")?.clone(),
        ),
    ] {
        mlx_rs::transforms::eval([&w, &s, &b])?;
        let (n, k) = (w.shape()[0], w.shape()[1] * 16);
        for m in [512, 1024] {
            let mut state = 0x5eed_cafe_u64;
            let input: Vec<f32> = (0..m * k)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 32) as u32 as f32 / u32::MAX as f32) * 0.2 - 0.1
                })
                .collect();
            let x = Array::from_slice(&input, &[m, k]);
            x.eval()?;
            let reference = values(&operation(Path::StockF32, &x, &w, &s, &b, None)?)?;
            let cast_reference = values(&operation(Path::StockF16, &x, &w, &s, &b, None)?)?;
            for prepared_phase in [false, true] {
                // Prepared paths retain one FP16 projection only. They are optimistic
                // controls, not a feasible cache of all model weights on a 32GB host.
                let dense = if prepared_phase {
                    let d = dequantize(&w, &s, Some(&b), 128, 2, QuantMode::Affine)?;
                    d.eval()?;
                    Some(d)
                } else {
                    None
                };
                let paths: &[Path] = if prepared_phase {
                    &[
                        Path::StockF32,
                        Path::DenseF16,
                        Path::Strassen1,
                        Path::Strassen2,
                    ]
                } else {
                    &[
                        Path::StockF32,
                        Path::StockF16,
                        Path::DenseF32,
                        Path::DenseF16,
                        Path::Strassen1,
                        Path::Strassen2,
                    ]
                };
                let mut timing = vec![Vec::<f64>::new(); paths.len()];
                let mut peaks = vec![(0_usize, 0_usize, 0_usize); paths.len()];
                let mut numerical = Vec::new();
                for &path in paths {
                    for _ in 0..2 {
                        operation(path, &x, &w, &s, &b, dense.as_ref())?.eval()?;
                    }
                    let got = values(&operation(path, &x, &w, &s, &b, dense.as_ref())?)?;
                    numerical.push(json!({"vs_native_stock":errors(&got,&reference)?,
                        "vs_cast_stock":errors(&got,&cast_reference)?}));
                }
                for round in 0..7 {
                    for offset in 0..paths.len() {
                        let index = (round + offset) % paths.len();
                        let before = memory(false)?;
                        reset_peak()?;
                        let start = Instant::now();
                        let y = operation(paths[index], &x, &w, &s, &b, dense.as_ref())?;
                        y.eval()?;
                        timing[index].push(start.elapsed().as_secs_f64() * 1000.0);
                        let peak = memory(true)?;
                        drop(y);
                        peaks[index].0 = peaks[index].0.max(before);
                        peaks[index].1 = peaks[index].1.max(peak);
                        peaks[index].2 = peaks[index].2.max(peak.saturating_sub(before));
                    }
                }
                let median = |samples: &[f64]| {
                    let mut sorted = samples.to_vec();
                    sorted.sort_by(f64::total_cmp);
                    sorted[sorted.len() / 2]
                };
                let stock_ms = median(&timing[0]);
                for (i, &path) in paths.iter().enumerate() {
                    let ms = median(&timing[i]);
                    let record = json!({"kind":"measurement","projection":label,"m":m,"n":n,"k":k,
                        "phase":if prepared_phase {"prepared_dense_control"} else {"transient"},
                        "path":format!("{path:?}"),"samples_ms":timing[i],"median_ms":ms,
                        "min_ms":timing[i].iter().copied().fold(f64::INFINITY,f64::min),
                        "max_ms":timing[i].iter().copied().fold(0.0_f64,f64::max),
                        "speedup_vs_native_stock":stock_ms/ms,"error":numerical[i],
                        "active_before_bytes":peaks[i].0,"absolute_peak_bytes":peaks[i].1,
                        "incremental_peak_bytes":peaks[i].2,
                        "prepared_weight_bytes":if prepared_phase {n as usize*k as usize*2} else {0}});
                    writeln!(output, "{record}")?;
                    output.flush()?;
                    println!(
                        "{label} M={m} {:?} {}: {ms:.3}ms {:.3}x peak_delta={}MiB",
                        path,
                        if prepared_phase {
                            "prepared"
                        } else {
                            "transient"
                        },
                        stock_ms / ms,
                        peaks[i].2 / 1048576
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_matches_independent_f64_oracle() {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        let (m, k, n) = (8usize, 12usize, 16usize);
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 23) as f32 - 11.0) / 8.0)
            .collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) / 8.0)
            .collect();
        let xa = Array::from_slice(&x, &[m as i32, k as i32]);
        let wa = Array::from_slice(&w, &[n as i32, k as i32]);
        for depth in [1, 2] {
            let actual = bilinear(&xa, &wa, depth).unwrap();
            actual.eval().unwrap();
            let values = actual.as_slice::<f32>();
            for row in 0..m {
                for col in 0..n {
                    let want: f64 = (0..k)
                        .map(|j| f64::from(x[row * k + j]) * f64::from(w[col * k + j]))
                        .sum();
                    assert!(
                        (f64::from(values[row * n + col]) - want).abs() < 1e-6,
                        "depth={depth} row={row} col={col} got={} want={want}",
                        values[row * n + col]
                    );
                }
            }
        }
    }
}
