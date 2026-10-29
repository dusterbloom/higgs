//! Zero-centered RMSNorm checkpoints come in two forms: raw `w` (HF, Clover's
//! Kapitsa pack) and mlx-lm's offset `1 + w`. The loader must tell them apart
//! from the weights themselves, using only the per-layer input norms.
use higgs_models::qwen3_next::norm_convention_probe::needs_offset;
use mlx_rs::Array;

fn checkpoint(input_norm_means: &[f32]) -> Vec<(String, Array)> {
    let mut tensors: Vec<(String, Array)> = input_norm_means
        .iter()
        .enumerate()
        .map(|(layer, &mean)| {
            let w: Vec<f32> = (0..8).map(|i| mean + (i as f32 - 3.5) * 0.02).collect();
            (
                format!("model.layers.{layer}.input_layernorm.weight"),
                Array::from_slice(&w, &[8]),
            )
        })
        .collect();
    // Decoys the decision must ignore: a large final norm (raw Clover: 1.73)
    // and the never-offset GDN gated norm.
    tensors.push((
        "model.norm.weight".into(),
        Array::from_slice(&[1.7_f32; 8], &[8]),
    ));
    tensors.push((
        "model.layers.0.linear_attn.norm.weight".into(),
        Array::from_slice(&[0.9_f32; 8], &[8]),
    ));
    tensors
}

fn decide(input_norm_means: &[f32]) -> bool {
    let tensors = checkpoint(input_norm_means);
    let view: Vec<(&str, Array)> = tensors
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();
    needs_offset(&view).unwrap()
}

#[test]
fn input_norm_medians_decide_the_convention() {
    // Measured shapes: raw Clover/Escha layers span -0.27..0.08; one outlier
    // layer must not flip the decision either way.
    assert!(
        decide(&[-0.04, -0.27, 0.08, 0.02, 0.9]),
        "raw norms need 1 + w"
    );
    assert!(
        !decide(&[0.96, 0.73, 1.08, 1.02, 0.1]),
        "offset norms load as-is"
    );
    assert!(!decide(&[]), "no layer norms: leave weights untouched");
}
