use std::path::{Path, PathBuf};

use higgs_models::{AnyModel, load_tokenizer as shared_load_tokenizer, registry, transformer};

use crate::error::EngineError;

/// Configuration for loading a model from a directory.
#[derive(Debug)]
pub struct ModelConfig {
    pub model_dir: PathBuf,
    pub model_type: String,
}

impl ModelConfig {
    /// Detect model type and create a config from a model directory.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, EngineError> {
        let model_dir = dir.as_ref().to_path_buf();
        let model_type = registry::detect_model_type(&model_dir)?;

        if !registry::is_supported(&model_type) {
            return Err(EngineError::Model(
                higgs_models::error::ModelError::UnsupportedModel(model_type),
            ));
        }

        Ok(Self {
            model_dir,
            model_type,
        })
    }
}

/// Pending ANE GDN setup work that must complete on the inference worker
/// thread (P0.8 Stage 2). Carries Send-safe dequantized projection weights
/// produced on the main thread; the consumer calls
/// [`higgs_models::qwen3_next::Qwen3NextCausalLM::finalize_ane_gdn_inline`]
/// on the inference thread to compile + install the kernels there.
///
/// `None` from [`load_model`] means either ANE GDN is disabled
/// (`HIGGS_TARGET_ANE_GDN!=1`), the model is non–Qwen3-Next, or the legacy
/// worker fallback was used (`HIGGS_ANE_GDN_WORKER=1`).
#[cfg(feature = "ane")]
pub struct PendingAneGdn {
    pub weights: Vec<higgs_models::qwen3_next_ane_worker::GdnLayerWeights>,
    pub seq_len: i32,
}

#[cfg(not(feature = "ane"))]
pub struct PendingAneGdn;

/// Pending ANE `lm_head` setup work that must complete on the inference
/// worker thread. Carries Send-safe dequantized `[vocab, hidden]` weights
/// produced on the main thread; the consumer calls
/// [`higgs_models::qwen3_next::Qwen3NextCausalLM::finalize_ane_lm_head_inline`]
/// on the inference thread to compile + install the kernel there.
///
/// Shipstuff's mlx-ane-sd measured target `lm_head` on ANE at ~+20% decode
/// throughput on Qwen3-4B dense; the `lm_draft` step in DFlash verify
/// (~24 ms/round per session 2 notes) is the analogous drafter-side win.
/// Both collapse to the same compiled kernel when `seq_len` covers both the
/// drafter block size and the target verify slice.
///
/// `None` from [`load_model`] means either ANE `lm_head` is disabled
/// (`HIGGS_TARGET_ANE_LM_HEAD!=1`), the model is non–Qwen3-Next, the model
/// has tied word embeddings (not supported in step 1), or prep failed.
#[cfg(feature = "ane")]
pub struct PendingAneLmHead {
    pub weights: Vec<f32>,
    pub hidden: usize,
    pub vocab: usize,
    pub seq_len: i32,
}

#[cfg(not(feature = "ane"))]
pub struct PendingAneLmHead;

/// Load a model from a directory, auto-detecting the architecture.
///
/// Once the model is constructed, `maybe_enable_ane_gdn` runs and — if
/// `HIGGS_TARGET_ANE_GDN=1` and the model is Qwen3-Next-family — either
/// (a) returns a `PendingAneGdn` payload for the caller to finalize on the
/// inference worker thread (P0.8 Stage 2 inline path, the default), or
/// (b) attaches the legacy mpsc GDN ANE worker thread to every linear
/// layer immediately (when `HIGGS_ANE_GDN_WORKER=1` — kept as the
/// regression safety valve).
pub fn load_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<(AnyModel, Option<PendingAneGdn>, Option<PendingAneLmHead>), EngineError> {
    let config = ModelConfig::from_dir(&model_dir)?;

    let mut model = load_model_inner(&config)?;
    let pending_gdn = maybe_enable_ane_gdn(&mut model);
    let pending_lm_head = maybe_enable_ane_lm_head(&mut model);
    Ok((model, pending_gdn, pending_lm_head))
}

fn load_model_inner(config: &ModelConfig) -> Result<AnyModel, EngineError> {
    match config.model_type.as_str() {
        "qwen2" | "qwen3" | "llama" | "mistral" => {
            let model = transformer::load_model(&config.model_dir).map_err(EngineError::Model)?;
            Ok(AnyModel::Transformer(model))
        }
        "qwen3_next" => {
            let model = higgs_models::qwen3_next::load_qwen3_next_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Qwen3Next(model))
        }
        "qwen3_5" => {
            let model = higgs_models::qwen3_next::load_qwen3_5_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Qwen3Next(model))
        }
        "qwen3_5_moe" => {
            let model = higgs_models::qwen3_next::load_qwen3_5_moe_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Qwen3Next(model))
        }
        "qwen3_moe" => {
            let model = higgs_models::qwen3_moe::load_qwen3_moe_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Qwen3Moe(model))
        }
        "gemma2" => {
            let model = higgs_models::gemma2::load_gemma2_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Gemma2(model))
        }
        "phi3" => {
            let model = higgs_models::phi3::load_phi3_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Phi3(model))
        }
        "starcoder2" => {
            let model = higgs_models::starcoder2::load_starcoder2_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Starcoder2(model))
        }
        "llava-qwen2" => {
            let model = higgs_models::llava_qwen2::load_llava_qwen2_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::LlavaQwen2(model))
        }
        "deepseek_v2" => {
            let model = higgs_models::deepseek_v2::load_deepseek_v2_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::DeepSeekV2(model))
        }
        other => Err(EngineError::Model(
            higgs_models::error::ModelError::UnsupportedModel(other.to_owned()),
        )),
    }
}

/// Wave 4: opt-in GDN-on-ANE offload via env-var.
///
/// When `HIGGS_TARGET_ANE_GDN=1`, attaches the model-wide
/// `qwen-gdn-ane-worker` thread to every linear layer of a Qwen3-Next-family
/// model. The worker handle is `Send + Sync`, so unlike the inline
/// `Vec<Arc<GdnAneLayerKernels>>` path (Wave 1/2), this survives the model
/// being moved into the inference worker thread (`batch_engine.rs:117`,
/// `simple.rs`).
///
/// Single-bucket only — `seq_len=32` is hard-coded for now (covers the
/// drafter target verify shape; runtime seqs > 32 fall back to Metal). When
/// Wave 3's bridge bug is fixed and multi-bucket lands, this becomes a
/// configurable bucket list. Other model families silently no-op — this hook
/// only fires for `AnyModel::Qwen3Next`.
#[cfg(feature = "ane")]
fn maybe_enable_ane_gdn(model: &mut AnyModel) -> Option<PendingAneGdn> {
    if std::env::var("HIGGS_TARGET_ANE_GDN").as_deref() != Ok("1") {
        return None;
    }
    let AnyModel::Qwen3Next(qwen) = model else {
        tracing::debug!(
            "HIGGS_TARGET_ANE_GDN=1 set but model is not Qwen3Next — skipping ANE GDN setup"
        );
        return None;
    };
    const ANE_GDN_SEQ_LEN: i32 = 32;

    // Regression safety valve: HIGGS_ANE_GDN_WORKER=1 forces the legacy
    // mpsc worker path (compiles immediately on a dedicated worker thread,
    // attaches a Send+Sync handle). Default (env unset / "0") returns a
    // `PendingAneGdn` payload that the caller finalizes on the inference
    // worker thread — see P0.8 Stage 2.
    if std::env::var("HIGGS_ANE_GDN_WORKER").as_deref() == Ok("1") {
        match qwen.enable_ane_gdn_all_layers_via_worker(ANE_GDN_SEQ_LEN) {
            Ok(report) => {
                tracing::info!(
                    ?report,
                    seq_len = ANE_GDN_SEQ_LEN,
                    "ANE GDN legacy worker enabled (HIGGS_ANE_GDN_WORKER=1)"
                );
            }
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "HIGGS_ANE_GDN_WORKER=1 fallback failed — falling back to Metal"
                );
            }
        }
        return None;
    }

    match qwen.prepare_ane_gdn_weights(ANE_GDN_SEQ_LEN) {
        Ok((weights, seq_len)) => {
            tracing::info!(
                n_layers = weights.len(),
                seq_len,
                "ANE GDN inline prep complete on main thread — finalize pending on inference worker"
            );
            Some(PendingAneGdn { weights, seq_len })
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "HIGGS_TARGET_ANE_GDN=1 set but prepare_ane_gdn_weights failed — falling back to Metal"
            );
            None
        }
    }
}

#[cfg(not(feature = "ane"))]
fn maybe_enable_ane_gdn(_model: &mut AnyModel) -> Option<PendingAneGdn> {
    if std::env::var("HIGGS_TARGET_ANE_GDN").as_deref() == Ok("1") {
        tracing::warn!(
            "HIGGS_TARGET_ANE_GDN=1 set but binary built without `ane` feature — ignoring"
        );
    }
    None
}

/// Opt-in `lm_head`-on-ANE offload via env-var.
///
/// When `HIGGS_TARGET_ANE_LM_HEAD=1`, dequantize `lm_head` weights on the
/// main thread and hand the finalize step to the inference worker (same
/// pattern as [`maybe_enable_ane_gdn`] P0.8 Stage 2). Seq bucket defaults to
/// 32 — covers both the DFlash drafter's `block_size=16` output and target
/// verify's sliced-last-token path (seq=1). Runtime seqs > 32 fall back to
/// the Metal / QLinear path inside `project_logits`.
///
/// Other model families silently no-op — this hook only fires for
/// `AnyModel::Qwen3Next`. Tied-embedding models also no-op in step 1
/// (prep returns `Ok(None)`).
#[cfg(feature = "ane")]
fn maybe_enable_ane_lm_head(model: &mut AnyModel) -> Option<PendingAneLmHead> {
    if std::env::var("HIGGS_TARGET_ANE_LM_HEAD").as_deref() != Ok("1") {
        return None;
    }
    let AnyModel::Qwen3Next(qwen) = model else {
        tracing::debug!(
            "HIGGS_TARGET_ANE_LM_HEAD=1 set but model is not Qwen3Next — skipping ANE lm_head setup"
        );
        return None;
    };
    // Override with HIGGS_ANE_LM_HEAD_SEQ if set (stretch buckets during
    // development). Default 32 covers drafter block=16 + target verify slice=1.
    let seq_len: i32 = std::env::var("HIGGS_ANE_LM_HEAD_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    match qwen.prepare_lm_head_weights() {
        Ok(Some((weights, hidden, vocab))) => {
            tracing::info!(
                hidden,
                vocab,
                seq_len,
                weight_bytes = weights.len() * 4,
                "ANE lm_head prep complete on main thread — finalize pending on inference worker"
            );
            Some(PendingAneLmHead {
                weights,
                hidden,
                vocab,
                seq_len,
            })
        }
        Ok(None) => {
            tracing::warn!(
                "HIGGS_TARGET_ANE_LM_HEAD=1 set but model has tied word embeddings — \
                 skipping (not supported in step 1)"
            );
            None
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "HIGGS_TARGET_ANE_LM_HEAD=1 set but prepare_lm_head_weights failed — \
                 falling back to Metal"
            );
            None
        }
    }
}

#[cfg(not(feature = "ane"))]
fn maybe_enable_ane_lm_head(_model: &mut AnyModel) -> Option<PendingAneLmHead> {
    if std::env::var("HIGGS_TARGET_ANE_LM_HEAD").as_deref() == Ok("1") {
        tracing::warn!(
            "HIGGS_TARGET_ANE_LM_HEAD=1 set but binary built without `ane` feature — ignoring"
        );
    }
    None
}

/// Load a DFlash block-diffusion drafter from a model directory.
pub fn load_dflash_drafter<P: AsRef<Path>>(
    model_dir: P,
) -> Result<higgs_models::dflash::DFlashDrafter, EngineError> {
    higgs_models::dflash::load_dflash_drafter(model_dir.as_ref()).map_err(EngineError::Model)
}

/// Load a tokenizer from a model directory.
pub fn load_tokenizer<P: AsRef<Path>>(model_dir: P) -> Result<tokenizers::Tokenizer, EngineError> {
    shared_load_tokenizer(model_dir).map_err(|e| EngineError::Tokenization(e.to_string()))
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use higgs_models::error::ModelError;

    /// Create a temp dir with a config.json containing the given `model_type` and
    /// return the `ModelConfig` result.
    fn config_for_model(model_type: &str) -> (tempfile::TempDir, Result<ModelConfig, EngineError>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            format!(r#"{{"model_type": "{model_type}"}}"#),
        )
        .unwrap();
        let result = ModelConfig::from_dir(dir.path());
        (dir, result)
    }

    /// Write arbitrary content to config.json in a temp dir and return
    /// the `ModelConfig` result.
    fn config_from_raw(content: &str) -> (tempfile::TempDir, Result<ModelConfig, EngineError>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), content).unwrap();
        let result = ModelConfig::from_dir(dir.path());
        (dir, result)
    }

    #[test]
    fn model_config_from_dir_qwen2() {
        let (dir, result) = config_for_model("qwen2");
        let config = result.unwrap();
        assert_eq!(config.model_type, "qwen2");
        assert_eq!(config.model_dir, dir.path());
    }

    #[test]
    fn model_config_from_dir_qwen3() {
        let (_dir, result) = config_for_model("qwen3");
        assert_eq!(result.unwrap().model_type, "qwen3");
    }

    #[test]
    fn model_config_from_dir_llama() {
        let (_dir, result) = config_for_model("llama");
        assert_eq!(result.unwrap().model_type, "llama");
    }

    #[test]
    fn model_config_from_dir_mistral() {
        let (_dir, result) = config_for_model("mistral");
        assert_eq!(result.unwrap().model_type, "mistral");
    }

    #[test]
    fn model_config_from_dir_qwen3_next() {
        let (_dir, result) = config_for_model("qwen3_next");
        assert_eq!(result.unwrap().model_type, "qwen3_next");
    }

    #[test]
    fn model_config_from_dir_qwen3_moe() {
        let (_dir, result) = config_for_model("qwen3_moe");
        assert_eq!(result.unwrap().model_type, "qwen3_moe");
    }

    #[test]
    fn model_config_from_dir_gemma2() {
        let (_dir, result) = config_for_model("gemma2");
        assert_eq!(result.unwrap().model_type, "gemma2");
    }

    #[test]
    fn model_config_from_dir_phi3() {
        let (_dir, result) = config_for_model("phi3");
        assert_eq!(result.unwrap().model_type, "phi3");
    }

    #[test]
    fn model_config_from_dir_starcoder2() {
        let (_dir, result) = config_for_model("starcoder2");
        assert_eq!(result.unwrap().model_type, "starcoder2");
    }

    #[test]
    fn model_config_from_dir_deepseek_v2() {
        let (_dir, result) = config_for_model("deepseek_v2");
        assert_eq!(result.unwrap().model_type, "deepseek_v2");
    }

    #[test]
    fn model_config_from_dir_qwen3_5() {
        let (_dir, result) = config_for_model("qwen3_5");
        assert_eq!(result.unwrap().model_type, "qwen3_5");
    }

    #[test]
    fn model_config_from_dir_qwen3_5_moe() {
        let (_dir, result) = config_for_model("qwen3_5_moe");
        assert_eq!(result.unwrap().model_type, "qwen3_5_moe");
    }

    #[test]
    fn model_config_from_dir_unsupported_model_type() {
        let (_dir, result) = config_for_model("gpt2");
        match result {
            Err(e) => assert!(e.to_string().contains("gpt2")),
            Ok(_) => panic!("Expected error for unsupported model type"),
        }
    }

    #[test]
    fn model_config_from_dir_missing_config_json() {
        let dir = tempfile::tempdir().unwrap();
        let err = ModelConfig::from_dir(dir.path()).unwrap_err();
        assert!(matches!(err, EngineError::Model(ModelError::Io(_))));
    }

    #[test]
    fn model_config_from_dir_invalid_json() {
        let (_dir, result) = config_from_raw("not valid json {{{");
        let err = result.unwrap_err();
        assert!(matches!(err, EngineError::Model(ModelError::Json(_))));
    }

    #[test]
    fn model_config_from_dir_missing_model_type_field() {
        let (_dir, result) = config_from_raw(r#"{"vocab_size": 32000, "hidden_size": 4096}"#);
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            EngineError::Model(ModelError::UnsupportedModel(_))
        ));
    }

    #[test]
    fn load_tokenizer_missing_tokenizer_json() {
        let dir = tempfile::tempdir().unwrap();
        match load_tokenizer(dir.path()) {
            Err(e) => assert!(e.to_string().contains("Tokenization error")),
            Ok(_) => panic!("Expected error for missing tokenizer.json"),
        }
    }
}
