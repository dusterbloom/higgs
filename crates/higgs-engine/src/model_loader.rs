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

/// Load a model from a directory, auto-detecting the architecture.
///
/// Once the model is constructed, `maybe_enable_ane_gdn` runs and — if
/// `HIGGS_TARGET_ANE_GDN=1` and the model is Qwen3-Next-family — attaches
/// the GDN ANE worker thread to every linear layer.
pub fn load_model<P: AsRef<Path>>(model_dir: P) -> Result<AnyModel, EngineError> {
    let config = ModelConfig::from_dir(&model_dir)?;

    let mut model = load_model_inner(&config)?;
    maybe_enable_ane_gdn(&mut model);
    Ok(model)
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
fn maybe_enable_ane_gdn(model: &mut AnyModel) {
    if std::env::var("HIGGS_TARGET_ANE_GDN").as_deref() != Ok("1") {
        return;
    }
    let AnyModel::Qwen3Next(qwen) = model else {
        tracing::debug!(
            "HIGGS_TARGET_ANE_GDN=1 set but model is not Qwen3Next — skipping ANE GDN setup"
        );
        return;
    };
    const ANE_GDN_SEQ_LEN: i32 = 32;
    match qwen.enable_ane_gdn_all_layers_via_worker(ANE_GDN_SEQ_LEN) {
        Ok(report) => {
            tracing::info!(
                ?report,
                seq_len = ANE_GDN_SEQ_LEN,
                "ANE GDN worker enabled — all linear layers offloaded"
            );
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "HIGGS_TARGET_ANE_GDN=1 set but enable_ane_gdn_all_layers_via_worker failed — \
                 falling back to Metal"
            );
        }
    }
}

#[cfg(not(feature = "ane"))]
fn maybe_enable_ane_gdn(_model: &mut AnyModel) {
    if std::env::var("HIGGS_TARGET_ANE_GDN").as_deref() == Ok("1") {
        tracing::warn!(
            "HIGGS_TARGET_ANE_GDN=1 set but binary built without `ane` feature — ignoring"
        );
    }
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
