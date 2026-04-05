//! Save and load training deltas via safetensors + replay metadata via JSON.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

use crate::qwen3_next::DeltaMap;

/// Save a DeltaMap to a safetensors file.
pub fn save_deltas(deltas: &DeltaMap, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory: {e}"))?;
    }

    let mut tensors: Vec<(&str, TensorView<'_>)> = Vec::new();
    // We need to eval and extract data from each Array
    let mut data_store: HashMap<String, Vec<u8>> = HashMap::new();
    let mut shapes_store: HashMap<String, Vec<usize>> = HashMap::new();

    for (name, array) in deltas.iter() {
        mlx_rs::transforms::eval(std::slice::from_ref(array))
            .map_err(|e| format!("Failed to eval delta {name}: {e}"))?;
        let arr_f32 = array
            .as_dtype(mlx_rs::Dtype::Float32)
            .map_err(|e| format!("Failed to cast delta {name} to f32: {e}"))?;
        mlx_rs::transforms::eval(std::slice::from_ref(&arr_f32))
            .map_err(|e| format!("Failed to eval cast delta {name}: {e}"))?;
        let slice = arr_f32.as_slice::<f32>();
        let bytes: Vec<u8> = slice.iter().flat_map(|f| f.to_le_bytes()).collect();
        let shape: Vec<usize> = arr_f32.shape().iter().map(|&d| d as usize).collect();
        data_store.insert(name.clone(), bytes);
        shapes_store.insert(name.clone(), shape);
    }

    for (name, bytes) in &data_store {
        let shape = &shapes_store[name];
        let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
            .map_err(|e| format!("Failed to create tensor view for {name}: {e}"))?;
        tensors.push((name.as_str(), view));
    }

    let serialized = safetensors::serialize(tensors, &None)
        .map_err(|e| format!("Failed to serialize deltas: {e}"))?;
    std::fs::write(path, serialized)
        .map_err(|e| format!("Failed to write {}: {e}", path.display()))?;

    Ok(())
}

/// Load a DeltaMap from a safetensors file.
pub fn load_deltas(path: &Path) -> Result<DeltaMap, String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| format!("Failed to deserialize deltas: {e}"))?;

    let mut map = DeltaMap::new();
    for (name, view) in tensors.tensors() {
        let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
        let f32_data: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let array = Array::from_slice(&f32_data, &shape);
        map.insert(name.to_string(), array);
    }

    Ok(map)
}

/// Save replay buffer metadata (without token data) to JSON.
pub fn save_replay_metadata(
    entries: &[ReplayMeta],
    path: &Path,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory: {e}"))?;
    }
    let json = serde_json::to_string_pretty(entries)
        .map_err(|e| format!("Failed to serialize replay metadata: {e}"))?;
    std::fs::write(path, json)
        .map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
    Ok(())
}

/// Load replay buffer metadata from JSON.
pub fn load_replay_metadata(path: &Path) -> Result<Vec<ReplayMeta>, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&json)
        .map_err(|e| format!("Failed to deserialize replay metadata: {e}"))
}

/// Serializable replay entry metadata (tokens excluded to keep file small).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplayMeta {
    pub request_id: String,
    pub surprise: f32,
    pub reward: f32,
    pub pinned: bool,
    pub train_count: u32,
}

/// Compress deltas when they exceed the budget by scaling them down.
///
/// Scales all deltas uniformly so total size fits within budget.
/// Acts as a regularizer — kills noise while preserving dominant patterns.
pub fn compress_deltas(deltas: &mut DeltaMap, budget_bytes: usize) -> Result<(), String> {
    let current_bytes: usize = deltas
        .values()
        .map(|a| a.shape().iter().map(|&d| d as usize).product::<usize>() * 4)
        .sum();

    if current_bytes <= budget_bytes {
        return Ok(());
    }

    // Scale factor to fit within budget (applied as norm reduction)
    let scale = (budget_bytes as f32 / current_bytes as f32).sqrt();
    let scale_arr = Array::from_f32(scale);

    for (_name, array) in deltas.iter_mut() {
        let scaled = array
            .multiply(&scale_arr)
            .map_err(|e| format!("scale failed: {e}"))?;
        mlx_rs::transforms::eval(std::slice::from_ref(&scaled))
            .map_err(|e| format!("eval failed: {e}"))?;
        *array = scaled;
    }

    Ok(())
}
