//! Parity probe: run the higgs Qwen-VL tower on /tmp/parity_in.png and
//! write /tmp/higgs_feats.npy for comparison against the mlx-vlm reference.
fn main() {
    match higgs_models::qwen_vl::debug_vision_parity("/Users/peppi/AI-Models/qwen3-vl-4b-4bit") {
        Ok(msg) => println!("higgs tower: {msg}"),
        Err(e) => println!("ERR {e}"),
    }
}
