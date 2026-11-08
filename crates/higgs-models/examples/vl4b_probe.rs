//! Probe: load the on-disk Qwen3-VL-4B group-64 checkpoint through the
//! same adapter path the engine uses. Run with HIGGS_BONSAI_TG_LUT4=0 to
//! skip the row4 promotion and expose the raw load result.
use std::path::Path;

fn main() {
    let dir = Path::new(
        "/Users/peppi/AI-Models/qwen3-vl-4b-4bit",
    );
    let detected = higgs_models::adapter::detect(dir).expect("detect failed");
    println!(
        "detected model_type={} adapter-compatible",
        detected.model_type
    );
    let selected = higgs_models::adapter::resolve(&detected).expect("resolve failed");
    let start = std::time::Instant::now();
    match selected.load(&detected, false) {
        Ok(_) => println!("LOADED OK in {:?}", start.elapsed()),
        Err(e) => println!("LOAD FAILED after {:?}: {}", start.elapsed(), e),
    }
}
