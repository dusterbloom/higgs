// Build script — clippy strict lints don't apply here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Accelerate framework for cblas_sgemm (Apple's BLAS on AMX).
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Compile and link the CoreML bridge (Obj-C → CoreML framework).
    #[cfg(target_os = "macos")]
    {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let bridge_dir = manifest_dir.join("bridge").join("coreml");
        let bridge_src = bridge_dir.join("coreml_bridge.m");
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let lib_path = out_dir.join("libcoreml_bridge.a");

        // Compile Obj-C to static library (easier for cargo linking than dylib)
        let status = Command::new("xcrun")
            .args([
                "clang",
                "-O2",
                "-Wall",
                "-Wno-deprecated-declarations",
                "-fobjc-arc",
                "-fPIC",
                "-c",
                bridge_src.to_str().unwrap(),
                "-o",
            ])
            .arg(out_dir.join("coreml_bridge.o").to_str().unwrap())
            .status()
            .expect("failed to compile coreml_bridge.m");
        assert!(status.success(), "coreml_bridge.m compilation failed");

        let status = Command::new("ar")
            .args(["rcs"])
            .arg(lib_path.to_str().unwrap())
            .arg(out_dir.join("coreml_bridge.o").to_str().unwrap())
            .status()
            .expect("failed to create static library");
        assert!(status.success(), "ar failed");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=coreml_bridge");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rerun-if-changed={}", bridge_src.display());
        println!(
            "cargo:rerun-if-changed={}",
            bridge_dir.join("coreml_bridge.h").display()
        );
    }
}
