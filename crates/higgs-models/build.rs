fn main() {
    #[cfg(feature = "ane")]
    build_ane_bridge();
}

/// Compile the Objective-C ANE bridge when the `ane` feature is enabled.
///
/// Requires macOS with the private `AppleNeuralEngine.framework`.
/// The bridge wraps `_ANEInMemoryModel` and `_ANEClient` for direct ANE access.
#[cfg(feature = "ane")]
fn build_ane_bridge() {
    use std::env;
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bridge_dir = manifest_dir.join("bridge").join("ane");

    println!("cargo:rerun-if-changed=bridge/ane/ane_bridge.m");
    println!("cargo:rerun-if-changed=bridge/ane/ane_bridge.h");

    // Compile ane_bridge.m into a static library.
    let mut build = cc::Build::new();
    build
        .file(bridge_dir.join("ane_bridge.m"))
        .flag("-fobjc-arc")
        .flag("-fPIC")
        .flag("-O2")
        .flag("-Wno-deprecated-declarations")
        // Link Apple frameworks.
        .flag("-framework")
        .flag("Foundation")
        .flag("-framework")
        .flag("IOSurface")
        .flag("-framework")
        .flag("Accelerate")
        // dlopen is used to load the private ANE framework at runtime.
        .flag("-ldl")
        .include(&bridge_dir);

    build.compile("ane_bridge");

    println!("cargo:rustc-link-lib=static=ane_bridge");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOSurface");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=dylib=dl");
}
