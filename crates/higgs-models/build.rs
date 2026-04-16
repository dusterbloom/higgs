fn main() {
    #[cfg(feature = "ane")]
    {
        println!("cargo:rerun-if-changed=bridge/ane/ane_bridge.m");
        println!("cargo:rerun-if-changed=bridge/ane/ane_bridge.h");
        println!("cargo:rerun-if-changed=bridge/ane/ane_bridge_mlmodel.m");
        println!("cargo:rerun-if-changed=bridge/ane/ane_bridge_mlmodel.h");

        cc::Build::new()
            .file("bridge/ane/ane_bridge.m")
            .file("bridge/ane/ane_bridge_mlmodel.m")
            .include("bridge/ane")
            .flag("-fobjc-arc")
            .flag("-fmodules")
            .compile("ane_bridge");

        println!("cargo:rustc-link-lib=framework=Foundation");
        // Public CoreML framework — needed by the lm_head LUT6 path
        // (see `ane_bridge_mlmodel.m`). The private-API bridge in
        // `ane_bridge.m` only needs Foundation + dlopen of the
        // AppleNeuralEngine private framework.
        println!("cargo:rustc-link-lib=framework=CoreML");
    }
}
