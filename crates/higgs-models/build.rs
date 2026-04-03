fn main() {
    #[cfg(feature = "ane")]
    {
        println!("cargo:rerun-if-changed=bridge/ane/ane_bridge.m");
        println!("cargo:rerun-if-changed=bridge/ane/ane_bridge.h");

        cc::Build::new()
            .file("bridge/ane/ane_bridge.m")
            .include("bridge/ane")
            .flag("-fobjc-arc")
            .flag("-fmodules")
            .compile("ane_bridge");

        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
