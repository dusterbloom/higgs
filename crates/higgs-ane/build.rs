fn main() {
    // Accelerate framework for cblas_sgemm (Apple's BLAS on AMX).
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
