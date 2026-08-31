//! GGUF model format support for higgs.
//!
//! Phase 1: Q4_K dequant + scalar dot-product kernel (correctness).
//! Phase 2: simdgroup GEMM (performance).
//! Phase 3: IQ4_XS, IQ3, IQ2 vector codebook kernels (70B unlock).

pub mod dequant;
pub mod model;
pub mod parser;
pub mod q4_k;
