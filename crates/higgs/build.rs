use std::{env, fs, path::PathBuf, process};

fn main() {
    // Find mlx.metallib in the mlx-sys build output and copy it next to the binary.
    // MLX's runtime uses dladdr to look for mlx.metallib next to the executable.
    let Ok(out_dir) = env::var("OUT_DIR").map(PathBuf::from) else {
        return;
    };

    // OUT_DIR is target/<profile>/build/<crate>-<hash>/out
    // Walk up to target/<profile>/build/ to search mlx-sys-*/out/
    let Some(build_dir) = out_dir.ancestors().nth(2) else {
        return;
    };

    let Ok(entries) = fs::read_dir(build_dir) else {
        return;
    };

    for entry in entries.flatten() {
        let entry_name = entry.file_name();
        let is_mlx_sys = entry_name
            .to_str()
            .is_some_and(|s| s.starts_with("mlx-sys-"));
        if !is_mlx_sys {
            continue;
        }

        let metallib = entry.path().join("out/build/lib/mlx.metallib");
        if !metallib.exists() {
            continue;
        }

        // Copy to target profile dir (e.g. target/release/) so the binary finds it via dladdr
        if let Some(profile_dir) = out_dir.ancestors().nth(3) {
            let dst = profile_dir.join("mlx.metallib");
            if let Err(err) = fs::copy(&metallib, &dst) {
                eprintln!(
                    "failed to copy {} to {}: {}",
                    metallib.display(),
                    dst.display(),
                    err
                );
                process::exit(1);
            }
            println!("cargo:warning=Copied mlx.metallib to {}", dst.display());
        }

        break;
    }
}
