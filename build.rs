#[cfg(target_os = "macos")]
fn generate_bindings(header: &str, output_file_name: &str) {
    use std::path::PathBuf;

    let bindings = bindgen::Builder::default()
        .header(header)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("unable to generate native bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is not set"));
    bindings
        .write_to_file(out_path.join(output_file_name))
        .expect("unable to write native bindings");
}

#[cfg(target_os = "macos")]
fn cc_config() {
    let engine_metal_enabled = std::env::var_os("CARGO_FEATURE_ENGINE_METAL").is_some();

    generate_bindings(
        "src/decoder/avfoundation/native.hpp",
        "decoder_avfoundation_bindings.rs",
    );
    generate_bindings(
        "src/encoder/avfoundation/native.hpp",
        "encoder_avfoundation_bindings.rs",
    );

    println!("cargo:rerun-if-changed=src/decoder/avfoundation/native.hpp");
    println!("cargo:rerun-if-changed=src/decoder/avfoundation/native.mm");
    println!("cargo:rerun-if-changed=src/encoder/avfoundation/native.hpp");
    println!("cargo:rerun-if-changed=src/encoder/avfoundation/native.mm");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ENGINE_METAL");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .flag("-std=c++17")
        .file("src/decoder/avfoundation/native.mm")
        .file("src/encoder/avfoundation/native.mm");

    if engine_metal_enabled {
        println!("cargo:rerun-if-changed=src/engine/metal/native.h");
        println!("cargo:rerun-if-changed=src/engine/metal/native.mm");
        build.file("src/engine/metal/native.mm");
    }

    build.compile("segmenter_native");

    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-lib=static=segmenter_native");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=CoreMedia");
    println!("cargo:rustc-link-lib=framework=CoreVideo");
    println!("cargo:rustc-link-lib=framework=Foundation");
    if engine_metal_enabled {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }
}

#[cfg(not(target_os = "macos"))]
fn cc_config() {}

fn main() {
    cc_config();
}
