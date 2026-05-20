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

    cc::Build::new()
        .cpp(true)
        .file("src/decoder/avfoundation/native.mm")
        .file("src/encoder/avfoundation/native.mm")
        .compile("segmenter_native");

    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-lib=static=segmenter_native");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=CoreMedia");
    println!("cargo:rustc-link-lib=framework=CoreVideo");
    println!("cargo:rustc-link-lib=framework=Foundation");
}

#[cfg(not(target_os = "macos"))]
fn cc_config() {}

fn main() {
    cc_config();
}
