use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context};

use crate::frame::VideoFrame;

#[cfg(any(
    all(target_os = "macos", feature = "engine-onnx"),
    all(
        target_os = "windows",
        any(feature = "engine-onnx", feature = "windows-default-onnx")
    )
))]
mod onnx;

#[cfg(all(target_os = "macos", feature = "engine-metal"))]
mod metal;

pub trait Engine {
    fn segment(&mut self, frame: &VideoFrame) -> anyhow::Result<VideoFrame>;
}

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub model_path: PathBuf,
    pub downsample_ratio: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFormat {
    Onnx,
    RvmMetal,
}

pub fn create_engine(options: EngineOptions) -> anyhow::Result<Box<dyn Engine>> {
    match detect_model_format(&options.model_path)? {
        ModelFormat::Onnx => create_onnx_engine(options),
        ModelFormat::RvmMetal => create_metal_engine(options),
    }
}

pub fn engine_label_for_model(path: &Path) -> anyhow::Result<&'static str> {
    match detect_model_format(path)? {
        ModelFormat::Onnx => Ok("ONNX"),
        ModelFormat::RvmMetal => Ok("Metal"),
    }
}

fn detect_model_format(path: &Path) -> anyhow::Result<ModelFormat> {
    const RVM_METAL_MAGIC: &[u8; 8] = b"RVMMETAL";

    let mut file = File::open(path).with_context(|| {
        format!(
            "failed to open model for format detection: {}",
            path.display()
        )
    })?;
    let mut header = [0_u8; 8];
    let bytes_read = file
        .read(&mut header)
        .with_context(|| format!("failed to read model header: {}", path.display()))?;

    if bytes_read == header.len() && &header == RVM_METAL_MAGIC {
        Ok(ModelFormat::RvmMetal)
    } else {
        Ok(ModelFormat::Onnx)
    }
}

#[cfg(any(
    all(target_os = "macos", feature = "engine-onnx"),
    all(
        target_os = "windows",
        any(feature = "engine-onnx", feature = "windows-default-onnx")
    )
))]
fn create_onnx_engine(options: EngineOptions) -> anyhow::Result<Box<dyn Engine>> {
    Ok(Box::new(onnx::OnnxEngine::new(
        options.model_path,
        options.downsample_ratio,
    )?))
}

#[cfg(not(any(
    all(target_os = "macos", feature = "engine-onnx"),
    all(
        target_os = "windows",
        any(feature = "engine-onnx", feature = "windows-default-onnx")
    )
)))]
fn create_onnx_engine(_options: EngineOptions) -> anyhow::Result<Box<dyn Engine>> {
    bail!("the .onnx engine is not enabled; rebuild with --features engine-onnx")
}

#[cfg(all(target_os = "macos", feature = "engine-metal"))]
fn create_metal_engine(options: EngineOptions) -> anyhow::Result<Box<dyn Engine>> {
    Ok(Box::new(metal::MetalEngine::new(
        options.model_path,
        options.downsample_ratio,
    )?))
}

#[cfg(all(target_os = "macos", not(feature = "engine-metal")))]
fn create_metal_engine(_options: EngineOptions) -> anyhow::Result<Box<dyn Engine>> {
    bail!("the .rvmmetal engine is not enabled; rebuild with --features engine-metal")
}

#[cfg(not(target_os = "macos"))]
fn create_metal_engine(_options: EngineOptions) -> anyhow::Result<Box<dyn Engine>> {
    bail!("the .rvmmetal engine is only available on macOS with --features engine-metal")
}

#[cfg(test)]
mod tests {
    use super::{detect_model_format, ModelFormat};

    use std::{
        fs,
        io::Write,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn detects_rvm_metal_from_magic_header() {
        let path = write_model_file("renamed-model.bin", b"RVMMETAL\x02\0\0\0").unwrap();

        assert_eq!(detect_model_format(&path).unwrap(), ModelFormat::RvmMetal);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn treats_non_metal_files_as_onnx() {
        let path = write_model_file("renamed-model.rvmmetal", b"\x08\x06\x12\x07pytorch").unwrap();

        assert_eq!(detect_model_format(&path).unwrap(), ModelFormat::Onnx);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn treats_short_files_as_onnx() {
        let path = write_model_file("model-without-extension", b"RVM").unwrap();

        assert_eq!(detect_model_format(&path).unwrap(), ModelFormat::Onnx);

        let _ = fs::remove_file(path);
    }

    fn write_model_file(name: &str, contents: &[u8]) -> anyhow::Result<PathBuf> {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "segmenter-test-{}-{name}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        write_file(&path, contents)?;
        Ok(path)
    }

    fn write_file(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
        let mut file = fs::File::create(path)?;
        file.write_all(contents)?;
        Ok(())
    }
}
