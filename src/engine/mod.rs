use std::path::{Path, PathBuf};

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

pub fn create_engine(options: EngineOptions) -> anyhow::Result<Box<dyn Engine>> {
    match model_extension(&options.model_path)?.as_str() {
        "onnx" => create_onnx_engine(options),
        "rvmmetal" => create_metal_engine(options),
        extension => bail!("unsupported model extension .{extension}; use .onnx or .rvmmetal"),
    }
}

pub fn engine_label_for_model(path: &Path) -> anyhow::Result<&'static str> {
    match model_extension(path)?.as_str() {
        "onnx" => Ok("ONNX"),
        "rvmmetal" => Ok("Metal"),
        extension => bail!("unsupported model extension .{extension}; use .onnx or .rvmmetal"),
    }
}

fn model_extension(path: &Path) -> anyhow::Result<String> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .with_context(|| format!("model path must have an extension: {}", path.display()))
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
