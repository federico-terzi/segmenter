use std::path::{Path, PathBuf};

use anyhow::{bail, Context};

use crate::frame::VideoFrame;

mod onnx;

#[cfg(target_os = "macos")]
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
        "onnx" => Ok(Box::new(onnx::OnnxEngine::new(
            options.model_path,
            options.downsample_ratio,
        )?)),
        "rvmmetal" => create_metal_engine(options),
        extension => bail!("unsupported model extension .{extension}; use .onnx or .rvmmetal"),
    }
}

fn model_extension(path: &Path) -> anyhow::Result<String> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .with_context(|| format!("model path must have an extension: {}", path.display()))
}

#[cfg(target_os = "macos")]
fn create_metal_engine(options: EngineOptions) -> anyhow::Result<Box<dyn Engine>> {
    Ok(Box::new(metal::MetalEngine::new(
        options.model_path,
        options.downsample_ratio,
    )?))
}

#[cfg(not(target_os = "macos"))]
fn create_metal_engine(_options: EngineOptions) -> anyhow::Result<Box<dyn Engine>> {
    bail!("the .rvmmetal engine is only available on macOS")
}
