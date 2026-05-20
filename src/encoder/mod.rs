use std::path::Path;

use crate::frame::VideoFrame;

pub trait VideoEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> anyhow::Result<()>;
    fn finalize(&mut self) -> anyhow::Result<()>;
}

#[cfg(target_os = "macos")]
mod avfoundation;

#[cfg(target_os = "windows")]
mod mediafoundation;

#[cfg(target_os = "macos")]
pub fn create_video_encoder(output: &Path) -> anyhow::Result<Box<dyn VideoEncoder>> {
    Ok(Box::new(avfoundation::AvFoundationEncoder::new(output)?))
}

#[cfg(target_os = "windows")]
pub fn create_video_encoder(output: &Path) -> anyhow::Result<Box<dyn VideoEncoder>> {
    Ok(Box::new(mediafoundation::MediaFoundationEncoder::new(
        output,
    )?))
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn create_video_encoder(_output: &Path) -> anyhow::Result<Box<dyn VideoEncoder>> {
    anyhow::bail!("native video encoding is only implemented on macOS and Windows")
}
