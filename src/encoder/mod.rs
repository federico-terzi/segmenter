use std::path::Path;

use crate::frame::{MediaTime, VideoFrame, VideoFrameInfo};

#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeOptions {
    #[cfg(target_os = "macos")]
    pub h264_fast: bool,
}

pub trait VideoEncoder {
    /// Optional early initialization from the engine's guaranteed output layout.
    fn prepare(&mut self, _info: VideoFrameInfo) -> anyhow::Result<()> {
        Ok(())
    }

    fn set_expected_duration(&mut self, _duration: Option<MediaTime>) -> anyhow::Result<()> {
        Ok(())
    }

    fn set_expected_frame_rate(&mut self, _rate: Option<f64>) -> anyhow::Result<()> {
        Ok(())
    }

    fn send_frame(&mut self, frame: &VideoFrame) -> anyhow::Result<()>;
    fn finalize(&mut self) -> anyhow::Result<()>;
}

#[cfg(target_os = "macos")]
mod avfoundation;

#[cfg(target_os = "windows")]
mod mediafoundation;

#[cfg(target_os = "macos")]
pub fn create_video_encoder(
    output: &Path,
    options: EncodeOptions,
) -> anyhow::Result<Box<dyn VideoEncoder>> {
    Ok(Box::new(avfoundation::AvFoundationEncoder::new(
        output, options,
    )?))
}

#[cfg(target_os = "windows")]
pub fn create_video_encoder(
    output: &Path,
    _options: EncodeOptions,
) -> anyhow::Result<Box<dyn VideoEncoder>> {
    Ok(Box::new(mediafoundation::MediaFoundationEncoder::new(
        output,
    )?))
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn create_video_encoder(
    _output: &Path,
    _options: EncodeOptions,
) -> anyhow::Result<Box<dyn VideoEncoder>> {
    anyhow::bail!("native video encoding is only implemented on macOS and Windows")
}
