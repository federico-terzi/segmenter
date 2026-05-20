use std::path::Path;

use crate::frame::{MediaTime, VideoFrame};

pub trait VideoDecoder {
    fn read_frame(&mut self) -> anyhow::Result<Option<VideoFrame>>;

    fn duration(&self) -> Option<MediaTime> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeOptions {
    pub max_dimension: Option<u32>,
}

#[cfg(target_os = "macos")]
mod avfoundation;

#[cfg(target_os = "windows")]
mod mediafoundation;

#[cfg(target_os = "macos")]
pub fn open_video_decoder(
    input: &Path,
    options: DecodeOptions,
) -> anyhow::Result<Box<dyn VideoDecoder>> {
    Ok(Box::new(avfoundation::AvFoundationDecoder::new(
        input, options,
    )?))
}

#[cfg(target_os = "windows")]
pub fn open_video_decoder(
    input: &Path,
    options: DecodeOptions,
) -> anyhow::Result<Box<dyn VideoDecoder>> {
    Ok(Box::new(mediafoundation::MediaFoundationDecoder::new(
        input, options,
    )?))
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn open_video_decoder(
    _input: &Path,
    _options: DecodeOptions,
) -> anyhow::Result<Box<dyn VideoDecoder>> {
    anyhow::bail!("native video decoding is only implemented on macOS and Windows")
}
