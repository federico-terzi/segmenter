mod native;

use std::{
    ffi::{c_void, CString},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Context};
use native::{
    sege_finalize_asset_writer, sege_finalize_asset_writer_input, sege_initialize_asset_writer,
    sege_initialize_asset_writer_input, sege_release_asset_writer, sege_release_asset_writer_input,
    sege_send_video_sample, sege_start_asset_writer, sege_wait_for_asset_writer_input_ready,
    SEGEncodeTime, SEGEncodeVideoSample, SEGEncodeVideoSamplePlane, SEGEncodeWriterInputOptions,
    SEGEncodeWriterOptions, SEGE_FORMAT_MOV, SEGE_FORMAT_MP4, SEGE_SUCCESS, SEGE_VIDEO_CODEC_H264,
    SEGE_VIDEO_FORMAT_BGRA,
};
use objc::rc::autoreleasepool;

use crate::{
    encoder::VideoEncoder,
    frame::{PixelFormat, VideoFrame},
};

pub struct AvFoundationEncoder {
    writer: Arc<AvAssetWriter>,
    input: Option<Arc<AvAssetWriterInput>>,
    output: PathBuf,
    finalized: bool,
}

impl AvFoundationEncoder {
    pub fn new(output: &Path) -> anyhow::Result<Self> {
        if output.exists() {
            std::fs::remove_file(output).with_context(|| {
                format!("failed to remove existing output {}", output.display())
            })?;
        }

        Ok(Self {
            writer: Arc::new(AvAssetWriter::new(output)?),
            input: None,
            output: output.to_path_buf(),
            finalized: false,
        })
    }

    fn initialize(&mut self, frame: &VideoFrame) -> anyhow::Result<()> {
        let input = Arc::new(AvAssetWriterInput::new(
            self.writer.clone(),
            frame.width,
            frame.height,
        )?);
        self.writer.start(frame.time)?;
        self.input = Some(input);
        Ok(())
    }
}

impl VideoEncoder for AvFoundationEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> anyhow::Result<()> {
        autoreleasepool(|| {
            if self.finalized {
                bail!("encoder has already been finalized");
            }
            if frame.format != PixelFormat::Bgra {
                bail!("AVFoundation encoder only accepts BGRA frames");
            }
            if self.input.is_none() {
                self.initialize(frame)?;
            }

            let input = self
                .input
                .as_ref()
                .context("encoder input was not initialized")?;
            if input.width != frame.width || input.height != frame.height {
                bail!(
                    "all encoded frames must have the same dimensions; encoder is {}x{}, frame is {}x{}",
                    input.width,
                    input.height,
                    frame.width,
                    frame.height
                );
            }
            input.send_frame(frame)
        })
    }

    fn finalize(&mut self) -> anyhow::Result<()> {
        autoreleasepool(|| {
            if self.finalized {
                bail!("encoder has already been finalized");
            }
            let input = self.input.as_ref().with_context(|| {
                format!(
                    "cannot finalize encoder for {} before sending any frames",
                    self.output.display()
                )
            })?;

            input.finalize()?;
            self.writer.finalize()?;
            self.finalized = true;
            Ok(())
        })
    }
}

unsafe impl Send for AvFoundationEncoder {}

struct AvAssetWriter {
    ptr: *mut c_void,
    _options: Arc<SEGEncodeWriterOptions>,
}

impl AvAssetWriter {
    fn new(output: &Path) -> anyhow::Result<Self> {
        let native_file_path =
            CString::new(output.to_string_lossy().as_bytes()).with_context(|| {
                format!(
                    "output path contained an interior NUL: {}",
                    output.display()
                )
            })?;
        let native_file_path_bytes = native_file_path.as_bytes_with_nul();
        let mut file_path = [0; 1024];
        if native_file_path_bytes.len() > file_path.len() {
            bail!(
                "output path is too long for AVFoundation bridge: {}",
                output.display()
            );
        }
        for (slot, byte) in file_path
            .iter_mut()
            .zip(native_file_path_bytes.iter().copied())
        {
            *slot = byte as i8;
        }

        let format = match output
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref()
        {
            Some("mp4") => SEGE_FORMAT_MP4,
            Some("mov") => SEGE_FORMAT_MOV,
            _ => bail!("unsupported output extension for AVFoundation encoder"),
        };

        let options = Arc::new(SEGEncodeWriterOptions { file_path, format });
        let mut error_code = 0;
        let ptr = unsafe { sege_initialize_asset_writer(options.as_ref(), &mut error_code) };
        if ptr.is_null() {
            bail!("failed to initialize AVAssetWriter: {error_code}");
        }

        Ok(Self {
            ptr,
            _options: options,
        })
    }

    fn start(&self, time: crate::frame::MediaTime) -> anyhow::Result<()> {
        let native_time = SEGEncodeTime {
            value: time.value,
            timescale: time.timescale,
        };
        let result = unsafe { sege_start_asset_writer(self.ptr, &native_time) };
        if result != SEGE_SUCCESS {
            bail!("failed to start AVAssetWriter: {result}");
        }

        Ok(())
    }

    fn finalize(&self) -> anyhow::Result<()> {
        let result = unsafe { sege_finalize_asset_writer(self.ptr) };
        if result != SEGE_SUCCESS {
            bail!("failed to finalize AVAssetWriter: {result}");
        }

        Ok(())
    }
}

impl Drop for AvAssetWriter {
    fn drop(&mut self) {
        unsafe { sege_release_asset_writer(self.ptr) };
    }
}

struct AvAssetWriterInput {
    ptr: *mut c_void,
    writer: Arc<AvAssetWriter>,
    width: u32,
    height: u32,
    _options: Arc<SEGEncodeWriterInputOptions>,
}

impl AvAssetWriterInput {
    fn new(writer: Arc<AvAssetWriter>, width: u32, height: u32) -> anyhow::Result<Self> {
        let options = Arc::new(SEGEncodeWriterInputOptions {
            asset_writer: writer.ptr,
            video_codec: SEGE_VIDEO_CODEC_H264,
            video_width: width,
            video_height: height,
        });
        let mut error_code = 0;
        let ptr = unsafe { sege_initialize_asset_writer_input(options.as_ref(), &mut error_code) };
        if ptr.is_null() {
            bail!("failed to initialize AVAssetWriterInput: {error_code}");
        }

        Ok(Self {
            ptr,
            writer,
            width,
            height,
            _options: options,
        })
    }

    fn send_frame(&self, frame: &VideoFrame) -> anyhow::Result<()> {
        let ready = unsafe { sege_wait_for_asset_writer_input_ready(self.ptr) };
        if ready != SEGE_SUCCESS {
            bail!("failed waiting for AVAssetWriterInput readiness: {ready}");
        }

        let sample = SEGEncodeVideoSample {
            format: SEGE_VIDEO_FORMAT_BGRA,
            width: frame.width,
            height: frame.height,
            planes: [SEGEncodeVideoSamplePlane {
                data: frame.data.as_ptr() as *const c_void,
                size: frame.data.len() as u64,
                bytes_per_row: frame.bytes_per_row,
            }],
            planes_count: 1,
            pts: SEGEncodeTime {
                value: frame.time.value,
                timescale: frame.time.timescale,
            },
        };

        let result = unsafe { sege_send_video_sample(self.writer.ptr, self.ptr, &sample) };
        if result != SEGE_SUCCESS {
            bail!("failed to send video sample to AVAssetWriterInput: {result}");
        }

        Ok(())
    }

    fn finalize(&self) -> anyhow::Result<()> {
        let result = unsafe { sege_finalize_asset_writer_input(self.ptr) };
        if result != SEGE_SUCCESS {
            bail!("failed to finalize AVAssetWriterInput: {result}");
        }

        Ok(())
    }
}

impl Drop for AvAssetWriterInput {
    fn drop(&mut self) {
        unsafe { sege_release_asset_writer_input(self.ptr) };
    }
}
