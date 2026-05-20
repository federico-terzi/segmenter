use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use windows::{
    core::{GUID, PCWSTR},
    Win32::{
        Media::MediaFoundation::{
            IMFAttributes, IMFMediaType, IMFSinkWriter, MFCreateMediaType, MFCreateMemoryBuffer,
            MFCreateSample, MFCreateSinkWriterFromURL, MFMediaType_Video, MFShutdown, MFStartup,
            MFVideoFormat_ARGB32, MFVideoFormat_H264, MFVideoInterlace_Progressive,
            MFSTARTUP_NOSOCKET, MF_API_VERSION, MF_MT_AVG_BITRATE, MF_MT_DEFAULT_STRIDE,
            MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE,
            MF_MT_PIXEL_ASPECT_RATIO, MF_MT_SUBTYPE,
        },
        System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED},
    },
};

use crate::{
    encoder::VideoEncoder,
    frame::{MediaTime, PixelFormat, VideoFrame},
};

const DEFAULT_FINAL_FRAME_DURATION_HNS: i64 = 10_000_000 / 30;
const NOMINAL_FRAME_RATE_NUMERATOR: u32 = 30;
const NOMINAL_FRAME_RATE_DENOMINATOR: u32 = 1;
const DEFAULT_BYTES_PER_PIXEL_PER_SECOND: u32 = 4;
const MIN_AVG_BITRATE: u32 = 1_000_000;
const MAX_AVG_BITRATE: u32 = 20_000_000;

pub struct MediaFoundationEncoder {
    output: PathBuf,
    sink_writer: Option<IMFSinkWriter>,
    stream_index: Option<u32>,
    width: u32,
    height: u32,
    pending_frame: Option<PendingFrame>,
    last_duration_hns: Option<i64>,
    finalized: bool,
    com_initialized: bool,
    mf_started: bool,
}

impl MediaFoundationEncoder {
    pub fn new(output: &Path) -> anyhow::Result<Self> {
        validate_output_extension(output)?;
        if output.exists() {
            std::fs::remove_file(output).with_context(|| {
                format!("failed to remove existing output {}", output.display())
            })?;
        }

        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }
            .ok()
            .context("failed to initialize COM for Media Foundation encoder")?;
        let mut encoder = Self {
            output: output.to_path_buf(),
            sink_writer: None,
            stream_index: None,
            width: 0,
            height: 0,
            pending_frame: None,
            last_duration_hns: None,
            finalized: false,
            com_initialized: true,
            mf_started: false,
        };

        encoder.start_media_foundation()?;
        Ok(encoder)
    }

    fn start_media_foundation(&mut self) -> anyhow::Result<()> {
        unsafe { MFStartup(MF_API_VERSION, MFSTARTUP_NOSOCKET) }
            .context("failed to initialize Media Foundation encoder")?;
        self.mf_started = true;
        Ok(())
    }

    fn initialize(&mut self, frame: &VideoFrame) -> anyhow::Result<()> {
        let mut path_wide: Vec<u16> = self.output.to_string_lossy().encode_utf16().collect();
        path_wide.push(0);

        let sink_writer =
            unsafe { MFCreateSinkWriterFromURL(PCWSTR(path_wide.as_ptr()), None, None) }
                .with_context(|| {
                    format!(
                        "failed to create Media Foundation sink writer for {}",
                        self.output.display()
                    )
                })?;

        let output_type = create_h264_output_type(frame.width, frame.height)?;
        let stream_index = unsafe { sink_writer.AddStream(&output_type) }
            .context("failed to add H.264 stream to Media Foundation sink writer")?;

        let input_type = create_bgra_input_type(frame.width, frame.height, frame.bytes_per_row)?;
        unsafe {
            sink_writer
                .SetInputMediaType(stream_index, &input_type, None)
                .context("failed to set Media Foundation sink writer input type")?;
            sink_writer
                .BeginWriting()
                .context("failed to begin writing Media Foundation output")?;
        }

        self.width = frame.width;
        self.height = frame.height;
        self.stream_index = Some(stream_index);
        self.sink_writer = Some(sink_writer);
        Ok(())
    }

    fn write_pending_frame(&mut self, duration_hns: i64) -> anyhow::Result<()> {
        if duration_hns <= 0 {
            bail!("encoded frame duration must be positive, got {duration_hns}");
        }

        let pending = self
            .pending_frame
            .take()
            .context("encoder did not have a pending frame to write")?;
        let sink_writer = self
            .sink_writer
            .as_ref()
            .context("Media Foundation sink writer was not initialized")?;
        let stream_index = self
            .stream_index
            .context("Media Foundation sink writer stream was not initialized")?;
        let sample = create_sample(&pending, duration_hns)?;

        unsafe {
            sink_writer
                .WriteSample(stream_index, &sample)
                .context("failed to write Media Foundation video sample")?;
        }
        self.last_duration_hns = Some(duration_hns);
        Ok(())
    }
}

impl Drop for MediaFoundationEncoder {
    fn drop(&mut self) {
        self.sink_writer = None;
        unsafe {
            if self.mf_started {
                let _ = MFShutdown();
            }
            if self.com_initialized {
                CoUninitialize();
            }
        }
    }
}

impl VideoEncoder for MediaFoundationEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> anyhow::Result<()> {
        if self.finalized {
            bail!("encoder has already been finalized");
        }
        if frame.format != PixelFormat::Bgra {
            bail!("Media Foundation encoder only accepts BGRA frames");
        }
        if self.sink_writer.is_none() {
            self.initialize(frame)?;
        }
        if frame.width != self.width || frame.height != self.height {
            bail!(
                "all encoded frames must have the same dimensions; encoder is {}x{}, frame is {}x{}",
                self.width,
                self.height,
                frame.width,
                frame.height
            );
        }

        let current_time_hns = media_time_to_hns(frame.time)?;
        if let Some(pending) = self.pending_frame.as_ref() {
            let duration_hns = current_time_hns - pending.time_hns;
            if duration_hns <= 0 {
                bail!(
                    "video frame timestamps must be strictly increasing; previous PTS was {}, current PTS is {}",
                    pending.time_hns,
                    current_time_hns
                );
            }
            self.write_pending_frame(duration_hns)?;
        }

        self.pending_frame = Some(PendingFrame {
            data: frame.data.clone(),
            time_hns: current_time_hns,
        });
        Ok(())
    }

    fn finalize(&mut self) -> anyhow::Result<()> {
        if self.finalized {
            bail!("encoder has already been finalized");
        }
        if self.sink_writer.is_none() {
            bail!(
                "cannot finalize encoder for {} before sending any frames",
                self.output.display()
            );
        }

        if self.pending_frame.is_some() {
            let duration_hns = self
                .last_duration_hns
                .unwrap_or(DEFAULT_FINAL_FRAME_DURATION_HNS);
            self.write_pending_frame(duration_hns)?;
        }

        let sink_writer = self
            .sink_writer
            .as_ref()
            .context("Media Foundation sink writer was not initialized")?;
        unsafe {
            sink_writer
                .Finalize()
                .context("failed to finalize Media Foundation output")?;
        }
        self.finalized = true;
        Ok(())
    }
}

unsafe impl Send for MediaFoundationEncoder {}

struct PendingFrame {
    data: Vec<u8>,
    time_hns: i64,
}

fn validate_output_extension(output: &Path) -> anyhow::Result<()> {
    match output
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .as_deref()
    {
        Some("mp4") => Ok(()),
        Some("mov") => bail!("Windows video encoding currently supports .mp4 output only"),
        _ => bail!("unsupported output extension for Media Foundation encoder; use .mp4"),
    }
}

fn create_h264_output_type(width: u32, height: u32) -> anyhow::Result<IMFMediaType> {
    let media_type = unsafe { MFCreateMediaType() }.context("failed to create H.264 media type")?;
    let bitrate = avg_bitrate(width, height);

    unsafe {
        media_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .context("failed to set H.264 media major type")?;
        media_type
            .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264)
            .context("failed to set H.264 media subtype")?;
        media_type
            .SetUINT32(&MF_MT_AVG_BITRATE, bitrate)
            .context("failed to set H.264 average bitrate")?;
        media_type
            .SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)
            .context("failed to set H.264 interlace mode")?;
        mf_set_attribute_size(&media_type, &MF_MT_FRAME_SIZE, width, height)
            .context("failed to set H.264 frame size")?;
        mf_set_attribute_ratio(
            &media_type,
            &MF_MT_FRAME_RATE,
            NOMINAL_FRAME_RATE_NUMERATOR,
            NOMINAL_FRAME_RATE_DENOMINATOR,
        )
        .context("failed to set H.264 nominal frame rate")?;
        mf_set_attribute_ratio(&media_type, &MF_MT_PIXEL_ASPECT_RATIO, 1, 1)
            .context("failed to set H.264 pixel aspect ratio")?;
    }

    Ok(media_type)
}

fn create_bgra_input_type(
    width: u32,
    height: u32,
    bytes_per_row: u32,
) -> anyhow::Result<IMFMediaType> {
    let media_type = unsafe { MFCreateMediaType() }.context("failed to create BGRA media type")?;

    unsafe {
        media_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .context("failed to set BGRA media major type")?;
        media_type
            .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_ARGB32)
            .context("failed to set BGRA media subtype")?;
        media_type
            .SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)
            .context("failed to set BGRA interlace mode")?;
        media_type
            .SetUINT32(&MF_MT_DEFAULT_STRIDE, bytes_per_row)
            .context("failed to set BGRA stride")?;
        mf_set_attribute_size(&media_type, &MF_MT_FRAME_SIZE, width, height)
            .context("failed to set BGRA frame size")?;
        mf_set_attribute_ratio(
            &media_type,
            &MF_MT_FRAME_RATE,
            NOMINAL_FRAME_RATE_NUMERATOR,
            NOMINAL_FRAME_RATE_DENOMINATOR,
        )
        .context("failed to set BGRA nominal frame rate")?;
        mf_set_attribute_ratio(&media_type, &MF_MT_PIXEL_ASPECT_RATIO, 1, 1)
            .context("failed to set BGRA pixel aspect ratio")?;
    }

    Ok(media_type)
}

fn create_sample(
    frame: &PendingFrame,
    duration_hns: i64,
) -> anyhow::Result<windows::Win32::Media::MediaFoundation::IMFSample> {
    let buffer_len = u32::try_from(frame.data.len())
        .context("frame buffer is too large for Media Foundation")?;
    let buffer = unsafe { MFCreateMemoryBuffer(buffer_len) }
        .context("failed to create Media Foundation frame buffer")?;
    let mut target = std::ptr::null_mut();

    unsafe {
        buffer
            .Lock(&mut target, None, None)
            .context("failed to lock Media Foundation frame buffer")?;
    }
    scopeguard::defer! {
        unsafe {
            let _ = buffer.Unlock();
        }
    }

    if target.is_null() {
        bail!("Media Foundation returned a null frame buffer");
    }
    unsafe {
        std::ptr::copy_nonoverlapping(frame.data.as_ptr(), target, frame.data.len());
        buffer
            .SetCurrentLength(buffer_len)
            .context("failed to set Media Foundation frame buffer length")?;
    }

    let sample = unsafe { MFCreateSample() }.context("failed to create Media Foundation sample")?;
    unsafe {
        sample
            .SetSampleTime(frame.time_hns)
            .context("failed to set Media Foundation sample time")?;
        sample
            .SetSampleDuration(duration_hns)
            .context("failed to set Media Foundation sample duration")?;
        sample
            .AddBuffer(&buffer)
            .context("failed to attach frame buffer to Media Foundation sample")?;
    }

    Ok(sample)
}

fn media_time_to_hns(time: MediaTime) -> anyhow::Result<i64> {
    let value = (time.value as i128)
        .checked_mul(10_000_000)
        .context("media timestamp overflowed during conversion")?
        / time.timescale as i128;
    i64::try_from(value).context("media timestamp is outside Media Foundation range")
}

fn avg_bitrate(width: u32, height: u32) -> u32 {
    width
        .saturating_mul(height)
        .saturating_mul(DEFAULT_BYTES_PER_PIXEL_PER_SECOND)
        .clamp(MIN_AVG_BITRATE, MAX_AVG_BITRATE)
}

#[allow(non_snake_case)]
unsafe fn mf_set_attribute_size(
    attributes: &IMFAttributes,
    key: &GUID,
    width: u32,
    height: u32,
) -> windows::core::Result<()> {
    attributes.SetUINT64(key, ((width as u64) << 32) | height as u64)
}

#[allow(non_snake_case)]
unsafe fn mf_set_attribute_ratio(
    attributes: &IMFAttributes,
    key: &GUID,
    numerator: u32,
    denominator: u32,
) -> windows::core::Result<()> {
    attributes.SetUINT64(key, ((numerator as u64) << 32) | denominator as u64)
}
