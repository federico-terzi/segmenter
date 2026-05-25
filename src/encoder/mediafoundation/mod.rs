use std::{
    collections::VecDeque,
    mem::ManuallyDrop,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context};
use windows::{
    core::{Interface, GUID, PCWSTR, VARIANT},
    Win32::{
        Media::MediaFoundation::{
            CLSID_MSH264EncoderMFT, CODECAPI_AVEncMPVDefaultBPictureCount,
            CODECAPI_AVLowLatencyMode, ICodecAPI, IMFAttributes, IMFMediaType, IMFSinkWriter,
            IMFTransform, MFCreateMediaType, MFCreateMemoryBuffer, MFCreateSample,
            MFCreateSinkWriterFromURL, MFMediaType_Video, MFShutdown, MFStartup,
            MFVideoFormat_H264, MFVideoFormat_NV12, MFVideoInterlace_Progressive,
            MFSTARTUP_NOSOCKET, MFT_MESSAGE_COMMAND_DRAIN, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING,
            MFT_MESSAGE_NOTIFY_END_OF_STREAM, MFT_MESSAGE_NOTIFY_START_OF_STREAM,
            MFT_OUTPUT_DATA_BUFFER, MFT_OUTPUT_DATA_BUFFER_INCOMPLETE,
            MFT_OUTPUT_STREAM_PROVIDES_SAMPLES, MF_API_VERSION, MF_E_TRANSFORM_NEED_MORE_INPUT,
            MF_E_TRANSFORM_STREAM_CHANGE, MF_MT_AVG_BITRATE, MF_MT_DEFAULT_STRIDE,
            MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE,
            MF_MT_PIXEL_ASPECT_RATIO, MF_MT_SUBTYPE,
        },
        System::Com::{
            CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER,
            COINIT_MULTITHREADED,
        },
    },
};

use crate::{
    encoder::VideoEncoder,
    frame::{MediaTime, PixelFormat, VideoFrame},
};

const DEFAULT_FINAL_FRAME_DURATION_HNS: i64 = 10_000_000 / 30;
const DEFAULT_NOMINAL_FRAME_RATE_NUMERATOR: u32 = 30;
const DEFAULT_NOMINAL_FRAME_RATE_DENOMINATOR: u32 = 1;
const DEFAULT_MP4_TIMESCALE: u32 = 30_000;
const MP4_TIMESCALE_FRAME_RATE_DENOMINATOR: u32 = 1_000;
const MAX_CUSTOM_MP4_TIMESCALE: u32 = 90_000;
const DEFAULT_BYTES_PER_PIXEL_PER_SECOND: u32 = 4;
const MIN_AVG_BITRATE: u32 = 1_000_000;
const MAX_AVG_BITRATE: u32 = 20_000_000;

pub struct MediaFoundationEncoder {
    output: PathBuf,
    sink_writer: Option<IMFSinkWriter>,
    video_encoder: Option<IMFTransform>,
    stream_index: Option<u32>,
    width: u32,
    height: u32,
    encoder_output_buffer_len: u32,
    encoder_provides_output_samples: bool,
    encoded_timings: VecDeque<SampleTiming>,
    pending_frame: Option<PendingFrame>,
    last_duration_hns: Option<i64>,
    expected_duration_hns: Option<i64>,
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
            video_encoder: None,
            stream_index: None,
            width: 0,
            height: 0,
            encoder_output_buffer_len: 0,
            encoder_provides_output_samples: false,
            encoded_timings: VecDeque::new(),
            pending_frame: None,
            last_duration_hns: None,
            expected_duration_hns: None,
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
        if frame.width % 2 != 0 || frame.height % 2 != 0 {
            bail!(
                "Media Foundation H.264 encoding requires even dimensions, got {}x{}",
                frame.width,
                frame.height
            );
        }

        let nominal_frame_rate = nominal_frame_rate_for_timestamps(frame.time.timescale);
        let output_type = create_h264_output_type(frame.width, frame.height, nominal_frame_rate)?;
        let input_type = create_nv12_input_type(frame.width, frame.height, nominal_frame_rate)?;
        let video_encoder = create_h264_encoder(&output_type, &input_type)?;
        let output_type = unsafe { video_encoder.GetOutputCurrentType(0) }
            .unwrap_or_else(|_| output_type.clone());
        let output_stream_info = unsafe { video_encoder.GetOutputStreamInfo(0) }
            .context("failed to get H.264 encoder output stream info")?;
        let encoder_provides_output_samples =
            output_stream_info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 as u32 != 0;
        let encoder_output_buffer_len = if encoder_provides_output_samples {
            0
        } else {
            output_stream_info
                .cbSize
                .max(frame.width.saturating_mul(frame.height).saturating_mul(4))
        };

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

        let stream_index = unsafe { sink_writer.AddStream(&output_type) }
            .context("failed to add H.264 stream to Media Foundation sink writer")?;

        unsafe {
            sink_writer
                .SetInputMediaType(stream_index, &output_type, None)
                .context("failed to set Media Foundation sink writer input type")?;
            sink_writer
                .BeginWriting()
                .context("failed to begin writing Media Foundation output")?;
        }

        self.width = frame.width;
        self.height = frame.height;
        self.encoder_output_buffer_len = encoder_output_buffer_len;
        self.encoder_provides_output_samples = encoder_provides_output_samples;
        self.stream_index = Some(stream_index);
        self.video_encoder = Some(video_encoder);
        self.sink_writer = Some(sink_writer);
        Ok(())
    }

    fn encode_pending_frame(&mut self, duration_hns: i64) -> anyhow::Result<()> {
        if duration_hns <= 0 {
            bail!("encoded frame duration must be positive, got {duration_hns}");
        }

        let pending = self
            .pending_frame
            .take()
            .context("encoder did not have a pending frame to write")?;
        let video_encoder = self
            .video_encoder
            .as_ref()
            .context("Media Foundation H.264 encoder was not initialized")?;
        let nv12 = bgra_to_nv12(&pending, self.width, self.height)?;
        let sample = create_sample(&nv12, pending.time_hns, duration_hns)?;

        unsafe {
            video_encoder
                .ProcessInput(0, &sample, 0)
                .context("failed to send frame to Media Foundation H.264 encoder")?;
        }
        self.encoded_timings.push_back(SampleTiming {
            time_hns: pending.time_hns,
            duration_hns,
        });
        self.last_duration_hns = Some(duration_hns);
        self.drain_encoder(false)?;
        Ok(())
    }

    fn finish_encoder(&mut self) -> anyhow::Result<()> {
        let video_encoder = self
            .video_encoder
            .as_ref()
            .context("Media Foundation H.264 encoder was not initialized")?;

        unsafe {
            video_encoder
                .ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0)
                .context("failed to notify Media Foundation encoder end of stream")?;
            video_encoder
                .ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0)
                .context("failed to drain Media Foundation encoder")?;
        }
        self.drain_encoder(true)?;
        if !self.encoded_timings.is_empty() {
            bail!(
                "Media Foundation H.264 encoder did not produce {} queued frame(s)",
                self.encoded_timings.len()
            );
        }

        Ok(())
    }

    fn drain_encoder(&mut self, final_drain: bool) -> anyhow::Result<()> {
        let video_encoder = self
            .video_encoder
            .as_ref()
            .context("Media Foundation H.264 encoder was not initialized")?
            .clone();

        loop {
            let mut output_buffer = self.create_output_data_buffer()?;
            let mut status = 0;
            let result = unsafe {
                video_encoder.ProcessOutput(
                    0,
                    std::slice::from_mut(&mut output_buffer),
                    &mut status,
                )
            };

            match result {
                Ok(()) => {
                    let sample = take_output_sample(&mut output_buffer);
                    drop(take_output_events(&mut output_buffer));
                    let sample =
                        sample.context("Media Foundation H.264 encoder returned no sample")?;
                    self.write_encoded_sample(&sample)?;

                    if output_buffer.dwStatus & MFT_OUTPUT_DATA_BUFFER_INCOMPLETE.0 as u32 != 0 {
                        continue;
                    }
                }
                Err(error) if error.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => {
                    drop(take_output_sample(&mut output_buffer));
                    drop(take_output_events(&mut output_buffer));
                    break;
                }
                Err(error) if error.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                    drop(take_output_sample(&mut output_buffer));
                    drop(take_output_events(&mut output_buffer));
                    bail!("Media Foundation H.264 encoder changed output streams unexpectedly");
                }
                Err(error) => {
                    drop(take_output_sample(&mut output_buffer));
                    drop(take_output_events(&mut output_buffer));
                    return Err(error).context("failed to get Media Foundation encoded sample");
                }
            }
        }

        if final_drain && !self.encoded_timings.is_empty() {
            bail!(
                "Media Foundation H.264 encoder still has {} queued frame(s) after drain",
                self.encoded_timings.len()
            );
        }

        Ok(())
    }

    fn create_output_data_buffer(&self) -> anyhow::Result<MFT_OUTPUT_DATA_BUFFER> {
        let mut output_buffer = MFT_OUTPUT_DATA_BUFFER::default();
        if !self.encoder_provides_output_samples {
            let sample = create_empty_sample(self.encoder_output_buffer_len)?;
            output_buffer.pSample = ManuallyDrop::new(Some(sample));
        }

        Ok(output_buffer)
    }

    fn write_encoded_sample(
        &mut self,
        sample: &windows::Win32::Media::MediaFoundation::IMFSample,
    ) -> anyhow::Result<()> {
        let timing = self
            .encoded_timings
            .pop_front()
            .context("Media Foundation H.264 encoder produced an unexpected sample")?;
        let sink_writer = self
            .sink_writer
            .as_ref()
            .context("Media Foundation sink writer was not initialized")?;
        let stream_index = self
            .stream_index
            .context("Media Foundation sink writer stream was not initialized")?;

        unsafe {
            sample
                .SetSampleTime(timing.time_hns)
                .context("failed to set encoded sample time")?;
            sample
                .SetSampleDuration(timing.duration_hns)
                .context("failed to set encoded sample duration")?;
            sink_writer
                .WriteSample(stream_index, sample)
                .context("failed to write encoded sample to Media Foundation sink writer")?;
        }

        Ok(())
    }

    fn final_frame_duration_hns(&self) -> i64 {
        self.pending_frame
            .as_ref()
            .and_then(|pending| {
                let duration = self.expected_duration_hns? - pending.time_hns;
                (duration > 0).then_some(duration)
            })
            .or(self.last_duration_hns)
            .unwrap_or(DEFAULT_FINAL_FRAME_DURATION_HNS)
    }
}

impl Drop for MediaFoundationEncoder {
    fn drop(&mut self) {
        self.video_encoder = None;
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
    fn set_expected_duration(&mut self, duration: Option<MediaTime>) -> anyhow::Result<()> {
        self.expected_duration_hns = duration.map(media_time_to_hns).transpose()?;
        Ok(())
    }

    fn send_frame(&mut self, frame: &VideoFrame) -> anyhow::Result<()> {
        if self.finalized {
            bail!("encoder has already been finalized");
        }
        if frame.format != PixelFormat::Bgra {
            bail!("Media Foundation encoder only accepts BGRA frames");
        }
        if self.video_encoder.is_none() {
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
            self.encode_pending_frame(duration_hns)?;
        }

        self.pending_frame = Some(PendingFrame {
            data: frame.data.clone(),
            time_hns: current_time_hns,
            bytes_per_row: frame.bytes_per_row,
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
            let duration_hns = self.final_frame_duration_hns();
            self.encode_pending_frame(duration_hns)?;
        }
        self.finish_encoder()?;

        let sink_writer = self
            .sink_writer
            .as_ref()
            .context("Media Foundation sink writer was not initialized")?;
        let stream_index = self
            .stream_index
            .context("Media Foundation sink writer stream was not initialized")?;
        unsafe {
            sink_writer
                .NotifyEndOfSegment(stream_index)
                .context("failed to notify Media Foundation sink writer end of segment")?;
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
    bytes_per_row: u32,
}

struct SampleTiming {
    time_hns: i64,
    duration_hns: i64,
}

#[derive(Clone, Copy)]
struct NominalFrameRate {
    numerator: u32,
    denominator: u32,
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

fn nominal_frame_rate_for_timestamps(timestamp_timescale: i32) -> NominalFrameRate {
    let default = NominalFrameRate {
        numerator: DEFAULT_NOMINAL_FRAME_RATE_NUMERATOR,
        denominator: DEFAULT_NOMINAL_FRAME_RATE_DENOMINATOR,
    };
    let Ok(timestamp_timescale) = u32::try_from(timestamp_timescale) else {
        return default;
    };

    if timestamp_timescale == 0
        || DEFAULT_MP4_TIMESCALE % timestamp_timescale == 0
        || timestamp_timescale > MAX_CUSTOM_MP4_TIMESCALE
    {
        return default;
    }

    NominalFrameRate {
        numerator: timestamp_timescale,
        denominator: MP4_TIMESCALE_FRAME_RATE_DENOMINATOR,
    }
}

fn create_h264_output_type(
    width: u32,
    height: u32,
    nominal_frame_rate: NominalFrameRate,
) -> anyhow::Result<IMFMediaType> {
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
            nominal_frame_rate.numerator,
            nominal_frame_rate.denominator,
        )
        .context("failed to set H.264 nominal frame rate")?;
        mf_set_attribute_ratio(&media_type, &MF_MT_PIXEL_ASPECT_RATIO, 1, 1)
            .context("failed to set H.264 pixel aspect ratio")?;
    }

    Ok(media_type)
}

fn create_nv12_input_type(
    width: u32,
    height: u32,
    nominal_frame_rate: NominalFrameRate,
) -> anyhow::Result<IMFMediaType> {
    let media_type = unsafe { MFCreateMediaType() }.context("failed to create NV12 media type")?;
    let bytes_per_row = width;

    unsafe {
        media_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .context("failed to set NV12 media major type")?;
        media_type
            .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)
            .context("failed to set NV12 media subtype")?;
        media_type
            .SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)
            .context("failed to set NV12 interlace mode")?;
        media_type
            .SetUINT32(&MF_MT_DEFAULT_STRIDE, bytes_per_row)
            .context("failed to set NV12 stride")?;
        mf_set_attribute_size(&media_type, &MF_MT_FRAME_SIZE, width, height)
            .context("failed to set NV12 frame size")?;
        mf_set_attribute_ratio(
            &media_type,
            &MF_MT_FRAME_RATE,
            nominal_frame_rate.numerator,
            nominal_frame_rate.denominator,
        )
        .context("failed to set NV12 nominal frame rate")?;
        mf_set_attribute_ratio(&media_type, &MF_MT_PIXEL_ASPECT_RATIO, 1, 1)
            .context("failed to set NV12 pixel aspect ratio")?;
    }

    Ok(media_type)
}

fn create_h264_encoder(
    output_type: &IMFMediaType,
    input_type: &IMFMediaType,
) -> anyhow::Result<IMFTransform> {
    let video_encoder: IMFTransform = unsafe {
        CoCreateInstance(&CLSID_MSH264EncoderMFT, None, CLSCTX_INPROC_SERVER)
            .context("failed to create Media Foundation H.264 encoder")?
    };
    configure_h264_encoder(&video_encoder)?;
    unsafe {
        video_encoder
            .SetOutputType(0, output_type, 0)
            .context("failed to set Media Foundation H.264 encoder output type")?;
        video_encoder
            .SetInputType(0, input_type, 0)
            .context("failed to set Media Foundation H.264 encoder input type")?;
        video_encoder
            .ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)
            .context("failed to begin Media Foundation encoder streaming")?;
        video_encoder
            .ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)
            .context("failed to start Media Foundation encoder stream")?;
    }

    Ok(video_encoder)
}

fn configure_h264_encoder(video_encoder: &IMFTransform) -> anyhow::Result<()> {
    let codec_api: ICodecAPI = video_encoder
        .cast()
        .context("Media Foundation H.264 encoder does not expose ICodecAPI")?;
    set_codec_api_value(
        &codec_api,
        &CODECAPI_AVEncMPVDefaultBPictureCount,
        VARIANT::from(0_u32),
        "disable Media Foundation H.264 B-pictures",
    )?;
    set_codec_api_value(
        &codec_api,
        &CODECAPI_AVLowLatencyMode,
        VARIANT::from(true),
        "enable Media Foundation H.264 low-latency mode",
    )?;
    Ok(())
}

fn set_codec_api_value(
    codec_api: &ICodecAPI,
    key: &GUID,
    value: VARIANT,
    context: &'static str,
) -> anyhow::Result<()> {
    unsafe { codec_api.SetValue(key, &value) }.with_context(|| format!("failed to {context}"))
}

fn create_sample(
    data: &[u8],
    time_hns: i64,
    duration_hns: i64,
) -> anyhow::Result<windows::Win32::Media::MediaFoundation::IMFSample> {
    let buffer_len =
        u32::try_from(data.len()).context("frame buffer is too large for Media Foundation")?;
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
        std::ptr::copy_nonoverlapping(data.as_ptr(), target, data.len());
        buffer
            .SetCurrentLength(buffer_len)
            .context("failed to set Media Foundation frame buffer length")?;
    }

    let sample = unsafe { MFCreateSample() }.context("failed to create Media Foundation sample")?;
    unsafe {
        sample
            .SetSampleTime(time_hns)
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

fn create_empty_sample(
    buffer_len: u32,
) -> anyhow::Result<windows::Win32::Media::MediaFoundation::IMFSample> {
    if buffer_len == 0 {
        bail!("Media Foundation encoder reported a zero-length output buffer");
    }

    let buffer = unsafe { MFCreateMemoryBuffer(buffer_len) }
        .context("failed to create Media Foundation encoder output buffer")?;
    let sample = unsafe { MFCreateSample() }.context("failed to create Media Foundation sample")?;
    unsafe {
        sample
            .AddBuffer(&buffer)
            .context("failed to attach encoder output buffer to sample")?;
    }

    Ok(sample)
}

fn take_output_sample(
    output_buffer: &mut MFT_OUTPUT_DATA_BUFFER,
) -> Option<windows::Win32::Media::MediaFoundation::IMFSample> {
    unsafe { ManuallyDrop::take(&mut output_buffer.pSample) }
}

fn take_output_events(
    output_buffer: &mut MFT_OUTPUT_DATA_BUFFER,
) -> Option<windows::Win32::Media::MediaFoundation::IMFCollection> {
    unsafe { ManuallyDrop::take(&mut output_buffer.pEvents) }
}

fn bgra_to_nv12(frame: &PendingFrame, width: u32, height: u32) -> anyhow::Result<Vec<u8>> {
    let width = width as usize;
    let height = height as usize;
    let bytes_per_row = frame.bytes_per_row as usize;
    let y_plane_len = width
        .checked_mul(height)
        .context("NV12 luma plane size overflowed")?;
    let uv_plane_len = y_plane_len / 2;
    let mut nv12 = vec![0_u8; y_plane_len + uv_plane_len];

    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = bgra_pixel(frame, x, y, bytes_per_row)?;
            let (yy, _, _) = rgb_to_yuv_limited(r, g, b);
            nv12[y * width + x] = yy;
        }
    }

    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let mut u_sum = 0_u32;
            let mut v_sum = 0_u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let (r, g, b) = bgra_pixel(frame, x + dx, y + dy, bytes_per_row)?;
                    let (_, u, v) = rgb_to_yuv_limited(r, g, b);
                    u_sum += u as u32;
                    v_sum += v as u32;
                }
            }

            let uv_offset = y_plane_len + (y / 2) * width + x;
            nv12[uv_offset] = (u_sum / 4) as u8;
            nv12[uv_offset + 1] = (v_sum / 4) as u8;
        }
    }

    Ok(nv12)
}

fn bgra_pixel(
    frame: &PendingFrame,
    x: usize,
    y: usize,
    bytes_per_row: usize,
) -> anyhow::Result<(u8, u8, u8)> {
    let offset = y
        .checked_mul(bytes_per_row)
        .and_then(|offset| offset.checked_add(x.checked_mul(4)?))
        .context("BGRA frame offset overflowed during NV12 conversion")?;
    let b = *frame
        .data
        .get(offset)
        .context("BGRA frame was too short during NV12 conversion")?;
    let g = frame.data[offset + 1];
    let r = frame.data[offset + 2];
    Ok((r, g, b))
}

fn rgb_to_yuv_limited(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    (clamp_u8(y), clamp_u8(u), clamp_u8(v))
}

fn clamp_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn media_time_to_hns(time: MediaTime) -> anyhow::Result<i64> {
    let numerator = (time.value as i128)
        .checked_mul(10_000_000)
        .context("media timestamp overflowed during conversion")?;
    let timescale = time.timescale as i128;
    let value = if numerator >= 0 {
        (numerator + timescale / 2) / timescale
    } else {
        (numerator - timescale / 2) / timescale
    };
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
