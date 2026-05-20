use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use windows::{
    core::{GUID, PWSTR},
    Win32::{
        Media::MediaFoundation::{
            IMFAttributes, IMFMediaType, IMFSourceReader, MFCreateAttributes, MFCreateMediaType,
            MFCreateSourceReaderFromURL, MFMediaType_Video, MFShutdown, MFStartup,
            MFVideoFormat_ARGB32, MFSTARTUP_NOSOCKET, MF_API_VERSION, MF_MT_DEFAULT_STRIDE,
            MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_PD_DURATION,
            MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READERF_ERROR, MF_SOURCE_READER_ALL_STREAMS,
            MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            MF_SOURCE_READER_MEDIASOURCE,
        },
        System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED},
    },
};

use crate::{
    decoder::{DecodeOptions, VideoDecoder},
    frame::{MediaTime, VideoFrame},
};

pub struct MediaFoundationDecoder {
    source_reader: IMFSourceReader,
    input: PathBuf,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    duration: Option<MediaTime>,
    com_initialized: bool,
    mf_started: bool,
}

impl MediaFoundationDecoder {
    pub fn new(input: &Path, options: DecodeOptions) -> anyhow::Result<Self> {
        if !input.exists() {
            bail!("input file does not exist: {}", input.display());
        }

        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }
            .context("failed to initialize COM for Media Foundation")?;
        let mut decoder = Self {
            source_reader: unsafe { std::mem::zeroed() },
            input: input.to_path_buf(),
            width: 0,
            height: 0,
            bytes_per_row: 0,
            duration: None,
            com_initialized: true,
            mf_started: false,
        };

        let result = decoder.initialize(input, options);
        if result.is_err() {
            decoder.com_initialized = true;
        }

        result.map(|_| decoder)
    }

    fn initialize(&mut self, input: &Path, options: DecodeOptions) -> anyhow::Result<()> {
        unsafe { MFStartup(MF_API_VERSION, MFSTARTUP_NOSOCKET) }
            .context("failed to initialize Media Foundation")?;
        self.mf_started = true;

        let mut path_wide: Vec<u16> = input.to_string_lossy().encode_utf16().collect();
        path_wide.push(0);

        let attributes = unsafe {
            let mut attributes: Option<IMFAttributes> = None;
            MFCreateAttributes(&mut attributes, 1)
                .context("failed to create Media Foundation source reader attributes")?;
            let attributes =
                attributes.context("Media Foundation did not return source reader attributes")?;
            attributes
                .SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1)
                .context("failed to enable Media Foundation video processing")?;
            attributes
        };

        self.source_reader = unsafe {
            MFCreateSourceReaderFromURL(PWSTR(path_wide.as_mut_ptr()), &attributes)
                .context("failed to create Media Foundation source reader")?
        };
        self.duration = media_duration(&self.source_reader);

        unsafe {
            self.source_reader
                .SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)
                .context("failed to disable non-video streams")?;
            self.source_reader
                .SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, true)
                .context("failed to enable first video stream")?;
        }

        let (original_width, original_height) = native_dimensions(&self.source_reader)?;
        let target_size = resize_dimensions(original_width, original_height, options.max_dimension);
        let media_type = configure_bgra_format(&self.source_reader, target_size)?;
        let (width, height) = media_type_dimensions(&media_type)?;
        let bytes_per_row = media_type_stride(&media_type).unwrap_or(width.saturating_mul(4));

        self.width = width;
        self.height = height;
        self.bytes_per_row = bytes_per_row;
        Ok(())
    }
}

impl Drop for MediaFoundationDecoder {
    fn drop(&mut self) {
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

impl VideoDecoder for MediaFoundationDecoder {
    fn read_frame(&mut self) -> anyhow::Result<Option<VideoFrame>> {
        let mut stream_flags = 0;
        let mut timestamp_100ns = 0;
        let mut sample = None;

        unsafe {
            self.source_reader
                .ReadSample(
                    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                    0,
                    None,
                    Some(&mut stream_flags),
                    Some(&mut timestamp_100ns),
                    Some(&mut sample),
                )
                .with_context(|| {
                    format!(
                        "failed to read Media Foundation sample from {}",
                        self.input.display()
                    )
                })?;
        }

        if stream_flags & MF_SOURCE_READERF_ERROR.0 as u32 != 0 {
            bail!("Media Foundation reported an error while reading the input video");
        }
        if stream_flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
            return Ok(None);
        }

        let Some(sample) = sample else {
            return Ok(None);
        };

        let buffer = unsafe { sample.ConvertToContiguousBuffer() }
            .context("failed to convert Media Foundation sample to a contiguous buffer")?;

        let mut ptr = std::ptr::null_mut();
        let mut _max_len = 0;
        let mut current_len = 0;
        unsafe {
            buffer
                .Lock(&mut ptr, Some(&mut _max_len), Some(&mut current_len))
                .context("failed to lock Media Foundation sample buffer")?;
        }
        scopeguard::defer! {
            unsafe {
                let _ = buffer.Unlock();
            }
        }

        let required_len = self.bytes_per_row as usize * self.height as usize;
        if (current_len as usize) < required_len {
            bail!(
                "Media Foundation returned a frame buffer that is too short; got {} bytes, need {required_len}",
                current_len
            );
        }
        let source = unsafe { std::slice::from_raw_parts(ptr, required_len) };
        let data = source.to_vec();
        let time = MediaTime::new(timestamp_100ns, 10_000_000)?;

        VideoFrame::new_bgra(self.width, self.height, self.bytes_per_row, time, data).map(Some)
    }

    fn duration(&self) -> Option<MediaTime> {
        self.duration
    }
}

fn native_dimensions(source_reader: &IMFSourceReader) -> anyhow::Result<(u32, u32)> {
    let native_media_type = unsafe {
        source_reader.GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, 0)
    }
    .context("failed to get native video media type")?;

    media_type_dimensions(&native_media_type)
}

fn configure_bgra_format(
    source_reader: &IMFSourceReader,
    target_size: Option<(u32, u32)>,
) -> anyhow::Result<IMFMediaType> {
    let media_type = unsafe { MFCreateMediaType() }.context("failed to create video media type")?;

    unsafe {
        media_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .context("failed to set video media major type")?;
        media_type
            .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_ARGB32)
            .context("failed to set video media subtype")?;
        if let Some((width, height)) = target_size {
            mf_set_attribute_size(&media_type, &MF_MT_FRAME_SIZE, width, height)?;
        }
        source_reader
            .SetCurrentMediaType(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                None,
                &media_type,
            )
            .context("failed to set Media Foundation output media type")?;

        source_reader
            .GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32)
            .context("failed to get configured Media Foundation media type")
    }
}

fn media_type_dimensions(media_type: &IMFMediaType) -> anyhow::Result<(u32, u32)> {
    let frame_size = unsafe { media_type.GetUINT64(&MF_MT_FRAME_SIZE) }
        .context("failed to get video frame size")?;
    Ok(((frame_size >> 32) as u32, (frame_size & 0xFFFF_FFFF) as u32))
}

fn media_type_stride(media_type: &IMFMediaType) -> Option<u32> {
    unsafe { media_type.GetUINT32(&MF_MT_DEFAULT_STRIDE).ok() }
}

fn media_duration(source_reader: &IMFSourceReader) -> Option<MediaTime> {
    let value = unsafe {
        source_reader
            .GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE.0 as u32, &MF_PD_DURATION)
            .ok()
    }?;
    let duration_100ns = i64::try_from(&value).ok()?;
    MediaTime::new(duration_100ns, 10_000_000).ok()
}

fn resize_dimensions(width: u32, height: u32, max_dimension: Option<u32>) -> Option<(u32, u32)> {
    let max_dimension = max_dimension?;
    let largest = width.max(height);
    if largest <= max_dimension {
        return None;
    }

    let factor = largest as f32 / max_dimension as f32;
    let mut resized_width = (width as f32 / factor).floor().max(2.0) as u32;
    let mut resized_height = (height as f32 / factor).floor().max(2.0) as u32;

    if resized_width % 2 != 0 {
        resized_width -= 1;
    }
    if resized_height % 2 != 0 {
        resized_height -= 1;
    }

    Some((resized_width.max(2), resized_height.max(2)))
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
