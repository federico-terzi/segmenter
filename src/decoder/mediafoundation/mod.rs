use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context};
use windows::{
    core::{GUID, PWSTR},
    Win32::{
        Media::MediaFoundation::{
            IMFAttributes, IMFMediaType, IMFSourceReader, MFCreateAttributes, MFCreateMediaType,
            MFCreateSourceReaderFromURL, MFGetStrideForBitmapInfoHeader, MFMediaType_Video,
            MFShutdown, MFStartup, MFVideoFormat_RGB32, MFSTARTUP_NOSOCKET, MF_API_VERSION,
            MF_MT_DEFAULT_STRIDE, MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
            MF_PD_DURATION, MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READERF_ERROR,
            MF_SOURCE_READER_ALL_STREAMS, MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,
            MF_SOURCE_READER_FIRST_VIDEO_STREAM, MF_SOURCE_READER_MEDIASOURCE,
        },
        System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED},
    },
};

use crate::{
    decoder::{DecodeOptions, VideoDecoder},
    frame::{MediaTime, VideoFrame},
};

pub struct MediaFoundationDecoder {
    source_reader: Option<IMFSourceReader>,
    input: PathBuf,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    source_width: u32,
    source_height: u32,
    source_bytes_per_row: u32,
    software_resize: Option<(u32, u32)>,
    timestamp_clock: TimestampClock,
    duration: Option<MediaTime>,
    com_initialized: bool,
    mf_started: bool,
}

impl MediaFoundationDecoder {
    pub fn new(input: &Path, options: DecodeOptions) -> anyhow::Result<Self> {
        if !input.exists() {
            bail!("input file does not exist: {}", input.display());
        }

        let mp4_timing = mp4_video_timing(input).unwrap_or_default();

        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }
            .ok()
            .context("failed to initialize COM for Media Foundation")?;
        let mut decoder = Self {
            source_reader: None,
            input: input.to_path_buf(),
            width: 0,
            height: 0,
            bytes_per_row: 0,
            source_width: 0,
            source_height: 0,
            source_bytes_per_row: 0,
            software_resize: None,
            timestamp_clock: TimestampClock::new(mp4_timing.timescale),
            duration: None,
            com_initialized: true,
            mf_started: false,
        };

        let result = decoder.initialize(input, options, mp4_timing);
        result.map(|_| decoder)
    }

    fn initialize(
        &mut self,
        input: &Path,
        options: DecodeOptions,
        mp4_timing: Mp4VideoTiming,
    ) -> anyhow::Result<()> {
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

        self.source_reader = Some(unsafe {
            MFCreateSourceReaderFromURL(PWSTR(path_wide.as_mut_ptr()), &attributes)
                .context("failed to create Media Foundation source reader")?
        });
        let source_reader = self
            .source_reader
            .as_ref()
            .context("Media Foundation did not return a source reader")?;
        self.duration = media_duration(source_reader, mp4_timing);

        unsafe {
            source_reader
                .SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)
                .context("failed to disable non-video streams")?;
            source_reader
                .SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, true)
                .context("failed to enable first video stream")?;
        }

        let native_media_type = native_media_type(source_reader)?;
        let (original_width, original_height) = media_type_dimensions(&native_media_type)?;
        let target_size = resize_dimensions(original_width, original_height, options.max_dimension);
        let (media_type, software_resize) = match configure_bgra_format(source_reader, target_size)
        {
            Ok(media_type) => (media_type, None),
            Err(error) => {
                let Some(target_size) = target_size else {
                    return Err(error);
                };
                let media_type = configure_bgra_format(source_reader, None).with_context(|| {
                    format!(
                        "failed to set native Media Foundation output media type after resized output was rejected: {error:#}"
                    )
                })?;
                (media_type, Some(target_size))
            }
        };
        let (source_width, source_height) = media_type_dimensions(&media_type)?;
        let source_bytes_per_row = media_type_stride(&media_type, source_width)?;
        let (width, height, bytes_per_row) = match software_resize {
            Some((width, height)) => (
                width,
                height,
                width
                    .checked_mul(4)
                    .context("resized row width overflowed")?,
            ),
            None => (source_width, source_height, source_bytes_per_row),
        };

        self.width = width;
        self.height = height;
        self.bytes_per_row = bytes_per_row;
        self.source_width = source_width;
        self.source_height = source_height;
        self.source_bytes_per_row = source_bytes_per_row;
        self.software_resize = software_resize;
        Ok(())
    }
}

impl Drop for MediaFoundationDecoder {
    fn drop(&mut self) {
        self.source_reader = None;
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
        let source_reader = self
            .source_reader
            .as_ref()
            .context("Media Foundation source reader was not initialized")?;

        unsafe {
            source_reader
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

        if ptr.is_null() {
            bail!("Media Foundation returned a null frame buffer");
        }

        let source_bytes_per_row = sample_bytes_per_row(
            self.source_width,
            self.source_height,
            self.source_bytes_per_row,
            current_len,
        )?;
        let required_len = frame_buffer_len(source_bytes_per_row, self.source_height)?;
        if (current_len as usize) < required_len {
            bail!(
                "Media Foundation returned a frame buffer that is too short; got {} bytes, need {required_len}",
                current_len
            );
        }
        let source = unsafe { std::slice::from_raw_parts(ptr, required_len) };
        let time = self.timestamp_clock.media_time_from_hns(timestamp_100ns)?;

        match self.software_resize {
            Some((width, height)) => resize_bgra(
                source,
                self.source_width,
                self.source_height,
                source_bytes_per_row,
                width,
                height,
                time,
            )
            .map(Some),
            None => VideoFrame::new_bgra(
                self.width,
                self.height,
                source_bytes_per_row,
                time,
                source.to_vec(),
            )
            .map(Some),
        }
    }

    fn duration(&self) -> Option<MediaTime> {
        self.duration
    }
}

fn native_media_type(source_reader: &IMFSourceReader) -> anyhow::Result<IMFMediaType> {
    unsafe { source_reader.GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, 0) }
        .context("failed to get native video media type")
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
            .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)
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

fn media_type_stride(media_type: &IMFMediaType, width: u32) -> anyhow::Result<u32> {
    let stride = match unsafe { media_type.GetUINT32(&MF_MT_DEFAULT_STRIDE) } {
        Ok(stride) => i32::from_ne_bytes(stride.to_ne_bytes()),
        Err(_) => unsafe { MFGetStrideForBitmapInfoHeader(MFVideoFormat_RGB32.data1, width) }
            .context("failed to get BGRA stride from Media Foundation")?,
    };
    absolute_bgra_stride(stride, width)
}

fn absolute_bgra_stride(stride: i32, width: u32) -> anyhow::Result<u32> {
    if stride == i32::MIN {
        bail!("Media Foundation returned an invalid BGRA stride");
    }

    let bytes_per_row = stride.unsigned_abs();
    let min_bytes_per_row = width
        .checked_mul(4)
        .context("Media Foundation BGRA row width overflowed")?;
    Ok(bytes_per_row.max(min_bytes_per_row))
}

fn frame_buffer_len(bytes_per_row: u32, height: u32) -> anyhow::Result<usize> {
    (bytes_per_row as usize)
        .checked_mul(height as usize)
        .context("Media Foundation frame buffer length overflowed")
}

fn sample_bytes_per_row(
    width: u32,
    height: u32,
    media_type_bytes_per_row: u32,
    buffer_len: u32,
) -> anyhow::Result<u32> {
    let min_bytes_per_row = width
        .checked_mul(4)
        .context("Media Foundation BGRA row width overflowed")?;
    let reported_bytes_per_row = media_type_bytes_per_row.max(min_bytes_per_row);
    let reported_visible_len = reported_bytes_per_row as u64 * height as u64;

    if buffer_len as u64 <= reported_visible_len {
        return Ok(reported_bytes_per_row);
    }

    // Some H.264 decoders expose visible width in MF_MT_DEFAULT_STRIDE while
    // each sample is padded to coded width, e.g. 1080 BGRA pixels in a 1088-wide
    // coded frame. In that case the buffer length reveals the actual row pitch.
    if height != 0 && buffer_len % height == 0 {
        let inferred_bytes_per_row = buffer_len / height;
        if inferred_bytes_per_row >= reported_bytes_per_row && inferred_bytes_per_row % 4 == 0 {
            return Ok(inferred_bytes_per_row);
        }
    }

    Ok(reported_bytes_per_row)
}

fn media_duration(
    source_reader: &IMFSourceReader,
    mp4_timing: Mp4VideoTiming,
) -> Option<MediaTime> {
    let duration = media_foundation_duration(source_reader);
    if duration.is_some_and(|duration| mp4_timing.matches_unknown_mdhd_duration(duration)) {
        return mp4_timing.duration;
    }

    duration.or(mp4_timing.duration)
}

fn media_foundation_duration(source_reader: &IMFSourceReader) -> Option<MediaTime> {
    let value = unsafe {
        source_reader
            .GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE.0 as u32, &MF_PD_DURATION)
            .ok()
    }?;
    let duration_100ns = i64::try_from(&value).ok()?;
    if duration_100ns <= 0 {
        return None;
    }
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

fn resize_bgra(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    source_bytes_per_row: u32,
    target_width: u32,
    target_height: u32,
    time: MediaTime,
) -> anyhow::Result<VideoFrame> {
    let target_bytes_per_row = target_width
        .checked_mul(4)
        .context("resized row width overflowed")?;
    let mut data = vec![0_u8; target_bytes_per_row as usize * target_height as usize];

    let source_x = sample_coordinates(target_width as usize, source_width as usize);
    let source_y = sample_coordinates(target_height as usize, source_height as usize);
    for (target_y, sample_y) in source_y.into_iter().enumerate() {
        for (target_x, sample_x) in source_x.iter().copied().enumerate() {
            let target_offset = target_y * target_bytes_per_row as usize + target_x * 4;
            for channel in 0..4 {
                data[target_offset + channel] =
                    sample_bgra_channel(source, source_bytes_per_row, sample_x, sample_y, channel)?;
            }
        }
    }

    VideoFrame::new_bgra(
        target_width,
        target_height,
        target_bytes_per_row,
        time,
        data,
    )
}

fn mp4_video_timing(path: &Path) -> anyhow::Result<Mp4VideoTiming> {
    if !matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref(),
        Some("mp4" | "m4v" | "mov")
    ) {
        return Ok(Mp4VideoTiming::default());
    }

    let mut file = File::open(path).with_context(|| {
        format!(
            "failed to open {} for MP4 timestamp inspection",
            path.display()
        )
    })?;
    let len = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();

    while let Some(box_header) = read_mp4_box(&mut file, len)? {
        if box_header.kind == *b"moov" {
            return find_video_track_timing(&mut file, box_header.end);
        }
        file.seek(SeekFrom::Start(box_header.end))
            .context("failed to skip MP4 box")?;
    }

    Ok(Mp4VideoTiming::default())
}

fn find_video_track_timing(file: &mut File, moov_end: u64) -> anyhow::Result<Mp4VideoTiming> {
    while let Some(box_header) = read_mp4_box(file, moov_end)? {
        if box_header.kind == *b"trak" {
            let track = read_track_info(file, box_header.end)?;
            if track.handler_type == Some(*b"vide") {
                return Ok(track.video_timing());
            }
        }
        file.seek(SeekFrom::Start(box_header.end))
            .context("failed to skip MP4 track box")?;
    }

    Ok(Mp4VideoTiming::default())
}

fn read_track_info(file: &mut File, trak_end: u64) -> anyhow::Result<Mp4TrackInfo> {
    let mut track = Mp4TrackInfo::default();

    while let Some(box_header) = read_mp4_box(file, trak_end)? {
        if box_header.kind == *b"mdia" {
            track = read_media_info(file, box_header.end)?;
        }
        file.seek(SeekFrom::Start(box_header.end))
            .context("failed to skip MP4 track child box")?;
    }

    Ok(track)
}

fn read_media_info(file: &mut File, mdia_end: u64) -> anyhow::Result<Mp4TrackInfo> {
    let mut track = Mp4TrackInfo::default();

    while let Some(box_header) = read_mp4_box(file, mdia_end)? {
        match &box_header.kind {
            b"mdhd" => {
                let timing = read_mdhd_timing(file, box_header.end)?;
                track.timescale = timing.timescale;
                track.mdhd_duration_units = timing.duration_units;
                track.unknown_mdhd_duration_units = timing.unknown_duration_units;
            }
            b"hdlr" => track.handler_type = read_hdlr_type(file, box_header.end)?,
            b"minf" => track.stts_duration_units = read_minf_stts_duration(file, box_header.end)?,
            _ => {}
        }
        file.seek(SeekFrom::Start(box_header.end))
            .context("failed to skip MP4 media child box")?;
    }

    Ok(track)
}

fn read_mdhd_timing(file: &mut File, box_end: u64) -> anyhow::Result<MdhdTiming> {
    let payload_start = file
        .stream_position()
        .context("failed to read MP4 mdhd position")?;
    let payload_len = box_end.saturating_sub(payload_start);
    let bytes_to_read = payload_len.min(32) as usize;
    let mut payload = vec![0_u8; bytes_to_read];
    file.read_exact(&mut payload)
        .context("failed to read MP4 mdhd payload")?;

    mdhd_timing_from_payload(&payload)
}

fn mdhd_timing_from_payload(payload: &[u8]) -> anyhow::Result<MdhdTiming> {
    let version = *payload.first().context("MP4 mdhd box was empty")?;
    let (timescale_offset, duration_offset, duration_len) = match version {
        0 => (12, 16, 4),
        1 => (20, 24, 8),
        _ => return Ok(MdhdTiming::default()),
    };
    let timescale = read_be_u32(
        payload
            .get(timescale_offset..timescale_offset + 4)
            .context("MP4 mdhd box was too short for timescale")?,
    );
    if timescale == 0 {
        return Ok(MdhdTiming::default());
    }

    let duration = payload
        .get(duration_offset..duration_offset + duration_len)
        .context("MP4 mdhd box was too short for duration")?;
    let duration_units = if duration_len == 4 {
        read_be_u32(duration) as u64
    } else {
        read_be_u64(duration)
    };
    let unknown_duration_units = match (duration_len, duration_units) {
        (_, 0) => None,
        (4, units) if units == u32::MAX as u64 => Some(units),
        (8, u64::MAX) => Some(u64::MAX),
        _ => None,
    };
    let duration_units =
        (duration_units != 0 && unknown_duration_units.is_none()).then_some(duration_units);

    Ok(MdhdTiming {
        timescale: Some(timescale),
        duration_units,
        unknown_duration_units,
    })
}

fn read_hdlr_type(file: &mut File, box_end: u64) -> anyhow::Result<Option<[u8; 4]>> {
    let payload_start = file
        .stream_position()
        .context("failed to read MP4 hdlr position")?;
    let payload_len = box_end.saturating_sub(payload_start);
    let bytes_to_read = payload_len.min(16) as usize;
    let mut payload = vec![0_u8; bytes_to_read];
    file.read_exact(&mut payload)
        .context("failed to read MP4 hdlr payload")?;

    let handler = payload
        .get(8..12)
        .context("MP4 hdlr box was too short for handler type")?;
    Ok(Some([handler[0], handler[1], handler[2], handler[3]]))
}

fn read_minf_stts_duration(file: &mut File, minf_end: u64) -> anyhow::Result<Option<u64>> {
    let mut duration = None;
    while let Some(box_header) = read_mp4_box(file, minf_end)? {
        if box_header.kind == *b"stbl" {
            duration = read_stbl_stts_duration(file, box_header.end)?;
        }
        file.seek(SeekFrom::Start(box_header.end))
            .context("failed to skip MP4 media information child box")?;
    }

    Ok(duration)
}

fn read_stbl_stts_duration(file: &mut File, stbl_end: u64) -> anyhow::Result<Option<u64>> {
    let mut duration = None;
    while let Some(box_header) = read_mp4_box(file, stbl_end)? {
        if box_header.kind == *b"stts" {
            duration = read_stts_duration(file, box_header.end)?;
        }
        file.seek(SeekFrom::Start(box_header.end))
            .context("failed to skip MP4 sample table child box")?;
    }

    Ok(duration)
}

fn read_stts_duration(file: &mut File, box_end: u64) -> anyhow::Result<Option<u64>> {
    let payload_start = file
        .stream_position()
        .context("failed to read MP4 stts position")?;
    let payload_len = box_end.saturating_sub(payload_start);
    if payload_len < 8 {
        bail!("MP4 stts box was too short");
    }

    let mut header = [0_u8; 8];
    file.read_exact(&mut header)
        .context("failed to read MP4 stts header")?;
    let entry_count = read_be_u32(&header[4..8]) as u64;
    let entries_len = entry_count
        .checked_mul(8)
        .context("MP4 stts entry table size overflowed")?;
    if payload_len - 8 < entries_len {
        bail!("MP4 stts box was too short for entries");
    }

    let mut duration = 0_u64;
    for _ in 0..entry_count {
        let mut entry = [0_u8; 8];
        file.read_exact(&mut entry)
            .context("failed to read MP4 stts entry")?;
        let sample_count = read_be_u32(&entry[0..4]) as u64;
        let sample_delta = read_be_u32(&entry[4..8]) as u64;
        let entry_duration = sample_count
            .checked_mul(sample_delta)
            .context("MP4 stts duration overflowed")?;
        duration = duration
            .checked_add(entry_duration)
            .context("MP4 stts duration overflowed")?;
    }

    Ok((duration != 0).then_some(duration))
}

fn read_mp4_box(file: &mut File, parent_end: u64) -> anyhow::Result<Option<Mp4BoxHeader>> {
    let start = file
        .stream_position()
        .context("failed to read MP4 box position")?;
    if start + 8 > parent_end {
        return Ok(None);
    }

    let mut header = [0_u8; 8];
    file.read_exact(&mut header)
        .context("failed to read MP4 box header")?;
    let size32 = read_be_u32(&header[0..4]) as u64;
    let kind = [header[4], header[5], header[6], header[7]];

    let mut header_size = 8_u64;
    let size = match size32 {
        0 => parent_end - start,
        1 => {
            let mut large_size = [0_u8; 8];
            file.read_exact(&mut large_size)
                .context("failed to read MP4 largesize box header")?;
            header_size += 8;
            read_be_u64(&large_size)
        }
        size => size,
    };
    if kind == *b"uuid" {
        header_size += 16;
        file.seek(SeekFrom::Current(16))
            .context("failed to skip MP4 uuid extended type")?;
    }
    let end = start
        .checked_add(size)
        .context("invalid MP4 box size while reading timestamps")?;
    if size < header_size || end > parent_end {
        bail!("invalid MP4 box size while reading timestamps");
    }

    Ok(Some(Mp4BoxHeader { kind, end }))
}

fn read_be_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn read_be_u64(bytes: &[u8]) -> u64 {
    u64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

#[derive(Debug)]
struct Mp4BoxHeader {
    kind: [u8; 4],
    end: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Mp4VideoTiming {
    timescale: Option<u32>,
    duration: Option<MediaTime>,
    unknown_mdhd_duration: Option<Mp4UnknownDuration>,
}

impl Mp4VideoTiming {
    fn matches_unknown_mdhd_duration(self, duration: MediaTime) -> bool {
        self.unknown_mdhd_duration
            .is_some_and(|unknown| unknown.matches(duration))
    }
}

#[derive(Debug, Clone, Copy)]
struct Mp4UnknownDuration {
    units: u64,
    timescale: u32,
}

impl Mp4UnknownDuration {
    fn matches(self, duration: MediaTime) -> bool {
        let unknown_seconds = self.units as f64 / self.timescale as f64;
        let tolerance_seconds = 1.0_f64.max(2.0 / self.timescale as f64);
        (duration.as_seconds() - unknown_seconds).abs() <= tolerance_seconds
    }
}

#[derive(Default)]
struct MdhdTiming {
    timescale: Option<u32>,
    duration_units: Option<u64>,
    unknown_duration_units: Option<u64>,
}

#[derive(Default)]
struct Mp4TrackInfo {
    handler_type: Option<[u8; 4]>,
    timescale: Option<u32>,
    mdhd_duration_units: Option<u64>,
    unknown_mdhd_duration_units: Option<u64>,
    stts_duration_units: Option<u64>,
}

impl Mp4TrackInfo {
    fn video_timing(self) -> Mp4VideoTiming {
        let duration_units = self.mdhd_duration_units.or(self.stts_duration_units);
        Mp4VideoTiming {
            timescale: self.timescale,
            duration: self.timescale.and_then(|timescale| {
                duration_units.and_then(|units| media_time_from_mp4_units(units, timescale))
            }),
            unknown_mdhd_duration: self.timescale.and_then(|timescale| {
                self.unknown_mdhd_duration_units
                    .map(|units| Mp4UnknownDuration { units, timescale })
            }),
        }
    }
}

fn media_time_from_mp4_units(units: u64, timescale: u32) -> Option<MediaTime> {
    if units == 0 {
        return None;
    }
    let value = i64::try_from(units).ok()?;
    let timescale = i32::try_from(timescale).ok()?;
    MediaTime::new(value, timescale).ok()
}

fn sample_coordinates(target_len: usize, source_len: usize) -> Vec<SampleCoordinate> {
    (0..target_len)
        .map(|target| source_coordinate(target, target_len, source_len))
        .collect()
}

fn source_coordinate(target: usize, target_len: usize, source_len: usize) -> SampleCoordinate {
    if source_len <= 1 || target_len <= 1 {
        return SampleCoordinate {
            lower: 0,
            upper: 0,
            weight: 0.0,
        };
    }

    let source = ((target as f32 + 0.5) * source_len as f32 / target_len as f32 - 0.5)
        .clamp(0.0, (source_len - 1) as f32);
    let lower = source.floor() as usize;
    let upper = (lower + 1).min(source_len - 1);

    SampleCoordinate {
        lower,
        upper,
        weight: source - lower as f32,
    }
}

fn sample_bgra_channel(
    source: &[u8],
    bytes_per_row: u32,
    x: SampleCoordinate,
    y: SampleCoordinate,
    channel: usize,
) -> anyhow::Result<u8> {
    let top = mix(
        source_channel(source, bytes_per_row, x.lower, y.lower, channel)? as f32,
        source_channel(source, bytes_per_row, x.upper, y.lower, channel)? as f32,
        x.weight,
    );
    let bottom = mix(
        source_channel(source, bytes_per_row, x.lower, y.upper, channel)? as f32,
        source_channel(source, bytes_per_row, x.upper, y.upper, channel)? as f32,
        x.weight,
    );

    Ok(mix(top, bottom, y.weight).round().clamp(0.0, 255.0) as u8)
}

fn source_channel(
    source: &[u8],
    bytes_per_row: u32,
    x: usize,
    y: usize,
    channel: usize,
) -> anyhow::Result<u8> {
    let offset = y
        .checked_mul(bytes_per_row as usize)
        .and_then(|offset| offset.checked_add(x.checked_mul(4)?))
        .and_then(|offset| offset.checked_add(channel))
        .context("resized frame source offset overflowed")?;
    source
        .get(offset)
        .copied()
        .context("resized frame source offset was out of bounds")
}

fn mix(from: f32, to: f32, amount: f32) -> f32 {
    from + (to - from) * amount
}

#[derive(Debug, Clone, Copy)]
struct SampleCoordinate {
    lower: usize,
    upper: usize,
    weight: f32,
}

#[derive(Debug, Clone, Copy)]
struct TimestampScale {
    numerator: u32,
    denominator: u32,
}

struct TimestampClock {
    preferred_scale: Option<TimestampScale>,
    observed_hns: Vec<i64>,
    scale: Option<TimestampScale>,
    disabled: bool,
}

impl TimestampClock {
    fn new(preferred_timescale: Option<u32>) -> Self {
        Self {
            preferred_scale: preferred_timescale.and_then(|numerator| {
                let _ = i32::try_from(numerator).ok()?;
                (numerator > 0).then_some(TimestampScale {
                    numerator,
                    denominator: 1,
                })
            }),
            observed_hns: Vec::new(),
            scale: None,
            disabled: false,
        }
    }

    fn media_time_from_hns(&mut self, timestamp_100ns: i64) -> anyhow::Result<MediaTime> {
        self.observed_hns.push(timestamp_100ns);

        if let Some(scale) = self.scale {
            if scale.fits(timestamp_100ns) {
                return scale.media_time_from_hns(timestamp_100ns);
            }
            self.scale = None;
        }

        if !self.disabled && self.observed_hns.len() >= MIN_TIMESTAMPS_FOR_CLOCK {
            self.scale = timestamp_scale_for_observed(&self.observed_hns);
            if let Some(scale) = self.scale {
                return scale.media_time_from_hns(timestamp_100ns);
            }
            self.disabled = true;
        }

        if let Some(scale) = self.preferred_scale {
            if scale.fits(timestamp_100ns) {
                return scale.media_time_from_hns(timestamp_100ns);
            }
            self.preferred_scale = None;
        }

        MediaTime::new(timestamp_100ns, 10_000_000)
    }
}

impl TimestampScale {
    fn media_time_from_hns(self, timestamp_100ns: i64) -> anyhow::Result<MediaTime> {
        let value = self.timestamp_units(timestamp_100ns);
        let value =
            i64::try_from(value).context("native-frame-clock timestamp is outside range")?;
        let timescale = i32::try_from(self.numerator)
            .context("native-frame-clock timescale is outside range")?;
        MediaTime::new(value, timescale)
    }

    fn fits(self, timestamp_100ns: i64) -> bool {
        let value = self.timestamp_units(timestamp_100ns);
        let scaled_hns = value * 10_000_000_i128;
        let original_hns = timestamp_100ns as i128 * self.numerator as i128;
        (scaled_hns - original_hns).abs()
            <= TIMESTAMP_QUANTIZATION_TOLERANCE_HNS * self.numerator as i128
    }

    fn timestamp_units(self, timestamp_100ns: i64) -> i128 {
        let numerator = timestamp_100ns as i128 * self.numerator as i128;
        let units = rounded_div(numerator, 10_000_000_i128);
        rounded_div(units, self.denominator as i128) * self.denominator as i128
    }
}

const MIN_TIMESTAMPS_FOR_CLOCK: usize = 8;
const TIMESTAMP_QUANTIZATION_TOLERANCE_HNS: i128 = 1_000;

fn timestamp_scale_for_observed(observed_hns: &[i64]) -> Option<TimestampScale> {
    standard_timestamp_scales()
        .into_iter()
        .find(|scale| observed_hns.iter().copied().all(|time| scale.fits(time)))
}

fn standard_timestamp_scales() -> [TimestampScale; 21] {
    [
        TimestampScale {
            numerator: 24_000,
            denominator: 1001,
        },
        TimestampScale {
            numerator: 24,
            denominator: 1,
        },
        TimestampScale {
            numerator: 25,
            denominator: 1,
        },
        TimestampScale {
            numerator: 30_000,
            denominator: 1001,
        },
        TimestampScale {
            numerator: 30,
            denominator: 1,
        },
        TimestampScale {
            numerator: 48_000,
            denominator: 1001,
        },
        TimestampScale {
            numerator: 48,
            denominator: 1,
        },
        TimestampScale {
            numerator: 50,
            denominator: 1,
        },
        TimestampScale {
            numerator: 60_000,
            denominator: 1001,
        },
        TimestampScale {
            numerator: 60,
            denominator: 1,
        },
        TimestampScale {
            numerator: 100,
            denominator: 1,
        },
        TimestampScale {
            numerator: 120_000,
            denominator: 1001,
        },
        TimestampScale {
            numerator: 120,
            denominator: 1,
        },
        TimestampScale {
            numerator: 150,
            denominator: 1,
        },
        TimestampScale {
            numerator: 240,
            denominator: 1,
        },
        TimestampScale {
            numerator: 600,
            denominator: 1,
        },
        TimestampScale {
            numerator: 1_000,
            denominator: 1,
        },
        TimestampScale {
            numerator: 15_360,
            denominator: 1,
        },
        TimestampScale {
            numerator: 30_000,
            denominator: 1,
        },
        TimestampScale {
            numerator: 60_000,
            denominator: 1,
        },
        TimestampScale {
            numerator: 90_000,
            denominator: 1,
        },
    ]
}

fn rounded_div(numerator: i128, denominator: i128) -> i128 {
    if numerator >= 0 {
        (numerator + denominator / 2) / denominator
    } else {
        (numerator - denominator / 2) / denominator
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn timestamp_clock_snaps_vfr_gop_drift_to_observed_video_grid() {
        let mut clock = TimestampClock::new(None);
        let timestamps_hns = [
            0, 333_333, 666_667, 1_000_000, 1_250_000, 1_583_333, 1_916_667, 2_250_000, 2_583_333,
            2_916_667, 3_333_333, 3_666_667, 39_500_000, 39_832_667,
        ];

        let mut last = MediaTime::new(0, 10_000_000).unwrap();
        for timestamp_hns in timestamps_hns {
            last = clock.media_time_from_hns(timestamp_hns).unwrap();
        }

        assert_eq!(
            last,
            MediaTime {
                value: 478,
                timescale: 120
            }
        );
    }

    #[test]
    fn timestamp_clock_keeps_ntsc_frame_grid() {
        let mut clock = TimestampClock::new(None);
        let mut last = MediaTime::new(0, 10_000_000).unwrap();
        for frame in 0..16 {
            let frame = frame as i64;
            let timestamp_hns = ((frame * 1001 * 10_000_000) as f64 / 30_000_f64).round() as i64;
            last = clock.media_time_from_hns(timestamp_hns).unwrap();
        }

        assert_eq!(
            last,
            MediaTime {
                value: 15_015,
                timescale: 30_000
            }
        );
    }

    #[test]
    fn timestamp_clock_prefers_observed_video_grid_over_container_grid() {
        let mut clock = TimestampClock::new(Some(15_360));
        let timestamps_hns = [
            0, 333_333, 666_667, 1_000_000, 1_250_000, 1_583_333, 1_916_667, 2_250_000, 39_832_667,
        ];

        let mut last = MediaTime::new(0, 10_000_000).unwrap();
        for timestamp_hns in timestamps_hns {
            last = clock.media_time_from_hns(timestamp_hns).unwrap();
        }

        assert_eq!(
            last,
            MediaTime {
                value: 478,
                timescale: 120
            }
        );
    }

    #[test]
    fn timestamp_clock_uses_container_grid_for_irregular_mp4_times() {
        let mut clock = TimestampClock::new(Some(15_360));
        let timestamp_hns = 2_333_984;

        assert_eq!(
            clock.media_time_from_hns(timestamp_hns).unwrap(),
            MediaTime {
                value: 3_585,
                timescale: 15_360
            }
        );
    }

    #[test]
    fn timestamp_clock_ignores_unrepresentable_container_timescale() {
        let mut clock = TimestampClock::new(Some(u32::MAX));

        assert_eq!(
            clock.media_time_from_hns(0).unwrap(),
            MediaTime {
                value: 0,
                timescale: 10_000_000
            }
        );
    }

    #[test]
    fn read_mp4_box_rejects_overflowing_large_size() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("segmenter_mp4_overflow_{unique}.mp4"));

        {
            let mut file = File::create(&path).unwrap();
            file.write_all(&8_u32.to_be_bytes()).unwrap();
            file.write_all(b"ftyp").unwrap();
            file.write_all(&1_u32.to_be_bytes()).unwrap();
            file.write_all(b"free").unwrap();
            file.write_all(&u64::MAX.to_be_bytes()).unwrap();
        }

        let mut file = File::open(&path).unwrap();
        let first = read_mp4_box(&mut file, u64::MAX).unwrap().unwrap();
        assert_eq!(first.end, 8);

        file.seek(SeekFrom::Start(first.end)).unwrap();
        assert!(read_mp4_box(&mut file, u64::MAX).is_err());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn mdhd_all_ones_duration_is_marked_unknown() {
        let mut payload = [0_u8; 20];
        payload[12..16].copy_from_slice(&10_240_u32.to_be_bytes());
        payload[16..20].copy_from_slice(&u32::MAX.to_be_bytes());

        let timing = mdhd_timing_from_payload(&payload).unwrap();

        assert_eq!(timing.timescale, Some(10_240));
        assert_eq!(timing.duration_units, None);
        assert_eq!(timing.unknown_duration_units, Some(u32::MAX as u64));
    }

    #[test]
    fn mp4_timing_uses_stts_duration_when_mdhd_duration_is_unknown() {
        let track = Mp4TrackInfo {
            timescale: Some(10_240),
            unknown_mdhd_duration_units: Some(u32::MAX as u64),
            stts_duration_units: Some(10_240),
            ..Mp4TrackInfo::default()
        };

        assert_eq!(
            track.video_timing().duration,
            Some(MediaTime {
                value: 10_240,
                timescale: 10_240
            })
        );
    }

    #[test]
    fn mp4_unknown_duration_matches_media_foundation_rounded_sentinel() {
        let timing = Mp4VideoTiming {
            timescale: Some(10_240),
            unknown_mdhd_duration: Some(Mp4UnknownDuration {
                units: u32::MAX as u64,
                timescale: 10_240,
            }),
            ..Mp4VideoTiming::default()
        };

        assert!(timing.matches_unknown_mdhd_duration(MediaTime {
            value: 4_194_304_000_000,
            timescale: 10_000_000,
        }));
    }

    #[test]
    fn absolute_bgra_stride_accepts_signed_stride() {
        assert_eq!(absolute_bgra_stride(-4320, 1080).unwrap(), 4320);
    }

    #[test]
    fn sample_bytes_per_row_uses_reported_stride_for_visible_buffer() {
        assert_eq!(
            sample_bytes_per_row(1080, 1920, 4320, 4320 * 1920).unwrap(),
            4320
        );
    }

    #[test]
    fn sample_bytes_per_row_infers_coded_width_padding() {
        assert_eq!(
            sample_bytes_per_row(1080, 1920, 4320, 4352 * 1920).unwrap(),
            4352
        );
    }

    #[test]
    fn sample_bytes_per_row_ignores_bottom_padding() {
        assert_eq!(
            sample_bytes_per_row(1920, 1080, 7680, 7680 * 1088).unwrap(),
            7680
        );
    }

    #[test]
    fn sample_bytes_per_row_enforces_minimum_bgra_row_width() {
        assert_eq!(
            sample_bytes_per_row(1080, 1920, 0, 4320 * 1920).unwrap(),
            4320
        );
    }
}
