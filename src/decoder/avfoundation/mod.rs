mod native;

use std::{
    ffi::{c_void, CString},
    path::Path,
    sync::Arc,
};

use anyhow::{bail, Context};
use native::{
    segd_get_asset_duration, segd_get_asset_rotation, segd_get_sample_data, segd_initialize_asset,
    segd_initialize_asset_reader, segd_initialize_video_track_output, segd_lock_sample,
    segd_read_sample, segd_release_asset, segd_release_asset_reader, segd_release_sample,
    segd_release_track_output, segd_start_asset_reader, segd_unlock_sample, SEGDecodeOptions,
    SEGDecodeTime, SEGDecodedSample, SEGD_NOT_FOUND_ERROR, SEGD_READ_SAMPLE_CANCELLED,
    SEGD_READ_SAMPLE_COMPLETED, SEGD_READ_SAMPLE_FAILED, SEGD_READ_SAMPLE_NO_SAMPLE,
    SEGD_READ_SAMPLE_SUCCESS, SEGD_READ_SAMPLE_UNKNOWN, SEGD_SUCCESS, SEGD_VIDEO_FORMAT_BGRA,
};

use crate::{
    decoder::{DecodeOptions, VideoDecoder},
    frame::{DisplayRotation, MediaTime, VideoFrame},
};

pub struct AvFoundationDecoder {
    _asset: Arc<AvAsset>,
    _reader: Arc<AvAssetReader>,
    track_output: Arc<AvAssetReaderTrackOutput>,
    display_rotation: DisplayRotation,
    duration: Option<MediaTime>,
}

impl AvFoundationDecoder {
    pub fn new(input: &Path, options: DecodeOptions) -> anyhow::Result<Self> {
        if !input.exists() {
            bail!("input file does not exist: {}", input.display());
        }

        let native_options = Arc::new(native_options(input, options)?);
        let asset = Arc::new(AvAsset::new(native_options.clone())?);
        let duration = asset.duration();
        let display_rotation = asset.display_rotation()?;
        let reader = Arc::new(AvAssetReader::new(native_options.clone(), asset.clone())?);
        let track_output = Arc::new(AvAssetReaderTrackOutput::new(
            native_options,
            asset.clone(),
            reader.clone(),
        )?);

        reader.start()?;

        Ok(Self {
            _asset: asset,
            _reader: reader,
            track_output,
            display_rotation,
            duration,
        })
    }
}

impl VideoDecoder for AvFoundationDecoder {
    fn read_frame(&mut self) -> anyhow::Result<Option<VideoFrame>> {
        read_video_frame(
            &self.track_output.reader,
            &self.track_output,
            self.display_rotation,
        )
    }

    fn duration(&self) -> Option<MediaTime> {
        self.duration
    }
}

fn native_options(input: &Path, options: DecodeOptions) -> anyhow::Result<SEGDecodeOptions> {
    let native_file_path = CString::new(input.to_string_lossy().as_bytes())
        .with_context(|| format!("input path contained an interior NUL: {}", input.display()))?;
    let native_file_path_bytes = native_file_path.as_bytes_with_nul();
    let mut file_path = [0; 1024];
    if native_file_path_bytes.len() > file_path.len() {
        bail!(
            "input path is too long for AVFoundation bridge: {}",
            input.display()
        );
    }
    for (slot, byte) in file_path
        .iter_mut()
        .zip(native_file_path_bytes.iter().copied())
    {
        *slot = byte as i8;
    }

    Ok(SEGDecodeOptions {
        file_path,
        output_video_format: SEGD_VIDEO_FORMAT_BGRA,
        max_dimension: options.max_dimension.unwrap_or(0),
    })
}

struct AvAsset {
    ptr: *mut c_void,
    _options: Arc<SEGDecodeOptions>,
}

impl AvAsset {
    fn new(options: Arc<SEGDecodeOptions>) -> anyhow::Result<Self> {
        let mut error_code = 0;
        let ptr = unsafe { segd_initialize_asset(options.as_ref(), &mut error_code) };
        if ptr.is_null() {
            bail!("failed to initialize AVAsset: {error_code}");
        }

        Ok(Self {
            ptr,
            _options: options,
        })
    }

    fn duration(&self) -> Option<MediaTime> {
        let mut duration = SEGDecodeTime {
            value: 0,
            timescale: 0,
        };
        let result = unsafe { segd_get_asset_duration(self.ptr, &mut duration) };
        if result != SEGD_SUCCESS {
            return None;
        }

        MediaTime::new(duration.value, duration.timescale).ok()
    }

    fn display_rotation(&self) -> anyhow::Result<DisplayRotation> {
        let mut rotation = 0;
        let result = unsafe { segd_get_asset_rotation(self.ptr, &mut rotation) };
        if result == SEGD_NOT_FOUND_ERROR {
            return Ok(DisplayRotation::None);
        }
        if result != SEGD_SUCCESS {
            bail!("failed to read AVFoundation video rotation: {result}");
        }

        DisplayRotation::from_clockwise_degrees(rotation)
            .with_context(|| format!("unsupported AVFoundation video rotation {rotation} degrees"))
    }
}

impl Drop for AvAsset {
    fn drop(&mut self) {
        unsafe { segd_release_asset(self.ptr) };
    }
}

struct AvAssetReader {
    ptr: *mut c_void,
    _asset: Arc<AvAsset>,
    _options: Arc<SEGDecodeOptions>,
}

impl AvAssetReader {
    fn new(options: Arc<SEGDecodeOptions>, asset: Arc<AvAsset>) -> anyhow::Result<Self> {
        let mut error_code = 0;
        let ptr =
            unsafe { segd_initialize_asset_reader(options.as_ref(), asset.ptr, &mut error_code) };
        if ptr.is_null() {
            bail!("failed to initialize AVAssetReader: {error_code}");
        }

        Ok(Self {
            ptr,
            _asset: asset,
            _options: options,
        })
    }

    fn start(&self) -> anyhow::Result<()> {
        let result = unsafe { segd_start_asset_reader(self.ptr) };
        if result != SEGD_SUCCESS {
            bail!("failed to start AVAssetReader: {result}");
        }

        Ok(())
    }
}

impl Drop for AvAssetReader {
    fn drop(&mut self) {
        unsafe { segd_release_asset_reader(self.ptr) };
    }
}

struct AvAssetReaderTrackOutput {
    ptr: *mut c_void,
    reader: Arc<AvAssetReader>,
    _asset: Arc<AvAsset>,
    _options: Arc<SEGDecodeOptions>,
}

impl AvAssetReaderTrackOutput {
    fn new(
        options: Arc<SEGDecodeOptions>,
        asset: Arc<AvAsset>,
        reader: Arc<AvAssetReader>,
    ) -> anyhow::Result<Self> {
        let mut error_code = 0;
        let ptr = unsafe {
            segd_initialize_video_track_output(
                options.as_ref(),
                asset.ptr,
                reader.ptr,
                &mut error_code,
            )
        };
        if ptr.is_null() {
            bail!("failed to initialize AVAssetReaderTrackOutput: {error_code}");
        }

        Ok(Self {
            ptr,
            reader,
            _asset: asset,
            _options: options,
        })
    }
}

impl Drop for AvAssetReaderTrackOutput {
    fn drop(&mut self) {
        unsafe { segd_release_track_output(self.ptr) };
    }
}

fn read_video_frame(
    reader: &AvAssetReader,
    track_output: &AvAssetReaderTrackOutput,
    display_rotation: DisplayRotation,
) -> anyhow::Result<Option<VideoFrame>> {
    let mut sample = SEGDecodedSample {
        sample_buffer: std::ptr::null_mut(),
        width: 0,
        height: 0,
        pts: SEGDecodeTime {
            value: 0,
            timescale: 0,
        },
    };

    let result = unsafe { segd_read_sample(reader.ptr, track_output.ptr, &mut sample) };
    match result {
        SEGD_READ_SAMPLE_SUCCESS => {}
        SEGD_READ_SAMPLE_COMPLETED | SEGD_READ_SAMPLE_NO_SAMPLE => return Ok(None),
        SEGD_READ_SAMPLE_UNKNOWN => bail!("failed to read AVFoundation sample: unknown status"),
        SEGD_READ_SAMPLE_CANCELLED => bail!("failed to read AVFoundation sample: cancelled"),
        SEGD_READ_SAMPLE_FAILED => bail!("failed to read AVFoundation sample: failed"),
        other => bail!("failed to read AVFoundation sample: unexpected status {other}"),
    }

    let sample = NativeSample { sample };
    let lock_result = unsafe { segd_lock_sample(&sample.sample) };
    if lock_result != SEGD_SUCCESS {
        bail!("failed to lock AVFoundation sample: {lock_result}");
    }
    scopeguard::defer! {
        unsafe {
            segd_unlock_sample(&sample.sample);
        }
    }

    let sample_data = unsafe { segd_get_sample_data(&sample.sample) };
    if sample_data.valid == 0 || sample_data.format != SEGD_VIDEO_FORMAT_BGRA {
        bail!("AVFoundation returned a non-BGRA sample");
    }
    let plane = sample_data.planes[0];
    if plane.is_null() {
        bail!("AVFoundation returned a BGRA sample with no pixel data");
    }
    let plane_size = sample_data.planes_size[0] as usize;
    let source = unsafe { std::slice::from_raw_parts(plane as *const u8, plane_size) };
    let data = source.to_vec();
    let time = MediaTime::new(sample.sample.pts.value, sample.sample.pts.timescale)?;

    let frame = VideoFrame::new_bgra(
        sample.sample.width,
        sample.sample.height,
        sample_data.bytes_per_row[0],
        time,
        data,
    )?;

    frame.rotated(display_rotation).map(Some)
}

struct NativeSample {
    sample: SEGDecodedSample,
}

impl Drop for NativeSample {
    fn drop(&mut self) {
        unsafe { segd_release_sample(&self.sample) };
    }
}
