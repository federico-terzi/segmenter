//! Match the basic Media Foundation processor's limited-range integer color
//! conversion, then the segmenter's existing BGRA bilinear resize. Sparse resizes
//! convert sampled pixels only; denser output uses a vectorized RGB intermediate.
use super::{media_type_dimensions, mix, sample_coordinates, PendingSample};
use windows::Win32::Media::MediaFoundation::{
    IMFMediaType, IMFSample, IMFSourceReader, MFCreateMediaType, MFMediaType_Video,
    MFNominalRange_16_235, MFSampleExtension_Interlaced, MFVideoFormat_NV12,
    MFVideoInterlace_MixedInterlaceOrProgressive, MFVideoInterlace_Progressive,
    MFVideoTransferMatrix_BT601, MFVideoTransferMatrix_BT709, MF_MT_DEFAULT_STRIDE,
    MF_MT_GEOMETRIC_APERTURE, MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE,
    MF_MT_MINIMUM_DISPLAY_APERTURE, MF_MT_PAN_SCAN_APERTURE, MF_MT_PAN_SCAN_ENABLED, MF_MT_SUBTYPE,
    MF_MT_VIDEO_NOMINAL_RANGE, MF_MT_YUV_MATRIX, MF_SOURCE_READERF_ENDOFSTREAM,
    MF_SOURCE_READERF_ERROR, MF_SOURCE_READER_FIRST_VIDEO_STREAM,
};

use crate::frame::{MediaTime, VideoFrame};
use anyhow::{bail, Context};

#[cfg(target_arch = "x86_64")]
mod avx2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Matrix {
    Bt601,
    Bt709,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    pub(super) stride: usize,
    pub(super) coded_height: usize,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) matrix: Matrix,
    pub(super) mixed_interlace: bool,
}

impl Layout {
    fn validate(self, len: usize) -> anyhow::Result<usize> {
        if self.width == 0
            || self.height == 0
            || self.stride < self.width as usize
            || self.stride % 2 != 0
            || self.coded_height % 2 != 0
            || self.coded_height < self.height as usize
        {
            bail!("invalid NV12 frame layout: {self:?}");
        }
        let uv_offset = self
            .stride
            .checked_mul(self.coded_height)
            .context("NV12 plane size overflowed")?;
        let required = uv_offset
            .checked_add(uv_offset / 2)
            .context("NV12 buffer size overflowed")?;
        if len < required {
            bail!("NV12 buffer is too short: got {len}, need {required}");
        }
        Ok(uv_offset)
    }
}

pub(super) fn to_bgra(
    source: &[u8],
    layout: Layout,
    width: u32,
    height: u32,
    time: MediaTime,
) -> anyhow::Result<VideoFrame> {
    let uv_offset = layout.validate(source.len())?;
    if width == 0 || height == 0 {
        bail!("NV12 output dimensions must be non-zero");
    }
    let bytes_per_row = width
        .checked_mul(4)
        .context("NV12 output row size overflowed")?;
    let len = (bytes_per_row as usize)
        .checked_mul(height as usize)
        .context("NV12 output size overflowed")?;
    #[cfg(target_arch = "x86_64")]
    if (width != layout.width || height != layout.height)
        && std::is_x86_feature_detected!("avx2")
        && u64::from(width) * u64::from(height)
            >= u64::from(layout.width) * u64::from(layout.height) / 8
    {
        // For denser output, vector-convert once instead of four scalar color
        // conversions per bilinear output pixel. The RGB resize is unchanged.
        let full = to_bgra(source, layout, layout.width, layout.height, time)?;
        return super::resize_bgra(
            &full.data,
            full.width,
            full.height,
            full.bytes_per_row,
            width,
            height,
            time,
        );
    }
    let mut data = vec![0; len];
    let coefficients = match layout.matrix {
        Matrix::Bt601 => [409, 100, 208, 516],
        Matrix::Bt709 => [459, 55, 136, 541],
    };
    #[cfg(target_arch = "x86_64")]
    if width == layout.width && height == layout.height && std::is_x86_feature_detected!("avx2") {
        // Layout, plane length, and destination length were checked above.
        unsafe {
            avx2::convert_frame(source, layout, uv_offset, coefficients, &mut data);
        }
        return VideoFrame::new_bgra(width, height, bytes_per_row, time, data);
    }
    let pixel = |x: usize, y: usize| {
        let luma = source[y * layout.stride + x];
        let chroma = uv_offset + (y / 2) * layout.stride + (x & !1);
        convert(luma, source[chroma], source[chroma + 1], coefficients)
    };
    if width == layout.width && height == layout.height {
        for (y, row) in data.chunks_exact_mut(bytes_per_row as usize).enumerate() {
            for (x, output) in row.chunks_exact_mut(4).enumerate() {
                output.copy_from_slice(&pixel(x, y));
            }
        }
    } else {
        let xs = sample_coordinates(width as usize, layout.width as usize);
        let ys = sample_coordinates(height as usize, layout.height as usize);
        let half_weights = xs
            .iter()
            .chain(&ys)
            .all(|coordinate| coordinate.weight == 0.5);
        for (row, y) in data.chunks_exact_mut(bytes_per_row as usize).zip(ys) {
            for (output, x) in row.chunks_exact_mut(4).zip(&xs) {
                let tl = pixel(x.lower, y.lower);
                let tr = pixel(x.upper, y.lower);
                let bl = pixel(x.lower, y.upper);
                let br = pixel(x.upper, y.upper);
                if half_weights {
                    // All intermediate halves/quarters of byte values are
                    // exact in f32. Round their sum once, just like bilinear.
                    for c in 0..3 {
                        output[c] = ((u16::from(tl[c])
                            + u16::from(tr[c])
                            + u16::from(bl[c])
                            + u16::from(br[c])
                            + 2)
                            >> 2) as u8;
                    }
                } else {
                    for c in 0..3 {
                        let top = mix(tl[c] as f32, tr[c] as f32, x.weight);
                        let bottom = mix(bl[c] as f32, br[c] as f32, x.weight);
                        output[c] = mix(top, bottom, y.weight).round().clamp(0.0, 255.0) as u8;
                    }
                }
                output[3] = 255;
            }
        }
    }
    VideoFrame::new_bgra(width, height, bytes_per_row, time, data)
}

fn convert(y: u8, u: u8, v: u8, [rv, gu, gv, bu]: [i32; 4]) -> [u8; 4] {
    let y = 298 * (y as i32 - 16) + 128;
    let u = u as i32 - 128;
    let v = v as i32 - 128;
    [
        ((y + bu * u) >> 8).clamp(0, 255) as u8,
        ((y - gu * u - gv * v) >> 8).clamp(0, 255) as u8,
        ((y + rv * v) >> 8).clamp(0, 255) as u8,
        255,
    ]
}

pub(super) fn same_rgb_frame(a: &VideoFrame, b: &VideoFrame) -> bool {
    if (a.width, a.height, a.time) != (b.width, b.height, b.time) {
        return false;
    }
    let row_len = a.width as usize * 4;
    a.data
        .chunks_exact(a.bytes_per_row as usize)
        .take(a.height as usize)
        .zip(
            b.data
                .chunks_exact(b.bytes_per_row as usize)
                .take(b.height as usize),
        )
        .all(|(a, b)| {
            a[..row_len]
                .iter()
                .zip(&b[..row_len])
                .enumerate()
                .all(|(i, (a, b))| i % 4 == 3 || a == b)
        })
}

// The decoder can expose color metadata and padded coded dimensions only after
// its first sample. Hold that sample so negotiation never consumes a frame.
pub(super) fn configure(
    reader: &IMFSourceReader,
    width: u32,
    height: u32,
) -> anyhow::Result<(Layout, PendingSample)> {
    let media_type = unsafe { MFCreateMediaType()? };
    let mut sample = None;
    let mut flags = 0;
    let mut timestamp = 0;
    unsafe {
        media_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
        media_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)?;
        reader.SetCurrentMediaType(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
            None,
            &media_type,
        )?;
        reader.ReadSample(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
            0,
            None,
            Some(&mut flags),
            Some(&mut timestamp),
            Some(&mut sample),
        )?;
    }
    anyhow::ensure!(
        flags & (MF_SOURCE_READERF_ERROR.0 | MF_SOURCE_READERF_ENDOFSTREAM.0) as u32 == 0,
        "native decoder did not produce a first video frame"
    );
    let sample = sample.context("native decoder did not produce a first sample")?;
    let layout = current_layout(reader, width, height)?;
    ensure_progressive(&sample, layout)?;
    Ok((layout, PendingSample { sample, timestamp }))
}

pub(super) fn ensure_progressive(sample: &IMFSample, layout: Layout) -> anyhow::Result<()> {
    let interlaced = unsafe { sample.GetUINT32(&MFSampleExtension_Interlaced) };
    anyhow::ensure!(
        interlaced.unwrap_or(u32::from(layout.mixed_interlace)) == 0,
        "native sample is interlaced or has no progressive flag"
    );
    Ok(())
}

pub(super) fn current_layout(
    reader: &IMFSourceReader,
    width: u32,
    height: u32,
) -> anyhow::Result<Layout> {
    let media_type =
        unsafe { reader.GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32)? };
    let (coded_width, coded_height) = media_type_dimensions(&media_type)?;
    validate_aperture(&media_type, width, height)?;
    unsafe {
        anyhow::ensure!(
            media_type.GetGUID(&MF_MT_SUBTYPE)? == MFVideoFormat_NV12,
            "native output is not NV12"
        );
        // Do not guess colorimetry or replace deinterlacing. Unsupported inputs
        // reopen through the original RGB processor before emitting any frame.
        let interlace_mode = media_type.GetUINT32(&MF_MT_INTERLACE_MODE)?;
        let mixed_interlace =
            interlace_mode == MFVideoInterlace_MixedInterlaceOrProgressive.0 as u32;
        anyhow::ensure!(
            interlace_mode == MFVideoInterlace_Progressive.0 as u32 || mixed_interlace,
            "native video is not progressive (mode {interlace_mode})"
        );
        anyhow::ensure!(
            media_type.GetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE)? == MFNominalRange_16_235.0 as u32,
            "native video is not explicitly limited-range"
        );
        let matrix = match media_type.GetUINT32(&MF_MT_YUV_MATRIX)? {
            value if value == MFVideoTransferMatrix_BT709.0 as u32 => Matrix::Bt709,
            value if value == MFVideoTransferMatrix_BT601.0 as u32 => Matrix::Bt601,
            value => bail!("unsupported native YUV matrix {value}"),
        };
        let stride = media_type.GetUINT32(&MF_MT_DEFAULT_STRIDE)? as i32;
        anyhow::ensure!(
            width > 0
                && height > 0
                && coded_width >= width
                && coded_height >= height
                && stride > 0
                && stride as u32 >= coded_width
                && stride % 2 == 0
                && coded_height % 2 == 0,
            "unsupported native NV12 dimensions or stride"
        );
        Ok(Layout {
            stride: stride as usize,
            coded_height: coded_height as usize,
            width,
            height,
            matrix,
            mixed_interlace,
        })
    }
}

fn validate_aperture(media_type: &IMFMediaType, width: u32, height: u32) -> anyhow::Result<()> {
    anyhow::ensure!(
        unsafe { media_type.GetUINT32(&MF_MT_PAN_SCAN_ENABLED) }.unwrap_or(0) == 0,
        "native video requires pan-scan processing"
    );
    for key in [
        MF_MT_MINIMUM_DISPLAY_APERTURE,
        MF_MT_GEOMETRIC_APERTURE,
        MF_MT_PAN_SCAN_APERTURE,
    ] {
        if let Ok(len) = unsafe { media_type.GetBlobSize(&key) } {
            // MFVideoArea: two four-byte MFOffsets, then two signed dimensions.
            anyhow::ensure!(len == 16, "invalid native video aperture");
            let mut area = [0; 16];
            unsafe {
                media_type.GetBlob(&key, &mut area, None)?;
            }
            anyhow::ensure!(
                area[..8] == [0; 8]
                    && u32::from_le_bytes(area[8..12].try_into().unwrap()) == width
                    && u32::from_le_bytes(area[12..16].try_into().unwrap()) == height,
                "native video requires cropping or aperture scaling"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_aperture_and_per_sample_interlace_flags() -> anyhow::Result<()> {
        use windows::Win32::{
            Media::MediaFoundation::{MFShutdown, MFStartup, MFSTARTUP_NOSOCKET, MF_API_VERSION},
            System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED},
        };
        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.ok()?;
        scopeguard::defer! { unsafe { CoUninitialize(); } }
        unsafe {
            MFStartup(MF_API_VERSION, MFSTARTUP_NOSOCKET)?;
        }
        scopeguard::defer! { unsafe { let _ = MFShutdown(); } }
        let media_type = unsafe { MFCreateMediaType()? };
        let mut area = [0; 16];
        area[8..12].copy_from_slice(&8u32.to_le_bytes());
        area[12..16].copy_from_slice(&6u32.to_le_bytes());
        unsafe {
            media_type.SetBlob(&MF_MT_MINIMUM_DISPLAY_APERTURE, &area)?;
        }
        validate_aperture(&media_type, 8, 6)?;
        assert!(validate_aperture(&media_type, 8, 4).is_err());
        area[0] = 1; // Even a fractional offset must not be ignored.
        unsafe {
            media_type.SetBlob(&MF_MT_MINIMUM_DISPLAY_APERTURE, &area)?;
        }
        assert!(validate_aperture(&media_type, 8, 6).is_err());
        let sample = unsafe { windows::Win32::Media::MediaFoundation::MFCreateSample()? };
        let layout = Layout {
            width: 8,
            height: 6,
            stride: 8,
            coded_height: 6,
            matrix: Matrix::Bt709,
            mixed_interlace: true,
        };
        assert!(ensure_progressive(&sample, layout).is_err());
        ensure_progressive(
            &sample,
            Layout {
                mixed_interlace: false,
                ..layout
            },
        )?;
        unsafe {
            sample.SetUINT32(&MFSampleExtension_Interlaced, 0)?;
        }
        ensure_progressive(&sample, layout)?;
        unsafe {
            sample.SetUINT32(&MFSampleExtension_Interlaced, 1)?;
        }
        assert!(ensure_progressive(&sample, layout).is_err());
        Ok(())
    }

    fn layout(width: u32, height: u32, matrix: Matrix) -> Layout {
        Layout {
            width,
            height,
            stride: (width as usize + 7) & !1,
            coded_height: (height as usize + 7) & !1,
            matrix,
            mixed_interlace: false,
        }
    }

    #[test]
    fn fused_conversion_matches_convert_then_original_resize() {
        let time = MediaTime::new(31, 30).unwrap();
        for matrix in [Matrix::Bt601, Matrix::Bt709] {
            for (sw, sh, tw, th) in [
                (17, 13, 6, 4),
                (3, 2, 10, 8),
                (1, 5, 7, 1),
                (9, 1, 1, 7),
                (8, 6, 8, 6),
                (32, 24, 8, 6),
                (8, 6, 4, 3),
            ] {
                let layout = layout(sw, sh, matrix);
                let source: Vec<u8> = (0..layout.stride * layout.coded_height * 3 / 2)
                    .map(|i| ((i * 73 + i / 7) % 256) as u8)
                    .collect();
                let full = to_bgra(&source, layout, sw, sh, time).unwrap();
                let actual = to_bgra(&source, layout, tw, th, time).unwrap();
                let expected =
                    super::super::resize_bgra(&full.data, sw, sh, full.bytes_per_row, tw, th, time)
                        .unwrap();
                assert_eq!(
                    actual.data, expected.data,
                    "{matrix:?} {sw}x{sh} -> {tw}x{th}"
                );
                assert_eq!(actual.time, time);
            }
        }
    }

    #[test]
    fn neutral_chroma_is_exact_and_opaque() {
        for y in 0..=255 {
            let gray = ((298 * (i32::from(y) - 16) + 128) >> 8).clamp(0, 255) as u8;
            assert_eq!(
                convert(y, 128, 128, [459, 55, 136, 541]),
                [gray, gray, gray, 255]
            );
        }
    }

    #[test]
    fn rejects_invalid_layouts_and_short_buffers() {
        let valid = layout(8, 6, Matrix::Bt709);
        assert!(valid.validate(1).is_err());
        for invalid in [
            Layout { stride: 7, ..valid },
            Layout {
                coded_height: 5,
                ..valid
            },
            Layout { width: 0, ..valid },
            Layout {
                coded_height: usize::MAX - 1,
                ..valid
            },
        ] {
            assert!(invalid.validate(usize::MAX).is_err());
        }
        let source = vec![0; valid.stride * valid.coded_height * 3 / 2];
        assert!(to_bgra(&source, valid, 0, 1, MediaTime::new(0, 1).unwrap()).is_err());
    }
}
