use anyhow::{bail, Context};
use half::f16;
use ndarray::ArrayView4;
use std::sync::OnceLock;

use crate::frame::{validate_bgra_buffer, MediaTime, VideoFrame};

pub(super) fn preprocess_f16(frame: &VideoFrame, tensor: &mut [f16]) -> anyhow::Result<()> {
    static NORMALIZED: OnceLock<[f16; 256]> = OnceLock::new();
    // Match the original division and f16 rounding exactly for every u8 value.
    let normalized =
        NORMALIZED.get_or_init(|| std::array::from_fn(|i| f16::from_f32(i as f32 / 255.0)));
    preprocess(frame, tensor, normalized)
}

pub(super) fn preprocess_f32(frame: &VideoFrame, tensor: &mut [f32]) -> anyhow::Result<()> {
    static NORMALIZED: OnceLock<[f32; 256]> = OnceLock::new();
    let normalized = NORMALIZED.get_or_init(|| std::array::from_fn(|i| i as f32 / 255.0));
    preprocess(frame, tensor, normalized)
}

fn preprocess<T: Copy>(
    frame: &VideoFrame,
    tensor: &mut [T],
    normalized: &[T; 256],
) -> anyhow::Result<()> {
    validate_bgra_buffer(
        frame.width,
        frame.height,
        frame.bytes_per_row,
        frame.data.len(),
    )?;
    let width = frame.width as usize;
    let height = frame.height as usize;
    let plane_len = width
        .checked_mul(height)
        .context("input tensor size overflowed")?;
    if Some(tensor.len()) != plane_len.checked_mul(3) {
        bail!("input tensor size does not match frame dimensions");
    }
    let (red, rest) = tensor.split_at_mut(plane_len);
    let (green, blue) = rest.split_at_mut(plane_len);
    for (y, row) in frame
        .data
        .chunks_exact(frame.bytes_per_row as usize)
        .take(height)
        .enumerate()
    {
        let offset = y * width;
        for (x, pixel) in row[..width * 4].chunks_exact(4).enumerate() {
            red[offset + x] = normalized[pixel[2] as usize];
            green[offset + x] = normalized[pixel[1] as usize];
            blue[offset + x] = normalized[pixel[0] as usize];
        }
    }
    Ok(())
}

pub(super) fn alpha_to_mask_f16(
    alpha: ArrayView4<'_, f16>,
    width: u32,
    height: u32,
    time: MediaTime,
) -> anyhow::Result<VideoFrame> {
    let shape = alpha.shape();
    validate_alpha_shape(shape)?;

    if same_size_sampling_is_identity(shape, width, height) {
        if let Some(values) = alpha.as_slice() {
            static QUANTIZED: OnceLock<Box<[u16]>> = OnceLock::new();
            let quantized = QUANTIZED.get_or_init(|| {
                (0..=u16::MAX)
                    .map(|bits| quantize_alpha(f16::from_bits(bits).to_f32()))
                    .collect()
            });
            return direct_alpha_to_mask(
                values.iter().map(|v| quantized[v.to_bits() as usize]),
                width,
                height,
                time,
            );
        }
    }

    let source_x = sample_coordinates(width as usize, shape[3]);
    let source_y = sample_coordinates(height as usize, shape[2]);
    let bytes_per_row = width.checked_mul(4).context("mask row width overflowed")?;
    let mut data = vec![0_u8; bytes_per_row as usize * height as usize];

    for (target_y, sample_y) in source_y.into_iter().enumerate() {
        for (target_x, sample_x) in source_x.iter().copied().enumerate() {
            let value = sample_alpha_f16(alpha, sample_x, sample_y);
            write_mask_pixel(&mut data, bytes_per_row, target_x, target_y, value)?;
        }
    }

    VideoFrame::new_bgra(width, height, bytes_per_row, time, data)
}

pub(super) fn alpha_to_mask_f32(
    alpha: ArrayView4<'_, f32>,
    width: u32,
    height: u32,
    time: MediaTime,
) -> anyhow::Result<VideoFrame> {
    let shape = alpha.shape();
    validate_alpha_shape(shape)?;

    if same_size_sampling_is_identity(shape, width, height) {
        if let Some(values) = alpha.as_slice() {
            return direct_alpha_to_mask(
                values.iter().map(|&v| quantize_alpha(v)),
                width,
                height,
                time,
            );
        }
    }

    let source_x = sample_coordinates(width as usize, shape[3]);
    let source_y = sample_coordinates(height as usize, shape[2]);
    let bytes_per_row = width.checked_mul(4).context("mask row width overflowed")?;
    let mut data = vec![0_u8; bytes_per_row as usize * height as usize];

    for (target_y, sample_y) in source_y.into_iter().enumerate() {
        for (target_x, sample_x) in source_x.iter().copied().enumerate() {
            let value = sample_alpha_f32(alpha, sample_x, sample_y);
            write_mask_pixel(&mut data, bytes_per_row, target_x, target_y, value)?;
        }
    }

    VideoFrame::new_bgra(width, height, bytes_per_row, time, data)
}

fn same_size_sampling_is_identity(shape: &[usize], width: u32, height: u32) -> bool {
    fn exact_grid(len: usize) -> bool {
        // Equal dimensions alone do not guarantee identity: the original f32
        // coordinate calculation can round differently at large dimensions.
        // These bounds keep its half-integer products exactly representable;
        // avoid scanning the coordinate grid on every ordinary video frame.
        if len <= 2048 || (len <= 4096 && len % 2 == 0) {
            return true;
        }
        (0..len).all(|index| {
            let coordinate = source_coordinate(index, len, len);
            coordinate.lower == index && coordinate.weight == 0.0
        })
    }
    shape[2] == height as usize
        && shape[3] == width as usize
        && exact_grid(width as usize)
        && exact_grid(height as usize)
}

// 256 is a sentinel for non-finite values, never a valid mask byte.
fn quantize_alpha(value: f32) -> u16 {
    if value.is_finite() {
        (value.clamp(0.0, 1.0) * 255.0).round() as u16
    } else {
        256
    }
}

fn direct_alpha_to_mask(
    values: impl Iterator<Item = u16>,
    width: u32,
    height: u32,
    time: MediaTime,
) -> anyhow::Result<VideoFrame> {
    let bytes_per_row = width.checked_mul(4).context("mask row width overflowed")?;
    let len = (bytes_per_row as usize)
        .checked_mul(height as usize)
        .context("mask size overflowed")?;
    let mut data = vec![0; len];
    for (pixel, value) in data.chunks_exact_mut(4).zip(values) {
        if value > 255 {
            bail!("RVM alpha output contained a non-finite value");
        }
        let alpha = value as u8;
        pixel.copy_from_slice(&[alpha, alpha, alpha, 255]);
    }
    VideoFrame::new_bgra(width, height, bytes_per_row, time, data)
}

fn write_mask_pixel(
    data: &mut [u8],
    bytes_per_row: u32,
    x: usize,
    y: usize,
    value: f32,
) -> anyhow::Result<()> {
    if !value.is_finite() {
        bail!("RVM alpha output contained a non-finite value");
    }

    let alpha = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    let offset = y * bytes_per_row as usize + x * 4;
    data[offset] = alpha;
    data[offset + 1] = alpha;
    data[offset + 2] = alpha;
    data[offset + 3] = 255;

    Ok(())
}

fn validate_alpha_shape(shape: &[usize]) -> anyhow::Result<()> {
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] == 0 || shape[3] == 0 {
        bail!("RVM alpha output had invalid shape {:?}", shape);
    }

    Ok(())
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

fn sample_alpha_f16(alpha: ArrayView4<'_, f16>, x: SampleCoordinate, y: SampleCoordinate) -> f32 {
    let top = mix(
        alpha[[0, 0, y.lower, x.lower]].to_f32(),
        alpha[[0, 0, y.lower, x.upper]].to_f32(),
        x.weight,
    );
    let bottom = mix(
        alpha[[0, 0, y.upper, x.lower]].to_f32(),
        alpha[[0, 0, y.upper, x.upper]].to_f32(),
        x.weight,
    );

    mix(top, bottom, y.weight)
}

fn sample_alpha_f32(alpha: ArrayView4<'_, f32>, x: SampleCoordinate, y: SampleCoordinate) -> f32 {
    let top = mix(
        alpha[[0, 0, y.lower, x.lower]],
        alpha[[0, 0, y.lower, x.upper]],
        x.weight,
    );
    let bottom = mix(
        alpha[[0, 0, y.upper, x.lower]],
        alpha[[0, 0, y.upper, x.upper]],
        x.weight,
    );

    mix(top, bottom, y.weight)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, Array4};

    #[test]
    fn fp32_identity_path_matches_original_at_rounding_thresholds_and_large_sizes() {
        for width in [270, 480, 720, 1920, 3840, 4097, 5002] {
            let values = (0..width * 2)
                .map(|i| {
                    let threshold = ((i % 255) as f32 + 0.5) / 255.0;
                    f32::from_bits(threshold.to_bits() + (i % 3) as u32 - 1)
                })
                .collect();
            let alpha = Array4::from_shape_vec([1, 1, 2, width], values).unwrap();
            let frame =
                alpha_to_mask_f32(alpha.view(), width as u32, 2, MediaTime::new(0, 1).unwrap())
                    .unwrap();
            for (i, pixel) in frame.data.chunks_exact(4).enumerate() {
                let reference = sample_alpha_f32(
                    alpha.view(),
                    source_coordinate(i % width, width, width),
                    source_coordinate(i / width, 2, 2),
                );
                assert_eq!(
                    pixel[0],
                    (reference.clamp(0.0, 1.0) * 255.0).round() as u8,
                    "width={width} pixel={i}"
                );
            }
        }
    }

    #[test]
    fn preprocessing_preserves_every_byte_value_and_ignores_row_padding() {
        let mut frame =
            VideoFrame::new_bgra(256, 2, 1032, MediaTime::new(7, 30).unwrap(), vec![99; 2064])
                .unwrap();
        for y in 0..2 {
            for x in 0..256 {
                let p = y * 1032 + x * 4;
                frame.data[p..p + 4].copy_from_slice(&[
                    x as u8,
                    (255 - x) as u8,
                    ((x + 113) % 256) as u8,
                    255,
                ]);
            }
        }
        let mut fp16 = vec![f16::ZERO; 1536];
        let mut fp32 = vec![0.0; 1536];
        preprocess_f16(&frame, &mut fp16).unwrap();
        preprocess_f32(&frame, &mut fp32).unwrap();
        for c in 0..3 {
            for y in 0..2 {
                for x in 0..256 {
                    let value = frame.data[y * 1032 + x * 4 + 2 - c] as f32 / 255.0;
                    let p = c * 512 + y * 256 + x;
                    assert_eq!(fp32[p].to_bits(), value.to_bits());
                    assert_eq!(fp16[p].to_bits(), f16::from_f32(value).to_bits());
                }
            }
        }
        frame.data.truncate(10);
        assert!(preprocess_f16(&frame, &mut fp16).is_err());
    }

    #[test]
    fn fp16_mask_conversion_matches_original_for_every_finite_half() {
        let values: Vec<_> = (0..=u16::MAX)
            .map(f16::from_bits)
            .map(|v| if v.is_finite() { v } else { f16::ZERO })
            .collect();
        let alpha = Array4::from_shape_vec([1, 1, 256, 256], values).unwrap();
        let result =
            alpha_to_mask_f16(alpha.view(), 256, 256, MediaTime::new(0, 1).unwrap()).unwrap();
        for (index, pixel) in result.data.chunks_exact(4).enumerate() {
            let x = source_coordinate(index % 256, 256, 256);
            let y = source_coordinate(index / 256, 256, 256);
            let reference =
                (sample_alpha_f16(alpha.view(), x, y).clamp(0.0, 1.0) * 255.0).round() as u8;
            assert_eq!(pixel, &[reference, reference, reference, 255]);
        }
    }

    #[test]
    fn mask_resampling_and_noncontiguous_tensors_preserve_bilinear_values() {
        let alpha = Array4::from_shape_vec([1, 1, 2, 2], vec![0.0, 0.25, 0.75, 1.0]).unwrap();
        let time = MediaTime::new(0, 1).unwrap();
        let result = alpha_to_mask_f32(alpha.view(), 4, 4, time).unwrap();
        assert_eq!(
            result
                .data
                .chunks_exact(4)
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            vec![0, 16, 48, 64, 48, 64, 96, 112, 143, 159, 191, 207, 191, 207, 239, 255]
        );
        let reversed = alpha.slice(s![.., .., .., ..;-1]);
        let result = alpha_to_mask_f32(reversed, 2, 2, time).unwrap();
        assert_eq!(
            result
                .data
                .chunks_exact(4)
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            vec![64, 0, 255, 191]
        );
    }

    #[test]
    fn masks_reject_nonfinite_values_in_fast_and_resampled_paths() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let alpha = Array4::from_elem([1, 1, 2, 2], value);
            for size in [2, 4] {
                let time = MediaTime::new(0, 1).unwrap();
                assert!(alpha_to_mask_f32(alpha.view(), size, size, time).is_err());
                assert!(
                    alpha_to_mask_f16(alpha.mapv(f16::from_f32).view(), size, size, time).is_err()
                );
            }
        }
    }
}
