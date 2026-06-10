use anyhow::{bail, Context};
use half::f16;
use ndarray::{Array4, ArrayView4};

use crate::frame::{MediaTime, VideoFrame};

pub(super) fn preprocess_f16(frame: &VideoFrame) -> anyhow::Result<Array4<f16>> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut tensor = Array4::<f16>::from_elem([1, 3, height, width], f16::from_f32(0.0));

    for y in 0..height {
        for x in 0..width {
            let source = frame.checked_pixel_offset(x as u32, y as u32)?;
            tensor[[0, 0, y, x]] = f16::from_f32(frame.data[source + 2] as f32 / 255.0);
            tensor[[0, 1, y, x]] = f16::from_f32(frame.data[source + 1] as f32 / 255.0);
            tensor[[0, 2, y, x]] = f16::from_f32(frame.data[source] as f32 / 255.0);
        }
    }

    Ok(tensor)
}

pub(super) fn preprocess_f32(frame: &VideoFrame) -> anyhow::Result<Array4<f32>> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut tensor = Array4::<f32>::zeros([1, 3, height, width]);

    for y in 0..height {
        for x in 0..width {
            let source = frame.checked_pixel_offset(x as u32, y as u32)?;
            tensor[[0, 0, y, x]] = frame.data[source + 2] as f32 / 255.0;
            tensor[[0, 1, y, x]] = frame.data[source + 1] as f32 / 255.0;
            tensor[[0, 2, y, x]] = frame.data[source] as f32 / 255.0;
        }
    }

    Ok(tensor)
}

pub(super) fn alpha_to_mask_f16(
    alpha: ArrayView4<'_, f16>,
    width: u32,
    height: u32,
    time: MediaTime,
) -> anyhow::Result<VideoFrame> {
    let shape = alpha.shape();
    validate_alpha_shape(shape)?;

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
