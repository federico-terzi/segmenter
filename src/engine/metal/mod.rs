use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int},
    path::PathBuf,
    ptr::NonNull,
};

use anyhow::{bail, Context};

use crate::{
    engine::Engine,
    frame::{MediaTime, PixelFormat, VideoFrame},
};

const ERROR_LEN: usize = 4096;

#[repr(C)]
struct SegmenterRvmMetalContext {
    _private: [u8; 0],
}

extern "C" {
    fn segmenter_rvm_metal_create(
        model_path: *const c_char,
        error: *mut c_char,
        error_len: usize,
    ) -> *mut SegmenterRvmMetalContext;
    fn segmenter_rvm_metal_run(
        context: *mut SegmenterRvmMetalContext,
        input_nchw: *const f32,
        input_len: usize,
        width: u32,
        height: u32,
        downsample_ratio: f32,
        alpha_nchw: *mut f32,
        alpha_len: usize,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;
    fn segmenter_rvm_metal_destroy(context: *mut SegmenterRvmMetalContext);
}

pub struct MetalEngine {
    context: NonNull<SegmenterRvmMetalContext>,
    downsample_ratio: f32,
}

impl MetalEngine {
    pub fn new(model_path: PathBuf, downsample_ratio: f32) -> anyhow::Result<Self> {
        if !model_path.exists() {
            bail!(
                "RVM Metal model path does not exist: {}",
                model_path.display()
            );
        }
        if !downsample_ratio.is_finite() || downsample_ratio <= 0.0 || downsample_ratio > 1.0 {
            bail!(
                "RVM downsample ratio must be finite and within (0, 1], got {}",
                downsample_ratio
            );
        }

        let path = CString::new(model_path.as_os_str().to_string_lossy().as_bytes())
            .context("RVM Metal model path contained an interior NUL byte")?;
        let mut error = ErrorBuffer::new();
        let context =
            unsafe { segmenter_rvm_metal_create(path.as_ptr(), error.as_mut_ptr(), ERROR_LEN) };
        let context = NonNull::new(context)
            .ok_or_else(|| error.into_error("failed to create RVM Metal backend"))?;

        Ok(Self {
            context,
            downsample_ratio,
        })
    }
}

impl Engine for MetalEngine {
    fn segment(&mut self, frame: &VideoFrame) -> anyhow::Result<VideoFrame> {
        if frame.format != PixelFormat::Bgra {
            bail!("RVM Metal engine only accepts BGRA frames");
        }

        let input = preprocess_f32(frame)?;
        let mut alpha = vec![0.0_f32; frame.width as usize * frame.height as usize];
        let mut error = ErrorBuffer::new();
        let status = unsafe {
            segmenter_rvm_metal_run(
                self.context.as_ptr(),
                input.as_ptr(),
                input.len(),
                frame.width,
                frame.height,
                self.downsample_ratio,
                alpha.as_mut_ptr(),
                alpha.len(),
                error.as_mut_ptr(),
                ERROR_LEN,
            )
        };
        if status != 0 {
            return Err(error.into_error("RVM Metal inference failed"));
        }

        alpha_to_mask(&alpha, frame.width, frame.height, frame.time)
    }
}

impl Drop for MetalEngine {
    fn drop(&mut self) {
        unsafe { segmenter_rvm_metal_destroy(self.context.as_ptr()) }
    }
}

fn preprocess_f32(frame: &VideoFrame) -> anyhow::Result<Vec<f32>> {
    let plane_len = frame.width as usize * frame.height as usize;
    let mut tensor = vec![0.0_f32; plane_len * 3];

    for y in 0..frame.height as usize {
        for x in 0..frame.width as usize {
            let pixel_index = y * frame.width as usize + x;
            let source = frame.checked_pixel_offset(x as u32, y as u32)?;
            tensor[pixel_index] = frame.data[source + 2] as f32 / 255.0;
            tensor[plane_len + pixel_index] = frame.data[source + 1] as f32 / 255.0;
            tensor[2 * plane_len + pixel_index] = frame.data[source] as f32 / 255.0;
        }
    }

    Ok(tensor)
}

fn alpha_to_mask(
    alpha: &[f32],
    width: u32,
    height: u32,
    time: MediaTime,
) -> anyhow::Result<VideoFrame> {
    let expected = width as usize * height as usize;
    if alpha.len() != expected {
        bail!(
            "RVM Metal alpha output had invalid length: expected {}, got {}",
            expected,
            alpha.len()
        );
    }

    let bytes_per_row = width
        .checked_mul(4)
        .context("RVM Metal mask row width overflowed")?;
    let mut data = vec![0_u8; bytes_per_row as usize * height as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let value = alpha[y * width as usize + x];
            if !value.is_finite() {
                bail!("RVM Metal alpha output contained a non-finite value");
            }
            let alpha = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            let offset = y * bytes_per_row as usize + x * 4;
            data[offset] = alpha;
            data[offset + 1] = alpha;
            data[offset + 2] = alpha;
            data[offset + 3] = 255;
        }
    }

    VideoFrame::new_bgra(width, height, bytes_per_row, time, data)
}

struct ErrorBuffer {
    bytes: Vec<c_char>,
}

impl ErrorBuffer {
    fn new() -> Self {
        Self {
            bytes: vec![0; ERROR_LEN],
        }
    }

    fn as_mut_ptr(&mut self) -> *mut c_char {
        self.bytes.as_mut_ptr()
    }

    fn into_error(self, fallback: &'static str) -> anyhow::Error {
        let message = unsafe { CStr::from_ptr(self.bytes.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if message.is_empty() {
            anyhow::anyhow!(fallback)
        } else {
            anyhow::anyhow!(message)
        }
    }
}
