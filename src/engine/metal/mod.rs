use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int},
    path::PathBuf,
    ptr::NonNull,
};

use anyhow::{bail, Context};

use crate::{
    engine::Engine,
    frame::{validate_bgra_buffer, PixelFormat, VideoFrame},
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
        bgra: *const u8,
        input_len: usize,
        width: u32,
        height: u32,
        bytes_per_row: u32,
        downsample_ratio: f32,
        mask: *mut u8,
        mask_len: usize,
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

        validate_bgra_buffer(
            frame.width,
            frame.height,
            frame.bytes_per_row,
            frame.data.len(),
        )?;
        let bytes_per_row = frame
            .width
            .checked_mul(4)
            .context("RVM Metal mask row width overflowed")?;
        let mut mask = vec![0u8; bytes_per_row as usize * frame.height as usize];
        let mut error = ErrorBuffer::new();
        let status = unsafe {
            segmenter_rvm_metal_run(
                self.context.as_ptr(),
                frame.data.as_ptr(),
                frame.data.len(),
                frame.width,
                frame.height,
                frame.bytes_per_row,
                self.downsample_ratio,
                mask.as_mut_ptr(),
                mask.len(),
                error.as_mut_ptr(),
                ERROR_LEN,
            )
        };
        if status != 0 {
            return Err(error.into_error("RVM Metal inference failed"));
        }

        VideoFrame::new_bgra(frame.width, frame.height, bytes_per_row, frame.time, mask)
    }
}

impl Drop for MetalEngine {
    fn drop(&mut self) {
        unsafe { segmenter_rvm_metal_destroy(self.context.as_ptr()) }
    }
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
