use std::{
    collections::VecDeque,
    ffi::{c_void, CStr, CString},
    os::raw::{c_char, c_int},
    path::PathBuf,
    ptr::NonNull,
};

use anyhow::{bail, Context};

use crate::{
    engine::Engine,
    frame::{validate_bgra_buffer, FrameData, FramePool, MediaTime, PixelFormat, VideoFrame},
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
    fn segmenter_rvm_metal_queue_capacity(context: *mut SegmenterRvmMetalContext) -> usize;
    fn segmenter_rvm_metal_destroy(context: *mut SegmenterRvmMetalContext);
    fn segmenter_rvm_metal_submit(
        context: *mut SegmenterRvmMetalContext,
        bgra: *const u8,
        input_len: usize,
        width: u32,
        height: u32,
        bytes_per_row: u32,
        downsample_ratio: f32,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;
    fn segmenter_rvm_metal_submit_pixel_buffer(
        context: *mut SegmenterRvmMetalContext,
        pixel_buffer: *mut c_void,
        downsample_ratio: f32,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;
    fn segmenter_rvm_metal_receive_buffer(
        context: *mut SegmenterRvmMetalContext,
        lease: *mut *mut c_void,
        data: *mut *const u8,
        length: *mut usize,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;
    fn segmenter_rvm_metal_release_buffer(lease: *mut c_void);
}

pub struct MetalEngine {
    context: NonNull<SegmenterRvmMetalContext>,
    downsample_ratio: f32,
    pending: VecDeque<(u32, u32, MediaTime)>,
    completed: VecDeque<VideoFrame>,
    pool: FramePool,
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
            pending: VecDeque::new(),
            completed: VecDeque::new(),
            pool: FramePool::default(),
        })
    }

    fn receive(&mut self) -> anyhow::Result<Option<VideoFrame>> {
        let Some(&(width, height, time)) = self.pending.front() else {
            return Ok(None);
        };
        let stride = width.checked_mul(4).context("Metal mask row overflow")?;
        let mut lease = std::ptr::null_mut();
        let mut pixels = std::ptr::null();
        let mut len = 0;
        let mut error = ErrorBuffer::new();
        let status = unsafe {
            segmenter_rvm_metal_receive_buffer(
                self.context.as_ptr(),
                &mut lease,
                &mut pixels,
                &mut len,
                error.as_mut_ptr(),
                ERROR_LEN,
            )
        };
        if status != 0 {
            return Err(error.into_error("RVM Metal receive failed"));
        }
        if lease.is_null() || pixels.is_null() {
            if !lease.is_null() {
                unsafe { segmenter_rvm_metal_release_buffer(lease) };
            }
            bail!("Metal returned an empty mask buffer");
        }
        // Native receive waits for the GPU and transfers exclusive ownership of
        // this immutable mask. The pool cannot reuse it until the encoder drops it.
        let mask = unsafe {
            FrameData::from_native(
                lease,
                pixels,
                len,
                std::ptr::null_mut(),
                segmenter_rvm_metal_release_buffer,
            )
        };
        self.pending.pop_front();
        VideoFrame::from_bgra_data(width, height, stride, time, mask)
            .map(|frame| Some(frame.into_grayscale_mask()))
    }
}

impl Engine for MetalEngine {
    fn output_frame_info(&self, frame: &VideoFrame) -> Option<crate::frame::VideoFrameInfo> {
        let mut info = frame.info();
        info.grayscale_mask = true;
        Some(info)
    }

    fn enqueue(&mut self, frame: &VideoFrame) -> anyhow::Result<Option<VideoFrame>> {
        if frame.format != PixelFormat::Bgra {
            bail!("RVM Metal engine only accepts BGRA frames");
        }
        validate_bgra_buffer(
            frame.width,
            frame.height,
            frame.bytes_per_row,
            frame.data.len(),
        )?;
        if self
            .pending
            .back()
            .is_some_and(|&(w, h, _)| (w, h) != (frame.width, frame.height))
        {
            while let Some(mask) = self.receive()? {
                self.completed.push_back(mask);
            }
        }
        let mut error = ErrorBuffer::new();
        let status = unsafe {
            if !frame.data.pixel_buffer().is_null() {
                segmenter_rvm_metal_submit_pixel_buffer(
                    self.context.as_ptr(),
                    frame.data.pixel_buffer(),
                    self.downsample_ratio,
                    error.as_mut_ptr(),
                    ERROR_LEN,
                )
            } else {
                segmenter_rvm_metal_submit(
                    self.context.as_ptr(),
                    frame.data.as_ptr(),
                    frame.data.len(),
                    frame.width,
                    frame.height,
                    frame.bytes_per_row,
                    self.downsample_ratio,
                    error.as_mut_ptr(),
                    ERROR_LEN,
                )
            }
        };
        if status != 0 {
            return Err(error.into_error("RVM Metal submit failed"));
        }
        self.pending
            .push_back((frame.width, frame.height, frame.time));
        // Drain before the bounded native ring is reused.
        if self.pending.len()
            == unsafe { segmenter_rvm_metal_queue_capacity(self.context.as_ptr()) }
        {
            let mask = self.receive()?.context("Metal queue unexpectedly empty")?;
            self.completed.push_back(mask);
        }
        Ok(self.completed.pop_front())
    }

    fn drain(&mut self) -> anyhow::Result<Option<VideoFrame>> {
        match self.completed.pop_front() {
            Some(mask) => Ok(Some(mask)),
            None => self.receive(),
        }
    }

    fn segment(&mut self, frame: &VideoFrame) -> anyhow::Result<VideoFrame> {
        if !self.pending.is_empty() || !self.completed.is_empty() {
            bail!("drain queued Metal frames before synchronous inference");
        }
        if !frame.data.pixel_buffer().is_null() {
            if let Some(mask) = self.enqueue(frame)? {
                return Ok(mask);
            }
            return self.receive()?.context("Metal synchronous queue was empty");
        }
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
        let mut mask = self
            .pool
            .take(bytes_per_row as usize * frame.height as usize);
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
            .map(|frame| frame.with_pool(&self.pool).into_grayscale_mask())
    }
}

impl Drop for MetalEngine {
    fn drop(&mut self) {
        unsafe { segmenter_rvm_metal_destroy(self.context.as_ptr()) }
    }
}

struct ErrorBuffer {
    bytes: [c_char; ERROR_LEN],
}

impl ErrorBuffer {
    fn new() -> Self {
        Self {
            bytes: [0; ERROR_LEN],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Metal device and SEGMENTER_TEST_MODEL (or the local RVM model)"]
    fn queued_masks_match_synchronous_with_padding_and_resolution_changes() {
        let model = std::env::var_os("SEGMENTER_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("rvm_mobilenetv3_fp16.rvmmetal"));
        let mut synchronous = MetalEngine::new(model.clone(), 0.25).unwrap();
        let mut queued = MetalEngine::new(model.clone(), 0.25).unwrap();
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        for i in 0..11 {
            let (width, height) = if i < 5 { (171, 129) } else { (128, 96) };
            let stride = width * 4 + 28;
            let mut data = vec![0xa5; (stride * height) as usize];
            for y in 0..height {
                for x in 0..width {
                    let offset = (y * stride + x * 4) as usize;
                    data[offset..offset + 4].copy_from_slice(&[
                        (x + i * 17) as u8,
                        (y + i * 31) as u8,
                        (x + y) as u8,
                        255,
                    ]);
                }
            }
            let frame = VideoFrame::new_bgra(
                width,
                height,
                stride,
                MediaTime::new(i as i64 * 1001, 30000).unwrap(),
                data,
            )
            .unwrap();
            expected.push(synchronous.segment(&frame).unwrap());
            if let Some(mask) = queued.enqueue(&frame).unwrap() {
                actual.push(mask);
            }
        }
        while let Some(mask) = queued.drain().unwrap() {
            actual.push(mask);
        }
        assert!(queued.drain().unwrap().is_none());
        // Leased masks remain readable after the producing engine and runtime
        // are destroyed, including masks retained across a resolution change.
        drop(queued);
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!((a.width, a.height, a.time), (b.width, b.height, b.time));
            assert_eq!(a.data.len(), b.data.len());
            // Synchronous execution selects a one-frame graph; the queued graph
            // batches convolutions. Their FP16 reduction order can differ slightly.
            let differences: Vec<_> = a
                .data
                .iter()
                .zip(b.data.iter())
                .map(|(x, y)| x.abs_diff(*y))
                .collect();
            let max_error = differences.iter().copied().max().unwrap_or(0);
            let mean_error =
                differences.iter().map(|&x| x as f64).sum::<f64>() / differences.len() as f64;
            assert!(
                max_error <= 5 && mean_error < 0.01,
                "queued mask differs from synchronous mask: max={max_error}, mean={mean_error}"
            );
        }
        // Destruction must wait for any outstanding GPU users of the ring.
        let mut abandoned = MetalEngine::new(model, 0.25).unwrap();
        assert!(abandoned.enqueue(&expected[0]).unwrap().is_none());
        assert!(abandoned.enqueue(&expected[1]).unwrap().is_none());
        drop(abandoned);
    }
}
