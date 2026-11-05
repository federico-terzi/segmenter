use anyhow::{bail, Context};

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::FrameData;
#[cfg(not(target_os = "macos"))]
pub type FrameData = Vec<u8>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayRotation {
    None,
    Clockwise90,
    Clockwise180,
    Clockwise270,
}

impl DisplayRotation {
    pub fn from_clockwise_degrees(degrees: i32) -> anyhow::Result<Self> {
        match degrees.rem_euclid(360) {
            0 => Ok(Self::None),
            90 => Ok(Self::Clockwise90),
            180 => Ok(Self::Clockwise180),
            270 => Ok(Self::Clockwise270),
            degrees => bail!(
                "unsupported video display rotation {degrees} degrees; expected 0, 90, 180, or 270"
            ),
        }
    }

    fn output_dimensions(self, width: u32, height: u32) -> (u32, u32) {
        match self {
            Self::None | Self::Clockwise180 => (width, height),
            Self::Clockwise90 | Self::Clockwise270 => (height, width),
        }
    }

    fn source_coordinate(
        self,
        target_x: u32,
        target_y: u32,
        width: u32,
        height: u32,
    ) -> (u32, u32) {
        match self {
            Self::None => (target_x, target_y),
            Self::Clockwise90 => (target_y, height - 1 - target_x),
            Self::Clockwise180 => (width - 1 - target_x, height - 1 - target_y),
            Self::Clockwise270 => (width - 1 - target_y, target_x),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaTime {
    pub value: i64,
    pub timescale: i32,
}

impl MediaTime {
    pub fn new(value: i64, timescale: i32) -> anyhow::Result<Self> {
        if timescale <= 0 {
            bail!("media timestamp timescale must be positive, got {timescale}");
        }

        Ok(Self { value, timescale })
    }

    pub fn as_seconds(self) -> f64 {
        self.value as f64 / self.timescale as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Bgra,
}

/// Pixel layout and timing without retaining the frame's image storage.
#[derive(Debug, Clone, Copy)]
pub struct VideoFrameInfo {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub time: MediaTime,
    #[cfg(target_os = "macos")]
    pub(crate) grayscale_mask: bool,
}

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub format: PixelFormat,
    pub time: MediaTime,
    pub data: FrameData,
    #[cfg(target_os = "macos")]
    pool: Option<FramePool>,
    #[cfg(target_os = "macos")]
    pub(crate) grayscale_mask: bool,
}

/// Reuse CPU staging allocations across the decoder, inference, and encoder
/// threads. The bounded cache avoids allocating and faulting multi-MB buffers
/// for every frame, while ownership of each live frame remains exclusive.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Default)]
pub(crate) struct FramePool(std::sync::Arc<std::sync::Mutex<Vec<Vec<u8>>>>);

#[cfg(target_os = "macos")]
impl FramePool {
    pub(crate) fn take(&self, len: usize) -> Vec<u8> {
        let mut data = self
            .0
            .lock()
            .ok()
            .and_then(|mut pool| pool.pop())
            .unwrap_or_default();
        data.resize(len, 0);
        data
    }
}

#[cfg(target_os = "macos")]
impl Drop for VideoFrame {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            if let Ok(mut buffers) = pool.0.lock() {
                if buffers.len() < 8 {
                    if let Some(data) = self.data.take_owned() {
                        buffers.push(data);
                    }
                }
            }
        }
    }
}

impl VideoFrame {
    pub(crate) fn info(&self) -> VideoFrameInfo {
        VideoFrameInfo {
            width: self.width,
            height: self.height,
            format: self.format,
            time: self.time,
            #[cfg(target_os = "macos")]
            grayscale_mask: self.grayscale_mask,
        }
    }

    pub fn new_bgra(
        width: u32,
        height: u32,
        bytes_per_row: u32,
        time: MediaTime,
        data: Vec<u8>,
    ) -> anyhow::Result<Self> {
        #[cfg(target_os = "macos")]
        let data = FrameData::from(data);
        Self::from_bgra_data(width, height, bytes_per_row, time, data)
    }

    pub(crate) fn from_bgra_data(
        width: u32,
        height: u32,
        bytes_per_row: u32,
        time: MediaTime,
        data: FrameData,
    ) -> anyhow::Result<Self> {
        validate_bgra_buffer(width, height, bytes_per_row, data.len())?;

        Ok(Self {
            width,
            height,
            bytes_per_row,
            format: PixelFormat::Bgra,
            time,
            data,
            #[cfg(target_os = "macos")]
            pool: None,
            #[cfg(target_os = "macos")]
            grayscale_mask: false,
        })
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn with_pool(mut self, pool: &FramePool) -> Self {
        if matches!(self.data, FrameData::Owned(_)) {
            self.pool = Some(pool.clone());
        }
        self
    }

    /// Only producers that write equal B/G/R bytes and opaque alpha may set this.
    #[cfg(target_os = "macos")]
    pub(crate) fn into_grayscale_mask(mut self) -> Self {
        self.grayscale_mask = true;
        self
    }

    pub fn checked_pixel_offset(&self, x: u32, y: u32) -> anyhow::Result<usize> {
        if x >= self.width || y >= self.height {
            bail!(
                "pixel coordinate ({x}, {y}) is outside frame bounds {}x{}",
                self.width,
                self.height
            );
        }

        let offset = y as usize * self.bytes_per_row as usize + x as usize * 4;
        let last = offset
            .checked_add(3)
            .context("frame pixel offset overflowed")?;
        if last >= self.data.len() {
            bail!(
                "frame data was too short for pixel coordinate ({x}, {y}); offset {offset}, length {}",
                self.data.len()
            );
        }

        Ok(offset)
    }

    pub fn rotated(self, rotation: DisplayRotation) -> anyhow::Result<Self> {
        if rotation == DisplayRotation::None {
            return Ok(self);
        }
        if self.format != PixelFormat::Bgra {
            bail!("video frame rotation is only implemented for BGRA frames");
        }

        let (width, height) = rotation.output_dimensions(self.width, self.height);
        let bytes_per_row = width
            .checked_mul(4)
            .context("rotated frame row width overflowed")?;
        let len = (bytes_per_row as usize)
            .checked_mul(height as usize)
            .context("rotated frame buffer length overflowed")?;
        let mut data = vec![0_u8; len];

        for target_y in 0..height {
            for target_x in 0..width {
                let (source_x, source_y) =
                    rotation.source_coordinate(target_x, target_y, self.width, self.height);
                let source_offset = self.checked_pixel_offset(source_x, source_y)?;
                let target_offset =
                    target_y as usize * bytes_per_row as usize + target_x as usize * 4;
                data[target_offset..target_offset + 4]
                    .copy_from_slice(&self.data[source_offset..source_offset + 4]);
            }
        }

        Self::new_bgra(width, height, bytes_per_row, self.time, data)
    }
}

pub fn validate_bgra_buffer(
    width: u32,
    height: u32,
    bytes_per_row: u32,
    len: usize,
) -> anyhow::Result<()> {
    if width == 0 || height == 0 {
        bail!("video frame dimensions must be non-zero, got {width}x{height}");
    }
    let min_row_bytes = width
        .checked_mul(4)
        .context("video frame row width overflowed")?;
    if bytes_per_row < min_row_bytes {
        bail!(
            "BGRA bytes_per_row must be at least width * 4; got {bytes_per_row}, need {min_row_bytes}"
        );
    }
    let min_len = bytes_per_row as usize * height as usize;
    if len < min_len {
        bail!("BGRA frame buffer is too short; got {len} bytes, need at least {min_len}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn frame_pool_reuses_only_released_buffers_and_handles_size_changes() {
        let pool = FramePool::default();
        let data = pool.take(16);
        let address = data.as_ptr();
        let frame = VideoFrame::new_bgra(2, 2, 8, MediaTime::new(0, 30).unwrap(), data)
            .unwrap()
            .with_pool(&pool);
        let other = pool.take(16);
        assert_ne!(other.as_ptr(), address);
        drop(frame);
        let reused = pool.take(16);
        assert_eq!(reused.as_ptr(), address);
        assert_eq!(pool.take(32).len(), 32);
    }

    fn test_frame() -> VideoFrame {
        let pixels = [
            [1, 0, 0, 255],
            [2, 0, 0, 255],
            [3, 0, 0, 255],
            [4, 0, 0, 255],
            [5, 0, 0, 255],
            [6, 0, 0, 255],
        ];
        VideoFrame::new_bgra(
            3,
            2,
            12,
            MediaTime::new(0, 1).unwrap(),
            pixels.into_iter().flatten().collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn first_channel(frame: &VideoFrame) -> Vec<u8> {
        let mut values = Vec::new();
        for y in 0..frame.height {
            for x in 0..frame.width {
                values.push(frame.data[frame.checked_pixel_offset(x, y).unwrap()]);
            }
        }
        values
    }

    #[test]
    fn rotates_bgra_clockwise_90() {
        let frame = test_frame().rotated(DisplayRotation::Clockwise90).unwrap();

        assert_eq!((frame.width, frame.height), (2, 3));
        assert_eq!(first_channel(&frame), vec![4, 1, 5, 2, 6, 3]);
    }

    #[test]
    fn rotates_bgra_clockwise_180() {
        let frame = test_frame().rotated(DisplayRotation::Clockwise180).unwrap();

        assert_eq!((frame.width, frame.height), (3, 2));
        assert_eq!(first_channel(&frame), vec![6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn rotates_bgra_clockwise_270() {
        let frame = test_frame().rotated(DisplayRotation::Clockwise270).unwrap();

        assert_eq!((frame.width, frame.height), (2, 3));
        assert_eq!(first_channel(&frame), vec![3, 6, 2, 5, 1, 4]);
    }
}
