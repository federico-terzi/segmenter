use anyhow::{bail, Context};

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

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub format: PixelFormat,
    pub time: MediaTime,
    pub data: Vec<u8>,
}

impl VideoFrame {
    pub fn new_bgra(
        width: u32,
        height: u32,
        bytes_per_row: u32,
        time: MediaTime,
        data: Vec<u8>,
    ) -> anyhow::Result<Self> {
        validate_bgra_buffer(width, height, bytes_per_row, data.len())?;

        Ok(Self {
            width,
            height,
            bytes_per_row,
            format: PixelFormat::Bgra,
            time,
            data,
        })
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
