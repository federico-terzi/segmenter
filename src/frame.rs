use anyhow::{bail, Context};

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
            pixels.into_iter().flatten().collect(),
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
