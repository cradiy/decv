use std::sync::Arc;

use crate::{MediaError, MediaTime, PixelFormat, Result, Size, VideoFormat};

/// One immutable CPU-backed image plane.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CpuPlane {
    pub bytes: Arc<[u8]>,
    pub offset: usize,
    pub stride: usize,
    pub rows: usize,
}

impl CpuPlane {
    #[inline]
    pub fn new(bytes: impl Into<Arc<[u8]>>, offset: usize, stride: usize, rows: usize) -> Self {
        Self {
            bytes: bytes.into(),
            offset,
            stride,
            rows,
        }
    }
}

/// CPU-backed storage for all planes of one decoded frame.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CpuFrame {
    pub planes: Vec<CpuPlane>,
}

impl CpuFrame {
    #[inline]
    pub fn new(planes: Vec<CpuPlane>) -> Self {
        Self { planes }
    }
}

/// Ownership-preserving storage for a decoded frame.
///
/// Platform-native variants can be added without changing the frame metadata
/// contract. The backing allocation must remain immutable while any clone of
/// the frame is alive.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FrameStorage {
    Cpu(CpuFrame),
}

/// A decoded frame with presentation metadata and owned immutable storage.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedVideoFrame {
    pub id: u64,
    pub pts: Option<MediaTime>,
    pub duration: Option<MediaTime>,
    pub format: VideoFormat,
    pub storage: FrameStorage,
}

impl DecodedVideoFrame {
    #[inline]
    pub const fn new(
        id: u64,
        pts: Option<MediaTime>,
        duration: Option<MediaTime>,
        format: VideoFormat,
        storage: FrameStorage,
    ) -> Self {
        Self {
            id,
            pts,
            duration,
            format,
            storage,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.format.validate()?;
        if self.duration.is_some_and(|duration| duration.value < 0) {
            return Err(MediaError::InvalidFrameStorage(
                "frame duration must not be negative",
            ));
        }

        match &self.storage {
            FrameStorage::Cpu(frame) => validate_cpu_frame(frame, self.format),
        }
    }
}

fn validate_cpu_frame(frame: &CpuFrame, format: VideoFormat) -> Result<()> {
    if frame.planes.len() != format.pixel_format.plane_count() {
        return Err(MediaError::InvalidFrameStorage(
            "plane count does not match pixel format",
        ));
    }

    let size = format.coded_size;
    match format.pixel_format {
        PixelFormat::Nv12 => {
            validate_plane(&frame.planes[0], size.width, size.height)?;
            validate_plane(&frame.planes[1], size.width, size.height / 2)?;
        }
        PixelFormat::Bgra8 | PixelFormat::Rgba8 => {
            let row_bytes = size
                .width
                .checked_mul(4)
                .ok_or(MediaError::IntegerOverflow)?;
            validate_plane(&frame.planes[0], row_bytes, size.height)?;
        }
        PixelFormat::I420 => {
            validate_plane(&frame.planes[0], size.width, size.height)?;
            let chroma = Size::new(size.width / 2, size.height / 2);
            validate_plane(&frame.planes[1], chroma.width, chroma.height)?;
            validate_plane(&frame.planes[2], chroma.width, chroma.height)?;
        }
        PixelFormat::P010 => {
            let row_bytes = size
                .width
                .checked_mul(2)
                .ok_or(MediaError::IntegerOverflow)?;
            validate_plane(&frame.planes[0], row_bytes, size.height)?;
            validate_plane(&frame.planes[1], row_bytes, size.height / 2)?;
        }
    }

    Ok(())
}

fn validate_plane(plane: &CpuPlane, row_bytes: u32, rows: u32) -> Result<()> {
    let row_bytes = usize::try_from(row_bytes).map_err(|_| MediaError::IntegerOverflow)?;
    let rows = usize::try_from(rows).map_err(|_| MediaError::IntegerOverflow)?;

    if plane.rows != rows {
        return Err(MediaError::InvalidFrameStorage(
            "plane row count does not match pixel format",
        ));
    }
    if plane.stride < row_bytes {
        return Err(MediaError::InvalidFrameStorage(
            "plane stride is smaller than its visible row",
        ));
    }

    let body = plane
        .stride
        .checked_mul(rows.saturating_sub(1))
        .ok_or(MediaError::IntegerOverflow)?;
    let required_len = plane
        .offset
        .checked_add(body)
        .and_then(|value| value.checked_add(row_bytes))
        .ok_or(MediaError::IntegerOverflow)?;

    if required_len > plane.bytes.len() {
        return Err(MediaError::InvalidFrameStorage(
            "plane layout exceeds its backing allocation",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{CpuFrame, CpuPlane, DecodedVideoFrame, FrameStorage};
    use crate::{ColorInfo, MediaError, PixelFormat, Rect, Size, VideoFormat};

    fn nv12_format() -> VideoFormat {
        VideoFormat {
            coded_size: Size::new(4, 4),
            visible_rect: Rect::new(0, 0, 4, 4),
            display_size: Size::new(4, 4),
            pixel_format: PixelFormat::Nv12,
            color: ColorInfo::default(),
        }
    }

    #[test]
    fn validates_strided_nv12_storage() {
        let allocation: Arc<[u8]> = vec![0; 48].into();
        let frame = DecodedVideoFrame {
            id: 1,
            pts: None,
            duration: None,
            format: nv12_format(),
            storage: FrameStorage::Cpu(CpuFrame {
                planes: vec![
                    CpuPlane {
                        bytes: allocation.clone(),
                        offset: 0,
                        stride: 8,
                        rows: 4,
                    },
                    CpuPlane {
                        bytes: allocation,
                        offset: 32,
                        stride: 8,
                        rows: 2,
                    },
                ],
            }),
        };

        assert_eq!(frame.validate(), Ok(()));
    }

    #[test]
    fn rejects_a_plane_that_exceeds_its_allocation() {
        let allocation: Arc<[u8]> = vec![0; 39].into();
        let frame = DecodedVideoFrame {
            id: 1,
            pts: None,
            duration: None,
            format: nv12_format(),
            storage: FrameStorage::Cpu(CpuFrame {
                planes: vec![
                    CpuPlane {
                        bytes: allocation.clone(),
                        offset: 0,
                        stride: 8,
                        rows: 4,
                    },
                    CpuPlane {
                        bytes: allocation,
                        offset: 32,
                        stride: 8,
                        rows: 2,
                    },
                ],
            }),
        };

        assert_eq!(
            frame.validate(),
            Err(MediaError::InvalidFrameStorage(
                "plane layout exceeds its backing allocation"
            ))
        );
    }
}
