use std::{
    fmt,
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use aligned_vec::AVec;

use crate::{MediaError, MediaTime, PixelFormat, Result, Size, VideoFormat};

pub const CPU_BUFFER_ALIGNMENT: usize = 64;

/// Cache-line-aligned byte storage used by software codecs.
///
/// Keeping this type in `decv-core` lets codecs share their native aligned
/// picture allocation with an output frame instead of copying into `Arc<[u8]>`.
#[derive(Clone, PartialEq, Eq)]
pub struct AlignedBytes {
    inner: AVec<u8>,
}

impl AlignedBytes {
    #[inline]
    pub fn zeroed(len: usize) -> Self {
        let mut inner = AVec::with_capacity(CPU_BUFFER_ALIGNMENT, len);
        inner.resize(len, 0);
        Self { inner }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: AVec::with_capacity(CPU_BUFFER_ALIGNMENT, capacity),
        }
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.inner.alignment()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[inline]
    pub fn resize(&mut self, len: usize, value: u8) {
        self.inner.resize(len, value);
    }

    /// Changes the initialized byte length.
    ///
    /// # Safety
    ///
    /// The newly exposed range must be completely initialized before it is
    /// read or before this value is dropped.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        // SAFETY: Forwarded to AVec with the caller's initialization contract.
        unsafe { self.inner.set_len(len) };
    }
}

impl fmt::Debug for AlignedBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlignedBytes")
            .field("len", &self.len())
            .field("capacity", &self.capacity())
            .field("alignment", &self.alignment())
            .finish()
    }
}

impl Deref for AlignedBytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner.as_slice()
    }
}

impl DerefMut for AlignedBytes {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut_slice()
    }
}

impl AsRef<[u8]> for AlignedBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl AsMut<[u8]> for AlignedBytes {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self
    }
}

/// Immutable shared backing allocation for a CPU image plane.
#[derive(Debug, Clone)]
pub enum CpuBuffer {
    Bytes(Arc<[u8]>),
    /// Native-endian 16-bit samples exposed as their byte representation.
    Words(Arc<[u16]>),
    Aligned(Arc<AlignedBytes>),
}

impl CpuBuffer {
    #[inline]
    pub fn ptr_eq(left: &Self, right: &Self) -> bool {
        match (left, right) {
            (Self::Bytes(left), Self::Bytes(right)) => Arc::ptr_eq(left, right),
            (Self::Words(left), Self::Words(right)) => Arc::ptr_eq(left, right),
            (Self::Aligned(left), Self::Aligned(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }
}

impl Deref for CpuBuffer {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Bytes(bytes) => bytes,
            Self::Words(words) => {
                // SAFETY: `u16` has no invalid bit patterns, the resulting byte
                // slice covers exactly the same allocation, and its lifetime is
                // bounded by the borrowed `Arc<[u16]>`.
                unsafe {
                    std::slice::from_raw_parts(
                        words.as_ptr().cast::<u8>(),
                        words.len() * size_of::<u16>(),
                    )
                }
            }
            Self::Aligned(bytes) => bytes,
        }
    }
}

impl AsRef<[u8]> for CpuBuffer {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl From<Arc<[u8]>> for CpuBuffer {
    #[inline]
    fn from(bytes: Arc<[u8]>) -> Self {
        Self::Bytes(bytes)
    }
}

impl From<Vec<u8>> for CpuBuffer {
    #[inline]
    fn from(bytes: Vec<u8>) -> Self {
        Self::Bytes(bytes.into())
    }
}

impl From<Arc<[u16]>> for CpuBuffer {
    #[inline]
    fn from(words: Arc<[u16]>) -> Self {
        Self::Words(words)
    }
}

impl From<Arc<AlignedBytes>> for CpuBuffer {
    #[inline]
    fn from(bytes: Arc<AlignedBytes>) -> Self {
        Self::Aligned(bytes)
    }
}

impl From<AlignedBytes> for CpuBuffer {
    #[inline]
    fn from(bytes: AlignedBytes) -> Self {
        Self::Aligned(Arc::new(bytes))
    }
}

/// One immutable CPU-backed image plane.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CpuPlane {
    pub bytes: CpuBuffer,
    pub offset: usize,
    pub stride: usize,
    pub rows: usize,
}

impl CpuPlane {
    #[inline]
    pub fn new(bytes: impl Into<CpuBuffer>, offset: usize, stride: usize, rows: usize) -> Self {
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
        PixelFormat::I420 | PixelFormat::I422 | PixelFormat::I440 | PixelFormat::I444 => {
            validate_plane(&frame.planes[0], size.width, size.height)?;
            let (subsampling_x, subsampling_y) = format
                .pixel_format
                .chroma_subsampling()
                .expect("planar YUV format has chroma subsampling");
            let chroma = Size::new(size.width >> subsampling_x, size.height >> subsampling_y);
            validate_plane(&frame.planes[1], chroma.width, chroma.height)?;
            validate_plane(&frame.planes[2], chroma.width, chroma.height)?;
        }
        PixelFormat::PlanarYuv16 {
            subsampling_x,
            subsampling_y,
            ..
        } => {
            let luma_row_bytes = size
                .width
                .checked_mul(2)
                .ok_or(MediaError::IntegerOverflow)?;
            validate_plane(&frame.planes[0], luma_row_bytes, size.height)?;
            let chroma_width = (size.width >> subsampling_x)
                .checked_mul(2)
                .ok_or(MediaError::IntegerOverflow)?;
            let chroma_height = size.height >> subsampling_y;
            validate_plane(&frame.planes[1], chroma_width, chroma_height)?;
            validate_plane(&frame.planes[2], chroma_width, chroma_height)?;
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

    use super::{
        AlignedBytes, CPU_BUFFER_ALIGNMENT, CpuBuffer, CpuFrame, CpuPlane, DecodedVideoFrame,
        FrameStorage,
    };
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
    fn aligned_buffers_preserve_alignment_and_shared_identity() {
        let bytes = Arc::new(AlignedBytes::zeroed(257));
        assert_eq!(bytes.as_ptr().addr() % CPU_BUFFER_ALIGNMENT, 0);
        assert!(bytes.iter().all(|&byte| byte == 0));

        let first = CpuBuffer::from(bytes.clone());
        let second = CpuBuffer::from(bytes);
        assert!(CpuBuffer::ptr_eq(&first, &second));
    }

    #[test]
    fn word_buffers_expose_native_endian_bytes_without_copying() {
        let words: Arc<[u16]> = vec![0x1234, 0xabcd].into();
        let first = CpuBuffer::from(words.clone());
        let second = CpuBuffer::from(words);
        let expected = [0x1234u16.to_ne_bytes(), 0xabcdu16.to_ne_bytes()].concat();

        assert_eq!(first.as_ref(), expected);
        assert!(CpuBuffer::ptr_eq(&first, &second));
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
                        bytes: allocation.clone().into(),
                        offset: 0,
                        stride: 8,
                        rows: 4,
                    },
                    CpuPlane {
                        bytes: allocation.into(),
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
                        bytes: allocation.clone().into(),
                        offset: 0,
                        stride: 8,
                        rows: 4,
                    },
                    CpuPlane {
                        bytes: allocation.into(),
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

    #[test]
    fn validates_planar_444_storage() {
        let format = VideoFormat {
            coded_size: Size::new(4, 2),
            visible_rect: Rect::new(0, 0, 4, 2),
            display_size: Size::new(4, 2),
            pixel_format: PixelFormat::I444,
            color: ColorInfo::default(),
        };
        let frame = DecodedVideoFrame {
            id: 2,
            pts: None,
            duration: None,
            format,
            storage: FrameStorage::Cpu(CpuFrame::new(
                (0..3).map(|_| CpuPlane::new(vec![0; 8], 0, 4, 2)).collect(),
            )),
        };
        assert_eq!(frame.validate(), Ok(()));
    }

    #[test]
    fn validates_high_bit_depth_planar_storage() {
        let format = VideoFormat {
            coded_size: Size::new(4, 2),
            visible_rect: Rect::new(0, 0, 4, 2),
            display_size: Size::new(4, 2),
            pixel_format: PixelFormat::PlanarYuv16 {
                bit_depth: 10,
                subsampling_x: 1,
                subsampling_y: 0,
            },
            color: ColorInfo::default(),
        };
        let frame = DecodedVideoFrame {
            id: 3,
            pts: None,
            duration: None,
            format,
            storage: FrameStorage::Cpu(CpuFrame::new(vec![
                CpuPlane::new(Arc::<[u16]>::from(vec![0; 8]), 0, 8, 2),
                CpuPlane::new(Arc::<[u16]>::from(vec![0; 4]), 0, 4, 2),
                CpuPlane::new(Arc::<[u16]>::from(vec![0; 4]), 0, 4, 2),
            ])),
        };
        assert_eq!(frame.validate(), Ok(()));
    }
}
