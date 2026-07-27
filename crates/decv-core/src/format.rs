use crate::{ColorInfo, MediaError, Result};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    #[inline]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    #[inline]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    #[inline]
    pub const fn size(self) -> Size {
        Size::new(self.width, self.height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelFormat {
    Nv12,
    Bgra8,
    Rgba8,
    I420,
    I422,
    I440,
    I444,
    P010,
}

impl PixelFormat {
    #[inline]
    pub const fn plane_count(self) -> usize {
        match self {
            Self::Nv12 | Self::P010 => 2,
            Self::Bgra8 | Self::Rgba8 => 1,
            Self::I420 | Self::I422 | Self::I440 | Self::I444 => 3,
        }
    }

    #[inline]
    pub const fn is_chroma_subsampled_420(self) -> bool {
        matches!(self, Self::Nv12 | Self::I420 | Self::P010)
    }

    #[inline]
    pub const fn chroma_subsampling(self) -> Option<(u8, u8)> {
        match self {
            Self::Nv12 | Self::I420 | Self::P010 => Some((1, 1)),
            Self::I422 => Some((1, 0)),
            Self::I440 => Some((0, 1)),
            Self::I444 => Some((0, 0)),
            Self::Bgra8 | Self::Rgba8 => None,
        }
    }
}

/// Image layout and presentation metadata shared by a stream and its frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct VideoFormat {
    pub coded_size: Size,
    pub visible_rect: Rect,
    pub display_size: Size,
    pub pixel_format: PixelFormat,
    pub color: ColorInfo,
}

impl VideoFormat {
    #[inline]
    pub const fn new(
        coded_size: Size,
        visible_rect: Rect,
        display_size: Size,
        pixel_format: PixelFormat,
        color: ColorInfo,
    ) -> Self {
        Self {
            coded_size,
            visible_rect,
            display_size,
            pixel_format,
            color,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.coded_size.is_empty() {
            return Err(MediaError::InvalidVideoFormat(
                "coded size must be non-zero",
            ));
        }
        if self.visible_rect.size().is_empty() {
            return Err(MediaError::InvalidVideoFormat(
                "visible rectangle must be non-zero",
            ));
        }
        if self.display_size.is_empty() {
            return Err(MediaError::InvalidVideoFormat(
                "display size must be non-zero",
            ));
        }

        let visible_right = self
            .visible_rect
            .x
            .checked_add(self.visible_rect.width)
            .ok_or(MediaError::IntegerOverflow)?;
        let visible_bottom = self
            .visible_rect
            .y
            .checked_add(self.visible_rect.height)
            .ok_or(MediaError::IntegerOverflow)?;

        if visible_right > self.coded_size.width || visible_bottom > self.coded_size.height {
            return Err(MediaError::InvalidVideoFormat(
                "visible rectangle exceeds coded size",
            ));
        }

        if let Some((subsampling_x, subsampling_y)) = self.pixel_format.chroma_subsampling() {
            if subsampling_x != 0 && !self.coded_size.width.is_multiple_of(2) {
                return Err(MediaError::InvalidVideoFormat(
                    "horizontally subsampled coded width must be even",
                ));
            }
            if subsampling_y != 0 && !self.coded_size.height.is_multiple_of(2) {
                return Err(MediaError::InvalidVideoFormat(
                    "vertically subsampled coded height must be even",
                ));
            }
        }

        Ok(())
    }
}
