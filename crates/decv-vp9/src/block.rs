#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum BlockSize {
    B4x4,
    B4x8,
    B8x4,
    B8x8,
    B8x16,
    B16x8,
    B16x16,
    B16x32,
    B32x16,
    B32x32,
    B32x64,
    B64x32,
    B64x64,
}

impl BlockSize {
    pub(crate) const fn width_4x4(self) -> usize {
        [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16][self as usize]
    }

    pub(crate) const fn height_4x4(self) -> usize {
        [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16][self as usize]
    }

    pub(crate) const fn width_mi(self) -> usize {
        self.width_4x4().div_ceil(2)
    }

    pub(crate) const fn height_mi(self) -> usize {
        self.height_4x4().div_ceil(2)
    }

    pub(crate) const fn maximum_transform(self) -> TransformSize {
        match self {
            Self::B4x4 | Self::B4x8 | Self::B8x4 => TransformSize::Tx4x4,
            Self::B8x8 | Self::B8x16 | Self::B16x8 => TransformSize::Tx8x8,
            Self::B16x16 | Self::B16x32 | Self::B32x16 => TransformSize::Tx16x16,
            _ => TransformSize::Tx32x32,
        }
    }

    pub(crate) const fn partition_subsize(self, partition: Partition) -> Option<Self> {
        match (self, partition) {
            (size, Partition::None) => Some(size),
            (Self::B8x8, Partition::Horizontal) => Some(Self::B8x4),
            (Self::B16x16, Partition::Horizontal) => Some(Self::B16x8),
            (Self::B32x32, Partition::Horizontal) => Some(Self::B32x16),
            (Self::B64x64, Partition::Horizontal) => Some(Self::B64x32),
            (Self::B8x8, Partition::Vertical) => Some(Self::B4x8),
            (Self::B16x16, Partition::Vertical) => Some(Self::B8x16),
            (Self::B32x32, Partition::Vertical) => Some(Self::B16x32),
            (Self::B64x64, Partition::Vertical) => Some(Self::B32x64),
            (Self::B8x8, Partition::Split) => Some(Self::B4x4),
            (Self::B16x16, Partition::Split) => Some(Self::B8x8),
            (Self::B32x32, Partition::Split) => Some(Self::B16x16),
            (Self::B64x64, Partition::Split) => Some(Self::B32x32),
            _ => None,
        }
    }

    pub(crate) const fn partition_context(self) -> (u8, u8) {
        [
            (15, 15),
            (15, 14),
            (14, 15),
            (14, 14),
            (14, 12),
            (12, 14),
            (12, 12),
            (12, 8),
            (8, 12),
            (8, 8),
            (8, 0),
            (0, 8),
            (0, 0),
        ][self as usize]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Partition {
    None,
    Horizontal,
    Vertical,
    Split,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum TransformSize {
    Tx4x4,
    Tx8x8,
    Tx16x16,
    Tx32x32,
}

impl TransformSize {
    pub(crate) const fn width_4x4(self) -> usize {
        1 << self as usize
    }

    pub(crate) const fn coefficient_count(self) -> usize {
        16 << ((self as usize) * 2)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum IntraMode {
    Dc,
    Vertical,
    Horizontal,
    D45,
    D135,
    D117,
    D153,
    D207,
    D63,
    TrueMotion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransformType {
    DctDct,
    AdstDct,
    DctAdst,
    AdstAdst,
}

impl IntraMode {
    pub(crate) const fn transform_type(self) -> TransformType {
        match self {
            Self::Dc | Self::D45 => TransformType::DctDct,
            Self::Vertical | Self::D117 | Self::D63 => TransformType::AdstDct,
            Self::Horizontal | Self::D153 | Self::D207 => TransformType::DctAdst,
            Self::D135 | Self::TrueMotion => TransformType::AdstAdst,
        }
    }
}
