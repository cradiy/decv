//! CABAC residual-block categories and coded-block neighbour state.

use crate::{H264Error, Result};

const LUMA_BLOCK_COORDINATES: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
];
const CHROMA_BLOCK_COORDINATES: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

/// The six residual block categories used by progressive 8-bit 4:2:0 CABAC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CabacResidualCategory {
    LumaDc,
    LumaAc,
    Luma4x4,
    ChromaDc,
    ChromaAc,
    Luma8x8,
}

impl CabacResidualCategory {
    #[inline]
    pub const fn maximum_coefficients(self) -> u8 {
        match self {
            Self::LumaDc | Self::Luma4x4 => 16,
            Self::LumaAc | Self::ChromaAc => 15,
            Self::ChromaDc => 4,
            Self::Luma8x8 => 64,
        }
    }

    #[inline]
    pub const fn coded_block_flag_context_base(self) -> Option<usize> {
        match self {
            Self::LumaDc => Some(85),
            Self::LumaAc => Some(89),
            Self::Luma4x4 => Some(93),
            Self::ChromaDc => Some(97),
            Self::ChromaAc => Some(101),
            // For 4:2:0, coded_block_pattern already implies an 8x8 luma
            // block is coded, so no coded_block_flag bin is present.
            Self::Luma8x8 => None,
        }
    }

    #[inline]
    pub const fn significant_coeff_context_base(self) -> usize {
        match self {
            Self::LumaDc => 105,
            Self::LumaAc => 120,
            Self::Luma4x4 => 134,
            Self::ChromaDc => 149,
            Self::ChromaAc => 152,
            Self::Luma8x8 => 402,
        }
    }

    #[inline]
    pub const fn last_significant_coeff_context_base(self) -> usize {
        match self {
            Self::LumaDc => 166,
            Self::LumaAc => 181,
            Self::Luma4x4 => 195,
            Self::ChromaDc => 210,
            Self::ChromaAc => 213,
            Self::Luma8x8 => 417,
        }
    }

    #[inline]
    pub const fn coefficient_level_context_base(self) -> usize {
        match self {
            Self::LumaDc => 227,
            Self::LumaAc => 237,
            Self::Luma4x4 => 247,
            Self::ChromaDc => 257,
            Self::ChromaAc => 266,
            Self::Luma8x8 => 426,
        }
    }
}

/// Identifies one transform block within a macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CabacResidualBlock {
    LumaDc,
    LumaAc(u8),
    Luma4x4(u8),
    ChromaDc { plane: u8 },
    ChromaAc { plane: u8, block: u8 },
    Luma8x8(u8),
}

impl CabacResidualBlock {
    #[inline]
    pub const fn category(self) -> CabacResidualCategory {
        match self {
            Self::LumaDc => CabacResidualCategory::LumaDc,
            Self::LumaAc(_) => CabacResidualCategory::LumaAc,
            Self::Luma4x4(_) => CabacResidualCategory::Luma4x4,
            Self::ChromaDc { .. } => CabacResidualCategory::ChromaDc,
            Self::ChromaAc { .. } => CabacResidualCategory::ChromaAc,
            Self::Luma8x8(_) => CabacResidualCategory::Luma8x8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockState {
    slice_id: u32,
    coded: bool,
}

#[derive(Debug, Clone)]
struct BlockGrid {
    width: usize,
    height: usize,
    entries: Vec<Option<BlockState>>,
}

impl BlockGrid {
    fn new(width: usize, height: usize) -> Result<Self> {
        let length = width
            .checked_mul(height)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width,
            height,
            entries: vec![None; length],
        })
    }

    fn neighbour_coded(
        &self,
        x: Option<usize>,
        y: Option<usize>,
        slice_id: u32,
        unavailable_value: bool,
    ) -> bool {
        let (Some(x), Some(y)) = (x, y) else {
            return unavailable_value;
        };
        if x >= self.width || y >= self.height {
            return unavailable_value;
        }
        self.entries[y * self.width + x]
            .filter(|state| state.slice_id == slice_id)
            .map_or(unavailable_value, |state| state.coded)
    }

    fn record(&mut self, x: usize, y: usize, slice_id: u32, coded: bool) -> Result<()> {
        if x >= self.width || y >= self.height {
            return Err(H264Error::InvalidSyntax(
                "CABAC residual block lies outside the picture",
            ));
        }
        self.entries[y * self.width + x] = Some(BlockState { slice_id, coded });
        Ok(())
    }
}

/// Per-picture coded-block state used by CABAC residual context derivation.
#[derive(Debug, Clone)]
pub struct CabacResidualState {
    width_in_macroblocks: usize,
    height_in_macroblocks: usize,
    luma: BlockGrid,
    chroma_cb: BlockGrid,
    chroma_cr: BlockGrid,
    luma_dc: BlockGrid,
    chroma_dc_cb: BlockGrid,
    chroma_dc_cr: BlockGrid,
}

impl CabacResidualState {
    pub fn new(width_in_macroblocks: usize, height_in_macroblocks: usize) -> Result<Self> {
        if width_in_macroblocks == 0 || height_in_macroblocks == 0 {
            return Err(H264Error::InvalidSyntax(
                "CABAC residual-state dimensions must be non-zero",
            ));
        }
        let luma_width = width_in_macroblocks
            .checked_mul(4)
            .ok_or(H264Error::IntegerOverflow)?;
        let luma_height = height_in_macroblocks
            .checked_mul(4)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma_width = width_in_macroblocks
            .checked_mul(2)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma_height = height_in_macroblocks
            .checked_mul(2)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width_in_macroblocks,
            height_in_macroblocks,
            luma: BlockGrid::new(luma_width, luma_height)?,
            chroma_cb: BlockGrid::new(chroma_width, chroma_height)?,
            chroma_cr: BlockGrid::new(chroma_width, chroma_height)?,
            luma_dc: BlockGrid::new(width_in_macroblocks, height_in_macroblocks)?,
            chroma_dc_cb: BlockGrid::new(width_in_macroblocks, height_in_macroblocks)?,
            chroma_dc_cr: BlockGrid::new(width_in_macroblocks, height_in_macroblocks)?,
        })
    }

    /// Returns the context index for `coded_block_flag`, or `None` when the
    /// category has no coded-block flag in a 4:2:0 stream.
    pub fn coded_block_flag_context_index(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        current_is_intra: bool,
        block: CabacResidualBlock,
    ) -> Result<Option<usize>> {
        let Some(base) = block.category().coded_block_flag_context_base() else {
            self.validate_block(macroblock_address, block)?;
            return Ok(None);
        };
        let (grid, x, y) = self.grid_and_coordinates(macroblock_address, block)?;
        let left = grid.neighbour_coded(x.checked_sub(1), Some(y), slice_id, current_is_intra);
        let top = grid.neighbour_coded(Some(x), y.checked_sub(1), slice_id, current_is_intra);
        Ok(Some(base + usize::from(left) + 2 * usize::from(top)))
    }

    pub fn record_block(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        block: CabacResidualBlock,
        coded: bool,
    ) -> Result<()> {
        self.validate_block(macroblock_address, block)?;
        if let CabacResidualBlock::Luma8x8(index) = block {
            let macroblock_x = macroblock_address % self.width_in_macroblocks;
            let macroblock_y = macroblock_address / self.width_in_macroblocks;
            let base_x = macroblock_x * 4 + usize::from(index % 2) * 2;
            let base_y = macroblock_y * 4 + usize::from(index / 2) * 2;
            for y in base_y..base_y + 2 {
                for x in base_x..base_x + 2 {
                    self.luma.record(x, y, slice_id, coded)?;
                }
            }
            return Ok(());
        }
        let (grid, x, y) = self.grid_and_coordinates_mut(macroblock_address, block)?;
        grid.record(x, y, slice_id, coded)
    }

    pub fn record_pcm_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<()> {
        self.validate_macroblock(macroblock_address)?;
        self.record_block(
            macroblock_address,
            slice_id,
            CabacResidualBlock::LumaDc,
            true,
        )?;
        for block in 0..16 {
            self.record_block(
                macroblock_address,
                slice_id,
                CabacResidualBlock::Luma4x4(block),
                true,
            )?;
        }
        for plane in 0..2 {
            self.record_block(
                macroblock_address,
                slice_id,
                CabacResidualBlock::ChromaDc { plane },
                true,
            )?;
            for block in 0..4 {
                self.record_block(
                    macroblock_address,
                    slice_id,
                    CabacResidualBlock::ChromaAc { plane, block },
                    true,
                )?;
            }
        }
        Ok(())
    }

    fn grid_and_coordinates(
        &self,
        macroblock_address: usize,
        block: CabacResidualBlock,
    ) -> Result<(&BlockGrid, usize, usize)> {
        self.validate_block(macroblock_address, block)?;
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        match block {
            CabacResidualBlock::LumaDc => Ok((&self.luma_dc, macroblock_x, macroblock_y)),
            CabacResidualBlock::LumaAc(index) | CabacResidualBlock::Luma4x4(index) => {
                let (x, y) = LUMA_BLOCK_COORDINATES[usize::from(index)];
                Ok((&self.luma, macroblock_x * 4 + x, macroblock_y * 4 + y))
            }
            CabacResidualBlock::ChromaDc { plane: 0 } => {
                Ok((&self.chroma_dc_cb, macroblock_x, macroblock_y))
            }
            CabacResidualBlock::ChromaDc { plane: 1 } => {
                Ok((&self.chroma_dc_cr, macroblock_x, macroblock_y))
            }
            CabacResidualBlock::ChromaAc { plane, block } => {
                let (x, y) = CHROMA_BLOCK_COORDINATES[usize::from(block)];
                let grid = if plane == 0 {
                    &self.chroma_cb
                } else {
                    &self.chroma_cr
                };
                Ok((grid, macroblock_x * 2 + x, macroblock_y * 2 + y))
            }
            CabacResidualBlock::Luma8x8(index) => {
                let x = macroblock_x * 4 + usize::from(index % 2) * 2;
                let y = macroblock_y * 4 + usize::from(index / 2) * 2;
                Ok((&self.luma, x, y))
            }
            CabacResidualBlock::ChromaDc { .. } => unreachable!("plane validated below"),
        }
    }

    fn grid_and_coordinates_mut(
        &mut self,
        macroblock_address: usize,
        block: CabacResidualBlock,
    ) -> Result<(&mut BlockGrid, usize, usize)> {
        self.validate_block(macroblock_address, block)?;
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        match block {
            CabacResidualBlock::LumaDc => Ok((&mut self.luma_dc, macroblock_x, macroblock_y)),
            CabacResidualBlock::LumaAc(index) | CabacResidualBlock::Luma4x4(index) => {
                let (x, y) = LUMA_BLOCK_COORDINATES[usize::from(index)];
                Ok((&mut self.luma, macroblock_x * 4 + x, macroblock_y * 4 + y))
            }
            CabacResidualBlock::ChromaDc { plane: 0 } => {
                Ok((&mut self.chroma_dc_cb, macroblock_x, macroblock_y))
            }
            CabacResidualBlock::ChromaDc { plane: 1 } => {
                Ok((&mut self.chroma_dc_cr, macroblock_x, macroblock_y))
            }
            CabacResidualBlock::ChromaAc { plane, block } => {
                let (x, y) = CHROMA_BLOCK_COORDINATES[usize::from(block)];
                let grid = if plane == 0 {
                    &mut self.chroma_cb
                } else {
                    &mut self.chroma_cr
                };
                Ok((grid, macroblock_x * 2 + x, macroblock_y * 2 + y))
            }
            CabacResidualBlock::Luma8x8(index) => {
                let x = macroblock_x * 4 + usize::from(index % 2) * 2;
                let y = macroblock_y * 4 + usize::from(index / 2) * 2;
                Ok((&mut self.luma, x, y))
            }
            CabacResidualBlock::ChromaDc { .. } => unreachable!("plane validated below"),
        }
    }

    fn validate_block(&self, macroblock_address: usize, block: CabacResidualBlock) -> Result<()> {
        self.validate_macroblock(macroblock_address)?;
        match block {
            CabacResidualBlock::LumaAc(index) | CabacResidualBlock::Luma4x4(index)
                if index >= 16 =>
            {
                Err(H264Error::InvalidSyntax(
                    "CABAC luma residual block index exceeds 15",
                ))
            }
            CabacResidualBlock::ChromaDc { plane } if plane >= 2 => Err(H264Error::InvalidSyntax(
                "CABAC chroma residual plane exceeds 1",
            )),
            CabacResidualBlock::ChromaAc { plane, block } if plane >= 2 || block >= 4 => Err(
                H264Error::InvalidSyntax("CABAC chroma AC block identifier is out of range"),
            ),
            CabacResidualBlock::Luma8x8(index) if index >= 4 => Err(H264Error::InvalidSyntax(
                "CABAC luma 8x8 block index exceeds 3",
            )),
            _ => Ok(()),
        }
    }

    fn validate_macroblock(&self, macroblock_address: usize) -> Result<()> {
        if macroblock_address >= self.width_in_macroblocks * self.height_in_macroblocks {
            return Err(H264Error::InvalidSyntax(
                "CABAC residual macroblock address exceeds the picture",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_normative_context_offsets_and_block_sizes() {
        let expected = [
            (CabacResidualCategory::LumaDc, Some(85), 105, 166, 227, 16),
            (CabacResidualCategory::LumaAc, Some(89), 120, 181, 237, 15),
            (CabacResidualCategory::Luma4x4, Some(93), 134, 195, 247, 16),
            (CabacResidualCategory::ChromaDc, Some(97), 149, 210, 257, 4),
            (
                CabacResidualCategory::ChromaAc,
                Some(101),
                152,
                213,
                266,
                15,
            ),
            (CabacResidualCategory::Luma8x8, None, 402, 417, 426, 64),
        ];
        for (category, coded, significant, last, level, maximum) in expected {
            assert_eq!(category.coded_block_flag_context_base(), coded);
            assert_eq!(category.significant_coeff_context_base(), significant);
            assert_eq!(category.last_significant_coeff_context_base(), last);
            assert_eq!(category.coefficient_level_context_base(), level);
            assert_eq!(category.maximum_coefficients(), maximum);
        }
    }

    #[test]
    fn derives_coded_block_contexts_from_available_blocks() {
        let mut state = CabacResidualState::new(1, 1).unwrap();
        assert_eq!(
            state
                .coded_block_flag_context_index(0, 1, true, CabacResidualBlock::Luma4x4(0),)
                .unwrap(),
            Some(96)
        );
        assert_eq!(
            state
                .coded_block_flag_context_index(0, 1, false, CabacResidualBlock::Luma4x4(0),)
                .unwrap(),
            Some(93)
        );

        state
            .record_block(0, 1, CabacResidualBlock::Luma4x4(0), true)
            .unwrap();
        assert_eq!(
            state
                .coded_block_flag_context_index(0, 1, false, CabacResidualBlock::Luma4x4(1),)
                .unwrap(),
            Some(94)
        );
        assert_eq!(
            state
                .coded_block_flag_context_index(0, 2, false, CabacResidualBlock::Luma4x4(1),)
                .unwrap(),
            Some(93)
        );
    }

    #[test]
    fn records_eight_by_eight_and_pcm_availability() {
        let mut state = CabacResidualState::new(2, 1).unwrap();
        state
            .record_block(0, 3, CabacResidualBlock::Luma8x8(1), true)
            .unwrap();
        assert_eq!(
            state
                .coded_block_flag_context_index(1, 3, false, CabacResidualBlock::Luma4x4(0),)
                .unwrap(),
            Some(94)
        );

        state.record_pcm_macroblock(0, 4).unwrap();
        assert_eq!(
            state
                .coded_block_flag_context_index(
                    1,
                    4,
                    false,
                    CabacResidualBlock::ChromaAc { plane: 1, block: 0 },
                )
                .unwrap(),
            Some(102)
        );
        assert_eq!(
            state
                .coded_block_flag_context_index(1, 4, false, CabacResidualBlock::Luma8x8(0),)
                .unwrap(),
            None
        );
    }

    #[test]
    fn rejects_out_of_range_block_identifiers() {
        let mut state = CabacResidualState::new(1, 1).unwrap();
        assert!(
            state
                .record_block(0, 1, CabacResidualBlock::Luma4x4(16), true)
                .is_err()
        );
        assert!(
            state
                .record_block(
                    0,
                    1,
                    CabacResidualBlock::ChromaAc { plane: 2, block: 0 },
                    true,
                )
                .is_err()
        );
        assert!(
            state
                .record_block(1, 1, CabacResidualBlock::LumaDc, true)
                .is_err()
        );
    }
}
