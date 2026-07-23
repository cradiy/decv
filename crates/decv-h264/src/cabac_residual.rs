//! CABAC residual-block categories and coded-block neighbour state.

use crate::{
    CabacSyntaxDecoder, CodedBlockPattern, H264Error, InterResidual, IntraLumaPrediction,
    IntraMacroblockHeader, IntraResidual, ResidualBlock, Result,
};

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
const SIGNIFICANT_COEFF_OFFSETS_8X8: [u8; 63] = [
    0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8,
    7, 7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9, 11, 12,
    13, 11, 14, 10, 12,
];
const LAST_SIGNIFICANT_COEFF_OFFSETS_8X8: [u8; 63] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8,
];
const COEFFICIENT_LEVEL_ONE_CONTEXTS: [usize; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
const COEFFICIENT_LEVEL_GREATER_THAN_ONE_CONTEXTS: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 9];
const LEVEL_ONE_TRANSITIONS: [usize; 8] = [1, 2, 3, 3, 4, 5, 6, 7];
const LEVEL_GREATER_THAN_ONE_TRANSITIONS: [usize; 8] = [4, 4, 4, 4, 5, 6, 7, 7];
const MAXIMUM_COEFFICIENT_ESCAPE_PREFIX: u8 = 23;

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

/// Significant coefficient positions in increasing scan order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CabacSignificanceMap {
    indices: [u8; 64],
    count: u8,
}

impl CabacSignificanceMap {
    #[inline]
    pub const fn count(self) -> u8 {
        self.count
    }

    #[inline]
    pub fn indices(&self) -> &[u8] {
        &self.indices[..usize::from(self.count)]
    }
}

/// Decoded transform coefficients in scan order.
///
/// Only the prefix returned by [`Self::coefficients`] belongs to the selected
/// residual category. The fixed backing storage also accommodates one complete
/// 8x8 transform block without allocating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CabacCoefficientBlock {
    coefficients: [i32; 64],
    coefficient_count: u8,
    maximum_coefficients: u8,
}

impl CabacCoefficientBlock {
    #[inline]
    pub const fn coefficient_count(self) -> u8 {
        self.coefficient_count
    }

    #[inline]
    pub fn coefficients(&self) -> &[i32] {
        &self.coefficients[..usize::from(self.maximum_coefficients)]
    }
}

/// Decodes `significant_coeff_flag` and `last_significant_coeff_flag` for one
/// block whose coded-block flag is already known to be one.
pub fn decode_cabac_significance_map(
    syntax: &mut CabacSyntaxDecoder<'_, '_>,
    category: CabacResidualCategory,
) -> Result<CabacSignificanceMap> {
    decode_significance_map_with(category, |context_index| syntax.decision(context_index))
}

fn decode_significance_map_with(
    category: CabacResidualCategory,
    mut decision: impl FnMut(usize) -> Result<u8>,
) -> Result<CabacSignificanceMap> {
    let maximum = usize::from(category.maximum_coefficients());
    let mut indices = [0; 64];
    let mut count = 0usize;
    for coefficient_index in 0..maximum - 1 {
        let significant_offset = if category == CabacResidualCategory::Luma8x8 {
            usize::from(SIGNIFICANT_COEFF_OFFSETS_8X8[coefficient_index])
        } else {
            coefficient_index
        };
        if decision(category.significant_coeff_context_base() + significant_offset)? == 0 {
            continue;
        }
        indices[count] = u8::try_from(coefficient_index).map_err(|_| H264Error::IntegerOverflow)?;
        count += 1;
        let last_offset = if category == CabacResidualCategory::Luma8x8 {
            usize::from(LAST_SIGNIFICANT_COEFF_OFFSETS_8X8[coefficient_index])
        } else {
            coefficient_index
        };
        if decision(category.last_significant_coeff_context_base() + last_offset)? != 0 {
            return Ok(CabacSignificanceMap {
                indices,
                count: u8::try_from(count).map_err(|_| H264Error::IntegerOverflow)?,
            });
        }
    }

    indices[count] = u8::try_from(maximum - 1).map_err(|_| H264Error::IntegerOverflow)?;
    count += 1;
    Ok(CabacSignificanceMap {
        indices,
        count: u8::try_from(count).map_err(|_| H264Error::IntegerOverflow)?,
    })
}

/// Decodes coefficient magnitudes and signs for an existing significance map.
pub fn decode_cabac_coefficient_levels(
    syntax: &mut CabacSyntaxDecoder<'_, '_>,
    category: CabacResidualCategory,
    significance_map: CabacSignificanceMap,
) -> Result<CabacCoefficientBlock> {
    decode_coefficient_levels_with(category, significance_map, |context_index| {
        if let Some(context_index) = context_index {
            syntax.decision(context_index)
        } else {
            syntax.bypass()
        }
    })
}

/// Decodes a complete coded CABAC residual block.
pub fn decode_cabac_coefficient_block(
    syntax: &mut CabacSyntaxDecoder<'_, '_>,
    category: CabacResidualCategory,
) -> Result<CabacCoefficientBlock> {
    let significance_map = decode_cabac_significance_map(syntax, category)?;
    decode_cabac_coefficient_levels(syntax, category, significance_map)
}

fn decode_coefficient_levels_with(
    category: CabacResidualCategory,
    significance_map: CabacSignificanceMap,
    mut decode: impl FnMut(Option<usize>) -> Result<u8>,
) -> Result<CabacCoefficientBlock> {
    let context_base = category.coefficient_level_context_base();
    let mut coefficients = [0; 64];
    let mut node_context = 0usize;

    for &coefficient_index in significance_map.indices().iter().rev() {
        let first_context = context_base + COEFFICIENT_LEVEL_ONE_CONTEXTS[node_context];
        let magnitude = if decode(Some(first_context))? == 0 {
            node_context = LEVEL_ONE_TRANSITIONS[node_context];
            1
        } else {
            let repeated_context =
                context_base + COEFFICIENT_LEVEL_GREATER_THAN_ONE_CONTEXTS[node_context];
            node_context = LEVEL_GREATER_THAN_ONE_TRANSITIONS[node_context];
            let mut magnitude = 2u32;
            while magnitude < 15 && decode(Some(repeated_context))? != 0 {
                magnitude += 1;
            }
            if magnitude == 15 {
                let mut prefix = 0u8;
                while decode(None)? != 0 {
                    prefix = prefix.checked_add(1).ok_or(H264Error::IntegerOverflow)?;
                    if prefix > MAXIMUM_COEFFICIENT_ESCAPE_PREFIX {
                        return Err(H264Error::InvalidSyntax(
                            "CABAC coefficient escape prefix is too long",
                        ));
                    }
                }
                let mut suffix = 0u32;
                for _ in 0..prefix {
                    suffix = (suffix << 1) | u32::from(decode(None)?);
                }
                (1u32 << prefix)
                    .checked_add(suffix)
                    .and_then(|value| value.checked_add(14))
                    .ok_or(H264Error::IntegerOverflow)?
            } else {
                magnitude
            }
        };

        let magnitude = i32::try_from(magnitude).map_err(|_| H264Error::IntegerOverflow)?;
        let coefficient = if decode(None)? == 0 {
            magnitude
        } else {
            -magnitude
        };
        coefficients[usize::from(coefficient_index)] = coefficient;
    }

    Ok(CabacCoefficientBlock {
        coefficients,
        coefficient_count: significance_map.count,
        maximum_coefficients: category.maximum_coefficients(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockState {
    slice_id: u32,
    coded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MacroblockResidualSnapshot {
    luma: [Option<BlockState>; 16],
    chroma_cb: [Option<BlockState>; 4],
    chroma_cr: [Option<BlockState>; 4],
    luma_dc: Option<BlockState>,
    chroma_dc_cb: Option<BlockState>,
    chroma_dc_cr: Option<BlockState>,
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

    /// Decodes and records every residual block of one progressive 4:2:0
    /// intra macroblock.
    ///
    /// Neighbour state is committed only after the complete macroblock
    /// succeeds. A CABAC syntax error remains fatal to the surrounding slice,
    /// so the arithmetic decoder itself is not rewound.
    pub fn decode_intra_residual(
        &mut self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        header: &IntraMacroblockHeader,
    ) -> Result<IntraResidual> {
        validate_coded_block_pattern(header.coded_block_pattern)?;
        self.validate_macroblock(macroblock_address)?;
        let snapshot = self.snapshot_macroblock(macroblock_address);
        match self.decode_intra_residual_inner(syntax, macroblock_address, slice_id, header) {
            Ok(residual) => Ok(residual),
            Err(error) => {
                self.restore_macroblock(macroblock_address, snapshot);
                Err(error)
            }
        }
    }

    /// Decodes and records every residual block of one progressive 4:2:0
    /// inter macroblock.
    pub fn decode_inter_residual(
        &mut self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        coded_block_pattern: CodedBlockPattern,
        transform_size_8x8: bool,
    ) -> Result<InterResidual> {
        validate_coded_block_pattern(coded_block_pattern)?;
        self.validate_macroblock(macroblock_address)?;
        let snapshot = self.snapshot_macroblock(macroblock_address);
        match self.decode_inter_residual_inner(
            syntax,
            macroblock_address,
            slice_id,
            coded_block_pattern,
            transform_size_8x8,
        ) {
            Ok(residual) => Ok(residual),
            Err(error) => {
                self.restore_macroblock(macroblock_address, snapshot);
                Err(error)
            }
        }
    }

    fn decode_intra_residual_inner(
        &mut self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        header: &IntraMacroblockHeader,
    ) -> Result<IntraResidual> {
        let intra_16x16 = matches!(
            header.luma_prediction,
            IntraLumaPrediction::SixteenBySixteen { .. }
        );
        let luma_dc = if intra_16x16 {
            Some(self.decode_residual_block(
                syntax,
                macroblock_address,
                slice_id,
                true,
                CabacResidualBlock::LumaDc,
            )?)
        } else {
            None
        };
        let transform_size_8x8 =
            matches!(header.luma_prediction, IntraLumaPrediction::EightByEight(_));
        let luma = self.decode_luma_residual(
            syntax,
            macroblock_address,
            slice_id,
            true,
            header.coded_block_pattern.luma,
            intra_16x16,
            transform_size_8x8,
        )?;
        let (chroma_dc, chroma_ac) = self.decode_chroma_residual(
            syntax,
            macroblock_address,
            slice_id,
            true,
            header.coded_block_pattern.chroma,
        )?;
        Ok(IntraResidual {
            luma_dc,
            luma,
            chroma_dc,
            chroma_ac,
        })
    }

    fn decode_inter_residual_inner(
        &mut self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        coded_block_pattern: CodedBlockPattern,
        transform_size_8x8: bool,
    ) -> Result<InterResidual> {
        let luma = self.decode_luma_residual(
            syntax,
            macroblock_address,
            slice_id,
            false,
            coded_block_pattern.luma,
            false,
            transform_size_8x8,
        )?;
        let (chroma_dc, chroma_ac) = self.decode_chroma_residual(
            syntax,
            macroblock_address,
            slice_id,
            false,
            coded_block_pattern.chroma,
        )?;
        Ok(InterResidual {
            luma,
            chroma_dc,
            chroma_ac,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_luma_residual(
        &mut self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        current_is_intra: bool,
        coded_block_pattern_luma: u8,
        intra_16x16: bool,
        transform_size_8x8: bool,
    ) -> Result<[ResidualBlock; 16]> {
        let maximum_coefficients = if intra_16x16 { 15 } else { 16 };
        let mut luma = [ResidualBlock::empty(maximum_coefficients); 16];

        if transform_size_8x8 {
            for region in 0..4u8 {
                let block = CabacResidualBlock::Luma8x8(region);
                if coded_block_pattern_luma & (1 << region) == 0 {
                    self.record_block(macroblock_address, slice_id, block, false)?;
                    continue;
                }
                let coefficients = self
                    .decode_coded_coefficient_block(
                        syntax,
                        macroblock_address,
                        slice_id,
                        current_is_intra,
                        block,
                    )?
                    .expect("4:2:0 luma 8x8 blocks have no coded_block_flag");
                let split = split_luma_8x8(coefficients)?;
                luma[usize::from(region) * 4..usize::from(region + 1) * 4].copy_from_slice(&split);
            }
            return Ok(luma);
        }

        for block_index in 0..16u8 {
            let block = if intra_16x16 {
                CabacResidualBlock::LumaAc(block_index)
            } else {
                CabacResidualBlock::Luma4x4(block_index)
            };
            if coded_block_pattern_luma & (1 << (block_index / 4)) == 0 {
                self.record_block(macroblock_address, slice_id, block, false)?;
                continue;
            }
            luma[usize::from(block_index)] = self.decode_residual_block(
                syntax,
                macroblock_address,
                slice_id,
                current_is_intra,
                block,
            )?;
        }
        Ok(luma)
    }

    fn decode_chroma_residual(
        &mut self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        current_is_intra: bool,
        coded_block_pattern_chroma: u8,
    ) -> Result<([ResidualBlock; 2], [[ResidualBlock; 4]; 2])> {
        let mut chroma_dc = [ResidualBlock::empty(4); 2];
        for plane in 0..2u8 {
            let block = CabacResidualBlock::ChromaDc { plane };
            if coded_block_pattern_chroma == 0 {
                self.record_block(macroblock_address, slice_id, block, false)?;
            } else {
                chroma_dc[usize::from(plane)] = self.decode_residual_block(
                    syntax,
                    macroblock_address,
                    slice_id,
                    current_is_intra,
                    block,
                )?;
            }
        }

        let mut chroma_ac = [[ResidualBlock::empty(15); 4]; 2];
        for plane in 0..2u8 {
            for block_index in 0..4u8 {
                let block = CabacResidualBlock::ChromaAc {
                    plane,
                    block: block_index,
                };
                if coded_block_pattern_chroma < 2 {
                    self.record_block(macroblock_address, slice_id, block, false)?;
                } else {
                    chroma_ac[usize::from(plane)][usize::from(block_index)] = self
                        .decode_residual_block(
                            syntax,
                            macroblock_address,
                            slice_id,
                            current_is_intra,
                            block,
                        )?;
                }
            }
        }
        Ok((chroma_dc, chroma_ac))
    }

    fn decode_residual_block(
        &mut self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        current_is_intra: bool,
        block: CabacResidualBlock,
    ) -> Result<ResidualBlock> {
        let maximum_coefficients = block.category().maximum_coefficients();
        let coefficients = self.decode_coded_coefficient_block(
            syntax,
            macroblock_address,
            slice_id,
            current_is_intra,
            block,
        )?;
        coefficients.map_or_else(
            || Ok(ResidualBlock::empty(maximum_coefficients)),
            coefficient_block_to_residual,
        )
    }

    fn decode_coded_coefficient_block(
        &mut self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        current_is_intra: bool,
        block: CabacResidualBlock,
    ) -> Result<Option<CabacCoefficientBlock>> {
        let coded = match self.coded_block_flag_context_index(
            macroblock_address,
            slice_id,
            current_is_intra,
            block,
        )? {
            Some(context_index) => syntax.decision(context_index)? != 0,
            None => true,
        };
        if !coded {
            self.record_block(macroblock_address, slice_id, block, false)?;
            return Ok(None);
        }

        let coefficients = decode_cabac_coefficient_block(syntax, block.category())?;
        self.record_block(macroblock_address, slice_id, block, true)?;
        Ok(Some(coefficients))
    }

    fn snapshot_macroblock(&self, macroblock_address: usize) -> MacroblockResidualSnapshot {
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let mut snapshot = MacroblockResidualSnapshot {
            luma: [None; 16],
            chroma_cb: [None; 4],
            chroma_cr: [None; 4],
            luma_dc: self.luma_dc.entries[macroblock_address],
            chroma_dc_cb: self.chroma_dc_cb.entries[macroblock_address],
            chroma_dc_cr: self.chroma_dc_cr.entries[macroblock_address],
        };
        for (index, &(block_x, block_y)) in LUMA_BLOCK_COORDINATES.iter().enumerate() {
            let x = macroblock_x * 4 + block_x;
            let y = macroblock_y * 4 + block_y;
            snapshot.luma[index] = self.luma.entries[y * self.luma.width + x];
        }
        for (index, &(block_x, block_y)) in CHROMA_BLOCK_COORDINATES.iter().enumerate() {
            let x = macroblock_x * 2 + block_x;
            let y = macroblock_y * 2 + block_y;
            snapshot.chroma_cb[index] = self.chroma_cb.entries[y * self.chroma_cb.width + x];
            snapshot.chroma_cr[index] = self.chroma_cr.entries[y * self.chroma_cr.width + x];
        }
        snapshot
    }

    fn restore_macroblock(
        &mut self,
        macroblock_address: usize,
        snapshot: MacroblockResidualSnapshot,
    ) {
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        self.luma_dc.entries[macroblock_address] = snapshot.luma_dc;
        self.chroma_dc_cb.entries[macroblock_address] = snapshot.chroma_dc_cb;
        self.chroma_dc_cr.entries[macroblock_address] = snapshot.chroma_dc_cr;
        for (index, &(block_x, block_y)) in LUMA_BLOCK_COORDINATES.iter().enumerate() {
            let x = macroblock_x * 4 + block_x;
            let y = macroblock_y * 4 + block_y;
            self.luma.entries[y * self.luma.width + x] = snapshot.luma[index];
        }
        for (index, &(block_x, block_y)) in CHROMA_BLOCK_COORDINATES.iter().enumerate() {
            let x = macroblock_x * 2 + block_x;
            let y = macroblock_y * 2 + block_y;
            self.chroma_cb.entries[y * self.chroma_cb.width + x] = snapshot.chroma_cb[index];
            self.chroma_cr.entries[y * self.chroma_cr.width + x] = snapshot.chroma_cr[index];
        }
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

fn validate_coded_block_pattern(coded_block_pattern: CodedBlockPattern) -> Result<()> {
    if coded_block_pattern.luma > 15 || coded_block_pattern.chroma > 2 {
        return Err(H264Error::InvalidSyntax(
            "coded block pattern exceeds 4:2:0 macroblock bounds",
        ));
    }
    Ok(())
}

fn coefficient_block_to_residual(block: CabacCoefficientBlock) -> Result<ResidualBlock> {
    if block.maximum_coefficients > 16 {
        return Err(H264Error::InvalidSyntax(
            "CABAC coefficient block does not fit a 4x4 residual",
        ));
    }
    let mut coefficients = [0; 16];
    coefficients[..usize::from(block.maximum_coefficients)].copy_from_slice(block.coefficients());
    Ok(ResidualBlock {
        coefficients,
        total_coeff: block.coefficient_count,
        max_num_coeff: block.maximum_coefficients,
    })
}

fn split_luma_8x8(block: CabacCoefficientBlock) -> Result<[ResidualBlock; 4]> {
    if block.maximum_coefficients != 64 {
        return Err(H264Error::InvalidSyntax(
            "CABAC luma 8x8 split requires 64 coefficients",
        ));
    }
    let mut output = [ResidualBlock::empty(16); 4];
    for (coefficient_index, &coefficient) in block.coefficients().iter().enumerate() {
        let output_block = coefficient_index % 4;
        output[output_block].coefficients[coefficient_index / 4] = coefficient;
        output[output_block].total_coeff += u8::from(coefficient != 0);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use bit_readers::BitReader;

    use crate::{
        CabacContextSet, CabacDecoder, CabacInitializationTable, IntraPredictionModeSyntax,
    };

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

    #[test]
    fn decodes_linear_significance_and_last_flags() {
        let mut bins = VecDeque::from([0, 1, 0, 0, 1, 1]);
        let mut visited = Vec::new();
        let map = decode_significance_map_with(CabacResidualCategory::Luma4x4, |context_index| {
            visited.push(context_index);
            Ok(bins.pop_front().unwrap())
        })
        .unwrap();
        assert_eq!(map.indices(), [1, 3]);
        assert_eq!(visited, [134, 135, 196, 136, 137, 198]);
    }

    #[test]
    fn infers_the_final_coefficient_when_no_last_flag_is_set() {
        let mut visited = Vec::new();
        let map = decode_significance_map_with(CabacResidualCategory::LumaDc, |context_index| {
            visited.push(context_index);
            Ok(0)
        })
        .unwrap();
        assert_eq!(map.indices(), [15]);
        assert_eq!(visited, (105..120).collect::<Vec<_>>());
    }

    #[test]
    fn applies_non_linear_eight_by_eight_significance_contexts() {
        let mut bins = VecDeque::from([0, 0, 0, 0, 0, 0, 1, 1]);
        let mut visited = Vec::new();
        let map = decode_significance_map_with(CabacResidualCategory::Luma8x8, |context_index| {
            visited.push(context_index);
            Ok(bins.pop_front().unwrap())
        })
        .unwrap();
        assert_eq!(map.indices(), [6]);
        assert_eq!(visited, [402, 403, 404, 405, 406, 407, 407, 418]);
    }

    #[test]
    fn decodes_coefficient_levels_in_reverse_scan_order() {
        let map = CabacSignificanceMap {
            indices: {
                let mut indices = [0; 64];
                indices[..3].copy_from_slice(&[1, 3, 6]);
                indices
            },
            count: 3,
        };
        let mut bins = VecDeque::from([0, 0, 1, 1, 1, 0, 1, 0, 1]);
        let mut visited = Vec::new();
        let block =
            decode_coefficient_levels_with(CabacResidualCategory::Luma4x4, map, |context| {
                visited.push(context);
                Ok(bins.pop_front().unwrap())
            })
            .unwrap();

        assert_eq!(block.coefficient_count(), 3);
        assert_eq!(&block.coefficients()[..7], [0, -1, 0, -4, 0, 0, 1]);
        assert_eq!(
            visited,
            [
                Some(248),
                None,
                Some(249),
                Some(252),
                Some(252),
                Some(252),
                None,
                Some(247),
                None,
            ]
        );
    }

    #[test]
    fn decodes_coefficient_escape_prefix_and_suffix() {
        let map = CabacSignificanceMap {
            indices: [0; 64],
            count: 1,
        };
        let mut bins = VecDeque::from(
            [1].into_iter()
                .chain(std::iter::repeat_n(1, 13))
                .chain([1, 1, 0, 1, 0, 0])
                .collect::<Vec<_>>(),
        );
        let block = decode_coefficient_levels_with(CabacResidualCategory::LumaDc, map, |_| {
            Ok(bins.pop_front().unwrap())
        })
        .unwrap();

        assert_eq!(block.coefficient_count(), 1);
        assert_eq!(block.coefficients()[0], 20);
        assert!(bins.is_empty());
    }

    #[test]
    fn rejects_unbounded_coefficient_escape_prefixes() {
        let map = CabacSignificanceMap {
            indices: [0; 64],
            count: 1,
        };
        let mut decision_count = 0;
        let error =
            decode_coefficient_levels_with(CabacResidualCategory::ChromaDc, map, |context| {
                if context.is_some() {
                    decision_count += 1;
                    Ok(1)
                } else {
                    Ok(1)
                }
            })
            .unwrap_err();

        assert_eq!(
            error,
            H264Error::InvalidSyntax("CABAC coefficient escape prefix is too long")
        );
        assert_eq!(decision_count, 14);
    }

    #[test]
    fn converts_small_blocks_and_interleaves_eight_by_eight_blocks() {
        let small = CabacCoefficientBlock {
            coefficients: {
                let mut coefficients = [0; 64];
                coefficients[0] = 7;
                coefficients[14] = -3;
                coefficients
            },
            coefficient_count: 2,
            maximum_coefficients: 15,
        };
        assert_eq!(
            coefficient_block_to_residual(small).unwrap(),
            ResidualBlock {
                coefficients: {
                    let mut coefficients = [0; 16];
                    coefficients[0] = 7;
                    coefficients[14] = -3;
                    coefficients
                },
                total_coeff: 2,
                max_num_coeff: 15,
            }
        );

        let block_8x8 = CabacCoefficientBlock {
            coefficients: std::array::from_fn(|index| i32::try_from(index + 1).unwrap()),
            coefficient_count: 64,
            maximum_coefficients: 64,
        };
        let split = split_luma_8x8(block_8x8).unwrap();
        for (block_index, block) in split.iter().enumerate() {
            assert_eq!(block.total_coeff, 16);
            assert_eq!(block.max_num_coeff, 16);
            assert_eq!(
                block.coefficients,
                std::array::from_fn(|index| i32::try_from(index * 4 + block_index + 1).unwrap())
            );
        }
    }

    #[test]
    fn assembles_and_records_an_inferred_zero_intra_macroblock() {
        let mode = IntraPredictionModeSyntax {
            use_predicted: true,
            remaining_mode: None,
        };
        let header = IntraMacroblockHeader {
            luma_prediction: IntraLumaPrediction::FourByFour([mode; 16]),
            chroma_prediction_mode: 0,
            coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
            qp_delta: 0,
        };
        let mut arithmetic = CabacDecoder::new(BitReader::new(&[0; 2])).unwrap();
        let mut contexts = CabacContextSet::new(CabacInitializationTable::Intra, 26).unwrap();
        let mut syntax = CabacSyntaxDecoder::new(&mut arithmetic, &mut contexts);
        let mut state = CabacResidualState::new(2, 1).unwrap();

        let residual = state
            .decode_intra_residual(&mut syntax, 0, 7, &header)
            .unwrap();

        assert!(residual.luma_dc.is_none());
        assert!(residual.luma.iter().all(|block| block.total_coeff == 0));
        assert!(
            residual
                .chroma_dc
                .iter()
                .chain(residual.chroma_ac.iter().flatten())
                .all(|block| block.total_coeff == 0)
        );
        assert_eq!(
            state
                .coded_block_flag_context_index(1, 7, true, CabacResidualBlock::Luma4x4(0),)
                .unwrap(),
            Some(95)
        );
    }

    #[test]
    fn restores_only_the_current_macroblock_state_after_residual_failure() {
        let mode = IntraPredictionModeSyntax {
            use_predicted: true,
            remaining_mode: None,
        };
        let header = IntraMacroblockHeader {
            luma_prediction: IntraLumaPrediction::FourByFour([mode; 16]),
            chroma_prediction_mode: 0,
            coded_block_pattern: CodedBlockPattern {
                luma: 15,
                chroma: 2,
            },
            qp_delta: 0,
        };
        let mut arithmetic = CabacDecoder::new(BitReader::new(&[0; 2])).unwrap();
        let mut contexts = CabacContextSet::new(CabacInitializationTable::Intra, 26).unwrap();
        let mut syntax = CabacSyntaxDecoder::new(&mut arithmetic, &mut contexts);
        let mut state = CabacResidualState::new(2, 1).unwrap();
        state
            .record_block(1, 3, CabacResidualBlock::Luma4x4(0), true)
            .unwrap();
        let before_current = state.snapshot_macroblock(0);
        let before_other = state.snapshot_macroblock(1);

        assert!(
            state
                .decode_intra_residual(&mut syntax, 0, 3, &header)
                .is_err()
        );
        assert_eq!(state.snapshot_macroblock(0), before_current);
        assert_eq!(state.snapshot_macroblock(1), before_other);
    }
}
