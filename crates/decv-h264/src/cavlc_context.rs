//! Neighbouring-block state used to derive the CAVLC `nC` context.

use bit_readers::BitReader;

use crate::{
    BMacroblockContext, BSliceMacroblock, CodedBlockPattern, CoeffTokenContext,
    DecodedBSliceMacroblock, DecodedIntraMacroblock, DecodedPSliceMacroblock, H264Error,
    InterResidual, IntraLumaPrediction, IntraMacroblock, IntraMacroblockHeader, IntraResidual,
    PInterMacroblockHeader, PMacroblockContext, PSliceMacroblock, ResidualBlock, Result,
    decode_residual_block, parse_cavlc_b_macroblock, parse_cavlc_intra_macroblock,
    parse_cavlc_p_macroblock,
};

const DECODED_BIT: u32 = 1 << 5;
const SLICE_SHIFT: u32 = 6;
const MAX_SLICE_ID: u32 = u32::MAX >> SLICE_SHIFT;

// H.264 4x4 block scanning order inside one 16x16 luma macroblock.
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

#[derive(Debug, Clone)]
struct BlockGrid {
    width: usize,
    height: usize,
    entries: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MacroblockSnapshot {
    luma: [u32; 16],
    chroma_cb: [u32; 4],
    chroma_cr: [u32; 4],
}

impl BlockGrid {
    fn new(width: usize, height: usize) -> Result<Self> {
        let len = width
            .checked_mul(height)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width,
            height,
            entries: vec![0; len],
        })
    }

    #[inline]
    fn neighbour_total(&self, x: usize, y: usize, slice_id: u32) -> u8 {
        let left = x
            .checked_sub(1)
            .and_then(|left_x| self.decoded_total(left_x, y, slice_id));
        let top = y
            .checked_sub(1)
            .and_then(|top_y| self.decoded_total(x, top_y, slice_id));
        match (left, top) {
            (Some(left), Some(top)) => (left + top + 1) >> 1,
            (Some(value), None) | (None, Some(value)) => value,
            (None, None) => 0,
        }
    }

    #[inline]
    fn decoded_total(&self, x: usize, y: usize, slice_id: u32) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let entry = self.entries[y * self.width + x];
        if entry & DECODED_BIT != 0 && entry >> SLICE_SHIFT == slice_id {
            Some((entry & 0x1f) as u8)
        } else {
            None
        }
    }

    #[inline]
    fn record(&mut self, x: usize, y: usize, slice_id: u32, total_coeff: u8) {
        self.entries[y * self.width + x] =
            (slice_id << SLICE_SHIFT) | DECODED_BIT | u32::from(total_coeff);
    }

    fn clear(&mut self) {
        self.entries.fill(0);
    }

    fn inactive(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            entries: Vec::new(),
        }
    }
}

/// Per-picture CAVLC context state for 4:2:0 streams.
///
/// A new instance is used for each decoded picture. Call [`Self::begin_slice`]
/// before decoding every slice; entries from another slice are intentionally
/// treated as unavailable, as required by H.264 neighbour derivation.
#[derive(Debug, Clone)]
pub struct CavlcNeighborState {
    width_in_mbs: usize,
    height_in_mbs: usize,
    slice_id: u32,
    luma: BlockGrid,
    chroma_cb: BlockGrid,
    chroma_cr: BlockGrid,
}

impl CavlcNeighborState {
    pub fn new(width_in_mbs: u32, height_in_mbs: u32) -> Result<Self> {
        let width_in_mbs = usize::try_from(width_in_mbs).map_err(|_| H264Error::IntegerOverflow)?;
        let height_in_mbs =
            usize::try_from(height_in_mbs).map_err(|_| H264Error::IntegerOverflow)?;
        if width_in_mbs == 0 || height_in_mbs == 0 {
            return Err(H264Error::InvalidSyntax(
                "CAVLC context dimensions must be non-zero",
            ));
        }
        let luma_width = width_in_mbs
            .checked_mul(4)
            .ok_or(H264Error::IntegerOverflow)?;
        let luma_height = height_in_mbs
            .checked_mul(4)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma_width = width_in_mbs
            .checked_mul(2)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma_height = height_in_mbs
            .checked_mul(2)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width_in_mbs,
            height_in_mbs,
            slice_id: 0,
            luma: BlockGrid::new(luma_width, luma_height)?,
            chroma_cb: BlockGrid::new(chroma_width, chroma_height)?,
            chroma_cr: BlockGrid::new(chroma_width, chroma_height)?,
        })
    }

    #[inline(never)]
    pub(crate) fn new_inactive(width_in_mbs: u32, height_in_mbs: u32) -> Result<Self> {
        let width_in_mbs = usize::try_from(width_in_mbs).map_err(|_| H264Error::IntegerOverflow)?;
        let height_in_mbs =
            usize::try_from(height_in_mbs).map_err(|_| H264Error::IntegerOverflow)?;
        if width_in_mbs == 0 || height_in_mbs == 0 {
            return Err(H264Error::InvalidSyntax(
                "CAVLC context dimensions must be non-zero",
            ));
        }
        let luma_width = width_in_mbs
            .checked_mul(4)
            .ok_or(H264Error::IntegerOverflow)?;
        let luma_height = height_in_mbs
            .checked_mul(4)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma_width = width_in_mbs
            .checked_mul(2)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma_height = height_in_mbs
            .checked_mul(2)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width_in_mbs,
            height_in_mbs,
            slice_id: 0,
            luma: BlockGrid::inactive(luma_width, luma_height),
            chroma_cb: BlockGrid::inactive(chroma_width, chroma_height),
            chroma_cr: BlockGrid::inactive(chroma_width, chroma_height),
        })
    }

    #[cfg(test)]
    pub(crate) fn has_backing_storage(&self) -> bool {
        !self.luma.entries.is_empty()
            && !self.chroma_cb.entries.is_empty()
            && !self.chroma_cr.entries.is_empty()
    }

    /// Starts a new slice and invalidates cross-slice neighbour availability.
    pub fn begin_slice(&mut self) {
        if self.slice_id == MAX_SLICE_ID {
            self.luma.clear();
            self.chroma_cb.clear();
            self.chroma_cr.clear();
            self.slice_id = 1;
        } else {
            self.slice_id += 1;
        }
    }

    /// Parses and coefficient-decodes one complete CAVLC I macroblock.
    ///
    /// This is the transactional entry point intended for slice decoding:
    /// syntax bits and neighbour state are committed only after the whole
    /// macroblock succeeds.
    pub fn decode_intra_macroblock(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        transform_8x8_mode_enabled: bool,
    ) -> Result<DecodedIntraMacroblock> {
        self.ensure_slice_started()?;
        self.ensure_macroblock(mb_x, mb_y)?;
        let mut probe = *reader;
        let decoded = match parse_cavlc_intra_macroblock(&mut probe, transform_8x8_mode_enabled)? {
            IntraMacroblock::Predicted(header) => {
                let residual = self.decode_intra_residual(&mut probe, mb_x, mb_y, &header)?;
                DecodedIntraMacroblock {
                    macroblock: IntraMacroblock::Predicted(header),
                    residual: Some(residual),
                }
            }
            IntraMacroblock::Pcm(pcm) => {
                self.record_pcm_macroblock(mb_x, mb_y)?;
                DecodedIntraMacroblock {
                    macroblock: IntraMacroblock::Pcm(pcm),
                    residual: None,
                }
            }
        };
        *reader = probe;
        Ok(decoded)
    }

    /// Parses and coefficient-decodes one complete non-skipped CAVLC P-slice
    /// macroblock, including embedded Intra macroblock types.
    pub fn decode_p_macroblock(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        context: PMacroblockContext,
    ) -> Result<DecodedPSliceMacroblock> {
        self.ensure_slice_started()?;
        self.ensure_macroblock(mb_x, mb_y)?;
        let snapshot = self.snapshot_macroblock(mb_x, mb_y)?;
        let mut probe = *reader;
        let result = (|| {
            Ok(match parse_cavlc_p_macroblock(&mut probe, context)? {
                PSliceMacroblock::Inter(header) => {
                    let residual = self.decode_inter_residual(&mut probe, mb_x, mb_y, &header)?;
                    DecodedPSliceMacroblock::Inter { header, residual }
                }
                PSliceMacroblock::Intra(IntraMacroblock::Predicted(header)) => {
                    let residual = self.decode_intra_residual(&mut probe, mb_x, mb_y, &header)?;
                    DecodedPSliceMacroblock::Intra(DecodedIntraMacroblock {
                        macroblock: IntraMacroblock::Predicted(header),
                        residual: Some(residual),
                    })
                }
                PSliceMacroblock::Intra(IntraMacroblock::Pcm(pcm)) => {
                    self.record_pcm_macroblock(mb_x, mb_y)?;
                    DecodedPSliceMacroblock::Intra(DecodedIntraMacroblock {
                        macroblock: IntraMacroblock::Pcm(pcm),
                        residual: None,
                    })
                }
            })
        })();
        match result {
            Ok(decoded) => {
                *reader = probe;
                Ok(decoded)
            }
            Err(error) => {
                self.restore_macroblock(mb_x, mb_y, snapshot);
                Err(error)
            }
        }
    }

    /// Parses and coefficient-decodes one complete non-skipped CAVLC B-slice
    /// macroblock, including embedded Intra macroblock types.
    pub fn decode_b_macroblock(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        context: BMacroblockContext,
    ) -> Result<DecodedBSliceMacroblock> {
        self.ensure_slice_started()?;
        self.ensure_macroblock(mb_x, mb_y)?;
        let snapshot = self.snapshot_macroblock(mb_x, mb_y)?;
        let mut probe = *reader;
        let result = (|| {
            Ok(match parse_cavlc_b_macroblock(&mut probe, context)? {
                BSliceMacroblock::Inter(header) => {
                    let residual = self.decode_inter_residual_pattern(
                        &mut probe,
                        mb_x,
                        mb_y,
                        header.coded_block_pattern,
                    )?;
                    DecodedBSliceMacroblock::Inter { header, residual }
                }
                BSliceMacroblock::Intra(IntraMacroblock::Predicted(header)) => {
                    let residual = self.decode_intra_residual(&mut probe, mb_x, mb_y, &header)?;
                    DecodedBSliceMacroblock::Intra(DecodedIntraMacroblock {
                        macroblock: IntraMacroblock::Predicted(header),
                        residual: Some(residual),
                    })
                }
                BSliceMacroblock::Intra(IntraMacroblock::Pcm(pcm)) => {
                    self.record_pcm_macroblock(mb_x, mb_y)?;
                    DecodedBSliceMacroblock::Intra(DecodedIntraMacroblock {
                        macroblock: IntraMacroblock::Pcm(pcm),
                        residual: None,
                    })
                }
            })
        })();
        match result {
            Ok(decoded) => {
                *reader = probe;
                Ok(decoded)
            }
            Err(error) => {
                self.restore_macroblock(mb_x, mb_y, snapshot);
                Err(error)
            }
        }
    }

    #[inline]
    pub fn luma_context(&self, mb_x: u32, mb_y: u32, block_index: u8) -> Result<CoeffTokenContext> {
        self.ensure_slice_started()?;
        let (x, y) = self.luma_position(mb_x, mb_y, block_index)?;
        Ok(CoeffTokenContext::NeighborTotal(self.luma.neighbour_total(
            x,
            y,
            self.slice_id,
        )))
    }

    #[inline]
    pub fn record_luma(
        &mut self,
        mb_x: u32,
        mb_y: u32,
        block_index: u8,
        total_coeff: u8,
    ) -> Result<()> {
        self.ensure_total_coeff(total_coeff)?;
        self.ensure_slice_started()?;
        let (x, y) = self.luma_position(mb_x, mb_y, block_index)?;
        self.luma.record(x, y, self.slice_id, total_coeff);
        Ok(())
    }

    /// Derives `nC`, decodes a luma 4x4/AC block, and records TotalCoeff.
    pub fn decode_luma_block(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        block_index: u8,
        max_num_coeff: u8,
    ) -> Result<ResidualBlock> {
        let context = self.luma_context(mb_x, mb_y, block_index)?;
        let block = decode_residual_block(reader, context, max_num_coeff)?;
        self.record_luma(mb_x, mb_y, block_index, block.total_coeff)?;
        Ok(block)
    }

    /// Decodes Intra16x16 DC using block zero's neighbour context.
    ///
    /// DC TotalCoeff is intentionally not recorded: neighbouring `nC`
    /// derivation uses only the separately decoded AC block totals.
    pub fn decode_intra16x16_dc(
        &self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
    ) -> Result<ResidualBlock> {
        let context = self.luma_context(mb_x, mb_y, 0)?;
        decode_residual_block(reader, context, 16)
    }

    /// Decodes all residual blocks of one predicted 4:2:0 intra macroblock.
    ///
    /// The bit reader and every neighbour-state entry touched by the
    /// macroblock are committed together. Both are restored on failure.
    pub fn decode_intra_residual(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        header: &IntraMacroblockHeader,
    ) -> Result<IntraResidual> {
        self.ensure_slice_started()?;
        if header.coded_block_pattern.luma > 15 || header.coded_block_pattern.chroma > 2 {
            return Err(H264Error::InvalidSyntax(
                "coded block pattern exceeds 4:2:0 macroblock bounds",
            ));
        }
        let snapshot = self.snapshot_macroblock(mb_x, mb_y)?;
        let mut probe = *reader;
        match self.decode_intra_residual_inner(&mut probe, mb_x, mb_y, header) {
            Ok(residual) => {
                *reader = probe;
                Ok(residual)
            }
            Err(error) => {
                self.restore_macroblock(mb_x, mb_y, snapshot);
                Err(error)
            }
        }
    }

    /// Decodes all residual blocks of one frame-coded 4:2:0 inter
    /// macroblock, committing reader position and neighbour totals together.
    pub fn decode_inter_residual(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        header: &PInterMacroblockHeader,
    ) -> Result<InterResidual> {
        self.decode_inter_residual_pattern(reader, mb_x, mb_y, header.coded_block_pattern)
    }

    fn decode_inter_residual_pattern(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        coded_block_pattern: CodedBlockPattern,
    ) -> Result<InterResidual> {
        self.ensure_slice_started()?;
        if coded_block_pattern.luma > 15 || coded_block_pattern.chroma > 2 {
            return Err(H264Error::InvalidSyntax(
                "coded block pattern exceeds 4:2:0 macroblock bounds",
            ));
        }
        let snapshot = self.snapshot_macroblock(mb_x, mb_y)?;
        let mut probe = *reader;
        match self.decode_inter_residual_inner(&mut probe, mb_x, mb_y, coded_block_pattern) {
            Ok(residual) => {
                *reader = probe;
                Ok(residual)
            }
            Err(error) => {
                self.restore_macroblock(mb_x, mb_y, snapshot);
                Err(error)
            }
        }
    }

    #[inline]
    pub fn chroma_cb_context(
        &self,
        mb_x: u32,
        mb_y: u32,
        block_index: u8,
    ) -> Result<CoeffTokenContext> {
        self.chroma_context(&self.chroma_cb, mb_x, mb_y, block_index)
    }

    #[inline]
    pub fn chroma_cr_context(
        &self,
        mb_x: u32,
        mb_y: u32,
        block_index: u8,
    ) -> Result<CoeffTokenContext> {
        self.chroma_context(&self.chroma_cr, mb_x, mb_y, block_index)
    }

    #[inline]
    pub fn record_chroma_cb(
        &mut self,
        mb_x: u32,
        mb_y: u32,
        block_index: u8,
        total_coeff: u8,
    ) -> Result<()> {
        self.record_chroma(true, mb_x, mb_y, block_index, total_coeff)
    }

    #[inline]
    pub fn record_chroma_cr(
        &mut self,
        mb_x: u32,
        mb_y: u32,
        block_index: u8,
        total_coeff: u8,
    ) -> Result<()> {
        self.record_chroma(false, mb_x, mb_y, block_index, total_coeff)
    }

    /// Derives `nC`, decodes a Cb AC block, and records TotalCoeff.
    pub fn decode_chroma_cb_ac(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        block_index: u8,
    ) -> Result<ResidualBlock> {
        let context = self.chroma_cb_context(mb_x, mb_y, block_index)?;
        let block = decode_residual_block(reader, context, 15)?;
        self.record_chroma_cb(mb_x, mb_y, block_index, block.total_coeff)?;
        Ok(block)
    }

    /// Derives `nC`, decodes a Cr AC block, and records TotalCoeff.
    pub fn decode_chroma_cr_ac(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        block_index: u8,
    ) -> Result<ResidualBlock> {
        let context = self.chroma_cr_context(mb_x, mb_y, block_index)?;
        let block = decode_residual_block(reader, context, 15)?;
        self.record_chroma_cr(mb_x, mb_y, block_index, block.total_coeff)?;
        Ok(block)
    }

    /// Records the inferred TotalCoeff of every 4x4 block in a skipped or
    /// otherwise coefficient-free macroblock.
    pub fn record_zero_macroblock(&mut self, mb_x: u32, mb_y: u32) -> Result<()> {
        self.record_uniform_macroblock(mb_x, mb_y, 0)
    }

    /// I_PCM neighbouring blocks contribute nA/nB equal to 16.
    pub fn record_pcm_macroblock(&mut self, mb_x: u32, mb_y: u32) -> Result<()> {
        self.record_uniform_macroblock(mb_x, mb_y, 16)
    }

    fn decode_intra_residual_inner(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        header: &IntraMacroblockHeader,
    ) -> Result<IntraResidual> {
        let intra_16x16 = matches!(
            header.luma_prediction,
            IntraLumaPrediction::SixteenBySixteen { .. }
        );
        let luma_max_num_coeff = if intra_16x16 { 15 } else { 16 };
        let luma_dc = if intra_16x16 {
            Some(self.decode_intra16x16_dc(reader, mb_x, mb_y)?)
        } else {
            None
        };

        let mut luma = [ResidualBlock::empty(16); 16];
        for block_index in 0..16u8 {
            let region_8x8 = block_index / 4;
            if header.coded_block_pattern.luma & (1 << region_8x8) != 0 {
                luma[usize::from(block_index)] =
                    self.decode_luma_block(reader, mb_x, mb_y, block_index, luma_max_num_coeff)?;
            } else {
                luma[usize::from(block_index)] = ResidualBlock::empty(luma_max_num_coeff);
                self.record_luma(mb_x, mb_y, block_index, 0)?;
            }
        }

        let mut chroma_dc = [ResidualBlock::empty(4); 2];
        if header.coded_block_pattern.chroma != 0 {
            for block in &mut chroma_dc {
                *block = decode_residual_block(reader, CoeffTokenContext::ChromaDc420, 4)?;
            }
        }

        let mut chroma_ac = [[ResidualBlock::empty(15); 4]; 2];
        if header.coded_block_pattern.chroma == 2 {
            for block_index in 0..4u8 {
                chroma_ac[0][usize::from(block_index)] =
                    self.decode_chroma_cb_ac(reader, mb_x, mb_y, block_index)?;
            }
            for block_index in 0..4u8 {
                chroma_ac[1][usize::from(block_index)] =
                    self.decode_chroma_cr_ac(reader, mb_x, mb_y, block_index)?;
            }
        } else {
            for block_index in 0..4u8 {
                self.record_chroma_cb(mb_x, mb_y, block_index, 0)?;
                self.record_chroma_cr(mb_x, mb_y, block_index, 0)?;
            }
        }

        Ok(IntraResidual {
            luma_dc,
            luma,
            chroma_dc,
            chroma_ac,
        })
    }

    fn decode_inter_residual_inner(
        &mut self,
        reader: &mut BitReader<'_>,
        mb_x: u32,
        mb_y: u32,
        coded_block_pattern: CodedBlockPattern,
    ) -> Result<InterResidual> {
        let mut luma = [ResidualBlock::empty(16); 16];
        for block_index in 0..16u8 {
            let region_8x8 = block_index / 4;
            if coded_block_pattern.luma & (1 << region_8x8) != 0 {
                luma[usize::from(block_index)] =
                    self.decode_luma_block(reader, mb_x, mb_y, block_index, 16)?;
            } else {
                self.record_luma(mb_x, mb_y, block_index, 0)?;
            }
        }

        let mut chroma_dc = [ResidualBlock::empty(4); 2];
        if coded_block_pattern.chroma != 0 {
            for block in &mut chroma_dc {
                *block = decode_residual_block(reader, CoeffTokenContext::ChromaDc420, 4)?;
            }
        }

        let mut chroma_ac = [[ResidualBlock::empty(15); 4]; 2];
        if coded_block_pattern.chroma == 2 {
            for block_index in 0..4u8 {
                chroma_ac[0][usize::from(block_index)] =
                    self.decode_chroma_cb_ac(reader, mb_x, mb_y, block_index)?;
            }
            for block_index in 0..4u8 {
                chroma_ac[1][usize::from(block_index)] =
                    self.decode_chroma_cr_ac(reader, mb_x, mb_y, block_index)?;
            }
        } else {
            for block_index in 0..4u8 {
                self.record_chroma_cb(mb_x, mb_y, block_index, 0)?;
                self.record_chroma_cr(mb_x, mb_y, block_index, 0)?;
            }
        }

        Ok(InterResidual {
            luma,
            chroma_dc,
            chroma_ac,
        })
    }

    fn chroma_context(
        &self,
        grid: &BlockGrid,
        mb_x: u32,
        mb_y: u32,
        block_index: u8,
    ) -> Result<CoeffTokenContext> {
        self.ensure_slice_started()?;
        let (x, y) = self.chroma_position(mb_x, mb_y, block_index)?;
        Ok(CoeffTokenContext::NeighborTotal(grid.neighbour_total(
            x,
            y,
            self.slice_id,
        )))
    }

    fn record_chroma(
        &mut self,
        cb: bool,
        mb_x: u32,
        mb_y: u32,
        block_index: u8,
        total_coeff: u8,
    ) -> Result<()> {
        self.ensure_total_coeff(total_coeff)?;
        self.ensure_slice_started()?;
        let (x, y) = self.chroma_position(mb_x, mb_y, block_index)?;
        let grid = if cb {
            &mut self.chroma_cb
        } else {
            &mut self.chroma_cr
        };
        grid.record(x, y, self.slice_id, total_coeff);
        Ok(())
    }

    fn record_uniform_macroblock(&mut self, mb_x: u32, mb_y: u32, total_coeff: u8) -> Result<()> {
        self.ensure_total_coeff(total_coeff)?;
        self.ensure_slice_started()?;
        self.ensure_macroblock(mb_x, mb_y)?;
        for block_index in 0..16 {
            let (x, y) = self.luma_position(mb_x, mb_y, block_index)?;
            self.luma.record(x, y, self.slice_id, total_coeff);
        }
        for block_index in 0..4 {
            let (x, y) = self.chroma_position(mb_x, mb_y, block_index)?;
            self.chroma_cb.record(x, y, self.slice_id, total_coeff);
            self.chroma_cr.record(x, y, self.slice_id, total_coeff);
        }
        Ok(())
    }

    pub(crate) fn snapshot_macroblock(&self, mb_x: u32, mb_y: u32) -> Result<MacroblockSnapshot> {
        self.ensure_macroblock(mb_x, mb_y)?;
        let mut snapshot = MacroblockSnapshot {
            luma: [0; 16],
            chroma_cb: [0; 4],
            chroma_cr: [0; 4],
        };
        for block_index in 0..16u8 {
            let (x, y) = self.luma_position(mb_x, mb_y, block_index)?;
            snapshot.luma[usize::from(block_index)] = self.luma.entries[y * self.luma.width + x];
        }
        for block_index in 0..4u8 {
            let (x, y) = self.chroma_position(mb_x, mb_y, block_index)?;
            let index = usize::from(block_index);
            snapshot.chroma_cb[index] = self.chroma_cb.entries[y * self.chroma_cb.width + x];
            snapshot.chroma_cr[index] = self.chroma_cr.entries[y * self.chroma_cr.width + x];
        }
        Ok(snapshot)
    }

    pub(crate) fn restore_macroblock(
        &mut self,
        mb_x: u32,
        mb_y: u32,
        snapshot: MacroblockSnapshot,
    ) {
        for block_index in 0..16u8 {
            let (x, y) = self
                .luma_position(mb_x, mb_y, block_index)
                .expect("snapshot coordinates were already validated");
            self.luma.entries[y * self.luma.width + x] = snapshot.luma[usize::from(block_index)];
        }
        for block_index in 0..4u8 {
            let (x, y) = self
                .chroma_position(mb_x, mb_y, block_index)
                .expect("snapshot coordinates were already validated");
            let index = usize::from(block_index);
            self.chroma_cb.entries[y * self.chroma_cb.width + x] = snapshot.chroma_cb[index];
            self.chroma_cr.entries[y * self.chroma_cr.width + x] = snapshot.chroma_cr[index];
        }
    }

    #[inline]
    fn luma_position(&self, mb_x: u32, mb_y: u32, block_index: u8) -> Result<(usize, usize)> {
        self.ensure_macroblock(mb_x, mb_y)?;
        let (block_x, block_y) = *LUMA_BLOCK_COORDINATES
            .get(usize::from(block_index))
            .ok_or(H264Error::InvalidSyntax("luma4x4BlkIdx exceeds 15"))?;
        Ok((
            usize::try_from(mb_x).map_err(|_| H264Error::IntegerOverflow)? * 4 + block_x,
            usize::try_from(mb_y).map_err(|_| H264Error::IntegerOverflow)? * 4 + block_y,
        ))
    }

    #[inline]
    fn chroma_position(&self, mb_x: u32, mb_y: u32, block_index: u8) -> Result<(usize, usize)> {
        self.ensure_macroblock(mb_x, mb_y)?;
        let (block_x, block_y) = *CHROMA_BLOCK_COORDINATES
            .get(usize::from(block_index))
            .ok_or(H264Error::InvalidSyntax("chroma4x4BlkIdx exceeds 3"))?;
        Ok((
            usize::try_from(mb_x).map_err(|_| H264Error::IntegerOverflow)? * 2 + block_x,
            usize::try_from(mb_y).map_err(|_| H264Error::IntegerOverflow)? * 2 + block_y,
        ))
    }

    #[inline]
    fn ensure_macroblock(&self, mb_x: u32, mb_y: u32) -> Result<()> {
        let mb_x = usize::try_from(mb_x).map_err(|_| H264Error::IntegerOverflow)?;
        let mb_y = usize::try_from(mb_y).map_err(|_| H264Error::IntegerOverflow)?;
        if mb_x >= self.width_in_mbs || mb_y >= self.height_in_mbs {
            return Err(H264Error::InvalidSyntax(
                "macroblock coordinates exceed the CAVLC context picture",
            ));
        }
        Ok(())
    }

    #[inline]
    fn ensure_total_coeff(&self, total_coeff: u8) -> Result<()> {
        if total_coeff > 16 {
            return Err(H264Error::InvalidSyntax("TotalCoeff exceeds 16"));
        }
        Ok(())
    }

    #[inline]
    fn ensure_slice_started(&self) -> Result<()> {
        if self.slice_id == 0 {
            return Err(H264Error::InvalidSyntax(
                "begin_slice must be called before deriving CAVLC contexts",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use bit_readers::BitReader;

    use super::CavlcNeighborState;
    use crate::{
        BMacroblockContext, CodedBlockPattern, CoeffTokenContext, DecodedBSliceMacroblock,
        DecodedPSliceMacroblock, H264Error, IntraLumaPrediction, IntraMacroblock,
        IntraMacroblockHeader, IntraPredictionModeSyntax, PInterMacroblockHeader,
        PMacroblockContext, PPartitionMode, PPartitionMotion,
    };

    #[test]
    fn derives_none_one_and_two_neighbour_contexts_in_scan_order() {
        let mut state = CavlcNeighborState::new(1, 1).unwrap();
        state.begin_slice();

        assert_eq!(
            state.luma_context(0, 0, 0),
            Ok(CoeffTokenContext::NeighborTotal(0))
        );
        state.record_luma(0, 0, 0, 2).unwrap();
        assert_eq!(
            state.luma_context(0, 0, 1),
            Ok(CoeffTokenContext::NeighborTotal(2))
        );
        state.record_luma(0, 0, 1, 4).unwrap();
        assert_eq!(
            state.luma_context(0, 0, 2),
            Ok(CoeffTokenContext::NeighborTotal(2))
        );
        state.record_luma(0, 0, 2, 6).unwrap();
        assert_eq!(
            state.luma_context(0, 0, 3),
            Ok(CoeffTokenContext::NeighborTotal(5))
        );
    }

    #[test]
    fn derives_macroblock_edges_and_hides_other_slices() {
        let mut state = CavlcNeighborState::new(2, 2).unwrap();
        state.begin_slice();
        state.record_zero_macroblock(0, 0).unwrap();
        state.record_luma(0, 0, 5, 8).unwrap();
        assert_eq!(
            state.luma_context(1, 0, 0),
            Ok(CoeffTokenContext::NeighborTotal(8))
        );

        state.begin_slice();
        assert_eq!(
            state.luma_context(1, 0, 0),
            Ok(CoeffTokenContext::NeighborTotal(0))
        );
        state.record_zero_macroblock(0, 1).unwrap();
        state.record_luma(0, 1, 5, 6).unwrap();
        assert_eq!(
            state.luma_context(1, 1, 0),
            Ok(CoeffTokenContext::NeighborTotal(6))
        );
    }

    #[test]
    fn keeps_cb_and_cr_contexts_independent() {
        let mut state = CavlcNeighborState::new(1, 1).unwrap();
        state.begin_slice();
        state.record_chroma_cb(0, 0, 0, 2).unwrap();
        state.record_chroma_cr(0, 0, 0, 6).unwrap();
        assert_eq!(
            state.chroma_cb_context(0, 0, 1),
            Ok(CoeffTokenContext::NeighborTotal(2))
        );
        assert_eq!(
            state.chroma_cr_context(0, 0, 1),
            Ok(CoeffTokenContext::NeighborTotal(6))
        );
    }

    #[test]
    fn records_zero_and_pcm_macroblocks_with_distinct_availability() {
        let mut state = CavlcNeighborState::new(3, 1).unwrap();
        state.begin_slice();
        state.record_zero_macroblock(0, 0).unwrap();
        state.record_luma(1, 0, 0, 6).unwrap();
        assert_eq!(
            state.luma_context(1, 0, 2),
            // The decoded zero block on the left is available, so nC is
            // rounded from (0 + 6) / 2 rather than copied from the top.
            Ok(CoeffTokenContext::NeighborTotal(3))
        );

        state.record_pcm_macroblock(1, 0).unwrap();
        assert_eq!(
            state.luma_context(2, 0, 0),
            Ok(CoeffTokenContext::NeighborTotal(16))
        );
        assert_eq!(
            state.chroma_cb_context(2, 0, 0),
            Ok(CoeffTokenContext::NeighborTotal(16))
        );
    }

    #[test]
    fn validates_lifecycle_dimensions_and_indices() {
        assert!(matches!(
            CavlcNeighborState::new(0, 1),
            Err(H264Error::InvalidSyntax(_))
        ));
        let mut state = CavlcNeighborState::new(1, 1).unwrap();
        assert!(matches!(
            state.luma_context(0, 0, 0),
            Err(H264Error::InvalidSyntax(_))
        ));
        state.begin_slice();
        assert!(matches!(
            state.record_luma(0, 0, 16, 0),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert!(matches!(
            state.record_luma(0, 0, 0, 17),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert!(matches!(
            state.chroma_cb_context(1, 0, 0),
            Err(H264Error::InvalidSyntax(_))
        ));
    }

    #[test]
    fn decodes_and_records_blocks_as_one_state_transition() {
        // First block: TotalCoeff=2, two positive trailing ones, no zeros.
        // Second block: TotalCoeff=0 using the nC=2 coeff_token table.
        let data = bit_string("0010011111");
        let mut reader = BitReader::new(&data);
        let mut state = CavlcNeighborState::new(1, 1).unwrap();
        state.begin_slice();

        let first = state.decode_luma_block(&mut reader, 0, 0, 0, 16).unwrap();
        assert_eq!(first.total_coeff, 2);
        assert_eq!(&first.coefficients[..2], &[1, 1]);
        let second = state.decode_luma_block(&mut reader, 0, 0, 1, 16).unwrap();
        assert_eq!(second.total_coeff, 0);
        assert_eq!(reader.bit_position(), 10);

        let mut truncated = BitReader::new(&[0]);
        let mut clean_state = CavlcNeighborState::new(1, 1).unwrap();
        clean_state.begin_slice();
        assert!(
            clean_state
                .decode_luma_block(&mut truncated, 0, 0, 0, 16)
                .is_err()
        );
        assert_eq!(truncated.bit_position(), 0);
        assert_eq!(
            clean_state.luma_context(0, 0, 1),
            Ok(CoeffTokenContext::NeighborTotal(0))
        );
    }

    #[test]
    fn decodes_intra_macroblock_residual_layouts() {
        let mut state = CavlcNeighborState::new(3, 1).unwrap();
        state.begin_slice();

        // One coded 8x8 luma region contains four empty 4x4 blocks.
        let data = bit_string("1111");
        let mut reader = BitReader::new(&data);
        let residual = state
            .decode_intra_residual(&mut reader, 0, 0, &header_4x4(1, 0))
            .unwrap();
        assert!(residual.luma_dc.is_none());
        assert!(residual.luma.iter().all(|block| block.total_coeff == 0));
        assert_eq!(reader.bit_position(), 4);

        // Intra16x16 always carries a separately decoded DC block.
        let data = bit_string("1");
        let mut reader = BitReader::new(&data);
        let residual = state
            .decode_intra_residual(&mut reader, 1, 0, &header_16x16(0, 0))
            .unwrap();
        assert_eq!(residual.luma_dc.unwrap().total_coeff, 0);
        assert!(residual.luma.iter().all(|block| block.max_num_coeff == 15));
        assert_eq!(reader.bit_position(), 1);

        // Chroma pattern one carries Cb and Cr DC but no chroma AC.
        let data = bit_string("0101");
        let mut reader = BitReader::new(&data);
        let residual = state
            .decode_intra_residual(&mut reader, 2, 0, &header_4x4(0, 1))
            .unwrap();
        assert!(
            residual
                .chroma_dc
                .iter()
                .all(|block| block.total_coeff == 0)
        );
        assert!(
            residual
                .chroma_ac
                .iter()
                .flatten()
                .all(|block| block.total_coeff == 0)
        );
        assert_eq!(reader.bit_position(), 4);
    }

    #[test]
    fn rolls_back_reader_and_neighbours_on_macroblock_failure() {
        let mut state = CavlcNeighborState::new(1, 1).unwrap();
        state.begin_slice();
        state.record_luma(0, 0, 0, 6).unwrap();

        // The first block succeeds and overwrites block zero, then the second
        // coeff_token runs out of input.
        let mut reader = BitReader::new(&[0b1000_0000]);
        assert!(
            state
                .decode_intra_residual(&mut reader, 0, 0, &header_4x4(1, 0))
                .is_err()
        );
        assert_eq!(reader.bit_position(), 0);
        assert_eq!(
            state.luma_context(0, 0, 1),
            Ok(CoeffTokenContext::NeighborTotal(6))
        );
    }

    #[test]
    fn decodes_inter_residual_and_rolls_back_atomically() {
        let mut state = CavlcNeighborState::new(2, 1).unwrap();
        state.begin_slice();
        let data = bit_string("1111");
        let mut reader = BitReader::new(&data);
        let residual = state
            .decode_inter_residual(&mut reader, 0, 0, &inter_header(1, 0, false))
            .unwrap();
        assert!(residual.luma.iter().all(|block| block.total_coeff == 0));
        assert!(residual.luma.iter().all(|block| block.max_num_coeff == 16));
        assert_eq!(reader.bit_position(), 4);

        state.record_luma(1, 0, 0, 6).unwrap();
        let mut truncated = BitReader::new(&[0b1000_0000]);
        assert!(
            state
                .decode_inter_residual(&mut truncated, 1, 0, &inter_header(1, 0, false))
                .is_err()
        );
        assert_eq!(truncated.bit_position(), 0);
        assert_eq!(
            state.luma_context(1, 0, 1),
            Ok(CoeffTokenContext::NeighborTotal(6))
        );
    }

    #[test]
    fn parses_and_decodes_complete_intra_macroblocks_transactionally() {
        let bits = format!("1{}100100", "1".repeat(16));
        let data = bit_string(&bits);
        let mut reader = BitReader::new(&data);
        let mut state = CavlcNeighborState::new(2, 1).unwrap();
        state.begin_slice();
        let decoded = state
            .decode_intra_macroblock(&mut reader, 0, 0, false)
            .unwrap();
        let IntraMacroblock::Predicted(header) = decoded.macroblock else {
            panic!("expected predicted macroblock");
        };
        let residual = decoded.residual.unwrap();
        assert!(matches!(
            header.luma_prediction,
            IntraLumaPrediction::FourByFour(_)
        ));
        assert!(!header.has_residual());
        assert!(residual.luma.iter().all(|block| block.total_coeff == 0));
        assert_eq!(reader.bit_position(), bits.len());

        // mb_type=1, chroma mode=0, qp_delta=0, and an empty luma DC block.
        let data = bit_string("010111");
        let mut reader = BitReader::new(&data);
        let decoded = state
            .decode_intra_macroblock(&mut reader, 1, 0, false)
            .unwrap();
        let IntraMacroblock::Predicted(_) = decoded.macroblock else {
            panic!("expected predicted macroblock");
        };
        let residual = decoded.residual.unwrap();
        assert_eq!(residual.luma_dc.unwrap().total_coeff, 0);
        assert_eq!(reader.bit_position(), 6);
    }

    #[test]
    fn complete_macroblock_failure_restores_header_bits_and_state() {
        let mut state = CavlcNeighborState::new(1, 1).unwrap();
        state.begin_slice();
        state.record_luma(0, 0, 0, 6).unwrap();

        // A valid Intra16x16 header is followed by a truncated DC coeff_token.
        let mut reader = BitReader::new(&[0b0101_1000]);
        assert!(
            state
                .decode_intra_macroblock(&mut reader, 0, 0, false)
                .is_err()
        );
        assert_eq!(reader.bit_position(), 0);
        assert_eq!(
            state.luma_context(0, 0, 1),
            Ok(CoeffTokenContext::NeighborTotal(6))
        );
    }

    #[test]
    fn decodes_complete_inter_and_embedded_intra_p_macroblocks() {
        let mut state = CavlcNeighborState::new(2, 1).unwrap();
        state.begin_slice();
        let context = PMacroblockContext {
            num_ref_idx_l0_active: 1,
            transform_8x8_mode_enabled: false,
        };

        // P_L0_16x16, zero MVD, coded_block_pattern zero.
        let data = bit_string("1111");
        let mut reader = BitReader::new(&data);
        let decoded = state
            .decode_p_macroblock(&mut reader, 0, 0, context)
            .unwrap();
        let DecodedPSliceMacroblock::Inter { header, residual } = decoded else {
            panic!("expected inter P macroblock");
        };
        assert!(!header.coded_block_pattern.has_residual());
        assert!(residual.luma.iter().all(|block| block.total_coeff == 0));
        assert_eq!(reader.bit_position(), 4);

        // P mb_type 5 maps to I_NxN, followed by predicted Intra4x4 modes.
        let bits = format!("00110{}100100", "1".repeat(16));
        let data = bit_string(&bits);
        let mut reader = BitReader::new(&data);
        let decoded = state
            .decode_p_macroblock(&mut reader, 1, 0, context)
            .unwrap();
        let DecodedPSliceMacroblock::Intra(decoded) = decoded else {
            panic!("expected embedded intra P macroblock");
        };
        assert!(matches!(
            decoded.macroblock,
            IntraMacroblock::Predicted(IntraMacroblockHeader {
                luma_prediction: IntraLumaPrediction::FourByFour(_),
                ..
            })
        ));
        assert_eq!(reader.bit_position(), bits.len());
    }

    #[test]
    fn complete_p_macroblock_failure_restores_bits_and_neighbours() {
        let mut state = CavlcNeighborState::new(1, 1).unwrap();
        state.begin_slice();
        state.record_luma(0, 0, 0, 6).unwrap();
        let mut reader = BitReader::new(&[0b1110_0000]);
        assert!(
            state
                .decode_p_macroblock(
                    &mut reader,
                    0,
                    0,
                    PMacroblockContext {
                        num_ref_idx_l0_active: 1,
                        transform_8x8_mode_enabled: false,
                    }
                )
                .is_err()
        );
        assert_eq!(reader.bit_position(), 0);
        assert_eq!(
            state.luma_context(0, 0, 1),
            Ok(CoeffTokenContext::NeighborTotal(6))
        );
    }

    #[test]
    fn decodes_complete_inter_and_embedded_intra_b_macroblocks() {
        let mut state = CavlcNeighborState::new(2, 1).unwrap();
        state.begin_slice();

        // B_L0_16x16, zero MVD, and coded_block_pattern zero.
        let bits = "010111";
        let data = bit_string(bits);
        let mut reader = BitReader::new(&data);
        let decoded = state
            .decode_b_macroblock(&mut reader, 0, 0, b_context())
            .unwrap();
        let DecodedBSliceMacroblock::Inter { header, residual } = decoded else {
            panic!("expected B inter macroblock");
        };
        assert_eq!(
            header.coded_block_pattern,
            CodedBlockPattern { luma: 0, chroma: 0 }
        );
        assert!(residual.luma.iter().all(|block| block.total_coeff == 0));
        assert_eq!(reader.bit_position(), bits.len());

        // B mb_type 23 maps to I_NxN with predicted modes and no residual.
        let bits = format!("000011000{}100100", "1".repeat(16));
        let data = bit_string(&bits);
        let mut reader = BitReader::new(&data);
        let decoded = state
            .decode_b_macroblock(&mut reader, 1, 0, b_context())
            .unwrap();
        assert!(matches!(
            decoded,
            DecodedBSliceMacroblock::Intra(crate::DecodedIntraMacroblock {
                macroblock: IntraMacroblock::Predicted(IntraMacroblockHeader {
                    luma_prediction: IntraLumaPrediction::FourByFour(_),
                    ..
                }),
                ..
            })
        ));
        assert_eq!(reader.bit_position(), bits.len());
    }

    #[test]
    fn complete_b_macroblock_failure_restores_bits_and_neighbours() {
        let mut state = CavlcNeighborState::new(1, 1).unwrap();
        state.begin_slice();
        state.record_luma(0, 0, 0, 6).unwrap();

        let mut reader = BitReader::new(&[0]);
        assert!(
            state
                .decode_b_macroblock(&mut reader, 0, 0, b_context())
                .is_err()
        );
        assert_eq!(reader.bit_position(), 0);
        assert_eq!(
            state.luma_context(0, 0, 1),
            Ok(CoeffTokenContext::NeighborTotal(6))
        );
    }

    fn header_4x4(luma: u8, chroma: u8) -> IntraMacroblockHeader {
        IntraMacroblockHeader {
            luma_prediction: IntraLumaPrediction::FourByFour(
                [IntraPredictionModeSyntax {
                    use_predicted: true,
                    remaining_mode: None,
                }; 16],
            ),
            chroma_prediction_mode: 0,
            coded_block_pattern: CodedBlockPattern { luma, chroma },
            qp_delta: 0,
        }
    }

    fn header_16x16(luma: u8, chroma: u8) -> IntraMacroblockHeader {
        IntraMacroblockHeader {
            luma_prediction: IntraLumaPrediction::SixteenBySixteen { mode: 0 },
            chroma_prediction_mode: 0,
            coded_block_pattern: CodedBlockPattern { luma, chroma },
            qp_delta: 0,
        }
    }

    fn inter_header(luma: u8, chroma: u8, transform_size_8x8: bool) -> PInterMacroblockHeader {
        PInterMacroblockHeader {
            partition_mode: PPartitionMode::L0_16x16,
            partitions: vec![PPartitionMotion {
                reference_index: 0,
                differences: Vec::new().into(),
            }]
            .into(),
            coded_block_pattern: CodedBlockPattern { luma, chroma },
            transform_size_8x8,
            qp_delta: 0,
        }
    }

    fn b_context() -> BMacroblockContext {
        BMacroblockContext {
            num_ref_idx_l0_active: 1,
            num_ref_idx_l1_active: 1,
            transform_8x8_mode_enabled: false,
            direct_8x8_inference: true,
        }
    }

    fn bit_string(bits: &str) -> Vec<u8> {
        let mut bytes = vec![0; bits.len().div_ceil(8)];
        for (index, bit) in bits.bytes().enumerate() {
            if bit == b'1' {
                bytes[index / 8] |= 1 << (7 - index % 8);
            }
        }
        bytes
    }
}
