//! Neighbouring-block state used to derive the CAVLC `nC` context.

use bit_readers::BitReader;

use crate::{CoeffTokenContext, H264Error, ResidualBlock, Result, decode_residual_block};

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
    use crate::{CoeffTokenContext, H264Error};

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
