//! Binary arithmetic decoding primitives used by H.264 CABAC slice data.

use bit_readers::BitReader;

use crate::{H264Error, Result};

const INITIAL_RANGE: u16 = 510;

// H.264 Table 9-44, indexed by pStateIdx and qCodIRangeIdx.
const RANGE_TAB_LPS: [[u16; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [95, 116, 137, 158],
    [90, 110, 130, 150],
    [85, 104, 123, 142],
    [81, 99, 117, 135],
    [77, 94, 111, 128],
    [73, 89, 105, 122],
    [69, 85, 100, 116],
    [66, 80, 95, 110],
    [62, 76, 90, 104],
    [59, 72, 86, 99],
    [56, 69, 81, 94],
    [53, 65, 77, 89],
    [51, 62, 73, 85],
    [48, 59, 69, 80],
    [46, 56, 66, 76],
    [43, 53, 63, 72],
    [41, 50, 59, 69],
    [39, 48, 56, 65],
    [37, 45, 54, 62],
    [35, 43, 51, 59],
    [33, 41, 48, 56],
    [32, 39, 46, 53],
    [30, 37, 43, 50],
    [29, 35, 41, 48],
    [27, 33, 39, 45],
    [26, 31, 37, 43],
    [24, 30, 35, 41],
    [23, 28, 33, 39],
    [22, 27, 32, 37],
    [21, 26, 30, 35],
    [20, 24, 29, 33],
    [19, 23, 27, 31],
    [18, 22, 26, 30],
    [17, 21, 25, 28],
    [16, 20, 23, 27],
    [15, 19, 22, 25],
    [14, 18, 21, 24],
    [14, 17, 20, 23],
    [13, 16, 19, 22],
    [12, 15, 18, 21],
    [12, 14, 17, 20],
    [11, 14, 16, 19],
    [11, 13, 15, 18],
    [10, 12, 15, 17],
    [10, 12, 14, 16],
    [9, 11, 13, 15],
    [9, 11, 12, 14],
    [8, 10, 12, 14],
    [8, 9, 11, 13],
    [7, 9, 11, 12],
    [7, 9, 10, 12],
    [7, 8, 10, 11],
    [6, 8, 9, 11],
    [6, 7, 9, 10],
    [6, 7, 8, 9],
    [2, 2, 2, 2],
];

// H.264 Tables 9-45 and 9-46.
const TRANS_IDX_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];
const TRANS_IDX_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21,
    21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34,
    34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// One adaptive CABAC probability state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CabacContextState {
    probability_state: u8,
    most_probable_symbol: u8,
}

impl CabacContextState {
    pub fn new(probability_state: u8, most_probable_symbol: u8) -> Result<Self> {
        if probability_state >= 64 {
            return Err(H264Error::InvalidSyntax(
                "CABAC probability state exceeds 63",
            ));
        }
        if most_probable_symbol > 1 {
            return Err(H264Error::InvalidSyntax(
                "CABAC most-probable symbol is not a bit",
            ));
        }
        Ok(Self {
            probability_state,
            most_probable_symbol,
        })
    }

    #[inline]
    pub const fn probability_state(self) -> u8 {
        self.probability_state
    }

    #[inline]
    pub const fn most_probable_symbol(self) -> u8 {
        self.most_probable_symbol
    }
}

/// Consumes the all-one padding inserted before CABAC slice data.
///
/// The operation is atomic: malformed or truncated alignment leaves `reader`
/// at its original position.
pub fn consume_cabac_alignment(reader: &mut BitReader<'_>) -> Result<()> {
    let mut probe = *reader;
    while probe.bit_offset() != 0 {
        if probe.read_bit().ok_or(H264Error::UnexpectedEof)? != 1 {
            return Err(H264Error::InvalidSyntax(
                "CABAC alignment contains a zero bit",
            ));
        }
    }
    *reader = probe;
    Ok(())
}

/// H.264 CABAC binary arithmetic decoder.
///
/// This layer deliberately knows nothing about H.264 syntax-element context
/// selection. Callers supply the context state for decision bins and use the
/// separate bypass and terminate operations where the syntax requires them.
#[derive(Debug)]
pub struct CabacDecoder<'reader, 'data> {
    reader: &'reader mut BitReader<'data>,
    range: u16,
    offset: u16,
}

impl<'reader, 'data> CabacDecoder<'reader, 'data> {
    /// Initializes `codIRange` and the nine-bit `codIOffset`.
    ///
    /// The input must already be byte-aligned after
    /// [`consume_cabac_alignment`].
    pub fn new(reader: &'reader mut BitReader<'data>) -> Result<Self> {
        if reader.bit_offset() != 0 {
            return Err(H264Error::InvalidSyntax(
                "CABAC arithmetic data is not byte-aligned",
            ));
        }
        let mut probe = *reader;
        let offset = probe
            .read_bits_const::<9>()
            .ok_or(H264Error::UnexpectedEof)? as u16;
        if offset >= INITIAL_RANGE {
            return Err(H264Error::InvalidSyntax(
                "CABAC initial offset is not smaller than the initial range",
            ));
        }
        *reader = probe;
        Ok(Self {
            reader,
            range: INITIAL_RANGE,
            offset,
        })
    }

    /// Decodes one context-modelled decision bin.
    ///
    /// Reader, arithmetic state, and probability state are committed together
    /// only after any required renormalization bits are available.
    #[inline]
    pub fn decode_decision(&mut self, context: &mut CabacContextState) -> Result<u8> {
        let mut reader = *self.reader;
        let mut range = self.range;
        let mut offset = self.offset;
        let mut next_context = *context;

        let range_index = usize::from((range >> 6) & 3);
        let lps_range = RANGE_TAB_LPS[usize::from(next_context.probability_state)][range_index];
        range -= lps_range;

        let bin = if offset < range {
            next_context.probability_state =
                TRANS_IDX_MPS[usize::from(next_context.probability_state)];
            next_context.most_probable_symbol
        } else {
            offset -= range;
            range = lps_range;
            let bin = 1 - next_context.most_probable_symbol;
            if next_context.probability_state == 0 {
                next_context.most_probable_symbol ^= 1;
            }
            next_context.probability_state =
                TRANS_IDX_LPS[usize::from(next_context.probability_state)];
            bin
        };

        renormalize(&mut reader, &mut range, &mut offset)?;
        *self.reader = reader;
        self.range = range;
        self.offset = offset;
        *context = next_context;
        Ok(bin)
    }

    /// Decodes one equiprobable bypass bin without changing a context model.
    #[inline]
    pub fn decode_bypass(&mut self) -> Result<u8> {
        let bit = self.reader.read_bit().ok_or(H264Error::UnexpectedEof)?;
        self.offset = (self.offset << 1) | u16::from(bit);
        if self.offset < self.range {
            Ok(0)
        } else {
            self.offset -= self.range;
            Ok(1)
        }
    }

    /// Decodes `end_of_slice_flag` using the fixed termination probability.
    #[inline]
    pub fn decode_terminate(&mut self) -> Result<u8> {
        let mut reader = *self.reader;
        let mut range = self.range - 2;
        let mut offset = self.offset;
        if offset >= range {
            self.range = range;
            return Ok(1);
        }

        renormalize(&mut reader, &mut range, &mut offset)?;
        *self.reader = reader;
        self.range = range;
        self.offset = offset;
        Ok(0)
    }

    #[inline]
    pub const fn range(&self) -> u16 {
        self.range
    }

    #[inline]
    pub const fn offset(&self) -> u16 {
        self.offset
    }
}

#[inline]
fn renormalize(reader: &mut BitReader<'_>, range: &mut u16, offset: &mut u16) -> Result<()> {
    while *range < 256 {
        *range <<= 1;
        *offset = (*offset << 1) | u16::from(reader.read_bit().ok_or(H264Error::UnexpectedEof)?);
    }
    debug_assert!((256..=510).contains(range));
    debug_assert!(*offset < *range);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_alignment_atomically() {
        let mut valid = BitReader::new(&[0b1011_1111, 0]);
        valid.skip_bits(3);
        consume_cabac_alignment(&mut valid).unwrap();
        assert_eq!(valid.bit_position(), 8);

        let mut invalid = BitReader::new(&[0b1010_1111, 0]);
        invalid.skip_bits(3);
        let original = invalid.bit_position();
        assert!(matches!(
            consume_cabac_alignment(&mut invalid),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert_eq!(invalid.bit_position(), original);
    }

    #[test]
    fn initializes_range_and_offset_from_nine_bits() {
        let mut reader = BitReader::new(&[0b0011_0010, 0b1000_0000]);
        let decoder = CabacDecoder::new(&mut reader).unwrap();
        assert_eq!(decoder.range(), 510);
        assert_eq!(decoder.offset(), 101);
        assert_eq!(reader.bit_position(), 9);
    }

    #[test]
    fn rejects_invalid_initial_offsets_without_consuming_input() {
        for bytes in [[0xff, 0x00], [0xff, 0x80]] {
            let mut reader = BitReader::new(&bytes);
            assert!(matches!(
                CabacDecoder::new(&mut reader),
                Err(H264Error::InvalidSyntax(_))
            ));
            assert_eq!(reader.bit_position(), 0);
        }
    }

    #[test]
    fn decodes_mps_and_lps_with_normative_state_transitions() {
        let mut mps_reader = BitReader::new(&[0, 0]);
        let mut mps_decoder = CabacDecoder::new(&mut mps_reader).unwrap();
        let mut mps_context = CabacContextState::new(0, 0).unwrap();
        assert_eq!(mps_decoder.decode_decision(&mut mps_context).unwrap(), 0);
        assert_eq!(mps_context, CabacContextState::new(1, 0).unwrap());
        assert_eq!((mps_decoder.range(), mps_decoder.offset()), (270, 0));

        // Initial offset 400 followed by a one renormalization bit.
        let mut lps_reader = BitReader::new(&[0b1100_1000, 0b0100_0000]);
        let mut lps_decoder = CabacDecoder::new(&mut lps_reader).unwrap();
        let mut lps_context = CabacContextState::new(0, 0).unwrap();
        assert_eq!(lps_decoder.decode_decision(&mut lps_context).unwrap(), 1);
        assert_eq!(lps_context, CabacContextState::new(0, 1).unwrap());
        assert_eq!((lps_decoder.range(), lps_decoder.offset()), (480, 261));
    }

    #[test]
    fn decodes_bypass_and_termination_bins() {
        let mut bypass_reader = BitReader::new(&[0b0011_0010, 0b1100_0000]);
        let mut bypass = CabacDecoder::new(&mut bypass_reader).unwrap();
        assert_eq!(bypass.decode_bypass().unwrap(), 0);
        assert_eq!(bypass.offset(), 203);

        let mut zero_reader = BitReader::new(&[0, 0]);
        let mut zero = CabacDecoder::new(&mut zero_reader).unwrap();
        assert_eq!(zero.decode_terminate().unwrap(), 0);
        assert_eq!((zero.range(), zero.offset()), (508, 0));

        let mut one_reader = BitReader::new(&[0b1111_1110, 0b1000_0000]);
        let mut one = CabacDecoder::new(&mut one_reader).unwrap();
        assert_eq!(one.decode_terminate().unwrap(), 1);
        assert_eq!((one.range(), one.offset()), (508, 509));
    }

    #[test]
    fn decision_failure_is_atomic() {
        let mut reader = BitReader::new(&[0, 0]);
        let mut decoder = CabacDecoder::new(&mut reader).unwrap();
        let mut context = CabacContextState::new(0, 0).unwrap();

        // Repeated MPS decisions eventually exhaust the seven bits left after
        // the nine-bit initial offset. The failing decision must not commit.
        loop {
            let range = decoder.range();
            let offset = decoder.offset();
            let state = context;
            match decoder.decode_decision(&mut context) {
                Ok(_) => {}
                Err(H264Error::UnexpectedEof) => {
                    assert_eq!((decoder.range(), decoder.offset()), (range, offset));
                    assert_eq!(context, state);
                    break;
                }
                Err(error) => panic!("unexpected CABAC error: {error}"),
            }
        }
    }
}
