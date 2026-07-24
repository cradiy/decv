//! Binary arithmetic decoding primitives used by H.264 CABAC slice data.

use bit_readers::BitReader;

use crate::cabac_context_tables::{CABAC_CONTEXT_COUNT, CABAC_INIT_I, CABAC_INIT_PB};
use crate::{H264Error, PcmMacroblock, Result, SliceType};

const INITIAL_RANGE: u16 = 510;

// H.264 Table 9-44, indexed by pStateIdx and qCodIRangeIdx.
const RANGE_TAB_LPS: [[u8; 4]; 64] = [
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

const fn packed_transitions(lps: bool) -> [u8; 128] {
    let mut transitions = [0; 128];
    let mut packed = 0;
    while packed < transitions.len() {
        let probability_state = packed >> 1;
        let mut most_probable_symbol = packed & 1;
        let next_probability_state = if lps {
            if probability_state == 0 {
                most_probable_symbol ^= 1;
            }
            TRANS_IDX_LPS[probability_state]
        } else {
            TRANS_IDX_MPS[probability_state]
        };
        transitions[packed] = (next_probability_state << 1) | most_probable_symbol as u8;
        packed += 1;
    }
    transitions
}

const TRANS_STATE_MPS: [u8; 128] = packed_transitions(false);
const TRANS_STATE_LPS: [u8; 128] = packed_transitions(true);

/// One adaptive CABAC probability state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CabacContextState {
    /// `(pStateIdx << 1) | valMPS`.
    packed: u8,
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
            packed: (probability_state << 1) | most_probable_symbol,
        })
    }

    #[inline]
    pub const fn probability_state(self) -> u8 {
        self.packed >> 1
    }

    #[inline]
    pub const fn most_probable_symbol(self) -> u8 {
        self.packed & 1
    }

    #[inline]
    fn initialize(m: i8, n: i8, slice_qp_y: u8) -> Self {
        let pre_context_state = ((i32::from(m) * i32::from(slice_qp_y)) >> 4) + i32::from(n);
        let pre_context_state = pre_context_state.clamp(1, 126) as u8;
        if pre_context_state <= 63 {
            Self {
                packed: (63 - pre_context_state) << 1,
            }
        } else {
            Self {
                packed: ((pre_context_state - 64) << 1) | 1,
            }
        }
    }
}

/// Selects one of the normative CABAC initialization parameter tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CabacInitializationTable {
    Intra,
    Inter0,
    Inter1,
    Inter2,
}

impl CabacInitializationTable {
    pub fn for_slice(slice_type: SliceType, cabac_init_idc: Option<u8>) -> Result<Self> {
        if slice_type.is_intra() {
            if cabac_init_idc.is_some() {
                return Err(H264Error::InvalidSyntax(
                    "intra CABAC slice unexpectedly carries cabac_init_idc",
                ));
            }
            return Ok(Self::Intra);
        }
        match cabac_init_idc {
            Some(0) => Ok(Self::Inter0),
            Some(1) => Ok(Self::Inter1),
            Some(2) => Ok(Self::Inter2),
            Some(_) => Err(H264Error::InvalidSyntax("cabac_init_idc exceeds 2")),
            None => Err(H264Error::InvalidSyntax(
                "inter CABAC slice is missing cabac_init_idc",
            )),
        }
    }
}

/// The 460 CABAC context models used by 8-bit 4:2:0 H.264 syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CabacContextSet {
    states: [CabacContextState; CABAC_CONTEXT_COUNT],
}

impl CabacContextSet {
    pub fn new(table: CabacInitializationTable, slice_qp_y: u8) -> Result<Self> {
        if slice_qp_y > 51 {
            return Err(H264Error::InvalidSyntax(
                "CABAC slice QP exceeds the 8-bit range",
            ));
        }
        let parameters = match table {
            CabacInitializationTable::Intra => &CABAC_INIT_I,
            CabacInitializationTable::Inter0 => &CABAC_INIT_PB[0],
            CabacInitializationTable::Inter1 => &CABAC_INIT_PB[1],
            CabacInitializationTable::Inter2 => &CABAC_INIT_PB[2],
        };
        Ok(Self {
            states: std::array::from_fn(|index| {
                let (m, n) = parameters[index];
                CabacContextState::initialize(m, n, slice_qp_y)
            }),
        })
    }

    #[inline]
    pub fn get(&self, context_index: usize) -> Result<CabacContextState> {
        self.states
            .get(context_index)
            .copied()
            .ok_or(H264Error::InvalidSyntax(
                "CABAC context index exceeds the 8-bit 4:2:0 set",
            ))
    }

    #[inline]
    pub fn get_mut(&mut self, context_index: usize) -> Result<&mut CabacContextState> {
        self.states
            .get_mut(context_index)
            .ok_or(H264Error::InvalidSyntax(
                "CABAC context index exceeds the 8-bit 4:2:0 set",
            ))
    }

    /// Returns a context selected by codec-internal, spec-bounded syntax.
    ///
    /// # Safety
    ///
    /// `context_index` must be smaller than the normative 8-bit 4:2:0 context
    /// count. Public callers must use [`Self::get_mut`] instead.
    #[inline]
    pub(crate) unsafe fn get_mut_unchecked(
        &mut self,
        context_index: usize,
    ) -> &mut CabacContextState {
        debug_assert!(context_index < CABAC_CONTEXT_COUNT);
        // SAFETY: The caller guarantees the documented context-index bound.
        unsafe { self.states.get_unchecked_mut(context_index) }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.states.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.states.is_empty()
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
#[derive(Debug, Clone)]
pub struct CabacDecoder<'data> {
    reader: BitReader<'data>,
    range: u16,
    offset: u16,
    terminated: bool,
}

impl<'data> CabacDecoder<'data> {
    /// Initializes `codIRange` and the nine-bit `codIOffset`.
    ///
    /// The input must already be byte-aligned after
    /// [`consume_cabac_alignment`].
    pub fn new(mut reader: BitReader<'data>) -> Result<Self> {
        if reader.bit_offset() != 0 {
            return Err(H264Error::InvalidSyntax(
                "CABAC arithmetic data is not byte-aligned",
            ));
        }
        let mut probe = reader;
        let offset = probe
            .read_bits_const::<9>()
            .ok_or(H264Error::UnexpectedEof)? as u16;
        if offset >= INITIAL_RANGE {
            return Err(H264Error::InvalidSyntax(
                "CABAC initial offset is not smaller than the initial range",
            ));
        }
        reader = probe;
        Ok(Self {
            reader,
            range: INITIAL_RANGE,
            offset,
            terminated: false,
        })
    }

    /// Decodes one context-modelled decision bin.
    ///
    /// Reader, arithmetic state, and probability state are committed together
    /// only after any required renormalization bits are available.
    #[inline]
    pub fn decode_decision(&mut self, context: &mut CabacContextState) -> Result<u8> {
        let mut range = self.range;
        let mut offset = self.offset;
        let packed_state = context.packed;
        debug_assert!(packed_state < 128);

        let range_index = usize::from((range >> 6) & 3);
        let probability_state = usize::from(packed_state >> 1);
        // SAFETY: CabacContextState constructors and transition tables keep
        // pStateIdx below 64, while the range-derived index is in 0..4.
        let lps_range = u16::from(unsafe {
            *RANGE_TAB_LPS
                .get_unchecked(probability_state)
                .get_unchecked(range_index)
        });
        range -= lps_range;

        let (bin, next_state) = if offset < range {
            // SAFETY: A packed CABAC context is always in 0..128.
            let next_state = unsafe { *TRANS_STATE_MPS.get_unchecked(usize::from(packed_state)) };
            (packed_state & 1, next_state)
        } else {
            offset -= range;
            range = lps_range;
            // SAFETY: A packed CABAC context is always in 0..128.
            let next_state = unsafe { *TRANS_STATE_LPS.get_unchecked(usize::from(packed_state)) };
            ((packed_state & 1) ^ 1, next_state)
        };

        // BitReader reads are failure-atomic, so this can operate directly on
        // the decoder reader without snapshotting and copying its 40 bytes.
        renormalize(&mut self.reader, &mut range, &mut offset)?;
        self.range = range;
        self.offset = offset;
        context.packed = next_state;
        Ok(bin)
    }

    /// Decodes one equiprobable bypass bin without changing a context model.
    #[inline]
    pub fn decode_bypass(&mut self) -> Result<u8> {
        let bit = self.reader.read_bit().ok_or(H264Error::UnexpectedEof)?;
        // Keep the arithmetic state local until the bin is known. Besides
        // preserving the EOF-before-commit rule, this lets optimized builds
        // lower the unpredictable comparison to a conditional selection.
        let range = self.range;
        let mut offset = (self.offset << 1) | u16::from(bit);
        let bin = if offset < range {
            0
        } else {
            offset -= range;
            1
        };
        self.offset = offset;
        Ok(bin)
    }

    /// Decodes `end_of_slice_flag` using the fixed termination probability.
    #[inline]
    pub fn decode_terminate(&mut self) -> Result<u8> {
        let mut range = self.range - 2;
        let mut offset = self.offset;
        if offset >= range {
            self.range = range;
            self.terminated = true;
            return Ok(1);
        }

        // BitReader reads are failure-atomic; arithmetic fields are committed
        // only after renormalization succeeds.
        renormalize(&mut self.reader, &mut range, &mut offset)?;
        self.range = range;
        self.offset = offset;
        Ok(0)
    }

    /// Leaves arithmetic coding for one 8-bit 4:2:0 I_PCM macroblock, reads
    /// its raw samples, and restarts the arithmetic engine after the samples.
    ///
    /// CABAC keeps nine lookahead bits in `codIOffset`. Rewinding those bits
    /// recovers the syntax position at which `pcm_alignment_zero_bit` begins.
    pub fn decode_pcm_420_8bit_and_restart(&mut self) -> Result<PcmMacroblock> {
        if !self.terminated {
            return Err(H264Error::InvalidSyntax(
                "CABAC I_PCM transition requires a terminating bin",
            ));
        }
        let mut reader = self
            .reader
            .rewound_by(9)
            .ok_or(H264Error::IntegerOverflow)?;
        while reader.bit_offset() != 0 {
            if reader.read_bit().ok_or(H264Error::UnexpectedEof)? != 0 {
                return Err(H264Error::InvalidSyntax(
                    "CABAC pcm_alignment_zero_bit is not zero",
                ));
            }
        }

        let mut luma = Box::new([0; 256]);
        for sample in luma.iter_mut() {
            *sample = reader
                .read_bits_const::<8>()
                .ok_or(H264Error::UnexpectedEof)? as u8;
        }
        let mut chroma = Box::new([0; 128]);
        for sample in chroma.iter_mut() {
            *sample = reader
                .read_bits_const::<8>()
                .ok_or(H264Error::UnexpectedEof)? as u8;
        }
        let restarted = Self::new(reader)?;
        *self = restarted;
        Ok(PcmMacroblock { luma, chroma })
    }

    #[inline]
    pub const fn range(&self) -> u16 {
        self.range
    }

    #[inline]
    pub const fn offset(&self) -> u16 {
        self.offset
    }

    #[inline]
    pub fn bit_position(&self) -> usize {
        self.reader.bit_position()
    }

    #[inline]
    pub const fn reader(&self) -> &BitReader<'data> {
        &self.reader
    }

    #[inline]
    pub fn into_reader(self) -> BitReader<'data> {
        self.reader
    }
}

#[inline]
fn renormalize(reader: &mut BitReader<'_>, range: &mut u16, offset: &mut u16) -> Result<()> {
    if *range < 256 {
        let shift = range.leading_zeros() - 7;
        let bits = reader.read_bits(shift).ok_or(H264Error::UnexpectedEof)? as u16;
        *range <<= shift;
        *offset = (*offset << shift) | bits;
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
        let reader = BitReader::new(&[0b0011_0010, 0b1000_0000]);
        let decoder = CabacDecoder::new(reader).unwrap();
        assert_eq!(decoder.range(), 510);
        assert_eq!(decoder.offset(), 101);
        assert_eq!(decoder.bit_position(), 9);
    }

    #[test]
    fn rejects_invalid_initial_offsets_without_consuming_input() {
        for bytes in [[0xff, 0x00], [0xff, 0x80]] {
            let reader = BitReader::new(&bytes);
            assert!(matches!(
                CabacDecoder::new(reader),
                Err(H264Error::InvalidSyntax(_))
            ));
            assert_eq!(reader.bit_position(), 0);
        }
    }

    #[test]
    fn decodes_mps_and_lps_with_normative_state_transitions() {
        let mps_reader = BitReader::new(&[0, 0]);
        let mut mps_decoder = CabacDecoder::new(mps_reader).unwrap();
        let mut mps_context = CabacContextState::new(0, 0).unwrap();
        assert_eq!(mps_decoder.decode_decision(&mut mps_context).unwrap(), 0);
        assert_eq!(mps_context, CabacContextState::new(1, 0).unwrap());
        assert_eq!((mps_decoder.range(), mps_decoder.offset()), (270, 0));

        // Initial offset 400 followed by a one renormalization bit.
        let lps_reader = BitReader::new(&[0b1100_1000, 0b0100_0000]);
        let mut lps_decoder = CabacDecoder::new(lps_reader).unwrap();
        let mut lps_context = CabacContextState::new(0, 0).unwrap();
        assert_eq!(lps_decoder.decode_decision(&mut lps_context).unwrap(), 1);
        assert_eq!(lps_context, CabacContextState::new(0, 1).unwrap());
        assert_eq!((lps_decoder.range(), lps_decoder.offset()), (480, 261));
    }

    #[test]
    fn decodes_bypass_and_termination_bins() {
        let bypass_reader = BitReader::new(&[0b0011_0010, 0b1100_0000]);
        let mut bypass = CabacDecoder::new(bypass_reader).unwrap();
        assert_eq!(bypass.decode_bypass().unwrap(), 0);
        assert_eq!(bypass.offset(), 203);

        let zero_reader = BitReader::new(&[0, 0]);
        let mut zero = CabacDecoder::new(zero_reader).unwrap();
        assert_eq!(zero.decode_terminate().unwrap(), 0);
        assert_eq!((zero.range(), zero.offset()), (508, 0));

        let one_reader = BitReader::new(&[0b1111_1110, 0b1000_0000]);
        let mut one = CabacDecoder::new(one_reader).unwrap();
        assert_eq!(one.decode_terminate().unwrap(), 1);
        assert_eq!((one.range(), one.offset()), (508, 509));
    }

    #[test]
    fn reads_pcm_samples_from_before_the_nine_bit_lookahead_and_restarts() {
        let mut data = vec![0; 386];
        data[0] = 0xfe;
        for (index, byte) in data[..384].iter_mut().enumerate().skip(2) {
            *byte = index as u8;
        }
        let mut decoder = CabacDecoder::new(BitReader::new(&data)).unwrap();
        assert_eq!(decoder.decode_terminate().unwrap(), 1);

        let pcm = decoder.decode_pcm_420_8bit_and_restart().unwrap();

        assert_eq!(pcm.luma[0], 0xfe);
        assert_eq!(pcm.luma[1], 0);
        assert_eq!(pcm.luma[255], 255);
        assert_eq!(pcm.chroma[0], 0);
        assert_eq!(pcm.chroma[127], 127);
        assert_eq!(decoder.range(), 510);
        assert_eq!(decoder.offset(), 0);
        assert_eq!(decoder.bit_position(), 384 * 8 + 9);
    }

    #[test]
    fn decision_failure_is_atomic() {
        let reader = BitReader::new(&[0, 0]);
        let mut decoder = CabacDecoder::new(reader).unwrap();
        let mut context = CabacContextState::new(0, 0).unwrap();

        // Repeated MPS decisions eventually exhaust the seven bits left after
        // the nine-bit initial offset. The failing decision must not commit.
        loop {
            let range = decoder.range();
            let offset = decoder.offset();
            let bit_position = decoder.bit_position();
            let state = context;
            match decoder.decode_decision(&mut context) {
                Ok(_) => {}
                Err(H264Error::UnexpectedEof) => {
                    assert_eq!((decoder.range(), decoder.offset()), (range, offset));
                    assert_eq!(decoder.bit_position(), bit_position);
                    assert_eq!(context, state);
                    break;
                }
                Err(error) => panic!("unexpected CABAC error: {error}"),
            }
        }
    }

    #[test]
    fn termination_failure_is_atomic() {
        let mut decoder = CabacDecoder::new(BitReader::new(&[0, 0])).unwrap();
        loop {
            let range = decoder.range();
            let offset = decoder.offset();
            let bit_position = decoder.bit_position();
            match decoder.decode_terminate() {
                Ok(0) => {}
                Ok(1) => panic!("zero input unexpectedly terminated CABAC"),
                Ok(_) => unreachable!(),
                Err(H264Error::UnexpectedEof) => {
                    assert_eq!((decoder.range(), decoder.offset()), (range, offset));
                    assert_eq!(decoder.bit_position(), bit_position);
                    break;
                }
                Err(error) => panic!("unexpected CABAC error: {error}"),
            }
        }
    }

    #[test]
    fn initializes_normative_intra_contexts_at_qp_boundaries() {
        let qp0 = CabacContextSet::new(CabacInitializationTable::Intra, 0).unwrap();
        assert_eq!(qp0.get(0).unwrap(), CabacContextState::new(62, 0).unwrap());
        assert_eq!(qp0.get(1).unwrap(), CabacContextState::new(9, 0).unwrap());
        assert_eq!(qp0.get(2).unwrap(), CabacContextState::new(10, 1).unwrap());

        let qp51 = CabacContextSet::new(CabacInitializationTable::Intra, 51).unwrap();
        assert_eq!(qp51.get(0).unwrap(), CabacContextState::new(15, 0).unwrap());
        assert_eq!(
            qp51.get(459).unwrap(),
            CabacContextState::new(47, 1).unwrap()
        );
    }

    #[test]
    fn selects_each_inter_initialization_table() {
        let expected = [
            (CabacInitializationTable::Inter0, (6, 1), (62, 1)),
            (CabacInitializationTable::Inter1, (3, 0), (62, 1)),
            (CabacInitializationTable::Inter2, (0, 0), (32, 1)),
        ];
        for (table, first, last) in expected {
            let contexts = CabacContextSet::new(table, 26).unwrap();
            assert_eq!(
                contexts.get(11).unwrap(),
                CabacContextState::new(first.0, first.1).unwrap()
            );
            assert_eq!(
                contexts.get(459).unwrap(),
                CabacContextState::new(last.0, last.1).unwrap()
            );
        }
    }

    #[test]
    fn validates_context_initialization_inputs_and_indices() {
        assert!(matches!(
            CabacContextSet::new(CabacInitializationTable::Intra, 52),
            Err(H264Error::InvalidSyntax(_))
        ));
        let mut contexts = CabacContextSet::new(CabacInitializationTable::Intra, 26).unwrap();
        assert_eq!(contexts.len(), 460);
        assert!(!contexts.is_empty());
        assert!(contexts.get(460).is_err());
        assert!(contexts.get_mut(460).is_err());
    }

    #[test]
    fn packs_every_valid_context_state_into_one_byte() {
        assert_eq!(std::mem::size_of::<CabacContextState>(), 1);
        for probability_state in 0..64 {
            for most_probable_symbol in 0..=1 {
                let state =
                    CabacContextState::new(probability_state, most_probable_symbol).unwrap();
                assert_eq!(state.probability_state(), probability_state);
                assert_eq!(state.most_probable_symbol(), most_probable_symbol);
                assert_eq!(
                    TRANS_STATE_MPS[usize::from(state.packed)],
                    (TRANS_IDX_MPS[usize::from(probability_state)] << 1) | most_probable_symbol
                );
                assert_eq!(
                    TRANS_STATE_LPS[usize::from(state.packed)],
                    (TRANS_IDX_LPS[usize::from(probability_state)] << 1)
                        | (most_probable_symbol ^ u8::from(probability_state == 0))
                );
            }
        }
    }

    #[test]
    fn maps_slice_headers_to_initialization_tables() {
        assert_eq!(
            CabacInitializationTable::for_slice(SliceType::I, None).unwrap(),
            CabacInitializationTable::Intra
        );
        assert_eq!(
            CabacInitializationTable::for_slice(SliceType::B, Some(2)).unwrap(),
            CabacInitializationTable::Inter2
        );
        assert!(CabacInitializationTable::for_slice(SliceType::P, None).is_err());
        assert!(CabacInitializationTable::for_slice(SliceType::I, Some(0)).is_err());
    }

    #[test]
    fn all_context_tables_match_the_normative_fixed_vector() {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for table in [
            CabacInitializationTable::Intra,
            CabacInitializationTable::Inter0,
            CabacInitializationTable::Inter1,
            CabacInitializationTable::Inter2,
        ] {
            for qp in [0, 26, 51] {
                let contexts = CabacContextSet::new(table, qp).unwrap();
                for state in contexts.states {
                    hash ^=
                        u64::from((state.probability_state() << 1) | state.most_probable_symbol());
                    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        assert_eq!(hash, 0xfc67_0f17_aec4_54d8);
    }
}
