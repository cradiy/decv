//! Reusable CABAC binarization readers.

use bit_readers::BitReader;

use crate::{
    BSubMacroblockType, CabacContextSet, CabacDecoder, CabacInitializationTable, CodedBlockPattern,
    H264Error, IntraLumaPrediction, IntraMacroblockHeader, IntraPredictionModeSyntax,
    PSubMacroblockType, PcmMacroblock, Result, SliceType, consume_cabac_alignment,
};

/// CABAC-decoded `mb_type` for a P slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CabacPMacroblockType {
    L0_16x16,
    L0_16x8,
    L0_8x16,
    EightByEight,
    /// Embedded I macroblock type in the ordinary I table's 0..=25 range.
    Intra(u8),
}

/// CABAC-decoded `mb_type` for a B slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CabacBMacroblockType {
    /// Index in the ordinary B inter table, in the range 0..=22.
    Inter(u8),
    /// Embedded I macroblock type in the ordinary I table's 0..=25 range.
    Intra(u8),
}

/// CABAC-decoded I-macroblock syntax before residual or I_PCM sample decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CabacIntraMacroblockSyntax {
    Predicted(IntraMacroblockHeader),
    Pcm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CabacMacroblockSummary {
    pub skipped: bool,
    /// B_Direct or B_Skip. False for P and intra macroblocks.
    pub direct: bool,
    pub intra16_or_pcm: bool,
    pub intra_chroma_prediction: Option<u8>,
    pub coded_block_pattern: CodedBlockPattern,
    pub transform_size_8x8: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompletedCabacMacroblock {
    slice_id: u32,
    summary: CabacMacroblockSummary,
}

/// Completed-neighbour state used for macroblock-level CABAC contexts.
#[derive(Debug, Clone)]
pub struct CabacMacroblockState {
    width_in_macroblocks: usize,
    height_in_macroblocks: usize,
    first_slice_id: u32,
    completed: Vec<Option<CompletedCabacMacroblock>>,
}

impl CabacMacroblockState {
    pub fn new(width_in_macroblocks: usize, height_in_macroblocks: usize) -> Result<Self> {
        if width_in_macroblocks == 0 || height_in_macroblocks == 0 {
            return Err(H264Error::InvalidSyntax(
                "CABAC macroblock-state dimensions must be non-zero",
            ));
        }
        let macroblock_count = width_in_macroblocks
            .checked_mul(height_in_macroblocks)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width_in_macroblocks,
            height_in_macroblocks,
            first_slice_id: 0,
            completed: vec![None; macroblock_count],
        })
    }

    pub(crate) fn reset_for_picture(
        &mut self,
        width_in_macroblocks: usize,
        height_in_macroblocks: usize,
        first_slice_id: u32,
        clear_entries: bool,
    ) -> Result<()> {
        if width_in_macroblocks == 0 || height_in_macroblocks == 0 {
            return Err(H264Error::InvalidSyntax(
                "CABAC macroblock-state dimensions must be non-zero",
            ));
        }
        let macroblock_count = width_in_macroblocks
            .checked_mul(height_in_macroblocks)
            .ok_or(H264Error::IntegerOverflow)?;
        self.width_in_macroblocks = width_in_macroblocks;
        self.height_in_macroblocks = height_in_macroblocks;
        self.first_slice_id = first_slice_id;
        if self.completed.len() != macroblock_count {
            self.completed = vec![None; macroblock_count];
        } else if clear_entries {
            self.completed.fill(None);
        }
        Ok(())
    }

    /// Decodes progressive-frame `mb_skip_flag` for a P/SP or B slice.
    pub fn decode_skip_flag(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        slice_type: SliceType,
    ) -> Result<bool> {
        let context_index =
            self.skip_flag_context_index(macroblock_address, slice_id, slice_type)?;
        Ok(syntax.decision_known(context_index)? != 0)
    }

    /// Decodes an I/SI-slice `mb_type` value in the range 0..=25.
    pub fn decode_intra_macroblock_type(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        slice_type: SliceType,
    ) -> Result<u8> {
        if !slice_type.is_intra() {
            return Err(H264Error::InvalidSyntax(
                "intra CABAC mb_type requested for an inter slice",
            ));
        }
        let mut context_index = 3;
        for neighbour in self.left_and_top(macroblock_address, slice_id)? {
            if neighbour.is_some_and(|macroblock| macroblock.summary.intra16_or_pcm) {
                context_index += 1;
            }
        }
        if syntax.decision_known(context_index)? == 0 {
            return Ok(0);
        }
        if syntax.terminate()? != 0 {
            return Ok(25);
        }
        let mut macroblock_type = 1;
        macroblock_type += 12 * syntax.decision_known(6)?;
        if syntax.decision_known(7)? != 0 {
            macroblock_type += 4 + 4 * syntax.decision_known(8)?;
        }
        macroblock_type += 2 * syntax.decision_known(9)?;
        macroblock_type += syntax.decision_known(10)?;
        Ok(macroblock_type)
    }

    /// Decodes one non-skipped progressive P-slice `mb_type`.
    pub fn decode_p_macroblock_type(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
    ) -> Result<CabacPMacroblockType> {
        decode_p_macroblock_type_with(|request| match request {
            CabacBinRequest::Decision(context_index) => syntax.decision_known(context_index),
            CabacBinRequest::Terminate => syntax.terminate(),
        })
    }

    /// Decodes one P-slice `sub_mb_type`.
    pub fn decode_p_sub_macroblock_type(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
    ) -> Result<PSubMacroblockType> {
        decode_p_sub_macroblock_type_with(|context_index| syntax.decision_known(context_index))
    }

    /// Decodes one non-skipped progressive B-slice `mb_type`.
    pub fn decode_b_macroblock_type(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<CabacBMacroblockType> {
        let context_increment =
            self.b_macroblock_type_context_increment(macroblock_address, slice_id)?;
        decode_b_macroblock_type_with(context_increment, |request| match request {
            CabacBinRequest::Decision(context_index) => syntax.decision_known(context_index),
            CabacBinRequest::Terminate => syntax.terminate(),
        })
    }

    /// Decodes one B-slice `sub_mb_type`.
    pub fn decode_b_sub_macroblock_type(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
    ) -> Result<BSubMacroblockType> {
        decode_b_sub_macroblock_type_with(|context_index| syntax.decision_known(context_index))
    }

    /// Decodes the complete prediction/header syntax of one I macroblock.
    ///
    /// I_PCM sample extraction is deliberately left to the slice layer because
    /// it temporarily leaves arithmetic coding and then reinitializes CABAC.
    pub fn decode_intra_macroblock_syntax(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        transform_8x8_mode_enabled: bool,
        previous_qp_delta_nonzero: bool,
    ) -> Result<CabacIntraMacroblockSyntax> {
        let macroblock_type =
            self.decode_intra_macroblock_type(syntax, macroblock_address, slice_id, SliceType::I)?;
        self.decode_intra_macroblock_syntax_for_type(
            syntax,
            macroblock_address,
            slice_id,
            macroblock_type,
            transform_8x8_mode_enabled,
            previous_qp_delta_nonzero,
        )
    }

    /// Decodes the prediction/header syntax following an already decoded
    /// intra `mb_type`. Inter slices use this for their embedded I macroblocks.
    pub(crate) fn decode_intra_macroblock_syntax_for_type(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        macroblock_type: u8,
        transform_8x8_mode_enabled: bool,
        previous_qp_delta_nonzero: bool,
    ) -> Result<CabacIntraMacroblockSyntax> {
        match macroblock_type {
            0 => {
                let transform_size_8x8 = transform_8x8_mode_enabled
                    && self.decode_transform_size_8x8_flag(syntax, macroblock_address, slice_id)?;
                let luma_prediction = if transform_size_8x8 {
                    let mut modes = [empty_intra_prediction_mode(); 4];
                    for mode in &mut modes {
                        *mode = syntax.intra_prediction_mode()?;
                    }
                    IntraLumaPrediction::EightByEight(modes)
                } else {
                    let mut modes = [empty_intra_prediction_mode(); 16];
                    for mode in &mut modes {
                        *mode = syntax.intra_prediction_mode()?;
                    }
                    IntraLumaPrediction::FourByFour(modes)
                };
                let chroma_prediction_mode =
                    self.decode_intra_chroma_prediction_mode(syntax, macroblock_address, slice_id)?;
                let coded_block_pattern =
                    self.decode_coded_block_pattern(syntax, macroblock_address, slice_id)?;
                let qp_delta = if coded_block_pattern.has_residual() {
                    syntax.macroblock_qp_delta(previous_qp_delta_nonzero)?
                } else {
                    0
                };
                Ok(CabacIntraMacroblockSyntax::Predicted(
                    IntraMacroblockHeader {
                        luma_prediction,
                        chroma_prediction_mode,
                        coded_block_pattern,
                        qp_delta,
                    },
                ))
            }
            1..=24 => {
                let (mode, coded_block_pattern) =
                    intra16x16_fields_from_macroblock_type(macroblock_type);
                let chroma_prediction_mode =
                    self.decode_intra_chroma_prediction_mode(syntax, macroblock_address, slice_id)?;
                let qp_delta = syntax.macroblock_qp_delta(previous_qp_delta_nonzero)?;
                Ok(CabacIntraMacroblockSyntax::Predicted(
                    IntraMacroblockHeader {
                        luma_prediction: IntraLumaPrediction::SixteenBySixteen { mode },
                        chroma_prediction_mode,
                        coded_block_pattern,
                        qp_delta,
                    },
                ))
            }
            25 => Ok(CabacIntraMacroblockSyntax::Pcm),
            _ => Err(H264Error::InvalidSyntax(
                "CABAC intra macroblock type exceeds 25",
            )),
        }
    }

    pub fn decode_intra_chroma_prediction_mode(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<u8> {
        let context_increment = self
            .left_and_top(macroblock_address, slice_id)?
            .into_iter()
            .filter(|neighbour| {
                neighbour.is_some_and(|macroblock| {
                    macroblock.summary.intra_chroma_prediction.unwrap_or(0) != 0
                })
            })
            .count();
        decode_intra_chroma_prediction_mode(64 + context_increment, |context_index| {
            syntax.decision_known(context_index)
        })
    }

    pub fn decode_coded_block_pattern(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<CodedBlockPattern> {
        let [left, top] = self.left_and_top(macroblock_address, slice_id)?;
        let left_pattern = left.map(|macroblock| macroblock.summary.coded_block_pattern);
        let top_pattern = top.map(|macroblock| macroblock.summary.coded_block_pattern);
        let luma = decode_luma_coded_block_pattern(left_pattern, top_pattern, |context_index| {
            syntax.decision_known(context_index)
        })?;
        let chroma =
            decode_chroma_coded_block_pattern(left_pattern, top_pattern, |context_index| {
                syntax.decision_known(context_index)
            })?;
        Ok(CodedBlockPattern { luma, chroma })
    }

    pub fn decode_transform_size_8x8_flag(
        &self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<bool> {
        let context_increment = self
            .left_and_top(macroblock_address, slice_id)?
            .into_iter()
            .filter(|neighbour| {
                neighbour.is_some_and(|macroblock| macroblock.summary.transform_size_8x8)
            })
            .count();
        Ok(syntax.decision_known(399 + context_increment)? != 0)
    }

    /// Records the properties needed by later macroblocks in the same slice.
    pub fn record_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        summary: CabacMacroblockSummary,
    ) -> Result<()> {
        if summary.intra_chroma_prediction.is_some_and(|mode| mode > 3)
            || summary.coded_block_pattern.luma > 15
            || summary.coded_block_pattern.chroma > 2
        {
            return Err(H264Error::InvalidSyntax(
                "CABAC macroblock summary contains an out-of-range value",
            ));
        }
        let slot = self
            .completed
            .get_mut(macroblock_address)
            .ok_or(H264Error::InvalidSyntax(
                "CABAC macroblock address exceeds the picture",
            ))?;
        if slot.is_some_and(|macroblock| macroblock.slice_id >= self.first_slice_id) {
            return Err(H264Error::InvalidSyntax(
                "CABAC macroblock was already completed",
            ));
        }
        *slot = Some(CompletedCabacMacroblock { slice_id, summary });
        Ok(())
    }

    fn skip_flag_context_index(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        slice_type: SliceType,
    ) -> Result<usize> {
        let base = match slice_type {
            SliceType::P | SliceType::Sp => 11,
            SliceType::B => 24,
            SliceType::I | SliceType::Si => {
                return Err(H264Error::InvalidSyntax(
                    "intra slices do not carry mb_skip_flag",
                ));
            }
        };
        let context_increment = self
            .left_and_top(macroblock_address, slice_id)?
            .into_iter()
            .filter(|neighbour| neighbour.is_some_and(|macroblock| !macroblock.summary.skipped))
            .count();
        Ok(base + context_increment)
    }

    fn b_macroblock_type_context_increment(
        &self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<usize> {
        Ok(self
            .left_and_top(macroblock_address, slice_id)?
            .into_iter()
            .filter(|neighbour| neighbour.is_some_and(|macroblock| !macroblock.summary.direct))
            .count())
    }

    fn left_and_top(
        &self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<[Option<CompletedCabacMacroblock>; 2]> {
        if macroblock_address >= self.width_in_macroblocks * self.height_in_macroblocks {
            return Err(H264Error::InvalidSyntax(
                "CABAC macroblock address exceeds the picture",
            ));
        }
        let left = if !macroblock_address.is_multiple_of(self.width_in_macroblocks) {
            self.completed[macroblock_address - 1]
                .filter(|macroblock| macroblock.slice_id == slice_id)
        } else {
            None
        };
        let top = if macroblock_address >= self.width_in_macroblocks {
            self.completed[macroblock_address - self.width_in_macroblocks]
                .filter(|macroblock| macroblock.slice_id == slice_id)
        } else {
            None
        };
        Ok([left, top])
    }
}

const fn empty_intra_prediction_mode() -> IntraPredictionModeSyntax {
    IntraPredictionModeSyntax {
        use_predicted: false,
        remaining_mode: None,
    }
}

const fn intra16x16_fields_from_macroblock_type(macroblock_type: u8) -> (u8, CodedBlockPattern) {
    debug_assert!(macroblock_type >= 1 && macroblock_type <= 24);
    let type_index = macroblock_type - 1;
    (
        type_index % 4,
        CodedBlockPattern {
            luma: if type_index >= 12 { 15 } else { 0 },
            chroma: (type_index / 4) % 3,
        },
    )
}

/// Long-lived arithmetic and probability state for one CABAC slice.
#[derive(Debug, Clone)]
pub struct CabacSliceDecoder<'data> {
    arithmetic: CabacDecoder<'data>,
    contexts: CabacContextSet,
}

impl<'data> CabacSliceDecoder<'data> {
    pub fn new(
        rbsp: &'data [u8],
        header_bit_size: usize,
        slice_type: SliceType,
        cabac_init_idc: Option<u8>,
        slice_qp_y: u8,
    ) -> Result<Self> {
        let mut reader = BitReader::new(rbsp);
        if !reader.skip_bits(header_bit_size) {
            return Err(H264Error::UnexpectedEof);
        }
        consume_cabac_alignment(&mut reader)?;
        let table = CabacInitializationTable::for_slice(slice_type, cabac_init_idc)?;
        Ok(Self {
            arithmetic: CabacDecoder::new(reader)?,
            contexts: CabacContextSet::new(table, slice_qp_y)?,
        })
    }

    #[inline]
    pub fn syntax(&mut self) -> CabacSyntaxDecoder<'_, 'data> {
        CabacSyntaxDecoder::new(&mut self.arithmetic, &mut self.contexts)
    }

    #[inline]
    pub fn bit_position(&self) -> usize {
        self.arithmetic.bit_position()
    }

    #[inline]
    pub const fn contexts(&self) -> &CabacContextSet {
        &self.contexts
    }

    /// Reads one raw I_PCM macroblock and restarts arithmetic decoding while
    /// preserving the adaptive probability contexts.
    pub fn decode_pcm_420_8bit_and_restart(&mut self) -> Result<PcmMacroblock> {
        self.arithmetic.decode_pcm_420_8bit_and_restart()
    }

    pub fn into_parts(self) -> (CabacDecoder<'data>, CabacContextSet) {
        (self.arithmetic, self.contexts)
    }
}

/// Couples the arithmetic engine with one slice's adaptive context models.
///
/// Higher H.264 syntax layers use this type to decode bin strings without
/// reaching into either object's storage representation.
#[derive(Debug)]
pub struct CabacSyntaxDecoder<'syntax, 'data> {
    arithmetic: &'syntax mut CabacDecoder<'data>,
    contexts: &'syntax mut CabacContextSet,
}

impl<'syntax, 'data> CabacSyntaxDecoder<'syntax, 'data> {
    pub const fn new(
        arithmetic: &'syntax mut CabacDecoder<'data>,
        contexts: &'syntax mut CabacContextSet,
    ) -> Self {
        Self {
            arithmetic,
            contexts,
        }
    }

    /// Decodes one decision bin and updates the selected context in place.
    #[inline]
    pub fn decision(&mut self, context_index: usize) -> Result<u8> {
        self.arithmetic
            .decode_decision(self.contexts.get_mut(context_index)?)
    }

    /// Fast path for context indices derived from bounded H.264 syntax.
    #[inline]
    pub(crate) fn decision_known(&mut self, context_index: usize) -> Result<u8> {
        // SAFETY: Every internal call site derives its index from normative
        // context bases plus a syntax increment whose range is validated by
        // the owning decoder state.
        let context = unsafe { self.contexts.get_mut_unchecked(context_index) };
        self.arithmetic.decode_decision(context)
    }

    /// Decodes one bypass bin.
    #[inline]
    pub fn bypass(&mut self) -> Result<u8> {
        self.arithmetic.decode_bypass()
    }

    /// Decodes one terminate bin.
    #[inline]
    pub fn terminate(&mut self) -> Result<u8> {
        self.arithmetic.decode_terminate()
    }

    pub fn intra_prediction_mode(&mut self) -> Result<IntraPredictionModeSyntax> {
        decode_intra_prediction_mode(|context_index| self.decision_known(context_index))
    }

    pub fn macroblock_qp_delta(&mut self, previous_delta_nonzero: bool) -> Result<i8> {
        decode_macroblock_qp_delta(previous_delta_nonzero, |context_index| {
            self.decision_known(context_index)
        })
    }

    /// Decodes a fixed-length bypass-coded bin string, MSB first.
    pub fn bypass_bits(&mut self, bit_count: u8) -> Result<u32> {
        decode_fixed_length(bit_count, || self.bypass())
    }

    /// Decodes truncated unary with an explicit context progression.
    ///
    /// `context_indices[0]` selects the first bin. Later entries select later
    /// bins; once the list is exhausted its final entry is repeated. A string
    /// of `maximum_value` one-bins represents the maximum without a terminating
    /// zero-bin.
    pub fn truncated_unary(
        &mut self,
        context_indices: &[usize],
        maximum_value: u32,
    ) -> Result<u32> {
        decode_truncated_unary(context_indices, maximum_value, |context_index| {
            self.decision_known(context_index)
        })
    }

    /// Decodes unary with an explicit upper bound for malformed streams.
    ///
    /// Unlike truncated unary, the maximum legal value still requires a
    /// terminating zero-bin. A longer run of one-bins is rejected.
    pub fn unary(&mut self, context_indices: &[usize], maximum_value: u32) -> Result<u32> {
        decode_unary(context_indices, maximum_value, |context_index| {
            self.decision_known(context_index)
        })
    }
}

fn decode_fixed_length(mut bit_count: u8, mut decode: impl FnMut() -> Result<u8>) -> Result<u32> {
    if bit_count > 32 {
        return Err(H264Error::InvalidSyntax(
            "CABAC fixed-length binarization exceeds 32 bits",
        ));
    }
    let mut value = 0u32;
    while bit_count != 0 {
        value = (value << 1) | u32::from(decode()?);
        bit_count -= 1;
    }
    Ok(value)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CabacBinRequest {
    Decision(usize),
    Terminate,
}

fn decode_p_macroblock_type_with(
    mut decode: impl FnMut(CabacBinRequest) -> Result<u8>,
) -> Result<CabacPMacroblockType> {
    if decode(CabacBinRequest::Decision(14))? == 0 {
        if decode(CabacBinRequest::Decision(15))? == 0 {
            return Ok(if decode(CabacBinRequest::Decision(16))? == 0 {
                CabacPMacroblockType::L0_16x16
            } else {
                CabacPMacroblockType::EightByEight
            });
        }
        return Ok(if decode(CabacBinRequest::Decision(17))? == 0 {
            CabacPMacroblockType::L0_8x16
        } else {
            CabacPMacroblockType::L0_16x8
        });
    }

    Ok(CabacPMacroblockType::Intra(
        decode_embedded_intra_macroblock_type(17, &mut decode)?,
    ))
}

fn decode_embedded_intra_macroblock_type(
    context_base: usize,
    mut decode: impl FnMut(CabacBinRequest) -> Result<u8>,
) -> Result<u8> {
    if decode(CabacBinRequest::Decision(context_base))? == 0 {
        return Ok(0);
    }
    if decode(CabacBinRequest::Terminate)? != 0 {
        return Ok(25);
    }
    let mut macroblock_type = 1;
    macroblock_type += 12 * decode(CabacBinRequest::Decision(context_base + 1))?;
    if decode(CabacBinRequest::Decision(context_base + 2))? != 0 {
        macroblock_type += 4 + 4 * decode(CabacBinRequest::Decision(context_base + 2))?;
    }
    macroblock_type += 2 * decode(CabacBinRequest::Decision(context_base + 3))?;
    macroblock_type += decode(CabacBinRequest::Decision(context_base + 3))?;
    Ok(macroblock_type)
}

fn decode_p_sub_macroblock_type_with(
    mut decision: impl FnMut(usize) -> Result<u8>,
) -> Result<PSubMacroblockType> {
    if decision(21)? != 0 {
        return Ok(PSubMacroblockType::L0_8x8);
    }
    if decision(22)? == 0 {
        return Ok(PSubMacroblockType::L0_8x4);
    }
    Ok(if decision(23)? != 0 {
        PSubMacroblockType::L0_4x8
    } else {
        PSubMacroblockType::L0_4x4
    })
}

fn decode_b_macroblock_type_with(
    context_increment: usize,
    mut decode: impl FnMut(CabacBinRequest) -> Result<u8>,
) -> Result<CabacBMacroblockType> {
    if context_increment > 2 {
        return Err(H264Error::InvalidSyntax(
            "CABAC B mb_type context increment exceeds 2",
        ));
    }
    if decode(CabacBinRequest::Decision(27 + context_increment))? == 0 {
        return Ok(CabacBMacroblockType::Inter(0));
    }
    if decode(CabacBinRequest::Decision(30))? == 0 {
        return Ok(CabacBMacroblockType::Inter(
            1 + decode(CabacBinRequest::Decision(32))?,
        ));
    }

    let mut bits = decode(CabacBinRequest::Decision(31))? << 3;
    bits |= decode(CabacBinRequest::Decision(32))? << 2;
    bits |= decode(CabacBinRequest::Decision(32))? << 1;
    bits |= decode(CabacBinRequest::Decision(32))?;
    match bits {
        0..=7 => Ok(CabacBMacroblockType::Inter(bits + 3)),
        13 => Ok(CabacBMacroblockType::Intra(
            decode_embedded_intra_macroblock_type(32, decode)?,
        )),
        14 => Ok(CabacBMacroblockType::Inter(11)),
        15 => Ok(CabacBMacroblockType::Inter(22)),
        8..=12 => Ok(CabacBMacroblockType::Inter(
            (bits << 1) + decode(CabacBinRequest::Decision(32))? - 4,
        )),
        _ => unreachable!("four decoded bins fit in 0..=15"),
    }
}

fn decode_b_sub_macroblock_type_with(
    mut decision: impl FnMut(usize) -> Result<u8>,
) -> Result<BSubMacroblockType> {
    if decision(36)? == 0 {
        return Ok(BSubMacroblockType::Direct8x8);
    }
    if decision(37)? == 0 {
        return Ok(if decision(39)? == 0 {
            BSubMacroblockType::List0_8x8
        } else {
            BSubMacroblockType::List1_8x8
        });
    }

    let index = if decision(38)? == 0 {
        3 + 2 * decision(39)? + decision(39)?
    } else if decision(39)? != 0 {
        11 + decision(39)?
    } else {
        7 + 2 * decision(39)? + decision(39)?
    };
    Ok([
        BSubMacroblockType::Direct8x8,
        BSubMacroblockType::List0_8x8,
        BSubMacroblockType::List1_8x8,
        BSubMacroblockType::Bi8x8,
        BSubMacroblockType::List0_8x4,
        BSubMacroblockType::List0_4x8,
        BSubMacroblockType::List1_8x4,
        BSubMacroblockType::List1_4x8,
        BSubMacroblockType::Bi8x4,
        BSubMacroblockType::Bi4x8,
        BSubMacroblockType::List0_4x4,
        BSubMacroblockType::List1_4x4,
        BSubMacroblockType::Bi4x4,
    ][usize::from(index)])
}

fn decode_truncated_unary(
    context_indices: &[usize],
    maximum_value: u32,
    mut decode: impl FnMut(usize) -> Result<u8>,
) -> Result<u32> {
    if maximum_value == 0 {
        return Ok(0);
    }
    validate_context_progression(context_indices)?;
    for value in 0..maximum_value {
        let context_index = progressing_context_index(context_indices, value);
        if decode(context_index)? == 0 {
            return Ok(value);
        }
    }
    Ok(maximum_value)
}

fn decode_unary(
    context_indices: &[usize],
    maximum_value: u32,
    mut decode: impl FnMut(usize) -> Result<u8>,
) -> Result<u32> {
    validate_context_progression(context_indices)?;
    for value in 0..=maximum_value {
        let context_index = progressing_context_index(context_indices, value);
        if decode(context_index)? == 0 {
            return Ok(value);
        }
    }
    Err(H264Error::InvalidSyntax(
        "CABAC unary binarization exceeds its syntax bound",
    ))
}

fn validate_context_progression(context_indices: &[usize]) -> Result<()> {
    if context_indices.is_empty() {
        return Err(H264Error::InvalidSyntax(
            "CABAC context progression is empty",
        ));
    }
    Ok(())
}

fn progressing_context_index(context_indices: &[usize], bin_index: u32) -> usize {
    context_indices[usize::try_from(bin_index)
        .unwrap_or(usize::MAX)
        .min(context_indices.len() - 1)]
}

#[cfg(test)]
fn decode_intra_macroblock_type(
    first_context_index: usize,
    mut decision: impl FnMut(usize) -> Result<u8>,
    mut terminate: impl FnMut() -> Result<u8>,
) -> Result<u8> {
    if decision(first_context_index)? == 0 {
        return Ok(0);
    }
    if terminate()? != 0 {
        return Ok(25);
    }

    let mut macroblock_type = 1;
    macroblock_type += 12 * decision(6)?;
    if decision(7)? != 0 {
        macroblock_type += 4 + 4 * decision(8)?;
    }
    macroblock_type += 2 * decision(9)?;
    macroblock_type += decision(10)?;
    Ok(macroblock_type)
}

fn decode_intra_prediction_mode(
    mut decision: impl FnMut(usize) -> Result<u8>,
) -> Result<IntraPredictionModeSyntax> {
    if decision(68)? != 0 {
        return Ok(IntraPredictionModeSyntax {
            use_predicted: true,
            remaining_mode: None,
        });
    }
    let mut remaining_mode = 0;
    for bit_index in 0..3 {
        remaining_mode |= decision(69)? << bit_index;
    }
    Ok(IntraPredictionModeSyntax {
        use_predicted: false,
        remaining_mode: Some(remaining_mode),
    })
}

fn decode_intra_chroma_prediction_mode(
    first_context_index: usize,
    mut decision: impl FnMut(usize) -> Result<u8>,
) -> Result<u8> {
    if decision(first_context_index)? == 0 {
        return Ok(0);
    }
    if decision(67)? == 0 {
        return Ok(1);
    }
    Ok(2 + decision(67)?)
}

fn decode_luma_coded_block_pattern(
    left: Option<CodedBlockPattern>,
    top: Option<CodedBlockPattern>,
    mut decision: impl FnMut(usize) -> Result<u8>,
) -> Result<u8> {
    let left = left.map_or(0x0f, |pattern| pattern.luma);
    let top = top.map_or(0x0f, |pattern| pattern.luma);
    let mut current = 0u8;

    let context = usize::from(left & 0x02 == 0) + 2 * usize::from(top & 0x04 == 0);
    current |= decision(73 + context)?;
    let context = usize::from(current & 0x01 == 0) + 2 * usize::from(top & 0x08 == 0);
    current |= decision(73 + context)? << 1;
    let context = usize::from(left & 0x08 == 0) + 2 * usize::from(current & 0x01 == 0);
    current |= decision(73 + context)? << 2;
    let context = usize::from(current & 0x04 == 0) + 2 * usize::from(current & 0x02 == 0);
    current |= decision(73 + context)? << 3;
    Ok(current)
}

fn decode_chroma_coded_block_pattern(
    left: Option<CodedBlockPattern>,
    top: Option<CodedBlockPattern>,
    mut decision: impl FnMut(usize) -> Result<u8>,
) -> Result<u8> {
    let left = left.map_or(0, |pattern| pattern.chroma);
    let top = top.map_or(0, |pattern| pattern.chroma);
    let context = usize::from(left > 0) + 2 * usize::from(top > 0);
    if decision(77 + context)? == 0 {
        return Ok(0);
    }
    let context = 4 + usize::from(left == 2) + 2 * usize::from(top == 2);
    Ok(1 + decision(77 + context)?)
}

fn decode_macroblock_qp_delta(
    previous_delta_nonzero: bool,
    mut decision: impl FnMut(usize) -> Result<u8>,
) -> Result<i8> {
    if decision(60 + usize::from(previous_delta_nonzero))? == 0 {
        return Ok(0);
    }
    let mut code = 1u16;
    let mut context_index = 62;
    while decision(context_index)? != 0 {
        context_index = 63;
        code += 1;
        if code > 102 {
            return Err(H264Error::InvalidSyntax(
                "CABAC mb_qp_delta exceeds the 8-bit QP range",
            ));
        }
    }
    let magnitude = ((code + 1) >> 1) as i16;
    let delta = if code & 1 != 0 { magnitude } else { -magnitude };
    i8::try_from(delta).map_err(|_| H264Error::IntegerOverflow)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    fn summary(
        skipped: bool,
        intra16_or_pcm: bool,
        intra_chroma_prediction: Option<u8>,
        luma: u8,
        chroma: u8,
    ) -> CabacMacroblockSummary {
        CabacMacroblockSummary {
            skipped,
            direct: false,
            intra16_or_pcm,
            intra_chroma_prediction,
            coded_block_pattern: CodedBlockPattern { luma, chroma },
            transform_size_8x8: false,
        }
    }

    #[test]
    fn decodes_fixed_length_bits_most_significant_first() {
        let mut bins = VecDeque::from([1, 0, 1, 1]);
        assert_eq!(
            decode_fixed_length(4, || Ok(bins.pop_front().unwrap())).unwrap(),
            0b1011
        );
        assert!(decode_fixed_length(33, || Ok(0)).is_err());
    }

    #[test]
    fn truncated_unary_progresses_then_repeats_the_last_context() {
        let mut bins = VecDeque::from([1, 1, 1, 0]);
        let mut visited = Vec::new();
        let value = decode_truncated_unary(&[10, 11, 12], 5, |context_index| {
            visited.push(context_index);
            Ok(bins.pop_front().unwrap())
        })
        .unwrap();
        assert_eq!(value, 3);
        assert_eq!(visited, [10, 11, 12, 12]);
    }

    #[test]
    fn truncated_unary_omits_the_terminal_zero_at_the_maximum() {
        let mut bins = VecDeque::from([1, 1, 1]);
        assert_eq!(
            decode_truncated_unary(&[20], 3, |_| Ok(bins.pop_front().unwrap())).unwrap(),
            3
        );
        assert!(bins.is_empty());
        assert_eq!(
            decode_truncated_unary(&[], 0, |_| unreachable!()).unwrap(),
            0
        );
    }

    #[test]
    fn unary_requires_a_terminal_zero_and_enforces_its_bound() {
        let mut valid = VecDeque::from([1, 1, 0]);
        assert_eq!(
            decode_unary(&[30, 31], 2, |_| Ok(valid.pop_front().unwrap())).unwrap(),
            2
        );

        let mut invalid = VecDeque::from([1, 1, 1]);
        assert!(matches!(
            decode_unary(&[30], 2, |_| Ok(invalid.pop_front().unwrap())),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert!(decode_unary(&[], 2, |_| Ok(0)).is_err());
    }

    #[test]
    fn initializes_a_slice_session_after_header_alignment() {
        let rbsp = [0b1011_1111, 0b0011_0010, 0b1000_0000];
        let slice = CabacSliceDecoder::new(&rbsp, 3, SliceType::I, None, 0).unwrap();
        assert_eq!(slice.bit_position(), 17);
        assert_eq!(
            slice.contexts().get(0).unwrap(),
            crate::CabacContextState::new(62, 0).unwrap()
        );
        let (arithmetic, contexts) = slice.into_parts();
        assert_eq!(arithmetic.offset(), 101);
        assert_eq!(contexts.len(), 460);
    }

    #[test]
    fn rejects_malformed_slice_alignment_and_header_bounds() {
        let malformed = [0b1010_1111, 0, 0];
        assert!(matches!(
            CabacSliceDecoder::new(&malformed, 3, SliceType::I, None, 26),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert!(matches!(
            CabacSliceDecoder::new(&[0], 9, SliceType::I, None, 26),
            Err(H264Error::UnexpectedEof)
        ));
    }

    #[test]
    fn derives_skip_contexts_from_completed_same_slice_neighbours() {
        let mut state = CabacMacroblockState::new(2, 2).unwrap();
        assert_eq!(
            state.skip_flag_context_index(0, 1, SliceType::P).unwrap(),
            11
        );
        state
            .record_macroblock(0, 1, summary(false, false, None, 0, 0))
            .unwrap();
        assert_eq!(
            state.skip_flag_context_index(1, 1, SliceType::P).unwrap(),
            12
        );
        state
            .record_macroblock(1, 1, summary(true, false, None, 0, 0))
            .unwrap();
        assert_eq!(
            state.skip_flag_context_index(2, 1, SliceType::P).unwrap(),
            12
        );
        assert_eq!(
            state.skip_flag_context_index(2, 2, SliceType::P).unwrap(),
            11
        );
        assert_eq!(
            state.skip_flag_context_index(2, 1, SliceType::B).unwrap(),
            25
        );
        assert!(state.skip_flag_context_index(2, 1, SliceType::I).is_err());
        assert!(
            state
                .record_macroblock(0, 1, summary(false, false, None, 0, 0))
                .is_err()
        );
        assert!(state.skip_flag_context_index(4, 1, SliceType::P).is_err());
    }

    #[test]
    fn reuses_macroblock_storage_across_picture_generations() {
        let mut state = CabacMacroblockState::new(2, 1).unwrap();
        let storage = state.completed.as_ptr();
        state
            .record_macroblock(0, 3, summary(false, false, None, 0, 0))
            .unwrap();

        state.reset_for_picture(2, 1, 4, false).unwrap();
        assert_eq!(state.completed.as_ptr(), storage);
        assert_eq!(state.left_and_top(1, 4).unwrap(), [None, None]);
        state
            .record_macroblock(0, 4, summary(true, false, None, 0, 0))
            .unwrap();
        assert!(
            state
                .record_macroblock(0, 5, summary(false, false, None, 0, 0))
                .is_err()
        );

        state.reset_for_picture(2, 1, 6, false).unwrap();
        state
            .record_macroblock(0, 6, summary(false, false, None, 0, 0))
            .unwrap();
        state.reset_for_picture(2, 1, 1, true).unwrap();
        assert_eq!(state.completed.as_ptr(), storage);
        assert!(state.completed.iter().all(Option::is_none));
    }

    #[test]
    fn decodes_intra_macroblock_type_binarizations() {
        let mut visited = Vec::new();
        assert_eq!(
            decode_intra_macroblock_type(
                5,
                |context_index| {
                    visited.push(context_index);
                    Ok(0)
                },
                || unreachable!(),
            )
            .unwrap(),
            0
        );
        assert_eq!(visited, [5]);

        let mut bins = VecDeque::from([1]);
        assert_eq!(
            decode_intra_macroblock_type(3, |_| Ok(bins.pop_front().unwrap()), || Ok(1)).unwrap(),
            25
        );

        let mut bins = VecDeque::from([1, 1, 1, 1, 1, 0]);
        let mut visited = Vec::new();
        assert_eq!(
            decode_intra_macroblock_type(
                3,
                |context_index| {
                    visited.push(context_index);
                    Ok(bins.pop_front().unwrap())
                },
                || Ok(0),
            )
            .unwrap(),
            23
        );
        assert_eq!(visited, [3, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn intra_macroblock_context_counts_only_intra16_and_pcm_neighbours() {
        let mut state = CabacMacroblockState::new(2, 2).unwrap();
        state
            .record_macroblock(0, 7, summary(false, true, Some(0), 0, 0))
            .unwrap();
        state
            .record_macroblock(1, 7, summary(false, false, Some(0), 0, 0))
            .unwrap();
        let neighbours = state.left_and_top(2, 7).unwrap();
        assert_eq!(
            neighbours
                .into_iter()
                .filter(|entry| {
                    entry.is_some_and(|macroblock| macroblock.summary.intra16_or_pcm)
                })
                .count(),
            1
        );
    }

    #[test]
    fn decodes_luma_and_chroma_coded_block_patterns() {
        let left = CodedBlockPattern {
            luma: 0b1010,
            chroma: 2,
        };
        let top = CodedBlockPattern {
            luma: 0b0101,
            chroma: 1,
        };
        let mut bins = VecDeque::from([1, 0, 1, 1]);
        let mut visited = Vec::new();
        assert_eq!(
            decode_luma_coded_block_pattern(Some(left), Some(top), |context_index| {
                visited.push(context_index);
                Ok(bins.pop_front().unwrap())
            })
            .unwrap(),
            0b1101
        );
        assert_eq!(visited, [73, 75, 73, 75]);

        let mut bins = VecDeque::from([1, 1]);
        let mut visited = Vec::new();
        assert_eq!(
            decode_chroma_coded_block_pattern(Some(left), Some(top), |context_index| {
                visited.push(context_index);
                Ok(bins.pop_front().unwrap())
            })
            .unwrap(),
            2
        );
        assert_eq!(visited, [80, 82]);
    }

    #[test]
    fn validates_recorded_macroblock_summary_values() {
        let mut state = CabacMacroblockState::new(1, 1).unwrap();
        assert!(
            state
                .record_macroblock(0, 1, summary(false, false, Some(4), 0, 0))
                .is_err()
        );
        assert!(
            state
                .record_macroblock(0, 1, summary(false, false, None, 16, 0))
                .is_err()
        );
    }

    #[test]
    fn decodes_intra_luma_and_chroma_prediction_modes() {
        let mut predicted = VecDeque::from([1]);
        assert_eq!(
            decode_intra_prediction_mode(|_| Ok(predicted.pop_front().unwrap())).unwrap(),
            IntraPredictionModeSyntax {
                use_predicted: true,
                remaining_mode: None,
            }
        );

        let mut explicit = VecDeque::from([0, 1, 0, 1]);
        assert_eq!(
            decode_intra_prediction_mode(|_| Ok(explicit.pop_front().unwrap())).unwrap(),
            IntraPredictionModeSyntax {
                use_predicted: false,
                remaining_mode: Some(5),
            }
        );

        let mut chroma = VecDeque::from([1, 1, 0]);
        let mut visited = Vec::new();
        assert_eq!(
            decode_intra_chroma_prediction_mode(66, |context_index| {
                visited.push(context_index);
                Ok(chroma.pop_front().unwrap())
            })
            .unwrap(),
            2
        );
        assert_eq!(visited, [66, 67, 67]);
    }

    #[test]
    fn derives_every_intra16x16_header_field_from_mb_type() {
        for macroblock_type in 1..=24 {
            let (mode, coded_block_pattern) =
                intra16x16_fields_from_macroblock_type(macroblock_type);
            let type_index = macroblock_type - 1;
            assert_eq!(mode, type_index % 4);
            assert_eq!(coded_block_pattern.chroma, (type_index / 4) % 3);
            assert_eq!(
                coded_block_pattern.luma,
                if macroblock_type >= 13 { 15 } else { 0 }
            );
        }
    }

    #[test]
    fn decodes_every_p_macroblock_partition_shape() {
        let cases = [
            ([0, 0, 0], CabacPMacroblockType::L0_16x16),
            ([0, 0, 1], CabacPMacroblockType::EightByEight),
            ([0, 1, 0], CabacPMacroblockType::L0_8x16),
            ([0, 1, 1], CabacPMacroblockType::L0_16x8),
        ];
        for (bins, expected) in cases {
            let mut bins = VecDeque::from(bins);
            let mut visited = Vec::new();
            let actual = decode_p_macroblock_type_with(|request| match request {
                CabacBinRequest::Decision(context_index) => {
                    visited.push(context_index);
                    Ok(bins.pop_front().unwrap())
                }
                CabacBinRequest::Terminate => {
                    unreachable!("inter P types do not use termination")
                }
            })
            .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(
                visited,
                if matches!(
                    expected,
                    CabacPMacroblockType::L0_16x16 | CabacPMacroblockType::EightByEight
                ) {
                    vec![14, 15, 16]
                } else {
                    vec![14, 15, 17]
                }
            );
        }
    }

    #[test]
    fn decodes_embedded_p_intra_and_pcm_types() {
        let mut nxn_bins = VecDeque::from([1, 0]);
        assert_eq!(
            decode_p_macroblock_type_with(|request| match request {
                CabacBinRequest::Decision(_) => Ok(nxn_bins.pop_front().unwrap()),
                CabacBinRequest::Terminate => unreachable!(),
            })
            .unwrap(),
            CabacPMacroblockType::Intra(0)
        );

        let mut pcm_bins = VecDeque::from([1, 1]);
        assert_eq!(
            decode_p_macroblock_type_with(|request| match request {
                CabacBinRequest::Decision(_) => Ok(pcm_bins.pop_front().unwrap()),
                CabacBinRequest::Terminate => Ok(1),
            })
            .unwrap(),
            CabacPMacroblockType::Intra(25)
        );
    }

    #[test]
    fn decodes_every_p_sub_macroblock_shape() {
        let cases: &[(&[u8], PSubMacroblockType)] = &[
            (&[1], PSubMacroblockType::L0_8x8),
            (&[0, 0], PSubMacroblockType::L0_8x4),
            (&[0, 1, 1], PSubMacroblockType::L0_4x8),
            (&[0, 1, 0], PSubMacroblockType::L0_4x4),
        ];
        for &(bins, expected) in cases {
            let mut bins: VecDeque<_> = bins.iter().copied().collect();
            assert_eq!(
                decode_p_sub_macroblock_type_with(|_| Ok(bins.pop_front().unwrap())).unwrap(),
                expected
            );
            assert!(bins.is_empty());
        }
    }

    #[test]
    fn decodes_every_b_inter_macroblock_type() {
        for expected_type in 0..=22u8 {
            let mut bins = match expected_type {
                0 => vec![0],
                1 | 2 => vec![1, 0, expected_type - 1],
                3..=10 => {
                    let bits = expected_type - 3;
                    vec![
                        1,
                        1,
                        (bits >> 3) & 1,
                        (bits >> 2) & 1,
                        (bits >> 1) & 1,
                        bits & 1,
                    ]
                }
                11 => vec![1, 1, 1, 1, 1, 0],
                12..=21 => {
                    let code = expected_type + 4;
                    let bits = code >> 1;
                    vec![
                        1,
                        1,
                        (bits >> 3) & 1,
                        (bits >> 2) & 1,
                        (bits >> 1) & 1,
                        bits & 1,
                        code & 1,
                    ]
                }
                22 => vec![1, 1, 1, 1, 1, 1],
                _ => unreachable!(),
            };
            let mut visited = Vec::new();
            let actual = decode_b_macroblock_type_with(usize::from(expected_type % 3), |request| {
                match request {
                    CabacBinRequest::Decision(context_index) => {
                        visited.push(context_index);
                        Ok(bins.remove(0))
                    }
                    CabacBinRequest::Terminate => {
                        unreachable!("inter B types do not use termination")
                    }
                }
            })
            .unwrap();
            assert_eq!(actual, CabacBMacroblockType::Inter(expected_type));
            assert_eq!(visited[0], 27 + usize::from(expected_type % 3));
            assert!(bins.is_empty());
        }
    }

    #[test]
    fn decodes_embedded_b_intra_and_pcm_types() {
        let mut nxn_bins = VecDeque::from([1, 1, 1, 1, 0, 1, 0]);
        assert_eq!(
            decode_b_macroblock_type_with(0, |request| match request {
                CabacBinRequest::Decision(_) => Ok(nxn_bins.pop_front().unwrap()),
                CabacBinRequest::Terminate => unreachable!(),
            })
            .unwrap(),
            CabacBMacroblockType::Intra(0)
        );
        assert!(nxn_bins.is_empty());

        let mut pcm_bins = VecDeque::from([1, 1, 1, 1, 0, 1, 1]);
        assert_eq!(
            decode_b_macroblock_type_with(0, |request| match request {
                CabacBinRequest::Decision(_) => Ok(pcm_bins.pop_front().unwrap()),
                CabacBinRequest::Terminate => Ok(1),
            })
            .unwrap(),
            CabacBMacroblockType::Intra(25)
        );
        assert!(pcm_bins.is_empty());
    }

    #[test]
    fn decodes_every_b_sub_macroblock_shape() {
        let cases: &[(&[u8], BSubMacroblockType)] = &[
            (&[0], BSubMacroblockType::Direct8x8),
            (&[1, 0, 0], BSubMacroblockType::List0_8x8),
            (&[1, 0, 1], BSubMacroblockType::List1_8x8),
            (&[1, 1, 0, 0, 0], BSubMacroblockType::Bi8x8),
            (&[1, 1, 0, 0, 1], BSubMacroblockType::List0_8x4),
            (&[1, 1, 0, 1, 0], BSubMacroblockType::List0_4x8),
            (&[1, 1, 0, 1, 1], BSubMacroblockType::List1_8x4),
            (&[1, 1, 1, 0, 0, 0], BSubMacroblockType::List1_4x8),
            (&[1, 1, 1, 0, 0, 1], BSubMacroblockType::Bi8x4),
            (&[1, 1, 1, 0, 1, 0], BSubMacroblockType::Bi4x8),
            (&[1, 1, 1, 0, 1, 1], BSubMacroblockType::List0_4x4),
            (&[1, 1, 1, 1, 0], BSubMacroblockType::List1_4x4),
            (&[1, 1, 1, 1, 1], BSubMacroblockType::Bi4x4),
        ];
        for &(bins, expected) in cases {
            let mut bins: VecDeque<_> = bins.iter().copied().collect();
            assert_eq!(
                decode_b_sub_macroblock_type_with(|_| Ok(bins.pop_front().unwrap())).unwrap(),
                expected
            );
            assert!(bins.is_empty());
        }
    }

    #[test]
    fn derives_b_type_context_from_non_direct_neighbours() {
        let mut state = CabacMacroblockState::new(2, 2).unwrap();
        let mut direct = summary(true, false, None, 0, 0);
        direct.direct = true;
        state.record_macroblock(0, 9, direct).unwrap();
        state
            .record_macroblock(1, 9, summary(false, false, None, 0, 0))
            .unwrap();
        assert_eq!(state.b_macroblock_type_context_increment(2, 9), Ok(0));
        assert_eq!(state.b_macroblock_type_context_increment(3, 9), Ok(1));
        assert_eq!(state.b_macroblock_type_context_increment(3, 10), Ok(0));
    }

    #[test]
    fn decodes_macroblock_qp_delta_mapping_and_context_progression() {
        let mut zero = VecDeque::from([0]);
        assert_eq!(
            decode_macroblock_qp_delta(false, |_| Ok(zero.pop_front().unwrap())).unwrap(),
            0
        );

        let mut positive = VecDeque::from([1, 0]);
        let mut visited = Vec::new();
        assert_eq!(
            decode_macroblock_qp_delta(false, |context_index| {
                visited.push(context_index);
                Ok(positive.pop_front().unwrap())
            })
            .unwrap(),
            1
        );
        assert_eq!(visited, [60, 62]);

        let mut negative = VecDeque::from([1, 1, 0]);
        let mut visited = Vec::new();
        assert_eq!(
            decode_macroblock_qp_delta(true, |context_index| {
                visited.push(context_index);
                Ok(negative.pop_front().unwrap())
            })
            .unwrap(),
            -1
        );
        assert_eq!(visited, [61, 62, 63]);
    }
}
