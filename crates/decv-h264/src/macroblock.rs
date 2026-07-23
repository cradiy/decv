//! Macroblock syntax, state, and reconstruction orchestration.

use bit_readers::BitReader;

use crate::{H264Error, Result};

const INTRA_CODED_BLOCK_PATTERNS_420: [u8; 48] = [
    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26, 28,
    35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodedBlockPattern {
    /// Four bits, one for each 8x8 luma region.
    pub luma: u8,
    /// Zero means no chroma coefficients, one adds DC, and two adds DC + AC.
    pub chroma: u8,
}

impl CodedBlockPattern {
    #[inline]
    pub const fn has_residual(self) -> bool {
        self.luma != 0 || self.chroma != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntraPredictionModeSyntax {
    /// Use the mode predicted from the neighbouring blocks.
    pub use_predicted: bool,
    /// The three-bit rem_intra prediction mode when `use_predicted` is false.
    pub remaining_mode: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntraLumaPrediction {
    FourByFour([IntraPredictionModeSyntax; 16]),
    EightByEight([IntraPredictionModeSyntax; 4]),
    SixteenBySixteen { mode: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntraMacroblockHeader {
    pub luma_prediction: IntraLumaPrediction,
    pub chroma_prediction_mode: u8,
    pub coded_block_pattern: CodedBlockPattern,
    /// Zero when mb_qp_delta is absent and therefore inferred.
    pub qp_delta: i8,
}

impl IntraMacroblockHeader {
    #[inline]
    pub const fn has_residual(&self) -> bool {
        matches!(
            self.luma_prediction,
            IntraLumaPrediction::SixteenBySixteen { .. }
        ) || self.coded_block_pattern.has_residual()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcmMacroblock {
    pub luma: Box<[u8; 256]>,
    /// Interleaved only in syntax order: 64 Cb samples followed by 64 Cr.
    pub chroma: Box<[u8; 128]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntraMacroblock {
    Predicted(IntraMacroblockHeader),
    Pcm(PcmMacroblock),
}

/// Parses the non-residual portion of one CAVLC I-slice macroblock.
///
/// For predicted macroblocks the reader stops immediately before residual
/// coefficient syntax. I_PCM samples are consumed completely. Any failure
/// leaves the reader unchanged.
pub fn parse_cavlc_intra_macroblock(
    reader: &mut BitReader<'_>,
    transform_8x8_mode_enabled: bool,
) -> Result<IntraMacroblock> {
    let mut probe = *reader;
    let mb_type = probe.read_ue().ok_or(H264Error::UnexpectedEof)?;
    let macroblock = match mb_type {
        0 => IntraMacroblock::Predicted(parse_intra_nxn(&mut probe, transform_8x8_mode_enabled)?),
        1..=24 => IntraMacroblock::Predicted(parse_intra_16x16(&mut probe, mb_type)?),
        25 => IntraMacroblock::Pcm(parse_pcm(&mut probe)?),
        _ => {
            return Err(H264Error::InvalidSyntax(
                "mb_type exceeds the I-slice macroblock table",
            ));
        }
    };
    *reader = probe;
    Ok(macroblock)
}

fn parse_intra_nxn(
    reader: &mut BitReader<'_>,
    transform_8x8_mode_enabled: bool,
) -> Result<IntraMacroblockHeader> {
    let transform_size_8x8 = transform_8x8_mode_enabled && read_flag(reader)?;
    let luma_prediction = if transform_size_8x8 {
        let mut modes = [IntraPredictionModeSyntax {
            use_predicted: false,
            remaining_mode: None,
        }; 4];
        for mode in &mut modes {
            *mode = parse_intra_prediction_mode(reader)?;
        }
        IntraLumaPrediction::EightByEight(modes)
    } else {
        let mut modes = [IntraPredictionModeSyntax {
            use_predicted: false,
            remaining_mode: None,
        }; 16];
        for mode in &mut modes {
            *mode = parse_intra_prediction_mode(reader)?;
        }
        IntraLumaPrediction::FourByFour(modes)
    };

    let chroma_prediction_mode = parse_chroma_prediction_mode(reader)?;
    let coded_block_pattern = parse_intra_coded_block_pattern(reader)?;
    let qp_delta = if coded_block_pattern.has_residual() {
        parse_qp_delta(reader)?
    } else {
        0
    };
    Ok(IntraMacroblockHeader {
        luma_prediction,
        chroma_prediction_mode,
        coded_block_pattern,
        qp_delta,
    })
}

fn parse_intra_16x16(reader: &mut BitReader<'_>, mb_type: u32) -> Result<IntraMacroblockHeader> {
    let type_index = mb_type - 1;
    let mode = (type_index % 4) as u8;
    let chroma = ((type_index / 4) % 3) as u8;
    let luma = if type_index >= 12 { 15 } else { 0 };
    let chroma_prediction_mode = parse_chroma_prediction_mode(reader)?;
    let qp_delta = parse_qp_delta(reader)?;
    Ok(IntraMacroblockHeader {
        luma_prediction: IntraLumaPrediction::SixteenBySixteen { mode },
        chroma_prediction_mode,
        coded_block_pattern: CodedBlockPattern { luma, chroma },
        qp_delta,
    })
}

fn parse_intra_prediction_mode(reader: &mut BitReader<'_>) -> Result<IntraPredictionModeSyntax> {
    let use_predicted = read_flag(reader)?;
    let remaining_mode = if use_predicted {
        None
    } else {
        Some(
            reader
                .read_bits_const::<3>()
                .ok_or(H264Error::UnexpectedEof)? as u8,
        )
    };
    Ok(IntraPredictionModeSyntax {
        use_predicted,
        remaining_mode,
    })
}

fn parse_chroma_prediction_mode(reader: &mut BitReader<'_>) -> Result<u8> {
    let mode = reader.read_ue().ok_or(H264Error::UnexpectedEof)?;
    u8::try_from(mode)
        .ok()
        .filter(|&mode| mode <= 3)
        .ok_or(H264Error::InvalidSyntax("intra_chroma_pred_mode exceeds 3"))
}

fn parse_intra_coded_block_pattern(reader: &mut BitReader<'_>) -> Result<CodedBlockPattern> {
    let code_num = reader.read_ue().ok_or(H264Error::UnexpectedEof)?;
    let value = *INTRA_CODED_BLOCK_PATTERNS_420
        .get(usize::try_from(code_num).map_err(|_| H264Error::IntegerOverflow)?)
        .ok_or(H264Error::InvalidSyntax(
            "coded_block_pattern codeNum exceeds 47",
        ))?;
    Ok(CodedBlockPattern {
        luma: value & 0x0f,
        chroma: value >> 4,
    })
}

fn parse_qp_delta(reader: &mut BitReader<'_>) -> Result<i8> {
    let delta = reader.read_se().ok_or(H264Error::UnexpectedEof)?;
    i8::try_from(delta)
        .ok()
        .filter(|&delta| (-26..=25).contains(&delta))
        .ok_or(H264Error::InvalidSyntax(
            "mb_qp_delta is outside the 8-bit range",
        ))
}

fn parse_pcm(reader: &mut BitReader<'_>) -> Result<PcmMacroblock> {
    while reader.bit_offset() != 0 {
        if reader.read_bit().ok_or(H264Error::UnexpectedEof)? != 0 {
            return Err(H264Error::InvalidSyntax(
                "pcm_alignment_zero_bit is not zero",
            ));
        }
    }

    let mut luma = Box::new([0u8; 256]);
    for sample in luma.iter_mut() {
        *sample = read_u8(reader)?;
    }
    let mut chroma = Box::new([0u8; 128]);
    for sample in chroma.iter_mut() {
        *sample = read_u8(reader)?;
    }
    Ok(PcmMacroblock { luma, chroma })
}

#[inline]
fn read_flag(reader: &mut BitReader<'_>) -> Result<bool> {
    reader
        .read_bit()
        .map(|value| value != 0)
        .ok_or(H264Error::UnexpectedEof)
}

#[inline]
fn read_u8(reader: &mut BitReader<'_>) -> Result<u8> {
    reader
        .read_bits_const::<8>()
        .map(|value| value as u8)
        .ok_or(H264Error::UnexpectedEof)
}

#[cfg(test)]
mod tests {
    use bit_readers::BitReader;

    use super::{
        CodedBlockPattern, IntraLumaPrediction, IntraMacroblock, IntraMacroblockHeader,
        IntraPredictionModeSyntax, parse_cavlc_intra_macroblock,
    };
    use crate::H264Error;

    #[test]
    fn parses_intra_4x4_without_residual() {
        let mut writer = BitWriter::default();
        writer.write_ue(0);
        for _ in 0..16 {
            writer.write_flag(true);
        }
        writer.write_ue(0);
        writer.write_ue(3); // maps to coded_block_pattern = 0

        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        let parsed = parse_cavlc_intra_macroblock(&mut reader, false).unwrap();
        assert_eq!(
            parsed,
            IntraMacroblock::Predicted(IntraMacroblockHeader {
                luma_prediction: IntraLumaPrediction::FourByFour(
                    [IntraPredictionModeSyntax {
                        use_predicted: true,
                        remaining_mode: None,
                    }; 16]
                ),
                chroma_prediction_mode: 0,
                coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
                qp_delta: 0,
            })
        );
        assert_eq!(reader.bit_position(), writer.bit_len);
    }

    #[test]
    fn parses_intra_8x8_modes_coded_pattern_and_qp() {
        let mut writer = BitWriter::default();
        writer.write_ue(0);
        writer.write_flag(true);
        for remaining_mode in 0..4 {
            writer.write_flag(false);
            writer.write_bits(remaining_mode, 3);
        }
        writer.write_ue(2);
        writer.write_ue(0); // maps to 47: all luma plus chroma DC + AC
        writer.write_se(1);

        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        let parsed = parse_cavlc_intra_macroblock(&mut reader, true).unwrap();
        let IntraMacroblock::Predicted(header) = parsed else {
            panic!("expected predicted macroblock");
        };
        assert_eq!(
            header.luma_prediction,
            IntraLumaPrediction::EightByEight([
                IntraPredictionModeSyntax {
                    use_predicted: false,
                    remaining_mode: Some(0),
                },
                IntraPredictionModeSyntax {
                    use_predicted: false,
                    remaining_mode: Some(1),
                },
                IntraPredictionModeSyntax {
                    use_predicted: false,
                    remaining_mode: Some(2),
                },
                IntraPredictionModeSyntax {
                    use_predicted: false,
                    remaining_mode: Some(3),
                },
            ])
        );
        assert_eq!(header.chroma_prediction_mode, 2);
        assert_eq!(
            header.coded_block_pattern,
            CodedBlockPattern {
                luma: 15,
                chroma: 2,
            }
        );
        assert_eq!(header.qp_delta, 1);
        assert_eq!(reader.bit_position(), writer.bit_len);
    }

    #[test]
    fn derives_intra_16x16_fields_from_mb_type() {
        let mut writer = BitWriter::default();
        writer.write_ue(23);
        writer.write_ue(3);
        writer.write_se(-2);

        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        assert_eq!(
            parse_cavlc_intra_macroblock(&mut reader, true),
            Ok(IntraMacroblock::Predicted(IntraMacroblockHeader {
                luma_prediction: IntraLumaPrediction::SixteenBySixteen { mode: 2 },
                chroma_prediction_mode: 3,
                coded_block_pattern: CodedBlockPattern {
                    luma: 15,
                    chroma: 2,
                },
                qp_delta: -2,
            }))
        );
        assert_eq!(reader.bit_position(), writer.bit_len);
    }

    #[test]
    fn parses_aligned_pcm_samples() {
        let mut writer = BitWriter::default();
        writer.write_ue(25);
        writer.byte_align_zero();
        for value in 0..384 {
            writer.write_bits((value & 0xff) as u32, 8);
        }

        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        let IntraMacroblock::Pcm(pcm) = parse_cavlc_intra_macroblock(&mut reader, false).unwrap()
        else {
            panic!("expected PCM macroblock");
        };
        assert_eq!(pcm.luma[0], 0);
        assert_eq!(pcm.luma[255], 255);
        assert_eq!(pcm.chroma[0], 0);
        assert_eq!(pcm.chroma[127], 127);
        assert_eq!(reader.bit_position(), writer.bit_len);
    }

    #[test]
    fn rejects_invalid_or_truncated_macroblocks_atomically() {
        for writer in [
            invalid_mb_type(),
            invalid_chroma_mode(),
            invalid_qp_delta(),
            invalid_pcm_alignment(),
        ] {
            let data = writer.finish();
            let mut reader = BitReader::new(&data);
            assert!(matches!(
                parse_cavlc_intra_macroblock(&mut reader, false),
                Err(H264Error::InvalidSyntax(_))
            ));
            assert_eq!(reader.bit_position(), 0);
        }

        let mut reader = BitReader::new(&[0]);
        assert_eq!(
            parse_cavlc_intra_macroblock(&mut reader, false),
            Err(H264Error::UnexpectedEof)
        );
        assert_eq!(reader.bit_position(), 0);
    }

    fn invalid_mb_type() -> BitWriter {
        let mut writer = BitWriter::default();
        writer.write_ue(26);
        writer
    }

    fn invalid_chroma_mode() -> BitWriter {
        let mut writer = BitWriter::default();
        writer.write_ue(1);
        writer.write_ue(4);
        writer
    }

    fn invalid_qp_delta() -> BitWriter {
        let mut writer = BitWriter::default();
        writer.write_ue(1);
        writer.write_ue(0);
        writer.write_se(26);
        writer
    }

    fn invalid_pcm_alignment() -> BitWriter {
        let mut writer = BitWriter::default();
        writer.write_ue(25);
        writer.write_flag(true);
        writer
    }

    #[derive(Default)]
    struct BitWriter {
        bits: Vec<u8>,
        bit_len: usize,
    }

    impl BitWriter {
        fn write_flag(&mut self, value: bool) {
            self.write_bits(u32::from(value), 1);
        }

        fn write_bits(&mut self, value: u32, count: usize) {
            for shift in (0..count).rev() {
                self.bits.push(((value >> shift) & 1) as u8);
                self.bit_len += 1;
            }
        }

        fn write_ue(&mut self, value: u32) {
            let code_num = u64::from(value) + 1;
            let width = 64 - code_num.leading_zeros() as usize;
            self.bits
                .extend(std::iter::repeat_n(0, width.saturating_sub(1)));
            self.bit_len += width.saturating_sub(1);
            self.write_bits(code_num as u32, width);
        }

        fn write_se(&mut self, value: i32) {
            let code_num = if value <= 0 {
                value.unsigned_abs() * 2
            } else {
                value as u32 * 2 - 1
            };
            self.write_ue(code_num);
        }

        fn byte_align_zero(&mut self) {
            while !self.bit_len.is_multiple_of(8) {
                self.write_flag(false);
            }
        }

        fn finish(&self) -> Vec<u8> {
            let mut bytes = vec![0; self.bits.len().div_ceil(8)];
            for (index, &bit) in self.bits.iter().enumerate() {
                bytes[index / 8] |= bit << (7 - index % 8);
            }
            bytes
        }
    }
}
