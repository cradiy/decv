//! Macroblock syntax, state, and reconstruction orchestration.

use bit_readers::BitReader;

use crate::{H264Error, ResidualBlock, Result};

const INTRA_CODED_BLOCK_PATTERNS_420: [u8; 48] = [
    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26, 28,
    35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];
const INTER_CODED_BLOCK_PATTERNS_420: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34,
    36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntraResidual {
    pub luma_dc: Option<ResidualBlock>,
    pub luma: [ResidualBlock; 16],
    pub chroma_dc: [ResidualBlock; 2],
    pub chroma_ac: [[ResidualBlock; 4]; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterResidual {
    pub luma: [ResidualBlock; 16],
    pub chroma_dc: [ResidualBlock; 2],
    pub chroma_ac: [[ResidualBlock; 4]; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedIntraMacroblock {
    pub macroblock: IntraMacroblock,
    /// Present for predicted macroblocks and absent for I_PCM.
    pub residual: Option<IntraResidual>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionVectorDifference {
    /// Horizontal displacement difference in quarter-luma-sample units.
    pub x: i16,
    /// Vertical displacement difference in quarter-luma-sample units.
    pub y: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PSubMacroblockType {
    L0_8x8,
    L0_8x4,
    L0_4x8,
    L0_4x4,
}

impl PSubMacroblockType {
    #[inline]
    pub const fn partition_count(self) -> usize {
        match self {
            Self::L0_8x8 => 1,
            Self::L0_8x4 | Self::L0_4x8 => 2,
            Self::L0_4x4 => 4,
        }
    }

    #[inline]
    pub const fn partition_size(self) -> (u8, u8) {
        match self {
            Self::L0_8x8 => (8, 8),
            Self::L0_8x4 => (8, 4),
            Self::L0_4x8 => (4, 8),
            Self::L0_4x4 => (4, 4),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PPartitionMode {
    L0_16x16,
    L0_16x8,
    L0_8x16,
    L0_8x8 {
        sub_macroblocks: [PSubMacroblockType; 4],
        reference_index_forced_zero: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PPartitionMotion {
    pub reference_index: u8,
    /// One entry for ordinary macroblock partitions, or one per sub-partition
    /// for P_8x8/P_8x8ref0.
    pub differences: Vec<MotionVectorDifference>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PInterMacroblockHeader {
    pub partition_mode: PPartitionMode,
    pub partitions: Vec<PPartitionMotion>,
    pub coded_block_pattern: CodedBlockPattern,
    pub transform_size_8x8: bool,
    /// Zero when `mb_qp_delta` is absent and inferred.
    pub qp_delta: i8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PSliceMacroblock {
    Inter(PInterMacroblockHeader),
    Intra(IntraMacroblock),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedPSliceMacroblock {
    Inter {
        header: PInterMacroblockHeader,
        residual: InterResidual,
    },
    Intra(DecodedIntraMacroblock),
}

/// Syntax context needed to decode a frame-coded CAVLC P macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PMacroblockContext {
    pub num_ref_idx_l0_active: u8,
    pub transform_8x8_mode_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BPredictionMode {
    Direct,
    List0,
    List1,
    Bi,
}

impl BPredictionMode {
    #[inline]
    pub const fn uses_list0(self) -> bool {
        matches!(self, Self::List0 | Self::Bi)
    }

    #[inline]
    pub const fn uses_list1(self) -> bool {
        matches!(self, Self::List1 | Self::Bi)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BSubMacroblockType {
    Direct8x8,
    List0_8x8,
    List1_8x8,
    Bi8x8,
    List0_8x4,
    List0_4x8,
    List1_8x4,
    List1_4x8,
    Bi8x4,
    Bi4x8,
    List0_4x4,
    List1_4x4,
    Bi4x4,
}

impl BSubMacroblockType {
    #[inline]
    pub const fn prediction(self) -> BPredictionMode {
        match self {
            Self::Direct8x8 => BPredictionMode::Direct,
            Self::List0_8x8 | Self::List0_8x4 | Self::List0_4x8 | Self::List0_4x4 => {
                BPredictionMode::List0
            }
            Self::List1_8x8 | Self::List1_8x4 | Self::List1_4x8 | Self::List1_4x4 => {
                BPredictionMode::List1
            }
            Self::Bi8x8 | Self::Bi8x4 | Self::Bi4x8 | Self::Bi4x4 => BPredictionMode::Bi,
        }
    }

    #[inline]
    pub const fn partition_count(self) -> usize {
        match self {
            Self::Direct8x8 | Self::List0_8x8 | Self::List1_8x8 | Self::Bi8x8 => 1,
            Self::List0_8x4
            | Self::List0_4x8
            | Self::List1_8x4
            | Self::List1_4x8
            | Self::Bi8x4
            | Self::Bi4x8 => 2,
            Self::List0_4x4 | Self::List1_4x4 | Self::Bi4x4 => 4,
        }
    }

    #[inline]
    pub const fn partition_size(self) -> (u8, u8) {
        match self {
            Self::Direct8x8 | Self::List0_8x8 | Self::List1_8x8 | Self::Bi8x8 => (8, 8),
            Self::List0_8x4 | Self::List1_8x4 | Self::Bi8x4 => (8, 4),
            Self::List0_4x8 | Self::List1_4x8 | Self::Bi4x8 => (4, 8),
            Self::List0_4x4 | Self::List1_4x4 | Self::Bi4x4 => (4, 4),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BPartitionMode {
    Direct16x16,
    SixteenBySixteen,
    SixteenByEight,
    EightBySixteen,
    EightByEight {
        sub_macroblocks: [BSubMacroblockType; 4],
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BPartitionMotion {
    pub prediction: BPredictionMode,
    pub reference_index_l0: Option<u8>,
    pub reference_index_l1: Option<u8>,
    /// Empty for an unused list or Direct; otherwise one entry for an
    /// ordinary macroblock partition or one per sub-macroblock partition.
    pub differences_l0: Vec<MotionVectorDifference>,
    pub differences_l1: Vec<MotionVectorDifference>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BInterMacroblockHeader {
    pub partition_mode: BPartitionMode,
    pub partitions: Vec<BPartitionMotion>,
    pub coded_block_pattern: CodedBlockPattern,
    pub transform_size_8x8: bool,
    /// Zero when `mb_qp_delta` is absent and inferred.
    pub qp_delta: i8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BSliceMacroblock {
    Inter(BInterMacroblockHeader),
    Intra(IntraMacroblock),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedBSliceMacroblock {
    Inter {
        header: BInterMacroblockHeader,
        residual: InterResidual,
    },
    Intra(DecodedIntraMacroblock),
}

/// Syntax context needed to decode a frame-coded CAVLC B macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BMacroblockContext {
    pub num_ref_idx_l0_active: u8,
    pub num_ref_idx_l1_active: u8,
    pub transform_8x8_mode_enabled: bool,
    pub direct_8x8_inference: bool,
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
    let macroblock = parse_intra_macroblock_type(&mut probe, mb_type, transform_8x8_mode_enabled)?;
    *reader = probe;
    Ok(macroblock)
}

/// Parses one non-skipped, frame-coded CAVLC P-slice macroblock up to the
/// residual coefficient syntax.
///
/// P-slice mb_type values 5 through 30 are mapped to their I-slice
/// counterparts, including complete I_PCM sample consumption. Any failure
/// leaves the reader unchanged.
pub fn parse_cavlc_p_macroblock(
    reader: &mut BitReader<'_>,
    context: PMacroblockContext,
) -> Result<PSliceMacroblock> {
    if context.num_ref_idx_l0_active == 0 || context.num_ref_idx_l0_active > 32 {
        return Err(H264Error::InvalidSyntax(
            "P macroblock active reference count is outside 1..=32",
        ));
    }

    let mut probe = *reader;
    let mb_type = probe.read_ue().ok_or(H264Error::UnexpectedEof)?;
    let macroblock = match mb_type {
        0..=4 => PSliceMacroblock::Inter(parse_p_inter_macroblock(&mut probe, mb_type, context)?),
        5..=30 => PSliceMacroblock::Intra(parse_intra_macroblock_type(
            &mut probe,
            mb_type - 5,
            context.transform_8x8_mode_enabled,
        )?),
        _ => {
            return Err(H264Error::InvalidSyntax(
                "mb_type exceeds the P-slice macroblock table",
            ));
        }
    };
    *reader = probe;
    Ok(macroblock)
}

/// Parses one non-skipped, frame-coded CAVLC B-slice macroblock up to the
/// residual coefficient syntax.
///
/// B-slice mb_type values 23 through 48 are mapped to their I-slice
/// counterparts. Any failure leaves the reader unchanged.
pub fn parse_cavlc_b_macroblock(
    reader: &mut BitReader<'_>,
    context: BMacroblockContext,
) -> Result<BSliceMacroblock> {
    if context.num_ref_idx_l0_active == 0
        || context.num_ref_idx_l0_active > 32
        || context.num_ref_idx_l1_active == 0
        || context.num_ref_idx_l1_active > 32
    {
        return Err(H264Error::InvalidSyntax(
            "B macroblock active reference count is outside 1..=32",
        ));
    }

    let mut probe = *reader;
    let mb_type = probe.read_ue().ok_or(H264Error::UnexpectedEof)?;
    let macroblock = match mb_type {
        0..=22 => BSliceMacroblock::Inter(parse_b_inter_macroblock(&mut probe, mb_type, context)?),
        23..=48 => BSliceMacroblock::Intra(parse_intra_macroblock_type(
            &mut probe,
            mb_type - 23,
            context.transform_8x8_mode_enabled,
        )?),
        _ => {
            return Err(H264Error::InvalidSyntax(
                "mb_type exceeds the B-slice macroblock table",
            ));
        }
    };
    *reader = probe;
    Ok(macroblock)
}

/// Parses CAVLC `mb_skip_run` and bounds it to the remaining picture.
pub fn parse_cavlc_mb_skip_run(
    reader: &mut BitReader<'_>,
    remaining_macroblocks: usize,
) -> Result<usize> {
    let mut probe = *reader;
    let value = probe.read_ue().ok_or(H264Error::UnexpectedEof)?;
    let value = usize::try_from(value).map_err(|_| H264Error::IntegerOverflow)?;
    if value > remaining_macroblocks {
        return Err(H264Error::InvalidSyntax(
            "mb_skip_run exceeds the remaining picture",
        ));
    }
    *reader = probe;
    Ok(value)
}

fn parse_intra_macroblock_type(
    reader: &mut BitReader<'_>,
    mb_type: u32,
    transform_8x8_mode_enabled: bool,
) -> Result<IntraMacroblock> {
    match mb_type {
        0 => Ok(IntraMacroblock::Predicted(parse_intra_nxn(
            reader,
            transform_8x8_mode_enabled,
        )?)),
        1..=24 => Ok(IntraMacroblock::Predicted(parse_intra_16x16(
            reader, mb_type,
        )?)),
        25 => Ok(IntraMacroblock::Pcm(parse_pcm(reader)?)),
        _ => Err(H264Error::InvalidSyntax(
            "mb_type exceeds the I-slice macroblock table",
        )),
    }
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

fn parse_p_inter_macroblock(
    reader: &mut BitReader<'_>,
    mb_type: u32,
    context: PMacroblockContext,
) -> Result<PInterMacroblockHeader> {
    let (partition_mode, partitions, permits_transform_8x8) = match mb_type {
        0 => (
            PPartitionMode::L0_16x16,
            parse_macroblock_partition_motion(reader, 1, context.num_ref_idx_l0_active)?,
            true,
        ),
        1 => (
            PPartitionMode::L0_16x8,
            parse_macroblock_partition_motion(reader, 2, context.num_ref_idx_l0_active)?,
            true,
        ),
        2 => (
            PPartitionMode::L0_8x16,
            parse_macroblock_partition_motion(reader, 2, context.num_ref_idx_l0_active)?,
            true,
        ),
        3 | 4 => {
            let forced_zero = mb_type == 4;
            let mut sub_macroblocks = [PSubMacroblockType::L0_8x8; 4];
            for sub_type in &mut sub_macroblocks {
                *sub_type = parse_p_sub_macroblock_type(reader)?;
            }
            let mut reference_indices = [0; 4];
            if !forced_zero {
                for reference_index in &mut reference_indices {
                    *reference_index =
                        parse_reference_index(reader, context.num_ref_idx_l0_active)?;
                }
            }
            let mut partitions = Vec::with_capacity(4);
            for (sub_type, reference_index) in sub_macroblocks.into_iter().zip(reference_indices) {
                let mut differences = Vec::with_capacity(sub_type.partition_count());
                for _ in 0..sub_type.partition_count() {
                    differences.push(parse_motion_vector_difference(reader)?);
                }
                partitions.push(PPartitionMotion {
                    reference_index,
                    differences,
                });
            }
            (
                PPartitionMode::L0_8x8 {
                    sub_macroblocks,
                    reference_index_forced_zero: forced_zero,
                },
                partitions,
                sub_macroblocks
                    .iter()
                    .all(|sub_type| *sub_type == PSubMacroblockType::L0_8x8),
            )
        }
        _ => unreachable!("caller restricts P inter mb_type to 0..=4"),
    };

    let coded_block_pattern = parse_inter_coded_block_pattern(reader)?;
    let transform_size_8x8 = coded_block_pattern.luma != 0
        && context.transform_8x8_mode_enabled
        && permits_transform_8x8
        && read_flag(reader)?;
    let qp_delta = if coded_block_pattern.has_residual() {
        parse_qp_delta(reader)?
    } else {
        0
    };
    Ok(PInterMacroblockHeader {
        partition_mode,
        partitions,
        coded_block_pattern,
        transform_size_8x8,
        qp_delta,
    })
}

fn parse_b_inter_macroblock(
    reader: &mut BitReader<'_>,
    mb_type: u32,
    context: BMacroblockContext,
) -> Result<BInterMacroblockHeader> {
    let (partition_mode, partitions, permits_transform_8x8) = match mb_type {
        0 => (
            BPartitionMode::Direct16x16,
            vec![empty_b_partition(BPredictionMode::Direct)],
            context.direct_8x8_inference,
        ),
        1..=3 => {
            let prediction = match mb_type {
                1 => BPredictionMode::List0,
                2 => BPredictionMode::List1,
                3 => BPredictionMode::Bi,
                _ => unreachable!(),
            };
            (
                BPartitionMode::SixteenBySixteen,
                parse_b_partition_motion(reader, &[prediction], context)?,
                true,
            )
        }
        4..=21 => {
            let partition_mode = if mb_type.is_multiple_of(2) {
                BPartitionMode::SixteenByEight
            } else {
                BPartitionMode::EightBySixteen
            };
            let pair_index =
                usize::try_from((mb_type - 4) / 2).map_err(|_| H264Error::IntegerOverflow)?;
            let predictions = [
                [BPredictionMode::List0, BPredictionMode::List0],
                [BPredictionMode::List1, BPredictionMode::List1],
                [BPredictionMode::List0, BPredictionMode::List1],
                [BPredictionMode::List1, BPredictionMode::List0],
                [BPredictionMode::List0, BPredictionMode::Bi],
                [BPredictionMode::List1, BPredictionMode::Bi],
                [BPredictionMode::Bi, BPredictionMode::List0],
                [BPredictionMode::Bi, BPredictionMode::List1],
                [BPredictionMode::Bi, BPredictionMode::Bi],
            ][pair_index];
            (
                partition_mode,
                parse_b_partition_motion(reader, &predictions, context)?,
                true,
            )
        }
        22 => parse_b_sub_macroblocks(reader, context)?,
        _ => unreachable!("caller restricts B inter mb_type to 0..=22"),
    };

    let coded_block_pattern = parse_inter_coded_block_pattern(reader)?;
    let transform_size_8x8 = coded_block_pattern.luma != 0
        && context.transform_8x8_mode_enabled
        && permits_transform_8x8
        && read_flag(reader)?;
    let qp_delta = if coded_block_pattern.has_residual() {
        parse_qp_delta(reader)?
    } else {
        0
    };
    Ok(BInterMacroblockHeader {
        partition_mode,
        partitions,
        coded_block_pattern,
        transform_size_8x8,
        qp_delta,
    })
}

fn parse_b_partition_motion(
    reader: &mut BitReader<'_>,
    predictions: &[BPredictionMode],
    context: BMacroblockContext,
) -> Result<Vec<BPartitionMotion>> {
    let mut partitions = predictions
        .iter()
        .copied()
        .map(empty_b_partition)
        .collect::<Vec<_>>();
    for partition in &mut partitions {
        if partition.prediction.uses_list0() {
            partition.reference_index_l0 = Some(parse_reference_index(
                reader,
                context.num_ref_idx_l0_active,
            )?);
        }
    }
    for partition in &mut partitions {
        if partition.prediction.uses_list1() {
            partition.reference_index_l1 = Some(parse_reference_index(
                reader,
                context.num_ref_idx_l1_active,
            )?);
        }
    }
    for partition in &mut partitions {
        if partition.prediction.uses_list0() {
            partition
                .differences_l0
                .push(parse_motion_vector_difference(reader)?);
        }
    }
    for partition in &mut partitions {
        if partition.prediction.uses_list1() {
            partition
                .differences_l1
                .push(parse_motion_vector_difference(reader)?);
        }
    }
    Ok(partitions)
}

fn parse_b_sub_macroblocks(
    reader: &mut BitReader<'_>,
    context: BMacroblockContext,
) -> Result<(BPartitionMode, Vec<BPartitionMotion>, bool)> {
    let mut sub_macroblocks = [BSubMacroblockType::Direct8x8; 4];
    for sub_type in &mut sub_macroblocks {
        *sub_type = parse_b_sub_macroblock_type(reader)?;
    }
    let mut partitions = sub_macroblocks
        .iter()
        .map(|sub_type| empty_b_partition(sub_type.prediction()))
        .collect::<Vec<_>>();
    for partition in &mut partitions {
        if partition.prediction.uses_list0() {
            partition.reference_index_l0 = Some(parse_reference_index(
                reader,
                context.num_ref_idx_l0_active,
            )?);
        }
    }
    for partition in &mut partitions {
        if partition.prediction.uses_list1() {
            partition.reference_index_l1 = Some(parse_reference_index(
                reader,
                context.num_ref_idx_l1_active,
            )?);
        }
    }
    for (partition, sub_type) in partitions.iter_mut().zip(sub_macroblocks) {
        if partition.prediction.uses_list0() {
            for _ in 0..sub_type.partition_count() {
                partition
                    .differences_l0
                    .push(parse_motion_vector_difference(reader)?);
            }
        }
    }
    for (partition, sub_type) in partitions.iter_mut().zip(sub_macroblocks) {
        if partition.prediction.uses_list1() {
            for _ in 0..sub_type.partition_count() {
                partition
                    .differences_l1
                    .push(parse_motion_vector_difference(reader)?);
            }
        }
    }
    let permits_transform_8x8 = sub_macroblocks.iter().all(|sub_type| {
        sub_type.partition_size() == (8, 8)
            && (*sub_type != BSubMacroblockType::Direct8x8 || context.direct_8x8_inference)
    });
    Ok((
        BPartitionMode::EightByEight { sub_macroblocks },
        partitions,
        permits_transform_8x8,
    ))
}

#[inline]
fn empty_b_partition(prediction: BPredictionMode) -> BPartitionMotion {
    BPartitionMotion {
        prediction,
        reference_index_l0: None,
        reference_index_l1: None,
        differences_l0: Vec::new(),
        differences_l1: Vec::new(),
    }
}

fn parse_b_sub_macroblock_type(reader: &mut BitReader<'_>) -> Result<BSubMacroblockType> {
    match reader.read_ue().ok_or(H264Error::UnexpectedEof)? {
        0 => Ok(BSubMacroblockType::Direct8x8),
        1 => Ok(BSubMacroblockType::List0_8x8),
        2 => Ok(BSubMacroblockType::List1_8x8),
        3 => Ok(BSubMacroblockType::Bi8x8),
        4 => Ok(BSubMacroblockType::List0_8x4),
        5 => Ok(BSubMacroblockType::List0_4x8),
        6 => Ok(BSubMacroblockType::List1_8x4),
        7 => Ok(BSubMacroblockType::List1_4x8),
        8 => Ok(BSubMacroblockType::Bi8x4),
        9 => Ok(BSubMacroblockType::Bi4x8),
        10 => Ok(BSubMacroblockType::List0_4x4),
        11 => Ok(BSubMacroblockType::List1_4x4),
        12 => Ok(BSubMacroblockType::Bi4x4),
        _ => Err(H264Error::InvalidSyntax(
            "sub_mb_type exceeds the B-slice table",
        )),
    }
}

fn parse_macroblock_partition_motion(
    reader: &mut BitReader<'_>,
    partition_count: usize,
    num_ref_idx_l0_active: u8,
) -> Result<Vec<PPartitionMotion>> {
    let mut reference_indices = Vec::with_capacity(partition_count);
    for _ in 0..partition_count {
        reference_indices.push(parse_reference_index(reader, num_ref_idx_l0_active)?);
    }
    let mut partitions = Vec::with_capacity(partition_count);
    for reference_index in reference_indices {
        partitions.push(PPartitionMotion {
            reference_index,
            differences: vec![parse_motion_vector_difference(reader)?],
        });
    }
    Ok(partitions)
}

fn parse_p_sub_macroblock_type(reader: &mut BitReader<'_>) -> Result<PSubMacroblockType> {
    match reader.read_ue().ok_or(H264Error::UnexpectedEof)? {
        0 => Ok(PSubMacroblockType::L0_8x8),
        1 => Ok(PSubMacroblockType::L0_8x4),
        2 => Ok(PSubMacroblockType::L0_4x8),
        3 => Ok(PSubMacroblockType::L0_4x4),
        _ => Err(H264Error::InvalidSyntax(
            "sub_mb_type exceeds the P-slice table",
        )),
    }
}

fn parse_reference_index(reader: &mut BitReader<'_>, active_count: u8) -> Result<u8> {
    match active_count {
        0 => Err(H264Error::InvalidSyntax(
            "P macroblock has no active reference pictures",
        )),
        1 => Ok(0),
        2 => Ok(u8::from(!read_flag(reader)?)),
        _ => {
            let index = reader.read_ue().ok_or(H264Error::UnexpectedEof)?;
            u8::try_from(index)
                .ok()
                .filter(|&index| index < active_count)
                .ok_or(H264Error::InvalidSyntax(
                    "ref_idx_l0 exceeds the active reference list",
                ))
        }
    }
}

fn parse_motion_vector_difference(reader: &mut BitReader<'_>) -> Result<MotionVectorDifference> {
    let x = reader.read_se().ok_or(H264Error::UnexpectedEof)?;
    let y = reader.read_se().ok_or(H264Error::UnexpectedEof)?;
    Ok(MotionVectorDifference {
        x: i16::try_from(x).map_err(|_| {
            H264Error::InvalidSyntax("horizontal mvd_l0 is outside the supported range")
        })?,
        y: i16::try_from(y).map_err(|_| {
            H264Error::InvalidSyntax("vertical mvd_l0 is outside the supported range")
        })?,
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

fn parse_inter_coded_block_pattern(reader: &mut BitReader<'_>) -> Result<CodedBlockPattern> {
    let code_num = reader.read_ue().ok_or(H264Error::UnexpectedEof)?;
    let value = *INTER_CODED_BLOCK_PATTERNS_420
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
        BMacroblockContext, BPartitionMode, BPredictionMode, BSliceMacroblock, BSubMacroblockType,
        CodedBlockPattern, IntraLumaPrediction, IntraMacroblock, IntraMacroblockHeader,
        IntraPredictionModeSyntax, MotionVectorDifference, PInterMacroblockHeader,
        PMacroblockContext, PPartitionMode, PPartitionMotion, PSliceMacroblock, PSubMacroblockType,
        parse_cavlc_b_macroblock, parse_cavlc_intra_macroblock, parse_cavlc_mb_skip_run,
        parse_cavlc_p_macroblock,
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
    fn parses_p_16x16_with_inferred_reference_index() {
        let mut writer = BitWriter::default();
        writer.write_ue(0);
        writer.write_se(2);
        writer.write_se(-1);
        writer.write_ue(0); // inter coded_block_pattern = 0

        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        assert_eq!(
            parse_cavlc_p_macroblock(
                &mut reader,
                PMacroblockContext {
                    num_ref_idx_l0_active: 1,
                    transform_8x8_mode_enabled: false,
                },
            ),
            Ok(PSliceMacroblock::Inter(PInterMacroblockHeader {
                partition_mode: PPartitionMode::L0_16x16,
                partitions: vec![PPartitionMotion {
                    reference_index: 0,
                    differences: vec![MotionVectorDifference { x: 2, y: -1 }],
                }],
                coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
                transform_size_8x8: false,
                qp_delta: 0,
            }))
        );
        assert_eq!(reader.bit_position(), writer.bit_len);
    }

    #[test]
    fn parses_p_16x8_reference_indices_transform_and_qp() {
        let mut writer = BitWriter::default();
        writer.write_ue(1);
        writer.write_flag(true); // te(v) value 0
        writer.write_flag(false); // te(v) value 1
        for (x, y) in [(1, 2), (-3, 4)] {
            writer.write_se(x);
            writer.write_se(y);
        }
        writer.write_ue(2); // inter coded_block_pattern = 1
        writer.write_flag(true);
        writer.write_se(-2);

        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        let parsed = parse_cavlc_p_macroblock(
            &mut reader,
            PMacroblockContext {
                num_ref_idx_l0_active: 2,
                transform_8x8_mode_enabled: true,
            },
        )
        .unwrap();
        let PSliceMacroblock::Inter(header) = parsed else {
            panic!("expected inter macroblock");
        };
        assert_eq!(header.partition_mode, PPartitionMode::L0_16x8);
        assert_eq!(
            header
                .partitions
                .iter()
                .map(|partition| partition.reference_index)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            header
                .partitions
                .iter()
                .map(|partition| partition.differences[0])
                .collect::<Vec<_>>(),
            [
                MotionVectorDifference { x: 1, y: 2 },
                MotionVectorDifference { x: -3, y: 4 },
            ]
        );
        assert_eq!(
            header.coded_block_pattern,
            CodedBlockPattern { luma: 1, chroma: 0 }
        );
        assert!(header.transform_size_8x8);
        assert_eq!(header.qp_delta, -2);
        assert_eq!(reader.bit_position(), writer.bit_len);
    }

    #[test]
    fn parses_every_p_sub_macroblock_shape_in_syntax_order() {
        let mut writer = BitWriter::default();
        writer.write_ue(3);
        for sub_type in 0..4 {
            writer.write_ue(sub_type);
        }
        for index in 0..9 {
            writer.write_se(index);
            writer.write_se(-index);
        }
        writer.write_ue(0);

        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        let PSliceMacroblock::Inter(header) = parse_cavlc_p_macroblock(
            &mut reader,
            PMacroblockContext {
                num_ref_idx_l0_active: 1,
                transform_8x8_mode_enabled: true,
            },
        )
        .unwrap() else {
            panic!("expected inter macroblock");
        };
        assert_eq!(
            header.partition_mode,
            PPartitionMode::L0_8x8 {
                sub_macroblocks: [
                    PSubMacroblockType::L0_8x8,
                    PSubMacroblockType::L0_8x4,
                    PSubMacroblockType::L0_4x8,
                    PSubMacroblockType::L0_4x4,
                ],
                reference_index_forced_zero: false,
            }
        );
        assert_eq!(
            header
                .partitions
                .iter()
                .map(|partition| partition.differences.len())
                .collect::<Vec<_>>(),
            [1, 2, 2, 4]
        );
        assert!(!header.transform_size_8x8);
        assert_eq!(reader.bit_position(), writer.bit_len);
    }

    #[test]
    fn p_8x8ref0_omits_reference_index_syntax() {
        let mut writer = BitWriter::default();
        writer.write_ue(4);
        for _ in 0..4 {
            writer.write_ue(0);
        }
        for _ in 0..4 {
            writer.write_se(0);
            writer.write_se(0);
        }
        writer.write_ue(0);

        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        let PSliceMacroblock::Inter(header) = parse_cavlc_p_macroblock(
            &mut reader,
            PMacroblockContext {
                num_ref_idx_l0_active: 3,
                transform_8x8_mode_enabled: false,
            },
        )
        .unwrap() else {
            panic!("expected inter macroblock");
        };
        assert!(
            header
                .partitions
                .iter()
                .all(|part| part.reference_index == 0)
        );
        assert!(matches!(
            header.partition_mode,
            PPartitionMode::L0_8x8 {
                reference_index_forced_zero: true,
                ..
            }
        ));
        assert_eq!(reader.bit_position(), writer.bit_len);
    }

    #[test]
    fn maps_every_ordinary_b_inter_macroblock_type() {
        let modes = [
            (1, vec![BPredictionMode::List0]),
            (2, vec![BPredictionMode::List1]),
            (3, vec![BPredictionMode::Bi]),
            (4, vec![BPredictionMode::List0, BPredictionMode::List0]),
            (5, vec![BPredictionMode::List0, BPredictionMode::List0]),
            (6, vec![BPredictionMode::List1, BPredictionMode::List1]),
            (7, vec![BPredictionMode::List1, BPredictionMode::List1]),
            (8, vec![BPredictionMode::List0, BPredictionMode::List1]),
            (9, vec![BPredictionMode::List0, BPredictionMode::List1]),
            (10, vec![BPredictionMode::List1, BPredictionMode::List0]),
            (11, vec![BPredictionMode::List1, BPredictionMode::List0]),
            (12, vec![BPredictionMode::List0, BPredictionMode::Bi]),
            (13, vec![BPredictionMode::List0, BPredictionMode::Bi]),
            (14, vec![BPredictionMode::List1, BPredictionMode::Bi]),
            (15, vec![BPredictionMode::List1, BPredictionMode::Bi]),
            (16, vec![BPredictionMode::Bi, BPredictionMode::List0]),
            (17, vec![BPredictionMode::Bi, BPredictionMode::List0]),
            (18, vec![BPredictionMode::Bi, BPredictionMode::List1]),
            (19, vec![BPredictionMode::Bi, BPredictionMode::List1]),
            (20, vec![BPredictionMode::Bi, BPredictionMode::Bi]),
            (21, vec![BPredictionMode::Bi, BPredictionMode::Bi]),
        ];
        for (mb_type, predictions) in modes {
            let mut writer = BitWriter::default();
            writer.write_ue(mb_type);
            for prediction in &predictions {
                if prediction.uses_list0() {
                    writer.write_se(mb_type as i32);
                    writer.write_se(0);
                }
            }
            for prediction in &predictions {
                if prediction.uses_list1() {
                    writer.write_se(-(mb_type as i32));
                    writer.write_se(0);
                }
            }
            writer.write_ue(0);

            let data = writer.finish();
            let mut reader = BitReader::new(&data);
            let BSliceMacroblock::Inter(header) =
                parse_cavlc_b_macroblock(&mut reader, b_context()).unwrap()
            else {
                panic!("expected B inter macroblock type {mb_type}");
            };
            assert_eq!(
                header
                    .partitions
                    .iter()
                    .map(|partition| partition.prediction)
                    .collect::<Vec<_>>(),
                predictions,
                "mb_type={mb_type}"
            );
            let expected_mode = match mb_type {
                1..=3 => BPartitionMode::SixteenBySixteen,
                value if value.is_multiple_of(2) => BPartitionMode::SixteenByEight,
                _ => BPartitionMode::EightBySixteen,
            };
            assert_eq!(header.partition_mode, expected_mode, "mb_type={mb_type}");
            assert_eq!(reader.bit_position(), writer.bit_len, "mb_type={mb_type}");
        }
    }

    #[test]
    fn parses_b_direct_and_bidirectional_reference_syntax() {
        let mut direct = BitWriter::default();
        direct.write_ue(0);
        direct.write_ue(0);
        let data = direct.finish();
        let mut reader = BitReader::new(&data);
        let BSliceMacroblock::Inter(header) =
            parse_cavlc_b_macroblock(&mut reader, b_context()).unwrap()
        else {
            panic!("expected direct macroblock");
        };
        assert_eq!(header.partition_mode, BPartitionMode::Direct16x16);
        assert_eq!(header.partitions[0].prediction, BPredictionMode::Direct);
        assert_eq!(reader.bit_position(), direct.bit_len);

        let mut bi = BitWriter::default();
        bi.write_ue(20);
        bi.write_flag(true);
        bi.write_flag(false);
        bi.write_flag(false);
        bi.write_flag(true);
        for (x, y) in [(1, 2), (3, 4), (-1, -2), (-3, -4)] {
            bi.write_se(x);
            bi.write_se(y);
        }
        bi.write_ue(2);
        bi.write_flag(true);
        bi.write_se(-1);
        let data = bi.finish();
        let mut reader = BitReader::new(&data);
        let BSliceMacroblock::Inter(header) = parse_cavlc_b_macroblock(
            &mut reader,
            BMacroblockContext {
                num_ref_idx_l0_active: 2,
                num_ref_idx_l1_active: 2,
                transform_8x8_mode_enabled: true,
                direct_8x8_inference: true,
            },
        )
        .unwrap() else {
            panic!("expected bidirectional macroblock");
        };
        assert_eq!(header.partition_mode, BPartitionMode::SixteenByEight);
        assert_eq!(header.partitions[0].reference_index_l0, Some(0));
        assert_eq!(header.partitions[1].reference_index_l0, Some(1));
        assert_eq!(header.partitions[0].reference_index_l1, Some(1));
        assert_eq!(header.partitions[1].reference_index_l1, Some(0));
        assert_eq!(
            header.partitions[0].differences_l0,
            [MotionVectorDifference { x: 1, y: 2 }]
        );
        assert_eq!(
            header.partitions[0].differences_l1,
            [MotionVectorDifference { x: -1, y: -2 }]
        );
        assert!(header.transform_size_8x8);
        assert_eq!(header.qp_delta, -1);
        assert_eq!(reader.bit_position(), bi.bit_len);
    }

    #[test]
    fn parses_mixed_b_sub_macroblock_shapes_in_list_order() {
        let mut writer = BitWriter::default();
        writer.write_ue(22);
        for sub_type in [0, 4, 7, 12] {
            writer.write_ue(sub_type);
        }
        for index in 0..6 {
            writer.write_se(index);
            writer.write_se(-index);
        }
        for index in 6..12 {
            writer.write_se(index);
            writer.write_se(-index);
        }
        writer.write_ue(0);

        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        let BSliceMacroblock::Inter(header) =
            parse_cavlc_b_macroblock(&mut reader, b_context()).unwrap()
        else {
            panic!("expected B_8x8 macroblock");
        };
        assert_eq!(
            header.partition_mode,
            BPartitionMode::EightByEight {
                sub_macroblocks: [
                    BSubMacroblockType::Direct8x8,
                    BSubMacroblockType::List0_8x4,
                    BSubMacroblockType::List1_4x8,
                    BSubMacroblockType::Bi4x4,
                ],
            }
        );
        assert_eq!(
            header
                .partitions
                .iter()
                .map(|partition| partition.differences_l0.len())
                .collect::<Vec<_>>(),
            [0, 2, 0, 4]
        );
        assert_eq!(
            header
                .partitions
                .iter()
                .map(|partition| partition.differences_l1.len())
                .collect::<Vec<_>>(),
            [0, 0, 2, 4]
        );
        assert!(!header.transform_size_8x8);
        assert_eq!(reader.bit_position(), writer.bit_len);
    }

    #[test]
    fn maps_b_intra_types_and_rejects_invalid_syntax_atomically() {
        let mut intra = BitWriter::default();
        intra.write_ue(23);
        for _ in 0..16 {
            intra.write_flag(true);
        }
        intra.write_ue(0);
        intra.write_ue(3);
        let data = intra.finish();
        let mut reader = BitReader::new(&data);
        assert!(matches!(
            parse_cavlc_b_macroblock(&mut reader, b_context()),
            Ok(BSliceMacroblock::Intra(IntraMacroblock::Predicted(_)))
        ));

        let mut invalid_type = BitWriter::default();
        invalid_type.write_ue(49);
        let mut invalid_sub_type = BitWriter::default();
        invalid_sub_type.write_ue(22);
        invalid_sub_type.write_ue(13);
        for writer in [invalid_type, invalid_sub_type] {
            let data = writer.finish();
            let mut reader = BitReader::new(&data);
            assert!(parse_cavlc_b_macroblock(&mut reader, b_context()).is_err());
            assert_eq!(reader.bit_position(), 0);
        }
    }

    #[test]
    fn maps_p_slice_intra_types_and_bounds_skip_runs() {
        let mut writer = BitWriter::default();
        writer.write_ue(5);
        for _ in 0..16 {
            writer.write_flag(true);
        }
        writer.write_ue(0);
        writer.write_ue(3);
        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        assert!(matches!(
            parse_cavlc_p_macroblock(
                &mut reader,
                PMacroblockContext {
                    num_ref_idx_l0_active: 1,
                    transform_8x8_mode_enabled: false,
                },
            ),
            Ok(PSliceMacroblock::Intra(IntraMacroblock::Predicted(_)))
        ));

        let mut writer = BitWriter::default();
        writer.write_ue(3);
        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        assert_eq!(parse_cavlc_mb_skip_run(&mut reader, 3), Ok(3));
        let mut reader = BitReader::new(&data);
        assert!(parse_cavlc_mb_skip_run(&mut reader, 2).is_err());
        assert_eq!(reader.bit_position(), 0);
    }

    #[test]
    fn rejects_invalid_p_syntax_atomically() {
        let context = PMacroblockContext {
            num_ref_idx_l0_active: 3,
            transform_8x8_mode_enabled: false,
        };
        let mut invalid_type = BitWriter::default();
        invalid_type.write_ue(31);
        let mut invalid_sub_type = BitWriter::default();
        invalid_sub_type.write_ue(3);
        invalid_sub_type.write_ue(4);
        let mut invalid_reference = BitWriter::default();
        invalid_reference.write_ue(0);
        invalid_reference.write_ue(3);
        for writer in [invalid_type, invalid_sub_type, invalid_reference] {
            let data = writer.finish();
            let mut reader = BitReader::new(&data);
            assert!(parse_cavlc_p_macroblock(&mut reader, context).is_err());
            assert_eq!(reader.bit_position(), 0);
        }
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

    fn b_context() -> BMacroblockContext {
        BMacroblockContext {
            num_ref_idx_l0_active: 1,
            num_ref_idx_l1_active: 1,
            transform_8x8_mode_enabled: false,
            direct_8x8_inference: true,
        }
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
