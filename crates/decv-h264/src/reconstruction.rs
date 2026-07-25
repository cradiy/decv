//! Conversion from decoded transform coefficients to spatial residual samples.

use crate::transform::PreparedInverseScale4x4;
use crate::{
    BInterMacroblockHeader, Block4x4, Block8x8, ColorComponent, H264Error, InterResidual,
    IntraLumaPrediction, IntraMacroblockHeader, IntraResidual, MacroblockQuantizer,
    PInterMacroblockHeader, PredictionClass, ResolvedScalingLists4x4, ResolvedScalingLists8x8,
    Result, ScanMode, inverse_transform_chroma_dc_420, inverse_transform_luma_dc_4x4,
    reconstruct_residual_8x8,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconstructedLumaResidual {
    FourByFour(Box<[Block4x4; 16]>),
    EightByEight(Box<[Block8x8; 4]>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructedIntraResidual {
    pub luma: ReconstructedLumaResidual,
    pub chroma_cb: [Block4x4; 4],
    pub chroma_cr: [Block4x4; 4],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructedInterResidual {
    // Kept inline so one P/B macroblock batch owns all residual samples in
    // its single `Vec` allocation. The previous box performed a separate
    // allocator round-trip for every inter macroblock carrying residuals.
    data: ReconstructedInterResidualData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReconstructedInterResidualData {
    luma: ReconstructedInterLumaResidual,
    chroma_cb: [Block4x4; 4],
    chroma_cr: [Block4x4; 4],
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ReconstructedInterLumaResidual {
    FourByFour([Block4x4; 16]),
    EightByEight([Block8x8; 4]),
}

pub(crate) enum ReconstructedInterLumaResidualRef<'a> {
    FourByFour(&'a [Block4x4; 16]),
    EightByEight(&'a [Block8x8; 4]),
}

impl ReconstructedInterResidual {
    pub(crate) fn four_by_four(
        luma: [Block4x4; 16],
        chroma_cb: [Block4x4; 4],
        chroma_cr: [Block4x4; 4],
    ) -> Self {
        Self {
            data: ReconstructedInterResidualData {
                luma: ReconstructedInterLumaResidual::FourByFour(luma),
                chroma_cb,
                chroma_cr,
            },
        }
    }

    pub(crate) fn eight_by_eight(
        luma: [Block8x8; 4],
        chroma_cb: [Block4x4; 4],
        chroma_cr: [Block4x4; 4],
    ) -> Self {
        Self {
            data: ReconstructedInterResidualData {
                luma: ReconstructedInterLumaResidual::EightByEight(luma),
                chroma_cb,
                chroma_cr,
            },
        }
    }

    pub fn luma_4x4(&self) -> Option<&[Block4x4; 16]> {
        match &self.data.luma {
            ReconstructedInterLumaResidual::FourByFour(luma) => Some(luma),
            ReconstructedInterLumaResidual::EightByEight(_) => None,
        }
    }

    pub fn luma_8x8(&self) -> Option<&[Block8x8; 4]> {
        match &self.data.luma {
            ReconstructedInterLumaResidual::FourByFour(_) => None,
            ReconstructedInterLumaResidual::EightByEight(luma) => Some(luma),
        }
    }

    pub(crate) fn luma(&self) -> ReconstructedInterLumaResidualRef<'_> {
        match &self.data.luma {
            ReconstructedInterLumaResidual::FourByFour(luma) => {
                ReconstructedInterLumaResidualRef::FourByFour(luma)
            }
            ReconstructedInterLumaResidual::EightByEight(luma) => {
                ReconstructedInterLumaResidualRef::EightByEight(luma)
            }
        }
    }

    pub fn chroma_cb(&self) -> &[Block4x4; 4] {
        &self.data.chroma_cb
    }

    pub fn chroma_cr(&self) -> &[Block4x4; 4] {
        &self.data.chroma_cr
    }

    pub(crate) fn luma_4x4_mut(&mut self) -> Option<&mut [Block4x4; 16]> {
        match &mut self.data.luma {
            ReconstructedInterLumaResidual::FourByFour(luma) => Some(luma),
            ReconstructedInterLumaResidual::EightByEight(_) => None,
        }
    }

    pub(crate) fn luma_8x8_mut(&mut self) -> Option<&mut [Block8x8; 4]> {
        match &mut self.data.luma {
            ReconstructedInterLumaResidual::FourByFour(_) => None,
            ReconstructedInterLumaResidual::EightByEight(luma) => Some(luma),
        }
    }

    pub(crate) fn chroma_cb_mut(&mut self) -> &mut [Block4x4; 4] {
        &mut self.data.chroma_cb
    }

    pub(crate) fn chroma_cr_mut(&mut self) -> &mut [Block4x4; 4] {
        &mut self.data.chroma_cr
    }
}

/// Applies the 8-bit 4x4 or 8x8 inverse transform pipeline for one intra
/// macroblock, including CAVLC coefficient interleaving for 8x8 blocks.
pub fn reconstruct_intra_residual(
    header: &IntraMacroblockHeader,
    residual: &IntraResidual,
    quantizer: MacroblockQuantizer,
    scaling_lists: &ResolvedScalingLists4x4,
    scaling_lists_8x8: &ResolvedScalingLists8x8,
    scan_mode: ScanMode,
) -> Result<ReconstructedIntraResidual> {
    if quantizer.transform_bypass {
        return Err(H264Error::UnsupportedFeature(
            "transform-bypass macroblock reconstruction",
        ));
    }
    let luma_scaling = scaling_lists.get(PredictionClass::Intra, ColorComponent::Luma);
    let luma = match header.luma_prediction {
        IntraLumaPrediction::SixteenBySixteen { .. } => {
            let mut luma = [[[0; 4]; 4]; 16];
            let prepared_scale = PreparedInverseScale4x4::new(quantizer.luma, luma_scaling)?;
            let dc = residual.luma_dc.as_ref().ok_or(H264Error::InvalidSyntax(
                "Intra16x16 residual is missing its luma DC block",
            ))?;
            ensure_block_size(dc.max_num_coeff, 16)?;
            let transformed_dc = inverse_transform_luma_dc_4x4(
                &dc.coefficients,
                scan_mode,
                quantizer.luma,
                luma_scaling,
            )?;
            for (index, block) in residual.luma.iter().enumerate() {
                ensure_block_size(block.max_num_coeff, 15)?;
                let (block_x, block_y) = LUMA_BLOCK_COORDINATES[index];
                let coefficients =
                    merge_dc_and_ac(transformed_dc[block_y][block_x], &block.coefficients);
                prepared_scale.reconstruct_into(
                    &coefficients,
                    scan_mode,
                    true,
                    &mut luma[index],
                )?;
            }
            ReconstructedLumaResidual::FourByFour(Box::new(luma))
        }
        IntraLumaPrediction::FourByFour(_) => {
            let mut luma = [[[0; 4]; 4]; 16];
            let prepared_scale = PreparedInverseScale4x4::new(quantizer.luma, luma_scaling)?;
            if residual.luma_dc.is_some() {
                return Err(H264Error::InvalidSyntax(
                    "Intra4x4 residual unexpectedly contains luma DC",
                ));
            }
            for (output, block) in luma.iter_mut().zip(&residual.luma) {
                ensure_block_size(block.max_num_coeff, 16)?;
                prepared_scale.reconstruct_into(&block.coefficients, scan_mode, false, output)?;
            }
            ReconstructedLumaResidual::FourByFour(Box::new(luma))
        }
        IntraLumaPrediction::EightByEight(_) => {
            if residual.luma_dc.is_some() {
                return Err(H264Error::InvalidSyntax(
                    "Intra8x8 residual unexpectedly contains luma DC",
                ));
            }
            let mut luma = [[[0; 8]; 8]; 4];
            for (block_8x8, output) in luma.iter_mut().enumerate() {
                let mut coefficients = [0; 64];
                for block_4x4 in 0..4 {
                    let source = &residual.luma[block_8x8 * 4 + block_4x4];
                    ensure_block_size(source.max_num_coeff, 16)?;
                    for index in 0..16 {
                        coefficients[4 * index + block_4x4] = source.coefficients[index];
                    }
                }
                *output = reconstruct_residual_8x8(
                    &coefficients,
                    scan_mode,
                    quantizer.luma,
                    scaling_lists_8x8.get(PredictionClass::Intra),
                )?;
            }
            ReconstructedLumaResidual::EightByEight(Box::new(luma))
        }
    };

    let chroma_cb = reconstruct_chroma(
        &residual.chroma_dc[0],
        &residual.chroma_ac[0],
        quantizer.chroma_cb,
        scaling_lists.get(PredictionClass::Intra, ColorComponent::Cb),
        scan_mode,
    )?;
    let chroma_cr = reconstruct_chroma(
        &residual.chroma_dc[1],
        &residual.chroma_ac[1],
        quantizer.chroma_cr,
        scaling_lists.get(PredictionClass::Intra, ColorComponent::Cr),
        scan_mode,
    )?;

    Ok(ReconstructedIntraResidual {
        luma,
        chroma_cb,
        chroma_cr,
    })
}

/// Applies the inverse transform pipeline for one progressive 8-bit 4:2:0
/// inter macroblock.
pub fn reconstruct_inter_residual(
    header: &PInterMacroblockHeader,
    residual: &InterResidual,
    quantizer: MacroblockQuantizer,
    scaling_lists: &ResolvedScalingLists4x4,
    scaling_lists_8x8: &ResolvedScalingLists8x8,
    scan_mode: ScanMode,
) -> Result<ReconstructedInterResidual> {
    reconstruct_inter_residual_with_transform_size(
        header.transform_size_8x8,
        residual,
        quantizer,
        scaling_lists,
        scaling_lists_8x8,
        scan_mode,
    )
}

/// Applies the inverse transform pipeline for one progressive 8-bit 4:2:0
/// B-slice inter macroblock.
///
/// P and B inter prediction share the same transform and quantization
/// processes; only their prediction samples come from different lists.
pub fn reconstruct_b_inter_residual(
    header: &BInterMacroblockHeader,
    residual: &InterResidual,
    quantizer: MacroblockQuantizer,
    scaling_lists: &ResolvedScalingLists4x4,
    scaling_lists_8x8: &ResolvedScalingLists8x8,
    scan_mode: ScanMode,
) -> Result<ReconstructedInterResidual> {
    reconstruct_inter_residual_with_transform_size(
        header.transform_size_8x8,
        residual,
        quantizer,
        scaling_lists,
        scaling_lists_8x8,
        scan_mode,
    )
}

fn reconstruct_inter_residual_with_transform_size(
    transform_size_8x8: bool,
    residual: &InterResidual,
    quantizer: MacroblockQuantizer,
    scaling_lists: &ResolvedScalingLists4x4,
    scaling_lists_8x8: &ResolvedScalingLists8x8,
    scan_mode: ScanMode,
) -> Result<ReconstructedInterResidual> {
    if quantizer.transform_bypass {
        return Err(H264Error::UnsupportedFeature(
            "transform-bypass macroblock reconstruction",
        ));
    }
    let mut output = if transform_size_8x8 {
        ReconstructedInterResidual::eight_by_eight(
            [[[0; 8]; 8]; 4],
            [[[0; 4]; 4]; 4],
            [[[0; 4]; 4]; 4],
        )
    } else {
        ReconstructedInterResidual::four_by_four(
            [[[0; 4]; 4]; 16],
            [[[0; 4]; 4]; 4],
            [[[0; 4]; 4]; 4],
        )
    };
    if transform_size_8x8 {
        let luma = output
            .luma_8x8_mut()
            .expect("the output transform size was selected above");
        for (block_8x8, block_output) in luma.iter_mut().enumerate() {
            let source_blocks = &residual.luma[block_8x8 * 4..block_8x8 * 4 + 4];
            for source in source_blocks {
                ensure_block_size(source.max_num_coeff, 16)?;
            }
            if source_blocks.iter().all(|block| block.total_coeff == 0) {
                continue;
            }
            let mut coefficients = [0; 64];
            for block_4x4 in 0..4 {
                let source = &source_blocks[block_4x4];
                for index in 0..16 {
                    coefficients[4 * index + block_4x4] = source.coefficients[index];
                }
            }
            *block_output = reconstruct_residual_8x8(
                &coefficients,
                scan_mode,
                quantizer.luma,
                scaling_lists_8x8.get(PredictionClass::Inter),
            )?;
        }
    } else {
        let luma = output
            .luma_4x4_mut()
            .expect("the output transform size was selected above");
        let scaling = scaling_lists.get(PredictionClass::Inter, ColorComponent::Luma);
        let prepared_scale = PreparedInverseScale4x4::new(quantizer.luma, scaling)?;
        for (block_output, block) in luma.iter_mut().zip(&residual.luma) {
            ensure_block_size(block.max_num_coeff, 16)?;
            if block.total_coeff == 0 {
                continue;
            }
            prepared_scale.reconstruct_into(&block.coefficients, scan_mode, false, block_output)?;
        }
    }
    reconstruct_chroma_into(
        &residual.chroma_dc[0],
        &residual.chroma_ac[0],
        quantizer.chroma_cb,
        scaling_lists.get(PredictionClass::Inter, ColorComponent::Cb),
        scan_mode,
        output.chroma_cb_mut(),
    )?;
    reconstruct_chroma_into(
        &residual.chroma_dc[1],
        &residual.chroma_ac[1],
        quantizer.chroma_cr,
        scaling_lists.get(PredictionClass::Inter, ColorComponent::Cr),
        scan_mode,
        output.chroma_cr_mut(),
    )?;
    Ok(output)
}

fn reconstruct_chroma(
    dc: &crate::ResidualBlock,
    ac: &[crate::ResidualBlock; 4],
    qp: u8,
    scaling_list: &[u8; 16],
    scan_mode: ScanMode,
) -> Result<[Block4x4; 4]> {
    let mut output = [[[0; 4]; 4]; 4];
    reconstruct_chroma_into(dc, ac, qp, scaling_list, scan_mode, &mut output)?;
    Ok(output)
}

fn reconstruct_chroma_into(
    dc: &crate::ResidualBlock,
    ac: &[crate::ResidualBlock; 4],
    qp: u8,
    scaling_list: &[u8; 16],
    scan_mode: ScanMode,
    output: &mut [Block4x4; 4],
) -> Result<()> {
    ensure_block_size(dc.max_num_coeff, 4)?;
    let dc_values: [i32; 4] = dc.coefficients[..4]
        .try_into()
        .expect("the source slice has a fixed length");
    let transformed_dc = inverse_transform_chroma_dc_420(&dc_values, qp, scaling_list)?;
    let prepared_scale = PreparedInverseScale4x4::new(qp, scaling_list)?;
    for (index, block) in ac.iter().enumerate() {
        ensure_block_size(block.max_num_coeff, 15)?;
        let (block_x, block_y) = CHROMA_BLOCK_COORDINATES[index];
        let dc = transformed_dc[block_y][block_x];
        if dc == 0 && block.total_coeff == 0 {
            continue;
        }
        let coefficients = merge_dc_and_ac(dc, &block.coefficients);
        prepared_scale.reconstruct_into(&coefficients, scan_mode, true, &mut output[index])?;
    }
    Ok(())
}

fn merge_dc_and_ac(dc: i32, ac: &[i32; 16]) -> [i32; 16] {
    let mut coefficients = [0; 16];
    coefficients[0] = dc;
    coefficients[1..].copy_from_slice(&ac[..15]);
    coefficients
}

fn ensure_block_size(actual: u8, expected: u8) -> Result<()> {
    if actual != expected {
        return Err(H264Error::InvalidSyntax(
            "residual block size does not match its transform path",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ReconstructedIntraResidual, ReconstructedLumaResidual};
    use crate::{
        BInterMacroblockHeader, BPartitionMode, BPartitionMotion, BPredictionMode,
        CodedBlockPattern, FLAT_SCALING_LIST_8X8, H264Error, InterResidual, IntraLumaPrediction,
        IntraMacroblockHeader, IntraPredictionModeSyntax, IntraResidual, MacroblockQuantizer,
        MotionVectorDifference, PInterMacroblockHeader, PPartitionMode, PPartitionMotion,
        ResidualBlock, ScanMode, reconstruct_residual_4x4, reconstruct_residual_8x8,
        resolve_scaling_lists_4x4, resolve_scaling_lists_8x8,
    };

    fn reconstruct_intra_residual(
        header: &IntraMacroblockHeader,
        residual: &IntraResidual,
        quantizer: MacroblockQuantizer,
        scaling_lists: &crate::ResolvedScalingLists4x4,
        scan_mode: ScanMode,
    ) -> crate::Result<ReconstructedIntraResidual> {
        let scaling_lists_8x8 = resolve_scaling_lists_8x8(None, None)?;
        super::reconstruct_intra_residual(
            header,
            residual,
            quantizer,
            scaling_lists,
            &scaling_lists_8x8,
            scan_mode,
        )
    }

    fn luma_4x4(residual: &ReconstructedIntraResidual) -> &[[[i32; 4]; 4]; 16] {
        let ReconstructedLumaResidual::FourByFour(blocks) = &residual.luma else {
            panic!("expected 4x4 luma residual");
        };
        blocks
    }

    fn header(luma_prediction: IntraLumaPrediction) -> IntraMacroblockHeader {
        IntraMacroblockHeader {
            luma_prediction,
            chroma_prediction_mode: 0,
            coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
            qp_delta: 0,
        }
    }

    fn residual(luma_block_size: u8, luma_dc: Option<ResidualBlock>) -> IntraResidual {
        IntraResidual {
            luma_dc,
            luma: [ResidualBlock::empty(luma_block_size); 16],
            chroma_dc: [ResidualBlock::empty(4); 2],
            chroma_ac: [[ResidualBlock::empty(15); 4]; 2],
        }
    }

    fn quantizer() -> MacroblockQuantizer {
        MacroblockQuantizer {
            luma: 0,
            chroma_cb: 0,
            chroma_cr: 0,
            transform_bypass: false,
        }
    }

    #[test]
    fn reconstructs_zero_and_impulse_4x4_residuals() {
        let header = header(IntraLumaPrediction::FourByFour(
            [IntraPredictionModeSyntax {
                use_predicted: true,
                remaining_mode: None,
            }; 16],
        ));
        let scaling = resolve_scaling_lists_4x4(None, None).unwrap();
        let zero = reconstruct_intra_residual(
            &header,
            &residual(16, None),
            quantizer(),
            &scaling,
            ScanMode::Frame,
        )
        .unwrap();
        assert_eq!(
            zero,
            ReconstructedIntraResidual {
                luma: ReconstructedLumaResidual::FourByFour(Box::new([[[0; 4]; 4]; 16])),
                chroma_cb: [[[0; 4]; 4]; 4],
                chroma_cr: [[[0; 4]; 4]; 4],
            }
        );

        let mut impulse = residual(16, None);
        impulse.luma[0].coefficients[0] = 64;
        impulse.luma[0].total_coeff = 1;
        let reconstructed =
            reconstruct_intra_residual(&header, &impulse, quantizer(), &scaling, ScanMode::Frame)
                .unwrap();
        assert_eq!(luma_4x4(&reconstructed)[0], [[10; 4]; 4]);
        assert_eq!(luma_4x4(&reconstructed)[1], [[0; 4]; 4]);
    }

    #[test]
    fn transforms_intra16_luma_dc_before_each_ac_block() {
        let header = header(IntraLumaPrediction::SixteenBySixteen { mode: 2 });
        let scaling = resolve_scaling_lists_4x4(None, None).unwrap();
        let mut coefficients = residual(15, Some(ResidualBlock::empty(16)));
        coefficients.luma_dc.as_mut().unwrap().coefficients[0] = 64;
        let reconstructed = reconstruct_intra_residual(
            &header,
            &coefficients,
            quantizer(),
            &scaling,
            ScanMode::Frame,
        )
        .unwrap();
        assert!(
            luma_4x4(&reconstructed)
                .iter()
                .all(|block| block == &[[3; 4]; 4])
        );

        let mut ac_only = residual(15, Some(ResidualBlock::empty(16)));
        ac_only.luma[0].coefficients[0] = 64;
        let reconstructed =
            reconstruct_intra_residual(&header, &ac_only, quantizer(), &scaling, ScanMode::Frame)
                .unwrap();
        let mut merged = [0; 16];
        merged[1] = 64;
        let expected = reconstruct_residual_4x4(
            &merged,
            ScanMode::Frame,
            0,
            scaling.get(crate::PredictionClass::Intra, crate::ColorComponent::Luma),
            true,
        )
        .unwrap();
        assert_eq!(luma_4x4(&reconstructed)[0], expected);
    }

    #[test]
    fn transforms_chroma_dc_and_keeps_components_independent() {
        let header = header(IntraLumaPrediction::FourByFour(
            [IntraPredictionModeSyntax {
                use_predicted: true,
                remaining_mode: None,
            }; 16],
        ));
        let scaling = resolve_scaling_lists_4x4(None, None).unwrap();
        let mut coefficients = residual(16, None);
        coefficients.chroma_dc[0].coefficients[0] = 64;
        let reconstructed = reconstruct_intra_residual(
            &header,
            &coefficients,
            quantizer(),
            &scaling,
            ScanMode::Frame,
        )
        .unwrap();
        assert!(
            reconstructed
                .chroma_cb
                .iter()
                .all(|block| block == &[[5; 4]; 4])
        );
        assert!(
            reconstructed
                .chroma_cr
                .iter()
                .all(|block| block == &[[0; 4]; 4])
        );
    }

    #[test]
    fn reconstructs_interleaved_cavlc_8x8_coefficients() {
        let scaling = resolve_scaling_lists_4x4(None, None).unwrap();
        let eight = header(IntraLumaPrediction::EightByEight(
            [IntraPredictionModeSyntax {
                use_predicted: true,
                remaining_mode: None,
            }; 4],
        ));
        let mut coefficients = residual(16, None);
        coefficients.luma[3].coefficients[2] = 64;
        let reconstructed = reconstruct_intra_residual(
            &eight,
            &coefficients,
            quantizer(),
            &scaling,
            ScanMode::Frame,
        );
        let ReconstructedLumaResidual::EightByEight(blocks) = reconstructed.unwrap().luma else {
            panic!("expected 8x8 luma residual");
        };
        let mut interleaved = [0; 64];
        interleaved[11] = 64;
        assert_eq!(
            blocks[0],
            reconstruct_residual_8x8(&interleaved, ScanMode::Frame, 0, &FLAT_SCALING_LIST_8X8,)
                .unwrap()
        );
        assert!(blocks[1..].iter().all(|block| block == &[[0; 8]; 8]));
    }

    #[test]
    fn rejects_mismatched_or_unsupported_transform_paths() {
        let scaling = resolve_scaling_lists_4x4(None, None).unwrap();
        let four = header(IntraLumaPrediction::FourByFour(
            [IntraPredictionModeSyntax {
                use_predicted: true,
                remaining_mode: None,
            }; 16],
        ));
        assert_eq!(
            reconstruct_intra_residual(
                &four,
                &residual(15, None),
                quantizer(),
                &scaling,
                ScanMode::Frame,
            ),
            Err(H264Error::InvalidSyntax(
                "residual block size does not match its transform path"
            ))
        );

        let bypass = MacroblockQuantizer {
            transform_bypass: true,
            ..quantizer()
        };
        assert!(matches!(
            reconstruct_intra_residual(
                &four,
                &residual(16, None),
                bypass,
                &scaling,
                ScanMode::Frame,
            ),
            Err(H264Error::UnsupportedFeature(_))
        ));
    }

    #[test]
    fn reconstructed_inter_residual_is_stored_inline() {
        assert_eq!(
            std::mem::size_of::<super::ReconstructedInterResidual>(),
            std::mem::size_of::<super::ReconstructedInterResidualData>()
        );
    }

    #[test]
    fn reconstructs_inter_four_and_eight_by_eight_residuals() {
        let scaling = resolve_scaling_lists_4x4(None, None).unwrap();
        let scaling_8x8 = resolve_scaling_lists_8x8(None, None).unwrap();
        let mut residual = InterResidual {
            luma: [ResidualBlock::empty(16); 16],
            chroma_dc: [ResidualBlock::empty(4); 2],
            chroma_ac: [[ResidualBlock::empty(15); 4]; 2],
        };
        residual.luma[0].coefficients[0] = 64;
        residual.luma[0].total_coeff = 1;

        let four = super::reconstruct_inter_residual(
            &inter_header(false),
            &residual,
            quantizer(),
            &scaling,
            &scaling_8x8,
            ScanMode::Frame,
        )
        .unwrap();
        let four_blocks = four.luma_4x4().expect("expected 4x4 residual blocks");
        assert_eq!(four_blocks[0], [[10; 4]; 4]);

        let eight = super::reconstruct_inter_residual(
            &inter_header(true),
            &residual,
            quantizer(),
            &scaling,
            &scaling_8x8,
            ScanMode::Frame,
        )
        .unwrap();
        let eight_blocks = eight.luma_8x8().expect("expected 8x8 residual blocks");
        let mut interleaved = [0; 64];
        interleaved[0] = 64;
        assert_eq!(
            eight_blocks[0],
            reconstruct_residual_8x8(&interleaved, ScanMode::Frame, 0, &FLAT_SCALING_LIST_8X8)
                .unwrap()
        );

        let b = super::reconstruct_b_inter_residual(
            &b_inter_header(true),
            &residual,
            quantizer(),
            &scaling,
            &scaling_8x8,
            ScanMode::Frame,
        )
        .unwrap();
        assert_eq!(b, eight);
    }

    fn inter_header(transform_size_8x8: bool) -> PInterMacroblockHeader {
        PInterMacroblockHeader {
            partition_mode: PPartitionMode::L0_16x16,
            partitions: vec![PPartitionMotion {
                reference_index: 0,
                differences: Vec::new().into(),
            }]
            .into(),
            coded_block_pattern: CodedBlockPattern { luma: 1, chroma: 0 },
            transform_size_8x8,
            qp_delta: 0,
        }
    }

    fn b_inter_header(transform_size_8x8: bool) -> BInterMacroblockHeader {
        BInterMacroblockHeader {
            partition_mode: BPartitionMode::SixteenBySixteen,
            partitions: vec![BPartitionMotion {
                prediction: BPredictionMode::Bi,
                reference_index_l0: Some(0),
                reference_index_l1: Some(0),
                differences_l0: vec![MotionVectorDifference { x: 0, y: 0 }].into(),
                differences_l1: vec![MotionVectorDifference { x: 0, y: 0 }].into(),
            }]
            .into(),
            coded_block_pattern: CodedBlockPattern { luma: 1, chroma: 0 },
            transform_size_8x8,
            qp_delta: 0,
        }
    }
}
