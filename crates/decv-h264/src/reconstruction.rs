//! Conversion from decoded transform coefficients to spatial residual samples.

use crate::{
    Block4x4, ColorComponent, H264Error, IntraLumaPrediction, IntraMacroblockHeader, IntraResidual,
    MacroblockQuantizer, PredictionClass, ResolvedScalingLists4x4, Result, ScanMode,
    inverse_transform_chroma_dc_420, inverse_transform_luma_dc_4x4, reconstruct_residual_4x4,
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
pub struct ReconstructedIntraResidual {
    pub luma: [Block4x4; 16],
    pub chroma_cb: [Block4x4; 4],
    pub chroma_cr: [Block4x4; 4],
}

/// Applies the 8-bit 4x4 inverse-scan, inverse-scaling, and inverse-transform
/// pipeline for one intra macroblock.
pub fn reconstruct_intra_residual(
    header: &IntraMacroblockHeader,
    residual: &IntraResidual,
    quantizer: MacroblockQuantizer,
    scaling_lists: &ResolvedScalingLists4x4,
    scan_mode: ScanMode,
) -> Result<ReconstructedIntraResidual> {
    if quantizer.transform_bypass {
        return Err(H264Error::UnsupportedFeature(
            "transform-bypass macroblock reconstruction",
        ));
    }
    if matches!(header.luma_prediction, IntraLumaPrediction::EightByEight(_)) {
        return Err(H264Error::UnsupportedFeature(
            "8x8 inverse transform reconstruction",
        ));
    }

    let luma_scaling = scaling_lists.get(PredictionClass::Intra, ColorComponent::Luma);
    let mut luma = [[[0; 4]; 4]; 16];
    match header.luma_prediction {
        IntraLumaPrediction::SixteenBySixteen { .. } => {
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
                luma[index] = reconstruct_residual_4x4(
                    &coefficients,
                    scan_mode,
                    quantizer.luma,
                    luma_scaling,
                    true,
                )?;
            }
        }
        IntraLumaPrediction::FourByFour(_) => {
            if residual.luma_dc.is_some() {
                return Err(H264Error::InvalidSyntax(
                    "Intra4x4 residual unexpectedly contains luma DC",
                ));
            }
            for (output, block) in luma.iter_mut().zip(&residual.luma) {
                ensure_block_size(block.max_num_coeff, 16)?;
                *output = reconstruct_residual_4x4(
                    &block.coefficients,
                    scan_mode,
                    quantizer.luma,
                    luma_scaling,
                    false,
                )?;
            }
        }
        IntraLumaPrediction::EightByEight(_) => unreachable!("rejected above"),
    }

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

fn reconstruct_chroma(
    dc: &crate::ResidualBlock,
    ac: &[crate::ResidualBlock; 4],
    qp: u8,
    scaling_list: &[u8; 16],
    scan_mode: ScanMode,
) -> Result<[Block4x4; 4]> {
    ensure_block_size(dc.max_num_coeff, 4)?;
    let dc_values: [i32; 4] = dc.coefficients[..4]
        .try_into()
        .expect("the source slice has a fixed length");
    let transformed_dc = inverse_transform_chroma_dc_420(&dc_values, qp, scaling_list)?;
    let mut output = [[[0; 4]; 4]; 4];
    for (index, block) in ac.iter().enumerate() {
        ensure_block_size(block.max_num_coeff, 15)?;
        let (block_x, block_y) = CHROMA_BLOCK_COORDINATES[index];
        let coefficients = merge_dc_and_ac(transformed_dc[block_y][block_x], &block.coefficients);
        output[index] = reconstruct_residual_4x4(&coefficients, scan_mode, qp, scaling_list, true)?;
    }
    Ok(output)
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
    use super::{ReconstructedIntraResidual, reconstruct_intra_residual};
    use crate::{
        CodedBlockPattern, H264Error, IntraLumaPrediction, IntraMacroblockHeader,
        IntraPredictionModeSyntax, IntraResidual, MacroblockQuantizer, ResidualBlock, ScanMode,
        reconstruct_residual_4x4, resolve_scaling_lists_4x4,
    };

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
                luma: [[[0; 4]; 4]; 16],
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
        assert_eq!(reconstructed.luma[0], [[10; 4]; 4]);
        assert_eq!(reconstructed.luma[1], [[0; 4]; 4]);
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
        assert!(reconstructed.luma.iter().all(|block| block == &[[3; 4]; 4]));

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
        assert_eq!(reconstructed.luma[0], expected);
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

        let eight = header(IntraLumaPrediction::EightByEight(
            [IntraPredictionModeSyntax {
                use_predicted: true,
                remaining_mode: None,
            }; 4],
        ));
        assert!(matches!(
            reconstruct_intra_residual(
                &eight,
                &residual(16, None),
                quantizer(),
                &scaling,
                ScanMode::Frame,
            ),
            Err(H264Error::UnsupportedFeature(_))
        ));

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
}
