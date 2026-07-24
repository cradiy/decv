//! Complete progressive P-macroblock assembly from CABAC syntax elements.

use crate::{
    CabacIntraMacroblockSyntax, CabacMacroblockState, CabacMacroblockSummary, CabacMotionPartition,
    CabacMotionSyntaxState, CabacPMacroblockType, CabacResidualState, CabacSliceDecoder,
    CodedBlockPattern, DecodedIntraMacroblock, DecodedPSliceMacroblock, H264Error,
    IntraLumaPrediction, IntraMacroblock, PInterMacroblockHeader, PPartitionMode, PPartitionMotion,
    PSubMacroblockType, Result, SliceType,
};

/// Slice-level inputs that affect one progressive CABAC P macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CabacPMacroblockContext {
    pub num_ref_idx_l0_active: u8,
    pub transform_8x8_mode_enabled: bool,
    pub previous_qp_delta_nonzero: bool,
}

/// A decoded P macroblock before motion-vector prediction or pixel
/// reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CabacPMacroblock {
    Skip,
    Decoded(Box<DecodedPSliceMacroblock>),
}

impl CabacPMacroblock {
    /// Returns the explicit QP delta, or the inferred zero for P_Skip/I_PCM.
    pub fn qp_delta(&self) -> i8 {
        match self {
            Self::Skip => 0,
            Self::Decoded(decoded) => match decoded.as_ref() {
                DecodedPSliceMacroblock::Inter { header, .. } => header.qp_delta,
                DecodedPSliceMacroblock::Intra(decoded) => match &decoded.macroblock {
                    IntraMacroblock::Predicted(header) => header.qp_delta,
                    IntraMacroblock::Pcm(_) => 0,
                },
            },
        }
    }
}

/// One completed CABAC P macroblock and its slice termination flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CabacPMacroblockDecode {
    pub macroblock: CabacPMacroblock,
    pub end_of_slice: bool,
}

/// Macroblock-level and List-0 motion syntax state for one CABAC P slice.
#[derive(Debug, Clone)]
pub struct CabacPMacroblockState {
    macroblocks: CabacMacroblockState,
    motion_l0: CabacMotionSyntaxState,
}

impl CabacPMacroblockState {
    pub fn new(width_in_macroblocks: usize, height_in_macroblocks: usize) -> Result<Self> {
        Ok(Self {
            macroblocks: CabacMacroblockState::new(width_in_macroblocks, height_in_macroblocks)?,
            motion_l0: CabacMotionSyntaxState::new(width_in_macroblocks, height_in_macroblocks)?,
        })
    }

    /// Decodes, assembles, and records one progressive 4:2:0 P macroblock.
    ///
    /// Motion and residual neighbour state are transactional. Arithmetic
    /// decoding itself cannot be rewound, so any returned error remains fatal
    /// to the enclosing slice.
    pub fn decode_macroblock(
        &mut self,
        cabac: &mut CabacSliceDecoder<'_>,
        residuals: &mut CabacResidualState,
        macroblock_address: usize,
        slice_id: u32,
        context: CabacPMacroblockContext,
    ) -> Result<CabacPMacroblockDecode> {
        if context.num_ref_idx_l0_active == 0 {
            return Err(H264Error::InvalidSyntax(
                "CABAC P slice has no active List-0 references",
            ));
        }
        let motion_snapshot = self.motion_l0.snapshot_macroblock(macroblock_address)?;
        residuals.validate_macroblock(macroblock_address)?;
        let residual_snapshot = residuals.snapshot_macroblock(macroblock_address);
        match self.decode_macroblock_inner(cabac, residuals, macroblock_address, slice_id, context)
        {
            Ok(decoded) => Ok(decoded),
            Err(error) => {
                self.motion_l0
                    .restore_macroblock(macroblock_address, motion_snapshot)?;
                residuals.restore_macroblock(macroblock_address, residual_snapshot);
                Err(error)
            }
        }
    }

    /// Decoder-internal fast path for a picture that is discarded on error.
    pub(crate) fn decode_macroblock_terminal(
        &mut self,
        cabac: &mut CabacSliceDecoder<'_>,
        residuals: &mut CabacResidualState,
        macroblock_address: usize,
        slice_id: u32,
        context: CabacPMacroblockContext,
    ) -> Result<CabacPMacroblockDecode> {
        if context.num_ref_idx_l0_active == 0 {
            return Err(H264Error::InvalidSyntax(
                "CABAC P slice has no active List-0 references",
            ));
        }
        self.motion_l0.validate_macroblock(macroblock_address)?;
        residuals.validate_macroblock(macroblock_address)?;
        self.decode_macroblock_inner(cabac, residuals, macroblock_address, slice_id, context)
    }

    fn decode_macroblock_inner(
        &mut self,
        cabac: &mut CabacSliceDecoder<'_>,
        residuals: &mut CabacResidualState,
        macroblock_address: usize,
        slice_id: u32,
        context: CabacPMacroblockContext,
    ) -> Result<CabacPMacroblockDecode> {
        let skipped = {
            let mut syntax = cabac.syntax();
            self.macroblocks.decode_skip_flag(
                &mut syntax,
                macroblock_address,
                slice_id,
                SliceType::P,
            )?
        };
        if skipped {
            residuals.record_zero_inter_macroblock_terminal(macroblock_address, slice_id)?;
            self.motion_l0
                .record_skip_macroblock(macroblock_address, slice_id)?;
            let end_of_slice = decode_end_of_slice(cabac)?;
            self.macroblocks.record_macroblock(
                macroblock_address,
                slice_id,
                CabacMacroblockSummary {
                    skipped: true,
                    direct: false,
                    intra16_or_pcm: false,
                    intra_chroma_prediction: None,
                    coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
                    transform_size_8x8: false,
                },
            )?;
            return Ok(CabacPMacroblockDecode {
                macroblock: CabacPMacroblock::Skip,
                end_of_slice,
            });
        }

        let macroblock_type = {
            let mut syntax = cabac.syntax();
            self.macroblocks.decode_p_macroblock_type(&mut syntax)?
        };
        let (macroblock, summary) = match macroblock_type {
            CabacPMacroblockType::Intra(macroblock_type) => self.decode_intra_macroblock(
                cabac,
                residuals,
                macroblock_address,
                slice_id,
                macroblock_type,
                context,
            )?,
            inter_type => self.decode_inter_macroblock(
                cabac,
                residuals,
                macroblock_address,
                slice_id,
                inter_type,
                context,
            )?,
        };
        let end_of_slice = decode_end_of_slice(cabac)?;
        self.macroblocks
            .record_macroblock(macroblock_address, slice_id, summary)?;
        Ok(CabacPMacroblockDecode {
            macroblock: CabacPMacroblock::Decoded(Box::new(macroblock)),
            end_of_slice,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_intra_macroblock(
        &mut self,
        cabac: &mut CabacSliceDecoder<'_>,
        residuals: &mut CabacResidualState,
        macroblock_address: usize,
        slice_id: u32,
        macroblock_type: u8,
        context: CabacPMacroblockContext,
    ) -> Result<(DecodedPSliceMacroblock, CabacMacroblockSummary)> {
        let syntax = {
            let mut syntax = cabac.syntax();
            self.macroblocks.decode_intra_macroblock_syntax_for_type(
                &mut syntax,
                macroblock_address,
                slice_id,
                macroblock_type,
                context.transform_8x8_mode_enabled,
                context.previous_qp_delta_nonzero,
            )?
        };
        self.motion_l0
            .record_intra_macroblock(macroblock_address, slice_id)?;
        match syntax {
            CabacIntraMacroblockSyntax::Predicted(header) => {
                let residual = {
                    let mut syntax = cabac.syntax();
                    residuals.decode_intra_residual_terminal(
                        &mut syntax,
                        macroblock_address,
                        slice_id,
                        &header,
                    )?
                };
                let summary = CabacMacroblockSummary {
                    skipped: false,
                    direct: false,
                    intra16_or_pcm: matches!(
                        header.luma_prediction,
                        IntraLumaPrediction::SixteenBySixteen { .. }
                    ),
                    intra_chroma_prediction: Some(header.chroma_prediction_mode),
                    coded_block_pattern: header.coded_block_pattern,
                    transform_size_8x8: matches!(
                        header.luma_prediction,
                        IntraLumaPrediction::EightByEight(_)
                    ),
                };
                Ok((
                    DecodedPSliceMacroblock::Intra(DecodedIntraMacroblock {
                        macroblock: IntraMacroblock::Predicted(header),
                        residual: Some(residual),
                    }),
                    summary,
                ))
            }
            CabacIntraMacroblockSyntax::Pcm => {
                let pcm = cabac.decode_pcm_420_8bit_and_restart()?;
                residuals.record_pcm_macroblock(macroblock_address, slice_id)?;
                Ok((
                    DecodedPSliceMacroblock::Intra(DecodedIntraMacroblock {
                        macroblock: IntraMacroblock::Pcm(pcm),
                        residual: None,
                    }),
                    CabacMacroblockSummary {
                        skipped: false,
                        direct: false,
                        intra16_or_pcm: true,
                        intra_chroma_prediction: None,
                        coded_block_pattern: CodedBlockPattern {
                            luma: 15,
                            chroma: 2,
                        },
                        transform_size_8x8: false,
                    },
                ))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_inter_macroblock(
        &mut self,
        cabac: &mut CabacSliceDecoder<'_>,
        residuals: &mut CabacResidualState,
        macroblock_address: usize,
        slice_id: u32,
        macroblock_type: CabacPMacroblockType,
        context: CabacPMacroblockContext,
    ) -> Result<(DecodedPSliceMacroblock, CabacMacroblockSummary)> {
        let (partition_mode, plans, permits_transform_8x8) =
            self.decode_partition_plans(cabac, macroblock_type)?;

        let mut reference_indices = Vec::with_capacity(plans.len());
        {
            let mut syntax = cabac.syntax();
            for plan in &plans {
                reference_indices.push(self.motion_l0.decode_reference_index(
                    &mut syntax,
                    macroblock_address,
                    slice_id,
                    plan.reference,
                    context.num_ref_idx_l0_active,
                    false,
                )?);
            }
        }

        let mut partitions = Vec::with_capacity(plans.len());
        {
            let mut syntax = cabac.syntax();
            for (plan, reference_index) in plans.iter().zip(reference_indices) {
                let mut differences = Vec::with_capacity(plan.motion.len());
                for &partition in &plan.motion {
                    differences.push(self.motion_l0.decode_motion_vector_difference(
                        &mut syntax,
                        macroblock_address,
                        slice_id,
                        partition,
                    )?);
                }
                partitions.push(PPartitionMotion {
                    reference_index,
                    differences,
                });
            }
        }

        let coded_block_pattern = {
            let mut syntax = cabac.syntax();
            self.macroblocks.decode_coded_block_pattern(
                &mut syntax,
                macroblock_address,
                slice_id,
            )?
        };
        let transform_size_8x8 = if coded_block_pattern.luma != 0
            && context.transform_8x8_mode_enabled
            && permits_transform_8x8
        {
            let mut syntax = cabac.syntax();
            self.macroblocks.decode_transform_size_8x8_flag(
                &mut syntax,
                macroblock_address,
                slice_id,
            )?
        } else {
            false
        };
        let qp_delta = if coded_block_pattern.has_residual() {
            let mut syntax = cabac.syntax();
            syntax.macroblock_qp_delta(context.previous_qp_delta_nonzero)?
        } else {
            0
        };
        let residual = {
            let mut syntax = cabac.syntax();
            residuals.decode_inter_residual_terminal(
                &mut syntax,
                macroblock_address,
                slice_id,
                coded_block_pattern,
                transform_size_8x8,
            )?
        };
        let header = PInterMacroblockHeader {
            partition_mode,
            partitions,
            coded_block_pattern,
            transform_size_8x8,
            qp_delta,
        };
        Ok((
            DecodedPSliceMacroblock::Inter { header, residual },
            CabacMacroblockSummary {
                skipped: false,
                direct: false,
                intra16_or_pcm: false,
                intra_chroma_prediction: None,
                coded_block_pattern,
                transform_size_8x8,
            },
        ))
    }

    fn decode_partition_plans(
        &self,
        cabac: &mut CabacSliceDecoder<'_>,
        macroblock_type: CabacPMacroblockType,
    ) -> Result<(PPartitionMode, Vec<MotionPartitionPlan>, bool)> {
        match macroblock_type {
            CabacPMacroblockType::L0_16x16 => Ok((
                PPartitionMode::L0_16x16,
                vec![single_partition_plan(0, 0, 16, 16)],
                true,
            )),
            CabacPMacroblockType::L0_16x8 => Ok((
                PPartitionMode::L0_16x8,
                vec![
                    single_partition_plan(0, 0, 16, 8),
                    single_partition_plan(0, 8, 16, 8),
                ],
                true,
            )),
            CabacPMacroblockType::L0_8x16 => Ok((
                PPartitionMode::L0_8x16,
                vec![
                    single_partition_plan(0, 0, 8, 16),
                    single_partition_plan(8, 0, 8, 16),
                ],
                true,
            )),
            CabacPMacroblockType::EightByEight => {
                let mut sub_macroblocks = [PSubMacroblockType::L0_8x8; 4];
                {
                    let mut syntax = cabac.syntax();
                    for sub_type in &mut sub_macroblocks {
                        *sub_type = self.macroblocks.decode_p_sub_macroblock_type(&mut syntax)?;
                    }
                }
                let plans = p8x8_partition_plans(sub_macroblocks);
                let permits_transform_8x8 = sub_macroblocks
                    .iter()
                    .all(|sub_type| *sub_type == PSubMacroblockType::L0_8x8);
                Ok((
                    PPartitionMode::L0_8x8 {
                        sub_macroblocks,
                        reference_index_forced_zero: false,
                    },
                    plans,
                    permits_transform_8x8,
                ))
            }
            CabacPMacroblockType::Intra(_) => Err(H264Error::InvalidSyntax(
                "intra type passed to CABAC P inter assembler",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MotionPartitionPlan {
    reference: CabacMotionPartition,
    motion: Vec<CabacMotionPartition>,
}

fn single_partition_plan(x: u8, y: u8, width: u8, height: u8) -> MotionPartitionPlan {
    let partition = CabacMotionPartition {
        x,
        y,
        width,
        height,
    };
    MotionPartitionPlan {
        reference: partition,
        motion: vec![partition],
    }
}

fn p8x8_partition_plans(sub_macroblocks: [PSubMacroblockType; 4]) -> Vec<MotionPartitionPlan> {
    sub_macroblocks
        .into_iter()
        .enumerate()
        .map(|(index, sub_type)| {
            let base_x = u8::try_from(index % 2).expect("P sub-macroblock column fits") * 8;
            let base_y = u8::try_from(index / 2).expect("P sub-macroblock row fits") * 8;
            let reference = CabacMotionPartition {
                x: base_x,
                y: base_y,
                width: 8,
                height: 8,
            };
            let (width, height) = sub_type.partition_size();
            let motion = (0..sub_type.partition_count())
                .map(|partition_index| CabacMotionPartition {
                    x: base_x
                        + u8::try_from(partition_index % usize::from(8 / width))
                            .expect("P sub-partition column fits")
                            * width,
                    y: base_y
                        + u8::try_from(partition_index / usize::from(8 / width))
                            .expect("P sub-partition row fits")
                            * height,
                    width,
                    height,
                })
                .collect();
            MotionPartitionPlan { reference, motion }
        })
        .collect()
}

fn decode_end_of_slice(cabac: &mut CabacSliceDecoder<'_>) -> Result<bool> {
    let mut syntax = cabac.syntax();
    Ok(syntax.terminate()? != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_p8x8_subpartitions_in_raster_syntax_order() {
        let plans = p8x8_partition_plans([
            PSubMacroblockType::L0_8x8,
            PSubMacroblockType::L0_8x4,
            PSubMacroblockType::L0_4x8,
            PSubMacroblockType::L0_4x4,
        ]);
        assert_eq!(
            plans.iter().map(|plan| plan.reference).collect::<Vec<_>>(),
            [
                CabacMotionPartition {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                },
                CabacMotionPartition {
                    x: 8,
                    y: 0,
                    width: 8,
                    height: 8,
                },
                CabacMotionPartition {
                    x: 0,
                    y: 8,
                    width: 8,
                    height: 8,
                },
                CabacMotionPartition {
                    x: 8,
                    y: 8,
                    width: 8,
                    height: 8,
                },
            ]
        );
        assert_eq!(plans[0].motion, [plans[0].reference]);
        assert_eq!(
            plans[1].motion,
            [
                CabacMotionPartition {
                    x: 8,
                    y: 0,
                    width: 8,
                    height: 4,
                },
                CabacMotionPartition {
                    x: 8,
                    y: 4,
                    width: 8,
                    height: 4,
                },
            ]
        );
        assert_eq!(
            plans[2].motion,
            [
                CabacMotionPartition {
                    x: 0,
                    y: 8,
                    width: 4,
                    height: 8,
                },
                CabacMotionPartition {
                    x: 4,
                    y: 8,
                    width: 4,
                    height: 8,
                },
            ]
        );
        assert_eq!(
            plans[3].motion,
            [
                CabacMotionPartition {
                    x: 8,
                    y: 8,
                    width: 4,
                    height: 4,
                },
                CabacMotionPartition {
                    x: 12,
                    y: 8,
                    width: 4,
                    height: 4,
                },
                CabacMotionPartition {
                    x: 8,
                    y: 12,
                    width: 4,
                    height: 4,
                },
                CabacMotionPartition {
                    x: 12,
                    y: 12,
                    width: 4,
                    height: 4,
                },
            ]
        );
    }

    #[test]
    fn reports_inferred_and_explicit_qp_deltas() {
        assert_eq!(CabacPMacroblock::Skip.qp_delta(), 0);
        let decoded = CabacPMacroblock::Decoded(Box::new(DecodedPSliceMacroblock::Intra(
            DecodedIntraMacroblock {
                macroblock: IntraMacroblock::Pcm(crate::PcmMacroblock {
                    luma: Box::new([0; 256]),
                    chroma: Box::new([0; 128]),
                }),
                residual: None,
            },
        )));
        assert_eq!(decoded.qp_delta(), 0);
    }
}
