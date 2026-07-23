//! Complete progressive B-macroblock assembly from CABAC syntax elements.

use crate::{
    BInterMacroblockHeader, BPartitionMode, BPartitionMotion, BPredictionMode, BSubMacroblockType,
    CabacBMacroblockType, CabacIntraMacroblockSyntax, CabacMacroblockState, CabacMacroblockSummary,
    CabacMotionPartition, CabacMotionSyntaxState, CabacResidualState, CabacSliceDecoder,
    CodedBlockPattern, DecodedBSliceMacroblock, DecodedIntraMacroblock, H264Error,
    IntraLumaPrediction, IntraMacroblock, Result, SliceType,
};

/// Slice-level inputs that affect one progressive CABAC B macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CabacBMacroblockContext {
    pub num_ref_idx_l0_active: u8,
    pub num_ref_idx_l1_active: u8,
    pub transform_8x8_mode_enabled: bool,
    pub direct_8x8_inference: bool,
    pub previous_qp_delta_nonzero: bool,
}

/// A decoded B macroblock before motion-vector prediction or reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CabacBMacroblock {
    Skip,
    Decoded(Box<DecodedBSliceMacroblock>),
}

impl CabacBMacroblock {
    pub fn qp_delta(&self) -> i8 {
        match self {
            Self::Skip => 0,
            Self::Decoded(decoded) => match decoded.as_ref() {
                DecodedBSliceMacroblock::Inter { header, .. } => header.qp_delta,
                DecodedBSliceMacroblock::Intra(decoded) => match &decoded.macroblock {
                    IntraMacroblock::Predicted(header) => header.qp_delta,
                    IntraMacroblock::Pcm(_) => 0,
                },
            },
        }
    }
}

/// One completed CABAC B macroblock and its slice termination flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CabacBMacroblockDecode {
    pub macroblock: CabacBMacroblock,
    pub end_of_slice: bool,
}

/// Macroblock-level and per-list motion syntax state for one CABAC B slice.
#[derive(Debug, Clone)]
pub struct CabacBMacroblockState {
    macroblocks: CabacMacroblockState,
    motion_l0: CabacMotionSyntaxState,
    motion_l1: CabacMotionSyntaxState,
}

impl CabacBMacroblockState {
    pub fn new(width_in_macroblocks: usize, height_in_macroblocks: usize) -> Result<Self> {
        Ok(Self {
            macroblocks: CabacMacroblockState::new(width_in_macroblocks, height_in_macroblocks)?,
            motion_l0: CabacMotionSyntaxState::new(width_in_macroblocks, height_in_macroblocks)?,
            motion_l1: CabacMotionSyntaxState::new(width_in_macroblocks, height_in_macroblocks)?,
        })
    }

    /// Decodes, assembles, and records one progressive 4:2:0 B macroblock.
    pub fn decode_macroblock(
        &mut self,
        cabac: &mut CabacSliceDecoder<'_>,
        residuals: &mut CabacResidualState,
        macroblock_address: usize,
        slice_id: u32,
        context: CabacBMacroblockContext,
    ) -> Result<CabacBMacroblockDecode> {
        if context.num_ref_idx_l0_active == 0 || context.num_ref_idx_l1_active == 0 {
            return Err(H264Error::InvalidSyntax(
                "CABAC B slice has an empty active reference list",
            ));
        }
        let motion_l0_snapshot = self.motion_l0.snapshot_macroblock(macroblock_address)?;
        let motion_l1_snapshot = self.motion_l1.snapshot_macroblock(macroblock_address)?;
        residuals.validate_macroblock(macroblock_address)?;
        let residual_snapshot = residuals.snapshot_macroblock(macroblock_address);
        match self.decode_macroblock_inner(cabac, residuals, macroblock_address, slice_id, context)
        {
            Ok(decoded) => Ok(decoded),
            Err(error) => {
                self.motion_l0
                    .restore_macroblock(macroblock_address, motion_l0_snapshot)?;
                self.motion_l1
                    .restore_macroblock(macroblock_address, motion_l1_snapshot)?;
                residuals.restore_macroblock(macroblock_address, residual_snapshot);
                Err(error)
            }
        }
    }

    fn decode_macroblock_inner(
        &mut self,
        cabac: &mut CabacSliceDecoder<'_>,
        residuals: &mut CabacResidualState,
        macroblock_address: usize,
        slice_id: u32,
        context: CabacBMacroblockContext,
    ) -> Result<CabacBMacroblockDecode> {
        let skipped = {
            let mut syntax = cabac.syntax();
            self.macroblocks.decode_skip_flag(
                &mut syntax,
                macroblock_address,
                slice_id,
                SliceType::B,
            )?
        };
        if skipped {
            {
                let mut syntax = cabac.syntax();
                residuals.decode_inter_residual(
                    &mut syntax,
                    macroblock_address,
                    slice_id,
                    CodedBlockPattern { luma: 0, chroma: 0 },
                    false,
                )?;
            }
            self.motion_l0
                .record_direct_macroblock(macroblock_address, slice_id)?;
            self.motion_l1
                .record_direct_macroblock(macroblock_address, slice_id)?;
            let end_of_slice = decode_end_of_slice(cabac)?;
            self.macroblocks.record_macroblock(
                macroblock_address,
                slice_id,
                CabacMacroblockSummary {
                    skipped: true,
                    direct: true,
                    intra16_or_pcm: false,
                    intra_chroma_prediction: None,
                    coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
                    transform_size_8x8: false,
                },
            )?;
            return Ok(CabacBMacroblockDecode {
                macroblock: CabacBMacroblock::Skip,
                end_of_slice,
            });
        }

        let macroblock_type = {
            let mut syntax = cabac.syntax();
            self.macroblocks
                .decode_b_macroblock_type(&mut syntax, macroblock_address, slice_id)?
        };
        let (macroblock, summary) = match macroblock_type {
            CabacBMacroblockType::Intra(macroblock_type) => self.decode_intra_macroblock(
                cabac,
                residuals,
                macroblock_address,
                slice_id,
                macroblock_type,
                context,
            )?,
            CabacBMacroblockType::Inter(macroblock_type) => self.decode_inter_macroblock(
                cabac,
                residuals,
                macroblock_address,
                slice_id,
                macroblock_type,
                context,
            )?,
        };
        let end_of_slice = decode_end_of_slice(cabac)?;
        self.macroblocks
            .record_macroblock(macroblock_address, slice_id, summary)?;
        Ok(CabacBMacroblockDecode {
            macroblock: CabacBMacroblock::Decoded(Box::new(macroblock)),
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
        context: CabacBMacroblockContext,
    ) -> Result<(DecodedBSliceMacroblock, CabacMacroblockSummary)> {
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
        self.motion_l1
            .record_intra_macroblock(macroblock_address, slice_id)?;
        match syntax {
            CabacIntraMacroblockSyntax::Predicted(header) => {
                let residual = {
                    let mut syntax = cabac.syntax();
                    residuals.decode_intra_residual(
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
                    DecodedBSliceMacroblock::Intra(DecodedIntraMacroblock {
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
                    DecodedBSliceMacroblock::Intra(DecodedIntraMacroblock {
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
        macroblock_type: u8,
        context: CabacBMacroblockContext,
    ) -> Result<(DecodedBSliceMacroblock, CabacMacroblockSummary)> {
        let (partition_mode, plans, permits_transform_8x8) =
            self.decode_partition_plans(cabac, macroblock_type, context.direct_8x8_inference)?;
        self.initialize_inferred_list_states(macroblock_address, slice_id, &plans)?;

        let mut partitions = plans
            .iter()
            .map(|plan| BPartitionMotion {
                prediction: plan.prediction,
                reference_index_l0: None,
                reference_index_l1: None,
                differences_l0: Vec::new(),
                differences_l1: Vec::new(),
            })
            .collect::<Vec<_>>();

        {
            let mut syntax = cabac.syntax();
            for (plan, partition) in plans.iter().zip(&mut partitions) {
                if plan.prediction.uses_list0() {
                    partition.reference_index_l0 = Some(self.motion_l0.decode_reference_index(
                        &mut syntax,
                        macroblock_address,
                        slice_id,
                        plan.reference,
                        context.num_ref_idx_l0_active,
                        true,
                    )?);
                }
            }
        }
        {
            let mut syntax = cabac.syntax();
            for (plan, partition) in plans.iter().zip(&mut partitions) {
                if plan.prediction.uses_list1() {
                    partition.reference_index_l1 = Some(self.motion_l1.decode_reference_index(
                        &mut syntax,
                        macroblock_address,
                        slice_id,
                        plan.reference,
                        context.num_ref_idx_l1_active,
                        true,
                    )?);
                }
            }
        }
        {
            let mut syntax = cabac.syntax();
            for (plan, partition) in plans.iter().zip(&mut partitions) {
                if plan.prediction.uses_list0() {
                    for &motion in &plan.motion {
                        partition.differences_l0.push(
                            self.motion_l0.decode_motion_vector_difference(
                                &mut syntax,
                                macroblock_address,
                                slice_id,
                                motion,
                            )?,
                        );
                    }
                }
            }
        }
        {
            let mut syntax = cabac.syntax();
            for (plan, partition) in plans.iter().zip(&mut partitions) {
                if plan.prediction.uses_list1() {
                    for &motion in &plan.motion {
                        partition.differences_l1.push(
                            self.motion_l1.decode_motion_vector_difference(
                                &mut syntax,
                                macroblock_address,
                                slice_id,
                                motion,
                            )?,
                        );
                    }
                }
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
            residuals.decode_inter_residual(
                &mut syntax,
                macroblock_address,
                slice_id,
                coded_block_pattern,
                transform_size_8x8,
            )?
        };
        let header = BInterMacroblockHeader {
            partition_mode,
            partitions,
            coded_block_pattern,
            transform_size_8x8,
            qp_delta,
        };
        Ok((
            DecodedBSliceMacroblock::Inter { header, residual },
            CabacMacroblockSummary {
                skipped: false,
                direct: macroblock_type == 0,
                intra16_or_pcm: false,
                intra_chroma_prediction: None,
                coded_block_pattern,
                transform_size_8x8,
            },
        ))
    }

    fn initialize_inferred_list_states(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        plans: &[BMotionPartitionPlan],
    ) -> Result<()> {
        for plan in plans {
            match plan.prediction {
                BPredictionMode::Direct => {
                    self.motion_l0.record_direct_partition(
                        macroblock_address,
                        slice_id,
                        plan.reference,
                    )?;
                    self.motion_l1.record_direct_partition(
                        macroblock_address,
                        slice_id,
                        plan.reference,
                    )?;
                }
                BPredictionMode::List0 => self.motion_l1.record_unused_partition(
                    macroblock_address,
                    slice_id,
                    plan.reference,
                )?,
                BPredictionMode::List1 => self.motion_l0.record_unused_partition(
                    macroblock_address,
                    slice_id,
                    plan.reference,
                )?,
                BPredictionMode::Bi => {}
            }
        }
        Ok(())
    }

    fn decode_partition_plans(
        &self,
        cabac: &mut CabacSliceDecoder<'_>,
        macroblock_type: u8,
        direct_8x8_inference: bool,
    ) -> Result<(BPartitionMode, Vec<BMotionPartitionPlan>, bool)> {
        match macroblock_type {
            0..=21 => {
                let (mode, plans) = ordinary_b_partition_plans(macroblock_type);
                Ok((mode, plans, macroblock_type != 0 || direct_8x8_inference))
            }
            22 => {
                let mut sub_macroblocks = [BSubMacroblockType::Direct8x8; 4];
                {
                    let mut syntax = cabac.syntax();
                    for sub_type in &mut sub_macroblocks {
                        *sub_type = self.macroblocks.decode_b_sub_macroblock_type(&mut syntax)?;
                    }
                }
                let plans = b8x8_partition_plans(sub_macroblocks);
                let permits_transform_8x8 = sub_macroblocks.iter().all(|sub_type| {
                    sub_type.partition_size() == (8, 8)
                        && (*sub_type != BSubMacroblockType::Direct8x8 || direct_8x8_inference)
                });
                Ok((
                    BPartitionMode::EightByEight { sub_macroblocks },
                    plans,
                    permits_transform_8x8,
                ))
            }
            _ => Err(H264Error::InvalidSyntax(
                "CABAC B inter macroblock type exceeds 22",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BMotionPartitionPlan {
    reference: CabacMotionPartition,
    motion: Vec<CabacMotionPartition>,
    prediction: BPredictionMode,
}

fn single_partition_plan(
    x: u8,
    y: u8,
    width: u8,
    height: u8,
    prediction: BPredictionMode,
) -> BMotionPartitionPlan {
    let partition = CabacMotionPartition {
        x,
        y,
        width,
        height,
    };
    BMotionPartitionPlan {
        reference: partition,
        motion: if prediction == BPredictionMode::Direct {
            Vec::new()
        } else {
            vec![partition]
        },
        prediction,
    }
}

fn b8x8_partition_plans(sub_macroblocks: [BSubMacroblockType; 4]) -> Vec<BMotionPartitionPlan> {
    sub_macroblocks
        .into_iter()
        .enumerate()
        .map(|(index, sub_type)| {
            let base_x = u8::try_from(index % 2).expect("B sub-macroblock column fits") * 8;
            let base_y = u8::try_from(index / 2).expect("B sub-macroblock row fits") * 8;
            let reference = CabacMotionPartition {
                x: base_x,
                y: base_y,
                width: 8,
                height: 8,
            };
            let prediction = sub_type.prediction();
            let motion = if prediction == BPredictionMode::Direct {
                Vec::new()
            } else {
                let (width, height) = sub_type.partition_size();
                (0..sub_type.partition_count())
                    .map(|partition_index| CabacMotionPartition {
                        x: base_x
                            + u8::try_from(partition_index % usize::from(8 / width))
                                .expect("B sub-partition column fits")
                                * width,
                        y: base_y
                            + u8::try_from(partition_index / usize::from(8 / width))
                                .expect("B sub-partition row fits")
                                * height,
                        width,
                        height,
                    })
                    .collect()
            };
            BMotionPartitionPlan {
                reference,
                motion,
                prediction,
            }
        })
        .collect()
}

fn ordinary_b_partition_plans(macroblock_type: u8) -> (BPartitionMode, Vec<BMotionPartitionPlan>) {
    match macroblock_type {
        0 => (
            BPartitionMode::Direct16x16,
            vec![single_partition_plan(0, 0, 16, 16, BPredictionMode::Direct)],
        ),
        1..=3 => (
            BPartitionMode::SixteenBySixteen,
            vec![single_partition_plan(
                0,
                0,
                16,
                16,
                [
                    BPredictionMode::List0,
                    BPredictionMode::List1,
                    BPredictionMode::Bi,
                ][usize::from(macroblock_type - 1)],
            )],
        ),
        4..=21 => {
            let mode = if macroblock_type.is_multiple_of(2) {
                BPartitionMode::SixteenByEight
            } else {
                BPartitionMode::EightBySixteen
            };
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
            ][usize::from((macroblock_type - 4) / 2)];
            let plans = if matches!(mode, BPartitionMode::SixteenByEight) {
                vec![
                    single_partition_plan(0, 0, 16, 8, predictions[0]),
                    single_partition_plan(0, 8, 16, 8, predictions[1]),
                ]
            } else {
                vec![
                    single_partition_plan(0, 0, 8, 16, predictions[0]),
                    single_partition_plan(8, 0, 8, 16, predictions[1]),
                ]
            };
            (mode, plans)
        }
        _ => unreachable!("ordinary B macroblock type is in 0..=21"),
    }
}

fn decode_end_of_slice(cabac: &mut CabacSliceDecoder<'_>) -> Result<bool> {
    let mut syntax = cabac.syntax();
    Ok(syntax.terminate()? != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_all_ordinary_b_macroblock_layouts() {
        for macroblock_type in 0..=21 {
            let (mode, plans) = ordinary_b_partition_plans(macroblock_type);
            match macroblock_type {
                0 => {
                    assert_eq!(mode, BPartitionMode::Direct16x16);
                    assert_eq!(plans[0].prediction, BPredictionMode::Direct);
                }
                1..=3 => {
                    assert_eq!(mode, BPartitionMode::SixteenBySixteen);
                    assert_eq!(plans.len(), 1);
                }
                _ if macroblock_type.is_multiple_of(2) => {
                    assert_eq!(mode, BPartitionMode::SixteenByEight);
                    assert_eq!(plans.len(), 2);
                    assert!(plans.iter().all(|plan| plan.reference.width == 16));
                }
                _ => {
                    assert_eq!(mode, BPartitionMode::EightBySixteen);
                    assert_eq!(plans.len(), 2);
                    assert!(plans.iter().all(|plan| plan.reference.height == 16));
                }
            }
        }
    }

    #[test]
    fn expands_mixed_b_subpartitions_in_raster_order() {
        let plans = b8x8_partition_plans([
            BSubMacroblockType::Direct8x8,
            BSubMacroblockType::List0_8x4,
            BSubMacroblockType::List1_4x8,
            BSubMacroblockType::Bi4x4,
        ]);
        assert!(plans[0].motion.is_empty());
        assert_eq!(plans[1].motion.len(), 2);
        assert_eq!(plans[2].motion.len(), 2);
        assert_eq!(plans[3].motion.len(), 4);
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
}
