//! Stateful reconstruction of progressive 8-bit 4:2:0 intra pictures.

use bit_readers::BitReader;
use decv_core::{DecodedVideoFrame, MediaTime, Size, VideoFormat};
use rayon::prelude::*;

use crate::deblock::{DeblockListMotion, DeblockMotion, MacroblockDeblockInfo, filter_420_picture};
use crate::inter_reconstruction::{
    BPredictionWeightMode, ImplicitWeightReference, reconstruct_b_macroblock_from_lists_with_mode,
    reconstruct_b_macroblock_pixels_from_lists_into_with_scratch,
    reconstruct_b_macroblock_pixels_from_lists_with_scratch,
    reconstruct_p_macroblock_pixels_from_list_into_with_scratch,
    reconstruct_p_macroblock_pixels_from_list_with_scratch,
    reconstruct_p_skip_macroblock_from_list_420, reconstruct_weighted_p_macroblock_from_list_420,
    reconstruct_weighted_p_skip_macroblock_from_list_420,
};
use crate::motion_field::MotionFieldBuilder;
use crate::parallelism::ReconstructionExecutor;
use crate::picture_surface::StagedMacroblockPixels;
use crate::rbsp::more_rbsp_data;
use crate::{
    ActiveParameterSets, BMacroblockContext, BMotionState, BPartitionMode, BSubMacroblockType,
    CabacBMacroblock, CabacBMacroblockContext, CabacBMacroblockState, CabacIntraMacroblockSyntax,
    CabacMacroblockState, CabacMacroblockSummary, CabacPMacroblock, CabacPMacroblockContext,
    CabacPMacroblockState, CabacResidualState, CabacSliceDecoder, CavlcNeighborState, ChromaPlane,
    DeblockingFilter, DecodedBSliceMacroblock, DecodedIntraMacroblock, DecodedPSliceMacroblock,
    DirectMotionContext, DirectReference, EntropyCodingMode, H264Error, InterPrediction420,
    InterResidual, IntraLumaPrediction, IntraMacroblock, IntraMacroblockHeader, IntraModeState,
    IntraPredictionModeSyntax, IntraReferenceAvailability, MacroblockQuantizer,
    MacroblockQuantizerState, PMacroblockContext, PMotionState, ParsedSliceHeader,
    PredictionWeightTable, ReconstructedInterResidual, ReconstructedIntraResidual,
    ReconstructedLumaResidual, ReferenceId, ReferenceMotionField, ResolvedBListMotion,
    ResolvedBMacroblock, ResolvedPMacroblock, ResolvedScalingLists4x4, ResolvedScalingLists8x8,
    Result, ScanMode, SliceType, SpatialDirectContext, TemporalDirectContext, WeightedBiprediction,
    Yuv420Picture, consume_rbsp_trailing_bits, derive_chroma_qp, parse_cavlc_mb_skip_run,
    predict_intra_4x4, predict_intra_8x8, predict_intra_16x16, predict_intra_chroma_420,
    reconstruct_b_inter_residual, reconstruct_inter_residual, reconstruct_intra_residual,
    reconstruct_p_macroblock_from_list_420, resolve_scaling_lists_4x4, resolve_scaling_lists_8x8,
};

const LUMA_4X4_COORDINATES: [(usize, usize); 16] = [
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
const LUMA_8X8_COORDINATES: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

#[derive(Debug, Clone, Copy)]
struct IntraSliceConfig {
    header_bit_size: usize,
    first_macroblock: usize,
    slice_qp_y: u8,
    transform_8x8_mode: bool,
    chroma_cb_offset: i8,
    chroma_cr_offset: i8,
    transform_bypass_enabled: bool,
    deblocking_filter: DeblockingFilter,
}

#[derive(Debug, Clone, Copy)]
enum BSlicePredictionWeights<'a> {
    Default,
    Explicit(&'a PredictionWeightTable),
    Implicit {
        current_picture_order_count: i32,
        list0: &'a [Option<ImplicitWeightReference>],
        list1: &'a [Option<ImplicitWeightReference>],
    },
}

#[derive(Debug, Clone, Copy)]
enum BDirectPrediction<'a> {
    Spatial(Option<SpatialDirectContext<'a>>),
    Temporal(Option<TemporalDirectContext<'a>>),
}

impl<'a> BDirectPrediction<'a> {
    fn context(self) -> Result<DirectMotionContext<'a>> {
        match self {
            Self::Spatial(Some(context)) => Ok(DirectMotionContext::Spatial(context)),
            Self::Spatial(None) => Err(H264Error::InvalidSyntax(
                "spatial Direct requires active List 1 reference metadata",
            )),
            Self::Temporal(Some(context)) => Ok(DirectMotionContext::Temporal(context)),
            Self::Temporal(None) => Err(H264Error::InvalidSyntax(
                "temporal Direct requires active reference metadata",
            )),
        }
    }

    fn resolve(
        self,
        state: &mut BMotionState,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<ResolvedBMacroblock> {
        match self.context()? {
            DirectMotionContext::Spatial(context) => {
                state.resolve_spatial_direct_macroblock(macroblock_address, slice_id, context)
            }
            DirectMotionContext::Temporal(context) => {
                state.resolve_temporal_direct_macroblock(macroblock_address, slice_id, context)
            }
        }
    }

    fn resolve_mixed(
        self,
        state: &mut BMotionState,
        macroblock_address: usize,
        slice_id: u32,
        header: &crate::BInterMacroblockHeader,
    ) -> Result<ResolvedBMacroblock> {
        state.resolve_mixed_direct_8x8_macroblock(
            macroblock_address,
            slice_id,
            header,
            self.context()?,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct BReconstructionModes<'a> {
    weights: BSlicePredictionWeights<'a>,
    direct: BDirectPrediction<'a>,
}

#[derive(Debug)]
struct PendingBInterMacroblock {
    address: usize,
    macroblock_x: usize,
    macroblock_y: usize,
    motion: ResolvedBMacroblock,
    residual: Option<ReconstructedInterResidual>,
    deblock: MacroblockDeblockInfo,
}

#[derive(Debug)]
struct PendingPInterMacroblock {
    address: usize,
    macroblock_x: usize,
    macroblock_y: usize,
    motion: ResolvedPMacroblock,
    residual: Option<ReconstructedInterResidual>,
    deblock: MacroblockDeblockInfo,
}

#[derive(Debug, Clone, Copy)]
struct DeblockReferenceList {
    ids: [u8; 32],
    len: u8,
}

impl DeblockReferenceList {
    #[inline]
    fn get(&self, reference_index: u8) -> Option<u8> {
        (reference_index < self.len)
            .then(|| self.ids[usize::from(reference_index)])
            .filter(|id| *id != 0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReconstructionReferenceList<'a> {
    pictures: &'a [Option<&'a Yuv420Picture>],
    reference_ids: Option<&'a [Option<ReferenceId>]>,
    implicit_weights: Option<&'a [Option<ImplicitWeightReference>]>,
    direct_references: Option<&'a [Option<DirectReference<'a>>]>,
}

impl<'a> ReconstructionReferenceList<'a> {
    pub const fn new(pictures: &'a [Option<&'a Yuv420Picture>]) -> Self {
        Self {
            pictures,
            reference_ids: None,
            implicit_weights: None,
            direct_references: None,
        }
    }

    pub const fn with_ids(
        pictures: &'a [Option<&'a Yuv420Picture>],
        reference_ids: &'a [Option<ReferenceId>],
    ) -> Self {
        Self {
            pictures,
            reference_ids: Some(reference_ids),
            implicit_weights: None,
            direct_references: None,
        }
    }

    pub const fn with_metadata(
        pictures: &'a [Option<&'a Yuv420Picture>],
        reference_ids: &'a [Option<ReferenceId>],
        implicit_weights: &'a [Option<ImplicitWeightReference>],
        direct_references: &'a [Option<DirectReference<'a>>],
    ) -> Self {
        Self {
            pictures,
            reference_ids: Some(reference_ids),
            implicit_weights: Some(implicit_weights),
            direct_references: Some(direct_references),
        }
    }
}

/// Reconstructs one progressively scanned intra picture in macroblock order.
#[derive(Debug, Clone)]
pub struct IntraPictureReconstructor {
    width_in_macroblocks: usize,
    picture: Yuv420Picture,
    cavlc: CavlcNeighborState,
    cabac_residual: CabacResidualState,
    modes: IntraModeState,
    motion: PMotionState,
    b_motion: BMotionState,
    reference_motion: MotionFieldBuilder,
    completed: Vec<bool>,
    deblock: Vec<MacroblockDeblockInfo>,
    // Stable picture-local tokens preserve reference identity across slices
    // without storing pointer-sized addresses in every deblocking grid cell.
    deblock_reference_addresses: [usize; 32],
    deblock_reference_count: u8,
    scaling_lists: ResolvedScalingLists4x4,
    scaling_lists_8x8: ResolvedScalingLists8x8,
    scan_mode: ScanMode,
    constrained_intra_prediction: bool,
    reconstruction_executor: ReconstructionExecutor,
    next_slice_id: u32,
}

impl IntraPictureReconstructor {
    pub fn new(
        coded_size: Size,
        scaling_lists: ResolvedScalingLists4x4,
        constrained_intra_prediction: bool,
    ) -> Result<Self> {
        Self::new_with_scaling_lists(
            coded_size,
            scaling_lists,
            resolve_scaling_lists_8x8(None, None)?,
            constrained_intra_prediction,
        )
    }

    pub fn new_with_scaling_lists(
        coded_size: Size,
        scaling_lists: ResolvedScalingLists4x4,
        scaling_lists_8x8: ResolvedScalingLists8x8,
        constrained_intra_prediction: bool,
    ) -> Result<Self> {
        Self::new_with_executor(
            coded_size,
            scaling_lists,
            scaling_lists_8x8,
            constrained_intra_prediction,
            ReconstructionExecutor::serial(),
        )
    }

    fn new_with_executor(
        coded_size: Size,
        scaling_lists: ResolvedScalingLists4x4,
        scaling_lists_8x8: ResolvedScalingLists8x8,
        constrained_intra_prediction: bool,
        reconstruction_executor: ReconstructionExecutor,
    ) -> Result<Self> {
        let picture = Yuv420Picture::new(coded_size)?;
        let width_in_macroblocks =
            usize::try_from(coded_size.width / 16).map_err(|_| H264Error::IntegerOverflow)?;
        let height_in_macroblocks =
            usize::try_from(coded_size.height / 16).map_err(|_| H264Error::IntegerOverflow)?;
        let macroblock_count = width_in_macroblocks
            .checked_mul(height_in_macroblocks)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width_in_macroblocks,
            picture,
            cavlc: CavlcNeighborState::new(coded_size.width / 16, coded_size.height / 16)?,
            cabac_residual: CabacResidualState::new(width_in_macroblocks, height_in_macroblocks)?,
            modes: IntraModeState::new(width_in_macroblocks, height_in_macroblocks)?,
            motion: PMotionState::new(width_in_macroblocks, height_in_macroblocks)?,
            b_motion: BMotionState::new(width_in_macroblocks, height_in_macroblocks)?,
            reference_motion: MotionFieldBuilder::new(coded_size)?,
            completed: vec![false; macroblock_count],
            deblock: vec![MacroblockDeblockInfo::default(); macroblock_count],
            deblock_reference_addresses: [0; 32],
            deblock_reference_count: 0,
            scaling_lists,
            scaling_lists_8x8,
            scan_mode: ScanMode::Frame,
            constrained_intra_prediction,
            reconstruction_executor,
            next_slice_id: 0,
        })
    }

    pub fn from_parameter_sets(parameter_sets: &ActiveParameterSets) -> Result<Self> {
        Self::from_parameter_sets_with_executor(parameter_sets, ReconstructionExecutor::serial())
    }

    pub(crate) fn from_parameter_sets_with_executor(
        parameter_sets: &ActiveParameterSets,
        reconstruction_executor: ReconstructionExecutor,
    ) -> Result<Self> {
        let sps = &parameter_sets.sequence;
        let pps = &parameter_sets.picture;
        let scaling_lists = resolve_scaling_lists_4x4(
            sps.scaling_matrices.as_ref(),
            pps.scaling_matrices.as_ref(),
        )?;
        let scaling_lists_8x8 = resolve_scaling_lists_8x8(
            sps.scaling_matrices.as_ref(),
            pps.scaling_matrices.as_ref(),
        )?;
        Self::new_with_executor(
            sps.coded_size,
            scaling_lists,
            scaling_lists_8x8,
            pps.constrained_intra_prediction,
            reconstruction_executor,
        )
    }

    #[inline]
    pub fn picture(&self) -> &Yuv420Picture {
        &self.picture
    }

    /// Decodes and reconstructs one progressively scanned CAVLC I slice.
    pub fn decode_cavlc_intra_slice(
        &mut self,
        rbsp: &[u8],
        parsed: &ParsedSliceHeader,
    ) -> Result<usize> {
        let header = &parsed.header;
        let sps = &parsed.parameter_sets.sequence;
        let pps = &parsed.parameter_sets.picture;
        if header.slice_type != SliceType::I {
            return Err(H264Error::UnsupportedFeature(
                "slice-data reconstruction currently requires an I slice",
            ));
        }
        if pps.entropy_coding_mode != EntropyCodingMode::Cavlc {
            return Err(H264Error::UnsupportedFeature(
                "CABAC slice-data reconstruction",
            ));
        }
        if pps.num_slice_groups != 1 {
            return Err(H264Error::UnsupportedFeature(
                "FMO slice-data reconstruction",
            ));
        }
        if header.field_picture || sps.mb_adaptive_frame_field {
            return Err(H264Error::UnsupportedFeature(
                "field and MBAFF slice-data reconstruction",
            ));
        }
        if sps.coded_size != self.picture.coded_size()
            || pps.constrained_intra_prediction != self.constrained_intra_prediction
        {
            return Err(H264Error::InvalidSyntax(
                "slice parameter sets do not match the picture reconstructor",
            ));
        }
        let config = IntraSliceConfig {
            header_bit_size: header.bit_size,
            first_macroblock: usize::try_from(header.first_mb_in_slice)
                .map_err(|_| H264Error::IntegerOverflow)?,
            slice_qp_y: header.slice_qp_y,
            transform_8x8_mode: pps.transform_8x8_mode,
            chroma_cb_offset: pps.chroma_qp_index_offset,
            chroma_cr_offset: pps.second_chroma_qp_index_offset,
            transform_bypass_enabled: sps.qpprime_y_zero_transform_bypass,
            deblocking_filter: header.deblocking_filter.unwrap_or_default(),
        };
        self.decode_cavlc_intra_slice_data(rbsp, config)
    }

    /// Decodes and reconstructs one progressively scanned CABAC I slice.
    pub fn decode_cabac_intra_slice(
        &mut self,
        rbsp: &[u8],
        parsed: &ParsedSliceHeader,
    ) -> Result<usize> {
        let header = &parsed.header;
        let sps = &parsed.parameter_sets.sequence;
        let pps = &parsed.parameter_sets.picture;
        if header.slice_type != SliceType::I {
            return Err(H264Error::InvalidSyntax(
                "CABAC I reconstruction requires an I slice header",
            ));
        }
        if pps.entropy_coding_mode != EntropyCodingMode::Cabac
            || pps.num_slice_groups != 1
            || header.field_picture
            || sps.mb_adaptive_frame_field
        {
            return Err(H264Error::UnsupportedFeature(
                "CABAC I reconstruction currently requires progressive non-FMO input",
            ));
        }
        if sps.coded_size != self.picture.coded_size()
            || pps.constrained_intra_prediction != self.constrained_intra_prediction
        {
            return Err(H264Error::InvalidSyntax(
                "slice parameter sets do not match the picture reconstructor",
            ));
        }
        let config = IntraSliceConfig {
            header_bit_size: header.bit_size,
            first_macroblock: usize::try_from(header.first_mb_in_slice)
                .map_err(|_| H264Error::IntegerOverflow)?,
            slice_qp_y: header.slice_qp_y,
            transform_8x8_mode: pps.transform_8x8_mode,
            chroma_cb_offset: pps.chroma_qp_index_offset,
            chroma_cr_offset: pps.second_chroma_qp_index_offset,
            transform_bypass_enabled: sps.qpprime_y_zero_transform_bypass,
            deblocking_filter: header.deblocking_filter.unwrap_or_default(),
        };
        self.decode_cabac_intra_slice_data(rbsp, config)
    }

    /// Decodes one progressive CAVLC P slice against an already constructed
    /// active List 0.
    pub fn decode_cavlc_p_slice(
        &mut self,
        rbsp: &[u8],
        parsed: &ParsedSliceHeader,
        references_l0: ReconstructionReferenceList<'_>,
    ) -> Result<usize> {
        let header = &parsed.header;
        let sps = &parsed.parameter_sets.sequence;
        let pps = &parsed.parameter_sets.picture;
        if header.slice_type != SliceType::P {
            return Err(H264Error::InvalidSyntax(
                "P slice reconstruction requires a P slice header",
            ));
        }
        if pps.entropy_coding_mode != EntropyCodingMode::Cavlc
            || pps.num_slice_groups != 1
            || header.field_picture
            || sps.mb_adaptive_frame_field
        {
            return Err(H264Error::UnsupportedFeature(
                "P reconstruction currently requires progressive non-FMO CAVLC",
            ));
        }
        if sps.coded_size != self.picture.coded_size()
            || pps.constrained_intra_prediction != self.constrained_intra_prediction
        {
            return Err(H264Error::InvalidSyntax(
                "slice parameter sets do not match the picture reconstructor",
            ));
        }
        let config = IntraSliceConfig {
            header_bit_size: header.bit_size,
            first_macroblock: usize::try_from(header.first_mb_in_slice)
                .map_err(|_| H264Error::IntegerOverflow)?,
            slice_qp_y: header.slice_qp_y,
            transform_8x8_mode: pps.transform_8x8_mode,
            chroma_cb_offset: pps.chroma_qp_index_offset,
            chroma_cr_offset: pps.second_chroma_qp_index_offset,
            transform_bypass_enabled: sps.qpprime_y_zero_transform_bypass,
            deblocking_filter: header.deblocking_filter.unwrap_or_default(),
        };
        self.decode_cavlc_p_slice_data(
            rbsp,
            config,
            header.num_ref_idx_l0_active,
            references_l0,
            header.prediction_weights.as_ref(),
        )
    }

    /// Decodes one progressive CABAC P slice against an already constructed
    /// active List 0.
    pub fn decode_cabac_p_slice(
        &mut self,
        rbsp: &[u8],
        parsed: &ParsedSliceHeader,
        references_l0: ReconstructionReferenceList<'_>,
    ) -> Result<usize> {
        let header = &parsed.header;
        let sps = &parsed.parameter_sets.sequence;
        let pps = &parsed.parameter_sets.picture;
        if header.slice_type != SliceType::P {
            return Err(H264Error::InvalidSyntax(
                "CABAC P reconstruction requires a P slice header",
            ));
        }
        if pps.entropy_coding_mode != EntropyCodingMode::Cabac
            || pps.num_slice_groups != 1
            || header.field_picture
            || sps.mb_adaptive_frame_field
        {
            return Err(H264Error::UnsupportedFeature(
                "CABAC P reconstruction currently requires progressive non-FMO input",
            ));
        }
        if sps.coded_size != self.picture.coded_size()
            || pps.constrained_intra_prediction != self.constrained_intra_prediction
        {
            return Err(H264Error::InvalidSyntax(
                "slice parameter sets do not match the picture reconstructor",
            ));
        }
        let config = IntraSliceConfig {
            header_bit_size: header.bit_size,
            first_macroblock: usize::try_from(header.first_mb_in_slice)
                .map_err(|_| H264Error::IntegerOverflow)?,
            slice_qp_y: header.slice_qp_y,
            transform_8x8_mode: pps.transform_8x8_mode,
            chroma_cb_offset: pps.chroma_qp_index_offset,
            chroma_cr_offset: pps.second_chroma_qp_index_offset,
            transform_bypass_enabled: sps.qpprime_y_zero_transform_bypass,
            deblocking_filter: header.deblocking_filter.unwrap_or_default(),
        };
        self.decode_cabac_p_slice_data(
            rbsp,
            config,
            header.cabac_init_idc,
            header.num_ref_idx_l0_active,
            references_l0,
            header.prediction_weights.as_ref(),
        )
    }

    /// Decodes one progressive CAVLC B slice using default, unweighted
    /// explicit List 0/List 1 prediction.
    ///
    /// Direct and skipped B macroblocks require co-located motion and POC
    /// scaling, so they remain explicitly unsupported.
    pub fn decode_cavlc_b_slice(
        &mut self,
        rbsp: &[u8],
        parsed: &ParsedSliceHeader,
        references_l0: ReconstructionReferenceList<'_>,
        references_l1: ReconstructionReferenceList<'_>,
        current_picture_order_count: i32,
    ) -> Result<usize> {
        let header = &parsed.header;
        let sps = &parsed.parameter_sets.sequence;
        let pps = &parsed.parameter_sets.picture;
        if header.slice_type != SliceType::B {
            return Err(H264Error::InvalidSyntax(
                "B slice reconstruction requires a B slice header",
            ));
        }
        if pps.entropy_coding_mode != EntropyCodingMode::Cavlc
            || pps.num_slice_groups != 1
            || header.field_picture
            || sps.mb_adaptive_frame_field
        {
            return Err(H264Error::UnsupportedFeature(
                "B reconstruction currently requires progressive non-FMO CAVLC",
            ));
        }
        let prediction_weights = match pps.weighted_biprediction {
            WeightedBiprediction::Default => BSlicePredictionWeights::Default,
            WeightedBiprediction::Explicit => {
                BSlicePredictionWeights::Explicit(header.prediction_weights.as_ref().ok_or(
                    H264Error::InvalidSyntax(
                        "explicit weighted B slice is missing pred_weight_table",
                    ),
                )?)
            }
            WeightedBiprediction::Implicit => BSlicePredictionWeights::Implicit {
                current_picture_order_count,
                list0: references_l0
                    .implicit_weights
                    .ok_or(H264Error::InvalidSyntax(
                        "implicit weighted B List 0 is missing reference metadata",
                    ))?,
                list1: references_l1
                    .implicit_weights
                    .ok_or(H264Error::InvalidSyntax(
                        "implicit weighted B List 1 is missing reference metadata",
                    ))?,
            },
        };
        let direct_prediction = if header.direct_spatial_mv_prediction == Some(true) {
            let colocated = references_l1
                .direct_references
                .and_then(|references| references.first())
                .copied()
                .flatten();
            BDirectPrediction::Spatial(colocated.map(|colocated| SpatialDirectContext {
                colocated_motion: colocated.motion,
                colocated_long_term: colocated.long_term,
                direct_8x8_inference: sps.direct_8x8_inference,
                num_ref_idx_l0_active: header.num_ref_idx_l0_active,
                num_ref_idx_l1_active: header.num_ref_idx_l1_active,
            }))
        } else {
            let context = references_l0
                .direct_references
                .zip(references_l1.direct_references)
                .and_then(|(list0, list1)| {
                    list1
                        .first()
                        .copied()
                        .flatten()
                        .map(|colocated| TemporalDirectContext {
                            current_picture_order_count,
                            colocated,
                            references_l0: list0,
                            direct_8x8_inference: sps.direct_8x8_inference,
                            num_ref_idx_l1_active: header.num_ref_idx_l1_active,
                        })
                });
            BDirectPrediction::Temporal(context)
        };
        if sps.coded_size != self.picture.coded_size()
            || pps.constrained_intra_prediction != self.constrained_intra_prediction
        {
            return Err(H264Error::InvalidSyntax(
                "slice parameter sets do not match the picture reconstructor",
            ));
        }
        let config = IntraSliceConfig {
            header_bit_size: header.bit_size,
            first_macroblock: usize::try_from(header.first_mb_in_slice)
                .map_err(|_| H264Error::IntegerOverflow)?,
            slice_qp_y: header.slice_qp_y,
            transform_8x8_mode: pps.transform_8x8_mode,
            chroma_cb_offset: pps.chroma_qp_index_offset,
            chroma_cr_offset: pps.second_chroma_qp_index_offset,
            transform_bypass_enabled: sps.qpprime_y_zero_transform_bypass,
            deblocking_filter: header.deblocking_filter.unwrap_or_default(),
        };
        self.decode_cavlc_b_slice_data(
            rbsp,
            config,
            BMacroblockContext {
                num_ref_idx_l0_active: header.num_ref_idx_l0_active,
                num_ref_idx_l1_active: header.num_ref_idx_l1_active,
                transform_8x8_mode_enabled: pps.transform_8x8_mode,
                direct_8x8_inference: sps.direct_8x8_inference,
            },
            references_l0,
            references_l1,
            BReconstructionModes {
                weights: prediction_weights,
                direct: direct_prediction,
            },
        )
    }

    /// Decodes one progressive CABAC B slice using the active reference lists.
    pub fn decode_cabac_b_slice(
        &mut self,
        rbsp: &[u8],
        parsed: &ParsedSliceHeader,
        references_l0: ReconstructionReferenceList<'_>,
        references_l1: ReconstructionReferenceList<'_>,
        current_picture_order_count: i32,
    ) -> Result<usize> {
        let header = &parsed.header;
        let sps = &parsed.parameter_sets.sequence;
        let pps = &parsed.parameter_sets.picture;
        if header.slice_type != SliceType::B {
            return Err(H264Error::InvalidSyntax(
                "CABAC B reconstruction requires a B slice header",
            ));
        }
        if pps.entropy_coding_mode != EntropyCodingMode::Cabac
            || pps.num_slice_groups != 1
            || header.field_picture
            || sps.mb_adaptive_frame_field
        {
            return Err(H264Error::UnsupportedFeature(
                "CABAC B reconstruction currently requires progressive non-FMO input",
            ));
        }
        let prediction_weights = match pps.weighted_biprediction {
            WeightedBiprediction::Default => BSlicePredictionWeights::Default,
            WeightedBiprediction::Explicit => {
                BSlicePredictionWeights::Explicit(header.prediction_weights.as_ref().ok_or(
                    H264Error::InvalidSyntax(
                        "explicit weighted B slice is missing pred_weight_table",
                    ),
                )?)
            }
            WeightedBiprediction::Implicit => BSlicePredictionWeights::Implicit {
                current_picture_order_count,
                list0: references_l0
                    .implicit_weights
                    .ok_or(H264Error::InvalidSyntax(
                        "implicit weighted B List 0 is missing reference metadata",
                    ))?,
                list1: references_l1
                    .implicit_weights
                    .ok_or(H264Error::InvalidSyntax(
                        "implicit weighted B List 1 is missing reference metadata",
                    ))?,
            },
        };
        let direct_prediction = if header.direct_spatial_mv_prediction == Some(true) {
            let colocated = references_l1
                .direct_references
                .and_then(|references| references.first())
                .copied()
                .flatten();
            BDirectPrediction::Spatial(colocated.map(|colocated| SpatialDirectContext {
                colocated_motion: colocated.motion,
                colocated_long_term: colocated.long_term,
                direct_8x8_inference: sps.direct_8x8_inference,
                num_ref_idx_l0_active: header.num_ref_idx_l0_active,
                num_ref_idx_l1_active: header.num_ref_idx_l1_active,
            }))
        } else {
            let context = references_l0
                .direct_references
                .zip(references_l1.direct_references)
                .and_then(|(list0, list1)| {
                    list1
                        .first()
                        .copied()
                        .flatten()
                        .map(|colocated| TemporalDirectContext {
                            current_picture_order_count,
                            colocated,
                            references_l0: list0,
                            direct_8x8_inference: sps.direct_8x8_inference,
                            num_ref_idx_l1_active: header.num_ref_idx_l1_active,
                        })
                });
            BDirectPrediction::Temporal(context)
        };
        if sps.coded_size != self.picture.coded_size()
            || pps.constrained_intra_prediction != self.constrained_intra_prediction
        {
            return Err(H264Error::InvalidSyntax(
                "slice parameter sets do not match the picture reconstructor",
            ));
        }
        let config = IntraSliceConfig {
            header_bit_size: header.bit_size,
            first_macroblock: usize::try_from(header.first_mb_in_slice)
                .map_err(|_| H264Error::IntegerOverflow)?,
            slice_qp_y: header.slice_qp_y,
            transform_8x8_mode: pps.transform_8x8_mode,
            chroma_cb_offset: pps.chroma_qp_index_offset,
            chroma_cr_offset: pps.second_chroma_qp_index_offset,
            transform_bypass_enabled: sps.qpprime_y_zero_transform_bypass,
            deblocking_filter: header.deblocking_filter.unwrap_or_default(),
        };
        self.decode_cabac_b_slice_data(
            rbsp,
            config,
            header.cabac_init_idc,
            CabacBMacroblockContext {
                num_ref_idx_l0_active: header.num_ref_idx_l0_active,
                num_ref_idx_l1_active: header.num_ref_idx_l1_active,
                transform_8x8_mode_enabled: pps.transform_8x8_mode,
                direct_8x8_inference: sps.direct_8x8_inference,
                previous_qp_delta_nonzero: false,
            },
            references_l0,
            references_l1,
            BReconstructionModes {
                weights: prediction_weights,
                direct: direct_prediction,
            },
        )
    }

    fn decode_cavlc_intra_slice_data(
        &mut self,
        rbsp: &[u8],
        config: IntraSliceConfig,
    ) -> Result<usize> {
        let mut reader = BitReader::new(rbsp);
        if !reader.skip_bits(config.header_bit_size) {
            return Err(H264Error::UnexpectedEof);
        }
        self.next_slice_id = self
            .next_slice_id
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        let slice_id = self.next_slice_id;
        self.cavlc.begin_slice();
        let mut quantizers = MacroblockQuantizerState::new(
            config.slice_qp_y,
            config.chroma_cb_offset,
            config.chroma_cr_offset,
            config.transform_bypass_enabled,
        )?;
        let mut macroblock_address = config.first_macroblock;
        let mut decoded_count = 0usize;
        let pcm_chroma_qp = [
            derive_chroma_qp(0, config.chroma_cb_offset),
            derive_chroma_qp(0, config.chroma_cr_offset),
        ];
        while more_rbsp_data(&reader) {
            if macroblock_address >= self.completed.len() {
                return Err(H264Error::InvalidSyntax(
                    "slice data exceeds the reconstructed picture",
                ));
            }
            let macroblock_x = u32::try_from(macroblock_address % self.width_in_macroblocks)
                .map_err(|_| H264Error::IntegerOverflow)?;
            let macroblock_y = u32::try_from(macroblock_address / self.width_in_macroblocks)
                .map_err(|_| H264Error::IntegerOverflow)?;
            let cavlc_snapshot = self.cavlc.snapshot_macroblock(macroblock_x, macroblock_y)?;
            let decoded = self.cavlc.decode_intra_macroblock(
                &mut reader,
                macroblock_x,
                macroblock_y,
                config.transform_8x8_mode,
            )?;
            let qp_delta = match &decoded.macroblock {
                IntraMacroblock::Predicted(header) => header.qp_delta,
                IntraMacroblock::Pcm(_) => 0,
            };
            if let Err(error) = quantizers.with_macroblock(qp_delta, |quantizer| {
                self.reconstruct_macroblock_with_deblocking(
                    macroblock_address,
                    slice_id,
                    &decoded,
                    quantizer,
                    config.deblocking_filter,
                    pcm_chroma_qp,
                )
            }) {
                self.cavlc
                    .restore_macroblock(macroblock_x, macroblock_y, cavlc_snapshot);
                return Err(error);
            }
            macroblock_address = macroblock_address
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            decoded_count = decoded_count
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
        }
        consume_rbsp_trailing_bits(&mut reader)?;
        Ok(decoded_count)
    }

    fn decode_cabac_intra_slice_data(
        &mut self,
        rbsp: &[u8],
        config: IntraSliceConfig,
    ) -> Result<usize> {
        self.next_slice_id = self
            .next_slice_id
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        let slice_id = self.next_slice_id;
        let mut cabac = CabacSliceDecoder::new(
            rbsp,
            config.header_bit_size,
            SliceType::I,
            None,
            config.slice_qp_y,
        )?;
        let height_in_macroblocks = self.completed.len() / self.width_in_macroblocks;
        let mut macroblocks =
            CabacMacroblockState::new(self.width_in_macroblocks, height_in_macroblocks)?;
        let mut quantizers = MacroblockQuantizerState::new(
            config.slice_qp_y,
            config.chroma_cb_offset,
            config.chroma_cr_offset,
            config.transform_bypass_enabled,
        )?;
        let pcm_chroma_qp = [
            derive_chroma_qp(0, config.chroma_cb_offset),
            derive_chroma_qp(0, config.chroma_cr_offset),
        ];
        let mut macroblock_address = config.first_macroblock;
        let mut decoded_count = 0usize;
        let mut previous_qp_delta_nonzero = false;

        loop {
            self.macroblock_coordinates(macroblock_address)?;
            let syntax = {
                let mut syntax = cabac.syntax();
                macroblocks.decode_intra_macroblock_syntax(
                    &mut syntax,
                    macroblock_address,
                    slice_id,
                    config.transform_8x8_mode,
                    previous_qp_delta_nonzero,
                )?
            };
            let (decoded, qp_delta, summary) = match syntax {
                CabacIntraMacroblockSyntax::Predicted(header) => {
                    let residual = {
                        let mut syntax = cabac.syntax();
                        self.cabac_residual.decode_intra_residual_terminal(
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
                    let qp_delta = header.qp_delta;
                    (
                        DecodedIntraMacroblock {
                            macroblock: IntraMacroblock::Predicted(header),
                            residual: Some(residual),
                        },
                        qp_delta,
                        summary,
                    )
                }
                CabacIntraMacroblockSyntax::Pcm => {
                    let pcm = cabac.decode_pcm_420_8bit_and_restart()?;
                    self.cabac_residual
                        .record_pcm_macroblock(macroblock_address, slice_id)?;
                    (
                        DecodedIntraMacroblock {
                            macroblock: IntraMacroblock::Pcm(pcm),
                            residual: None,
                        },
                        0,
                        CabacMacroblockSummary {
                            skipped: false,
                            direct: false,
                            intra16_or_pcm: true,
                            intra_chroma_prediction: None,
                            coded_block_pattern: crate::CodedBlockPattern {
                                luma: 15,
                                chroma: 2,
                            },
                            transform_size_8x8: false,
                        },
                    )
                }
            };
            let end_of_slice = {
                let mut syntax = cabac.syntax();
                syntax.terminate()? != 0
            };
            quantizers.with_macroblock(qp_delta, |quantizer| {
                self.reconstruct_macroblock_with_deblocking(
                    macroblock_address,
                    slice_id,
                    &decoded,
                    quantizer,
                    config.deblocking_filter,
                    pcm_chroma_qp,
                )
            })?;

            macroblocks.record_macroblock(macroblock_address, slice_id, summary)?;
            previous_qp_delta_nonzero = qp_delta != 0;
            macroblock_address = macroblock_address
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            decoded_count = decoded_count
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            if end_of_slice {
                return Ok(decoded_count);
            }
            if macroblock_address >= self.completed.len() {
                return Err(H264Error::InvalidSyntax(
                    "CABAC slice has no terminating end_of_slice_flag",
                ));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_cabac_p_slice_data(
        &mut self,
        rbsp: &[u8],
        config: IntraSliceConfig,
        cabac_init_idc: Option<u8>,
        num_ref_idx_l0_active: u8,
        references_l0: ReconstructionReferenceList<'_>,
        prediction_weights: Option<&PredictionWeightTable>,
    ) -> Result<usize> {
        self.next_slice_id = self
            .next_slice_id
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        let slice_id = self.next_slice_id;
        let mut cabac = CabacSliceDecoder::new(
            rbsp,
            config.header_bit_size,
            SliceType::P,
            cabac_init_idc,
            config.slice_qp_y,
        )?;
        let height_in_macroblocks = self.completed.len() / self.width_in_macroblocks;
        let mut macroblocks =
            CabacPMacroblockState::new(self.width_in_macroblocks, height_in_macroblocks)?;
        let mut quantizers = MacroblockQuantizerState::new(
            config.slice_qp_y,
            config.chroma_cb_offset,
            config.chroma_cr_offset,
            config.transform_bypass_enabled,
        )?;
        let pcm_chroma_qp = [
            derive_chroma_qp(0, config.chroma_cb_offset),
            derive_chroma_qp(0, config.chroma_cr_offset),
        ];
        let reference_ids_l0 = references_l0.reference_ids;
        let deblock_references_l0 = self.deblock_reference_list(references_l0.pictures)?;
        let references_l0 = references_l0.pictures;
        let mut macroblock_address = config.first_macroblock;
        let mut decoded_count = 0usize;
        let mut previous_qp_delta_nonzero = false;
        let inter_batch_limit = self
            .width_in_macroblocks
            .checked_mul(4)
            .ok_or(H264Error::IntegerOverflow)?;
        let mut pending_inter = Vec::with_capacity(inter_batch_limit);

        loop {
            let (macroblock_x, macroblock_y) = self.macroblock_coordinates(macroblock_address)?;
            let decoded = macroblocks.decode_macroblock_terminal(
                &mut cabac,
                &mut self.cabac_residual,
                macroblock_address,
                slice_id,
                CabacPMacroblockContext {
                    num_ref_idx_l0_active,
                    transform_8x8_mode_enabled: config.transform_8x8_mode,
                    previous_qp_delta_nonzero,
                },
            )?;
            let qp_delta = decoded.macroblock.qp_delta();
            let reconstruction =
                quantizers.with_macroblock(qp_delta, |quantizer| match &decoded.macroblock {
                    CabacPMacroblock::Skip => {
                        let motion = self
                            .motion
                            .resolve_skip_macroblock(macroblock_address, slice_id)?;
                        if let Err(error) = self.reference_motion.record_p(
                            macroblock_address,
                            &motion,
                            reference_ids_l0,
                        ) {
                            self.motion.clear_macroblock(macroblock_address)?;
                            return Err(error);
                        }
                        let deblock = match inter_deblock_info(
                            slice_id,
                            quantizer,
                            false,
                            config.deblocking_filter,
                            &motion,
                            None,
                            &deblock_references_l0,
                        ) {
                            Ok(deblock) => deblock,
                            Err(error) => {
                                self.motion.clear_macroblock(macroblock_address)?;
                                self.reference_motion.clear_macroblock(macroblock_address)?;
                                return Err(error);
                            }
                        };
                        pending_inter.push(PendingPInterMacroblock {
                            address: macroblock_address,
                            macroblock_x,
                            macroblock_y,
                            motion,
                            residual: None,
                            deblock,
                        });
                        Ok(())
                    }
                    CabacPMacroblock::Decoded(decoded) => match decoded.as_ref() {
                        DecodedPSliceMacroblock::Inter { header, residual } => {
                            let reconstructed = reconstruct_inter_residual(
                                header,
                                residual,
                                quantizer,
                                &self.scaling_lists,
                                &self.scaling_lists_8x8,
                                self.scan_mode,
                            )?;
                            let motion = self.motion.resolve_inter_macroblock(
                                macroblock_address,
                                slice_id,
                                header,
                            )?;
                            if let Err(error) = self.reference_motion.record_p(
                                macroblock_address,
                                &motion,
                                reference_ids_l0,
                            ) {
                                self.motion.clear_macroblock(macroblock_address)?;
                                return Err(error);
                            }
                            let deblock = match inter_deblock_info(
                                slice_id,
                                quantizer,
                                header.transform_size_8x8,
                                config.deblocking_filter,
                                &motion,
                                Some(residual),
                                &deblock_references_l0,
                            ) {
                                Ok(deblock) => deblock,
                                Err(error) => {
                                    self.motion.clear_macroblock(macroblock_address)?;
                                    self.reference_motion.clear_macroblock(macroblock_address)?;
                                    return Err(error);
                                }
                            };
                            pending_inter.push(PendingPInterMacroblock {
                                address: macroblock_address,
                                macroblock_x,
                                macroblock_y,
                                motion,
                                residual: Some(reconstructed),
                                deblock,
                            });
                            Ok(())
                        }
                        DecodedPSliceMacroblock::Intra(decoded) => {
                            self.finish_pending_p_inter(
                                &mut pending_inter,
                                references_l0,
                                prediction_weights,
                                slice_id,
                            )?;
                            self.reconstruct_macroblock_with_deblocking(
                                macroblock_address,
                                slice_id,
                                decoded,
                                quantizer,
                                config.deblocking_filter,
                                pcm_chroma_qp,
                            )
                        }
                    },
                });
            reconstruction?;
            if pending_inter.len() >= inter_batch_limit {
                self.finish_pending_p_inter(
                    &mut pending_inter,
                    references_l0,
                    prediction_weights,
                    slice_id,
                )?;
            }

            previous_qp_delta_nonzero = qp_delta != 0;
            macroblock_address = macroblock_address
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            decoded_count = decoded_count
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            if decoded.end_of_slice {
                self.finish_pending_p_inter(
                    &mut pending_inter,
                    references_l0,
                    prediction_weights,
                    slice_id,
                )?;
                return Ok(decoded_count);
            }
            if macroblock_address >= self.completed.len() {
                return Err(H264Error::InvalidSyntax(
                    "CABAC P slice has no terminating end_of_slice_flag",
                ));
            }
        }
    }

    fn decode_cavlc_p_slice_data(
        &mut self,
        rbsp: &[u8],
        config: IntraSliceConfig,
        num_ref_idx_l0_active: u8,
        references_l0: ReconstructionReferenceList<'_>,
        prediction_weights: Option<&PredictionWeightTable>,
    ) -> Result<usize> {
        let reference_ids_l0 = references_l0.reference_ids;
        let deblock_references_l0 = self.deblock_reference_list(references_l0.pictures)?;
        let references_l0 = references_l0.pictures;
        let mut reader = BitReader::new(rbsp);
        if !reader.skip_bits(config.header_bit_size) {
            return Err(H264Error::UnexpectedEof);
        }
        self.next_slice_id = self
            .next_slice_id
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        let slice_id = self.next_slice_id;
        self.cavlc.begin_slice();
        let mut quantizers = MacroblockQuantizerState::new(
            config.slice_qp_y,
            config.chroma_cb_offset,
            config.chroma_cr_offset,
            config.transform_bypass_enabled,
        )?;
        let mut macroblock_address = config.first_macroblock;
        let mut decoded_count = 0usize;
        let pcm_chroma_qp = [
            derive_chroma_qp(0, config.chroma_cb_offset),
            derive_chroma_qp(0, config.chroma_cr_offset),
        ];

        while more_rbsp_data(&reader) {
            let remaining = self.completed.len().checked_sub(macroblock_address).ok_or(
                H264Error::InvalidSyntax("P slice starts beyond the reconstructed picture"),
            )?;
            let skip_run = parse_cavlc_mb_skip_run(&mut reader, remaining)?;
            for _ in 0..skip_run {
                let (macroblock_x, macroblock_y) =
                    self.macroblock_coordinates(macroblock_address)?;
                let cavlc_snapshot = self.cavlc.snapshot_macroblock(
                    u32::try_from(macroblock_x).map_err(|_| H264Error::IntegerOverflow)?,
                    u32::try_from(macroblock_y).map_err(|_| H264Error::IntegerOverflow)?,
                )?;
                self.cavlc.record_zero_macroblock(
                    u32::try_from(macroblock_x).map_err(|_| H264Error::IntegerOverflow)?,
                    u32::try_from(macroblock_y).map_err(|_| H264Error::IntegerOverflow)?,
                )?;
                let motion = self
                    .motion
                    .resolve_skip_macroblock(macroblock_address, slice_id)?;
                if let Err(error) =
                    self.reference_motion
                        .record_p(macroblock_address, &motion, reference_ids_l0)
                {
                    self.motion.clear_macroblock(macroblock_address)?;
                    self.cavlc.restore_macroblock(
                        u32::try_from(macroblock_x).map_err(|_| H264Error::IntegerOverflow)?,
                        u32::try_from(macroblock_y).map_err(|_| H264Error::IntegerOverflow)?,
                        cavlc_snapshot,
                    );
                    return Err(error);
                }
                let reconstruction = if let Some(weights) = prediction_weights {
                    reconstruct_weighted_p_skip_macroblock_from_list_420(
                        &mut self.picture,
                        references_l0,
                        macroblock_x,
                        macroblock_y,
                        &motion,
                        weights,
                    )
                } else {
                    reconstruct_p_skip_macroblock_from_list_420(
                        &mut self.picture,
                        references_l0,
                        macroblock_x,
                        macroblock_y,
                        &motion,
                    )
                };
                let result = reconstruction
                    .and_then(|()| self.modes.record_inter(macroblock_address, slice_id));
                if let Err(error) = result {
                    self.motion.clear_macroblock(macroblock_address)?;
                    self.reference_motion.clear_macroblock(macroblock_address)?;
                    self.cavlc.restore_macroblock(
                        u32::try_from(macroblock_x).map_err(|_| H264Error::IntegerOverflow)?,
                        u32::try_from(macroblock_y).map_err(|_| H264Error::IntegerOverflow)?,
                        cavlc_snapshot,
                    );
                    return Err(error);
                }
                self.complete_inter_macroblock(
                    macroblock_address,
                    inter_deblock_info(
                        slice_id,
                        quantizers.derive(0)?,
                        false,
                        config.deblocking_filter,
                        &motion,
                        None,
                        &deblock_references_l0,
                    )?,
                );
                macroblock_address += 1;
                decoded_count += 1;
            }
            if !more_rbsp_data(&reader) {
                break;
            }

            let (macroblock_x, macroblock_y) = self.macroblock_coordinates(macroblock_address)?;
            let mb_x = u32::try_from(macroblock_x).map_err(|_| H264Error::IntegerOverflow)?;
            let mb_y = u32::try_from(macroblock_y).map_err(|_| H264Error::IntegerOverflow)?;
            let cavlc_snapshot = self.cavlc.snapshot_macroblock(mb_x, mb_y)?;
            let decoded = self.cavlc.decode_p_macroblock(
                &mut reader,
                mb_x,
                mb_y,
                PMacroblockContext {
                    num_ref_idx_l0_active,
                    transform_8x8_mode_enabled: config.transform_8x8_mode,
                },
            )?;
            let qp_delta = match &decoded {
                DecodedPSliceMacroblock::Inter { header, .. } => header.qp_delta,
                DecodedPSliceMacroblock::Intra(decoded) => match &decoded.macroblock {
                    IntraMacroblock::Predicted(header) => header.qp_delta,
                    IntraMacroblock::Pcm(_) => 0,
                },
            };
            let result = quantizers.with_macroblock(qp_delta, |quantizer| match &decoded {
                DecodedPSliceMacroblock::Inter { header, residual } => {
                    let reconstructed = reconstruct_inter_residual(
                        header,
                        residual,
                        quantizer,
                        &self.scaling_lists,
                        &self.scaling_lists_8x8,
                        self.scan_mode,
                    )?;
                    let motion = self.motion.resolve_inter_macroblock(
                        macroblock_address,
                        slice_id,
                        header,
                    )?;
                    if let Err(error) = self.reference_motion.record_p(
                        macroblock_address,
                        &motion,
                        reference_ids_l0,
                    ) {
                        self.motion.clear_macroblock(macroblock_address)?;
                        return Err(error);
                    }
                    let reconstruction = if let Some(weights) = prediction_weights {
                        reconstruct_weighted_p_macroblock_from_list_420(
                            &mut self.picture,
                            references_l0,
                            macroblock_x,
                            macroblock_y,
                            &motion,
                            &reconstructed,
                            weights,
                        )
                    } else {
                        reconstruct_p_macroblock_from_list_420(
                            &mut self.picture,
                            references_l0,
                            macroblock_x,
                            macroblock_y,
                            &motion,
                            &reconstructed,
                        )
                    };
                    if let Err(error) = reconstruction
                        .and_then(|()| self.modes.record_inter(macroblock_address, slice_id))
                    {
                        self.motion.clear_macroblock(macroblock_address)?;
                        self.reference_motion.clear_macroblock(macroblock_address)?;
                        return Err(error);
                    }
                    self.complete_inter_macroblock(
                        macroblock_address,
                        inter_deblock_info(
                            slice_id,
                            quantizer,
                            header.transform_size_8x8,
                            config.deblocking_filter,
                            &motion,
                            Some(residual),
                            &deblock_references_l0,
                        )?,
                    );
                    Ok(())
                }
                DecodedPSliceMacroblock::Intra(decoded) => self
                    .reconstruct_macroblock_with_deblocking(
                        macroblock_address,
                        slice_id,
                        decoded,
                        quantizer,
                        config.deblocking_filter,
                        pcm_chroma_qp,
                    ),
            });
            if let Err(error) = result {
                self.cavlc.restore_macroblock(mb_x, mb_y, cavlc_snapshot);
                return Err(error);
            }
            macroblock_address = macroblock_address
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            decoded_count = decoded_count
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
        }
        consume_rbsp_trailing_bits(&mut reader)?;
        Ok(decoded_count)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_cabac_b_slice_data(
        &mut self,
        rbsp: &[u8],
        config: IntraSliceConfig,
        cabac_init_idc: Option<u8>,
        mut context: CabacBMacroblockContext,
        references_l0: ReconstructionReferenceList<'_>,
        references_l1: ReconstructionReferenceList<'_>,
        modes: BReconstructionModes<'_>,
    ) -> Result<usize> {
        let reference_ids_l0 = references_l0.reference_ids;
        let reference_ids_l1 = references_l1.reference_ids;
        let deblock_references_l0 = self.deblock_reference_list(references_l0.pictures)?;
        let deblock_references_l1 = self.deblock_reference_list(references_l1.pictures)?;
        let references_l0 = references_l0.pictures;
        let references_l1 = references_l1.pictures;
        self.next_slice_id = self
            .next_slice_id
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        let slice_id = self.next_slice_id;
        let mut cabac = CabacSliceDecoder::new(
            rbsp,
            config.header_bit_size,
            SliceType::B,
            cabac_init_idc,
            config.slice_qp_y,
        )?;
        let height_in_macroblocks = self.completed.len() / self.width_in_macroblocks;
        let mut macroblocks =
            CabacBMacroblockState::new(self.width_in_macroblocks, height_in_macroblocks)?;
        let mut quantizers = MacroblockQuantizerState::new(
            config.slice_qp_y,
            config.chroma_cb_offset,
            config.chroma_cr_offset,
            config.transform_bypass_enabled,
        )?;
        let pcm_chroma_qp = [
            derive_chroma_qp(0, config.chroma_cb_offset),
            derive_chroma_qp(0, config.chroma_cr_offset),
        ];
        let mut macroblock_address = config.first_macroblock;
        let mut decoded_count = 0usize;
        let inter_batch_limit = self
            .width_in_macroblocks
            .checked_mul(4)
            .ok_or(H264Error::IntegerOverflow)?;
        let mut pending_inter = Vec::with_capacity(inter_batch_limit);

        loop {
            let (macroblock_x, macroblock_y) = self.macroblock_coordinates(macroblock_address)?;
            let decoded = macroblocks.decode_macroblock_terminal(
                &mut cabac,
                &mut self.cabac_residual,
                macroblock_address,
                slice_id,
                context,
            )?;
            let qp_delta = decoded.macroblock.qp_delta();
            let reconstruction =
                quantizers.with_macroblock(qp_delta, |quantizer| match &decoded.macroblock {
                    CabacBMacroblock::Skip => {
                        let motion = modes.direct.resolve(
                            &mut self.b_motion,
                            macroblock_address,
                            slice_id,
                        )?;
                        if let Err(error) = self.reference_motion.record_b(
                            macroblock_address,
                            &motion,
                            reference_ids_l0,
                            reference_ids_l1,
                        ) {
                            self.b_motion.clear_macroblock(macroblock_address)?;
                            return Err(error);
                        }
                        let deblock = match b_inter_deblock_info(
                            slice_id,
                            quantizer,
                            false,
                            config.deblocking_filter,
                            &motion,
                            None,
                            &deblock_references_l0,
                            &deblock_references_l1,
                        ) {
                            Ok(deblock) => deblock,
                            Err(error) => {
                                self.b_motion.clear_macroblock(macroblock_address)?;
                                self.reference_motion.clear_macroblock(macroblock_address)?;
                                return Err(error);
                            }
                        };
                        pending_inter.push(PendingBInterMacroblock {
                            address: macroblock_address,
                            macroblock_x,
                            macroblock_y,
                            motion,
                            residual: None,
                            deblock,
                        });
                        Ok(())
                    }
                    CabacBMacroblock::Decoded(decoded) => match decoded.as_ref() {
                        DecodedBSliceMacroblock::Inter { header, residual } => {
                            let reconstructed = reconstruct_b_inter_residual(
                                header,
                                residual,
                                quantizer,
                                &self.scaling_lists,
                                &self.scaling_lists_8x8,
                                self.scan_mode,
                            )?;
                            let motion = if is_fully_direct_b_macroblock(header) {
                                modes.direct.resolve(
                                    &mut self.b_motion,
                                    macroblock_address,
                                    slice_id,
                                )?
                            } else if has_direct_b_sub_macroblock(header) {
                                modes.direct.resolve_mixed(
                                    &mut self.b_motion,
                                    macroblock_address,
                                    slice_id,
                                    header,
                                )?
                            } else {
                                self.b_motion.resolve_inter_macroblock(
                                    macroblock_address,
                                    slice_id,
                                    header,
                                )?
                            };
                            if let Err(error) = self.reference_motion.record_b(
                                macroblock_address,
                                &motion,
                                reference_ids_l0,
                                reference_ids_l1,
                            ) {
                                self.b_motion.clear_macroblock(macroblock_address)?;
                                return Err(error);
                            }
                            let deblock = match b_inter_deblock_info(
                                slice_id,
                                quantizer,
                                header.transform_size_8x8,
                                config.deblocking_filter,
                                &motion,
                                Some(residual),
                                &deblock_references_l0,
                                &deblock_references_l1,
                            ) {
                                Ok(deblock) => deblock,
                                Err(error) => {
                                    self.b_motion.clear_macroblock(macroblock_address)?;
                                    self.reference_motion.clear_macroblock(macroblock_address)?;
                                    return Err(error);
                                }
                            };
                            pending_inter.push(PendingBInterMacroblock {
                                address: macroblock_address,
                                macroblock_x,
                                macroblock_y,
                                motion,
                                residual: Some(reconstructed),
                                deblock,
                            });
                            Ok(())
                        }
                        DecodedBSliceMacroblock::Intra(decoded) => {
                            self.finish_pending_b_inter(
                                &mut pending_inter,
                                references_l0,
                                references_l1,
                                modes.weights,
                                slice_id,
                            )?;
                            self.reconstruct_macroblock_with_deblocking(
                                macroblock_address,
                                slice_id,
                                decoded,
                                quantizer,
                                config.deblocking_filter,
                                pcm_chroma_qp,
                            )
                        }
                    },
                });
            reconstruction?;
            if pending_inter.len() >= inter_batch_limit {
                self.finish_pending_b_inter(
                    &mut pending_inter,
                    references_l0,
                    references_l1,
                    modes.weights,
                    slice_id,
                )?;
            }

            context.previous_qp_delta_nonzero = qp_delta != 0;
            macroblock_address = macroblock_address
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            decoded_count = decoded_count
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            if decoded.end_of_slice {
                self.finish_pending_b_inter(
                    &mut pending_inter,
                    references_l0,
                    references_l1,
                    modes.weights,
                    slice_id,
                )?;
                return Ok(decoded_count);
            }
            if macroblock_address >= self.completed.len() {
                return Err(H264Error::InvalidSyntax(
                    "CABAC B slice has no terminating end_of_slice_flag",
                ));
            }
        }
    }

    fn decode_cavlc_b_slice_data(
        &mut self,
        rbsp: &[u8],
        config: IntraSliceConfig,
        context: BMacroblockContext,
        references_l0: ReconstructionReferenceList<'_>,
        references_l1: ReconstructionReferenceList<'_>,
        modes: BReconstructionModes<'_>,
    ) -> Result<usize> {
        let reference_ids_l0 = references_l0.reference_ids;
        let reference_ids_l1 = references_l1.reference_ids;
        let deblock_references_l0 = self.deblock_reference_list(references_l0.pictures)?;
        let deblock_references_l1 = self.deblock_reference_list(references_l1.pictures)?;
        let references_l0 = references_l0.pictures;
        let references_l1 = references_l1.pictures;
        let mut reader = BitReader::new(rbsp);
        if !reader.skip_bits(config.header_bit_size) {
            return Err(H264Error::UnexpectedEof);
        }
        self.next_slice_id = self
            .next_slice_id
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        let slice_id = self.next_slice_id;
        self.cavlc.begin_slice();
        let mut quantizers = MacroblockQuantizerState::new(
            config.slice_qp_y,
            config.chroma_cb_offset,
            config.chroma_cr_offset,
            config.transform_bypass_enabled,
        )?;
        let mut macroblock_address = config.first_macroblock;
        let mut decoded_count = 0usize;
        let pcm_chroma_qp = [
            derive_chroma_qp(0, config.chroma_cb_offset),
            derive_chroma_qp(0, config.chroma_cr_offset),
        ];

        while more_rbsp_data(&reader) {
            let remaining = self.completed.len().checked_sub(macroblock_address).ok_or(
                H264Error::InvalidSyntax("B slice starts beyond the reconstructed picture"),
            )?;
            let skip_run = parse_cavlc_mb_skip_run(&mut reader, remaining)?;
            for _ in 0..skip_run {
                let (macroblock_x, macroblock_y) =
                    self.macroblock_coordinates(macroblock_address)?;
                let mb_x = u32::try_from(macroblock_x).map_err(|_| H264Error::IntegerOverflow)?;
                let mb_y = u32::try_from(macroblock_y).map_err(|_| H264Error::IntegerOverflow)?;
                let cavlc_snapshot = self.cavlc.snapshot_macroblock(mb_x, mb_y)?;
                self.cavlc.record_zero_macroblock(mb_x, mb_y)?;
                let motion =
                    match modes
                        .direct
                        .resolve(&mut self.b_motion, macroblock_address, slice_id)
                    {
                        Ok(motion) => motion,
                        Err(error) => {
                            self.cavlc.restore_macroblock(mb_x, mb_y, cavlc_snapshot);
                            return Err(error);
                        }
                    };
                if let Err(error) = self.reference_motion.record_b(
                    macroblock_address,
                    &motion,
                    reference_ids_l0,
                    reference_ids_l1,
                ) {
                    self.b_motion.clear_macroblock(macroblock_address)?;
                    self.cavlc.restore_macroblock(mb_x, mb_y, cavlc_snapshot);
                    return Err(error);
                }
                let quantizer = quantizers.derive(0)?;
                let deblock = match b_inter_deblock_info(
                    slice_id,
                    quantizer,
                    false,
                    config.deblocking_filter,
                    &motion,
                    None,
                    &deblock_references_l0,
                    &deblock_references_l1,
                ) {
                    Ok(deblock) => deblock,
                    Err(error) => {
                        self.b_motion.clear_macroblock(macroblock_address)?;
                        self.reference_motion.clear_macroblock(macroblock_address)?;
                        self.cavlc.restore_macroblock(mb_x, mb_y, cavlc_snapshot);
                        return Err(error);
                    }
                };
                let picture_snapshot = self
                    .picture
                    .snapshot_macroblock(macroblock_x, macroblock_y)?;
                let result = reconstruct_b_prediction(
                    &mut self.picture,
                    references_l0,
                    references_l1,
                    macroblock_x,
                    macroblock_y,
                    &motion,
                    None,
                    modes.weights,
                )
                .and_then(|()| self.modes.record_inter(macroblock_address, slice_id));
                if let Err(error) = result {
                    self.picture
                        .restore_macroblock(macroblock_x, macroblock_y, &picture_snapshot);
                    self.b_motion.clear_macroblock(macroblock_address)?;
                    self.reference_motion.clear_macroblock(macroblock_address)?;
                    self.modes.clear_macroblock(macroblock_address)?;
                    self.cavlc.restore_macroblock(mb_x, mb_y, cavlc_snapshot);
                    return Err(error);
                }
                self.complete_inter_macroblock(macroblock_address, deblock);
                macroblock_address = macroblock_address
                    .checked_add(1)
                    .ok_or(H264Error::IntegerOverflow)?;
                decoded_count = decoded_count
                    .checked_add(1)
                    .ok_or(H264Error::IntegerOverflow)?;
            }
            if !more_rbsp_data(&reader) {
                break;
            }

            let (macroblock_x, macroblock_y) = self.macroblock_coordinates(macroblock_address)?;
            let mb_x = u32::try_from(macroblock_x).map_err(|_| H264Error::IntegerOverflow)?;
            let mb_y = u32::try_from(macroblock_y).map_err(|_| H264Error::IntegerOverflow)?;
            let cavlc_snapshot = self.cavlc.snapshot_macroblock(mb_x, mb_y)?;
            let decoded = self
                .cavlc
                .decode_b_macroblock(&mut reader, mb_x, mb_y, context)?;
            let qp_delta = match &decoded {
                DecodedBSliceMacroblock::Inter { header, .. } => header.qp_delta,
                DecodedBSliceMacroblock::Intra(decoded) => match &decoded.macroblock {
                    IntraMacroblock::Predicted(header) => header.qp_delta,
                    IntraMacroblock::Pcm(_) => 0,
                },
            };
            let result = quantizers.with_macroblock(qp_delta, |quantizer| match &decoded {
                DecodedBSliceMacroblock::Inter { header, residual } => {
                    let reconstructed = reconstruct_b_inter_residual(
                        header,
                        residual,
                        quantizer,
                        &self.scaling_lists,
                        &self.scaling_lists_8x8,
                        self.scan_mode,
                    )?;
                    let motion = if is_fully_direct_b_macroblock(header) {
                        modes
                            .direct
                            .resolve(&mut self.b_motion, macroblock_address, slice_id)?
                    } else if has_direct_b_sub_macroblock(header) {
                        modes.direct.resolve_mixed(
                            &mut self.b_motion,
                            macroblock_address,
                            slice_id,
                            header,
                        )?
                    } else {
                        self.b_motion.resolve_inter_macroblock(
                            macroblock_address,
                            slice_id,
                            header,
                        )?
                    };
                    if let Err(error) = self.reference_motion.record_b(
                        macroblock_address,
                        &motion,
                        reference_ids_l0,
                        reference_ids_l1,
                    ) {
                        self.b_motion.clear_macroblock(macroblock_address)?;
                        return Err(error);
                    }
                    let deblock = match b_inter_deblock_info(
                        slice_id,
                        quantizer,
                        header.transform_size_8x8,
                        config.deblocking_filter,
                        &motion,
                        Some(residual),
                        &deblock_references_l0,
                        &deblock_references_l1,
                    ) {
                        Ok(deblock) => deblock,
                        Err(error) => {
                            self.b_motion.clear_macroblock(macroblock_address)?;
                            self.reference_motion.clear_macroblock(macroblock_address)?;
                            return Err(error);
                        }
                    };
                    let picture_snapshot = self
                        .picture
                        .snapshot_macroblock(macroblock_x, macroblock_y)?;
                    let reconstruction = reconstruct_b_prediction(
                        &mut self.picture,
                        references_l0,
                        references_l1,
                        macroblock_x,
                        macroblock_y,
                        &motion,
                        Some(&reconstructed),
                        modes.weights,
                    );
                    if let Err(error) = reconstruction
                        .and_then(|()| self.modes.record_inter(macroblock_address, slice_id))
                    {
                        self.picture.restore_macroblock(
                            macroblock_x,
                            macroblock_y,
                            &picture_snapshot,
                        );
                        self.b_motion.clear_macroblock(macroblock_address)?;
                        self.reference_motion.clear_macroblock(macroblock_address)?;
                        self.modes.clear_macroblock(macroblock_address)?;
                        return Err(error);
                    }
                    self.complete_inter_macroblock(macroblock_address, deblock);
                    Ok(())
                }
                DecodedBSliceMacroblock::Intra(decoded) => self
                    .reconstruct_macroblock_with_deblocking(
                        macroblock_address,
                        slice_id,
                        decoded,
                        quantizer,
                        config.deblocking_filter,
                        pcm_chroma_qp,
                    ),
            });
            if let Err(error) = result {
                self.cavlc.restore_macroblock(mb_x, mb_y, cavlc_snapshot);
                return Err(error);
            }
            macroblock_address = macroblock_address
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            decoded_count = decoded_count
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
        }
        consume_rbsp_trailing_bits(&mut reader)?;
        Ok(decoded_count)
    }

    fn macroblock_coordinates(&self, address: usize) -> Result<(usize, usize)> {
        self.validate_new_macroblock(address)
    }

    fn deblock_reference_list(
        &mut self,
        references: &[Option<&Yuv420Picture>],
    ) -> Result<DeblockReferenceList> {
        if references.len() > 32 {
            return Err(H264Error::InvalidSyntax(
                "deblocking reference list exceeds the AVC limit",
            ));
        }
        let len = references.len() as u8;

        let mut ids = [0u8; 32];
        for (index, reference) in references.iter().enumerate() {
            let Some(reference) = reference else {
                continue;
            };
            let address = std::ptr::from_ref(*reference).addr();
            let count = usize::from(self.deblock_reference_count);
            let token_index = self.deblock_reference_addresses[..count]
                .iter()
                .position(|candidate| *candidate == address)
                .unwrap_or(count);
            if token_index == count {
                if count == self.deblock_reference_addresses.len() {
                    return Err(H264Error::InvalidSyntax(
                        "picture uses too many distinct deblocking references",
                    ));
                }
                self.deblock_reference_addresses[count] = address;
                self.deblock_reference_count = self
                    .deblock_reference_count
                    .checked_add(1)
                    .ok_or(H264Error::IntegerOverflow)?;
            }
            ids[index] = u8::try_from(token_index + 1).map_err(|_| H264Error::IntegerOverflow)?;
        }
        Ok(DeblockReferenceList { ids, len })
    }

    fn complete_inter_macroblock(
        &mut self,
        macroblock_address: usize,
        deblock: MacroblockDeblockInfo,
    ) {
        self.deblock[macroblock_address] = deblock;
        self.completed[macroblock_address] = true;
    }

    fn finish_pending_p_inter(
        &mut self,
        jobs: &mut Vec<PendingPInterMacroblock>,
        references_l0: &[Option<&Yuv420Picture>],
        weights: Option<&PredictionWeightTable>,
        slice_id: u32,
    ) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        let coded_size = self.picture.coded_size();
        let reconstruct = |job: &PendingPInterMacroblock, scratch: &mut InterPrediction420| {
            reconstruct_p_macroblock_pixels_from_list_with_scratch(
                coded_size,
                references_l0,
                job.macroblock_x,
                job.macroblock_y,
                &job.motion,
                job.residual.as_ref(),
                weights,
                scratch,
            )
            .map(|pixels| StagedMacroblockPixels::new(job.address, pixels))
        };
        let staged = if jobs.len() > 1
            && let Some(pool) = self.reconstruction_executor.pool()
        {
            let reconstructed: Vec<Result<StagedMacroblockPixels>> = pool.install(|| {
                jobs.par_iter()
                    .map_init(InterPrediction420::empty, |scratch, job| {
                        reconstruct(job, scratch)
                    })
                    .collect()
            });
            reconstructed
                .into_iter()
                .collect::<Result<Vec<StagedMacroblockPixels>>>()?
        } else {
            let mut scratch = InterPrediction420::empty();
            let mut staged = Vec::with_capacity(jobs.len());
            for job in jobs.iter() {
                staged.push(StagedMacroblockPixels::empty(job.address));
                reconstruct_p_macroblock_pixels_from_list_into_with_scratch(
                    coded_size,
                    references_l0,
                    job.macroblock_x,
                    job.macroblock_y,
                    &job.motion,
                    job.residual.as_ref(),
                    weights,
                    &mut scratch,
                    staged
                        .last_mut()
                        .expect("the staged macroblock was just appended")
                        .pixels_mut(),
                )?;
            }
            staged
        };

        for (recorded_modes, job) in jobs.iter().enumerate() {
            if let Err(error) = self.modes.record_inter(job.address, slice_id) {
                for recorded in &jobs[..recorded_modes] {
                    self.modes.clear_macroblock(recorded.address)?;
                }
                return Err(error);
            }
        }
        if let Err(error) = self.picture.commit_macroblock_batch(&staged) {
            for job in jobs.iter() {
                self.modes.clear_macroblock(job.address)?;
            }
            return Err(error);
        }
        for job in jobs.iter() {
            self.complete_inter_macroblock(job.address, job.deblock);
        }
        jobs.clear();
        Ok(())
    }

    fn finish_pending_b_inter(
        &mut self,
        jobs: &mut Vec<PendingBInterMacroblock>,
        references_l0: &[Option<&Yuv420Picture>],
        references_l1: &[Option<&Yuv420Picture>],
        weights: BSlicePredictionWeights<'_>,
        slice_id: u32,
    ) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        let weight_mode = match weights {
            BSlicePredictionWeights::Default => BPredictionWeightMode::Default,
            BSlicePredictionWeights::Explicit(table) => BPredictionWeightMode::Explicit(table),
            BSlicePredictionWeights::Implicit {
                current_picture_order_count,
                list0,
                list1,
            } => BPredictionWeightMode::Implicit {
                current_picture_order_count,
                list0,
                list1,
            },
        };
        let coded_size = self.picture.coded_size();
        let reconstruct =
            |job: &PendingBInterMacroblock,
             scratch: &mut (InterPrediction420, InterPrediction420)| {
                let (prediction_l0, prediction_l1) = scratch;
                reconstruct_b_macroblock_pixels_from_lists_with_scratch(
                    coded_size,
                    references_l0,
                    references_l1,
                    job.macroblock_x,
                    job.macroblock_y,
                    &job.motion,
                    job.residual.as_ref(),
                    weight_mode,
                    prediction_l0,
                    prediction_l1,
                )
                .map(|pixels| StagedMacroblockPixels::new(job.address, pixels))
            };
        let staged = if jobs.len() > 1
            && let Some(pool) = self.reconstruction_executor.pool()
        {
            let reconstructed: Vec<Result<StagedMacroblockPixels>> = pool.install(|| {
                jobs.par_iter()
                    .map_init(
                        || (InterPrediction420::empty(), InterPrediction420::empty()),
                        |scratch, job| reconstruct(job, scratch),
                    )
                    .collect()
            });
            reconstructed
                .into_iter()
                .collect::<Result<Vec<StagedMacroblockPixels>>>()?
        } else {
            let mut scratch = (InterPrediction420::empty(), InterPrediction420::empty());
            let mut staged = Vec::with_capacity(jobs.len());
            for job in jobs.iter() {
                staged.push(StagedMacroblockPixels::empty(job.address));
                let (prediction_l0, prediction_l1) = &mut scratch;
                reconstruct_b_macroblock_pixels_from_lists_into_with_scratch(
                    coded_size,
                    references_l0,
                    references_l1,
                    job.macroblock_x,
                    job.macroblock_y,
                    &job.motion,
                    job.residual.as_ref(),
                    weight_mode,
                    prediction_l0,
                    prediction_l1,
                    staged
                        .last_mut()
                        .expect("the staged macroblock was just appended")
                        .pixels_mut(),
                )?;
            }
            staged
        };

        for (recorded_modes, job) in jobs.iter().enumerate() {
            if let Err(error) = self.modes.record_inter(job.address, slice_id) {
                for recorded in &jobs[..recorded_modes] {
                    self.modes.clear_macroblock(recorded.address)?;
                }
                return Err(error);
            }
        }
        if let Err(error) = self.picture.commit_macroblock_batch(&staged) {
            for job in jobs.iter() {
                self.modes.clear_macroblock(job.address)?;
            }
            return Err(error);
        }
        for job in jobs.iter() {
            self.complete_inter_macroblock(job.address, job.deblock);
        }
        jobs.clear();
        Ok(())
    }

    /// Reconstructs one decoded CAVLC intra macroblock.
    ///
    /// Prediction and residual matrices are fully prepared before any picture
    /// sample is changed. Consequently malformed directional modes do not
    /// leave a partially written macroblock.
    pub fn reconstruct_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        decoded: &DecodedIntraMacroblock,
        quantizer: MacroblockQuantizer,
    ) -> Result<()> {
        self.reconstruct_macroblock_with_deblocking(
            macroblock_address,
            slice_id,
            decoded,
            quantizer,
            DeblockingFilter::default(),
            [0, 0],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reconstruct_macroblock_with_deblocking(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        decoded: &DecodedIntraMacroblock,
        quantizer: MacroblockQuantizer,
        deblocking_filter: DeblockingFilter,
        pcm_chroma_qp: [u8; 2],
    ) -> Result<()> {
        let (macroblock_x, macroblock_y) = self.validate_new_macroblock(macroblock_address)?;
        let is_pcm = matches!(&decoded.macroblock, IntraMacroblock::Pcm(_));
        let transform_8x8 = matches!(
            &decoded.macroblock,
            IntraMacroblock::Predicted(IntraMacroblockHeader {
                luma_prediction: IntraLumaPrediction::EightByEight(_),
                ..
            })
        );
        match &decoded.macroblock {
            IntraMacroblock::Pcm(pcm) => {
                if decoded.residual.is_some() {
                    return Err(H264Error::InvalidSyntax(
                        "I_PCM macroblock unexpectedly contains residual blocks",
                    ));
                }
                self.modes
                    .record_other_intra(macroblock_address, slice_id)?;
                self.picture
                    .write_pcm_macroblock(macroblock_x, macroblock_y, pcm)?;
            }
            IntraMacroblock::Predicted(header) => {
                let residual = decoded.residual.as_ref().ok_or(H264Error::InvalidSyntax(
                    "predicted intra macroblock is missing residual blocks",
                ))?;
                let reconstructed = reconstruct_intra_residual(
                    header,
                    residual,
                    quantizer,
                    &self.scaling_lists,
                    &self.scaling_lists_8x8,
                    self.scan_mode,
                )?;
                match header.luma_prediction {
                    IntraLumaPrediction::SixteenBySixteen { mode } => {
                        self.reconstruct_intra16x16(
                            macroblock_address,
                            macroblock_x,
                            macroblock_y,
                            slice_id,
                            mode,
                            header.chroma_prediction_mode,
                            &reconstructed,
                        )?;
                        self.modes
                            .record_other_intra(macroblock_address, slice_id)?;
                    }
                    IntraLumaPrediction::FourByFour(syntax) => {
                        let snapshot = self
                            .picture
                            .snapshot_macroblock(macroblock_x, macroblock_y)?;
                        if let Err(error) = self.reconstruct_intra4x4(
                            macroblock_address,
                            macroblock_x,
                            macroblock_y,
                            slice_id,
                            &syntax,
                            header.chroma_prediction_mode,
                            &reconstructed,
                        ) {
                            self.picture
                                .restore_macroblock(macroblock_x, macroblock_y, &snapshot);
                            self.modes.clear_macroblock(macroblock_address)?;
                            return Err(error);
                        }
                    }
                    IntraLumaPrediction::EightByEight(syntax) => {
                        let snapshot = self
                            .picture
                            .snapshot_macroblock(macroblock_x, macroblock_y)?;
                        if let Err(error) = self.reconstruct_intra8x8(
                            macroblock_address,
                            macroblock_x,
                            macroblock_y,
                            slice_id,
                            &syntax,
                            header.chroma_prediction_mode,
                            &reconstructed,
                        ) {
                            self.picture
                                .restore_macroblock(macroblock_x, macroblock_y, &snapshot);
                            self.modes.clear_macroblock(macroblock_address)?;
                            return Err(error);
                        }
                    }
                }
            }
        }
        self.reference_motion.record_intra(macroblock_address)?;
        if let Err(error) = self
            .b_motion
            .record_intra_macroblock(macroblock_address, slice_id)
        {
            self.reference_motion.clear_macroblock(macroblock_address)?;
            return Err(error);
        }
        if let Err(error) = self
            .motion
            .record_intra_macroblock(macroblock_address, slice_id)
        {
            self.b_motion.clear_macroblock(macroblock_address)?;
            self.reference_motion.clear_macroblock(macroblock_address)?;
            return Err(error);
        }
        self.deblock[macroblock_address] = MacroblockDeblockInfo {
            slice_id,
            is_intra: true,
            luma_qp: if is_pcm { 0 } else { quantizer.luma },
            cb_qp: if is_pcm {
                pcm_chroma_qp[0]
            } else {
                quantizer.chroma_cb
            },
            cr_qp: if is_pcm {
                pcm_chroma_qp[1]
            } else {
                quantizer.chroma_cr
            },
            transform_8x8,
            internal_edges_zero: false,
            luma_nonzero: 0,
            motion: [DeblockMotion::default(); 16],
            filter: deblocking_filter,
        };
        self.completed[macroblock_address] = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn reconstruct_intra4x4(
        &mut self,
        macroblock_address: usize,
        macroblock_x: usize,
        macroblock_y: usize,
        slice_id: u32,
        syntax: &[IntraPredictionModeSyntax; 16],
        chroma_mode: u8,
        residual: &ReconstructedIntraResidual,
    ) -> Result<()> {
        let ReconstructedLumaResidual::FourByFour(luma_residual) = &residual.luma else {
            return Err(H264Error::InvalidSyntax(
                "Intra4x4 macroblock has non-4x4 residual samples",
            ));
        };
        let macroblock_availability = self.macroblock_availability(macroblock_address, slice_id);
        let cb_references = self.picture.intra_chroma_references(
            ChromaPlane::Cb,
            macroblock_x,
            macroblock_y,
            macroblock_availability,
        )?;
        let cr_references = self.picture.intra_chroma_references(
            ChromaPlane::Cr,
            macroblock_x,
            macroblock_y,
            macroblock_availability,
        )?;
        let cb_prediction = predict_intra_chroma_420(chroma_mode, &cb_references)?;
        let cr_prediction = predict_intra_chroma_420(chroma_mode, &cr_references)?;
        let modes = self.modes.derive_intra4x4(
            macroblock_address,
            slice_id,
            syntax,
            self.constrained_intra_prediction,
        )?;

        for (index, &(block_x, block_y)) in LUMA_4X4_COORDINATES.iter().enumerate() {
            let availability =
                self.intra4x4_availability(macroblock_address, slice_id, index, block_x, block_y);
            let x = macroblock_x * 16 + block_x * 4;
            let y = macroblock_y * 16 + block_y * 4;
            let references = self.picture.intra4x4_references(x, y, availability)?;
            let prediction = predict_intra_4x4(modes[index], &references)?;
            self.picture
                .write_luma_4x4(x, y, &prediction, &luma_residual[index])?;
        }

        let cb_residual = assemble_chroma_residual(&residual.chroma_cb);
        let cr_residual = assemble_chroma_residual(&residual.chroma_cr);
        self.picture.write_chroma_8x8(
            ChromaPlane::Cb,
            macroblock_x,
            macroblock_y,
            &cb_prediction,
            &cb_residual,
        )?;
        self.picture.write_chroma_8x8(
            ChromaPlane::Cr,
            macroblock_x,
            macroblock_y,
            &cr_prediction,
            &cr_residual,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reconstruct_intra8x8(
        &mut self,
        macroblock_address: usize,
        macroblock_x: usize,
        macroblock_y: usize,
        slice_id: u32,
        syntax: &[IntraPredictionModeSyntax; 4],
        chroma_mode: u8,
        residual: &ReconstructedIntraResidual,
    ) -> Result<()> {
        let ReconstructedLumaResidual::EightByEight(luma_residual) = &residual.luma else {
            return Err(H264Error::InvalidSyntax(
                "Intra8x8 macroblock has non-8x8 residual samples",
            ));
        };
        let macroblock_availability = self.macroblock_availability(macroblock_address, slice_id);
        let cb_references = self.picture.intra_chroma_references(
            ChromaPlane::Cb,
            macroblock_x,
            macroblock_y,
            macroblock_availability,
        )?;
        let cr_references = self.picture.intra_chroma_references(
            ChromaPlane::Cr,
            macroblock_x,
            macroblock_y,
            macroblock_availability,
        )?;
        let cb_prediction = predict_intra_chroma_420(chroma_mode, &cb_references)?;
        let cr_prediction = predict_intra_chroma_420(chroma_mode, &cr_references)?;
        let modes = self.modes.derive_intra8x8(
            macroblock_address,
            slice_id,
            syntax,
            self.constrained_intra_prediction,
        )?;

        for (index, &(block_x, block_y)) in LUMA_8X8_COORDINATES.iter().enumerate() {
            let availability =
                self.intra8x8_availability(macroblock_address, slice_id, index, block_x, block_y);
            let x = macroblock_x * 16 + block_x * 8;
            let y = macroblock_y * 16 + block_y * 8;
            let references = self.picture.intra8x8_references(x, y, availability)?;
            let prediction = predict_intra_8x8(modes[index], &references)?;
            self.picture
                .write_luma_8x8(x, y, &prediction, &luma_residual[index])?;
        }

        let cb_residual = assemble_chroma_residual(&residual.chroma_cb);
        let cr_residual = assemble_chroma_residual(&residual.chroma_cr);
        self.picture.write_chroma_8x8(
            ChromaPlane::Cb,
            macroblock_x,
            macroblock_y,
            &cb_prediction,
            &cb_residual,
        )?;
        self.picture.write_chroma_8x8(
            ChromaPlane::Cr,
            macroblock_x,
            macroblock_y,
            &cr_prediction,
            &cr_residual,
        )
    }

    pub fn into_nv12_frame(
        self,
        id: u64,
        pts: Option<MediaTime>,
        duration: Option<MediaTime>,
        format: VideoFormat,
    ) -> Result<DecodedVideoFrame> {
        self.into_deblocked_picture()?
            .into_nv12_frame(id, pts, duration, format)
    }

    pub(crate) fn into_deblocked_picture(self) -> Result<Yuv420Picture> {
        Ok(self.into_deblocked_reference_picture()?.0)
    }

    pub(crate) fn into_deblocked_reference_picture(
        mut self,
    ) -> Result<(Yuv420Picture, ReferenceMotionField)> {
        if self.completed.iter().any(|&completed| !completed) {
            return Err(H264Error::InvalidSyntax(
                "cannot output an incomplete reconstructed picture",
            ));
        }
        filter_420_picture(&mut self.picture, &self.deblock, self.width_in_macroblocks)?;
        let motion = self.reference_motion.finish()?;
        Ok((self.picture, motion))
    }

    #[allow(clippy::too_many_arguments)]
    fn reconstruct_intra16x16(
        &mut self,
        macroblock_address: usize,
        macroblock_x: usize,
        macroblock_y: usize,
        slice_id: u32,
        luma_mode: u8,
        chroma_mode: u8,
        residual: &ReconstructedIntraResidual,
    ) -> Result<()> {
        let ReconstructedLumaResidual::FourByFour(luma_residual) = &residual.luma else {
            return Err(H264Error::InvalidSyntax(
                "Intra16x16 macroblock has non-4x4 residual samples",
            ));
        };
        let availability = self.macroblock_availability(macroblock_address, slice_id);
        let luma_references =
            self.picture
                .intra16x16_references(macroblock_x, macroblock_y, availability)?;
        let luma_prediction = predict_intra_16x16(luma_mode, &luma_references)?;
        let cb_references = self.picture.intra_chroma_references(
            ChromaPlane::Cb,
            macroblock_x,
            macroblock_y,
            availability,
        )?;
        let cr_references = self.picture.intra_chroma_references(
            ChromaPlane::Cr,
            macroblock_x,
            macroblock_y,
            availability,
        )?;
        let cb_prediction = predict_intra_chroma_420(chroma_mode, &cb_references)?;
        let cr_prediction = predict_intra_chroma_420(chroma_mode, &cr_references)?;

        let luma_residual = assemble_luma_residual(luma_residual);
        let cb_residual = assemble_chroma_residual(&residual.chroma_cb);
        let cr_residual = assemble_chroma_residual(&residual.chroma_cr);
        self.picture.write_luma_16x16(
            macroblock_x,
            macroblock_y,
            &luma_prediction,
            &luma_residual,
        )?;
        self.picture.write_chroma_8x8(
            ChromaPlane::Cb,
            macroblock_x,
            macroblock_y,
            &cb_prediction,
            &cb_residual,
        )?;
        self.picture.write_chroma_8x8(
            ChromaPlane::Cr,
            macroblock_x,
            macroblock_y,
            &cr_prediction,
            &cr_residual,
        )
    }

    fn macroblock_availability(
        &self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> IntraReferenceAvailability {
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let top = (macroblock_y > 0).then(|| macroblock_address - self.width_in_macroblocks);
        let left = (macroblock_x > 0).then(|| macroblock_address - 1);
        let top_left = (macroblock_x > 0 && macroblock_y > 0)
            .then(|| macroblock_address - self.width_in_macroblocks - 1);
        IntraReferenceAvailability {
            top: top.is_some_and(|address| self.is_available(address, slice_id)),
            left: left.is_some_and(|address| self.is_available(address, slice_id)),
            top_left: top_left.is_some_and(|address| self.is_available(address, slice_id)),
            top_right: false,
        }
    }

    fn intra4x4_availability(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        block_index: usize,
        block_x: usize,
        block_y: usize,
    ) -> IntraReferenceAvailability {
        let x = (block_x * 4) as isize;
        let y = (block_y * 4) as isize;
        IntraReferenceAvailability {
            top: (0..4).all(|offset| {
                self.luma_reference_sample_available(
                    macroblock_address,
                    slice_id,
                    block_index,
                    x + offset,
                    y - 1,
                )
            }),
            left: (0..4).all(|offset| {
                self.luma_reference_sample_available(
                    macroblock_address,
                    slice_id,
                    block_index,
                    x - 1,
                    y + offset,
                )
            }),
            top_left: self.luma_reference_sample_available(
                macroblock_address,
                slice_id,
                block_index,
                x - 1,
                y - 1,
            ),
            top_right: !matches!(block_index, 3 | 11)
                && (4..8).all(|offset| {
                    self.luma_reference_sample_available(
                        macroblock_address,
                        slice_id,
                        block_index,
                        x + offset,
                        y - 1,
                    )
                }),
        }
    }

    fn intra8x8_availability(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        block_index: usize,
        block_x: usize,
        block_y: usize,
    ) -> IntraReferenceAvailability {
        let x = (block_x * 8) as isize;
        let y = (block_y * 8) as isize;
        IntraReferenceAvailability {
            top: (0..8).all(|offset| {
                self.luma8x8_reference_sample_available(
                    macroblock_address,
                    slice_id,
                    block_index,
                    x + offset,
                    y - 1,
                )
            }),
            left: (0..8).all(|offset| {
                self.luma8x8_reference_sample_available(
                    macroblock_address,
                    slice_id,
                    block_index,
                    x - 1,
                    y + offset,
                )
            }),
            top_left: self.luma8x8_reference_sample_available(
                macroblock_address,
                slice_id,
                block_index,
                x - 1,
                y - 1,
            ),
            top_right: (8..16).all(|offset| {
                self.luma8x8_reference_sample_available(
                    macroblock_address,
                    slice_id,
                    block_index,
                    x + offset,
                    y - 1,
                )
            }),
        }
    }

    fn luma8x8_reference_sample_available(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        block_index: usize,
        local_x: isize,
        local_y: isize,
    ) -> bool {
        if (0..16).contains(&local_x) && (0..16).contains(&local_y) {
            let neighbor_index = local_y as usize / 8 * 2 + local_x as usize / 8;
            return neighbor_index < block_index;
        }
        self.external_luma_reference_sample_available(
            macroblock_address,
            slice_id,
            local_x,
            local_y,
        )
    }

    fn luma_reference_sample_available(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        block_index: usize,
        local_x: isize,
        local_y: isize,
    ) -> bool {
        if (0..16).contains(&local_x) && (0..16).contains(&local_y) {
            let cell_x = local_x as usize / 4;
            let cell_y = local_y as usize / 4;
            return luma4x4_index(cell_x, cell_y) < block_index;
        }

        self.external_luma_reference_sample_available(
            macroblock_address,
            slice_id,
            local_x,
            local_y,
        )
    }

    fn external_luma_reference_sample_available(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        local_x: isize,
        local_y: isize,
    ) -> bool {
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let global_x = macroblock_x as isize * 16 + local_x;
        let global_y = macroblock_y as isize * 16 + local_y;
        if global_x < 0
            || global_y < 0
            || global_x >= (self.width_in_macroblocks * 16) as isize
            || global_y >= (self.completed.len() / self.width_in_macroblocks * 16) as isize
        {
            return false;
        }
        let neighbor_x = global_x as usize / 16;
        let neighbor_y = global_y as usize / 16;
        let neighbor_address = neighbor_y * self.width_in_macroblocks + neighbor_x;
        self.is_available(neighbor_address, slice_id)
    }

    fn is_available(&self, address: usize, slice_id: u32) -> bool {
        self.completed[address]
            && self.deblock[address].slice_id == slice_id
            && (!self.constrained_intra_prediction || self.deblock[address].is_intra)
    }

    fn validate_new_macroblock(&self, address: usize) -> Result<(usize, usize)> {
        if address >= self.completed.len() {
            return Err(H264Error::InvalidSyntax(
                "macroblock address exceeds reconstructed picture",
            ));
        }
        if self.completed[address] {
            return Err(H264Error::InvalidSyntax(
                "macroblock was already reconstructed",
            ));
        }
        Ok((
            address % self.width_in_macroblocks,
            address / self.width_in_macroblocks,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_b_prediction(
    picture: &mut Yuv420Picture,
    references_l0: &[Option<&Yuv420Picture>],
    references_l1: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedBMacroblock,
    residual: Option<&ReconstructedInterResidual>,
    weights: BSlicePredictionWeights<'_>,
) -> Result<()> {
    let weight_mode = match weights {
        BSlicePredictionWeights::Default => BPredictionWeightMode::Default,
        BSlicePredictionWeights::Explicit(weights) => BPredictionWeightMode::Explicit(weights),
        BSlicePredictionWeights::Implicit {
            current_picture_order_count,
            list0,
            list1,
        } => BPredictionWeightMode::Implicit {
            current_picture_order_count,
            list0,
            list1,
        },
    };
    reconstruct_b_macroblock_from_lists_with_mode(
        picture,
        references_l0,
        references_l1,
        macroblock_x,
        macroblock_y,
        motion,
        residual,
        weight_mode,
    )
}

fn is_fully_direct_b_macroblock(header: &crate::BInterMacroblockHeader) -> bool {
    match &header.partition_mode {
        BPartitionMode::Direct16x16 => true,
        BPartitionMode::EightByEight { sub_macroblocks } => sub_macroblocks
            .iter()
            .all(|sub_type| *sub_type == BSubMacroblockType::Direct8x8),
        BPartitionMode::SixteenBySixteen
        | BPartitionMode::SixteenByEight
        | BPartitionMode::EightBySixteen => false,
    }
}

fn has_direct_b_sub_macroblock(header: &crate::BInterMacroblockHeader) -> bool {
    matches!(
        &header.partition_mode,
        BPartitionMode::EightByEight { sub_macroblocks }
            if sub_macroblocks.contains(&BSubMacroblockType::Direct8x8)
    )
}

fn inter_deblock_info(
    slice_id: u32,
    quantizer: MacroblockQuantizer,
    transform_8x8: bool,
    filter: DeblockingFilter,
    resolved: &ResolvedPMacroblock,
    residual: Option<&InterResidual>,
    references_l0: &DeblockReferenceList,
) -> Result<MacroblockDeblockInfo> {
    let mut motion = [DeblockMotion::default(); 16];
    let mut first_motion = None;
    let mut uniform_motion = true;
    for partition in &resolved.partitions {
        let reference_id =
            references_l0
                .get(partition.reference_index)
                .ok_or(H264Error::InvalidSyntax(
                    "P macroblock selects a missing List-0 reference picture",
                ))?;
        let start_x = usize::from(partition.x) / 4;
        let start_y = usize::from(partition.y) / 4;
        let end_x = usize::from(partition.x + partition.width) / 4;
        let end_y = usize::from(partition.y + partition.height) / 4;
        if !partition.x.is_multiple_of(4)
            || !partition.y.is_multiple_of(4)
            || !partition.width.is_multiple_of(4)
            || !partition.height.is_multiple_of(4)
            || end_x > 4
            || end_y > 4
        {
            return Err(H264Error::InvalidSyntax(
                "P partition is not aligned to the deblocking grid",
            ));
        }
        let partition_motion = DeblockMotion::new(
            DeblockListMotion {
                reference_id,
                vector: partition.motion_vector,
            },
            DeblockListMotion::default(),
        );
        if let Some(first_motion) = first_motion {
            uniform_motion &= partition_motion == first_motion;
        } else {
            first_motion = Some(partition_motion);
        }
        for y in start_y..end_y {
            for x in start_x..end_x {
                motion[y * 4 + x] = partition_motion;
            }
        }
    }

    let luma_nonzero = inter_luma_nonzero(transform_8x8, residual);
    Ok(MacroblockDeblockInfo {
        slice_id,
        is_intra: false,
        luma_qp: quantizer.luma,
        cb_qp: quantizer.chroma_cb,
        cr_qp: quantizer.chroma_cr,
        transform_8x8,
        internal_edges_zero: luma_nonzero == 0 && first_motion.is_some() && uniform_motion,
        luma_nonzero,
        motion,
        filter,
    })
}

#[allow(clippy::too_many_arguments)]
fn b_inter_deblock_info(
    slice_id: u32,
    quantizer: MacroblockQuantizer,
    transform_8x8: bool,
    filter: DeblockingFilter,
    resolved: &ResolvedBMacroblock,
    residual: Option<&InterResidual>,
    references_l0: &DeblockReferenceList,
    references_l1: &DeblockReferenceList,
) -> Result<MacroblockDeblockInfo> {
    let mut motion = [DeblockMotion::default(); 16];
    let mut first_motion = None;
    let mut uniform_motion = true;
    for partition in &resolved.partitions {
        let start_x = usize::from(partition.x) / 4;
        let start_y = usize::from(partition.y) / 4;
        let end_x = usize::from(partition.x + partition.width) / 4;
        let end_y = usize::from(partition.y + partition.height) / 4;
        if !partition.x.is_multiple_of(4)
            || !partition.y.is_multiple_of(4)
            || !partition.width.is_multiple_of(4)
            || !partition.height.is_multiple_of(4)
            || end_x > 4
            || end_y > 4
        {
            return Err(H264Error::InvalidSyntax(
                "B partition is not aligned to the deblocking grid",
            ));
        }
        let list0 = b_deblock_list_motion(partition.list0, references_l0, "List 0")?;
        let list1 = b_deblock_list_motion(partition.list1, references_l1, "List 1")?;
        let partition_motion = DeblockMotion::new(list0, list1);
        if let Some(first_motion) = first_motion {
            uniform_motion &= partition_motion == first_motion;
        } else {
            first_motion = Some(partition_motion);
        }
        for y in start_y..end_y {
            for x in start_x..end_x {
                motion[y * 4 + x] = partition_motion;
            }
        }
    }
    let luma_nonzero = inter_luma_nonzero(transform_8x8, residual);
    Ok(MacroblockDeblockInfo {
        slice_id,
        is_intra: false,
        luma_qp: quantizer.luma,
        cb_qp: quantizer.chroma_cb,
        cr_qp: quantizer.chroma_cr,
        transform_8x8,
        internal_edges_zero: luma_nonzero == 0 && first_motion.is_some() && uniform_motion,
        luma_nonzero,
        motion,
        filter,
    })
}

fn b_deblock_list_motion(
    motion: Option<ResolvedBListMotion>,
    references: &DeblockReferenceList,
    list_name: &'static str,
) -> Result<DeblockListMotion> {
    let Some(motion) = motion else {
        return Ok(DeblockListMotion::default());
    };
    let reference_id = references
        .get(motion.reference_index)
        .ok_or(match list_name {
            "List 0" => {
                H264Error::InvalidSyntax("B macroblock selects a missing List-0 reference picture")
            }
            _ => {
                H264Error::InvalidSyntax("B macroblock selects a missing List-1 reference picture")
            }
        })?;
    Ok(DeblockListMotion {
        reference_id,
        vector: motion.motion_vector,
    })
}

fn inter_luma_nonzero(transform_8x8: bool, residual: Option<&InterResidual>) -> u16 {
    let mut luma_nonzero = 0u16;
    let Some(residual) = residual else {
        return luma_nonzero;
    };
    if transform_8x8 {
        for (block_8x8, &(region_x, region_y)) in LUMA_8X8_COORDINATES.iter().enumerate() {
            if residual.luma[block_8x8 * 4..block_8x8 * 4 + 4]
                .iter()
                .any(|block| block.total_coeff != 0)
            {
                for y in region_y * 2..region_y * 2 + 2 {
                    for x in region_x * 2..region_x * 2 + 2 {
                        luma_nonzero |= 1 << (y * 4 + x);
                    }
                }
            }
        }
    } else {
        for (index, block) in residual.luma.iter().enumerate() {
            let (x, y) = LUMA_4X4_COORDINATES[index];
            if block.total_coeff != 0 {
                luma_nonzero |= 1 << (y * 4 + x);
            }
        }
    }
    luma_nonzero
}

fn luma4x4_index(cell_x: usize, cell_y: usize) -> usize {
    8 * (cell_y / 2) + 4 * (cell_x / 2) + 2 * (cell_y % 2) + cell_x % 2
}

fn assemble_luma_residual(blocks: &[[[i32; 4]; 4]; 16]) -> [[i32; 16]; 16] {
    let mut output = [[0; 16]; 16];
    const COORDINATES: [(usize, usize); 16] = [
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
    for (block, (block_x, block_y)) in blocks.iter().zip(COORDINATES) {
        for row in 0..4 {
            output[block_y * 4 + row][block_x * 4..block_x * 4 + 4].copy_from_slice(&block[row]);
        }
    }
    output
}

fn assemble_chroma_residual(blocks: &[[[i32; 4]; 4]; 4]) -> [[i32; 8]; 8] {
    let mut output = [[0; 8]; 8];
    for (index, block) in blocks.iter().enumerate() {
        let block_x = index % 2;
        let block_y = index / 2;
        for row in 0..4 {
            output[block_y * 4 + row][block_x * 4..block_x * 4 + 4].copy_from_slice(&block[row]);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use decv_core::{ColorInfo, FrameStorage, PixelFormat, Rect, Size, VideoFormat};

    use super::{
        BDirectPrediction, BReconstructionModes, BSlicePredictionWeights,
        IntraPictureReconstructor, IntraSliceConfig, ReconstructionReferenceList,
    };
    use crate::{
        BMacroblockContext, CodedBlockPattern, DeblockingFilter, DecodedIntraMacroblock, H264Error,
        IntraLumaPrediction, IntraMacroblock, IntraMacroblockHeader, IntraPredictionModeSyntax,
        IntraResidual, MacroblockQuantizer, PcmMacroblock, ResidualBlock,
        resolve_scaling_lists_4x4,
    };

    const PREDICTED_MODE: IntraPredictionModeSyntax = IntraPredictionModeSyntax {
        use_predicted: true,
        remaining_mode: None,
    };

    fn quantizer() -> MacroblockQuantizer {
        MacroblockQuantizer {
            luma: 0,
            chroma_cb: 0,
            chroma_cr: 0,
            transform_bypass: false,
        }
    }

    fn predicted(luma_mode: u8, chroma_mode: u8) -> DecodedIntraMacroblock {
        DecodedIntraMacroblock {
            macroblock: IntraMacroblock::Predicted(IntraMacroblockHeader {
                luma_prediction: IntraLumaPrediction::SixteenBySixteen { mode: luma_mode },
                chroma_prediction_mode: chroma_mode,
                coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
                qp_delta: 0,
            }),
            residual: Some(IntraResidual {
                luma_dc: Some(ResidualBlock::empty(16)),
                luma: [ResidualBlock::empty(15); 16],
                chroma_dc: [ResidualBlock::empty(4); 2],
                chroma_ac: [[ResidualBlock::empty(15); 4]; 2],
            }),
        }
    }

    fn pcm(luma: u8, cb: u8, cr: u8) -> DecodedIntraMacroblock {
        DecodedIntraMacroblock {
            macroblock: IntraMacroblock::Pcm(PcmMacroblock {
                luma: Box::new([luma; 256]),
                chroma: Box::new(std::array::from_fn(
                    |index| if index < 64 { cb } else { cr },
                )),
            }),
            residual: None,
        }
    }

    fn predicted4x4(
        modes: [IntraPredictionModeSyntax; 16],
        chroma_mode: u8,
    ) -> DecodedIntraMacroblock {
        DecodedIntraMacroblock {
            macroblock: IntraMacroblock::Predicted(IntraMacroblockHeader {
                luma_prediction: IntraLumaPrediction::FourByFour(modes),
                chroma_prediction_mode: chroma_mode,
                coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
                qp_delta: 0,
            }),
            residual: Some(IntraResidual {
                luma_dc: None,
                luma: [ResidualBlock::empty(16); 16],
                chroma_dc: [ResidualBlock::empty(4); 2],
                chroma_ac: [[ResidualBlock::empty(15); 4]; 2],
            }),
        }
    }

    fn predicted8x8(
        modes: [IntraPredictionModeSyntax; 4],
        chroma_mode: u8,
    ) -> DecodedIntraMacroblock {
        DecodedIntraMacroblock {
            macroblock: IntraMacroblock::Predicted(IntraMacroblockHeader {
                luma_prediction: IntraLumaPrediction::EightByEight(modes),
                chroma_prediction_mode: chroma_mode,
                coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
                qp_delta: 0,
            }),
            residual: Some(IntraResidual {
                luma_dc: None,
                luma: [ResidualBlock::empty(16); 16],
                chroma_dc: [ResidualBlock::empty(4); 2],
                chroma_ac: [[ResidualBlock::empty(15); 4]; 2],
            }),
        }
    }

    fn reconstructor(size: Size) -> IntraPictureReconstructor {
        IntraPictureReconstructor::new(size, resolve_scaling_lists_4x4(None, None).unwrap(), false)
            .unwrap()
    }

    fn p_slice_config() -> IntraSliceConfig {
        IntraSliceConfig {
            header_bit_size: 0,
            first_macroblock: 0,
            slice_qp_y: 26,
            transform_8x8_mode: false,
            chroma_cb_offset: 0,
            chroma_cr_offset: 0,
            transform_bypass_enabled: false,
            deblocking_filter: DeblockingFilter::default(),
        }
    }

    fn b_macroblock_context() -> BMacroblockContext {
        BMacroblockContext {
            num_ref_idx_l0_active: 1,
            num_ref_idx_l1_active: 1,
            transform_8x8_mode_enabled: false,
            direct_8x8_inference: true,
        }
    }

    fn b_reconstruction_modes() -> BReconstructionModes<'static> {
        BReconstructionModes {
            weights: BSlicePredictionWeights::Default,
            direct: BDirectPrediction::Temporal(None),
        }
    }

    fn constant_picture(value: u8) -> crate::Yuv420Picture {
        let mut picture = crate::Yuv420Picture::new(Size::new(16, 16)).unwrap();
        let (luma, cb, cr) = picture.planes_mut();
        luma.fill(value);
        cb.fill(value + 1);
        cr.fill(value + 2);
        picture
    }

    #[test]
    fn deblock_reference_tokens_preserve_identity_across_reordered_lists() {
        let first = constant_picture(16);
        let second = constant_picture(32);
        let mut reconstructor = reconstructor(Size::new(16, 16));

        let initial = reconstructor
            .deblock_reference_list(&[Some(&first), None, Some(&second), Some(&first)])
            .unwrap();
        let first_id = initial.get(0).unwrap();
        let second_id = initial.get(2).unwrap();
        assert_ne!(first_id, second_id);
        assert_eq!(initial.get(1), None);
        assert_eq!(initial.get(3), Some(first_id));
        assert_eq!(initial.get(4), None);

        let reordered = reconstructor
            .deblock_reference_list(&[Some(&second), Some(&first)])
            .unwrap();
        assert_eq!(reordered.get(0), Some(second_id));
        assert_eq!(reordered.get(1), Some(first_id));
        assert_eq!(reconstructor.deblock_reference_count, 2);
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

    #[test]
    fn decodes_complete_inter_and_skipped_cavlc_p_slices() {
        let reference = constant_picture(70);
        let references = [Some(&reference)];

        // mb_skip_run=0, P_L0_16x16, MVD=(0,0), CBP=0, trailing bits.
        let mut inter = reconstructor(Size::new(16, 16));
        assert_eq!(
            inter.decode_cavlc_p_slice_data(
                &bit_string("111111"),
                p_slice_config(),
                1,
                ReconstructionReferenceList::new(&references),
                None,
            ),
            Ok(1)
        );
        let (luma, cb, cr) = inter.picture().planes();
        assert!(luma.iter().all(|&sample| sample == 70));
        assert!(cb.iter().all(|&sample| sample == 71));
        assert!(cr.iter().all(|&sample| sample == 72));
        assert!(
            inter
                .into_nv12_frame(1, None, None, format(Size::new(16, 16)))
                .is_ok()
        );

        // mb_skip_run=1 followed directly by rbsp_trailing_bits.
        let mut skipped = reconstructor(Size::new(16, 16));
        assert_eq!(
            skipped.decode_cavlc_p_slice_data(
                &bit_string("0101"),
                p_slice_config(),
                1,
                ReconstructionReferenceList::new(&references),
                None,
            ),
            Ok(1)
        );
        assert!(
            skipped
                .picture()
                .planes()
                .0
                .iter()
                .all(|&sample| sample == 70)
        );
        assert!(
            skipped
                .into_nv12_frame(2, None, None, format(Size::new(16, 16)))
                .is_ok()
        );
    }

    #[test]
    fn decodes_explicit_list0_and_bidirectional_cavlc_b_slices() {
        let reference_l0 = constant_picture(40);
        let reference_l1 = constant_picture(100);
        let references_l0 = [Some(&reference_l0)];
        let references_l1 = [Some(&reference_l1)];

        // mb_skip_run=0, B_L0_16x16, MVD L0=(0,0), CBP=0, trailing bits.
        let mut list0 = reconstructor(Size::new(16, 16));
        assert_eq!(
            list0.decode_cavlc_b_slice_data(
                &bit_string("10101111"),
                p_slice_config(),
                b_macroblock_context(),
                ReconstructionReferenceList::new(&references_l0),
                ReconstructionReferenceList::new(&references_l1),
                b_reconstruction_modes(),
            ),
            Ok(1)
        );
        let (luma, cb, cr) = list0.picture().planes();
        assert!(luma.iter().all(|&sample| sample == 40));
        assert!(cb.iter().all(|&sample| sample == 41));
        assert!(cr.iter().all(|&sample| sample == 42));

        // mb_skip_run=0, B_Bi_16x16, zero MVDs for both lists, CBP=0.
        let mut bidirectional = reconstructor(Size::new(16, 16));
        assert_eq!(
            bidirectional.decode_cavlc_b_slice_data(
                &bit_string("100100111111"),
                p_slice_config(),
                b_macroblock_context(),
                ReconstructionReferenceList::new(&references_l0),
                ReconstructionReferenceList::new(&references_l1),
                b_reconstruction_modes(),
            ),
            Ok(1)
        );
        let (luma, cb, cr) = bidirectional.picture().planes();
        assert!(luma.iter().all(|&sample| sample == 70));
        assert!(cb.iter().all(|&sample| sample == 71));
        assert!(cr.iter().all(|&sample| sample == 72));

        let mut direct = reconstructor(Size::new(16, 16));
        assert!(matches!(
            direct.decode_cavlc_b_slice_data(
                &bit_string("0101"),
                p_slice_config(),
                b_macroblock_context(),
                ReconstructionReferenceList::new(&references_l0),
                ReconstructionReferenceList::new(&references_l1),
                b_reconstruction_modes(),
            ),
            Err(H264Error::InvalidSyntax(_))
        ));
    }

    fn format(size: Size) -> VideoFormat {
        VideoFormat {
            coded_size: size,
            visible_rect: Rect::new(0, 0, size.width, size.height),
            display_size: size,
            pixel_format: PixelFormat::Nv12,
            color: ColorInfo::default(),
        }
    }

    #[test]
    fn reconstructs_flat_intra16_picture_to_nv12() {
        let size = Size::new(16, 16);
        let mut reconstructor = reconstructor(size);
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted(2, 0), quantizer())
            .unwrap();
        let frame = reconstructor
            .into_nv12_frame(1, None, None, format(size))
            .unwrap();
        let cpu = match frame.storage {
            FrameStorage::Cpu(cpu) => cpu,
            _ => panic!("expected CPU frame"),
        };
        assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
    }

    #[test]
    fn decodes_one_complete_cavlc_intra_slice_from_rbsp_bits() {
        // mb_type=I_NxN; sixteen predicted-mode flags; chroma mode=DC;
        // coded_block_pattern codeNum=3 (no residual); rbsp_trailing_bits.
        let rbsp = [0xff, 0xff, 0xc9];
        let size = Size::new(16, 16);
        let mut reconstructor = reconstructor(size);
        assert_eq!(
            reconstructor.decode_cavlc_intra_slice_data(
                &rbsp,
                IntraSliceConfig {
                    header_bit_size: 0,
                    first_macroblock: 0,
                    slice_qp_y: 26,
                    transform_8x8_mode: false,
                    chroma_cb_offset: 0,
                    chroma_cr_offset: 0,
                    transform_bypass_enabled: false,
                    deblocking_filter: DeblockingFilter::default(),
                }
            ),
            Ok(1)
        );
        let frame = reconstructor
            .into_nv12_frame(1, None, None, format(size))
            .unwrap();
        let cpu = match frame.storage {
            FrameStorage::Cpu(cpu) => cpu,
            _ => panic!("expected CPU frame"),
        };
        assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
    }

    #[test]
    fn decodes_one_complete_cavlc_intra8_slice_from_rbsp_bits() {
        // mb_type=I_NxN; transform_size_8x8_flag; four predicted-mode flags;
        // chroma mode=DC; coded_block_pattern codeNum=3 (no residual);
        // rbsp_trailing_bits.
        let rbsp = [0xfe, 0x48];
        let size = Size::new(16, 16);
        let mut reconstructor = reconstructor(size);
        assert_eq!(
            reconstructor.decode_cavlc_intra_slice_data(
                &rbsp,
                IntraSliceConfig {
                    header_bit_size: 0,
                    first_macroblock: 0,
                    slice_qp_y: 26,
                    transform_8x8_mode: true,
                    chroma_cb_offset: 0,
                    chroma_cr_offset: 0,
                    transform_bypass_enabled: false,
                    deblocking_filter: DeblockingFilter::default(),
                }
            ),
            Ok(1)
        );
        assert!(
            reconstructor
                .into_nv12_frame(1, None, None, format(size))
                .is_ok()
        );
    }

    #[test]
    fn reconstructs_intra4_blocks_in_normative_scan_order() {
        let size = Size::new(16, 16);
        let mut reconstructor = reconstructor(size);
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted4x4([PREDICTED_MODE; 16], 0), quantizer())
            .unwrap();
        let frame = reconstructor
            .into_nv12_frame(1, None, None, format(size))
            .unwrap();
        let cpu = match frame.storage {
            FrameStorage::Cpu(cpu) => cpu,
            _ => panic!("expected CPU frame"),
        };
        assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
    }

    #[test]
    fn reconstructs_intra8_blocks_with_internal_references() {
        let size = Size::new(16, 16);
        let mut reconstructor = reconstructor(size);
        let explicit = |remaining_mode| IntraPredictionModeSyntax {
            use_predicted: false,
            remaining_mode: Some(remaining_mode),
        };
        reconstructor
            .reconstruct_macroblock(
                0,
                1,
                &predicted8x8(
                    [PREDICTED_MODE, explicit(1), explicit(0), PREDICTED_MODE],
                    0,
                ),
                quantizer(),
            )
            .unwrap();
        let frame = reconstructor
            .into_nv12_frame(1, None, None, format(size))
            .unwrap();
        let cpu = match frame.storage {
            FrameStorage::Cpu(cpu) => cpu,
            _ => panic!("expected CPU frame"),
        };
        assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
    }

    #[test]
    fn reconstructs_intra4_from_a_completed_left_macroblock() {
        let size = Size::new(32, 16);
        let mut reconstructor = reconstructor(size);
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted(2, 0), quantizer())
            .unwrap();
        let mut syntax = [PREDICTED_MODE; 16];
        syntax[0] = IntraPredictionModeSyntax {
            use_predicted: false,
            remaining_mode: Some(1),
        };
        reconstructor
            .reconstruct_macroblock(1, 1, &predicted4x4(syntax, 1), quantizer())
            .unwrap();
    }

    #[test]
    fn rolls_back_partial_intra4_picture_and_mode_state() {
        let size = Size::new(16, 16);
        let mut reconstructor = reconstructor(size);
        let mut invalid = [PREDICTED_MODE; 16];
        invalid[0] = IntraPredictionModeSyntax {
            use_predicted: false,
            remaining_mode: Some(0),
        };
        assert!(
            reconstructor
                .reconstruct_macroblock(0, 1, &predicted4x4(invalid, 0), quantizer())
                .is_err()
        );
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted4x4([PREDICTED_MODE; 16], 0), quantizer())
            .unwrap();
    }

    #[test]
    fn uses_completed_left_macroblock_for_horizontal_prediction() {
        let size = Size::new(32, 16);
        let mut reconstructor = reconstructor(size);
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted(2, 0), quantizer())
            .unwrap();
        reconstructor
            .reconstruct_macroblock(1, 1, &predicted(1, 1), quantizer())
            .unwrap();
        assert!(
            reconstructor
                .into_nv12_frame(1, None, None, format(size))
                .is_ok()
        );
    }

    #[test]
    fn failed_directional_prediction_leaves_macroblock_retryable() {
        let size = Size::new(16, 16);
        let mut reconstructor = reconstructor(size);
        assert!(
            reconstructor
                .reconstruct_macroblock(0, 1, &predicted(0, 2), quantizer())
                .is_err()
        );
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted(2, 0), quantizer())
            .unwrap();
        assert!(
            reconstructor
                .into_nv12_frame(1, None, None, format(size))
                .is_ok()
        );
    }

    #[test]
    fn hides_references_across_slice_boundaries() {
        let size = Size::new(32, 16);
        let mut reconstructor = reconstructor(size);
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted(2, 0), quantizer())
            .unwrap();
        assert!(
            reconstructor
                .reconstruct_macroblock(1, 2, &predicted(1, 1), quantizer())
                .is_err()
        );
        reconstructor
            .reconstruct_macroblock(1, 2, &predicted(2, 0), quantizer())
            .unwrap();
    }

    #[test]
    fn writes_pcm_and_rejects_incomplete_output() {
        let size = Size::new(32, 16);
        let mut incomplete = reconstructor(size);
        incomplete
            .reconstruct_macroblock(0, 1, &pcm(5, 10, 20), quantizer())
            .unwrap();
        assert!(
            incomplete
                .into_nv12_frame(1, None, None, format(size))
                .is_err()
        );

        let size = Size::new(16, 16);
        let mut complete = reconstructor(size);
        complete
            .reconstruct_macroblock(0, 1, &pcm(5, 10, 20), quantizer())
            .unwrap();
        let frame = complete
            .into_nv12_frame(1, None, None, format(size))
            .unwrap();
        let cpu = match frame.storage {
            FrameStorage::Cpu(cpu) => cpu,
            _ => panic!("expected CPU frame"),
        };
        assert!(cpu.planes[0].bytes[..256].iter().all(|&sample| sample == 5));
        assert_eq!(&cpu.planes[1].bytes[..4], &[10, 20, 10, 20]);
    }
}
