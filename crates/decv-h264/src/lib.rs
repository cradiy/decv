//! A pure-Rust H.264 decoder.
//!
//! The crate is intentionally organized as a sequence of representation
//! layers: Annex-B bytes, NAL units, RBSP syntax, parameter sets, slices, and
//! finally reconstructed pictures.

mod annex_b;
mod b_motion;
mod cavlc;
mod cavlc_context;
pub mod deblock;
mod decoder;
mod dpb;
mod error;
mod inter_prediction;
mod inter_reconstruction;
mod intra_modes;
mod intra_reconstruction;
mod macroblock;
mod motion;
mod nal;
mod parameter_sets;
mod picture;
mod picture_surface;
mod pps;
mod prediction;
mod quantization;
mod rbsp;
mod reconstruction;
mod slice;
mod sps;
mod transform;

pub use annex_b::{AnnexBNalUnit, AnnexBReader};
pub use b_motion::BMotionState;
pub use cavlc::{
    CoeffToken, CoeffTokenContext, ResidualBlock, decode_coeff_token, decode_residual_block,
};
pub use cavlc_context::CavlcNeighborState;
pub use decoder::{H264Decoder, H264StreamParser, ParserEvent};
pub use dpb::{
    ActiveBReferenceLists, ActiveReferenceList, DecodedPictureBuffer, DefaultBReferenceLists,
    DefaultReferenceList, DpbReference, ReferenceKind, ReferencePicture,
};
pub use error::{H264Error, Result};
pub use inter_prediction::InterPrediction420;
pub use inter_reconstruction::{
    reconstruct_b_macroblock_from_lists_420, reconstruct_p_macroblock_420,
    reconstruct_p_macroblock_from_list_420, reconstruct_weighted_p_macroblock_from_list_420,
};
pub use intra_modes::IntraModeState;
pub use intra_reconstruction::IntraPictureReconstructor;
pub use macroblock::{
    BInterMacroblockHeader, BMacroblockContext, BPartitionMode, BPartitionMotion, BPredictionMode,
    BSliceMacroblock, BSubMacroblockType, CodedBlockPattern, DecodedBSliceMacroblock,
    DecodedIntraMacroblock, DecodedPSliceMacroblock, InterResidual, IntraLumaPrediction,
    IntraMacroblock, IntraMacroblockHeader, IntraPredictionModeSyntax, IntraResidual,
    MotionVectorDifference, PInterMacroblockHeader, PMacroblockContext, PPartitionMode,
    PPartitionMotion, PSliceMacroblock, PSubMacroblockType, PcmMacroblock,
    parse_cavlc_b_macroblock, parse_cavlc_intra_macroblock, parse_cavlc_mb_skip_run,
    parse_cavlc_p_macroblock,
};
pub use motion::{
    MotionVector, PMotionState, ResolvedBListMotion, ResolvedBMacroblock, ResolvedBPartition,
    ResolvedPMacroblock, ResolvedPPartition,
};
pub use nal::{NalHeader, NalUnit, NalUnitType};
pub use parameter_sets::{ActiveParameterSets, ParameterSetStore};
pub use picture::{FieldOrderCount, PictureOrderCount, PictureOrderCountState};
pub use picture_surface::{ChromaPlane, IntraReferenceAvailability, Yuv420Picture};
pub use pps::{
    EntropyCodingMode, PictureParameterSet, SliceGroupMap, SliceGroupRectangle,
    WeightedBiprediction,
};
pub use prediction::{
    Intra4x4References, Intra8x8References, Intra16x16References, IntraChroma420References,
    Prediction4x4, Prediction8x8, Prediction16x16, filter_intra_8x8_references, predict_intra_4x4,
    predict_intra_8x8, predict_intra_16x16, predict_intra_chroma_420,
};
pub use quantization::{MacroblockQuantizer, MacroblockQuantizerState, derive_chroma_qp};
pub use rbsp::{consume_rbsp_trailing_bits, decode_rbsp};
pub use reconstruction::{
    ReconstructedInterResidual, ReconstructedIntraResidual, ReconstructedLumaResidual,
    reconstruct_inter_residual, reconstruct_intra_residual,
};
pub use slice::{
    DeblockingFilter, MemoryManagementOperation, ParsedSliceHeader, PredictionWeight,
    PredictionWeightTable, ReferenceListModification, ReferencePictureMarking, SliceHeader,
    SlicePictureOrder, SliceType, WeightOffset,
};
pub use sps::{
    BitstreamRestrictions, PicOrderCount, Profile, SampleAspectRatio, ScalingList, ScalingMatrices,
    SequenceParameterSet, TimingInfo, VuiParameters,
};
pub use transform::{
    Block4x4, Block8x8, ColorComponent, DEFAULT_SCALING_LIST_4X4_INTER,
    DEFAULT_SCALING_LIST_4X4_INTRA, DEFAULT_SCALING_LIST_8X8_INTER, DEFAULT_SCALING_LIST_8X8_INTRA,
    FLAT_SCALING_LIST_4X4, FLAT_SCALING_LIST_8X8, PredictionClass, ResolvedScalingLists4x4,
    ResolvedScalingLists8x8, ScanMode, inverse_scale_4x4, inverse_scale_8x8, inverse_scan_4x4,
    inverse_scan_8x8, inverse_scan_scaling_list_4x4, inverse_scan_scaling_list_8x8,
    inverse_transform_4x4, inverse_transform_8x8, inverse_transform_chroma_dc_420,
    inverse_transform_luma_dc_4x4, reconstruct_residual_4x4, reconstruct_residual_8x8,
    resolve_scaling_lists_4x4, resolve_scaling_lists_8x8,
};
