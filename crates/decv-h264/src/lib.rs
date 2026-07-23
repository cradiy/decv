//! A pure-Rust H.264 decoder.
//!
//! The crate is intentionally organized as a sequence of representation
//! layers: Annex-B bytes, NAL units, RBSP syntax, parameter sets, slices, and
//! finally reconstructed pictures.

mod annex_b;
mod cavlc;
mod cavlc_context;
mod deblock;
mod decoder;
mod dpb;
mod error;
mod macroblock;
mod nal;
mod parameter_sets;
mod picture;
mod pps;
mod prediction;
mod rbsp;
mod slice;
mod sps;
mod transform;

pub use annex_b::{AnnexBNalUnit, AnnexBReader};
pub use cavlc::{
    CoeffToken, CoeffTokenContext, ResidualBlock, decode_coeff_token, decode_residual_block,
};
pub use cavlc_context::CavlcNeighborState;
pub use decoder::{H264StreamParser, ParserEvent};
pub use error::{H264Error, Result};
pub use macroblock::{
    CodedBlockPattern, DecodedIntraMacroblock, IntraLumaPrediction, IntraMacroblock,
    IntraMacroblockHeader, IntraPredictionModeSyntax, IntraResidual, PcmMacroblock,
    parse_cavlc_intra_macroblock,
};
pub use nal::{NalHeader, NalUnit, NalUnitType};
pub use parameter_sets::{ActiveParameterSets, ParameterSetStore};
pub use picture::{FieldOrderCount, PictureOrderCount, PictureOrderCountState};
pub use pps::{
    EntropyCodingMode, PictureParameterSet, SliceGroupMap, SliceGroupRectangle,
    WeightedBiprediction,
};
pub use rbsp::{consume_rbsp_trailing_bits, decode_rbsp};
pub use slice::{
    DeblockingFilter, MemoryManagementOperation, ParsedSliceHeader, PredictionWeight,
    PredictionWeightTable, ReferenceListModification, ReferencePictureMarking, SliceHeader,
    SlicePictureOrder, SliceType, WeightOffset,
};
pub use sps::{
    BitstreamRestrictions, PicOrderCount, Profile, SampleAspectRatio, ScalingList, ScalingMatrices,
    SequenceParameterSet, TimingInfo, VuiParameters,
};
