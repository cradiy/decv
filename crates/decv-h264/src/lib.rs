//! A pure-Rust H.264 decoder.
//!
//! The crate is intentionally organized as a sequence of representation
//! layers: Annex-B bytes, NAL units, RBSP syntax, parameter sets, slices, and
//! finally reconstructed pictures.

mod annex_b;
mod cavlc;
mod deblock;
mod decoder;
mod dpb;
mod error;
mod macroblock;
mod nal;
mod picture;
mod pps;
mod prediction;
mod rbsp;
mod slice;
mod sps;
mod transform;

pub use annex_b::{AnnexBNalUnit, AnnexBReader};
pub use error::{H264Error, Result};
pub use nal::{NalHeader, NalUnit, NalUnitType};
pub use pps::{
    EntropyCodingMode, PictureParameterSet, SliceGroupMap, SliceGroupRectangle,
    WeightedBiprediction,
};
pub use rbsp::{consume_rbsp_trailing_bits, decode_rbsp};
pub use sps::{
    BitstreamRestrictions, PicOrderCount, Profile, SampleAspectRatio, ScalingList, ScalingMatrices,
    SequenceParameterSet, TimingInfo, VuiParameters,
};
